import {
  type AdapterStatusName,
  ZgmlError,
  assertNativeHandleAlive,
  checkStatusOk,
} from "../runtime/native_status.js";

export type BunStatusName = AdapterStatusName;
export type BunNativeHandle = number;
export type BunHandleOut = BigUint64Array;

export type BunStatusHelpers = Readonly<{
  check(code: number): void;
  handleOut(): BunHandleOut;
  readHandle(out: BunHandleOut): BunNativeHandle;
  assertAlive(handle: BunNativeHandle, name: string): void;
}>;

export type BunStatusHelpersOptions = Readonly<{
  ok?: number;
  statusName: BunStatusName;
  readPointer(out: BunHandleOut): BunNativeHandle;
}>;

export { ZgmlError };

export function createBunStatusHelpers(options: BunStatusHelpersOptions): BunStatusHelpers {
  const ok = options.ok ?? 0;
  return Object.freeze({
    check(code) {
      checkStatusOk(code, options.statusName, ok);
    },
    handleOut() {
      return new BigUint64Array(1);
    },
    readHandle(out) {
      return options.readPointer(out);
    },
    assertAlive(handle, name) {
      assertNativeHandleAlive(handle, name);
    },
  });
}
