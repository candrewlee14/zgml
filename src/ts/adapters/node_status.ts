import {
  type AdapterStatusName,
  ZgmlError,
  assertNativeHandleAlive,
  checkStatusOk,
  requireNativeHandleReturned,
} from "../runtime/native_status.js";

export type NodeStatusName = AdapterStatusName;

export type NodeStatusHelpers = Readonly<{
  ZgmlError: typeof ZgmlError;
  check(code: number): void;
  handleOut(): [unknown];
  readHandle<T = unknown>(out: readonly [T | null | undefined], name: string): T;
  assertAlive(handle: unknown, name: string): void;
}>;

export type NodeStatusHelpersOptions = Readonly<{
  ok?: number;
  statusName: NodeStatusName;
}>;

export { ZgmlError };

export function createNodeStatusHelpers(options: NodeStatusHelpersOptions): NodeStatusHelpers {
  const ok = options.ok ?? 0;

  return Object.freeze({
    ZgmlError,
    check(code) {
      checkStatusOk(code, options.statusName, ok);
    },
    handleOut(): [unknown] {
      return [null];
    },
    readHandle(out, name) {
      return requireNativeHandleReturned(out[0], name);
    },
    assertAlive(handle, name) {
      assertNativeHandleAlive(handle, name);
    },
  });
}
