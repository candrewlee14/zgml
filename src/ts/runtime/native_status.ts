import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";

export type AdapterStatusName = (code: number) => string;

export class ZgmlError extends Error {
  readonly code: number;
  readonly status: string;

  constructor(code: number, status: string) {
    super(`zgml ${status} (${code})`);
    this.code = code;
    this.status = status;
  }
}

export function checkStatusOk(code: number, statusName: AdapterStatusName, ok = 0): void {
  if (code !== ok) throw new ZgmlError(code, statusName(code));
}

export function missingNativeHandleError(name: string): Error {
  return new Error(`${name} handle was not returned`);
}

export function freedNativeHandleError(name: string): Error {
  return new Error(`${name} was already freed`);
}

export function requireNativeHandleReturned<T>(handle: T | null | undefined, name: string): T {
  if (!handle) throw missingNativeHandleError(name);
  return handle;
}

export function assertNativeHandleAlive(handle: unknown, name: string): void {
  if (!handle) throw freedNativeHandleError(name);
}

export const nativeStatusManifest = Object.freeze({
  kind: "zgml-native-status-runtime-policy",
  ...tsRuntimeManifestPolicy("src/ts/runtime/native_status.ts", "Native status/error contract -> host adapters"),
});
