"use strict";

import {
  dlopen as bunDlopen,
  FFIType as bunFfiType,
  ptr as bunPtr,
  read as bunRead,
} from "bun:ffi";
import { nativeAdapterManifestPolicy } from "./native.js";

export type BunNativeHandle = number;
export type BunPointerInput = ArrayBuffer | ArrayBufferView;
export type BunFfiSymbolType =
  | "cstring"
  | "i32"
  | "ptr"
  | "u32"
  | "u64";

export type BunFfiTypes = Readonly<Record<BunFfiSymbolType, string>>;
export type BunFfiSymbolDefinition = Readonly<{
  args?: readonly string[];
  returns?: string;
}>;
export type BunFfiSymbolsDefinition = Readonly<Record<string, BunFfiSymbolDefinition>>;
export type BunFfiLoadedLibrary<TSymbols extends Record<string, unknown> = Record<string, unknown>> = Readonly<{
  symbols: TSymbols;
}>;

export const FFIType = bunFfiType as BunFfiTypes;

export function dlopen<TSymbols extends Record<string, unknown>>(
  path: string,
  symbols: BunFfiSymbolsDefinition,
): BunFfiLoadedLibrary<TSymbols> {
  return bunDlopen(path, symbols) as BunFfiLoadedLibrary<TSymbols>;
}

export function ptr(value: BunPointerInput): BunNativeHandle {
  return bunPtr(value);
}

export function readPointer(value: BunPointerInput, offset = 0): BunNativeHandle {
  return bunRead.ptr(ptr(value), offset);
}

export const bunFfiIntrinsicsManifest = Object.freeze({
  kind: "zgml-bun-ffi-intrinsics",
  ...nativeAdapterManifestPolicy("src/ts/adapters/bun_ffi_intrinsics.ts"),
});
