import {
  nativeBufferCreateDescriptorFields,
  nativeBufferExternalResourceDescriptorFields,
  type NativeBufferCreateDescriptorFields,
  type NativeBufferExternalResourceDescriptorFields,
  type NativeBufferExternalResourceInfo,
} from "../runtime/native_buffer.js";
import { setUSize } from "./bun_abi_words.js";
import { ptr } from "./bun_ffi_intrinsics.js";
import type { BunHandleOut, BunNativeHandle, BunStatusHelpers } from "./bun_status.js";

export type BunNativeBufferSymbols = Readonly<{
  zgml_buffer_create(desc: BigUint64Array, outBuffer: BunHandleOut): number;
  zgml_buffer_wrap(data: BunNativeHandle, byteLen: bigint, outBuffer: BunHandleOut): number;
  zgml_buffer_wrap_resource(desc: BigUint64Array, outBuffer: BunHandleOut): number;
  zgml_buffer_size(buffer: BunNativeHandle): bigint | number;
  zgml_buffer_write(buffer: BunNativeHandle, byteOffset: bigint, src: ArrayBufferView, byteLen: bigint): number;
  zgml_buffer_read(buffer: BunNativeHandle, byteOffset: bigint, dst: ArrayBufferView, byteLen: bigint): number;
  zgml_buffer_free(buffer: BunNativeHandle): void;
}>;

export type BunNativeBufferSyscallsOptions = Readonly<{
  symbols: BunNativeBufferSymbols;
  check: BunStatusHelpers["check"];
  handleOut: BunStatusHelpers["handleOut"];
  readHandle: BunStatusHelpers["readHandle"];
}>;

export type BunNativeBufferSyscalls = Readonly<{
  createBuffer(byteLength: number): BunNativeHandle;
  wrapBytes(data: ArrayBufferView, byteLength: number): BunNativeHandle;
  wrapExternalResource(resource: NativeBufferExternalResourceInfo): BunNativeHandle;
  bufferSize(handle: BunNativeHandle): number;
  writeBuffer(handle: BunNativeHandle, byteOffset: number, data: ArrayBufferView, byteLength: number): void;
  readBuffer(handle: BunNativeHandle, byteOffset: number, target: ArrayBufferView, byteLength: number): void;
  freeBuffer(handle: BunNativeHandle): void;
}>;

function bufferDesc(fields: NativeBufferCreateDescriptorFields): BigUint64Array {
  const buf = new BigUint64Array(1);
  const view = new DataView(buf.buffer);
  setUSize(view, 0, fields.byteLength);
  return buf;
}

function externalResourceDesc(fields: NativeBufferExternalResourceDescriptorFields): BigUint64Array {
  const desc = new BigUint64Array(4);
  const view = new DataView(desc.buffer);
  view.setUint32(0, fields.placement, true);
  view.setUint32(4, fields.accessFlags, true);
  setUSize(view, 8, fields.handle);
  setUSize(view, 16, fields.byteOffset);
  setUSize(view, 24, fields.byteLength);
  return desc;
}

export function createBunNativeBufferSyscalls(options: BunNativeBufferSyscallsOptions): BunNativeBufferSyscalls {
  const { symbols, check, handleOut, readHandle } = options;

  return Object.freeze({
    createBuffer(byteLength) {
      const out = handleOut();
      check(symbols.zgml_buffer_create(bufferDesc(nativeBufferCreateDescriptorFields(byteLength)), out));
      return readHandle(out);
    },
    wrapBytes(data, byteLength) {
      const out = handleOut();
      check(symbols.zgml_buffer_wrap(ptr(data), BigInt(byteLength), out));
      return readHandle(out);
    },
    wrapExternalResource(resource) {
      const out = handleOut();
      check(symbols.zgml_buffer_wrap_resource(
        externalResourceDesc(nativeBufferExternalResourceDescriptorFields(resource)),
        out,
      ));
      return readHandle(out);
    },
    bufferSize(handle) {
      return Number(symbols.zgml_buffer_size(handle));
    },
    writeBuffer(handle, byteOffset, data, byteLength) {
      check(symbols.zgml_buffer_write(handle, BigInt(byteOffset), data, BigInt(byteLength)));
    },
    readBuffer(handle, byteOffset, target, byteLength) {
      check(symbols.zgml_buffer_read(handle, BigInt(byteOffset), target, BigInt(byteLength)));
    },
    freeBuffer(handle) {
      symbols.zgml_buffer_free(handle);
    },
  });
}
