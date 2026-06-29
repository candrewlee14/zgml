import {
  nativeBufferCreateDescriptorFields,
  nativeBufferExternalResourceDescriptorFields,
  type NativeBufferExternalResourceInfo,
} from "../runtime/native_buffer.js";
import type { NodeNativeSymbols } from "./node_symbols.js";
import type { NodeStatusHelpers } from "./node_status.js";

export type NodeNativeBufferSymbols = Pick<
  NodeNativeSymbols,
  | "bufferCreate"
  | "bufferWrap"
  | "bufferWrapResource"
  | "bufferSize"
  | "bufferWrite"
  | "bufferRead"
  | "bufferFree"
>;

export type NodeNativeBufferSyscallsOptions = Readonly<{
  symbols: NodeNativeBufferSymbols;
  check: NodeStatusHelpers["check"];
  handleOut: NodeStatusHelpers["handleOut"];
  readHandle: NodeStatusHelpers["readHandle"];
}>;

export type NodeNativeBufferSyscalls = Readonly<{
  createBuffer(byteLength: number): unknown;
  wrapBytes(data: ArrayBufferView, byteLength: number): unknown;
  wrapExternalResource(resource: NativeBufferExternalResourceInfo): unknown;
  bufferSize(handle: unknown): number;
  writeBuffer(handle: unknown, byteOffset: number, data: ArrayBufferView, byteLength: number): void;
  readBuffer(handle: unknown, byteOffset: number, target: ArrayBufferView, byteLength: number): void;
  freeBuffer(handle: unknown): void;
}>;

export function createNodeNativeBufferSyscalls(options: NodeNativeBufferSyscallsOptions): NodeNativeBufferSyscalls {
  const { symbols, check, handleOut, readHandle } = options;

  return Object.freeze({
    createBuffer(byteLength) {
      const out = handleOut();
      const fields = nativeBufferCreateDescriptorFields(byteLength);
      check(Number(symbols.bufferCreate({ byte_len: fields.byteLength }, out)));
      return readHandle(out, "buffer");
    },
    wrapBytes(data, byteLength) {
      const out = handleOut();
      check(Number(symbols.bufferWrap(data, byteLength, out)));
      return readHandle(out, "buffer");
    },
    wrapExternalResource(resource) {
      const out = handleOut();
      const fields = nativeBufferExternalResourceDescriptorFields(resource);
      check(Number(symbols.bufferWrapResource({
        placement: fields.placement,
        access_flags: fields.accessFlags,
        handle: fields.handle,
        byte_offset: fields.byteOffset,
        byte_len: fields.byteLength,
      }, out)));
      return readHandle(out, "buffer");
    },
    bufferSize(handle) {
      return Number(symbols.bufferSize(handle));
    },
    writeBuffer(handle, byteOffset, data, byteLength) {
      check(Number(symbols.bufferWrite(handle, byteOffset, data, byteLength)));
    },
    readBuffer(handle, byteOffset, target, byteLength) {
      check(Number(symbols.bufferRead(handle, byteOffset, target, byteLength)));
    },
    freeBuffer(handle) {
      symbols.bufferFree(handle);
    },
  });
}
