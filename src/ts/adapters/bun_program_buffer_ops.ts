import {
  setPtr,
  setUSize,
} from "./bun_abi_words.js";
import {
  programDeviceBufferImportDescriptorFields,
  resolveProgramDeviceBufferImportSource,
  type ProgramDeviceBufferImportDescriptorFields,
} from "../runtime/program_buffer_desc.js";
import type {
  ProgramDeviceBufferImportOptions,
} from "../public_api.js";

type NativeHandle = number;
type NativeOut = BigUint64Array;

type DeviceBufferImportInfo = Readonly<{
  placement: number;
  placementName: string;
  deviceHandle: NativeHandle;
  bufferHandle: NativeHandle;
  byteOffset: number;
  byteLength: number;
}>;

type BunProgramBufferSymbols = Readonly<{
  programCreateBuffer(handle: NativeHandle, kind: number, out: NativeOut): number;
  programCreateDeviceBuffer(handle: NativeHandle, kind: number, placement: number, out: NativeOut): number;
  programGetDeviceHandle(handle: NativeHandle, placement: number, out: NativeOut): number;
  programImportDeviceBuffer(handle: NativeHandle, kind: number, desc: BigUint64Array, out: NativeOut): number;
  bufferSize(handle: NativeHandle): bigint | number;
}>;

export type BunProgramBufferOpsOptions<TNativeBuffer> = Readonly<{
  symbols: BunProgramBufferSymbols;
  programBufferKinds: Readonly<Record<string, number>>;
  webgpuImportSourceKey: unknown;
  webgpuInterop: unknown;
  check(code: number): void;
  handleOut(): NativeOut;
  readHandle(out: NativeOut): NativeHandle;
  readDeviceHandle(out: NativeOut): NativeHandle;
  pointerForHandle(handle: NativeHandle): NativeHandle;
  programBufferKindId(kind: unknown): number;
  backendId(placement: unknown): number;
  normalizeWebGpuImportSource(source: unknown, importSourceKey: unknown): unknown;
  nativeBufferDeviceImportOptions(value: unknown, programHandle: NativeHandle): unknown | null;
  programDeviceBufferImportInfo(
    options: ProgramDeviceBufferImportOptions,
    config: { readonly defaults: unknown; readonly webgpuInterop: unknown },
  ): DeviceBufferImportInfo;
  createNativeBuffer(handle: NativeHandle, byteLength: number, placementName?: string): TNativeBuffer;
}>;

export function createBunProgramBufferOps<TNativeBuffer>(options: BunProgramBufferOpsOptions<TNativeBuffer>) {
  const {
    symbols,
    programBufferKinds,
    webgpuImportSourceKey,
    webgpuInterop,
    check,
    handleOut,
    readHandle,
    readDeviceHandle,
    pointerForHandle,
    programBufferKindId,
    backendId,
    normalizeWebGpuImportSource,
    nativeBufferDeviceImportOptions,
    programDeviceBufferImportInfo,
    createNativeBuffer,
  } = options;

  function nativeBufferFromHandle(bufferHandle: NativeHandle, placementName?: string): TNativeBuffer {
    return createNativeBuffer(bufferHandle, Number(symbols.bufferSize(bufferHandle)), placementName);
  }

  function programCreateOutputBuffer(handle: NativeHandle): TNativeBuffer {
    const out = handleOut();
    check(symbols.programCreateBuffer(handle, programBufferKinds.output, out));
    return nativeBufferFromHandle(readHandle(out));
  }

  function programCreateBuffer(handle: NativeHandle, kind: unknown): TNativeBuffer {
    const out = handleOut();
    check(symbols.programCreateBuffer(handle, programBufferKindId(kind), out));
    return nativeBufferFromHandle(readHandle(out));
  }

  function programCreateDeviceBuffer(handle: NativeHandle, kind: unknown, placement: unknown): TNativeBuffer {
    const out = handleOut();
    check(symbols.programCreateDeviceBuffer(handle, programBufferKindId(kind), backendId(placement), out));
    return nativeBufferFromHandle(readHandle(out), String(placement));
  }

  function programDeviceHandle(handle: NativeHandle, placement: unknown = "webgpu"): NativeHandle {
    const out = handleOut();
    check(symbols.programGetDeviceHandle(handle, backendId(placement), out));
    return readDeviceHandle(out);
  }

  function resolveDeviceBufferImportInfo(
    programHandle: NativeHandle,
    source: unknown,
    defaults: unknown = {},
  ): DeviceBufferImportInfo {
    const optionsForImport = resolveProgramDeviceBufferImportSource({
      source,
      programHandle,
      webgpuImportSourceKey,
      normalizeWebGpuImportSource,
      nativeBufferDeviceImportOptions,
    });
    return programDeviceBufferImportInfo(optionsForImport as ProgramDeviceBufferImportOptions, { defaults, webgpuInterop });
  }

  function deviceBufferImportDesc(fields: ProgramDeviceBufferImportDescriptorFields<NativeHandle>): BigUint64Array {
    const buf = new BigUint64Array(5);
    const view = new DataView(buf.buffer);
    view.setUint32(0, fields.placement, true);
    view.setUint32(4, fields.reserved, true);
    setPtr(view, 8, pointerForHandle(fields.deviceHandle));
    setPtr(view, 16, pointerForHandle(fields.bufferHandle));
    setUSize(view, 24, fields.byteOffset);
    setUSize(view, 32, fields.byteLength);
    return buf;
  }

  function programImportDeviceBuffer(
    handle: NativeHandle,
    kind: unknown,
    source: unknown,
    defaults: unknown = {},
  ): TNativeBuffer {
    const importInfo = resolveDeviceBufferImportInfo(handle, source, defaults);
    const out = handleOut();
    check(symbols.programImportDeviceBuffer(
      handle,
      programBufferKindId(kind),
      deviceBufferImportDesc(programDeviceBufferImportDescriptorFields(importInfo)),
      out,
    ));
    return nativeBufferFromHandle(readHandle(out), importInfo.placementName);
  }

  return Object.freeze({
    programCreateOutputBuffer,
    programCreateBuffer,
    programCreateDeviceBuffer,
    programDeviceHandle,
    programImportDeviceBuffer,
  });
}
