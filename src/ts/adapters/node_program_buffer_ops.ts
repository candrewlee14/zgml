import type {
  ProgramDeviceBufferImportOptions,
} from "../public_api.js";
import {
  programDeviceBufferImportDescriptorFields,
  programDeviceBufferImportDescriptorRecord,
  resolveProgramDeviceBufferImportSource,
} from "../runtime/program_buffer_desc.js";

type NativeHandle = unknown;
type NativeOut = [NativeHandle | null];

type NativeBufferLike = unknown;

type DeviceBufferImportInfo = {
  readonly placement: number;
  readonly placementName: string;
  readonly deviceHandle: unknown;
  readonly bufferHandle: unknown;
  readonly byteOffset: number;
  readonly byteLength: number;
};

type ProgramBufferSymbols = Readonly<{
  programCreateBuffer(handle: NativeHandle, kind: number, out: NativeOut): number;
  programCreateDeviceBuffer(handle: NativeHandle, kind: number, placement: number, out: NativeOut): number;
  programGetDeviceHandle(handle: NativeHandle, placement: number, out: [number]): number;
  programImportDeviceBuffer(
    handle: NativeHandle,
    kind: number,
    desc: Record<string, unknown>,
    out: NativeOut,
  ): number;
  bufferSize(handle: NativeHandle): unknown;
}>;

export type NodeProgramBufferOpsOptions<TNativeBuffer> = Readonly<{
  symbols: ProgramBufferSymbols;
  programBufferKinds: Readonly<{ output: number }>;
  webgpuImportSourceKey: unknown;
  check(code: number): void;
  handleOut(): NativeOut;
  readHandle<T = NativeHandle>(out: readonly [T | null | undefined], name: string): T;
  programBufferKindId(kind: unknown): number;
  backendId(placementName: unknown): number;
  normalizeWebGpuImportSource(source: unknown, importSourceKey: unknown): unknown;
  programDeviceBufferImportInfo(
    options: ProgramDeviceBufferImportOptions,
    config: { readonly defaults: unknown; readonly webgpuInterop: unknown },
  ): DeviceBufferImportInfo;
  webgpuInterop: unknown;
  createNativeBuffer(handle: NativeHandle, byteLength: number, placementName?: string): TNativeBuffer;
  nativeBufferDeviceImportOptions(value: unknown, programHandle: NativeHandle): unknown | null;
}>;

export function createNodeProgramBufferOps<TNativeBuffer>(options: NodeProgramBufferOpsOptions<TNativeBuffer>) {
  const {
    symbols,
    programBufferKinds,
    webgpuImportSourceKey,
    check,
    handleOut,
    readHandle,
    programBufferKindId,
    backendId,
    normalizeWebGpuImportSource,
    programDeviceBufferImportInfo,
    webgpuInterop,
    createNativeBuffer,
    nativeBufferDeviceImportOptions,
  } = options;

  function nativeBufferFromHandle(bufferHandle: NativeHandle, placementName?: string): TNativeBuffer {
    return createNativeBuffer(bufferHandle, Number(symbols.bufferSize(bufferHandle)), placementName);
  }

  function programCreateOutputBuffer(handle: NativeHandle): TNativeBuffer {
    const out = handleOut();
    check(symbols.programCreateBuffer(handle, programBufferKinds.output, out));
    return nativeBufferFromHandle(readHandle(out, "buffer"));
  }

  function programCreateBuffer(handle: NativeHandle, kind: unknown): TNativeBuffer {
    const out = handleOut();
    check(symbols.programCreateBuffer(handle, programBufferKindId(kind), out));
    return nativeBufferFromHandle(readHandle(out, "buffer"));
  }

  function programCreateDeviceBuffer(handle: NativeHandle, kind: unknown, placementName = "webgpu"): TNativeBuffer {
    const out = handleOut();
    check(symbols.programCreateDeviceBuffer(handle, programBufferKindId(kind), backendId(placementName), out));
    return nativeBufferFromHandle(readHandle(out, "buffer"), placementName);
  }

  function programDeviceHandle(handle: NativeHandle, placementName = "webgpu"): number {
    const out: [number] = [0];
    check(symbols.programGetDeviceHandle(handle, backendId(placementName), out));
    return Number(out[0]);
  }

  function programImportDeviceBuffer(
    handle: NativeHandle,
    kind: unknown,
    source: NativeBufferLike | Record<string, unknown> = {},
    defaults: unknown = {},
  ): TNativeBuffer {
    const normalizedSource = resolveProgramDeviceBufferImportSource({
      source,
      programHandle: handle,
      webgpuImportSourceKey,
      normalizeWebGpuImportSource,
      nativeBufferDeviceImportOptions,
    });
    const importInfo = programDeviceBufferImportInfo(normalizedSource as ProgramDeviceBufferImportOptions, { defaults, webgpuInterop });
    const out = handleOut();
    check(symbols.programImportDeviceBuffer(
      handle,
      programBufferKindId(kind),
      programDeviceBufferImportDescriptorRecord(programDeviceBufferImportDescriptorFields(importInfo)),
      out,
    ));
    return nativeBufferFromHandle(readHandle(out, "buffer"), importInfo.placementName);
  }

  return Object.freeze({
    programCreateOutputBuffer,
    programCreateBuffer,
    programCreateDeviceBuffer,
    programDeviceHandle,
    programImportDeviceBuffer,
  });
}
