"use strict";

import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";

export type ProgramDeviceBufferImportDescriptorInput<THandle = unknown> = Readonly<{
  placement: number;
  deviceHandle: THandle;
  bufferHandle: THandle;
  byteOffset: number;
  byteLength: number;
}>;

export type ProgramDeviceBufferImportDescriptorFields<THandle = unknown> = Readonly<{
  placement: number;
  reserved: 0;
  deviceHandle: THandle;
  bufferHandle: THandle;
  byteOffset: number;
  byteLength: number;
}>;

export type ProgramDeviceBufferImportDescriptorRecord<THandle = unknown> = Readonly<{
  placement: number;
  reserved: 0;
  device_handle: THandle;
  buffer_handle: THandle;
  byte_offset: number;
  byte_len: number;
}>;

export type ProgramDeviceBufferImportSourceResolverOptions = Readonly<{
  source: unknown;
  programHandle: unknown;
  webgpuImportSourceKey: unknown;
  normalizeWebGpuImportSource(source: unknown, importSourceKey: unknown): unknown;
  nativeBufferDeviceImportOptions(value: unknown, programHandle: unknown): unknown | null;
}>;

export function resolveProgramDeviceBufferImportSource(
  options: ProgramDeviceBufferImportSourceResolverOptions,
): unknown {
  return options.nativeBufferDeviceImportOptions(options.source, options.programHandle)
    ?? options.normalizeWebGpuImportSource(options.source, options.webgpuImportSourceKey);
}

export function programDeviceBufferImportDescriptorFields<THandle>(
  input: ProgramDeviceBufferImportDescriptorInput<THandle>,
): ProgramDeviceBufferImportDescriptorFields<THandle> {
  return Object.freeze({
    placement: input.placement,
    reserved: 0,
    deviceHandle: input.deviceHandle,
    bufferHandle: input.bufferHandle,
    byteOffset: input.byteOffset,
    byteLength: input.byteLength,
  });
}

export function programDeviceBufferImportDescriptorRecord<THandle>(
  fields: ProgramDeviceBufferImportDescriptorFields<THandle>,
): ProgramDeviceBufferImportDescriptorRecord<THandle> {
  return Object.freeze({
    placement: fields.placement,
    reserved: fields.reserved,
    device_handle: fields.deviceHandle,
    buffer_handle: fields.bufferHandle,
    byte_offset: fields.byteOffset,
    byte_len: fields.byteLength,
  });
}

export const programBufferDescriptorManifest = Object.freeze({
  kind: "zgml-program-buffer-descriptor",
  ...tsRuntimeManifestPolicy("src/ts/runtime/program_buffer_desc.ts", "ProgramDevice -> buffer import descriptor -> Program"),
});
