"use strict";

import { nativeAdapterManifestPolicy } from "./native.js";

export type WebGpuInteropSymbols = Readonly<{
  deviceHandle: typeof webgpuDeviceHandle;
  bufferHandle: typeof webgpuBufferHandle;
  byteLength: typeof webgpuByteLength;
  byteOffset: typeof webgpuByteOffset;
  placement: typeof webgpuPlacement;
  importSource: typeof webgpuImportSource;
}>;

export const webgpuDeviceHandle: unique symbol = Symbol.for("zgml.webgpu.deviceHandle") as never;
export const webgpuBufferHandle: unique symbol = Symbol.for("zgml.webgpu.bufferHandle") as never;
export const webgpuByteLength: unique symbol = Symbol.for("zgml.webgpu.byteLength") as never;
export const webgpuByteOffset: unique symbol = Symbol.for("zgml.webgpu.byteOffset") as never;
export const webgpuPlacement: unique symbol = Symbol.for("zgml.webgpu.placement") as never;
export const webgpuImportSource: unique symbol = Symbol.for("zgml.webgpu.importSource") as never;

export const webgpuInterop: WebGpuInteropSymbols = Object.freeze({
  deviceHandle: webgpuDeviceHandle,
  bufferHandle: webgpuBufferHandle,
  byteLength: webgpuByteLength,
  byteOffset: webgpuByteOffset,
  placement: webgpuPlacement,
  importSource: webgpuImportSource,
});

export const webgpuInteropManifest = Object.freeze({
  kind: "zgml-webgpu-interop-symbols",
  ...nativeAdapterManifestPolicy("src/ts/adapters/webgpu_interop.ts"),
  symbolRegistry: "Symbol.for",
});
