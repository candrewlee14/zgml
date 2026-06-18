"use strict";

import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";
import {
  backendIds,
  backendId,
  resourceAccessFlags,
} from "./abi.js";
import type {
  ByteLike,
  BufferInspection,
  IndexLike,
  NativeBuffer as PublicNativeBuffer,
  NativeBufferByteRangeElementType,
  NativeBufferByteRangeInfo,
  NativeBufferByteRangeOperation,
  NativeBufferDeviceImportInfo,
  NativeBufferExternalResourceInfo,
  ExternalResourceOptions,
  ProgramDeviceBufferImportOptions,
  ProgramDeviceBufferImportInfo,
} from "../public_api.js";

export type {
  ByteLike,
  BufferInspection,
  IndexLike,
  NativeBufferByteRangeElementType,
  NativeBufferByteRangeInfo,
  NativeBufferByteRangeOperation,
  NativeBufferDeviceImportInfo,
  NativeBufferExternalResourceInfo,
  ExternalResourceOptions,
  ProgramDeviceBufferImportOptions,
  ProgramDeviceBufferImportInfo,
} from "../public_api.js";

type AnyRecord = Record<string, any>;

export type NativeBufferLifetimeTarget = PublicNativeBuffer & Readonly<{
  assertAlive(): void;
}>;

export type NativeBufferLifetimeHelpers<Buffer extends NativeBufferLifetimeTarget = NativeBufferLifetimeTarget> = Readonly<{
  requireNativeBuffer(value: unknown, name: string): Buffer | null;
  uniqueNativeBuffers(values: readonly unknown[]): readonly Buffer[];
}>;

export type NativeBufferLifetimeHelpersOptions<Buffer extends NativeBufferLifetimeTarget = NativeBufferLifetimeTarget> = Readonly<{
  isNativeBuffer: (value: unknown) => value is Buffer;
}>;

export type NativeBufferInitOptions = Readonly<{
  backing?: ArrayBufferView | null;
  externalResource?: boolean;
  devicePlacement?: unknown | null;
}>;

export type NativeBufferSurfaceClass<Buffer extends PublicNativeBuffer = PublicNativeBuffer> = {
  new (handle: unknown, byteLength: number, options?: NativeBufferInitOptions): Buffer;
  readonly prototype: Buffer;
  create(byteLength: number): Buffer;
  wrapBytes(data: ArrayBufferView): Buffer;
  wrapFloat32(data: Float32Array): Buffer;
  externalResource(options: ExternalResourceOptions): Buffer;
  fromFloat32(values: IndexLike): Buffer;
  fromBytes(values: ByteLike): Buffer;
  fromNativeHandle(handle: unknown, byteLength: number, devicePlacement?: unknown | null): Buffer;
};

export type NativeBufferFacade<Buffer extends PublicNativeBuffer = PublicNativeBuffer> = Readonly<{
  initialize(buffer: Buffer, handle: unknown, byteLength: unknown, init?: NativeBufferInitOptions): void;
  assertAlive(buffer: Buffer): void;
  fromNativeHandle(handle: unknown, byteLength: unknown, devicePlacement?: unknown | null): Buffer;
  create(byteLength: unknown): Buffer;
  wrapBytes(data: ArrayBufferView): Buffer;
  wrapFloat32(data: unknown): Buffer;
  externalResource(options: ExternalResourceOptions): Buffer;
  fromFloat32(values: unknown): Buffer;
  fromBytes(values: unknown): Buffer;
  handleValue(buffer: Buffer): unknown;
  size(buffer: Buffer): number;
  inspect(buffer: Buffer): BufferInspection;
  deviceImportOptions(buffer: Buffer, programHandle: unknown): NativeBufferDeviceImportInfo;
  writeFloat32(buffer: Buffer, values: unknown, byteOffset?: unknown): Buffer;
  writeBytes(buffer: Buffer, values: unknown, byteOffset?: unknown): Buffer;
  readFloat32(buffer: Buffer, length?: unknown, byteOffset?: unknown): Float32Array;
  readBytes(buffer: Buffer, length?: unknown, byteOffset?: unknown): Uint8Array;
  readFloat32Into(buffer: Buffer, target: Float32Array, length?: unknown, byteOffset?: unknown): Float32Array;
  readBytesInto(buffer: Buffer, target: Uint8Array, byteLength?: unknown, byteOffset?: unknown): Uint8Array;
  free(buffer: Buffer): void;
  dispose(buffer: Buffer): void;
}>;

type NativeBufferF32Callback = (values: unknown) => Float32Array;
type NativeBufferByteViewCallback = (values: unknown, label: string) => ArrayBufferView;
type NativeBufferHandleCallback = (byteLength: number) => unknown;
type NativeBufferWrapBytesCallback = (data: ArrayBufferView, byteLength: number) => unknown;
type NativeBufferWrapExternalResourceCallback = (resource: NativeBufferExternalResourceInfo) => unknown;
type NativeBufferHandlePredicate = {
  bivarianceHack(handle: any): boolean;
}["bivarianceHack"];
type NativeBufferSizeCallback = {
  bivarianceHack(handle: any): number;
}["bivarianceHack"];
type NativeBufferInspectionCallback = {
  bivarianceHack(handle: any): unknown;
}["bivarianceHack"];
type NativeBufferIoCallback = {
  bivarianceHack(handle: any, byteOffset: number, data: ArrayBufferView, byteLength: number): void;
}["bivarianceHack"];
type NativeBufferFreeCallback = {
  bivarianceHack(handle: any): void;
}["bivarianceHack"];
type NativeBufferProgramDeviceHandleCallback = {
  bivarianceHack(programHandle: any, placement: unknown): number;
}["bivarianceHack"];

export const nativeBufferManifest = Object.freeze({
  kind: "zgml-native-buffer",
  ...tsRuntimeManifestPolicy("src/ts/runtime/native_buffer.ts", "Program -> Session -> NativeBuffer"),
});

export function readImportField(source: unknown, keys: readonly unknown[]) {
  if (typeof source !== "object" || source === null) return undefined;
  const record = source as AnyRecord;
  for (const key of keys) {
    const value = record[key as any];
    if (value !== undefined) return value;
  }
  return undefined;
}

export function readImportString(source: unknown, keys: readonly unknown[]) {
  const value = readImportField(source, keys);
  return typeof value === "string" ? value : undefined;
}

export function normalizeWebGpuImportSource(source: unknown, importSourceKey: unknown) {
  const factory = readImportField(source, [importSourceKey]);
  if (typeof factory === "function") return factory.call(source);
  return source;
}

function isNonNegativeSafeInteger(value: unknown) {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function isPositiveSafeInteger(value: unknown) {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function isRecord(value: unknown): value is AnyRecord {
  return value !== null && typeof value === "object";
}

export function programDeviceBufferImportSignature(info: {
  readonly placementName: string;
  readonly placement: number;
  readonly deviceHandle: number;
  readonly bufferHandle: number;
  readonly byteOffset: number;
  readonly explicitByteLength?: number;
  readonly byteLength: number;
}) {
  return [
    "native-buffer-device-buffer-import",
    `placement=${info.placementName}`,
    `placementId=${info.placement}`,
    `device=${info.deviceHandle}`,
    `buffer=${info.bufferHandle}`,
    `offset=${info.byteOffset}`,
    `explicitBytes=${info.explicitByteLength ?? "none"}`,
    `bytes=${info.byteLength}`,
  ].join("|");
}

export function nativeBufferDeviceImportSignature(info: {
  readonly placement: string;
  readonly deviceHandle: number;
  readonly bufferHandle: number;
  readonly byteOffset: number;
  readonly byteLength: number;
}) {
  return [
    "native-buffer-device-import",
    `placement=${info.placement}`,
    `device=${info.deviceHandle}`,
    `buffer=${info.bufferHandle}`,
    `offset=${info.byteOffset}`,
    `bytes=${info.byteLength}`,
  ].join("|");
}

export function nativeBufferExternalResourceSignature(info: {
  readonly placementName: string;
  readonly placement: number;
  readonly accessFlags: number;
  readonly handle: number;
  readonly byteOffset: number;
  readonly byteLength: number;
}) {
  return [
    "native-buffer-external-resource",
    `placement=${info.placementName}`,
    `placementId=${info.placement}`,
    `access=${info.accessFlags}`,
    `handle=${info.handle}`,
    `offset=${info.byteOffset}`,
    `bytes=${info.byteLength}`,
  ].join("|");
}

export function programDeviceBufferImportInfo(options: ProgramDeviceBufferImportOptions, config: AnyRecord): ProgramDeviceBufferImportInfo {
  const defaults = (config && config.defaults) || {};
  const webgpuInterop = (config && config.webgpuInterop) || {};
  const placementName =
    readImportString(options, [webgpuInterop.placement, "placement", "backend", "device"]) ??
    defaults.placement ??
    "webgpu";
  const placement = backendId(placementName);
  const deviceHandle =
    readImportField(options, [webgpuInterop.deviceHandle, "deviceHandle", "wgpuDeviceHandle"]) ??
    defaults.deviceHandle;
  const bufferHandle = readImportField(options, [webgpuInterop.bufferHandle, "bufferHandle", "wgpuBufferHandle", "handle"]);
  const byteOffset = readImportField(options, [webgpuInterop.byteOffset, "byteOffset"]) ?? 0;
  const explicitByteLength = readImportField(options, [webgpuInterop.byteLength, "byteLength", "byteLen", "size"]);
  const byteLength = explicitByteLength ?? 0;
  if (!Number.isSafeInteger(deviceHandle) || deviceHandle <= 0) {
    throw new Error(`invalid device handle: ${deviceHandle}`);
  }
  if (!Number.isSafeInteger(bufferHandle) || bufferHandle <= 0) {
    throw new Error(`invalid device buffer handle: ${bufferHandle}`);
  }
  if (!Number.isSafeInteger(byteOffset) || byteOffset < 0) {
    throw new Error(`invalid device buffer byteOffset: ${byteOffset}`);
  }
  if (explicitByteLength !== undefined && (!Number.isSafeInteger(byteLength) || byteLength <= 0)) {
    throw new Error(`invalid device buffer byteLength: ${byteLength}`);
  }
  const signature = programDeviceBufferImportSignature({
    placementName,
    placement,
    deviceHandle,
    bufferHandle,
    byteOffset,
    explicitByteLength,
    byteLength,
  });
  const info: AnyRecord = {
    kind: "zgml.native-buffer.device-buffer-import",
    signature,
    placementName,
    placement,
    deviceHandle,
    bufferHandle,
    byteOffset,
    explicitByteLength,
    byteLength,
  };
  return Object.freeze(info) as ProgramDeviceBufferImportInfo;
}

export function nativeBufferDeviceImportOptions(placementName: string | null | undefined, inspection: AnyRecord | null | undefined, deviceHandle: unknown, label = "importDeviceBuffer(NativeBuffer)"): NativeBufferDeviceImportInfo {
  if (placementName === null || placementName === undefined) {
    throw new Error(`${label} requires a zgml-created device buffer`);
  }
  if (!inspection || typeof inspection !== "object") {
    throw new Error(`${label} requires NativeBuffer inspection`);
  }
  if (
    inspection.storage !== "external-resource" ||
    inspection.placement !== placementName ||
    !Number.isSafeInteger(inspection.handle) ||
    inspection.handle <= 0
  ) {
    throw new Error(`NativeBuffer is not an importable ${placementName} device buffer`);
  }
  if (!Number.isSafeInteger(deviceHandle) || (deviceHandle as number) <= 0) {
    throw new Error(`invalid ${placementName} device handle: ${deviceHandle}`);
  }
  if (!Number.isSafeInteger(inspection.byteOffset) || inspection.byteOffset < 0) {
    throw new Error(`invalid ${placementName} device buffer byteOffset: ${inspection.byteOffset}`);
  }
  if (!Number.isSafeInteger(inspection.byteLength) || inspection.byteLength <= 0) {
    throw new Error(`invalid ${placementName} device buffer byteLength: ${inspection.byteLength}`);
  }
  const info: NativeBufferDeviceImportInfo = Object.freeze({
    kind: "zgml.native-buffer.device-import",
    signature: nativeBufferDeviceImportSignature({
      placement: placementName,
      deviceHandle: deviceHandle as number,
      bufferHandle: inspection.handle,
      byteOffset: inspection.byteOffset,
      byteLength: inspection.byteLength,
    }),
    placement: placementName,
    deviceHandle: deviceHandle as number,
    bufferHandle: inspection.handle,
    byteOffset: inspection.byteOffset,
    byteLength: inspection.byteLength,
  });
  return info;
}

export function externalResourceInfo(options: ExternalResourceOptions): NativeBufferExternalResourceInfo {
  const placementName = options.placement ?? options.backend ?? "webgpu";
  const placement = backendId(placementName);
  if (placement === backendIds.auto) {
    throw new Error("NativeBuffer.externalResource placement cannot be auto");
  }
  const byteOffset = options.byteOffset ?? 0;
  if (!Number.isSafeInteger(options.handle) || options.handle <= 0) {
    throw new Error(`invalid external resource handle: ${options.handle}`);
  }
  if (!Number.isSafeInteger(byteOffset) || byteOffset < 0) {
    throw new Error(`invalid external resource byteOffset: ${byteOffset}`);
  }
  if (!Number.isSafeInteger(options.byteLength) || options.byteLength <= 0) {
    throw new Error(`invalid external resource byteLength: ${options.byteLength}`);
  }
  const info: NativeBufferExternalResourceInfo = Object.freeze({
    kind: "zgml.native-buffer.external-resource",
    signature: nativeBufferExternalResourceSignature({
      placementName,
      placement,
      accessFlags: resourceAccessFlags(options.access),
      handle: options.handle,
      byteOffset,
      byteLength: options.byteLength,
    }),
    placementName,
    placement,
    accessFlags: resourceAccessFlags(options.access),
    handle: options.handle,
    byteOffset,
    byteLength: options.byteLength,
  });
  return info;
}

export function nativeBufferWrapByteLength(data: unknown) {
  if (!ArrayBuffer.isView(data) || data.byteLength <= 0) {
    throw new Error("NativeBuffer.wrapBytes requires a non-empty typed array or DataView");
  }
  return data.byteLength;
}

export function validateNativeBufferWrapFloat32(data: unknown) {
  if (!(data instanceof Float32Array)) {
    throw new Error("NativeBuffer.wrapFloat32 requires a Float32Array");
  }
  return data;
}

export function nativeBufferCreateByteLength(byteLength: unknown) {
  if (!Number.isSafeInteger(byteLength) || (byteLength as number) <= 0) {
    throw new Error(`invalid zgml buffer byte length: ${byteLength}`);
  }
  return byteLength as number;
}

export type NativeBufferCreateDescriptorFields = Readonly<{
  byteLength: number;
}>;

export type NativeBufferExternalResourceDescriptorFields = Readonly<{
  placement: number;
  accessFlags: number;
  handle: number;
  byteOffset: number;
  byteLength: number;
}>;

export function nativeBufferCreateDescriptorFields(byteLength: number): NativeBufferCreateDescriptorFields {
  return Object.freeze({
    byteLength,
  });
}

export function nativeBufferExternalResourceDescriptorFields(
  resource: NativeBufferExternalResourceInfo,
): NativeBufferExternalResourceDescriptorFields {
  return Object.freeze({
    placement: resource.placement,
    accessFlags: resource.accessFlags,
    handle: resource.handle,
    byteOffset: resource.byteOffset,
    byteLength: resource.byteLength,
  });
}

function nativeBufferByteRangeSignature(fields: {
  readonly operation: NativeBufferByteRangeOperation;
  readonly elementType: NativeBufferByteRangeElementType;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly length: number | null;
}) {
  return [
    "native-buffer-byte-range",
    `op=${fields.operation}`,
    `element=${fields.elementType}`,
    `offset=${fields.byteOffset}`,
    `bytes=${fields.byteLength}`,
    `length=${fields.length === null ? "null" : fields.length}`,
  ].join("|");
}

export function nativeBufferByteRangeInfo(fields: {
  readonly operation: NativeBufferByteRangeOperation;
  readonly elementType: NativeBufferByteRangeElementType;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly length?: number | null;
}): NativeBufferByteRangeInfo {
  const length = fields.length ?? null;
  return Object.freeze({
    kind: "zgml.native-buffer.byte-range",
    signature: nativeBufferByteRangeSignature({
      operation: fields.operation,
      elementType: fields.elementType,
      byteOffset: fields.byteOffset,
      byteLength: fields.byteLength,
      length,
    }),
    operation: fields.operation,
    elementType: fields.elementType,
    byteOffset: fields.byteOffset,
    byteLength: fields.byteLength,
    length,
  });
}

export function isNativeBufferByteRangeInfo(info: unknown): info is NativeBufferByteRangeInfo {
  if (!isRecord(info) || !Object.isFrozen(info)) return false;
  if (info.kind !== "zgml.native-buffer.byte-range" || typeof info.signature !== "string") return false;
  if (!["write", "read-float32-into", "read-bytes-into"].includes(info.operation)) return false;
  if (!["bytes", "f32"].includes(info.elementType)) return false;
  if (!isNonNegativeSafeInteger(info.byteOffset) || !isNonNegativeSafeInteger(info.byteLength)) return false;
  if (info.length !== null && !isNonNegativeSafeInteger(info.length)) return false;
  return info.signature === nativeBufferByteRangeSignature({
    operation: info.operation,
    elementType: info.elementType,
    byteOffset: info.byteOffset,
    byteLength: info.byteLength,
    length: info.length,
  });
}

export function requireNativeBufferByteRangeInfo(info: unknown): NativeBufferByteRangeInfo {
  if (isNativeBufferByteRangeInfo(info)) return info;
  throw new Error("expected frozen NativeBuffer byte-range evidence");
}

export const assertNativeBufferByteRangeInfo = requireNativeBufferByteRangeInfo;
export const assert_native_buffer_byte_range_info = requireNativeBufferByteRangeInfo;

export function matchesNativeBufferByteRangeSignature(info: unknown, signature: unknown) {
  return isNativeBufferByteRangeInfo(info) && typeof signature === "string" && info.signature === signature;
}

export const matches_native_buffer_byte_range_signature = matchesNativeBufferByteRangeSignature;

export function isProgramDeviceBufferImportInfo(info: unknown): info is ProgramDeviceBufferImportInfo {
  if (!isRecord(info) || !Object.isFrozen(info)) return false;
  if (info.kind !== "zgml.native-buffer.device-buffer-import" || typeof info.signature !== "string") return false;
  if (typeof info.placementName !== "string" || info.placementName.length === 0) return false;
  if (!isNonNegativeSafeInteger(info.placement)) return false;
  if (!isPositiveSafeInteger(info.deviceHandle) || !isPositiveSafeInteger(info.bufferHandle)) return false;
  if (!isNonNegativeSafeInteger(info.byteOffset) || !isNonNegativeSafeInteger(info.byteLength)) return false;
  if (info.explicitByteLength !== undefined && !isPositiveSafeInteger(info.explicitByteLength)) return false;
  return info.signature === programDeviceBufferImportSignature(info as ProgramDeviceBufferImportInfo);
}

export function requireProgramDeviceBufferImportInfo(info: unknown): ProgramDeviceBufferImportInfo {
  if (isProgramDeviceBufferImportInfo(info)) return info;
  throw new Error("expected frozen ProgramDevice buffer import evidence");
}

export const assertProgramDeviceBufferImportInfo = requireProgramDeviceBufferImportInfo;
export const assert_program_device_buffer_import_info = requireProgramDeviceBufferImportInfo;

export function matchesProgramDeviceBufferImportSignature(info: unknown, signature: unknown) {
  return isProgramDeviceBufferImportInfo(info) && typeof signature === "string" && info.signature === signature;
}

export const matches_program_device_buffer_import_signature = matchesProgramDeviceBufferImportSignature;

export function isNativeBufferDeviceImportInfo(info: unknown): info is NativeBufferDeviceImportInfo {
  if (!isRecord(info) || !Object.isFrozen(info)) return false;
  if (info.kind !== "zgml.native-buffer.device-import" || typeof info.signature !== "string") return false;
  if (typeof info.placement !== "string" || info.placement.length === 0) return false;
  if (!isPositiveSafeInteger(info.deviceHandle) || !isPositiveSafeInteger(info.bufferHandle)) return false;
  if (!isNonNegativeSafeInteger(info.byteOffset) || !isPositiveSafeInteger(info.byteLength)) return false;
  return info.signature === nativeBufferDeviceImportSignature(info as NativeBufferDeviceImportInfo);
}

export function requireNativeBufferDeviceImportInfo(info: unknown): NativeBufferDeviceImportInfo {
  if (isNativeBufferDeviceImportInfo(info)) return info;
  throw new Error("expected frozen NativeBuffer device import evidence");
}

export const assertNativeBufferDeviceImportInfo = requireNativeBufferDeviceImportInfo;
export const assert_native_buffer_device_import_info = requireNativeBufferDeviceImportInfo;

export function matchesNativeBufferDeviceImportSignature(info: unknown, signature: unknown) {
  return isNativeBufferDeviceImportInfo(info) && typeof signature === "string" && info.signature === signature;
}

export const matches_native_buffer_device_import_signature = matchesNativeBufferDeviceImportSignature;

export function isNativeBufferExternalResourceInfo(info: unknown): info is NativeBufferExternalResourceInfo {
  if (!isRecord(info) || !Object.isFrozen(info)) return false;
  if (info.kind !== "zgml.native-buffer.external-resource" || typeof info.signature !== "string") return false;
  if (typeof info.placementName !== "string" || info.placementName.length === 0) return false;
  if (!isNonNegativeSafeInteger(info.placement) || !isNonNegativeSafeInteger(info.accessFlags)) return false;
  if (!isPositiveSafeInteger(info.handle) || !isNonNegativeSafeInteger(info.byteOffset) || !isPositiveSafeInteger(info.byteLength)) return false;
  return info.signature === nativeBufferExternalResourceSignature(info as NativeBufferExternalResourceInfo);
}

export function requireNativeBufferExternalResourceInfo(info: unknown): NativeBufferExternalResourceInfo {
  if (isNativeBufferExternalResourceInfo(info)) return info;
  throw new Error("expected frozen NativeBuffer external-resource evidence");
}

export const assertNativeBufferExternalResourceInfo = requireNativeBufferExternalResourceInfo;
export const assert_native_buffer_external_resource_info = requireNativeBufferExternalResourceInfo;

export function matchesNativeBufferExternalResourceSignature(info: unknown, signature: unknown) {
  return isNativeBufferExternalResourceInfo(info) && typeof signature === "string" && info.signature === signature;
}

export const matches_native_buffer_external_resource_signature = matchesNativeBufferExternalResourceSignature;

export function nativeBufferWriteInfo(bufferByteLength: unknown, dataByteLength: unknown, byteOffset: unknown = 0, label = "NativeBuffer.write") {
  if (!Number.isSafeInteger(bufferByteLength) || (bufferByteLength as number) < 0) {
    throw new Error(`${label} requires a valid buffer byteLength`);
  }
  if (!Number.isSafeInteger(dataByteLength) || (dataByteLength as number) < 0) {
    throw new Error(`invalid ${label} byteLength: ${dataByteLength}`);
  }
  const normalizedDataByteLength = dataByteLength as number;
  const normalizedOffsetValue = byteOffset ?? 0;
  if (!Number.isSafeInteger(normalizedOffsetValue) || (normalizedOffsetValue as number) < 0) {
    throw new Error(`invalid ${label} byteOffset: ${byteOffset}`);
  }
  const normalizedOffset = normalizedOffsetValue as number;
  if (normalizedOffset + normalizedDataByteLength > (bufferByteLength as number)) {
    throw new Error(`${label} byte range ${normalizedOffset}..${normalizedOffset + normalizedDataByteLength} exceeds buffer byteLength ${bufferByteLength}`);
  }
  return nativeBufferByteRangeInfo({
    operation: "write",
    elementType: "bytes",
    byteOffset: normalizedOffset,
    byteLength: normalizedDataByteLength,
  });
}

export function validateNativeBufferReadFloat32Into(target: unknown, length: unknown) {
  if (!(target instanceof Float32Array)) {
    throw new Error("NativeBuffer.readFloat32Into requires a Float32Array target");
  }
  if (!Number.isSafeInteger(length) || (length as number) < 0 || (length as number) > target.length) {
    throw new Error(`invalid NativeBuffer.readFloat32Into length: ${length}`);
  }
  return length as number;
}

export function validateNativeBufferReadBytesInto(target: unknown, byteLength: unknown) {
  if (!(target instanceof Uint8Array)) {
    throw new Error("NativeBuffer.readBytesInto requires a Uint8Array target");
  }
  if (!Number.isSafeInteger(byteLength) || (byteLength as number) < 0 || (byteLength as number) > target.byteLength) {
    throw new Error(`invalid NativeBuffer.readBytesInto byteLength: ${byteLength}`);
  }
  return byteLength as number;
}

function validateNativeBufferReadOffset(bufferByteLength: unknown, byteOffset: unknown, label: string) {
  if (!Number.isSafeInteger(bufferByteLength) || (bufferByteLength as number) < 0) {
    throw new Error(`${label} requires a valid buffer byteLength`);
  }
  const normalizedOffset = byteOffset ?? 0;
  if (!Number.isSafeInteger(normalizedOffset) || (normalizedOffset as number) < 0) {
    throw new Error(`invalid ${label} byteOffset: ${byteOffset}`);
  }
  if ((normalizedOffset as number) > (bufferByteLength as number)) {
    throw new Error(`${label} byteOffset ${normalizedOffset} exceeds buffer byteLength ${bufferByteLength}`);
  }
  return normalizedOffset as number;
}

export function nativeBufferReadFloat32IntoInfo(bufferByteLength: unknown, target: Float32Array, length: unknown, byteOffset: unknown = 0) {
  const label = "NativeBuffer.readFloat32Into";
  const normalizedLength = validateNativeBufferReadFloat32Into(target, length) as number;
  const normalizedOffset = validateNativeBufferReadOffset(bufferByteLength, byteOffset, label);
  if (normalizedOffset % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error(`${label} byteOffset must be ${Float32Array.BYTES_PER_ELEMENT}-byte aligned, got ${normalizedOffset}`);
  }
  const byteLength = normalizedLength * Float32Array.BYTES_PER_ELEMENT;
  if (normalizedOffset + byteLength > (bufferByteLength as number)) {
    throw new Error(`${label} byte range ${normalizedOffset}..${normalizedOffset + byteLength} exceeds buffer byteLength ${bufferByteLength}`);
  }
  return nativeBufferByteRangeInfo({
    operation: "read-float32-into",
    elementType: "f32",
    length: normalizedLength,
    byteOffset: normalizedOffset,
    byteLength,
  });
}

export function nativeBufferReadBytesIntoInfo(bufferByteLength: unknown, target: Uint8Array, byteLength: unknown, byteOffset: unknown = 0) {
  const label = "NativeBuffer.readBytesInto";
  const normalizedByteLength = validateNativeBufferReadBytesInto(target, byteLength) as number;
  const normalizedOffset = validateNativeBufferReadOffset(bufferByteLength, byteOffset, label);
  if (normalizedOffset + normalizedByteLength > (bufferByteLength as number)) {
    throw new Error(`${label} byte range ${normalizedOffset}..${normalizedOffset + normalizedByteLength} exceeds buffer byteLength ${bufferByteLength}`);
  }
  return nativeBufferByteRangeInfo({
    operation: "read-bytes-into",
    elementType: "bytes",
    byteLength: normalizedByteLength,
    byteOffset: normalizedOffset,
    length: normalizedByteLength,
  });
}

export function nativeBufferReadFloat32Target(bufferByteLength: unknown, length: unknown, byteOffset: unknown = 0) {
  const label = "NativeBuffer.readFloat32";
  const normalizedOffset = validateNativeBufferReadOffset(bufferByteLength, byteOffset, label);
  if (normalizedOffset % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error(`${label} byteOffset must be ${Float32Array.BYTES_PER_ELEMENT}-byte aligned, got ${normalizedOffset}`);
  }
  const available = Math.floor(((bufferByteLength as number) - normalizedOffset) / Float32Array.BYTES_PER_ELEMENT);
  const normalizedLength = length ?? available;
  if (!Number.isSafeInteger(normalizedLength) || (normalizedLength as number) < 0) {
    throw new Error(`invalid ${label} length: ${length}`);
  }
  const readByteLength = (normalizedLength as number) * Float32Array.BYTES_PER_ELEMENT;
  if (normalizedOffset + readByteLength > (bufferByteLength as number)) {
    throw new Error(`${label} byte range ${normalizedOffset}..${normalizedOffset + readByteLength} exceeds buffer byteLength ${bufferByteLength}`);
  }
  return {
    target: new Float32Array(normalizedLength as number),
    length: normalizedLength,
    byteOffset: normalizedOffset,
  };
}

export function nativeBufferReadBytesTarget(bufferByteLength: unknown, byteLength: unknown, byteOffset: unknown = 0) {
  const label = "NativeBuffer.readBytes";
  const normalizedOffset = validateNativeBufferReadOffset(bufferByteLength, byteOffset, label);
  const normalizedByteLength = byteLength ?? ((bufferByteLength as number) - normalizedOffset);
  if (!Number.isSafeInteger(normalizedByteLength) || (normalizedByteLength as number) < 0) {
    throw new Error(`invalid ${label} length: ${byteLength}`);
  }
  if (normalizedOffset + (normalizedByteLength as number) > (bufferByteLength as number)) {
    throw new Error(`${label} byte range ${normalizedOffset}..${normalizedOffset + (normalizedByteLength as number)} exceeds buffer byteLength ${bufferByteLength}`);
  }
  return {
    target: new Uint8Array(normalizedByteLength as number),
    byteLength: normalizedByteLength,
    byteOffset: normalizedOffset,
  };
}

export type NativeBufferFacadeHelpersOptions = Readonly<{
  NativeBuffer: NativeBufferSurfaceClass;
  nullHandle?: unknown;
  f32: NativeBufferF32Callback;
  byteView: NativeBufferByteViewCallback;
  isLiveHandle: NativeBufferHandlePredicate;
  createBuffer: NativeBufferHandleCallback;
  wrapBytes: NativeBufferWrapBytesCallback;
  wrapExternalResource: NativeBufferWrapExternalResourceCallback;
  bufferSize: NativeBufferSizeCallback;
  inspectBuffer: NativeBufferInspectionCallback;
  writeBuffer: NativeBufferIoCallback;
  readBuffer: NativeBufferIoCallback;
  freeBuffer: NativeBufferFreeCallback;
  programDeviceHandle: NativeBufferProgramDeviceHandleCallback;
}>;

export function createNativeBufferFacadeHelpers<Buffer extends PublicNativeBuffer = PublicNativeBuffer>(options: NativeBufferFacadeHelpersOptions): NativeBufferFacade<Buffer> {
  const NativeBufferClass = options.NativeBuffer as NativeBufferSurfaceClass<Buffer>;
  const nullHandle = Object.prototype.hasOwnProperty.call(options, "nullHandle")
    ? options.nullHandle
    : null;
  const f32 = options.f32;
  const byteView = options.byteView;
  const isLiveHandle = options.isLiveHandle;
  const createBuffer = options.createBuffer;
  const wrapBytes = options.wrapBytes;
  const wrapExternalResource = options.wrapExternalResource;
  const bufferSize = options.bufferSize;
  const inspectBuffer = options.inspectBuffer;
  const writeBuffer = options.writeBuffer;
  const readBuffer = options.readBuffer;
  const freeBuffer = options.freeBuffer;
  const programDeviceHandle = options.programDeviceHandle;

  if (typeof NativeBufferClass !== "function") {
    throw new Error("createNativeBufferFacadeHelpers requires NativeBuffer");
  }
  if (typeof f32 !== "function") {
    throw new Error("createNativeBufferFacadeHelpers requires f32");
  }
  if (typeof byteView !== "function") {
    throw new Error("createNativeBufferFacadeHelpers requires byteView");
  }
  if (typeof isLiveHandle !== "function") {
    throw new Error("createNativeBufferFacadeHelpers requires isLiveHandle");
  }
  if (
    typeof createBuffer !== "function" ||
    typeof wrapBytes !== "function" ||
    typeof wrapExternalResource !== "function" ||
    typeof bufferSize !== "function" ||
    typeof inspectBuffer !== "function" ||
    typeof writeBuffer !== "function" ||
    typeof readBuffer !== "function" ||
    typeof freeBuffer !== "function" ||
    typeof programDeviceHandle !== "function"
  ) {
    throw new Error("createNativeBufferFacadeHelpers requires native buffer callbacks");
  }

  function initialize(buffer: AnyRecord, handle: unknown, byteLength: unknown, init: NativeBufferInitOptions = {}) {
    buffer.handle = handle;
    buffer.byteLength = byteLength;
    buffer.backing = init.backing ?? null;
    buffer.externalResource = Boolean(init.externalResource);
    buffer.devicePlacement = init.devicePlacement ?? null;
  }

  function assertAlive(buffer: AnyRecord) {
    if (!isLiveHandle(buffer.handle)) {
      throw new Error("buffer was already freed");
    }
  }

  function fromNativeHandle(handle: unknown, byteLength: unknown, devicePlacement = null) {
    return new NativeBufferClass(handle, nativeBufferCreateByteLength(byteLength), { devicePlacement });
  }

  function create(byteLength: unknown) {
    const normalizedByteLength = nativeBufferCreateByteLength(byteLength);
    return new NativeBufferClass(createBuffer(normalizedByteLength), normalizedByteLength);
  }

  function wrapBytesBuffer(data: ArrayBufferView) {
    const byteLength = nativeBufferWrapByteLength(data);
    return new NativeBufferClass(wrapBytes(data, byteLength), byteLength, { backing: data });
  }

  function wrapFloat32(data: unknown) {
    validateNativeBufferWrapFloat32(data);
    return wrapBytesBuffer(data as Float32Array);
  }

  function externalResource(options: ExternalResourceOptions) {
    const resource = externalResourceInfo(options);
    return new NativeBufferClass(wrapExternalResource(resource), resource.byteLength, { externalResource: true });
  }

  function fromFloat32(values: unknown) {
    const data = f32(values);
    const buffer = create(data.byteLength);
    writeFloat32(buffer, data);
    return buffer;
  }

  function fromBytes(values: unknown) {
    const data = byteView(values, "NativeBuffer.fromBytes values");
    const buffer = create(data.byteLength);
    writeBytes(buffer, data);
    return buffer;
  }

  function handleValue(buffer: AnyRecord) {
    assertAlive(buffer);
    return buffer.handle;
  }

  function size(buffer: AnyRecord) {
    assertAlive(buffer);
    return Number(bufferSize(buffer.handle));
  }

  function inspect(buffer: AnyRecord) {
    assertAlive(buffer);
    return inspectBuffer(buffer.handle) as BufferInspection;
  }

  function deviceImportOptions(buffer: AnyRecord, programHandle: unknown) {
    assertAlive(buffer);
    if (buffer.devicePlacement === null) {
      return nativeBufferDeviceImportOptions(null, null, null);
    }
    const placement = buffer.devicePlacement;
    return nativeBufferDeviceImportOptions(
      placement,
      inspect(buffer),
      programDeviceHandle(programHandle, placement),
    );
  }

  function writeFloat32(buffer: AnyRecord, values: unknown, byteOffset: unknown = 0) {
    assertAlive(buffer);
    const data = f32(values);
    const write = nativeBufferWriteInfo(buffer.byteLength, data.byteLength, byteOffset, "NativeBuffer.writeFloat32");
    writeBuffer(buffer.handle, write.byteOffset, data, write.byteLength);
    return buffer;
  }

  function writeBytes(buffer: AnyRecord, values: unknown, byteOffset: unknown = 0) {
    assertAlive(buffer);
    const data = byteView(values, "NativeBuffer.writeBytes values");
    const write = nativeBufferWriteInfo(buffer.byteLength, data.byteLength, byteOffset, "NativeBuffer.writeBytes");
    writeBuffer(buffer.handle, write.byteOffset, data, write.byteLength);
    return buffer;
  }

  function readFloat32(buffer: AnyRecord, length: unknown, byteOffset: unknown = 0) {
    assertAlive(buffer);
    const target = nativeBufferReadFloat32Target(buffer.byteLength, length, byteOffset);
    readFloat32Into(buffer, target.target, target.length, target.byteOffset);
    return target.target;
  }

  function readBytes(buffer: AnyRecord, length: unknown, byteOffset: unknown = 0) {
    assertAlive(buffer);
    const target = nativeBufferReadBytesTarget(buffer.byteLength, length, byteOffset);
    readBytesInto(buffer, target.target, target.byteLength, target.byteOffset);
    return target.target;
  }

  function readFloat32Into(buffer: AnyRecord, target: Float32Array, length: unknown = target && target.length, byteOffset: unknown = 0) {
    assertAlive(buffer);
    const read = nativeBufferReadFloat32IntoInfo(buffer.byteLength, target, length, byteOffset);
    readBuffer(buffer.handle, read.byteOffset, target, read.byteLength);
    return target;
  }

  function readBytesInto(buffer: AnyRecord, target: Uint8Array, byteLength: unknown = target && target.byteLength, byteOffset: unknown = 0) {
    assertAlive(buffer);
    const read = nativeBufferReadBytesIntoInfo(buffer.byteLength, target, byteLength, byteOffset);
    readBuffer(buffer.handle, read.byteOffset, target, read.byteLength);
    return target;
  }

  function free(buffer: AnyRecord) {
    if (!isLiveHandle(buffer.handle)) return;
    freeBuffer(buffer.handle);
    buffer.handle = nullHandle;
    buffer.byteLength = 0;
    buffer.backing = null;
    buffer.externalResource = false;
    buffer.devicePlacement = null;
  }

  function dispose(buffer: AnyRecord) {
    return free(buffer);
  }

  return Object.freeze({
    initialize,
    assertAlive,
    fromNativeHandle,
    create,
    wrapBytes: wrapBytesBuffer,
    wrapFloat32,
    externalResource,
    fromFloat32,
    fromBytes,
    handleValue,
    size,
    inspect,
    deviceImportOptions,
    writeFloat32,
    writeBytes,
    readFloat32,
    readBytes,
    readFloat32Into,
    readBytesInto,
    free,
    dispose,
  }) as NativeBufferFacade<Buffer>;
}

export function createNativeBufferLifetimeHelpers<Buffer extends NativeBufferLifetimeTarget = NativeBufferLifetimeTarget>(
  options: NativeBufferLifetimeHelpersOptions<Buffer>,
): NativeBufferLifetimeHelpers<Buffer> {
  const isNativeBuffer = options && options.isNativeBuffer;
  if (typeof isNativeBuffer !== "function") {
    throw new Error("createNativeBufferLifetimeHelpers requires isNativeBuffer");
  }

  function requireNativeBuffer(value: unknown, name: string): Buffer | null {
    if (value == null) return null;
    if (!isNativeBuffer(value)) throw new Error(`${name} must be a zgml NativeBuffer`);
    value.assertAlive();
    return value;
  }

  function uniqueNativeBuffers(values: readonly unknown[]): readonly Buffer[] {
    const out: Buffer[] = [];
    const seen = new Set<Buffer>();
    for (const value of values) {
      if (!isNativeBuffer(value) || seen.has(value)) continue;
      value.assertAlive();
      seen.add(value);
      out.push(value);
    }
    return out;
  }

  return Object.freeze({
    requireNativeBuffer,
    uniqueNativeBuffers,
  }) as NativeBufferLifetimeHelpers<Buffer>;
}
