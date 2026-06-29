import type {
  ByteLike,
  BufferInspection,
  ExternalResourceOptions,
  IndexLike,
  NativeBufferDeviceImportInfo,
  NativeBufferFacade,
  NativeBufferInitOptions,
  NativeBufferSurfaceClass,
} from "../runtime/native_buffer.js";

const disposeSymbol: symbol = (Symbol as any).dispose;

let nativeBufferFacade: NativeBufferFacade<NativeBuffer> | null = null;

function facade(): NativeBufferFacade<NativeBuffer> {
  if (nativeBufferFacade === null) {
    throw new Error("Adapter NativeBuffer facade has not been initialized");
  }
  return nativeBufferFacade;
}

export function setAdapterNativeBufferFacade(facadeValue: NativeBufferFacade<NativeBuffer>): void {
  nativeBufferFacade = facadeValue;
}

export class NativeBuffer {
  handle: unknown = 0;
  byteLength = 0;
  backing: ArrayBufferView | null = null;
  externalResource = false;
  devicePlacement: unknown = null;

  constructor(handle: unknown, byteLength: number, options: NativeBufferInitOptions = {}) {
    facade().initialize(this, handle, byteLength, options);
  }

  static create(byteLength: number): NativeBuffer {
    return facade().create(byteLength);
  }

  static wrapBytes(data: ArrayBufferView): NativeBuffer {
    return facade().wrapBytes(data);
  }

  static wrapFloat32(data: Float32Array): NativeBuffer {
    return facade().wrapFloat32(data);
  }

  static externalResource(options: ExternalResourceOptions): NativeBuffer {
    return facade().externalResource(options);
  }

  static fromFloat32(values: IndexLike): NativeBuffer {
    return facade().fromFloat32(values);
  }

  static fromBytes(values: ByteLike): NativeBuffer {
    return facade().fromBytes(values);
  }

  static fromNativeHandle(handle: unknown, byteLength: number, devicePlacement: unknown | null = null): NativeBuffer {
    return facade().fromNativeHandle(handle, byteLength, devicePlacement);
  }

  get handleValue(): number {
    return facade().handleValue(this) as number;
  }

  assertAlive(): void {
    return facade().assertAlive(this);
  }

  size(): number {
    return facade().size(this);
  }

  inspect(): BufferInspection {
    return facade().inspect(this);
  }

  deviceImportOptions(programHandle: unknown): NativeBufferDeviceImportInfo {
    return facade().deviceImportOptions(this, programHandle);
  }

  writeFloat32(values: unknown, byteOffset = 0): this {
    return facade().writeFloat32(this, values, byteOffset) as this;
  }

  writeBytes(values: unknown, byteOffset = 0): this {
    return facade().writeBytes(this, values, byteOffset) as this;
  }

  readFloat32(length?: number, byteOffset = 0): Float32Array {
    return facade().readFloat32(this, length, byteOffset);
  }

  readBytes(length?: number, byteOffset = 0): Uint8Array {
    return facade().readBytes(this, length, byteOffset);
  }

  readFloat32Into(target: Float32Array, length = target?.length, byteOffset = 0): Float32Array {
    return facade().readFloat32Into(this, target, length, byteOffset);
  }

  readBytesInto(target: Uint8Array, byteLength = target?.byteLength, byteOffset = 0): Uint8Array {
    return facade().readBytesInto(this, target, byteLength, byteOffset);
  }

  free(): void {
    return facade().free(this);
  }

  dispose(): void {
    return facade().dispose(this);
  }

  [disposeSymbol](): void {
    return facade().dispose(this);
  }
}

export const NativeBufferSurface = NativeBuffer as unknown as NativeBufferSurfaceClass<NativeBuffer>;
