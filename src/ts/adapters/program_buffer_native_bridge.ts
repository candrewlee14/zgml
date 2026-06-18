type NativeHandle = unknown;

export type AdapterProgramBufferNativeBuffer = Readonly<{
  deviceImportOptions(programHandle: NativeHandle): unknown;
}>;

export type AdapterProgramBufferNativeBufferClass<
  TNativeBuffer extends AdapterProgramBufferNativeBuffer,
  TPlacement = unknown,
> = (abstract new (...args: never[]) => TNativeBuffer) & {
  fromNativeHandle(handle: NativeHandle, byteLength: number, placementName?: TPlacement): TNativeBuffer;
};

export type AdapterProgramBufferNativeBridgeOptions<
  TNativeBuffer extends AdapterProgramBufferNativeBuffer,
  TPlacement = unknown,
> = Readonly<{
  getNativeBufferClass(): AdapterProgramBufferNativeBufferClass<TNativeBuffer, TPlacement>;
}>;

export function createAdapterProgramBufferNativeBridge<
  TNativeBuffer extends AdapterProgramBufferNativeBuffer,
  TPlacement = unknown,
>(
  options: AdapterProgramBufferNativeBridgeOptions<TNativeBuffer, TPlacement>,
) {
  function nativeBufferClass(): AdapterProgramBufferNativeBufferClass<TNativeBuffer, TPlacement> {
    return options.getNativeBufferClass();
  }

  function createNativeBuffer(handle: NativeHandle, byteLength: number, placementName?: TPlacement): TNativeBuffer {
    return nativeBufferClass().fromNativeHandle(handle, byteLength, placementName);
  }

  function nativeBufferDeviceImportOptions(value: unknown, programHandle: NativeHandle): unknown | null {
    const NativeBuffer = nativeBufferClass();
    return value instanceof NativeBuffer ? value.deviceImportOptions(programHandle) : null;
  }

  return Object.freeze({
    createNativeBuffer,
    nativeBufferDeviceImportOptions,
  });
}
