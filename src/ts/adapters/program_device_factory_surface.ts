type NativeHandle = unknown;

export type AdapterProgramDeviceConstructor<THandle, TPlacement, TProgramDevice> = new (
  handle: THandle,
  placement?: TPlacement,
) => TProgramDevice;

export type AdapterProgramDeviceFactorySurfaceOptions<
  THandle = NativeHandle,
  TPlacement = unknown,
  TProgramDevice = unknown,
> = Readonly<{
  getProgramDeviceClass(): AdapterProgramDeviceConstructor<THandle, TPlacement, TProgramDevice>;
}>;

export function createAdapterProgramDeviceFactory<
  THandle = NativeHandle,
  TPlacement = unknown,
  TProgramDevice = unknown,
>(
  options: AdapterProgramDeviceFactorySurfaceOptions<THandle, TPlacement, TProgramDevice>,
) {
  return function createProgramDevice(handle: THandle, placement?: TPlacement): TProgramDevice {
    const ProgramDevice = options.getProgramDeviceClass();
    return new ProgramDevice(handle, placement);
  };
}
