import type { SharedFrontendRuntime } from "./shared_frontend_runtime.js";

type NativeBufferFacadeHelpersOptions = Parameters<SharedFrontendRuntime["createNativeBufferFacadeHelpers"]>[0];

export type AdapterNativeBufferFacadeSurfaceOptions = Readonly<{
  sharedFrontend: SharedFrontendRuntime;
  NativeBuffer: NativeBufferFacadeHelpersOptions["NativeBuffer"];
  nullHandle?: NativeBufferFacadeHelpersOptions["nullHandle"];
  f32: NativeBufferFacadeHelpersOptions["f32"];
  byteView: NativeBufferFacadeHelpersOptions["byteView"];
  isLiveHandle: NativeBufferFacadeHelpersOptions["isLiveHandle"];
  createBuffer: NativeBufferFacadeHelpersOptions["createBuffer"];
  wrapBytes: NativeBufferFacadeHelpersOptions["wrapBytes"];
  wrapExternalResource: NativeBufferFacadeHelpersOptions["wrapExternalResource"];
  bufferSize: NativeBufferFacadeHelpersOptions["bufferSize"];
  inspectBuffer: NativeBufferFacadeHelpersOptions["inspectBuffer"];
  writeBuffer: NativeBufferFacadeHelpersOptions["writeBuffer"];
  readBuffer: NativeBufferFacadeHelpersOptions["readBuffer"];
  freeBuffer: NativeBufferFacadeHelpersOptions["freeBuffer"];
  programDeviceHandle: NativeBufferFacadeHelpersOptions["programDeviceHandle"];
}>;

export function createAdapterNativeBufferFacadeSurface(options: AdapterNativeBufferFacadeSurfaceOptions) {
  return options.sharedFrontend.createNativeBufferFacadeHelpers({
    NativeBuffer: options.NativeBuffer,
    nullHandle: options.nullHandle,
    f32: options.f32,
    byteView: options.byteView,
    isLiveHandle: options.isLiveHandle,
    createBuffer: options.createBuffer,
    wrapBytes: options.wrapBytes,
    wrapExternalResource: options.wrapExternalResource,
    bufferSize: options.bufferSize,
    inspectBuffer: options.inspectBuffer,
    writeBuffer: options.writeBuffer,
    readBuffer: options.readBuffer,
    freeBuffer: options.freeBuffer,
    programDeviceHandle: options.programDeviceHandle,
  });
}
