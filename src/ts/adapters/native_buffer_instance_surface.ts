import {
  createHostNativeBufferInstanceSurface,
  type HostNativeBufferConstructor,
  type HostNativeBufferInstanceSurface,
} from "../runtime/host_adapter_surfaces.js";

export type AdapterNativeBufferConstructor<TNativeBuffer extends object = object> =
  HostNativeBufferConstructor<TNativeBuffer>;

export type AdapterNativeBufferInstanceSurfaceOptions<TNativeBuffer extends object = object> = Readonly<{
  getNativeBufferClass(): AdapterNativeBufferConstructor<TNativeBuffer> | null | undefined;
  uninitializedMessage: string;
}>;

export type AdapterNativeBufferInstanceSurface<TNativeBuffer extends object = object> =
  HostNativeBufferInstanceSurface<TNativeBuffer>;

export function createAdapterNativeBufferInstanceSurface<TNativeBuffer extends object = object>(
  options: AdapterNativeBufferInstanceSurfaceOptions<TNativeBuffer>,
): AdapterNativeBufferInstanceSurface<TNativeBuffer> {
  return createHostNativeBufferInstanceSurface({
    getNativeBufferClass: options.getNativeBufferClass,
    uninitializedMessage: options.uninitializedMessage,
  });
}
