import {
  createHostModelSourceSurface,
  type HostModelSourceFacade,
  type HostModelSourceSurface,
} from "../runtime/host_adapter_surfaces.js";

export type AdapterModelSourceFacade = HostModelSourceFacade;

export type AdapterModelSourceSurfaceOptions = Readonly<{
  getModelSourceFacade(): AdapterModelSourceFacade | null | undefined;
  uninitializedMessage: string;
}>;

export type AdapterModelSourceSurface = HostModelSourceSurface;

export function createAdapterModelSourceSurface(
  options: AdapterModelSourceSurfaceOptions,
): AdapterModelSourceSurface {
  return createHostModelSourceSurface({
    getModelSourceFacade: options.getModelSourceFacade,
    uninitializedMessage: options.uninitializedMessage,
  });
}
