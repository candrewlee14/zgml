import {
  createHostModuleCompilerSurface,
  type HostModuleCompilerSurface,
  type HostTraceModuleCompiler,
} from "../runtime/host_adapter_surfaces.js";

export type AdapterTraceModuleCompiler = HostTraceModuleCompiler;

export type AdapterModuleCompilerSurfaceOptions = Readonly<{
  getTraceModuleCompiler(): AdapterTraceModuleCompiler | null | undefined;
  uninitializedMessage: string;
}>;

export type AdapterModuleCompilerSurface = HostModuleCompilerSurface;

export function createAdapterModuleCompilerSurface(
  options: AdapterModuleCompilerSurfaceOptions,
): AdapterModuleCompilerSurface {
  return createHostModuleCompilerSurface({
    getTraceModuleCompiler: options.getTraceModuleCompiler,
    uninitializedMessage: options.uninitializedMessage,
  });
}
