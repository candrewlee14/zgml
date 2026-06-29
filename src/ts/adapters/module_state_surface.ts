import type { SharedFrontendRuntime } from "./shared_frontend_runtime.js";

type ModuleStateHelpersOptions = Parameters<SharedFrontendRuntime["createModuleStateHelpers"]>[0];

export type AdapterModuleStateSurfaceOptions = Readonly<{
  sharedFrontend: SharedFrontendRuntime;
  Tensor: ModuleStateHelpersOptions["Tensor"];
  f32WithLength: ModuleStateHelpersOptions["f32WithLength"];
  defaultLayout: ModuleStateHelpersOptions["defaultLayout"];
  optionalParameterFields?: ModuleStateHelpersOptions["optionalParameterFields"];
}>;

export function createAdapterModuleStateSurface(options: AdapterModuleStateSurfaceOptions) {
  return options.sharedFrontend.createModuleStateHelpers({
    Tensor: options.Tensor,
    f32WithLength: options.f32WithLength,
    defaultLayout: options.defaultLayout,
    optionalParameterFields: options.optionalParameterFields,
  });
}
