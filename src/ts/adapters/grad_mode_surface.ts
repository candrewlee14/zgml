type GradModeCallback = <T>(fn: () => T) => T;

export type AdapterGradModeSurface = Readonly<{
  isGradEnabled: () => boolean;
  is_grad_enabled: () => boolean;
  setGradEnabled: (enabled: boolean) => boolean;
  set_grad_enabled: (enabled: boolean) => boolean;
  noGrad: GradModeCallback;
  no_grad: GradModeCallback;
  inferenceMode: GradModeCallback;
  inference_mode: GradModeCallback;
  enableGrad: GradModeCallback;
  enable_grad: GradModeCallback;
}>;

export function createAdapterGradModeSurface(surface: AdapterGradModeSurface): AdapterGradModeSurface {
  return Object.freeze({
    isGradEnabled: surface.isGradEnabled,
    is_grad_enabled: surface.is_grad_enabled,
    setGradEnabled: surface.setGradEnabled,
    set_grad_enabled: surface.set_grad_enabled,
    noGrad: surface.noGrad,
    no_grad: surface.no_grad,
    inferenceMode: surface.inferenceMode,
    inference_mode: surface.inference_mode,
    enableGrad: surface.enableGrad,
    enable_grad: surface.enable_grad,
  });
}
