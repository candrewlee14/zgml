import type { SharedFrontendRuntime } from "./shared_frontend_runtime.js";

type LlamaSessionFacadeOptions = Parameters<SharedFrontendRuntime["createLlamaSessionFacadeHelpers"]>[0];

export type AdapterLlamaSessionFacadeSurfaceOptions = Readonly<{
  sharedFrontend: SharedFrontendRuntime;
  isNativeBuffer: LlamaSessionFacadeOptions["isNativeBuffer"];
  f32: LlamaSessionFacadeOptions["f32"];
  createProgramOutputBuffer: LlamaSessionFacadeOptions["createProgramOutputBuffer"];
  sessionTensorHelpers: LlamaSessionFacadeOptions["sessionTensorHelpers"];
  executeTokenWindow: LlamaSessionFacadeOptions["executeTokenWindow"];
  stepToken: LlamaSessionFacadeOptions["stepToken"];
  advanceToken: LlamaSessionFacadeOptions["advanceToken"];
  assertSessionAlive: LlamaSessionFacadeOptions["assertSessionAlive"];
  sessionPosition: LlamaSessionFacadeOptions["sessionPosition"];
  sessionInspect: LlamaSessionFacadeOptions["sessionInspect"];
  sessionReset: LlamaSessionFacadeOptions["sessionReset"];
  sessionRuntimeProfile: LlamaSessionFacadeOptions["sessionRuntimeProfile"];
  sessionResetRuntimeProfile: LlamaSessionFacadeOptions["sessionResetRuntimeProfile"];
  sessionFree: LlamaSessionFacadeOptions["sessionFree"];
  nullSessionHandle: LlamaSessionFacadeOptions["nullSessionHandle"];
  argmaxLogits: LlamaSessionFacadeOptions["argmaxLogits"];
  sampleLogits: LlamaSessionFacadeOptions["sampleLogits"];
  executeArgmaxWindow: LlamaSessionFacadeOptions["executeArgmaxWindow"];
  executeSampleWindow: LlamaSessionFacadeOptions["executeSampleWindow"];
  generateArgmaxWindow: LlamaSessionFacadeOptions["generateArgmaxWindow"];
  generateSampleWindow: LlamaSessionFacadeOptions["generateSampleWindow"];
}>;

export function createAdapterLlamaSessionFacadeSurface(options: AdapterLlamaSessionFacadeSurfaceOptions) {
  return options.sharedFrontend.createLlamaSessionFacadeHelpers({
    isNativeBuffer: options.isNativeBuffer,
    f32: options.f32,
    createProgramOutputBuffer: options.createProgramOutputBuffer,
    sessionTensorHelpers: options.sessionTensorHelpers,
    executeTokenWindow: options.executeTokenWindow,
    stepToken: options.stepToken,
    advanceToken: options.advanceToken,
    assertSessionAlive: options.assertSessionAlive,
    sessionPosition: options.sessionPosition,
    sessionInspect: options.sessionInspect,
    sessionReset: options.sessionReset,
    sessionRuntimeProfile: options.sessionRuntimeProfile,
    sessionResetRuntimeProfile: options.sessionResetRuntimeProfile,
    sessionFree: options.sessionFree,
    nullSessionHandle: options.nullSessionHandle,
    argmaxLogits: options.argmaxLogits,
    sampleLogits: options.sampleLogits,
    executeArgmaxWindow: options.executeArgmaxWindow,
    executeSampleWindow: options.executeSampleWindow,
    generateArgmaxWindow: options.generateArgmaxWindow,
    generateSampleWindow: options.generateSampleWindow,
  });
}
