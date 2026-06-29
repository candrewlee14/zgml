import type { SharedFrontendRuntime } from "./shared_frontend_runtime.js";
import type { NativeBufferLifetimeTarget } from "../runtime/native_buffer.js";
import type { ProgramBindingSurfaceOptions } from "../runtime/program_binding_surface.js";

type GenericSessionFacadeOptions = Parameters<SharedFrontendRuntime["createGenericSessionFacadeHelpers"]>[0];

export type AdapterSessionRuntimeSurfaceOptions<TNativeBuffer extends NativeBufferLifetimeTarget = NativeBufferLifetimeTarget> = Readonly<{
  sharedFrontend: SharedFrontendRuntime;
  isNativeBuffer: ProgramBindingSurfaceOptions<TNativeBuffer>["isNativeBuffer"];
  f32: ProgramBindingSurfaceOptions<TNativeBuffer>["f32"];
  prepareF32: GenericSessionFacadeOptions["prepareF32"] & ProgramBindingSurfaceOptions<TNativeBuffer>["prepareF32"];
  valueShape: NonNullable<GenericSessionFacadeOptions["valueShape"]> & ProgramBindingSurfaceOptions<TNativeBuffer>["valueShape"];
  sessionTensorHelpers: GenericSessionFacadeOptions["sessionTensorHelpers"];
  assertSessionAlive: GenericSessionFacadeOptions["assertSessionAlive"];
  sessionInspect: GenericSessionFacadeOptions["sessionInspect"];
  sessionUploadPersistent: GenericSessionFacadeOptions["sessionUploadPersistent"];
  sessionUploadPersistentRange: GenericSessionFacadeOptions["sessionUploadPersistentRange"];
  sessionReset: GenericSessionFacadeOptions["sessionReset"];
  sessionRuntimeProfile: GenericSessionFacadeOptions["sessionRuntimeProfile"];
  sessionResetRuntimeProfile: GenericSessionFacadeOptions["sessionResetRuntimeProfile"];
  sessionFree: GenericSessionFacadeOptions["sessionFree"];
  nullSessionHandle: GenericSessionFacadeOptions["nullSessionHandle"];
  stepSession: GenericSessionFacadeOptions["stepSession"];
  prepareStepSession?: GenericSessionFacadeOptions["prepareStepSession"];
  stepNoOutput: GenericSessionFacadeOptions["stepNoOutput"];
}>;

export function createAdapterSessionRuntimeSurface<TNativeBuffer extends NativeBufferLifetimeTarget = NativeBufferLifetimeTarget>(
  options: AdapterSessionRuntimeSurfaceOptions<TNativeBuffer>,
) {
  const genericSessionFacade = options.sharedFrontend.createGenericSessionFacadeHelpers({
    isNativeBuffer: options.isNativeBuffer,
    f32: options.f32,
    prepareF32: options.prepareF32,
    valueShape: options.valueShape,
    sessionTensorHelpers: options.sessionTensorHelpers,
    assertSessionAlive: options.assertSessionAlive,
    sessionInspect: options.sessionInspect,
    sessionUploadPersistent: options.sessionUploadPersistent,
    sessionUploadPersistentRange: options.sessionUploadPersistentRange,
    sessionReset: options.sessionReset,
    sessionRuntimeProfile: options.sessionRuntimeProfile,
    sessionResetRuntimeProfile: options.sessionResetRuntimeProfile,
    sessionFree: options.sessionFree,
    nullSessionHandle: options.nullSessionHandle,
    stepSession: options.stepSession,
    prepareStepSession: options.prepareStepSession,
    stepNoOutput: options.stepNoOutput,
  });

  const programBindingSurface = options.sharedFrontend.createProgramBindingSurface<TNativeBuffer>({
    isNativeBuffer: options.isNativeBuffer,
    f32: options.f32,
    prepareF32: options.prepareF32,
    valueShape: options.valueShape,
    createNativeBufferLifetimeHelpers: options.sharedFrontend.createNativeBufferLifetimeHelpers,
    createProgramModuleBindingHelpers: options.sharedFrontend.createProgramModuleBindingHelpers,
    createProgramBindValidationHelpers: options.sharedFrontend.createProgramBindValidationHelpers,
    createProgramBindPreparationHelpers: options.sharedFrontend.createProgramBindPreparationHelpers,
    createProgramNativeBufferBindFieldHelpers: options.sharedFrontend.createProgramNativeBufferBindFieldHelpers,
  });

  return Object.freeze({
    genericSessionFacade,
    programBindingSurface,
    requireNativeBuffer: programBindingSurface.requireNativeBuffer,
    uniqueNativeBuffers: programBindingSurface.uniqueNativeBuffers,
    bindModuleThroughProgram: programBindingSurface.bindModuleThroughProgram,
    validateProgramBindParams: programBindingSurface.validateProgramBindParams,
    programBindingPlan: programBindingSurface.programBindingPlan,
    prepareProgramBind: programBindingSurface.prepareProgramBind,
    freeOwnedProgramBindBuffers: programBindingSurface.freeOwnedProgramBindBuffers,
    programNativeBufferBindFields: programBindingSurface.programNativeBufferBindFields,
  });
}
