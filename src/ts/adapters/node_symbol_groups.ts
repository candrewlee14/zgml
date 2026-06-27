"use strict";

import type {
  NodeNativeSymbols,
} from "./node_symbols.js";

export type NodeSymbolGroups = ReturnType<typeof createNodeSymbolGroups>;

export function createNodeSymbolGroups(symbols: NodeNativeSymbols) {
  return Object.freeze({
    inspection: Object.freeze({
      modelInspect: symbols.modelInspect,
      programCompile: symbols.programCompile,
      bufferInspect: symbols.bufferInspect,
      sessionInspect: symbols.sessionInspect,
      sessionPosition: symbols.sessionPosition,
      sessionReset: symbols.sessionReset,
      sessionRuntimeProfile: symbols.sessionRuntimeProfile,
      sessionResetRuntimeProfile: symbols.sessionResetRuntimeProfile,
      programInspect: symbols.programInspect,
      programGetRequirements: symbols.programGetRequirements,
      programCheckModelCompatibility: symbols.programCheckModelCompatibility,
      llamaProgramInspect: symbols.llamaProgramInspect,
      programRuntimeProfile: symbols.programRuntimeProfile,
    }),
    modelSource: Object.freeze({
      modelCreate: symbols.modelCreate,
      modelLoadPath: symbols.modelLoadPath,
      modelLoadSafetensorsData: symbols.modelLoadSafetensorsData,
      modelProbePath: symbols.modelProbePath,
      modelProbeSafetensorsData: symbols.modelProbeSafetensorsData,
      modelProbeSafetensorsHeader: symbols.modelProbeSafetensorsHeader,
      supportedCheckpointCount: symbols.supportedCheckpointCount,
      supportedCheckpointInspect: symbols.supportedCheckpointInspect,
    }),
    llamaSessionBind: Object.freeze({
      llamaProgramGetKvCacheRequirements: symbols.llamaProgramGetKvCacheRequirements,
      llamaSessionBindBuffers: symbols.llamaSessionBindBuffers,
      llamaSessionBindModelBuffers: symbols.llamaSessionBindModelBuffers,
      sessionBindBuffers: symbols.sessionBindBuffers,
      sessionBindModelBuffers: symbols.sessionBindModelBuffers,
      sessionBind: symbols.sessionBind,
      sessionBindModel: symbols.sessionBindModel,
    }),
    moduleProgram: Object.freeze({
      moduleProgramCompile: symbols.moduleProgramCompile,
    }),
    llamaToken: Object.freeze({
      sessionStepToken: symbols.sessionStepToken,
      sessionAdvanceToken: symbols.sessionAdvanceToken,
      sessionExecuteTokens: symbols.sessionExecuteTokens,
      sessionArgmaxToken: symbols.sessionArgmaxToken,
      sessionExecuteArgmaxTokens: symbols.sessionExecuteArgmaxTokens,
      sessionGenerateArgmaxTokens: symbols.sessionGenerateArgmaxTokens,
      sessionSampleToken: symbols.sessionSampleToken,
      sessionExecuteSampleTokens: symbols.sessionExecuteSampleTokens,
      sessionGenerateSampleTokens: symbols.sessionGenerateSampleTokens,
    }),
    session: Object.freeze({
      sessionUploadPersistent: symbols.sessionUploadPersistent,
      sessionUploadPersistentRange: symbols.sessionUploadPersistentRange,
      sessionStep: symbols.sessionStep,
      sessionStepDirect: symbols.sessionStepDirect,
      sessionStepNoOutput: symbols.sessionStepNoOutput,
    }),
    nativeLifecycle: Object.freeze({
      sessionFree: symbols.sessionFree,
      programResetRuntimeProfile: symbols.programResetRuntimeProfile,
      programFree: symbols.programFree,
      modelFree: symbols.modelFree,
    }),
    nativeBuffer: Object.freeze({
      bufferCreate: symbols.bufferCreate,
      bufferWrap: symbols.bufferWrap,
      bufferWrapResource: symbols.bufferWrapResource,
      bufferSize: symbols.bufferSize,
      bufferWrite: symbols.bufferWrite,
      bufferRead: symbols.bufferRead,
      bufferFree: symbols.bufferFree,
    }),
    nativeEager: Object.freeze({
      eagerLinearF32: symbols.eagerLinearF32,
      eagerMatmulF32: symbols.eagerMatmulF32,
      eagerLinearActivationF32: symbols.eagerLinearActivationF32,
      eagerActivationF32: symbols.eagerActivationF32,
      eagerElementwiseF32: symbols.eagerElementwiseF32,
      eagerReduceF32: symbols.eagerReduceF32,
      eagerSoftmaxF32: symbols.eagerSoftmaxF32,
    }),
    nativeTraining: Object.freeze({
      trainLinearMseSgdF32: symbols.trainLinearMseSgdF32,
      trainMlpReluCrossEntropyAdamF32: symbols.trainMlpReluCrossEntropyAdamF32,
      trainMlpReluCrossEntropyAdamWF32: symbols.trainMlpReluCrossEntropyAdamWF32,
    }),
    programBind: Object.freeze({
      sessionBind: symbols.sessionBind,
      sessionBindBuffers: symbols.sessionBindBuffers,
    }),
    programBuffer: Object.freeze({
      programCreateBuffer: symbols.programCreateBuffer,
      programCreateDeviceBuffer: symbols.programCreateDeviceBuffer,
      programGetDeviceHandle: symbols.programGetDeviceHandle,
      programImportDeviceBuffer: symbols.programImportDeviceBuffer,
      bufferSize: symbols.bufferSize,
    }),
    genericFamily: Object.freeze({
      modelCreate: symbols.modelCreate,
    }),
  });
}
