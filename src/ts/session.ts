"use strict";

import { tsProductManifestPolicy } from "./internal/product_manifest.js";

export type {
  LlamaSessionStepContract,
  LlamaStepParams,
  ProgramInputBinding,
  ProgramOutputBinding,
  PublicSessionNamespace,
  Session,
  SessionBufferSizing,
  SessionCallProfile,
  SessionDefaultOutputKind,
  SessionExecuteIntoParams,
  SessionExecuteParams,
  SessionExecuteTensorParams,
  SessionExecutionPlan,
  SessionInspection,
  SessionNoOutputEffect,
  SessionNamespace,
  SessionReadOutputTensorOptions,
  SessionStepContract,
  SessionStepParams,
  SessionStepTensorOptions,
  SessionStepParamsCompatibility,
  SessionStepParamsDiagnostic,
  SessionStepParamsDiagnosticCode,
  SessionStepParamsElementType,
  SessionStepParamsHotPathBlocker,
  SessionStepParamsHotPathStatus,
  SessionStepParamsInputOwnership,
  SessionStepParamsInputSource,
  SessionStepParamsOutputEffect,
  SessionStepParamsOutputOwnership,
  SessionStepParamsOutputReturnOwnership,
  SessionStepParamsOutputTarget,
  SessionStepParamsStateEffect,
} from "./public_api.js";

export {
  acceptsSessionExecutionPlan,
  assert_session_execution_plan,
  assertSessionExecutionPlan,
  matchesSessionExecutionPlanSignature,
  requireSessionExecutionPlan,
} from "./runtime/execution_plan.js";
export {
  createSessionLiveFacadeHelpers,
  createSessionLayoutFacadeHelpers,
  createSessionBufferSizingFacadeHelpers,
  createSessionCallProfileFacadeHelpers,
  createSessionRuntimeProfileFacadeHelpers,
  createSessionStepParamsFacadeHelpers,
  createSessionParameterFacadeHelpers,
  createSessionLifecycleFacadeHelpers,
  createSessionReadbackFacadeHelpers,
  createGenericSessionExecutionFacadeHelpers,
  createGenericSessionCoreStepFacadeHelpers,
  createGenericSessionStepFacadeHelpers,
  createLlamaSessionStepFacadeHelpers,
  createLlamaSessionExecutionFacadeHelpers,
  createLlamaSessionPrefillFacadeHelpers,
  createLlamaSessionTokenSelectionFacadeHelpers,
} from "./runtime/session_facade.js";
export {
  createGenericSessionStepContract,
  createLlamaSessionStepContract,
} from "./runtime/session_contract.js";
export {
  createSessionTensorHelpers,
} from "./runtime/session_tensor.js";
export {
  bindLlamaSessionHandleByPolicy,
  createLlamaSessionFacadeHelpers,
  createGenericSessionFacadeHelpers,
  sessionFacadeCompositionManifest,
} from "./runtime/session_facade_composition.js";
export {
  acceptsSessionCallProfile,
  assert_session_call_profile,
  assertSessionCallProfile,
  matchesSessionCallProfileSignature,
  requireSessionCallProfile,
  sessionCallProfileFields,
} from "./runtime/session_profile.js";

export const sessionManifest = Object.freeze({
  kind: "zgml-session",
  ...tsProductManifestPolicy("src/ts/session.ts"),
  runtimePath: "Program -> Session -> StepParams",
  genericSessionComposition: "ts",
  llamaSessionComposition: "ts",
});
