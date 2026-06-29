"use strict";

import {
  acceptsStepParamsCompatibility,
  canExecuteStepParamsCompatibility,
  requireCanExecuteStepParamsCompatibility,
  acceptsAllocationFreeStepParamsCompatibility,
  requireAllocationFreeStepParamsCompatibility,
  acceptsRuntimeOutputAllocationFreeStepParamsCompatibility,
  requireRuntimeOutputAllocationFreeStepParamsCompatibility,
  acceptsNoReadbackStepParamsCompatibility,
  requireNoReadbackStepParamsCompatibility,
  acceptsReadbackFreeStepParamsCompatibility,
  requireReadbackFreeStepParamsCompatibility,
  acceptsHotStepParamsCompatibility,
  requireHotStepParamsCompatibility,
  matchesStepContractSignature,
  matchesStepParamsSignature,
  matchesStepParamsCompatibility,
} from "./step_params.js";

import {
  bumpSessionCallProfile as bumpSessionCallProfileValue,
  sessionCallProfile as sessionCallProfileValue,
  matchesSessionCallProfileSignature as matchesSessionCallProfileSignatureValue,
  resetSessionCallProfile as resetSessionCallProfileValue,
} from "./session_profile.js";

import {
  sessionBufferSizing,
} from "./session_contract.js";

import {
  sessionParameterNames,
  sessionParameterInfos,
  sessionParameterInfo,
  sessionParameterPersistentBindingIndex,
  sessionParameterInfoByNameForUpload,
  type SessionParameterLayoutInput,
} from "./session_parameters.js";

import {
  sessionBufferSlotNames,
  sessionBufferSlot,
  requireSessionKvCacheLayout,
  hostBoundInput as hostBoundInputValue,
  nativeBoundOutput as nativeBoundOutputValue,
  hostBoundOutput as hostBoundOutputValue,
} from "./session_layout.js";

import {
  tokenId,
  tokenWindow,
} from "../core/token.js";

import {
  assertFloat32OutputBuffer,
  noOutputTokenWindowOptions,
  outputTokenWindowOptions,
  scalarTokenExecuteOptions,
  scalarTokenOutputOptions,
  scalarTokenSampleOptions,
  stepParamsValidationError,
  validateLogitsOutputBuffer,
} from "./step_params.js";

import {
  genericSessionExecuteIntoPlan,
  genericSessionExecutePlan,
  genericSessionExecuteTensorPlan,
  llamaSessionArgmaxExecutionPlan,
  llamaSessionGeneratedOutput,
  llamaSessionNativeArgmaxGenerationPlan,
  llamaSessionNativeSampleGenerationPlan,
  llamaSessionPrefillTensorPlan,
  llamaSessionExecuteIntoPlan,
  llamaSessionExecutePlan,
  llamaSessionExecuteTensorPlan,
  llamaSessionExecuteTokenOutputTarget,
  llamaSessionExecuteTokenResult,
  llamaSessionSampleExecutionPlan,
  llamaSessionSampleOptions,
  llamaSessionStepTensorOutputTarget,
  requireLlamaSessionLogitsOutput,
  genericSessionExplicitInput,
  genericSessionExplicitOutput,
  genericSessionStepOutputTarget,
  genericSessionStepResult,
  genericSessionStepParamsCompatibility,
  assertGenericSessionNoOutputStepResult,
} from "./session_values.js";

import type {
  LlamaExecuteTensorParams,
  LlamaLogitsTensorOptions,
  SessionExecuteIntoParams,
  SessionExecuteParams,
  SessionExecuteTensorParams,
  SessionReadOutputTensorOptions,
  SessionStepTensorOptions,
  TokenSampleOptions,
  TokenWindowOptions,
  ProgramBufferLayout,
} from "../public_api.js";
import type {
  TensorOutputOptions,
} from "./session_tensor.js";

type AnyRecord = Record<string, any>;
const noFastExecuteIntoInput = Symbol("zgml.no-fast-execute-into-input");

type StepParamsCompatibilityFn<TSession = unknown> = (session: TSession, params?: unknown) => AnyRecord;
type StepContractFn<TSession = unknown> = (session: TSession) => AnyRecord;
type AssertLiveSessionFn<TSession = unknown> = (session: TSession) => void;
type SessionNumberFn<TSession = unknown> = (session: TSession) => number;
type SessionStringFn<TSession = unknown> = (session: TSession) => string;
type SessionRecordFn<TSession = unknown> = (session: TSession) => AnyRecord;
type SessionVoidFn<TSession = unknown> = (session: TSession) => unknown;
type SessionParameterLayoutFn<TSession = unknown> = (session: TSession) => SessionParameterLayoutInput;
type SessionUploadPersistentFn<TSession = unknown> = (session: TSession) => unknown;
type SessionUploadPersistentRangeFn<TSession = unknown> = (session: TSession, first: unknown, len: unknown) => unknown;
type BumpSessionCallProfileFn<TSession = unknown> = (session: TSession, field: string) => void;
type SessionHandleFn<TSession = unknown, THandle = unknown> = (session: TSession) => THandle;
type HandleAssertFn<THandle = unknown> = (handle: THandle) => void;
type HandleRecordFn<THandle = unknown> = (handle: THandle) => AnyRecord;
type HandleValueFn<THandle = unknown> = (handle: THandle) => unknown;
type SessionActionFn<TSession = unknown> = (session: TSession) => unknown;
type SessionValueFn<TSession = unknown> = (session: TSession) => unknown;
type SessionBufferLayoutFn<TSession = unknown> = (session: TSession) => ProgramBufferLayout | null | undefined;
type SessionOptionalValueFn<TSession = unknown> = (session: TSession) => unknown;
type SessionReadOutputIntoCoreFn<TSession = unknown> = (
  session: TSession,
  outputValues: unknown,
  length?: unknown,
  byteOffset?: unknown,
) => unknown;
type SessionReadOutputTensorTargetFn<TSession = unknown> = (session: TSession, options: SessionReadOutputTensorOptions) => AnyRecord;
type SessionActiveOutputValuesFn = (values: Float32Array, length: number) => Float32Array;
type SessionOutputTensorFn<TSession = unknown> = (session: TSession, values: Float32Array, options: TensorOutputOptions, target: AnyRecord) => unknown;
type GenericSessionStepCoreFn<TSession = unknown> = (session: TSession, inputValues: unknown, outputValues?: unknown) => unknown;
type GenericSessionStepIntoCoreFn<TSession = unknown> = (session: TSession, outputValues: unknown, inputValues?: unknown) => unknown;
type GenericSessionPrepareExecuteIntoCoreFn<TSession = unknown> = (session: TSession, outputValues: unknown, inputValues?: unknown) => () => Float32Array;
type GenericSessionAdvanceCoreFn<TSession = unknown> = (session: TSession, inputValues: unknown) => unknown;
type GenericPrepareHostValueFn = (value: unknown) => { data: Float32Array; shape: number[] };
type GenericValidateHostValueShapeFn = (
  prepared: { data: Float32Array; shape: number[] },
  expectedShape: readonly number[],
  label: string,
  options?: AnyRecord,
) => void;

function inputOnlyExecuteIntoParam(params: unknown): unknown | typeof noFastExecuteIntoInput {
  if (params === undefined || params === null) return noFastExecuteIntoInput;
  if (typeof params !== "object" || Array.isArray(params) || ArrayBuffer.isView(params)) return noFastExecuteIntoInput;
  const record = params as AnyRecord;
  if (!Object.prototype.hasOwnProperty.call(record, "input")) return noFastExecuteIntoInput;
  if (Object.prototype.hasOwnProperty.call(record, "output")) return noFastExecuteIntoInput;
  for (const key in record) {
    if (Object.prototype.hasOwnProperty.call(record, key) && key !== "input") return noFastExecuteIntoInput;
  }
  return record.input;
}
type GenericStepSessionFn = (handle: unknown, input: unknown, output: unknown, outputLen: number) => number;
type GenericPrepareStepSessionFn = (handle: unknown, input: unknown, output: unknown, outputLen: number) => () => number;
type GenericStepNoOutputFn = (handle: unknown, input: unknown) => number;
type GenericSessionOutputTensorForSessionFn = (
  desc: unknown,
  values: unknown,
  options: TensorOutputOptions,
  boundOutputShape?: unknown,
) => unknown;
type LlamaStepTokenFn = (
  handle: unknown,
  token: unknown,
  outputValues: unknown,
  boundOutput: unknown,
  vocabSize: number,
) => unknown;
type LlamaAdvanceTokenFn = (handle: unknown, token: unknown) => unknown;
type LlamaTensorOutputTargetFn = (options: TensorOutputOptions, label: string) => unknown;
type LlamaOutputTensorFn = (values: unknown, options: TensorOutputOptions) => unknown;
type LlamaExecuteTokensFn<TSession = unknown> = (session: TSession, tokens: unknown, options?: unknown) => unknown;
type LlamaOutputTokenWindowOptionsFn<TSession = unknown> = (
  session: TSession,
  options: TokenWindowOptions,
  outputValues: unknown,
) => unknown;
type ValueToF32Fn = (value: unknown) => Float32Array;
type IsNativeBufferFn = (value: unknown) => boolean;
type LlamaArgmaxLogitsFn = (handle: unknown, data: Float32Array | null) => unknown;
type LlamaSampleLogitsFn = (handle: unknown, data: Float32Array | null, options: TokenSampleOptions) => unknown;
type LlamaExecuteArgmaxWindowFn = (handle: unknown, window: unknown) => unknown;
type LlamaExecuteSampleWindowFn = (handle: unknown, window: unknown, sampleOptions: TokenSampleOptions) => unknown;
type LlamaGenerateArgmaxWindowFn = (handle: unknown, window: unknown, outputTokens: Uint32Array) => unknown;
type LlamaGenerateSampleWindowFn = (
  handle: unknown,
  window: unknown,
  outputTokens: Uint32Array,
  sampleOptions: TokenSampleOptions,
) => unknown;
type LlamaExecuteTokenWindowFn = (
  handle: unknown,
  window: unknown,
  wantsLogits: boolean,
  descOutput: unknown,
) => number;
type LlamaScalarTokenWindowFn<TSession = unknown> = (session: TSession, token: unknown) => unknown;
type LlamaScalarTokenExecuteOptionsFn<TSession = unknown> = (session: TSession, options: TokenWindowOptions & { output?: unknown }) => unknown;
type LlamaScalarTokenOutputOptionsFn<TSession = unknown> = (session: TSession, outputValues: unknown) => unknown;
type LlamaNoOutputTokenWindowOptionsFn<TSession = unknown> = (session: TSession, options: TokenWindowOptions) => unknown;
type ScratchFn<TSession = unknown> = (session: TSession) => AnyRecord;
type TokenScratchFn<TSession = unknown> = (session: TSession) => Uint32Array;

export type SessionLiveFacadeHelpersOptions<
  TSession = unknown,
  THandle = unknown,
> = Readonly<{
  readonly handle: SessionHandleFn<TSession, THandle>;
  readonly assertSessionAlive: HandleAssertFn<THandle>;
  readonly inspect: HandleRecordFn<THandle>;
  readonly position?: HandleValueFn<THandle>;
}>;

export type SessionLayoutFacadeHelpersOptions<TSession = unknown> = Readonly<{
  readonly assertLiveSession: AssertLiveSessionFn<TSession>;
  readonly bufferLayout: SessionBufferLayoutFn<TSession>;
  readonly bufferSlotMethodName: string;
  readonly kvCacheLayout?: SessionOptionalValueFn<TSession>;
  readonly inputShape?: SessionValueFn<TSession>;
  readonly outputShape: SessionValueFn<TSession>;
  readonly inputLen: SessionNumberFn<TSession>;
  readonly outputLen: SessionNumberFn<TSession>;
  readonly inputByteLength: SessionNumberFn<TSession>;
  readonly outputByteLength: SessionNumberFn<TSession>;
  readonly weightsLen: SessionNumberFn<TSession>;
  readonly weightsByteLength: SessionNumberFn<TSession>;
  readonly biasLen: SessionNumberFn<TSession>;
  readonly biasByteLength: SessionNumberFn<TSession>;
  readonly parameterLen: SessionNumberFn<TSession>;
  readonly parameterByteLength: SessionNumberFn<TSession>;
}>;

export type SessionStepParamsFacadeHelpersOptions<TSession = unknown> = Readonly<{
  readonly stepParamsCompatibility: StepParamsCompatibilityFn<TSession>;
  readonly stepContract: StepContractFn<TSession>;
}>;

export type SessionCallProfileFacadeHelpersOptions<TSession = unknown> = Readonly<{
  readonly assertLiveSession: AssertLiveSessionFn<TSession>;
}>;

export type SessionBufferSizingFacadeHelpersOptions<TSession = unknown> = Readonly<{
  readonly assertLiveSession: AssertLiveSessionFn<TSession>;
  readonly modelKind: SessionStringFn<TSession>;
  readonly inputLen: SessionNumberFn<TSession>;
  readonly inputByteLength: SessionNumberFn<TSession>;
  readonly outputLen: SessionNumberFn<TSession>;
  readonly outputByteLength: SessionNumberFn<TSession>;
  readonly weightsLen: SessionNumberFn<TSession>;
  readonly weightsByteLength: SessionNumberFn<TSession>;
  readonly biasLen: SessionNumberFn<TSession>;
  readonly biasByteLength: SessionNumberFn<TSession>;
  readonly parameterLen: SessionNumberFn<TSession>;
  readonly parameterByteLength: SessionNumberFn<TSession>;
}>;

export type SessionRuntimeProfileFacadeHelpersOptions<TSession = unknown> = Readonly<{
  readonly assertLiveSession: AssertLiveSessionFn<TSession>;
  readonly runtimeProfile: SessionRecordFn<TSession>;
  readonly resetRuntimeProfile: SessionVoidFn<TSession>;
}>;

export type SessionParameterFacadeHelpersOptions<TSession = unknown> = Readonly<{
  readonly assertLiveSession: AssertLiveSessionFn<TSession>;
  readonly bumpSessionCallProfile: BumpSessionCallProfileFn<TSession>;
  readonly parameterLayout: SessionParameterLayoutFn<TSession>;
  readonly uploadPersistent: SessionUploadPersistentFn<TSession>;
  readonly uploadPersistentRange: SessionUploadPersistentRangeFn<TSession>;
}>;

export type SessionLifecycleFacadeHelpersOptions<TSession = unknown> = Readonly<{
  readonly assertLiveSession: AssertLiveSessionFn<TSession>;
  readonly bumpSessionCallProfile: BumpSessionCallProfileFn<TSession>;
  readonly reset: SessionActionFn<TSession>;
  readonly free: SessionActionFn<TSession>;
}>;

export type SessionReadbackFacadeHelpersOptions<TSession = unknown> = Readonly<{
  readonly bumpSessionCallProfile: BumpSessionCallProfileFn<TSession>;
  readonly readOutputIntoCore: SessionReadOutputIntoCoreFn<TSession>;
  readonly readOutputTensorTarget: SessionReadOutputTensorTargetFn<TSession>;
  readonly activeOutputValues: SessionActiveOutputValuesFn;
  readonly outputTensor: SessionOutputTensorFn<TSession>;
}>;

export type GenericSessionExecutionFacadeHelpersOptions<TSession extends AnyRecord = AnyRecord> = Readonly<{
  readonly bumpSessionCallProfile: BumpSessionCallProfileFn<TSession>;
  readonly stepCore: GenericSessionStepCoreFn<TSession>;
  readonly stepIntoCore: GenericSessionStepIntoCoreFn<TSession>;
  readonly prepareExecuteIntoCore: GenericSessionPrepareExecuteIntoCoreFn<TSession>;
  readonly advanceCore: GenericSessionAdvanceCoreFn<TSession>;
  readonly outputTensorForSession: GenericSessionOutputTensorForSessionFn;
}>;

export type GenericSessionCoreStepFacadeHelpersOptions<TSession extends AnyRecord = AnyRecord> = Readonly<{
  readonly assertLiveSession: AssertLiveSessionFn<TSession>;
  readonly isNativeBuffer: IsNativeBufferFn;
  readonly prepareHostValue: GenericPrepareHostValueFn;
  readonly validateHostValueShape: GenericValidateHostValueShapeFn;
  readonly stepSession: GenericStepSessionFn;
  readonly prepareStepSession?: GenericPrepareStepSessionFn;
  readonly stepNoOutput: GenericStepNoOutputFn;
  readonly stepContract: StepContractFn<TSession>;
}>;

export type GenericSessionStepFacadeHelpersOptions<TSession extends AnyRecord = AnyRecord> = Readonly<{
  readonly bumpSessionCallProfile: BumpSessionCallProfileFn<TSession>;
  readonly stepCore: GenericSessionStepCoreFn<TSession>;
  readonly stepIntoCore: GenericSessionStepIntoCoreFn<TSession>;
  readonly outputTensorForSession: GenericSessionOutputTensorForSessionFn;
}>;

export type LlamaSessionStepFacadeHelpersOptions<TSession extends AnyRecord = AnyRecord> = Readonly<{
  readonly assertLiveSession: AssertLiveSessionFn<TSession>;
  readonly bumpSessionCallProfile: BumpSessionCallProfileFn<TSession>;
  readonly stepToken: LlamaStepTokenFn;
  readonly advanceToken: LlamaAdvanceTokenFn;
  readonly outputTarget: LlamaTensorOutputTargetFn;
  readonly outputTensor: LlamaOutputTensorFn;
}>;

export type LlamaSessionExecutionFacadeHelpersOptions<TSession extends AnyRecord = AnyRecord> = Readonly<{
  readonly bumpSessionCallProfile: BumpSessionCallProfileFn<TSession>;
  readonly isNativeBuffer: IsNativeBufferFn;
  readonly executeTokenWindow: LlamaExecuteTokenWindowFn;
  readonly scalarTokenWindow: LlamaScalarTokenWindowFn<TSession>;
  readonly scalarTokenExecuteOptions: LlamaScalarTokenExecuteOptionsFn<TSession>;
  readonly outputTokenWindowOptions: LlamaOutputTokenWindowOptionsFn<TSession>;
  readonly scalarTokenOutputOptions: LlamaScalarTokenOutputOptionsFn<TSession>;
  readonly noOutputTokenWindowOptions: LlamaNoOutputTokenWindowOptionsFn<TSession>;
  readonly outputTarget: LlamaTensorOutputTargetFn;
  readonly outputTensor: LlamaOutputTensorFn;
}>;

export type LlamaSessionPrefillFacadeHelpersOptions<TSession = unknown> = Readonly<{
  readonly bumpSessionCallProfile: BumpSessionCallProfileFn<TSession>;
  readonly executeTokens: LlamaExecuteTokensFn<TSession>;
  readonly outputTokenWindowOptions: LlamaOutputTokenWindowOptionsFn<TSession>;
  readonly outputTarget: LlamaTensorOutputTargetFn;
  readonly outputTensor: LlamaOutputTensorFn;
}>;

export type LlamaSessionTokenSelectionFacadeHelpersOptions<TSession extends AnyRecord = AnyRecord> = Readonly<{
  readonly f32: ValueToF32Fn;
  readonly isNativeBuffer: IsNativeBufferFn;
  readonly argmaxLogits: LlamaArgmaxLogitsFn;
  readonly sampleLogits: LlamaSampleLogitsFn;
  readonly executeArgmaxWindow: LlamaExecuteArgmaxWindowFn;
  readonly executeSampleWindow: LlamaExecuteSampleWindowFn;
  readonly generateArgmaxWindow: LlamaGenerateArgmaxWindowFn;
  readonly generateSampleWindow: LlamaGenerateSampleWindowFn;
}>;

export type LlamaTokenScratchFacadeHelpersOptions<TSession = unknown> = Readonly<{
  readonly scalarTokenScratch: TokenScratchFn<TSession>;
  readonly scalarTokenSampleOptionsScratch: ScratchFn<TSession>;
  readonly noOutputTokenWindowOptionsScratch: ScratchFn<TSession>;
  readonly outputTokenWindowOptionsScratch: ScratchFn<TSession>;
  readonly scalarTokenExecuteOptionsScratch: ScratchFn<TSession>;
}>;

function sessionExecutionPlanSignature(contract: AnyRecord, compatibility: AnyRecord) {
  return [
    "session-execution-plan",
    `contract=${contract.signature}`,
    `stepParams=${compatibility.stepParamsSignature}`,
    `accepted=${compatibility.accepted === true ? 1 : 0}`,
    `canExecute=${compatibility.canExecute === true ? 1 : 0}`,
    `hotPath=${compatibility.hotPath === true ? 1 : 0}`,
    `status=${compatibility.hotPathStatus}`,
    `state=${compatibility.stateEffect}`,
    `input=${compatibility.inputSource}`,
    `output=${compatibility.outputTarget}`,
    `effect=${compatibility.outputEffect}`,
  ].join("|");
}

export function createSessionLiveFacadeHelpers<
  TSession = unknown,
  THandle = unknown,
>(options: SessionLiveFacadeHelpersOptions<TSession, THandle>) {
  const handleForSession = options && options.handle;
  const assertSessionAlive = options && options.assertSessionAlive;
  const inspectHandle = options && options.inspect;
  const positionHandle = options && options.position;
  if (
    typeof handleForSession !== "function" ||
    typeof assertSessionAlive !== "function" ||
    typeof inspectHandle !== "function"
  ) {
    throw new Error("createSessionLiveFacadeHelpers requires handle, assertSessionAlive, and inspect callbacks");
  }
  if (positionHandle !== undefined && typeof positionHandle !== "function") {
    throw new Error("createSessionLiveFacadeHelpers position callback must be a function");
  }

  function handle(session: TSession): THandle {
    return handleForSession(session);
  }

  function assertLiveSession(session: TSession) {
    assertSessionAlive(handle(session));
  }

  function inspect(session: TSession) {
    assertLiveSession(session);
    return inspectHandle(handle(session));
  }

  function position(session: TSession) {
    if (typeof positionHandle !== "function") {
      throw new Error("Session.position is unavailable");
    }
    assertLiveSession(session);
    return positionHandle(handle(session));
  }

  return Object.freeze({
    handle,
    assertLiveSession,
    inspect,
    position,
  });
}

export function createSessionLayoutFacadeHelpers<TSession = unknown>(options: SessionLayoutFacadeHelpersOptions<TSession>) {
  const assertLiveSession = options && options.assertLiveSession;
  const bufferLayoutForSession = options && options.bufferLayout;
  const bufferSlotMethodName = options && options.bufferSlotMethodName;
  const kvCacheLayoutForSession = options && options.kvCacheLayout;
  const inputShapeForSession = options && options.inputShape;
  const callbacks = [
    options && options.outputShape,
    options && options.inputLen,
    options && options.outputLen,
    options && options.inputByteLength,
    options && options.outputByteLength,
    options && options.weightsLen,
    options && options.weightsByteLength,
    options && options.biasLen,
    options && options.biasByteLength,
    options && options.parameterLen,
    options && options.parameterByteLength,
  ];
  if (
    typeof assertLiveSession !== "function" ||
    typeof bufferLayoutForSession !== "function" ||
    typeof bufferSlotMethodName !== "string" ||
    bufferSlotMethodName.length === 0 ||
    callbacks.some((callback) => typeof callback !== "function")
  ) {
    throw new Error("createSessionLayoutFacadeHelpers requires assertLiveSession, bufferLayout, bufferSlotMethodName, and layout callbacks");
  }
  if (kvCacheLayoutForSession !== undefined && typeof kvCacheLayoutForSession !== "function") {
    throw new Error("createSessionLayoutFacadeHelpers kvCacheLayout callback must be a function");
  }
  if (inputShapeForSession !== undefined && typeof inputShapeForSession !== "function") {
    throw new Error("createSessionLayoutFacadeHelpers inputShape callback must be a function");
  }

  function liveValue(session: TSession, callback: SessionValueFn<TSession> | SessionNumberFn<TSession>) {
    assertLiveSession(session);
    return callback(session);
  }

  function bufferLayout(session: TSession) {
    assertLiveSession(session);
    return bufferLayoutForSession(session);
  }

  function bufferSlotNames(session: TSession) {
    return sessionBufferSlotNames(bufferLayout(session));
  }

  function bufferSlot(session: TSession, nameOrKind: unknown) {
    return sessionBufferSlot(bufferLayout(session), nameOrKind, bufferSlotMethodName);
  }

  function kvCacheLayout(session: TSession) {
    if (typeof kvCacheLayoutForSession !== "function") {
      throw new Error("Session.kvCacheLayout is unavailable");
    }
    assertLiveSession(session);
    return requireSessionKvCacheLayout(kvCacheLayoutForSession(session), "LLaMA Session kvCacheLayout");
  }

  function inputShape(session: TSession) {
    if (typeof inputShapeForSession !== "function") {
      throw new Error("Session.inputShape is unavailable");
    }
    return liveValue(session, inputShapeForSession);
  }

  function outputShape(session: TSession) {
    return liveValue(session, options.outputShape);
  }

  function inputLen(session: TSession) {
    return liveValue(session, options.inputLen);
  }

  function outputLen(session: TSession) {
    return liveValue(session, options.outputLen);
  }

  function inputByteLength(session: TSession) {
    return liveValue(session, options.inputByteLength);
  }

  function outputByteLength(session: TSession) {
    return liveValue(session, options.outputByteLength);
  }

  function weightsLen(session: TSession) {
    return liveValue(session, options.weightsLen);
  }

  function weightsByteLength(session: TSession) {
    return liveValue(session, options.weightsByteLength);
  }

  function biasLen(session: TSession) {
    return liveValue(session, options.biasLen);
  }

  function biasByteLength(session: TSession) {
    return liveValue(session, options.biasByteLength);
  }

  function parameterLen(session: TSession) {
    return liveValue(session, options.parameterLen);
  }

  function parameterByteLength(session: TSession) {
    return liveValue(session, options.parameterByteLength);
  }

  return Object.freeze({
    bufferLayout,
    bufferSlotNames,
    bufferSlot,
    kvCacheLayout,
    inputShape,
    outputShape,
    inputLen,
    outputLen,
    inputByteLength,
    outputByteLength,
    weightsLen,
    weightsByteLength,
    biasLen,
    biasByteLength,
    parameterLen,
    parameterByteLength,
  });
}

export function createSessionStepParamsFacadeHelpers<TSession = unknown>(options: SessionStepParamsFacadeHelpersOptions<TSession>) {
  const stepParamsCompatibilityForSession = options && options.stepParamsCompatibility;
  const stepContractForSession = options && options.stepContract;
  if (typeof stepParamsCompatibilityForSession !== "function" || typeof stepContractForSession !== "function") {
    throw new Error("createSessionStepParamsFacadeHelpers requires stepParamsCompatibility and stepContract callbacks");
  }

  function stepParamsCompatibility(session: TSession, params?: unknown) {
    return stepParamsCompatibilityForSession(session, params);
  }

  function preflightStepParams(session: TSession, params?: unknown) {
    return stepParamsCompatibility(session, params);
  }

  function acceptsStepParams(session: TSession, params?: unknown) {
    return acceptsStepParamsCompatibility(stepParamsCompatibility(session, params));
  }

  function canExecuteStepParams(session: TSession, params?: unknown) {
    return canExecuteStepParamsCompatibility(stepParamsCompatibility(session, params));
  }

  function requireCanExecuteStepParams(session: TSession, params?: unknown) {
    return requireCanExecuteStepParamsCompatibility(stepParamsCompatibility(session, params));
  }

  function acceptsAllocationFreeStepParams(session: TSession, params?: unknown) {
    return acceptsAllocationFreeStepParamsCompatibility(stepParamsCompatibility(session, params));
  }

  function requireAllocationFreeStepParams(session: TSession, params?: unknown) {
    return requireAllocationFreeStepParamsCompatibility(stepParamsCompatibility(session, params));
  }

  function acceptsRuntimeOutputAllocationFreeStepParams(session: TSession, params?: unknown) {
    return acceptsRuntimeOutputAllocationFreeStepParamsCompatibility(stepParamsCompatibility(session, params));
  }

  function requireRuntimeOutputAllocationFreeStepParams(session: TSession, params?: unknown) {
    return requireRuntimeOutputAllocationFreeStepParamsCompatibility(stepParamsCompatibility(session, params));
  }

  function acceptsNoReadbackStepParams(session: TSession, params?: unknown) {
    return acceptsNoReadbackStepParamsCompatibility(stepParamsCompatibility(session, params));
  }

  function requireNoReadbackStepParams(session: TSession, params?: unknown) {
    return requireNoReadbackStepParamsCompatibility(stepParamsCompatibility(session, params));
  }

  function acceptsReadbackFreeStepParams(session: TSession, params?: unknown) {
    return acceptsReadbackFreeStepParamsCompatibility(stepParamsCompatibility(session, params));
  }

  function requireReadbackFreeStepParams(session: TSession, params?: unknown) {
    return requireReadbackFreeStepParamsCompatibility(stepParamsCompatibility(session, params));
  }

  function acceptsHotStepParams(session: TSession, params?: unknown) {
    return acceptsHotStepParamsCompatibility(stepParamsCompatibility(session, params));
  }

  function requireHotStepParams(session: TSession, params?: unknown) {
    return requireHotStepParamsCompatibility(stepParamsCompatibility(session, params));
  }

  function executionPlan(session: TSession, params?: unknown) {
    const contract = stepContractForSession(session);
    const compatibility = stepParamsCompatibility(session, params);
    const signature = sessionExecutionPlanSignature(contract, compatibility);
    return Object.freeze({
      kind: "zgml.session.execution-plan",
      signature,
      contract,
      compatibility,
      accepted: compatibility.accepted === true,
      canExecute: compatibility.canExecute === true,
      hotPath: compatibility.hotPath === true,
      hotPathStatus: compatibility.hotPathStatus,
      hotPathBlockers: compatibility.hotPathBlockers,
      contractKind: contract.kind,
      contractSignature: contract.signature,
      stepParamsSignature: compatibility.stepParamsSignature,
      defaultHotPath: contract.defaultHotPath === true,
      defaultAllocationFree: contract.defaultAllocationFree === true,
      defaultReadbackRequired: contract.defaultReadbackRequired === true,
      allocationFree: compatibility.allocationFree === true,
      runtimeOutputAllocationFree: compatibility.runtimeOutputAllocationFree === true,
      readbackFree: compatibility.readbackFree === true,
      readbackRequired: compatibility.readbackRequired === true,
      stateEffect: compatibility.stateEffect,
      inputSource: compatibility.inputSource,
      inputOwnership: compatibility.inputOwnership,
      readsInput: compatibility.readsInput === true,
      outputTarget: compatibility.outputTarget,
      outputEffect: compatibility.outputEffect,
      outputOwnership: compatibility.outputOwnership,
      outputReturnOwnership: compatibility.outputReturnOwnership,
      writesOutput: compatibility.writesOutput === true,
      inputElementType: compatibility.inputElementType,
      outputElementType: compatibility.outputElementType,
      inputElementLength: compatibility.inputElementLength,
      outputElementLength: compatibility.outputElementLength,
      inputShape: compatibility.inputShape,
      outputShape: compatibility.outputShape,
      inputShapeSignature: compatibility.inputShapeSignature,
      outputShapeSignature: compatibility.outputShapeSignature,
      inputByteLength: compatibility.inputByteLength,
      outputByteLength: compatibility.outputByteLength,
      rejectionCode: compatibility.rejectionCode,
      diagnostics: compatibility.diagnostics,
    });
  }

  function hotPathPlan(session: TSession, params?: unknown) {
    return executionPlan(session, params);
  }

  function requireExecutionPlan(session: TSession, params?: unknown) {
    const plan = executionPlan(session, params);
    requireCanExecuteStepParamsCompatibility(plan.compatibility);
    return plan;
  }

  function matchesStepContractSignatureValue(session: TSession, signature: string) {
    return matchesStepContractSignature(stepContractForSession(session), signature);
  }

  function matchesStepParamsSignatureValue(session: TSession, params: unknown, signature: string) {
    return matchesStepParamsSignature(stepParamsCompatibility(session, params), signature);
  }

  function matchesStepParamsCompatibilityValue(session: TSession, params: unknown, compatibility: unknown) {
    return matchesStepParamsCompatibility(stepParamsCompatibility(session, params), compatibility);
  }

  return Object.freeze({
    stepParamsCompatibility,
    preflightStepParams,
    acceptsStepParams,
    canExecuteStepParams,
    requireCanExecuteStepParams,
    acceptsAllocationFreeStepParams,
    requireAllocationFreeStepParams,
    acceptsRuntimeOutputAllocationFreeStepParams,
    requireRuntimeOutputAllocationFreeStepParams,
    acceptsNoReadbackStepParams,
    requireNoReadbackStepParams,
    acceptsReadbackFreeStepParams,
    requireReadbackFreeStepParams,
    acceptsHotStepParams,
    requireHotStepParams,
    executionPlan,
    requireExecutionPlan,
    hotPathPlan,
    matchesStepContractSignature: matchesStepContractSignatureValue,
    matchesStepParamsSignature: matchesStepParamsSignatureValue,
    matchesStepParamsCompatibility: matchesStepParamsCompatibilityValue,
  });
}

export function createSessionCallProfileFacadeHelpers<TSession = unknown>(options: SessionCallProfileFacadeHelpersOptions<TSession>) {
  const assertLiveSession = options && options.assertLiveSession;
  if (typeof assertLiveSession !== "function") {
    throw new Error("createSessionCallProfileFacadeHelpers requires assertLiveSession callback");
  }

  function bumpSessionCallProfile(session: TSession, field: string) {
    bumpSessionCallProfileValue(session as AnyRecord, field);
  }

  function sessionCallProfile(session: TSession) {
    assertLiveSession(session);
    return sessionCallProfileValue(session as AnyRecord);
  }

  function matchesSessionCallProfileSignature(session: TSession, signature: unknown) {
    assertLiveSession(session);
    return matchesSessionCallProfileSignatureValue(session as AnyRecord, signature);
  }

  function resetSessionCallProfile(session: TSession) {
    assertLiveSession(session);
    resetSessionCallProfileValue(session as AnyRecord);
  }

  return Object.freeze({
    bumpSessionCallProfile,
    sessionCallProfile,
    session_call_profile: sessionCallProfile,
    matchesSessionCallProfileSignature,
    matches_session_call_profile_signature: matchesSessionCallProfileSignature,
    resetSessionCallProfile,
    reset_session_call_profile: resetSessionCallProfile,
  });
}

export function createSessionBufferSizingFacadeHelpers<TSession = unknown>(options: SessionBufferSizingFacadeHelpersOptions<TSession>) {
  const assertLiveSession = options && options.assertLiveSession;
  const modelKind = options && options.modelKind;
  const callbacks = [
    modelKind,
    options && options.inputLen,
    options && options.inputByteLength,
    options && options.outputLen,
    options && options.outputByteLength,
    options && options.weightsLen,
    options && options.weightsByteLength,
    options && options.biasLen,
    options && options.biasByteLength,
    options && options.parameterLen,
    options && options.parameterByteLength,
  ];
  if (typeof assertLiveSession !== "function" || callbacks.some((callback) => typeof callback !== "function")) {
    throw new Error("createSessionBufferSizingFacadeHelpers requires assertLiveSession and sizing callbacks");
  }

  function bufferSizing(session: TSession) {
    assertLiveSession(session);
    return sessionBufferSizing({
      kind: "session-buffer-sizing",
      scalarType: "f32",
      scalarBytes: 4,
      inputLen: options.inputLen(session),
      inputByteLength: options.inputByteLength(session),
      outputLen: options.outputLen(session),
      outputByteLength: options.outputByteLength(session),
      weightsLen: options.weightsLen(session),
      weightsByteLength: options.weightsByteLength(session),
      biasLen: options.biasLen(session),
      biasByteLength: options.biasByteLength(session),
      parameterLen: options.parameterLen(session),
      parameterByteLength: options.parameterByteLength(session),
      modelKind: modelKind(session),
    });
  }

  function matchesBufferSizingSignature(session: TSession, signature: unknown) {
    return typeof signature === "string" && signature.length > 0 && bufferSizing(session).signature === signature;
  }

  return Object.freeze({
    bufferSizing,
    matchesBufferSizingSignature,
  });
}

export function createSessionRuntimeProfileFacadeHelpers<TSession = unknown>(options: SessionRuntimeProfileFacadeHelpersOptions<TSession>) {
  const assertLiveSession = options && options.assertLiveSession;
  const runtimeProfileForSession = options && options.runtimeProfile;
  const resetRuntimeProfileForSession = options && options.resetRuntimeProfile;
  if (
    typeof assertLiveSession !== "function" ||
    typeof runtimeProfileForSession !== "function" ||
    typeof resetRuntimeProfileForSession !== "function"
  ) {
    throw new Error("createSessionRuntimeProfileFacadeHelpers requires assertLiveSession, runtimeProfile, and resetRuntimeProfile callbacks");
  }

  function runtimeProfile(session: TSession) {
    assertLiveSession(session);
    return runtimeProfileForSession(session);
  }

  function matchesRuntimeProfileSignature(session: TSession, signature: unknown) {
    return typeof signature === "string" && runtimeProfile(session).signature === signature;
  }

  function resetRuntimeProfile(session: TSession) {
    assertLiveSession(session);
    return resetRuntimeProfileForSession(session);
  }

  return Object.freeze({
    runtimeProfile,
    runtime_profile: runtimeProfile,
    matchesRuntimeProfileSignature,
    matches_runtime_profile_signature: matchesRuntimeProfileSignature,
    resetRuntimeProfile,
    reset_runtime_profile: resetRuntimeProfile,
  });
}

export function createSessionParameterFacadeHelpers<TSession = unknown>(options: SessionParameterFacadeHelpersOptions<TSession>) {
  const assertLiveSession = options && options.assertLiveSession;
  const bumpSessionCallProfile = options && options.bumpSessionCallProfile;
  const parameterLayoutForSession = options && options.parameterLayout;
  const uploadPersistent = options && options.uploadPersistent;
  const uploadPersistentRange = options && options.uploadPersistentRange;
  if (
    typeof assertLiveSession !== "function" ||
    typeof bumpSessionCallProfile !== "function" ||
    typeof parameterLayoutForSession !== "function" ||
    typeof uploadPersistent !== "function" ||
    typeof uploadPersistentRange !== "function"
  ) {
    throw new Error("createSessionParameterFacadeHelpers requires assertLiveSession, bumpSessionCallProfile, parameterLayout, uploadPersistent, and uploadPersistentRange callbacks");
  }

  function parameterLayout(session: TSession) {
    assertLiveSession(session);
    return parameterLayoutForSession(session) || null;
  }

  function parameterNames(session: TSession) {
    return sessionParameterNames(parameterLayout(session));
  }

  function parameterInfos(session: TSession) {
    return sessionParameterInfos(parameterLayout(session));
  }

  function parameterInfo(session: TSession, nameOrIndex: unknown) {
    return sessionParameterInfo(parameterLayout(session), nameOrIndex);
  }

  function uploadParameters(session: TSession) {
    assertLiveSession(session);
    const out = uploadPersistent(session);
    bumpSessionCallProfile(session, "uploadParametersCount");
    return out;
  }

  function uploadParameter(session: TSession, index: unknown) {
    assertLiveSession(session);
    const out = uploadPersistentRange(session, index, 1);
    bumpSessionCallProfile(session, "uploadParameterCount");
    return out;
  }

  function uploadParameterByName(session: TSession, name: unknown) {
    assertLiveSession(session);
    const layout = parameterLayout(session);
    const entry = sessionParameterInfoByNameForUpload(layout, name);
    const out = uploadPersistentRange(session, sessionParameterPersistentBindingIndex(entry, layout), 1);
    bumpSessionCallProfile(session, "uploadParameterByNameCount");
    return out;
  }

  function uploadParameterRange(session: TSession, first: unknown, len: unknown) {
    assertLiveSession(session);
    const out = uploadPersistentRange(session, first, len);
    bumpSessionCallProfile(session, "uploadParameterRangeCount");
    return out;
  }

  return Object.freeze({
    uploadParameters,
    uploadParameter,
    uploadParameterByName,
    uploadParameterRange,
    parameterLayout,
    parameterNames,
    parameterInfos,
    parameterInfo,
  });
}

export function createSessionLifecycleFacadeHelpers<TSession = unknown>(options: SessionLifecycleFacadeHelpersOptions<TSession>) {
  const assertLiveSession = options && options.assertLiveSession;
  const bumpSessionCallProfile = options && options.bumpSessionCallProfile;
  const resetSession = options && options.reset;
  const freeSession = options && options.free;
  if (
    typeof assertLiveSession !== "function" ||
    typeof bumpSessionCallProfile !== "function" ||
    typeof resetSession !== "function" ||
    typeof freeSession !== "function"
  ) {
    throw new Error("createSessionLifecycleFacadeHelpers requires assertLiveSession, bumpSessionCallProfile, reset, and free callbacks");
  }

  function reset(session: TSession) {
    assertLiveSession(session);
    const out = resetSession(session);
    bumpSessionCallProfile(session, "resetCount");
    return out;
  }

  function free(session: TSession) {
    return freeSession(session);
  }

  function dispose(session: TSession) {
    return free(session);
  }

  return Object.freeze({
    reset,
    free,
    dispose,
  });
}

export function createSessionReadbackFacadeHelpers<TSession = unknown>(options: SessionReadbackFacadeHelpersOptions<TSession>) {
  const bumpSessionCallProfile = options && options.bumpSessionCallProfile;
  const readOutputIntoCore = options && options.readOutputIntoCore;
  const readOutputTensorTarget = options && options.readOutputTensorTarget;
  const activeOutputValues = options && options.activeOutputValues;
  const outputTensor = options && options.outputTensor;
  if (
    typeof bumpSessionCallProfile !== "function" ||
    typeof readOutputIntoCore !== "function" ||
    typeof readOutputTensorTarget !== "function" ||
    typeof activeOutputValues !== "function" ||
    typeof outputTensor !== "function"
  ) {
    throw new Error("createSessionReadbackFacadeHelpers requires bumpSessionCallProfile, readOutputIntoCore, readOutputTensorTarget, activeOutputValues, and outputTensor callbacks");
  }

  function readOutputInto(session: TSession, outputValues: unknown, length?: unknown, byteOffset?: unknown) {
    const out = readOutputIntoCore(session, outputValues, length, byteOffset);
    bumpSessionCallProfile(session, "readOutputIntoCount");
    return out;
  }

  function readOutputTensor(session: TSession, tensorOptions: SessionReadOutputTensorOptions = {}) {
    const optionsForRead = tensorOptions || {};
    const target = readOutputTensorTarget(session, optionsForRead);
    readOutputIntoCore(session, target.values, target.length, target.byteOffset);
    const activeValues = activeOutputValues(target.values, target.length);
    const out = outputTensor(session, activeValues, optionsForRead, target);
    bumpSessionCallProfile(session, "readOutputTensorCount");
    return out;
  }

  return Object.freeze({
    readOutputInto,
    readOutputTensor,
  });
}

export function createGenericSessionExecutionFacadeHelpers<TSession extends AnyRecord = AnyRecord>(options: GenericSessionExecutionFacadeHelpersOptions<TSession>) {
  const bumpSessionCallProfile = options && options.bumpSessionCallProfile;
  const stepCore = options && options.stepCore;
  const stepIntoCore = options && options.stepIntoCore;
  const prepareExecuteIntoCore = options && options.prepareExecuteIntoCore;
  const advanceCore = options && options.advanceCore;
  const outputTensorForSession = options && options.outputTensorForSession;
  if (
    typeof bumpSessionCallProfile !== "function" ||
    typeof stepCore !== "function" ||
    typeof stepIntoCore !== "function" ||
    typeof prepareExecuteIntoCore !== "function" ||
    typeof advanceCore !== "function" ||
    typeof outputTensorForSession !== "function"
  ) {
    throw new Error("createGenericSessionExecutionFacadeHelpers requires bumpSessionCallProfile, stepCore, stepIntoCore, prepareExecuteIntoCore, advanceCore, and outputTensorForSession callbacks");
  }

  function execute(session: TSession, params: SessionExecuteParams = {}) {
    const plan = genericSessionExecutePlan(params, "execute");
    if (plan.noOutput) {
      advanceCore(session, plan.input);
      bumpSessionCallProfile(session, "executeCount");
      return undefined;
    }
    const out = stepCore(session, plan.input, plan.output);
    bumpSessionCallProfile(session, "executeCount");
    return out;
  }

  function executeTensor(session: TSession, params: SessionExecuteTensorParams = {}) {
    const plan = genericSessionExecuteTensorPlan(params);
    const values = stepCore(session, plan.input, plan.output);
    const out = outputTensorForSession(session.desc, values, plan.stepParams, session.boundOutputShape || undefined);
    bumpSessionCallProfile(session, "executeTensorCount");
    return out;
  }

  function executeInto(session: TSession, outputValues: unknown, params?: SessionExecuteIntoParams) {
    assertFloat32OutputBuffer(outputValues, "executeInto");
    if (params === undefined || params === null) {
      const out = stepIntoCore(session, outputValues, undefined);
      bumpSessionCallProfile(session, "executeIntoCount");
      return out;
    }
    const fastInput = inputOnlyExecuteIntoParam(params);
    if (fastInput !== noFastExecuteIntoInput) {
      const out = stepIntoCore(session, outputValues, fastInput);
      bumpSessionCallProfile(session, "executeIntoCount");
      return out;
    }
    const plan = genericSessionExecuteIntoPlan(params);
    const out = stepIntoCore(session, outputValues, plan.input);
    bumpSessionCallProfile(session, "executeIntoCount");
    return out;
  }

  function prepareExecuteInto(session: TSession, outputValues: unknown, params?: SessionExecuteIntoParams) {
    assertFloat32OutputBuffer(outputValues, "prepareExecuteInto");
    let runner: () => Float32Array;
    if (params === undefined || params === null) {
      runner = prepareExecuteIntoCore(session, outputValues, undefined);
    } else {
      const fastInput = inputOnlyExecuteIntoParam(params);
      if (fastInput !== noFastExecuteIntoInput) {
        runner = prepareExecuteIntoCore(session, outputValues, fastInput);
      } else {
        const plan = genericSessionExecuteIntoPlan(params);
        runner = prepareExecuteIntoCore(session, outputValues, plan.input);
      }
    }
    bumpSessionCallProfile(session, "prepareExecuteIntoCount");
    return runner;
  }

  function advance(session: TSession, inputValues: unknown) {
    const out = advanceCore(session, inputValues);
    bumpSessionCallProfile(session, "advanceCount");
    return out;
  }

  return Object.freeze({
    execute,
    executeTensor,
    executeInto,
    prepareExecuteInto,
    advance,
  });
}

export function createGenericSessionCoreStepFacadeHelpers<TSession extends AnyRecord = AnyRecord>(options: GenericSessionCoreStepFacadeHelpersOptions<TSession>) {
  const assertLiveSession = options && options.assertLiveSession;
  const isNativeBuffer = options && options.isNativeBuffer;
  const prepareHostValue = options && options.prepareHostValue;
  const validateHostValueShape = options && options.validateHostValueShape;
  const stepSession = options && options.stepSession;
  const prepareStepSession = typeof (options && options.prepareStepSession) === "function"
    ? options.prepareStepSession
    : null;
  const stepNoOutput = options && options.stepNoOutput;
  const stepContract = options && options.stepContract;
  if (
    typeof assertLiveSession !== "function" ||
    typeof isNativeBuffer !== "function" ||
    typeof prepareHostValue !== "function" ||
    typeof validateHostValueShape !== "function" ||
    typeof stepSession !== "function" ||
    typeof stepNoOutput !== "function" ||
    typeof stepContract !== "function"
  ) {
    throw new Error("createGenericSessionCoreStepFacadeHelpers requires assertLiveSession, isNativeBuffer, host value, native step, and stepContract callbacks");
  }
  const valueDeps = { prepareHostValue, validateHostValueShape };

  function hostBoundInput(session: TSession) {
    return hostBoundInputValue(session.boundInput, isNativeBuffer);
  }

  function nativeBoundOutput(session: TSession) {
    return nativeBoundOutputValue(session.boundOutput, isNativeBuffer);
  }

  function hostBoundOutput(session: TSession) {
    return hostBoundOutputValue(session.boundOutput, isNativeBuffer);
  }

  function explicitInput(session: TSession, inputValues: unknown, label: string) {
    return genericSessionExplicitInput({
      desc: session.desc,
      boundInput: session.boundInput,
      hostBoundInput: hostBoundInput(session),
    }, inputValues, label, valueDeps);
  }

  function explicitOutput(session: TSession, outputValues: unknown, label: string) {
    return genericSessionExplicitOutput(session.desc, outputValues, label, valueDeps);
  }

  function explicitStepIntoOutput(session: TSession, outputValues: unknown) {
    if (outputValues instanceof Float32Array) {
      if (outputValues.length < session.desc.outputLen) {
        throw stepParamsValidationError(
          "invalid-output",
          `session.step output length ${outputValues.length} is smaller than Program output length ${session.desc.outputLen}`,
          { actualLength: outputValues.length, expectedLength: session.desc.outputLen },
        );
      }
      return outputValues;
    }
    const output = explicitOutput(session, outputValues, "session.step output");
    if (output === null) {
      throw new Error("session.step output requires an output buffer");
    }
    return output;
  }

  function stepCore(session: TSession, inputValues: unknown, outputValues?: unknown) {
    assertLiveSession(session);
    const input = explicitInput(session, inputValues, "session.step input");
    const target = genericSessionStepOutputTarget(
      session.desc,
      outputValues,
      nativeBoundOutput(session),
      hostBoundOutput(session),
      "session.step output",
      valueDeps,
    );
    const outputLen = stepSession(session.handle, input, target.descOutput, session.desc.outputLen);
    return genericSessionStepResult(outputLen, target.output, nativeBoundOutput(session), target.useNativeOutput);
  }

  function stepIntoCore(session: TSession, outputValues: unknown, inputValues?: unknown) {
    assertLiveSession(session);
    const input = explicitInput(session, inputValues, "session.step input");
    const output = explicitStepIntoOutput(session, outputValues);
    const outputLen = stepSession(session.handle, input, output, session.desc.outputLen);
    return outputLen === output.length ? output : output.subarray(0, outputLen);
  }

  function prepareExecuteIntoCore(session: TSession, outputValues: unknown, inputValues?: unknown) {
    assertLiveSession(session);
    const input = explicitInput(session, inputValues, "session.prepareExecuteInto input");
    const output = explicitStepIntoOutput(session, outputValues);
    const outputLen = session.desc.outputLen;
    const preparedStep = prepareStepSession
      ? prepareStepSession(session.handle, input, output, outputLen)
      : null;
    if (preparedStep && output.length === outputLen) {
      return function preparedExactExecuteInto() {
        preparedStep();
        return output;
      };
    }
    return function preparedExecuteInto() {
      const actualOutputLen = preparedStep ? preparedStep() : stepSession(session.handle, input, output, outputLen);
      return actualOutputLen === output.length ? output : output.subarray(0, actualOutputLen);
    };
  }

  function advanceCore(session: TSession, inputValues: unknown) {
    assertLiveSession(session);
    const input = explicitInput(session, inputValues, "session.advance input");
    const outputLen = stepNoOutput(session.handle, input);
    assertGenericSessionNoOutputStepResult(outputLen);
  }

  function stepParamsCompatibility(session: TSession, params: unknown = {}) {
    return genericSessionStepParamsCompatibility({
      desc: session.desc,
      boundInput: session.boundInput,
      hostBoundInput: hostBoundInput(session),
    }, stepContract(session) as any, params, valueDeps);
  }

  return Object.freeze({
    hostBoundInput,
    nativeBoundOutput,
    hostBoundOutput,
    explicitInput,
    explicitOutput,
    stepCore,
    stepIntoCore,
    prepareExecuteIntoCore,
    advanceCore,
    stepParamsCompatibility,
  });
}

export function createGenericSessionStepFacadeHelpers<TSession extends AnyRecord = AnyRecord>(options: GenericSessionStepFacadeHelpersOptions<TSession>) {
  const bumpSessionCallProfile = options && options.bumpSessionCallProfile;
  const stepCore = options && options.stepCore;
  const stepIntoCore = options && options.stepIntoCore;
  const outputTensorForSession = options && options.outputTensorForSession;
  if (
    typeof bumpSessionCallProfile !== "function" ||
    typeof stepCore !== "function" ||
    typeof stepIntoCore !== "function" ||
    typeof outputTensorForSession !== "function"
  ) {
    throw new Error("createGenericSessionStepFacadeHelpers requires bumpSessionCallProfile, stepCore, stepIntoCore, and outputTensorForSession callbacks");
  }

  function step(session: TSession, inputValues: unknown, outputValues?: unknown) {
    const out = stepCore(session, inputValues, outputValues);
    bumpSessionCallProfile(session, "stepCount");
    return out;
  }

  function stepTensor(session: TSession, inputValues: unknown, tensorOptions: SessionStepTensorOptions = {}) {
    const options = tensorOptions || {};
    const output = options.output !== undefined ? options.output : undefined;
    const values = stepCore(session, inputValues, output);
    const out = outputTensorForSession(session.desc, values, options, session.boundOutputShape || undefined);
    bumpSessionCallProfile(session, "stepTensorCount");
    return out;
  }

  function stepInto(session: TSession, outputValues: unknown, inputValues: unknown) {
    assertFloat32OutputBuffer(outputValues, "stepInto");
    const out = stepIntoCore(session, outputValues, inputValues);
    bumpSessionCallProfile(session, "stepIntoCount");
    return out;
  }

  return Object.freeze({
    step,
    stepTensor,
    stepInto,
  });
}

export function createLlamaSessionStepFacadeHelpers<TSession extends AnyRecord = AnyRecord>(options: LlamaSessionStepFacadeHelpersOptions<TSession>) {
  const assertLiveSession = options && options.assertLiveSession;
  const bumpSessionCallProfile = options && options.bumpSessionCallProfile;
  const stepToken = options && options.stepToken;
  const advanceToken = options && options.advanceToken;
  const outputTarget = options && options.outputTarget;
  const outputTensor = options && options.outputTensor;
  if (
    typeof assertLiveSession !== "function" ||
    typeof bumpSessionCallProfile !== "function" ||
    typeof stepToken !== "function" ||
    typeof advanceToken !== "function" ||
    typeof outputTarget !== "function" ||
    typeof outputTensor !== "function"
  ) {
    throw new Error("createLlamaSessionStepFacadeHelpers requires assertLiveSession, bumpSessionCallProfile, stepToken, advanceToken, outputTarget, and outputTensor callbacks");
  }

  function step(session: TSession, token: unknown, outputValues?: unknown) {
    assertLiveSession(session);
    if (outputValues !== undefined) {
      validateLogitsOutputBuffer(outputValues, session.vocabSize, "session.step");
    }
    const out = stepToken(session.handle, token, outputValues, session.boundOutput, session.vocabSize);
    bumpSessionCallProfile(session, "stepCount");
    return out;
  }

  function stepTensor(session: TSession, token: unknown, options: LlamaLogitsTensorOptions = {}) {
    assertLiveSession(session);
    const opts = options || {};
    const output = llamaSessionStepTensorOutputTarget(session.vocabSize, opts, outputTarget);
    const values = stepToken(session.handle, token, output, session.boundOutput, session.vocabSize);
    const out = outputTensor(values, opts);
    bumpSessionCallProfile(session, "stepTensorCount");
    return out;
  }

  function stepInto(session: TSession, outputValues: unknown, token: unknown) {
    assertFloat32OutputBuffer(outputValues, "stepInto");
    assertLiveSession(session);
    validateLogitsOutputBuffer(outputValues, session.vocabSize, "session.stepInto");
    const out = stepToken(session.handle, token, outputValues, session.boundOutput, session.vocabSize);
    bumpSessionCallProfile(session, "stepIntoCount");
    return out;
  }

  function advance(session: TSession, token: unknown) {
    assertLiveSession(session);
    const out = advanceToken(session.handle, token);
    bumpSessionCallProfile(session, "advanceCount");
    return out;
  }

  return Object.freeze({
    step,
    stepTensor,
    stepInto,
    advance,
  });
}

export function createLlamaSessionExecutionFacadeHelpers<TSession extends AnyRecord = AnyRecord>(options: LlamaSessionExecutionFacadeHelpersOptions<TSession>) {
  const bumpSessionCallProfile = options && options.bumpSessionCallProfile;
  const isNativeBuffer = options && options.isNativeBuffer;
  const executeTokenWindow = options && options.executeTokenWindow;
  const scalarTokenWindowForSession = options && options.scalarTokenWindow;
  const scalarTokenExecuteOptionsForSession = options && options.scalarTokenExecuteOptions;
  const outputTokenWindowOptionsForSession = options && options.outputTokenWindowOptions;
  const scalarTokenOutputOptionsForSession = options && options.scalarTokenOutputOptions;
  const noOutputTokenWindowOptionsForSession = options && options.noOutputTokenWindowOptions;
  const outputTarget = options && options.outputTarget;
  const outputTensor = options && options.outputTensor;
  if (
    typeof bumpSessionCallProfile !== "function" ||
    typeof isNativeBuffer !== "function" ||
    typeof executeTokenWindow !== "function" ||
    typeof scalarTokenWindowForSession !== "function" ||
    typeof scalarTokenExecuteOptionsForSession !== "function" ||
    typeof outputTokenWindowOptionsForSession !== "function" ||
    typeof scalarTokenOutputOptionsForSession !== "function" ||
    typeof noOutputTokenWindowOptionsForSession !== "function" ||
    typeof outputTarget !== "function" ||
    typeof outputTensor !== "function"
  ) {
    throw new Error("createLlamaSessionExecutionFacadeHelpers requires bumpSessionCallProfile, isNativeBuffer, executeTokenWindow, token option helpers, outputTarget, and outputTensor callbacks");
  }

  function nativeBoundOutput(session: TSession) {
    return isNativeBuffer(session.boundOutput) ? session.boundOutput : null;
  }

  function executeTokens(session: TSession, tokens: unknown, options: unknown = {}) {
    const target = llamaSessionExecuteTokenOutputTarget(
      session.vocabSize,
      (options || {}) as AnyRecord,
      nativeBoundOutput(session),
    );
    const window = tokenWindow(tokens as any, target.opts, "execute token window");
    const outputLen = executeTokenWindow(session.handle, window, target.wantsLogits, target.descOutput);
    return llamaSessionExecuteTokenResult(
      outputLen,
      target.output,
      nativeBoundOutput(session),
      target.wantsLogits,
      target.useNativeOutput,
    );
  }

  function executeCore(session: TSession, params: unknown, method: string) {
    const plan = llamaSessionExecutePlan(params, method);
    if (plan.hasTokens) {
      return executeTokens(session, plan.stepParams.tokens, plan.stepParams);
    }
    return executeTokens(
      session,
      scalarTokenWindowForSession(session, plan.stepParams.token),
      scalarTokenExecuteOptionsForSession(session, plan.stepParams),
    );
  }

  function execute(session: TSession, params: unknown) {
    const out = executeCore(session, params, "execute");
    bumpSessionCallProfile(session, "executeCount");
    return out;
  }

  function executeTensor(session: TSession, params: LlamaExecuteTensorParams) {
    const plan = llamaSessionExecuteTensorPlan(params, outputTarget);
    const values = executeCore(session, plan.stepParams, "executeTensor");
    const out = outputTensor(requireLlamaSessionLogitsOutput(values), params);
    bumpSessionCallProfile(session, "executeTensorCount");
    return out;
  }

  function executeInto(session: TSession, outputValues: unknown, params: unknown) {
    assertFloat32OutputBuffer(outputValues, "executeInto");
    const plan = llamaSessionExecuteIntoPlan(params);
    const out = plan.hasTokens
      ? executeTokens(session, plan.stepParams.tokens, outputTokenWindowOptionsForSession(session, plan.stepParams, outputValues))
      : executeTokens(
        session,
        scalarTokenWindowForSession(session, plan.stepParams.token),
        scalarTokenOutputOptionsForSession(session, outputValues),
      );
    bumpSessionCallProfile(session, "executeIntoCount");
    return out;
  }

  function advanceTokens(session: TSession, tokens: unknown, options: TokenWindowOptions = {}) {
    executeTokens(session, tokens, noOutputTokenWindowOptionsForSession(session, options || {}));
    bumpSessionCallProfile(session, "advanceCount");
  }

  return Object.freeze({
    executeTokens,
    execute,
    executeTensor,
    executeInto,
    advanceTokens,
  });
}

export function createLlamaSessionPrefillFacadeHelpers<TSession = unknown>(options: LlamaSessionPrefillFacadeHelpersOptions<TSession>) {
  const bumpSessionCallProfile = options && options.bumpSessionCallProfile;
  const executeTokens = options && options.executeTokens;
  const outputTokenWindowOptionsForSession = options && options.outputTokenWindowOptions;
  const outputTarget = options && options.outputTarget;
  const outputTensor = options && options.outputTensor;
  if (
    typeof bumpSessionCallProfile !== "function" ||
    typeof executeTokens !== "function" ||
    typeof outputTokenWindowOptionsForSession !== "function" ||
    typeof outputTarget !== "function" ||
    typeof outputTensor !== "function"
  ) {
    throw new Error("createLlamaSessionPrefillFacadeHelpers requires bumpSessionCallProfile, executeTokens, outputTokenWindowOptions, outputTarget, and outputTensor callbacks");
  }

  function prefill(session: TSession, tokens: unknown, outputValues: unknown, options: TokenWindowOptions = {}) {
    const out = executeTokens(session, tokens, outputTokenWindowOptionsForSession(session, options || {}, outputValues));
    bumpSessionCallProfile(session, "prefillCount");
    return out;
  }

  function prefillTensor(session: TSession, tokens: unknown, options: TokenWindowOptions = {}) {
    const plan = llamaSessionPrefillTensorPlan(options || {}, outputTarget);
    const values = executeTokens(session, tokens, outputTokenWindowOptionsForSession(session, plan.options, plan.output));
    const out = outputTensor(values, plan.options);
    bumpSessionCallProfile(session, "prefillTensorCount");
    return out;
  }

  function prefillInto(session: TSession, outputValues: unknown, tokens: unknown, options: TokenWindowOptions = {}) {
    assertFloat32OutputBuffer(outputValues, "prefillInto");
    const out = executeTokens(session, tokens, outputTokenWindowOptionsForSession(session, options || {}, outputValues));
    bumpSessionCallProfile(session, "prefillIntoCount");
    return out;
  }

  return Object.freeze({
    prefill,
    prefillTensor,
    prefillInto,
  });
}

export function createLlamaSessionTokenSelectionFacadeHelpers<TSession extends AnyRecord = AnyRecord>(options: LlamaSessionTokenSelectionFacadeHelpersOptions<TSession>) {
  const f32 = options && options.f32;
  const isNativeBuffer = options && options.isNativeBuffer;
  const argmaxLogits = options && options.argmaxLogits;
  const sampleLogits = options && options.sampleLogits;
  const executeArgmaxWindow = options && options.executeArgmaxWindow;
  const executeSampleWindow = options && options.executeSampleWindow;
  const generateArgmaxWindow = options && options.generateArgmaxWindow;
  const generateSampleWindow = options && options.generateSampleWindow;
  if (
    typeof f32 !== "function" ||
    typeof isNativeBuffer !== "function" ||
    typeof argmaxLogits !== "function" ||
    typeof sampleLogits !== "function" ||
    typeof executeArgmaxWindow !== "function" ||
    typeof executeSampleWindow !== "function" ||
    typeof generateArgmaxWindow !== "function" ||
    typeof generateSampleWindow !== "function"
  ) {
    throw new Error("createLlamaSessionTokenSelectionFacadeHelpers requires f32, isNativeBuffer, token selection, execution, and generation callbacks");
  }

  function argmaxToken(session: TSession, logits: unknown) {
    const data = logits == null ? null : f32(logits);
    return argmaxLogits(session.handle, data);
  }

  function sampleToken(session: TSession, logits: unknown, options: TokenSampleOptions = {}) {
    const data = logits == null ? null : f32(logits);
    return sampleLogits(session.handle, data, llamaSessionSampleOptions(options || {}));
  }

  function executeTokensArgmax(session: TSession, tokens: unknown, options: TokenWindowOptions = {}) {
    const plan = llamaSessionArgmaxExecutionPlan(session.boundOutput, isNativeBuffer, tokens, options || {});
    return executeArgmaxWindow(session.handle, plan.window);
  }

  function stepArgmax(session: TSession, token: unknown) {
    return session.executeTokensArgmax(session.scalarTokenWindow(token), session.scalarTokenWindowOptionsScratch);
  }

  function executeTokensSample(session: TSession, tokens: unknown, options: TokenSampleOptions = {}) {
    const plan = llamaSessionSampleExecutionPlan(session.boundOutput, isNativeBuffer, tokens, options || {});
    return executeSampleWindow(session.handle, plan.window, plan.sampleOptions);
  }

  function stepSample(session: TSession, token: unknown, options: TokenSampleOptions = {}) {
    return session.executeTokensSample(
      session.scalarTokenWindow(token),
      session.scalarTokenSampleOptions(options || {}),
    );
  }

  function generateTokensArgmaxInto(session: TSession, tokens: unknown, outputTokens: Uint32Array, options: TokenWindowOptions = {}) {
    const plan = llamaSessionNativeArgmaxGenerationPlan(session.boundOutput, isNativeBuffer, tokens, outputTokens, options || {});
    return generateArgmaxWindow(session.handle, plan.window, plan.outputTokens);
  }

  function generateTokensArgmax(session: TSession, tokens: unknown, maxTokens: number, options: TokenWindowOptions = {}) {
    return generateTokensArgmaxInto(session, tokens, llamaSessionGeneratedOutput(maxTokens), options || {});
  }

  function generateTokensSampleInto(session: TSession, tokens: unknown, outputTokens: Uint32Array, options: TokenSampleOptions = {}) {
    const plan = llamaSessionNativeSampleGenerationPlan(session.boundOutput, isNativeBuffer, tokens, outputTokens, options || {});
    return generateSampleWindow(session.handle, plan.window, plan.outputTokens, plan.sampleOptions);
  }

  function generateTokensSample(session: TSession, tokens: unknown, maxTokens: number, options: TokenSampleOptions = {}) {
    return generateTokensSampleInto(session, tokens, llamaSessionGeneratedOutput(maxTokens), options || {});
  }

  return Object.freeze({
    argmaxToken,
    sampleToken,
    executeTokensArgmax,
    stepArgmax,
    executeTokensSample,
    stepSample,
    generateTokensArgmax,
    generateTokensArgmaxInto,
    generateTokensSample,
    generateTokensSampleInto,
  });
}

export function createLlamaTokenScratchFacadeHelpers<TSession = unknown>(options: LlamaTokenScratchFacadeHelpersOptions<TSession>) {
  const scalarTokenScratch = options && options.scalarTokenScratch;
  const scalarTokenSampleOptionsScratch = options && options.scalarTokenSampleOptionsScratch;
  const noOutputTokenWindowOptionsScratch = options && options.noOutputTokenWindowOptionsScratch;
  const outputTokenWindowOptionsScratch = options && options.outputTokenWindowOptionsScratch;
  const scalarTokenExecuteOptionsScratch = options && options.scalarTokenExecuteOptionsScratch;
  const callbacks = [
    scalarTokenScratch,
    scalarTokenSampleOptionsScratch,
    noOutputTokenWindowOptionsScratch,
    outputTokenWindowOptionsScratch,
    scalarTokenExecuteOptionsScratch,
  ];
  if (callbacks.some((callback) => typeof callback !== "function")) {
    throw new Error("createLlamaTokenScratchFacadeHelpers requires scalar token and option scratch callbacks");
  }

  function scalarTokenWindow(session: TSession, token: unknown) {
    const scratch = scalarTokenScratch(session);
    scratch[0] = tokenId(token);
    return scratch;
  }

  function sessionScalarTokenSampleOptions(session: TSession, options: TokenSampleOptions = {}) {
    return scalarTokenSampleOptions(options, scalarTokenSampleOptionsScratch(session));
  }

  function sessionNoOutputTokenWindowOptions(session: TSession, options: TokenWindowOptions = {}) {
    return noOutputTokenWindowOptions(options, noOutputTokenWindowOptionsScratch(session));
  }

  function sessionOutputTokenWindowOptions(session: TSession, options: TokenWindowOptions = {}, outputValues: unknown) {
    return outputTokenWindowOptions(options, outputTokenWindowOptionsScratch(session), outputValues);
  }

  function sessionScalarTokenExecuteOptions(session: TSession, options: TokenWindowOptions & { output?: unknown } = {}) {
    return scalarTokenExecuteOptions(options, scalarTokenExecuteOptionsScratch(session));
  }

  function sessionScalarTokenOutputOptions(session: TSession, outputValues: unknown) {
    return scalarTokenOutputOptions(scalarTokenExecuteOptionsScratch(session), outputValues);
  }

  return Object.freeze({
    scalarTokenWindow,
    sessionScalarTokenSampleOptions,
    sessionNoOutputTokenWindowOptions,
    sessionOutputTokenWindowOptions,
    sessionScalarTokenExecuteOptions,
    sessionScalarTokenOutputOptions,
  });
}
