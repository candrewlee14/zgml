"use strict";

import {
  createShapedF32Helpers,
} from "../core/shape.js";
import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";
import type {
  NativeBuffer,
  ProgramBufferLayout,
  RuntimeProfile,
  SessionInspection,
  TokenSampleOptions,
  TokenWindowOptions,
} from "../public_api.js";
import {
  programOutputShape,
  type SessionTensorHelpers,
} from "./session_tensor.js";
import {
  assertFloat32OutputBuffer,
  assertNoInlineOutput,
  validateLlamaTokenStepParams,
  noOutputTokenWindowOptions,
  outputTokenWindowOptions,
  scalarTokenExecuteOptions,
  scalarTokenOutputOptions,
  scalarTokenSampleOptions,
} from "./step_params.js";
import {
  createSessionLiveFacadeHelpers,
  createSessionStepParamsFacadeHelpers,
  createSessionCallProfileFacadeHelpers,
  createSessionBufferSizingFacadeHelpers,
  createSessionRuntimeProfileFacadeHelpers,
  createSessionParameterFacadeHelpers,
  createSessionLifecycleFacadeHelpers,
  createSessionLayoutFacadeHelpers,
  createSessionReadbackFacadeHelpers,
  createGenericSessionExecutionFacadeHelpers,
  createGenericSessionCoreStepFacadeHelpers,
  createGenericSessionStepFacadeHelpers,
  createLlamaSessionStepFacadeHelpers,
  createLlamaSessionExecutionFacadeHelpers,
  createLlamaSessionPrefillFacadeHelpers,
  createLlamaSessionTokenSelectionFacadeHelpers,
  createLlamaTokenScratchFacadeHelpers,
} from "./session_facade.js";
import {
  createLlamaSessionStepContract,
  createGenericSessionStepContract,
} from "./session_contract.js";
import {
  bindLlamaSessionHandleByPolicy,
  type LlamaBindOptionsRecord,
} from "./session_binding.js";
import * as sessionBinding from "./session_binding.js";
import * as sessionLayout from "./session_layout.js";
import * as sessionLifecycle from "./session_lifecycle.js";
import * as sessionValues from "./session_values.js";

type AnyRecord = Record<string, any>;
type NullableRecord = AnyRecord | null | undefined;
type NativeHandle = unknown;
type BoundaryCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];
type SessionOutputBuffer = NativeBuffer | Readonly<Record<string, unknown>>;
type SessionStepValue = unknown;
type TokenWindow = unknown;
type TokenWindowScratch = TokenWindowOptions & { output?: unknown };
type TokenSampleScratch = TokenSampleOptions;
type RuntimeProfileEvidence = RuntimeProfile | Readonly<Record<string, unknown>>;
type SessionInspectionEvidence = SessionInspection | Readonly<Record<string, unknown>>;
type PreparedF32 = Readonly<{ data: Float32Array; shape: readonly number[] }>;
type LlamaSessionRecord = AnyRecord & {
  readonly handle: NativeHandle;
  readonly boundBufferLayout: ProgramBufferLayout;
  readonly boundKvCacheLayout: unknown;
  readonly boundOutput: unknown;
  readonly vocabSize: number;
  readonly scalarTokenScratch: Uint32Array;
  readonly scalarTokenSampleOptionsScratch: TokenSampleScratch;
  readonly noOutputTokenWindowOptionsScratch: TokenWindowScratch;
  readonly outputTokenWindowOptionsScratch: TokenWindowScratch;
  readonly scalarTokenExecuteOptionsScratch: TokenWindowScratch;
  readonly scalarTokenWindow: BoundaryCallback<[token: unknown], unknown>;
  readonly scalarTokenExecuteOptions: BoundaryCallback<[options: TokenWindowOptions & { output?: unknown }], unknown>;
  readonly outputTokenWindowOptions: BoundaryCallback<[options: TokenWindowOptions, outputValues: unknown], unknown>;
  readonly scalarTokenOutputOptions: BoundaryCallback<[outputValues: unknown], unknown>;
  readonly noOutputTokenWindowOptions: BoundaryCallback<[options: TokenWindowOptions], unknown>;
  readonly executeTokens: BoundaryCallback<[tokens: unknown, options?: unknown], unknown>;
  readonly executeTokensArgmax: BoundaryCallback<[tokens: unknown, options?: unknown], unknown>;
  readonly executeTokensSample: BoundaryCallback<[tokens: unknown, options?: unknown], unknown>;
  readonly scalarTokenSampleOptions: BoundaryCallback<[options: TokenSampleOptions], unknown>;
};
type GenericSessionRecord = AnyRecord & {
  readonly handle: NativeHandle;
  readonly desc: AnyRecord & {
    readonly inputLen: number;
    readonly outputLen: number;
    readonly inputShape?: readonly number[];
    readonly outputShape?: readonly number[];
  };
  readonly boundBufferLayout: ProgramBufferLayout;
  readonly boundInput: unknown;
  readonly boundOutput: unknown;
  readonly boundOutputShape?: readonly number[] | null;
  readonly boundParameterLayout?: NullableRecord;
};

export type LlamaSessionFacadeOptions = {
  readonly isNativeBuffer: BoundaryCallback<[value: unknown], boolean>;
  readonly f32: BoundaryCallback<[value: unknown], Float32Array>;
  readonly createProgramOutputBuffer: BoundaryCallback<[programHandle: NativeHandle], SessionOutputBuffer>;
  readonly sessionTensorHelpers: SessionTensorHelpers;
  readonly executeTokenWindow: BoundaryCallback<[handle: NativeHandle, window: TokenWindow, wantsLogits: boolean, output: SessionStepValue], number>;
  readonly stepToken: BoundaryCallback<[
    handle: NativeHandle,
    token: unknown,
    outputValues: SessionStepValue,
    boundOutput: SessionStepValue,
    vocabSize: number,
  ], unknown>;
  readonly advanceToken: BoundaryCallback<[handle: NativeHandle, token: unknown], unknown>;
  readonly assertSessionAlive: BoundaryCallback<[handle: NativeHandle], void>;
  readonly sessionPosition: BoundaryCallback<[handle: NativeHandle], unknown>;
  readonly sessionInspect: BoundaryCallback<[handle: NativeHandle], SessionInspectionEvidence>;
  readonly sessionReset: BoundaryCallback<[handle: NativeHandle], unknown>;
  readonly sessionRuntimeProfile: BoundaryCallback<[handle: NativeHandle], RuntimeProfileEvidence>;
  readonly sessionResetRuntimeProfile: BoundaryCallback<[handle: NativeHandle], unknown>;
  readonly sessionFree: BoundaryCallback<[handle: NativeHandle], unknown>;
  readonly nullSessionHandle: NativeHandle;
  readonly argmaxLogits: BoundaryCallback<[handle: NativeHandle, data: Float32Array | null], unknown>;
  readonly sampleLogits: BoundaryCallback<[handle: NativeHandle, data: Float32Array | null, options: TokenSampleOptions], unknown>;
  readonly executeArgmaxWindow: BoundaryCallback<[handle: NativeHandle, window: TokenWindow], unknown>;
  readonly executeSampleWindow: BoundaryCallback<[handle: NativeHandle, window: TokenWindow, options: TokenSampleOptions], unknown>;
  readonly generateArgmaxWindow: BoundaryCallback<[handle: NativeHandle, window: TokenWindow, outputTokens: Uint32Array], unknown>;
  readonly generateSampleWindow: BoundaryCallback<[
    handle: NativeHandle,
    window: TokenWindow,
    outputTokens: Uint32Array,
    options: TokenSampleOptions,
  ], unknown>;
};

export type GenericSessionFacadeOptions = {
  readonly isNativeBuffer: BoundaryCallback<[value: unknown], boolean>;
  readonly f32: BoundaryCallback<[value: unknown], Float32Array>;
  readonly prepareF32: BoundaryCallback<[value: unknown], PreparedF32>;
  readonly valueShape?: BoundaryCallback<[value: unknown], readonly number[] | null>;
  readonly sessionTensorHelpers: SessionTensorHelpers;
  readonly stepSession: BoundaryCallback<[handle: NativeHandle, input: SessionStepValue, output: SessionStepValue, outputLen: number], number>;
  readonly prepareStepSession?: BoundaryCallback<[handle: NativeHandle, input: SessionStepValue, output: SessionStepValue, outputLen: number], () => number>;
  readonly stepNoOutput: BoundaryCallback<[handle: NativeHandle, input: SessionStepValue], number>;
  readonly assertSessionAlive: BoundaryCallback<[handle: NativeHandle], void>;
  readonly sessionInspect: BoundaryCallback<[handle: NativeHandle], SessionInspectionEvidence>;
  readonly sessionUploadPersistent: BoundaryCallback<[handle: NativeHandle], unknown>;
  readonly sessionUploadPersistentRange: BoundaryCallback<[handle: NativeHandle, first: number, len: number], unknown>;
  readonly sessionReset: BoundaryCallback<[handle: NativeHandle], unknown>;
  readonly sessionRuntimeProfile: BoundaryCallback<[handle: NativeHandle], RuntimeProfileEvidence>;
  readonly sessionResetRuntimeProfile: BoundaryCallback<[handle: NativeHandle], unknown>;
  readonly sessionFree: BoundaryCallback<[handle: NativeHandle], unknown>;
  readonly nullSessionHandle: NativeHandle;
};

function hasOwn(value: unknown, key: string): boolean {
  return !!value && Object.prototype.hasOwnProperty.call(value, key);
}

function numberFrom<TSession>(callback: (session: TSession) => unknown, session: TSession): number {
  return Number(callback(session));
}

function shapeFrom<TSession>(callback: (session: TSession) => unknown, session: TSession): readonly number[] {
  return callback(session) as readonly number[];
}

export function createLlamaSessionFacadeHelpers(options: LlamaSessionFacadeOptions) {
  const isNativeBuffer = options && options.isNativeBuffer;
  const f32 = options && options.f32;
  if (typeof isNativeBuffer !== "function") {
    throw new Error("createLlamaSessionFacadeHelpers requires isNativeBuffer");
  }
  if (typeof f32 !== "function") {
    throw new Error("createLlamaSessionFacadeHelpers requires f32");
  }
  const createProgramOutputBuffer = options && options.createProgramOutputBuffer;
  if (typeof createProgramOutputBuffer !== "function") {
    throw new Error("createLlamaSessionFacadeHelpers requires createProgramOutputBuffer");
  }
  const sessionTensorHelpers = options && options.sessionTensorHelpers;
  if (
    !sessionTensorHelpers ||
    typeof sessionTensorHelpers.outputTarget !== "function" ||
    typeof sessionTensorHelpers.outputTensor !== "function"
  ) {
    throw new Error("createLlamaSessionFacadeHelpers requires sessionTensorHelpers");
  }
  const executeTokenWindow = options && options.executeTokenWindow;
  if (typeof executeTokenWindow !== "function") {
    throw new Error("createLlamaSessionFacadeHelpers requires executeTokenWindow");
  }
  const stepToken = options && options.stepToken;
  if (typeof stepToken !== "function") {
    throw new Error("createLlamaSessionFacadeHelpers requires stepToken");
  }
  const advanceToken = options && options.advanceToken;
  if (typeof advanceToken !== "function") {
    throw new Error("createLlamaSessionFacadeHelpers requires advanceToken");
  }
  const assertSessionAlive = options && options.assertSessionAlive;
  const sessionPosition = options && options.sessionPosition;
  const sessionInspect = options && options.sessionInspect;
  const sessionReset = options && options.sessionReset;
  const sessionRuntimeProfile = options && options.sessionRuntimeProfile;
  const sessionResetRuntimeProfile = options && options.sessionResetRuntimeProfile;
  const sessionFree = options && options.sessionFree;
  const hasNullSessionHandle = hasOwn(options, "nullSessionHandle");
  const nullSessionHandle = hasNullSessionHandle ? options.nullSessionHandle : null;
  if (
    typeof assertSessionAlive !== "function" ||
    typeof sessionPosition !== "function" ||
    typeof sessionInspect !== "function" ||
    typeof sessionReset !== "function" ||
    typeof sessionRuntimeProfile !== "function" ||
    typeof sessionResetRuntimeProfile !== "function" ||
    typeof sessionFree !== "function" ||
    !hasNullSessionHandle
  ) {
    throw new Error("createLlamaSessionFacadeHelpers requires session lifecycle callbacks");
  }
  const argmaxLogits = options && options.argmaxLogits;
  const sampleLogits = options && options.sampleLogits;
  const executeArgmaxWindow = options && options.executeArgmaxWindow;
  const executeSampleWindow = options && options.executeSampleWindow;
  const generateArgmaxWindow = options && options.generateArgmaxWindow;
  const generateSampleWindow = options && options.generateSampleWindow;
  if (
    typeof argmaxLogits !== "function" ||
    typeof sampleLogits !== "function" ||
    typeof executeArgmaxWindow !== "function" ||
    typeof executeSampleWindow !== "function" ||
    typeof generateArgmaxWindow !== "function" ||
    typeof generateSampleWindow !== "function"
  ) {
    throw new Error("createLlamaSessionFacadeHelpers requires token selection and generation callbacks");
  }

  function normalizeLlamaBindOptions(programHandle: unknown, optionsForBind: LlamaBindOptionsRecord = {}) {
    return sessionBinding.normalizeLlamaBindOptions(programHandle, optionsForBind, {
      createProgramOutputBuffer,
      isNativeBuffer,
    });
  }

  function bindLlamaProgramSession(
    programHandle: unknown,
    program: unknown,
    optionsForBind: LlamaBindOptionsRecord,
    bindSessionHandle: (...args: any[]) => unknown,
    createSession: (...args: any[]) => unknown,
  ) {
    return sessionBinding.bindLlamaProgramSession(programHandle, program, optionsForBind, {
      bindSessionHandle,
      createProgramOutputBuffer,
      createSession,
      isNativeBuffer,
    });
  }

  function requireNativeOutputBinding(boundOutput: unknown, method: string) {
    return sessionLayout.requireNativeOutputBinding(boundOutput, isNativeBuffer, method);
  }

  function validateStepParams(params: unknown, method: string) {
    return validateLlamaTokenStepParams(params, method);
  }

  function stepParamsCompatibility(session: LlamaSessionRecord, params?: unknown) {
    const contract = stepContract(session);
    return sessionValues.llamaSessionStepParamsCompatibility(contract, params);
  }

  const liveFacade = createSessionLiveFacadeHelpers<LlamaSessionRecord, NativeHandle>({
    handle(session: LlamaSessionRecord) {
      return session.handle;
    },
    assertSessionAlive,
    inspect: sessionInspect,
    position: sessionPosition,
  });
  const {
    assertLiveSession,
    inspect,
    position,
  } = liveFacade;

  const stepParamsFacade = createSessionStepParamsFacadeHelpers<LlamaSessionRecord>({
    stepContract,
    stepParamsCompatibility,
  });
  const {
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
    matchesStepContractSignature,
    matchesStepParamsSignature,
    matchesStepParamsCompatibility,
  } = stepParamsFacade;

  function defaultSessionBufferLayout(vocabSize: number, bufferLayout: unknown = null) {
    return sessionLayout.defaultLlamaSessionBufferLayout(vocabSize, bufferLayout);
  }

  function createSessionScratch() {
    return sessionLayout.createLlamaSessionScratch();
  }

  const layoutFacade = createSessionLayoutFacadeHelpers<LlamaSessionRecord>({
    assertLiveSession,
    bufferLayout(session: LlamaSessionRecord) {
      return session.boundBufferLayout;
    },
    bufferSlotMethodName: "TinyLlamaSession.bufferSlot",
    kvCacheLayout(session: LlamaSessionRecord) {
      return session.boundKvCacheLayout;
    },
    outputShape(session: LlamaSessionRecord) {
      return sessionLayout.llamaSessionOutputShape(session.vocabSize);
    },
    inputLen() {
      return sessionLayout.llamaSessionInputLen();
    },
    outputLen(session: LlamaSessionRecord) {
      return sessionLayout.llamaSessionOutputLen(session.vocabSize);
    },
    inputByteLength() {
      return sessionLayout.llamaSessionInputByteLength();
    },
    outputByteLength(session: LlamaSessionRecord) {
      return sessionLayout.llamaSessionOutputByteLength(session.vocabSize);
    },
    weightsLen() {
      return sessionLayout.zeroSessionElementLength();
    },
    weightsByteLength() {
      return sessionLayout.zeroSessionByteLength();
    },
    biasLen() {
      return sessionLayout.zeroSessionElementLength();
    },
    biasByteLength() {
      return sessionLayout.zeroSessionByteLength();
    },
    parameterLen() {
      return sessionLayout.zeroSessionElementLength();
    },
    parameterByteLength() {
      return sessionLayout.zeroSessionByteLength();
    },
  });
  const {
    bufferLayout,
    bufferSlotNames,
    bufferSlot,
    kvCacheLayout,
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
  } = layoutFacade;

  const bufferSizingFacade = createSessionBufferSizingFacadeHelpers<LlamaSessionRecord>({
    assertLiveSession,
    modelKind(session: LlamaSessionRecord) {
      return inspect(session).modelKind;
    },
    inputLen(session: LlamaSessionRecord) {
      return numberFrom(inputLen, session);
    },
    inputByteLength(session: LlamaSessionRecord) {
      return numberFrom(inputByteLength, session);
    },
    outputLen(session: LlamaSessionRecord) {
      return numberFrom(outputLen, session);
    },
    outputByteLength(session: LlamaSessionRecord) {
      return numberFrom(outputByteLength, session);
    },
    weightsLen(session: LlamaSessionRecord) {
      return numberFrom(weightsLen, session);
    },
    weightsByteLength(session: LlamaSessionRecord) {
      return numberFrom(weightsByteLength, session);
    },
    biasLen(session: LlamaSessionRecord) {
      return numberFrom(biasLen, session);
    },
    biasByteLength(session: LlamaSessionRecord) {
      return numberFrom(biasByteLength, session);
    },
    parameterLen(session: LlamaSessionRecord) {
      return numberFrom(parameterLen, session);
    },
    parameterByteLength(session: LlamaSessionRecord) {
      return numberFrom(parameterByteLength, session);
    },
  });
  const {
    bufferSizing,
    matchesBufferSizingSignature,
  } = bufferSizingFacade;

  function stepContract(session: LlamaSessionRecord) {
    assertLiveSession(session);
    const inspection = inspect(session);
    const hasBoundOutput = isNativeBuffer(session.boundOutput);
    return createLlamaSessionStepContract({
      vocabSize: session.vocabSize,
      outputShape: shapeFrom(outputShape, session),
      outputByteLength: numberFrom(outputByteLength, session),
      contextLength: inspection.contextLen,
      position: inspection.position,
      hasBoundOutput,
    });
  }

  const runtimeProfileFacade = createSessionRuntimeProfileFacadeHelpers<LlamaSessionRecord>({
    assertLiveSession,
    runtimeProfile(session: LlamaSessionRecord) {
      return sessionRuntimeProfile(session.handle);
    },
    resetRuntimeProfile(session: LlamaSessionRecord) {
      return sessionResetRuntimeProfile(session.handle);
    },
  });
  const {
    runtimeProfile,
    matchesRuntimeProfileSignature,
    resetRuntimeProfile,
  } = runtimeProfileFacade;

  const callProfileFacade = createSessionCallProfileFacadeHelpers<LlamaSessionRecord>({
    assertLiveSession,
  });
  const {
    bumpSessionCallProfile,
    sessionCallProfile,
    matchesSessionCallProfileSignature,
    resetSessionCallProfile,
  } = callProfileFacade;

  const lifecycleFacade = createSessionLifecycleFacadeHelpers<LlamaSessionRecord>({
    assertLiveSession,
    bumpSessionCallProfile,
    reset(session: LlamaSessionRecord) {
      return sessionReset(session.handle);
    },
    free(session: LlamaSessionRecord) {
      return sessionLifecycle.freeLlamaSessionResources(session, {
        nullSessionHandle,
        sessionFree,
      });
    },
  });
  const {
    reset,
    free,
    dispose,
  } = lifecycleFacade;

  const tokenScratchFacade = createLlamaTokenScratchFacadeHelpers<LlamaSessionRecord>({
    scalarTokenScratch(session: LlamaSessionRecord) {
      return session.scalarTokenScratch;
    },
    scalarTokenSampleOptionsScratch(session: LlamaSessionRecord) {
      return session.scalarTokenSampleOptionsScratch;
    },
    noOutputTokenWindowOptionsScratch(session: LlamaSessionRecord) {
      return session.noOutputTokenWindowOptionsScratch;
    },
    outputTokenWindowOptionsScratch(session: LlamaSessionRecord) {
      return session.outputTokenWindowOptionsScratch;
    },
    scalarTokenExecuteOptionsScratch(session: LlamaSessionRecord) {
      return session.scalarTokenExecuteOptionsScratch;
    },
  });
  const {
    scalarTokenWindow,
    sessionScalarTokenSampleOptions,
    sessionNoOutputTokenWindowOptions,
    sessionOutputTokenWindowOptions,
    sessionScalarTokenExecuteOptions,
    sessionScalarTokenOutputOptions,
  } = tokenScratchFacade;

  function readOutputIntoCore(
    session: LlamaSessionRecord,
    outputValues: unknown,
    length?: unknown,
    byteOffset?: unknown,
  ) {
    requireNativeOutputBinding(session.boundOutput, "readOutputInto");
    const outputLenForSession = length === undefined ? session.vocabSize : length;
    const outputByteOffset = byteOffset === undefined ? 0 : byteOffset;
    return sessionTensorHelpers.readOutputInto(
      session.boundOutput,
      outputValues,
      outputLenForSession,
      outputByteOffset,
      "TinyLlamaSession.readOutputInto",
    );
  }

  const readbackFacade = createSessionReadbackFacadeHelpers<LlamaSessionRecord>({
    bumpSessionCallProfile,
    readOutputIntoCore,
    readOutputTensorTarget(session: LlamaSessionRecord, optionsForRead: AnyRecord = {}) {
      return sessionValues.llamaSessionReadOutputTensorTarget(
        session.vocabSize,
        optionsForRead,
        sessionTensorHelpers.outputTarget,
      );
    },
    activeOutputValues: sessionValues.genericSessionActiveOutputValues,
    outputTensor(_session: LlamaSessionRecord, activeValues: Float32Array, _options: AnyRecord, target: AnyRecord) {
      return sessionTensorHelpers.outputTensor(activeValues, target.opts, [activeValues.length]);
    },
  });
  const {
    readOutputInto,
    readOutputTensor,
  } = readbackFacade;

  const executionFacade = createLlamaSessionExecutionFacadeHelpers<LlamaSessionRecord>({
    bumpSessionCallProfile,
    isNativeBuffer,
    executeTokenWindow,
    scalarTokenWindow(session: LlamaSessionRecord, token: unknown) {
      return session.scalarTokenWindow(token);
    },
    scalarTokenExecuteOptions(session: LlamaSessionRecord, optionsForExecute: AnyRecord) {
      return session.scalarTokenExecuteOptions(optionsForExecute);
    },
    outputTokenWindowOptions(session: LlamaSessionRecord, optionsForOutput: AnyRecord, outputValues: unknown) {
      return session.outputTokenWindowOptions(optionsForOutput, outputValues);
    },
    scalarTokenOutputOptions(session: LlamaSessionRecord, outputValues: unknown) {
      return session.scalarTokenOutputOptions(outputValues);
    },
    noOutputTokenWindowOptions(session: LlamaSessionRecord, optionsForNoOutput: AnyRecord) {
      return session.noOutputTokenWindowOptions(optionsForNoOutput);
    },
    outputTarget: sessionTensorHelpers.outputTarget,
    outputTensor: sessionTensorHelpers.outputTensor,
  });
  const {
    executeTokens,
    execute,
    executeTensor,
    executeInto,
    advanceTokens,
  } = executionFacade;

  const stepFacade = createLlamaSessionStepFacadeHelpers<LlamaSessionRecord>({
    assertLiveSession,
    bumpSessionCallProfile,
    stepToken,
    advanceToken,
    outputTarget: sessionTensorHelpers.outputTarget,
    outputTensor: sessionTensorHelpers.outputTensor,
  });
  const {
    step,
    stepTensor,
    stepInto,
    advance,
  } = stepFacade;

  const prefillFacade = createLlamaSessionPrefillFacadeHelpers<LlamaSessionRecord>({
    bumpSessionCallProfile,
    executeTokens(session: LlamaSessionRecord, tokens: unknown, optionsForExecute?: unknown) {
      return session.executeTokens(tokens, optionsForExecute);
    },
    outputTokenWindowOptions(session: LlamaSessionRecord, optionsForOutput: AnyRecord, outputValues: unknown) {
      return session.outputTokenWindowOptions(optionsForOutput, outputValues);
    },
    outputTarget: sessionTensorHelpers.outputTarget,
    outputTensor: sessionTensorHelpers.outputTensor,
  });
  const {
    prefill,
    prefillTensor,
    prefillInto,
  } = prefillFacade;

  const tokenSelectionFacade = createLlamaSessionTokenSelectionFacadeHelpers<LlamaSessionRecord>({
    f32,
    isNativeBuffer,
    argmaxLogits,
    sampleLogits,
    executeArgmaxWindow,
    executeSampleWindow,
    generateArgmaxWindow,
    generateSampleWindow,
  });
  const {
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
  } = tokenSelectionFacade;

  return Object.freeze({
    normalizeLlamaBindOptions,
    bindLlamaProgramSession,
    requireNativeOutputBinding,
    assertFloat32OutputBuffer,
    validateStepParams,
    stepParamsCompatibility,
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
    matchesStepContractSignature,
    matchesStepParamsSignature,
    matchesStepParamsCompatibility,
    assertNoInlineOutput,
    noOutputTokenWindowOptions,
    outputTokenWindowOptions,
    scalarTokenExecuteOptions,
    scalarTokenOutputOptions,
    scalarTokenSampleOptions,
    defaultSessionBufferLayout,
    createSessionScratch,
    bufferLayout,
    bufferSlotNames,
    bufferSlot,
    kvCacheLayout,
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
    bufferSizing,
    matchesBufferSizingSignature,
    outputShape,
    stepContract,
    preflightStepParams,
    position,
    inspect,
    reset,
    sessionCallProfile,
    session_call_profile: sessionCallProfile,
    matchesSessionCallProfileSignature,
    matches_session_call_profile_signature: matchesSessionCallProfileSignature,
    resetSessionCallProfile,
    reset_session_call_profile: resetSessionCallProfile,
    runtimeProfile,
    matchesRuntimeProfileSignature,
    resetRuntimeProfile,
    free,
    dispose,
    scalarTokenWindow,
    sessionScalarTokenSampleOptions,
    sessionNoOutputTokenWindowOptions,
    sessionOutputTokenWindowOptions,
    sessionScalarTokenExecuteOptions,
    sessionScalarTokenOutputOptions,
    readOutputInto,
    readOutputTensor,
    executeTokens,
    execute,
    executeTensor,
    executeInto,
    step,
    advance,
    advanceTokens,
    stepTensor,
    stepInto,
    prefill,
    prefillTensor,
    prefillInto,
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

export function createGenericSessionFacadeHelpers(options: GenericSessionFacadeOptions) {
  const isNativeBuffer = options && options.isNativeBuffer;
  const f32 = options && options.f32;
  const prepareF32 = options && options.prepareF32;
  const valueShape = typeof (options && options.valueShape) === "function" ? options.valueShape : () => null;
  const sessionTensorHelpers = options && options.sessionTensorHelpers;
  const stepSession = options && options.stepSession;
  const prepareStepSession = typeof (options && options.prepareStepSession) === "function" ? options.prepareStepSession : undefined;
  const stepNoOutput = options && options.stepNoOutput;
  const assertSessionAlive = options && options.assertSessionAlive;
  const sessionInspect = options && options.sessionInspect;
  const sessionUploadPersistent = options && options.sessionUploadPersistent;
  const sessionUploadPersistentRange = options && options.sessionUploadPersistentRange;
  const sessionReset = options && options.sessionReset;
  const sessionRuntimeProfile = options && options.sessionRuntimeProfile;
  const sessionResetRuntimeProfile = options && options.sessionResetRuntimeProfile;
  const sessionFree = options && options.sessionFree;
  const hasNullSessionHandle = hasOwn(options, "nullSessionHandle");
  const nullSessionHandle = hasNullSessionHandle ? options.nullSessionHandle : null;

  if (typeof isNativeBuffer !== "function") {
    throw new Error("createGenericSessionFacadeHelpers requires isNativeBuffer");
  }
  if (typeof f32 !== "function") {
    throw new Error("createGenericSessionFacadeHelpers requires f32");
  }
  if (typeof prepareF32 !== "function") {
    throw new Error("createGenericSessionFacadeHelpers requires prepareF32");
  }
  const {
    prepareHostValue,
    validateHostValueShape,
  } = createShapedF32Helpers({ f32, prepareF32, valueShape, label: "createGenericSessionFacadeHelpers" });
  if (
    !sessionTensorHelpers ||
    typeof sessionTensorHelpers.outputTarget !== "function" ||
    typeof sessionTensorHelpers.readOutputInto !== "function" ||
    typeof sessionTensorHelpers.outputTensorForSession !== "function"
  ) {
    throw new Error("createGenericSessionFacadeHelpers requires sessionTensorHelpers");
  }
  if (typeof stepSession !== "function") {
    throw new Error("createGenericSessionFacadeHelpers requires stepSession");
  }
  if (typeof stepNoOutput !== "function") {
    throw new Error("createGenericSessionFacadeHelpers requires stepNoOutput");
  }
  if (
    typeof assertSessionAlive !== "function" ||
    typeof sessionInspect !== "function" ||
    typeof sessionUploadPersistent !== "function" ||
    typeof sessionUploadPersistentRange !== "function" ||
    typeof sessionReset !== "function" ||
    typeof sessionRuntimeProfile !== "function" ||
    typeof sessionResetRuntimeProfile !== "function" ||
    typeof sessionFree !== "function" ||
    !hasNullSessionHandle
  ) {
    throw new Error("createGenericSessionFacadeHelpers requires session lifecycle callbacks");
  }

  const liveFacade = createSessionLiveFacadeHelpers<GenericSessionRecord, NativeHandle>({
    handle(session: GenericSessionRecord) {
      return session.handle;
    },
    assertSessionAlive,
    inspect: sessionInspect,
  });
  const {
    assertLiveSession,
    inspect,
  } = liveFacade;

  const layoutFacade = createSessionLayoutFacadeHelpers<GenericSessionRecord>({
    assertLiveSession,
    bufferLayout(session: GenericSessionRecord) {
      return session.boundBufferLayout;
    },
    bufferSlotMethodName: "Session.bufferSlot",
    inputShape(session: GenericSessionRecord) {
      return sessionLayout.genericSessionInputShape(session.desc);
    },
    outputShape(session: GenericSessionRecord) {
      return sessionLayout.genericSessionOutputShape(session.desc, session.boundOutputShape);
    },
    inputLen(session: GenericSessionRecord) {
      return sessionLayout.genericSessionInputLen(session.desc);
    },
    outputLen(session: GenericSessionRecord) {
      return sessionLayout.genericSessionOutputLen(session.desc);
    },
    inputByteLength(session: GenericSessionRecord) {
      return sessionLayout.genericSessionInputByteLength(session.desc);
    },
    outputByteLength(session: GenericSessionRecord) {
      return sessionLayout.genericSessionOutputByteLength(session.desc);
    },
    weightsLen(session: GenericSessionRecord) {
      return sessionLayout.sessionWeightsLen(session.boundBufferLayout);
    },
    weightsByteLength(session: GenericSessionRecord) {
      return sessionLayout.sessionWeightsByteLength(session.boundBufferLayout);
    },
    biasLen(session: GenericSessionRecord) {
      return sessionLayout.sessionBiasLen(session.boundBufferLayout);
    },
    biasByteLength(session: GenericSessionRecord) {
      return sessionLayout.sessionBiasByteLength(session.boundBufferLayout);
    },
    parameterLen(session: GenericSessionRecord) {
      return sessionLayout.sessionParameterLen(session.boundBufferLayout);
    },
    parameterByteLength(session: GenericSessionRecord) {
      return sessionLayout.sessionParameterByteLength(session.boundBufferLayout);
    },
  });
  const {
    bufferLayout,
    bufferSlotNames,
    bufferSlot,
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
  } = layoutFacade;

  const bufferSizingFacade = createSessionBufferSizingFacadeHelpers<GenericSessionRecord>({
    assertLiveSession,
    modelKind(session: GenericSessionRecord) {
      return session.desc.modelKind || "tiny-linear";
    },
    inputLen(session: GenericSessionRecord) {
      return numberFrom(inputLen, session);
    },
    inputByteLength(session: GenericSessionRecord) {
      return numberFrom(inputByteLength, session);
    },
    outputLen(session: GenericSessionRecord) {
      return numberFrom(outputLen, session);
    },
    outputByteLength(session: GenericSessionRecord) {
      return numberFrom(outputByteLength, session);
    },
    weightsLen(session: GenericSessionRecord) {
      return numberFrom(weightsLen, session);
    },
    weightsByteLength(session: GenericSessionRecord) {
      return numberFrom(weightsByteLength, session);
    },
    biasLen(session: GenericSessionRecord) {
      return numberFrom(biasLen, session);
    },
    biasByteLength(session: GenericSessionRecord) {
      return numberFrom(biasByteLength, session);
    },
    parameterLen(session: GenericSessionRecord) {
      return numberFrom(parameterLen, session);
    },
    parameterByteLength(session: GenericSessionRecord) {
      return numberFrom(parameterByteLength, session);
    },
  });
  const {
    bufferSizing,
    matchesBufferSizingSignature,
  } = bufferSizingFacade;

  function boundBufferKind(value: unknown) {
    return sessionLayout.boundBufferKind(value, isNativeBuffer);
  }

  function stepContract(session: GenericSessionRecord) {
    assertLiveSession(session);
    const boundInput = boundBufferKind(session.boundInput);
    const boundOutput = boundBufferKind(session.boundOutput);
    return createGenericSessionStepContract({
      inputLen: session.desc.inputLen,
      outputLen: session.desc.outputLen,
      inputByteLength: numberFrom(inputByteLength, session),
      outputByteLength: numberFrom(outputByteLength, session),
      inputShape: shapeFrom(inputShape, session),
      outputShape: shapeFrom(outputShape, session),
      programOutputShape: programOutputShape(session.desc),
      boundInput,
      boundOutput,
    });
  }

  const callProfileFacade = createSessionCallProfileFacadeHelpers<GenericSessionRecord>({
    assertLiveSession,
  });
  const {
    bumpSessionCallProfile,
    sessionCallProfile,
    matchesSessionCallProfileSignature,
    resetSessionCallProfile,
  } = callProfileFacade;

  const parameterFacade = createSessionParameterFacadeHelpers<GenericSessionRecord>({
    assertLiveSession,
    bumpSessionCallProfile,
    parameterLayout(session: GenericSessionRecord): NullableRecord {
      return session.boundParameterLayout || null;
    },
    uploadPersistent(session: GenericSessionRecord) {
      return sessionUploadPersistent(session.handle);
    },
    uploadPersistentRange(session: GenericSessionRecord, first: unknown, len: unknown) {
      return sessionUploadPersistentRange(session.handle, Number(first), Number(len));
    },
  });
  const {
    uploadParameters,
    uploadParameter,
    uploadParameterByName,
    uploadParameterRange,
    parameterLayout,
    parameterNames,
    parameterInfos,
    parameterInfo,
  } = parameterFacade;

  const coreStepFacade = createGenericSessionCoreStepFacadeHelpers<GenericSessionRecord>({
    assertLiveSession,
    isNativeBuffer,
    prepareHostValue,
    validateHostValueShape(prepared: AnyRecord, expectedShape: readonly number[], label: string, validateOptions?: AnyRecord) {
      return validateHostValueShape(prepared as any, expectedShape, label, validateOptions);
    },
    stepSession,
    prepareStepSession,
    stepNoOutput,
    stepContract,
  });
  const {
    stepCore,
    stepIntoCore,
    prepareExecuteIntoCore,
    advanceCore,
    stepParamsCompatibility,
  } = coreStepFacade;

  const stepFacade = createGenericSessionStepFacadeHelpers<GenericSessionRecord>({
    bumpSessionCallProfile,
    stepCore,
    stepIntoCore,
    outputTensorForSession: sessionTensorHelpers.outputTensorForSession,
  });
  const {
    step,
    stepTensor,
    stepInto,
  } = stepFacade;

  const stepParamsFacade = createSessionStepParamsFacadeHelpers<GenericSessionRecord>({
    stepContract,
    stepParamsCompatibility,
  });
  const {
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
    matchesStepContractSignature,
    matchesStepParamsSignature,
    matchesStepParamsCompatibility,
  } = stepParamsFacade;

  function readOutputIntoCore(
    session: GenericSessionRecord,
    outputValues: unknown,
    length?: unknown,
    byteOffset?: unknown,
  ) {
    assertLiveSession(session);
    const outputLenForSession = length === undefined ? session.desc.outputLen : length;
    const outputByteOffset = byteOffset === undefined ? 0 : byteOffset;
    return sessionTensorHelpers.readOutputInto(
      session.boundOutput,
      outputValues,
      outputLenForSession,
      outputByteOffset,
    );
  }

  const readbackFacade = createSessionReadbackFacadeHelpers<GenericSessionRecord>({
    bumpSessionCallProfile,
    readOutputIntoCore,
    readOutputTensorTarget(session: GenericSessionRecord, tensorOptions: AnyRecord = {}) {
      return sessionValues.genericSessionReadOutputTensorTarget(
        session.desc,
        tensorOptions,
        sessionTensorHelpers.outputTarget,
      );
    },
    activeOutputValues: sessionValues.genericSessionActiveOutputValues,
    outputTensor(session: GenericSessionRecord, activeValues: Float32Array, tensorOptions: AnyRecord) {
      return sessionTensorHelpers.outputTensorForSession(
        session.desc,
        activeValues,
        tensorOptions,
        session.boundOutputShape || undefined,
        "Session.readOutputTensor output shape",
      );
    },
  });
  const {
    readOutputInto,
    readOutputTensor,
  } = readbackFacade;

  const executionFacade = createGenericSessionExecutionFacadeHelpers<GenericSessionRecord>({
    bumpSessionCallProfile,
    stepCore,
    stepIntoCore,
    prepareExecuteIntoCore,
    advanceCore,
    outputTensorForSession: sessionTensorHelpers.outputTensorForSession,
  });
  const {
    execute,
    executeTensor,
    executeInto,
    prepareExecuteInto,
    advance,
  } = executionFacade;

  const runtimeProfileFacade = createSessionRuntimeProfileFacadeHelpers<GenericSessionRecord>({
    assertLiveSession,
    runtimeProfile(session: GenericSessionRecord) {
      return sessionRuntimeProfile(session.handle);
    },
    resetRuntimeProfile(session: GenericSessionRecord) {
      return sessionResetRuntimeProfile(session.handle);
    },
  });
  const {
    runtimeProfile,
    matchesRuntimeProfileSignature,
    resetRuntimeProfile,
  } = runtimeProfileFacade;

  const lifecycleFacade = createSessionLifecycleFacadeHelpers<GenericSessionRecord>({
    assertLiveSession,
    bumpSessionCallProfile,
    reset(session: GenericSessionRecord) {
      return sessionReset(session.handle);
    },
    free(session: GenericSessionRecord) {
      return sessionLifecycle.freeGenericSessionResources(session, {
        nullSessionHandle,
        sessionFree,
      });
    },
  });
  const {
    reset,
    free,
    dispose,
  } = lifecycleFacade;

  return Object.freeze({
    inspect,
    bufferLayout,
    bufferSlotNames,
    bufferSlot,
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
    bufferSizing,
    matchesBufferSizingSignature,
    sessionCallProfile,
    session_call_profile: sessionCallProfile,
    matchesSessionCallProfileSignature,
    matches_session_call_profile_signature: matchesSessionCallProfileSignature,
    resetSessionCallProfile,
    reset_session_call_profile: resetSessionCallProfile,
    stepContract,
    preflightStepParams,
    stepParamsCompatibility,
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
    matchesStepContractSignature,
    matchesStepParamsSignature,
    matchesStepParamsCompatibility,
    uploadParameters,
    uploadParameter,
    uploadParameterByName,
    uploadParameterRange,
    parameterLayout,
    parameterNames,
    parameterInfos,
    parameterInfo,
    step,
    stepTensor,
    stepInto,
    execute,
    executeTensor,
    executeInto,
    prepareExecuteInto,
    readOutputInto,
    readOutputTensor,
    advance,
    reset,
    runtimeProfile,
    matchesRuntimeProfileSignature,
    resetRuntimeProfile,
    free,
    dispose,
  });
}

export {
  bindLlamaSessionHandleByPolicy,
};

export const sessionFacadeCompositionManifest = Object.freeze({
  kind: "zgml-session-facade-composition",
  ...tsRuntimeManifestPolicy("src/ts/runtime/session_facade_composition.ts", "Program -> Session -> StepParams"),
  genericSessionComposition: "ts",
  llamaSessionComposition: "ts",
  hostBoundary: "native adapter callbacks",
});
