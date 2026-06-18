"use strict";

import {
  normalizeTokenSampleOptions,
  tokenId,
  tokenOutputArray,
  validateMaxTokens,
  tokenWindow,
} from "../core/token.js";
import {
  programInputShape,
  programOutputShape,
} from "./session_tensor.js";
import {
  requireNativeOutputBinding,
} from "./session_layout.js";
import type {
  ExecuteTokensOptions,
  LlamaExecuteParams,
  LlamaExecuteTensorParams,
  LlamaLogitsTensorOptions,
  LlamaTokenWindowTensorOptions,
  SessionReadOutputTensorOptions,
  TokenSampleOptions,
  TokenIds,
  TokenWindowOptions,
} from "../public_api.js";
import type {
  TensorOutputOptions,
} from "./session_tensor.js";
import {
  assertNoInlineOutput,
  stepParamsCompatibilityResult,
  stepParamsValidationError,
  validateLlamaTokenStepParams,
  validateLogitsOutputBuffer,
  validateTokenWindowFitsContract,
  validateExecuteStepParams,
} from "./step_params.js";

type UnknownRecord = Record<string, unknown>;
type StepParamsRecord = UnknownRecord & TokenWindowOptions & {
  input?: unknown;
  output?: unknown;
  token?: unknown;
  tokens?: TokenIds;
};
type StepParamsDiagnosticLike = {
  readonly stepParamsDiagnosticCode?: unknown;
};

type PreparedHostValue = {
  data: Float32Array;
  shape: number[];
};

type GenericSessionDesc = {
  inputLen: number;
  outputLen: number;
  inputShape?: readonly number[];
  outputShape?: readonly number[];
};

type BaseStepContract = Readonly<UnknownRecord & {
  readonly kind: string;
  readonly signature: string;
  readonly position?: number;
}>;

type GenericSessionStepContract = BaseStepContract & {
  readonly scalarType: string;
  readonly scalarBytes: number;
  readonly inputLen: number;
  readonly outputLen: number;
  readonly inputShape: readonly number[];
  readonly outputShape: readonly number[];
  readonly boundInput: string;
  readonly defaultOutput: string;
};

type LlamaStepContract = BaseStepContract & {
  readonly scalarType: string;
  readonly scalarBytes: number;
  readonly tokenIdBytes: number;
  readonly outputLen: number;
  readonly outputShape: readonly number[];
  readonly defaultOutput: string;
};

type GenericSessionValueDeps = {
  prepareHostValue: (value: unknown) => PreparedHostValue;
  validateHostValueShape: (
    prepared: PreparedHostValue,
    expectedShape: readonly number[],
    label: string,
    options?: UnknownRecord,
  ) => void;
};

type GenericSessionValueContext = {
  desc: GenericSessionDesc;
  boundInput?: unknown;
  hostBoundInput?: unknown;
};

type NativeReadbackOutput = {
  readFloat32: (length: number) => Float32Array;
};

type TokenWindowInput = {
  readonly length: number;
  readonly [index: number]: unknown;
};

export type GenericSessionStepOutputTarget = Readonly<{
  output: unknown;
  descOutput: unknown;
  useNativeOutput: boolean;
}>;

export type SessionReadOutputTensorTarget = Readonly<{
  length: unknown;
  byteOffset: unknown;
  values: unknown;
}>;

export type LlamaReadOutputTensorTarget = Readonly<{
  opts: SessionReadOutputTensorOptions;
  length: unknown;
  byteOffset: unknown;
  values: unknown;
}>;

export type LlamaExecuteTokenOutputTarget = Readonly<{
  opts: ExecuteTokensOptions;
  wantsLogits: boolean;
  output: unknown;
  descOutput: unknown;
  useNativeOutput: boolean;
}>;

export type LlamaExecutePlan = Readonly<{
  stepParams: LlamaExecuteParams;
  hasTokens: boolean;
}>;

export type LlamaPrefillTensorPlan = Readonly<{
  options: LlamaTokenWindowTensorOptions;
  output: unknown;
}>;

export type LlamaWindowPlan = Readonly<{
  window: ReturnType<typeof tokenWindow>;
}>;

export type LlamaSampleWindowPlan = Readonly<{
  window: ReturnType<typeof tokenWindow>;
  sampleOptions: ReturnType<typeof normalizeTokenSampleOptions>;
}>;

export type LlamaGenerationPlan = Readonly<{
  window: ReturnType<typeof tokenWindow>;
  outputTokens: Uint32Array;
}>;

export type LlamaSampleGenerationPlan = Readonly<{
  window: ReturnType<typeof tokenWindow>;
  outputTokens: Uint32Array;
  sampleOptions: ReturnType<typeof normalizeTokenSampleOptions>;
}>;

export type GenericSessionExecutePlan<TStepParams extends StepParamsRecord = StepParamsRecord> = Readonly<{
  stepParams: TStepParams;
  input: unknown;
  output: unknown;
  noOutput: boolean;
}>;

function hasOwn(value: UnknownRecord, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function hasStepParamsDiagnosticCode(error: unknown): error is StepParamsDiagnosticLike {
  return Boolean(error && typeof error === "object" && (error as StepParamsDiagnosticLike).stepParamsDiagnosticCode);
}

function tokenWindowInput(tokens: unknown): TokenWindowInput {
  return tokens as TokenWindowInput;
}

export function genericSessionExplicitInput(
  context: GenericSessionValueContext,
  inputValues: unknown,
  label: string,
  deps: GenericSessionValueDeps,
): Float32Array | unknown | null {
  const desc = context.desc;
  if (inputValues === undefined) {
    if (context.boundInput) return context.hostBoundInput ?? null;
    if (desc.inputLen === 0) return null;
    throw stepParamsValidationError(
      "invalid-input",
      `${label} requires input because the Session has no bound input`,
      { actualLength: 0, expectedLength: desc.inputLen },
    );
  }
  const prepared = deps.prepareHostValue(inputValues);
  if (prepared.data.length !== desc.inputLen) {
    throw stepParamsValidationError(
      "invalid-input",
      `${label} length ${prepared.data.length} does not match Program input length ${desc.inputLen}`,
      { actualLength: prepared.data.length, expectedLength: desc.inputLen },
    );
  }
  try {
    const expectedShape = programInputShape(desc);
    deps.validateHostValueShape(prepared, expectedShape, label);
  } catch (err) {
    throw stepParamsValidationError(
      "invalid-input",
      err && (err as Error).message ? (err as Error).message : String(err),
      { actualShape: prepared.shape.slice(), expectedShape: programInputShape(desc) },
    );
  }
  return prepared.data;
}

export function genericSessionExplicitOutput(
  desc: GenericSessionDesc,
  outputValues: unknown,
  label: string,
  deps: GenericSessionValueDeps,
): Float32Array | null {
  if (outputValues === undefined) return null;
  const prepared = deps.prepareHostValue(outputValues);
  if (prepared.data.length < desc.outputLen) {
    throw stepParamsValidationError(
      "invalid-output",
      `${label} length ${prepared.data.length} is smaller than Program output length ${desc.outputLen}`,
      { actualLength: prepared.data.length, expectedLength: desc.outputLen },
    );
  }
  try {
    const expectedShape = programOutputShape(desc);
    deps.validateHostValueShape(prepared, expectedShape, label, {
      allowFlatCapacity: true,
      allowSameElementShape: true,
    });
  } catch (err) {
    throw stepParamsValidationError(
      "invalid-output",
      err && (err as Error).message ? (err as Error).message : String(err),
      { actualShape: prepared.shape.slice(), expectedShape: programOutputShape(desc) },
    );
  }
  return prepared.data;
}

export function genericSessionStepOutputTarget(
  desc: GenericSessionDesc,
  outputValues: unknown,
  nativeOutput: unknown,
  hostOutput: unknown,
  label: string,
  deps: GenericSessionValueDeps,
): GenericSessionStepOutputTarget {
  const preparedOutput = genericSessionExplicitOutput(desc, outputValues, label, deps);
  const useNativeOutput = nativeOutput !== null && nativeOutput !== undefined && outputValues === undefined;
  const output = preparedOutput
    || (useNativeOutput ? null : hostOutput)
    || (useNativeOutput ? null : new Float32Array(desc.outputLen));
  return Object.freeze({
    output,
    descOutput: useNativeOutput ? null : output,
    useNativeOutput,
  });
}

export function genericSessionStepResult(
  outputLen: number,
  output: unknown,
  nativeOutput: unknown,
  useNativeOutput: boolean,
) {
  if (useNativeOutput) {
    return (nativeOutput as NativeReadbackOutput).readFloat32(outputLen);
  }
  const outputValues = output as Float32Array;
  return outputLen === outputValues.length ? outputValues : outputValues.subarray(0, outputLen);
}

export function genericSessionReadOutputTensorTarget(
  desc: GenericSessionDesc,
  tensorOptions: SessionReadOutputTensorOptions = {},
  outputTarget: (options: TensorOutputOptions, label: string) => unknown,
): SessionReadOutputTensorTarget {
  const options = tensorOptions || {};
  const length = options.length !== undefined ? options.length : desc.outputLen;
  const byteOffset = options.byteOffset !== undefined ? options.byteOffset : 0;
  const values = outputTarget(options, "Session.readOutputTensor output helpers") || new Float32Array(length);
  return Object.freeze({ length, byteOffset, values });
}

export function genericSessionActiveOutputValues(values: Float32Array, length: number): Float32Array {
  return values.length === length ? values : values.subarray(0, length);
}

export function llamaSessionReadOutputTensorTarget(
  vocabSize: number,
  options: SessionReadOutputTensorOptions = {},
  outputTarget: (options: TensorOutputOptions, label: string) => unknown,
): LlamaReadOutputTensorTarget {
  const opts = options || {};
  const length = opts.length !== undefined ? opts.length : vocabSize;
  const byteOffset = opts.byteOffset !== undefined ? opts.byteOffset : 0;
  const values = outputTarget(opts, "LLaMA readOutputTensor output helpers") || new Float32Array(length);
  return Object.freeze({ opts, length, byteOffset, values });
}

export function llamaSessionExecuteTokenOutputTarget(
  vocabSize: number,
  options: ExecuteTokensOptions = {},
  nativeOutput: unknown,
): LlamaExecuteTokenOutputTarget {
  const opts = options || {};
  const wantsLogits = opts.output !== false;
  if (wantsLogits && opts.output !== undefined) {
    validateLogitsOutputBuffer(opts.output, vocabSize, "session.execute");
  }
  const useNativeOutput = wantsLogits && nativeOutput !== null && nativeOutput !== undefined && opts.output === undefined;
  const output = wantsLogits
    ? (opts.output ?? (useNativeOutput ? null : new Float32Array(vocabSize)))
    : null;
  return Object.freeze({
    opts,
    wantsLogits,
    output,
    descOutput: !wantsLogits || useNativeOutput ? null : output,
    useNativeOutput,
  });
}

export function llamaSessionExecuteTokenResult(
  outputLen: number,
  output: unknown,
  nativeOutput: unknown,
  wantsLogits: boolean,
  useNativeOutput: boolean,
) {
  if (!wantsLogits) return undefined;
  if (useNativeOutput) {
    return (nativeOutput as NativeReadbackOutput).readFloat32(outputLen);
  }
  const outputValues = output as Float32Array;
  return outputLen === outputValues.length ? outputValues : outputValues.subarray(0, outputLen);
}

export function llamaSessionTensorOutputTarget(
  options: LlamaLogitsTensorOptions = {},
  outputTarget: (options: TensorOutputOptions, label: string) => unknown,
) {
  return outputTarget(options || {}, "LLaMA Tensor output helpers");
}

export function llamaSessionStepTensorOutputTarget(
  vocabSize: number,
  options: LlamaLogitsTensorOptions = {},
  outputTarget: (options: TensorOutputOptions, label: string) => unknown,
) {
  const output = llamaSessionTensorOutputTarget(options, outputTarget);
  if (output !== undefined) {
    validateLogitsOutputBuffer(output, vocabSize, "session.stepTensor");
  }
  return output;
}

export function requireLlamaSessionLogitsOutput(values: unknown) {
  if (values === undefined) {
    throw new Error("session.executeTensor requires logits output");
  }
  return values;
}

export function llamaSessionExecutePlan(params: unknown, method: string): LlamaExecutePlan {
  const stepParams = params as LlamaExecuteParams;
  const { hasTokens } = validateLlamaTokenStepParams(stepParams, method);
  return Object.freeze({ stepParams, hasTokens });
}

export function llamaSessionExecuteTensorPlan(
  params: LlamaExecuteTensorParams,
  outputTarget: (options: TensorOutputOptions, label: string) => unknown,
): LlamaExecutePlan {
  const stepParams = {
    ...(params || {}),
    output: llamaSessionTensorOutputTarget(params || {}, outputTarget),
  } as LlamaExecuteParams;
  return llamaSessionExecutePlan(stepParams, "executeTensor");
}

export function llamaSessionExecuteIntoPlan(params: unknown): LlamaExecutePlan {
  const plan = llamaSessionExecutePlan(params, "executeInto");
  assertNoInlineOutput(plan.stepParams, "executeInto");
  return plan;
}

export function llamaSessionPrefillTensorPlan(
  options: LlamaTokenWindowTensorOptions = {},
  outputTarget: (options: TensorOutputOptions, label: string) => unknown,
): LlamaPrefillTensorPlan {
  return Object.freeze({
    options: options || {},
    output: llamaSessionTensorOutputTarget(options || {}, outputTarget),
  });
}

export function llamaSessionGenerationOutput(outputTokens: unknown) {
  return tokenOutputArray(outputTokens as Uint32Array);
}

export function llamaSessionGeneratedOutput(maxTokens: number) {
  validateMaxTokens(maxTokens);
  return new Uint32Array(maxTokens);
}

export function llamaSessionSampleOptions(options: TokenSampleOptions = {}) {
  return normalizeTokenSampleOptions(options);
}

export function llamaSessionArgmaxWindow(tokens: unknown, options: TokenWindowOptions = {}) {
  return tokenWindow(tokenWindowInput(tokens), options || {}, "argmax token window");
}

export function llamaSessionSampleWindow(tokens: unknown, options: TokenSampleOptions = {}) {
  return tokenWindow(tokenWindowInput(tokens), options || {}, "sample token window");
}

export function llamaSessionArgmaxExecutionPlan(
  boundOutput: unknown,
  isNativeBuffer: (value: unknown) => boolean,
  tokens: unknown,
  options: TokenWindowOptions = {},
): LlamaWindowPlan {
  requireNativeOutputBinding(boundOutput, isNativeBuffer, "executeTokensArgmax");
  return Object.freeze({ window: llamaSessionArgmaxWindow(tokens, options) });
}

export function llamaSessionSampleExecutionPlan(
  boundOutput: unknown,
  isNativeBuffer: (value: unknown) => boolean,
  tokens: unknown,
  options: TokenSampleOptions = {},
): LlamaSampleWindowPlan {
  requireNativeOutputBinding(boundOutput, isNativeBuffer, "executeTokensSample");
  return Object.freeze({
    window: llamaSessionSampleWindow(tokens, options),
    sampleOptions: llamaSessionSampleOptions(options || {}),
  });
}

export function llamaSessionArgmaxGenerationPlan(tokens: unknown, outputTokens: unknown, options: TokenWindowOptions = {}): LlamaGenerationPlan {
  return Object.freeze({
    window: tokenWindow(tokenWindowInput(tokens), options || {}, "argmax generation prompt window"),
    outputTokens: llamaSessionGenerationOutput(outputTokens),
  });
}

export function llamaSessionSampleGenerationPlan(tokens: unknown, outputTokens: unknown, options: TokenSampleOptions = {}): LlamaSampleGenerationPlan {
  return Object.freeze({
    window: tokenWindow(tokenWindowInput(tokens), options || {}, "sample generation prompt window"),
    outputTokens: llamaSessionGenerationOutput(outputTokens),
    sampleOptions: llamaSessionSampleOptions(options || {}),
  });
}

export function llamaSessionNativeArgmaxGenerationPlan(
  boundOutput: unknown,
  isNativeBuffer: (value: unknown) => boolean,
  tokens: unknown,
  outputTokens: unknown,
  options: TokenWindowOptions = {},
): LlamaGenerationPlan {
  requireNativeOutputBinding(boundOutput, isNativeBuffer, "generateTokensArgmaxInto");
  return llamaSessionArgmaxGenerationPlan(tokens, outputTokens, options);
}

export function llamaSessionNativeSampleGenerationPlan(
  boundOutput: unknown,
  isNativeBuffer: (value: unknown) => boolean,
  tokens: unknown,
  outputTokens: unknown,
  options: TokenSampleOptions = {},
): LlamaSampleGenerationPlan {
  requireNativeOutputBinding(boundOutput, isNativeBuffer, "generateTokensSampleInto");
  return llamaSessionSampleGenerationPlan(tokens, outputTokens, options);
}

export function genericSessionExecutePlan(params: unknown = {}, method = "execute"): GenericSessionExecutePlan {
  const stepParams = validateExecuteStepParams(params, method) as StepParamsRecord;
  return Object.freeze({
    stepParams,
    input: hasOwn(stepParams, "input") ? stepParams.input : undefined,
    output: hasOwn(stepParams, "output") ? stepParams.output : undefined,
    noOutput: stepParams.output === false,
  });
}

export function genericSessionExecuteTensorPlan(params: unknown = {}): GenericSessionExecutePlan<StepParamsRecord & TensorOutputOptions> {
  const plan = genericSessionExecutePlan(params, "executeTensor");
  if (plan.noOutput) {
    throw new Error("session.executeTensor requires output");
  }
  return plan as GenericSessionExecutePlan<StepParamsRecord & TensorOutputOptions>;
}

export function genericSessionExecuteIntoPlan(params: unknown = {}): GenericSessionExecutePlan {
  const plan = genericSessionExecutePlan(params, "executeInto");
  assertNoInlineOutput(plan.stepParams, "executeInto");
  return plan;
}

export function assertGenericSessionNoOutputStepResult(outputLen: number) {
  if (outputLen !== 0) {
    throw new Error(`expected no-output tiny linear step, got ${outputLen} outputs`);
  }
}

export function llamaSessionStepParamsCompatibility(contract: LlamaStepContract, params: unknown) {
  try {
    const stepParams = params as StepParamsRecord;
    const { hasTokens } = validateLlamaTokenStepParams(stepParams, "acceptsStepParams");
    const hasInlineOutput = hasOwn(stepParams, "output");
    let inputTokenCount = 1;
    if (hasInlineOutput) {
      validateLogitsOutputBuffer(stepParams.output, contract.outputLen, "session.acceptsStepParams", { allowFalse: true });
    }
    if (hasTokens) {
      try {
        const window = tokenWindow(stepParams.tokens, stepParams, "session.acceptsStepParams tokens");
        inputTokenCount = window.tokensLen;
        validateTokenWindowFitsContract(contract, window.tokensLen, "session.acceptsStepParams tokens");
      } catch (err) {
        if (hasStepParamsDiagnosticCode(err)) throw err;
        throw stepParamsValidationError("invalid-token-window", err && (err as Error).message ? (err as Error).message : String(err));
      }
    } else {
      try {
        tokenId(stepParams.token);
        validateTokenWindowFitsContract(contract, 1, "session.acceptsStepParams token");
      } catch (err) {
        if (hasStepParamsDiagnosticCode(err)) throw err;
        throw stepParamsValidationError("invalid-token", err && (err as Error).message ? (err as Error).message : String(err));
      }
    }
    return stepParamsCompatibilityResult(contract, null, {
      stateEffect: "advance",
      allocationFree: stepParams.output === false || (hasInlineOutput && stepParams.output instanceof Float32Array),
      inputSource: hasTokens ? "tokens" : "token",
      outputTarget: stepParams.output === false
        ? "none"
        : hasInlineOutput
          ? "inline"
          : contract.defaultOutput,
      inputElementType: "token-u32",
      outputElementType: stepParams.output === false ? "none" : contract.scalarType,
      inputElementLength: inputTokenCount,
      outputElementLength: stepParams.output === false ? 0 : contract.outputLen,
      inputShape: [inputTokenCount],
      outputShape: stepParams.output === false ? [] : contract.outputShape,
      inputByteLength: inputTokenCount * contract.tokenIdBytes,
      outputByteLength: stepParams.output === false ? 0 : contract.outputLen * contract.scalarBytes,
    });
  } catch (err) {
    return stepParamsCompatibilityResult(contract, err);
  }
}

export function genericSessionStepParamsCompatibility(
  context: GenericSessionValueContext,
  contract: GenericSessionStepContract,
  params: unknown = {},
  deps: GenericSessionValueDeps,
) {
  try {
    const stepParams = validateExecuteStepParams(params, "acceptsStepParams");
    const input = hasOwn(stepParams, "input") ? stepParams.input : undefined;
    genericSessionExplicitInput(context, input, "session.acceptsStepParams input", deps);
    const hasInlineOutput = hasOwn(stepParams, "output");
    if (stepParams.output !== false && hasInlineOutput) {
      genericSessionExplicitOutput(context.desc, stepParams.output, "session.acceptsStepParams output", deps);
    }
    return stepParamsCompatibilityResult(contract, null, {
      stateEffect: stepParams.output === false ? "advance" : "none",
      allocationFree: stepParams.output === false || hasInlineOutput || contract.defaultOutput === "bound-host",
      inputSource: hasOwn(stepParams, "input")
        ? "inline"
        : contract.boundInput === "host"
          ? "bound-host"
          : contract.boundInput === "native"
            ? "bound-native"
            : "none",
      outputTarget: stepParams.output === false
        ? "none"
        : hasInlineOutput
          ? "inline"
          : contract.defaultOutput,
      inputElementType: contract.scalarType,
      outputElementType: stepParams.output === false ? "none" : contract.scalarType,
      inputElementLength: contract.inputLen,
      outputElementLength: stepParams.output === false ? 0 : contract.outputLen,
      inputShape: input === undefined && contract.boundInput === "none" ? [] : contract.inputShape,
      outputShape: stepParams.output === false ? [] : contract.outputShape,
      inputByteLength: contract.inputLen * contract.scalarBytes,
      outputByteLength: stepParams.output === false ? 0 : contract.outputLen * contract.scalarBytes,
    });
  } catch (err) {
    return stepParamsCompatibilityResult(contract, err);
  }
}
