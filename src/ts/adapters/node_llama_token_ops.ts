import {
  llamaTokenGenerationDescriptorFields,
  llamaTokenLogitsDescriptorFields,
  llamaTokenSampleGenerationDescriptorFields,
  llamaTokenSampleLogitsDescriptorFields,
  llamaTokenSampleWindowDescriptorFields,
  llamaTokenWindowInputDescriptorFields,
  llamaTokenWindowDescriptorFields,
  llamaTokenStepOutputResult,
  llamaTokenStepOutputTarget,
  type LlamaTokenNativeReadback,
  type NormalizedTokenSampleOptions,
} from "../runtime/llama_token_output.js";

export type NativeHandle = unknown;

export type NativeBufferReadback = LlamaTokenNativeReadback;

export type TokenWindow = {
  tokens: Uint32Array;
  tokensLen: number;
};

export type TokenSelectionAbiRecord = {
  token: unknown;
  logit: number;
};

export type TokenGenerateAbiRecord = {
  tokens_generated: unknown;
  last_token: unknown;
  last_logit: number;
};

export type NodeLlamaTokenSymbols = Readonly<{
  sessionStepToken(handle: NativeHandle, desc: Record<string, unknown>, result: Record<string, unknown>): number;
  sessionExecuteTokens(handle: NativeHandle, desc: Record<string, unknown>, result: Record<string, unknown>): number;
  sessionAdvanceToken(handle: NativeHandle, desc: Record<string, unknown>): number;
  sessionArgmaxToken(handle: NativeHandle, desc: Record<string, unknown> | null, result: Record<string, unknown>): number;
  sessionExecuteArgmaxTokens(handle: NativeHandle, desc: Record<string, unknown>, result: Record<string, unknown>): number;
  sessionGenerateArgmaxTokens(handle: NativeHandle, desc: Record<string, unknown>, result: Record<string, unknown>): number;
  sessionSampleToken(handle: NativeHandle, desc: Record<string, unknown>, result: Record<string, unknown>): number;
  sessionExecuteSampleTokens(handle: NativeHandle, desc: Record<string, unknown>, result: Record<string, unknown>): number;
  sessionGenerateSampleTokens(handle: NativeHandle, desc: Record<string, unknown>, result: Record<string, unknown>): number;
}>;

export type NodeLlamaTokenOps = Readonly<{
  sessionStepToken(
    handle: NativeHandle,
    token: unknown,
    outputValues: Float32Array | undefined,
    boundOutput: unknown,
    vocabSize: number,
  ): Float32Array;
  executeLlamaTokenWindow(
    handle: NativeHandle,
    window: TokenWindow,
    wantsLogits: boolean,
    output: Float32Array | null,
  ): number;
  sessionAdvanceToken(handle: NativeHandle, token: unknown): void;
  llamaArgmaxLogits(handle: NativeHandle, data: Float32Array | null): unknown;
  executeLlamaArgmaxWindow(handle: NativeHandle, window: TokenWindow): unknown;
  generateLlamaArgmaxWindow(handle: NativeHandle, window: TokenWindow, outTokens: Uint32Array): unknown;
  llamaSampleLogits(handle: NativeHandle, data: Float32Array | null, sample: NormalizedTokenSampleOptions): unknown;
  executeLlamaSampleWindow(handle: NativeHandle, window: TokenWindow, sample: NormalizedTokenSampleOptions): unknown;
  generateLlamaSampleWindow(
    handle: NativeHandle,
    window: TokenWindow,
    outTokens: Uint32Array,
    sample: NormalizedTokenSampleOptions,
  ): unknown;
}>;

export type NodeLlamaTokenOpsOptions = Readonly<{
  symbols: NodeLlamaTokenSymbols;
  check(code: number): void;
  tokenId(value: unknown): number;
  isNativeBuffer(value: unknown): value is NativeBufferReadback;
  tokenSelectionResultFromAbiRecord(result: TokenSelectionAbiRecord): unknown;
  tokenGenerateResultFromAbiRecord(outputTokens: Uint32Array, result: TokenGenerateAbiRecord): unknown;
}>;

export function createNodeLlamaTokenOps(options: NodeLlamaTokenOpsOptions): NodeLlamaTokenOps {
  const {
    symbols,
    check,
    tokenId,
    isNativeBuffer,
    tokenSelectionResultFromAbiRecord,
    tokenGenerateResultFromAbiRecord,
  } = options;

  return {
    sessionStepToken(handle, token, outputValues, boundOutput, vocabSize) {
      const target = llamaTokenStepOutputTarget(outputValues, boundOutput, vocabSize, isNativeBuffer);
      const result: Record<string, unknown> = {};
      check(symbols.sessionStepToken(handle, {
        token: tokenId(token),
        output: target.descOutput,
        output_len: target.descOutput ? target.descOutput.length : 0,
      }, result));
      const outputLen = Number(result.output_len);
      return llamaTokenStepOutputResult(outputLen, target);
    },

    executeLlamaTokenWindow(handle, window, wantsLogits, output) {
      const fields = llamaTokenWindowDescriptorFields(window, wantsLogits, output);
      const result: Record<string, unknown> = {};
      check(symbols.sessionExecuteTokens(handle, {
        tokens: fields.tokens,
        tokens_len: fields.tokensLen,
        output_policy: fields.outputPolicy,
        reserved: fields.reserved,
        output: fields.output,
        output_len: fields.outputLen,
      }, result));
      return Number(result.output_len);
    },

    sessionAdvanceToken(handle, token) {
      check(symbols.sessionAdvanceToken(handle, {
        token: tokenId(token),
      }));
    },

    llamaArgmaxLogits(handle, data) {
      const result = {} as TokenSelectionAbiRecord;
      const fields = llamaTokenLogitsDescriptorFields(data);
      check(symbols.sessionArgmaxToken(handle, fields.logits ? {
        logits: fields.logits,
        logits_len: fields.logitsLen,
        reserved: fields.reserved,
      } : null, result as Record<string, unknown>));
      return tokenSelectionResultFromAbiRecord(result);
    },

    executeLlamaArgmaxWindow(handle, window) {
      const result = {} as TokenSelectionAbiRecord;
      const fields = llamaTokenWindowInputDescriptorFields(window);
      check(symbols.sessionExecuteArgmaxTokens(handle, {
        tokens: fields.tokens,
        tokens_len: fields.tokensLen,
        reserved: fields.reserved,
      }, result as Record<string, unknown>));
      return tokenSelectionResultFromAbiRecord(result);
    },

    generateLlamaArgmaxWindow(handle, window, outTokens) {
      const result = {} as TokenGenerateAbiRecord;
      const fields = llamaTokenGenerationDescriptorFields(window, outTokens);
      check(symbols.sessionGenerateArgmaxTokens(handle, {
        tokens: fields.tokens,
        tokens_len: fields.tokensLen,
        output_tokens: fields.outputTokens,
        output_tokens_len: fields.outputTokensLen,
        reserved: fields.reserved,
      }, result as Record<string, unknown>));
      return tokenGenerateResultFromAbiRecord(outTokens, result);
    },

    llamaSampleLogits(handle, data, sample) {
      const result = {} as TokenSelectionAbiRecord;
      const fields = llamaTokenSampleLogitsDescriptorFields(data, sample);
      check(symbols.sessionSampleToken(handle, {
        logits: fields.logits,
        logits_len: fields.logitsLen,
        top_k: fields.top_k,
        seed: fields.seed,
        temperature: fields.temperature,
        reserved: fields.reserved,
      }, result as Record<string, unknown>));
      return tokenSelectionResultFromAbiRecord(result);
    },

    executeLlamaSampleWindow(handle, window, sample) {
      const result = {} as TokenSelectionAbiRecord;
      const fields = llamaTokenSampleWindowDescriptorFields(window, sample);
      check(symbols.sessionExecuteSampleTokens(handle, {
        tokens: fields.tokens,
        tokens_len: fields.tokensLen,
        top_k: fields.top_k,
        seed: fields.seed,
        temperature: fields.temperature,
        reserved: fields.reserved,
      }, result as Record<string, unknown>));
      return tokenSelectionResultFromAbiRecord(result);
    },

    generateLlamaSampleWindow(handle, window, outTokens, sample) {
      const result = {} as TokenGenerateAbiRecord;
      const fields = llamaTokenSampleGenerationDescriptorFields(window, outTokens, sample);
      check(symbols.sessionGenerateSampleTokens(handle, {
        tokens: fields.tokens,
        tokens_len: fields.tokensLen,
        output_tokens: fields.outputTokens,
        output_tokens_len: fields.outputTokensLen,
        top_k: fields.top_k,
        seed: fields.seed,
        temperature: fields.temperature,
        reserved: fields.reserved,
      }, result as Record<string, unknown>));
      return tokenGenerateResultFromAbiRecord(outTokens, result);
    },
  };
}
