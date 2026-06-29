import {
  setPtr,
  setUSize,
} from "./bun_abi_words.js";
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
  type LlamaTokenGenerationDescriptorFields,
  type LlamaTokenLogitsDescriptorFields,
  type LlamaTokenSampleGenerationDescriptorFields,
  type LlamaTokenSampleLogitsDescriptorFields,
  type LlamaTokenSampleWindowDescriptorFields,
  type LlamaTokenWindowInputDescriptorFields,
  type LlamaTokenWindowDescriptorFields,
  type NormalizedTokenSampleOptions,
} from "../runtime/llama_token_output.js";

type NativeHandle = number;

type NativeBufferReadback = LlamaTokenNativeReadback;

type TokenWindow = {
  tokens: Uint32Array;
  tokensLen: number;
};

type TokenSelectionResult = Readonly<{
  token: number;
  logit: number;
}>;

type TokenGenerateResult = Readonly<{
  tokens: Uint32Array;
  lastToken: number;
  lastLogit: number;
}>;

type BunLlamaTokenSymbols = Readonly<{
  sessionStepToken(handle: NativeHandle, desc: BigUint64Array, result: BigUint64Array): number;
  sessionAdvanceToken(handle: NativeHandle, desc: BigUint64Array): number;
  sessionExecuteTokens(handle: NativeHandle, desc: BigUint64Array, result: BigUint64Array): number;
  sessionArgmaxToken(handle: NativeHandle, desc: BigUint64Array | NativeHandle, result: BigUint64Array): number;
  sessionExecuteArgmaxTokens(handle: NativeHandle, desc: BigUint64Array, result: BigUint64Array): number;
  sessionGenerateArgmaxTokens(handle: NativeHandle, desc: BigUint64Array, result: BigUint64Array): number;
  sessionSampleToken(handle: NativeHandle, desc: BigUint64Array, result: BigUint64Array): number;
  sessionExecuteSampleTokens(handle: NativeHandle, desc: BigUint64Array, result: BigUint64Array): number;
  sessionGenerateSampleTokens(handle: NativeHandle, desc: BigUint64Array, result: BigUint64Array): number;
}>;

export type BunLlamaTokenOpsOptions = Readonly<{
  symbols: BunLlamaTokenSymbols;
  check(code: number): void;
  pointerFor(value: Float32Array | Uint32Array): NativeHandle;
  tokenId(value: unknown): number;
  isNativeBuffer(value: unknown): value is NativeBufferReadback;
  tokenSelectionResultFromAbiWords(words: BigUint64Array): TokenSelectionResult;
  tokenGenerateResultFromAbiWords(outputTokens: Uint32Array, words: BigUint64Array): TokenGenerateResult;
}>;

export function createBunLlamaTokenOps(options: BunLlamaTokenOpsOptions) {
  const {
    symbols,
    check,
    pointerFor,
    tokenId,
    isNativeBuffer,
    tokenSelectionResultFromAbiWords,
    tokenGenerateResultFromAbiWords,
  } = options;

  function tokenExecuteDesc(fields: LlamaTokenWindowDescriptorFields): BigUint64Array {
    const buf = new BigUint64Array(5);
    const view = new DataView(buf.buffer);
    setPtr(view, 0, pointerFor(fields.tokens));
    setUSize(view, 8, fields.tokensLen);
    view.setUint32(16, fields.outputPolicy, true);
    view.setUint32(20, fields.reserved, true);
    setPtr(view, 24, fields.output ? pointerFor(fields.output) : 0);
    setUSize(view, 32, fields.outputLen);
    return buf;
  }

  function tokenStepDesc(token: number, output: Float32Array | null): BigUint64Array {
    const buf = new BigUint64Array(3);
    const view = new DataView(buf.buffer);
    view.setUint32(0, tokenId(token), true);
    setPtr(view, 8, output ? pointerFor(output) : 0);
    setUSize(view, 16, output ? output.length : 0);
    return buf;
  }

  function tokenAdvanceDesc(token: number): BigUint64Array {
    const buf = new BigUint64Array(1);
    const view = new DataView(buf.buffer);
    view.setUint32(0, tokenId(token), true);
    return buf;
  }

  function tokenArgmaxDesc(fields: LlamaTokenLogitsDescriptorFields): NativeHandle | BigUint64Array {
    if (!fields.logits) return 0;
    const buf = new BigUint64Array(3);
    const view = new DataView(buf.buffer);
    setPtr(view, 0, pointerFor(fields.logits));
    setUSize(view, 8, fields.logitsLen);
    view.setUint32(16, fields.reserved, true);
    return buf;
  }

  function tokenExecuteArgmaxDesc(fields: LlamaTokenWindowInputDescriptorFields): BigUint64Array {
    const buf = new BigUint64Array(3);
    const view = new DataView(buf.buffer);
    setPtr(view, 0, pointerFor(fields.tokens));
    setUSize(view, 8, fields.tokensLen);
    view.setUint32(16, fields.reserved, true);
    return buf;
  }

  function tokenGenerateArgmaxDesc(fields: LlamaTokenGenerationDescriptorFields): BigUint64Array {
    const buf = new BigUint64Array(5);
    const view = new DataView(buf.buffer);
    setPtr(view, 0, pointerFor(fields.tokens));
    setUSize(view, 8, fields.tokensLen);
    setPtr(view, 16, pointerFor(fields.outputTokens));
    setUSize(view, 24, fields.outputTokensLen);
    view.setUint32(32, fields.reserved, true);
    return buf;
  }

  function tokenSampleDesc(fields: LlamaTokenSampleLogitsDescriptorFields): BigUint64Array {
    const buf = new BigUint64Array(4);
    const view = new DataView(buf.buffer);
    setPtr(view, 0, fields.logits ? pointerFor(fields.logits) : 0);
    setUSize(view, 8, fields.logitsLen);
    view.setUint32(16, fields.topK, true);
    view.setUint32(20, fields.seed, true);
    view.setFloat32(24, fields.temperature, true);
    view.setUint32(28, fields.reserved, true);
    return buf;
  }

  function tokenExecuteSampleDesc(fields: LlamaTokenSampleWindowDescriptorFields): BigUint64Array {
    const buf = new BigUint64Array(4);
    const view = new DataView(buf.buffer);
    setPtr(view, 0, pointerFor(fields.tokens));
    setUSize(view, 8, fields.tokensLen);
    view.setUint32(16, fields.topK, true);
    view.setUint32(20, fields.seed, true);
    view.setFloat32(24, fields.temperature, true);
    view.setUint32(28, fields.reserved, true);
    return buf;
  }

  function tokenGenerateSampleDesc(fields: LlamaTokenSampleGenerationDescriptorFields): BigUint64Array {
    const buf = new BigUint64Array(6);
    const view = new DataView(buf.buffer);
    setPtr(view, 0, pointerFor(fields.tokens));
    setUSize(view, 8, fields.tokensLen);
    setPtr(view, 16, pointerFor(fields.outputTokens));
    setUSize(view, 24, fields.outputTokensLen);
    view.setUint32(32, fields.topK, true);
    view.setUint32(36, fields.seed, true);
    view.setFloat32(40, fields.temperature, true);
    view.setUint32(44, fields.reserved, true);
    return buf;
  }

  function sessionStepToken(
    handle: NativeHandle,
    token: number,
    outputValues: Float32Array | undefined,
    boundOutput: unknown,
    vocabSize: number,
  ): Float32Array {
    const target = llamaTokenStepOutputTarget(outputValues, boundOutput, vocabSize, isNativeBuffer);
    const result = new BigUint64Array(1);
    check(symbols.sessionStepToken(handle, tokenStepDesc(token, target.descOutput), result));
    const outputLen = Number(result[0]);
    return llamaTokenStepOutputResult(outputLen, target);
  }

  function executeLlamaTokenWindow(
    handle: NativeHandle,
    window: TokenWindow,
    wantsLogits: boolean,
    output: Float32Array | null,
  ): number {
    const result = new BigUint64Array(1);
    check(symbols.sessionExecuteTokens(
      handle,
      tokenExecuteDesc(llamaTokenWindowDescriptorFields(window, wantsLogits, output)),
      result,
    ));
    return Number(result[0]);
  }

  function sessionAdvanceToken(handle: NativeHandle, token: number): void {
    check(symbols.sessionAdvanceToken(handle, tokenAdvanceDesc(token)));
  }

  function llamaArgmaxLogits(handle: NativeHandle, data: Float32Array | null): TokenSelectionResult {
    const result = new BigUint64Array(1);
    check(symbols.sessionArgmaxToken(handle, tokenArgmaxDesc(llamaTokenLogitsDescriptorFields(data)), result));
    return tokenSelectionResultFromAbiWords(result);
  }

  function executeLlamaArgmaxWindow(handle: NativeHandle, window: TokenWindow): TokenSelectionResult {
    const result = new BigUint64Array(1);
    check(symbols.sessionExecuteArgmaxTokens(handle, tokenExecuteArgmaxDesc(llamaTokenWindowInputDescriptorFields(window)), result));
    return tokenSelectionResultFromAbiWords(result);
  }

  function generateLlamaArgmaxWindow(
    handle: NativeHandle,
    window: TokenWindow,
    outTokens: Uint32Array,
  ): TokenGenerateResult {
    const result = new BigUint64Array(2);
    check(symbols.sessionGenerateArgmaxTokens(
      handle,
      tokenGenerateArgmaxDesc(llamaTokenGenerationDescriptorFields(window, outTokens)),
      result,
    ));
    return tokenGenerateResultFromAbiWords(outTokens, result);
  }

  function llamaSampleLogits(
    handle: NativeHandle,
    data: Float32Array | null,
    sample: NormalizedTokenSampleOptions,
  ): TokenSelectionResult {
    const result = new BigUint64Array(1);
    check(symbols.sessionSampleToken(handle, tokenSampleDesc(llamaTokenSampleLogitsDescriptorFields(data, sample)), result));
    return tokenSelectionResultFromAbiWords(result);
  }

  function executeLlamaSampleWindow(
    handle: NativeHandle,
    window: TokenWindow,
    sample: NormalizedTokenSampleOptions,
  ): TokenSelectionResult {
    const result = new BigUint64Array(1);
    check(symbols.sessionExecuteSampleTokens(handle, tokenExecuteSampleDesc(llamaTokenSampleWindowDescriptorFields(window, sample)), result));
    return tokenSelectionResultFromAbiWords(result);
  }

  function generateLlamaSampleWindow(
    handle: NativeHandle,
    window: TokenWindow,
    outTokens: Uint32Array,
    sample: NormalizedTokenSampleOptions,
  ): TokenGenerateResult {
    const result = new BigUint64Array(2);
    check(symbols.sessionGenerateSampleTokens(
      handle,
      tokenGenerateSampleDesc(llamaTokenSampleGenerationDescriptorFields(window, outTokens, sample)),
      result,
    ));
    return tokenGenerateResultFromAbiWords(outTokens, result);
  }

  return Object.freeze({
    sessionStepToken,
    executeLlamaTokenWindow,
    sessionAdvanceToken,
    llamaArgmaxLogits,
    executeLlamaArgmaxWindow,
    generateLlamaArgmaxWindow,
    llamaSampleLogits,
    executeLlamaSampleWindow,
    generateLlamaSampleWindow,
  });
}
