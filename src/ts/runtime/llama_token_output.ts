"use strict";

import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";
import {
  tokenSampleAbiFields,
  type NormalizedTokenSampleOptions,
  type TokenSampleAbiFields,
} from "../core/token.js";

export type { NormalizedTokenSampleOptions } from "../core/token.js";

export type LlamaTokenNativeReadback = {
  readFloat32(length: number): Float32Array;
};

export type LlamaTokenStepOutputTarget = Readonly<{
  nativeOutput: LlamaTokenNativeReadback | null;
  output: Float32Array | null;
  descOutput: Float32Array | null;
  useNativeOutput: boolean;
}>;

export type LlamaTokenWindow = Readonly<{
  tokens: Uint32Array;
  tokensLen: number;
}>;

export type LlamaTokenWindowOutputPolicy = 0 | 1;

export type LlamaTokenWindowDescriptorFields = Readonly<{
  tokens: Uint32Array;
  tokensLen: number;
  outputPolicy: LlamaTokenWindowOutputPolicy;
  reserved: 0;
  output: Float32Array | null;
  outputLen: number;
}>;

export type LlamaTokenLogitsDescriptorFields = Readonly<{
  logits: Float32Array | null;
  logitsLen: number;
  reserved: 0;
}>;

export type LlamaTokenWindowInputDescriptorFields = Readonly<{
  tokens: Uint32Array;
  tokensLen: number;
  reserved: 0;
}>;

export type LlamaTokenGenerationDescriptorFields = Readonly<{
  tokens: Uint32Array;
  tokensLen: number;
  outputTokens: Uint32Array;
  outputTokensLen: number;
  reserved: 0;
}>;

export type LlamaTokenSampleLogitsDescriptorFields = Readonly<{
  logits: Float32Array | null;
  logitsLen: number;
}> & TokenSampleAbiFields;

export type LlamaTokenSampleWindowDescriptorFields = Readonly<{
  tokens: Uint32Array;
  tokensLen: number;
}> & TokenSampleAbiFields;

export type LlamaTokenSampleGenerationDescriptorFields = Readonly<{
  tokens: Uint32Array;
  tokensLen: number;
  outputTokens: Uint32Array;
  outputTokensLen: number;
}> & TokenSampleAbiFields;

export const llamaTokenWindowOutputPolicies = Object.freeze({
  none: 0 as LlamaTokenWindowOutputPolicy,
  logits: 1 as LlamaTokenWindowOutputPolicy,
});

export function llamaTokenWindowOutputPolicy(wantsLogits: boolean): LlamaTokenWindowOutputPolicy {
  return wantsLogits ? llamaTokenWindowOutputPolicies.logits : llamaTokenWindowOutputPolicies.none;
}

export function llamaTokenWindowDescriptorFields(
  window: LlamaTokenWindow,
  wantsLogits: boolean,
  output: Float32Array | null,
): LlamaTokenWindowDescriptorFields {
  return Object.freeze({
    tokens: window.tokens,
    tokensLen: window.tokensLen,
    outputPolicy: llamaTokenWindowOutputPolicy(wantsLogits),
    reserved: 0,
    output,
    outputLen: output ? output.length : 0,
  });
}

export function llamaTokenLogitsDescriptorFields(logits: Float32Array | null): LlamaTokenLogitsDescriptorFields {
  return Object.freeze({
    logits,
    logitsLen: logits ? logits.length : 0,
    reserved: 0,
  });
}

export function llamaTokenWindowInputDescriptorFields(window: LlamaTokenWindow): LlamaTokenWindowInputDescriptorFields {
  return Object.freeze({
    tokens: window.tokens,
    tokensLen: window.tokensLen,
    reserved: 0,
  });
}

export function llamaTokenGenerationDescriptorFields(
  window: LlamaTokenWindow,
  outputTokens: Uint32Array,
): LlamaTokenGenerationDescriptorFields {
  return Object.freeze({
    tokens: window.tokens,
    tokensLen: window.tokensLen,
    outputTokens,
    outputTokensLen: outputTokens.length,
    reserved: 0,
  });
}

export function llamaTokenSampleLogitsDescriptorFields(
  logits: Float32Array | null,
  sample: NormalizedTokenSampleOptions,
): LlamaTokenSampleLogitsDescriptorFields {
  const sampleFields = tokenSampleAbiFields(sample);
  return Object.freeze({
    logits,
    logitsLen: logits ? logits.length : 0,
    topK: sampleFields.topK,
    top_k: sampleFields.top_k,
    seed: sampleFields.seed,
    temperature: sampleFields.temperature,
    reserved: sampleFields.reserved,
  });
}

export function llamaTokenSampleWindowDescriptorFields(
  window: LlamaTokenWindow,
  sample: NormalizedTokenSampleOptions,
): LlamaTokenSampleWindowDescriptorFields {
  const sampleFields = tokenSampleAbiFields(sample);
  return Object.freeze({
    tokens: window.tokens,
    tokensLen: window.tokensLen,
    topK: sampleFields.topK,
    top_k: sampleFields.top_k,
    seed: sampleFields.seed,
    temperature: sampleFields.temperature,
    reserved: sampleFields.reserved,
  });
}

export function llamaTokenSampleGenerationDescriptorFields(
  window: LlamaTokenWindow,
  outputTokens: Uint32Array,
  sample: NormalizedTokenSampleOptions,
): LlamaTokenSampleGenerationDescriptorFields {
  const sampleFields = tokenSampleAbiFields(sample);
  return Object.freeze({
    tokens: window.tokens,
    tokensLen: window.tokensLen,
    outputTokens,
    outputTokensLen: outputTokens.length,
    topK: sampleFields.topK,
    top_k: sampleFields.top_k,
    seed: sampleFields.seed,
    temperature: sampleFields.temperature,
    reserved: sampleFields.reserved,
  });
}

export function llamaTokenStepOutputTarget(
  outputValues: Float32Array | undefined,
  boundOutput: unknown,
  vocabSize: number,
  isNativeBuffer: (value: unknown) => value is LlamaTokenNativeReadback,
): LlamaTokenStepOutputTarget {
  const nativeOutput = isNativeBuffer(boundOutput) ? boundOutput : null;
  const useNativeOutput = nativeOutput !== null && outputValues === undefined;
  const output = outputValues ?? (useNativeOutput ? null : new Float32Array(vocabSize));
  return Object.freeze({
    nativeOutput,
    output,
    descOutput: useNativeOutput ? null : output,
    useNativeOutput,
  });
}

export function llamaTokenStepOutputResult(
  outputLen: number,
  target: LlamaTokenStepOutputTarget,
): Float32Array {
  if (target.useNativeOutput) {
    if (target.nativeOutput === null) {
      throw new Error("LLaMA token step expected a native output buffer");
    }
    return target.nativeOutput.readFloat32(outputLen);
  }
  if (target.output === null) {
    throw new Error("LLaMA token step did not produce a host output buffer");
  }
  return outputLen === target.output.length ? target.output : target.output.subarray(0, outputLen);
}

export const llamaTokenOutputManifest = Object.freeze({
  kind: "zgml-llama-token-output",
  ...tsRuntimeManifestPolicy("src/ts/runtime/llama_token_output.ts", "LLaMA Session -> token StepParams -> output"),
});
