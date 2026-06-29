import type {
  TokenSampleOptions as PublicTokenSampleOptions,
  TokenWindowOptions as PublicTokenWindowOptions,
} from "../public_api.js";

export type TokenWindowOptions = PublicTokenWindowOptions;
export type TokenSampleOptions = PublicTokenSampleOptions;

export type NormalizedTokenSampleOptions = Readonly<{
  topK: number;
  top_k: number;
  temperature: number;
  seed: number;
  reserved: 0;
}>;

export type TokenSampleAbiFields = Readonly<{
  topK: number;
  top_k: number;
  temperature: number;
  seed: number;
  reserved: 0;
}>;

type TokenContainer = {
  readonly length: number;
  readonly [index: number]: unknown;
};

export function tokenId(value: unknown, label = "token id"): number {
  const token = Number(value);
  if (!Number.isInteger(token) || token < 0 || token > 0xffffffff) {
    throw new Error(`invalid ${label}: ${value}`);
  }
  return token;
}

export function tokenWindowLength(options: TokenWindowOptions = {}, containerLength: number, label = "token window"): number {
  const length = options?.tokensLen ??
    options?.tokenLength ??
    options?.activeTokenLength ??
    options?.activeTokenCount ??
    containerLength;
  if (!Number.isSafeInteger(length) || length <= 0) {
    throw new Error(`${label} length must be a positive safe integer, got ${length}`);
  }
  if (!Number.isSafeInteger(containerLength) || length > containerLength) {
    throw new Error(`${label} length ${length} exceeds token container length ${containerLength}`);
  }
  return length;
}

export function tokenWindow(data: TokenContainer | null | undefined, options: TokenWindowOptions = {}, label = "token window") {
  if (data == null || !Number.isSafeInteger(data.length)) {
    throw new Error(`${label} must be an array-like token container`);
  }
  const tokensLen = tokenWindowLength(options, data.length, label);
  if (data instanceof Uint32Array) {
    return { tokens: data, tokensLen };
  }
  const tokens = new Uint32Array(tokensLen);
  for (let i = 0; i < tokensLen; i += 1) {
    tokens[i] = tokenId(data[i], `token id at ${i}`);
  }
  return { tokens, tokensLen };
}

export function validateMaxTokens(maxTokens: number): void {
  if (!Number.isSafeInteger(maxTokens) || maxTokens <= 0) {
    throw new Error(`maxTokens must be a positive safe integer, got ${maxTokens}`);
  }
}

export function tokenOutputArray(outputTokens: Uint32Array): Uint32Array {
  if (!(outputTokens instanceof Uint32Array)) {
    throw new Error("generated token output must be a Uint32Array");
  }
  if (outputTokens.length === 0) {
    throw new Error("generated token output must not be empty");
  }
  return outputTokens;
}

export function normalizeTokenSampleOptions(options: TokenSampleOptions = {}): NormalizedTokenSampleOptions {
  const topK = options.topK ?? options.top_k ?? 40;
  const temperature = options.temperature ?? 1.0;
  const seed = options.seed ?? 0;
  if (!Number.isInteger(topK) || topK < 1 || topK > 256) {
    throw new Error(`sample topK must be an integer in [1, 256], got ${topK}`);
  }
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error(`sample seed must be a uint32, got ${seed}`);
  }
  if (typeof temperature !== "number" || !Number.isFinite(temperature) || temperature <= 0) {
    throw new Error(`sample temperature must be finite and > 0, got ${temperature}`);
  }
  return Object.freeze({
    topK,
    top_k: topK,
    temperature,
    seed,
    reserved: 0,
  });
}

export function tokenSampleAbiFields(sample: Pick<NormalizedTokenSampleOptions, "topK" | "top_k" | "temperature" | "seed" | "reserved">): TokenSampleAbiFields {
  const topK = sample.topK ?? sample.top_k;
  return Object.freeze({
    topK,
    top_k: topK,
    temperature: sample.temperature,
    seed: sample.seed,
    reserved: sample.reserved,
  });
}

export function tokenSelectionResultFromAbiRecord(result: { token: unknown; logit: number }) {
  return Object.freeze({
    token: Number(result.token),
    logit: result.logit,
  });
}

export function tokenSelectionResultFromAbiWords(words: BigUint64Array) {
  const view = new DataView(words.buffer, words.byteOffset, words.byteLength);
  return Object.freeze({
    token: view.getUint32(0, true),
    logit: view.getFloat32(4, true),
  });
}

export function tokenGenerateResultFromAbiRecord(
  outputTokens: Uint32Array,
  result: { tokens_generated: unknown; last_token: unknown; last_logit: number },
) {
  const tokensGenerated = Number(result.tokens_generated);
  return Object.freeze({
    tokens: outputTokens.subarray(0, tokensGenerated),
    lastToken: Number(result.last_token),
    lastLogit: result.last_logit,
  });
}

export function tokenGenerateResultFromAbiWords(outputTokens: Uint32Array, words: BigUint64Array) {
  const view = new DataView(words.buffer, words.byteOffset, words.byteLength);
  const tokensGenerated = Number(words[0]);
  return Object.freeze({
    tokens: outputTokens.subarray(0, tokensGenerated),
    lastToken: view.getUint32(8, true),
    lastLogit: view.getFloat32(12, true),
  });
}
