import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";

export type LlamaNativeBufferLike<Handle = unknown> = {
  readonly handle?: Handle;
  readonly handleValue?: Handle;
  readonly byteLength: number;
  assertAlive?(): void;
};

export type LlamaKvCacheRequirements = {
  readonly layers: number;
  readonly kBufferByteLength: number;
  readonly vBufferByteLength: number;
};

export type LlamaKvCacheBufferPair<Buffer> = {
  readonly k: Buffer;
  readonly v: Buffer;
};

export type LlamaKvCacheBindDescriptorFields<Handle> = Readonly<{
  kHandles: readonly Handle[];
  vHandles: readonly Handle[];
  len: number;
}>;

export type LlamaSessionBufferBindOptions = Readonly<{
  output?: unknown;
  kvCache?: unknown;
}>;

export type LlamaSessionBufferBindInspection = Readonly<{
  vocabSize: number;
}>;

export type LlamaSessionBufferBindDescriptorFields<Output, KvCache> = Readonly<{
  output: Output | null;
  outputLen: number;
  kvCache: KvCache | null;
}>;

export type RequireLlamaNativeBuffer<Buffer> = (value: unknown, name: string) => Buffer | null;

export function requireLlamaNativeOutput<Buffer>(
  output: unknown,
  requireNativeBuffer: RequireLlamaNativeBuffer<Buffer>,
): Buffer | null {
  const nativeOutput = requireNativeBuffer(output, "output");
  const maybeLive = nativeOutput as { assertAlive?: unknown } | null;
  if (typeof maybeLive?.assertAlive === "function") {
    maybeLive.assertAlive();
  }
  return nativeOutput;
}

export function normalizeLlamaKvCacheBuffers<Buffer extends { readonly byteLength: number }>(
  kvCache: unknown,
  requirements: LlamaKvCacheRequirements,
  requireNativeBuffer: RequireLlamaNativeBuffer<Buffer>,
): readonly LlamaKvCacheBufferPair<Buffer>[] | null {
  if (kvCache == null) return null;
  const candidate = kvCache as { readonly k?: unknown; readonly v?: unknown };
  const k = candidate.k;
  const v = candidate.v;
  if (!Array.isArray(k) || !Array.isArray(v)) {
    throw new Error("LLaMA kvCache must be { k: NativeBuffer[], v: NativeBuffer[] }");
  }
  if (k.length !== requirements.layers || v.length !== requirements.layers) {
    throw new Error(`LLaMA kvCache must have ${requirements.layers} K and V buffers`);
  }
  const pairs: LlamaKvCacheBufferPair<Buffer>[] = [];
  for (let i = 0; i < requirements.layers; i += 1) {
    const kb = requireNativeBuffer(k[i], `kvCache.k[${i}]`);
    const vb = requireNativeBuffer(v[i], `kvCache.v[${i}]`);
    if (kb === null || vb === null) {
      throw new Error("LLaMA kvCache buffers must be NativeBuffer instances");
    }
    if (kb.byteLength < requirements.kBufferByteLength) {
      throw new Error(`kvCache.k[${i}] is too small: ${kb.byteLength} < ${requirements.kBufferByteLength}`);
    }
    if (vb.byteLength < requirements.vBufferByteLength) {
      throw new Error(`kvCache.v[${i}] is too small: ${vb.byteLength} < ${requirements.vBufferByteLength}`);
    }
    pairs.push(Object.freeze({ k: kb, v: vb }));
  }
  return Object.freeze(pairs);
}

export function llamaKvCacheBindDescriptorFields<Buffer extends { readonly byteLength: number }, Handle>(
  kvCache: unknown,
  requirements: LlamaKvCacheRequirements,
  requireNativeBuffer: RequireLlamaNativeBuffer<Buffer>,
  handleFor: (buffer: Buffer) => Handle,
): LlamaKvCacheBindDescriptorFields<Handle> | null {
  const buffers = normalizeLlamaKvCacheBuffers(kvCache, requirements, requireNativeBuffer);
  if (buffers === null) return null;
  const kHandles: Handle[] = [];
  const vHandles: Handle[] = [];
  for (const pair of buffers) {
    kHandles.push(handleFor(pair.k));
    vHandles.push(handleFor(pair.v));
  }
  return Object.freeze({
    kHandles: Object.freeze(kHandles),
    vHandles: Object.freeze(vHandles),
    len: buffers.length,
  });
}

export function llamaSessionBufferBindDescriptorFields<Buffer, KvCache>(
  bindOptions: LlamaSessionBufferBindOptions,
  inspection: LlamaSessionBufferBindInspection,
  requirements: LlamaKvCacheRequirements,
  requireNativeBuffer: RequireLlamaNativeBuffer<Buffer>,
  kvCacheFieldsFor: (kvCache: unknown, requirements: LlamaKvCacheRequirements) => KvCache | null,
): LlamaSessionBufferBindDescriptorFields<Buffer, KvCache> | null {
  const output = requireNativeBuffer(bindOptions.output, "output");
  const kvCache = kvCacheFieldsFor(bindOptions.kvCache, requirements);
  if (!output && !kvCache) return null;
  return Object.freeze({
    output,
    outputLen: output ? inspection.vocabSize : 0,
    kvCache,
  });
}

export const llamaSessionBindDescriptorManifest = Object.freeze({
  kind: "zgml-llama-session-bind-descriptor",
  ...tsRuntimeManifestPolicy("src/ts/runtime/llama_session_bind_desc.ts", "LLaMA Program -> Session bind descriptor"),
});
