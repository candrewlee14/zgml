import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";

export type NativeBufferLike = {
  readonly byteLength: number;
  free(): void;
};

export type LlamaKvCacheRequirements = {
  readonly layers: number;
  readonly kBufferByteLength?: number;
  readonly vBufferByteLength?: number;
  readonly [key: string]: unknown;
};

export type LlamaKvCacheSlot = {
  readonly kind: string;
  readonly layer: number;
  readonly byteLength: number;
  readonly [key: string]: unknown;
};

export type LlamaKvCacheLayout = {
  readonly kind: "zgml.llama.kv-cache.layout";
  readonly signature: string;
  readonly layers?: number;
  readonly k: readonly LlamaKvCacheSlot[];
  readonly v: readonly LlamaKvCacheSlot[];
  readonly [key: string]: unknown;
};

export type LlamaKvCacheBufferOpsOptions<TNativeBuffer extends NativeBufferLike> = Readonly<{
  createNativeBuffer(byteLength: number): TNativeBuffer;
  requireNativeBuffer(value: unknown, label: string): TNativeBuffer | null;
  assertLlamaKvCacheResourceByteLength(label: string, buffer: TNativeBuffer | null, byteLength: number): TNativeBuffer;
  llamaKvCacheResourceFactory(options: unknown): ((slot: unknown) => unknown) | Function | null;
  llamaKvCacheResourceSlot(
    requirements: LlamaKvCacheRequirements,
    layout: LlamaKvCacheLayout,
    slot: LlamaKvCacheSlot,
  ): unknown;
  llamaKvCacheLayoutFromRequirements(requirements: any): LlamaKvCacheLayout;
}>;

export function createLlamaKvCacheBufferOps<TNativeBuffer extends NativeBufferLike>(
  options: LlamaKvCacheBufferOpsOptions<TNativeBuffer>,
) {
  const {
    createNativeBuffer,
    requireNativeBuffer,
    assertLlamaKvCacheResourceByteLength,
    llamaKvCacheResourceFactory,
    llamaKvCacheResourceSlot,
    llamaKvCacheLayoutFromRequirements,
  } = options;

  function freeKvCacheBuffers(k: readonly TNativeBuffer[], v: readonly TNativeBuffer[]): void {
    for (const buffer of k) buffer.free();
    for (const buffer of v) buffer.free();
  }

  function createLlamaKvCacheBuffers(
    requirements: LlamaKvCacheRequirements,
    cacheOptions: unknown = {},
    createBuffer?: (slot: LlamaKvCacheSlot) => TNativeBuffer,
  ): { readonly k: TNativeBuffer[]; readonly v: TNativeBuffer[] } {
    const makeResource = llamaKvCacheResourceFactory(cacheOptions) as ((slot: unknown) => unknown) | null;
    const layout = llamaKvCacheLayoutFromRequirements(requirements);
    const makeBuffer = (slot: LlamaKvCacheSlot): TNativeBuffer => {
      if (createBuffer !== undefined) return createBuffer(slot);
      if (makeResource === null) return createNativeBuffer(slot.byteLength);
      const label = `kvCache.${slot.kind}[${slot.layer}]`;
      const buffer = requireNativeBuffer(makeResource(llamaKvCacheResourceSlot(requirements, layout, slot)), label);
      return assertLlamaKvCacheResourceByteLength(label, buffer, slot.byteLength);
    };

    const k: TNativeBuffer[] = [];
    const v: TNativeBuffer[] = [];
    try {
      for (let layer = 0; layer < requirements.layers; layer += 1) {
        k.push(makeBuffer(layout.k[layer] as LlamaKvCacheSlot));
        v.push(makeBuffer(layout.v[layer] as LlamaKvCacheSlot));
      }
    } catch (err) {
      freeKvCacheBuffers(k, v);
      throw err;
    }
    return { k, v };
  }

  return Object.freeze({
    createLlamaKvCacheBuffers,
    freeKvCacheBuffers,
  });
}

export const llamaKvCacheManifest = Object.freeze({
  kind: "zgml-llama-kv-cache-runtime-policy",
  ...tsRuntimeManifestPolicy("src/ts/runtime/llama_kv_cache.ts", "LLaMA Program -> KV-cache resources"),
});
