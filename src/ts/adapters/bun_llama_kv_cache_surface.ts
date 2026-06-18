import type {
  NativeBufferLike,
  LlamaKvCacheRequirements,
  LlamaKvCacheSlot,
} from "../runtime/llama_kv_cache.js";
import type * as PublicApi from "../public_api.js";

export type BunLlamaKvCacheCreateBuffers<TNativeBuffer extends NativeBufferLike> = (
  requirements: LlamaKvCacheRequirements,
  options?: PublicApi.LlamaKvCacheCreateOptions,
  createBuffer?: (slot: LlamaKvCacheSlot) => TNativeBuffer,
) => { readonly k: TNativeBuffer[]; readonly v: TNativeBuffer[] };

export type BunLlamaKvCacheFreeBuffers<TNativeBuffer extends NativeBufferLike> = (
  k: readonly TNativeBuffer[],
  v: readonly TNativeBuffer[],
) => void;

export type BunLlamaKvCacheSurfaceOptions<TNativeBuffer extends NativeBufferLike> = Readonly<{
  createLlamaKvCacheBuffers: BunLlamaKvCacheCreateBuffers<TNativeBuffer>;
  freeKvCacheBuffers: BunLlamaKvCacheFreeBuffers<TNativeBuffer>;
}>;

export type BunLlamaKvCache<TNativeBuffer extends NativeBufferLike> =
  Readonly<{
    k: TNativeBuffer[];
    v: TNativeBuffer[];
    free(): void;
    dispose(): void;
    [Symbol.dispose](): void;
  }>;

export type BunLlamaKvCacheConstructor<TNativeBuffer extends NativeBufferLike> = {
  new(
    requirements: PublicApi.LlamaKvCacheRequirements,
    options?: PublicApi.LlamaKvCacheCreateOptions,
    createBuffer?: (slot: PublicApi.LlamaKvCacheLayoutSlot) => TNativeBuffer,
  ): BunLlamaKvCache<TNativeBuffer>;
};

export function createBunLlamaKvCacheClass<TNativeBuffer extends NativeBufferLike>(
  options: BunLlamaKvCacheSurfaceOptions<TNativeBuffer>,
): BunLlamaKvCacheConstructor<TNativeBuffer> {
  class LlamaKvCache implements BunLlamaKvCache<TNativeBuffer> {
    public k: TNativeBuffer[];
    public v: TNativeBuffer[];

    constructor(
      requirements: PublicApi.LlamaKvCacheRequirements,
      cacheOptions: PublicApi.LlamaKvCacheCreateOptions = {},
      createBuffer?: (slot: PublicApi.LlamaKvCacheLayoutSlot) => TNativeBuffer,
    ) {
      const buffers = options.createLlamaKvCacheBuffers(
        requirements,
        cacheOptions,
        createBuffer as ((slot: LlamaKvCacheSlot) => TNativeBuffer) | undefined,
      );
      this.k = buffers.k;
      this.v = buffers.v;
    }

    free(): void {
      options.freeKvCacheBuffers(this.k, this.v);
      this.k = [];
      this.v = [];
    }

    dispose(): void {
      this.free();
    }

    [Symbol.dispose](): void {
      this.free();
    }
  }

  return LlamaKvCache;
}
