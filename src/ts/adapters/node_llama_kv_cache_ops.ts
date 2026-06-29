import {
  createLlamaKvCacheBufferOps,
  type LlamaKvCacheBufferOpsOptions,
  type LlamaKvCacheRequirements,
  type LlamaKvCacheSlot,
  type NativeBufferLike,
} from "../runtime/llama_kv_cache.js";

export type NodeLlamaKvCacheOpsOptions<TKvCache> =
  LlamaKvCacheBufferOpsOptions<NativeBufferLike> & Readonly<{
    createKvCache(k: NativeBufferLike[], v: NativeBufferLike[]): TKvCache;
    llamaKvCacheRequirements(programHandle: unknown): LlamaKvCacheRequirements;
  }>;

export function createNodeLlamaKvCacheOps<TKvCache>(options: NodeLlamaKvCacheOpsOptions<TKvCache>) {
  const {
    createKvCache,
    llamaKvCacheRequirements,
  } = options;
  const {
    createLlamaKvCacheBuffers,
    freeKvCacheBuffers,
  } = createLlamaKvCacheBufferOps(options);

  function createLlamaKvCache(requirements: LlamaKvCacheRequirements, cacheOptions: unknown = {}): TKvCache {
    const buffers = createLlamaKvCacheBuffers(requirements, cacheOptions);
    return createKvCache(buffers.k, buffers.v);
  }

  function createProgramDeviceKvCache(
    programHandle: unknown,
    createBuffer: (kind: "kv-k" | "kv-v") => NativeBufferLike,
  ): TKvCache {
    const requirements = llamaKvCacheRequirements(programHandle);
    const k: NativeBufferLike[] = [];
    const v: NativeBufferLike[] = [];
    try {
      for (let layer = 0; layer < requirements.layers; layer += 1) {
        k.push(createBuffer("kv-k"));
        v.push(createBuffer("kv-v"));
      }
    } catch (err) {
      freeKvCacheBuffers(k, v);
      throw err;
    }
    return createKvCache(k, v);
  }

  function createProgramBufferFactoryKvCache(
    requirements: LlamaKvCacheRequirements,
    cacheOptions: unknown,
    createBuffer: unknown,
  ): TKvCache {
    const buffers = createLlamaKvCacheBuffers(
      requirements,
      cacheOptions,
      typeof createBuffer === "function"
        ? (slot: LlamaKvCacheSlot) => (createBuffer as (slot: LlamaKvCacheSlot) => NativeBufferLike)(slot)
        : undefined,
    );
    return createKvCache(buffers.k, buffers.v);
  }

  return Object.freeze({
    createLlamaKvCache,
    createProgramDeviceKvCache,
    createProgramBufferFactoryKvCache,
  });
}
