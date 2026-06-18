import {
  createLlamaKvCacheBufferOps,
  type LlamaKvCacheBufferOpsOptions,
  type NativeBufferLike,
} from "../runtime/llama_kv_cache.js";

export type BunLlamaKvCacheOpsOptions<TNativeBuffer extends NativeBufferLike> =
  LlamaKvCacheBufferOpsOptions<TNativeBuffer>;

export function createBunLlamaKvCacheOps<TNativeBuffer extends NativeBufferLike>(
  options: BunLlamaKvCacheOpsOptions<TNativeBuffer>,
) {
  return createLlamaKvCacheBufferOps(options);
}
