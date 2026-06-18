type NativeBufferFactory<TBuffer> = Readonly<{
  create(byteLength: number): TBuffer;
}>;

type LlamaKvCacheConstructor<TBuffer, TKvCache> = new (k: TBuffer[], v: TBuffer[]) => TKvCache;

export type NodeLlamaKvCacheFactorySurfaceOptions<TBuffer, TKvCache> = Readonly<{
  getNativeBufferClass(): NativeBufferFactory<TBuffer>;
  getKvCacheClass(): LlamaKvCacheConstructor<TBuffer, TKvCache>;
}>;

export function createNodeLlamaKvCacheFactorySurface<TBuffer, TKvCache>(
  options: NodeLlamaKvCacheFactorySurfaceOptions<TBuffer, TKvCache>,
) {
  function createNativeBuffer(byteLength: number): TBuffer {
    return options.getNativeBufferClass().create(byteLength);
  }

  function createKvCache(k: TBuffer[], v: TBuffer[]): TKvCache {
    const LlamaKvCache = options.getKvCacheClass();
    return new LlamaKvCache(k, v);
  }

  return Object.freeze({
    createNativeBuffer,
    createKvCache,
  });
}
