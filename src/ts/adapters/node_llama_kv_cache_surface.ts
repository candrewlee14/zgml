const disposeSymbol: symbol = (Symbol as any).dispose;

export type NodeLlamaKvCacheBuffer = {
  free(): void;
};

export type NodeLlamaKvCacheResources<TBuffer extends NodeLlamaKvCacheBuffer = NodeLlamaKvCacheBuffer> = {
  k: TBuffer[];
  v: TBuffer[];
};

export type NodeLlamaKvCacheSurface<TBuffer extends NodeLlamaKvCacheBuffer = NodeLlamaKvCacheBuffer> =
  NodeLlamaKvCacheResources<TBuffer> & {
    free(): void;
    dispose(): void;
    [disposeSymbol](): void;
  };

export type NodeLlamaKvCacheConstructor<TBuffer extends NodeLlamaKvCacheBuffer = NodeLlamaKvCacheBuffer> = {
  new(k: TBuffer[], v: TBuffer[]): NodeLlamaKvCacheSurface<TBuffer>;
};

export function createNodeLlamaKvCacheClass<
  TBuffer extends NodeLlamaKvCacheBuffer = NodeLlamaKvCacheBuffer,
>(): NodeLlamaKvCacheConstructor<TBuffer> {
  class LlamaKvCache implements NodeLlamaKvCacheSurface<TBuffer> {
    k: TBuffer[];
    v: TBuffer[];

    constructor(k: TBuffer[], v: TBuffer[]) {
      this.k = k;
      this.v = v;
    }

    free(): void {
      for (const buffer of this.k) buffer.free();
      for (const buffer of this.v) buffer.free();
      this.k = [];
      this.v = [];
    }

    dispose(): void {
      this.free();
    }

    [disposeSymbol](): void {
      this.free();
    }
  }

  return LlamaKvCache;
}
