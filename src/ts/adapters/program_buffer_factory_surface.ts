export type AdapterProgramBufferFactories<
  THandle = unknown,
  TNativeBuffer = unknown,
  TLlamaKvCache = unknown,
> = Readonly<{
  createProgramOutputBuffer(handle: THandle, options?: unknown): TNativeBuffer;
  createProgramBuffer(handle: THandle, kind: unknown, options?: unknown): TNativeBuffer;
  createProgramKvCacheBuffer(handle: THandle, kind: "kv-k" | "kv-v", options?: unknown): TNativeBuffer;
  createProgramNamedBuffer(handle: THandle, kind: unknown, options?: unknown): TNativeBuffer;
  createProgramLlamaKvCache(handle: THandle, options?: unknown): TLlamaKvCache;
}>;

export type AdapterProgramBufferFactorySurfaceOptions<
  THandle = unknown,
  TNativeBuffer = unknown,
  TLlamaKvCache = unknown,
> = Readonly<{
  getProgramBufferFactories(): AdapterProgramBufferFactories<THandle, TNativeBuffer, TLlamaKvCache> | null | undefined;
  uninitializedMessage?: string;
}>;

export function createAdapterProgramBufferFactorySurface<
  THandle = unknown,
  TNativeBuffer = unknown,
  TLlamaKvCache = unknown,
>(
  options: AdapterProgramBufferFactorySurfaceOptions<THandle, TNativeBuffer, TLlamaKvCache>,
) {
  function factories(): AdapterProgramBufferFactories<THandle, TNativeBuffer, TLlamaKvCache> {
    const value = options.getProgramBufferFactories();
    if (!value) throw new Error(options.uninitializedMessage ?? "Program buffer factory surface is not initialized");
    return value;
  }

  function createProgramOutputBuffer(handle: THandle, createOptions: unknown = {}): TNativeBuffer {
    return factories().createProgramOutputBuffer(handle, createOptions);
  }

  function createProgramBuffer(handle: THandle, kind: unknown, createOptions: unknown = {}): TNativeBuffer {
    return factories().createProgramBuffer(handle, kind, createOptions);
  }

  function createProgramKvCacheBuffer(handle: THandle, kind: "kv-k" | "kv-v", createOptions: unknown = {}): TNativeBuffer {
    return factories().createProgramKvCacheBuffer(handle, kind, createOptions);
  }

  function createProgramNamedBuffer(handle: THandle, kind: unknown, createOptions: unknown = {}): TNativeBuffer {
    return factories().createProgramNamedBuffer(handle, kind, createOptions);
  }

  function createProgramLlamaKvCache(handle: THandle, createOptions: unknown = {}): TLlamaKvCache {
    return factories().createProgramLlamaKvCache(handle, createOptions);
  }

  return Object.freeze({
    createProgramOutputBuffer,
    createProgramBuffer,
    createProgramKvCacheBuffer,
    createProgramNamedBuffer,
    createProgramLlamaKvCache,
  });
}
