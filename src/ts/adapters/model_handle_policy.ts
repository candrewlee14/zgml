type NativeHandle = unknown;

type ModelInstance = Readonly<{
  handle: NativeHandle;
}>;

type ModelConstructor = abstract new (...args: never[]) => object;

type SharedFrontendModelHandleHelpers = Readonly<{
  createModelHandleHelpers<THandle = NativeHandle>(options: {
    isBindableModel(model: unknown): boolean;
    isCompatibleModel(model: unknown): boolean;
    getModelHandle(model: unknown): THandle;
    assertAlive(handle: THandle, label: string): void;
    nullHandle?: THandle | null;
  }): unknown;
}>;

export type AdapterModelHandlePolicyOptions<THandle = NativeHandle, THelpers = unknown> = Readonly<{
  sharedFrontend: SharedFrontendModelHandleHelpers;
  TinyLinearModel: ModelConstructor;
  TinyMlpModel: ModelConstructor;
  TinyLlamaModel: ModelConstructor;
  SmolLM135MModel: ModelConstructor;
  SmolLM2_360MModel: ModelConstructor;
  LlamaModel: ModelConstructor;
  assertAlive(handle: THandle, label: string): void;
  nullHandle?: THandle | null;
}>;

export function createAdapterModelHandlePolicy<THandle = NativeHandle, THelpers = unknown>(
  options: AdapterModelHandlePolicyOptions<THandle, THelpers>,
): THelpers {
  const bindableModelClasses = Object.freeze([
    options.TinyLlamaModel,
    options.SmolLM135MModel,
    options.SmolLM2_360MModel,
    options.LlamaModel,
  ]);
  const compatibleModelClasses = Object.freeze([
    options.TinyLinearModel,
    options.TinyMlpModel,
    ...bindableModelClasses,
  ]);

  function isInstanceOfAny(value: unknown, constructors: readonly ModelConstructor[]): value is ModelInstance {
    return constructors.some((constructor) => value instanceof constructor);
  }

  return options.sharedFrontend.createModelHandleHelpers<THandle>({
    isBindableModel: (model) => isInstanceOfAny(model, bindableModelClasses),
    isCompatibleModel: (model) => isInstanceOfAny(model, compatibleModelClasses),
    getModelHandle: (model) => (model as ModelInstance).handle as THandle,
    assertAlive: options.assertAlive,
    nullHandle: options.nullHandle,
  }) as THelpers;
}
