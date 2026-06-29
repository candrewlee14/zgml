import {
  createGenericFamilySurface,
  type GenericFamilySurface,
  type GenericFamilySurfaceOptions,
} from "../runtime/generic_family_surface.js";
import type * as PublicApi from "../public_api.js";

type AdapterGenericFamilySymbols = Readonly<{
  modelCreate(desc: unknown, out: unknown): unknown;
}>;

export type AdapterGenericFamilyModelHandleSurfaceOptions = Omit<
  GenericFamilySurfaceOptions,
  "createTinyLinearModelHandle" | "createTinyMlpModelHandle"
> & Readonly<{
  symbols: AdapterGenericFamilySymbols;
  check(status: unknown): unknown;
  handleOut(): unknown;
  readHandle(out: unknown, label?: string): unknown;
  modelDesc(desc: PublicApi.TinyLinearDesc | PublicApi.TinyMlpDesc): unknown;
  modelHandleLabel?: string;
}>;

export function createAdapterGenericFamilyModelHandleSurface(
  options: AdapterGenericFamilyModelHandleSurfaceOptions,
): GenericFamilySurface {
  const {
    symbols,
    check,
    handleOut,
    readHandle,
    modelDesc,
    modelHandleLabel,
  } = options;

  function createModelHandle(desc: PublicApi.TinyLinearDesc | PublicApi.TinyMlpDesc): unknown {
    const out = handleOut();
    check(symbols.modelCreate(modelDesc(desc), out));
    return readHandle(out, modelHandleLabel);
  }

  return createGenericFamilySurface({
    ...options,
    createTinyLinearModelHandle: createModelHandle,
    createTinyMlpModelHandle: createModelHandle,
  });
}
