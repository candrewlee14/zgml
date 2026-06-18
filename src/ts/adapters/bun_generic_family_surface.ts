import {
  type GenericFamilySurface,
  type GenericFamilySurfaceOptions,
} from "../runtime/generic_family_surface.js";
import {
  createAdapterGenericFamilyModelHandleSurface,
} from "./generic_family_model_handle_surface.js";
import type * as PublicApi from "../public_api.js";

type BunGenericFamilySymbols = Readonly<{
  modelCreate(desc: unknown, out: unknown): unknown;
}>;

export type BunGenericFamilySurfaceOptions = Omit<
  GenericFamilySurfaceOptions,
  "createTinyLinearModelHandle" | "createTinyMlpModelHandle"
> & Readonly<{
  symbols: BunGenericFamilySymbols;
  check(status: unknown): unknown;
  handleOut(): unknown;
  readHandle(out: unknown): unknown;
  modelDesc(desc: PublicApi.TinyLinearDesc | PublicApi.TinyMlpDesc): unknown;
}>;

export function createBunGenericFamilySurface(options: BunGenericFamilySurfaceOptions): GenericFamilySurface {
  return createAdapterGenericFamilyModelHandleSurface(options);
}
