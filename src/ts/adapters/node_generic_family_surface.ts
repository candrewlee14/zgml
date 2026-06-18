import {
  type GenericFamilySurface,
  type GenericFamilySurfaceOptions,
} from "../runtime/generic_family_surface.js";
import {
  genericModelNativeDesc,
} from "../runtime/generic_model_desc.js";
import {
  createAdapterGenericFamilyModelHandleSurface,
} from "./generic_family_model_handle_surface.js";
import type * as PublicApi from "../public_api.js";

type NodeGenericFamilySymbols = Readonly<{
  modelCreate(desc: unknown, out: unknown): unknown;
}>;

export type NodeGenericFamilySurfaceOptions = Omit<
  GenericFamilySurfaceOptions,
  "createTinyLinearModelHandle" | "createTinyMlpModelHandle"
> & Readonly<{
  tinyLinearKind: number;
  tinyMlpKind: number;
  symbols: NodeGenericFamilySymbols;
  check(status: unknown): unknown;
  handleOut(): unknown;
  readHandle(out: unknown, label?: string): unknown;
}>;

export function createNodeGenericFamilySurface(options: NodeGenericFamilySurfaceOptions): GenericFamilySurface {
  const { tinyLinearKind, tinyMlpKind } = options;

  return createAdapterGenericFamilyModelHandleSurface({
    ...options,
    modelDesc: (desc: PublicApi.TinyLinearDesc | PublicApi.TinyMlpDesc) =>
      genericModelNativeDesc(desc, { tinyLinearKind, tinyMlpKind }),
    modelHandleLabel: "model",
  });
}
