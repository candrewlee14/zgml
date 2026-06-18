import {
  requiredNativeApiExports,
  type RequiredNativeApiExport,
} from "../runtime/native_api_contract.js";

export type AdapterPublicRuntimeExports = Readonly<Record<RequiredNativeApiExport, unknown>>;

export function createAdapterPublicRuntimeExports<const TExports extends AdapterPublicRuntimeExports>(
  exportsSurface: TExports,
  label = "Adapter public runtime exports",
): TExports {
  const missing = requiredNativeApiExports.filter((name) => !(name in exportsSurface));
  if (missing.length > 0) {
    throw new Error(`${label} missing native API contract keys: ${missing.join(", ")}`);
  }
  return exportsSurface;
}
