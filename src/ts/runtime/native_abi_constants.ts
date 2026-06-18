import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";

type NumericAbiMap = Readonly<Record<string, number>>;

export function adapterModelKindAliases(modelKinds: NumericAbiMap) {
  return Object.freeze({
    autoKind: modelKinds.auto,
    tinyLinearKind: modelKinds.tinyLinear,
    tinyLlamaKind: modelKinds.tinyLlama,
    smollm135mKind: modelKinds.smollm135m,
    tinyLlama2LayerKind: modelKinds.tinyLlama2Layer,
    tinyMlpKind: modelKinds.tinyMlp,
    moduleKind: modelKinds.module,
  });
}

export function adapterBufferStorageAliases(bufferStorageIds: NumericAbiMap) {
  return Object.freeze({
    bufferStorageExternalResource: bufferStorageIds.externalResource,
  });
}

export const nativeAbiConstantsManifest = Object.freeze({
  kind: "zgml-native-abi-constants",
  ...tsRuntimeManifestPolicy("src/ts/runtime/native_abi_constants.ts", "Native ABI constants -> host adapters"),
});
