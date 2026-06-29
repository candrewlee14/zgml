import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";

export type TensorDataLike = {
  readonly data: ArrayLike<number>;
};

export type TensorDataConstructor<TTensor extends TensorDataLike = TensorDataLike> =
  abstract new (...args: never[]) => TTensor;

export type NumericIndexValues = Uint32Array | Int32Array | Float32Array | readonly number[];

export function indexValuesFromTensorOrArray(
  data: unknown,
  name: string,
  TensorCtor: TensorDataConstructor,
): ArrayLike<number> {
  if (data instanceof TensorCtor) return data.data;
  if (
    Array.isArray(data) ||
    data instanceof Uint32Array ||
    data instanceof Int32Array ||
    data instanceof Float32Array
  ) {
    return data;
  }
  throw new Error(`${name} must be an array or numeric typed array`);
}

export const indexValuesManifest = Object.freeze({
  kind: "zgml-index-values-core-policy",
  ...tsRuntimeManifestPolicy("src/ts/core/index_values.ts", "Tensor/index values -> embedding/class-index inputs"),
});
