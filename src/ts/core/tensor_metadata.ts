import type { TensorDevice, TensorDType, TensorInspection } from "../public_api.js";
import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";

export type TensorMetadataTarget = {
  readonly data: { readonly length: number };
  readonly shape: readonly number[];
  readonly requiresGrad: boolean;
  readonly _prev?: readonly unknown[];
};

export type TensorMetadataOpsOptions<TTensor extends TensorMetadataTarget> = Readonly<{
  readonly dtype: (tensor: TTensor) => TensorDType;
  readonly device: (tensor: TTensor) => TensorDevice;
  readonly numel: (tensor: TTensor) => number;
  readonly rowMajorStrides: (shape: readonly number[]) => readonly number[];
  readonly normalizeDim: (dim: number, rank: number, label: string) => number;
}>;

export function createTensorMetadataOps<TTensor extends TensorMetadataTarget>(options: TensorMetadataOpsOptions<TTensor>) {
  const { dtype, device, numel, rowMajorStrides, normalizeDim } = options;

  function isCpu(tensor: TTensor): boolean {
    return device(tensor) === "cpu";
  }

  function is_cpu(tensor: TTensor): boolean {
    return isCpu(tensor);
  }

  function isFloatingPoint(tensor: TTensor): boolean {
    return dtype(tensor) === "f32";
  }

  function is_floating_point(tensor: TTensor): boolean {
    return isFloatingPoint(tensor);
  }

  function isLeaf(tensor: TTensor): boolean {
    return (tensor._prev ?? []).length === 0;
  }

  function is_leaf(tensor: TTensor): boolean {
    return isLeaf(tensor);
  }

  function elementSize(): number {
    return Float32Array.BYTES_PER_ELEMENT;
  }

  function element_size(): number {
    return elementSize();
  }

  function nbytes(tensor: TTensor): number {
    return numel(tensor) * elementSize();
  }

  function stride(tensor: TTensor, dim?: number): readonly number[] | number {
    const strides = rowMajorStrides(tensor.shape);
    if (dim === undefined) return Object.freeze(strides.slice());
    return strides[normalizeDim(dim, tensor.shape.length, "Tensor.stride")];
  }

  function strides(tensor: TTensor): readonly number[] {
    return stride(tensor) as readonly number[];
  }

  function storageOffset(): number {
    return 0;
  }

  function storage_offset(): number {
    return storageOffset();
  }

  function isContiguous(): boolean {
    return true;
  }

  function is_contiguous(): boolean {
    return isContiguous();
  }

  function contiguous(tensor: TTensor): TTensor {
    return tensor;
  }

  function inspect(tensor: TTensor): TensorInspection {
    const inspection: TensorInspection = {
      dtype: dtype(tensor),
      device: device(tensor),
      shape: Object.freeze(tensor.shape.slice()),
      rank: tensor.shape.length,
      length: tensor.data.length,
      strides: Object.freeze(rowMajorStrides(tensor.shape).slice()),
      storageOffset: storageOffset(),
      elementSize: elementSize(),
      byteLength: nbytes(tensor),
      contiguous: isContiguous(),
      requiresGrad: tensor.requiresGrad,
      isLeaf: isLeaf(tensor),
    };
    return Object.freeze(inspection);
  }

  return Object.freeze({
    isCpu,
    is_cpu,
    isFloatingPoint,
    is_floating_point,
    isLeaf,
    is_leaf,
    elementSize,
    element_size,
    nbytes,
    stride,
    strides,
    storageOffset,
    storage_offset,
    isContiguous,
    is_contiguous,
    contiguous,
    inspect,
  });
}

export const tensorMetadataManifest = Object.freeze({
  kind: "zgml-tensor-metadata",
  ...tsRuntimeManifestPolicy("src/ts/core/tensor_metadata.ts", "Tensor metadata -> inspection"),
});
