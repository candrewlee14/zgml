type TensorLikeConstructOptions = Readonly<Record<string, unknown> & {
  requiresGrad?: boolean;
}>;

type TensorPreparedData = Readonly<{
  data: Float32Array;
  shape: number[];
}>;

type TensorLikeClass = new (
  data: Float32Array,
  shape: readonly number[],
  options?: TensorLikeConstructOptions,
) => TensorLikeInstance;

type TensorLikeInstance = {
  data: Float32Array;
  shape: readonly number[];
  length: number;
  requiresGrad: boolean;
  grad: Float32Array | null;
};

export type TensorDataHelpersOptions = {
  getTensorClass?: () => TensorLikeClass;
  Tensor?: TensorLikeClass;
};

export function createTensorDataHelpers(options: TensorDataHelpersOptions = {}) {
  const getTensorClass = typeof options.getTensorClass === "function"
    ? options.getTensorClass
    : () => options.Tensor;

  function tensorClass(): TensorLikeClass {
    const TensorClass = getTensorClass();
    if (typeof TensorClass !== "function") {
      throw new Error("tensor data helpers require a Tensor constructor");
    }
    return TensorClass;
  }

  function flattenArrayTensorInput(value: readonly unknown[], label = "tensor data") {
    const values: number[] = [];

    function visit(node: unknown, path: string): number[] {
      if (!Array.isArray(node)) {
        if (typeof node !== "number") {
          throw new Error(`${label}${path} must contain numbers, got ${typeof node}`);
        }
        values.push(node);
        return [];
      }
      if (node.length === 0) {
        throw new Error(`${label}${path} must not contain empty arrays`);
      }
      let childShape: number[] | null = null;
      for (let i = 0; i < node.length; i += 1) {
        const nextPath = `${path}[${i}]`;
        const shape = visit(node[i], nextPath);
        if (childShape === null) {
          childShape = shape;
        } else if (childShape.length !== shape.length || childShape.some((dim, dimIndex) => dim !== shape[dimIndex])) {
          throw new Error(`${label} must be rectangular; shape at ${nextPath} differs from earlier entries`);
        }
      }
      return [node.length, ...(childShape ?? [])];
    }

    const shape = visit(value, "");
    return {
      data: Float32Array.from(values),
      shape,
    };
  }

  function prepareF32(data: unknown): TensorPreparedData {
    if (typeof data === "number") return { data: Float32Array.of(data), shape: [1] };
    if (data instanceof Float32Array) return { data, shape: [data.length] };
    if (Array.isArray(data)) return flattenArrayTensorInput(data);
    const out = Float32Array.from(data as ArrayLike<number>);
    return { data: out, shape: [out.length] };
  }

  function rawF32(data: unknown): Float32Array {
    return prepareF32(data).data;
  }

  function f32(data: unknown): Float32Array {
    return data instanceof tensorClass() ? data.data : rawF32(data);
  }

  function byteView(data: unknown, name = "bytes"): Uint8Array {
    if (data instanceof Uint8Array) return data;
    if (ArrayBuffer.isView(data)) {
      return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    }
    if (data instanceof ArrayBuffer) return new Uint8Array(data);
    if (Array.isArray(data)) return Uint8Array.from(data);
    throw new Error(`${name} must be a Uint8Array, ArrayBuffer, typed array, DataView, or number array`);
  }

  function addTensorGrad(tensor: TensorLikeInstance, grad: Float32Array): void {
    if (!tensor.requiresGrad) return;
    if (!tensor.grad) tensor.grad = new Float32Array(tensor.length);
    for (let i = 0; i < grad.length; i += 1) tensor.grad[i] += grad[i];
  }

  function scalarTensor(value: number, requiresGrad = false): TensorLikeInstance {
    return new (tensorClass())(Float32Array.of(value), [1], { requiresGrad });
  }

  function requirePositiveInteger(value: number, name: string): number {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new Error(`${name} must be a positive safe integer, got ${value}`);
    }
    return value;
  }

  function zerosF32(length: number): Float32Array {
    return new Float32Array(length);
  }

  function sameShape(actual: readonly number[], expected: readonly number[]): boolean {
    return actual.length === expected.length && actual.every((dim, index) => dim === expected[index]);
  }

  function preparedTensorLike(data: unknown): TensorPreparedData {
    if (data instanceof tensorClass()) {
      return {
        data: data.data,
        shape: Array.isArray(data.shape) ? Array.from(data.shape) : [data.data.length],
      };
    }
    return prepareF32(data);
  }

  function f32WithLength(data: unknown, length: number, name: string, expectedShape?: readonly number[]): Float32Array {
    const prepared = preparedTensorLike(data);
    const out = prepared.data;
    if (out.length !== length) {
      throw new Error(`${name} length must be ${length}, got ${out.length}`);
    }
    if (expectedShape !== undefined && !sameShape(prepared.shape, expectedShape) && !sameShape(prepared.shape, [length])) {
      throw new Error(`${name} shape must be [${expectedShape.join(", ")}] or flat length ${length}, got [${prepared.shape.join(", ")}]`);
    }
    return out;
  }

  function defaultedF32(
    data: unknown,
    length: number,
    name: string,
    fallback: (length: number) => Float32Array,
    expectedShape?: readonly number[],
  ): Float32Array {
    return data === undefined ? fallback(length) : f32WithLength(data, length, name, expectedShape);
  }

  return Object.freeze({
    rawF32,
    f32,
    byteView,
    addTensorGrad,
    scalarTensor,
    requirePositiveInteger,
    zerosF32,
    f32WithLength,
    defaultedF32,
    prepareF32,
  });
}
