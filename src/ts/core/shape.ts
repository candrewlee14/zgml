export type Shape = readonly number[];

export type PreparedHostValue = {
  data: Float32Array;
  shape: number[];
  shapeEvidence: boolean;
};

export type ShapedF32HelpersOptions = {
  f32: (value: unknown) => Float32Array;
  prepareF32: (value: unknown) => { data: Float32Array; shape: readonly number[] };
  valueShape?: (value: unknown) => readonly number[] | null;
  label?: string;
};

export type HostShapeValidationOptions = {
  allowSameElementShape?: boolean;
  allowFlatCapacity?: boolean;
};

export function shapeScalarCount(shape: Shape): number {
  let n = 1;
  for (const dim of shape) n *= dim;
  return n;
}

export function sameShape(actual: Shape, expected: Shape): boolean {
  return actual.length === expected.length && actual.every((dim, index) => dim === expected[index]);
}

export function isFlatShape(shape: Shape, length: number): boolean {
  return sameShape(shape, [length]);
}

export function hasNestedArrayShape(value: unknown): boolean {
  return Array.isArray(value) && value.some(Array.isArray);
}

export function createShapedF32Helpers(options: ShapedF32HelpersOptions) {
  const f32 = options && options.f32;
  const prepareF32 = options && options.prepareF32;
  let valueShape: (value: unknown) => readonly number[] | null = () => null;
  if (typeof options.valueShape === "function") {
    valueShape = options.valueShape;
  }
  const label = (options && options.label) || "host tensor shape helpers";
  if (typeof f32 !== "function") {
    throw new Error(`${label} require f32`);
  }
  if (typeof prepareF32 !== "function") {
    throw new Error(`${label} require prepareF32`);
  }

  function prepareHostValue(value: unknown): PreparedHostValue {
    const explicitShape = valueShape(value);
    if (Array.isArray(explicitShape)) {
      const data = f32(value);
      return { data, shape: explicitShape.slice(), shapeEvidence: true };
    }
    const prepared = prepareF32(value);
    return {
      data: prepared.data,
      shape: prepared.shape.slice(),
      shapeEvidence: hasNestedArrayShape(value),
    };
  }

  function validateHostValueShape(
    prepared: PreparedHostValue,
    expectedShape: Shape,
    valueLabel: string,
    validateOptions: HostShapeValidationOptions = {},
  ): void {
    if (!Array.isArray(expectedShape) || expectedShape.length === 0) return;
    if (sameShape(prepared.shape, expectedShape)) return;
    const expectedLength = shapeScalarCount(expectedShape);
    if (validateOptions.allowSameElementShape && prepared.data.length === expectedLength) return;
    if (
      prepared.shapeEvidence &&
      validateOptions.allowFlatCapacity &&
      prepared.data.length > expectedLength &&
      isFlatShape(prepared.shape, prepared.data.length)
    ) {
      return;
    }
    if (!prepared.shapeEvidence && isFlatShape(prepared.shape, prepared.data.length)) return;
    throw new Error(
      `${valueLabel} shape must be [${expectedShape.join(", ")}] or flat length ${prepared.data.length}, got [${prepared.shape.join(", ")}]`,
    );
  }

  return Object.freeze({
    prepareHostValue,
    validateHostValueShape,
  });
}

export function validateTensorShape(shape: Shape, length: number, name = "tensor shape"): readonly number[] {
  if (!Array.isArray(shape) || shape.length === 0) {
    throw new Error(`${name} must be a non-empty array`);
  }
  let product = 1;
  for (const dim of shape) {
    if (!Number.isSafeInteger(dim) || dim <= 0) {
      throw new Error(`${name} dimensions must be positive safe integers, got ${dim}`);
    }
    product *= dim;
  }
  if (product !== length) {
    throw new Error(`${name} product ${product} must match tensor length ${length}`);
  }
  return Object.freeze(shape.slice());
}

export function shapeProduct(shape: Shape): number {
  let product = 1;
  for (const dim of shape) product *= dim;
  return product;
}

export function normalizeFactoryShape(shape: number | Shape, name = "tensor shape"): { shape: number[]; length: number } {
  const dims = Number.isSafeInteger(shape) ? [shape as number] : shape;
  if (!Array.isArray(dims) || dims.length === 0) {
    throw new Error(`${name} must be a positive integer or non-empty array`);
  }
  let length = 1;
  for (const dim of dims) {
    if (!Number.isSafeInteger(dim) || dim <= 0) {
      throw new Error(`${name} dimensions must be positive safe integers, got ${dim}`);
    }
    length *= dim;
  }
  return { shape: dims.slice(), length };
}

export function rowMajorStrides(shape: Shape): number[] {
  const strides = new Array<number>(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i -= 1) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

export function inferredOperandShape(data: { length: number }, baseShape: Shape): number[] {
  if (data.length === 1) return [1];
  if (data.length === shapeProduct(baseShape)) return baseShape.slice();
  if (baseShape.length > 1 && data.length === baseShape[baseShape.length - 1]) return [data.length];
  return [data.length];
}

export function broadcastPlan(lhsShape: Shape, rhsShape: Shape, label: string) {
  const rank = Math.max(lhsShape.length, rhsShape.length);
  const outShape = new Array<number>(rank);
  for (let i = 0; i < rank; i += 1) {
    const lhsDim = lhsShape[lhsShape.length - rank + i] ?? 1;
    const rhsDim = rhsShape[rhsShape.length - rank + i] ?? 1;
    if (lhsDim !== rhsDim && lhsDim !== 1 && rhsDim !== 1) {
      throw new Error(`tensor ${label} shape mismatch: [${lhsShape.join(",")}] cannot broadcast with [${rhsShape.join(",")}]`);
    }
    outShape[i] = Math.max(lhsDim, rhsDim);
  }

  const outLen = shapeProduct(outShape);
  const outStrides = rowMajorStrides(outShape);
  const lhsStrides = rowMajorStrides(lhsShape);
  const rhsStrides = rowMajorStrides(rhsShape);
  const lhsIndex = new Uint32Array(outLen);
  const rhsIndex = new Uint32Array(outLen);
  for (let flat = 0; flat < outLen; flat += 1) {
    let li = 0;
    let ri = 0;
    for (let dim = 0; dim < rank; dim += 1) {
      const coord = Math.floor(flat / outStrides[dim]) % outShape[dim];
      const lhsDimIndex = dim - (rank - lhsShape.length);
      if (lhsDimIndex >= 0 && lhsShape[lhsDimIndex] !== 1) li += coord * lhsStrides[lhsDimIndex];
      const rhsDimIndex = dim - (rank - rhsShape.length);
      if (rhsDimIndex >= 0 && rhsShape[rhsDimIndex] !== 1) ri += coord * rhsStrides[rhsDimIndex];
    }
    lhsIndex[flat] = li;
    rhsIndex[flat] = ri;
  }
  return { shape: outShape, lhsIndex, rhsIndex };
}

export function normalizeDim(dim: number, rank: number, label: string): number {
  if (!Number.isSafeInteger(dim)) throw new Error(`${label} dim must be an integer, got ${dim}`);
  const axis = dim < 0 ? rank + dim : dim;
  if (axis < 0 || axis >= rank) throw new Error(`${label} dim ${dim} is out of range for rank ${rank}`);
  return axis;
}

export function normalizeInsertDim(dim: number, rank: number, label: string): number {
  if (!Number.isSafeInteger(dim)) throw new Error(`${label} dim must be an integer, got ${dim}`);
  const axis = dim < 0 ? rank + 1 + dim : dim;
  if (axis < 0 || axis > rank) throw new Error(`${label} dim ${dim} is out of range for rank ${rank}`);
  return axis;
}

export function dimReductionPlan(shape: Shape, dim: number, label: string) {
  const axis = normalizeDim(dim, shape.length, label);
  const outShape = shape.slice();
  outShape[axis] = 1;
  const inStrides = rowMajorStrides(shape);
  const outStrides = rowMajorStrides(outShape);
  const inToOut = new Uint32Array(shapeProduct(shape));
  for (let flat = 0; flat < inToOut.length; flat += 1) {
    let outIndex = 0;
    for (let d = 0; d < shape.length; d += 1) {
      if (d === axis) continue;
      const coord = Math.floor(flat / inStrides[d]) % shape[d];
      outIndex += coord * outStrides[d];
    }
    inToOut[flat] = outIndex;
  }
  return { axis, shape: outShape, inToOut, reduceLen: shape[axis] };
}

export function normalizeViewShape(shape: number | Shape, length: number, name: string): number[] {
  const dims = Number.isSafeInteger(shape) ? [shape as number] : shape;
  if (!Array.isArray(dims) || dims.length === 0) {
    throw new Error(`${name} shape must be an integer, -1, or non-empty shape array`);
  }

  let known = 1;
  let inferIndex = -1;
  const out = dims.map((dim, index) => {
    if (dim === -1) {
      if (inferIndex !== -1) throw new Error(`${name} shape can infer at most one dimension`);
      inferIndex = index;
      return dim;
    }
    if (!Number.isSafeInteger(dim) || dim <= 0) {
      throw new Error(`${name} shape dimensions must be positive safe integers or one -1, got ${dim}`);
    }
    known *= dim;
    return dim;
  });

  if (inferIndex !== -1) {
    if (length % known !== 0) {
      throw new Error(`${name} inferred shape product ${known} must divide tensor length ${length}`);
    }
    out[inferIndex] = length / known;
  }
  const product = shapeProduct(out);
  if (product !== length) {
    throw new Error(`${name} shape product ${product} must match tensor length ${length}`);
  }
  return out;
}

export function traceIndexForDim(index: number, dim: number, label: string): number {
  if (!Number.isSafeInteger(index)) throw new Error(`${label} index must be a safe integer, got ${index}`);
  const resolved = index < 0 ? dim + index : index;
  if (resolved < 0 || resolved >= dim) {
    throw new Error(`${label} index ${index} is out of range for dimension length ${dim}`);
  }
  return resolved;
}

export function traceSliceBound(value: number | null | undefined, dim: number, fallback: number, label: string): number {
  if (value === undefined || value === null) return fallback;
  if (!Number.isSafeInteger(value)) throw new Error(`${label} must be a safe integer, got ${value}`);
  const resolved = value < 0 ? dim + value : value;
  return Math.max(0, Math.min(dim, resolved));
}
