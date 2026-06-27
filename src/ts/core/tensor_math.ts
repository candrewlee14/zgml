"use strict";

import {
  broadcastPlan,
  dimReductionPlan,
  inferredOperandShape,
  normalizeDim,
  rowMajorStrides,
  shapeProduct,
  validateTensorShape,
} from "./shape.js";
import {
  geluDerivativeScalar,
  geluScalar,
  sigmoidDerivativeScalar,
  sigmoidScalar,
  siluDerivativeScalar,
  siluScalar,
} from "./activation.js";
import {
  isGradEnabled,
} from "./grad_mode.js";

type AnyRecord = Record<string, any>;
type TensorMathConstructOptions = Readonly<Record<string, unknown> & {
  requiresGrad?: boolean;
  requires_grad?: boolean;
  prev?: readonly TensorMathTensor[];
}>;
type TensorMathTensor = AnyRecord & {
  data: Float32Array;
  length: number;
  shape: readonly number[];
  requiresGrad: boolean;
  grad: Float32Array | null;
  _prev?: readonly TensorMathTensor[];
  _backward?: (grad: Float32Array | null) => void;
};
type TensorConstructor = new (
  data: Float32Array,
  shape: readonly number[],
  options?: TensorMathConstructOptions,
) => TensorMathTensor;
type BivariantCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];
type BinaryOp = (a: number, b: number) => number;
type BinaryGrad = (grad: number, a: number, b: number) => number;
type UnaryOp = (value: number, index?: number) => number;
type ComparisonOp = (a: number, b: number) => boolean;
type IsCloseOptions = Readonly<{
  rtol?: number;
  atol?: number;
  equalNan?: boolean;
  equal_nan?: boolean;
}>;
type F32Callback = (data: unknown) => Float32Array;
type AddTensorGradCallback = BivariantCallback<[tensor: TensorMathTensor, grad: Float32Array], void>;
type ScalarTensorCallback = (value: number, requiresGrad?: boolean) => TensorMathTensor;
type NativeEagerMatmulInto = (
  output: Float32Array,
  lhs: unknown,
  rhs: unknown,
  options?: Record<string, unknown>,
) => Float32Array;
type NativeEagerElementwiseInto = (
  output: Float32Array,
  lhs: unknown,
  rhs: unknown,
  options: Readonly<{ op: string }>,
) => Float32Array;

export type TensorMathHelpersOptions = Readonly<{
  Tensor?: TensorConstructor;
  getTensorClass?: () => TensorConstructor | undefined;
  f32: F32Callback;
  addTensorGrad: AddTensorGradCallback;
  scalarTensor: ScalarTensorCallback;
  isGradEnabled?: () => boolean;
  nativeEagerMatmulInto?: NativeEagerMatmulInto;
  nativeEagerElementwiseInto?: NativeEagerElementwiseInto;
  nativeEagerElementwiseMinLength?: number;
}>;

export function createTensorMathHelpers(options: TensorMathHelpersOptions) {
  const getTensorClass = typeof options.getTensorClass === "function"
    ? options.getTensorClass
    : () => options.Tensor;
  const f32 = options.f32;
  const addTensorGrad = options.addTensorGrad;
  const scalarTensor = options.scalarTensor;
  const nativeEagerMatmulInto = options.nativeEagerMatmulInto;
  const nativeEagerElementwiseInto = options.nativeEagerElementwiseInto;
  const nativeEagerElementwiseMinLength = Number.isSafeInteger(options.nativeEagerElementwiseMinLength) && Number(options.nativeEagerElementwiseMinLength) >= 0
    ? Number(options.nativeEagerElementwiseMinLength)
    : 512;
  const gradModeEnabled = typeof options.isGradEnabled === "function"
    ? options.isGradEnabled
    : isGradEnabled;
  if (typeof f32 !== "function" || typeof addTensorGrad !== "function" || typeof scalarTensor !== "function") {
    throw new Error("tensor math helpers require f32, addTensorGrad, and scalarTensor hooks");
  }

  function tensorClass() {
    const TensorClass = getTensorClass();
    if (typeof TensorClass !== "function") {
      throw new Error("tensor math helpers require a Tensor constructor");
    }
    return TensorClass as TensorConstructor;
  }

  function sameShape(left: readonly number[], right: readonly number[]) {
    return left.length === right.length && left.every((value, index) => value === right[index]);
  }

  function nativeElementwiseBinaryInto(
    output: Float32Array,
    tensor: TensorMathTensor,
    rhsTensor: TensorMathTensor | null,
    rhs: Float32Array,
    rhsShape: readonly number[],
    op: string,
  ) {
    if (typeof nativeEagerElementwiseInto !== "function") return false;
    if (output.length < nativeEagerElementwiseMinLength) return false;
    if (!sameShape(tensor.shape, rhsShape) && rhs.length !== 1) return false;
    nativeEagerElementwiseInto(output, tensor, rhsTensor ?? rhs, { op });
    return true;
  }

  function nativeElementwiseUnaryInto(output: Float32Array, tensor: TensorMathTensor, op: string) {
    if (typeof nativeEagerElementwiseInto !== "function") return false;
    if (output.length < nativeEagerElementwiseMinLength) return false;
    nativeEagerElementwiseInto(output, tensor, null, { op });
    return true;
  }

  function binary(tensor: TensorMathTensor, other: unknown, op: BinaryOp, gradLeft: BinaryGrad, gradRight: BinaryGrad, label: string, nativeOp = label) {
    const TensorClass = tensorClass();
    const rhsTensor = other instanceof TensorClass ? other as TensorMathTensor : null;
    const rhs = f32(other);
    const rhsShape = rhsTensor ? rhsTensor.shape : inferredOperandShape(rhs, tensor.shape);
    const plan = broadcastPlan(tensor.shape, rhsShape, label);
    const out = new Float32Array(shapeProduct(plan.shape));
    const gradEnabled = gradModeEnabled();
    const needsGrad = gradEnabled && (tensor.requiresGrad || Boolean(rhsTensor && rhsTensor.requiresGrad));
    if (!gradEnabled && nativeElementwiseBinaryInto(out, tensor, rhsTensor, rhs, rhsShape, nativeOp)) {
      // Native eager elementwise currently accepts same-shape or scalar RHS. General broadcasting stays in TS.
    } else {
      for (let i = 0; i < out.length; i += 1) out[i] = op(tensor.data[plan.lhsIndex[i]], rhs[plan.rhsIndex[i]]);
    }
    const result = new TensorClass(out, plan.shape, {
      requiresGrad: needsGrad,
      prev: needsGrad ? [tensor, ...(rhsTensor ? [rhsTensor] : [])] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      if (tensor.requiresGrad) {
        const leftGrad = new Float32Array(tensor.length);
        for (let i = 0; i < grad.length; i += 1) {
          const li = plan.lhsIndex[i];
          const ri = plan.rhsIndex[i];
          leftGrad[li] += gradLeft(grad[i], tensor.data[li], rhs[ri]);
        }
        addTensorGrad(tensor, leftGrad);
      }
      if (rhsTensor && rhsTensor.requiresGrad) {
        const rightGrad = new Float32Array(rhsTensor.length);
        for (let i = 0; i < grad.length; i += 1) {
          const li = plan.lhsIndex[i];
          const ri = plan.rhsIndex[i];
          rightGrad[ri] += gradRight(grad[i], tensor.data[li], rhs[ri]);
        }
        addTensorGrad(rhsTensor, rightGrad);
      }
    };
    return result;
  }

  function add(tensor: TensorMathTensor, other: unknown) {
    return binary(tensor, other, (a, b) => a + b, (g) => g, (g) => g, "add");
  }

  function compare(tensor: TensorMathTensor, other: unknown, op: ComparisonOp, label: string) {
    const TensorClass = tensorClass();
    const rhsTensor = other instanceof TensorClass ? other as TensorMathTensor : null;
    const rhs = f32(other);
    const rhsShape = rhsTensor ? rhsTensor.shape : inferredOperandShape(rhs, tensor.shape);
    const plan = broadcastPlan(tensor.shape, rhsShape, label);
    const out = new Float32Array(shapeProduct(plan.shape));
    for (let i = 0; i < out.length; i += 1) {
      out[i] = op(tensor.data[plan.lhsIndex[i]], rhs[plan.rhsIndex[i]]) ? 1 : 0;
    }
    return new TensorClass(out, plan.shape);
  }

  function isclose(tensor: TensorMathTensor, other: unknown, options: IsCloseOptions = {}) {
    const rtol = options.rtol ?? 1e-5;
    const atol = options.atol ?? 1e-8;
    const equalNan = options.equalNan ?? options.equal_nan ?? false;
    if (!Number.isFinite(rtol) || rtol < 0) throw new Error(`Tensor.isclose rtol must be a non-negative finite number, got ${rtol}`);
    if (!Number.isFinite(atol) || atol < 0) throw new Error(`Tensor.isclose atol must be a non-negative finite number, got ${atol}`);
    return compare(tensor, other, (a, b) => {
      if (Object.is(a, b) || a === b) return true;
      if (Number.isNaN(a) || Number.isNaN(b)) return equalNan && Number.isNaN(a) && Number.isNaN(b);
      return Math.abs(a - b) <= atol + rtol * Math.abs(b);
    }, "isclose");
  }

  function where(condition: TensorMathTensor, input: unknown, other: unknown) {
    const TensorClass = tensorClass();
    const inputTensor = input instanceof TensorClass ? input as TensorMathTensor : null;
    const otherTensor = other instanceof TensorClass ? other as TensorMathTensor : null;
    const inputData = f32(input);
    const inputShape = inputTensor ? inputTensor.shape : inferredOperandShape(inputData, condition.shape);
    const otherData = f32(other);
    const valuePlan = broadcastPlan(inputShape, otherTensor ? otherTensor.shape : inferredOperandShape(otherData, inputShape), "where values");
    const resultPlan = broadcastPlan(condition.shape, valuePlan.shape, "where condition");
    const conditionPlan = broadcastPlan(condition.shape, resultPlan.shape, "where condition");
    const inputPlan = broadcastPlan(inputShape, resultPlan.shape, "where input");
    const otherPlan = broadcastPlan(otherTensor ? otherTensor.shape : inferredOperandShape(otherData, resultPlan.shape), resultPlan.shape, "where other");
    const out = new Float32Array(shapeProduct(resultPlan.shape));
    for (let i = 0; i < out.length; i += 1) {
      out[i] = condition.data[conditionPlan.lhsIndex[i]] !== 0
        ? inputData[inputPlan.lhsIndex[i]]
        : otherData[otherPlan.lhsIndex[i]];
    }
    const needsGrad = gradModeEnabled() && (Boolean(inputTensor && inputTensor.requiresGrad) || Boolean(otherTensor && otherTensor.requiresGrad));
    const result = new TensorClass(out, resultPlan.shape, {
      requiresGrad: needsGrad,
      prev: needsGrad ? [ ...(inputTensor ? [inputTensor] : []), ...(otherTensor ? [otherTensor] : []) ] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      if (inputTensor && inputTensor.requiresGrad) {
        const inputGrad = new Float32Array(inputTensor.length);
        for (let i = 0; i < grad.length; i += 1) {
          if (condition.data[conditionPlan.lhsIndex[i]] !== 0) inputGrad[inputPlan.lhsIndex[i]] += grad[i];
        }
        addTensorGrad(inputTensor, inputGrad);
      }
      if (otherTensor && otherTensor.requiresGrad) {
        const otherGrad = new Float32Array(otherTensor.length);
        for (let i = 0; i < grad.length; i += 1) {
          if (condition.data[conditionPlan.lhsIndex[i]] === 0) otherGrad[otherPlan.lhsIndex[i]] += grad[i];
        }
        addTensorGrad(otherTensor, otherGrad);
      }
    };
    return result;
  }

  function maskedFill(tensor: TensorMathTensor, mask: unknown, value: unknown) {
    const TensorClass = tensorClass();
    const maskTensor = mask instanceof TensorClass
      ? mask as TensorMathTensor
      : new TensorClass(f32(mask), inferredOperandShape(f32(mask), tensor.shape));
    return where(maskTensor, value, tensor);
  }

  function masked_fill(tensor: TensorMathTensor, mask: unknown, value: unknown) {
    return maskedFill(tensor, mask, value);
  }

  function eq(tensor: TensorMathTensor, other: unknown) {
    return compare(tensor, other, (a, b) => Object.is(a, b) || a === b, "eq");
  }

  function ne(tensor: TensorMathTensor, other: unknown) {
    return compare(tensor, other, (a, b) => !(Object.is(a, b) || a === b), "ne");
  }

  function lt(tensor: TensorMathTensor, other: unknown) {
    return compare(tensor, other, (a, b) => a < b, "lt");
  }

  function le(tensor: TensorMathTensor, other: unknown) {
    return compare(tensor, other, (a, b) => a <= b, "le");
  }

  function gt(tensor: TensorMathTensor, other: unknown) {
    return compare(tensor, other, (a, b) => a > b, "gt");
  }

  function ge(tensor: TensorMathTensor, other: unknown) {
    return compare(tensor, other, (a, b) => a >= b, "ge");
  }

  function sub(tensor: TensorMathTensor, other: unknown) {
    return binary(tensor, other, (a, b) => a - b, (g) => g, (g) => -g, "sub");
  }

  function mul(tensor: TensorMathTensor, other: unknown) {
    return binary(tensor, other, (a, b) => a * b, (g, _a, b) => g * b, (g, a) => g * a, "mul");
  }

  function div(tensor: TensorMathTensor, other: unknown) {
    return binary(tensor, other, (a, b) => a / b, (g, _a, b) => g / b, (g, a, b) => -g * a / (b * b), "div");
  }

  function maximum(tensor: TensorMathTensor, other: unknown) {
    return binary(
      tensor,
      other,
      (a, b) => Math.max(a, b),
      (g, a, b) => a > b ? g : a === b ? g * 0.5 : 0,
      (g, a, b) => b > a ? g : a === b ? g * 0.5 : 0,
      "maximum",
    );
  }

  function minimum(tensor: TensorMathTensor, other: unknown) {
    return binary(
      tensor,
      other,
      (a, b) => Math.min(a, b),
      (g, a, b) => a < b ? g : a === b ? g * 0.5 : 0,
      (g, a, b) => b < a ? g : a === b ? g * 0.5 : 0,
      "minimum",
    );
  }

  function sqr(tensor: TensorMathTensor) {
    if (!gradModeEnabled()) {
      const TensorClass = tensorClass();
      const out = new Float32Array(tensor.length);
      if (nativeElementwiseUnaryInto(out, tensor, "sqr")) return new TensorClass(out, tensor.shape);
    }
    return mul(tensor, tensor);
  }

  function square(tensor: TensorMathTensor) {
    return sqr(tensor);
  }

  function pow(tensor: TensorMathTensor, exponent: number) {
    const value = Number(exponent);
    if (!Number.isFinite(value)) throw new Error(`Tensor.pow exponent must be finite, got ${exponent}`);
    return unary(
      tensor,
      (x) => Math.pow(x, value),
      (x) => value === 0 ? 0 : value * Math.pow(x, value - 1),
    );
  }

  function recip(tensor: TensorMathTensor) {
    return unary(tensor, (x) => 1 / x, (x) => -1 / (x * x), "recip");
  }

  function reciprocal(tensor: TensorMathTensor) {
    return recip(tensor);
  }

  function abs(tensor: TensorMathTensor) {
    return unary(tensor, Math.abs, (x) => x < 0 ? -1 : x > 0 ? 1 : 0, "abs");
  }

  function sgn(tensor: TensorMathTensor) {
    return unary(tensor, (x) => x < 0 ? -1 : x > 0 ? 1 : 0, () => 0);
  }

  function sign(tensor: TensorMathTensor) {
    return sgn(tensor);
  }

  function step(tensor: TensorMathTensor) {
    return unary(tensor, (x) => x > 0 ? 1 : 0, () => 0);
  }

  function isnan(tensor: TensorMathTensor) {
    return map(tensor, (x) => Number.isNaN(x) ? 1 : 0);
  }

  function isinf(tensor: TensorMathTensor) {
    return map(tensor, (x) => x === Infinity || x === -Infinity ? 1 : 0);
  }

  function isfinite(tensor: TensorMathTensor) {
    return map(tensor, (x) => Number.isFinite(x) ? 1 : 0);
  }

  function floor(tensor: TensorMathTensor) {
    return unary(tensor, Math.floor, () => 0);
  }

  function ceil(tensor: TensorMathTensor) {
    return unary(tensor, Math.ceil, () => 0);
  }

  function round(tensor: TensorMathTensor) {
    return unary(tensor, Math.round, () => 0);
  }

  function trunc(tensor: TensorMathTensor) {
    return unary(tensor, Math.trunc, () => 0);
  }

  function sqrt(tensor: TensorMathTensor) {
    return unary(tensor, Math.sqrt, (x) => 0.5 / Math.sqrt(x), "sqrt");
  }

  function rsqrt(tensor: TensorMathTensor) {
    return unary(tensor, (x) => 1 / Math.sqrt(x), (x) => -0.5 / Math.pow(x, 1.5));
  }

  function clamp(tensor: TensorMathTensor, min: number | null = null, max: number | null = null) {
    const hasMin = min !== undefined && min !== null;
    const hasMax = max !== undefined && max !== null;
    if (!hasMin && !hasMax) throw new Error("Tensor.clamp requires min, max, or both");
    const minValue = hasMin ? Number(min) : -Infinity;
    const maxValue = hasMax ? Number(max) : Infinity;
    if (!Number.isFinite(minValue) && hasMin) throw new Error(`Tensor.clamp min must be finite, got ${min}`);
    if (!Number.isFinite(maxValue) && hasMax) throw new Error(`Tensor.clamp max must be finite, got ${max}`);
    if (minValue > maxValue) throw new Error(`Tensor.clamp min ${minValue} must be <= max ${maxValue}`);

    return unary(
      tensor,
      (x) => Math.min(Math.max(x, minValue), maxValue),
      (x) => x >= minValue && x <= maxValue ? 1 : 0,
    );
  }

  function clip(tensor: TensorMathTensor, min: number | null = null, max: number | null = null) {
    return clamp(tensor, min, max);
  }

  function matmul(tensor: TensorMathTensor, other: unknown, otherShape?: readonly number[]) {
    const TensorClass = tensorClass();
    const rhsTensor = other instanceof TensorClass ? other as TensorMathTensor : null;
    if (tensor.shape.length !== 2) throw new Error("tensor matmul lhs must be rank-2");
    const rhs = f32(other);
    const rhsShape = rhsTensor ? rhsTensor.shape : validateTensorShape(otherShape ?? [], rhs.length, "tensor matmul rhs shape");
    if (rhsShape.length !== 2) throw new Error("tensor matmul rhs must be rank-2");
    const lhsRows = tensor.shape[0];
    const lhsCols = tensor.shape[1];
    const rhsRows = rhsShape[0];
    const rhsCols = rhsShape[1];
    if (lhsCols !== rhsRows) {
      throw new Error(`tensor matmul shape mismatch: ${lhsRows}x${lhsCols} cannot multiply ${rhsRows}x${rhsCols}`);
    }
    const out = new Float32Array(lhsRows * rhsCols);
    const gradEnabled = gradModeEnabled();
    const needsGrad = gradEnabled && (tensor.requiresGrad || Boolean(rhsTensor && rhsTensor.requiresGrad));
    if (!gradEnabled && typeof nativeEagerMatmulInto === "function") {
      nativeEagerMatmulInto(out, tensor, rhsTensor ?? rhs, {
        rows: lhsRows,
        shared: lhsCols,
        cols: rhsCols,
      });
    } else {
      for (let r = 0; r < lhsRows; r += 1) {
        for (let c = 0; c < rhsCols; c += 1) {
          let acc = 0;
          for (let k = 0; k < lhsCols; k += 1) {
            acc += tensor.data[r * lhsCols + k] * rhs[k * rhsCols + c];
          }
          out[r * rhsCols + c] = acc;
        }
      }
    }
    const result = new TensorClass(out, [lhsRows, rhsCols], {
      requiresGrad: needsGrad,
      prev: needsGrad ? [tensor, ...(rhsTensor ? [rhsTensor] : [])] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      if (tensor.requiresGrad) {
        const leftGrad = new Float32Array(tensor.length);
        for (let r = 0; r < lhsRows; r += 1) {
          for (let k = 0; k < lhsCols; k += 1) {
            let acc = 0;
            for (let c = 0; c < rhsCols; c += 1) acc += grad[r * rhsCols + c] * rhs[k * rhsCols + c];
            leftGrad[r * lhsCols + k] += acc;
          }
        }
        addTensorGrad(tensor, leftGrad);
      }
      if (rhsTensor && rhsTensor.requiresGrad) {
        const rightGrad = new Float32Array(rhsTensor.length);
        for (let k = 0; k < lhsCols; k += 1) {
          for (let c = 0; c < rhsCols; c += 1) {
            let acc = 0;
            for (let r = 0; r < lhsRows; r += 1) acc += tensor.data[r * lhsCols + k] * grad[r * rhsCols + c];
            rightGrad[k * rhsCols + c] += acc;
          }
        }
        addTensorGrad(rhsTensor, rightGrad);
      }
    };
    return result;
  }

  function dot(tensor: TensorMathTensor, other: unknown, otherShape?: readonly number[]) {
    const TensorClass = tensorClass();
    const rhsTensor = other instanceof TensorClass ? other as TensorMathTensor : null;
    if (tensor.shape.length !== 1) throw new Error("tensor dot lhs must be rank-1");
    const rhs = f32(other);
    const rhsShape = rhsTensor ? rhsTensor.shape : validateTensorShape(otherShape ?? [rhs.length], rhs.length, "tensor dot rhs shape");
    if (rhsShape.length !== 1) throw new Error("tensor dot rhs must be rank-1");
    if (tensor.shape[0] !== rhsShape[0]) {
      throw new Error(`tensor dot shape mismatch: [${tensor.shape[0]}] cannot dot [${rhsShape[0]}]`);
    }
    let acc = 0;
    for (let i = 0; i < tensor.length; i += 1) acc += tensor.data[i] * rhs[i];
    const needsGrad = gradModeEnabled() && (tensor.requiresGrad || Boolean(rhsTensor && rhsTensor.requiresGrad));
    const result = new TensorClass(Float32Array.of(acc), [1], {
      requiresGrad: needsGrad,
      prev: needsGrad ? [tensor, ...(rhsTensor ? [rhsTensor] : [])] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const upstream = grad[0];
      if (tensor.requiresGrad) {
        const leftGrad = new Float32Array(tensor.length);
        for (let i = 0; i < tensor.length; i += 1) leftGrad[i] = upstream * rhs[i];
        addTensorGrad(tensor, leftGrad);
      }
      if (rhsTensor && rhsTensor.requiresGrad) {
        const rightGrad = new Float32Array(rhsTensor.length);
        for (let i = 0; i < rhsTensor.length; i += 1) rightGrad[i] = upstream * tensor.data[i];
        addTensorGrad(rhsTensor, rightGrad);
      }
    };
    return result;
  }

  function trace(tensor: TensorMathTensor) {
    const TensorClass = tensorClass();
    if (tensor.shape.length !== 2) throw new Error("tensor trace input must be rank-2");
    const rows = tensor.shape[0];
    const cols = tensor.shape[1];
    const diagonal = Math.min(rows, cols);
    let acc = 0;
    for (let i = 0; i < diagonal; i += 1) acc += tensor.data[i * cols + i];
    const result = new TensorClass(Float32Array.of(acc), [1], {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < diagonal; i += 1) inGrad[i * cols + i] = grad[0];
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function diagonal(tensor: TensorMathTensor) {
    const TensorClass = tensorClass();
    if (tensor.shape.length !== 2) throw new Error("tensor diagonal input must be rank-2");
    const rows = tensor.shape[0];
    const cols = tensor.shape[1];
    const diagonalLength = Math.min(rows, cols);
    const out = new Float32Array(diagonalLength);
    for (let i = 0; i < diagonalLength; i += 1) out[i] = tensor.data[i * cols + i];
    const result = new TensorClass(out, [diagonalLength], {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < diagonalLength; i += 1) inGrad[i * cols + i] = grad[i];
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function bmm(tensor: TensorMathTensor, other: unknown, otherShape?: readonly number[]) {
    const TensorClass = tensorClass();
    const rhsTensor = other instanceof TensorClass ? other as TensorMathTensor : null;
    if (tensor.shape.length !== 3) throw new Error("tensor bmm lhs must be rank-3");
    const rhs = f32(other);
    const rhsShape = rhsTensor ? rhsTensor.shape : validateTensorShape(otherShape ?? [], rhs.length, "tensor bmm rhs shape");
    if (rhsShape.length !== 3) throw new Error("tensor bmm rhs must be rank-3");
    const batch = tensor.shape[0];
    const lhsRows = tensor.shape[1];
    const lhsCols = tensor.shape[2];
    const rhsBatch = rhsShape[0];
    const rhsRows = rhsShape[1];
    const rhsCols = rhsShape[2];
    if (batch !== rhsBatch || lhsCols !== rhsRows) {
      throw new Error(`tensor bmm shape mismatch: [${batch},${lhsRows},${lhsCols}] cannot multiply [${rhsBatch},${rhsRows},${rhsCols}]`);
    }
    const out = new Float32Array(batch * lhsRows * rhsCols);
    for (let b = 0; b < batch; b += 1) {
      const lhsBatchOffset = b * lhsRows * lhsCols;
      const rhsBatchOffset = b * rhsRows * rhsCols;
      const outBatchOffset = b * lhsRows * rhsCols;
      for (let r = 0; r < lhsRows; r += 1) {
        for (let c = 0; c < rhsCols; c += 1) {
          let acc = 0;
          for (let k = 0; k < lhsCols; k += 1) {
            acc += tensor.data[lhsBatchOffset + r * lhsCols + k] * rhs[rhsBatchOffset + k * rhsCols + c];
          }
          out[outBatchOffset + r * rhsCols + c] = acc;
        }
      }
    }
    const needsGrad = gradModeEnabled() && (tensor.requiresGrad || Boolean(rhsTensor && rhsTensor.requiresGrad));
    const result = new TensorClass(out, [batch, lhsRows, rhsCols], {
      requiresGrad: needsGrad,
      prev: needsGrad ? [tensor, ...(rhsTensor ? [rhsTensor] : [])] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      if (tensor.requiresGrad) {
        const leftGrad = new Float32Array(tensor.length);
        for (let b = 0; b < batch; b += 1) {
          const lhsBatchOffset = b * lhsRows * lhsCols;
          const rhsBatchOffset = b * rhsRows * rhsCols;
          const outBatchOffset = b * lhsRows * rhsCols;
          for (let r = 0; r < lhsRows; r += 1) {
            for (let k = 0; k < lhsCols; k += 1) {
              let acc = 0;
              for (let c = 0; c < rhsCols; c += 1) {
                acc += grad[outBatchOffset + r * rhsCols + c] * rhs[rhsBatchOffset + k * rhsCols + c];
              }
              leftGrad[lhsBatchOffset + r * lhsCols + k] += acc;
            }
          }
        }
        addTensorGrad(tensor, leftGrad);
      }
      if (rhsTensor && rhsTensor.requiresGrad) {
        const rightGrad = new Float32Array(rhsTensor.length);
        for (let b = 0; b < batch; b += 1) {
          const lhsBatchOffset = b * lhsRows * lhsCols;
          const rhsBatchOffset = b * rhsRows * rhsCols;
          const outBatchOffset = b * lhsRows * rhsCols;
          for (let k = 0; k < lhsCols; k += 1) {
            for (let c = 0; c < rhsCols; c += 1) {
              let acc = 0;
              for (let r = 0; r < lhsRows; r += 1) {
                acc += tensor.data[lhsBatchOffset + r * lhsCols + k] * grad[outBatchOffset + r * rhsCols + c];
              }
              rightGrad[rhsBatchOffset + k * rhsCols + c] += acc;
            }
          }
        }
        addTensorGrad(rhsTensor, rightGrad);
      }
    };
    return result;
  }

  function map(tensor: TensorMathTensor, fn: UnaryOp) {
    const TensorClass = tensorClass();
    const out = new Float32Array(tensor.length);
    for (let i = 0; i < out.length; i += 1) out[i] = fn(tensor.data[i], i);
    return new TensorClass(out, tensor.shape);
  }

  function unary(tensor: TensorMathTensor, fn: UnaryOp, derivative: UnaryOp, nativeOp?: string) {
    const TensorClass = tensorClass();
    const out = new Float32Array(tensor.length);
    const gradEnabled = gradModeEnabled();
    if (!gradEnabled && nativeOp && nativeElementwiseUnaryInto(out, tensor, nativeOp)) {
      // Native eager elementwise owns the no-grad tensor-sized path for supported unary ops.
    } else {
      for (let i = 0; i < out.length; i += 1) out[i] = fn(tensor.data[i]);
    }
    const result = new TensorClass(out, tensor.shape, {
      requiresGrad: gradEnabled && tensor.requiresGrad,
      prev: gradEnabled && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < inGrad.length; i += 1) inGrad[i] = grad[i] * derivative(tensor.data[i]);
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function relu(tensor: TensorMathTensor) {
    return unary(tensor, (x) => Math.max(0, x), (x) => x > 0 ? 1 : 0);
  }

  function gelu(tensor: TensorMathTensor) {
    return unary(tensor, geluScalar, geluDerivativeScalar);
  }

  function silu(tensor: TensorMathTensor) {
    return unary(tensor, siluScalar, siluDerivativeScalar);
  }

  function sigmoid(tensor: TensorMathTensor) {
    return unary(tensor, sigmoidScalar, sigmoidDerivativeScalar);
  }

  function tanh(tensor: TensorMathTensor) {
    return unary(tensor, Math.tanh, (x) => {
      const value = Math.tanh(x);
      return 1 - value * value;
    });
  }

  function sin(tensor: TensorMathTensor) {
    return unary(tensor, Math.sin, Math.cos);
  }

  function cos(tensor: TensorMathTensor) {
    return unary(tensor, Math.cos, (x) => -Math.sin(x));
  }

  function tan(tensor: TensorMathTensor) {
    return unary(tensor, Math.tan, (x) => {
      const value = Math.cos(x);
      return 1 / (value * value);
    });
  }

  function exp(tensor: TensorMathTensor) {
    return unary(tensor, Math.exp, Math.exp, "exp");
  }

  function expm1(tensor: TensorMathTensor) {
    return unary(tensor, Math.expm1, Math.exp);
  }

  function log(tensor: TensorMathTensor) {
    return unary(tensor, Math.log, (x) => 1 / x, "log");
  }

  function log1p(tensor: TensorMathTensor) {
    return unary(tensor, Math.log1p, (x) => 1 / (1 + x));
  }

  function neg(tensor: TensorMathTensor) {
    return unary(tensor, (x) => -x, () => -1, "neg");
  }

  function negative(tensor: TensorMathTensor) {
    return neg(tensor);
  }

  function sumDim(tensor: TensorMathTensor, dim: number) {
    const TensorClass = tensorClass();
    const plan = dimReductionPlan(tensor.shape, dim, "sumDim");
    const out = new Float32Array(shapeProduct(plan.shape));
    for (let i = 0; i < tensor.length; i += 1) out[plan.inToOut[i]] += tensor.data[i];
    const result = new TensorClass(out, plan.shape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < tensor.length; i += 1) inGrad[i] = grad[plan.inToOut[i]];
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function meanDim(tensor: TensorMathTensor, dim: number) {
    return div(sumDim(tensor, dim), scalarTensor(dimReductionPlan(tensor.shape, dim, "meanDim").reduceLen));
  }

  function prodDim(tensor: TensorMathTensor, dim: number) {
    const TensorClass = tensorClass();
    const plan = dimReductionPlan(tensor.shape, dim, "prodDim");
    const out = new Float32Array(shapeProduct(plan.shape));
    const zeroCounts = new Uint32Array(out.length);
    const nonZeroProducts = new Float32Array(out.length);
    out.fill(1);
    nonZeroProducts.fill(1);
    for (let i = 0; i < tensor.length; i += 1) {
      const oi = plan.inToOut[i];
      const value = tensor.data[i];
      out[oi] *= value;
      if (value === 0) zeroCounts[oi] += 1;
      else nonZeroProducts[oi] *= value;
    }
    const result = new TensorClass(out, plan.shape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < tensor.length; i += 1) {
        const oi = plan.inToOut[i];
        const value = tensor.data[i];
        if (zeroCounts[oi] === 0) inGrad[i] = grad[oi] * out[oi] / value;
        else if (zeroCounts[oi] === 1 && value === 0) inGrad[i] = grad[oi] * nonZeroProducts[oi];
      }
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function cumsum(tensor: TensorMathTensor, dim = -1) {
    const TensorClass = tensorClass();
    const axis = normalizeDim(dim, tensor.shape.length, "cumsum");
    const out = cumsumData(tensor.data, tensor.shape, axis, false);
    const result = new TensorClass(out, tensor.shape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      addTensorGrad(tensor, cumsumData(grad, tensor.shape, axis, true));
    };
    return result;
  }

  function cumsumData(data: Float32Array, shape: readonly number[], axis: number, reverse: boolean) {
    const strides = rowMajorStrides(shape);
    const axisLen = shape[axis];
    const axisStride = strides[axis];
    const outerShape = shape.filter((_: number, index: number) => index !== axis);
    const outerLen = outerShape.length === 0 ? 1 : shapeProduct(outerShape);
    const outerStrides = outerShape.length === 0 ? [] : rowMajorStrides(outerShape);
    const out = new Float32Array(data.length);
    for (let outerFlat = 0; outerFlat < outerLen; outerFlat += 1) {
      let base = 0;
      let outerDim = 0;
      for (let dimIndex = 0; dimIndex < shape.length; dimIndex += 1) {
        if (dimIndex === axis) continue;
        const coord = outerShape.length === 0 ? 0 : Math.floor(outerFlat / outerStrides[outerDim]) % outerShape[outerDim];
        base += coord * strides[dimIndex];
        outerDim += 1;
      }
      let acc = 0;
      for (let offset = 0; offset < axisLen; offset += 1) {
        const axisIndex = reverse ? axisLen - 1 - offset : offset;
        const flat = base + axisIndex * axisStride;
        acc += data[flat];
        out[flat] = acc;
      }
    }
    return out;
  }

  function maxDim(tensor: TensorMathTensor, dim: number) {
    const TensorClass = tensorClass();
    const plan = dimReductionPlan(tensor.shape, dim, "maxDim");
    const out = new Float32Array(shapeProduct(plan.shape));
    const counts = new Uint32Array(out.length);
    out.fill(-Infinity);
    for (let i = 0; i < tensor.length; i += 1) {
      const oi = plan.inToOut[i];
      const value = tensor.data[i];
      if (value > out[oi]) {
        out[oi] = value;
        counts[oi] = 1;
      } else if (value === out[oi]) {
        counts[oi] += 1;
      }
    }
    const result = new TensorClass(out, plan.shape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < tensor.length; i += 1) {
        const oi = plan.inToOut[i];
        if (tensor.data[i] === out[oi]) inGrad[i] += grad[oi] / counts[oi];
      }
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function minDim(tensor: TensorMathTensor, dim: number) {
    const TensorClass = tensorClass();
    const plan = dimReductionPlan(tensor.shape, dim, "minDim");
    const out = new Float32Array(shapeProduct(plan.shape));
    const counts = new Uint32Array(out.length);
    out.fill(Infinity);
    for (let i = 0; i < tensor.length; i += 1) {
      const oi = plan.inToOut[i];
      const value = tensor.data[i];
      if (value < out[oi]) {
        out[oi] = value;
        counts[oi] = 1;
      } else if (value === out[oi]) {
        counts[oi] += 1;
      }
    }
    const result = new TensorClass(out, plan.shape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < tensor.length; i += 1) {
        const oi = plan.inToOut[i];
        if (tensor.data[i] === out[oi]) inGrad[i] += grad[oi] / counts[oi];
      }
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function argmaxDim(tensor: TensorMathTensor, dim: number) {
    const TensorClass = tensorClass();
    const plan = dimReductionPlan(tensor.shape, dim, "argmaxDim");
    const out = new Float32Array(shapeProduct(plan.shape));
    const values = new Float32Array(out.length);
    const strides = rowMajorStrides(tensor.shape);
    values.fill(-Infinity);
    for (let i = 0; i < tensor.length; i += 1) {
      const oi = plan.inToOut[i];
      const value = tensor.data[i];
      if (value > values[oi]) {
        values[oi] = value;
        out[oi] = Math.floor(i / strides[plan.axis]) % plan.reduceLen;
      }
    }
    return new TensorClass(out, plan.shape);
  }

  function argminDim(tensor: TensorMathTensor, dim: number) {
    const TensorClass = tensorClass();
    const plan = dimReductionPlan(tensor.shape, dim, "argminDim");
    const out = new Float32Array(shapeProduct(plan.shape));
    const values = new Float32Array(out.length);
    const strides = rowMajorStrides(tensor.shape);
    values.fill(Infinity);
    for (let i = 0; i < tensor.length; i += 1) {
      const oi = plan.inToOut[i];
      const value = tensor.data[i];
      if (value < values[oi]) {
        values[oi] = value;
        out[oi] = Math.floor(i / strides[plan.axis]) % plan.reduceLen;
      }
    }
    return new TensorClass(out, plan.shape);
  }

  function anyDim(tensor: TensorMathTensor, dim: number) {
    const TensorClass = tensorClass();
    const plan = dimReductionPlan(tensor.shape, dim, "anyDim");
    const out = new Float32Array(shapeProduct(plan.shape));
    for (let i = 0; i < tensor.length; i += 1) {
      if (tensor.data[i] !== 0) out[plan.inToOut[i]] = 1;
    }
    return new TensorClass(out, plan.shape);
  }

  function allDim(tensor: TensorMathTensor, dim: number) {
    const TensorClass = tensorClass();
    const plan = dimReductionPlan(tensor.shape, dim, "allDim");
    const out = new Float32Array(shapeProduct(plan.shape));
    out.fill(1);
    for (let i = 0; i < tensor.length; i += 1) {
      if (tensor.data[i] === 0) out[plan.inToOut[i]] = 0;
    }
    return new TensorClass(out, plan.shape);
  }

  function softmax(tensor: TensorMathTensor) {
    return softmaxDim(tensor, -1);
  }

  function softmaxDim(tensor: TensorMathTensor, dim: number) {
    const TensorClass = tensorClass();
    const plan = dimReductionPlan(tensor.shape, dim, "softmaxDim");
    const maxValues = new Float32Array(shapeProduct(plan.shape));
    const denom = new Float32Array(maxValues.length);
    maxValues.fill(-Infinity);
    for (let i = 0; i < tensor.length; i += 1) {
      const oi = plan.inToOut[i];
      if (tensor.data[i] > maxValues[oi]) maxValues[oi] = tensor.data[i];
    }
    const out = new Float32Array(tensor.length);
    for (let i = 0; i < tensor.length; i += 1) {
      const oi = plan.inToOut[i];
      const value = Math.exp(tensor.data[i] - maxValues[oi]);
      out[i] = value;
      denom[oi] += value;
    }
    for (let i = 0; i < tensor.length; i += 1) out[i] /= denom[plan.inToOut[i]];
    const result = new TensorClass(out, tensor.shape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const dot = new Float32Array(maxValues.length);
      for (let i = 0; i < tensor.length; i += 1) dot[plan.inToOut[i]] += grad[i] * out[i];
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < tensor.length; i += 1) {
        inGrad[i] = out[i] * (grad[i] - dot[plan.inToOut[i]]);
      }
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function logSoftmax(tensor: TensorMathTensor) {
    return logSoftmaxDim(tensor, -1);
  }

  function logSoftmaxDim(tensor: TensorMathTensor, dim: number) {
    const TensorClass = tensorClass();
    const plan = dimReductionPlan(tensor.shape, dim, "logSoftmaxDim");
    const maxValues = new Float32Array(shapeProduct(plan.shape));
    const denom = new Float32Array(maxValues.length);
    maxValues.fill(-Infinity);
    for (let i = 0; i < tensor.length; i += 1) {
      const oi = plan.inToOut[i];
      if (tensor.data[i] > maxValues[oi]) maxValues[oi] = tensor.data[i];
    }
    for (let i = 0; i < tensor.length; i += 1) {
      const oi = plan.inToOut[i];
      denom[oi] += Math.exp(tensor.data[i] - maxValues[oi]);
    }
    const out = new Float32Array(tensor.length);
    const probs = new Float32Array(tensor.length);
    for (let i = 0; i < tensor.length; i += 1) {
      const oi = plan.inToOut[i];
      const shifted = tensor.data[i] - maxValues[oi];
      out[i] = shifted - Math.log(denom[oi]);
      probs[i] = Math.exp(shifted) / denom[oi];
    }
    const result = new TensorClass(out, tensor.shape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const sumGrad = new Float32Array(maxValues.length);
      for (let i = 0; i < tensor.length; i += 1) sumGrad[plan.inToOut[i]] += grad[i];
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < tensor.length; i += 1) {
        inGrad[i] = grad[i] - probs[i] * sumGrad[plan.inToOut[i]];
      }
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function logsumexp(tensor: TensorMathTensor, dim?: number) {
    if (dim !== undefined) return logsumexpDim(tensor, dim);
    const TensorClass = tensorClass();
    let maxValue = -Infinity;
    for (let i = 0; i < tensor.length; i += 1) {
      if (tensor.data[i] > maxValue) maxValue = tensor.data[i];
    }
    let denom = 0;
    for (let i = 0; i < tensor.length; i += 1) denom += Math.exp(tensor.data[i] - maxValue);
    const out = new Float32Array([maxValue + Math.log(denom)]);
    const result = new TensorClass(out, [1], {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < tensor.length; i += 1) {
        inGrad[i] = grad[0] * Math.exp(tensor.data[i] - maxValue) / denom;
      }
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function logSumExp(tensor: TensorMathTensor, dim?: number) {
    return logsumexp(tensor, dim);
  }

  function logsumexpDim(tensor: TensorMathTensor, dim: number) {
    const TensorClass = tensorClass();
    const plan = dimReductionPlan(tensor.shape, dim, "logsumexp");
    const maxValues = new Float32Array(shapeProduct(plan.shape));
    const denom = new Float32Array(maxValues.length);
    maxValues.fill(-Infinity);
    for (let i = 0; i < tensor.length; i += 1) {
      const oi = plan.inToOut[i];
      if (tensor.data[i] > maxValues[oi]) maxValues[oi] = tensor.data[i];
    }
    for (let i = 0; i < tensor.length; i += 1) {
      const oi = plan.inToOut[i];
      denom[oi] += Math.exp(tensor.data[i] - maxValues[oi]);
    }
    const out = new Float32Array(maxValues.length);
    for (let i = 0; i < out.length; i += 1) out[i] = maxValues[i] + Math.log(denom[i]);
    const result = new TensorClass(out, plan.shape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < tensor.length; i += 1) {
        const oi = plan.inToOut[i];
        inGrad[i] = grad[oi] * Math.exp(tensor.data[i] - maxValues[oi]) / denom[oi];
      }
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function variance(tensor: TensorMathTensor, dim?: number, correction = 0) {
    if (dim !== undefined) return varianceDim(tensor, dim, correction);
    const TensorClass = tensorClass();
    const denom = varianceDenominator(tensor.length, correction, "variance");
    let meanValue = 0;
    for (let i = 0; i < tensor.length; i += 1) meanValue += tensor.data[i];
    meanValue /= tensor.length;
    let acc = 0;
    for (let i = 0; i < tensor.length; i += 1) {
      const centered = tensor.data[i] - meanValue;
      acc += centered * centered;
    }
    const out = new Float32Array([acc / denom]);
    const result = new TensorClass(out, [1], {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < tensor.length; i += 1) {
        inGrad[i] = grad[0] * 2 * (tensor.data[i] - meanValue) / denom;
      }
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function var_(tensor: TensorMathTensor, dim?: number, correction = 0) {
    return variance(tensor, dim, correction);
  }

  function std(tensor: TensorMathTensor, dim?: number, correction = 0) {
    if (dim !== undefined) return stdDim(tensor, dim, correction);
    const TensorClass = tensorClass();
    const denom = varianceDenominator(tensor.length, correction, "std");
    let meanValue = 0;
    for (let i = 0; i < tensor.length; i += 1) meanValue += tensor.data[i];
    meanValue /= tensor.length;
    let acc = 0;
    for (let i = 0; i < tensor.length; i += 1) {
      const centered = tensor.data[i] - meanValue;
      acc += centered * centered;
    }
    const stdValue = Math.sqrt(acc / denom);
    const out = new Float32Array([stdValue]);
    const result = new TensorClass(out, [1], {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      if (stdValue !== 0) {
        for (let i = 0; i < tensor.length; i += 1) {
          inGrad[i] = grad[0] * (tensor.data[i] - meanValue) / (denom * stdValue);
        }
      }
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function norm(tensor: TensorMathTensor, dim?: number, p = 2) {
    if (dim !== undefined) return normDim(tensor, dim, p);
    const TensorClass = tensorClass();
    const power = normPower(p);
    let acc = 0;
    for (let i = 0; i < tensor.length; i += 1) acc += Math.abs(tensor.data[i]) ** power;
    const value = acc ** (1 / power);
    const out = new Float32Array([value]);
    const result = new TensorClass(out, [1], {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      if (value !== 0) {
        const denom = value ** (power - 1);
        for (let i = 0; i < tensor.length; i += 1) {
          const input = tensor.data[i];
          if (input !== 0) inGrad[i] = grad[0] * Math.sign(input) * (Math.abs(input) ** (power - 1)) / denom;
        }
      }
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function normDim(tensor: TensorMathTensor, dim: number, p = 2) {
    const TensorClass = tensorClass();
    const power = normPower(p);
    const plan = dimReductionPlan(tensor.shape, dim, "norm");
    const out = new Float32Array(shapeProduct(plan.shape));
    for (let i = 0; i < tensor.length; i += 1) {
      out[plan.inToOut[i]] += Math.abs(tensor.data[i]) ** power;
    }
    for (let i = 0; i < out.length; i += 1) out[i] = out[i] ** (1 / power);
    const result = new TensorClass(out, plan.shape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < tensor.length; i += 1) {
        const oi = plan.inToOut[i];
        const denom = out[oi] ** (power - 1);
        const input = tensor.data[i];
        if (denom !== 0 && input !== 0) {
          inGrad[i] = grad[oi] * Math.sign(input) * (Math.abs(input) ** (power - 1)) / denom;
        }
      }
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function normPower(p: number) {
    const value = Number(p);
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(`Tensor.norm p must be finite and > 0, got ${p}`);
    }
    return value;
  }

  function varianceDim(tensor: TensorMathTensor, dim: number, correction = 0) {
    const TensorClass = tensorClass();
    const plan = dimReductionPlan(tensor.shape, dim, "variance");
    const denom = varianceDenominator(plan.reduceLen, correction, "variance");
    const meanValues = new Float32Array(shapeProduct(plan.shape));
    for (let i = 0; i < tensor.length; i += 1) meanValues[plan.inToOut[i]] += tensor.data[i];
    for (let i = 0; i < meanValues.length; i += 1) meanValues[i] /= plan.reduceLen;
    const out = new Float32Array(meanValues.length);
    for (let i = 0; i < tensor.length; i += 1) {
      const oi = plan.inToOut[i];
      const centered = tensor.data[i] - meanValues[oi];
      out[oi] += centered * centered;
    }
    for (let i = 0; i < out.length; i += 1) out[i] /= denom;
    const result = new TensorClass(out, plan.shape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < tensor.length; i += 1) {
        const oi = plan.inToOut[i];
        inGrad[i] = grad[oi] * 2 * (tensor.data[i] - meanValues[oi]) / denom;
      }
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function stdDim(tensor: TensorMathTensor, dim: number, correction = 0) {
    const TensorClass = tensorClass();
    const plan = dimReductionPlan(tensor.shape, dim, "std");
    const denom = varianceDenominator(plan.reduceLen, correction, "std");
    const meanValues = new Float32Array(shapeProduct(plan.shape));
    for (let i = 0; i < tensor.length; i += 1) meanValues[plan.inToOut[i]] += tensor.data[i];
    for (let i = 0; i < meanValues.length; i += 1) meanValues[i] /= plan.reduceLen;
    const out = new Float32Array(meanValues.length);
    for (let i = 0; i < tensor.length; i += 1) {
      const oi = plan.inToOut[i];
      const centered = tensor.data[i] - meanValues[oi];
      out[oi] += centered * centered;
    }
    for (let i = 0; i < out.length; i += 1) out[i] = Math.sqrt(out[i] / denom);
    const result = new TensorClass(out, plan.shape, {
      requiresGrad: gradModeEnabled() && tensor.requiresGrad,
      prev: gradModeEnabled() && tensor.requiresGrad ? [tensor] : [],
    });
    result._backward = (grad: Float32Array | null) => {
      if (!grad) return;
      const inGrad = new Float32Array(tensor.length);
      for (let i = 0; i < tensor.length; i += 1) {
        const oi = plan.inToOut[i];
        if (out[oi] !== 0) inGrad[i] = grad[oi] * (tensor.data[i] - meanValues[oi]) / (denom * out[oi]);
      }
      addTensorGrad(tensor, inGrad);
    };
    return result;
  }

  function varianceDenominator(count: number, correction: number, label: string) {
    const value = Number(correction);
    if (!Number.isFinite(value) || value < 0) {
      throw new Error(`Tensor.${label} correction must be finite and >= 0, got ${correction}`);
    }
    const denom = count - value;
    if (denom <= 0) {
      throw new Error(`Tensor.${label} correction ${correction} must be less than reduction size ${count}`);
    }
    return denom;
  }

  function sum(tensor: TensorMathTensor, dim?: number) {
    if (dim !== undefined) return sumDim(tensor, dim);
    let acc = 0;
    for (const value of tensor.data) acc += value;
    const result = scalarTensor(acc, gradModeEnabled() && tensor.requiresGrad);
    if (gradModeEnabled() && tensor.requiresGrad) {
      result._prev = [tensor];
      result._backward = (grad: Float32Array | null) => {
        if (!grad) return;
        const inGrad = new Float32Array(tensor.length);
        inGrad.fill(grad[0]);
        addTensorGrad(tensor, inGrad);
      };
    }
    return result;
  }

  function max(tensor: TensorMathTensor, dim?: number) {
    if (dim !== undefined) return maxDim(tensor, dim);
    let acc = -Infinity;
    for (const value of tensor.data) acc = Math.max(acc, value);
    const result = scalarTensor(acc, gradModeEnabled() && tensor.requiresGrad);
    if (gradModeEnabled() && tensor.requiresGrad) {
      let count = 0;
      for (const value of tensor.data) if (value === acc) count += 1;
      result._prev = [tensor];
      result._backward = (grad: Float32Array | null) => {
        if (!grad) return;
        const inGrad = new Float32Array(tensor.length);
        for (let i = 0; i < tensor.length; i += 1) {
          if (tensor.data[i] === acc) inGrad[i] = grad[0] / count;
        }
        addTensorGrad(tensor, inGrad);
      };
    }
    return result;
  }

  function min(tensor: TensorMathTensor, dim?: number) {
    if (dim !== undefined) return minDim(tensor, dim);
    let acc = Infinity;
    for (const value of tensor.data) acc = Math.min(acc, value);
    const result = scalarTensor(acc, gradModeEnabled() && tensor.requiresGrad);
    if (gradModeEnabled() && tensor.requiresGrad) {
      let count = 0;
      for (const value of tensor.data) if (value === acc) count += 1;
      result._prev = [tensor];
      result._backward = (grad: Float32Array | null) => {
        if (!grad) return;
        const inGrad = new Float32Array(tensor.length);
        for (let i = 0; i < tensor.length; i += 1) {
          if (tensor.data[i] === acc) inGrad[i] = grad[0] / count;
        }
        addTensorGrad(tensor, inGrad);
      };
    }
    return result;
  }

  function any(tensor: TensorMathTensor, dim?: number) {
    if (dim !== undefined) return anyDim(tensor, dim);
    for (const value of tensor.data) {
      if (value !== 0) return scalarTensor(1, false);
    }
    return scalarTensor(0, false);
  }

  function all(tensor: TensorMathTensor, dim?: number) {
    if (dim !== undefined) return allDim(tensor, dim);
    for (const value of tensor.data) {
      if (value === 0) return scalarTensor(0, false);
    }
    return scalarTensor(1, false);
  }

  function mean(tensor: TensorMathTensor, dim?: number) {
    if (dim !== undefined) return meanDim(tensor, dim);
    const result = div(sum(tensor), scalarTensor(tensor.data.length));
    result.shape = Object.freeze([1]);
    return result;
  }

  function prod(tensor: TensorMathTensor, dim?: number) {
    if (dim !== undefined) return prodDim(tensor, dim);
    let acc = 1;
    let nonZeroProduct = 1;
    let zeroCount = 0;
    for (const value of tensor.data) {
      acc *= value;
      if (value === 0) zeroCount += 1;
      else nonZeroProduct *= value;
    }
    const result = scalarTensor(acc, gradModeEnabled() && tensor.requiresGrad);
    if (gradModeEnabled() && tensor.requiresGrad) {
      result._prev = [tensor];
      result._backward = (grad: Float32Array | null) => {
        if (!grad) return;
        const inGrad = new Float32Array(tensor.length);
        for (let i = 0; i < tensor.length; i += 1) {
          const value = tensor.data[i];
          if (zeroCount === 0) inGrad[i] = grad[0] * acc / value;
          else if (zeroCount === 1 && value === 0) inGrad[i] = grad[0] * nonZeroProduct;
        }
        addTensorGrad(tensor, inGrad);
      };
    }
    return result;
  }

  function argmax(tensor: TensorMathTensor, dim?: number) {
    if (dim !== undefined) return argmaxDim(tensor, dim);
    let bestIndex = 0;
    let bestValue = -Infinity;
    for (let i = 0; i < tensor.length; i += 1) {
      const value = tensor.data[i];
      if (value > bestValue) {
        bestValue = value;
        bestIndex = i;
      }
    }
    return scalarTensor(bestIndex, false);
  }

  function argmin(tensor: TensorMathTensor, dim?: number) {
    if (dim !== undefined) return argminDim(tensor, dim);
    let bestIndex = 0;
    let bestValue = Infinity;
    for (let i = 0; i < tensor.length; i += 1) {
      const value = tensor.data[i];
      if (value < bestValue) {
        bestValue = value;
        bestIndex = i;
      }
    }
    return scalarTensor(bestIndex, false);
  }

  return Object.freeze({
    binary,
    compare,
    where,
    add,
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    isclose,
    sub,
    mul,
    div,
    maximum,
    minimum,
    sqr,
    square,
    pow,
    recip,
    reciprocal,
    matmul,
    mm: matmul,
    dot,
    trace,
    diagonal,
    bmm,
    map,
    unary,
    relu,
    gelu,
    silu,
    sigmoid,
    tanh,
    sin,
    cos,
    tan,
    exp,
    expm1,
    log,
    log1p,
    neg,
    negative,
    abs,
    sgn,
    sign,
    step,
    isnan,
    isinf,
    isfinite,
    floor,
    ceil,
    round,
    trunc,
    sqrt,
    rsqrt,
    clamp,
    clip,
    sumDim,
    meanDim,
    prodDim,
    maxDim,
    minDim,
    argmaxDim,
    argminDim,
    anyDim,
    allDim,
    softmax,
    softmaxDim,
    logSoftmax,
    logSoftmaxDim,
    logsumexp,
    logSumExp,
    logsumexpDim,
    variance,
    var: var_,
    std,
    norm,
    normDim,
    cumsum,
    varianceDim,
    stdDim,
    sum,
    max,
    min,
    maskedFill,
    masked_fill,
    any,
    all,
    mean,
    prod,
    argmax,
    argmin,
  });
}

export type TensorMathSurfaceHelpersOptions<TTensor> = Readonly<{
  tensorMathHelpers: AnyRecord;
  meanSquaredError: (tensor: TTensor, target: unknown) => TTensor;
}>;

export function createTensorMathSurfaceHelpers<TTensor>(options: TensorMathSurfaceHelpersOptions<TTensor>) {
  const helpers = options.tensorMathHelpers;
  const meanSquaredError = options.meanSquaredError;
  return Object.freeze({
    binary: (
      tensor: TTensor,
      other: unknown,
      op: BinaryOp,
      gradLeft: BinaryGrad,
      gradRight: BinaryGrad,
      label: string,
    ) => helpers.binary(tensor, other, op, gradLeft, gradRight, label) as TTensor,
    add: (tensor: TTensor, other: unknown) => helpers.add(tensor, other) as TTensor,
    sub: (tensor: TTensor, other: unknown) => helpers.sub(tensor, other) as TTensor,
    mul: (tensor: TTensor, other: unknown) => helpers.mul(tensor, other) as TTensor,
    div: (tensor: TTensor, other: unknown) => helpers.div(tensor, other) as TTensor,
    maximum: (tensor: TTensor, other: unknown) => helpers.maximum(tensor, other) as TTensor,
    minimum: (tensor: TTensor, other: unknown) => helpers.minimum(tensor, other) as TTensor,
    where: (tensor: TTensor, input: unknown, other: unknown) => helpers.where(tensor, input, other) as TTensor,
    maskedFill: (tensor: TTensor, mask: unknown, value: unknown) => helpers.maskedFill(tensor, mask, value) as TTensor,
    masked_fill: (tensor: TTensor, mask: unknown, value: unknown) => helpers.masked_fill(tensor, mask, value) as TTensor,
    eq: (tensor: TTensor, other: unknown) => helpers.eq(tensor, other) as TTensor,
    ne: (tensor: TTensor, other: unknown) => helpers.ne(tensor, other) as TTensor,
    lt: (tensor: TTensor, other: unknown) => helpers.lt(tensor, other) as TTensor,
    le: (tensor: TTensor, other: unknown) => helpers.le(tensor, other) as TTensor,
    gt: (tensor: TTensor, other: unknown) => helpers.gt(tensor, other) as TTensor,
    ge: (tensor: TTensor, other: unknown) => helpers.ge(tensor, other) as TTensor,
    isclose: (tensor: TTensor, other: unknown, options?: IsCloseOptions) => helpers.isclose(tensor, other, options) as TTensor,
    sqr: (tensor: TTensor) => helpers.sqr(tensor) as TTensor,
    square: (tensor: TTensor) => helpers.square(tensor) as TTensor,
    pow: (tensor: TTensor, exponent: number) => helpers.pow(tensor, exponent) as TTensor,
    recip: (tensor: TTensor) => helpers.recip(tensor) as TTensor,
    reciprocal: (tensor: TTensor) => helpers.reciprocal(tensor) as TTensor,
    abs: (tensor: TTensor) => helpers.abs(tensor) as TTensor,
    sgn: (tensor: TTensor) => helpers.sgn(tensor) as TTensor,
    sign: (tensor: TTensor) => helpers.sign(tensor) as TTensor,
    step: (tensor: TTensor) => helpers.step(tensor) as TTensor,
    isnan: (tensor: TTensor) => helpers.isnan(tensor) as TTensor,
    isinf: (tensor: TTensor) => helpers.isinf(tensor) as TTensor,
    isfinite: (tensor: TTensor) => helpers.isfinite(tensor) as TTensor,
    floor: (tensor: TTensor) => helpers.floor(tensor) as TTensor,
    ceil: (tensor: TTensor) => helpers.ceil(tensor) as TTensor,
    round: (tensor: TTensor) => helpers.round(tensor) as TTensor,
    trunc: (tensor: TTensor) => helpers.trunc(tensor) as TTensor,
    sqrt: (tensor: TTensor) => helpers.sqrt(tensor) as TTensor,
    rsqrt: (tensor: TTensor) => helpers.rsqrt(tensor) as TTensor,
    clamp: (tensor: TTensor, min?: number | null, max?: number | null) => helpers.clamp(tensor, min, max) as TTensor,
    clip: (tensor: TTensor, min?: number | null, max?: number | null) => helpers.clip(tensor, min, max) as TTensor,
    matmul: (tensor: TTensor, other: unknown, otherShape?: readonly number[]) => helpers.matmul(tensor, other, otherShape) as TTensor,
    mm: (tensor: TTensor, other: unknown, otherShape?: readonly number[]) => helpers.mm(tensor, other, otherShape) as TTensor,
    dot: (tensor: TTensor, other: unknown, otherShape?: readonly number[]) => helpers.dot(tensor, other, otherShape) as TTensor,
    trace: (tensor: TTensor) => helpers.trace(tensor) as TTensor,
    diagonal: (tensor: TTensor) => helpers.diagonal(tensor) as TTensor,
    bmm: (tensor: TTensor, other: unknown, otherShape?: readonly number[]) => helpers.bmm(tensor, other, otherShape) as TTensor,
    map: (tensor: TTensor, fn: UnaryOp) => helpers.map(tensor, fn) as TTensor,
    unary: (tensor: TTensor, fn: (value: number) => number, derivative: (value: number) => number) => (
      helpers.unary(tensor, fn, derivative) as TTensor
    ),
    relu: (tensor: TTensor) => helpers.relu(tensor) as TTensor,
    gelu: (tensor: TTensor) => helpers.gelu(tensor) as TTensor,
    silu: (tensor: TTensor) => helpers.silu(tensor) as TTensor,
    sigmoid: (tensor: TTensor) => helpers.sigmoid(tensor) as TTensor,
    tanh: (tensor: TTensor) => helpers.tanh(tensor) as TTensor,
    sin: (tensor: TTensor) => helpers.sin(tensor) as TTensor,
    cos: (tensor: TTensor) => helpers.cos(tensor) as TTensor,
    tan: (tensor: TTensor) => helpers.tan(tensor) as TTensor,
    exp: (tensor: TTensor) => helpers.exp(tensor) as TTensor,
    expm1: (tensor: TTensor) => helpers.expm1(tensor) as TTensor,
    log: (tensor: TTensor) => helpers.log(tensor) as TTensor,
    log1p: (tensor: TTensor) => helpers.log1p(tensor) as TTensor,
    neg: (tensor: TTensor) => helpers.neg(tensor) as TTensor,
    negative: (tensor: TTensor) => helpers.negative(tensor) as TTensor,
    sumDim: (tensor: TTensor, dim: number) => helpers.sumDim(tensor, dim) as TTensor,
    meanDim: (tensor: TTensor, dim: number) => helpers.meanDim(tensor, dim) as TTensor,
    prodDim: (tensor: TTensor, dim: number) => helpers.prodDim(tensor, dim) as TTensor,
    maxDim: (tensor: TTensor, dim: number) => helpers.maxDim(tensor, dim) as TTensor,
    minDim: (tensor: TTensor, dim: number) => helpers.minDim(tensor, dim) as TTensor,
    argmaxDim: (tensor: TTensor, dim: number) => helpers.argmaxDim(tensor, dim) as TTensor,
    argminDim: (tensor: TTensor, dim: number) => helpers.argminDim(tensor, dim) as TTensor,
    anyDim: (tensor: TTensor, dim: number) => helpers.anyDim(tensor, dim) as TTensor,
    allDim: (tensor: TTensor, dim: number) => helpers.allDim(tensor, dim) as TTensor,
    softmax: (tensor: TTensor) => helpers.softmax(tensor) as TTensor,
    softmaxDim: (tensor: TTensor, dim: number) => helpers.softmaxDim(tensor, dim) as TTensor,
    logSoftmax: (tensor: TTensor) => helpers.logSoftmax(tensor) as TTensor,
    logSoftmaxDim: (tensor: TTensor, dim: number) => helpers.logSoftmaxDim(tensor, dim) as TTensor,
    logsumexp: (tensor: TTensor, dim?: number) => helpers.logsumexp(tensor, dim) as TTensor,
    logSumExp: (tensor: TTensor, dim?: number) => helpers.logSumExp(tensor, dim) as TTensor,
    logsumexpDim: (tensor: TTensor, dim: number) => helpers.logsumexpDim(tensor, dim) as TTensor,
    variance: (tensor: TTensor, dim?: number, correction?: number) => helpers.variance(tensor, dim, correction) as TTensor,
    var: (tensor: TTensor, dim?: number, correction?: number) => helpers.var(tensor, dim, correction) as TTensor,
    std: (tensor: TTensor, dim?: number, correction?: number) => helpers.std(tensor, dim, correction) as TTensor,
    norm: (tensor: TTensor, dim?: number, p?: number) => helpers.norm(tensor, dim, p) as TTensor,
    normDim: (tensor: TTensor, dim: number, p?: number) => helpers.normDim(tensor, dim, p) as TTensor,
    varianceDim: (tensor: TTensor, dim: number, correction?: number) => helpers.varianceDim(tensor, dim, correction) as TTensor,
    stdDim: (tensor: TTensor, dim: number, correction?: number) => helpers.stdDim(tensor, dim, correction) as TTensor,
    sum: (tensor: TTensor, dim?: number) => helpers.sum(tensor, dim) as TTensor,
    prod: (tensor: TTensor, dim?: number) => helpers.prod(tensor, dim) as TTensor,
    cumsum: (tensor: TTensor, dim?: number) => helpers.cumsum(tensor, dim) as TTensor,
    max: (tensor: TTensor, dim?: number) => helpers.max(tensor, dim) as TTensor,
    min: (tensor: TTensor, dim?: number) => helpers.min(tensor, dim) as TTensor,
    any: (tensor: TTensor, dim?: number) => helpers.any(tensor, dim) as TTensor,
    all: (tensor: TTensor, dim?: number) => helpers.all(tensor, dim) as TTensor,
    mean: (tensor: TTensor, dim?: number) => helpers.mean(tensor, dim) as TTensor,
    argmax: (tensor: TTensor, dim?: number) => helpers.argmax(tensor, dim) as TTensor,
    argmin: (tensor: TTensor, dim?: number) => helpers.argmin(tensor, dim) as TTensor,
    meanSquaredError,
  });
}
