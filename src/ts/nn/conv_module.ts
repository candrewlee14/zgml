import { initializeModuleMode } from "../train/module_mode.js";
import { annotateModuleBindings } from "../runtime/module_bindings.js";
import {
  leafChildren,
  leafNamedChildren,
  leafNamedModules,
  moduleListSelf,
  moduleTraversalOptions,
} from "./module_tree.js";
import {
  createStatefulModuleStateHooks,
  installStatefulModuleStateMethods,
} from "./stateful_module_state.js";
import {
  createSingleModuleCompileHooks,
  installSingleModuleCompileMethods,
  type SingleModuleCompileHooks,
  type SingleModuleCompileHooksInput,
  type SingleModuleCompilePrototype,
} from "./single_module_compile.js";
import type { CompileOptions } from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
type BivariantCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];

export type ConvTensorConstructOptions = Readonly<Record<string, unknown> & {
  requiresGrad?: boolean;
  requires_grad?: boolean;
  prev?: readonly unknown[];
  backward?: (grad: Float32Array | null) => void;
}>;
export type ConvTensor = {
  readonly data: Float32Array;
  readonly shape: readonly number[];
  readonly length: number;
  readonly rank?: number;
  readonly grad?: Float32Array | null;
  readonly requiresGrad?: boolean;
  readonly requires_grad?: boolean;
  zeroGrad(options?: Readonly<Record<string, unknown>>): void;
};
export type ConvGradTensor = ConvTensor & {
  readonly grad: Float32Array | null;
  readonly requiresGrad: boolean;
};
export type ConvParameter = {
  readonly name: string;
  readonly data: Float32Array;
  readonly grad?: Float32Array | null;
  readonly layout?: string;
  tensor: ConvTensor;
  readonly requiresGrad?: boolean;
  readonly requires_grad?: boolean;
};
type TensorConstructor = new (values: Float32Array, shape?: readonly number[], options?: ConvTensorConstructOptions) => ConvTensor;
type TensorGradAdder = (tensor: ConvGradTensor, grad: Float32Array) => void;
type RequirePositiveInteger = BivariantCallback<[value: unknown, label: string], number>;
type DefaultedF32 = (values: unknown, length: number, label: string, fallback: (length: number) => Float32Array, shape?: readonly number[]) => Float32Array;
type ParameterFactory = (name: string, values: Float32Array, shape: readonly number[], layout: string) => ConvParameter;
type ParameterView = BivariantCallback<[prefix: string, parameter: ConvParameter], ConvParameter>;

function convGradTensor(tensor: ConvTensor): ConvGradTensor {
  return tensor as ConvGradTensor;
}

function onesF32(length: number) {
  const values = new Float32Array(length);
  values.fill(1);
  return values;
}

function pair(value: unknown, label: string, defaultValue: readonly [number, number]) {
  if (value === undefined || value === null) return defaultValue;
  if (Number.isSafeInteger(value) && (value as number) >= 0) return [value as number, value as number] as const;
  if (Array.isArray(value) && value.length === 2 && value.every((dim) => Number.isSafeInteger(dim) && dim >= 0)) {
    return [value[0] as number, value[1] as number] as const;
  }
  throw new Error(`${label} must be a non-negative integer or [height, width] pair`);
}

function positivePair(value: unknown, label: string, defaultValue: readonly [number, number]) {
  const result = pair(value, label, defaultValue);
  if (result[0] <= 0 || result[1] <= 0) throw new Error(`${label} dimensions must be positive`);
  return result;
}

export type Conv2dModuleClassOptions = Readonly<Record<string, unknown> & {
  Tensor: TensorConstructor;
  addTensorGrad: TensorGradAdder;
  requirePositiveInteger: RequirePositiveInteger;
  defaultedF32: DefaultedF32;
  zerosF32: (length: number) => Float32Array;
  makeParameter: ParameterFactory;
  parameterView: ParameterView;
} & SingleModuleCompileHooksInput>;

export function createConv2dModuleClass(options: Conv2dModuleClassOptions) {
  const TensorClass = options.Tensor;
  const addTensorGrad = options.addTensorGrad;
  const gradModeEnabled = typeof options.isGradEnabled === "function"
    ? options.isGradEnabled as () => boolean
    : () => true;
  const requirePositiveInteger = options.requirePositiveInteger;
  const defaultedF32 = options.defaultedF32;
  const zerosF32 = options.zerosF32;
  const makeParameter = options.makeParameter;
  const parameterView = options.parameterView;
  const compileHooks = createSingleModuleCompileHooks(options, "Conv2dModule", { requirePackParameters: false });
  const stateHooks = createStatefulModuleStateHooks(options, "Conv2dModule");
  if (
    typeof TensorClass !== "function" ||
    typeof addTensorGrad !== "function" ||
    typeof requirePositiveInteger !== "function" ||
    typeof defaultedF32 !== "function" ||
    typeof zerosF32 !== "function" ||
    typeof makeParameter !== "function" ||
    typeof parameterView !== "function"
  ) {
    throw new Error("Conv2dModule factory requires tensor, gradient, state, parameter, and compile-support hooks");
  }

  class Conv2dModule {
    kind: "conv2d";
    inChannels: number;
    outChannels: number;
    kernelSize: readonly [number, number];
    stride: readonly [number, number];
    padding: readonly [number, number];
    dilation: readonly [number, number];
    groups: number;
    weight: Float32Array;
    bias: Float32Array | null;
    weightParam: ConvParameter;
    biasParam: ConvParameter | null;
    training?: boolean;

    constructor(inChannels: unknown, outChannels: unknown, kernelSize: unknown, config: UnknownRecord = {}) {
      initializeModuleMode(this);
      this.kind = "conv2d";
      this.inChannels = requirePositiveInteger(inChannels, "conv2d inChannels");
      this.outChannels = requirePositiveInteger(outChannels, "conv2d outChannels");
      this.kernelSize = positivePair(kernelSize, "conv2d kernelSize", [1, 1]);
      this.stride = positivePair(config.stride, "conv2d stride", [1, 1]);
      this.padding = pair(config.padding, "conv2d padding", [0, 0]);
      this.dilation = positivePair(config.dilation, "conv2d dilation", [1, 1]);
      this.groups = requirePositiveInteger(config.groups ?? 1, "conv2d groups");
      if (this.groups !== 1) throw new Error("conv2d currently supports groups=1");
      const [kh, kw] = this.kernelSize;
      this.weight = defaultedF32(
        config.weight ?? config.weights,
        this.outChannels * this.inChannels * kh * kw,
        "conv2d weight",
        onesF32,
        [this.outChannels, this.inChannels, kh, kw],
      );
      this.bias = config.bias === false
        ? null
        : defaultedF32(
          config.biasValues ?? (config.bias === undefined || config.bias === true ? undefined : config.bias),
          this.outChannels,
          "conv2d bias",
          zerosF32,
          [this.outChannels],
        );
      this.weightParam = makeParameter("weight", this.weight, [this.outChannels, this.inChannels, kh, kw], "row-major:conv2d.weight[out_channels,in_channels,kernel_h,kernel_w]");
      this.biasParam = this.bias ? makeParameter("bias", this.bias, [this.outChannels], "row-major:conv2d.bias[out_channels]") : null;
    }

    outputHW(height: number, width: number) {
      const [kh, kw] = this.kernelSize;
      const [sh, sw] = this.stride;
      const [ph, pw] = this.padding;
      const [dh, dw] = this.dilation;
      const outH = Math.floor((height + 2 * ph - dh * (kh - 1) - 1) / sh + 1);
      const outW = Math.floor((width + 2 * pw - dw * (kw - 1) - 1) / sw + 1);
      if (!Number.isSafeInteger(outH) || !Number.isSafeInteger(outW) || outH <= 0 || outW <= 0) {
        throw new Error(`conv2d output spatial shape must be positive, got [${outH},${outW}]`);
      }
      return [outH, outW] as const;
    }

    forward(inputValues: unknown) {
      const input = inputValues instanceof TensorClass ? inputValues as ConvTensor : new TensorClass(inputValues as Float32Array);
      const rank = input.shape.length;
      const batched = rank === 4;
      if (!batched && rank !== 3) throw new Error(`conv2d input shape must be [channels,height,width] or [batch,channels,height,width], got [${input.shape.join(",")}]`);
      const batch = batched ? input.shape[0]! : 1;
      const channels = batched ? input.shape[1]! : input.shape[0]!;
      const height = batched ? input.shape[2]! : input.shape[1]!;
      const width = batched ? input.shape[3]! : input.shape[2]!;
      if (channels !== this.inChannels) throw new Error(`conv2d input channels must be ${this.inChannels}, got ${channels}`);
      const [outH, outW] = this.outputHW(height, width);
      const [kh, kw] = this.kernelSize;
      const [sh, sw] = this.stride;
      const [ph, pw] = this.padding;
      const [dh, dw] = this.dilation;
      const outShape = batched ? [batch, this.outChannels, outH, outW] : [this.outChannels, outH, outW];
      const out = new Float32Array(batch * this.outChannels * outH * outW);
      for (let n = 0; n < batch; n += 1) {
        for (let oc = 0; oc < this.outChannels; oc += 1) {
          for (let oh = 0; oh < outH; oh += 1) {
            for (let ow = 0; ow < outW; ow += 1) {
              let sum = this.bias ? this.bias[oc]! : 0;
              for (let ic = 0; ic < this.inChannels; ic += 1) {
                for (let ky = 0; ky < kh; ky += 1) {
                  const ih = oh * sh + ky * dh - ph;
                  if (ih < 0 || ih >= height) continue;
                  for (let kx = 0; kx < kw; kx += 1) {
                    const iw = ow * sw + kx * dw - pw;
                    if (iw < 0 || iw >= width) continue;
                    const inputIndex = batched
                      ? ((n * this.inChannels + ic) * height + ih) * width + iw
                      : (ic * height + ih) * width + iw;
                    const weightIndex = ((oc * this.inChannels + ic) * kh + ky) * kw + kx;
                    sum += input.data[inputIndex]! * this.weight[weightIndex]!;
                  }
                }
              }
              out[((n * this.outChannels + oc) * outH + oh) * outW + ow] = sum;
            }
          }
        }
      }
      const needsGrad = gradModeEnabled() && Boolean(input.requiresGrad || this.weightParam.tensor.requiresGrad || this.biasParam?.tensor.requiresGrad);
      return new TensorClass(out, outShape, {
        requiresGrad: needsGrad,
        prev: needsGrad ? [input, this.weightParam.tensor, ...(this.biasParam ? [this.biasParam.tensor] : [])] : [],
        backward: (grad: Float32Array | null | undefined) => {
          if (!grad) return;
          const inputGrad = input.requiresGrad ? new Float32Array(input.length) : null;
          const weightGrad = this.weightParam.tensor.requiresGrad ? new Float32Array(this.weight.length) : null;
          const biasGrad = this.biasParam?.tensor.requiresGrad ? new Float32Array(this.outChannels) : null;
          for (let n = 0; n < batch; n += 1) {
            for (let oc = 0; oc < this.outChannels; oc += 1) {
              for (let oh = 0; oh < outH; oh += 1) {
                for (let ow = 0; ow < outW; ow += 1) {
                  const g = grad[((n * this.outChannels + oc) * outH + oh) * outW + ow]!;
                  if (biasGrad) biasGrad[oc] += g;
                  for (let ic = 0; ic < this.inChannels; ic += 1) {
                    for (let ky = 0; ky < kh; ky += 1) {
                      const ih = oh * sh + ky * dh - ph;
                      if (ih < 0 || ih >= height) continue;
                      for (let kx = 0; kx < kw; kx += 1) {
                        const iw = ow * sw + kx * dw - pw;
                        if (iw < 0 || iw >= width) continue;
                        const inputIndex = batched
                          ? ((n * this.inChannels + ic) * height + ih) * width + iw
                          : (ic * height + ih) * width + iw;
                        const weightIndex = ((oc * this.inChannels + ic) * kh + ky) * kw + kx;
                        if (inputGrad) inputGrad[inputIndex] += g * this.weight[weightIndex]!;
                        if (weightGrad) weightGrad[weightIndex] += g * input.data[inputIndex]!;
                      }
                    }
                  }
                }
              }
            }
          }
          if (inputGrad) addTensorGrad(convGradTensor(input), inputGrad);
          if (weightGrad) addTensorGrad(convGradTensor(this.weightParam.tensor), weightGrad);
          if (biasGrad && this.biasParam) addTensorGrad(convGradTensor(this.biasParam.tensor), biasGrad);
        },
      });
    }

    parameters(prefixOrOptions: unknown = "", options = {}) {
      const { prefix } = moduleTraversalOptions(prefixOrOptions, options);
      const params = [parameterView(prefix, this.weightParam)];
      if (this.biasParam) params.push(parameterView(prefix, this.biasParam));
      return params;
    }

    namedParameters(prefixOrOptions: unknown = "", options = {}) {
      return this.parameters(prefixOrOptions, options);
    }

    children() {
      return leafChildren();
    }

    modules() {
      return moduleListSelf(this);
    }

    namedChildren() {
      return leafNamedChildren();
    }

    namedModules(prefix = "") {
      return leafNamedModules(this, prefix);
    }
  }

  installStatefulModuleStateMethods(Conv2dModule.prototype, stateHooks);
  installSingleModuleCompileMethods(Conv2dModule.prototype as SingleModuleCompilePrototype, compileHooks, {
    bindParameters(module: UnknownRecord, _hooks: SingleModuleCompileHooks, bindOptions: CompileOptions = {}) {
      return annotateModuleBindings(
        module.bias instanceof Float32Array ? { weights: module.weight, bias: module.bias } : { weights: module.weight },
        module,
        bindOptions,
      );
    },
    fallbackReason: "nn.Conv2d is outside the native Program compiler subset for this input shape/config",
  });
  return Conv2dModule;
}
