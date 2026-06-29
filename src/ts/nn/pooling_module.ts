import { initializeModuleMode } from "../train/module_mode.js";
import {
  leafChildren,
  leafNamedChildren,
  leafNamedModules,
  moduleListSelf,
} from "./module_tree.js";
import {
  createSingleModuleCompileHooks,
  emptySingleModuleBindings,
  installSingleModuleCompileMethods,
  type SingleModuleCompileHooksInput,
  type SingleModuleCompilePrototype,
} from "./single_module_compile.js";

type UnknownRecord = Record<string, unknown>;

export type PoolTensorConstructOptions = Readonly<Record<string, unknown> & {
  requiresGrad?: boolean;
  requires_grad?: boolean;
  prev?: readonly unknown[];
  backward?: (grad: Float32Array | null) => void;
}>;
export type PoolTensor = {
  readonly data: Float32Array;
  readonly shape: readonly number[];
  readonly length: number;
  readonly grad?: Float32Array | null;
  readonly requiresGrad?: boolean;
  readonly requires_grad?: boolean;
};
export type PoolGradTensor = PoolTensor & {
  readonly grad: Float32Array | null;
  readonly requiresGrad: boolean;
};
type TensorConstructor = new (values: Float32Array, shape?: readonly number[], options?: PoolTensorConstructOptions) => PoolTensor;
type TensorGradAdder = (tensor: PoolGradTensor, grad: Float32Array) => void;
export type NativeEagerPool2dInto = (output: Float32Array, input: PoolTensor, options: Readonly<Record<string, unknown>>) => Float32Array;

function poolGradTensor(tensor: PoolTensor): PoolGradTensor {
  return tensor as PoolGradTensor;
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

function boolOption(value: unknown, label: string, defaultValue = false) {
  if (value === undefined || value === null) return defaultValue;
  if (typeof value === "boolean") return value;
  throw new Error(`${label} must be a boolean`);
}

export type PoolingModuleClassOptions = Readonly<Record<string, unknown> & {
  Tensor: TensorConstructor;
  addTensorGrad: TensorGradAdder;
  isGradEnabled?: () => boolean;
  nativeEagerPool2dInto?: NativeEagerPool2dInto;
} & SingleModuleCompileHooksInput>;

export function createMaxPool2dModuleClass(options: PoolingModuleClassOptions) {
  const TensorClass = options.Tensor;
  const addTensorGrad = options.addTensorGrad;
  const gradModeEnabled = typeof options.isGradEnabled === "function"
    ? options.isGradEnabled as () => boolean
    : () => true;
  const nativeEagerPool2dInto = typeof options.nativeEagerPool2dInto === "function" ? options.nativeEagerPool2dInto : null;
  const compileHooks = createSingleModuleCompileHooks(options, "MaxPool2dModule", { requirePackParameters: false });
  if (
    typeof TensorClass !== "function" ||
    typeof addTensorGrad !== "function"
  ) {
    throw new Error("MaxPool2dModule factory requires tensor, gradient, and compile-support hooks");
  }

  class MaxPool2dModule {
    kind: "maxPool2d";
    kernelSize: readonly [number, number];
    stride: readonly [number, number];
    padding: readonly [number, number];
    dilation: readonly [number, number];
    ceilMode: boolean;
    training?: boolean;

    constructor(kernelSize: unknown, config: UnknownRecord = {}) {
      initializeModuleMode(this);
      this.kind = "maxPool2d";
      this.kernelSize = positivePair(kernelSize, "maxPool2d kernelSize", [1, 1]);
      this.stride = positivePair(config.stride, "maxPool2d stride", this.kernelSize);
      this.padding = pair(config.padding, "maxPool2d padding", [0, 0]);
      this.dilation = positivePair(config.dilation, "maxPool2d dilation", [1, 1]);
      this.ceilMode = boolOption(config.ceilMode ?? config.ceil_mode, "maxPool2d ceilMode", false);
    }

    outputHW(height: number, width: number) {
      const [kh, kw] = this.kernelSize;
      const [sh, sw] = this.stride;
      const [ph, pw] = this.padding;
      const [dh, dw] = this.dilation;
      const output = this.ceilMode ? Math.ceil : Math.floor;
      const outH = output((height + 2 * ph - dh * (kh - 1) - 1) / sh + 1);
      const outW = output((width + 2 * pw - dw * (kw - 1) - 1) / sw + 1);
      if (!Number.isSafeInteger(outH) || !Number.isSafeInteger(outW) || outH <= 0 || outW <= 0) {
        throw new Error(`maxPool2d output spatial shape must be positive, got [${outH},${outW}]`);
      }
      return [outH, outW] as const;
    }

    forward(inputValues: unknown) {
      const input = inputValues instanceof TensorClass ? inputValues as PoolTensor : new TensorClass(inputValues as Float32Array);
      const rank = input.shape.length;
      const batched = rank === 4;
      if (!batched && rank !== 3) throw new Error(`maxPool2d input shape must be [channels,height,width] or [batch,channels,height,width], got [${input.shape.join(",")}]`);
      const batch = batched ? input.shape[0]! : 1;
      const channels = batched ? input.shape[1]! : input.shape[0]!;
      const height = batched ? input.shape[2]! : input.shape[1]!;
      const width = batched ? input.shape[3]! : input.shape[2]!;
      const [outH, outW] = this.outputHW(height, width);
      const [kh, kw] = this.kernelSize;
      const [sh, sw] = this.stride;
      const [ph, pw] = this.padding;
      const [dh, dw] = this.dilation;
      const outShape = batched ? [batch, channels, outH, outW] : [channels, outH, outW];
      const out = new Float32Array(batch * channels * outH * outW);
      const gradEnabled = gradModeEnabled();
      const needsGrad = gradEnabled && Boolean(input.requiresGrad || input.requires_grad);
      let maxIndices: Int32Array | null = null;
      if (!needsGrad && nativeEagerPool2dInto) {
        nativeEagerPool2dInto(out, input, {
          op: "max",
          batch,
          channels,
          height,
          width,
          kernelH: kh,
          kernelW: kw,
          strideH: sh,
          strideW: sw,
          paddingH: ph,
          paddingW: pw,
          dilationH: dh,
          dilationW: dw,
          outH,
          outW,
          ceilMode: this.ceilMode,
          countIncludePad: true,
        });
      } else {
        maxIndices = new Int32Array(out.length);
        maxIndices.fill(-1);
        for (let n = 0; n < batch; n += 1) {
          for (let c = 0; c < channels; c += 1) {
            for (let oh = 0; oh < outH; oh += 1) {
              for (let ow = 0; ow < outW; ow += 1) {
                let max = -Infinity;
                let maxIndex = -1;
                for (let ky = 0; ky < kh; ky += 1) {
                  const ih = oh * sh + ky * dh - ph;
                  if (ih < 0 || ih >= height) continue;
                  for (let kx = 0; kx < kw; kx += 1) {
                    const iw = ow * sw + kx * dw - pw;
                    if (iw < 0 || iw >= width) continue;
                    const inputIndex = batched
                      ? ((n * channels + c) * height + ih) * width + iw
                      : (c * height + ih) * width + iw;
                    const value = input.data[inputIndex]!;
                    if (value > max) {
                      max = value;
                      maxIndex = inputIndex;
                    }
                  }
                }
                const outIndex = ((n * channels + c) * outH + oh) * outW + ow;
                out[outIndex] = max;
                maxIndices[outIndex] = maxIndex;
              }
            }
          }
        }
      }
      return new TensorClass(out, outShape, {
        requiresGrad: needsGrad,
        prev: needsGrad ? [input] : [],
        backward: (grad: Float32Array | null | undefined) => {
          if (!grad || !needsGrad) return;
          if (!maxIndices) throw new Error("maxPool2d backward requires forward max indices");
          const inputGrad = new Float32Array(input.length);
          for (let i = 0; i < grad.length; i += 1) {
            const inputIndex = maxIndices[i]!;
            if (inputIndex >= 0) inputGrad[inputIndex] += grad[i]!;
          }
          addTensorGrad(poolGradTensor(input), inputGrad);
        },
      });
    }

    parameters() {
      return [];
    }

    namedParameters() {
      return [];
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

    stateDict() {
      return Object.freeze({});
    }

    loadStateDict(source: unknown) {
      const keys = source && typeof source === "object" ? Object.keys(source as Record<string, unknown>) : [];
      if (keys.length !== 0) throw new Error(`maxPool2d loadStateDict expected no state entries, got ${keys.join(",")}`);
      return this;
    }
  }

  installSingleModuleCompileMethods(MaxPool2dModule.prototype as SingleModuleCompilePrototype, compileHooks, {
    bindParameters: emptySingleModuleBindings,
    fallbackReason: "nn.MaxPool2d is outside the native Program compiler subset for this input shape/config",
  });
  return MaxPool2dModule;
}

export function createAvgPool2dModuleClass(options: PoolingModuleClassOptions) {
  const TensorClass = options.Tensor;
  const addTensorGrad = options.addTensorGrad;
  const gradModeEnabled = typeof options.isGradEnabled === "function"
    ? options.isGradEnabled as () => boolean
    : () => true;
  const nativeEagerPool2dInto = typeof options.nativeEagerPool2dInto === "function" ? options.nativeEagerPool2dInto : null;
  const compileHooks = createSingleModuleCompileHooks(options, "AvgPool2dModule", { requirePackParameters: false });
  if (
    typeof TensorClass !== "function" ||
    typeof addTensorGrad !== "function"
  ) {
    throw new Error("AvgPool2dModule factory requires tensor, gradient, and compile-support hooks");
  }

  class AvgPool2dModule {
    kind: "avgPool2d";
    kernelSize: readonly [number, number];
    stride: readonly [number, number];
    padding: readonly [number, number];
    ceilMode: boolean;
    countIncludePad: boolean;
    training?: boolean;

    constructor(kernelSize: unknown, config: UnknownRecord = {}) {
      initializeModuleMode(this);
      this.kind = "avgPool2d";
      this.kernelSize = positivePair(kernelSize, "avgPool2d kernelSize", [1, 1]);
      this.stride = positivePair(config.stride, "avgPool2d stride", this.kernelSize);
      this.padding = pair(config.padding, "avgPool2d padding", [0, 0]);
      this.ceilMode = boolOption(config.ceilMode ?? config.ceil_mode, "avgPool2d ceilMode", false);
      this.countIncludePad = boolOption(config.countIncludePad ?? config.count_include_pad, "avgPool2d countIncludePad", true);
    }

    outputHW(height: number, width: number) {
      const [kh, kw] = this.kernelSize;
      const [sh, sw] = this.stride;
      const [ph, pw] = this.padding;
      const output = this.ceilMode ? Math.ceil : Math.floor;
      const outH = output((height + 2 * ph - kh) / sh + 1);
      const outW = output((width + 2 * pw - kw) / sw + 1);
      if (!Number.isSafeInteger(outH) || !Number.isSafeInteger(outW) || outH <= 0 || outW <= 0) {
        throw new Error(`avgPool2d output spatial shape must be positive, got [${outH},${outW}]`);
      }
      return [outH, outW] as const;
    }

    forward(inputValues: unknown) {
      const input = inputValues instanceof TensorClass ? inputValues as PoolTensor : new TensorClass(inputValues as Float32Array);
      const rank = input.shape.length;
      const batched = rank === 4;
      if (!batched && rank !== 3) throw new Error(`avgPool2d input shape must be [channels,height,width] or [batch,channels,height,width], got [${input.shape.join(",")}]`);
      const batch = batched ? input.shape[0]! : 1;
      const channels = batched ? input.shape[1]! : input.shape[0]!;
      const height = batched ? input.shape[2]! : input.shape[1]!;
      const width = batched ? input.shape[3]! : input.shape[2]!;
      const [outH, outW] = this.outputHW(height, width);
      const [kh, kw] = this.kernelSize;
      const [sh, sw] = this.stride;
      const [ph, pw] = this.padding;
      const outShape = batched ? [batch, channels, outH, outW] : [channels, outH, outW];
      const out = new Float32Array(batch * channels * outH * outW);
      const gradEnabled = gradModeEnabled();
      const needsGrad = gradEnabled && Boolean(input.requiresGrad || input.requires_grad);
      let counts: Float32Array | null = null;
      if (!needsGrad && nativeEagerPool2dInto) {
        nativeEagerPool2dInto(out, input, {
          op: "avg",
          batch,
          channels,
          height,
          width,
          kernelH: kh,
          kernelW: kw,
          strideH: sh,
          strideW: sw,
          paddingH: ph,
          paddingW: pw,
          dilationH: 1,
          dilationW: 1,
          outH,
          outW,
          ceilMode: this.ceilMode,
          countIncludePad: this.countIncludePad,
        });
      } else {
        counts = new Float32Array(out.length);
        for (let n = 0; n < batch; n += 1) {
          for (let c = 0; c < channels; c += 1) {
            for (let oh = 0; oh < outH; oh += 1) {
              for (let ow = 0; ow < outW; ow += 1) {
                let sum = 0;
                let count = this.countIncludePad ? kh * kw : 0;
                for (let ky = 0; ky < kh; ky += 1) {
                  const ih = oh * sh + ky - ph;
                  if (ih < 0 || ih >= height) continue;
                  for (let kx = 0; kx < kw; kx += 1) {
                    const iw = ow * sw + kx - pw;
                    if (iw < 0 || iw >= width) continue;
                    const inputIndex = batched
                      ? ((n * channels + c) * height + ih) * width + iw
                      : (c * height + ih) * width + iw;
                    sum += input.data[inputIndex]!;
                    if (!this.countIncludePad) count += 1;
                  }
                }
                const outIndex = ((n * channels + c) * outH + oh) * outW + ow;
                counts[outIndex] = count;
                out[outIndex] = count === 0 ? 0 : sum / count;
              }
            }
          }
        }
      }
      return new TensorClass(out, outShape, {
        requiresGrad: needsGrad,
        prev: needsGrad ? [input] : [],
        backward: (grad: Float32Array | null | undefined) => {
          if (!grad || !needsGrad) return;
          if (!counts) throw new Error("avgPool2d backward requires forward counts");
          const inputGrad = new Float32Array(input.length);
          for (let n = 0; n < batch; n += 1) {
            for (let c = 0; c < channels; c += 1) {
              for (let oh = 0; oh < outH; oh += 1) {
                for (let ow = 0; ow < outW; ow += 1) {
                  const outIndex = ((n * channels + c) * outH + oh) * outW + ow;
                  const scale = counts[outIndex] === 0 ? 0 : grad[outIndex]! / counts[outIndex]!;
                  for (let ky = 0; ky < kh; ky += 1) {
                    const ih = oh * sh + ky - ph;
                    if (ih < 0 || ih >= height) continue;
                    for (let kx = 0; kx < kw; kx += 1) {
                      const iw = ow * sw + kx - pw;
                      if (iw < 0 || iw >= width) continue;
                      const inputIndex = batched
                        ? ((n * channels + c) * height + ih) * width + iw
                        : (c * height + ih) * width + iw;
                      inputGrad[inputIndex] += scale;
                    }
                  }
                }
              }
            }
          }
          addTensorGrad(poolGradTensor(input), inputGrad);
        },
      });
    }

    parameters() {
      return [];
    }

    namedParameters() {
      return [];
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

    stateDict() {
      return Object.freeze({});
    }

    loadStateDict(source: unknown) {
      const keys = source && typeof source === "object" ? Object.keys(source as Record<string, unknown>) : [];
      if (keys.length !== 0) throw new Error(`avgPool2d loadStateDict expected no state entries, got ${keys.join(",")}`);
      return this;
    }
  }

  installSingleModuleCompileMethods(AvgPool2dModule.prototype as SingleModuleCompilePrototype, compileHooks, {
    bindParameters: emptySingleModuleBindings,
    fallbackReason: "nn.AvgPool2d is outside the native Program compiler subset for this input shape/config",
  });
  return AvgPool2dModule;
}
