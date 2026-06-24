"use strict";

import {
  nativePackageSpineContractManifest,
  requiredNativePackageSpineExports,
  type NativePackageSpineContractManifest,
  type RequiredNativePackageSpineExport,
} from "./native_package_spine_contract.js";
import { frontendManifest } from "../frontend_manifest.js";
import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";

export {
  nativePackageSpineContractManifest,
  requiredNativePackageSpineExports,
  type NativePackageSpineContractManifest,
  type RequiredNativePackageSpineExport,
} from "./native_package_spine_contract.js";

export const nativeApiContractManifest = Object.freeze({
  kind: "zgml-native-api-contract",
  ...tsRuntimeManifestPolicy("src/ts/runtime/native_api_contract.ts", "Program -> Session -> StepParams"),
  productSourceOfTruth: frontendManifest.productSourceOfTruth,
  productSemanticsOwner: frontendManifest.productSemanticsOwner,
  nativeAlignment: frontendManifest.nativeAlignment,
  nativeProductPolicy: frontendManifest.nativeProductPolicy,
  handwrittenFrontendMirrors: frontendManifest.handwrittenFrontendMirrors,
  packageSpine: nativePackageSpineContractManifest,
});

export const requiredNativePackageSpinePrefixExports = Object.freeze([
  "frontendManifest",
  "nodeAdapterEvidence",
  "nativeAdapterEvidence",
  "resolveNativeLibraryPath",
] as const);

export const requiredNativePackageSpineSuffixExports = Object.freeze([
  "program",
  "session",
  "stepParams",
  "compile",
] as const);

export const requiredNativeTensorFactoryExports = Object.freeze([
  "Tensor",
  "tensor",
  "asTensor",
  "as_tensor",
  "asarray",
  "fromNumpy",
  "from_numpy",
  "parameter",
  "param",
  "empty",
  "emptyLike",
  "empty_like",
  "zeros",
  "zerosLike",
  "zeros_like",
  "ones",
  "onesLike",
  "ones_like",
  "eye",
  "full",
  "fullLike",
  "full_like",
  "scalar",
  "rand",
  "randLike",
  "rand_like",
  "randn",
  "randnLike",
  "randn_like",
  "randInt",
  "randint",
  "randPerm",
  "randperm",
  "manualSeed",
  "manual_seed",
  "initialSeed",
  "initial_seed",
  "seededRng",
  "linspace",
  "arange",
  "cat",
  "concat",
  "concatenate",
  "stack",
  "vstack",
  "hstack",
  "einsum",
  "allclose",
  "equal",
] as const);

export const requiredNativeTensorViewExports = Object.freeze([
  "to",
  "cpu",
  "float",
  "float32",
  "typeAs",
  "type_as",
  "clone",
  "detach",
  "reshape",
  "view",
  "broadcastTo",
  "expand",
  "repeat",
  "tile",
  "flatten",
  "squeeze",
  "unsqueeze",
  "transpose",
  "permute",
  "flip",
  "roll",
  "select",
  "narrow",
  "slice",
  "indexSelect",
  "index_select",
  "gather",
  "take",
  "argsort",
  "sort",
  "topk",
  "scatterAdd",
  "scatter_add",
  "split",
  "chunk",
  "unbind",
] as const);

export const requiredNativeTensorMathExports = Object.freeze([
  "add",
  "sub",
  "mul",
  "div",
  "eq",
  "ne",
  "lt",
  "le",
  "gt",
  "ge",
  "pow",
  "neg",
  "negative",
  "exp",
  "expm1",
  "log",
  "log1p",
  "sqr",
  "square",
  "recip",
  "reciprocal",
  "abs",
  "sgn",
  "sign",
  "step",
  "isnan",
  "isinf",
  "isfinite",
  "floor",
  "ceil",
  "round",
  "trunc",
  "sqrt",
  "rsqrt",
  "relu",
  "gelu",
  "silu",
  "sigmoid",
  "tanh",
  "sin",
  "cos",
  "tan",
  "maximum",
  "minimum",
  "where",
  "maskedFill",
  "masked_fill",
  "isclose",
  "sum",
  "prod",
  "cumsum",
  "mean",
  "max",
  "min",
  "any",
  "all",
  "argmax",
  "argmin",
  "variance",
  "std",
  "norm",
  "softmax",
  "softmax_dim",
  "softmaxDim",
  "logSoftmax",
  "log_softmax",
  "log_softmax_dim",
  "logSoftmaxDim",
  "logsumexp",
  "logSumExp",
  "clamp",
  "clip",
  "matmul",
  "mm",
  "dot",
  "trace",
  "diagonal",
  "bmm",
] as const);

export const requiredNativeGradModeExports = Object.freeze([
  "isGradEnabled",
  "is_grad_enabled",
  "setGradEnabled",
  "set_grad_enabled",
  "noGrad",
  "no_grad",
  "inferenceMode",
  "inference_mode",
  "enableGrad",
  "enable_grad",
] as const);

export const requiredNativeFrontendNamespaceExports = Object.freeze([
  "gradMode",
  "zgml",
  "torch",
  "nn",
  "F",
  "data",
  "loss",
  "optim",
  "train",
  "checkpoint",
  "save",
  "load",
  "compile",
] as const);

export const requiredNativeExecutableRuntimeExports = Object.freeze([
  "Program",
  "Session",
  "NativeBuffer",
  "ProgramDevice",
  "TinyLinear",
  "TinyLinearModel",
  "TinyLinearProgram",
  "TinyLinearSession",
  "TinyMlp",
  "TinyMlpModel",
] as const);

export const requiredNativeLlmRuntimeExports = Object.freeze([
  "TinyLlama",
  "TinyLlamaModel",
  "TinyLlamaProgram",
  "TinyLlamaSession",
  "SmolLM135M",
  "SmolLM135MModel",
  "SmolLM135MProgram",
  "SmolLM135MSession",
  "Llama",
  "LlamaModel",
  "LlamaProgram",
  "LlamaSession",
  "LlamaKvCache",
] as const);

export const requiredNativeModelSourceExports = Object.freeze([
  "loadModel",
  "loadSafetensorsData",
  "probeModel",
  "probeSafetensorsData",
  "probeSafetensorsHeader",
  "supportedCheckpointModels",
] as const);

export const requiredNativeInspectionExports = Object.freeze([
  "runtimeInfo",
  "loadedRuntimeInfo",
  "abiStructSize",
  "abiStructSizes",
  "webgpuInterop",
  "ZgmlError",
] as const);

export const requiredNativeApiExports = Object.freeze([
  ...requiredNativeTensorFactoryExports,
  ...requiredNativeTensorViewExports,
  ...requiredNativeTensorMathExports,
  ...requiredNativeGradModeExports,
  ...requiredNativeFrontendNamespaceExports,
  ...requiredNativeExecutableRuntimeExports,
  ...requiredNativeLlmRuntimeExports,
  ...requiredNativeModelSourceExports,
  ...requiredNativeInspectionExports,
] as const);

export const requiredNativeApiSentinelExports = Object.freeze([
  "add",
  "sub",
  "mul",
  "div",
  "tanh",
  "empty",
  "emptyLike",
  "empty_like",
  "zerosLike",
  "zeros_like",
  "onesLike",
  "ones_like",
  "fullLike",
  "full_like",
  "randLike",
  "rand_like",
  "randnLike",
  "randn_like",
  "randInt",
  "randint",
  "randPerm",
  "randperm",
  "eye",
] as const);

export type RequiredNativeApiExport = typeof requiredNativeApiExports[number];
export type RequiredNativeApiSentinelExport = typeof requiredNativeApiSentinelExports[number];
export type RequiredNativePackageSpinePrefixExport = typeof requiredNativePackageSpinePrefixExports[number];
export type RequiredNativePackageSpineSuffixExport = typeof requiredNativePackageSpineSuffixExports[number];
export type RequiredNativeTensorFactoryExport = typeof requiredNativeTensorFactoryExports[number];
export type RequiredNativeTensorViewExport = typeof requiredNativeTensorViewExports[number];
export type RequiredNativeTensorMathExport = typeof requiredNativeTensorMathExports[number];
export type RequiredNativeGradModeExport = typeof requiredNativeGradModeExports[number];
export type RequiredNativeFrontendNamespaceExport = typeof requiredNativeFrontendNamespaceExports[number];
export type RequiredNativeExecutableRuntimeExport = typeof requiredNativeExecutableRuntimeExports[number];
export type RequiredNativeLlmRuntimeExport = typeof requiredNativeLlmRuntimeExports[number];
export type RequiredNativeModelSourceExport = typeof requiredNativeModelSourceExports[number];
export type RequiredNativeInspectionExport = typeof requiredNativeInspectionExports[number];

export type NativeApiContractSignatureParts = Readonly<{
  kind: string;
  source: string;
  api: number;
  spine: number;
  sentinels: number;
}>;

type ExportOwner = object;
type ExportKey<Owner extends ExportOwner> = Extract<keyof Owner, string>;

export function sortedExportKeys<const Owner extends ExportOwner>(owner: Owner): ExportKey<Owner>[] {
  return Object.keys(owner).sort() as ExportKey<Owner>[];
}

export function missingExports<const Key extends string>(owner: ExportOwner, keys: readonly Key[]): Key[] {
  return keys.filter((key) => !(key in owner));
}

export function missingRequiredNativeApiSentinelExports(): RequiredNativeApiSentinelExport[] {
  const nativeExportOwner = Object.fromEntries(requiredNativeApiExports.map((key) => [key, true]));
  return missingExports(nativeExportOwner, requiredNativeApiSentinelExports);
}

export function extraExports<const Left extends ExportOwner, const Right extends ExportOwner>(
  left: Left,
  right: Right,
): Exclude<ExportKey<Left>, ExportKey<Right>>[] {
  const rightKeys = new Set(sortedExportKeys(right));
  return sortedExportKeys(left).filter((key) => !rightKeys.has(key as unknown as ExportKey<Right>)) as Exclude<ExportKey<Left>, ExportKey<Right>>[];
}

export function formatNativeApiContractSignature(parts: NativeApiContractSignatureParts): string {
  return [
    parts.kind,
    `source=${parts.source}`,
    `api=${parts.api}`,
    `spine=${parts.spine}`,
    `sentinels=${parts.sentinels}`,
  ].join("|");
}

export function nativeApiContractSignatureParts(): NativeApiContractSignatureParts {
  return Object.freeze({
    kind: nativeApiContractManifest.kind,
    source: nativeApiContractManifest.source,
    api: requiredNativeApiExports.length,
    spine: requiredNativePackageSpineExports.length,
    sentinels: requiredNativeApiSentinelExports.length,
  });
}

export function nativeApiContractSignature(): string {
  return formatNativeApiContractSignature(nativeApiContractSignatureParts());
}
