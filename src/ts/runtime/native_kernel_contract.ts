"use strict";

import {
  moduleActivationIds,
  moduleOpIds,
} from "./abi.js";
import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";

export type NativeModuleOpDesc = Readonly<{
  kind: number;
  activation?: number;
  flags?: number;
  reserved?: number;
  a?: number;
  b?: number;
  c?: number;
  eps?: number;
  readonly [key: string]: unknown;
}>;
export type NativeModuleOpDescAbiFields = Readonly<{
  kind: number;
  activation: number;
  flags: number;
  reserved: number;
  a: number;
  b: number;
  c: number;
  eps: number;
}>;

const moduleActivationNamesById = Object.freeze(Object.fromEntries(
  Object.entries(moduleActivationIds).map(([name, id]) => [id, name]),
));

export function kernelNameForNativeModuleDesc(desc: NativeModuleOpDesc | null | undefined): string {
  switch (desc?.kind) {
    case moduleOpIds.linear: return "linear";
    case moduleOpIds.activation: return moduleActivationNamesById[desc.activation ?? 0] ?? "activation";
    case moduleOpIds.activationChain: return "activation-chain";
    case moduleOpIds.softmax: return "softmax";
    case moduleOpIds.logSoftmax: return "log-softmax";
    case moduleOpIds.reduceSum: return "sum";
    case moduleOpIds.reduceMean: return "mean";
    case moduleOpIds.reduceMax: return "max";
    case moduleOpIds.reduceMin: return "min";
    case moduleOpIds.reshape: return "reshape";
    case moduleOpIds.broadcastTo: return "broadcast";
    case moduleOpIds.narrow: return "narrow";
    case moduleOpIds.slice: return "slice";
    case moduleOpIds.transpose: return "transpose";
    case moduleOpIds.layerNorm: return "layer-norm";
    case moduleOpIds.rmsNorm: return "rms-norm";
    case moduleOpIds.embedding: return "embedding";
    case moduleOpIds.maxPool2d: return "max-pool2d";
    case moduleOpIds.avgPool2d: return "avg-pool2d";
    case moduleOpIds.conv2d: return "conv2d";
    case moduleOpIds.add: return "add";
    case moduleOpIds.featureAffine: return "affine";
    default: return "unknown";
  }
}

export function kernelNamesForNativeModuleDescs(descs: readonly NativeModuleOpDesc[]): readonly string[] {
  return Object.freeze(descs.map(kernelNameForNativeModuleDesc));
}

export function nativeModuleDescSignature(desc: NativeModuleOpDesc | null | undefined): string {
  return desc
    ? [
        desc.kind ?? "",
        desc.activation ?? 0,
        desc.flags ?? 0,
        desc.reserved ?? 0,
        desc.a ?? 0,
        desc.b ?? 0,
        desc.c ?? 0,
        desc.eps ?? 0,
      ].join(":")
    : "";
}

export function nativeModuleDescSignatures(descs: readonly NativeModuleOpDesc[]): readonly string[] {
  return Object.freeze(descs.map(nativeModuleDescSignature));
}

function nativeModuleDescUSize(name: string, value: unknown): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${name} must be a non-negative safe integer, got ${value}`);
  }
  return value as number;
}

function nativeModuleDescU32(name: string, value: unknown): number {
  const normalized = nativeModuleDescUSize(name, value);
  if (normalized > 0xffffffff) {
    throw new Error(`${name} must fit uint32_t, got ${value}`);
  }
  return normalized;
}

export function nativeModuleOpDescAbiFields(desc: NativeModuleOpDesc, label = "native module op descriptor"): NativeModuleOpDescAbiFields {
  return Object.freeze({
    kind: nativeModuleDescUSize(`${label}.kind`, desc.kind),
    activation: nativeModuleDescUSize(`${label}.activation`, desc.activation ?? 0),
    flags: nativeModuleDescUSize(`${label}.flags`, desc.flags ?? 0),
    reserved: nativeModuleDescU32(`${label}.reserved`, desc.reserved ?? 0),
    a: nativeModuleDescUSize(`${label}.a`, desc.a ?? 0),
    b: nativeModuleDescUSize(`${label}.b`, desc.b ?? 0),
    c: nativeModuleDescUSize(`${label}.c`, desc.c ?? 0),
    eps: desc.eps ?? 0,
  });
}

export const nativeKernelContractManifest = Object.freeze({
  kind: "zgml-native-kernel-contract",
  ...tsRuntimeManifestPolicy("src/ts/runtime/native_kernel_contract.ts", "KernelPlan -> native descriptor contract -> Program"),
});
