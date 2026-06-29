"use strict";

import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";
import type * as PublicApi from "../public_api.js";

export type GenericModelKindAliases = Readonly<{
  tinyLinearKind: number;
  tinyMlpKind: number;
}>;

export type GenericModelNativeDesc = Readonly<{
  kind: number;
  activation: number;
  input_len: number;
  output_len: number;
  hidden_len: number;
}>;

export type GenericModelProgramLengthAccessors = Readonly<{
  packedProgramWeightsLen(desc: PublicApi.TinyLinearDesc | PublicApi.TinyMlpDesc): number;
  packedProgramBiasLen(desc: PublicApi.TinyLinearDesc | PublicApi.TinyMlpDesc): number;
}>;

export type GenericModelProgramRequirements = Readonly<{
  scalarBytes: number;
  inputLen: number;
  inputByteLength: number;
  outputLen: number;
  outputByteLength: number;
  weightsLen: number;
  weightsByteLength: number;
  biasLen: number;
  biasByteLength: number;
  parameterLen: number;
  parameterByteLength: number;
}>;

export function genericModelNativeDesc(
  desc: PublicApi.TinyLinearDesc | PublicApi.TinyMlpDesc,
  kinds: GenericModelKindAliases,
): GenericModelNativeDesc {
  const hiddenLen = desc.hiddenLen ?? 0;
  const isMlp = hiddenLen !== 0;
  return Object.freeze({
    kind: isMlp ? kinds.tinyMlpKind : kinds.tinyLinearKind,
    activation: isMlp ? (desc.activation ?? 0) : 0,
    input_len: desc.inputLen,
    output_len: desc.outputLen,
    hidden_len: isMlp ? hiddenLen : 0,
  });
}

export function genericModelProgramRequirements(
  desc: PublicApi.TinyLinearDesc | PublicApi.TinyMlpDesc,
  accessors: GenericModelProgramLengthAccessors,
  scalarBytes = 4,
): GenericModelProgramRequirements {
  const weightsLen = accessors.packedProgramWeightsLen(desc);
  const biasLen = accessors.packedProgramBiasLen(desc);
  const parameterLen = weightsLen + biasLen;
  return Object.freeze({
    scalarBytes,
    inputLen: desc.inputLen,
    inputByteLength: desc.inputLen * scalarBytes,
    outputLen: desc.outputLen,
    outputByteLength: desc.outputLen * scalarBytes,
    weightsLen,
    weightsByteLength: weightsLen * scalarBytes,
    biasLen,
    biasByteLength: biasLen * scalarBytes,
    parameterLen,
    parameterByteLength: parameterLen * scalarBytes,
  });
}

export const genericModelDescriptorManifest = Object.freeze({
  kind: "zgml-generic-model-descriptor",
  ...tsRuntimeManifestPolicy("src/ts/runtime/generic_model_desc.ts", "Tensor/nn generic model -> Program requirements"),
});
