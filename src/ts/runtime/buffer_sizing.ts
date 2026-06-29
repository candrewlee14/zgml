"use strict";

export type BufferSizingSignatureInput = Readonly<{
  kind?: unknown;
  modelKind?: unknown;
  scalarType?: unknown;
  scalarBytes?: unknown;
  inputLen?: unknown;
  inputByteLength?: unknown;
  outputLen?: unknown;
  outputByteLength?: unknown;
  weightsLen?: unknown;
  weightsByteLength?: unknown;
  biasLen?: unknown;
  biasByteLength?: unknown;
  parameterLen?: unknown;
  parameterByteLength?: unknown;
}>;

export function bufferSizingEvidenceSignature(sizing: BufferSizingSignatureInput): string {
  return [
    sizing.kind,
    `model=${sizing.modelKind}`,
    `scalar=${sizing.scalarType}`,
    `scalarBytes=${sizing.scalarBytes}`,
    `input=${sizing.inputLen}`,
    `inputBytes=${sizing.inputByteLength}`,
    `output=${sizing.outputLen}`,
    `outputBytes=${sizing.outputByteLength}`,
    `weights=${sizing.weightsLen}`,
    `weightsBytes=${sizing.weightsByteLength}`,
    `bias=${sizing.biasLen}`,
    `biasBytes=${sizing.biasByteLength}`,
    `parameters=${sizing.parameterLen}`,
    `parameterBytes=${sizing.parameterByteLength}`,
  ].join("|");
}
