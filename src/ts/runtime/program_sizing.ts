import type {
  ProgramBufferSizing,
  ProgramRequirements,
} from "../public_api.js";
import { bufferSizingEvidenceSignature, type BufferSizingSignatureInput } from "./buffer_sizing.js";

type UnknownRecord = Record<string, unknown>;
type ProgramScalarEvidence = Readonly<UnknownRecord & {
  scalarBytes?: number;
}>;
type ProgramRequirementsEvidence = Readonly<ProgramScalarEvidence & {
  modelKind: ProgramRequirements["modelKind"];
  inputLen: number;
  inputByteLength?: number;
  outputLen: number;
  outputByteLength?: number;
  weightsLen?: number;
  weightsByteLength?: number;
  biasLen?: number;
  biasByteLength?: number;
  parameterLen?: number;
  parameterByteLength?: number;
}>;
type ProgramRequirementsInput = ProgramRequirements | ProgramRequirementsEvidence;
type MutableProgramBufferSizing = Omit<ProgramBufferSizing, "signature"> & {
  signature?: string;
};

function scalarBytes(requirements: ProgramScalarEvidence) {
  return Number(requirements.scalarBytes ?? 4);
}

export function programInputLen(requirements: ProgramRequirementsInput): number {
  return requirements.inputLen;
}

export function programOutputLen(requirements: ProgramRequirementsInput): number {
  return requirements.outputLen;
}

export function programInputByteLength(requirements: ProgramRequirementsInput): number {
  return Number(requirements.inputByteLength ?? requirements.inputLen * scalarBytes(requirements));
}

export function programOutputByteLength(requirements: ProgramRequirementsInput): number {
  return Number(requirements.outputByteLength ?? requirements.outputLen * scalarBytes(requirements));
}

export function programWeightsLen(requirements: ProgramRequirementsInput): number {
  return Number(requirements.weightsLen ?? 0);
}

export function programWeightsByteLength(requirements: ProgramRequirementsInput): number {
  return Number(requirements.weightsByteLength ?? programWeightsLen(requirements) * scalarBytes(requirements));
}

export function programBiasLen(requirements: ProgramRequirementsInput): number {
  return Number(requirements.biasLen ?? 0);
}

export function programBiasByteLength(requirements: ProgramRequirementsInput): number {
  return Number(requirements.biasByteLength ?? programBiasLen(requirements) * scalarBytes(requirements));
}

export function programParameterLen(requirements: ProgramRequirementsInput): number {
  return Number(requirements.parameterLen ?? programWeightsLen(requirements) + programBiasLen(requirements));
}

export function programParameterByteLength(requirements: ProgramRequirementsInput): number {
  return Number(requirements.parameterByteLength ?? programParameterLen(requirements) * scalarBytes(requirements));
}

export function programBufferSizingSignature(sizing: ProgramBufferSizing | BufferSizingSignatureInput): string {
  return bufferSizingEvidenceSignature(sizing);
}

export function programBufferSizingFromRequirements(requirements: ProgramRequirementsInput): ProgramBufferSizing {
  const sizing: MutableProgramBufferSizing = {
    kind: "program-buffer-sizing",
    scalarType: "f32",
    scalarBytes: scalarBytes(requirements),
    inputLen: programInputLen(requirements),
    inputByteLength: programInputByteLength(requirements),
    outputLen: programOutputLen(requirements),
    outputByteLength: programOutputByteLength(requirements),
    weightsLen: programWeightsLen(requirements),
    weightsByteLength: programWeightsByteLength(requirements),
    biasLen: programBiasLen(requirements),
    biasByteLength: programBiasByteLength(requirements),
    parameterLen: programParameterLen(requirements),
    parameterByteLength: programParameterByteLength(requirements),
    modelKind: requirements.modelKind,
  } satisfies Omit<ProgramBufferSizing, "signature">;
  sizing.signature = programBufferSizingSignature(sizing);
  return Object.freeze(sizing) as ProgramBufferSizing;
}

export function programMatchesBufferSizingSignature(requirements: ProgramRequirementsInput, signature: unknown): boolean {
  return typeof signature === "string" && signature.length > 0 && programBufferSizingFromRequirements(requirements).signature === signature;
}

export type ProgramSizingAccessorsOptions<TProgram extends object = object> = Readonly<{
  requirements: (program: TProgram) => ProgramRequirementsInput;
}>;

export function createProgramSizingAccessors<TProgram extends object = object>(options: ProgramSizingAccessorsOptions<TProgram>) {
  const programRequirements = options.requirements;
  if (typeof programRequirements !== "function") {
    throw new Error("createProgramSizingAccessors requires requirements");
  }

  function requirements(program: TProgram): ProgramRequirementsInput {
    return programRequirements(program);
  }

  function inputLen(program: TProgram) {
    return programInputLen(requirements(program));
  }

  function outputLen(program: TProgram) {
    return programOutputLen(requirements(program));
  }

  function inputByteLength(program: TProgram) {
    return programInputByteLength(requirements(program));
  }

  function outputByteLength(program: TProgram) {
    return programOutputByteLength(requirements(program));
  }

  function weightsLen(program: TProgram) {
    return programWeightsLen(requirements(program));
  }

  function weightsByteLength(program: TProgram) {
    return programWeightsByteLength(requirements(program));
  }

  function biasLen(program: TProgram) {
    return programBiasLen(requirements(program));
  }

  function biasByteLength(program: TProgram) {
    return programBiasByteLength(requirements(program));
  }

  function parameterLen(program: TProgram) {
    return programParameterLen(requirements(program));
  }

  function parameterByteLength(program: TProgram) {
    return programParameterByteLength(requirements(program));
  }

  function bufferSizing(program: TProgram): ProgramBufferSizing {
    return programBufferSizingFromRequirements(requirements(program));
  }

  function matchesBufferSizingSignature(program: TProgram, signature: unknown) {
    return programMatchesBufferSizingSignature(requirements(program), signature);
  }

  return Object.freeze({
    inputLen,
    outputLen,
    inputByteLength,
    outputByteLength,
    weightsLen,
    weightsByteLength,
    biasLen,
    biasByteLength,
    parameterLen,
    parameterByteLength,
    bufferSizing,
    matchesBufferSizingSignature,
  });
}
