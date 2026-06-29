"use strict";

import {
  shapeSignature,
  stepParamsOutputEffect,
  stepParamsOutputOwnership,
  stepParamsOutputReturnOwnership,
} from "./step_params.js";
import { bufferSizingEvidenceSignature } from "./buffer_sizing.js";

export type SessionBufferSizingInput = Readonly<{
  kind: string;
  modelKind: unknown;
  scalarType: string;
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

export type SessionBufferSizing = Readonly<SessionBufferSizingInput & {
  signature: string;
}>;

export type LlamaSessionStepContractOptions = Readonly<{
  vocabSize: number;
  outputShape: readonly number[];
  outputByteLength: number;
  contextLength: number;
  position: number;
  hasBoundOutput: boolean;
}>;

export type GenericSessionStepContractOptions = Readonly<{
  inputLen: number;
  outputLen: number;
  inputShape: readonly number[];
  outputShape: readonly number[];
  programOutputShape: readonly number[];
  inputByteLength: number;
  outputByteLength: number;
  boundInput: string;
  boundOutput: string;
}>;

export function bufferSizingSignature(sizing: SessionBufferSizingInput): string {
  return bufferSizingEvidenceSignature(sizing);
}

export function sessionBufferSizing(sizing: SessionBufferSizingInput): SessionBufferSizing {
  const out = { ...sizing } as SessionBufferSizingInput & { signature: string };
  out.signature = bufferSizingSignature(out);
  return Object.freeze(out);
}

export function createLlamaSessionStepContract(options: LlamaSessionStepContractOptions) {
  const boundOutput = options.hasBoundOutput ? "native" : "none";
  const defaultOutput = options.hasBoundOutput ? "bound-native-readback" : "allocated";
  const defaultReadbackRequired = defaultOutput === "bound-native-readback";
  const defaultAllocationFree = false;
  const defaultHotPath = defaultAllocationFree && !defaultReadbackRequired;
  const defaultOutputEffect = stepParamsOutputEffect(defaultOutput);
  const defaultOutputOwnership = stepParamsOutputOwnership(defaultOutput);
  const defaultOutputReturnOwnership = stepParamsOutputReturnOwnership(defaultOutput);
  const remainingContext = Math.max(0, options.contextLength - options.position);
  const signature = [
    "llama-session-step",
    "scalar=f32",
    "scalarBytes=4",
    "tokenIdBytes=4",
    `logits=${options.vocabSize}`,
    `outputShape=${shapeSignature(options.outputShape)}`,
    `outputBytes=${options.outputByteLength}`,
    "outputSlot=output",
    "outputSlotRole=step-output",
    `context=${options.contextLength}`,
    `boundOutput=${boundOutput}`,
    `defaultOutput=${defaultOutput}`,
    `defaultReadback=${defaultReadbackRequired ? 1 : 0}`,
    `defaultAllocationFree=${defaultAllocationFree ? 1 : 0}`,
    `defaultHotPath=${defaultHotPath ? 1 : 0}`,
    `defaultOutputEffect=${defaultOutputEffect}`,
    `defaultOutputOwnership=${defaultOutputOwnership}`,
    `defaultOutputReturnOwnership=${defaultOutputReturnOwnership}`,
    "token=1",
    "tokens=1",
    "inlineOutput=1",
    "noOutput=1",
    "noOutputEffect=advance-state",
  ].join("|");
  return Object.freeze({
    kind: "llama-session-step-contract",
    signature,
    scalarType: "f32",
    scalarBytes: 4,
    tokenIdBytes: 4,
    logitsLen: options.vocabSize,
    outputLen: options.vocabSize,
    outputByteLength: options.outputByteLength,
    outputSlotName: "output",
    outputSlotRole: "step-output",
    outputShape: options.outputShape,
    contextLength: options.contextLength,
    position: options.position,
    remainingContext,
    boundOutput,
    defaultOutput,
    defaultReadbackRequired,
    defaultAllocationFree,
    defaultHotPath,
    defaultOutputEffect,
    defaultOutputOwnership,
    defaultOutputReturnOwnership,
    hasBoundOutput: options.hasBoundOutput,
    canReadOutput: options.hasBoundOutput,
    acceptsToken: true,
    acceptsTokens: true,
    acceptsInlineOutput: true,
    acceptsNoOutput: true,
    noOutputEffect: "advance-state",
  });
}

export function createGenericSessionStepContract(options: GenericSessionStepContractOptions) {
  const defaultOutput = options.boundOutput === "host"
    ? "bound-host"
    : options.boundOutput === "native"
      ? "bound-native-readback"
      : "allocated";
  const defaultReadbackRequired = defaultOutput === "bound-native-readback";
  const defaultAllocationFree = defaultOutput === "bound-host";
  const defaultHotPath = defaultAllocationFree && !defaultReadbackRequired;
  const defaultOutputEffect = stepParamsOutputEffect(defaultOutput);
  const defaultOutputOwnership = stepParamsOutputOwnership(defaultOutput);
  const defaultOutputReturnOwnership = stepParamsOutputReturnOwnership(defaultOutput);
  const acceptsNoInput = options.boundInput !== "none" || options.inputLen === 0;
  const signature = [
    "session-step",
    "scalar=f32",
    "scalarBytes=4",
    `input=${options.inputLen}`,
    `inputShape=${shapeSignature(options.inputShape)}`,
    `inputBytes=${options.inputByteLength}`,
    "inputSlot=input",
    "inputSlotRole=step-input",
    `output=${options.outputLen}`,
    `outputShape=${shapeSignature(options.outputShape)}`,
    `programOutputShape=${shapeSignature(options.programOutputShape)}`,
    `outputBytes=${options.outputByteLength}`,
    "outputSlot=output",
    "outputSlotRole=step-output",
    `boundInput=${options.boundInput}`,
    `boundOutput=${options.boundOutput}`,
    `defaultOutput=${defaultOutput}`,
    `defaultReadback=${defaultReadbackRequired ? 1 : 0}`,
    `defaultAllocationFree=${defaultAllocationFree ? 1 : 0}`,
    `defaultHotPath=${defaultHotPath ? 1 : 0}`,
    `defaultOutputEffect=${defaultOutputEffect}`,
    `defaultOutputOwnership=${defaultOutputOwnership}`,
    `defaultOutputReturnOwnership=${defaultOutputReturnOwnership}`,
    "inlineInput=1",
    `noInput=${acceptsNoInput ? 1 : 0}`,
    "inlineOutput=1",
    "noOutput=1",
    "noOutputEffect=advance-state",
  ].join("|");
  return Object.freeze({
    kind: "session-step-contract",
    signature,
    scalarType: "f32",
    scalarBytes: 4,
    inputLen: options.inputLen,
    outputLen: options.outputLen,
    inputByteLength: options.inputByteLength,
    outputByteLength: options.outputByteLength,
    inputSlotName: "input",
    outputSlotName: "output",
    inputSlotRole: "step-input",
    outputSlotRole: "step-output",
    inputShape: options.inputShape,
    outputShape: options.outputShape,
    programOutputShape: options.programOutputShape,
    boundInput: options.boundInput,
    boundOutput: options.boundOutput,
    defaultOutput,
    defaultReadbackRequired,
    defaultAllocationFree,
    defaultHotPath,
    defaultOutputEffect,
    defaultOutputOwnership,
    defaultOutputReturnOwnership,
    hasBoundInput: options.boundInput !== "none",
    hasBoundOutput: options.boundOutput !== "none",
    canReadOutput: options.boundOutput !== "none",
    acceptsInlineInput: true,
    acceptsNoInput,
    requiresInput: !acceptsNoInput,
    acceptsInlineOutput: true,
    acceptsNoOutput: true,
    noOutputEffect: "advance-state",
  });
}
