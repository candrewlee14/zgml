"use strict";

import {
  programBufferLayoutFromRequirements,
  programBufferSlot,
  programBufferSlotNames,
} from "./program_buffers.js";
import {
  programInputShape,
  programOutputShape,
} from "./session_tensor.js";
import type {
  ProgramBufferLayout,
  ProgramBufferLayoutSlot,
  ProgramRequirements,
} from "../public_api.js";

type ProgramDesc = Readonly<{
  inputShape?: readonly number[];
  outputShape?: readonly number[];
  inputLen: number;
  outputLen: number;
}>;
type LlamaSessionRequirements = Readonly<Record<string, unknown> & {
  modelKind: ProgramRequirements["modelKind"];
  scalarBytes: number;
  inputLen: number;
  outputLen: number;
  weightsLen: number;
  biasLen: number;
  outputByteLength: number;
}>;

function programBufferLayoutOrNull(layout: unknown): ProgramBufferLayout | null {
  if (!layout || typeof layout !== "object") return null;
  const candidate = layout as Partial<ProgramBufferLayout>;
  return Array.isArray(candidate.slots) ? candidate as ProgramBufferLayout : null;
}

export function sessionBufferSlotNames(layout: ProgramBufferLayout | null | undefined) {
  return programBufferSlotNames(layout);
}

export function sessionBufferSlot(layout: ProgramBufferLayout | null | undefined, nameOrKind: unknown, label = "Session.bufferSlot") {
  return programBufferSlot(layout, nameOrKind, label);
}

export function requireSessionKvCacheLayout(layout: unknown, label = "LLaMA Session kvCacheLayout") {
  if (!layout) {
    throw new Error(`${label} is unavailable`);
  }
  return layout;
}

export function requireNativeOutputBinding(
  boundOutput: unknown,
  isNativeBuffer: (value: unknown) => boolean,
  method: string,
) {
  if (!isNativeBuffer(boundOutput)) {
    throw new Error(`${method} requires a NativeBuffer output binding`);
  }
}

export function defaultLlamaSessionBufferLayout(vocabSize: number, bufferLayout: unknown = null): ProgramBufferLayout {
  const suppliedLayout = programBufferLayoutOrNull(bufferLayout);
  if (suppliedLayout) return suppliedLayout;
  const requirements: LlamaSessionRequirements = {
    modelKind: "tiny-llama",
    scalarBytes: Float32Array.BYTES_PER_ELEMENT,
    inputLen: 0,
    outputLen: vocabSize,
    weightsLen: 0,
    biasLen: 0,
    outputByteLength: vocabSize * Float32Array.BYTES_PER_ELEMENT,
  };
  return programBufferLayoutFromRequirements(requirements);
}

export function createLlamaSessionScratch() {
  return {
    scalarTokenScratch: new Uint32Array(1),
    scalarTokenWindowOptionsScratch: { tokensLen: 1 },
    noOutputTokenWindowOptionsScratch: {
      tokensLen: undefined,
      tokenLength: undefined,
      activeTokenLength: undefined,
      activeTokenCount: undefined,
      output: false,
    },
    outputTokenWindowOptionsScratch: {
      tokensLen: undefined,
      tokenLength: undefined,
      activeTokenLength: undefined,
      activeTokenCount: undefined,
      output: undefined,
    },
    scalarTokenExecuteOptionsScratch: {
      tokensLen: 1,
      tokenLength: undefined,
      activeTokenLength: undefined,
      activeTokenCount: undefined,
      output: undefined,
    },
    scalarTokenSampleOptionsScratch: {
      tokensLen: 1,
      topK: undefined,
      top_k: undefined,
      temperature: undefined,
      seed: undefined,
    },
  };
}

export function llamaSessionOutputShape(vocabSize: number): readonly number[] {
  return Object.freeze([vocabSize]);
}

export function llamaSessionInputLen(): number {
  return 0;
}

export function llamaSessionOutputLen(vocabSize: number): number {
  return vocabSize;
}

export function llamaSessionInputByteLength(): number {
  return 0;
}

export function llamaSessionOutputByteLength(vocabSize: number): number {
  return vocabSize * Float32Array.BYTES_PER_ELEMENT;
}

export function zeroSessionElementLength(): number {
  return 0;
}

export function zeroSessionByteLength(): number {
  return 0;
}

function layoutSlot(layout: ProgramBufferLayout | null | undefined, name: keyof Pick<ProgramBufferLayout, "weights" | "bias">): ProgramBufferLayoutSlot | null {
  return layout?.[name] ?? null;
}

function slotElementCount(layout: ProgramBufferLayout | null | undefined, name: keyof Pick<ProgramBufferLayout, "weights" | "bias">): number {
  return Number(layout?.[name]?.elementCount ?? 0);
}

function slotByteLength(layout: ProgramBufferLayout | null | undefined, name: keyof Pick<ProgramBufferLayout, "weights" | "bias">): number {
  return Number(layoutSlot(layout, name)?.byteLength ?? 0);
}

export function sessionWeightsLen(layout: ProgramBufferLayout | null | undefined): number {
  return slotElementCount(layout, "weights");
}

export function sessionWeightsByteLength(layout: ProgramBufferLayout | null | undefined): number {
  return slotByteLength(layout, "weights");
}

export function sessionBiasLen(layout: ProgramBufferLayout | null | undefined): number {
  return slotElementCount(layout, "bias");
}

export function sessionBiasByteLength(layout: ProgramBufferLayout | null | undefined): number {
  return slotByteLength(layout, "bias");
}

export function sessionParameterLen(layout: ProgramBufferLayout | null | undefined): number {
  return sessionWeightsLen(layout) + sessionBiasLen(layout);
}

export function sessionParameterByteLength(layout: ProgramBufferLayout | null | undefined): number {
  return sessionWeightsByteLength(layout) + sessionBiasByteLength(layout);
}

export function genericSessionInputShape(desc: ProgramDesc): readonly number[] {
  return programInputShape(desc);
}

export function genericSessionOutputShape(desc: ProgramDesc, boundOutputShape?: readonly number[] | null): readonly number[] {
  return Object.freeze((boundOutputShape || programOutputShape(desc)).slice());
}

export function genericSessionInputLen(desc: ProgramDesc): number {
  return desc.inputLen;
}

export function genericSessionOutputLen(desc: ProgramDesc): number {
  return desc.outputLen;
}

export function genericSessionInputByteLength(desc: ProgramDesc): number {
  return desc.inputLen * Float32Array.BYTES_PER_ELEMENT;
}

export function genericSessionOutputByteLength(desc: ProgramDesc): number {
  return desc.outputLen * Float32Array.BYTES_PER_ELEMENT;
}

export function boundBufferKind(value: unknown, isNativeBuffer: (value: unknown) => boolean): string {
  if (!value) return "none";
  return isNativeBuffer(value) ? "native" : "host";
}

export function hostBoundInput(boundInput: unknown, isNativeBuffer: (value: unknown) => boolean) {
  return isNativeBuffer(boundInput) ? null : boundInput;
}

export function nativeBoundOutput(boundOutput: unknown, isNativeBuffer: (value: unknown) => boolean) {
  return isNativeBuffer(boundOutput) ? boundOutput : null;
}

export function hostBoundOutput(boundOutput: unknown, isNativeBuffer: (value: unknown) => boolean) {
  return boundOutput && !isNativeBuffer(boundOutput) ? boundOutput : null;
}
