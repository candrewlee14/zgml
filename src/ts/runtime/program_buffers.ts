"use strict";

import type {
  ProgramBufferKind as PublicProgramBufferKind,
  ProgramBufferLayout,
  ProgramBufferLayoutSlot,
  ProgramBufferSlot as PublicProgramBufferSlot,
  ProgramOutputBufferSlot,
  ProgramRequirements,
} from "../public_api.js";

type BufferRole = ProgramBufferLayoutSlot["role"];
type ProgramBufferKind = PublicProgramBufferKind | "output";
type LlamaKvBufferKind = "kv-k" | "kv-v";

type UnsignedProgramBufferLayout = Readonly<{
  kind: "zgml.program.buffer-layout";
  scalarType: "f32";
  scalarBytes: number;
  slots: readonly ProgramBufferLayoutSlot[];
  input: ProgramBufferLayoutSlot;
  output: ProgramBufferLayoutSlot;
  weights: ProgramBufferLayoutSlot;
  bias: ProgramBufferLayoutSlot;
}>;

type ProgramRequirementsInput = ProgramRequirements | Readonly<Record<string, unknown>>;
type ProgramFactorySlot = PublicProgramBufferSlot | ProgramOutputBufferSlot;

type ByteLengthResource = {
  readonly byteLength: number;
};

type LlamaKvRequirements = {
  readonly scalarBytes?: number;
  readonly layers?: number;
  readonly contextLength?: number;
  readonly kBufferByteLength: number;
  readonly vBufferByteLength: number;
  readonly bufferByteLength?: number;
  readonly [key: string]: unknown;
};

type LlamaKvSlot = Readonly<{
  kind: "k" | "v";
  name: LlamaKvBufferKind;
  role: "persistent";
  scalarType: "f32";
  scalarBytes: number;
  layer: number;
  elementOffset: number;
  elementCount: number;
  byteOffset: number;
  byteLength: number;
}>;

function kernelBufferLayoutSlot(name: ProgramBufferKind, elementCount: number, role: BufferRole): ProgramBufferLayoutSlot {
  return {
    name,
    role,
    scalarType: "f32",
    scalarBytes: 4,
    elementOffset: 0,
    elementCount,
    byteOffset: 0,
    byteLength: elementCount * 4,
  };
}

export function kernelBufferLayout(inputLen: number, outputLen: number, weightsLen: number, biasLen: number): ProgramBufferLayout {
  const slots = [
    kernelBufferLayoutSlot("input", inputLen, "step-input"),
    kernelBufferLayoutSlot("output", outputLen, "step-output"),
    kernelBufferLayoutSlot("weights", weightsLen, "persistent"),
    kernelBufferLayoutSlot("bias", biasLen, "persistent"),
  ];
  return withProgramBufferLayoutSignature({
    kind: "zgml.program.buffer-layout",
    scalarType: "f32",
    scalarBytes: 4,
    slots,
    input: slots[0],
    output: slots[1],
    weights: slots[2],
    bias: slots[3],
  });
}

function freezeKernelBufferLayoutSlot(slot: ProgramBufferLayoutSlot): ProgramBufferLayoutSlot {
  return Object.freeze({
    name: slot.name,
    role: slot.role,
    scalarType: slot.scalarType,
    scalarBytes: slot.scalarBytes,
    elementOffset: slot.elementOffset,
    elementCount: slot.elementCount,
    byteOffset: slot.byteOffset,
    byteLength: slot.byteLength,
  });
}

export function freezeKernelBufferLayout(layout: ProgramBufferLayout): ProgramBufferLayout {
  const slots = layout.slots.map(freezeKernelBufferLayoutSlot);
  return Object.freeze(withProgramBufferLayoutSignature({
    kind: "zgml.program.buffer-layout",
    scalarType: layout.scalarType,
    scalarBytes: layout.scalarBytes,
    slots: Object.freeze(slots),
    input: slots[0],
    output: slots[1],
    weights: slots[2],
    bias: slots[3],
  }));
}

export function programBufferLayoutFromRequirements(requirements: ProgramRequirementsInput): ProgramBufferLayout {
  const scalarBytes = Number(requirements.scalarBytes ?? 4);
  const inputLen = Number(requirements.inputLen ?? 0);
  const outputLen = Number(requirements.outputLen ?? 0);
  const weightsLen = Number(requirements.weightsLen ?? 0);
  const biasLen = Number(requirements.biasLen ?? 0);
  const inputByteLength = Number(requirements.inputByteLength ?? inputLen * scalarBytes);
  const outputByteLength = Number(requirements.outputByteLength ?? outputLen * scalarBytes);
  const weightsByteLength = Number(requirements.weightsByteLength ?? weightsLen * scalarBytes);
  const biasByteLength = Number(requirements.biasByteLength ?? biasLen * scalarBytes);
  const slot = (name: ProgramBufferKind, elementCount: number, byteLength: number, role: BufferRole): ProgramBufferLayoutSlot => ({
    name,
    role,
    scalarType: "f32",
    scalarBytes,
    elementOffset: 0,
    elementCount,
    byteOffset: 0,
    byteLength,
  });
  const slots = [
    slot("input", inputLen, inputByteLength, "step-input"),
    slot("output", outputLen, outputByteLength, "step-output"),
    slot("weights", weightsLen, weightsByteLength, "persistent"),
    slot("bias", biasLen, biasByteLength, "persistent"),
  ];
  return freezeKernelBufferLayout({
    kind: "zgml.program.buffer-layout",
    signature: "",
    scalarType: "f32",
    scalarBytes,
    slots,
    input: slots[0],
    output: slots[1],
    weights: slots[2],
    bias: slots[3],
  });
}

export function programBufferLayoutSignature(layout: { kind?: unknown; scalarType?: unknown; scalarBytes?: unknown; slots?: readonly ProgramBufferLayoutSlot[] }): string {
  const slots = Array.isArray(layout.slots) ? layout.slots : [];
  return [
    String(layout.kind ?? "zgml.program.buffer-layout"),
    `scalar=${String(layout.scalarType ?? "f32")}`,
    `scalarBytes=${String(layout.scalarBytes ?? 4)}`,
    ...slots.map((slot) => [
      slot.name,
      slot.role,
      slot.elementOffset,
      slot.elementCount,
      slot.byteOffset,
      slot.byteLength,
    ].join(":")),
  ].join("|");
}

function withProgramBufferLayoutSignature(layout: UnsignedProgramBufferLayout): ProgramBufferLayout {
  return {
    ...layout,
    signature: programBufferLayoutSignature(layout),
  };
}

export function programBufferSlotNames(layout: ProgramBufferLayout | null | undefined): readonly string[] {
  const slots = layout && Array.isArray(layout.slots) ? layout.slots : [];
  return Object.freeze(slots.map((slot) => slot.name));
}

export function programBufferSlot(layout: ProgramBufferLayout | null | undefined, nameOrKind: unknown, methodName = "Program.bufferSlot"): ProgramBufferLayoutSlot | null {
  if (typeof nameOrKind !== "string") {
    throw new Error(`${methodName} requires a buffer slot name or kind`);
  }
  if (!layout || !Array.isArray(layout.slots)) return null;
  const keyedSlot = (layout as Record<string, unknown>)[nameOrKind];
  if (keyedSlot && typeof keyedSlot === "object") {
    return keyedSlot as ProgramBufferLayoutSlot;
  }
  return layout.slots.find((slot) => slot && slot.name === nameOrKind) || null;
}

export function programBufferFactorySlotFromRequirements(requirements: ProgramRequirementsInput, kind: ProgramBufferKind): ProgramFactorySlot {
  const layout = programBufferLayoutFromRequirements(requirements);
  const slot = layout[kind];
  if (!slot || typeof slot !== "object") throw new Error(`unknown Program buffer kind: ${kind}`);
  return Object.freeze({
    ...slot,
    kind,
    elementLength: slot.elementCount,
    requirements,
    layout,
    slot,
  }) as ProgramFactorySlot;
}

function programBufferResourceFactoryName(kind: string): string {
  return `create${kind[0].toUpperCase()}${kind.slice(1)}Buffer`;
}

export function requireProgramBufferResourceFactory<T>(kind: string, value: T): T {
  if (typeof value !== "function") {
    throw new Error(`Program ${programBufferResourceFactoryName(kind)} resource option must be a function`);
  }
  return value;
}

export function assertProgramBufferSlotRequired<T extends { byteLength: number }>(slot: T, kind: string): T {
  if (slot.byteLength <= 0) throw new Error(`Program has no ${kind} buffer requirement`);
  return slot;
}

export function assertProgramBufferResourceByteLength<T extends ByteLengthResource>(name: string, buffer: T, slot: { byteLength: number }): T {
  if (buffer.byteLength < slot.byteLength) {
    throw new Error(`${name} is too small: ${buffer.byteLength} < ${slot.byteLength}`);
  }
  return buffer;
}

export function llamaKvCacheResourceFactory(options: { resource?: unknown; externalResource?: unknown } = {}) {
  const makeResource = options.resource ?? options.externalResource ?? null;
  if (makeResource !== null && typeof makeResource !== "function") {
    throw new Error("LLaMA createKvCache resource option must be a function");
  }
  return makeResource;
}

export function llamaKvCacheResourceSlot(requirements: LlamaKvRequirements, layout: unknown, slot: Record<string, unknown>) {
  return Object.freeze({
    ...slot,
    requirements,
    layout,
    slot,
  });
}

export function llamaProgramKvCacheBufferSlot(requirements: LlamaKvRequirements, kind: LlamaKvBufferKind) {
  if (kind !== "kv-k" && kind !== "kv-v") throw new Error(`unknown LLaMA KV cache buffer kind: ${kind}`);
  const byteLength = kind === "kv-k" ? requirements.kBufferByteLength : requirements.vBufferByteLength;
  return Object.freeze({
    kind,
    byteLength,
    elementLength: byteLength / Number(requirements.scalarBytes),
    requirements,
  });
}

export function assertLlamaKvCacheResourceByteLength<T extends ByteLengthResource>(name: string, buffer: T, byteLength: number): T {
  if (buffer.byteLength < byteLength) {
    throw new Error(`${name} is too small: ${buffer.byteLength} < ${byteLength}`);
  }
  return buffer;
}

export function llamaKvCacheLayoutFromRequirements(requirements: LlamaKvRequirements) {
  const scalarBytes = Number(requirements.scalarBytes ?? 4);
  const layers = Number(requirements.layers ?? 0);
  const kByteLength = Number(requirements.kBufferByteLength ?? 0);
  const vByteLength = Number(requirements.vBufferByteLength ?? 0);
  const makeSlot = (kind: "k" | "v", layer: number, byteLength: number): LlamaKvSlot => Object.freeze({
    kind,
    name: kind === "k" ? "kv-k" : "kv-v",
    role: "persistent",
    scalarType: "f32",
    scalarBytes,
    layer,
    elementOffset: 0,
    elementCount: byteLength / scalarBytes,
    byteOffset: 0,
    byteLength,
  });
  const k: LlamaKvSlot[] = [];
  const v: LlamaKvSlot[] = [];
  for (let layer = 0; layer < layers; layer += 1) {
    k.push(makeSlot("k", layer, kByteLength));
    v.push(makeSlot("v", layer, vByteLength));
  }
  const layout = {
    kind: "zgml.llama.kv-cache.layout",
    scalarType: "f32",
    scalarBytes,
    layers,
    contextLength: Number(requirements.contextLength ?? 0),
    kBufferByteLength: kByteLength,
    vBufferByteLength: vByteLength,
    bufferByteLength: Number(requirements.bufferByteLength ?? (kByteLength + vByteLength) * layers),
    slots: Object.freeze(k.flatMap((slot, index) => [slot, v[index]])),
    k: Object.freeze(k),
    v: Object.freeze(v),
  };
  return Object.freeze({
    ...layout,
    signature: llamaKvCacheLayoutSignature(layout),
  });
}

export function llamaKvCacheLayoutSignature(layout: {
  kind?: unknown;
  scalarType?: unknown;
  scalarBytes?: unknown;
  layers?: unknown;
  contextLength?: unknown;
  kBufferByteLength?: unknown;
  vBufferByteLength?: unknown;
  bufferByteLength?: unknown;
  slots?: readonly LlamaKvSlot[];
}) {
  const slots = Array.isArray(layout.slots) ? layout.slots : [];
  return [
    String(layout.kind ?? "zgml.llama.kv-cache.layout"),
    `scalar=${String(layout.scalarType ?? "f32")}`,
    `scalarBytes=${String(layout.scalarBytes ?? 4)}`,
    `layers=${String(layout.layers ?? 0)}`,
    `context=${String(layout.contextLength ?? 0)}`,
    `kBytes=${String(layout.kBufferByteLength ?? 0)}`,
    `vBytes=${String(layout.vBufferByteLength ?? 0)}`,
    `bufferBytes=${String(layout.bufferByteLength ?? 0)}`,
    ...slots.map((slot) => [
      slot.layer,
      slot.name,
      slot.kind,
      slot.elementOffset,
      slot.elementCount,
      slot.byteOffset,
      slot.byteLength,
    ].join(":")),
  ].join("|");
}
