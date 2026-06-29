import { programBufferKindId } from "./abi.js";
import type {
  LlamaKvCacheCreateOptions,
  LlamaKvCacheLayoutSlot,
  ProgramBufferKind,
  ProgramBufferSlot,
  ProgramDeviceBufferKind,
  ProgramOutputBufferSlot,
  ProgramRequirements,
  ZgmlBackend,
} from "../public_api.js";
import {
  assertLlamaKvCacheResourceByteLength,
  assertProgramBufferResourceByteLength,
  assertProgramBufferSlotRequired,
  llamaKvCacheResourceFactory,
  llamaProgramKvCacheBufferSlot,
  programBufferFactorySlotFromRequirements,
  requireProgramBufferResourceFactory,
} from "./program_buffers.js";

type ProgramKvCacheBufferKind = "kv-k" | "kv-v";
type ProgramBufferResourceOptions = Readonly<{
  resource?: unknown;
  externalResource?: unknown;
  placement?: ZgmlBackend | string;
  backend?: ZgmlBackend | string;
  device?: ZgmlBackend | string;
}>;
type ProgramOutputBufferFactoryOptions = ProgramBufferResourceOptions;
type ProgramBufferFactoryOptions = ProgramBufferResourceOptions;
type LlamaKvCacheFactoryOptions = ProgramBufferResourceOptions;
type ProgramBufferFactoryNativeBuffer = { readonly byteLength: number };
type ProgramLlamaKvCacheRequirements = Readonly<{
  scalarBytes?: number;
  layers?: number;
  contextLength?: number;
  kBufferByteLength: number;
  vBufferByteLength: number;
  bufferByteLength?: number;
  [key: string]: unknown;
}>;

export type ProgramBufferFactoryHelpersOptions<
  THandle = unknown,
  TNativeBuffer extends ProgramBufferFactoryNativeBuffer = ProgramBufferFactoryNativeBuffer,
  TLlamaKvCache = unknown,
> = Readonly<{
  programRequirements: (handle: THandle) => ProgramRequirements;
  llamaKvCacheRequirements: (handle: THandle) => ProgramLlamaKvCacheRequirements;
  createHostBuffer: (handle: THandle, kind: ProgramDeviceBufferKind | unknown) => TNativeBuffer;
  createHostOutputBuffer?: (handle: THandle, kind: "output") => TNativeBuffer;
  createDeviceBuffer: (handle: THandle, kind: ProgramDeviceBufferKind | unknown, placement: ZgmlBackend | string | unknown) => TNativeBuffer;
  requireNativeBuffer: (value: unknown, name: string) => TNativeBuffer | null;
  createLlamaKvCache: (
    requirements: ProgramLlamaKvCacheRequirements,
    createOptions: LlamaKvCacheFactoryOptions,
    resourceFactory: ((slot: LlamaKvCacheLayoutSlot) => TNativeBuffer) | null,
  ) => TLlamaKvCache;
}>;

function resourceFactoryOption(createOptions: ProgramBufferResourceOptions | null | undefined): unknown {
  return (createOptions && (createOptions.resource ?? createOptions.externalResource)) ?? null;
}

function placementOption(createOptions: ProgramBufferResourceOptions | null | undefined): ZgmlBackend | string | null {
  return (createOptions && (createOptions.placement ?? createOptions.backend ?? createOptions.device)) ?? null;
}

function requireBuffer<TNativeBuffer extends ProgramBufferFactoryNativeBuffer>(
  requireNativeBuffer: (value: unknown, name: string) => TNativeBuffer | null,
  value: unknown,
  name: string,
): TNativeBuffer {
  const buffer = requireNativeBuffer(value, name);
  if (buffer == null) throw new Error(`${name} must be a NativeBuffer`);
  return buffer;
}

export function createProgramBufferFactoryHelpers<
  THandle = unknown,
  TNativeBuffer extends ProgramBufferFactoryNativeBuffer = ProgramBufferFactoryNativeBuffer,
  TLlamaKvCache = unknown,
>(options: ProgramBufferFactoryHelpersOptions<THandle, TNativeBuffer, TLlamaKvCache>) {
  const programRequirements = options.programRequirements;
  const llamaKvCacheRequirements = options.llamaKvCacheRequirements;
  const createHostBuffer = options.createHostBuffer;
  const createHostOutputBuffer = options.createHostOutputBuffer;
  const createDeviceBuffer = options.createDeviceBuffer;
  const requireNativeBuffer = options.requireNativeBuffer;
  const createLlamaKvCache = options.createLlamaKvCache;

  if (typeof programRequirements !== "function") {
    throw new Error("createProgramBufferFactoryHelpers requires programRequirements");
  }
  if (typeof llamaKvCacheRequirements !== "function") {
    throw new Error("createProgramBufferFactoryHelpers requires llamaKvCacheRequirements");
  }
  if (typeof createHostBuffer !== "function") {
    throw new Error("createProgramBufferFactoryHelpers requires createHostBuffer");
  }
  if (createHostOutputBuffer !== undefined && typeof createHostOutputBuffer !== "function") {
    throw new Error("createProgramBufferFactoryHelpers createHostOutputBuffer must be a function");
  }
  if (typeof createDeviceBuffer !== "function") {
    throw new Error("createProgramBufferFactoryHelpers requires createDeviceBuffer");
  }
  if (typeof requireNativeBuffer !== "function") {
    throw new Error("createProgramBufferFactoryHelpers requires requireNativeBuffer");
  }
  if (typeof createLlamaKvCache !== "function") {
    throw new Error("createProgramBufferFactoryHelpers requires createLlamaKvCache");
  }

  function createProgramOutputBuffer(handle: THandle, createOptions: ProgramOutputBufferFactoryOptions = {}): TNativeBuffer {
    const makeResource = resourceFactoryOption(createOptions);
    const placement = placementOption(createOptions);
    if (makeResource === null && placement !== null) return createDeviceBuffer(handle, "output", placement);
    if (makeResource === null) return (createHostOutputBuffer || createHostBuffer)(handle, "output");
    const resourceFactory = requireProgramBufferResourceFactory("output", makeResource);
    const requirements = programRequirements(handle);
    const slot = programBufferFactorySlotFromRequirements(requirements, "output") as ProgramOutputBufferSlot;
    const buffer = requireBuffer(requireNativeBuffer, (resourceFactory as (slot: unknown) => unknown)(slot), "output");
    return assertProgramBufferResourceByteLength("output", buffer, slot) as TNativeBuffer;
  }

  function createProgramBuffer(handle: THandle, kind: ProgramBufferKind, createOptions: ProgramBufferFactoryOptions = {}): TNativeBuffer {
    const requirements = programRequirements(handle);
    const slot = assertProgramBufferSlotRequired(
      programBufferFactorySlotFromRequirements(requirements, kind) as ProgramBufferSlot,
      String(kind),
    );
    const makeResource = resourceFactoryOption(createOptions);
    const placement = placementOption(createOptions);
    if (makeResource === null && placement !== null) return createDeviceBuffer(handle, kind, placement);
    if (makeResource === null) return createHostBuffer(handle, kind);
    const resourceFactory = requireProgramBufferResourceFactory(String(kind), makeResource);
    const buffer = requireBuffer(requireNativeBuffer, (resourceFactory as (slot: unknown) => unknown)(slot), String(kind));
    return assertProgramBufferResourceByteLength(String(kind), buffer, slot) as TNativeBuffer;
  }

  function createProgramKvCacheBuffer(handle: THandle, kind: ProgramKvCacheBufferKind, createOptions: LlamaKvCacheFactoryOptions = {}): TNativeBuffer {
    const requirements = llamaKvCacheRequirements(handle);
    const makeResource = resourceFactoryOption(createOptions);
    const placement = placementOption(createOptions);
    if (makeResource === null && placement !== null) return createDeviceBuffer(handle, kind, placement);
    if (makeResource === null) return createHostBuffer(handle, kind);
    if (typeof makeResource !== "function") {
      throw new Error(`Program create ${kind} resource option must be a function`);
    }
    const slot = llamaProgramKvCacheBufferSlot(requirements, kind);
    const buffer = requireBuffer(requireNativeBuffer, (makeResource as (slot: unknown) => unknown)(slot), kind);
    return assertLlamaKvCacheResourceByteLength(String(kind), buffer, slot.byteLength) as TNativeBuffer;
  }

  function createProgramNamedBuffer(handle: THandle, kind: ProgramDeviceBufferKind, createOptions: ProgramBufferResourceOptions = {}): TNativeBuffer {
    programBufferKindId(kind as string);
    if (kind === "output") return createProgramOutputBuffer(handle, createOptions);
    if (kind === "kv-k" || kind === "kv-v") return createProgramKvCacheBuffer(handle, kind, createOptions);
    return createProgramBuffer(handle, kind, createOptions);
  }

  function createProgramLlamaKvCache(handle: THandle, createOptions: LlamaKvCacheFactoryOptions = {}): TLlamaKvCache {
    const requirements = llamaKvCacheRequirements(handle);
    const makeResource = llamaKvCacheResourceFactory(createOptions);
    const placement = placementOption(createOptions);
    if (makeResource !== null || placement === null) {
      return createLlamaKvCache(requirements, createOptions, null);
    }
    return createLlamaKvCache(requirements, createOptions, (slot: LlamaKvCacheLayoutSlot) =>
      createProgramKvCacheBuffer(handle, slot.kind === "k" ? "kv-k" : "kv-v", { placement }));
  }

  return Object.freeze({
    createProgramOutputBuffer,
    createProgramBuffer,
    createProgramKvCacheBuffer,
    createProgramNamedBuffer,
    createProgramLlamaKvCache,
  });
}
