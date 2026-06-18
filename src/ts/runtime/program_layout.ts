import {
  type LlamaKvCacheLayout,
  type LlamaKvCacheRequirements,
  type ProgramBufferLayout,
  type ProgramBufferLayoutSlot,
} from "../public_api.js";
import {
  llamaKvCacheLayoutFromRequirements,
  programBufferSlot,
  programBufferSlotNames,
} from "./program_buffers.js";

type UnknownRecord = Record<string, unknown>;
type ProgramHandleRecord<THandle = unknown> = UnknownRecord & {
  handle: THandle;
  bufferLayoutSnapshot?: ProgramBufferLayout;
};
type ProgramBufferLayoutPolicy<THandle = unknown> = Readonly<{
  bufferLayout(handle: THandle): ProgramBufferLayout;
}>;

export type ProgramLayoutAccessorsOptions<THandle = unknown> = Readonly<{
  programPolicy: ProgramBufferLayoutPolicy<THandle>;
  assertProgramAlive?: (handle: THandle, label: string) => void;
  llamaKvCacheRequirements?: (handle: THandle) => LlamaKvCacheRequirements;
  bufferSlotMethodName?: string;
}>;

export function createProgramLayoutAccessors<THandle = unknown>(options: ProgramLayoutAccessorsOptions<THandle>) {
  const programPolicy = options.programPolicy;
  const assertProgramAlive = options.assertProgramAlive;
  const llamaKvCacheRequirementsCallback = options.llamaKvCacheRequirements;
  const bufferSlotMethodName = options.bufferSlotMethodName;

  if (!programPolicy || typeof programPolicy.bufferLayout !== "function") {
    throw new Error("createProgramLayoutAccessors requires programPolicy.bufferLayout");
  }
  if (assertProgramAlive !== undefined && typeof assertProgramAlive !== "function") {
    throw new Error("createProgramLayoutAccessors assertProgramAlive must be a function");
  }
  if (llamaKvCacheRequirementsCallback !== undefined && typeof llamaKvCacheRequirementsCallback !== "function") {
    throw new Error("createProgramLayoutAccessors llamaKvCacheRequirements must be a function");
  }

  function bufferLayout(program: ProgramHandleRecord<THandle>): ProgramBufferLayout {
    if (!program.bufferLayoutSnapshot) {
      program.bufferLayoutSnapshot = programPolicy.bufferLayout(program.handle);
    }
    return program.bufferLayoutSnapshot;
  }

  function bufferSlotNames(program: ProgramHandleRecord<THandle>): readonly string[] {
    return programBufferSlotNames(bufferLayout(program));
  }

  function bufferSlot(program: ProgramHandleRecord<THandle>, nameOrKind: unknown): ProgramBufferLayoutSlot | null {
    return programBufferSlot(bufferLayout(program), nameOrKind, bufferSlotMethodName || "Program.bufferSlot");
  }

  function kvCacheRequirements(program: ProgramHandleRecord<THandle>): LlamaKvCacheRequirements {
    if (typeof assertProgramAlive !== "function" || typeof llamaKvCacheRequirementsCallback !== "function") {
      throw new Error("createProgramLayoutAccessors requires LLaMA KV-cache callbacks");
    }
    assertProgramAlive(program.handle, "program");
    return llamaKvCacheRequirementsCallback(program.handle);
  }

  function kvCacheLayout(program: ProgramHandleRecord<THandle>): LlamaKvCacheLayout {
    if (typeof assertProgramAlive !== "function") {
      throw new Error("createProgramLayoutAccessors requires LLaMA KV-cache callbacks");
    }
    assertProgramAlive(program.handle, "program");
    return llamaKvCacheLayoutFromRequirements(kvCacheRequirements(program)) as LlamaKvCacheLayout;
  }

  return Object.freeze({
    bufferLayout,
    bufferSlotNames,
    bufferSlot,
    kvCacheRequirements,
    kvCacheLayout,
  });
}
