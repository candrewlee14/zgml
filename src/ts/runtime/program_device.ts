import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";
import type {
  LlamaKvCache,
  NativeBuffer,
  ProgramDeviceBufferKind,
  ProgramDeviceInfo,
  ProgramDeviceImportBufferSource,
  ZgmlBackend,
} from "../public_api.js";

export type {
  ProgramDeviceBufferKind,
  ProgramDeviceInfo,
  ProgramDeviceImportBufferSource,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
type ProgramDeviceNativeBuffer = { readonly byteLength: number };
type ProgramDevicePlacement = ZgmlBackend | string;
type ProgramDeviceImportContext = Readonly<{
  placement: ProgramDevicePlacement;
  deviceHandle: number;
}>;
type ProgramDeviceInfoEvidenceRecord = Readonly<UnknownRecord & {
  kind?: unknown;
  placement?: unknown;
  handle?: unknown;
  signature?: unknown;
}>;

export const programDeviceManifest = Object.freeze({
  kind: "zgml-program-device",
  ...tsRuntimeManifestPolicy("src/ts/runtime/program_device.ts", "Program -> ProgramDevice -> NativeBuffer"),
});

function isRecord(value: unknown): value is UnknownRecord {
  return value !== null && typeof value === "object";
}

function programDeviceInfoSignature(placement: string, handle: number) {
  return `program-device-info|placement=${placement}|handle=${handle}`;
}

export function programDeviceInfo(placement: unknown, handle: unknown): ProgramDeviceInfo {
  if (typeof placement !== "string" || placement.length === 0) {
    throw new Error(`invalid ProgramDevice placement: ${placement}`);
  }
  if (typeof handle !== "number" || !Number.isSafeInteger(handle) || handle <= 0) {
    throw new Error(`invalid ProgramDevice handle: ${handle}`);
  }
  const normalizedHandle = handle;
  const normalizedPlacement = placement as ZgmlBackend;
  return Object.freeze({
    kind: "zgml.program-device.info",
    signature: programDeviceInfoSignature(normalizedPlacement, normalizedHandle),
    placement: normalizedPlacement,
    handle: normalizedHandle,
  });
}

export function isProgramDeviceInfo(info: unknown): info is ProgramDeviceInfo {
  if (!isRecord(info) || !Object.isFrozen(info)) return false;
  const record = info as ProgramDeviceInfoEvidenceRecord;
  if (record.kind !== "zgml.program-device.info") return false;
  if (typeof record.placement !== "string" || record.placement.length === 0) return false;
  if (typeof record.handle !== "number" || !Number.isSafeInteger(record.handle) || record.handle <= 0) return false;
  return record.signature === programDeviceInfoSignature(record.placement, record.handle);
}

export function requireProgramDeviceInfo(info: unknown): ProgramDeviceInfo {
  if (isProgramDeviceInfo(info)) return info;
  throw new Error("expected frozen ProgramDevice info evidence");
}

export const assertProgramDeviceInfo = requireProgramDeviceInfo;
export const assert_program_device_info = requireProgramDeviceInfo;

export function matchesProgramDeviceInfoSignature(info: unknown, signature: unknown) {
  return isProgramDeviceInfo(info) && typeof signature === "string" && info.signature === signature;
}

export const matches_program_device_info_signature = matchesProgramDeviceInfoSignature;

export type ProgramDeviceClassOptions<
  TNativeBuffer extends ProgramDeviceNativeBuffer = NativeBuffer,
  TLlamaKvCache = LlamaKvCache,
  THandle = unknown,
> = Readonly<Record<string, unknown> & {
  assertProgramAlive: (handle: THandle, label: string) => void;
  programDeviceHandle: (handle: THandle, placement: ProgramDevicePlacement) => number;
  createDeviceBuffer: (handle: THandle, kind: ProgramDeviceBufferKind, placement: ProgramDevicePlacement) => TNativeBuffer;
  importDeviceBuffer: (
    handle: THandle,
    kind: ProgramDeviceBufferKind,
    source: ProgramDeviceImportBufferSource | TNativeBuffer,
    context?: ProgramDeviceImportContext,
  ) => TNativeBuffer;
  isNativeBuffer: (value: unknown) => value is TNativeBuffer;
  createKvCache: (handle: THandle, createBuffer: (kind: ProgramDeviceBufferKind) => TNativeBuffer) => TLlamaKvCache;
}>;

export function createProgramDeviceClass<
  TNativeBuffer extends ProgramDeviceNativeBuffer = NativeBuffer,
  TLlamaKvCache = LlamaKvCache,
  THandle = unknown,
>(options: ProgramDeviceClassOptions<TNativeBuffer, TLlamaKvCache, THandle>) {
  const assertProgramAlive = options.assertProgramAlive;
  const programDeviceHandle = options.programDeviceHandle;
  const createDeviceBuffer = options.createDeviceBuffer;
  const importDeviceBufferCallback = options.importDeviceBuffer;
  const isNativeBuffer = options.isNativeBuffer;
  const createKvCache = options.createKvCache;

  if (typeof assertProgramAlive !== "function") {
    throw new Error("createProgramDeviceClass requires assertProgramAlive");
  }
  if (typeof programDeviceHandle !== "function") {
    throw new Error("createProgramDeviceClass requires programDeviceHandle");
  }
  if (typeof createDeviceBuffer !== "function") {
    throw new Error("createProgramDeviceClass requires createDeviceBuffer");
  }
  if (typeof importDeviceBufferCallback !== "function") {
    throw new Error("createProgramDeviceClass requires importDeviceBuffer");
  }
  if (typeof isNativeBuffer !== "function") {
    throw new Error("createProgramDeviceClass requires isNativeBuffer");
  }
  if (typeof createKvCache !== "function") {
    throw new Error("createProgramDeviceClass requires createKvCache");
  }

  return class ProgramDevice {
    programHandle: THandle;
    placement: ProgramDevicePlacement;
    handle: number;

    constructor(programHandle: THandle, placement: ProgramDevicePlacement = "webgpu") {
      assertProgramAlive(programHandle, "program");
      this.programHandle = programHandle;
      this.placement = placement;
      this.handle = programDeviceHandle(programHandle, placement);
    }

    info(): ProgramDeviceInfo {
      return programDeviceInfo(this.placement, this.handle);
    }

    matchesInfoSignature(signature: unknown): boolean {
      return matchesProgramDeviceInfoSignature(this.info(), signature);
    }

    createBuffer(kind: ProgramDeviceBufferKind): TNativeBuffer {
      assertProgramAlive(this.programHandle, "program");
      return createDeviceBuffer(this.programHandle, kind, this.placement);
    }

    createOutputBuffer(): TNativeBuffer {
      return this.createBuffer("output");
    }

    createWeightsBuffer(): TNativeBuffer {
      return this.createBuffer("weights");
    }

    createBiasBuffer(): TNativeBuffer {
      return this.createBuffer("bias");
    }

    createInputBuffer(): TNativeBuffer {
      return this.createBuffer("input");
    }

    createKvCache(): TLlamaKvCache {
      return createKvCache(this.programHandle, (kind: ProgramDeviceBufferKind) => this.createBuffer(kind));
    }

    importBuffer(kind: ProgramDeviceBufferKind, source: ProgramDeviceImportBufferSource | TNativeBuffer): TNativeBuffer {
      assertProgramAlive(this.programHandle, "program");
      if (isNativeBuffer(source)) return importDeviceBufferCallback(this.programHandle, kind, source);
      return importDeviceBufferCallback(this.programHandle, kind, source, {
        placement: this.placement,
        deviceHandle: this.handle,
      });
    }
  };
}
