import type {
  NativeBuffer,
  ProgramDevice,
  ProgramDeviceBufferImportSource,
  ProgramDeviceBufferKind,
  ZgmlBackend,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
type ProgramResourceRecord<THandle = unknown> = UnknownRecord & {
  handle: THandle;
};
type ProgramBufferResourceCreateOptions = Readonly<{
  resource?: unknown;
  externalResource?: unknown;
  placement?: ZgmlBackend | string;
  backend?: ZgmlBackend | string;
  device?: ZgmlBackend | string;
}>;
type ProgramNamedBufferCreateOptions = ProgramBufferResourceCreateOptions;

export type ProgramResourceAccessorsOptions<
  THandle = unknown,
  TNativeBuffer extends NativeBuffer = NativeBuffer,
  TProgramDevice extends ProgramDevice = ProgramDevice,
> = Readonly<{
  assertProgramAlive: (handle: THandle, label: string) => void;
  createProgramNamedBuffer: (handle: THandle, kind: ProgramDeviceBufferKind | unknown, createOptions?: ProgramNamedBufferCreateOptions) => TNativeBuffer;
  createProgramOutputBuffer: (handle: THandle, createOptions?: ProgramBufferResourceCreateOptions) => TNativeBuffer;
  createProgramBuffer?: (handle: THandle, kind: ProgramDeviceBufferKind, createOptions?: ProgramBufferResourceCreateOptions) => TNativeBuffer;
  programDeviceHandle: (handle: THandle, placement?: ZgmlBackend | string) => number;
  createProgramDevice: (handle: THandle, placement?: ZgmlBackend | string) => TProgramDevice;
  importDeviceBuffer: (handle: THandle, kind: ProgramDeviceBufferKind | unknown, source: ProgramDeviceBufferImportSource | unknown) => TNativeBuffer;
}>;

export function createProgramResourceAccessors<
  THandle = unknown,
  TNativeBuffer extends NativeBuffer = NativeBuffer,
  TProgramDevice extends ProgramDevice = ProgramDevice,
>(options: ProgramResourceAccessorsOptions<THandle, TNativeBuffer, TProgramDevice>) {
  const assertProgramAlive = options.assertProgramAlive;
  const createProgramNamedBuffer = options.createProgramNamedBuffer;
  const createProgramOutputBuffer = options.createProgramOutputBuffer;
  const createProgramBuffer = options.createProgramBuffer;
  const programDeviceHandle = options.programDeviceHandle;
  const createProgramDevice = options.createProgramDevice;
  const importDeviceBufferCallback = options.importDeviceBuffer;

  if (typeof assertProgramAlive !== "function") {
    throw new Error("createProgramResourceAccessors requires assertProgramAlive");
  }
  if (typeof createProgramNamedBuffer !== "function") {
    throw new Error("createProgramResourceAccessors requires createProgramNamedBuffer");
  }
  if (typeof createProgramOutputBuffer !== "function") {
    throw new Error("createProgramResourceAccessors requires createProgramOutputBuffer");
  }
  if (typeof programDeviceHandle !== "function") {
    throw new Error("createProgramResourceAccessors requires programDeviceHandle");
  }
  if (typeof createProgramDevice !== "function") {
    throw new Error("createProgramResourceAccessors requires createProgramDevice");
  }
  if (typeof importDeviceBufferCallback !== "function") {
    throw new Error("createProgramResourceAccessors requires importDeviceBuffer");
  }

  function assertAlive(program: ProgramResourceRecord<THandle>): void {
    assertProgramAlive(program.handle, "program");
  }

  function createBuffer(program: ProgramResourceRecord<THandle>, kind: ProgramDeviceBufferKind, createOptions: ProgramNamedBufferCreateOptions = {}): TNativeBuffer {
    assertAlive(program);
    return createProgramNamedBuffer(program.handle, kind, createOptions);
  }

  function createOutputBuffer(program: ProgramResourceRecord<THandle>, createOptions: ProgramBufferResourceCreateOptions = {}): TNativeBuffer {
    assertAlive(program);
    return createProgramOutputBuffer(program.handle, createOptions);
  }

  function createRoleBuffer(program: ProgramResourceRecord<THandle>, kind: ProgramDeviceBufferKind, createOptions: ProgramBufferResourceCreateOptions = {}): TNativeBuffer {
    assertAlive(program);
    if (typeof createProgramBuffer !== "function") {
      throw new Error("createProgramResourceAccessors requires createProgramBuffer");
    }
    return createProgramBuffer(program.handle, kind, createOptions);
  }

  function createWeightsBuffer(program: ProgramResourceRecord<THandle>, createOptions: ProgramBufferResourceCreateOptions = {}): TNativeBuffer {
    return createRoleBuffer(program, "weights", createOptions);
  }

  function createBiasBuffer(program: ProgramResourceRecord<THandle>, createOptions: ProgramBufferResourceCreateOptions = {}): TNativeBuffer {
    return createRoleBuffer(program, "bias", createOptions);
  }

  function createInputBuffer(program: ProgramResourceRecord<THandle>, createOptions: ProgramBufferResourceCreateOptions = {}): TNativeBuffer {
    return createRoleBuffer(program, "input", createOptions);
  }

  function deviceHandle(program: ProgramResourceRecord<THandle>, placement: ZgmlBackend | string = "webgpu"): number {
    assertAlive(program);
    return programDeviceHandle(program.handle, placement);
  }

  function device(program: ProgramResourceRecord<THandle>, placement: ZgmlBackend | string = "webgpu"): TProgramDevice {
    assertAlive(program);
    return createProgramDevice(program.handle, placement);
  }

  function importDeviceBuffer(
    program: ProgramResourceRecord<THandle>,
    kind: ProgramDeviceBufferKind,
    source: ProgramDeviceBufferImportSource | unknown = {},
  ): TNativeBuffer {
    assertAlive(program);
    return importDeviceBufferCallback(program.handle, kind, source);
  }

  return Object.freeze({
    createBuffer,
    createOutputBuffer,
    createWeightsBuffer,
    createBiasBuffer,
    createInputBuffer,
    deviceHandle,
    device,
    importDeviceBuffer,
  });
}
