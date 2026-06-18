import {
  programBindDescriptorFields,
  programBindDescriptorRecord,
} from "../runtime/program_bind_desc.js";
import type {
  ProgramModuleCompatibility,
} from "../public_api.js";
import type {
  ProgramBindingDesc,
  ProgramBindingsInput,
} from "../runtime/tensor_placement.js";

type NativeHandle = unknown;
type NativeOut = [NativeHandle | null];

type PreparedHostBind = {
  readonly kind: "host";
  readonly layout: unknown;
  readonly weights: { readonly length: number };
  readonly bias: { readonly length: number } | null;
  readonly input: unknown;
  readonly output: unknown;
  readonly outputShape: unknown;
};

type PreparedNativeBind = {
  readonly kind: "native";
  readonly layout: unknown;
  readonly bindParams: ProgramBindingsInput;
  readonly ownedBuffers: readonly { free(): void }[];
  readonly boundInput: unknown;
  readonly boundOutput: unknown;
  readonly outputShape: unknown;
};

type PreparedBind = PreparedHostBind | PreparedNativeBind;

type ProgramLike = {
  readonly handle: NativeHandle;
  bufferLayout?: () => unknown;
  inputShape?: () => readonly number[] | null;
  outputShape?: () => readonly number[] | null;
  createBuffer?: (kind: string) => { free(): void; writeFloat32(values: unknown): unknown };
  compileEvidence?: () => unknown;
  moduleCompatibility?: (module: unknown, options: unknown) => ProgramModuleCompatibility | null | undefined;
};

type NodeProgramBindSymbols = Readonly<{
  sessionBind(handle: NativeHandle, desc: Record<string, unknown>, out: NativeOut): number;
  sessionBindBuffers(handle: NativeHandle, desc: Record<string, unknown>, out: NativeOut): number;
}>;

export type NodeProgramBindOpsOptions<TSession> = Readonly<{
  symbols: NodeProgramBindSymbols;
  check(code: number): void;
  handleOut(): NativeOut;
  readHandle<T = NativeHandle>(out: readonly [T | null | undefined], name: string): T;
  prepareProgramBind(program: ProgramLike, desc: ProgramBindingDesc, params: ProgramBindingsInput): PreparedBind;
  freeOwnedProgramBindBuffers(buffers: readonly { free(): void }[]): void;
  bufferBindDescForTinyLinear(desc: ProgramBindingDesc, params: ProgramBindingsInput): Record<string, unknown>;
  createSession(program: ProgramLike, sessionHandle: NativeHandle, prepared: PreparedBind): TSession;
}>;

export function createNodeProgramBindOps<TSession>(options: NodeProgramBindOpsOptions<TSession>) {
  const {
    symbols,
    check,
    handleOut,
    readHandle,
    prepareProgramBind,
    freeOwnedProgramBindBuffers,
    bufferBindDescForTinyLinear,
    createSession,
  } = options;

  function hostBindDesc(prepared: PreparedHostBind): Record<string, unknown> {
    return programBindDescriptorRecord(programBindDescriptorFields({
      weights: prepared.weights,
      bias: prepared.bias,
    })) as Record<string, unknown>;
  }

  function bindProgram(program: ProgramLike, desc: ProgramBindingDesc, params: ProgramBindingsInput): TSession {
    const prepared = prepareProgramBind(program, desc, params);
    const out = handleOut();
    if (prepared.kind === "native") {
      try {
        check(symbols.sessionBindBuffers(program.handle, bufferBindDescForTinyLinear(desc, prepared.bindParams), out));
      } catch (err) {
        freeOwnedProgramBindBuffers(prepared.ownedBuffers);
        throw err;
      }
    } else {
      check(symbols.sessionBind(program.handle, hostBindDesc(prepared), out));
    }
    return createSession(program, readHandle(out, "session"), prepared);
  }

  return Object.freeze({
    bindProgram,
  });
}
