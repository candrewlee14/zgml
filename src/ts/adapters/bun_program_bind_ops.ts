import {
  setPtr,
  setUSize,
} from "./bun_abi_words.js";
import {
  programBindDescriptorFields,
  type ProgramBindDescriptorFields,
} from "../runtime/program_bind_desc.js";
import type {
  ModuleKernelParameterLayout,
  ProgramBufferLayout,
  ProgramModuleCompatibility,
  TensorShape,
} from "../public_api.js";
import type {
  ProgramBindingDesc,
  ProgramBindingsInput,
} from "../runtime/tensor_placement.js";

type NativeHandle = number;
type NativeOut = BigUint64Array;

type FloatArrayLike = {
  readonly length: number;
};

type NativeBufferLike = {
  readonly handleValue: NativeHandle;
  writeFloat32(values: unknown): unknown;
  free(): void;
};

type PreparedOutputShape = TensorShape | readonly number[] | null;
type PreparedBufferLayout = ProgramBufferLayout | Readonly<Record<string, unknown>> | null;

type PreparedHostBind = {
  readonly kind: "host";
  readonly layout: PreparedBufferLayout;
  readonly weights: FloatArrayLike;
  readonly bias: FloatArrayLike | null;
  readonly input: FloatArrayLike | null;
  readonly output: FloatArrayLike | null;
  readonly outputShape: PreparedOutputShape;
};

type PreparedNativeBind<TNativeBuffer extends NativeBufferLike> = {
  readonly kind: "native";
  readonly layout: PreparedBufferLayout;
  readonly bindParams: ProgramBindingsInput;
  readonly ownedBuffers: readonly TNativeBuffer[];
  readonly boundInput: FloatArrayLike | TNativeBuffer | null;
  readonly boundOutput: FloatArrayLike | TNativeBuffer | null;
  readonly outputShape: PreparedOutputShape;
};

type PreparedBind<TNativeBuffer extends NativeBufferLike> =
  | PreparedHostBind
  | PreparedNativeBind<TNativeBuffer>;

type ProgramLike<TNativeBuffer extends NativeBufferLike> = {
  readonly handle: NativeHandle;
  bufferLayout?: () => ProgramBufferLayout | null;
  inputShape?: () => readonly number[] | null;
  outputShape?: () => readonly number[] | null;
  createBuffer?: (kind: string) => TNativeBuffer;
  compileEvidence?: () => unknown;
  moduleCompatibility?: (module: unknown, options: unknown) => ProgramModuleCompatibility | null | undefined;
  parameterLayout(): ModuleKernelParameterLayout | Readonly<Record<string, unknown>> | null;
};

type ProgramNativeBufferBindFields<TNativeBuffer extends NativeBufferLike> = {
  readonly weights: TNativeBuffer | null;
  readonly weightsLen: number;
  readonly bias: TNativeBuffer | null;
  readonly biasLen: number;
  readonly input: TNativeBuffer | null;
  readonly inputLen: number;
  readonly output: TNativeBuffer | null;
  readonly outputLen: number;
};

type BunProgramBindSymbols = Readonly<{
  sessionBind(handle: NativeHandle, desc: BigUint64Array, out: NativeOut): number;
  sessionBindBuffers(handle: NativeHandle, desc: BigUint64Array, out: NativeOut): number;
}>;

export type BunProgramBindOpsOptions<TNativeBuffer extends NativeBufferLike, TSession> = Readonly<{
  symbols: BunProgramBindSymbols;
  check(code: number): void;
  handleOut(): NativeOut;
  readHandle(out: NativeOut): NativeHandle;
  pointerFor(value: FloatArrayLike): NativeHandle;
  prepareProgramBind(program: ProgramLike<TNativeBuffer>, desc: ProgramBindingDesc, params: ProgramBindingsInput): PreparedBind<TNativeBuffer>;
  freeOwnedProgramBindBuffers(buffers: readonly TNativeBuffer[]): void;
  programNativeBufferBindFields(desc: ProgramBindingDesc, params: ProgramBindingsInput): ProgramNativeBufferBindFields<TNativeBuffer>;
  createSession(program: ProgramLike<TNativeBuffer>, desc: unknown, sessionHandle: NativeHandle, prepared: PreparedBind<TNativeBuffer>): TSession;
}>;

export function createBunProgramBindOps<TNativeBuffer extends NativeBufferLike, TSession>(
  options: BunProgramBindOpsOptions<TNativeBuffer, TSession>,
) {
  const {
    symbols,
    check,
    handleOut,
    readHandle,
    pointerFor,
    prepareProgramBind,
    freeOwnedProgramBindBuffers,
    programNativeBufferBindFields,
    createSession,
  } = options;

  function hostBindDesc(
    weights: FloatArrayLike,
    bias: FloatArrayLike | null,
    input: FloatArrayLike | null,
    output: FloatArrayLike | null,
  ): BigUint64Array {
    const fields = programBindDescriptorFields({ weights, bias, input, output });
    return packedBindDesc(fields, (value) => value && value.length !== 0 ? pointerFor(value) : 0);
  }

  function packedBindDesc<TValue>(
    fields: ProgramBindDescriptorFields<TValue>,
    handleFor: (value: TValue | null) => NativeHandle,
  ): BigUint64Array {
    const buf = new BigUint64Array(8);
    const view = new DataView(buf.buffer);
    setPtr(view, 0, handleFor(fields.weights));
    setUSize(view, 8, fields.weightsLen);
    setPtr(view, 16, handleFor(fields.bias));
    setUSize(view, 24, fields.biasLen);
    setPtr(view, 32, handleFor(fields.input));
    setUSize(view, 40, fields.inputLen);
    setPtr(view, 48, handleFor(fields.output));
    setUSize(view, 56, fields.outputLen);
    return buf;
  }

  function bufferBindDesc(desc: ProgramBindingDesc, params: ProgramBindingsInput): BigUint64Array {
    const fields = programNativeBufferBindFields(desc, params);
    return packedBindDesc(fields, (buffer) => buffer?.handleValue ?? 0);
  }

  function bindProgram(program: ProgramLike<TNativeBuffer>, desc: ProgramBindingDesc, params: ProgramBindingsInput): TSession {
    const prepared = prepareProgramBind(program, desc, params);
    const out = handleOut();
    if (prepared.kind === "native") {
      try {
        check(symbols.sessionBindBuffers(program.handle, bufferBindDesc(desc, prepared.bindParams), out));
      } catch (err) {
        freeOwnedProgramBindBuffers(prepared.ownedBuffers);
        throw err;
      }
      return createSession(program, desc, readHandle(out), prepared);
    }
    check(symbols.sessionBind(program.handle, hostBindDesc(prepared.weights, prepared.bias, null, null), out));
    return createSession(program, desc, readHandle(out), prepared);
  }

  return Object.freeze({
    bindProgram,
  });
}
