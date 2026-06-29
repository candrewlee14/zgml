import type {
  ModuleKernelParameterLayout,
  ProgramBufferLayout,
  TensorShape,
} from "../public_api.js";
import type {
  AdapterProgramDesc,
} from "./program_factory_surface.js";

type AdapterSessionProgramLike<TDesc extends AdapterProgramDesc = AdapterProgramDesc> = Readonly<{
  desc?: TDesc;
  parameterLayout(): ModuleKernelParameterLayout | Readonly<Record<string, unknown>> | null;
}>;

type AdapterSessionOutputShape = TensorShape | readonly number[] | null;
type AdapterSessionBufferLayout = ProgramBufferLayout | Readonly<Record<string, unknown>> | null;
type AdapterSessionParameterLayout = ModuleKernelParameterLayout | Readonly<Record<string, unknown>> | null;

type PreparedHostBind<TInput = unknown, TOutput = unknown> = Readonly<{
  kind: "host";
  input: TInput;
  output: TOutput;
  outputShape: AdapterSessionOutputShape;
  layout: AdapterSessionBufferLayout;
}>;

type PreparedNativeBind<TInput = unknown, TOutput = unknown, TOwnedBuffer = unknown> = Readonly<{
  kind: "native";
  boundInput: TInput;
  boundOutput: TOutput;
  ownedBuffers: readonly TOwnedBuffer[];
  outputShape: AdapterSessionOutputShape;
  layout: AdapterSessionBufferLayout;
}>;

export type AdapterPreparedSessionBind<TInput = unknown, TOutput = unknown, TOwnedBuffer = unknown> =
  | PreparedHostBind<TInput, TOutput>
  | PreparedNativeBind<TInput, TOutput, TOwnedBuffer>;

export type AdapterSessionConstructor<
  TSession,
  THandle = unknown,
  TDesc extends AdapterProgramDesc = AdapterProgramDesc,
  TInput = unknown,
  TOutput = unknown,
  TOwnedBuffer = unknown,
> = new (
  handle: THandle,
  desc: TDesc | undefined,
  input: TInput,
  output: TOutput,
  ownedBuffers: readonly TOwnedBuffer[],
  outputShape: AdapterSessionOutputShape,
  layout: AdapterSessionBufferLayout,
  parameterLayout: AdapterSessionParameterLayout,
) => TSession;

export type AdapterSessionFactorySurfaceOptions<
  TSession,
  THandle = unknown,
  TDesc extends AdapterProgramDesc = AdapterProgramDesc,
  TInput = unknown,
  TOutput = unknown,
  TOwnedBuffer = unknown,
> = Readonly<{
  getSessionClass(): AdapterSessionConstructor<TSession, THandle, TDesc, TInput, TOutput, TOwnedBuffer>;
  copyOwnedBuffers?: boolean;
}>;

export function createAdapterSessionFactory<
  TSession,
  THandle = unknown,
  TDesc extends AdapterProgramDesc = AdapterProgramDesc,
  TInput = unknown,
  TOutput = unknown,
  TOwnedBuffer = unknown,
>(
  options: AdapterSessionFactorySurfaceOptions<TSession, THandle, TDesc, TInput, TOutput, TOwnedBuffer>,
) {
  return function createSession(
    program: AdapterSessionProgramLike<TDesc>,
    descOrSessionHandle: TDesc | THandle,
    sessionHandleOrPrepared: THandle | AdapterPreparedSessionBind<TInput, TOutput, TOwnedBuffer>,
    maybePrepared?: AdapterPreparedSessionBind<TInput, TOutput, TOwnedBuffer>,
  ): TSession {
    const Session = options.getSessionClass();
    const desc = maybePrepared ? descOrSessionHandle : program.desc;
    const sessionHandle = maybePrepared ? sessionHandleOrPrepared : descOrSessionHandle;
    const prepared = (maybePrepared ?? sessionHandleOrPrepared) as AdapterPreparedSessionBind<TInput, TOutput, TOwnedBuffer>;
    if (prepared.kind === "native") {
      return new Session(
        sessionHandle as THandle,
        desc as TDesc | undefined,
        prepared.boundInput,
        prepared.boundOutput,
        options.copyOwnedBuffers ? prepared.ownedBuffers.slice() : prepared.ownedBuffers,
        prepared.outputShape,
        prepared.layout,
        program.parameterLayout(),
      );
    }
    return new Session(
      sessionHandle as THandle,
      desc as TDesc | undefined,
      prepared.input,
      prepared.output,
      [] as readonly TOwnedBuffer[],
      prepared.outputShape,
      prepared.layout,
      program.parameterLayout(),
    );
  };
}
