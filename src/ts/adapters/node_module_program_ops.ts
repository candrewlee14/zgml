import {
  attachProgramCompileEvidence,
} from "../runtime/module_program_evidence.js";
import {
  programBindDescriptorRecord,
} from "../runtime/program_bind_desc.js";
import {
  moduleProgramAbiDescriptorFields,
  type ModuleProgramDesc,
} from "../runtime/module_program_desc.js";
import type {
  ProgramCompileEvidence,
} from "../public_api.js";
import type {
  ProgramBindingDesc,
  ProgramBindingsInput,
} from "../runtime/tensor_placement.js";

type NativeHandle = unknown;
type NativeOut = [NativeHandle | null];

type ModuleProgramArtifacts = {
  readonly desc: Readonly<ModuleProgramDesc>;
  readonly evidence: ProgramCompileEvidence | Readonly<Record<string, unknown>> | null;
  readonly packed: {
    readonly inputShape: readonly number[];
    readonly opWords: unknown;
    readonly opCount: number;
  };
};

type ProgramNativeBufferBindFields = {
  readonly weights?: { readonly handle: NativeHandle } | null;
  readonly weightsLen: number;
  readonly bias?: { readonly handle: NativeHandle } | null;
  readonly biasLen: number;
  readonly input?: { readonly handle: NativeHandle } | null;
  readonly inputLen: number;
  readonly output?: { readonly handle: NativeHandle } | null;
  readonly outputLen: number;
};

type NodeModuleProgramSymbols = Readonly<{
  moduleProgramCompile(desc: Record<string, unknown>, compile: Record<string, unknown>, out: NativeOut): number;
}>;

export type NodeModuleProgramOpsOptions<TProgram> = Readonly<{
  symbols: NodeModuleProgramSymbols;
  check(code: number): void;
  handleOut(): NativeOut;
  readHandle<T = NativeHandle>(out: readonly [T | null | undefined], name: string): T;
  compileDesc(options: unknown): Record<string, unknown>;
  moduleProgramCompileArtifactsFromCompiledSpec(spec: unknown): ModuleProgramArtifacts;
  programNativeBufferBindFields(desc: ProgramBindingDesc, params: ProgramBindingsInput): ProgramNativeBufferBindFields;
  createProgram(handle: NativeHandle, desc: Readonly<ModuleProgramDesc>, evidence: ProgramCompileEvidence | Readonly<Record<string, unknown>> | null): TProgram;
}>;

export function createNodeModuleProgramOps<TProgram>(options: NodeModuleProgramOpsOptions<TProgram>) {
  const {
    symbols,
    check,
    handleOut,
    readHandle,
    compileDesc,
    moduleProgramCompileArtifactsFromCompiledSpec,
    programNativeBufferBindFields,
    createProgram,
  } = options;

  function moduleProgramAbiDesc(packed: ModuleProgramArtifacts["packed"]): { readonly desc: Record<string, unknown> } {
    const fields = moduleProgramAbiDescriptorFields(packed);
    return {
      desc: {
        input_shape: fields.inputShape,
        input_rank: fields.inputRank,
        ops: fields.opWords,
        op_count: fields.opCount,
      },
    };
  }

  function compileModuleProgram(spec: unknown, compileOptions: unknown = {}): TProgram {
    const out = handleOut();
    const artifacts = moduleProgramCompileArtifactsFromCompiledSpec(spec);
    const abiDesc = moduleProgramAbiDesc(artifacts.packed);
    check(symbols.moduleProgramCompile(abiDesc.desc, compileDesc(compileOptions), out));
    return createProgram(readHandle(out, "program"), artifacts.desc, artifacts.evidence);
  }

  function bufferBindDescForTinyLinear(desc: ProgramBindingDesc, params: ProgramBindingsInput): Record<string, unknown> {
    const fields = programNativeBufferBindFields(desc, params);
    return programBindDescriptorRecord(fields, (buffer) => buffer ? buffer.handle : null) as Record<string, unknown>;
  }

  return Object.freeze({
    compileModuleProgram,
    attachProgramCompileEvidence,
    bufferBindDescForTinyLinear,
  });
}
