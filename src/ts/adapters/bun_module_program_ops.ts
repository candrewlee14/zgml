import {
  setPtr,
  setUSize,
} from "./bun_abi_words.js";
import {
  attachProgramCompileEvidence,
} from "../runtime/module_program_evidence.js";
import {
  moduleProgramAbiDescriptorFields,
  type ModuleProgramDesc,
} from "../runtime/module_program_desc.js";
import type {
  ProgramCompileEvidence,
  ProgramRequirements,
} from "../public_api.js";

type NativeHandle = number;
type NativeOut = BigUint64Array;

type ModuleProgramArtifacts = {
  readonly desc: Readonly<ModuleProgramDesc>;
  readonly evidence: ProgramCompileEvidence | Readonly<Record<string, unknown>> | null;
  readonly packed: {
    readonly inputShape: BigUint64Array;
    readonly opWords: BigUint64Array | null;
    readonly opCount: number;
  };
};

type BunModuleProgramSymbols = Readonly<{
  moduleProgramCompile(moduleDesc: BigUint64Array, compileDesc: NativeHandle | BigUint64Array, outProgram: NativeOut): number;
  moduleProgramGetRequirements(moduleDesc: BigUint64Array, compileDesc: NativeHandle | BigUint64Array, outRequirements: BigUint64Array): number;
}>;

export type BunModuleProgramOpsOptions<TProgram> = Readonly<{
  symbols: BunModuleProgramSymbols;
  check(code: number): void;
  handleOut(): NativeOut;
  readHandle(out: NativeOut): NativeHandle;
  pointerFor(value: BigUint64Array): NativeHandle;
  compileDesc(options: unknown): NativeHandle | BigUint64Array;
  moduleProgramCompileArtifactsFromCompiledSpec(spec: unknown): ModuleProgramArtifacts;
  programRequirementsFromAbiWords(out: BigUint64Array): ProgramRequirements;
  createProgram(handle: NativeHandle, desc: Readonly<ModuleProgramDesc>, evidence: ProgramCompileEvidence | Readonly<Record<string, unknown>> | null): TProgram;
}>;

export function createBunModuleProgramOps<TProgram>(options: BunModuleProgramOpsOptions<TProgram>) {
  const {
    symbols,
    check,
    handleOut,
    readHandle,
    pointerFor,
    compileDesc,
    moduleProgramCompileArtifactsFromCompiledSpec,
    programRequirementsFromAbiWords,
    createProgram,
  } = options;

  function moduleProgramAbiDesc(packed: ModuleProgramArtifacts["packed"]): BigUint64Array {
    const fields = moduleProgramAbiDescriptorFields(packed);
    const moduleDesc = new BigUint64Array(4);
    const view = new DataView(moduleDesc.buffer);
    setPtr(view, 0, pointerFor(fields.inputShape));
    setUSize(view, 8, fields.inputRank);
    setPtr(view, 16, fields.opWords ? pointerFor(fields.opWords) : 0);
    setUSize(view, 24, fields.opCount);
    return moduleDesc;
  }

  function compileModuleProgram(spec: unknown, compileOptions: unknown = {}): TProgram {
    const out = handleOut();
    const artifacts = moduleProgramCompileArtifactsFromCompiledSpec(spec);
    check(symbols.moduleProgramCompile(moduleProgramAbiDesc(artifacts.packed), compileDesc(compileOptions), out));
    return createProgram(readHandle(out), artifacts.desc, artifacts.evidence);
  }

  function moduleProgramRequirements(spec: unknown, compileOptions: unknown = {}): ProgramRequirements {
    const out = new BigUint64Array(16);
    const artifacts = moduleProgramCompileArtifactsFromCompiledSpec(spec);
    check(symbols.moduleProgramGetRequirements(moduleProgramAbiDesc(artifacts.packed), compileDesc(compileOptions), out));
    return programRequirementsFromAbiWords(out);
  }

  return Object.freeze({
    compileModuleProgram,
    moduleProgramRequirements,
    attachProgramCompileEvidence,
  });
}
