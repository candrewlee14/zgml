import type {
  ProgramCompileEvidence,
  TinyLinearDesc,
  TinyMlpDesc,
} from "../public_api.js";
import type {
  ModuleProgramDesc,
} from "../runtime/module_program_desc.js";

export type AdapterProgramDesc = TinyLinearDesc | TinyMlpDesc | Readonly<ModuleProgramDesc>;
export type AdapterProgramCompileEvidence = ProgramCompileEvidence | Readonly<Record<string, unknown>> | null;

export type AdapterProgramConstructor<
  TProgram,
  THandle = unknown,
  TDesc extends AdapterProgramDesc = AdapterProgramDesc,
  TEvidence extends AdapterProgramCompileEvidence = AdapterProgramCompileEvidence,
> = new (
  handle: THandle,
  desc: TDesc,
  evidence: TEvidence,
) => TProgram;

export type AdapterProgramFactorySurfaceOptions<
  TProgram,
  THandle = unknown,
  TDesc extends AdapterProgramDesc = AdapterProgramDesc,
  TEvidence extends AdapterProgramCompileEvidence = AdapterProgramCompileEvidence,
> = Readonly<{
  getProgramClass(): AdapterProgramConstructor<TProgram, THandle, TDesc, TEvidence>;
}>;

export function createAdapterProgramFactory<
  TProgram,
  THandle = unknown,
  TDesc extends AdapterProgramDesc = AdapterProgramDesc,
  TEvidence extends AdapterProgramCompileEvidence = AdapterProgramCompileEvidence,
>(
  options: AdapterProgramFactorySurfaceOptions<TProgram, THandle, TDesc, TEvidence>,
) {
  return function createProgram(
    handle: THandle,
    desc: TDesc,
    evidence: TEvidence,
  ): TProgram {
    const Program = options.getProgramClass();
    return new Program(handle, desc, evidence);
  };
}
