import {
  type ModuleKernelParameterLayout,
  type ModuleKernelParameterLayoutEntry,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
type ProgramParameterEntryEvidence = Readonly<UnknownRecord & ModuleKernelParameterLayoutEntry>;
type ProgramParameterLayoutLike = Readonly<UnknownRecord & {
  parameters?: readonly ProgramParameterEntryEvidence[];
}>;
type ProgramParameterLayoutInput = ModuleKernelParameterLayout | ProgramParameterLayoutLike | null | undefined;
type ProgramParameterRecord = ModuleKernelParameterLayoutEntry | ProgramParameterEntryEvidence;
type ProgramParameterLayoutProvider<TProgram extends object = object> = (program: TProgram) => ProgramParameterLayoutInput;

function parametersFromLayout(layout: ProgramParameterLayoutInput): readonly ProgramParameterRecord[] {
  return layout && Array.isArray(layout.parameters) ? layout.parameters : [];
}

export function programParameterNames(layout: ProgramParameterLayoutInput): readonly string[] {
  return Object.freeze(parametersFromLayout(layout).map((param) => String(param.name)));
}

export function programParameterInfos(layout: ProgramParameterLayoutInput): readonly ModuleKernelParameterLayoutEntry[] {
  return Object.freeze(parametersFromLayout(layout).slice());
}

export function programParameterInfo(
  layout: ProgramParameterLayoutInput,
  nameOrIndex: unknown,
  label = "Program.parameterInfo",
): ModuleKernelParameterLayoutEntry | null {
  const parameters = parametersFromLayout(layout);
  if (parameters.length === 0) return null;
  if (typeof nameOrIndex === "number") {
    if (!Number.isInteger(nameOrIndex) || nameOrIndex < 0) {
      throw new Error(`${label} index must be a non-negative integer`);
    }
    return parameters[nameOrIndex] || null;
  }
  if (typeof nameOrIndex === "string") {
    return parameters.find((param) => param && param.name === nameOrIndex) || null;
  }
  throw new Error(`${label} requires a parameter name or index`);
}

export type ProgramParameterAccessorsOptions<TProgram extends object = object> = Readonly<{
  parameterLayout: ProgramParameterLayoutProvider<TProgram>;
}>;

export function createProgramParameterAccessors<TProgram extends object = object>(options: ProgramParameterAccessorsOptions<TProgram>) {
  const parameterLayout = options.parameterLayout;
  if (typeof parameterLayout !== "function") {
    throw new Error("createProgramParameterAccessors requires parameterLayout");
  }

  function layout(program: TProgram): ProgramParameterLayoutInput {
    return parameterLayout(program);
  }

  function parameterNames(program: TProgram): readonly string[] {
    return programParameterNames(layout(program));
  }

  function parameterInfos(program: TProgram): readonly ModuleKernelParameterLayoutEntry[] {
    return programParameterInfos(layout(program));
  }

  function parameterInfo(program: TProgram, nameOrIndex: unknown): ModuleKernelParameterLayoutEntry | null {
    return programParameterInfo(layout(program), nameOrIndex);
  }

  return Object.freeze({
    parameterNames,
    parameterInfos,
    parameterInfo,
  });
}
