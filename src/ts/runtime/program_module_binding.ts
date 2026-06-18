type UnknownRecord = Record<string, unknown>;

import type {
  ModuleBindings,
  ModuleParameterPlacementOptions,
  ProgramModuleCompatibility,
  Session,
} from "../public_api.js";

export type ProgramModuleBindingOwnedBuffer = Readonly<{
  free(): void;
}>;

type ProgramModuleBindingSession<TNativeBuffer extends ProgramModuleBindingOwnedBuffer = ProgramModuleBindingOwnedBuffer> = Session & {
  _ownBuffers(ownedBuffers: readonly TNativeBuffer[]): Session;
};

type ProgramModuleBindingProgram<TNativeBuffer extends ProgramModuleBindingOwnedBuffer = ProgramModuleBindingOwnedBuffer> = Readonly<UnknownRecord & {
  bind(bindings: ModuleBindings): ProgramModuleBindingSession<TNativeBuffer>;
  moduleCompatibility(module: unknown, options: ModuleParameterPlacementOptions): ProgramModuleCompatibility;
}>;

type ProgramModuleBindingModule = Readonly<UnknownRecord & {
  placeParameters(program: unknown, options: ModuleParameterPlacementOptions): ModuleBindings;
}>;

export type ProgramModuleBindingOptions<TNativeBuffer extends ProgramModuleBindingOwnedBuffer = ProgramModuleBindingOwnedBuffer> = Readonly<{
  uniqueNativeBuffers(values: readonly unknown[]): readonly TNativeBuffer[];
}>;

export type ProgramModuleBindingHelpers<TNativeBuffer extends ProgramModuleBindingOwnedBuffer = ProgramModuleBindingOwnedBuffer> = Readonly<{
  bindModuleThroughProgram(program: unknown, module: unknown, options?: ModuleParameterPlacementOptions): Session;
}>;

export function createProgramModuleBindingHelpers<TNativeBuffer extends ProgramModuleBindingOwnedBuffer = ProgramModuleBindingOwnedBuffer>(
  options: ProgramModuleBindingOptions<TNativeBuffer>,
): ProgramModuleBindingHelpers<TNativeBuffer> {
  const uniqueNativeBuffers = options && options.uniqueNativeBuffers;
  if (typeof uniqueNativeBuffers !== "function") {
    throw new Error("createProgramModuleBindingHelpers requires uniqueNativeBuffers");
  }

  function bindModuleThroughProgram(program: unknown, module: unknown, options: ModuleParameterPlacementOptions = {}): Session {
    const programRecord = program as ProgramModuleBindingProgram<TNativeBuffer> | null | undefined;
    const moduleRecord = module as ProgramModuleBindingModule | null | undefined;
    if (!programRecord || typeof programRecord.bind !== "function" || typeof programRecord.moduleCompatibility !== "function") {
      throw new Error("Program.bindModule requires a Program with bind and moduleCompatibility");
    }
    if (!moduleRecord || typeof moduleRecord.placeParameters !== "function") {
      throw new Error("Program.bindModule requires an nn module with placeParameters(program, options)");
    }
    const compatibility = programRecord.moduleCompatibility(module, options);
    if (!compatibility.compatible) {
      throw new Error(compatibility.reason || "module is incompatible with this Program");
    }
    const bindings = moduleRecord.placeParameters(program, options);
    const ownedBuffers = uniqueNativeBuffers([bindings.weights, bindings.bias]);
    try {
      return programRecord.bind(bindings)._ownBuffers(ownedBuffers);
    } catch (err) {
      for (const buffer of ownedBuffers) buffer.free();
      throw err;
    }
  }

  return Object.freeze({
    bindModuleThroughProgram,
  }) as ProgramModuleBindingHelpers<TNativeBuffer>;
}
