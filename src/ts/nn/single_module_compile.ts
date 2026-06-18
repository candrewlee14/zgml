import { annotateModuleBindings } from "../runtime/module_bindings.js";
import { compileSupportRejectionReason } from "../runtime/compile_support.js";
import type {
  CompileOptions,
  ModuleCompileSupport,
  ModuleParameterPlacementOptions,
  NnModule,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
type ModuleCompileOptions = Readonly<CompileOptions & Record<string, unknown>>;
type ModulePlacementOptions = Readonly<ModuleParameterPlacementOptions>;
type ModuleCompileSupportFn = (options?: ModuleCompileOptions) => ModuleCompileSupport;
export type SingleModuleRecord = Readonly<Partial<Record<keyof NnModule, unknown>> & {
  readonly kind?: unknown;
  readonly training?: NnModule["training"];
  readonly forward?: (...args: readonly unknown[]) => unknown;
  readonly compileSupport?: ModuleCompileSupportFn;
}>;
type ModuleRecord = SingleModuleRecord;
type ModuleWithSingleCompileSupport = ModuleRecord & {
  readonly compileSupport: ModuleCompileSupportFn;
};
export type SingleCompiledSpec = Readonly<Record<string, unknown> & {
  readonly kind?: unknown;
}>;
type HookCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];

export type SingleModuleProgramAnalysis = Readonly<UnknownRecord & {
  readonly supported: boolean;
  readonly reason?: unknown;
  readonly support?: UnknownRecord;
  readonly compiled?: SingleCompiledSpec | null;
}>;

export type SingleModuleCompileCoreHooksInput = Readonly<Record<string, unknown> & {
  analyzeSingleModuleProgram: HookCallback<[module: ModuleRecord, options: ModuleCompileOptions], SingleModuleProgramAnalysis>;
  moduleCompileSupport: HookCallback<[supported: boolean, reason: unknown, support?: UnknownRecord], unknown>;
  compileModuleProgram: HookCallback<[compiled: SingleCompiledSpec, options: ModuleCompileOptions], unknown>;
  packedSequentialProgramParameters?: HookCallback<[compiled: SingleCompiledSpec], unknown>;
}>;

export type SingleModuleCompileHooksInput = Readonly<SingleModuleCompileCoreHooksInput & {
  placeModuleParameterBindings: HookCallback<[module: ModuleRecord, program: unknown, options: ModulePlacementOptions], unknown>;
}>;

export type SingleModuleCompileHookOverrides = Readonly<{
  readonly packParameters?: HookCallback<[compiled: SingleCompiledSpec], unknown>;
  readonly requirePackParameters?: boolean;
}>;

export type SingleModuleCompileHooks = Readonly<{
  readonly analyzeSingleModuleProgram: HookCallback<[module: ModuleRecord, options: ModuleCompileOptions], SingleModuleProgramAnalysis>;
  readonly moduleCompileSupport: HookCallback<[supported: boolean, reason: unknown, support?: UnknownRecord], unknown>;
  readonly compileModuleProgram: HookCallback<[compiled: SingleCompiledSpec, options: ModuleCompileOptions], unknown>;
  readonly placeModuleParameterBindings: HookCallback<[module: ModuleRecord, program: unknown, options: ModulePlacementOptions], unknown>;
  readonly packParameters?: HookCallback<[compiled: SingleCompiledSpec], unknown>;
}>;

export type SingleModuleCompileInstallOptions = Readonly<{
  readonly compileSupport?: boolean;
  readonly bindParameters?: HookCallback<[module: ModuleRecord, hooks: SingleModuleCompileHooks, bindOptions: ModuleCompileOptions], unknown>;
  readonly fallbackReason?: string;
}>;

export type SingleModuleCompilePrototype = object & {
  compileSupport?: (this: ModuleRecord, compileOptions?: ModuleCompileOptions) => unknown;
  canCompile?: (this: ModuleRecord, compileOptions?: ModuleCompileOptions) => unknown;
  bindParameters?: (this: ModuleRecord, bindOptions?: ModuleCompileOptions) => unknown;
  placeParameters?: (this: ModuleRecord, program: unknown, placementOptions?: ModulePlacementOptions) => unknown;
  compile?: (this: ModuleRecord, compileOptions?: ModuleCompileOptions) => unknown;
};

export function createSingleModuleCompileHooks(
  options: SingleModuleCompileHooksInput,
  label: string,
  extra: SingleModuleCompileHookOverrides = {},
): SingleModuleCompileHooks {
  const analyzeSingleModuleProgram = options && options.analyzeSingleModuleProgram;
  const moduleCompileSupport = options && options.moduleCompileSupport;
  const compileModuleProgram = options && options.compileModuleProgram;
  const placeModuleParameterBindings = options && options.placeModuleParameterBindings;
  const packParameters = extra.packParameters ?? (options && options.packedSequentialProgramParameters);
  if (
    typeof analyzeSingleModuleProgram !== "function" ||
    typeof moduleCompileSupport !== "function" ||
    typeof compileModuleProgram !== "function" ||
    typeof placeModuleParameterBindings !== "function"
  ) {
    throw new Error(`${label} factory requires analysis, placement, and compile hooks`);
  }
  if (extra.requirePackParameters !== false && packParameters !== undefined && typeof packParameters !== "function") {
    throw new Error(`${label} factory requires a parameter packing hook`);
  }
  return Object.freeze({ analyzeSingleModuleProgram, moduleCompileSupport, compileModuleProgram, placeModuleParameterBindings, packParameters });
}

export function singleModuleCompileSupport(module: ModuleRecord, hooks: SingleModuleCompileHooks, options: ModuleCompileOptions = {}) {
  const analysis = hooks.analyzeSingleModuleProgram(module, options);
  return hooks.moduleCompileSupport(analysis.supported, analysis.reason, analysis.support);
}

function requireSingleCompileSupport(module: ModuleRecord): ModuleWithSingleCompileSupport {
  if (typeof module.compileSupport !== "function") {
    throw new Error("single-module Program compile support has not been installed");
  }
  return module as ModuleWithSingleCompileSupport;
}

export function singleModuleCanCompile(module: ModuleRecord, options: ModuleCompileOptions = {}) {
  return requireSingleCompileSupport(module).compileSupport(options).supported;
}

export function singleModuleCompiledSpec(module: ModuleRecord, hooks: SingleModuleCompileHooks, options: ModuleCompileOptions = {}) {
  return hooks.analyzeSingleModuleProgram(module, options).compiled;
}

export function singleModuleCompile(module: ModuleRecord, hooks: SingleModuleCompileHooks, options: ModuleCompileOptions = {}, fallbackReason?: string) {
  const compiled = singleModuleCompiledSpec(module, hooks, options);
  if (compiled && compiled.kind === "module") return hooks.compileModuleProgram(compiled, options);
  throw new Error(compileSupportRejectionReason(requireSingleCompileSupport(module).compileSupport(options), { fallbackReason }));
}

export function singleModulePlaceParameters(module: ModuleRecord, hooks: SingleModuleCompileHooks, program: unknown, options: ModulePlacementOptions = {}) {
  return hooks.placeModuleParameterBindings(module, program, options);
}

export function packedSingleModuleBindings(module: ModuleRecord, hooks: SingleModuleCompileHooks, options: ModuleCompileOptions = {}) {
  if (typeof hooks.packParameters !== "function") {
    throw new Error(`${module.kind || "module"} parameter packing hook is unavailable`);
  }
  const compiled = singleModuleCompiledSpec(module, hooks, options);
  if (compiled) return annotateModuleBindings(hooks.packParameters(compiled), module, options);
  throw new Error(compileSupportRejectionReason(requireSingleCompileSupport(module).compileSupport(options)));
}

export function emptySingleModuleBindings(module: ModuleRecord, options: ModuleCompileOptions = {}) {
  return annotateModuleBindings({ weights: new Float32Array(0) }, module, options);
}

export function installSingleModuleCompileMethods(proto: SingleModuleCompilePrototype, hooks: SingleModuleCompileHooks, options: SingleModuleCompileInstallOptions = {}) {
  const bindParametersCallback = options.bindParameters;
  if (options.compileSupport !== false) {
    proto.compileSupport = function compileSupport(this: ModuleRecord, compileOptions: ModuleCompileOptions = {}) {
      return singleModuleCompileSupport(this, hooks, compileOptions);
    };
  }
  proto.canCompile = function canCompile(this: ModuleRecord, compileOptions: ModuleCompileOptions = {}) {
    return singleModuleCanCompile(this, compileOptions);
  };
  proto.bindParameters = typeof bindParametersCallback === "function"
    ? function bindParameters(this: ModuleRecord, bindOptions: ModuleCompileOptions = {}) {
      return bindParametersCallback(this, hooks, bindOptions);
    }
    : function bindParameters(this: ModuleRecord, bindOptions: ModuleCompileOptions = {}) {
      return packedSingleModuleBindings(this, hooks, bindOptions);
    };
  proto.placeParameters = function placeParameters(this: ModuleRecord, program: unknown, placementOptions: ModulePlacementOptions = {}) {
    return singleModulePlaceParameters(this, hooks, program, placementOptions);
  };
  proto.compile = function compile(this: ModuleRecord, compileOptions: ModuleCompileOptions = {}) {
    return singleModuleCompile(this, hooks, compileOptions, options.fallbackReason);
  };
}
