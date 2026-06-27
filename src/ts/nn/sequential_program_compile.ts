import { programCompileEvidenceFromCompiledSpec } from "../runtime/trace_compiler.js";
import type { ModuleFacadeTarget } from "../runtime/module_facade.js";
import type { SequentialCompiledProgramSpec } from "../runtime/module_compiler_policy.js";
import type {
  CompiledTrainingStep,
  CompileOptions,
  ModuleCompileSupport,
  ModuleParameterPlacementOptions,
  ModuleProgramTrace,
  NnModule,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
export type SequentialModuleRecord = ModuleFacadeTarget & Readonly<{
  readonly layers?: readonly SequentialModuleRecord[];
  readonly weight?: unknown;
  readonly bias?: unknown;
  readonly inFeatures?: unknown;
  readonly outFeatures?: unknown;
}>;
type ModuleRecord = SequentialModuleRecord;
type ModuleCompileOptions = Readonly<CompileOptions & Record<string, unknown>>;
type ModulePlacementOptions = Readonly<ModuleParameterPlacementOptions>;
type ModuleCompileSupportFn = (options?: ModuleCompileOptions) => ModuleCompileSupport;
type ModuleWithSequentialCompileSupport = ModuleRecord & {
  readonly compileSupport: ModuleCompileSupportFn;
};
export type SequentialCompiledSpec = SequentialCompiledProgramSpec;
type HookCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];
export type SequentialProgramAnalysis = Readonly<UnknownRecord & {
  readonly supported: boolean;
  readonly reason?: unknown;
  readonly support?: UnknownRecord;
  readonly compiled?: SequentialCompiledSpec | null;
  readonly trace?: ModuleProgramTrace | null;
}>;
type SequentialLayersForModule<TModule extends ModuleRecord = ModuleRecord> = (module: TModule) => readonly ModuleRecord[];
type SequentialCompileTinyLinearFn<
  TModule extends ModuleRecord = ModuleRecord,
  THooks extends SequentialProgramCompileHooks = SequentialProgramCompileHooks,
> = (
  compiled: SequentialCompiledSpec,
  analysis: SequentialProgramAnalysis,
  options: ModuleCompileOptions,
  hooks: THooks,
  module: TModule,
) => unknown;

export type SequentialProgramCompileCoreHooksInput = Readonly<Record<string, unknown> & {
  analyzeSequentialProgram: HookCallback<[layers: readonly ModuleRecord[], options: ModuleCompileOptions], SequentialProgramAnalysis>;
  moduleCompileSupport: HookCallback<[supported: boolean, reason: unknown, support?: UnknownRecord], unknown>;
  compileModuleProgram: HookCallback<[compiled: SequentialCompiledSpec, options: ModuleCompileOptions], unknown>;
  compileTrainingStep?: HookCallback<[module: ModuleRecord, optimizer: unknown, options?: UnknownRecord], CompiledTrainingStep>;
  packedSequentialProgramParameters?: HookCallback<[compiled: SequentialCompiledSpec], unknown>;
  attachProgramCompileEvidence?: HookCallback<[program: unknown, evidence: unknown], unknown>;
}>;

export type SequentialProgramCompileHooksInput = Readonly<SequentialProgramCompileCoreHooksInput & {
  placeModuleParameterBindings: HookCallback<[module: ModuleRecord, program: unknown, options: ModulePlacementOptions], unknown>;
}>;

export type SequentialProgramCompileHookOverrides = Readonly<{
  readonly packParameters?: (compiled: SequentialCompiledSpec) => unknown;
  readonly requirePackParameters?: boolean;
}>;

export type SequentialProgramCompileHooks = Readonly<{
  readonly analyzeSequentialProgram: (layers: readonly ModuleRecord[], options: ModuleCompileOptions) => SequentialProgramAnalysis;
  readonly moduleCompileSupport: (supported: boolean, reason: unknown, support?: UnknownRecord) => unknown;
  readonly compileModuleProgram: (compiled: SequentialCompiledSpec, options: ModuleCompileOptions) => unknown;
  readonly placeModuleParameterBindings: (module: ModuleRecord, program: unknown, options: ModulePlacementOptions) => unknown;
  readonly compileTrainingStep?: (module: ModuleRecord, optimizer: unknown, options?: UnknownRecord) => CompiledTrainingStep;
  readonly attachProgramCompileEvidence?: (program: unknown, evidence: unknown) => unknown;
  readonly packParameters?: (compiled: SequentialCompiledSpec) => unknown;
}>;

type SequentialCompilePrototype = {
  compileSupport?: (this: ModuleRecord, compileOptions?: ModuleCompileOptions) => unknown;
  canCompile?: (this: ModuleRecord, compileOptions?: ModuleCompileOptions) => boolean;
  bindParameters?: (this: ModuleRecord, bindOptions?: ModuleCompileOptions) => unknown;
  placeParameters?: (this: ModuleRecord, program: unknown, placementOptions?: ModulePlacementOptions) => unknown;
  compile?: (this: ModuleRecord, compileOptions?: ModuleCompileOptions) => unknown;
  compileForTraining?: (this: ModuleRecord, optimizer: unknown, compileOptions?: UnknownRecord) => CompiledTrainingStep;
  compile_for_training?: (this: ModuleRecord, optimizer: unknown, compileOptions?: UnknownRecord) => CompiledTrainingStep;
  trainingStep?: (this: ModuleRecord, optimizer: unknown, compileOptions?: UnknownRecord) => CompiledTrainingStep;
  training_step?: (this: ModuleRecord, optimizer: unknown, compileOptions?: UnknownRecord) => CompiledTrainingStep;
};

export type SequentialProgramCompileInstallOptions<
  THooks extends SequentialProgramCompileHooks = SequentialProgramCompileHooks,
> = Readonly<{
  readonly layersForModule: SequentialLayersForModule;
  readonly compileTinyLinear: SequentialCompileTinyLinearFn<ModuleRecord, THooks>;
  readonly bindParameters?: HookCallback<[
    module: ModuleRecord,
    hooks: THooks,
    bindOptions: ModuleCompileOptions,
    layersForModule: SequentialLayersForModule,
  ], unknown>;
  readonly fallbackReason?: string;
}>;

export function createSequentialProgramCompileHooks(
  options: SequentialProgramCompileHooksInput,
  label: string,
  extra: SequentialProgramCompileHookOverrides = {},
): SequentialProgramCompileHooks {
  const analyzeSequentialProgram = options && options.analyzeSequentialProgram;
  const moduleCompileSupport = options && options.moduleCompileSupport;
  const compileModuleProgram = options && options.compileModuleProgram;
  const placeModuleParameterBindings = options && options.placeModuleParameterBindings;
  const compileTrainingStep = options && options.compileTrainingStep;
  const attachProgramCompileEvidence = options && options.attachProgramCompileEvidence;
  const packParameters = extra.packParameters ?? (options && options.packedSequentialProgramParameters);
  if (
    typeof analyzeSequentialProgram !== "function" ||
    typeof moduleCompileSupport !== "function" ||
    typeof compileModuleProgram !== "function" ||
    typeof placeModuleParameterBindings !== "function"
  ) {
    throw new Error(`${label} factory requires sequential analysis, placement, and compile hooks`);
  }
  if (extra.requirePackParameters !== false && typeof packParameters !== "function") {
    throw new Error(`${label} factory requires a sequential parameter packing hook`);
  }
  return Object.freeze({
    analyzeSequentialProgram,
    moduleCompileSupport,
    compileModuleProgram,
    placeModuleParameterBindings,
    compileTrainingStep: typeof compileTrainingStep === "function" ? compileTrainingStep : undefined,
    attachProgramCompileEvidence,
    packParameters,
  });
}

export function sequentialProgramAnalysis<TModule extends ModuleRecord>(
  module: TModule,
  hooks: SequentialProgramCompileHooks,
  options: ModuleCompileOptions,
  layersForModule: SequentialLayersForModule<TModule>,
): SequentialProgramAnalysis {
  return hooks.analyzeSequentialProgram(layersForModule(module), options);
}

export function sequentialProgramCompileSupport<TModule extends ModuleRecord>(
  module: TModule,
  hooks: SequentialProgramCompileHooks,
  options: ModuleCompileOptions = {},
  layersForModule: SequentialLayersForModule<TModule>,
) {
  const analysis = sequentialProgramAnalysis(module, hooks, options, layersForModule);
  return hooks.moduleCompileSupport(analysis.supported, analysis.reason, analysis.support);
}

function requireSequentialCompileSupport(module: ModuleRecord): ModuleWithSequentialCompileSupport {
  if (typeof module.compileSupport !== "function") {
    throw new Error("sequential Program compile support has not been installed");
  }
  return module as ModuleWithSequentialCompileSupport;
}

export function sequentialProgramCanCompile(module: ModuleRecord, options: ModuleCompileOptions = {}) {
  return requireSequentialCompileSupport(module).compileSupport(options).supported;
}

export function sequentialProgramUnsupportedReason(module: ModuleRecord, options: ModuleCompileOptions = {}, fallbackReason?: string) {
  return requireSequentialCompileSupport(module).compileSupport(options).reason || fallbackReason || "nn module compile is unsupported";
}

export function sequentialProgramPlaceParameters(
  module: ModuleRecord,
  hooks: SequentialProgramCompileHooks,
  program: unknown,
  options: ModulePlacementOptions = {},
) {
  return hooks.placeModuleParameterBindings(module, program, options);
}

export function attachTinyLinearEvidence(
  program: unknown,
  hooks: SequentialProgramCompileHooks,
  compiled: SequentialCompiledSpec,
  analysis?: SequentialProgramAnalysis,
) {
  if (typeof hooks.attachProgramCompileEvidence !== "function") return program;
  const evidenceSpec = compiled.kind === "tiny-linear" && analysis?.trace
    ? { ...compiled, trace: analysis.trace }
    : compiled;
  return hooks.attachProgramCompileEvidence(program, programCompileEvidenceFromCompiledSpec(evidenceSpec));
}

export function sequentialProgramCompile<TModule extends ModuleRecord>(
  module: TModule,
  hooks: SequentialProgramCompileHooks,
  options: ModuleCompileOptions = {},
  layersForModule: SequentialLayersForModule<TModule>,
  compileTinyLinear: SequentialCompileTinyLinearFn<TModule>,
  fallbackReason?: string,
) {
  const analysis = sequentialProgramAnalysis(module, hooks, options, layersForModule);
  const compiled = analysis.compiled;
  if (compiled && compiled.kind === "tiny-linear") return compileTinyLinear(compiled, analysis, options, hooks, module);
  if (compiled && compiled.kind === "module") return hooks.compileModuleProgram(compiled, options);
  throw new Error(sequentialProgramUnsupportedReason(module, options, fallbackReason));
}

export function installSequentialProgramCompileMethods(
  proto: object,
  hooks: SequentialProgramCompileHooks,
  options: SequentialProgramCompileInstallOptions,
) {
  const target = proto as SequentialCompilePrototype;
  const layersForModule = options.layersForModule;
  const compileTinyLinear = options.compileTinyLinear;
  const bindParametersCallback = options.bindParameters;
  if (typeof layersForModule !== "function" || typeof compileTinyLinear !== "function") {
    throw new Error("sequential Program compile methods require layer and tiny-linear callbacks");
  }
  target.compileSupport = function compileSupport(this: ModuleRecord, compileOptions: ModuleCompileOptions = {}) {
    return sequentialProgramCompileSupport(this, hooks, compileOptions, layersForModule);
  };
  target.canCompile = function canCompile(this: ModuleRecord, compileOptions: ModuleCompileOptions = {}) {
    return sequentialProgramCanCompile(this, compileOptions);
  };
  if (typeof bindParametersCallback === "function") {
    target.bindParameters = function bindParameters(this: ModuleRecord, bindOptions: ModuleCompileOptions = {}) {
      return bindParametersCallback(this, hooks, bindOptions, layersForModule);
    };
  }
  target.placeParameters = function placeParameters(this: ModuleRecord, program: unknown, placementOptions: ModulePlacementOptions = {}) {
    return sequentialProgramPlaceParameters(this, hooks, program, placementOptions);
  };
  target.compile = function compile(this: ModuleRecord, compileOptions: ModuleCompileOptions = {}) {
    return sequentialProgramCompile(this, hooks, compileOptions, layersForModule, compileTinyLinear, options.fallbackReason);
  };
  const compileForTraining = function compileForTraining(this: ModuleRecord, optimizer: unknown, compileOptions: UnknownRecord = {}) {
    if (typeof hooks.compileTrainingStep !== "function") {
      throw new Error("module.compileForTraining requires a native Node or Bun training runtime");
    }
    return hooks.compileTrainingStep(this, optimizer, compileOptions);
  };
  target.compileForTraining = compileForTraining;
  target.compile_for_training = compileForTraining;
  target.trainingStep = compileForTraining;
  target.training_step = compileForTraining;
}
