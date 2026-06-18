import { completeProgramCompileEvidence, type CompileEvidence } from "./compiler_signatures.js";
import type { LlamaBindOptionsRecord } from "./session_binding.js";
import type {
  ModuleCompleteCompilerSignatures,
  ModuleKernelMemoryLayout,
  ModuleKernelParameterLayout,
  ModuleKernelPlan,
  ModuleKernelShapeConstraints,
  ModuleParameterPlacementOptions,
  ModuleProgramTrace,
  ModuleTensorProgramIr,
  ProgramCompileEvidence,
  ProgramExecutionCapabilities,
  ProgramExecutionMode,
  ProgramModelCompatibility,
  ProgramModuleCompatibility,
  ProgramRequirements,
  RuntimeProfile,
  ZgmlBackend,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
type ProgramEvidenceRecord = Readonly<Record<string, unknown>>;
type ProgramCompileEvidenceInput = CompileEvidence | null | undefined;
type LlamaProgramInspectionRecord = Readonly<UnknownRecord & {
  vocabSize: number;
}>;
type ProgramHandleRecord<THandle = unknown> = UnknownRecord & {
  handle: THandle;
  compileEvidenceSnapshot?: unknown;
  inspection?: LlamaProgramInspectionRecord | null;
};
type ProgramModuleOptions = Readonly<ModuleParameterPlacementOptions>;
type LlamaKvCacheCreateOptionsRecord = Readonly<{
  resource?: unknown;
  externalResource?: unknown;
  placement?: ZgmlBackend | string;
  backend?: ZgmlBackend | string;
  device?: ZgmlBackend | string;
}>;

type UnknownFunction<TArgs extends unknown[] = unknown[], TReturn = unknown> = (...args: TArgs) => TReturn;

function requireFunction<TArgs extends unknown[] = unknown[], TReturn = unknown>(value: unknown, label: string): UnknownFunction<TArgs, TReturn> {
  if (typeof value !== "function") throw new Error(`${label} must be a function`);
  return value as UnknownFunction<TArgs, TReturn>;
}

type ProgramPolicyAccessorsPolicy<
  THandle = unknown,
  TProgram extends ProgramHandleRecord<THandle> = ProgramHandleRecord<THandle>,
> = Readonly<Record<string, unknown> & {
  requirements(handle: THandle): ProgramRequirements;
  trace(handle: THandle, compileEvidence: ProgramCompileEvidence | null | undefined): ModuleProgramTrace | null;
  compilerSignatures(handle: THandle, compileEvidence: ProgramCompileEvidence | null | undefined): ModuleCompleteCompilerSignatures | null;
  tensorProgramIr(handle: THandle, compileEvidence: ProgramCompileEvidence | null | undefined): ModuleTensorProgramIr | null;
  kernelPlan(handle: THandle, compileEvidence: ProgramCompileEvidence | null | undefined): ModuleKernelPlan | null;
  shapeConstraints(handle: THandle, compileEvidence: ProgramCompileEvidence | null | undefined): ModuleKernelShapeConstraints | null;
  memoryLayout(handle: THandle, compileEvidence: ProgramCompileEvidence | null | undefined): ModuleKernelMemoryLayout | null;
  parameterLayout(handle: THandle, compileEvidence: ProgramCompileEvidence | null | undefined): ModuleKernelParameterLayout | null;
  modelCompatibility(handle: THandle, model: unknown): ProgramModelCompatibility | ProgramEvidenceRecord;
  acceptsModel(handle: THandle, model: unknown): boolean;
  moduleCompatibility(handle: THandle, compileEvidence: ProgramCompileEvidence | null | undefined, module: unknown, moduleOptions?: ProgramModuleOptions): ProgramModuleCompatibility | null;
  acceptsModule(handle: THandle, compileEvidence: ProgramCompileEvidence | null | undefined, module: unknown, moduleOptions?: ProgramModuleOptions): boolean;
  capabilities(handle: THandle): ProgramExecutionCapabilities;
  canExecute(handle: THandle): boolean;
  canBindExternalResources(handle: THandle): boolean;
  hasFullDispatchPlan(handle: THandle): boolean;
  executionMode(handle: THandle): ProgramExecutionMode | string;
  matchesCapabilitySignature(handle: THandle, signature: unknown): boolean;
  runtimeProfile(handle: THandle): RuntimeProfile | ProgramEvidenceRecord;
  matchesRuntimeProfileSignature(handle: THandle, signature: unknown): boolean;
  resetRuntimeProfile(handle: THandle): unknown;
  free(program: TProgram): void;
  dispose(program: TProgram): void;
}>;

export type GenericProgramPolicyAccessorsOptions<
  THandle = unknown,
  TProgram extends ProgramHandleRecord<THandle> = ProgramHandleRecord<THandle>,
> = Readonly<Partial<{
  programPolicy: ProgramPolicyAccessorsPolicy<THandle, TProgram>;
  assertProgramAlive: (handle: THandle, label: string) => void;
  bindModuleThroughProgram: (program: TProgram, module: unknown, moduleOptions?: ProgramModuleOptions) => unknown;
}>>;

export type LlamaProgramPolicyAccessorsOptions<
  THandle = unknown,
  TLlamaKvCache = unknown,
  TProgram extends ProgramHandleRecord<THandle> = ProgramHandleRecord<THandle>,
> = Readonly<Partial<{
  programPolicy: Pick<
    ProgramPolicyAccessorsPolicy<THandle, TProgram>,
    | "requirements"
    | "modelCompatibility"
    | "acceptsModel"
    | "capabilities"
    | "canExecute"
    | "canBindExternalResources"
    | "hasFullDispatchPlan"
    | "executionMode"
    | "matchesCapabilitySignature"
    | "runtimeProfile"
    | "matchesRuntimeProfileSignature"
    | "resetRuntimeProfile"
    | "free"
    | "dispose"
  >;
  assertProgramAlive: (handle: THandle, label: string) => void;
  inspectLlamaProgram: (handle: THandle) => LlamaProgramInspectionRecord;
  inspectExecutableProgram: (handle: THandle) => unknown;
  createProgramLlamaKvCache: (handle: THandle, createOptions?: LlamaKvCacheCreateOptionsRecord) => TLlamaKvCache;
  bindLlamaProgramSession: (
    programHandle: THandle,
    program: TProgram,
    bindOptions: LlamaBindOptionsRecord,
    bindSessionHandle: unknown,
    createSession: unknown,
  ) => unknown;
}>>;

export function createGenericProgramPolicyAccessors<
  THandle = unknown,
  TProgram extends ProgramHandleRecord<THandle> = ProgramHandleRecord<THandle>,
>(options: GenericProgramPolicyAccessorsOptions<THandle, TProgram>) {
  const programPolicy = options.programPolicy;
  const assertProgramAlive = options.assertProgramAlive;
  const bindModuleThroughProgram = options.bindModuleThroughProgram;

  if (!programPolicy) throw new Error("createGenericProgramPolicyAccessors requires programPolicy");
  const policy = programPolicy;
  const assertAlive = requireFunction<[THandle, string], void>(assertProgramAlive, "createGenericProgramPolicyAccessors assertProgramAlive");
  const bindModuleCallback = requireFunction<[TProgram, unknown, ProgramModuleOptions?], unknown>(bindModuleThroughProgram, "createGenericProgramPolicyAccessors bindModuleThroughProgram");

  const requiredPolicyMethods = [
    "requirements",
    "trace",
    "compilerSignatures",
    "tensorProgramIr",
    "kernelPlan",
    "shapeConstraints",
    "memoryLayout",
    "parameterLayout",
    "modelCompatibility",
    "acceptsModel",
    "moduleCompatibility",
    "acceptsModule",
    "capabilities",
    "canExecute",
    "canBindExternalResources",
    "hasFullDispatchPlan",
    "executionMode",
    "matchesCapabilitySignature",
    "runtimeProfile",
    "matchesRuntimeProfileSignature",
    "resetRuntimeProfile",
    "free",
    "dispose",
  ];
  for (const method of requiredPolicyMethods) {
    requireFunction(policy[method], `createGenericProgramPolicyAccessors programPolicy.${method}`);
  }

  function evidence(program: TProgram): ProgramCompileEvidence | null | undefined {
    return program.compileEvidenceSnapshot as ProgramCompileEvidence | null | undefined;
  }

  function compileEvidence(program: TProgram): ProgramCompileEvidence | null | undefined {
    return evidence(program);
  }

  function withCompileEvidence(program: TProgram, value: unknown) {
    program.compileEvidenceSnapshot = completeProgramCompileEvidence(value as ProgramCompileEvidenceInput);
    return program;
  }

  function requirements(program: TProgram) {
    return policy.requirements(program.handle);
  }

  function trace(program: TProgram) {
    return policy.trace(program.handle, evidence(program));
  }

  function compilerSignatures(program: TProgram) {
    return policy.compilerSignatures(program.handle, evidence(program));
  }

  function tensorProgramIr(program: TProgram) {
    return policy.tensorProgramIr(program.handle, evidence(program));
  }

  function kernelPlan(program: TProgram) {
    return policy.kernelPlan(program.handle, evidence(program));
  }

  function shapeConstraints(program: TProgram) {
    return policy.shapeConstraints(program.handle, evidence(program));
  }

  function memoryLayout(program: TProgram) {
    return policy.memoryLayout(program.handle, evidence(program));
  }

  function parameterLayout(program: TProgram) {
    return policy.parameterLayout(program.handle, evidence(program));
  }

  function modelCompatibility(program: TProgram, model: unknown) {
    return policy.modelCompatibility(program.handle, model);
  }

  function acceptsModel(program: TProgram, model: unknown) {
    return policy.acceptsModel(program.handle, model);
  }

  function moduleCompatibility(program: TProgram, module: unknown, moduleOptions: ProgramModuleOptions = {}) {
    return policy.moduleCompatibility(program.handle, evidence(program), module, moduleOptions);
  }

  function acceptsModule(program: TProgram, module: unknown, moduleOptions: ProgramModuleOptions = {}) {
    return policy.acceptsModule(program.handle, evidence(program), module, moduleOptions);
  }

  function capabilities(program: TProgram) {
    return policy.capabilities(program.handle);
  }

  function canExecute(program: TProgram) {
    return policy.canExecute(program.handle);
  }

  function canBindExternalResources(program: TProgram) {
    return policy.canBindExternalResources(program.handle);
  }

  function hasFullDispatchPlan(program: TProgram) {
    return policy.hasFullDispatchPlan(program.handle);
  }

  function executionMode(program: TProgram) {
    return policy.executionMode(program.handle);
  }

  function matchesCapabilitySignature(program: TProgram, signature: unknown) {
    return policy.matchesCapabilitySignature(program.handle, signature);
  }

  function runtimeProfile(program: TProgram) {
    return policy.runtimeProfile(program.handle);
  }

  function matchesRuntimeProfileSignature(program: TProgram, signature: unknown) {
    return policy.matchesRuntimeProfileSignature(program.handle, signature);
  }

  function resetRuntimeProfile(program: TProgram) {
    return policy.resetRuntimeProfile(program.handle);
  }

  function bindModule(program: TProgram, module: unknown, moduleOptions: ProgramModuleOptions = {}) {
    assertAlive(program.handle, "program");
    return bindModuleCallback(program, module, moduleOptions);
  }

  function free(program: TProgram) {
    return policy.free(program);
  }

  function dispose(program: TProgram) {
    return policy.dispose(program);
  }

  return Object.freeze({
    compileEvidence,
    withCompileEvidence,
    requirements,
    trace,
    compilerSignatures,
    tensorProgramIr,
    kernelPlan,
    shapeConstraints,
    memoryLayout,
    parameterLayout,
    modelCompatibility,
    acceptsModel,
    moduleCompatibility,
    acceptsModule,
    capabilities,
    canExecute,
    canBindExternalResources,
    hasFullDispatchPlan,
    executionMode,
    matchesCapabilitySignature,
    runtimeProfile,
    matchesRuntimeProfileSignature,
    resetRuntimeProfile,
    bindModule,
    free,
    dispose,
  });
}

export function createLlamaProgramPolicyAccessors<
  THandle = unknown,
  TLlamaKvCache = unknown,
  TProgram extends ProgramHandleRecord<THandle> = ProgramHandleRecord<THandle>,
>(options: LlamaProgramPolicyAccessorsOptions<THandle, TLlamaKvCache, TProgram>) {
  const programPolicy = options.programPolicy;
  const assertProgramAlive = options.assertProgramAlive;
  const inspectLlamaProgram = options.inspectLlamaProgram;
  const inspectExecutableProgram = options.inspectExecutableProgram;
  const createProgramLlamaKvCache = options.createProgramLlamaKvCache;
  const bindLlamaProgramSession = options.bindLlamaProgramSession;

  if (!programPolicy) throw new Error("createLlamaProgramPolicyAccessors requires programPolicy");
  const policy = programPolicy;
  const assertAlive = requireFunction<[THandle, string], void>(assertProgramAlive, "createLlamaProgramPolicyAccessors assertProgramAlive");
  const inspectLlama = requireFunction<[THandle], LlamaProgramInspectionRecord>(inspectLlamaProgram, "createLlamaProgramPolicyAccessors inspectLlamaProgram");
  const inspectExecutableCallback = requireFunction<[THandle], unknown>(inspectExecutableProgram, "createLlamaProgramPolicyAccessors inspectExecutableProgram");
  const createKvCacheCallback = requireFunction<[THandle, LlamaKvCacheCreateOptionsRecord?], TLlamaKvCache>(createProgramLlamaKvCache, "createLlamaProgramPolicyAccessors createProgramLlamaKvCache");
  const bindSessionCallback = requireFunction<[THandle, TProgram, LlamaBindOptionsRecord, unknown, unknown], unknown>(bindLlamaProgramSession, "createLlamaProgramPolicyAccessors bindLlamaProgramSession");

  const requiredPolicyMethods = [
    "requirements",
    "modelCompatibility",
    "acceptsModel",
    "capabilities",
    "canExecute",
    "canBindExternalResources",
    "hasFullDispatchPlan",
    "executionMode",
    "matchesCapabilitySignature",
    "runtimeProfile",
    "matchesRuntimeProfileSignature",
    "resetRuntimeProfile",
    "free",
    "dispose",
  ];
  for (const method of requiredPolicyMethods) {
    requireFunction(policy[method as keyof typeof policy], `createLlamaProgramPolicyAccessors programPolicy.${method}`);
  }

  function inspect(program: TProgram) {
    assertAlive(program.handle, "program");
    if (!program.inspection) program.inspection = inspectLlama(program.handle);
    return program.inspection;
  }

  function vocabSize(program: TProgram) {
    return inspect(program).vocabSize;
  }

  function requirements(program: TProgram) {
    return policy.requirements(program.handle);
  }

  function modelCompatibility(program: TProgram, model: unknown) {
    return policy.modelCompatibility(program.handle, model);
  }

  function acceptsModel(program: TProgram, model: unknown) {
    return policy.acceptsModel(program.handle, model);
  }

  function capabilities(program: TProgram) {
    return policy.capabilities(program.handle);
  }

  function canExecute(program: TProgram) {
    return policy.canExecute(program.handle);
  }

  function canBindExternalResources(program: TProgram) {
    return policy.canBindExternalResources(program.handle);
  }

  function hasFullDispatchPlan(program: TProgram) {
    return policy.hasFullDispatchPlan(program.handle);
  }

  function executionMode(program: TProgram) {
    return policy.executionMode(program.handle);
  }

  function matchesCapabilitySignature(program: TProgram, signature: unknown) {
    return policy.matchesCapabilitySignature(program.handle, signature);
  }

  function createKvCache(program: TProgram, createOptions: LlamaKvCacheCreateOptionsRecord = {}): TLlamaKvCache {
    assertAlive(program.handle, "program");
    return createKvCacheCallback(program.handle, createOptions) as TLlamaKvCache;
  }

  function inspectExecutable(program: TProgram) {
    assertAlive(program.handle, "program");
    return inspectExecutableCallback(program.handle);
  }

  function runtimeProfile(program: TProgram) {
    return policy.runtimeProfile(program.handle);
  }

  function matchesRuntimeProfileSignature(program: TProgram, signature: unknown) {
    return policy.matchesRuntimeProfileSignature(program.handle, signature);
  }

  function resetRuntimeProfile(program: TProgram) {
    return policy.resetRuntimeProfile(program.handle);
  }

  function bind(program: TProgram, bindOptions: LlamaBindOptionsRecord, bindSessionHandle: unknown, createSession: unknown) {
    assertAlive(program.handle, "program");
    return bindSessionCallback(program.handle, program, bindOptions, bindSessionHandle, createSession);
  }

  function free(program: TProgram) {
    return policy.free(program);
  }

  function dispose(program: TProgram) {
    return policy.dispose(program);
  }

  return Object.freeze({
    inspect,
    vocabSize,
    requirements,
    modelCompatibility,
    acceptsModel,
    capabilities,
    canExecute,
    canBindExternalResources,
    hasFullDispatchPlan,
    executionMode,
    matchesCapabilitySignature,
    createKvCache,
    inspectExecutable,
    runtimeProfile,
    matchesRuntimeProfileSignature,
    resetRuntimeProfile,
    bind,
    free,
    dispose,
  });
}
