import { createProgramLayoutAccessors } from "./program_layout.js";
import { createProgramParameterAccessors } from "./program_parameters.js";
import {
  createGenericProgramPolicyAccessors,
  createLlamaProgramPolicyAccessors,
} from "./program_policy_accessors.js";
import { createProgramResourceAccessors } from "./program_resources.js";
import {
  createGenericProgramShapeAccessors,
  createLlamaProgramShapeAccessors,
} from "./program_shapes.js";
import { createProgramSizingAccessors } from "./program_sizing.js";
import {
  programBindingPlanRejectionReason,
  programExecutionPlanRejectionReason,
} from "./execution_plan.js";
import type { LlamaBindOptionsRecord } from "./session_binding.js";
import type {
  ProgramBindingDesc,
  ProgramBindingPlanProgram,
} from "./tensor_placement.js";
import type {
  LlamaKvCache,
  LlamaKvCacheRequirements,
  ModuleParameterPlacementOptions,
  NativeBuffer,
  ProgramBindingPlan,
  ProgramBindings,
  ModuleKernelPlan,
  ProgramBufferLayout,
  ProgramBufferSizing,
  ProgramDevice,
  ProgramDeviceBufferImportSource,
  ProgramDeviceBufferKind,
  ProgramExecutionCapabilities,
  ProgramExecutionPlan,
  ProgramInspection,
  ProgramModelCompatibility,
  ProgramModuleCompatibility,
  ProgramRequirements,
  RuntimeProfile,
  ZgmlBackend,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
type UnknownCallback = (...args: readonly unknown[]) => unknown;

type ProgramEvidenceRecord = Readonly<Record<string, unknown>>;
type ProgramHandleRecord<THandle = unknown> = UnknownRecord & {
  handle: THandle;
  desc?: ProgramEvidenceRecord;
  compileEvidenceSnapshot?: unknown;
};
type ProgramBindingPlanEvidence = ProgramBindingPlan | ProgramEvidenceRecord;
type ProgramFacadeNativeBuffer = { readonly byteLength: number };
type ProgramFacadeModuleOptions = Readonly<ModuleParameterPlacementOptions>;
type ProgramFacadeBindingParams = ProgramBindings;
type ProgramFacadeResourceOptions = Readonly<{
  resource?: unknown;
  externalResource?: unknown;
  placement?: ZgmlBackend | string;
  backend?: ZgmlBackend | string;
  device?: ZgmlBackend | string;
}>;
type ProgramFacadePolicy<THandle = unknown> = Readonly<Record<string, unknown> & {
  requirements(handle: THandle): ProgramRequirements;
  bufferLayout(handle: THandle): ProgramBufferLayout;
  trace(handle: THandle, compileEvidence: unknown): unknown;
  compilerSignatures(handle: THandle, compileEvidence: unknown): unknown;
  tensorProgramIr(handle: THandle, compileEvidence: unknown): unknown;
  kernelPlan(handle: THandle, compileEvidence: unknown): ModuleKernelPlan | ProgramEvidenceRecord | null;
  shapeConstraints(handle: THandle, compileEvidence: unknown): ProgramEvidenceRecord | null;
  memoryLayout(handle: THandle, compileEvidence: unknown): ProgramEvidenceRecord | null;
  parameterLayout(handle: THandle, compileEvidence: unknown): ProgramEvidenceRecord | null;
  modelCompatibility(handle: THandle, model: unknown): ProgramModelCompatibility | ProgramEvidenceRecord;
  acceptsModel(handle: THandle, model: unknown): boolean;
  moduleCompatibility(handle: THandle, compileEvidence: unknown, module: unknown, moduleOptions?: ProgramFacadeModuleOptions): ProgramModuleCompatibility | null;
  acceptsModule(handle: THandle, compileEvidence: unknown, module: unknown, moduleOptions?: ProgramFacadeModuleOptions): boolean;
  capabilities(handle: THandle): ProgramExecutionCapabilities;
  canExecute(handle: THandle): boolean;
  canBindExternalResources(handle: THandle): boolean;
  hasFullDispatchPlan(handle: THandle): boolean;
  executionMode(handle: THandle): ProgramExecutionCapabilities["mode"] | string;
  matchesCapabilitySignature(handle: THandle, signature: unknown): boolean;
  runtimeProfile(handle: THandle): RuntimeProfile | ProgramEvidenceRecord;
  matchesRuntimeProfileSignature(handle: THandle, signature: unknown): boolean;
  resetRuntimeProfile(handle: THandle): unknown;
  free(program: ProgramHandleRecord<THandle>): void;
  dispose(program: ProgramHandleRecord<THandle>): void;
}>;

export type GenericProgramFacadeHelpersOptions<
  THandle = unknown,
  TNativeBuffer extends NativeBuffer = NativeBuffer,
  TProgramDevice = ProgramDevice,
  TProgramDesc = unknown,
  TOutputBufferOptions = ProgramFacadeResourceOptions,
  TCreateBufferOptions = ProgramFacadeResourceOptions,
  TProgramBufferOptions = ProgramFacadeResourceOptions,
> = Readonly<Partial<{
  assertProgramAlive: (handle: THandle, label: string) => void;
  programPolicy: ProgramFacadePolicy<THandle>;
  inspectExecutableProgram: (handle: THandle) => ProgramInspection | ProgramEvidenceRecord | unknown;
  programInputShape: (desc: TProgramDesc) => readonly number[] | null;
  programOutputShape: (desc: TProgramDesc) => readonly number[] | null;
  createProgramOutputBuffer: (handle: THandle, createOptions?: TOutputBufferOptions) => TNativeBuffer;
  createProgramNamedBuffer: (handle: THandle, kind: ProgramDeviceBufferKind, createOptions?: TCreateBufferOptions) => TNativeBuffer;
  createProgramBuffer: (handle: THandle, kind: ProgramDeviceBufferKind, createOptions?: TProgramBufferOptions) => TNativeBuffer;
  programDeviceHandle: (handle: THandle, placement?: ZgmlBackend | string) => number;
  createProgramDevice: (handle: THandle, placement?: ZgmlBackend | string) => TProgramDevice;
  importDeviceBuffer: (handle: THandle, kind: ProgramDeviceBufferKind, source: ProgramDeviceBufferImportSource | unknown) => TNativeBuffer;
  bindModuleThroughProgram: (program: ProgramHandleRecord<THandle>, module: unknown, moduleOptions?: ProgramFacadeModuleOptions) => unknown;
  programBindingPlan: (program: ProgramHandleRecord<THandle> & ProgramBindingPlanProgram, desc: ProgramBindingDesc, params?: ProgramFacadeBindingParams) => ProgramBindingPlanEvidence;
}>>;

export type LlamaProgramFacadeHelpersOptions<
  THandle = unknown,
  TNativeBuffer extends NativeBuffer = NativeBuffer,
  TProgramDevice = ProgramDevice,
  TLlamaKvCache = LlamaKvCache,
  TOutputBufferOptions = ProgramFacadeResourceOptions,
  TCreateBufferOptions = ProgramFacadeResourceOptions,
  TLlamaKvCacheOptions = ProgramFacadeResourceOptions,
> = Readonly<Partial<{
  assertProgramAlive: (handle: THandle, label: string) => void;
  programPolicy: ProgramFacadePolicy<THandle>;
  inspectLlamaProgram: (handle: THandle) => unknown;
  inspectExecutableProgram: (handle: THandle) => ProgramInspection | ProgramEvidenceRecord | unknown;
  llamaKvCacheRequirements: (handle: THandle) => LlamaKvCacheRequirements | ProgramEvidenceRecord;
  createProgramOutputBuffer: (handle: THandle, createOptions?: TOutputBufferOptions) => TNativeBuffer;
  createProgramNamedBuffer: (handle: THandle, kind: ProgramDeviceBufferKind, createOptions?: TCreateBufferOptions) => TNativeBuffer;
  programDeviceHandle: (handle: THandle, placement?: ZgmlBackend | string) => number;
  createProgramDevice: (handle: THandle, placement?: ZgmlBackend | string) => TProgramDevice;
  importDeviceBuffer: (handle: THandle, kind: ProgramDeviceBufferKind, source: ProgramDeviceBufferImportSource | unknown) => TNativeBuffer;
  createProgramLlamaKvCache: (handle: THandle, createOptions?: TLlamaKvCacheOptions) => TLlamaKvCache;
  bindLlamaProgramSession: (
    programHandle: THandle,
    program: unknown,
    bindOptions: LlamaBindOptionsRecord,
    bindSessionHandle: UnknownCallback,
    createSession: UnknownCallback,
  ) => unknown;
}>>;

function shapeSignature(shape: readonly unknown[] | null | undefined) {
  return Array.isArray(shape) ? shape.join("x") : "null";
}

function programExecutionPlanSignature(fields: {
  readonly programKind: string;
  readonly requirements: ProgramRequirements;
  readonly capabilities: ProgramExecutionCapabilities;
  readonly bufferSizing: ProgramBufferSizing;
  readonly bufferLayout: ProgramBufferLayout;
  readonly inputShape: readonly unknown[] | null | undefined;
  readonly outputShape: readonly unknown[] | null | undefined;
  readonly kernelPlan: ModuleKernelPlan | null | undefined;
  readonly moduleCompatibility: ProgramModuleCompatibility | null | undefined;
}) {
  return [
    "program-execution-plan",
    `kind=${fields.programKind}`,
    `requirements=${fields.requirements.signature}`,
    `capabilities=${fields.capabilities.signature}`,
    `canExecute=${fields.capabilities.canExecute === true ? 1 : 0}`,
    `mode=${fields.capabilities.mode}`,
    `bufferSizing=${fields.bufferSizing.signature}`,
    `bufferLayout=${fields.bufferLayout.signature}`,
    `inputShape=${shapeSignature(fields.inputShape)}`,
    `outputShape=${shapeSignature(fields.outputShape)}`,
    `kernelOps=${fields.kernelPlan && typeof fields.kernelPlan.opCount === "number" ? fields.kernelPlan.opCount : "null"}`,
    `module=${fields.moduleCompatibility ? fields.moduleCompatibility.signature : "null"}`,
  ].join("|");
}

export function createGenericProgramFacadeHelpers<
  THandle = unknown,
  TNativeBuffer extends NativeBuffer = NativeBuffer,
  TProgramDevice = ProgramDevice,
  TProgramDesc = unknown,
  TOutputBufferOptions = ProgramFacadeResourceOptions,
  TCreateBufferOptions = ProgramFacadeResourceOptions,
  TProgramBufferOptions = ProgramFacadeResourceOptions,
>(options: GenericProgramFacadeHelpersOptions<THandle, TNativeBuffer, TProgramDevice, TProgramDesc, TOutputBufferOptions, TCreateBufferOptions, TProgramBufferOptions>) {
  const assertProgramAlive = options.assertProgramAlive;
  const programPolicy = options.programPolicy;
  const inspectExecutableProgram = options.inspectExecutableProgram;
  const programInputShape = options.programInputShape;
  const programOutputShape = options.programOutputShape;
  const createProgramOutputBuffer = options.createProgramOutputBuffer;
  const createProgramNamedBuffer = options.createProgramNamedBuffer;
  const createProgramBuffer = options.createProgramBuffer;
  const programDeviceHandle = options.programDeviceHandle;
  const createProgramDevice = options.createProgramDevice;
  const importDeviceBufferCallback = options.importDeviceBuffer;
  const bindModuleThroughProgram = options.bindModuleThroughProgram;
  const programBindingPlanCallback = options.programBindingPlan;

  if (typeof assertProgramAlive !== "function") {
    throw new Error("createGenericProgramFacadeHelpers requires assertProgramAlive");
  }
  if (
    !programPolicy ||
    typeof programPolicy.requirements !== "function" ||
    typeof programPolicy.bufferLayout !== "function" ||
    typeof programPolicy.trace !== "function" ||
    typeof programPolicy.compilerSignatures !== "function" ||
    typeof programPolicy.tensorProgramIr !== "function" ||
    typeof programPolicy.kernelPlan !== "function" ||
    typeof programPolicy.shapeConstraints !== "function" ||
    typeof programPolicy.memoryLayout !== "function" ||
    typeof programPolicy.parameterLayout !== "function" ||
    typeof programPolicy.modelCompatibility !== "function" ||
    typeof programPolicy.acceptsModel !== "function" ||
    typeof programPolicy.moduleCompatibility !== "function" ||
    typeof programPolicy.acceptsModule !== "function" ||
    typeof programPolicy.capabilities !== "function" ||
    typeof programPolicy.canExecute !== "function" ||
    typeof programPolicy.canBindExternalResources !== "function" ||
    typeof programPolicy.hasFullDispatchPlan !== "function" ||
    typeof programPolicy.executionMode !== "function" ||
    typeof programPolicy.matchesCapabilitySignature !== "function" ||
    typeof programPolicy.runtimeProfile !== "function" ||
    typeof programPolicy.matchesRuntimeProfileSignature !== "function" ||
    typeof programPolicy.resetRuntimeProfile !== "function" ||
    typeof programPolicy.free !== "function" ||
    typeof programPolicy.dispose !== "function"
  ) {
    throw new Error("createGenericProgramFacadeHelpers requires programPolicy");
  }
  if (
    typeof inspectExecutableProgram !== "function" ||
    typeof programInputShape !== "function" ||
    typeof programOutputShape !== "function" ||
    typeof createProgramOutputBuffer !== "function" ||
    typeof createProgramNamedBuffer !== "function" ||
    typeof createProgramBuffer !== "function" ||
    typeof programDeviceHandle !== "function" ||
    typeof createProgramDevice !== "function" ||
    typeof importDeviceBufferCallback !== "function" ||
    typeof bindModuleThroughProgram !== "function" ||
    typeof programBindingPlanCallback !== "function"
  ) {
    throw new Error("createGenericProgramFacadeHelpers requires Program callbacks");
  }

  const programResources = createProgramResourceAccessors({
    assertProgramAlive,
    createProgramNamedBuffer,
    createProgramOutputBuffer,
    createProgramBuffer,
    programDeviceHandle,
    createProgramDevice,
    importDeviceBuffer: importDeviceBufferCallback,
  } as any);

  const programLayout = createProgramLayoutAccessors({
    programPolicy,
    bufferSlotMethodName: "Program.bufferSlot",
  });

  const programAccessors = createGenericProgramPolicyAccessors({
    programPolicy,
    assertProgramAlive,
    bindModuleThroughProgram,
  } as any);

  const programSizing = createProgramSizingAccessors({
    requirements: programAccessors.requirements,
  });

  const programShapes = createGenericProgramShapeAccessors({
    assertProgramAlive,
    shapeConstraints: programAccessors.shapeConstraints,
    inputShapeFallback: programInputShape,
    outputShapeFallback: programOutputShape,
  } as any);

  const programParameters = createProgramParameterAccessors({
    parameterLayout: programAccessors.parameterLayout,
  } as any);

  function inspect(program: ProgramHandleRecord<THandle>) {
    assertProgramAlive!(program.handle, "program");
    return inspectExecutableProgram!(program.handle);
  }

  function executionPlan(program: ProgramHandleRecord<THandle>, module: unknown = null, moduleOptions: ProgramFacadeModuleOptions = {}): ProgramExecutionPlan {
    const requirements = programAccessors.requirements(program);
    const capabilities = programAccessors.capabilities(program);
    const compileEvidence = programAccessors.compileEvidence(program);
    const kernelPlan = programAccessors.kernelPlan(program);
    const parameterLayout = programAccessors.parameterLayout(program);
    const bufferSizing = programSizing.bufferSizing(program);
    const bufferLayout = programLayout.bufferLayout(program);
    const inputShape = programShapes.inputShape(program);
    const outputShape = programShapes.outputShape(program);
    const moduleCompatibility = module === null || module === undefined
      ? null
      : programAccessors.moduleCompatibility(program, module, moduleOptions);
    const signature = programExecutionPlanSignature({
      programKind: "generic",
      requirements,
      capabilities,
      bufferSizing,
      bufferLayout,
      inputShape,
      outputShape,
      kernelPlan,
      moduleCompatibility,
    });
    return Object.freeze({
      kind: "zgml.program.execution-plan",
      signature,
      programKind: "generic",
      requirements,
      capabilities,
      compileEvidence: compileEvidence ?? null,
      canExecute: capabilities.canExecute === true,
      executionMode: capabilities.mode,
      canBindExternalResources: capabilities.canBindExternalResources === true,
      hasFullDispatchPlan: capabilities.hasFullDispatchPlan === true,
      capabilitySignature: capabilities.signature,
      bufferSizing,
      bufferLayout,
      inputShape,
      outputShape,
      trace: programAccessors.trace(program),
      tensorProgramIr: programAccessors.tensorProgramIr(program),
      kernelPlan,
      memoryLayout: programAccessors.memoryLayout(program),
      shapeConstraints: programAccessors.shapeConstraints(program),
      parameterLayout,
      parameterNames: programParameters.parameterNames(program),
      parameterInfos: programParameters.parameterInfos(program),
      moduleCompatibility,
      acceptsModule: moduleCompatibility ? moduleCompatibility.compatible === true : null,
      diagnostics: capabilities.diagnostics,
    });
  }

  function bindingPlan(program: ProgramHandleRecord<THandle>, params: ProgramFacadeBindingParams = {}) {
    assertProgramAlive!(program.handle, "program");
    return programBindingPlanCallback!(program, program.desc ?? {}, params);
  }

  function requireExecutionPlan(program: ProgramHandleRecord<THandle>, module: unknown = null, moduleOptions: ProgramFacadeModuleOptions = {}): ProgramExecutionPlan {
    const plan = executionPlan(program, module, moduleOptions);
    if (plan.canExecute === true && plan.acceptsModule !== false) return plan;
    throw new Error(`Program.requireExecutionPlan rejected execution plan: ${programExecutionPlanRejectionReason(plan)}`);
  }

  function requireBindingPlan(program: ProgramHandleRecord<THandle>, params: ProgramFacadeBindingParams = {}) {
    const plan = bindingPlan(program, params);
    if (plan && plan.accepted === true && plan.canBind === true) return plan;
    throw new Error(`Program.requireBindingPlan rejected bindings: ${programBindingPlanRejectionReason(plan)}`);
  }

  return Object.freeze({
    inspect,
    compileEvidence: programAccessors.compileEvidence,
    withCompileEvidence: programAccessors.withCompileEvidence,
    requirements: programAccessors.requirements,
    inputLen: programSizing.inputLen,
    outputLen: programSizing.outputLen,
    inputByteLength: programSizing.inputByteLength,
    outputByteLength: programSizing.outputByteLength,
    weightsLen: programSizing.weightsLen,
    weightsByteLength: programSizing.weightsByteLength,
    biasLen: programSizing.biasLen,
    biasByteLength: programSizing.biasByteLength,
    parameterLen: programSizing.parameterLen,
    parameterByteLength: programSizing.parameterByteLength,
    bufferSizing: programSizing.bufferSizing,
    matchesBufferSizingSignature: programSizing.matchesBufferSizingSignature,
    bufferLayout: programLayout.bufferLayout,
    bufferSlotNames: programLayout.bufferSlotNames,
    bufferSlot: programLayout.bufferSlot,
    inputShape: programShapes.inputShape,
    outputShape: programShapes.outputShape,
    trace: programAccessors.trace,
    compilerSignatures: programAccessors.compilerSignatures,
    tensorProgramIr: programAccessors.tensorProgramIr,
    kernelPlan: programAccessors.kernelPlan,
    shapeConstraints: programAccessors.shapeConstraints,
    memoryLayout: programAccessors.memoryLayout,
    parameterLayout: programAccessors.parameterLayout,
    parameterNames: programParameters.parameterNames,
    parameterInfos: programParameters.parameterInfos,
    parameterInfo: programParameters.parameterInfo,
    modelCompatibility: programAccessors.modelCompatibility,
    acceptsModel: programAccessors.acceptsModel,
    moduleCompatibility: programAccessors.moduleCompatibility,
    acceptsModule: programAccessors.acceptsModule,
    capabilities: programAccessors.capabilities,
    executionPlan,
    requireExecutionPlan,
    bindingPlan,
    requireBindingPlan,
    canExecute: programAccessors.canExecute,
    canBindExternalResources: programAccessors.canBindExternalResources,
    hasFullDispatchPlan: programAccessors.hasFullDispatchPlan,
    executionMode: programAccessors.executionMode,
    matchesCapabilitySignature: programAccessors.matchesCapabilitySignature,
    createBuffer: programResources.createBuffer,
    createOutputBuffer: programResources.createOutputBuffer,
    createWeightsBuffer: programResources.createWeightsBuffer,
    createBiasBuffer: programResources.createBiasBuffer,
    createInputBuffer: programResources.createInputBuffer,
    deviceHandle: programResources.deviceHandle,
    device: programResources.device,
    importDeviceBuffer: programResources.importDeviceBuffer,
    runtimeProfile: programAccessors.runtimeProfile,
    runtime_profile: programAccessors.runtimeProfile,
    matchesRuntimeProfileSignature: programAccessors.matchesRuntimeProfileSignature,
    matches_runtime_profile_signature: programAccessors.matchesRuntimeProfileSignature,
    resetRuntimeProfile: programAccessors.resetRuntimeProfile,
    reset_runtime_profile: programAccessors.resetRuntimeProfile,
    bindModule: programAccessors.bindModule,
    free: programAccessors.free,
    dispose: programAccessors.dispose,
  });
}

export function createLlamaProgramFacadeHelpers<
  THandle = unknown,
  TNativeBuffer extends NativeBuffer = NativeBuffer,
  TProgramDevice = ProgramDevice,
  TLlamaKvCache = LlamaKvCache,
  TOutputBufferOptions = ProgramFacadeResourceOptions,
  TCreateBufferOptions = ProgramFacadeResourceOptions,
  TLlamaKvCacheOptions = ProgramFacadeResourceOptions,
>(options: LlamaProgramFacadeHelpersOptions<THandle, TNativeBuffer, TProgramDevice, TLlamaKvCache, TOutputBufferOptions, TCreateBufferOptions, TLlamaKvCacheOptions>) {
  const assertProgramAlive = options.assertProgramAlive;
  const programPolicy = options.programPolicy;
  const inspectLlamaProgram = options.inspectLlamaProgram;
  const inspectExecutableProgram = options.inspectExecutableProgram;
  const llamaKvCacheRequirements = options.llamaKvCacheRequirements;
  const createProgramOutputBuffer = options.createProgramOutputBuffer;
  const createProgramNamedBuffer = options.createProgramNamedBuffer;
  const programDeviceHandle = options.programDeviceHandle;
  const createProgramDevice = options.createProgramDevice;
  const importDeviceBufferCallback = options.importDeviceBuffer;
  const createProgramLlamaKvCache = options.createProgramLlamaKvCache;
  const bindLlamaProgramSession = options.bindLlamaProgramSession;

  if (typeof assertProgramAlive !== "function") {
    throw new Error("createLlamaProgramFacadeHelpers requires assertProgramAlive");
  }
  if (
    !programPolicy ||
    typeof programPolicy.requirements !== "function" ||
    typeof programPolicy.bufferLayout !== "function" ||
    typeof programPolicy.modelCompatibility !== "function" ||
    typeof programPolicy.acceptsModel !== "function" ||
    typeof programPolicy.capabilities !== "function" ||
    typeof programPolicy.canExecute !== "function" ||
    typeof programPolicy.canBindExternalResources !== "function" ||
    typeof programPolicy.hasFullDispatchPlan !== "function" ||
    typeof programPolicy.executionMode !== "function" ||
    typeof programPolicy.matchesCapabilitySignature !== "function" ||
    typeof programPolicy.runtimeProfile !== "function" ||
    typeof programPolicy.matchesRuntimeProfileSignature !== "function" ||
    typeof programPolicy.resetRuntimeProfile !== "function" ||
    typeof programPolicy.free !== "function" ||
    typeof programPolicy.dispose !== "function"
  ) {
    throw new Error("createLlamaProgramFacadeHelpers requires programPolicy");
  }
  if (
    typeof inspectLlamaProgram !== "function" ||
    typeof inspectExecutableProgram !== "function" ||
    typeof llamaKvCacheRequirements !== "function" ||
    typeof createProgramOutputBuffer !== "function" ||
    typeof createProgramNamedBuffer !== "function" ||
    typeof programDeviceHandle !== "function" ||
    typeof createProgramDevice !== "function" ||
    typeof importDeviceBufferCallback !== "function" ||
    typeof createProgramLlamaKvCache !== "function" ||
    typeof bindLlamaProgramSession !== "function"
  ) {
    throw new Error("createLlamaProgramFacadeHelpers requires LLaMA Program callbacks");
  }

  const programResources = createProgramResourceAccessors({
    assertProgramAlive,
    createProgramNamedBuffer,
    createProgramOutputBuffer,
    programDeviceHandle,
    createProgramDevice,
    importDeviceBuffer: importDeviceBufferCallback,
  } as any);

  const programLayout = createProgramLayoutAccessors({
    programPolicy,
    assertProgramAlive,
    llamaKvCacheRequirements,
    bufferSlotMethodName: "TinyLlamaProgram.bufferSlot",
  } as any);

  const programAccessors = createLlamaProgramPolicyAccessors({
    programPolicy,
    assertProgramAlive,
    inspectLlamaProgram,
    inspectExecutableProgram,
    createProgramLlamaKvCache,
    bindLlamaProgramSession,
  } as any);

  const programSizing = createProgramSizingAccessors({
    requirements: programAccessors.requirements,
  });

  const programShapes = createLlamaProgramShapeAccessors({
    inspect: programAccessors.inspect,
  } as any);

  function executionPlan(program: ProgramHandleRecord<THandle>): ProgramExecutionPlan {
    const requirements = programAccessors.requirements(program);
    const capabilities = programAccessors.capabilities(program);
    const bufferSizing = programSizing.bufferSizing(program);
    const bufferLayout = programLayout.bufferLayout(program);
    const outputShape = programShapes.outputShape(program);
    const signature = programExecutionPlanSignature({
      programKind: "llama",
      requirements,
      capabilities,
      bufferSizing,
      bufferLayout,
      inputShape: null,
      outputShape,
      kernelPlan: null,
      moduleCompatibility: null,
    });
    return Object.freeze({
      kind: "zgml.program.execution-plan",
      signature,
      programKind: "llama",
      requirements,
      capabilities,
      compileEvidence: null,
      canExecute: capabilities.canExecute === true,
      executionMode: capabilities.mode,
      canBindExternalResources: capabilities.canBindExternalResources === true,
      hasFullDispatchPlan: capabilities.hasFullDispatchPlan === true,
      capabilitySignature: capabilities.signature,
      bufferSizing,
      bufferLayout,
      inputShape: null,
      outputShape,
      trace: null,
      tensorProgramIr: null,
      kernelPlan: null,
      memoryLayout: null,
      shapeConstraints: null,
      parameterLayout: null,
      parameterNames: Object.freeze([]),
      parameterInfos: Object.freeze([]),
      moduleCompatibility: null,
      acceptsModule: null,
      diagnostics: capabilities.diagnostics,
    });
  }

  function requireExecutionPlan(program: ProgramHandleRecord<THandle>): ProgramExecutionPlan {
    const plan = executionPlan(program);
    if (plan.canExecute === true) return plan;
    throw new Error(`Program.requireExecutionPlan rejected execution plan: ${programExecutionPlanRejectionReason(plan)}`);
  }

  return Object.freeze({
    inspect: programAccessors.inspect,
    vocabSize: programAccessors.vocabSize,
    requirements: programAccessors.requirements,
    inputLen: programSizing.inputLen,
    outputLen: programSizing.outputLen,
    inputByteLength: programSizing.inputByteLength,
    outputByteLength: programSizing.outputByteLength,
    weightsLen: programSizing.weightsLen,
    weightsByteLength: programSizing.weightsByteLength,
    biasLen: programSizing.biasLen,
    biasByteLength: programSizing.biasByteLength,
    parameterLen: programSizing.parameterLen,
    parameterByteLength: programSizing.parameterByteLength,
    bufferSizing: programSizing.bufferSizing,
    matchesBufferSizingSignature: programSizing.matchesBufferSizingSignature,
    bufferLayout: programLayout.bufferLayout,
    bufferSlotNames: programLayout.bufferSlotNames,
    bufferSlot: programLayout.bufferSlot,
    outputShape: programShapes.outputShape,
    modelCompatibility: programAccessors.modelCompatibility,
    acceptsModel: programAccessors.acceptsModel,
    capabilities: programAccessors.capabilities,
    executionPlan,
    requireExecutionPlan,
    canExecute: programAccessors.canExecute,
    canBindExternalResources: programAccessors.canBindExternalResources,
    hasFullDispatchPlan: programAccessors.hasFullDispatchPlan,
    executionMode: programAccessors.executionMode,
    matchesCapabilitySignature: programAccessors.matchesCapabilitySignature,
    createBuffer: programResources.createBuffer,
    createOutputBuffer: programResources.createOutputBuffer,
    deviceHandle: programResources.deviceHandle,
    device: programResources.device,
    importDeviceBuffer: programResources.importDeviceBuffer,
    kvCacheRequirements: programLayout.kvCacheRequirements,
    kvCacheLayout: programLayout.kvCacheLayout,
    createKvCache: programAccessors.createKvCache,
    inspectExecutable: programAccessors.inspectExecutable,
    runtimeProfile: programAccessors.runtimeProfile,
    runtime_profile: programAccessors.runtimeProfile,
    matchesRuntimeProfileSignature: programAccessors.matchesRuntimeProfileSignature,
    matches_runtime_profile_signature: programAccessors.matchesRuntimeProfileSignature,
    resetRuntimeProfile: programAccessors.resetRuntimeProfile,
    reset_runtime_profile: programAccessors.resetRuntimeProfile,
    bind: programAccessors.bind,
    free: programAccessors.free,
    dispose: programAccessors.dispose,
  });
}
