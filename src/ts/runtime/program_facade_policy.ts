import { programBufferLayoutFromRequirements } from "./program_buffers.js";
import {
  programCapabilityCanBindExternalResources,
  programCapabilityCanExecute,
  programCapabilityHasFullDispatchPlan,
  programCompatibilityAccepted,
  programExecutionModeFromCapabilities,
} from "./inspection.js";
import { programCompilerSignaturesFromCompileEvidence } from "./compiler_signatures.js";
import {
  programKernelPlanFromCompileEvidence,
  programMemoryLayoutFromKernelPlan,
  programParameterLayoutFromKernelPlan,
  programShapeConstraintsFromKernelPlan,
  programTensorProgramIrFromCompileEvidence,
  programTraceFromCompileEvidence,
} from "./trace_compiler.js";
import { moduleCompatibilityForProgramEvidence } from "./module_compatibility.js";
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
  ProgramBufferLayout,
  ProgramExecutionCapabilities,
  ProgramExecutionMode,
  ProgramModelCompatibility,
  ProgramRequirements,
  RuntimeProfile,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
type ProgramCompileEvidenceRecord = ProgramCompileEvidence & UnknownRecord;
type ProgramCompatibilityModuleSource = Readonly<{
  compileSupport?: unknown;
}>;
type ProgramLifecycleRecord<THandle> = {
  handle: THandle;
};

function evidenceRecord(value: unknown): ProgramCompileEvidenceRecord | null | undefined {
  return value as ProgramCompileEvidenceRecord | null | undefined;
}

function moduleSource(value: unknown): ProgramCompatibilityModuleSource | null | undefined {
  if (value == null) return value;
  if (typeof value !== "object" && typeof value !== "function") return null;
  return value as ProgramCompatibilityModuleSource;
}

type ProgramRuntimeProfileEvidence = RuntimeProfile | Readonly<Record<string, unknown>>;
type ProgramExecutionCapabilitiesEvidence = ProgramExecutionCapabilities;
type ProgramModelCompatibilityEvidence = ProgramModelCompatibility | Readonly<Record<string, unknown>>;
type ProgramRequirementsEvidence = ProgramRequirements;
type ProgramModuleOptions = Readonly<ModuleParameterPlacementOptions>;

export type ProgramFacadePolicyHelpersOptions<THandle = unknown> = Readonly<{
  assertProgramAlive: (handle: THandle, label: string) => void;
  programRequirements: (handle: THandle) => ProgramRequirementsEvidence;
  programModelCompatibility: (handle: THandle, model: unknown) => ProgramModelCompatibilityEvidence;
  programExecutionCapabilities: (handle: THandle) => ProgramExecutionCapabilitiesEvidence;
  programRuntimeProfile: (handle: THandle) => ProgramRuntimeProfileEvidence;
  programResetRuntimeProfile: (handle: THandle) => unknown;
  programFree: (handle: THandle) => unknown;
  nullProgramHandle: THandle;
}>;

export function createProgramFacadePolicyHelpers<THandle = unknown>(options: ProgramFacadePolicyHelpersOptions<THandle>) {
  const assertProgramAlive = options.assertProgramAlive;
  const programRequirements = options.programRequirements;
  const programModelCompatibility = options.programModelCompatibility;
  const programExecutionCapabilities = options.programExecutionCapabilities;
  const programRuntimeProfile = options.programRuntimeProfile;
  const programResetRuntimeProfile = options.programResetRuntimeProfile;
  const programFree = options.programFree;
  const hasNullProgramHandle = Object.prototype.hasOwnProperty.call(options, "nullProgramHandle");
  const nullProgramHandle = hasNullProgramHandle ? options.nullProgramHandle : null as THandle;

  if (typeof assertProgramAlive !== "function") {
    throw new Error("createProgramFacadePolicyHelpers requires assertProgramAlive");
  }
  if (typeof programRequirements !== "function") {
    throw new Error("createProgramFacadePolicyHelpers requires programRequirements");
  }
  if (typeof programModelCompatibility !== "function") {
    throw new Error("createProgramFacadePolicyHelpers requires programModelCompatibility");
  }
  if (typeof programExecutionCapabilities !== "function") {
    throw new Error("createProgramFacadePolicyHelpers requires programExecutionCapabilities");
  }
  if (
    typeof programRuntimeProfile !== "function" ||
    typeof programResetRuntimeProfile !== "function" ||
    typeof programFree !== "function" ||
    !hasNullProgramHandle
  ) {
    throw new Error("createProgramFacadePolicyHelpers requires program lifecycle callbacks");
  }

  function requirements(handle: THandle): ProgramRequirementsEvidence {
    assertProgramAlive(handle, "program");
    return programRequirements(handle);
  }

  function bufferLayout(handle: THandle): ProgramBufferLayout {
    return programBufferLayoutFromRequirements(requirements(handle));
  }

  function trace(handle: THandle, compileEvidence: unknown): ModuleProgramTrace | null {
    assertProgramAlive(handle, "program");
    return programTraceFromCompileEvidence(evidenceRecord(compileEvidence));
  }

  function compilerSignatures(handle: THandle, compileEvidence: unknown): ModuleCompleteCompilerSignatures | null {
    assertProgramAlive(handle, "program");
    return programCompilerSignaturesFromCompileEvidence(evidenceRecord(compileEvidence));
  }

  function tensorProgramIr(handle: THandle, compileEvidence: unknown): ModuleTensorProgramIr | null {
    assertProgramAlive(handle, "program");
    return programTensorProgramIrFromCompileEvidence(evidenceRecord(compileEvidence));
  }

  function kernelPlan(handle: THandle, compileEvidence: unknown): ModuleKernelPlan | null {
    assertProgramAlive(handle, "program");
    return programKernelPlanFromCompileEvidence(evidenceRecord(compileEvidence));
  }

  function shapeConstraints(handle: THandle, compileEvidence: unknown): ModuleKernelShapeConstraints | null {
    return programShapeConstraintsFromKernelPlan(kernelPlan(handle, compileEvidence));
  }

  function memoryLayout(handle: THandle, compileEvidence: unknown): ModuleKernelMemoryLayout | null {
    return programMemoryLayoutFromKernelPlan(kernelPlan(handle, compileEvidence));
  }

  function parameterLayout(handle: THandle, compileEvidence: unknown): ModuleKernelParameterLayout | null {
    return programParameterLayoutFromKernelPlan(kernelPlan(handle, compileEvidence));
  }

  function modelCompatibility(handle: THandle, model: unknown): ProgramModelCompatibilityEvidence {
    assertProgramAlive(handle, "program");
    return programModelCompatibility(handle, model);
  }

  function acceptsModel(handle: THandle, model: unknown): boolean {
    return programCompatibilityAccepted(modelCompatibility(handle, model));
  }

  function moduleCompatibility(handle: THandle, compileEvidence: unknown, module: unknown, moduleOptions: ProgramModuleOptions = {}) {
    assertProgramAlive(handle, "program");
    return moduleCompatibilityForProgramEvidence(evidenceRecord(compileEvidence), moduleSource(module), moduleOptions);
  }

  function acceptsModule(handle: THandle, compileEvidence: unknown, module: unknown, moduleOptions: ProgramModuleOptions = {}): boolean {
    return programCompatibilityAccepted(moduleCompatibility(handle, compileEvidence, module, moduleOptions));
  }

  function capabilities(handle: THandle): ProgramExecutionCapabilitiesEvidence {
    assertProgramAlive(handle, "program");
    return programExecutionCapabilities(handle);
  }

  function canExecute(handle: THandle): boolean {
    return programCapabilityCanExecute(capabilities(handle));
  }

  function canBindExternalResources(handle: THandle): boolean {
    return programCapabilityCanBindExternalResources(capabilities(handle));
  }

  function hasFullDispatchPlan(handle: THandle): boolean {
    return programCapabilityHasFullDispatchPlan(capabilities(handle));
  }

  function executionMode(handle: THandle): ProgramExecutionMode {
    return programExecutionModeFromCapabilities(capabilities(handle)) as ProgramExecutionMode;
  }

  function matchesCapabilitySignature(handle: THandle, signature: unknown): boolean {
    return typeof signature === "string" && capabilities(handle).signature === signature;
  }

  function runtimeProfile(handle: THandle): ProgramRuntimeProfileEvidence {
    assertProgramAlive(handle, "program");
    return programRuntimeProfile(handle);
  }

  function matchesRuntimeProfileSignature(handle: THandle, signature: unknown): boolean {
    return typeof signature === "string" && runtimeProfile(handle).signature === signature;
  }

  function resetRuntimeProfile(handle: THandle): unknown {
    assertProgramAlive(handle, "program");
    return programResetRuntimeProfile(handle);
  }

  function free(program: ProgramLifecycleRecord<THandle>): void {
    if (program.handle !== nullProgramHandle) {
      programFree(program.handle);
      program.handle = nullProgramHandle;
    }
  }

  function dispose(program: ProgramLifecycleRecord<THandle>): void {
    return free(program);
  }

  return Object.freeze({
    requirements,
    bufferLayout,
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
    free,
    dispose,
  });
}
