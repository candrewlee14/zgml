"use strict";

import { tsProductManifestPolicy } from "./internal/product_manifest.js";

export type {
  ExternalResourceAccess,
  ExternalResourceAccessName,
  ExternalResourceOptions,
  HostGpuBufferImportSource,
  Program,
  ProgramBindingDiagnostic,
  ProgramBindingMode,
  ProgramBindingPlan,
  ProgramBindings,
  ProgramBufferCreateOptions,
  ProgramBufferKind,
  ProgramBufferLayout,
  ProgramBufferLayoutSlot,
  ProgramBufferSlot,
  ProgramBufferSizing,
  ProgramCompileEvidence,
  ProgramCreateBufferOptions,
  ProgramDevice,
  ProgramDeviceBufferKind,
  ProgramDeviceBufferImportDescriptor,
  ProgramDeviceBufferImportOptions,
  ProgramDeviceBufferImportSource,
  ProgramDeviceImportBufferSource,
  ProgramExecutionCapabilities,
  ProgramExecutionPlan,
  ProgramInputBinding,
  ProgramInspection,
  ProgramModelCompatibility,
  ProgramModuleCompatibility,
  ProgramNamespace,
  ProgramOutputBufferCreateOptions,
  ProgramOutputBufferSlot,
  ProgramOutputBinding,
  ProgramRequirements,
  ProgramRuntimeDiagnostic,
  PublicProgramNamespace,
  WebGpuInteropImportFields,
  WebGpuInteropImportSource,
  WebGpuInteropSymbols,
} from "./public_api.js";

export {
  acceptsProgramBindingPlan,
  acceptsProgramExecutionPlan,
  assert_program_binding_plan,
  assert_program_execution_plan,
  assertProgramBindingPlan,
  assertProgramExecutionPlan,
  matchesProgramBindingPlanSignature,
  matchesProgramExecutionPlanSignature,
  requireProgramBindingPlan,
  requireProgramExecutionPlan,
} from "./runtime/execution_plan.js";
export {
  isProgramCompileEvidence,
  requireProgramCompileEvidence,
  assertProgramCompileEvidence,
  assert_program_compile_evidence,
  matchesProgramCompileEvidenceSignature,
  matches_program_compile_evidence_signature,
  programCompileEvidenceSignature,
} from "./runtime/compiler_signatures.js";
export {
  createGenericProgramFacadeHelpers,
  createLlamaProgramFacadeHelpers,
} from "./runtime/program_facade.js";
export {
  createProgramFacadePolicyHelpers,
} from "./runtime/program_facade_policy.js";
export {
  createGenericProgramPolicyAccessors,
  createLlamaProgramPolicyAccessors,
} from "./runtime/program_policy_accessors.js";
export {
  createProgramBufferFactoryHelpers,
} from "./runtime/program_buffer_factory.js";
export {
  createProgramLayoutAccessors,
} from "./runtime/program_layout.js";
export {
  createProgramParameterAccessors,
} from "./runtime/program_parameters.js";
export {
  createProgramResourceAccessors,
} from "./runtime/program_resources.js";
export {
  createGenericProgramShapeAccessors,
  createLlamaProgramShapeAccessors,
} from "./runtime/program_shapes.js";
export {
  createProgramSizingAccessors,
} from "./runtime/program_sizing.js";
export {
  createProgramModuleBindingHelpers,
} from "./runtime/program_module_binding.js";
export {
  createProgramDeviceClass,
} from "./runtime/program_device.js";

export const programManifest = Object.freeze({
  kind: "zgml-program",
  ...tsProductManifestPolicy("src/ts/program.ts"),
  runtimePath: "Program -> Session -> StepParams",
});
