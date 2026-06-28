import {
  nn,
  tensor,
  type ModuleCompleteCompilerSignatures,
  type ModuleKernelBufferLayout,
  type ModuleKernelMemoryLayout,
  type ModuleKernelParameterLayout,
  type ModuleKernelPlan,
  type ModuleProgramTrace,
  type ModuleTensorProgramIr,
  type ModuleBindings,
  type ProgramBindingPlan,
  type ProgramBufferLayoutSlot,
  type Program,
  type ProgramCompileEvidence,
  type ProgramCompatibleModule,
  type ProgramExecutionCapabilities,
  type ProgramExecutionPlan,
  type ProgramModuleCompatibility,
  type ProgramRequirements,
  type RuntimeProfile,
  type RuntimeProfileExpectation,
  type Session,
  type SessionExecutionPlan,
  type SessionStepParamsCompatibility,
  type SessionStepContract,
  type Tensor,
} from "zgml";
import {
  nn as bunNn,
  tensor as bunTensor,
  type ModuleCompleteCompilerSignatures as BunModuleCompleteCompilerSignatures,
  type ModuleKernelBufferLayout as BunModuleKernelBufferLayout,
  type ModuleKernelMemoryLayout as BunModuleKernelMemoryLayout,
  type ModuleKernelParameterLayout as BunModuleKernelParameterLayout,
  type ModuleKernelPlan as BunModuleKernelPlan,
  type ModuleProgramTrace as BunModuleProgramTrace,
  type ModuleTensorProgramIr as BunModuleTensorProgramIr,
  type ModuleBindings as BunModuleBindings,
  type ProgramBindingPlan as BunProgramBindingPlan,
  type ProgramBufferLayoutSlot as BunProgramBufferLayoutSlot,
  type Program as BunProgram,
  type ProgramCompileEvidence as BunProgramCompileEvidence,
  type ProgramCompatibleModule as BunProgramCompatibleModule,
  type ProgramExecutionCapabilities as BunProgramExecutionCapabilities,
  type ProgramExecutionPlan as BunProgramExecutionPlan,
  type ProgramModuleCompatibility as BunProgramModuleCompatibility,
  type ProgramRequirements as BunProgramRequirements,
  type RuntimeProfile as BunRuntimeProfile,
  type RuntimeProfileExpectation as BunRuntimeProfileExpectation,
  type Session as BunSession,
  type SessionExecutionPlan as BunSessionExecutionPlan,
  type SessionStepParamsCompatibility as BunSessionStepParamsCompatibility,
  type SessionStepContract as BunSessionStepContract,
  type Tensor as BunTensor,
} from "zgml/bun";

const model = nn.linear(2, 1, { weights: [0.5, -0.5], bias: [0] });
const eager: Tensor = model.forward(tensor([1, 2], [2] as const));
const program: Program<readonly [2], readonly [1]> = model.compile({ backend: "cpu" });
const capabilities: ProgramExecutionCapabilities = program.capabilities();
const requirements: ProgramRequirements = program.requirements();
const compileEvidence: ProgramCompileEvidence | null = program.compileEvidence();
const trace: ModuleProgramTrace | null = program.trace();
const compilerSignatures: ModuleCompleteCompilerSignatures | null = program.compilerSignatures();
const tensorProgramIr: ModuleTensorProgramIr | null = program.tensorProgramIr();
const kernelPlan: ModuleKernelPlan | null = program.kernelPlan();
const bufferLayout: ModuleKernelBufferLayout | null = program.compileEvidence()?.kernelPlan.bufferLayout ?? null;
const programBufferLayout = program.bufferLayout();
const programBufferSlotNames: readonly string[] = program.bufferSlotNames();
const programInputSlot: ProgramBufferLayoutSlot | null = program.bufferSlot("input");
const programOutputSlot: ProgramBufferLayoutSlot | null = program.bufferSlot("output");
const memoryLayout: ModuleKernelMemoryLayout | null = program.memoryLayout();
const parameterLayout: ModuleKernelParameterLayout | null = program.parameterLayout();
const compatibility: ProgramModuleCompatibility = program.moduleCompatibility(model);
const executionPlan: ProgramExecutionPlan<readonly [2], readonly [1]> = program.executionPlan(model);
const modelBindings: ModuleBindings<readonly [2], readonly [1]> = model.bindParameters({ inputShape: [2] as const });
const modelBindingsAlias: ModuleBindings<readonly [2], readonly [1]> = model.bind_parameters({ inputShape: [2] as const });
const bindingPlan: ProgramBindingPlan<readonly [2], readonly [1]> = program.bindingPlan(modelBindings);
const bindingPlanAlias: ProgramBindingPlan<readonly [2], readonly [1]> = program.binding_plan(modelBindingsAlias);
const requiredBindingPlan: ProgramBindingPlan<readonly [2], readonly [1]> = program.requireBindingPlan(modelBindings);
const requiredBindingPlanAlias: ProgramBindingPlan<readonly [2], readonly [1]> = program.require_binding_plan(
  modelBindingsAlias,
);
const runtimeProfile: RuntimeProfile = program.runtimeProfile();
const runtimeProfileAlias: RuntimeProfile = program.runtime_profile();
const runtimeProfileExpectation: RuntimeProfileExpectation = program.runtimeProfileExpectation();
const hotRuntimeProfile: RuntimeProfileExpectation = program.requireHotRuntimeProfile();
const noFallbackRuntimeProfile: boolean = program.runtimeProfileHasNoFallback();
const noSyncRuntimeProfile: boolean = program.runtimeProfileHasNoSync();
const noInvalidRuntimePatchProfile: boolean = program.runtimeProfileHasNoInvalidRuntimePatches();
const accepted: boolean = program.acceptsModule(model);
const session: Session<readonly [2], readonly [1]> = program.bindModule(model);
const compiled: Tensor<readonly [1]> = session.stepTensor(tensor([1, 2], [2] as const));
const callableCompiled = model.inference({ backend: "cpu", inputShape: [2] as const });
const callableCompiledTensor: Tensor<readonly [1]> = callableCompiled(tensor([1, 2], [2] as const));
const compatibleModule: ProgramCompatibleModule<readonly [2], readonly [1], typeof model> = model;
const incompatibleModel = nn.linear(3, 1);
const incompatibleModuleCompatibility: ProgramModuleCompatibility = program.moduleCompatibility(incompatibleModel);
const incompatibleModuleAccepted: boolean = program.acceptsModule(incompatibleModel);
// @ts-expect-error Program binding preserves the compiled input/output shape contract.
const invalidBoundSession: Session<readonly [2], readonly [1]> = program.bindModule(incompatibleModel);
// @ts-expect-error Program execution plans preserve the compiled input/output shape contract.
const invalidModuleExecutionPlan: ProgramExecutionPlan<readonly [2], readonly [1]> = program.executionPlan(incompatibleModel);
// @ts-expect-error ProgramCompatibleModule narrows incompatible modules to never.
const invalidCompatibleModule: ProgramCompatibleModule<readonly [2], readonly [1], typeof incompatibleModel> = incompatibleModel;
const contract: SessionStepContract = session.stepContract();
const sessionBufferSlotNames: readonly string[] = session.bufferSlotNames();
const sessionInputSlot: ProgramBufferLayoutSlot | null = session.bufferSlot("input");
const sessionRuntimeProfile: RuntimeProfile = session.runtimeProfile();
const sessionRuntimeProfileAlias: RuntimeProfile = session.runtime_profile();
const sessionRuntimeProfileExpectation: RuntimeProfileExpectation = session.runtimeProfileExpectation();
const sessionHotRuntimeProfile: RuntimeProfileExpectation = session.requireHotRuntimeProfile();
const sessionNoFallbackRuntimeProfile: boolean = session.runtimeProfileHasNoFallback();
const sessionNoSyncRuntimeProfile: boolean = session.runtimeProfileHasNoSync();
const sessionNoInvalidRuntimePatchProfile: boolean = session.runtimeProfileHasNoInvalidRuntimePatches();
const hotParams = { input: tensor([1, 2], [2] as const), output: false };
const hotCompatibility: SessionStepParamsCompatibility = session.requireHotStepParams(hotParams);
const hotCompatibilityAlias: SessionStepParamsCompatibility = session.require_hot_step_params(hotParams);
const hotPlan: SessionExecutionPlan<readonly [2], readonly [1]> = session.hotPathPlan(hotParams);
const hotPlanAlias: SessionExecutionPlan<readonly [2], readonly [1]> = session.hot_path_plan(hotParams);
const rejectedCompatibility: SessionStepParamsCompatibility = session.stepParamsCompatibility({
  input: tensor([1, 2, 3], [3] as const),
  output: false,
});
const rejectedCompatibilityAlias: SessionStepParamsCompatibility = session.step_params_compatibility({
  input: tensor([1, 2, 3], [3] as const),
  output: false,
});
const inputShape: readonly number[] = program.inputShape();
const outputShape: readonly number[] = program.outputShape();

void eager;
void capabilities;
void requirements;
void compileEvidence;
void trace;
void compilerSignatures;
void tensorProgramIr;
void kernelPlan;
void bufferLayout;
void programBufferLayout;
void programBufferSlotNames;
void programInputSlot;
void programOutputSlot;
void memoryLayout;
void parameterLayout;
void compatibility;
void executionPlan;
void modelBindings;
void modelBindingsAlias;
void bindingPlan;
void bindingPlanAlias;
void requiredBindingPlan;
void requiredBindingPlanAlias;
void runtimeProfile;
void runtimeProfileAlias;
void runtimeProfileExpectation;
void hotRuntimeProfile;
void noFallbackRuntimeProfile;
void noSyncRuntimeProfile;
void noInvalidRuntimePatchProfile;
void accepted;
void compiled;
void callableCompiledTensor;
void compatibleModule;
void incompatibleModuleCompatibility;
void incompatibleModuleAccepted;
void invalidBoundSession;
void invalidModuleExecutionPlan;
void invalidCompatibleModule;
void contract;
void sessionBufferSlotNames;
void sessionInputSlot;
void sessionRuntimeProfile;
void sessionRuntimeProfileAlias;
void sessionRuntimeProfileExpectation;
void sessionHotRuntimeProfile;
void sessionNoFallbackRuntimeProfile;
void sessionNoSyncRuntimeProfile;
void sessionNoInvalidRuntimePatchProfile;
void hotCompatibility;
void hotCompatibilityAlias;
void hotPlan;
void hotPlanAlias;
void rejectedCompatibility;
void rejectedCompatibilityAlias;
void inputShape;
void outputShape;

const bunModel = bunNn.linear(2, 1, { weights: [0.5, -0.5], bias: [0] });
const bunEager: BunTensor = bunModel.forward(bunTensor([1, 2], [2] as const));
const bunProgram: BunProgram<readonly [2], readonly [1]> = bunModel.compile({ backend: "cpu" });
const bunCapabilities: BunProgramExecutionCapabilities = bunProgram.capabilities();
const bunRequirements: BunProgramRequirements = bunProgram.requirements();
const bunCompileEvidence: BunProgramCompileEvidence | null = bunProgram.compileEvidence();
const bunTrace: BunModuleProgramTrace | null = bunProgram.trace();
const bunCompilerSignatures: BunModuleCompleteCompilerSignatures | null = bunProgram.compilerSignatures();
const bunTensorProgramIr: BunModuleTensorProgramIr | null = bunProgram.tensorProgramIr();
const bunKernelPlan: BunModuleKernelPlan | null = bunProgram.kernelPlan();
const bunBufferLayout: BunModuleKernelBufferLayout | null = bunProgram.compileEvidence()?.kernelPlan.bufferLayout ?? null;
const bunProgramBufferLayout = bunProgram.bufferLayout();
const bunProgramBufferSlotNames: readonly string[] = bunProgram.bufferSlotNames();
const bunProgramInputSlot: BunProgramBufferLayoutSlot | null = bunProgram.bufferSlot("input");
const bunProgramOutputSlot: BunProgramBufferLayoutSlot | null = bunProgram.bufferSlot("output");
const bunMemoryLayout: BunModuleKernelMemoryLayout | null = bunProgram.memoryLayout();
const bunParameterLayout: BunModuleKernelParameterLayout | null = bunProgram.parameterLayout();
const bunCompatibility: BunProgramModuleCompatibility = bunProgram.moduleCompatibility(bunModel);
const bunExecutionPlan: BunProgramExecutionPlan<readonly [2], readonly [1]> = bunProgram.executionPlan(bunModel);
const bunModelBindings: BunModuleBindings<readonly [2], readonly [1]> = bunModel.bindParameters({ inputShape: [2] as const });
const bunModelBindingsAlias: BunModuleBindings<readonly [2], readonly [1]> = bunModel.bind_parameters({
  inputShape: [2] as const,
});
const bunBindingPlan: BunProgramBindingPlan<readonly [2], readonly [1]> = bunProgram.bindingPlan(
  bunModelBindings,
);
const bunBindingPlanAlias: BunProgramBindingPlan<readonly [2], readonly [1]> = bunProgram.binding_plan(
  bunModelBindingsAlias,
);
const bunRequiredBindingPlan: BunProgramBindingPlan<readonly [2], readonly [1]> = bunProgram.requireBindingPlan(
  bunModelBindings,
);
const bunRequiredBindingPlanAlias: BunProgramBindingPlan<readonly [2], readonly [1]> = bunProgram.require_binding_plan(
  bunModelBindingsAlias,
);
const bunRuntimeProfile: BunRuntimeProfile = bunProgram.runtimeProfile();
const bunRuntimeProfileAlias: BunRuntimeProfile = bunProgram.runtime_profile();
const bunRuntimeProfileExpectation: BunRuntimeProfileExpectation = bunProgram.runtimeProfileExpectation();
const bunHotRuntimeProfile: BunRuntimeProfileExpectation = bunProgram.requireHotRuntimeProfile();
const bunNoFallbackRuntimeProfile: boolean = bunProgram.runtimeProfileHasNoFallback();
const bunNoSyncRuntimeProfile: boolean = bunProgram.runtimeProfileHasNoSync();
const bunNoInvalidRuntimePatchProfile: boolean = bunProgram.runtimeProfileHasNoInvalidRuntimePatches();
const bunAccepted: boolean = bunProgram.acceptsModule(bunModel);
const bunSession: BunSession<readonly [2], readonly [1]> = bunProgram.bindModule(bunModel);
const bunCompiled: BunTensor<readonly [1]> = bunSession.stepTensor(bunTensor([1, 2], [2] as const));
const bunCallableCompiled = bunModel.inference({ backend: "cpu", inputShape: [2] as const });
const bunCallableCompiledTensor: BunTensor<readonly [1]> = bunCallableCompiled(bunTensor([1, 2], [2] as const));
const bunCompatibleModule: BunProgramCompatibleModule<readonly [2], readonly [1], typeof bunModel> = bunModel;
const bunIncompatibleModel = bunNn.linear(3, 1);
const bunIncompatibleModuleCompatibility: BunProgramModuleCompatibility = bunProgram.moduleCompatibility(bunIncompatibleModel);
const bunIncompatibleModuleAccepted: boolean = bunProgram.acceptsModule(bunIncompatibleModel);
// @ts-expect-error Bun Program binding preserves the compiled input/output shape contract.
const bunInvalidBoundSession: BunSession<readonly [2], readonly [1]> = bunProgram.bindModule(bunIncompatibleModel);
// @ts-expect-error Bun Program execution plans preserve the compiled input/output shape contract.
const bunInvalidModuleExecutionPlan: BunProgramExecutionPlan<readonly [2], readonly [1]> = bunProgram.executionPlan(bunIncompatibleModel);
// @ts-expect-error Bun ProgramCompatibleModule narrows incompatible modules to never.
const bunInvalidCompatibleModule: BunProgramCompatibleModule<readonly [2], readonly [1], typeof bunIncompatibleModel> = bunIncompatibleModel;
const bunContract: BunSessionStepContract = bunSession.stepContract();
const bunSessionBufferSlotNames: readonly string[] = bunSession.bufferSlotNames();
const bunSessionInputSlot: BunProgramBufferLayoutSlot | null = bunSession.bufferSlot("input");
const bunSessionRuntimeProfile: BunRuntimeProfile = bunSession.runtimeProfile();
const bunSessionRuntimeProfileAlias: BunRuntimeProfile = bunSession.runtime_profile();
const bunSessionRuntimeProfileExpectation: BunRuntimeProfileExpectation = bunSession.runtimeProfileExpectation();
const bunSessionHotRuntimeProfile: BunRuntimeProfileExpectation = bunSession.requireHotRuntimeProfile();
const bunSessionNoFallbackRuntimeProfile: boolean = bunSession.runtimeProfileHasNoFallback();
const bunSessionNoSyncRuntimeProfile: boolean = bunSession.runtimeProfileHasNoSync();
const bunSessionNoInvalidRuntimePatchProfile: boolean = bunSession.runtimeProfileHasNoInvalidRuntimePatches();
const bunHotParams = { input: bunTensor([1, 2], [2] as const), output: false };
const bunHotCompatibility: BunSessionStepParamsCompatibility = bunSession.requireHotStepParams(bunHotParams);
const bunHotCompatibilityAlias: BunSessionStepParamsCompatibility = bunSession.require_hot_step_params(bunHotParams);
const bunHotPlan: BunSessionExecutionPlan<readonly [2], readonly [1]> = bunSession.hotPathPlan(bunHotParams);
const bunHotPlanAlias: BunSessionExecutionPlan<readonly [2], readonly [1]> = bunSession.hot_path_plan(bunHotParams);
const bunRejectedCompatibility: BunSessionStepParamsCompatibility = bunSession.stepParamsCompatibility({
  input: bunTensor([1, 2, 3], [3] as const),
  output: false,
});
const bunRejectedCompatibilityAlias: BunSessionStepParamsCompatibility = bunSession.step_params_compatibility({
  input: bunTensor([1, 2, 3], [3] as const),
  output: false,
});
const bunInputShape: readonly number[] = bunProgram.inputShape();
const bunOutputShape: readonly number[] = bunProgram.outputShape();

void bunEager;
void bunCapabilities;
void bunRequirements;
void bunCompileEvidence;
void bunTrace;
void bunCompilerSignatures;
void bunTensorProgramIr;
void bunKernelPlan;
void bunBufferLayout;
void bunProgramBufferLayout;
void bunProgramBufferSlotNames;
void bunProgramInputSlot;
void bunProgramOutputSlot;
void bunMemoryLayout;
void bunParameterLayout;
void bunCompatibility;
void bunExecutionPlan;
void bunModelBindings;
void bunModelBindingsAlias;
void bunBindingPlan;
void bunBindingPlanAlias;
void bunRequiredBindingPlan;
void bunRequiredBindingPlanAlias;
void bunRuntimeProfile;
void bunRuntimeProfileAlias;
void bunRuntimeProfileExpectation;
void bunHotRuntimeProfile;
void bunNoFallbackRuntimeProfile;
void bunNoSyncRuntimeProfile;
void bunNoInvalidRuntimePatchProfile;
void bunAccepted;
void bunCompiled;
void bunCallableCompiledTensor;
void bunCompatibleModule;
void bunIncompatibleModuleCompatibility;
void bunIncompatibleModuleAccepted;
void bunInvalidBoundSession;
void bunInvalidModuleExecutionPlan;
void bunInvalidCompatibleModule;
void bunContract;
void bunSessionBufferSlotNames;
void bunSessionInputSlot;
void bunSessionRuntimeProfile;
void bunSessionRuntimeProfileAlias;
void bunSessionRuntimeProfileExpectation;
void bunSessionHotRuntimeProfile;
void bunSessionNoFallbackRuntimeProfile;
void bunSessionNoSyncRuntimeProfile;
void bunSessionNoInvalidRuntimePatchProfile;
void bunHotCompatibility;
void bunHotCompatibilityAlias;
void bunHotPlan;
void bunHotPlanAlias;
void bunRejectedCompatibility;
void bunRejectedCompatibilityAlias;
void bunInputShape;
void bunOutputShape;
