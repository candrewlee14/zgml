import {
  requireNoFallbackRuntimeProfile,
  requireNoSyncRuntimeProfile,
  requireHotRuntimeProfile,
  requireRuntimePatchValidProfile,
  runtimeProfileExpectation,
  runtimeProfileHasNoFallback,
  runtimeProfileHasNoInvalidRuntimePatches,
  runtimeProfileHasNoSync,
} from "./runtime_profile_expectation.js";
import {
  genericModelProgramRequirements,
} from "./generic_model_desc.js";
import type {
  ModuleProgramDesc,
} from "./module_program_desc.js";
import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";
import type * as PublicApi from "../public_api.js";

const disposeSymbol: symbol = (Symbol as any).dispose;

type UnknownRecord = Record<string, unknown>;
type SurfaceMethod = (...args: unknown[]) => unknown;
type SurfaceFacade = Readonly<Record<string, unknown>>;
type SurfaceMethodBag = Record<string, SurfaceMethod>;
type GenericTinyDesc = PublicApi.TinyLinearDesc | PublicApi.TinyMlpDesc;
type GenericProgramDesc = GenericTinyDesc | Readonly<ModuleProgramDesc>;
type GenericSessionOutputShape = PublicApi.TensorShape | readonly unknown[] | null;

type SurfaceClass<Instance, Args extends readonly unknown[] = readonly unknown[]> = {
  new (...args: Args): Instance;
  readonly prototype: Instance;
};

type GenericModelConstructorArgs<TDesc> = [handle: unknown, desc: TDesc];
type GenericProgramConstructorArgs = [
  handle: unknown,
  desc: GenericProgramDesc,
  compileEvidence?: unknown,
];
type GenericSessionConstructorArgs = [
  handle: unknown,
  desc: GenericProgramDesc | undefined,
  boundInput?: unknown,
  boundOutput?: unknown,
  ownedBuffers?: readonly unknown[],
  boundOutputShape?: GenericSessionOutputShape,
  bufferLayout?: unknown,
  parameterLayout?: unknown,
];

export type GenericFamilySurfaceOptions = Readonly<{
  createTinyLinearModelHandle(desc: PublicApi.TinyLinearDesc): unknown;
  createTinyMlpModelHandle(desc: PublicApi.TinyMlpDesc): unknown;
  modelFacadePolicy: SurfaceFacade;
  genericProgramFacade: SurfaceFacade;
  genericSessionFacade: SurfaceFacade;
  bindProgram(program: unknown, desc: PublicApi.TinyLinearDesc | PublicApi.TinyMlpDesc, params: unknown): unknown;
  packedProgramWeightsLen(desc: PublicApi.TinyLinearDesc | PublicApi.TinyMlpDesc): number;
  packedProgramBiasLen(desc: PublicApi.TinyLinearDesc | PublicApi.TinyMlpDesc): number;
  programBufferLayoutFromRequirements: unknown;
}>;

export type GenericFamilySurface = Readonly<{
  TinyLinearModel: SurfaceClass<PublicApi.TinyLinearModel, GenericModelConstructorArgs<PublicApi.TinyLinearDesc>> & Pick<typeof PublicApi.TinyLinearModel, "create">;
  TinyMlpModel: SurfaceClass<PublicApi.TinyMlpModel, GenericModelConstructorArgs<PublicApi.TinyMlpDesc>> & Pick<typeof PublicApi.TinyMlpModel, "create">;
  Program: SurfaceClass<PublicApi.Program, GenericProgramConstructorArgs>;
  Session: SurfaceClass<PublicApi.Session, GenericSessionConstructorArgs>;
  TinyLinearProgram: SurfaceClass<PublicApi.Program, GenericProgramConstructorArgs>;
  TinyLinearSession: SurfaceClass<PublicApi.Session, GenericSessionConstructorArgs>;
}>;

export function createGenericFamilySurface(options: GenericFamilySurfaceOptions): GenericFamilySurface {
  const {
    createTinyLinearModelHandle,
    createTinyMlpModelHandle,
    modelFacadePolicy: rawModelFacadePolicy,
    genericProgramFacade: rawGenericProgramFacade,
    genericSessionFacade: rawGenericSessionFacade,
    bindProgram,
    packedProgramWeightsLen,
    packedProgramBiasLen,
    programBufferLayoutFromRequirements: rawProgramBufferLayoutFromRequirements,
  } = options;
  const modelFacadePolicy = rawModelFacadePolicy as SurfaceMethodBag;
  const genericProgramFacade = rawGenericProgramFacade as SurfaceMethodBag;
  const genericSessionFacade = rawGenericSessionFacade as SurfaceMethodBag;
  const programBufferLayoutFromRequirements = rawProgramBufferLayoutFromRequirements as (requirements: UnknownRecord) => PublicApi.ProgramBufferLayout;

  class TinyLinearModel {
    handle: unknown;
    desc: PublicApi.TinyLinearDesc;

    constructor(handle: unknown, desc: PublicApi.TinyLinearDesc) {
      this.handle = handle;
      this.desc = desc;
    }

    static create(desc: PublicApi.TinyLinearDesc) {
      return new TinyLinearModel(createTinyLinearModelHandle(desc), desc);
    }

    inspect() { return modelFacadePolicy.inspect(this); }
    compile(optionsForCompile = {}) { return modelFacadePolicy.compile(this, optionsForCompile, (programHandle: unknown) => new Program(programHandle, this.desc)); }
    free() { return modelFacadePolicy.free(this); }
    dispose() { return modelFacadePolicy.dispose(this); }
    [disposeSymbol]() { return modelFacadePolicy.dispose(this); }
  }

  class TinyMlpModel {
    handle: unknown;
    desc: PublicApi.TinyMlpDesc;

    constructor(handle: unknown, desc: PublicApi.TinyMlpDesc) {
      this.handle = handle;
      this.desc = desc;
    }

    static create(desc: PublicApi.TinyMlpDesc) {
      return new TinyMlpModel(createTinyMlpModelHandle(desc), desc);
    }

    inspect() { return modelFacadePolicy.inspect(this); }
    compile(optionsForCompile = {}) { return modelFacadePolicy.compile(this, optionsForCompile, (programHandle: unknown) => new Program(programHandle, this.desc)); }
    free() { return modelFacadePolicy.free(this); }
    dispose() { return modelFacadePolicy.dispose(this); }
    [disposeSymbol]() { return modelFacadePolicy.dispose(this); }
  }

  class Program {
    handle: unknown;
    desc: GenericProgramDesc;
    compileEvidenceSnapshot: unknown | null;

    constructor(handle: unknown, desc: GenericProgramDesc, compileEvidence = null) {
      this.handle = handle;
      this.desc = desc;
      this.compileEvidenceSnapshot = null;
      genericProgramFacade.withCompileEvidence(this, compileEvidence);
    }

    inspect() { return genericProgramFacade.inspect(this); }
    compileEvidence() { return genericProgramFacade.compileEvidence(this); }
    _withCompileEvidence(evidence: unknown) { return genericProgramFacade.withCompileEvidence(this, evidence); }
    requirements() { return genericProgramFacade.requirements(this); }
    inputLen() { return genericProgramFacade.inputLen(this); }
    outputLen() { return genericProgramFacade.outputLen(this); }
    inputByteLength() { return genericProgramFacade.inputByteLength(this); }
    outputByteLength() { return genericProgramFacade.outputByteLength(this); }
    weightsLen() { return genericProgramFacade.weightsLen(this); }
    weightsByteLength() { return genericProgramFacade.weightsByteLength(this); }
    biasLen() { return genericProgramFacade.biasLen(this); }
    biasByteLength() { return genericProgramFacade.biasByteLength(this); }
    parameterLen() { return genericProgramFacade.parameterLen(this); }
    parameterByteLength() { return genericProgramFacade.parameterByteLength(this); }
    bufferSizing() { return genericProgramFacade.bufferSizing(this); }
    matchesBufferSizingSignature(signature: unknown) { return genericProgramFacade.matchesBufferSizingSignature(this, signature); }
    bufferLayout() { return genericProgramFacade.bufferLayout(this); }
    bufferSlotNames() { return genericProgramFacade.bufferSlotNames(this); }
    bufferSlot(nameOrKind: unknown) { return genericProgramFacade.bufferSlot(this, nameOrKind); }
    inputShape() { return genericProgramFacade.inputShape(this); }
    outputShape() { return genericProgramFacade.outputShape(this); }
    trace() { return genericProgramFacade.trace(this); }
    compilerSignatures() { return genericProgramFacade.compilerSignatures(this); }
    tensorProgramIr() { return genericProgramFacade.tensorProgramIr(this); }
    kernelPlan() { return genericProgramFacade.kernelPlan(this); }
    shapeConstraints() { return genericProgramFacade.shapeConstraints(this); }
    memoryLayout() { return genericProgramFacade.memoryLayout(this); }
    parameterLayout() { return genericProgramFacade.parameterLayout(this); }
    parameterNames() { return genericProgramFacade.parameterNames(this); }
    parameterInfos() { return genericProgramFacade.parameterInfos(this); }
    parameterInfo(nameOrIndex: unknown) { return genericProgramFacade.parameterInfo(this, nameOrIndex); }
    modelCompatibility(model: unknown) { return genericProgramFacade.modelCompatibility(this, model); }
    acceptsModel(model: unknown) { return genericProgramFacade.acceptsModel(this, model); }
    moduleCompatibility(module: unknown, optionsForModule = {}) { return genericProgramFacade.moduleCompatibility(this, module, optionsForModule); }
    acceptsModule(module: unknown, optionsForModule = {}) { return genericProgramFacade.acceptsModule(this, module, optionsForModule); }
    capabilities() { return genericProgramFacade.capabilities(this); }
    executionPlan(module: unknown = null, optionsForModule = {}) { return genericProgramFacade.executionPlan(this, module, optionsForModule); }
    execution_plan(module: unknown = null, optionsForModule = {}) { return genericProgramFacade.executionPlan(this, module, optionsForModule); }
    requireExecutionPlan(module: unknown = null, optionsForModule = {}) { return genericProgramFacade.requireExecutionPlan(this, module, optionsForModule); }
    require_execution_plan(module: unknown = null, optionsForModule = {}) { return genericProgramFacade.requireExecutionPlan(this, module, optionsForModule); }
    bindingPlan(params: unknown = {}) { return genericProgramFacade.bindingPlan(this, params); }
    binding_plan(params: unknown = {}) { return genericProgramFacade.bindingPlan(this, params); }
    requireBindingPlan(params: unknown = {}) { return genericProgramFacade.requireBindingPlan(this, params); }
    require_binding_plan(params: unknown = {}) { return genericProgramFacade.requireBindingPlan(this, params); }
    canExecute() { return genericProgramFacade.canExecute(this); }
    can_execute() { return genericProgramFacade.canExecute(this); }
    canBindExternalResources() { return genericProgramFacade.canBindExternalResources(this); }
    can_bind_external_resources() { return genericProgramFacade.canBindExternalResources(this); }
    hasFullDispatchPlan() { return genericProgramFacade.hasFullDispatchPlan(this); }
    has_full_dispatch_plan() { return genericProgramFacade.hasFullDispatchPlan(this); }
    executionMode() { return genericProgramFacade.executionMode(this); }
    execution_mode() { return genericProgramFacade.executionMode(this); }
    matchesCapabilitySignature(signature: unknown) { return genericProgramFacade.matchesCapabilitySignature(this, signature); }
    matches_capability_signature(signature: unknown) { return genericProgramFacade.matchesCapabilitySignature(this, signature); }
    createBuffer(kind: unknown, optionsForBuffer = {}) { return genericProgramFacade.createBuffer(this, kind, optionsForBuffer); }
    createOutputBuffer(optionsForBuffer = {}) { return genericProgramFacade.createOutputBuffer(this, optionsForBuffer); }
    createWeightsBuffer(optionsForBuffer = {}) { return genericProgramFacade.createWeightsBuffer(this, optionsForBuffer); }
    createBiasBuffer(optionsForBuffer = {}) { return genericProgramFacade.createBiasBuffer(this, optionsForBuffer); }
    createInputBuffer(optionsForBuffer = {}) { return genericProgramFacade.createInputBuffer(this, optionsForBuffer); }
    deviceHandle(placement = "webgpu") { return genericProgramFacade.deviceHandle(this, placement); }
    device(placement = "webgpu") { return genericProgramFacade.device(this, placement); }
    importDeviceBuffer(kind: unknown, optionsForBuffer = {}) { return genericProgramFacade.importDeviceBuffer(this, kind, optionsForBuffer); }
    runtimeProfile() { return genericProgramFacade.runtimeProfile(this); }
    runtime_profile() { return genericProgramFacade.runtimeProfile(this); }
    matchesRuntimeProfileSignature(signature: unknown) { return genericProgramFacade.matchesRuntimeProfileSignature(this, signature); }
    matches_runtime_profile_signature(signature: unknown) { return genericProgramFacade.matchesRuntimeProfileSignature(this, signature); }
    resetRuntimeProfile() { return genericProgramFacade.resetRuntimeProfile(this); }
    reset_runtime_profile() { return genericProgramFacade.resetRuntimeProfile(this); }
    runtimeProfileExpectation() { return runtimeProfileExpectation(this.runtimeProfile()); }
    runtime_profile_expectation() { return runtimeProfileExpectation(this.runtimeProfile()); }
    runtimeProfileHasNoFallback() { return runtimeProfileHasNoFallback(this.runtimeProfile()); }
    runtime_profile_has_no_fallback() { return runtimeProfileHasNoFallback(this.runtimeProfile()); }
    runtimeProfileHasNoSync() { return runtimeProfileHasNoSync(this.runtimeProfile()); }
    runtime_profile_has_no_sync() { return runtimeProfileHasNoSync(this.runtimeProfile()); }
    runtimeProfileHasNoInvalidRuntimePatches() { return runtimeProfileHasNoInvalidRuntimePatches(this.runtimeProfile()); }
    runtime_profile_has_no_invalid_runtime_patches() { return runtimeProfileHasNoInvalidRuntimePatches(this.runtimeProfile()); }
    requireNoFallbackRuntimeProfile() { return requireNoFallbackRuntimeProfile(this.runtimeProfile()); }
    require_no_fallback_runtime_profile() { return requireNoFallbackRuntimeProfile(this.runtimeProfile()); }
    requireNoSyncRuntimeProfile() { return requireNoSyncRuntimeProfile(this.runtimeProfile()); }
    require_no_sync_runtime_profile() { return requireNoSyncRuntimeProfile(this.runtimeProfile()); }
    requireRuntimePatchValidProfile() { return requireRuntimePatchValidProfile(this.runtimeProfile()); }
    require_runtime_patch_valid_profile() { return requireRuntimePatchValidProfile(this.runtimeProfile()); }
    requireHotRuntimeProfile() { return requireHotRuntimeProfile(this.runtimeProfile()); }
    require_hot_runtime_profile() { return requireHotRuntimeProfile(this.runtimeProfile()); }
    bind(params: unknown, optionsForModule = {}) {
      if (params && typeof (params as Record<string, unknown>).placeParameters === "function") {
        return genericProgramFacade.bindModule(this, params, optionsForModule);
      }
      return bindProgram(this, this.desc as GenericTinyDesc, params);
    }
    bindModule(module: unknown, optionsForModule = {}) { return genericProgramFacade.bindModule(this, module, optionsForModule); }
    free() { return genericProgramFacade.free(this); }
    dispose() { return genericProgramFacade.dispose(this); }
    [disposeSymbol]() { return genericProgramFacade.dispose(this); }
  }

  class Session {
    handle: unknown;
    desc: GenericProgramDesc | undefined;
    boundInput: unknown;
    boundOutput: unknown;
    ownedBuffers: unknown[];
    boundOutputShape: unknown;
    boundParameterLayout: unknown;
    boundBufferLayout: unknown;

    constructor(
      handle: unknown,
      desc: GenericProgramDesc | undefined,
      boundInput = null,
      boundOutput = null,
      ownedBuffers: readonly unknown[] = [],
      boundOutputShape: GenericSessionOutputShape = null,
      bufferLayout: unknown = null,
      parameterLayout: unknown = null,
    ) {
      this.handle = handle;
      this.desc = desc;
      this.boundInput = boundInput;
      this.boundOutput = boundOutput;
      this.ownedBuffers = ownedBuffers.slice();
      this.boundOutputShape = Array.isArray(boundOutputShape) ? boundOutputShape.slice() : boundOutputShape;
      this.boundParameterLayout = parameterLayout || null;
      this.boundBufferLayout = bufferLayout || programBufferLayoutFromRequirements(genericModelProgramRequirements(desc as GenericTinyDesc, {
        packedProgramWeightsLen,
        packedProgramBiasLen,
      }));
    }

    inspect() { return genericSessionFacade.inspect(this); }
    bufferLayout() { return genericSessionFacade.bufferLayout(this); }
    bufferSlotNames() { return genericSessionFacade.bufferSlotNames(this); }
    bufferSlot(nameOrKind: unknown) { return genericSessionFacade.bufferSlot(this, nameOrKind); }
    inputLen() { return genericSessionFacade.inputLen(this); }
    outputLen() { return genericSessionFacade.outputLen(this); }
    inputByteLength() { return genericSessionFacade.inputByteLength(this); }
    outputByteLength() { return genericSessionFacade.outputByteLength(this); }
    weightsLen() { return genericSessionFacade.weightsLen(this); }
    weightsByteLength() { return genericSessionFacade.weightsByteLength(this); }
    biasLen() { return genericSessionFacade.biasLen(this); }
    biasByteLength() { return genericSessionFacade.biasByteLength(this); }
    parameterLen() { return genericSessionFacade.parameterLen(this); }
    parameterByteLength() { return genericSessionFacade.parameterByteLength(this); }
    bufferSizing() { return genericSessionFacade.bufferSizing(this); }
    matchesBufferSizingSignature(signature: unknown) { return genericSessionFacade.matchesBufferSizingSignature(this, signature); }
    inputShape() { return genericSessionFacade.inputShape(this); }
    outputShape() { return genericSessionFacade.outputShape(this); }
    sessionCallProfile() { return genericSessionFacade.sessionCallProfile(this); }
    session_call_profile() { return genericSessionFacade.sessionCallProfile(this); }
    matchesSessionCallProfileSignature(signature: unknown) { return genericSessionFacade.matchesSessionCallProfileSignature(this, signature); }
    matches_session_call_profile_signature(signature: unknown) { return genericSessionFacade.matchesSessionCallProfileSignature(this, signature); }
    resetSessionCallProfile() { return genericSessionFacade.resetSessionCallProfile(this); }
    reset_session_call_profile() { return genericSessionFacade.resetSessionCallProfile(this); }
    stepContract() { return genericSessionFacade.stepContract(this); }
    step_contract() { return genericSessionFacade.stepContract(this); }
    stepParamsCompatibility(params = {}) { return genericSessionFacade.stepParamsCompatibility(this, params); }
    step_params_compatibility(params = {}) { return genericSessionFacade.stepParamsCompatibility(this, params); }
    preflightStepParams(params = {}) { return genericSessionFacade.preflightStepParams(this, params); }
    preflight_step_params(params = {}) { return genericSessionFacade.preflightStepParams(this, params); }
    acceptsStepParams(params = {}) { return genericSessionFacade.acceptsStepParams(this, params); }
    accepts_step_params(params = {}) { return genericSessionFacade.acceptsStepParams(this, params); }
    canExecuteStepParams(params = {}) { return genericSessionFacade.canExecuteStepParams(this, params); }
    can_execute_step_params(params = {}) { return genericSessionFacade.canExecuteStepParams(this, params); }
    requireCanExecuteStepParams(params = {}) { return genericSessionFacade.requireCanExecuteStepParams(this, params); }
    require_can_execute_step_params(params = {}) { return genericSessionFacade.requireCanExecuteStepParams(this, params); }
    acceptsAllocationFreeStepParams(params = {}) { return genericSessionFacade.acceptsAllocationFreeStepParams(this, params); }
    accepts_allocation_free_step_params(params = {}) { return genericSessionFacade.acceptsAllocationFreeStepParams(this, params); }
    requireAllocationFreeStepParams(params = {}) { return genericSessionFacade.requireAllocationFreeStepParams(this, params); }
    require_allocation_free_step_params(params = {}) { return genericSessionFacade.requireAllocationFreeStepParams(this, params); }
    acceptsRuntimeOutputAllocationFreeStepParams(params = {}) { return genericSessionFacade.acceptsRuntimeOutputAllocationFreeStepParams(this, params); }
    accepts_runtime_output_allocation_free_step_params(params = {}) { return genericSessionFacade.acceptsRuntimeOutputAllocationFreeStepParams(this, params); }
    requireRuntimeOutputAllocationFreeStepParams(params = {}) { return genericSessionFacade.requireRuntimeOutputAllocationFreeStepParams(this, params); }
    require_runtime_output_allocation_free_step_params(params = {}) { return genericSessionFacade.requireRuntimeOutputAllocationFreeStepParams(this, params); }
    acceptsNoReadbackStepParams(params = {}) { return genericSessionFacade.acceptsNoReadbackStepParams(this, params); }
    accepts_no_readback_step_params(params = {}) { return genericSessionFacade.acceptsNoReadbackStepParams(this, params); }
    requireNoReadbackStepParams(params = {}) { return genericSessionFacade.requireNoReadbackStepParams(this, params); }
    require_no_readback_step_params(params = {}) { return genericSessionFacade.requireNoReadbackStepParams(this, params); }
    acceptsReadbackFreeStepParams(params = {}) { return genericSessionFacade.acceptsReadbackFreeStepParams(this, params); }
    accepts_readback_free_step_params(params = {}) { return genericSessionFacade.acceptsReadbackFreeStepParams(this, params); }
    requireReadbackFreeStepParams(params = {}) { return genericSessionFacade.requireReadbackFreeStepParams(this, params); }
    require_readback_free_step_params(params = {}) { return genericSessionFacade.requireReadbackFreeStepParams(this, params); }
    acceptsHotStepParams(params = {}) { return genericSessionFacade.acceptsHotStepParams(this, params); }
    accepts_hot_step_params(params = {}) { return genericSessionFacade.acceptsHotStepParams(this, params); }
    requireHotStepParams(params = {}) { return genericSessionFacade.requireHotStepParams(this, params); }
    require_hot_step_params(params = {}) { return genericSessionFacade.requireHotStepParams(this, params); }
    executionPlan(params = {}) { return genericSessionFacade.executionPlan(this, params); }
    execution_plan(params = {}) { return genericSessionFacade.executionPlan(this, params); }
    requireExecutionPlan(params = {}) { return genericSessionFacade.requireExecutionPlan(this, params); }
    require_execution_plan(params = {}) { return genericSessionFacade.requireExecutionPlan(this, params); }
    hotPathPlan(params = {}) { return genericSessionFacade.hotPathPlan(this, params); }
    hot_path_plan(params = {}) { return genericSessionFacade.hotPathPlan(this, params); }
    matchesStepContractSignature(signature: unknown) { return genericSessionFacade.matchesStepContractSignature(this, signature); }
    matches_step_contract_signature(signature: unknown) { return genericSessionFacade.matchesStepContractSignature(this, signature); }
    matchesStepParamsSignature(params = {}, signature: unknown) { return genericSessionFacade.matchesStepParamsSignature(this, params, signature); }
    matches_step_params_signature(params = {}, signature: unknown) { return genericSessionFacade.matchesStepParamsSignature(this, params, signature); }
    matchesStepParamsCompatibility(params = {}, compatibility: unknown) { return genericSessionFacade.matchesStepParamsCompatibility(this, params, compatibility); }
    matches_step_params_compatibility(params = {}, compatibility: unknown) { return genericSessionFacade.matchesStepParamsCompatibility(this, params, compatibility); }
    uploadParameters() { return genericSessionFacade.uploadParameters(this); }
    uploadParameter(index: unknown) { return genericSessionFacade.uploadParameter(this, index); }
    uploadParameterByName(name: unknown) { return genericSessionFacade.uploadParameterByName(this, name); }
    uploadParameterRange(first: unknown, len: unknown) { return genericSessionFacade.uploadParameterRange(this, first, len); }
    parameterLayout() { return genericSessionFacade.parameterLayout(this); }
    parameterNames() { return genericSessionFacade.parameterNames(this); }
    parameterInfos() { return genericSessionFacade.parameterInfos(this); }
    parameterInfo(nameOrIndex: unknown) { return genericSessionFacade.parameterInfo(this, nameOrIndex); }
    step(inputValues: unknown, outputValues?: unknown) { return genericSessionFacade.step(this, inputValues, outputValues); }
    stepTensor(inputValues: unknown, optionsForStep = {}) { return genericSessionFacade.stepTensor(this, inputValues, optionsForStep); }
    stepInto(outputValues: unknown, inputValues: unknown) { return genericSessionFacade.stepInto(this, outputValues, inputValues); }
    execute(params = {}) { return genericSessionFacade.execute(this, params); }
    executeTensor(params = {}) { return genericSessionFacade.executeTensor(this, params); }
    executeInto(outputValues: unknown, params = {}) { return genericSessionFacade.executeInto(this, outputValues, params); }
    readOutputInto(outputValues: unknown, length = this.desc?.outputLen ?? 0, byteOffset = 0) { return genericSessionFacade.readOutputInto(this, outputValues, length, byteOffset); }
    readOutputTensor(optionsForRead = {}) { return genericSessionFacade.readOutputTensor(this, optionsForRead); }
    advance(inputValues: unknown) { return genericSessionFacade.advance(this, inputValues); }
    reset() { return genericSessionFacade.reset(this); }
    runtimeProfile() { return genericSessionFacade.runtimeProfile(this); }
    runtime_profile() { return genericSessionFacade.runtimeProfile(this); }
    matchesRuntimeProfileSignature(signature: unknown) { return genericSessionFacade.matchesRuntimeProfileSignature(this, signature); }
    matches_runtime_profile_signature(signature: unknown) { return genericSessionFacade.matchesRuntimeProfileSignature(this, signature); }
    resetRuntimeProfile() { return genericSessionFacade.resetRuntimeProfile(this); }
    reset_runtime_profile() { return genericSessionFacade.resetRuntimeProfile(this); }
    runtimeProfileExpectation() { return runtimeProfileExpectation(this.runtimeProfile()); }
    runtime_profile_expectation() { return runtimeProfileExpectation(this.runtimeProfile()); }
    runtimeProfileHasNoFallback() { return runtimeProfileHasNoFallback(this.runtimeProfile()); }
    runtime_profile_has_no_fallback() { return runtimeProfileHasNoFallback(this.runtimeProfile()); }
    runtimeProfileHasNoSync() { return runtimeProfileHasNoSync(this.runtimeProfile()); }
    runtime_profile_has_no_sync() { return runtimeProfileHasNoSync(this.runtimeProfile()); }
    runtimeProfileHasNoInvalidRuntimePatches() { return runtimeProfileHasNoInvalidRuntimePatches(this.runtimeProfile()); }
    runtime_profile_has_no_invalid_runtime_patches() { return runtimeProfileHasNoInvalidRuntimePatches(this.runtimeProfile()); }
    requireNoFallbackRuntimeProfile() { return requireNoFallbackRuntimeProfile(this.runtimeProfile()); }
    require_no_fallback_runtime_profile() { return requireNoFallbackRuntimeProfile(this.runtimeProfile()); }
    requireNoSyncRuntimeProfile() { return requireNoSyncRuntimeProfile(this.runtimeProfile()); }
    require_no_sync_runtime_profile() { return requireNoSyncRuntimeProfile(this.runtimeProfile()); }
    requireRuntimePatchValidProfile() { return requireRuntimePatchValidProfile(this.runtimeProfile()); }
    require_runtime_patch_valid_profile() { return requireRuntimePatchValidProfile(this.runtimeProfile()); }
    requireHotRuntimeProfile() { return requireHotRuntimeProfile(this.runtimeProfile()); }
    require_hot_runtime_profile() { return requireHotRuntimeProfile(this.runtimeProfile()); }
    _ownBuffers(buffers: readonly unknown[]) { this.ownedBuffers.push(...buffers); return this; }
    free() { return genericSessionFacade.free(this); }
    dispose() { return genericSessionFacade.dispose(this); }
    [disposeSymbol]() { return genericSessionFacade.dispose(this); }
  }

  return Object.freeze({
    TinyLinearModel,
    TinyMlpModel,
    Program,
    Session,
    TinyLinearProgram: Program,
    TinyLinearSession: Session,
  }) as unknown as GenericFamilySurface;
}

export const genericFamilySurfaceManifest = Object.freeze({
  kind: "zgml-generic-family-surface",
  ...tsRuntimeManifestPolicy("src/ts/runtime/generic_family_surface.ts", "Generic model handles -> Program -> Session surface"),
});
