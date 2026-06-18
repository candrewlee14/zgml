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
import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";
import type * as PublicApi from "../public_api.js";

const disposeSymbol: symbol = (Symbol as any).dispose;

type LlamaSurfaceCallback = (...args: any[]) => any;
type LlamaModelFamilyFacade = Readonly<Record<
  "createTinyLlama" |
  "load" |
  "loadSafetensorsData" |
  "probe" |
  "probeHeader" |
  "inspect" |
  "compile" |
  "free" |
  "dispose",
  LlamaSurfaceCallback
>>;
type LlamaProgramFacade = Readonly<Record<string, LlamaSurfaceCallback>>;
type LlamaSessionScratch = Record<string, unknown>;

export type LlamaFamilySurfaceOptions = Readonly<{
  autoKind: unknown;
  tinyLlamaKind: unknown;
  smollm135mKind: unknown;
  llamaModelFamilyFacade: LlamaModelFamilyFacade;
  llamaProgramFacade: LlamaProgramFacade;
  bindLlamaSessionHandle: LlamaSurfaceCallback;
  defaultLlamaSessionBufferLayout: (vocabSize: number, bufferLayout: unknown) => unknown;
  createLlamaSessionScratch: () => LlamaSessionScratch;
  llamaSessionPositionFacade: LlamaSurfaceCallback;
  inspectLlamaSessionFacade: LlamaSurfaceCallback;
  assertSessionAlive: (handle: unknown) => void;
  llamaSessionBufferLayout: LlamaSurfaceCallback;
  llamaSessionBufferSlotNames: LlamaSurfaceCallback;
  llamaSessionBufferSlot: LlamaSurfaceCallback;
  llamaSessionKvCacheLayout: LlamaSurfaceCallback;
  llamaSessionInputLen: LlamaSurfaceCallback;
  llamaSessionOutputLen: LlamaSurfaceCallback;
  llamaSessionInputByteLength: LlamaSurfaceCallback;
  llamaSessionOutputByteLength: LlamaSurfaceCallback;
  llamaSessionWeightsLen: LlamaSurfaceCallback;
  llamaSessionWeightsByteLength: LlamaSurfaceCallback;
  llamaSessionBiasLen: LlamaSurfaceCallback;
  llamaSessionBiasByteLength: LlamaSurfaceCallback;
  llamaSessionParameterLen: LlamaSurfaceCallback;
  llamaSessionParameterByteLength: LlamaSurfaceCallback;
  llamaSessionBufferSizing: LlamaSurfaceCallback;
  llamaSessionMatchesBufferSizingSignature: LlamaSurfaceCallback;
  llamaSessionOutputShape: LlamaSurfaceCallback;
  llamaSessionStepContract: LlamaSurfaceCallback;
  llamaSessionStepParamsCompatibility: LlamaSurfaceCallback;
  llamaSessionPreflightStepParams: LlamaSurfaceCallback;
  llamaSessionAcceptsStepParams: LlamaSurfaceCallback;
  llamaSessionCanExecuteStepParams: LlamaSurfaceCallback;
  llamaSessionRequireCanExecuteStepParams: LlamaSurfaceCallback;
  llamaSessionAcceptsAllocationFreeStepParams: LlamaSurfaceCallback;
  llamaSessionRequireAllocationFreeStepParams: LlamaSurfaceCallback;
  llamaSessionAcceptsRuntimeOutputAllocationFreeStepParams: LlamaSurfaceCallback;
  llamaSessionRequireRuntimeOutputAllocationFreeStepParams: LlamaSurfaceCallback;
  llamaSessionAcceptsNoReadbackStepParams: LlamaSurfaceCallback;
  llamaSessionRequireNoReadbackStepParams: LlamaSurfaceCallback;
  llamaSessionAcceptsReadbackFreeStepParams: LlamaSurfaceCallback;
  llamaSessionRequireReadbackFreeStepParams: LlamaSurfaceCallback;
  llamaSessionAcceptsHotStepParams: LlamaSurfaceCallback;
  llamaSessionRequireHotStepParams: LlamaSurfaceCallback;
  llamaSessionExecutionPlan: LlamaSurfaceCallback;
  llamaSessionRequireExecutionPlan: LlamaSurfaceCallback;
  llamaSessionHotPathPlan: LlamaSurfaceCallback;
  llamaSessionMatchesStepContractSignature: LlamaSurfaceCallback;
  llamaSessionMatchesStepParamsSignature: LlamaSurfaceCallback;
  llamaSessionMatchesStepParamsCompatibility: LlamaSurfaceCallback;
  stepLlama: LlamaSurfaceCallback;
  stepLlamaTensor: LlamaSurfaceCallback;
  stepLlamaInto: LlamaSurfaceCallback;
  llamaScalarTokenWindow: LlamaSurfaceCallback;
  llamaScalarTokenSampleOptions: LlamaSurfaceCallback;
  llamaNoOutputTokenWindowOptions: LlamaSurfaceCallback;
  llamaOutputTokenWindowOptions: LlamaSurfaceCallback;
  llamaScalarTokenExecuteOptions: LlamaSurfaceCallback;
  llamaScalarTokenOutputOptions: LlamaSurfaceCallback;
  advanceLlama: LlamaSurfaceCallback;
  advanceLlamaTokens: LlamaSurfaceCallback;
  executeLlamaTokensArgmax: LlamaSurfaceCallback;
  stepLlamaArgmax: LlamaSurfaceCallback;
  executeLlamaTokensSample: LlamaSurfaceCallback;
  stepLlamaSample: LlamaSurfaceCallback;
  generateLlamaTokensArgmax: LlamaSurfaceCallback;
  generateLlamaTokensArgmaxInto: LlamaSurfaceCallback;
  generateLlamaTokensSample: LlamaSurfaceCallback;
  generateLlamaTokensSampleInto: LlamaSurfaceCallback;
  executeLlamaTokens: LlamaSurfaceCallback;
  executeLlama: LlamaSurfaceCallback;
  executeLlamaTensor: LlamaSurfaceCallback;
  executeLlamaInto: LlamaSurfaceCallback;
  argmaxLlamaToken: LlamaSurfaceCallback;
  sampleLlamaToken: LlamaSurfaceCallback;
  prefillLlama: LlamaSurfaceCallback;
  prefillLlamaTensor: LlamaSurfaceCallback;
  prefillLlamaInto: LlamaSurfaceCallback;
  readLlamaOutputInto: LlamaSurfaceCallback;
  readLlamaOutputTensor: LlamaSurfaceCallback;
  llamaSessionCallProfileFacade: LlamaSurfaceCallback;
  llamaSessionMatchesCallProfileSignature: LlamaSurfaceCallback;
  resetLlamaSessionCallProfileFacade: LlamaSurfaceCallback;
  resetLlamaSessionFacade: LlamaSurfaceCallback;
  llamaSessionRuntimeProfileFacade: LlamaSurfaceCallback;
  llamaSessionMatchesRuntimeProfileSignature: LlamaSurfaceCallback;
  resetLlamaSessionRuntimeProfileFacade: LlamaSurfaceCallback;
  freeLlamaSessionFacade: LlamaSurfaceCallback;
  disposeLlamaSessionFacade: LlamaSurfaceCallback;
}>;

type SurfaceClass<Instance, Args extends readonly unknown[] = readonly unknown[]> = {
  new (...args: Args): Instance;
  readonly prototype: Instance;
};

type LlamaModelConstructorArgs = [handle: unknown];
type LlamaProgramConstructorArgs = [handle: unknown];
type LlamaSessionConstructorArgs = [
  handle: unknown,
  vocabSize: number,
  boundOutput?: unknown,
  ownedOutput?: unknown,
  bufferLayout?: unknown,
  kvCacheLayout?: unknown,
];

export type LlamaFamilySurface = Readonly<{
  TinyLlamaModel: SurfaceClass<PublicApi.TinyLlamaModel, LlamaModelConstructorArgs> & Pick<typeof PublicApi.TinyLlamaModel, "create" | "load" | "loadSafetensorsData">;
  TinyLlamaProgram: SurfaceClass<PublicApi.TinyLlamaProgram, LlamaProgramConstructorArgs>;
  TinyLlamaSession: SurfaceClass<PublicApi.TinyLlamaSession, LlamaSessionConstructorArgs>;
  LlamaModel: SurfaceClass<PublicApi.LlamaModel, LlamaModelConstructorArgs> & Pick<typeof PublicApi.LlamaModel, "load" | "loadSafetensorsData">;
  LlamaProgram: SurfaceClass<PublicApi.LlamaProgram, LlamaProgramConstructorArgs>;
  LlamaSession: SurfaceClass<PublicApi.LlamaSession, LlamaSessionConstructorArgs>;
  SmolLM135MModel: SurfaceClass<PublicApi.SmolLM135MModel, LlamaModelConstructorArgs> & Pick<typeof PublicApi.SmolLM135MModel, "load" | "loadSafetensorsData">;
  SmolLM135MProgram: SurfaceClass<PublicApi.SmolLM135MProgram, LlamaProgramConstructorArgs>;
  SmolLM135MSession: SurfaceClass<PublicApi.SmolLM135MSession, LlamaSessionConstructorArgs>;
}>;

export function createLlamaFamilySurface(options: LlamaFamilySurfaceOptions): LlamaFamilySurface {
  const {
    autoKind,
    tinyLlamaKind,
    smollm135mKind,
    llamaModelFamilyFacade,
    llamaProgramFacade,
    bindLlamaSessionHandle,
    defaultLlamaSessionBufferLayout,
    createLlamaSessionScratch,
    llamaSessionPositionFacade,
    inspectLlamaSessionFacade,
    assertSessionAlive,
    llamaSessionBufferLayout,
    llamaSessionBufferSlotNames,
    llamaSessionBufferSlot,
    llamaSessionKvCacheLayout,
    llamaSessionInputLen,
    llamaSessionOutputLen,
    llamaSessionInputByteLength,
    llamaSessionOutputByteLength,
    llamaSessionWeightsLen,
    llamaSessionWeightsByteLength,
    llamaSessionBiasLen,
    llamaSessionBiasByteLength,
    llamaSessionParameterLen,
    llamaSessionParameterByteLength,
    llamaSessionBufferSizing,
    llamaSessionMatchesBufferSizingSignature,
    llamaSessionOutputShape,
    llamaSessionStepContract,
    llamaSessionStepParamsCompatibility,
    llamaSessionPreflightStepParams,
    llamaSessionAcceptsStepParams,
    llamaSessionCanExecuteStepParams,
    llamaSessionRequireCanExecuteStepParams,
    llamaSessionAcceptsAllocationFreeStepParams,
    llamaSessionRequireAllocationFreeStepParams,
    llamaSessionAcceptsRuntimeOutputAllocationFreeStepParams,
    llamaSessionRequireRuntimeOutputAllocationFreeStepParams,
    llamaSessionAcceptsNoReadbackStepParams,
    llamaSessionRequireNoReadbackStepParams,
    llamaSessionAcceptsReadbackFreeStepParams,
    llamaSessionRequireReadbackFreeStepParams,
    llamaSessionAcceptsHotStepParams,
    llamaSessionRequireHotStepParams,
    llamaSessionExecutionPlan,
    llamaSessionRequireExecutionPlan,
    llamaSessionHotPathPlan,
    llamaSessionMatchesStepContractSignature,
    llamaSessionMatchesStepParamsSignature,
    llamaSessionMatchesStepParamsCompatibility,
    stepLlama,
    stepLlamaTensor,
    stepLlamaInto,
    llamaScalarTokenWindow,
    llamaScalarTokenSampleOptions,
    llamaNoOutputTokenWindowOptions,
    llamaOutputTokenWindowOptions,
    llamaScalarTokenExecuteOptions,
    llamaScalarTokenOutputOptions,
    advanceLlama,
    advanceLlamaTokens,
    executeLlamaTokensArgmax,
    stepLlamaArgmax,
    executeLlamaTokensSample,
    stepLlamaSample,
    generateLlamaTokensArgmax,
    generateLlamaTokensArgmaxInto,
    generateLlamaTokensSample,
    generateLlamaTokensSampleInto,
    executeLlamaTokens,
    executeLlama,
    executeLlamaTensor,
    executeLlamaInto,
    argmaxLlamaToken,
    sampleLlamaToken,
    prefillLlama,
    prefillLlamaTensor,
    prefillLlamaInto,
    readLlamaOutputInto,
    readLlamaOutputTensor,
    llamaSessionCallProfileFacade,
    llamaSessionMatchesCallProfileSignature,
    resetLlamaSessionCallProfileFacade,
    resetLlamaSessionFacade,
    llamaSessionRuntimeProfileFacade,
    llamaSessionMatchesRuntimeProfileSignature,
    resetLlamaSessionRuntimeProfileFacade,
    freeLlamaSessionFacade,
    disposeLlamaSessionFacade,
  } = options;

  class TinyLlamaModel {
    handle: unknown;

    constructor(handle: unknown) {
      this.handle = handle;
    }

    static create() {
      return llamaModelFamilyFacade.createTinyLlama((handle: unknown) => new TinyLlamaModel(handle));
    }

    static load(modelPath: unknown) {
      return llamaModelFamilyFacade.load(tinyLlamaKind, modelPath, (handle: unknown) => new TinyLlamaModel(handle));
    }

    static loadSafetensorsData(data: unknown) {
      return llamaModelFamilyFacade.loadSafetensorsData(tinyLlamaKind, data, (handle: unknown) => new TinyLlamaModel(handle));
    }

    static probe(source: unknown) {
      return llamaModelFamilyFacade.probe("tiny-llama", source);
    }

    static probeSafetensorsHeader(header: unknown) {
      return llamaModelFamilyFacade.probeHeader("tiny-llama", header);
    }

    inspect() {
      return llamaModelFamilyFacade.inspect(this);
    }

    compile(optionsForCompile = {}) {
      return llamaModelFamilyFacade.compile(this, optionsForCompile, (programHandle: unknown) => new TinyLlamaProgram(programHandle));
    }

    free() {
      return llamaModelFamilyFacade.free(this);
    }

    dispose() {
      return llamaModelFamilyFacade.dispose(this);
    }

    [disposeSymbol]() {
      return llamaModelFamilyFacade.dispose(this);
    }
  }

  class TinyLlamaProgram {
    handle: unknown;
    inspection: unknown | null;

    constructor(handle: unknown) {
      this.handle = handle;
      this.inspection = null;
    }

    inspect() { return llamaProgramFacade.inspect(this); }
    get vocabSize() { return llamaProgramFacade.vocabSize(this); }
    requirements() { return llamaProgramFacade.requirements(this); }
    inputLen() { return llamaProgramFacade.inputLen(this); }
    outputLen() { return llamaProgramFacade.outputLen(this); }
    inputByteLength() { return llamaProgramFacade.inputByteLength(this); }
    outputByteLength() { return llamaProgramFacade.outputByteLength(this); }
    weightsLen() { return llamaProgramFacade.weightsLen(this); }
    weightsByteLength() { return llamaProgramFacade.weightsByteLength(this); }
    biasLen() { return llamaProgramFacade.biasLen(this); }
    biasByteLength() { return llamaProgramFacade.biasByteLength(this); }
    parameterLen() { return llamaProgramFacade.parameterLen(this); }
    parameterByteLength() { return llamaProgramFacade.parameterByteLength(this); }
    bufferSizing() { return llamaProgramFacade.bufferSizing(this); }
    matchesBufferSizingSignature(signature: unknown) { return llamaProgramFacade.matchesBufferSizingSignature(this, signature); }
    bufferLayout() { return llamaProgramFacade.bufferLayout(this); }
    bufferSlotNames() { return llamaProgramFacade.bufferSlotNames(this); }
    bufferSlot(nameOrKind: unknown) { return llamaProgramFacade.bufferSlot(this, nameOrKind); }
    outputShape() { return llamaProgramFacade.outputShape(this); }
    modelCompatibility(model: unknown) { return llamaProgramFacade.modelCompatibility(this, model); }
    acceptsModel(model: unknown) { return llamaProgramFacade.acceptsModel(this, model); }
    capabilities() { return llamaProgramFacade.capabilities(this); }
    executionPlan() { return llamaProgramFacade.executionPlan(this); }
    execution_plan() { return llamaProgramFacade.executionPlan(this); }
    requireExecutionPlan() { return llamaProgramFacade.requireExecutionPlan(this); }
    require_execution_plan() { return llamaProgramFacade.requireExecutionPlan(this); }
    canExecute() { return llamaProgramFacade.canExecute(this); }
    can_execute() { return llamaProgramFacade.canExecute(this); }
    canBindExternalResources() { return llamaProgramFacade.canBindExternalResources(this); }
    can_bind_external_resources() { return llamaProgramFacade.canBindExternalResources(this); }
    hasFullDispatchPlan() { return llamaProgramFacade.hasFullDispatchPlan(this); }
    has_full_dispatch_plan() { return llamaProgramFacade.hasFullDispatchPlan(this); }
    executionMode() { return llamaProgramFacade.executionMode(this); }
    execution_mode() { return llamaProgramFacade.executionMode(this); }
    matchesCapabilitySignature(signature: unknown) { return llamaProgramFacade.matchesCapabilitySignature(this, signature); }
    matches_capability_signature(signature: unknown) { return llamaProgramFacade.matchesCapabilitySignature(this, signature); }
    createBuffer(kind: unknown, optionsForBuffer = {}) { return llamaProgramFacade.createBuffer(this, kind, optionsForBuffer); }
    createOutputBuffer(optionsForBuffer = {}) { return llamaProgramFacade.createOutputBuffer(this, optionsForBuffer); }
    deviceHandle(placement = "webgpu") { return llamaProgramFacade.deviceHandle(this, placement); }
    device(placement = "webgpu") { return llamaProgramFacade.device(this, placement); }
    importDeviceBuffer(kind: unknown, optionsForBuffer = {}) { return llamaProgramFacade.importDeviceBuffer(this, kind, optionsForBuffer); }
    kvCacheRequirements() { return llamaProgramFacade.kvCacheRequirements(this); }
    kvCacheLayout() { return llamaProgramFacade.kvCacheLayout(this); }
    createKvCache(optionsForCache = {}) { return llamaProgramFacade.createKvCache(this, optionsForCache); }
    inspectExecutable() { return llamaProgramFacade.inspectExecutable(this); }
    runtimeProfile() { return llamaProgramFacade.runtimeProfile(this); }
    runtime_profile() { return llamaProgramFacade.runtimeProfile(this); }
    matchesRuntimeProfileSignature(signature: unknown) { return llamaProgramFacade.matchesRuntimeProfileSignature(this, signature); }
    matches_runtime_profile_signature(signature: unknown) { return llamaProgramFacade.matchesRuntimeProfileSignature(this, signature); }
    resetRuntimeProfile() { return llamaProgramFacade.resetRuntimeProfile(this); }
    reset_runtime_profile() { return llamaProgramFacade.resetRuntimeProfile(this); }
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

    bind(optionsForBind = {}) {
      return llamaProgramFacade.bind(this, optionsForBind, bindLlamaSessionHandle, (
        sessionHandle: unknown,
        inspection: { readonly vocabSize: number },
        boundOutput: unknown,
        ownedOutput: unknown,
        layout: unknown,
        kvLayout: unknown,
      ) => new TinyLlamaSession(sessionHandle, inspection.vocabSize, boundOutput, ownedOutput, layout, kvLayout));
    }

    free() { return llamaProgramFacade.free(this); }
    dispose() { return llamaProgramFacade.dispose(this); }
    [disposeSymbol]() { return llamaProgramFacade.dispose(this); }
  }

  class TinyLlamaSession {
    handle: unknown;
    vocabSize: number;
    boundOutput: unknown;
    ownedOutput: unknown;
    boundBufferLayout: unknown;
    boundKvCacheLayout: unknown;

    constructor(handle: unknown, vocabSize: number, boundOutput: unknown = null, ownedOutput: unknown = null, bufferLayout: unknown = null, kvCacheLayout: unknown = null) {
      this.handle = handle;
      this.vocabSize = vocabSize;
      this.boundOutput = boundOutput;
      this.ownedOutput = ownedOutput;
      this.boundBufferLayout = defaultLlamaSessionBufferLayout(vocabSize, bufferLayout);
      this.boundKvCacheLayout = kvCacheLayout;
      Object.assign(this, createLlamaSessionScratch());
    }

    position() { return llamaSessionPositionFacade(this); }
    inspect() { return inspectLlamaSessionFacade(this); }
    bufferLayout() { assertSessionAlive(this.handle); return llamaSessionBufferLayout(this); }
    bufferSlotNames() { assertSessionAlive(this.handle); return llamaSessionBufferSlotNames(this); }
    bufferSlot(nameOrKind: unknown) { assertSessionAlive(this.handle); return llamaSessionBufferSlot(this, nameOrKind); }
    kvCacheLayout() { assertSessionAlive(this.handle); return llamaSessionKvCacheLayout(this); }
    inputLen() { assertSessionAlive(this.handle); return llamaSessionInputLen(this); }
    outputLen() { assertSessionAlive(this.handle); return llamaSessionOutputLen(this); }
    inputByteLength() { return llamaSessionInputByteLength(this); }
    outputByteLength() { return llamaSessionOutputByteLength(this); }
    weightsLen() { return llamaSessionWeightsLen(this); }
    weightsByteLength() { return llamaSessionWeightsByteLength(this); }
    biasLen() { return llamaSessionBiasLen(this); }
    biasByteLength() { return llamaSessionBiasByteLength(this); }
    parameterLen() { return llamaSessionParameterLen(this); }
    parameterByteLength() { return llamaSessionParameterByteLength(this); }
    bufferSizing() { return llamaSessionBufferSizing(this); }
    matchesBufferSizingSignature(signature: unknown) { return llamaSessionMatchesBufferSizingSignature(this, signature); }
    outputShape() { assertSessionAlive(this.handle); return llamaSessionOutputShape(this); }
    stepContract() { assertSessionAlive(this.handle); return llamaSessionStepContract(this); }
    step_contract() { assertSessionAlive(this.handle); return llamaSessionStepContract(this); }
    stepParamsCompatibility(params: unknown) { assertSessionAlive(this.handle); return llamaSessionStepParamsCompatibility(this, params); }
    step_params_compatibility(params: unknown) { assertSessionAlive(this.handle); return llamaSessionStepParamsCompatibility(this, params); }
    preflightStepParams(params: unknown) { assertSessionAlive(this.handle); return llamaSessionPreflightStepParams(this, params); }
    preflight_step_params(params: unknown) { assertSessionAlive(this.handle); return llamaSessionPreflightStepParams(this, params); }
    acceptsStepParams(params: unknown) { assertSessionAlive(this.handle); return llamaSessionAcceptsStepParams(this, params); }
    accepts_step_params(params: unknown) { assertSessionAlive(this.handle); return llamaSessionAcceptsStepParams(this, params); }
    canExecuteStepParams(params: unknown) { assertSessionAlive(this.handle); return llamaSessionCanExecuteStepParams(this, params); }
    can_execute_step_params(params: unknown) { assertSessionAlive(this.handle); return llamaSessionCanExecuteStepParams(this, params); }
    requireCanExecuteStepParams(params: unknown) { assertSessionAlive(this.handle); return llamaSessionRequireCanExecuteStepParams(this, params); }
    require_can_execute_step_params(params: unknown) { assertSessionAlive(this.handle); return llamaSessionRequireCanExecuteStepParams(this, params); }
    acceptsAllocationFreeStepParams(params: unknown) { assertSessionAlive(this.handle); return llamaSessionAcceptsAllocationFreeStepParams(this, params); }
    accepts_allocation_free_step_params(params: unknown) { assertSessionAlive(this.handle); return llamaSessionAcceptsAllocationFreeStepParams(this, params); }
    requireAllocationFreeStepParams(params: unknown) { assertSessionAlive(this.handle); return llamaSessionRequireAllocationFreeStepParams(this, params); }
    require_allocation_free_step_params(params: unknown) { assertSessionAlive(this.handle); return llamaSessionRequireAllocationFreeStepParams(this, params); }
    acceptsRuntimeOutputAllocationFreeStepParams(params: unknown) { assertSessionAlive(this.handle); return llamaSessionAcceptsRuntimeOutputAllocationFreeStepParams(this, params); }
    accepts_runtime_output_allocation_free_step_params(params: unknown) { assertSessionAlive(this.handle); return llamaSessionAcceptsRuntimeOutputAllocationFreeStepParams(this, params); }
    requireRuntimeOutputAllocationFreeStepParams(params: unknown) { assertSessionAlive(this.handle); return llamaSessionRequireRuntimeOutputAllocationFreeStepParams(this, params); }
    require_runtime_output_allocation_free_step_params(params: unknown) { assertSessionAlive(this.handle); return llamaSessionRequireRuntimeOutputAllocationFreeStepParams(this, params); }
    acceptsNoReadbackStepParams(params: unknown) { assertSessionAlive(this.handle); return llamaSessionAcceptsNoReadbackStepParams(this, params); }
    accepts_no_readback_step_params(params: unknown) { assertSessionAlive(this.handle); return llamaSessionAcceptsNoReadbackStepParams(this, params); }
    requireNoReadbackStepParams(params: unknown) { assertSessionAlive(this.handle); return llamaSessionRequireNoReadbackStepParams(this, params); }
    require_no_readback_step_params(params: unknown) { assertSessionAlive(this.handle); return llamaSessionRequireNoReadbackStepParams(this, params); }
    acceptsReadbackFreeStepParams(params: unknown) { assertSessionAlive(this.handle); return llamaSessionAcceptsReadbackFreeStepParams(this, params); }
    accepts_readback_free_step_params(params: unknown) { assertSessionAlive(this.handle); return llamaSessionAcceptsReadbackFreeStepParams(this, params); }
    requireReadbackFreeStepParams(params: unknown) { assertSessionAlive(this.handle); return llamaSessionRequireReadbackFreeStepParams(this, params); }
    require_readback_free_step_params(params: unknown) { assertSessionAlive(this.handle); return llamaSessionRequireReadbackFreeStepParams(this, params); }
    acceptsHotStepParams(params: unknown) { assertSessionAlive(this.handle); return llamaSessionAcceptsHotStepParams(this, params); }
    accepts_hot_step_params(params: unknown) { assertSessionAlive(this.handle); return llamaSessionAcceptsHotStepParams(this, params); }
    requireHotStepParams(params: unknown) { assertSessionAlive(this.handle); return llamaSessionRequireHotStepParams(this, params); }
    require_hot_step_params(params: unknown) { assertSessionAlive(this.handle); return llamaSessionRequireHotStepParams(this, params); }
    executionPlan(params: unknown) { assertSessionAlive(this.handle); return llamaSessionExecutionPlan(this, params); }
    execution_plan(params: unknown) { assertSessionAlive(this.handle); return llamaSessionExecutionPlan(this, params); }
    requireExecutionPlan(params: unknown) { assertSessionAlive(this.handle); return llamaSessionRequireExecutionPlan(this, params); }
    require_execution_plan(params: unknown) { assertSessionAlive(this.handle); return llamaSessionRequireExecutionPlan(this, params); }
    hotPathPlan(params: unknown) { assertSessionAlive(this.handle); return llamaSessionHotPathPlan(this, params); }
    hot_path_plan(params: unknown) { assertSessionAlive(this.handle); return llamaSessionHotPathPlan(this, params); }
    matchesStepContractSignature(signature: unknown) { assertSessionAlive(this.handle); return llamaSessionMatchesStepContractSignature(this, signature); }
    matches_step_contract_signature(signature: unknown) { assertSessionAlive(this.handle); return llamaSessionMatchesStepContractSignature(this, signature); }
    matchesStepParamsSignature(params: unknown, signature: unknown) { assertSessionAlive(this.handle); return llamaSessionMatchesStepParamsSignature(this, params, signature); }
    matches_step_params_signature(params: unknown, signature: unknown) { assertSessionAlive(this.handle); return llamaSessionMatchesStepParamsSignature(this, params, signature); }
    matchesStepParamsCompatibility(params: unknown, compatibility: unknown) { return llamaSessionMatchesStepParamsCompatibility(this, params, compatibility); }
    matches_step_params_compatibility(params: unknown, compatibility: unknown) { return llamaSessionMatchesStepParamsCompatibility(this, params, compatibility); }
    step(token: unknown, outputValues?: unknown) { return stepLlama(this, token, outputValues); }
    stepTensor(token: unknown, optionsForStep = {}) { return stepLlamaTensor(this, token, optionsForStep); }
    stepInto(outputValues: unknown, token: unknown) { return stepLlamaInto(this, outputValues, token); }
    scalarTokenWindow(token: unknown) { return llamaScalarTokenWindow(this, token); }
    scalarTokenSampleOptions(optionsForToken = {}) { return llamaScalarTokenSampleOptions(this, optionsForToken); }
    noOutputTokenWindowOptions(optionsForToken = {}) { return llamaNoOutputTokenWindowOptions(this, optionsForToken); }
    outputTokenWindowOptions(optionsForToken = {}, outputValues?: unknown) { return llamaOutputTokenWindowOptions(this, optionsForToken, outputValues); }
    scalarTokenExecuteOptions(optionsForToken = {}) { return llamaScalarTokenExecuteOptions(this, optionsForToken); }
    scalarTokenOutputOptions(outputValues: unknown) { return llamaScalarTokenOutputOptions(this, outputValues); }
    advance(token: unknown) { return advanceLlama(this, token); }
    advanceTokens(tokens: unknown, optionsForTokens = {}) { return advanceLlamaTokens(this, tokens, optionsForTokens); }
    advance_tokens(tokens: unknown, optionsForTokens = {}) { return advanceLlamaTokens(this, tokens, optionsForTokens); }
    executeTokensArgmax(tokens: unknown, optionsForTokens = {}) { assertSessionAlive(this.handle); return executeLlamaTokensArgmax(this, tokens, optionsForTokens); }
    execute_tokens_argmax(tokens: unknown, optionsForTokens = {}) { assertSessionAlive(this.handle); return executeLlamaTokensArgmax(this, tokens, optionsForTokens); }
    stepArgmax(token: unknown) { return stepLlamaArgmax(this, token); }
    step_argmax(token: unknown) { return stepLlamaArgmax(this, token); }
    executeTokensSample(tokens: unknown, optionsForTokens = {}) { assertSessionAlive(this.handle); return executeLlamaTokensSample(this, tokens, optionsForTokens); }
    execute_tokens_sample(tokens: unknown, optionsForTokens = {}) { assertSessionAlive(this.handle); return executeLlamaTokensSample(this, tokens, optionsForTokens); }
    stepSample(token: unknown, optionsForToken = {}) { return stepLlamaSample(this, token, optionsForToken); }
    step_sample(token: unknown, optionsForToken = {}) { return stepLlamaSample(this, token, optionsForToken); }
    generateTokensArgmax(tokens: unknown, maxTokens: unknown, optionsForTokens = {}) { assertSessionAlive(this.handle); return generateLlamaTokensArgmax(this, tokens, maxTokens, optionsForTokens); }
    generate_tokens_argmax(tokens: unknown, maxTokens: unknown, optionsForTokens = {}) { assertSessionAlive(this.handle); return generateLlamaTokensArgmax(this, tokens, maxTokens, optionsForTokens); }
    generateTokensArgmaxInto(tokens: unknown, outputTokens: unknown, optionsForTokens = {}) { assertSessionAlive(this.handle); return generateLlamaTokensArgmaxInto(this, tokens, outputTokens, optionsForTokens); }
    generate_tokens_argmax_into(tokens: unknown, outputTokens: unknown, optionsForTokens = {}) { assertSessionAlive(this.handle); return generateLlamaTokensArgmaxInto(this, tokens, outputTokens, optionsForTokens); }
    generateTokensSample(tokens: unknown, maxTokens: unknown, optionsForTokens = {}) { assertSessionAlive(this.handle); return generateLlamaTokensSample(this, tokens, maxTokens, optionsForTokens); }
    generate_tokens_sample(tokens: unknown, maxTokens: unknown, optionsForTokens = {}) { assertSessionAlive(this.handle); return generateLlamaTokensSample(this, tokens, maxTokens, optionsForTokens); }
    generateTokensSampleInto(tokens: unknown, outputTokens: unknown, optionsForTokens = {}) { assertSessionAlive(this.handle); return generateLlamaTokensSampleInto(this, tokens, outputTokens, optionsForTokens); }
    generate_tokens_sample_into(tokens: unknown, outputTokens: unknown, optionsForTokens = {}) { assertSessionAlive(this.handle); return generateLlamaTokensSampleInto(this, tokens, outputTokens, optionsForTokens); }
    executeTokens(tokens: unknown, optionsForTokens = {}) { assertSessionAlive(this.handle); return executeLlamaTokens(this, tokens, optionsForTokens); }
    execute_tokens(tokens: unknown, optionsForTokens = {}) { assertSessionAlive(this.handle); return executeLlamaTokens(this, tokens, optionsForTokens); }
    execute(params: unknown) { return executeLlama(this, params); }
    executeTensor(params: unknown) { return executeLlamaTensor(this, params); }
    execute_tensor(params: unknown) { return executeLlamaTensor(this, params); }
    executeInto(outputValues: unknown, params: unknown) { return executeLlamaInto(this, outputValues, params); }
    execute_into(outputValues: unknown, params: unknown) { return executeLlamaInto(this, outputValues, params); }
    argmaxToken(logits: unknown) { assertSessionAlive(this.handle); return argmaxLlamaToken(this, logits); }
    argmax_token(logits: unknown) { assertSessionAlive(this.handle); return argmaxLlamaToken(this, logits); }
    sampleToken(logits: unknown, optionsForToken = {}) { assertSessionAlive(this.handle); return sampleLlamaToken(this, logits, optionsForToken); }
    sample_token(logits: unknown, optionsForToken = {}) { assertSessionAlive(this.handle); return sampleLlamaToken(this, logits, optionsForToken); }
    prefill(tokens: unknown, outputValues?: unknown, optionsForTokens = {}) { return prefillLlama(this, tokens, outputValues, optionsForTokens); }
    prefillTensor(tokens: unknown, optionsForTokens = {}) { return prefillLlamaTensor(this, tokens, optionsForTokens); }
    prefill_tensor(tokens: unknown, optionsForTokens = {}) { return prefillLlamaTensor(this, tokens, optionsForTokens); }
    prefillInto(outputValues: unknown, tokens: unknown, optionsForTokens = {}) { return prefillLlamaInto(this, outputValues, tokens, optionsForTokens); }
    prefill_into(outputValues: unknown, tokens: unknown, optionsForTokens = {}) { return prefillLlamaInto(this, outputValues, tokens, optionsForTokens); }
    readOutputInto(outputValues: unknown, length = this.vocabSize, byteOffset = 0) { assertSessionAlive(this.handle); return readLlamaOutputInto(this, outputValues, length, byteOffset); }
    readOutputTensor(optionsForRead = {}) { assertSessionAlive(this.handle); return readLlamaOutputTensor(this, optionsForRead); }
    sessionCallProfile() { return llamaSessionCallProfileFacade(this); }
    session_call_profile() { return llamaSessionCallProfileFacade(this); }
    matchesSessionCallProfileSignature(signature: unknown) { return llamaSessionMatchesCallProfileSignature(this, signature); }
    matches_session_call_profile_signature(signature: unknown) { return llamaSessionMatchesCallProfileSignature(this, signature); }
    resetSessionCallProfile() { return resetLlamaSessionCallProfileFacade(this); }
    reset_session_call_profile() { return resetLlamaSessionCallProfileFacade(this); }
    reset() { return resetLlamaSessionFacade(this); }
    runtimeProfile() { return llamaSessionRuntimeProfileFacade(this); }
    runtime_profile() { return llamaSessionRuntimeProfileFacade(this); }
    matchesRuntimeProfileSignature(signature: unknown) { return llamaSessionMatchesRuntimeProfileSignature(this, signature); }
    matches_runtime_profile_signature(signature: unknown) { return llamaSessionMatchesRuntimeProfileSignature(this, signature); }
    resetRuntimeProfile() { return resetLlamaSessionRuntimeProfileFacade(this); }
    reset_runtime_profile() { return resetLlamaSessionRuntimeProfileFacade(this); }
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
    free() { return freeLlamaSessionFacade(this); }
    dispose() { return disposeLlamaSessionFacade(this); }
    [disposeSymbol]() { return disposeLlamaSessionFacade(this); }
  }

  class LlamaModel extends TinyLlamaModel {
    static load(modelPath: unknown) {
      return llamaModelFamilyFacade.load(autoKind, modelPath, (handle: unknown) => new LlamaModel(handle));
    }

    static loadSafetensorsData(data: unknown) {
      return llamaModelFamilyFacade.loadSafetensorsData(autoKind, data, (handle: unknown) => new LlamaModel(handle));
    }

    static probe(source: unknown) {
      return llamaModelFamilyFacade.probe(null, source);
    }

    static probeSafetensorsHeader(header: unknown) {
      return llamaModelFamilyFacade.probeHeader(null, header);
    }

    compile(optionsForCompile = {}) {
      return llamaModelFamilyFacade.compile(this, optionsForCompile, (programHandle: unknown) => new LlamaProgram(programHandle));
    }
  }

  class LlamaProgram extends TinyLlamaProgram {
    bind(optionsForBind = {}) {
      return llamaProgramFacade.bind(this, optionsForBind, bindLlamaSessionHandle, (
        sessionHandle: unknown,
        inspection: { readonly vocabSize: number },
        boundOutput: unknown,
        ownedOutput: unknown,
        layout: unknown,
        kvLayout: unknown,
      ) => new LlamaSession(sessionHandle, inspection.vocabSize, boundOutput, ownedOutput, layout, kvLayout));
    }
  }

  class LlamaSession extends TinyLlamaSession {}

  class SmolLM135MModel extends LlamaModel {
    static load(modelPath: unknown) {
      return llamaModelFamilyFacade.load(smollm135mKind, modelPath, (handle: unknown) => new SmolLM135MModel(handle));
    }

    static loadSafetensorsData(data: unknown) {
      return llamaModelFamilyFacade.loadSafetensorsData(smollm135mKind, data, (handle: unknown) => new SmolLM135MModel(handle));
    }

    static probe(source: unknown) {
      return llamaModelFamilyFacade.probe("smollm-135m", source);
    }

    static probeSafetensorsHeader(header: unknown) {
      return llamaModelFamilyFacade.probeHeader("smollm-135m", header);
    }

    compile(optionsForCompile = {}) {
      return llamaModelFamilyFacade.compile(this, optionsForCompile, (programHandle: unknown) => new SmolLM135MProgram(programHandle));
    }
  }

  class SmolLM135MProgram extends LlamaProgram {
    bind(optionsForBind = {}) {
      return llamaProgramFacade.bind(this, optionsForBind, bindLlamaSessionHandle, (
        sessionHandle: unknown,
        inspection: { readonly vocabSize: number },
        boundOutput: unknown,
        ownedOutput: unknown,
        layout: unknown,
        kvLayout: unknown,
      ) => new SmolLM135MSession(sessionHandle, inspection.vocabSize, boundOutput, ownedOutput, layout, kvLayout));
    }
  }

  class SmolLM135MSession extends LlamaSession {
    static vocabSize = 49152;
  }

  return Object.freeze({
    TinyLlamaModel,
    TinyLlamaProgram,
    TinyLlamaSession,
    LlamaModel,
    LlamaProgram,
    LlamaSession,
    SmolLM135MModel,
    SmolLM135MProgram,
    SmolLM135MSession,
  }) as unknown as LlamaFamilySurface;
}

export const llamaFamilySurfaceManifest = Object.freeze({
  kind: "zgml-llama-family-surface",
  ...tsRuntimeManifestPolicy("src/ts/runtime/llama_family_surface.ts", "LLaMA model handles -> Program -> Session surface"),
});
