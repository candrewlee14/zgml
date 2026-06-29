import { createAdapterProgramDeviceFactory } from "./program_device_factory_surface.js";
import type { SharedFrontendRuntime } from "./shared_frontend_runtime.js";
import type { ProgramDeviceClassOptions } from "../runtime/program_device.js";
import type { ProgramBufferFactoryHelpersOptions } from "../runtime/program_buffer_factory.js";
import type { ProgramFacadePolicyHelpersOptions } from "../runtime/program_facade_policy.js";
import type {
  GenericProgramFacadeHelpersOptions,
  LlamaProgramFacadeHelpersOptions,
} from "../runtime/program_facade.js";
import type {
  LlamaModelFamilyFacadeHelpersOptions,
  ModelFacadePolicyHelpersOptions,
} from "../runtime/model_source.js";
import type { NativeBuffer } from "../public_api.js";

type AdapterProgramDeviceOptions<THandle, TNativeBuffer extends NativeBuffer, TLlamaKvCache> =
  ProgramDeviceClassOptions<TNativeBuffer, TLlamaKvCache, THandle>;
type AdapterProgramBufferFactoryOptions<THandle, TNativeBuffer extends NativeBuffer, TLlamaKvCache> =
  ProgramBufferFactoryHelpersOptions<THandle, TNativeBuffer, TLlamaKvCache>;
type AdapterGenericProgramOptions<THandle, TNativeBuffer extends NativeBuffer, TProgramDesc> =
  GenericProgramFacadeHelpersOptions<THandle, TNativeBuffer, unknown, TProgramDesc>;
type AdapterLlamaProgramOptions<THandle, TNativeBuffer extends NativeBuffer, TLlamaKvCache> =
  LlamaProgramFacadeHelpersOptions<THandle, TNativeBuffer, unknown, TLlamaKvCache>;
type AdapterModelFamilyOptions<TInspection, TProgram> =
  LlamaModelFamilyFacadeHelpersOptions<unknown, TInspection, TProgram>;

export type AdapterProgramRuntimeSurfaceOptions<
  THandle = unknown,
  TNativeBuffer extends NativeBuffer = NativeBuffer,
  TLlamaKvCache = unknown,
  TModelHandle = unknown,
  TInspection = unknown,
  TProgramDesc = unknown,
> = Readonly<{
  sharedFrontend: SharedFrontendRuntime;
  assertAlive: AdapterProgramDeviceOptions<THandle, TNativeBuffer, TLlamaKvCache>["assertProgramAlive"] &
    ProgramFacadePolicyHelpersOptions<THandle>["assertProgramAlive"] &
    ModelFacadePolicyHelpersOptions<TModelHandle>["assertModelAlive"];
  programDeviceHandle: AdapterProgramDeviceOptions<THandle, TNativeBuffer, TLlamaKvCache>["programDeviceHandle"];
  programCreateDeviceBuffer: AdapterProgramDeviceOptions<THandle, TNativeBuffer, TLlamaKvCache>["createDeviceBuffer"] &
    AdapterProgramBufferFactoryOptions<THandle, TNativeBuffer, TLlamaKvCache>["createDeviceBuffer"];
  programImportDeviceBuffer: AdapterProgramDeviceOptions<THandle, TNativeBuffer, TLlamaKvCache>["importDeviceBuffer"];
  isNativeBuffer: AdapterProgramDeviceOptions<THandle, TNativeBuffer, TLlamaKvCache>["isNativeBuffer"];
  createProgramDeviceKvCache: AdapterProgramDeviceOptions<THandle, TNativeBuffer, TLlamaKvCache>["createKvCache"];
  programRequirements: AdapterProgramBufferFactoryOptions<THandle, TNativeBuffer, TLlamaKvCache>["programRequirements"] &
    ProgramFacadePolicyHelpersOptions<THandle>["programRequirements"];
  llamaKvCacheRequirements: AdapterProgramBufferFactoryOptions<THandle, TNativeBuffer, TLlamaKvCache>["llamaKvCacheRequirements"] &
    NonNullable<AdapterLlamaProgramOptions<THandle, TNativeBuffer, TLlamaKvCache>["llamaKvCacheRequirements"]>;
  programCreateBuffer: AdapterProgramBufferFactoryOptions<THandle, TNativeBuffer, TLlamaKvCache>["createHostBuffer"];
  programCreateOutputBuffer: NonNullable<AdapterProgramBufferFactoryOptions<THandle, TNativeBuffer, TLlamaKvCache>["createHostOutputBuffer"]>;
  requireNativeBuffer: AdapterProgramBufferFactoryOptions<THandle, TNativeBuffer, TLlamaKvCache>["requireNativeBuffer"];
  createProgramBufferFactoryKvCache: AdapterProgramBufferFactoryOptions<THandle, TNativeBuffer, TLlamaKvCache>["createLlamaKvCache"];
  programModelCompatibility: ProgramFacadePolicyHelpersOptions<THandle>["programModelCompatibility"];
  programExecutionCapabilities: ProgramFacadePolicyHelpersOptions<THandle>["programExecutionCapabilities"];
  programRuntimeProfile: ProgramFacadePolicyHelpersOptions<THandle>["programRuntimeProfile"];
  programResetRuntimeProfile: ProgramFacadePolicyHelpersOptions<THandle>["programResetRuntimeProfile"];
  programFree: ProgramFacadePolicyHelpersOptions<THandle>["programFree"];
  nullProgramHandle: ProgramFacadePolicyHelpersOptions<THandle>["nullProgramHandle"];
  inspectModelHandle: ModelFacadePolicyHelpersOptions<TModelHandle>["modelInspect"];
  compileModelProgramHandle: ModelFacadePolicyHelpersOptions<TModelHandle>["modelCompile"];
  modelFree: ModelFacadePolicyHelpersOptions<TModelHandle>["modelFree"];
  nullModelHandle: ModelFacadePolicyHelpersOptions<TModelHandle>["nullModelHandle"];
  inspectExecutableProgram: NonNullable<AdapterGenericProgramOptions<THandle, TNativeBuffer, TProgramDesc>["inspectExecutableProgram"]> &
    NonNullable<AdapterLlamaProgramOptions<THandle, TNativeBuffer, TLlamaKvCache>["inspectExecutableProgram"]>;
  programInputShape: NonNullable<AdapterGenericProgramOptions<THandle, TNativeBuffer, TProgramDesc>["programInputShape"]>;
  programOutputShape: NonNullable<AdapterGenericProgramOptions<THandle, TNativeBuffer, TProgramDesc>["programOutputShape"]>;
  createProgramOutputBuffer: NonNullable<AdapterGenericProgramOptions<THandle, TNativeBuffer, TProgramDesc>["createProgramOutputBuffer"]> &
    NonNullable<AdapterLlamaProgramOptions<THandle, TNativeBuffer, TLlamaKvCache>["createProgramOutputBuffer"]>;
  createProgramNamedBuffer: NonNullable<AdapterGenericProgramOptions<THandle, TNativeBuffer, TProgramDesc>["createProgramNamedBuffer"]> &
    NonNullable<AdapterLlamaProgramOptions<THandle, TNativeBuffer, TLlamaKvCache>["createProgramNamedBuffer"]>;
  createProgramBuffer: NonNullable<AdapterGenericProgramOptions<THandle, TNativeBuffer, TProgramDesc>["createProgramBuffer"]>;
  bindModuleThroughProgram: NonNullable<AdapterGenericProgramOptions<THandle, TNativeBuffer, TProgramDesc>["bindModuleThroughProgram"]>;
  programBindingPlan: NonNullable<AdapterGenericProgramOptions<THandle, TNativeBuffer, TProgramDesc>["programBindingPlan"]>;
  createTinyLlamaModelHandle: AdapterModelFamilyOptions<unknown, unknown>["createTinyLlamaModelHandle"];
  loadModelPath: AdapterModelFamilyOptions<unknown, unknown>["loadModelPath"];
  loadSafetensorsDataHandle: AdapterModelFamilyOptions<unknown, unknown>["loadSafetensorsDataHandle"];
  probeModel: AdapterModelFamilyOptions<unknown, unknown>["probeModel"];
  probeSafetensorsHeader: AdapterModelFamilyOptions<unknown, unknown>["probeSafetensorsHeader"];
  inspectLlamaProgram: NonNullable<AdapterLlamaProgramOptions<THandle, TNativeBuffer, TLlamaKvCache>["inspectLlamaProgram"]>;
  createProgramLlamaKvCache: NonNullable<AdapterLlamaProgramOptions<THandle, TNativeBuffer, TLlamaKvCache>["createProgramLlamaKvCache"]>;
  bindLlamaProgramSession: NonNullable<AdapterLlamaProgramOptions<THandle, TNativeBuffer, TLlamaKvCache>["bindLlamaProgramSession"]>;
}>;

export function createAdapterProgramRuntimeSurface<
  THandle = unknown,
  TNativeBuffer extends NativeBuffer = NativeBuffer,
  TLlamaKvCache = unknown,
  TModelHandle = unknown,
  TInspection = unknown,
  TProgramDesc = unknown,
>(options: AdapterProgramRuntimeSurfaceOptions<THandle, TNativeBuffer, TLlamaKvCache, TModelHandle, TInspection, TProgramDesc>) {
  const ProgramDevice = options.sharedFrontend.createProgramDeviceClass<TNativeBuffer, TLlamaKvCache, THandle>({
    assertProgramAlive: options.assertAlive,
    programDeviceHandle: options.programDeviceHandle,
    createDeviceBuffer: options.programCreateDeviceBuffer,
    importDeviceBuffer: options.programImportDeviceBuffer,
    isNativeBuffer: options.isNativeBuffer,
    createKvCache: options.createProgramDeviceKvCache,
  });

  const createProgramDevice = createAdapterProgramDeviceFactory({
    getProgramDeviceClass: () => ProgramDevice,
  });

  const programBufferFactories = options.sharedFrontend.createProgramBufferFactoryHelpers<THandle, TNativeBuffer, TLlamaKvCache>({
    programRequirements: options.programRequirements,
    llamaKvCacheRequirements: options.llamaKvCacheRequirements,
    createHostBuffer: options.programCreateBuffer,
    createHostOutputBuffer: options.programCreateOutputBuffer,
    createDeviceBuffer: options.programCreateDeviceBuffer,
    requireNativeBuffer: options.requireNativeBuffer,
    createLlamaKvCache: options.createProgramBufferFactoryKvCache,
  });

  const programFacadePolicy = options.sharedFrontend.createProgramFacadePolicyHelpers<THandle>({
    assertProgramAlive: options.assertAlive,
    programRequirements: options.programRequirements,
    programModelCompatibility: options.programModelCompatibility,
    programExecutionCapabilities: options.programExecutionCapabilities,
    programRuntimeProfile: options.programRuntimeProfile,
    programResetRuntimeProfile: options.programResetRuntimeProfile,
    programFree: options.programFree,
    nullProgramHandle: options.nullProgramHandle,
  });

  const modelFacadePolicy = options.sharedFrontend.createModelFacadePolicyHelpers<TModelHandle>({
    assertModelAlive: options.assertAlive,
    modelInspect: options.inspectModelHandle,
    modelCompile: options.compileModelProgramHandle,
    modelFree: options.modelFree,
    nullModelHandle: options.nullModelHandle,
  });

  const programDeviceHandleForFacade: NonNullable<AdapterGenericProgramOptions<THandle, TNativeBuffer, TProgramDesc>["programDeviceHandle"]> =
    (handle, placement = "webgpu") => options.programDeviceHandle(handle, placement);
  const importDeviceBufferForFacade: NonNullable<AdapterGenericProgramOptions<THandle, TNativeBuffer, TProgramDesc>["importDeviceBuffer"]> =
    (handle, kind, source) => options.programImportDeviceBuffer(
      handle,
      kind,
      source as Parameters<typeof options.programImportDeviceBuffer>[2],
    );

  const genericProgramFacade = options.sharedFrontend.createGenericProgramFacadeHelpers<THandle, TNativeBuffer, unknown, TProgramDesc>({
    assertProgramAlive: options.assertAlive,
    programPolicy: programFacadePolicy,
    inspectExecutableProgram: options.inspectExecutableProgram,
    programInputShape: options.programInputShape,
    programOutputShape: options.programOutputShape,
    createProgramOutputBuffer: options.createProgramOutputBuffer,
    createProgramNamedBuffer: options.createProgramNamedBuffer,
    createProgramBuffer: options.createProgramBuffer,
    programDeviceHandle: programDeviceHandleForFacade,
    createProgramDevice,
    importDeviceBuffer: importDeviceBufferForFacade,
    bindModuleThroughProgram: options.bindModuleThroughProgram,
    programBindingPlan: options.programBindingPlan,
  });

  const llamaModelFamilyFacade = options.sharedFrontend.createLlamaModelFamilyFacadeHelpers<unknown, unknown, unknown>({
    modelPolicy: modelFacadePolicy,
    createTinyLlamaModelHandle: options.createTinyLlamaModelHandle,
    loadModelPath: options.loadModelPath,
    loadSafetensorsDataHandle: options.loadSafetensorsDataHandle,
    probeModel: options.probeModel,
    probeSafetensorsHeader: options.probeSafetensorsHeader,
  });

  const llamaProgramFacade = options.sharedFrontend.createLlamaProgramFacadeHelpers<THandle, TNativeBuffer, unknown, TLlamaKvCache>({
    assertProgramAlive: options.assertAlive,
    programPolicy: programFacadePolicy,
    inspectLlamaProgram: options.inspectLlamaProgram,
    inspectExecutableProgram: options.inspectExecutableProgram,
    llamaKvCacheRequirements: options.llamaKvCacheRequirements,
    createProgramOutputBuffer: options.createProgramOutputBuffer,
    createProgramNamedBuffer: options.createProgramNamedBuffer,
    programDeviceHandle: programDeviceHandleForFacade,
    createProgramDevice,
    importDeviceBuffer: importDeviceBufferForFacade,
    createProgramLlamaKvCache: options.createProgramLlamaKvCache,
    bindLlamaProgramSession: options.bindLlamaProgramSession,
  });

  return Object.freeze({
    ProgramDevice,
    createProgramDevice,
    programBufferFactories,
    programFacadePolicy,
    modelFacadePolicy,
    genericProgramFacade,
    llamaModelFamilyFacade,
    llamaProgramFacade,
  });
}
