"use strict";

import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";
import {
  modelKinds,
  modelKindName,
} from "./abi.js";
import { supportedCheckpointModelsFromCatalog } from "./model_source_catalog.js";

declare const TextEncoder: {
  new(): { encode(value: string): Uint8Array };
};

type AnyRecord = Record<string, any>;
type LoadModelKind = "auto" | "tiny-llama" | "tiny-llama-2layer" | "smollm-135m" | "smollm2-360m";
type LoadModelOptions = string | { kind?: string; modelKind?: string };
type BytesSource = Uint8Array | ArrayBuffer;
type BivariantCallback<Args extends readonly unknown[], Return> = {
  bivarianceHack(...args: Args): Return;
}["bivarianceHack"];
type SafetensorsOpenFileCallback<Fd> = (modelPath: string) => Fd;
type SafetensorsCloseFileCallback<Fd> = BivariantCallback<[fd: Fd], void>;
type SafetensorsReadFileCallback<Fd> = BivariantCallback<[fd: Fd, buffer: Uint8Array, offset: number, length: number, position: number], number>;
type SafetensorsAllocBytesCallback = (byteLength: number) => Uint8Array;
type SafetensorsReadU64LECallback = (bytes: Uint8Array) => bigint;
type ModelProbeOptions = Readonly<{ kind: LoadModelKind }>;
type ModelSourcePathLoader<Model> = (modelPath: string) => Model;
type ModelSourceDataLoader<Model> = (data: BytesSource) => Model;
type ModelSourcePathLoaderMap<Model> = Readonly<Record<LoadModelKind, ModelSourcePathLoader<Model>>>;
type ModelSourceDataLoaderMap<Model> = Readonly<Record<LoadModelKind, ModelSourceDataLoader<Model>>>;
type ModelPolicyFacade<Model = unknown, Inspection = unknown, Program = unknown> = Readonly<{
  inspect: BivariantCallback<[model: Model], Inspection>;
  compile: BivariantCallback<[model: Model, options: unknown, createProgram: (handle: unknown) => Program], Program>;
  free: BivariantCallback<[model: Model], unknown>;
  dispose: BivariantCallback<[model: Model], unknown>;
}>;

export type SafetensorsFileHeaderHelpersOptions<Fd = unknown> = Readonly<{
  openFile: SafetensorsOpenFileCallback<Fd>;
  closeFile: SafetensorsCloseFileCallback<Fd>;
  readFile: SafetensorsReadFileCallback<Fd>;
  allocBytes: SafetensorsAllocBytesCallback;
  readU64LE: SafetensorsReadU64LECallback;
}>;

export type ModelSourceFacadeHelpersOptions<Inspection = unknown, Model = unknown> = Readonly<{
  readSafetensorsHeaderFile: (modelPath: string) => Uint8Array;
  probePath: (modelPath: string, kind: LoadModelKind) => Inspection;
  probeSafetensorsData: BivariantCallback<[data: BytesSource, options: ModelProbeOptions], Inspection>;
  probeSafetensorsHeader: BivariantCallback<[header: string | Uint8Array, options: ModelProbeOptions], Inspection>;
  loadPathByKind: ModelSourcePathLoaderMap<Model>;
  loadSafetensorsDataByKind: ModelSourceDataLoaderMap<Model>;
}>;

export type SupportedCheckpointCatalogHelpersOptions = Readonly<{
  countSupportedCheckpoints: () => number;
  inspectSupportedCheckpoint: (index: number) => unknown;
}>;

export type ModelHandleHelpersOptions<Handle = unknown> = Readonly<{
  isBindableModel: (model: unknown) => boolean;
  isCompatibleModel: (model: unknown) => boolean;
  getModelHandle: (model: unknown) => Handle;
  assertAlive: BivariantCallback<[handle: Handle, label: string], void>;
  nullHandle?: Handle | null;
}>;

export type ModelFacadePolicyHelpersOptions<Handle = unknown> = Readonly<{
  assertModelAlive: BivariantCallback<[handle: Handle, label: string], void>;
  modelInspect: BivariantCallback<[handle: Handle], unknown>;
  modelCompile: BivariantCallback<[handle: Handle, options: unknown], unknown>;
  modelFree: BivariantCallback<[handle: Handle], void>;
  nullModelHandle: Handle;
}>;

export type LlamaModelFamilyFacadeHelpersOptions<Model = unknown, Inspection = unknown, Program = unknown> = Readonly<{
  modelPolicy: ModelPolicyFacade<Model, Inspection, Program>;
  createTinyLlamaModelHandle: () => unknown;
  loadModelPath: BivariantCallback<[nativeKind: unknown, modelPath: string], unknown>;
  loadSafetensorsDataHandle: BivariantCallback<[nativeKind: unknown, data: BytesSource], unknown>;
  probeModel: BivariantCallback<[source: unknown, options?: ModelProbeOptions], Inspection>;
  probeSafetensorsHeader: BivariantCallback<[header: string | Uint8Array, options?: ModelProbeOptions], Inspection>;
}>;

export const modelSourceManifest = Object.freeze({
  kind: "zgml-model-source",
  ...tsRuntimeManifestPolicy("src/ts/runtime/model_source.ts", "ModelSource -> Program -> Session"),
});

const safetensorsTextEncoder = new TextEncoder();

export function safetensorsHeaderBytes(header: string | Uint8Array): Uint8Array {
  if (typeof header === "string") return safetensorsTextEncoder.encode(header);
  if (header instanceof Uint8Array) return header;
  throw new Error("safetensors header must be a string, Buffer, or Uint8Array");
}

export function safetensorsDataBytes(data: BytesSource): Uint8Array {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  throw new Error("safetensors data must be a Buffer, Uint8Array, or ArrayBuffer");
}

export function isSafetensorsDataSource(source: unknown): source is BytesSource {
  return source instanceof Uint8Array || source instanceof ArrayBuffer;
}

export function isSafetensorsPath(modelPath: unknown): modelPath is string {
  return typeof modelPath === "string" && modelPath.toLowerCase().endsWith(".safetensors");
}

export function normalizeLoadModelKind(options: LoadModelOptions = {}): LoadModelKind {
  const kind = typeof options === "string"
    ? options
    : (options.kind ?? options.modelKind ?? "auto");
  switch (kind) {
    case "auto":
    case "llama":
    case "llama-auto":
      return "auto";
    case "tiny-llama":
    case "tinyllama":
      return "tiny-llama";
    case "tiny-llama-2layer":
    case "tiny-llama-2-layer":
    case "tinyllama2layer":
      return "tiny-llama-2layer";
    case "smollm":
    case "smollm135m":
    case "smollm-135m":
      return "smollm-135m";
    case "smollm2":
    case "smollm2360m":
    case "smollm2-360m":
      return "smollm2-360m";
    default:
      throw new Error(`unknown zgml model load kind: ${kind}`);
  }
}

export function modelLoadKindId(kind: LoadModelKind): number {
  switch (kind) {
    case "auto":
      return modelKinds.auto;
    case "tiny-llama":
      return modelKinds.tinyLlama;
    case "tiny-llama-2layer":
      return modelKinds.tinyLlama2Layer;
    case "smollm-135m":
      return modelKinds.smollm135m;
    case "smollm2-360m":
      return modelKinds.smollm2_360m;
    default:
      throw new Error("unreachable model load kind");
  }
}

export function createSafetensorsFileHeaderHelpers<Fd = unknown>(options: SafetensorsFileHeaderHelpersOptions<Fd>) {
  const openFile = options.openFile;
  const closeFile = options.closeFile;
  const readFile = options.readFile;
  const allocBytes = options.allocBytes;
  const readU64LE = options.readU64LE;
  if (typeof openFile !== "function") {
    throw new Error("createSafetensorsFileHeaderHelpers requires openFile");
  }
  if (typeof closeFile !== "function") {
    throw new Error("createSafetensorsFileHeaderHelpers requires closeFile");
  }
  if (typeof readFile !== "function") {
    throw new Error("createSafetensorsFileHeaderHelpers requires readFile");
  }
  if (typeof allocBytes !== "function") {
    throw new Error("createSafetensorsFileHeaderHelpers requires allocBytes");
  }
  if (typeof readU64LE !== "function") {
    throw new Error("createSafetensorsFileHeaderHelpers requires readU64LE");
  }

  const openFileFn = openFile;
  const closeFileFn = closeFile;
  const readFileFn = readFile;
  const allocBytesFn = allocBytes;
  const readU64LEFn = readU64LE;

  function readExact(fd: Fd, buffer: Uint8Array, position: number, label: string) {
    let offset = 0;
    while (offset < buffer.length) {
      const n = readFileFn(fd, buffer, offset, buffer.length - offset, position + offset);
      if (n === 0) throw new Error(`truncated safetensors ${label}`);
      offset += n;
    }
  }

  function readSafetensorsHeaderFile(modelPath: string) {
    const fd = openFileFn(modelPath);
    try {
      const lenBytes = allocBytesFn(8);
      readExact(fd, lenBytes, 0, "header length");
      const headerLen = readU64LEFn(lenBytes);
      if (headerLen > BigInt(Number.MAX_SAFE_INTEGER)) {
        throw new Error(`safetensors header is too large: ${headerLen}`);
      }
      const header = allocBytesFn(Number(headerLen));
      readExact(fd, header, 8, "header");
      return header;
    } finally {
      closeFileFn(fd);
    }
  }

  return Object.freeze({ readSafetensorsHeaderFile });
}

export function createModelSourceFacadeHelpers<Inspection = unknown, Model = unknown>(options: ModelSourceFacadeHelpersOptions<Inspection, Model>) {
  const readSafetensorsHeaderFile = options.readSafetensorsHeaderFile;
  const probePath = options.probePath;
  const probeSafetensorsData = options.probeSafetensorsData;
  const probeSafetensorsHeader = options.probeSafetensorsHeader;
  const loadPathByKind = options.loadPathByKind;
  const loadSafetensorsDataByKind = options.loadSafetensorsDataByKind;
  if (typeof readSafetensorsHeaderFile !== "function") {
    throw new Error("createModelSourceFacadeHelpers requires readSafetensorsHeaderFile");
  }
  if (typeof probePath !== "function") {
    throw new Error("createModelSourceFacadeHelpers requires probePath");
  }
  if (typeof probeSafetensorsData !== "function") {
    throw new Error("createModelSourceFacadeHelpers requires probeSafetensorsData");
  }
  if (typeof probeSafetensorsHeader !== "function") {
    throw new Error("createModelSourceFacadeHelpers requires probeSafetensorsHeader");
  }
  if (!loadPathByKind || typeof loadPathByKind !== "object") {
    throw new Error("createModelSourceFacadeHelpers requires loadPathByKind");
  }
  if (!loadSafetensorsDataByKind || typeof loadSafetensorsDataByKind !== "object") {
    throw new Error("createModelSourceFacadeHelpers requires loadSafetensorsDataByKind");
  }

  const readSafetensorsHeaderFileFn = readSafetensorsHeaderFile;
  const probePathFn = probePath;
  const probeSafetensorsDataFn = probeSafetensorsData;
  const probeSafetensorsHeaderFn = probeSafetensorsHeader;

  function requireModelPathSource(source: unknown) {
    if (typeof source === "string") return source;
    throw new Error("model source must be a path string or safetensors bytes");
  }

  function modelLoader<Input, Output>(loaders: Readonly<Record<LoadModelKind, (input: Input) => Output>>, kind: LoadModelKind, label: string) {
    const loader = loaders[kind];
    if (typeof loader !== "function") {
      throw new Error(`unreachable model ${label} kind`);
    }
    return loader;
  }

  function probeModel(source: unknown, options: LoadModelOptions = {}): Inspection {
    const kind = normalizeLoadModelKind(options);
    if (isSafetensorsDataSource(source)) return probeSafetensorsDataFn(source, { kind });
    const path = requireModelPathSource(source);
    if (isSafetensorsPath(path)) {
      if (kind === "auto") return probePathFn(path, kind);
      return probeSafetensorsHeaderFn(readSafetensorsHeaderFileFn(path), { kind });
    }
    return probePathFn(path, kind);
  }

  function loadModel(source: unknown, options: LoadModelOptions = {}): Model {
    if (isSafetensorsDataSource(source)) return loadSafetensorsData(source, options);
    const path = requireModelPathSource(source);
    const kind = normalizeLoadModelKind(options);
    return modelLoader(loadPathByKind, kind, "load")(path);
  }

  function loadSafetensorsData(data: BytesSource, options: LoadModelOptions = {}): Model {
    const kind = normalizeLoadModelKind(options);
    return modelLoader(loadSafetensorsDataByKind, kind, "safetensors data load")(data);
  }

  return Object.freeze({
    probeModel,
    loadModel,
    loadSafetensorsData,
  });
}

export function createSupportedCheckpointCatalogHelpers(options: SupportedCheckpointCatalogHelpersOptions) {
  const countSupportedCheckpoints = options.countSupportedCheckpoints;
  const inspectSupportedCheckpoint = options.inspectSupportedCheckpoint;
  if (typeof countSupportedCheckpoints !== "function") {
    throw new Error("createSupportedCheckpointCatalogHelpers requires countSupportedCheckpoints");
  }
  if (typeof inspectSupportedCheckpoint !== "function") {
    throw new Error("createSupportedCheckpointCatalogHelpers requires inspectSupportedCheckpoint");
  }

  const countSupportedCheckpointsFn = countSupportedCheckpoints;
  const inspectSupportedCheckpointFn = inspectSupportedCheckpoint;

  function supportedCheckpointModels() {
    return supportedCheckpointModelsFromCatalog(
      () => countSupportedCheckpointsFn(),
      (index) => inspectSupportedCheckpointFn(index),
    );
  }

  return Object.freeze({ supportedCheckpointModels });
}

export function createModelHandleHelpers<Handle = unknown>(options: ModelHandleHelpersOptions<Handle>) {
  const isBindableModel = options.isBindableModel;
  const isCompatibleModel = options.isCompatibleModel;
  const getModelHandle = options.getModelHandle;
  const assertAlive = options.assertAlive;
  const nullHandle: Handle | null = Object.prototype.hasOwnProperty.call(options, "nullHandle")
    ? (options.nullHandle ?? null)
    : null;
  if (typeof isBindableModel !== "function") {
    throw new Error("createModelHandleHelpers requires isBindableModel");
  }
  if (typeof isCompatibleModel !== "function") {
    throw new Error("createModelHandleHelpers requires isCompatibleModel");
  }
  if (typeof getModelHandle !== "function") {
    throw new Error("createModelHandleHelpers requires getModelHandle");
  }
  if (typeof assertAlive !== "function") {
    throw new Error("createModelHandleHelpers requires assertAlive");
  }

  const isBindableModelFn = isBindableModel;
  const isCompatibleModelFn = isCompatibleModel;
  const getModelHandleFn = getModelHandle;
  const assertAliveFn = assertAlive;

  function liveModelHandle(model: unknown) {
    const handle = getModelHandleFn(model);
    assertAliveFn(handle, "model");
    return handle;
  }

  function modelHandleForBind(model: unknown) {
    if (model == null) return nullHandle;
    if (isBindableModelFn(model)) return liveModelHandle(model);
    throw new Error("model must be a zgml LLaMA model handle");
  }

  function modelHandleForCompatibility(model: unknown) {
    if (isCompatibleModelFn(model)) return liveModelHandle(model);
    throw new Error("model must be a zgml model handle");
  }

  return Object.freeze({
    modelHandleForBind,
    modelHandleForCompatibility,
  });
}

export function createModelFacadePolicyHelpers<Handle = unknown>(options: ModelFacadePolicyHelpersOptions<Handle>) {
  const assertModelAlive = options.assertModelAlive;
  const modelInspect = options.modelInspect;
  const modelCompile = options.modelCompile;
  const modelFree = options.modelFree;
  const hasNullModelHandle = Object.prototype.hasOwnProperty.call(options, "nullModelHandle");
  const nullModelHandle = hasNullModelHandle ? options.nullModelHandle : null;

  if (typeof assertModelAlive !== "function") {
    throw new Error("createModelFacadePolicyHelpers requires assertModelAlive");
  }
  if (typeof modelInspect !== "function") {
    throw new Error("createModelFacadePolicyHelpers requires modelInspect");
  }
  if (typeof modelCompile !== "function") {
    throw new Error("createModelFacadePolicyHelpers requires modelCompile");
  }
  if (typeof modelFree !== "function" || !hasNullModelHandle) {
    throw new Error("createModelFacadePolicyHelpers requires model lifecycle callbacks");
  }

  const assertModelAliveFn = assertModelAlive;
  const modelInspectFn = modelInspect;
  const modelCompileFn = modelCompile;
  const modelFreeFn = modelFree;

  function inspect(model: AnyRecord) {
    assertModelAliveFn(model.handle, "model");
    return modelInspectFn(model.handle);
  }

  function compile(model: AnyRecord, options: unknown, createProgram: (handle: unknown) => unknown) {
    assertModelAliveFn(model.handle, "model");
    if (typeof createProgram !== "function") {
      throw new Error("model compile requires a Program factory");
    }
    return createProgram(modelCompileFn(model.handle, options));
  }

  function free(model: AnyRecord) {
    if (model.handle !== nullModelHandle) {
      modelFreeFn(model.handle);
      model.handle = nullModelHandle;
    }
  }

  function dispose(model: AnyRecord) {
    return free(model);
  }

  return Object.freeze({
    inspect,
    compile,
    free,
    dispose,
  });
}

export function createLlamaModelFamilyFacadeHelpers<Model = unknown, Inspection = unknown, Program = unknown>(options: LlamaModelFamilyFacadeHelpersOptions<Model, Inspection, Program>) {
  const modelPolicy = options.modelPolicy;
  const createTinyLlamaModelHandle = options.createTinyLlamaModelHandle;
  const loadModelPath = options.loadModelPath;
  const loadSafetensorsDataHandle = options.loadSafetensorsDataHandle;
  const probeModel = options.probeModel;
  const probeSafetensorsHeader = options.probeSafetensorsHeader;

  if (
    !modelPolicy ||
    typeof modelPolicy.inspect !== "function" ||
    typeof modelPolicy.compile !== "function" ||
    typeof modelPolicy.free !== "function" ||
    typeof modelPolicy.dispose !== "function"
  ) {
    throw new Error("createLlamaModelFamilyFacadeHelpers requires modelPolicy");
  }
  if (
    typeof createTinyLlamaModelHandle !== "function" ||
    typeof loadModelPath !== "function" ||
    typeof loadSafetensorsDataHandle !== "function" ||
    typeof probeModel !== "function" ||
    typeof probeSafetensorsHeader !== "function"
  ) {
    throw new Error("createLlamaModelFamilyFacadeHelpers requires model callbacks");
  }

  const createTinyLlamaModelHandleFn = createTinyLlamaModelHandle;
  const loadModelPathFn = loadModelPath;
  const loadSafetensorsDataHandleFn = loadSafetensorsDataHandle;
  const probeModelFn = probeModel;
  const probeSafetensorsHeaderFn = probeSafetensorsHeader;

  function probeOptions(probeKind: LoadModelOptions | null): ModelProbeOptions | undefined {
    return probeKind == null ? undefined : { kind: normalizeLoadModelKind(probeKind) };
  }

  function createTinyLlama(createModel: (handle: unknown) => unknown) {
    if (typeof createModel !== "function") throw new Error("model create requires a Model factory");
    return createModel(createTinyLlamaModelHandleFn());
  }

  function load(nativeKind: unknown, modelPath: string, createModel: (handle: unknown) => unknown) {
    if (typeof createModel !== "function") throw new Error("model load requires a Model factory");
    return createModel(loadModelPathFn(nativeKind, modelPath));
  }

  function loadSafetensorsData(nativeKind: unknown, data: BytesSource, createModel: (handle: unknown) => unknown) {
    if (typeof createModel !== "function") throw new Error("safetensors data load requires a Model factory");
    return createModel(loadSafetensorsDataHandleFn(nativeKind, data));
  }

  function probe(probeKind: LoadModelOptions | null, source: unknown): Inspection {
    return probeModelFn(source, probeOptions(probeKind));
  }

  function probeHeader(probeKind: LoadModelOptions | null, header: string | Uint8Array): Inspection {
    return probeSafetensorsHeaderFn(header, probeOptions(probeKind));
  }

  function inspect(model: Model): Inspection {
    return modelPolicy.inspect(model);
  }

  function compile(model: Model, compileOptions: unknown, createProgram: (handle: unknown) => Program): Program {
    return modelPolicy.compile(model, compileOptions, createProgram);
  }

  function free(model: Model) {
    return modelPolicy.free(model);
  }

  function dispose(model: Model) {
    return modelPolicy.dispose(model);
  }

  return Object.freeze({
    createTinyLlama,
    load,
    loadSafetensorsData,
    probe,
    probeHeader,
    inspect,
    compile,
    free,
    dispose,
  });
}

export { modelKindName };
