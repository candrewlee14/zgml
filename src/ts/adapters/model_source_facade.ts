import {
  createModelSourceFacadeHelpers,
} from "../runtime/model_source.js";

type BytesSource = Uint8Array | ArrayBuffer;

type LoadableModel = Readonly<{
  load(source: string): unknown;
  loadSafetensorsData(data: BytesSource): unknown;
}>;

type ConstructableModel = LoadableModel & {
  new(handle: unknown): unknown;
};

export type AdapterModelSourceFacadeOptions = Readonly<{
  readSafetensorsHeaderFile(modelPath: string): Uint8Array;
  probeModelPath(modelPath: string, kind: unknown): unknown;
  probeSafetensorsData(data: BytesSource, options: unknown): unknown;
  probeSafetensorsHeader(header: string | Uint8Array, options: unknown): unknown;
  loadModelPath(nativeKind: unknown, modelPath: string): unknown;
  loadSafetensorsDataHandle(nativeKind: unknown, data: BytesSource): unknown;
  tinyLlama2LayerKind: unknown;
  smollm2_360mKind: unknown;
  LlamaModel: ConstructableModel;
  TinyLlamaModel: LoadableModel;
  SmolLM135MModel: LoadableModel;
  SmolLM2_360MModel: ConstructableModel;
}>;

export function createAdapterModelSourceFacade(options: AdapterModelSourceFacadeOptions) {
  return createModelSourceFacadeHelpers({
    readSafetensorsHeaderFile: options.readSafetensorsHeaderFile,
    probePath: options.probeModelPath,
    probeSafetensorsData: options.probeSafetensorsData,
    probeSafetensorsHeader: options.probeSafetensorsHeader,
    loadPathByKind: {
      auto: (source) => options.LlamaModel.load(source),
      "tiny-llama": (source) => options.TinyLlamaModel.load(source),
      "tiny-llama-2layer": (source) => new options.LlamaModel(options.loadModelPath(options.tinyLlama2LayerKind, source)),
      "smollm-135m": (source) => options.SmolLM135MModel.load(source),
      "smollm2-360m": (source) => new options.SmolLM2_360MModel(options.loadModelPath(options.smollm2_360mKind, source)),
    },
    loadSafetensorsDataByKind: {
      auto: (data) => options.LlamaModel.loadSafetensorsData(data),
      "tiny-llama": (data) => options.TinyLlamaModel.loadSafetensorsData(data),
      "tiny-llama-2layer": (data) => new options.LlamaModel(options.loadSafetensorsDataHandle(options.tinyLlama2LayerKind, data)),
      "smollm-135m": (data) => options.SmolLM135MModel.loadSafetensorsData(data),
      "smollm2-360m": (data) => new options.SmolLM2_360MModel(options.loadSafetensorsDataHandle(options.smollm2_360mKind, data)),
    },
  });
}
