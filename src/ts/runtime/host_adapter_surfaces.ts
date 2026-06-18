import { indexValuesFromTensorOrArray, type TensorDataConstructor, type TensorDataLike } from "../core/index_values.js";
import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";

export type HostAdapterIndexValuesTensorConstructor<TTensor extends TensorDataLike = TensorDataLike> =
  TensorDataConstructor<TTensor>;

export type HostAdapterIndexValuesSurfaceOptions<TTensor extends TensorDataLike = TensorDataLike> = Readonly<{
  Tensor: HostAdapterIndexValuesTensorConstructor<TTensor>;
}>;

export type HostAdapterIndexValuesSurface = Readonly<{
  indexValues(data: unknown, name: string): ArrayLike<number>;
}>;

export function createHostAdapterIndexValuesSurface<TTensor extends TensorDataLike = TensorDataLike>(
  options: HostAdapterIndexValuesSurfaceOptions<TTensor>,
): HostAdapterIndexValuesSurface {
  function indexValues(data: unknown, name: string): ArrayLike<number> {
    return indexValuesFromTensorOrArray(data, name, options.Tensor);
  }

  return Object.freeze({ indexValues });
}

export type HostTraceModuleCompiler = Readonly<{
  analyze(layers: unknown, options: unknown): unknown;
  analyzeSingle(layer: unknown, options: unknown): unknown;
  trace(layers: unknown, options: unknown): unknown;
  packParameters(spec: unknown): unknown;
}>;

export type HostModuleCompilerSurfaceOptions = Readonly<{
  getTraceModuleCompiler(): HostTraceModuleCompiler | null | undefined;
  uninitializedMessage: string;
}>;

export type HostModuleCompilerSurface = Readonly<{
  analyzeSequentialProgram(layers: unknown, options?: unknown): unknown;
  analyzeSingleModuleProgram(layer: unknown, options?: unknown): unknown;
  traceSequentialProgram(layers: unknown, options?: unknown): unknown;
  packedSequentialProgramParameters(spec: unknown): unknown;
}>;

export function createHostModuleCompilerSurface(options: HostModuleCompilerSurfaceOptions): HostModuleCompilerSurface {
  function traceModuleCompiler(): HostTraceModuleCompiler {
    const compiler = options.getTraceModuleCompiler();
    if (!compiler) {
      throw new Error(options.uninitializedMessage);
    }
    return compiler;
  }

  function analyzeSequentialProgram(layers: unknown, compileOptions: unknown = {}): unknown {
    return traceModuleCompiler().analyze(layers, compileOptions);
  }

  function analyzeSingleModuleProgram(layer: unknown, compileOptions: unknown = {}): unknown {
    return traceModuleCompiler().analyzeSingle(layer, compileOptions);
  }

  function traceSequentialProgram(layers: unknown, traceOptions: unknown = {}): unknown {
    return traceModuleCompiler().trace(layers, traceOptions);
  }

  function packedSequentialProgramParameters(spec: unknown): unknown {
    return traceModuleCompiler().packParameters(spec);
  }

  return Object.freeze({
    analyzeSequentialProgram,
    analyzeSingleModuleProgram,
    traceSequentialProgram,
    packedSequentialProgramParameters,
  });
}

export type HostModelSourceFacade = Readonly<{
  probeModel(source: unknown, options: unknown): unknown;
  loadModel(source: unknown, options: unknown): unknown;
  loadSafetensorsData(data: unknown, options: unknown): unknown;
}>;

export type HostModelSourceSurfaceOptions = Readonly<{
  getModelSourceFacade(): HostModelSourceFacade | null | undefined;
  uninitializedMessage: string;
}>;

export type HostModelSourceSurface = Readonly<{
  probeModel(source: unknown, options?: unknown): unknown;
  loadModel(source: unknown, options?: unknown): unknown;
  loadSafetensorsData(data: unknown, options?: unknown): unknown;
}>;

export function createHostModelSourceSurface(options: HostModelSourceSurfaceOptions): HostModelSourceSurface {
  function modelSourceFacade(): HostModelSourceFacade {
    const facade = options.getModelSourceFacade();
    if (!facade) {
      throw new Error(options.uninitializedMessage);
    }
    return facade;
  }

  function probeModel(source: unknown, loadOptions: unknown = {}): unknown {
    return modelSourceFacade().probeModel(source, loadOptions);
  }

  function loadModel(source: unknown, loadOptions: unknown = {}): unknown {
    return modelSourceFacade().loadModel(source, loadOptions);
  }

  function loadSafetensorsData(data: unknown, loadOptions: unknown = {}): unknown {
    return modelSourceFacade().loadSafetensorsData(data, loadOptions);
  }

  return Object.freeze({
    probeModel,
    loadModel,
    loadSafetensorsData,
  });
}

export type HostNativeBufferConstructor<TNativeBuffer extends object = object> = abstract new (
  ...args: any[]
) => TNativeBuffer;

export type HostNativeBufferInstanceSurfaceOptions<TNativeBuffer extends object = object> = Readonly<{
  getNativeBufferClass(): HostNativeBufferConstructor<TNativeBuffer> | null | undefined;
  uninitializedMessage: string;
}>;

export type HostNativeBufferInstanceSurface<TNativeBuffer extends object = object> = Readonly<{
  isNativeBuffer(value: unknown): value is TNativeBuffer;
}>;

export function createHostNativeBufferInstanceSurface<TNativeBuffer extends object = object>(
  options: HostNativeBufferInstanceSurfaceOptions<TNativeBuffer>,
): HostNativeBufferInstanceSurface<TNativeBuffer> {
  function nativeBufferClass(): HostNativeBufferConstructor<TNativeBuffer> {
    const NativeBuffer = options.getNativeBufferClass();
    if (!NativeBuffer) {
      throw new Error(options.uninitializedMessage);
    }
    return NativeBuffer;
  }

  function isNativeBuffer(value: unknown): value is TNativeBuffer {
    return value instanceof nativeBufferClass();
  }

  return Object.freeze({ isNativeBuffer });
}

export const hostAdapterSurfacesManifest = Object.freeze({
  kind: "zgml-host-adapter-surfaces",
  ...tsRuntimeManifestPolicy("src/ts/runtime/host_adapter_surfaces.ts", "Host adapter callback surfaces"),
});
