import { frontendManifest } from "../frontend_manifest.js";

type NativeCoreFeatureMap = Readonly<Record<string, boolean>>;
type NativeCoreRuntimeInfo = Readonly<{
  abiVersion: number;
  featureFlags: bigint;
  features: NativeCoreFeatureMap;
}>;

export type NativeCoreHost = "node" | "bun";

export type NativeCoreEvidence = Readonly<{
  kind: "zgml.native-core";
  host: NativeCoreHost;
  productApi: "typescript";
  nativeCore: "zig-c-abi";
  boundary: Readonly<{
    userApi: "typescript";
    tensorRuntime: "zig";
    ffi: "c-abi";
    moduleCompiler: typeof frontendManifest.moduleCompilerCore;
    compileEvidence: typeof frontendManifest.nativeCompileEvidence;
    hotPath: "program-session";
    training: "zig-ffi-kernels";
  }>;
  productSourceOfTruth: typeof frontendManifest.productSourceOfTruth;
  nativeProductPolicy: typeof frontendManifest.nativeProductPolicy;
  runtimePath: "JS/TS API -> Zig C ABI -> Program/Session kernels";
  abiVersion: number;
  featureFlags: bigint;
  domains: Readonly<{
    tensorStorage: boolean;
    eagerKernels: boolean;
    programSession: boolean;
    trainingStep: boolean;
    llmSession: boolean;
    modelSource: boolean;
    webgpuInterop: boolean;
  }>;
  eagerOps: Readonly<{
    linear: boolean;
    linearActivation: boolean;
    matmul: boolean;
    activation: boolean;
    elementwise: boolean;
    reduce: boolean;
    dot: boolean;
    softmax: boolean;
    conv2d: boolean;
    pool2d: boolean;
  }>;
  unsupportedHotPathPolicy: typeof frontendManifest.unsupportedHotPathPolicy;
  signature: string;
}>;

export type NativeCoreSurfaceOptions = Readonly<{
  host: NativeCoreHost;
  runtimeInfo: () => NativeCoreRuntimeInfo;
}>;

function feature(features: NativeCoreFeatureMap, name: string): boolean {
  return features[name] === true;
}

export function createAdapterNativeCoreSurface(options: NativeCoreSurfaceOptions) {
  function nativeCore(): NativeCoreEvidence {
    const info = options.runtimeInfo();
    const features = info.features ?? {};
    const eagerOps = Object.freeze({
      linear: feature(features, "nativeEagerLinear"),
      linearActivation: feature(features, "nativeEagerLinearActivation"),
      matmul: feature(features, "nativeEagerMatmul"),
      activation: feature(features, "nativeEagerActivation"),
      elementwise: feature(features, "nativeEagerElementwise"),
      reduce: feature(features, "nativeEagerReduce"),
      dot: feature(features, "nativeEagerDot"),
      softmax: feature(features, "nativeEagerSoftmax"),
      conv2d: feature(features, "nativeEagerConv2d"),
      pool2d: feature(features, "nativeEagerPool2d"),
    });
    const domains = Object.freeze({
      tensorStorage: feature(features, "bufferHandle") && feature(features, "nativeBufferIo"),
      eagerKernels: Object.values(eagerOps).some(Boolean),
      programSession: feature(features, "nativeModuleProgram") && feature(features, "programRequirements"),
      trainingStep: feature(features, "nativeTrainingStep"),
      llmSession: feature(features, "sessionModelBinding") && feature(features, "llamaKvCacheRequirements"),
      modelSource: feature(features, "safetensorsDataLoad") || feature(features, "supportedCheckpoints"),
      webgpuInterop: feature(features, "programDeviceBufferImport") || feature(features, "nativeWgpuExecution"),
    });
    return Object.freeze({
      kind: "zgml.native-core",
      host: options.host,
      productApi: "typescript",
      nativeCore: "zig-c-abi",
      boundary: Object.freeze({
        userApi: "typescript",
        tensorRuntime: "zig",
        ffi: "c-abi",
        moduleCompiler: frontendManifest.moduleCompilerCore,
        compileEvidence: frontendManifest.nativeCompileEvidence,
        hotPath: "program-session",
        training: "zig-ffi-kernels",
      }),
      productSourceOfTruth: frontendManifest.productSourceOfTruth,
      nativeProductPolicy: frontendManifest.nativeProductPolicy,
      runtimePath: "JS/TS API -> Zig C ABI -> Program/Session kernels",
      abiVersion: info.abiVersion,
      featureFlags: info.featureFlags,
      domains,
      eagerOps,
      unsupportedHotPathPolicy: frontendManifest.unsupportedHotPathPolicy,
      signature: `${options.host}:ts-api:zig-c-abi:abi${info.abiVersion}`,
    });
  }

  return Object.freeze({
    nativeCore,
    native_core: nativeCore,
  });
}
