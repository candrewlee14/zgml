import type { SharedFrontendRuntime } from "./shared_frontend_runtime.js";
import type { TensorHostSurfaceOptions } from "../core/tensor_host_surface.js";
import type { TensorMetadataOpsOptions } from "../core/tensor_metadata.js";

type AdapterTensorInstance = {
  data: Float32Array;
  shape: readonly number[];
  length: number;
  requiresGrad: boolean;
  grad: Float32Array | null;
};
type AdapterTensorConstructOptions<TTensor extends AdapterTensorInstance = AdapterTensorInstance> =
  Readonly<Record<string, unknown> & {
    requiresGrad?: boolean;
    requires_grad?: boolean;
    prev?: readonly unknown[];
    backward?: (grad: Float32Array | null) => void;
  }>;
type AdapterTensorConstructor<TTensor extends AdapterTensorInstance = AdapterTensorInstance> =
  new(data: unknown, shape?: unknown, options?: AdapterTensorConstructOptions<TTensor>) => TTensor;

export type AdapterTensorRuntimeSurfaceOptions<
  TTensor extends AdapterTensorInstance = AdapterTensorInstance,
  TNativeBuffer = unknown,
> = Readonly<{
  sharedFrontend: SharedFrontendRuntime;
  Tensor: AdapterTensorConstructor<TTensor>;
  isGradEnabled: TensorHostSurfaceOptions<TTensor, TNativeBuffer>["isGradEnabled"];
  meanSquaredError: (tensor: TTensor, target: unknown) => TTensor;
  dtype: TensorMetadataOpsOptions<TTensor>["dtype"];
  device: TensorMetadataOpsOptions<TTensor>["device"];
  rowMajorStrides: TensorMetadataOpsOptions<TTensor>["rowMajorStrides"];
  normalizeDim: TensorMetadataOpsOptions<TTensor>["normalizeDim"];
  isNativeBuffer: TensorHostSurfaceOptions<TTensor, TNativeBuffer>["isNativeBuffer"];
  nativeBufferFromFloat32: TensorHostSurfaceOptions<TTensor, TNativeBuffer>["nativeBufferFromFloat32"];
}>;

export function createAdapterTensorRuntimeSurface<
  TTensor extends AdapterTensorInstance = AdapterTensorInstance,
  TNativeBuffer = unknown,
>(
  options: AdapterTensorRuntimeSurfaceOptions<TTensor, TNativeBuffer>,
) {
  const getTensorClass = () => options.Tensor;
  const tensorDataHelpers = options.sharedFrontend.createTensorDataHelpers({ getTensorClass });
  const {
    rawF32,
    prepareF32,
    f32,
    byteView,
    addTensorGrad,
    scalarTensor,
    requirePositiveInteger,
    zerosF32,
    f32WithLength,
    defaultedF32,
  } = tensorDataHelpers;
  const tensorCoreHelpers = options.sharedFrontend.createTensorCoreHelpers({
    getTensorClass,
    f32WithLength,
    prepareF32,
    addTensorGrad,
  });
  const tensorMathHelpers = options.sharedFrontend.createTensorMathHelpers({
    getTensorClass,
    f32,
    addTensorGrad,
    scalarTensor,
    isGradEnabled: options.isGradEnabled,
  });
  const tensorMathSurfaceHelpers = options.sharedFrontend.createTensorMathSurfaceHelpers({
    tensorMathHelpers,
    meanSquaredError: options.meanSquaredError,
  });
  const sessionTensorHelpers = options.sharedFrontend.createSessionTensorHelpers({ getTensorClass });
  const tensorMetadataHelpers = options.sharedFrontend.createTensorMetadataOps({
    dtype: options.dtype,
    device: options.device,
    numel: (tensor: unknown) => tensorCoreHelpers.numel(tensor as TTensor),
    rowMajorStrides: options.rowMajorStrides,
    normalizeDim: options.normalizeDim,
  });
  const tensorGradStateHelpers = options.sharedFrontend.createTensorGradStateHelpers();
  const tensorHostSurface = options.sharedFrontend.createTensorHostSurface({
    Tensor: options.Tensor,
    addTensorGrad,
    isGradEnabled: options.isGradEnabled,
    tensorCoreHelpers,
    tensorMetadataHelpers,
    tensorGradStateHelpers,
    rawF32,
    prepareF32,
    isTensor: (value: unknown): value is TTensor => value instanceof options.Tensor,
    isNativeBuffer: options.isNativeBuffer,
    nativeBufferFromFloat32: options.nativeBufferFromFloat32,
    createTensorFactoryHelpers: options.sharedFrontend.createTensorFactoryHelpers,
    createTensorViewHelpers: options.sharedFrontend.createTensorViewHelpers,
    createTensorViewSurfaceHelpers: options.sharedFrontend.createTensorViewSurfaceHelpers,
    createTensorPlacementHelpers: options.sharedFrontend.createTensorPlacementHelpers,
    createTensorInfoSurfaceHelpers: options.sharedFrontend.createTensorInfoSurfaceHelpers,
    createTensorJoinHelpers: options.sharedFrontend.createTensorJoinHelpers,
    createTensorIndexHelpers: options.sharedFrontend.createTensorIndexHelpers,
    createTensorIndexSurfaceHelpers: options.sharedFrontend.createTensorIndexSurfaceHelpers,
    createTensorFacadeHelpers: options.sharedFrontend.createTensorFacadeHelpers,
    createTensorNativeSurfaceHelpers: options.sharedFrontend.createTensorNativeSurfaceHelpers,
    createTensorStaticSurfaceFromFacade: options.sharedFrontend.createTensorStaticSurfaceFromFacade,
  });
  return {
    ...tensorDataHelpers,
    tensorCoreHelpers,
    tensorMathHelpers,
    tensorMathSurfaceHelpers,
    sessionTensorHelpers,
    tensorMetadataHelpers,
    tensorGradStateHelpers,
    tensorHostSurface,
  };
}
