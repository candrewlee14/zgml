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
type NativeEagerMatmulInto = (
  output: Float32Array,
  lhs: unknown,
  rhs: unknown,
  options?: Record<string, unknown>,
) => Float32Array;
type NativeEagerBmmInto = (
  output: Float32Array,
  lhs: unknown,
  rhs: unknown,
  options?: Record<string, unknown>,
) => Float32Array;
type NativeEagerElementwiseInto = (
  output: Float32Array,
  lhs: unknown,
  rhs: unknown,
  options: Readonly<{ op: string }>,
) => Float32Array;
type NativeEagerActivationInto = (
  output: Float32Array,
  input: unknown,
  options: Readonly<{ activation: string }>,
) => Float32Array;
type NativeEagerActivationEnabled = (activation: string, outputLength: number) => boolean;
type NativeEagerWhereInto = (
  output: Float32Array,
  condition: unknown,
  input: unknown,
  other: unknown,
) => Float32Array;
type NativeEagerClampInto = (
  output: Float32Array,
  input: unknown,
  options: Readonly<{ min?: number; max?: number }>,
) => Float32Array;
type NativeEagerReduceInto = (
  output: Float32Array,
  input: unknown,
  options: Readonly<{ op: string }>,
) => Float32Array;
type NativeEagerDotInto = (
  output: Float32Array,
  lhs: unknown,
  rhs: unknown,
) => Float32Array;
type NativeEagerSoftmaxInto = (
  output: Float32Array,
  input: unknown,
  options: Readonly<{ dim: number; logSoftmax?: boolean }>,
) => Float32Array;
type NativeEagerPermuteInto = (
  output: Float32Array,
  input: unknown,
  options: Readonly<{
    outputShape: Uint32Array;
    inputStrides: Uint32Array;
    axes: Uint32Array;
  }>,
) => Float32Array;
type NativeEagerTakeInto = (
  output: Float32Array,
  input: unknown,
  index: unknown,
) => Float32Array;
type NativeEagerIndexSelectInto = (
  output: Float32Array,
  input: unknown,
  index: unknown,
  options: Readonly<{ outer: number; axisLen: number; inner: number }>,
) => Float32Array;
type NativeEagerGatherInto = (
  output: Float32Array,
  input: unknown,
  index: unknown,
  options: Readonly<{
    outputShape: Uint32Array;
    inputStrides: Uint32Array;
    axis: number;
    axisLen: number;
  }>,
) => Float32Array;

export type AdapterTensorRuntimeSurfaceOptions<
  TTensor extends AdapterTensorInstance = AdapterTensorInstance,
  TNativeBuffer = unknown,
> = Readonly<{
  sharedFrontend: SharedFrontendRuntime;
  Tensor: AdapterTensorConstructor<TTensor>;
  isGradEnabled: TensorHostSurfaceOptions<TTensor, TNativeBuffer>["isGradEnabled"];
  nativeEagerMatmulInto?: NativeEagerMatmulInto;
  nativeEagerBmmInto?: NativeEagerBmmInto;
  nativeEagerBmmMinMultiplyAdds?: number;
  nativeEagerElementwiseInto?: NativeEagerElementwiseInto;
  nativeEagerElementwiseMinLength?: number;
  nativeEagerUnaryOpEnabled?: (op: string, outputLength: number) => boolean;
  nativeEagerActivationInto?: NativeEagerActivationInto;
  nativeEagerActivationMinLength?: number;
  nativeEagerActivationEnabled?: NativeEagerActivationEnabled;
  nativeEagerWhereInto?: NativeEagerWhereInto;
  nativeEagerClampInto?: NativeEagerClampInto;
  nativeEagerReduceInto?: NativeEagerReduceInto;
  nativeEagerReduceDimInto?: (
    output: Float32Array,
    input: unknown,
    options: Readonly<{ op: string; outer: number; reduce: number; inner: number }>,
  ) => Float32Array;
  nativeEagerArgReduceDimInto?: (
    output: Float32Array,
    input: unknown,
    options: Readonly<{ op: string; outer: number; reduce: number; inner: number }>,
  ) => Float32Array;
  nativeEagerCumsumInto?: (
    output: Float32Array,
    input: unknown,
    options: Readonly<{ outer: number; axis: number; inner: number; reverse?: boolean }>,
  ) => Float32Array;
  nativeEagerMomentInto?: (
    output: Float32Array,
    input: unknown,
    options: Readonly<{ outer: number; reduce: number; inner: number; correction?: number; sqrtOutput?: boolean }>,
  ) => Float32Array;
  nativeEagerReduceMinLength?: number;
  nativeEagerDotInto?: NativeEagerDotInto;
  nativeEagerSoftmaxInto?: NativeEagerSoftmaxInto;
  nativeEagerPermuteInto?: NativeEagerPermuteInto;
  nativeEagerTakeInto?: NativeEagerTakeInto;
  nativeEagerIndexSelectInto?: NativeEagerIndexSelectInto;
  nativeEagerGatherInto?: NativeEagerGatherInto;
  nativeFullF32?: (output: Float32Array, value: number) => void;
  nativeArangeF32?: (output: Float32Array, start: number, step: number) => void;
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
    nativeEagerMatmulInto: options.nativeEagerMatmulInto,
    nativeEagerBmmInto: options.nativeEagerBmmInto,
    nativeEagerBmmMinMultiplyAdds: options.nativeEagerBmmMinMultiplyAdds,
    nativeEagerElementwiseInto: options.nativeEagerElementwiseInto,
    nativeEagerElementwiseMinLength: options.nativeEagerElementwiseMinLength,
    nativeEagerUnaryOpEnabled: options.nativeEagerUnaryOpEnabled,
    nativeEagerActivationInto: options.nativeEagerActivationInto,
    nativeEagerActivationMinLength: options.nativeEagerActivationMinLength,
    nativeEagerActivationEnabled: options.nativeEagerActivationEnabled,
    nativeEagerWhereInto: options.nativeEagerWhereInto,
    nativeEagerClampInto: options.nativeEagerClampInto,
    nativeEagerReduceInto: options.nativeEagerReduceInto,
    nativeEagerReduceDimInto: options.nativeEagerReduceDimInto,
    nativeEagerArgReduceDimInto: options.nativeEagerArgReduceDimInto,
    nativeEagerCumsumInto: options.nativeEagerCumsumInto,
    nativeEagerMomentInto: options.nativeEagerMomentInto,
    nativeEagerReduceMinLength: options.nativeEagerReduceMinLength,
    nativeEagerDotInto: options.nativeEagerDotInto,
    nativeEagerSoftmaxInto: options.nativeEagerSoftmaxInto,
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
    nativeFullF32: options.nativeFullF32,
    nativeArangeF32: options.nativeArangeF32,
    nativePermuteF32: options.nativeEagerPermuteInto,
    nativeTakeF32: options.nativeEagerTakeInto,
    nativeIndexSelectF32: options.nativeEagerIndexSelectInto,
    nativeGatherF32: options.nativeEagerGatherInto,
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
