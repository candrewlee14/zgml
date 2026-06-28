import {
  createTensorRootOps,
} from "./tensor_root.js";
import type {
  createTensorDataHelpers,
} from "./tensor_data.js";
import type {
  createTensorCoreHelpers,
  createTensorInfoSurfaceHelpers,
} from "./tensor_core.js";
import type {
  createTensorFactoryHelpers,
} from "./tensor_factory.js";
import type {
  createTensorFacadeHelpers,
  createTensorNativeSurfaceHelpers,
} from "./tensor_facade.js";
import type {
  createTensorViewHelpers,
  createTensorViewSurfaceHelpers,
} from "./tensor_view.js";
import type {
  createTensorJoinHelpers,
} from "./tensor_join.js";
import type {
  createTensorIndexHelpers,
  createTensorIndexSurfaceHelpers,
} from "./tensor_index.js";
import type {
  createTensorPlacementHelpers,
} from "../runtime/tensor_placement.js";
import type {
  createTensorStaticSurfaceFromFacade,
} from "./tensor_static_surface.js";
import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";
type AnyRecord = Record<string, any>;

type TensorHostGradStateTarget = {
  readonly length: number;
  requiresGrad: boolean;
  requires_grad?: boolean;
  grad: Float32Array | null;
};

type TensorHostValueTarget = TensorHostGradStateTarget & {
  data: Float32Array;
  shape: readonly number[];
};

export type TensorHostConstructOptions<TTensor extends TensorHostValueTarget = TensorHostValueTarget> =
  Readonly<Record<string, unknown> & {
    requiresGrad?: boolean;
    requires_grad?: boolean;
    prev?: readonly unknown[];
    backward?: (grad: Float32Array | null) => void;
  }>;

export type TensorHostConstructor<TTensor extends TensorHostValueTarget> = new(
  data: unknown,
  shape?: unknown,
  options?: TensorHostConstructOptions<TTensor>,
) => TTensor;

export type TensorHostGradStateHelpers<TTensor> = Readonly<{
  requiresGrad(tensor: TTensor): boolean;
  setRequiresGrad(tensor: TTensor, value: unknown): void;
  requiresGrad_(tensor: TTensor, value?: boolean): TTensor;
  detach_(tensor: TTensor): TTensor;
}>;

export type TensorHostPreparedF32 = Readonly<{
  data: Float32Array;
  shape: readonly number[];
}>;

export type TensorHostSurfaceOptions<TTensor extends TensorHostValueTarget, TNativeBuffer = unknown> = Readonly<{
  readonly Tensor: TensorHostConstructor<TTensor>;
  readonly addTensorGrad: (tensor: TTensor, grad: Float32Array) => void;
  readonly isGradEnabled: () => boolean;
  readonly tensorCoreHelpers: ReturnType<typeof createTensorCoreHelpers>;
  readonly tensorMetadataHelpers: AnyRecord;
  readonly tensorGradStateHelpers: TensorHostGradStateHelpers<TensorHostGradStateTarget>;
  readonly rawF32: (data: unknown) => Float32Array;
  readonly prepareF32: (data: unknown) => TensorHostPreparedF32;
  readonly isTensor: (value: unknown) => value is TTensor;
  readonly isNativeBuffer: (value: unknown) => value is TNativeBuffer;
  readonly nativeBufferFromFloat32: (data: Float32Array) => TNativeBuffer;
  readonly nativeFullF32?: (output: Float32Array, value: number) => void;
  readonly nativeArangeF32?: (output: Float32Array, start: number, step: number) => void;
  readonly createTensorFactoryHelpers: typeof createTensorFactoryHelpers;
  readonly createTensorViewHelpers: typeof createTensorViewHelpers;
  readonly createTensorViewSurfaceHelpers: typeof createTensorViewSurfaceHelpers;
  readonly createTensorPlacementHelpers: typeof createTensorPlacementHelpers;
  readonly createTensorInfoSurfaceHelpers: typeof createTensorInfoSurfaceHelpers;
  readonly createTensorJoinHelpers: typeof createTensorJoinHelpers;
  readonly createTensorIndexHelpers: typeof createTensorIndexHelpers;
  readonly createTensorIndexSurfaceHelpers: typeof createTensorIndexSurfaceHelpers;
  readonly createTensorFacadeHelpers: typeof createTensorFacadeHelpers;
  readonly createTensorNativeSurfaceHelpers: typeof createTensorNativeSurfaceHelpers;
  readonly createTensorStaticSurfaceFromFacade: typeof createTensorStaticSurfaceFromFacade;
}>;

export type TensorHostSurface<TTensor extends TensorHostValueTarget> = Readonly<{
  tensorFactoryHelpers: ReturnType<typeof createTensorFactoryHelpers>;
  tensorViewHelpers: ReturnType<typeof createTensorViewHelpers>;
  tensorViewSurfaceHelpers: ReturnType<typeof createTensorViewSurfaceHelpers<TTensor>>;
  tensorPlacementHelpers: ReturnType<typeof createTensorPlacementHelpers>;
  tensorInfoSurfaceHelpers: ReturnType<typeof createTensorInfoSurfaceHelpers<TTensor>>;
  tensorJoinHelpers: ReturnType<typeof createTensorJoinHelpers>;
  tensorIndexHelpers: ReturnType<typeof createTensorIndexHelpers>;
  tensorIndexSurfaceHelpers: ReturnType<typeof createTensorIndexSurfaceHelpers<TTensor>>;
  tensorFacade: ReturnType<typeof createTensorFacadeHelpers>;
  tensorNativeSurfaceHelpers: ReturnType<typeof createTensorNativeSurfaceHelpers<TTensor>>;
  tensorRootOps: ReturnType<typeof createTensorRootOps<TTensor>>;
  tensorStaticHelpers: ReturnType<typeof createTensorStaticSurfaceFromFacade<TTensor>>;
}>;

export function createTensorHostSurface<TTensor extends TensorHostValueTarget, TNativeBuffer = unknown>(
  options: TensorHostSurfaceOptions<TTensor, TNativeBuffer>,
): TensorHostSurface<TTensor> {
  const tensorFactoryHelpers = options.createTensorFactoryHelpers({
    Tensor: options.Tensor,
    nativeFullF32: options.nativeFullF32,
    nativeArangeF32: options.nativeArangeF32,
  });
  const tensorViewHelpers = options.createTensorViewHelpers({
    Tensor: options.Tensor,
    addTensorGrad: options.addTensorGrad,
    isGradEnabled: options.isGradEnabled,
  });
  const tensorViewSurfaceHelpers = options.createTensorViewSurfaceHelpers({ tensorViewHelpers });
  const tensorPlacementHelpers = options.createTensorPlacementHelpers({
    cloneTensor: (tensorValue: unknown) => tensorViewHelpers.clone(tensorValue as TTensor),
  });
  const tensorInfoSurfaceHelpers = options.createTensorInfoSurfaceHelpers({
    tensorCoreHelpers: options.tensorCoreHelpers,
    tensorMetadataHelpers: options.tensorMetadataHelpers,
    tensorGradStateHelpers: options.tensorGradStateHelpers,
    tensorPlacementHelpers,
  });
  const tensorJoinHelpers = options.createTensorJoinHelpers({
    Tensor: options.Tensor,
    addTensorGrad: options.addTensorGrad,
    isGradEnabled: options.isGradEnabled,
  });
  const tensorIndexHelpers = options.createTensorIndexHelpers();
  const tensorIndexSurfaceHelpers = options.createTensorIndexSurfaceHelpers({ tensorIndexHelpers });
  const tensorFacade = options.createTensorFacadeHelpers({
    Tensor: options.Tensor,
    rawF32: options.rawF32,
    prepareF32: options.prepareF32,
    isTensor: options.isTensor,
    isNativeBuffer: options.isNativeBuffer,
    nativeBufferFromFloat32: options.nativeBufferFromFloat32,
    tensorFactory: tensorFactoryHelpers,
    tensorPlacement: tensorPlacementHelpers,
    tensorJoin: tensorJoinHelpers,
  });
  const tensorNativeSurfaceHelpers = options.createTensorNativeSurfaceHelpers({ tensorFacade });
  const tensorRootOps = createTensorRootOps({
    Tensor: options.Tensor,
    tensorFacade: tensorFacade as any,
  });
  const tensorStaticHelpers = options.createTensorStaticSurfaceFromFacade({
    tensorCoreHelpers: options.tensorCoreHelpers as unknown as Parameters<typeof createTensorStaticSurfaceFromFacade<TTensor>>[0]["tensorCoreHelpers"],
    tensorFacade: tensorFacade as unknown as Parameters<typeof createTensorStaticSurfaceFromFacade<TTensor>>[0]["tensorFacade"],
    rootOps: tensorRootOps as unknown as Parameters<typeof createTensorStaticSurfaceFromFacade<TTensor>>[0]["rootOps"],
  });

  return Object.freeze({
    tensorFactoryHelpers,
    tensorViewHelpers,
    tensorViewSurfaceHelpers,
    tensorPlacementHelpers,
    tensorInfoSurfaceHelpers,
    tensorJoinHelpers,
    tensorIndexHelpers,
    tensorIndexSurfaceHelpers,
    tensorFacade,
    tensorNativeSurfaceHelpers,
    tensorRootOps,
    tensorStaticHelpers,
  }) as unknown as TensorHostSurface<TTensor>;
}

export const tensorHostSurfaceManifest = Object.freeze({
  kind: "zgml-tensor-host-surface",
  ...tsRuntimeManifestPolicy("src/ts/core/tensor_host_surface.ts", "Tensor host facade composition"),
});
