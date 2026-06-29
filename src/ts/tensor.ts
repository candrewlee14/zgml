"use strict";

import { tsProductManifestPolicy } from "./internal/product_manifest.js";

export type {
  AllCloseOptions,
  ArangeRangeShape,
  ArangeShape,
  BroadcastShape,
  FlattenShape,
  IndexLike,
  MatmulShape,
  NarrowShape,
  OneHotShape,
  PermuteShape,
  RandomIntTensorOptions,
  RandomTensorOptions,
  RandomUniformTensorOptions,
  ReductionShape,
  ReshapeShape,
  SelectShape,
  SliceShape,
  SqueezeAllShape,
  SqueezeShape,
  Tensor,
  TensorCatShape,
  TensorData,
  TensorDevice,
  TensorDeviceLike,
  TensorDType,
  TensorDTypeLike,
  TensorFromNativeBufferOptions,
  TensorGatherShape,
  TensorInspection,
  TensorIndexSelectShape,
  TensorJSON,
  TensorLike,
  TensorLikeShape,
  TensorNativeBufferOptions,
  TensorNestedArray,
  TensorOptions,
  TensorShape,
  TensorShapeOf,
  TensorStackShape,
  TensorShapeTuple,
  TensorHStackShape,
  TensorToOptions,
  TensorToTarget,
  TensorVStackShape,
  TransposeShape,
  UnsqueezeShape,
  WhereShape,
} from "./public_api.js";

export {
  createTensorCoreHelpers,
} from "./core/tensor_core.js";
export {
  createTensorDataHelpers,
} from "./core/tensor_data.js";
export {
  createTensorFacadeHelpers,
  normalizeShapeModuleShape,
} from "./core/tensor_facade.js";
export {
  createTensorFactoryHelpers,
  initialSeed,
  initial_seed,
  manual_seed,
  manualSeed,
  seededRng,
} from "./core/tensor_factory.js";
export {
  createTensorIndexHelpers,
} from "./core/tensor_index.js";
export {
  createTensorJoinHelpers,
} from "./core/tensor_join.js";
export {
  createTensorMathHelpers,
} from "./core/tensor_math.js";
export {
  createTensorViewHelpers,
} from "./core/tensor_view.js";

export const tensorManifest = Object.freeze({
  kind: "zgml-tensor",
  ...tsProductManifestPolicy("src/ts/tensor.ts"),
  factoryModules: Object.freeze([
    "core/tensor_core",
    "core/tensor_data",
    "core/tensor_facade",
    "core/tensor_factory",
    "core/tensor_index",
    "core/tensor_join",
    "core/tensor_math",
    "core/tensor_view",
  ]),
});
