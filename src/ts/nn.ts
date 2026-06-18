"use strict";

import { tsProductManifestPolicy } from "./internal/product_manifest.js";

export type {
  ActivationModule,
  AvgPool2dModule,
  CompilableActivationModule,
  CustomModule,
  DropoutModule,
  Conv2dForwardShape,
  Conv2dModule,
  EmbeddingForwardShape,
  EmbeddingModule,
  FeatureNormModule,
  KaimingInitOptions,
  KaimingNormalOptions,
  KaimingUniformOptions,
  LinearForwardShape,
  LinearModule,
  LoadStateDictOptions,
  LogSoftmaxModule,
  ModuleBase,
  ModuleBufferTraversalOptions,
  ModuleDict,
  ModuleForwardShape,
  ModuleList,
  ModuleParameterInfo,
  ModuleParameterTraversalOptions,
  ModuleStateDict,
  ModuleStateDictOptions,
  ModuleStateEntry,
  ModuleStateSnapshot,
  ModuleStateSnapshotEntry,
  ModuleStateValue,
  ModuleTargetForwardShape,
  ModuleTraversalEntry,
  NnBuffer,
  NnBufferOptions,
  NnCompilableModule,
  NnAvgPool2dConfig,
  NnConv2dConfig,
  NnDropoutConfig,
  NnEmbeddingConfig,
  NnFunctionalNamespace,
  NnInitNamespace,
  NnInitTarget,
  NnLinearConfig,
  NnLossConstructorName,
  NnLossConstructors,
  NnModule,
  NnModuleConfig,
  MaxPool2dForwardShape,
  MaxPool2dModule,
  NnMaxPool2dConfig,
  NnNamespace,
  NnNormConfig,
  NnParameter,
  NnParameterOptions,
  ParameterDict,
  ParameterList,
  PublicNnNamespace,
  ReductionModule,
  ShapeModule,
  ShapeModuleForwardShape,
  ShapeModuleKind,
  SequentialForwardShape,
  SequentialModule,
  SoftmaxModule,
  ZeroGradOptions,
} from "./public_api.js";

export {
  acceptsModuleBindingPlan,
  acceptsModuleCompilePlan,
  assert_module_binding_plan,
  assert_module_compile_plan,
  assertModuleBindingPlan,
  assertModuleCompilePlan,
  matchesModuleCompilePlanSignature,
  matchesModuleBindingPlanSignature,
  requireModuleBindingPlan,
  requireModuleCompilePlan,
} from "./runtime/execution_plan.js";
export {
  createConv2dModuleClass,
} from "./nn/conv_module.js";
export {
  createAvgPool2dModuleClass,
  createMaxPool2dModuleClass,
} from "./nn/pooling_module.js";
export {
  createEmbeddingModuleClass,
} from "./nn/embedding_module.js";
export {
  createFeatureNormModuleClass,
} from "./nn/feature_norm_module.js";
export {
  createLinearModuleClass,
} from "./nn/linear_module.js";
export {
  createNnNamespace,
} from "./nn/namespace.js";
export {
  createActivationModuleClass,
  createDropoutModuleClass,
  createReductionModuleClass,
  createSoftmaxModuleClass,
} from "./nn/parameterless_modules.js";
export {
  createSequentialModuleClass,
} from "./nn/sequential_module.js";
export {
  createShapeModuleClass,
} from "./nn/shape_module.js";

export const nnManifest = Object.freeze({
  kind: "zgml-nn",
  ...tsProductManifestPolicy("src/ts/nn.ts"),
  factoryModules: Object.freeze([
    "nn/embedding_module",
    "nn/feature_norm_module",
    "nn/conv_module",
    "nn/pooling_module",
    "nn/linear_module",
    "nn/namespace",
    "nn/parameterless_modules",
    "nn/sequential_module",
    "nn/shape_module",
  ]),
});
