// @ts-nocheck
"use strict";

const shape = require(["..", "core", "shape.cjs"].join("/"));
const { createTensorCoreHelpers } = require(["..", "core", "tensor_core.cjs"].join("/"));
const { createTensorDataHelpers } = require(["..", "core", "tensor_data.cjs"].join("/"));
const { createTensorFactoryHelpers } = require(["..", "core", "tensor_factory.cjs"].join("/"));
const { createTensorMathHelpers } = require(["..", "core", "tensor_math.cjs"].join("/"));
const tensorFacadePolicy = require(["..", "core", "tensor_facade.cjs"].join("/"));
const { createTensorIndexHelpers } = require(["..", "core", "tensor_index.cjs"].join("/"));
const { createTensorJoinHelpers } = require(["..", "core", "tensor_join.cjs"].join("/"));
const { createTensorViewHelpers } = require(["..", "core", "tensor_view.cjs"].join("/"));
const indexValues = require(["..", "core", "index_values.cjs"].join("/"));
const tensorMetadata = require(["..", "core", "tensor_metadata.cjs"].join("/"));
const tensorRoot = require(["..", "core", "tensor_root.cjs"].join("/"));
const tensorHostSurface = require(["..", "core", "tensor_host_surface.cjs"].join("/"));
const activation = require(["..", "core", "activation.cjs"].join("/"));
const gradMode = require(["..", "core", "grad_mode.cjs"].join("/"));
const token = require(["..", "core", "token.cjs"].join("/"));
const compileDiagnostics = require(["..", "runtime", "compile_diagnostics.cjs"].join("/"));
const compilerSignatures = require(["..", "runtime", "compiler_signatures.cjs"].join("/"));
const moduleProgramDesc = require(["..", "runtime", "module_program_desc.cjs"].join("/"));
const moduleProgramEvidence = require(["..", "runtime", "module_program_evidence.cjs"].join("/"));
const moduleBindings = require(["..", "runtime", "module_bindings.cjs"].join("/"));
const moduleCompilerPolicy = require(["..", "runtime", "module_compiler_policy.cjs"].join("/"));
const moduleFacadePolicy = require(["..", "runtime", "module_facade.cjs"].join("/"));
const sessionTensor = require(["..", "runtime", "session_tensor.cjs"].join("/"));
const stepParams = require(["..", "runtime", "step_params.cjs"].join("/"));
const sessionFacade = require(["..", "runtime", "session_facade.cjs"].join("/"));
const sessionContract = require(["..", "runtime", "session_contract.cjs"].join("/"));
const sessionProfile = require(["..", "runtime", "session_profile.cjs"].join("/"));
const sessionParameters = require(["..", "runtime", "session_parameters.cjs"].join("/"));
const sessionLayout = require(["..", "runtime", "session_layout.cjs"].join("/"));
const sessionBinding = require(["..", "runtime", "session_binding.cjs"].join("/"));
const sessionLifecycle = require(["..", "runtime", "session_lifecycle.cjs"].join("/"));
const sessionValues = require(["..", "runtime", "session_values.cjs"].join("/"));
const sessionStepIo = require(["..", "runtime", "session_step_io.cjs"].join("/"));
const executionPlan = require(["..", "runtime", "execution_plan.cjs"].join("/"));
const programBuffers = require(["..", "runtime", "program_buffers.cjs"].join("/"));
const programBufferFactory = require(["..", "runtime", "program_buffer_factory.cjs"].join("/"));
const programFacadePolicy = require(["..", "runtime", "program_facade_policy.cjs"].join("/"));
const programModuleBinding = require(["..", "runtime", "program_module_binding.cjs"].join("/"));
const programDevicePolicy = require(["..", "runtime", "program_device.cjs"].join("/"));
const programSizing = require(["..", "runtime", "program_sizing.cjs"].join("/"));
const programParameters = require(["..", "runtime", "program_parameters.cjs"].join("/"));
const programShapes = require(["..", "runtime", "program_shapes.cjs"].join("/"));
const programResources = require(["..", "runtime", "program_resources.cjs"].join("/"));
const programLayoutAccess = require(["..", "runtime", "program_layout.cjs"].join("/"));
const programPolicyAccess = require(["..", "runtime", "program_policy_accessors.cjs"].join("/"));
const programFacade = require(["..", "runtime", "program_facade.cjs"].join("/"));
const moduleCompatibility = require(["..", "runtime", "module_compatibility.cjs"].join("/"));
const abi = require(["..", "runtime", "abi.cjs"].join("/"));
const inspection = require(["..", "runtime", "inspection.cjs"].join("/"));
const modelSource = require(["..", "runtime", "model_source.cjs"].join("/"));
const tensorProgramIr = require(["..", "runtime", "tensor_program_ir.cjs"].join("/"));
const kernelPlanPolicy = require(["..", "runtime", "kernel_plan.cjs"].join("/"));
const traceCompiler = require(["..", "runtime", "trace_compiler.cjs"].join("/"));
const compileNamespace = require(["..", "compile.cjs"].join("/"));
const lazyNamespace = require(["..", "lazy.cjs"].join("/"));
const inspectionNamespace = require(["..", "inspection.cjs"].join("/"));
const sessionNamespace = require(["..", "session.cjs"].join("/"));
const tensorPlacement = require(["..", "runtime", "tensor_placement.cjs"].join("/"));
const nativeBufferPolicy = require(["..", "runtime", "native_buffer.cjs"].join("/"));
const nativeAbiConstants = require(["..", "runtime", "native_abi_constants.cjs"].join("/"));
const nativeApiContract = require(["..", "runtime", "native_api_contract.cjs"].join("/"));
const nativeStatus = require(["..", "runtime", "native_status.cjs"].join("/"));
const nativeKernelContract = require(["..", "runtime", "native_kernel_contract.cjs"].join("/"));
const sessionFacadeComposition = require(["..", "runtime", "session_facade_composition.cjs"].join("/"));
const dataNamespace = require(["..", "data.cjs"].join("/"));
const nnShapeModule = require(["..", "nn", "shape_module.cjs"].join("/"));
const nnParameterlessModules = require(["..", "nn", "parameterless_modules.cjs"].join("/"));
const nnLinearModule = require(["..", "nn", "linear_module.cjs"].join("/"));
const nnEmbeddingModule = require(["..", "nn", "embedding_module.cjs"].join("/"));
const nnConvModule = require(["..", "nn", "conv_module.cjs"].join("/"));
const nnPoolingModule = require(["..", "nn", "pooling_module.cjs"].join("/"));
const nnSequentialModule = require(["..", "nn", "sequential_module.cjs"].join("/"));
const nnFeatureNormModule = require(["..", "nn", "feature_norm_module.cjs"].join("/"));
const nnModule = require(["..", "nn.cjs"].join("/"));
const nnNamespace = require(["..", "nn", "namespace.cjs"].join("/"));
const moduleMode = require(["..", "train", "module_mode.cjs"].join("/"));
const moduleState = require(["..", "train", "module_state.cjs"].join("/"));
const training = require(["..", "train", "training.cjs"].join("/"));
const trainNamespacePolicy = require(["..", "train.cjs"].join("/"));
const { createTensorRootOps } = tensorRoot;
const { createTensorMetadataOps } = tensorMetadata;
const genericFamilySurfacePolicy = require(["..", "runtime", "generic_family_surface.cjs"].join("/"));
const { createGenericFamilySurface } = genericFamilySurfacePolicy;
const llamaFamilySurfacePolicy = require(["..", "runtime", "llama_family_surface.cjs"].join("/"));
const genericModelDesc = require(["..", "runtime", "generic_model_desc.cjs"].join("/"));
const programBindDesc = require(["..", "runtime", "program_bind_desc.cjs"].join("/"));
const programBindingSurface = require(["..", "runtime", "program_binding_surface.cjs"].join("/"));
const programBufferDesc = require(["..", "runtime", "program_buffer_desc.cjs"].join("/"));
const modelSourceDesc = require(["..", "runtime", "model_source_desc.cjs"].join("/"));
const llamaKvCache = require(["..", "runtime", "llama_kv_cache.cjs"].join("/"));
const llamaSessionBindDesc = require(["..", "runtime", "llama_session_bind_desc.cjs"].join("/"));
const llamaTokenOutput = require(["..", "runtime", "llama_token_output.cjs"].join("/"));
const hostAdapterSurfaces = require(["..", "runtime", "host_adapter_surfaces.cjs"].join("/"));
const {
  createHostAdapterIndexValuesSurface,
  createHostModelSourceSurface,
  createHostModuleCompilerSurface,
  createHostNativeBufferInstanceSurface,
} = hostAdapterSurfaces;
const { createNodeLlamaKvCacheClass } = require(["..", "adapters", "node_llama_kv_cache_surface.cjs"].join("/"));
const { createBunLlamaKvCacheClass } = require(["..", "adapters", "bun_llama_kv_cache_surface.cjs"].join("/"));
const { createAdapterNativeBufferInstanceSurface } = require(["..", "adapters", "native_buffer_instance_surface.cjs"].join("/"));
const { createAdapterProgramBufferFactorySurface } = require(["..", "adapters", "program_buffer_factory_surface.cjs"].join("/"));
const { createAdapterSafetensorsFileHeaderHelpers } = require(["..", "adapters", "safetensors_file_header.cjs"].join("/"));
const { createAdapterModelSourceFacade } = require(["..", "adapters", "model_source_facade.cjs"].join("/"));
const { createAdapterModelSourceSurface } = require(["..", "adapters", "model_source_surface.cjs"].join("/"));
const { createAdapterGenericFamilyModelHandleSurface } = require(["..", "adapters", "generic_family_model_handle_surface.cjs"].join("/"));
const { createAdapterIndexValuesSurface } = require(["..", "adapters", "index_values_surface.cjs"].join("/"));
const { createAdapterModuleCompilerSurface } = require(["..", "adapters", "module_compiler_surface.cjs"].join("/"));
const { createTensorGradStateHelpers } = require(["..", "core", "tensor_grad_state.cjs"].join("/"));
const llamaProgramInspection = require(["..", "runtime", "llama_program_inspection.cjs"].join("/"));
const concreteRuntimeLoader = require(["..", "adapters", "concrete_runtime_loader.cjs"].join("/"));
const runtimeLoadCandidates = require(["..", "runtime", "runtime_load_candidates.cjs"].join("/"));
const nodeConcreteRuntime = require(["..", "adapters", "node_concrete_runtime.cjs"].join("/"));
const bunConcreteRuntime = require(["..", "adapters", "bun_concrete_runtime.cjs"].join("/"));
const sharedFrontendRuntime = require(["..", "adapters", "shared_frontend_runtime.cjs"].join("/"));
const sharedFrontend = require(["..", "shared_frontend.cjs"].join("/"));
const tsFrontend = require(["..", "index.cjs"].join("/"));
const { expectFrontendManifest: expectTsFrontendManifest } = require([".", "smoke_contracts.cjs"].join("/"));
const tsNode = require(["..", "node.cjs"].join("/"));
const tsBun = require(["..", "bun.cjs"].join("/"));

function normalize(value) {
  if (typeof value === "bigint") return value.toString();
  if (ArrayBuffer.isView(value)) return Array.from(value).map(normalize);
  if (Array.isArray(value)) return value.map(normalize);
  if (value && typeof value === "object") {
    const out = {};
    for (const key of Object.keys(value)) out[key] = normalize(value[key]);
    return out;
  }
  return value;
}

function expectSame(actual, expected, label) {
  const actualJson = JSON.stringify(normalize(actual));
  const expectedJson = JSON.stringify(normalize(expected));
  if (actualJson !== expectedJson) {
    throw new Error(`${label} mismatch:\nactual   ${actualJson}\nexpected ${expectedJson}`);
  }
}

function expectApprox(actual, expected, tolerance, label) {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${label} mismatch:\nactual   ${actual}\nexpected ${expected} +/- ${tolerance}`);
  }
}

function expectThrow(fn, expectedMessage, label) {
  try {
    fn();
  } catch (error) {
    if ((error && error.message) === expectedMessage) return;
    throw new Error(`${label} threw wrong error:\nactual   ${error && error.message}\nexpected ${expectedMessage}`);
  }
  throw new Error(`${label} did not throw`);
}

const expectedExports = [
  "broadcastPlan",
  "createShapedF32Helpers",
  "dimReductionPlan",
  "hasNestedArrayShape",
  "inferredOperandShape",
  "isFlatShape",
  "normalizeDim",
  "normalizeFactoryShape",
  "normalizeInsertDim",
  "normalizeViewShape",
  "rowMajorStrides",
  "sameShape",
  "shapeProduct",
  "shapeScalarCount",
  "traceIndexForDim",
  "traceSliceBound",
  "validateTensorShape",
];

for (const name of expectedExports) {
  if (typeof shape[name] !== "function") throw new Error(`compiled TS shape helper is missing ${name}`);
}

expectSame(shape.shapeScalarCount([2, 3, 4]), 24, "shapeScalarCount");
expectSame(shape.sameShape([2, 3], [2, 3]), true, "sameShape true");
expectSame(shape.sameShape([2, 3], [3, 2]), false, "sameShape false");
expectSame(shape.isFlatShape([6], 6), true, "isFlatShape");
expectSame(shape.hasNestedArrayShape([[1], [2]]), true, "hasNestedArrayShape");
expectSame(shape.validateTensorShape([2, 3], 6, "x"), [2, 3], "validateTensorShape");
expectSame(Object.isFrozen(shape.validateTensorShape([2, 3], 6, "x")), true, "validateTensorShape frozen");
expectSame(shape.shapeProduct([2, 3, 4]), 24, "shapeProduct");
expectSame(shape.normalizeFactoryShape([2, 3], "x"), { shape: [2, 3], length: 6 }, "normalizeFactoryShape array");
expectSame(shape.normalizeFactoryShape(4, "x"), { shape: [4], length: 4 }, "normalizeFactoryShape scalar");
expectSame(shape.normalizeViewShape([2, -1], 6, "view"), [2, 3], "normalizeViewShape");
expectSame(shape.rowMajorStrides([2, 3, 4]), [12, 4, 1], "rowMajorStrides");
expectSame(shape.inferredOperandShape(new Float32Array(3), [2, 3]), [3], "inferredOperandShape trailing dim");
expectSame(shape.inferredOperandShape(new Float32Array(6), [2, 3]), [2, 3], "inferredOperandShape full shape");
expectSame(shape.broadcastPlan([2, 3], [3], "add"), {
  shape: [2, 3],
  lhsIndex: [0, 1, 2, 3, 4, 5],
  rhsIndex: [0, 1, 2, 0, 1, 2],
}, "broadcastPlan");
expectSame(shape.normalizeDim(-1, 3, "x"), 2, "normalizeDim");
expectSame(shape.normalizeInsertDim(-1, 3, "x"), 3, "normalizeInsertDim");
expectSame(shape.dimReductionPlan([2, 3], 1, "sum"), {
  axis: 1,
  shape: [2, 1],
  inToOut: [0, 0, 0, 1, 1, 1],
  reduceLen: 3,
}, "dimReductionPlan");
expectSame(shape.traceIndexForDim(-1, 4, "x"), 3, "traceIndexForDim");
expectSame(shape.traceSliceBound(-2, 5, 0, "x"), 3, "traceSliceBound");

expectThrow(
  () => shape.validateTensorShape([2, 0], 0, "bad"),
  "bad dimensions must be positive safe integers, got 0",
  "validateTensorShape bad dim",
);
expectThrow(
  () => shape.broadcastPlan([2, 3], [4], "add"),
  "tensor add shape mismatch: [2,3] cannot broadcast with [4]",
  "broadcastPlan mismatch",
);
expectThrow(
  () => shape.normalizeViewShape([2, -1, -1], 8, "view"),
  "view shape can infer at most one dimension",
  "normalizeViewShape double infer",
);

const f32 = (value) => new Float32Array(Array.isArray(value) ? value.flat(Infinity) : [value]);
const prepareF32 = (value) => ({
  data: f32(value),
  shape: Array.isArray(value) && Array.isArray(value[0]) ? [value.length, value[0].length] : [f32(value).length],
});
const helpers = shape.createShapedF32Helpers({ f32, prepareF32 });
const prepared = helpers.prepareHostValue([[1, 2], [3, 4]]);
expectSame(prepared, { data: [1, 2, 3, 4], shape: [2, 2], shapeEvidence: true }, "prepareHostValue");
expectSame(helpers.validateHostValueShape(prepared, [2, 2], "value"), undefined, "validateHostValueShape");
expectThrow(
  () => helpers.validateHostValueShape(prepared, [4, 1], "value"),
  "value shape must be [4, 1] or flat length 4, got [2, 2]",
  "validateHostValueShape mismatch",
);

class TensorDataSmokeTensor {
  constructor(data, shape, options = {}) {
    this.data = data instanceof Float32Array ? data : Float32Array.from(data);
    this.shape = shape.slice();
    this.length = this.data.length;
    this.rank = this.shape.length;
    this.requiresGrad = Boolean(options.requiresGrad);
    this.grad = options.grad ?? (this.requiresGrad ? new Float32Array(this.length) : null);
    this._prev = options.prev ?? [];
    this._backward = options.backward ?? (() => {});
  }

  select(dim, index) {
    if (dim !== 0) throw new Error("TensorDataSmokeTensor.select only supports dim 0");
    const rowLength = this.shape.slice(1).reduce((acc, value) => acc * value, 1);
    const start = index * rowLength;
    return new TensorDataSmokeTensor(this.data.slice(start, start + rowLength), this.shape.slice(1));
  }
}

const tensorData = createTensorDataHelpers({ Tensor: TensorDataSmokeTensor });
const preparedTensorData = tensorData.prepareF32([[1, 2], [3, 4]]);
expectSame(preparedTensorData, { data: [1, 2, 3, 4], shape: [2, 2] }, "tensor data prepareF32 nested");
expectSame(tensorData.rawF32([1, 2, 3]), [1, 2, 3], "tensor data rawF32");
const existingTensor = new TensorDataSmokeTensor(Float32Array.of(5, 6), [2]);
expectSame(tensorData.f32(existingTensor), [5, 6], "tensor data f32 Tensor");
expectSame(tensorData.byteView(new Uint16Array([0x1234, 0x5678])), [0x34, 0x12, 0x78, 0x56], "tensor data byteView");
const gradTensor = new TensorDataSmokeTensor(Float32Array.of(1, 2), [2], { requiresGrad: true });
tensorData.addTensorGrad(gradTensor, Float32Array.of(3, 4));
tensorData.addTensorGrad(gradTensor, Float32Array.of(1, 2));
expectSame(gradTensor.grad, [4, 6], "tensor data addTensorGrad");
const scalar = tensorData.scalarTensor(7, true);
expectSame({ data: scalar.data, shape: scalar.shape, requiresGrad: scalar.requiresGrad }, { data: [7], shape: [1], requiresGrad: true }, "tensor data scalarTensor");
expectSame(tensorData.requirePositiveInteger(3, "n"), 3, "tensor data requirePositiveInteger");
expectSame(tensorData.zerosF32(3), [0, 0, 0], "tensor data zerosF32");
expectSame(tensorData.f32WithLength([[1, 2]], 2, "weights", [1, 2]), [1, 2], "tensor data f32WithLength shaped");
expectSame(tensorData.defaultedF32(undefined, 2, "bias", tensorData.zerosF32), [0, 0], "tensor data defaultedF32 fallback");

expectThrow(
  () => tensorData.prepareF32([[1], [2, 3]]),
  "tensor data must be rectangular; shape at [1] differs from earlier entries",
  "tensor data ragged nested input",
);
expectThrow(
  () => tensorData.f32WithLength([1, 2, 3], 2, "weights"),
  "weights length must be 2, got 3",
  "tensor data length mismatch",
);
expectThrow(
  () => tensorData.f32WithLength([[1, 2]], 2, "weights", [2, 1]),
  "weights shape must be [2, 1] or flat length 2, got [1, 2]",
  "tensor data shape mismatch",
);

let TensorCoreSmokeTensor = TensorDataSmokeTensor;
const tensorCore = createTensorCoreHelpers({
  getTensorClass: () => TensorCoreSmokeTensor,
  f32WithLength: tensorData.f32WithLength,
  prepareF32: tensorData.prepareF32,
  addTensorGrad: tensorData.addTensorGrad,
});
const coreTensor = new TensorCoreSmokeTensor(Float32Array.of(1, 2, 3), [3], { requiresGrad: true });
expectSame(tensorCore.toFloat32Array(coreTensor), [1, 2, 3], "tensor core toFloat32Array");
expectSame(tensorCore.ndim(coreTensor), 1, "tensor core ndim");
expectSame(tensorCore.dim(coreTensor), 1, "tensor core dim");
expectSame(tensorCore.numel(coreTensor), 3, "tensor core numel");
expectSame(tensorCore.size(coreTensor), [3], "tensor core size shape");
expectSame(Object.isFrozen(tensorCore.size(coreTensor)), true, "tensor core size frozen");
expectSame(tensorCore.size(coreTensor, -1), 3, "tensor core size dim");
expectSame(tensorCore.allclose(coreTensor, [1, 2.000001, 3], { atol: 1e-4 }), true, "tensor core allclose");
expectSame(tensorCore.equal(coreTensor, [1, 2, 3]), true, "tensor core equal");
expectSame(tensorCore.toJSON(coreTensor), { dtype: "f32", shape: [3], data: [1, 2, 3], requiresGrad: true }, "tensor core toJSON preserves requiresGrad");
expectSame(tensorCore.fromJSON({ dtype: "f32", shape: [2], data: [4, 5], requiresGrad: true }).requiresGrad, true, "tensor core fromJSON preserves requiresGrad");
expectSame(tensorCore.fromJSON({ dtype: "f32", shape: [2], data: [4, 5], requires_grad: true }).requiresGrad, true, "tensor core fromJSON accepts requires_grad alias");
expectSame(tensorCore.fromJSON({ dtype: "f32", shape: [2], data: [4, 5] }).data, [4, 5], "tensor core fromJSON");
coreTensor.grad.set([9, 9, 9]);
tensorCore.zeroGrad(coreTensor);
expectSame(coreTensor.grad, [0, 0, 0], "tensor core zeroGrad");
const backwardRoot = new TensorCoreSmokeTensor(Float32Array.of(10), [1], { requiresGrad: true });
const backwardParent = new TensorCoreSmokeTensor(Float32Array.of(3), [1], { requiresGrad: true });
backwardRoot._prev = [backwardParent];
backwardRoot._backward = (grad) => {
  tensorData.addTensorGrad(backwardParent, Float32Array.of(grad[0] * 2));
};
tensorCore.backward(backwardRoot);
expectSame(backwardRoot.grad, [1], "tensor core backward root grad");
expectSame(backwardParent.grad, [2], "tensor core backward parent grad");
expectThrow(
  () => tensorCore.item(coreTensor),
  "Tensor.item() requires a scalar tensor, got length 3",
  "tensor core item rejects non scalar",
);
expectThrow(
  () => tensorCore.backward(coreTensor),
  "Tensor.backward() on a non-scalar tensor requires an explicit gradient",
  "tensor core backward rejects non scalar without gradient",
);

const tensorMath = createTensorMathHelpers({
  getTensorClass: () => TensorDataSmokeTensor,
  f32: tensorData.f32,
  addTensorGrad: tensorData.addTensorGrad,
  scalarTensor: (value, requiresGrad = false) =>
    new TensorDataSmokeTensor(Float32Array.of(value), [1], { requiresGrad }),
  isGradEnabled: () => true,
});
const mathA = new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4), [2, 2], { requiresGrad: true });
const mathB = new TensorDataSmokeTensor(Float32Array.of(10, 20), [1, 2], { requiresGrad: true });
const addResult = tensorMath.add(mathA, mathB);
expectSame({ data: addResult.data, shape: addResult.shape, requiresGrad: addResult.requiresGrad }, {
  data: [11, 22, 13, 24],
  shape: [2, 2],
  requiresGrad: true,
}, "tensor math add broadcast");
addResult._backward(Float32Array.of(1, 1, 1, 1));
expectSame(mathA.grad, [1, 1, 1, 1], "tensor math add lhs grad");
expectSame(mathB.grad, [2, 2], "tensor math add rhs grad");
expectSame(tensorMath.mul(mathA, 2).data, [2, 4, 6, 8], "tensor math scalar mul");
const math3d = new TensorDataSmokeTensor(Float32Array.from({ length: 24 }, (_value, index) => index + 1), [2, 3, 4]);
const mathTrailing = new TensorDataSmokeTensor(Float32Array.of(10, 20, 30, 40), [4]);
expectSame({ data: tensorMath.add(math3d, mathTrailing).data.slice(0, 8), shape: tensorMath.add(math3d, mathTrailing).shape }, {
  data: [11, 22, 33, 44, 15, 26, 37, 48],
  shape: [2, 3, 4],
}, "tensor math add 3d trailing broadcast");
expectSame({ data: tensorMath.where(tensorMath.gt(mathA, 2), mathB, 0).data, shape: tensorMath.where(tensorMath.gt(mathA, 2), mathB, 0).shape }, {
  data: [0, 0, 10, 20],
  shape: [2, 2],
}, "tensor math where broadcast values");
const clampInput = new TensorDataSmokeTensor(Float32Array.of(-2, -1, 0, 3), [4], { requiresGrad: true });
const clamped = tensorMath.clamp(clampInput, -1, 2);
expectSame(clamped.data, [-1, -1, 0, 2], "tensor math clamp");
clamped._backward(Float32Array.of(1, 1, 1, 1));
expectSame(clampInput.grad, [0, 1, 1, 0], "tensor math clamp grad");
const lhs = new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4), [2, 2], { requiresGrad: true });
const rhs = new TensorDataSmokeTensor(Float32Array.of(5, 6, 7, 8), [2, 2], { requiresGrad: true });
const mm = tensorMath.matmul(lhs, rhs);
expectSame({ data: mm.data, shape: mm.shape }, { data: [19, 22, 43, 50], shape: [2, 2] }, "tensor math matmul");
mm._backward(Float32Array.of(1, 1, 1, 1));
expectSame(lhs.grad, [11, 15, 11, 15], "tensor math matmul lhs grad");
expectSame(rhs.grad, [4, 4, 6, 6], "tensor math matmul rhs grad");
let nativeMatmulCalls = 0;
let nativeBmmCalls = 0;
const noGradNativeMatmul = createTensorMathHelpers({
  getTensorClass: () => TensorDataSmokeTensor,
  f32: tensorData.f32,
  addTensorGrad: tensorData.addTensorGrad,
  scalarTensor: (value, requiresGrad = false) =>
    new TensorDataSmokeTensor(Float32Array.of(value), [1], { requiresGrad }),
  isGradEnabled: () => false,
  nativeEagerMatmulInto(output, left, right, options) {
    nativeMatmulCalls += 1;
    if (nativeMatmulCalls === 1) {
      expectSame(options, { rows: 2, shared: 2, cols: 2 }, "tensor math native matmul options");
    }
    const leftData = left.data ?? left;
    const rightData = right.data ?? right;
    const rows = options.rows;
    const shared = options.shared;
    const cols = options.cols;
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        let acc = 0;
        for (let k = 0; k < shared; k += 1) acc += leftData[r * shared + k] * rightData[k * cols + c];
        output[r * cols + c] = acc;
      }
    }
    return output;
  },
  nativeEagerBmmInto(output, left, right, options) {
    nativeBmmCalls += 1;
    expectSame(options, { batch: 2, rows: 8, shared: 8, cols: 8 }, "tensor math native bmm options");
    const leftData = left.data ?? left;
    const rightData = right.data ?? right;
    const batch = options.batch;
    const rows = options.rows;
    const shared = options.shared;
    const cols = options.cols;
    const lhsBatchLen = rows * shared;
    const rhsBatchLen = shared * cols;
    const outBatchLen = rows * cols;
    for (let b = 0; b < batch; b += 1) {
      for (let r = 0; r < rows; r += 1) {
        for (let c = 0; c < cols; c += 1) {
          let acc = 0;
          for (let k = 0; k < shared; k += 1) {
            acc += leftData[b * lhsBatchLen + r * shared + k] * rightData[b * rhsBatchLen + k * cols + c];
          }
          output[b * outBatchLen + r * cols + c] = acc;
        }
      }
    }
    return output;
  },
});
const nativeMm = noGradNativeMatmul.matmul(
  new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4), [2, 2]),
  new TensorDataSmokeTensor(Float32Array.of(5, 6, 7, 8), [2, 2]),
);
expectSame({ data: nativeMm.data, shape: nativeMm.shape, requiresGrad: nativeMm.requiresGrad }, {
  data: [19, 22, 43, 50],
  shape: [2, 2],
  requiresGrad: false,
}, "tensor math no-grad native matmul hook");
expectSame(nativeMatmulCalls, 1, "tensor math no-grad native matmul hook count");
const nativeBmmValues = Float32Array.from({ length: 2 * 8 * 8 }, (_value, index) => index % 11 - 5);
const nativeBmmIdentity = Float32Array.from({ length: 2 * 8 * 8 }, (_value, index) => {
  const matrixIndex = index % 64;
  return Math.floor(matrixIndex / 8) === matrixIndex % 8 ? 1 : 0;
});
const nativeBmm = noGradNativeMatmul.bmm(
  new TensorDataSmokeTensor(nativeBmmValues, [2, 8, 8]),
  new TensorDataSmokeTensor(nativeBmmIdentity, [2, 8, 8]),
);
expectSame({ data: nativeBmm.data, shape: nativeBmm.shape, requiresGrad: nativeBmm.requiresGrad }, {
  data: Array.from(nativeBmmValues),
  shape: [2, 8, 8],
  requiresGrad: false,
}, "tensor math no-grad native bmm hook");
expectSame(nativeMatmulCalls, 1, "tensor math no-grad native bmm avoids per-batch matmul hook");
expectSame(nativeBmmCalls, 1, "tensor math no-grad native bmm hook count");
const nativeActivationCalls = [];
const noGradNativeActivation = createTensorMathHelpers({
  getTensorClass: () => TensorDataSmokeTensor,
  f32: tensorData.f32,
  addTensorGrad: tensorData.addTensorGrad,
  scalarTensor: (value, requiresGrad = false) =>
    new TensorDataSmokeTensor(Float32Array.of(value), [1], { requiresGrad }),
  isGradEnabled: () => false,
  nativeEagerActivationMinLength: 0,
  nativeEagerActivationEnabled: () => true,
  nativeEagerActivationInto(output, input, options) {
    nativeActivationCalls.push(options.activation);
    const inputData = input.data ?? input;
    for (let i = 0; i < output.length; i += 1) {
      switch (options.activation) {
        case "relu": output[i] = Math.max(0, inputData[i]); break;
        case "tanh": output[i] = Math.tanh(inputData[i]); break;
        default: throw new Error(`unexpected native activation ${options.activation}`);
      }
    }
    return output;
  },
});
expectSame(noGradNativeActivation.relu(
  new TensorDataSmokeTensor(Float32Array.of(-2, -1, 0, 3), [2, 2]),
).data, [0, 0, 0, 3], "tensor math no-grad native relu hook");
const nativeActivationTanh = noGradNativeActivation.tanh(
  new TensorDataSmokeTensor(Float32Array.of(0, 1), [2]),
);
expectApprox(nativeActivationTanh.data[0], 0, 1e-7, "tensor math no-grad native tanh hook 0");
expectApprox(nativeActivationTanh.data[1], Math.tanh(1), 1e-7, "tensor math no-grad native tanh hook 1");
expectSame(nativeActivationCalls, ["relu", "tanh"], "tensor math no-grad native activation hook count");
const nativeElementwiseCalls: string[] = [];
const nativeElementwiseReduceCalls: string[] = [];
const nativeDotCalls: string[] = [];
const nativeWhereCalls: string[] = [];
const noGradNativeElementwise = createTensorMathHelpers({
  getTensorClass: () => TensorDataSmokeTensor,
  f32: tensorData.f32,
  addTensorGrad: tensorData.addTensorGrad,
  scalarTensor: (value, requiresGrad = false) =>
    new TensorDataSmokeTensor(Float32Array.of(value), [1], { requiresGrad }),
  isGradEnabled: () => false,
  nativeEagerElementwiseMinLength: 0,
  nativeEagerReduceMinLength: 0,
  nativeEagerElementwiseInto(output, left, right, options) {
    nativeElementwiseCalls.push(options.rows && options.cols ? `${options.op}:${options.broadcast ?? "rhs"}:${options.rows}x${options.cols}` : options.op);
    const leftData = left.data ?? left;
    const rightData = right === null ? null : right.data ?? right;
    for (let i = 0; i < output.length; i += 1) {
      const lhsIndex = options.broadcast === "lhs" && options.cols ? i % options.cols : i;
      const rhsIndex = rightData === null || rightData.length === 1
        ? 0
        : options.broadcast === "rhs" && options.cols
          ? i % options.cols
          : i;
      const lhsValue = leftData[lhsIndex];
      const rhsValue = rightData === null ? 0 : rightData[rhsIndex];
      switch (options.op) {
        case "add": output[i] = lhsValue + rhsValue; break;
        case "mul": output[i] = lhsValue * rhsValue; break;
        case "sqr": output[i] = lhsValue * lhsValue; break;
        case "recip": output[i] = 1 / lhsValue; break;
        case "sqrt": output[i] = Math.sqrt(lhsValue); break;
        case "rsqrt": output[i] = 1 / Math.sqrt(lhsValue); break;
        case "eq": output[i] = Object.is(lhsValue, rhsValue) || lhsValue === rhsValue ? 1 : 0; break;
        case "lt": output[i] = lhsValue < rhsValue ? 1 : 0; break;
        case "maximum": output[i] = Math.max(lhsValue, rhsValue); break;
        case "minimum": output[i] = Math.min(lhsValue, rhsValue); break;
        default: throw new Error(`unexpected native elementwise op ${options.op}`);
      }
    }
    return output;
  },
  nativeEagerReduceInto(output, input, options) {
    nativeElementwiseReduceCalls.push(options.op);
    const data = input.data ?? input;
    switch (options.op) {
      case "sum": output[0] = Array.from(data).reduce((acc, value) => acc + value, 0); break;
      default: throw new Error(`unexpected native elementwise reduce op ${options.op}`);
    }
    return output;
  },
  nativeEagerDotInto(output, left, right) {
    nativeDotCalls.push("dot");
    const leftData = left.data ?? left;
    const rightData = right.data ?? right;
    output[0] = Array.from(leftData).reduce((acc, value, index) => acc + value * rightData[index], 0);
    return output;
  },
  nativeEagerWhereInto(output, condition, input, other) {
    nativeWhereCalls.push("where");
    const conditionData = condition.data ?? condition;
    const inputData = input.data ?? input;
    const otherData = other.data ?? other;
    for (let i = 0; i < output.length; i += 1) {
      output[i] = conditionData[i] !== 0
        ? inputData[inputData.length === 1 ? 0 : i]
        : otherData[otherData.length === 1 ? 0 : i];
    }
    return output;
  },
  nativeEagerClampInto(output, input, options) {
    nativeElementwiseCalls.push("clamp");
    const inputData = input.data ?? input;
    const hasMin = options.min !== undefined;
    const hasMax = options.max !== undefined;
    for (let i = 0; i < output.length; i += 1) {
      let value = inputData[i];
      if (hasMin) value = Math.max(value, options.min);
      if (hasMax) value = Math.min(value, options.max);
      output[i] = value;
    }
    return output;
  },
});
expectSame(noGradNativeElementwise.mul(
  new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4), [2, 2]),
  2,
).data, [2, 4, 6, 8], "tensor math no-grad native scalar mul hook");
expectSame(noGradNativeElementwise.add(
  new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4), [2, 2]),
  new TensorDataSmokeTensor(Float32Array.of(10, 20, 30, 40), [2, 2]),
).data, [11, 22, 33, 44], "tensor math no-grad native same-shape add hook");
expectSame(noGradNativeElementwise.add(
  new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4, 5, 6, 7, 8), [2, 4]),
  new TensorDataSmokeTensor(Float32Array.of(10, 20, 30, 40), [4]),
).data, [11, 22, 33, 44, 15, 26, 37, 48], "tensor math no-grad native row-broadcast add hook");
expectSame(noGradNativeElementwise.add(
  new TensorDataSmokeTensor(Float32Array.of(10, 20, 30, 40), [4]),
  new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4, 5, 6, 7, 8), [2, 4]),
).data, [11, 22, 33, 44, 15, 26, 37, 48], "tensor math no-grad native lhs row-broadcast add hook");
expectSame(noGradNativeElementwise.sqr(
  new TensorDataSmokeTensor(Float32Array.of(1, -2, 3, -4), [2, 2]),
).data, [1, 4, 9, 16], "tensor math no-grad native unary sqr hook");
expectSame(noGradNativeElementwise.pow(
  new TensorDataSmokeTensor(Float32Array.of(1, -2, 3, -4), [2, 2]),
  2,
).data, [1, 4, 9, 16], "tensor math no-grad native pow square hook");
expectSame(noGradNativeElementwise.pow(
  new TensorDataSmokeTensor(Float32Array.of(1, 4, 9, 16), [2, 2]),
  0.5,
).data, [1, 2, 3, 4], "tensor math no-grad native pow sqrt hook");
expectSame(noGradNativeElementwise.pow(
  new TensorDataSmokeTensor(Float32Array.of(1, 2, 4, 8), [2, 2]),
  -1,
).data, [1, 0.5, 0.25, 0.125], "tensor math no-grad native pow reciprocal hook");
expectSame(noGradNativeElementwise.pow(
  new TensorDataSmokeTensor(Float32Array.of(1, 4, 9, 16), [2, 2]),
  -0.5,
).data, [1, 0.5, 0.3333333432674408, 0.25], "tensor math no-grad native pow rsqrt hook");
expectSame(noGradNativeElementwise.lt(
  new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4), [2, 2]),
  3,
).data, [1, 1, 0, 0], "tensor math native scalar lt hook");
expectSame(noGradNativeElementwise.eq(
  new TensorDataSmokeTensor(Float32Array.of(1, Number.NaN, -0, 4), [2, 2]),
  new TensorDataSmokeTensor(Float32Array.of(1, Number.NaN, 0, 5), [2, 2]),
).data, [1, 1, 1, 0], "tensor math native same-shape eq hook");
expectSame(noGradNativeElementwise.clamp(
  new TensorDataSmokeTensor(Float32Array.of(-2, -1, 0, 3), [2, 2]),
  -1,
  2,
).data, [-1, -1, 0, 2], "tensor math native clamp hook");
expectSame(noGradNativeElementwise.where(
  new TensorDataSmokeTensor(Float32Array.of(1, 0, -1, 0), [2, 2]),
  new TensorDataSmokeTensor(Float32Array.of(10, 20, 30, 40), [2, 2]),
  -5,
).data, [10, -5, 30, -5], "tensor math native where hook");
expectSame(noGradNativeElementwise.dot(
  new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4), [4]),
  new TensorDataSmokeTensor(Float32Array.of(0.5, 1.5, -1, 2), [4]),
).data, [8.5], "tensor math no-grad native dot hook");
expectSame(noGradNativeElementwise.add(math3d, mathTrailing).data.slice(0, 4), [11, 22, 33, 44], "tensor math no-grad native rank-3 row-broadcast hook");
expectSame(noGradNativeElementwise.add(mathTrailing, math3d).data.slice(0, 4), [11, 22, 33, 44], "tensor math no-grad native rank-3 lhs row-broadcast hook");
expectSame(nativeElementwiseCalls, ["mul", "add", "add:rhs:2x4", "add:lhs:2x4", "sqr", "sqr", "sqrt", "recip", "rsqrt", "lt", "eq", "clamp", "add:rhs:6x4", "add:lhs:6x4"], "tensor math no-grad native elementwise hook count");
expectSame(nativeDotCalls, ["dot"], "tensor math no-grad native dot hook count");
expectSame(nativeElementwiseReduceCalls, [], "tensor math no-grad native dot avoids composed reduce fallback");
expectSame(nativeWhereCalls, ["where"], "tensor math no-grad native where hook count");
const nativeReduceCalls: string[] = [];
const noGradNativeReduce = createTensorMathHelpers({
  getTensorClass: () => TensorDataSmokeTensor,
  f32: tensorData.f32,
  addTensorGrad: tensorData.addTensorGrad,
  scalarTensor: (value, requiresGrad = false) =>
    new TensorDataSmokeTensor(Float32Array.of(value), [1], { requiresGrad }),
  isGradEnabled: () => false,
  nativeEagerReduceMinLength: 0,
  nativeEagerReduceInto(output, input, options) {
    nativeReduceCalls.push(options.op);
    const data = input.data;
    switch (options.op) {
      case "sum": output[0] = Array.from(data).reduce((acc, value) => acc + value, 0); break;
      case "mean": output[0] = Array.from(data).reduce((acc, value) => acc + value, 0) / data.length; break;
      case "max": output[0] = Math.max(...data); break;
      case "min": output[0] = Math.min(...data); break;
      case "prod": output[0] = Array.from(data).reduce((acc, value) => acc * value, 1); break;
      default: throw new Error(`unexpected native reduce op ${options.op}`);
    }
    return output;
  },
});
const reduceInput = new TensorDataSmokeTensor(Float32Array.of(-2, 4, 0.5, 3), [4]);
expectSame(noGradNativeReduce.sum(reduceInput).data, [5.5], "tensor math no-grad native sum hook");
expectSame(noGradNativeReduce.mean(reduceInput).data, [1.375], "tensor math no-grad native mean hook");
expectSame(noGradNativeReduce.max(reduceInput).data, [4], "tensor math no-grad native max hook");
expectSame(noGradNativeReduce.min(reduceInput).data, [-2], "tensor math no-grad native min hook");
expectSame(noGradNativeReduce.prod(reduceInput).data, [-12], "tensor math no-grad native prod hook");
expectSame(noGradNativeReduce.sumDim(new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4), [2, 2]), 1).data, [3, 7], "tensor math dim reduction keeps TS path");
expectSame(nativeReduceCalls, ["sum", "mean", "max", "min", "prod"], "tensor math no-grad native reduce hook count");
const nativeSoftmaxCalls: string[] = [];
const noGradNativeSoftmax = createTensorMathHelpers({
  getTensorClass: () => TensorDataSmokeTensor,
  f32: tensorData.f32,
  addTensorGrad: tensorData.addTensorGrad,
  scalarTensor: (value, requiresGrad = false) =>
    new TensorDataSmokeTensor(Float32Array.of(value), [1], { requiresGrad }),
  isGradEnabled: () => false,
  nativeEagerSoftmaxInto(output, input, options) {
    nativeSoftmaxCalls.push(options.logSoftmax ? "logSoftmax" : "softmax");
    expectSame({ dim: options.dim, shape: input.shape }, { dim: -1, shape: [2, 3] }, "tensor math no-grad native softmax options");
    const data = input.data;
    for (let row = 0; row < 2; row += 1) {
      const base = row * 3;
      const max = Math.max(data[base], data[base + 1], data[base + 2]);
      const e0 = Math.exp(data[base] - max);
      const e1 = Math.exp(data[base + 1] - max);
      const e2 = Math.exp(data[base + 2] - max);
      const denom = e0 + e1 + e2;
      if (options.logSoftmax) {
        const logDenom = Math.log(denom);
        output[base] = data[base] - max - logDenom;
        output[base + 1] = data[base + 1] - max - logDenom;
        output[base + 2] = data[base + 2] - max - logDenom;
      } else {
        output[base] = e0 / denom;
        output[base + 1] = e1 / denom;
        output[base + 2] = e2 / denom;
      }
    }
    return output;
  },
});
const softmaxInput = new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4, 5, 6), [2, 3]);
Array.from(noGradNativeSoftmax.softmax(softmaxInput).data).forEach((value, index) => expectApprox(value, [
  0.09003057, 0.24472847, 0.66524096,
  0.09003057, 0.24472847, 0.66524096,
][index], 1e-6, "tensor math no-grad native softmax hook"));
Array.from(noGradNativeSoftmax.logSoftmax(softmaxInput).data).forEach((value, index) => expectApprox(value, [
  -2.407606, -1.407606, -0.407606,
  -2.407606, -1.407606, -0.407606,
][index], 1e-6, "tensor math no-grad native logSoftmax hook"));
expectSame(nativeSoftmaxCalls, ["softmax", "logSoftmax"], "tensor math no-grad native softmax hook count");
const red = new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4, 5, 6), [2, 3], { requiresGrad: true });
const summed = tensorMath.sumDim(red, 1);
expectSame({ data: summed.data, shape: summed.shape }, { data: [6, 15], shape: [2, 1] }, "tensor math sumDim");
summed._backward(Float32Array.of(10, 20));
expectSame(red.grad, [10, 10, 10, 20, 20, 20], "tensor math sumDim grad");
red.grad.fill(0);
const maxed = tensorMath.maxDim(red, 1);
expectSame({ data: maxed.data, shape: maxed.shape }, { data: [3, 6], shape: [2, 1] }, "tensor math maxDim");
maxed._backward(Float32Array.of(1, 2));
expectSame(red.grad, [0, 0, 1, 0, 0, 2], "tensor math maxDim grad");
const red3d = new TensorDataSmokeTensor(Float32Array.from({ length: 24 }, (_value, index) => index + 1), [2, 3, 4]);
expectSame({ data: tensorMath.sumDim(red3d, 0).data, shape: tensorMath.sumDim(red3d, 0).shape }, {
  data: [14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36],
  shape: [1, 3, 4],
}, "tensor math sumDim 3d dim0");
expectSame({ data: tensorMath.meanDim(red3d, -1).data, shape: tensorMath.meanDim(red3d, -1).shape }, {
  data: [2.5, 6.5, 10.5, 14.5, 18.5, 22.5],
  shape: [2, 3, 1],
}, "tensor math meanDim 3d trailing");
expectSame({ data: tensorMath.maxDim(red3d, 1).data, shape: tensorMath.maxDim(red3d, 1).shape }, {
  data: [9, 10, 11, 12, 21, 22, 23, 24],
  shape: [2, 1, 4],
}, "tensor math maxDim 3d middle");
const logits = new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 1), [2, 2], { requiresGrad: true });
expectSame(
  Array.from(tensorMath.softmaxDim(logits, 1).data, (value) => Number(value.toFixed(6))),
  [0.268941, 0.731059, 0.880797, 0.119203],
  "tensor math softmaxDim",
);
expectSame(
  Array.from(tensorMath.logSoftmaxDim(logits, 1).data, (value) => Number(value.toFixed(6))),
  [-1.313262, -0.313262, -0.126928, -2.126928],
  "tensor math logSoftmaxDim",
);
expectThrow(
  () => tensorMath.clamp(mathA),
  "Tensor.clamp requires min, max, or both",
  "tensor math clamp rejects missing bounds",
);
expectThrow(
  () => tensorMath.matmul(new TensorDataSmokeTensor(Float32Array.of(1), [1]), rhs),
  "tensor matmul lhs must be rank-2",
  "tensor math matmul rejects lhs rank",
);

const tensorFactory = createTensorFactoryHelpers({ Tensor: TensorDataSmokeTensor });
expectSame(tensorFactory.full([2, 2], 3).data, [3, 3, 3, 3], "tensor factory full");
expectSame(tensorFactory.zeros(3).shape, [3], "tensor factory zeros shape");
expectSame(tensorFactory.ones([2]).data, [1, 1], "tensor factory ones");
expectSame(tensorFactory.eye(3).data, [1, 0, 0, 0, 1, 0, 0, 0, 1], "tensor factory eye");
expectSame(tensorFactory.scalar(4).data, [4], "tensor factory scalar");
expectSame(tensorFactory.rand([3], { rng: () => 0.25 }).data, [0.25, 0.25, 0.25], "tensor factory rand rng");
expectSame(tensorFactory.rand([3], { seed: 123 }).data, tensorFactory.rand([3], { seed: 123 }).data, "tensor factory rand seed option");
expectSame(tensorFactory.manualSeed(456), 456, "tensor factory manualSeed returns normalized seed");
expectSame(tensorFactory.initialSeed(), 456, "tensor factory initialSeed after manualSeed");
expectSame(tensorFactory.rand([3]).data, tensorFactory.manual_seed(456) && tensorFactory.rand([3]).data, "tensor factory manual_seed resets default RNG");
const seededRngA = tensorFactory.seededRng(789);
const seededRngB = tensorFactory.seededRng(789);
expectSame([seededRngA(), seededRngA()], [seededRngB(), seededRngB()], "tensor factory seededRng is deterministic");
const randnValues = [0.5, 0, 0.5, 0.25];
expectSame(
  Array.from(tensorFactory.randn([4], { rng: () => randnValues.shift(), mean: 1, std: 2 }).data, (value) => Number(value.toFixed(6))),
  [3.35482, 1, 1, 3.35482],
  "tensor factory randn rng",
);
expectSame(tensorFactory.linspace(0, 1, 4).data, [0, 0.25, 0.5, 0.75], "tensor factory linspace shorthand");
expectSame(tensorFactory.linspace([3], -1, 2).data, [-1, 0, 1], "tensor factory linspace shaped");
expectSame(tensorFactory.arange(4).data, [0, 1, 2, 3], "tensor factory arange end");
expectSame(tensorFactory.arange(4, 0, -2).data, [4, 2], "tensor factory arange descending");
expectThrow(
  () => tensorFactory.full([1], Infinity),
  "full tensor value must be finite, got Infinity",
  "tensor factory full rejects infinite value",
);
expectThrow(
  () => tensorFactory.arange(0, 0),
  "arange produced an empty tensor",
  "tensor factory arange rejects empty",
);
expectThrow(
  () => tensorFactory.eye(0),
  "eye size must be a positive safe integer, got 0",
  "tensor factory eye rejects invalid size",
);

const facadeNativeBuffers = new WeakSet();
function makeFacadeNativeBuffer(data, byteOffset = 0) {
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const storage = new Uint8Array(byteOffset + bytes.byteLength);
  storage.set(bytes, byteOffset);
  const backing = storage.buffer;
  const buffer = {
    byteLength: storage.byteLength,
    writes: [],
    readFloat32(length, offset = 0) {
      return new Float32Array(backing, offset, length);
    },
    readFloat32Into(target, length = target.length, offset = 0) {
      target.set(new Float32Array(backing, offset, length));
      return target;
    },
    writeFloat32(value) {
      this.writes.push(Array.from(value));
    },
  };
  facadeNativeBuffers.add(buffer);
  return buffer;
}
const facadePlacements = [];
const tensorFacadeHelpers = tensorFacadePolicy.createTensorFacadeHelpers({
  Tensor: TensorDataSmokeTensor,
  rawF32: tensorData.rawF32,
  prepareF32: tensorData.prepareF32,
  isTensor: (value) => value instanceof TensorDataSmokeTensor,
  isNativeBuffer: (value) => facadeNativeBuffers.has(value),
  nativeBufferFromFloat32: (value) => makeFacadeNativeBuffer(value),
  tensorFactory,
  tensorPlacement: {
    validateProgramPlacement(tensor, program, kind) {
      facadePlacements.push({ shape: tensor.shape, kind, programName: program.name });
    },
  },
  tensorJoin: {
    cat: (tensors, dim) => ({ op: "cat", tensors, dim }),
    stack: (tensors, dim) => ({ op: "stack", tensors, dim }),
    einsum: (equation, tensors, ...moreTensors) => ({ op: "einsum", equation, tensors, moreTensors }),
  },
});
const initializedTensor = {};
tensorFacadeHelpers.initialize(initializedTensor, [[1, 2]], undefined, { requiresGrad: true });
expectSame({
  data: initializedTensor.data,
  shape: initializedTensor.shape,
  requiresGrad: initializedTensor.requiresGrad,
  grad: initializedTensor.grad,
}, {
  data: [1, 2],
  shape: [1, 2],
  requiresGrad: true,
  grad: [0, 0],
}, "tensor facade initialize");
expectSame(tensorFacadeHelpers.parameter([1, 2], [2]).requiresGrad, true, "tensor facade parameter requires grad");
const facadeReadBuffer = makeFacadeNativeBuffer(Float32Array.of(5, 6), Float32Array.BYTES_PER_ELEMENT);
expectSame(tensorFacadeHelpers.fromNativeBuffer(facadeReadBuffer, [2], { byteOffset: 4 }).data, [5, 6], "tensor facade fromNativeBuffer");
const facadeCopyTarget = new TensorDataSmokeTensor(Float32Array.of(0, 0), [2]);
expectSame(tensorFacadeHelpers.copyFromNativeBuffer_(facadeCopyTarget, facadeReadBuffer, { byteOffset: 4 }), facadeCopyTarget, "tensor facade copyFromNativeBuffer_ returns tensor");
expectSame(facadeCopyTarget.data, [5, 6], "tensor facade copyFromNativeBuffer_ fills existing tensor");
expectSame(tensorFacadeHelpers.copy_from_native_buffer_(facadeCopyTarget, facadeReadBuffer, { byteOffset: 4 }), facadeCopyTarget, "tensor facade copy_from_native_buffer_ returns tensor");
const placedTensor = new TensorDataSmokeTensor(Float32Array.of(7, 8), [2]);
const programBuffer = makeFacadeNativeBuffer(new Float32Array(2));
const placedBuffer = tensorFacadeHelpers.place(placedTensor, {
  name: "smoke-program",
  createBuffer(kind, options) {
    programBuffer.kind = kind;
    programBuffer.options = options;
    return programBuffer;
  },
}, "input", { program: "removed", kind: "removed", label: "kept" });
expectSame(facadePlacements, [{ shape: [2], kind: "input", programName: "smoke-program" }], "tensor facade validates placement");
expectSame({ kind: placedBuffer.kind, options: placedBuffer.options, writes: placedBuffer.writes }, {
  kind: "input",
  options: { label: "kept" },
  writes: [[7, 8]],
}, "tensor facade place writes");
expectSame(tensorFacadeHelpers.cat(["a", "b"], 1), { op: "cat", tensors: ["a", "b"], dim: 1 }, "tensor facade cat delegates");
expectSame(tensorFacadeHelpers.einsum("i,i->", ["a", "b"]), { op: "einsum", equation: "i,i->", tensors: ["a", "b"], moreTensors: [] }, "tensor facade einsum delegates");
expectSame(tensorFacadeHelpers.eye(2).data, [1, 0, 0, 1], "tensor facade eye delegates");
expectSame(tensorFacadeHelpers.arange(3).data, [0, 1, 2], "tensor facade arange delegates");
expectSame(tensorFacadePolicy.normalizeShapeModuleShape([-1, 2], "view shape", true), [-1, 2], "tensor facade shape module infer");
expectThrow(
  () => tensorFacadePolicy.normalizeShapeModuleShape([-1, -1], "view shape", true),
  "view shape can infer at most one dimension",
  "tensor facade shape module rejects double infer",
);
expectThrow(
  () => tensorFacadeHelpers.fromNativeBuffer({}, [1]),
  "Tensor.fromNativeBuffer requires a zgml NativeBuffer",
  "tensor facade rejects non native buffer",
);

const tensorJoin = createTensorJoinHelpers({
  Tensor: TensorDataSmokeTensor,
  addTensorGrad: tensorData.addTensorGrad,
  isGradEnabled: () => true,
});
const joinA = new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4), [2, 2], { requiresGrad: true });
const joinB = new TensorDataSmokeTensor(Float32Array.of(5, 6, 7, 8), [2, 2], { requiresGrad: true });
const catResult = tensorJoin.cat([joinA, joinB], 1);
expectSame({ data: catResult.data, shape: catResult.shape, requiresGrad: catResult.requiresGrad }, {
  data: [1, 2, 5, 6, 3, 4, 7, 8],
  shape: [2, 4],
  requiresGrad: true,
}, "tensor join cat dim 1");
catResult._backward(Float32Array.of(10, 20, 30, 40, 50, 60, 70, 80));
expectSame(joinA.grad, [10, 20, 50, 60], "tensor join cat backward left");
expectSame(joinB.grad, [30, 40, 70, 80], "tensor join cat backward right");
joinA.grad.fill(0);
joinB.grad.fill(0);
const join4d = new TensorDataSmokeTensor(Float32Array.from({ length: 120 }, (_value, index) => index + 1), [2, 3, 4, 5]);
expectSame(tensorJoin.cat([join4d, join4d], -1).shape, [2, 3, 4, 10], "tensor join cat 4d trailing");
const stackResult = tensorJoin.stack([joinA, joinB], 0);
expectSame({ data: stackResult.data, shape: stackResult.shape, requiresGrad: stackResult.requiresGrad }, {
  data: [1, 2, 3, 4, 5, 6, 7, 8],
  shape: [2, 2, 2],
  requiresGrad: true,
}, "tensor join stack dim 0");
stackResult._backward(Float32Array.of(1, 2, 3, 4, 5, 6, 7, 8));
expectSame(joinA.grad, [1, 2, 3, 4], "tensor join stack backward first");
expectSame(joinB.grad, [5, 6, 7, 8], "tensor join stack backward second");
expectSame(tensorJoin.stack([join4d, join4d], 2).shape, [2, 3, 2, 4, 5], "tensor join stack 4d dim 2");
joinA.grad.fill(0);
joinB.grad.fill(0);
const einsumMatmul = tensorJoin.einsum("ij,jk->ik", [joinA, joinB]);
expectSame({ data: einsumMatmul.data, shape: einsumMatmul.shape, requiresGrad: einsumMatmul.requiresGrad }, {
  data: [19, 22, 43, 50],
  shape: [2, 2],
  requiresGrad: true,
}, "tensor join einsum matmul");
einsumMatmul._backward(Float32Array.of(1, 1, 1, 1));
expectSame(joinA.grad, [11, 15, 11, 15], "tensor join einsum backward lhs");
expectSame(joinB.grad, [4, 4, 6, 6], "tensor join einsum backward rhs");
joinA.grad.fill(0);
const einsumTrace = tensorJoin.einsum("ii->", [joinA]);
expectSame({ data: einsumTrace.data, shape: einsumTrace.shape }, { data: [5], shape: [1] }, "tensor join einsum repeated-label scalar");
einsumTrace._backward(Float32Array.of(2));
expectSame(joinA.grad, [2, 0, 0, 2], "tensor join einsum trace backward diagonal");
expectSame(tensorJoin.einsum("ij,jk->ik", joinA, joinB).data, [19, 22, 43, 50], "tensor join einsum variadic operands");
const einsumBatchLhs = new TensorDataSmokeTensor(Float32Array.from({ length: 8 }, (_value, index) => index + 1), [2, 2, 2], { requiresGrad: true });
const einsumBatchRhs = new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4), [2, 2], { requiresGrad: true });
const einsumEllipsis = tensorJoin.einsum("...ij,jk->...ik", [einsumBatchLhs, einsumBatchRhs]);
expectSame({ data: einsumEllipsis.data, shape: einsumEllipsis.shape }, {
  data: [7, 10, 15, 22, 23, 34, 31, 46],
  shape: [2, 2, 2],
}, "tensor join einsum ellipsis batch matmul");
einsumEllipsis._backward(Float32Array.from({ length: 8 }, () => 1));
expectSame(einsumBatchLhs.grad, [3, 7, 3, 7, 3, 7, 3, 7], "tensor join einsum ellipsis backward lhs");
expectSame(einsumBatchRhs.grad, [16, 16, 20, 20], "tensor join einsum ellipsis backward rhs");
expectSame(tensorJoin.einsum("...i->...", [new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4, 5, 6), [2, 3])]).data, [6, 15], "tensor join einsum implicit ellipsis reduction");
const einsumBroadcastLhs = new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4), [1, 2, 2], { requiresGrad: true });
const einsumBroadcastRhs = new TensorDataSmokeTensor(Float32Array.of(5, 6, 7, 8, 1, 0, 0, 1), [2, 2, 2], { requiresGrad: true });
const einsumBroadcast = tensorJoin.einsum("bij,bjk->bik", [einsumBroadcastLhs, einsumBroadcastRhs]);
expectSame({ data: einsumBroadcast.data, shape: einsumBroadcast.shape }, {
  data: [19, 22, 43, 50, 1, 2, 3, 4],
  shape: [2, 2, 2],
}, "tensor join einsum broadcast batch matmul");
einsumBroadcast._backward(Float32Array.from({ length: 8 }, () => 1));
expectSame(einsumBroadcastLhs.grad, [12, 16, 12, 16], "tensor join einsum broadcast backward accumulates size-one axis");
expectThrow(
  () => tensorJoin.cat([joinA, new TensorDataSmokeTensor(Float32Array.of(1, 2, 3), [3])], 0),
  "cat input 1 rank must be 2, got 1",
  "tensor join cat rejects rank mismatch",
);
expectThrow(
  () => tensorJoin.stack([joinA, new TensorDataSmokeTensor(Float32Array.of(1, 2), [2])]),
  "stack input 1 shape [2] must match [2,2]",
  "tensor join stack rejects shape mismatch",
);
expectThrow(
  () => tensorJoin.einsum("ij,jk->ik", [joinA, new TensorDataSmokeTensor(Float32Array.of(1, 2, 3), [3, 1])]),
  "einsum label j has inconsistent dimensions 2 and 3",
  "tensor join einsum rejects non-broadcast dimensions",
);

const dataNs = dataNamespace.createDataNamespace({
  stack: (tensors, dim = 0) => tensorJoin.stack(tensors, dim),
});
const dataInput = new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4), [2, 2]);
const dataTarget = new TensorDataSmokeTensor(Float32Array.of(10, 20), [2, 1]);
const dataSet = dataNs.tensorDataset(dataInput, dataTarget);
expectSame(Object.isFrozen(dataSet), true, "data namespace tensorDataset frozen");
class CustomDataSmokeDataset extends dataNs.Dataset {
  length = 2;
  sample(index) {
    return Object.freeze({
      kind: "zgml.data.sample",
      index,
      input: new TensorDataSmokeTensor(index === 0 ? Float32Array.of(1, 2) : Float32Array.of(3, 4), [2]),
      target: new TensorDataSmokeTensor(Float32Array.of(index), [1]),
    });
  }
}
const customDataSet = new CustomDataSmokeDataset();
expectSame({
  kind: customDataSet.kind,
  len: customDataSet.__len__(),
  sample: customDataSet.__getitem__(1).input.data,
  batchShape: customDataSet.batch([0, 1]).input.shape,
  loaderShape: dataNs.dataLoader(customDataSet, { batch_size: 2 }).__getitem__(0).target.shape,
}, {
  kind: "zgml.data.dataset",
  len: 2,
  sample: [3, 4],
  batchShape: [2, 2],
  loaderShape: [2, 1],
}, "data namespace Dataset subclasses compose with DataLoader");
expectSame({ kind: dataSet.kind, length: dataSet.length, len: dataSet.len(), size: dataSet.size() }, {
  kind: "zgml.data.tensor-dataset",
  length: 2,
  len: 2,
  size: 2,
}, "data namespace tensorDataset length helpers");
expectSame(dataSet.sample(1).input.data, [3, 4], "data namespace tensorDataset sample");
expectSame(Array.from(dataSet).map((sample) => sample.index), [0, 1], "data namespace tensorDataset iterable samples");
expectSame(Array.from(dataSet.__iter__()).map((sample) => sample.index), [0, 1], "data namespace tensorDataset __iter__ samples");
const sequentialSampler = new dataNs.SequentialSampler(dataSet);
const randomSampler = new dataNs.RandomSampler(dataSet, { seed: 7 });
const replacementSampler = new dataNs.RandomSampler(dataSet, { replacement: true, num_samples: 3, seed: 7 });
const batchSampler = new dataNs.BatchSampler(sequentialSampler, 1, false);
expectSame({
  sequentialKind: sequentialSampler.kind,
  sequentialLen: sequentialSampler.__len__(),
  sequentialOrder: Array.from(sequentialSampler),
  randomKind: randomSampler.kind,
  randomLen: randomSampler.len(),
  randomOrder: Array.from(randomSampler),
  replacementLen: replacementSampler.size(),
  replacementOrder: Array.from(replacementSampler),
  batchKind: batchSampler.kind,
  batchLen: batchSampler.__len__(),
  batchOrder: Array.from(batchSampler),
  loaderSamplerIndices: Array.from(dataNs.dataLoader(dataSet, { sampler: [1, 0], batch_size: 1 })).map((batch) => batch.indices),
  loaderBatchSamplerIndices: Array.from(dataNs.dataLoader(dataSet, { batch_sampler: batchSampler })).map((batch) => batch.indices),
  defaultCollateBatch: dataNs.default_collate([dataSet.sample(1), dataSet.sample(0)], { batch_index: 2 }).indices,
  defaultCollateInput: dataNs.defaultCollate([dataSet.sample(1), dataSet.sample(0)]).input.data,
  loaderCollate: ((batch) => ({
    kind: batch.kind,
    indices: batch.indices,
    batchIndex: batch.batchIndex,
    input: batch.input.data,
    inputShape: batch.input.shape,
    target: batch.target.data,
    targetShape: batch.target.shape,
  }))(Array.from(dataNs.dataLoader(dataSet, {
      sampler: [1, 0],
      batch_size: 2,
      collate_fn: dataNs.default_collate,
    }))[0]),
}, {
  sequentialKind: "zgml.data.sequential-sampler",
  sequentialLen: 2,
  sequentialOrder: [0, 1],
  randomKind: "zgml.data.random-sampler",
  randomLen: 2,
  randomOrder: [1, 0],
  replacementLen: 3,
  replacementOrder: [0, 1, 1],
  batchKind: "zgml.data.batch-sampler",
  batchLen: 2,
  batchOrder: [[0], [1]],
  loaderSamplerIndices: [[1], [0]],
  loaderBatchSamplerIndices: [[0], [1]],
  defaultCollateBatch: [1, 0],
  defaultCollateInput: [3, 4, 1, 2],
  loaderCollate: {
    kind: "zgml.data.batch",
    indices: [1, 0],
    batchIndex: 0,
    input: [3, 4, 1, 2],
    inputShape: [2, 2],
    target: [20, 10],
    targetShape: [2, 1],
  },
}, "data namespace PyTorch-style samplers drive DataLoader order");
const dataSplitInput = new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4, 5, 6), [3, 2]);
const dataSplitTarget = new TensorDataSmokeTensor(Float32Array.of(10, 20, 30), [3, 1]);
const dataSplitSet = dataNs.tensorDataset(dataSplitInput, dataSplitTarget);
const dataSplitPair = dataNs.random_split(dataSplitSet, [1, 2], { shuffle: false });
expectSame({
  frozen: Object.isFrozen(dataSplitPair),
  firstLength: dataSplitPair[0].length,
  secondLength: dataSplitPair[1].__len__(),
  firstIndices: dataSplitPair[0].indices,
  secondIndices: dataSplitPair[1].indices,
  iterIndex: dataSplitPair[1].__iter__().next().value.index,
}, {
  frozen: true,
  firstLength: 1,
  secondLength: 2,
  firstIndices: [0],
  secondIndices: [1, 2],
  iterIndex: 1,
}, "data namespace random_split deterministic subset datasets");
const dataSubset = dataNs.subset(dataSplitSet, [2, 0]);
expectSame({
  kind: dataSubset.kind,
  length: dataSubset.length,
  indices: dataSubset.indices,
  firstIndex: dataSubset.sample(0).index,
  batchIndices: dataSubset.batch([0, 1]).indices,
}, {
  kind: "zgml.data.tensor-dataset",
  length: 2,
  indices: [2, 0],
  firstIndex: 2,
  batchIndices: [2, 0],
}, "data namespace subset preserves original row indices");
const dataTake = dataNs.take(dataSplitSet, 2);
expectSame({
  length: dataTake.length,
  indices: dataTake.indices,
  secondIndex: dataTake.__getitem__(1).index,
}, {
  length: 2,
  indices: [0, 1],
  secondIndex: 1,
}, "data namespace take returns prefix subset");
const dataConcat = dataNs.concatDataset([dataTake, dataSubset]);
expectSame({
  frozen: Object.isFrozen(dataConcat),
  kind: dataConcat.kind,
  length: dataConcat.length,
  offsets: dataConcat.offsets,
  len: dataConcat.__len__(),
  sourceIndex: dataConcat.sample(2).sourceIndex,
  source_index: dataConcat.sample(2).source_index,
  localIndex: dataConcat.sample(2).localIndex,
  local_index: dataConcat.sample(2).local_index,
  batchInputShape: dataConcat.batch([0, 2]).input.shape,
  batchIndices: dataConcat.batch([0, 2]).indices,
  batchSourceIndices: dataConcat.batch([0, 2]).source_indices,
  aliasInput: dataNs.concat_dataset([dataSubset, dataTake]).sample(0).input.data,
}, {
  frozen: true,
  kind: "zgml.data.concat-dataset",
  length: 4,
  offsets: [0, 2],
  len: 4,
  sourceIndex: 1,
  source_index: 1,
  localIndex: 0,
  local_index: 0,
  batchInputShape: [2, 2],
  batchIndices: [0, 2],
  batchSourceIndices: [0, 1],
  aliasInput: [5, 6],
}, "data namespace concatDataset composes datasets and batches");
const dataMapped = dataNs.mapDataset(dataSet, (sample) => ({
  input: tensorMath.mul(sample.input, 2),
  target: sample.target,
  sourceIndex: sample.index,
}));
expectSame({
  frozen: Object.isFrozen(dataMapped),
  kind: dataMapped.kind,
  length: dataMapped.__len__(),
  sourceIndex: dataMapped.at(1).sourceIndex,
  sampleInput: dataMapped.sample(1).input.data,
  batchShape: dataMapped.batch([0, 1]).input.shape,
  batchSource: dataMapped.batch([0, 1]).sourceIndex,
}, {
  frozen: true,
  kind: "zgml.data.mapped-dataset",
  length: 2,
  sourceIndex: 1,
  sampleInput: [6, 8],
  batchShape: [2, 2],
  batchSource: [0, 1],
}, "data namespace mapDataset transforms samples and batches");
expectSame(dataNs.map_dataset(dataSet, (sample) => ({ input: sample.input })).sample(0).input.data, [1, 2], "data namespace map_dataset alias");
const dataLoaderSmoke = dataNs.dataLoader(dataSet, { batch_size: 2, drop_last: false });
expectSame(Object.isFrozen(dataLoaderSmoke), true, "data namespace DataLoader frozen");
expectSame({
  kind: dataLoaderSmoke.kind,
  length: dataLoaderSmoke.length,
  sampleCount: dataLoaderSmoke.sampleCount,
  sample_count: dataLoaderSmoke.sample_count,
  batchSize: dataLoaderSmoke.batchSize,
  batchCount: dataLoaderSmoke.batchCount,
  batch_count: dataLoaderSmoke.batch_count,
  len: dataLoaderSmoke.len(),
  size: dataLoaderSmoke.size(),
}, {
  kind: "zgml.data.batches",
  length: 2,
  sampleCount: 2,
  sample_count: 2,
  batchSize: 2,
  batchCount: 1,
  batch_count: 1,
  len: 1,
  size: 1,
}, "data namespace DataLoader length helpers");
expectSame(Array.from(dataLoaderSmoke)[0].input.shape, [2, 2], "data namespace DataLoader batch shape");

const tensorView = createTensorViewHelpers({
  Tensor: TensorDataSmokeTensor,
  addTensorGrad: tensorData.addTensorGrad,
  isGradEnabled: () => true,
});
const viewTensor = new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4, 5, 6), [2, 3], { requiresGrad: true });
const reshaped = tensorView.reshape(viewTensor, [3, 2]);
expectSame({ data: reshaped.data, shape: reshaped.shape, requiresGrad: reshaped.requiresGrad }, {
  data: [1, 2, 3, 4, 5, 6],
  shape: [3, 2],
  requiresGrad: true,
}, "tensor view reshape");
reshaped._backward(Float32Array.of(1, 2, 3, 4, 5, 6));
expectSame(viewTensor.grad, [1, 2, 3, 4, 5, 6], "tensor view reshape backward");
viewTensor.grad.fill(0);
const transposed = tensorView.transpose(viewTensor);
expectSame({ data: transposed.data, shape: transposed.shape }, {
  data: [1, 4, 2, 5, 3, 6],
  shape: [3, 2],
}, "tensor view transpose");
transposed._backward(Float32Array.of(10, 20, 30, 40, 50, 60));
expectSame(viewTensor.grad, [10, 30, 50, 20, 40, 60], "tensor view transpose backward");
viewTensor.grad.fill(0);
const permuteTensor = new TensorDataSmokeTensor(Float32Array.from({ length: 12 }, (_, index) => index + 1), [2, 3, 2], { requiresGrad: true });
const permuted = tensorView.permute(permuteTensor, [1, 0, 2]);
expectSame({ data: permuted.data, shape: permuted.shape }, {
  data: [1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12],
  shape: [3, 2, 2],
}, "tensor view permute");
permuted._backward(Float32Array.from({ length: 12 }, (_, index) => (index + 1) * 10));
expectSame(permuteTensor.grad, [10, 20, 50, 60, 90, 100, 30, 40, 70, 80, 110, 120], "tensor view permute backward");
permuteTensor.grad.fill(0);
const flipped = tensorView.flip(permuteTensor, [0, -1]);
expectSame({ data: flipped.data, shape: flipped.shape }, {
  data: [8, 7, 10, 9, 12, 11, 2, 1, 4, 3, 6, 5],
  shape: [2, 3, 2],
}, "tensor view flip");
flipped._backward(Float32Array.from({ length: 12 }, (_, index) => index + 1));
expectSame(permuteTensor.grad, [8, 7, 10, 9, 12, 11, 2, 1, 4, 3, 6, 5], "tensor view flip backward");
permuteTensor.grad.fill(0);
const rolled = tensorView.roll(permuteTensor, [1, -1], [0, 2]);
expectSame({ data: rolled.data, shape: rolled.shape }, {
  data: [8, 7, 10, 9, 12, 11, 2, 1, 4, 3, 6, 5],
  shape: [2, 3, 2],
}, "tensor view roll dims");
rolled._backward(Float32Array.from({ length: 12 }, (_, index) => index + 1));
expectSame(permuteTensor.grad, [8, 7, 10, 9, 12, 11, 2, 1, 4, 3, 6, 5], "tensor view roll backward");
expectSame(tensorView.roll(viewTensor, 2).data, [5, 6, 1, 2, 3, 4], "tensor view roll flattened");
expectSame(tensorView.permute(new TensorDataSmokeTensor(Float32Array.from({ length: 120 }, (_, index) => index + 1), [2, 3, 4, 5]), [0, -1, 1, -2]).shape, [2, 5, 3, 4], "tensor view permute 4d mixed axes");
const selected = tensorView.select(viewTensor, 1, -1);
expectSame({ data: selected.data, shape: selected.shape }, {
  data: [3, 6],
  shape: [2],
}, "tensor view select negative");
selected._backward(Float32Array.of(7, 8));
expectSame(viewTensor.grad, [0, 0, 7, 0, 0, 8], "tensor view select backward");
viewTensor.grad.fill(0);
const sliced = tensorView.slice(viewTensor, 1, 0, null, 2);
expectSame({ data: sliced.data, shape: sliced.shape }, {
  data: [1, 3, 4, 6],
  shape: [2, 2],
}, "tensor view slice step");
sliced._backward(Float32Array.of(1, 2, 3, 4));
expectSame(viewTensor.grad, [1, 0, 2, 3, 0, 4], "tensor view slice backward");
viewTensor.grad.fill(0);
const splitPair = tensorView.split(viewTensor, [1, 2], 1);
expectSame(splitPair.map((part) => ({ data: part.data, shape: part.shape })), [
  { data: [1, 4], shape: [2, 1] },
  { data: [2, 3, 5, 6], shape: [2, 2] },
], "tensor view split sections");
splitPair[1]._backward(Float32Array.of(10, 20, 30, 40));
expectSame(viewTensor.grad, [0, 10, 20, 0, 30, 40], "tensor view split backward");
viewTensor.grad.fill(0);
const chunks = tensorView.chunk(viewTensor, 2, 1);
expectSame(chunks.map((part) => ({ data: part.data, shape: part.shape })), [
  { data: [1, 2, 4, 5], shape: [2, 2] },
  { data: [3, 6], shape: [2, 1] },
], "tensor view chunk uneven");
const unbound = tensorView.unbind(viewTensor, 0);
expectSame(unbound.map((part) => ({ data: part.data, shape: part.shape })), [
  { data: [1, 2, 3], shape: [3] },
  { data: [4, 5, 6], shape: [3] },
], "tensor view unbind");
const broadcastSource = new TensorDataSmokeTensor(Float32Array.of(2, 3), [1, 2], { requiresGrad: true });
const broadcasted = tensorView.broadcastTo(broadcastSource, [3, 2]);
expectSame({ data: broadcasted.data, shape: broadcasted.shape }, {
  data: [2, 3, 2, 3, 2, 3],
  shape: [3, 2],
}, "tensor view broadcastTo");
broadcasted._backward(Float32Array.of(1, 2, 3, 4, 5, 6));
expectSame(broadcastSource.grad, [9, 12], "tensor view broadcast backward");
const repeated = tensorView.repeat(broadcastSource, [2, 3]);
expectSame({ data: repeated.data, shape: repeated.shape }, {
  data: [2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
  shape: [2, 6],
}, "tensor view repeat");
repeated._backward(Float32Array.from({ length: 12 }, () => 1));
expectSame(broadcastSource.grad, [15, 18], "tensor view repeat backward");
expectSame(tensorView.tile(new TensorDataSmokeTensor(Float32Array.of(5, 6), [2]), [2, 2]).shape, [2, 4], "tensor view tile rank extension");
expectSame(tensorView.flatten(new TensorDataSmokeTensor(Float32Array.of(1, 2, 3, 4), [1, 2, 2]), 1).shape, [1, 4], "tensor view flatten");
expectSame(tensorView.flatten(new TensorDataSmokeTensor(Float32Array.from({ length: 24 }, (_value, index) => index + 1), [2, 3, 4])).shape, [24], "tensor view flatten 3d all");
expectSame(tensorView.flatten(new TensorDataSmokeTensor(Float32Array.from({ length: 120 }, (_value, index) => index + 1), [2, 3, 4, 5]), 1, -2).shape, [2, 12, 5], "tensor view flatten 4d middle");
const viewTensor3d = new TensorDataSmokeTensor(Float32Array.from({ length: 24 }, (_value, index) => index + 1), [2, 3, 4]);
expectSame({ data: tensorView.select(viewTensor3d, -1, 2).data, shape: tensorView.select(viewTensor3d, -1, 2).shape }, {
  data: [3, 7, 11, 15, 19, 23],
  shape: [2, 3],
}, "tensor view select 3d trailing");
expectSame({ data: tensorView.narrow(viewTensor3d, 1, 1, 2).data, shape: tensorView.narrow(viewTensor3d, 1, 1, 2).shape }, {
  data: [5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 21, 22, 23, 24],
  shape: [2, 2, 4],
}, "tensor view narrow 3d middle");
const transposed3d = tensorView.transpose(viewTensor3d, 0, -1);
expectSame({ data: transposed3d.data, shape: transposed3d.shape }, {
  data: [1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23, 4, 16, 8, 20, 12, 24],
  shape: [4, 3, 2],
}, "tensor view transpose 3d outer");
expectSame(tensorView.squeeze(new TensorDataSmokeTensor(Float32Array.of(1, 2), [1, 2, 1])).shape, [2], "tensor view squeeze");
expectSame(tensorView.squeeze(new TensorDataSmokeTensor(Float32Array.from({ length: 24 }, (_value, index) => index + 1), [2, 1, 3, 4]), 1).shape, [2, 3, 4], "tensor view squeeze 4d middle");
expectSame(tensorView.unsqueeze(new TensorDataSmokeTensor(Float32Array.of(1, 2), [2]), 0).shape, [1, 2], "tensor view unsqueeze");
expectSame(tensorView.unsqueeze(viewTensor3d, -2).shape, [2, 3, 1, 4], "tensor view unsqueeze 3d middle");
expectSame(tensorView.narrow(viewTensor, 1, 1, 2).data, [2, 3, 5, 6], "tensor view narrow");
expectThrow(
  () => tensorView.transpose(new TensorDataSmokeTensor(Float32Array.of(1), [1])),
  "transpose requires a tensor with rank >= 2",
  "tensor view transpose rejects rank one",
);
expectThrow(
  () => tensorView.permute(viewTensor, [0, 0]),
  "permute dims must be a permutation of tensor axes; duplicate axis 0",
  "tensor view permute rejects duplicate axes",
);
expectThrow(
  () => tensorView.flip(viewTensor, [1, -1]),
  "flip dims must be unique; duplicate axis 1",
  "tensor view flip rejects duplicate axes",
);
expectThrow(
  () => tensorView.roll(viewTensor, [1, 2], null),
  "roll shifts must have length 1 when dims is omitted",
  "tensor view roll rejects multi-shift without dims",
);
expectThrow(
  () => tensorView.roll(viewTensor, [1, 2], [1, -1]),
  "roll dims must be unique; duplicate axis 1",
  "tensor view roll rejects duplicate axes",
);
expectThrow(
  () => tensorView.split(viewTensor, [1, 1], 1),
  "split sections must sum to dimension length 3, got 2",
  "tensor view split rejects bad section sum",
);
expectThrow(
  () => tensorView.chunk(viewTensor, 0, 1),
  "chunk chunks must be a positive safe integer, got 0",
  "tensor view chunk rejects invalid chunks",
);
expectThrow(
  () => tensorView.slice(viewTensor, 1, 0, 3, 0),
  "slice step must be a positive safe integer, got 0",
  "tensor view slice rejects bad step",
);

const tensorIndex = createTensorIndexHelpers();
const indexTensor = { data: Float32Array.of(1, 2, 3, 4, 5, 6), shape: [2, 3] };
expectSame(tensorIndex.flatIndex(indexTensor, [1, -1], "index smoke"), 5, "tensor index flatIndex negative");
expectSame(tensorIndex.get(indexTensor, [0, 2]), 3, "tensor index get");
tensorIndex.set(indexTensor, [1, 0], 9);
expectSame(indexTensor.data, [1, 2, 3, 9, 5, 6], "tensor index set");
tensorIndex.setFlexible(indexTensor, 0, [1, 20]);
expectSame(indexTensor.data, [1, 20, 3, 9, 5, 6], "tensor index setFlexible varargs");
tensorIndex.setFlexible(indexTensor, [1, 2], [60]);
expectSame(indexTensor.data, [1, 20, 3, 9, 5, 60], "tensor index setFlexible array index");
expectSame(tensorIndex.toArray(indexTensor), [[1, 20, 3], [9, 5, 60]], "tensor index toArray");
expectThrow(
  () => tensorIndex.get(indexTensor, [2, 0]),
  "Tensor.get dim 0 index 2 is out of range for dimension length 2",
  "tensor index rejects out of range",
);

expectSame(Number(activation.geluScalar(0).toFixed(6)), 0, "activation geluScalar zero");
expectSame(Number(activation.geluScalar(1).toFixed(6)), 0.841192, "activation geluScalar one");
expectSame(Number(activation.geluDerivativeScalar(1).toFixed(6)), 1.082964, "activation gelu derivative");
expectSame(Number(activation.siluScalar(2).toFixed(6)), 1.761594, "activation siluScalar");
expectSame(Number(activation.siluDerivativeScalar(2).toFixed(6)), 1.090784, "activation silu derivative");
expectSame(Number(activation.sigmoidScalar(0).toFixed(6)), 0.5, "activation sigmoidScalar");
expectSame(Number(activation.sigmoidDerivativeScalar(0).toFixed(6)), 0.25, "activation sigmoid derivative");

expectSame(gradMode.isGradEnabled(), true, "grad mode initially enabled");
expectSame(gradMode.setGradEnabled(false), true, "grad mode set false returns previous");
expectSame(gradMode.isGradEnabled(), false, "grad mode disabled");
expectSame(gradMode.enableGrad(() => gradMode.isGradEnabled()), true, "enableGrad callback");
expectSame(gradMode.isGradEnabled(), false, "enableGrad restores previous disabled state");
expectSame(gradMode.noGrad(() => gradMode.isGradEnabled()), false, "noGrad callback");
expectThrow(
  () => gradMode.noGrad(() => {
    throw new Error("boom");
  }),
  "boom",
  "grad mode propagates callback error",
);
expectSame(gradMode.isGradEnabled(), false, "grad mode restores after callback error");
expectSame(gradMode.setGradEnabled(true), false, "grad mode restore true returns previous");
expectSame(gradMode.inferenceMode(() => gradMode.isGradEnabled()), false, "inferenceMode callback");
expectSame(gradMode.isGradEnabled(), true, "grad mode finally restored");
expectThrow(
  () => gradMode.noGrad(null),
  "grad mode helper requires a callback",
  "grad mode callback validation",
);

expectSame(token.tokenId("7"), 7, "token tokenId string");
expectSame(token.tokenWindowLength({ tokensLen: 2 }, 4, "prompt"), 2, "token window length");
expectSame(token.tokenWindow([1, 2, 3], { tokensLen: 2 }, "prompt"), { tokens: [1, 2], tokensLen: 2 }, "token window array");
const tokenCarrier = new Uint32Array([4, 5, 6]);
const tokenWindow = token.tokenWindow(tokenCarrier, { tokenLength: 2 }, "prompt");
expectSame(tokenWindow, { tokens: [4, 5, 6], tokensLen: 2 }, "token window Uint32Array");
if (tokenWindow.tokens !== tokenCarrier) throw new Error("token window must preserve Uint32Array storage");
expectSame(token.validateMaxTokens(3), undefined, "token validateMaxTokens");
const outputTokens = new Uint32Array([9, 8, 7]);
if (token.tokenOutputArray(outputTokens) !== outputTokens) throw new Error("token output array must return caller storage");
const normalizedSampleOptions = token.normalizeTokenSampleOptions({ top_k: 3, temperature: 0.5, seed: 11 });
expectSame(normalizedSampleOptions, {
  topK: 3,
  top_k: 3,
  temperature: 0.5,
  seed: 11,
  reserved: 0,
}, "token sample options");
expectSame(Object.isFrozen(normalizedSampleOptions), true, "token sample options frozen");
const sampleAbiFields = token.tokenSampleAbiFields(normalizedSampleOptions);
expectSame(sampleAbiFields, {
  topK: 3,
  top_k: 3,
  temperature: 0.5,
  seed: 11,
  reserved: 0,
}, "token sample ABI fields");
expectSame(Object.isFrozen(sampleAbiFields), true, "token sample ABI fields frozen");
expectSame(token.tokenSelectionResultFromAbiRecord({ token: 7, logit: 1.25 }), { token: 7, logit: 1.25 }, "token selection record");
const selectionWords = new BigUint64Array(1);
const selectionView = new DataView(selectionWords.buffer);
selectionView.setUint32(0, 8, true);
selectionView.setFloat32(4, -0.5, true);
expectSame(token.tokenSelectionResultFromAbiWords(selectionWords), { token: 8, logit: -0.5 }, "token selection words");
const generated = new Uint32Array([3, 4, 5]);
expectSame(token.tokenGenerateResultFromAbiRecord(generated, {
  tokens_generated: 2,
  last_token: 4,
  last_logit: 0.75,
}), { tokens: [3, 4], lastToken: 4, lastLogit: 0.75 }, "token generate record");
const generateWords = new BigUint64Array(2);
generateWords[0] = 2n;
const generateView = new DataView(generateWords.buffer);
generateView.setUint32(8, 5, true);
generateView.setFloat32(12, 0.125, true);
expectSame(token.tokenGenerateResultFromAbiWords(generated, generateWords), { tokens: [3, 4], lastToken: 5, lastLogit: 0.125 }, "token generate words");

expectThrow(() => token.tokenId(-1), "invalid token id: -1", "token invalid id");
expectThrow(() => token.tokenWindow([1], { tokensLen: 2 }, "prompt"), "prompt length 2 exceeds token container length 1", "token window too long");
expectThrow(() => token.validateMaxTokens(0), "maxTokens must be a positive safe integer, got 0", "token invalid maxTokens");
expectThrow(() => token.tokenOutputArray(new Uint32Array(0)), "generated token output must not be empty", "token empty output");
expectThrow(() => token.normalizeTokenSampleOptions({ topK: 0 }), "sample topK must be an integer in [1, 256], got 0", "token invalid topK");

const partialCompilerEvidence = Object.freeze({
  irSignature: "ir-a",
  kernelPlanSignature: "kernel-a",
  memoryLayoutSignature: "memory-a",
  parameterLayoutSignature: "",
  bufferLayoutSignature: "buffer-a",
});
expectSame(compilerSignatures.compilerSignaturesFromCompilerEvidence(partialCompilerEvidence), {
  ir: "ir-a",
  kernelPlan: "kernel-a",
  memoryLayout: "memory-a",
  parameterLayout: "",
  bufferLayout: "buffer-a",
}, "compiler signatures from flat evidence");
expectSame(
  compilerSignatures.compilerEvidenceSignature(partialCompilerEvidence, "kernelPlan"),
  "kernel-a",
  "compiler evidence signature field",
);
expectSame(compilerSignatures.programCompilerSignaturesFromCompileEvidence(partialCompilerEvidence), {
  ir: "ir-a",
  kernelPlan: "kernel-a",
  memoryLayout: "memory-a",
  parameterLayout: "",
  bufferLayout: "buffer-a",
}, "program compiler signatures allow empty parameter layout");
const completedCompilerEvidence = compilerSignatures.completeProgramCompileEvidence(partialCompilerEvidence);
expectSame(Object.isFrozen(completedCompilerEvidence), true, "completed compiler evidence frozen");
expectSame(completedCompilerEvidence.compilerSignatures, {
  ir: "ir-a",
  kernelPlan: "kernel-a",
  memoryLayout: "memory-a",
  parameterLayout: "",
  bufferLayout: "buffer-a",
}, "completed compiler evidence signatures");
expectSame(compilerSignatures.completeProgramCompileEvidence(completedCompilerEvidence), completedCompilerEvidence, "canonical compiler evidence identity");

const packedModuleDesc = moduleProgramDesc.packModuleProgramDesc({
  inputShape: [2, 3],
  ops: [
    { kind: 1, activation: 2, flags: 3, reserved: 4, a: 5, b: 6, c: 7, eps: 0.25 },
  ],
});
expectSame(packedModuleDesc.inputShape, [2n, 3n], "module Program packed input shape");
expectSame(packedModuleDesc.opCount, 1, "module Program packed op count");
const packedOpView = new DataView(packedModuleDesc.opWords.buffer);
expectSame(packedOpView.getUint32(0, true), 1, "module Program packed op kind");
expectSame(packedOpView.getUint32(4, true), 2, "module Program packed op activation");
expectSame(packedOpView.getUint32(8, true), 3, "module Program packed op flags");
expectSame(packedOpView.getUint32(12, true), 4, "module Program packed op reserved");
expectSame(packedOpView.getBigUint64(16, true), 5n, "module Program packed op a");
expectSame(packedOpView.getBigUint64(24, true), 6n, "module Program packed op b");
expectSame(packedOpView.getBigUint64(32, true), 7n, "module Program packed op c");
expectSame(packedOpView.getFloat64(40, true), 0.25, "module Program packed op eps");
expectSame(moduleProgramDesc.createModuleProgramDescPackerHelpers().pack({ inputShape: [1], ops: [] }), {
  inputShape: [1n],
  opWords: null,
  opCount: 0,
}, "module Program packer helper");
expectThrow(
  () => moduleProgramDesc.packModuleProgramDesc({ inputShape: [], ops: [] }),
  "module Program inputShape must be a non-empty shape array",
  "module Program empty input shape",
);
expectThrow(
  () => moduleProgramDesc.packModuleProgramDesc({ inputShape: [1], ops: [{ kind: -1 }] }),
  "module Program ops[0].kind must be a non-negative safe integer, got -1",
  "module Program invalid op kind",
);

const bindingTarget = {};
const bindingMetadataModule = {
  compileSupport(options) {
    return { options };
  },
};
moduleBindings.annotateModuleBindings(bindingTarget, bindingMetadataModule, { inputShape: [2, 3], outputShape: [4], tag: "x" });
const bindingDescriptor = Object.getOwnPropertyDescriptor(bindingTarget, moduleBindings.moduleBindingsSymbol);
if (!bindingDescriptor || bindingDescriptor.enumerable !== true || bindingDescriptor.configurable !== false || bindingDescriptor.writable !== false) {
  throw new Error("module bindings metadata descriptor must be enumerable, frozen, and non-configurable");
}
const bindingMetadata = moduleBindings.moduleBindingMetadata(bindingTarget);
expectSame(Object.isFrozen(bindingMetadata), true, "module bindings metadata frozen");
expectSame(Object.isFrozen(bindingMetadata.options), true, "module bindings options frozen");
expectSame(Object.isFrozen(bindingMetadata.options.inputShape), true, "module bindings inputShape frozen");
expectSame(bindingMetadata.options, { inputShape: [2, 3], outputShape: [4], tag: "x" }, "module bindings normalized options");
moduleBindings.annotateModuleBindings(bindingTarget, { compileSupport: () => ({ changed: true }) }, { inputShape: [9] });
expectSame(moduleBindings.moduleBindingMetadata(bindingTarget), bindingMetadata, "module bindings immutable annotation");
moduleBindings.validateProgramBindModuleCompatibility({
  moduleCompatibility(module, options) {
    return module === bindingMetadataModule && options === bindingMetadata.options
      ? { compatible: true }
      : { compatible: false, reason: "unexpected metadata" };
  },
}, bindingTarget);
expectThrow(
  () => moduleBindings.validateProgramBindModuleCompatibility({
    moduleCompatibility() {
      return { compatible: false, reason: "shape mismatch" };
    },
  }, bindingTarget),
  "Program.bind ModuleBindings are not compatible with this Program: shape mismatch",
  "module bindings compatibility error",
);

class SessionSmokeTensor {
  constructor(values, shape, options = {}) {
    this.data = values;
    this.shape = shape;
    this.options = options;
  }
}
const sessionTensorHelpers = sessionTensor.createSessionTensorHelpers({ getTensorClass: () => SessionSmokeTensor });
expectSame(sessionTensor.programInputShape({ inputShape: [2, 3], inputLen: 6, outputLen: 1 }), [2, 3], "session tensor programInputShape");
expectSame(sessionTensor.programOutputShape({ inputLen: 1, outputLen: 4 }), [4], "session tensor programOutputShape fallback");
expectSame(sessionTensor.sessionOutputTensorShape({ outputShape: [2, 2] }, 4), [2, 2], "session tensor output shape desc");
expectThrow(
  () => sessionTensor.sessionOutputTensorShape({ outputShape: [2, 2] }, 3, undefined, "read tensor"),
  "read tensor has 4 elements, expected 3",
  "session tensor output shape mismatch",
);
const tensorOutput = new SessionSmokeTensor(new Float32Array([1, 2]), [2]);
expectSame(sessionTensorHelpers.outputTarget({ output: tensorOutput }), [1, 2], "session tensor output target tensor");
expectThrow(
  () => sessionTensorHelpers.outputTarget({ output: false }, "tensor output"),
  "tensor output require logits output",
  "session tensor output false",
);
const madeTensor = sessionTensorHelpers.outputTensorForSession({ outputShape: [2] }, new Float32Array([3, 4]), {}, undefined);
expectSame(madeTensor instanceof SessionSmokeTensor, true, "session tensor constructed Tensor");
expectSame(madeTensor.shape, [2], "session tensor constructed shape");
const readTarget = new Float32Array(2);
expectSame(sessionTensorHelpers.readOutputInto(new Float32Array([0, 5, 6]), readTarget, 2, 4), [5, 6], "session tensor read f32 range");
let nativeReadArgs = null;
const nativeReadTarget = new Float32Array(1);
sessionTensorHelpers.readOutputInto({
  readFloat32Into(target, length, byteOffset) {
    nativeReadArgs = { target, length, byteOffset };
    target[0] = 9;
    return target;
  },
}, nativeReadTarget, 1, 0);
expectSame(nativeReadArgs.length, 1, "session tensor native read length");
expectSame(nativeReadTarget, [9], "session tensor native read target");
expectThrow(
  () => sessionTensorHelpers.readOutputInto(new Float32Array(1), new Float32Array(1), 1, 2),
  "Session.readOutputInto byteOffset must be aligned to f32",
  "session tensor read alignment",
);

const stepParamsAccepted = stepParams.stepParamsCompatibilityResult({
  kind: "llama-token",
  signature: "kind=llama-token|position=3",
  position: 3,
}, null, {
  stateEffect: "advance",
  allocationFree: true,
  inputSource: "token",
  outputTarget: "inline",
  inputElementType: "token-u32",
  outputElementType: "f32",
  inputElementLength: 1,
  outputElementLength: 5,
  inputShape: [1],
  outputShape: [5],
  inputByteLength: 4,
  outputByteLength: 20,
});
expectSame(stepParamsAccepted.accepted, true, "step params accepted");
expectSame(stepParamsAccepted.kind, "zgml.step-params.compatibility", "step params compatibility kind");
expectSame(stepParamsAccepted.hotPath, true, "step params hot path");
expectSame(stepParamsAccepted.inputOwnership, "caller", "step params input ownership");
expectSame(stepParamsAccepted.outputReturnOwnership, "caller", "step params output return ownership");
expectSame(stepParamsAccepted.stepParamsSignature.includes("inputShape=1|outputShape=5"), true, "step params signature shapes");
expectSame(Object.isFrozen(stepParamsAccepted.outputShape), true, "step params accepted output shape frozen");
expectSame(stepParams.isStepParamsCompatibility(stepParamsAccepted), true, "step params compatibility evidence predicate accepts accepted evidence");
expectSame(stepParams.requireStepParamsCompatibility(stepParamsAccepted), stepParamsAccepted, "step params require compatibility returns evidence");
expectSame(stepParams.assertStepParamsCompatibility(stepParamsAccepted), stepParamsAccepted, "step params assert compatibility alias");
expectSame(stepParams.assert_step_params_compatibility(stepParamsAccepted), stepParamsAccepted, "step params assert compatibility snake alias");
expectSame(stepParams.acceptsStepParamsCompatibility(stepParamsAccepted), true, "step params accepts predicate");
expectSame(stepParams.canExecuteStepParamsCompatibility(stepParamsAccepted), true, "step params can execute predicate");
expectSame(stepParams.requireCanExecuteStepParamsCompatibility(stepParamsAccepted), stepParamsAccepted, "step params require can-execute returns evidence");
expectSame(stepParams.acceptsAllocationFreeStepParamsCompatibility(stepParamsAccepted), true, "step params allocation-free predicate");
expectSame(stepParams.acceptsRuntimeOutputAllocationFreeStepParamsCompatibility(stepParamsAccepted), true, "step params runtime allocation-free predicate");
expectSame(stepParams.acceptsNoReadbackStepParamsCompatibility(stepParamsAccepted), true, "step params no readback predicate");
expectSame(stepParams.acceptsReadbackFreeStepParamsCompatibility(stepParamsAccepted), true, "step params readback-free predicate");
expectSame(stepParams.acceptsHotStepParamsCompatibility(stepParamsAccepted), true, "step params hot predicate");
expectSame(stepParams.requireAllocationFreeStepParamsCompatibility(stepParamsAccepted), stepParamsAccepted, "step params require allocation-free returns evidence");
expectSame(stepParams.requireRuntimeOutputAllocationFreeStepParamsCompatibility(stepParamsAccepted), stepParamsAccepted, "step params require runtime allocation-free returns evidence");
expectSame(stepParams.requireNoReadbackStepParamsCompatibility(stepParamsAccepted), stepParamsAccepted, "step params require no-readback returns evidence");
expectSame(stepParams.requireReadbackFreeStepParamsCompatibility(stepParamsAccepted), stepParamsAccepted, "step params require readback-free returns evidence");
expectSame(stepParams.requireHotStepParamsCompatibility(stepParamsAccepted), stepParamsAccepted, "step params require hot returns evidence");
expectSame(stepParams.matchesStepContractSignature({ signature: "contract-a" }, "contract-a"), true, "step params contract signature predicate");
expectSame(stepParams.matchesStepParamsSignature(stepParamsAccepted, stepParamsAccepted.stepParamsSignature), true, "step params signature predicate");
expectSame(stepParams.matchesStepParamsCompatibility(stepParamsAccepted, stepParamsAccepted), true, "step params compatibility evidence predicate");
expectSame(stepParams.matchesStepParamsCompatibility(stepParamsAccepted, { stepParamsSignature: stepParamsAccepted.stepParamsSignature }), false, "step params compatibility rejects bare signature object");
const stepParamsRequireHotRejected = stepParams.stepParamsCompatibilityResult(
  { kind: "rejected-step", signature: "kind=rejected-step" },
  stepParams.stepParamsValidationError("invalid-input", "bad input"),
);
expectSame(stepParams.isStepParamsCompatibility(stepParamsRequireHotRejected), true, "step params compatibility evidence predicate accepts rejected evidence");
expectSame(stepParams.acceptsStepParamsCompatibility(stepParamsRequireHotRejected), false, "step params accepts predicate rejects non-executable evidence");
expectSame(stepParams.requireStepParamsCompatibility(stepParamsRequireHotRejected), stepParamsRequireHotRejected, "step params require compatibility accepts rejected evidence");
expectSame(stepParams.matchesStepParamsCompatibility(stepParamsAccepted, stepParamsRequireHotRejected), false, "step params compatibility mismatch detects different evidence");
expectThrow(
  () => stepParams.requireStepParamsCompatibility({ ...stepParamsAccepted, outputShape: stepParamsAccepted.outputShape.slice() }),
  "StepParamsCompatibility is not valid evidence: outputShape is not frozen evidence",
  "step params require compatibility rejects mutable output shape",
);
expectThrow(
  () => stepParams.requireCanExecuteStepParamsCompatibility(stepParamsRequireHotRejected),
  "StepParams are not executable: status=rejected",
  "step params require can-execute rejects non-executable evidence",
);
expectThrow(
  () => stepParams.requireAllocationFreeStepParamsCompatibility(stepParamsRequireHotRejected),
  "StepParams are not allocation-free compatible: status=rejected allocationFree=false",
  "step params require allocation-free rejects non-allocation-free evidence",
);
expectThrow(
  () => stepParams.requireRuntimeOutputAllocationFreeStepParamsCompatibility(stepParamsRequireHotRejected),
  "StepParams are not runtime-output-allocation-free compatible: status=rejected runtimeOutputAllocationFree=false",
  "step params require runtime allocation-free rejects allocating evidence",
);
expectThrow(
  () => stepParams.requireNoReadbackStepParamsCompatibility({
    ...stepParamsAccepted,
    readbackRequired: true,
  }),
  "StepParams are not no-readback compatible: status=accepted readbackRequired=true",
  "step params require no-readback rejects readback evidence",
);
expectThrow(
  () => stepParams.requireReadbackFreeStepParamsCompatibility(stepParamsRequireHotRejected),
  "StepParams are not readback-free compatible: status=rejected readbackFree=false",
  "step params require readback-free rejects readback evidence",
);
expectThrow(
  () => stepParams.requireHotStepParamsCompatibility(stepParamsRequireHotRejected),
  "StepParams are not hot-path compatible: status=rejected blockers=rejected rejection=invalid-input",
  "step params require hot rejects non-hot evidence",
);
const sessionFacadeContract = {
  kind: "facade",
  signature: "kind=facade|position=7",
  position: 7,
};
const sessionFacadeCompatibility = stepParams.stepParamsCompatibilityResult(sessionFacadeContract, null, {
  stateEffect: "advance",
  allocationFree: true,
  inputSource: "bound-host",
  outputTarget: "bound-host",
  inputElementType: "f32",
  outputElementType: "f32",
  inputElementLength: 2,
  outputElementLength: 2,
  inputShape: [2],
  outputShape: [2],
  inputByteLength: 8,
  outputByteLength: 8,
});
const sessionFacadeCalls = [];
const sessionFacadeHelpers = sessionFacade.createSessionStepParamsFacadeHelpers({
  stepContract(session) {
    return session.contract;
  },
  stepParamsCompatibility(session, params) {
    sessionFacadeCalls.push({ session, params });
    return session.compatibility;
  },
});
const sessionFacadeSession = {
  contract: sessionFacadeContract,
  compatibility: sessionFacadeCompatibility,
};
expectSame(Object.isFrozen(sessionFacadeHelpers), true, "session facade helpers frozen");
expectSame(sessionFacadeHelpers.preflightStepParams(sessionFacadeSession, { input: "x" }), sessionFacadeCompatibility, "session facade preflight alias");
expectSame(sessionFacadeHelpers.acceptsStepParams(sessionFacadeSession, {}), true, "session facade accepts predicate");
expectSame(sessionFacadeHelpers.canExecuteStepParams(sessionFacadeSession, {}), true, "session facade can execute predicate");
expectSame(sessionFacadeHelpers.requireCanExecuteStepParams(sessionFacadeSession, {}), sessionFacadeCompatibility, "session facade require can-execute evidence");
expectSame(sessionFacadeHelpers.acceptsAllocationFreeStepParams(sessionFacadeSession, {}), true, "session facade allocation-free predicate");
expectSame(sessionFacadeHelpers.acceptsRuntimeOutputAllocationFreeStepParams(sessionFacadeSession, {}), true, "session facade runtime allocation-free predicate");
expectSame(sessionFacadeHelpers.acceptsNoReadbackStepParams(sessionFacadeSession, {}), true, "session facade no readback predicate");
expectSame(sessionFacadeHelpers.acceptsReadbackFreeStepParams(sessionFacadeSession, {}), true, "session facade readback-free predicate");
expectSame(sessionFacadeHelpers.acceptsHotStepParams(sessionFacadeSession, {}), true, "session facade hot predicate");
expectSame(sessionFacadeHelpers.requireAllocationFreeStepParams(sessionFacadeSession, {}), sessionFacadeCompatibility, "session facade require allocation-free evidence");
expectSame(sessionFacadeHelpers.requireRuntimeOutputAllocationFreeStepParams(sessionFacadeSession, {}), sessionFacadeCompatibility, "session facade require runtime allocation-free evidence");
expectSame(sessionFacadeHelpers.requireNoReadbackStepParams(sessionFacadeSession, {}), sessionFacadeCompatibility, "session facade require no-readback evidence");
expectSame(sessionFacadeHelpers.requireReadbackFreeStepParams(sessionFacadeSession, {}), sessionFacadeCompatibility, "session facade require readback-free evidence");
expectSame(sessionFacadeHelpers.requireHotStepParams(sessionFacadeSession, {}), sessionFacadeCompatibility, "session facade require hot evidence");
expectSame(sessionFacadeHelpers.matchesStepContractSignature(sessionFacadeSession, sessionFacadeContract.signature), true, "session facade contract signature match");
expectSame(sessionFacadeHelpers.matchesStepParamsSignature(sessionFacadeSession, {}, sessionFacadeCompatibility.stepParamsSignature), true, "session facade params signature match");
expectSame(sessionFacadeHelpers.matchesStepParamsCompatibility(sessionFacadeSession, {}, sessionFacadeCompatibility), true, "session facade params compatibility match");
const sessionFacadeExecutionPlan = sessionFacadeHelpers.executionPlan(sessionFacadeSession, {});
expectSame(sessionFacadeExecutionPlan.kind, "zgml.session.execution-plan", "session facade execution plan kind");
expectSame(sessionFacadeExecutionPlan.signature.startsWith("session-execution-plan|contract=kind=facade|position=7|"), true, "session facade execution plan signature prefix");
expectSame(sessionFacadeExecutionPlan.signature.includes(`stepParams=${sessionFacadeCompatibility.stepParamsSignature}`), true, "session facade execution plan step params signature");
expectSame(sessionFacadeHelpers.requireExecutionPlan(sessionFacadeSession, {}).signature, sessionFacadeExecutionPlan.signature, "session facade require execution plan accepts executable params");
expectSame(executionPlan.acceptsSessionExecutionPlan(sessionFacadeExecutionPlan), true, "execution plan helper accepts executable Session plan");
expectSame(executionPlan.requireSessionExecutionPlan(sessionFacadeExecutionPlan), sessionFacadeExecutionPlan, "execution plan helper requires executable Session plan");
expectSame(executionPlan.assertSessionExecutionPlan(sessionFacadeExecutionPlan), sessionFacadeExecutionPlan, "execution plan helper assert Session alias");
expectSame(executionPlan.assert_session_execution_plan(sessionFacadeExecutionPlan), sessionFacadeExecutionPlan, "execution plan helper assert Session snake alias");
expectSame(executionPlan.matchesSessionExecutionPlanSignature(sessionFacadeExecutionPlan, sessionFacadeExecutionPlan.signature), true, "execution plan helper matches Session signature");
expectSame(sessionFacadeExecutionPlan.signature, sessionFacadeHelpers.hotPathPlan(sessionFacadeSession, {}).signature, "session facade hot path plan signature");
expectThrow(
  () => executionPlan.requireSessionExecutionPlan({ ...sessionFacadeExecutionPlan, canExecute: false, compatibility: stepParamsRequireHotRejected }),
  "StepParams are not executable: status=rejected",
  "execution plan helper rejects non-executable Session plan",
);
expectSame(sessionFacadeCalls.length, 19, "session facade compatibility call count");
expectThrow(
  () => sessionFacade.createSessionStepParamsFacadeHelpers({ stepContract() {} }),
  "createSessionStepParamsFacadeHelpers requires stepParamsCompatibility and stepContract callbacks",
  "session facade requires callbacks",
);

const genericFamilyRequireCalls = [];
expectSame(genericModelDesc.genericModelDescriptorManifest.policyOwner, "src/ts/runtime/generic_model_desc.ts", "generic model descriptor runtime manifest owner");
expectSame(genericModelDesc.genericModelNativeDesc({ inputLen: 2, outputLen: 1, activation: 7 }, {
  tinyLinearKind: 11,
  tinyMlpKind: 12,
}), {
  kind: 11,
  activation: 0,
  input_len: 2,
  output_len: 1,
  hidden_len: 0,
}, "generic model descriptor normalizes tiny linear native ABI record");
expectSame(genericModelDesc.genericModelNativeDesc({ inputLen: 2, outputLen: 1, hiddenLen: 3, activation: 7 }, {
  tinyLinearKind: 11,
  tinyMlpKind: 12,
}), {
  kind: 12,
  activation: 7,
  input_len: 2,
  output_len: 1,
  hidden_len: 3,
}, "generic model descriptor normalizes tiny mlp native ABI record");
expectSame(Object.isFrozen(genericModelDesc.genericModelNativeDesc({ inputLen: 2, outputLen: 1 }, {
  tinyLinearKind: 11,
  tinyMlpKind: 12,
})), true, "generic model descriptor evidence is frozen");
const genericModelRequirementAccessorCalls = [];
const genericModelRequirements = genericModelDesc.genericModelProgramRequirements({ inputLen: 2, outputLen: 3, weightsLen: 5, biasLen: 1 }, {
  packedProgramWeightsLen(desc) {
    genericModelRequirementAccessorCalls.push(["weights", desc.weightsLen]);
    return desc.weightsLen;
  },
  packedProgramBiasLen(desc) {
    genericModelRequirementAccessorCalls.push(["bias", desc.biasLen]);
    return desc.biasLen;
  },
});
expectSame(genericModelRequirements, {
  scalarBytes: 4,
  inputLen: 2,
  inputByteLength: 8,
  outputLen: 3,
  outputByteLength: 12,
  weightsLen: 5,
  weightsByteLength: 20,
  biasLen: 1,
  biasByteLength: 4,
  parameterLen: 6,
  parameterByteLength: 24,
}, "generic model descriptor builds Program requirements");
expectSame(Object.isFrozen(genericModelRequirements), true, "generic model Program requirements are frozen");
expectSame(genericModelRequirementAccessorCalls, [["weights", 5], ["bias", 1]], "generic model Program requirements use packed length accessors once");
const bindWeights = { length: 4, handle: 101 };
const bindBias = { length: 2, handle: 102 };
const bindInput = { length: 3, handle: 103 };
const bindOutput = { length: 5, handle: 104 };
const bindDescriptorFields = programBindDesc.programBindDescriptorFields({
  weights: bindWeights,
  bias: bindBias,
  input: bindInput,
  output: bindOutput,
});
expectSame(bindDescriptorFields, {
  weights: bindWeights,
  weightsLen: 4,
  bias: bindBias,
  biasLen: 2,
  input: bindInput,
  inputLen: 3,
  output: bindOutput,
  outputLen: 5,
}, "program bind descriptor fields normalize values and lengths");
expectSame(Object.isFrozen(bindDescriptorFields), true, "program bind descriptor fields are frozen");
expectSame(programBindDesc.programBindDescriptorRecord(bindDescriptorFields, (value) => value ? value.handle : null), {
  weights: 101,
  weights_len: 4,
  bias: 102,
  bias_len: 2,
  input: 103,
  input_len: 3,
  output: 104,
  output_len: 5,
}, "program bind descriptor record projects native handles");
expectSame(Object.isFrozen(programBindDesc.programBindDescriptorRecord(bindDescriptorFields)), true, "program bind descriptor record is frozen");
const outputOnlyBindFields = programBindDesc.programOutputBindDescriptorFields({ handle: 777 }, 13);
expectSame(outputOnlyBindFields, {
  weights: null,
  weightsLen: 0,
  bias: null,
  biasLen: 0,
  input: null,
  inputLen: 0,
  output: { handle: 777 },
  outputLen: 13,
}, "program bind descriptor output-only fields preserve explicit length");
expectSame(programBindDesc.programBindDescriptorRecord(outputOnlyBindFields, (value) => value ? value.handle : null), {
  weights: null,
  weights_len: 0,
  bias: null,
  bias_len: 0,
  input: null,
  input_len: 0,
  output: 777,
  output_len: 13,
}, "program bind descriptor output-only record");
expectSame(programBindDesc.programOutputBindDescriptorFields(null, 13).outputLen, 0, "program bind descriptor output-only null length");
const deviceImportFields = programBufferDesc.programDeviceBufferImportDescriptorFields({
  placement: 2,
  deviceHandle: 301,
  bufferHandle: 302,
  byteOffset: 16,
  byteLength: 64,
});
expectSame(deviceImportFields, {
  placement: 2,
  reserved: 0,
  deviceHandle: 301,
  bufferHandle: 302,
  byteOffset: 16,
  byteLength: 64,
}, "program buffer import descriptor fields");
expectSame(Object.isFrozen(deviceImportFields), true, "program buffer import descriptor fields frozen");
expectSame(programBufferDesc.programDeviceBufferImportDescriptorRecord(deviceImportFields), {
  placement: 2,
  reserved: 0,
  device_handle: 301,
  buffer_handle: 302,
  byte_offset: 16,
  byte_len: 64,
}, "program buffer import descriptor record");
expectSame(Object.isFrozen(programBufferDesc.programDeviceBufferImportDescriptorRecord(deviceImportFields)), true, "program buffer import descriptor record frozen");
let llamaNativeReadbackLength = null;
expectSame(llamaTokenOutput.llamaTokenOutputManifest.policyOwner, "src/ts/runtime/llama_token_output.ts", "llama token output runtime manifest owner");
expectSame(llamaKvCache.llamaKvCacheManifest.policyOwner, "src/ts/runtime/llama_kv_cache.ts", "llama kv-cache runtime manifest owner");
expectSame(llamaSessionBindDesc.llamaSessionBindDescriptorManifest.policyOwner, "src/ts/runtime/llama_session_bind_desc.ts", "llama session bind descriptor runtime manifest owner");
expectSame(moduleProgramEvidence.moduleProgramEvidenceManifest.policyOwner, "src/ts/runtime/module_program_evidence.ts", "module Program evidence runtime manifest owner");
expectSame(moduleProgramEvidence.attachProgramCompileEvidence({ keep: true }, { evidence: true }), { keep: true }, "module Program evidence no hook leaves program");
expectSame(
  moduleProgramEvidence.attachProgramCompileEvidence(
    { _withCompileEvidence(evidence) { return { evidence }; } },
    { ok: true },
  ),
  { evidence: { ok: true } },
  "module Program evidence attaches through retained Program hook",
);
expectSame(nativeAbiConstants.nativeAbiConstantsManifest.policyOwner, "src/ts/runtime/native_abi_constants.ts", "native ABI constants runtime manifest owner");
expectSame(nativeStatus.nativeStatusManifest.policyOwner, "src/ts/runtime/native_status.ts", "native status runtime manifest owner");
expectThrow(
  () => nativeStatus.checkStatusOk(7, (code) => `status-${code}`),
  "zgml status-7 (7)",
  "native status check error",
);
const llamaNativeOutput = {
  readFloat32(length) {
    llamaNativeReadbackLength = length;
    return new Float32Array([9, 8, 7]).subarray(0, length);
  },
};
const llamaTokenNativeTarget = llamaTokenOutput.llamaTokenStepOutputTarget(
  undefined,
  llamaNativeOutput,
  4,
  (value) => value === llamaNativeOutput,
);
expectSame(llamaTokenNativeTarget, {
  nativeOutput: llamaNativeOutput,
  output: null,
  descOutput: null,
  useNativeOutput: true,
}, "llama token output target uses bound native output");
expectSame(Object.isFrozen(llamaTokenNativeTarget), true, "llama token output target is frozen");
expectSame(llamaTokenOutput.llamaTokenStepOutputResult(2, llamaTokenNativeTarget), [9, 8], "llama token output reads native output");
expectSame(llamaNativeReadbackLength, 2, "llama token output readback length");
const llamaProvidedOutput = new Float32Array([1, 2, 3]);
const llamaTokenProvidedTarget = llamaTokenOutput.llamaTokenStepOutputTarget(
  llamaProvidedOutput,
  llamaNativeOutput,
  4,
  (value) => value === llamaNativeOutput,
);
expectSame(llamaTokenProvidedTarget.descOutput === llamaProvidedOutput, true, "llama token output prefers explicit host output");
expectSame(llamaTokenOutput.llamaTokenStepOutputResult(2, llamaTokenProvidedTarget), [1, 2], "llama token output slices explicit host output");
const llamaTokenAllocatedTarget = llamaTokenOutput.llamaTokenStepOutputTarget(
  undefined,
  null,
  3,
  () => false,
);
expectSame(llamaTokenAllocatedTarget.output.length, 3, "llama token output allocates host logits");
expectThrow(
  () => llamaTokenOutput.llamaTokenStepOutputResult(1, {
    nativeOutput: null,
    output: null,
    descOutput: null,
    useNativeOutput: false,
  }),
  "LLaMA token step did not produce a host output buffer",
  "llama token output rejects missing host output",
);
const genericFamilyHotCompatibility = stepParams.stepParamsCompatibilityResult({
  kind: "session-step-contract",
  signature: "generic-family-hot",
}, null, {
  allocationFree: true,
  inputSource: "inline",
  outputTarget: "inline",
  inputElementType: "f32",
  outputElementType: "f32",
  inputElementLength: 2,
  outputElementLength: 1,
  inputShape: [2],
  outputShape: [1],
  inputByteLength: 8,
  outputByteLength: 4,
});
const genericFamilySurface = createGenericFamilySurface({
  createTinyLinearModelHandle: () => 1,
  createTinyMlpModelHandle: () => 2,
  modelFacadePolicy: {},
  genericProgramFacade: {
    withCompileEvidence(program, evidence) {
      program.compileEvidenceSnapshot = evidence;
      return program;
    },
  },
  genericSessionFacade: {
    requireCanExecuteStepParams(session, params) {
      genericFamilyRequireCalls.push({ method: "requireCanExecuteStepParams", session, params });
      return genericFamilyHotCompatibility;
    },
    requireAllocationFreeStepParams(session, params) {
      genericFamilyRequireCalls.push({ method: "requireAllocationFreeStepParams", session, params });
      return genericFamilyHotCompatibility;
    },
    requireRuntimeOutputAllocationFreeStepParams(session, params) {
      genericFamilyRequireCalls.push({ method: "requireRuntimeOutputAllocationFreeStepParams", session, params });
      return genericFamilyHotCompatibility;
    },
    requireNoReadbackStepParams(session, params) {
      genericFamilyRequireCalls.push({ method: "requireNoReadbackStepParams", session, params });
      return genericFamilyHotCompatibility;
    },
    requireReadbackFreeStepParams(session, params) {
      genericFamilyRequireCalls.push({ method: "requireReadbackFreeStepParams", session, params });
      return genericFamilyHotCompatibility;
    },
    requireHotStepParams(session, params) {
      genericFamilyRequireCalls.push({ method: "requireHotStepParams", session, params });
      return genericFamilyHotCompatibility;
    },
  },
  bindProgram: () => null,
  packedProgramWeightsLen: () => 0,
  packedProgramBiasLen: () => 0,
  programBufferLayoutFromRequirements: () => ({ slots: [] }),
});
expectSame(genericFamilySurfacePolicy.genericFamilySurfaceManifest.policyOwner, "src/ts/runtime/generic_family_surface.ts", "generic family surface runtime manifest owner");
expectSame(hostAdapterSurfaces.hostAdapterSurfacesManifest.policyOwner, "src/ts/runtime/host_adapter_surfaces.ts", "host adapter surfaces runtime manifest owner");
expectSame(llamaFamilySurfacePolicy.llamaFamilySurfaceManifest.policyOwner, "src/ts/runtime/llama_family_surface.ts", "llama family surface runtime manifest owner");
expectSame(programBindingSurface.programBindingSurfaceManifest.policyOwner, "src/ts/runtime/program_binding_surface.ts", "program binding surface runtime manifest owner");
expectSame(tensorHostSurface.tensorHostSurfaceManifest.policyOwner, "src/ts/core/tensor_host_surface.ts", "tensor host surface core manifest owner");
const llamaFamilySurfaceCalls = [];
function llamaFamilySurfaceCallback(name) {
  return (...args) => {
    llamaFamilySurfaceCalls.push({ name, args });
    return { name, args };
  };
}
const llamaFamilySurfaceStubFacade = new Proxy({}, {
  get(_target, property) {
    return llamaFamilySurfaceCallback(String(property));
  },
});
const llamaFamilyRuntimeSurface = llamaFamilySurfacePolicy.createLlamaFamilySurface(new Proxy({}, {
  get(_target, property) {
    if (property === "llamaModelFamilyFacade" || property === "llamaProgramFacade") return llamaFamilySurfaceStubFacade;
    if (property === "defaultLlamaSessionBufferLayout") return (vocabSize, bufferLayout) => ({ vocabSize, bufferLayout });
    if (property === "createLlamaSessionScratch") return () => ({});
    if (property === "assertSessionAlive") return (handle) => {
      if (handle === 0) throw new Error("dead llama session");
    };
    return llamaFamilySurfaceCallback(String(property));
  },
}));
const llamaFamilyRuntimeSession = new llamaFamilyRuntimeSurface.TinyLlamaSession(99, 8);
for (const method of [
  "advance_tokens",
  "execute_tokens",
  "execute_tensor",
  "execute_into",
  "prefill_tensor",
  "prefill_into",
  "step_argmax",
  "step_sample",
  "execute_tokens_argmax",
  "execute_tokens_sample",
  "generate_tokens_argmax",
  "generate_tokens_argmax_into",
  "generate_tokens_sample",
  "generate_tokens_sample_into",
  "argmax_token",
  "sample_token",
]) {
  expectSame(typeof llamaFamilyRuntimeSession[method], "function", `llama family Session snake alias ${method} exists`);
}
expectSame(typeof llamaFamilyRuntimeSession.execute_tensor, "function", "llama family Session snake alias execute_tensor exists");
expectSame(llamaFamilyRuntimeSession.advance_tokens([1, 2], { tokensLen: 2 }).name, "advanceLlamaTokens", "llama family advance_tokens delegates");
expectSame(llamaFamilyRuntimeSession.execute_tokens([1], { tokensLen: 1 }).name, "executeLlamaTokens", "llama family execute_tokens delegates");
expectSame(llamaFamilyRuntimeSession.execute_tensor({ token: 1 }).name, "executeLlamaTensor", "llama family execute_tensor delegates");
expectSame(llamaFamilyRuntimeSession.execute_into(new Float32Array(8), { token: 1 }).name, "executeLlamaInto", "llama family execute_into delegates");
expectSame(llamaFamilyRuntimeSession.prefill_tensor([1], { tokensLen: 1 }).name, "prefillLlamaTensor", "llama family prefill_tensor delegates");
expectSame(llamaFamilyRuntimeSession.prefill_into(new Float32Array(8), [1], { tokensLen: 1 }).name, "prefillLlamaInto", "llama family prefill_into delegates");
expectSame(llamaFamilyRuntimeSession.step_argmax(1).name, "stepLlamaArgmax", "llama family step_argmax delegates");
expectSame(llamaFamilyRuntimeSession.step_sample(1, { topK: 1 }).name, "stepLlamaSample", "llama family step_sample delegates");
expectSame(llamaFamilyRuntimeSession.execute_tokens_argmax([1], { tokensLen: 1 }).name, "executeLlamaTokensArgmax", "llama family execute_tokens_argmax delegates");
expectSame(llamaFamilyRuntimeSession.execute_tokens_sample([1], { tokensLen: 1 }).name, "executeLlamaTokensSample", "llama family execute_tokens_sample delegates");
expectSame(llamaFamilyRuntimeSession.generate_tokens_argmax([1], 1).name, "generateLlamaTokensArgmax", "llama family generate_tokens_argmax delegates");
expectSame(llamaFamilyRuntimeSession.generate_tokens_argmax_into([1], new Uint32Array(1)).name, "generateLlamaTokensArgmaxInto", "llama family generate_tokens_argmax_into delegates");
expectSame(llamaFamilyRuntimeSession.generate_tokens_sample([1], 1, { topK: 1 }).name, "generateLlamaTokensSample", "llama family generate_tokens_sample delegates");
expectSame(llamaFamilyRuntimeSession.generate_tokens_sample_into([1], new Uint32Array(1), { topK: 1 }).name, "generateLlamaTokensSampleInto", "llama family generate_tokens_sample_into delegates");
expectSame(llamaFamilyRuntimeSession.argmax_token().name, "argmaxLlamaToken", "llama family argmax_token delegates");
expectSame(llamaFamilyRuntimeSession.sample_token(undefined, { topK: 1 }).name, "sampleLlamaToken", "llama family sample_token delegates");
expectSame(llamaFamilySurfaceCalls.some((call) => call.name === "executeLlamaTensor"), true, "llama family snake aliases call LLaMA callbacks");
const genericFamilySession = new genericFamilySurface.Session(9, { inputLen: 2, outputLen: 1 });
for (const method of [
  "requireCanExecuteStepParams",
  "requireAllocationFreeStepParams",
  "requireRuntimeOutputAllocationFreeStepParams",
  "requireNoReadbackStepParams",
  "requireReadbackFreeStepParams",
  "requireHotStepParams",
]) {
  expectSame(
    genericFamilySession[method]({ input: [1, 2], output: false }),
    genericFamilyHotCompatibility,
    `generic family Session ${method} delegates`,
  );
}
expectSame(genericFamilyRequireCalls.map((call) => call.method), [
  "requireCanExecuteStepParams",
  "requireAllocationFreeStepParams",
  "requireRuntimeOutputAllocationFreeStepParams",
  "requireNoReadbackStepParams",
  "requireReadbackFreeStepParams",
  "requireHotStepParams",
], "generic family require method calls");
expectSame(genericFamilyRequireCalls[0].session, genericFamilySession, "generic family require session");
expectSame(genericFamilyRequireCalls[0].params, { input: [1, 2], output: false }, "generic family require params");

const adapterGenericFamilyModelHandleCalls = [];
const adapterGenericFamilySurface = createAdapterGenericFamilyModelHandleSurface({
  symbols: {
    modelCreate(desc, out) {
      adapterGenericFamilyModelHandleCalls.push(["modelCreate", desc, out]);
      out.handle = `handle:${desc.kind}:${desc.inputLen}:${desc.outputLen}`;
      return 0;
    },
  },
  check(status) {
    adapterGenericFamilyModelHandleCalls.push(["check", status]);
  },
  handleOut() {
    const out = {};
    adapterGenericFamilyModelHandleCalls.push(["handleOut", out]);
    return out;
  },
  readHandle(out, label) {
    adapterGenericFamilyModelHandleCalls.push(["readHandle", out, label]);
    return out.handle;
  },
  modelDesc(desc) {
    return {
      kind: desc.hiddenLen ? "mlp" : "linear",
      inputLen: desc.inputLen,
      outputLen: desc.outputLen,
    };
  },
  modelHandleLabel: "generic-model",
  modelFacadePolicy: {},
  genericProgramFacade: { withCompileEvidence(program) { return program; } },
  genericSessionFacade: {},
  bindProgram: () => null,
  packedProgramWeightsLen: () => 0,
  packedProgramBiasLen: () => 0,
  programBufferLayoutFromRequirements: () => ({ slots: [] }),
});
expectSame(
  adapterGenericFamilySurface.TinyLinearModel.create({ inputLen: 3, outputLen: 2 }).handle,
  "handle:linear:3:2",
  "Adapter generic family model-handle surface creates TinyLinear handles",
);
expectSame(
  adapterGenericFamilySurface.TinyMlpModel.create({ inputLen: 3, hiddenLen: 4, outputLen: 2 }).handle,
  "handle:mlp:3:2",
  "Adapter generic family model-handle surface creates TinyMlp handles",
);
expectSame(adapterGenericFamilyModelHandleCalls.map((call) => call[0]), [
  "handleOut",
  "modelCreate",
  "check",
  "readHandle",
  "handleOut",
  "modelCreate",
  "check",
  "readHandle",
], "Adapter generic family model-handle call order");
expectSame(adapterGenericFamilyModelHandleCalls[3][2], "generic-model", "Adapter generic family model-handle read label");

const sessionLiveFacadeChecks = [];
const sessionLiveFacadeHelpers = sessionFacade.createSessionLiveFacadeHelpers({
  handle(session) {
    return session.handle;
  },
  assertSessionAlive(handle) {
    sessionLiveFacadeChecks.push(handle);
    if (handle === 0) throw new Error("session is dead");
  },
  inspect(handle) {
    return { handle, signature: `inspect-${handle}` };
  },
  position(handle) {
    return handle + 10;
  },
});
expectSame(Object.isFrozen(sessionLiveFacadeHelpers), true, "session live facade helpers frozen");
expectSame(sessionLiveFacadeHelpers.handle({ handle: 3 }), 3, "session live facade handle");
sessionLiveFacadeHelpers.assertLiveSession({ handle: 3 });
expectSame(sessionLiveFacadeHelpers.inspect({ handle: 4 }), { handle: 4, signature: "inspect-4" }, "session live facade inspect");
expectSame(sessionLiveFacadeHelpers.position({ handle: 5 }), 15, "session live facade position");
expectSame(sessionLiveFacadeChecks, [3, 4, 5], "session live facade live checks");
expectThrow(
  () => sessionLiveFacadeHelpers.inspect({ handle: 0 }),
  "session is dead",
  "session live facade dead guard",
);
const sessionLiveNoPositionHelpers = sessionFacade.createSessionLiveFacadeHelpers({
  handle(session) { return session.handle; },
  assertSessionAlive() {},
  inspect(handle) { return { handle }; },
});
expectThrow(
  () => sessionLiveNoPositionHelpers.position({ handle: 1 }),
  "Session.position is unavailable",
  "session live facade missing position",
);
expectThrow(
  () => sessionFacade.createSessionLiveFacadeHelpers({ handle() {}, assertSessionAlive() {} }),
  "createSessionLiveFacadeHelpers requires handle, assertSessionAlive, and inspect callbacks",
  "session live facade requires callbacks",
);
expectThrow(
  () => sessionFacade.createSessionLiveFacadeHelpers({ handle() {}, assertSessionAlive() {}, inspect() {}, position: 1 }),
  "createSessionLiveFacadeHelpers position callback must be a function",
  "session live facade position callback",
);
const sessionLayoutFacadeChecks = [];
const sessionLayoutFacadeHelpers = sessionFacade.createSessionLayoutFacadeHelpers({
  assertLiveSession(session) {
    sessionLayoutFacadeChecks.push(session.handle);
    if (session.handle === 0) throw new Error("session is dead");
  },
  bufferLayout(session) {
    return session.layout;
  },
  bufferSlotMethodName: "SmokeSession.bufferSlot",
  kvCacheLayout(session) {
    return session.kv;
  },
  inputShape(session) {
    return session.inputShape;
  },
  outputShape(session) {
    return session.outputShape;
  },
  inputLen(session) {
    return session.inputLen;
  },
  outputLen(session) {
    return session.outputLen;
  },
  inputByteLength(session) {
    return session.inputBytes;
  },
  outputByteLength(session) {
    return session.outputBytes;
  },
  weightsLen(session) {
    return session.weightsLen;
  },
  weightsByteLength(session) {
    return session.weightsBytes;
  },
  biasLen(session) {
    return session.biasLen;
  },
  biasByteLength(session) {
    return session.biasBytes;
  },
  parameterLen(session) {
    return session.parameterLen;
  },
  parameterByteLength(session) {
    return session.parameterBytes;
  },
});
const sessionLayoutFacadeTarget = {
  handle: 12,
  layout: {
    input: { name: "input", byteLength: 8 },
    output: { name: "output", byteLength: 12 },
    slots: [
      { name: "input", kind: "input", byteLength: 8 },
      { name: "output", kind: "output", byteLength: 12 },
    ],
  },
  kv: { key: { byteLength: 4 } },
  inputShape: [2],
  outputShape: [3],
  inputLen: 2,
  outputLen: 3,
  inputBytes: 8,
  outputBytes: 12,
  weightsLen: 6,
  weightsBytes: 24,
  biasLen: 3,
  biasBytes: 12,
  parameterLen: 9,
  parameterBytes: 36,
};
expectSame(Object.isFrozen(sessionLayoutFacadeHelpers), true, "session layout facade helpers frozen");
expectSame(sessionLayoutFacadeHelpers.bufferSlotNames(sessionLayoutFacadeTarget), ["input", "output"], "session layout facade slot names");
expectSame(sessionLayoutFacadeHelpers.bufferSlot(sessionLayoutFacadeTarget, "output").byteLength, 12, "session layout facade slot lookup");
expectSame(sessionLayoutFacadeHelpers.kvCacheLayout(sessionLayoutFacadeTarget), { key: { byteLength: 4 } }, "session layout facade kv cache");
expectSame(sessionLayoutFacadeHelpers.inputShape(sessionLayoutFacadeTarget), [2], "session layout facade input shape");
expectSame(sessionLayoutFacadeHelpers.outputShape(sessionLayoutFacadeTarget), [3], "session layout facade output shape");
expectSame(sessionLayoutFacadeHelpers.inputLen(sessionLayoutFacadeTarget), 2, "session layout facade input len");
expectSame(sessionLayoutFacadeHelpers.outputByteLength(sessionLayoutFacadeTarget), 12, "session layout facade output bytes");
expectSame(sessionLayoutFacadeHelpers.parameterByteLength(sessionLayoutFacadeTarget), 36, "session layout facade parameter bytes");
expectThrow(
  () => sessionLayoutFacadeHelpers.bufferSlot(sessionLayoutFacadeTarget, 1),
  "SmokeSession.bufferSlot requires a buffer slot name or kind",
  "session layout facade slot error label",
);
expectThrow(
  () => sessionLayoutFacadeHelpers.outputShape({ handle: 0 }),
  "session is dead",
  "session layout facade live guard",
);
const sessionLayoutNoOptionalHelpers = sessionFacade.createSessionLayoutFacadeHelpers({
  assertLiveSession() {},
  bufferLayout() { return {}; },
  bufferSlotMethodName: "Session.bufferSlot",
  outputShape() { return [1]; },
  inputLen() { return 0; },
  outputLen() { return 1; },
  inputByteLength() { return 0; },
  outputByteLength() { return 4; },
  weightsLen() { return 0; },
  weightsByteLength() { return 0; },
  biasLen() { return 0; },
  biasByteLength() { return 0; },
  parameterLen() { return 0; },
  parameterByteLength() { return 0; },
});
expectThrow(
  () => sessionLayoutNoOptionalHelpers.inputShape({ handle: 1 }),
  "Session.inputShape is unavailable",
  "session layout facade missing input shape",
);
expectThrow(
  () => sessionLayoutNoOptionalHelpers.kvCacheLayout({ handle: 1 }),
  "Session.kvCacheLayout is unavailable",
  "session layout facade missing kv cache",
);
expectThrow(
  () => sessionFacade.createSessionLayoutFacadeHelpers({ assertLiveSession() {}, bufferLayout() {}, bufferSlotMethodName: "" }),
  "createSessionLayoutFacadeHelpers requires assertLiveSession, bufferLayout, bufferSlotMethodName, and layout callbacks",
  "session layout facade requires callbacks",
);
const sessionCallFacadeChecks = [];
const sessionCallFacadeHelpers = sessionFacade.createSessionCallProfileFacadeHelpers({
  assertLiveSession(session) {
    sessionCallFacadeChecks.push(session);
    if (!session || session.dead) throw new Error("session is dead");
  },
});
const sessionCallFacadeTarget = {};
expectSame(Object.isFrozen(sessionCallFacadeHelpers), true, "session call facade helpers frozen");
sessionCallFacadeHelpers.bumpSessionCallProfile(sessionCallFacadeTarget, "stepCount");
sessionCallFacadeHelpers.bumpSessionCallProfile(sessionCallFacadeTarget, "executeCount");
const sessionCallFacadeProfile = sessionCallFacadeHelpers.sessionCallProfile(sessionCallFacadeTarget);
expectSame(sessionCallFacadeProfile.stepCount, 1, "session call facade step count");
expectSame(sessionCallFacadeProfile.executeCount, 1, "session call facade execute count");
expectSame(sessionCallFacadeHelpers.matchesSessionCallProfileSignature(sessionCallFacadeTarget, sessionCallFacadeProfile.signature), true, "session call facade signature match");
expectSame(sessionCallFacadeHelpers.session_call_profile(sessionCallFacadeTarget).signature, sessionCallFacadeProfile.signature, "session call facade snake profile alias");
expectSame(sessionCallFacadeHelpers.matches_session_call_profile_signature(sessionCallFacadeTarget, sessionCallFacadeProfile.signature), true, "session call facade snake signature alias");
sessionCallFacadeHelpers.resetSessionCallProfile(sessionCallFacadeTarget);
expectSame(sessionCallFacadeHelpers.sessionCallProfile(sessionCallFacadeTarget).stepCount, 0, "session call facade reset");
sessionCallFacadeHelpers.reset_session_call_profile(sessionCallFacadeTarget);
expectSame(sessionCallFacadeChecks.length, 7, "session call facade live checks");
expectThrow(
  () => sessionCallFacadeHelpers.sessionCallProfile({ dead: true }),
  "session is dead",
  "session call facade live guard",
);
expectThrow(
  () => sessionFacade.createSessionCallProfileFacadeHelpers({}),
  "createSessionCallProfileFacadeHelpers requires assertLiveSession callback",
  "session call facade requires callback",
);
const sessionSizingFacadeChecks = [];
const sessionSizingFacadeHelpers = sessionFacade.createSessionBufferSizingFacadeHelpers({
  assertLiveSession(session) {
    sessionSizingFacadeChecks.push(session);
    if (!session || session.dead) throw new Error("session is dead");
  },
  modelKind(session) { return session.modelKind; },
  inputLen(session) { return session.inputLen; },
  inputByteLength(session) { return session.inputByteLength; },
  outputLen(session) { return session.outputLen; },
  outputByteLength(session) { return session.outputByteLength; },
  weightsLen(session) { return session.weightsLen; },
  weightsByteLength(session) { return session.weightsByteLength; },
  biasLen(session) { return session.biasLen; },
  biasByteLength(session) { return session.biasByteLength; },
  parameterLen(session) { return session.parameterLen; },
  parameterByteLength(session) { return session.parameterByteLength; },
});
const sessionSizingFacadeTarget = {
  modelKind: "facade-linear",
  inputLen: 2,
  inputByteLength: 8,
  outputLen: 3,
  outputByteLength: 12,
  weightsLen: 6,
  weightsByteLength: 24,
  biasLen: 3,
  biasByteLength: 12,
  parameterLen: 9,
  parameterByteLength: 36,
};
const sessionSizingFacadeSnapshot = sessionSizingFacadeHelpers.bufferSizing(sessionSizingFacadeTarget);
expectSame(Object.isFrozen(sessionSizingFacadeHelpers), true, "session sizing facade helpers frozen");
expectSame(sessionSizingFacadeSnapshot.signature, "session-buffer-sizing|model=facade-linear|scalar=f32|scalarBytes=4|input=2|inputBytes=8|output=3|outputBytes=12|weights=6|weightsBytes=24|bias=3|biasBytes=12|parameters=9|parameterBytes=36", "session sizing facade signature");
expectSame(Object.isFrozen(sessionSizingFacadeSnapshot), true, "session sizing facade snapshot frozen");
expectSame(sessionSizingFacadeHelpers.matchesBufferSizingSignature(sessionSizingFacadeTarget, sessionSizingFacadeSnapshot.signature), true, "session sizing facade signature match");
expectSame(sessionSizingFacadeHelpers.matchesBufferSizingSignature(sessionSizingFacadeTarget, `${sessionSizingFacadeSnapshot.signature}|changed`), false, "session sizing facade signature mismatch");
expectSame(sessionSizingFacadeChecks.length, 3, "session sizing facade live checks");
expectThrow(
  () => sessionSizingFacadeHelpers.bufferSizing({ dead: true }),
  "session is dead",
  "session sizing facade live guard",
);
expectThrow(
  () => sessionFacade.createSessionBufferSizingFacadeHelpers({ assertLiveSession() {} }),
  "createSessionBufferSizingFacadeHelpers requires assertLiveSession and sizing callbacks",
  "session sizing facade requires callbacks",
);
const sessionRuntimeFacadeChecks = [];
const sessionRuntimeFacadeHelpers = sessionFacade.createSessionRuntimeProfileFacadeHelpers({
  assertLiveSession(session) {
    sessionRuntimeFacadeChecks.push(session);
    if (!session || session.dead) throw new Error("session is dead");
  },
  runtimeProfile(session) {
    return session.profile;
  },
  resetRuntimeProfile(session) {
    session.profile = { signature: "runtime-reset", stepCount: 0 };
    return session.profile;
  },
});
const sessionRuntimeFacadeTarget = {
  profile: { signature: "runtime-profile", stepCount: 2 },
};
expectSame(Object.isFrozen(sessionRuntimeFacadeHelpers), true, "session runtime facade helpers frozen");
expectSame(sessionRuntimeFacadeHelpers.runtimeProfile(sessionRuntimeFacadeTarget), { signature: "runtime-profile", stepCount: 2 }, "session runtime facade profile");
expectSame(sessionRuntimeFacadeHelpers.runtime_profile(sessionRuntimeFacadeTarget), { signature: "runtime-profile", stepCount: 2 }, "session runtime facade profile alias");
expectSame(sessionRuntimeFacadeHelpers.matchesRuntimeProfileSignature(sessionRuntimeFacadeTarget, "runtime-profile"), true, "session runtime facade signature match");
expectSame(sessionRuntimeFacadeHelpers.matches_runtime_profile_signature(sessionRuntimeFacadeTarget, "runtime-profile"), true, "session runtime facade signature match alias");
expectSame(sessionRuntimeFacadeHelpers.matchesRuntimeProfileSignature(sessionRuntimeFacadeTarget, "wrong"), false, "session runtime facade signature mismatch");
expectSame(sessionRuntimeFacadeHelpers.matchesRuntimeProfileSignature(sessionRuntimeFacadeTarget, null), false, "session runtime facade signature invalid");
expectSame(sessionRuntimeFacadeHelpers.resetRuntimeProfile(sessionRuntimeFacadeTarget), { signature: "runtime-reset", stepCount: 0 }, "session runtime facade reset");
expectSame(sessionRuntimeFacadeHelpers.reset_runtime_profile(sessionRuntimeFacadeTarget), { signature: "runtime-reset", stepCount: 0 }, "session runtime facade reset alias");
expectSame(sessionRuntimeFacadeChecks.length, 7, "session runtime facade live checks");
expectThrow(
  () => sessionRuntimeFacadeHelpers.runtimeProfile({ dead: true }),
  "session is dead",
  "session runtime facade live guard",
);
expectThrow(
  () => sessionFacade.createSessionRuntimeProfileFacadeHelpers({ assertLiveSession() {}, runtimeProfile() {} }),
  "createSessionRuntimeProfileFacadeHelpers requires assertLiveSession, runtimeProfile, and resetRuntimeProfile callbacks",
  "session runtime facade requires callbacks",
);
const sessionParameterFacadeChecks = [];
const sessionParameterFacadeUploads = [];
const sessionParameterFacadeBumps = [];
const sessionParameterFacadeHelpers = sessionFacade.createSessionParameterFacadeHelpers({
  assertLiveSession(session) {
    sessionParameterFacadeChecks.push(session);
    if (!session || session.dead) throw new Error("session is dead");
  },
  bumpSessionCallProfile(session, field) {
    sessionParameterFacadeBumps.push({ session, field });
  },
  parameterLayout(session) {
    return session.layout;
  },
  uploadPersistent(session) {
    sessionParameterFacadeUploads.push({ kind: "all", handle: session.handle });
    return { uploaded: "all", handle: session.handle };
  },
  uploadPersistentRange(session, first, len) {
    sessionParameterFacadeUploads.push({ kind: "range", handle: session.handle, first, len });
    return { uploaded: "range", handle: session.handle, first, len };
  },
});
const sessionParameterFacadeTarget = {
  handle: 42,
  layout: {
    weightsLen: 6,
    biasLen: 3,
    parameters: [
      { name: "weight", binding: "weights", shape: [3, 2] },
      { name: "bias", binding: "bias", shape: [3] },
    ],
  },
};
expectSame(Object.isFrozen(sessionParameterFacadeHelpers), true, "session parameter facade helpers frozen");
expectSame(sessionParameterFacadeHelpers.parameterNames(sessionParameterFacadeTarget), ["weight", "bias"], "session parameter facade names");
expectSame(sessionParameterFacadeHelpers.parameterInfo(sessionParameterFacadeTarget, "bias"), { name: "bias", binding: "bias", shape: [3] }, "session parameter facade info by name");
expectSame(sessionParameterFacadeHelpers.parameterInfos(sessionParameterFacadeTarget).length, 2, "session parameter facade infos length");
expectSame(sessionParameterFacadeHelpers.uploadParameters(sessionParameterFacadeTarget), { uploaded: "all", handle: 42 }, "session parameter facade upload all");
expectSame(sessionParameterFacadeHelpers.uploadParameter(sessionParameterFacadeTarget, 0), { uploaded: "range", handle: 42, first: 0, len: 1 }, "session parameter facade upload index");
expectSame(sessionParameterFacadeHelpers.uploadParameterByName(sessionParameterFacadeTarget, "bias"), { uploaded: "range", handle: 42, first: 1, len: 1 }, "session parameter facade upload by name");
expectSame(sessionParameterFacadeHelpers.uploadParameterRange(sessionParameterFacadeTarget, 0, 2), { uploaded: "range", handle: 42, first: 0, len: 2 }, "session parameter facade upload range");
expectSame(sessionParameterFacadeBumps.map((entry) => entry.field), ["uploadParametersCount", "uploadParameterCount", "uploadParameterByNameCount", "uploadParameterRangeCount"], "session parameter facade profile bumps");
expectSame(sessionParameterFacadeUploads, [
  { kind: "all", handle: 42 },
  { kind: "range", handle: 42, first: 0, len: 1 },
  { kind: "range", handle: 42, first: 1, len: 1 },
  { kind: "range", handle: 42, first: 0, len: 2 },
], "session parameter facade upload calls");
expectThrow(
  () => sessionParameterFacadeHelpers.parameterNames({ dead: true }),
  "session is dead",
  "session parameter facade live guard",
);
expectThrow(
  () => sessionParameterFacadeHelpers.uploadParameterByName(sessionParameterFacadeTarget, "missing"),
  "Session parameter missing was not found",
  "session parameter facade missing name",
);
expectThrow(
  () => sessionFacade.createSessionParameterFacadeHelpers({ assertLiveSession() {} }),
  "createSessionParameterFacadeHelpers requires assertLiveSession, bumpSessionCallProfile, parameterLayout, uploadPersistent, and uploadPersistentRange callbacks",
  "session parameter facade requires callbacks",
);
const sessionLifecycleFacadeChecks = [];
const sessionLifecycleFacadeBumps = [];
const sessionLifecycleFacadeActions = [];
const sessionLifecycleFacadeHelpers = sessionFacade.createSessionLifecycleFacadeHelpers({
  assertLiveSession(session) {
    sessionLifecycleFacadeChecks.push(session.handle);
    if (session.handle === 0) throw new Error("session is dead");
  },
  bumpSessionCallProfile(session, field) {
    sessionLifecycleFacadeBumps.push({ handle: session.handle, field });
  },
  reset(session) {
    sessionLifecycleFacadeActions.push({ kind: "reset", handle: session.handle });
    return { reset: session.handle };
  },
  free(session) {
    sessionLifecycleFacadeActions.push({ kind: "free", handle: session.handle });
    session.handle = 0;
  },
});
const sessionLifecycleFacadeTarget = { handle: 8 };
expectSame(Object.isFrozen(sessionLifecycleFacadeHelpers), true, "session lifecycle facade helpers frozen");
expectSame(sessionLifecycleFacadeHelpers.reset(sessionLifecycleFacadeTarget), { reset: 8 }, "session lifecycle facade reset");
expectSame(sessionLifecycleFacadeBumps, [{ handle: 8, field: "resetCount" }], "session lifecycle facade reset bump");
sessionLifecycleFacadeHelpers.free(sessionLifecycleFacadeTarget);
expectSame(sessionLifecycleFacadeTarget.handle, 0, "session lifecycle facade free mutates handle");
sessionLifecycleFacadeHelpers.dispose(sessionLifecycleFacadeTarget);
expectSame(sessionLifecycleFacadeActions, [
  { kind: "reset", handle: 8 },
  { kind: "free", handle: 8 },
  { kind: "free", handle: 0 },
], "session lifecycle facade actions");
expectSame(sessionLifecycleFacadeChecks, [8], "session lifecycle facade live checks");
expectThrow(
  () => sessionLifecycleFacadeHelpers.reset({ handle: 0 }),
  "session is dead",
  "session lifecycle facade reset live guard",
);
expectThrow(
  () => sessionFacade.createSessionLifecycleFacadeHelpers({ assertLiveSession() {}, reset() {}, free() {} }),
  "createSessionLifecycleFacadeHelpers requires assertLiveSession, bumpSessionCallProfile, reset, and free callbacks",
  "session lifecycle facade requires callbacks",
);
const sessionReadbackFacadeBumps = [];
const sessionReadbackFacadeReads = [];
const sessionReadbackFacadeHelpers = sessionFacade.createSessionReadbackFacadeHelpers({
  bumpSessionCallProfile(session, field) {
    sessionReadbackFacadeBumps.push({ handle: session.handle, field });
  },
  readOutputIntoCore(session, outputValues, length = session.defaultLength, byteOffset = 0) {
    sessionReadbackFacadeReads.push({ handle: session.handle, length, byteOffset });
    outputValues.set(session.output.subarray(byteOffset / 4, byteOffset / 4 + length));
    return outputValues;
  },
  readOutputTensorTarget(session, options) {
    return {
      values: options.output || new Float32Array(session.defaultLength),
      length: options.length ?? session.defaultLength,
      byteOffset: options.byteOffset ?? 0,
      opts: options,
    };
  },
  activeOutputValues(values, length) {
    return values.length === length ? values : values.subarray(0, length);
  },
  outputTensor(session, activeValues, options, target) {
    return { handle: session.handle, data: activeValues, shape: options.shape || [target.length] };
  },
});
const sessionReadbackFacadeTarget = {
  handle: 9,
  defaultLength: 3,
  output: new Float32Array([5, 6, 7, 8]),
};
const sessionReadbackInto = new Float32Array(2);
expectSame(sessionReadbackFacadeHelpers.readOutputInto(sessionReadbackFacadeTarget, sessionReadbackInto, 2, 4), [6, 7], "session readback facade read into");
const sessionReadbackTensor = sessionReadbackFacadeHelpers.readOutputTensor(sessionReadbackFacadeTarget, {
  output: new Float32Array(4),
  length: 3,
  shape: [1, 3],
});
expectSame(sessionReadbackTensor.shape, [1, 3], "session readback facade tensor shape");
expectSame(sessionReadbackTensor.data, [5, 6, 7], "session readback facade active tensor data");
expectSame(sessionReadbackFacadeBumps.map((entry) => entry.field), ["readOutputIntoCount", "readOutputTensorCount"], "session readback facade profile bumps");
expectSame(sessionReadbackFacadeReads, [
  { handle: 9, length: 2, byteOffset: 4 },
  { handle: 9, length: 3, byteOffset: 0 },
], "session readback facade core reads");
expectThrow(
  () => sessionFacade.createSessionReadbackFacadeHelpers({ bumpSessionCallProfile() {}, readOutputIntoCore() {} }),
  "createSessionReadbackFacadeHelpers requires bumpSessionCallProfile, readOutputIntoCore, readOutputTensorTarget, activeOutputValues, and outputTensor callbacks",
  "session readback facade requires callbacks",
);
const genericCoreStepCalls = [];
const genericCoreStepHelpers = sessionFacade.createGenericSessionCoreStepFacadeHelpers({
  assertLiveSession(session) {
    if (session.handle === 0) throw new Error("generic session is dead");
  },
  isNativeBuffer(value) {
    return value && value.native === true;
  },
  prepareHostValue(value) {
    const data = value instanceof Float32Array ? value : new Float32Array(value);
    return { data, shape: [data.length] };
  },
  validateHostValueShape(prepared, expectedShape, label) {
    if (prepared.data.length !== expectedShape.reduce((acc, dim) => acc * dim, 1)) {
      throw new Error(`${label} shape mismatch`);
    }
  },
  stepSession(handle, input, output) {
    genericCoreStepCalls.push({ kind: "step", handle, input, output });
    if (output instanceof Float32Array) output.set([7, 8]);
    return 2;
  },
  prepareStepSession(handle, input, output) {
    genericCoreStepCalls.push({ kind: "prepare-step", handle, input, output });
    return () => {
      genericCoreStepCalls.push({ kind: "prepared-step", handle, input, output });
      if (output instanceof Float32Array) output.set([9, 10]);
      return 2;
    };
  },
  stepNoOutput(handle, input) {
    genericCoreStepCalls.push({ kind: "advance", handle, input });
    return 0;
  },
  stepContract() {
    return {
      kind: "generic-session-step-contract",
      signature: "generic",
      inputLen: 2,
      outputLen: 2,
      inputShape: [2],
      outputShape: [2],
      defaultOutput: "allocated-readback",
    };
  },
});
const genericCoreStepSession = {
  handle: 10,
  desc: { inputLen: 2, outputLen: 2, inputShape: [2], outputShape: [2] },
  boundInput: new Float32Array([1, 2]),
  boundOutput: new Float32Array(2),
};
expectSame(Object.isFrozen(genericCoreStepHelpers), true, "generic core step facade helpers frozen");
expectSame(genericCoreStepHelpers.hostBoundInput(genericCoreStepSession), [1, 2], "generic core step host bound input");
expectSame(genericCoreStepHelpers.nativeBoundOutput(genericCoreStepSession), null, "generic core step no native output");
expectSame(genericCoreStepHelpers.hostBoundOutput(genericCoreStepSession), [0, 0], "generic core step host bound output");
expectSame(genericCoreStepHelpers.explicitInput(genericCoreStepSession, undefined, "session.step input"), [1, 2], "generic core step explicit bound input");
expectSame(genericCoreStepHelpers.explicitOutput(genericCoreStepSession, new Float32Array(2), "session.step output"), [0, 0], "generic core step explicit output");
expectSame(genericCoreStepHelpers.stepCore(genericCoreStepSession, undefined), [7, 8], "generic core step stepCore host output");
const genericCoreStepIntoOutput = new Float32Array(2);
expectSame(genericCoreStepHelpers.stepIntoCore(genericCoreStepSession, genericCoreStepIntoOutput, [5, 6]), genericCoreStepIntoOutput, "generic core stepIntoCore returns caller Float32Array");
expectSame(genericCoreStepIntoOutput, [7, 8], "generic core stepIntoCore writes caller Float32Array");
const genericCorePreparedOutput = new Float32Array(2);
const genericCorePrepared = genericCoreStepHelpers.prepareExecuteIntoCore(genericCoreStepSession, genericCorePreparedOutput, [6, 7]);
expectSame(genericCorePrepared(), genericCorePreparedOutput, "generic core prepared executeInto returns caller Float32Array");
expectSame(genericCorePreparedOutput, [9, 10], "generic core prepared executeInto uses prepared native step");
expectSame(genericCoreStepCalls.slice(-2).map((entry) => entry.kind), ["prepare-step", "prepared-step"], "generic core prepared executeInto native step path");
try {
  genericCoreStepHelpers.stepIntoCore(genericCoreStepSession, new Float32Array(1), [5, 6]);
  throw new Error("generic core stepIntoCore small Float32Array did not throw");
} catch (error) {
  expectSame(error.stepParamsDiagnosticCode, "invalid-output", "generic core stepIntoCore small Float32Array diagnostic code");
  expectSame(error.stepParamsDiagnosticDetails, { actualLength: 1, expectedLength: 2 }, "generic core stepIntoCore small Float32Array diagnostic details");
}
expectSame(genericCoreStepHelpers.advanceCore(genericCoreStepSession, [3, 4]), undefined, "generic core step advanceCore");
const genericCoreNativeSession = {
  ...genericCoreStepSession,
  boundOutput: { native: true, readFloat32: (length) => Float32Array.from(Array.from({ length }, (_, index) => index + 30)) },
};
expectSame(genericCoreStepHelpers.stepCore(genericCoreNativeSession, [5, 6]), [30, 31], "generic core step native readback");
const genericCoreCompatibility = genericCoreStepHelpers.stepParamsCompatibility(genericCoreStepSession, {
  input: new Float32Array([1, 2]),
  output: new Float32Array(2),
});
expectSame(genericCoreCompatibility.accepted, true, "generic core step compatibility accepted");
expectThrow(
  () => genericCoreStepHelpers.stepCore({ ...genericCoreStepSession, handle: 0 }, [1, 2]),
  "generic session is dead",
  "generic core step live guard",
);
expectThrow(
  () => sessionFacade.createGenericSessionCoreStepFacadeHelpers({ assertLiveSession() {}, isNativeBuffer() {} }),
  "createGenericSessionCoreStepFacadeHelpers requires assertLiveSession, isNativeBuffer, host value, native step, and stepContract callbacks",
  "generic core step facade requires callbacks",
);
const genericExecutionFacadeBumps = [];
const genericExecutionFacadeSteps = [];
const genericExecutionFacadeAdvances = [];
const genericExecutionFacadeHelpers = sessionFacade.createGenericSessionExecutionFacadeHelpers({
  bumpSessionCallProfile(session, field) {
    genericExecutionFacadeBumps.push({ handle: session.handle, field });
  },
  stepCore(session, inputValues, outputValues) {
    genericExecutionFacadeSteps.push({ handle: session.handle, inputValues, outputValues });
    if (outputValues instanceof Float32Array) {
      outputValues.set([3, 4]);
      return outputValues;
    }
    return new Float32Array([1, 2]);
  },
  stepIntoCore(session, outputValues, inputValues) {
    genericExecutionFacadeSteps.push({ handle: session.handle, inputValues, outputValues });
    outputValues.set([3, 4]);
    return outputValues;
  },
  prepareExecuteIntoCore(session, outputValues, inputValues) {
    return () => {
      genericExecutionFacadeSteps.push({ handle: session.handle, inputValues, outputValues });
      outputValues.set([3, 4]);
      return outputValues;
    };
  },
  advanceCore(session, inputValues) {
    genericExecutionFacadeAdvances.push({ handle: session.handle, inputValues });
  },
  outputTensorForSession(desc, values, options, boundOutputShape) {
    return { desc, values, options, boundOutputShape };
  },
});
const genericExecutionFacadeSession = {
  handle: 10,
  desc: { inputLen: 2, outputLen: 2, inputShape: [2], outputShape: [2] },
  boundOutputShape: [1, 2],
};
expectSame(Object.isFrozen(genericExecutionFacadeHelpers), true, "generic execution facade helpers frozen");
expectSame(genericExecutionFacadeHelpers.execute(genericExecutionFacadeSession, { input: [1, 2] }), [1, 2], "generic execution facade execute");
expectSame(genericExecutionFacadeHelpers.execute(genericExecutionFacadeSession, { input: [2, 3], output: false }), undefined, "generic execution facade execute no output");
const genericExecutionIntoOutput = new Float32Array(2);
expectSame(genericExecutionFacadeHelpers.executeInto(genericExecutionFacadeSession, genericExecutionIntoOutput, { input: [4, 5] }), [3, 4], "generic execution facade execute into");
const genericExecutionPreparedOutput = new Float32Array(2);
const genericExecutionPrepared = genericExecutionFacadeHelpers.prepareExecuteInto(genericExecutionFacadeSession, genericExecutionPreparedOutput, { input: [5, 6] });
expectSame(genericExecutionPrepared(), [3, 4], "generic execution facade prepared execute into");
const genericExecutionTensor = genericExecutionFacadeHelpers.executeTensor(genericExecutionFacadeSession, { input: [6, 7], shape: [2] });
expectSame(genericExecutionTensor.values, [1, 2], "generic execution facade execute tensor values");
expectSame(genericExecutionTensor.boundOutputShape, [1, 2], "generic execution facade execute tensor bound shape");
expectSame(genericExecutionFacadeHelpers.advance(genericExecutionFacadeSession, [8, 9]), undefined, "generic execution facade advance");
expectSame(genericExecutionFacadeBumps.map((entry) => entry.field), [
  "executeCount",
  "executeCount",
  "executeIntoCount",
  "prepareExecuteIntoCount",
  "executeTensorCount",
  "advanceCount",
], "generic execution facade profile bumps");
expectSame(genericExecutionFacadeSteps.length, 4, "generic execution facade step count");
expectSame(genericExecutionFacadeAdvances, [
  { handle: 10, inputValues: [2, 3] },
  { handle: 10, inputValues: [8, 9] },
], "generic execution facade advance calls");
expectThrow(
  () => genericExecutionFacadeHelpers.executeTensor(genericExecutionFacadeSession, { output: false }),
  "session.executeTensor requires output",
  "generic execution facade executeTensor rejects no output",
);
expectThrow(
  () => genericExecutionFacadeHelpers.executeInto(genericExecutionFacadeSession, [], {}),
  "session.executeInto requires a Float32Array output buffer",
  "generic execution facade executeInto output type",
);
expectThrow(
  () => sessionFacade.createGenericSessionExecutionFacadeHelpers({ bumpSessionCallProfile() {}, stepCore() {} }),
  "createGenericSessionExecutionFacadeHelpers requires bumpSessionCallProfile, stepCore, stepIntoCore, prepareExecuteIntoCore, advanceCore, and outputTensorForSession callbacks",
  "generic execution facade requires callbacks",
);
const genericStepFacadeBumps = [];
const genericStepFacadeSteps = [];
const genericStepFacadeHelpers = sessionFacade.createGenericSessionStepFacadeHelpers({
  bumpSessionCallProfile(session, field) {
    genericStepFacadeBumps.push({ handle: session.handle, field });
  },
  stepCore(session, inputValues, outputValues) {
    genericStepFacadeSteps.push({ handle: session.handle, inputValues, outputValues });
    if (outputValues instanceof Float32Array) {
      outputValues.set([9, 10]);
      return outputValues;
    }
    return new Float32Array([11, 12]);
  },
  stepIntoCore(session, outputValues, inputValues) {
    genericStepFacadeSteps.push({ handle: session.handle, inputValues, outputValues });
    outputValues.set([9, 10]);
    return outputValues;
  },
  outputTensorForSession(desc, values, options, boundOutputShape) {
    return { desc, values, options, boundOutputShape };
  },
});
const genericStepFacadeSession = {
  handle: 11,
  desc: { inputLen: 2, outputLen: 2, inputShape: [2], outputShape: [2] },
  boundOutputShape: [2],
};
expectSame(Object.isFrozen(genericStepFacadeHelpers), true, "generic step facade helpers frozen");
expectSame(genericStepFacadeHelpers.step(genericStepFacadeSession, [1, 2]), [11, 12], "generic step facade step");
const genericStepIntoOutput = new Float32Array(2);
expectSame(genericStepFacadeHelpers.stepInto(genericStepFacadeSession, genericStepIntoOutput, [3, 4]), [9, 10], "generic step facade step into");
const genericStepTensor = genericStepFacadeHelpers.stepTensor(genericStepFacadeSession, [5, 6], { shape: [2] });
expectSame(genericStepTensor.values, [11, 12], "generic step facade step tensor values");
expectSame(genericStepTensor.boundOutputShape, [2], "generic step facade step tensor bound shape");
expectSame(genericStepFacadeBumps.map((entry) => entry.field), [
  "stepCount",
  "stepIntoCount",
  "stepTensorCount",
], "generic step facade profile bumps");
expectSame(genericStepFacadeSteps.length, 3, "generic step facade step count");
expectThrow(
  () => genericStepFacadeHelpers.stepInto(genericStepFacadeSession, [], [1, 2]),
  "session.stepInto requires a Float32Array output buffer",
  "generic step facade stepInto output type",
);
expectThrow(
  () => sessionFacade.createGenericSessionStepFacadeHelpers({ bumpSessionCallProfile() {}, stepCore() {} }),
  "createGenericSessionStepFacadeHelpers requires bumpSessionCallProfile, stepCore, stepIntoCore, and outputTensorForSession callbacks",
  "generic step facade requires callbacks",
);
const llamaStepFacadeBumps = [];
const llamaStepFacadeSteps = [];
const llamaStepFacadeAdvances = [];
const llamaStepFacadeHelpers = sessionFacade.createLlamaSessionStepFacadeHelpers({
  assertLiveSession(session) {
    if (session.handle === 0) throw new Error("llama session is dead");
  },
  bumpSessionCallProfile(session, field) {
    llamaStepFacadeBumps.push({ handle: session.handle, field });
  },
  stepToken(handle, tokenId, outputValues, boundOutput, vocabSize) {
    llamaStepFacadeSteps.push({ handle, tokenId, outputValues, boundOutput, vocabSize });
    if (outputValues instanceof Float32Array) {
      outputValues.set([0.25, 0.5, 0.75]);
      return outputValues;
    }
    return new Float32Array([1, 2, 3]);
  },
  advanceToken(handle, tokenId) {
    llamaStepFacadeAdvances.push({ handle, tokenId });
  },
  outputTarget(options, label) {
    llamaStepFacadeSteps.push({ kind: "target", label, output: options.output });
    return options.output;
  },
  outputTensor(values, options) {
    return { values, options };
  },
});
const llamaStepFacadeSession = {
  handle: 12,
  vocabSize: 3,
  boundOutput: "native-output",
};
expectSame(Object.isFrozen(llamaStepFacadeHelpers), true, "llama step facade helpers frozen");
expectSame(llamaStepFacadeHelpers.step(llamaStepFacadeSession, 4), [1, 2, 3], "llama step facade step");
const llamaStepIntoOutput = new Float32Array(3);
expectSame(llamaStepFacadeHelpers.stepInto(llamaStepFacadeSession, llamaStepIntoOutput, 5), [0.25, 0.5, 0.75], "llama step facade step into");
const llamaStepTensorOutput = new Float32Array(3);
const llamaStepTensor = llamaStepFacadeHelpers.stepTensor(llamaStepFacadeSession, 6, { output: llamaStepTensorOutput, shape: [3] });
expectSame(llamaStepTensor.values, [0.25, 0.5, 0.75], "llama step facade step tensor values");
expectSame(llamaStepTensor.options.shape, [3], "llama step facade step tensor options");
expectSame(llamaStepFacadeHelpers.advance(llamaStepFacadeSession, 7), undefined, "llama step facade advance");
expectSame(llamaStepFacadeBumps.map((entry) => entry.field), [
  "stepCount",
  "stepIntoCount",
  "stepTensorCount",
  "advanceCount",
], "llama step facade profile bumps");
expectSame(llamaStepFacadeAdvances, [{ handle: 12, tokenId: 7 }], "llama step facade advance calls");
expectThrow(
  () => llamaStepFacadeHelpers.stepInto(llamaStepFacadeSession, [], 8),
  "session.stepInto requires a Float32Array output buffer",
  "llama step facade stepInto output type",
);
expectThrow(
  () => llamaStepFacadeHelpers.step({ ...llamaStepFacadeSession, handle: 0 }, 9),
  "llama session is dead",
  "llama step facade live guard",
);
expectThrow(
  () => sessionFacade.createLlamaSessionStepFacadeHelpers({ assertLiveSession() {}, stepToken() {} }),
  "createLlamaSessionStepFacadeHelpers requires assertLiveSession, bumpSessionCallProfile, stepToken, advanceToken, outputTarget, and outputTensor callbacks",
  "llama step facade requires callbacks",
);
const llamaExecutionFacadeBumps = [];
const llamaExecutionFacadeCalls = [];
const llamaExecutionFacadeHelpers = sessionFacade.createLlamaSessionExecutionFacadeHelpers({
  bumpSessionCallProfile(session, field) {
    llamaExecutionFacadeBumps.push({ handle: session.handle, field });
  },
  isNativeBuffer(value) {
    return value && value.native === true;
  },
  executeTokenWindow(handle, window, wantsLogits, descOutput) {
    llamaExecutionFacadeCalls.push({ handle, window, wantsLogits, descOutput });
    if (descOutput instanceof Float32Array) {
      descOutput.set([1, 2, 3]);
    }
    return wantsLogits ? 3 : 0;
  },
  scalarTokenWindow(session, tokenId) {
    session.scalarTokenScratch[0] = tokenId;
    return session.scalarTokenScratch;
  },
  scalarTokenExecuteOptions(_session, options) {
    return { ...options, tokensLen: 1 };
  },
  outputTokenWindowOptions(_session, options, outputValues) {
    return { ...options, output: outputValues };
  },
  scalarTokenOutputOptions(_session, outputValues) {
    return { tokensLen: 1, output: outputValues };
  },
  noOutputTokenWindowOptions(_session, options) {
    return { ...options, output: false };
  },
  outputTarget(options) {
    return options.output !== undefined ? options.output : new Float32Array(3);
  },
  outputTensor(values, options) {
    return { values, options };
  },
});
const llamaExecutionFacadeSession = {
  handle: 13,
  vocabSize: 3,
  boundOutput: null,
  scalarTokenScratch: new Uint32Array(1),
};
expectSame(Object.isFrozen(llamaExecutionFacadeHelpers), true, "llama execution facade helpers frozen");
expectSame(llamaExecutionFacadeHelpers.executeTokens(llamaExecutionFacadeSession, [1, 2], { tokensLen: 2 }), [1, 2, 3], "llama execution facade execute tokens");
expectSame(llamaExecutionFacadeHelpers.execute(llamaExecutionFacadeSession, { token: 4 }), [1, 2, 3], "llama execution facade execute scalar");
const llamaExecutionTensor = llamaExecutionFacadeHelpers.executeTensor(llamaExecutionFacadeSession, { token: 5, shape: [3] });
expectSame(llamaExecutionTensor.values, [1, 2, 3], "llama execution facade execute tensor values");
expectSame(llamaExecutionTensor.options.shape, [3], "llama execution facade execute tensor options");
const llamaExecutionIntoOutput = new Float32Array(3);
expectSame(llamaExecutionFacadeHelpers.executeInto(llamaExecutionFacadeSession, llamaExecutionIntoOutput, { tokens: [6, 7], tokensLen: 2 }), [1, 2, 3], "llama execution facade execute into");
expectSame(llamaExecutionFacadeHelpers.advanceTokens(llamaExecutionFacadeSession, [8, 9], { tokensLen: 2 }), undefined, "llama execution facade advance tokens");
expectSame(llamaExecutionFacadeBumps.map((entry) => entry.field), [
  "executeCount",
  "executeTensorCount",
  "executeIntoCount",
  "advanceCount",
], "llama execution facade profile bumps");
expectSame(llamaExecutionFacadeCalls.length, 5, "llama execution facade native call count");
expectThrow(
  () => llamaExecutionFacadeHelpers.executeInto(llamaExecutionFacadeSession, [], { token: 1 }),
  "session.executeInto requires a Float32Array output buffer",
  "llama execution facade executeInto output type",
);
expectThrow(
  () => llamaExecutionFacadeHelpers.executeTensor(llamaExecutionFacadeSession, { token: 1, output: false }),
  "session.executeTensor requires logits output",
  "llama execution facade execute tensor requires output",
);
expectThrow(
  () => sessionFacade.createLlamaSessionExecutionFacadeHelpers({ bumpSessionCallProfile() {}, isNativeBuffer() {} }),
  "createLlamaSessionExecutionFacadeHelpers requires bumpSessionCallProfile, isNativeBuffer, executeTokenWindow, token option helpers, outputTarget, and outputTensor callbacks",
  "llama execution facade requires callbacks",
);
const llamaPrefillFacadeBumps = [];
const llamaPrefillFacadeExecutes = [];
const llamaPrefillFacadeHelpers = sessionFacade.createLlamaSessionPrefillFacadeHelpers({
  bumpSessionCallProfile(session, field) {
    llamaPrefillFacadeBumps.push({ handle: session.handle, field });
  },
  executeTokens(session, tokens, options) {
    llamaPrefillFacadeExecutes.push({ handle: session.handle, tokens, options });
    if (options.output instanceof Float32Array) {
      options.output.set([1, 2, 3]);
      return options.output;
    }
    return new Float32Array([4, 5, 6]);
  },
  outputTokenWindowOptions(_session, options, outputValues) {
    return { ...options, output: outputValues };
  },
  outputTarget(options, label) {
    return options.output || new Float32Array(3);
  },
  outputTensor(values, options) {
    return { values, options };
  },
});
const llamaPrefillFacadeSession = { handle: 13 };
expectSame(Object.isFrozen(llamaPrefillFacadeHelpers), true, "llama prefill facade helpers frozen");
expectSame(llamaPrefillFacadeHelpers.prefill(llamaPrefillFacadeSession, [1, 2], new Float32Array(3), { activeTokenCount: 2 }), [1, 2, 3], "llama prefill facade prefill");
const llamaPrefillTensorOutput = new Float32Array(3);
const llamaPrefillTensor = llamaPrefillFacadeHelpers.prefillTensor(llamaPrefillFacadeSession, [3, 4], { output: llamaPrefillTensorOutput, shape: [3] });
expectSame(llamaPrefillTensor.values, [1, 2, 3], "llama prefill facade tensor values");
expectSame(llamaPrefillTensor.options.shape, [3], "llama prefill facade tensor options");
const llamaPrefillIntoOutput = new Float32Array(3);
expectSame(llamaPrefillFacadeHelpers.prefillInto(llamaPrefillFacadeSession, llamaPrefillIntoOutput, [5, 6], { tokenLength: 2 }), [1, 2, 3], "llama prefill facade prefill into");
expectSame(llamaPrefillFacadeBumps.map((entry) => entry.field), [
  "prefillCount",
  "prefillTensorCount",
  "prefillIntoCount",
], "llama prefill facade profile bumps");
expectSame(llamaPrefillFacadeExecutes.length, 3, "llama prefill facade execute count");
expectThrow(
  () => llamaPrefillFacadeHelpers.prefillInto(llamaPrefillFacadeSession, [], [7]),
  "session.prefillInto requires a Float32Array output buffer",
  "llama prefill facade prefillInto output type",
);
expectThrow(
  () => sessionFacade.createLlamaSessionPrefillFacadeHelpers({ bumpSessionCallProfile() {}, executeTokens() {} }),
  "createLlamaSessionPrefillFacadeHelpers requires bumpSessionCallProfile, executeTokens, outputTokenWindowOptions, outputTarget, and outputTensor callbacks",
  "llama prefill facade requires callbacks",
);
const llamaTokenSelectionCalls = [];
const llamaTokenSelectionFacadeHelpers = sessionFacade.createLlamaSessionTokenSelectionFacadeHelpers({
  f32(value) {
    llamaTokenSelectionCalls.push({ kind: "f32", value });
    return value instanceof Float32Array ? value : new Float32Array(value);
  },
  isNativeBuffer(value) {
    return value && value.native === true;
  },
  argmaxLogits(handle, data) {
    llamaTokenSelectionCalls.push({ kind: "argmax", handle, data });
    return { token: 2, logit: 9 };
  },
  sampleLogits(handle, data, options) {
    llamaTokenSelectionCalls.push({ kind: "sample", handle, data, options });
    return { token: 3, logit: 8 };
  },
  executeArgmaxWindow(handle, window) {
    llamaTokenSelectionCalls.push({ kind: "executeArgmax", handle, window });
    return { token: 4, logit: 7 };
  },
  executeSampleWindow(handle, window, sampleOptions) {
    llamaTokenSelectionCalls.push({ kind: "executeSample", handle, window, sampleOptions });
    return { token: 5, logit: 6 };
  },
  generateArgmaxWindow(handle, window, outputTokens) {
    llamaTokenSelectionCalls.push({ kind: "generateArgmax", handle, window, outputTokens });
    outputTokens.set([6, 7]);
    return { tokens_generated: 2, last_token: 7, last_logit: 5 };
  },
  generateSampleWindow(handle, window, outputTokens, sampleOptions) {
    llamaTokenSelectionCalls.push({ kind: "generateSample", handle, window, outputTokens, sampleOptions });
    outputTokens.set([8, 9]);
    return { tokens_generated: 2, last_token: 9, last_logit: 4 };
  },
});
const llamaTokenSelectionSession = {
  handle: 14,
  boundOutput: { native: true },
  scalarTokenScratch: new Uint32Array(1),
  scalarTokenSampleOptionsScratch: {},
  scalarTokenWindow(tokenId) {
    this.scalarTokenScratch[0] = tokenId;
    return this.scalarTokenScratch;
  },
  scalarTokenSampleOptions(options) {
    return { ...options, tokensLen: 1 };
  },
  executeTokensArgmax(tokens, options) {
    return llamaTokenSelectionFacadeHelpers.executeTokensArgmax(this, tokens, options);
  },
  executeTokensSample(tokens, options) {
    return llamaTokenSelectionFacadeHelpers.executeTokensSample(this, tokens, options);
  },
};
expectSame(Object.isFrozen(llamaTokenSelectionFacadeHelpers), true, "llama token selection facade helpers frozen");
expectSame(llamaTokenSelectionFacadeHelpers.argmaxToken(llamaTokenSelectionSession, [1, 2]), { token: 2, logit: 9 }, "llama token selection argmax");
expectSame(llamaTokenSelectionFacadeHelpers.sampleToken(llamaTokenSelectionSession, null, { top_k: 2, seed: 11 }), { token: 3, logit: 8 }, "llama token selection sample");
expectSame(llamaTokenSelectionFacadeHelpers.executeTokensArgmax(llamaTokenSelectionSession, [1, 2], { tokensLen: 2 }), { token: 4, logit: 7 }, "llama token selection execute argmax");
expectSame(llamaTokenSelectionFacadeHelpers.stepArgmax(llamaTokenSelectionSession, 12), { token: 4, logit: 7 }, "llama token selection step argmax");
expectSame(llamaTokenSelectionFacadeHelpers.executeTokensSample(llamaTokenSelectionSession, [3, 4], { top_k: 3 }), { token: 5, logit: 6 }, "llama token selection execute sample");
expectSame(llamaTokenSelectionFacadeHelpers.stepSample(llamaTokenSelectionSession, 13, { top_k: 4 }), { token: 5, logit: 6 }, "llama token selection step sample");
const llamaGenerateArgmaxOut = new Uint32Array(2);
expectSame(llamaTokenSelectionFacadeHelpers.generateTokensArgmaxInto(llamaTokenSelectionSession, [1], llamaGenerateArgmaxOut), {
  tokens_generated: 2,
  last_token: 7,
  last_logit: 5,
}, "llama token selection generate argmax into");
expectSame(llamaGenerateArgmaxOut, [6, 7], "llama token selection generate argmax output");
const llamaGenerateSampleOut = new Uint32Array(2);
expectSame(llamaTokenSelectionFacadeHelpers.generateTokensSampleInto(llamaTokenSelectionSession, [1], llamaGenerateSampleOut, { top_k: 2 }), {
  tokens_generated: 2,
  last_token: 9,
  last_logit: 4,
}, "llama token selection generate sample into");
expectSame(llamaGenerateSampleOut, [8, 9], "llama token selection generate sample output");
expectSame(llamaTokenSelectionFacadeHelpers.generateTokensArgmax(llamaTokenSelectionSession, [1], 2).tokens_generated, 2, "llama token selection generate argmax allocates");
expectSame(llamaTokenSelectionFacadeHelpers.generateTokensSample(llamaTokenSelectionSession, [1], 2, { seed: 1 }).tokens_generated, 2, "llama token selection generate sample allocates");
expectSame(llamaTokenSelectionCalls.some((call) => call.kind === "f32"), true, "llama token selection converts logits");
expectThrow(
  () => llamaTokenSelectionFacadeHelpers.executeTokensArgmax({ ...llamaTokenSelectionSession, boundOutput: null }, [1]),
  "executeTokensArgmax requires a NativeBuffer output binding",
  "llama token selection argmax requires native output",
);
expectThrow(
  () => sessionFacade.createLlamaSessionTokenSelectionFacadeHelpers({ f32() {}, isNativeBuffer() {} }),
  "createLlamaSessionTokenSelectionFacadeHelpers requires f32, isNativeBuffer, token selection, execution, and generation callbacks",
  "llama token selection facade requires callbacks",
);
const llamaTokenScratchHelpers = sessionFacade.createLlamaTokenScratchFacadeHelpers({
  scalarTokenScratch(session) { return session.scalarTokenScratch; },
  scalarTokenSampleOptionsScratch(session) { return session.scalarTokenSampleOptionsScratch; },
  noOutputTokenWindowOptionsScratch(session) { return session.noOutputTokenWindowOptionsScratch; },
  outputTokenWindowOptionsScratch(session) { return session.outputTokenWindowOptionsScratch; },
  scalarTokenExecuteOptionsScratch(session) { return session.scalarTokenExecuteOptionsScratch; },
});
const llamaTokenScratchSession = {
  scalarTokenScratch: new Uint32Array(1),
  scalarTokenSampleOptionsScratch: {},
  noOutputTokenWindowOptionsScratch: {},
  outputTokenWindowOptionsScratch: {},
  scalarTokenExecuteOptionsScratch: {},
};
expectSame(Object.isFrozen(llamaTokenScratchHelpers), true, "llama token scratch facade helpers frozen");
expectSame(llamaTokenScratchHelpers.scalarTokenWindow(llamaTokenScratchSession, 17), [17], "llama token scratch scalar token window");
expectSame(llamaTokenScratchHelpers.sessionNoOutputTokenWindowOptions(llamaTokenScratchSession, { tokensLen: 2 }), {
  tokensLen: 2,
  tokenLength: undefined,
  activeTokenLength: undefined,
  activeTokenCount: undefined,
  output: false,
}, "llama token scratch no output options");
expectSame(llamaTokenScratchHelpers.sessionOutputTokenWindowOptions(llamaTokenScratchSession, { activeTokenCount: 3 }, "out"), {
  tokensLen: undefined,
  tokenLength: undefined,
  activeTokenLength: undefined,
  activeTokenCount: 3,
  output: "out",
}, "llama token scratch output options");
expectSame(llamaTokenScratchHelpers.sessionScalarTokenExecuteOptions(llamaTokenScratchSession, { output: "logits" }), {
  tokensLen: 1,
  tokenLength: undefined,
  activeTokenLength: undefined,
  activeTokenCount: undefined,
  output: "logits",
}, "llama token scratch scalar execute options");
expectSame(llamaTokenScratchHelpers.sessionScalarTokenOutputOptions(llamaTokenScratchSession, "target"), {
  tokensLen: 1,
  tokenLength: undefined,
  activeTokenLength: undefined,
  activeTokenCount: undefined,
  output: "target",
}, "llama token scratch scalar output options");
expectSame(llamaTokenScratchHelpers.sessionScalarTokenSampleOptions(llamaTokenScratchSession, { top_k: 5, temperature: 0.5, seed: 7 }), {
  tokensLen: 1,
  topK: undefined,
  top_k: 5,
  temperature: 0.5,
  seed: 7,
}, "llama token scratch sample options");
expectThrow(
  () => llamaTokenScratchHelpers.scalarTokenWindow(llamaTokenScratchSession, -1),
  "invalid token id: -1",
  "llama token scratch invalid token",
);
expectThrow(
  () => sessionFacade.createLlamaTokenScratchFacadeHelpers({ scalarTokenScratch() {} }),
  "createLlamaTokenScratchFacadeHelpers requires scalar token and option scratch callbacks",
  "llama token scratch facade requires callbacks",
);
const stepParamsError = stepParams.stepParamsValidationError("invalid-output", "bad output", {
  actualShape: [2, 2],
  expectedShape: [4],
});
const stepParamsRejected = stepParams.stepParamsCompatibilityResult({
  kind: "generic",
  signature: "kind=generic",
}, stepParamsError);
expectSame(stepParamsRejected.accepted, false, "step params rejected");
expectSame(stepParamsRejected.rejectionCode, "invalid-output", "step params rejection code");
expectSame(stepParamsRejected.diagnostics[0], {
  code: "invalid-output",
  message: "bad output",
  actualShape: [2, 2],
  expectedShape: [4],
}, "step params diagnostic details");
expectSame(Object.isFrozen(stepParamsRejected.diagnostics), true, "step params diagnostics frozen");
expectSame(Object.isFrozen(stepParamsRejected.diagnostics[0].actualShape), true, "step params diagnostic shape frozen");
expectSame(stepParams.acceptsStepParamsCompatibility(stepParamsRejected), false, "step params rejected accepts predicate");
expectSame(stepParams.matchesStepParamsCompatibility(stepParamsAccepted, null), false, "step params missing compatibility predicate");
expectSame(stepParams.validateLlamaTokenStepParams({ token: 1 }, "step"), { hasTokens: false, hasToken: true }, "step params token validation");
expectSame(stepParams.validateLlamaTokenStepParams({ tokens: new Uint32Array([1]) }, "step"), { hasTokens: true, hasToken: false }, "step params tokens validation");
expectThrow(
  () => stepParams.validateLlamaTokenStepParams({ token: 1, tokens: new Uint32Array([1]) }, "step"),
  "session.step requires exactly one of token or tokens",
  "step params token choice mismatch",
);
expectThrow(
  () => stepParams.validateLlamaTokenStepParams(null, "step"),
  "session.step requires a StepParams object",
  "step params invalid object",
);
stepParams.validateLogitsOutputBuffer(new Float32Array(3), 3, "logits");
stepParams.validateLogitsOutputBuffer(false, 3, "logits", { allowFalse: true });
expectThrow(
  () => stepParams.validateLogitsOutputBuffer(new Float32Array(2), 3, "logits"),
  "logits output length 2 is smaller than logits length 3",
  "step params logits output too small",
);
stepParams.validateTokenWindowFitsContract({ remainingContext: 2, contextLength: 4, position: 2 }, 2, "tokens");
expectThrow(
  () => stepParams.validateTokenWindowFitsContract({ remainingContext: 1, contextLength: 4, position: 3 }, 2, "tokens"),
  "tokens length 2 exceeds remaining context 1",
  "step params token window too long",
);
expectSame(stepParams.validateExecuteStepParams(undefined, "execute"), {}, "step params default execute params");
expectThrow(
  () => stepParams.validateExecuteStepParams(new Float32Array(1), "execute"),
  "session.execute requires a StepParams object",
  "step params execute invalid object",
);
expectThrow(
  () => stepParams.assertNoInlineOutput({ output: new Float32Array(1) }, "executeInto"),
  "session.executeInto takes the output buffer as its first argument",
  "step params no inline output",
);
const tokenWindowScratch = {};
expectSame(stepParams.outputTokenWindowOptions({ tokensLen: 2, tokenLength: 9 }, tokenWindowScratch, false), {
  tokensLen: 2,
  tokenLength: 9,
  activeTokenLength: undefined,
  activeTokenCount: undefined,
  output: false,
}, "step params token window scratch");
expectSame(stepParams.scalarTokenExecuteOptions({ output: "out", activeTokenCount: 3 }, tokenWindowScratch), {
  tokensLen: 1,
  tokenLength: undefined,
  activeTokenLength: undefined,
  activeTokenCount: 3,
  output: "out",
}, "step params scalar execute scratch");
expectSame(stepParams.scalarTokenOutputOptions(tokenWindowScratch, "target"), {
  tokensLen: 1,
  tokenLength: undefined,
  activeTokenLength: undefined,
  activeTokenCount: undefined,
  output: "target",
}, "step params scalar output scratch");
const sampleScratch = {};
expectSame(stepParams.scalarTokenSampleOptions({ topK: 4, top_k: 5, temperature: 0.7, seed: 8 }, sampleScratch), {
  tokensLen: 1,
  topK: 4,
  top_k: 5,
  temperature: 0.7,
  seed: 8,
}, "step params scalar sample scratch");

const sessionSizing = sessionContract.sessionBufferSizing({
  kind: "session-buffer-sizing",
  scalarType: "f32",
  scalarBytes: 4,
  inputLen: 2,
  inputByteLength: 8,
  outputLen: 3,
  outputByteLength: 12,
  weightsLen: 6,
  weightsByteLength: 24,
  biasLen: 3,
  biasByteLength: 12,
  parameterLen: 9,
  parameterByteLength: 36,
  modelKind: "tiny-linear",
});
expectSame(sessionSizing.signature, "session-buffer-sizing|model=tiny-linear|scalar=f32|scalarBytes=4|input=2|inputBytes=8|output=3|outputBytes=12|weights=6|weightsBytes=24|bias=3|biasBytes=12|parameters=9|parameterBytes=36", "session contract sizing signature");
expectSame(Object.isFrozen(sessionSizing), true, "session contract sizing frozen");
const llamaStepContract = sessionContract.createLlamaSessionStepContract({
  vocabSize: 5,
  outputShape: [5],
  outputByteLength: 20,
  contextLength: 8,
  position: 3,
  hasBoundOutput: true,
});
expectSame(llamaStepContract.defaultOutput, "bound-native-readback", "llama session contract default output");
expectSame(llamaStepContract.remainingContext, 5, "llama session contract remaining context");
expectSame(llamaStepContract.signature.includes("boundOutput=native|defaultOutput=bound-native-readback"), true, "llama session contract signature");
const llamaTokenCompatibility = sessionValues.llamaSessionStepParamsCompatibility(llamaStepContract, {
  token: 7,
  output: new Float32Array(5),
});
expectSame(llamaTokenCompatibility.accepted, true, "session values llama token compatibility accepted");
expectSame(llamaTokenCompatibility.inputSource, "token", "session values llama token input source");
expectSame(llamaTokenCompatibility.outputTarget, "inline", "session values llama token output target");
expectSame(llamaTokenCompatibility.inputByteLength, 4, "session values llama token input bytes");
const llamaTokensCompatibility = sessionValues.llamaSessionStepParamsCompatibility(llamaStepContract, {
  tokens: [1, 2, 3],
  tokensLen: 2,
  output: false,
});
expectSame(llamaTokensCompatibility.accepted, true, "session values llama tokens compatibility accepted");
expectSame(llamaTokensCompatibility.inputSource, "tokens", "session values llama tokens input source");
expectSame(llamaTokensCompatibility.inputElementLength, 2, "session values llama tokens input length");
expectSame(llamaTokensCompatibility.outputTarget, "none", "session values llama tokens no output target");
expectSame(llamaTokensCompatibility.outputShape, [], "session values llama tokens no output shape");
const llamaInvalidOutputCompatibility = sessionValues.llamaSessionStepParamsCompatibility(llamaStepContract, {
  token: 7,
  output: new Float32Array(4),
});
expectSame(llamaInvalidOutputCompatibility.accepted, false, "session values llama output rejection");
expectSame(llamaInvalidOutputCompatibility.rejectionCode, "invalid-output", "session values llama output rejection code");
const llamaContextRejectedCompatibility = sessionValues.llamaSessionStepParamsCompatibility(llamaStepContract, {
  tokens: [1, 2, 3, 4, 5, 6],
});
expectSame(llamaContextRejectedCompatibility.accepted, false, "session values llama context rejection");
expectSame(llamaContextRejectedCompatibility.rejectionCode, "invalid-token-window", "session values llama context rejection code");
const llamaReadTensorTarget = sessionValues.llamaSessionReadOutputTensorTarget(5, { length: 3, byteOffset: 4 }, () => new Float32Array([1, 2, 3, 4]));
expectSame(llamaReadTensorTarget.length, 3, "session values llama read tensor length");
expectSame(llamaReadTensorTarget.byteOffset, 4, "session values llama read tensor byte offset");
expectSame(sessionValues.genericSessionActiveOutputValues(llamaReadTensorTarget.values, llamaReadTensorTarget.length), [1, 2, 3], "session values llama read tensor active values");
const llamaAllocatedExecuteTarget = sessionValues.llamaSessionExecuteTokenOutputTarget(5, {}, null);
expectSame(llamaAllocatedExecuteTarget.wantsLogits, true, "session values llama execute wants logits");
expectSame(llamaAllocatedExecuteTarget.output.length, 5, "session values llama execute allocates logits");
expectSame(llamaAllocatedExecuteTarget.descOutput.length, 5, "session values llama execute desc output allocated");
const llamaInlineExecuteTarget = sessionValues.llamaSessionExecuteTokenOutputTarget(5, { output: new Float32Array([1, 2, 3, 4, 5, 6]) }, null);
expectSame(llamaInlineExecuteTarget.output, [1, 2, 3, 4, 5, 6], "session values llama execute inline output");
expectSame(sessionValues.llamaSessionExecuteTokenResult(5, llamaInlineExecuteTarget.output, null, true, false), [1, 2, 3, 4, 5], "session values llama execute inline result trim");
const llamaNativeReadback = { readFloat32: (length) => Float32Array.from(Array.from({ length }, (_, index) => index + 20)) };
const llamaNativeExecuteTarget = sessionValues.llamaSessionExecuteTokenOutputTarget(5, {}, llamaNativeReadback);
expectSame(llamaNativeExecuteTarget.output, null, "session values llama execute native output");
expectSame(llamaNativeExecuteTarget.descOutput, null, "session values llama execute native desc output");
expectSame(llamaNativeExecuteTarget.useNativeOutput, true, "session values llama execute native flag");
expectSame(sessionValues.llamaSessionExecuteTokenResult(3, llamaNativeExecuteTarget.output, llamaNativeReadback, true, true), [20, 21, 22], "session values llama execute native result");
const llamaNoOutputExecuteTarget = sessionValues.llamaSessionExecuteTokenOutputTarget(5, { output: false }, llamaNativeReadback);
expectSame(llamaNoOutputExecuteTarget.wantsLogits, false, "session values llama execute no output flag");
expectSame(llamaNoOutputExecuteTarget.output, null, "session values llama execute no output target");
expectSame(sessionValues.llamaSessionExecuteTokenResult(0, null, llamaNativeReadback, false, false), undefined, "session values llama execute no output result");
expectThrow(
  () => sessionValues.llamaSessionExecuteTokenOutputTarget(5, { output: new Float32Array(4) }, null),
  "session.execute output length 4 is smaller than logits length 5",
  "session values llama execute invalid output",
);
expectSame(sessionValues.llamaSessionExecutePlan({ token: 1 }, "execute").hasTokens, false, "session values llama execute scalar plan");
expectSame(sessionValues.llamaSessionExecutePlan({ tokens: [1, 2] }, "execute").hasTokens, true, "session values llama execute tokens plan");
const llamaExecuteTensorPlan = sessionValues.llamaSessionExecuteTensorPlan({ token: 1, output: "ignored" }, () => new Float32Array(5));
expectSame(llamaExecuteTensorPlan.hasTokens, false, "session values llama execute tensor scalar plan");
expectSame(llamaExecuteTensorPlan.stepParams.output, [0, 0, 0, 0, 0], "session values llama execute tensor output target");
expectThrow(
  () => sessionValues.requireLlamaSessionLogitsOutput(undefined),
  "session.executeTensor requires logits output",
  "session values llama require logits output",
);
expectSame(sessionValues.requireLlamaSessionLogitsOutput(new Float32Array([1])), [1], "session values llama logits output ok");
expectThrow(
  () => sessionValues.llamaSessionExecuteIntoPlan({ token: 1, output: new Float32Array(5) }),
  "session.executeInto takes the output buffer as its first argument",
  "session values llama execute into rejects inline output",
);
expectSame(sessionValues.llamaSessionExecuteIntoPlan({ token: 1 }).hasTokens, false, "session values llama execute into scalar plan");
expectSame(sessionValues.llamaSessionTensorOutputTarget({ output: "x" }, (options) => options.output), "x", "session values llama tensor output target");
expectSame(sessionValues.llamaSessionStepTensorOutputTarget(5, { output: new Float32Array(5) }, (options) => options.output), [0, 0, 0, 0, 0], "session values llama step tensor output target");
expectThrow(
  () => sessionValues.llamaSessionStepTensorOutputTarget(5, { output: new Float32Array(4) }, (options) => options.output),
  "session.stepTensor output length 4 is smaller than logits length 5",
  "session values llama step tensor invalid output",
);
const llamaPrefillTensorPlan = sessionValues.llamaSessionPrefillTensorPlan({ output: "out", shape: [5] }, (options) => options.output);
expectSame(llamaPrefillTensorPlan.options.shape, [5], "session values llama prefill tensor plan options");
expectSame(llamaPrefillTensorPlan.output, "out", "session values llama prefill tensor output");
const generatedOutput = sessionValues.llamaSessionGeneratedOutput(3);
expectSame(generatedOutput.length, 3, "session values llama generated output length");
const callerGenerationOutput = new Uint32Array([1, 2]);
if (sessionValues.llamaSessionGenerationOutput(callerGenerationOutput) !== callerGenerationOutput) {
  throw new Error("session values llama generation output must return caller storage");
}
expectSame(sessionValues.llamaSessionSampleOptions({ top_k: 2, temperature: 0.25, seed: 3 }), {
  topK: 2,
  top_k: 2,
  temperature: 0.25,
  seed: 3,
  reserved: 0,
}, "session values llama sample options");
expectSame(sessionValues.llamaSessionArgmaxWindow([7, 8, 9], { tokensLen: 2 }), { tokens: [7, 8], tokensLen: 2 }, "session values llama argmax window");
expectSame(sessionValues.llamaSessionSampleWindow([4, 5, 6], { tokenLength: 2 }), { tokens: [4, 5], tokensLen: 2 }, "session values llama sample window");
const nativeOutputPredicate = (value) => Boolean(value && value.native);
expectSame(sessionValues.llamaSessionArgmaxExecutionPlan({ native: true }, nativeOutputPredicate, [7, 8], { tokensLen: 1 }).window.tokensLen, 1, "session values llama argmax execution plan");
const sampleExecutionPlan = sessionValues.llamaSessionSampleExecutionPlan({ native: true }, nativeOutputPredicate, [7, 8], { tokensLen: 1, topK: 2 });
expectSame(sampleExecutionPlan.window.tokensLen, 1, "session values llama sample execution window");
expectSame(sampleExecutionPlan.sampleOptions.topK, 2, "session values llama sample execution options");
expectThrow(
  () => sessionValues.llamaSessionArgmaxExecutionPlan(null, nativeOutputPredicate, [7], {}),
  "executeTokensArgmax requires a NativeBuffer output binding",
  "session values llama argmax execution requires native output",
);
const argmaxGenerationOut = new Uint32Array(3);
const argmaxGenerationPlan = sessionValues.llamaSessionArgmaxGenerationPlan([1, 2, 3], argmaxGenerationOut, { tokensLen: 2 });
expectSame(argmaxGenerationPlan.window.tokensLen, 2, "session values llama argmax generation window");
if (argmaxGenerationPlan.outputTokens !== argmaxGenerationOut) {
  throw new Error("session values llama argmax generation output must preserve caller storage");
}
expectSame(sessionValues.llamaSessionNativeArgmaxGenerationPlan({ native: true }, nativeOutputPredicate, [1, 2], argmaxGenerationOut, { tokensLen: 1 }).window.tokensLen, 1, "session values llama native argmax generation plan");
const sampleGenerationOut = new Uint32Array(2);
const sampleGenerationPlan = sessionValues.llamaSessionSampleGenerationPlan([1, 2, 3], sampleGenerationOut, { tokensLen: 1, topK: 3, temperature: 0.5, seed: 4 });
expectSame(sampleGenerationPlan.window.tokensLen, 1, "session values llama sample generation window");
if (sampleGenerationPlan.outputTokens !== sampleGenerationOut) {
  throw new Error("session values llama sample generation output must preserve caller storage");
}
expectSame(sampleGenerationPlan.sampleOptions, {
  topK: 3,
  top_k: 3,
  temperature: 0.5,
  seed: 4,
  reserved: 0,
}, "session values llama sample generation options");
expectSame(sessionValues.llamaSessionNativeSampleGenerationPlan({ native: true }, nativeOutputPredicate, [1, 2], sampleGenerationOut, { tokensLen: 1 }).window.tokensLen, 1, "session values llama native sample generation plan");
expectThrow(
  () => sessionValues.llamaSessionNativeSampleGenerationPlan(null, nativeOutputPredicate, [1], sampleGenerationOut, {}),
  "generateTokensSampleInto requires a NativeBuffer output binding",
  "session values llama sample generation requires native output",
);
expectThrow(
  () => sessionValues.llamaSessionGeneratedOutput(0),
  "maxTokens must be a positive safe integer, got 0",
  "session values llama generated output invalid max",
);
expectThrow(
  () => sessionValues.llamaSessionGenerationOutput(new Uint32Array(0)),
  "generated token output must not be empty",
  "session values llama generation output empty",
);
expectThrow(
  () => sessionValues.llamaSessionSampleOptions({ topK: 0 }),
  "sample topK must be an integer in [1, 256], got 0",
  "session values llama sample options invalid topK",
);
const genericStepContract = sessionContract.createGenericSessionStepContract({
  inputLen: 2,
  outputLen: 3,
  inputShape: [1, 2],
  outputShape: [3],
  programOutputShape: [3],
  inputByteLength: 8,
  outputByteLength: 12,
  boundInput: "host",
  boundOutput: "host",
});
expectSame(genericStepContract.defaultOutput, "bound-host", "generic session contract default output");
expectSame(genericStepContract.acceptsNoInput, true, "generic session contract accepts bound input");
expectSame(genericStepContract.defaultHotPath, true, "generic session contract bound host hot path");
expectSame(genericStepContract.signature.includes("inputShape=1x2|inputBytes=8"), true, "generic session contract signature");
expectSame(sessionStepIo.hasSessionStepIo(null, null), false, "session step io detects empty descriptor");
expectSame(sessionStepIo.hasSessionStepIo(new Float32Array(2), null), true, "session step io detects input descriptor");
const sessionStepFields = sessionStepIo.sessionStepIoFields(new Float32Array(2), new Float32Array(3));
expectSame(sessionStepFields, {
  input: new Float32Array(2),
  inputLen: 2,
  output: new Float32Array(3),
  outputLen: 3,
}, "session step io fields");
expectSame(Object.isFrozen(sessionStepFields), true, "session step io fields are frozen");
expectSame(sessionStepIo.sessionStepIoFields(null, null), null, "session step io fields empty");
expectSame(sessionStepIo.sessionStepNoOutputFields(new Float32Array(2)), {
  input: new Float32Array(2),
  inputLen: 2,
  output: null,
  outputLen: 0,
}, "session step no-output fields");
expectSame(sessionStepIo.sessionStepIoRecord(new Float32Array(2), new Float32Array(3)), {
  input: new Float32Array(2),
  input_len: 2,
  output: new Float32Array(3),
  output_len: 3,
}, "session step io record");
expectSame(sessionStepIo.sessionStepNoOutputRecord(new Float32Array(2)), {
  input: new Float32Array(2),
  input_len: 2,
  output: null,
  output_len: 0,
}, "session step no-output record");
expectSame(sessionStepIo.outputLenFromStepResultRecord({ output_len: 7 }), 7, "session step record result output len");
expectSame(sessionStepIo.outputLenFromStepResultWords(BigUint64Array.of(8n)), 8, "session step word result output len");

const sessionProfileTarget = {};
expectSame(sessionProfile.sessionCallProfile(sessionProfileTarget).signature, "session-call-profile|stepCount=0|stepTensorCount=0|stepIntoCount=0|executeCount=0|executeTensorCount=0|executeIntoCount=0|prepareExecuteIntoCount=0|prefillCount=0|prefillTensorCount=0|prefillIntoCount=0|readOutputIntoCount=0|readOutputTensorCount=0|advanceCount=0|resetCount=0|uploadParametersCount=0|uploadParameterCount=0|uploadParameterByNameCount=0|uploadParameterRangeCount=0", "session profile empty signature");
sessionProfile.bumpSessionCallProfile(sessionProfileTarget, "stepCount");
sessionProfile.bumpSessionCallProfile(sessionProfileTarget, "readOutputTensorCount");
const sessionProfileSnapshot = sessionProfile.sessionCallProfile(sessionProfileTarget);
expectSame(sessionProfileSnapshot.kind, "zgml.session.call-profile", "session profile kind");
expectSame(sessionProfileSnapshot.stepCount, 1, "session profile step bump");
expectSame(sessionProfileSnapshot.readOutputTensorCount, 1, "session profile read tensor bump");
expectSame(sessionProfile.acceptsSessionCallProfile(sessionProfileSnapshot), true, "session profile predicate accepts snapshot");
expectSame(sessionProfile.requireSessionCallProfile(sessionProfileSnapshot), sessionProfileSnapshot, "session profile require returns evidence");
expectSame(sessionProfile.assertSessionCallProfile(sessionProfileSnapshot), sessionProfileSnapshot, "session profile assert alias");
expectSame(sessionProfile.assert_session_call_profile(sessionProfileSnapshot), sessionProfileSnapshot, "session profile assert snake alias");
expectSame(sessionProfile.matchesSessionCallProfileSignature(sessionProfileTarget, sessionProfileSnapshot.signature), true, "session profile signature match");
expectSame(sessionProfile.matchesSessionCallProfileSignature(sessionProfileSnapshot, sessionProfileSnapshot.signature), true, "session profile signature match accepts snapshot");
expectSame(sessionNamespace.acceptsSessionCallProfile(sessionProfileSnapshot), true, "session namespace accepts Session call profile");
expectSame(sessionNamespace.requireSessionCallProfile(sessionProfileSnapshot), sessionProfileSnapshot, "session namespace requires Session call profile");
expectSame(inspectionNamespace.acceptsSessionCallProfile(sessionProfileSnapshot), true, "inspection namespace accepts Session call profile");
expectSame(inspectionNamespace.requireSessionCallProfile(sessionProfileSnapshot), sessionProfileSnapshot, "inspection namespace requires Session call profile");
expectThrow(
  () => sessionProfile.requireSessionCallProfile({ ...sessionProfileSnapshot, stepCount: -1 }),
  "SessionCallProfile is not valid evidence: stepCount is not a non-negative integer",
  "session profile require rejects malformed count",
);
expectSame(Object.isFrozen(sessionProfileSnapshot), true, "session profile snapshot frozen");
expectThrow(
  () => sessionProfile.bumpSessionCallProfile(sessionProfileTarget, "notAField"),
  "unknown Session call profile field: notAField",
  "session profile unknown field",
);
sessionProfile.resetSessionCallProfile(sessionProfileTarget);
expectSame(sessionProfile.sessionCallProfile(sessionProfileTarget).stepCount, 0, "session profile reset");

const sessionParameterLayout = {
  weightsLen: 6,
  biasLen: 3,
  parameters: [
    { name: "weight", binding: "weights", shape: [3, 2] },
    { name: "bias", binding: "bias", shape: [3] },
  ],
};
expectSame(sessionParameters.sessionParameterNames(sessionParameterLayout), ["weight", "bias"], "session parameter names");
expectSame(sessionParameters.sessionParameterInfos(sessionParameterLayout).length, 2, "session parameter infos");
expectSame(sessionParameters.sessionParameterInfo(sessionParameterLayout, "bias").binding, "bias", "session parameter info by name");
expectSame(sessionParameters.sessionParameterInfo(sessionParameterLayout, 0).name, "weight", "session parameter info by index");
expectSame(sessionParameters.sessionParameterInfo(null, "weight"), null, "session parameter missing layout");
expectSame(sessionParameters.sessionParameterPersistentBindingIndex(sessionParameterLayout.parameters[0], sessionParameterLayout), 0, "session parameter weights binding index");
expectSame(sessionParameters.sessionParameterPersistentBindingIndex(sessionParameterLayout.parameters[1], sessionParameterLayout), 1, "session parameter bias binding index");
expectSame(sessionParameters.sessionParameterInfoByNameForUpload(sessionParameterLayout, "weight").binding, "weights", "session parameter upload lookup");
expectThrow(
  () => sessionParameters.sessionParameterInfo(sessionParameterLayout, -1),
  "Session.parameterInfo index must be a non-negative integer",
  "session parameter invalid index",
);
expectThrow(
  () => sessionParameters.sessionParameterInfo(sessionParameterLayout, {}),
  "Session.parameterInfo requires a parameter name or index",
  "session parameter invalid key",
);
expectThrow(
  () => sessionParameters.sessionParameterInfoByNameForUpload(sessionParameterLayout, ""),
  "Session.uploadParameterByName requires a non-empty parameter name",
  "session parameter empty upload name",
);
expectThrow(
  () => sessionParameters.sessionParameterInfoByNameForUpload(sessionParameterLayout, "missing"),
  "Session parameter missing was not found",
  "session parameter missing upload name",
);
expectThrow(
  () => sessionParameters.sessionParameterPersistentBindingIndex({ name: "scale", binding: "scale" }, sessionParameterLayout),
  "Session parameter scale uses unsupported binding scale",
  "session parameter unsupported binding",
);

const sessionLayoutSmoke = {
  slots: [
    { name: "input", role: "step-input", elementCount: 2, byteLength: 8 },
    { name: "output", role: "step-output", elementCount: 3, byteLength: 12 },
    { name: "weights", role: "persistent", elementCount: 6, byteLength: 24 },
    { name: "bias", role: "persistent", elementCount: 3, byteLength: 12 },
  ],
  input: { name: "input", role: "step-input", elementCount: 2, byteLength: 8 },
  output: { name: "output", role: "step-output", elementCount: 3, byteLength: 12 },
  weights: { name: "weights", role: "persistent", elementCount: 6, byteLength: 24 },
  bias: { name: "bias", role: "persistent", elementCount: 3, byteLength: 12 },
};
expectSame(sessionLayout.sessionBufferSlotNames(sessionLayoutSmoke), ["input", "output", "weights", "bias"], "session layout slot names");
expectSame(sessionLayout.sessionBufferSlot(sessionLayoutSmoke, "weights").byteLength, 24, "session layout slot lookup");
expectSame(sessionLayout.requireSessionKvCacheLayout({ key: 1 }), { key: 1 }, "session layout kv cache");
expectThrow(
  () => sessionLayout.requireSessionKvCacheLayout(null),
  "LLaMA Session kvCacheLayout is unavailable",
  "session layout missing kv cache",
);
sessionLayout.requireNativeOutputBinding({ native: true }, (value) => value.native === true, "argmax");
expectThrow(
  () => sessionLayout.requireNativeOutputBinding({}, () => false, "argmax"),
  "argmax requires a NativeBuffer output binding",
  "session layout require native output",
);
const defaultLlamaLayout = sessionLayout.defaultLlamaSessionBufferLayout(4);
expectSame(defaultLlamaLayout.output.byteLength, 16, "session layout default llama output bytes");
const suppliedLlamaLayout = defaultLlamaLayout;
expectSame(sessionLayout.defaultLlamaSessionBufferLayout(4, suppliedLlamaLayout), suppliedLlamaLayout, "session layout supplied llama layout");
const llamaScratch = sessionLayout.createLlamaSessionScratch();
expectSame(llamaScratch.scalarTokenScratch.length, 1, "session layout llama scalar scratch");
expectSame(llamaScratch.noOutputTokenWindowOptionsScratch.output, false, "session layout llama no output scratch");
expectSame(llamaScratch.scalarTokenSampleOptionsScratch.tokensLen, 1, "session layout llama sample scratch");
expectSame(sessionLayout.llamaSessionOutputShape(7), [7], "session layout llama output shape");
expectSame(sessionLayout.llamaSessionInputLen(), 0, "session layout llama input len");
expectSame(sessionLayout.llamaSessionOutputByteLength(7), 28, "session layout llama output bytes");
expectSame(sessionLayout.sessionWeightsLen(sessionLayoutSmoke), 6, "session layout weights len");
expectSame(sessionLayout.sessionBiasByteLength(sessionLayoutSmoke), 12, "session layout bias bytes");
expectSame(sessionLayout.sessionParameterLen(sessionLayoutSmoke), 9, "session layout parameter len");
expectSame(sessionLayout.sessionParameterByteLength(sessionLayoutSmoke), 36, "session layout parameter bytes");
expectSame(sessionLayout.genericSessionInputShape({ inputShape: [2, 1], inputLen: 2, outputLen: 3 }), [2, 1], "session layout generic input shape");
expectSame(sessionLayout.genericSessionOutputShape({ outputShape: [3], inputLen: 2, outputLen: 3 }, [1, 3]), [1, 3], "session layout generic bound output shape");
expectSame(sessionLayout.genericSessionInputByteLength({ inputLen: 2 }), 8, "session layout generic input bytes");
expectSame(sessionLayout.boundBufferKind(null, () => false), "none", "session layout bound none");
expectSame(sessionLayout.boundBufferKind({ native: true }, (value) => value.native === true), "native", "session layout bound native");
expectSame(sessionLayout.boundBufferKind(new Float32Array(1), () => false), "host", "session layout bound host");
expectSame(sessionLayout.hostBoundInput({ native: true }, (value) => value.native === true), null, "session layout host input native");
expectSame(sessionLayout.hostBoundInput(new Float32Array([1]), () => false), [1], "session layout host input host");
expectSame(sessionLayout.nativeBoundOutput({ native: true }, (value) => value.native === true), { native: true }, "session layout native output native");
expectSame(sessionLayout.nativeBoundOutput(new Float32Array(1), () => false), null, "session layout native output host");
expectSame(sessionLayout.hostBoundOutput(new Float32Array([2]), () => false), [2], "session layout host output host");
expectSame(sessionLayout.hostBoundOutput({ native: true }, (value) => value.native === true), null, "session layout host output native");

let sessionBindingBufferId = 0;
const sessionBindingDeps = {
  createProgramOutputBuffer: (handle) => ({ native: true, handle, id: ++sessionBindingBufferId, freed: false, free() { this.freed = true; } }),
  isNativeBuffer: (value) => Boolean(value && value.native),
};
const nativeBind = sessionBinding.normalizeLlamaBindOptions("program-handle", { output: "native", label: "x" }, sessionBindingDeps);
expectSame(nativeBind.options.label, "x", "session binding preserves bind options");
expectSame(nativeBind.options.output.native, true, "session binding native output option");
expectSame(nativeBind.boundOutput.id, nativeBind.ownedOutput.id, "session binding owns allocated native output");
const callerNativeOutput = { native: true, id: "caller" };
const callerNativeBind = sessionBinding.normalizeLlamaBindOptions("program-handle", { output: callerNativeOutput }, sessionBindingDeps);
expectSame(callerNativeBind.boundOutput, callerNativeOutput, "session binding caller native output");
expectSame(callerNativeBind.ownedOutput, null, "session binding caller native is not owned");
const hostOutputBind = sessionBinding.normalizeLlamaBindOptions("program-handle", { output: new Float32Array(2) }, sessionBindingDeps);
expectSame(hostOutputBind.boundOutput, null, "session binding host output is not persistent");
expectSame(hostOutputBind.ownedOutput, null, "session binding host output has no ownership");
const sessionBindingProgram = {
  inspect: () => ({ vocabSize: 4 }),
  bufferLayout: () => ({ output: { byteLength: 16 } }),
  kvCacheLayout: () => ({ key: { byteLength: 8 } }),
};
const boundSession = sessionBinding.bindLlamaProgramSession("program-handle", sessionBindingProgram, { output: true }, {
  ...sessionBindingDeps,
  bindSessionHandle: (handle, bindOptions, inspect) => ({ sessionHandle: `${handle}:session`, output: bindOptions.output, vocabSize: inspect.vocabSize }),
  createSession: (sessionHandle, inspect, boundOutput, ownedOutput, layout, kvLayout) => ({
    sessionHandle,
    inspect,
    boundOutput,
    ownedOutput,
    layout,
    kvLayout,
  }),
});
expectSame(boundSession.sessionHandle.sessionHandle, "program-handle:session", "session binding session handle");
expectSame(boundSession.boundOutput.native, true, "session binding bound allocated output");
expectSame(boundSession.ownedOutput.native, true, "session binding owned allocated output");
expectSame(boundSession.layout.output.byteLength, 16, "session binding buffer layout");
expectSame(boundSession.kvLayout.key.byteLength, 8, "session binding kv layout");
let freedOwnedOutput = null;
expectThrow(
  () => sessionBinding.bindLlamaProgramSession("program-handle", sessionBindingProgram, { output: "native" }, {
    createProgramOutputBuffer: (handle) => {
      freedOwnedOutput = { native: true, handle, freed: false, free() { this.freed = true; } };
      return freedOwnedOutput;
    },
    isNativeBuffer: (value) => Boolean(value && value.native),
    bindSessionHandle: () => ({ sessionHandle: "x" }),
    createSession: () => {
      throw new Error("create failed");
    },
  }),
  "create failed",
  "session binding create failure",
);
expectSame(freedOwnedOutput.freed, true, "session binding frees owned output on failure");
expectThrow(
  () => sessionBinding.bindLlamaProgramSession("program-handle", {}, {}, {
    ...sessionBindingDeps,
    bindSessionHandle: () => ({}),
    createSession: () => ({}),
  }),
  "LLaMA Program.bind requires a Program with inspect, bufferLayout, and kvCacheLayout",
  "session binding invalid program",
);
expectThrow(
  () => sessionBinding.bindLlamaProgramSession("program-handle", sessionBindingProgram, {}, {
    ...sessionBindingDeps,
    createSession: () => ({}),
  }),
  "LLaMA Program.bind requires bindSessionHandle",
  "session binding missing bind callback",
);
const bindPolicyCalls = [];
const bindPolicy = {
  isNativeBuffer: (value) => Boolean(value && value.native),
  modelHandleForBind: (model) => model || null,
  hasSourceModel: (model) => model !== null,
  bindKvCache: (handle, options, inspect) => {
    bindPolicyCalls.push(["kv", handle, options.kvCache, inspect.kind]);
    return "kv";
  },
  bindModelKvCache: (handle, model, options, inspect) => {
    bindPolicyCalls.push(["model-kv", handle, model.id, options.kvCache, inspect.kind]);
    return "model-kv";
  },
  bindNativeOutput: (handle, options, inspect) => {
    bindPolicyCalls.push(["native-output", handle, options.output.id, inspect.kind]);
    return "native-output";
  },
  bindModelNativeOutput: (handle, model, options, inspect) => {
    bindPolicyCalls.push(["model-native-output", handle, model.id, options.output.id, inspect.kind]);
    return "model-native-output";
  },
  bindPlain: (handle, options, inspect) => {
    bindPolicyCalls.push(["plain", handle, options.name, inspect.kind]);
    return "plain";
  },
  bindModelPlain: (handle, model, options, inspect) => {
    bindPolicyCalls.push(["model-plain", handle, model.id, options.name, inspect.kind]);
    return "model-plain";
  },
};
expectSame(sessionBinding.bindLlamaSessionHandleByPolicy("program", { kvCache: "cache" }, { kind: "inspect" }, bindPolicy), "kv", "session binding kv route");
expectSame(sessionBinding.bindLlamaSessionHandleByPolicy("program", { kvCache: "cache", model: { id: "m" } }, { kind: "inspect" }, bindPolicy), "model-kv", "session binding model kv route");
expectSame(sessionBinding.bindLlamaSessionHandleByPolicy("program", { output: { native: true, id: "out" } }, { kind: "inspect" }, bindPolicy), "native-output", "session binding native output route");
expectSame(sessionBinding.bindLlamaSessionHandleByPolicy("program", { output: { native: true, id: "out" }, model: { id: "m" } }, { kind: "inspect" }, bindPolicy), "model-native-output", "session binding model native output route");
expectSame(sessionBinding.bindLlamaSessionHandleByPolicy("program", { name: "x" }, { kind: "inspect" }, bindPolicy), "plain", "session binding plain route");
expectSame(sessionBinding.bindLlamaSessionHandleByPolicy("program", { name: "x", model: { id: "m" } }, { kind: "inspect" }, bindPolicy), "model-plain", "session binding model plain route");
expectSame(bindPolicyCalls, [
  ["kv", "program", "cache", "inspect"],
  ["model-kv", "program", "m", "cache", "inspect"],
  ["native-output", "program", "out", "inspect"],
  ["model-native-output", "program", "m", "out", "inspect"],
  ["plain", "program", "x", "inspect"],
  ["model-plain", "program", "m", "x", "inspect"],
], "session binding route calls");
expectThrow(
  () => sessionBinding.bindLlamaSessionHandleByPolicy("program", { output: new Float32Array(1) }, { kind: "inspect" }, bindPolicy),
  "LLaMA persistent output binding requires a zgml NativeBuffer; pass Float32Array outputs per call",
  "session binding rejects host persistent output",
);
expectThrow(
  () => sessionBinding.bindLlamaSessionHandleByPolicy("program", {}, { kind: "inspect" }, { ...bindPolicy, bindPlain: undefined }),
  "LLaMA Session bind policy requires bindPlain",
  "session binding missing policy function",
);
const freedSessionHandles = [];
const lifecycleDeps = {
  nullSessionHandle: null,
  sessionFree: (handle) => freedSessionHandles.push(handle),
};
const llamaLifecycleOutput = { freed: false, free() { this.freed = true; } };
const llamaLifecycleSession = {
  handle: "llama-session",
  boundOutput: { native: true },
  ownedOutput: llamaLifecycleOutput,
};
sessionLifecycle.freeLlamaSessionResources(llamaLifecycleSession, lifecycleDeps);
expectSame(llamaLifecycleSession.handle, null, "session lifecycle llama handle cleared");
expectSame(llamaLifecycleSession.boundOutput, null, "session lifecycle llama bound output cleared");
expectSame(llamaLifecycleSession.ownedOutput, null, "session lifecycle llama owned output cleared");
expectSame(llamaLifecycleOutput.freed, true, "session lifecycle llama owned output freed");
expectSame(freedSessionHandles, ["llama-session"], "session lifecycle llama handle freed once");
sessionLifecycle.freeLlamaSessionResources(llamaLifecycleSession, lifecycleDeps);
expectSame(freedSessionHandles, ["llama-session"], "session lifecycle llama idempotent free");
const genericLifecycleBuffers = [
  { freed: false, free() { this.freed = true; } },
  { freed: false, free() { this.freed = true; } },
];
const genericLifecycleSession = {
  handle: "generic-session",
  ownedBuffers: genericLifecycleBuffers,
};
sessionLifecycle.freeGenericSessionResources(genericLifecycleSession, lifecycleDeps);
expectSame(genericLifecycleSession.handle, null, "session lifecycle generic handle cleared");
expectSame(genericLifecycleSession.ownedBuffers, [], "session lifecycle generic owned buffers cleared");
expectSame(genericLifecycleBuffers.map((buffer) => buffer.freed), [true, true], "session lifecycle generic owned buffers freed");
expectSame(freedSessionHandles, ["llama-session", "generic-session"], "session lifecycle generic handle freed");
sessionLifecycle.freeGenericSessionResources(genericLifecycleSession, lifecycleDeps);
expectSame(freedSessionHandles, ["llama-session", "generic-session"], "session lifecycle generic idempotent free");
const sessionValueDeps = shape.createShapedF32Helpers({ f32, prepareF32, label: "session values smoke" });
const sessionValueDesc = { inputLen: 2, outputLen: 3, inputShape: [2], outputShape: [3] };
expectSame(
  sessionValues.genericSessionExplicitInput({ desc: sessionValueDesc }, [1, 2], "session.step input", sessionValueDeps),
  [1, 2],
  "session values explicit input",
);
expectSame(
  sessionValues.genericSessionExplicitInput({ desc: sessionValueDesc, boundInput: [3, 4], hostBoundInput: new Float32Array([3, 4]) }, undefined, "session.step input", sessionValueDeps),
  [3, 4],
  "session values host bound input",
);
expectSame(
  sessionValues.genericSessionExplicitOutput(sessionValueDesc, [1, 2, 3, 4], "session.step output", sessionValueDeps),
  [1, 2, 3, 4],
  "session values explicit output capacity",
);
expectThrow(
  () => sessionValues.genericSessionExplicitInput({ desc: sessionValueDesc }, undefined, "session.step input", sessionValueDeps),
  "session.step input requires input because the Session has no bound input",
  "session values missing input",
);
expectThrow(
  () => sessionValues.genericSessionExplicitInput({ desc: sessionValueDesc }, [[1, 2]], "session.step input", sessionValueDeps),
  "session.step input shape must be [2] or flat length 2, got [1, 2]",
  "session values input shape mismatch",
);
expectThrow(
  () => sessionValues.genericSessionExplicitOutput(sessionValueDesc, [1, 2], "session.step output", sessionValueDeps),
  "session.step output length 2 is smaller than Program output length 3",
  "session values small output",
);
const sessionValueContract = sessionContract.createGenericSessionStepContract({
  inputLen: 2,
  outputLen: 3,
  inputShape: [2],
  outputShape: [3],
  programOutputShape: [3],
  inputByteLength: 8,
  outputByteLength: 12,
  boundInput: "none",
  boundOutput: "none",
});
const sessionValueCompatibility = sessionValues.genericSessionStepParamsCompatibility(
  { desc: sessionValueDesc },
  sessionValueContract,
  { input: [1, 2], output: [0, 0, 0] },
  sessionValueDeps,
);
expectSame(sessionValueCompatibility.accepted, true, "session values compatibility accepted");
expectSame(sessionValueCompatibility.inputSource, "inline", "session values compatibility input source");
expectSame(sessionValueCompatibility.outputTarget, "inline", "session values compatibility output target");
const sessionValueAdvanceCompatibility = sessionValues.genericSessionStepParamsCompatibility(
  { desc: sessionValueDesc },
  sessionValueContract,
  { input: [1, 2], output: false },
  sessionValueDeps,
);
expectSame(sessionValueAdvanceCompatibility.stateEffect, "advance", "session values compatibility no output effect");
expectSame(sessionValueAdvanceCompatibility.outputShape, [], "session values compatibility no output shape");
const sessionValueRejectedCompatibility = sessionValues.genericSessionStepParamsCompatibility(
  { desc: sessionValueDesc },
  sessionValueContract,
  {},
  sessionValueDeps,
);
expectSame(sessionValueRejectedCompatibility.accepted, false, "session values compatibility rejected");
expectSame(sessionValueRejectedCompatibility.rejectionCode, "invalid-input", "session values compatibility rejection code");
const inlineOutputTarget = sessionValues.genericSessionStepOutputTarget(
  sessionValueDesc,
  [9, 8, 7, 6],
  null,
  null,
  "session.step output",
  sessionValueDeps,
);
expectSame(inlineOutputTarget.output, [9, 8, 7, 6], "session values inline output target");
expectSame(inlineOutputTarget.descOutput, [9, 8, 7, 6], "session values inline desc output");
expectSame(sessionValues.genericSessionStepResult(3, inlineOutputTarget.output, null, false), [9, 8, 7], "session values trims output result");
const hostBoundTarget = sessionValues.genericSessionStepOutputTarget(
  sessionValueDesc,
  undefined,
  null,
  new Float32Array([1, 2, 3]),
  "session.step output",
  sessionValueDeps,
);
expectSame(hostBoundTarget.output, [1, 2, 3], "session values host bound output target");
const nativeReadback = {
  readFloat32: (length) => Float32Array.from(Array.from({ length }, (_, index) => index + 10)),
};
const nativeOutputTarget = sessionValues.genericSessionStepOutputTarget(
  sessionValueDesc,
  undefined,
  nativeReadback,
  new Float32Array([1, 2, 3]),
  "session.step output",
  sessionValueDeps,
);
expectSame(nativeOutputTarget.output, null, "session values native output target");
expectSame(nativeOutputTarget.descOutput, null, "session values native desc output");
expectSame(nativeOutputTarget.useNativeOutput, true, "session values native flag");
expectSame(sessionValues.genericSessionStepResult(3, nativeOutputTarget.output, nativeReadback, true), [10, 11, 12], "session values native readback result");
const readTensorTarget = sessionValues.genericSessionReadOutputTensorTarget(
  sessionValueDesc,
  { length: 2, byteOffset: 4 },
  () => new Float32Array([5, 6, 7]),
);
expectSame(readTensorTarget.length, 2, "session values read tensor length");
expectSame(readTensorTarget.byteOffset, 4, "session values read tensor byte offset");
expectSame(readTensorTarget.values, [5, 6, 7], "session values read tensor target values");
expectSame(sessionValues.genericSessionActiveOutputValues(readTensorTarget.values, readTensorTarget.length), [5, 6], "session values active read tensor values");
const allocatedReadTensorTarget = sessionValues.genericSessionReadOutputTensorTarget(
  sessionValueDesc,
  {},
  () => null,
);
expectSame(allocatedReadTensorTarget.length, 3, "session values allocated read tensor default length");
expectSame(allocatedReadTensorTarget.values.length, 3, "session values allocated read tensor target");
const executePlan = sessionValues.genericSessionExecutePlan({ input: [1, 2], output: [0, 0, 0] }, "execute");
expectSame(executePlan.input, [1, 2], "session values execute plan input");
expectSame(executePlan.output, [0, 0, 0], "session values execute plan output");
expectSame(executePlan.noOutput, false, "session values execute plan output flag");
const advancePlan = sessionValues.genericSessionExecutePlan({ input: [1, 2], output: false }, "execute");
expectSame(advancePlan.input, [1, 2], "session values advance plan input");
expectSame(advancePlan.output, false, "session values advance plan output");
expectSame(advancePlan.noOutput, true, "session values advance plan flag");
expectThrow(
  () => sessionValues.genericSessionExecuteTensorPlan({ input: [1, 2], output: false }),
  "session.executeTensor requires output",
  "session values execute tensor rejects no output",
);
expectThrow(
  () => sessionValues.genericSessionExecuteIntoPlan({ input: [1, 2], output: [0, 0, 0] }),
  "session.executeInto takes the output buffer as its first argument",
  "session values execute into rejects inline output",
);
expectSame(sessionValues.genericSessionExecuteIntoPlan({ input: [1, 2] }).input, [1, 2], "session values execute into plan input");
expectSame(sessionValues.assertGenericSessionNoOutputStepResult(0), undefined, "session values no output result ok");
expectThrow(
  () => sessionValues.assertGenericSessionNoOutputStepResult(1),
  "expected no-output tiny linear step, got 1 outputs",
  "session values no output result mismatch",
);

const kernelLayout = programBuffers.kernelBufferLayout(2, 3, 6, 3);
expectSame(kernelLayout.kind, "zgml.program.buffer-layout", "program buffers kernel layout kind");
expectSame(kernelLayout.signature, "zgml.program.buffer-layout|scalar=f32|scalarBytes=4|input:step-input:0:2:0:8|output:step-output:0:3:0:12|weights:persistent:0:6:0:24|bias:persistent:0:3:0:12", "program buffers kernel layout signature");
expectSame(kernelLayout.input.byteLength, 8, "program buffers kernel input bytes");
expectSame(programBuffers.freezeKernelBufferLayout(kernelLayout), {
  kind: "zgml.program.buffer-layout",
  scalarType: "f32",
  scalarBytes: 4,
  slots: [
    { name: "input", role: "step-input", scalarType: "f32", scalarBytes: 4, elementOffset: 0, elementCount: 2, byteOffset: 0, byteLength: 8 },
    { name: "output", role: "step-output", scalarType: "f32", scalarBytes: 4, elementOffset: 0, elementCount: 3, byteOffset: 0, byteLength: 12 },
    { name: "weights", role: "persistent", scalarType: "f32", scalarBytes: 4, elementOffset: 0, elementCount: 6, byteOffset: 0, byteLength: 24 },
    { name: "bias", role: "persistent", scalarType: "f32", scalarBytes: 4, elementOffset: 0, elementCount: 3, byteOffset: 0, byteLength: 12 },
  ],
  input: { name: "input", role: "step-input", scalarType: "f32", scalarBytes: 4, elementOffset: 0, elementCount: 2, byteOffset: 0, byteLength: 8 },
  output: { name: "output", role: "step-output", scalarType: "f32", scalarBytes: 4, elementOffset: 0, elementCount: 3, byteOffset: 0, byteLength: 12 },
  weights: { name: "weights", role: "persistent", scalarType: "f32", scalarBytes: 4, elementOffset: 0, elementCount: 6, byteOffset: 0, byteLength: 24 },
  bias: { name: "bias", role: "persistent", scalarType: "f32", scalarBytes: 4, elementOffset: 0, elementCount: 3, byteOffset: 0, byteLength: 12 },
  signature: "zgml.program.buffer-layout|scalar=f32|scalarBytes=4|input:step-input:0:2:0:8|output:step-output:0:3:0:12|weights:persistent:0:6:0:24|bias:persistent:0:3:0:12",
}, "program buffers frozen kernel layout");
const requirements = { scalarBytes: 4, inputLen: 2, outputLen: 3, weightsLen: 6, biasLen: 3 };
const programLayout = programBuffers.programBufferLayoutFromRequirements(requirements);
expectSame(Object.isFrozen(programLayout), true, "program buffers layout frozen");
expectSame(programLayout.kind, "zgml.program.buffer-layout", "program buffers layout kind");
expectSame(programLayout.signature, kernelLayout.signature, "program buffers layout signature");
expectSame(programBuffers.programBufferSlotNames(programLayout), ["input", "output", "weights", "bias"], "program buffers slot names");
expectSame(programBuffers.programBufferSlot(programLayout, "output").byteLength, 12, "program buffers slot lookup");
expectSame(programBuffers.programBufferFactorySlotFromRequirements(requirements, "weights").elementLength, 6, "program buffers factory slot");
const bufferFactory = () => ({ byteLength: 24 });
expectSame(programBuffers.requireProgramBufferResourceFactory("weights", bufferFactory), bufferFactory, "program buffers factory validation");
expectSame(programBuffers.assertProgramBufferSlotRequired(programLayout.output, "output"), programLayout.output, "program buffers required slot");
expectSame(programBuffers.assertProgramBufferResourceByteLength("weights buffer", { byteLength: 24 }, programLayout.weights), { byteLength: 24 }, "program buffers resource byte length");
expectThrow(
  () => programBuffers.programBufferSlot(programLayout, 1),
  "Program.bufferSlot requires a buffer slot name or kind",
  "program buffers invalid slot name",
);
expectThrow(
  () => programBuffers.requireProgramBufferResourceFactory("weights", null),
  "Program createWeightsBuffer resource option must be a function",
  "program buffers invalid factory",
);
expectThrow(
  () => programBuffers.assertProgramBufferResourceByteLength("weights buffer", { byteLength: 8 }, programLayout.weights),
  "weights buffer is too small: 8 < 24",
  "program buffers undersized resource",
);
const kvFactory = () => ({ byteLength: 1 });
expectSame(programBuffers.llamaKvCacheResourceFactory({ externalResource: kvFactory }), kvFactory, "program buffers kv factory");
const kvRequirements = { scalarBytes: 4, layers: 2, contextLength: 8, kBufferByteLength: 16, vBufferByteLength: 32 };
const kvLayout = programBuffers.llamaKvCacheLayoutFromRequirements(kvRequirements);
expectSame(kvLayout.kind, "zgml.llama.kv-cache.layout", "program buffers kv layout kind");
expectSame(kvLayout.signature, "zgml.llama.kv-cache.layout|scalar=f32|scalarBytes=4|layers=2|context=8|kBytes=16|vBytes=32|bufferBytes=96|0:kv-k:k:0:4:0:16|0:kv-v:v:0:8:0:32|1:kv-k:k:0:4:0:16|1:kv-v:v:0:8:0:32", "program buffers kv layout signature");
expectSame(kvLayout.slots.map((slot) => `${slot.layer}:${slot.name}:${slot.byteLength}`), ["0:kv-k:16", "0:kv-v:32", "1:kv-k:16", "1:kv-v:32"], "program buffers kv layout slots");
expectSame(programBuffers.llamaProgramKvCacheBufferSlot(kvRequirements, "kv-v").elementLength, 8, "program buffers kv slot element length");
expectSame(programBuffers.assertLlamaKvCacheResourceByteLength("kv", { byteLength: 32 }, 32), { byteLength: 32 }, "program buffers kv byte length");
expectThrow(
  () => programBuffers.llamaKvCacheResourceFactory({ resource: 1 }),
  "LLaMA createKvCache resource option must be a function",
  "program buffers invalid kv factory",
);

const bufferFactoryCalls = [];
const bufferFactoryRequirements = {
  scalarBytes: 4,
  inputLen: 2,
  outputLen: 3,
  weightsLen: 6,
  biasLen: 3,
};
const bufferFactoryKvRequirements = {
  scalarBytes: 4,
  layers: 1,
  contextLength: 4,
  kBufferByteLength: 16,
  vBufferByteLength: 32,
};
const bufferHelpers = programBufferFactory.createProgramBufferFactoryHelpers({
  programRequirements(handle) {
    bufferFactoryCalls.push(["requirements", handle]);
    return bufferFactoryRequirements;
  },
  llamaKvCacheRequirements(handle) {
    bufferFactoryCalls.push(["kv-requirements", handle]);
    return bufferFactoryKvRequirements;
  },
  createHostBuffer(handle, kind) {
    bufferFactoryCalls.push(["host", handle, kind]);
    return { route: "host", handle, kind };
  },
  createHostOutputBuffer(handle, kind) {
    bufferFactoryCalls.push(["host-output", handle, kind]);
    return { route: "host-output", handle, kind };
  },
  createDeviceBuffer(handle, kind, placement) {
    bufferFactoryCalls.push(["device", handle, kind, placement]);
    return { route: "device", handle, kind, placement };
  },
  requireNativeBuffer(buffer, kind) {
    bufferFactoryCalls.push(["native", kind, buffer.byteLength]);
    return buffer;
  },
  createLlamaKvCache(requirements, options, createSlotBuffer) {
    bufferFactoryCalls.push(["kv-cache", requirements.layers, options.placement ?? null, typeof createSlotBuffer]);
    return {
      route: "kv-cache",
      requirements,
      created: createSlotBuffer ? [
        createSlotBuffer({ kind: "k" }),
        createSlotBuffer({ kind: "v" }),
      ] : [],
    };
  },
});
expectSame(Object.isFrozen(bufferHelpers), true, "program buffer factory helpers frozen");
expectSame(bufferHelpers.createProgramOutputBuffer(9), { route: "host-output", handle: 9, kind: "output" }, "program buffer factory host output");
expectSame(bufferHelpers.createProgramBuffer(9, "weights"), { route: "host", handle: 9, kind: "weights" }, "program buffer factory host weights");
expectSame(bufferHelpers.createProgramNamedBuffer(9, "bias", { placement: "webgpu" }), { route: "device", handle: 9, kind: "bias", placement: "webgpu" }, "program buffer factory device named buffer");
let outputSlot = null;
const outputResource = bufferHelpers.createProgramOutputBuffer(9, {
  resource(slot) {
    outputSlot = slot;
    return { byteLength: slot.byteLength, tag: "output-resource" };
  },
});
expectSame({ resource: outputResource, slotKind: outputSlot.kind, slotBytes: outputSlot.byteLength }, {
  resource: { byteLength: 12, tag: "output-resource" },
  slotKind: "output",
  slotBytes: 12,
}, "program buffer factory output resource");
let weightsSlot = null;
const weightsResource = bufferHelpers.createProgramBuffer(9, "weights", {
  externalResource(slot) {
    weightsSlot = slot;
    return { byteLength: slot.byteLength, tag: "weights-resource" };
  },
});
expectSame({ resource: weightsResource, slotKind: weightsSlot.kind, slotBytes: weightsSlot.byteLength }, {
  resource: { byteLength: 24, tag: "weights-resource" },
  slotKind: "weights",
  slotBytes: 24,
}, "program buffer factory weights resource");
const kvResource = bufferHelpers.createProgramKvCacheBuffer(9, "kv-v", {
  resource(slot) {
    return { byteLength: slot.byteLength, tag: slot.kind };
  },
});
expectSame(kvResource, { byteLength: 32, tag: "kv-v" }, "program buffer factory kv resource");
expectSame(bufferHelpers.createProgramNamedBuffer(9, "kv-k", { device: "webgpu" }), {
  route: "device",
  handle: 9,
  kind: "kv-k",
  placement: "webgpu",
}, "program buffer factory kv device named buffer");
expectSame(bufferHelpers.createProgramLlamaKvCache(9, { placement: "webgpu" }), {
  route: "kv-cache",
  requirements: bufferFactoryKvRequirements,
  created: [
    { route: "device", handle: 9, kind: "kv-k", placement: "webgpu" },
    { route: "device", handle: 9, kind: "kv-v", placement: "webgpu" },
  ],
}, "program buffer factory kv cache device routing");
expectSame(bufferHelpers.createProgramLlamaKvCache(9, { resource: () => ({ byteLength: 64 }) }).created, [], "program buffer factory kv cache resource callback stays delegated");
expectThrow(
  () => programBufferFactory.createProgramBufferFactoryHelpers({}),
  "createProgramBufferFactoryHelpers requires programRequirements",
  "program buffer factory rejects missing requirements",
);
expectThrow(
  () => programBufferFactory.createProgramBufferFactoryHelpers({
    programRequirements: () => ({}),
    llamaKvCacheRequirements: () => ({}),
    createHostBuffer: () => ({}),
    createHostOutputBuffer: 1,
    createDeviceBuffer: () => ({}),
    requireNativeBuffer: () => ({}),
    createLlamaKvCache: () => ({}),
  }),
  "createProgramBufferFactoryHelpers createHostOutputBuffer must be a function",
  "program buffer factory rejects bad host output callback",
);
expectThrow(
  () => bufferHelpers.createProgramBuffer(9, "weights", { resource: () => ({ byteLength: 8 }) }),
  "weights is too small: 8 < 24",
  "program buffer factory rejects undersized resource",
);
expectThrow(
  () => bufferHelpers.createProgramNamedBuffer(9, "not-a-buffer"),
  "unknown Program buffer kind: not-a-buffer",
  "program buffer factory rejects invalid kind",
);
expectThrow(
  () => bufferHelpers.createProgramKvCacheBuffer(9, "kv-k", { resource: 1 }),
  "Program create kv-k resource option must be a function",
  "program buffer factory rejects bad kv resource",
);

const moduleBindFreed = [];
const moduleBindHelpers = programModuleBinding.createProgramModuleBindingHelpers({
  uniqueNativeBuffers(buffers) {
    return buffers.filter(Boolean);
  },
});
const moduleBindWeights = { kind: "weights", free: () => moduleBindFreed.push("weights") };
const moduleBindBias = { kind: "bias", free: () => moduleBindFreed.push("bias") };
const moduleBindModule = {
  placeParameters(program, options) {
    return { weights: moduleBindWeights, bias: moduleBindBias, programName: program.name, options };
  },
};
const moduleBindProgram = {
  name: "program",
  moduleCompatibility(module, options) {
    return { compatible: module === moduleBindModule && options.inputShape[0] === 2, reason: null };
  },
  bind(bindings) {
    return {
      bindings,
      _ownBuffers(ownedBuffers) {
        return { session: "bound", bindings, ownedBuffers };
      },
    };
  },
};
const moduleBindResult = moduleBindHelpers.bindModuleThroughProgram(moduleBindProgram, moduleBindModule, { inputShape: [2] });
expectSame(Object.isFrozen(moduleBindHelpers), true, "program module binding helpers frozen");
expectSame(moduleBindResult.session, "bound", "program module binding success");
expectSame(moduleBindResult.bindings.programName, "program", "program module binding placement gets program");
expectSame(moduleBindResult.bindings.options, { inputShape: [2] }, "program module binding placement gets options");
expectSame(moduleBindResult.ownedBuffers, [moduleBindWeights, moduleBindBias], "program module binding owns unique buffers");
const placedModuleParameterWrites = [];
const placedModule = {
  compileSupport(options) {
    return { supported: true, options };
  },
  bindParameters(options) {
    return {
      weights: [1, 2],
      bias: Float32Array.of(3),
      options,
    };
  },
};
const placedProgram = {
  inputShape() {
    return [2];
  },
  moduleCompatibility(module, options) {
    return { compatible: module === placedModule && options.inputShape[0] === 2, reason: "shape mismatch" };
  },
  bufferLayout() {
    return {
      weights: { scalarType: "f32", elementCount: 2 },
      bias: { scalarType: "f32", elementCount: 1 },
    };
  },
  createBuffer(name, options) {
    return {
      name,
      options,
      writeFloat32(values) {
        placedModuleParameterWrites.push([name, Array.from(values), options]);
      },
    };
  },
};
const placedModuleParameters = moduleBindings.placeModuleParameterBindings(placedModule, placedProgram, { weights: { placement: "webgpu" } });
expectSame(placedModuleParameterWrites, [
  ["weights", [1, 2], { placement: "webgpu" }],
  ["bias", [3], { weights: { placement: "webgpu" }, inputShape: [2] }],
], "module bindings place parameters writes buffers");
const placedMetadata = moduleBindings.moduleBindingMetadata(placedModuleParameters);
expectSame({
  hasWeights: Boolean(placedModuleParameters.weights),
  hasBias: Boolean(placedModuleParameters.bias),
  options: placedMetadata.options,
  supportOptions: placedMetadata.support.options,
}, {
  hasWeights: true,
  hasBias: true,
  options: { weights: { placement: "webgpu" }, inputShape: [2] },
  supportOptions: { weights: { placement: "webgpu" }, inputShape: [2] },
}, "module bindings place parameters annotates metadata");
expectThrow(
  () => moduleBindings.placeModuleParameterBindings({}, placedProgram),
  "nn.placeParameters requires a module with native Program support",
  "module bindings place parameters rejects bad module",
);
expectThrow(
  () => moduleBindings.placeModuleParameterBindings(placedModule, {}),
  "nn.placeParameters requires a compiled zgml Program",
  "module bindings place parameters rejects bad program",
);
expectThrow(
  () => moduleBindings.placeModuleParameterBindings(placedModule, { ...placedProgram, moduleCompatibility: () => ({ compatible: false, reason: "bad shape" }) }),
  "nn.placeParameters module is not compatible with this Program: bad shape",
  "module bindings place parameters rejects incompatible program",
);
expectThrow(
  () => moduleBindings.placeModuleParameterBindings(placedModule, { ...placedProgram, bufferLayout: () => ({ weights: { scalarType: "i32", elementCount: 2 } }) }),
  "nn.placeParameters weights slot dtype i32 is unsupported; eager zgml module parameters store f32 values",
  "module bindings place parameters rejects non-f32 slot",
);
expectThrow(
  () => moduleBindings.placeModuleParameterBindings(placedModule, { ...placedProgram, bufferLayout: () => ({ weights: { scalarType: "f32", elementCount: 3 } }) }),
  "nn.placeParameters weights length 2 does not match Program weights slot length 3",
  "module bindings place parameters rejects slot length mismatch",
);
const packedSequentialModule = moduleBindings.packedSequentialModuleParameters({
  kind: "module",
  desc: { weightsLen: 5, biasLen: 3 },
  entries: [
    { layer: { weight: Float32Array.of(1, 2), bias: Float32Array.of(3) } },
    { layer: { weight: Float32Array.of(4, 5, 6), bias: Float32Array.of(7, 8) } },
  ],
});
expectSame(packedSequentialModule, { weights: [1, 2, 4, 5, 6], bias: [3, 7, 8] }, "module bindings pack Sequential module parameters");
const packedSequentialNoBias = moduleBindings.packedSequentialModuleParameters({
  kind: "module",
  desc: { weightsLen: 2, biasLen: 0 },
  entries: [
    { layer: { weight: Float32Array.of(9, 10) } },
  ],
});
expectSame(Object.keys(packedSequentialNoBias), ["weights"], "module bindings pack Sequential omits empty bias");
expectSame(packedSequentialNoBias.weights, [9, 10], "module bindings pack Sequential no-bias weights");
const tinyLinearWeights = Float32Array.of(11, 12);
const tinyLinearBias = Float32Array.of(13);
expectSame(moduleBindings.packedSequentialProgramParameters({
  kind: "tiny-linear",
  layer: { bindParameters: () => ({ weights: tinyLinearWeights, bias: tinyLinearBias }) },
}), { weights: [11, 12], bias: [13] }, "module bindings pack tiny-linear parameters");
expectSame(moduleBindings.packedSequentialProgramParameters({
  kind: "module",
  desc: { weightsLen: 1, biasLen: 0 },
  entries: [{ layer: { weight: Float32Array.of(14) } }],
}), { weights: [14] }, "module bindings pack module program parameters");
expectThrow(
  () => moduleBindings.packedSequentialProgramParameters({ kind: "mystery" }),
  "unsupported compiled Sequential program kind: mystery",
  "module bindings pack rejects unknown Sequential kind",
);
class PolicyLinearModule {
  constructor(inFeatures, outFeatures, weight, bias) {
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.weight = weight;
    this.bias = bias;
  }
  parameters(prefix) {
    return [
      { name: `${prefix}.weight`, tensor: { shape: [this.outFeatures, this.inFeatures] }, layout: "row-major", data: this.weight },
    ];
  }
}
class PolicyActivationModule {
  constructor(kind = "relu") {
    this.kind = kind;
  }
}
class PolicySequentialModule {
  constructor(layers) {
    this.layers = layers;
  }
}
const policyLinear = new PolicyLinearModule(2, 3, Float32Array.of(1, 2, 3, 4, 5, 6), Float32Array.of(7, 8, 9));
const policyOptions = {
  LinearModule: PolicyLinearModule,
  ActivationModule: PolicyActivationModule,
  SequentialModule: PolicySequentialModule,
};
const policyNested = new PolicySequentialModule([
  policyLinear,
  new PolicySequentialModule([new PolicyActivationModule()]),
]);
const flattenedPolicyEntries = moduleCompilerPolicy.flattenSequentialEntries([policyNested], policyOptions);
expectSame(flattenedPolicyEntries.map((entry) => entry.path), [[0, 0], [0, 1, 0]], "module compiler policy flattens nested Sequential paths");
expectSame(moduleCompilerPolicy.flattenSequentialLayers([policyNested], policyOptions).length, 2, "module compiler policy flattens layers");
const policyLinearSpec = moduleCompilerPolicy.compiledSequentialLinearSpec([policyLinear], policyOptions);
expectSame({
  kind: policyLinearSpec.kind,
  nativePath: policyLinearSpec.nativePath,
  modelKind: policyLinearSpec.modelKind,
  layerCount: policyLinearSpec.layerCount,
}, {
  kind: "tiny-linear",
  nativePath: "tiny-linear",
  modelKind: "tiny-linear",
  layerCount: 1,
}, "module compiler policy detects tiny-linear spec");
expectSame(moduleCompilerPolicy.compiledSequentialProgramSpec([policyLinear], policyOptions).kind, "tiny-linear", "module compiler policy detects program spec");
expectSame(moduleCompilerPolicy.defaultTraceInputShape([{ layer: policyLinear, path: [0] }], policyOptions), [2], "module compiler policy defaults trace input shape");
expectSame(moduleCompilerPolicy.defaultTraceInputShape([{ layer: policyLinear, path: [0] }], { ...policyOptions, inputShape: [4, 2] }), [4, 2], "module compiler policy honors explicit trace input shape");
expectSame(moduleCompilerPolicy.sequentialProgramSupportDetails(policyLinearSpec), {
  nativePath: "tiny-linear",
  modelKind: "tiny-linear",
  layerCount: 1,
  inputLen: 2,
  outputLen: 3,
  weightsLen: 6,
  biasLen: 3,
}, "module compiler policy support details");
expectSame(moduleCompilerPolicy.traceCompilerUnsupportedReason({ diagnostic: { message: "nope" } }, "fallback"), "nope", "module compiler policy diagnostic reason");
expectSame(moduleCompilerPolicy.traceCompilerUnsupportedReason(null, "fallback"), "fallback", "module compiler policy fallback reason");
expectSame(moduleCompilerPolicy.layerParametersTrace(policyLinear, [1, 2]), [
  { name: "1.2.weight", shape: [3, 2], layout: "row-major", scalarCount: 6 },
], "module compiler policy parameter trace");
expectSame(moduleCompilerPolicy.normalizeTraceShape([1, 2], "trace"), [1, 2], "module compiler policy normalizes trace shape");
expectSame(moduleCompilerPolicy.copyShape([5, 6]), [5, 6], "module compiler policy copies shape");
expectSame(moduleCompilerPolicy.withTraceShapes({ op: "x" }, [2], [3]), { op: "x", inputShape: [2], outputShape: [3] }, "module compiler policy attaches trace shapes");
expectSame(moduleCompilerPolicy.singleLinearCompatibilityAllowed([policyLinear], { ...policyOptions, inputShape: [2] }, [2]), true, "module compiler policy allows compatible single linear");
expectSame(moduleCompilerPolicy.singleLinearCompatibilityAllowed([policyLinear], { ...policyOptions, inputShape: [4, 2] }, [4, 2]), false, "module compiler policy rejects batched tiny-linear compatibility path");
expectSame(moduleCompilerPolicy.inferLinearShape(policyLinear, [2]), [3], "module compiler policy infers vector linear shape");
expectSame(moduleCompilerPolicy.inferLinearShape(policyLinear, [4, 2]), [4, 3], "module compiler policy infers batch linear shape");
expectSame(moduleCompilerPolicy.inferEmbeddingShape({ embeddingDim: 5 }, [2, 3]), [2, 3, 5], "module compiler policy infers embedding shape");
expectSame(moduleCompilerPolicy.inferConv2dShape({
  inChannels: 1,
  outChannels: 2,
  kernelSize: [2, 2],
  stride: [1, 1],
  padding: [0, 0],
  dilation: [1, 1],
}, [1, 3, 3]), [2, 2, 2], "module compiler policy infers unbatched conv2d shape");
expectSame(moduleCompilerPolicy.inferConv2dShape({
  inChannels: 1,
  outChannels: 2,
  kernelSize: [3, 3],
  stride: [2, 2],
  padding: [1, 1],
  dilation: [1, 1],
}, [4, 1, 5, 5]), [4, 2, 3, 3], "module compiler policy infers batched conv2d shape");
expectSame(moduleCompilerPolicy.inferConv2dShape({
  inChannels: 1,
  outChannels: 1,
  kernelSize: [2, 2],
  stride: [2, 2],
  padding: [1, 1],
  dilation: [2, 2],
}, [1, 5, 5]), [1, 3, 3], "module compiler policy infers strided padded dilated conv2d shape");
expectSame(moduleCompilerPolicy.inferMaxPool2dShape({
  kernelSize: [2, 2],
  stride: [2, 2],
  padding: [0, 0],
  dilation: [1, 1],
  ceilMode: false,
}, [4, 2, 5, 5]), [4, 2, 2, 2], "module compiler policy infers batched maxPool2d shape");
expectSame(moduleCompilerPolicy.inferAvgPool2dShape({
  kernelSize: [2, 2],
  stride: [2, 2],
  padding: [0, 0],
  ceilMode: false,
}, [4, 2, 5, 5]), [4, 2, 2, 2], "module compiler policy infers batched avgPool2d shape");
expectSame(moduleCompilerPolicy.inferSoftmaxShape({ dim: -1 }, [2, 3]), [2, 3], "module compiler policy infers softmax shape");
expectSame(moduleCompilerPolicy.inferReductionShape({ kind: "sum", dim: -1 }, [2, 3]), [2, 1], "module compiler policy infers reduction shape");
expectSame(moduleCompilerPolicy.inferFeatureNormShape({ kind: "layerNorm", features: 3 }, [2, 3]), [2, 3], "module compiler policy infers feature norm shape");
expectSame(moduleCompilerPolicy.inferFeatureNormShape({ kind: "batchNorm1d", features: 3 }, [2, 3]), [2, 3], "module compiler policy infers batchNorm1d shape");
expectSame(moduleCompilerPolicy.inferShapeModuleShape({ kind: "view", shape: [3, 2] }, [2, 3]), [3, 2], "module compiler policy infers view shape");
expectSame(moduleCompilerPolicy.inferShapeModuleShape({ kind: "flatten", startDim: 1, endDim: -1 }, [2, 3, 4]), [2, 12], "module compiler policy infers flatten shape");
expectSame(moduleCompilerPolicy.inferShapeModuleShape({ kind: "squeeze", squeezeAll: false, dim: 1 }, [2, 1, 3]), [2, 3], "module compiler policy infers squeeze shape");
expectSame(moduleCompilerPolicy.inferShapeModuleShape({ kind: "unsqueeze", dim: -1 }, [2, 3]), [2, 3, 1], "module compiler policy infers unsqueeze shape");
expectSame(moduleCompilerPolicy.inferShapeModuleShape({ kind: "expand", shape: [2, 3] }, [1, 3]), [2, 3], "module compiler policy infers expand shape");
expectSame(moduleCompilerPolicy.inferShapeModuleShape({ kind: "narrow", dim: 1, start: 1, length: 2 }, [2, 4]), [2, 2], "module compiler policy infers narrow shape");
expectSame(moduleCompilerPolicy.inferShapeModuleShape({ kind: "select", dim: 1, index: -1 }, [2, 4]), [2], "module compiler policy infers select shape");
expectSame(moduleCompilerPolicy.inferShapeModuleShape({ kind: "slice", dim: 1, start: 1, end: 4, step: 2 }, [2, 5]), [2, 2], "module compiler policy infers slice shape");
expectSame(moduleCompilerPolicy.inferShapeModuleShape({ kind: "transpose", dim0: 0, dim1: 2 }, [2, 3, 4]), [4, 3, 2], "module compiler policy infers transpose shape");
expectSame(moduleCompilerPolicy.inferShapeModuleShape({ kind: "permute", dims: [1, 0, 2] }, [2, 3, 4]), [3, 2, 4], "module compiler policy infers permute shape");
const policyLinearTraceOp = moduleCompilerPolicy.traceOpForEntry({ layer: policyLinear, path: [0] }, 0, policyOptions, [2]);
expectSame({
  index: policyLinearTraceOp.index,
  path: policyLinearTraceOp.path,
  op: policyLinearTraceOp.op,
  inFeatures: policyLinearTraceOp.inFeatures,
  outFeatures: policyLinearTraceOp.outFeatures,
  bias: policyLinearTraceOp.bias,
  inputShape: policyLinearTraceOp.inputShape,
  outputShape: policyLinearTraceOp.outputShape,
  parameters: policyLinearTraceOp.parameters,
}, {
  index: 0,
  path: "0",
  op: "linear",
  inFeatures: 2,
  outFeatures: 3,
  bias: true,
  inputShape: [2],
  outputShape: [3],
  parameters: [{ name: "0.weight", shape: [3, 2], layout: "row-major", scalarCount: 6 }],
}, "module compiler policy builds linear trace op");
const policyConv2dTraceOp = moduleCompilerPolicy.traceOpForEntry({
  layer: {
    kind: "conv2d",
    inChannels: 1,
    outChannels: 2,
    kernelSize: [2, 2],
    stride: [1, 1],
    padding: [0, 0],
    dilation: [1, 1],
    groups: 1,
    weight: new Float32Array(8),
    parameters: () => [{ name: "0.weight", tensor: { shape: [2, 1, 2, 2] }, layout: "row-major:conv2d.weight", data: new Float32Array(8) }],
  },
  path: [0],
}, 0, policyOptions, [1, 3, 3]);
expectSame({
  op: policyConv2dTraceOp.op,
  inputShape: policyConv2dTraceOp.inputShape,
  outputShape: policyConv2dTraceOp.outputShape,
  parameters: policyConv2dTraceOp.parameters,
}, {
  op: "conv2d",
  inputShape: [1, 3, 3],
  outputShape: [2, 2, 2],
  parameters: [{ name: "0.weight", shape: [2, 1, 2, 2], layout: "row-major:conv2d.weight", scalarCount: 8 }],
}, "module compiler policy builds conv2d trace op with output shape");
const policyMaxPool2dTraceOp = moduleCompilerPolicy.traceOpForEntry({
  layer: {
    kind: "maxPool2d",
    kernelSize: [2, 2],
    stride: [2, 2],
    padding: [0, 0],
    dilation: [1, 1],
    ceilMode: false,
    parameters: () => [],
  },
  path: [1],
}, 1, policyOptions, [1, 4, 4]);
expectSame({
  op: policyMaxPool2dTraceOp.op,
  inputShape: policyMaxPool2dTraceOp.inputShape,
  outputShape: policyMaxPool2dTraceOp.outputShape,
  kernelSize: policyMaxPool2dTraceOp.kernelSize,
  stride: policyMaxPool2dTraceOp.stride,
  parameters: policyMaxPool2dTraceOp.parameters,
}, {
  op: "maxPool2d",
  inputShape: [1, 4, 4],
  outputShape: [1, 2, 2],
  kernelSize: [2, 2],
  stride: [2, 2],
  parameters: [],
}, "module compiler policy builds maxPool2d trace op with output shape");
const policyAvgPool2dTraceOp = moduleCompilerPolicy.traceOpForEntry({
  layer: {
    kind: "avgPool2d",
    kernelSize: [2, 2],
    stride: [2, 2],
    padding: [0, 0],
    ceilMode: false,
    countIncludePad: true,
    parameters: () => [],
  },
  path: [2],
}, 2, policyOptions, [1, 4, 4]);
expectSame({
  op: policyAvgPool2dTraceOp.op,
  inputShape: policyAvgPool2dTraceOp.inputShape,
  outputShape: policyAvgPool2dTraceOp.outputShape,
  kernelSize: policyAvgPool2dTraceOp.kernelSize,
  stride: policyAvgPool2dTraceOp.stride,
  countIncludePad: policyAvgPool2dTraceOp.countIncludePad,
  parameters: policyAvgPool2dTraceOp.parameters,
}, {
  op: "avgPool2d",
  inputShape: [1, 4, 4],
  outputShape: [1, 2, 2],
  kernelSize: [2, 2],
  stride: [2, 2],
  countIncludePad: true,
  parameters: [],
}, "module compiler policy builds avgPool2d trace op with output shape");
const policyBatchNorm1dTraceOp = moduleCompilerPolicy.traceOpForEntry({
  layer: {
    kind: "batchNorm1d",
    features: 3,
    eps: 1e-5,
    momentum: 0.1,
    trackRunningStats: true,
    training: false,
    weight: new Float32Array([1, 1, 1]),
    bias: new Float32Array([0, 0, 0]),
    parameters: () => [
      { name: "2.weight", tensor: { shape: [3] }, layout: "row-major:batchNorm1d.weight[features]", data: new Float32Array([1, 1, 1]) },
      { name: "2.bias", tensor: { shape: [3] }, layout: "row-major:batchNorm1d.bias[features]", data: new Float32Array([0, 0, 0]) },
    ],
  },
  path: [2],
}, 2, policyOptions, [2, 3]);
expectSame({
  op: policyBatchNorm1dTraceOp.op,
  inputShape: policyBatchNorm1dTraceOp.inputShape,
  outputShape: policyBatchNorm1dTraceOp.outputShape,
  features: policyBatchNorm1dTraceOp.features,
  momentum: policyBatchNorm1dTraceOp.momentum,
  trackRunningStats: policyBatchNorm1dTraceOp.trackRunningStats,
  training: policyBatchNorm1dTraceOp.training,
  parameters: policyBatchNorm1dTraceOp.parameters,
}, {
  op: "batchNorm1d",
  inputShape: [2, 3],
  outputShape: [2, 3],
  features: 3,
  momentum: 0.1,
  trackRunningStats: true,
  training: false,
  parameters: [
    { name: "2.weight", shape: [3], layout: "row-major:batchNorm1d.weight[features]", scalarCount: 3 },
    { name: "2.bias", shape: [3], layout: "row-major:batchNorm1d.bias[features]", scalarCount: 3 },
  ],
}, "module compiler policy builds batchNorm1d trace op with output shape");
const policySequentialTrace = moduleCompilerPolicy.traceSequentialEntries([
  { layer: policyLinear, path: [0] },
  { layer: new PolicyActivationModule(), path: [1] },
], policyOptions, { inputShape: [2] });
expectSame({
  kind: policySequentialTrace.kind,
  normalized: policySequentialTrace.normalized,
  layerCount: policySequentialTrace.layerCount,
  opCount: policySequentialTrace.opCount,
  parameterCount: policySequentialTrace.parameterCount,
  parameterScalarCount: policySequentialTrace.parameterScalarCount,
  shapeKnown: policySequentialTrace.shapeKnown,
  inputShape: policySequentialTrace.inputShape,
  outputShape: policySequentialTrace.outputShape,
  ops: policySequentialTrace.ops.map((op) => ({ op: op.op, path: op.path, inputShape: op.inputShape, outputShape: op.outputShape })),
}, {
  kind: "sequential",
  normalized: true,
  layerCount: 2,
  opCount: 2,
  parameterCount: 1,
  parameterScalarCount: 6,
  shapeKnown: true,
  inputShape: [2],
  outputShape: [3],
  ops: [
    { op: "linear", path: "0", inputShape: [2], outputShape: [3] },
    { op: "activation", path: "1", inputShape: [3], outputShape: [3] },
  ],
}, "module compiler policy builds Sequential trace");
const policyCompiledModule = moduleCompilerPolicy.compiledSequentialModuleSpec([{ layer: policyLinear, path: [0] }], policyOptions);
expectSame({
  kind: policyCompiledModule.kind,
  nativePath: policyCompiledModule.nativePath,
  modelKind: policyCompiledModule.modelKind,
  layerCount: policyCompiledModule.layerCount,
  desc: policyCompiledModule.desc && {
    inputLen: policyCompiledModule.desc.inputLen,
    outputLen: policyCompiledModule.desc.outputLen,
    weightsLen: policyCompiledModule.desc.weightsLen,
    biasLen: policyCompiledModule.desc.biasLen,
  },
}, {
  kind: "module",
  nativePath: "device-program",
  modelKind: "module",
  layerCount: 1,
  desc: { inputLen: 2, outputLen: 3, weightsLen: 6, biasLen: 0 },
}, "module compiler policy compiles Sequential module spec");
const policyCompiledNormalizedModule = moduleCompilerPolicy.compiledSequentialProgramSpecForNormalizedLayers(
  [policyLinear, new PolicyActivationModule()],
  { ...policyOptions, inputShape: [2] },
);
expectSame({
  kind: policyCompiledNormalizedModule.kind,
  nativePath: policyCompiledNormalizedModule.nativePath,
  modelKind: policyCompiledNormalizedModule.modelKind,
  layerCount: policyCompiledNormalizedModule.layerCount,
  opCount: policyCompiledNormalizedModule.ir.opCount,
  dispatchCount: policyCompiledNormalizedModule.kernelPlan.dispatchCount,
  desc: policyCompiledNormalizedModule.desc && {
    inputLen: policyCompiledNormalizedModule.desc.inputLen,
    outputLen: policyCompiledNormalizedModule.desc.outputLen,
    weightsLen: policyCompiledNormalizedModule.desc.weightsLen,
    biasLen: policyCompiledNormalizedModule.desc.biasLen,
  },
}, {
  kind: "module",
  nativePath: "device-program",
  modelKind: "module",
  layerCount: 2,
  opCount: 2,
  dispatchCount: 1,
  desc: { inputLen: 2, outputLen: 3, weightsLen: 6, biasLen: 0 },
}, "module compiler policy compiles normalized layer lists through trace artifacts");
const policyFrozenTrace = moduleCompilerPolicy.traceSequentialProgram([policyLinear], policyOptions, { inputShape: [2] });
expectSame({
  frozen: Object.isFrozen(policyFrozenTrace),
  opFrozen: Object.isFrozen(policyFrozenTrace.ops[0]),
  outputShape: policyFrozenTrace.outputShape,
}, {
  frozen: true,
  opFrozen: true,
  outputShape: [3],
}, "module compiler policy traces frozen Sequential program");
const policyAnalysis = moduleCompilerPolicy.analyzeSequentialProgram([policyLinear], policyOptions);
expectSame({
  supported: policyAnalysis.supported,
  reason: policyAnalysis.reason,
  compiledKind: policyAnalysis.compiled.kind,
  hasSupport: Boolean(policyAnalysis.support),
}, {
  supported: true,
  reason: null,
  compiledKind: "tiny-linear",
  hasSupport: true,
}, "module compiler policy analyzes supported Sequential program");
const policyCompiler = moduleCompilerPolicy.createTraceModuleCompiler({
  getConstructors: () => policyOptions,
});
expectSame(policyCompiler.trace([policyLinear], { inputShape: [2] }).outputShape, [3], "module compiler policy factory traces");
expectSame(policyCompiler.analyze([policyLinear]).supported, true, "module compiler policy factory analyzes");
expectSame(policyCompiler.packParameters({ kind: "tiny-linear", layer: { bindParameters: () => ({ weights: Float32Array.of(1) }) } }), { weights: [1] }, "module compiler policy factory packs parameters");
const nodeCompilerForwardCalls = [];
const nodeSmokeTraceModuleCompiler = {
  analyze(layers, options) {
    nodeCompilerForwardCalls.push(["analyze", layers, options]);
    return { layers, options, kind: "analysis" };
  },
  analyzeSingle(layer, options) {
    nodeCompilerForwardCalls.push(["analyzeSingle", layer, options]);
    return { layer, options, kind: "single-analysis" };
  },
  trace(layers, options) {
    nodeCompilerForwardCalls.push(["trace", layers, options]);
    return { layers, options, kind: "trace" };
  },
  packParameters(spec) {
    nodeCompilerForwardCalls.push(["pack", spec]);
    return { spec, kind: "packed" };
  },
};
const adapterModuleCompilerSurface = createAdapterModuleCompilerSurface({
  getTraceModuleCompiler: () => nodeSmokeTraceModuleCompiler,
  uninitializedMessage: "Adapter Module compiler surface is not initialized",
});
expectSame(adapterModuleCompilerSurface.analyzeSequentialProgram(["linear"], { backend: "cpu" }), { layers: ["linear"], options: { backend: "cpu" }, kind: "analysis" }, "Adapter module compiler surface forwards Sequential analysis");
expectSame(adapterModuleCompilerSurface.analyzeSingleModuleProgram("linear"), { layer: "linear", options: {}, kind: "single-analysis" }, "Adapter module compiler surface defaults single-module options");
expectSame(adapterModuleCompilerSurface.traceSequentialProgram(["linear"]), { layers: ["linear"], options: {}, kind: "trace" }, "Adapter module compiler surface defaults trace options");
expectSame(adapterModuleCompilerSurface.packedSequentialProgramParameters({ kind: "tiny-linear" }), { spec: { kind: "tiny-linear" }, kind: "packed" }, "Adapter module compiler surface forwards parameter packing");
expectSame(nodeCompilerForwardCalls.map((call) => call[0]), ["analyze", "analyzeSingle", "trace", "pack"], "Adapter module compiler surface call order");
expectThrow(
  () => createAdapterModuleCompilerSurface({
    getTraceModuleCompiler: () => null,
    uninitializedMessage: "Adapter Module compiler surface is not initialized",
  }).traceSequentialProgram([]),
  "Adapter Module compiler surface is not initialized",
  "Adapter module compiler surface requires initialization",
);
const hostCompilerForwardCalls = [];
const hostModuleCompilerSurface = createHostModuleCompilerSurface({
  getTraceModuleCompiler: () => ({
    analyze(layers, options) {
      hostCompilerForwardCalls.push(["analyze", layers, options]);
      return { layers, options, host: true };
    },
    analyzeSingle(layer, options) {
      hostCompilerForwardCalls.push(["single", layer, options]);
      return { layer, options, host: true };
    },
    trace(layers, options) {
      hostCompilerForwardCalls.push(["trace", layers, options]);
      return { layers, options, host: true };
    },
    packParameters(spec) {
      hostCompilerForwardCalls.push(["pack", spec]);
      return { spec, host: true };
    },
  }),
  uninitializedMessage: "host compiler missing",
});
expectSame(hostModuleCompilerSurface.analyzeSequentialProgram(["relu"]), { layers: ["relu"], options: {}, host: true }, "host Module compiler surface defaults analysis options");
expectSame(hostModuleCompilerSurface.analyzeSingleModuleProgram("relu", { backend: "cpu" }), { layer: "relu", options: { backend: "cpu" }, host: true }, "host Module compiler surface forwards single-module options");
expectSame(hostModuleCompilerSurface.traceSequentialProgram(["relu"]), { layers: ["relu"], options: {}, host: true }, "host Module compiler surface defaults trace options");
expectSame(hostModuleCompilerSurface.packedSequentialProgramParameters({ kind: "host" }), { spec: { kind: "host" }, host: true }, "host Module compiler surface forwards parameter packing");
expectSame(hostCompilerForwardCalls.map((call) => call[0]), ["analyze", "single", "trace", "pack"], "host Module compiler surface call order");
expectThrow(
  () => createHostModuleCompilerSurface({ getTraceModuleCompiler: () => null, uninitializedMessage: "host compiler missing" }).traceSequentialProgram([]),
  "host compiler missing",
  "host Module compiler surface requires initialization",
);
const policyCycle = new PolicySequentialModule([]);
policyCycle.layers.push(policyCycle);
expectThrow(
  () => moduleCompilerPolicy.flattenSequentialEntries([policyCycle], policyOptions),
  "nested nn.Sequential graph contains a cycle",
  "module compiler policy rejects Sequential cycles",
);
expectThrow(
  () => moduleCompilerPolicy.requireSequentialCompilerOptions({ LinearModule: PolicyLinearModule }),
  "Sequential compiler helpers require LinearModule and ActivationModule constructors",
  "module compiler policy rejects missing constructors",
);
expectThrow(
  () => moduleCompilerPolicy.normalizeTraceShape([1, 0], "trace"),
  "trace dimensions must be positive integers",
  "module compiler policy rejects invalid trace shape",
);
expectThrow(
  () => moduleCompilerPolicy.inferLinearShape(policyLinear, [5]),
  "linear trace input shape must be [2] or [batch, 2], got [5]",
  "module compiler policy rejects bad linear shape",
);
expectThrow(
  () => moduleCompilerPolicy.inferConv2dShape({ inChannels: 2, outChannels: 1, kernelSize: 3 }, [1, 3, 3]),
  "conv2d trace input channels must be 2, got 1",
  "module compiler policy rejects bad conv2d channels",
);
expectThrow(
  () => moduleCompilerPolicy.inferSoftmaxShape({ dim: 2 }, [2, 3]),
  "softmax trace dim 2 is out of range for rank 2",
  "module compiler policy rejects bad softmax dim",
);
expectThrow(
  () => moduleCompilerPolicy.inferFeatureNormShape({ kind: "layerNorm", features: 4 }, [2, 3]),
  "layerNorm trace input shape [2,3] must have scalar count divisible by features 4",
  "module compiler policy rejects bad feature norm shape",
);
expectThrow(
  () => moduleCompilerPolicy.inferShapeModuleShape({ kind: "slice", dim: 0, start: 0, end: 2, step: 0 }, [3]),
  "slice trace step must be a positive safe integer, got 0",
  "module compiler policy rejects bad slice step",
);
const moduleFacadeTraceCalls = [];
const moduleFacadeHelpers = moduleFacadePolicy.createModuleFacadeHelpers({
  traceSequentialProgram(layers, options) {
    moduleFacadeTraceCalls.push([layers.map((layer) => layer.name), options]);
    return { kind: "sequential-trace", layers: layers.length, options };
  },
});
const moduleFacadeModule = {
  name: "facade-module",
  forward(input) {
    return input;
  },
  compileSupport(options) {
    return {
      supported: options.mode !== "off",
      reason: options.mode === "off" ? "facade module disabled" : null,
      compilerSignatures: { ir: "ir:sig", kernelPlan: "kp:sig", memoryLayout: "mem:sig", parameterLayout: "param:sig", bufferLayout: "buf:sig" },
      trace: { inputShape: [1, 2], outputShape: [2, 1] },
      ir: { kind: "ir" },
      kernelPlan: {
        shapeConstraints: { inputShape: [3], outputShape: [4] },
        memoryLayout: { bytes: 16 },
        parameterLayout: { params: 1 },
        bufferLayout: { slots: [] },
      },
    };
  },
  compile(options) {
    return { kind: "compiled", options };
  },
  bindParameters(options) {
    return { weights: Float32Array.of(1), options };
  },
};
expectSame(Object.isFrozen(moduleFacadeHelpers), true, "module facade helpers frozen");
expectSame(moduleFacadeHelpers.traceModule(moduleFacadeModule, { label: "trace" }), { kind: "sequential-trace", layers: 1, options: { label: "trace" } }, "module facade trace fallback");
expectSame(moduleFacadeTraceCalls, [[["facade-module"], { label: "trace" }]], "module facade trace passes module list");
expectSame(moduleFacadeHelpers.compileSupportForModule(moduleFacadeModule, { mode: "on" }).supported, true, "module facade compileSupport delegates");
expectSame(moduleFacadeHelpers.requireCompileSupportForModule(moduleFacadeModule, { mode: "on" }).supported, true, "module facade requireCompileSupport accepts supported evidence");
const moduleFacadeRequiredPlan = moduleFacadeHelpers.requireCompilePlanForModule(moduleFacadeModule, { mode: "on" });
expectSame(moduleFacadeRequiredPlan.kind, "zgml.nn.compile-explanation", "module facade requireCompilePlan returns explanation");
expectSame(moduleFacadeRequiredPlan.supported, true, "module facade requireCompilePlan accepts supported evidence");
expectThrow(
  () => moduleFacadeHelpers.requireCompileSupportForModule(moduleFacadeModule, { mode: "off" }),
  "nn.requireCompileSupport rejected unsupported module: facade module disabled",
  "module facade requireCompileSupport rejects unsupported evidence",
);
expectThrow(
  () => moduleFacadeHelpers.requireCompilePlanForModule(moduleFacadeModule, { mode: "off" }),
  "nn.requireCompilePlan rejected unsupported module: facade module disabled",
  "module facade requireCompilePlan rejects unsupported explanation",
);
const requiredCompileSupport = compileNamespace.requireCompileSupport(moduleFacadeModule, { mode: "on" });
expectSame(requiredCompileSupport.supported, true, "compile namespace requireCompileSupport returns supported evidence");
expectSame(compileNamespace.require_compile_support(moduleFacadeModule, { mode: "on" }), requiredCompileSupport, "compile namespace require_compile_support alias returns evidence");
expectSame(compileNamespace.assertCompileSupport(moduleFacadeModule, { mode: "on" }).supported, true, "compile namespace assertCompileSupport alias returns evidence");
const requiredCompilePlan = compileNamespace.requireCompilePlan(moduleFacadeModule, { mode: "on" });
expectSame(requiredCompilePlan.supported, true, "compile namespace requireCompilePlan returns supported evidence");
expectSame(compileNamespace.require_compile_plan(moduleFacadeModule, { mode: "on" }).signature, requiredCompilePlan.signature, "compile namespace require_compile_plan alias returns evidence");
expectSame(compileNamespace.assertCompilePlan(moduleFacadeModule, { mode: "on" }).signature, requiredCompilePlan.signature, "compile namespace assertCompilePlan alias returns evidence");
expectSame(compileNamespace.assert_compile_plan(moduleFacadeModule, { mode: "on" }).signature, requiredCompilePlan.signature, "compile namespace assert_compile_plan alias returns evidence");
const rawLayerListCompileDiagnostic = compileNamespace.compile([moduleFacadeModule], { inputShape: [3] });
expectSame(Object.isFrozen(rawLayerListCompileDiagnostic), true, "compile namespace returns frozen raw Sequential layer list diagnostic");
expectSame(rawLayerListCompileDiagnostic, {
  kind: "zgml.compile.raw-sequential-layer-list",
  supported: false,
  reason: "raw Sequential layer lists support compile evidence only",
  evidencePath: "compile.trace|compile.compileSupport|compile.requireCompileSupport|compile.explain|compile.requireCompilePlan",
  programPath: "nn.compile(layers)",
}, "compile namespace returns diagnostic for raw Sequential layer list Program creation");
expectSame(executionPlan.acceptsModuleCompilePlan(moduleFacadeRequiredPlan), true, "compile plan helper accepts supported explanation");
expectSame(compileNamespace.acceptsModuleCompilePlan(moduleFacadeRequiredPlan), true, "compile namespace accepts compile plan evidence");
expectSame(compileNamespace.requireModuleCompilePlan(moduleFacadeRequiredPlan), moduleFacadeRequiredPlan, "compile namespace requires compile plan evidence");
expectSame(compileNamespace.assertModuleCompilePlan(moduleFacadeRequiredPlan), moduleFacadeRequiredPlan, "compile namespace assert compile plan alias");
expectSame(compileNamespace.assert_module_compile_plan(moduleFacadeRequiredPlan), moduleFacadeRequiredPlan, "compile namespace assert compile plan snake alias");
expectSame(compileNamespace.matchesModuleCompilePlanSignature(moduleFacadeRequiredPlan, moduleFacadeRequiredPlan.signature), true, "compile namespace matches compile plan signature");
expectSame(nnModule.acceptsModuleCompilePlan(moduleFacadeRequiredPlan), true, "nn module accepts compile plan evidence");
expectSame(nnModule.requireModuleCompilePlan(moduleFacadeRequiredPlan), moduleFacadeRequiredPlan, "nn module requires compile plan evidence");
expectSame(inspectionNamespace.acceptsModuleCompilePlan(moduleFacadeRequiredPlan), true, "inspection namespace accepts compile plan evidence");
expectSame(inspectionNamespace.requireModuleCompilePlan(moduleFacadeRequiredPlan), moduleFacadeRequiredPlan, "inspection namespace requires compile plan evidence");
expectThrow(
  () => compileNamespace.requireCompileSupport({ compileSupport: () => ({ supported: false, reason: "unsupported smoke module" }) }),
  "compile.requireCompileSupport rejected unsupported target: unsupported smoke module",
  "compile namespace requireCompileSupport rejects unsupported evidence",
);
expectThrow(
  () => compileNamespace.requireCompilePlan({ compileSupport: () => ({ supported: false, reason: "unsupported smoke module" }) }),
  "compile.requireCompilePlan rejected unsupported target: unsupported smoke module",
  "compile namespace requireCompilePlan rejects unsupported evidence",
);
expectThrow(
  () => executionPlan.requireModuleCompilePlan(compileNamespace.explain(moduleFacadeModule, { mode: "off" })),
  "ModuleCompilePlan is not compilable: facade module disabled",
  "compile plan helper rejects unsupported explanation",
);
expectSame(moduleFacadeHelpers.compilerSignaturesForModule(moduleFacadeModule, {}).ir, "ir:sig", "module facade compiler signatures");
expectSame(moduleFacadeHelpers.tensorProgramIrForModule(moduleFacadeModule, {}), { kind: "ir" }, "module facade tensor program ir");
expectSame(moduleFacadeHelpers.kernelPlanForModule(moduleFacadeModule, {}).shapeConstraints.inputShape, [3], "module facade kernel plan");
expectSame(moduleFacadeHelpers.bufferLayoutForModule(moduleFacadeModule, {}), { slots: [] }, "module facade buffer layout");
expectSame(moduleFacadeHelpers.memoryLayoutForModule(moduleFacadeModule, {}), { bytes: 16 }, "module facade memory layout");
expectSame(moduleFacadeHelpers.inputShapeForModule(moduleFacadeModule, {}), [3], "module facade input shape prefers kernel plan");
expectSame(moduleFacadeHelpers.outputShapeForModule(moduleFacadeModule, {}), [4], "module facade output shape prefers kernel plan");
expectSame(moduleFacadeHelpers.shapeConstraintsForModule(moduleFacadeModule, {}), { inputShape: [3], outputShape: [4] }, "module facade shape constraints");
expectSame(moduleFacadeHelpers.parameterLayoutForModule(moduleFacadeModule, {}), { params: 1 }, "module facade parameter layout");
expectSame(moduleFacadeHelpers.canCompileModule(moduleFacadeModule, { mode: "on" }), true, "module facade canCompile");
expectSame(moduleFacadeHelpers.compileModule(moduleFacadeModule, { backend: "cpu" }), { kind: "compiled", options: { backend: "cpu" } }, "module facade compile");
expectSame(moduleFacadeHelpers.bindModuleParameters(moduleFacadeModule, { tag: "bind" }).weights, [1], "module facade bind parameters");
const moduleFacadeBindings = moduleFacadeHelpers.bindModuleParameters(moduleFacadeModule, { mode: "on" });
moduleBindings.annotateModuleBindings(moduleFacadeBindings, moduleFacadeModule, { mode: "on" });
const moduleFacadeBindingPlan = moduleFacadeHelpers.bindingPlanForModuleBindings(moduleFacadeBindings);
expectSame(moduleFacadeHelpers.requireBindingPlanForModuleBindings(moduleFacadeBindings).moduleBindings, true, "module facade require binding plan accepts ModuleBindings");
expectSame(executionPlan.acceptsModuleBindingPlan(moduleFacadeBindingPlan), true, "binding plan helper accepts ModuleBindings plan");
expectSame(executionPlan.requireModuleBindingPlan(moduleFacadeBindingPlan), moduleFacadeBindingPlan, "binding plan helper requires ModuleBindings plan");
expectSame(executionPlan.assertModuleBindingPlan(moduleFacadeBindingPlan), moduleFacadeBindingPlan, "binding plan helper assert ModuleBindings alias");
expectSame(executionPlan.assert_module_binding_plan(moduleFacadeBindingPlan), moduleFacadeBindingPlan, "binding plan helper assert ModuleBindings snake alias");
expectSame(executionPlan.matchesModuleBindingPlanSignature(moduleFacadeBindingPlan, moduleFacadeBindingPlan.signature), true, "binding plan helper matches ModuleBindings signature");
expectThrow(
  () => moduleFacadeHelpers.requireBindingPlanForModuleBindings({ weights: Float32Array.of(1) }),
  "nn.requireBindingPlan rejected bindings: bindings are not ModuleBindings from nn.bindParameters(...) or nn.placeParameters(...)",
  "module facade require binding plan rejects raw bindings",
);
expectThrow(
  () => executionPlan.requireModuleBindingPlan(moduleFacadeHelpers.bindingPlanForModuleBindings({ weights: Float32Array.of(1) })),
  "ModuleBindingPlan is not bindable: bindings are not ModuleBindings from nn.bindParameters(...) or nn.placeParameters(...)",
  "binding plan helper rejects raw ModuleBindings plan",
);
expectSame(moduleFacadeHelpers.placeModuleParameters(placedModule, placedProgram, { inputShape: [2] }).weights.name, "weights", "module facade place parameters");
expectThrow(
  () => moduleFacadePolicy.createModuleFacadeHelpers({}),
  "module facade helpers require traceSequentialProgram",
  "module facade rejects missing trace callback",
);
expectThrow(
  () => moduleFacadeHelpers.traceModule({}),
  "nn.trace requires an nn module",
  "module facade rejects trace without module",
);
expectThrow(
  () => moduleFacadeHelpers.compileSupportForModule({}),
  "nn.compileSupport requires an nn module",
  "module facade rejects compileSupport without module",
);
expectThrow(
  () => moduleFacadeHelpers.compileModule({}),
  "nn.compile requires a module with native Program support",
  "module facade rejects compile without module",
);
expectThrow(
  () => moduleFacadeHelpers.bindModuleParameters({}),
  "nn.bindParameters requires a module with native Program support",
  "module facade rejects bind without module",
);
expectThrow(
  () => programModuleBinding.createProgramModuleBindingHelpers({}),
  "createProgramModuleBindingHelpers requires uniqueNativeBuffers",
  "program module binding rejects missing unique buffers",
);
expectThrow(
  () => moduleBindHelpers.bindModuleThroughProgram({}, moduleBindModule),
  "Program.bindModule requires a Program with bind and moduleCompatibility",
  "program module binding rejects bad program",
);
expectThrow(
  () => moduleBindHelpers.bindModuleThroughProgram(moduleBindProgram, {}),
  "Program.bindModule requires an nn module with placeParameters(program, options)",
  "program module binding rejects bad module",
);
expectThrow(
  () => moduleBindHelpers.bindModuleThroughProgram(moduleBindProgram, moduleBindModule, { inputShape: [3] }),
  "module is incompatible with this Program",
  "program module binding rejects incompatible module fallback reason",
);
const rejectedProgram = {
  bind: moduleBindProgram.bind,
  moduleCompatibility: () => ({ compatible: false, reason: "shape mismatch" }),
};
expectThrow(
  () => moduleBindHelpers.bindModuleThroughProgram(rejectedProgram, moduleBindModule, {}),
  "shape mismatch",
  "program module binding rejects incompatible module reason",
);
const throwingProgram = {
  moduleCompatibility: () => ({ compatible: true, reason: null }),
  bind() {
    throw new Error("bind failed");
  },
};
expectThrow(
  () => moduleBindHelpers.bindModuleThroughProgram(throwingProgram, moduleBindModule, {}),
  "bind failed",
  "program module binding rethrows bind failure",
);
expectSame(moduleBindFreed, ["weights", "bias"], "program module binding frees owned buffers on bind failure");

const programDeviceCalls = [];
const nativeDeviceSource = { native: true, id: "native-buffer" };
const ProgramDeviceSmoke = programDevicePolicy.createProgramDeviceClass({
  assertProgramAlive(handle, label) {
    programDeviceCalls.push(["alive", handle, label]);
    if (handle === 0) throw new Error(`${label} is freed`);
  },
  programDeviceHandle(handle, placement) {
    programDeviceCalls.push(["device-handle", handle, placement]);
    return handle * 10 + (placement === "webgpu" ? 1 : 2);
  },
  createDeviceBuffer(handle, kind, placement) {
    programDeviceCalls.push(["create-buffer", handle, kind, placement]);
    return { route: "device-buffer", handle, kind, placement };
  },
  importDeviceBuffer(handle, kind, source, options) {
    programDeviceCalls.push(["import", handle, kind, source.id ?? source.tag, options ?? null]);
    return { route: "import", handle, kind, source, options: options ?? null };
  },
  isNativeBuffer(source) {
    return Boolean(source && source.native);
  },
  createKvCache(handle, createBuffer) {
    programDeviceCalls.push(["kv-cache", handle]);
    return {
      route: "kv-cache",
      k: createBuffer("kv-k"),
      v: createBuffer("kv-v"),
    };
  },
});
const programDevice = new ProgramDeviceSmoke(11);
expectSame({
  programHandle: programDevice.programHandle,
  placement: programDevice.placement,
  handle: programDevice.handle,
}, {
  programHandle: 11,
  placement: "webgpu",
  handle: 111,
}, "program device initializes with default placement");
const programDeviceInfo = programDevice.info();
expectSame(programDeviceInfo, {
  kind: "zgml.program-device.info",
  signature: "program-device-info|placement=webgpu|handle=111",
  placement: "webgpu",
  handle: 111,
}, "program device info evidence");
expectSame(Object.isFrozen(programDeviceInfo), true, "program device info is frozen");
expectSame(programDevicePolicy.isProgramDeviceInfo(programDeviceInfo), true, "program device info predicate");
expectSame(programDevicePolicy.requireProgramDeviceInfo(programDeviceInfo).handle, 111, "program device info require");
expectSame(programDevicePolicy.assertProgramDeviceInfo(programDeviceInfo).placement, "webgpu", "program device info assert");
expectSame(programDevicePolicy.assert_program_device_info(programDeviceInfo).signature, programDeviceInfo.signature, "program device info snake assert");
expectSame(programDevice.matchesInfoSignature(programDeviceInfo.signature), true, "program device class info signature");
expectSame(programDevicePolicy.matchesProgramDeviceInfoSignature(programDeviceInfo, programDeviceInfo.signature), true, "program device info signature");
expectSame(programDevicePolicy.matches_program_device_info_signature(programDeviceInfo, "wrong"), false, "program device info snake signature mismatch");
expectSame(programDevicePolicy.isProgramDeviceInfo({ ...programDeviceInfo }), false, "program device info rejects mutable record");
expectThrow(
  () => programDevicePolicy.requireProgramDeviceInfo({
    ...programDeviceInfo,
    signature: "program-device-info|placement=webgpu|handle=112",
  }),
  "expected frozen ProgramDevice info evidence",
  "program device info rejects mismatched signature",
);
expectSame(programDevice.createOutputBuffer(), { route: "device-buffer", handle: 11, kind: "output", placement: "webgpu" }, "program device output buffer");
expectSame(programDevice.createWeightsBuffer(), { route: "device-buffer", handle: 11, kind: "weights", placement: "webgpu" }, "program device weights buffer");
expectSame(programDevice.createBiasBuffer(), { route: "device-buffer", handle: 11, kind: "bias", placement: "webgpu" }, "program device bias buffer");
expectSame(programDevice.createInputBuffer(), { route: "device-buffer", handle: 11, kind: "input", placement: "webgpu" }, "program device input buffer");
expectSame(programDevice.createBuffer("kv-k"), { route: "device-buffer", handle: 11, kind: "kv-k", placement: "webgpu" }, "program device generic buffer");
expectSame(programDevice.createKvCache(), {
  route: "kv-cache",
  k: { route: "device-buffer", handle: 11, kind: "kv-k", placement: "webgpu" },
  v: { route: "device-buffer", handle: 11, kind: "kv-v", placement: "webgpu" },
}, "program device kv cache");
expectSame(programDevice.importBuffer("input", nativeDeviceSource), {
  route: "import",
  handle: 11,
  kind: "input",
  source: nativeDeviceSource,
  options: null,
}, "program device imports native source directly");
expectSame(programDevice.importBuffer("output", { tag: "descriptor" }), {
  route: "import",
  handle: 11,
  kind: "output",
  source: { tag: "descriptor" },
  options: { placement: "webgpu", deviceHandle: 111 },
}, "program device imports descriptor with same-device defaults");
const metalDevice = new ProgramDeviceSmoke(12, "metal");
expectSame(metalDevice.createBuffer("input"), { route: "device-buffer", handle: 12, kind: "input", placement: "metal" }, "program device custom placement");
expectThrow(
  () => programDevicePolicy.createProgramDeviceClass({}),
  "createProgramDeviceClass requires assertProgramAlive",
  "program device rejects missing assert",
);
expectThrow(
  () => new ProgramDeviceSmoke(0),
  "program is freed",
  "program device rejects dead program",
);
expectThrow(
  () => new programDevicePolicy.createProgramDeviceClass({
    assertProgramAlive: () => {},
    programDeviceHandle: () => 1,
    createDeviceBuffer: () => ({}),
    importDeviceBuffer: () => ({}),
    isNativeBuffer: () => false,
  })(1),
  "createProgramDeviceClass requires createKvCache",
  "program device rejects missing kv cache hook",
);

const compatibleEvidence = {
  kind: "module",
  nativePath: "module",
  modelKind: "module",
  layerCount: 1,
  inputLen: 2,
  outputLen: 3,
  weightsLen: 6,
  biasLen: 3,
  inputShape: [2],
  outputShape: [3],
  compilerSignatures: {
    ir: "ir-a",
    kernelPlan: "kernel-a",
    memoryLayout: "memory-a",
    parameterLayout: "",
    bufferLayout: "buffer-a",
  },
  kernelPlan: { opCount: 1, dispatchCount: 1, descriptorCount: 1, elidedOpCount: 0, inputLen: 2, outputLen: 3, weightsLen: 6, biasLen: 3, inputShape: [2], outputShape: [3] },
};
const compatibleSupport = {
  supported: true,
  nativePath: "module",
  modelKind: "module",
  layerCount: 1,
  inputLen: 2,
  outputLen: 3,
  weightsLen: 6,
  biasLen: 3,
  inputShape: [2],
  outputShape: [3],
  compilerSignatures: compatibleEvidence.compilerSignatures,
  kernelPlan: { opCount: 1, dispatchCount: 1, descriptorCount: 1, elidedOpCount: 0, inputLen: 2, outputLen: 3, weightsLen: 6, biasLen: 3, inputShape: [2], outputShape: [3] },
};
expectSame(moduleCompatibility.moduleCompatibilityForSupportEvidence(compatibleEvidence, compatibleSupport), {
  kind: "zgml.program.module-compatibility",
  signature: "program-module-compatibility|compatible=1|program=module|module=module|diagnostics=0|codes=",
  compatible: true,
  reason: null,
  programKind: "module",
  moduleKind: "module",
  diagnostics: [],
}, "module compatibility accepted");
expectSame(moduleCompatibility.moduleOptionsWithEvidenceShape({ trace: { inputShape: [2, 3] } }, { tag: "x" }), {
  tag: "x",
  inputShape: [2, 3],
}, "module compatibility injects input shape");
expectSame(moduleCompatibility.moduleOptionsWithEvidenceShape({ kind: "tiny-linear", nativePath: "tiny-linear", trace: { inputShape: [2] } }, { tag: "x", inputShape: [9] }), {
  tag: "x",
}, "module compatibility preserves tiny-linear support lane");
expectSame(moduleCompatibility.moduleCompatibilityForProgramEvidence(compatibleEvidence, null).diagnostics[0], {
  code: "module-missing-compiler",
  message: "module does not expose compileSupport for compatibility preflight",
}, "module compatibility missing compiler diagnostic");
const mismatchSupport = {
  ...compatibleSupport,
  inputShape: [1, 2],
  compilerSignatures: { ...compatibleSupport.compilerSignatures, ir: "ir-b" },
};
const mismatch = moduleCompatibility.moduleCompatibilityForSupportEvidence(compatibleEvidence, mismatchSupport);
expectSame(mismatch.compatible, false, "module compatibility mismatch rejected");
expectSame(mismatch.signature.startsWith("program-module-compatibility|compatible=0|program=module|module=module|"), true, "module compatibility rejected signature");
expectSame(mismatch.diagnostics.some((diagnostic) => diagnostic.code === "input-shape-mismatch"), true, "module compatibility shape diagnostic");
expectSame(mismatch.diagnostics.some((diagnostic) => diagnostic.code === "ir-mismatch" && diagnostic.signatureKind === "ir"), true, "module compatibility signature diagnostic");

const policyAliveChecks = [];
const policyFreedHandles = [];
const policyResetHandles = [];
const policyRequirements = { scalarBytes: 4, inputLen: 2, outputLen: 3, weightsLen: 6, biasLen: 3 };
const policyCapabilities = Object.freeze({
  signature: "program-capabilities|backend=cpu|mode=executable",
  mode: "executable",
  canExecute: true,
  canBindExternalResources: true,
  hasFullDispatchPlan: true,
});
const policyRuntimeProfile = Object.freeze({
  signature: "runtime-profile|callCount=1",
  callCount: 1,
});
const policy = programFacadePolicy.createProgramFacadePolicyHelpers({
  assertProgramAlive(handle, label) {
    policyAliveChecks.push({ handle, label });
    if (handle === 0) throw new Error(`${label} is freed`);
  },
  programRequirements: () => policyRequirements,
  programModelCompatibility: (_handle, model) => Object.freeze({
    compatible: model && model.kind === "module",
    reason: model && model.kind === "module" ? null : "model mismatch",
  }),
  programExecutionCapabilities: () => policyCapabilities,
  programRuntimeProfile: () => policyRuntimeProfile,
  programResetRuntimeProfile(handle) {
    policyResetHandles.push(handle);
    return { reset: handle };
  },
  programFree(handle) {
    policyFreedHandles.push(handle);
  },
  nullProgramHandle: 0,
});
expectSame(Object.isFrozen(policy), true, "program facade policy frozen");
expectSame(policy.requirements(7), policyRequirements, "program facade policy requirements");
expectSame(policy.bufferLayout(7).weights.byteLength, 24, "program facade policy buffer layout");
expectSame(policy.trace(7, { trace: { ops: [{ op: "linear" }] } }), { ops: [{ op: "linear" }] }, "program facade policy trace evidence");
expectSame(policy.compilerSignatures(7, compatibleEvidence), compatibleEvidence.compilerSignatures, "program facade policy compiler signatures");
expectSame(policy.tensorProgramIr(7, { ir: { kind: "ir" } }), { kind: "ir" }, "program facade policy tensor ir");
expectSame(policy.kernelPlan(7, compatibleEvidence), compatibleEvidence.kernelPlan, "program facade policy kernel plan");
expectSame(policy.shapeConstraints(7, { kernelPlan: { shapeConstraints: { inputShape: [2] } } }), { inputShape: [2] }, "program facade policy shape constraints");
expectSame(policy.memoryLayout(7, { kernelPlan: { memoryLayout: { bytes: 44 } } }), { bytes: 44 }, "program facade policy memory layout");
expectSame(policy.parameterLayout(7, { kernelPlan: { parameterLayout: { weights: [0, 24] } } }), { weights: [0, 24] }, "program facade policy parameter layout");
expectSame(policy.modelCompatibility(7, { kind: "module" }).compatible, true, "program facade policy model compatibility");
expectSame(policy.acceptsModel(7, { kind: "module" }), true, "program facade policy accepts model");
expectSame(policy.acceptsModel(7, { kind: "llama" }), false, "program facade policy rejects model");
const compatibleModule = {
  compileSupport: () => compatibleSupport,
};
expectSame(policy.moduleCompatibility(7, compatibleEvidence, compatibleModule).compatible, true, "program facade policy module compatibility");
expectSame(policy.acceptsModule(7, compatibleEvidence, compatibleModule), true, "program facade policy accepts module");
expectSame(policy.capabilities(7), policyCapabilities, "program facade policy capabilities");
expectSame(policy.canExecute(7), true, "program facade policy can execute");
expectSame(policy.canBindExternalResources(7), true, "program facade policy can bind resources");
expectSame(policy.hasFullDispatchPlan(7), true, "program facade policy full dispatch");
expectSame(policy.executionMode(7), "executable", "program facade policy execution mode");
expectSame(policy.matchesCapabilitySignature(7, policyCapabilities.signature), true, "program facade policy capability signature");
expectSame(policy.matchesCapabilitySignature(7, "wrong"), false, "program facade policy capability signature mismatch");
expectSame(policy.runtimeProfile(7), policyRuntimeProfile, "program facade policy runtime profile");
expectSame(policy.matchesRuntimeProfileSignature(7, policyRuntimeProfile.signature), true, "program facade policy runtime signature");
expectSame(policy.matchesRuntimeProfileSignature(7, null), false, "program facade policy runtime signature invalid");
expectSame(policy.resetRuntimeProfile(7), { reset: 7 }, "program facade policy reset runtime profile");
const policyProgram = { handle: 7 };
policy.free(policyProgram);
policy.dispose(policyProgram);
expectSame({ program: policyProgram, freed: policyFreedHandles, reset: policyResetHandles }, {
  program: { handle: 0 },
  freed: [7],
  reset: [7],
}, "program facade policy lifecycle");
expectSame(policyAliveChecks.some((entry) => entry.handle === 7 && entry.label === "program"), true, "program facade policy alive checks");
expectThrow(
  () => programFacadePolicy.createProgramFacadePolicyHelpers({}),
  "createProgramFacadePolicyHelpers requires assertProgramAlive",
  "program facade policy rejects missing assert",
);
expectThrow(
  () => policy.requirements(0),
  "program is freed",
  "program facade policy alive rejection",
);

const sizingRequirements = {
  modelKind: "module",
  scalarBytes: 2,
  inputLen: 3,
  outputLen: 4,
  weightsLen: 5,
  biasLen: 2,
};
const sizingSnapshot = programSizing.programBufferSizingFromRequirements(sizingRequirements);
expectSame(sizingSnapshot, {
  kind: "program-buffer-sizing",
  scalarType: "f32",
  scalarBytes: 2,
  inputLen: 3,
  inputByteLength: 6,
  outputLen: 4,
  outputByteLength: 8,
  weightsLen: 5,
  weightsByteLength: 10,
  biasLen: 2,
  biasByteLength: 4,
  parameterLen: 7,
  parameterByteLength: 14,
  modelKind: "module",
  signature: "program-buffer-sizing|model=module|scalar=f32|scalarBytes=2|input=3|inputBytes=6|output=4|outputBytes=8|weights=5|weightsBytes=10|bias=2|biasBytes=4|parameters=7|parameterBytes=14",
}, "program sizing fallback snapshot");
expectSame(Object.isFrozen(sizingSnapshot), true, "program sizing snapshot frozen");
expectSame(programSizing.programMatchesBufferSizingSignature(sizingRequirements, sizingSnapshot.signature), true, "program sizing signature match");
expectSame(programSizing.programMatchesBufferSizingSignature(sizingRequirements, ""), false, "program sizing rejects empty signature");
const explicitSizingRequirements = {
  modelKind: "tiny-llama",
  scalarBytes: 4,
  inputLen: 10,
  inputByteLength: 44,
  outputLen: 11,
  outputByteLength: 48,
  weightsLen: 12,
  weightsByteLength: 52,
  biasLen: 13,
  biasByteLength: 56,
  parameterLen: 14,
  parameterByteLength: 60,
};
expectSame(programSizing.programInputByteLength(explicitSizingRequirements), 44, "program sizing explicit input bytes");
expectSame(programSizing.programParameterLen(explicitSizingRequirements), 14, "program sizing explicit parameter len");
expectSame(programSizing.programBufferSizingFromRequirements(explicitSizingRequirements).signature, "program-buffer-sizing|model=tiny-llama|scalar=f32|scalarBytes=4|input=10|inputBytes=44|output=11|outputBytes=48|weights=12|weightsBytes=52|bias=13|biasBytes=56|parameters=14|parameterBytes=60", "program sizing explicit signature");
const sizingAccessorCalls = [];
const sizingAccessors = programSizing.createProgramSizingAccessors({
  requirements(program) {
    sizingAccessorCalls.push(program.handle);
    return sizingRequirements;
  },
});
expectSame(Object.isFrozen(sizingAccessors), true, "program sizing accessors frozen");
expectSame(sizingAccessors.inputLen({ handle: 1 }), 3, "program sizing access input len");
expectSame(sizingAccessors.outputLen({ handle: 2 }), 4, "program sizing access output len");
expectSame(sizingAccessors.inputByteLength({ handle: 3 }), 6, "program sizing access input bytes");
expectSame(sizingAccessors.outputByteLength({ handle: 4 }), 8, "program sizing access output bytes");
expectSame(sizingAccessors.weightsLen({ handle: 5 }), 5, "program sizing access weights len");
expectSame(sizingAccessors.weightsByteLength({ handle: 6 }), 10, "program sizing access weights bytes");
expectSame(sizingAccessors.biasLen({ handle: 7 }), 2, "program sizing access bias len");
expectSame(sizingAccessors.biasByteLength({ handle: 8 }), 4, "program sizing access bias bytes");
expectSame(sizingAccessors.parameterLen({ handle: 9 }), 7, "program sizing access parameter len");
expectSame(sizingAccessors.parameterByteLength({ handle: 10 }), 14, "program sizing access parameter bytes");
expectSame(sizingAccessors.bufferSizing({ handle: 11 }).signature, sizingSnapshot.signature, "program sizing access buffer sizing");
expectSame(sizingAccessors.matchesBufferSizingSignature({ handle: 12 }, sizingSnapshot.signature), true, "program sizing access signature match");
expectSame(sizingAccessorCalls, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "program sizing access requirements calls");
expectThrow(
  () => programSizing.createProgramSizingAccessors({}),
  "createProgramSizingAccessors requires requirements",
  "program sizing access rejects missing requirements",
);

const parameterLayoutSmoke = {
  parameters: [
    { name: "weight", offset: 0, length: 4 },
    { name: "bias", offset: 4, length: 2 },
  ],
};
const parameterNamesSmoke = programParameters.programParameterNames(parameterLayoutSmoke);
const parameterInfosSmoke = programParameters.programParameterInfos(parameterLayoutSmoke);
expectSame(parameterNamesSmoke, ["weight", "bias"], "program parameters names");
expectSame(Object.isFrozen(parameterNamesSmoke), true, "program parameters names frozen");
expectSame(parameterInfosSmoke, parameterLayoutSmoke.parameters, "program parameters infos");
expectSame(parameterInfosSmoke === parameterLayoutSmoke.parameters, false, "program parameters infos copies array");
expectSame(Object.isFrozen(parameterInfosSmoke), true, "program parameters infos frozen");
expectSame(programParameters.programParameterInfo(parameterLayoutSmoke, 1), parameterLayoutSmoke.parameters[1], "program parameters info by index");
expectSame(programParameters.programParameterInfo(parameterLayoutSmoke, "weight"), parameterLayoutSmoke.parameters[0], "program parameters info by name");
expectSame(programParameters.programParameterInfo(parameterLayoutSmoke, "missing"), null, "program parameters missing name");
expectSame(programParameters.programParameterNames({}), [], "program parameters empty names");
expectSame(programParameters.programParameterInfo({}, "weight"), null, "program parameters empty info");
expectThrow(
  () => programParameters.programParameterInfo(parameterLayoutSmoke, -1),
  "Program.parameterInfo index must be a non-negative integer",
  "program parameters rejects negative index",
);
expectThrow(
  () => programParameters.programParameterInfo(parameterLayoutSmoke, true),
  "Program.parameterInfo requires a parameter name or index",
  "program parameters rejects invalid key",
);
const parameterAccessors = programParameters.createProgramParameterAccessors({
  parameterLayout(program) {
    return program.layout;
  },
});
expectSame(Object.isFrozen(parameterAccessors), true, "program parameter accessors frozen");
expectSame(parameterAccessors.parameterNames({ layout: parameterLayoutSmoke }), ["weight", "bias"], "program parameter access names");
expectSame(parameterAccessors.parameterInfos({ layout: parameterLayoutSmoke }), parameterLayoutSmoke.parameters, "program parameter access infos");
expectSame(parameterAccessors.parameterInfo({ layout: parameterLayoutSmoke }, "bias"), parameterLayoutSmoke.parameters[1], "program parameter access info");
expectThrow(
  () => programParameters.createProgramParameterAccessors({}),
  "createProgramParameterAccessors requires parameterLayout",
  "program parameter access rejects missing layout",
);

const fallbackShapeCalls = [];
const fallbackProgramShape = (desc) => {
  fallbackShapeCalls.push(desc);
  return Object.freeze([desc.fallbackLen]);
};
const evidenceInputShape = programShapes.programInputShapeFromEvidence(
  { fallbackLen: 9 },
  { inputShape: [2, 3], outputShape: [6] },
  fallbackProgramShape,
);
expectSame(evidenceInputShape, [2, 3], "program shapes evidence input");
expectSame(Object.isFrozen(evidenceInputShape), true, "program shapes evidence input frozen");
expectSame(fallbackShapeCalls, [], "program shapes skips fallback with evidence");
const fallbackOutputShape = programShapes.programOutputShapeFromEvidence(
  { fallbackLen: 5 },
  {},
  fallbackProgramShape,
);
expectSame(fallbackOutputShape, [5], "program shapes fallback output");
expectSame(fallbackShapeCalls, [{ fallbackLen: 5 }], "program shapes fallback called");
expectSame(programShapes.llamaProgramOutputShapeFromInspection({ vocabSize: 32000 }), [32000], "program shapes llama output");
expectSame(Object.isFrozen(programShapes.llamaProgramOutputShapeFromInspection({ vocabSize: 7 })), true, "program shapes llama output frozen");
const shapeAccessCalls = [];
const genericShapeAccessors = programShapes.createGenericProgramShapeAccessors({
  assertProgramAlive(handle, label) {
    shapeAccessCalls.push(["alive", handle, label]);
    if (handle === 0) throw new Error(`${label} is freed`);
  },
  shapeConstraints(program) {
    shapeAccessCalls.push(["constraints", program.handle]);
    return program.constraints;
  },
  inputShapeFallback(desc) {
    shapeAccessCalls.push(["inputFallback", desc.id]);
    return Object.freeze([desc.inputLen]);
  },
  outputShapeFallback(desc) {
    shapeAccessCalls.push(["outputFallback", desc.id]);
    return Object.freeze([desc.outputLen]);
  },
});
expectSame(Object.isFrozen(genericShapeAccessors), true, "program shape accessors frozen");
expectSame(genericShapeAccessors.inputShape({ handle: 31, desc: { id: "a", inputLen: 9 }, constraints: { inputShape: [2, 2] } }), [2, 2], "program shape access input evidence");
expectSame(genericShapeAccessors.outputShape({ handle: 32, desc: { id: "b", outputLen: 5 }, constraints: {} }), [5], "program shape access output fallback");
expectSame(shapeAccessCalls, [
  ["alive", 31, "program"],
  ["constraints", 31],
  ["alive", 32, "program"],
  ["constraints", 32],
  ["outputFallback", "b"],
], "program shape access calls");
expectThrow(
  () => genericShapeAccessors.inputShape({ handle: 0, desc: {}, constraints: {} }),
  "program is freed",
  "program shape access liveness",
);
const llamaShapeAccessors = programShapes.createLlamaProgramShapeAccessors({
  inspect(program) {
    return program.inspection;
  },
});
expectSame(llamaShapeAccessors.outputShape({ inspection: { vocabSize: 99 } }), [99], "program shape access llama output");
expectThrow(
  () => programShapes.createGenericProgramShapeAccessors({}),
  "createGenericProgramShapeAccessors requires assertProgramAlive",
  "program shape access rejects missing assert",
);
expectThrow(
  () => programShapes.createLlamaProgramShapeAccessors({}),
  "createLlamaProgramShapeAccessors requires inspect",
  "program shape access rejects missing inspect",
);

const resourceCalls = [];
const resourceAccessors = programResources.createProgramResourceAccessors({
  assertProgramAlive(handle, label) {
    resourceCalls.push(["alive", handle, label]);
    if (handle === 0) throw new Error(`${label} is freed`);
  },
  createProgramNamedBuffer(handle, kind, options) {
    resourceCalls.push(["named", handle, kind, options]);
    return { handle, kind, options };
  },
  createProgramOutputBuffer(handle, options) {
    resourceCalls.push(["output", handle, options]);
    return { handle, kind: "output", options };
  },
  createProgramBuffer(handle, kind, options) {
    resourceCalls.push(["role", handle, kind, options]);
    return { handle, kind, options };
  },
  programDeviceHandle(handle, placement) {
    resourceCalls.push(["device-handle", handle, placement]);
    return { handle, placement };
  },
  createProgramDevice(handle, placement) {
    resourceCalls.push(["device", handle, placement]);
    return { handle, placement };
  },
  importDeviceBuffer(handle, kind, source) {
    resourceCalls.push(["import", handle, kind, source]);
    return { handle, kind, source };
  },
});
const resourceProgram = { handle: 12 };
expectSame(Object.isFrozen(resourceAccessors), true, "program resources accessors frozen");
expectSame(resourceAccessors.createBuffer(resourceProgram, "weights", { resource: true }), { handle: 12, kind: "weights", options: { resource: true } }, "program resources createBuffer");
expectSame(resourceAccessors.createOutputBuffer(resourceProgram, { placement: "cpu" }), { handle: 12, kind: "output", options: { placement: "cpu" } }, "program resources output buffer");
expectSame(resourceAccessors.createWeightsBuffer(resourceProgram, { tag: "w" }), { handle: 12, kind: "weights", options: { tag: "w" } }, "program resources weights buffer");
expectSame(resourceAccessors.createBiasBuffer(resourceProgram, { tag: "b" }), { handle: 12, kind: "bias", options: { tag: "b" } }, "program resources bias buffer");
expectSame(resourceAccessors.createInputBuffer(resourceProgram, { tag: "i" }), { handle: 12, kind: "input", options: { tag: "i" } }, "program resources input buffer");
expectSame(resourceAccessors.deviceHandle(resourceProgram), { handle: 12, placement: "webgpu" }, "program resources default device handle");
expectSame(resourceAccessors.device(resourceProgram, "metal"), { handle: 12, placement: "metal" }, "program resources device");
expectSame(resourceAccessors.importDeviceBuffer(resourceProgram, "output", { native: true }), { handle: 12, kind: "output", source: { native: true } }, "program resources import device buffer");
expectSame(resourceCalls.filter((call) => call[0] === "alive").length, 8, "program resources liveness count");
expectThrow(
  () => resourceAccessors.createBuffer({ handle: 0 }, "input"),
  "program is freed",
  "program resources liveness error",
);
const resourceAccessorsWithoutRoleBuffers = programResources.createProgramResourceAccessors({
  assertProgramAlive() {},
  createProgramNamedBuffer() {},
  createProgramOutputBuffer() {},
  programDeviceHandle() {},
  createProgramDevice() {},
  importDeviceBuffer() {},
});
expectThrow(
  () => resourceAccessorsWithoutRoleBuffers.createWeightsBuffer({ handle: 1 }),
  "createProgramResourceAccessors requires createProgramBuffer",
  "program resources rejects missing role buffer callback",
);
expectThrow(
  () => programResources.createProgramResourceAccessors({}),
  "createProgramResourceAccessors requires assertProgramAlive",
  "program resources rejects missing assert",
);

let layoutPolicyCalls = 0;
const programLayoutAccessors = programLayoutAccess.createProgramLayoutAccessors({
  programPolicy: {
    bufferLayout(handle) {
      layoutPolicyCalls += 1;
      return {
        scalarType: "f32",
        scalarBytes: 4,
        slots: [
          { name: "input", byteLength: 8 },
          { name: "output", byteLength: 12 },
        ],
        input: { name: "input", byteLength: 8 },
        output: { name: "output", byteLength: 12 },
        handle,
      };
    },
  },
  bufferSlotMethodName: "SmokeProgram.bufferSlot",
});
const layoutProgram = { handle: 44 };
expectSame(programLayoutAccessors.bufferLayout(layoutProgram).handle, 44, "program layout access buffer layout");
expectSame(programLayoutAccessors.bufferLayout(layoutProgram).handle, 44, "program layout access cached layout");
expectSame(layoutPolicyCalls, 1, "program layout access cache count");
expectSame(programLayoutAccessors.bufferSlotNames(layoutProgram), ["input", "output"], "program layout access slot names");
expectSame(programLayoutAccessors.bufferSlot(layoutProgram, "output").byteLength, 12, "program layout access slot");
expectThrow(
  () => programLayoutAccessors.bufferSlot(layoutProgram, 1),
  "SmokeProgram.bufferSlot requires a buffer slot name or kind",
  "program layout access slot label",
);
const kvLayoutCalls = [];
const llamaLayoutAccessors = programLayoutAccess.createProgramLayoutAccessors({
  programPolicy: { bufferLayout: () => ({ slots: [] }) },
  assertProgramAlive(handle, label) {
    kvLayoutCalls.push(["alive", handle, label]);
    if (handle === 0) throw new Error(`${label} is freed`);
  },
  llamaKvCacheRequirements(handle) {
    kvLayoutCalls.push(["requirements", handle]);
    return {
      scalarBytes: 4,
      layers: 2,
      contextLength: 8,
      kBufferByteLength: 16,
      vBufferByteLength: 32,
      bufferByteLength: 48,
    };
  },
});
expectSame(llamaLayoutAccessors.kvCacheRequirements({ handle: 45 }).layers, 2, "program layout access kv requirements");
expectSame(llamaLayoutAccessors.kvCacheLayout({ handle: 45 }).k[0].byteLength, 16, "program layout access kv layout");
expectSame(kvLayoutCalls, [
  ["alive", 45, "program"],
  ["requirements", 45],
  ["alive", 45, "program"],
  ["alive", 45, "program"],
  ["requirements", 45],
], "program layout access kv calls");
expectThrow(
  () => llamaLayoutAccessors.kvCacheRequirements({ handle: 0 }),
  "program is freed",
  "program layout access kv liveness",
);
expectThrow(
  () => programLayoutAccess.createProgramLayoutAccessors({}),
  "createProgramLayoutAccessors requires programPolicy.bufferLayout",
  "program layout access rejects missing policy",
);

const policyAccessorCalls = [];
const genericPolicyAccessors = programPolicyAccess.createGenericProgramPolicyAccessors({
  assertProgramAlive(handle, label) {
    policyAccessorCalls.push(["alive", handle, label]);
    if (handle === 0) throw new Error(`${label} is freed`);
  },
  bindModuleThroughProgram(program, module, options) {
    policyAccessorCalls.push(["bindModule", program.handle, module, options]);
    return { bound: module.name, options };
  },
  programPolicy: {
    requirements(handle) {
      policyAccessorCalls.push(["requirements", handle]);
      return { inputLen: 2, outputLen: 3 };
    },
    trace(handle, evidence) {
      policyAccessorCalls.push(["trace", handle, evidence.tag]);
      return evidence.trace;
    },
    compilerSignatures(handle, evidence) {
      policyAccessorCalls.push(["compilerSignatures", handle, evidence.tag]);
      return evidence.compilerSignatures;
    },
    tensorProgramIr(handle, evidence) {
      policyAccessorCalls.push(["ir", handle, evidence.tag]);
      return evidence.ir;
    },
    kernelPlan(handle, evidence) {
      policyAccessorCalls.push(["kernelPlan", handle, evidence.tag]);
      return evidence.kernelPlan;
    },
    shapeConstraints(handle, evidence) {
      policyAccessorCalls.push(["shapeConstraints", handle, evidence.tag]);
      return evidence.kernelPlan.shapeConstraints;
    },
    memoryLayout(handle, evidence) {
      policyAccessorCalls.push(["memoryLayout", handle, evidence.tag]);
      return evidence.kernelPlan.memoryLayout;
    },
    parameterLayout(handle, evidence) {
      policyAccessorCalls.push(["parameterLayout", handle, evidence.tag]);
      return evidence.kernelPlan.parameterLayout;
    },
    modelCompatibility(handle, model) {
      policyAccessorCalls.push(["modelCompatibility", handle, model.name]);
      return { compatible: model.name === "ok" };
    },
    acceptsModel(handle, model) {
      policyAccessorCalls.push(["acceptsModel", handle, model.name]);
      return model.name === "ok";
    },
    moduleCompatibility(handle, evidence, module, options) {
      policyAccessorCalls.push(["moduleCompatibility", handle, evidence.tag, module.name, options]);
      return { compatible: module.name === "ok" };
    },
    acceptsModule(handle, evidence, module, options) {
      policyAccessorCalls.push(["acceptsModule", handle, evidence.tag, module.name, options]);
      return module.name === "ok";
    },
    capabilities(handle) {
      policyAccessorCalls.push(["capabilities", handle]);
      return { signature: "cap", canExecute: true };
    },
    canExecute(handle) {
      policyAccessorCalls.push(["canExecute", handle]);
      return true;
    },
    canBindExternalResources(handle) {
      policyAccessorCalls.push(["canBindExternalResources", handle]);
      return true;
    },
    hasFullDispatchPlan(handle) {
      policyAccessorCalls.push(["hasFullDispatchPlan", handle]);
      return true;
    },
    executionMode(handle) {
      policyAccessorCalls.push(["executionMode", handle]);
      return "executable";
    },
    matchesCapabilitySignature(handle, signature) {
      policyAccessorCalls.push(["matchesCapabilitySignature", handle, signature]);
      return signature === "cap";
    },
    runtimeProfile(handle) {
      policyAccessorCalls.push(["runtimeProfile", handle]);
      return { signature: "runtime" };
    },
    matchesRuntimeProfileSignature(handle, signature) {
      policyAccessorCalls.push(["matchesRuntimeProfileSignature", handle, signature]);
      return signature === "runtime";
    },
    resetRuntimeProfile(handle) {
      policyAccessorCalls.push(["resetRuntimeProfile", handle]);
      return { reset: handle };
    },
    free(program) {
      policyAccessorCalls.push(["free", program.handle]);
      program.handle = 0;
    },
    dispose(program) {
      policyAccessorCalls.push(["dispose", program.handle]);
      program.handle = 0;
    },
  },
});
const policyAccessorProgram = {
  handle: 51,
  compileEvidenceSnapshot: {
    tag: "evidence",
    trace: { ops: [] },
    ir: { kind: "ir" },
    kernelPlan: {
      memoryLayout: { bytes: 1 },
      parameterLayout: { parameters: [{ name: "weight" }] },
      shapeConstraints: { inputShape: [1] },
    },
    compilerSignatures: {
      ir: "ir",
      kernelPlan: "kernel",
      memoryLayout: "memory",
      parameterLayout: "parameter",
      bufferLayout: "buffer",
    },
  },
};
expectSame(Object.isFrozen(genericPolicyAccessors), true, "program policy accessors frozen");
expectSame(genericPolicyAccessors.compileEvidence(policyAccessorProgram).tag, "evidence", "program policy access compile evidence");
const normalizedEvidenceProgram = genericPolicyAccessors.withCompileEvidence({ handle: 52 }, policyAccessorProgram.compileEvidenceSnapshot);
expectSame(Object.isFrozen(normalizedEvidenceProgram.compileEvidenceSnapshot), true, "program policy access normalizes evidence");
expectSame(genericPolicyAccessors.requirements(policyAccessorProgram), { inputLen: 2, outputLen: 3 }, "program policy access requirements");
expectSame(genericPolicyAccessors.trace(policyAccessorProgram), { ops: [] }, "program policy access trace");
expectSame(genericPolicyAccessors.compilerSignatures(policyAccessorProgram).kernelPlan, "kernel", "program policy access signatures");
expectSame(genericPolicyAccessors.tensorProgramIr(policyAccessorProgram), { kind: "ir" }, "program policy access ir");
expectSame(genericPolicyAccessors.kernelPlan(policyAccessorProgram).memoryLayout, { bytes: 1 }, "program policy access kernel plan");
expectSame(genericPolicyAccessors.shapeConstraints(policyAccessorProgram), { inputShape: [1] }, "program policy access shape constraints");
expectSame(genericPolicyAccessors.memoryLayout(policyAccessorProgram), { bytes: 1 }, "program policy access memory layout");
expectSame(genericPolicyAccessors.parameterLayout(policyAccessorProgram), { parameters: [{ name: "weight" }] }, "program policy access parameter layout");
expectSame(genericPolicyAccessors.modelCompatibility(policyAccessorProgram, { name: "ok" }), { compatible: true }, "program policy access model compatibility");
expectSame(genericPolicyAccessors.acceptsModel(policyAccessorProgram, { name: "ok" }), true, "program policy access accepts model");
expectSame(genericPolicyAccessors.moduleCompatibility(policyAccessorProgram, { name: "ok" }, { tag: "m" }), { compatible: true }, "program policy access module compatibility");
expectSame(genericPolicyAccessors.acceptsModule(policyAccessorProgram, { name: "ok" }, { tag: "m" }), true, "program policy access accepts module");
expectSame(genericPolicyAccessors.capabilities(policyAccessorProgram), { signature: "cap", canExecute: true }, "program policy access capabilities");
expectSame(genericPolicyAccessors.canExecute(policyAccessorProgram), true, "program policy access canExecute");
expectSame(genericPolicyAccessors.canBindExternalResources(policyAccessorProgram), true, "program policy access canBindExternalResources");
expectSame(genericPolicyAccessors.hasFullDispatchPlan(policyAccessorProgram), true, "program policy access hasFullDispatchPlan");
expectSame(genericPolicyAccessors.executionMode(policyAccessorProgram), "executable", "program policy access executionMode");
expectSame(genericPolicyAccessors.matchesCapabilitySignature(policyAccessorProgram, "cap"), true, "program policy access capability signature");
expectSame(genericPolicyAccessors.runtimeProfile(policyAccessorProgram), { signature: "runtime" }, "program policy access runtime profile");
expectSame(genericPolicyAccessors.matchesRuntimeProfileSignature(policyAccessorProgram, "runtime"), true, "program policy access runtime signature");
expectSame(genericPolicyAccessors.resetRuntimeProfile(policyAccessorProgram), { reset: 51 }, "program policy access reset runtime");
expectSame(genericPolicyAccessors.bindModule(policyAccessorProgram, { name: "ok" }, { tag: "bind" }), { bound: "ok", options: { tag: "bind" } }, "program policy access bind module");
expectThrow(
  () => genericPolicyAccessors.bindModule({ handle: 0 }, { name: "ok" }),
  "program is freed",
  "program policy access bind liveness",
);
const freeProgram = { handle: 53 };
genericPolicyAccessors.free(freeProgram);
expectSame(freeProgram.handle, 0, "program policy access free");
const disposeProgram = { handle: 54 };
genericPolicyAccessors.dispose(disposeProgram);
expectSame(disposeProgram.handle, 0, "program policy access dispose");
expectThrow(
  () => programPolicyAccess.createGenericProgramPolicyAccessors({ programPolicy: {} }),
  "createGenericProgramPolicyAccessors assertProgramAlive must be a function",
  "program policy access rejects missing assert",
);

const llamaPolicyCalls = [];
const llamaPolicyAccessors = programPolicyAccess.createLlamaProgramPolicyAccessors({
  assertProgramAlive(handle, label) {
    llamaPolicyCalls.push(["alive", handle, label]);
    if (handle === 0) throw new Error(`${label} is freed`);
  },
  inspectLlamaProgram(handle) {
    llamaPolicyCalls.push(["inspect", handle]);
    return { vocabSize: 123, hiddenSize: 16 };
  },
  inspectExecutableProgram(handle) {
    llamaPolicyCalls.push(["inspectExecutable", handle]);
    return { executable: handle };
  },
  createProgramLlamaKvCache(handle, options) {
    llamaPolicyCalls.push(["createKvCache", handle, options]);
    return { handle, options };
  },
  bindLlamaProgramSession(handle, program, options, bindSessionHandle, createSession) {
    llamaPolicyCalls.push(["bind", handle, program === llamaPolicyProgram, options, bindSessionHandle, createSession]);
    return { session: handle, options };
  },
  programPolicy: {
    requirements(handle) {
      llamaPolicyCalls.push(["requirements", handle]);
      return { inputLen: 1, outputLen: 123 };
    },
    modelCompatibility(handle, model) {
      llamaPolicyCalls.push(["modelCompatibility", handle, model.name]);
      return { compatible: model.name === "llama" };
    },
    acceptsModel(handle, model) {
      llamaPolicyCalls.push(["acceptsModel", handle, model.name]);
      return model.name === "llama";
    },
    capabilities(handle) {
      llamaPolicyCalls.push(["capabilities", handle]);
      return { signature: "llama-cap" };
    },
    canExecute(handle) {
      llamaPolicyCalls.push(["canExecute", handle]);
      return true;
    },
    canBindExternalResources(handle) {
      llamaPolicyCalls.push(["canBindExternalResources", handle]);
      return true;
    },
    hasFullDispatchPlan(handle) {
      llamaPolicyCalls.push(["hasFullDispatchPlan", handle]);
      return true;
    },
    executionMode(handle) {
      llamaPolicyCalls.push(["executionMode", handle]);
      return "executable";
    },
    matchesCapabilitySignature(handle, signature) {
      llamaPolicyCalls.push(["matchesCapabilitySignature", handle, signature]);
      return signature === "llama-cap";
    },
    runtimeProfile(handle) {
      llamaPolicyCalls.push(["runtimeProfile", handle]);
      return { signature: "llama-runtime" };
    },
    matchesRuntimeProfileSignature(handle, signature) {
      llamaPolicyCalls.push(["matchesRuntimeProfileSignature", handle, signature]);
      return signature === "llama-runtime";
    },
    resetRuntimeProfile(handle) {
      llamaPolicyCalls.push(["resetRuntimeProfile", handle]);
      return { reset: handle };
    },
    free(program) {
      llamaPolicyCalls.push(["free", program.handle]);
      program.handle = 0;
    },
    dispose(program) {
      llamaPolicyCalls.push(["dispose", program.handle]);
      program.handle = 0;
    },
  },
});
const llamaPolicyProgram = { handle: 61 };
expectSame(Object.isFrozen(llamaPolicyAccessors), true, "llama policy accessors frozen");
expectSame(llamaPolicyAccessors.inspect(llamaPolicyProgram), { vocabSize: 123, hiddenSize: 16 }, "llama policy access inspect");
expectSame(llamaPolicyAccessors.inspect(llamaPolicyProgram), { vocabSize: 123, hiddenSize: 16 }, "llama policy access inspect cached");
expectSame(llamaPolicyCalls.filter((call) => call[0] === "inspect").length, 1, "llama policy access inspect cache count");
expectSame(llamaPolicyAccessors.vocabSize(llamaPolicyProgram), 123, "llama policy access vocab size");
expectSame(llamaPolicyAccessors.requirements(llamaPolicyProgram), { inputLen: 1, outputLen: 123 }, "llama policy access requirements");
expectSame(llamaPolicyAccessors.modelCompatibility(llamaPolicyProgram, { name: "llama" }), { compatible: true }, "llama policy access model compatibility");
expectSame(llamaPolicyAccessors.acceptsModel(llamaPolicyProgram, { name: "llama" }), true, "llama policy access accepts model");
expectSame(llamaPolicyAccessors.capabilities(llamaPolicyProgram), { signature: "llama-cap" }, "llama policy access capabilities");
expectSame(llamaPolicyAccessors.canExecute(llamaPolicyProgram), true, "llama policy access canExecute");
expectSame(llamaPolicyAccessors.canBindExternalResources(llamaPolicyProgram), true, "llama policy access canBindExternalResources");
expectSame(llamaPolicyAccessors.hasFullDispatchPlan(llamaPolicyProgram), true, "llama policy access hasFullDispatchPlan");
expectSame(llamaPolicyAccessors.executionMode(llamaPolicyProgram), "executable", "llama policy access execution mode");
expectSame(llamaPolicyAccessors.matchesCapabilitySignature(llamaPolicyProgram, "llama-cap"), true, "llama policy access capability signature");
expectSame(llamaPolicyAccessors.createKvCache(llamaPolicyProgram, { placement: "cpu" }), { handle: 61, options: { placement: "cpu" } }, "llama policy access create kv cache");
expectSame(llamaPolicyAccessors.inspectExecutable(llamaPolicyProgram), { executable: 61 }, "llama policy access inspect executable");
expectSame(llamaPolicyAccessors.runtimeProfile(llamaPolicyProgram), { signature: "llama-runtime" }, "llama policy access runtime profile");
expectSame(llamaPolicyAccessors.matchesRuntimeProfileSignature(llamaPolicyProgram, "llama-runtime"), true, "llama policy access runtime signature");
expectSame(llamaPolicyAccessors.resetRuntimeProfile(llamaPolicyProgram), { reset: 61 }, "llama policy access reset runtime");
const llamaCreateSession = () => "session";
expectSame(llamaPolicyAccessors.bind(llamaPolicyProgram, { output: true }, 77, llamaCreateSession), { session: 61, options: { output: true } }, "llama policy access bind");
expectThrow(
  () => llamaPolicyAccessors.bind({ handle: 0 }, {}, 1, llamaCreateSession),
  "program is freed",
  "llama policy access bind liveness",
);
const llamaFreeProgram = { handle: 62 };
llamaPolicyAccessors.free(llamaFreeProgram);
expectSame(llamaFreeProgram.handle, 0, "llama policy access free");
const llamaDisposeProgram = { handle: 63 };
llamaPolicyAccessors.dispose(llamaDisposeProgram);
expectSame(llamaDisposeProgram.handle, 0, "llama policy access dispose");
expectThrow(
  () => programPolicyAccess.createLlamaProgramPolicyAccessors({ programPolicy: {} }),
  "createLlamaProgramPolicyAccessors assertProgramAlive must be a function",
  "llama policy access rejects missing assert",
);

function makeProgramFacadePolicy(requirements) {
  return {
    requirements: () => requirements,
    bufferLayout: () => ({
      slots: [{ name: "input", byteLength: requirements.inputByteLength ?? requirements.inputLen * 4 }],
      input: { name: "input", byteLength: requirements.inputByteLength ?? requirements.inputLen * 4 },
    }),
    trace: (_handle, evidence) => evidence && evidence.trace,
    compilerSignatures: (_handle, evidence) => evidence && evidence.compilerSignatures,
    tensorProgramIr: (_handle, evidence) => evidence && evidence.ir,
    kernelPlan: (_handle, evidence) => evidence && evidence.kernelPlan,
    shapeConstraints: (_handle, evidence) => (evidence && evidence.shapeConstraints) || {},
    memoryLayout: (_handle, evidence) => evidence && evidence.memoryLayout,
    parameterLayout: (_handle, evidence) => (evidence && evidence.parameterLayout) || { parameters: [] },
    modelCompatibility: (_handle, model) => ({ compatible: model && model.name === "ok" }),
    acceptsModel: (_handle, model) => !!(model && model.name === "ok"),
    moduleCompatibility: (_handle, _evidence, module) => ({ compatible: module && module.name === "ok" }),
    acceptsModule: (_handle, _evidence, module) => !!(module && module.name === "ok"),
    capabilities: () => Object.freeze({
      signature: `facade-cap|canExecute=${requirements.canExecute === false ? 0 : 1}`,
      canExecute: requirements.canExecute !== false,
      mode: requirements.canExecute === false ? "compile-only" : "executable",
      canBindExternalResources: true,
      hasFullDispatchPlan: requirements.canExecute !== false,
      diagnostics: Object.freeze(requirements.canExecute === false
        ? [Object.freeze({ code: "execution-unavailable", message: "Program has no executable dispatch plan" })]
        : []),
    }),
    canExecute: () => requirements.canExecute !== false,
    canBindExternalResources: () => true,
    hasFullDispatchPlan: () => requirements.canExecute !== false,
    executionMode: () => requirements.canExecute === false ? "compile-only" : "executable",
    matchesCapabilitySignature: (_handle, signature) => signature === `facade-cap|canExecute=${requirements.canExecute === false ? 0 : 1}`,
    runtimeProfile: () => ({ signature: "facade-runtime" }),
    matchesRuntimeProfileSignature: (_handle, signature) => signature === "facade-runtime",
    resetRuntimeProfile: (handle) => ({ reset: handle }),
    free(program) { program.handle = 0; },
    dispose(program) { program.handle = 0; },
  };
}
const facadeGenericCalls = [];
const genericFacade = programFacade.createGenericProgramFacadeHelpers({
  assertProgramAlive(handle, label) {
    facadeGenericCalls.push(["alive", handle, label]);
    if (handle === 0) throw new Error(`${label} is freed`);
  },
  programPolicy: makeProgramFacadePolicy({
    modelKind: "module",
    scalarBytes: 4,
    inputLen: 2,
    outputLen: 3,
    weightsLen: 4,
    biasLen: 1,
  }),
  inspectExecutableProgram(handle) {
    return { executable: handle };
  },
  programInputShape(desc) {
    return [desc.inputLen];
  },
  programOutputShape(desc) {
    return [desc.outputLen];
  },
  createProgramOutputBuffer(handle, options) {
    return { handle, kind: "output", options };
  },
  createProgramNamedBuffer(handle, kind, options) {
    return { handle, kind, options };
  },
  createProgramBuffer(handle, kind, options) {
    return { handle, kind, options };
  },
  programDeviceHandle(handle, placement) {
    return { handle, placement };
  },
  createProgramDevice(handle, placement) {
    return { handle, placement };
  },
  importDeviceBuffer(handle, kind, source) {
    return { handle, kind, source };
  },
  bindModuleThroughProgram(program, module, options) {
    return { program: program.handle, module: module.name, options };
  },
  programBindingPlan(program, desc, params) {
    if (params && params.reject) {
      return Object.freeze({
        kind: "zgml.program.binding-plan",
        signature: "program-binding-plan|accepted=0|mode=invalid|reason=bad",
        accepted: false,
        canBind: false,
        mode: "invalid",
        reason: "bad facade binding",
        diagnostics: Object.freeze([{ code: "binding-invalid", message: "bad facade binding" }]),
      });
    }
    return Object.freeze({ kind: "zgml.program.binding-plan", signature: `program-binding-plan|accepted=1|mode=host|program=${program.handle}|input=${desc.inputLen}`, accepted: true, canBind: true, mode: "host", program: program.handle, inputLen: desc.inputLen, params, diagnostics: Object.freeze([]) });
  },
});
const genericFacadeProgram = {
  handle: 71,
  desc: { inputLen: 2, outputLen: 3 },
  compileEvidenceSnapshot: {
    shapeConstraints: { inputShape: [1, 2] },
    parameterLayout: { parameters: [{ name: "weight" }] },
    trace: { ops: [] },
    compilerSignatures: { ir: "i", kernelPlan: "k", memoryLayout: "m", parameterLayout: "p", bufferLayout: "b" },
  },
};
expectSame(Object.isFrozen(genericFacade), true, "program facade generic frozen");
expectSame(genericFacade.inspect(genericFacadeProgram), { executable: 71 }, "program facade generic inspect");
expectSame(genericFacade.inputLen(genericFacadeProgram), 2, "program facade generic input len");
expectSame(genericFacade.outputShape(genericFacadeProgram), [3], "program facade generic output shape fallback");
expectSame(genericFacade.inputShape(genericFacadeProgram), [1, 2], "program facade generic input shape evidence");
expectSame(genericFacade.parameterNames(genericFacadeProgram), ["weight"], "program facade generic parameter names");
expectSame(genericFacade.createWeightsBuffer(genericFacadeProgram, { tag: "w" }), { handle: 71, kind: "weights", options: { tag: "w" } }, "program facade generic weights buffer");
expectSame(genericFacade.bindModule(genericFacadeProgram, { name: "ok" }, { tag: "bind" }), { program: 71, module: "ok", options: { tag: "bind" } }, "program facade generic bind module");
expectSame(genericFacade.bindingPlan(genericFacadeProgram, { input: [1, 2] }), { kind: "zgml.program.binding-plan", signature: "program-binding-plan|accepted=1|mode=host|program=71|input=2", accepted: true, canBind: true, mode: "host", program: 71, inputLen: 2, params: { input: [1, 2] }, diagnostics: [] }, "program facade generic bindingPlan");
const genericFacadeExecutionPlan = genericFacade.executionPlan(genericFacadeProgram);
expectSame(genericFacade.requireExecutionPlan(genericFacadeProgram).signature, genericFacadeExecutionPlan.signature, "program facade generic requireExecutionPlan");
expectSame(executionPlan.acceptsProgramExecutionPlan(genericFacadeExecutionPlan), true, "execution plan helper accepts executable Program plan");
expectSame(executionPlan.requireProgramExecutionPlan(genericFacadeExecutionPlan), genericFacadeExecutionPlan, "execution plan helper requires executable Program plan");
expectSame(executionPlan.assertProgramExecutionPlan(genericFacadeExecutionPlan), genericFacadeExecutionPlan, "execution plan helper assert Program alias");
expectSame(executionPlan.assert_program_execution_plan(genericFacadeExecutionPlan), genericFacadeExecutionPlan, "execution plan helper assert Program snake alias");
expectSame(executionPlan.matchesProgramExecutionPlanSignature(genericFacadeExecutionPlan, genericFacadeExecutionPlan.signature), true, "execution plan helper matches Program signature");
expectSame(genericFacade.requireExecutionPlan(genericFacadeProgram, { name: "ok" }).acceptsModule, true, "program facade generic requireExecutionPlan accepts compatible module");
expectThrow(
  () => genericFacade.requireExecutionPlan(genericFacadeProgram, { name: "bad" }),
  "Program.requireExecutionPlan rejected execution plan: module is incompatible with this Program",
  "program facade generic requireExecutionPlan rejects incompatible module",
);
expectThrow(
  () => executionPlan.requireProgramExecutionPlan(genericFacade.executionPlan(genericFacadeProgram, { name: "bad" })),
  "ProgramExecutionPlan is not executable: module is incompatible with this Program",
  "execution plan helper rejects incompatible Program plan",
);
const nonExecutableGenericFacade = programFacade.createGenericProgramFacadeHelpers({
  assertProgramAlive(handle, label) {
    if (handle === 0) throw new Error(`${label} is freed`);
  },
  programPolicy: makeProgramFacadePolicy({
    modelKind: "module",
    scalarBytes: 4,
    inputLen: 2,
    outputLen: 3,
    weightsLen: 4,
    biasLen: 1,
    canExecute: false,
  }),
  inspectExecutableProgram: (handle) => ({ executable: handle }),
  programInputShape: (desc) => [desc.inputLen],
  programOutputShape: (desc) => [desc.outputLen],
  createProgramOutputBuffer: (handle, options) => ({ handle, kind: "output", options }),
  createProgramNamedBuffer: (handle, kind, options) => ({ handle, kind, options }),
  createProgramBuffer: (handle, kind, options) => ({ handle, kind, options }),
  programDeviceHandle: (handle, placement) => ({ handle, placement }),
  createProgramDevice: (handle, placement) => ({ handle, placement }),
  importDeviceBuffer: (handle, kind, source) => ({ handle, kind, source }),
  bindModuleThroughProgram: (program, module, options) => ({ program: program.handle, module: module.name, options }),
  programBindingPlan: (program, desc, params) => Object.freeze({ kind: "zgml.program.binding-plan", accepted: true, canBind: true, program: program.handle, inputLen: desc.inputLen, params, diagnostics: Object.freeze([]) }),
});
expectSame(nonExecutableGenericFacade.executionPlan(genericFacadeProgram).canExecute, false, "program facade generic executionPlan records non-executable Program");
expectSame(executionPlan.acceptsProgramExecutionPlan(nonExecutableGenericFacade.executionPlan(genericFacadeProgram)), false, "execution plan helper rejects non-executable Program predicate");
expectThrow(
  () => nonExecutableGenericFacade.requireExecutionPlan(genericFacadeProgram),
  "Program.requireExecutionPlan rejected execution plan: Program has no executable dispatch plan",
  "program facade generic requireExecutionPlan rejects non-executable Program",
);
expectThrow(
  () => executionPlan.requireProgramExecutionPlan(nonExecutableGenericFacade.executionPlan(genericFacadeProgram)),
  "ProgramExecutionPlan is not executable: Program has no executable dispatch plan",
  "execution plan helper rejects non-executable Program plan",
);
const genericFacadeBindingPlan = genericFacade.bindingPlan(genericFacadeProgram, { input: [1, 2] });
expectSame(genericFacade.requireBindingPlan(genericFacadeProgram, { input: [1, 2] }).accepted, true, "program facade generic requireBindingPlan");
expectSame(executionPlan.acceptsProgramBindingPlan(genericFacadeBindingPlan), true, "binding plan helper accepts Program binding plan");
expectSame(executionPlan.requireProgramBindingPlan(genericFacadeBindingPlan), genericFacadeBindingPlan, "binding plan helper requires Program binding plan");
expectSame(executionPlan.assertProgramBindingPlan(genericFacadeBindingPlan), genericFacadeBindingPlan, "binding plan helper assert Program binding alias");
expectSame(executionPlan.assert_program_binding_plan(genericFacadeBindingPlan), genericFacadeBindingPlan, "binding plan helper assert Program binding snake alias");
expectSame(executionPlan.matchesProgramBindingPlanSignature(genericFacadeBindingPlan, genericFacadeBindingPlan.signature), true, "binding plan helper matches Program binding signature");
expectThrow(
  () => genericFacade.requireBindingPlan(genericFacadeProgram, { reject: true }),
  "Program.requireBindingPlan rejected bindings: bad facade binding",
  "program facade generic requireBindingPlan rejects invalid plan",
);
expectThrow(
  () => executionPlan.requireProgramBindingPlan(genericFacade.bindingPlan(genericFacadeProgram, { reject: true })),
  "ProgramBindingPlan is not bindable: bad facade binding",
  "binding plan helper rejects invalid Program binding plan",
);
expectThrow(
  () => programFacade.createGenericProgramFacadeHelpers({}),
  "createGenericProgramFacadeHelpers requires assertProgramAlive",
  "program facade generic rejects missing assert",
);

const llamaFacade = programFacade.createLlamaProgramFacadeHelpers({
  assertProgramAlive(handle, label) {
    if (handle === 0) throw new Error(`${label} is freed`);
  },
  programPolicy: makeProgramFacadePolicy({
    modelKind: "tiny-llama",
    scalarBytes: 4,
    inputLen: 1,
    outputLen: 99,
    weightsLen: 4,
    biasLen: 0,
  }),
  inspectLlamaProgram: () => ({ vocabSize: 99 }),
  inspectExecutableProgram: (handle) => ({ executable: handle }),
  llamaKvCacheRequirements: () => ({
    scalarBytes: 4,
    layers: 1,
    contextLength: 4,
    kBufferByteLength: 16,
    vBufferByteLength: 16,
    bufferByteLength: 32,
  }),
  createProgramOutputBuffer: (handle, options) => ({ handle, kind: "output", options }),
  createProgramNamedBuffer: (handle, kind, options) => ({ handle, kind, options }),
  programDeviceHandle: (handle, placement) => ({ handle, placement }),
  createProgramDevice: (handle, placement) => ({ handle, placement }),
  importDeviceBuffer: (handle, kind, source) => ({ handle, kind, source }),
  createProgramLlamaKvCache: (handle, options) => ({ handle, options }),
  bindLlamaProgramSession: (handle, _program, options) => ({ handle, options }),
});
const llamaFacadeProgram = { handle: 72 };
expectSame(Object.isFrozen(llamaFacade), true, "program facade llama frozen");
expectSame(llamaFacade.vocabSize(llamaFacadeProgram), 99, "program facade llama vocab");
expectSame(llamaFacade.outputShape(llamaFacadeProgram), [99], "program facade llama output shape");
expectSame(llamaFacade.requireExecutionPlan(llamaFacadeProgram).signature, llamaFacade.executionPlan(llamaFacadeProgram).signature, "program facade llama requireExecutionPlan");
expectSame(llamaFacade.kvCacheLayout(llamaFacadeProgram).k[0].byteLength, 16, "program facade llama kv layout");
expectSame(llamaFacade.createKvCache(llamaFacadeProgram, { placement: "cpu" }), { handle: 72, options: { placement: "cpu" } }, "program facade llama kv cache");
expectSame(llamaFacade.bind(llamaFacadeProgram, { output: true }), { handle: 72, options: { output: true } }, "program facade llama bind");
expectThrow(
  () => programFacade.createLlamaProgramFacadeHelpers({}),
  "createLlamaProgramFacadeHelpers requires assertProgramAlive",
  "program facade llama rejects missing assert",
);

expectSame(abi.modelKindName(abi.modelKinds.module), "module", "abi modelKindName");
expectSame(abi.backendId("webgpu"), 3, "abi backendId");
expectSame(abi.backendName(2n), "metal", "abi backendName bigint");
expectSame(abi.bufferStorageName(abi.bufferStorageIds.externalResource), "external-resource", "abi bufferStorageName");
expectSame(abi.resourceAccessFlags(["read", "write"]), 3, "abi resource access array");
expectSame(abi.accessFlagsToObject(3), { read: true, write: true }, "abi access flags object");
expectSame(abi.programBufferKindId("kv-v"), 6, "abi program buffer kind");
expectSame(abi.abiStructKindId("moduleDesc"), 41, "abi struct kind");
expectSame(abi.normalizeCompileOptions({ backend: "cpu", contextLength: 8, batch: 2 }), {
  backend: "cpu",
  backendId: 1,
  contextLength: 8,
  batch: 2,
}, "abi normalize compile options");
const defaultCompileOptions = abi.normalizeCompileOptions();
expectSame(abi.isDefaultCompileEnvelope(defaultCompileOptions), true, "abi detects default compile envelope");
expectSame(abi.isDefaultCompileEnvelope(abi.normalizeCompileOptions({ backend: "cpu" })), false, "abi rejects non-default compile envelope");
const compileDescFields = abi.compileDescFieldsFromOptions(abi.normalizeCompileOptions({ backend: "webgpu", contextLength: 12, batch: 3 }));
expectSame(compileDescFields, {
  backend: 3,
  reserved: 0,
  contextLen: 12,
  batch: 3,
}, "abi compile desc fields projection");
expectSame(Object.isFrozen(compileDescFields), true, "abi compile desc fields frozen");
expectSame(abi.compileDescRecordFromFields(compileDescFields), {
  backend: 3,
  reserved: 0,
  context_len: 12,
  batch: 3,
}, "abi compile desc record from fields");
expectSame(abi.compileDescRecordFromOptions(abi.normalizeCompileOptions({ backend: "webgpu", contextLength: 12, batch: 3 })), {
  backend: 3,
  reserved: 0,
  context_len: 12,
  batch: 3,
}, "abi compile desc record projection");
const allRequiredFeatures = abi.requiredRuntimeFeatureMask;
const runtimeFeatures = abi.decodeRuntimeFeatures(allRequiredFeatures);
expectSame(runtimeFeatures.bufferHandle, true, "abi decode required feature");
expectSame(runtimeFeatures.experimentalLlamaWgpuExecution, false, "abi decode optional feature");
const runtimeInfo = {
  abiVersion: abi.expectedRuntimeAbiVersion,
  tokenIdBytes: abi.expectedRuntimeTokenIdBytes,
  featureFlags: allRequiredFeatures,
};
expectSame(abi.assertCompatibleRuntimeInfo(runtimeInfo), runtimeInfo, "abi compatible runtime info");
const runtimeFacade = abi.createRuntimeAbiFacadeHelpers({
  readRuntimeInfo() {
    return { ...runtimeInfo, sizeTBytes: 8, pointerBytes: 8 };
  },
  readAbiStructSize(kind) {
    return kind === abi.abiStructKindId("runtimeInfo") ? 32 : 8;
  },
});
expectSame(runtimeFacade.runtimeInfo().features.bufferHandle, true, "abi facade runtime info");
expectSame(runtimeFacade.abiStructSize("runtimeInfo"), 32, "abi facade struct size");
expectSame(runtimeFacade.assertCompatibleRuntime().abiVersion, abi.expectedRuntimeAbiVersion, "abi facade compatible runtime");
expectThrow(() => abi.backendId("gpu"), "unknown zgml backend: gpu", "abi invalid backend");
expectThrow(() => abi.resourceAccessFlags({}), "external resource access must include read or write", "abi invalid access object");
expectThrow(
  () => abi.assertCompatibleRuntimeInfo({ abiVersion: 0, tokenIdBytes: 4, featureFlags: allRequiredFeatures }),
  "unsupported zgml C ABI version 0; expected 6",
  "abi invalid version",
);
expectSame(nativeKernelContract.nativeKernelContractManifest.policyOwner, "src/ts/runtime/native_kernel_contract.ts", "native kernel contract manifest owner");
expectSame(
  nativeKernelContract.kernelNameForNativeModuleDesc({ kind: abi.moduleOpIds.linear, activation: 0, flags: abi.moduleFlags.bias, a: 2, b: 3, c: 0, eps: 0 }),
  "linear",
  "native kernel contract descriptor kernel name",
);
expectSame(
  nativeKernelContract.kernelNameForNativeModuleDesc({ kind: abi.moduleOpIds.maxPool2d, activation: 0, flags: 0, a: 2, b: 2, c: 0, eps: 0 }),
  "max-pool2d",
  "native kernel contract MaxPool2d descriptor kernel name",
);
expectSame(
  nativeKernelContract.kernelNameForNativeModuleDesc({ kind: abi.moduleOpIds.avgPool2d, activation: 0, flags: 0, a: 2, b: 2, c: 0, eps: 0 }),
  "avg-pool2d",
  "native kernel contract AvgPool2d descriptor kernel name",
);
expectSame(
  nativeKernelContract.nativeModuleDescSignature({ kind: abi.moduleOpIds.linear, activation: 0, flags: abi.moduleFlags.bias, a: 2, b: 3, c: 0, eps: 0 }),
  `${abi.moduleOpIds.linear}:0:${abi.moduleFlags.bias}:0:2:3:0:0`,
  "native kernel contract descriptor signature",
);
expectSame(
  nativeKernelContract.kernelNamesForNativeModuleDescs([
    { kind: abi.moduleOpIds.linear },
    { kind: abi.moduleOpIds.activation, activation: abi.moduleActivationIds.relu },
  ]),
  ["linear", "relu"],
  "native kernel contract descriptor kernel names",
);
expectSame(
  nativeKernelContract.nativeModuleDescSignatures([{ kind: abi.moduleOpIds.reshape, a: 2, b: 3 }]),
  [`${abi.moduleOpIds.reshape}:0:0:0:2:3:0:0`],
  "native kernel contract descriptor signatures",
);
expectSame(
  nativeKernelContract.nativeModuleOpDescAbiFields({ kind: abi.moduleOpIds.slice, reserved: 2, a: 1, b: 3, c: 4 }),
  { kind: abi.moduleOpIds.slice, activation: 0, flags: 0, reserved: 2, a: 1, b: 3, c: 4, eps: 0 },
  "native kernel contract descriptor ABI fields",
);
expectThrow(
  () => nativeKernelContract.nativeModuleOpDescAbiFields({ kind: abi.moduleOpIds.slice, reserved: 0x100000000 }),
  "native module op descriptor.reserved must fit uint32_t, got 4294967296",
  "native kernel contract descriptor ABI fields reject oversized reserved",
);

const runtimeProfile = inspection.runtimeProfileFromAbiRecord({
  call_count: 1,
  backend_op_count: 2,
  fallback_op_count: 0,
  backend_dispatch_count: 2,
  sync_count: 3,
  runtime_patch_call_count: 4,
  runtime_patch_changed_count: 5,
  runtime_patch_invalid_count: 6,
  runtime_patch_holes: 7,
  runtime_patch_cache_write_pos_holes: 8,
  runtime_patch_attention_seq_kv_holes: 9,
  runtime_patch_stencil_hash: 10n,
  command_count: 11,
  command_stencil_hash: 12n,
  command_op_count: 13,
  command_row_count: 14,
  command_projection_count: 15,
  command_attention_count: 16,
  command_movement_count: 17,
  command_elementwise_count: 18,
  command_rope_count: 19,
});
expectSame(runtimeProfile.kind, "zgml.runtime.profile", "inspection runtime profile kind");
expectSame(runtimeProfile.signature.startsWith("runtime-profile|callCount=1"), true, "inspection runtime profile signature");
expectSame(inspection.acceptsRuntimeProfile(runtimeProfile), true, "inspection runtime profile predicate accepts ABI profile");
expectSame(inspection.requireRuntimeProfile(runtimeProfile), runtimeProfile, "inspection runtime profile require returns evidence");
expectSame(inspection.assertRuntimeProfile(runtimeProfile), runtimeProfile, "inspection runtime profile assert alias");
expectSame(inspection.assert_runtime_profile(runtimeProfile), runtimeProfile, "inspection runtime profile assert snake alias");
expectSame(inspection.matchesRuntimeProfileSignature(runtimeProfile, runtimeProfile.signature), true, "inspection runtime profile signature match");
expectSame(inspection.matchesRuntimeProfileSignature(runtimeProfile, "wrong"), false, "inspection runtime profile signature mismatch");
expectThrow(
  () => inspection.requireRuntimeProfile({ ...runtimeProfile, syncCount: -1 }),
  "RuntimeProfile is not valid evidence: syncCount is not a non-negative integer",
  "inspection runtime profile require rejects malformed count",
);
const programInspection = inspection.programInspectionFromAbiRecord({
  backend: abi.backendIds.cpu,
  execution_supported: 1,
  external_resources_supported: 1,
  buffer_count: 4,
  buffer_element_count: 20,
  buffer_byte_len: 80,
  initial_upload_count: 1,
  qweight_count: 0,
  op_count: 3,
  command_count: 2,
  command_stencil_hash: 123n,
  runtime_patch_holes: 0,
  runtime_patch_cache_write_pos_holes: 0,
  runtime_patch_attention_seq_kv_holes: 0,
  runtime_patch_max_cache_write_pos: 0xffffffffffffffffn,
  runtime_patch_max_attention_seq_kv: 9n,
  runtime_patch_stencil_hash: 456n,
  command_op_count: 3,
  command_row_count: 1,
  command_projection_count: 1,
  command_attention_count: 1,
  command_movement_count: 0,
  command_elementwise_count: 0,
  command_rope_count: 0,
  backend_dispatch_count: 3,
  dispatch_plan_supported: 1,
  dispatch_plan_covered_op_count: 3,
  dispatch_plan_first_unsupported_op: 0xffffffffffffffffn,
  dispatch_plan_projection_count: 1,
  dispatch_plan_row_count: 1,
  dispatch_plan_attention_count: 1,
  dispatch_plan_movement_count: 0,
  dispatch_plan_elementwise_count: 0,
  dispatch_plan_rope_count: 0,
  dispatch_plan_quantized_projection_count: 0,
  binding_requirement_hash: 77n,
  persistent_requirement_count: 2,
  step_input_requirement_count: 1,
  step_output_requirement_count: 1,
});
expectSame(programInspection.kind, "zgml.program.inspection", "inspection program kind");
expectSame(programInspection.backend, "cpu", "inspection program backend");
expectSame(programInspection.runtimePatchMaxCacheWritePos, null, "inspection optional sentinel");
expectSame(programInspection.diagnostics, [], "inspection program diagnostics");
const capabilities = inspection.programExecutionCapabilitiesFromInspection(programInspection);
expectSame(capabilities.kind, "zgml.program.capabilities", "inspection program capabilities kind");
expectSame(capabilities.hasFullDispatchPlan, true, "inspection capabilities full plan");
expectSame(inspection.programCapabilityCanExecute(capabilities), true, "inspection can execute");
expectSame(inspection.programCapabilityCanBindExternalResources(capabilities), true, "inspection can bind resources");
expectSame(inspection.programCapabilityHasFullDispatchPlan(capabilities), true, "inspection full dispatch predicate");
expectSame(inspection.programExecutionModeFromCapabilities(null), "compile-only", "inspection missing capability mode");
expectSame(inspection.programRuntimeDiagnostics({
  backend: "webgpu",
  executionSupported: false,
  externalResourcesSupported: true,
  dispatchPlanSupported: false,
  opCount: 3,
  dispatchPlanCoveredOpCount: 0,
  dispatchPlanFirstUnsupportedOp: null,
}, "resource-probe")[0].code, "execution-unavailable", "inspection runtime diagnostic");
expectSame(inspection.sessionInspectionFromAbiRecord({
  model_kind: abi.modelKinds.tinyLlama,
  backend: abi.backendIds.cpu,
  output_storage: abi.bufferStorageIds.host,
  kv_cache_storage: abi.bufferStorageIds.externalResource,
  position: 2,
  context_len: 8,
  persistent_binding_count: 3,
  step_input_count: 1,
  step_output_count: 1,
  host_binding_count: 2,
  resource_binding_count: 1,
  binding_shape_hash: 99n,
}).kind, "zgml.session.inspection", "inspection session kind");
expectSame(inspection.sessionInspectionFromAbiRecord({
  model_kind: abi.modelKinds.tinyLlama,
  backend: abi.backendIds.cpu,
  output_storage: abi.bufferStorageIds.host,
  kv_cache_storage: abi.bufferStorageIds.externalResource,
  position: 2,
  context_len: 8,
  persistent_binding_count: 3,
  step_input_count: 1,
  step_output_count: 1,
  host_binding_count: 2,
  resource_binding_count: 1,
  binding_shape_hash: 99n,
}).kvCacheStorage, "external-resource", "inspection session storage");
const bufferInspection = inspection.bufferInspectionFromAbiRecord({
  storage: abi.bufferStorageIds.externalResource,
  placement: abi.backendIds.webgpu,
  access_flags: abi.resourceAccessIds.readwrite,
  byte_len: 16,
  handle: 7,
  byte_offset: 4,
  resource_byte_len: 32,
});
expectSame(bufferInspection.kind, "zgml.buffer.inspection", "inspection buffer kind");
expectSame(bufferInspection.access, { read: true, write: true }, "inspection buffer access");
const modelInspection = inspection.modelInspectionFromAbiRecord({
  model_kind: abi.modelKinds.tinyLlama,
  input_len: 1,
  output_len: 2,
  vocab_size: 3,
  max_seq_len: 4,
  d_model: 5,
  n_layers: 6,
  n_heads: 7,
  n_kv_heads: 8,
  d_ff: 9,
  rope_base: 10000,
  rms_norm_eps: 0.00001,
  tied_lm_head: 1,
});
expectSame(modelInspection.kind, "zgml.model.inspection", "inspection model inspection kind");
expectSame(modelInspection.modelKind, "tiny-llama", "inspection model kind");
expectSame(modelInspection.signature.startsWith("model-inspection|modelKind=tiny-llama|"), true, "inspection model signature");
const programRequirements = inspection.programRequirementsFromAbiRecord({
  model_kind: abi.modelKinds.module,
  scalar_bytes: 4,
  token_id_bytes: 4,
  input_len: 2,
  input_byte_len: 8,
  output_len: 3,
  weights_len: 6,
  weights_byte_len: 24,
  bias_len: 3,
  bias_byte_len: 12,
  parameter_len: 9,
  parameter_byte_len: 36,
  logits_len: 3,
  output_byte_len: 12,
  context_len: 0,
  batch: 0,
  max_token_window: 0,
});
expectSame(programRequirements.kind, "zgml.program.requirements", "inspection program requirements kind");
expectSame(programRequirements.signature.startsWith("program-requirements|modelKind=module|"), true, "inspection program requirements signature");
expectSame(programRequirements.weightsByteLength, 24, "inspection program requirements");
const kvCacheRequirements = inspection.llamaKvCacheRequirementsFromAbiRecord({
  model_kind: abi.modelKinds.tinyLlama,
  scalar_bytes: 4,
  n_layers: 2,
  k_buffer_byte_len: 16,
  v_buffer_byte_len: 32,
  buffer_byte_len: 96,
  context_len: 8,
});
expectSame(kvCacheRequirements.kind, "zgml.llama.kv-cache.requirements", "inspection kv requirements kind");
expectSame(kvCacheRequirements.signature.startsWith("llama-kv-cache-requirements|modelKind=tiny-llama|"), true, "inspection kv requirements signature");
expectSame(kvCacheRequirements.layers, 2, "inspection kv requirements");
expectSame(inspection.programModelCompatibilityFromAbiRecord({
  program_model_kind: abi.modelKinds.tinyLlama,
  model_kind: abi.modelKinds.tinyLlama,
  compatible: 1,
}), {
  kind: "zgml.program.model-compatibility",
  signature: "program-model-compatibility|program=tiny-llama|model=tiny-llama|compatible=1",
  programModelKind: "tiny-llama",
  modelKind: "tiny-llama",
  compatible: true,
}, "inspection model compatibility");
const llamaProgramInspectionEvidence = llamaProgramInspection.llamaProgramInspectionFromRecord({
  vocab_size: 10,
  max_seq_len: 8,
  context_len: 8,
  batch: 1,
  d_model: 4,
  n_layers: 1,
  n_heads: 1,
  n_kv_heads: 1,
  semantic_stage_count: 3,
  semantic_token_count: 1,
  semantic_layer_stage_count: 1,
  semantic_terminal_stage_count: 1,
  semantic_runtime_patch_holes: 2,
  semantic_runtime_patch_cache_write_pos_holes: 1,
  semantic_runtime_patch_attention_seq_kv_holes: 1,
});
expectSame(llamaProgramInspection.llamaProgramInspectionManifest.policyOwner, "src/ts/runtime/llama_program_inspection.ts", "llama program inspection runtime manifest owner");
expectSame(llamaProgramInspectionEvidence.kind, "zgml.llama.program.inspection", "llama program inspection kind");
expectSame(llamaProgramInspectionEvidence.signature.startsWith("llama-program-inspection|vocabSize=10|"), true, "llama program inspection signature");

const modelPathFields = modelSourceDesc.modelPathDescriptorFields(7, { length: 3, byteLength: 99, label: "path" });
expectSame(modelSourceDesc.modelSourceDescriptorManifest.policyOwner, "src/ts/runtime/model_source_desc.ts", "model source descriptor runtime manifest owner");
expectSame(modelPathFields.pathLen, 3, "model source descriptor path length prefers length");
expectSame(Object.isFrozen(modelPathFields), true, "model source descriptor path fields frozen");
expectSame(modelSourceDesc.modelPathDescriptorRecord(modelPathFields), {
  kind: 7,
  path: { length: 3, byteLength: 99, label: "path" },
  path_len: 3,
}, "model source descriptor path record");
expectSame(Object.isFrozen(modelSourceDesc.modelPathDescriptorRecord(modelPathFields)), true, "model source descriptor path record frozen");
const modelDataFields = modelSourceDesc.safetensorsDataDescriptorFields(8, { byteLength: 5, label: "data" });
expectSame(modelDataFields, {
  kind: 8,
  reserved: 0,
  data: { byteLength: 5, label: "data" },
  dataLen: 5,
}, "model source descriptor data fields");
expectSame(modelSourceDesc.safetensorsDataDescriptorRecord(modelDataFields), {
  kind: 8,
  reserved: 0,
  data: { byteLength: 5, label: "data" },
  data_len: 5,
}, "model source descriptor data record");
const modelHeaderFields = modelSourceDesc.safetensorsHeaderDescriptorFields(9, { length: 4, byteLength: 4, label: "header" });
expectSame(modelSourceDesc.safetensorsHeaderDescriptorRecord(modelHeaderFields), {
  kind: 9,
  reserved: 0,
  header: { length: 4, byteLength: 4, label: "header" },
  header_len: 4,
}, "model source descriptor header record");
expectSame(modelSourceDesc.tinyLlamaModelDescriptorRecord(modelSourceDesc.tinyLlamaModelDescriptorFields(10)), {
  kind: 10,
  input_len: 0,
  output_len: 0,
}, "model source descriptor tiny llama record");

expectSame(modelSource.safetensorsHeaderBytes("ab"), [97, 98], "model source header string bytes");
const headerView = new Uint8Array([1, 2]);
expectSame(modelSource.safetensorsHeaderBytes(headerView) === headerView, true, "model source header view identity");
expectSame(modelSource.safetensorsDataBytes(new Uint8Array([3, 4])), [3, 4], "model source data view");
expectSame(modelSource.safetensorsDataBytes(new ArrayBuffer(2)), [0, 0], "model source data buffer");
expectSame(modelSource.isSafetensorsDataSource(new Uint8Array(1)), true, "model source data source");
expectSame(modelSource.isSafetensorsPath("X.SAFETENSORS"), true, "model source safetensors path");
expectSame(modelSource.normalizeLoadModelKind("tinyllama2layer"), "tiny-llama-2layer", "model source kind alias");
expectSame(modelSource.normalizeLoadModelKind({ modelKind: "smollm" }), "smollm-135m", "model source option alias");
expectSame(modelSource.modelLoadKindId("smollm-135m"), abi.modelKinds.smollm135m, "model source kind id");
expectSame(modelSource.modelKindName(abi.modelKinds.tinyLlama2Layer), "tiny-llama-2layer", "model source modelKindName export");
expectThrow(
  () => modelSource.normalizeLoadModelKind("mystery"),
  "unknown zgml model load kind: mystery",
  "model source invalid kind",
);

const safetensorsFile = new Uint8Array(11);
new DataView(safetensorsFile.buffer).setBigUint64(0, 3n, true);
safetensorsFile.set([9, 8, 7], 8);
let closedFd = null;
const safetensorsFileHelpers = modelSource.createSafetensorsFileHeaderHelpers({
  openFile(path) {
    expectSame(path, "model.safetensors", "model source open path");
    return 12;
  },
  closeFile(fd) {
    closedFd = fd;
  },
  readFile(fd, target, offset, length, position) {
    expectSame(fd, 12, "model source read fd");
    const chunk = safetensorsFile.subarray(position, position + length);
    target.set(chunk, offset);
    return chunk.length;
  },
  allocBytes(length) {
    return new Uint8Array(length);
  },
  readU64LE(bytes) {
    return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getBigUint64(0, true);
  },
});
expectSame(safetensorsFileHelpers.readSafetensorsHeaderFile("model.safetensors"), [9, 8, 7], "model source header file");
expectSame(closedFd, 12, "model source header close");

let adapterClosedFd = null;
const adapterSafetensorsFileHelpers = createAdapterSafetensorsFileHeaderHelpers({
  openSync(path, flags) {
    expectSame({ path, flags }, { path: "adapter.safetensors", flags: "r" }, "Adapter safetensors file open");
    return 31;
  },
  closeSync(fd) {
    adapterClosedFd = fd;
  },
  readSync(fd, target, offset, length, position) {
    expectSame(fd, 31, "Adapter safetensors file fd");
    const chunk = safetensorsFile.subarray(position, position + length);
    target.set(chunk, offset);
    return chunk.length;
  },
});
expectSame(adapterSafetensorsFileHelpers.readSafetensorsHeaderFile("adapter.safetensors"), [9, 8, 7], "Adapter safetensors header file");
expectSame(adapterClosedFd, 31, "Adapter safetensors header close");

const sourceFacade = modelSource.createModelSourceFacadeHelpers({
  readSafetensorsHeaderFile(path) {
    return new Uint8Array([path.length]);
  },
  probePath(path, kind) {
    return { route: "path", path, kind };
  },
  probeSafetensorsData(data, options) {
    return { route: "data", bytes: data.byteLength, kind: options.kind };
  },
  probeSafetensorsHeader(header, options) {
    return { route: "header", header: Array.from(header), kind: options.kind };
  },
  loadPathByKind: {
    auto(path) { return { route: "load-path", path }; },
    "tiny-llama"(path) { return { route: "load-path-tiny", path }; },
    "tiny-llama-2layer"(path) { return { route: "load-path-2layer", path }; },
    "smollm-135m"(path) { return { route: "load-path-smollm", path }; },
  },
  loadSafetensorsDataByKind: {
    auto(data) { return { route: "load-data", bytes: data.byteLength }; },
    "tiny-llama"(data) { return { route: "load-data-tiny", bytes: data.byteLength }; },
    "tiny-llama-2layer"(data) { return { route: "load-data-2layer", bytes: data.byteLength }; },
    "smollm-135m"(data) { return { route: "load-data-smollm", bytes: data.byteLength }; },
  },
});
expectSame(sourceFacade.probeModel("plain.gguf", "tiny-llama"), {
  route: "path",
  path: "plain.gguf",
  kind: "tiny-llama",
}, "model source probe path");
expectSame(sourceFacade.probeModel("weights.safetensors", "smollm"), {
  route: "header",
  header: [19],
  kind: "smollm-135m",
}, "model source probe safetensors path");
expectSame(sourceFacade.probeModel(new Uint8Array([1, 2, 3])), {
  route: "data",
  bytes: 3,
  kind: "auto",
}, "model source probe data");
expectSame(sourceFacade.loadModel("plain.gguf", "tiny-llama-2layer"), {
  route: "load-path-2layer",
  path: "plain.gguf",
}, "model source load path");
expectSame(sourceFacade.loadModel(new Uint8Array([1, 2]), "tiny-llama"), {
  route: "load-data-tiny",
  bytes: 2,
}, "model source load data");

class AdapterLlamaModel {
  constructor(handle) {
    this.handle = handle;
  }
  static load(source) {
    return { route: "llama-path", source };
  }
  static loadSafetensorsData(data) {
    return { route: "llama-data", bytes: data.byteLength };
  }
}
class AdapterTinyLlamaModel {
  static load(source) {
    return { route: "tiny-path", source };
  }
  static loadSafetensorsData(data) {
    return { route: "tiny-data", bytes: data.byteLength };
  }
}
class AdapterSmolLMModel {
  static load(source) {
    return { route: "smol-path", source };
  }
  static loadSafetensorsData(data) {
    return { route: "smol-data", bytes: data.byteLength };
  }
}
const adapterModelSourceFacade = createAdapterModelSourceFacade({
  readSafetensorsHeaderFile(path) {
    return new Uint8Array([path.length]);
  },
  probeModelPath(path, kind) {
    return { route: "probe-path", path, kind };
  },
  probeSafetensorsData(data, options) {
    return { route: "probe-data", bytes: data.byteLength, kind: options.kind };
  },
  probeSafetensorsHeader(header, options) {
    return { route: "probe-header", header: Array.from(header), kind: options.kind };
  },
  loadModelPath(nativeKind, path) {
    return { nativeKind, path };
  },
  loadSafetensorsDataHandle(nativeKind, data) {
    return { nativeKind, bytes: data.byteLength };
  },
  tinyLlama2LayerKind: 42,
  LlamaModel: AdapterLlamaModel,
  TinyLlamaModel: AdapterTinyLlamaModel,
  SmolLM135MModel: AdapterSmolLMModel,
});
expectSame(adapterModelSourceFacade.probeModel("weights.safetensors", "tiny-llama"), {
  route: "probe-header",
  header: [19],
  kind: "tiny-llama",
}, "Adapter model-source facade probes safetensors path headers");
expectSame(adapterModelSourceFacade.loadModel("plain.gguf", "smollm"), {
  route: "smol-path",
  source: "plain.gguf",
}, "Adapter model-source facade routes smollm path loads");
expectSame(adapterModelSourceFacade.loadModel("plain.gguf", "tiny-llama-2layer"), {
  handle: { nativeKind: 42, path: "plain.gguf" },
}, "Adapter model-source facade routes 2-layer path loads through native handle");
expectSame(adapterModelSourceFacade.loadSafetensorsData(new Uint8Array([1, 2, 3]), "tiny-llama-2layer"), {
  handle: { nativeKind: 42, bytes: 3 },
}, "Adapter model-source facade routes 2-layer data loads through native handle");

const adapterModelSourceSurfaceCalls = [];
const adapterModelSourceSurface = createAdapterModelSourceSurface({
  getModelSourceFacade: () => ({
    probeModel(source, options) {
      adapterModelSourceSurfaceCalls.push(["probe", source, options]);
      return { source, options, route: "probe" };
    },
    loadModel(source, options) {
      adapterModelSourceSurfaceCalls.push(["load", source, options]);
      return { source, options, route: "load" };
    },
    loadSafetensorsData(data, options) {
      adapterModelSourceSurfaceCalls.push(["load-data", data, options]);
      return { bytes: data.byteLength, options, route: "load-data" };
    },
  }),
  uninitializedMessage: "Adapter model-source surface is not initialized",
});
expectSame(adapterModelSourceSurface.probeModel("plain.gguf", { modelKind: "tiny-llama" }), { source: "plain.gguf", options: { modelKind: "tiny-llama" }, route: "probe" }, "Adapter model-source surface forwards probeModel");
expectSame(adapterModelSourceSurface.loadModel("plain.gguf"), { source: "plain.gguf", options: {}, route: "load" }, "Adapter model-source surface defaults loadModel options");
expectSame(adapterModelSourceSurface.loadSafetensorsData(new Uint8Array([1, 2, 3])), { bytes: 3, options: {}, route: "load-data" }, "Adapter model-source surface defaults loadSafetensorsData options");
expectSame(adapterModelSourceSurfaceCalls.map((call) => call[0]), ["probe", "load", "load-data"], "Adapter model-source surface call order");
expectThrow(
  () => createAdapterModelSourceSurface({
    getModelSourceFacade: () => null,
    uninitializedMessage: "Adapter model-source surface is not initialized",
  }).probeModel("plain.gguf"),
  "Adapter model-source surface is not initialized",
  "Adapter model-source surface requires initialization",
);
const hostModelSourceSurfaceCalls = [];
const hostModelSourceSurface = createHostModelSourceSurface({
  getModelSourceFacade: () => ({
    probeModel(source, options) {
      hostModelSourceSurfaceCalls.push(["probe", source, options]);
      return { source, options, host: true };
    },
    loadModel(source, options) {
      hostModelSourceSurfaceCalls.push(["load", source, options]);
      return { source, options, host: true };
    },
    loadSafetensorsData(data, options) {
      hostModelSourceSurfaceCalls.push(["load-data", data, options]);
      return { bytes: data.byteLength, options, host: true };
    },
  }),
  uninitializedMessage: "host model source missing",
});
expectSame(hostModelSourceSurface.probeModel("plain.gguf"), { source: "plain.gguf", options: {}, host: true }, "host model-source surface defaults probe options");
expectSame(hostModelSourceSurface.loadModel("plain.gguf", { modelKind: "tiny-llama" }), { source: "plain.gguf", options: { modelKind: "tiny-llama" }, host: true }, "host model-source surface forwards load options");
expectSame(hostModelSourceSurface.loadSafetensorsData(new Uint8Array([1, 2, 3, 4])), { bytes: 4, options: {}, host: true }, "host model-source surface defaults load-data options");
expectSame(hostModelSourceSurfaceCalls.map((call) => call[0]), ["probe", "load", "load-data"], "host model-source surface call order");
expectThrow(
  () => createHostModelSourceSurface({ getModelSourceFacade: () => null, uninitializedMessage: "host model source missing" }).probeModel("plain.gguf"),
  "host model source missing",
  "host model-source surface requires initialization",
);

const checkpointCatalog = modelSource.createSupportedCheckpointCatalogHelpers({
  countSupportedCheckpoints() { return 2; },
  inspectSupportedCheckpoint(index) { return { index }; },
});
const supportedModels = checkpointCatalog.supportedCheckpointModels();
expectSame(supportedModels, [{ index: 0 }, { index: 1 }], "model source supported checkpoints");
expectSame(Object.isFrozen(supportedModels), true, "model source supported checkpoints frozen");

const modelHandleHelpers = modelSource.createModelHandleHelpers({
  nullHandle: 0,
  isBindableModel(model) { return model && model.family === "llama"; },
  isCompatibleModel(model) { return model && model.handle != null; },
  getModelHandle(model) { return model.handle; },
  assertAlive(handle, label) {
    if (handle === 0) throw new Error(`${label} is freed`);
  },
});
expectSame(modelHandleHelpers.modelHandleForBind(null), 0, "model source null bind handle");
expectSame(modelHandleHelpers.modelHandleForBind({ family: "llama", handle: 7 }), 7, "model source bind handle");
expectSame(modelHandleHelpers.modelHandleForCompatibility({ handle: 8 }), 8, "model source compatibility handle");
expectThrow(
  () => modelHandleHelpers.modelHandleForBind({ family: "other", handle: 7 }),
  "model must be a zgml LLaMA model handle",
  "model source invalid bind model",
);

const freedModels = [];
const modelPolicy = modelSource.createModelFacadePolicyHelpers({
  nullModelHandle: 0,
  assertModelAlive(handle, label) {
    if (handle === 0) throw new Error(`${label} is freed`);
  },
  modelInspect(handle) { return { handle }; },
  modelCompile(handle, options) { return { handle, options }; },
  modelFree(handle) { freedModels.push(handle); },
});
const policyModel = { handle: 11 };
expectSame(modelPolicy.inspect(policyModel), { handle: 11 }, "model source policy inspect");
expectSame(modelPolicy.compile(policyModel, { backend: "cpu" }, (handle) => ({ program: handle })), {
  program: { handle: 11, options: { backend: "cpu" } },
}, "model source policy compile");
modelPolicy.free(policyModel);
expectSame({ handle: policyModel.handle, freedModels }, { handle: 0, freedModels: [11] }, "model source policy free");

const llamaFamily = modelSource.createLlamaModelFamilyFacadeHelpers({
  modelPolicy,
  createTinyLlamaModelHandle() { return 21; },
  loadModelPath(kind, path) { return { kind, path }; },
  loadSafetensorsDataHandle(kind, data) { return { kind, bytes: data.byteLength }; },
  probeModel(source, options) { return { source, options }; },
  probeSafetensorsHeader(header, options) { return { header: Array.from(modelSource.safetensorsHeaderBytes(header)), options }; },
});
expectSame(llamaFamily.createTinyLlama((handle) => ({ handle })), { handle: 21 }, "model source family create");
expectSame(llamaFamily.load(abi.modelKinds.tinyLlama, "x.gguf", (handle) => ({ handle })), {
  handle: { kind: abi.modelKinds.tinyLlama, path: "x.gguf" },
}, "model source family load");
expectSame(llamaFamily.loadSafetensorsData(abi.modelKinds.smollm135m, new Uint8Array(4), (handle) => ({ handle })), {
  handle: { kind: abi.modelKinds.smollm135m, bytes: 4 },
}, "model source family data load");
expectSame(llamaFamily.probe("tiny-llama", "x.gguf"), {
  source: "x.gguf",
  options: { kind: "tiny-llama" },
}, "model source family probe");
expectSame(llamaFamily.probeHeader(null, "hi"), {
  header: [104, 105],
  options: undefined,
}, "model source family probe header");

const irTrace = {
  inputShape: [1, 2],
  outputShape: [1, 3],
  parameterCount: 2,
  parameterScalarCount: 9,
  ops: [{
    index: 0,
    path: "0",
    op: "linear",
    inputShape: [1, 2],
    outputShape: [1, 3],
    inFeatures: 2,
    outFeatures: 3,
    bias: true,
    parameters: [
      { name: "0.weight", shape: [3, 2], scalarCount: 6 },
      { name: "0.bias", shape: [3], scalarCount: 3 },
    ],
  }],
};
const ir = tensorProgramIr.buildTensorProgramIrForTrace(irTrace);
expectSame(ir.kind, "tensor-program-ir", "tensor program ir kind");
expectSame(ir.inputShape, [1, 2], "tensor program ir input shape");
expectSame(ir.outputShape, [1, 3], "tensor program ir output shape");
expectSame(ir.valueCount, 4, "tensor program ir value count");
expectSame(ir.ops[0].inputValueIds, [0, 1, 2], "tensor program ir op input ids");
expectSame(ir.ops[0].parameterValueIds, [1, 2], "tensor program ir op parameter ids");
expectSame(ir.ops[0].attrs, { inFeatures: 2, outFeatures: 3, bias: true }, "tensor program ir attrs");
expectSame(ir.values[1].binding, "weights", "tensor program ir weight binding");
expectSame(ir.values[2].binding, "bias", "tensor program ir bias binding");
expectSame(ir.values[3].strides, [3, 1], "tensor program ir row-major strides");
expectSame(Object.isFrozen(ir), true, "tensor program ir frozen root");
expectSame(Object.isFrozen(ir.ops[0].attrs), true, "tensor program ir frozen attrs");
const irSignature = tensorProgramIr.tensorProgramIrSignature(ir);
expectSame(ir.signature, irSignature, "tensor program ir intrinsic signature");
expectSame(irSignature.startsWith("tensor-program-ir:1:1x2:1x3:2:3:0:3:4:1:2:9::"), true, "tensor program ir signature header");
expectSame(irSignature.includes("inFeatures=2,outFeatures=3"), true, "tensor program ir signature attrs sorted");
expectSame(tensorProgramIr.tensorProgramIrSignature(null), "", "tensor program ir empty signature");

const directCompileDiagnostic = compileDiagnostics.compileDiagnostic("shape-mismatch", "bad shape", { inputShape: [2, 3] });
expectSame(directCompileDiagnostic, {
  code: "shape-mismatch",
  message: "bad shape",
  inputShape: [2, 3],
}, "compile diagnostics direct diagnostic shape");
expectSame(Object.isFrozen(directCompileDiagnostic.inputShape), true, "compile diagnostics direct frozen shape");
const kernelizerDiagnostic = compileDiagnostics.diagnosticForKernelizerOp(
  { index: 7, op: "permute", path: "0", inputShape: [1, 2, 3], outputShape: [1, 3, 2] },
  "unsupported-view",
  "rank-3 permute unsupported",
);
expectSame(kernelizerDiagnostic.stage, "kernelizer", "compile diagnostics kernelizer stage");
expectSame(kernelizerDiagnostic.opIndex, 7, "compile diagnostics kernelizer op index");
expectSame(Object.isFrozen(compileDiagnostics.freezeCompileDiagnostics([{
  code: "unsupported-op",
  message: "not now",
  inputShape: [1],
}])[0].inputShape), true, "compile diagnostics freeze list shapes");
const traceDiagnostic = traceCompiler.compileDiagnostic("shape-mismatch", "bad shape", { inputShape: [2, 3] });
expectSame(traceDiagnostic, {
  code: "shape-mismatch",
  message: "bad shape",
  inputShape: [2, 3],
}, "trace compiler diagnostic shape");
expectSame(Object.isFrozen(traceDiagnostic.inputShape), true, "trace compiler diagnostic frozen shape");
const frozenTrace = traceCompiler.freezeSequentialTrace({
  inputShape: [1, 2],
  outputShape: [1, 3],
  ops: [{
    index: 0,
    op: "linear",
    path: "0",
    inputShape: [1, 2],
    outputShape: [1, 3],
    parameters: [{ name: "weight", shape: [3, 2] }],
  }],
});
expectSame(Object.isFrozen(frozenTrace.ops[0].parameters[0].shape), true, "trace compiler frozen parameter shape");
expectSame(traceCompiler.tensorProgramIrForTrace({ shapeKnown: false, ops: [] }).diagnostic.code, "missing-input-shape", "trace compiler missing shape diagnostic");
const irOnlyTrace = { ...irTrace, shapeKnown: true };
const traceIrResult = traceCompiler.tensorProgramIrForTrace(irOnlyTrace);
expectSame(traceIrResult.diagnostic, null, "trace compiler trace to ir diagnostic");
expectSame(traceIrResult.ir.valueCount, 4, "trace compiler trace to ir");
expectSame(traceIrResult.ir.signature, tensorProgramIr.tensorProgramIrSignature(traceIrResult.ir), "trace compiler ir intrinsic signature");
const traceArtifacts = traceCompiler.traceCompilerArtifacts(irOnlyTrace);
expectSame(traceArtifacts.diagnostic, null, "trace compiler artifacts diagnostic");
expectSame(traceArtifacts.ir.signature, traceArtifacts.irSignature ?? tensorProgramIr.tensorProgramIrSignature(traceArtifacts.ir), "trace compiler artifacts ir signature");
expectSame(traceArtifacts.kernelPlan.signature, kernelPlanPolicy.kernelPlanSignature(traceArtifacts.kernelPlan), "trace compiler kernel plan intrinsic signature");
expectSame(traceArtifacts.kernelPlan.memoryLayout.signature, kernelPlanPolicy.memoryLayoutSignature(traceArtifacts.kernelPlan.memoryLayout), "trace compiler memory layout intrinsic signature");
expectSame(traceArtifacts.kernelPlan.parameterLayout.signature, kernelPlanPolicy.parameterLayoutSignature(traceArtifacts.kernelPlan.parameterLayout), "trace compiler parameter layout intrinsic signature");
expectSame(traceArtifacts.kernelPlan.bufferLayout.signature, programBuffers.programBufferLayoutSignature(traceArtifacts.kernelPlan.bufferLayout), "trace compiler buffer layout intrinsic signature");
expectSame(kernelPlanPolicy.bufferLayoutSignature(traceArtifacts.kernelPlan.bufferLayout), compilerSignatures.compilerSignatureEvidence(traceArtifacts.ir, traceArtifacts.kernelPlan).bufferLayoutSignature, "trace compiler compact buffer layout signature evidence");
expectSame(traceArtifacts.kernelPlan.signature, compilerSignatures.compilerSignatureEvidence(traceArtifacts.ir, traceArtifacts.kernelPlan).kernelPlanSignature, "trace compiler kernel plan signature evidence");
const programCompileSpec = traceCompiler.compiledSequentialModuleSpecFromTrace([{ path: "0" }], irOnlyTrace, traceArtifacts);
const programCompileEvidence = traceCompiler.programCompileEvidenceFromCompiledSpec(programCompileSpec);
const nativeModuleRequirements = Object.freeze({
  kind: "zgml.program.requirements",
  signature: "module-native-requirements-smoke",
  modelKind: "module",
  scalarBytes: 4,
  tokenIdBytes: 4,
  inputLen: 2,
  inputByteLength: 8,
  outputLen: 3,
  weightsLen: 6,
  weightsByteLength: 24,
  biasLen: 3,
  biasByteLength: 12,
  parameterLen: 9,
  parameterByteLength: 36,
  logitsLen: 0,
  outputByteLength: 12,
  contextLen: 0,
  batch: 0,
  maxTokenWindow: 0,
});
const nativeRequirementsEvidence = moduleProgramEvidence.moduleProgramEvidenceWithNativeRequirements(programCompileEvidence, nativeModuleRequirements, programCompileSpec.desc);
expectSame(nativeRequirementsEvidence.nativeRequirements, nativeModuleRequirements, "module Program compile evidence retains native requirements");
expectSame(nativeRequirementsEvidence.nativeRequirementsSource, "zig-module-program", "module Program compile evidence records native requirements source");
expectSame(nativeRequirementsEvidence.inputLen, nativeModuleRequirements.inputLen, "module Program compile evidence uses native input len");
expectSame(nativeRequirementsEvidence.signature, compilerSignatures.programCompileEvidenceSignature(nativeRequirementsEvidence), "module Program native requirements evidence signature");
expectSame(compilerSignatures.isProgramCompileEvidence(nativeRequirementsEvidence), true, "module Program native requirements evidence predicate");
expectThrow(
  () => moduleProgramEvidence.moduleProgramEvidenceWithNativeRequirements(programCompileEvidence, Object.freeze({ ...nativeModuleRequirements, weightsLen: 7 }), programCompileSpec.desc),
  "module Program native requirements drift: weightsLen TS=6 Zig=7",
  "module Program native requirements reject TS/Zig length drift",
);
expectSame(Object.isFrozen(programCompileEvidence), true, "program compile evidence frozen");
expectSame(Object.isFrozen(programCompileEvidence.compilerSignatures), true, "program compile evidence signatures frozen");
expectSame(programCompileEvidence.signature, compilerSignatures.programCompileEvidenceSignature(programCompileEvidence), "program compile evidence signature");
expectSame(compilerSignatures.isProgramCompileEvidence(programCompileEvidence), true, "program compile evidence predicate");
expectSame(compilerSignatures.requireProgramCompileEvidence(programCompileEvidence).kernelPlan.signature, traceArtifacts.kernelPlan.signature, "program compile evidence require");
expectSame(compilerSignatures.assertProgramCompileEvidence(programCompileEvidence).ir.signature, traceArtifacts.ir.signature, "program compile evidence assert");
expectSame(compilerSignatures.assert_program_compile_evidence(programCompileEvidence).signature, programCompileEvidence.signature, "program compile evidence snake assert");
expectSame(compilerSignatures.matchesProgramCompileEvidenceSignature(programCompileEvidence, programCompileEvidence.signature), true, "program compile evidence signature match");
expectSame(compilerSignatures.matches_program_compile_evidence_signature(programCompileEvidence, "wrong"), false, "program compile evidence snake signature mismatch");
expectSame(compilerSignatures.isProgramCompileEvidence({ ...programCompileEvidence }), false, "program compile evidence rejects mutable evidence");
expectThrow(
  () => compilerSignatures.requireProgramCompileEvidence(Object.freeze({ ...programCompileEvidence, signature: "wrong" })),
  "expected frozen ProgramCompileEvidence",
  "program compile evidence rejects mismatched signature",
);
expectThrow(
  () => lazyNamespace.input([2]).linear(1).compile({ backend: "cpu" }),
  "LazyTensor.compile requires a native adapter Program compiler; use torch.compile.compile(lazyGraph) in Node/Bun or inspect compileSupport() in source-only runtimes",
  "source-only LazyTensor.compile rejects missing native compiler",
);
const permuteTrace = {
  kind: "sequential",
  normalized: true,
  shapeKnown: true,
  inputShape: [2, 3],
  outputShape: [3, 2],
  layerCount: 1,
  opCount: 1,
  parameterCount: 0,
  parameterScalarCount: 0,
  ops: [{
    index: 0,
    path: "0",
    op: "permute",
    dims: [1, 0],
    inputShape: [2, 3],
    outputShape: [3, 2],
    parameters: [],
  }],
};
const permuteArtifacts = traceCompiler.traceCompilerArtifacts(permuteTrace);
expectSame(permuteArtifacts.diagnostic, null, "trace compiler rank-2 permute lowers");
expectSame(permuteArtifacts.ir.ops[0].attrs, { dims: [1, 0] }, "trace compiler rank-2 permute ir attrs");
expectSame(permuteArtifacts.kernelPlan.ops[0].nativeKernels, ["transpose"], "trace compiler rank-2 permute kernel");
const rank3PermuteArtifacts = traceCompiler.traceCompilerArtifacts({
  ...permuteTrace,
  inputShape: [1, 2, 3],
  outputShape: [1, 3, 2],
  ops: [{
    ...permuteTrace.ops[0],
    dims: [0, 2, 1],
    inputShape: [1, 2, 3],
    outputShape: [1, 3, 2],
  }],
});
expectSame(rank3PermuteArtifacts.diagnostic, null, "trace compiler rank-3 single-axis permute lowers");
expectSame(rank3PermuteArtifacts.kernelPlan.ops[0].nativeKernels, ["transpose"], "trace compiler rank-3 single-axis permute kernel");
const rank3CyclePermuteArtifacts = traceCompiler.traceCompilerArtifacts({
  ...permuteTrace,
  inputShape: [1, 2, 3],
  outputShape: [2, 3, 1],
  ops: [{
    ...permuteTrace.ops[0],
    dims: [1, 2, 0],
    inputShape: [1, 2, 3],
    outputShape: [2, 3, 1],
  }],
});
expectSame({
  diagnostic: rank3CyclePermuteArtifacts.diagnostic,
  nativeKernels: rank3CyclePermuteArtifacts.kernelPlan.ops[0].nativeKernels,
}, {
  diagnostic: null,
  nativeKernels: ["transpose", "transpose"],
}, "trace compiler rank-3 cycle permute lowers through transpose chain");
const supportDetails = traceCompiler.supportDetailsWithTrace({ composable: true }, irOnlyTrace, traceIrResult.ir, null);
expectSame(supportDetails.inputShape, [1, 2], "trace compiler support input shape");
expectSame(Object.isFrozen(supportDetails.trace.ops[0]), true, "trace compiler support frozen trace op");
const unsupportedSupport = traceCompiler.sequentialUnsupportedSupportDetails(
  [{ path: "0" }],
  irOnlyTrace,
  traceCompiler.compileDiagnostic("unsupported-op", "nope"),
);
expectSame(unsupportedSupport.diagnostics[0].code, "unsupported-op", "trace compiler unsupported support diagnostic");
expectSame(Object.isFrozen(unsupportedSupport.diagnostics), true, "trace compiler unsupported diagnostics frozen");
const mismatchAnalysis = traceCompiler.shapeMismatchAnalysis([{ path: "0" }], [1, 2], new Error("wrong"));
expectSame(mismatchAnalysis.support.diagnostics[0], {
  code: "shape-mismatch",
  message: "wrong",
  stage: "trace",
  inputShape: [1, 2],
}, "trace compiler mismatch analysis");
expectSame(traceCompiler.programTraceFromCompileEvidence({ trace: frozenTrace }), frozenTrace, "trace compiler trace evidence");
expectSame(traceCompiler.programTensorProgramIrFromCompileEvidence({ ir }), ir, "trace compiler ir evidence");
expectSame(traceCompiler.programKernelPlanFromCompileEvidence({ kernelPlan: { tag: "plan" } }), { tag: "plan" }, "trace compiler kernel evidence");

expectSame(tensorPlacement.packedProgramWeightsLen({ inputLen: 2, outputLen: 3 }), 6, "tensor placement packed weights fallback");
expectSame(tensorPlacement.packedProgramBiasLen({ outputLen: 3 }), 3, "tensor placement packed bias fallback");
let clonedTensor = null;
const placementHelpers = tensorPlacement.createTensorPlacementHelpers({
  cloneTensor(tensor) {
    clonedTensor = { ...tensor, cloned: true };
    return clonedTensor;
  },
});
const placementTensor = { data: new Float32Array([1, 2]), shape: [2] };
expectSame(placementHelpers.dtype(), "f32", "tensor placement dtype");
expectSame(placementHelpers.device(), "cpu", "tensor placement device");
expectSame(placementHelpers.to(placementTensor, "cpu") === placementTensor, true, "tensor placement to cpu identity");
expectSame(placementHelpers.to(placementTensor, { copy: true }) === clonedTensor, true, "tensor placement to copy");
expectSame(placementHelpers.cpu(placementTensor) === placementTensor, true, "tensor placement cpu identity");
expectSame(placementHelpers.float32(placementTensor) === placementTensor, true, "tensor placement float32 identity");
expectSame(placementHelpers.typeAs(placementTensor, placementTensor) === placementTensor, true, "tensor placement typeAs identity");
expectSame(placementHelpers.type_as(placementTensor, placementTensor, { copy: true }) === clonedTensor, true, "tensor placement type_as copy");
const placementProgram = {
  bufferLayout() {
    return { input: { scalarType: "f32", elementCount: 2 } };
  },
  inputShape() {
    return [2];
  },
};
expectSame(placementHelpers.validateProgramPlacement(placementTensor, placementProgram, "input"), undefined, "tensor placement valid placement");
expectThrow(
  () => placementHelpers.validateProgramPlacement(placementTensor, placementProgram, "output"),
  "Tensor.place output is not a Program buffer slot",
  "tensor placement missing slot",
);
expectThrow(
  () => placementHelpers.to(placementTensor, "webgpu"),
  'Tensor.to device webgpu requires an explicit compiled Program; use Tensor.place(program, kind) or program.device("webgpu")',
  "tensor placement webgpu eager rejection",
);
expectThrow(
  () => placementHelpers.validateProgramPlacement({ data: new Float32Array([1, 2]), shape: [1, 2] }, placementProgram, "input"),
  "Tensor.place input shape must be [2], got [1, 2]",
  "tensor placement shape mismatch",
);

const nativeBuffer = { native: true, byteLength: 16, writes: [], freed: false };
const isNativeBuffer = (value) => Boolean(value && value.native);
const f32Smoke = (value) => value instanceof Float32Array ? value : Float32Array.from(value ?? []);
const prepareF32Smoke = (value) => ({ data: f32Smoke(value), shape: Array.isArray(value) ? [value.length] : [f32Smoke(value).length] });
const bindLayout = {
  weights: { elementCount: 4, byteLength: 16 },
  bias: { elementCount: 2, byteLength: 8 },
  input: { elementCount: 2, byteLength: 8 },
  output: { elementCount: 2, byteLength: 8 },
};
const zeroParameterBindLayout = {
  weights: { elementCount: 0, byteLength: 0 },
  bias: { elementCount: 0, byteLength: 0 },
  input: { elementCount: 2, byteLength: 8 },
  output: { elementCount: 2, byteLength: 8 },
};
const bindValidation = tensorPlacement.createProgramBindValidationHelpers({
  isNativeBuffer,
  nativeBufferByteLength(buffer) { return buffer.byteLength; },
  f32: f32Smoke,
  prepareF32: prepareF32Smoke,
});
expectSame(bindValidation.validateProgramBindParams(bindLayout, {
  weights: [1, 2, 3, 4],
  bias: [1, 2],
  input: [3, 4],
  output: [0, 0, 0],
}, { input: [2], output: [2] }), undefined, "tensor placement bind validation");
expectThrow(
  () => bindValidation.validateProgramBindParams(bindLayout, { weights: [1, 2] }),
  "weights length 2 does not match Program weights slot length 4",
  "tensor placement bind validation mismatch",
);
expectThrow(
  () => bindValidation.validateProgramBindSlot(bindLayout, "weights", { native: true, byteLength: 8 }),
  "weights is too small for Program weights slot: 8 < 16",
  "tensor placement native buffer too small",
);

const createdBuffers = [];
const bindProgram = {
  bufferLayout() { return bindLayout; },
  inputShape() { return [2]; },
  outputShape() { return [2]; },
  createBuffer(kind) {
    const buffer = {
      native: true,
      kind,
      byteLength: bindLayout[kind].byteLength,
      writes: [],
      free() { this.freed = true; },
      writeFloat32(data) { this.writes.push(Array.from(data)); },
    };
    createdBuffers.push(buffer);
    return buffer;
  },
};
const zeroParameterBindProgram = {
  ...bindProgram,
  bufferLayout() { return zeroParameterBindLayout; },
};
const bindPreparation = tensorPlacement.createProgramBindPreparationHelpers({
  isNativeBuffer,
  f32: f32Smoke,
  validateProgramBindParams: bindValidation.validateProgramBindParams,
  outputShape(value) { return value && value.shape ? value.shape : null; },
});
const hostPrepared = bindPreparation.prepareProgramBind(bindProgram, { inputLen: 2, outputLen: 2 }, {
  weights: [1, 2, 3, 4],
  bias: [5, 6],
  input: [7, 8],
  output: [0, 0],
});
expectSame(hostPrepared.kind, "host", "tensor placement host bind kind");
expectSame(hostPrepared.weights, [1, 2, 3, 4], "tensor placement host weights");
const zeroParameterHostPrepared = bindPreparation.prepareProgramBind(zeroParameterBindProgram, { inputLen: 2, outputLen: 2, weightsLen: 0, biasLen: 0 }, {
  input: [7, 8],
  output: [0, 0],
});
expectSame(zeroParameterHostPrepared.kind, "host", "tensor placement zero-parameter host bind kind");
expectSame(Array.from(zeroParameterHostPrepared.weights), [], "tensor placement zero-parameter host weights");
const explicitZeroParameterHostPrepared = bindPreparation.prepareProgramBind(zeroParameterBindProgram, { inputLen: 2, outputLen: 2, weightsLen: 0, biasLen: 0 }, {
  weights: [],
  input: [7, 8],
  output: [0, 0],
});
expectSame(Array.from(explicitZeroParameterHostPrepared.weights), [], "tensor placement explicit zero-parameter host weights");
const nativePrepared = bindPreparation.prepareProgramBind(bindProgram, { inputLen: 2, outputLen: 2, weightsLen: 4, biasLen: 2 }, {
  weights: [1, 2, 3, 4],
  bias: [5, 6],
  input: nativeBuffer,
  output: [0, 0],
});
expectSame(nativePrepared.kind, "native", "tensor placement native bind kind");
expectSame(nativePrepared.ownedBuffers.length, 2, "tensor placement owned upload buffers");
expectSame(nativePrepared.ownedBuffers[0].writes[0], [1, 2, 3, 4], "tensor placement owned weights upload");
bindPreparation.freeOwnedProgramBindBuffers(nativePrepared.ownedBuffers);
expectSame(nativePrepared.ownedBuffers.every((buffer) => buffer.freed), true, "tensor placement owned free");

const nativeBindFields = tensorPlacement.createProgramNativeBufferBindFieldHelpers({
  isNativeBuffer,
  requireNativeBuffer(buffer, label) { return { label, kind: buffer.kind ?? label }; },
  f32: f32Smoke,
});
expectSame(nativeBindFields.programNativeBufferBindFields({ inputLen: 2, outputLen: 2, weightsLen: 4, biasLen: 2 }, {
  weights: { ...nativeBuffer, kind: "weights" },
  bias: { ...nativeBuffer, kind: "bias" },
  input: { ...nativeBuffer, kind: "input" },
  output: { ...nativeBuffer, kind: "output" },
}), {
  weights: { label: "weights", kind: "weights" },
  weightsLen: 4,
  bias: { label: "bias", kind: "bias" },
  biasLen: 2,
  input: { label: "input", kind: "input" },
  inputLen: 2,
  output: { label: "output", kind: "output" },
  outputLen: 2,
}, "tensor placement native bind fields");
expectThrow(
  () => nativeBindFields.programNativeBufferBindFields({ inputLen: 2, outputLen: 2, weightsLen: 4 }, { weights: [1, 2, 3, 4] }),
  "weights must be a zgml NativeBuffer",
  "tensor placement native weights required",
);

expectSame(nativeBufferPolicy.readImportField({ a: 1 }, ["x", "a"]), 1, "native buffer read import field");
expectSame(nativeBufferPolicy.readImportString({ a: 1, b: "ok" }, ["a", "b"]), undefined, "native buffer read import string non-string");
expectSame(nativeBufferPolicy.normalizeWebGpuImportSource({
  getSource() { return { handle: 3 }; },
}, "getSource"), { handle: 3 }, "native buffer normalize import source");
expectSame(nativeBufferPolicy.programDeviceBufferImportInfo({
  deviceHandle: 10,
  handle: 11,
  byteOffset: 4,
  byteLength: 16,
}, {}), {
  kind: "zgml.native-buffer.device-buffer-import",
  signature: "native-buffer-device-buffer-import|placement=webgpu|placementId=3|device=10|buffer=11|offset=4|explicitBytes=16|bytes=16",
  placementName: "webgpu",
  placement: abi.backendIds.webgpu,
  deviceHandle: 10,
  bufferHandle: 11,
  byteOffset: 4,
  explicitByteLength: 16,
  byteLength: 16,
}, "native buffer device import info");
const nativeBufferProgramImportEvidence = nativeBufferPolicy.programDeviceBufferImportInfo({
  deviceHandle: 10,
  handle: 11,
  byteOffset: 4,
  byteLength: 16,
}, {});
expectSame(nativeBufferPolicy.isProgramDeviceBufferImportInfo(nativeBufferProgramImportEvidence), true, "native buffer ProgramDevice import predicate");
expectSame(nativeBufferPolicy.requireProgramDeviceBufferImportInfo(nativeBufferProgramImportEvidence).bufferHandle, 11, "native buffer ProgramDevice import require");
expectSame(nativeBufferPolicy.assertProgramDeviceBufferImportInfo(nativeBufferProgramImportEvidence).byteLength, 16, "native buffer ProgramDevice import assert");
expectSame(nativeBufferPolicy.assert_program_device_buffer_import_info(nativeBufferProgramImportEvidence).placementName, "webgpu", "native buffer ProgramDevice import snake assert");
expectSame(nativeBufferPolicy.matchesProgramDeviceBufferImportSignature(nativeBufferProgramImportEvidence, nativeBufferProgramImportEvidence.signature), true, "native buffer ProgramDevice import signature");
expectSame(nativeBufferPolicy.matches_program_device_buffer_import_signature(nativeBufferProgramImportEvidence, "wrong"), false, "native buffer ProgramDevice import signature mismatch");
expectSame(nativeBufferPolicy.isProgramDeviceBufferImportInfo({ ...nativeBufferProgramImportEvidence }), false, "native buffer ProgramDevice import rejects mutable evidence");
expectThrow(
  () => nativeBufferPolicy.requireProgramDeviceBufferImportInfo(Object.freeze({
    ...nativeBufferProgramImportEvidence,
    signature: "wrong",
  })),
  "expected frozen ProgramDevice buffer import evidence",
  "native buffer ProgramDevice import rejects mismatched signature",
);
expectSame(nativeBufferPolicy.nativeBufferDeviceImportOptions("webgpu", {
  storage: "external-resource",
  placement: "webgpu",
  handle: 12,
  byteOffset: 0,
  byteLength: 32,
}, 13), {
  kind: "zgml.native-buffer.device-import",
  signature: "native-buffer-device-import|placement=webgpu|device=13|buffer=12|offset=0|bytes=32",
  placement: "webgpu",
  deviceHandle: 13,
  bufferHandle: 12,
  byteOffset: 0,
  byteLength: 32,
}, "native buffer device import options");
const nativeBufferDeviceImportEvidence = nativeBufferPolicy.nativeBufferDeviceImportOptions("webgpu", {
  storage: "external-resource",
  placement: "webgpu",
  handle: 12,
  byteOffset: 0,
  byteLength: 32,
}, 13);
expectSame(nativeBufferPolicy.isNativeBufferDeviceImportInfo(nativeBufferDeviceImportEvidence), true, "native buffer device import predicate");
expectSame(nativeBufferPolicy.requireNativeBufferDeviceImportInfo(nativeBufferDeviceImportEvidence).bufferHandle, 12, "native buffer device import require");
expectSame(nativeBufferPolicy.assertNativeBufferDeviceImportInfo(nativeBufferDeviceImportEvidence).deviceHandle, 13, "native buffer device import assert");
expectSame(nativeBufferPolicy.assert_native_buffer_device_import_info(nativeBufferDeviceImportEvidence).byteLength, 32, "native buffer device import snake assert");
expectSame(nativeBufferPolicy.matchesNativeBufferDeviceImportSignature(nativeBufferDeviceImportEvidence, nativeBufferDeviceImportEvidence.signature), true, "native buffer device import signature");
expectSame(nativeBufferPolicy.matches_native_buffer_device_import_signature(nativeBufferDeviceImportEvidence, "wrong"), false, "native buffer device import signature mismatch");
expectSame(nativeBufferPolicy.isNativeBufferDeviceImportInfo({ ...nativeBufferDeviceImportEvidence }), false, "native buffer device import rejects mutable evidence");
expectSame(nativeBufferPolicy.externalResourceInfo({
  placement: "webgpu",
  access: ["read", "write"],
  handle: 99,
  byteOffset: 8,
  byteLength: 64,
}), {
  kind: "zgml.native-buffer.external-resource",
  signature: "native-buffer-external-resource|placement=webgpu|placementId=3|access=3|handle=99|offset=8|bytes=64",
  placementName: "webgpu",
  placement: abi.backendIds.webgpu,
  accessFlags: abi.resourceAccessIds.readwrite,
  handle: 99,
  byteOffset: 8,
  byteLength: 64,
}, "native buffer external resource info");
const nativeBufferExternalResourceEvidence = nativeBufferPolicy.externalResourceInfo({
  placement: "webgpu",
  access: ["read", "write"],
  handle: 99,
  byteOffset: 8,
  byteLength: 64,
});
expectSame(nativeBufferPolicy.isNativeBufferExternalResourceInfo(nativeBufferExternalResourceEvidence), true, "native buffer external resource predicate");
expectSame(nativeBufferPolicy.requireNativeBufferExternalResourceInfo(nativeBufferExternalResourceEvidence).handle, 99, "native buffer external resource require");
expectSame(nativeBufferPolicy.assertNativeBufferExternalResourceInfo(nativeBufferExternalResourceEvidence).byteOffset, 8, "native buffer external resource assert");
expectSame(nativeBufferPolicy.assert_native_buffer_external_resource_info(nativeBufferExternalResourceEvidence).byteLength, 64, "native buffer external resource snake assert");
expectSame(nativeBufferPolicy.matchesNativeBufferExternalResourceSignature(nativeBufferExternalResourceEvidence, nativeBufferExternalResourceEvidence.signature), true, "native buffer external resource signature");
expectSame(nativeBufferPolicy.matches_native_buffer_external_resource_signature(nativeBufferExternalResourceEvidence, "wrong"), false, "native buffer external resource signature mismatch");
expectSame(nativeBufferPolicy.isNativeBufferExternalResourceInfo({ ...nativeBufferExternalResourceEvidence }), false, "native buffer external resource rejects mutable evidence");
const nativeBufferWriteEvidence = nativeBufferPolicy.nativeBufferWriteInfo(16, 8, 4, "write");
expectSame(nativeBufferWriteEvidence, {
  kind: "zgml.native-buffer.byte-range",
  signature: "native-buffer-byte-range|op=write|element=bytes|offset=4|bytes=8|length=null",
  operation: "write",
  elementType: "bytes",
  byteOffset: 4,
  byteLength: 8,
  length: null,
}, "native buffer write info");
expectSame(Object.isFrozen(nativeBufferWriteEvidence), true, "native buffer write info is frozen evidence");
expectSame(nativeBufferPolicy.isNativeBufferByteRangeInfo(nativeBufferWriteEvidence), true, "native buffer byte range evidence predicate");
expectSame(nativeBufferPolicy.requireNativeBufferByteRangeInfo(nativeBufferWriteEvidence).byteLength, 8, "native buffer byte range require");
expectSame(nativeBufferPolicy.assertNativeBufferByteRangeInfo(nativeBufferWriteEvidence).byteOffset, 4, "native buffer byte range assert");
expectSame(nativeBufferPolicy.assert_native_buffer_byte_range_info(nativeBufferWriteEvidence).operation, "write", "native buffer byte range snake assert");
expectSame(nativeBufferPolicy.matchesNativeBufferByteRangeSignature(nativeBufferWriteEvidence, nativeBufferWriteEvidence.signature), true, "native buffer byte range signature");
expectSame(nativeBufferPolicy.matches_native_buffer_byte_range_signature(nativeBufferWriteEvidence, "wrong"), false, "native buffer byte range snake signature mismatch");
expectSame(nativeBufferPolicy.isNativeBufferByteRangeInfo({
  ...nativeBufferWriteEvidence,
}), false, "native buffer byte range rejects mutable evidence");
expectThrow(
  () => nativeBufferPolicy.requireNativeBufferByteRangeInfo({
    ...nativeBufferWriteEvidence,
    signature: "native-buffer-byte-range|op=write|element=bytes|offset=4|bytes=7|length=null",
  }),
  "expected frozen NativeBuffer byte-range evidence",
  "native buffer byte range rejects mismatched signature",
);
expectSame(nativeBufferPolicy.nativeBufferReadFloat32Target(16, undefined, 4).length, 3, "native buffer read f32 target");
expectSame(nativeBufferPolicy.nativeBufferReadBytesTarget(16, undefined, 4).byteLength, 12, "native buffer read bytes target");
expectSame(nativeBufferPolicy.nativeBufferReadFloat32IntoInfo(16, new Float32Array(2), 2, 4), {
  kind: "zgml.native-buffer.byte-range",
  signature: "native-buffer-byte-range|op=read-float32-into|element=f32|offset=4|bytes=8|length=2",
  operation: "read-float32-into",
  elementType: "f32",
  byteOffset: 4,
  byteLength: 8,
  length: 2,
}, "native buffer read f32 into info");
expectSame(nativeBufferPolicy.nativeBufferReadBytesIntoInfo(16, new Uint8Array(4), 4, 8), {
  kind: "zgml.native-buffer.byte-range",
  signature: "native-buffer-byte-range|op=read-bytes-into|element=bytes|offset=8|bytes=4|length=4",
  operation: "read-bytes-into",
  elementType: "bytes",
  byteOffset: 8,
  byteLength: 4,
  length: 4,
}, "native buffer read bytes into info");
expectThrow(
  () => nativeBufferPolicy.nativeBufferWriteInfo(4, 8, 0, "write"),
  "write byte range 0..8 exceeds buffer byteLength 4",
  "native buffer write range rejection",
);
expectThrow(
  () => nativeBufferPolicy.nativeBufferReadFloat32Target(16, 1, 2),
  "NativeBuffer.readFloat32 byteOffset must be 4-byte aligned, got 2",
  "native buffer read f32 alignment rejection",
);

let nextNativeHandle = 200;
const nativeWrites = [];
const nativeReads = [];
const nativeFreed = [];
class NativeBufferSmoke {
  constructor(handle, byteLength, options = {}) {
    nativeFacade.initialize(this, handle, byteLength, options);
  }
  assertAlive() { return nativeFacade.assertAlive(this); }
}
const nativeFacade = nativeBufferPolicy.createNativeBufferFacadeHelpers({
  NativeBuffer: NativeBufferSmoke,
  nullHandle: 0,
  f32: f32Smoke,
  byteView(value) { return value instanceof Uint8Array ? value : new Uint8Array(value); },
  isLiveHandle(handle) { return handle !== 0 && handle != null; },
  createBuffer(byteLength) { return nextNativeHandle += byteLength; },
  wrapBytes(data) { return nextNativeHandle += data.byteLength; },
  wrapExternalResource(resource) { return resource.handle; },
  bufferSize(handle) { return handle === 216 ? 16 : 8; },
  inspectBuffer(handle) {
    return { storage: "external-resource", placement: "webgpu", handle, byteOffset: 0, byteLength: 16 };
  },
  writeBuffer(handle, byteOffset, data, byteLength) {
    nativeWrites.push({ handle, byteOffset, data: Array.from(data), byteLength });
  },
  readBuffer(handle, byteOffset, target, byteLength) {
    nativeReads.push({ handle, byteOffset, byteLength });
    if (target instanceof Float32Array) target.set(Float32Array.from([1, 2]).subarray(0, target.length));
    else target.set(Uint8Array.from([3, 4, 5, 6]).subarray(0, target.byteLength));
  },
  freeBuffer(handle) { nativeFreed.push(handle); },
  programDeviceHandle() { return 77; },
});
const createdNative = nativeFacade.create(16);
expectSame({ handle: createdNative.handle, byteLength: createdNative.byteLength }, { handle: 216, byteLength: 16 }, "native buffer facade create");
nativeFacade.writeFloat32(createdNative, [1, 2], 0);
expectSame(nativeWrites[0], { handle: 216, byteOffset: 0, data: [1, 2], byteLength: 8 }, "native buffer facade write f32");
expectSame(nativeFacade.readFloat32(createdNative, 2, 0), [1, 2], "native buffer facade read f32");
expectSame(nativeFacade.readBytes(createdNative, 4, 0), [3, 4, 5, 6], "native buffer facade read bytes");
createdNative.devicePlacement = "webgpu";
expectSame(nativeFacade.deviceImportOptions(createdNative, 1), {
  kind: "zgml.native-buffer.device-import",
  signature: "native-buffer-device-import|placement=webgpu|device=77|buffer=216|offset=0|bytes=16",
  placement: "webgpu",
  deviceHandle: 77,
  bufferHandle: 216,
  byteOffset: 0,
  byteLength: 16,
}, "native buffer facade device import");
nativeFacade.free(createdNative);
expectSame({ handle: createdNative.handle, byteLength: createdNative.byteLength, nativeFreed }, { handle: 0, byteLength: 0, nativeFreed: [216] }, "native buffer facade free");
expectThrow(
  () => nativeFacade.assertAlive(createdNative),
  "buffer was already freed",
  "native buffer facade freed rejection",
);
const lifetime = nativeBufferPolicy.createNativeBufferLifetimeHelpers({
  isNativeBuffer(value) { return value && value.isNative; },
});
const liveA = { isNative: true, assertAliveCount: 0, assertAlive() { this.assertAliveCount += 1; } };
const liveB = { isNative: true, assertAliveCount: 0, assertAlive() { this.assertAliveCount += 1; } };
expectSame(lifetime.requireNativeBuffer(liveA, "weights"), liveA, "native buffer lifetime require");
expectSame(lifetime.uniqueNativeBuffers([liveA, liveA, {}, liveB]), [liveA, liveB], "native buffer lifetime unique");
expectSame({ a: liveA.assertAliveCount, b: liveB.assertAliveCount }, { a: 2, b: 1 }, "native buffer lifetime assert counts");

const modeLeaf = { training: true };
const trainCalls = [];
const trainableLeaf = { train(mode) { trainCalls.push(mode); this.training = mode; } };
const modeParent = { layers: [modeLeaf, trainableLeaf] };
moduleMode.initializeModuleMode(modeParent);
expectSame(modeParent.training, true, "module mode initialize");
expectSame(moduleMode.setModuleTraining(modeParent, false), modeParent, "module mode set returns module");
expectSame(modeParent.training, false, "module mode set false");
moduleMode.setChildModuleTraining(modeParent, true);
expectSame({ parent: modeParent.training, leaf: modeLeaf.training, calls: trainCalls }, {
  parent: true,
  leaf: true,
  calls: [true],
}, "module mode child propagation");
expectSame(moduleMode.normalizeModuleTrainingMode(false), false, "module mode normalize false");
expectThrow(
  () => moduleMode.normalizeModuleTrainingMode("yes"),
  "module train mode must be boolean, got string",
  "module mode rejects non-boolean",
);

class ShapeModuleSmokeTensor {
  constructor(data, shape = [data.length]) {
    this.data = data instanceof Float32Array ? data : Float32Array.from(data);
    this.shape = shape.slice();
  }
  reshape(shape) { return { op: "reshape", shape: shape.slice() }; }
  view(shape) { return { op: "view", shape: shape.slice() }; }
  flatten(startDim, endDim) { return { op: "flatten", startDim, endDim }; }
  squeeze(dim) { return { op: "squeeze", dim: dim ?? null }; }
  unsqueeze(dim) { return { op: "unsqueeze", dim }; }
  broadcastTo(shape) { return { op: "broadcastTo", shape: shape.slice() }; }
  expand(shape) { return { op: "expand", shape: shape.slice() }; }
  narrow(dim, start, length) { return { op: "narrow", dim, start, length }; }
  select(dim, index) { return { op: "select", dim, index }; }
  slice(dim, start, end, step) { return { op: "slice", dim, start, end, step }; }
  transpose(dim0, dim1) { return { op: "transpose", dim0, dim1 }; }
  permute(dims) { return { op: "permute", dims: dims.slice() }; }
}
const shapeModulePlacements = [];
const shapeModulePrograms = [];
const ShapeModule = nnShapeModule.createShapeModuleClass({
  Tensor: ShapeModuleSmokeTensor,
  f32: (value) => value instanceof Float32Array ? value : Float32Array.from(Array.isArray(value) ? value.flat(Infinity) : [value]),
  stateKeys: (source) => Object.keys(source ?? {}),
  setRequiresGrad: (module, requiresGrad) => {
    module.requiresGrad = Boolean(requiresGrad);
    return module;
  },
  analyzeSingleModuleProgram: (module) => ({
    supported: module.kind !== "slice",
    reason: module.kind === "slice" ? "slice unsupported" : null,
    support: { nativePath: "device-program" },
    compiled: module.kind === "slice" ? null : { kind: "module", moduleKind: module.kind },
  }),
  moduleCompileSupport: (supported, reason, support) => ({ supported, reason, ...support }),
  compileModuleProgram: (compiled, options) => {
    shapeModulePrograms.push({ compiled, options });
    return { programKind: compiled.moduleKind };
  },
  packedSequentialProgramParameters: () => ({ weights: new Float32Array(0) }),
  placeModuleParameterBindings: (module, program, options) => {
    shapeModulePlacements.push({ kind: module.kind, program, options });
    return { placed: module.kind };
  },
});
const reshapeModule = new ShapeModule("reshape", [3, -1]);
expectSame({ kind: reshapeModule.kind, shape: reshapeModule.shape, training: reshapeModule.training }, {
  kind: "reshape",
  shape: [3, -1],
  training: true,
}, "nn shape module reshape initializes");
expectSame(reshapeModule.forward(new ShapeModuleSmokeTensor(Float32Array.of(1, 2, 3, 4, 5, 6), [2, 3])), {
  op: "reshape",
  shape: [3, -1],
}, "nn shape module tensor forward");
expectSame(new ShapeModule("transpose", -1, 0).forward([1, 2, 3]), {
  op: "transpose",
  dim0: -1,
  dim1: 0,
}, "nn shape module host forward");
expectSame(new ShapeModule("permute", [1, 0]).forward([1, 2, 3]), {
  op: "permute",
  dims: [1, 0],
}, "nn shape module permute host forward");
expectSame(new ShapeModule("squeeze").forward(new ShapeModuleSmokeTensor(Float32Array.of(1), [1])), {
  op: "squeeze",
  dim: null,
}, "nn shape module squeeze-all forward");
expectSame(reshapeModule.parameters(), [], "nn shape module parameters");
expectSame(reshapeModule.namedModules(), [{ name: "", module: reshapeModule }], "nn shape module namedModules");
expectSame(reshapeModule.requiresGrad_(false), reshapeModule, "nn shape module requiresGrad returns self");
expectSame(reshapeModule.requiresGrad, false, "nn shape module requiresGrad hook");
expectSame(reshapeModule.eval().training, false, "nn shape module eval");
expectSame(reshapeModule.train(true).training, true, "nn shape module train");
expectSame(reshapeModule.loadStateDict({}, { strict: true }), reshapeModule, "nn shape module load empty state");
expectThrow(
  () => reshapeModule.loadStateDict({ weight: Float32Array.of(1) }),
  "reshape has no parameters",
  "nn shape module rejects state",
);
expectSame(reshapeModule.compileSupport(), { supported: true, reason: null, nativePath: "device-program" }, "nn shape module compile support");
expectSame(new ShapeModule("slice", 0, 0, 1).compileSupport(), { supported: false, reason: "slice unsupported", nativePath: "device-program" }, "nn shape module unsupported compile support");
expectSame(reshapeModule.bindParameters().weights, [], "nn shape module bind parameters");
expectSame(reshapeModule.placeParameters({ name: "program" }, { label: "x" }), { placed: "reshape" }, "nn shape module place parameters");
expectSame(shapeModulePlacements, [{ kind: "reshape", program: { name: "program" }, options: { label: "x" } }], "nn shape module placement hook");
expectSame(reshapeModule.compile({ inputShape: [2, 3] }), { programKind: "reshape" }, "nn shape module compile");
expectSame(shapeModulePrograms, [{ compiled: { kind: "module", moduleKind: "reshape" }, options: { inputShape: [2, 3] } }], "nn shape module compile hook");
expectThrow(
  () => new ShapeModule("narrow", 0, 0, 0),
  "narrow length must be a positive safe integer, got 0",
  "nn shape module rejects narrow length",
);
expectThrow(
  () => new ShapeModule("unknown"),
  "unsupported shape module kind: unknown",
  "nn shape module rejects unknown kind",
);

class ParameterlessModuleSmokeTensor {
  constructor(data, shape = [data.length], options = {}) {
    this.data = data instanceof Float32Array ? data : Float32Array.from(data);
    this.shape = shape.slice();
    this.length = this.data.length;
    this.requiresGrad = Boolean(options.requiresGrad);
    this.grad = options.grad ?? (this.requiresGrad ? new Float32Array(this.length) : null);
    this._prev = options.prev ?? [];
    this._backward = options.backward ?? (() => {});
  }
  gelu() { return { op: "gelu" }; }
  relu() { return { op: "relu" }; }
  silu() { return { op: "silu" }; }
  sigmoid() { return { op: "sigmoid" }; }
  exp() { return { op: "exp" }; }
  log() { return { op: "log" }; }
  neg() { return { op: "neg" }; }
  recip() { return { op: "recip" }; }
  abs() { return { op: "abs" }; }
  sgn() { return { op: "sgn" }; }
  step() { return { op: "step" }; }
  sqrt() { return { op: "sqrt" }; }
  square() { return { op: "square" }; }
  softmaxDim(dim) { return { op: "softmaxDim", dim }; }
  logSoftmaxDim(dim) { return { op: "logSoftmaxDim", dim }; }
  sumDim(dim) { return { op: "sumDim", dim }; }
  meanDim(dim) { return { op: "meanDim", dim }; }
  prodDim(dim) { return { op: "prodDim", dim }; }
  maxDim(dim) { return { op: "maxDim", dim }; }
  minDim(dim) { return { op: "minDim", dim }; }
  argmaxDim(dim) { return { op: "argmaxDim", dim }; }
  argminDim(dim) { return { op: "argminDim", dim }; }
}
const parameterlessPlacements = [];
const parameterlessPrograms = [];
const parameterlessHooks = {
  Tensor: ParameterlessModuleSmokeTensor,
  f32: (value) => value instanceof Float32Array ? value : Float32Array.from(Array.isArray(value) ? value.flat(Infinity) : [value]),
  stateKeys: (source) => Object.keys(source ?? {}),
  setRequiresGrad: (module, requiresGrad) => {
    module.requiresGrad = Boolean(requiresGrad);
    return module;
  },
  analyzeSingleModuleProgram: (module) => ({
    supported: module.kind !== "unsupportedCompile",
    reason: module.kind === "unsupportedCompile" ? "unsupported compile" : null,
    support: { nativePath: "device-program" },
    compiled: module.kind === "unsupportedCompile" ? null : { kind: "module", moduleKind: module.kind },
  }),
  moduleCompileSupport: (supported, reason, support) => ({ supported, reason, ...support }),
  compileModuleProgram: (compiled, options) => {
    parameterlessPrograms.push({ compiled, options });
    return { programKind: compiled.moduleKind };
  },
  placeModuleParameterBindings: (module, program, options) => {
    parameterlessPlacements.push({ kind: module.kind, program, options });
    return { placed: module.kind };
  },
  addTensorGrad: (tensor, grad) => {
    if (!tensor.grad) tensor.grad = new Float32Array(tensor.length);
    for (let i = 0; i < grad.length; i += 1) tensor.grad[i] += grad[i];
  },
  isGradEnabled: () => true,
};
const ActivationModule = nnParameterlessModules.createActivationModuleClass(parameterlessHooks);
const SoftmaxModule = nnParameterlessModules.createSoftmaxModuleClass({
  ...parameterlessHooks,
  kind: "softmax",
  tensorMethod: "softmaxDim",
});
const LogSoftmaxModule = nnParameterlessModules.createSoftmaxModuleClass({
  ...parameterlessHooks,
  kind: "logSoftmax",
  tensorMethod: "logSoftmaxDim",
  parameterLabel: "logSoftmax",
});
const ReductionModule = nnParameterlessModules.createReductionModuleClass(parameterlessHooks);
const DropoutModule = nnParameterlessModules.createDropoutModuleClass(parameterlessHooks);
const AvgPool2dModule = nnPoolingModule.createAvgPool2dModuleClass(parameterlessHooks);
const avgPool2d = new AvgPool2dModule(2);
const avgPoolInput = new ParameterlessModuleSmokeTensor(Float32Array.of(1, 2, 3, 4), [1, 2, 2], { requiresGrad: true });
const avgPoolOut = avgPool2d.forward(avgPoolInput);
expectSame({ kind: avgPool2d.kind, kernelSize: avgPool2d.kernelSize, stride: avgPool2d.stride, data: avgPoolOut.data, shape: avgPoolOut.shape }, {
  kind: "avgPool2d",
  kernelSize: [2, 2],
  stride: [2, 2],
  data: [2.5],
  shape: [1, 1, 1],
}, "nn avgPool2d tensor forward");
avgPoolOut._backward(Float32Array.of(4));
expectSame(avgPoolInput.grad, [1, 1, 1, 1], "nn avgPool2d tensor grad");
expectSame(new AvgPool2dModule(2, { padding: 1, stride: 1, count_include_pad: false }).forward(new ParameterlessModuleSmokeTensor(Float32Array.of(4), [1, 1, 1])).data, [4, 4, 4, 4], "nn avgPool2d excludes padding from divisor");
expectSame(avgPool2d.compileSupport(), { supported: true, reason: null, nativePath: "device-program" }, "nn avgPool2d compile support is honest");
expectSame(avgPool2d.compile({ inputShape: [1, 2, 2] }), { programKind: "avgPool2d" }, "nn avgPool2d direct compile");
const reluModule = new ActivationModule("relu", (value) => Math.max(0, value));
expectSame({ kind: reluModule.kind, training: reluModule.training }, { kind: "relu", training: true }, "nn activation initializes");
expectSame(reluModule.forward(Float32Array.of(-1, 2)), [0, 2], "nn activation host forward");
expectSame(reluModule.forward(new ParameterlessModuleSmokeTensor(Float32Array.of(-1, 2))), { op: "relu" }, "nn activation tensor forward");
expectSame(reluModule.parameters(), [], "nn activation parameters");
expectSame(reluModule.requiresGrad_(false), reluModule, "nn activation requiresGrad returns self");
expectSame(reluModule.requiresGrad, false, "nn activation requiresGrad hook");
expectSame(reluModule.eval().training, false, "nn activation eval");
expectSame(reluModule.train(true).training, true, "nn activation train");
expectSame(reluModule.loadStateDict({}, { strict: true }), reluModule, "nn activation load empty state");
expectThrow(
  () => reluModule.loadStateDict({ weight: Float32Array.of(1) }),
  "relu has no parameters",
  "nn activation rejects state",
);
expectSame(reluModule.compileSupport(), { supported: true, reason: null, nativePath: "device-program" }, "nn activation compile support");
expectSame(reluModule.bindParameters().weights, [], "nn activation bind parameters");
expectSame(reluModule.placeParameters({ name: "program" }, { label: "activation" }), { placed: "relu" }, "nn activation place parameters");
expectSame(reluModule.compile({ inputShape: [2] }), { programKind: "relu" }, "nn activation compile");
expectThrow(
  () => new ActivationModule("unsupportedActivation", (value) => value).forward(new ParameterlessModuleSmokeTensor(Float32Array.of(1))),
  "unsupported activation module kind: unsupportedActivation",
  "nn activation rejects unsupported tensor kind",
);
const softmaxModule = new SoftmaxModule(1);
expectSame(softmaxModule.forward(new ParameterlessModuleSmokeTensor(Float32Array.of(1, 2), [1, 2])), { op: "softmaxDim", dim: 1 }, "nn softmax tensor forward");
expectSame(softmaxModule.forward([1, 2]), { op: "softmaxDim", dim: 1 }, "nn softmax host forward");
expectThrow(
  () => softmaxModule.loadStateDict({ weight: Float32Array.of(1) }),
  "softmax has no parameters",
  "nn softmax rejects state",
);
const logSoftmaxModule = new LogSoftmaxModule(-1);
expectSame(logSoftmaxModule.forward(new ParameterlessModuleSmokeTensor(Float32Array.of(1, 2), [1, 2])), { op: "logSoftmaxDim", dim: -1 }, "nn logSoftmax tensor forward");
expectThrow(
  () => logSoftmaxModule.loadStateDict({ weight: Float32Array.of(1) }),
  "logSoftmax has no parameters",
  "nn logSoftmax rejects state",
);
expectThrow(
  () => nnParameterlessModules.createSoftmaxModuleClass({ ...parameterlessHooks, kind: "bad", tensorMethod: "softmaxDim" }),
  "SoftmaxModule factory requires kind and tensorMethod",
  "nn softmax factory rejects bad kind",
);
const sumModule = new ReductionModule("sum", 0);
expectSame(sumModule.forward(new ParameterlessModuleSmokeTensor(Float32Array.of(1, 2), [2])), { op: "sumDim", dim: 0 }, "nn reduction tensor forward");
expectSame(new ReductionModule("mean", -1).forward([1, 2]), { op: "meanDim", dim: -1 }, "nn reduction host forward");
expectSame(new ReductionModule("prod", 1).forward(new ParameterlessModuleSmokeTensor(Float32Array.of(1, 2), [1, 2])), { op: "prodDim", dim: 1 }, "nn reduction prod forward");
expectSame(new ReductionModule("max", 1).forward(new ParameterlessModuleSmokeTensor(Float32Array.of(1, 2), [1, 2])), { op: "maxDim", dim: 1 }, "nn reduction max forward");
expectSame(new ReductionModule("min", 0).forward(new ParameterlessModuleSmokeTensor(Float32Array.of(1, 2), [1, 2])), { op: "minDim", dim: 0 }, "nn reduction min forward");
expectSame(new ReductionModule("argmax", 0).forward(new ParameterlessModuleSmokeTensor(Float32Array.of(1, 2), [1, 2])), { op: "argmaxDim", dim: 0 }, "nn reduction argmax forward");
expectSame(new ReductionModule("argmin", 0).forward(new ParameterlessModuleSmokeTensor(Float32Array.of(1, 2), [1, 2])), { op: "argminDim", dim: 0 }, "nn reduction argmin forward");
expectThrow(
  () => new ReductionModule("median", 0),
  "unsupported reduction module kind: median",
  "nn reduction rejects unsupported kind",
);
expectSame(parameterlessPlacements, [{ kind: "relu", program: { name: "program" }, options: { label: "activation" } }], "nn parameterless placement hook");
expectSame(parameterlessPrograms, [
  { compiled: { kind: "module", moduleKind: "avgPool2d" }, options: { inputShape: [1, 2, 2] } },
  { compiled: { kind: "module", moduleKind: "relu" }, options: { inputShape: [2] } },
], "nn parameterless compile hook");
const dropoutRngValues = [0.25, 0.75, 0.9];
const dropoutModule = new DropoutModule(0.5, { rng: () => dropoutRngValues.shift() });
expectSame({ kind: dropoutModule.kind, p: dropoutModule.p, training: dropoutModule.training }, {
  kind: "dropout",
  p: 0.5,
  training: true,
}, "nn dropout initializes");
expectSame(dropoutModule.forward(Float32Array.of(1, 2, 3)), [0, 4, 6], "nn dropout host train forward");
dropoutModule.eval();
expectSame(dropoutModule.forward(Float32Array.of(1, 2)), [1, 2], "nn dropout host eval forward");
const dropoutEvalTensor = new ParameterlessModuleSmokeTensor(Float32Array.of(3, 4), [2], { requiresGrad: true });
expectSame(dropoutModule.forward(dropoutEvalTensor), dropoutEvalTensor, "nn dropout tensor eval identity");
const dropoutTensorRng = [0.25, 0.75];
const dropoutTensorModule = new DropoutModule(0.5, { rng: () => dropoutTensorRng.shift() });
const dropoutTensor = new ParameterlessModuleSmokeTensor(Float32Array.of(2, 4), [2], { requiresGrad: true });
const dropoutOut = dropoutTensorModule.forward(dropoutTensor);
expectSame({ data: dropoutOut.data, shape: dropoutOut.shape, requiresGrad: dropoutOut.requiresGrad, prev: dropoutOut._prev }, {
  data: [0, 8],
  shape: [2],
  requiresGrad: true,
  prev: [dropoutTensor],
}, "nn dropout tensor train forward");
dropoutOut._backward(Float32Array.of(1, 1));
expectSame(dropoutTensor.grad, [0, 2], "nn dropout tensor grad");
const dropoutNoGradTensor = new ParameterlessModuleSmokeTensor(Float32Array.of(2), [1], { requiresGrad: true });
const dropoutNoGradModule = nnParameterlessModules.createDropoutModuleClass({ ...parameterlessHooks, isGradEnabled: () => false });
expectSame(new dropoutNoGradModule(0.5, { rng: () => 0.75 }).forward(dropoutNoGradTensor).requiresGrad, false, "nn dropout respects grad mode");
expectSame(new DropoutModule(1, { rng: () => 0.75 }).forward(Float32Array.of(1, 2)), [0, 0], "nn dropout p one");
expectSame(new DropoutModule(0, { rng: () => 0 }).forward(Float32Array.of(1, 2)), [1, 2], "nn dropout p zero");
expectSame(new DropoutModule(0.5, { training: false }).training, false, "nn dropout constructor training false");
expectThrow(
  () => new DropoutModule(-0.1),
  "dropout probability must be between 0 and 1, got -0.1",
  "nn dropout rejects probability",
);
expectThrow(
  () => new DropoutModule(0.5, { rng: 1 }),
  "dropout rng must be a function",
  "nn dropout rejects rng",
);
expectThrow(
  () => new DropoutModule(0.5, { training: "yes" }),
  "module train mode must be boolean, got string",
  "nn dropout rejects non boolean training",
);
const unsupportedDropoutModule = nnParameterlessModules.createDropoutModuleClass({
  ...parameterlessHooks,
  analyzeSingleModuleProgram: () => ({
    supported: false,
    reason: "dropout unsupported in training mode",
    support: { nativePath: "device-program" },
    compiled: null,
  }),
});
expectThrow(
  () => new unsupportedDropoutModule().bindParameters(),
  "dropout unsupported in training mode",
  "nn dropout bind rejects unsupported compile",
);
expectThrow(
  () => new unsupportedDropoutModule().compile(),
  "dropout unsupported in training mode",
  "nn dropout compile rejects unsupported compile",
);

class ModuleStateSmokeTensor {
  constructor(data, shape, options = {}) {
    this.data = data;
    this.shape = shape.slice();
    this.length = data.length;
    this.grad = options.grad ?? null;
    this.requiresGrad = Boolean(options.requiresGrad);
  }
  zeroGrad() {
    if (this.grad) this.grad.fill(0);
  }
  set requires_grad(value) {
    this.requiresGrad = Boolean(value);
  }
  get requires_grad() {
    return this.requiresGrad;
  }
}
const moduleStateHelpers = moduleState.createModuleStateHelpers({
  Tensor: ModuleStateSmokeTensor,
  f32WithLength(value, length, label) {
    const data = value instanceof Float32Array ? value : Float32Array.from(value);
    if (data.length !== length) throw new Error(`${label} length must be ${length}, got ${data.length}`);
    return data;
  },
});
const stateParam = moduleStateHelpers.makeParameter("weight", Float32Array.of(1, 2), [2]);
expectSame({
  name: stateParam.name,
  grad: stateParam.grad,
  tensorShape: stateParam.tensor.shape,
  requiresGrad: stateParam.tensor.requiresGrad,
}, {
  name: "weight",
  grad: [0, 0],
  tensorShape: [2],
  requiresGrad: true,
}, "module state make parameter");
expectSame(moduleStateHelpers.parameterView("head", stateParam).name, "head.weight", "module state parameter view");
expectSame(moduleStateHelpers.parameterNames([stateParam], "head"), ["head.weight"], "module state parameter names");
expectSame(moduleStateHelpers.parameterInfos([stateParam])[0], {
  name: "weight",
  index: 0,
  scalarCount: 2,
  shape: [2],
  layout: "row-major",
  requiresGrad: true,
  requires_grad: true,
}, "module state parameter infos");
expectSame(moduleStateHelpers.parameterInfo([stateParam], "weight").index, 0, "module state parameterInfo by name");
const stateSnapshot = moduleStateHelpers.stateDict([stateParam], "head");
expectSame(Object.isFrozen(stateSnapshot["head.weight"].shape), true, "module state frozen snapshot shape");
expectSame(stateSnapshot["head.weight"].signature.startsWith("module-state-entry|name=null|shape=2|layout=row-major|"), true, "module state snapshot signature");
expectSame(stateSnapshot["head.weight"].data, [1, 2], "module state snapshot data");
moduleStateHelpers.loadStateDict([stateParam], { "head.weight": { shape: [2], layout: "row-major", data: [3, 4] } }, { prefix: "head", validateOnly: true });
expectSame(stateParam.data, [1, 2], "module state validateOnly does not mutate");
moduleStateHelpers.loadStateDict([stateParam], { "head.weight": { shape: [2], layout: "row-major", data: [3, 4] } }, { prefix: "head" });
expectSame(stateParam.data, [3, 4], "module state load mutates");
stateParam.grad.set([5, 6]);
moduleStateHelpers.zeroGrad([stateParam]);
expectSame(stateParam.grad, [0, 0], "module state zeroGrad");
moduleStateHelpers.setRequiresGrad([stateParam], false);
expectSame(stateParam.tensor.requiresGrad, false, "module state setRequiresGrad false");
class ModuleStateNoAliasTensor {
  constructor(data, shape, options = {}) {
    this.data = data;
    this.shape = shape.slice();
    this.length = data.length;
    this.grad = options.grad ?? null;
    this.requiresGrad = Boolean(options.requiresGrad);
  }
  zeroGrad() {
    if (this.grad) this.grad.fill(0);
  }
}
const moduleStateNoAliasHelpers = moduleState.createModuleStateHelpers({
  Tensor: ModuleStateNoAliasTensor,
  f32WithLength(value, length, label) {
    const data = value instanceof Float32Array ? value : Float32Array.from(value);
    if (data.length !== length) throw new Error(`${label} length must be ${length}, got ${data.length}`);
    return data;
  },
});
const noAliasParam = moduleStateNoAliasHelpers.makeParameter("weight", Float32Array.of(1), [1]);
moduleStateNoAliasHelpers.setRequiresGrad([noAliasParam], false);
expectSame(noAliasParam.tensor.requiresGrad, false, "module state setRequiresGrad writes requiresGrad");
moduleStateNoAliasHelpers.setRequiresGrad(noAliasParam.tensor, true);
expectSame(noAliasParam.tensor.requiresGrad, true, "module state tensor setRequiresGrad writes requiresGrad");
expectSame(moduleStateHelpers.finiteConfigNumber({ lr: 0.2 }, "lr", 0.1, (value) => value > 0), 0.2, "module state finite config");
const optimizerEntry = moduleStateHelpers.optimizerStateEntry("m.0", stateParam, Float32Array.of(1, 2));
expectSame(optimizerEntry.signature.startsWith("optimizer-state-entry|name=m.0|shape=2|layout=row-major|"), true, "module state optimizer entry signature");
expectSame({ name: optimizerEntry.name, shape: optimizerEntry.shape, layout: optimizerEntry.layout, data: optimizerEntry.data }, { name: "m.0", shape: [2], layout: "row-major", data: [1, 2] }, "module state optimizer entry");
const optimizerDict = moduleStateHelpers.optimizerStateDict("adam", [stateParam], [optimizerEntry], 7);
expectSame(optimizerDict.signature.startsWith("optimizer-state|kind=adam|step=7|params=1|"), true, "module state optimizer dict signature");
const optimizerEntries = moduleStateHelpers.optimizerStateEntries(optimizerDict, "adam", 1);
expectSame(optimizerEntries.byName.get("m.0").data, [1, 2], "module state optimizer entries");
expectSame(moduleStateHelpers.optimizerStepFromState(optimizerDict, 0, true), 7, "module state optimizer step");
const optimizerTarget = new Float32Array(2);
const optimizerUpdate = moduleStateHelpers.loadOptimizerTensorState(optimizerEntries.byName, new Set(), "m.0", stateParam, optimizerTarget, true);
expectSame(optimizerUpdate.data, [1, 2], "module state optimizer tensor load");
expectThrow(
  () => moduleStateHelpers.parameterInfo([stateParam], -1),
  "module parameterInfo index must be a non-negative integer",
  "module state parameterInfo rejects negative index",
);
expectThrow(
  () => moduleStateHelpers.loadStateDict([stateParam], { weight: { shape: [3], data: [1, 2, 3] } }),
  "state dict weight shape must be [2], got [3]",
  "module state load rejects shape",
);

class LinearSmokeTensor {
  constructor(data, shape = [data.length], options = {}) {
    this.data = data instanceof Float32Array ? data : Float32Array.from(data);
    this.shape = shape.slice();
    this.length = this.data.length;
    this.rank = this.shape.length;
    this.grad = options.grad ?? null;
    this.requiresGrad = Boolean(options.requiresGrad);
    this._prev = options.prev ?? [];
    this._backward = options.backward ?? (() => {});
  }
  zeroGrad() {
    if (this.grad) this.grad.fill(0);
  }
  set requires_grad(value) {
    this.requiresGrad = Boolean(value);
  }
  get requires_grad() {
    return this.requiresGrad;
  }
  reshape(shape) {
    return new LinearSmokeTensor(new Float32Array(this.data), shape);
  }
  matmul(rhs) {
    if (this.rank !== 2 || rhs.rank !== 2) throw new Error("linear smoke matmul requires rank-2");
    const rows = this.shape[0];
    const inner = this.shape[1];
    const cols = rhs.shape[1];
    const out = new Float32Array(rows * cols);
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        let sum = 0;
        for (let k = 0; k < inner; k += 1) sum += this.data[row * inner + k] * rhs.data[k * cols + col];
        out[row * cols + col] = sum;
      }
    }
    return new LinearSmokeTensor(out, [rows, cols]);
  }
  add(rhs) {
    const out = new Float32Array(this.length);
    if (rhs.length === this.length) {
      for (let i = 0; i < out.length; i += 1) out[i] = this.data[i] + rhs.data[i];
    } else if (this.rank === 2 && rhs.rank === 1 && rhs.length === this.shape[1]) {
      for (let row = 0; row < this.shape[0]; row += 1) {
        for (let col = 0; col < this.shape[1]; col += 1) out[row * this.shape[1] + col] = this.data[row * this.shape[1] + col] + rhs.data[col];
      }
    } else {
      throw new Error("linear smoke add shape mismatch");
    }
    return new LinearSmokeTensor(out, this.shape);
  }
}
function linearF32WithLength(value, length, label) {
  const data = value instanceof Float32Array ? value : Float32Array.from(Array.isArray(value) ? value.flat(Infinity) : value);
  if (data.length !== length) throw new Error(`${label} length must be ${length}, got ${data.length}`);
  return data;
}
const linearStateHelpers = moduleState.createModuleStateHelpers({
  Tensor: LinearSmokeTensor,
  f32WithLength: linearF32WithLength,
});
const linearPrograms = [];
const linearPlacements = [];
const LinearModule = nnLinearModule.createLinearModuleClass({
  Tensor: LinearSmokeTensor,
  prepareF32: tensorData.prepareF32,
  f32WithLength: linearF32WithLength,
  requirePositiveInteger: tensorData.requirePositiveInteger,
  defaultedF32: tensorData.defaultedF32,
  zerosF32: tensorData.zerosF32,
  makeParameter: linearStateHelpers.makeParameter,
  parameterView: linearStateHelpers.parameterView,
  parameterNames: linearStateHelpers.parameterNames,
  parameterInfos: linearStateHelpers.parameterInfos,
  parameterInfo: linearStateHelpers.parameterInfo,
  zeroGrad: linearStateHelpers.zeroGrad,
  setRequiresGrad: linearStateHelpers.setRequiresGrad,
  stateDict: linearStateHelpers.stateDict,
  loadStateDict: linearStateHelpers.loadStateDict,
  analyzeSequentialProgram: (layers, options = {}) => {
    const layer = layers[0];
    if (options.mode === "module") {
      return {
        supported: true,
        reason: null,
        support: { nativePath: "device-program" },
        compiled: { kind: "module", moduleKind: layer.kind },
      };
    }
    if (options.mode === "unsupported") {
      return {
        supported: false,
        reason: "linear unsupported",
        support: { nativePath: "none" },
        compiled: null,
      };
    }
    return {
      supported: true,
      reason: null,
      support: { nativePath: "tiny-linear", modelKind: "tiny-linear" },
      trace: { ops: [{ op: "linear" }] },
      compiled: {
        kind: "tiny-linear",
        nativePath: "tiny-linear",
        modelKind: "tiny-linear",
        layer,
        layerCount: 1,
      },
    };
  },
  moduleCompileSupport: linearStateHelpers.moduleCompileSupport,
  TinyLinearModel: {
    create(config) {
      return {
        compile(options) {
          return { kind: "tiny-program", config, options };
        },
      };
    },
  },
  compileModuleProgram: (compiled, options) => {
    linearPrograms.push({ compiled, options });
    return { kind: "module-program", moduleKind: compiled.moduleKind };
  },
  attachProgramCompileEvidence: (program, evidence) => ({ ...program, evidence }),
  placeModuleParameterBindings: (module, program, options) => {
    linearPlacements.push({ kind: module.kind, program, options });
    return { placed: module.kind };
  },
});
let conv2dGradEnabled = false;
const nativeConv2dCalls = [];
const Conv2dModule = nnConvModule.createConv2dModuleClass({
  Tensor: LinearSmokeTensor,
  addTensorGrad: (tensor, grad) => {
    if (!tensor.grad) tensor.grad = new Float32Array(tensor.length);
    for (let i = 0; i < grad.length; i += 1) tensor.grad[i] += grad[i];
  },
  isGradEnabled: () => conv2dGradEnabled,
  requirePositiveInteger: tensorData.requirePositiveInteger,
  defaultedF32: tensorData.defaultedF32,
  zerosF32: tensorData.zerosF32,
  makeParameter: linearStateHelpers.makeParameter,
  parameterView: linearStateHelpers.parameterView,
  nativeEagerConv2dInto: (output, input, weights, options) => {
    nativeConv2dCalls.push({ inputShape: input.shape, weightShape: weights.shape, options });
    output.fill(42);
    return output;
  },
  parameterNames: linearStateHelpers.parameterNames,
  parameterInfos: linearStateHelpers.parameterInfos,
  parameterInfo: linearStateHelpers.parameterInfo,
  zeroGrad: linearStateHelpers.zeroGrad,
  setRequiresGrad: linearStateHelpers.setRequiresGrad,
  stateDict: linearStateHelpers.stateDict,
  loadStateDict: linearStateHelpers.loadStateDict,
  moduleCompileSupport: linearStateHelpers.moduleCompileSupport,
  analyzeSingleModuleProgram: () => ({
    supported: true,
    reason: null,
    support: { nativePath: "device-program" },
    compiled: { kind: "module", moduleKind: "conv2d" },
  }),
  compileModuleProgram: (compiled) => ({ kind: "module-program", moduleKind: compiled.moduleKind }),
  placeModuleParameterBindings: (module) => ({ placed: module.kind }),
  packedSequentialProgramParameters: () => ({ weights: [], bias: [] }),
});
const conv2dModule = new Conv2dModule(1, 1, [2, 2], {
  weight: [1, 0, 0, 1],
  bias: [0.5],
});
const conv2dInput = new LinearSmokeTensor(Float32Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9), [1, 3, 3]);
expectSame(conv2dModule.forward(conv2dInput).data, [42, 42, 42, 42], "nn conv2d no-grad routes through native eager hook");
expectSame(nativeConv2dCalls.length, 1, "nn conv2d native eager hook is called once in no-grad");
expectSame(nativeConv2dCalls[0].options, {
  bias: conv2dModule.biasParam.tensor,
  batch: 1,
  inChannels: 1,
  height: 3,
  width: 3,
  outChannels: 1,
  kernelH: 2,
  kernelW: 2,
  strideH: 1,
  strideW: 1,
  paddingH: 0,
  paddingW: 0,
  dilationH: 1,
  dilationW: 1,
  outH: 2,
  outW: 2,
}, "nn conv2d native eager hook receives shape contract");
conv2dGradEnabled = true;
expectSame(conv2dModule.forward(conv2dInput).data, [6.5, 8.5, 12.5, 14.5], "nn conv2d grad-enabled forward stays on TS autograd path");
expectSame(nativeConv2dCalls.length, 1, "nn conv2d does not route grad-enabled forward through native eager hook");
conv2dGradEnabled = false;
const linearModule = new LinearModule(2, 3, {
  weights: [1, 2, 3, 4, 5, 6],
  bias: [0.5, -0.5, 1],
});
expectSame({
  kind: linearModule.kind,
  inFeatures: linearModule.inFeatures,
  outFeatures: linearModule.outFeatures,
  weight: linearModule.weight,
  bias: linearModule.bias,
  training: linearModule.training,
}, {
  kind: "linear",
  inFeatures: 2,
  outFeatures: 3,
  weight: [1, 2, 3, 4, 5, 6],
  bias: [0.5, -0.5, 1],
  training: true,
}, "nn linear initializes");
expectSame(linearModule.forward([2, 3]).data, [14.5, 18.5, 25], "nn linear vector forward");
expectSame(linearModule.forward([[1, 2], [3, 4]]).shape, [2, 3], "nn linear nested array preserves batch shape");
expectSame(linearModule.forward([[1, 2], [3, 4]]).data, [9.5, 11.5, 16, 19.5, 25.5, 34], "nn linear nested array batch forward");
expectSame(linearModule.forward(new LinearSmokeTensor(Float32Array.of(1, 2, 3, 4), [2, 2])).data, [9.5, 11.5, 16, 19.5, 25.5, 34], "nn linear batch forward");
const traceableLinearModule = tsNode.nn.sequential([
  tsNode.nn.linear(2, 3, {
    weights: [1, 2, 3, 4, 5, 6],
    bias: [0.5, -0.5, 1],
  }),
]);
const traceableLinearLayerList = [
  tsNode.nn.linear(2, 3, {
    weights: [1, 2, 3, 4, 5, 6],
    bias: [0.5, -0.5, 1],
  }),
];
expectSame(compileNamespace.trace(traceableLinearLayerList, { inputShape: [2] }).kind, "sequential", "compile namespace traces raw Sequential layer lists");
expectSame(compileNamespace.requireCompileSupport(traceableLinearLayerList, { inputShape: [2] }).supported, true, "compile namespace supports raw Sequential layer list evidence");
expectSame(compileNamespace.canCompile(traceableLinearLayerList, { inputShape: [2] }), true, "compile namespace canCompile raw Sequential layer lists");
expectSame(compileNamespace.compilerSignatures(traceableLinearLayerList, { inputShape: [2] }).kernelPlan.length > 0, true, "compile namespace exposes raw Sequential layer list compiler signatures");
expectSame(compileNamespace.tensorProgramIr(traceableLinearLayerList, { inputShape: [2] }).kind, "tensor-program-ir", "compile namespace exposes raw Sequential layer list tensor Program IR");
expectSame(compileNamespace.kernelPlan(traceableLinearLayerList, { inputShape: [2] }).kind, "native-module-kernel-plan", "compile namespace exposes raw Sequential layer list kernel plan");
expectSame(compileNamespace.bufferLayout(traceableLinearLayerList, { inputShape: [2] }).input.elementCount, 2, "compile namespace exposes raw Sequential layer list buffer layout");
expectSame(compileNamespace.memoryLayout(traceableLinearLayerList, { inputShape: [2] }).values[0].byteLength, 8, "compile namespace exposes raw Sequential layer list memory layout");
expectSame(compileNamespace.shapeConstraints(traceableLinearLayerList, { inputShape: [2] }).inputShape, [2], "compile namespace exposes raw Sequential layer list shape constraints");
expectSame(compileNamespace.parameterLayout(traceableLinearLayerList, { inputShape: [2] }).weightsLen + compileNamespace.parameterLayout(traceableLinearLayerList, { inputShape: [2] }).biasLen, 9, "compile namespace exposes raw Sequential layer list parameter layout");
expectSame(tsNode.nn.parameterNames(traceableLinearLayerList, "model"), ["model.0.weight", "model.0.bias"], "nn namespace exposes raw Sequential layer list parameter names through Sequential");
expectSame(tsNode.nn.parameterInfo(traceableLinearLayerList, "model.0.weight", "model").shape, [2, 3], "nn namespace exposes raw Sequential layer list parameter info through Sequential");
const rawLayerListState = tsNode.nn.stateDict(traceableLinearLayerList, "model");
expectSame(rawLayerListState["model.0.weight"].shape, [2, 3], "nn namespace stateDict accepts raw Sequential layer lists through Sequential");
expectSame(tsNode.nn.loadStateDict(traceableLinearLayerList, rawLayerListState, { prefix: "model", strict: true, validateOnly: true }).length, 1, "nn namespace loadStateDict validates raw Sequential layer lists through Sequential");
expectSame(tsNode.nn.requiresGrad(traceableLinearLayerList, false).length, 1, "nn namespace requiresGrad accepts raw Sequential layer lists through Sequential");
tsNode.nn.zeroGrad(traceableLinearLayerList);
expectSame(tsNode.nn.forward(traceableLinearLayerList, tsNode.tensor([1, 2], [2])).data, [9.5, 11.5, 16], "nn namespace forwards raw Sequential layer lists through Sequential");
expectSame(tsNode.nn.children(traceableLinearLayerList).length, 1, "nn namespace traverses raw Sequential layer list children through Sequential");
expectSame(tsNode.nn.namedModules(traceableLinearLayerList).map((entry) => entry.name), ["", "0"], "nn namespace traverses raw Sequential layer list modules through Sequential");
expectSame(tsNode.nn.trace(traceableLinearLayerList, { inputShape: [2] }).kind, "sequential", "nn namespace traces raw Sequential layer lists through Sequential");
expectSame(tsNode.nn.requireCompileSupport(traceableLinearLayerList, { inputShape: [2] }).supported, true, "nn namespace supports raw Sequential layer list evidence through Sequential");
const nnRawLayerListProgram = tsNode.nn.compile(traceableLinearLayerList, { inputShape: [2] });
expectSame(nnRawLayerListProgram.outputShape(), [3], "nn namespace compiles raw Sequential layer lists through Sequential");
nnRawLayerListProgram.free();
const linearCompileAnalysis = compileNamespace.analyze(traceableLinearModule, { inputShape: [2] });
expectSame(Object.isFrozen(linearCompileAnalysis), true, "compile analysis is frozen evidence");
expectSame(Object.isFrozen(linearCompileAnalysis.artifacts), true, "compile analysis artifacts are frozen evidence");
expectSame(linearCompileAnalysis.kind, "zgml.compile.analysis", "compile analysis kind");
expectSame(
  linearCompileAnalysis.signature,
  "compile-analysis|traceKind=sequential|layers=1|ops=1|params=2|paramScalars=9|shapeKnown=1|input=2|output=3|ir=tensor-program-ir:1:2:3:2:3:0:3:4:1:2:9::0:input:2:f32:1:2:4:8:row-major:1:0:dense::::::|1:parameter:2x3:f32:2:6:4:24:row-major:3x1:0:dense:0.weight:weights:row-major:linear.weight[in_features,out_features]:0:linear:0|2:parameter:3:f32:1:3:4:12:row-major:1:0:dense:0.bias:bias:row-major:linear.bias[out_features]:0:linear:0|3:op-output:3:f32:1:3:4:12:row-major:1:0:dense::::0:linear:0::0:0:linear:2:3:2:3:0x1x2:3:1x2:9:bias=true,inFeatures=2,outFeatures=3|kernel=dispatch=1:descriptor=1::linear:linear:2:3:f32:4:9:1:1:linear:0+1+2=>3:1::::1:0:2:0:2:3:0:0::elided::|diagnostic=null",
  "compile analysis signature",
);
expectSame(compileNamespace.isCompileAnalysis(linearCompileAnalysis), true, "compile analysis predicate");
expectSame(compileNamespace.requireCompileAnalysis(linearCompileAnalysis).artifacts.kernelPlan?.opCount, 1, "compile analysis require");
expectSame(compileNamespace.assertCompileAnalysis(linearCompileAnalysis).trace.layerCount, 1, "compile analysis assert");
expectSame(compileNamespace.assert_compile_analysis(linearCompileAnalysis).signature, linearCompileAnalysis.signature, "compile analysis snake assert");
expectSame(compileNamespace.compileAnalysisSignature(linearCompileAnalysis), linearCompileAnalysis.signature, "compile analysis signature helper");
expectSame(compileNamespace.matchesCompileAnalysisSignature(linearCompileAnalysis, linearCompileAnalysis.signature), true, "compile analysis signature match");
expectSame(compileNamespace.matches_compile_analysis_signature(linearCompileAnalysis, "wrong"), false, "compile analysis snake signature mismatch");
expectSame(compileNamespace.isCompileAnalysis({ ...linearCompileAnalysis }), false, "compile analysis rejects mutable evidence");
expectThrow(
  () => compileNamespace.requireCompileAnalysis(Object.freeze({ ...linearCompileAnalysis, signature: "wrong" })),
  "expected frozen CompileAnalysis evidence",
  "compile analysis rejects mismatched signature",
);
expectSame(linearModule.parameterNames(), ["weight", "bias"], "nn linear parameter names");
expectSame(linearModule.parameterInfos()[0], {
  name: "weight",
  index: 0,
  scalarCount: 6,
  shape: [2, 3],
  layout: "row-major:linear.weight[in_features,out_features]",
  requiresGrad: true,
  requires_grad: true,
}, "nn linear parameter info");
expectSame(linearModule.parameterInfo("bias").shape, [3], "nn linear parameterInfo by name");
linearModule.weightParam.grad.set([1, 2, 3, 4, 5, 6]);
linearModule.zeroGrad();
expectSame(linearModule.weightParam.grad, [0, 0, 0, 0, 0, 0], "nn linear zeroGrad");
linearModule.requiresGrad_(false);
expectSame(linearModule.weightParam.tensor.requiresGrad, false, "nn linear requiresGrad false");
expectSame(linearModule.eval().training, false, "nn linear eval");
expectSame(linearModule.train(true).training, true, "nn linear train");
const linearState = linearModule.stateDict();
expectSame(linearState.weight.data, [1, 2, 3, 4, 5, 6], "nn linear stateDict weight");
linearModule.loadStateDict({ weight: { shape: [2, 3], data: [6, 5, 4, 3, 2, 1] }, bias: { shape: [3], data: [1, 1, 1] } });
expectSame({ weight: linearModule.weight, bias: linearModule.bias }, { weight: [6, 5, 4, 3, 2, 1], bias: [1, 1, 1] }, "nn linear loadStateDict");
expectSame(linearModule.compileSupport().nativePath, "tiny-linear", "nn linear compileSupport tiny");
expectSame(linearModule.canCompile(), true, "nn linear canCompile");
const linearBindings = linearModule.bindParameters({ inputShape: [2] });
expectSame({ weights: linearBindings.weights, bias: linearBindings.bias }, {
  weights: [6, 5, 4, 3, 2, 1],
  bias: [1, 1, 1],
}, "nn linear bind parameters");
expectSame(linearModule.placeParameters({ name: "program" }, { label: "linear" }), { placed: "linear" }, "nn linear place parameters");
const tinyProgram = linearModule.compile({ backend: "cpu" });
expectSame({
  kind: tinyProgram.kind,
  config: tinyProgram.config,
  evidenceKind: tinyProgram.evidence.kind,
  evidenceNativePath: tinyProgram.evidence.nativePath,
}, {
  kind: "tiny-program",
  config: { inputLen: 2, outputLen: 3 },
  evidenceKind: "tiny-linear",
  evidenceNativePath: "tiny-linear",
}, "nn linear tiny compile");
expectSame(linearModule.compile({ mode: "module" }), { kind: "module-program", moduleKind: "linear" }, "nn linear module compile");
expectSame(linearPrograms, [{ compiled: { kind: "module", moduleKind: "linear" }, options: { mode: "module" } }], "nn linear module compile hook");
expectSame(linearPlacements, [{ kind: "linear", program: { name: "program" }, options: { label: "linear" } }], "nn linear placement hook");
expectThrow(
  () => new LinearModule(0, 1),
  "linear inFeatures must be a positive safe integer, got 0",
  "nn linear rejects inFeatures",
);
expectThrow(
  () => new LinearModule(2, 3, { weights: [1, 2] }),
  "linear weight length must be 6, got 2",
  "nn linear rejects weight length",
);
expectThrow(
  () => linearModule.forward(new LinearSmokeTensor(Float32Array.of(1, 2, 3), [3])),
  "linear input length must be 2, got 3",
  "nn linear rejects vector length",
);
expectThrow(
  () => linearModule.forward(new LinearSmokeTensor(Float32Array.of(1, 2, 3), [1, 3])),
  "linear input shape must be [2] or [batch, 2], got [1,3]",
  "nn linear rejects matrix shape",
);
expectThrow(
  () => linearModule.compile({ mode: "unsupported" }),
  "linear unsupported",
  "nn linear rejects unsupported compile",
);

const embeddingPrograms = [];
const embeddingPlacements = [];
const embeddingTrace = { ops: [{ op: "embedding" }] };
const EmbeddingModule = nnEmbeddingModule.createEmbeddingModuleClass({
  Tensor: LinearSmokeTensor,
  indexValues: (value, label) => {
    if (value instanceof LinearSmokeTensor) return Array.from(value.data);
    if (value instanceof TensorDataSmokeTensor) return Array.from(value.data);
    if (!Array.isArray(value) && !(value instanceof Uint32Array) && !(value instanceof Int32Array)) throw new Error(`${label} must be an index array`);
    return Array.from(value);
  },
  addTensorGrad: (tensor, grad) => {
    if (!tensor.grad) tensor.grad = new Float32Array(tensor.length);
    for (let i = 0; i < grad.length; i += 1) tensor.grad[i] += grad[i];
  },
  isGradEnabled: () => true,
  requirePositiveInteger: tensorData.requirePositiveInteger,
  defaultedF32: tensorData.defaultedF32,
  zerosF32: tensorData.zerosF32,
  makeParameter: linearStateHelpers.makeParameter,
  parameterView: linearStateHelpers.parameterView,
  parameterNames: linearStateHelpers.parameterNames,
  parameterInfos: linearStateHelpers.parameterInfos,
  parameterInfo: linearStateHelpers.parameterInfo,
  zeroGrad: linearStateHelpers.zeroGrad,
  setRequiresGrad: linearStateHelpers.setRequiresGrad,
  stateDict: linearStateHelpers.stateDict,
  loadStateDict: linearStateHelpers.loadStateDict,
  traceSequentialProgram: () => embeddingTrace,
  freezeSequentialTrace: (trace) => Object.freeze({ ...trace, ops: Object.freeze(trace.ops.slice()) }),
  analyzeSingleModuleProgram: (module, options = {}) => {
    if (options.mode === "unsupported") {
      return {
        supported: false,
        reason: "embedding unsupported",
        support: { nativePath: "none" },
        compiled: null,
      };
    }
    return {
      supported: true,
      reason: null,
      support: { nativePath: "device-program" },
      compiled: { kind: "module", moduleKind: module.kind },
    };
  },
  moduleCompileSupport: linearStateHelpers.moduleCompileSupport,
  packedSequentialProgramParameters: () => ({ weights: new Float32Array(0) }),
  compileModuleProgram: (compiled, options) => {
    embeddingPrograms.push({ compiled, options });
    return { kind: "embedding-program", moduleKind: compiled.moduleKind };
  },
  placeModuleParameterBindings: (module, program, options) => {
    embeddingPlacements.push({ kind: module.kind, program, options });
    return { placed: module.kind };
  },
});
const embeddingModule = new EmbeddingModule(4, 2, { weight: [0, 1, 10, 11, 20, 21, 30, 31] });
expectSame({
  kind: embeddingModule.kind,
  numEmbeddings: embeddingModule.numEmbeddings,
  embeddingDim: embeddingModule.embeddingDim,
  weight: embeddingModule.weight,
  training: embeddingModule.training,
}, {
  kind: "embedding",
  numEmbeddings: 4,
  embeddingDim: 2,
  weight: [0, 1, 10, 11, 20, 21, 30, 31],
  training: true,
}, "nn embedding initializes");
const embeddingOut = embeddingModule.forward([1, 3, 1]);
expectSame({ data: embeddingOut.data, shape: embeddingOut.shape, requiresGrad: embeddingOut.requiresGrad, prev: embeddingOut._prev }, {
  data: [10, 11, 30, 31, 10, 11],
  shape: [3, 2],
  requiresGrad: true,
  prev: [embeddingModule.weightParam.tensor],
}, "nn embedding forward");
const embeddingGridOut = embeddingModule.forward(new LinearSmokeTensor(Float32Array.of(1, 3, 1, 0, 2, 3), [2, 3]));
expectSame({ data: embeddingGridOut.data, shape: embeddingGridOut.shape }, {
  data: [10, 11, 30, 31, 10, 11, 0, 1, 20, 21, 30, 31],
  shape: [2, 3, 2],
}, "nn embedding forward preserves tensor index shape");
embeddingOut._backward(Float32Array.of(1, 2, 3, 4, 5, 6));
expectSame(embeddingModule.weightParam.tensor.grad, [0, 0, 6, 8, 0, 0, 3, 4], "nn embedding repeated index grad");
expectSame(embeddingModule.parameterNames(), ["weight"], "nn embedding parameter names");
expectSame(embeddingModule.parameterInfos()[0], {
  name: "weight",
  index: 0,
  scalarCount: 8,
  shape: [4, 2],
  layout: "row-major:embedding.weight[num_embeddings,embedding_dim]",
  requiresGrad: true,
  requires_grad: true,
}, "nn embedding parameter info");
embeddingModule.zeroGrad();
expectSame(embeddingModule.weightParam.grad, [0, 0, 0, 0, 0, 0, 0, 0], "nn embedding zeroGrad");
embeddingModule.requiresGrad_(false);
expectSame(embeddingModule.weightParam.tensor.requiresGrad, false, "nn embedding requiresGrad false");
embeddingModule.requiresGrad_(true);
expectSame(embeddingModule.eval().training, false, "nn embedding eval");
expectSame(embeddingModule.train(true).training, true, "nn embedding train");
const embeddingState = embeddingModule.stateDict();
expectSame(embeddingState.weight.data, [0, 1, 10, 11, 20, 21, 30, 31], "nn embedding stateDict");
embeddingModule.loadStateDict({ weight: { shape: [4, 2], data: [1, 2, 3, 4, 5, 6, 7, 8] } });
expectSame(embeddingModule.weight, [1, 2, 3, 4, 5, 6, 7, 8], "nn embedding loadStateDict");
const embeddingMissingInputSupport = embeddingModule.compileSupport();
expectSame({
  supported: embeddingMissingInputSupport.supported,
  reason: embeddingMissingInputSupport.reason,
  diagnosticCode: embeddingMissingInputSupport.diagnostics[0].code,
  traceFrozen: Object.isFrozen(embeddingMissingInputSupport.trace.ops),
}, {
  supported: false,
  reason: "nn.Embedding.compile requires inputShape because output length depends on index count",
  diagnosticCode: "missing-input-shape",
  traceFrozen: true,
}, "nn embedding compileSupport missing inputShape");
expectSame(embeddingModule.compileSupport({ inputShape: [3] }).supported, true, "nn embedding compileSupport");
expectSame(embeddingModule.canCompile({ inputShape: [3] }), true, "nn embedding canCompile");
expectSame(embeddingModule.bindParameters({ inputShape: [3] }).weights, [], "nn embedding bind parameters");
expectSame(embeddingModule.placeParameters({ name: "program" }, { label: "embedding" }), { placed: "embedding" }, "nn embedding place parameters");
expectSame(embeddingModule.compile({ inputShape: [3] }), { kind: "embedding-program", moduleKind: "embedding" }, "nn embedding compile");
expectSame(embeddingPrograms, [{ compiled: { kind: "module", moduleKind: "embedding" }, options: { inputShape: [3] } }], "nn embedding compile hook");
expectSame(embeddingPlacements, [{ kind: "embedding", program: { name: "program" }, options: { label: "embedding" } }], "nn embedding placement hook");
expectThrow(
  () => new EmbeddingModule(0, 2),
  "embedding numEmbeddings must be a positive safe integer, got 0",
  "nn embedding rejects numEmbeddings",
);
expectThrow(
  () => new EmbeddingModule(2, 2, { weights: [1, 2] }),
  "embedding weight length must be 4, got 2",
  "nn embedding rejects weight length",
);
expectThrow(
  () => embeddingModule.forward([4]),
  "embedding index 4 at position 0 is out of range 0..3",
  "nn embedding rejects index range",
);
expectThrow(
  () => embeddingModule.forward([-1]),
  "embedding index -1 at position 0 is out of range 0..3",
  "nn embedding rejects negative index",
);
expectThrow(
  () => embeddingModule.compile({ mode: "unsupported", inputShape: [1] }),
  "embedding unsupported",
  "nn embedding rejects unsupported compile",
);

function makeSequentialParam(name, data, shape, layout = "row-major:test") {
  return {
    name,
    data: Float32Array.from(data),
    grad: new Float32Array(data.length),
    layout,
    tensor: new LinearSmokeTensor(Float32Array.from(data), shape, { grad: new Float32Array(data.length), requiresGrad: true }),
  };
}
function makeSequentialLayer(kind, params = []) {
  const layer = {
    kind,
    training: true,
    forward(input) {
      return `${input}|${kind}`;
    },
    parameters(prefix = "") {
      return params.map((param) => linearStateHelpers.parameterView(prefix, param));
    },
    children() { return Object.freeze([]); },
    modules() { return Object.freeze([this]); },
    namedChildren() { return Object.freeze([]); },
    namedModules(prefix = "") { return Object.freeze([{ name: String(prefix), module: this }]); },
    train(mode = true) {
      this.training = Boolean(mode);
      return this;
    },
  };
  return layer;
}
const seqParamA = makeSequentialParam("weight", [1, 2], [2], "row-major:a");
const seqParamB = makeSequentialParam("bias", [3], [1], "row-major:b");
const seqLayerA = makeSequentialLayer("first", [seqParamA]);
const seqLayerB0 = makeSequentialLayer("inner-first");
const seqLayerB1 = makeSequentialLayer("inner-second", [seqParamB]);
const SequentialModule = nnSequentialModule.createSequentialModuleClass({
  Tensor: LinearSmokeTensor,
  f32: (value) => value,
  zeroGrad: linearStateHelpers.zeroGrad,
  parameterNames: linearStateHelpers.parameterNames,
  parameterInfos: linearStateHelpers.parameterInfos,
  parameterInfo: linearStateHelpers.parameterInfo,
  setRequiresGrad: linearStateHelpers.setRequiresGrad,
  stateDict: linearStateHelpers.stateDict,
  loadStateDict: linearStateHelpers.loadStateDict,
  traceSequentialProgram: (layers, options = {}) => Object.freeze({
    kind: "trace",
    layerKinds: Object.freeze(layers.map((layer) => layer.kind)),
    options,
  }),
  analyzeSequentialProgram: (layers, options = {}) => {
    if (options.mode === "unsupported") {
      return { supported: false, reason: "sequential unsupported", support: { nativePath: "none" }, compiled: null };
    }
    if (options.mode === "tiny") {
      return {
        supported: true,
        reason: null,
        support: { nativePath: "tiny-linear" },
        compiled: {
          kind: "tiny-linear",
          nativePath: "tiny-linear",
          modelKind: "tiny-linear",
          layer: layers[0],
          layerCount: layers.length,
          trace: { ops: [{ op: "linear" }] },
        },
      };
    }
    return {
      supported: true,
      reason: null,
      support: { nativePath: "device-program" },
      compiled: { kind: "module", moduleKind: "sequential", layerCount: layers.length },
    };
  },
  moduleCompileSupport: linearStateHelpers.moduleCompileSupport,
  packedSequentialProgramParameters: () => ({ weights: Float32Array.of(9) }),
  compileModuleProgram: (compiled, options) => ({ kind: "sequential-program", compiled, options }),
  attachProgramCompileEvidence: (program, evidence) => ({ ...program, evidence }),
  placeModuleParameterBindings: (module, program, options) => ({ placed: module.layers.length, program, options }),
});
const innerSequential = new SequentialModule([seqLayerB0, seqLayerB1]);
const sequentialModule = new SequentialModule([seqLayerA, innerSequential]);
const namedSequentialModule = new SequentialModule({ stem: seqLayerA, tail: innerSequential });
const emptySequentialModule = new SequentialModule();
expectSame(emptySequentialModule.length, 0, "nn sequential empty length");
expectSame(emptySequentialModule.forward("input"), "input", "nn sequential empty forward is identity");
expectSame(emptySequentialModule.children(), [], "nn sequential empty children");
expectSame(emptySequentialModule.namedModules().map((entry) => entry.name), [""], "nn sequential empty namedModules");
expectSame(emptySequentialModule.parameterNames(), [], "nn sequential empty parameterNames");
expectSame(namedSequentialModule.forward("input"), "input|first|inner-first|inner-second", "nn sequential named forward");
expectSame(namedSequentialModule.namedChildren().map((entry) => entry.name), ["stem", "tail"], "nn sequential namedChildren preserve object keys");
expectSame(namedSequentialModule.namedModules().map((entry) => entry.name), ["", "stem", "tail", "tail.0", "tail.1"], "nn sequential namedModules preserve object keys");
expectSame(namedSequentialModule.parameterNames(), ["stem.weight", "tail.1.bias"], "nn sequential named parameterNames");
expectSame(namedSequentialModule.stateDict()["stem.weight"].data, [1, 2], "nn sequential named stateDict");
expectSame(sequentialModule.length, 2, "nn sequential length");
expectSame(sequentialModule.at(0), seqLayerA, "nn sequential at first");
expectSame(sequentialModule.at(-1), innerSequential, "nn sequential at negative");
expectSame(Array.from(sequentialModule), [seqLayerA, innerSequential], "nn sequential iterator");
expectSame(sequentialModule.forward("input"), "input|first|inner-first|inner-second", "nn sequential forward");
expectSame(sequentialModule.children(), [seqLayerA, innerSequential], "nn sequential children");
expectSame(sequentialModule.modules().map((module) => module.kind ?? "sequential").join("|"), "sequential|first|sequential|inner-first|inner-second", "nn sequential modules");
expectSame(sequentialModule.namedChildren().map((entry) => entry.name), ["0", "1"], "nn sequential namedChildren");
expectSame(sequentialModule.namedModules().map((entry) => entry.name), ["", "0", "1", "1.0", "1.1"], "nn sequential namedModules");
expectSame(sequentialModule.namedModules("model").map((entry) => entry.name), ["model", "model.0", "model.1", "model.1.0", "model.1.1"], "nn sequential prefixed namedModules");
expectSame(sequentialModule.parameterNames(), ["0.weight", "1.1.bias"], "nn sequential parameterNames");
expectSame(sequentialModule.parameterNames("model"), ["model.0.weight", "model.1.1.bias"], "nn sequential prefixed parameterNames");
const sequentialPop = new SequentialModule([seqLayerA, innerSequential]);
expectSame(sequentialPop.pop(), innerSequential, "nn sequential pop default removes last layer");
expectSame(sequentialPop.pop(0), seqLayerA, "nn sequential pop removes indexed layer");
expectSame({ length: sequentialPop.length, names: sequentialPop.namedChildren().map((entry) => entry.name) }, { length: 0, names: [] }, "nn sequential pop updates layer names");
const sequentialClear = new SequentialModule({ stem: seqLayerA, tail: innerSequential });
expectSame(sequentialClear.clear(), sequentialClear, "nn sequential clear returns self");
expectSame({ length: sequentialClear.length, names: sequentialClear.namedChildren().map((entry) => entry.name), params: sequentialClear.parameterNames() }, { length: 0, names: [], params: [] }, "nn sequential clear removes layers and names");
expectSame(sequentialModule.parameterInfos()[1], {
  name: "1.1.bias",
  index: 1,
  scalarCount: 1,
  shape: [1],
  layout: "row-major:b",
  requiresGrad: true,
  requires_grad: true,
}, "nn sequential parameterInfos");
expectSame(sequentialModule.parameterInfos("model")[1].name, "model.1.1.bias", "nn sequential prefixed parameterInfos");
expectSame(sequentialModule.parameterInfo("0.weight").shape, [2], "nn sequential parameterInfo by name");
expectSame(sequentialModule.parameterInfo("model.0.weight", "model").shape, [2], "nn sequential prefixed parameterInfo by name");
seqParamA.grad.set([4, 5]);
sequentialModule.zeroGrad();
expectSame(seqParamA.grad, [0, 0], "nn sequential zeroGrad");
sequentialModule.requiresGrad_(false);
expectSame(seqParamA.tensor.requiresGrad, false, "nn sequential requiresGrad false");
sequentialModule.train(false);
expectSame({ parent: sequentialModule.training, first: seqLayerA.training, inner: innerSequential.training, innerFirst: seqLayerB0.training }, {
  parent: false,
  first: false,
  inner: false,
  innerFirst: false,
}, "nn sequential train propagates");
sequentialModule.train(true);
expectSame(sequentialModule.stateDict()["0.weight"].data, [1, 2], "nn sequential stateDict");
sequentialModule.loadStateDict({ "0.weight": { shape: [2], data: [7, 8] }, "1.1.bias": { shape: [1], data: [9] } });
expectSame({ a: seqParamA.data, b: seqParamB.data }, { a: [7, 8], b: [9] }, "nn sequential loadStateDict");
expectSame(sequentialModule.trace({ inputShape: [2] }).layerKinds, ["first", undefined], "nn sequential trace");
expectSame(sequentialModule.compileSupport({ mode: "module" }).nativePath, "device-program", "nn sequential compileSupport");
expectSame(sequentialModule.canCompile({ mode: "module" }), true, "nn sequential canCompile");
expectSame(sequentialModule.bindParameters({ mode: "module" }).weights, [9], "nn sequential bind parameters");
expectSame(sequentialModule.placeParameters({ name: "program" }, { label: "seq" }), { placed: 2, program: { name: "program" }, options: { label: "seq" } }, "nn sequential place parameters");
expectSame(sequentialModule.compile({ mode: "module" }).kind, "sequential-program", "nn sequential module compile");
seqLayerA.compile = (options) => ({ kind: "tiny-program", options });
const tinySequentialProgram = sequentialModule.compile({ mode: "tiny" });
expectSame({
  kind: tinySequentialProgram.kind,
  evidenceKind: tinySequentialProgram.evidence.kind,
  evidenceLayerCount: tinySequentialProgram.evidence.layerCount,
}, {
  kind: "tiny-program",
  evidenceKind: "tiny-linear",
  evidenceLayerCount: 2,
}, "nn sequential tiny compile");
expectThrow(
  () => sequentialModule.at(2),
  "Sequential index 2 is out of range for length 2",
  "nn sequential rejects out of range",
);
expectThrow(
  () => sequentialModule.at(0.5),
  "Sequential index must be a safe integer, got 0.5",
  "nn sequential rejects non integer",
);
expectThrow(
  () => sequentialModule.parameterInfo({}),
  "module parameterInfo requires a parameter name or index",
  "nn sequential rejects parameterInfo type",
);
expectThrow(
  () => sequentialModule.compile({ mode: "unsupported" }),
  "sequential unsupported",
  "nn sequential rejects unsupported compile",
);

const featureNormPrograms = [];
const featureNormPlacements = [];
const featureNormAddGrad = (tensor, grad) => {
  if (!tensor.grad) tensor.grad = new Float32Array(tensor.length);
  for (let i = 0; i < grad.length; i += 1) tensor.grad[i] += grad[i];
};
const FeatureNormModule = nnFeatureNormModule.createFeatureNormModuleClass({
  Tensor: LinearSmokeTensor,
  f32: (value) => {
    if (value instanceof LinearSmokeTensor) return value.data;
    if (value instanceof Float32Array) return value;
    return Float32Array.from(Array.isArray(value) ? value.flat(Infinity) : [value]);
  },
  addTensorGrad: featureNormAddGrad,
  isGradEnabled: () => true,
  requirePositiveInteger: tensorData.requirePositiveInteger,
  defaultedF32: tensorData.defaultedF32,
  zerosF32: tensorData.zerosF32,
  makeParameter: linearStateHelpers.makeParameter,
  parameterView: linearStateHelpers.parameterView,
  parameterNames: linearStateHelpers.parameterNames,
  parameterInfos: linearStateHelpers.parameterInfos,
  parameterInfo: linearStateHelpers.parameterInfo,
  zeroGrad: linearStateHelpers.zeroGrad,
  setRequiresGrad: linearStateHelpers.setRequiresGrad,
  stateDict: linearStateHelpers.stateDict,
  loadStateDict: linearStateHelpers.loadStateDict,
  analyzeSingleModuleProgram: (module, options = {}) => {
    if (options.mode === "unsupported") {
      return { supported: false, reason: "feature norm unsupported", support: { nativePath: "none" }, compiled: null };
    }
    return {
      supported: true,
      reason: null,
      support: { nativePath: "device-program" },
      compiled: { kind: "module", moduleKind: module.kind },
    };
  },
  moduleCompileSupport: linearStateHelpers.moduleCompileSupport,
  compileModuleProgram: (compiled, options) => {
    featureNormPrograms.push({ compiled, options });
    return { kind: "feature-norm-program", moduleKind: compiled.moduleKind };
  },
  placeModuleParameterBindings: (module, program, options) => {
    featureNormPlacements.push({ kind: module.kind, program, options });
    return { placed: module.kind };
  },
});
const layerNormModule = new FeatureNormModule("layerNorm", 2, {
  eps: 0,
  weight: [1, 2],
  bias: [0.5, -0.5],
});
expectSame({
  kind: layerNormModule.kind,
  features: layerNormModule.features,
  eps: layerNormModule.eps,
  weight: layerNormModule.weight,
  bias: layerNormModule.bias,
  training: layerNormModule.training,
}, {
  kind: "layerNorm",
  features: 2,
  eps: 0,
  weight: [1, 2],
  bias: [0.5, -0.5],
  training: true,
}, "nn feature norm layerNorm initializes");
expectSame(layerNormModule.forward([1, 2, 3, 1]), [-0.5, 1.5, 1.5, -2.5], "nn feature norm layerNorm forward");
const layerNormInput = new LinearSmokeTensor(Float32Array.of(1, 2, 3, 1), [2, 2], { requiresGrad: true });
const layerNormOut = layerNormModule.forward(layerNormInput);
expectSame({
  data: layerNormOut.data,
  shape: layerNormOut.shape,
  requiresGrad: layerNormOut.requiresGrad,
  prev: layerNormOut._prev,
}, {
  data: [-0.5, 1.5, 1.5, -2.5],
  shape: [2, 2],
  requiresGrad: true,
  prev: [layerNormInput, layerNormModule.weightParam.tensor, layerNormModule.biasParam.tensor],
}, "nn feature norm layerNorm tensor forward");
layerNormOut._backward(Float32Array.of(1, 1, 1, 1));
expectSame(Array.from(layerNormInput.grad, (value) => Number(value.toFixed(6))), [0, 0, 0, 0], "nn feature norm layerNorm input grad");
expectSame(Array.from(layerNormModule.weightParam.tensor.grad, (value) => Number(value.toFixed(6))), [0, 0], "nn feature norm layerNorm weight grad");
expectSame(layerNormModule.biasParam.tensor.grad, [2, 2], "nn feature norm layerNorm bias grad");
expectSame(layerNormModule.parameterNames(), ["weight", "bias"], "nn feature norm parameter names");
expectSame(layerNormModule.parameterInfos()[0], {
  name: "weight",
  index: 0,
  scalarCount: 2,
  shape: [2],
  layout: "row-major:layerNorm.weight[features]",
  requiresGrad: true,
  requires_grad: true,
}, "nn feature norm parameter info");
expectSame(layerNormModule.parameterInfo("bias").shape, [2], "nn feature norm parameterInfo by name");
layerNormModule.zeroGrad();
expectSame(layerNormModule.biasParam.grad, [0, 0], "nn feature norm zeroGrad");
layerNormModule.requiresGrad_(false);
expectSame(layerNormModule.weightParam.tensor.requiresGrad, false, "nn feature norm requiresGrad false");
layerNormModule.requiresGrad_(true);
expectSame(layerNormModule.eval().training, false, "nn feature norm eval");
expectSame(layerNormModule.train(true).training, true, "nn feature norm train");
const featureNormState = layerNormModule.stateDict();
expectSame(featureNormState.weight.data, [1, 2], "nn feature norm stateDict weight");
layerNormModule.loadStateDict({ weight: { shape: [2], data: [3, 4] }, bias: { shape: [2], data: [5, 6] } });
expectSame({ weight: layerNormModule.weight, bias: layerNormModule.bias }, { weight: [3, 4], bias: [5, 6] }, "nn feature norm loadStateDict");
expectSame(layerNormModule.compileSupport({ mode: "module" }).nativePath, "device-program", "nn feature norm compileSupport");
expectSame(layerNormModule.canCompile({ mode: "module" }), true, "nn feature norm canCompile");
const layerNormBindings = layerNormModule.bindParameters({ mode: "module" });
expectSame({ weights: layerNormBindings.weights, bias: layerNormBindings.bias }, {
  weights: [3, 4],
  bias: [5, 6],
}, "nn feature norm bind parameters");
expectSame(layerNormModule.placeParameters({ name: "program" }, { label: "feature-norm" }), { placed: "layerNorm" }, "nn feature norm place parameters");
expectSame(layerNormModule.compile({ mode: "module" }), { kind: "feature-norm-program", moduleKind: "layerNorm" }, "nn feature norm compile");
const rmsNormModule = new FeatureNormModule("rmsNorm", 2, {
  eps: 0,
  weights: [1, 2],
  bias: [9, 9],
});
expectSame(rmsNormModule.bias, null, "nn feature norm rmsNorm ignores bias");
expectSame(Array.from(rmsNormModule.forward([3, 4, 1, 2]), (value) => Number(value.toFixed(6))), [
  0.848528,
  2.262742,
  0.632456,
  2.529822,
], "nn feature norm rmsNorm forward");
expectSame(rmsNormModule.parameterNames(), ["weight"], "nn feature norm rmsNorm parameter names");
expectSame(rmsNormModule.bindParameters().weights, [1, 2], "nn feature norm rmsNorm bind parameters");
const batchNormModule = new FeatureNormModule("batchNorm1d", 2, {
  eps: 0,
  momentum: 0.5,
  weight: [1, 1],
  bias: [0, 0],
});
expectSame({
  kind: batchNormModule.kind,
  features: batchNormModule.features,
  trackRunningStats: batchNormModule.trackRunningStats,
  runningMean: batchNormModule.runningMean.data,
  runningVar: batchNormModule.runningVar.data,
}, {
  kind: "batchNorm1d",
  features: 2,
  trackRunningStats: true,
  runningMean: [0, 0],
  runningVar: [1, 1],
}, "nn feature norm batchNorm1d initializes");
expectSame(Array.from(batchNormModule.forward([1, 3, 3, 7]), (value) => Number(value.toFixed(6))), [
  -1,
  -1,
  1,
  1,
], "nn feature norm batchNorm1d train forward");
expectSame({
  runningMean: Array.from(batchNormModule.runningMean.data, (value) => Number(value.toFixed(6))),
  runningVar: Array.from(batchNormModule.runningVar.data, (value) => Number(value.toFixed(6))),
  numBatchesTracked: batchNormModule.numBatchesTracked.data[0],
}, {
  runningMean: [1, 2.5],
  runningVar: [1.5, 4.5],
  numBatchesTracked: 1,
}, "nn feature norm batchNorm1d running stats");
const batchNormState = batchNormModule.stateDict();
expectSame(batchNormState.runningMean.data, [1, 2.5], "nn feature norm batchNorm1d stateDict running mean");
batchNormModule.loadStateDict({
  weight: { shape: [2], data: [2, 3] },
  bias: { shape: [2], data: [0.5, -0.5] },
  runningMean: { shape: [2], data: [1, 1] },
  runningVar: { shape: [2], data: [4, 9] },
  numBatchesTracked: { shape: [1], data: [8] },
});
expectSame({
  weight: batchNormModule.weight,
  bias: batchNormModule.bias,
  runningMean: batchNormModule.runningMean.data,
  runningVar: batchNormModule.runningVar.data,
  numBatchesTracked: batchNormModule.numBatchesTracked.data,
}, {
  weight: [2, 3],
  bias: [0.5, -0.5],
  runningMean: [1, 1],
  runningVar: [4, 9],
  numBatchesTracked: [8],
}, "nn feature norm batchNorm1d loadStateDict");
batchNormModule.eval();
expectSame(Array.from(batchNormModule.forward([3, 4]), (value) => Number(value.toFixed(6))), [
  2.5,
  2.5,
], "nn feature norm batchNorm1d eval forward");
expectSame({
  weights: batchNormModule.bindParameters().weights,
  bias: batchNormModule.bindParameters().bias,
}, {
  weights: [1, 1],
  bias: [-0.5, -1.5],
}, "nn feature norm batchNorm1d eval bind parameters");
const noAffineLayerNorm = new FeatureNormModule("layerNorm", 2, { affine: false, eps: 0 });
expectSame(noAffineLayerNorm.parameters(), [], "nn feature norm no-affine parameters");
expectSame(noAffineLayerNorm.bindParameters().weights, [], "nn feature norm no-affine bind parameters");
expectSame(noAffineLayerNorm.forward([1, 2]), [-1, 1], "nn feature norm no-affine forward");
expectSame(featureNormPrograms, [{ compiled: { kind: "module", moduleKind: "layerNorm" }, options: { mode: "module" } }], "nn feature norm compile hook");
expectSame(featureNormPlacements, [{ kind: "layerNorm", program: { name: "program" }, options: { label: "feature-norm" } }], "nn feature norm placement hook");
expectThrow(
  () => new FeatureNormModule("layerNorm", 0),
  "layerNorm features must be a positive safe integer, got 0",
  "nn feature norm rejects features",
);
expectThrow(
  () => new FeatureNormModule("layerNorm", 2, { weight: [1] }),
  "layerNorm weight length must be 2, got 1",
  "nn feature norm rejects weight length",
);
expectThrow(
  () => layerNormModule.forward([1, 2, 3]),
  "layerNorm input length 3 must be divisible by features 2",
  "nn feature norm rejects input length",
);
expectThrow(
  () => layerNormModule.compile({ mode: "unsupported" }),
  "feature norm unsupported",
  "nn feature norm rejects unsupported compile",
);

class NamespaceLinear {
  constructor(inFeatures, outFeatures, config = {}) {
    this.kind = "linear";
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.config = config;
  }

  forward(input) {
    return { kind: "linear-forward", input };
  }

  parameters() {
    return [{ name: "weight" }];
  }

  namedChildren(prefix = "") {
    return [{ name: prefix ? `${prefix}.child` : "child", module: this }];
  }

  namedModules(prefix = "") {
    return [{ name: prefix || "", module: this }];
  }
}
class NamespaceEmbedding {
  constructor(numEmbeddings, embeddingDim, config = {}) {
    this.kind = "embedding";
    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;
    this.config = config;
  }
}
class NamespaceSequential {
  constructor(layers) {
    this.kind = "sequential";
    this.layers = layers;
    this.training = true;
  }

  parameters() {
    return [{ name: "0.weight" }];
  }

  namedParameters(prefix = "") {
    return this.parameters().map((param) => ({ ...param, name: prefix ? `${prefix}.${param.name}` : param.name }));
  }

  parameterNames(prefix = "") {
    return this.parameters().map((param) => prefix ? `${prefix}.${param.name}` : param.name);
  }

  parameterInfos(prefix = "") {
    return this.parameters().map((param, index) => ({ name: prefix ? `${prefix}.${param.name}` : param.name, index }));
  }

  parameterInfo(nameOrIndex, prefix = "") {
    return this.parameterInfos(prefix).find((entry) => entry.name === nameOrIndex || entry.index === nameOrIndex) ?? null;
  }

  forward(input) {
    return { kind: "sequential-forward", input, layers: this.layers };
  }

  children() {
    return this.layers;
  }

  modules() {
    return [this, ...this.layers];
  }

  namedChildren(prefix = "") {
    return this.layers.map((layer, index) => ({ name: prefix ? `${prefix}.${index}` : String(index), module: layer }));
  }

  namedModules(prefix = "") {
    return [{ name: prefix || "", module: this }, ...this.namedChildren(prefix)];
  }

  train(mode = true) {
    this.training = mode;
    for (const layer of this.layers) {
      if (layer && typeof layer === "object") layer.training = mode;
    }
    return this;
  }

  eval() {
    return this.train(false);
  }
}
class NamespaceActivation {
  constructor(kind, fn) {
    this.kind = kind;
    this.fn = fn;
  }
}
class NamespaceSoftmax {
  constructor(dim = -1) {
    this.kind = "softmax";
    this.dim = dim;
  }
}
class NamespaceLogSoftmax {
  constructor(dim = -1) {
    this.kind = "logSoftmax";
    this.dim = dim;
  }
}
class NamespaceReduction {
  constructor(kind, dim = -1) {
    this.kind = kind;
    this.dim = dim;
  }
}
class NamespaceDropout {
  constructor(p = 0.5, config = {}) {
    this.kind = "dropout";
    this.p = p;
    this.config = config;
  }
}
class NamespaceShape {
  constructor(kind, ...args) {
    this.kind = kind;
    this.args = args;
  }
}
class NamespaceFeatureNorm {
  constructor(kind, features, config = {}) {
    this.kind = kind;
    this.features = features;
    this.config = config;
  }
}
class NamespaceConv2d {
  constructor(inChannels, outChannels, kernelSize, config = {}) {
    this.kind = "conv2d";
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelSize;
    this.config = config;
  }
}
class NamespaceMaxPool2d {
  constructor(kernelSize, config = {}) {
    this.kind = "maxPool2d";
    this.kernelSize = kernelSize;
    this.config = config;
  }
}
class NamespaceAvgPool2d {
  constructor(kernelSize, config = {}) {
    this.kind = "avgPool2d";
    this.kernelSize = kernelSize;
    this.config = config;
  }
}
class NamespaceMSELoss {
  constructor() {
    this.kind = "mse-loss";
  }
}
class NamespaceL1Loss {
  constructor() {
    this.kind = "l1-loss";
  }
}
class NamespaceHuberLoss {
  constructor(options = {}) {
    this.kind = "huber-loss";
    this.options = options;
  }
}
class NamespaceSmoothL1Loss {
  constructor(options = {}) {
    this.kind = "smooth-l1-loss";
    this.options = options;
  }
}
class NamespaceBCELoss {
  constructor(options = {}) {
    this.kind = "bce-loss";
    this.options = options;
  }
}
class NamespaceBCEWithLogitsLoss {
  constructor() {
    this.kind = "bce-with-logits-loss";
  }
}
class NamespaceCrossEntropyLoss {
  constructor(options = {}) {
    this.kind = "cross-entropy-loss";
    this.options = options;
  }
}
class NamespaceNLLLoss {
  constructor(options = {}) {
    this.kind = "nll-loss";
    this.options = options;
  }
}
const namespaceHelperCalls = [];
const namespaceCompileTargets = [];
const nnGeneratedNamespace = nnNamespace.createNnNamespace({
  LinearModule: NamespaceLinear,
  EmbeddingModule: NamespaceEmbedding,
  SequentialModule: NamespaceSequential,
  ActivationModule: NamespaceActivation,
  SoftmaxModule: NamespaceSoftmax,
  LogSoftmaxModule: NamespaceLogSoftmax,
  ReductionModule: NamespaceReduction,
  DropoutModule: NamespaceDropout,
  ShapeModule: NamespaceShape,
  FeatureNormModule: NamespaceFeatureNorm,
  Conv2dModule: NamespaceConv2d,
  AvgPool2dModule: NamespaceAvgPool2d,
  MaxPool2dModule: NamespaceMaxPool2d,
  MSELoss: NamespaceMSELoss,
  L1Loss: NamespaceL1Loss,
  HuberLoss: NamespaceHuberLoss,
  SmoothL1Loss: NamespaceSmoothL1Loss,
  BCELoss: NamespaceBCELoss,
  BCEWithLogitsLoss: NamespaceBCEWithLogitsLoss,
  CrossEntropyLoss: NamespaceCrossEntropyLoss,
  NLLLoss: NamespaceNLLLoss,
  geluScalar: (x) => x + 1,
  siluScalar: (x) => x + 2,
  sigmoidScalar: (x) => x + 3,
  f32WithLength: (value, length, label) => {
    const data = value instanceof Float32Array ? value : Float32Array.from(value);
    if (data.length !== length) throw new Error(`${label} length must be ${length}, got ${data.length}`);
    return data;
  },
  makeParameter: (name, data, shape = [data.length], layout = "row-major") => ({ name, data, shape, layout }),
  resolveParameters: (module) => {
    namespaceHelperCalls.push(["parameters", module]);
    return module && typeof module.parameters === "function" ? module.parameters() : [{ name: "p" }];
  },
  parameterNames: () => ["weight"],
  parameterInfos: () => [{ name: "weight" }],
  parameterInfo: () => ({ name: "weight" }),
  zeroGrad: () => "zero",
  setRequiresGrad: (target) => target?.kind === "sequential" ? target : "requires",
  stateDict: () => ({ weight: [1] }),
  loadStateDict: () => "load",
  traceSequentialProgram: () => ({ kind: "trace" }),
  traceModule: () => ({ kind: "trace" }),
  compileSupportForModule: () => ({ supported: true }),
  requireCompileSupportForModule: () => ({ kind: "required-support" }),
  compilerSignaturesForModule: () => ["sig"],
  tensorProgramIrForModule: () => ({ kind: "ir" }),
  kernelPlanForModule: () => ({ kind: "plan" }),
  bufferLayoutForModule: () => ({ input: {} }),
  memoryLayoutForModule: () => ({ bytes: 0 }),
  inputShapeForModule: () => [1],
  outputShapeForModule: () => [2],
  shapeConstraintsForModule: () => ({ static: true }),
  parameterLayoutForModule: () => ({ weight: {} }),
  explainModule: () => ({ reason: "ok" }),
  requireCompilePlanForModule: () => ({ kind: "required-plan" }),
  canCompileModule: () => true,
  compileModule: (target) => namespaceCompileTargets.push(target) && ({ kind: "program", target }),
  bindModuleParameters: (target) => ({ weights: [], target }),
  placeModuleParameters: (target) => ({ placed: true, target }),
  bindingPlanForModuleBindings: () => ({ kind: "binding-plan" }),
  requireBindingPlanForModuleBindings: () => ({ kind: "required-binding-plan" }),
});
expectSame(Object.isFrozen(nnGeneratedNamespace), true, "nn namespace is frozen");
expectSame(nnGeneratedNamespace.Linear, NamespaceLinear, "nn namespace Linear constructor");
expectSame(nnGeneratedNamespace.Conv2d, NamespaceConv2d, "nn namespace Conv2d constructor");
expectSame(nnGeneratedNamespace.AvgPool2d, NamespaceAvgPool2d, "nn namespace AvgPool2d constructor");
expectSame(nnGeneratedNamespace.MaxPool2d, NamespaceMaxPool2d, "nn namespace MaxPool2d constructor");
expectSame(nnGeneratedNamespace.linear(2, 3, { bias: false }), {
  kind: "linear",
  inFeatures: 2,
  outFeatures: 3,
  config: { bias: false },
}, "nn namespace linear factory");
expectSame(nnGeneratedNamespace.conv2d(1, 2, [3, 3], { padding: 1 }), {
  kind: "conv2d",
  inChannels: 1,
  outChannels: 2,
  kernelSize: [3, 3],
  config: { padding: 1 },
}, "nn namespace conv2d factory");
expectSame(nnGeneratedNamespace.avgPool2d(2, { stride: 1 }), {
  kind: "avgPool2d",
  kernelSize: 2,
  config: { stride: 1 },
}, "nn namespace avgPool2d factory");
expectSame(nnGeneratedNamespace.avg_pool2d([2, 2], { count_include_pad: false }), {
  kind: "avgPool2d",
  kernelSize: [2, 2],
  config: { count_include_pad: false },
}, "nn namespace avg_pool2d factory");
expectSame(nnGeneratedNamespace.maxPool2d([2, 2], { stride: 2 }), {
  kind: "maxPool2d",
  kernelSize: [2, 2],
  config: { stride: 2 },
}, "nn namespace maxPool2d factory");
expectSame(nnGeneratedNamespace.max_pool2d(2, { padding: 1 }), {
  kind: "maxPool2d",
  kernelSize: 2,
  config: { padding: 1 },
}, "nn namespace max_pool2d factory");
expectSame(nnGeneratedNamespace.embedding(4, 5).kind, "embedding", "nn namespace embedding factory");
expectSame(nnGeneratedNamespace.sequential(["a"]).layers, ["a"], "nn namespace sequential factory");
expectSame(nnGeneratedNamespace.relu().kind, "relu", "nn namespace relu factory");
expectSame(nnGeneratedNamespace.gelu().fn(2), 3, "nn namespace gelu scalar");
expectSame(nnGeneratedNamespace.silu().fn(2), 4, "nn namespace silu scalar");
expectSame(nnGeneratedNamespace.sigmoid().fn(2), 5, "nn namespace sigmoid scalar");
expectSame(nnGeneratedNamespace.sgn().fn(-1), -1, "nn namespace sgn scalar");
expectSame(nnGeneratedNamespace.sign().kind, "sgn", "nn namespace sign alias");
expectSame(nnGeneratedNamespace.square().fn(3), 9, "nn namespace square scalar");
expectSame(nnGeneratedNamespace.sqr().kind, "square", "nn namespace sqr alias");
expectSame(new nnGeneratedNamespace.MSELoss(), { kind: "mse-loss" }, "nn namespace MSELoss constructor");
expectSame(new nnGeneratedNamespace.L1Loss(), { kind: "l1-loss" }, "nn namespace L1Loss constructor");
expectSame(new nnGeneratedNamespace.HuberLoss({ delta: 0.5 }), {
  kind: "huber-loss",
  options: { delta: 0.5 },
}, "nn namespace HuberLoss constructor");
expectSame(new nnGeneratedNamespace.SmoothL1Loss({ beta: 0.5 }), {
  kind: "smooth-l1-loss",
  options: { beta: 0.5 },
}, "nn namespace SmoothL1Loss constructor");
expectSame(new nnGeneratedNamespace.BCELoss({ eps: 1e-6 }), {
  kind: "bce-loss",
  options: { eps: 1e-6 },
}, "nn namespace BCELoss constructor");
expectSame(new nnGeneratedNamespace.BCEWithLogitsLoss(), {
  kind: "bce-with-logits-loss",
}, "nn namespace BCEWithLogitsLoss constructor");
expectSame(new nnGeneratedNamespace.CrossEntropyLoss({ classes: 2 }), {
  kind: "cross-entropy-loss",
  options: { classes: 2 },
}, "nn namespace CrossEntropyLoss constructor");
expectSame(new nnGeneratedNamespace.NLLLoss({ classes: 2 }), {
  kind: "nll-loss",
  options: { classes: 2 },
}, "nn namespace NLLLoss constructor");
expectSame(nnGeneratedNamespace.softmax(1), { kind: "softmax", dim: 1 }, "nn namespace softmax factory");
expectSame(nnGeneratedNamespace.logSoftmax(0), { kind: "logSoftmax", dim: 0 }, "nn namespace logSoftmax factory");
expectSame(nnGeneratedNamespace.log_softmax(0), { kind: "logSoftmax", dim: 0 }, "nn namespace log_softmax alias");
expectSame(nnGeneratedNamespace.sum(1), { kind: "sum", dim: 1 }, "nn namespace sum factory");
expectSame(nnGeneratedNamespace.mean(0), { kind: "mean", dim: 0 }, "nn namespace mean factory");
expectSame(nnGeneratedNamespace.max(-1), { kind: "max", dim: -1 }, "nn namespace max factory");
expectSame(nnGeneratedNamespace.dropout(0.25, { seed: 7 }), { kind: "dropout", p: 0.25, config: { seed: 7 } }, "nn namespace dropout factory");
expectSame(nnGeneratedNamespace.identity(), { kind: "identity", args: [] }, "nn namespace identity factory");
expectSame(nnGeneratedNamespace.reshape([2, 3]), { kind: "reshape", args: [[2, 3]] }, "nn namespace reshape factory");
expectSame(nnGeneratedNamespace.flatten(1, 2), { kind: "flatten", args: [1, 2] }, "nn namespace flatten factory");
expectSame(nnGeneratedNamespace.squeeze(), { kind: "squeeze", args: [null] }, "nn namespace squeeze default");
expectSame(nnGeneratedNamespace.unsqueeze(0), { kind: "unsqueeze", args: [0] }, "nn namespace unsqueeze factory");
expectSame(nnGeneratedNamespace.transpose(0, 1), { kind: "transpose", args: [0, 1] }, "nn namespace transpose factory");
expectSame(nnGeneratedNamespace.permute([1, 0]), { kind: "permute", args: [[1, 0]] }, "nn namespace permute factory");
expectSame(nnGeneratedNamespace.broadcastTo([2]), { kind: "broadcastTo", args: [[2]] }, "nn namespace broadcastTo factory");
expectSame(nnGeneratedNamespace.expand([3]), { kind: "expand", args: [[3]] }, "nn namespace expand factory");
expectSame(nnGeneratedNamespace.narrow(1, 2, 3), { kind: "narrow", args: [1, 2, 3] }, "nn namespace narrow factory");
expectSame(nnGeneratedNamespace.select(1, 0), { kind: "select", args: [1, 0] }, "nn namespace select factory");
expectSame(nnGeneratedNamespace.slice(1, 2, 3, 4), { kind: "slice", args: [1, 2, 3, 4] }, "nn namespace slice factory");
expectSame(new nnGeneratedNamespace.LayerNorm(8, { eps: 1e-6 }), {
  kind: "layerNorm",
  features: 8,
  config: { eps: 0.000001 },
}, "nn namespace LayerNorm constructor");
expectSame(new nnGeneratedNamespace.RMSNorm(8), { kind: "rmsNorm", features: 8, config: {} }, "nn namespace RMSNorm constructor");
expectSame(nnGeneratedNamespace.layerNorm(4).kind, "layerNorm", "nn namespace layerNorm factory");
expectSame(nnGeneratedNamespace.layer_norm(4).kind, "layerNorm", "nn namespace layer_norm alias");
expectSame(nnGeneratedNamespace.rmsNorm(4).kind, "rmsNorm", "nn namespace rmsNorm factory");
expectSame(nnGeneratedNamespace.rms_norm(4).kind, "rmsNorm", "nn namespace rms_norm alias");
expectSame(nnGeneratedNamespace.parameters({ id: 1 }), [{ name: "p" }], "nn namespace parameters helper");
expectSame(nnGeneratedNamespace.namedParameters({ id: 2 }), [{ name: "p" }], "nn namespace namedParameters alias");
expectSame(nnGeneratedNamespace.namedParameters({ id: 3 }, "model"), [{ name: "model.p" }], "nn namespace prefixed namedParameters helper");
expectSame(namespaceHelperCalls, [["parameters", { id: 1 }], ["parameters", { id: 2 }], ["parameters", { id: 3 }]], "nn namespace parameters delegates");
expectSame(nnGeneratedNamespace.parameterNames(), ["weight"], "nn namespace parameterNames helper");
expectSame(nnGeneratedNamespace.parameterInfos(), [{ name: "weight" }], "nn namespace parameterInfos helper");
expectSame(nnGeneratedNamespace.parameterInfo(), { name: "weight" }, "nn namespace parameterInfo helper");
const namespaceMetadataModule = {
  parameterNames: (prefix = "") => [prefix ? `${prefix}.custom` : "custom"],
  parameterInfos: (prefix = "") => Object.freeze([{ name: prefix ? `${prefix}.custom` : "custom", custom: true }]),
  parameterInfo: (nameOrIndex, prefix = "") => ({ nameOrIndex, prefix, custom: true }),
  namedChildren: (prefix = "") => [{ name: prefix ? `${prefix}.child` : "child" }],
  namedModules: (prefix = "") => [{ name: prefix || "" }],
};
expectSame(nnGeneratedNamespace.parameterNames(namespaceMetadataModule, "model"), ["model.custom"], "nn namespace delegates module parameterNames");
expectSame(nnGeneratedNamespace.parameterInfos(namespaceMetadataModule, "model"), [{ name: "model.custom", custom: true }], "nn namespace delegates module parameterInfos");
expectSame(nnGeneratedNamespace.parameterInfo(namespaceMetadataModule, "model.custom", "model"), {
  nameOrIndex: "model.custom",
  prefix: "model",
  custom: true,
}, "nn namespace delegates module parameterInfo");
const namespaceAliasModule = new nnGeneratedNamespace.Linear(2, 3);
expectSame(namespaceAliasModule.named_children("model").map((entry) => entry.name), ["model.child"], "nn module named_children alias");
expectSame(namespaceAliasModule.named_modules("model").map((entry) => entry.name), ["model"], "nn module named_modules alias");
expectSame(namespaceAliasModule.requireCompileSupport(), { kind: "required-support" }, "nn module requireCompileSupport method");
expectSame(namespaceAliasModule.require_compile_support(), { kind: "required-support" }, "nn module require_compile_support alias");
expectSame(namespaceAliasModule.requireCompilePlan(), { kind: "required-plan" }, "nn module requireCompilePlan method");
expectSame(namespaceAliasModule.require_compile_plan(), { kind: "required-plan" }, "nn module require_compile_plan alias");
expectSame(namespaceAliasModule.assertCompilePlan(), { kind: "required-plan" }, "nn module assertCompilePlan alias");
expectSame(namespaceAliasModule.assert_compile_plan(), { kind: "required-plan" }, "nn module assert_compile_plan alias");
expectSame(nnGeneratedNamespace.named_children(namespaceMetadataModule, "model"), [{ name: "model.child" }], "nn namespace named_children alias");
expectSame(nnGeneratedNamespace.named_modules(namespaceMetadataModule, "model"), [{ name: "model" }], "nn namespace named_modules alias");
expectSame(nnGeneratedNamespace.zeroGrad(), "zero", "nn namespace zeroGrad helper");
expectSame(nnGeneratedNamespace.requiresGrad(), "requires", "nn namespace requiresGrad helper");
expectSame(nnGeneratedNamespace.requiresGrad_(), "requires", "nn namespace requiresGrad_ helper");
expectSame(nnGeneratedNamespace.stateDict(), { weight: [1] }, "nn namespace stateDict helper");
expectSame(nnGeneratedNamespace.loadStateDict(), "load", "nn namespace loadStateDict helper");
expectSame(nnGeneratedNamespace.trace(), { kind: "trace" }, "nn namespace trace helper");
expectSame(nnGeneratedNamespace.compileSupport(), { supported: true }, "nn namespace compileSupport helper");
expectSame(nnGeneratedNamespace.requireCompileSupport(), { kind: "required-support" }, "nn namespace requireCompileSupport helper");
expectSame(nnGeneratedNamespace.require_compile_support(), { kind: "required-support" }, "nn namespace require_compile_support helper");
expectSame(nnGeneratedNamespace.compilerSignatures(), ["sig"], "nn namespace compilerSignatures helper");
expectSame(nnGeneratedNamespace.tensorProgramIr(), { kind: "ir" }, "nn namespace tensorProgramIr helper");
expectSame(nnGeneratedNamespace.kernelPlan(), { kind: "plan" }, "nn namespace kernelPlan helper");
expectSame(nnGeneratedNamespace.bufferLayout(), { input: {} }, "nn namespace bufferLayout helper");
expectSame(nnGeneratedNamespace.memoryLayout(), { bytes: 0 }, "nn namespace memoryLayout helper");
expectSame(nnGeneratedNamespace.inputShape(), [1], "nn namespace inputShape helper");
expectSame(nnGeneratedNamespace.outputShape(), [2], "nn namespace outputShape helper");
expectSame(nnGeneratedNamespace.shapeConstraints(), { static: true }, "nn namespace shapeConstraints helper");
expectSame(nnGeneratedNamespace.parameterLayout(), { weight: {} }, "nn namespace parameterLayout helper");
expectSame(nnGeneratedNamespace.explain(), { reason: "ok" }, "nn namespace explain helper");
expectSame(nnGeneratedNamespace.requireCompilePlan(), { kind: "required-plan" }, "nn namespace requireCompilePlan helper");
expectSame(nnGeneratedNamespace.require_compile_plan(), { kind: "required-plan" }, "nn namespace require_compile_plan alias");
expectSame(nnGeneratedNamespace.assertCompilePlan(), { kind: "required-plan" }, "nn namespace assertCompilePlan alias");
expectSame(nnGeneratedNamespace.assert_compile_plan(), { kind: "required-plan" }, "nn namespace assert_compile_plan alias");
expectSame(nnGeneratedNamespace.canCompile(), true, "nn namespace canCompile helper");
expectSame(nnGeneratedNamespace.compile().kind, "program", "nn namespace compile helper");
expectSame(nnGeneratedNamespace.bindParameters().weights, [], "nn namespace bindParameters helper");
expectSame(nnGeneratedNamespace.placeParameters().placed, true, "nn namespace placeParameters helper");
const namespaceRawLayerList = [new nnGeneratedNamespace.Linear(2, 3)];
expectSame(nnGeneratedNamespace.parameters(namespaceRawLayerList), [{ name: "0.weight" }], "nn namespace parameters accepts raw Sequential layer lists");
expectSame(nnGeneratedNamespace.namedParameters(namespaceRawLayerList, "model"), [{ name: "model.0.weight" }], "nn namespace namedParameters accepts raw Sequential layer lists");
expectSame(nnGeneratedNamespace.parameterNames(namespaceRawLayerList, "model"), ["model.0.weight"], "nn namespace parameterNames accepts raw Sequential layer lists");
expectSame(nnGeneratedNamespace.parameterInfos(namespaceRawLayerList, "model"), [{ name: "model.0.weight", index: 0 }], "nn namespace parameterInfos accepts raw Sequential layer lists");
expectSame(nnGeneratedNamespace.parameterInfo(namespaceRawLayerList, "model.0.weight", "model"), { name: "model.0.weight", index: 0 }, "nn namespace parameterInfo accepts raw Sequential layer lists");
expectSame(nnGeneratedNamespace.requiresGrad(namespaceRawLayerList, false).training, true, "nn namespace requiresGrad accepts raw Sequential layer lists");
expectSame(nnGeneratedNamespace.stateDict(["plain-param-array"]), { weight: [1] }, "nn namespace stateDict leaves plain arrays as parameter arrays");
expectSame(nnGeneratedNamespace.forward(["layer"], "input"), {
  kind: "sequential-forward",
  input: "input",
  layers: ["layer"],
}, "nn namespace forwards raw Sequential layer lists through Sequential");
expectSame(nnGeneratedNamespace.children(["layer"]), ["layer"], "nn namespace children accepts raw Sequential layer lists");
expectSame(nnGeneratedNamespace.modules(["layer"]).map((entry) => entry.kind ?? entry), ["sequential", "layer"], "nn namespace modules accepts raw Sequential layer lists");
expectSame(nnGeneratedNamespace.namedChildren(["layer"], "model"), [{ name: "model.0", module: "layer" }], "nn namespace namedChildren accepts raw Sequential layer lists");
expectSame(nnGeneratedNamespace.namedModules(["layer"], "model").map((entry) => entry.name), ["model", "model.0"], "nn namespace namedModules accepts raw Sequential layer lists");
expectSame(nnGeneratedNamespace.train([{ kind: "layer" }], false).training, false, "nn namespace trains raw Sequential layer lists through Sequential");
expectSame(nnGeneratedNamespace.eval([{ kind: "layer" }]).training, false, "nn namespace evals raw Sequential layer lists through Sequential");
expectSame(nnGeneratedNamespace.compile(["layer"]).target, { kind: "sequential", layers: ["layer"], training: true }, "nn namespace compiles raw Sequential layer lists through Sequential");
expectSame(nnGeneratedNamespace.bindParameters(["layer"]).target, { kind: "sequential", layers: ["layer"], training: true }, "nn namespace binds raw Sequential layer lists through Sequential");
expectSame(nnGeneratedNamespace.placeParameters(["layer"], { kind: "program" }).target, { kind: "sequential", layers: ["layer"], training: true }, "nn namespace places raw Sequential layer list parameters through Sequential");
expectSame(nnGeneratedNamespace.bindingPlan(), { kind: "binding-plan" }, "nn namespace bindingPlan helper");
expectSame(nnGeneratedNamespace.requireBindingPlan(), { kind: "required-binding-plan" }, "nn namespace requireBindingPlan helper");
expectThrow(
  () => nnNamespace.createNnNamespace({}),
  "nn namespace factory requires module constructors, loss constructors, state helpers, scalar activations, and compile helpers",
  "nn namespace rejects missing hooks",
);

const lossTrainHelpers = training.createLossTrainHelpers({
  Tensor: ModuleStateSmokeTensor,
  f32(value) {
    return value instanceof Float32Array ? value : Float32Array.from(value);
  },
  f32WithLength(value, length, label) {
    const data = value instanceof Float32Array ? value : Float32Array.from(value);
    if (data.length !== length) throw new Error(`${label} length must be ${length}, got ${data.length}`);
    return data;
  },
  indexValues(value) {
    return Array.from(value);
  },
  addTensorGrad(tensor, grad) {
    if (!tensor.grad) tensor.grad = new Float32Array(grad.length);
    for (let i = 0; i < grad.length; i += 1) tensor.grad[i] += grad[i];
  },
  zeroGrad: moduleStateHelpers.zeroGrad,
  resolveParameters: moduleStateHelpers.resolveParameters,
  isGradEnabled: () => true,
});
expectSame(lossTrainHelpers.meanSquaredError([1, 3], [2, 5]), 2.5, "training mse arrays");
expectSame(lossTrainHelpers.meanAbsoluteError([1, 3], [2, 5]), 1.5, "training l1 arrays");
expectSame(lossTrainHelpers.huber([1, 3], [2, 5]), 1, "training huber arrays");
expectSame(lossTrainHelpers.smoothL1([1, 3], [2, 5]), 1, "training smooth l1 arrays");
expectApprox(lossTrainHelpers.binaryCrossEntropy([0.25, 0.75], [0, 1]), 0.2876820724517809, 1e-12, "training bce arrays");
expectApprox(lossTrainHelpers.binaryCrossEntropyWithLogits([0, 2], [0, 1]), 0.41003759580145885, 1e-12, "training bce with logits arrays");
expectSame(lossTrainHelpers.classTargets([1, 0]), [1, 0], "training class targets");
expectApprox(lossTrainHelpers.crossEntropy([1, 3, 2, 0], [1, 0], { classes: 2 }), 0.126928011043, 1e-9, "training cross entropy arrays");
expectApprox(lossTrainHelpers.cross_entropy([1, 3, 2, 0], [1, 0], { numClasses: 2 }), 0.126928011043, 1e-9, "training cross_entropy alias arrays");
expectApprox(lossTrainHelpers.nllLoss([-2, -0.1, -0.3, -1], [1, 0], { classes: 2 }), 0.2, 1e-7, "training nll loss arrays");
expectApprox(lossTrainHelpers.negative_log_likelihood([-2, -0.1, -0.3, -1], [1, 0], { classes: 2 }), 0.2, 1e-7, "training negative_log_likelihood alias arrays");
expectApprox(lossTrainHelpers.nll_loss([-2, -0.1, -0.3, -1], [1, 0], { numClasses: 2 }), 0.2, 1e-7, "training nll_loss alias arrays");
expectSame(lossTrainHelpers.train.accuracy([1, 3, 2, 0], [1, 0], { classes: 2 }), 1, "training classification accuracy");
expectSame(lossTrainHelpers.train.classification_accuracy([1, 3, 2, 0], [1, 1], { classes: 2 }), 0.5, "training classification accuracy alias");
expectSame(lossTrainHelpers.train.topKAccuracy([3, 2, 1, 1, 3, 2], [1, 2], { classes: 3, k: 2 }), 1, "training top-k accuracy");
expectSame(lossTrainHelpers.train.top_k_accuracy([3, 2, 1, 1, 3, 2], [1, 2], { classes: 3, top_k: 1 }), 0, "training top-k accuracy alias");
expectSame(lossTrainHelpers.train.confusionMatrix([5, 1, 1, 5, 5, 1], [0, 0, 1], { classes: 2 }), [[1, 1], [1, 0]], "training confusion matrix");
expectSame(lossTrainHelpers.train.confusion_matrix([5, 1, 1, 5, 5, 1], [0, 0, 1], { classes: 2 }), [[1, 1], [1, 0]], "training confusion matrix alias");
const classificationReport = lossTrainHelpers.train.classificationReport([5, 1, 1, 5, 5, 1], [0, 0, 1], { classes: 2 });
expectApprox(classificationReport.accuracy, 1 / 3, 1e-12, "training classification report accuracy");
expectApprox(classificationReport.macroF1, 0.25, 1e-12, "training classification report macro f1");
expectApprox(classificationReport.weighted_f1, 1 / 3, 1e-12, "training classification report weighted f1 alias");
expectSame(classificationReport.perClass[0].support, 2, "training classification report class support");
expectSame(lossTrainHelpers.train.classification_report([5, 1, 1, 5, 5, 1], [0, 0, 1], { classes: 2 }).confusion_matrix, [[1, 1], [1, 0]], "training classification report alias");
expectSame(Array.from(lossTrainHelpers.train.classPredictions([0.1, 0.9, 2, 1], { classes: 2 })), [1, 0], "training class predictions");
expectSame(Array.from(lossTrainHelpers.train.predict_classes([0.1, 0.9, 2, 1], { numClasses: 2 })), [1, 0], "training class predictions alias");
expectSame(lossTrainHelpers.train.binaryAccuracy([0.1, 0.9, 0.7, 0.2], [0, 1, 1, 0]), 1, "training binary accuracy");
expectSame(lossTrainHelpers.train.binary_accuracy([0.1, 0.9, 0.4, 0.2], [0, 1, 1, 0]), 0.75, "training binary accuracy alias");
expectSame(lossTrainHelpers.train.binaryLogitsAccuracy([-2, 3, 1, -1], [0, 1, 1, 0]), 1, "training binary logits accuracy");
expectSame(lossTrainHelpers.train.binary_logits_accuracy([-2, 3, -1, -1], [0, 1, 1, 0]), 0.75, "training binary logits accuracy alias");
const trainEvaluateSteps = [];
const trainEvaluateEvidence = lossTrainHelpers.train.evaluate([
  { input: new ModuleStateSmokeTensor([1], [1]), target: new ModuleStateSmokeTensor([2], [1]), indices: [4] },
  { input: new ModuleStateSmokeTensor([3], [1]), target: new ModuleStateSmokeTensor([4], [1]), indices: [5] },
], (batch, context) => {
  expectSame(Object.isFrozen(context), true, "training evaluate freezes context");
  expectSame(context.sampleIndices, context.sample_indices, "training evaluate aliases sample indices");
  expectSame(context.sampleIndices, [context.batchIndex + 4], "training evaluate carries sample indices");
  void batch;
  return new ModuleStateSmokeTensor(Float32Array.of(1), [1]);
}, {
  maxSteps: 1,
  onStep: (evidence) => trainEvaluateSteps.push(evidence),
});
expectSame(lossTrainHelpers.train.isTrainEvaluateEvidence(trainEvaluateEvidence), true, "training evaluate evidence validator");
expectSame(lossTrainHelpers.train.requireTrainEvaluateEvidence(trainEvaluateEvidence).steps, 1, "training evaluate evidence require");
expectSame(lossTrainHelpers.train.assert_train_evaluate_evidence(trainEvaluateEvidence).stoppedEarly, true, "training evaluate evidence assert alias");
expectSame(lossTrainHelpers.train.matchesTrainEvaluateEvidenceSignature(trainEvaluateEvidence, trainEvaluateEvidence.signature), true, "training evaluate evidence signature");
expectSame(lossTrainHelpers.train.isTrainEvaluateEvidence({ ...trainEvaluateEvidence }), false, "training evaluate evidence rejects unfrozen clone");
expectSame(trainEvaluateSteps.length, 1, "training evaluate step callback count");
expectSame(lossTrainHelpers.train.isTrainEvaluateStepEvidence(trainEvaluateSteps[0]), true, "training evaluate step evidence validator");
expectSame(lossTrainHelpers.train.requireTrainEvaluateStepEvidence(trainEvaluateSteps[0]).step, 1, "training evaluate step evidence require");
expectSame(lossTrainHelpers.train.assert_train_evaluate_step_evidence(trainEvaluateSteps[0]).loss, 1, "training evaluate step evidence assert alias");
expectSame(lossTrainHelpers.train.matchesTrainEvaluateStepEvidenceSignature(trainEvaluateSteps[0], trainEvaluateSteps[0].signature), true, "training evaluate step evidence signature");
expectSame(lossTrainHelpers.train.evaluate_loss([
  { input: new ModuleStateSmokeTensor([1], [1]), target: new ModuleStateSmokeTensor([1], [1]) },
], () => new ModuleStateSmokeTensor(Float32Array.of(0), [1]), { max_steps: 1 }).meanLoss, 0, "training evaluate_loss alias");
const trainPredictSteps = [];
const trainPredictEvidence = lossTrainHelpers.train.predict([
  { input: new ModuleStateSmokeTensor([1], [1]), indices: [7] },
  { input: new ModuleStateSmokeTensor([2], [1]), indices: [8] },
], (batch, context) => {
  expectSame(Object.isFrozen(context), true, "training predict freezes context");
  expectSame(context.sampleIndices, context.sample_indices, "training predict aliases sample indices");
  expectSame(context.sampleIndices, [7], "training predict carries sample indices");
  return batch.input;
}, {
  maxSteps: 1,
  onStep: (evidence) => trainPredictSteps.push(evidence),
});
expectSame({
  frozen: Object.isFrozen(trainPredictEvidence),
  kind: trainPredictEvidence.kind,
  steps: trainPredictEvidence.steps,
  stopped: trainPredictEvidence.stoppedEarly,
  outputsFrozen: Object.isFrozen(trainPredictEvidence.outputs),
  outputCount: trainPredictEvidence.outputs.length,
  lastAlias: trainPredictEvidence.lastOutput === trainPredictEvidence.last_output,
  stepKind: trainPredictSteps[0].kind,
  stepFrozen: Object.isFrozen(trainPredictSteps[0]),
  stepOutput: trainPredictSteps[0].output.data,
}, {
  frozen: true,
  kind: "zgml.train.predict",
  steps: 1,
  stopped: true,
  outputsFrozen: true,
  outputCount: 1,
  lastAlias: true,
  stepKind: "zgml.train.predict-step",
  stepFrozen: true,
  stepOutput: [1],
}, "training predict evidence and step callback");
expectSame(lossTrainHelpers.train.predict_batches([
  { input: new ModuleStateSmokeTensor([3], [1]) },
], (batch) => batch.input, { max_steps: 1 }).outputs[0].data, [3], "training predict_batches alias");
const trainEvidenceOptimizer = {
  params: [],
  stepCount: 0,
  stateDict() {
    return { kind: "sgd", step: this.stepCount, paramCount: 0 };
  },
  step() {
    this.stepCount += 1;
  },
  zeroGrad() {},
};
const trainStepEvidence = lossTrainHelpers.train.step(trainEvidenceOptimizer, { inspect: true });
expectSame(trainStepEvidence, {
  kind: "zgml.train.step",
  optimizerKind: "sgd",
  parameterCount: 0,
  beforeStep: 0,
  afterStep: 1,
  stepAdvanced: 1,
  hadLoss: false,
  lossScalar: null,
  gradientProvided: false,
  clipGradNormApplied: false,
  clipGradValueApplied: false,
  gradNormBeforeClip: null,
  gradNormAfterClip: null,
  zeroGradApplied: true,
  gradientsCleared: null,
  signature: "train-step|optimizer=sgd|parameters=0|before=0|after=1|advanced=1|loss=0|lossScalar=null|gradient=0|clipNorm=0|clipValue=0|gradNormBeforeClip=null|gradNormAfterClip=null|zeroGrad=1|cleared=null",
}, "training step evidence");
const trainFitEarlyStopEvidence = lossTrainHelpers.train.fit(trainEvidenceOptimizer, [
  { input: new ModuleStateSmokeTensor([1], [1]) },
  { input: new ModuleStateSmokeTensor([2], [1]) },
], () => Object.assign(new ModuleStateSmokeTensor([1], [1]), { backward() {} }), {
  epochs: 3,
  early_stopping: { patience: 0, min_delta: 0, mode: "min" },
});
expectSame({
  valid: lossTrainHelpers.train.isTrainFitEvidence(trainFitEarlyStopEvidence),
  steps: trainFitEarlyStopEvidence.steps,
  stopped: trainFitEarlyStopEvidence.stoppedEarly,
  reason: trainFitEarlyStopEvidence.stop_reason,
  bestLoss: trainFitEarlyStopEvidence.best_loss,
  bestStep: trainFitEarlyStopEvidence.best_step,
}, {
  valid: true,
  steps: 2,
  stopped: true,
  reason: "early-stopping",
  bestLoss: 1,
  bestStep: 1,
}, "training fit early stopping evidence");
expectSame(lossTrainHelpers.train.isTrainStepEvidence(trainStepEvidence), true, "training step evidence namespace predicate");
expectSame(trainNamespacePolicy.isTrainStepEvidence(trainStepEvidence), true, "training step evidence standalone predicate");
expectSame(trainNamespacePolicy.requireTrainStepEvidence(trainStepEvidence).afterStep, 1, "training step evidence standalone require");
expectSame(trainNamespacePolicy.assertTrainStepEvidence(trainStepEvidence).optimizerKind, "sgd", "training step evidence standalone assert");
expectSame(trainNamespacePolicy.assert_train_step_evidence(trainStepEvidence).signature, trainStepEvidence.signature, "training step evidence snake assert");
expectSame(trainNamespacePolicy.matchesTrainStepEvidenceSignature(trainStepEvidence, trainStepEvidence.signature), true, "training step evidence signature");
expectSame(trainNamespacePolicy.matches_train_step_evidence_signature(trainStepEvidence, "wrong"), false, "training step evidence snake signature mismatch");
expectSame(trainNamespacePolicy.isTrainStepEvidence({ ...trainStepEvidence }), false, "training step evidence rejects mutable record");
expectThrow(
  () => trainNamespacePolicy.requireTrainStepEvidence(Object.freeze({ ...trainStepEvidence, signature: "wrong" })),
  "expected frozen TrainStepEvidence",
  "training step evidence rejects mismatched signature",
);

const optimizerClasses = training.createOptimizerClasses({
  resolveParameters: moduleStateHelpers.resolveParameters,
  finiteConfigNumber: moduleStateHelpers.finiteConfigNumber,
  zeroGrad: moduleStateHelpers.zeroGrad,
  optimizerStateEntry: moduleStateHelpers.optimizerStateEntry,
  optimizerStateDict: moduleStateHelpers.optimizerStateDict,
  optimizerStateEntries: moduleStateHelpers.optimizerStateEntries,
  optimizerStepFromState: moduleStateHelpers.optimizerStepFromState,
  loadOptimizerTensorState: moduleStateHelpers.loadOptimizerTensorState,
  rejectUnexpectedOptimizerState: moduleStateHelpers.rejectUnexpectedOptimizerState,
});
const optimNamespace = training.createOptimNamespace({
  ...optimizerClasses,
  zeroGrad: moduleStateHelpers.zeroGrad,
});
const trainingParam = moduleStateHelpers.makeParameter("weight", Float32Array.of(1, -1), [2]);
trainingParam.grad.set([0.5, -0.25]);
const sgd = optimNamespace.sgd([trainingParam], { lr: 0.1, momentum: 0.5 });
sgd.step();
expectSame(trainingParam.data, [0.949999988079071, -0.9750000238418579], "training sgd update");
const sgdState = optimNamespace.stateDict(sgd);
expectSame(sgdState.kind, "sgd", "training sgd state kind");
expectSame(sgdState.signature.startsWith("optimizer-state|kind=sgd|step=1|params=1|"), true, "training sgd state signature");
expectSame(sgdState.step, 1, "training sgd state step");
expectSame(optimNamespace.config(sgd).signature.startsWith("optimizer-config|kind=sgd|"), true, "training sgd config signature");
expectSame(optimNamespace.config(sgd).step, 1, "training sgd config step");
const sgd2 = optimNamespace.sgd([trainingParam], { lr: 0.1, momentum: 0.5 });
optimNamespace.loadStateDict(sgd2, sgdState);
expectSame(sgd2.velocity[0], [0.5, -0.25], "training sgd load state");
expectSame(sgd2.stateDict().step, 1, "training sgd load step");

const checkpointHelpers = training.createCheckpointHelpers({
  moduleStateDict: moduleStateHelpers.stateDict,
  loadModuleStateDict: moduleStateHelpers.loadStateDict,
  optimizerStateDict: optimNamespace.stateDict,
  loadOptimizerStateDict: optimNamespace.loadStateDict,
});
const checkpointSnapshot = checkpointHelpers.create({
  metadata: { epoch: 1 },
  model: [trainingParam],
  optimizer: sgd,
});
const checkpointInfo = checkpointHelpers.inspect(checkpointSnapshot);
expectSame(checkpointInfo.signature.startsWith("checkpoint-inspection|format=zgml.checkpoint|version=1|"), true, "training checkpoint inspection signature");
expectSame(checkpointInfo.modelParameters[0].signature.startsWith("checkpoint-tensor-inspection|name=weight|"), true, "training checkpoint tensor inspection signature");
expectSame({
  hasModel: checkpointInfo.hasModel,
  hasOptimizer: checkpointInfo.hasOptimizer,
  modelParameterNames: checkpointInfo.modelParameterNames,
  optimizerEntryNames: checkpointInfo.optimizerEntryNames,
}, {
  hasModel: true,
  hasOptimizer: true,
  modelParameterNames: ["weight"],
  optimizerEntryNames: ["velocity.0"],
}, "training checkpoint inspect");
trainingParam.data.set([9, 9]);
sgd.velocity[0].set([9, 9]);
checkpointHelpers.restore(checkpointSnapshot, { model: [trainingParam], optimizer: sgd });
expectSame(trainingParam.data, [0.949999988079071, -0.9750000238418579], "training checkpoint restores model");
expectSame(sgd.velocity[0], [0.5, -0.25], "training checkpoint restores optimizer");

class TensorRootOpsSmokeTensor {
  constructor(data, shape = [Array.isArray(data) ? data.length : 1], options = {}) {
    this.data = data;
    this.shape = shape;
    this.options = options;
    this.calls = [];
  }
  record(name, args) {
    this.calls.push({ name, args });
    return new TensorRootOpsSmokeTensor([name], [1], { from: name, args });
  }
  allclose(other, options) {
    this.calls.push({ name: "allclose", args: [other, options] });
    return other === "expected" && options.atol === 0.25;
  }
  equal(other) {
    this.calls.push({ name: "equal", args: [other] });
    return other === "same";
  }
  add(other) {
    return this.record("add", [other]);
  }
  sub(other) {
    return this.record("sub", [other]);
  }
  mul(other) {
    return this.record("mul", [other]);
  }
  div(other) {
    return this.record("div", [other]);
  }
  pow(exponent) {
    return this.record("pow", [exponent]);
  }
  maximum(other) {
    return this.record("maximum", [other]);
  }
  minimum(other) {
    return this.record("minimum", [other]);
  }
  where(input, other) {
    return this.record("where", [input, other]);
  }
  any(dim) {
    return this.record("any", [dim]);
  }
  all(dim) {
    return this.record("all", [dim]);
  }
  clamp(min, max) {
    return this.record("clamp", [min, max]);
  }
  clip(min, max) {
    return this.record("clip", [min, max]);
  }
}

const tensorRootFacadeCalls = [];
const tensorRootOps = createTensorRootOps({
  Tensor: TensorRootOpsSmokeTensor,
  tensorFacade: {
    tensor(data, shape, options) {
      tensorRootFacadeCalls.push({ name: "tensor", args: [data, shape, options] });
      return new TensorRootOpsSmokeTensor(data, shape, options);
    },
    parameter(data, shape, options) {
      tensorRootFacadeCalls.push({ name: "parameter", args: [data, shape, options] });
      return new TensorRootOpsSmokeTensor(data, shape, { ...options, parameter: true });
    },
    cat(tensors, dim) {
      tensorRootFacadeCalls.push({ name: "cat", args: [tensors.length, dim] });
      return new TensorRootOpsSmokeTensor(["cat"], [tensors.length], { dim });
    },
    stack(tensors, dim) {
      tensorRootFacadeCalls.push({ name: "stack", args: [tensors.length, dim] });
      return new TensorRootOpsSmokeTensor(["stack"], [tensors.length], { dim });
    },
  },
});
expectSame(tensorRoot.tensorRootManifest.policyOwner, "src/ts/core/tensor_root.ts", "core tensor root manifest owner");
expectSame(Object.isFrozen(tensorRootOps), true, "core tensor root ops frozen");
expectSame(tensorRootOps.tensor([1, 2], [2], { requiresGrad: true }).options, { requiresGrad: true }, "core tensor root tensor delegates");
expectSame(tensorRootOps.parameter([3], { requiresGrad: true }).options, { parameter: true }, "core tensor root parameter shape/options overload delegates");
expectSame(tensorRootOps.cat([new TensorRootOpsSmokeTensor([1]), new TensorRootOpsSmokeTensor([2])], 1).options, { dim: 1 }, "core tensor root cat delegates");
expectSame(tensorRootOps.concat([new TensorRootOpsSmokeTensor([1]), new TensorRootOpsSmokeTensor([2])], 1).options, { dim: 1 }, "core tensor root concat delegates");
expectSame(tensorRootOps.concatenate([new TensorRootOpsSmokeTensor([1]), new TensorRootOpsSmokeTensor([2])], 1).options, { dim: 1 }, "core tensor root concatenate delegates");
expectSame(tensorRootOps.stack([new TensorRootOpsSmokeTensor([1])]).options, { dim: 0 }, "core tensor root stack default dim");
expectSame(tensorRootOps.allclose(new TensorRootOpsSmokeTensor([1]), "expected", { atol: 0.25 }), true, "core tensor root allclose instance delegates");
expectSame(tensorRootOps.equal(new TensorRootOpsSmokeTensor([1]), "same"), true, "core tensor root equal instance delegates");
expectSame(tensorRootOps.hasShape(new TensorRootOpsSmokeTensor([1, 2], [2]), [2]), true, "core tensor root hasShape accepts matching shape");
expectSame(tensorRootOps.hasShape(new TensorRootOpsSmokeTensor([1, 2], [2]), [1, 2]), false, "core tensor root hasShape rejects mismatched shape");
expectSame(tensorRootOps.requireShape(new TensorRootOpsSmokeTensor([1, 2], [2]), [2]).shape, [2], "core tensor root requireShape returns tensor");
expectThrow(
  () => tensorRootOps.requireShape(new TensorRootOpsSmokeTensor([1, 2], [2]), [1, 2]),
  "Tensor.requireShape expected [1, 2], got [2]",
  "core tensor root requireShape mismatch",
);
expectSame(tensorRootOps.add(new TensorRootOpsSmokeTensor([1]), [2]).options, { from: "add", args: [[2]] }, "core tensor root add delegates");
expectSame(tensorRootOps.sub(new TensorRootOpsSmokeTensor([1]), [2]).options, { from: "sub", args: [[2]] }, "core tensor root sub delegates");
expectSame(tensorRootOps.mul(new TensorRootOpsSmokeTensor([1]), [2]).options, { from: "mul", args: [[2]] }, "core tensor root mul delegates");
expectSame(tensorRootOps.div(new TensorRootOpsSmokeTensor([1]), [2]).options, { from: "div", args: [[2]] }, "core tensor root div delegates");
expectSame(tensorRootOps.pow([2], 3).options, { from: "pow", args: [3] }, "core tensor root coerces non-tensor input");
expectSame(tensorRootOps.maximum(new TensorRootOpsSmokeTensor([1]), [2]).options, { from: "maximum", args: [[2]] }, "core tensor root maximum delegates");
expectSame(tensorRootOps.minimum(new TensorRootOpsSmokeTensor([1]), [0]).options, { from: "minimum", args: [[0]] }, "core tensor root minimum delegates");
expectSame(tensorRootOps.where(new TensorRootOpsSmokeTensor([1]), [9], [0]).options, { from: "where", args: [[9], [0]] }, "core tensor root where delegates");
expectSame(tensorRootOps.any(new TensorRootOpsSmokeTensor([1]), -1).options, { from: "any", args: [-1] }, "core tensor root any delegates");
expectSame(tensorRootOps.all(new TensorRootOpsSmokeTensor([1])).options, { from: "all", args: [undefined] }, "core tensor root all delegates");
expectSame(tensorRootOps.clamp(new TensorRootOpsSmokeTensor([1]), 0, 1).options, { from: "clamp", args: [0, 1] }, "core tensor root clamp delegates");
expectSame(tensorRootOps.clip(new TensorRootOpsSmokeTensor([1]), null, 1).options, { from: "clip", args: [null, 1] }, "core tensor root clip delegates");
expectSame(tensorRootFacadeCalls.map((call) => call.name), ["tensor", "parameter", "cat", "cat", "cat", "stack", "tensor"], "core tensor root facade calls");

const metadataOps = createTensorMetadataOps({
  dtype: (tensor) => tensor.dtype,
  device: (tensor) => tensor.device,
  numel: (tensor) => tensor.numel,
  rowMajorStrides: shape.rowMajorStrides,
  normalizeDim: shape.normalizeDim,
});
const metadataLeafTensor = {
  dtype: "f32",
  device: "cpu",
  data: Float32Array.of(1, 2, 3, 4, 5, 6),
  shape: [2, 3],
  numel: 6,
  requiresGrad: true,
  _prev: [],
  storageOffset: () => 0,
  elementSize: () => 4,
  nbytes: () => 24,
  isContiguous: () => true,
};
const metadataNonLeafTensor = {
  ...metadataLeafTensor,
  _prev: [{}],
};
expectSame(tensorMetadata.tensorMetadataManifest.policyOwner, "src/ts/core/tensor_metadata.ts", "core tensor metadata manifest owner");
expectSame(Object.isFrozen(metadataOps), true, "core tensor metadata ops frozen");
expectSame(metadataOps.isCpu(metadataLeafTensor), true, "core tensor metadata isCpu");
expectSame(metadataOps.isFloatingPoint(metadataLeafTensor), true, "core tensor metadata isFloatingPoint");
expectSame(metadataOps.isLeaf(metadataLeafTensor), true, "core tensor metadata leaf");
expectSame(metadataOps.isLeaf(metadataNonLeafTensor), false, "core tensor metadata non-leaf");
expectSame(metadataOps.elementSize(metadataLeafTensor), 4, "core tensor metadata elementSize");
expectSame(metadataOps.nbytes(metadataLeafTensor), 24, "core tensor metadata nbytes");
expectSame(metadataOps.stride(metadataLeafTensor), [3, 1], "core tensor metadata strides");
expectSame(Object.isFrozen(metadataOps.stride(metadataLeafTensor)), true, "core tensor metadata strides frozen");
expectSame(metadataOps.stride(metadataLeafTensor, -1), 1, "core tensor metadata stride dim");
expectSame(metadataOps.storageOffset(metadataLeafTensor), 0, "core tensor metadata storageOffset");
expectSame(metadataOps.isContiguous(metadataLeafTensor), true, "core tensor metadata isContiguous");
expectSame(metadataOps.inspect(metadataLeafTensor), {
  dtype: "f32",
  device: "cpu",
  shape: [2, 3],
  rank: 2,
  length: 6,
  strides: [3, 1],
  storageOffset: 0,
  elementSize: 4,
  byteLength: 24,
  contiguous: true,
  requiresGrad: true,
  isLeaf: true,
}, "core tensor metadata inspect");
expectSame(Object.isFrozen(metadataOps.inspect(metadataLeafTensor)), true, "core tensor metadata inspect frozen");
expectThrow(
  () => metadataOps.stride(metadataLeafTensor, 2),
  "Tensor.stride dim 2 is out of range for rank 2",
  "core tensor metadata stride dim bounds",
);

const gradStateOps = createTensorGradStateHelpers();
const gradStateTensor = {
  length: 3,
  requiresGrad: false,
  grad: null,
  _prev: [{}],
  _backward: () => {
    throw new Error("old backward should be cleared");
  },
};
expectSame(Object.isFrozen(gradStateOps), true, "adapter tensor grad-state ops frozen");
expectSame(gradStateOps.setRequiresGrad(gradStateTensor, true), undefined, "adapter tensor grad-state setRequiresGrad mutates in place");
expectSame(gradStateTensor.requiresGrad, true, "adapter tensor grad-state enables grad");
expectSame(gradStateTensor.requires_grad, true, "adapter tensor grad-state writes requires_grad alias");
expectSame(gradStateTensor.grad, [0, 0, 0], "adapter tensor grad-state allocates grad buffer");
const existingGrad = gradStateTensor.grad;
gradStateOps.setRequiresGrad(gradStateTensor, true);
expectSame(gradStateTensor.grad === existingGrad, true, "adapter tensor grad-state preserves existing grad buffer");
gradStateOps.detach_(gradStateTensor);
expectSame({
  requiresGrad: gradStateTensor.requiresGrad,
  requires_grad: gradStateTensor.requires_grad,
  grad: gradStateTensor.grad,
  prev: gradStateTensor._prev,
}, {
  requiresGrad: false,
  requires_grad: false,
  grad: null,
  prev: [],
}, "adapter tensor grad-state detach clears graph state");
expectSame(gradStateOps.detach_(gradStateTensor), gradStateTensor, "adapter tensor grad-state detach returns tensor");
expectSame(gradStateTensor._backward(), undefined, "adapter tensor grad-state detach clears backward callback");

expectSame(sessionFacadeComposition.sessionFacadeCompositionManifest, {
  kind: "zgml-session-facade-composition",
  source: "ts",
  policyOwner: "src/ts/runtime/session_facade_composition.ts",
  runtimePath: "Program -> Session -> StepParams",
  genericSessionComposition: "ts",
  llamaSessionComposition: "ts",
  hostBoundary: "native adapter callbacks",
}, "TS session facade composition manifest");
expectSame(sharedFrontend.sharedFrontendManifest, {
  kind: "zgml-shared-frontend",
  source: "ts",
  policyOwner: "src/ts/shared_frontend.ts",
  productSourceOfTruth: "ts-api-zig-core",
  productSemanticsOwner: "src/ts/** + src/**/*.zig",
  nativeProductPolicy: "required-core",
  handwrittenFrontendMirrors: false,
  genericSessionComposition: "ts",
  llamaSessionComposition: "ts",
  legacyCjsBridge: false,
}, "TS shared frontend manifest");
expectSame(runtimeLoadCandidates.runtimeLoadCandidatesManifest, {
  kind: "zgml-runtime-load-candidates",
  source: "ts",
  policyOwner: "src/ts/runtime/runtime_load_candidates.ts",
  runtimePath: "TS package artifact -> concrete runtime load",
  packageArtifactFirst: true,
  runtimeKinds: ["node-ffi-runtime", "bun-ffi-runtime", "shared-frontend-runtime"],
  sourceCheckoutFallbackKinds: ["bun-ffi-runtime"],
}, "TS runtime load candidates manifest");
expectSame(runtimeLoadCandidates.runtimeLoadCandidateEvidence("node-ffi-runtime"), {
  kind: "zgml-runtime-load-candidate-evidence",
  runtimeKind: "node-ffi-runtime",
  source: "ts",
  policyOwner: "src/ts/runtime/runtime_load_candidates.ts",
  runtimePath: "TS package artifact -> concrete runtime load",
  candidates: [
    "./adapters/node_ffi_runtime.cjs",
    "./node_ffi_runtime.cjs",
    "./node_ffi_runtime.js",
    "../../dist/adapters/node_ffi_runtime.cjs",
    "../../../dist/adapters/node_ffi_runtime.cjs",
  ],
  packageCandidates: [
    "./adapters/node_ffi_runtime.cjs",
    "./node_ffi_runtime.cjs",
    "./node_ffi_runtime.js",
    "../../dist/adapters/node_ffi_runtime.cjs",
    "../../../dist/adapters/node_ffi_runtime.cjs",
  ],
  sourceCheckoutFallbacks: [],
  packageArtifactFirst: true,
  hasSourceCheckoutFallback: false,
}, "TS Node runtime load candidate evidence");
expectSame(runtimeLoadCandidates.runtimeLoadCandidateEvidence("bun-ffi-runtime"), {
  kind: "zgml-runtime-load-candidate-evidence",
  runtimeKind: "bun-ffi-runtime",
  source: "ts",
  policyOwner: "src/ts/runtime/runtime_load_candidates.ts",
  runtimePath: "TS package artifact -> concrete runtime load",
  candidates: [
    "./adapters/bun_ffi_runtime.cjs",
    "./bun_ffi_runtime.cjs",
    "./bun_ffi_runtime.ts",
    "../../src/ts/adapters/bun_ffi_runtime.ts",
    "../../../src/ts/adapters/bun_ffi_runtime.ts",
  ],
  packageCandidates: [
    "./adapters/bun_ffi_runtime.cjs",
  ],
  sourceCheckoutFallbacks: [
    "./bun_ffi_runtime.cjs",
    "./bun_ffi_runtime.ts",
    "../../src/ts/adapters/bun_ffi_runtime.ts",
    "../../../src/ts/adapters/bun_ffi_runtime.ts",
  ],
  packageArtifactFirst: true,
  hasSourceCheckoutFallback: true,
}, "TS Bun runtime load candidate evidence");
expectSame(runtimeLoadCandidates.runtimeLoadCandidateEvidence("shared-frontend-runtime"), {
  kind: "zgml-runtime-load-candidate-evidence",
  runtimeKind: "shared-frontend-runtime",
  source: "ts",
  policyOwner: "src/ts/runtime/runtime_load_candidates.ts",
  runtimePath: "TS package artifact -> concrete runtime load",
  candidates: [
    "../shared_frontend.cjs",
    "../shared_frontend.js",
    "../../dist/shared_frontend.cjs",
    "../../../dist/shared_frontend.cjs",
  ],
  packageCandidates: [
    "../shared_frontend.cjs",
    "../shared_frontend.js",
    "../../dist/shared_frontend.cjs",
    "../../../dist/shared_frontend.cjs",
  ],
  sourceCheckoutFallbacks: [],
  packageArtifactFirst: true,
  hasSourceCheckoutFallback: false,
}, "TS shared frontend runtime load candidate evidence");
expectSame(concreteRuntimeLoader.concreteRuntimeLoaderEvidence(), {
  kind: "zgml-concrete-runtime-loader-evidence",
  source: "ts",
  policyOwner: "src/ts/adapters/concrete_runtime_loader.ts",
  productSemanticsOwner: "src/ts/** + src/**/*.zig",
  productLanguage: "typescript",
  nativeAlignment: "zig-core-contract-tested",
  handwrittenFrontendMirrors: false,
  checkedContract: nativeApiContract.nativeApiContractSignature(),
  runtimeType: "NativeRuntime",
}, "TS concrete runtime loader evidence");
expectSame(concreteRuntimeLoader.concreteRuntimeLoaderManifest.evidence, concreteRuntimeLoader.concreteRuntimeLoaderEvidence(), "TS concrete runtime loader manifest carries evidence");
expectSame(nodeConcreteRuntime.nodeConcreteRuntimeManifest.kind, "zgml-node-concrete-runtime-loader", "TS Node concrete runtime manifest kind");
expectSame(nodeConcreteRuntime.nodeConcreteRuntimeManifest.runtimeLoad, runtimeLoadCandidates.runtimeLoadCandidateEvidence("node-ffi-runtime"), "TS Node concrete runtime manifest carries runtime-load evidence");
expectSame(nodeConcreteRuntime.nodeConcreteRuntimeManifest.loader, concreteRuntimeLoader.concreteRuntimeLoaderEvidence(), "TS Node concrete runtime manifest carries loader evidence");
expectSame(nodeConcreteRuntime.nodeConcreteRuntimeManifest.checkedContract, nativeApiContract.nativeApiContractSignature(), "TS Node concrete runtime manifest carries native contract signature");
expectSame(bunConcreteRuntime.bunConcreteRuntimeManifest.kind, "zgml-bun-concrete-runtime-loader", "TS Bun concrete runtime manifest kind");
expectSame(bunConcreteRuntime.bunConcreteRuntimeManifest.runtimeLoad, runtimeLoadCandidates.runtimeLoadCandidateEvidence("bun-ffi-runtime"), "TS Bun concrete runtime manifest carries runtime-load evidence");
expectSame(bunConcreteRuntime.bunConcreteRuntimeManifest.loader, concreteRuntimeLoader.concreteRuntimeLoaderEvidence(), "TS Bun concrete runtime manifest carries loader evidence");
expectSame(bunConcreteRuntime.bunConcreteRuntimeManifest.checkedContract, nativeApiContract.nativeApiContractSignature(), "TS Bun concrete runtime manifest carries native contract signature");
expectSame(sharedFrontendRuntime.sharedFrontendRuntimeManifest.kind, "zgml-shared-frontend-runtime-loader", "TS shared frontend runtime manifest kind");
expectSame(sharedFrontendRuntime.sharedFrontendRuntimeManifest.runtimeLoad, runtimeLoadCandidates.runtimeLoadCandidateEvidence("shared-frontend-runtime"), "TS shared frontend runtime manifest carries runtime-load evidence");
expectSame(sharedFrontendRuntime.sharedFrontendRuntimeManifest.checkedRuntimeManifest, {
  manifestKind: "zgml-shared-frontend",
  source: "ts",
  policyOwner: "src/ts/shared_frontend.ts",
  legacyCjsBridge: false,
}, "TS shared frontend runtime manifest carries checked runtime manifest");
expectSame(sharedFrontendRuntime.assertSharedFrontendRuntimeContract(sharedFrontend, "TS source smoke"), undefined, "TS shared frontend runtime contract accepts authored shared frontend");
expectSame(sharedFrontendRuntime.sharedFrontendRuntimeManifest.legacyCjsFallback, false, "TS shared frontend runtime manifest rejects legacy CJS fallback");

expectTsFrontendManifest(tsFrontend, "TS frontend");
expectSame(tsFrontend.shape.shapeScalarCount([2, 3, 4]), 24, "TS frontend re-exports core shape namespace");
expectSame(tsFrontend.token.tokenId(7), 7, "TS frontend re-exports token namespace");
expectSame(typeof tsFrontend.nnLinear.createLinearModuleClass, "function", "TS frontend re-exports nn namespace slices");
expectSame(typeof tsFrontend.sessionValues.genericSessionStepParamsCompatibility, "function", "TS frontend re-exports runtime session values");
expectSame(tsFrontend.nativeApiContract.nativeApiContractManifest.policyOwner, "src/ts/runtime/native_api_contract.ts", "TS frontend re-exports native API contract");
expectSame(tsFrontend.nativeApiContract.nativePackageSpineContractManifest.source, "generated-ts", "TS frontend re-exports generated native package spine contract");
expectSame(tsFrontend.nativeApiContract.nativeApiContractManifest.packageSpine, nativeApiContract.nativePackageSpineContractManifest, "TS native API contract embeds generated package spine evidence");
expectSame(nativeApiContract.nativeApiContractSignature(), nativeApiContract.formatNativeApiContractSignature(nativeApiContract.nativeApiContractSignatureParts()), "TS native API contract signature");
expectSame(tsFrontend.nativeApiContract.requiredNativePackageSpinePrefixExports, nativeApiContract.requiredNativePackageSpinePrefixExports, "TS frontend re-exports native package spine prefix policy");
expectSame(tsFrontend.nativeApiContract.requiredNativePackageSpineSuffixExports, nativeApiContract.requiredNativePackageSpineSuffixExports, "TS frontend re-exports native package spine suffix policy");
expectSame(tsFrontend.nativeApiContract.requiredNativeApiSentinelExports, nativeApiContract.requiredNativeApiSentinelExports, "TS frontend re-exports native API sentinels");
expectSame(nativeApiContract.missingRequiredNativeApiSentinelExports(), [], "TS native API contract sentinel coverage");
expectSame(nativeApiContract.missingExports({ Tensor: true }, ["Tensor"]), [], "TS native API contract missingExports present");
expectSame(nativeApiContract.missingExports({}, ["Tensor"]), ["Tensor"], "TS native API contract missingExports absent");
expectSame(tsNode.frontendManifest.source, "ts", "TS Node entrypoint re-exports frontend manifest");
expectSame(tsNode.nodeAdapterEvidence, {
  kind: "native-adapter",
  host: "node",
  frontendSource: "ts",
  productSemanticsOwner: "src/ts/** + src/**/*.zig",
  nativeAlignment: "zig-core-contract-tested",
  ownsFrontendPolicy: false,
  ownsNativeLoading: true,
  capabilities: [
    { kind: "loadLibrary", required: true },
    { kind: "bindAbi", required: true },
    { kind: "readFile", required: false },
    { kind: "pathJoin", required: false },
    { kind: "ffiCall", required: true },
  ],
  signature: "node:ts-frontend:native-loader:loadLibrary+bindAbi+readFile+pathJoin+ffiCall",
}, "TS Node adapter evidence");
expectSame(tsNode.nodeAdapterManifest.kind, "zgml-node-adapter", "TS Node adapter manifest kind");
expectSame(tsNode.nodeAdapterManifest.adapterEvidence, tsNode.nodeAdapterEvidence, "TS Node adapter manifest carries adapter evidence");
expectSame(tsNode.nodeAdapterManifest.concreteRuntime.runtimeLoad, runtimeLoadCandidates.runtimeLoadCandidateEvidence("node-ffi-runtime"), "TS Node adapter manifest carries concrete runtime load evidence");
expectSame(tsNode.adapterOwnsFrontendPolicy(tsNode.nodeAdapterEvidence), false, "TS Node adapter does not own frontend policy");
expectSame(tsBun.bunAdapterEvidence.host, "bun", "TS Bun adapter evidence host");
expectSame(tsBun.bunAdapterEvidence.productSemanticsOwner, "src/ts/** + src/**/*.zig", "TS Bun adapter evidence product owner");
expectSame(tsBun.bunAdapterEvidence.nativeAlignment, "zig-core-contract-tested", "TS Bun adapter evidence native alignment");
expectSame(tsBun.bunAdapterEvidence.signature, "bun:ts-frontend:native-loader:loadLibrary+bindAbi+readFile+pathJoin+ffiCall", "TS Bun adapter evidence signature");
expectSame(tsBun.bunAdapterManifest.kind, "zgml-bun-adapter", "TS Bun adapter manifest kind");
expectSame(tsBun.bunAdapterManifest.adapterEvidence, tsBun.bunAdapterEvidence, "TS Bun adapter manifest carries adapter evidence");
expectSame(tsBun.bunAdapterManifest.concreteRuntime.runtimeLoad, runtimeLoadCandidates.runtimeLoadCandidateEvidence("bun-ffi-runtime"), "TS Bun adapter manifest carries concrete runtime load evidence");
expectSame(tsNode.nativeLibraryExtensionForPlatform("darwin"), "dylib", "TS native adapter darwin library extension");
expectSame(tsNode.nativeLibraryExtensionForPlatform("win32"), "dll", "TS native adapter windows library extension");
expectSame(tsNode.nativeLibraryExtensionForPlatform("linux"), "so", "TS native adapter linux library extension");
expectSame(tsNode.nativeLibraryFilename("so"), "libzgml_c.so", "TS native adapter library filename");
const nativePathExists = new Set(["/pkg/zig-out/lib/libzgml_c.so", "/repo/zig-out/lib/libzgml_c.so"]);
const resolveNativePathOptions = {
  moduleDir: "/pkg",
  cwd: "/repo",
  extension: "so",
  joinPath: (...parts) => parts.join("/"),
  exists: (path) => nativePathExists.has(path),
};
expectSame(
  tsNode.resolveNativeLibraryPath({ ...resolveNativePathOptions, envLibraryPath: "/custom/libzgml_c.so" }),
  "/custom/libzgml_c.so",
  "TS native adapter prefers env library path",
);
expectSame(
  tsNode.resolveNativeLibraryPath(resolveNativePathOptions),
  "/pkg/zig-out/lib/libzgml_c.so",
  "TS native adapter prefers package-local library path",
);
expectSame(
  tsNode.resolveNativeLibraryPath({
    ...resolveNativePathOptions,
    exists: (path) => path === "/repo/zig-out/lib/libzgml_c.so",
  }),
  "/repo/zig-out/lib/libzgml_c.so",
  "TS native adapter falls back to cwd library path after package-local candidates",
);
expectSame(
  tsNode.resolveNativeLibraryPath({ ...resolveNativePathOptions, cwd: "/missing", exists: () => false }),
  "/pkg/zig-out/lib/libzgml_c.so",
  "TS native adapter returns default path when no candidate exists",
);
expectSame(
  tsNode.nativeLibraryMissingMessage("/tmp/libzgml_c.dylib", "dylib"),
  'zgml native library not found at /tmp/libzgml_c.dylib. Run "zig build ffi-c" from the zgml package root or set ZGML_C_DYLIB to a built libzgml_c.dylib.',
  "TS native adapter missing library message",
);
expectSame(
  tsNode.nodeKoffiFallbackPath({ moduleDir: "/pkg/js", joinPath: (...parts) => parts.join("/") }),
  "/pkg/js/../examples/node_ffi/node_modules/koffi",
  "TS native adapter Node koffi fallback path",
);
const NodeLlamaKvCache = createNodeLlamaKvCacheClass();
const freedNodeKvCacheBuffers = [];
const nodeKvCache = new NodeLlamaKvCache(
  [{ free: () => freedNodeKvCacheBuffers.push("k0") }, { free: () => freedNodeKvCacheBuffers.push("k1") }],
  [{ free: () => freedNodeKvCacheBuffers.push("v0") }],
);
nodeKvCache.dispose();
expectSame(freedNodeKvCacheBuffers, ["k0", "k1", "v0"], "Node LLaMA KV-cache dispose frees K/V buffers");
expectSame(nodeKvCache.k, [], "Node LLaMA KV-cache dispose clears K buffers");
expectSame(nodeKvCache.v, [], "Node LLaMA KV-cache dispose clears V buffers");
const freedBunKvCacheBuffers = [];
const BunLlamaKvCache = createBunLlamaKvCacheClass({
  createLlamaKvCacheBuffers(requirements, options, createBuffer) {
    expectSame(requirements, { layers: 1 }, "Bun LLaMA KV-cache forwards requirements");
    expectSame(options, { tag: "cache" }, "Bun LLaMA KV-cache forwards create options");
    expectSame(typeof createBuffer, "function", "Bun LLaMA KV-cache forwards createBuffer");
    return {
      k: [createBuffer({ kind: "k", layer: 0, byteLength: 4 })],
      v: [{ byteLength: 8, free: () => freedBunKvCacheBuffers.push("v0") }],
    };
  },
  freeKvCacheBuffers(k, v) {
    for (const buffer of k) buffer.free();
    for (const buffer of v) buffer.free();
  },
});
const bunKvCache = new BunLlamaKvCache(
  { layers: 1 },
  { tag: "cache" },
  (slot) => ({ byteLength: slot.byteLength, free: () => freedBunKvCacheBuffers.push(`${slot.kind}${slot.layer}`) }),
);
bunKvCache.dispose();
expectSame(freedBunKvCacheBuffers, ["k0", "v0"], "Bun LLaMA KV-cache dispose frees K/V buffers");
expectSame(bunKvCache.k, [], "Bun LLaMA KV-cache dispose clears K buffers");
expectSame(bunKvCache.v, [], "Bun LLaMA KV-cache dispose clears V buffers");
class AdapterNativeBufferInstanceSmoke {}
const adapterNativeBufferInstanceSurface = createAdapterNativeBufferInstanceSurface({
  getNativeBufferClass: () => AdapterNativeBufferInstanceSmoke,
  uninitializedMessage: "Adapter NativeBuffer instance surface is not initialized",
});
expectSame(adapterNativeBufferInstanceSurface.isNativeBuffer(new AdapterNativeBufferInstanceSmoke()), true, "Adapter NativeBuffer instance surface accepts class instances");
expectSame(adapterNativeBufferInstanceSurface.isNativeBuffer({}), false, "Adapter NativeBuffer instance surface rejects plain objects");
expectThrow(
  () => createAdapterNativeBufferInstanceSurface({
    getNativeBufferClass: () => null,
    uninitializedMessage: "Adapter NativeBuffer instance surface is not initialized",
  }).isNativeBuffer({}),
  "Adapter NativeBuffer instance surface is not initialized",
  "Adapter NativeBuffer instance surface requires initialization",
);
class HostNativeBufferInstanceSmoke {}
const hostNativeBufferInstanceSurface = createHostNativeBufferInstanceSurface({
  getNativeBufferClass: () => HostNativeBufferInstanceSmoke,
  uninitializedMessage: "host NativeBuffer missing",
});
expectSame(hostNativeBufferInstanceSurface.isNativeBuffer(new HostNativeBufferInstanceSmoke()), true, "host NativeBuffer instance surface accepts class instances");
expectSame(hostNativeBufferInstanceSurface.isNativeBuffer({}), false, "host NativeBuffer instance surface rejects plain objects");
expectThrow(
  () => createHostNativeBufferInstanceSurface({ getNativeBufferClass: () => null, uninitializedMessage: "host NativeBuffer missing" }).isNativeBuffer({}),
  "host NativeBuffer missing",
  "host NativeBuffer instance surface requires initialization",
);
const adapterProgramBufferFactoryCalls = [];
const adapterProgramBufferFactorySurface = createAdapterProgramBufferFactorySurface({
  getProgramBufferFactories: () => ({
    createProgramOutputBuffer(handle, options) {
      adapterProgramBufferFactoryCalls.push(["output", handle, options]);
      return { kind: "output", handle, options };
    },
    createProgramBuffer(handle, kind, options) {
      adapterProgramBufferFactoryCalls.push(["buffer", handle, kind, options]);
      return { kind, handle, options };
    },
    createProgramKvCacheBuffer(handle, kind, options) {
      adapterProgramBufferFactoryCalls.push(["kv", handle, kind, options]);
      return { kind, handle, options };
    },
    createProgramNamedBuffer(handle, kind, options) {
      adapterProgramBufferFactoryCalls.push(["named", handle, kind, options]);
      return { kind, handle, options };
    },
    createProgramLlamaKvCache(handle, options) {
      adapterProgramBufferFactoryCalls.push(["llama-kv", handle, options]);
      return { handle, options };
    },
  }),
});
expectSame(adapterProgramBufferFactorySurface.createProgramOutputBuffer(3), { kind: "output", handle: 3, options: {} }, "Adapter Program buffer factory output default options");
expectSame(adapterProgramBufferFactorySurface.createProgramBuffer(3, "weights", { placement: "webgpu" }), { kind: "weights", handle: 3, options: { placement: "webgpu" } }, "Adapter Program buffer factory buffer forwards kind/options");
expectSame(adapterProgramBufferFactorySurface.createProgramKvCacheBuffer(3, "kv-k"), { kind: "kv-k", handle: 3, options: {} }, "Adapter Program buffer factory KV default options");
expectSame(adapterProgramBufferFactorySurface.createProgramNamedBuffer(3, "bias"), { kind: "bias", handle: 3, options: {} }, "Adapter Program buffer factory named default options");
expectSame(adapterProgramBufferFactorySurface.createProgramLlamaKvCache(3, { resource: null }), { handle: 3, options: { resource: null } }, "Adapter Program buffer factory LLaMA KV forwards options");
expectSame(adapterProgramBufferFactoryCalls.map((call) => call[0]), ["output", "buffer", "kv", "named", "llama-kv"], "Adapter Program buffer factory surface call order");
expectThrow(
  () => createAdapterProgramBufferFactorySurface({ getProgramBufferFactories: () => null }).createProgramOutputBuffer(1),
  "Program buffer factory surface is not initialized",
  "Adapter Program buffer factory surface requires initialization",
);
class AdapterIndexValuesSmokeTensor {
  constructor(data) {
    this.data = data;
  }
}
const adapterIndexValuesSurface = createAdapterIndexValuesSurface({ Tensor: AdapterIndexValuesSmokeTensor });
const adapterIndexTensorValues = new Float32Array([1, 2, 3]);
const adapterIndexTypedArrayValues = new Uint32Array([3, 2, 1]);
expectSame(adapterIndexValuesSurface.indexValues(new AdapterIndexValuesSmokeTensor(adapterIndexTensorValues), "indices"), adapterIndexTensorValues, "Adapter index-values surface unwraps Tensor data");
expectSame(adapterIndexValuesSurface.indexValues(adapterIndexTypedArrayValues, "indices"), adapterIndexTypedArrayValues, "Adapter index-values surface accepts numeric typed arrays");
expectSame(adapterIndexValuesSurface.indexValues([1, 0], "indices"), [1, 0], "Adapter index-values surface accepts arrays");
expectSame(indexValues.indexValuesManifest.policyOwner, "src/ts/core/index_values.ts", "index values core manifest owner");
expectSame(indexValues.indexValuesFromTensorOrArray([1, 0], "indices", AdapterIndexValuesSmokeTensor), [1, 0], "core index-values accepts arrays");
expectThrow(
  () => adapterIndexValuesSurface.indexValues({ data: [1] }, "indices"),
  "indices must be an array or numeric typed array",
  "Adapter index-values surface rejects unsupported values",
);
class HostIndexValuesSmokeTensor {
  constructor(data) {
    this.data = data;
  }
}
const hostIndexValuesSurface = createHostAdapterIndexValuesSurface({ Tensor: HostIndexValuesSmokeTensor });
const hostIndexTensorValues = new Float32Array([5, 6]);
expectSame(hostIndexValuesSurface.indexValues(new HostIndexValuesSmokeTensor(hostIndexTensorValues), "indices"), hostIndexTensorValues, "host index-values surface unwraps Tensor data");
expectSame(hostIndexValuesSurface.indexValues([2, 1], "indices"), [2, 1], "host index-values surface accepts arrays");
expectThrow(
  () => hostIndexValuesSurface.indexValues({ data: [1] }, "indices"),
  "indices must be an array or numeric typed array",
  "host index-values surface rejects unsupported values",
);

console.log("zgml TS source smoke ok: core/shape, tensor_core, tensor_data, tensor_math, tensor_factory, tensor_facade, tensor_join, tensor_view, tensor_index, tensor_metadata, tensor_root, activation, grad_mode, token, compile-diagnostics, compiler-signature, module Program descriptor, native kernel contract, module binding, module compiler policy, module facade, Session tensor, StepParams, Session contract, Session profile, Session parameters, Session layout, Session binding, Session lifecycle, Session values, Program buffer, Program buffer factory, Program facade policy, Program module binding, Program device, Program sizing, Program parameters, Program shapes, Program resources, Program layout, Program policy accessors, LLaMA Program policy accessors, Program facade factories, module compatibility, ABI, inspection, model-source, Tensor Program IR, trace-compiler, tensor-placement, native-buffer, data namespace, adapter tensor grad-state ops, host Adapter forwarding surfaces, adapter generic family model-handle surface, adapter Node/Bun LLaMA KV-cache surfaces, adapter Node Program buffer factory surface, adapter index-values surface, adapter module compiler surface, adapter model-source surface, adapter NativeBuffer instance surface, nn shape-module, nn parameterless modules, nn linear-module, nn embedding-module, nn sequential-module, nn feature-norm-module, nn namespace, module-mode, module-state, training behavior, and TS package spine");

export {};
