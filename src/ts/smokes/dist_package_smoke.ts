"use strict";

declare const console: { log: (message?: unknown, ...optionalParams: unknown[]) => void };
declare const require: (id: string) => any;

const frontend = require(["..", "index.cjs"].join("/"));
const {
  expectFrontendManifest,
  expectNativeAdapterEvidence,
  expectStepParamsNamespace,
  expectTsProductManifest,
} = require([".", "smoke_contracts.cjs"].join("/"));
const node = require(["..", "node.cjs"].join("/"));
const bun = require(["..", "bun.cjs"].join("/"));
const checkpoint = require(["..", "checkpoint.cjs"].join("/"));
const compile = require(["..", "compile.cjs"].join("/"));
const inspection = require(["..", "inspection.cjs"].join("/"));
const loss = require(["..", "loss.cjs"].join("/"));
const modelSource = require(["..", "model_source.cjs"].join("/"));
const optim = require(["..", "optim.cjs"].join("/"));
const train = require(["..", "train.cjs"].join("/"));
const tensor = require(["..", "tensor.cjs"].join("/"));
const nn = require(["..", "nn.cjs"].join("/"));
const program = require(["..", "program.cjs"].join("/"));
const session = require(["..", "session.cjs"].join("/"));
const stepParams = require(["..", "step_params.cjs"].join("/"));
const nativeBuffer = require(["..", "native_buffer.cjs"].join("/"));
const programDevice = require(["..", "program_device.cjs"].join("/"));
const nativeAdapter = require(["..", "adapters", "native.cjs"].join("/"));
const nodeAdapter = require(["..", "adapters", "node.cjs"].join("/"));
const concreteNodeRuntime = require(["..", "adapters", "node_ffi_runtime.cjs"].join("/"));
const bunAdapter = require(["..", "adapters", "bun.cjs"].join("/"));
const coreIndexValues = require(["..", "core", "index_values.cjs"].join("/"));
const coreTensorMetadata = require(["..", "core", "tensor_metadata.cjs"].join("/"));
const coreTensorRoot = require(["..", "core", "tensor_root.cjs"].join("/"));
const coreTensorHostSurface = require(["..", "core", "tensor_host_surface.cjs"].join("/"));
const coreShape = require(["..", "core", "shape.cjs"].join("/"));
const runtimeGenericFamilySurface = require(["..", "runtime", "generic_family_surface.cjs"].join("/"));
const runtimeGenericModelDesc = require(["..", "runtime", "generic_model_desc.cjs"].join("/"));
const runtimeHostAdapterSurfaces = require(["..", "runtime", "host_adapter_surfaces.cjs"].join("/"));
const topLevelKernelPlan = require(["..", "kernel_plan.cjs"].join("/"));
const runtimeKernelPlan = require(["..", "runtime", "kernel_plan.cjs"].join("/"));
const runtimeLlamaFamilySurface = require(["..", "runtime", "llama_family_surface.cjs"].join("/"));
const runtimeLlamaKvCache = require(["..", "runtime", "llama_kv_cache.cjs"].join("/"));
const runtimeLlamaProgramInspection = require(["..", "runtime", "llama_program_inspection.cjs"].join("/"));
const runtimeLlamaSessionBindDesc = require(["..", "runtime", "llama_session_bind_desc.cjs"].join("/"));
const runtimeLlamaTokenOutput = require(["..", "runtime", "llama_token_output.cjs"].join("/"));
const runtimeModuleProgramEvidence = require(["..", "runtime", "module_program_evidence.cjs"].join("/"));
const runtimeNativeAbiConstants = require(["..", "runtime", "native_abi_constants.cjs"].join("/"));
const runtimeNativeKernelContract = require(["..", "runtime", "native_kernel_contract.cjs"].join("/"));
const runtimeNativeApiContract = require(["..", "runtime", "native_api_contract.cjs"].join("/"));
const runtimeNativeStatus = require(["..", "runtime", "native_status.cjs"].join("/"));
const runtimeInspection = require(["..", "runtime", "inspection.cjs"].join("/"));
const runtimeModelSource = require(["..", "runtime", "model_source.cjs"].join("/"));
const runtimeModelSourceDesc = require(["..", "runtime", "model_source_desc.cjs"].join("/"));
const runtimeNativeBuffer = require(["..", "runtime", "native_buffer.cjs"].join("/"));
const runtimeProgramBindingSurface = require(["..", "runtime", "program_binding_surface.cjs"].join("/"));
const runtimeProgramBindDesc = require(["..", "runtime", "program_bind_desc.cjs"].join("/"));
const runtimeProgramBufferDesc = require(["..", "runtime", "program_buffer_desc.cjs"].join("/"));
const runtimeProgramDevice = require(["..", "runtime", "program_device.cjs"].join("/"));
const runtimeLoadCandidates = require(["..", "runtime", "runtime_load_candidates.cjs"].join("/"));
const runtimeStepParams = require(["..", "runtime", "step_params.cjs"].join("/"));
const nnLinear = require(["..", "nn", "linear_module.cjs"].join("/"));
const trainMode = require(["..", "train", "module_mode.cjs"].join("/"));

expectFrontendManifest(frontend, "dist/index.cjs");

expectFrontendManifest(node, "dist/node.cjs");

const missingNodeNative = runtimeNativeApiContract.missingExports(node, runtimeNativeApiContract.requiredNativeApiExports);
if (missingNodeNative.length !== 0) {
  throw new Error(`dist/node.cjs must bridge the full TS-declared native API: ${missingNodeNative.join(", ")}`);
}

const missingNativeSentinels = runtimeNativeApiContract.missingRequiredNativeApiSentinelExports();
if (missingNativeSentinels.length !== 0) {
  throw new Error(`dist/runtime/native_api_contract.cjs must include the TS-owned PyTorch-like root helpers: ${missingNativeSentinels.join(", ")}`);
}

for (const key of runtimeNativeApiContract.requiredNativeApiExports) {
  if (node[key] !== concreteNodeRuntime[key]) {
    throw new Error(`dist/node.cjs native bridge export ${key} must match the concrete Node runtime`);
  }
}

expectFrontendManifest(bun, "dist/bun.cjs");

if (frontend.nn.nnManifest.source !== "ts") {
  throw new Error("dist/index.cjs must expose the TS-authored nn namespace");
}

expectTsProductManifest(frontend.nn.nnManifest, "src/ts/nn.ts", "dist/index.cjs nn");
expectTsProductManifest(frontend.compile.compileManifest, "src/ts/compile.ts", "dist/index.cjs compile");
expectTsProductManifest(frontend.inspection.inspectionManifest, "src/ts/inspection.ts", "dist/index.cjs inspection");
expectTsProductManifest(frontend.modelSource.modelSourceManifest, "src/ts/model_source.ts", "dist/index.cjs model source");
expectTsProductManifest(checkpoint.checkpointManifest, "src/ts/checkpoint.ts", "dist/checkpoint.cjs");
expectTsProductManifest(compile.compileManifest, "src/ts/compile.ts", "dist/compile.cjs");
expectTsProductManifest(inspection.inspectionManifest, "src/ts/inspection.ts", "dist/inspection.cjs");
expectTsProductManifest(loss.lossManifest, "src/ts/loss.ts", "dist/loss.cjs");
expectTsProductManifest(modelSource.modelSourceManifest, "src/ts/model_source.ts", "dist/model_source.cjs");
expectTsProductManifest(optim.optimManifest, "src/ts/optim.ts", "dist/optim.cjs");
expectTsProductManifest(train.trainManifest, "src/ts/train.ts", "dist/train.cjs");
expectTsProductManifest(tensor.tensorManifest, "src/ts/tensor.ts", "dist/tensor.cjs");
expectTsProductManifest(nn.nnManifest, "src/ts/nn.ts", "dist/nn.cjs");
expectTsProductManifest(program.programManifest, "src/ts/program.ts", "dist/program.cjs");
expectTsProductManifest(session.sessionManifest, "src/ts/session.ts", "dist/session.cjs");
expectTsProductManifest(stepParams.stepParamsManifest, "src/ts/step_params.ts", "dist/step_params.cjs");
expectTsProductManifest(nativeBuffer.nativeBufferManifest, "src/ts/native_buffer.ts", "dist/native_buffer.cjs");
expectTsProductManifest(programDevice.programDeviceManifest, "src/ts/program_device.ts", "dist/program_device.cjs");

if (frontend.compile.compileManifest.runtimePath !== "Trace -> TensorProgramIr -> KernelPlan -> Program") {
  throw new Error("dist/index.cjs must expose the TS-authored compile namespace");
}

if (frontend.inspection.inspectionManifest.policyOwner !== "src/ts/inspection.ts") {
  throw new Error("dist/index.cjs must expose the TS-authored inspection namespace");
}

if (frontend.modelSource.modelSourceManifest.policyOwner !== "src/ts/model_source.ts") {
  throw new Error("dist/index.cjs must expose the TS-authored model source namespace");
}

if (checkpoint.checkpointManifest.source !== "ts" || typeof checkpoint.createCheckpointHelpers !== "function") {
  throw new Error("dist/checkpoint.cjs must expose the TS-authored checkpoint namespace");
}

if (
  compile.compileManifest.source !== "ts" ||
  typeof compile.traceCompilerArtifacts !== "function" ||
  typeof compile.requireCompileSupport !== "function" ||
  typeof compile.require_compile_support !== "function"
) {
  throw new Error("dist/compile.cjs must expose the TS-authored compile namespace");
}

if (
  inspection.inspectionManifest.source !== "ts" ||
  inspection.programExecutionModeFromCapabilities(null) !== "compile-only" ||
  typeof inspection.runtimeProfileExpectation !== "function" ||
  typeof inspection.runtimeProfileHasNoFallback !== "function" ||
  typeof inspection.runtimeProfileHasNoSync !== "function" ||
  typeof inspection.runtimeProfileHasNoInvalidRuntimePatches !== "function" ||
  typeof inspection.requireNoFallbackRuntimeProfile !== "function" ||
  typeof inspection.requireNoSyncRuntimeProfile !== "function" ||
  typeof inspection.requireRuntimePatchValidProfile !== "function" ||
  typeof inspection.requireHotRuntimeProfile !== "function"
) {
  throw new Error("dist/inspection.cjs must expose the TS-authored inspection namespace");
}

if (loss.lossManifest.source !== "ts" || typeof loss.createLossTrainHelpers !== "function") {
  throw new Error("dist/loss.cjs must expose the TS-authored loss namespace");
}

if (modelSource.modelSourceManifest.source !== "ts" || modelSource.normalizeLoadModelKind("tinyllama2layer") !== "tiny-llama-2layer") {
  throw new Error("dist/model_source.cjs must expose the TS-authored model source namespace");
}

if (optim.optimManifest.source !== "ts" || typeof optim.createOptimNamespace !== "function") {
  throw new Error("dist/optim.cjs must expose the TS-authored optim namespace");
}

if (train.trainManifest.source !== "ts" || typeof train.createLossTrainHelpers !== "function") {
  throw new Error("dist/train.cjs must expose the TS-authored train namespace");
}

if (tensor.tensorManifest.source !== "ts" || typeof tensor.createTensorFacadeHelpers !== "function") {
  throw new Error("dist/tensor.cjs must expose the TS-authored tensor namespace");
}

if (nn.nnManifest.source !== "ts" || typeof nn.createNnNamespace !== "function") {
  throw new Error("dist/nn.cjs must expose the TS-authored nn namespace");
}

if (program.programManifest.source !== "ts" || typeof program.createGenericProgramFacadeHelpers !== "function") {
  throw new Error("dist/program.cjs must expose the TS-authored Program namespace");
}

if (session.sessionManifest.source !== "ts" || typeof session.createSessionLiveFacadeHelpers !== "function") {
  throw new Error("dist/session.cjs must expose the TS-authored Session namespace");
}

expectStepParamsNamespace(stepParams, "dist/step_params.cjs", "dist");
expectStepParamsNamespace(frontend.stepParams, "dist/index.cjs stepParams", "dist");

if (nativeBuffer.nativeBufferManifest.source !== "ts" || typeof nativeBuffer.nativeBufferWriteInfo !== "function") {
  throw new Error("dist/native_buffer.cjs must expose the TS-authored NativeBuffer policy namespace");
}

if (programDevice.programDeviceManifest.source !== "ts" || typeof programDevice.createProgramDeviceClass !== "function") {
  throw new Error("dist/program_device.cjs must expose the TS-authored ProgramDevice policy namespace");
}

const nativeNodeEvidence = nativeAdapter.nativeAdapterEvidence("node");
expectNativeAdapterEvidence(nativeNodeEvidence, "node", "dist/adapters/native.cjs");

expectNativeAdapterEvidence(nodeAdapter.nodeAdapterEvidence, "node", "dist/adapters/node.cjs");
expectNativeAdapterEvidence(bunAdapter.bunAdapterEvidence, "bun", "dist/adapters/bun.cjs");

if (coreShape.shapeScalarCount([2, 3]) !== 6) {
  throw new Error("dist/core/shape.cjs must expose shape helpers");
}

if (
  coreIndexValues.indexValuesManifest.policyOwner !== "src/ts/core/index_values.ts" ||
  coreIndexValues.indexValuesFromTensorOrArray(new Uint32Array([1]), "idx", class {}).length !== 1
) {
  throw new Error("dist/core/index_values.cjs must expose index-values normalization policy");
}

if (
  coreTensorMetadata.tensorMetadataManifest.policyOwner !== "src/ts/core/tensor_metadata.ts" ||
  coreTensorMetadata.createTensorMetadataOps({
    dtype: () => "f32",
    device: () => "cpu",
    numel: () => 1,
    rowMajorStrides: () => [1],
    normalizeDim: () => 0,
  }).inspect({ data: Float32Array.of(1), shape: [1], requiresGrad: false }).rank !== 1
) {
  throw new Error("dist/core/tensor_metadata.cjs must expose Tensor metadata policy");
}

class DistTensorRootSmokeTensor {
  data: unknown;
  shape: readonly number[];

  constructor(data: unknown, shape: readonly number[] = [Array.isArray(data) ? data.length : 1]) {
    this.data = data;
    this.shape = shape;
  }
  allclose() { return true; }
  equal() { return true; }
}
const distTensorRootOps = coreTensorRoot.createTensorRootOps({
  Tensor: DistTensorRootSmokeTensor,
  tensor: undefined,
  tensorFacade: {
    tensor: (data: unknown, shape?: readonly number[]) => new DistTensorRootSmokeTensor(data, shape),
  },
});
if (
  coreTensorRoot.tensorRootManifest.policyOwner !== "src/ts/core/tensor_root.ts" ||
  distTensorRootOps.hasShape(new DistTensorRootSmokeTensor([1], [1]), [1]) !== true ||
  distTensorRootOps.requireShape(new DistTensorRootSmokeTensor([1], [1]), [1]).shape[0] !== 1
) {
  throw new Error("dist/core/tensor_root.cjs must expose Tensor root helper policy");
}

if (runtimeKernelPlan.kernelPlanManifest.policyOwner !== "src/ts/runtime/kernel_plan.ts") {
  throw new Error("dist/runtime/kernel_plan.cjs must expose KernelPlan policy");
}

if (
  topLevelKernelPlan.kernelPlanManifest.policyOwner !== "src/ts/runtime/kernel_plan.ts" ||
  typeof topLevelKernelPlan.kernelPlanSignature !== "function" ||
  typeof topLevelKernelPlan.acceptsKernelPlan !== "function" ||
  typeof topLevelKernelPlan.requireKernelPlan !== "function" ||
  typeof topLevelKernelPlan.matchesKernelPlanSignature !== "function"
) {
  throw new Error("dist/kernel_plan.cjs must expose the top-level TS KernelPlan facade");
}

if (
  runtimeNativeKernelContract.nativeKernelContractManifest.policyOwner !== "src/ts/runtime/native_kernel_contract.ts" ||
  runtimeNativeKernelContract.kernelNameForNativeModuleDesc({ kind: 1 }) !== "linear" ||
  runtimeNativeKernelContract.nativeModuleOpDescAbiFields({ kind: 8, a: 2 }).a !== 2
) {
  throw new Error("dist/runtime/native_kernel_contract.cjs must expose native descriptor policy");
}

if (
  runtimeGenericModelDesc.genericModelDescriptorManifest.policyOwner !== "src/ts/runtime/generic_model_desc.ts" ||
  runtimeGenericFamilySurface.genericFamilySurfaceManifest.policyOwner !== "src/ts/runtime/generic_family_surface.ts" ||
  runtimeHostAdapterSurfaces.hostAdapterSurfacesManifest.policyOwner !== "src/ts/runtime/host_adapter_surfaces.ts" ||
  runtimeLlamaFamilySurface.llamaFamilySurfaceManifest.policyOwner !== "src/ts/runtime/llama_family_surface.ts" ||
  runtimeProgramBindingSurface.programBindingSurfaceManifest.policyOwner !== "src/ts/runtime/program_binding_surface.ts" ||
  coreTensorHostSurface.tensorHostSurfaceManifest.policyOwner !== "src/ts/core/tensor_host_surface.ts" ||
  runtimeGenericModelDesc.genericModelNativeDesc({ inputLen: 2, outputLen: 1 }, { tinyLinearKind: 3, tinyMlpKind: 4 }).kind !== 3 ||
  runtimeLlamaKvCache.llamaKvCacheManifest.policyOwner !== "src/ts/runtime/llama_kv_cache.ts" ||
  runtimeLlamaProgramInspection.llamaProgramInspectionManifest.policyOwner !== "src/ts/runtime/llama_program_inspection.ts" ||
  runtimeLlamaProgramInspection.llamaProgramInspectionFromWords([10, 8, 8, 1, 4, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1]).kind !== "zgml.llama.program.inspection" ||
  runtimeLlamaSessionBindDesc.llamaSessionBindDescriptorManifest.policyOwner !== "src/ts/runtime/llama_session_bind_desc.ts" ||
  runtimeLlamaTokenOutput.llamaTokenOutputManifest.policyOwner !== "src/ts/runtime/llama_token_output.ts" ||
  runtimeLlamaTokenOutput.llamaTokenWindowDescriptorFields({ tokens: new Uint32Array([1]), tokensLen: 1 }, true, null).outputPolicy !== 1 ||
  runtimeModuleProgramEvidence.moduleProgramEvidenceManifest.policyOwner !== "src/ts/runtime/module_program_evidence.ts" ||
  runtimeModuleProgramEvidence.attachProgramCompileEvidence({ keep: true }, { evidence: true }).keep !== true ||
  runtimeInspection.inspectionManifest.policyOwner !== "src/ts/runtime/inspection.ts" ||
  runtimeModelSource.modelSourceManifest.policyOwner !== "src/ts/runtime/model_source.ts" ||
  runtimeModelSourceDesc.modelSourceDescriptorManifest.policyOwner !== "src/ts/runtime/model_source_desc.ts" ||
  runtimeModelSourceDesc.modelPathDescriptorRecord(runtimeModelSourceDesc.modelPathDescriptorFields(3, { length: 4, byteLength: 9 })).path_len !== 4 ||
  runtimeNativeAbiConstants.nativeAbiConstantsManifest.policyOwner !== "src/ts/runtime/native_abi_constants.ts" ||
  runtimeNativeAbiConstants.adapterModelKindAliases({ auto: 0, tinyLinear: 1, tinyLlama: 2, smollm135m: 3, tinyLlama2Layer: 4, tinyMlp: 5, module: 6 }).moduleKind !== 6 ||
  runtimeNativeStatus.nativeStatusManifest.policyOwner !== "src/ts/runtime/native_status.ts" ||
  runtimeNativeBuffer.nativeBufferManifest.policyOwner !== "src/ts/runtime/native_buffer.ts" ||
  runtimeProgramBindDesc.programBindDescriptorManifest.policyOwner !== "src/ts/runtime/program_bind_desc.ts" ||
  runtimeProgramBindDesc.programBindDescriptorRecord(runtimeProgramBindDesc.programOutputBindDescriptorFields({ handle: 1 }, 2)).output_len !== 2 ||
  runtimeProgramBufferDesc.programBufferDescriptorManifest.policyOwner !== "src/ts/runtime/program_buffer_desc.ts" ||
  runtimeProgramBufferDesc.programDeviceBufferImportDescriptorRecord(runtimeProgramBufferDesc.programDeviceBufferImportDescriptorFields({ placement: 1, deviceHandle: 2, bufferHandle: 3, byteOffset: 4, byteLength: 5 })).byte_len !== 5 ||
  runtimeLoadCandidates.runtimeLoadCandidatesManifest.policyOwner !== "src/ts/runtime/runtime_load_candidates.ts" ||
  runtimeLoadCandidates.runtimeLoadCandidateEvidence("node-ffi-runtime").packageArtifactFirst !== true ||
  runtimeProgramDevice.programDeviceManifest.policyOwner !== "src/ts/runtime/program_device.ts"
) {
  throw new Error("dist/runtime/*.cjs must expose runtime-owned TS manifests");
}

if (runtimeNativeApiContract.nativeApiContractSignature() !== runtimeNativeApiContract.formatNativeApiContractSignature(runtimeNativeApiContract.nativeApiContractSignatureParts())) {
  throw new Error("dist/runtime/native_api_contract.cjs must expose the TS native API contract");
}

expectStepParamsNamespace(runtimeStepParams, "dist/runtime/step_params.cjs", "dist");

if (typeof nnLinear.createLinearModuleClass !== "function") {
  throw new Error("dist/nn/linear_module.cjs must expose nn Linear helpers");
}

if (typeof trainMode.setModuleTraining !== "function") {
  throw new Error("dist/train/module_mode.cjs must expose training mode helpers");
}

console.log("zgml dist package smoke ok");

export {};
