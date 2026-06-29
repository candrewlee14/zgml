// @ts-nocheck
"use strict";

const frontend = require(["zgml", "frontend"].join("/"));
const {
  expectConcreteRuntimeLoaderEvidence,
  expectFrontendManifest,
  expectNativeAdapterEvidence,
  expectStepParamsNamespace,
  expectTsProductManifest,
} = require([".", "smoke_contracts.cjs"].join("/"));
const browser = require(["zgml", "browser"].join("/"));
const checkpoint = require(["zgml", "checkpoint"].join("/"));
const compile = require(["zgml", "compile"].join("/"));
const inspection = require(["zgml", "inspection"].join("/"));
const lazy = require(["zgml", "lazy"].join("/"));
const loss = require(["zgml", "loss"].join("/"));
const modelSource = require(["zgml", "model_source"].join("/"));
const optim = require(["zgml", "optim"].join("/"));
const train = require(["zgml", "train"].join("/"));
const tensor = require(["zgml", "tensor"].join("/"));
const nn = require(["zgml", "nn"].join("/"));
const program = require(["zgml", "program"].join("/"));
const session = require(["zgml", "session"].join("/"));
const stepParamsFacade = require(["zgml", "step_params"].join("/"));
const nativeBuffer = require(["zgml", "native_buffer"].join("/"));
const programDevice = require(["zgml", "program_device"].join("/"));
const nativeAdapter = require(["zgml", "adapters", "native"].join("/"));
const nodeAdapter = require(["zgml", "adapters", "node"].join("/"));
const bunAdapter = require(["zgml", "adapters", "bun"].join("/"));
const coreIndexValues = require(["zgml", "core", "index_values"].join("/"));
const coreTensorMetadata = require(["zgml", "core", "tensor_metadata"].join("/"));
const coreTensorRoot = require(["zgml", "core", "tensor_root"].join("/"));
const coreTensorHostSurface = require(["zgml", "core", "tensor_host_surface"].join("/"));
const coreShape = require(["zgml", "core", "shape"].join("/"));
const runtimeGenericFamilySurface = require(["zgml", "runtime", "generic_family_surface"].join("/"));
const runtimeGenericModelDesc = require(["zgml", "runtime", "generic_model_desc"].join("/"));
const runtimeHostAdapterSurfaces = require(["zgml", "runtime", "host_adapter_surfaces"].join("/"));
const topLevelKernelPlan = require(["zgml", "kernel_plan"].join("/"));
const runtimeKernelPlan = require(["zgml", "runtime", "kernel_plan"].join("/"));
const runtimeLlamaFamilySurface = require(["zgml", "runtime", "llama_family_surface"].join("/"));
const runtimeLlamaKvCache = require(["zgml", "runtime", "llama_kv_cache"].join("/"));
const runtimeLlamaProgramInspection = require(["zgml", "runtime", "llama_program_inspection"].join("/"));
const runtimeLlamaSessionBindDesc = require(["zgml", "runtime", "llama_session_bind_desc"].join("/"));
const runtimeLlamaTokenOutput = require(["zgml", "runtime", "llama_token_output"].join("/"));
const runtimeModuleProgramEvidence = require(["zgml", "runtime", "module_program_evidence"].join("/"));
const runtimeNativeAbiConstants = require(["zgml", "runtime", "native_abi_constants"].join("/"));
const runtimeNativeKernelContract = require(["zgml", "runtime", "native_kernel_contract"].join("/"));
const runtimeNativeApiContract = require(["zgml", "runtime", "native_api_contract"].join("/"));
const runtimeNativeStatus = require(["zgml", "runtime", "native_status"].join("/"));
const runtimeInspection = require(["zgml", "runtime", "inspection"].join("/"));
const runtimeModelSource = require(["zgml", "runtime", "model_source"].join("/"));
const runtimeModelSourceDesc = require(["zgml", "runtime", "model_source_desc"].join("/"));
const runtimeNativeBuffer = require(["zgml", "runtime", "native_buffer"].join("/"));
const runtimeProgramBindingSurface = require(["zgml", "runtime", "program_binding_surface"].join("/"));
const runtimeProgramBindDesc = require(["zgml", "runtime", "program_bind_desc"].join("/"));
const runtimeProgramBufferDesc = require(["zgml", "runtime", "program_buffer_desc"].join("/"));
const runtimeProgramDevice = require(["zgml", "runtime", "program_device"].join("/"));
const runtimeLoadCandidates = require(["zgml", "runtime", "runtime_load_candidates"].join("/"));
const runtimeStepParams = require(["zgml", "runtime", "step_params"].join("/"));
const nnLinear = require(["zgml", "nn", "linear_module"].join("/"));
const trainMode = require(["zgml", "train", "module_mode"].join("/"));
const packageJson = require(["..", "..", "package.json"].join("/"));

if (!require.resolve(["zgml", "frontend"].join("/")).endsWith("/dist/index.cjs")) {
  throw new Error("zgml/frontend must resolve to the tsdown dist package artifact");
}

if (!require.resolve(["zgml", "browser"].join("/")).endsWith("/dist/browser.cjs")) {
  throw new Error("zgml/browser must resolve to its tsdown browser package artifact");
}

if (
  coreIndexValues.indexValuesManifest.policyOwner !== "src/ts/core/index_values.ts" ||
  coreIndexValues.indexValuesFromTensorOrArray([1], "idx", class {}).length !== 1
) {
  throw new Error("zgml/core/index_values must expose index-values normalization policy");
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
  throw new Error("zgml/core/tensor_metadata must expose Tensor metadata policy");
}

class FrontendTensorRootSmokeTensor {
  constructor(data, shape = [data.length]) {
    this.data = data;
    this.shape = shape;
  }
  allclose() { return true; }
  equal() { return true; }
}
const frontendTensorRootOps = coreTensorRoot.createTensorRootOps({
  Tensor: FrontendTensorRootSmokeTensor,
  tensorFacade: {
    tensor: (data, shape) => new FrontendTensorRootSmokeTensor(data, shape),
  },
});
if (
  coreTensorRoot.tensorRootManifest.policyOwner !== "src/ts/core/tensor_root.ts" ||
  frontendTensorRootOps.hasShape(new FrontendTensorRootSmokeTensor([1], [1]), [1]) !== true ||
  frontendTensorRootOps.requireShape(new FrontendTensorRootSmokeTensor([1], [1]), [1]).shape[0] !== 1
) {
  throw new Error("zgml/core/tensor_root must expose Tensor root helper policy");
}

if (browser.frontendManifest.source !== "ts" || browser.frontendManifest.runtimePath !== frontend.frontendManifest.runtimePath) {
  throw new Error("zgml/browser must expose the same TS-authored frontend contract as zgml/frontend");
}

if (
  browser.browserManifest?.kind !== "zgml-browser-frontend" ||
  browser.browserManifest?.policyOwner !== "src/ts/browser.ts" ||
  browser.browserManifest?.frontendEntry !== "src/ts/index.ts" ||
  browser.browserManifest?.nativeLoader !== false ||
  browser.browserManifest?.adapterRole !== "browser-safe-frontend"
) {
  throw new Error("zgml/browser must expose a TS-authored browser frontend seam without a native loader");
}

function distBackedSubpathEntries() {
  const entries = [];
  for (const [subpath, exportSpec] of Object.entries(packageJson.exports ?? {})) {
    if (subpath === "." || subpath === "./node" || subpath.includes("*")) continue;
    if (!exportSpec || typeof exportSpec !== "object") continue;
    const target = exportSpec.require ?? exportSpec.default;
    if (typeof target !== "string" || !target.startsWith("./dist/")) continue;
    entries.push([
      ["zgml", subpath.slice(2)].filter(Boolean).join("/"),
      target.slice(1),
    ]);
  }
  return Object.freeze(entries.sort(([left], [right]) => left.localeCompare(right)));
}

const distBackedSubpaths = distBackedSubpathEntries();

for (const [subpath, suffix] of distBackedSubpaths) {
  if (!require.resolve(subpath).endsWith(suffix)) {
    throw new Error(`${subpath} must resolve to the tsdown dist package artifact`);
  }
}

expectFrontendManifest(frontend, "zgml/frontend");

if (frontend.frontendManifest.runtimePath !== "Program -> Session -> StepParams") {
  throw new Error("zgml/frontend must expose Program -> Session -> StepParams evidence");
}

expectStepParamsNamespace(frontend.stepParams, "zgml/frontend stepParams", "frontend");
expectStepParamsNamespace(stepParamsFacade, "zgml/step_params", "frontend");
expectStepParamsNamespace(runtimeStepParams, "zgml/runtime/step_params", "frontend");

if (frontend.shape.shapeScalarCount([2, 3, 4]) !== 24) {
  throw new Error("zgml/frontend shape helpers are unavailable");
}

if (frontend.token.tokenId(7) !== 7) {
  throw new Error("zgml/frontend token helpers are unavailable");
}

expectTsProductManifest(frontend.loss.lossManifest, "src/ts/loss.ts", "zgml/frontend loss");
expectTsProductManifest(frontend.optim.optimManifest, "src/ts/optim.ts", "zgml/frontend optim");
expectTsProductManifest(frontend.train.trainManifest, "src/ts/train.ts", "zgml/frontend train");
expectTsProductManifest(frontend.checkpoint.checkpointManifest, "src/ts/checkpoint.ts", "zgml/frontend checkpoint");
expectTsProductManifest(frontend.inspection.inspectionManifest, "src/ts/inspection.ts", "zgml/frontend inspection");
expectTsProductManifest(frontend.lazy.lazyManifest, "src/ts/lazy.ts", "zgml/frontend lazy");
expectTsProductManifest(frontend.modelSource.modelSourceManifest, "src/ts/model_source.ts", "zgml/frontend model source");
expectTsProductManifest(frontend.tensor.tensorManifest, "src/ts/tensor.ts", "zgml/frontend tensor");
expectTsProductManifest(frontend.nn.nnManifest, "src/ts/nn.ts", "zgml/frontend nn");
expectTsProductManifest(frontend.program.programManifest, "src/ts/program.ts", "zgml/frontend Program");
expectTsProductManifest(frontend.session.sessionManifest, "src/ts/session.ts", "zgml/frontend Session");
expectTsProductManifest(frontend.stepParamsFacade.stepParamsManifest, "src/ts/step_params.ts", "zgml/frontend StepParams");

if (frontend.loss.lossManifest.policyOwner !== "src/ts/loss.ts") {
  throw new Error("zgml/frontend must expose the TS-authored loss namespace");
}

if (frontend.optim.optimManifest.policyOwner !== "src/ts/optim.ts") {
  throw new Error("zgml/frontend must expose the TS-authored optim namespace");
}

if (frontend.train.trainManifest.policyOwner !== "src/ts/train.ts") {
  throw new Error("zgml/frontend must expose the TS-authored train namespace");
}

if (frontend.checkpoint.checkpointManifest.policyOwner !== "src/ts/checkpoint.ts") {
  throw new Error("zgml/frontend must expose the TS-authored checkpoint namespace");
}

if (frontend.compile.compileManifest.runtimePath !== "Trace -> TensorProgramIr -> KernelPlan -> Program") {
  throw new Error("zgml/frontend must expose the TS-authored compile namespace");
}

const lazyGraph = frontend.lazy.input([2], "x").linear(3, { name: "0" }).relu().linear(1, { name: "2" });
const lazyArtifacts = lazyGraph.artifacts();
const lazyModule = browser.nn.sequential([
  browser.nn.linear(2, 3),
  browser.nn.relu(),
  browser.nn.linear(3, 1),
]);
const lazyModuleGraph = frontend.lazy.fromModule(lazyModule, { inputShape: [2] });
const lazyModuleArtifacts = frontend.lazy.moduleArtifacts(lazyModule, { inputShape: [2] });
if (
  frontend.lazy.lazyManifest.runtimePath !== "LazyTensor -> Trace -> TensorProgramIr -> KernelPlan -> Program" ||
  lazy.lazyManifest.policyOwner !== "src/ts/lazy.ts" ||
  lazyArtifacts.trace.opCount !== 3 ||
  lazyArtifacts.ir?.ops.length !== 3 ||
  lazyArtifacts.kernelPlan?.dispatchCount !== 2 ||
  lazyGraph.compileSupport().supported !== true ||
  lazyGraph.trace().ops.map((op) => op.op).join("|") !== "linear|activation|linear" ||
  lazyModuleGraph.trace().ops.map((op) => op.op).join("|") !== "linear|activation|linear" ||
  lazyModuleArtifacts.kernelPlan?.dispatchCount !== 2 ||
  frontend.lazy.moduleCompileSupport(lazyModule, { inputShape: [2] }).supported !== true
) {
  throw new Error("zgml/lazy must expose a TS-authored LazyTensor -> IR -> KernelPlan frontend");
}

if (frontend.inspection.inspectionManifest.policyOwner !== "src/ts/inspection.ts") {
  throw new Error("zgml/frontend must expose the TS-authored inspection namespace");
}

if (
  typeof inspection.runtimeProfileExpectation !== "function" ||
  typeof inspection.runtimeProfileHasNoFallback !== "function" ||
  typeof inspection.runtimeProfileHasNoSync !== "function" ||
  typeof inspection.runtimeProfileHasNoInvalidRuntimePatches !== "function" ||
  typeof inspection.requireNoFallbackRuntimeProfile !== "function" ||
  typeof inspection.requireNoSyncRuntimeProfile !== "function" ||
  typeof inspection.requireRuntimePatchValidProfile !== "function" ||
  typeof inspection.requireHotRuntimeProfile !== "function"
) {
  throw new Error("zgml/inspection must expose runtime profile expectation helpers");
}

if (frontend.modelSource.modelSourceManifest.policyOwner !== "src/ts/model_source.ts") {
  throw new Error("zgml/frontend must expose the TS-authored model source namespace");
}

if (frontend.tensor.tensorManifest.policyOwner !== "src/ts/tensor.ts") {
  throw new Error("zgml/frontend must expose the TS-authored tensor namespace");
}

if (frontend.nn.nnManifest.policyOwner !== "src/ts/nn.ts") {
  throw new Error("zgml/frontend must expose the TS-authored nn namespace");
}

if (frontend.program.programManifest.runtimePath !== "Program -> Session -> StepParams") {
  throw new Error("zgml/frontend must expose the TS-authored Program namespace");
}

if (frontend.session.sessionManifest.runtimePath !== "Program -> Session -> StepParams") {
  throw new Error("zgml/frontend must expose the TS-authored Session namespace");
}

if (frontend.stepParamsFacade.stepParamsManifest.runtimePath !== "Program -> Session -> StepParams") {
  throw new Error("zgml/frontend must expose the TS-authored StepParams namespace");
}

expectTsProductManifest(loss.lossManifest, "src/ts/loss.ts", "zgml/loss");
expectTsProductManifest(optim.optimManifest, "src/ts/optim.ts", "zgml/optim");
expectTsProductManifest(train.trainManifest, "src/ts/train.ts", "zgml/train");
expectTsProductManifest(checkpoint.checkpointManifest, "src/ts/checkpoint.ts", "zgml/checkpoint");
expectTsProductManifest(compile.compileManifest, "src/ts/compile.ts", "zgml/compile");
expectTsProductManifest(inspection.inspectionManifest, "src/ts/inspection.ts", "zgml/inspection");
expectTsProductManifest(modelSource.modelSourceManifest, "src/ts/model_source.ts", "zgml/model_source");
expectTsProductManifest(tensor.tensorManifest, "src/ts/tensor.ts", "zgml/tensor");
expectTsProductManifest(nn.nnManifest, "src/ts/nn.ts", "zgml/nn");
expectTsProductManifest(program.programManifest, "src/ts/program.ts", "zgml/program");
expectTsProductManifest(session.sessionManifest, "src/ts/session.ts", "zgml/session");
expectTsProductManifest(stepParamsFacade.stepParamsManifest, "src/ts/step_params.ts", "zgml/step_params");
expectTsProductManifest(nativeBuffer.nativeBufferManifest, "src/ts/native_buffer.ts", "zgml/native_buffer");
expectTsProductManifest(programDevice.programDeviceManifest, "src/ts/program_device.ts", "zgml/program_device");

if (loss.lossManifest.source !== "ts" || typeof loss.createLossTrainHelpers !== "function") {
  throw new Error("zgml/loss must expose TS-authored loss helper policy");
}

if (optim.optimManifest.source !== "ts" || typeof optim.createOptimNamespace !== "function") {
  throw new Error("zgml/optim must expose TS-authored optimizer helper policy");
}

if (train.trainManifest.source !== "ts" || typeof train.createLossTrainHelpers !== "function") {
  throw new Error("zgml/train must expose TS-authored train helper policy");
}

if (checkpoint.checkpointManifest.source !== "ts" || typeof checkpoint.createCheckpointHelpers !== "function") {
  throw new Error("zgml/checkpoint must expose TS-authored checkpoint helper policy");
}

if (
  compile.compileManifest.source !== "ts" ||
  typeof compile.traceCompilerArtifacts !== "function" ||
  typeof compile.requireCompileSupport !== "function" ||
  typeof compile.require_compile_support !== "function"
) {
  throw new Error("zgml/compile must expose TS-authored compile helper policy");
}

if (lazy.lazyManifest.source !== "ts" || typeof lazy.input !== "function" || lazy.input([1]).tensorProgramIr()?.kind !== "tensor-program-ir") {
  throw new Error("zgml/lazy must expose TS-authored lazy tensor helper policy");
}

if (inspection.inspectionManifest.source !== "ts" || inspection.programExecutionModeFromCapabilities(null) !== "compile-only") {
  throw new Error("zgml/inspection must expose TS-authored inspection evidence policy");
}

if (modelSource.modelSourceManifest.source !== "ts" || modelSource.normalizeLoadModelKind("smollm") !== "smollm-135m") {
  throw new Error("zgml/model_source must expose TS-authored model source policy");
}

if (tensor.tensorManifest.source !== "ts" || typeof tensor.createTensorFacadeHelpers !== "function") {
  throw new Error("zgml/tensor must expose TS-authored tensor helper policy");
}

if (nn.nnManifest.source !== "ts" || typeof nn.createNnNamespace !== "function") {
  throw new Error("zgml/nn must expose TS-authored nn helper policy");
}

if (program.programManifest.source !== "ts" || typeof program.createGenericProgramFacadeHelpers !== "function") {
  throw new Error("zgml/program must expose TS-authored Program helper policy");
}

if (session.sessionManifest.source !== "ts" || typeof session.createSessionLiveFacadeHelpers !== "function") {
  throw new Error("zgml/session must expose TS-authored Session helper policy");
}

if (stepParamsFacade.stepParamsManifest.source !== "ts" || typeof stepParamsFacade.stepParamsCompatibilityResult !== "function") {
  throw new Error("zgml/step_params must expose TS-authored StepParams helper policy");
}

if (nativeBuffer.nativeBufferManifest.source !== "ts" || typeof nativeBuffer.nativeBufferWriteInfo !== "function") {
  throw new Error("zgml/native_buffer must expose TS-authored NativeBuffer helper policy");
}

if (programDevice.programDeviceManifest.source !== "ts" || typeof programDevice.createProgramDeviceClass !== "function") {
  throw new Error("zgml/program_device must expose TS-authored ProgramDevice helper policy");
}

const missingNativeSentinels = runtimeNativeApiContract.missingRequiredNativeApiSentinelExports();
if (missingNativeSentinels.length !== 0) {
  throw new Error(`zgml/runtime/native_api_contract must include the TS-owned PyTorch-like root helpers: ${missingNativeSentinels.join(", ")}`);
}

const compatibility = frontend.stepParams.stepParamsCompatibilityResult({
  kind: "frontend-smoke",
  signature: "frontend-smoke",
}, null);
if (!compatibility.accepted || compatibility.contractSignature !== "frontend-smoke") {
  throw new Error("zgml/frontend StepParams helpers are unavailable");
}

expectNativeAdapterEvidence(nodeAdapter.nodeAdapterEvidence, "node", "zgml/adapters/node");

if (nodeAdapter.nodeAdapterManifest?.concreteRuntime?.runtimeLoad?.runtimeKind !== "node-ffi-runtime") {
  throw new Error("zgml/adapters/node must expose concrete runtime-load evidence");
}

expectConcreteRuntimeLoaderEvidence(nodeAdapter.nodeAdapterManifest?.concreteRuntime?.loader, "zgml/adapters/node");

expectNativeAdapterEvidence(bunAdapter.bunAdapterEvidence, "bun", "zgml/adapters/bun");

if (bunAdapter.bunAdapterManifest?.concreteRuntime?.runtimeLoad?.runtimeKind !== "bun-ffi-runtime") {
  throw new Error("zgml/adapters/bun must expose concrete runtime-load evidence");
}

expectConcreteRuntimeLoaderEvidence(bunAdapter.bunAdapterManifest?.concreteRuntime?.loader, "zgml/adapters/bun");

if (nativeAdapter.nativeLibraryFilename("so") !== "libzgml_c.so") {
  throw new Error("zgml/adapters/native helpers are unavailable");
}

if (coreShape.shapeScalarCount([2, 5]) !== 10) {
  throw new Error("zgml/core/* package subpaths must expose TS core helpers");
}

if (runtimeKernelPlan.kernelPlanManifest.policyOwner !== "src/ts/runtime/kernel_plan.ts") {
  throw new Error("zgml/runtime/kernel_plan must expose the TS KernelPlan facade manifest");
}

if (
  topLevelKernelPlan.kernelPlanManifest.policyOwner !== "src/ts/runtime/kernel_plan.ts" ||
  typeof topLevelKernelPlan.kernelPlanSignature !== "function" ||
  typeof topLevelKernelPlan.acceptsKernelPlan !== "function" ||
  typeof topLevelKernelPlan.requireKernelPlan !== "function" ||
  typeof topLevelKernelPlan.matchesKernelPlanSignature !== "function"
) {
  throw new Error("zgml/kernel_plan must expose the top-level TS KernelPlan facade");
}

if (
  runtimeNativeKernelContract.nativeKernelContractManifest.policyOwner !== "src/ts/runtime/native_kernel_contract.ts" ||
  runtimeNativeKernelContract.kernelNameForNativeModuleDesc({ kind: 1 }) !== "linear" ||
  runtimeNativeKernelContract.nativeModuleOpDescAbiFields({ kind: 8, a: 2 }).a !== 2
) {
  throw new Error("zgml/runtime/native_kernel_contract must expose native descriptor policy");
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
  runtimeNativeAbiConstants.adapterBufferStorageAliases({ externalResource: 7 }).bufferStorageExternalResource !== 7 ||
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
  throw new Error("zgml/runtime/* package subpaths must expose runtime-owned TS manifests");
}

if (runtimeNativeApiContract.nativeApiContractSignature() !== runtimeNativeApiContract.formatNativeApiContractSignature(runtimeNativeApiContract.nativeApiContractSignatureParts())) {
  throw new Error("zgml/runtime/native_api_contract must expose the TS native API contract");
}

const subpathCompatibility = runtimeStepParams.stepParamsCompatibilityResult({
  kind: "subpath-smoke",
  signature: "subpath-smoke",
}, null);
if (!subpathCompatibility.accepted || subpathCompatibility.contractSignature !== "subpath-smoke") {
  throw new Error("zgml/runtime/* package subpaths must expose TS runtime helpers");
}

if (typeof nnLinear.createLinearModuleClass !== "function") {
  throw new Error("zgml/nn/* package subpaths must expose TS nn helpers");
}

if (typeof trainMode.setModuleTraining !== "function") {
  throw new Error("zgml/train/* package subpaths must expose TS train helpers");
}

console.log("zgml frontend package smoke ok");
