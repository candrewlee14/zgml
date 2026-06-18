import { HostResourceRegistry, WasmExternalResourceBridge, WasmSessionBindingBridge, WasmWebGpuDevice, WasmWebGpuLlamaAttentionProjectionExecutor, WasmWebGpuLlamaBlockPipelineExecutor, WasmWebGpuLlamaBlockProjectionExecutor, WasmWebGpuLlamaEmbeddingProjectionExecutor, WasmWebGpuLlamaKvProjectionExecutor, WasmWebGpuLlamaResourceProgram, WasmWebGpuLlamaResourceSession, WasmWebGpuLlamaRmsNormProjectionExecutor, WasmWebGpuLlamaTokenWindowExecutor, WasmWebGpuLlamaTokenWindowPatternExecutor, WasmWebGpuSessionRuntime, WasmWebGpuTinyLinearProgram, biasedQwen2ProofConfig, externalResourceAccessFlags, llama3RopeProofConfig, mistralSlidingWindowProofConfig, qwen3QkNormProofConfig, requestBrowserWebGpuDeviceInfo, safetensorsModelResourceSpecs, smollm3NopeProofConfig, wasmWebgpuInterop, webgpuBufferUsage } from "./host_resources.mjs";

const logEl = document.getElementById("log");
const ok = 0;
const invalidArgument = 1;
const shapeMismatch = 3;
const unsupported = 5;
const executeOutputNone = 0;
const executeOutputLogits = 1;
const autoKind = 0;
const tinyLinearKind = 1;
const tinyLlamaKind = 2;
const smollm135mKind = 3;
const tinyLlama2LayerKind = 4;
const backendCpu = 1;
const backendWebGpu = 3;
const tinyLlamaVocabSize = 8;
const expectedAbiVersion = 6;
const featureBufferHandle = 1n << 0n;
const featureModelAuto = 1n << 1n;
const featureRuntimeProfile = 1n << 2n;
const featureWebGpuCompileOnly = 1n << 3n;
const featureWasmExports = 1n << 4n;
const featureNativeBufferIo = 1n << 5n;
const featureNativeArgmax = 1n << 6n;
const featureNativeExecuteArgmax = 1n << 7n;
const featureNativeTopKSample = 1n << 8n;
const featureProgramRequirements = 1n << 9n;
const featureProgramOutputBuffer = 1n << 10n;
const featureNativeGenerateSample = 1n << 11n;
const featureNativeGenerateArgmax = 1n << 12n;
const featureSessionModelBinding = 1n << 13n;
const featureExternalBuffer = 1n << 14n;
const featureExternalResourceBuffer = 1n << 15n;
const featureLlamaKvResourceBinding = 1n << 16n;
const featureLlamaKvCacheRequirements = 1n << 17n;
const featureProgramBufferFactory = 1n << 18n;
const featureExternalResourceAccess = 1n << 19n;
const featureModelInspection = 1n << 20n;
const featureProgramModelCompatibility = 1n << 21n;
const featureBufferInspection = 1n << 22n;
const featureSessionInspection = 1n << 23n;
const featureProgramResourceInspection = 1n << 24n;
const featureProgramMemoryInspection = 1n << 25n;
const featureProgramShapeInspection = 1n << 26n;
const featureProgramPatchEnvelopeInspection = 1n << 27n;
const featureSessionBindingShapeInspection = 1n << 28n;
const featureNativeWgpuExecution = 1n << 29n;
const featureProgramDeviceBuffer = 1n << 30n;
const featureProgramDeviceBufferImport = 1n << 31n;
const featureProgramDispatchPlanInspection = 1n << 32n;
const featureAbiStructSize = 1n << 33n;
const featureExperimentalLlamaWgpuExecution = 1n << 34n;
const featureModelPathProbe = 1n << 35n;
const featureSupportedCheckpoints = 1n << 36n;
const featureSafetensorsHeaderProbe = 1n << 37n;
const featureSafetensorsDataLoad = 1n << 38n;
const featureSafetensorsDataProbe = 1n << 39n;
const featureNativeTinyMlp = 1n << 40n;
const bufferStorageHost = 1;
const bufferStorageExternalResource = 2;
const resourceAccessReadWrite = 3;
const programBufferWeights = 1;
const programBufferBias = 2;
const programBufferInput = 3;
const programBufferOutput = 4;
const abiStructRuntimeInfo = 1;
const abiStructModelDesc = 2;
const abiStructModelInspection = 4;
const abiStructProgramModelCompatibility = 5;
const abiStructSessionInspection = 6;
const abiStructCompileDesc = 7;
const abiStructBindDesc = 8;
const abiStructBufferDesc = 9;
const abiStructBufferInspection = 10;
const abiStructExternalResourceDesc = 11;
const abiStructDeviceBufferImportDesc = 12;
const abiStructBufferBindDesc = 13;
const abiStructLlamaKvCacheBindDesc = 14;
const abiStructLlamaBufferBindDesc = 15;
const abiStructStepDesc = 16;
const abiStructStepResult = 17;
const abiStructTokenStepDesc = 18;
const abiStructTokenAdvanceDesc = 19;
const abiStructTokenAdvanceTokensDesc = 20;
const abiStructTokenPrefillDesc = 21;
const abiStructTokenExecuteDesc = 22;
const abiStructTokenArgmaxDesc = 23;
const abiStructTokenArgmaxResult = 24;
const abiStructTokenExecuteArgmaxDesc = 25;
const abiStructTokenGenerateArgmaxDesc = 26;
const abiStructTokenGenerateArgmaxResult = 27;
const abiStructTokenSampleDesc = 28;
const abiStructTokenSampleResult = 29;
const abiStructTokenExecuteSampleDesc = 30;
const abiStructTokenGenerateSampleDesc = 31;
const abiStructTokenGenerateSampleResult = 32;
const abiStructProgramRequirements = 33;
const abiStructLlamaKvCacheRequirements = 34;
const abiStructProgramInspection = 35;
const abiStructLlamaProgramInspection = 36;
const abiStructRuntimeProfile = 37;
const abiStructSafetensorsHeaderProbeDesc = 38;
const abiStructSafetensorsDataLoadDesc = 39;

let runtimeInfoSize = 24;
let modelDescSize = 12;
let modelInspectionSize = 104;
let compileDescSize = 16;
let bindDescSize = 32;
let bufferDescSize = 4;
let bufferInspectionSize = 48;
let externalResourceDescSize = 20;
let llamaKvCacheBindDescSize = 12;
let llamaBufferBindDescSize = 12;
let stepDescSize = 16;
let stepResultSize = 4;
let tokenStepDescSize = 12;
let tokenAdvanceDescSize = 4;
let tokenAdvanceTokensDescSize = 8;
let tokenPrefillDescSize = 16;
let tokenExecuteDescSize = 24;
let tokenArgmaxDescSize = 12;
let tokenArgmaxResultSize = 8;
let tokenExecuteArgmaxDescSize = 12;
let tokenGenerateArgmaxDescSize = 20;
let tokenGenerateArgmaxResultSize = 12;
let tokenSampleDescSize = 24;
let tokenSampleResultSize = 8;
let tokenExecuteSampleDescSize = 24;
let tokenGenerateSampleDescSize = 32;
let tokenGenerateSampleResultSize = 12;
const handleSize = 4;
let deviceBufferImportDescSize = 24;
let programRequirementsSize = 52;
let programModelCompatibilitySize = 16;
let sessionInspectionSize = 80;
let programInspectionSize = 280;
let llamaProgramInspectionSize = 120;
let llamaKvCacheRequirementsSize = 32;
let runtimeProfileSize = 168;
let safetensorsHeaderProbeDescSize = 16;
let safetensorsDataLoadDescSize = 16;

const tinyLlamaDefaultShape = Object.freeze({
  ffnSize: 8,
  hiddenSize: 4,
  vocabSize: tinyLlamaVocabSize,
});

function tinyLlamaRootTensorSpecsFor(shape = tinyLlamaDefaultShape) {
  return [
    ["model.embed_tokens.weight", [shape.vocabSize, shape.hiddenSize]],
    ["model.norm.weight", [shape.hiddenSize]],
    ["lm_head.weight", [shape.vocabSize, shape.hiddenSize]],
  ];
}

const tinyLlamaRootTensorSpecs = tinyLlamaRootTensorSpecsFor();

function tinyLlamaLayerTensorSpecs(layer, shape = tinyLlamaDefaultShape) {
  const qSize = shape.qSize ?? shape.hiddenSize;
  const kvSize = shape.kvSize ?? shape.hiddenSize;
  return [
    [`model.layers.${layer}.self_attn.q_proj.weight`, [qSize, shape.hiddenSize]],
    [`model.layers.${layer}.self_attn.k_proj.weight`, [kvSize, shape.hiddenSize]],
    [`model.layers.${layer}.self_attn.v_proj.weight`, [kvSize, shape.hiddenSize]],
    [`model.layers.${layer}.self_attn.o_proj.weight`, [shape.hiddenSize, qSize]],
    [`model.layers.${layer}.mlp.gate_proj.weight`, [shape.ffnSize, shape.hiddenSize]],
    [`model.layers.${layer}.mlp.up_proj.weight`, [shape.ffnSize, shape.hiddenSize]],
    [`model.layers.${layer}.mlp.down_proj.weight`, [shape.hiddenSize, shape.ffnSize]],
    [`model.layers.${layer}.input_layernorm.weight`, [shape.hiddenSize]],
    [`model.layers.${layer}.post_attention_layernorm.weight`, [shape.hiddenSize]],
  ];
}

function tinyLlamaQkNormLayerTensorSpecs(layer, shape = tinyLlamaDefaultShape) {
  const headSize = shape.attentionHeadSize ?? shape.headSize ?? shape.hiddenSize;
  return [
    ...tinyLlamaLayerTensorSpecs(layer, shape),
    [`model.layers.${layer}.self_attn.q_norm.weight`, [headSize]],
    [`model.layers.${layer}.self_attn.k_norm.weight`, [headSize]],
  ];
}

function tinyLlamaBiasedLayerTensorSpecs(layer, shape = tinyLlamaDefaultShape) {
  const qSize = shape.qSize ?? shape.hiddenSize;
  const kvSize = shape.kvSize ?? shape.hiddenSize;
  return [
    ...tinyLlamaLayerTensorSpecs(layer, shape),
    [`model.layers.${layer}.self_attn.q_proj.bias`, [qSize]],
    [`model.layers.${layer}.self_attn.k_proj.bias`, [kvSize]],
    [`model.layers.${layer}.self_attn.v_proj.bias`, [kvSize]],
    [`model.layers.${layer}.self_attn.o_proj.bias`, [shape.hiddenSize]],
    [`model.layers.${layer}.mlp.gate_proj.bias`, [shape.ffnSize]],
    [`model.layers.${layer}.mlp.up_proj.bias`, [shape.ffnSize]],
    [`model.layers.${layer}.mlp.down_proj.bias`, [shape.hiddenSize]],
  ];
}

const tinyLlamaTensorSpecs = [
  ...tinyLlamaRootTensorSpecs,
  ...tinyLlamaLayerTensorSpecs(0),
];

const tinyLlamaTwoLayerTensorSpecs = [
  ...tinyLlamaRootTensorSpecs,
  ...tinyLlamaLayerTensorSpecs(0),
  ...tinyLlamaLayerTensorSpecs(1),
];

const tinyLlamaThreeLayerTensorSpecs = [
  ...tinyLlamaRootTensorSpecs,
  ...tinyLlamaLayerTensorSpecs(0),
  ...tinyLlamaLayerTensorSpecs(1),
  ...tinyLlamaLayerTensorSpecs(2),
];

const tinyLlamaTiedRootTensorSpecs = tinyLlamaRootTensorSpecs.filter(([name]) => name !== "lm_head.weight");

const tinyLlamaTwoLayerTiedTensorSpecs = [
  ...tinyLlamaTiedRootTensorSpecs,
  ...tinyLlamaLayerTensorSpecs(0),
  ...tinyLlamaLayerTensorSpecs(1),
];

const widerTinyLlamaShape = Object.freeze({
  contextLength: 3,
  ffnSize: 12,
  hiddenSize: 6,
  layers: 2,
  vocabSize: 10,
});

const widerTinyLlamaTwoLayerTensorSpecs = [
  ...tinyLlamaRootTensorSpecsFor(widerTinyLlamaShape),
  ...tinyLlamaLayerTensorSpecs(0, widerTinyLlamaShape),
  ...tinyLlamaLayerTensorSpecs(1, widerTinyLlamaShape),
];

const realisticHeadTinyLlamaShape = Object.freeze({
  attentionHeadSize: 128,
  contextLength: 96,
  ffnSize: 32,
  hiddenSize: 128,
  kvSize: 128,
  layers: 1,
  vocabSize: 5,
});

const realisticHeadTinyLlamaTensorSpecs = [
  ...tinyLlamaRootTensorSpecsFor(realisticHeadTinyLlamaShape),
  ...tinyLlamaLayerTensorSpecs(0, realisticHeadTinyLlamaShape),
];

const realisticGqaTinyLlamaShape = Object.freeze({
  attentionHeadSize: 128,
  contextLength: 2,
  ffnSize: 8,
  hiddenSize: 256,
  kvSize: 128,
  layers: 2,
  vocabSize: 6,
});

const realisticGqaTinyLlamaTwoLayerTensorSpecs = [
  ...tinyLlamaRootTensorSpecsFor(realisticGqaTinyLlamaShape),
  ...tinyLlamaLayerTensorSpecs(0, realisticGqaTinyLlamaShape),
  ...tinyLlamaLayerTensorSpecs(1, realisticGqaTinyLlamaShape),
];

const groupedTinyLlamaShape = Object.freeze({
  attentionHeadSize: 2,
  contextLength: 3,
  ffnSize: 8,
  hiddenSize: 8,
  kvSize: 4,
  layers: 2,
  vocabSize: 6,
});

const groupedTinyLlamaTwoLayerTensorSpecs = [
  ...tinyLlamaRootTensorSpecsFor(groupedTinyLlamaShape),
  ...tinyLlamaLayerTensorSpecs(0, groupedTinyLlamaShape),
  ...tinyLlamaLayerTensorSpecs(1, groupedTinyLlamaShape),
];

const qkNormGroupedTinyLlamaTwoLayerTensorSpecs = [
  ...tinyLlamaRootTensorSpecsFor(groupedTinyLlamaShape),
  ...tinyLlamaQkNormLayerTensorSpecs(0, groupedTinyLlamaShape),
  ...tinyLlamaQkNormLayerTensorSpecs(1, groupedTinyLlamaShape),
];

const biasedGroupedTinyLlamaTwoLayerTensorSpecs = [
  ...tinyLlamaRootTensorSpecsFor(groupedTinyLlamaShape),
  ["lm_head.bias", [groupedTinyLlamaShape.vocabSize]],
  ...tinyLlamaBiasedLayerTensorSpecs(0, groupedTinyLlamaShape),
  ...tinyLlamaBiasedLayerTensorSpecs(1, groupedTinyLlamaShape),
];

const longContextGqaTinyLlamaShape = Object.freeze({
  ...groupedTinyLlamaShape,
  contextLength: 32,
});

const longContextGqaTinyLlamaTwoLayerTensorSpecs = [
  ...tinyLlamaRootTensorSpecsFor(longContextGqaTinyLlamaShape),
  ...tinyLlamaLayerTensorSpecs(0, longContextGqaTinyLlamaShape),
  ...tinyLlamaLayerTensorSpecs(1, longContextGqaTinyLlamaShape),
];

const longSlidingWindowTinyLlamaShape = Object.freeze({
  ...groupedTinyLlamaShape,
  contextLength: 96,
});

const longSlidingWindowTinyLlamaTwoLayerTensorSpecs = [
  ...tinyLlamaRootTensorSpecsFor(longSlidingWindowTinyLlamaShape),
  ...tinyLlamaLayerTensorSpecs(0, longSlidingWindowTinyLlamaShape),
  ...tinyLlamaLayerTensorSpecs(1, longSlidingWindowTinyLlamaShape),
];

const mhaTinyLlamaShape = Object.freeze({
  attentionHeadSize: 2,
  contextLength: 3,
  ffnSize: 8,
  hiddenSize: 8,
  kvSize: 8,
  layers: 2,
  vocabSize: 6,
});

const mhaTinyLlamaTwoLayerTensorSpecs = [
  ...tinyLlamaRootTensorSpecsFor(mhaTinyLlamaShape),
  ...tinyLlamaLayerTensorSpecs(0, mhaTinyLlamaShape),
  ...tinyLlamaLayerTensorSpecs(1, mhaTinyLlamaShape),
];

const mqaTinyLlamaShape = Object.freeze({
  attentionHeadSize: 2,
  contextLength: 3,
  ffnSize: 8,
  hiddenSize: 8,
  kvSize: 2,
  layers: 2,
  vocabSize: 6,
});

const mqaTinyLlamaTwoLayerTensorSpecs = [
  ...tinyLlamaRootTensorSpecsFor(mqaTinyLlamaShape),
  ...tinyLlamaLayerTensorSpecs(0, mqaTinyLlamaShape),
  ...tinyLlamaLayerTensorSpecs(1, mqaTinyLlamaShape),
];

function tinyLlamaModelResourceSpecs(source = tinyLlamaModelResourceFileBytes(), options = {}) {
  return safetensorsModelResourceSpecs(source, options);
}

const tinyLlamaModelResourceRequirements = tinyLlamaModelResourceSpecs();
const tinyLlamaModelResourceRoles = tinyLlamaModelResourceRequirements.map((spec) => spec.role);

function expectLlamaLayerRoleDefaults() {
  const kv = new WasmWebGpuLlamaKvProjectionExecutor({ layer: 2 });
  if (
    kv.layer !== 2 ||
    kv.inputNormRole !== "model.layers.2.input_layernorm.weight" ||
    kv.kProjRole !== "model.layers.2.self_attn.k_proj.weight" ||
    kv.vProjRole !== "model.layers.2.self_attn.v_proj.weight"
  ) {
    throw new Error("LLaMA K/V projection executor did not derive layer-indexed roles");
  }

  const attention = new WasmWebGpuLlamaAttentionProjectionExecutor({ layer: 3 });
  if (
    attention.layer !== 3 ||
    attention.inputNormRole !== "model.layers.3.input_layernorm.weight" ||
    attention.qProjRole !== "model.layers.3.self_attn.q_proj.weight" ||
    attention.kProjRole !== "model.layers.3.self_attn.k_proj.weight" ||
    attention.vProjRole !== "model.layers.3.self_attn.v_proj.weight" ||
    attention.oProjRole !== "model.layers.3.self_attn.o_proj.weight"
  ) {
    throw new Error("LLaMA attention projection executor did not derive layer-indexed roles");
  }

  const block = new WasmWebGpuLlamaBlockProjectionExecutor({ layer: 4 });
  if (
    block.layer !== 4 ||
    block.inputNormRole !== "model.layers.4.input_layernorm.weight" ||
    block.qProjRole !== "model.layers.4.self_attn.q_proj.weight" ||
    block.kProjRole !== "model.layers.4.self_attn.k_proj.weight" ||
    block.vProjRole !== "model.layers.4.self_attn.v_proj.weight" ||
    block.oProjRole !== "model.layers.4.self_attn.o_proj.weight" ||
    block.postNormRole !== "model.layers.4.post_attention_layernorm.weight" ||
    block.gateProjRole !== "model.layers.4.mlp.gate_proj.weight" ||
    block.upProjRole !== "model.layers.4.mlp.up_proj.weight" ||
    block.downProjRole !== "model.layers.4.mlp.down_proj.weight"
  ) {
    throw new Error("LLaMA block projection executor did not derive layer-indexed roles");
  }

  const override = new WasmWebGpuLlamaBlockProjectionExecutor({
    gateProjRole: "custom.gate",
    layer: 5,
  });
  if (
    override.layer !== 5 ||
    override.inputNormRole !== "model.layers.5.input_layernorm.weight" ||
    override.gateProjRole !== "custom.gate" ||
    override.upProjRole !== "model.layers.5.mlp.up_proj.weight"
  ) {
    throw new Error("LLaMA block projection executor layer roles did not preserve explicit overrides");
  }

  let rejected = false;
  try {
    new WasmWebGpuLlamaKvProjectionExecutor({ layer: -1 });
  } catch {
    rejected = true;
  }
  if (!rejected) {
    throw new Error("LLaMA projection executors accepted a negative layer index");
  }
}

function expectHostResourceDescriptorImmutability() {
  const device = new WasmWebGpuDevice();
  const resource = device.createBuffer({
    byteLength: 16,
    label: "fake.immutable.descriptor",
    usage: webgpuBufferUsage.STORAGE | webgpuBufferUsage.COPY_SRC | webgpuBufferUsage.COPY_DST,
  });
  const descriptor = device.descriptor(resource, { access: "read", byteLength: 16, byteOffset: 0 });
  if (!Object.isFrozen(descriptor)) {
    throw new Error("host WebGPU resource descriptor is not immutable");
  }
  let rejected = false;
  try {
    descriptor.byteLength = 8;
  } catch {
    rejected = true;
  }
  if (!rejected || descriptor.byteLength !== 16) {
    throw new Error("host WebGPU resource descriptor byteLength mutation was not rejected");
  }
  let oversizedDescriptorRejected = false;
  try {
    device.descriptor(resource, { access: "read", byteLength: 16, byteOffset: 8 });
  } catch (err) {
    oversizedDescriptorRejected = String(err && err.message ? err.message : err).includes("range");
  }
  if (!oversizedDescriptorRejected) {
    throw new Error("host WebGPU resource descriptor accepted an oversized view");
  }
  let oversizedTableDescriptorRejected = false;
  try {
    device.resourceTable([{
      role: "fake.oversized.descriptor",
      descriptor: {
        ...descriptor,
        byteLength: 16,
        byteOffset: 8,
      },
    }]);
  } catch (err) {
    oversizedTableDescriptorRejected = String(err && err.message ? err.message : err).includes("range");
  }
  if (!oversizedTableDescriptorRejected) {
    throw new Error("host WebGPU resource table accepted an oversized descriptor view");
  }
  device.destroy();
  if (device.resourceFor(descriptor.handle) !== null) {
    throw new Error("host WebGPU resource handle still resolved after device destroy");
  }
  expectDestroyedWasmWebGpuResourceRejected(device, resource, "device-owned host WebGPU resource", 16);
}

function expectExecutorOwnedResourceCleanupForgetsHostResources() {
  const device = new WasmWebGpuDevice();
  const executor = new WasmWebGpuLlamaAttentionProjectionExecutor();
  const resource = device.createBuffer({
    byteLength: 16,
    label: "fake.executor-owned.resource",
    usage: webgpuBufferUsage.STORAGE | webgpuBufferUsage.COPY_SRC | webgpuBufferUsage.COPY_DST,
  });
  const descriptor = device.descriptor(resource, { access: "readwrite", byteLength: 16 });
  executor.ownedResources.push(resource);
  executor.destroy();
  if (device.resourceFor(descriptor.handle) !== null) {
    throw new Error("executor-owned host WebGPU resource handle still resolved after executor destroy");
  }
  expectDestroyedWasmWebGpuResourceRejected(device, resource, "executor-owned host WebGPU resource", 16);
  device.destroy();
}

function expectImportedLlamaBindingDescriptorsAreSnapshotted() {
  const device = new WasmWebGpuDevice();
  const program = new WasmWebGpuLlamaResourceProgram({
    contextLength: 4,
    device,
    kvCacheRequirements: {
      kBufferByteLength: 16,
      layers: 1,
      vBufferByteLength: 16,
    },
    program: 0x71000041,
    sessionBindingBridge: new WasmSessionBindingBridge(),
    vocabSize: tinyLlamaVocabSize,
  });
  let nextBufferHandle = 0x71010000;
  const binding = (label, byteLength, access) => {
    const resource = device.createBuffer({
      byteLength,
      label,
      usage: webgpuBufferUsage.STORAGE | webgpuBufferUsage.COPY_SRC | webgpuBufferUsage.COPY_DST,
    });
    return {
      buffer: nextBufferHandle++,
      descriptor: {
        ...device.descriptor(resource, { access, byteLength }),
      },
    };
  };
  const output = binding("fake.snapshot.output", tinyLlamaVocabSize * Float32Array.BYTES_PER_ELEMENT, "write");
  const k = binding("fake.snapshot.k", 16, "readwrite");
  const v = binding("fake.snapshot.v", 16, "readwrite");
  const prepared = program.prepareSessionResources({
    kvCache: [{ k, v }],
    output,
  });
  const session = program.createResourceSession(0x71000042, prepared);
  output.descriptor.byteLength = 4;
  output.descriptor.handle += 1000;
  k.descriptor.byteLength = 4;
  v.descriptor.access = "read";
  try {
    if (
      !Object.isFrozen(prepared.output.descriptor) ||
      !Object.isFrozen(prepared.kvCache[0].k.descriptor) ||
      !Object.isFrozen(session.bindings.output) ||
      session.bindings.output.byteLength !== tinyLlamaVocabSize * Float32Array.BYTES_PER_ELEMENT ||
      session.bindings.output.handle === output.descriptor.handle ||
      session.bindings.kvCache[0].k.byteLength !== 16 ||
      externalResourceAccessFlags(session.bindings.kvCache[0].v.access) !== externalResourceAccessFlags("readwrite") ||
      session.table.hash !== device.resourceTable(prepared.resourceDescriptors).hash
    ) {
      throw new Error("imported LLaMA binding descriptor mutation changed prepared Session evidence");
    }
  } finally {
    session.dispose();
    device.destroy();
  }
}

function expectLlamaProjectionGpuBindGroupCache() {
  const hadGpuBufferUsage = Object.prototype.hasOwnProperty.call(globalThis, "GPUBufferUsage");
  const previousGpuBufferUsage = globalThis.GPUBufferUsage;
  if (typeof globalThis.GPUBufferUsage !== "object") {
    globalThis.GPUBufferUsage = webgpuBufferUsage;
  }
  let bindGroupCreateCount = 0;
  let submitCount = 0;
  let kvTokenBufferCreateCount = 0;
  let tokenBufferCreateCount = 0;
  const kvTokenBufferSizes = [];
  const tokenBufferSizes = [];
  const computePipelineLabels = [];
  const uploadSources = [];
  const fakeGpuDevice = {
    limits: { minStorageBufferOffsetAlignment: 4 },
    queue: {
      submit() {
        submitCount += 1;
      },
      writeBuffer(_buffer, _offset, source) {
        uploadSources.push(source);
      },
    },
    createBindGroup(desc) {
      bindGroupCreateCount += 1;
      return { desc, kind: "fake-bind-group" };
    },
    createBuffer(desc) {
      if (desc.label === "zgml.wasm.webgpu.llama-kv-projection.tokens") {
        kvTokenBufferCreateCount += 1;
        kvTokenBufferSizes.push(desc.size);
      }
      if (desc.label === "zgml.wasm.webgpu.llama-block-projection.tokens") {
        tokenBufferCreateCount += 1;
        tokenBufferSizes.push(desc.size);
      }
      return {
        ...desc,
        destroy() {},
        size: desc.size,
      };
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            dispatchWorkgroups() {},
            end() {},
            setBindGroup() {},
            setPipeline() {},
          };
        },
        finish() {
          return { kind: "fake-command-buffer" };
        },
      };
    },
    createComputePipeline(desc = {}) {
      computePipelineLabels.push(desc.label ?? "");
      return {
        getBindGroupLayout() {
          return { kind: "fake-bind-group-layout" };
        },
      };
    },
    createShaderModule(desc) {
      return { desc, kind: "fake-shader-module" };
    },
  };
  const fakeDevice = new WasmWebGpuDevice({ device: fakeGpuDevice });
  const fakeDescriptor = (label, byteLength, access = "read") => {
    const resource = fakeDevice.createBuffer({
      byteLength,
      label,
      usage: webgpuBufferUsage.STORAGE | webgpuBufferUsage.COPY_SRC | webgpuBufferUsage.COPY_DST,
    });
    return fakeDevice.descriptor(resource, { access, byteLength });
  };
  const fakeModelPack = (label, offsets, byteLength = 4096) => ({
    byteLength,
    descriptor: fakeDescriptor(label, byteLength),
    offsets,
  });
  const executor = new WasmWebGpuLlamaEmbeddingProjectionExecutor();
  const kvExecutor = new WasmWebGpuLlamaKvProjectionExecutor();
  const mockKvExecutor = new WasmWebGpuLlamaKvProjectionExecutor();
  const attentionExecutor = new WasmWebGpuLlamaAttentionProjectionExecutor();
  const blockExecutor = new WasmWebGpuLlamaBlockProjectionExecutor();
  const longDecodeBlockExecutor = new WasmWebGpuLlamaBlockProjectionExecutor({ scalarProjectionMinHiddenSize: 1 });
  try {
    const projection = {
      embedding: {
        byteLength: tinyLlamaVocabSize * 4 * Float32Array.BYTES_PER_ELEMENT,
        descriptor: fakeDescriptor("fake.embedding", tinyLlamaVocabSize * 4 * Float32Array.BYTES_PER_ELEMENT),
      },
      hiddenSize: 4,
      lmHead: {
        byteLength: tinyLlamaVocabSize * 4 * Float32Array.BYTES_PER_ELEMENT,
        descriptor: fakeDescriptor("fake.lm_head", tinyLlamaVocabSize * 4 * Float32Array.BYTES_PER_ELEMENT),
      },
      vocabSize: tinyLlamaVocabSize,
    };
    const output = fakeDescriptor("fake.output", tinyLlamaVocabSize * Float32Array.BYTES_PER_ELEMENT, "write");
    const context = {
      execution: {
        outputPolicy: executeOutputLogits,
        tokens: [1],
      },
      session: {
        bindings: { output },
        device: fakeDevice,
      },
    };
    executor.submitGpuProjection(context, projection);
    executor.submitGpuProjection(context, projection);
    if (bindGroupCreateCount !== 1 || submitCount !== 2) {
      throw new Error(`LLaMA projection GPU bind-group cache created ${bindGroupCreateCount} groups for ${submitCount} submissions`);
    }
    if (uploadSources.length !== 2 || uploadSources[0] !== uploadSources[1]) {
      throw new Error("LLaMA projection GPU params upload did not reuse the executor upload buffer");
    }

    const attentionBindGroupsBefore = bindGroupCreateCount;
    const attentionProjection = {
      embedding: {
        byteLength: tinyLlamaVocabSize * 4 * Float32Array.BYTES_PER_ELEMENT,
        descriptor: fakeDescriptor("fake.attention.embedding", tinyLlamaVocabSize * 4 * Float32Array.BYTES_PER_ELEMENT),
      },
      epsilon: 1e-5,
      hiddenSize: 4,
      inputNorm: {
        byteLength: 4 * Float32Array.BYTES_PER_ELEMENT,
        descriptor: fakeDescriptor("fake.attention.input_norm", 4 * Float32Array.BYTES_PER_ELEMENT),
      },
      kProj: {
        byteLength: 4 * 4 * Float32Array.BYTES_PER_ELEMENT,
        descriptor: fakeDescriptor("fake.attention.k_proj", 4 * 4 * Float32Array.BYTES_PER_ELEMENT),
      },
      lmHead: {
        byteLength: tinyLlamaVocabSize * 4 * Float32Array.BYTES_PER_ELEMENT,
        descriptor: fakeDescriptor("fake.attention.lm_head", tinyLlamaVocabSize * 4 * Float32Array.BYTES_PER_ELEMENT),
      },
      modelPack: fakeModelPack("fake.attention.model-pack", {
        embedding: 0,
        inputNorm: 256,
        qProj: 512,
        kProj: 768,
        vProj: 1024,
        oProj: 1280,
        norm: 1536,
        lmHead: 1792,
      }),
      norm: {
        byteLength: 4 * Float32Array.BYTES_PER_ELEMENT,
        descriptor: fakeDescriptor("fake.attention.norm", 4 * Float32Array.BYTES_PER_ELEMENT),
      },
      oProj: {
        byteLength: 4 * 4 * Float32Array.BYTES_PER_ELEMENT,
        descriptor: fakeDescriptor("fake.attention.o_proj", 4 * 4 * Float32Array.BYTES_PER_ELEMENT),
      },
      qProj: {
        byteLength: 4 * 4 * Float32Array.BYTES_PER_ELEMENT,
        descriptor: fakeDescriptor("fake.attention.q_proj", 4 * 4 * Float32Array.BYTES_PER_ELEMENT),
        rows: 4,
      },
      vProj: {
        byteLength: 4 * 4 * Float32Array.BYTES_PER_ELEMENT,
        descriptor: fakeDescriptor("fake.attention.v_proj", 4 * 4 * Float32Array.BYTES_PER_ELEMENT),
      },
      vocabSize: tinyLlamaVocabSize,
    };
    const attentionContext = {
      execution: {
        outputPolicy: executeOutputLogits,
        tokens: [1],
      },
      session: {
        bindings: { output },
        device: fakeDevice,
      },
      stepParams: {
        contextLength: 4,
        endPosition: 1,
        startPosition: 0,
      },
    };
    const attentionSlots = () => ({
      k: {
        descriptor: fakeDescriptor("fake.attention.k_cache", 4 * 4 * Float32Array.BYTES_PER_ELEMENT),
        stride: 4,
      },
      v: {
        descriptor: fakeDescriptor("fake.attention.v_cache", 4 * 4 * Float32Array.BYTES_PER_ELEMENT),
        stride: 4,
      },
    });
    const firstSlots = attentionSlots();
    const secondSlots = attentionSlots();
    attentionExecutor.submitGpuProjection(attentionContext, attentionProjection, firstSlots);
    attentionExecutor.submitGpuProjection(attentionContext, attentionProjection, firstSlots);
    attentionExecutor.submitGpuProjection(attentionContext, attentionProjection, secondSlots);
    if (bindGroupCreateCount - attentionBindGroupsBefore !== 2) {
      throw new Error(`LLaMA attention GPU bind-group cache did not account for K/V slot descriptors: ${bindGroupCreateCount - attentionBindGroupsBefore}`);
    }

    const attentionOptionSnapshots = [];
    const originalAttentionProjectionForSession = attentionExecutor.projectionForSession.bind(attentionExecutor);
    const originalAttentionSlotsMethod = attentionExecutor.attentionSlots.bind(attentionExecutor);
    const originalAttentionSubmitProjection = attentionExecutor.submitGpuProjection.bind(attentionExecutor);
    attentionExecutor.projectionForSession = () => attentionProjection;
    attentionExecutor.attentionSlots = () => firstSlots;
    attentionExecutor.submitGpuProjection = (contextArg, projectionArg, slotsArg, optionsArg) => {
      attentionOptionSnapshots.push({
        outputPolicy: optionsArg.outputPolicy,
        position: optionsArg.position,
        ref: optionsArg,
        seq: optionsArg.seq,
        token: optionsArg.token,
      });
      return originalAttentionSubmitProjection(contextArg, projectionArg, slotsArg, optionsArg);
    };
    const attentionWindowTokens = [4, 5, 6];
    const attentionWindowResult = attentionExecutor.executeTokensSync({
      execution: {
        outputLen: tinyLlamaVocabSize,
        outputPolicy: executeOutputLogits,
        outputPtr: 0,
        tokens: attentionWindowTokens,
        tokensLen: attentionWindowTokens.length,
      },
      runtime: {
        status: { ok, unsupported },
      },
      session: {
        bindings: { output },
        device: fakeDevice,
        vocabSize: tinyLlamaVocabSize,
      },
      stepParams: {
        contextLength: 4,
        endPosition: 4,
        startPosition: 1,
        tokenCount: attentionWindowTokens.length,
      },
    });
    attentionExecutor.projectionForSession = originalAttentionProjectionForSession;
    attentionExecutor.attentionSlots = originalAttentionSlotsMethod;
    attentionExecutor.submitGpuProjection = originalAttentionSubmitProjection;
    if (
      attentionWindowResult.status !== ok ||
      attentionWindowResult.backendDispatchCount !== 3 ||
      attentionWindowResult.commandCount !== 3
    ) {
      throw new Error("LLaMA attention GPU token window did not dispatch each token through the projection path");
    }
    if (
      attentionOptionSnapshots.length !== 3 ||
      attentionOptionSnapshots[0].ref !== attentionExecutor.projectionStepOptionsScratch ||
      attentionOptionSnapshots[1].ref !== attentionExecutor.projectionStepOptionsScratch ||
      attentionOptionSnapshots[2].ref !== attentionExecutor.projectionStepOptionsScratch ||
      attentionOptionSnapshots[0].outputPolicy !== executeOutputNone ||
      attentionOptionSnapshots[1].outputPolicy !== executeOutputNone ||
      attentionOptionSnapshots[2].outputPolicy !== executeOutputLogits ||
      attentionOptionSnapshots[0].position !== 1 ||
      attentionOptionSnapshots[1].position !== 2 ||
      attentionOptionSnapshots[2].position !== 3 ||
      attentionOptionSnapshots[0].seq !== 2 ||
      attentionOptionSnapshots[1].seq !== 3 ||
      attentionOptionSnapshots[2].seq !== 4 ||
      attentionOptionSnapshots[0].token !== 4 ||
      attentionOptionSnapshots[1].token !== 5 ||
      attentionOptionSnapshots[2].token !== 6
    ) {
      throw new Error("LLaMA attention GPU token window did not reuse step options scratch");
    }

    const kvTokenBuffersBefore = kvTokenBufferCreateCount;
    const kvTokenBufferSizesBefore = kvTokenBufferSizes.length;
    const kvBindGroupsBefore = bindGroupCreateCount;
    const kvSubmitsBefore = submitCount;
    const kvUploadsBefore = uploadSources.length;
    const kvWindowContext = (tokens) => ({
      execution: {
        outputPolicy: executeOutputNone,
        tokens,
      },
      session: {
        device: fakeDevice,
      },
      stepParams: {
        contextLength: 4,
        endPosition: tokens.length,
        startPosition: 0,
        tokenCount: tokens.length,
      },
    });
    const kvProjection = { ...attentionProjection };
    kvExecutor.submitGpuProjection(kvWindowContext([0, 1]), kvProjection, firstSlots);
    kvExecutor.submitGpuProjection(kvWindowContext([2, 3]), kvProjection, firstSlots);
    kvExecutor.submitGpuProjection(kvWindowContext([0, 1, 2]), kvProjection, firstSlots);
    kvExecutor.submitGpuProjection(kvWindowContext([3, 2, 1, 0]), kvProjection, firstSlots);
    const kvWindowUploads = uploadSources.slice(kvUploadsBefore);
    const kvWindowTokenBufferSizes = kvTokenBufferSizes.slice(kvTokenBufferSizesBefore);
    if (bindGroupCreateCount - kvBindGroupsBefore !== 2) {
      throw new Error(`LLaMA K/V projection GPU bind-group cache did not track token buffer bucket growth: ${bindGroupCreateCount - kvBindGroupsBefore}`);
    }
    if (kvTokenBufferCreateCount - kvTokenBuffersBefore !== 2) {
      throw new Error(`LLaMA K/V projection GPU token buffer was not reused across capacity buckets: ${kvTokenBufferCreateCount - kvTokenBuffersBefore}`);
    }
    if (submitCount - kvSubmitsBefore !== 4) {
      throw new Error(`LLaMA K/V projection GPU submit evidence mismatch: ${submitCount - kvSubmitsBefore}`);
    }
    if (
      kvWindowTokenBufferSizes.length !== 2 ||
      kvWindowTokenBufferSizes[0] !== 2 * Uint32Array.BYTES_PER_ELEMENT ||
      kvWindowTokenBufferSizes[1] !== 4 * Uint32Array.BYTES_PER_ELEMENT
    ) {
      throw new Error(`LLaMA K/V projection GPU token buffers did not grow by reusable capacity bucket: ${kvWindowTokenBufferSizes.join(",")}`);
    }
    if (
      kvWindowUploads.length !== 8 ||
      kvWindowUploads[0] !== kvWindowUploads[2] ||
      kvWindowUploads[0] !== kvWindowUploads[4] ||
      kvWindowUploads[0] !== kvWindowUploads[6] ||
      kvWindowUploads[1] !== kvWindowUploads[3] ||
      kvWindowUploads[1] === kvWindowUploads[5] ||
      kvWindowUploads[5] !== kvWindowUploads[7]
    ) {
      throw new Error("LLaMA K/V projection GPU params/tokens uploads did not reuse buffers by capacity");
    }

    const mockOnlyDevice = new WasmWebGpuDevice();
    const mockOnlyDescriptor = (label, byteLength, access = "read") => {
      const resource = mockOnlyDevice.createBuffer({
        byteLength,
        label,
        usage: webgpuBufferUsage.STORAGE | webgpuBufferUsage.COPY_SRC | webgpuBufferUsage.COPY_DST,
      });
      return mockOnlyDevice.descriptor(resource, { access, byteLength });
    };
    const mockKvProjection = {
      embedding: {
        descriptor: mockOnlyDescriptor("mock.kv.embedding", 4 * 2 * Float32Array.BYTES_PER_ELEMENT),
        elementCount: 8,
      },
      epsilon: 1e-5,
      hiddenSize: 2,
      inputNorm: {
        descriptor: mockOnlyDescriptor("mock.kv.input_norm", 2 * Float32Array.BYTES_PER_ELEMENT),
        elementCount: 2,
      },
      kProj: {
        descriptor: mockOnlyDescriptor("mock.kv.k_proj", 2 * 2 * Float32Array.BYTES_PER_ELEMENT),
        elementCount: 4,
      },
      layer: 0,
      vProj: {
        descriptor: mockOnlyDescriptor("mock.kv.v_proj", 2 * 2 * Float32Array.BYTES_PER_ELEMENT),
        elementCount: 4,
      },
    };
    const mockK = mockOnlyDescriptor("mock.kv.k_cache", 4 * 2 * Float32Array.BYTES_PER_ELEMENT, "write");
    const mockV = mockOnlyDescriptor("mock.kv.v_cache", 4 * 2 * Float32Array.BYTES_PER_ELEMENT, "write");
    mockKvExecutor.projectionForSession = () => mockKvProjection;
    mockKvExecutor.kvProjectionSlots = (contextArg) => ({
      k: {
        byteOffset: 0,
        descriptor: mockK,
        elementLength: contextArg.stepParams.tokenCount * 2,
        kind: "k",
        stride: 2,
      },
      v: {
        byteOffset: 0,
        descriptor: mockV,
        elementLength: contextArg.stepParams.tokenCount * 2,
        kind: "v",
        stride: 2,
      },
    });
    const mockKvContext = (tokens) => ({
      execution: {
        outputPolicy: executeOutputNone,
        tokens,
      },
      runtime: {
        status: { ok, unsupported },
      },
      session: {
        device: mockOnlyDevice,
      },
      stepParams: {
        contextLength: 4,
        endPosition: tokens.length,
        startPosition: 0,
        tokenCount: tokens.length,
      },
    });
    const firstMockKvResult = mockKvExecutor.executeTokensSync(mockKvContext([0]));
    const firstMockKScratch = mockKvExecutor.mockKValuesScratch;
    const firstMockVScratch = mockKvExecutor.mockVValuesScratch;
    const secondMockKvResult = mockKvExecutor.executeTokensSync(mockKvContext([1, 2]));
    const secondMockKScratch = mockKvExecutor.mockKValuesScratch;
    const secondMockVScratch = mockKvExecutor.mockVValuesScratch;
    const thirdMockKvResult = mockKvExecutor.executeTokensSync(mockKvContext([3]));
    if (
      firstMockKvResult.status !== ok ||
      secondMockKvResult.status !== ok ||
      thirdMockKvResult.status !== ok ||
      firstMockKvResult.fallbackOpCount !== 1 ||
      secondMockKvResult.fallbackOpCount !== 1 ||
      thirdMockKvResult.fallbackOpCount !== 1 ||
      firstMockKScratch.length !== 2 ||
      firstMockVScratch.length !== 2 ||
      secondMockKScratch.length !== 4 ||
      secondMockVScratch.length !== 4 ||
      mockKvExecutor.mockKValuesScratch !== secondMockKScratch ||
      mockKvExecutor.mockVValuesScratch !== secondMockVScratch ||
      firstMockKScratch === secondMockKScratch ||
      firstMockVScratch === secondMockVScratch
    ) {
      throw new Error("LLaMA K/V projection mock fallback did not grow and reuse executor-owned value scratch");
    }
    mockOnlyDevice.destroy();

    const blockBindGroupsBefore = bindGroupCreateCount;
    const firstActivation = fakeDescriptor("fake.block.activation.0", 4 * 4 * Float32Array.BYTES_PER_ELEMENT, "readwrite");
    const secondActivation = fakeDescriptor("fake.block.activation.1", 4 * 4 * Float32Array.BYTES_PER_ELEMENT, "readwrite");
    const firstActivationInput = fakeDescriptor("fake.block.activation-input.0", 4 * 4 * Float32Array.BYTES_PER_ELEMENT, "read");
    const secondActivationInput = fakeDescriptor("fake.block.activation-input.1", 4 * 4 * Float32Array.BYTES_PER_ELEMENT, "read");
    const blockProjection = {
      ...attentionProjection,
      activation: { descriptor: firstActivation },
      activationInput: { descriptor: firstActivationInput },
      downProj: {
        byteLength: 4 * 8 * Float32Array.BYTES_PER_ELEMENT,
        descriptor: fakeDescriptor("fake.block.down_proj", 4 * 8 * Float32Array.BYTES_PER_ELEMENT),
      },
      ffnSize: 8,
      gateProj: {
        byteLength: 8 * 4 * Float32Array.BYTES_PER_ELEMENT,
        descriptor: fakeDescriptor("fake.block.gate_proj", 8 * 4 * Float32Array.BYTES_PER_ELEMENT),
      },
      postNorm: {
        byteLength: 4 * Float32Array.BYTES_PER_ELEMENT,
        descriptor: fakeDescriptor("fake.block.post_norm", 4 * Float32Array.BYTES_PER_ELEMENT),
      },
      modelPack: fakeModelPack("fake.block.model-pack", {
        embedding: 0,
        inputNorm: 256,
        qProj: 512,
        kProj: 768,
        vProj: 1024,
        oProj: 1280,
        postNorm: 1536,
        gateProj: 1792,
        upProj: 2048,
        downProj: 2304,
        norm: 2560,
        lmHead: 2816,
      }),
      upProj: {
        byteLength: 8 * 4 * Float32Array.BYTES_PER_ELEMENT,
        descriptor: fakeDescriptor("fake.block.up_proj", 8 * 4 * Float32Array.BYTES_PER_ELEMENT),
      },
    };
    blockExecutor.submitGpuProjection(attentionContext, blockProjection, firstSlots);
    blockExecutor.submitGpuProjection(attentionContext, blockProjection, firstSlots);
    blockProjection.activation = { descriptor: secondActivation };
    blockExecutor.submitGpuProjection(attentionContext, blockProjection, firstSlots);
    blockProjection.activationInput = { descriptor: secondActivationInput };
    blockExecutor.submitGpuProjection(attentionContext, blockProjection, firstSlots);
    if (bindGroupCreateCount - blockBindGroupsBefore !== 3) {
      throw new Error(`LLaMA block GPU bind-group cache did not account for activation descriptors: ${bindGroupCreateCount - blockBindGroupsBefore}`);
    }

    const blockWindowBindGroupsBefore = bindGroupCreateCount;
    const blockWindowSubmitsBefore = submitCount;
    const blockWindowTokenBuffersBefore = tokenBufferCreateCount;
    const blockWindowUploadsBefore = uploadSources.length;
    const blockWindowContext = (tokens) => ({
      execution: {
        outputPolicy: executeOutputNone,
        tokens,
      },
      session: {
        bindings: { output },
        device: fakeDevice,
      },
      stepParams: {
        contextLength: 4,
        endPosition: tokens.length,
        startPosition: 0,
        tokenCount: tokens.length,
      },
    });
    blockProjection.activation = { descriptor: firstActivation };
    blockProjection.activationInput = { descriptor: firstActivationInput };
    blockExecutor.submitGpuWindowProjection(blockWindowContext([0, 1]), blockProjection, firstSlots);
    blockExecutor.submitGpuWindowProjection(blockWindowContext([2, 3]), blockProjection, firstSlots);
    blockExecutor.submitGpuWindowProjection(blockWindowContext([0, 1, 2]), blockProjection, firstSlots);
    blockExecutor.submitGpuWindowProjection(blockWindowContext([3, 2, 1, 0]), blockProjection, firstSlots);
    const blockWindowUploads = uploadSources.slice(blockWindowUploadsBefore);
    if (bindGroupCreateCount - blockWindowBindGroupsBefore !== 2) {
      throw new Error(`LLaMA block window GPU bind-group cache did not track token buffer growth: ${bindGroupCreateCount - blockWindowBindGroupsBefore}`);
    }
    if (tokenBufferCreateCount - blockWindowTokenBuffersBefore !== 2) {
      throw new Error(`LLaMA block window GPU token buffer was not reused until capacity growth: ${tokenBufferCreateCount - blockWindowTokenBuffersBefore}`);
    }
    if (submitCount - blockWindowSubmitsBefore !== 4) {
      throw new Error(`LLaMA block window GPU submit evidence mismatch: ${submitCount - blockWindowSubmitsBefore}`);
    }
    const blockWindowTokenBufferSizes = tokenBufferSizes.slice(blockWindowTokenBuffersBefore);
    if (
      blockWindowTokenBufferSizes.length !== 2 ||
      blockWindowTokenBufferSizes[0] !== 2 * Uint32Array.BYTES_PER_ELEMENT ||
      blockWindowTokenBufferSizes[1] !== 4 * Uint32Array.BYTES_PER_ELEMENT
    ) {
      throw new Error(`LLaMA block window GPU token buffers did not grow by reusable capacity bucket: ${blockWindowTokenBufferSizes.join(",")}`);
    }
    if (
      blockWindowUploads.length !== 8 ||
      blockWindowUploads[0] !== blockWindowUploads[2] ||
      blockWindowUploads[0] !== blockWindowUploads[4] ||
      blockWindowUploads[0] !== blockWindowUploads[6] ||
      blockWindowUploads[1] !== blockWindowUploads[3] ||
      blockWindowUploads[1] === blockWindowUploads[5] ||
      blockWindowUploads[5] !== blockWindowUploads[7]
    ) {
      throw new Error("LLaMA block window GPU params/tokens uploads did not reuse buffers by capacity");
    }

    const longWindowBindGroupsBefore = bindGroupCreateCount;
    const longWindowSubmitsBefore = submitCount;
    const longWindowTokenBuffersBefore = tokenBufferCreateCount;
    const longWindowUploadsBefore = uploadSources.length;
    const longWindowSlots = {
      k: {
        descriptor: fakeDescriptor("fake.block.long-window.k_cache", 96 * 4 * Float32Array.BYTES_PER_ELEMENT, "readwrite"),
        stride: 4,
      },
      v: {
        descriptor: fakeDescriptor("fake.block.long-window.v_cache", 96 * 4 * Float32Array.BYTES_PER_ELEMENT, "readwrite"),
        stride: 4,
      },
    };
    const longWindowProjection = {
      ...blockProjection,
      slidingWindow: 2,
    };
    const longWindowTokens = Array.from({ length: 66 }, (_unused, index) => index % tinyLlamaVocabSize);
    const originalLongProjectionForSession = blockExecutor.projectionForSession.bind(blockExecutor);
    const originalLongAttentionSlots = blockExecutor.attentionSlots.bind(blockExecutor);
    const originalLongBlockRecord = blockExecutor.record;
    const longWindowRecords = [];
    blockExecutor.projectionForSession = () => longWindowProjection;
    blockExecutor.attentionSlots = () => longWindowSlots;
    blockExecutor.record = (call) => longWindowRecords.push(call);
    const longWindowResult = blockExecutor.executeTokensSync({
      execution: {
        outputLen: 0,
        outputPolicy: executeOutputNone,
        outputPtr: 0,
        tokens: longWindowTokens,
        tokensLen: longWindowTokens.length,
      },
      runtime: {
        status: { ok, unsupported },
      },
      session: {
        bindings: { output },
        device: fakeDevice,
        vocabSize: tinyLlamaVocabSize,
      },
      stepParams: {
        contextLength: 96,
        endPosition: longWindowTokens.length,
        kind: "llama-token-window",
        logitsLength: 0,
        outputKind: "none",
        outputPolicy: executeOutputNone,
        requestedOutputLength: 0,
        startPosition: 0,
        tokenCount: longWindowTokens.length,
      },
    });
    blockExecutor.projectionForSession = originalLongProjectionForSession;
    blockExecutor.attentionSlots = originalLongAttentionSlots;
    blockExecutor.record = originalLongBlockRecord;
    const longWindowUploads = uploadSources.slice(longWindowUploadsBefore);
    if (
      longWindowResult.status !== ok ||
      longWindowResult.backendDispatchCount !== 1 ||
      longWindowResult.commandCount !== 1 ||
      longWindowResult.fallbackOpCount !== 0 ||
      submitCount - longWindowSubmitsBefore !== 1 ||
      bindGroupCreateCount - longWindowBindGroupsBefore !== 1 ||
      tokenBufferCreateCount - longWindowTokenBuffersBefore !== 1 ||
      tokenBufferSizes.at(-1) !== 512 ||
      longWindowUploads.length !== 2 ||
      longWindowUploads[0] !== blockExecutor.paramsBytes ||
      !(longWindowUploads[1] instanceof ArrayBuffer) ||
      longWindowRecords.length !== 1 ||
      longWindowRecords[0].windowDispatchCount !== 1 ||
      longWindowRecords[0].scalarDispatchCount !== 0 ||
      longWindowRecords[0].genericDispatchCount !== 0
    ) {
      throw new Error(`LLaMA long sliding-window block GPU path did not stay batched: ${JSON.stringify({
        bindGroups: bindGroupCreateCount - longWindowBindGroupsBefore,
        commandCount: longWindowResult.commandCount,
        fallbackOpCount: longWindowResult.fallbackOpCount,
        lastTokenBufferSize: tokenBufferSizes.at(-1),
        records: longWindowRecords,
        status: longWindowResult.status,
        submits: submitCount - longWindowSubmitsBefore,
        tokenBuffers: tokenBufferCreateCount - longWindowTokenBuffersBefore,
        uploads: longWindowUploads.length,
      })}`);
    }

    const longDecodeBindGroupsBefore = bindGroupCreateCount;
    const longDecodeSubmitsBefore = submitCount;
    const longDecodeTokenBuffersBefore = tokenBufferCreateCount;
    const longDecodeUploadsBefore = uploadSources.length;
    const longDecodePipelinesBefore = computePipelineLabels.length;
    const originalLongDecodeProjectionForSession = longDecodeBlockExecutor.projectionForSession.bind(longDecodeBlockExecutor);
    const originalLongDecodeAttentionSlots = longDecodeBlockExecutor.attentionSlots.bind(longDecodeBlockExecutor);
    const originalLongDecodeRecord = longDecodeBlockExecutor.record;
    const longDecodeRecords = [];
    longDecodeBlockExecutor.projectionForSession = () => longWindowProjection;
    longDecodeBlockExecutor.attentionSlots = () => longWindowSlots;
    longDecodeBlockExecutor.record = (call) => longDecodeRecords.push(call);
    const longDecodeResult = longDecodeBlockExecutor.executeTokensSync({
      execution: {
        outputLen: tinyLlamaVocabSize,
        outputPolicy: executeOutputLogits,
        outputPtr: 0,
        tokens: [5],
        tokensLen: 1,
      },
      runtime: {
        status: { ok, unsupported },
      },
      session: {
        bindings: { output },
        device: fakeDevice,
        vocabSize: tinyLlamaVocabSize,
      },
      stepParams: {
        contextLength: 96,
        endPosition: 67,
        kind: "llama-token-window",
        logitsLength: tinyLlamaVocabSize,
        outputKind: "bound-logits",
        outputPolicy: executeOutputLogits,
        requestedOutputLength: tinyLlamaVocabSize,
        startPosition: 66,
        tokenCount: 1,
      },
    });
    longDecodeBlockExecutor.projectionForSession = originalLongDecodeProjectionForSession;
    longDecodeBlockExecutor.attentionSlots = originalLongDecodeAttentionSlots;
    longDecodeBlockExecutor.record = originalLongDecodeRecord;
    const longDecodeUploads = uploadSources.slice(longDecodeUploadsBefore);
    const longDecodePipelines = computePipelineLabels.slice(longDecodePipelinesBefore);
    if (
      longDecodeResult.status !== ok ||
      longDecodeResult.backendDispatchCount !== 1 ||
      longDecodeResult.commandCount !== 1 ||
      longDecodeResult.fallbackOpCount !== 0 ||
      submitCount - longDecodeSubmitsBefore !== 1 ||
      bindGroupCreateCount - longDecodeBindGroupsBefore !== 1 ||
      tokenBufferCreateCount - longDecodeTokenBuffersBefore !== 0 ||
      longDecodeUploads.length !== 1 ||
      longDecodeUploads[0] !== longDecodeBlockExecutor.paramsBytes ||
      longDecodePipelines.length !== 1 ||
      longDecodePipelines[0] !== "zgml.wasm.webgpu.llama-block-projection.scalar" ||
      longDecodeRecords.length !== 1 ||
      longDecodeRecords[0].scalarDispatchCount !== 1 ||
      longDecodeRecords[0].windowDispatchCount !== 0 ||
      longDecodeRecords[0].genericDispatchCount !== 0
    ) {
      throw new Error(`LLaMA long sliding-window block decode did not use the scalar GPU path: ${JSON.stringify({
        bindGroups: bindGroupCreateCount - longDecodeBindGroupsBefore,
        commandCount: longDecodeResult.commandCount,
        fallbackOpCount: longDecodeResult.fallbackOpCount,
        pipelines: longDecodePipelines,
        records: longDecodeRecords,
        status: longDecodeResult.status,
        submits: submitCount - longDecodeSubmitsBefore,
        tokenBuffers: tokenBufferCreateCount - longDecodeTokenBuffersBefore,
        uploads: longDecodeUploads.length,
      })}`);
    }

    const prefixSnapshots = [];
    const terminalOptionSnapshots = [];
    const originalProjectionForSession = blockExecutor.projectionForSession.bind(blockExecutor);
    const originalAttentionSlots = blockExecutor.attentionSlots.bind(blockExecutor);
    const originalSubmitWindow = blockExecutor.submitGpuWindowProjection.bind(blockExecutor);
    const originalSubmitProjection = blockExecutor.submitGpuProjection.bind(blockExecutor);
    blockExecutor.projectionForSession = () => blockProjection;
    blockExecutor.attentionSlots = () => firstSlots;
    blockExecutor.submitGpuWindowProjection = (prefixContext, projectionArg, slotsArg) => {
      prefixSnapshots.push({
        execution: prefixContext.execution,
        outputPolicy: prefixContext.execution.outputPolicy,
        ref: prefixContext,
        stepParams: prefixContext.stepParams,
        tokenCount: prefixContext.stepParams.tokenCount,
        tokens: prefixContext.execution.tokens,
        tokensLen: prefixContext.execution.tokensLen,
      });
      return originalSubmitWindow(prefixContext, projectionArg, slotsArg);
    };
    blockExecutor.submitGpuProjection = (contextArg, projectionArg, slotsArg, optionsArg) => {
      terminalOptionSnapshots.push({
        outputPolicy: optionsArg.outputPolicy,
        position: optionsArg.position,
        ref: optionsArg,
        seq: optionsArg.seq,
        token: optionsArg.token,
      });
      return originalSubmitProjection(contextArg, projectionArg, slotsArg, optionsArg);
    };
    const logitsPrefixContext = (tokens) => ({
      execution: {
        outputLen: tinyLlamaVocabSize,
        outputPolicy: executeOutputLogits,
        outputPtr: 0,
        tokens,
        tokensLen: tokens.length,
      },
      runtime: {
        status: { ok, unsupported },
      },
      session: {
        bindings: { output },
        device: fakeDevice,
        vocabSize: tinyLlamaVocabSize,
      },
      stepParams: {
        contextLength: 4,
        endPosition: tokens.length,
        kind: "llama-token-window",
        logitsLength: tinyLlamaVocabSize,
        outputKind: "bound-logits",
        outputPolicy: executeOutputLogits,
        requestedOutputLength: tinyLlamaVocabSize,
        startPosition: 0,
        tokenCount: tokens.length,
      },
    });
    const firstPrefixTokens = [0, 1, 2];
    const secondPrefixTokens = [3, 4, 5];
    const firstPrefixResult = blockExecutor.executeTokensSync(logitsPrefixContext(firstPrefixTokens));
    const secondPrefixResult = blockExecutor.executeTokensSync(logitsPrefixContext(secondPrefixTokens));
    blockExecutor.projectionForSession = originalProjectionForSession;
    blockExecutor.attentionSlots = originalAttentionSlots;
    blockExecutor.submitGpuWindowProjection = originalSubmitWindow;
    blockExecutor.submitGpuProjection = originalSubmitProjection;
    if (
      firstPrefixResult.status !== ok ||
      secondPrefixResult.status !== ok ||
      firstPrefixResult.backendDispatchCount !== 2 ||
      secondPrefixResult.backendDispatchCount !== 2 ||
      firstPrefixResult.commandCount !== 2 ||
      secondPrefixResult.commandCount !== 2
    ) {
      throw new Error("LLaMA block logits-prefix GPU path did not split into prefix window plus terminal logits dispatch");
    }
    if (
      prefixSnapshots.length !== 2 ||
      prefixSnapshots[0].ref !== blockExecutor.prefixContextScratch ||
      prefixSnapshots[1].ref !== blockExecutor.prefixContextScratch ||
      prefixSnapshots[0].execution !== blockExecutor.prefixContextScratch.execution ||
      prefixSnapshots[0].stepParams !== blockExecutor.prefixContextScratch.stepParams ||
      prefixSnapshots[0].tokens !== firstPrefixTokens ||
      prefixSnapshots[1].tokens !== secondPrefixTokens ||
      prefixSnapshots[0].tokensLen !== 2 ||
      prefixSnapshots[1].tokensLen !== 2 ||
      prefixSnapshots[0].tokenCount !== 2 ||
      prefixSnapshots[1].tokenCount !== 2 ||
      prefixSnapshots[0].outputPolicy !== executeOutputNone ||
      prefixSnapshots[1].outputPolicy !== executeOutputNone
    ) {
      throw new Error("LLaMA block logits-prefix GPU path did not reuse a borrowed-prefix context");
    }
    if (
      terminalOptionSnapshots.length !== 2 ||
      terminalOptionSnapshots[0].ref !== blockExecutor.projectionStepOptionsScratch ||
      terminalOptionSnapshots[1].ref !== blockExecutor.projectionStepOptionsScratch ||
      terminalOptionSnapshots[0].outputPolicy !== executeOutputLogits ||
      terminalOptionSnapshots[1].outputPolicy !== executeOutputLogits ||
      terminalOptionSnapshots[0].position !== 2 ||
      terminalOptionSnapshots[1].position !== 2 ||
      terminalOptionSnapshots[0].seq !== 3 ||
      terminalOptionSnapshots[1].seq !== 3 ||
      terminalOptionSnapshots[0].token !== 2 ||
      terminalOptionSnapshots[1].token !== 5
    ) {
      throw new Error("LLaMA block logits-prefix GPU terminal dispatch did not reuse step options scratch");
    }
  } finally {
    executor.destroy();
    kvExecutor.destroy();
    mockKvExecutor.destroy();
    attentionExecutor.destroy();
    blockExecutor.destroy();
    longDecodeBlockExecutor.destroy();
    fakeDevice.destroy();
    if (hadGpuBufferUsage) {
      globalThis.GPUBufferUsage = previousGpuBufferUsage;
    } else {
      delete globalThis.GPUBufferUsage;
    }
  }
}

function expectLlamaPatternParamsBufferReuse() {
  const hadGpuBufferUsage = Object.prototype.hasOwnProperty.call(globalThis, "GPUBufferUsage");
  const previousGpuBufferUsage = globalThis.GPUBufferUsage;
  if (typeof globalThis.GPUBufferUsage !== "object") {
    globalThis.GPUBufferUsage = webgpuBufferUsage;
  }
  let bindGroupCreateCount = 0;
  let paramsBufferCreateCount = 0;
  let submitCount = 0;
  const uploadSources = [];
  const fakeGpuDevice = {
    limits: { minStorageBufferOffsetAlignment: 4 },
    queue: {
      submit() {
        submitCount += 1;
      },
      writeBuffer(_buffer, _offset, source) {
        uploadSources.push(source);
      },
    },
    createBindGroup(desc) {
      bindGroupCreateCount += 1;
      return { desc, kind: "fake-bind-group" };
    },
    createBuffer(desc) {
      if (desc.label === "zgml.wasm.webgpu.llama-token-window-pattern.params") {
        paramsBufferCreateCount += 1;
      }
      return {
        ...desc,
        destroy() {},
        size: desc.size,
      };
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            dispatchWorkgroups() {},
            end() {},
            setBindGroup() {},
            setPipeline() {},
          };
        },
        finish() {
          return { kind: "fake-command-buffer" };
        },
      };
    },
    createComputePipeline() {
      return {
        getBindGroupLayout() {
          return { kind: "fake-bind-group-layout" };
        },
      };
    },
    createShaderModule(desc) {
      return { desc, kind: "fake-shader-module" };
    },
  };
  const fakeDevice = new WasmWebGpuDevice({ device: fakeGpuDevice });
  const fakeDescriptor = (label, byteLength = tinyLlamaVocabSize * Float32Array.BYTES_PER_ELEMENT) => {
    const resource = fakeDevice.createBuffer({
      byteLength,
      label,
      usage: webgpuBufferUsage.STORAGE | webgpuBufferUsage.COPY_SRC | webgpuBufferUsage.COPY_DST,
    });
    return fakeDevice.descriptor(resource, { access: "readwrite", byteLength });
  };
  const executor = new WasmWebGpuLlamaTokenWindowPatternExecutor();
  const kvExecutor = new WasmWebGpuLlamaTokenWindowPatternExecutor();
  const logitsExecutor = new WasmWebGpuLlamaTokenWindowPatternExecutor();
  try {
    const first = fakeDescriptor("fake.pattern.output.0");
    const second = fakeDescriptor("fake.pattern.output.1");
    const makeWrites = (base) => [
      {
        descriptor: first,
        elementLength: tinyLlamaVocabSize,
        elementOffset: 0,
        base,
        label: "fake pattern 0",
      },
      {
        descriptor: second,
        elementLength: tinyLlamaVocabSize,
        elementOffset: 0,
        base: base + 100,
        label: "fake pattern 1",
      },
    ];
    const firstDispatches = executor.submitGpuPatternWrites(fakeDevice, makeWrites(10));
    const secondDispatches = executor.submitGpuPatternWrites(fakeDevice, makeWrites(20));
    if (firstDispatches !== 2 || secondDispatches !== 2 || submitCount !== 2) {
      throw new Error(`LLaMA pattern GPU dispatch evidence mismatch: ${firstDispatches}/${secondDispatches}/${submitCount}`);
    }
    if (paramsBufferCreateCount !== 2) {
      throw new Error(`LLaMA pattern GPU params created ${paramsBufferCreateCount} buffers for two reused write slots`);
    }
    if (bindGroupCreateCount !== 2) {
      throw new Error(`LLaMA pattern GPU bind groups created ${bindGroupCreateCount} groups for two reused write slots`);
    }
    if (uploadSources.length !== 4 || uploadSources.some((source) => source !== uploadSources[0])) {
      throw new Error("LLaMA pattern GPU params upload did not reuse the executor upload buffer");
    }

    const kvParamsBefore = paramsBufferCreateCount;
    const kvBindGroupsBefore = bindGroupCreateCount;
    const kvSubmitsBefore = submitCount;
    const kvUploadsBefore = uploadSources.length;
    const kvContext = {
      session: {
        bindings: {
          kvCache: [{
            k: fakeDescriptor("fake.pattern.k.0"),
            v: fakeDescriptor("fake.pattern.v.0"),
          }],
        },
        device: fakeDevice,
      },
      stepParams: {
        contextLength: 4,
        startPosition: 1,
        tokenCount: 2,
      },
    };
    const firstKvEvidence = kvExecutor.writeKvCache(kvContext);
    const secondKvEvidence = kvExecutor.writeKvCache(kvContext);
    if (
      firstKvEvidence !== kvExecutor.kvEvidenceScratch ||
      secondKvEvidence !== kvExecutor.kvEvidenceScratch ||
      firstKvEvidence.backendDispatchCount !== 2 ||
      firstKvEvidence.commandCount !== 2 ||
      firstKvEvidence.fallbackOpCount !== 0 ||
      firstKvEvidence.kvWriteCount !== 2 ||
      secondKvEvidence.backendDispatchCount !== 2 ||
      secondKvEvidence.kvWriteCount !== 2 ||
      submitCount - kvSubmitsBefore !== 2
    ) {
      throw new Error(`LLaMA pattern GPU K/V direct-write evidence mismatch: ${JSON.stringify({ firstKvEvidence, secondKvEvidence, submitCount })}`);
    }
    if (paramsBufferCreateCount - kvParamsBefore !== 2) {
      throw new Error(`LLaMA pattern GPU K/V params created ${paramsBufferCreateCount - kvParamsBefore} buffers for two reused write slots`);
    }
    if (bindGroupCreateCount - kvBindGroupsBefore !== 2) {
      throw new Error(`LLaMA pattern GPU K/V bind groups created ${bindGroupCreateCount - kvBindGroupsBefore} groups for two reused write slots`);
    }
    const kvUploadSources = uploadSources.slice(kvUploadsBefore);
    if (kvUploadSources.length !== 4 || kvUploadSources.some((source) => source !== kvUploadSources[0])) {
      throw new Error("LLaMA pattern GPU K/V params upload did not reuse the executor upload buffer");
    }

    const logitsParamsBefore = paramsBufferCreateCount;
    const logitsBindGroupsBefore = bindGroupCreateCount;
    const logitsSubmitsBefore = submitCount;
    const logitsUploadsBefore = uploadSources.length;
    const logits = fakeDescriptor("fake.pattern.logits");
    const firstLogitsDispatches = logitsExecutor.submitGpuPatternWrite(
      fakeDevice,
      logits,
      tinyLlamaVocabSize,
      0,
      tinyLlamaVocabSize * Float32Array.BYTES_PER_ELEMENT,
      30,
      "fake pattern logits",
    );
    const secondLogitsDispatches = logitsExecutor.submitGpuPatternWrite(
      fakeDevice,
      logits,
      tinyLlamaVocabSize,
      0,
      tinyLlamaVocabSize * Float32Array.BYTES_PER_ELEMENT,
      40,
      "fake pattern logits",
    );
    if (firstLogitsDispatches !== 1 || secondLogitsDispatches !== 1 || submitCount - logitsSubmitsBefore !== 2) {
      throw new Error(`LLaMA pattern GPU logits direct-write evidence mismatch: ${firstLogitsDispatches}/${secondLogitsDispatches}/${submitCount}`);
    }
    if (paramsBufferCreateCount - logitsParamsBefore !== 1) {
      throw new Error(`LLaMA pattern GPU logits params created ${paramsBufferCreateCount - logitsParamsBefore} buffers for one reused write slot`);
    }
    if (bindGroupCreateCount - logitsBindGroupsBefore !== 1) {
      throw new Error(`LLaMA pattern GPU logits bind groups created ${bindGroupCreateCount - logitsBindGroupsBefore} groups for one reused write slot`);
    }
    const logitsUploadSources = uploadSources.slice(logitsUploadsBefore);
    if (logitsUploadSources.length !== 2 || logitsUploadSources[0] !== logitsUploadSources[1]) {
      throw new Error("LLaMA pattern GPU logits params upload did not reuse the executor upload buffer");
    }
  } finally {
    executor.destroy();
    kvExecutor.destroy();
    logitsExecutor.destroy();
    fakeDevice.destroy();
    if (hadGpuBufferUsage) {
      globalThis.GPUBufferUsage = previousGpuBufferUsage;
    } else {
      delete globalThis.GPUBufferUsage;
    }
  }
}

function expectLlamaTokenWindowExecutorReusesAdapterContext() {
  const device = new WasmWebGpuDevice();
  const descriptor = (label, elementCount) => {
    const byteLength = elementCount * Float32Array.BYTES_PER_ELEMENT;
    const resource = device.createBuffer({
      byteLength,
      label,
      usage: webgpuBufferUsage.STORAGE | webgpuBufferUsage.COPY_SRC | webgpuBufferUsage.COPY_DST,
    });
    return device.descriptor(resource, { access: "readwrite", byteLength });
  };
  const output = descriptor("adapter.output", 4);
  const kCache = descriptor("adapter.k", 8);
  const vCache = descriptor("adapter.v", 8);
  const logitsScratch = new Float32Array(4);
  const kScratch = new Float32Array(2);
  const vScratch = new Float32Array(2);
  const snapshots = [];
  let callIndex = 0;
  const executor = new WasmWebGpuLlamaTokenWindowExecutor((context) => {
    const base = callIndex === 0 ? 10 : 30;
    snapshots.push({
      context,
      kByteOffset: context.kvCacheWindow[0].byteOffset,
      kSlot: context.kvCacheWindow[0],
      vSlot: context.kvCacheWindow[1],
      window: context.kvCacheWindow,
    });
    for (let i = 0; i < logitsScratch.length; i += 1) logitsScratch[i] = base + i;
    for (let i = 0; i < kScratch.length; i += 1) {
      kScratch[i] = base + 100 + i;
      vScratch[i] = base + 200 + i;
    }
    context.writeLogits(logitsScratch);
    context.writeKvCacheWindow((slot) => slot.kind === "k" ? kScratch : vScratch);
    callIndex += 1;
    return { status: ok, outputLength: 4, backendDispatchCount: 0, fallbackOpCount: 3, commandCount: 3 };
  });
  const session = {
    bindings: {
      kvCache: [{ k: kCache, v: vCache }],
      output,
    },
    contextLength: 4,
    device,
    modelHandle: 0,
    vocabSize: 4,
  };
  const runtime = { memoryView: () => { throw new Error("adapter smoke should use bound output"); } };
  const run = (startPosition, token) => executor.executeTokensSync({
    bindings: session.bindings,
    device,
    execution: {
      outputLen: 0,
      outputPolicy: executeOutputLogits,
      outputPtr: 0,
      tokens: [token],
      tokensLen: 1,
    },
    runtime,
    session,
    stepParams: {
      contextLength: 4,
      endPosition: startPosition + 1,
      logitsLength: 4,
      outputKind: "bound-logits",
      outputPolicy: executeOutputLogits,
      requestedOutputLength: 0,
      startPosition,
      tokenCount: 1,
    },
  });
  try {
    const first = run(0, 1);
    const second = run(1, 2);
    if (first.status !== ok || second.status !== ok || snapshots.length !== 2) {
      throw new Error("LLaMA token-window adapter smoke did not execute twice");
    }
    if (
      snapshots[0].context !== executor.contextScratch ||
      snapshots[1].context !== executor.contextScratch ||
      snapshots[0].context !== snapshots[1].context ||
      snapshots[0].window !== executor.kvCacheWindowScratch ||
      snapshots[1].window !== executor.kvCacheWindowScratch ||
      snapshots[0].window !== snapshots[1].window ||
      snapshots[0].kSlot !== snapshots[1].kSlot ||
      snapshots[0].vSlot !== snapshots[1].vSlot ||
      snapshots[0].kByteOffset !== 0 ||
      snapshots[1].kByteOffset !== 2 * Float32Array.BYTES_PER_ELEMENT
    ) {
      throw new Error("LLaMA token-window adapter did not reuse augmented context and K/V slot scratch");
    }
    expectClose(Array.from(device.readFloat32Sync(output, 4)), [30, 31, 32, 33]);
    expectClose(Array.from(device.readFloat32Sync(kCache, 4)), [110, 111, 130, 131]);
    expectClose(Array.from(device.readFloat32Sync(vCache, 4)), [210, 211, 230, 231]);
  } finally {
    device.destroy();
  }
}

async function expectLlamaDeviceTopKOneUsesArgmaxSelector() {
  const hadGpuBufferUsage = Object.prototype.hasOwnProperty.call(globalThis, "GPUBufferUsage");
  const previousGpuBufferUsage = globalThis.GPUBufferUsage;
  if (typeof globalThis.GPUBufferUsage !== "object") {
    globalThis.GPUBufferUsage = webgpuBufferUsage;
  }
  let argmaxSelectCount = 0;
  let topKSelectCount = 0;
  const fakeDevice = new WasmWebGpuDevice({
    device: {
      createBuffer(desc) {
        return {
          ...desc,
          destroy() {},
          size: desc.size,
        };
      },
    },
  });
  const runtime = new WasmWebGpuSessionRuntime({
    invalidArgument,
    ok,
    shapeMismatch,
    unsupported,
  });
	  const session = new WasmWebGpuLlamaResourceSession({
    contextLength: 4,
    device: fakeDevice,
    deviceArgmaxSelector: {
      async select() {
        argmaxSelectCount += 1;
        return Object.freeze({
          backendDispatchCount: 1,
          logit: 7.5,
          resultReadCount: 1,
          syncCount: 1,
          token: 3,
        });
      },
    },
    deviceTopKSelector: {
      async select() {
        topKSelectCount += 1;
        throw new Error("topK=1 device sampling should use argmax selector");
      },
    },
    nativeSession: false,
    runtime,
    sessionHandle: 0x71000001,
    vocabSize: 8,
  });
  try {
    const selected = await session.sampleDevice({
      seed: 123,
      temperature: 0.25,
      topK: 1,
    });
    if (
      selected.status !== ok ||
      selected.token !== 3 ||
      selected.logit !== 7.5 ||
      selected.backendDispatchCount !== 1 ||
      selected.resultReadCount !== 1 ||
      selected.syncCount !== 1 ||
      argmaxSelectCount !== 1 ||
      topKSelectCount !== 0
    ) {
      throw new Error(`LLaMA device topK=1 sampling did not route through argmax selector: ${JSON.stringify({ argmaxSelectCount, selected, topKSelectCount })}`);
    }
    const profile = session.runtimeProfile();
    if (
      profile.selectionCallCount !== 1 ||
      profile.selectionBackendDispatchCount !== 1 ||
      profile.selectionResultReadCount !== 1 ||
      profile.selectionSyncCount !== 1 ||
      profile.selectionFallbackOpCount !== 0
    ) {
      throw new Error(`unexpected LLaMA device topK=1 sampling profile: ${JSON.stringify(profile)}`);
    }
  } finally {
    session.dispose();
    fakeDevice.destroy();
    if (hadGpuBufferUsage) {
      globalThis.GPUBufferUsage = previousGpuBufferUsage;
    } else {
      delete globalThis.GPUBufferUsage;
    }
  }
}

async function expectLlamaDeviceSelectionRejectsMalformedSelectorEvidence() {
  const hadGpuBufferUsage = Object.prototype.hasOwnProperty.call(globalThis, "GPUBufferUsage");
  const previousGpuBufferUsage = globalThis.GPUBufferUsage;
  if (typeof globalThis.GPUBufferUsage !== "object") {
    globalThis.GPUBufferUsage = webgpuBufferUsage;
  }
  const fakeDevice = new WasmWebGpuDevice({
    device: {
      createBuffer(desc) {
        return {
          ...desc,
          destroy() {},
          size: desc.size,
        };
      },
    },
  });
  const runtime = new WasmWebGpuSessionRuntime({
    invalidArgument,
    ok,
    shapeMismatch,
    unsupported,
  });
  const expectRejected = async (label, options) => {
    const session = new WasmWebGpuLlamaResourceSession({
      contextLength: 4,
      device: options.device ?? fakeDevice,
      deviceArgmaxSelector: options.deviceArgmaxSelector ?? {
        async select() {
          return Object.freeze({
            backendDispatchCount: 1,
            logit: 7.5,
            resultReadCount: 1,
            syncCount: 1,
            token: 3,
          });
        },
      },
      deviceTopKSelector: options.deviceTopKSelector ?? {
        async select() {
          return Object.freeze({
            backendDispatchCount: 1,
            logit: 6.5,
            resultReadCount: 1,
            syncCount: 1,
            token: 2,
          });
        },
      },
      nativeSession: false,
      runtime,
      sessionHandle: options.sessionHandle,
      vocabSize: 8,
    });
    try {
      const selected = options.sample === true
        ? await session.sampleDevice({ seed: 123, temperature: 1, topK: 2 })
        : await session.argmaxDevice();
      if (
        selected.status !== invalidArgument ||
        selected.position !== 0 ||
        selected.backendDispatchCount !== 0 ||
        selected.resultReadCount !== 0 ||
        selected.syncCount !== 0 ||
        selected.token !== 0 ||
        selected.logit !== 0
      ) {
        throw new Error(`LLaMA device selection accepted malformed ${label} evidence: ${JSON.stringify(selected)}`);
      }
      const profile = session.runtimeProfile();
      if (
        profile.selectionCallCount !== 1 ||
        profile.selectionBackendDispatchCount !== 0 ||
        profile.selectionResultReadCount !== 0 ||
        profile.selectionSyncCount !== 0 ||
        profile.selectionFallbackOpCount !== 0
      ) {
        throw new Error(`LLaMA device selection recorded malformed ${label} evidence in profile: ${JSON.stringify(profile)}`);
      }
    } finally {
      session.dispose();
    }
  };
  try {
    await expectRejected("out-of-vocab token", {
      sessionHandle: 0x71000011,
      deviceArgmaxSelector: {
        async select() {
          return Object.freeze({
            backendDispatchCount: 1,
            logit: 7.5,
            resultReadCount: 1,
            syncCount: 1,
            token: 8,
          });
        },
      },
    });
    await expectRejected("negative read counter", {
      sample: true,
      sessionHandle: 0x71000012,
      deviceTopKSelector: {
        async select() {
          return Object.freeze({
            backendDispatchCount: 1,
            logit: 6.5,
            resultReadCount: -1,
            syncCount: 1,
            token: 2,
          });
        },
      },
    });
    await expectRejected("zero-work success", {
      sessionHandle: 0x71000013,
      deviceArgmaxSelector: {
        async select() {
          return Object.freeze({
            backendDispatchCount: 0,
            logit: 7.5,
            resultReadCount: 0,
            syncCount: 0,
            token: 3,
          });
        },
      },
    });
    await expectRejected("unowned GPU dispatch", {
      device: {
        canCreateGpuBuffers() {
          return true;
        },
      },
      sessionHandle: 0x71000014,
    });
  } finally {
    fakeDevice.destroy();
    if (hadGpuBufferUsage) {
      globalThis.GPUBufferUsage = previousGpuBufferUsage;
    } else {
      delete globalThis.GPUBufferUsage;
    }
  }
}

async function expectLlamaFullLogitsSamplerReusesScratch() {
  const runtime = new WasmWebGpuSessionRuntime({
    invalidArgument,
    ok,
    shapeMismatch,
    unsupported,
  });
  const outputResource = { label: "llama.output" };
  const logits = [1, 9, 8, 7];
  const session = new WasmWebGpuLlamaResourceSession({
    bindings: { output: outputResource },
    contextLength: 8,
    device: {
      canCreateGpuBuffers() {
        return false;
      },
      async readFloat32Into(resource, target, length) {
        if (resource !== outputResource || length !== logits.length) {
          throw new Error(`unexpected LLaMA sampler readback request: ${JSON.stringify({ length, resource })}`);
        }
        target.set(logits);
        return target;
      },
    },
    nativeSession: false,
    runtime,
    sessionHandle: 0x71000009,
    tokenExecutor() {
      return {
        backendDispatchCount: 0,
        backendOpCount: 0,
        commandCount: 1,
        commandOpCount: 1,
        fallbackOpCount: 1,
        outputLength: logits.length,
        status: ok,
      };
    },
    vocabSize: 4,
  });
  try {
    const first = await session.sample({ logits, seed: 10752, temperature: 1, topK: 3 });
    const firstTokens = session.fullLogitsSampleScratch.candidateTokens;
    const firstLogits = session.fullLogitsSampleScratch.candidateLogits;
    if (first.status !== ok || firstTokens?.length !== 3 || firstLogits?.length !== 3) {
      throw new Error(`LLaMA full-logits sampler did not allocate initial scratch: ${JSON.stringify({ first, tokensLength: firstTokens?.length, logitsLength: firstLogits?.length })}`);
    }

    const smaller = await session.sampleDevice({ logits, seed: 11776, temperature: 1, topK: 2 });
    if (
      smaller.status !== ok ||
      session.fullLogitsSampleScratch.candidateTokens !== firstTokens ||
      session.fullLogitsSampleScratch.candidateLogits !== firstLogits
    ) {
      throw new Error("LLaMA full-logits sampler did not reuse scratch for smaller topK sample-device fallback");
    }

    const executed = await session.executeTokensSample([0], { seed: 11776, temperature: 1, topK: 2 });
    if (
      executed.status !== ok ||
      session.fullLogitsSampleScratch.candidateTokens !== firstTokens ||
      session.fullLogitsSampleScratch.candidateLogits !== firstLogits
    ) {
      throw new Error("LLaMA full-logits sampler did not reuse scratch for execute-and-sample");
    }

    const grown = await session.sample({ logits, seed: 14336, temperature: 1, topK: 4 });
    if (
      grown.status !== ok ||
      session.fullLogitsSampleScratch.candidateTokens === firstTokens ||
      session.fullLogitsSampleScratch.candidateLogits === firstLogits ||
      session.fullLogitsSampleScratch.candidateTokens?.length !== 4 ||
      session.fullLogitsSampleScratch.candidateLogits?.length !== 4
    ) {
      throw new Error("LLaMA full-logits sampler did not grow scratch exactly when topK capacity grew");
    }
  } finally {
    session.dispose();
  }
  if (session.fullLogitsSampleScratch.candidateTokens !== null || session.fullLogitsSampleScratch.candidateLogits !== null) {
    throw new Error("LLaMA full-logits sampler scratch was not cleared on Session dispose");
  }
}

async function expectLlamaFullLogitsReadbackReusesScratch() {
  const runtime = new WasmWebGpuSessionRuntime({
    invalidArgument,
    ok,
    shapeMismatch,
    unsupported,
  });
  const outputResource = { label: "llama.output" };
  const readTargets = [];
  const device = {
    canCreateGpuBuffers() {
      return false;
    },
    async readFloat32Into(resource, target, length) {
      if (resource !== outputResource || length !== 4) {
        throw new Error(`unexpected LLaMA readback request: ${JSON.stringify({ length, resource })}`);
      }
      readTargets.push(target);
      target.set([1, 9, 8, 7]);
      return target;
    },
  };
  const session = new WasmWebGpuLlamaResourceSession({
    bindings: { output: outputResource },
    contextLength: 8,
    device,
    nativeSession: false,
    runtime,
    sessionHandle: 0x7100000a,
    tokenExecutor() {
      return {
        backendDispatchCount: 0,
        backendOpCount: 0,
        commandCount: 1,
        commandOpCount: 1,
        fallbackOpCount: 1,
        outputLength: 4,
        status: ok,
      };
    },
    vocabSize: 4,
  });
  try {
    const sampled = await session.sample({ seed: 10752, temperature: 1, topK: 3 });
    const scratch = session.fullLogitsOutputScratch;
    if (sampled.status !== ok || !(scratch instanceof Float32Array) || readTargets[0] !== scratch) {
      throw new Error("LLaMA full-logits sample did not read into Session-owned output scratch");
    }

    const argmax = await session.argmax();
    if (argmax.status !== ok || argmax.token !== 1 || readTargets[1] !== scratch || session.fullLogitsOutputScratch !== scratch) {
      throw new Error("LLaMA full-logits argmax did not reuse Session-owned output scratch");
    }

    const executed = await session.executeTokensSample([0], { seed: 11776, temperature: 1, topK: 2 });
    if (executed.status !== ok || readTargets[2] !== scratch || session.fullLogitsOutputScratch !== scratch) {
      throw new Error("LLaMA execute-and-sample did not reuse Session-owned output scratch");
    }

    const staleLogits = [100, 0, 0, 0];
    const deviceArgmax = await session.executeTokensArgmaxDevice([0], { gpu: false, logits: staleLogits });
    if (deviceArgmax.status !== ok || deviceArgmax.token !== 1 || readTargets[3] !== scratch || session.fullLogitsOutputScratch !== scratch) {
      throw new Error("LLaMA execute-and-argmax-device used stale caller logits instead of Session output scratch");
    }

    const deviceSampled = await session.executeTokensSampleDevice([0], { gpu: false, logits: staleLogits, seed: 11776, temperature: 1, topK: 1 });
    if (deviceSampled.status !== ok || deviceSampled.token !== 1 || readTargets[4] !== scratch || session.fullLogitsOutputScratch !== scratch) {
      throw new Error("LLaMA execute-and-sample-device used stale caller logits instead of Session output scratch");
    }

    const included = await session.sample({ includeLogits: true, seed: 14336, temperature: 1, topK: 2 });
    if (
      included.status !== ok ||
      !(included.logits instanceof Float32Array) ||
      included.logits === scratch ||
      readTargets[5] !== included.logits
    ) {
      throw new Error("LLaMA includeLogits path did not preserve fresh public logits ownership");
    }

    const callerTarget = new Float32Array(4);
    const into = await session.outputInto(callerTarget);
    if (into !== callerTarget || readTargets[6] !== callerTarget) {
      throw new Error("LLaMA outputInto did not fill caller-owned logits storage");
    }

    const profile = session.runtimeProfile();
    if (profile.outputReadCount !== 7 || profile.syncCount !== 7) {
      throw new Error(`unexpected LLaMA readback scratch profile: ${JSON.stringify(profile)}`);
    }
  } finally {
    session.dispose();
  }
  if (session.fullLogitsOutputScratch !== null) {
    throw new Error("LLaMA full-logits output scratch was not cleared on Session dispose");
  }
}

async function expectLlamaGenerationUsesPrivateSelectionScratch() {
  const runtime = new WasmWebGpuSessionRuntime({
    invalidArgument,
    ok,
    shapeMismatch,
    unsupported,
  });
  let nextHandle = 0x71000010;
  async function expectPrivateScratch(methodName, privateExecuteName, options = {}) {
    const outputResource = { label: `${methodName}.output` };
    const calls = [];
    const decodedOptions = [];
    const selectionTargets = [];
    const syncOptions = [];
    const windows = [];
    const session = new WasmWebGpuLlamaResourceSession({
      bindings: { output: outputResource },
      contextLength: 16,
      device: {
        canCreateGpuBuffers() {
          return false;
        },
        async readFloat32Into(resource, target, length) {
          if (resource !== outputResource || length !== tinyLlamaVocabSize) {
            throw new Error(`unexpected LLaMA ${methodName} private generation readback: ${JSON.stringify({ length, resource })}`);
          }
          target.fill(-100);
          target[1] = 100;
          return target;
        },
      },
      nativeSession: false,
      runtime,
      sessionHandle: nextHandle,
      tokenExecutor(context) {
        calls.push({
          context,
          execution: context.execution,
          stepParams: context.stepParams,
          tokens: context.execution.tokens,
          tokensLen: context.execution.tokensLen,
          values: Array.from({ length: context.execution.tokensLen }, (_unused, index) => context.execution.tokens[index]),
        });
        return {
          backendDispatchCount: 0,
          backendOpCount: 0,
          commandCount: 1,
          commandOpCount: 1,
          fallbackOpCount: 1,
          outputLength: tinyLlamaVocabSize,
          status: ok,
        };
      },
      vocabSize: tinyLlamaVocabSize,
    });
    nextHandle += 1;
    for (const publicName of [
      "executeTokensArgmax",
      "executeTokensArgmaxDevice",
      "executeTokensSample",
      "executeTokensSampleDevice",
    ]) {
      session[publicName] = async () => {
        throw new Error(`LLaMA generation called public ${publicName}`);
      };
    }
    const originalPrivateExecute = session[privateExecuteName].bind(session);
    const originalSyncFromList = session.executeTokensSyncFromListInto.bind(session);
    const originalDecoded = session.executeDecodedTokensSyncInto.bind(session);
    session.executeTokensSyncFromListInto = (target, tokens, executeOptions, forcedOutputPolicy) => {
      syncOptions.push({
        executeOptions,
        forcedOutputPolicy,
        tokens,
      });
      return originalSyncFromList(target, tokens, executeOptions, forcedOutputPolicy);
    };
    session.executeDecodedTokensSyncInto = (target, execution, resultPtr, executeRuntime, executeOptions) => {
      decodedOptions.push(executeOptions);
      return originalDecoded(target, execution, resultPtr, executeRuntime, executeOptions);
    };
    session[privateExecuteName] = async (target, tokens, callOptions, internal) => {
      selectionTargets.push(target);
      windows.push({
        callOptions,
        includeLogits: callOptions.includeLogits,
        seed: callOptions.seed,
        tokens,
        values: Array.from(tokens),
      });
      return originalPrivateExecute(target, tokens, callOptions, internal);
    };
    try {
      const output = new Uint32Array(3);
      const prompt = Uint32Array.of(0, 1);
      const callerOptions = {
        ...options,
        includeLogits: true,
      };
      const result = await session[methodName](prompt, output, callerOptions);
      if (
        result.status !== ok ||
        result.tokensGenerated !== 3 ||
        result.lastToken !== 1 ||
        result.lastLogit !== 100 ||
        result.position !== 4 ||
        output[0] !== 1 ||
        output[1] !== 1 ||
        output[2] !== 1 ||
        !(result.logits instanceof Float32Array)
      ) {
        throw new Error(`LLaMA ${methodName} private-scratch generation returned wrong result: ${JSON.stringify({ output: Array.from(output), result })}`);
      }
      if (
        windows.length !== 3 ||
        calls.length !== 3 ||
        syncOptions.length !== 3 ||
        decodedOptions.length !== 3 ||
        windows[0].callOptions !== session.generationCallOptionsScratch ||
        windows[0].callOptions !== windows[1].callOptions ||
        windows[1].callOptions !== windows[2].callOptions ||
        syncOptions[0].executeOptions !== windows[0].callOptions ||
        syncOptions[1].executeOptions !== windows[1].callOptions ||
        syncOptions[2].executeOptions !== windows[2].callOptions ||
        decodedOptions[0] !== windows[0].callOptions ||
        decodedOptions[1] !== windows[1].callOptions ||
        decodedOptions[2] !== windows[2].callOptions ||
        syncOptions[0].forcedOutputPolicy !== 1 ||
        syncOptions[1].forcedOutputPolicy !== 1 ||
        syncOptions[2].forcedOutputPolicy !== 1 ||
        windows[0].tokens !== prompt ||
        windows[0].tokens === windows[1].tokens ||
        windows[1].tokens !== session.generationContinuationScratch ||
        windows[1].tokens !== windows[2].tokens ||
        windows[0].values.length !== 2 ||
        windows[0].values[0] !== 0 ||
        windows[0].values[1] !== 1 ||
        windows[1].values.length !== 1 ||
        windows[1].values[0] !== 1 ||
        windows[2].values.length !== 1 ||
        windows[2].values[0] !== 1
      ) {
        throw new Error(`LLaMA ${methodName} did not use the expected private generation token/options windows: ${JSON.stringify(windows.map((entry) => entry.values))}`);
      }
      if (
        selectionTargets.length !== 3 ||
        selectionTargets[0] !== session.generationSelectionScratch ||
        selectionTargets[1] !== session.generationSelectionScratch ||
        selectionTargets[2] !== session.generationSelectionScratch
      ) {
        throw new Error(`LLaMA ${methodName} did not reuse the private generation selection scratch`);
      }
      if (windows[0].includeLogits !== false || windows[1].includeLogits !== false || windows[2].includeLogits !== true) {
        throw new Error(`LLaMA ${methodName} did not preserve final-only includeLogits semantics`);
      }
      if (options.seed !== undefined && (windows[0].seed !== options.seed || windows[1].seed !== ((options.seed + 1) >>> 0) || windows[2].seed !== ((options.seed + 2) >>> 0))) {
        throw new Error(`LLaMA ${methodName} did not preserve per-token sample seed progression`);
      }
      if (callerOptions.includeLogits !== true || (options.seed !== undefined && callerOptions.seed !== options.seed)) {
        throw new Error(`LLaMA ${methodName} mutated caller generation options`);
      }
    } finally {
      session.dispose();
    }
  }

  await expectPrivateScratch("generateTokensArgmaxInto", "executeTokensArgmaxInto");
  await expectPrivateScratch("generateTokensArgmaxDeviceInto", "executeTokensArgmaxDeviceInto", { gpu: false });
  await expectPrivateScratch("generateTokensSampleInto", "executeTokensSampleInto", {
    seed: 17,
    temperature: 0.75,
    topK: 3,
  });
  await expectPrivateScratch("generateTokensSampleDeviceInto", "executeTokensSampleDeviceInto", {
    gpu: false,
    seed: 23,
    temperature: 0.5,
    topK: 4,
  });
}

async function expectLlamaGenerationBorrowsHotTokenWindow() {
  const runtime = new WasmWebGpuSessionRuntime({
    invalidArgument,
    ok,
    shapeMismatch,
    unsupported,
  });
  let nextHandle = 0x71000020;
  async function expectBorrow(methodName, options = {}) {
    const outputResource = { label: `${methodName}.output` };
    const calls = [];
    const session = new WasmWebGpuLlamaResourceSession({
      bindings: { output: outputResource },
      contextLength: 8,
      device: {
        canCreateGpuBuffers() {
          return false;
        },
        async readFloat32Into(resource, target, length) {
          if (resource !== outputResource || length !== tinyLlamaVocabSize) {
            throw new Error(`unexpected LLaMA ${methodName} generation readback: ${JSON.stringify({ length, resource })}`);
          }
          target.fill(-100);
          target[1] = 100;
          return target;
        },
      },
      nativeSession: false,
      runtime,
      sessionHandle: nextHandle,
      tokenExecutor(context) {
        calls.push({
          context,
          execution: context.execution,
          stepParams: context.stepParams,
          tokens: context.execution.tokens,
          tokensLen: context.execution.tokensLen,
          values: Array.from({ length: context.execution.tokensLen }, (_unused, index) => context.execution.tokens[index]),
        });
        return {
          backendDispatchCount: 0,
          backendOpCount: 0,
          commandCount: 1,
          commandOpCount: 1,
          fallbackOpCount: 1,
          outputLength: tinyLlamaVocabSize,
          status: ok,
        };
      },
      vocabSize: tinyLlamaVocabSize,
    });
    nextHandle += 1;
    try {
      const out = new Uint32Array(3);
      const prompt = [0, 1, tinyLlamaVocabSize + 1];
      prompt[Symbol.iterator] = () => {
        throw new Error("LLaMA generation should index borrowed prompt tokensLen window instead of iterating");
      };
      const callOptions = { ...options, includeLogits: true, promptTokensLen: 2 };
      const result = await session[methodName](prompt, out, callOptions);
      if (
        result.status !== ok ||
        result.tokensGenerated !== 3 ||
        result.lastToken !== 1 ||
        result.position !== 4 ||
        out[0] !== 1 ||
        out[1] !== 1 ||
        out[2] !== 1 ||
        !(result.logits instanceof Float32Array)
      ) {
        throw new Error(`LLaMA ${methodName} borrowed-token generation returned unexpected result: ${JSON.stringify({ out: Array.from(out), result })}`);
      }
      if (
        calls.length !== 3 ||
        calls[0].context !== calls[1].context ||
        calls[1].context !== calls[2].context ||
        calls[0].execution !== calls[1].execution ||
        calls[1].execution !== calls[2].execution ||
        calls[0].stepParams !== calls[1].stepParams ||
        calls[1].stepParams !== calls[2].stepParams ||
        calls[0].tokens !== prompt ||
        calls[1].tokens !== calls[2].tokens ||
        calls[0].tokens === calls[1].tokens ||
        calls[0].tokensLen !== 2 ||
        calls[1].tokensLen !== 1 ||
        calls[2].tokensLen !== 1 ||
        calls[0].values.length !== 2 ||
        calls[0].values[0] !== 0 ||
        calls[0].values[1] !== 1 ||
        calls[1].values.length !== 1 ||
        calls[1].values[0] !== 1 ||
        calls[2].values.length !== 1 ||
        calls[2].values[0] !== 1
      ) {
        throw new Error(`LLaMA ${methodName} did not borrow/reuse hot token execution state: ${JSON.stringify(calls.map((call) => call.values))}`);
      }
      if (
        calls[0].context !== session.tokenExecutorContextScratch ||
        session.lastTokenExecution !== session.tokenExecutionScratch ||
        session.lastStepParams !== session.stepParamsScratch ||
        session.tokenExecutionHistory.length !== 0 ||
        session.stepParamsHistory.length !== 0
      ) {
        throw new Error(`LLaMA ${methodName} did not keep diagnostic history out of the hot generation lane`);
      }
      if (
        callOptions.includeLogits !== true ||
        callOptions.promptTokensLen !== 2 ||
        (options.seed !== undefined && callOptions.seed !== options.seed)
      ) {
        throw new Error(`LLaMA ${methodName} mutated borrowed-token caller options`);
      }
      const preflightProfile = session.runtimeProfile();
      const preflightCalls = calls.length;
      const preflightPosition = session.position;
      session.maxTokenWindow = 1;
      const rejectedOut = new Uint32Array([777, 777]);
      const rejectedOptions = { ...options, includeLogits: true, promptTokensLen: 2 };
      const rejected = await session[methodName]([0, 1], rejectedOut, rejectedOptions);
      if (
        rejected.status !== shapeMismatch ||
        rejected.tokensGenerated !== 0 ||
        rejected.lastToken !== 0 ||
        rejected.lastLogit !== 0 ||
        rejected.position !== preflightPosition ||
        rejected.logits !== undefined ||
        calls.length !== preflightCalls ||
        session.position !== preflightPosition ||
        rejectedOut[0] !== 777 ||
        rejectedOut[1] !== 777
      ) {
        throw new Error(`LLaMA ${methodName} did not preflight-reject oversized generation prompt windows: ${JSON.stringify({ rejected, out: Array.from(rejectedOut), calls: calls.length, preflightCalls, position: session.position, preflightPosition })}`);
      }
      const rejectedProfile = session.runtimeProfile();
      if (JSON.stringify(rejectedProfile) !== JSON.stringify(preflightProfile)) {
        throw new Error(`LLaMA ${methodName} oversized generation prompt preflight mutated profile: ${JSON.stringify({ before: preflightProfile, after: rejectedProfile })}`);
      }
    } finally {
      session.dispose();
    }
  }

  await expectBorrow("generateTokensArgmaxInto");
  await expectBorrow("generateTokensArgmaxDeviceInto", { gpu: false });
  await expectBorrow("generateTokensSampleInto", { seed: 31, temperature: 1, topK: 1 });
  await expectBorrow("generateTokensSampleDeviceInto", { gpu: false, seed: 37, temperature: 1, topK: 1 });
}

async function expectLlamaScalarStepsBorrowTokenScratch() {
  const runtime = new WasmWebGpuSessionRuntime({
    invalidArgument,
    ok,
    shapeMismatch,
    unsupported,
  });
  const outputResource = { label: "scalar-step.output" };
  const calls = [];
  const session = new WasmWebGpuLlamaResourceSession({
    bindings: { output: outputResource },
    contextLength: 16,
    device: {
      canCreateGpuBuffers() {
        return false;
      },
      async readFloat32Into(resource, target, length) {
        if (resource !== outputResource || length !== tinyLlamaVocabSize) {
          throw new Error(`unexpected scalar-step readback: ${JSON.stringify({ length, resource })}`);
        }
        for (let i = 0; i < target.length; i += 1) target[i] = i;
        return target;
      },
    },
    nativeSession: false,
    runtime,
    sessionHandle: 0x71000030,
    tokenExecutor(context) {
      calls.push({
        context,
        execution: context.execution,
        outputPolicy: context.execution.outputPolicy,
        stepParams: context.stepParams,
        tokens: context.execution.tokens,
        tokensLen: context.execution.tokensLen,
        value: context.execution.tokens[0],
      });
      return {
        backendDispatchCount: 0,
        backendOpCount: 0,
        commandCount: 1,
        commandOpCount: 1,
        fallbackOpCount: 1,
        outputLength: context.execution.outputPolicy === executeOutputLogits ? tinyLlamaVocabSize : 0,
        status: ok,
      };
    },
    vocabSize: tinyLlamaVocabSize,
  });
  try {
    const stepOptions = { readLogits: false };
    const argmaxOptions = { includeLogits: false };
    const deviceArgmaxOptions = { gpu: false };
    const sampleOptions = { seed: 11, temperature: 1, topK: 1 };
    const deviceSampleOptions = { gpu: false, seed: 13, temperature: 1, topK: 1 };
    const step = await session.step(3, stepOptions);
    const argmax = await session.stepArgmax(2, argmaxOptions);
    const deviceArgmax = await session.stepArgmaxDevice(4, deviceArgmaxOptions);
    const sample = await session.stepSample(1, sampleOptions);
    const deviceSample = await session.stepSampleDevice(5, deviceSampleOptions);
    if (
      step.status !== ok ||
      argmax.status !== ok ||
      deviceArgmax.status !== ok ||
      sample.status !== ok ||
      deviceSample.status !== ok ||
      argmax.token !== tinyLlamaVocabSize - 1 ||
      deviceArgmax.token !== tinyLlamaVocabSize - 1 ||
      sample.token !== tinyLlamaVocabSize - 1 ||
      deviceSample.token !== tinyLlamaVocabSize - 1 ||
      session.position !== 5
    ) {
      throw new Error(`LLaMA scalar step helpers returned unexpected results: ${JSON.stringify({ argmax, deviceArgmax, deviceSample, position: session.position, sample, step })}`);
    }
    if (
      calls.length !== 5 ||
      calls.some((call) => call.tokens !== session.scalarTokenScratch) ||
      calls.some((call) => call.tokensLen !== 1) ||
      calls.map((call) => call.value).join(",") !== "3,2,4,1,5" ||
      calls.some((call) => call.outputPolicy !== executeOutputLogits) ||
      calls.some((call) => call.context !== session.tokenExecutorContextScratch) ||
      calls.some((call) => call.execution !== session.tokenExecutionScratch) ||
      calls.some((call) => call.stepParams !== session.stepParamsScratch) ||
      session.tokenExecutorOutcomeScratch.status !== ok ||
      session.tokenExecutorOutcomeScratch.outputLength !== tinyLlamaVocabSize ||
      session.tokenExecutorOutcomeScratch.backendDispatchCount !== 0 ||
      session.tokenExecutorOutcomeScratch.fallbackOpCount !== 1 ||
      session.lastTokenExecution !== session.tokenExecutionScratch ||
      session.lastStepParams !== session.stepParamsScratch ||
      session.tokenExecutionHistory.length !== 0 ||
      session.stepParamsHistory.length !== 0
    ) {
      throw new Error(`LLaMA scalar step helpers did not borrow scalar token scratch: ${JSON.stringify(calls.map((call) => ({ tokensLen: call.tokensLen, value: call.value })))}`);
    }
    if (
      Object.prototype.hasOwnProperty.call(stepOptions, "borrowTokens") ||
      Object.prototype.hasOwnProperty.call(argmaxOptions, "borrowTokens") ||
      Object.prototype.hasOwnProperty.call(deviceArgmaxOptions, "borrowTokens") ||
      Object.prototype.hasOwnProperty.call(sampleOptions, "borrowTokens") ||
      Object.prototype.hasOwnProperty.call(deviceSampleOptions, "borrowTokens")
    ) {
      throw new Error("LLaMA scalar step helpers mutated caller options with borrowTokens");
    }
  } finally {
    session.dispose();
  }
}

async function expectLlamaAdvanceTokensUsesNoOutputInternal() {
  const runtime = new WasmWebGpuSessionRuntime({
    invalidArgument,
    ok,
    shapeMismatch,
    unsupported,
  });
  const outputResource = { label: "advance-no-output.output" };
  const calls = [];
  let readCount = 0;
  const session = new WasmWebGpuLlamaResourceSession({
    bindings: { output: outputResource },
    contextLength: 16,
    device: {
      canCreateGpuBuffers() {
        return false;
      },
      async readFloat32Into() {
        readCount += 1;
        throw new Error("advanceTokens should not read logits");
      },
    },
    nativeSession: false,
    runtime,
    sessionHandle: 0x71000031,
    tokenExecutor(context) {
      calls.push({
        context,
        execution: context.execution,
        outputPolicy: context.execution.outputPolicy,
        stepParams: context.stepParams,
        tokens: context.execution.tokens,
      });
      return {
        backendDispatchCount: 0,
        backendOpCount: 0,
        commandCount: 1,
        commandOpCount: 1,
        fallbackOpCount: 1,
        outputLength: context.execution.outputPolicy === executeOutputLogits ? tinyLlamaVocabSize : 0,
        status: ok,
      };
    },
    vocabSize: tinyLlamaVocabSize,
  });
  const originalSyncFromList = session.executeTokensSyncFromListInto.bind(session);
  session.executeTokensSyncFromListInto = (target, tokens, executeOptions, forcedOutputPolicy, internal) => {
    calls.push({
      kind: "sync",
      executeOptions,
      forcedOutputPolicy,
      internal,
      tokens,
    });
    return originalSyncFromList(target, tokens, executeOptions, forcedOutputPolicy, internal);
  };
  try {
    const callerOptions = { outputPolicy: executeOutputLogits, readLogits: true };
    const tokens = [0, 1, 2];
    const result = await session.advanceTokens(tokens, callerOptions);
    if (
      result.status !== ok ||
      result.outputLen !== 0 ||
      result.position !== 3 ||
      readCount !== 0
    ) {
      throw new Error(`LLaMA advanceTokens no-output result mismatch: ${JSON.stringify({ readCount, result })}`);
    }
    if (
      calls.length !== 2 ||
      calls[0].kind !== "sync" ||
      calls[0].executeOptions !== callerOptions ||
      calls[0].internal !== session.noOutputExecuteInternal ||
      calls[0].forcedOutputPolicy !== executeOutputNone ||
      calls[0].tokens !== tokens ||
      calls[1].tokens !== tokens ||
      calls[1].context !== session.tokenExecutorContextScratch ||
      calls[1].execution !== session.tokenExecutionScratch ||
      calls[1].stepParams !== session.stepParamsScratch ||
      calls[1].outputPolicy !== executeOutputNone ||
      session.tokenExecutorOutcomeScratch.status !== ok ||
      session.tokenExecutorOutcomeScratch.outputLength !== 0 ||
      session.tokenExecutorOutcomeScratch.backendDispatchCount !== 0 ||
      session.tokenExecutorOutcomeScratch.fallbackOpCount !== 1 ||
      session.lastTokenExecution !== session.tokenExecutionScratch ||
      session.lastStepParams !== session.stepParamsScratch ||
      session.tokenExecutionHistory.length !== 0 ||
      session.stepParamsHistory.length !== 0 ||
      callerOptions.outputPolicy !== executeOutputLogits ||
      callerOptions.readLogits !== true
    ) {
      throw new Error("LLaMA advanceTokens did not preserve caller options while using no-output internal execution");
    }
  } finally {
    session.dispose();
  }
}

function expectLlamaExecutorOutcomeScratchClearsInvalidResults() {
  const runtime = new WasmWebGpuSessionRuntime({
    invalidArgument,
    ok,
    shapeMismatch,
    unsupported,
  });
  let callCount = 0;
  const session = new WasmWebGpuLlamaResourceSession({
    bindings: { output: { label: "outcome-scratch.output" } },
    contextLength: 8,
    device: {
      canCreateGpuBuffers() {
        return false;
      },
    },
    nativeSession: false,
    runtime,
    sessionHandle: 0x71000033,
    tokenExecutor() {
      callCount += 1;
      if (callCount === 1) {
        return {
          backendDispatchCount: 0,
          backendOpCount: 0,
          commandCount: 7,
          commandOpCount: 7,
          fallbackOpCount: 7,
          genericDispatchCount: 3,
          outputLength: 0,
          scalarDispatchCount: 2,
          status: ok,
          windowDispatchCount: 1,
        };
      }
      return {
        backendDispatchCount: -1,
        outputLength: 0,
        status: ok,
      };
    },
    vocabSize: tinyLlamaVocabSize,
  });
  try {
    const first = session.executeTokensSyncFromListInto({}, [0], {}, executeOutputNone, {
      borrowTokens: true,
      recordHistory: false,
      reuseExecution: true,
      reuseExecutorContext: true,
      reuseStepParams: true,
    });
    if (
      first.status !== ok ||
      session.tokenExecutorOutcomeScratch.backendDispatchCount !== 0 ||
      session.tokenExecutorOutcomeScratch.fallbackOpCount !== 7 ||
      session.tokenExecutorOutcomeScratch.genericDispatchCount !== 3 ||
      session.tokenExecutorOutcomeScratch.scalarDispatchCount !== 2 ||
      session.tokenExecutorOutcomeScratch.windowDispatchCount !== 1
    ) {
      throw new Error(`LLaMA executor outcome scratch did not record first valid result: ${JSON.stringify(first)}`);
    }
    const second = session.executeTokensSyncFromListInto({}, [1], {}, executeOutputNone, {
      borrowTokens: true,
      recordHistory: false,
      reuseExecution: true,
      reuseExecutorContext: true,
      reuseStepParams: true,
    });
    if (
      second.status !== invalidArgument ||
      session.tokenExecutorOutcomeScratch.status !== invalidArgument ||
      second.outputLen !== 0 ||
      session.tokenExecutorOutcomeScratch.outputLength !== 0 ||
      session.tokenExecutorOutcomeScratch.backendDispatchCount !== 0 ||
      session.tokenExecutorOutcomeScratch.backendOpCount !== 0 ||
      session.tokenExecutorOutcomeScratch.commandCount !== 0 ||
      session.tokenExecutorOutcomeScratch.commandOpCount !== 0 ||
      session.tokenExecutorOutcomeScratch.fallbackOpCount !== 0 ||
      session.tokenExecutorOutcomeScratch.genericDispatchCount !== 0 ||
      session.tokenExecutorOutcomeScratch.scalarDispatchCount !== 0 ||
      session.tokenExecutorOutcomeScratch.windowDispatchCount !== 0
    ) {
      throw new Error(`LLaMA executor outcome scratch did not clear invalid result counters: ${JSON.stringify(second)}`);
    }
  } finally {
    session.dispose();
  }
}

function expectLlamaValidationUsesTokensLenWindow() {
  const runtime = new WasmWebGpuSessionRuntime({
    invalidArgument,
    ok,
    shapeMismatch,
    unsupported,
  });
  const dispatched = [];
  const session = new WasmWebGpuLlamaResourceSession({
    bindings: { output: { label: "validation-window.output" } },
    contextLength: 8,
    device: {
      canCreateGpuBuffers() {
        return false;
      },
    },
    nativeSession: false,
    runtime,
    sessionHandle: 0x71000032,
    tokenExecutor(context) {
      dispatched.push({
        tokenCount: context.stepParams.tokenCount,
        tokenValue: context.execution.tokens[0],
      });
      return {
        backendDispatchCount: 0,
        backendOpCount: 0,
        commandCount: 1,
        commandOpCount: 1,
        fallbackOpCount: 1,
        outputLength: 0,
        status: ok,
      };
    },
    vocabSize: tinyLlamaVocabSize,
  });
  try {
    const borrowedTokens = [1, tinyLlamaVocabSize + 1];
    borrowedTokens[Symbol.iterator] = () => {
      throw new Error("LLaMA token validation should index by tokensLen instead of iterating");
    };
    const result = session.executeDecodedTokensSyncInto({}, {
      outputLen: 0,
      outputPolicy: executeOutputNone,
      outputPtr: 0,
      tokens: borrowedTokens,
      tokensLen: 1,
    }, 0, runtime, {
      recordHistory: false,
      reuseExecutorContext: true,
      reuseStepParams: true,
    });
    if (
      result.status !== ok ||
      result.outputLen !== 0 ||
      session.position !== 1 ||
      dispatched.length !== 1 ||
      dispatched[0].tokenCount !== 1 ||
      dispatched[0].tokenValue !== 1
    ) {
      throw new Error(`LLaMA token validation did not honor tokensLen window: ${JSON.stringify({ dispatched, position: session.position, result })}`);
    }
    const publicTokens = [2, tinyLlamaVocabSize + 2];
    publicTokens[Symbol.iterator] = () => {
      throw new Error("LLaMA public token normalization should index borrowed tokensLen window instead of iterating");
    };
    const publicResult = session.executeTokensSyncFromListInto({}, publicTokens, {
      borrowTokens: true,
      outputPolicy: executeOutputNone,
      recordHistory: false,
      reuseExecution: true,
      reuseExecutorContext: true,
      reuseStepParams: true,
      tokensLen: 1,
    });
    if (
      publicResult.status !== ok ||
      publicResult.outputLen !== 0 ||
      session.position !== 2 ||
      dispatched.length !== 2 ||
      dispatched[1].tokenCount !== 1 ||
      dispatched[1].tokenValue !== 2 ||
      session.tokenExecutionScratch.tokens !== publicTokens ||
      session.tokenExecutionScratch.tokensLen !== 1
    ) {
      throw new Error(`LLaMA public token execution did not honor borrowed tokensLen window: ${JSON.stringify({ dispatched, position: session.position, publicResult })}`);
    }
    const undersized = session.executeDecodedTokensSyncInto({}, {
      outputLen: 0,
      outputPolicy: executeOutputNone,
      outputPtr: 0,
      tokens: [1],
      tokensLen: 2,
    }, 0, runtime, {
      recordHistory: false,
      reuseExecutorContext: true,
      reuseStepParams: true,
    });
	    if (
	      undersized.status !== shapeMismatch ||
	      session.position !== 2 ||
	      dispatched.length !== 2
	    ) {
	      throw new Error(`LLaMA token validation did not reject undersized borrowed token container: ${JSON.stringify({ dispatched, position: session.position, undersized })}`);
	    }
	    session.maxTokenWindow = 1;
	    const tooWide = session.executeDecodedTokensSyncInto({}, {
	      outputLen: 0,
	      outputPolicy: executeOutputNone,
	      outputPtr: 0,
	      tokens: [3, 4],
	      tokensLen: 2,
	    }, 0, runtime, {
	      recordHistory: false,
	      reuseExecutorContext: true,
	      reuseStepParams: true,
	    });
	    if (
	      tooWide.status !== shapeMismatch ||
	      session.position !== 2 ||
	      dispatched.length !== 2
	    ) {
	      throw new Error(`LLaMA token validation did not reject a window wider than the compiled max token window: ${JSON.stringify({ dispatched, position: session.position, tooWide })}`);
	    }
	  } finally {
    session.dispose();
  }
}

function expectLlamaDiagnosticHistoryIsBounded() {
  const runtime = new WasmWebGpuSessionRuntime({
    invalidArgument,
    ok,
    shapeMismatch,
    unsupported,
  });
  let calls = 0;
  const session = new WasmWebGpuLlamaResourceSession({
    bindings: { output: { label: "bounded-history.output" } },
    contextLength: 8,
    device: {
      canCreateGpuBuffers() {
        return false;
      },
    },
    diagnosticHistoryLimit: 2,
    nativeSession: false,
    runtime,
    sessionHandle: 0x71000034,
    tokenExecutor() {
      calls += 1;
      return {
        backendDispatchCount: 0,
        backendOpCount: 0,
        commandCount: 1,
        commandOpCount: 1,
        fallbackOpCount: 1,
        outputLength: 0,
        status: ok,
      };
    },
    vocabSize: tinyLlamaVocabSize,
  });
  try {
    for (let token = 0; token < 4; token += 1) {
      const tokens = [token, 0xffffffff];
      const execution = {
        outputLen: 0,
        outputPolicy: executeOutputNone,
        outputPtr: 0,
        tokens,
        tokensLen: 1,
      };
      const result = session.executeDecodedTokensSyncInto({}, execution, 0, runtime, { reuseStepParams: true });
      if (result.status !== ok) {
        throw new Error(`bounded LLaMA diagnostic history execution failed: ${JSON.stringify(result)}`);
      }
      tokens[0] = 0xffffffff;
      execution.tokensLen = 2;
    }
    if (
      calls !== 4 ||
      session.position !== 4 ||
      session.tokenExecutionHistory.length !== 2 ||
      session.stepParamsHistory.length !== 2 ||
      !Object.isFrozen(session.tokenExecutionHistory[0]) ||
      !Object.isFrozen(session.tokenExecutionHistory[0].tokens) ||
      !Object.isFrozen(session.stepParamsHistory[0]) ||
      session.tokenExecutionHistory[0].tokens[0] !== 2 ||
      session.tokenExecutionHistory[1].tokens[0] !== 3 ||
      session.tokenExecutionHistory[0].tokens.length !== 1 ||
      session.tokenExecutionHistory[1].tokens.length !== 1 ||
      session.stepParamsHistory[0].startPosition !== 2 ||
      session.stepParamsHistory[0].endPosition !== 3 ||
      session.stepParamsHistory[1].startPosition !== 3 ||
      session.stepParamsHistory[1].endPosition !== 4
    ) {
      throw new Error(`LLaMA diagnostic histories were not bounded to the newest records: ${JSON.stringify({
        calls,
        position: session.position,
        stepHistory: session.stepParamsHistory,
        tokenHistory: session.tokenExecutionHistory,
      })}`);
    }
  } finally {
    session.dispose();
  }

  const disabled = new WasmWebGpuLlamaResourceSession({
    bindings: { output: { label: "disabled-history.output" } },
    contextLength: 4,
    device: {
      canCreateGpuBuffers() {
        return false;
      },
    },
    diagnosticHistoryLimit: 0,
    nativeSession: false,
    runtime,
    sessionHandle: 0x71000035,
    tokenExecutor() {
      return {
        backendDispatchCount: 0,
        backendOpCount: 0,
        commandCount: 1,
        commandOpCount: 1,
        fallbackOpCount: 1,
        outputLength: 0,
        status: ok,
      };
    },
    vocabSize: tinyLlamaVocabSize,
  });
  try {
    const result = disabled.executeDecodedTokensSyncInto({}, {
      outputLen: 0,
      outputPolicy: executeOutputNone,
      outputPtr: 0,
      tokens: [1],
      tokensLen: 1,
    }, 0, runtime);
    if (
      result.status !== ok ||
      disabled.lastTokenExecution === null ||
      disabled.lastStepParams === null ||
      disabled.tokenExecutionHistory.length !== 0 ||
      disabled.stepParamsHistory.length !== 0
    ) {
      throw new Error("LLaMA diagnostic history limit 0 did not preserve last evidence while disabling retained history");
    }
  } finally {
    disabled.dispose();
  }
}

async function expectWasmWebGpuDeviceReusesReadbackStaging() {
  const hadGpuBufferUsage = Object.prototype.hasOwnProperty.call(globalThis, "GPUBufferUsage");
  const previousGpuBufferUsage = globalThis.GPUBufferUsage;
  const hadGpuMapMode = Object.prototype.hasOwnProperty.call(globalThis, "GPUMapMode");
  const previousGpuMapMode = globalThis.GPUMapMode;
  globalThis.GPUBufferUsage = webgpuBufferUsage;
  globalThis.GPUMapMode = { READ: 1 };
  let copyCount = 0;
  let mapCount = 0;
  let readbackCreateCount = 0;
  let readbackDestroyCount = 0;
  let submitCount = 0;
  let unmapCount = 0;
  function sourceBytes(source, sourceOffset, byteLength) {
    if (source instanceof ArrayBuffer) return new Uint8Array(source, sourceOffset, byteLength);
    if (ArrayBuffer.isView(source)) return new Uint8Array(source.buffer, source.byteOffset + sourceOffset, byteLength);
    throw new Error("fake WebGPU write source must be bytes");
  }
  const fakeGpuDevice = {
    queue: {
      submit() {
        submitCount += 1;
      },
      writeBuffer(buffer, offset, source, sourceOffset = 0, byteLength = source.byteLength - sourceOffset) {
        new Uint8Array(buffer.data, offset, byteLength).set(sourceBytes(source, sourceOffset, byteLength));
      },
    },
    createBuffer(desc) {
      const isReadback = String(desc.label ?? "").includes("readback");
      if (isReadback) readbackCreateCount += 1;
      return {
        ...desc,
        data: new ArrayBuffer(desc.size),
        destroyed: false,
        mapState: "unmapped",
        size: desc.size,
        async mapAsync(mode) {
          if (mode !== 1) throw new Error(`unexpected fake WebGPU map mode: ${mode}`);
          if (this.destroyed) throw new Error("fake WebGPU buffer was destroyed before map");
          this.mapState = "mapped";
          mapCount += 1;
        },
        getMappedRange() {
          if (this.mapState !== "mapped") throw new Error("fake WebGPU buffer is not mapped");
          return this.data;
        },
        unmap() {
          if (this.mapState === "mapped") {
            this.mapState = "unmapped";
            unmapCount += 1;
          }
        },
        destroy() {
          if (!this.destroyed && isReadback) readbackDestroyCount += 1;
          this.destroyed = true;
        },
      };
    },
    createCommandEncoder() {
      return {
        copyBufferToBuffer(source, sourceOffset, target, targetOffset, byteLength) {
          copyCount += 1;
          new Uint8Array(target.data, targetOffset, byteLength).set(new Uint8Array(source.data, sourceOffset, byteLength));
        },
        finish() {
          return { kind: "fake-command-buffer" };
        },
      };
    },
  };
  const device = new WasmWebGpuDevice({ device: fakeGpuDevice });
  try {
    const source = device.createBuffer({
      byteLength: 32,
      label: "fake.source",
      usage: webgpuBufferUsage.STORAGE | webgpuBufferUsage.COPY_SRC | webgpuBufferUsage.COPY_DST,
    });
    device.writeFloat32(source, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));

    const first = new Float32Array(4);
    await device.readFloat32Into(source, first, 4);
    expectClose(Array.from(first), [1, 2, 3, 4]);

    const second = new Float32Array(2);
    await device.readFloat32Into(source, second, 2, 2 * Float32Array.BYTES_PER_ELEMENT);
    expectClose(Array.from(second), [3, 4]);

    const bytes = await device.readBytes(source, 8);
    const byteView = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    expectClose([byteView.getFloat32(0, true), byteView.getFloat32(4, true)], [1, 2]);

    const callerBytes = new Uint8Array(8);
    const intoBytes = await device.readBytesInto(source, callerBytes, 8, 2 * Float32Array.BYTES_PER_ELEMENT);
    const intoByteView = new DataView(callerBytes.buffer, callerBytes.byteOffset, callerBytes.byteLength);
    if (intoBytes !== callerBytes) {
      throw new Error("WebGPU readBytesInto did not return caller-owned byte storage");
    }
    expectClose([intoByteView.getFloat32(0, true), intoByteView.getFloat32(4, true)], [3, 4]);

    if (
      readbackCreateCount !== 1 ||
      readbackDestroyCount !== 0 ||
      device.readbackStaging?.byteLength !== 16 ||
      device.ownedResources.length !== 2
    ) {
      throw new Error(`WebGPU readback staging was not reused for same-capacity reads: ${JSON.stringify({ byteLength: device.readbackStaging?.byteLength, ownedResources: device.ownedResources.length, readbackCreateCount, readbackDestroyCount })}`);
    }

    const largerBytes = await device.readBytes(source, 32);
    if (
      largerBytes.byteLength !== 32 ||
      readbackCreateCount !== 2 ||
      readbackDestroyCount !== 1 ||
      device.readbackStaging?.byteLength !== 32 ||
      device.ownedResources.length !== 2
    ) {
      throw new Error(`WebGPU readback staging did not grow exactly once: ${JSON.stringify({ byteLength: device.readbackStaging?.byteLength, ownedResources: device.ownedResources.length, larger: largerBytes.byteLength, readbackCreateCount, readbackDestroyCount })}`);
    }

    const third = new Float32Array(8);
    await device.readFloat32Into(source, third, 8);
    expectClose(Array.from(third), [1, 2, 3, 4, 5, 6, 7, 8]);
    if (readbackCreateCount !== 2 || readbackDestroyCount !== 1 || device.ownedResources.length !== 2) {
      throw new Error(`WebGPU readback staging did not reuse grown capacity: ${JSON.stringify({ ownedResources: device.ownedResources.length, readbackCreateCount, readbackDestroyCount })}`);
    }

    const overlappingA = new Float32Array(2);
    const overlappingB = new Float32Array(2);
    const overlappingFirst = device.readFloat32Into(source, overlappingA, 2, 0);
    const overlappingSecond = device.readFloat32Into(source, overlappingB, 2, 2 * Float32Array.BYTES_PER_ELEMENT);
    if (
      readbackCreateCount !== 3 ||
      readbackDestroyCount !== 1 ||
      device.readbackStaging?.byteLength !== 32 ||
      device.ownedResources.length !== 3
    ) {
      throw new Error(`WebGPU overlapping readback did not create one temporary staging buffer: ${JSON.stringify({ byteLength: device.readbackStaging?.byteLength, ownedResources: device.ownedResources.length, readbackCreateCount, readbackDestroyCount })}`);
    }
    await Promise.all([overlappingFirst, overlappingSecond]);
    expectClose(Array.from(overlappingA), [1, 2]);
    expectClose(Array.from(overlappingB), [3, 4]);
    if (
      readbackCreateCount !== 3 ||
      readbackDestroyCount !== 2 ||
      device.readbackStaging?.byteLength !== 32 ||
      device.ownedResources.length !== 2
    ) {
      throw new Error(`WebGPU overlapping readback temporary staging was not released: ${JSON.stringify({ byteLength: device.readbackStaging?.byteLength, ownedResources: device.ownedResources.length, readbackCreateCount, readbackDestroyCount })}`);
    }

    if (copyCount !== 8 || submitCount !== 8 || mapCount !== 8 || unmapCount !== 8) {
      throw new Error(`unexpected WebGPU readback staging counters: ${JSON.stringify({ copyCount, mapCount, submitCount, unmapCount })}`);
    }
    device.destroy();
    if (readbackDestroyCount !== 3) {
      throw new Error(`WebGPU readback staging was not destroyed with device: ${JSON.stringify({ readbackDestroyCount })}`);
    }
  } finally {
    device.destroy();
    if (hadGpuBufferUsage) {
      globalThis.GPUBufferUsage = previousGpuBufferUsage;
    } else {
      delete globalThis.GPUBufferUsage;
    }
    if (hadGpuMapMode) {
      globalThis.GPUMapMode = previousGpuMapMode;
    } else {
      delete globalThis.GPUMapMode;
    }
  }
}

async function expectLlamaDeviceArgmaxSelectorResourceReuseAndValidation() {
  const hadGpuBufferUsage = Object.prototype.hasOwnProperty.call(globalThis, "GPUBufferUsage");
  const previousGpuBufferUsage = globalThis.GPUBufferUsage;
  if (typeof globalThis.GPUBufferUsage !== "object") {
    globalThis.GPUBufferUsage = webgpuBufferUsage;
  }
  let bindGroupCreateCount = 0;
  let paramsBufferCreateCount = 0;
  let resultBufferCreateCount = 0;
  let resultDestroyCount = 0;
  let submitCount = 0;
  let readCount = 0;
  let invalidResult = false;
  const readTargets = [];
  const uploadSources = [];
  const fakeGpuDevice = {
    limits: { minStorageBufferOffsetAlignment: 4 },
    queue: {
      submit() {
        submitCount += 1;
      },
      writeBuffer(_buffer, _offset, source) {
        uploadSources.push(source);
      },
    },
    createBindGroup(desc) {
      bindGroupCreateCount += 1;
      return { desc, kind: "fake-bind-group" };
    },
    createBuffer(desc) {
      const isResult = desc.label === "zgml.wasm.webgpu.llama-device-argmax.result";
      if (desc.label === "zgml.wasm.webgpu.llama-device-argmax.params") {
        paramsBufferCreateCount += 1;
      } else if (isResult) {
        resultBufferCreateCount += 1;
      }
      return {
        ...desc,
        destroyed: false,
        destroy() {
          if (!this.destroyed && isResult) resultDestroyCount += 1;
          this.destroyed = true;
        },
        size: desc.size,
      };
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            dispatchWorkgroups() {},
            end() {},
            setBindGroup() {},
            setPipeline() {},
          };
        },
        finish() {
          return { kind: "fake-command-buffer" };
        },
      };
    },
    createComputePipeline() {
      return {
        getBindGroupLayout() {
          return { kind: "fake-bind-group-layout" };
        },
      };
    },
    createShaderModule(desc) {
      return { desc, kind: "fake-shader-module" };
    },
  };
  const fakeDevice = new WasmWebGpuDevice({ device: fakeGpuDevice });
  fakeDevice.readBytesInto = async (_descriptor, target, byteLength) => {
    if (byteLength !== 8) throw new Error(`unexpected LLaMA argmax readback byte length: ${byteLength}`);
    readCount += 1;
    readTargets.push(target);
    const view = new DataView(target.buffer, target.byteOffset, byteLength);
    view.setUint32(0, invalidResult ? tinyLlamaVocabSize : 6, true);
    view.setFloat32(4, invalidResult ? 1 : 12.5, true);
    return target;
  };
  const fakeDescriptor = (label, byteLength, access = "read") => {
    const resource = fakeDevice.createBuffer({
      byteLength,
      label,
      usage: webgpuBufferUsage.STORAGE | webgpuBufferUsage.COPY_SRC | webgpuBufferUsage.COPY_DST,
    });
    return fakeDevice.descriptor(resource, { access, byteLength });
  };
  const runtime = new WasmWebGpuSessionRuntime({
    invalidArgument,
    ok,
    shapeMismatch,
    unsupported,
  });
  const session = new WasmWebGpuLlamaResourceSession({
    bindings: {
      output: fakeDescriptor("fake.argmax.logits", tinyLlamaVocabSize * Float32Array.BYTES_PER_ELEMENT, "readwrite"),
    },
    contextLength: 4,
    device: fakeDevice,
    nativeSession: false,
    runtime,
	    sessionHandle: 0x71000003,
	    vocabSize: tinyLlamaVocabSize,
	  });
	  let argmaxResultHandle = null;
	  let argmaxResultResource = null;
	  try {
    const first = await session.sampleDevice({ topK: 1 });
    const second = await session.sampleDevice({ topK: 1 });
    if (
      first.status !== ok ||
      first.token !== 6 ||
      first.logit !== 12.5 ||
      second.status !== ok ||
      second.token !== 6 ||
      second.logit !== 12.5
    ) {
      throw new Error(`LLaMA device argmax selector returned unexpected candidates: ${JSON.stringify({ first, second })}`);
    }
    if (paramsBufferCreateCount !== 1 || resultBufferCreateCount !== 1 || bindGroupCreateCount !== 1) {
      throw new Error(`LLaMA device argmax selector did not reuse cached GPU objects: ${JSON.stringify({ bindGroupCreateCount, paramsBufferCreateCount, resultBufferCreateCount })}`);
    }
	    if (resultDestroyCount !== 0 || submitCount !== 2 || readCount !== 2 || uploadSources.length !== 2 || uploadSources[0] !== uploadSources[1]) {
	      throw new Error(`LLaMA device argmax selector reuse evidence mismatch: ${JSON.stringify({ readCount, resultDestroyCount, submitCount, uploadCount: uploadSources.length })}`);
	    }
	    if (fakeDevice.ownedResources.length !== 2) {
	      throw new Error(`LLaMA device argmax selector did not retain exactly output plus result resources: ${fakeDevice.ownedResources.length}`);
	    }
	    argmaxResultHandle = session.deviceArgmaxSelector.resultDescriptor?.handle;
	    argmaxResultResource = session.deviceArgmaxSelector.resultResource;
	    if (!Number.isSafeInteger(argmaxResultHandle) || fakeDevice.resourceFor(argmaxResultHandle) !== session.deviceArgmaxSelector.resultResource) {
	      throw new Error("LLaMA device argmax selector result handle was not registered");
	    }
	    if (readTargets[0] !== readTargets[1] || readTargets[0] !== session.deviceArgmaxSelector.resultBytes) {
	      throw new Error("LLaMA device argmax selector did not reuse result byte scratch");
	    }
    const profile = session.runtimeProfile();
    if (
      profile.selectionCallCount !== 2 ||
      profile.selectionBackendDispatchCount !== 2 ||
      profile.selectionResultReadCount !== 2 ||
      profile.selectionSyncCount !== 2 ||
      profile.selectionFallbackOpCount !== 0
    ) {
      throw new Error(`unexpected LLaMA device argmax selector profile: ${JSON.stringify(profile)}`);
    }

    invalidResult = true;
    let rejected = false;
    try {
      await session.deviceArgmaxSelector.select(session);
    } catch (err) {
      if (!String(err?.message ?? err).includes("invalid candidate")) throw err;
      rejected = true;
    }
    if (!rejected) {
      throw new Error("LLaMA device argmax selector accepted an out-of-vocab GPU result");
    }
    if (paramsBufferCreateCount !== 1 || resultBufferCreateCount !== 1 || bindGroupCreateCount !== 1 || submitCount !== 3 || readCount !== 3 || uploadSources.length !== 3) {
      throw new Error(`LLaMA device argmax invalid-result path rebuilt cached GPU objects: ${JSON.stringify({ bindGroupCreateCount, paramsBufferCreateCount, readCount, resultBufferCreateCount, submitCount, uploadCount: uploadSources.length })}`);
    }
	    if (readTargets[2] !== readTargets[0]) {
	      throw new Error("LLaMA device argmax invalid-result path did not reuse result byte scratch");
	    }
	  } finally {
	    session.dispose();
	    if (fakeDevice.ownedResources.length !== 1) {
	      throw new Error(`LLaMA device argmax selector did not forget result resource on dispose: ${fakeDevice.ownedResources.length}`);
	    }
	    if (Number.isSafeInteger(argmaxResultHandle) && fakeDevice.resourceFor(argmaxResultHandle) !== null) {
	      throw new Error("LLaMA device argmax selector result handle still resolved after dispose");
	    }
	    expectDestroyedWasmWebGpuResourceRejected(fakeDevice, argmaxResultResource, "LLaMA device argmax selector result", 8);
	    fakeDevice.destroy();
	    if (hadGpuBufferUsage) {
      globalThis.GPUBufferUsage = previousGpuBufferUsage;
    } else {
      delete globalThis.GPUBufferUsage;
    }
  }
}

function expectDestroyedWasmWebGpuResourceRejected(device, resource, label, byteLength = 8) {
  if (!resource || typeof resource !== "object") {
    throw new Error(`${label} did not retain a destroyed resource object`);
  }
  let descriptorRejected = false;
  try {
    device.descriptor(resource, { access: "readwrite", byteLength });
  } catch (err) {
    if (!String(err?.message ?? err).includes("already destroyed")) throw err;
    descriptorRejected = true;
  }
  if (!descriptorRejected) {
    throw new Error(`${label} descriptor accepted a destroyed resource`);
  }
  let writeRejected = false;
  try {
    device.writeBytes(resource, new Uint8Array(byteLength));
  } catch (err) {
    if (!String(err?.message ?? err).includes("already destroyed")) throw err;
    writeRejected = true;
  }
  if (!writeRejected) {
    throw new Error(`${label} write accepted a destroyed resource`);
  }
}

async function expectLlamaDeviceTopKSelectorResourceReuse() {
  const hadGpuBufferUsage = Object.prototype.hasOwnProperty.call(globalThis, "GPUBufferUsage");
  const previousGpuBufferUsage = globalThis.GPUBufferUsage;
  if (typeof globalThis.GPUBufferUsage !== "object") {
    globalThis.GPUBufferUsage = webgpuBufferUsage;
  }
  let bindGroupCreateCount = 0;
  let paramsBufferCreateCount = 0;
  let resultBufferCreateCount = 0;
  let resultDestroyCount = 0;
  let submitCount = 0;
  let readCount = 0;
  const readTargets = [];
  const uploadSources = [];
  const fakeGpuDevice = {
    limits: { minStorageBufferOffsetAlignment: 4 },
    queue: {
      submit() {
        submitCount += 1;
      },
      writeBuffer(_buffer, _offset, source) {
        uploadSources.push(source);
      },
    },
    createBindGroup(desc) {
      bindGroupCreateCount += 1;
      return { desc, kind: "fake-bind-group" };
    },
    createBuffer(desc) {
      const isResult = desc.label === "zgml.wasm.webgpu.llama-device-topk.result";
      if (desc.label === "zgml.wasm.webgpu.llama-device-topk.params") {
        paramsBufferCreateCount += 1;
      } else if (isResult) {
        resultBufferCreateCount += 1;
      }
      return {
        ...desc,
        destroyed: false,
        destroy() {
          if (!this.destroyed && isResult) resultDestroyCount += 1;
          this.destroyed = true;
        },
        size: desc.size,
      };
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            dispatchWorkgroups() {},
            end() {},
            setBindGroup() {},
            setPipeline() {},
          };
        },
        finish() {
          return { kind: "fake-command-buffer" };
        },
      };
    },
    createComputePipeline() {
      return {
        getBindGroupLayout() {
          return { kind: "fake-bind-group-layout" };
        },
      };
    },
    createShaderModule(desc) {
      return { desc, kind: "fake-shader-module" };
    },
  };
  const fakeDevice = new WasmWebGpuDevice({ device: fakeGpuDevice });
  fakeDevice.readBytesInto = async (_descriptor, target, byteLength) => {
    readCount += 1;
    readTargets.push(target);
    const view = new DataView(target.buffer, target.byteOffset, byteLength);
    for (let i = 0; i < byteLength / 8; i += 1) {
      view.setUint32(i * 8, i, true);
      view.setFloat32(i * 8 + 4, 10 - i, true);
    }
    return target;
  };
  const fakeDescriptor = (label, byteLength, access = "read") => {
    const resource = fakeDevice.createBuffer({
      byteLength,
      label,
      usage: webgpuBufferUsage.STORAGE | webgpuBufferUsage.COPY_SRC | webgpuBufferUsage.COPY_DST,
    });
    return fakeDevice.descriptor(resource, { access, byteLength });
  };
  const runtime = new WasmWebGpuSessionRuntime({
    invalidArgument,
    ok,
    shapeMismatch,
    unsupported,
  });
	  const session = new WasmWebGpuLlamaResourceSession({
    bindings: {
      output: fakeDescriptor("fake.topk.logits", tinyLlamaVocabSize * Float32Array.BYTES_PER_ELEMENT, "readwrite"),
    },
    contextLength: 4,
    device: fakeDevice,
    nativeSession: false,
    runtime,
	    sessionHandle: 0x71000002,
	    vocabSize: tinyLlamaVocabSize,
	  });
	  let topKResultHandle = null;
	  let topKResultResource = null;
	  let topKResultByteLength = 0;
	  try {
	    const first = await session.sampleDevice({ seed: 10752, temperature: 1, topK: 3 });
	    const second = await session.sampleDevice({ seed: 11776, temperature: 1, topK: 2 });
	    const oldTopKResultHandle = session.deviceTopKSelector.resultDescriptor?.handle;
	    const oldTopKResultResource = session.deviceTopKSelector.resultResource;
	    const oldTopKResultByteLength = session.deviceTopKSelector.resultDescriptor?.byteLength ?? 0;
	    const third = await session.sampleDevice({ seed: 14336, temperature: 1, topK: 4 });
    if (first.status !== ok || second.status !== ok || third.status !== ok) {
      throw new Error(`LLaMA device top-k reuse proof selection failed: ${JSON.stringify({ first, second, third })}`);
    }
    if (first.token !== 1 || first.logit !== 9 || second.token !== 1 || second.logit !== 9 || third.token !== 2 || third.logit !== 8) {
      throw new Error(`LLaMA device top-k selector returned unexpected sampled candidates: ${JSON.stringify({ first, second, third })}`);
    }
    if (paramsBufferCreateCount !== 1 || resultBufferCreateCount !== 2 || bindGroupCreateCount !== 2) {
      throw new Error(`LLaMA device top-k selector did not reuse cached GPU objects by capacity: ${JSON.stringify({ bindGroupCreateCount, paramsBufferCreateCount, resultBufferCreateCount })}`);
    }
	    if (resultDestroyCount !== 1) {
	      throw new Error(`LLaMA device top-k selector did not destroy the old result buffer on capacity growth: ${resultDestroyCount}`);
	    }
	    if (fakeDevice.ownedResources.length !== 2) {
	      throw new Error(`LLaMA device top-k selector did not forget old result resource on capacity growth: ${fakeDevice.ownedResources.length}`);
	    }
	    if (Number.isSafeInteger(oldTopKResultHandle) && fakeDevice.resourceFor(oldTopKResultHandle) !== null) {
	      throw new Error("LLaMA device top-k selector old result handle still resolved after capacity growth");
	    }
	    expectDestroyedWasmWebGpuResourceRejected(fakeDevice, oldTopKResultResource, "LLaMA device top-k selector old result", oldTopKResultByteLength);
	    topKResultHandle = session.deviceTopKSelector.resultDescriptor?.handle;
	    topKResultResource = session.deviceTopKSelector.resultResource;
	    topKResultByteLength = session.deviceTopKSelector.resultDescriptor?.byteLength ?? 0;
	    if (!Number.isSafeInteger(topKResultHandle) || fakeDevice.resourceFor(topKResultHandle) !== session.deviceTopKSelector.resultResource) {
	      throw new Error("LLaMA device top-k selector result handle was not registered after capacity growth");
	    }
	    if (submitCount !== 3 || readCount !== 3 || uploadSources.length !== 3) {
	      throw new Error(`LLaMA device top-k selector dispatch/read/upload evidence mismatch: ${JSON.stringify({ readCount, submitCount, uploadCount: uploadSources.length })}`);
	    }
    if (
      readTargets.length !== 3 ||
      readTargets[0] !== readTargets[1] ||
      readTargets[2] === readTargets[0] ||
      readTargets[2] !== session.deviceTopKSelector.resultBytes ||
      session.deviceTopKSelector.resultByteCapacity !== 32
    ) {
      throw new Error(`LLaMA device top-k selector did not reuse/grow result byte scratch by capacity: ${JSON.stringify({ capacity: session.deviceTopKSelector.resultByteCapacity, readTargets: readTargets.map((target) => target.byteLength) })}`);
    }
    if (uploadSources[0] !== uploadSources[1] || uploadSources[0] !== uploadSources[2]) {
      throw new Error("LLaMA device top-k selector did not reuse the params upload view");
    }
    const profile = session.runtimeProfile();
    if (
      profile.selectionCallCount !== 3 ||
      profile.selectionBackendDispatchCount !== 3 ||
      profile.selectionResultReadCount !== 3 ||
      profile.selectionSyncCount !== 3 ||
      profile.selectionFallbackOpCount !== 0
    ) {
      throw new Error(`unexpected LLaMA device top-k selector profile: ${JSON.stringify(profile)}`);
    }
	  } finally {
	    session.dispose();
	    if (fakeDevice.ownedResources.length !== 1) {
	      throw new Error(`LLaMA device top-k selector did not forget result resource on dispose: ${fakeDevice.ownedResources.length}`);
	    }
	    if (Number.isSafeInteger(topKResultHandle) && fakeDevice.resourceFor(topKResultHandle) !== null) {
	      throw new Error("LLaMA device top-k selector result handle still resolved after dispose");
	    }
	    expectDestroyedWasmWebGpuResourceRejected(fakeDevice, topKResultResource, "LLaMA device top-k selector result", topKResultByteLength);
	    fakeDevice.destroy();
	    if (hadGpuBufferUsage) {
      globalThis.GPUBufferUsage = previousGpuBufferUsage;
    } else {
      delete globalThis.GPUBufferUsage;
    }
  }
}

function expectTinyLlamaModelResources(resources, label) {
  if (!Array.isArray(resources) || resources.length !== tinyLlamaTensorSpecs.length) {
    throw new Error(`${label} retained ${resources?.length ?? 0} model resources, expected ${tinyLlamaTensorSpecs.length}`);
  }
  let dataByteOffset = 0;
  for (let i = 0; i < tinyLlamaTensorSpecs.length; i += 1) {
    const [name, shape] = tinyLlamaTensorSpecs[i];
    const entry = resources[i];
    const elementCount = shape.reduce((count, dim) => count * dim, 1);
    const byteLength = elementCount * Float32Array.BYTES_PER_ELEMENT;
    if (
      entry.role !== `llama.weight.${name}` ||
      entry.dtype !== "f32" ||
      entry.elementCount !== elementCount ||
      entry.byteLength !== byteLength ||
      entry.dataByteOffset !== dataByteOffset ||
      entry.dataByteLength !== byteLength ||
      !Array.isArray(entry.shape) ||
      entry.shape.length !== shape.length ||
      entry.shape.some((dim, j) => dim !== shape[j])
    ) {
      throw new Error(`${label} retained wrong model resource metadata for ${name}`);
    }
    dataByteOffset += byteLength;
  }
}

function safetensorsDataStart(bytes) {
  return 8 + Number(new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getBigUint64(0, true));
}

function safetensorsF32(source, spec, index) {
  const dtype = spec.dtype ?? "f32";
  return readSafetensorsScalar(
    new DataView(source.buffer, source.byteOffset, source.byteLength),
    safetensorsDataStart(source) + spec.dataByteOffset + index * safetensorsScalarBytes(dtype),
    dtype,
  );
}

function tinyLlamaRootTensorNames(options = {}) {
  const modelResourceOptions = options.modelResourceOptions ?? options;
  return {
    embedding: modelResourceOptions.embeddingTensor ?? "model.embed_tokens.weight",
    lmHead: modelResourceOptions.lmHeadTensor ?? "lm_head.weight",
    lmHeadBias: modelResourceOptions.lmHeadBiasTensor ?? "lm_head.bias",
    norm: modelResourceOptions.normTensor ?? "model.norm.weight",
  };
}

function tinyLlamaEmbeddingProjectionLogits(source, token, options = {}) {
  const modelResourceOptions = options.modelResourceOptions ?? options;
  const rootTensors = tinyLlamaRootTensorNames(options);
  const specs = new Map(tinyLlamaModelResourceSpecs(source, modelResourceOptions).map((spec) => [spec.tensor, spec]));
  const embedding = specs.get(rootTensors.embedding);
  const lmHead = specs.get(rootTensors.lmHead);
  const hidden = embedding.shape[1];
  const vocab = embedding.shape[0];
  const logits = [];
  for (let row = 0; row < vocab; row += 1) {
    let acc = Math.fround(0);
    for (let col = 0; col < hidden; col += 1) {
      const embed = safetensorsF32(source, embedding, token * hidden + col);
      const weight = safetensorsF32(source, lmHead, row * hidden + col);
      acc = Math.fround(acc + Math.fround(weight * embed));
    }
    logits.push(acc);
  }
  return tinyAddOptionalBias(source, specs, rootTensors.lmHeadBias, logits);
}

function tinyLlamaRmsNormProjectionLogits(source, token, epsilonOrOptions = 1e-5) {
  const options = tinyLlamaProjectionOptions(epsilonOrOptions);
  const epsilon = options.epsilon;
  const modelResourceOptions = options.modelResourceOptions ?? options;
  const rootTensors = tinyLlamaRootTensorNames(options);
  const specs = new Map(tinyLlamaModelResourceSpecs(source, modelResourceOptions).map((spec) => [spec.tensor, spec]));
  const embedding = specs.get(rootTensors.embedding);
  const norm = specs.get(rootTensors.norm);
  const lmHead = specs.get(rootTensors.lmHead);
  const hidden = embedding.shape[1];
  const vocab = embedding.shape[0];
  let sumSquares = Math.fround(0);
  for (let col = 0; col < hidden; col += 1) {
    const value = safetensorsF32(source, embedding, token * hidden + col);
    sumSquares = Math.fround(sumSquares + Math.fround(value * value));
  }
  const invRms = Math.fround(1 / Math.sqrt(Math.fround(sumSquares / hidden + epsilon)));
  const logits = [];
  for (let row = 0; row < vocab; row += 1) {
    let acc = Math.fround(0);
    for (let col = 0; col < hidden; col += 1) {
      const embed = safetensorsF32(source, embedding, token * hidden + col);
      const normValue = safetensorsF32(source, norm, col);
      const weight = safetensorsF32(source, lmHead, row * hidden + col);
      const hiddenValue = Math.fround(Math.fround(embed * normValue) * invRms);
      acc = Math.fround(acc + Math.fround(weight * hiddenValue));
    }
    logits.push(acc);
  }
  return tinyAddOptionalBias(source, specs, rootTensors.lmHeadBias, logits);
}

function tinyLlamaKvProjectionCache(source, tokens, epsilon = 1e-5) {
  const specs = new Map(tinyLlamaModelResourceSpecs(source).map((spec) => [spec.tensor, spec]));
  const embedding = specs.get("model.embed_tokens.weight");
  const norm = specs.get("model.layers.0.input_layernorm.weight");
  const kProj = specs.get("model.layers.0.self_attn.k_proj.weight");
  const vProj = specs.get("model.layers.0.self_attn.v_proj.weight");
  const kNorm = specs.get("model.layers.0.self_attn.k_norm.weight");
  const hidden = embedding.shape[1];
  const stride = kProj.shape[0];
  const headSize = kNorm?.shape?.[0] ?? stride;
  const kNormValues = kNorm === undefined ? null : tinySafetensorsVector(source, kNorm);
  const out = {
    k: [],
    v: [],
  };
  for (const token of tokens) {
    let sumSquares = Math.fround(0);
    for (let col = 0; col < hidden; col += 1) {
      const value = safetensorsF32(source, embedding, token * hidden + col);
      sumSquares = Math.fround(sumSquares + Math.fround(value * value));
    }
    const invRms = Math.fround(1 / Math.sqrt(Math.fround(sumSquares / hidden + epsilon)));
    const tokenK = [];
    for (let row = 0; row < stride; row += 1) {
      let kAcc = Math.fround(0);
      let vAcc = Math.fround(0);
      for (let col = 0; col < hidden; col += 1) {
        const embed = safetensorsF32(source, embedding, token * hidden + col);
        const normValue = safetensorsF32(source, norm, col);
        const hiddenValue = Math.fround(Math.fround(embed * normValue) * invRms);
        const kWeight = safetensorsF32(source, kProj, row * hidden + col);
        const vWeight = safetensorsF32(source, vProj, row * hidden + col);
        kAcc = Math.fround(kAcc + Math.fround(kWeight * hiddenValue));
        vAcc = Math.fround(vAcc + Math.fround(vWeight * hiddenValue));
      }
      tokenK.push(kAcc);
      out.v.push(vAcc);
    }
    out.k.push(...(kNormValues === null ? tokenK : tinyRmsNormVectorByHead(tokenK, kNormValues, headSize, epsilon)));
  }
  return out;
}

function tinyRmsNormVector(values, normValues, epsilon) {
  let sumSquares = Math.fround(0);
  for (let i = 0; i < values.length; i += 1) {
    sumSquares = Math.fround(sumSquares + Math.fround(values[i] * values[i]));
  }
  const invRms = Math.fround(1 / Math.sqrt(Math.fround(sumSquares / values.length + epsilon)));
  return values.map((value, i) => Math.fround(Math.fround(value * normValues[i]) * invRms));
}

function tinySafetensorsVector(source, spec) {
  const out = [];
  for (let i = 0; i < spec.shape[0]; i += 1) {
    out.push(safetensorsF32(source, spec, i));
  }
  return out;
}

function tinyRmsNormVectorByHead(values, normValues, headSize, epsilon) {
  const out = values.slice();
  for (let headOffset = 0; headOffset < values.length; headOffset += headSize) {
    let sumSquares = Math.fround(0);
    for (let i = 0; i < headSize; i += 1) {
      const value = values[headOffset + i];
      sumSquares = Math.fround(sumSquares + Math.fround(value * value));
    }
    const invRms = Math.fround(1 / Math.sqrt(Math.fround(sumSquares / headSize + epsilon)));
    for (let i = 0; i < headSize; i += 1) {
      out[headOffset + i] = Math.fround(Math.fround(values[headOffset + i] * normValues[i]) * invRms);
    }
  }
  return out;
}

function tinyMatVec(source, spec, input) {
  const rows = spec.shape[0];
  const cols = spec.shape[1];
  const out = [];
  for (let row = 0; row < rows; row += 1) {
    let acc = Math.fround(0);
    for (let col = 0; col < cols; col += 1) {
      acc = Math.fround(acc + Math.fround(safetensorsF32(source, spec, row * cols + col) * input[col]));
    }
    out.push(acc);
  }
  return out;
}

function tinyAddOptionalBias(source, specs, tensorName, values) {
  const spec = specs.get(tensorName);
  if (!spec) return values;
  return values.map((value, i) => Math.fround(value + safetensorsF32(source, spec, i)));
}

function tinyRopeInvFrequency(pair, headSize, base = 10000, scaling = 1) {
  const invFrequency = 1 / Math.pow(base, (2 * pair) / headSize);
  const scale = typeof scaling === "number" ? scaling : (scaling.ropeScale ?? scaling.scale ?? 1);
  const kind = typeof scaling === "number"
    ? (Math.abs(scale - 1) > 1e-6 ? "linear" : "none")
    : (scaling.ropeKind ?? scaling.kind ?? (Math.abs(scale - 1) > 1e-6 ? "linear" : "none"));
  if (kind === "linear") return invFrequency / scale;
  if (kind === "llama3" || kind === "llama_3") {
    const lowFreqFactor = scaling.ropeLowFreqFactor ?? scaling.lowFreqFactor ?? 1;
    const highFreqFactor = scaling.ropeHighFreqFactor ?? scaling.highFreqFactor ?? 4;
    const originalContextLength = scaling.ropeOriginalContextLength ?? scaling.originalContextLength ?? 8192;
    const wavelen = (2 * Math.PI) / invFrequency;
    const lowFreqWavelen = originalContextLength / lowFreqFactor;
    const highFreqWavelen = originalContextLength / highFreqFactor;
    if (wavelen > lowFreqWavelen) return invFrequency / scale;
    if (wavelen < highFreqWavelen) return invFrequency;
    const smoothFactor = (originalContextLength / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
    return Math.fround(Math.fround(1 - smoothFactor) * Math.fround(invFrequency / scale) + Math.fround(smoothFactor * invFrequency));
  }
  return invFrequency;
}

function tinyRopeVector(values, position, base = 10000, scale = 1) {
  const out = Array(values.length).fill(0);
  const half = Math.floor(values.length / 2);
  for (let i = 0; i < half; i += 1) {
    const freq = position * tinyRopeInvFrequency(i, values.length, base, scale);
    const cos = Math.fround(Math.cos(freq));
    const sin = Math.fround(Math.sin(freq));
    const lo = values[i];
    const hi = values[i + half];
    out[i] = Math.fround(Math.fround(lo * cos) - Math.fround(hi * sin));
    out[i + half] = Math.fround(Math.fround(hi * cos) + Math.fround(lo * sin));
  }
  return out;
}

function tinyRopeVectorByHead(values, position, headSize = values.length, base = 10000, scale = 1) {
  if (headSize === values.length) return tinyRopeVector(values, position, base, scale);
  const out = Array(values.length).fill(0);
  const half = Math.floor(headSize / 2);
  for (let headOffset = 0; headOffset < values.length; headOffset += headSize) {
    for (let i = 0; i < half; i += 1) {
      const freq = position * tinyRopeInvFrequency(i, headSize, base, scale);
      const cos = Math.fround(Math.cos(freq));
      const sin = Math.fround(Math.sin(freq));
      const loIndex = headOffset + i;
      const hiIndex = headOffset + i + half;
      const lo = values[loIndex];
      const hi = values[hiIndex];
      out[loIndex] = Math.fround(Math.fround(lo * cos) - Math.fround(hi * sin));
      out[hiIndex] = Math.fround(Math.fround(hi * cos) + Math.fround(lo * sin));
    }
  }
  return out;
}

function tinyLlamaProjectionOptions(epsilonOrOptions = 1e-5, options = {}) {
  if (typeof epsilonOrOptions === "object" && epsilonOrOptions !== null) {
    return {
      ...epsilonOrOptions,
      epsilon: epsilonOrOptions.epsilon ?? 1e-5,
    };
  }
  return {
    ...options,
    epsilon: epsilonOrOptions,
  };
}

function tinyLlamaAttentionProjectionStep(source, cache, token, position, epsilonOrOptions = 1e-5, maybeOptions = {}) {
  const options = tinyLlamaProjectionOptions(epsilonOrOptions, maybeOptions);
  const epsilon = options.epsilon;
  const layer = options.layer ?? 0;
  const layerRole = (suffix) => `model.layers.${layer}.${suffix}`;
  const modelResourceOptions = options.modelResourceOptions ?? options;
  const rootTensors = tinyLlamaRootTensorNames(options);
  const specs = new Map(tinyLlamaModelResourceSpecs(source, modelResourceOptions).map((spec) => [spec.tensor, spec]));
  const embedding = specs.get(rootTensors.embedding);
  const inputNorm = specs.get(layerRole("input_layernorm.weight"));
  const qProj = specs.get(layerRole("self_attn.q_proj.weight"));
  const kProj = specs.get(layerRole("self_attn.k_proj.weight"));
  const vProj = specs.get(layerRole("self_attn.v_proj.weight"));
  const oProj = specs.get(layerRole("self_attn.o_proj.weight"));
  const finalNorm = specs.get(rootTensors.norm);
  const lmHead = specs.get(rootTensors.lmHead);
  const hidden = embedding.shape[1];
  const vocab = embedding.shape[0];
  const embeddingValues = [];
  const inputNormValues = [];
  const finalNormValues = [];
  const inputHidden = options.inputHidden ?? options.hidden ?? null;
  for (let col = 0; col < hidden; col += 1) {
    embeddingValues.push(inputHidden === null ? safetensorsF32(source, embedding, token * hidden + col) : inputHidden[col]);
    inputNormValues.push(safetensorsF32(source, inputNorm, col));
    finalNormValues.push(safetensorsF32(source, finalNorm, col));
  }
  const normed = tinyRmsNormVector(embeddingValues, inputNormValues, epsilon);
  const qRows = qProj.shape[0];
  const kvRows = kProj.shape[0];
  const attentionHeadSize = options.attentionHeadSize ?? options.headSize ?? (kvRows === hidden ? hidden : kvRows);
  const queryHeadCount = qRows / attentionHeadSize;
  const kvHeadCount = kvRows / attentionHeadSize;
  const queryHeadsPerKvHead = queryHeadCount / kvHeadCount;
  const ropeBase = options.ropeBase ?? options.ropeTheta ?? 10000;
  const ropeScaling = {
    ropeHighFreqFactor: options.ropeHighFreqFactor ?? options.rope_high_freq_factor ?? 4,
    ropeKind: options.ropeKind ?? options.ropeType ?? options.rope_kind ?? options.rope_type,
    ropeLowFreqFactor: options.ropeLowFreqFactor ?? options.rope_low_freq_factor ?? 1,
    ropeOriginalContextLength: options.ropeOriginalContextLength ?? options.rope_original_context_length ?? 8192,
    ropeScale: options.ropeScale ?? options.rope_scale ?? options.ropeScalingFactor ?? options.rope_scaling_factor ?? 1,
  };
  const useRope = options.useRope ?? options.use_rope ?? true;
  const qProjected = tinyAddOptionalBias(source, specs, layerRole("self_attn.q_proj.bias"), tinyMatVec(source, qProj, normed));
  const kProjected = tinyAddOptionalBias(source, specs, layerRole("self_attn.k_proj.bias"), tinyMatVec(source, kProj, normed));
  const qNorm = specs.get(layerRole("self_attn.q_norm.weight"));
  const kNorm = specs.get(layerRole("self_attn.k_norm.weight"));
  if ((qNorm === undefined) !== (kNorm === undefined)) {
    throw new Error("tiny LLaMA reference needs both Q and K projection norm tensors");
  }
  const qNormed = qNorm === undefined
    ? qProjected
    : tinyRmsNormVectorByHead(qProjected, tinySafetensorsVector(source, qNorm), attentionHeadSize, epsilon);
  const kNormed = kNorm === undefined
    ? kProjected
    : tinyRmsNormVectorByHead(kProjected, tinySafetensorsVector(source, kNorm), attentionHeadSize, epsilon);
  const q = useRope ? tinyRopeVectorByHead(qNormed, position, attentionHeadSize, ropeBase, ropeScaling) : qNormed;
  const k = useRope ? tinyRopeVectorByHead(kNormed, position, attentionHeadSize, ropeBase, ropeScaling) : kNormed;
  const v = tinyAddOptionalBias(source, specs, layerRole("self_attn.v_proj.bias"), tinyMatVec(source, vProj, normed));
  cache.k.splice(position * kvRows, kvRows, ...k);
  cache.v.splice(position * kvRows, kvRows, ...v);
  const seq = position + 1;
  const slidingWindow =
    options.slidingWindow ??
    options.sliding_window ??
    options.attentionSlidingWindow ??
    options.attention_sliding_window ??
    null;
  const attentionStart = Number.isSafeInteger(slidingWindow) && slidingWindow > 0 && seq > slidingWindow
    ? seq - slidingWindow
    : 0;
  const scale = Math.fround(1 / Math.sqrt(attentionHeadSize));
  const attention = Array(qRows).fill(0);
  for (let queryHead = 0; queryHead < queryHeadCount; queryHead += 1) {
    const kvHead = Math.floor(queryHead / queryHeadsPerKvHead);
    const qOffset = queryHead * attentionHeadSize;
    const kvOffset = kvHead * attentionHeadSize;
    const scores = [];
    let maxScore = -Infinity;
    for (let pos = attentionStart; pos < seq; pos += 1) {
      let dot = Math.fround(0);
      for (let row = 0; row < attentionHeadSize; row += 1) {
        dot = Math.fround(dot + Math.fround(q[qOffset + row] * cache.k[pos * kvRows + kvOffset + row]));
      }
      const score = Math.fround(dot * scale);
      scores.push(score);
      if (score > maxScore) maxScore = score;
    }
    let denom = Math.fround(0);
    for (let pos = attentionStart; pos < seq; pos += 1) {
      const weight = Math.fround(Math.exp(Math.fround(scores[pos - attentionStart] - maxScore)));
      denom = Math.fround(denom + weight);
      for (let row = 0; row < attentionHeadSize; row += 1) {
        attention[qOffset + row] = Math.fround(attention[qOffset + row] + Math.fround(weight * cache.v[pos * kvRows + kvOffset + row]));
      }
    }
    const invDenom = denom > 0 ? Math.fround(1 / denom) : Math.fround(0);
    for (let row = 0; row < attentionHeadSize; row += 1) {
      attention[qOffset + row] = Math.fround(attention[qOffset + row] * invDenom);
    }
  }
  const attnProjected = tinyAddOptionalBias(source, specs, layerRole("self_attn.o_proj.bias"), tinyMatVec(source, oProj, attention));
  const afterAttention = embeddingValues.map((value, i) => Math.fround(value + attnProjected[i]));
  const finalHidden = tinyRmsNormVector(afterAttention, finalNormValues, epsilon);
  return {
    afterAttention,
    k,
    logits: tinyAddOptionalBias(source, specs, rootTensors.lmHeadBias, tinyMatVec(source, lmHead, finalHidden).slice(0, vocab)),
    v,
  };
}

function tinySilu(value) {
  return Math.fround(value / Math.fround(1 + Math.exp(-value)));
}

function tinyLlamaBlockProjectionStep(source, cache, token, position, epsilonOrOptions = 1e-5, maybeOptions = {}) {
  const options = tinyLlamaProjectionOptions(epsilonOrOptions, maybeOptions);
  const epsilon = options.epsilon;
  const layer = options.layer ?? 0;
  const layerRole = (suffix) => `model.layers.${layer}.${suffix}`;
  const modelResourceOptions = options.modelResourceOptions ?? options;
  const rootTensors = tinyLlamaRootTensorNames(options);
  const specs = new Map(tinyLlamaModelResourceSpecs(source, modelResourceOptions).map((spec) => [spec.tensor, spec]));
  const postNorm = specs.get(layerRole("post_attention_layernorm.weight"));
  const gateProj = specs.get(layerRole("mlp.gate_proj.weight"));
  const upProj = specs.get(layerRole("mlp.up_proj.weight"));
  const downProj = specs.get(layerRole("mlp.down_proj.weight"));
  const embedding = specs.get(rootTensors.embedding);
  const finalNorm = specs.get(rootTensors.norm);
  const lmHead = specs.get(rootTensors.lmHead);
  const attention = tinyLlamaAttentionProjectionStep(source, cache, token, position, options);
  const hidden = attention.afterAttention.length;
  const postNormValues = [];
  const finalNormValues = [];
  for (let col = 0; col < hidden; col += 1) {
    postNormValues.push(safetensorsF32(source, postNorm, col));
    finalNormValues.push(safetensorsF32(source, finalNorm, col));
  }
  const normedFfn = tinyRmsNormVector(attention.afterAttention, postNormValues, epsilon);
  const gate = tinyAddOptionalBias(source, specs, layerRole("mlp.gate_proj.bias"), tinyMatVec(source, gateProj, normedFfn));
  const up = tinyAddOptionalBias(source, specs, layerRole("mlp.up_proj.bias"), tinyMatVec(source, upProj, normedFfn));
  const ffnInput = gate.map((value, i) => Math.fround(tinySilu(value) * up[i]));
  const ffn = tinyAddOptionalBias(source, specs, layerRole("mlp.down_proj.bias"), tinyMatVec(source, downProj, ffnInput));
  const blockHidden = attention.afterAttention.map((value, i) => Math.fround(value + ffn[i]));
  const finalHidden = tinyRmsNormVector(blockHidden, finalNormValues, epsilon);
  return {
    blockHidden,
    k: attention.k,
    logits: tinyAddOptionalBias(source, specs, rootTensors.lmHeadBias, tinyMatVec(source, lmHead, finalHidden).slice(0, embedding.shape[0])),
    v: attention.v,
  };
}

function expectByteArray(actual, expected, label) {
  if (actual.byteLength !== expected.byteLength) {
    throw new Error(`${label} byte length mismatch: ${actual.byteLength} != ${expected.byteLength}`);
  }
  for (let i = 0; i < actual.byteLength; i += 1) {
    if (actual[i] !== expected[i]) {
      throw new Error(`${label} byte ${i}: expected ${expected[i]}, got ${actual[i]}`);
    }
  }
}

async function expectTinyLlamaModelResourceBytes(device, resources, source, label) {
  const dataStart = safetensorsDataStart(source);
  for (const resource of resources) {
    const expected = source.subarray(
      dataStart + resource.dataByteOffset,
      dataStart + resource.dataByteOffset + resource.byteLength,
    );
    const actual = await device.readBytes(resource.descriptor, resource.byteLength);
    expectByteArray(actual, expected, `${label} ${resource.role}`);
  }
}

let exportsRef;
let hostDevice;
let hostRuntime;

function abiStructSize(kind, label) {
  const size = Number(exportsRef.zgml_abi_struct_size(kind));
  if (size === 0) throw new Error(`unknown ABI struct size for ${label}`);
  return size;
}

function initAbiStructSizes() {
  runtimeInfoSize = abiStructSize(abiStructRuntimeInfo, "runtime info");
  modelDescSize = abiStructSize(abiStructModelDesc, "model desc");
  modelInspectionSize = abiStructSize(abiStructModelInspection, "model inspection");
  compileDescSize = abiStructSize(abiStructCompileDesc, "compile desc");
  bindDescSize = abiStructSize(abiStructBindDesc, "bind desc");
  const bufferBindDescSize = abiStructSize(abiStructBufferBindDesc, "buffer bind desc");
  if (bufferBindDescSize !== bindDescSize) throw new Error("unexpected distinct Wasm bind descriptor sizes");
  bufferDescSize = abiStructSize(abiStructBufferDesc, "buffer desc");
  bufferInspectionSize = abiStructSize(abiStructBufferInspection, "buffer inspection");
  externalResourceDescSize = abiStructSize(abiStructExternalResourceDesc, "external resource desc");
  llamaKvCacheBindDescSize = abiStructSize(abiStructLlamaKvCacheBindDesc, "LLaMA KV cache bind desc");
  llamaBufferBindDescSize = abiStructSize(abiStructLlamaBufferBindDesc, "LLaMA buffer bind desc");
  stepDescSize = abiStructSize(abiStructStepDesc, "step desc");
  stepResultSize = abiStructSize(abiStructStepResult, "step result");
  tokenStepDescSize = abiStructSize(abiStructTokenStepDesc, "token step desc");
  tokenAdvanceDescSize = abiStructSize(abiStructTokenAdvanceDesc, "token advance desc");
  tokenAdvanceTokensDescSize = abiStructSize(abiStructTokenAdvanceTokensDesc, "token advance tokens desc");
  tokenPrefillDescSize = abiStructSize(abiStructTokenPrefillDesc, "token prefill desc");
  tokenExecuteDescSize = abiStructSize(abiStructTokenExecuteDesc, "token execute desc");
  tokenArgmaxDescSize = abiStructSize(abiStructTokenArgmaxDesc, "token argmax desc");
  tokenArgmaxResultSize = abiStructSize(abiStructTokenArgmaxResult, "token argmax result");
  tokenExecuteArgmaxDescSize = abiStructSize(abiStructTokenExecuteArgmaxDesc, "token execute argmax desc");
  tokenGenerateArgmaxDescSize = abiStructSize(abiStructTokenGenerateArgmaxDesc, "token generate argmax desc");
  tokenGenerateArgmaxResultSize = abiStructSize(abiStructTokenGenerateArgmaxResult, "token generate argmax result");
  tokenSampleDescSize = abiStructSize(abiStructTokenSampleDesc, "token sample desc");
  tokenSampleResultSize = abiStructSize(abiStructTokenSampleResult, "token sample result");
  tokenExecuteSampleDescSize = abiStructSize(abiStructTokenExecuteSampleDesc, "token execute sample desc");
  tokenGenerateSampleDescSize = abiStructSize(abiStructTokenGenerateSampleDesc, "token generate sample desc");
  tokenGenerateSampleResultSize = abiStructSize(abiStructTokenGenerateSampleResult, "token generate sample result");
  deviceBufferImportDescSize = abiStructSize(abiStructDeviceBufferImportDesc, "device-buffer import desc");
  programRequirementsSize = abiStructSize(abiStructProgramRequirements, "program requirements");
  programModelCompatibilitySize = abiStructSize(abiStructProgramModelCompatibility, "program model compatibility");
  sessionInspectionSize = abiStructSize(abiStructSessionInspection, "session inspection");
  programInspectionSize = abiStructSize(abiStructProgramInspection, "program inspection");
  llamaProgramInspectionSize = abiStructSize(abiStructLlamaProgramInspection, "LLaMA program inspection");
  llamaKvCacheRequirementsSize = abiStructSize(abiStructLlamaKvCacheRequirements, "LLaMA KV cache requirements");
  runtimeProfileSize = abiStructSize(abiStructRuntimeProfile, "runtime profile");
  safetensorsHeaderProbeDescSize = abiStructSize(abiStructSafetensorsHeaderProbeDesc, "safetensors header probe desc");
  safetensorsDataLoadDescSize = abiStructSize(abiStructSafetensorsDataLoadDesc, "safetensors data load desc");
  if (exportsRef.zgml_abi_struct_size(0) !== 0) throw new Error("unknown ABI struct kind returned a size");
}

function append(message) {
  logEl.textContent = `${logEl.textContent === "running..." ? "" : logEl.textContent}${message}\n`;
  document.documentElement.dataset.zgmlWasmSmokeLastMessage = message;
  console.log(message);
}

function markBrowserSmokeStage(stage) {
  document.documentElement.dataset.zgmlWasmSmokeStage = stage;
  console.log(`zgml browser smoke stage ${stage}`);
}

const browserSmokeParams = new URLSearchParams(globalThis.location?.search ?? "");
const browserLlamaProfileLabelFilter = new Set(
  browserSmokeParams.getAll("llamaProfileLabel")
    .flatMap((value) => value.split(","))
    .map((value) => value.trim())
    .filter(Boolean),
);
document.documentElement.dataset.zgmlWasmLlamaProfileFilter = [...browserLlamaProfileLabelFilter].join(",");

const browserFocusedLlamaProfileLabels = browserLlamaProfileLabelFilter.size !== 0;

function shouldRunPackedLlamaProfile(label) {
  return (
    !browserFocusedLlamaProfileLabels ||
    browserLlamaProfileLabelFilter.has(label) ||
    browserLlamaProfileLabelFilter.has(`greedy-${label}`)
  );
}

const browserLlamaProfileEvidence = {
  adapterLimitStorageCallCount: 0,
  backendDispatchCount: 0,
  commandCount: 0,
  executorBackendDispatchCount: 0,
  fallbackOpCount: 0,
  genericDispatchCount: 0,
  gpuStorageCallCount: 0,
  labels: [],
  maxRequiredStorageBufferCount: 0,
  mockStorageCallCount: 0,
  outputReadCount: 0,
  profileCount: 0,
  scalarDispatchCount: 0,
  selectionBackendDispatchCount: 0,
  selectionReadCount: 0,
  storageCallCount: 0,
  storageLabels: [],
  syncCount: 0,
  unknownStorageCallCount: 0,
  windowDispatchCount: 0,
};

function writeBrowserLlamaProfileEvidence() {
  document.documentElement.dataset.zgmlWasmLlamaStorageCallCount = String(browserLlamaProfileEvidence.storageCallCount);
  document.documentElement.dataset.zgmlWasmLlamaGpuStorageCalls = String(browserLlamaProfileEvidence.gpuStorageCallCount);
  document.documentElement.dataset.zgmlWasmLlamaMockStorageCalls = String(browserLlamaProfileEvidence.mockStorageCallCount);
  document.documentElement.dataset.zgmlWasmLlamaAdapterLimitStorageCalls = String(browserLlamaProfileEvidence.adapterLimitStorageCallCount);
  document.documentElement.dataset.zgmlWasmLlamaUnknownStorageCalls = String(browserLlamaProfileEvidence.unknownStorageCallCount);
  document.documentElement.dataset.zgmlWasmLlamaMaxRequiredStorageBuffers = String(browserLlamaProfileEvidence.maxRequiredStorageBufferCount);
  document.documentElement.dataset.zgmlWasmLlamaStorageLabels = browserLlamaProfileEvidence.storageLabels.join(",");
  document.documentElement.dataset.zgmlWasmLlamaProfileCount = String(browserLlamaProfileEvidence.profileCount);
  document.documentElement.dataset.zgmlWasmLlamaBackendDispatches = String(browserLlamaProfileEvidence.backendDispatchCount);
  document.documentElement.dataset.zgmlWasmLlamaExecutorBackendDispatches = String(browserLlamaProfileEvidence.executorBackendDispatchCount);
  document.documentElement.dataset.zgmlWasmLlamaSelectionBackendDispatches = String(browserLlamaProfileEvidence.selectionBackendDispatchCount);
  document.documentElement.dataset.zgmlWasmLlamaFallbackOps = String(browserLlamaProfileEvidence.fallbackOpCount);
  document.documentElement.dataset.zgmlWasmLlamaCommandCount = String(browserLlamaProfileEvidence.commandCount);
  document.documentElement.dataset.zgmlWasmLlamaGenericDispatches = String(browserLlamaProfileEvidence.genericDispatchCount);
  document.documentElement.dataset.zgmlWasmLlamaScalarDispatches = String(browserLlamaProfileEvidence.scalarDispatchCount);
  document.documentElement.dataset.zgmlWasmLlamaWindowDispatches = String(browserLlamaProfileEvidence.windowDispatchCount);
  document.documentElement.dataset.zgmlWasmLlamaOutputReads = String(browserLlamaProfileEvidence.outputReadCount);
  document.documentElement.dataset.zgmlWasmLlamaSelectionReads = String(browserLlamaProfileEvidence.selectionReadCount);
  document.documentElement.dataset.zgmlWasmLlamaSyncs = String(browserLlamaProfileEvidence.syncCount);
  document.documentElement.dataset.zgmlWasmLlamaProfileLabels = browserLlamaProfileEvidence.labels.join(",");
}

function recordBrowserLlamaStorageCallEvidence(label, call) {
  if (
    call === null ||
    call === undefined ||
    typeof call.usesGpuStorageBuffers !== "boolean" ||
    !Number.isSafeInteger(call.requiredStorageBufferCount)
  ) {
    return;
  }
  browserLlamaProfileEvidence.storageCallCount += 1;
  browserLlamaProfileEvidence.maxRequiredStorageBufferCount = Math.max(
    browserLlamaProfileEvidence.maxRequiredStorageBufferCount,
    call.requiredStorageBufferCount,
  );
  if (call.usesGpuStorageBuffers === true) {
    browserLlamaProfileEvidence.gpuStorageCallCount += 1;
  } else if (call.deviceMode === "mock") {
    browserLlamaProfileEvidence.mockStorageCallCount += 1;
  } else if (call.deviceMode === "gpu-buffer") {
    browserLlamaProfileEvidence.adapterLimitStorageCallCount += 1;
  } else {
    browserLlamaProfileEvidence.unknownStorageCallCount += 1;
  }
  browserLlamaProfileEvidence.storageLabels.push(label);
  writeBrowserLlamaProfileEvidence();
}

function recordBrowserLlamaPipelineCallEvidence(calls, label, call) {
  calls.push(call);
  recordBrowserLlamaStorageCallEvidence(label, call);
}

function recordBrowserLlamaProfileEvidence(label, profile) {
  browserLlamaProfileEvidence.profileCount += 1;
  const executorBackendDispatchCount = Number(profile.executorBackendDispatchCount ?? 0);
  const selectionBackendDispatchCount = Number(profile.selectionBackendDispatchCount ?? 0);
  browserLlamaProfileEvidence.backendDispatchCount += executorBackendDispatchCount + selectionBackendDispatchCount;
  browserLlamaProfileEvidence.executorBackendDispatchCount += executorBackendDispatchCount;
  browserLlamaProfileEvidence.fallbackOpCount += Number(profile.executorFallbackOpCount ?? 0) + Number(profile.selectionFallbackOpCount ?? 0);
  browserLlamaProfileEvidence.commandCount +=
    Number(profile.executorCommandCount ?? 0) +
    Number(profile.selectionBackendDispatchCount ?? 0) +
    Number(profile.selectionFallbackOpCount ?? 0);
  browserLlamaProfileEvidence.genericDispatchCount += Number(profile.executorGenericDispatchCount ?? 0);
  browserLlamaProfileEvidence.outputReadCount += Number(profile.outputReadCount ?? 0);
  browserLlamaProfileEvidence.scalarDispatchCount += Number(profile.executorScalarDispatchCount ?? 0);
  browserLlamaProfileEvidence.selectionBackendDispatchCount += selectionBackendDispatchCount;
  browserLlamaProfileEvidence.selectionReadCount += Number(profile.selectionResultReadCount ?? 0);
  browserLlamaProfileEvidence.syncCount += Number(profile.syncCount ?? 0) + Number(profile.selectionSyncCount ?? 0);
  browserLlamaProfileEvidence.windowDispatchCount += Number(profile.executorWindowDispatchCount ?? 0);
  browserLlamaProfileEvidence.labels.push(label);
  document.documentElement.dataset.zgmlWasmLlamaLastProfileLabel = label;
  writeBrowserLlamaProfileEvidence();
  console.log(
    `zgml llama profile ${label}: backendDispatches=${profile.executorBackendDispatchCount ?? 0}` +
    ` selectionDispatches=${profile.selectionBackendDispatchCount ?? 0}` +
    ` fallbackOps=${profile.executorFallbackOpCount ?? 0}` +
    ` familyDispatches=${profile.executorGenericDispatchCount ?? 0}/${profile.executorScalarDispatchCount ?? 0}/${profile.executorWindowDispatchCount ?? 0}` +
    ` outputReads=${profile.outputReadCount ?? 0}` +
    ` selectionReads=${profile.selectionResultReadCount ?? 0}`,
  );
}

function appendBrowserLlamaProfileEvidence() {
  writeBrowserLlamaProfileEvidence();
  append(
    `zgml browser wasm LLaMA profile evidence: profiles=${browserLlamaProfileEvidence.profileCount}` +
    ` backendDispatches=${browserLlamaProfileEvidence.backendDispatchCount}` +
    ` executorDispatches=${browserLlamaProfileEvidence.executorBackendDispatchCount}` +
    ` selectionDispatches=${browserLlamaProfileEvidence.selectionBackendDispatchCount}` +
    ` fallbackOps=${browserLlamaProfileEvidence.fallbackOpCount}` +
    ` commandCount=${browserLlamaProfileEvidence.commandCount}` +
    ` familyDispatches=${browserLlamaProfileEvidence.genericDispatchCount}/${browserLlamaProfileEvidence.scalarDispatchCount}/${browserLlamaProfileEvidence.windowDispatchCount}` +
    ` storageCalls=${browserLlamaProfileEvidence.storageCallCount}/${browserLlamaProfileEvidence.gpuStorageCallCount}/${browserLlamaProfileEvidence.mockStorageCallCount}/${browserLlamaProfileEvidence.adapterLimitStorageCallCount}/${browserLlamaProfileEvidence.unknownStorageCallCount}` +
    ` outputReads=${browserLlamaProfileEvidence.outputReadCount}` +
    ` selectionReads=${browserLlamaProfileEvidence.selectionReadCount}` +
    ` syncs=${browserLlamaProfileEvidence.syncCount}`,
  );
}

async function requestOptionalBrowserGpuDeviceInfo() {
  try {
    return await requestBrowserWebGpuDeviceInfo();
  } catch (err) {
    return {
      adapterFeatures: [],
      adapterLimits: {},
      available: false,
      device: null,
      deviceFeatures: [],
      deviceLimits: {},
      reason: err && err.message ? err.message : String(err),
    };
  }
}

function recordBrowserWebGpuEvidence(info, device) {
  const deviceLimits = info.deviceLimits ?? {};
  const adapterLimits = info.adapterLimits ?? {};
  const maxStorageBuffers = device.maxStorageBuffersPerShaderStage();
  const alignment = device.storageBufferOffsetAlignment();
  document.documentElement.dataset.zgmlWasmGpuResources = device.mode;
  document.documentElement.dataset.zgmlWasmGpuAvailable = info.available ? "true" : "false";
  document.documentElement.dataset.zgmlWasmGpuReason = info.reason ?? "";
  document.documentElement.dataset.zgmlWasmGpuMaxStorageBuffers = String(maxStorageBuffers);
  document.documentElement.dataset.zgmlWasmGpuStorageAlignment = String(alignment);
  document.documentElement.dataset.zgmlWasmGpuCanBindBlockPipeline = device.canBindStorageBuffers(6) ? "true" : "false";
  const source = info.available ? "device" : "fallback";
  append(
    `zgml browser wasm host resources: ${device.mode}` +
    ` available=${info.available ? "yes" : "no"}` +
    ` reason=${info.reason ?? ""}` +
    ` maxStorageBuffers=${maxStorageBuffers}` +
    ` storageAlignment=${alignment}` +
    ` source=${source}`,
  );
  if (info.available) {
    append(
      `zgml browser wasm WebGPU device limits: maxStorageBufferBindingSize=${deviceLimits.maxStorageBufferBindingSize ?? "unknown"}` +
      ` maxBufferSize=${deviceLimits.maxBufferSize ?? "unknown"}` +
      ` features=${(info.deviceFeatures ?? []).join(",") || "none"}`,
    );
  } else if (Object.keys(adapterLimits).length !== 0) {
    append(
      `zgml browser wasm WebGPU adapter limits before device failure: maxStorageBuffers=${adapterLimits.maxStorageBuffersPerShaderStage ?? "unknown"}` +
      ` storageAlignment=${adapterLimits.minStorageBufferOffsetAlignment ?? "unknown"}`,
    );
  }
}

function makeZgmlHostImports() {
  return {
    zgml_wasm_host_session_step(sessionHandle, descPtr = 0, resultPtr = 0) {
      return hostRuntime ? hostRuntime.stepSync(sessionHandle, descPtr, resultPtr) : unsupported;
    },
    zgml_wasm_host_session_step_no_output(sessionHandle, descPtr = 0, resultPtr = 0) {
      return hostRuntime ? hostRuntime.stepNoOutputSync(sessionHandle, descPtr, resultPtr) : unsupported;
    },
    zgml_wasm_host_session_execute_tokens(sessionHandle, descPtr = 0, resultPtr = 0) {
      return hostRuntime ? hostRuntime.executeTokensSync(sessionHandle, descPtr, resultPtr) : unsupported;
    },
    zgml_wasm_host_session_position(sessionHandle, outPositionPtr = 0) {
      return hostRuntime ? hostRuntime.positionSync(sessionHandle, outPositionPtr) : unsupported;
    },
    zgml_wasm_host_session_runtime_profile(sessionHandle, outProfilePtr = 0) {
      return hostRuntime ? hostRuntime.runtimeProfileSync(sessionHandle, outProfilePtr) : unsupported;
    },
    zgml_wasm_host_session_reset(sessionHandle) {
      return hostRuntime ? hostRuntime.resetSync(sessionHandle) : unsupported;
    },
    zgml_wasm_host_session_reset_runtime_profile(sessionHandle) {
      return hostRuntime ? hostRuntime.resetRuntimeProfileSync(sessionHandle) : unsupported;
    },
    zgml_wasm_host_session_free(sessionHandle) {
      return hostRuntime ? hostRuntime.releaseHostSession(sessionHandle) : unsupported;
    },
  };
}

function memory() {
  return exportsRef.memory;
}

function view() {
  return new DataView(memory().buffer);
}

function writeU32(ptr, offset, value) {
  view().setUint32(ptr + offset, value, true);
}

function zero(ptr, len) {
  new Uint8Array(memory().buffer, ptr, len).fill(0);
}

function writeBytes(ptr, bytes) {
  new Uint8Array(memory().buffer, ptr, bytes.length).set(bytes);
}

function makeWasiShim() {
  const errnoSuccess = 0;
  const errnoBadf = 8;
  const textDecoder = new TextDecoder();

  function unsupported() {
    return errnoBadf;
  }

  return {
    random_get(ptr, len) {
      crypto.getRandomValues(new Uint8Array(memory().buffer, ptr, len));
      return errnoSuccess;
    },
    clock_time_get(_, __, outPtr) {
      view().setBigUint64(outPtr, BigInt(Math.trunc(performance.now() * 1_000_000)), true);
      return errnoSuccess;
    },
    clock_res_get(_, outPtr) {
      view().setBigUint64(outPtr, 1_000_000n, true);
      return errnoSuccess;
    },
    environ_sizes_get(countPtr, sizePtr) {
      writeU32(countPtr, 0, 0);
      writeU32(sizePtr, 0, 0);
      return errnoSuccess;
    },
    environ_get() {
      return errnoSuccess;
    },
    fd_fdstat_get(fd, outPtr) {
      if (fd > 2) return errnoBadf;
      zero(outPtr, 24);
      return errnoSuccess;
    },
    fd_write(fd, iovsPtr, iovsLen, nwrittenPtr) {
      if (fd > 2) return errnoBadf;
      let written = 0;
      let text = "";
      for (let i = 0; i < iovsLen; i += 1) {
        const base = view().getUint32(iovsPtr + i * 8, true);
        const len = view().getUint32(iovsPtr + i * 8 + 4, true);
        written += len;
        text += textDecoder.decode(new Uint8Array(memory().buffer, base, len));
      }
      if (text) append(text.trimEnd());
      writeU32(nwrittenPtr, 0, written);
      return errnoSuccess;
    },
    poll_oneoff(_, __, ___, nsubscriptionsPtr) {
      writeU32(nsubscriptionsPtr, 0, 0);
      return errnoSuccess;
    },
    fd_read: unsupported,
    fd_pread: unsupported,
    fd_pwrite: unsupported,
    fd_close: unsupported,
    fd_seek: unsupported,
    fd_sync: unsupported,
    fd_filestat_get: unsupported,
    fd_filestat_set_size: unsupported,
    fd_filestat_set_times: unsupported,
    fd_readdir: unsupported,
    path_open: unsupported,
    path_link: unsupported,
    path_readlink: unsupported,
    path_symlink: unsupported,
    path_rename: unsupported,
    path_remove_directory: unsupported,
    path_unlink_file: unsupported,
    path_filestat_get: unsupported,
    path_create_directory: unsupported,
  };
}

function alloc(size) {
  const ptr = exportsRef.zgml_wasm_alloc(size);
  if (ptr === 0) throw new Error(`zgml_wasm_alloc(${size}) returned null`);
  return ptr;
}

function free(ptr, size) {
  if (ptr !== 0) exportsRef.zgml_wasm_free(ptr, size);
}

function check(code) {
  if (code !== ok) throw new Error(`zgml status ${code}`);
}

function expectStatus(actual, expected, message) {
  if (actual !== expected) throw new Error(`${message}: expected status ${expected}, got ${actual}`);
}

function u32(ptr, offset = 0) {
  return view().getUint32(ptr + offset, true);
}

function u64(ptr, offset = 0) {
  return view().getBigUint64(ptr + offset, true);
}

function f64(ptr, offset = 0) {
  return view().getFloat64(ptr + offset, true);
}

function writeF32Array(ptr, values) {
  new Float32Array(memory().buffer, ptr, values.length).set(values);
}

function writeU32Array(ptr, values) {
  new Uint32Array(memory().buffer, ptr, values.length).set(values);
}

function readF32Array(ptr, len) {
  return Array.from(new Float32Array(memory().buffer, ptr, len));
}

function expectClose(actual, expected, tolerance = 1e-5) {
  if (actual.length !== expected.length) {
    throw new Error(`expected ${expected.length} outputs, got ${actual.length}`);
  }
  for (let i = 0; i < actual.length; i += 1) {
    if (Math.abs(actual[i] - expected[i]) > tolerance) {
      throw new Error(`output[${i}] expected ${expected[i]}, got ${actual[i]}`);
    }
  }
}

function argmaxOf(values) {
  if (values.length === 0) throw new Error("cannot select argmax from empty values");
  let token = 0;
  let logit = values[0];
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > logit) {
      token = i;
      logit = values[i];
    }
  }
  return { token, logit };
}

function sampleRandomUnit(seed) {
  let x = (seed === 0 ? 0x9e3779b9 : seed) >>> 0;
  x = (x ^ ((x << 13) >>> 0)) >>> 0;
  x = (x ^ (x >>> 17)) >>> 0;
  x = (x ^ ((x << 5) >>> 0)) >>> 0;
  return x / 4294967296.0;
}

function sampleTopKOf(values, options = {}) {
  const topK = options.topK ?? options.top_k ?? 40;
  const temperature = options.temperature ?? 1.0;
  const seed = options.seed ?? 0;
  if (!Number.isSafeInteger(topK) || topK <= 0 || topK > 256) throw new Error("invalid topK");
  if (!Number.isFinite(temperature) || temperature <= 0) throw new Error("invalid temperature");
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error("invalid seed");
  if (values.length === 0) throw new Error("cannot sample from empty values");
  const effectiveK = Math.min(topK, values.length);
  const candidates = [];
  for (let i = 0; i < values.length; i += 1) {
    const logit = Number(values[i]);
    if (!Number.isFinite(logit)) throw new Error(`logit ${i} is not finite`);
    const candidate = { token: i, logit };
    let index = candidates.length;
    while (index > 0 && candidate.logit > candidates[index - 1].logit) index -= 1;
    if (candidates.length < effectiveK) {
      candidates.splice(index, 0, candidate);
    } else if (logit > candidates[candidates.length - 1].logit) {
      candidates.pop();
      if (index > candidates.length) index = candidates.length;
      candidates.splice(index, 0, candidate);
    }
  }
  if (effectiveK === 1) return candidates[0];
  const maxLogit = candidates[0].logit;
  const weights = candidates.map((candidate) => Math.exp((candidate.logit - maxLogit) / temperature));
  let threshold = sampleRandomUnit(seed >>> 0) * weights.reduce((sum, weight) => sum + weight, 0);
  for (let i = 0; i < candidates.length; i += 1) {
    if (threshold < weights[i]) return candidates[i];
    threshold -= weights[i];
  }
  return candidates[candidates.length - 1];
}

function canUseGpuStorageBuffers(device, count) {
  return device.canCreateGpuBuffers() && device.canBindStorageBuffers(count);
}

function expectedLlamaExecutorDeviceMode(device) {
  return device.canCreateGpuBuffers() ? "gpu-buffer" : "mock";
}

const llamaBlockScalarWindowMaxContextLength = 64;

function canUseLlamaBlockWindowGpuPath(device, tokenCount, options = {}) {
  const contextLength = options.contextLength ?? llamaBlockScalarWindowMaxContextLength;
  const activeSeq = Math.min(contextLength, (options.startPosition ?? 0) + tokenCount);
  const slidingWindow = options.slidingWindow ?? 0;
  const scoreSlots = slidingWindow > 0 ? Math.min(activeSeq, slidingWindow) : activeSeq;
  return (
    canUseGpuStorageBuffers(device, 6) &&
    tokenCount > 1 &&
    scoreSlots <= llamaBlockScalarWindowMaxContextLength
  );
}

function expectedLlamaBlockNoOutputPrefillCommandCount(device, layers, tokenCount, options = {}) {
  return canUseLlamaBlockWindowGpuPath(device, tokenCount, options) ? layers : layers * tokenCount;
}

function expectedLlamaBlockPipelineCommandCount(device, layers, tokenCount, outputPolicy, options = {}) {
  const finalLogitsCommandCount = options.splitTerminalLogits && outputPolicy === executeOutputLogits ? 1 : 0;
  if (!canUseLlamaBlockWindowGpuPath(device, tokenCount, options)) {
    return layers * tokenCount + finalLogitsCommandCount;
  }
  if (outputPolicy === executeOutputNone) return layers;
  if (options.splitTerminalLogits) return layers + finalLogitsCommandCount;
  return Math.max(0, layers - 1) + (tokenCount > 2 ? 2 : tokenCount);
}

function expectedLlamaBlockPipelineDispatchFamilies(device, layers, tokenCount, outputPolicy, options = {}) {
  const counts = { generic: 0, scalar: 0, window: 0 };
  if (!canUseGpuStorageBuffers(device, 6)) return counts;
  const splitTerminalLogits = options.splitTerminalLogits && outputPolicy === executeOutputLogits;
  if (canUseLlamaBlockWindowGpuPath(device, tokenCount, options)) {
    if (outputPolicy === executeOutputNone || splitTerminalLogits) {
      counts.window = layers;
      return counts;
    }
    if (tokenCount > 2) {
      counts.window = layers;
      const hiddenSize = options.hiddenSize ?? 64;
      const scalarMinHiddenSize = options.scalarProjectionMinHiddenSize ?? 64;
      const contextLength = options.contextLength ?? llamaBlockScalarWindowMaxContextLength;
      const activeSeq = Math.min(contextLength, (options.startPosition ?? 0) + tokenCount);
      const slidingWindow = options.slidingWindow ?? 0;
      const scoreSlots = slidingWindow > 0 ? Math.min(activeSeq, slidingWindow) : activeSeq;
      if (hiddenSize >= scalarMinHiddenSize && scoreSlots <= llamaBlockScalarWindowMaxContextLength) {
        counts.scalar = 1;
      } else {
        counts.generic = 1;
      }
      return counts;
    }
  }
  const hiddenSize = options.hiddenSize ?? 64;
  const scalarMinHiddenSize = options.scalarProjectionMinHiddenSize ?? 64;
  const contextLength = options.contextLength ?? llamaBlockScalarWindowMaxContextLength;
  const activeSeq = Math.min(contextLength, (options.startPosition ?? 0) + tokenCount);
  const slidingWindow = options.slidingWindow ?? 0;
  const scoreSlots = slidingWindow > 0 ? Math.min(activeSeq, slidingWindow) : activeSeq;
  if (tokenCount === 1 && hiddenSize >= scalarMinHiddenSize && scoreSlots <= llamaBlockScalarWindowMaxContextLength) {
    counts.scalar = layers;
  } else {
    counts.generic = layers * tokenCount;
  }
  return counts;
}

function checkRuntimeInfo() {
  const info = keep(alloc(runtimeInfoSize), runtimeInfoSize);
  zero(info, runtimeInfoSize);
  check(exportsRef.zgml_get_runtime_info(info));
  const featureFlags = u64(info, 16);
  const features = runtimeFeatures(featureFlags);
  if (
    u32(info, 0) !== expectedAbiVersion ||
    u32(info, 4) !== 4 ||
    u32(info, 8) !== 4 ||
    u32(info, 12) !== 4 ||
    !features.bufferHandle ||
    !features.modelAuto ||
    !features.runtimeProfile ||
    !features.webGpuCompileOnly ||
    !features.wasmExports ||
    !features.nativeBufferIo ||
    !features.nativeArgmax ||
    !features.nativeExecuteArgmax ||
    !features.nativeTopKSample ||
    !features.programRequirements ||
    !features.programOutputBuffer ||
    !features.nativeGenerateSample ||
    !features.nativeGenerateArgmax ||
    !features.sessionModelBinding ||
    !features.externalBuffer ||
    !features.externalResourceBuffer ||
    !features.llamaKvResourceBinding ||
    !features.llamaKvCacheRequirements ||
    !features.programBufferFactory ||
    !features.externalResourceAccess ||
    !features.modelInspection ||
    !features.programModelCompatibility ||
    !features.bufferInspection ||
    !features.sessionInspection ||
    !features.programResourceInspection ||
    !features.programMemoryInspection ||
    !features.programShapeInspection ||
    !features.programPatchEnvelopeInspection ||
    !features.sessionBindingShapeInspection ||
    !features.programDeviceBuffer ||
    !features.programDeviceBufferImport ||
    !features.programDispatchPlanInspection ||
    !features.abiStructSize ||
    !features.modelPathProbe ||
    !features.supportedCheckpoints ||
    !features.safetensorsHeaderProbe ||
    !features.safetensorsDataLoad ||
    !features.safetensorsDataProbe ||
    !features.nativeTinyMlp ||
    features.nativeWgpuExecution ||
    features.experimentalLlamaWgpuExecution
  ) {
    throw new Error("unexpected zgml browser wasm runtime info");
  }
}

function runtimeFeatures(featureFlags) {
  return {
    bufferHandle: (featureFlags & featureBufferHandle) !== 0n,
    modelAuto: (featureFlags & featureModelAuto) !== 0n,
    runtimeProfile: (featureFlags & featureRuntimeProfile) !== 0n,
    webGpuCompileOnly: (featureFlags & featureWebGpuCompileOnly) !== 0n,
    wasmExports: (featureFlags & featureWasmExports) !== 0n,
    nativeBufferIo: (featureFlags & featureNativeBufferIo) !== 0n,
    nativeArgmax: (featureFlags & featureNativeArgmax) !== 0n,
    nativeExecuteArgmax: (featureFlags & featureNativeExecuteArgmax) !== 0n,
    nativeTopKSample: (featureFlags & featureNativeTopKSample) !== 0n,
    programRequirements: (featureFlags & featureProgramRequirements) !== 0n,
    programOutputBuffer: (featureFlags & featureProgramOutputBuffer) !== 0n,
    nativeGenerateSample: (featureFlags & featureNativeGenerateSample) !== 0n,
    nativeGenerateArgmax: (featureFlags & featureNativeGenerateArgmax) !== 0n,
    sessionModelBinding: (featureFlags & featureSessionModelBinding) !== 0n,
    externalBuffer: (featureFlags & featureExternalBuffer) !== 0n,
    externalResourceBuffer: (featureFlags & featureExternalResourceBuffer) !== 0n,
    llamaKvResourceBinding: (featureFlags & featureLlamaKvResourceBinding) !== 0n,
    llamaKvCacheRequirements: (featureFlags & featureLlamaKvCacheRequirements) !== 0n,
    programBufferFactory: (featureFlags & featureProgramBufferFactory) !== 0n,
    externalResourceAccess: (featureFlags & featureExternalResourceAccess) !== 0n,
    modelInspection: (featureFlags & featureModelInspection) !== 0n,
    programModelCompatibility: (featureFlags & featureProgramModelCompatibility) !== 0n,
    bufferInspection: (featureFlags & featureBufferInspection) !== 0n,
    sessionInspection: (featureFlags & featureSessionInspection) !== 0n,
    programResourceInspection: (featureFlags & featureProgramResourceInspection) !== 0n,
    programMemoryInspection: (featureFlags & featureProgramMemoryInspection) !== 0n,
    programShapeInspection: (featureFlags & featureProgramShapeInspection) !== 0n,
    programPatchEnvelopeInspection: (featureFlags & featureProgramPatchEnvelopeInspection) !== 0n,
    sessionBindingShapeInspection: (featureFlags & featureSessionBindingShapeInspection) !== 0n,
    nativeWgpuExecution: (featureFlags & featureNativeWgpuExecution) !== 0n,
    programDeviceBuffer: (featureFlags & featureProgramDeviceBuffer) !== 0n,
    programDeviceBufferImport: (featureFlags & featureProgramDeviceBufferImport) !== 0n,
    programDispatchPlanInspection: (featureFlags & featureProgramDispatchPlanInspection) !== 0n,
    abiStructSize: (featureFlags & featureAbiStructSize) !== 0n,
    experimentalLlamaWgpuExecution: (featureFlags & featureExperimentalLlamaWgpuExecution) !== 0n,
    modelPathProbe: (featureFlags & featureModelPathProbe) !== 0n,
    supportedCheckpoints: (featureFlags & featureSupportedCheckpoints) !== 0n,
    safetensorsHeaderProbe: (featureFlags & featureSafetensorsHeaderProbe) !== 0n,
    safetensorsDataLoad: (featureFlags & featureSafetensorsDataLoad) !== 0n,
    safetensorsDataProbe: (featureFlags & featureSafetensorsDataProbe) !== 0n,
    nativeTinyMlp: (featureFlags & featureNativeTinyMlp) !== 0n,
  };
}

const allocations = [];
const buffers = [];
const hostResources = new HostResourceRegistry();
hostDevice = new WasmWebGpuDevice({ registry: hostResources });
const keep = (ptr, size) => {
  allocations.push([ptr, size]);
  return ptr;
};
const hostResourceBridge = new WasmExternalResourceBridge({
  exports: () => exportsRef,
  alloc,
  keep,
  view,
  zero,
  check,
  externalResourceDescSize: () => externalResourceDescSize,
  handleSize,
  trackBuffer: (handle) => buffers.push(handle),
});
const sessionBindingBridge = new WasmSessionBindingBridge({
  exports: () => exportsRef,
  alloc,
  keep,
  view,
  zero,
	  check,
	  bindDescSize: () => bindDescSize,
	  programModelCompatibilitySize: () => programModelCompatibilitySize,
	  programRequirementsSize: () => programRequirementsSize,
	  llamaKvCacheBindDescSize: () => llamaKvCacheBindDescSize,
  llamaBufferBindDescSize: () => llamaBufferBindDescSize,
  llamaKvCacheRequirementsSize: () => llamaKvCacheRequirementsSize,
  handleSize,
});

function createBuffer(byteLen) {
  const desc = keep(alloc(bufferDescSize), bufferDescSize);
  writeU32(desc, 0, byteLen);
  const out = keep(alloc(handleSize), handleSize);
  check(exportsRef.zgml_buffer_create(desc, out));
  const handle = u32(out);
  if (handle === 0) throw new Error("buffer handle was not returned");
  buffers.push(handle);
  return handle;
}

function wrapBuffer(dataPtr, byteLen) {
  const out = keep(alloc(handleSize), handleSize);
  check(exportsRef.zgml_buffer_wrap(dataPtr, byteLen, out));
  const handle = u32(out);
  if (handle === 0) throw new Error("wrapped buffer handle was not returned");
  buffers.push(handle);
  return handle;
}

function wrapResourceBuffer(placement, handleValue, byteOffset, byteLen, accessFlags = resourceAccessReadWrite) {
  return hostResourceBridge.wrapResourceBuffer({
    access: accessFlags,
    byteLength: byteLen,
    byteOffset,
    handle: handleValue,
    placement,
  });
}

function hostGpuBuffer(label, byteLength, usage) {
  return hostDevice.createBuffer({ label, byteLength, ...(usage === undefined ? {} : { usage }) });
}

function wrapHostResourceBuffer(placement, resource, options = {}) {
  return hostResourceBridge.wrapHostResourceBuffer(hostDevice, resource, {
    placement,
    ...options,
  });
}

function expectHostResourceBuffer(buffer, descriptor, label) {
  const inspection = inspectBuffer(buffer);
  if (
      u32(inspection, 0) !== bufferStorageExternalResource ||
      u32(inspection, 4) !== backendWebGpu ||
    u32(inspection, 8) !== externalResourceAccessFlags(descriptor.access) ||
    u64(inspection, 16) !== BigInt(descriptor.byteLength) ||
    u64(inspection, 24) !== BigInt(descriptor.handle) ||
    u64(inspection, 32) !== BigInt(descriptor.byteOffset) ||
    u64(inspection, 40) !== BigInt(descriptor.byteLength) ||
    descriptor.deviceHandle !== hostDevice.deviceHandle ||
    hostDevice.resourceFor(descriptor.handle) !== descriptor.resource
  ) {
    throw new Error(`unexpected ${label} host resource buffer inspection`);
  }
}

function expectHostResourceTable(descriptors, label) {
  const table = hostDevice.resourceTable(descriptors);
  const byteLength = descriptors.reduce((sum, descriptor) => sum + descriptor.byteLength, 0);
  if (
    table.deviceHandle !== hostDevice.deviceHandle ||
    table.resourceCount !== descriptors.length ||
    table.byteLength !== byteLength ||
    table.hash === 0n ||
    table.descriptorHash === 0n ||
    table.descriptors.length !== descriptors.length
  ) {
    throw new Error(`unexpected ${label} host resource table`);
  }
  if (
    !Object.isFrozen(table) ||
    !Object.isFrozen(table.descriptors) ||
    table.descriptors.some((entry) => !Object.isFrozen(entry))
  ) {
    throw new Error(`${label} host resource table evidence is not immutable`);
  }
  let mutationRejected = false;
  try {
    table.descriptors[0].byteLength = 0;
  } catch {
    mutationRejected = true;
  }
  if (!mutationRejected || table.descriptors[0].byteLength !== descriptors[0].byteLength) {
    throw new Error(`${label} host resource table descriptor mutation was not rejected`);
  }
  const stableTable = hostDevice.resourceTable(descriptors);
  if (stableTable.hash !== table.hash || stableTable.descriptorHash !== table.descriptorHash) {
    throw new Error(`unstable ${label} host resource table hash`);
  }
  let mixedDeviceRejected = false;
  try {
    hostDevice.resourceTable([
      descriptors[0],
      {
        ...descriptors[0],
        deviceHandle: hostDevice.deviceHandle + 1,
        handle: descriptors[0].handle + 0x100,
      },
    ]);
  } catch (err) {
    mixedDeviceRejected = String(err && err.message ? err.message : err).includes("device");
  }
  if (!mixedDeviceRejected) throw new Error(`expected ${label} mixed-device resource table to reject`);
}

function resourceTablePhysicalDescriptors(table) {
  return table.descriptors.map((entry) => ({
    access: entry.access,
    byteLength: entry.byteLength,
    byteOffset: entry.byteOffset,
    deviceHandle: entry.deviceHandle,
    handle: entry.handle,
    placement: entry.placement,
  }));
}

function expectHostResourceTableRoles(table, expectedRoles, label) {
  const actualRoles = table.descriptors.map((entry) => entry.role);
  if (
    actualRoles.length !== expectedRoles.length ||
    actualRoles.some((role, i) => role !== expectedRoles[i])
  ) {
    throw new Error(`${label} resource table roles mismatch: ${actualRoles.join(",")}`);
  }
  const physicalDescriptors = resourceTablePhysicalDescriptors(table);
  const rolelessTable = hostDevice.resourceTable(physicalDescriptors);
  if (rolelessTable.descriptorHash !== table.descriptorHash) {
    throw new Error(`${label} resource table descriptor hash changed without physical descriptor changes`);
  }
  const rotatedRoles = expectedRoles.length > 1
    ? expectedRoles.slice(1).concat(expectedRoles[0])
    : expectedRoles.map((role) => `${role}.changed`);
  const swappedRoleTable = hostDevice.resourceTable(physicalDescriptors.map((descriptor, i) => ({
    role: rotatedRoles[i],
    descriptor,
  })));
  if (swappedRoleTable.descriptorHash !== table.descriptorHash || swappedRoleTable.hash === table.hash) {
    throw new Error(`${label} resource table hash did not include semantic roles`);
  }
  if (physicalDescriptors.length > 0) {
    const lowByteRoleTable = hostDevice.resourceTable([{
      role: "semantic.\u0001",
      descriptor: physicalDescriptors[0],
    }]);
    const highByteRoleTable = hostDevice.resourceTable([{
      role: "semantic.\u0101",
      descriptor: physicalDescriptors[0],
    }]);
    if (
      lowByteRoleTable.descriptorHash !== highByteRoleTable.descriptorHash ||
      lowByteRoleTable.hash === highByteRoleTable.hash
    ) {
      throw new Error(`${label} resource table semantic hash collapsed distinct UTF-16 role code units`);
    }
  }
  if (expectedRoles.length > 1) {
    let duplicateRoleRejected = false;
    try {
      hostDevice.resourceTable(physicalDescriptors.map((descriptor) => ({
        role: expectedRoles[0],
        descriptor,
      })));
    } catch (err) {
      duplicateRoleRejected = String(err && err.message ? err.message : err).includes("duplicate role");
    }
    if (!duplicateRoleRejected) {
      throw new Error(`${label} resource table accepted duplicate semantic roles`);
    }
  }
}

function expectHostResourceUsageValidation() {
  const invalid = hostGpuBuffer("zgml.invalid-usage", 16, webgpuBufferUsage.COPY_SRC);
  let invalidRejected = false;
  try {
    wrapHostResourceBuffer(backendWebGpu, invalid, { access: "read" });
  } catch (err) {
    invalidRejected = String(err && err.message ? err.message : err).includes("usage");
  }
  if (!invalidRejected) throw new Error("expected host WebGPU resource usage validation to reject");

  const storageOnly = hostGpuBuffer("zgml.storage-only", 16, webgpuBufferUsage.STORAGE);
  const { buffer, descriptor } = wrapHostResourceBuffer(backendWebGpu, storageOnly, {
    access: "read",
    hostTransfer: false,
  });
  expectHostResourceBuffer(buffer, descriptor, "storage-only host resource");

  const callerOwnedResource = {
    destroyed: false,
    destroy() {
      this.destroyed = true;
    },
    label: "zgml.caller-owned-resource",
    size: 16,
    usage: webgpuBufferUsage.STORAGE | webgpuBufferUsage.COPY_DST,
  };
  let unregisteredCallerOwnedRejected = false;
  try {
    wrapHostResourceBuffer(backendWebGpu, callerOwnedResource, { access: "read" });
  } catch (err) {
    unregisteredCallerOwnedRejected = String(err && err.message ? err.message : err).includes("provenance");
  }
  if (!unregisteredCallerOwnedRejected) throw new Error("expected unregistered caller-owned WebGPU resource to reject");
  if (hostDevice.importBuffer(callerOwnedResource, { access: "read", byteLength: 16 }) !== callerOwnedResource) {
    throw new Error("imported caller-owned WebGPU resource did not preserve object identity");
  }
  const { buffer: importedBuffer, descriptor: importedDescriptor } = wrapHostResourceBuffer(backendWebGpu, callerOwnedResource, { access: "read" });
  expectHostResourceBuffer(importedBuffer, importedDescriptor, "imported caller-owned host resource");

  const otherHostDevice = new WasmWebGpuDevice();
  const foreignResource = otherHostDevice.createBuffer({
    label: "zgml.foreign-device-resource",
    byteLength: 16,
  });
  let foreignResourceRejected = false;
  try {
    wrapHostResourceBuffer(backendWebGpu, foreignResource, { access: "read" });
  } catch (err) {
    foreignResourceRejected = String(err && err.message ? err.message : err).includes("device");
  } finally {
    otherHostDevice.destroy();
  }
  if (!foreignResourceRejected) throw new Error("expected host WebGPU foreign-device resource validation to reject");

  const directResourceOffset = hostDevice.storageBufferOffsetAlignment();
  let unalignedDirectViewRejected = false;
  try {
    const unaligned = hostGpuBuffer("zgml.unaligned-direct-view", directResourceOffset + 16);
    wrapHostResourceBuffer(backendWebGpu, unaligned, {
      access: "read",
      byteLength: 16,
      byteOffset: Float32Array.BYTES_PER_ELEMENT,
    });
  } catch (err) {
    unalignedDirectViewRejected = String(err && err.message ? err.message : err).includes("alignment");
  }
  if (!unalignedDirectViewRejected) throw new Error("expected direct host WebGPU resource unaligned view validation to reject");

  let oversizedDirectViewRejected = false;
  try {
    const short = hostGpuBuffer("zgml.oversized-direct-view", directResourceOffset + 12);
    wrapHostResourceBuffer(backendWebGpu, short, {
      access: "read",
      byteLength: 16,
      byteOffset: directResourceOffset,
    });
  } catch (err) {
    oversizedDirectViewRejected = String(err && err.message ? err.message : err).includes("range");
  }
  if (!oversizedDirectViewRejected) throw new Error("expected direct host WebGPU resource oversized view validation to reject");

  const wrongDevice = {
    [wasmWebgpuInterop.deviceHandle]: hostDevice.deviceHandle + 1,
    deviceHandle: hostDevice.deviceHandle + 1,
    label: "zgml.wrong-device",
    size: 16,
    usage: webgpuBufferUsage.STORAGE | webgpuBufferUsage.COPY_DST,
  };
  let wrongDeviceRejected = false;
  try {
    wrapHostResourceBuffer(backendWebGpu, wrongDevice, { access: "read" });
  } catch (err) {
    wrongDeviceRejected = String(err && err.message ? err.message : err).includes("device");
  }
  if (!wrongDeviceRejected) throw new Error("expected host WebGPU resource device provenance validation to reject");
}

function createProgramBuffer(program, kind) {
  const out = keep(alloc(handleSize), handleSize);
  check(exportsRef.zgml_program_create_buffer(program, kind, out));
  const handle = u32(out);
  if (handle === 0) throw new Error("program buffer handle was not returned");
  buffers.push(handle);
  return handle;
}

function expectPortableDeviceBufferApisUnsupported(program, label) {
  const outBuffer = keep(alloc(handleSize), handleSize);
  writeU32(outBuffer, 0, 0xffffffff);
  expectStatus(exportsRef.zgml_program_create_device_buffer(program, programBufferWeights, backendCpu, outBuffer), unsupported, `${label} CPU device-buffer factory`);
  if (u32(outBuffer) !== 0) throw new Error(`${label} CPU device-buffer factory returned a handle`);

  writeU32(outBuffer, 0, 0xffffffff);
  expectStatus(exportsRef.zgml_program_create_device_buffer(program, programBufferWeights, backendWebGpu, outBuffer), unsupported, `${label} WebGPU device-buffer factory`);
  if (u32(outBuffer) !== 0) throw new Error(`${label} WebGPU device-buffer factory returned a handle`);

  const outDevice = keep(alloc(4), 4);
  writeU32(outDevice, 0, 0xffffffff);
  expectStatus(exportsRef.zgml_program_get_device_handle(program, backendWebGpu, outDevice), unsupported, `${label} WebGPU device handle`);
  if (u32(outDevice) !== 0) throw new Error(`${label} WebGPU device handle was not cleared`);

  const importDesc = keep(alloc(deviceBufferImportDescSize), deviceBufferImportDescSize);
  zero(importDesc, deviceBufferImportDescSize);
  writeU32(importDesc, 0, backendWebGpu);
  writeU32(importDesc, 8, 77);
  writeU32(importDesc, 12, 88);
  writeU32(outBuffer, 0, 0xffffffff);
  expectStatus(exportsRef.zgml_program_import_device_buffer(program, programBufferWeights, importDesc, outBuffer), unsupported, `${label} WebGPU device-buffer import`);
  if (u32(outBuffer) !== 0) throw new Error(`${label} WebGPU device-buffer import returned a handle`);
}

function bufferSize(buffer) {
  return exportsRef.zgml_buffer_size(buffer);
}

function inspectBuffer(buffer) {
  const out = keep(alloc(bufferInspectionSize), bufferInspectionSize);
  zero(out, bufferInspectionSize);
  check(exportsRef.zgml_buffer_inspect(buffer, out));
  return out;
}

function bufferWrite(buffer, byteOffset, srcPtr, byteLen) {
  check(exportsRef.zgml_buffer_write(buffer, byteOffset, srcPtr, byteLen));
}

function bufferRead(buffer, byteOffset, dstPtr, byteLen) {
  check(exportsRef.zgml_buffer_read(buffer, byteOffset, dstPtr, byteLen));
}

function createModel(kind, inputLen, outputLen) {
  const desc = keep(alloc(modelDescSize), modelDescSize);
  zero(desc, modelDescSize);
  writeU32(desc, 0, kind);
  writeU32(desc, 4, 0);
  writeU32(desc, 8, inputLen);
  writeU32(desc, 12, outputLen);
  writeU32(desc, 16, 0);

  const out = keep(alloc(handleSize), handleSize);
  check(exportsRef.zgml_model_create(desc, out));
  const handle = u32(out);
  if (handle === 0) throw new Error("model handle was not returned");
  return handle;
}

function inspectModel(model) {
  const out = keep(alloc(modelInspectionSize), modelInspectionSize);
  zero(out, modelInspectionSize);
  check(exportsRef.zgml_model_inspect(model, out));
  return out;
}

function loadSafetensorsData(kind, bytes, reserved = 0) {
  const dataPtr = keep(alloc(bytes.length), bytes.length);
  writeBytes(dataPtr, bytes);
  const desc = keep(alloc(safetensorsDataLoadDescSize), safetensorsDataLoadDescSize);
  zero(desc, safetensorsDataLoadDescSize);
  writeU32(desc, 0, kind);
  writeU32(desc, 4, reserved);
  writeU32(desc, 8, dataPtr);
  writeU32(desc, 12, bytes.length);
  const out = keep(alloc(handleSize), handleSize);
  writeU32(out, 0, 0xffffffff);
  const status = exportsRef.zgml_model_load_safetensors_data(desc, out);
  return { status, handle: u32(out) };
}

function probeSafetensorsData(kind, bytes, reserved = 0) {
  const dataPtr = keep(alloc(bytes.length), bytes.length);
  writeBytes(dataPtr, bytes);
  const desc = keep(alloc(safetensorsDataLoadDescSize), safetensorsDataLoadDescSize);
  zero(desc, safetensorsDataLoadDescSize);
  writeU32(desc, 0, kind);
  writeU32(desc, 4, reserved);
  writeU32(desc, 8, dataPtr);
  writeU32(desc, 12, bytes.length);
  const out = keep(alloc(modelInspectionSize), modelInspectionSize);
  zero(out, modelInspectionSize);
  const status = exportsRef.zgml_model_probe_safetensors_data(desc, out);
  return { status, out };
}

function expectTinyLlamaCheckpointInspection(inspection, label, options = {}) {
  const kind = options.kind ?? tinyLlamaKind;
  const layers = options.layers ?? 1;
  const maxSeqLen = options.maxSeqLen ?? (kind === tinyLlama2LayerKind ? 32 : 8);
  if (
    u32(inspection, 0) !== kind ||
    u64(inspection, 24) !== 8n ||
    u64(inspection, 32) !== BigInt(maxSeqLen) ||
    u64(inspection, 40) !== 4n ||
    u64(inspection, 48) !== BigInt(layers) ||
    u64(inspection, 56) !== 1n ||
    u64(inspection, 64) !== 1n ||
    u64(inspection, 96) !== 0n
  ) {
    throw new Error(`unexpected ${label} tiny LLaMA checkpoint inspection`);
  }
}

function expectSmolLMCheckpointInspection(inspection, label) {
  if (
    u32(inspection, 0) !== smollm135mKind ||
    u64(inspection, 24) !== 49152n ||
    u64(inspection, 32) !== 2048n ||
    u64(inspection, 40) !== 576n ||
    u64(inspection, 48) !== 30n ||
    u64(inspection, 64) !== 3n ||
    u64(inspection, 96) !== 1n
  ) {
    throw new Error(`unexpected ${label} SmolLM checkpoint inspection`);
  }
}

function expectTinyLlamaLoadedModelStep(model, label) {
  let program = 0;
  let session = 0;
  try {
    program = compileProgram(model, 4);
    session = bindSession(program);
    const logits = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(logits, Array(tinyLlamaVocabSize + 1).fill(-123));
    const step = tokenStep(session, 0, logits, tinyLlamaVocabSize + 1);
    check(step.status);
    if (
      step.outputLen !== tinyLlamaVocabSize ||
      sessionPosition(session) !== 1 ||
      readF32Array(logits + tinyLlamaVocabSize * 4, 1)[0] !== -123
    ) {
      throw new Error(`unexpected ${label} tiny checkpoint step state`);
    }
    for (const value of readF32Array(logits, tinyLlamaVocabSize)) {
      if (Math.abs(value) > 1e-6) {
        throw new Error(`expected zero ${label} tiny checkpoint logits`);
      }
    }
  } finally {
    if (session !== 0) exportsRef.zgml_session_free(session);
    if (program !== 0) exportsRef.zgml_program_free(program);
  }
}

function expectTinyLlama2LayerNativeKvBinding(model, label) {
  let program = 0;
  let session = 0;
  try {
    program = compileProgram(model, 4);
    const requirements = programRequirements(program);
    if (
      u32(requirements, 0) !== tinyLlama2LayerKind ||
      requirementOutputLen(requirements) !== tinyLlamaVocabSize ||
      requirementOutputByteLen(requirements) !== tinyLlamaVocabSize * 4 ||
      requirementContextLen(requirements) !== 4
    ) {
      throw new Error(`unexpected ${label} two-layer tiny LLaMA requirements: kind=${u32(requirements, 0)} outputLen=${requirementOutputLen(requirements)} outputBytes=${requirementOutputByteLen(requirements)} context=${requirementContextLen(requirements)}`);
    }

    const kvRequirements = keep(alloc(llamaKvCacheRequirementsSize), llamaKvCacheRequirementsSize);
    zero(kvRequirements, llamaKvCacheRequirementsSize);
    check(exportsRef.zgml_llama_program_get_kv_cache_requirements(program, kvRequirements));
    if (
      u32(kvRequirements, 0) !== tinyLlama2LayerKind ||
      u32(kvRequirements, 8) !== 2 ||
      u32(kvRequirements, 16) !== 4 ||
      u32(kvRequirements, 20) !== 64 ||
      u32(kvRequirements, 24) !== 64
    ) {
      throw new Error(`unexpected ${label} two-layer tiny LLaMA KV requirements`);
    }

    const outputBuffer = createProgramBuffer(program, programBufferOutput);
    const cacheByteLen = u32(kvRequirements, 20);
    const kCache0 = createBuffer(cacheByteLen);
    const vCache0 = createBuffer(cacheByteLen);
    const kCache1 = createBuffer(cacheByteLen);
    const vCache1 = createBuffer(cacheByteLen);
    const kCacheHandles = keep(alloc(handleSize * 2), handleSize * 2);
    const vCacheHandles = keep(alloc(handleSize * 2), handleSize * 2);
    writeU32(kCacheHandles, 0, kCache0);
    writeU32(kCacheHandles, 4, kCache1);
    writeU32(vCacheHandles, 0, vCache0);
    writeU32(vCacheHandles, 4, vCache1);

    const badKvDesc = keep(alloc(llamaKvCacheBindDescSize), llamaKvCacheBindDescSize);
    zero(badKvDesc, llamaKvCacheBindDescSize);
    writeU32(badKvDesc, 0, kCacheHandles);
    writeU32(badKvDesc, 4, vCacheHandles);
    writeU32(badKvDesc, 8, 1);
    const badBindDesc = keep(alloc(llamaBufferBindDescSize), llamaBufferBindDescSize);
    zero(badBindDesc, llamaBufferBindDescSize);
    writeU32(badBindDesc, 8, badKvDesc);
    const badOut = keep(alloc(handleSize), handleSize);
    writeU32(badOut, 0, 0xffffffff);
    expectStatus(exportsRef.zgml_llama_session_bind_buffers(program, badBindDesc, badOut), shapeMismatch, `${label} one-layer KV bind against two-layer checkpoint`);
    if (u32(badOut) !== 0) throw new Error(`${label} bad two-layer KV bind returned a session handle`);

    const kvDesc = keep(alloc(llamaKvCacheBindDescSize), llamaKvCacheBindDescSize);
    zero(kvDesc, llamaKvCacheBindDescSize);
    writeU32(kvDesc, 0, kCacheHandles);
    writeU32(kvDesc, 4, vCacheHandles);
    writeU32(kvDesc, 8, 2);
    const bindDesc = keep(alloc(llamaBufferBindDescSize), llamaBufferBindDescSize);
    zero(bindDesc, llamaBufferBindDescSize);
    writeU32(bindDesc, 0, outputBuffer);
    writeU32(bindDesc, 4, tinyLlamaVocabSize);
    writeU32(bindDesc, 8, kvDesc);
    session = bindLlamaBufferSession(program, bindDesc);

    const inspection = inspectSession(session);
    if (
      u32(inspection, 0) !== tinyLlama2LayerKind ||
      u32(inspection, 4) !== backendCpu ||
      u32(inspection, 8) !== bufferStorageHost ||
      u32(inspection, 12) !== bufferStorageHost ||
      u64(inspection, 16) !== 0n ||
      u64(inspection, 24) !== 4n ||
      u64(inspection, 56) === 0n ||
      u64(inspection, 64) !== 0n ||
      u64(inspection, 72) === 0n
    ) {
      throw new Error(`unexpected ${label} two-layer tiny LLaMA Session inspection`);
    }

    const logits = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(logits, Array(tinyLlamaVocabSize + 1).fill(-2020));
    const step = tokenStep(session, 1, logits, tinyLlamaVocabSize + 1);
    check(step.status);
    if (
      step.outputLen !== tinyLlamaVocabSize ||
      sessionPosition(session) !== 1 ||
      readF32Array(logits + tinyLlamaVocabSize * 4, 1)[0] !== -2020
    ) {
      throw new Error(`unexpected ${label} two-layer tiny LLaMA step state`);
    }
  } finally {
    if (session !== 0) exportsRef.zgml_session_free(session);
    if (program !== 0) exportsRef.zgml_program_free(program);
  }
}

function checkSupportedCheckpointCatalog() {
  const count = Number(exportsRef.zgml_supported_checkpoint_count());
  if (count !== 3) throw new Error(`expected 3 supported checkpoint envelopes, got ${count}`);

  const out = keep(alloc(modelInspectionSize), modelInspectionSize);
  zero(out, modelInspectionSize);
  check(exportsRef.zgml_supported_checkpoint_inspect(0, out));
  if (
    u32(out, 0) !== tinyLlamaKind ||
    u64(out, 24) !== 8n ||
    u64(out, 32) !== 8n ||
    u64(out, 40) !== 4n ||
    u64(out, 48) !== 1n ||
    u64(out, 64) !== 1n ||
    u64(out, 96) !== 0n
  ) {
    throw new Error("unexpected tiny LLaMA supported checkpoint envelope");
  }

  zero(out, modelInspectionSize);
  check(exportsRef.zgml_supported_checkpoint_inspect(1, out));
  expectTinyLlamaCheckpointInspection(out, "two-layer supported catalog", { kind: tinyLlama2LayerKind, layers: 2 });

  zero(out, modelInspectionSize);
  check(exportsRef.zgml_supported_checkpoint_inspect(2, out));
  expectSmolLMCheckpointInspection(out, "supported catalog");

  writeU32(out, 0, 99);
  writeU32(out, 24, 123);
  expectStatus(exportsRef.zgml_supported_checkpoint_inspect(count, out), invalidArgument, "invalid supported checkpoint index");
  if (u32(out, 0) !== 0 || u64(out, 24) !== 0n) throw new Error("invalid supported checkpoint inspect did not clear output");
}

function smollmSafetensorsHeader() {
  const tensors = {};
  const add = (name, shape) => {
    tensors[name] = { dtype: "F16", shape, data_offsets: [0, 0] };
  };
  add("model.embed_tokens.weight", [49152, 576]);
  add("model.norm.weight", [576]);
  for (const layer of [0, 29]) {
    add(`model.layers.${layer}.self_attn.q_proj.weight`, [576, 576]);
    add(`model.layers.${layer}.self_attn.k_proj.weight`, [192, 576]);
    add(`model.layers.${layer}.self_attn.v_proj.weight`, [192, 576]);
    add(`model.layers.${layer}.self_attn.o_proj.weight`, [576, 576]);
    add(`model.layers.${layer}.mlp.gate_proj.weight`, [1536, 576]);
    add(`model.layers.${layer}.mlp.up_proj.weight`, [1536, 576]);
    add(`model.layers.${layer}.mlp.down_proj.weight`, [576, 1536]);
    add(`model.layers.${layer}.input_layernorm.weight`, [576]);
    add(`model.layers.${layer}.post_attention_layernorm.weight`, [576]);
  }
  return JSON.stringify(tensors);
}

function elementCount(shape) {
  return shape.reduce((n, dim) => n * dim, 1);
}

function safetensorsScalarBytes(dtype = "F32") {
  switch (String(dtype).toLowerCase()) {
    case "f32":
    case "float32":
      return 4;
    case "f16":
    case "float16":
    case "bf16":
    case "bfloat16":
      return 2;
    default:
      throw new Error(`unsupported test safetensors dtype: ${dtype}`);
  }
}

function float32ToFloat16Bits(value) {
  if (Number.isNaN(value)) return 0x7e00;
  const sign = Object.is(value, -0) || value < 0 ? 0x8000 : 0;
  const abs = Math.abs(value);
  if (abs === 0) return sign;
  if (!Number.isFinite(abs)) return sign | 0x7c00;
  if (abs < 2 ** -24) return sign;
  if (abs < 2 ** -14) return sign | Math.min(0x03ff, Math.round(abs / (2 ** -24)));
  let exponent = Math.floor(Math.log2(abs));
  let mantissa = Math.round((abs / (2 ** exponent) - 1) * 0x400);
  if (mantissa === 0x400) {
    exponent += 1;
    mantissa = 0;
  }
  if (exponent > 15) return sign | 0x7c00;
  return sign | ((exponent + 15) << 10) | mantissa;
}

const bfloatWriteScratchF32 = new Float32Array(1);
const bfloatWriteScratchU32 = new Uint32Array(bfloatWriteScratchF32.buffer);

function float32ToBfloat16Bits(value) {
  bfloatWriteScratchF32[0] = value;
  const bits = bfloatWriteScratchU32[0];
  return (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16;
}

function float16BitsToNumber(bits) {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * (2 ** -14) * (fraction / 0x400);
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : NaN;
  return sign * (2 ** (exponent - 15)) * (1 + fraction / 0x400);
}

const bfloatReadScratchU32 = new Uint32Array(1);
const bfloatReadScratchF32 = new Float32Array(bfloatReadScratchU32.buffer);

function bfloat16BitsToNumber(bits) {
  bfloatReadScratchU32[0] = (bits & 0xffff) << 16;
  return bfloatReadScratchF32[0];
}

function writeSafetensorsScalar(view, byteOffset, dtype, value) {
  switch (String(dtype).toLowerCase()) {
    case "f32":
    case "float32":
      view.setFloat32(byteOffset, value, true);
      return;
    case "f16":
    case "float16":
      view.setUint16(byteOffset, float32ToFloat16Bits(value), true);
      return;
    case "bf16":
    case "bfloat16":
      view.setUint16(byteOffset, float32ToBfloat16Bits(value), true);
      return;
    default:
      throw new Error(`unsupported test safetensors dtype: ${dtype}`);
  }
}

function readSafetensorsScalar(view, byteOffset, dtype) {
  switch (String(dtype).toLowerCase()) {
    case "f32":
    case "float32":
      return view.getFloat32(byteOffset, true);
    case "f16":
    case "float16":
      return Math.fround(float16BitsToNumber(view.getUint16(byteOffset, true)));
    case "bf16":
    case "bfloat16":
      return Math.fround(bfloat16BitsToNumber(view.getUint16(byteOffset, true)));
    default:
      throw new Error(`unsupported test safetensors dtype: ${dtype}`);
  }
}

function tinyLlamaSafetensorsHeader(specs = tinyLlamaTensorSpecs) {
  const tensors = {};
  for (const [name, shape] of specs) {
    tensors[name] = { dtype: "F16", shape, data_offsets: [0, 0] };
  }
  return JSON.stringify(tensors);
}

function safetensorsHeaderBytes(header) {
  const headerBytes = new TextEncoder().encode(header);
  const bytes = new Uint8Array(8 + headerBytes.length);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  bytes.set(headerBytes, 8);
  return bytes;
}

function mutateSafetensorsHeaderBytes(bytes, mutateHeader) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = Number(view.getBigUint64(0, true));
  const dataStart = 8 + headerLength;
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, dataStart)));
  const payload = bytes.subarray(dataStart);
  mutateHeader(header, payload.byteLength);
  let headerJson = JSON.stringify(header);
  while ((8 + new TextEncoder().encode(headerJson).length) % Float32Array.BYTES_PER_ELEMENT !== 0) {
    headerJson += " ";
  }
  const headerBytes = new TextEncoder().encode(headerJson);
  const out = new Uint8Array(8 + headerBytes.length + payload.byteLength);
  new DataView(out.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  out.set(headerBytes, 8);
  out.set(payload, 8 + headerBytes.length);
  return out;
}

function tinyLlamaSafetensorsFileBytes(specs = tinyLlamaTensorSpecs, options = {}) {
  const tensors = {};
  if (options.metadata !== undefined && options.metadata !== null) {
    tensors.__metadata__ = { ...options.metadata };
  }
  const dtype = options.dtype ?? "F32";
  const scalarBytes = safetensorsScalarBytes(dtype);
  let offset = 0;
  for (const [name, shape] of specs) {
    const byteLength = elementCount(shape) * scalarBytes;
    tensors[name] = { dtype, shape, data_offsets: [offset, offset + byteLength] };
    offset += byteLength;
  }
  let header = JSON.stringify(tensors);
  while ((8 + new TextEncoder().encode(header).length) % Float32Array.BYTES_PER_ELEMENT !== 0) {
    header += " ";
  }
  const headerBytes = new TextEncoder().encode(header);
  const bytes = new Uint8Array(8 + headerBytes.length + offset);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  bytes.set(headerBytes, 8);
  return bytes;
}

function tinyLlamaModelResourceFileBytes(specs = tinyLlamaTensorSpecs, options = {}) {
  const bytes = tinyLlamaSafetensorsFileBytes(specs, options);
  const dataStart = safetensorsDataStart(bytes);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const dtype = options.dtype ?? "F32";
  const scalarBytes = safetensorsScalarBytes(dtype);
  const tensorIndexOffset = options.tensorIndexOffset ?? 0;
  let offset = 0;
  for (let tensorIndex = 0; tensorIndex < specs.length; tensorIndex += 1) {
    const [name, shape] = specs[tensorIndex];
    const count = elementCount(shape);
    for (let i = 0; i < count; i += 1) {
      let value = Math.fround((tensorIndex + tensorIndexOffset + 1) * 0.01 + (i + 1) * 0.001);
      if (name === "model.embed_tokens.weight") {
        const token = Math.floor(i / shape[1]);
        const col = i % shape[1];
        value = Math.fround((token + 1) * 0.1 + (col + 1) * 0.01);
      } else if (name === "lm_head.weight") {
        const row = Math.floor(i / shape[1]);
        const col = i % shape[1];
        value = Math.fround((row + 1) * 0.2 + (col + 1) * 0.03);
      } else if (name === "model.rotary_emb.inv_freq" || name.endsWith(".rotary_emb.inv_freq")) {
        const ropeBase = Number(options.ropeBase ?? options.ropeTheta ?? options.metadata?.rope_theta ?? options.metadata?.rope_base ?? 10000);
        const headSize = shape[0] * 2;
        const ropeScaling = {
          ropeHighFreqFactor: options.ropeHighFreqFactor ?? options.rope_high_freq_factor,
          ropeKind: options.ropeKind ?? options.ropeType ?? options.rope_kind ?? options.rope_type,
          ropeLowFreqFactor: options.ropeLowFreqFactor ?? options.rope_low_freq_factor,
          ropeOriginalContextLength: options.ropeOriginalContextLength ?? options.rope_original_context_length,
          ropeScale: options.ropeScale ?? options.rope_scale ?? options.ropeScalingFactor ?? options.rope_scaling_factor,
        };
        value = Math.fround(tinyRopeInvFrequency(i, headSize, ropeBase, ropeScaling));
      }
      writeSafetensorsScalar(view, dataStart + offset + i * scalarBytes, dtype, value);
    }
    offset += count * scalarBytes;
  }
  return bytes;
}

function probeSafetensorsHeader(header, kind = autoKind, reserved = 0) {
  const bytes = new TextEncoder().encode(header);
  const headerPtr = keep(alloc(bytes.length), bytes.length);
  writeBytes(headerPtr, bytes);
  const desc = keep(alloc(safetensorsHeaderProbeDescSize), safetensorsHeaderProbeDescSize);
  zero(desc, safetensorsHeaderProbeDescSize);
  writeU32(desc, 0, kind);
  writeU32(desc, 4, reserved);
  writeU32(desc, 8, headerPtr);
  writeU32(desc, 12, bytes.length);
  const out = keep(alloc(modelInspectionSize), modelInspectionSize);
  zero(out, modelInspectionSize);
  const status = exportsRef.zgml_model_probe_safetensors_header(desc, out);
  return { status, out };
}

function checkSafetensorsHeaderProbe() {
  const smollmHeader = smollmSafetensorsHeader();
  const probe = probeSafetensorsHeader(smollmHeader);
  check(probe.status);
  expectSmolLMCheckpointInspection(probe.out, "safetensors header probe");

  const smollmHeaderBytes = safetensorsHeaderBytes(smollmHeader);
  const smollmDataProbe = probeSafetensorsData(autoKind, smollmHeaderBytes);
  check(smollmDataProbe.status);
  expectSmolLMCheckpointInspection(smollmDataProbe.out, "auto safetensors data probe");

  const fixedSmollmDataProbe = probeSafetensorsData(smollm135mKind, smollmHeaderBytes);
  check(fixedSmollmDataProbe.status);
  expectSmolLMCheckpointInspection(fixedSmollmDataProbe.out, "fixed safetensors data probe");
  expectStatus(probeSafetensorsData(tinyLlamaKind, smollmHeaderBytes).status, unsupported, "tiny LLaMA safetensors data probe");

  const tinyHeader = tinyLlamaSafetensorsHeader();
  const tinyProbe = probeSafetensorsHeader(tinyHeader);
  check(tinyProbe.status);
  expectTinyLlamaCheckpointInspection(tinyProbe.out, "tiny LLaMA safetensors header probe");

  const twoLayerHeader = tinyLlamaSafetensorsHeader(tinyLlamaTwoLayerTensorSpecs);
  const twoLayerProbe = probeSafetensorsHeader(twoLayerHeader);
  check(twoLayerProbe.status);
  expectTinyLlamaCheckpointInspection(twoLayerProbe.out, "two-layer tiny LLaMA safetensors header probe", {
    kind: tinyLlama2LayerKind,
    layers: 2,
  });
  const twoLayerDataProbe = probeSafetensorsData(autoKind, safetensorsHeaderBytes(twoLayerHeader));
  check(twoLayerDataProbe.status);
  expectTinyLlamaCheckpointInspection(twoLayerDataProbe.out, "two-layer tiny LLaMA safetensors data probe", {
    kind: tinyLlama2LayerKind,
    layers: 2,
  });
  const fixedTwoLayerHeaderProbe = probeSafetensorsHeader(twoLayerHeader, tinyLlama2LayerKind);
  check(fixedTwoLayerHeaderProbe.status);
  expectTinyLlamaCheckpointInspection(fixedTwoLayerHeaderProbe.out, "fixed two-layer tiny LLaMA safetensors header probe", {
    kind: tinyLlama2LayerKind,
    layers: 2,
  });
  expectStatus(probeSafetensorsHeader(twoLayerHeader, tinyLlamaKind).status, unsupported, "one-layer tiny LLaMA two-layer safetensors header probe");
  expectStatus(probeSafetensorsHeader(tinyHeader, tinyLlama2LayerKind).status, unsupported, "two-layer tiny LLaMA one-layer safetensors header probe");
  expectStatus(probeSafetensorsHeader(smollmHeader, tinyLlamaKind).status, unsupported, "tiny LLaMA safetensors header probe");
  expectStatus(probeSafetensorsHeader(tinyHeader, smollm135mKind).status, unsupported, "SmolLM safetensors header probe");
  expectStatus(probeSafetensorsHeader(smollmHeader, autoKind, 1).status, invalidArgument, "reserved safetensors header probe");
}

function runTinyLlamaSafetensorsDataLoadSmoke() {
  const bytes = tinyLlamaSafetensorsFileBytes();
  const twoLayerBytes = tinyLlamaSafetensorsFileBytes(tinyLlamaTwoLayerTensorSpecs);
  let autoModel = 0;
  let fixedModel = 0;
  let autoTwoLayerModel = 0;
  let fixedTwoLayerModel = 0;
  try {
    const invalidProbe = probeSafetensorsData(autoKind, bytes, 1);
    expectStatus(invalidProbe.status, invalidArgument, "reserved safetensors data probe");
    if (u32(invalidProbe.out, 0) !== 0) throw new Error("reserved safetensors data probe returned an inspection");

    const rejectedProbe = probeSafetensorsData(smollm135mKind, bytes);
    expectStatus(rejectedProbe.status, unsupported, "SmolLM safetensors data probe");
    if (u32(rejectedProbe.out, 0) !== 0) throw new Error("SmolLM safetensors data probe returned an inspection");

    const fixedProbe = probeSafetensorsData(tinyLlamaKind, bytes);
    check(fixedProbe.status);
    expectTinyLlamaCheckpointInspection(fixedProbe.out, "fixed safetensors data probe");

    const autoProbe = probeSafetensorsData(autoKind, bytes);
    check(autoProbe.status);
    expectTinyLlamaCheckpointInspection(autoProbe.out, "auto safetensors data probe");

    const rejectedTwoLayerAsOneProbe = probeSafetensorsData(tinyLlamaKind, twoLayerBytes);
    expectStatus(rejectedTwoLayerAsOneProbe.status, unsupported, "one-layer tiny LLaMA two-layer safetensors data probe");
    if (u32(rejectedTwoLayerAsOneProbe.out, 0) !== 0) throw new Error("one-layer tiny LLaMA two-layer safetensors data probe returned an inspection");

    const rejectedOneLayerAsTwoProbe = probeSafetensorsData(tinyLlama2LayerKind, bytes);
    expectStatus(rejectedOneLayerAsTwoProbe.status, unsupported, "two-layer tiny LLaMA one-layer safetensors data probe");
    if (u32(rejectedOneLayerAsTwoProbe.out, 0) !== 0) throw new Error("two-layer tiny LLaMA one-layer safetensors data probe returned an inspection");

    const fixedTwoLayerProbe = probeSafetensorsData(tinyLlama2LayerKind, twoLayerBytes);
    check(fixedTwoLayerProbe.status);
    expectTinyLlamaCheckpointInspection(fixedTwoLayerProbe.out, "fixed two-layer safetensors data probe", {
      kind: tinyLlama2LayerKind,
      layers: 2,
    });

    const autoTwoLayerProbe = probeSafetensorsData(autoKind, twoLayerBytes);
    check(autoTwoLayerProbe.status);
    expectTinyLlamaCheckpointInspection(autoTwoLayerProbe.out, "auto two-layer safetensors data probe", {
      kind: tinyLlama2LayerKind,
      layers: 2,
    });

    const invalid = loadSafetensorsData(autoKind, bytes, 1);
    expectStatus(invalid.status, invalidArgument, "reserved safetensors data load");
    if (invalid.handle !== 0) throw new Error("reserved safetensors data load returned a model handle");

    const rejected = loadSafetensorsData(smollm135mKind, bytes);
    expectStatus(rejected.status, unsupported, "SmolLM safetensors data load");
    if (rejected.handle !== 0) throw new Error("SmolLM safetensors data load returned a model handle");

    const rejectedTwoLayerAsOne = loadSafetensorsData(tinyLlamaKind, twoLayerBytes);
    expectStatus(rejectedTwoLayerAsOne.status, unsupported, "one-layer tiny LLaMA two-layer safetensors data load");
    if (rejectedTwoLayerAsOne.handle !== 0) throw new Error("one-layer tiny LLaMA two-layer safetensors data load returned a model handle");

    const rejectedOneLayerAsTwo = loadSafetensorsData(tinyLlama2LayerKind, bytes);
    expectStatus(rejectedOneLayerAsTwo.status, unsupported, "two-layer tiny LLaMA one-layer safetensors data load");
    if (rejectedOneLayerAsTwo.handle !== 0) throw new Error("two-layer tiny LLaMA one-layer safetensors data load returned a model handle");

    const fixed = loadSafetensorsData(tinyLlamaKind, bytes);
    check(fixed.status);
    fixedModel = fixed.handle;
    if (fixedModel === 0) throw new Error("fixed safetensors data load returned no model handle");
    expectTinyLlamaCheckpointInspection(inspectModel(fixedModel), "fixed safetensors data load");

    const auto = loadSafetensorsData(autoKind, bytes);
    check(auto.status);
    autoModel = auto.handle;
    if (autoModel === 0) throw new Error("auto safetensors data load returned no model handle");
    expectTinyLlamaCheckpointInspection(inspectModel(autoModel), "auto safetensors data load");
    expectTinyLlamaLoadedModelStep(autoModel, "safetensors-data-loaded");

    const fixedTwoLayer = loadSafetensorsData(tinyLlama2LayerKind, twoLayerBytes);
    check(fixedTwoLayer.status);
    fixedTwoLayerModel = fixedTwoLayer.handle;
    if (fixedTwoLayerModel === 0) throw new Error("fixed two-layer safetensors data load returned no model handle");
    expectTinyLlamaCheckpointInspection(inspectModel(fixedTwoLayerModel), "fixed two-layer safetensors data load", {
      kind: tinyLlama2LayerKind,
      layers: 2,
    });

    const autoTwoLayer = loadSafetensorsData(autoKind, twoLayerBytes);
    check(autoTwoLayer.status);
    autoTwoLayerModel = autoTwoLayer.handle;
    if (autoTwoLayerModel === 0) throw new Error("auto two-layer safetensors data load returned no model handle");
    expectTinyLlamaCheckpointInspection(inspectModel(autoTwoLayerModel), "auto two-layer safetensors data load", {
      kind: tinyLlama2LayerKind,
      layers: 2,
    });
    expectTinyLlamaLoadedModelStep(autoTwoLayerModel, "two-layer safetensors-data-loaded");
    expectTinyLlama2LayerNativeKvBinding(autoTwoLayerModel, "two-layer safetensors-data-loaded");

    append("zgml browser wasm ffi tiny checkpoint data load smoke ok");
  } finally {
    if (fixedTwoLayerModel !== 0) exportsRef.zgml_model_free(fixedTwoLayerModel);
    if (autoTwoLayerModel !== 0) exportsRef.zgml_model_free(autoTwoLayerModel);
    if (fixedModel !== 0) exportsRef.zgml_model_free(fixedModel);
    if (autoModel !== 0) exportsRef.zgml_model_free(autoModel);
  }
}

function programModelCompatibility(program, model) {
  const out = keep(alloc(programModelCompatibilitySize), programModelCompatibilitySize);
  zero(out, programModelCompatibilitySize);
  check(exportsRef.zgml_program_check_model_compatibility(program, model, out));
  return out;
}

function compileProgram(model, contextLength, backend = backendCpu) {
  const desc = keep(alloc(compileDescSize), compileDescSize);
  writeU32(desc, 0, backend);
  writeU32(desc, 4, 0);
  writeU32(desc, 8, contextLength);
  writeU32(desc, 12, 1);

  const out = keep(alloc(handleSize), handleSize);
  check(exportsRef.zgml_program_compile(model, desc, out));
  const handle = u32(out);
  if (handle === 0) throw new Error("program handle was not returned");
  return handle;
}

async function runTinyLinearWebGpuCompileOnlySmoke() {
  let model = 0;
  let program = 0;
  try {
    model = createModel(tinyLinearKind, 2, 3);
    program = compileProgram(model, 4, backendWebGpu);

    const inspection = inspectProgram(program);
    if (u64(inspection, 0) !== BigInt(backendWebGpu) || u64(inspection, 8) !== 0n || u64(inspection, 16) !== 1n || u64(inspection, 24) !== 5n || u64(inspection, 32) !== 68n || u64(inspection, 40) !== 1n || u64(inspection, 48) !== 0n || u64(inspection, 56) === 0n || u64(inspection, 64) === 0n || u64(inspection, 160) === 0n || u64(inspection, 168) !== 17n || u64(inspection, 176) !== 0xffffffffn || u64(inspection, 184) !== 0xffffffffn) {
      throw new Error("expected WebGPU compile-only tiny linear shape evidence");
    }
    if (commandCategoryTotal(inspection) === 0n) {
      throw new Error("expected WebGPU compile-only tiny linear command category evidence");
    }
    expectNoDispatchPlan(inspection, "WebGPU compile-only tiny linear");
    const capabilities = programCapabilities(program);
    if (
      capabilities.backend !== backendWebGpu ||
      capabilities.mode !== "resource-probe" ||
      capabilities.canExecute ||
      !capabilities.canBindExternalResources ||
      capabilities.hasFullDispatchPlan
    ) {
      throw new Error("unexpected WebGPU tiny linear capabilities");
    }
    expectNoDispatchPlanCapabilities(capabilities, "WebGPU compile-only tiny linear");
    expectPortableDeviceBufferApisUnsupported(program, "portable WebGPU compile-only tiny linear");

    const weights = keep(alloc(6 * 4), 6 * 4);
    writeF32Array(weights, [1, 0, 0, 1, 1, 1]);
    const bindDesc = keep(alloc(bindDescSize), bindDescSize);
    zero(bindDesc, bindDescSize);
    writeU32(bindDesc, 0, weights);
    writeU32(bindDesc, 4, 6);
    const out = keep(alloc(handleSize), handleSize);
    writeU32(out, 0, 0xffffffff);
    expectStatus(exportsRef.zgml_session_bind(program, bindDesc, out), unsupported, "WebGPU compile-only tiny linear bind");
    if (u32(out) !== 0) throw new Error("WebGPU compile-only tiny linear bind returned a session handle");

    expectHostResourceUsageValidation();

    const hostWeights = hostGpuBuffer("tiny-linear.weights", 6 * 4);
    const hostBias = {
      [wasmWebgpuInterop.importSource]() {
        return hostGpuBuffer("tiny-linear.bias", 3 * 4);
      },
    };
    const hostInput = hostGpuBuffer("tiny-linear.input", 2 * 4);
    const hostOutput = hostGpuBuffer("tiny-linear.output", 3 * 4);
    const { buffer: resourceWeights, descriptor: resourceWeightsDescriptor } = wrapHostResourceBuffer(backendWebGpu, hostWeights, { access: "read" });
    const { buffer: resourceBias, descriptor: resourceBiasDescriptor } = wrapHostResourceBuffer(backendWebGpu, hostBias, { access: "read" });
    const { buffer: resourceInput, descriptor: resourceInputDescriptor } = wrapHostResourceBuffer(backendWebGpu, hostInput, { access: "read" });
    const { buffer: resourceOutput, descriptor: resourceOutputDescriptor } = wrapHostResourceBuffer(backendWebGpu, hostOutput, { access: "write" });
    expectHostResourceBuffer(resourceWeights, resourceWeightsDescriptor, "tiny-linear weights");
    expectHostResourceBuffer(resourceBias, resourceBiasDescriptor, "tiny-linear bias");
    expectHostResourceBuffer(resourceInput, resourceInputDescriptor, "tiny-linear input");
    expectHostResourceBuffer(resourceOutput, resourceOutputDescriptor, "tiny-linear output");
    expectHostResourceTable(
      [resourceWeightsDescriptor, resourceBiasDescriptor, resourceInputDescriptor, resourceOutputDescriptor],
      "tiny-linear WebGPU resource probe",
    );
    const resourceBindDesc = sessionBindingBridge.bufferBindDesc({
      weights: resourceWeights,
      weightsLen: 6,
      bias: resourceBias,
      biasLen: 3,
      input: resourceInput,
      inputLen: 2,
      output: resourceOutput,
      outputLen: 3,
    });
    const resourceSession = sessionBindingBridge.bindBufferSession(program, resourceBindDesc);
    try {
      const sessionInspection = inspectSession(resourceSession);
      if (
        u32(sessionInspection, 0) !== tinyLinearKind ||
        u32(sessionInspection, 4) !== backendWebGpu ||
        u32(sessionInspection, 8) !== bufferStorageExternalResource ||
        u32(sessionInspection, 12) !== 0 ||
        u64(sessionInspection, 16) !== 0n ||
        u64(sessionInspection, 24) !== 0n ||
        u64(sessionInspection, 32) !== 2n ||
        u64(sessionInspection, 40) !== 1n ||
        u64(sessionInspection, 48) !== 1n ||
        u64(sessionInspection, 56) !== 0n ||
        u64(sessionInspection, 64) !== 4n ||
        u64(sessionInspection, 72) === 0n
      ) {
        throw new Error("unexpected WebGPU tiny linear resource session inspection");
      }
      const resourceStepResult = keep(alloc(stepResultSize), stepResultSize);
      writeU32(resourceStepResult, 0, 0);
      expectStatus(exportsRef.zgml_session_step(resourceSession, 0, resourceStepResult), unsupported, "WebGPU tiny linear resource step");
      if (u32(resourceStepResult) !== 0) throw new Error("WebGPU tiny linear unsupported resource step returned output");
      writeU32(resourceStepResult, 0, 99);
      expectStatus(exportsRef.zgml_session_step_no_output(resourceSession, 0, resourceStepResult), unsupported, "WebGPU tiny linear resource no-output step");
      if (u32(resourceStepResult) !== 99) throw new Error("WebGPU tiny linear unsupported resource no-output step mutated result");
      expectNoRuntimeWork(resourceSession, "WebGPU tiny linear resource probe");
    } finally {
      exportsRef.zgml_session_free(resourceSession);
    }

    {
      const hostProgram = new WasmWebGpuTinyLinearProgram({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        rows: 3,
        cols: 2,
        freeSession: (session) => exportsRef.zgml_session_free(session),
      });
      const hostProgramInspection = hostProgram.inspect();
      if (
        hostProgramInspection.kind !== "tiny-linear-resource-program" ||
        hostProgramInspection.rows !== 3 ||
        hostProgramInspection.cols !== 2 ||
        hostProgramInspection.bindingRequirementHash === 0n ||
        hostProgramInspection.persistentRequirementCount !== 2 ||
        hostProgramInspection.stepInputRequirementCount !== 1 ||
        hostProgramInspection.stepOutputRequirementCount !== 1 ||
        hostProgramInspection.weightsByteLength !== 6 * Float32Array.BYTES_PER_ELEMENT ||
        hostProgramInspection.outputByteLength !== 3 * Float32Array.BYTES_PER_ELEMENT
      ) {
        throw new Error(`unexpected WebGPU tiny linear host Program inspection: ${JSON.stringify(hostProgramInspection, (_, value) => typeof value === "bigint" ? value.toString() : value)}`);
      }
      const hostExecutionOffset = hostDevice.storageBufferOffsetAlignment();
      let unalignedHostViewRejected = false;
      try {
        const unaligned = hostGpuBuffer("tiny-linear.host-program.unaligned", hostExecutionOffset + 6 * 4);
        hostProgram.createWeightsBuffer({ buffer: unaligned, byteOffset: Float32Array.BYTES_PER_ELEMENT });
      } catch (err) {
        unalignedHostViewRejected = String(err && err.message ? err.message : err).includes("alignment");
      }
      if (!unalignedHostViewRejected) throw new Error("expected tiny-linear host Program unaligned resource view to reject");
      let oversizedHostViewRejected = false;
      try {
        const shortInput = hostGpuBuffer("tiny-linear.host-program.short-input", hostExecutionOffset + 1 * 4);
        hostProgram.createInputBuffer({ buffer: shortInput, byteOffset: hostExecutionOffset });
      } catch (err) {
        oversizedHostViewRejected = String(err && err.message ? err.message : err).includes("range");
      }
      if (!oversizedHostViewRejected) throw new Error("expected tiny-linear host Program oversized resource view to reject");
      const hostExecutionWeights = hostGpuBuffer("tiny-linear.host-program.weights", hostExecutionOffset + 6 * 4);
      const hostExecutionBias = hostGpuBuffer("tiny-linear.host-program.bias", hostExecutionOffset + 3 * 4);
      const hostExecutionInput = hostGpuBuffer("tiny-linear.host-program.input", hostExecutionOffset + 2 * 4);
      const hostExecutionOutput = hostGpuBuffer("tiny-linear.host-program.output", hostExecutionOffset + 3 * 4);
      const wrappedWeights = hostProgram.createWeightsBuffer({ buffer: hostExecutionWeights, byteOffset: hostExecutionOffset });
      const wrappedBias = hostProgram.createBiasBuffer({ buffer: hostExecutionBias, byteOffset: hostExecutionOffset });
      const wrappedInput = hostProgram.createInputBuffer({ buffer: hostExecutionInput, byteOffset: hostExecutionOffset });
      const wrappedOutput = hostProgram.createOutputBuffer({ buffer: hostExecutionOutput, byteOffset: hostExecutionOffset });
      let wrongDeviceWrappedBindingRejected = false;
      try {
        hostProgram.createOutputBuffer({
          resource: () => ({
            ...wrappedOutput,
            descriptor: {
              ...wrappedOutput.descriptor,
              deviceHandle: hostDevice.deviceHandle + 1,
              handle: wrappedOutput.descriptor.handle + 0x100,
            },
          }),
        });
      } catch (err) {
        wrongDeviceWrappedBindingRejected = String(err && err.message ? err.message : err).includes("device");
      }
      if (!wrongDeviceWrappedBindingRejected) throw new Error("expected tiny-linear pre-wrapped wrong-device resource binding to reject");
      const badHostExecutionInput = hostGpuBuffer("tiny-linear.host-program.bad-input", hostExecutionOffset + 2 * 4);
      const badHostExecutionOutput = hostGpuBuffer("tiny-linear.host-program.bad-output", hostExecutionOffset + 3 * 4);
      const { buffer: writeOnlyInputBuffer, descriptor: writeOnlyInputDescriptor } = wrapHostResourceBuffer(backendWebGpu, badHostExecutionInput, { access: "write" });
      const { buffer: readOnlyOutputBuffer, descriptor: readOnlyOutputDescriptor } = wrapHostResourceBuffer(backendWebGpu, badHostExecutionOutput, { access: "read" });
      let writeOnlyWrappedInputRejected = false;
      try {
        hostProgram.createInputBuffer({
          resource: () => ({
            buffer: writeOnlyInputBuffer,
            descriptor: writeOnlyInputDescriptor,
          }),
        });
      } catch (err) {
        writeOnlyWrappedInputRejected = String(err && err.message ? err.message : err).includes("access");
      }
      if (!writeOnlyWrappedInputRejected) throw new Error("expected tiny-linear pre-wrapped write-only input resource to reject");
      let readOnlyWrappedOutputRejected = false;
      try {
        hostProgram.createOutputBuffer({
          resource: () => ({
            buffer: readOnlyOutputBuffer,
            descriptor: readOnlyOutputDescriptor,
          }),
        });
      } catch (err) {
        readOnlyWrappedOutputRejected = String(err && err.message ? err.message : err).includes("access");
      }
      if (!readOnlyWrappedOutputRejected) throw new Error("expected tiny-linear pre-wrapped read-only output resource to reject");
      const hostSession = hostProgram.bind({
        weights: wrappedWeights,
        bias: wrappedBias,
        input: wrappedInput,
        output: wrappedOutput,
      });
      try {
        if (
          hostSession.table.deviceHandle !== hostDevice.deviceHandle ||
          hostSession.table.resourceCount !== 4 ||
          hostSession.table.byteLength !== (6 + 3 + 2 + 3) * 4 ||
          hostSession.table.hash === 0n
        ) {
          throw new Error("unexpected tiny-linear host Program resource table");
        }
        expectHostResourceTableRoles(
          hostSession.table,
          ["tiny-linear.weights", "tiny-linear.bias", "tiny-linear.input", "tiny-linear.output"],
          "WebGPU tiny linear host Program",
        );
        const hostSessionInspection = inspectSession(hostSession.handle);
        if (
          u32(hostSessionInspection, 0) !== tinyLinearKind ||
          u32(hostSessionInspection, 4) !== backendWebGpu ||
          u32(hostSessionInspection, 8) !== bufferStorageExternalResource ||
          u64(hostSessionInspection, 64) !== 4n ||
          u64(hostSessionInspection, 72) === 0n
        ) {
          throw new Error("unexpected tiny-linear host Program Session inspection");
        }
        hostDevice.writeFloat32(hostSession.bindings.weights.descriptor, [1, 0, 0, 1, 1, 1]);
        hostDevice.writeFloat32(hostSession.bindings.bias.descriptor, [0, 0, 0]);
        hostDevice.writeFloat32(hostSession.bindings.input.descriptor, [2, 3]);
        hostDevice.writeFloat32(hostSession.bindings.output.descriptor, [-1, -1, -1]);
        const hostStepResult = keep(alloc(stepResultSize), stepResultSize);
        writeU32(hostStepResult, 0, 0xffffffff);
        expectStatus(exportsRef.zgml_session_step(hostSession.handle, 0, hostStepResult), ok, "browser host WebGPU tiny linear native-export step");
        if (u32(hostStepResult) !== 3) throw new Error("browser host WebGPU tiny linear step did not write output_len");
        expectHostTinyLinearProfile(hostSession.runtimeProfile(), {
          callCount: 1,
          backendDispatchCount: 1,
          noOutputCallCount: 0,
          outputElementCount: 3,
          outputReadCount: 0,
          syncCount: 0,
          lastOutputLength: 3,
        }, "browser host WebGPU tiny linear after logits step");
        expectClose(Array.from(await hostSession.output()), [2, 3, 5]);
        expectHostTinyLinearProfile(hostSession.runtimeProfile(), {
          callCount: 1,
          backendDispatchCount: 1,
          noOutputCallCount: 0,
          outputElementCount: 3,
          outputReadCount: 1,
          syncCount: 1,
          lastOutputLength: 3,
        }, "browser host WebGPU tiny linear after explicit output read");
        hostDevice.writeFloat32(hostSession.bindings.input.descriptor, [4, 1]);
        hostDevice.writeFloat32(hostSession.bindings.output.descriptor, [-8, -8, -8]);
        writeU32(hostStepResult, 0, 0xffffffff);
        expectStatus(exportsRef.zgml_session_step_no_output(hostSession.handle, 0, hostStepResult), ok, "browser host WebGPU tiny linear native-export no-output step");
        if (u32(hostStepResult) !== 0) throw new Error("browser host WebGPU tiny linear no-output step did not clear output_len");
        expectHostTinyLinearProfile(hostSession.runtimeProfile(), {
          callCount: 2,
          backendDispatchCount: 2,
          noOutputCallCount: 1,
          outputElementCount: 3,
          outputReadCount: 1,
          syncCount: 1,
          lastOutputLength: 0,
        }, "browser host WebGPU tiny linear after no-output step");
        expectClose(Array.from(await hostSession.output()), [4, 1, 5]);
        expectHostTinyLinearProfile(hostSession.runtimeProfile(), {
          callCount: 2,
          backendDispatchCount: 2,
          noOutputCallCount: 1,
          outputElementCount: 3,
          outputReadCount: 2,
          syncCount: 2,
          lastOutputLength: 0,
        }, "browser host WebGPU tiny linear after no-output explicit read");
        hostSession.resetRuntimeProfile();
        expectHostTinyLinearProfile(hostSession.runtimeProfile(), {
          callCount: 0,
          backendDispatchCount: 0,
          noOutputCallCount: 0,
          outputElementCount: 0,
          outputReadCount: 0,
          syncCount: 0,
          lastOutputLength: 0,
        }, "browser host WebGPU tiny linear after profile reset");

        const wrappedHostExports = hostRuntime.wrapExports(exportsRef);
        hostDevice.writeFloat32(hostSession.bindings.input.descriptor, [6, 2]);
        hostDevice.writeFloat32(hostSession.bindings.output.descriptor, [-9, -9, -9]);
        writeU32(hostStepResult, 0, 0xffffffff);
        const wrappedStepStatus = wrappedHostExports.zgml_session_step(hostSession.handle, 0, hostStepResult);
        if (typeof wrappedStepStatus !== "number") throw new Error("wrapped browser host WebGPU tiny linear step did not return a synchronous status");
        expectStatus(wrappedStepStatus, ok, "wrapped browser host WebGPU tiny linear native-export step");
        if (u32(hostStepResult) !== 3) throw new Error("wrapped browser host WebGPU tiny linear step did not write output_len");
        expectHostTinyLinearProfile(hostSession.runtimeProfile(), {
          callCount: 1,
          backendDispatchCount: 1,
          noOutputCallCount: 0,
          outputElementCount: 3,
          outputReadCount: 0,
          syncCount: 0,
          lastOutputLength: 3,
        }, "wrapped browser host WebGPU tiny linear after logits step");
        expectClose(Array.from(await hostSession.output()), [6, 2, 8]);
        expectHostTinyLinearProfile(hostSession.runtimeProfile(), {
          callCount: 1,
          backendDispatchCount: 1,
          noOutputCallCount: 0,
          outputElementCount: 3,
          outputReadCount: 1,
          syncCount: 1,
          lastOutputLength: 3,
        }, "wrapped browser host WebGPU tiny linear after explicit output read");

        hostDevice.writeFloat32(hostSession.bindings.input.descriptor, [1, 7]);
        hostDevice.writeFloat32(hostSession.bindings.output.descriptor, [-10, -10, -10]);
        writeU32(hostStepResult, 0, 0xffffffff);
        const wrappedNoOutputStatus = wrappedHostExports.zgml_session_step_no_output(hostSession.handle, 0, hostStepResult);
        if (typeof wrappedNoOutputStatus !== "number") throw new Error("wrapped browser host WebGPU tiny linear no-output step did not return a synchronous status");
        expectStatus(wrappedNoOutputStatus, ok, "wrapped browser host WebGPU tiny linear native-export no-output step");
        if (u32(hostStepResult) !== 0) throw new Error("wrapped browser host WebGPU tiny linear no-output step did not clear output_len");
        expectHostTinyLinearProfile(hostSession.runtimeProfile(), {
          callCount: 2,
          backendDispatchCount: 2,
          noOutputCallCount: 1,
          outputElementCount: 3,
          outputReadCount: 1,
          syncCount: 1,
          lastOutputLength: 0,
        }, "wrapped browser host WebGPU tiny linear after no-output step");
        expectClose(Array.from(await hostSession.output()), [1, 7, 8]);
        expectHostTinyLinearProfile(hostSession.runtimeProfile(), {
          callCount: 2,
          backendDispatchCount: 2,
          noOutputCallCount: 1,
          outputElementCount: 3,
          outputReadCount: 2,
          syncCount: 2,
          lastOutputLength: 0,
        }, "wrapped browser host WebGPU tiny linear after no-output explicit read");

        const freedHostSessionHandle = hostSession.handle;
        exportsRef.zgml_session_free(freedHostSessionHandle);
        if (!hostSession.freed) throw new Error("native-export zgml_session_free did not mark host Session freed");
        if (hostRuntime.hasClaimed(freedHostSessionHandle)) throw new Error("native-export zgml_session_free did not release host Session claim");
        expectStatus(hostRuntime.stepSync(freedHostSessionHandle, 0, hostStepResult), hostRuntime.status.unclaimed, "freed browser host WebGPU tiny linear host callback step");
        append(`zgml browser wasm ffi webgpu tiny linear native-export Program/Session/free ${hostDevice.mode} execution ok`);
      } finally {
        hostSession.free();
        hostProgram.destroy();
      }
    }

    append("zgml browser wasm ffi webgpu compile/resource-probe tiny linear smoke ok");
  } finally {
    if (program !== 0) exportsRef.zgml_program_free(program);
    if (model !== 0) exportsRef.zgml_model_free(model);
  }
}

async function runTinyLlamaWebGpuCompileOnlySmoke() {
  let model = 0;
  let compatibleModel = 0;
  let program = 0;
  try {
    model = createModel(tinyLlamaKind, 0, 0);
    compatibleModel = createModel(tinyLlamaKind, 0, 0);
    program = compileProgram(model, 4, backendWebGpu);

    const inspection = inspectProgram(program);
    if (u64(inspection, 0) !== BigInt(backendWebGpu) || u64(inspection, 8) !== 0n || u64(inspection, 16) !== 1n || u64(inspection, 24) === 0n || u64(inspection, 32) === 0n || u64(inspection, 56) === 0n || u64(inspection, 64) === 0n || u64(inspection, 72) === 0n || u64(inspection, 96) === 0n || u64(inspection, 160) === 0n || u64(inspection, 168) === 0n || u64(inspection, 176) !== 3n || u64(inspection, 184) !== 4n) {
      throw new Error("expected WebGPU compile-only executable shape evidence");
    }
    if (u64(inspection, 120) === 0n && u64(inspection, 128) === 0n) {
      throw new Error("expected WebGPU compile-only LLaMA command category evidence");
    }
    expectNoDispatchPlan(inspection, "WebGPU compile-only LLaMA");
    const capabilities = programCapabilities(program);
    if (
      capabilities.backend !== backendWebGpu ||
      capabilities.mode !== "resource-probe" ||
      capabilities.canExecute ||
      !capabilities.canBindExternalResources ||
      capabilities.hasFullDispatchPlan
    ) {
      throw new Error("unexpected WebGPU LLaMA capabilities");
    }
    expectNoDispatchPlanCapabilities(capabilities, "WebGPU compile-only LLaMA");

    const llamaInspection = keep(alloc(llamaProgramInspectionSize), llamaProgramInspectionSize);
    check(exportsRef.zgml_llama_program_inspect(program, llamaInspection));
    if (u64(llamaInspection, 0) !== 8n || u64(llamaInspection, 16) !== 4n || u64(llamaInspection, 96) !== u64(inspection, 72)) {
      throw new Error("unexpected WebGPU compile-only tiny LLaMA inspection");
    }

    const out = keep(alloc(handleSize), handleSize);
    writeU32(out, 0, 0xffffffff);
    expectStatus(exportsRef.zgml_session_bind(program, 0, out), unsupported, "WebGPU compile-only bind");
    if (u32(out) !== 0) throw new Error("WebGPU compile-only bind returned a session handle");

    const kvRequirements = keep(alloc(llamaKvCacheRequirementsSize), llamaKvCacheRequirementsSize);
    zero(kvRequirements, llamaKvCacheRequirementsSize);
    check(exportsRef.zgml_llama_program_get_kv_cache_requirements(program, kvRequirements));
    if (u32(kvRequirements, 0) !== tinyLlamaKind || u32(kvRequirements, 4) !== 4 || u32(kvRequirements, 8) !== 1 || u32(kvRequirements, 16) !== 4) {
      throw new Error("unexpected WebGPU tiny LLaMA KV requirements");
    }

    const hostLogits = hostGpuBuffer("tiny-llama.logits", tinyLlamaVocabSize * 4);
    const hostKCache = hostGpuBuffer("tiny-llama.k-cache.0", u32(kvRequirements, 20));
    const hostVCache = hostGpuBuffer("tiny-llama.v-cache.0", u32(kvRequirements, 24));
    const { buffer: outputResource, descriptor: outputResourceDescriptor } = wrapHostResourceBuffer(backendWebGpu, hostLogits, { access: "write" });
    const { buffer: kCacheResource, descriptor: kCacheDescriptor } = wrapHostResourceBuffer(backendWebGpu, hostKCache);
    const { buffer: vCacheResource, descriptor: vCacheDescriptor } = wrapHostResourceBuffer(backendWebGpu, hostVCache);
    expectHostResourceBuffer(outputResource, outputResourceDescriptor, "tiny LLaMA logits");
    expectHostResourceBuffer(kCacheResource, kCacheDescriptor, "tiny LLaMA K cache");
    expectHostResourceBuffer(vCacheResource, vCacheDescriptor, "tiny LLaMA V cache");
    expectHostResourceTable(
      [outputResourceDescriptor, kCacheDescriptor, vCacheDescriptor],
      "tiny LLaMA WebGPU resource probe",
    );
    const hostReadOnlyLogits = hostGpuBuffer("tiny-llama.bad-logits", tinyLlamaVocabSize * 4);
    const hostWriteOnlyKCache = hostGpuBuffer("tiny-llama.bad-write-k-cache.0", u32(kvRequirements, 20));
    const hostReadOnlyKCache = hostGpuBuffer("tiny-llama.bad-read-k-cache.0", u32(kvRequirements, 20));
    const { buffer: readOnlyOutputResource, descriptor: readOnlyOutputDescriptor } = wrapHostResourceBuffer(backendWebGpu, hostReadOnlyLogits, { access: "read" });
    const { buffer: writeOnlyKCacheResource, descriptor: writeOnlyKCacheDescriptor } = wrapHostResourceBuffer(backendWebGpu, hostWriteOnlyKCache, { access: "write" });
    const { buffer: readOnlyKCacheResource, descriptor: readOnlyKCacheDescriptor } = wrapHostResourceBuffer(backendWebGpu, hostReadOnlyKCache, { access: "read" });
    const badOutputBindDesc = sessionBindingBridge.llamaBufferBindDesc({
      output: readOnlyOutputResource,
      outputLen: tinyLlamaVocabSize,
      kvCache: [{ k: kCacheResource, v: vCacheResource }],
    });
    const badOutputOut = keep(alloc(handleSize), handleSize);
    writeU32(badOutputOut, 0, 0xffffffff);
    expectStatus(exportsRef.zgml_llama_session_bind_buffers(program, badOutputBindDesc, badOutputOut), unsupported, "WebGPU tiny LLaMA read-only output resource bind");
    if (u32(badOutputOut) !== 0) throw new Error("WebGPU tiny LLaMA read-only output bind returned a session handle");
    const writeOnlyKvBindDesc = sessionBindingBridge.llamaBufferBindDesc({
      output: outputResource,
      outputLen: tinyLlamaVocabSize,
      kvCache: [{ k: writeOnlyKCacheResource, v: vCacheResource }],
    });
    const writeOnlyKvOut = keep(alloc(handleSize), handleSize);
    writeU32(writeOnlyKvOut, 0, 0xffffffff);
    expectStatus(exportsRef.zgml_llama_session_bind_buffers(program, writeOnlyKvBindDesc, writeOnlyKvOut), unsupported, "WebGPU tiny LLaMA write-only KV resource bind");
    if (u32(writeOnlyKvOut) !== 0) throw new Error("WebGPU tiny LLaMA write-only KV bind returned a session handle");
    const readOnlyKvBindDesc = sessionBindingBridge.llamaBufferBindDesc({
      output: outputResource,
      outputLen: tinyLlamaVocabSize,
      kvCache: [{ k: readOnlyKCacheResource, v: vCacheResource }],
    });
    const readOnlyKvOut = keep(alloc(handleSize), handleSize);
    writeU32(readOnlyKvOut, 0, 0xffffffff);
    expectStatus(exportsRef.zgml_llama_session_bind_buffers(program, readOnlyKvBindDesc, readOnlyKvOut), unsupported, "WebGPU tiny LLaMA read-only KV resource bind");
    if (u32(readOnlyKvOut) !== 0) throw new Error("WebGPU tiny LLaMA read-only KV bind returned a session handle");
    const llamaResourceProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
    });
    const llamaExecutionOffset = hostDevice.storageBufferOffsetAlignment();
    let unalignedLlamaOutputRejected = false;
    try {
      const unalignedOutput = hostGpuBuffer("tiny-llama.unaligned-logits", llamaExecutionOffset + tinyLlamaVocabSize * 4);
      llamaResourceProgram.createOutputBuffer({ buffer: unalignedOutput, byteOffset: Float32Array.BYTES_PER_ELEMENT });
    } catch (err) {
      unalignedLlamaOutputRejected = String(err && err.message ? err.message : err).includes("alignment");
    }
    if (!unalignedLlamaOutputRejected) throw new Error("expected WebGPU tiny LLaMA unaligned output resource view to reject");
    let oversizedLlamaKvRejected = false;
    try {
      const shortKCache = hostGpuBuffer("tiny-llama.short-k-cache.0", llamaExecutionOffset + u32(kvRequirements, 20) - 4);
      llamaResourceProgram.createKvCache({
        byteOffset: llamaExecutionOffset,
        resource: ({ kind }) => kind === "k" ? shortKCache : hostVCache,
      });
    } catch (err) {
      oversizedLlamaKvRejected = String(err && err.message ? err.message : err).includes("range");
    }
    if (!oversizedLlamaKvRejected) throw new Error("expected WebGPU tiny LLaMA oversized K/V resource view to reject");
    const wrappedLlamaOutput = llamaResourceProgram.createOutputBuffer({ buffer: hostLogits });
    let wrongDeviceWrappedLlamaOutputRejected = false;
    try {
      llamaResourceProgram.createOutputBuffer({
        resource: () => ({
          ...wrappedLlamaOutput,
          descriptor: {
            ...wrappedLlamaOutput.descriptor,
            deviceHandle: hostDevice.deviceHandle + 1,
            handle: wrappedLlamaOutput.descriptor.handle + 0x100,
          },
        }),
      });
    } catch (err) {
      wrongDeviceWrappedLlamaOutputRejected = String(err && err.message ? err.message : err).includes("device");
    }
    if (!wrongDeviceWrappedLlamaOutputRejected) throw new Error("expected WebGPU tiny LLaMA pre-wrapped wrong-device output resource to reject");
    let readOnlyWrappedLlamaOutputRejected = false;
    try {
      llamaResourceProgram.createOutputBuffer({
        resource: () => ({
          buffer: readOnlyOutputResource,
          descriptor: readOnlyOutputDescriptor,
        }),
      });
    } catch (err) {
      readOnlyWrappedLlamaOutputRejected = String(err && err.message ? err.message : err).includes("access");
    }
    if (!readOnlyWrappedLlamaOutputRejected) throw new Error("expected WebGPU tiny LLaMA pre-wrapped read-only output resource to reject");
    let writeOnlyWrappedLlamaKvRejected = false;
    try {
      llamaResourceProgram.createKvCache({
        resource: ({ kind }) => kind === "k"
          ? { buffer: writeOnlyKCacheResource, descriptor: writeOnlyKCacheDescriptor }
          : { buffer: vCacheResource, descriptor: vCacheDescriptor },
      });
    } catch (err) {
      writeOnlyWrappedLlamaKvRejected = String(err && err.message ? err.message : err).includes("access");
    }
    if (!writeOnlyWrappedLlamaKvRejected) throw new Error("expected WebGPU tiny LLaMA pre-wrapped write-only K/V resource to reject");
    let readOnlyWrappedLlamaKvRejected = false;
    try {
      llamaResourceProgram.createKvCache({
        resource: ({ kind }) => kind === "k"
          ? { buffer: readOnlyKCacheResource, descriptor: readOnlyKCacheDescriptor }
          : { buffer: vCacheResource, descriptor: vCacheDescriptor },
      });
    } catch (err) {
      readOnlyWrappedLlamaKvRejected = String(err && err.message ? err.message : err).includes("access");
    }
    if (!readOnlyWrappedLlamaKvRejected) throw new Error("expected WebGPU tiny LLaMA pre-wrapped read-only K/V resource to reject");
    const wrappedLlamaKvCache = llamaResourceProgram.createKvCache({
      resource: ({ kind }) => kind === "k" ? hostKCache : hostVCache,
    });
    const llamaResources = {
      output: llamaResourceProgram.createOutputBuffer({ resource: () => wrappedLlamaOutput }),
      kvCache: llamaResourceProgram.createKvCache({
        resource: ({ kind, layer }) => wrappedLlamaKvCache[layer][kind],
      }),
    };
    const rawLlamaResources = {
      output: llamaResourceProgram.createOutputBuffer({ buffer: hostLogits }),
      kvCache: llamaResourceProgram.createKvCache({
        resource: ({ kind }) => kind === "k" ? hostKCache : hostVCache,
      }),
    };
    const llamaModelResourceBytes = tinyLlamaModelResourceFileBytes();
    const llamaModelResources = llamaResourceProgram.createModelResourcesFromSafetensors(llamaModelResourceBytes, {
      usage: webgpuBufferUsage.STORAGE | webgpuBufferUsage.COPY_SRC | webgpuBufferUsage.COPY_DST,
    });
    let duplicateLlamaModelResourceRejected = false;
    try {
      llamaResourceProgram.createModelResources([
        { role: "llama.weight.dup", shape: [4], dtype: "f32" },
        { role: "llama.weight.dup", shape: [4], dtype: "f32" },
      ]);
    } catch (err) {
      duplicateLlamaModelResourceRejected = String(err && err.message ? err.message : err).includes("duplicate");
    }
    if (!duplicateLlamaModelResourceRejected) throw new Error("expected WebGPU tiny LLaMA duplicate model resource role to reject");
    let writeOnlyLlamaModelResourceRejected = false;
    try {
      const writeOnlyModelWeight = hostGpuBuffer("tiny-llama.bad-write-model-weight", 4 * Float32Array.BYTES_PER_ELEMENT);
      llamaResourceProgram.bind({
        ...rawLlamaResources,
        modelResources: {
          "llama.weight.bad": {
            resource: writeOnlyModelWeight,
            byteLength: 4 * Float32Array.BYTES_PER_ELEMENT,
            access: "write",
          },
        },
      });
    } catch (err) {
      writeOnlyLlamaModelResourceRejected = String(err && err.message ? err.message : err).includes("readable");
    }
    if (!writeOnlyLlamaModelResourceRejected) throw new Error("expected WebGPU tiny LLaMA write-only model resource to reject");
    if (!browserFocusedLlamaProfileLabels) {
    const hostLlamaResourceSession = llamaResourceProgram.bind(llamaResources);
    const resourceSession = hostLlamaResourceSession.handle;
    let resourceBindingShapeHash = 0n;
    try {
      const sessionInspection = inspectSession(resourceSession);
      resourceBindingShapeHash = u64(sessionInspection, 72);
      if (
        u32(sessionInspection, 0) !== tinyLlamaKind ||
        u32(sessionInspection, 4) !== backendWebGpu ||
        u32(sessionInspection, 8) !== bufferStorageExternalResource ||
        u32(sessionInspection, 12) !== bufferStorageExternalResource ||
        u64(sessionInspection, 16) !== 0n ||
        u64(sessionInspection, 24) !== 4n ||
        u64(sessionInspection, 32) === 0n ||
        u64(sessionInspection, 40) === 0n ||
        u64(sessionInspection, 48) !== 1n ||
        u64(sessionInspection, 56) === 0n ||
        u64(sessionInspection, 64) !== 3n ||
        resourceBindingShapeHash === 0n
      ) {
        throw new Error("unexpected WebGPU tiny LLaMA resource session inspection");
      }
      expectHostResourceTableRoles(
        hostLlamaResourceSession.table,
        ["llama.output", "llama.k.0", "llama.v.0"],
        "WebGPU tiny LLaMA resource",
      );
      const logits = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
      writeF32Array(logits, Array(tinyLlamaVocabSize + 1).fill(-88));
      const step = tokenStep(resourceSession, 0, logits, tinyLlamaVocabSize + 1);
      expectStatus(step.status, unsupported, "WebGPU tiny LLaMA resource step");
      if (step.outputLen !== 0 || readF32Array(logits, 1)[0] !== -88 || sessionPosition(resourceSession) !== 0) {
        throw new Error("WebGPU tiny LLaMA unsupported resource step mutated state");
      }
      if (hostLlamaResourceSession.tokenExecutionAttempts !== 1) {
        throw new Error("WebGPU tiny LLaMA resource step did not reach the host token bridge");
      }
      expectLastLlamaTokenExecution(hostLlamaResourceSession, {
        tokens: [0],
        outputPolicy: executeOutputLogits,
        outputPtr: logits,
        outputLen: tinyLlamaVocabSize + 1,
      }, "WebGPU tiny LLaMA resource step");
      expectStatus(tokenAdvance(resourceSession, 0), unsupported, "WebGPU tiny LLaMA resource advance");
      if (sessionPosition(resourceSession) !== 0) {
        throw new Error("WebGPU tiny LLaMA unsupported resource advance mutated state");
      }
      if (hostLlamaResourceSession.tokenExecutionAttempts !== 2) {
        throw new Error("WebGPU tiny LLaMA resource advance did not reach the host token bridge");
      }
      expectLastLlamaTokenExecution(hostLlamaResourceSession, {
        tokens: [0],
        outputPolicy: executeOutputNone,
      }, "WebGPU tiny LLaMA resource advance");
      const tooManyResourceTokens = keep(alloc(5 * 4), 5 * 4);
      writeU32Array(tooManyResourceTokens, [0, 1, 2, 3, 4]);
      const tooManyResourceExecute = tokenExecute(resourceSession, tooManyResourceTokens, 5, executeOutputLogits, logits, tinyLlamaVocabSize + 1);
      expectStatus(tooManyResourceExecute.status, shapeMismatch, "WebGPU tiny LLaMA resource over-context execute");
      expectLastLlamaTokenExecution(hostLlamaResourceSession, {
        tokens: [0, 1, 2, 3, 4],
        outputPolicy: executeOutputLogits,
        outputPtr: logits,
        outputLen: tinyLlamaVocabSize + 1,
      }, "WebGPU tiny LLaMA resource over-context execute");
      const tooSmallResourceLogits = keep(alloc((tinyLlamaVocabSize - 1) * 4), (tinyLlamaVocabSize - 1) * 4);
      const tooSmallResourceExecute = tokenExecute(resourceSession, tooManyResourceTokens, 1, executeOutputLogits, tooSmallResourceLogits, tinyLlamaVocabSize - 1);
      expectStatus(tooSmallResourceExecute.status, shapeMismatch, "WebGPU tiny LLaMA resource too-small execute output");
      const invalidResourceToken = keep(alloc(4), 4);
      writeU32Array(invalidResourceToken, [tinyLlamaVocabSize]);
      const invalidResourceExecute = tokenExecute(resourceSession, invalidResourceToken, 1, executeOutputLogits, logits, tinyLlamaVocabSize + 1);
      expectStatus(invalidResourceExecute.status, shapeMismatch, "WebGPU tiny LLaMA resource invalid token execute");
      if (invalidResourceExecute.outputLen !== 0 || readF32Array(logits, 1)[0] !== -88 || sessionPosition(resourceSession) !== 0) {
        throw new Error("WebGPU tiny LLaMA invalid-token resource execute mutated state");
      }
      expectLastLlamaTokenExecution(hostLlamaResourceSession, {
        tokens: [tinyLlamaVocabSize],
        outputPolicy: executeOutputLogits,
        outputPtr: logits,
        outputLen: tinyLlamaVocabSize + 1,
      }, "WebGPU tiny LLaMA resource invalid token execute");
      const wrappedExports = hostRuntime.wrapExports(exportsRef);
      const proxyResourceTokens = keep(alloc(4), 4);
      writeU32Array(proxyResourceTokens, [0]);
      const proxyResourceExecute = tokenExecuteWith(wrappedExports, resourceSession, proxyResourceTokens, 1, executeOutputNone, 0, 0);
      expectStatus(proxyResourceExecute.status, unsupported, "WebGPU tiny LLaMA resource wrapped-export no-output execute");
      if (proxyResourceExecute.outputLen !== 0 || sessionPosition(resourceSession) !== 0) {
        throw new Error("WebGPU tiny LLaMA wrapped-export execute mutated state");
      }
      expectLastLlamaTokenExecution(hostLlamaResourceSession, {
        tokens: [0],
        outputPolicy: executeOutputNone,
      }, "WebGPU tiny LLaMA resource wrapped-export no-output execute");
      expectUnsupportedResourceSelection(resourceSession, hostLlamaResourceSession, "WebGPU tiny LLaMA resource");
      if (hostLlamaResourceSession.tokenExecutionAttempts !== 10) {
        throw new Error(`WebGPU tiny LLaMA resource selection/generation host attempts ${hostLlamaResourceSession.tokenExecutionAttempts}`);
      }
      expectHostLlamaResourceProfile(
        hostLlamaResourceSession.runtimeProfile(),
        expectedHostLlamaResourceProfile(tinyLlamaVocabSize),
        "WebGPU tiny LLaMA resource",
      );
      hostLlamaResourceSession.resetRuntimeProfile();
      expectHostLlamaResourceProfile(
        hostLlamaResourceSession.runtimeProfile(),
        emptyHostLlamaResourceProfile(),
        "WebGPU tiny LLaMA resource after profile reset",
      );
      expectNoRuntimeWork(resourceSession, "WebGPU tiny LLaMA unsupported resource execution");
    } finally {
      exportsRef.zgml_session_free(resourceSession);
      if (!hostLlamaResourceSession.freed || hostRuntime.lookup(resourceSession) !== null) {
        throw new Error("WebGPU tiny LLaMA host resource Session was not released through native free");
      }
      if (hostRuntime.hasClaimed(resourceSession)) {
        throw new Error("WebGPU tiny LLaMA host resource Session claim was not released through native free");
      }
    }

    const hostModelLlamaResourceSession = llamaResourceProgram.bind(rawLlamaResources, { model: compatibleModel });
    const modelResourceSession = hostModelLlamaResourceSession.handle;
    try {
      const modelSessionInspection = inspectSession(modelResourceSession);
      if (
        u32(modelSessionInspection, 0) !== tinyLlamaKind ||
        u32(modelSessionInspection, 4) !== backendWebGpu ||
        u32(modelSessionInspection, 8) !== bufferStorageExternalResource ||
        u32(modelSessionInspection, 12) !== bufferStorageExternalResource ||
        u64(modelSessionInspection, 16) !== 0n ||
        u64(modelSessionInspection, 24) !== 4n ||
        u64(modelSessionInspection, 32) === 0n ||
        u64(modelSessionInspection, 40) === 0n ||
        u64(modelSessionInspection, 48) !== 1n ||
        u64(modelSessionInspection, 56) === 0n ||
        u64(modelSessionInspection, 64) !== 3n ||
        u64(modelSessionInspection, 72) !== resourceBindingShapeHash
      ) {
        throw new Error("unexpected model-bound WebGPU tiny LLaMA resource session inspection");
      }
      expectHostResourceTableRoles(
        hostModelLlamaResourceSession.table,
        ["llama.output", "llama.k.0", "llama.v.0"],
        "model-bound WebGPU tiny LLaMA resource",
      );
      const modelLogits = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
      writeF32Array(modelLogits, Array(tinyLlamaVocabSize + 1).fill(-77));
      const modelStep = tokenStep(modelResourceSession, 0, modelLogits, tinyLlamaVocabSize + 1);
      expectStatus(modelStep.status, unsupported, "model-bound WebGPU tiny LLaMA resource step");
      if (modelStep.outputLen !== 0 || readF32Array(modelLogits, 1)[0] !== -77 || sessionPosition(modelResourceSession) !== 0) {
        throw new Error("model-bound WebGPU tiny LLaMA unsupported resource step mutated state");
      }
      if (hostModelLlamaResourceSession.tokenExecutionAttempts !== 1) {
        throw new Error("model-bound WebGPU tiny LLaMA resource step did not reach the host token bridge");
      }
      expectLastLlamaTokenExecution(hostModelLlamaResourceSession, {
        tokens: [0],
        outputPolicy: executeOutputLogits,
        outputPtr: modelLogits,
        outputLen: tinyLlamaVocabSize + 1,
      }, "model-bound WebGPU tiny LLaMA resource step");
      expectStatus(tokenAdvance(modelResourceSession, 0), unsupported, "model-bound WebGPU tiny LLaMA resource advance");
      if (sessionPosition(modelResourceSession) !== 0) {
        throw new Error("model-bound WebGPU tiny LLaMA unsupported resource advance mutated state");
      }
      if (hostModelLlamaResourceSession.tokenExecutionAttempts !== 2) {
        throw new Error("model-bound WebGPU tiny LLaMA resource advance did not reach the host token bridge");
      }
      expectLastLlamaTokenExecution(hostModelLlamaResourceSession, {
        tokens: [0],
        outputPolicy: executeOutputNone,
      }, "model-bound WebGPU tiny LLaMA resource advance");
      const modelTooManyResourceTokens = keep(alloc(5 * 4), 5 * 4);
      writeU32Array(modelTooManyResourceTokens, [0, 1, 2, 3, 4]);
      const modelTooManyResourceExecute = tokenExecute(modelResourceSession, modelTooManyResourceTokens, 5, executeOutputLogits, modelLogits, tinyLlamaVocabSize + 1);
      expectStatus(modelTooManyResourceExecute.status, shapeMismatch, "model-bound WebGPU tiny LLaMA resource over-context execute");
      expectLastLlamaTokenExecution(hostModelLlamaResourceSession, {
        tokens: [0, 1, 2, 3, 4],
        outputPolicy: executeOutputLogits,
        outputPtr: modelLogits,
        outputLen: tinyLlamaVocabSize + 1,
      }, "model-bound WebGPU tiny LLaMA resource over-context execute");
      const modelTooSmallResourceLogits = keep(alloc((tinyLlamaVocabSize - 1) * 4), (tinyLlamaVocabSize - 1) * 4);
      const modelTooSmallResourceExecute = tokenExecute(modelResourceSession, modelTooManyResourceTokens, 1, executeOutputLogits, modelTooSmallResourceLogits, tinyLlamaVocabSize - 1);
      expectStatus(modelTooSmallResourceExecute.status, shapeMismatch, "model-bound WebGPU tiny LLaMA resource too-small execute output");
      const modelInvalidResourceToken = keep(alloc(4), 4);
      writeU32Array(modelInvalidResourceToken, [tinyLlamaVocabSize]);
      const modelInvalidResourceExecute = tokenExecute(modelResourceSession, modelInvalidResourceToken, 1, executeOutputLogits, modelLogits, tinyLlamaVocabSize + 1);
      expectStatus(modelInvalidResourceExecute.status, shapeMismatch, "model-bound WebGPU tiny LLaMA resource invalid token execute");
      if (modelInvalidResourceExecute.outputLen !== 0 || readF32Array(modelLogits, 1)[0] !== -77 || sessionPosition(modelResourceSession) !== 0) {
        throw new Error("model-bound WebGPU tiny LLaMA invalid-token resource execute mutated state");
      }
      expectLastLlamaTokenExecution(hostModelLlamaResourceSession, {
        tokens: [tinyLlamaVocabSize],
        outputPolicy: executeOutputLogits,
        outputPtr: modelLogits,
        outputLen: tinyLlamaVocabSize + 1,
      }, "model-bound WebGPU tiny LLaMA resource invalid token execute");
      const wrappedModelExports = hostRuntime.wrapExports(exportsRef);
      const proxyModelResourceTokens = keep(alloc(4), 4);
      writeU32Array(proxyModelResourceTokens, [0]);
      const proxyModelResourceExecute = tokenExecuteWith(wrappedModelExports, modelResourceSession, proxyModelResourceTokens, 1, executeOutputNone, 0, 0);
      expectStatus(proxyModelResourceExecute.status, unsupported, "model-bound WebGPU tiny LLaMA resource wrapped-export no-output execute");
      if (proxyModelResourceExecute.outputLen !== 0 || sessionPosition(modelResourceSession) !== 0) {
        throw new Error("model-bound WebGPU tiny LLaMA wrapped-export execute mutated state");
      }
      expectLastLlamaTokenExecution(hostModelLlamaResourceSession, {
        tokens: [0],
        outputPolicy: executeOutputNone,
      }, "model-bound WebGPU tiny LLaMA resource wrapped-export no-output execute");
      expectUnsupportedResourceSelection(modelResourceSession, hostModelLlamaResourceSession, "model-bound WebGPU tiny LLaMA resource");
      if (hostModelLlamaResourceSession.tokenExecutionAttempts !== 10) {
        throw new Error(`model-bound WebGPU tiny LLaMA resource selection/generation host attempts ${hostModelLlamaResourceSession.tokenExecutionAttempts}`);
      }
      expectHostLlamaResourceProfile(
        hostModelLlamaResourceSession.runtimeProfile(),
        expectedHostLlamaResourceProfile(tinyLlamaVocabSize),
        "model-bound WebGPU tiny LLaMA resource",
      );
      hostModelLlamaResourceSession.resetRuntimeProfile();
      expectHostLlamaResourceProfile(
        hostModelLlamaResourceSession.runtimeProfile(),
        emptyHostLlamaResourceProfile(),
        "model-bound WebGPU tiny LLaMA resource after profile reset",
      );
      expectNoRuntimeWork(modelResourceSession, "model-bound WebGPU tiny LLaMA unsupported resource execution");
    } finally {
      exportsRef.zgml_session_free(modelResourceSession);
      if (!hostModelLlamaResourceSession.freed || hostRuntime.lookup(modelResourceSession) !== null) {
        throw new Error("model-bound WebGPU tiny LLaMA host resource Session was not released through native free");
      }
      if (hostRuntime.hasClaimed(modelResourceSession)) {
        throw new Error("model-bound WebGPU tiny LLaMA host resource Session claim was not released through native free");
      }
    }

    const directFreeLlamaResourceSession = llamaResourceProgram.bind(rawLlamaResources);
    const directFreeLlamaResourceHandle = directFreeLlamaResourceSession.handle;
    directFreeLlamaResourceSession.free();
    if (!directFreeLlamaResourceSession.freed || hostRuntime.lookup(directFreeLlamaResourceHandle) !== null) {
      throw new Error("WebGPU tiny LLaMA direct JS free did not release native resource Session");
    }
    if (hostRuntime.hasClaimed(directFreeLlamaResourceHandle)) {
      throw new Error("WebGPU tiny LLaMA direct JS free did not release resource Session claim");
    }
    expectStatus(
      hostRuntime.stepSync(directFreeLlamaResourceHandle, 0, 0),
      hostRuntime.status.unclaimed,
      "direct-freed WebGPU tiny LLaMA resource host callback step",
    );

    const hookExecutor = createHostLlamaTokenExecutor("WebGPU tiny LLaMA host token executor");
    const executableHostLlamaResourceProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      tokenExecutor: hookExecutor,
    });
    const hookSession = executableHostLlamaResourceProgram.bind(rawLlamaResources);
    const hookSessionHandle = hookSession.handle;
    try {
      const hookTokens = keep(alloc(2 * 4), 2 * 4);
      writeU32Array(hookTokens, [1, 2]);
      const hookLogits = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
      writeF32Array(hookLogits, Array(tinyLlamaVocabSize + 1).fill(-55));
      const hookExecute = tokenExecute(hookSessionHandle, hookTokens, 2, executeOutputLogits, hookLogits, tinyLlamaVocabSize + 1);
      expectStatus(hookExecute.status, ok, "WebGPU tiny LLaMA host-token executor caller-output execute");
      if (hookExecute.outputLen !== tinyLlamaVocabSize || sessionPosition(hookSessionHandle) !== 2) {
        throw new Error("WebGPU tiny LLaMA host-token executor did not advance caller-output execution");
      }
      expectClose(readF32Array(hookLogits, tinyLlamaVocabSize), [30, 31, 32, 33, 34, 35, 36, 37]);

      const hookAdvanceTokens = keep(alloc(4), 4);
      writeU32Array(hookAdvanceTokens, [3]);
      const hookAdvance = tokenExecute(hookSessionHandle, hookAdvanceTokens, 1, executeOutputNone, 0, 0);
      expectStatus(hookAdvance.status, ok, "WebGPU tiny LLaMA host-token executor no-output execute");
      if (hookAdvance.outputLen !== 0 || sessionPosition(hookSessionHandle) !== 3) {
        throw new Error("WebGPU tiny LLaMA host-token executor did not advance no-output execution");
      }

      const hookBoundTokens = keep(alloc(4), 4);
      writeU32Array(hookBoundTokens, [4]);
      const hookBound = tokenExecute(hookSessionHandle, hookBoundTokens, 1, executeOutputLogits, 0, 0);
      expectStatus(hookBound.status, ok, "WebGPU tiny LLaMA host-token executor bound-output execute");
      if (hookBound.outputLen !== tinyLlamaVocabSize || sessionPosition(hookSessionHandle) !== 4) {
        throw new Error("WebGPU tiny LLaMA host-token executor did not advance bound-output execution");
      }
      expectClose(Array.from(await hostDevice.readFloat32(hookSession.bindings.output, tinyLlamaVocabSize)), [43, 44, 45, 46, 47, 48, 49, 50]);

      const hookRejected = tokenExecute(hookSessionHandle, hookBoundTokens, 1, executeOutputNone, 0, 0);
      expectStatus(hookRejected.status, shapeMismatch, "WebGPU tiny LLaMA host-token executor over-context execute");
      if (hookRejected.outputLen !== 0 || sessionPosition(hookSessionHandle) !== 4) {
        throw new Error("WebGPU tiny LLaMA host-token executor over-context request mutated state");
      }
      const hookInspection = inspectSession(hookSessionHandle);
      if (u64(hookInspection, 16) !== 4n) {
        throw new Error("WebGPU tiny LLaMA host-token executor inspect position did not mirror host position");
      }
      hookExecutor.expect([
        { tokens: [1, 2], outputPolicy: executeOutputLogits, outputKind: "caller-logits", position: 0 },
        { tokens: [3], outputPolicy: executeOutputNone, outputKind: "none", position: 2 },
        { tokens: [4], outputPolicy: executeOutputLogits, outputKind: "bound-logits", position: 3 },
      ]);
      const hookLastStepParams = hookSession.stepParamsHistory.at(-1);
      if (
        hookSession.stepParamsHistory.length !== 3 ||
        !hookLastStepParams ||
        hookLastStepParams.endPosition !== 4 ||
        hookLastStepParams.outputKind !== "bound-logits"
      ) {
        throw new Error("WebGPU tiny LLaMA host-token executor did not retain StepParams evidence");
      }
      if (!hostDevice.canCreateGpuBuffers()) {
        hookExecutor.expectPatternScratchReuse(tinyLlamaVocabSize);
      }
      for (let i = 0; i < hookSession.stepParamsHistory.length; i += 1) {
        await expectHostLlamaKvWindow(hookSession, hookSession.stepParamsHistory[i], `WebGPU tiny LLaMA host-token executor KV write ${i}`);
      }
      const expectedHookBackendDispatchCount = hostDevice.canCreateGpuBuffers() ? 7 : 0;
      const expectedHookFallbackOpCount = hostDevice.canCreateGpuBuffers() ? 0 : 7;
      const hookProfile = hookSession.runtimeProfile();
      if (
        hookProfile.executorCallCount !== 3 ||
        hookProfile.executorOkCount !== 3 ||
        hookProfile.executorFailureCount !== 0 ||
        hookProfile.executorBackendOpCount !== expectedHookBackendDispatchCount ||
        hookProfile.executorBackendDispatchCount !== expectedHookBackendDispatchCount ||
        hookProfile.executorFallbackOpCount !== expectedHookFallbackOpCount ||
        hookProfile.executorCommandCount !== 7 ||
        hookProfile.executorCommandOpCount !== 7 ||
        hookProfile.validationFailureCount !== 1 ||
        hookProfile.unsupportedCallCount !== 0 ||
        hookProfile.lastStatus !== shapeMismatch
      ) {
        throw new Error(`unexpected WebGPU tiny LLaMA host-token executor profile: ${JSON.stringify(hookProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("host-token", hookProfile);
      const hookAbiProfile = sessionRuntimeProfile(hookSessionHandle);
      if (
        u64(hookAbiProfile, 0) !== 3n ||
        u64(hookAbiProfile, 8) !== BigInt(expectedHookBackendDispatchCount) ||
        u64(hookAbiProfile, 16) !== BigInt(expectedHookFallbackOpCount) ||
        u64(hookAbiProfile, 24) !== BigInt(expectedHookBackendDispatchCount) ||
        u64(hookAbiProfile, 32) !== 0n ||
        u64(hookAbiProfile, 40) !== 3n ||
        u64(hookAbiProfile, 48) !== 3n ||
        u64(hookAbiProfile, 56) !== 0n ||
        u64(hookAbiProfile, 96) !== 7n ||
        u64(hookAbiProfile, 112) !== 7n
      ) {
        throw new Error("unexpected WebGPU tiny LLaMA host-token executor ABI runtime profile");
      }
      resetSessionRuntimeProfile(hookSessionHandle);
      const resetHookProfile = hookSession.runtimeProfile();
      if (resetHookProfile.executorCallCount !== 0 || resetHookProfile.executorOkCount !== 0) {
        throw new Error("WebGPU tiny LLaMA host-token executor ABI profile reset did not clear host profile");
      }
      const resetHookAbiProfile = sessionRuntimeProfile(hookSessionHandle);
      if (u64(resetHookAbiProfile, 0) !== 0n || u64(resetHookAbiProfile, 24) !== 0n || u64(resetHookAbiProfile, 40) !== 0n) {
        throw new Error("WebGPU tiny LLaMA host-token executor ABI runtime profile did not reset");
      }
      expectStatus(exportsRef.zgml_session_reset(hookSessionHandle), ok, "WebGPU tiny LLaMA host-token executor raw ABI reset");
      if (sessionPosition(hookSessionHandle) !== 0) {
        throw new Error(`WebGPU tiny LLaMA host-token executor raw ABI reset left position ${sessionPosition(hookSessionHandle)}`);
      }
      const hookReplayTokens = keep(alloc(4), 4);
      writeU32Array(hookReplayTokens, [1]);
      const hookReplay = tokenExecute(hookSessionHandle, hookReplayTokens, 1, executeOutputNone, 0, 0);
      expectStatus(hookReplay.status, ok, "WebGPU tiny LLaMA host-token executor replay after raw ABI reset");
      if (hookReplay.outputLen !== 0 || sessionPosition(hookSessionHandle) !== 1) {
        throw new Error("WebGPU tiny LLaMA host-token executor replay after raw ABI reset did not restart at position zero");
      }
    } finally {
      exportsRef.zgml_session_free(hookSessionHandle);
      hookExecutor.destroy?.();
    }

    const modelHookExecutor = createHostLlamaTokenExecutor("model-bound WebGPU tiny LLaMA host token executor");
    const modelExecutableHostLlamaResourceProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      requiredModelResources: tinyLlamaModelResourceRequirements,
      allowExtraModelResources: false,
      tokenExecutor: modelHookExecutor,
    });
    let missingRequiredModelResourceRejected = false;
    try {
      modelExecutableHostLlamaResourceProgram.bind({
        ...rawLlamaResources,
        modelResources: llamaModelResources.slice(0, -1),
      }, { model: compatibleModel });
    } catch (err) {
      const message = String(err && err.message ? err.message : err);
      missingRequiredModelResourceRejected = message.includes("missing required") && message.includes(tinyLlamaModelResourceRoles.at(-1));
    }
    if (!missingRequiredModelResourceRejected) {
      throw new Error("expected model-bound WebGPU tiny LLaMA missing required model resource to reject");
    }
    let wrongDtypeModelResourceRejected = false;
    try {
      modelExecutableHostLlamaResourceProgram.bind({
        ...rawLlamaResources,
        modelResources: llamaModelResources.map((entry, index) => index === 0 ? { ...entry, dtype: "f16" } : entry),
      }, { model: compatibleModel });
    } catch (err) {
      wrongDtypeModelResourceRejected = String(err && err.message ? err.message : err).includes("dtype mismatch");
    }
    if (!wrongDtypeModelResourceRejected) {
      throw new Error("expected model-bound WebGPU tiny LLaMA wrong dtype model resource to reject");
    }
    let wrongShapeModelResourceRejected = false;
    try {
      modelExecutableHostLlamaResourceProgram.bind({
        ...rawLlamaResources,
        modelResources: llamaModelResources.map((entry, index) => index === 0 ? { ...entry, shape: [entry.elementCount] } : entry),
      }, { model: compatibleModel });
    } catch (err) {
      wrongShapeModelResourceRejected = String(err && err.message ? err.message : err).includes("shape mismatch");
    }
    if (!wrongShapeModelResourceRejected) {
      throw new Error("expected model-bound WebGPU tiny LLaMA wrong shape model resource to reject");
    }
    let wrongByteLengthModelResourceRejected = false;
    try {
      modelExecutableHostLlamaResourceProgram.bind({
        ...rawLlamaResources,
        modelResources: llamaModelResources.map((entry, index) => index === 0 ? { ...entry, byteLength: entry.byteLength - Float32Array.BYTES_PER_ELEMENT } : entry),
      }, { model: compatibleModel });
    } catch (err) {
      wrongByteLengthModelResourceRejected = String(err && err.message ? err.message : err).includes("byteLength mismatch");
    }
    if (!wrongByteLengthModelResourceRejected) {
      throw new Error("expected model-bound WebGPU tiny LLaMA wrong byteLength model resource to reject");
    }
    let extraModelResourceRejected = false;
    const extraLlamaModelResource = llamaResourceProgram.createModelResources([
      { role: "llama.weight.extra", shape: [1], dtype: "f32" },
    ], {
      usage: webgpuBufferUsage.STORAGE | webgpuBufferUsage.COPY_SRC | webgpuBufferUsage.COPY_DST,
    })[0];
    try {
      modelExecutableHostLlamaResourceProgram.bind({
        ...rawLlamaResources,
        modelResources: [...llamaModelResources, extraLlamaModelResource],
      }, { model: compatibleModel });
    } catch (err) {
      extraModelResourceRejected = String(err && err.message ? err.message : err).includes("unexpected LLaMA model resource");
    }
    if (!extraModelResourceRejected) {
      throw new Error("expected model-bound WebGPU tiny LLaMA extra model resource to reject");
    }
    const missingPreparedLayerExecutor = new WasmWebGpuLlamaBlockProjectionExecutor({ layer: 1 });
    const missingPreparedLayerProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      requiredModelResources: tinyLlamaModelResourceRequirements,
      allowExtraModelResources: false,
      tokenExecutor: missingPreparedLayerExecutor,
    });
    let missingPreparedLayerRejected = false;
    try {
      missingPreparedLayerProgram.bind({
        ...rawLlamaResources,
        modelResources: llamaModelResources,
      }, { model: compatibleModel });
    } catch (err) {
      const message = String(err && err.message ? err.message : err);
      missingPreparedLayerRejected = message.includes("needs LLaMA model resource") &&
        message.includes("llama.weight.model.layers.1.input_layernorm.weight");
    } finally {
      missingPreparedLayerExecutor.destroy?.();
    }
    if (!missingPreparedLayerRejected) {
      throw new Error("expected model-bound WebGPU tiny LLaMA missing prepared layer resource to reject at bind");
    }
    const malformedPreparedCacheExecutor = new WasmWebGpuLlamaKvProjectionExecutor({ layer: 0 });
    const malformedPreparedCacheProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      requiredModelResources: tinyLlamaModelResourceRequirements,
      allowExtraModelResources: false,
      tokenExecutor: malformedPreparedCacheExecutor,
    });
    const malformedKCacheByteLength = u32(kvRequirements, 20) + Float32Array.BYTES_PER_ELEMENT;
    const malformedKCacheResource = hostGpuBuffer(
      "tiny-llama.malformed-prepared-k-cache",
      malformedKCacheByteLength,
      webgpuBufferUsage.STORAGE | webgpuBufferUsage.COPY_SRC | webgpuBufferUsage.COPY_DST,
    );
    const malformedKCache = wrapHostResourceBuffer("webgpu", malformedKCacheResource, {
      access: "readwrite",
      byteLength: malformedKCacheByteLength,
    });
    const malformedKvCache = malformedPreparedCacheProgram.createKvCache({
      resource: ({ kind, layer }) => kind === "k" ? malformedKCache : rawLlamaResources.kvCache[layer].v,
    });
    let malformedPreparedCacheRejected = false;
    try {
      malformedPreparedCacheProgram.bind({
        output: rawLlamaResources.output,
        kvCache: malformedKvCache,
        modelResources: llamaModelResources,
      }, { model: compatibleModel });
    } catch (err) {
      malformedPreparedCacheRejected = String(err && err.message ? err.message : err).includes("element count is not divisible");
    } finally {
      malformedPreparedCacheExecutor.destroy?.();
    }
    if (!malformedPreparedCacheRejected) {
      throw new Error("expected model-bound WebGPU tiny LLaMA malformed prepared K cache to reject at bind");
    }
    const modelHookSession = modelExecutableHostLlamaResourceProgram.bind({
      ...rawLlamaResources,
      modelResources: llamaModelResources,
    }, { model: compatibleModel });
    const modelHookSessionHandle = modelHookSession.handle;
    try {
      if (modelHookSession.modelHandle !== compatibleModel || modelHookSession.modelBindingKind !== "compatible-model") {
        throw new Error("model-bound WebGPU tiny LLaMA host-token executor did not retain model identity");
      }
      if (
        !modelHookSession.modelTable ||
        !modelHookSession.modelManifest ||
        modelHookSession.modelManifest.hash === 0n ||
        modelHookSession.modelTable.hash === 0n ||
        modelHookSession.modelTable.descriptorHash === 0n
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA host-token executor did not retain model-resource table evidence");
      }
      if (
        !Object.isFrozen(modelHookSession.modelTable) ||
        !Object.isFrozen(modelHookSession.modelTable.descriptors) ||
        modelHookSession.modelTable.descriptors.some((entry) => !Object.isFrozen(entry)) ||
        !Object.isFrozen(modelHookSession.modelManifest) ||
        !Object.isFrozen(modelHookSession.modelManifest.entries) ||
        modelHookSession.modelManifest.entries.some((entry) => !Object.isFrozen(entry) || !Object.isFrozen(entry.shape)) ||
        !Object.isFrozen(modelHookSession.bindings) ||
        !Object.isFrozen(modelHookSession.bindings.kvCache) ||
        modelHookSession.bindings.kvCache.some((entry) => !Object.isFrozen(entry)) ||
        !Object.isFrozen(modelHookSession.modelResources) ||
        modelHookSession.modelResources.some((entry) => !Object.isFrozen(entry) || !Object.isFrozen(entry.shape))
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA model-resource evidence is not immutable");
      }
      let modelManifestMutationRejected = false;
      try {
        modelHookSession.modelManifest.entries[0].shape[0] = 999;
      } catch {
        modelManifestMutationRejected = true;
      }
      if (!modelManifestMutationRejected || modelHookSession.modelManifest.entries[0].shape[0] !== tinyLlamaTensorSpecs[0][1][0]) {
        throw new Error("model-bound WebGPU tiny LLaMA model manifest mutation was not rejected");
      }
      let modelResourceMutationRejected = false;
      try {
        modelHookSession.modelResources[0].shape[0] = 999;
      } catch {
        modelResourceMutationRejected = true;
      }
      if (!modelResourceMutationRejected || modelHookSession.modelResources[0].shape[0] !== tinyLlamaTensorSpecs[0][1][0]) {
        throw new Error("model-bound WebGPU tiny LLaMA model resource mutation was not rejected");
      }
      expectTinyLlamaModelResources(
        modelHookSession.modelResources,
        "model-bound WebGPU tiny LLaMA host-token executor",
      );
      await expectTinyLlamaModelResourceBytes(
        hostDevice,
        modelHookSession.modelResources,
        llamaModelResourceBytes,
        "model-bound WebGPU tiny LLaMA host-token executor checkpoint-backed model resource",
      );
      const modelHookTokens = keep(alloc(4), 4);
      writeU32Array(modelHookTokens, [0]);
      const modelHookAdvance = tokenExecute(modelHookSessionHandle, modelHookTokens, 1, executeOutputNone, 0, 0);
      expectStatus(modelHookAdvance.status, ok, "model-bound WebGPU tiny LLaMA host-token executor no-output execute");
      if (modelHookAdvance.outputLen !== 0 || sessionPosition(modelHookSessionHandle) !== 1) {
        throw new Error("model-bound WebGPU tiny LLaMA host-token executor did not advance no-output execution");
      }
      modelHookExecutor.expect([
        {
          tokens: [0],
          outputPolicy: executeOutputNone,
          outputKind: "none",
          position: 0,
          modelHandle: compatibleModel,
          modelResourceRoles: tinyLlamaModelResourceRoles,
        },
      ]);
      const modelHookStepParams = modelHookSession.stepParamsHistory.at(-1);
      if (
        modelHookSession.stepParamsHistory.length !== 1 ||
        !modelHookStepParams ||
        modelHookStepParams.startPosition !== 0 ||
        modelHookStepParams.endPosition !== 1 ||
        modelHookStepParams.outputKind !== "none"
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA host-token executor did not retain StepParams evidence");
      }
      await expectHostLlamaKvWindow(modelHookSession, modelHookStepParams, "model-bound WebGPU tiny LLaMA host-token executor KV write");
      const expectedModelHookBackendDispatchCount = hostDevice.canCreateGpuBuffers() ? 2 : 0;
      const expectedModelHookFallbackOpCount = hostDevice.canCreateGpuBuffers() ? 0 : 2;
      const modelHookProfile = modelHookSession.runtimeProfile();
      if (
        modelHookProfile.executorCallCount !== 1 ||
        modelHookProfile.executorOkCount !== 1 ||
        modelHookProfile.executorFailureCount !== 0 ||
        modelHookProfile.executorBackendOpCount !== expectedModelHookBackendDispatchCount ||
        modelHookProfile.executorBackendDispatchCount !== expectedModelHookBackendDispatchCount ||
        modelHookProfile.executorFallbackOpCount !== expectedModelHookFallbackOpCount ||
        modelHookProfile.executorCommandCount !== 2 ||
        modelHookProfile.executorCommandOpCount !== 2 ||
        modelHookProfile.validationFailureCount !== 0 ||
        modelHookProfile.unsupportedCallCount !== 0 ||
        modelHookProfile.lastStatus !== ok
      ) {
        throw new Error(`unexpected model-bound WebGPU tiny LLaMA host-token executor profile: ${JSON.stringify(modelHookProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("model-host-token", modelHookProfile);
      const modelHookAbiProfile = sessionRuntimeProfile(modelHookSessionHandle);
      if (
        u64(modelHookAbiProfile, 0) !== 1n ||
        u64(modelHookAbiProfile, 8) !== BigInt(expectedModelHookBackendDispatchCount) ||
        u64(modelHookAbiProfile, 16) !== BigInt(expectedModelHookFallbackOpCount) ||
        u64(modelHookAbiProfile, 24) !== BigInt(expectedModelHookBackendDispatchCount) ||
        u64(modelHookAbiProfile, 32) !== 0n ||
        u64(modelHookAbiProfile, 40) !== 1n ||
        u64(modelHookAbiProfile, 48) !== 1n ||
        u64(modelHookAbiProfile, 56) !== 0n ||
        u64(modelHookAbiProfile, 96) !== 2n ||
        u64(modelHookAbiProfile, 112) !== 2n
      ) {
        throw new Error("unexpected model-bound WebGPU tiny LLaMA host-token executor ABI runtime profile");
      }
      resetSessionRuntimeProfile(modelHookSessionHandle);
      const resetModelHookProfile = modelHookSession.runtimeProfile();
      if (resetModelHookProfile.executorCallCount !== 0 || resetModelHookProfile.executorOkCount !== 0) {
        throw new Error("model-bound WebGPU tiny LLaMA host-token executor ABI profile reset did not clear host profile");
      }
      const resetModelHookAbiProfile = sessionRuntimeProfile(modelHookSessionHandle);
      if (u64(resetModelHookAbiProfile, 0) !== 0n || u64(resetModelHookAbiProfile, 24) !== 0n || u64(resetModelHookAbiProfile, 40) !== 0n) {
        throw new Error("model-bound WebGPU tiny LLaMA host-token executor ABI runtime profile did not reset");
      }
    } finally {
      exportsRef.zgml_session_free(modelHookSessionHandle);
      modelHookExecutor.destroy?.();
    }

    const projectionExecutorCalls = [];
    const projectionExecutor = new WasmWebGpuLlamaEmbeddingProjectionExecutor({
      record(call) {
        projectionExecutorCalls.push(call);
      },
    });
    const projectionProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      requiredModelResources: tinyLlamaModelResourceRequirements,
      allowExtraModelResources: false,
      tokenExecutor: projectionExecutor,
    });
    const projectionSession = projectionProgram.bind({
      ...rawLlamaResources,
      modelResources: llamaModelResources,
    }, { model: compatibleModel });
    const projectionSessionHandle = projectionSession.handle;
    try {
      hostDevice.writeFloat32(projectionSession.bindings.output, Array(tinyLlamaVocabSize).fill(-321));
      const projectionTokens = keep(alloc(4), 4);
      writeU32Array(projectionTokens, [1]);
      const projectionExecute = tokenExecute(projectionSessionHandle, projectionTokens, 1, executeOutputLogits, 0, 0);
      expectStatus(projectionExecute.status, ok, "model-bound WebGPU tiny LLaMA embedding-projection executor execute");
      if (projectionExecute.outputLen !== tinyLlamaVocabSize || sessionPosition(projectionSessionHandle) !== 1) {
        throw new Error("model-bound WebGPU tiny LLaMA embedding-projection executor did not advance bound-logits execution");
      }
      expectClose(
        Array.from(await hostDevice.readFloat32(projectionSession.bindings.output, tinyLlamaVocabSize)),
        tinyLlamaEmbeddingProjectionLogits(llamaModelResourceBytes, 1),
      );
      const projectionFirstLogitsScratch = projectionExecutor.mockLogitsScratch;
      const projectionStepParams = projectionSession.stepParamsHistory.at(-1);
      if (
        projectionSession.stepParamsHistory.length !== 1 ||
        !projectionStepParams ||
        projectionStepParams.startPosition !== 0 ||
        projectionStepParams.endPosition !== 1 ||
        projectionStepParams.outputKind !== "bound-logits"
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA embedding-projection executor did not retain StepParams evidence");
      }
      const expectedProjectionBackendDispatchCount = hostDevice.canCreateGpuBuffers() ? 1 : 0;
      const expectedProjectionFallbackOpCount = hostDevice.canCreateGpuBuffers() ? 0 : 1;
      if (
        projectionExecutorCalls.length !== 1 ||
        projectionExecutorCalls[0].embeddingRole !== "llama.weight.model.embed_tokens.weight" ||
        projectionExecutorCalls[0].lmHeadRole !== "llama.weight.lm_head.weight" ||
        projectionExecutorCalls[0].hiddenSize !== 4 ||
        projectionExecutorCalls[0].token !== 1 ||
        projectionExecutorCalls[0].outputKind !== "bound-logits" ||
        projectionExecutorCalls[0].status !== ok ||
        projectionExecutorCalls[0].backendDispatchCount !== expectedProjectionBackendDispatchCount ||
        projectionExecutorCalls[0].fallbackOpCount !== expectedProjectionFallbackOpCount ||
        projectionExecutorCalls[0].commandCount !== 1 ||
        projectionExecutorCalls[0].modelManifestHash === 0n ||
        projectionExecutorCalls[0].modelTableHash === 0n ||
        projectionExecutorCalls[0].tableHash === 0n
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA embedding-projection executor evidence mismatch");
      }
      const projectionProfile = projectionSession.runtimeProfile();
      if (
        projectionProfile.executorCallCount !== 1 ||
        projectionProfile.executorOkCount !== 1 ||
        projectionProfile.executorFailureCount !== 0 ||
        projectionProfile.executorBackendOpCount !== expectedProjectionBackendDispatchCount ||
        projectionProfile.executorBackendDispatchCount !== expectedProjectionBackendDispatchCount ||
        projectionProfile.executorFallbackOpCount !== expectedProjectionFallbackOpCount ||
        projectionProfile.executorCommandCount !== 1 ||
        projectionProfile.executorCommandOpCount !== 1 ||
        projectionProfile.lastStatus !== ok ||
        projectionProfile.lastResultOutputLength !== tinyLlamaVocabSize
      ) {
        throw new Error(`unexpected model-bound WebGPU tiny LLaMA embedding-projection executor profile: ${JSON.stringify(projectionProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("embedding-projection", projectionProfile);
      const projectionAbiProfile = sessionRuntimeProfile(projectionSessionHandle);
      if (
        u64(projectionAbiProfile, 0) !== 1n ||
        u64(projectionAbiProfile, 8) !== BigInt(expectedProjectionBackendDispatchCount) ||
        u64(projectionAbiProfile, 16) !== BigInt(expectedProjectionFallbackOpCount) ||
        u64(projectionAbiProfile, 24) !== BigInt(expectedProjectionBackendDispatchCount) ||
        u64(projectionAbiProfile, 40) !== 1n ||
        u64(projectionAbiProfile, 48) !== 1n ||
        u64(projectionAbiProfile, 56) !== 0n ||
        u64(projectionAbiProfile, 96) !== 1n ||
        u64(projectionAbiProfile, 112) !== 1n
      ) {
        throw new Error("unexpected model-bound WebGPU tiny LLaMA embedding-projection executor ABI runtime profile");
      }
      hostDevice.writeFloat32(projectionSession.bindings.output, Array(tinyLlamaVocabSize).fill(-432));
      const projectionWindowTokens = keep(alloc(8), 8);
      writeU32Array(projectionWindowTokens, [0, 3]);
      const projectionWindowExecute = tokenExecute(projectionSessionHandle, projectionWindowTokens, 2, executeOutputLogits, 0, 0);
      expectStatus(projectionWindowExecute.status, ok, "model-bound WebGPU tiny LLaMA embedding-projection token-window execute");
      if (projectionWindowExecute.outputLen !== tinyLlamaVocabSize || sessionPosition(projectionSessionHandle) !== 3) {
        throw new Error("model-bound WebGPU tiny LLaMA embedding-projection executor did not advance token-window bound-logits execution");
      }
      expectClose(
        Array.from(await hostDevice.readFloat32(projectionSession.bindings.output, tinyLlamaVocabSize)),
        tinyLlamaEmbeddingProjectionLogits(llamaModelResourceBytes, 3),
      );
      if (
        !hostDevice.canCreateGpuBuffers() &&
        (
          projectionFirstLogitsScratch.length !== tinyLlamaVocabSize ||
          projectionExecutor.mockLogitsScratch !== projectionFirstLogitsScratch
        )
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA embedding-projection fallback did not reuse logits scratch");
      }
      const projectionWindowStepParams = projectionSession.stepParamsHistory.at(-1);
      if (
        projectionSession.stepParamsHistory.length !== 2 ||
        !projectionWindowStepParams ||
        projectionWindowStepParams.startPosition !== 1 ||
        projectionWindowStepParams.endPosition !== 3 ||
        projectionWindowStepParams.tokenCount !== 2 ||
        projectionWindowStepParams.outputKind !== "bound-logits"
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA embedding-projection token-window StepParams evidence mismatch");
      }
      if (
        projectionExecutorCalls.length !== 2 ||
        projectionExecutorCalls[1].token !== 3 ||
        projectionExecutorCalls[1].tokens.length !== 2 ||
        projectionExecutorCalls[1].tokens[0] !== 0 ||
        projectionExecutorCalls[1].tokens[1] !== 3 ||
        projectionExecutorCalls[1].tokenCount !== 2 ||
        projectionExecutorCalls[1].commandCount !== 1 ||
        projectionExecutorCalls[1].outputKind !== "bound-logits" ||
        projectionExecutorCalls[1].status !== ok
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA embedding-projection token-window executor evidence mismatch");
      }
      const expectedProjectionWindowBackendDispatchCount = hostDevice.canCreateGpuBuffers() ? 2 : 0;
      const expectedProjectionWindowFallbackOpCount = hostDevice.canCreateGpuBuffers() ? 0 : 2;
      const projectionWindowProfile = projectionSession.runtimeProfile();
      if (
        projectionWindowProfile.executorCallCount !== 2 ||
        projectionWindowProfile.executorOkCount !== 2 ||
        projectionWindowProfile.executorFailureCount !== 0 ||
        projectionWindowProfile.executorBackendOpCount !== expectedProjectionWindowBackendDispatchCount ||
        projectionWindowProfile.executorBackendDispatchCount !== expectedProjectionWindowBackendDispatchCount ||
        projectionWindowProfile.executorFallbackOpCount !== expectedProjectionWindowFallbackOpCount ||
        projectionWindowProfile.executorCommandCount !== 2 ||
        projectionWindowProfile.executorCommandOpCount !== 2 ||
        projectionWindowProfile.lastStatus !== ok ||
        projectionWindowProfile.lastResultOutputLength !== tinyLlamaVocabSize
      ) {
        throw new Error(`unexpected model-bound WebGPU tiny LLaMA embedding-projection token-window executor profile: ${JSON.stringify(projectionWindowProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("embedding-projection-window", projectionWindowProfile);
      const projectionWindowAbiProfile = sessionRuntimeProfile(projectionSessionHandle);
      if (
        u64(projectionWindowAbiProfile, 0) !== 2n ||
        u64(projectionWindowAbiProfile, 8) !== BigInt(expectedProjectionWindowBackendDispatchCount) ||
        u64(projectionWindowAbiProfile, 16) !== BigInt(expectedProjectionWindowFallbackOpCount) ||
        u64(projectionWindowAbiProfile, 24) !== BigInt(expectedProjectionWindowBackendDispatchCount) ||
        u64(projectionWindowAbiProfile, 40) !== 2n ||
        u64(projectionWindowAbiProfile, 48) !== 2n ||
        u64(projectionWindowAbiProfile, 56) !== 0n ||
        u64(projectionWindowAbiProfile, 96) !== 2n ||
        u64(projectionWindowAbiProfile, 112) !== 2n
      ) {
        throw new Error("unexpected model-bound WebGPU tiny LLaMA embedding-projection token-window ABI runtime profile");
      }
    } finally {
      exportsRef.zgml_session_free(projectionSessionHandle);
      projectionExecutor.destroy?.();
    }

    const rmsNormProjectionExecutorCalls = [];
    const rmsNormProjectionExecutor = new WasmWebGpuLlamaRmsNormProjectionExecutor({
      record(call) {
        rmsNormProjectionExecutorCalls.push(call);
      },
    });
    const rmsNormProjectionProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      requiredModelResources: tinyLlamaModelResourceRequirements,
      allowExtraModelResources: false,
      tokenExecutor: rmsNormProjectionExecutor,
    });
    const rmsNormProjectionSession = rmsNormProjectionProgram.bind({
      ...rawLlamaResources,
      modelResources: llamaModelResources,
    }, { model: compatibleModel });
    const rmsNormProjectionSessionHandle = rmsNormProjectionSession.handle;
    try {
      hostDevice.writeFloat32(rmsNormProjectionSession.bindings.output, Array(tinyLlamaVocabSize).fill(-654));
      const rmsNormProjectionTokens = keep(alloc(4), 4);
      writeU32Array(rmsNormProjectionTokens, [2]);
      const rmsNormProjectionExecute = tokenExecute(rmsNormProjectionSessionHandle, rmsNormProjectionTokens, 1, executeOutputLogits, 0, 0);
      expectStatus(rmsNormProjectionExecute.status, ok, "model-bound WebGPU tiny LLaMA RMSNorm-projection executor execute");
      if (rmsNormProjectionExecute.outputLen !== tinyLlamaVocabSize || sessionPosition(rmsNormProjectionSessionHandle) !== 1) {
        throw new Error("model-bound WebGPU tiny LLaMA RMSNorm-projection executor did not advance bound-logits execution");
      }
      expectClose(
        Array.from(await hostDevice.readFloat32(rmsNormProjectionSession.bindings.output, tinyLlamaVocabSize)),
        tinyLlamaRmsNormProjectionLogits(llamaModelResourceBytes, 2),
      );
      const rmsNormProjectionFirstLogitsScratch = rmsNormProjectionExecutor.mockLogitsScratch;
      const rmsNormProjectionStepParams = rmsNormProjectionSession.stepParamsHistory.at(-1);
      if (
        rmsNormProjectionSession.stepParamsHistory.length !== 1 ||
        !rmsNormProjectionStepParams ||
        rmsNormProjectionStepParams.startPosition !== 0 ||
        rmsNormProjectionStepParams.endPosition !== 1 ||
        rmsNormProjectionStepParams.outputKind !== "bound-logits"
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA RMSNorm-projection executor did not retain StepParams evidence");
      }
      const expectedRmsNormProjectionBackendDispatchCount = hostDevice.canCreateGpuBuffers() ? 1 : 0;
      const expectedRmsNormProjectionFallbackOpCount = hostDevice.canCreateGpuBuffers() ? 0 : 1;
      if (
        rmsNormProjectionExecutorCalls.length !== 1 ||
        rmsNormProjectionExecutorCalls[0].embeddingRole !== "llama.weight.model.embed_tokens.weight" ||
        rmsNormProjectionExecutorCalls[0].lmHeadRole !== "llama.weight.lm_head.weight" ||
        rmsNormProjectionExecutorCalls[0].normRole !== "llama.weight.model.norm.weight" ||
        rmsNormProjectionExecutorCalls[0].hiddenSize !== 4 ||
        rmsNormProjectionExecutorCalls[0].token !== 2 ||
        rmsNormProjectionExecutorCalls[0].outputKind !== "bound-logits" ||
        rmsNormProjectionExecutorCalls[0].status !== ok ||
        rmsNormProjectionExecutorCalls[0].backendDispatchCount !== expectedRmsNormProjectionBackendDispatchCount ||
        rmsNormProjectionExecutorCalls[0].fallbackOpCount !== expectedRmsNormProjectionFallbackOpCount ||
        rmsNormProjectionExecutorCalls[0].commandCount !== 1 ||
        rmsNormProjectionExecutorCalls[0].modelManifestHash === 0n ||
        rmsNormProjectionExecutorCalls[0].modelTableHash === 0n ||
        rmsNormProjectionExecutorCalls[0].tableHash === 0n
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA RMSNorm-projection executor evidence mismatch");
      }
      const rmsNormProjectionProfile = rmsNormProjectionSession.runtimeProfile();
      if (
        rmsNormProjectionProfile.executorCallCount !== 1 ||
        rmsNormProjectionProfile.executorOkCount !== 1 ||
        rmsNormProjectionProfile.executorFailureCount !== 0 ||
        rmsNormProjectionProfile.executorBackendOpCount !== expectedRmsNormProjectionBackendDispatchCount ||
        rmsNormProjectionProfile.executorBackendDispatchCount !== expectedRmsNormProjectionBackendDispatchCount ||
        rmsNormProjectionProfile.executorFallbackOpCount !== expectedRmsNormProjectionFallbackOpCount ||
        rmsNormProjectionProfile.executorCommandCount !== 1 ||
        rmsNormProjectionProfile.executorCommandOpCount !== 1 ||
        rmsNormProjectionProfile.lastStatus !== ok ||
        rmsNormProjectionProfile.lastResultOutputLength !== tinyLlamaVocabSize
      ) {
        throw new Error(`unexpected model-bound WebGPU tiny LLaMA RMSNorm-projection executor profile: ${JSON.stringify(rmsNormProjectionProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("rmsnorm-projection", rmsNormProjectionProfile);
      const rmsNormProjectionAbiProfile = sessionRuntimeProfile(rmsNormProjectionSessionHandle);
      if (
        u64(rmsNormProjectionAbiProfile, 0) !== 1n ||
        u64(rmsNormProjectionAbiProfile, 8) !== BigInt(expectedRmsNormProjectionBackendDispatchCount) ||
        u64(rmsNormProjectionAbiProfile, 16) !== BigInt(expectedRmsNormProjectionFallbackOpCount) ||
        u64(rmsNormProjectionAbiProfile, 24) !== BigInt(expectedRmsNormProjectionBackendDispatchCount) ||
        u64(rmsNormProjectionAbiProfile, 40) !== 1n ||
        u64(rmsNormProjectionAbiProfile, 48) !== 1n ||
        u64(rmsNormProjectionAbiProfile, 56) !== 0n ||
        u64(rmsNormProjectionAbiProfile, 96) !== 1n ||
        u64(rmsNormProjectionAbiProfile, 112) !== 1n
      ) {
        throw new Error("unexpected model-bound WebGPU tiny LLaMA RMSNorm-projection executor ABI runtime profile");
      }
      hostDevice.writeFloat32(rmsNormProjectionSession.bindings.output, Array(tinyLlamaVocabSize).fill(-765));
      const rmsNormProjectionWindowTokens = keep(alloc(8), 8);
      writeU32Array(rmsNormProjectionWindowTokens, [1, 3]);
      const rmsNormProjectionWindowExecute = tokenExecute(rmsNormProjectionSessionHandle, rmsNormProjectionWindowTokens, 2, executeOutputLogits, 0, 0);
      expectStatus(rmsNormProjectionWindowExecute.status, ok, "model-bound WebGPU tiny LLaMA RMSNorm-projection token-window execute");
      if (rmsNormProjectionWindowExecute.outputLen !== tinyLlamaVocabSize || sessionPosition(rmsNormProjectionSessionHandle) !== 3) {
        throw new Error("model-bound WebGPU tiny LLaMA RMSNorm-projection executor did not advance token-window bound-logits execution");
      }
      expectClose(
        Array.from(await hostDevice.readFloat32(rmsNormProjectionSession.bindings.output, tinyLlamaVocabSize)),
        tinyLlamaRmsNormProjectionLogits(llamaModelResourceBytes, 3),
      );
      if (
        !hostDevice.canCreateGpuBuffers() &&
        (
          rmsNormProjectionFirstLogitsScratch.length !== tinyLlamaVocabSize ||
          rmsNormProjectionExecutor.mockLogitsScratch !== rmsNormProjectionFirstLogitsScratch
        )
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA RMSNorm-projection fallback did not reuse logits scratch");
      }
      const rmsNormProjectionWindowStepParams = rmsNormProjectionSession.stepParamsHistory.at(-1);
      if (
        rmsNormProjectionSession.stepParamsHistory.length !== 2 ||
        !rmsNormProjectionWindowStepParams ||
        rmsNormProjectionWindowStepParams.startPosition !== 1 ||
        rmsNormProjectionWindowStepParams.endPosition !== 3 ||
        rmsNormProjectionWindowStepParams.tokenCount !== 2 ||
        rmsNormProjectionWindowStepParams.outputKind !== "bound-logits"
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA RMSNorm-projection token-window StepParams evidence mismatch");
      }
      if (
        rmsNormProjectionExecutorCalls.length !== 2 ||
        rmsNormProjectionExecutorCalls[1].token !== 3 ||
        rmsNormProjectionExecutorCalls[1].tokens.length !== 2 ||
        rmsNormProjectionExecutorCalls[1].tokens[0] !== 1 ||
        rmsNormProjectionExecutorCalls[1].tokens[1] !== 3 ||
        rmsNormProjectionExecutorCalls[1].tokenCount !== 2 ||
        rmsNormProjectionExecutorCalls[1].commandCount !== 1 ||
        rmsNormProjectionExecutorCalls[1].outputKind !== "bound-logits" ||
        rmsNormProjectionExecutorCalls[1].status !== ok
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA RMSNorm-projection token-window executor evidence mismatch");
      }
      const expectedRmsNormProjectionWindowBackendDispatchCount = hostDevice.canCreateGpuBuffers() ? 2 : 0;
      const expectedRmsNormProjectionWindowFallbackOpCount = hostDevice.canCreateGpuBuffers() ? 0 : 2;
      const rmsNormProjectionWindowProfile = rmsNormProjectionSession.runtimeProfile();
      if (
        rmsNormProjectionWindowProfile.executorCallCount !== 2 ||
        rmsNormProjectionWindowProfile.executorOkCount !== 2 ||
        rmsNormProjectionWindowProfile.executorFailureCount !== 0 ||
        rmsNormProjectionWindowProfile.executorBackendOpCount !== expectedRmsNormProjectionWindowBackendDispatchCount ||
        rmsNormProjectionWindowProfile.executorBackendDispatchCount !== expectedRmsNormProjectionWindowBackendDispatchCount ||
        rmsNormProjectionWindowProfile.executorFallbackOpCount !== expectedRmsNormProjectionWindowFallbackOpCount ||
        rmsNormProjectionWindowProfile.executorCommandCount !== 2 ||
        rmsNormProjectionWindowProfile.executorCommandOpCount !== 2 ||
        rmsNormProjectionWindowProfile.lastStatus !== ok ||
        rmsNormProjectionWindowProfile.lastResultOutputLength !== tinyLlamaVocabSize
      ) {
        throw new Error(`unexpected model-bound WebGPU tiny LLaMA RMSNorm-projection token-window executor profile: ${JSON.stringify(rmsNormProjectionWindowProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("rmsnorm-projection-window", rmsNormProjectionWindowProfile);
      const rmsNormProjectionWindowAbiProfile = sessionRuntimeProfile(rmsNormProjectionSessionHandle);
      if (
        u64(rmsNormProjectionWindowAbiProfile, 0) !== 2n ||
        u64(rmsNormProjectionWindowAbiProfile, 8) !== BigInt(expectedRmsNormProjectionWindowBackendDispatchCount) ||
        u64(rmsNormProjectionWindowAbiProfile, 16) !== BigInt(expectedRmsNormProjectionWindowFallbackOpCount) ||
        u64(rmsNormProjectionWindowAbiProfile, 24) !== BigInt(expectedRmsNormProjectionWindowBackendDispatchCount) ||
        u64(rmsNormProjectionWindowAbiProfile, 40) !== 2n ||
        u64(rmsNormProjectionWindowAbiProfile, 48) !== 2n ||
        u64(rmsNormProjectionWindowAbiProfile, 56) !== 0n ||
        u64(rmsNormProjectionWindowAbiProfile, 96) !== 2n ||
        u64(rmsNormProjectionWindowAbiProfile, 112) !== 2n
      ) {
        throw new Error("unexpected model-bound WebGPU tiny LLaMA RMSNorm-projection token-window ABI runtime profile");
      }
    } finally {
      exportsRef.zgml_session_free(rmsNormProjectionSessionHandle);
      rmsNormProjectionExecutor.destroy?.();
    }

    const kvProjectionExecutorCalls = [];
    const kvProjectionExecutor = new WasmWebGpuLlamaKvProjectionExecutor({
      record(call) {
        kvProjectionExecutorCalls.push(call);
      },
    });
    const kvProjectionProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      requiredModelResources: tinyLlamaModelResourceRequirements,
      allowExtraModelResources: false,
      tokenExecutor: kvProjectionExecutor,
    });
    const kvProjectionSession = kvProjectionProgram.bind({
      ...rawLlamaResources,
      modelResources: llamaModelResources,
    }, { model: compatibleModel });
    const kvProjectionSessionHandle = kvProjectionSession.handle;
    try {
      const kvProjectionK = kvProjectionSession.bindings.kvCache[0].k;
      const kvProjectionV = kvProjectionSession.bindings.kvCache[0].v;
      const kvProjectionKElements = Math.floor(kvProjectionK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const kvProjectionVElements = Math.floor(kvProjectionV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(kvProjectionK, Array(kvProjectionKElements).fill(-707));
      hostDevice.writeFloat32(kvProjectionV, Array(kvProjectionVElements).fill(-808));
      hostDevice.writeFloat32(kvProjectionSession.bindings.output, Array(tinyLlamaVocabSize).fill(-909));
      const kvProjectionBorrowedTokens = [3, tinyLlamaVocabSize + 1];
      kvProjectionBorrowedTokens[Symbol.iterator] = () => {
        throw new Error("LLaMA K/V projection executor should index the active tokensLen window");
      };
      const kvProjectionExecute = kvProjectionSession.executeDecodedTokensSyncInto({}, {
        outputLen: 0,
        outputPolicy: executeOutputNone,
        outputPtr: 0,
        tokens: kvProjectionBorrowedTokens,
        tokensLen: 1,
      }, 0, hostRuntime);
      expectStatus(kvProjectionExecute.status, ok, "model-bound WebGPU tiny LLaMA K/V projection executor no-output execute");
      if (kvProjectionExecute.outputLen !== 0 || sessionPosition(kvProjectionSessionHandle) !== 1) {
        throw new Error("model-bound WebGPU tiny LLaMA K/V projection executor did not advance no-output execution");
      }
      const expectedKvProjection = tinyLlamaKvProjectionCache(llamaModelResourceBytes, [3]);
      expectClose(
        Array.from(await hostDevice.readFloat32(kvProjectionK, expectedKvProjection.k.length, 0)),
        expectedKvProjection.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(kvProjectionV, expectedKvProjection.v.length, 0)),
        expectedKvProjection.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(kvProjectionK, expectedKvProjection.k.length, expectedKvProjection.k.length * Float32Array.BYTES_PER_ELEMENT)),
        Array(expectedKvProjection.k.length).fill(-707),
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(kvProjectionV, expectedKvProjection.v.length, expectedKvProjection.v.length * Float32Array.BYTES_PER_ELEMENT)),
        Array(expectedKvProjection.v.length).fill(-808),
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(kvProjectionSession.bindings.output, tinyLlamaVocabSize)),
        Array(tinyLlamaVocabSize).fill(-909),
      );
      const kvProjectionStepParams = kvProjectionSession.stepParamsHistory.at(-1);
      if (
        kvProjectionSession.stepParamsHistory.length !== 1 ||
        !kvProjectionStepParams ||
        kvProjectionStepParams.startPosition !== 0 ||
        kvProjectionStepParams.endPosition !== 1 ||
        kvProjectionStepParams.outputKind !== "none"
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA K/V projection executor did not retain StepParams evidence");
      }
      const expectedKvProjectionBackendDispatchCount = hostDevice.canCreateGpuBuffers() ? 1 : 0;
      const expectedKvProjectionFallbackOpCount = hostDevice.canCreateGpuBuffers() ? 0 : 1;
      if (
        kvProjectionExecutorCalls.length !== 1 ||
        kvProjectionExecutorCalls[0].embeddingRole !== "llama.weight.model.embed_tokens.weight" ||
        kvProjectionExecutorCalls[0].inputNormRole !== "llama.weight.model.layers.0.input_layernorm.weight" ||
        kvProjectionExecutorCalls[0].kProjRole !== "llama.weight.model.layers.0.self_attn.k_proj.weight" ||
        kvProjectionExecutorCalls[0].vProjRole !== "llama.weight.model.layers.0.self_attn.v_proj.weight" ||
        kvProjectionExecutorCalls[0].hiddenSize !== 4 ||
        kvProjectionExecutorCalls[0].kvStride !== 4 ||
        kvProjectionExecutorCalls[0].layer !== 0 ||
        kvProjectionExecutorCalls[0].tokens.length !== 1 ||
        kvProjectionExecutorCalls[0].tokens[0] !== 3 ||
        kvProjectionExecutorCalls[0].outputKind !== "none" ||
        kvProjectionExecutorCalls[0].status !== ok ||
        kvProjectionExecutorCalls[0].backendDispatchCount !== expectedKvProjectionBackendDispatchCount ||
        kvProjectionExecutorCalls[0].fallbackOpCount !== expectedKvProjectionFallbackOpCount ||
        kvProjectionExecutorCalls[0].commandCount !== 1 ||
        kvProjectionExecutorCalls[0].kvWriteCount !== 2 ||
        kvProjectionExecutorCalls[0].modelManifestHash === 0n ||
        kvProjectionExecutorCalls[0].modelTableHash === 0n ||
        kvProjectionExecutorCalls[0].tableHash === 0n
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA K/V projection executor evidence mismatch");
      }
      const kvProjectionProfile = kvProjectionSession.runtimeProfile();
      if (
        kvProjectionProfile.executorCallCount !== 1 ||
        kvProjectionProfile.executorOkCount !== 1 ||
        kvProjectionProfile.executorFailureCount !== 0 ||
        kvProjectionProfile.executorBackendOpCount !== expectedKvProjectionBackendDispatchCount ||
        kvProjectionProfile.executorBackendDispatchCount !== expectedKvProjectionBackendDispatchCount ||
        kvProjectionProfile.executorFallbackOpCount !== expectedKvProjectionFallbackOpCount ||
        kvProjectionProfile.executorCommandCount !== 1 ||
        kvProjectionProfile.executorCommandOpCount !== 1 ||
        kvProjectionProfile.lastStatus !== ok ||
        kvProjectionProfile.lastResultOutputLength !== 0
      ) {
        throw new Error(`unexpected model-bound WebGPU tiny LLaMA K/V projection executor profile: ${JSON.stringify(kvProjectionProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("kv-projection", kvProjectionProfile);
      const kvProjectionAbiProfile = sessionRuntimeProfile(kvProjectionSessionHandle);
      if (
        u64(kvProjectionAbiProfile, 0) !== 1n ||
        u64(kvProjectionAbiProfile, 8) !== BigInt(expectedKvProjectionBackendDispatchCount) ||
        u64(kvProjectionAbiProfile, 16) !== BigInt(expectedKvProjectionFallbackOpCount) ||
        u64(kvProjectionAbiProfile, 24) !== BigInt(expectedKvProjectionBackendDispatchCount) ||
        u64(kvProjectionAbiProfile, 32) !== 0n ||
        u64(kvProjectionAbiProfile, 40) !== 1n ||
        u64(kvProjectionAbiProfile, 48) !== 1n ||
        u64(kvProjectionAbiProfile, 56) !== 0n ||
        u64(kvProjectionAbiProfile, 96) !== 1n ||
        u64(kvProjectionAbiProfile, 112) !== 1n
      ) {
        throw new Error("unexpected model-bound WebGPU tiny LLaMA K/V projection executor ABI runtime profile");
      }
      const kvProjectionWindowTokens = keep(alloc(8), 8);
      writeU32Array(kvProjectionWindowTokens, [0, 2]);
      const kvProjectionWindowExecute = tokenExecute(kvProjectionSessionHandle, kvProjectionWindowTokens, 2, executeOutputNone, 0, 0);
      expectStatus(kvProjectionWindowExecute.status, ok, "model-bound WebGPU tiny LLaMA K/V projection executor token-window execute");
      if (kvProjectionWindowExecute.outputLen !== 0 || sessionPosition(kvProjectionSessionHandle) !== 3) {
        throw new Error("model-bound WebGPU tiny LLaMA K/V projection executor did not advance token-window no-output execution");
      }
      const expectedKvProjectionWindow = tinyLlamaKvProjectionCache(llamaModelResourceBytes, [0, 2]);
      expectClose(
        Array.from(await hostDevice.readFloat32(kvProjectionK, expectedKvProjectionWindow.k.length, expectedKvProjection.k.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedKvProjectionWindow.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(kvProjectionV, expectedKvProjectionWindow.v.length, expectedKvProjection.v.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedKvProjectionWindow.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(kvProjectionSession.bindings.output, tinyLlamaVocabSize)),
        Array(tinyLlamaVocabSize).fill(-909),
      );
      const kvProjectionWindowStepParams = kvProjectionSession.stepParamsHistory.at(-1);
      if (
        kvProjectionSession.stepParamsHistory.length !== 2 ||
        !kvProjectionWindowStepParams ||
        kvProjectionWindowStepParams.startPosition !== 1 ||
        kvProjectionWindowStepParams.endPosition !== 3 ||
        kvProjectionWindowStepParams.tokenCount !== 2 ||
        kvProjectionWindowStepParams.outputKind !== "none"
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA K/V projection token-window StepParams evidence mismatch");
      }
      if (
        kvProjectionExecutorCalls.length !== 2 ||
        kvProjectionExecutorCalls[1].tokens.length !== 2 ||
        kvProjectionExecutorCalls[1].tokens[0] !== 0 ||
        kvProjectionExecutorCalls[1].tokens[1] !== 2 ||
        kvProjectionExecutorCalls[1].tokenCount !== 2 ||
        kvProjectionExecutorCalls[1].outputKind !== "none" ||
        kvProjectionExecutorCalls[1].status !== ok ||
        kvProjectionExecutorCalls[1].commandCount !== 1 ||
        kvProjectionExecutorCalls[1].kvWriteCount !== 4
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA K/V projection token-window executor evidence mismatch");
      }
      const expectedKvProjectionWindowBackendDispatchCount = hostDevice.canCreateGpuBuffers() ? 2 : 0;
      const expectedKvProjectionWindowFallbackOpCount = hostDevice.canCreateGpuBuffers() ? 0 : 2;
      const expectedLastKvProjectionWindowBackendDispatchCount = hostDevice.canCreateGpuBuffers() ? 1 : 0;
      const expectedLastKvProjectionWindowFallbackOpCount = hostDevice.canCreateGpuBuffers() ? 0 : 1;
      const kvProjectionWindowProfile = kvProjectionSession.runtimeProfile();
      if (
        kvProjectionWindowProfile.executorCallCount !== 2 ||
        kvProjectionWindowProfile.executorOkCount !== 2 ||
        kvProjectionWindowProfile.executorFailureCount !== 0 ||
        kvProjectionWindowProfile.executorBackendOpCount !== expectedKvProjectionWindowBackendDispatchCount ||
        kvProjectionWindowProfile.executorBackendDispatchCount !== expectedKvProjectionWindowBackendDispatchCount ||
        kvProjectionWindowProfile.executorFallbackOpCount !== expectedKvProjectionWindowFallbackOpCount ||
        kvProjectionWindowProfile.executorCommandCount !== 2 ||
        kvProjectionWindowProfile.executorCommandOpCount !== 2 ||
        kvProjectionWindowProfile.lastStatus !== ok ||
        kvProjectionWindowProfile.lastResultOutputLength !== 0
      ) {
        throw new Error(`unexpected model-bound WebGPU tiny LLaMA K/V projection token-window executor profile: ${JSON.stringify(kvProjectionWindowProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("kv-projection-window", kvProjectionWindowProfile);
      const kvProjectionWindowAbiProfile = sessionRuntimeProfile(kvProjectionSessionHandle);
      if (
        u64(kvProjectionWindowAbiProfile, 0) !== 2n ||
        u64(kvProjectionWindowAbiProfile, 8) !== BigInt(expectedKvProjectionWindowBackendDispatchCount) ||
        u64(kvProjectionWindowAbiProfile, 16) !== BigInt(expectedKvProjectionWindowFallbackOpCount) ||
        u64(kvProjectionWindowAbiProfile, 24) !== BigInt(expectedKvProjectionWindowBackendDispatchCount) ||
        u64(kvProjectionWindowAbiProfile, 32) !== 0n ||
        u64(kvProjectionWindowAbiProfile, 40) !== 2n ||
        u64(kvProjectionWindowAbiProfile, 48) !== 2n ||
        u64(kvProjectionWindowAbiProfile, 56) !== 0n ||
        u64(kvProjectionWindowAbiProfile, 96) !== 2n ||
        u64(kvProjectionWindowAbiProfile, 112) !== 2n
      ) {
        throw new Error("unexpected model-bound WebGPU tiny LLaMA K/V projection token-window ABI runtime profile");
      }
      if (
        kvProjectionSession.tokenExecutorResultScratch.status !== ok ||
        kvProjectionSession.tokenExecutorResultScratch.outputLength !== 0 ||
        kvProjectionSession.tokenExecutorResultScratch.backendDispatchCount !== expectedLastKvProjectionWindowBackendDispatchCount ||
        kvProjectionSession.tokenExecutorResultScratch.commandCount !== 1 ||
        kvProjectionSession.tokenExecutorResultScratch.fallbackOpCount !== expectedLastKvProjectionWindowFallbackOpCount
      ) {
        throw new Error(
          `model-bound WebGPU tiny LLaMA K/V projection did not reuse executor result scratch: ` +
          `status=${kvProjectionSession.tokenExecutorResultScratch.status} ` +
          `outputLength=${kvProjectionSession.tokenExecutorResultScratch.outputLength} ` +
          `backendDispatchCount=${kvProjectionSession.tokenExecutorResultScratch.backendDispatchCount} ` +
          `commandCount=${kvProjectionSession.tokenExecutorResultScratch.commandCount} ` +
          `fallbackOpCount=${kvProjectionSession.tokenExecutorResultScratch.fallbackOpCount}`,
        );
      }
    } finally {
      exportsRef.zgml_session_free(kvProjectionSessionHandle);
      kvProjectionExecutor.destroy?.();
    }

    const qkNormKvProjectionTensorSpecs = [
      ...tinyLlamaRootTensorSpecs,
      ...tinyLlamaQkNormLayerTensorSpecs(0),
    ];
    const qkNormKvProjectionBytes = tinyLlamaModelResourceFileBytes(qkNormKvProjectionTensorSpecs);
    const qkNormKvProjectionExecutorCalls = [];
    const qkNormKvProjectionExecutor = new WasmWebGpuLlamaKvProjectionExecutor({
      attentionHeadSize: tinyLlamaDefaultShape.hiddenSize,
      qkProjectionNorm: true,
      record(call) {
        qkNormKvProjectionExecutorCalls.push(call);
      },
    });
    const qkNormKvProjectionProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      requiredModelResources: tinyLlamaModelResourceSpecs(qkNormKvProjectionBytes, { qkProjectionNorm: true }),
      allowExtraModelResources: false,
      tokenExecutor: qkNormKvProjectionExecutor,
    });
    const qkNormKvProjectionSession = qkNormKvProjectionProgram.bind({
      ...rawLlamaResources,
      modelResources: {
        safetensors: qkNormKvProjectionBytes,
        qkProjectionNorm: true,
      },
    }, { model: compatibleModel });
    const qkNormKvProjectionSessionHandle = qkNormKvProjectionSession.handle;
    try {
      const qkNormKvProjectionK = qkNormKvProjectionSession.bindings.kvCache[0].k;
      const qkNormKvProjectionV = qkNormKvProjectionSession.bindings.kvCache[0].v;
      const qkNormKvProjectionKElements = Math.floor(qkNormKvProjectionK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const qkNormKvProjectionVElements = Math.floor(qkNormKvProjectionV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(qkNormKvProjectionK, Array(qkNormKvProjectionKElements).fill(-757));
      hostDevice.writeFloat32(qkNormKvProjectionV, Array(qkNormKvProjectionVElements).fill(-858));
      const qkNormKvProjectionTokens = keep(alloc(4), 4);
      writeU32Array(qkNormKvProjectionTokens, [3]);
      const qkNormKvProjectionExecute = tokenExecute(qkNormKvProjectionSessionHandle, qkNormKvProjectionTokens, 1, executeOutputNone, 0, 0);
      expectStatus(qkNormKvProjectionExecute.status, ok, "model-bound WebGPU Q/K-norm LLaMA K/V projection executor no-output execute");
      if (qkNormKvProjectionExecute.outputLen !== 0 || sessionPosition(qkNormKvProjectionSessionHandle) !== 1) {
        throw new Error("model-bound WebGPU Q/K-norm LLaMA K/V projection executor did not advance no-output execution");
      }
      const expectedQkNormKvProjection = tinyLlamaKvProjectionCache(qkNormKvProjectionBytes, [3]);
      expectClose(
        Array.from(await hostDevice.readFloat32(qkNormKvProjectionK, expectedQkNormKvProjection.k.length, 0)),
        expectedQkNormKvProjection.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(qkNormKvProjectionV, expectedQkNormKvProjection.v.length, 0)),
        expectedQkNormKvProjection.v,
      );
      const expectedQkNormKvProjectionBackendDispatchCount = hostDevice.canCreateGpuBuffers() && hostDevice.canBindStorageBuffers(8) ? 1 : 0;
      const expectedQkNormKvProjectionFallbackOpCount = expectedQkNormKvProjectionBackendDispatchCount === 1 ? 0 : 1;
      if (
        qkNormKvProjectionExecutorCalls.length !== 1 ||
        qkNormKvProjectionExecutorCalls[0].qkProjectionNorm !== true ||
        qkNormKvProjectionExecutorCalls[0].attentionHeadSize !== tinyLlamaDefaultShape.hiddenSize ||
        qkNormKvProjectionExecutorCalls[0].kNormRole !== "llama.weight.model.layers.0.self_attn.k_norm.weight" ||
        qkNormKvProjectionExecutorCalls[0].backendDispatchCount !== expectedQkNormKvProjectionBackendDispatchCount ||
        qkNormKvProjectionExecutorCalls[0].fallbackOpCount !== expectedQkNormKvProjectionFallbackOpCount ||
        qkNormKvProjectionExecutorCalls[0].commandCount !== 1 ||
        qkNormKvProjectionExecutorCalls[0].kvWriteCount !== 2 ||
        qkNormKvProjectionExecutorCalls[0].status !== ok
      ) {
        throw new Error("model-bound WebGPU Q/K-norm LLaMA K/V projection executor evidence mismatch");
      }
      const qkNormKvProjectionProfile = qkNormKvProjectionSession.runtimeProfile();
      if (
        qkNormKvProjectionProfile.executorCallCount !== 1 ||
        qkNormKvProjectionProfile.executorOkCount !== 1 ||
        qkNormKvProjectionProfile.executorBackendDispatchCount !== expectedQkNormKvProjectionBackendDispatchCount ||
        qkNormKvProjectionProfile.executorFallbackOpCount !== expectedQkNormKvProjectionFallbackOpCount ||
        qkNormKvProjectionProfile.executorCommandCount !== 1 ||
        qkNormKvProjectionProfile.lastStatus !== ok
      ) {
        throw new Error(`unexpected model-bound WebGPU Q/K-norm LLaMA K/V projection profile: ${JSON.stringify(qkNormKvProjectionProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("kv-projection-qknorm", qkNormKvProjectionProfile);
    } finally {
      exportsRef.zgml_session_free(qkNormKvProjectionSessionHandle);
      qkNormKvProjectionExecutor.destroy?.();
    }

    const qkNormAttentionProjectionExecutorCalls = [];
    const qkNormAttentionProjectionExecutor = new WasmWebGpuLlamaAttentionProjectionExecutor({
      attentionHeadSize: tinyLlamaDefaultShape.hiddenSize,
      qkProjectionNorm: true,
      record(call) {
        qkNormAttentionProjectionExecutorCalls.push(call);
      },
    });
    const qkNormAttentionProjectionProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      requiredModelResources: tinyLlamaModelResourceSpecs(qkNormKvProjectionBytes, { qkProjectionNorm: true }),
      allowExtraModelResources: false,
      tokenExecutor: qkNormAttentionProjectionExecutor,
    });
    const qkNormAttentionProjectionSession = qkNormAttentionProjectionProgram.bind({
      ...rawLlamaResources,
      modelResources: {
        safetensors: qkNormKvProjectionBytes,
        qkProjectionNorm: true,
      },
    }, { model: compatibleModel });
    const qkNormAttentionProjectionSessionHandle = qkNormAttentionProjectionSession.handle;
    let qkNormAttentionProjectionModelPack = null;
    try {
      qkNormAttentionProjectionModelPack = qkNormAttentionProjectionExecutor.projectionForSession(qkNormAttentionProjectionSession).modelPack;
      const qkNormAttentionProjectionK = qkNormAttentionProjectionSession.bindings.kvCache[0].k;
      const qkNormAttentionProjectionV = qkNormAttentionProjectionSession.bindings.kvCache[0].v;
      const qkNormAttentionProjectionKElements = Math.floor(qkNormAttentionProjectionK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const qkNormAttentionProjectionVElements = Math.floor(qkNormAttentionProjectionV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(qkNormAttentionProjectionK, Array(qkNormAttentionProjectionKElements).fill(-959));
      hostDevice.writeFloat32(qkNormAttentionProjectionV, Array(qkNormAttentionProjectionVElements).fill(-969));
      hostDevice.writeFloat32(qkNormAttentionProjectionSession.bindings.output, Array(tinyLlamaVocabSize).fill(-979));
      const expectedQkNormAttentionCache = {
        k: Array(qkNormAttentionProjectionKElements).fill(-959),
        v: Array(qkNormAttentionProjectionVElements).fill(-969),
      };
      const qkNormAttentionSeedToken = keep(alloc(4), 4);
      writeU32Array(qkNormAttentionSeedToken, [1]);
      const qkNormAttentionSeed = tokenExecute(qkNormAttentionProjectionSessionHandle, qkNormAttentionSeedToken, 1, executeOutputNone, 0, 0);
      expectStatus(qkNormAttentionSeed.status, ok, "model-bound WebGPU Q/K-norm LLaMA attention projection no-output seed");
      if (qkNormAttentionSeed.outputLen !== 0 || sessionPosition(qkNormAttentionProjectionSessionHandle) !== 1) {
        throw new Error("model-bound WebGPU Q/K-norm LLaMA attention projection seed did not advance");
      }
      const expectedQkNormAttentionSeed = tinyLlamaAttentionProjectionStep(qkNormKvProjectionBytes, expectedQkNormAttentionCache, 1, 0);
      expectClose(
        Array.from(await hostDevice.readFloat32(qkNormAttentionProjectionK, expectedQkNormAttentionSeed.k.length, 0)),
        expectedQkNormAttentionSeed.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(qkNormAttentionProjectionV, expectedQkNormAttentionSeed.v.length, 0)),
        expectedQkNormAttentionSeed.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(qkNormAttentionProjectionSession.bindings.output, tinyLlamaVocabSize)),
        Array(tinyLlamaVocabSize).fill(-979),
      );
      const qkNormAttentionDecodeToken = keep(alloc(4), 4);
      writeU32Array(qkNormAttentionDecodeToken, [2]);
      const qkNormAttentionDecode = tokenExecute(qkNormAttentionProjectionSessionHandle, qkNormAttentionDecodeToken, 1, executeOutputLogits, 0, 0);
      expectStatus(qkNormAttentionDecode.status, ok, "model-bound WebGPU Q/K-norm LLaMA attention projection logits decode");
      if (qkNormAttentionDecode.outputLen !== tinyLlamaVocabSize || sessionPosition(qkNormAttentionProjectionSessionHandle) !== 2) {
        throw new Error("model-bound WebGPU Q/K-norm LLaMA attention projection decode did not advance bound-logits execution");
      }
      const expectedQkNormAttentionDecode = tinyLlamaAttentionProjectionStep(qkNormKvProjectionBytes, expectedQkNormAttentionCache, 2, 1);
      expectClose(
        Array.from(await hostDevice.readFloat32(qkNormAttentionProjectionSession.bindings.output, tinyLlamaVocabSize)),
        expectedQkNormAttentionDecode.logits,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(qkNormAttentionProjectionK, expectedQkNormAttentionDecode.k.length, expectedQkNormAttentionDecode.k.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedQkNormAttentionDecode.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(qkNormAttentionProjectionV, expectedQkNormAttentionDecode.v.length, expectedQkNormAttentionDecode.v.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedQkNormAttentionDecode.v,
      );
      const expectedQkNormAttentionBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 4) ? 2 : 0;
      const expectedQkNormAttentionFallbackOpCount = expectedQkNormAttentionBackendDispatchCount === 2 ? 0 : 2;
      if (
        qkNormAttentionProjectionExecutorCalls.length !== 2 ||
        qkNormAttentionProjectionExecutorCalls[1].qkProjectionNorm !== true ||
        qkNormAttentionProjectionExecutorCalls[1].attentionHeadSize !== tinyLlamaDefaultShape.hiddenSize ||
        qkNormAttentionProjectionExecutorCalls[1].qNormRole !== "llama.weight.model.layers.0.self_attn.q_norm.weight" ||
        qkNormAttentionProjectionExecutorCalls[1].kNormRole !== "llama.weight.model.layers.0.self_attn.k_norm.weight" ||
        qkNormAttentionProjectionExecutorCalls[1].modelPackFieldCount !== 10 ||
        qkNormAttentionProjectionModelPack.fieldCount !== 10 ||
        qkNormAttentionProjectionExecutorCalls[0].commandCount !== 1 ||
        qkNormAttentionProjectionExecutorCalls[1].commandCount !== 1 ||
        qkNormAttentionProjectionExecutorCalls[1].outputKind !== "bound-logits" ||
        qkNormAttentionProjectionExecutorCalls[1].status !== ok
      ) {
        throw new Error("model-bound WebGPU Q/K-norm LLaMA attention projection executor evidence mismatch");
      }
      const qkNormAttentionProjectionProfile = qkNormAttentionProjectionSession.runtimeProfile();
      if (
        qkNormAttentionProjectionProfile.executorCallCount !== 2 ||
        qkNormAttentionProjectionProfile.executorOkCount !== 2 ||
        qkNormAttentionProjectionProfile.executorBackendDispatchCount !== expectedQkNormAttentionBackendDispatchCount ||
        qkNormAttentionProjectionProfile.executorFallbackOpCount !== expectedQkNormAttentionFallbackOpCount ||
        qkNormAttentionProjectionProfile.executorCommandCount !== 2 ||
        qkNormAttentionProjectionProfile.lastStatus !== ok ||
        qkNormAttentionProjectionProfile.lastResultOutputLength !== tinyLlamaVocabSize
      ) {
        throw new Error(`unexpected model-bound WebGPU Q/K-norm LLaMA attention projection profile: ${JSON.stringify(qkNormAttentionProjectionProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("attention-projection-qknorm", qkNormAttentionProjectionProfile);
    } finally {
      exportsRef.zgml_session_free(qkNormAttentionProjectionSessionHandle);
      if (qkNormAttentionProjectionModelPack?.resource && "destroyed" in qkNormAttentionProjectionModelPack.resource && !qkNormAttentionProjectionModelPack.resource.destroyed) {
        throw new Error("model-bound WebGPU Q/K-norm LLaMA attention projection model pack was not destroyed on Session free");
      }
      qkNormAttentionProjectionExecutor.destroy?.();
    }

    const slidingAttentionConfig = mistralSlidingWindowProofConfig(groupedTinyLlamaShape);
    const slidingAttentionBytes = tinyLlamaModelResourceFileBytes(groupedTinyLlamaTwoLayerTensorSpecs, {
      metadata: { config: JSON.stringify(slidingAttentionConfig) },
    });
    const slidingAttentionShape = WasmWebGpuLlamaResourceProgram.inferSafetensorsShape(slidingAttentionBytes, {
      contextLength: groupedTinyLlamaShape.contextLength,
    });
    const slidingAttentionExecutorCalls = [];
    const slidingAttentionExecutor = new WasmWebGpuLlamaAttentionProjectionExecutor({
      attentionHeadSize: slidingAttentionShape.attentionHeadSize,
      epsilon: slidingAttentionShape.epsilon,
      ropeBase: slidingAttentionShape.ropeBase,
      ropeHighFreqFactor: slidingAttentionShape.ropeHighFreqFactor,
      ropeKind: slidingAttentionShape.ropeKind,
      ropeLowFreqFactor: slidingAttentionShape.ropeLowFreqFactor,
      ropeOriginalContextLength: slidingAttentionShape.ropeOriginalContextLength,
      ropeScale: slidingAttentionShape.ropeScale,
      slidingWindow: slidingAttentionShape.slidingWindow,
      record(call) {
        slidingAttentionExecutorCalls.push(call);
      },
    });
    const slidingAttentionProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      contextLength: groupedTinyLlamaShape.contextLength,
      safetensors: slidingAttentionBytes,
      allowMockFallback: true,
      tokenExecutor: slidingAttentionExecutor,
    });
    if (
      slidingAttentionShape.slidingWindow !== 2 ||
      slidingAttentionProgram.vocabSize !== groupedTinyLlamaShape.vocabSize ||
      slidingAttentionProgram.hiddenSize !== groupedTinyLlamaShape.hiddenSize ||
      slidingAttentionProgram.kvRequirements.layers !== groupedTinyLlamaShape.layers ||
      slidingAttentionProgram.kvRequirements.kBufferByteLength !== groupedTinyLlamaShape.contextLength * groupedTinyLlamaShape.kvSize * Float32Array.BYTES_PER_ELEMENT ||
      slidingAttentionExecutor.attentionHeadSize !== groupedTinyLlamaShape.attentionHeadSize ||
      slidingAttentionExecutor.slidingWindow !== 2 ||
      slidingAttentionExecutor.ropeBase !== 15000 ||
      slidingAttentionExecutor.epsilon !== 0.00006
    ) {
      throw new Error("host-only WebGPU Mistral standalone attention did not infer grouped sliding-window shape");
    }
    const slidingAttentionSession = slidingAttentionProgram.bindHostResources();
    const slidingAttentionSessionHandle = slidingAttentionSession.handle;
    const slidingAttentionHostExports = hostRuntime.wrapExports(exportsRef);
    let slidingAttentionModelPack = null;
    try {
      slidingAttentionModelPack = slidingAttentionExecutor.projectionForSession(slidingAttentionSession).modelPack;
      const slidingAttentionK = slidingAttentionSession.bindings.kvCache[0].k;
      const slidingAttentionV = slidingAttentionSession.bindings.kvCache[0].v;
      const slidingAttentionKElements = Math.floor(slidingAttentionK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const slidingAttentionVElements = Math.floor(slidingAttentionV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(slidingAttentionK, Array(slidingAttentionKElements).fill(-6810));
      hostDevice.writeFloat32(slidingAttentionV, Array(slidingAttentionVElements).fill(-6820));
      hostDevice.writeFloat32(slidingAttentionSession.bindings.output, Array(groupedTinyLlamaShape.vocabSize).fill(-6830));
      const slidingAttentionOptions = {
        attentionHeadSize: slidingAttentionShape.attentionHeadSize,
        epsilon: slidingAttentionShape.epsilon,
        ropeBase: slidingAttentionShape.ropeBase,
        ropeHighFreqFactor: slidingAttentionShape.ropeHighFreqFactor,
        ropeKind: slidingAttentionShape.ropeKind,
        ropeLowFreqFactor: slidingAttentionShape.ropeLowFreqFactor,
        ropeOriginalContextLength: slidingAttentionShape.ropeOriginalContextLength,
        ropeScale: slidingAttentionShape.ropeScale,
        slidingWindow: slidingAttentionShape.slidingWindow,
      };
      const expectedSlidingAttentionCache = {
        k: Array(slidingAttentionKElements).fill(-6810),
        v: Array(slidingAttentionVElements).fill(-6820),
      };
      const slidingAttentionPrefillTokens = keep(alloc(8), 8);
      writeU32Array(slidingAttentionPrefillTokens, [2, 3]);
      const slidingAttentionPrefill = tokenExecuteWith(slidingAttentionHostExports, slidingAttentionSessionHandle, slidingAttentionPrefillTokens, 2, executeOutputNone, 0, 0);
      expectStatus(slidingAttentionPrefill.status, ok, "host-only WebGPU Mistral standalone attention no-output prefill");
      if (slidingAttentionPrefill.outputLen !== 0 || sessionPositionWith(slidingAttentionHostExports, slidingAttentionSessionHandle) !== 2) {
        throw new Error("host-only WebGPU Mistral standalone attention prefill did not advance without output");
      }
      const expectedSlidingAttentionFirst = tinyLlamaAttentionProjectionStep(slidingAttentionBytes, expectedSlidingAttentionCache, 2, 0, slidingAttentionOptions);
      const expectedSlidingAttentionSecond = tinyLlamaAttentionProjectionStep(slidingAttentionBytes, expectedSlidingAttentionCache, 3, 1, slidingAttentionOptions);
      expectClose(
        Array.from(await hostDevice.readFloat32(slidingAttentionK, expectedSlidingAttentionFirst.k.length, 0)),
        expectedSlidingAttentionFirst.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(slidingAttentionV, expectedSlidingAttentionFirst.v.length, 0)),
        expectedSlidingAttentionFirst.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(slidingAttentionK, expectedSlidingAttentionSecond.k.length, expectedSlidingAttentionSecond.k.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedSlidingAttentionSecond.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(slidingAttentionV, expectedSlidingAttentionSecond.v.length, expectedSlidingAttentionSecond.v.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedSlidingAttentionSecond.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(slidingAttentionSession.bindings.output, groupedTinyLlamaShape.vocabSize)),
        Array(groupedTinyLlamaShape.vocabSize).fill(-6830),
      );
      const slidingAttentionPoisonK = Array(expectedSlidingAttentionFirst.k.length).fill(7.25);
      const slidingAttentionPoisonV = Array(expectedSlidingAttentionFirst.v.length).fill(-8.5);
      hostDevice.writeFloat32(slidingAttentionK, slidingAttentionPoisonK, 0);
      hostDevice.writeFloat32(slidingAttentionV, slidingAttentionPoisonV, 0);
      expectedSlidingAttentionCache.k.splice(0, slidingAttentionPoisonK.length, ...slidingAttentionPoisonK);
      expectedSlidingAttentionCache.v.splice(0, slidingAttentionPoisonV.length, ...slidingAttentionPoisonV);
      const slidingAttentionDecodeTokens = keep(alloc(4), 4);
      writeU32Array(slidingAttentionDecodeTokens, [5]);
      const slidingAttentionDecode = tokenExecuteWith(slidingAttentionHostExports, slidingAttentionSessionHandle, slidingAttentionDecodeTokens, 1, executeOutputLogits, 0, 0);
      expectStatus(slidingAttentionDecode.status, ok, "host-only WebGPU Mistral standalone attention logits decode");
      if (slidingAttentionDecode.outputLen !== groupedTinyLlamaShape.vocabSize || sessionPositionWith(slidingAttentionHostExports, slidingAttentionSessionHandle) !== 3) {
        throw new Error("host-only WebGPU Mistral standalone attention decode did not advance bound-logits execution");
      }
      const nonSlidingAttentionCache = {
        k: expectedSlidingAttentionCache.k.slice(),
        v: expectedSlidingAttentionCache.v.slice(),
      };
      const expectedSlidingAttentionDecode = tinyLlamaAttentionProjectionStep(slidingAttentionBytes, expectedSlidingAttentionCache, 5, 2, slidingAttentionOptions);
      const nonSlidingAttentionDecode = tinyLlamaAttentionProjectionStep(slidingAttentionBytes, nonSlidingAttentionCache, 5, 2, { ...slidingAttentionOptions, slidingWindow: null });
      if (!expectedSlidingAttentionDecode.logits.some((value, i) => Math.abs(value - nonSlidingAttentionDecode.logits[i]) > 1e-5)) {
        throw new Error("host-only WebGPU Mistral standalone attention sliding window did not change logits versus full-context attention");
      }
      expectClose(
        Array.from(await hostDevice.readFloat32(slidingAttentionSession.bindings.output, groupedTinyLlamaShape.vocabSize)),
        expectedSlidingAttentionDecode.logits,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(slidingAttentionK, expectedSlidingAttentionDecode.k.length, expectedSlidingAttentionDecode.k.length * 2 * Float32Array.BYTES_PER_ELEMENT)),
        expectedSlidingAttentionDecode.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(slidingAttentionV, expectedSlidingAttentionDecode.v.length, expectedSlidingAttentionDecode.v.length * 2 * Float32Array.BYTES_PER_ELEMENT)),
        expectedSlidingAttentionDecode.v,
      );
      const expectedSlidingAttentionBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 4) ? 3 : 0;
      const expectedSlidingAttentionFallbackOpCount = expectedSlidingAttentionBackendDispatchCount === 3 ? 0 : 3;
      if (
        slidingAttentionExecutorCalls.length !== 2 ||
        slidingAttentionExecutorCalls[0].tokenCount !== 2 ||
        slidingAttentionExecutorCalls[0].commandCount !== 2 ||
        slidingAttentionExecutorCalls[0].kvWriteCount !== 4 ||
        slidingAttentionExecutorCalls[0].outputKind !== "none" ||
        slidingAttentionExecutorCalls[1].token !== 5 ||
        slidingAttentionExecutorCalls[1].commandCount !== 1 ||
        slidingAttentionExecutorCalls[1].kvWriteCount !== 2 ||
        slidingAttentionExecutorCalls[1].outputKind !== "bound-logits" ||
        slidingAttentionExecutorCalls[1].slidingWindow !== 2 ||
        slidingAttentionExecutorCalls[1].queryHeadCount !== 4 ||
        slidingAttentionExecutorCalls[1].kvHeadCount !== 2 ||
        slidingAttentionExecutorCalls[1].modelPackFieldCount !== 8 ||
        slidingAttentionModelPack.fieldCount !== 8 ||
        slidingAttentionExecutorCalls[1].status !== ok
      ) {
        throw new Error("host-only WebGPU Mistral standalone attention executor evidence mismatch");
      }
      const slidingAttentionProfile = slidingAttentionSession.runtimeProfile();
      if (
        slidingAttentionProfile.executorCallCount !== 2 ||
        slidingAttentionProfile.executorOkCount !== 2 ||
        slidingAttentionProfile.executorBackendDispatchCount !== expectedSlidingAttentionBackendDispatchCount ||
        slidingAttentionProfile.executorFallbackOpCount !== expectedSlidingAttentionFallbackOpCount ||
        slidingAttentionProfile.executorCommandCount !== 3 ||
        slidingAttentionProfile.lastStatus !== ok ||
        slidingAttentionProfile.lastResultOutputLength !== groupedTinyLlamaShape.vocabSize
      ) {
        throw new Error(`unexpected host-only WebGPU Mistral standalone attention profile: ${JSON.stringify(slidingAttentionProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("attention-projection-sliding-window", slidingAttentionProfile);
    } finally {
      slidingAttentionHostExports.zgml_session_free(slidingAttentionSessionHandle);
      if (slidingAttentionSession.ownedResources.length !== 0) {
        throw new Error("host-only WebGPU Mistral standalone attention Session-owned resources were not released");
      }
      if (slidingAttentionModelPack?.resource && "destroyed" in slidingAttentionModelPack.resource && !slidingAttentionModelPack.resource.destroyed) {
        throw new Error("host-only WebGPU Mistral standalone attention model pack was not destroyed on Session free");
      }
      slidingAttentionExecutor.destroy?.();
    }

    const llama3AttentionConfig = llama3RopeProofConfig(groupedTinyLlamaShape);
    const llama3AttentionBytes = tinyLlamaModelResourceFileBytes(groupedTinyLlamaTwoLayerTensorSpecs);
    const llama3AttentionShape = WasmWebGpuLlamaResourceProgram.inferSafetensorsShape(llama3AttentionBytes, {
      config: llama3AttentionConfig,
      contextLength: groupedTinyLlamaShape.contextLength,
    });
    const llama3AttentionExecutorCalls = [];
    const llama3AttentionExecutor = new WasmWebGpuLlamaAttentionProjectionExecutor({
      attentionHeadSize: llama3AttentionShape.attentionHeadSize,
      epsilon: llama3AttentionShape.epsilon,
      ropeBase: llama3AttentionShape.ropeBase,
      ropeHighFreqFactor: llama3AttentionShape.ropeHighFreqFactor,
      ropeKind: llama3AttentionShape.ropeKind,
      ropeLowFreqFactor: llama3AttentionShape.ropeLowFreqFactor,
      ropeOriginalContextLength: llama3AttentionShape.ropeOriginalContextLength,
      ropeScale: llama3AttentionShape.ropeScale,
      slidingWindow: llama3AttentionShape.slidingWindow,
      record(call) {
        llama3AttentionExecutorCalls.push(call);
      },
    });
    const llama3AttentionProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      config: llama3AttentionConfig,
      contextLength: groupedTinyLlamaShape.contextLength,
      safetensors: llama3AttentionBytes,
      allowMockFallback: true,
      tokenExecutor: llama3AttentionExecutor,
    });
    if (
      llama3AttentionShape.ropeKind !== "llama3" ||
      llama3AttentionShape.ropeKindId !== 2 ||
      llama3AttentionShape.ropeScale !== 8 ||
      llama3AttentionShape.ropeLowFreqFactor !== 1 ||
      llama3AttentionShape.ropeHighFreqFactor !== 4 ||
      llama3AttentionShape.ropeOriginalContextLength !== 16 ||
      llama3AttentionProgram.kvRequirements.kBufferByteLength !== groupedTinyLlamaShape.contextLength * groupedTinyLlamaShape.kvSize * Float32Array.BYTES_PER_ELEMENT ||
      llama3AttentionExecutor.ropeKind !== "llama3" ||
      llama3AttentionExecutor.ropeScale !== 8 ||
      llama3AttentionExecutor.ropeBase !== 13500 ||
      llama3AttentionExecutor.epsilon !== 0.000045
    ) {
      throw new Error("host-only WebGPU Llama3 standalone attention did not infer grouped Llama3 RoPE shape");
    }
    const llama3AttentionSession = llama3AttentionProgram.bindHostResources();
    const llama3AttentionSessionHandle = llama3AttentionSession.handle;
    const llama3AttentionHostExports = hostRuntime.wrapExports(exportsRef);
    let llama3AttentionModelPack = null;
    try {
      llama3AttentionModelPack = llama3AttentionExecutor.projectionForSession(llama3AttentionSession).modelPack;
      const llama3AttentionK = llama3AttentionSession.bindings.kvCache[0].k;
      const llama3AttentionV = llama3AttentionSession.bindings.kvCache[0].v;
      const llama3AttentionKElements = Math.floor(llama3AttentionK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const llama3AttentionVElements = Math.floor(llama3AttentionV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(llama3AttentionK, Array(llama3AttentionKElements).fill(-6910));
      hostDevice.writeFloat32(llama3AttentionV, Array(llama3AttentionVElements).fill(-6920));
      hostDevice.writeFloat32(llama3AttentionSession.bindings.output, Array(groupedTinyLlamaShape.vocabSize).fill(-6930));
      const llama3AttentionOptions = {
        attentionHeadSize: llama3AttentionShape.attentionHeadSize,
        epsilon: llama3AttentionShape.epsilon,
        ropeBase: llama3AttentionShape.ropeBase,
        ropeHighFreqFactor: llama3AttentionShape.ropeHighFreqFactor,
        ropeKind: llama3AttentionShape.ropeKind,
        ropeLowFreqFactor: llama3AttentionShape.ropeLowFreqFactor,
        ropeOriginalContextLength: llama3AttentionShape.ropeOriginalContextLength,
        ropeScale: llama3AttentionShape.ropeScale,
      };
      const expectedLlama3AttentionCache = {
        k: Array(llama3AttentionKElements).fill(-6910),
        v: Array(llama3AttentionVElements).fill(-6920),
      };
      const llama3AttentionPrefillTokens = keep(alloc(4), 4);
      writeU32Array(llama3AttentionPrefillTokens, [2]);
      const llama3AttentionPrefill = tokenExecuteWith(llama3AttentionHostExports, llama3AttentionSessionHandle, llama3AttentionPrefillTokens, 1, executeOutputNone, 0, 0);
      expectStatus(llama3AttentionPrefill.status, ok, "host-only WebGPU Llama3 standalone attention no-output prefill");
      if (llama3AttentionPrefill.outputLen !== 0 || sessionPositionWith(llama3AttentionHostExports, llama3AttentionSessionHandle) !== 1) {
        throw new Error("host-only WebGPU Llama3 standalone attention prefill did not advance without output");
      }
      const expectedLlama3AttentionPrefill = tinyLlamaAttentionProjectionStep(llama3AttentionBytes, expectedLlama3AttentionCache, 2, 0, llama3AttentionOptions);
      expectClose(
        Array.from(await hostDevice.readFloat32(llama3AttentionK, expectedLlama3AttentionPrefill.k.length, 0)),
        expectedLlama3AttentionPrefill.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(llama3AttentionV, expectedLlama3AttentionPrefill.v.length, 0)),
        expectedLlama3AttentionPrefill.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(llama3AttentionSession.bindings.output, groupedTinyLlamaShape.vocabSize)),
        Array(groupedTinyLlamaShape.vocabSize).fill(-6930),
      );
      const llama3AttentionDecodeTokens = keep(alloc(4), 4);
      writeU32Array(llama3AttentionDecodeTokens, [5]);
      const llama3AttentionDecode = tokenExecuteWith(llama3AttentionHostExports, llama3AttentionSessionHandle, llama3AttentionDecodeTokens, 1, executeOutputLogits, 0, 0);
      expectStatus(llama3AttentionDecode.status, ok, "host-only WebGPU Llama3 standalone attention logits decode");
      if (llama3AttentionDecode.outputLen !== groupedTinyLlamaShape.vocabSize || sessionPositionWith(llama3AttentionHostExports, llama3AttentionSessionHandle) !== 2) {
        throw new Error("host-only WebGPU Llama3 standalone attention decode did not advance bound-logits execution");
      }
      const defaultRopeAttentionCache = {
        k: Array(llama3AttentionKElements).fill(-6910),
        v: Array(llama3AttentionVElements).fill(-6920),
      };
      const defaultRopeAttentionOptions = { ...llama3AttentionOptions, ropeKind: "none", ropeScale: 1 };
      tinyLlamaAttentionProjectionStep(llama3AttentionBytes, defaultRopeAttentionCache, 2, 0, defaultRopeAttentionOptions);
      const expectedLlama3AttentionDecode = tinyLlamaAttentionProjectionStep(llama3AttentionBytes, expectedLlama3AttentionCache, 5, 1, llama3AttentionOptions);
      const defaultRopeAttentionDecode = tinyLlamaAttentionProjectionStep(llama3AttentionBytes, defaultRopeAttentionCache, 5, 1, defaultRopeAttentionOptions);
      if (!expectedLlama3AttentionDecode.k.some((value, i) => Math.abs(value - defaultRopeAttentionDecode.k[i]) > 1e-5)) {
        throw new Error("host-only WebGPU Llama3 standalone attention RoPE scaling did not change K cache versus default RoPE");
      }
      expectClose(
        Array.from(await hostDevice.readFloat32(llama3AttentionSession.bindings.output, groupedTinyLlamaShape.vocabSize)),
        expectedLlama3AttentionDecode.logits,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(llama3AttentionK, expectedLlama3AttentionDecode.k.length, expectedLlama3AttentionDecode.k.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedLlama3AttentionDecode.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(llama3AttentionV, expectedLlama3AttentionDecode.v.length, expectedLlama3AttentionDecode.v.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedLlama3AttentionDecode.v,
      );
      const expectedLlama3AttentionBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 4) ? 2 : 0;
      const expectedLlama3AttentionFallbackOpCount = expectedLlama3AttentionBackendDispatchCount === 2 ? 0 : 2;
      if (
        llama3AttentionExecutorCalls.length !== 2 ||
        llama3AttentionExecutorCalls[0].commandCount !== 1 ||
        llama3AttentionExecutorCalls[0].kvWriteCount !== 2 ||
        llama3AttentionExecutorCalls[0].outputKind !== "none" ||
        llama3AttentionExecutorCalls[1].token !== 5 ||
        llama3AttentionExecutorCalls[1].commandCount !== 1 ||
        llama3AttentionExecutorCalls[1].kvWriteCount !== 2 ||
        llama3AttentionExecutorCalls[1].outputKind !== "bound-logits" ||
        llama3AttentionExecutorCalls[1].ropeKind !== "llama3" ||
        llama3AttentionExecutorCalls[1].ropeKindId !== 2 ||
        llama3AttentionExecutorCalls[1].ropeScale !== 8 ||
        llama3AttentionExecutorCalls[1].queryHeadCount !== 4 ||
        llama3AttentionExecutorCalls[1].kvHeadCount !== 2 ||
        llama3AttentionExecutorCalls[1].modelPackFieldCount !== 8 ||
        llama3AttentionModelPack.fieldCount !== 8 ||
        llama3AttentionExecutorCalls[1].status !== ok
      ) {
        throw new Error("host-only WebGPU Llama3 standalone attention executor evidence mismatch");
      }
      const llama3AttentionProfile = llama3AttentionSession.runtimeProfile();
      if (
        llama3AttentionProfile.executorCallCount !== 2 ||
        llama3AttentionProfile.executorOkCount !== 2 ||
        llama3AttentionProfile.executorBackendDispatchCount !== expectedLlama3AttentionBackendDispatchCount ||
        llama3AttentionProfile.executorFallbackOpCount !== expectedLlama3AttentionFallbackOpCount ||
        llama3AttentionProfile.executorCommandCount !== 2 ||
        llama3AttentionProfile.lastStatus !== ok ||
        llama3AttentionProfile.lastResultOutputLength !== groupedTinyLlamaShape.vocabSize
      ) {
        throw new Error(`unexpected host-only WebGPU Llama3 standalone attention profile: ${JSON.stringify(llama3AttentionProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("attention-projection-llama3-rope", llama3AttentionProfile);
    } finally {
      llama3AttentionHostExports.zgml_session_free(llama3AttentionSessionHandle);
      if (llama3AttentionSession.ownedResources.length !== 0) {
        throw new Error("host-only WebGPU Llama3 standalone attention Session-owned resources were not released");
      }
      if (llama3AttentionModelPack?.resource && "destroyed" in llama3AttentionModelPack.resource && !llama3AttentionModelPack.resource.destroyed) {
        throw new Error("host-only WebGPU Llama3 standalone attention model pack was not destroyed on Session free");
      }
      llama3AttentionExecutor.destroy?.();
    }

    const nopeAttentionConfig = smollm3NopeProofConfig(groupedTinyLlamaShape);
    const nopeAttentionBytes = tinyLlamaModelResourceFileBytes(groupedTinyLlamaTwoLayerTensorSpecs);
    const nopeAttentionShape = WasmWebGpuLlamaResourceProgram.inferSafetensorsShape(nopeAttentionBytes, {
      config: nopeAttentionConfig,
      contextLength: groupedTinyLlamaShape.contextLength,
    });
    const nopeAttentionExecutorCalls = [];
    const nopeAttentionExecutor = new WasmWebGpuLlamaAttentionProjectionExecutor({
      attentionHeadSize: nopeAttentionShape.attentionHeadSize,
      epsilon: nopeAttentionShape.epsilon,
      layer: 1,
      ropeBase: nopeAttentionShape.ropeBase,
      ropeHighFreqFactor: nopeAttentionShape.ropeHighFreqFactor,
      ropeKind: nopeAttentionShape.ropeKind,
      ropeLowFreqFactor: nopeAttentionShape.ropeLowFreqFactor,
      ropeOriginalContextLength: nopeAttentionShape.ropeOriginalContextLength,
      ropeScale: nopeAttentionShape.ropeScale,
      useRopeLayers: nopeAttentionShape.useRopeLayers,
      record(call) {
        nopeAttentionExecutorCalls.push(call);
      },
    });
    const nopeAttentionProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      config: nopeAttentionConfig,
      contextLength: groupedTinyLlamaShape.contextLength,
      safetensors: nopeAttentionBytes,
      allowMockFallback: true,
      tokenExecutor: nopeAttentionExecutor,
    });
    if (
      !Array.isArray(nopeAttentionShape.useRopeLayers) ||
      nopeAttentionShape.useRopeLayers.length !== groupedTinyLlamaShape.layers ||
      nopeAttentionShape.useRopeLayers[0] !== 1 ||
      nopeAttentionShape.useRopeLayers[1] !== 0 ||
      nopeAttentionProgram.kvRequirements.layers !== groupedTinyLlamaShape.layers ||
      nopeAttentionProgram.kvRequirements.kBufferByteLength !== groupedTinyLlamaShape.contextLength * groupedTinyLlamaShape.kvSize * Float32Array.BYTES_PER_ELEMENT ||
      nopeAttentionExecutor.layer !== 1 ||
      nopeAttentionExecutor.useRope !== false ||
      nopeAttentionExecutor.ropeBase !== 18000 ||
      nopeAttentionExecutor.epsilon !== 0.000085
    ) {
      throw new Error("host-only WebGPU SmolLM3 standalone attention did not infer grouped NoPE shape");
    }
    const nopeAttentionSession = nopeAttentionProgram.bindHostResources();
    const nopeAttentionSessionHandle = nopeAttentionSession.handle;
    const nopeAttentionHostExports = hostRuntime.wrapExports(exportsRef);
    let nopeAttentionModelPack = null;
    try {
      nopeAttentionModelPack = nopeAttentionExecutor.projectionForSession(nopeAttentionSession).modelPack;
      const nopeAttentionK = nopeAttentionSession.bindings.kvCache[1].k;
      const nopeAttentionV = nopeAttentionSession.bindings.kvCache[1].v;
      const nopeAttentionKElements = Math.floor(nopeAttentionK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const nopeAttentionVElements = Math.floor(nopeAttentionV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(nopeAttentionK, Array(nopeAttentionKElements).fill(-7010));
      hostDevice.writeFloat32(nopeAttentionV, Array(nopeAttentionVElements).fill(-7020));
      hostDevice.writeFloat32(nopeAttentionSession.bindings.output, Array(groupedTinyLlamaShape.vocabSize).fill(-7030));
      const nopeAttentionOptions = {
        attentionHeadSize: nopeAttentionShape.attentionHeadSize,
        epsilon: nopeAttentionShape.epsilon,
        layer: 1,
        ropeBase: nopeAttentionShape.ropeBase,
        ropeHighFreqFactor: nopeAttentionShape.ropeHighFreqFactor,
        ropeKind: nopeAttentionShape.ropeKind,
        ropeLowFreqFactor: nopeAttentionShape.ropeLowFreqFactor,
        ropeOriginalContextLength: nopeAttentionShape.ropeOriginalContextLength,
        ropeScale: nopeAttentionShape.ropeScale,
        useRope: false,
      };
      const expectedNopeAttentionCache = {
        k: Array(nopeAttentionKElements).fill(-7010),
        v: Array(nopeAttentionVElements).fill(-7020),
      };
      const nopeAttentionPrefillTokens = keep(alloc(4), 4);
      writeU32Array(nopeAttentionPrefillTokens, [2]);
      const nopeAttentionPrefill = tokenExecuteWith(nopeAttentionHostExports, nopeAttentionSessionHandle, nopeAttentionPrefillTokens, 1, executeOutputNone, 0, 0);
      expectStatus(nopeAttentionPrefill.status, ok, "host-only WebGPU SmolLM3 standalone attention no-output prefill");
      if (nopeAttentionPrefill.outputLen !== 0 || sessionPositionWith(nopeAttentionHostExports, nopeAttentionSessionHandle) !== 1) {
        throw new Error("host-only WebGPU SmolLM3 standalone attention prefill did not advance without output");
      }
      const expectedNopeAttentionPrefill = tinyLlamaAttentionProjectionStep(nopeAttentionBytes, expectedNopeAttentionCache, 2, 0, nopeAttentionOptions);
      expectClose(
        Array.from(await hostDevice.readFloat32(nopeAttentionK, expectedNopeAttentionPrefill.k.length, 0)),
        expectedNopeAttentionPrefill.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(nopeAttentionV, expectedNopeAttentionPrefill.v.length, 0)),
        expectedNopeAttentionPrefill.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(nopeAttentionSession.bindings.output, groupedTinyLlamaShape.vocabSize)),
        Array(groupedTinyLlamaShape.vocabSize).fill(-7030),
      );
      const nopeAttentionDecodeTokens = keep(alloc(4), 4);
      writeU32Array(nopeAttentionDecodeTokens, [5]);
      const nopeAttentionDecode = tokenExecuteWith(nopeAttentionHostExports, nopeAttentionSessionHandle, nopeAttentionDecodeTokens, 1, executeOutputLogits, 0, 0);
      expectStatus(nopeAttentionDecode.status, ok, "host-only WebGPU SmolLM3 standalone attention logits decode");
      if (nopeAttentionDecode.outputLen !== groupedTinyLlamaShape.vocabSize || sessionPositionWith(nopeAttentionHostExports, nopeAttentionSessionHandle) !== 2) {
        throw new Error("host-only WebGPU SmolLM3 standalone attention decode did not advance bound-logits execution");
      }
      const ropeAttentionCache = {
        k: Array(nopeAttentionKElements).fill(-7010),
        v: Array(nopeAttentionVElements).fill(-7020),
      };
      const ropeAttentionOptions = { ...nopeAttentionOptions, useRope: true };
      tinyLlamaAttentionProjectionStep(nopeAttentionBytes, ropeAttentionCache, 2, 0, ropeAttentionOptions);
      const expectedNopeAttentionDecode = tinyLlamaAttentionProjectionStep(nopeAttentionBytes, expectedNopeAttentionCache, 5, 1, nopeAttentionOptions);
      const ropeAttentionDecode = tinyLlamaAttentionProjectionStep(nopeAttentionBytes, ropeAttentionCache, 5, 1, ropeAttentionOptions);
      if (!expectedNopeAttentionDecode.k.some((value, i) => Math.abs(value - ropeAttentionDecode.k[i]) > 1e-5)) {
        throw new Error("host-only WebGPU SmolLM3 standalone attention NoPE did not change K cache versus RoPE");
      }
      expectClose(
        Array.from(await hostDevice.readFloat32(nopeAttentionSession.bindings.output, groupedTinyLlamaShape.vocabSize)),
        expectedNopeAttentionDecode.logits,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(nopeAttentionK, expectedNopeAttentionDecode.k.length, expectedNopeAttentionDecode.k.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedNopeAttentionDecode.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(nopeAttentionV, expectedNopeAttentionDecode.v.length, expectedNopeAttentionDecode.v.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedNopeAttentionDecode.v,
      );
      const expectedNopeAttentionBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 4) ? 2 : 0;
      const expectedNopeAttentionFallbackOpCount = expectedNopeAttentionBackendDispatchCount === 2 ? 0 : 2;
      if (
        nopeAttentionExecutorCalls.length !== 2 ||
        nopeAttentionExecutorCalls[0].commandCount !== 1 ||
        nopeAttentionExecutorCalls[0].kvWriteCount !== 2 ||
        nopeAttentionExecutorCalls[0].outputKind !== "none" ||
        nopeAttentionExecutorCalls[1].token !== 5 ||
        nopeAttentionExecutorCalls[1].commandCount !== 1 ||
        nopeAttentionExecutorCalls[1].kvWriteCount !== 2 ||
        nopeAttentionExecutorCalls[1].outputKind !== "bound-logits" ||
        nopeAttentionExecutorCalls[1].layer !== 1 ||
        nopeAttentionExecutorCalls[1].useRope !== false ||
        nopeAttentionExecutorCalls[1].queryHeadCount !== 4 ||
        nopeAttentionExecutorCalls[1].kvHeadCount !== 2 ||
        nopeAttentionExecutorCalls[1].modelPackFieldCount !== 8 ||
        nopeAttentionModelPack.fieldCount !== 8 ||
        nopeAttentionExecutorCalls[1].status !== ok
      ) {
        throw new Error("host-only WebGPU SmolLM3 standalone attention executor evidence mismatch");
      }
      const nopeAttentionProfile = nopeAttentionSession.runtimeProfile();
      if (
        nopeAttentionProfile.executorCallCount !== 2 ||
        nopeAttentionProfile.executorOkCount !== 2 ||
        nopeAttentionProfile.executorBackendDispatchCount !== expectedNopeAttentionBackendDispatchCount ||
        nopeAttentionProfile.executorFallbackOpCount !== expectedNopeAttentionFallbackOpCount ||
        nopeAttentionProfile.executorCommandCount !== 2 ||
        nopeAttentionProfile.lastStatus !== ok ||
        nopeAttentionProfile.lastResultOutputLength !== groupedTinyLlamaShape.vocabSize
      ) {
        throw new Error(`unexpected host-only WebGPU SmolLM3 standalone attention profile: ${JSON.stringify(nopeAttentionProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("attention-projection-smollm3-nope", nopeAttentionProfile);
    } finally {
      nopeAttentionHostExports.zgml_session_free(nopeAttentionSessionHandle);
      if (nopeAttentionSession.ownedResources.length !== 0) {
        throw new Error("host-only WebGPU SmolLM3 standalone attention Session-owned resources were not released");
      }
      if (nopeAttentionModelPack?.resource && "destroyed" in nopeAttentionModelPack.resource && !nopeAttentionModelPack.resource.destroyed) {
        throw new Error("host-only WebGPU SmolLM3 standalone attention model pack was not destroyed on Session free");
      }
      nopeAttentionExecutor.destroy?.();
    }

    const biasedQwen2AttentionConfig = biasedQwen2ProofConfig(groupedTinyLlamaShape);
    const biasedQwen2AttentionBytes = tinyLlamaModelResourceFileBytes(biasedGroupedTinyLlamaTwoLayerTensorSpecs);
    const unbiasedQwen2AttentionBytes = tinyLlamaModelResourceFileBytes(groupedTinyLlamaTwoLayerTensorSpecs);
    const biasedQwen2AttentionShape = WasmWebGpuLlamaResourceProgram.inferSafetensorsShape(biasedQwen2AttentionBytes, {
      config: biasedQwen2AttentionConfig,
      contextLength: groupedTinyLlamaShape.contextLength,
    });
    const biasedQwen2AttentionExecutorCalls = [];
    const biasedQwen2AttentionExecutor = new WasmWebGpuLlamaAttentionProjectionExecutor({
      attentionHeadSize: biasedQwen2AttentionShape.attentionHeadSize,
      epsilon: biasedQwen2AttentionShape.epsilon,
      layer: 0,
      ropeBase: biasedQwen2AttentionShape.ropeBase,
      ropeHighFreqFactor: biasedQwen2AttentionShape.ropeHighFreqFactor,
      ropeKind: biasedQwen2AttentionShape.ropeKind,
      ropeLowFreqFactor: biasedQwen2AttentionShape.ropeLowFreqFactor,
      ropeOriginalContextLength: biasedQwen2AttentionShape.ropeOriginalContextLength,
      ropeScale: biasedQwen2AttentionShape.ropeScale,
      record(call) {
        biasedQwen2AttentionExecutorCalls.push(call);
      },
    });
    const biasedQwen2AttentionProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      config: biasedQwen2AttentionConfig,
      contextLength: groupedTinyLlamaShape.contextLength,
      safetensors: biasedQwen2AttentionBytes,
      allowMockFallback: true,
      tokenExecutor: biasedQwen2AttentionExecutor,
    });
    if (
      biasedQwen2AttentionShape.attentionHeadSize !== groupedTinyLlamaShape.attentionHeadSize ||
      biasedQwen2AttentionProgram.kvRequirements.layers !== groupedTinyLlamaShape.layers ||
      biasedQwen2AttentionProgram.kvRequirements.kBufferByteLength !== groupedTinyLlamaShape.contextLength * groupedTinyLlamaShape.kvSize * Float32Array.BYTES_PER_ELEMENT ||
      biasedQwen2AttentionExecutor.layer !== 0 ||
      biasedQwen2AttentionExecutor.ropeBase !== 16500 ||
      biasedQwen2AttentionExecutor.epsilon !== 0.000075
    ) {
      throw new Error("host-only WebGPU biased Qwen2 standalone attention did not infer grouped biased shape");
    }
    const biasedQwen2AttentionSession = biasedQwen2AttentionProgram.bindHostResources();
    const biasedQwen2AttentionSessionHandle = biasedQwen2AttentionSession.handle;
    const biasedQwen2AttentionHostExports = hostRuntime.wrapExports(exportsRef);
    let biasedQwen2AttentionModelPack = null;
    try {
      biasedQwen2AttentionModelPack = biasedQwen2AttentionExecutor.projectionForSession(biasedQwen2AttentionSession).modelPack;
      const biasedQwen2AttentionK = biasedQwen2AttentionSession.bindings.kvCache[0].k;
      const biasedQwen2AttentionV = biasedQwen2AttentionSession.bindings.kvCache[0].v;
      const biasedQwen2AttentionKElements = Math.floor(biasedQwen2AttentionK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const biasedQwen2AttentionVElements = Math.floor(biasedQwen2AttentionV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(biasedQwen2AttentionK, Array(biasedQwen2AttentionKElements).fill(-7110));
      hostDevice.writeFloat32(biasedQwen2AttentionV, Array(biasedQwen2AttentionVElements).fill(-7120));
      hostDevice.writeFloat32(biasedQwen2AttentionSession.bindings.output, Array(groupedTinyLlamaShape.vocabSize).fill(-7130));
      const biasedQwen2AttentionOptions = {
        attentionHeadSize: biasedQwen2AttentionShape.attentionHeadSize,
        epsilon: biasedQwen2AttentionShape.epsilon,
        layer: 0,
        ropeBase: biasedQwen2AttentionShape.ropeBase,
        ropeHighFreqFactor: biasedQwen2AttentionShape.ropeHighFreqFactor,
        ropeKind: biasedQwen2AttentionShape.ropeKind,
        ropeLowFreqFactor: biasedQwen2AttentionShape.ropeLowFreqFactor,
        ropeOriginalContextLength: biasedQwen2AttentionShape.ropeOriginalContextLength,
        ropeScale: biasedQwen2AttentionShape.ropeScale,
      };
      const expectedBiasedQwen2AttentionCache = {
        k: Array(biasedQwen2AttentionKElements).fill(-7110),
        v: Array(biasedQwen2AttentionVElements).fill(-7120),
      };
      const biasedQwen2AttentionPrefillTokens = keep(alloc(4), 4);
      writeU32Array(biasedQwen2AttentionPrefillTokens, [2]);
      const biasedQwen2AttentionPrefill = tokenExecuteWith(biasedQwen2AttentionHostExports, biasedQwen2AttentionSessionHandle, biasedQwen2AttentionPrefillTokens, 1, executeOutputNone, 0, 0);
      expectStatus(biasedQwen2AttentionPrefill.status, ok, "host-only WebGPU biased Qwen2 standalone attention no-output prefill");
      if (biasedQwen2AttentionPrefill.outputLen !== 0 || sessionPositionWith(biasedQwen2AttentionHostExports, biasedQwen2AttentionSessionHandle) !== 1) {
        throw new Error("host-only WebGPU biased Qwen2 standalone attention prefill did not advance without output");
      }
      const expectedBiasedQwen2AttentionPrefill = tinyLlamaAttentionProjectionStep(biasedQwen2AttentionBytes, expectedBiasedQwen2AttentionCache, 2, 0, biasedQwen2AttentionOptions);
      expectClose(
        Array.from(await hostDevice.readFloat32(biasedQwen2AttentionK, expectedBiasedQwen2AttentionPrefill.k.length, 0)),
        expectedBiasedQwen2AttentionPrefill.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(biasedQwen2AttentionV, expectedBiasedQwen2AttentionPrefill.v.length, 0)),
        expectedBiasedQwen2AttentionPrefill.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(biasedQwen2AttentionSession.bindings.output, groupedTinyLlamaShape.vocabSize)),
        Array(groupedTinyLlamaShape.vocabSize).fill(-7130),
      );
      const biasedQwen2AttentionDecodeTokens = keep(alloc(4), 4);
      writeU32Array(biasedQwen2AttentionDecodeTokens, [5]);
      const biasedQwen2AttentionDecode = tokenExecuteWith(biasedQwen2AttentionHostExports, biasedQwen2AttentionSessionHandle, biasedQwen2AttentionDecodeTokens, 1, executeOutputLogits, 0, 0);
      expectStatus(biasedQwen2AttentionDecode.status, ok, "host-only WebGPU biased Qwen2 standalone attention logits decode");
      if (biasedQwen2AttentionDecode.outputLen !== groupedTinyLlamaShape.vocabSize || sessionPositionWith(biasedQwen2AttentionHostExports, biasedQwen2AttentionSessionHandle) !== 2) {
        throw new Error("host-only WebGPU biased Qwen2 standalone attention decode did not advance bound-logits execution");
      }
      const unbiasedQwen2AttentionCache = {
        k: Array(biasedQwen2AttentionKElements).fill(-7110),
        v: Array(biasedQwen2AttentionVElements).fill(-7120),
      };
      tinyLlamaAttentionProjectionStep(unbiasedQwen2AttentionBytes, unbiasedQwen2AttentionCache, 2, 0, biasedQwen2AttentionOptions);
      const expectedBiasedQwen2AttentionDecode = tinyLlamaAttentionProjectionStep(biasedQwen2AttentionBytes, expectedBiasedQwen2AttentionCache, 5, 1, biasedQwen2AttentionOptions);
      const unbiasedQwen2AttentionDecode = tinyLlamaAttentionProjectionStep(unbiasedQwen2AttentionBytes, unbiasedQwen2AttentionCache, 5, 1, biasedQwen2AttentionOptions);
      if (!expectedBiasedQwen2AttentionDecode.logits.some((value, i) => Math.abs(value - unbiasedQwen2AttentionDecode.logits[i]) > 1e-5)) {
        throw new Error("host-only WebGPU biased Qwen2 standalone attention bias did not change logits versus unbiased reference");
      }
      expectClose(
        Array.from(await hostDevice.readFloat32(biasedQwen2AttentionSession.bindings.output, groupedTinyLlamaShape.vocabSize)),
        expectedBiasedQwen2AttentionDecode.logits,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(biasedQwen2AttentionK, expectedBiasedQwen2AttentionDecode.k.length, expectedBiasedQwen2AttentionDecode.k.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedBiasedQwen2AttentionDecode.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(biasedQwen2AttentionV, expectedBiasedQwen2AttentionDecode.v.length, expectedBiasedQwen2AttentionDecode.v.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedBiasedQwen2AttentionDecode.v,
      );
      const expectedBiasedQwen2AttentionBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 4) ? 2 : 0;
      const expectedBiasedQwen2AttentionFallbackOpCount = expectedBiasedQwen2AttentionBackendDispatchCount === 2 ? 0 : 2;
      if (
        biasedQwen2AttentionExecutorCalls.length !== 2 ||
        biasedQwen2AttentionExecutorCalls[0].commandCount !== 1 ||
        biasedQwen2AttentionExecutorCalls[0].kvWriteCount !== 2 ||
        biasedQwen2AttentionExecutorCalls[0].outputKind !== "none" ||
        biasedQwen2AttentionExecutorCalls[1].token !== 5 ||
        biasedQwen2AttentionExecutorCalls[1].commandCount !== 1 ||
        biasedQwen2AttentionExecutorCalls[1].kvWriteCount !== 2 ||
        biasedQwen2AttentionExecutorCalls[1].outputKind !== "bound-logits" ||
        biasedQwen2AttentionExecutorCalls[1].layer !== 0 ||
        biasedQwen2AttentionExecutorCalls[1].biasMask !== 143 ||
        biasedQwen2AttentionExecutorCalls[1].queryHeadCount !== 4 ||
        biasedQwen2AttentionExecutorCalls[1].kvHeadCount !== 2 ||
        biasedQwen2AttentionExecutorCalls[1].modelPackFieldCount !== 13 ||
        biasedQwen2AttentionModelPack.fieldCount !== 13 ||
        biasedQwen2AttentionExecutorCalls[1].status !== ok
      ) {
        throw new Error("host-only WebGPU biased Qwen2 standalone attention executor evidence mismatch");
      }
      const biasedQwen2AttentionProfile = biasedQwen2AttentionSession.runtimeProfile();
      if (
        biasedQwen2AttentionProfile.executorCallCount !== 2 ||
        biasedQwen2AttentionProfile.executorOkCount !== 2 ||
        biasedQwen2AttentionProfile.executorBackendDispatchCount !== expectedBiasedQwen2AttentionBackendDispatchCount ||
        biasedQwen2AttentionProfile.executorFallbackOpCount !== expectedBiasedQwen2AttentionFallbackOpCount ||
        biasedQwen2AttentionProfile.executorCommandCount !== 2 ||
        biasedQwen2AttentionProfile.lastStatus !== ok ||
        biasedQwen2AttentionProfile.lastResultOutputLength !== groupedTinyLlamaShape.vocabSize
      ) {
        throw new Error(`unexpected host-only WebGPU biased Qwen2 standalone attention profile: ${JSON.stringify(biasedQwen2AttentionProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("attention-projection-biased-qwen2", biasedQwen2AttentionProfile);
    } finally {
      biasedQwen2AttentionHostExports.zgml_session_free(biasedQwen2AttentionSessionHandle);
      if (biasedQwen2AttentionSession.ownedResources.length !== 0) {
        throw new Error("host-only WebGPU biased Qwen2 standalone attention Session-owned resources were not released");
      }
      if (biasedQwen2AttentionModelPack?.resource && "destroyed" in biasedQwen2AttentionModelPack.resource && !biasedQwen2AttentionModelPack.resource.destroyed) {
        throw new Error("host-only WebGPU biased Qwen2 standalone attention model pack was not destroyed on Session free");
      }
      biasedQwen2AttentionExecutor.destroy?.();
    }

    const attentionProjectionExecutorCalls = [];
    const attentionProjectionExecutor = new WasmWebGpuLlamaAttentionProjectionExecutor({
      record(call) {
        attentionProjectionExecutorCalls.push(call);
      },
    });
    const attentionProjectionProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      requiredModelResources: tinyLlamaModelResourceRequirements,
      allowExtraModelResources: false,
      tokenExecutor: attentionProjectionExecutor,
    });
    const attentionProjectionSession = attentionProjectionProgram.bind({
      ...rawLlamaResources,
      modelResources: llamaModelResources,
    }, { model: compatibleModel });
    const attentionProjectionSessionHandle = attentionProjectionSession.handle;
    let attentionProjectionModelPack = null;
    try {
      attentionProjectionModelPack = attentionProjectionExecutor.projectionForSession(attentionProjectionSession).modelPack;
      const attentionProjectionK = attentionProjectionSession.bindings.kvCache[0].k;
      const attentionProjectionV = attentionProjectionSession.bindings.kvCache[0].v;
      const attentionProjectionKElements = Math.floor(attentionProjectionK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const attentionProjectionVElements = Math.floor(attentionProjectionV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(attentionProjectionK, Array(attentionProjectionKElements).fill(-1111));
      hostDevice.writeFloat32(attentionProjectionV, Array(attentionProjectionVElements).fill(-2222));
      hostDevice.writeFloat32(attentionProjectionSession.bindings.output, Array(tinyLlamaVocabSize).fill(-3333));
      const expectedAttentionCache = {
        k: Array(attentionProjectionKElements).fill(-1111),
        v: Array(attentionProjectionVElements).fill(-2222),
      };
      const attentionBorrowedTokenOne = [1, tinyLlamaVocabSize + 1];
      attentionBorrowedTokenOne[Symbol.iterator] = () => {
        throw new Error("LLaMA attention projection executor should index the active tokensLen window");
      };
      const attentionSeed = attentionProjectionSession.executeDecodedTokensSyncInto({}, {
        outputLen: 0,
        outputPolicy: executeOutputNone,
        outputPtr: 0,
        tokens: attentionBorrowedTokenOne,
        tokensLen: 1,
      }, 0, hostRuntime);
      expectStatus(attentionSeed.status, ok, "model-bound WebGPU tiny LLaMA attention projection no-output seed");
      if (attentionSeed.outputLen !== 0 || sessionPosition(attentionProjectionSessionHandle) !== 1) {
        throw new Error("model-bound WebGPU tiny LLaMA attention projection seed did not advance");
      }
      const expectedAttentionSeed = tinyLlamaAttentionProjectionStep(llamaModelResourceBytes, expectedAttentionCache, 1, 0);
      expectClose(
        Array.from(await hostDevice.readFloat32(attentionProjectionK, expectedAttentionSeed.k.length, 0)),
        expectedAttentionSeed.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(attentionProjectionV, expectedAttentionSeed.v.length, 0)),
        expectedAttentionSeed.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(attentionProjectionK, expectedAttentionSeed.k.length, expectedAttentionSeed.k.length * Float32Array.BYTES_PER_ELEMENT)),
        Array(expectedAttentionSeed.k.length).fill(-1111),
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(attentionProjectionSession.bindings.output, tinyLlamaVocabSize)),
        Array(tinyLlamaVocabSize).fill(-3333),
      );
      const attentionTokenTwo = keep(alloc(4), 4);
      writeU32Array(attentionTokenTwo, [2]);
      const attentionDecode = tokenExecute(attentionProjectionSessionHandle, attentionTokenTwo, 1, executeOutputLogits, 0, 0);
      expectStatus(attentionDecode.status, ok, "model-bound WebGPU tiny LLaMA attention projection logits decode");
      if (attentionDecode.outputLen !== tinyLlamaVocabSize || sessionPosition(attentionProjectionSessionHandle) !== 2) {
        throw new Error("model-bound WebGPU tiny LLaMA attention projection decode did not advance bound-logits execution");
      }
      const expectedAttentionDecode = tinyLlamaAttentionProjectionStep(llamaModelResourceBytes, expectedAttentionCache, 2, 1);
      expectClose(
        Array.from(await hostDevice.readFloat32(attentionProjectionSession.bindings.output, tinyLlamaVocabSize)),
        expectedAttentionDecode.logits,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(attentionProjectionK, expectedAttentionDecode.k.length, expectedAttentionDecode.k.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedAttentionDecode.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(attentionProjectionV, expectedAttentionDecode.v.length, expectedAttentionDecode.v.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedAttentionDecode.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(attentionProjectionK, expectedAttentionDecode.k.length, expectedAttentionDecode.k.length * 2 * Float32Array.BYTES_PER_ELEMENT)),
        Array(expectedAttentionDecode.k.length).fill(-1111),
      );
      const attentionSeedStepParams = attentionProjectionSession.stepParamsHistory[0];
      const attentionDecodeStepParams = attentionProjectionSession.stepParamsHistory[1];
      if (
        attentionProjectionSession.stepParamsHistory.length !== 2 ||
        !attentionSeedStepParams ||
        !attentionDecodeStepParams ||
        attentionSeedStepParams.startPosition !== 0 ||
        attentionSeedStepParams.endPosition !== 1 ||
        attentionSeedStepParams.outputKind !== "none" ||
        attentionDecodeStepParams.startPosition !== 1 ||
        attentionDecodeStepParams.endPosition !== 2 ||
        attentionDecodeStepParams.outputKind !== "bound-logits"
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA attention projection executor did not retain cache-handoff StepParams evidence");
      }
      const expectedAttentionBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 4) ? 2 : 0;
      const expectedAttentionFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 4) ? 0 : 2;
      if (
        attentionProjectionExecutorCalls.length !== 2 ||
        attentionProjectionExecutorCalls[0].token !== 1 ||
        attentionProjectionExecutorCalls[0].outputKind !== "none" ||
        attentionProjectionExecutorCalls[1].token !== 2 ||
        attentionProjectionExecutorCalls[1].outputKind !== "bound-logits" ||
        attentionProjectionExecutorCalls[1].qProjRole !== "llama.weight.model.layers.0.self_attn.q_proj.weight" ||
        attentionProjectionExecutorCalls[1].kProjRole !== "llama.weight.model.layers.0.self_attn.k_proj.weight" ||
        attentionProjectionExecutorCalls[1].vProjRole !== "llama.weight.model.layers.0.self_attn.v_proj.weight" ||
        attentionProjectionExecutorCalls[1].oProjRole !== "llama.weight.model.layers.0.self_attn.o_proj.weight" ||
        attentionProjectionExecutorCalls[1].hiddenSize !== 4 ||
        attentionProjectionExecutorCalls[1].kvStride !== 4 ||
        attentionProjectionExecutorCalls[1].layer !== 0 ||
        attentionProjectionExecutorCalls[0].status !== ok ||
        attentionProjectionExecutorCalls[1].status !== ok ||
        attentionProjectionExecutorCalls[0].commandCount !== 1 ||
        attentionProjectionExecutorCalls[1].commandCount !== 1 ||
        attentionProjectionExecutorCalls[0].kvWriteCount !== 2 ||
        attentionProjectionExecutorCalls[1].kvWriteCount !== 2 ||
        attentionProjectionExecutorCalls[1].modelManifestHash === 0n ||
        attentionProjectionExecutorCalls[1].modelTableHash === 0n ||
        attentionProjectionSession.modelPacks.length !== 1 ||
        attentionProjectionSession.modelPacks[0] !== attentionProjectionModelPack ||
        attentionProjectionModelPack.fieldCount !== 8 ||
        attentionProjectionModelPack.layoutHash === 0n ||
        attentionProjectionModelPack.descriptorHash === 0n ||
        attentionProjectionExecutorCalls[1].modelPackFieldCount !== attentionProjectionModelPack.fieldCount ||
        attentionProjectionExecutorCalls[1].modelPackByteLength !== attentionProjectionModelPack.byteLength ||
        attentionProjectionExecutorCalls[1].modelPackLayoutHash !== attentionProjectionModelPack.layoutHash ||
        attentionProjectionExecutorCalls[1].modelPackDescriptorHash !== attentionProjectionModelPack.descriptorHash ||
        attentionProjectionExecutorCalls[1].modelPackSourceDescriptorHash !== attentionProjectionModelPack.sourceDescriptorHash ||
        attentionProjectionExecutorCalls[1].modelPackModelManifestHash !== attentionProjectionSession.modelManifest.hash ||
        attentionProjectionExecutorCalls[1].modelPackModelTableHash !== attentionProjectionSession.modelTable.hash ||
        attentionProjectionExecutorCalls[1].tableHash === 0n
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA attention projection executor evidence mismatch");
      }
      const attentionProjectionProfile = attentionProjectionSession.runtimeProfile();
      if (
        attentionProjectionProfile.executorCallCount !== 2 ||
        attentionProjectionProfile.executorOkCount !== 2 ||
        attentionProjectionProfile.executorFailureCount !== 0 ||
        attentionProjectionProfile.executorBackendOpCount !== expectedAttentionBackendDispatchCount ||
        attentionProjectionProfile.executorBackendDispatchCount !== expectedAttentionBackendDispatchCount ||
        attentionProjectionProfile.executorFallbackOpCount !== expectedAttentionFallbackOpCount ||
        attentionProjectionProfile.executorCommandCount !== 2 ||
        attentionProjectionProfile.executorCommandOpCount !== 2 ||
        attentionProjectionProfile.lastStatus !== ok ||
        attentionProjectionProfile.lastResultOutputLength !== tinyLlamaVocabSize
      ) {
        throw new Error(`unexpected model-bound WebGPU tiny LLaMA attention projection executor profile: ${JSON.stringify(attentionProjectionProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("attention-projection", attentionProjectionProfile);
      const attentionProjectionAbiProfile = sessionRuntimeProfile(attentionProjectionSessionHandle);
      if (
        u64(attentionProjectionAbiProfile, 0) !== 2n ||
        u64(attentionProjectionAbiProfile, 8) !== BigInt(expectedAttentionBackendDispatchCount) ||
        u64(attentionProjectionAbiProfile, 16) !== BigInt(expectedAttentionFallbackOpCount) ||
        u64(attentionProjectionAbiProfile, 24) !== BigInt(expectedAttentionBackendDispatchCount) ||
        u64(attentionProjectionAbiProfile, 32) !== 0n ||
        u64(attentionProjectionAbiProfile, 40) !== 2n ||
        u64(attentionProjectionAbiProfile, 48) !== 2n ||
        u64(attentionProjectionAbiProfile, 56) !== 0n ||
        u64(attentionProjectionAbiProfile, 96) !== 2n ||
        u64(attentionProjectionAbiProfile, 112) !== 2n
      ) {
        throw new Error("unexpected model-bound WebGPU tiny LLaMA attention projection executor ABI runtime profile");
      }
      let attentionMockScratchAfterDecode = null;
      if (!canUseGpuStorageBuffers(hostDevice, 4)) {
        const scratch = attentionProjectionExecutor.mockProjectionScratch;
        attentionMockScratchAfterDecode = {
          afterAttention: scratch.afterAttention,
          attention: scratch.attention,
          attnProjected: scratch.attnProjected,
          finalHidden: scratch.finalHidden,
          hidden: scratch.hidden,
          k: scratch.k,
          logits: scratch.logits,
          normed: scratch.normed,
          q: scratch.q,
          result: scratch.result,
          v: scratch.v,
        };
        if (
          scratch.hidden.length !== 4 ||
          scratch.q.length !== 4 ||
          scratch.k.length !== 4 ||
          scratch.v.length !== 4 ||
          scratch.attention.length !== 4 ||
          scratch.afterAttention.length !== 4 ||
          scratch.logits.length !== tinyLlamaVocabSize ||
          scratch.result.logitsLength !== tinyLlamaVocabSize
        ) {
          throw new Error("model-bound WebGPU tiny LLaMA attention projection mock fallback did not retain projection scratch");
        }
      }
      hostDevice.writeFloat32(attentionProjectionSession.bindings.output, Array(tinyLlamaVocabSize).fill(-8888));
      const attentionPrefillTokens = keep(alloc(8), 8);
      writeU32Array(attentionPrefillTokens, [0, 3]);
      const attentionPrefill = tokenExecute(attentionProjectionSessionHandle, attentionPrefillTokens, 2, executeOutputLogits, 0, 0);
      expectStatus(attentionPrefill.status, ok, "model-bound WebGPU tiny LLaMA attention projection two-token prefill");
      if (attentionPrefill.outputLen !== tinyLlamaVocabSize || sessionPosition(attentionProjectionSessionHandle) !== 4) {
        throw new Error("model-bound WebGPU tiny LLaMA attention projection prefill did not advance bound-logits execution");
      }
      const expectedAttentionPrefillFirst = tinyLlamaAttentionProjectionStep(llamaModelResourceBytes, expectedAttentionCache, 0, 2);
      const expectedAttentionPrefillLast = tinyLlamaAttentionProjectionStep(llamaModelResourceBytes, expectedAttentionCache, 3, 3);
      expectClose(
        Array.from(await hostDevice.readFloat32(attentionProjectionSession.bindings.output, tinyLlamaVocabSize)),
        expectedAttentionPrefillLast.logits,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(attentionProjectionK, expectedAttentionPrefillFirst.k.length, expectedAttentionPrefillFirst.k.length * 2 * Float32Array.BYTES_PER_ELEMENT)),
        expectedAttentionPrefillFirst.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(attentionProjectionV, expectedAttentionPrefillFirst.v.length, expectedAttentionPrefillFirst.v.length * 2 * Float32Array.BYTES_PER_ELEMENT)),
        expectedAttentionPrefillFirst.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(attentionProjectionK, expectedAttentionPrefillLast.k.length, expectedAttentionPrefillLast.k.length * 3 * Float32Array.BYTES_PER_ELEMENT)),
        expectedAttentionPrefillLast.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(attentionProjectionV, expectedAttentionPrefillLast.v.length, expectedAttentionPrefillLast.v.length * 3 * Float32Array.BYTES_PER_ELEMENT)),
        expectedAttentionPrefillLast.v,
      );
      const attentionPrefillStepParams = attentionProjectionSession.stepParamsHistory[2];
      if (
        attentionProjectionSession.stepParamsHistory.length !== 3 ||
        !attentionPrefillStepParams ||
        attentionPrefillStepParams.outputKind !== "bound-logits" ||
        attentionPrefillStepParams.startPosition !== 2 ||
        attentionPrefillStepParams.endPosition !== 4 ||
        attentionPrefillStepParams.tokenCount !== 2
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA attention projection prefill StepParams evidence mismatch");
      }
      const attentionPrefillCall = attentionProjectionExecutorCalls[2];
      if (
        attentionProjectionExecutorCalls.length !== 3 ||
        !attentionPrefillCall ||
        attentionPrefillCall.token !== 3 ||
        attentionPrefillCall.tokenCount !== 2 ||
        attentionPrefillCall.tokens.length !== 2 ||
        attentionPrefillCall.tokens[0] !== 0 ||
        attentionPrefillCall.tokens[1] !== 3 ||
        attentionPrefillCall.commandCount !== 2 ||
        attentionPrefillCall.kvWriteCount !== 4 ||
        attentionPrefillCall.outputKind !== "bound-logits" ||
        attentionPrefillCall.status !== ok
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA attention projection prefill executor evidence mismatch");
      }
      const expectedAttentionPrefillBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 4) ? 4 : 0;
      const expectedAttentionPrefillFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 4) ? 0 : 4;
      const attentionPrefillProfile = attentionProjectionSession.runtimeProfile();
      if (
        attentionPrefillProfile.executorCallCount !== 3 ||
        attentionPrefillProfile.executorOkCount !== 3 ||
        attentionPrefillProfile.executorFailureCount !== 0 ||
        attentionPrefillProfile.executorBackendOpCount !== expectedAttentionPrefillBackendDispatchCount ||
        attentionPrefillProfile.executorBackendDispatchCount !== expectedAttentionPrefillBackendDispatchCount ||
        attentionPrefillProfile.executorFallbackOpCount !== expectedAttentionPrefillFallbackOpCount ||
        attentionPrefillProfile.executorCommandCount !== 4 ||
        attentionPrefillProfile.executorCommandOpCount !== 4 ||
        attentionPrefillProfile.lastStatus !== ok ||
        attentionPrefillProfile.lastResultOutputLength !== tinyLlamaVocabSize
      ) {
        throw new Error(`unexpected model-bound WebGPU tiny LLaMA attention projection prefill executor profile: ${JSON.stringify(attentionPrefillProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("attention-projection-prefill", attentionPrefillProfile);
      const attentionPrefillAbiProfile = sessionRuntimeProfile(attentionProjectionSessionHandle);
      if (
        u64(attentionPrefillAbiProfile, 0) !== 3n ||
        u64(attentionPrefillAbiProfile, 8) !== BigInt(expectedAttentionPrefillBackendDispatchCount) ||
        u64(attentionPrefillAbiProfile, 16) !== BigInt(expectedAttentionPrefillFallbackOpCount) ||
        u64(attentionPrefillAbiProfile, 24) !== BigInt(expectedAttentionPrefillBackendDispatchCount) ||
        u64(attentionPrefillAbiProfile, 32) !== 0n ||
        u64(attentionPrefillAbiProfile, 40) !== 3n ||
        u64(attentionPrefillAbiProfile, 48) !== 3n ||
        u64(attentionPrefillAbiProfile, 56) !== 0n ||
        u64(attentionPrefillAbiProfile, 96) !== 4n ||
        u64(attentionPrefillAbiProfile, 112) !== 4n
      ) {
        throw new Error("unexpected model-bound WebGPU tiny LLaMA attention projection prefill ABI runtime profile");
      }
      if (attentionMockScratchAfterDecode !== null) {
        const scratch = attentionProjectionExecutor.mockProjectionScratch;
        if (
          scratch.hidden !== attentionMockScratchAfterDecode.hidden ||
          scratch.normed !== attentionMockScratchAfterDecode.normed ||
          scratch.q !== attentionMockScratchAfterDecode.q ||
          scratch.k !== attentionMockScratchAfterDecode.k ||
          scratch.v !== attentionMockScratchAfterDecode.v ||
          scratch.attention !== attentionMockScratchAfterDecode.attention ||
          scratch.attnProjected !== attentionMockScratchAfterDecode.attnProjected ||
          scratch.afterAttention !== attentionMockScratchAfterDecode.afterAttention ||
          scratch.finalHidden !== attentionMockScratchAfterDecode.finalHidden ||
          scratch.logits !== attentionMockScratchAfterDecode.logits ||
          scratch.result !== attentionMockScratchAfterDecode.result ||
          scratch.scores.length < 4 ||
          scratch.result.logitsLength !== tinyLlamaVocabSize
        ) {
          throw new Error("model-bound WebGPU tiny LLaMA attention projection mock fallback did not reuse projection scratch");
        }
      }
    } finally {
      exportsRef.zgml_session_free(attentionProjectionSessionHandle);
      if (attentionProjectionSession.ownedResources.length !== 0) {
        throw new Error("model-bound WebGPU tiny LLaMA attention projection Session-owned resources were not released");
      }
      if (attentionProjectionModelPack?.resource && "destroyed" in attentionProjectionModelPack.resource && !attentionProjectionModelPack.resource.destroyed) {
        throw new Error("model-bound WebGPU tiny LLaMA attention projection model pack was not destroyed on Session free");
      }
      attentionProjectionExecutor.destroy?.();
    }

    const blockProjectionExecutorCalls = [];
    const blockProjectionExecutor = new WasmWebGpuLlamaBlockProjectionExecutor({
      record(call) {
        blockProjectionExecutorCalls.push(call);
      },
    });
    const blockProjectionProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      requiredModelResources: tinyLlamaModelResourceRequirements,
      allowExtraModelResources: false,
      tokenExecutor: blockProjectionExecutor,
    });
    const blockProjectionActivation = blockProjectionProgram.createActivationBuffer({ hiddenSize: 4 });
    let malformedBlockActivationRejected = false;
    try {
      const malformedActivation = blockProjectionProgram.createActivationBuffer({
        byteLength: 4 * Float32Array.BYTES_PER_ELEMENT,
      });
      blockProjectionProgram.bind({
        ...rawLlamaResources,
        activation: malformedActivation,
        modelResources: llamaModelResources,
      }, { model: compatibleModel });
    } catch (err) {
      malformedBlockActivationRejected = String(err && err.message ? err.message : err).includes("activation element count mismatch");
    }
    if (!malformedBlockActivationRejected) {
      throw new Error("expected model-bound WebGPU tiny LLaMA block projection malformed activation to reject at bind");
    }
    const blockProjectionInputActivation = blockProjectionProgram.createActivationBuffer({ hiddenSize: 4 });
    const blockProjectionInputOutputActivation = blockProjectionProgram.createActivationBuffer({ hiddenSize: 4 });
    let aliasedBlockActivationRejected = false;
    try {
      blockProjectionProgram.bind({
        ...rawLlamaResources,
        activation: blockProjectionInputActivation,
        activationInput: blockProjectionInputActivation,
        modelResources: llamaModelResources,
      }, { model: compatibleModel });
    } catch (err) {
      aliasedBlockActivationRejected = String(err && err.message ? err.message : err).includes("separate resource views");
    }
    if (!aliasedBlockActivationRejected) {
      throw new Error("expected model-bound WebGPU tiny LLaMA block projection aliased activation input/output to reject at bind");
    }
    const blockProjectionInputSession = blockProjectionProgram.bind({
      ...rawLlamaResources,
      activation: blockProjectionInputOutputActivation,
      activationInput: blockProjectionInputActivation,
      modelResources: llamaModelResources,
    }, { model: compatibleModel });
    const blockProjectionInputSessionHandle = blockProjectionInputSession.handle;
    try {
      const inputBlockK = blockProjectionInputSession.bindings.kvCache[0].k;
      const inputBlockV = blockProjectionInputSession.bindings.kvCache[0].v;
      const inputBlockActivation = blockProjectionInputSession.bindings.activation;
      const inputBlockActivationInput = blockProjectionInputSession.bindings.activationInput;
      const inputBlockKElements = Math.floor(inputBlockK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const inputBlockVElements = Math.floor(inputBlockV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const inputBlockActivationElements = Math.floor(inputBlockActivation.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const inputHidden = [0.25, -0.75, 1.5, -2.25];
      hostDevice.writeFloat32(inputBlockK, Array(inputBlockKElements).fill(-1010));
      hostDevice.writeFloat32(inputBlockV, Array(inputBlockVElements).fill(-2020));
      hostDevice.writeFloat32(inputBlockActivation, Array(inputBlockActivationElements).fill(-3030));
      hostDevice.writeFloat32(inputBlockActivationInput, Array(inputBlockActivationElements).fill(-4040));
      hostDevice.writeFloat32(inputBlockActivationInput, inputHidden, 0);
      hostDevice.writeFloat32(blockProjectionInputSession.bindings.output, Array(tinyLlamaVocabSize).fill(-5050));
      expectHostResourceTableRoles(
        blockProjectionInputSession.table,
        ["llama.output", "llama.k.0", "llama.v.0", "llama.activation", "llama.activation.input"],
        "WebGPU tiny LLaMA block projection activation input",
      );
      const inputToken = keep(alloc(4), 4);
      writeU32Array(inputToken, [1]);
      const inputStep = tokenExecute(blockProjectionInputSessionHandle, inputToken, 1, executeOutputLogits, 0, 0);
      expectStatus(inputStep.status, ok, "model-bound WebGPU tiny LLaMA block projection activation-input execute");
      if (inputStep.outputLen !== tinyLlamaVocabSize || sessionPosition(blockProjectionInputSessionHandle) !== 1) {
        throw new Error("model-bound WebGPU tiny LLaMA block projection activation-input step did not advance");
      }
      const expectedInputCache = {
        k: Array(inputBlockKElements).fill(-1010),
        v: Array(inputBlockVElements).fill(-2020),
      };
      const expectedInputBlock = tinyLlamaBlockProjectionStep(llamaModelResourceBytes, expectedInputCache, 1, 0, { inputHidden });
      expectClose(
        Array.from(await hostDevice.readFloat32(inputBlockK, expectedInputBlock.k.length, 0)),
        expectedInputBlock.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(inputBlockV, expectedInputBlock.v.length, 0)),
        expectedInputBlock.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(inputBlockActivation, expectedInputBlock.blockHidden.length, 0)),
        expectedInputBlock.blockHidden,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionInputSession.bindings.output, tinyLlamaVocabSize)),
        expectedInputBlock.logits,
      );
      const inputCall = blockProjectionExecutorCalls.at(-1);
      if (
        !inputCall ||
        inputCall.activationInputRole !== "llama.activation.input" ||
        inputCall.activationInputByteLength !== inputBlockActivationInput.byteLength ||
        inputCall.activationWriteCount !== 1 ||
        inputCall.status !== ok
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA block projection activation-input executor evidence mismatch");
      }
    } finally {
      exportsRef.zgml_session_free(blockProjectionInputSessionHandle);
    }
    blockProjectionExecutorCalls.length = 0;
    const blockProjectionSession = blockProjectionProgram.bind({
      ...rawLlamaResources,
      activation: blockProjectionActivation,
      modelResources: llamaModelResources,
    }, { model: compatibleModel });
    const blockProjectionSessionHandle = blockProjectionSession.handle;
    let blockProjectionModelPack = null;
    try {
      blockProjectionModelPack = blockProjectionExecutor.projectionForSession(blockProjectionSession).modelPack;
      const blockProjectionK = blockProjectionSession.bindings.kvCache[0].k;
      const blockProjectionV = blockProjectionSession.bindings.kvCache[0].v;
      const blockProjectionActivationDescriptor = blockProjectionSession.bindings.activation;
      const blockProjectionKElements = Math.floor(blockProjectionK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const blockProjectionVElements = Math.floor(blockProjectionV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const blockProjectionActivationElements = Math.floor(blockProjectionActivationDescriptor.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(blockProjectionK, Array(blockProjectionKElements).fill(-4444));
      hostDevice.writeFloat32(blockProjectionV, Array(blockProjectionVElements).fill(-5555));
      hostDevice.writeFloat32(blockProjectionActivationDescriptor, Array(blockProjectionActivationElements).fill(-8888));
      hostDevice.writeFloat32(blockProjectionSession.bindings.output, Array(tinyLlamaVocabSize).fill(-6666));
      expectHostResourceTableRoles(
        blockProjectionSession.table,
        ["llama.output", "llama.k.0", "llama.v.0", "llama.activation"],
        "WebGPU tiny LLaMA block projection activation",
      );
      const expectedBlockCache = {
        k: Array(blockProjectionKElements).fill(-4444),
        v: Array(blockProjectionVElements).fill(-5555),
      };
      const blockBorrowedTokenOne = [1, tinyLlamaVocabSize + 1];
      blockBorrowedTokenOne[Symbol.iterator] = () => {
        throw new Error("LLaMA block projection executor should index the active tokensLen window");
      };
      const blockSeed = blockProjectionSession.executeDecodedTokensSyncInto({}, {
        outputLen: 0,
        outputPolicy: executeOutputNone,
        outputPtr: 0,
        tokens: blockBorrowedTokenOne,
        tokensLen: 1,
      }, 0, hostRuntime);
      expectStatus(blockSeed.status, ok, "model-bound WebGPU tiny LLaMA block projection no-output seed");
      if (blockSeed.outputLen !== 0 || sessionPosition(blockProjectionSessionHandle) !== 1) {
        throw new Error("model-bound WebGPU tiny LLaMA block projection seed did not advance");
      }
      const expectedBlockSeed = tinyLlamaBlockProjectionStep(llamaModelResourceBytes, expectedBlockCache, 1, 0);
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionK, expectedBlockSeed.k.length, 0)),
        expectedBlockSeed.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionV, expectedBlockSeed.v.length, 0)),
        expectedBlockSeed.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionActivationDescriptor, expectedBlockSeed.blockHidden.length, 0)),
        expectedBlockSeed.blockHidden,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionActivationDescriptor, expectedBlockSeed.blockHidden.length, expectedBlockSeed.blockHidden.length * Float32Array.BYTES_PER_ELEMENT)),
        Array(expectedBlockSeed.blockHidden.length).fill(-8888),
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionSession.bindings.output, tinyLlamaVocabSize)),
        Array(tinyLlamaVocabSize).fill(-6666),
      );
      const blockTokenTwo = keep(alloc(4), 4);
      writeU32Array(blockTokenTwo, [2]);
      const blockDecode = tokenExecute(blockProjectionSessionHandle, blockTokenTwo, 1, executeOutputLogits, 0, 0);
      expectStatus(blockDecode.status, ok, "model-bound WebGPU tiny LLaMA block projection logits decode");
      if (blockDecode.outputLen !== tinyLlamaVocabSize || sessionPosition(blockProjectionSessionHandle) !== 2) {
        throw new Error("model-bound WebGPU tiny LLaMA block projection decode did not advance bound-logits execution");
      }
      const expectedBlockDecode = tinyLlamaBlockProjectionStep(llamaModelResourceBytes, expectedBlockCache, 2, 1);
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionSession.bindings.output, tinyLlamaVocabSize)),
        expectedBlockDecode.logits,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionK, expectedBlockDecode.k.length, expectedBlockDecode.k.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedBlockDecode.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionV, expectedBlockDecode.v.length, expectedBlockDecode.v.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedBlockDecode.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionActivationDescriptor, expectedBlockDecode.blockHidden.length, expectedBlockDecode.blockHidden.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedBlockDecode.blockHidden,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionK, expectedBlockDecode.k.length, expectedBlockDecode.k.length * 2 * Float32Array.BYTES_PER_ELEMENT)),
        Array(expectedBlockDecode.k.length).fill(-4444),
      );
      const blockSeedStepParams = blockProjectionSession.stepParamsHistory[0];
      const blockDecodeStepParams = blockProjectionSession.stepParamsHistory[1];
      if (
        blockProjectionSession.stepParamsHistory.length !== 2 ||
        !blockSeedStepParams ||
        !blockDecodeStepParams ||
        blockSeedStepParams.outputKind !== "none" ||
        blockDecodeStepParams.outputKind !== "bound-logits" ||
        blockSeedStepParams.startPosition !== 0 ||
        blockDecodeStepParams.startPosition !== 1
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA block projection executor did not retain cache-handoff StepParams evidence");
      }
      const expectedBlockBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? 2 : 0;
      const expectedBlockFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : 2;
      if (
        blockProjectionExecutorCalls.length !== 2 ||
        blockProjectionExecutorCalls[0].token !== 1 ||
        blockProjectionExecutorCalls[1].token !== 2 ||
        blockProjectionExecutorCalls[1].outputKind !== "bound-logits" ||
        blockProjectionExecutorCalls[1].postNormRole !== "llama.weight.model.layers.0.post_attention_layernorm.weight" ||
        blockProjectionExecutorCalls[1].gateProjRole !== "llama.weight.model.layers.0.mlp.gate_proj.weight" ||
        blockProjectionExecutorCalls[1].upProjRole !== "llama.weight.model.layers.0.mlp.up_proj.weight" ||
        blockProjectionExecutorCalls[1].downProjRole !== "llama.weight.model.layers.0.mlp.down_proj.weight" ||
        blockProjectionExecutorCalls[1].ffnSize !== 8 ||
        blockProjectionExecutorCalls[1].hiddenSize !== 4 ||
        blockProjectionExecutorCalls[1].activationRole !== "llama.activation" ||
        blockProjectionExecutorCalls[1].activationByteLength !== blockProjectionActivationDescriptor.byteLength ||
        blockProjectionExecutorCalls[0].activationWriteCount !== 1 ||
        blockProjectionExecutorCalls[1].activationWriteCount !== 1 ||
        blockProjectionExecutorCalls[0].status !== ok ||
        blockProjectionExecutorCalls[1].status !== ok ||
        blockProjectionExecutorCalls[0].commandCount !== 1 ||
        blockProjectionExecutorCalls[1].commandCount !== 1 ||
        blockProjectionExecutorCalls[0].kvWriteCount !== 2 ||
        blockProjectionExecutorCalls[1].kvWriteCount !== 2 ||
        blockProjectionExecutorCalls[1].modelManifestHash === 0n ||
        blockProjectionExecutorCalls[1].modelTableHash === 0n ||
        blockProjectionSession.modelPacks.length !== 1 ||
        blockProjectionSession.modelPacks[0] !== blockProjectionModelPack ||
        blockProjectionModelPack.fieldCount !== 12 ||
        blockProjectionModelPack.layoutHash === 0n ||
        blockProjectionModelPack.descriptorHash === 0n ||
        blockProjectionExecutorCalls[1].modelPackFieldCount !== blockProjectionModelPack.fieldCount ||
        blockProjectionExecutorCalls[1].modelPackByteLength !== blockProjectionModelPack.byteLength ||
        blockProjectionExecutorCalls[1].modelPackLayoutHash !== blockProjectionModelPack.layoutHash ||
        blockProjectionExecutorCalls[1].modelPackDescriptorHash !== blockProjectionModelPack.descriptorHash ||
        blockProjectionExecutorCalls[1].modelPackSourceDescriptorHash !== blockProjectionModelPack.sourceDescriptorHash ||
        blockProjectionExecutorCalls[1].modelPackModelManifestHash !== blockProjectionSession.modelManifest.hash ||
        blockProjectionExecutorCalls[1].modelPackModelTableHash !== blockProjectionSession.modelTable.hash ||
        blockProjectionExecutorCalls[1].tableHash === 0n
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA block projection executor evidence mismatch");
      }
      const blockProjectionProfile = blockProjectionSession.runtimeProfile();
      if (
        blockProjectionProfile.executorCallCount !== 2 ||
        blockProjectionProfile.executorOkCount !== 2 ||
        blockProjectionProfile.executorFailureCount !== 0 ||
        blockProjectionProfile.executorBackendOpCount !== expectedBlockBackendDispatchCount ||
        blockProjectionProfile.executorBackendDispatchCount !== expectedBlockBackendDispatchCount ||
        blockProjectionProfile.executorFallbackOpCount !== expectedBlockFallbackOpCount ||
        blockProjectionProfile.executorCommandCount !== 2 ||
        blockProjectionProfile.executorCommandOpCount !== 2 ||
        blockProjectionProfile.lastStatus !== ok ||
        blockProjectionProfile.lastResultOutputLength !== tinyLlamaVocabSize
      ) {
        throw new Error(`unexpected model-bound WebGPU tiny LLaMA block projection executor profile: ${JSON.stringify(blockProjectionProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("block-projection", blockProjectionProfile);
      const blockProjectionAbiProfile = sessionRuntimeProfile(blockProjectionSessionHandle);
      if (
        u64(blockProjectionAbiProfile, 0) !== 2n ||
        u64(blockProjectionAbiProfile, 8) !== BigInt(expectedBlockBackendDispatchCount) ||
        u64(blockProjectionAbiProfile, 16) !== BigInt(expectedBlockFallbackOpCount) ||
        u64(blockProjectionAbiProfile, 24) !== BigInt(expectedBlockBackendDispatchCount) ||
        u64(blockProjectionAbiProfile, 40) !== 2n ||
        u64(blockProjectionAbiProfile, 48) !== 2n ||
        u64(blockProjectionAbiProfile, 56) !== 0n ||
        u64(blockProjectionAbiProfile, 96) !== 2n ||
        u64(blockProjectionAbiProfile, 112) !== 2n
      ) {
        throw new Error("unexpected model-bound WebGPU tiny LLaMA block projection executor ABI runtime profile");
      }
      let blockMockScratchAfterDecode = null;
      if (!canUseGpuStorageBuffers(hostDevice, 6)) {
        const scratch = blockProjectionExecutor.mockBlockProjectionScratch;
        blockMockScratchAfterDecode = {
          attention: scratch.attention,
          blockHidden: scratch.blockHidden,
          ffn: scratch.ffn,
          ffnInput: scratch.ffnInput,
          finalHidden: scratch.finalHidden,
          gate: scratch.gate,
          logits: scratch.logits,
          normedFfn: scratch.normedFfn,
          result: scratch.result,
          up: scratch.up,
        };
        if (
          scratch.blockHidden.length !== 4 ||
          scratch.normedFfn.length !== 4 ||
          scratch.gate.length !== 8 ||
          scratch.up.length !== 8 ||
          scratch.ffnInput.length !== 8 ||
          scratch.ffn.length !== 4 ||
          scratch.logits.length !== tinyLlamaVocabSize ||
          scratch.result.blockHiddenLength !== 4 ||
          scratch.result.logitsLength !== tinyLlamaVocabSize
        ) {
          throw new Error("model-bound WebGPU tiny LLaMA block projection mock fallback did not retain block scratch");
        }
      }
      hostDevice.writeFloat32(blockProjectionSession.bindings.output, Array(tinyLlamaVocabSize).fill(-7777));
      const blockPrefillTokens = keep(alloc(8), 8);
      writeU32Array(blockPrefillTokens, [0, 3]);
      const blockPrefill = tokenExecute(blockProjectionSessionHandle, blockPrefillTokens, 2, executeOutputLogits, 0, 0);
      expectStatus(blockPrefill.status, ok, "model-bound WebGPU tiny LLaMA block projection two-token prefill");
      if (blockPrefill.outputLen !== tinyLlamaVocabSize || sessionPosition(blockProjectionSessionHandle) !== 4) {
        throw new Error("model-bound WebGPU tiny LLaMA block projection prefill did not advance bound-logits execution");
      }
      const expectedBlockPrefillFirst = tinyLlamaBlockProjectionStep(llamaModelResourceBytes, expectedBlockCache, 0, 2);
      const expectedBlockPrefillLast = tinyLlamaBlockProjectionStep(llamaModelResourceBytes, expectedBlockCache, 3, 3);
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionSession.bindings.output, tinyLlamaVocabSize)),
        expectedBlockPrefillLast.logits,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionK, expectedBlockPrefillFirst.k.length, expectedBlockPrefillFirst.k.length * 2 * Float32Array.BYTES_PER_ELEMENT)),
        expectedBlockPrefillFirst.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionV, expectedBlockPrefillFirst.v.length, expectedBlockPrefillFirst.v.length * 2 * Float32Array.BYTES_PER_ELEMENT)),
        expectedBlockPrefillFirst.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionK, expectedBlockPrefillLast.k.length, expectedBlockPrefillLast.k.length * 3 * Float32Array.BYTES_PER_ELEMENT)),
        expectedBlockPrefillLast.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionV, expectedBlockPrefillLast.v.length, expectedBlockPrefillLast.v.length * 3 * Float32Array.BYTES_PER_ELEMENT)),
        expectedBlockPrefillLast.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionActivationDescriptor, expectedBlockPrefillFirst.blockHidden.length, expectedBlockPrefillFirst.blockHidden.length * 2 * Float32Array.BYTES_PER_ELEMENT)),
        expectedBlockPrefillFirst.blockHidden,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(blockProjectionActivationDescriptor, expectedBlockPrefillLast.blockHidden.length, expectedBlockPrefillLast.blockHidden.length * 3 * Float32Array.BYTES_PER_ELEMENT)),
        expectedBlockPrefillLast.blockHidden,
      );
      const blockPrefillStepParams = blockProjectionSession.stepParamsHistory[2];
      if (
        blockProjectionSession.stepParamsHistory.length !== 3 ||
        !blockPrefillStepParams ||
        blockPrefillStepParams.outputKind !== "bound-logits" ||
        blockPrefillStepParams.startPosition !== 2 ||
        blockPrefillStepParams.endPosition !== 4 ||
        blockPrefillStepParams.tokenCount !== 2
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA block projection prefill StepParams evidence mismatch");
      }
      const blockPrefillCall = blockProjectionExecutorCalls[2];
      if (
        blockProjectionExecutorCalls.length !== 3 ||
        !blockPrefillCall ||
        blockPrefillCall.token !== 3 ||
        blockPrefillCall.tokenCount !== 2 ||
        blockPrefillCall.tokens.length !== 2 ||
        blockPrefillCall.tokens[0] !== 0 ||
        blockPrefillCall.tokens[1] !== 3 ||
        blockPrefillCall.commandCount !== 2 ||
        blockPrefillCall.kvWriteCount !== 4 ||
        blockPrefillCall.activationWriteCount !== 2 ||
        blockPrefillCall.outputKind !== "bound-logits" ||
        blockPrefillCall.status !== ok
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA block projection prefill executor evidence mismatch");
      }
      const expectedBlockPrefillBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? 4 : 0;
      const expectedBlockPrefillFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : 4;
      const blockPrefillProfile = blockProjectionSession.runtimeProfile();
      if (
        blockPrefillProfile.executorCallCount !== 3 ||
        blockPrefillProfile.executorOkCount !== 3 ||
        blockPrefillProfile.executorFailureCount !== 0 ||
        blockPrefillProfile.executorBackendOpCount !== expectedBlockPrefillBackendDispatchCount ||
        blockPrefillProfile.executorBackendDispatchCount !== expectedBlockPrefillBackendDispatchCount ||
        blockPrefillProfile.executorFallbackOpCount !== expectedBlockPrefillFallbackOpCount ||
        blockPrefillProfile.executorCommandCount !== 4 ||
        blockPrefillProfile.executorCommandOpCount !== 4 ||
        blockPrefillProfile.lastStatus !== ok ||
        blockPrefillProfile.lastResultOutputLength !== tinyLlamaVocabSize
      ) {
        throw new Error(`unexpected model-bound WebGPU tiny LLaMA block projection prefill executor profile: ${JSON.stringify(blockPrefillProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("block-projection-prefill", blockPrefillProfile);
      const blockPrefillAbiProfile = sessionRuntimeProfile(blockProjectionSessionHandle);
      if (
        u64(blockPrefillAbiProfile, 0) !== 3n ||
        u64(blockPrefillAbiProfile, 8) !== BigInt(expectedBlockPrefillBackendDispatchCount) ||
        u64(blockPrefillAbiProfile, 16) !== BigInt(expectedBlockPrefillFallbackOpCount) ||
        u64(blockPrefillAbiProfile, 24) !== BigInt(expectedBlockPrefillBackendDispatchCount) ||
        u64(blockPrefillAbiProfile, 40) !== 3n ||
        u64(blockPrefillAbiProfile, 48) !== 3n ||
        u64(blockPrefillAbiProfile, 56) !== 0n ||
        u64(blockPrefillAbiProfile, 96) !== 4n ||
        u64(blockPrefillAbiProfile, 112) !== 4n
      ) {
        throw new Error("unexpected model-bound WebGPU tiny LLaMA block projection prefill ABI runtime profile");
      }
      if (blockMockScratchAfterDecode !== null) {
        const scratch = blockProjectionExecutor.mockBlockProjectionScratch;
        if (
          scratch.attention !== blockMockScratchAfterDecode.attention ||
          scratch.blockHidden !== blockMockScratchAfterDecode.blockHidden ||
          scratch.normedFfn !== blockMockScratchAfterDecode.normedFfn ||
          scratch.gate !== blockMockScratchAfterDecode.gate ||
          scratch.up !== blockMockScratchAfterDecode.up ||
          scratch.ffnInput !== blockMockScratchAfterDecode.ffnInput ||
          scratch.ffn !== blockMockScratchAfterDecode.ffn ||
          scratch.finalHidden !== blockMockScratchAfterDecode.finalHidden ||
          scratch.logits !== blockMockScratchAfterDecode.logits ||
          scratch.result !== blockMockScratchAfterDecode.result ||
          scratch.attention.scores.length < 4 ||
          scratch.result.blockHiddenLength !== 4 ||
          scratch.result.logitsLength !== tinyLlamaVocabSize
        ) {
          throw new Error("model-bound WebGPU tiny LLaMA block projection mock fallback did not reuse block scratch");
        }
      }
    } finally {
      exportsRef.zgml_session_free(blockProjectionSessionHandle);
      if (blockProjectionSession.ownedResources.length !== 0) {
        throw new Error("model-bound WebGPU tiny LLaMA block projection Session-owned resources were not released");
      }
      if (blockProjectionModelPack?.resource && "destroyed" in blockProjectionModelPack.resource && !blockProjectionModelPack.resource.destroyed) {
        throw new Error("model-bound WebGPU tiny LLaMA block projection model pack was not destroyed on Session free");
      }
      blockProjectionExecutor.destroy?.();
    }

    const biasedQwen2BlockConfig = biasedQwen2ProofConfig(groupedTinyLlamaShape);
    const biasedQwen2BlockBytes = tinyLlamaModelResourceFileBytes(biasedGroupedTinyLlamaTwoLayerTensorSpecs);
    const unbiasedQwen2BlockBytes = tinyLlamaModelResourceFileBytes(groupedTinyLlamaTwoLayerTensorSpecs);
    const biasedQwen2BlockShape = WasmWebGpuLlamaResourceProgram.inferSafetensorsShape(biasedQwen2BlockBytes, {
      config: biasedQwen2BlockConfig,
      contextLength: groupedTinyLlamaShape.contextLength,
    });
    const biasedQwen2BlockExecutorCalls = [];
    const biasedQwen2BlockExecutor = new WasmWebGpuLlamaBlockProjectionExecutor({
      attentionHeadSize: biasedQwen2BlockShape.attentionHeadSize,
      epsilon: biasedQwen2BlockShape.epsilon,
      layer: 0,
      ropeBase: biasedQwen2BlockShape.ropeBase,
      ropeHighFreqFactor: biasedQwen2BlockShape.ropeHighFreqFactor,
      ropeKind: biasedQwen2BlockShape.ropeKind,
      ropeLowFreqFactor: biasedQwen2BlockShape.ropeLowFreqFactor,
      ropeOriginalContextLength: biasedQwen2BlockShape.ropeOriginalContextLength,
      ropeScale: biasedQwen2BlockShape.ropeScale,
      record(call) {
        biasedQwen2BlockExecutorCalls.push(call);
      },
    });
    const biasedQwen2BlockProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      config: biasedQwen2BlockConfig,
      contextLength: groupedTinyLlamaShape.contextLength,
      safetensors: biasedQwen2BlockBytes,
      allowMockFallback: true,
      tokenExecutor: biasedQwen2BlockExecutor,
    });
    if (
      biasedQwen2BlockShape.attentionHeadSize !== groupedTinyLlamaShape.attentionHeadSize ||
      biasedQwen2BlockShape.ffnSize !== groupedTinyLlamaShape.ffnSize ||
      biasedQwen2BlockProgram.kvRequirements.layers !== groupedTinyLlamaShape.layers ||
      biasedQwen2BlockExecutor.ropeBase !== 16500 ||
      biasedQwen2BlockExecutor.epsilon !== 0.000075
    ) {
      throw new Error("host-only WebGPU biased Qwen2 standalone block did not infer grouped biased shape");
    }
    const biasedQwen2BlockActivation = biasedQwen2BlockProgram.createActivationBuffer({ hiddenSize: groupedTinyLlamaShape.hiddenSize });
    const biasedQwen2BlockSession = biasedQwen2BlockProgram.bindHostResources({
      activation: biasedQwen2BlockActivation,
    });
    const biasedQwen2BlockSessionHandle = biasedQwen2BlockSession.handle;
    const biasedQwen2BlockHostExports = hostRuntime.wrapExports(exportsRef);
    let biasedQwen2BlockModelPack = null;
    try {
      biasedQwen2BlockModelPack = biasedQwen2BlockExecutor.projectionForSession(biasedQwen2BlockSession).modelPack;
      const biasedQwen2BlockK = biasedQwen2BlockSession.bindings.kvCache[0].k;
      const biasedQwen2BlockV = biasedQwen2BlockSession.bindings.kvCache[0].v;
      const biasedQwen2BlockActivationDescriptor = biasedQwen2BlockSession.bindings.activation;
      const biasedQwen2BlockKElements = Math.floor(biasedQwen2BlockK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const biasedQwen2BlockVElements = Math.floor(biasedQwen2BlockV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const biasedQwen2BlockActivationElements = Math.floor(biasedQwen2BlockActivationDescriptor.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(biasedQwen2BlockK, Array(biasedQwen2BlockKElements).fill(-7210));
      hostDevice.writeFloat32(biasedQwen2BlockV, Array(biasedQwen2BlockVElements).fill(-7220));
      hostDevice.writeFloat32(biasedQwen2BlockActivationDescriptor, Array(biasedQwen2BlockActivationElements).fill(-7230));
      hostDevice.writeFloat32(biasedQwen2BlockSession.bindings.output, Array(groupedTinyLlamaShape.vocabSize).fill(-7240));
      const biasedQwen2BlockOptions = {
        attentionHeadSize: biasedQwen2BlockShape.attentionHeadSize,
        epsilon: biasedQwen2BlockShape.epsilon,
        layer: 0,
        ropeBase: biasedQwen2BlockShape.ropeBase,
        ropeHighFreqFactor: biasedQwen2BlockShape.ropeHighFreqFactor,
        ropeKind: biasedQwen2BlockShape.ropeKind,
        ropeLowFreqFactor: biasedQwen2BlockShape.ropeLowFreqFactor,
        ropeOriginalContextLength: biasedQwen2BlockShape.ropeOriginalContextLength,
        ropeScale: biasedQwen2BlockShape.ropeScale,
      };
      const expectedBiasedQwen2BlockCache = {
        k: Array(biasedQwen2BlockKElements).fill(-7210),
        v: Array(biasedQwen2BlockVElements).fill(-7220),
      };
      const biasedQwen2BlockPrefillTokens = keep(alloc(4), 4);
      writeU32Array(biasedQwen2BlockPrefillTokens, [2]);
      const biasedQwen2BlockPrefill = tokenExecuteWith(biasedQwen2BlockHostExports, biasedQwen2BlockSessionHandle, biasedQwen2BlockPrefillTokens, 1, executeOutputNone, 0, 0);
      expectStatus(biasedQwen2BlockPrefill.status, ok, "host-only WebGPU biased Qwen2 standalone block no-output prefill");
      if (biasedQwen2BlockPrefill.outputLen !== 0 || sessionPositionWith(biasedQwen2BlockHostExports, biasedQwen2BlockSessionHandle) !== 1) {
        throw new Error("host-only WebGPU biased Qwen2 standalone block prefill did not advance without output");
      }
      const expectedBiasedQwen2BlockPrefill = tinyLlamaBlockProjectionStep(biasedQwen2BlockBytes, expectedBiasedQwen2BlockCache, 2, 0, biasedQwen2BlockOptions);
      expectClose(
        Array.from(await hostDevice.readFloat32(biasedQwen2BlockK, expectedBiasedQwen2BlockPrefill.k.length, 0)),
        expectedBiasedQwen2BlockPrefill.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(biasedQwen2BlockV, expectedBiasedQwen2BlockPrefill.v.length, 0)),
        expectedBiasedQwen2BlockPrefill.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(biasedQwen2BlockActivationDescriptor, expectedBiasedQwen2BlockPrefill.blockHidden.length, 0)),
        expectedBiasedQwen2BlockPrefill.blockHidden,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(biasedQwen2BlockSession.bindings.output, groupedTinyLlamaShape.vocabSize)),
        Array(groupedTinyLlamaShape.vocabSize).fill(-7240),
      );
      const biasedQwen2BlockDecodeTokens = keep(alloc(4), 4);
      writeU32Array(biasedQwen2BlockDecodeTokens, [5]);
      const biasedQwen2BlockDecode = tokenExecuteWith(biasedQwen2BlockHostExports, biasedQwen2BlockSessionHandle, biasedQwen2BlockDecodeTokens, 1, executeOutputLogits, 0, 0);
      expectStatus(biasedQwen2BlockDecode.status, ok, "host-only WebGPU biased Qwen2 standalone block logits decode");
      if (biasedQwen2BlockDecode.outputLen !== groupedTinyLlamaShape.vocabSize || sessionPositionWith(biasedQwen2BlockHostExports, biasedQwen2BlockSessionHandle) !== 2) {
        throw new Error("host-only WebGPU biased Qwen2 standalone block decode did not advance bound-logits execution");
      }
      const unbiasedQwen2BlockCache = {
        k: Array(biasedQwen2BlockKElements).fill(-7210),
        v: Array(biasedQwen2BlockVElements).fill(-7220),
      };
      tinyLlamaBlockProjectionStep(unbiasedQwen2BlockBytes, unbiasedQwen2BlockCache, 2, 0, biasedQwen2BlockOptions);
      const expectedBiasedQwen2BlockDecode = tinyLlamaBlockProjectionStep(biasedQwen2BlockBytes, expectedBiasedQwen2BlockCache, 5, 1, biasedQwen2BlockOptions);
      const unbiasedQwen2BlockDecode = tinyLlamaBlockProjectionStep(unbiasedQwen2BlockBytes, unbiasedQwen2BlockCache, 5, 1, biasedQwen2BlockOptions);
      if (!expectedBiasedQwen2BlockDecode.logits.some((value, i) => Math.abs(value - unbiasedQwen2BlockDecode.logits[i]) > 1e-5)) {
        throw new Error("host-only WebGPU biased Qwen2 standalone block bias did not change logits versus unbiased reference");
      }
      expectClose(
        Array.from(await hostDevice.readFloat32(biasedQwen2BlockSession.bindings.output, groupedTinyLlamaShape.vocabSize)),
        expectedBiasedQwen2BlockDecode.logits,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(biasedQwen2BlockK, expectedBiasedQwen2BlockDecode.k.length, expectedBiasedQwen2BlockDecode.k.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedBiasedQwen2BlockDecode.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(biasedQwen2BlockV, expectedBiasedQwen2BlockDecode.v.length, expectedBiasedQwen2BlockDecode.v.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedBiasedQwen2BlockDecode.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(biasedQwen2BlockActivationDescriptor, expectedBiasedQwen2BlockDecode.blockHidden.length, expectedBiasedQwen2BlockDecode.blockHidden.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedBiasedQwen2BlockDecode.blockHidden,
      );
      const expectedBiasedQwen2BlockBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? 2 : 0;
      const expectedBiasedQwen2BlockFallbackOpCount = expectedBiasedQwen2BlockBackendDispatchCount === 2 ? 0 : 2;
      if (
        biasedQwen2BlockExecutorCalls.length !== 2 ||
        biasedQwen2BlockExecutorCalls[0].commandCount !== 1 ||
        biasedQwen2BlockExecutorCalls[1].commandCount !== 1 ||
        biasedQwen2BlockExecutorCalls[0].kvWriteCount !== 2 ||
        biasedQwen2BlockExecutorCalls[1].kvWriteCount !== 2 ||
        biasedQwen2BlockExecutorCalls[0].activationWriteCount !== 1 ||
        biasedQwen2BlockExecutorCalls[1].activationWriteCount !== 1 ||
        biasedQwen2BlockExecutorCalls[1].outputKind !== "bound-logits" ||
        biasedQwen2BlockExecutorCalls[1].layer !== 0 ||
        biasedQwen2BlockExecutorCalls[1].biasMask !== 255 ||
        biasedQwen2BlockExecutorCalls[1].queryHeadCount !== 4 ||
        biasedQwen2BlockExecutorCalls[1].kvHeadCount !== 2 ||
        biasedQwen2BlockExecutorCalls[1].ffnSize !== groupedTinyLlamaShape.ffnSize ||
        biasedQwen2BlockExecutorCalls[1].activationRole !== "llama.activation" ||
        biasedQwen2BlockExecutorCalls[1].modelPackFieldCount !== 20 ||
        biasedQwen2BlockModelPack.fieldCount !== 20 ||
        biasedQwen2BlockExecutorCalls[1].status !== ok
      ) {
        throw new Error("host-only WebGPU biased Qwen2 standalone block executor evidence mismatch");
      }
      const biasedQwen2BlockProfile = biasedQwen2BlockSession.runtimeProfile();
      if (
        biasedQwen2BlockProfile.executorCallCount !== 2 ||
        biasedQwen2BlockProfile.executorOkCount !== 2 ||
        biasedQwen2BlockProfile.executorBackendDispatchCount !== expectedBiasedQwen2BlockBackendDispatchCount ||
        biasedQwen2BlockProfile.executorFallbackOpCount !== expectedBiasedQwen2BlockFallbackOpCount ||
        biasedQwen2BlockProfile.executorCommandCount !== 2 ||
        biasedQwen2BlockProfile.lastStatus !== ok ||
        biasedQwen2BlockProfile.lastResultOutputLength !== groupedTinyLlamaShape.vocabSize
      ) {
        throw new Error(`unexpected host-only WebGPU biased Qwen2 standalone block profile: ${JSON.stringify(biasedQwen2BlockProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("block-projection-biased-qwen2", biasedQwen2BlockProfile);
    } finally {
      biasedQwen2BlockHostExports.zgml_session_free(biasedQwen2BlockSessionHandle);
      if (biasedQwen2BlockSession.ownedResources.length !== 0) {
        throw new Error("host-only WebGPU biased Qwen2 standalone block Session-owned resources were not released");
      }
      if (biasedQwen2BlockModelPack?.resource && "destroyed" in biasedQwen2BlockModelPack.resource && !biasedQwen2BlockModelPack.resource.destroyed) {
        throw new Error("host-only WebGPU biased Qwen2 standalone block model pack was not destroyed on Session free");
      }
      biasedQwen2BlockExecutor.destroy?.();
    }

    const qwen3QkNormBlockConfig = qwen3QkNormProofConfig(groupedTinyLlamaShape);
    const qwen3QkNormBlockBytes = tinyLlamaModelResourceFileBytes(qkNormGroupedTinyLlamaTwoLayerTensorSpecs);
    const noQkNormBlockBytes = tinyLlamaModelResourceFileBytes(groupedTinyLlamaTwoLayerTensorSpecs);
    const qwen3QkNormBlockShape = WasmWebGpuLlamaResourceProgram.inferSafetensorsShape(qwen3QkNormBlockBytes, {
      config: qwen3QkNormBlockConfig,
      contextLength: groupedTinyLlamaShape.contextLength,
    });
    const qwen3QkNormBlockExecutorCalls = [];
    const qwen3QkNormBlockExecutor = new WasmWebGpuLlamaBlockProjectionExecutor({
      attentionHeadSize: qwen3QkNormBlockShape.attentionHeadSize,
      epsilon: qwen3QkNormBlockShape.epsilon,
      layer: 0,
      qkProjectionNorm: qwen3QkNormBlockShape.qkProjectionNorm,
      ropeBase: qwen3QkNormBlockShape.ropeBase,
      ropeHighFreqFactor: qwen3QkNormBlockShape.ropeHighFreqFactor,
      ropeKind: qwen3QkNormBlockShape.ropeKind,
      ropeLowFreqFactor: qwen3QkNormBlockShape.ropeLowFreqFactor,
      ropeOriginalContextLength: qwen3QkNormBlockShape.ropeOriginalContextLength,
      ropeScale: qwen3QkNormBlockShape.ropeScale,
      record(call) {
        qwen3QkNormBlockExecutorCalls.push(call);
      },
    });
    const qwen3QkNormBlockProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      config: qwen3QkNormBlockConfig,
      contextLength: groupedTinyLlamaShape.contextLength,
      safetensors: qwen3QkNormBlockBytes,
      allowMockFallback: true,
      tokenExecutor: qwen3QkNormBlockExecutor,
    });
    if (
      qwen3QkNormBlockShape.attentionHeadSize !== groupedTinyLlamaShape.attentionHeadSize ||
      qwen3QkNormBlockShape.ffnSize !== groupedTinyLlamaShape.ffnSize ||
      qwen3QkNormBlockShape.qkProjectionNorm !== true ||
      qwen3QkNormBlockProgram.kvRequirements.layers !== groupedTinyLlamaShape.layers ||
      qwen3QkNormBlockExecutor.qkProjectionNorm !== true ||
      qwen3QkNormBlockExecutor.ropeBase !== 17500 ||
      qwen3QkNormBlockExecutor.epsilon !== 0.000082
    ) {
      throw new Error("host-only WebGPU Qwen3 Q/K-norm standalone block did not infer grouped Q/K-norm shape");
    }
    const qwen3QkNormBlockActivation = qwen3QkNormBlockProgram.createActivationBuffer({ hiddenSize: groupedTinyLlamaShape.hiddenSize });
    const qwen3QkNormBlockSession = qwen3QkNormBlockProgram.bindHostResources({
      activation: qwen3QkNormBlockActivation,
    });
    const qwen3QkNormBlockSessionHandle = qwen3QkNormBlockSession.handle;
    const qwen3QkNormBlockHostExports = hostRuntime.wrapExports(exportsRef);
    let qwen3QkNormBlockModelPack = null;
    try {
      qwen3QkNormBlockModelPack = qwen3QkNormBlockExecutor.projectionForSession(qwen3QkNormBlockSession).modelPack;
      const qwen3QkNormBlockK = qwen3QkNormBlockSession.bindings.kvCache[0].k;
      const qwen3QkNormBlockV = qwen3QkNormBlockSession.bindings.kvCache[0].v;
      const qwen3QkNormBlockActivationDescriptor = qwen3QkNormBlockSession.bindings.activation;
      const qwen3QkNormBlockKElements = Math.floor(qwen3QkNormBlockK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const qwen3QkNormBlockVElements = Math.floor(qwen3QkNormBlockV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const qwen3QkNormBlockActivationElements = Math.floor(qwen3QkNormBlockActivationDescriptor.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(qwen3QkNormBlockK, Array(qwen3QkNormBlockKElements).fill(-7310));
      hostDevice.writeFloat32(qwen3QkNormBlockV, Array(qwen3QkNormBlockVElements).fill(-7320));
      hostDevice.writeFloat32(qwen3QkNormBlockActivationDescriptor, Array(qwen3QkNormBlockActivationElements).fill(-7330));
      hostDevice.writeFloat32(qwen3QkNormBlockSession.bindings.output, Array(groupedTinyLlamaShape.vocabSize).fill(-7340));
      const qwen3QkNormBlockOptions = {
        attentionHeadSize: qwen3QkNormBlockShape.attentionHeadSize,
        epsilon: qwen3QkNormBlockShape.epsilon,
        layer: 0,
        ropeBase: qwen3QkNormBlockShape.ropeBase,
        ropeHighFreqFactor: qwen3QkNormBlockShape.ropeHighFreqFactor,
        ropeKind: qwen3QkNormBlockShape.ropeKind,
        ropeLowFreqFactor: qwen3QkNormBlockShape.ropeLowFreqFactor,
        ropeOriginalContextLength: qwen3QkNormBlockShape.ropeOriginalContextLength,
        ropeScale: qwen3QkNormBlockShape.ropeScale,
      };
      const expectedQwen3QkNormBlockCache = {
        k: Array(qwen3QkNormBlockKElements).fill(-7310),
        v: Array(qwen3QkNormBlockVElements).fill(-7320),
      };
      const qwen3QkNormBlockPrefillTokens = keep(alloc(4), 4);
      writeU32Array(qwen3QkNormBlockPrefillTokens, [2]);
      const qwen3QkNormBlockPrefill = tokenExecuteWith(qwen3QkNormBlockHostExports, qwen3QkNormBlockSessionHandle, qwen3QkNormBlockPrefillTokens, 1, executeOutputNone, 0, 0);
      expectStatus(qwen3QkNormBlockPrefill.status, ok, "host-only WebGPU Qwen3 Q/K-norm standalone block no-output prefill");
      if (qwen3QkNormBlockPrefill.outputLen !== 0 || sessionPositionWith(qwen3QkNormBlockHostExports, qwen3QkNormBlockSessionHandle) !== 1) {
        throw new Error("host-only WebGPU Qwen3 Q/K-norm standalone block prefill did not advance without output");
      }
      const expectedQwen3QkNormBlockPrefill = tinyLlamaBlockProjectionStep(qwen3QkNormBlockBytes, expectedQwen3QkNormBlockCache, 2, 0, qwen3QkNormBlockOptions);
      expectClose(
        Array.from(await hostDevice.readFloat32(qwen3QkNormBlockK, expectedQwen3QkNormBlockPrefill.k.length, 0)),
        expectedQwen3QkNormBlockPrefill.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(qwen3QkNormBlockV, expectedQwen3QkNormBlockPrefill.v.length, 0)),
        expectedQwen3QkNormBlockPrefill.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(qwen3QkNormBlockActivationDescriptor, expectedQwen3QkNormBlockPrefill.blockHidden.length, 0)),
        expectedQwen3QkNormBlockPrefill.blockHidden,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(qwen3QkNormBlockSession.bindings.output, groupedTinyLlamaShape.vocabSize)),
        Array(groupedTinyLlamaShape.vocabSize).fill(-7340),
      );
      const qwen3QkNormBlockDecodeTokens = keep(alloc(4), 4);
      writeU32Array(qwen3QkNormBlockDecodeTokens, [5]);
      const qwen3QkNormBlockDecode = tokenExecuteWith(qwen3QkNormBlockHostExports, qwen3QkNormBlockSessionHandle, qwen3QkNormBlockDecodeTokens, 1, executeOutputLogits, 0, 0);
      expectStatus(qwen3QkNormBlockDecode.status, ok, "host-only WebGPU Qwen3 Q/K-norm standalone block logits decode");
      if (qwen3QkNormBlockDecode.outputLen !== groupedTinyLlamaShape.vocabSize || sessionPositionWith(qwen3QkNormBlockHostExports, qwen3QkNormBlockSessionHandle) !== 2) {
        throw new Error("host-only WebGPU Qwen3 Q/K-norm standalone block decode did not advance bound-logits execution");
      }
      const noQkNormBlockCache = {
        k: Array(qwen3QkNormBlockKElements).fill(-7310),
        v: Array(qwen3QkNormBlockVElements).fill(-7320),
      };
      const noQkNormBlockPrefill = tinyLlamaBlockProjectionStep(noQkNormBlockBytes, noQkNormBlockCache, 2, 0, qwen3QkNormBlockOptions);
      const expectedQwen3QkNormBlockDecode = tinyLlamaBlockProjectionStep(qwen3QkNormBlockBytes, expectedQwen3QkNormBlockCache, 5, 1, qwen3QkNormBlockOptions);
      const noQkNormBlockDecode = tinyLlamaBlockProjectionStep(noQkNormBlockBytes, noQkNormBlockCache, 5, 1, qwen3QkNormBlockOptions);
      if (
        !expectedQwen3QkNormBlockPrefill.k.some((value, i) => Math.abs(value - noQkNormBlockPrefill.k[i]) > 1e-5) &&
        !expectedQwen3QkNormBlockDecode.k.some((value, i) => Math.abs(value - noQkNormBlockDecode.k[i]) > 1e-5)
      ) {
        throw new Error("host-only WebGPU Qwen3 Q/K-norm standalone block K norm did not change cache rows versus no-norm reference");
      }
      expectClose(
        Array.from(await hostDevice.readFloat32(qwen3QkNormBlockSession.bindings.output, groupedTinyLlamaShape.vocabSize)),
        expectedQwen3QkNormBlockDecode.logits,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(qwen3QkNormBlockK, expectedQwen3QkNormBlockDecode.k.length, expectedQwen3QkNormBlockDecode.k.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedQwen3QkNormBlockDecode.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(qwen3QkNormBlockV, expectedQwen3QkNormBlockDecode.v.length, expectedQwen3QkNormBlockDecode.v.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedQwen3QkNormBlockDecode.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(qwen3QkNormBlockActivationDescriptor, expectedQwen3QkNormBlockDecode.blockHidden.length, expectedQwen3QkNormBlockDecode.blockHidden.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedQwen3QkNormBlockDecode.blockHidden,
      );
      const expectedQwen3QkNormBlockBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? 2 : 0;
      const expectedQwen3QkNormBlockFallbackOpCount = expectedQwen3QkNormBlockBackendDispatchCount === 2 ? 0 : 2;
      if (
        qwen3QkNormBlockExecutorCalls.length !== 2 ||
        qwen3QkNormBlockExecutorCalls[0].commandCount !== 1 ||
        qwen3QkNormBlockExecutorCalls[1].commandCount !== 1 ||
        qwen3QkNormBlockExecutorCalls[0].kvWriteCount !== 2 ||
        qwen3QkNormBlockExecutorCalls[1].kvWriteCount !== 2 ||
        qwen3QkNormBlockExecutorCalls[0].activationWriteCount !== 1 ||
        qwen3QkNormBlockExecutorCalls[1].activationWriteCount !== 1 ||
        qwen3QkNormBlockExecutorCalls[1].outputKind !== "bound-logits" ||
        qwen3QkNormBlockExecutorCalls[1].layer !== 0 ||
        qwen3QkNormBlockExecutorCalls[1].qkProjectionNorm !== true ||
        qwen3QkNormBlockExecutorCalls[1].qNormRole !== "llama.weight.model.layers.0.self_attn.q_norm.weight" ||
        qwen3QkNormBlockExecutorCalls[1].kNormRole !== "llama.weight.model.layers.0.self_attn.k_norm.weight" ||
        qwen3QkNormBlockExecutorCalls[1].biasMask !== 0 ||
        qwen3QkNormBlockExecutorCalls[1].queryHeadCount !== 4 ||
        qwen3QkNormBlockExecutorCalls[1].kvHeadCount !== 2 ||
        qwen3QkNormBlockExecutorCalls[1].ffnSize !== groupedTinyLlamaShape.ffnSize ||
        qwen3QkNormBlockExecutorCalls[1].activationRole !== "llama.activation" ||
        qwen3QkNormBlockExecutorCalls[1].modelPackFieldCount !== 14 ||
        qwen3QkNormBlockModelPack.fieldCount !== 14 ||
        qwen3QkNormBlockExecutorCalls[1].status !== ok
      ) {
        throw new Error("host-only WebGPU Qwen3 Q/K-norm standalone block executor evidence mismatch");
      }
      const qwen3QkNormBlockProfile = qwen3QkNormBlockSession.runtimeProfile();
      if (
        qwen3QkNormBlockProfile.executorCallCount !== 2 ||
        qwen3QkNormBlockProfile.executorOkCount !== 2 ||
        qwen3QkNormBlockProfile.executorBackendDispatchCount !== expectedQwen3QkNormBlockBackendDispatchCount ||
        qwen3QkNormBlockProfile.executorFallbackOpCount !== expectedQwen3QkNormBlockFallbackOpCount ||
        qwen3QkNormBlockProfile.executorCommandCount !== 2 ||
        qwen3QkNormBlockProfile.lastStatus !== ok ||
        qwen3QkNormBlockProfile.lastResultOutputLength !== groupedTinyLlamaShape.vocabSize
      ) {
        throw new Error(`unexpected host-only WebGPU Qwen3 Q/K-norm standalone block profile: ${JSON.stringify(qwen3QkNormBlockProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("block-projection-qknorm-qwen3", qwen3QkNormBlockProfile);
    } finally {
      qwen3QkNormBlockHostExports.zgml_session_free(qwen3QkNormBlockSessionHandle);
      if (qwen3QkNormBlockSession.ownedResources.length !== 0) {
        throw new Error("host-only WebGPU Qwen3 Q/K-norm standalone block Session-owned resources were not released");
      }
      if (qwen3QkNormBlockModelPack?.resource && "destroyed" in qwen3QkNormBlockModelPack.resource && !qwen3QkNormBlockModelPack.resource.destroyed) {
        throw new Error("host-only WebGPU Qwen3 Q/K-norm standalone block model pack was not destroyed on Session free");
      }
      qwen3QkNormBlockExecutor.destroy?.();
    }

    const readStandaloneBlockFloat32 = async (descriptor, length, byteOffset = 0) => (
      Array.from(await hostDevice.readFloat32(descriptor, length, byteOffset))
    );
    async function runStandaloneBlockVariantProof(proof) {
      const blockBytes = tinyLlamaModelResourceFileBytes(proof.tensorSpecs ?? groupedTinyLlamaTwoLayerTensorSpecs, proof.fileOptions ?? {});
      const blockShape = WasmWebGpuLlamaResourceProgram.inferSafetensorsShape(blockBytes, {
        config: proof.config,
        contextLength: groupedTinyLlamaShape.contextLength,
      });
      const layer = proof.layer ?? 0;
      const calls = [];
      const executor = new WasmWebGpuLlamaBlockProjectionExecutor({
        attentionHeadSize: blockShape.attentionHeadSize,
        epsilon: blockShape.epsilon,
        layer,
        qkProjectionNorm: blockShape.qkProjectionNorm,
        ropeBase: blockShape.ropeBase,
        ropeHighFreqFactor: blockShape.ropeHighFreqFactor,
        ropeKind: blockShape.ropeKind,
        ropeLowFreqFactor: blockShape.ropeLowFreqFactor,
        ropeOriginalContextLength: blockShape.ropeOriginalContextLength,
        ropeScale: blockShape.ropeScale,
        slidingWindow: blockShape.slidingWindow,
        useRopeLayers: blockShape.useRopeLayers,
        record(call) {
          calls.push(call);
        },
      });
      const blockProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        config: proof.config,
        contextLength: groupedTinyLlamaShape.contextLength,
        safetensors: blockBytes,
        allowMockFallback: true,
        tokenExecutor: executor,
      });
      proof.expectShape?.({ executor, program: blockProgram, shape: blockShape });
      const activation = blockProgram.createActivationBuffer({ hiddenSize: groupedTinyLlamaShape.hiddenSize });
      const session = blockProgram.bindHostResources({ activation });
      const sessionHandle = session.handle;
      const hostExports = hostRuntime.wrapExports(exportsRef);
      let modelPack = null;
      try {
        modelPack = executor.projectionForSession(session).modelPack;
        const k = session.bindings.kvCache[layer].k;
        const v = session.bindings.kvCache[layer].v;
        const activationDescriptor = session.bindings.activation;
        const kElements = Math.floor(k.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const vElements = Math.floor(v.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const activationElements = Math.floor(activationDescriptor.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const sentinel = proof.sentinelBase;
        hostDevice.writeFloat32(k, Array(kElements).fill(sentinel - 10));
        hostDevice.writeFloat32(v, Array(vElements).fill(sentinel - 20));
        hostDevice.writeFloat32(activationDescriptor, Array(activationElements).fill(sentinel - 30));
        hostDevice.writeFloat32(session.bindings.output, Array(groupedTinyLlamaShape.vocabSize).fill(sentinel - 40));
        const options = {
          attentionHeadSize: blockShape.attentionHeadSize,
          epsilon: blockShape.epsilon,
          layer,
          ropeBase: blockShape.ropeBase,
          ropeHighFreqFactor: blockShape.ropeHighFreqFactor,
          ropeKind: blockShape.ropeKind,
          ropeLowFreqFactor: blockShape.ropeLowFreqFactor,
          ropeOriginalContextLength: blockShape.ropeOriginalContextLength,
          ropeScale: blockShape.ropeScale,
          slidingWindow: blockShape.slidingWindow,
          useRope: executor.useRope,
        };
        const expectedCache = {
          k: Array(kElements).fill(sentinel - 10),
          v: Array(vElements).fill(sentinel - 20),
        };
        const prefillTokens = proof.prefillTokens ?? [2];
        const prefillPtr = keep(alloc(prefillTokens.length * Uint32Array.BYTES_PER_ELEMENT), prefillTokens.length * Uint32Array.BYTES_PER_ELEMENT);
        writeU32Array(prefillPtr, prefillTokens);
        const prefill = tokenExecuteWith(hostExports, sessionHandle, prefillPtr, prefillTokens.length, executeOutputNone, 0, 0);
        expectStatus(prefill.status, ok, `${proof.label} no-output prefill`);
        if (prefill.outputLen !== 0 || sessionPositionWith(hostExports, sessionHandle) !== prefillTokens.length) {
          throw new Error(`${proof.label} prefill did not advance without output`);
        }
        const expectedPrefill = [];
        for (let i = 0; i < prefillTokens.length; i += 1) {
          expectedPrefill.push(tinyLlamaBlockProjectionStep(blockBytes, expectedCache, prefillTokens[i], i, options));
        }
        for (let i = 0; i < expectedPrefill.length; i += 1) {
          const expected = expectedPrefill[i];
          expectClose(await readStandaloneBlockFloat32(k, expected.k.length, expected.k.length * i * Float32Array.BYTES_PER_ELEMENT), expected.k);
          expectClose(await readStandaloneBlockFloat32(v, expected.v.length, expected.v.length * i * Float32Array.BYTES_PER_ELEMENT), expected.v);
          expectClose(await readStandaloneBlockFloat32(activationDescriptor, expected.blockHidden.length, expected.blockHidden.length * i * Float32Array.BYTES_PER_ELEMENT), expected.blockHidden);
        }
        expectClose(await readStandaloneBlockFloat32(session.bindings.output, groupedTinyLlamaShape.vocabSize), Array(groupedTinyLlamaShape.vocabSize).fill(sentinel - 40));
        proof.afterPrefill?.({ expectedCache, hostDevice, k, v, rowLength: expectedPrefill[0].k.length });
        const cacheBeforeDecode = {
          k: expectedCache.k.slice(),
          v: expectedCache.v.slice(),
        };
        const decodeToken = proof.decodeToken ?? 5;
        const decodePtr = keep(alloc(4), 4);
        writeU32Array(decodePtr, [decodeToken]);
        const decode = tokenExecuteWith(hostExports, sessionHandle, decodePtr, 1, executeOutputLogits, 0, 0);
        expectStatus(decode.status, ok, `${proof.label} logits decode`);
        if (decode.outputLen !== groupedTinyLlamaShape.vocabSize || sessionPositionWith(hostExports, sessionHandle) !== prefillTokens.length + 1) {
          throw new Error(`${proof.label} decode did not advance bound-logits execution`);
        }
        const expectedDecode = tinyLlamaBlockProjectionStep(blockBytes, expectedCache, decodeToken, prefillTokens.length, options);
        proof.expectBehavior?.({
          blockBytes,
          cacheBeforeDecode,
          decodePosition: prefillTokens.length,
          decodeToken,
          expectedDecode,
          kElements,
          options,
          prefillTokens,
          vElements,
        });
        expectClose(await readStandaloneBlockFloat32(session.bindings.output, groupedTinyLlamaShape.vocabSize), expectedDecode.logits);
        expectClose(await readStandaloneBlockFloat32(k, expectedDecode.k.length, expectedDecode.k.length * prefillTokens.length * Float32Array.BYTES_PER_ELEMENT), expectedDecode.k);
        expectClose(await readStandaloneBlockFloat32(v, expectedDecode.v.length, expectedDecode.v.length * prefillTokens.length * Float32Array.BYTES_PER_ELEMENT), expectedDecode.v);
        expectClose(await readStandaloneBlockFloat32(activationDescriptor, expectedDecode.blockHidden.length, expectedDecode.blockHidden.length * prefillTokens.length * Float32Array.BYTES_PER_ELEMENT), expectedDecode.blockHidden);
        const expectedDispatchOptions = {
          contextLength: groupedTinyLlamaShape.contextLength,
          hiddenSize: groupedTinyLlamaShape.hiddenSize,
          slidingWindow: blockShape.slidingWindow ?? 0,
        };
        const expectedPrefillCommandCount = expectedLlamaBlockNoOutputPrefillCommandCount(
          hostDevice,
          1,
          prefillTokens.length,
          expectedDispatchOptions,
        );
        const expectedPrefillDispatchFamilies = expectedLlamaBlockPipelineDispatchFamilies(
          hostDevice,
          1,
          prefillTokens.length,
          executeOutputNone,
          expectedDispatchOptions,
        );
        const expectedDecodeCommandCount = expectedLlamaBlockPipelineCommandCount(
          hostDevice,
          1,
          1,
          executeOutputLogits,
          {
            ...expectedDispatchOptions,
            startPosition: prefillTokens.length,
          },
        );
        const expectedDecodeDispatchFamilies = expectedLlamaBlockPipelineDispatchFamilies(
          hostDevice,
          1,
          1,
          executeOutputLogits,
          {
            ...expectedDispatchOptions,
            startPosition: prefillTokens.length,
          },
        );
        const expectedCommandCount = expectedPrefillCommandCount + expectedDecodeCommandCount;
        const expectedBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? expectedCommandCount : 0;
        const expectedFallbackOpCount = expectedBackendDispatchCount === expectedCommandCount ? 0 : expectedCommandCount;
        if (
          calls.length !== 2 ||
          calls[0].tokenCount !== prefillTokens.length ||
          calls[0].backendDispatchCount !== (canUseGpuStorageBuffers(hostDevice, 6) ? expectedPrefillCommandCount : 0) ||
          calls[0].commandCount !== expectedPrefillCommandCount ||
          calls[0].fallbackOpCount !== (canUseGpuStorageBuffers(hostDevice, 6) ? 0 : expectedPrefillCommandCount) ||
          calls[0].genericDispatchCount !== expectedPrefillDispatchFamilies.generic ||
          calls[0].kvWriteCount !== prefillTokens.length * 2 ||
          calls[0].activationWriteCount !== prefillTokens.length ||
          calls[0].outputKind !== "none" ||
          calls[0].scalarDispatchCount !== expectedPrefillDispatchFamilies.scalar ||
          calls[0].windowDispatchCount !== expectedPrefillDispatchFamilies.window ||
          calls[1].backendDispatchCount !== (canUseGpuStorageBuffers(hostDevice, 6) ? expectedDecodeCommandCount : 0) ||
          calls[1].token !== decodeToken ||
          calls[1].commandCount !== expectedDecodeCommandCount ||
          calls[1].fallbackOpCount !== (canUseGpuStorageBuffers(hostDevice, 6) ? 0 : expectedDecodeCommandCount) ||
          calls[1].genericDispatchCount !== expectedDecodeDispatchFamilies.generic ||
          calls[1].kvWriteCount !== 2 ||
          calls[1].activationWriteCount !== 1 ||
          calls[1].outputKind !== "bound-logits" ||
          calls[1].scalarDispatchCount !== expectedDecodeDispatchFamilies.scalar ||
          calls[1].windowDispatchCount !== expectedDecodeDispatchFamilies.window ||
          calls[1].layer !== layer ||
          calls[1].queryHeadCount !== 4 ||
          calls[1].kvHeadCount !== 2 ||
          calls[1].ffnSize !== groupedTinyLlamaShape.ffnSize ||
          calls[1].activationRole !== "llama.activation" ||
          calls[1].modelPackFieldCount !== (proof.modelPackFieldCount ?? 12) ||
          modelPack.fieldCount !== (proof.modelPackFieldCount ?? 12) ||
          calls[1].status !== ok
        ) {
          throw new Error(`${proof.label} executor evidence mismatch`);
        }
        proof.expectCall?.(calls[1]);
        const profile = session.runtimeProfile();
        if (
          profile.executorCallCount !== 2 ||
          profile.executorOkCount !== 2 ||
          profile.executorBackendDispatchCount !== expectedBackendDispatchCount ||
          profile.executorFallbackOpCount !== expectedFallbackOpCount ||
          profile.executorCommandCount !== expectedCommandCount ||
          profile.lastStatus !== ok ||
          profile.lastResultOutputLength !== groupedTinyLlamaShape.vocabSize
        ) {
          throw new Error(`unexpected ${proof.label} profile: ${JSON.stringify(profile)}`);
        }
        if (proof.profileLabel) recordBrowserLlamaProfileEvidence(proof.profileLabel, profile);
      } finally {
        hostExports.zgml_session_free(sessionHandle);
        if (session.ownedResources.length !== 0) {
          throw new Error(`${proof.label} Session-owned resources were not released`);
        }
        if (modelPack?.resource && "destroyed" in modelPack.resource && !modelPack.resource.destroyed) {
          throw new Error(`${proof.label} model pack was not destroyed on Session free`);
        }
        executor.destroy?.();
      }
    }

    await runStandaloneBlockVariantProof({
      config: mistralSlidingWindowProofConfig(groupedTinyLlamaShape),
      fileOptions: { metadata: { config: JSON.stringify(mistralSlidingWindowProofConfig(groupedTinyLlamaShape)) } },
      label: "host-only WebGPU Mistral standalone block",
      prefillTokens: [2, 3],
      profileLabel: "block-projection-sliding-window",
      sentinelBase: -7410,
      expectShape({ executor, program: blockProgram, shape }) {
        if (
          shape.slidingWindow !== 2 ||
          blockProgram.kvRequirements.layers !== groupedTinyLlamaShape.layers ||
          executor.slidingWindow !== 2 ||
          executor.ropeBase !== 15000 ||
          executor.epsilon !== 0.00006
        ) {
          throw new Error("host-only WebGPU Mistral standalone block did not infer grouped sliding-window shape");
        }
      },
      afterPrefill({ expectedCache, hostDevice, k, rowLength, v }) {
        const poisonK = Array(rowLength).fill(7.25);
        const poisonV = Array(rowLength).fill(-8.5);
        hostDevice.writeFloat32(k, poisonK, 0);
        hostDevice.writeFloat32(v, poisonV, 0);
        expectedCache.k.splice(0, poisonK.length, ...poisonK);
        expectedCache.v.splice(0, poisonV.length, ...poisonV);
      },
      expectBehavior({ blockBytes, cacheBeforeDecode, decodePosition, decodeToken, expectedDecode, options }) {
        const fullContextDecode = tinyLlamaBlockProjectionStep(blockBytes, {
          k: cacheBeforeDecode.k.slice(),
          v: cacheBeforeDecode.v.slice(),
        }, decodeToken, decodePosition, { ...options, slidingWindow: null });
        if (!expectedDecode.logits.some((value, i) => Math.abs(value - fullContextDecode.logits[i]) > 1e-5)) {
          throw new Error("host-only WebGPU Mistral standalone block sliding window did not change logits versus full-context block");
        }
      },
      expectCall(call) {
        if (call.slidingWindow !== 2 || call.biasMask !== 0 || call.qkProjectionNorm !== false) {
          throw new Error("host-only WebGPU Mistral standalone block executor family evidence mismatch");
        }
      },
    });

    await runStandaloneBlockVariantProof({
      config: llama3RopeProofConfig(groupedTinyLlamaShape),
      label: "host-only WebGPU Llama3 standalone block",
      profileLabel: "block-projection-llama3-rope",
      sentinelBase: -7510,
      expectShape({ executor, program: blockProgram, shape }) {
        if (
          shape.ropeKind !== "llama3" ||
          shape.ropeKindId !== 2 ||
          shape.ropeScale !== 8 ||
          shape.ropeLowFreqFactor !== 1 ||
          shape.ropeHighFreqFactor !== 4 ||
          shape.ropeOriginalContextLength !== 16 ||
          blockProgram.kvRequirements.layers !== groupedTinyLlamaShape.layers ||
          executor.ropeKind !== "llama3" ||
          executor.ropeScale !== 8 ||
          executor.ropeBase !== 13500 ||
          executor.epsilon !== 0.000045
        ) {
          throw new Error("host-only WebGPU Llama3 standalone block did not infer grouped Llama3 RoPE shape");
        }
      },
      expectBehavior({ blockBytes, decodePosition, decodeToken, expectedDecode, kElements, options, prefillTokens, vElements }) {
        const defaultRopeCache = {
          k: Array(kElements).fill(-7520),
          v: Array(vElements).fill(-7530),
        };
        const defaultRopeOptions = { ...options, ropeKind: "none", ropeScale: 1 };
        for (let i = 0; i < prefillTokens.length; i += 1) {
          tinyLlamaBlockProjectionStep(blockBytes, defaultRopeCache, prefillTokens[i], i, defaultRopeOptions);
        }
        const defaultRopeDecode = tinyLlamaBlockProjectionStep(blockBytes, defaultRopeCache, decodeToken, decodePosition, defaultRopeOptions);
        if (!expectedDecode.k.some((value, i) => Math.abs(value - defaultRopeDecode.k[i]) > 1e-5)) {
          throw new Error("host-only WebGPU Llama3 standalone block RoPE scaling did not change K cache versus default RoPE");
        }
      },
      expectCall(call) {
        if (
          call.ropeKind !== "llama3" ||
          call.ropeKindId !== 2 ||
          call.ropeScale !== 8 ||
          call.biasMask !== 0 ||
          call.qkProjectionNorm !== false
        ) {
          throw new Error("host-only WebGPU Llama3 standalone block executor family evidence mismatch");
        }
      },
    });

    await runStandaloneBlockVariantProof({
      config: smollm3NopeProofConfig(groupedTinyLlamaShape),
      label: "host-only WebGPU SmolLM3 standalone block",
      layer: 1,
      profileLabel: "block-projection-smollm3-nope",
      sentinelBase: -7610,
      expectShape({ executor, program: blockProgram, shape }) {
        if (
          !Array.isArray(shape.useRopeLayers) ||
          shape.useRopeLayers.length !== groupedTinyLlamaShape.layers ||
          shape.useRopeLayers[0] !== 1 ||
          shape.useRopeLayers[1] !== 0 ||
          blockProgram.kvRequirements.layers !== groupedTinyLlamaShape.layers ||
          executor.layer !== 1 ||
          executor.useRope !== false ||
          executor.ropeBase !== 18000 ||
          executor.epsilon !== 0.000085
        ) {
          throw new Error("host-only WebGPU SmolLM3 standalone block did not infer grouped NoPE shape");
        }
      },
      expectBehavior({ blockBytes, decodePosition, decodeToken, expectedDecode, kElements, options, prefillTokens, vElements }) {
        const ropeCache = {
          k: Array(kElements).fill(-7620),
          v: Array(vElements).fill(-7630),
        };
        const ropeOptions = { ...options, useRope: true };
        for (let i = 0; i < prefillTokens.length; i += 1) {
          tinyLlamaBlockProjectionStep(blockBytes, ropeCache, prefillTokens[i], i, ropeOptions);
        }
        const ropeDecode = tinyLlamaBlockProjectionStep(blockBytes, ropeCache, decodeToken, decodePosition, ropeOptions);
        if (!expectedDecode.k.some((value, i) => Math.abs(value - ropeDecode.k[i]) > 1e-5)) {
          throw new Error("host-only WebGPU SmolLM3 standalone block NoPE did not change K cache versus RoPE");
        }
      },
      expectCall(call) {
        if (
          call.layer !== 1 ||
          call.useRope !== false ||
          call.biasMask !== 0 ||
          call.qkProjectionNorm !== false
        ) {
          throw new Error("host-only WebGPU SmolLM3 standalone block executor family evidence mismatch");
        }
      },
    });

    const blockPipelineExecutorCalls = [];
    const blockPipelineExecutor = new WasmWebGpuLlamaBlockPipelineExecutor({
      layers: [0, 0],
      record(call) {
        recordBrowserLlamaPipelineCallEvidence(blockPipelineExecutorCalls, "block-pipeline", call);
      },
      splitTerminalLogitsMinHiddenSize: 1,
    });
    const blockPipelineStageContexts = [];
    const blockPipelineStageSlotEvidence = [];
    for (const entry of blockPipelineExecutor.blocks) {
      const executor = entry.executor;
      executor.record = (_call, context) => {
        blockPipelineStageContexts.push(context);
        blockPipelineStageSlotEvidence.push({
          executor,
          k: executor.kvWindowScratch.k,
          slots: executor.kvWindowScratch,
          v: executor.kvWindowScratch.v,
        });
      };
    }
    const blockPipelineFinalLogitsContexts = [];
    blockPipelineExecutor.finalLogitsExecutor.record = (_call, context) => blockPipelineFinalLogitsContexts.push(context);
    const blockPipelineProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      requiredModelResources: tinyLlamaModelResourceRequirements,
      allowExtraModelResources: false,
      tokenExecutor: blockPipelineExecutor,
    });
    const blockPipelineActivation = blockPipelineProgram.createActivationBuffer({ hiddenSize: 4 });
    const blockPipelineSession = blockPipelineProgram.bind({
      ...rawLlamaResources,
      activation: blockPipelineActivation,
      modelResources: llamaModelResources,
    }, { model: compatibleModel });
    const blockPipelineSessionHandle = blockPipelineSession.handle;
    let blockPipelineWorkspaceResources = [];
    try {
      const pipelineK = blockPipelineSession.bindings.kvCache[0].k;
      const pipelineV = blockPipelineSession.bindings.kvCache[0].v;
      const pipelineActivation = blockPipelineSession.bindings.activation;
      const pipelineKElements = Math.floor(pipelineK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const pipelineVElements = Math.floor(pipelineV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const pipelineActivationElements = Math.floor(pipelineActivation.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(pipelineK, Array(pipelineKElements).fill(-6060));
      hostDevice.writeFloat32(pipelineV, Array(pipelineVElements).fill(-7070));
      hostDevice.writeFloat32(pipelineActivation, Array(pipelineActivationElements).fill(-8080));
      hostDevice.writeFloat32(blockPipelineSession.bindings.output, Array(tinyLlamaVocabSize).fill(-9090));
      const pipelineTokens = keep(alloc(8), 8);
      writeU32Array(pipelineTokens, [0, 3]);
      const pipelineExecute = tokenExecute(blockPipelineSessionHandle, pipelineTokens, 2, executeOutputLogits, 0, 0);
      expectStatus(pipelineExecute.status, ok, "model-bound WebGPU tiny LLaMA block pipeline prefill");
      if (pipelineExecute.outputLen !== tinyLlamaVocabSize || sessionPosition(blockPipelineSessionHandle) !== 2) {
        throw new Error("model-bound WebGPU tiny LLaMA block pipeline did not advance");
      }
      const expectedPipelineCache = {
        k: Array(pipelineKElements).fill(-6060),
        v: Array(pipelineVElements).fill(-7070),
      };
      const expectedPipelineStage0First = tinyLlamaBlockProjectionStep(llamaModelResourceBytes, expectedPipelineCache, 0, 0);
      const expectedPipelineStage0Last = tinyLlamaBlockProjectionStep(llamaModelResourceBytes, expectedPipelineCache, 3, 1);
      const expectedPipelineStage1First = tinyLlamaBlockProjectionStep(llamaModelResourceBytes, expectedPipelineCache, 0, 0, {
        inputHidden: expectedPipelineStage0First.blockHidden,
      });
      const expectedPipelineStage1Last = tinyLlamaBlockProjectionStep(llamaModelResourceBytes, expectedPipelineCache, 3, 1, {
        inputHidden: expectedPipelineStage0Last.blockHidden,
      });
      expectClose(
        Array.from(await hostDevice.readFloat32(blockPipelineSession.bindings.output, tinyLlamaVocabSize)),
        expectedPipelineStage1Last.logits,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(pipelineK, expectedPipelineStage1First.k.length, 0)),
        expectedPipelineStage1First.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(pipelineV, expectedPipelineStage1First.v.length, 0)),
        expectedPipelineStage1First.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(pipelineK, expectedPipelineStage1Last.k.length, expectedPipelineStage1Last.k.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedPipelineStage1Last.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(pipelineV, expectedPipelineStage1Last.v.length, expectedPipelineStage1Last.v.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedPipelineStage1Last.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(pipelineActivation, expectedPipelineStage1First.blockHidden.length, 0)),
        expectedPipelineStage1First.blockHidden,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(pipelineActivation, expectedPipelineStage1Last.blockHidden.length, expectedPipelineStage1Last.blockHidden.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedPipelineStage1Last.blockHidden,
      );
      const expectedPipelineCommandCount = expectedLlamaBlockPipelineCommandCount(hostDevice, 2, 2, executeOutputLogits, {
        contextLength: 4,
        hiddenSize: 4,
        splitTerminalLogits: true,
      });
      const expectedPipelineDispatchFamilies = expectedLlamaBlockPipelineDispatchFamilies(hostDevice, 2, 2, executeOutputLogits, {
        contextLength: 4,
        hiddenSize: 4,
        splitTerminalLogits: true,
      });
      const expectedPipelineBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? expectedPipelineCommandCount : 0;
      const expectedPipelineFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : expectedPipelineCommandCount;
      const expectedPipelineUsesGpuStorageBuffers = canUseGpuStorageBuffers(hostDevice, 6);
      const pipelineProfile = blockPipelineSession.runtimeProfile();
      if (
        pipelineProfile.executorCallCount !== 1 ||
        pipelineProfile.executorOkCount !== 1 ||
        pipelineProfile.executorBackendDispatchCount !== expectedPipelineBackendDispatchCount ||
        pipelineProfile.executorFallbackOpCount !== expectedPipelineFallbackOpCount ||
        pipelineProfile.executorCommandCount !== expectedPipelineCommandCount ||
        pipelineProfile.executorCommandOpCount !== expectedPipelineCommandCount ||
        pipelineProfile.executorGenericDispatchCount !== expectedPipelineDispatchFamilies.generic ||
        pipelineProfile.executorScalarDispatchCount !== expectedPipelineDispatchFamilies.scalar ||
        pipelineProfile.executorWindowDispatchCount !== expectedPipelineDispatchFamilies.window ||
        pipelineProfile.lastStatus !== ok ||
        pipelineProfile.lastResultOutputLength !== tinyLlamaVocabSize
      ) {
        throw new Error(`unexpected model-bound WebGPU tiny LLaMA block pipeline profile: ${JSON.stringify(pipelineProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("block-pipeline", pipelineProfile);
      if (
        blockPipelineExecutorCalls.length !== 1 ||
        blockPipelineExecutorCalls[0].stageCount !== 2 ||
        blockPipelineExecutorCalls[0].workspaceCount !== 1 ||
        blockPipelineExecutorCalls[0].commandCount !== expectedPipelineCommandCount ||
        blockPipelineExecutorCalls[0].deviceMode !== expectedLlamaExecutorDeviceMode(hostDevice) ||
        blockPipelineExecutorCalls[0].genericDispatchCount !== expectedPipelineDispatchFamilies.generic ||
        blockPipelineExecutorCalls[0].maxStorageBuffersPerShaderStage !== hostDevice.maxStorageBuffersPerShaderStage() ||
        blockPipelineExecutorCalls[0].requiredStorageBufferCount !== 6 ||
        blockPipelineExecutorCalls[0].scalarDispatchCount !== expectedPipelineDispatchFamilies.scalar ||
        blockPipelineExecutorCalls[0].splitTerminalLogits !== true ||
        blockPipelineExecutorCalls[0].status !== ok ||
        blockPipelineExecutorCalls[0].usesGpuStorageBuffers !== expectedPipelineUsesGpuStorageBuffers ||
        blockPipelineExecutorCalls[0].windowDispatchCount !== expectedPipelineDispatchFamilies.window
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA block pipeline evidence mismatch");
      }
      if (
        blockPipelineStageContexts.length !== 2 ||
        blockPipelineStageContexts[0] !== blockPipelineExecutor.stageContextScratch ||
        blockPipelineStageContexts[1] !== blockPipelineExecutor.stageContextScratch ||
        blockPipelineStageContexts[0].execution !== blockPipelineExecutor.stageContextScratch.execution ||
        blockPipelineStageContexts[0].stepParams !== blockPipelineExecutor.stageContextScratch.stepParams ||
        blockPipelineStageContexts[0].resultScratch !== blockPipelineExecutor.stageResultScratch ||
        blockPipelineFinalLogitsContexts.length !== 1 ||
        blockPipelineFinalLogitsContexts[0] !== blockPipelineExecutor.finalLogitsContextScratch ||
        blockPipelineFinalLogitsContexts[0].resultScratch !== blockPipelineExecutor.finalLogitsResultScratch ||
        blockPipelineExecutor.stageResultScratch.status !== ok ||
        blockPipelineExecutor.stageResultScratch.outputLength !== 0 ||
        blockPipelineExecutor.finalLogitsResultScratch.status !== ok ||
        blockPipelineExecutor.finalLogitsResultScratch.outputLength !== tinyLlamaVocabSize ||
        blockPipelineSession.tokenExecutorResultScratch.status !== ok ||
        blockPipelineSession.tokenExecutorResultScratch.outputLength !== tinyLlamaVocabSize ||
        blockPipelineSession.tokenExecutorResultScratch.commandCount !== expectedPipelineCommandCount ||
        blockPipelineSession.tokenExecutorResultScratch.genericDispatchCount !== expectedPipelineDispatchFamilies.generic ||
        blockPipelineSession.tokenExecutorResultScratch.scalarDispatchCount !== expectedPipelineDispatchFamilies.scalar ||
        blockPipelineSession.tokenExecutorResultScratch.usesGpuStorageBuffers !== expectedPipelineUsesGpuStorageBuffers ||
        blockPipelineSession.tokenExecutorResultScratch.windowDispatchCount !== expectedPipelineDispatchFamilies.window
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA block pipeline did not reuse context/result scratch");
      }
      if (
        !canUseGpuStorageBuffers(hostDevice, 6) &&
        blockPipelineExecutor.finalLogitsExecutor.mockLogitsScratch.length !== tinyLlamaVocabSize
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA block pipeline final-logits fallback did not retain logits scratch");
      }
      if (
        blockPipelineStageSlotEvidence.length !== 2 ||
        blockPipelineStageSlotEvidence[0].slots !== blockPipelineExecutor.blocks[0].executor.kvWindowScratch ||
        blockPipelineStageSlotEvidence[1].slots !== blockPipelineExecutor.blocks[1].executor.kvWindowScratch ||
        blockPipelineStageSlotEvidence[0].k !== blockPipelineExecutor.blocks[0].executor.kvWindowScratch.k ||
        blockPipelineStageSlotEvidence[0].v !== blockPipelineExecutor.blocks[0].executor.kvWindowScratch.v ||
        blockPipelineStageSlotEvidence[1].k !== blockPipelineExecutor.blocks[1].executor.kvWindowScratch.k ||
        blockPipelineStageSlotEvidence[1].v !== blockPipelineExecutor.blocks[1].executor.kvWindowScratch.v ||
        blockPipelineStageSlotEvidence[0].k.byteOffset !== 0 ||
        blockPipelineStageSlotEvidence[0].k.elementLength !== 2 * blockPipelineStageSlotEvidence[0].k.stride ||
        blockPipelineStageSlotEvidence[1].k.byteOffset !== 0 ||
        blockPipelineStageSlotEvidence[1].k.elementLength !== 2 * blockPipelineStageSlotEvidence[1].k.stride
      ) {
        throw new Error("model-bound WebGPU tiny LLaMA block pipeline did not reuse K/V slot-window scratch");
      }
      blockPipelineWorkspaceResources = blockPipelineSession.ownedResources.filter((resource) =>
        typeof resource?.label === "string" &&
        resource.label.includes("llama.block-pipeline.activation")
      );
      if (blockPipelineWorkspaceResources.length !== 1) {
        throw new Error(`model-bound WebGPU tiny LLaMA block pipeline expected one Session-owned activation workspace, found ${blockPipelineWorkspaceResources.length}`);
      }
      for (const resource of blockPipelineWorkspaceResources) {
        if ("destroyed" in resource && resource.destroyed) {
          throw new Error("model-bound WebGPU tiny LLaMA block pipeline activation workspace was destroyed before Session free");
        }
      }
    } finally {
      exportsRef.zgml_session_free(blockPipelineSessionHandle);
      if (blockPipelineSession.ownedResources.length !== 0) {
        throw new Error("model-bound WebGPU tiny LLaMA block pipeline Session-owned resources were not released");
      }
      for (const resource of blockPipelineWorkspaceResources) {
        if ("destroyed" in resource && !resource.destroyed) {
          throw new Error("model-bound WebGPU tiny LLaMA block pipeline activation workspace was not destroyed on Session free");
        }
      }
      blockPipelineExecutor.destroy?.();
    }

    const twoLayerPipelineExecutorCalls = [];
    const twoLayerModelResourceBytes = tinyLlamaModelResourceFileBytes(tinyLlamaTwoLayerTensorSpecs);
    const twoLayerModelResourceRequirements = tinyLlamaModelResourceSpecs(twoLayerModelResourceBytes);
    const strictUncatalogedModelProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: 2,
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      allowExtraModelResources: false,
    });
    let strictUncatalogedModelResourceRejected = false;
    try {
      strictUncatalogedModelProgram.bindHostResources({
        modelResources: [{
          role: "llama.weight.uncataloged",
          dtype: "f32",
          shape: [1],
          resource: hostGpuBuffer("tiny-llama.uncataloged-model-resource", Float32Array.BYTES_PER_ELEMENT),
        }],
      }, { model: compatibleModel });
    } catch (err) {
      strictUncatalogedModelResourceRejected = String(err && err.message ? err.message : err).includes("unexpected LLaMA model resource");
    }
    if (!strictUncatalogedModelResourceRejected) {
      throw new Error("strict WebGPU tiny LLaMA model-resource bind accepted uncataloged resources");
    }
    const arbitraryMapDefaults = {
      "llama.weight.manual_a": {
        dtype: "f32",
        shape: [2],
        resource: hostGpuBuffer("tiny-llama.manual-model-a", 2 * Float32Array.BYTES_PER_ELEMENT),
      },
      "llama.weight.manual_b": {
        dtype: "f32",
        shape: [1],
        resource: hostGpuBuffer("tiny-llama.manual-model-b", Float32Array.BYTES_PER_ELEMENT),
      },
    };
    const arbitraryMapProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: 1,
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      allowExtraModelResources: false,
      defaultModelResources: arbitraryMapDefaults,
    });
    if (
      arbitraryMapProgram.requiredModelResources.length !== 2 ||
      arbitraryMapProgram.requiredModelResources[0].role !== "llama.weight.manual_a" ||
      arbitraryMapProgram.requiredModelResources[0].byteLength !== 2 * Float32Array.BYTES_PER_ELEMENT ||
      arbitraryMapProgram.requiredModelResources[1].role !== "llama.weight.manual_b"
    ) {
      throw new Error("strict WebGPU tiny LLaMA arbitrary model-resource map did not derive a Program manifest");
    }
    const arbitraryMapSession = arbitraryMapProgram.bindHostResources({}, { model: compatibleModel });
    if (
      arbitraryMapSession.modelManifest?.resourceCount !== 2 ||
      arbitraryMapSession.modelTable?.resourceCount !== 2
    ) {
      throw new Error("strict WebGPU tiny LLaMA arbitrary model-resource map did not bind exact default resources");
    }
    arbitraryMapProgram.unregister(arbitraryMapSession);
    arbitraryMapSession.dispose();
    let arbitraryMapExtraRejected = false;
    try {
      arbitraryMapProgram.bindHostResources({
        modelResources: {
          ...arbitraryMapDefaults,
          "llama.weight.manual_extra": {
            dtype: "f32",
            shape: [1],
            resource: hostGpuBuffer("tiny-llama.manual-model-extra", Float32Array.BYTES_PER_ELEMENT),
          },
        },
      }, { model: compatibleModel });
    } catch (err) {
      arbitraryMapExtraRejected = String(err && err.message ? err.message : err).includes("unexpected LLaMA model resource");
    }
    if (!arbitraryMapExtraRejected) {
      throw new Error("strict WebGPU tiny LLaMA arbitrary model-resource Program accepted an extra role");
    }
    const strictDefaultTwoLayerExecutorCalls = [];
    const strictDefaultTwoLayerProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      contextLength: 4,
      model: compatibleModel,
      safetensors: twoLayerModelResourceBytes,
      executorOptions: {
        record(call) {
          recordBrowserLlamaPipelineCallEvidence(strictDefaultTwoLayerExecutorCalls, "strict-default-two-layer", call);
        },
      },
    });
    if (strictDefaultTwoLayerProgram.inspect().gpuDispatchRequired !== true) {
      throw new Error("default WebGPU tiny LLaMA safetensors Program did not require GPU dispatch");
    }
    const strictDefaultTwoLayerSession = strictDefaultTwoLayerProgram.bindHostResources();
    const strictDefaultTwoLayerHandle = strictDefaultTwoLayerSession.handle;
    const strictDefaultTwoLayerHostExports = hostRuntime.wrapExports(exportsRef);
    try {
      if (strictDefaultTwoLayerSession.requireGpuDispatch !== true) {
        throw new Error("default WebGPU tiny LLaMA safetensors Session did not inherit GPU-required execution");
      }
      const strictDefaultOutputSentinel = Array(tinyLlamaVocabSize).fill(-8140);
      const strictDefaultK = strictDefaultTwoLayerSession.bindings.kvCache[0].k;
      const strictDefaultV = strictDefaultTwoLayerSession.bindings.kvCache[0].v;
      const strictDefaultKElements = Math.floor(strictDefaultK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const strictDefaultVElements = Math.floor(strictDefaultV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const strictDefaultKSentinel = Array(strictDefaultKElements).fill(-8150);
      const strictDefaultVSentinel = Array(strictDefaultVElements).fill(-8160);
      hostDevice.writeFloat32(strictDefaultTwoLayerSession.bindings.output, strictDefaultOutputSentinel);
      hostDevice.writeFloat32(strictDefaultK, strictDefaultKSentinel);
      hostDevice.writeFloat32(strictDefaultV, strictDefaultVSentinel);
      const strictDefaultToken = keep(alloc(4), 4);
      writeU32Array(strictDefaultToken, [2]);
      const strictDefaultResult = tokenExecuteWith(
        strictDefaultTwoLayerHostExports,
        strictDefaultTwoLayerHandle,
        strictDefaultToken,
        1,
        executeOutputLogits,
        0,
        0,
      );
      const strictDefaultCanDispatch = canUseGpuStorageBuffers(hostDevice, 6);
      expectStatus(
        strictDefaultResult.status,
        strictDefaultCanDispatch ? ok : unsupported,
        "default WebGPU tiny LLaMA safetensors strict raw execute",
      );
      const strictDefaultProfile = strictDefaultTwoLayerSession.runtimeProfile();
      if (strictDefaultCanDispatch) {
        if (
          strictDefaultResult.outputLen !== tinyLlamaVocabSize ||
          sessionPositionWith(strictDefaultTwoLayerHostExports, strictDefaultTwoLayerHandle) !== 1 ||
          strictDefaultProfile.executorCallCount !== 1 ||
          strictDefaultProfile.executorOkCount !== 1 ||
          strictDefaultProfile.executorBackendDispatchCount <= 0 ||
          strictDefaultProfile.executorFallbackOpCount !== 0 ||
          strictDefaultProfile.lastStatus !== ok
        ) {
          throw new Error(`default WebGPU tiny LLaMA safetensors strict dispatch evidence mismatch: ${JSON.stringify(strictDefaultProfile)}`);
        }
        if (
          strictDefaultTwoLayerExecutorCalls.length !== 1 ||
          strictDefaultTwoLayerExecutorCalls[0].usesGpuStorageBuffers !== true ||
          strictDefaultTwoLayerExecutorCalls[0].backendDispatchCount <= 0 ||
          strictDefaultTwoLayerExecutorCalls[0].fallbackOpCount !== 0 ||
          strictDefaultTwoLayerExecutorCalls[0].status !== ok
        ) {
          throw new Error(`default WebGPU tiny LLaMA safetensors strict executor evidence mismatch: ${JSON.stringify(strictDefaultTwoLayerExecutorCalls)}`);
        }
      } else {
        if (
          strictDefaultResult.outputLen !== 0 ||
          sessionPositionWith(strictDefaultTwoLayerHostExports, strictDefaultTwoLayerHandle) !== 0 ||
          strictDefaultProfile.executorCallCount !== 0 ||
          strictDefaultProfile.executorBackendDispatchCount !== 0 ||
          strictDefaultProfile.executorFallbackOpCount !== 0 ||
          strictDefaultProfile.unsupportedCallCount !== 1 ||
          strictDefaultProfile.lastStatus !== unsupported ||
          strictDefaultProfile.lastResultOutputLength !== 0
        ) {
          throw new Error(`default WebGPU tiny LLaMA safetensors strict unsupported evidence mismatch: ${JSON.stringify(strictDefaultProfile)}`);
        }
        expectClose(
          Array.from(await hostDevice.readFloat32(strictDefaultTwoLayerSession.bindings.output, tinyLlamaVocabSize)),
          strictDefaultOutputSentinel,
        );
        expectClose(Array.from(await hostDevice.readFloat32(strictDefaultK, strictDefaultKElements)), strictDefaultKSentinel);
        expectClose(Array.from(await hostDevice.readFloat32(strictDefaultV, strictDefaultVElements)), strictDefaultVSentinel);
        if (strictDefaultTwoLayerExecutorCalls.length !== 0) {
          throw new Error("default WebGPU tiny LLaMA safetensors strict unsupported path invoked executor");
        }
      }
      recordBrowserLlamaProfileEvidence("strict-default-two-layer", strictDefaultProfile);
    } finally {
      strictDefaultTwoLayerSession.dispose();
    }
    const twoLayerPipelineProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      contextLength: 4,
      model: compatibleModel,
      safetensors: twoLayerModelResourceBytes,
      allowMockFallback: true,
      executorOptions: {
        record(call) {
          recordBrowserLlamaPipelineCallEvidence(twoLayerPipelineExecutorCalls, "two-layer-pipeline", call);
        },
      },
    });
    if (!(twoLayerPipelineProgram.tokenExecutor instanceof WasmWebGpuLlamaBlockPipelineExecutor)) {
      throw new Error("host-only WebGPU tiny LLaMA two-layer safetensors factory did not create a block pipeline executor");
    }
    if (
      twoLayerPipelineProgram.allowExtraModelResources !== false ||
      twoLayerPipelineProgram.defaultModelHandle !== compatibleModel ||
      twoLayerPipelineProgram.vocabSize !== tinyLlamaVocabSize ||
      twoLayerPipelineProgram.hiddenSize !== 4 ||
      twoLayerPipelineProgram.kvRequirements.layers !== 2 ||
      twoLayerPipelineProgram.kvRequirements.kBufferByteLength !== u32(kvRequirements, 20) ||
      twoLayerPipelineProgram.kvRequirements.vBufferByteLength !== u32(kvRequirements, 24)
    ) {
      throw new Error("host-only WebGPU tiny LLaMA two-layer Program did not infer shape from safetensors");
    }
    const expectedTwoLayerActivationByteLength = 4 * 4 * Float32Array.BYTES_PER_ELEMENT;
    const twoLayerProgramInspection = twoLayerPipelineProgram.inspect();
    const expectedTwoLayerModelRoles = twoLayerModelResourceRequirements.map((entry) => entry.role);
    if (
      twoLayerProgramInspection.kind !== "llama-resource-program" ||
      twoLayerProgramInspection.executionCoverage !== "bounded-proof" ||
      twoLayerProgramInspection.executionCoverageReason !== "llama-family-bounded-proof-executor" ||
      twoLayerProgramInspection.executionMode !== "proof-executor" ||
      twoLayerProgramInspection.tokenExecutionSupported !== true ||
      twoLayerProgramInspection.boundedProofExecutionSupported !== true ||
      twoLayerProgramInspection.fullDefaultExecutionSupported !== false ||
      twoLayerProgramInspection.tokenExecutorKind !== "tiny-llama-block-pipeline" ||
      twoLayerProgramInspection.modelKind !== tinyLlama2LayerKind ||
      twoLayerProgramInspection.vocabSize !== tinyLlamaVocabSize ||
      twoLayerProgramInspection.hiddenSize !== 4 ||
      twoLayerProgramInspection.contextLength !== 4 ||
      twoLayerProgramInspection.outputByteLength !== tinyLlamaVocabSize * Float32Array.BYTES_PER_ELEMENT ||
      twoLayerProgramInspection.terminalActivationByteLength !== expectedTwoLayerActivationByteLength ||
      twoLayerProgramInspection.kvCache.layers !== 2 ||
      twoLayerProgramInspection.kvCache.kBufferByteLength !== u32(kvRequirements, 20) ||
      twoLayerProgramInspection.kvCache.vBufferByteLength !== u32(kvRequirements, 24) ||
      twoLayerProgramInspection.bindingRequirementHash === 0n ||
      twoLayerProgramInspection.persistentRequirementCount !== expectedTwoLayerModelRoles.length + 4 ||
      twoLayerProgramInspection.stepInputRequirementCount !== 1 ||
      twoLayerProgramInspection.stepOutputRequirementCount !== 1 ||
      twoLayerProgramInspection.allowExtraModelResources !== false ||
      twoLayerProgramInspection.requiredModelResourceCount !== twoLayerModelResourceRequirements.length ||
      twoLayerProgramInspection.requiredModelResourceRoles.length !== expectedTwoLayerModelRoles.length ||
      twoLayerProgramInspection.requiredModelResourceRoles.some((role, index) => role !== expectedTwoLayerModelRoles[index]) ||
      twoLayerProgramInspection.requiredGpuStorageBufferCount !== 6 ||
      twoLayerProgramInspection.maxStorageBuffersPerShaderStage !== hostDevice.maxStorageBuffersPerShaderStage() ||
      twoLayerProgramInspection.canCreateGpuBuffers !== hostDevice.canCreateGpuBuffers() ||
      twoLayerProgramInspection.canBindRequiredGpuStorageBuffers !== canUseGpuStorageBuffers(hostDevice, 6) ||
      twoLayerProgramInspection.gpuDispatchRequired !== false ||
      twoLayerProgramInspection.defaultModelHandle !== compatibleModel ||
      twoLayerProgramInspection.hasDefaultModelResources !== true
    ) {
      throw new Error(`host-only WebGPU tiny LLaMA two-layer Program inspection mismatch: ${JSON.stringify(twoLayerProgramInspection, (_, value) => typeof value === "bigint" ? value.toString() : value)}`);
    }
    try {
      twoLayerProgramInspection.requiredModelResourceRoles[0] = "llama.weight.mutated";
    } catch {}
    if (
      twoLayerPipelineProgram.requiredModelResources[0].role !== expectedTwoLayerModelRoles[0] ||
      twoLayerProgramInspection.requiredModelResourceRoles[0] !== expectedTwoLayerModelRoles[0]
    ) {
      throw new Error("host-only WebGPU tiny LLaMA Program inspection exposed mutable resource roles");
    }
    if (
      twoLayerPipelineProgram.tokenExecutor.blocks.length !== 2 ||
      twoLayerPipelineProgram.tokenExecutor.blocks[0].executor.layer !== 0 ||
      twoLayerPipelineProgram.tokenExecutor.blocks[1].executor.layer !== 1
    ) {
      throw new Error("host-only WebGPU tiny LLaMA two-layer preset did not infer layers from K/V requirements");
    }
    const twoLayerInferredActivation = twoLayerPipelineProgram.createActivationBuffer();
    if (
      twoLayerPipelineProgram.activationByteLength() !== expectedTwoLayerActivationByteLength ||
      twoLayerInferredActivation.elementCount !== expectedTwoLayerActivationByteLength / Float32Array.BYTES_PER_ELEMENT ||
      twoLayerInferredActivation.descriptor.byteLength !== expectedTwoLayerActivationByteLength
    ) {
      throw new Error("host-only WebGPU tiny LLaMA two-layer Program did not infer activation size from safetensors");
    }
    twoLayerInferredActivation.resource?.destroy?.();
    const gpuRequiredTwoLayerRecord = twoLayerPipelineProgram.tokenExecutor.record;
    twoLayerPipelineProgram.tokenExecutor.record = null;
    const gpuRequiredTwoLayerSession = twoLayerPipelineProgram.bindHostResources({}, { requireGpuDispatch: true });
    const gpuRequiredTwoLayerHandle = gpuRequiredTwoLayerSession.handle;
    const gpuRequiredTwoLayerHostExports = hostRuntime.wrapExports(exportsRef);
    try {
      if (gpuRequiredTwoLayerSession.requireGpuDispatch !== true) {
        throw new Error("host-only WebGPU tiny LLaMA GPU-required Session did not retain requireGpuDispatch");
      }
      const gpuRequiredOutputSentinel = Array(tinyLlamaVocabSize).fill(-9140);
      const gpuRequiredK = gpuRequiredTwoLayerSession.bindings.kvCache[0].k;
      const gpuRequiredV = gpuRequiredTwoLayerSession.bindings.kvCache[0].v;
      const gpuRequiredKElements = Math.floor(gpuRequiredK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const gpuRequiredVElements = Math.floor(gpuRequiredV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const gpuRequiredKSentinel = Array(gpuRequiredKElements).fill(-9150);
      const gpuRequiredVSentinel = Array(gpuRequiredVElements).fill(-9160);
      hostDevice.writeFloat32(gpuRequiredTwoLayerSession.bindings.output, gpuRequiredOutputSentinel);
      hostDevice.writeFloat32(gpuRequiredK, gpuRequiredKSentinel);
      hostDevice.writeFloat32(gpuRequiredV, gpuRequiredVSentinel);
      const gpuRequiredToken = keep(alloc(4), 4);
      writeU32Array(gpuRequiredToken, [2]);
      const gpuRequiredDirectSelection = await gpuRequiredTwoLayerSession.argmaxDevice({
        logits: new Float32Array([1, 3, 2]),
      });
      expectStatus(gpuRequiredDirectSelection.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required direct device argmax fallback");
      const gpuRequiredDirectSample = await gpuRequiredTwoLayerSession.sampleDevice({
        logits: new Float32Array([1, 3, 2]),
        seed: 17,
        temperature: 1,
        topK: 1,
      });
      expectStatus(gpuRequiredDirectSample.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required direct device sample logits fallback");
      const gpuRequiredDirectSampleCpu = await gpuRequiredTwoLayerSession.sampleDevice({
        gpu: false,
        seed: 17,
        temperature: 1,
        topK: 1,
      });
      expectStatus(gpuRequiredDirectSampleCpu.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required direct device sample CPU fallback");
      const gpuRequiredHostArgmax = await gpuRequiredTwoLayerSession.argmax({
        logits: new Float32Array([1, 3, 2]),
      });
      expectStatus(gpuRequiredHostArgmax.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required direct host argmax fallback");
      const gpuRequiredHostSample = await gpuRequiredTwoLayerSession.sample({
        logits: new Float32Array([1, 3, 2]),
        seed: 17,
        temperature: 1,
        topK: 1,
      });
      expectStatus(gpuRequiredHostSample.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required direct host sample fallback");
      const gpuRequiredHostArgmaxReadback = await gpuRequiredTwoLayerSession.argmax();
      expectStatus(gpuRequiredHostArgmaxReadback.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required direct host argmax readback fallback");
      const gpuRequiredHostSampleReadback = await gpuRequiredTwoLayerSession.sample({
        seed: 17,
        temperature: 1,
        topK: 1,
      });
      expectStatus(gpuRequiredHostSampleReadback.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required direct host sample readback fallback");
      const gpuRequiredHostArgmaxSelection = await gpuRequiredTwoLayerSession.executeTokensArgmax([2]);
      expectStatus(gpuRequiredHostArgmaxSelection.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required execute host argmax fallback");
      const gpuRequiredHostSampleSelection = await gpuRequiredTwoLayerSession.executeTokensSample([2], {
        seed: 17,
        temperature: 1,
        topK: 1,
      });
      expectStatus(gpuRequiredHostSampleSelection.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required execute host sample fallback");
      const gpuRequiredHostGreedyTokens = new Uint32Array([81, 82]);
      const gpuRequiredHostGreedyGeneration = await gpuRequiredTwoLayerSession.generateTokensArgmaxInto(
        [2],
        gpuRequiredHostGreedyTokens,
      );
      expectStatus(gpuRequiredHostGreedyGeneration.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required host greedy generation fallback");
      if (
        gpuRequiredHostGreedyGeneration.tokensGenerated !== 0 ||
        gpuRequiredHostGreedyTokens[0] !== 81 ||
        gpuRequiredHostGreedyTokens[1] !== 82
      ) {
        throw new Error("host-only WebGPU tiny LLaMA GPU-required host greedy generation mutated output tokens");
      }
      const gpuRequiredHostSampleTokens = new Uint32Array([83, 84]);
      const gpuRequiredHostSampleGeneration = await gpuRequiredTwoLayerSession.generateTokensSampleInto(
        [2],
        gpuRequiredHostSampleTokens,
        {
          seed: 17,
          temperature: 1,
          topK: 1,
        },
      );
      expectStatus(gpuRequiredHostSampleGeneration.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required host sampled generation fallback");
      if (
        gpuRequiredHostSampleGeneration.tokensGenerated !== 0 ||
        gpuRequiredHostSampleTokens[0] !== 83 ||
        gpuRequiredHostSampleTokens[1] !== 84
      ) {
        throw new Error("host-only WebGPU tiny LLaMA GPU-required host sampled generation mutated output tokens");
      }
      const gpuRequiredArgmaxSelection = await gpuRequiredTwoLayerSession.executeTokensArgmaxDevice([2], { gpu: false });
      expectStatus(gpuRequiredArgmaxSelection.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required execute device argmax fallback");
      const gpuRequiredSampleSelection = await gpuRequiredTwoLayerSession.executeTokensSampleDevice([2], {
        gpu: false,
        seed: 17,
        temperature: 1,
        topK: 1,
      });
      expectStatus(gpuRequiredSampleSelection.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required execute device sample fallback");
      const gpuRequiredGreedyTokens = new Uint32Array([91, 92]);
      const gpuRequiredGreedyGeneration = await gpuRequiredTwoLayerSession.generateTokensArgmaxDeviceInto(
        [2],
        gpuRequiredGreedyTokens,
        { gpu: false },
      );
      expectStatus(gpuRequiredGreedyGeneration.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required device greedy generation fallback");
      if (gpuRequiredGreedyGeneration.tokensGenerated !== 0 || gpuRequiredGreedyTokens[0] !== 91 || gpuRequiredGreedyTokens[1] !== 92) {
        throw new Error("host-only WebGPU tiny LLaMA GPU-required greedy generation mutated output tokens");
      }
      const gpuRequiredSampleTokens = new Uint32Array([93, 94]);
      const gpuRequiredSampleGeneration = await gpuRequiredTwoLayerSession.generateTokensSampleDeviceInto(
        [2],
        gpuRequiredSampleTokens,
        {
          gpu: false,
          seed: 17,
          temperature: 1,
          topK: 1,
        },
      );
      expectStatus(gpuRequiredSampleGeneration.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required device sampled generation fallback");
      if (gpuRequiredSampleGeneration.tokensGenerated !== 0 || gpuRequiredSampleTokens[0] !== 93 || gpuRequiredSampleTokens[1] !== 94) {
        throw new Error("host-only WebGPU tiny LLaMA GPU-required sampled generation mutated output tokens");
      }
      const gpuRequiredSelectionProfile = gpuRequiredTwoLayerSession.runtimeProfile();
      if (
        gpuRequiredSelectionProfile.callCount !== 0 ||
        gpuRequiredSelectionProfile.executorCallCount !== 0 ||
        gpuRequiredSelectionProfile.executorBackendDispatchCount !== 0 ||
        gpuRequiredSelectionProfile.executorFallbackOpCount !== 0 ||
        gpuRequiredSelectionProfile.selectionCallCount !== 0 ||
        gpuRequiredSelectionProfile.selectionBackendDispatchCount !== 0 ||
        gpuRequiredSelectionProfile.selectionFallbackOpCount !== 0 ||
        gpuRequiredSelectionProfile.outputReadCount !== 0 ||
        gpuRequiredSelectionProfile.syncCount !== 0 ||
        gpuRequiredSelectionProfile.lastStatus !== null ||
        sessionPositionWith(gpuRequiredTwoLayerHostExports, gpuRequiredTwoLayerHandle) !== 0
      ) {
        throw new Error(`unexpected host-only WebGPU tiny LLaMA GPU-required rejected-selection profile: ${JSON.stringify(gpuRequiredSelectionProfile)}`);
      }
      expectClose(Array.from(await hostDevice.readFloat32(gpuRequiredTwoLayerSession.bindings.output, tinyLlamaVocabSize)), gpuRequiredOutputSentinel);
      expectClose(Array.from(await hostDevice.readFloat32(gpuRequiredK, gpuRequiredKElements)), gpuRequiredKSentinel);
      expectClose(Array.from(await hostDevice.readFloat32(gpuRequiredV, gpuRequiredVElements)), gpuRequiredVSentinel);
      const gpuRequiredResult = tokenExecuteWith(gpuRequiredTwoLayerHostExports, gpuRequiredTwoLayerHandle, gpuRequiredToken, 1, executeOutputLogits, 0, 0);
      const gpuRequiredProfile = gpuRequiredTwoLayerSession.runtimeProfile();
      if (canUseGpuStorageBuffers(hostDevice, 6)) {
        expectStatus(gpuRequiredResult.status, ok, "host-only WebGPU tiny LLaMA GPU-required two-layer execute");
        if (
          gpuRequiredResult.outputLen !== tinyLlamaVocabSize ||
          sessionPositionWith(gpuRequiredTwoLayerHostExports, gpuRequiredTwoLayerHandle) !== 1 ||
          gpuRequiredProfile.executorCallCount !== 1 ||
          gpuRequiredProfile.executorOkCount !== 1 ||
          gpuRequiredProfile.executorBackendDispatchCount <= 0 ||
          gpuRequiredProfile.executorFallbackOpCount !== 0
        ) {
          throw new Error(`unexpected host-only WebGPU tiny LLaMA GPU-required dispatch profile: ${JSON.stringify(gpuRequiredProfile)}`);
        }
      } else {
        expectStatus(gpuRequiredResult.status, unsupported, "host-only WebGPU tiny LLaMA GPU-required two-layer execute");
        if (
          gpuRequiredResult.outputLen !== 0 ||
          sessionPositionWith(gpuRequiredTwoLayerHostExports, gpuRequiredTwoLayerHandle) !== 0 ||
          gpuRequiredProfile.executorCallCount !== 0 ||
          gpuRequiredProfile.executorBackendDispatchCount !== 0 ||
          gpuRequiredProfile.executorFallbackOpCount !== 0 ||
          gpuRequiredProfile.unsupportedCallCount !== 1 ||
          gpuRequiredProfile.lastStatus !== unsupported
        ) {
          throw new Error(`unexpected host-only WebGPU tiny LLaMA GPU-required preflight profile: ${JSON.stringify(gpuRequiredProfile)}`);
        }
        expectClose(Array.from(await hostDevice.readFloat32(gpuRequiredTwoLayerSession.bindings.output, tinyLlamaVocabSize)), gpuRequiredOutputSentinel);
        expectClose(Array.from(await hostDevice.readFloat32(gpuRequiredK, gpuRequiredKElements)), gpuRequiredKSentinel);
        expectClose(Array.from(await hostDevice.readFloat32(gpuRequiredV, gpuRequiredVElements)), gpuRequiredVSentinel);
      }
    } finally {
      twoLayerPipelineProgram.tokenExecutor.record = gpuRequiredTwoLayerRecord;
      expectStatus(gpuRequiredTwoLayerHostExports.zgml_session_free(gpuRequiredTwoLayerHandle), ok, "host-only WebGPU tiny LLaMA GPU-required Session free");
    }
    let nativeDefaultTwoLayerModel = 0;
    let nativeDefaultTwoLayerCompatibleModel = 0;
    let nativeDefaultTwoLayerProgramHandle = 0;
    let nativeDefaultTwoLayerSession = null;
    let nativeDefaultTwoLayerSessionHandle = 0;
    const nativeDefaultTwoLayerExecutorCalls = [];
    try {
      nativeDefaultTwoLayerModel = createModel(tinyLlama2LayerKind, 0, 0);
      nativeDefaultTwoLayerCompatibleModel = createModel(tinyLlama2LayerKind, 0, 0);
      nativeDefaultTwoLayerProgramHandle = compileProgram(nativeDefaultTwoLayerModel, 4, backendWebGpu);
      const nativeDefaultTwoLayerKvRequirements = keep(alloc(llamaKvCacheRequirementsSize), llamaKvCacheRequirementsSize);
      zero(nativeDefaultTwoLayerKvRequirements, llamaKvCacheRequirementsSize);
      check(exportsRef.zgml_llama_program_get_kv_cache_requirements(nativeDefaultTwoLayerProgramHandle, nativeDefaultTwoLayerKvRequirements));
      const nativeDefaultTwoLayerProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program: nativeDefaultTwoLayerProgramHandle,
        contextLength: 4,
        model: nativeDefaultTwoLayerCompatibleModel,
        safetensors: twoLayerModelResourceBytes,
        allowMockFallback: true,
        executorOptions: {
          record(call) {
            recordBrowserLlamaPipelineCallEvidence(nativeDefaultTwoLayerExecutorCalls, "native-default-two-layer", call);
          },
        },
      });
      if (
        nativeDefaultTwoLayerProgram.kvRequirements.layers !== 2 ||
        nativeDefaultTwoLayerProgram.kvRequirements.kBufferByteLength !== u32(nativeDefaultTwoLayerKvRequirements, 20) ||
        nativeDefaultTwoLayerProgram.kvRequirements.vBufferByteLength !== u32(nativeDefaultTwoLayerKvRequirements, 24)
      ) {
        throw new Error("native-backed WebGPU tiny LLaMA default safetensors Program did not match native K/V requirements");
      }
      nativeDefaultTwoLayerSession = nativeDefaultTwoLayerProgram.bind();
      nativeDefaultTwoLayerSessionHandle = nativeDefaultTwoLayerSession.handle;
      if (nativeDefaultTwoLayerSession.nativeSession !== true || !hostRuntime.hasClaimed(nativeDefaultTwoLayerSessionHandle)) {
        throw new Error("native-backed WebGPU tiny LLaMA default safetensors Session was not registered for raw ABI execution");
      }
      if (
        nativeDefaultTwoLayerSession.modelHandle !== nativeDefaultTwoLayerCompatibleModel ||
        nativeDefaultTwoLayerSession.modelResources.length !== twoLayerModelResourceRequirements.length
      ) {
        throw new Error("native-backed WebGPU tiny LLaMA default safetensors bind did not preserve model resources");
      }
      await expectTinyLlamaModelResourceBytes(
        hostDevice,
        nativeDefaultTwoLayerSession.modelResources,
        twoLayerModelResourceBytes,
        "native-backed WebGPU tiny LLaMA default safetensors",
      );
      expectHostResourceTableRoles(
        nativeDefaultTwoLayerSession.table,
        ["llama.output", "llama.k.0", "llama.v.0", "llama.k.1", "llama.v.1"],
        "native-backed WebGPU tiny LLaMA default safetensors",
      );
      const nativeLayer0K = nativeDefaultTwoLayerSession.bindings.kvCache[0].k;
      const nativeLayer0V = nativeDefaultTwoLayerSession.bindings.kvCache[0].v;
      const nativeLayer1K = nativeDefaultTwoLayerSession.bindings.kvCache[1].k;
      const nativeLayer1V = nativeDefaultTwoLayerSession.bindings.kvCache[1].v;
      const nativeLayer0KElements = Math.floor(nativeLayer0K.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const nativeLayer0VElements = Math.floor(nativeLayer0V.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const nativeLayer1KElements = Math.floor(nativeLayer1K.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const nativeLayer1VElements = Math.floor(nativeLayer1V.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(nativeLayer0K, Array(nativeLayer0KElements).fill(-1510));
      hostDevice.writeFloat32(nativeLayer0V, Array(nativeLayer0VElements).fill(-1520));
      hostDevice.writeFloat32(nativeLayer1K, Array(nativeLayer1KElements).fill(-2510));
      hostDevice.writeFloat32(nativeLayer1V, Array(nativeLayer1VElements).fill(-2520));
      hostDevice.writeFloat32(nativeDefaultTwoLayerSession.bindings.output, Array(tinyLlamaVocabSize).fill(-6140));
      const nativeDefaultTokens = keep(alloc(4), 4);
      writeU32Array(nativeDefaultTokens, [2]);
      const nativeDefaultResult = tokenExecute(nativeDefaultTwoLayerSessionHandle, nativeDefaultTokens, 1, executeOutputLogits, 0, 0);
      expectStatus(nativeDefaultResult.status, ok, "native-backed WebGPU tiny LLaMA default safetensors raw execute");
      if (nativeDefaultResult.outputLen !== tinyLlamaVocabSize || sessionPosition(nativeDefaultTwoLayerSessionHandle) !== 1) {
        throw new Error("native-backed WebGPU tiny LLaMA default safetensors raw execute did not advance");
      }
      const nativeDefaultInspection = inspectSession(nativeDefaultTwoLayerSessionHandle);
      if (
        u32(nativeDefaultInspection, 0) !== tinyLlama2LayerKind ||
        u32(nativeDefaultInspection, 4) !== backendWebGpu ||
        u32(nativeDefaultInspection, 8) !== bufferStorageExternalResource ||
        u32(nativeDefaultInspection, 12) !== bufferStorageExternalResource ||
        u64(nativeDefaultInspection, 16) !== 1n ||
        u64(nativeDefaultInspection, 24) !== 4n ||
        u64(nativeDefaultInspection, 64) !== 5n ||
        u64(nativeDefaultInspection, 72) === 0n
      ) {
        throw new Error("native-backed WebGPU tiny LLaMA default safetensors raw inspect mismatch");
      }
      const expectedNativeLayer0Cache = {
        k: Array(nativeLayer0KElements).fill(-1510),
        v: Array(nativeLayer0VElements).fill(-1520),
      };
      const expectedNativeLayer1Cache = {
        k: Array(nativeLayer1KElements).fill(-2510),
        v: Array(nativeLayer1VElements).fill(-2520),
      };
      const expectedNativeLayer0 = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedNativeLayer0Cache, 2, 0, { layer: 0 });
      const expectedNativeLayer1 = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedNativeLayer1Cache, 2, 0, {
        inputHidden: expectedNativeLayer0.blockHidden,
        layer: 1,
      });
      expectClose(
        Array.from(await hostDevice.readFloat32(nativeDefaultTwoLayerSession.bindings.output, tinyLlamaVocabSize)),
        expectedNativeLayer1.logits,
      );
      expectClose(Array.from(await hostDevice.readFloat32(nativeLayer0K, expectedNativeLayer0.k.length, 0)), expectedNativeLayer0.k);
      expectClose(Array.from(await hostDevice.readFloat32(nativeLayer0V, expectedNativeLayer0.v.length, 0)), expectedNativeLayer0.v);
      expectClose(Array.from(await hostDevice.readFloat32(nativeLayer1K, expectedNativeLayer1.k.length, 0)), expectedNativeLayer1.k);
      expectClose(Array.from(await hostDevice.readFloat32(nativeLayer1V, expectedNativeLayer1.v.length, 0)), expectedNativeLayer1.v);
      const expectedNativeDefaultCommandCount = expectedLlamaBlockPipelineCommandCount(hostDevice, 2, 1, executeOutputLogits);
      const expectedNativeDefaultBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? expectedNativeDefaultCommandCount : 0;
      const expectedNativeDefaultFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : expectedNativeDefaultCommandCount;
      const expectedNativeDefaultUsesGpuStorageBuffers = canUseGpuStorageBuffers(hostDevice, 6);
      const nativeDefaultProfile = nativeDefaultTwoLayerSession.runtimeProfile();
      if (
        nativeDefaultProfile.executorCallCount !== 1 ||
        nativeDefaultProfile.executorOkCount !== 1 ||
        nativeDefaultProfile.executorBackendDispatchCount !== expectedNativeDefaultBackendDispatchCount ||
        nativeDefaultProfile.executorFallbackOpCount !== expectedNativeDefaultFallbackOpCount ||
        nativeDefaultProfile.executorCommandCount !== expectedNativeDefaultCommandCount ||
        nativeDefaultProfile.executorCommandOpCount !== expectedNativeDefaultCommandCount ||
        nativeDefaultProfile.lastStatus !== ok ||
        nativeDefaultProfile.lastResultOutputLength !== tinyLlamaVocabSize
      ) {
        throw new Error(`unexpected native-backed WebGPU tiny LLaMA default safetensors profile: ${JSON.stringify(nativeDefaultProfile)}`);
      }
      const nativeDefaultAbiProfile = sessionRuntimeProfile(nativeDefaultTwoLayerSessionHandle);
      if (
        u64(nativeDefaultAbiProfile, 0) !== 1n ||
        u64(nativeDefaultAbiProfile, 8) !== BigInt(expectedNativeDefaultBackendDispatchCount) ||
        u64(nativeDefaultAbiProfile, 16) !== BigInt(expectedNativeDefaultFallbackOpCount) ||
        u64(nativeDefaultAbiProfile, 24) !== BigInt(expectedNativeDefaultBackendDispatchCount) ||
        u64(nativeDefaultAbiProfile, 40) !== 1n ||
        u64(nativeDefaultAbiProfile, 48) !== 1n ||
        u64(nativeDefaultAbiProfile, 56) !== 0n ||
        u64(nativeDefaultAbiProfile, 96) !== BigInt(expectedNativeDefaultCommandCount) ||
        u64(nativeDefaultAbiProfile, 112) !== BigInt(expectedNativeDefaultCommandCount)
      ) {
        throw new Error("native-backed WebGPU tiny LLaMA default safetensors ABI profile mismatch");
      }
      if (
        nativeDefaultTwoLayerExecutorCalls.length !== 1 ||
        nativeDefaultTwoLayerExecutorCalls[0].stageCount !== 2 ||
        nativeDefaultTwoLayerExecutorCalls[0].commandCount !== expectedNativeDefaultCommandCount ||
        nativeDefaultTwoLayerExecutorCalls[0].deviceMode !== expectedLlamaExecutorDeviceMode(hostDevice) ||
        nativeDefaultTwoLayerExecutorCalls[0].maxStorageBuffersPerShaderStage !== hostDevice.maxStorageBuffersPerShaderStage() ||
        nativeDefaultTwoLayerExecutorCalls[0].requiredStorageBufferCount !== 6 ||
        nativeDefaultTwoLayerExecutorCalls[0].usesGpuStorageBuffers !== expectedNativeDefaultUsesGpuStorageBuffers ||
        nativeDefaultTwoLayerExecutorCalls[0].status !== ok
      ) {
        throw new Error("native-backed WebGPU tiny LLaMA default safetensors pipeline evidence mismatch");
      }
    } finally {
      if (nativeDefaultTwoLayerSessionHandle !== 0) exportsRef.zgml_session_free(nativeDefaultTwoLayerSessionHandle);
      if (nativeDefaultTwoLayerSession && nativeDefaultTwoLayerSession.ownedResources.length !== 0) {
        throw new Error("native-backed WebGPU tiny LLaMA default safetensors Session-owned resources were not released");
      }
      if (nativeDefaultTwoLayerProgramHandle !== 0) exportsRef.zgml_program_free(nativeDefaultTwoLayerProgramHandle);
      if (nativeDefaultTwoLayerCompatibleModel !== 0) exportsRef.zgml_model_free(nativeDefaultTwoLayerCompatibleModel);
      if (nativeDefaultTwoLayerModel !== 0) exportsRef.zgml_model_free(nativeDefaultTwoLayerModel);
    }
    const twoLayerPipelineSession = twoLayerPipelineProgram.bindHostResources();
    const twoLayerPipelineSessionHandle = twoLayerPipelineSession.handle;
    const twoLayerHostExports = hostRuntime.wrapExports(exportsRef);
    try {
      if (twoLayerPipelineSession.nativeSession !== false || !hostRuntime.hasClaimed(twoLayerPipelineSessionHandle)) {
        throw new Error("host-only WebGPU tiny LLaMA two-layer Session was not registered as host-only");
      }
      if (twoLayerPipelineSession.modelHandle !== compatibleModel) {
        throw new Error("host-only WebGPU tiny LLaMA two-layer Session did not inherit factory model handle");
      }
      if (twoLayerPipelineSession.modelResources.length !== twoLayerModelResourceRequirements.length) {
        throw new Error("host-only WebGPU tiny LLaMA two-layer byte-backed bind did not create model resources");
      }
      if (twoLayerPipelineSession.bindings.activation !== null) {
        throw new Error("host-only WebGPU tiny LLaMA minimal bind unexpectedly created a terminal activation resource");
      }
      const layer0K = twoLayerPipelineSession.bindings.kvCache[0].k;
      const layer0V = twoLayerPipelineSession.bindings.kvCache[0].v;
      const layer1K = twoLayerPipelineSession.bindings.kvCache[1].k;
      const layer1V = twoLayerPipelineSession.bindings.kvCache[1].v;
      const layer0KElements = Math.floor(layer0K.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const layer0VElements = Math.floor(layer0V.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const layer1KElements = Math.floor(layer1K.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const layer1VElements = Math.floor(layer1V.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(layer0K, Array(layer0KElements).fill(-1110));
      hostDevice.writeFloat32(layer0V, Array(layer0VElements).fill(-1120));
      hostDevice.writeFloat32(layer1K, Array(layer1KElements).fill(-2110));
      hostDevice.writeFloat32(layer1V, Array(layer1VElements).fill(-2120));
      hostDevice.writeFloat32(twoLayerPipelineSession.bindings.output, Array(tinyLlamaVocabSize).fill(-4140));
      expectHostResourceTableRoles(
        twoLayerPipelineSession.table,
        ["llama.output", "llama.k.0", "llama.v.0", "llama.k.1", "llama.v.1"],
        "host-only WebGPU tiny LLaMA two-layer minimal block pipeline",
      );
      const twoLayerTokens = keep(alloc(8), 8);
      writeU32Array(twoLayerTokens, [0, 3]);
      const twoLayerResult = tokenExecuteWith(twoLayerHostExports, twoLayerPipelineSessionHandle, twoLayerTokens, 2, executeOutputLogits, 0, 0);
      expectStatus(twoLayerResult.status, ok, "host-only WebGPU tiny LLaMA two-layer block pipeline execute");
      if (twoLayerResult.outputLen !== tinyLlamaVocabSize || sessionPositionWith(twoLayerHostExports, twoLayerPipelineSessionHandle) !== 2) {
        throw new Error("host-only WebGPU tiny LLaMA two-layer block pipeline did not emit bound logits and advance");
      }
      const twoLayerHostInspection = inspectSessionWith(twoLayerHostExports, twoLayerPipelineSessionHandle);
      if (
        u32(twoLayerHostInspection, 0) !== tinyLlama2LayerKind ||
        u32(twoLayerHostInspection, 4) !== backendWebGpu ||
        u32(twoLayerHostInspection, 8) !== bufferStorageExternalResource ||
        u32(twoLayerHostInspection, 12) !== bufferStorageExternalResource ||
        u64(twoLayerHostInspection, 16) !== 2n ||
        u64(twoLayerHostInspection, 24) !== 4n ||
        u64(twoLayerHostInspection, 40) !== 1n ||
        u64(twoLayerHostInspection, 48) !== 1n ||
        u64(twoLayerHostInspection, 56) !== 0n ||
        u64(twoLayerHostInspection, 64) !== BigInt(twoLayerPipelineSession.table.resourceCount) ||
        u64(twoLayerHostInspection, 72) !== twoLayerPipelineSession.table.hash
      ) {
        throw new Error("host-only WebGPU tiny LLaMA two-layer wrapped inspect mismatch");
      }
      const twoLayerSessionInspection = twoLayerPipelineSession.inspect();
      const expectedTwoLayerTableRoles = ["llama.output", "llama.k.0", "llama.v.0", "llama.k.1", "llama.v.1"];
      if (
        twoLayerSessionInspection.kind !== "llama-resource-session" ||
        twoLayerSessionInspection.executionCoverage !== "bounded-proof" ||
        twoLayerSessionInspection.executionCoverageReason !== "llama-family-bounded-proof-executor" ||
        twoLayerSessionInspection.executionMode !== "proof-executor" ||
        twoLayerSessionInspection.tokenExecutionSupported !== true ||
        twoLayerSessionInspection.boundedProofExecutionSupported !== true ||
        twoLayerSessionInspection.fullDefaultExecutionSupported !== false ||
        twoLayerSessionInspection.tokenExecutorKind !== "tiny-llama-block-pipeline" ||
        twoLayerSessionInspection.handle !== twoLayerPipelineSessionHandle ||
        twoLayerSessionInspection.nativeSession !== false ||
        twoLayerSessionInspection.modelBindingKind !== "compatible-model" ||
        twoLayerSessionInspection.modelHandle !== compatibleModel ||
        twoLayerSessionInspection.modelKind !== tinyLlama2LayerKind ||
        twoLayerSessionInspection.vocabSize !== tinyLlamaVocabSize ||
        twoLayerSessionInspection.contextLength !== 4 ||
        twoLayerSessionInspection.maxTokenWindow !== 4 ||
        twoLayerSessionInspection.position !== 2 ||
        twoLayerSessionInspection.outputByteLength !== tinyLlamaVocabSize * Float32Array.BYTES_PER_ELEMENT ||
        twoLayerSessionInspection.kvCacheLayers !== 2 ||
        twoLayerSessionInspection.requiredGpuStorageBufferCount !== 6 ||
        twoLayerSessionInspection.requireGpuDispatch !== false ||
        twoLayerSessionInspection.table.resourceCount !== expectedTwoLayerTableRoles.length ||
        twoLayerSessionInspection.table.hash !== twoLayerPipelineSession.table.hash ||
        twoLayerSessionInspection.table.roles.length !== expectedTwoLayerTableRoles.length ||
        twoLayerSessionInspection.table.roles.some((role, index) => role !== expectedTwoLayerTableRoles[index]) ||
        twoLayerSessionInspection.modelTable.resourceCount !== twoLayerModelResourceRequirements.length ||
        twoLayerSessionInspection.modelTable.hash !== twoLayerPipelineSession.modelTable.hash ||
        twoLayerSessionInspection.modelManifestHash !== twoLayerPipelineSession.modelManifest.hash ||
        twoLayerSessionInspection.modelResourceCount !== twoLayerModelResourceRequirements.length ||
        twoLayerSessionInspection.modelResourceRoles.length !== expectedTwoLayerModelRoles.length ||
        twoLayerSessionInspection.modelResourceRoles.some((role, index) => role !== expectedTwoLayerModelRoles[index]) ||
        twoLayerSessionInspection.program.kind !== "llama-resource-program" ||
        twoLayerSessionInspection.program.executionCoverage !== twoLayerProgramInspection.executionCoverage ||
        twoLayerSessionInspection.program.fullDefaultExecutionSupported !== false ||
        twoLayerSessionInspection.program.bindingRequirementHash !== twoLayerProgramInspection.bindingRequirementHash ||
        twoLayerSessionInspection.program.persistentRequirementCount !== expectedTwoLayerModelRoles.length + 4 ||
        twoLayerSessionInspection.program.stepInputRequirementCount !== 1 ||
        twoLayerSessionInspection.program.stepOutputRequirementCount !== 1 ||
        twoLayerSessionInspection.program.requiredModelResourceRoles.length !== expectedTwoLayerModelRoles.length ||
        twoLayerSessionInspection.program.requiredModelResourceRoles.some((role, index) => role !== expectedTwoLayerModelRoles[index])
      ) {
        throw new Error("host-only WebGPU tiny LLaMA two-layer JS Session inspection mismatch");
      }
      try {
        twoLayerSessionInspection.modelResourceRoles[0] = "llama.weight.mutated";
      } catch {}
      try {
        twoLayerSessionInspection.table.roles[0] = "llama.mutated";
      } catch {}
      if (
        twoLayerSessionInspection.modelResourceRoles[0] !== expectedTwoLayerModelRoles[0] ||
        twoLayerSessionInspection.table.roles[0] !== expectedTwoLayerTableRoles[0] ||
        twoLayerPipelineSession.modelResources[0].role !== expectedTwoLayerModelRoles[0]
      ) {
        throw new Error("host-only WebGPU tiny LLaMA JS Session inspection exposed mutable binding roles");
      }
      const expectedLayer0Cache = {
        k: Array(layer0KElements).fill(-1110),
        v: Array(layer0VElements).fill(-1120),
      };
      const expectedLayer1Cache = {
        k: Array(layer1KElements).fill(-2110),
        v: Array(layer1VElements).fill(-2120),
      };
      const expectedLayer0First = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedLayer0Cache, 0, 0, { layer: 0 });
      const expectedLayer0Last = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedLayer0Cache, 3, 1, { layer: 0 });
      const expectedLayer1First = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedLayer1Cache, 0, 0, {
        inputHidden: expectedLayer0First.blockHidden,
        layer: 1,
      });
      const expectedLayer1Last = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedLayer1Cache, 3, 1, {
        inputHidden: expectedLayer0Last.blockHidden,
        layer: 1,
      });
      expectClose(
        Array.from(await hostDevice.readFloat32(twoLayerPipelineSession.bindings.output, tinyLlamaVocabSize)),
        expectedLayer1Last.logits,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(layer0K, expectedLayer0First.k.length, 0)),
        expectedLayer0First.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(layer0V, expectedLayer0First.v.length, 0)),
        expectedLayer0First.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(layer0K, expectedLayer0Last.k.length, expectedLayer0Last.k.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedLayer0Last.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(layer0V, expectedLayer0Last.v.length, expectedLayer0Last.v.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedLayer0Last.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(layer1K, expectedLayer1First.k.length, 0)),
        expectedLayer1First.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(layer1V, expectedLayer1First.v.length, 0)),
        expectedLayer1First.v,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(layer1K, expectedLayer1Last.k.length, expectedLayer1Last.k.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedLayer1Last.k,
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(layer1V, expectedLayer1Last.v.length, expectedLayer1Last.v.length * Float32Array.BYTES_PER_ELEMENT)),
        expectedLayer1Last.v,
      );
      const expectedTwoLayerInitialCommandCount = expectedLlamaBlockPipelineCommandCount(hostDevice, 2, 2, executeOutputLogits);
      const expectedTwoLayerInitialBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? expectedTwoLayerInitialCommandCount : 0;
      const expectedTwoLayerInitialFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : expectedTwoLayerInitialCommandCount;
      const twoLayerProfile = twoLayerPipelineSession.runtimeProfile();
      if (
        twoLayerProfile.executorCallCount !== 1 ||
        twoLayerProfile.executorOkCount !== 1 ||
        twoLayerProfile.executorBackendDispatchCount !== expectedTwoLayerInitialBackendDispatchCount ||
        twoLayerProfile.executorFallbackOpCount !== expectedTwoLayerInitialFallbackOpCount ||
        twoLayerProfile.executorCommandCount !== expectedTwoLayerInitialCommandCount ||
        twoLayerProfile.executorCommandOpCount !== expectedTwoLayerInitialCommandCount ||
        twoLayerProfile.lastStatus !== ok ||
        twoLayerProfile.lastResultOutputLength !== tinyLlamaVocabSize
      ) {
        throw new Error(`unexpected host-only WebGPU tiny LLaMA two-layer pipeline profile: ${JSON.stringify(twoLayerProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("two-layer-pipeline", twoLayerProfile);
      const twoLayerAbiProfile = sessionRuntimeProfileWith(twoLayerHostExports, twoLayerPipelineSessionHandle);
      if (
        u64(twoLayerAbiProfile, 0) !== 1n ||
        u64(twoLayerAbiProfile, 8) !== BigInt(expectedTwoLayerInitialBackendDispatchCount) ||
        u64(twoLayerAbiProfile, 16) !== BigInt(expectedTwoLayerInitialFallbackOpCount) ||
        u64(twoLayerAbiProfile, 24) !== BigInt(expectedTwoLayerInitialBackendDispatchCount) ||
        u64(twoLayerAbiProfile, 40) !== 1n ||
        u64(twoLayerAbiProfile, 48) !== 1n ||
        u64(twoLayerAbiProfile, 56) !== 0n ||
        u64(twoLayerAbiProfile, 96) !== BigInt(expectedTwoLayerInitialCommandCount) ||
        u64(twoLayerAbiProfile, 112) !== BigInt(expectedTwoLayerInitialCommandCount)
      ) {
        throw new Error("unexpected host-only WebGPU tiny LLaMA two-layer pipeline ABI runtime profile");
      }
      if (
        twoLayerPipelineExecutorCalls.length !== 1 ||
        twoLayerPipelineExecutorCalls[0].stageCount !== 2 ||
        twoLayerPipelineExecutorCalls[0].workspaceCount !== 1 ||
        twoLayerPipelineExecutorCalls[0].layers.length !== 2 ||
        twoLayerPipelineExecutorCalls[0].layers[0] !== 0 ||
        twoLayerPipelineExecutorCalls[0].layers[1] !== 1 ||
        twoLayerPipelineExecutorCalls[0].commandCount !== expectedTwoLayerInitialCommandCount ||
        twoLayerPipelineExecutorCalls[0].status !== ok
      ) {
        throw new Error("host-only WebGPU tiny LLaMA two-layer block pipeline evidence mismatch");
      }
      const twoLayerResetCallStart = twoLayerPipelineExecutorCalls.length;
      expectStatus(twoLayerHostExports.zgml_session_reset(twoLayerPipelineSessionHandle), ok, "host-only WebGPU tiny LLaMA two-layer wrapped reset");
      if (sessionPositionWith(twoLayerHostExports, twoLayerPipelineSessionHandle) !== 0) {
        throw new Error("host-only WebGPU tiny LLaMA two-layer reset did not rewind position");
      }
      const twoLayerResetInspection = inspectSessionWith(twoLayerHostExports, twoLayerPipelineSessionHandle);
      if (
        u64(twoLayerResetInspection, 16) !== 0n ||
        u64(twoLayerResetInspection, 24) !== 4n ||
        u64(twoLayerResetInspection, 72) !== twoLayerPipelineSession.table.hash
      ) {
        throw new Error("host-only WebGPU tiny LLaMA two-layer wrapped inspect did not reflect reset");
      }
      const twoLayerReplayTokens = keep(alloc(4), 4);
      writeU32Array(twoLayerReplayTokens, [1]);
      const twoLayerReplay = tokenExecuteWith(twoLayerHostExports, twoLayerPipelineSessionHandle, twoLayerReplayTokens, 1, executeOutputLogits, 0, 0);
      expectStatus(twoLayerReplay.status, ok, "host-only WebGPU tiny LLaMA two-layer replay after reset");
      if (twoLayerReplay.outputLen !== tinyLlamaVocabSize || sessionPositionWith(twoLayerHostExports, twoLayerPipelineSessionHandle) !== 1) {
        throw new Error("host-only WebGPU tiny LLaMA two-layer replay after reset did not restart at position zero");
      }
      if (u64(inspectSessionWith(twoLayerHostExports, twoLayerPipelineSessionHandle), 16) !== 1n) {
        throw new Error("host-only WebGPU tiny LLaMA two-layer wrapped inspect did not reflect replay position");
      }
      const expectedResetLayer0Cache = {
        k: Array(layer0KElements).fill(-1110),
        v: Array(layer0VElements).fill(-1120),
      };
      const expectedResetLayer1Cache = {
        k: Array(layer1KElements).fill(-2110),
        v: Array(layer1VElements).fill(-2120),
      };
      const expectedResetLayer0 = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedResetLayer0Cache, 1, 0, { layer: 0 });
      const expectedResetLayer1 = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedResetLayer1Cache, 1, 0, {
        inputHidden: expectedResetLayer0.blockHidden,
        layer: 1,
      });
      expectClose(
        Array.from(await hostDevice.readFloat32(twoLayerPipelineSession.bindings.output, tinyLlamaVocabSize)),
        expectedResetLayer1.logits,
      );
      expectClose(Array.from(await hostDevice.readFloat32(layer0K, expectedResetLayer0.k.length, 0)), expectedResetLayer0.k);
      expectClose(Array.from(await hostDevice.readFloat32(layer0V, expectedResetLayer0.v.length, 0)), expectedResetLayer0.v);
      expectClose(Array.from(await hostDevice.readFloat32(layer1K, expectedResetLayer1.k.length, 0)), expectedResetLayer1.k);
      expectClose(Array.from(await hostDevice.readFloat32(layer1V, expectedResetLayer1.v.length, 0)), expectedResetLayer1.v);
      const twoLayerResetCalls = twoLayerPipelineExecutorCalls.slice(twoLayerResetCallStart);
      if (
        twoLayerResetCalls.length !== 1 ||
        twoLayerResetCalls[0].stageCount !== 2 ||
        twoLayerResetCalls[0].outputPolicy !== executeOutputLogits ||
        twoLayerResetCalls[0].commandCount !== expectedLlamaBlockPipelineCommandCount(hostDevice, 2, 1, executeOutputLogits) ||
        twoLayerResetCalls[0].status !== ok
      ) {
        throw new Error("host-only WebGPU tiny LLaMA two-layer replay-after-reset evidence mismatch");
      }
      const expectedTwoLayerCommandCount = expectedLlamaBlockPipelineCommandCount(hostDevice, 2, 1, executeOutputLogits) * 2;
      const expectedTwoLayerBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? expectedTwoLayerCommandCount : 0;
      const expectedTwoLayerFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : expectedTwoLayerCommandCount;
      const ergonomicCallStart = twoLayerPipelineExecutorCalls.length;
      const ergonomicSession = twoLayerPipelineProgram.bindHostResources();
      const ergonomicHandle = ergonomicSession.handle;
      try {
        const ergonomicLayer0K = ergonomicSession.bindings.kvCache[0].k;
        const ergonomicLayer0V = ergonomicSession.bindings.kvCache[0].v;
        const ergonomicLayer1K = ergonomicSession.bindings.kvCache[1].k;
        const ergonomicLayer1V = ergonomicSession.bindings.kvCache[1].v;
        const ergonomicLayer0KElements = Math.floor(ergonomicLayer0K.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const ergonomicLayer0VElements = Math.floor(ergonomicLayer0V.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const ergonomicLayer1KElements = Math.floor(ergonomicLayer1K.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const ergonomicLayer1VElements = Math.floor(ergonomicLayer1V.byteLength / Float32Array.BYTES_PER_ELEMENT);
        hostDevice.writeFloat32(ergonomicLayer0K, Array(ergonomicLayer0KElements).fill(-1310));
        hostDevice.writeFloat32(ergonomicLayer0V, Array(ergonomicLayer0VElements).fill(-1320));
        hostDevice.writeFloat32(ergonomicLayer1K, Array(ergonomicLayer1KElements).fill(-2310));
        hostDevice.writeFloat32(ergonomicLayer1V, Array(ergonomicLayer1VElements).fill(-2320));
        hostDevice.writeFloat32(ergonomicSession.bindings.output, Array(tinyLlamaVocabSize).fill(-5140));
        const ergonomicAdvance = await ergonomicSession.advanceTokens([0]);
        expectStatus(ergonomicAdvance.status, ok, "host-only WebGPU tiny LLaMA ergonomic advance");
        if (
          ergonomicAdvance.outputLen !== 0 ||
          ergonomicAdvance.position !== 1 ||
          sessionPositionWith(twoLayerHostExports, ergonomicHandle) !== 1
        ) {
          throw new Error("host-only WebGPU tiny LLaMA ergonomic advance did not update position without logits");
        }
        expectClose(
          Array.from(await hostDevice.readFloat32(ergonomicSession.bindings.output, tinyLlamaVocabSize)),
          Array(tinyLlamaVocabSize).fill(-5140),
        );
        const ergonomicStep = await ergonomicSession.step(3);
        expectStatus(ergonomicStep.status, ok, "host-only WebGPU tiny LLaMA ergonomic step");
        if (
          ergonomicStep.outputLen !== tinyLlamaVocabSize ||
          ergonomicStep.position !== 2 ||
          sessionPositionWith(twoLayerHostExports, ergonomicHandle) !== 2 ||
          !(ergonomicStep.logits instanceof Float32Array)
        ) {
          throw new Error("host-only WebGPU tiny LLaMA ergonomic step did not return bound logits and advance");
        }
        const expectedErgonomicLayer0Cache = {
          k: Array(ergonomicLayer0KElements).fill(-1310),
          v: Array(ergonomicLayer0VElements).fill(-1320),
        };
        const expectedErgonomicLayer1Cache = {
          k: Array(ergonomicLayer1KElements).fill(-2310),
          v: Array(ergonomicLayer1VElements).fill(-2320),
        };
        const expectedErgonomicLayer0First = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedErgonomicLayer0Cache, 0, 0, { layer: 0 });
        const expectedErgonomicLayer0Last = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedErgonomicLayer0Cache, 3, 1, { layer: 0 });
        const expectedErgonomicLayer1First = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedErgonomicLayer1Cache, 0, 0, {
          inputHidden: expectedErgonomicLayer0First.blockHidden,
          layer: 1,
        });
        const expectedErgonomicLayer1Last = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedErgonomicLayer1Cache, 3, 1, {
          inputHidden: expectedErgonomicLayer0Last.blockHidden,
          layer: 1,
        });
        expectClose(Array.from(ergonomicStep.logits), expectedErgonomicLayer1Last.logits);
        expectClose(
          Array.from(await hostDevice.readFloat32(ergonomicSession.bindings.output, tinyLlamaVocabSize)),
          expectedErgonomicLayer1Last.logits,
        );
        expectClose(
          Array.from(await hostDevice.readFloat32(ergonomicLayer0K, expectedErgonomicLayer0First.k.length, 0)),
          expectedErgonomicLayer0First.k,
        );
        expectClose(
          Array.from(await hostDevice.readFloat32(ergonomicLayer0V, expectedErgonomicLayer0First.v.length, 0)),
          expectedErgonomicLayer0First.v,
        );
        expectClose(
          Array.from(await hostDevice.readFloat32(ergonomicLayer0K, expectedErgonomicLayer0Last.k.length, expectedErgonomicLayer0Last.k.length * Float32Array.BYTES_PER_ELEMENT)),
          expectedErgonomicLayer0Last.k,
        );
        expectClose(
          Array.from(await hostDevice.readFloat32(ergonomicLayer0V, expectedErgonomicLayer0Last.v.length, expectedErgonomicLayer0Last.v.length * Float32Array.BYTES_PER_ELEMENT)),
          expectedErgonomicLayer0Last.v,
        );
        expectClose(
          Array.from(await hostDevice.readFloat32(ergonomicLayer1K, expectedErgonomicLayer1First.k.length, 0)),
          expectedErgonomicLayer1First.k,
        );
        expectClose(
          Array.from(await hostDevice.readFloat32(ergonomicLayer1V, expectedErgonomicLayer1First.v.length, 0)),
          expectedErgonomicLayer1First.v,
        );
        expectClose(
          Array.from(await hostDevice.readFloat32(ergonomicLayer1K, expectedErgonomicLayer1Last.k.length, expectedErgonomicLayer1Last.k.length * Float32Array.BYTES_PER_ELEMENT)),
          expectedErgonomicLayer1Last.k,
        );
        expectClose(
          Array.from(await hostDevice.readFloat32(ergonomicLayer1V, expectedErgonomicLayer1Last.v.length, expectedErgonomicLayer1Last.v.length * Float32Array.BYTES_PER_ELEMENT)),
          expectedErgonomicLayer1Last.v,
        );
        const ergonomicProfile = ergonomicSession.runtimeProfile();
        if (
          ergonomicProfile.callCount !== 2 ||
          ergonomicProfile.decodedCallCount !== 2 ||
          ergonomicProfile.noOutputCallCount !== 1 ||
          ergonomicProfile.logitsOutputCallCount !== 1 ||
          ergonomicProfile.boundOutputCallCount !== 1 ||
          ergonomicProfile.executorCallCount !== 2 ||
          ergonomicProfile.executorOkCount !== 2 ||
          ergonomicProfile.executorBackendDispatchCount !== expectedTwoLayerBackendDispatchCount ||
          ergonomicProfile.executorFallbackOpCount !== expectedTwoLayerFallbackOpCount ||
          ergonomicProfile.executorCommandCount !== 4 ||
          ergonomicProfile.executorCommandOpCount !== 4 ||
          ergonomicProfile.outputReadCount !== 1 ||
          ergonomicProfile.syncCount !== 1 ||
          ergonomicProfile.lastStatus !== ok ||
          ergonomicProfile.lastResultOutputLength !== tinyLlamaVocabSize
        ) {
          throw new Error(`unexpected host-only WebGPU tiny LLaMA ergonomic profile: ${JSON.stringify(ergonomicProfile)}`);
        }
        recordBrowserLlamaProfileEvidence("ergonomic-two-layer", ergonomicProfile);
        const ergonomicAbiProfile = sessionRuntimeProfileWith(twoLayerHostExports, ergonomicHandle);
        if (
          u64(ergonomicAbiProfile, 0) !== 2n ||
          u64(ergonomicAbiProfile, 8) !== BigInt(expectedTwoLayerBackendDispatchCount) ||
          u64(ergonomicAbiProfile, 16) !== BigInt(expectedTwoLayerFallbackOpCount) ||
          u64(ergonomicAbiProfile, 24) !== BigInt(expectedTwoLayerBackendDispatchCount) ||
          u64(ergonomicAbiProfile, 32) !== 1n ||
          u64(ergonomicAbiProfile, 40) !== 2n ||
          u64(ergonomicAbiProfile, 48) !== 2n ||
          u64(ergonomicAbiProfile, 56) !== 0n ||
          u64(ergonomicAbiProfile, 96) !== 4n ||
          u64(ergonomicAbiProfile, 112) !== 4n
        ) {
          throw new Error("unexpected host-only WebGPU tiny LLaMA ergonomic ABI runtime profile");
        }
        const ergonomicCalls = twoLayerPipelineExecutorCalls.slice(ergonomicCallStart);
        if (
          ergonomicCalls.length !== 2 ||
          ergonomicCalls[0].outputPolicy !== 0 ||
          ergonomicCalls[0].tokenCount !== 1 ||
          ergonomicCalls[0].commandCount !== 2 ||
          ergonomicCalls[1].outputPolicy !== 1 ||
          ergonomicCalls[1].tokenCount !== 1 ||
          ergonomicCalls[1].commandCount !== 2 ||
          ergonomicCalls[1].outputLength !== tinyLlamaVocabSize
        ) {
          throw new Error("host-only WebGPU tiny LLaMA ergonomic executor evidence mismatch");
        }
      } finally {
        expectStatus(twoLayerHostExports.zgml_session_free(ergonomicHandle), ok, "host-only WebGPU tiny LLaMA ergonomic Session free");
      }
      const greedyCallStart = twoLayerPipelineExecutorCalls.length;
      const greedySession = twoLayerPipelineProgram.bindHostResources();
      const greedyHandle = greedySession.handle;
      try {
        const greedyLayer0K = greedySession.bindings.kvCache[0].k;
        const greedyLayer0V = greedySession.bindings.kvCache[0].v;
        const greedyLayer1K = greedySession.bindings.kvCache[1].k;
        const greedyLayer1V = greedySession.bindings.kvCache[1].v;
        const greedyLayer0KElements = Math.floor(greedyLayer0K.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const greedyLayer0VElements = Math.floor(greedyLayer0V.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const greedyLayer1KElements = Math.floor(greedyLayer1K.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const greedyLayer1VElements = Math.floor(greedyLayer1V.byteLength / Float32Array.BYTES_PER_ELEMENT);
        hostDevice.writeFloat32(greedyLayer0K, Array(greedyLayer0KElements).fill(-1510));
        hostDevice.writeFloat32(greedyLayer0V, Array(greedyLayer0VElements).fill(-1520));
        hostDevice.writeFloat32(greedyLayer1K, Array(greedyLayer1KElements).fill(-2510));
        hostDevice.writeFloat32(greedyLayer1V, Array(greedyLayer1VElements).fill(-2520));
        hostDevice.writeFloat32(greedySession.bindings.output, Array(tinyLlamaVocabSize).fill(-6140));
        const expectedGreedyLayer0Cache = {
          k: Array(greedyLayer0KElements).fill(-1510),
          v: Array(greedyLayer0VElements).fill(-1520),
        };
        const expectedGreedyLayer1Cache = {
          k: Array(greedyLayer1KElements).fill(-2510),
          v: Array(greedyLayer1VElements).fill(-2520),
        };
        const expectedGreedyLayer0First = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedGreedyLayer0Cache, 0, 0, { layer: 0 });
        const expectedGreedyLayer1First = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedGreedyLayer1Cache, 0, 0, {
          inputHidden: expectedGreedyLayer0First.blockHidden,
          layer: 1,
        });
        const expectedGreedyFirst = argmaxOf(expectedGreedyLayer1First.logits);
        const expectedGreedyLayer0Second = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedGreedyLayer0Cache, expectedGreedyFirst.token, 1, { layer: 0 });
        const expectedGreedyLayer1Second = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedGreedyLayer1Cache, expectedGreedyFirst.token, 1, {
          inputHidden: expectedGreedyLayer0Second.blockHidden,
          layer: 1,
        });
        const expectedGreedySecond = argmaxOf(expectedGreedyLayer1Second.logits);
        const greedy = await greedySession.generateTokensArgmax([0], 2, { includeLogits: true });
        expectStatus(greedy.status, ok, "host-only WebGPU tiny LLaMA ergonomic greedy generation");
        if (
          !(greedy.tokens instanceof Uint32Array) ||
          greedy.tokensGenerated !== 2 ||
          greedy.tokens[0] !== expectedGreedyFirst.token ||
          greedy.tokens[1] !== expectedGreedySecond.token ||
          greedy.lastToken !== expectedGreedySecond.token ||
          Math.abs(greedy.lastLogit - expectedGreedySecond.logit) > 1e-5 ||
          greedy.position !== 2 ||
          sessionPositionWith(twoLayerHostExports, greedyHandle) !== 2 ||
          !(greedy.logits instanceof Float32Array)
        ) {
          throw new Error("host-only WebGPU tiny LLaMA ergonomic greedy generation returned unexpected tokens");
        }
        expectClose(Array.from(greedy.logits), expectedGreedyLayer1Second.logits);
        expectClose(
          Array.from(await hostDevice.readFloat32(greedySession.bindings.output, tinyLlamaVocabSize)),
          expectedGreedyLayer1Second.logits,
        );
        expectClose(
          Array.from(await hostDevice.readFloat32(greedyLayer0K, expectedGreedyLayer0First.k.length, 0)),
          expectedGreedyLayer0First.k,
        );
        expectClose(
          Array.from(await hostDevice.readFloat32(greedyLayer0K, expectedGreedyLayer0Second.k.length, expectedGreedyLayer0Second.k.length * Float32Array.BYTES_PER_ELEMENT)),
          expectedGreedyLayer0Second.k,
        );
        expectClose(
          Array.from(await hostDevice.readFloat32(greedyLayer1V, expectedGreedyLayer1First.v.length, 0)),
          expectedGreedyLayer1First.v,
        );
        expectClose(
          Array.from(await hostDevice.readFloat32(greedyLayer1V, expectedGreedyLayer1Second.v.length, expectedGreedyLayer1Second.v.length * Float32Array.BYTES_PER_ELEMENT)),
          expectedGreedyLayer1Second.v,
        );
        const greedyProfile = greedySession.runtimeProfile();
        if (
          greedyProfile.callCount !== 2 ||
          greedyProfile.decodedCallCount !== 2 ||
          greedyProfile.logitsOutputCallCount !== 2 ||
          greedyProfile.boundOutputCallCount !== 2 ||
          greedyProfile.executorCallCount !== 2 ||
          greedyProfile.executorOkCount !== 2 ||
          greedyProfile.executorBackendDispatchCount !== expectedTwoLayerBackendDispatchCount ||
          greedyProfile.executorFallbackOpCount !== expectedTwoLayerFallbackOpCount ||
          greedyProfile.executorCommandCount !== 4 ||
          greedyProfile.executorCommandOpCount !== 4 ||
          greedyProfile.outputReadCount !== 2 ||
          greedyProfile.syncCount !== 2 ||
          greedyProfile.lastStatus !== ok ||
          greedyProfile.lastResultOutputLength !== tinyLlamaVocabSize
        ) {
          throw new Error(`unexpected host-only WebGPU tiny LLaMA ergonomic greedy profile: ${JSON.stringify(greedyProfile)}`);
        }
        recordBrowserLlamaProfileEvidence("ergonomic-greedy", greedyProfile);
        const greedyAbiProfile = sessionRuntimeProfileWith(twoLayerHostExports, greedyHandle);
        if (
          u64(greedyAbiProfile, 0) !== 2n ||
          u64(greedyAbiProfile, 8) !== BigInt(expectedTwoLayerBackendDispatchCount) ||
          u64(greedyAbiProfile, 16) !== BigInt(expectedTwoLayerFallbackOpCount) ||
          u64(greedyAbiProfile, 24) !== BigInt(expectedTwoLayerBackendDispatchCount) ||
          u64(greedyAbiProfile, 32) !== 2n ||
          u64(greedyAbiProfile, 40) !== 2n ||
          u64(greedyAbiProfile, 48) !== 2n ||
          u64(greedyAbiProfile, 56) !== 0n ||
          u64(greedyAbiProfile, 96) !== 4n ||
          u64(greedyAbiProfile, 112) !== 4n
        ) {
          throw new Error("unexpected host-only WebGPU tiny LLaMA ergonomic greedy ABI runtime profile");
        }
        const greedyCalls = twoLayerPipelineExecutorCalls.slice(greedyCallStart);
        if (
          greedyCalls.length !== 2 ||
          greedyCalls[0].outputPolicy !== 1 ||
          greedyCalls[0].tokenCount !== 1 ||
          greedyCalls[0].commandCount !== 2 ||
          greedyCalls[1].outputPolicy !== 1 ||
          greedyCalls[1].tokenCount !== 1 ||
          greedyCalls[1].commandCount !== 2 ||
          greedyCalls[1].outputLength !== tinyLlamaVocabSize
        ) {
          throw new Error("host-only WebGPU tiny LLaMA ergonomic greedy executor evidence mismatch");
        }
      } finally {
        expectStatus(twoLayerHostExports.zgml_session_free(greedyHandle), ok, "host-only WebGPU tiny LLaMA ergonomic greedy Session free");
      }
      const deviceGreedyCallStart = twoLayerPipelineExecutorCalls.length;
      const deviceGreedySession = twoLayerPipelineProgram.bindHostResources();
      const deviceGreedyHandle = deviceGreedySession.handle;
      try {
        const deviceGreedyLayer0K = deviceGreedySession.bindings.kvCache[0].k;
        const deviceGreedyLayer0V = deviceGreedySession.bindings.kvCache[0].v;
        const deviceGreedyLayer1K = deviceGreedySession.bindings.kvCache[1].k;
        const deviceGreedyLayer1V = deviceGreedySession.bindings.kvCache[1].v;
        const deviceGreedyLayer0KElements = Math.floor(deviceGreedyLayer0K.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const deviceGreedyLayer0VElements = Math.floor(deviceGreedyLayer0V.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const deviceGreedyLayer1KElements = Math.floor(deviceGreedyLayer1K.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const deviceGreedyLayer1VElements = Math.floor(deviceGreedyLayer1V.byteLength / Float32Array.BYTES_PER_ELEMENT);
        hostDevice.writeFloat32(deviceGreedyLayer0K, Array(deviceGreedyLayer0KElements).fill(-1610));
        hostDevice.writeFloat32(deviceGreedyLayer0V, Array(deviceGreedyLayer0VElements).fill(-1620));
        hostDevice.writeFloat32(deviceGreedyLayer1K, Array(deviceGreedyLayer1KElements).fill(-2610));
        hostDevice.writeFloat32(deviceGreedyLayer1V, Array(deviceGreedyLayer1VElements).fill(-2620));
        hostDevice.writeFloat32(deviceGreedySession.bindings.output, Array(tinyLlamaVocabSize).fill(-6240));
        const expectedDeviceGreedyLayer0Cache = {
          k: Array(deviceGreedyLayer0KElements).fill(-1610),
          v: Array(deviceGreedyLayer0VElements).fill(-1620),
        };
        const expectedDeviceGreedyLayer1Cache = {
          k: Array(deviceGreedyLayer1KElements).fill(-2610),
          v: Array(deviceGreedyLayer1VElements).fill(-2620),
        };
        const expectedDeviceGreedyLayer0First = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedDeviceGreedyLayer0Cache, 0, 0, { layer: 0 });
        const expectedDeviceGreedyLayer1First = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedDeviceGreedyLayer1Cache, 0, 0, {
          inputHidden: expectedDeviceGreedyLayer0First.blockHidden,
          layer: 1,
        });
        const expectedDeviceGreedyFirst = argmaxOf(expectedDeviceGreedyLayer1First.logits);
        const expectedDeviceGreedyLayer0Second = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedDeviceGreedyLayer0Cache, expectedDeviceGreedyFirst.token, 1, { layer: 0 });
        const expectedDeviceGreedyLayer1Second = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedDeviceGreedyLayer1Cache, expectedDeviceGreedyFirst.token, 1, {
          inputHidden: expectedDeviceGreedyLayer0Second.blockHidden,
          layer: 1,
        });
        const expectedDeviceGreedySecond = argmaxOf(expectedDeviceGreedyLayer1Second.logits);
        const deviceGreedy = await deviceGreedySession.generateTokensArgmaxDevice([0], 2);
        expectStatus(deviceGreedy.status, ok, "host-only WebGPU tiny LLaMA device-argmax greedy generation");
        if (
          !(deviceGreedy.tokens instanceof Uint32Array) ||
          deviceGreedy.tokensGenerated !== 2 ||
          deviceGreedy.tokens[0] !== expectedDeviceGreedyFirst.token ||
          deviceGreedy.tokens[1] !== expectedDeviceGreedySecond.token ||
          deviceGreedy.lastToken !== expectedDeviceGreedySecond.token ||
          Math.abs(deviceGreedy.lastLogit - expectedDeviceGreedySecond.logit) > 1e-5 ||
          deviceGreedy.position !== 2 ||
          deviceGreedy.logits !== undefined ||
          sessionPositionWith(twoLayerHostExports, deviceGreedyHandle) !== 2
        ) {
          throw new Error("host-only WebGPU tiny LLaMA device-argmax greedy generation returned unexpected tokens");
        }
        expectClose(
          Array.from(await hostDevice.readFloat32(deviceGreedySession.bindings.output, tinyLlamaVocabSize)),
          expectedDeviceGreedyLayer1Second.logits,
        );
        const deviceGreedyUsesGpu = canUseGpuStorageBuffers(hostDevice, 6);
        const deviceGreedyProfile = deviceGreedySession.runtimeProfile();
        if (
          deviceGreedyProfile.callCount !== 2 ||
          deviceGreedyProfile.decodedCallCount !== 2 ||
          deviceGreedyProfile.logitsOutputCallCount !== 2 ||
          deviceGreedyProfile.boundOutputCallCount !== 2 ||
          deviceGreedyProfile.executorCallCount !== 2 ||
          deviceGreedyProfile.executorOkCount !== 2 ||
          deviceGreedyProfile.executorBackendDispatchCount !== expectedTwoLayerBackendDispatchCount ||
          deviceGreedyProfile.executorFallbackOpCount !== expectedTwoLayerFallbackOpCount ||
          deviceGreedyProfile.executorCommandCount !== 4 ||
          deviceGreedyProfile.executorCommandOpCount !== 4 ||
          deviceGreedyProfile.outputReadCount !== (deviceGreedyUsesGpu ? 0 : 2) ||
          deviceGreedyProfile.syncCount !== (deviceGreedyUsesGpu ? 0 : 2) ||
          deviceGreedyProfile.selectionCallCount !== 2 ||
          deviceGreedyProfile.selectionBackendDispatchCount !== (deviceGreedyUsesGpu ? 2 : 0) ||
          deviceGreedyProfile.selectionFallbackOpCount !== (deviceGreedyUsesGpu ? 0 : 2) ||
          deviceGreedyProfile.selectionResultReadCount !== (deviceGreedyUsesGpu ? 2 : 0) ||
          deviceGreedyProfile.selectionSyncCount !== (deviceGreedyUsesGpu ? 2 : 0) ||
          deviceGreedyProfile.lastStatus !== ok ||
          deviceGreedyProfile.lastResultOutputLength !== tinyLlamaVocabSize
        ) {
          throw new Error(`unexpected host-only WebGPU tiny LLaMA device-argmax greedy profile: ${JSON.stringify(deviceGreedyProfile)}`);
        }
        recordBrowserLlamaProfileEvidence("device-greedy", deviceGreedyProfile);
        const deviceGreedyCalls = twoLayerPipelineExecutorCalls.slice(deviceGreedyCallStart);
        if (
          deviceGreedyCalls.length !== 2 ||
          deviceGreedyCalls[0].outputPolicy !== 1 ||
          deviceGreedyCalls[0].tokenCount !== 1 ||
          deviceGreedyCalls[0].commandCount !== 2 ||
          deviceGreedyCalls[1].outputPolicy !== 1 ||
          deviceGreedyCalls[1].tokenCount !== 1 ||
          deviceGreedyCalls[1].commandCount !== 2 ||
          deviceGreedyCalls[1].outputLength !== tinyLlamaVocabSize
        ) {
          throw new Error("host-only WebGPU tiny LLaMA device-argmax greedy executor evidence mismatch");
        }
      } finally {
        expectStatus(twoLayerHostExports.zgml_session_free(deviceGreedyHandle), ok, "host-only WebGPU tiny LLaMA device-argmax greedy Session free");
      }
      const greedyOverContextSession = twoLayerPipelineProgram.bindHostResources();
      const greedyOverContextHandle = greedyOverContextSession.handle;
      try {
        const greedyOverContextOutput = new Uint32Array([777, 777, 777, 777]);
        const greedyOverContextSentinel = Array(tinyLlamaVocabSize).fill(-7141);
        const greedyOverContextK = greedyOverContextSession.bindings.kvCache[0].k;
        const greedyOverContextV = greedyOverContextSession.bindings.kvCache[0].v;
        const greedyOverContextKElements = Math.floor(greedyOverContextK.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const greedyOverContextVElements = Math.floor(greedyOverContextV.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const greedyOverContextKSentinel = Array(greedyOverContextKElements).fill(-1810);
        const greedyOverContextVSentinel = Array(greedyOverContextVElements).fill(-1820);
        hostDevice.writeFloat32(greedyOverContextSession.bindings.output, greedyOverContextSentinel);
        hostDevice.writeFloat32(greedyOverContextK, greedyOverContextKSentinel);
        hostDevice.writeFloat32(greedyOverContextV, greedyOverContextVSentinel);
        const greedyOverContext = await greedyOverContextSession.generateTokensArgmaxInto([0, 1], greedyOverContextOutput, { includeLogits: true });
        expectStatus(greedyOverContext.status, shapeMismatch, "host-only WebGPU tiny LLaMA ergonomic greedy over-context generation");
        if (
          greedyOverContext.tokensGenerated !== 0 ||
          greedyOverContext.lastToken !== 0 ||
          greedyOverContext.lastLogit !== 0 ||
          greedyOverContext.position !== 0 ||
          greedyOverContext.logits !== undefined ||
          sessionPositionWith(twoLayerHostExports, greedyOverContextHandle) !== 0 ||
          Array.from(greedyOverContextOutput).some((token) => token !== 777)
        ) {
          throw new Error("host-only WebGPU tiny LLaMA ergonomic greedy over-context generation mutated state");
        }
        expectClose(Array.from(await hostDevice.readFloat32(greedyOverContextSession.bindings.output, tinyLlamaVocabSize)), greedyOverContextSentinel);
        expectClose(Array.from(await hostDevice.readFloat32(greedyOverContextK, greedyOverContextKElements)), greedyOverContextKSentinel);
        expectClose(Array.from(await hostDevice.readFloat32(greedyOverContextV, greedyOverContextVElements)), greedyOverContextVSentinel);
        const greedyOverContextProfile = greedyOverContextSession.runtimeProfile();
        if (
          greedyOverContextProfile.callCount !== 0 ||
          greedyOverContextProfile.executorCallCount !== 0 ||
          greedyOverContextProfile.executorBackendDispatchCount !== 0 ||
          greedyOverContextProfile.executorFallbackOpCount !== 0 ||
          greedyOverContextProfile.outputReadCount !== 0 ||
          greedyOverContextProfile.syncCount !== 0 ||
          greedyOverContextProfile.lastStatus !== null
        ) {
          throw new Error(`unexpected host-only WebGPU tiny LLaMA greedy over-context generation profile: ${JSON.stringify(greedyOverContextProfile)}`);
        }
      } finally {
        expectStatus(twoLayerHostExports.zgml_session_free(greedyOverContextHandle), ok, "host-only WebGPU tiny LLaMA ergonomic greedy over-context Session free");
      }
      const sampledCallStart = twoLayerPipelineExecutorCalls.length;
      const sampledSession = twoLayerPipelineProgram.bindHostResources();
      const sampledHandle = sampledSession.handle;
      try {
        const directSampleExpected = sampleTopKOf(new Float32Array([-1.0, 3.0, 2.5]), {
          seed: 17,
          temperature: 0.75,
          topK: 2,
        });
        const directSample = await sampledSession.sample({
          logits: new Float32Array([-1.0, 3.0, 2.5]),
          seed: 17,
          temperature: 0.75,
          topK: 2,
        });
        expectStatus(directSample.status, ok, "host-only WebGPU tiny LLaMA ergonomic direct sample");
        if (directSample.token !== directSampleExpected.token || directSample.logit !== directSampleExpected.logit) {
          throw new Error("host-only WebGPU tiny LLaMA ergonomic direct sample mismatch");
        }
        const sampledLayer0K = sampledSession.bindings.kvCache[0].k;
        const sampledLayer0V = sampledSession.bindings.kvCache[0].v;
        const sampledLayer1K = sampledSession.bindings.kvCache[1].k;
        const sampledLayer1V = sampledSession.bindings.kvCache[1].v;
        const sampledLayer0KElements = Math.floor(sampledLayer0K.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const sampledLayer0VElements = Math.floor(sampledLayer0V.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const sampledLayer1KElements = Math.floor(sampledLayer1K.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const sampledLayer1VElements = Math.floor(sampledLayer1V.byteLength / Float32Array.BYTES_PER_ELEMENT);
        hostDevice.writeFloat32(sampledLayer0K, Array(sampledLayer0KElements).fill(-1710));
        hostDevice.writeFloat32(sampledLayer0V, Array(sampledLayer0VElements).fill(-1720));
        hostDevice.writeFloat32(sampledLayer1K, Array(sampledLayer1KElements).fill(-2710));
        hostDevice.writeFloat32(sampledLayer1V, Array(sampledLayer1VElements).fill(-2720));
        hostDevice.writeFloat32(sampledSession.bindings.output, Array(tinyLlamaVocabSize).fill(-7140));
        const expectedSampledLayer0Cache = {
          k: Array(sampledLayer0KElements).fill(-1710),
          v: Array(sampledLayer0VElements).fill(-1720),
        };
        const expectedSampledLayer1Cache = {
          k: Array(sampledLayer1KElements).fill(-2710),
          v: Array(sampledLayer1VElements).fill(-2720),
        };
        const sampleOptions = { seed: 321, temperature: 0.75, topK: 3 };
        const expectedSampledLayer0First = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedSampledLayer0Cache, 0, 0, { layer: 0 });
        const expectedSampledLayer1First = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedSampledLayer1Cache, 0, 0, {
          inputHidden: expectedSampledLayer0First.blockHidden,
          layer: 1,
        });
        const expectedSampledFirst = sampleTopKOf(expectedSampledLayer1First.logits, sampleOptions);
        const expectedSampledLayer0Second = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedSampledLayer0Cache, expectedSampledFirst.token, 1, { layer: 0 });
        const expectedSampledLayer1Second = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedSampledLayer1Cache, expectedSampledFirst.token, 1, {
          inputHidden: expectedSampledLayer0Second.blockHidden,
          layer: 1,
        });
        const expectedSampledSecond = sampleTopKOf(expectedSampledLayer1Second.logits, {
          ...sampleOptions,
          seed: sampleOptions.seed + 1,
        });
        const sampled = await sampledSession.generateTokensSample([0], 2, {
          ...sampleOptions,
          includeLogits: true,
        });
        expectStatus(sampled.status, ok, "host-only WebGPU tiny LLaMA ergonomic sampled generation");
        if (
          !(sampled.tokens instanceof Uint32Array) ||
          sampled.tokensGenerated !== 2 ||
          sampled.tokens[0] !== expectedSampledFirst.token ||
          sampled.tokens[1] !== expectedSampledSecond.token ||
          sampled.lastToken !== expectedSampledSecond.token ||
          Math.abs(sampled.lastLogit - expectedSampledSecond.logit) > 1e-5 ||
          sampled.position !== 2 ||
          sessionPositionWith(twoLayerHostExports, sampledHandle) !== 2 ||
          !(sampled.logits instanceof Float32Array)
        ) {
          throw new Error("host-only WebGPU tiny LLaMA ergonomic sampled generation returned unexpected tokens");
        }
        expectClose(Array.from(sampled.logits), expectedSampledLayer1Second.logits);
        expectClose(
          Array.from(await hostDevice.readFloat32(sampledSession.bindings.output, tinyLlamaVocabSize)),
          expectedSampledLayer1Second.logits,
        );
        const sampledProfile = sampledSession.runtimeProfile();
        if (
          sampledProfile.callCount !== 2 ||
          sampledProfile.decodedCallCount !== 2 ||
          sampledProfile.logitsOutputCallCount !== 2 ||
          sampledProfile.boundOutputCallCount !== 2 ||
          sampledProfile.executorCallCount !== 2 ||
          sampledProfile.executorOkCount !== 2 ||
          sampledProfile.executorBackendDispatchCount !== expectedTwoLayerBackendDispatchCount ||
          sampledProfile.executorFallbackOpCount !== expectedTwoLayerFallbackOpCount ||
          sampledProfile.executorCommandCount !== 4 ||
          sampledProfile.executorCommandOpCount !== 4 ||
          sampledProfile.outputReadCount !== 2 ||
          sampledProfile.syncCount !== 2 ||
          sampledProfile.lastStatus !== ok ||
          sampledProfile.lastResultOutputLength !== tinyLlamaVocabSize
        ) {
          throw new Error(`unexpected host-only WebGPU tiny LLaMA ergonomic sampled profile: ${JSON.stringify(sampledProfile)}`);
        }
        recordBrowserLlamaProfileEvidence("ergonomic-sampled", sampledProfile);
        const sampledAbiProfile = sessionRuntimeProfileWith(twoLayerHostExports, sampledHandle);
        if (
          u64(sampledAbiProfile, 0) !== 2n ||
          u64(sampledAbiProfile, 8) !== BigInt(expectedTwoLayerBackendDispatchCount) ||
          u64(sampledAbiProfile, 16) !== BigInt(expectedTwoLayerFallbackOpCount) ||
          u64(sampledAbiProfile, 24) !== BigInt(expectedTwoLayerBackendDispatchCount) ||
          u64(sampledAbiProfile, 32) !== 2n ||
          u64(sampledAbiProfile, 40) !== 2n ||
          u64(sampledAbiProfile, 48) !== 2n ||
          u64(sampledAbiProfile, 56) !== 0n ||
          u64(sampledAbiProfile, 96) !== 4n ||
          u64(sampledAbiProfile, 112) !== 4n
        ) {
          throw new Error("unexpected host-only WebGPU tiny LLaMA ergonomic sampled ABI runtime profile");
        }
        const sampledCalls = twoLayerPipelineExecutorCalls.slice(sampledCallStart);
        if (
          sampledCalls.length !== 2 ||
          sampledCalls[0].outputPolicy !== 1 ||
          sampledCalls[0].tokenCount !== 1 ||
          sampledCalls[0].commandCount !== 2 ||
          sampledCalls[1].outputPolicy !== 1 ||
          sampledCalls[1].tokenCount !== 1 ||
          sampledCalls[1].commandCount !== 2 ||
          sampledCalls[1].outputLength !== tinyLlamaVocabSize
        ) {
          throw new Error("host-only WebGPU tiny LLaMA ergonomic sampled executor evidence mismatch");
        }
        const deviceSampledCallStart = twoLayerPipelineExecutorCalls.length;
        const deviceSampledSession = twoLayerPipelineProgram.bindHostResources();
        const deviceSampledHandle = deviceSampledSession.handle;
        try {
          const deviceSampledLayer0K = deviceSampledSession.bindings.kvCache[0].k;
          const deviceSampledLayer0V = deviceSampledSession.bindings.kvCache[0].v;
          const deviceSampledLayer1K = deviceSampledSession.bindings.kvCache[1].k;
          const deviceSampledLayer1V = deviceSampledSession.bindings.kvCache[1].v;
          hostDevice.writeFloat32(deviceSampledLayer0K, Array(sampledLayer0KElements).fill(-1710));
          hostDevice.writeFloat32(deviceSampledLayer0V, Array(sampledLayer0VElements).fill(-1720));
          hostDevice.writeFloat32(deviceSampledLayer1K, Array(sampledLayer1KElements).fill(-2710));
          hostDevice.writeFloat32(deviceSampledLayer1V, Array(sampledLayer1VElements).fill(-2720));
          hostDevice.writeFloat32(deviceSampledSession.bindings.output, Array(tinyLlamaVocabSize).fill(-7240));
          const deviceSampled = await deviceSampledSession.generateTokensSampleDevice([0], 2, sampleOptions);
          expectStatus(deviceSampled.status, ok, "host-only WebGPU tiny LLaMA device-sampled generation");
          if (
            !(deviceSampled.tokens instanceof Uint32Array) ||
            deviceSampled.tokensGenerated !== 2 ||
            deviceSampled.tokens[0] !== expectedSampledFirst.token ||
            deviceSampled.tokens[1] !== expectedSampledSecond.token ||
            deviceSampled.lastToken !== expectedSampledSecond.token ||
            Math.abs(deviceSampled.lastLogit - expectedSampledSecond.logit) > 1e-5 ||
            deviceSampled.position !== 2 ||
            deviceSampled.logits !== undefined ||
            sessionPositionWith(twoLayerHostExports, deviceSampledHandle) !== 2
          ) {
            throw new Error("host-only WebGPU tiny LLaMA device-sampled generation returned unexpected tokens");
          }
          expectClose(
            Array.from(await hostDevice.readFloat32(deviceSampledSession.bindings.output, tinyLlamaVocabSize)),
            expectedSampledLayer1Second.logits,
          );
          const deviceSampledUsesGpu = canUseGpuStorageBuffers(hostDevice, 6);
          const deviceSampledProfile = deviceSampledSession.runtimeProfile();
          if (
            deviceSampledProfile.callCount !== 2 ||
            deviceSampledProfile.decodedCallCount !== 2 ||
            deviceSampledProfile.logitsOutputCallCount !== 2 ||
            deviceSampledProfile.boundOutputCallCount !== 2 ||
            deviceSampledProfile.executorCallCount !== 2 ||
            deviceSampledProfile.executorOkCount !== 2 ||
            deviceSampledProfile.executorBackendDispatchCount !== expectedTwoLayerBackendDispatchCount ||
            deviceSampledProfile.executorFallbackOpCount !== expectedTwoLayerFallbackOpCount ||
            deviceSampledProfile.executorCommandCount !== 4 ||
            deviceSampledProfile.executorCommandOpCount !== 4 ||
            deviceSampledProfile.outputReadCount !== (deviceSampledUsesGpu ? 0 : 2) ||
            deviceSampledProfile.syncCount !== (deviceSampledUsesGpu ? 0 : 2) ||
            deviceSampledProfile.selectionCallCount !== 2 ||
            deviceSampledProfile.selectionBackendDispatchCount !== (deviceSampledUsesGpu ? 2 : 0) ||
            deviceSampledProfile.selectionFallbackOpCount !== (deviceSampledUsesGpu ? 0 : 2) ||
            deviceSampledProfile.selectionResultReadCount !== (deviceSampledUsesGpu ? 2 : 0) ||
            deviceSampledProfile.selectionSyncCount !== (deviceSampledUsesGpu ? 2 : 0) ||
            deviceSampledProfile.lastStatus !== ok ||
            deviceSampledProfile.lastResultOutputLength !== tinyLlamaVocabSize
          ) {
            throw new Error(`unexpected host-only WebGPU tiny LLaMA device-sampled profile: ${JSON.stringify(deviceSampledProfile)}`);
          }
          recordBrowserLlamaProfileEvidence("device-sampled", deviceSampledProfile);
          const deviceSampledCalls = twoLayerPipelineExecutorCalls.slice(deviceSampledCallStart);
          if (
            deviceSampledCalls.length !== 2 ||
            deviceSampledCalls[0].outputPolicy !== 1 ||
            deviceSampledCalls[0].tokenCount !== 1 ||
            deviceSampledCalls[0].commandCount !== 2 ||
            deviceSampledCalls[1].outputPolicy !== 1 ||
            deviceSampledCalls[1].tokenCount !== 1 ||
            deviceSampledCalls[1].commandCount !== 2 ||
            deviceSampledCalls[1].outputLength !== tinyLlamaVocabSize
          ) {
            throw new Error("host-only WebGPU tiny LLaMA device-sampled executor evidence mismatch");
          }
        } finally {
          expectStatus(twoLayerHostExports.zgml_session_free(deviceSampledHandle), ok, "host-only WebGPU tiny LLaMA device-sampled Session free");
        }
      } finally {
        expectStatus(twoLayerHostExports.zgml_session_free(sampledHandle), ok, "host-only WebGPU tiny LLaMA ergonomic sampled Session free");
      }
      const sampledOverContextSession = twoLayerPipelineProgram.bindHostResources();
      const sampledOverContextHandle = sampledOverContextSession.handle;
      try {
        const sampledOverContextOutput = new Uint32Array([555, 555, 555, 555]);
        const sampledOverContextSentinel = Array(tinyLlamaVocabSize).fill(-8141);
        const sampledOverContextK = sampledOverContextSession.bindings.kvCache[0].k;
        const sampledOverContextV = sampledOverContextSession.bindings.kvCache[0].v;
        const sampledOverContextKElements = Math.floor(sampledOverContextK.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const sampledOverContextVElements = Math.floor(sampledOverContextV.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const sampledOverContextKSentinel = Array(sampledOverContextKElements).fill(-1910);
        const sampledOverContextVSentinel = Array(sampledOverContextVElements).fill(-1920);
        hostDevice.writeFloat32(sampledOverContextSession.bindings.output, sampledOverContextSentinel);
        hostDevice.writeFloat32(sampledOverContextK, sampledOverContextKSentinel);
        hostDevice.writeFloat32(sampledOverContextV, sampledOverContextVSentinel);
        const sampledOverContext = await sampledOverContextSession.generateTokensSampleInto([0, 1], sampledOverContextOutput, {
          seed: 9,
          temperature: 0.75,
          topK: 3,
          includeLogits: true,
        });
        expectStatus(sampledOverContext.status, shapeMismatch, "host-only WebGPU tiny LLaMA ergonomic sampled over-context generation");
        if (
          sampledOverContext.tokensGenerated !== 0 ||
          sampledOverContext.lastToken !== 0 ||
          sampledOverContext.lastLogit !== 0 ||
          sampledOverContext.position !== 0 ||
          sampledOverContext.logits !== undefined ||
          sessionPositionWith(twoLayerHostExports, sampledOverContextHandle) !== 0 ||
          Array.from(sampledOverContextOutput).some((token) => token !== 555)
        ) {
          throw new Error("host-only WebGPU tiny LLaMA ergonomic sampled over-context generation mutated state");
        }
        expectClose(Array.from(await hostDevice.readFloat32(sampledOverContextSession.bindings.output, tinyLlamaVocabSize)), sampledOverContextSentinel);
        expectClose(Array.from(await hostDevice.readFloat32(sampledOverContextK, sampledOverContextKElements)), sampledOverContextKSentinel);
        expectClose(Array.from(await hostDevice.readFloat32(sampledOverContextV, sampledOverContextVElements)), sampledOverContextVSentinel);
        const sampledOverContextProfile = sampledOverContextSession.runtimeProfile();
        if (
          sampledOverContextProfile.callCount !== 0 ||
          sampledOverContextProfile.executorCallCount !== 0 ||
          sampledOverContextProfile.executorBackendDispatchCount !== 0 ||
          sampledOverContextProfile.executorFallbackOpCount !== 0 ||
          sampledOverContextProfile.outputReadCount !== 0 ||
          sampledOverContextProfile.syncCount !== 0 ||
          sampledOverContextProfile.lastStatus !== null
        ) {
          throw new Error(`unexpected host-only WebGPU tiny LLaMA sampled over-context generation profile: ${JSON.stringify(sampledOverContextProfile)}`);
        }
      } finally {
        expectStatus(twoLayerHostExports.zgml_session_free(sampledOverContextHandle), ok, "host-only WebGPU tiny LLaMA ergonomic sampled over-context Session free");
      }
      const directDisposeHostSession = twoLayerPipelineProgram.bindHostResources();
      const directDisposeHostHandle = directDisposeHostSession.handle;
      directDisposeHostSession.dispose();
      if (!directDisposeHostSession.freed || hostRuntime.lookup(directDisposeHostHandle) !== null) {
        throw new Error("host-only WebGPU tiny LLaMA direct JS dispose did not release Session");
      }
      if (hostRuntime.hasClaimed(directDisposeHostHandle)) {
        throw new Error("host-only WebGPU tiny LLaMA direct JS dispose did not release Session claim");
      }
      expectStatus(
        hostRuntime.executeTokensSync(directDisposeHostHandle, 0, 0),
        hostRuntime.status.unclaimed,
        "direct-disposed host-only WebGPU tiny LLaMA host callback execute",
      );
    } finally {
      expectStatus(twoLayerHostExports.zgml_session_free(twoLayerPipelineSessionHandle), ok, "host-only WebGPU tiny LLaMA two-layer Session free");
      if (!twoLayerPipelineSession.freed || hostRuntime.lookup(twoLayerPipelineSessionHandle) !== null) {
        throw new Error("host-only WebGPU tiny LLaMA two-layer Session was not released");
      }
      if (hostRuntime.hasClaimed(twoLayerPipelineSessionHandle)) {
        throw new Error("host-only WebGPU tiny LLaMA two-layer Session claim was not released");
      }
      twoLayerPipelineProgram.tokenExecutor?.destroy?.();
    }

    const shardedTwoLayerShard0Specs = [
      ...tinyLlamaRootTensorSpecs,
      ...tinyLlamaLayerTensorSpecs(0),
    ];
    const shardedTwoLayerShard1Specs = tinyLlamaLayerTensorSpecs(1);
    const shardedTwoLayerBytes = [
      tinyLlamaModelResourceFileBytes(shardedTwoLayerShard0Specs),
      tinyLlamaModelResourceFileBytes(shardedTwoLayerShard1Specs, {
        tensorIndexOffset: shardedTwoLayerShard0Specs.length,
      }),
    ];
    const shardedTwoLayerShard0Name = "model-00001-of-00002.safetensors";
    const shardedTwoLayerShard1Name = "model-00002-of-00002.safetensors";
    const shardedTwoLayerTotalSize = shardedTwoLayerBytes.reduce((sum, bytes) => sum + bytes.byteLength - safetensorsDataStart(bytes), 0);
    const shardedTwoLayerSource = {
      index: {
        metadata: {
          total_size: shardedTwoLayerTotalSize,
        },
        weight_map: Object.fromEntries([
          ...shardedTwoLayerShard0Specs.map(([name]) => [name, shardedTwoLayerShard0Name]),
          ...shardedTwoLayerShard1Specs.map(([name]) => [name, shardedTwoLayerShard1Name]),
        ]),
      },
      shards: [
        { name: shardedTwoLayerShard0Name, data: shardedTwoLayerBytes[0] },
        { name: shardedTwoLayerShard1Name, data: shardedTwoLayerBytes[1] },
      ],
    };
    const shardedMissingOffsetBytes = mutateSafetensorsHeaderBytes(
      shardedTwoLayerBytes[1],
      (header) => {
        delete header["model.layers.1.self_attn.q_proj.weight"].data_offsets;
      },
    );
    let shardedMissingOffsetRejected = false;
    try {
      WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        contextLength: 4,
        safetensors: {
          ...shardedTwoLayerSource,
          shards: [
            shardedTwoLayerSource.shards[0],
            { name: shardedTwoLayerShard1Name, data: shardedMissingOffsetBytes },
          ],
        },
      });
    } catch (err) {
      if (!String(err?.message ?? err).includes("missing data_offsets")) throw err;
      shardedMissingOffsetRejected = true;
    }
    if (!shardedMissingOffsetRejected) {
      throw new Error("host-only WebGPU tiny LLaMA sharded safetensors accepted a named byte-backed shard with missing data_offsets");
    }
    const shardedTwoLayerExecutorCalls = [];
    const shardedTwoLayerProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      contextLength: 4,
      model: compatibleModel,
      safetensors: shardedTwoLayerSource,
      allowMockFallback: true,
      executorOptions: {
        record(call) {
          recordBrowserLlamaPipelineCallEvidence(shardedTwoLayerExecutorCalls, "sharded-two-layer", call);
        },
      },
    });
	    if (
	      shardedTwoLayerProgram.requiredModelResources.length !== twoLayerModelResourceRequirements.length ||
	      shardedTwoLayerProgram.tokenExecutor.blocks.length !== 2 ||
	      shardedTwoLayerProgram.tokenExecutor.blocks[0].executor.layer !== 0 ||
	      shardedTwoLayerProgram.tokenExecutor.blocks[1].executor.layer !== 1
	    ) {
	      throw new Error("host-only WebGPU tiny LLaMA sharded safetensors Program did not infer the two-layer shape");
	    }
	    let nativeShardedModel = 0;
	    let nativeShardedCompatibleModel = 0;
	    let nativeShardedProgramHandle = 0;
	    let nativeShardedProgram = null;
	    let nativeShardedSession = null;
	    let nativeShardedSessionHandle = 0;
	    const nativeShardedExecutorCalls = [];
	    try {
	      nativeShardedModel = createModel(tinyLlama2LayerKind, 0, 0);
	      nativeShardedCompatibleModel = createModel(tinyLlama2LayerKind, 0, 0);
	      nativeShardedProgramHandle = compileProgram(nativeShardedModel, 4, backendWebGpu);
	      const nativeShardedKvRequirements = keep(alloc(llamaKvCacheRequirementsSize), llamaKvCacheRequirementsSize);
	      zero(nativeShardedKvRequirements, llamaKvCacheRequirementsSize);
	      check(exportsRef.zgml_llama_program_get_kv_cache_requirements(nativeShardedProgramHandle, nativeShardedKvRequirements));
	      nativeShardedProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
	        device: hostDevice,
	        resourceBridge: hostResourceBridge,
	        sessionBindingBridge,
	        runtime: hostRuntime,
	        program: nativeShardedProgramHandle,
	        contextLength: 4,
	        model: nativeShardedCompatibleModel,
	        safetensors: shardedTwoLayerSource,
	        allowMockFallback: true,
	        executorOptions: {
	          record(call) {
	            recordBrowserLlamaPipelineCallEvidence(nativeShardedExecutorCalls, "native-sharded-two-layer", call);
	          },
	        },
	      });
	      if (
	        nativeShardedProgram.kvRequirements.layers !== 2 ||
	        nativeShardedProgram.kvRequirements.kBufferByteLength !== u32(nativeShardedKvRequirements, 20) ||
	        nativeShardedProgram.kvRequirements.vBufferByteLength !== u32(nativeShardedKvRequirements, 24) ||
	        nativeShardedProgram.requiredModelResources.length !== twoLayerModelResourceRequirements.length
	      ) {
	        throw new Error("native-backed WebGPU tiny LLaMA sharded safetensors Program did not match native K/V requirements and model manifest");
	      }
	      nativeShardedSession = nativeShardedProgram.bind();
	      nativeShardedSessionHandle = nativeShardedSession.handle;
	      if (nativeShardedSession.nativeSession !== true || !hostRuntime.hasClaimed(nativeShardedSessionHandle)) {
	        throw new Error("native-backed WebGPU tiny LLaMA sharded safetensors Session was not registered for raw ABI execution");
	      }
	      const nativeShard0ResourceCount = nativeShardedSession.modelResources.filter((entry) => entry.safetensorsShardIndex === 0).length;
	      const nativeShard1ResourceCount = nativeShardedSession.modelResources.filter((entry) => entry.safetensorsShardIndex === 1).length;
	      const nativeShard0NameCount = nativeShardedSession.modelResources.filter((entry) => entry.safetensorsShardName === shardedTwoLayerShard0Name).length;
	      const nativeShard1NameCount = nativeShardedSession.modelResources.filter((entry) => entry.safetensorsShardName === shardedTwoLayerShard1Name).length;
	      if (
	        nativeShardedSession.modelHandle !== nativeShardedCompatibleModel ||
	        nativeShardedSession.modelResources.length !== twoLayerModelResourceRequirements.length ||
	        nativeShard0ResourceCount !== shardedTwoLayerShard0Specs.length ||
	        nativeShard1ResourceCount !== shardedTwoLayerShard1Specs.length ||
	        nativeShard0NameCount !== shardedTwoLayerShard0Specs.length ||
	        nativeShard1NameCount !== shardedTwoLayerShard1Specs.length
	      ) {
	        throw new Error("native-backed WebGPU tiny LLaMA sharded safetensors bind lost shard provenance");
	      }
	      expectHostResourceTableRoles(
	        nativeShardedSession.table,
	        ["llama.output", "llama.k.0", "llama.v.0", "llama.k.1", "llama.v.1"],
	        "native-backed WebGPU tiny LLaMA sharded safetensors",
	      );
	      const nativeShardedLayer0K = nativeShardedSession.bindings.kvCache[0].k;
	      const nativeShardedLayer0V = nativeShardedSession.bindings.kvCache[0].v;
	      const nativeShardedLayer1K = nativeShardedSession.bindings.kvCache[1].k;
	      const nativeShardedLayer1V = nativeShardedSession.bindings.kvCache[1].v;
	      const nativeShardedLayer0KElements = Math.floor(nativeShardedLayer0K.byteLength / Float32Array.BYTES_PER_ELEMENT);
	      const nativeShardedLayer0VElements = Math.floor(nativeShardedLayer0V.byteLength / Float32Array.BYTES_PER_ELEMENT);
	      const nativeShardedLayer1KElements = Math.floor(nativeShardedLayer1K.byteLength / Float32Array.BYTES_PER_ELEMENT);
	      const nativeShardedLayer1VElements = Math.floor(nativeShardedLayer1V.byteLength / Float32Array.BYTES_PER_ELEMENT);
	      const expectedNativeShardedLayer0Cache = {
	        k: Array(nativeShardedLayer0KElements).fill(-3410),
	        v: Array(nativeShardedLayer0VElements).fill(-3420),
	      };
	      const expectedNativeShardedLayer1Cache = {
	        k: Array(nativeShardedLayer1KElements).fill(-4410),
	        v: Array(nativeShardedLayer1VElements).fill(-4420),
	      };
	      hostDevice.writeFloat32(nativeShardedLayer0K, expectedNativeShardedLayer0Cache.k);
	      hostDevice.writeFloat32(nativeShardedLayer0V, expectedNativeShardedLayer0Cache.v);
	      hostDevice.writeFloat32(nativeShardedLayer1K, expectedNativeShardedLayer1Cache.k);
	      hostDevice.writeFloat32(nativeShardedLayer1V, expectedNativeShardedLayer1Cache.v);
	      hostDevice.writeFloat32(nativeShardedSession.bindings.output, Array(tinyLlamaVocabSize).fill(-6160));
	      const nativeShardedTokenPtr = keep(alloc(4), 4);
	      writeU32Array(nativeShardedTokenPtr, [3]);
	      const nativeShardedResult = tokenExecute(nativeShardedSessionHandle, nativeShardedTokenPtr, 1, executeOutputLogits, 0, 0);
	      expectStatus(nativeShardedResult.status, ok, "native-backed WebGPU tiny LLaMA sharded safetensors raw execute");
	      if (nativeShardedResult.outputLen !== tinyLlamaVocabSize || sessionPosition(nativeShardedSessionHandle) !== 1) {
	        throw new Error("native-backed WebGPU tiny LLaMA sharded safetensors raw execute did not advance");
	      }
	      const nativeShardedInspection = inspectSession(nativeShardedSessionHandle);
	      if (
	        u32(nativeShardedInspection, 0) !== tinyLlama2LayerKind ||
	        u32(nativeShardedInspection, 4) !== backendWebGpu ||
	        u32(nativeShardedInspection, 8) !== bufferStorageExternalResource ||
	        u32(nativeShardedInspection, 12) !== bufferStorageExternalResource ||
	        u64(nativeShardedInspection, 16) !== 1n ||
	        u64(nativeShardedInspection, 24) !== 4n ||
	        u64(nativeShardedInspection, 64) !== 5n ||
	        u64(nativeShardedInspection, 72) === 0n
	      ) {
	        throw new Error("native-backed WebGPU tiny LLaMA sharded safetensors raw inspect mismatch");
	      }
	      const expectedNativeShardedLayer0 = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedNativeShardedLayer0Cache, 3, 0, { layer: 0 });
	      const expectedNativeShardedLayer1 = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedNativeShardedLayer1Cache, 3, 0, {
	        inputHidden: expectedNativeShardedLayer0.blockHidden,
	        layer: 1,
	      });
	      expectClose(
	        Array.from(await hostDevice.readFloat32(nativeShardedSession.bindings.output, tinyLlamaVocabSize)),
	        expectedNativeShardedLayer1.logits,
	      );
	      expectClose(Array.from(await hostDevice.readFloat32(nativeShardedLayer0K, expectedNativeShardedLayer0.k.length, 0)), expectedNativeShardedLayer0.k);
	      expectClose(Array.from(await hostDevice.readFloat32(nativeShardedLayer0V, expectedNativeShardedLayer0.v.length, 0)), expectedNativeShardedLayer0.v);
	      expectClose(Array.from(await hostDevice.readFloat32(nativeShardedLayer1K, expectedNativeShardedLayer1.k.length, 0)), expectedNativeShardedLayer1.k);
	      expectClose(Array.from(await hostDevice.readFloat32(nativeShardedLayer1V, expectedNativeShardedLayer1.v.length, 0)), expectedNativeShardedLayer1.v);
	      const expectedNativeShardedCommandCount = expectedLlamaBlockPipelineCommandCount(hostDevice, 2, 1, executeOutputLogits);
	      const expectedNativeShardedBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? expectedNativeShardedCommandCount : 0;
	      const expectedNativeShardedFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : expectedNativeShardedCommandCount;
	      const nativeShardedProfile = nativeShardedSession.runtimeProfile();
	      if (
	        nativeShardedProfile.executorCallCount !== 1 ||
	        nativeShardedProfile.executorOkCount !== 1 ||
	        nativeShardedProfile.executorBackendDispatchCount !== expectedNativeShardedBackendDispatchCount ||
	        nativeShardedProfile.executorFallbackOpCount !== expectedNativeShardedFallbackOpCount ||
	        nativeShardedProfile.executorCommandCount !== expectedNativeShardedCommandCount ||
	        nativeShardedProfile.executorCommandOpCount !== expectedNativeShardedCommandCount ||
	        nativeShardedProfile.lastStatus !== ok ||
	        nativeShardedProfile.lastResultOutputLength !== tinyLlamaVocabSize
	      ) {
	        throw new Error(`unexpected native-backed WebGPU tiny LLaMA sharded safetensors profile: ${JSON.stringify(nativeShardedProfile)}`);
	      }
	      const nativeShardedAbiProfile = sessionRuntimeProfile(nativeShardedSessionHandle);
	      if (
	        u64(nativeShardedAbiProfile, 0) !== 1n ||
	        u64(nativeShardedAbiProfile, 8) !== BigInt(expectedNativeShardedBackendDispatchCount) ||
	        u64(nativeShardedAbiProfile, 16) !== BigInt(expectedNativeShardedFallbackOpCount) ||
	        u64(nativeShardedAbiProfile, 24) !== BigInt(expectedNativeShardedBackendDispatchCount) ||
	        u64(nativeShardedAbiProfile, 40) !== 1n ||
	        u64(nativeShardedAbiProfile, 48) !== 1n ||
	        u64(nativeShardedAbiProfile, 56) !== 0n ||
	        u64(nativeShardedAbiProfile, 96) !== BigInt(expectedNativeShardedCommandCount) ||
	        u64(nativeShardedAbiProfile, 112) !== BigInt(expectedNativeShardedCommandCount)
	      ) {
	        throw new Error("native-backed WebGPU tiny LLaMA sharded safetensors ABI profile mismatch");
	      }
	      if (
	        nativeShardedExecutorCalls.length !== 1 ||
	        nativeShardedExecutorCalls[0].stageCount !== 2 ||
	        nativeShardedExecutorCalls[0].commandCount !== expectedNativeShardedCommandCount ||
	        nativeShardedExecutorCalls[0].status !== ok
	      ) {
	        throw new Error("native-backed WebGPU tiny LLaMA sharded safetensors executor evidence mismatch");
	      }
	    } finally {
	      if (nativeShardedSessionHandle !== 0) exportsRef.zgml_session_free(nativeShardedSessionHandle);
	      if (nativeShardedSession && nativeShardedSession.ownedResources.length !== 0) {
	        throw new Error("native-backed WebGPU tiny LLaMA sharded safetensors Session-owned resources were not released");
	      }
	      nativeShardedProgram?.tokenExecutor?.destroy?.();
	      if (nativeShardedProgramHandle !== 0) exportsRef.zgml_program_free(nativeShardedProgramHandle);
	      if (nativeShardedCompatibleModel !== 0) exportsRef.zgml_model_free(nativeShardedCompatibleModel);
	      if (nativeShardedModel !== 0) exportsRef.zgml_model_free(nativeShardedModel);
	    }
	    const shardedTwoLayerSession = shardedTwoLayerProgram.bindHostResources();
	    const shardedTwoLayerSessionHandle = shardedTwoLayerSession.handle;
    const shardedTwoLayerHostExports = hostRuntime.wrapExports(exportsRef);
    try {
      const shard0ResourceCount = shardedTwoLayerSession.modelResources.filter((entry) => entry.safetensorsShardIndex === 0).length;
      const shard1ResourceCount = shardedTwoLayerSession.modelResources.filter((entry) => entry.safetensorsShardIndex === 1).length;
      const shard0NameCount = shardedTwoLayerSession.modelResources.filter((entry) => entry.safetensorsShardName === shardedTwoLayerShard0Name).length;
      const shard1NameCount = shardedTwoLayerSession.modelResources.filter((entry) => entry.safetensorsShardName === shardedTwoLayerShard1Name).length;
      if (
        shardedTwoLayerSession.modelResources.length !== twoLayerModelResourceRequirements.length ||
        shard0ResourceCount !== shardedTwoLayerShard0Specs.length ||
        shard1ResourceCount !== shardedTwoLayerShard1Specs.length ||
        shard0NameCount !== shardedTwoLayerShard0Specs.length ||
        shard1NameCount !== shardedTwoLayerShard1Specs.length
      ) {
        throw new Error("host-only WebGPU tiny LLaMA sharded safetensors Session lost shard provenance");
      }
      const shardedLayer0K = shardedTwoLayerSession.bindings.kvCache[0].k;
      const shardedLayer0V = shardedTwoLayerSession.bindings.kvCache[0].v;
      const shardedLayer1K = shardedTwoLayerSession.bindings.kvCache[1].k;
      const shardedLayer1V = shardedTwoLayerSession.bindings.kvCache[1].v;
      const shardedLayer0KElements = Math.floor(shardedLayer0K.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const shardedLayer0VElements = Math.floor(shardedLayer0V.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const shardedLayer1KElements = Math.floor(shardedLayer1K.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const shardedLayer1VElements = Math.floor(shardedLayer1V.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(shardedLayer0K, Array(shardedLayer0KElements).fill(-3110));
      hostDevice.writeFloat32(shardedLayer0V, Array(shardedLayer0VElements).fill(-3120));
      hostDevice.writeFloat32(shardedLayer1K, Array(shardedLayer1KElements).fill(-4110));
      hostDevice.writeFloat32(shardedLayer1V, Array(shardedLayer1VElements).fill(-4120));
      hostDevice.writeFloat32(shardedTwoLayerSession.bindings.output, Array(tinyLlamaVocabSize).fill(-5150));
      const shardedTokenPtr = keep(alloc(4), 4);
      writeU32Array(shardedTokenPtr, [3]);
      const shardedResult = tokenExecuteWith(shardedTwoLayerHostExports, shardedTwoLayerSessionHandle, shardedTokenPtr, 1, executeOutputLogits, 0, 0);
      expectStatus(shardedResult.status, ok, "host-only WebGPU tiny LLaMA sharded safetensors block pipeline execute");
      if (shardedResult.outputLen !== tinyLlamaVocabSize || sessionPositionWith(shardedTwoLayerHostExports, shardedTwoLayerSessionHandle) !== 1) {
        throw new Error("host-only WebGPU tiny LLaMA sharded safetensors block pipeline did not emit bound logits and advance");
      }
      const expectedShardedLayer0Cache = {
        k: Array(shardedLayer0KElements).fill(-3110),
        v: Array(shardedLayer0VElements).fill(-3120),
      };
      const expectedShardedLayer1Cache = {
        k: Array(shardedLayer1KElements).fill(-4110),
        v: Array(shardedLayer1VElements).fill(-4120),
      };
      const expectedShardedLayer0 = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedShardedLayer0Cache, 3, 0, { layer: 0 });
      const expectedShardedLayer1 = tinyLlamaBlockProjectionStep(twoLayerModelResourceBytes, expectedShardedLayer1Cache, 3, 0, {
        inputHidden: expectedShardedLayer0.blockHidden,
        layer: 1,
      });
      expectClose(
        Array.from(await hostDevice.readFloat32(shardedTwoLayerSession.bindings.output, tinyLlamaVocabSize)),
        expectedShardedLayer1.logits,
      );
      expectClose(Array.from(await hostDevice.readFloat32(shardedLayer0K, expectedShardedLayer0.k.length, 0)), expectedShardedLayer0.k);
      expectClose(Array.from(await hostDevice.readFloat32(shardedLayer0V, expectedShardedLayer0.v.length, 0)), expectedShardedLayer0.v);
      expectClose(Array.from(await hostDevice.readFloat32(shardedLayer1K, expectedShardedLayer1.k.length, 0)), expectedShardedLayer1.k);
      expectClose(Array.from(await hostDevice.readFloat32(shardedLayer1V, expectedShardedLayer1.v.length, 0)), expectedShardedLayer1.v);
      const expectedShardedBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? 2 : 0;
      const expectedShardedFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : 2;
      const shardedProfile = shardedTwoLayerSession.runtimeProfile();
      if (
        shardedProfile.executorCallCount !== 1 ||
        shardedProfile.executorBackendDispatchCount !== expectedShardedBackendDispatchCount ||
        shardedProfile.executorFallbackOpCount !== expectedShardedFallbackOpCount ||
        shardedProfile.executorCommandCount !== 2 ||
        shardedProfile.executorCommandOpCount !== 2 ||
        shardedProfile.lastStatus !== ok ||
        shardedProfile.lastResultOutputLength !== tinyLlamaVocabSize
      ) {
        throw new Error(`unexpected host-only WebGPU tiny LLaMA sharded safetensors profile: ${JSON.stringify(shardedProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("sharded-two-layer", shardedProfile);
      if (
        shardedTwoLayerExecutorCalls.length !== 1 ||
        shardedTwoLayerExecutorCalls[0].stageCount !== 2 ||
        shardedTwoLayerExecutorCalls[0].commandCount !== 2 ||
        shardedTwoLayerExecutorCalls[0].status !== ok
      ) {
        throw new Error("host-only WebGPU tiny LLaMA sharded safetensors executor evidence mismatch");
      }
    } finally {
      expectStatus(shardedTwoLayerHostExports.zgml_session_free(shardedTwoLayerSessionHandle), ok, "host-only WebGPU tiny LLaMA sharded safetensors Session free");
      shardedTwoLayerProgram.tokenExecutor?.destroy?.();
    }

    let duplicateShardedTensorRejected = false;
    try {
      WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        contextLength: 4,
        safetensors: [shardedTwoLayerBytes[0], shardedTwoLayerBytes[0]],
      });
    } catch (err) {
      if (!String(err?.message ?? err).includes("duplicate safetensors tensor across shards")) throw err;
      duplicateShardedTensorRejected = true;
    }
    if (!duplicateShardedTensorRejected) {
      throw new Error("host-only WebGPU tiny LLaMA sharded safetensors accepted duplicate tensors");
    }

    let mismatchedShardedIndexRejected = false;
    try {
      WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        contextLength: 4,
        safetensors: {
          ...shardedTwoLayerSource,
          index: {
            weight_map: {
              ...shardedTwoLayerSource.index.weight_map,
              "model.layers.1.self_attn.q_proj.weight": shardedTwoLayerShard0Name,
            },
          },
        },
      });
    } catch (err) {
      if (!String(err?.message ?? err).includes("safetensors index weight_map mismatch")) throw err;
      mismatchedShardedIndexRejected = true;
    }
    if (!mismatchedShardedIndexRejected) {
      throw new Error("host-only WebGPU tiny LLaMA sharded safetensors accepted a mismatched index weight_map");
    }

    let mismatchedShardedTotalSizeRejected = false;
    try {
      WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        contextLength: 4,
        safetensors: {
          ...shardedTwoLayerSource,
          index: {
            ...shardedTwoLayerSource.index,
            metadata: {
              total_size: shardedTwoLayerTotalSize + Float32Array.BYTES_PER_ELEMENT,
            },
          },
        },
      });
    } catch (err) {
      if (!String(err?.message ?? err).includes("safetensors index metadata total_size mismatch")) throw err;
      mismatchedShardedTotalSizeRejected = true;
    }
    if (!mismatchedShardedTotalSizeRejected) {
      throw new Error("host-only WebGPU tiny LLaMA sharded safetensors accepted a mismatched index total_size");
    }

    for (const dtypeProof of [
      { dtype: "F16", sourceDtype: "f16" },
      { dtype: "BF16", sourceDtype: "bf16" },
    ]) {
      const halfPipelineExecutorCalls = [];
      const halfModelResourceBytes = tinyLlamaModelResourceFileBytes(tinyLlamaTwoLayerTensorSpecs, { dtype: dtypeProof.dtype });
      const halfPipelineProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        contextLength: 4,
        safetensors: halfModelResourceBytes,
        allowMockFallback: true,
        executorOptions: {
          record(call) {
            recordBrowserLlamaPipelineCallEvidence(halfPipelineExecutorCalls, `${dtypeProof.sourceDtype}-materialized-pipeline`, call);
          },
        },
      });
      let nativeHalfModel = 0;
      let nativeHalfCompatibleModel = 0;
      let nativeHalfProgramHandle = 0;
      let nativeHalfProgram = null;
      let nativeHalfSession = null;
      let nativeHalfSessionHandle = 0;
      const nativeHalfExecutorCalls = [];
      try {
        nativeHalfModel = createModel(tinyLlama2LayerKind, 0, 0);
        nativeHalfCompatibleModel = createModel(tinyLlama2LayerKind, 0, 0);
        nativeHalfProgramHandle = compileProgram(nativeHalfModel, 4, backendWebGpu);
        const nativeHalfKvRequirements = keep(alloc(llamaKvCacheRequirementsSize), llamaKvCacheRequirementsSize);
        zero(nativeHalfKvRequirements, llamaKvCacheRequirementsSize);
        check(exportsRef.zgml_llama_program_get_kv_cache_requirements(nativeHalfProgramHandle, nativeHalfKvRequirements));
        nativeHalfProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
          device: hostDevice,
          resourceBridge: hostResourceBridge,
          sessionBindingBridge,
          runtime: hostRuntime,
          program: nativeHalfProgramHandle,
          contextLength: 4,
          model: nativeHalfCompatibleModel,
          safetensors: halfModelResourceBytes,
          allowMockFallback: true,
          executorOptions: {
            record(call) {
              recordBrowserLlamaPipelineCallEvidence(nativeHalfExecutorCalls, `native-${dtypeProof.sourceDtype}-materialized-pipeline`, call);
            },
          },
        });
        if (
          nativeHalfProgram.kvRequirements.layers !== 2 ||
          nativeHalfProgram.kvRequirements.kBufferByteLength !== u32(nativeHalfKvRequirements, 20) ||
          nativeHalfProgram.kvRequirements.vBufferByteLength !== u32(nativeHalfKvRequirements, 24)
        ) {
          throw new Error(`native-backed WebGPU ${dtypeProof.dtype} LLaMA materialized Program did not match native K/V requirements`);
        }
        nativeHalfSession = nativeHalfProgram.bind();
        nativeHalfSessionHandle = nativeHalfSession.handle;
        if (nativeHalfSession.nativeSession !== true || !hostRuntime.hasClaimed(nativeHalfSessionHandle)) {
          throw new Error(`native-backed WebGPU ${dtypeProof.dtype} LLaMA materialized Session was not registered for raw ABI execution`);
        }
        if (
          nativeHalfSession.modelHandle !== nativeHalfCompatibleModel ||
          nativeHalfSession.modelResources.length !== nativeHalfProgram.requiredModelResources.length
        ) {
          throw new Error(`native-backed WebGPU ${dtypeProof.dtype} LLaMA materialized bind did not preserve model resources`);
        }
        const nativeHalfRawSpecs = new Map(tinyLlamaModelResourceSpecs(halfModelResourceBytes).map((spec) => [spec.tensor, spec]));
        const nativeHalfEmbeddingSpec = nativeHalfRawSpecs.get("model.embed_tokens.weight");
        const nativeHalfEmbeddingResource = nativeHalfSession.modelResources.find((entry) => entry.role === "llama.weight.model.embed_tokens.weight");
        if (
          !nativeHalfEmbeddingResource ||
          nativeHalfEmbeddingResource.dtype !== "f32" ||
          nativeHalfEmbeddingResource.sourceDtype !== dtypeProof.sourceDtype ||
          nativeHalfEmbeddingResource.byteLength !== nativeHalfEmbeddingResource.elementCount * Float32Array.BYTES_PER_ELEMENT ||
          nativeHalfEmbeddingResource.dataByteLength !== nativeHalfEmbeddingResource.elementCount * safetensorsScalarBytes(dtypeProof.dtype)
        ) {
          throw new Error(`native-backed WebGPU ${dtypeProof.dtype} LLaMA materialized model resource was not materialized to f32`);
        }
        expectClose(
          Array.from(await hostDevice.readFloat32(nativeHalfEmbeddingResource.descriptor, 4)),
          [0, 1, 2, 3].map((index) => safetensorsF32(halfModelResourceBytes, nativeHalfEmbeddingSpec, index)),
        );
        expectHostResourceTableRoles(
          nativeHalfSession.table,
          ["llama.output", "llama.k.0", "llama.v.0", "llama.k.1", "llama.v.1"],
          `native-backed WebGPU ${dtypeProof.dtype} LLaMA materialized`,
        );
        const nativeHalfLayer0K = nativeHalfSession.bindings.kvCache[0].k;
        const nativeHalfLayer0V = nativeHalfSession.bindings.kvCache[0].v;
        const nativeHalfLayer1K = nativeHalfSession.bindings.kvCache[1].k;
        const nativeHalfLayer1V = nativeHalfSession.bindings.kvCache[1].v;
        const nativeHalfLayer0KElements = Math.floor(nativeHalfLayer0K.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const nativeHalfLayer0VElements = Math.floor(nativeHalfLayer0V.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const nativeHalfLayer1KElements = Math.floor(nativeHalfLayer1K.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const nativeHalfLayer1VElements = Math.floor(nativeHalfLayer1V.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const expectedNativeHalfLayer0Cache = {
          k: Array(nativeHalfLayer0KElements).fill(-3310),
          v: Array(nativeHalfLayer0VElements).fill(-3320),
        };
        const expectedNativeHalfLayer1Cache = {
          k: Array(nativeHalfLayer1KElements).fill(-4310),
          v: Array(nativeHalfLayer1VElements).fill(-4320),
        };
        hostDevice.writeFloat32(nativeHalfLayer0K, expectedNativeHalfLayer0Cache.k);
        hostDevice.writeFloat32(nativeHalfLayer0V, expectedNativeHalfLayer0Cache.v);
        hostDevice.writeFloat32(nativeHalfLayer1K, expectedNativeHalfLayer1Cache.k);
        hostDevice.writeFloat32(nativeHalfLayer1V, expectedNativeHalfLayer1Cache.v);
        hostDevice.writeFloat32(nativeHalfSession.bindings.output, Array(tinyLlamaVocabSize).fill(-6140));
        const nativeHalfTokens = keep(alloc(4), 4);
        writeU32Array(nativeHalfTokens, [2]);
        const nativeHalfResult = tokenExecute(nativeHalfSessionHandle, nativeHalfTokens, 1, executeOutputLogits, 0, 0);
        expectStatus(nativeHalfResult.status, ok, `native-backed WebGPU ${dtypeProof.dtype} LLaMA materialized raw execute`);
        if (nativeHalfResult.outputLen !== tinyLlamaVocabSize || sessionPosition(nativeHalfSessionHandle) !== 1) {
          throw new Error(`native-backed WebGPU ${dtypeProof.dtype} LLaMA materialized raw execute did not advance`);
        }
        const nativeHalfInspection = inspectSession(nativeHalfSessionHandle);
        if (
          u32(nativeHalfInspection, 0) !== tinyLlama2LayerKind ||
          u32(nativeHalfInspection, 4) !== backendWebGpu ||
          u32(nativeHalfInspection, 8) !== bufferStorageExternalResource ||
          u32(nativeHalfInspection, 12) !== bufferStorageExternalResource ||
          u64(nativeHalfInspection, 16) !== 1n ||
          u64(nativeHalfInspection, 24) !== 4n ||
          u64(nativeHalfInspection, 64) !== 5n ||
          u64(nativeHalfInspection, 72) === 0n
        ) {
          throw new Error(`native-backed WebGPU ${dtypeProof.dtype} LLaMA materialized raw inspect mismatch`);
        }
        const expectedNativeHalfLayer0 = tinyLlamaBlockProjectionStep(halfModelResourceBytes, expectedNativeHalfLayer0Cache, 2, 0, { layer: 0 });
        const expectedNativeHalfLayer1 = tinyLlamaBlockProjectionStep(halfModelResourceBytes, expectedNativeHalfLayer1Cache, 2, 0, {
          inputHidden: expectedNativeHalfLayer0.blockHidden,
          layer: 1,
        });
        expectClose(
          Array.from(await hostDevice.readFloat32(nativeHalfSession.bindings.output, tinyLlamaVocabSize)),
          expectedNativeHalfLayer1.logits,
        );
        expectClose(Array.from(await hostDevice.readFloat32(nativeHalfLayer0K, expectedNativeHalfLayer0.k.length, 0)), expectedNativeHalfLayer0.k);
        expectClose(Array.from(await hostDevice.readFloat32(nativeHalfLayer0V, expectedNativeHalfLayer0.v.length, 0)), expectedNativeHalfLayer0.v);
        expectClose(Array.from(await hostDevice.readFloat32(nativeHalfLayer1K, expectedNativeHalfLayer1.k.length, 0)), expectedNativeHalfLayer1.k);
        expectClose(Array.from(await hostDevice.readFloat32(nativeHalfLayer1V, expectedNativeHalfLayer1.v.length, 0)), expectedNativeHalfLayer1.v);
        const expectedNativeHalfCommandCount = expectedLlamaBlockPipelineCommandCount(hostDevice, 2, 1, executeOutputLogits);
        const expectedNativeHalfBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? expectedNativeHalfCommandCount : 0;
        const expectedNativeHalfFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : expectedNativeHalfCommandCount;
        const nativeHalfProfile = nativeHalfSession.runtimeProfile();
        if (
          nativeHalfProfile.executorCallCount !== 1 ||
          nativeHalfProfile.executorOkCount !== 1 ||
          nativeHalfProfile.executorBackendDispatchCount !== expectedNativeHalfBackendDispatchCount ||
          nativeHalfProfile.executorFallbackOpCount !== expectedNativeHalfFallbackOpCount ||
          nativeHalfProfile.executorCommandCount !== expectedNativeHalfCommandCount ||
          nativeHalfProfile.executorCommandOpCount !== expectedNativeHalfCommandCount ||
          nativeHalfProfile.lastStatus !== ok ||
          nativeHalfProfile.lastResultOutputLength !== tinyLlamaVocabSize
        ) {
          throw new Error(`unexpected native-backed WebGPU ${dtypeProof.dtype} LLaMA materialized profile: ${JSON.stringify(nativeHalfProfile)}`);
        }
        const nativeHalfAbiProfile = sessionRuntimeProfile(nativeHalfSessionHandle);
        if (
          u64(nativeHalfAbiProfile, 0) !== 1n ||
          u64(nativeHalfAbiProfile, 8) !== BigInt(expectedNativeHalfBackendDispatchCount) ||
          u64(nativeHalfAbiProfile, 16) !== BigInt(expectedNativeHalfFallbackOpCount) ||
          u64(nativeHalfAbiProfile, 24) !== BigInt(expectedNativeHalfBackendDispatchCount) ||
          u64(nativeHalfAbiProfile, 40) !== 1n ||
          u64(nativeHalfAbiProfile, 48) !== 1n ||
          u64(nativeHalfAbiProfile, 56) !== 0n ||
          u64(nativeHalfAbiProfile, 96) !== BigInt(expectedNativeHalfCommandCount) ||
          u64(nativeHalfAbiProfile, 112) !== BigInt(expectedNativeHalfCommandCount)
        ) {
          throw new Error(`native-backed WebGPU ${dtypeProof.dtype} LLaMA materialized ABI profile mismatch`);
        }
        if (
          nativeHalfExecutorCalls.length !== 1 ||
          nativeHalfExecutorCalls[0].stageCount !== 2 ||
          nativeHalfExecutorCalls[0].commandCount !== expectedNativeHalfCommandCount ||
          nativeHalfExecutorCalls[0].status !== ok
        ) {
          throw new Error(`native-backed WebGPU ${dtypeProof.dtype} LLaMA materialized pipeline evidence mismatch`);
        }
      } finally {
        if (nativeHalfSessionHandle !== 0) exportsRef.zgml_session_free(nativeHalfSessionHandle);
        if (nativeHalfSession && nativeHalfSession.ownedResources.length !== 0) {
          throw new Error(`native-backed WebGPU ${dtypeProof.dtype} LLaMA materialized Session-owned resources were not released`);
        }
        nativeHalfProgram?.tokenExecutor?.destroy?.();
        if (nativeHalfProgramHandle !== 0) exportsRef.zgml_program_free(nativeHalfProgramHandle);
        if (nativeHalfCompatibleModel !== 0) exportsRef.zgml_model_free(nativeHalfCompatibleModel);
        if (nativeHalfModel !== 0) exportsRef.zgml_model_free(nativeHalfModel);
      }
      const halfPipelineSession = halfPipelineProgram.bindHostResources();
      const halfPipelineSessionHandle = halfPipelineSession.handle;
      const halfHostExports = hostRuntime.wrapExports(exportsRef);
      try {
        const rawSpecs = new Map(tinyLlamaModelResourceSpecs(halfModelResourceBytes).map((spec) => [spec.tensor, spec]));
        const embeddingSpec = rawSpecs.get("model.embed_tokens.weight");
        const embeddingResource = halfPipelineSession.modelResources.find((entry) => entry.role === "llama.weight.model.embed_tokens.weight");
        if (
          !embeddingResource ||
          embeddingResource.dtype !== "f32" ||
          embeddingResource.sourceDtype !== dtypeProof.sourceDtype ||
          embeddingResource.byteLength !== embeddingResource.elementCount * Float32Array.BYTES_PER_ELEMENT ||
          embeddingResource.dataByteLength !== embeddingResource.elementCount * safetensorsScalarBytes(dtypeProof.dtype)
        ) {
          throw new Error(`host-only WebGPU ${dtypeProof.dtype} LLaMA model resource was not materialized to f32`);
        }
        expectClose(
          Array.from(await hostDevice.readFloat32(embeddingResource.descriptor, 4)),
          [0, 1, 2, 3].map((index) => safetensorsF32(halfModelResourceBytes, embeddingSpec, index)),
        );
        const layer0K = halfPipelineSession.bindings.kvCache[0].k;
        const layer0V = halfPipelineSession.bindings.kvCache[0].v;
        const layer1K = halfPipelineSession.bindings.kvCache[1].k;
        const layer1V = halfPipelineSession.bindings.kvCache[1].v;
        const layer0KElements = Math.floor(layer0K.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const layer0VElements = Math.floor(layer0V.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const layer1KElements = Math.floor(layer1K.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const layer1VElements = Math.floor(layer1V.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const expectedLayer0Cache = {
          k: Array(layer0KElements).fill(-3110),
          v: Array(layer0VElements).fill(-3120),
        };
        const expectedLayer1Cache = {
          k: Array(layer1KElements).fill(-4110),
          v: Array(layer1VElements).fill(-4120),
        };
        hostDevice.writeFloat32(layer0K, expectedLayer0Cache.k);
        hostDevice.writeFloat32(layer0V, expectedLayer0Cache.v);
        hostDevice.writeFloat32(layer1K, expectedLayer1Cache.k);
        hostDevice.writeFloat32(layer1V, expectedLayer1Cache.v);
        hostDevice.writeFloat32(halfPipelineSession.bindings.output, Array(tinyLlamaVocabSize).fill(-5140));
        const halfTokens = keep(alloc(4), 4);
        writeU32Array(halfTokens, [2]);
        const halfResult = tokenExecuteWith(halfHostExports, halfPipelineSessionHandle, halfTokens, 1, executeOutputLogits, 0, 0);
        expectStatus(halfResult.status, ok, `host-only WebGPU ${dtypeProof.dtype} LLaMA materialized block pipeline execute`);
        const expectedLayer0 = tinyLlamaBlockProjectionStep(halfModelResourceBytes, expectedLayer0Cache, 2, 0, { layer: 0 });
        const expectedLayer1 = tinyLlamaBlockProjectionStep(halfModelResourceBytes, expectedLayer1Cache, 2, 0, {
          inputHidden: expectedLayer0.blockHidden,
          layer: 1,
        });
        expectClose(
          Array.from(await hostDevice.readFloat32(halfPipelineSession.bindings.output, tinyLlamaVocabSize)),
          expectedLayer1.logits,
        );
        expectClose(Array.from(await hostDevice.readFloat32(layer0K, expectedLayer0.k.length, 0)), expectedLayer0.k);
        expectClose(Array.from(await hostDevice.readFloat32(layer0V, expectedLayer0.v.length, 0)), expectedLayer0.v);
        expectClose(Array.from(await hostDevice.readFloat32(layer1K, expectedLayer1.k.length, 0)), expectedLayer1.k);
        expectClose(Array.from(await hostDevice.readFloat32(layer1V, expectedLayer1.v.length, 0)), expectedLayer1.v);
        const expectedHalfBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? 2 : 0;
        const expectedHalfFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : 2;
        const halfProfile = halfPipelineSession.runtimeProfile();
        if (
          halfProfile.executorCallCount !== 1 ||
          halfProfile.executorOkCount !== 1 ||
          halfProfile.executorBackendDispatchCount !== expectedHalfBackendDispatchCount ||
          halfProfile.executorFallbackOpCount !== expectedHalfFallbackOpCount ||
          halfProfile.executorCommandCount !== 2 ||
          halfProfile.executorCommandOpCount !== 2 ||
          halfProfile.lastStatus !== ok ||
          halfProfile.lastResultOutputLength !== tinyLlamaVocabSize
        ) {
          throw new Error(`unexpected host-only WebGPU ${dtypeProof.dtype} LLaMA materialized pipeline profile: ${JSON.stringify(halfProfile)}`);
        }
        if (
          halfPipelineExecutorCalls.length !== 1 ||
          halfPipelineExecutorCalls[0].stageCount !== 2 ||
          halfPipelineExecutorCalls[0].commandCount !== 2 ||
          halfPipelineExecutorCalls[0].status !== ok
        ) {
          throw new Error(`host-only WebGPU ${dtypeProof.dtype} LLaMA materialized pipeline evidence mismatch`);
        }
        recordBrowserLlamaProfileEvidence(`${dtypeProof.sourceDtype}-materialized-pipeline`, halfProfile);
      } finally {
        expectStatus(halfHostExports.zgml_session_free(halfPipelineSessionHandle), ok, `host-only WebGPU ${dtypeProof.dtype} LLaMA materialized Session free`);
        halfPipelineProgram.tokenExecutor?.destroy?.();
      }
    }

    const threeLayerPipelineExecutorCalls = [];
    const threeLayerModelResourceBytes = tinyLlamaModelResourceFileBytes(tinyLlamaThreeLayerTensorSpecs);
    const threeLayerModelResourceRequirements = tinyLlamaModelResourceSpecs(threeLayerModelResourceBytes);
    const threeLayerPipelineProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      contextLength: 4,
      safetensors: threeLayerModelResourceBytes,
      allowMockFallback: true,
      executorOptions: {
        record(call) {
          recordBrowserLlamaPipelineCallEvidence(threeLayerPipelineExecutorCalls, "three-layer-pipeline", call);
        },
      },
    });
    if (
      threeLayerPipelineProgram.kvRequirements.layers !== 3 ||
      threeLayerPipelineProgram.tokenExecutor.blocks.length !== 3 ||
      threeLayerPipelineProgram.tokenExecutor.blocks[0].executor.layer !== 0 ||
      threeLayerPipelineProgram.tokenExecutor.blocks[1].executor.layer !== 1 ||
      threeLayerPipelineProgram.tokenExecutor.blocks[2].executor.layer !== 2
    ) {
      throw new Error("host-only WebGPU tiny LLaMA three-layer safetensors Program did not infer pipeline layers");
    }
    const threeLayerPipelineSession = threeLayerPipelineProgram.bindHostResources();
    const threeLayerPipelineSessionHandle = threeLayerPipelineSession.handle;
    const threeLayerHostExports = hostRuntime.wrapExports(exportsRef);
    try {
      if (threeLayerPipelineSession.modelResources.length !== threeLayerModelResourceRequirements.length) {
        throw new Error("host-only WebGPU tiny LLaMA three-layer byte-backed bind did not create model resources");
      }
      expectHostResourceTableRoles(
        threeLayerPipelineSession.table,
        ["llama.output", "llama.k.0", "llama.v.0", "llama.k.1", "llama.v.1", "llama.k.2", "llama.v.2"],
        "host-only WebGPU tiny LLaMA three-layer block pipeline",
      );
      const threeLayerCaches = threeLayerPipelineSession.bindings.kvCache.map((layer, index) => {
        const kElements = Math.floor(layer.k.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const vElements = Math.floor(layer.v.byteLength / Float32Array.BYTES_PER_ELEMENT);
        hostDevice.writeFloat32(layer.k, Array(kElements).fill(-6100 - index * 100));
        hostDevice.writeFloat32(layer.v, Array(vElements).fill(-6200 - index * 100));
        return { k: layer.k, kElements, v: layer.v, vElements };
      });
      const threeLayerOutputSentinel = Array(tinyLlamaVocabSize).fill(-6160);
      hostDevice.writeFloat32(threeLayerPipelineSession.bindings.output, threeLayerOutputSentinel);
      const threeLayerPrefillTokens = [1, 2];
      const threeLayerPrefillTokensPtr = keep(alloc(8), 8);
      writeU32Array(threeLayerPrefillTokensPtr, threeLayerPrefillTokens);
      const threeLayerPrefill = tokenExecuteWith(threeLayerHostExports, threeLayerPipelineSessionHandle, threeLayerPrefillTokensPtr, 2, executeOutputNone, 0, 0);
      expectStatus(threeLayerPrefill.status, ok, "host-only WebGPU tiny LLaMA three-layer block pipeline no-output prefill");
      if (threeLayerPrefill.outputLen !== 0 || sessionPositionWith(threeLayerHostExports, threeLayerPipelineSessionHandle) !== 2) {
        throw new Error("host-only WebGPU tiny LLaMA three-layer block pipeline prefill did not advance without output");
      }
      expectClose(
        Array.from(await hostDevice.readFloat32(threeLayerPipelineSession.bindings.output, tinyLlamaVocabSize)),
        threeLayerOutputSentinel,
      );
      const threeLayerDecodeTokens = [3];
      const threeLayerDecodeTokensPtr = keep(alloc(4), 4);
      writeU32Array(threeLayerDecodeTokensPtr, threeLayerDecodeTokens);
      const threeLayerDecode = tokenExecuteWith(threeLayerHostExports, threeLayerPipelineSessionHandle, threeLayerDecodeTokensPtr, 1, executeOutputLogits, 0, 0);
      expectStatus(threeLayerDecode.status, ok, "host-only WebGPU tiny LLaMA three-layer block pipeline decode-after-prefill");
      if (threeLayerDecode.outputLen !== tinyLlamaVocabSize || sessionPositionWith(threeLayerHostExports, threeLayerPipelineSessionHandle) !== 3) {
        throw new Error("host-only WebGPU tiny LLaMA three-layer block pipeline decode did not emit bound logits and advance");
      }
      const expectedThreeLayerCaches = threeLayerCaches.map((cache, index) => ({
        k: Array(cache.kElements).fill(-6100 - index * 100),
        v: Array(cache.vElements).fill(-6200 - index * 100),
      }));
      const expectedThreeLayer = [];
      const runExpectedThreeLayerToken = (token, position) => {
        let inputHidden = null;
        const tokenLayers = [];
        for (let layer = 0; layer < 3; layer += 1) {
          const expected = tinyLlamaBlockProjectionStep(threeLayerModelResourceBytes, expectedThreeLayerCaches[layer], token, position, {
            ...(inputHidden === null ? {} : { inputHidden }),
            layer,
          });
          inputHidden = expected.blockHidden;
          tokenLayers.push(expected);
        }
        return tokenLayers;
      };
      for (let tokenIndex = 0; tokenIndex < threeLayerPrefillTokens.length; tokenIndex += 1) {
        expectedThreeLayer.push(runExpectedThreeLayerToken(threeLayerPrefillTokens[tokenIndex], tokenIndex));
      }
      for (let tokenIndex = 0; tokenIndex < threeLayerDecodeTokens.length; tokenIndex += 1) {
        const position = threeLayerPrefillTokens.length + tokenIndex;
        expectedThreeLayer.push(runExpectedThreeLayerToken(threeLayerDecodeTokens[tokenIndex], position));
      }
      const expectedThreeLayerLast = expectedThreeLayer.at(-1).at(-1);
      expectClose(
        Array.from(await hostDevice.readFloat32(threeLayerPipelineSession.bindings.output, tinyLlamaVocabSize)),
        expectedThreeLayerLast.logits,
      );
      for (let layer = 0; layer < 3; layer += 1) {
        for (let tokenIndex = 0; tokenIndex < expectedThreeLayer.length; tokenIndex += 1) {
          const expected = expectedThreeLayer[tokenIndex][layer];
          const offset = tokenIndex * expected.k.length * Float32Array.BYTES_PER_ELEMENT;
          expectClose(Array.from(await hostDevice.readFloat32(threeLayerCaches[layer].k, expected.k.length, offset)), expected.k);
          expectClose(Array.from(await hostDevice.readFloat32(threeLayerCaches[layer].v, expected.v.length, offset)), expected.v);
        }
      }
      const expectedThreeLayerPrefillCommandCount = expectedLlamaBlockNoOutputPrefillCommandCount(hostDevice, 3, threeLayerPrefillTokens.length);
      const expectedThreeLayerDecodeCommandCount = 3 * threeLayerDecodeTokens.length;
      const expectedThreeLayerCommandCount = expectedThreeLayerPrefillCommandCount + expectedThreeLayerDecodeCommandCount;
      const expectedThreeLayerBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? expectedThreeLayerCommandCount : 0;
      const expectedThreeLayerFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : expectedThreeLayerCommandCount;
      const threeLayerProfile = threeLayerPipelineSession.runtimeProfile();
      if (
        threeLayerProfile.callCount !== 2 ||
        threeLayerProfile.decodedCallCount !== 2 ||
        threeLayerProfile.noOutputCallCount !== 1 ||
        threeLayerProfile.logitsOutputCallCount !== 1 ||
        threeLayerProfile.boundOutputCallCount !== 1 ||
        threeLayerProfile.executorCallCount !== 2 ||
        threeLayerProfile.executorOkCount !== 2 ||
        threeLayerProfile.executorBackendDispatchCount !== expectedThreeLayerBackendDispatchCount ||
        threeLayerProfile.executorFallbackOpCount !== expectedThreeLayerFallbackOpCount ||
        threeLayerProfile.executorCommandCount !== expectedThreeLayerCommandCount ||
        threeLayerProfile.executorCommandOpCount !== expectedThreeLayerCommandCount ||
        threeLayerProfile.lastStatus !== ok ||
        threeLayerProfile.lastResultOutputLength !== tinyLlamaVocabSize
      ) {
        throw new Error(`unexpected host-only WebGPU tiny LLaMA three-layer pipeline profile: ${JSON.stringify(threeLayerProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("three-layer-pipeline", threeLayerProfile);
      const threeLayerAbiProfile = sessionRuntimeProfileWith(threeLayerHostExports, threeLayerPipelineSessionHandle);
      if (
        u64(threeLayerAbiProfile, 0) !== 2n ||
        u64(threeLayerAbiProfile, 8) !== BigInt(expectedThreeLayerBackendDispatchCount) ||
        u64(threeLayerAbiProfile, 16) !== BigInt(expectedThreeLayerFallbackOpCount) ||
        u64(threeLayerAbiProfile, 24) !== BigInt(expectedThreeLayerBackendDispatchCount) ||
        u64(threeLayerAbiProfile, 40) !== 2n ||
        u64(threeLayerAbiProfile, 48) !== 2n ||
        u64(threeLayerAbiProfile, 56) !== 0n ||
        u64(threeLayerAbiProfile, 96) !== BigInt(expectedThreeLayerCommandCount) ||
        u64(threeLayerAbiProfile, 112) !== BigInt(expectedThreeLayerCommandCount)
      ) {
        throw new Error("unexpected host-only WebGPU tiny LLaMA three-layer pipeline ABI runtime profile");
      }
      if (
        threeLayerPipelineExecutorCalls.length !== 2 ||
        threeLayerPipelineExecutorCalls[0].stageCount !== 3 ||
        threeLayerPipelineExecutorCalls[0].workspaceCount !== 2 ||
        threeLayerPipelineExecutorCalls[0].layers.length !== 3 ||
        threeLayerPipelineExecutorCalls[0].layers[0] !== 0 ||
        threeLayerPipelineExecutorCalls[0].layers[1] !== 1 ||
        threeLayerPipelineExecutorCalls[0].layers[2] !== 2 ||
        threeLayerPipelineExecutorCalls[0].commandCount !== expectedThreeLayerPrefillCommandCount ||
        threeLayerPipelineExecutorCalls[0].outputPolicy !== executeOutputNone ||
        threeLayerPipelineExecutorCalls[0].outputLength !== 0 ||
        threeLayerPipelineExecutorCalls[0].status !== ok ||
        threeLayerPipelineExecutorCalls[1].stageCount !== 3 ||
        threeLayerPipelineExecutorCalls[1].workspaceCount !== 2 ||
        threeLayerPipelineExecutorCalls[1].layers.length !== 3 ||
        threeLayerPipelineExecutorCalls[1].commandCount !== expectedThreeLayerDecodeCommandCount ||
        threeLayerPipelineExecutorCalls[1].outputPolicy !== executeOutputLogits ||
        threeLayerPipelineExecutorCalls[1].outputLength !== tinyLlamaVocabSize ||
        threeLayerPipelineExecutorCalls[1].status !== ok
      ) {
        throw new Error("host-only WebGPU tiny LLaMA three-layer block pipeline evidence mismatch");
      }
    } finally {
      expectStatus(threeLayerHostExports.zgml_session_free(threeLayerPipelineSessionHandle), ok, "host-only WebGPU tiny LLaMA three-layer Session free");
      threeLayerPipelineProgram.tokenExecutor?.destroy?.();
    }

    const widerPipelineExecutorCalls = [];
    const widerModelResourceBytes = tinyLlamaModelResourceFileBytes(widerTinyLlamaTwoLayerTensorSpecs);
    const widerModelResourceRequirements = tinyLlamaModelResourceSpecs(widerModelResourceBytes);
    const widerPipelineProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      contextLength: widerTinyLlamaShape.contextLength,
      safetensors: widerModelResourceBytes,
      allowMockFallback: true,
      executorOptions: {
        record(call) {
          recordBrowserLlamaPipelineCallEvidence(widerPipelineExecutorCalls, "wider-pipeline", call);
        },
      },
    });
    if (
      widerPipelineProgram.vocabSize !== widerTinyLlamaShape.vocabSize ||
      widerPipelineProgram.hiddenSize !== widerTinyLlamaShape.hiddenSize ||
      widerPipelineProgram.contextLength !== widerTinyLlamaShape.contextLength ||
      widerPipelineProgram.kvRequirements.layers !== widerTinyLlamaShape.layers ||
      widerPipelineProgram.kvRequirements.kBufferByteLength !== widerTinyLlamaShape.contextLength * widerTinyLlamaShape.hiddenSize * Float32Array.BYTES_PER_ELEMENT ||
      widerPipelineProgram.kvRequirements.vBufferByteLength !== widerTinyLlamaShape.contextLength * widerTinyLlamaShape.hiddenSize * Float32Array.BYTES_PER_ELEMENT ||
      widerPipelineProgram.tokenExecutor.blocks.length !== widerTinyLlamaShape.layers
    ) {
      throw new Error("host-only WebGPU wider LLaMA safetensors Program did not infer shape/layers/resources");
    }
    const widerActivation = widerPipelineProgram.createActivationBuffer({ label: "proof" });
    if (widerActivation.descriptor.byteLength !== widerTinyLlamaShape.contextLength * widerTinyLlamaShape.hiddenSize * Float32Array.BYTES_PER_ELEMENT) {
      throw new Error("host-only WebGPU wider LLaMA activation buffer was not sized from inferred shape");
    }
    const widerPipelineSession = widerPipelineProgram.bindHostResources();
    const widerPipelineSessionHandle = widerPipelineSession.handle;
    const widerHostExports = hostRuntime.wrapExports(exportsRef);
    try {
      if (
        widerPipelineSession.vocabSize !== widerTinyLlamaShape.vocabSize ||
        widerPipelineSession.contextLength !== widerTinyLlamaShape.contextLength ||
        widerPipelineSession.modelResources.length !== widerModelResourceRequirements.length ||
        widerPipelineSession.bindings.output.byteLength !== widerTinyLlamaShape.vocabSize * Float32Array.BYTES_PER_ELEMENT
      ) {
        throw new Error("host-only WebGPU wider LLaMA byte-backed bind did not preserve inferred shape");
      }
      expectHostResourceTableRoles(
        widerPipelineSession.table,
        ["llama.output", "llama.k.0", "llama.v.0", "llama.k.1", "llama.v.1"],
        "host-only WebGPU wider LLaMA block pipeline",
      );
      const widerCaches = widerPipelineSession.bindings.kvCache.map((layer, index) => {
        const kElements = Math.floor(layer.k.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const vElements = Math.floor(layer.v.byteLength / Float32Array.BYTES_PER_ELEMENT);
        if (
          kElements !== widerTinyLlamaShape.contextLength * widerTinyLlamaShape.hiddenSize ||
          vElements !== widerTinyLlamaShape.contextLength * widerTinyLlamaShape.hiddenSize
        ) {
          throw new Error("host-only WebGPU wider LLaMA K/V cache was not sized from inferred hidden/context");
        }
        hostDevice.writeFloat32(layer.k, Array(kElements).fill(-7100 - index * 100));
        hostDevice.writeFloat32(layer.v, Array(vElements).fill(-7200 - index * 100));
        return { k: layer.k, kElements, v: layer.v, vElements };
      });
      const widerOutputSentinel = Array(widerTinyLlamaShape.vocabSize).fill(-7160);
      hostDevice.writeFloat32(widerPipelineSession.bindings.output, widerOutputSentinel);
      const widerPrefillTokens = [1];
      const widerPrefillTokensPtr = keep(alloc(4), 4);
      writeU32Array(widerPrefillTokensPtr, widerPrefillTokens);
      const widerPrefill = tokenExecuteWith(widerHostExports, widerPipelineSessionHandle, widerPrefillTokensPtr, 1, executeOutputNone, 0, 0);
      expectStatus(widerPrefill.status, ok, "host-only WebGPU wider LLaMA block pipeline no-output prefill");
      if (widerPrefill.outputLen !== 0 || sessionPositionWith(widerHostExports, widerPipelineSessionHandle) !== 1) {
        throw new Error("host-only WebGPU wider LLaMA block pipeline prefill did not advance without output");
      }
      expectClose(
        Array.from(await hostDevice.readFloat32(widerPipelineSession.bindings.output, widerTinyLlamaShape.vocabSize)),
        widerOutputSentinel,
      );
      const widerDecodeTokens = [4];
      const widerDecodeTokensPtr = keep(alloc(4), 4);
      writeU32Array(widerDecodeTokensPtr, widerDecodeTokens);
      const widerDecode = tokenExecuteWith(widerHostExports, widerPipelineSessionHandle, widerDecodeTokensPtr, 1, executeOutputLogits, 0, 0);
      expectStatus(widerDecode.status, ok, "host-only WebGPU wider LLaMA block pipeline decode-after-prefill");
      if (widerDecode.outputLen !== widerTinyLlamaShape.vocabSize || sessionPositionWith(widerHostExports, widerPipelineSessionHandle) !== 2) {
        throw new Error("host-only WebGPU wider LLaMA block pipeline decode did not emit bound logits and advance");
      }
      const expectedWiderCaches = widerCaches.map((cache, index) => ({
        k: Array(cache.kElements).fill(-7100 - index * 100),
        v: Array(cache.vElements).fill(-7200 - index * 100),
      }));
      const expectedWider = [];
      const runExpectedWiderToken = (token, position) => {
        let inputHidden = null;
        const tokenLayers = [];
        for (let layer = 0; layer < widerTinyLlamaShape.layers; layer += 1) {
          const expected = tinyLlamaBlockProjectionStep(widerModelResourceBytes, expectedWiderCaches[layer], token, position, {
            ...(inputHidden === null ? {} : { inputHidden }),
            layer,
          });
          inputHidden = expected.blockHidden;
          tokenLayers.push(expected);
        }
        return tokenLayers;
      };
      for (let tokenIndex = 0; tokenIndex < widerPrefillTokens.length; tokenIndex += 1) {
        expectedWider.push(runExpectedWiderToken(widerPrefillTokens[tokenIndex], tokenIndex));
      }
      for (let tokenIndex = 0; tokenIndex < widerDecodeTokens.length; tokenIndex += 1) {
        const position = widerPrefillTokens.length + tokenIndex;
        expectedWider.push(runExpectedWiderToken(widerDecodeTokens[tokenIndex], position));
      }
      const expectedWiderLast = expectedWider.at(-1).at(-1);
      expectClose(
        Array.from(await hostDevice.readFloat32(widerPipelineSession.bindings.output, widerTinyLlamaShape.vocabSize)),
        expectedWiderLast.logits,
      );
      for (let layer = 0; layer < widerTinyLlamaShape.layers; layer += 1) {
        for (let tokenIndex = 0; tokenIndex < expectedWider.length; tokenIndex += 1) {
          const expected = expectedWider[tokenIndex][layer];
          const offset = tokenIndex * expected.k.length * Float32Array.BYTES_PER_ELEMENT;
          expectClose(Array.from(await hostDevice.readFloat32(widerCaches[layer].k, expected.k.length, offset)), expected.k);
          expectClose(Array.from(await hostDevice.readFloat32(widerCaches[layer].v, expected.v.length, offset)), expected.v);
        }
      }
      const expectedWiderBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? 4 : 0;
      const expectedWiderFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : 4;
      const widerProfile = widerPipelineSession.runtimeProfile();
      if (
        widerProfile.callCount !== 2 ||
        widerProfile.decodedCallCount !== 2 ||
        widerProfile.noOutputCallCount !== 1 ||
        widerProfile.logitsOutputCallCount !== 1 ||
        widerProfile.boundOutputCallCount !== 1 ||
        widerProfile.executorCallCount !== 2 ||
        widerProfile.executorOkCount !== 2 ||
        widerProfile.executorBackendDispatchCount !== expectedWiderBackendDispatchCount ||
        widerProfile.executorFallbackOpCount !== expectedWiderFallbackOpCount ||
        widerProfile.executorCommandCount !== 4 ||
        widerProfile.executorCommandOpCount !== 4 ||
        widerProfile.lastStatus !== ok ||
        widerProfile.lastResultOutputLength !== widerTinyLlamaShape.vocabSize
      ) {
        throw new Error(`unexpected host-only WebGPU wider LLaMA pipeline profile: ${JSON.stringify(widerProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("wider-pipeline", widerProfile);
      const widerAbiProfile = sessionRuntimeProfileWith(widerHostExports, widerPipelineSessionHandle);
      if (
        u64(widerAbiProfile, 0) !== 2n ||
        u64(widerAbiProfile, 8) !== BigInt(expectedWiderBackendDispatchCount) ||
        u64(widerAbiProfile, 16) !== BigInt(expectedWiderFallbackOpCount) ||
        u64(widerAbiProfile, 24) !== BigInt(expectedWiderBackendDispatchCount) ||
        u64(widerAbiProfile, 40) !== 2n ||
        u64(widerAbiProfile, 48) !== 2n ||
        u64(widerAbiProfile, 56) !== 0n ||
        u64(widerAbiProfile, 96) !== 4n ||
        u64(widerAbiProfile, 112) !== 4n
      ) {
        throw new Error("unexpected host-only WebGPU wider LLaMA pipeline ABI runtime profile");
      }
      if (
        widerPipelineExecutorCalls.length !== 2 ||
        widerPipelineExecutorCalls[0].stageCount !== widerTinyLlamaShape.layers ||
        widerPipelineExecutorCalls[0].workspaceCount !== 1 ||
        widerPipelineExecutorCalls[0].layers.length !== widerTinyLlamaShape.layers ||
        widerPipelineExecutorCalls[0].layers[0] !== 0 ||
        widerPipelineExecutorCalls[0].layers[1] !== 1 ||
        widerPipelineExecutorCalls[0].commandCount !== 2 ||
        widerPipelineExecutorCalls[0].outputPolicy !== executeOutputNone ||
        widerPipelineExecutorCalls[0].outputLength !== 0 ||
        widerPipelineExecutorCalls[0].hiddenSize !== widerTinyLlamaShape.hiddenSize ||
        widerPipelineExecutorCalls[0].status !== ok ||
        widerPipelineExecutorCalls[1].stageCount !== widerTinyLlamaShape.layers ||
        widerPipelineExecutorCalls[1].workspaceCount !== 1 ||
        widerPipelineExecutorCalls[1].layers.length !== widerTinyLlamaShape.layers ||
        widerPipelineExecutorCalls[1].commandCount !== 2 ||
        widerPipelineExecutorCalls[1].outputPolicy !== executeOutputLogits ||
        widerPipelineExecutorCalls[1].outputLength !== widerTinyLlamaShape.vocabSize ||
        widerPipelineExecutorCalls[1].hiddenSize !== widerTinyLlamaShape.hiddenSize ||
        widerPipelineExecutorCalls[1].status !== ok
      ) {
        throw new Error("host-only WebGPU wider LLaMA block pipeline evidence mismatch");
      }
    } finally {
      expectStatus(widerHostExports.zgml_session_free(widerPipelineSessionHandle), ok, "host-only WebGPU wider LLaMA Session free");
      widerPipelineProgram.tokenExecutor?.destroy?.();
    }

    const realisticHeadPipelineExecutorCalls = [];
    const realisticHeadConfig = mistralSlidingWindowProofConfig(realisticHeadTinyLlamaShape, {
      max_position_embeddings: realisticHeadTinyLlamaShape.contextLength,
    });
    const realisticHeadModelResourceBytes = tinyLlamaModelResourceFileBytes(realisticHeadTinyLlamaTensorSpecs, {
      metadata: {
        config: JSON.stringify(realisticHeadConfig),
        intermediate_size: String(realisticHeadTinyLlamaShape.ffnSize),
        num_attention_heads: "1",
        num_key_value_heads: "1",
      },
    });
    const realisticHeadModelResourceRequirements = tinyLlamaModelResourceSpecs(realisticHeadModelResourceBytes);
    const realisticHeadPipelineProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      config: realisticHeadConfig,
      contextLength: realisticHeadTinyLlamaShape.contextLength,
      safetensors: realisticHeadModelResourceBytes,
      allowMockFallback: true,
      executorOptions: {
        record(call) {
          recordBrowserLlamaPipelineCallEvidence(realisticHeadPipelineExecutorCalls, "realistic-head-pipeline", call);
        },
      },
    });
    if (
      realisticHeadPipelineProgram.vocabSize !== realisticHeadTinyLlamaShape.vocabSize ||
      realisticHeadPipelineProgram.hiddenSize !== realisticHeadTinyLlamaShape.hiddenSize ||
      realisticHeadPipelineProgram.ffnSize !== realisticHeadTinyLlamaShape.ffnSize ||
      realisticHeadPipelineProgram.contextLength !== realisticHeadTinyLlamaShape.contextLength ||
      realisticHeadPipelineProgram.kvRequirements.layers !== realisticHeadTinyLlamaShape.layers ||
      realisticHeadPipelineProgram.kvRequirements.kBufferByteLength !== realisticHeadTinyLlamaShape.contextLength * realisticHeadTinyLlamaShape.kvSize * Float32Array.BYTES_PER_ELEMENT ||
      realisticHeadPipelineProgram.kvRequirements.vBufferByteLength !== realisticHeadTinyLlamaShape.contextLength * realisticHeadTinyLlamaShape.kvSize * Float32Array.BYTES_PER_ELEMENT ||
      realisticHeadPipelineProgram.tokenExecutor.blocks.length !== realisticHeadTinyLlamaShape.layers ||
      realisticHeadPipelineProgram.tokenExecutor.blocks[0].executor.attentionHeadSize !== realisticHeadTinyLlamaShape.attentionHeadSize ||
      realisticHeadPipelineProgram.tokenExecutor.blocks[0].executor.ropeBase !== realisticHeadConfig.rope_theta ||
      realisticHeadPipelineProgram.tokenExecutor.blocks[0].executor.slidingWindow !== realisticHeadConfig.sliding_window
    ) {
      throw new Error("host-only WebGPU realistic-head sliding-window LLaMA safetensors Program did not infer d_head=128 shape/resources");
    }
    const realisticHeadActivation = realisticHeadPipelineProgram.createActivationBuffer({ label: "realistic-head-proof" });
    if (realisticHeadActivation.descriptor.byteLength !== realisticHeadTinyLlamaShape.contextLength * realisticHeadTinyLlamaShape.hiddenSize * Float32Array.BYTES_PER_ELEMENT) {
      throw new Error("host-only WebGPU realistic-head LLaMA activation buffer was not sized from inferred shape");
    }
    realisticHeadActivation.resource?.destroy?.();
    const realisticHeadPipelineSession = realisticHeadPipelineProgram.bindHostResources();
    const realisticHeadPipelineSessionHandle = realisticHeadPipelineSession.handle;
    const realisticHeadHostExports = hostRuntime.wrapExports(exportsRef);
    try {
      if (
        realisticHeadPipelineSession.vocabSize !== realisticHeadTinyLlamaShape.vocabSize ||
        realisticHeadPipelineSession.contextLength !== realisticHeadTinyLlamaShape.contextLength ||
        realisticHeadPipelineSession.modelResources.length !== realisticHeadModelResourceRequirements.length ||
        realisticHeadPipelineSession.bindings.output.byteLength !== realisticHeadTinyLlamaShape.vocabSize * Float32Array.BYTES_PER_ELEMENT
      ) {
        throw new Error("host-only WebGPU realistic-head LLaMA byte-backed bind did not preserve inferred shape");
      }
      expectHostResourceTableRoles(
        realisticHeadPipelineSession.table,
        ["llama.output", "llama.k.0", "llama.v.0"],
        "host-only WebGPU realistic-head LLaMA block pipeline",
      );
      const realisticHeadCaches = realisticHeadPipelineSession.bindings.kvCache.map((layer, index) => {
        const kElements = Math.floor(layer.k.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const vElements = Math.floor(layer.v.byteLength / Float32Array.BYTES_PER_ELEMENT);
        if (
          kElements !== realisticHeadTinyLlamaShape.contextLength * realisticHeadTinyLlamaShape.kvSize ||
          vElements !== realisticHeadTinyLlamaShape.contextLength * realisticHeadTinyLlamaShape.kvSize
        ) {
          throw new Error("host-only WebGPU realistic-head LLaMA K/V cache was not sized from inferred d_head");
        }
        hostDevice.writeFloat32(layer.k, Array(kElements).fill(-9100 - index * 100));
        hostDevice.writeFloat32(layer.v, Array(vElements).fill(-9200 - index * 100));
        return { k: layer.k, kElements, v: layer.v, vElements };
      });
      const realisticHeadOutputSentinel = Array(realisticHeadTinyLlamaShape.vocabSize).fill(-9160);
      hostDevice.writeFloat32(realisticHeadPipelineSession.bindings.output, realisticHeadOutputSentinel);
      const realisticHeadPrefillTokens = Array.from({ length: 66 }, (_unused, index) => (index + 1) % realisticHeadTinyLlamaShape.vocabSize);
      const realisticHeadPrefillTokensPtr = keep(
        alloc(realisticHeadPrefillTokens.length * Uint32Array.BYTES_PER_ELEMENT),
        realisticHeadPrefillTokens.length * Uint32Array.BYTES_PER_ELEMENT,
      );
      writeU32Array(realisticHeadPrefillTokensPtr, realisticHeadPrefillTokens);
      const realisticHeadPrefill = tokenExecuteWith(realisticHeadHostExports, realisticHeadPipelineSessionHandle, realisticHeadPrefillTokensPtr, realisticHeadPrefillTokens.length, executeOutputNone, 0, 0);
      expectStatus(realisticHeadPrefill.status, ok, "host-only WebGPU realistic-head sliding-window LLaMA block pipeline no-output prefill");
      if (realisticHeadPrefill.outputLen !== 0 || sessionPositionWith(realisticHeadHostExports, realisticHeadPipelineSessionHandle) !== realisticHeadPrefillTokens.length) {
        throw new Error("host-only WebGPU realistic-head sliding-window LLaMA block pipeline prefill did not advance without output");
      }
      expectClose(
        Array.from(await hostDevice.readFloat32(realisticHeadPipelineSession.bindings.output, realisticHeadTinyLlamaShape.vocabSize)),
        realisticHeadOutputSentinel,
      );
      const realisticHeadDecodeTokens = [2];
      const realisticHeadDecodeTokensPtr = keep(alloc(4), 4);
      writeU32Array(realisticHeadDecodeTokensPtr, realisticHeadDecodeTokens);
      const realisticHeadDecode = tokenExecuteWith(realisticHeadHostExports, realisticHeadPipelineSessionHandle, realisticHeadDecodeTokensPtr, 1, executeOutputLogits, 0, 0);
      expectStatus(realisticHeadDecode.status, ok, "host-only WebGPU realistic-head sliding-window LLaMA block pipeline decode-after-prefill");
      if (
        realisticHeadDecode.outputLen !== realisticHeadTinyLlamaShape.vocabSize ||
        sessionPositionWith(realisticHeadHostExports, realisticHeadPipelineSessionHandle) !== realisticHeadPrefillTokens.length + realisticHeadDecodeTokens.length
      ) {
        throw new Error("host-only WebGPU realistic-head sliding-window LLaMA block pipeline decode did not emit bound logits and advance");
      }
      const expectedRealisticHeadCaches = realisticHeadCaches.map((cache, index) => ({
        k: Array(cache.kElements).fill(-9100 - index * 100),
        v: Array(cache.vElements).fill(-9200 - index * 100),
      }));
      const expectedRealisticHead = [];
      const runExpectedRealisticHeadToken = (token, position) => {
        const expected = tinyLlamaBlockProjectionStep(realisticHeadModelResourceBytes, expectedRealisticHeadCaches[0], token, position, {
          attentionHeadSize: realisticHeadTinyLlamaShape.attentionHeadSize,
          epsilon: realisticHeadConfig.rms_norm_eps,
          layer: 0,
          ropeBase: realisticHeadConfig.rope_theta,
          slidingWindow: realisticHeadConfig.sliding_window,
        });
        return [expected];
      };
      for (let tokenIndex = 0; tokenIndex < realisticHeadPrefillTokens.length; tokenIndex += 1) {
        expectedRealisticHead.push(runExpectedRealisticHeadToken(realisticHeadPrefillTokens[tokenIndex], tokenIndex));
      }
      for (let tokenIndex = 0; tokenIndex < realisticHeadDecodeTokens.length; tokenIndex += 1) {
        const position = realisticHeadPrefillTokens.length + tokenIndex;
        expectedRealisticHead.push(runExpectedRealisticHeadToken(realisticHeadDecodeTokens[tokenIndex], position));
      }
      const expectedRealisticHeadLast = expectedRealisticHead.at(-1).at(-1);
      expectClose(
        Array.from(await hostDevice.readFloat32(realisticHeadPipelineSession.bindings.output, realisticHeadTinyLlamaShape.vocabSize)),
        expectedRealisticHeadLast.logits,
        1e-3,
      );
      for (let tokenIndex = 0; tokenIndex < expectedRealisticHead.length; tokenIndex += 1) {
        const expected = expectedRealisticHead[tokenIndex][0];
        const offset = tokenIndex * expected.k.length * Float32Array.BYTES_PER_ELEMENT;
        expectClose(Array.from(await hostDevice.readFloat32(realisticHeadCaches[0].k, expected.k.length, offset)), expected.k, 1e-3);
        expectClose(Array.from(await hostDevice.readFloat32(realisticHeadCaches[0].v, expected.v.length, offset)), expected.v, 1e-3);
      }
      const expectedRealisticHeadPrefillCommandCount = expectedLlamaBlockPipelineCommandCount(
        hostDevice,
        realisticHeadTinyLlamaShape.layers,
        realisticHeadPrefillTokens.length,
        executeOutputNone,
        {
          contextLength: realisticHeadTinyLlamaShape.contextLength,
          hiddenSize: realisticHeadTinyLlamaShape.hiddenSize,
          slidingWindow: realisticHeadConfig.sliding_window,
        },
      );
      const expectedRealisticHeadPrefillDispatchFamilies = expectedLlamaBlockPipelineDispatchFamilies(
        hostDevice,
        realisticHeadTinyLlamaShape.layers,
        realisticHeadPrefillTokens.length,
        executeOutputNone,
        {
          contextLength: realisticHeadTinyLlamaShape.contextLength,
          hiddenSize: realisticHeadTinyLlamaShape.hiddenSize,
          slidingWindow: realisticHeadConfig.sliding_window,
        },
      );
      const expectedRealisticHeadDecodeCommandCount = expectedLlamaBlockPipelineCommandCount(
        hostDevice,
        realisticHeadTinyLlamaShape.layers,
        realisticHeadDecodeTokens.length,
        executeOutputLogits,
        {
          contextLength: realisticHeadTinyLlamaShape.contextLength,
          hiddenSize: realisticHeadTinyLlamaShape.hiddenSize,
          slidingWindow: realisticHeadConfig.sliding_window,
          splitTerminalLogits: true,
          startPosition: realisticHeadPrefillTokens.length,
        },
      );
      const expectedRealisticHeadDecodeDispatchFamilies = expectedLlamaBlockPipelineDispatchFamilies(
        hostDevice,
        realisticHeadTinyLlamaShape.layers,
        realisticHeadDecodeTokens.length,
        executeOutputLogits,
        {
          contextLength: realisticHeadTinyLlamaShape.contextLength,
          hiddenSize: realisticHeadTinyLlamaShape.hiddenSize,
          slidingWindow: realisticHeadConfig.sliding_window,
          splitTerminalLogits: true,
          startPosition: realisticHeadPrefillTokens.length,
        },
      );
      const expectedRealisticHeadCommandCount = expectedRealisticHeadPrefillCommandCount + expectedRealisticHeadDecodeCommandCount;
      const expectedRealisticHeadBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? expectedRealisticHeadCommandCount : 0;
      const expectedRealisticHeadFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : expectedRealisticHeadCommandCount;
      const expectedRealisticHeadGenericDispatchCount =
        expectedRealisticHeadPrefillDispatchFamilies.generic + expectedRealisticHeadDecodeDispatchFamilies.generic;
      const expectedRealisticHeadScalarDispatchCount =
        expectedRealisticHeadPrefillDispatchFamilies.scalar + expectedRealisticHeadDecodeDispatchFamilies.scalar;
      const expectedRealisticHeadWindowDispatchCount =
        expectedRealisticHeadPrefillDispatchFamilies.window + expectedRealisticHeadDecodeDispatchFamilies.window;
      const realisticHeadProfile = realisticHeadPipelineSession.runtimeProfile();
      if (
        realisticHeadProfile.callCount !== 2 ||
        realisticHeadProfile.decodedCallCount !== 2 ||
        realisticHeadProfile.noOutputCallCount !== 1 ||
        realisticHeadProfile.logitsOutputCallCount !== 1 ||
        realisticHeadProfile.boundOutputCallCount !== 1 ||
        realisticHeadProfile.executorCallCount !== 2 ||
        realisticHeadProfile.executorOkCount !== 2 ||
        realisticHeadProfile.executorBackendDispatchCount !== expectedRealisticHeadBackendDispatchCount ||
        realisticHeadProfile.executorFallbackOpCount !== expectedRealisticHeadFallbackOpCount ||
        realisticHeadProfile.executorCommandCount !== expectedRealisticHeadCommandCount ||
        realisticHeadProfile.executorCommandOpCount !== expectedRealisticHeadCommandCount ||
        realisticHeadProfile.executorGenericDispatchCount !== expectedRealisticHeadGenericDispatchCount ||
        realisticHeadProfile.executorScalarDispatchCount !== expectedRealisticHeadScalarDispatchCount ||
        realisticHeadProfile.executorWindowDispatchCount !== expectedRealisticHeadWindowDispatchCount ||
        realisticHeadProfile.lastStatus !== ok ||
        realisticHeadProfile.lastResultOutputLength !== realisticHeadTinyLlamaShape.vocabSize
      ) {
        throw new Error(`unexpected host-only WebGPU realistic-head LLaMA pipeline profile: ${JSON.stringify(realisticHeadProfile)}`);
      }
      recordBrowserLlamaProfileEvidence("realistic-head-pipeline", realisticHeadProfile);
      const realisticHeadAbiProfile = sessionRuntimeProfileWith(realisticHeadHostExports, realisticHeadPipelineSessionHandle);
      if (
        u64(realisticHeadAbiProfile, 0) !== 2n ||
        u64(realisticHeadAbiProfile, 8) !== BigInt(expectedRealisticHeadBackendDispatchCount) ||
        u64(realisticHeadAbiProfile, 16) !== BigInt(expectedRealisticHeadFallbackOpCount) ||
        u64(realisticHeadAbiProfile, 24) !== BigInt(expectedRealisticHeadBackendDispatchCount) ||
        u64(realisticHeadAbiProfile, 40) !== 2n ||
        u64(realisticHeadAbiProfile, 48) !== 2n ||
        u64(realisticHeadAbiProfile, 56) !== 0n ||
        u64(realisticHeadAbiProfile, 96) !== BigInt(expectedRealisticHeadCommandCount) ||
        u64(realisticHeadAbiProfile, 112) !== BigInt(expectedRealisticHeadCommandCount)
      ) {
        throw new Error("unexpected host-only WebGPU realistic-head sliding-window LLaMA pipeline ABI runtime profile");
      }
      if (
        realisticHeadPipelineExecutorCalls.length !== 2 ||
        realisticHeadPipelineExecutorCalls[0].stageCount !== realisticHeadTinyLlamaShape.layers ||
        realisticHeadPipelineExecutorCalls[0].workspaceCount !== 0 ||
        realisticHeadPipelineExecutorCalls[0].layers.length !== realisticHeadTinyLlamaShape.layers ||
        realisticHeadPipelineExecutorCalls[0].layers[0] !== 0 ||
        realisticHeadPipelineExecutorCalls[0].commandCount !== expectedRealisticHeadPrefillCommandCount ||
        realisticHeadPipelineExecutorCalls[0].genericDispatchCount !== expectedRealisticHeadPrefillDispatchFamilies.generic ||
        realisticHeadPipelineExecutorCalls[0].outputPolicy !== executeOutputNone ||
        realisticHeadPipelineExecutorCalls[0].outputLength !== 0 ||
        realisticHeadPipelineExecutorCalls[0].scalarDispatchCount !== expectedRealisticHeadPrefillDispatchFamilies.scalar ||
        realisticHeadPipelineExecutorCalls[0].hiddenSize !== realisticHeadTinyLlamaShape.hiddenSize ||
        realisticHeadPipelineExecutorCalls[0].ffnSize !== realisticHeadTinyLlamaShape.ffnSize ||
        realisticHeadPipelineExecutorCalls[0].status !== ok ||
        realisticHeadPipelineExecutorCalls[0].windowDispatchCount !== expectedRealisticHeadPrefillDispatchFamilies.window ||
        realisticHeadPipelineExecutorCalls[1].stageCount !== realisticHeadTinyLlamaShape.layers ||
        realisticHeadPipelineExecutorCalls[1].workspaceCount !== 0 ||
        realisticHeadPipelineExecutorCalls[1].layers.length !== realisticHeadTinyLlamaShape.layers ||
        realisticHeadPipelineExecutorCalls[1].commandCount !== expectedRealisticHeadDecodeCommandCount ||
        realisticHeadPipelineExecutorCalls[1].genericDispatchCount !== expectedRealisticHeadDecodeDispatchFamilies.generic ||
        realisticHeadPipelineExecutorCalls[1].outputPolicy !== executeOutputLogits ||
        realisticHeadPipelineExecutorCalls[1].outputLength !== realisticHeadTinyLlamaShape.vocabSize ||
        realisticHeadPipelineExecutorCalls[1].scalarDispatchCount !== expectedRealisticHeadDecodeDispatchFamilies.scalar ||
        realisticHeadPipelineExecutorCalls[1].hiddenSize !== realisticHeadTinyLlamaShape.hiddenSize ||
        realisticHeadPipelineExecutorCalls[1].ffnSize !== realisticHeadTinyLlamaShape.ffnSize ||
        realisticHeadPipelineExecutorCalls[1].status !== ok ||
        realisticHeadPipelineExecutorCalls[1].windowDispatchCount !== expectedRealisticHeadDecodeDispatchFamilies.window
      ) {
        throw new Error("host-only WebGPU realistic-head sliding-window LLaMA block pipeline evidence mismatch");
      }
    } finally {
      expectStatus(realisticHeadHostExports.zgml_session_free(realisticHeadPipelineSessionHandle), ok, "host-only WebGPU realistic-head LLaMA Session free");
      realisticHeadPipelineProgram.tokenExecutor?.destroy?.();
    }
    }

    const packedTinyLlamaProofs = [
      {
        label: "metadata-mha-pipeline",
        metadata: {
          num_attention_heads: "4",
          num_key_value_heads: "4",
          intermediate_size: "8",
          pretraining_tp: "2",
          rms_norm_eps: "0.00002",
          rope_theta: "11000",
        },
        epsilon: 0.00002,
        ropeBase: 11000,
        shape: mhaTinyLlamaShape,
        specs: mhaTinyLlamaTwoLayerTensorSpecs,
      },
      {
        label: "gguf-metadata-gqa-pipeline",
        metadata: {
          "llama.attention.head_count": "4",
          "llama.attention.head_count_kv": "2",
          "llama.attention.layer_norm_rms_epsilon": "0.000086",
          "llama.block_count": String(groupedTinyLlamaShape.layers),
          "llama.context_length": String(groupedTinyLlamaShape.contextLength),
          "llama.embedding_length": String(groupedTinyLlamaShape.hiddenSize),
          "llama.feed_forward_length": String(groupedTinyLlamaShape.ffnSize),
          "llama.rope.freq_base": "18100",
          "llama.vocab_size": String(groupedTinyLlamaShape.vocabSize),
        },
        epsilon: 0.000086,
        omitContextLengthOption: true,
        ropeBase: 18100,
        shape: groupedTinyLlamaShape,
        specs: groupedTinyLlamaTwoLayerTensorSpecs,
      },
      {
        label: "qwen2-gguf-metadata-gqa-pipeline",
        metadata: {
          "general.architecture": "qwen2",
          "qwen2.attention.head_count": "4",
          "qwen2.attention.head_count_kv": "2",
          "qwen2.attention.key_length": String(groupedTinyLlamaShape.attentionHeadSize),
          "qwen2.attention.layer_norm_rms_epsilon": "0.000087",
          "qwen2.attention.value_length": String(groupedTinyLlamaShape.attentionHeadSize),
          "qwen2.block_count": String(groupedTinyLlamaShape.layers),
          "qwen2.context_length": String(groupedTinyLlamaShape.contextLength),
          "qwen2.embedding_length": String(groupedTinyLlamaShape.hiddenSize),
          "qwen2.feed_forward_length": String(groupedTinyLlamaShape.ffnSize),
          "qwen2.rope.dimension_count": String(groupedTinyLlamaShape.attentionHeadSize),
          "qwen2.rope.freq_base": "18200",
          "qwen2.vocab_size": String(groupedTinyLlamaShape.vocabSize),
          "tokenizer.ggml.scores": JSON.stringify(Array.from({ length: groupedTinyLlamaShape.vocabSize }, (_unused, token) => -0.01 * token)),
          "tokenizer.ggml.token_type": JSON.stringify(Array.from({ length: groupedTinyLlamaShape.vocabSize }, () => 1)),
          "tokenizer.ggml.tokens": JSON.stringify(Array.from({ length: groupedTinyLlamaShape.vocabSize }, (_unused, token) => `qwen2-token-${token}`)),
        },
        epsilon: 0.000087,
        omitContextLengthOption: true,
        ropeBase: 18200,
        shape: groupedTinyLlamaShape,
        specs: groupedTinyLlamaTwoLayerTensorSpecs,
      },
      {
        label: "config-gqa-pipeline",
        config: {
          architectures: ["LlamaForCausalLM"],
          attention_dropout: 0,
          attention_bias: false,
          bos_token_id: 1,
          eos_token_id: [2, 3],
          hidden_size: groupedTinyLlamaShape.hiddenSize,
          hidden_act: "silu",
          hidden_dropout: 0,
          intermediate_size: groupedTinyLlamaShape.ffnSize,
          max_position_embeddings: 16,
          mlp_bias: false,
          model_type: "llama",
          num_attention_heads: 4,
          num_hidden_layers: groupedTinyLlamaShape.layers,
          num_key_value_heads: 2,
          pad_token_id: 0,
          pretraining_tp: 2,
          partial_rotary_factor: 1,
          rms_norm_eps: 0.00003,
          rope_scaling: { rope_type: "default" },
          rope_theta: 12000,
          sliding_window: null,
          torch_dtype: "float32",
          use_sliding_window: false,
          use_cache: true,
          vocab_size: groupedTinyLlamaShape.vocabSize,
        },
        epsilon: 0.00003,
        ropeBase: 12000,
        shape: groupedTinyLlamaShape,
        specs: groupedTinyLlamaTwoLayerTensorSpecs,
      },
      {
        label: "long-context-gqa-pipeline",
        config: {
          architectures: ["LlamaForCausalLM"],
          attention_dropout: 0,
          attention_bias: false,
          hidden_size: longContextGqaTinyLlamaShape.hiddenSize,
          hidden_act: "silu",
          hidden_dropout: 0,
          intermediate_size: longContextGqaTinyLlamaShape.ffnSize,
          max_position_embeddings: longContextGqaTinyLlamaShape.contextLength,
          mlp_bias: false,
          model_type: "llama",
          num_attention_heads: 4,
          num_hidden_layers: longContextGqaTinyLlamaShape.layers,
          num_key_value_heads: 2,
          pretraining_tp: 3,
          partial_rotary_factor: 1,
          rms_norm_eps: 0.000031,
          rope_scaling: null,
          rope_theta: 12100,
          sliding_window: null,
          torch_dtype: "float32",
          use_sliding_window: false,
          use_cache: true,
          vocab_size: longContextGqaTinyLlamaShape.vocabSize,
        },
        decodeTokens: [5, 4, 3],
        epsilon: 0.000031,
        metadata: {
          max_position_embeddings: String(longContextGqaTinyLlamaShape.contextLength),
        },
        prefillTokens: [
          0, 1, 2, 3, 4, 5, 1, 2,
          3, 4, 5, 0, 1, 2, 3, 4,
          5,
        ],
        ropeBase: 12100,
        shape: longContextGqaTinyLlamaShape,
        specs: longContextGqaTinyLlamaTwoLayerTensorSpecs,
      },
      {
        label: "config-json-mqa-pipeline",
        config: JSON.stringify({
          architectures: ["LlamaForCausalLM"],
          attention_dropout: 0,
          attention_bias: false,
          bos_token_id: 1,
          eos_token_id: [2, 3],
          hidden_size: mqaTinyLlamaShape.hiddenSize,
          hidden_act: "swish",
          hidden_dropout: 0,
          intermediate_size: mqaTinyLlamaShape.ffnSize,
          max_position_embeddings: 16,
          mlp_bias: false,
          model_type: "llama",
          num_attention_heads: 4,
          num_hidden_layers: mqaTinyLlamaShape.layers,
          num_key_value_heads: 1,
          pad_token_id: 0,
          pretraining_tp: 1,
          partial_rotary_factor: 1,
          rms_norm_eps: 0.00004,
          rope_scaling: { type: "linear", factor: 2 },
          rope_theta: 13000,
          sliding_window: null,
          torch_dtype: "torch.float32",
          use_sliding_window: false,
          use_cache: true,
          vocab_size: mqaTinyLlamaShape.vocabSize,
        }),
        epsilon: 0.00004,
        ropeBase: 13000,
        ropeScale: 2,
        shape: mqaTinyLlamaShape,
        specs: mqaTinyLlamaTwoLayerTensorSpecs,
      },
      {
        label: "llama3-rope-gqa-pipeline",
        config: {
          architectures: ["LlamaForCausalLM"],
          attention_dropout: 0,
          attention_bias: false,
          hidden_size: groupedTinyLlamaShape.hiddenSize,
          hidden_act: "silu",
          hidden_dropout: 0,
          intermediate_size: groupedTinyLlamaShape.ffnSize,
          max_position_embeddings: 16,
          mlp_bias: false,
          model_type: "llama",
          num_attention_heads: 4,
          num_hidden_layers: groupedTinyLlamaShape.layers,
          num_key_value_heads: 2,
          pretraining_tp: 1,
          partial_rotary_factor: 1,
          rms_norm_eps: 0.000045,
          rope_scaling: {
            factor: 8,
            low_freq_factor: 1,
            high_freq_factor: 4,
            original_max_position_embeddings: 16,
            rope_type: "llama3",
          },
          rope_theta: 13500,
          sliding_window: null,
          torch_dtype: "float32",
          use_sliding_window: false,
          use_cache: true,
          vocab_size: groupedTinyLlamaShape.vocabSize,
        },
        epsilon: 0.000045,
        ropeBase: 13500,
        ropeHighFreqFactor: 4,
        ropeKind: "llama3",
        ropeLowFreqFactor: 1,
        ropeOriginalContextLength: 16,
        ropeScale: 8,
        shape: groupedTinyLlamaShape,
        specs: groupedTinyLlamaTwoLayerTensorSpecs,
      },
      {
        label: "metadata-config-gqa-pipeline",
        metadata: {
          config: JSON.stringify({
            architectures: ["LlamaForCausalLM"],
            attention_dropout: 0,
            attention_bias: false,
            bos_token_id: 1,
            eos_token_id: [2, 3],
            hidden_size: groupedTinyLlamaShape.hiddenSize,
            hidden_act: "silu",
            hidden_dropout_prob: 0,
            intermediate_size: groupedTinyLlamaShape.ffnSize,
            max_position_embeddings: 16,
            mlp_bias: false,
            model_type: "llama",
            num_attention_heads: 4,
            num_hidden_layers: groupedTinyLlamaShape.layers,
            num_key_value_heads: 2,
            pad_token_id: 0,
            pretraining_tp: 2,
            partial_rotary_factor: 1,
            rms_norm_eps: 0.00005,
            rope_scaling: {
              factor: 1,
              original_max_position_embeddings: 16,
              rope_type: "default",
            },
            rope_theta: 14000,
            sliding_window: null,
            torch_dtype: "auto",
            use_sliding_window: false,
            use_cache: true,
            vocab_size: groupedTinyLlamaShape.vocabSize,
          }),
        },
        epsilon: 0.00005,
        ropeBase: 14000,
        ropeOriginalContextLength: 16,
        shape: groupedTinyLlamaShape,
        specs: groupedTinyLlamaTwoLayerTensorSpecs,
      },
      {
        label: "mistral-gqa-pipeline",
        metadata: {
          config: JSON.stringify({
            architectures: ["MistralForCausalLM"],
            attention_dropout: 0,
            attention_bias: false,
            bos_token_id: 1,
            eos_token_id: [2, 3],
            hidden_size: groupedTinyLlamaShape.hiddenSize,
            hidden_act: "silu",
            hidden_dropout: 0,
            intermediate_size: groupedTinyLlamaShape.ffnSize,
            max_position_embeddings: 16,
            mlp_bias: false,
            model_type: "mistral",
            num_attention_heads: 4,
            num_hidden_layers: groupedTinyLlamaShape.layers,
            num_key_value_heads: 2,
            pad_token_id: 0,
            pretraining_tp: 1,
            partial_rotary_factor: 1,
            rms_norm_eps: 0.00006,
            rope_scaling: null,
            rope_theta: 15000,
            sliding_window: 2,
            torch_dtype: "float32",
            use_sliding_window: true,
            use_cache: true,
            vocab_size: groupedTinyLlamaShape.vocabSize,
          }),
        },
        decodeTokens: [5],
        epsilon: 0.00006,
        prefillTokens: [2, 3],
        ropeBase: 15000,
        shape: groupedTinyLlamaShape,
        slidingWindow: 2,
        specs: groupedTinyLlamaTwoLayerTensorSpecs,
      },
      {
        label: "long-sliding-window-mistral-pipeline",
        config: {
          architectures: ["MistralForCausalLM"],
          attention_dropout: 0,
          attention_bias: false,
          hidden_size: longSlidingWindowTinyLlamaShape.hiddenSize,
          hidden_act: "silu",
          hidden_dropout: 0,
          intermediate_size: longSlidingWindowTinyLlamaShape.ffnSize,
          max_position_embeddings: longSlidingWindowTinyLlamaShape.contextLength,
          mlp_bias: false,
          model_type: "mistral",
          num_attention_heads: 4,
          num_hidden_layers: longSlidingWindowTinyLlamaShape.layers,
          num_key_value_heads: 2,
          pretraining_tp: 1,
          partial_rotary_factor: 1,
          rms_norm_eps: 0.000061,
          rope_scaling: null,
          rope_theta: 15100,
          sliding_window: 2,
          torch_dtype: "float32",
          use_sliding_window: true,
          use_cache: true,
          vocab_size: longSlidingWindowTinyLlamaShape.vocabSize,
        },
        decodeTokens: [5],
        epsilon: 0.000061,
        prefillTokens: Array.from({ length: 66 }, (_unused, index) => index % longSlidingWindowTinyLlamaShape.vocabSize),
        ropeBase: 15100,
        shape: longSlidingWindowTinyLlamaShape,
        slidingWindow: 2,
        specs: longSlidingWindowTinyLlamaTwoLayerTensorSpecs,
      },
      {
        label: "biased-qwen2-gqa-pipeline",
        attentionBiasLayers: [0, 1],
        config: {
          architectures: ["Qwen2ForCausalLM"],
          attention_dropout: 0,
          attention_bias: true,
          bos_token_id: 1,
          eos_token_id: [2, 3],
          head_dim: groupedTinyLlamaShape.attentionHeadSize,
          hidden_size: groupedTinyLlamaShape.hiddenSize,
          hidden_act: "silu",
          hidden_dropout: 0,
          intermediate_size: groupedTinyLlamaShape.ffnSize,
          max_position_embeddings: 16,
          mlp_bias: true,
          model_type: "qwen2",
          num_attention_heads: 4,
          num_hidden_layers: groupedTinyLlamaShape.layers,
          num_key_value_heads: 2,
          pad_token_id: 0,
          pretraining_tp: 1,
          partial_rotary_factor: 1,
          rms_norm_eps: 0.000075,
          rope_scaling: null,
          rope_theta: 16500,
          sliding_window: null,
          torch_dtype: "float32",
          tie_word_embeddings: false,
          use_sliding_window: false,
          use_cache: true,
          vocab_size: groupedTinyLlamaShape.vocabSize,
        },
        epsilon: 0.000075,
        ropeBase: 16500,
        shape: groupedTinyLlamaShape,
        specs: biasedGroupedTinyLlamaTwoLayerTensorSpecs,
        mlpBiasLayers: [0, 1],
      },
      {
        label: "structural-biased-qwen2-gqa-pipeline",
        attentionBiasLayers: [0, 1],
        config: {
          architectures: ["Qwen2ForCausalLM"],
          attention_dropout: 0,
          bos_token_id: 1,
          eos_token_id: [2, 3],
          head_dim: groupedTinyLlamaShape.attentionHeadSize,
          hidden_size: groupedTinyLlamaShape.hiddenSize,
          hidden_act: "silu",
          hidden_dropout: 0,
          intermediate_size: groupedTinyLlamaShape.ffnSize,
          max_position_embeddings: 16,
          model_type: "qwen2",
          num_attention_heads: 4,
          num_hidden_layers: groupedTinyLlamaShape.layers,
          num_key_value_heads: 2,
          pad_token_id: 0,
          pretraining_tp: 1,
          partial_rotary_factor: 1,
          rms_norm_eps: 0.000076,
          rope_scaling: null,
          rope_theta: 16600,
          sliding_window: null,
          torch_dtype: "float32",
          tie_word_embeddings: false,
          use_sliding_window: false,
          use_cache: true,
          vocab_size: groupedTinyLlamaShape.vocabSize,
        },
        epsilon: 0.000076,
        ropeBase: 16600,
        shape: groupedTinyLlamaShape,
        specs: biasedGroupedTinyLlamaTwoLayerTensorSpecs,
        mlpBiasLayers: [0, 1],
      },
      {
        label: "qwen2-gqa-pipeline",
        metadata: {
          config: JSON.stringify({
            architectures: ["Qwen2ForCausalLM"],
            attention_dropout: 0,
            attention_bias: false,
            bos_token_id: 1,
            eos_token_id: [2, 3],
            head_dim: groupedTinyLlamaShape.attentionHeadSize,
            hidden_size: groupedTinyLlamaShape.hiddenSize,
            hidden_act: "silu",
            hidden_dropout: 0,
            intermediate_size: groupedTinyLlamaShape.ffnSize,
            max_position_embeddings: 16,
            mlp_bias: false,
            model_type: "qwen2",
            num_attention_heads: 4,
            num_hidden_layers: groupedTinyLlamaShape.layers,
            num_key_value_heads: 2,
            pad_token_id: 0,
            pretraining_tp: 1,
            partial_rotary_factor: 1,
            rms_norm_eps: 0.00007,
            rope_scaling: null,
            rope_theta: 16000,
            sliding_window: 16,
            torch_dtype: "bfloat16",
            tie_word_embeddings: false,
            use_sliding_window: true,
            use_cache: true,
            vocab_size: groupedTinyLlamaShape.vocabSize,
          }),
        },
        epsilon: 0.00007,
        ropeBase: 16000,
        shape: groupedTinyLlamaShape,
        slidingWindow: 16,
        specs: groupedTinyLlamaTwoLayerTensorSpecs,
      },
      {
        label: "qwen3-qknorm-gqa-pipeline",
        config: {
          architectures: ["Qwen3ForCausalLM"],
          attention_dropout: 0,
          attention_bias: false,
          bos_token_id: 1,
          eos_token_id: [2, 3],
          head_dim: groupedTinyLlamaShape.attentionHeadSize,
          hidden_size: groupedTinyLlamaShape.hiddenSize,
          hidden_act: "silu",
          hidden_dropout: 0,
          intermediate_size: groupedTinyLlamaShape.ffnSize,
          max_position_embeddings: 16,
          mlp_bias: false,
          model_type: "qwen3",
          num_attention_heads: 4,
          num_hidden_layers: groupedTinyLlamaShape.layers,
          num_key_value_heads: 2,
          pad_token_id: 0,
          pretraining_tp: 1,
          partial_rotary_factor: 1,
          qk_norm: true,
          rms_norm_eps: 0.000082,
          rope_scaling: null,
          rope_theta: 17500,
          sliding_window: null,
          torch_dtype: "bfloat16",
          tie_word_embeddings: false,
          use_sliding_window: false,
          use_cache: true,
          vocab_size: groupedTinyLlamaShape.vocabSize,
        },
        epsilon: 0.000082,
        qkProjectionNorm: true,
        ropeBase: 17500,
        shape: groupedTinyLlamaShape,
        specs: qkNormGroupedTinyLlamaTwoLayerTensorSpecs,
      },
      {
        label: "structural-qwen3-qknorm-gqa-pipeline",
        config: {
          architectures: ["Qwen3ForCausalLM"],
          attention_dropout: 0,
          attention_bias: false,
          bos_token_id: 1,
          eos_token_id: [2, 3],
          head_dim: groupedTinyLlamaShape.attentionHeadSize,
          hidden_size: groupedTinyLlamaShape.hiddenSize,
          hidden_act: "silu",
          hidden_dropout: 0,
          intermediate_size: groupedTinyLlamaShape.ffnSize,
          max_position_embeddings: 16,
          mlp_bias: false,
          model_type: "qwen3",
          num_attention_heads: 4,
          num_hidden_layers: groupedTinyLlamaShape.layers,
          num_key_value_heads: 2,
          pad_token_id: 0,
          pretraining_tp: 1,
          partial_rotary_factor: 1,
          rms_norm_eps: 0.000083,
          rope_scaling: null,
          rope_theta: 17600,
          sliding_window: null,
          torch_dtype: "bfloat16",
          tie_word_embeddings: false,
          use_sliding_window: false,
          use_cache: true,
          vocab_size: groupedTinyLlamaShape.vocabSize,
        },
        epsilon: 0.000083,
        qkProjectionNorm: true,
        ropeBase: 17600,
        shape: groupedTinyLlamaShape,
        specs: qkNormGroupedTinyLlamaTwoLayerTensorSpecs,
      },
      {
        label: "smollm3-nope-gqa-pipeline",
        config: {
          architectures: ["SmolLM3ForCausalLM"],
          attention_dropout: 0,
          attention_bias: false,
          bos_token_id: 1,
          eos_token_id: [2, 3],
          hidden_size: groupedTinyLlamaShape.hiddenSize,
          hidden_act: "silu",
          hidden_dropout: 0,
          intermediate_size: groupedTinyLlamaShape.ffnSize,
          max_position_embeddings: 16,
          mlp_bias: false,
          model_type: "smollm3",
          no_rope_layer_interval: 2,
          num_attention_heads: 4,
          num_hidden_layers: groupedTinyLlamaShape.layers,
          num_key_value_heads: 2,
          pad_token_id: 0,
          pretraining_tp: 2,
          partial_rotary_factor: 1,
          rms_norm_eps: 0.000085,
          rope_scaling: null,
          rope_theta: 18000,
          sliding_window: null,
          torch_dtype: "bfloat16",
          tie_word_embeddings: false,
          use_sliding_window: false,
          use_cache: true,
          vocab_size: groupedTinyLlamaShape.vocabSize,
        },
        epsilon: 0.000085,
        ropeBase: 18000,
        shape: groupedTinyLlamaShape,
        specs: groupedTinyLlamaTwoLayerTensorSpecs,
        useRopeLayers: [1, 0],
      },
      {
        label: "realistic-gqa-pipeline",
        config: {
          architectures: ["LlamaForCausalLM"],
          attention_dropout: 0,
          attention_bias: false,
          hidden_size: realisticGqaTinyLlamaShape.hiddenSize,
          hidden_act: "silu",
          hidden_dropout: 0,
          intermediate_size: realisticGqaTinyLlamaShape.ffnSize,
          max_position_embeddings: 16,
          mlp_bias: false,
          model_type: "llama",
          num_attention_heads: 2,
          num_hidden_layers: realisticGqaTinyLlamaShape.layers,
          num_key_value_heads: 1,
          pretraining_tp: 1,
          partial_rotary_factor: 1,
          rms_norm_eps: 0.00008,
          rope_scaling: null,
          rope_theta: 17000,
          sliding_window: null,
          torch_dtype: "float32",
          use_sliding_window: false,
          use_cache: true,
          vocab_size: realisticGqaTinyLlamaShape.vocabSize,
        },
        epsilon: 0.00008,
        ropeBase: 17000,
        shape: realisticGqaTinyLlamaShape,
        specs: realisticGqaTinyLlamaTwoLayerTensorSpecs,
      },
    ];

	    const nativeBackedDefaultFamilyLabels = new Set([
	      "gguf-metadata-gqa-pipeline",
	      "qwen2-gguf-metadata-gqa-pipeline",
	      "config-gqa-pipeline",
	      "long-context-gqa-pipeline",
	      "llama3-rope-gqa-pipeline",
	      "metadata-config-gqa-pipeline",
	      "mistral-gqa-pipeline",
	      "biased-qwen2-gqa-pipeline",
	      "structural-biased-qwen2-gqa-pipeline",
	      "qwen2-gqa-pipeline",
	      "qwen3-qknorm-gqa-pipeline",
	      "structural-qwen3-qknorm-gqa-pipeline",
	      "smollm3-nope-gqa-pipeline",
	    ]);
	    const nativeBackedMismatchedFamilyLabels = new Set([
	      "metadata-mha-pipeline",
	      "config-json-mqa-pipeline",
	    ]);

		    {
		      let nativeOutputMismatchModel = 0;
		      let nativeOutputMismatchProgramHandle = 0;
		      try {
	        nativeOutputMismatchModel = createModel(tinyLlama2LayerKind, 0, 0);
	        nativeOutputMismatchProgramHandle = compileProgram(nativeOutputMismatchModel, groupedTinyLlamaShape.contextLength, backendWebGpu);
	        const nativeOutputMismatchKvRequirements = sessionBindingBridge.llamaKvCacheRequirements(nativeOutputMismatchProgramHandle);
	        const nativeOutputMismatchProgram = new WasmWebGpuLlamaResourceProgram({
	          device: hostDevice,
	          resourceBridge: hostResourceBridge,
	          sessionBindingBridge,
	          runtime: hostRuntime,
	          program: nativeOutputMismatchProgramHandle,
	          contextLength: groupedTinyLlamaShape.contextLength,
	          kvCacheRequirements: nativeOutputMismatchKvRequirements,
	          modelKind: tinyLlama2LayerKind,
	          vocabSize: tinyLlamaVocabSize + 1,
	        });
	        const claimedSessionsBefore = hostRuntime.claimedSessionHandles.size;
	        const hostBuffersBefore = buffers.length;
	        let rejected = false;
	        try {
	          nativeOutputMismatchProgram.bind();
	        } catch (err) {
	          rejected = /output requirements mismatch/.test(String(err?.message ?? err));
	        }
	        if (!rejected) {
	          throw new Error("native-backed WebGPU LLaMA output envelope mismatch did not reject before bind");
	        }
	        if (
	          hostRuntime.claimedSessionHandles.size !== claimedSessionsBefore ||
	          buffers.length !== hostBuffersBefore
	        ) {
	          throw new Error("native-backed WebGPU LLaMA output preflight created resources or claimed a Session");
	        }
	      } finally {
	        if (nativeOutputMismatchProgramHandle !== 0) exportsRef.zgml_program_free(nativeOutputMismatchProgramHandle);
		        if (nativeOutputMismatchModel !== 0) exportsRef.zgml_model_free(nativeOutputMismatchModel);
		      }
		    }

			    {
			      let nativeModelCompatibilityMismatchModel = 0;
			      let nativeModelCompatibilityMismatchBoundModel = 0;
		      let nativeModelCompatibilityMismatchProgramHandle = 0;
		      try {
		        nativeModelCompatibilityMismatchModel = createModel(tinyLlama2LayerKind, 0, 0);
		        nativeModelCompatibilityMismatchBoundModel = createModel(tinyLlamaKind, 0, 0);
		        nativeModelCompatibilityMismatchProgramHandle = compileProgram(nativeModelCompatibilityMismatchModel, groupedTinyLlamaShape.contextLength, backendWebGpu);
		        const nativeModelCompatibilityMismatchKvRequirements = sessionBindingBridge.llamaKvCacheRequirements(nativeModelCompatibilityMismatchProgramHandle);
		        const nativeModelCompatibilityMismatchProgram = new WasmWebGpuLlamaResourceProgram({
		          device: hostDevice,
		          resourceBridge: hostResourceBridge,
		          sessionBindingBridge,
		          runtime: hostRuntime,
		          program: nativeModelCompatibilityMismatchProgramHandle,
		          contextLength: groupedTinyLlamaShape.contextLength,
		          kvCacheRequirements: nativeModelCompatibilityMismatchKvRequirements,
		          modelKind: tinyLlama2LayerKind,
		          vocabSize: tinyLlamaVocabSize,
		        });
		        const claimedSessionsBefore = hostRuntime.claimedSessionHandles.size;
		        const hostBuffersBefore = buffers.length;
		        let rejected = false;
		        try {
		          nativeModelCompatibilityMismatchProgram.bind({ model: nativeModelCompatibilityMismatchBoundModel });
		        } catch (err) {
		          rejected = /model compatibility mismatch/.test(String(err?.message ?? err));
		        }
		        if (!rejected) {
		          throw new Error("native-backed WebGPU LLaMA model compatibility mismatch did not reject before bind");
		        }
		        if (
		          hostRuntime.claimedSessionHandles.size !== claimedSessionsBefore ||
		          buffers.length !== hostBuffersBefore
		        ) {
		          throw new Error("native-backed WebGPU LLaMA model compatibility preflight created resources or claimed a Session");
		        }
		      } finally {
		        if (nativeModelCompatibilityMismatchProgramHandle !== 0) exportsRef.zgml_program_free(nativeModelCompatibilityMismatchProgramHandle);
		        if (nativeModelCompatibilityMismatchBoundModel !== 0) exportsRef.zgml_model_free(nativeModelCompatibilityMismatchBoundModel);
			        if (nativeModelCompatibilityMismatchModel !== 0) exportsRef.zgml_model_free(nativeModelCompatibilityMismatchModel);
			      }
			    }

		    {
		      let nativeEnvelopeMismatchModel = 0;
		      let nativeEnvelopeMismatchProgramHandle = 0;
		      try {
		        nativeEnvelopeMismatchModel = createModel(tinyLlama2LayerKind, 0, 0);
		        nativeEnvelopeMismatchProgramHandle = compileProgram(nativeEnvelopeMismatchModel, groupedTinyLlamaShape.contextLength, backendWebGpu);
		        const nativeEnvelopeMismatchRequirements = sessionBindingBridge.programRequirements(nativeEnvelopeMismatchProgramHandle);
		        const nativeEnvelopeMismatchKvRequirements = sessionBindingBridge.llamaKvCacheRequirements(nativeEnvelopeMismatchProgramHandle);
		        const nativeEnvelopeMismatchProgram = new WasmWebGpuLlamaResourceProgram({
		          device: hostDevice,
		          resourceBridge: hostResourceBridge,
		          sessionBindingBridge,
		          runtime: hostRuntime,
		          program: nativeEnvelopeMismatchProgramHandle,
		          contextLength: groupedTinyLlamaShape.contextLength,
		          kvCacheRequirements: nativeEnvelopeMismatchKvRequirements,
		          modelKind: tinyLlama2LayerKind,
		          vocabSize: tinyLlamaVocabSize,
		        });
		        const envelopeMismatchCases = [
		          {
		            label: "token-id width",
		            requirements: { ...nativeEnvelopeMismatchRequirements, tokenIdBytes: 8 },
		            pattern: /token-id width mismatch/,
		          },
		          {
		            label: "output byte length",
		            requirements: { ...nativeEnvelopeMismatchRequirements, outputByteLength: nativeEnvelopeMismatchRequirements.outputByteLength - Float32Array.BYTES_PER_ELEMENT },
		            pattern: /output byte requirements mismatch/,
		          },
		          {
		            label: "batch",
		            requirements: { ...nativeEnvelopeMismatchRequirements, batch: 2 },
		            pattern: /batch requirements mismatch/,
		          },
		          {
		            label: "max token window",
		            requirements: { ...nativeEnvelopeMismatchRequirements, maxTokenWindow: 0 },
		            pattern: /token-window requirements mismatch/,
		          },
		        ];
		        for (const envelopeMismatchCase of envelopeMismatchCases) {
		          const claimedSessionsBefore = hostRuntime.claimedSessionHandles.size;
		          const hostBuffersBefore = buffers.length;
		          let rejected = false;
		          try {
		            nativeEnvelopeMismatchProgram.bind({}, {
		              nativeProgramRequirements: envelopeMismatchCase.requirements,
		              nativeKvRequirements: nativeEnvelopeMismatchKvRequirements,
		            });
		          } catch (err) {
		            rejected = envelopeMismatchCase.pattern.test(String(err?.message ?? err));
		          }
		          if (!rejected) {
		            throw new Error(`native-backed WebGPU LLaMA ${envelopeMismatchCase.label} envelope mismatch did not reject before bind`);
		          }
		          if (
		            hostRuntime.claimedSessionHandles.size !== claimedSessionsBefore ||
		            buffers.length !== hostBuffersBefore
		          ) {
		            throw new Error(`native-backed WebGPU LLaMA ${envelopeMismatchCase.label} envelope preflight created resources or claimed a Session`);
		          }
		        }
		      } finally {
		        if (nativeEnvelopeMismatchProgramHandle !== 0) exportsRef.zgml_program_free(nativeEnvelopeMismatchProgramHandle);
		        if (nativeEnvelopeMismatchModel !== 0) exportsRef.zgml_model_free(nativeEnvelopeMismatchModel);
		      }
		    }

			    for (const packedTinyLlamaProof of packedTinyLlamaProofs) {
      if (!shouldRunPackedLlamaProfile(packedTinyLlamaProof.label)) continue;
      markBrowserSmokeStage(`tiny-llama-webgpu-${packedTinyLlamaProof.label}`);
      const groupedTinyLlamaShape = packedTinyLlamaProof.shape;
      const groupedTinyLlamaTwoLayerTensorSpecs = packedTinyLlamaProof.specs;
      const groupedPipelineExecutorCalls = [];
      const groupedModelResourceBytes = tinyLlamaModelResourceFileBytes(groupedTinyLlamaTwoLayerTensorSpecs, {
        metadata: packedTinyLlamaProof.metadata,
      });
      const groupedPipelineProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        config: packedTinyLlamaProof.config,
        ...(packedTinyLlamaProof.omitContextLengthOption === true ? {} : { contextLength: groupedTinyLlamaShape.contextLength }),
        safetensors: groupedModelResourceBytes,
        allowMockFallback: true,
        executorOptions: {
          record(call) {
            recordBrowserLlamaPipelineCallEvidence(groupedPipelineExecutorCalls, packedTinyLlamaProof.label, call);
          },
        },
      });
      const expectedGroupedRopeScale = packedTinyLlamaProof.ropeScale ?? 1;
      const expectedGroupedRopeKind = packedTinyLlamaProof.ropeKind ?? (Math.abs(expectedGroupedRopeScale - 1) > 1e-6 ? "linear" : "none");
      const expectedGroupedRopeLowFreqFactor = packedTinyLlamaProof.ropeLowFreqFactor ?? 1;
      const expectedGroupedRopeHighFreqFactor = packedTinyLlamaProof.ropeHighFreqFactor ?? 1;
      const expectedGroupedRopeOriginalContextLength = packedTinyLlamaProof.ropeOriginalContextLength ?? 1;
      const expectedGroupedUseRopeLayers = packedTinyLlamaProof.useRopeLayers ?? Array(groupedTinyLlamaShape.layers).fill(1);
      const expectedGroupedQkProjectionNorm = packedTinyLlamaProof.qkProjectionNorm ?? false;
      const expectedGroupedAttentionBiasLayers = packedTinyLlamaProof.attentionBiasLayers ?? [];
      const expectedGroupedMlpBiasLayers = packedTinyLlamaProof.mlpBiasLayers ?? [];
      if (
        groupedPipelineProgram.vocabSize !== groupedTinyLlamaShape.vocabSize ||
        groupedPipelineProgram.hiddenSize !== groupedTinyLlamaShape.hiddenSize ||
        groupedPipelineProgram.ffnSize !== groupedTinyLlamaShape.ffnSize ||
        groupedPipelineProgram.kvRequirements.layers !== groupedTinyLlamaShape.layers ||
        groupedPipelineProgram.kvRequirements.kBufferByteLength !== groupedTinyLlamaShape.contextLength * groupedTinyLlamaShape.kvSize * Float32Array.BYTES_PER_ELEMENT ||
        groupedPipelineProgram.kvRequirements.vBufferByteLength !== groupedTinyLlamaShape.contextLength * groupedTinyLlamaShape.kvSize * Float32Array.BYTES_PER_ELEMENT ||
        groupedPipelineProgram.tokenExecutor.blocks[0].executor.attentionHeadSize !== groupedTinyLlamaShape.attentionHeadSize ||
        groupedPipelineProgram.tokenExecutor.blocks[0].executor.ropeBase !== packedTinyLlamaProof.ropeBase ||
        groupedPipelineProgram.tokenExecutor.blocks[0].executor.ropeKind !== expectedGroupedRopeKind ||
        groupedPipelineProgram.tokenExecutor.blocks[0].executor.ropeLowFreqFactor !== expectedGroupedRopeLowFreqFactor ||
        groupedPipelineProgram.tokenExecutor.blocks[0].executor.ropeHighFreqFactor !== expectedGroupedRopeHighFreqFactor ||
        groupedPipelineProgram.tokenExecutor.blocks[0].executor.ropeOriginalContextLength !== expectedGroupedRopeOriginalContextLength ||
        groupedPipelineProgram.tokenExecutor.blocks[0].executor.ropeScale !== expectedGroupedRopeScale ||
        groupedPipelineProgram.tokenExecutor.blocks[0].executor.slidingWindow !== (packedTinyLlamaProof.slidingWindow ?? null) ||
        groupedPipelineProgram.tokenExecutor.blocks[0].executor.epsilon !== packedTinyLlamaProof.epsilon ||
        groupedPipelineProgram.tokenExecutor.blocks[0].executor.qkProjectionNorm !== expectedGroupedQkProjectionNorm
      ) {
        throw new Error(`host-only WebGPU grouped LLaMA safetensors Program did not infer packed K/V resources for ${packedTinyLlamaProof.label}: ${JSON.stringify({
          actual: {
            attentionHeadSize: groupedPipelineProgram.tokenExecutor.blocks[0].executor.attentionHeadSize,
            epsilon: groupedPipelineProgram.tokenExecutor.blocks[0].executor.epsilon,
            ffnSize: groupedPipelineProgram.ffnSize,
            hiddenSize: groupedPipelineProgram.hiddenSize,
            kvLayers: groupedPipelineProgram.kvRequirements.layers,
            ropeBase: groupedPipelineProgram.tokenExecutor.blocks[0].executor.ropeBase,
            ropeHighFreqFactor: groupedPipelineProgram.tokenExecutor.blocks[0].executor.ropeHighFreqFactor,
            ropeKind: groupedPipelineProgram.tokenExecutor.blocks[0].executor.ropeKind,
            ropeLowFreqFactor: groupedPipelineProgram.tokenExecutor.blocks[0].executor.ropeLowFreqFactor,
            ropeOriginalContextLength: groupedPipelineProgram.tokenExecutor.blocks[0].executor.ropeOriginalContextLength,
            ropeScale: groupedPipelineProgram.tokenExecutor.blocks[0].executor.ropeScale,
            qkProjectionNorm: groupedPipelineProgram.tokenExecutor.blocks[0].executor.qkProjectionNorm,
            slidingWindow: groupedPipelineProgram.tokenExecutor.blocks[0].executor.slidingWindow,
            vocabSize: groupedPipelineProgram.vocabSize,
          },
          expected: {
            attentionHeadSize: groupedTinyLlamaShape.attentionHeadSize,
            epsilon: packedTinyLlamaProof.epsilon,
            ffnSize: groupedTinyLlamaShape.ffnSize,
            hiddenSize: groupedTinyLlamaShape.hiddenSize,
            kvLayers: groupedTinyLlamaShape.layers,
            ropeBase: packedTinyLlamaProof.ropeBase,
            ropeHighFreqFactor: expectedGroupedRopeHighFreqFactor,
            ropeKind: expectedGroupedRopeKind,
            ropeLowFreqFactor: expectedGroupedRopeLowFreqFactor,
            ropeOriginalContextLength: expectedGroupedRopeOriginalContextLength,
            ropeScale: expectedGroupedRopeScale,
            qkProjectionNorm: expectedGroupedQkProjectionNorm,
            slidingWindow: packedTinyLlamaProof.slidingWindow ?? null,
            vocabSize: groupedTinyLlamaShape.vocabSize,
          },
        })}`);
      }
      const groupedProgramInspection = groupedPipelineProgram.inspect();
      if (
        JSON.stringify(groupedProgramInspection.attentionBiasLayers ?? []) !== JSON.stringify(expectedGroupedAttentionBiasLayers) ||
        JSON.stringify(groupedProgramInspection.mlpBiasLayers ?? []) !== JSON.stringify(expectedGroupedMlpBiasLayers)
      ) {
        throw new Error(`host-only WebGPU grouped LLaMA safetensors Program did not inspect bias layers for ${packedTinyLlamaProof.label}: ${JSON.stringify({
          actual: {
            attentionBiasLayers: groupedProgramInspection.attentionBiasLayers,
            mlpBiasLayers: groupedProgramInspection.mlpBiasLayers,
          },
          expected: {
            attentionBiasLayers: expectedGroupedAttentionBiasLayers,
            mlpBiasLayers: expectedGroupedMlpBiasLayers,
          },
        })}`);
      }
      for (let layer = 0; layer < groupedTinyLlamaShape.layers; layer += 1) {
        const expectedUseRope = expectedGroupedUseRopeLayers[layer] !== 0;
        const actualUseRope = groupedPipelineProgram.tokenExecutor.blocks[layer].executor.useRope;
        if (actualUseRope !== expectedUseRope) {
          throw new Error(`host-only WebGPU grouped LLaMA safetensors Program did not infer layer ${layer} useRope for ${packedTinyLlamaProof.label}: ${actualUseRope} != ${expectedUseRope}`);
        }
        const actualQkProjectionNorm = groupedPipelineProgram.tokenExecutor.blocks[layer].executor.qkProjectionNorm;
        if (actualQkProjectionNorm !== expectedGroupedQkProjectionNorm) {
          throw new Error(`host-only WebGPU grouped LLaMA safetensors Program did not infer layer ${layer} Q/K projection norm for ${packedTinyLlamaProof.label}: ${actualQkProjectionNorm} != ${expectedGroupedQkProjectionNorm}`);
        }
      }
      if (nativeBackedDefaultFamilyLabels.has(packedTinyLlamaProof.label)) {
        let nativeFamilyModel = 0;
        let nativeFamilyCompatibleModel = 0;
        let nativeFamilyProgramHandle = 0;
        let nativeFamilyProgram = null;
        let nativeFamilySession = null;
        let nativeFamilySessionHandle = 0;
        const nativeFamilyExecutorCalls = [];
        try {
          nativeFamilyModel = createModel(tinyLlama2LayerKind, 0, 0);
          nativeFamilyCompatibleModel = createModel(tinyLlama2LayerKind, 0, 0);
          nativeFamilyProgramHandle = compileProgram(nativeFamilyModel, groupedTinyLlamaShape.contextLength, backendWebGpu);
          const nativeFamilyKvRequirements = keep(alloc(llamaKvCacheRequirementsSize), llamaKvCacheRequirementsSize);
          zero(nativeFamilyKvRequirements, llamaKvCacheRequirementsSize);
          check(exportsRef.zgml_llama_program_get_kv_cache_requirements(nativeFamilyProgramHandle, nativeFamilyKvRequirements));
          nativeFamilyProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
            device: hostDevice,
            resourceBridge: hostResourceBridge,
            sessionBindingBridge,
            runtime: hostRuntime,
            program: nativeFamilyProgramHandle,
            config: packedTinyLlamaProof.config,
            ...(packedTinyLlamaProof.omitContextLengthOption === true ? {} : { contextLength: groupedTinyLlamaShape.contextLength }),
            model: nativeFamilyCompatibleModel,
            safetensors: groupedModelResourceBytes,
            allowMockFallback: true,
            executorOptions: {
              record(call) {
                recordBrowserLlamaPipelineCallEvidence(nativeFamilyExecutorCalls, `native-${packedTinyLlamaProof.label}`, call);
              },
            },
          });
          if (
            nativeFamilyProgram.kvRequirements.layers !== groupedTinyLlamaShape.layers ||
            nativeFamilyProgram.kvRequirements.kBufferByteLength !== u32(nativeFamilyKvRequirements, 20) ||
            nativeFamilyProgram.kvRequirements.vBufferByteLength !== u32(nativeFamilyKvRequirements, 24)
          ) {
            throw new Error(`native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} default safetensors Program did not match native K/V requirements`);
          }
	          nativeFamilySession = nativeFamilyProgram.bind();
          nativeFamilySessionHandle = nativeFamilySession.handle;
          if (nativeFamilySession.nativeSession !== true || !hostRuntime.hasClaimed(nativeFamilySessionHandle)) {
            throw new Error(`native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} default safetensors Session was not registered for raw ABI execution`);
          }
          if (
            nativeFamilySession.modelHandle !== nativeFamilyCompatibleModel ||
            nativeFamilySession.modelResources.length !== nativeFamilyProgram.requiredModelResources.length ||
            nativeFamilySession.bindings.output.byteLength !== tinyLlamaVocabSize * Float32Array.BYTES_PER_ELEMENT
          ) {
            throw new Error(`native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} default safetensors bind did not preserve model resources and ABI-sized output`);
          }
          await expectTinyLlamaModelResourceBytes(
            hostDevice,
            nativeFamilySession.modelResources,
            groupedModelResourceBytes,
            `native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} default safetensors`,
          );
          expectHostResourceTableRoles(
            nativeFamilySession.table,
            [
              "llama.output",
              ...Array.from({ length: groupedTinyLlamaShape.layers }, (_entry, layer) => [`llama.k.${layer}`, `llama.v.${layer}`]).flat(),
            ],
            `native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} default safetensors`,
          );
          const nativeFamilyCaches = nativeFamilySession.bindings.kvCache.map((layer, index) => {
            const kElements = Math.floor(layer.k.byteLength / Float32Array.BYTES_PER_ELEMENT);
            const vElements = Math.floor(layer.v.byteLength / Float32Array.BYTES_PER_ELEMENT);
            if (
              kElements !== groupedTinyLlamaShape.contextLength * groupedTinyLlamaShape.kvSize ||
              vElements !== groupedTinyLlamaShape.contextLength * groupedTinyLlamaShape.kvSize
            ) {
              throw new Error(`native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} K/V cache was not packed to K/V width`);
            }
            hostDevice.writeFloat32(layer.k, Array(kElements).fill(-9100 - index * 100));
            hostDevice.writeFloat32(layer.v, Array(vElements).fill(-9200 - index * 100));
            return { k: layer.k, kElements, v: layer.v, vElements };
          });
          const nativeFamilyOutputSentinel = Array(tinyLlamaVocabSize).fill(-9160);
          hostDevice.writeFloat32(nativeFamilySession.bindings.output, nativeFamilyOutputSentinel);
          const nativeFamilyPrefillTokens = packedTinyLlamaProof.prefillTokens ?? [2];
          const nativeFamilyPrefillTokensPtr = keep(alloc(nativeFamilyPrefillTokens.length * 4), nativeFamilyPrefillTokens.length * 4);
          writeU32Array(nativeFamilyPrefillTokensPtr, nativeFamilyPrefillTokens);
          const nativeFamilyPrefill = tokenExecute(nativeFamilySessionHandle, nativeFamilyPrefillTokensPtr, nativeFamilyPrefillTokens.length, executeOutputNone, 0, 0);
          expectStatus(nativeFamilyPrefill.status, ok, `native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} raw no-output prefill`);
          if (nativeFamilyPrefill.outputLen !== 0 || sessionPosition(nativeFamilySessionHandle) !== nativeFamilyPrefillTokens.length) {
            throw new Error(`native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} raw prefill did not advance without output`);
          }
          expectClose(
            Array.from(await hostDevice.readFloat32(nativeFamilySession.bindings.output, tinyLlamaVocabSize)),
            nativeFamilyOutputSentinel,
          );
          const nativeFamilyDecodeTokens = packedTinyLlamaProof.decodeTokens ?? [5];
          const nativeFamilyDecodeTokensPtr = keep(alloc(nativeFamilyDecodeTokens.length * 4), nativeFamilyDecodeTokens.length * 4);
          writeU32Array(nativeFamilyDecodeTokensPtr, nativeFamilyDecodeTokens);
          const nativeFamilyDecode = tokenExecute(nativeFamilySessionHandle, nativeFamilyDecodeTokensPtr, nativeFamilyDecodeTokens.length, executeOutputLogits, 0, 0);
          expectStatus(nativeFamilyDecode.status, ok, `native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} raw decode-after-prefill`);
          if (
            nativeFamilyDecode.outputLen !== groupedTinyLlamaShape.vocabSize ||
            sessionPosition(nativeFamilySessionHandle) !== nativeFamilyPrefillTokens.length + nativeFamilyDecodeTokens.length
          ) {
            throw new Error(`native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} raw decode did not emit family logits and advance`);
          }
          const nativeFamilyInspection = inspectSession(nativeFamilySessionHandle);
          if (
            u32(nativeFamilyInspection, 0) !== tinyLlama2LayerKind ||
            u32(nativeFamilyInspection, 4) !== backendWebGpu ||
            u32(nativeFamilyInspection, 8) !== bufferStorageExternalResource ||
            u32(nativeFamilyInspection, 12) !== bufferStorageExternalResource ||
            u64(nativeFamilyInspection, 16) !== BigInt(nativeFamilyPrefillTokens.length + nativeFamilyDecodeTokens.length) ||
            u64(nativeFamilyInspection, 24) !== BigInt(groupedTinyLlamaShape.contextLength) ||
            u64(nativeFamilyInspection, 64) !== BigInt(1 + groupedTinyLlamaShape.layers * 2) ||
            u64(nativeFamilyInspection, 72) === 0n
          ) {
            throw new Error(`native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} raw inspect mismatch`);
          }
          const expectedNativeFamilyCaches = nativeFamilyCaches.map((cache, index) => ({
            k: Array(cache.kElements).fill(-9100 - index * 100),
            v: Array(cache.vElements).fill(-9200 - index * 100),
          }));
          const expectedNativeFamily = [];
          const runExpectedNativeFamilyToken = (token, position) => {
            let inputHidden = null;
            const tokenLayers = [];
            for (let layer = 0; layer < groupedTinyLlamaShape.layers; layer += 1) {
              const expected = tinyLlamaBlockProjectionStep(groupedModelResourceBytes, expectedNativeFamilyCaches[layer], token, position, {
                attentionHeadSize: groupedTinyLlamaShape.attentionHeadSize,
                ...(inputHidden === null ? {} : { inputHidden }),
                epsilon: packedTinyLlamaProof.epsilon,
                layer,
                ropeBase: packedTinyLlamaProof.ropeBase,
                ropeHighFreqFactor: expectedGroupedRopeHighFreqFactor,
                ropeKind: expectedGroupedRopeKind,
                ropeLowFreqFactor: expectedGroupedRopeLowFreqFactor,
                ropeOriginalContextLength: expectedGroupedRopeOriginalContextLength,
                ropeScale: expectedGroupedRopeScale,
                slidingWindow: packedTinyLlamaProof.slidingWindow ?? null,
                useRope: expectedGroupedUseRopeLayers[layer] !== 0,
              });
              inputHidden = expected.blockHidden;
              tokenLayers.push(expected);
            }
            return tokenLayers;
          };
          for (let tokenIndex = 0; tokenIndex < nativeFamilyPrefillTokens.length; tokenIndex += 1) {
            expectedNativeFamily.push(runExpectedNativeFamilyToken(nativeFamilyPrefillTokens[tokenIndex], tokenIndex));
          }
          for (let tokenIndex = 0; tokenIndex < nativeFamilyDecodeTokens.length; tokenIndex += 1) {
            expectedNativeFamily.push(runExpectedNativeFamilyToken(nativeFamilyDecodeTokens[tokenIndex], nativeFamilyPrefillTokens.length + tokenIndex));
          }
          const expectedNativeFamilyLast = expectedNativeFamily.at(-1).at(-1);
          const nativeFamilyTolerance = groupedTinyLlamaShape.attentionHeadSize >= 128 ? 1e-3 : 1e-5;
          const nativeFamilyOutput = Array.from(await hostDevice.readFloat32(nativeFamilySession.bindings.output, tinyLlamaVocabSize));
          expectClose(nativeFamilyOutput.slice(0, groupedTinyLlamaShape.vocabSize), expectedNativeFamilyLast.logits, nativeFamilyTolerance);
          expectClose(nativeFamilyOutput.slice(groupedTinyLlamaShape.vocabSize), nativeFamilyOutputSentinel.slice(groupedTinyLlamaShape.vocabSize), nativeFamilyTolerance);
          for (let layer = 0; layer < groupedTinyLlamaShape.layers; layer += 1) {
            for (let tokenIndex = 0; tokenIndex < expectedNativeFamily.length; tokenIndex += 1) {
              const expected = expectedNativeFamily[tokenIndex][layer];
              const offset = tokenIndex * expected.k.length * Float32Array.BYTES_PER_ELEMENT;
              expectClose(Array.from(await hostDevice.readFloat32(nativeFamilyCaches[layer].k, expected.k.length, offset)), expected.k, nativeFamilyTolerance);
              expectClose(Array.from(await hostDevice.readFloat32(nativeFamilyCaches[layer].v, expected.v.length, offset)), expected.v, nativeFamilyTolerance);
            }
          }
          const expectedNativeFamilyPrefillCommandCount = expectedLlamaBlockNoOutputPrefillCommandCount(
            hostDevice,
            groupedTinyLlamaShape.layers,
            nativeFamilyPrefillTokens.length,
            {
              contextLength: groupedTinyLlamaShape.contextLength,
              hiddenSize: groupedTinyLlamaShape.hiddenSize,
              slidingWindow: packedTinyLlamaProof.slidingWindow ?? 0,
            },
          );
          const expectedNativeFamilyPrefillDispatchFamilies = expectedLlamaBlockPipelineDispatchFamilies(
            hostDevice,
            groupedTinyLlamaShape.layers,
            nativeFamilyPrefillTokens.length,
            executeOutputNone,
            {
              contextLength: groupedTinyLlamaShape.contextLength,
              hiddenSize: groupedTinyLlamaShape.hiddenSize,
              slidingWindow: packedTinyLlamaProof.slidingWindow ?? 0,
            },
          );
          const expectedNativeFamilyDecodeCommandCount = expectedLlamaBlockPipelineCommandCount(
            hostDevice,
            groupedTinyLlamaShape.layers,
            nativeFamilyDecodeTokens.length,
            executeOutputLogits,
            {
              contextLength: groupedTinyLlamaShape.contextLength,
              hiddenSize: groupedTinyLlamaShape.hiddenSize,
              slidingWindow: packedTinyLlamaProof.slidingWindow ?? 0,
              splitTerminalLogits: groupedTinyLlamaShape.hiddenSize >= 64,
              startPosition: nativeFamilyPrefillTokens.length,
            },
          );
          const expectedNativeFamilyDecodeDispatchFamilies = expectedLlamaBlockPipelineDispatchFamilies(
            hostDevice,
            groupedTinyLlamaShape.layers,
            nativeFamilyDecodeTokens.length,
            executeOutputLogits,
            {
              contextLength: groupedTinyLlamaShape.contextLength,
              hiddenSize: groupedTinyLlamaShape.hiddenSize,
              slidingWindow: packedTinyLlamaProof.slidingWindow ?? 0,
              splitTerminalLogits: groupedTinyLlamaShape.hiddenSize >= 64,
              startPosition: nativeFamilyPrefillTokens.length,
            },
          );
          const expectedNativeFamilyCommandCount = expectedNativeFamilyPrefillCommandCount + expectedNativeFamilyDecodeCommandCount;
          const expectedNativeFamilyBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? expectedNativeFamilyCommandCount : 0;
          const expectedNativeFamilyFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : expectedNativeFamilyCommandCount;
          const expectedNativeFamilyUsesGpuStorageBuffers = canUseGpuStorageBuffers(hostDevice, 6);
          const nativeFamilyProfile = nativeFamilySession.runtimeProfile();
          if (
            nativeFamilyProfile.callCount !== 2 ||
            nativeFamilyProfile.noOutputCallCount !== 1 ||
            nativeFamilyProfile.logitsOutputCallCount !== 1 ||
            nativeFamilyProfile.executorCallCount !== 2 ||
            nativeFamilyProfile.executorOkCount !== 2 ||
            nativeFamilyProfile.executorBackendDispatchCount !== expectedNativeFamilyBackendDispatchCount ||
            nativeFamilyProfile.executorFallbackOpCount !== expectedNativeFamilyFallbackOpCount ||
            nativeFamilyProfile.executorCommandCount !== expectedNativeFamilyCommandCount ||
            nativeFamilyProfile.executorCommandOpCount !== expectedNativeFamilyCommandCount ||
            nativeFamilyProfile.lastStatus !== ok ||
            nativeFamilyProfile.lastResultOutputLength !== groupedTinyLlamaShape.vocabSize
          ) {
            throw new Error(`unexpected native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} default safetensors profile: ${JSON.stringify(nativeFamilyProfile)}`);
          }
          const nativeFamilyAbiProfile = sessionRuntimeProfile(nativeFamilySessionHandle);
          if (
            u64(nativeFamilyAbiProfile, 0) !== 2n ||
            u64(nativeFamilyAbiProfile, 8) !== BigInt(expectedNativeFamilyBackendDispatchCount) ||
            u64(nativeFamilyAbiProfile, 16) !== BigInt(expectedNativeFamilyFallbackOpCount) ||
            u64(nativeFamilyAbiProfile, 24) !== BigInt(expectedNativeFamilyBackendDispatchCount) ||
            u64(nativeFamilyAbiProfile, 40) !== 2n ||
            u64(nativeFamilyAbiProfile, 48) !== 2n ||
            u64(nativeFamilyAbiProfile, 56) !== 0n ||
            u64(nativeFamilyAbiProfile, 96) !== BigInt(expectedNativeFamilyCommandCount) ||
            u64(nativeFamilyAbiProfile, 112) !== BigInt(expectedNativeFamilyCommandCount)
          ) {
            throw new Error(`native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} default safetensors ABI profile mismatch`);
          }
          if (
            nativeFamilyExecutorCalls.length !== 2 ||
            nativeFamilyExecutorCalls[0].stageCount !== groupedTinyLlamaShape.layers ||
            nativeFamilyExecutorCalls[0].commandCount !== expectedNativeFamilyPrefillCommandCount ||
            nativeFamilyExecutorCalls[0].deviceMode !== expectedLlamaExecutorDeviceMode(hostDevice) ||
            nativeFamilyExecutorCalls[0].genericDispatchCount !== expectedNativeFamilyPrefillDispatchFamilies.generic ||
            nativeFamilyExecutorCalls[0].maxStorageBuffersPerShaderStage !== hostDevice.maxStorageBuffersPerShaderStage() ||
            nativeFamilyExecutorCalls[0].outputPolicy !== executeOutputNone ||
            nativeFamilyExecutorCalls[0].requiredStorageBufferCount !== 6 ||
            nativeFamilyExecutorCalls[0].scalarDispatchCount !== expectedNativeFamilyPrefillDispatchFamilies.scalar ||
            nativeFamilyExecutorCalls[0].usesGpuStorageBuffers !== expectedNativeFamilyUsesGpuStorageBuffers ||
            nativeFamilyExecutorCalls[0].windowDispatchCount !== expectedNativeFamilyPrefillDispatchFamilies.window ||
            nativeFamilyExecutorCalls[1].stageCount !== groupedTinyLlamaShape.layers ||
            nativeFamilyExecutorCalls[1].commandCount !== expectedNativeFamilyDecodeCommandCount ||
            nativeFamilyExecutorCalls[1].deviceMode !== expectedLlamaExecutorDeviceMode(hostDevice) ||
            nativeFamilyExecutorCalls[1].genericDispatchCount !== expectedNativeFamilyDecodeDispatchFamilies.generic ||
            nativeFamilyExecutorCalls[1].maxStorageBuffersPerShaderStage !== hostDevice.maxStorageBuffersPerShaderStage() ||
            nativeFamilyExecutorCalls[1].outputPolicy !== executeOutputLogits ||
            nativeFamilyExecutorCalls[1].outputLength !== groupedTinyLlamaShape.vocabSize ||
            nativeFamilyExecutorCalls[1].requiredStorageBufferCount !== 6 ||
            nativeFamilyExecutorCalls[1].scalarDispatchCount !== expectedNativeFamilyDecodeDispatchFamilies.scalar ||
            nativeFamilyExecutorCalls[1].usesGpuStorageBuffers !== expectedNativeFamilyUsesGpuStorageBuffers ||
            nativeFamilyExecutorCalls[1].windowDispatchCount !== expectedNativeFamilyDecodeDispatchFamilies.window ||
            nativeFamilyExecutorCalls[1].status !== ok
          ) {
            throw new Error(`native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} default safetensors pipeline evidence mismatch`);
          }
        } finally {
          if (nativeFamilySessionHandle !== 0) exportsRef.zgml_session_free(nativeFamilySessionHandle);
          if (nativeFamilySession && nativeFamilySession.ownedResources.length !== 0) {
            throw new Error(`native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} default safetensors Session-owned resources were not released`);
          }
          nativeFamilyProgram?.tokenExecutor?.destroy?.();
	          if (nativeFamilyProgramHandle !== 0) exportsRef.zgml_program_free(nativeFamilyProgramHandle);
	          if (nativeFamilyCompatibleModel !== 0) exportsRef.zgml_model_free(nativeFamilyCompatibleModel);
	          if (nativeFamilyModel !== 0) exportsRef.zgml_model_free(nativeFamilyModel);
	        }
	      }
	      if (nativeBackedMismatchedFamilyLabels.has(packedTinyLlamaProof.label)) {
	        let nativeMismatchModel = 0;
	        let nativeMismatchCompatibleModel = 0;
	        let nativeMismatchProgramHandle = 0;
	        let nativeMismatchProgram = null;
	        try {
	          nativeMismatchModel = createModel(tinyLlama2LayerKind, 0, 0);
	          nativeMismatchCompatibleModel = createModel(tinyLlama2LayerKind, 0, 0);
	          nativeMismatchProgramHandle = compileProgram(nativeMismatchModel, groupedTinyLlamaShape.contextLength, backendWebGpu);
	          nativeMismatchProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
	            device: hostDevice,
	            resourceBridge: hostResourceBridge,
	            sessionBindingBridge,
	            runtime: hostRuntime,
	            program: nativeMismatchProgramHandle,
	            config: packedTinyLlamaProof.config,
	            contextLength: groupedTinyLlamaShape.contextLength,
	            model: nativeMismatchCompatibleModel,
	            safetensors: groupedModelResourceBytes,
	          });
	          const claimedSessionsBefore = hostRuntime.claimedSessionHandles.size;
	          const hostBuffersBefore = buffers.length;
	          let rejected = false;
	          try {
	            nativeMismatchProgram.bind({ outputLen: tinyLlamaVocabSize });
	          } catch (err) {
	            rejected = /K\/V requirements mismatch/.test(String(err?.message ?? err));
	          }
	          if (!rejected) {
	            throw new Error(`native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} did not reject mismatched K/V requirements before bind`);
	          }
	          if (
	            hostRuntime.claimedSessionHandles.size !== claimedSessionsBefore ||
	            buffers.length !== hostBuffersBefore
	          ) {
	            throw new Error(`native-backed WebGPU grouped LLaMA ${packedTinyLlamaProof.label} K/V preflight created resources or claimed a Session`);
	          }
	        } finally {
	          nativeMismatchProgram?.tokenExecutor?.destroy?.();
	          if (nativeMismatchProgramHandle !== 0) exportsRef.zgml_program_free(nativeMismatchProgramHandle);
	          if (nativeMismatchCompatibleModel !== 0) exportsRef.zgml_model_free(nativeMismatchCompatibleModel);
	          if (nativeMismatchModel !== 0) exportsRef.zgml_model_free(nativeMismatchModel);
	        }
	      }
	      const groupedPipelineSession = groupedPipelineProgram.bindHostResources();
      const groupedPipelineSessionHandle = groupedPipelineSession.handle;
      const groupedHostExports = hostRuntime.wrapExports(exportsRef);
      try {
      expectHostResourceTableRoles(
        groupedPipelineSession.table,
        ["llama.output", "llama.k.0", "llama.v.0", "llama.k.1", "llama.v.1"],
        "host-only WebGPU grouped LLaMA block pipeline",
      );
      const groupedCaches = groupedPipelineSession.bindings.kvCache.map((layer, index) => {
        const kElements = Math.floor(layer.k.byteLength / Float32Array.BYTES_PER_ELEMENT);
        const vElements = Math.floor(layer.v.byteLength / Float32Array.BYTES_PER_ELEMENT);
        if (
          kElements !== groupedTinyLlamaShape.contextLength * groupedTinyLlamaShape.kvSize ||
          vElements !== groupedTinyLlamaShape.contextLength * groupedTinyLlamaShape.kvSize
        ) {
          throw new Error("host-only WebGPU grouped LLaMA K/V cache was not packed to K/V width");
        }
        hostDevice.writeFloat32(layer.k, Array(kElements).fill(-8100 - index * 100));
        hostDevice.writeFloat32(layer.v, Array(vElements).fill(-8200 - index * 100));
        return { k: layer.k, kElements, v: layer.v, vElements };
      });
      const groupedOutputSentinel = Array(groupedTinyLlamaShape.vocabSize).fill(-8160);
      hostDevice.writeFloat32(groupedPipelineSession.bindings.output, groupedOutputSentinel);
      const groupedPrefillTokens = packedTinyLlamaProof.prefillTokens ?? [2];
      const groupedPrefillTokensPtr = keep(alloc(groupedPrefillTokens.length * 4), groupedPrefillTokens.length * 4);
      writeU32Array(groupedPrefillTokensPtr, groupedPrefillTokens);
      const groupedPrefill = tokenExecuteWith(groupedHostExports, groupedPipelineSessionHandle, groupedPrefillTokensPtr, groupedPrefillTokens.length, executeOutputNone, 0, 0);
      expectStatus(groupedPrefill.status, ok, "host-only WebGPU grouped LLaMA block pipeline no-output prefill");
      if (groupedPrefill.outputLen !== 0 || sessionPositionWith(groupedHostExports, groupedPipelineSessionHandle) !== groupedPrefillTokens.length) {
        throw new Error("host-only WebGPU grouped LLaMA block pipeline prefill did not advance without output");
      }
      expectClose(
        Array.from(await hostDevice.readFloat32(groupedPipelineSession.bindings.output, groupedTinyLlamaShape.vocabSize)),
        groupedOutputSentinel,
      );
      const groupedDecodeTokens = packedTinyLlamaProof.decodeTokens ?? [5];
      const groupedDecodeTokensPtr = keep(alloc(groupedDecodeTokens.length * 4), groupedDecodeTokens.length * 4);
      writeU32Array(groupedDecodeTokensPtr, groupedDecodeTokens);
      const groupedDecode = tokenExecuteWith(groupedHostExports, groupedPipelineSessionHandle, groupedDecodeTokensPtr, groupedDecodeTokens.length, executeOutputLogits, 0, 0);
      expectStatus(groupedDecode.status, ok, "host-only WebGPU grouped LLaMA block pipeline decode-after-prefill");
      if (
        groupedDecode.outputLen !== groupedTinyLlamaShape.vocabSize ||
        sessionPositionWith(groupedHostExports, groupedPipelineSessionHandle) !== groupedPrefillTokens.length + groupedDecodeTokens.length
      ) {
        throw new Error("host-only WebGPU grouped LLaMA block pipeline decode did not emit bound logits and advance");
      }
      const expectedGroupedCaches = groupedCaches.map((cache, index) => ({
        k: Array(cache.kElements).fill(-8100 - index * 100),
        v: Array(cache.vElements).fill(-8200 - index * 100),
      }));
      const expectedGrouped = [];
      const runExpectedGroupedToken = (token, position) => {
        let inputHidden = null;
        const tokenLayers = [];
        for (let layer = 0; layer < groupedTinyLlamaShape.layers; layer += 1) {
          const expected = tinyLlamaBlockProjectionStep(groupedModelResourceBytes, expectedGroupedCaches[layer], token, position, {
            attentionHeadSize: groupedTinyLlamaShape.attentionHeadSize,
            ...(inputHidden === null ? {} : { inputHidden }),
            epsilon: packedTinyLlamaProof.epsilon,
            layer,
            ropeBase: packedTinyLlamaProof.ropeBase,
            ropeHighFreqFactor: expectedGroupedRopeHighFreqFactor,
            ropeKind: expectedGroupedRopeKind,
            ropeLowFreqFactor: expectedGroupedRopeLowFreqFactor,
            ropeOriginalContextLength: expectedGroupedRopeOriginalContextLength,
            ropeScale: expectedGroupedRopeScale,
            slidingWindow: packedTinyLlamaProof.slidingWindow ?? null,
            useRope: expectedGroupedUseRopeLayers[layer] !== 0,
          });
          inputHidden = expected.blockHidden;
          tokenLayers.push(expected);
        }
        return tokenLayers;
      };
      for (let tokenIndex = 0; tokenIndex < groupedPrefillTokens.length; tokenIndex += 1) {
        expectedGrouped.push(runExpectedGroupedToken(groupedPrefillTokens[tokenIndex], tokenIndex));
      }
      for (let tokenIndex = 0; tokenIndex < groupedDecodeTokens.length; tokenIndex += 1) {
        const position = groupedPrefillTokens.length + tokenIndex;
        expectedGrouped.push(runExpectedGroupedToken(groupedDecodeTokens[tokenIndex], position));
      }
      const expectedGroupedLast = expectedGrouped.at(-1).at(-1);
      const groupedTolerance = groupedTinyLlamaShape.attentionHeadSize >= 128 ? 1e-3 : 1e-5;
      expectClose(
        Array.from(await hostDevice.readFloat32(groupedPipelineSession.bindings.output, groupedTinyLlamaShape.vocabSize)),
        expectedGroupedLast.logits,
        groupedTolerance,
      );
      for (let layer = 0; layer < groupedTinyLlamaShape.layers; layer += 1) {
        for (let tokenIndex = 0; tokenIndex < expectedGrouped.length; tokenIndex += 1) {
          const expected = expectedGrouped[tokenIndex][layer];
          const offset = tokenIndex * expected.k.length * Float32Array.BYTES_PER_ELEMENT;
          expectClose(Array.from(await hostDevice.readFloat32(groupedCaches[layer].k, expected.k.length, offset)), expected.k, groupedTolerance);
          expectClose(Array.from(await hostDevice.readFloat32(groupedCaches[layer].v, expected.v.length, offset)), expected.v, groupedTolerance);
        }
      }
      const expectedGroupedPrefillCommandCount = expectedLlamaBlockNoOutputPrefillCommandCount(
        hostDevice,
        groupedTinyLlamaShape.layers,
        groupedPrefillTokens.length,
        {
          contextLength: groupedTinyLlamaShape.contextLength,
          hiddenSize: groupedTinyLlamaShape.hiddenSize,
          slidingWindow: packedTinyLlamaProof.slidingWindow ?? 0,
        },
      );
      const expectedGroupedPrefillDispatchFamilies = expectedLlamaBlockPipelineDispatchFamilies(
        hostDevice,
        groupedTinyLlamaShape.layers,
        groupedPrefillTokens.length,
        executeOutputNone,
        {
          contextLength: groupedTinyLlamaShape.contextLength,
          hiddenSize: groupedTinyLlamaShape.hiddenSize,
          slidingWindow: packedTinyLlamaProof.slidingWindow ?? 0,
        },
      );
      const expectedGroupedDecodeCommandCount = expectedLlamaBlockPipelineCommandCount(
        hostDevice,
        groupedTinyLlamaShape.layers,
        groupedDecodeTokens.length,
        executeOutputLogits,
        {
          contextLength: groupedTinyLlamaShape.contextLength,
          hiddenSize: groupedTinyLlamaShape.hiddenSize,
          slidingWindow: packedTinyLlamaProof.slidingWindow ?? 0,
          splitTerminalLogits: groupedTinyLlamaShape.hiddenSize >= 64,
          startPosition: groupedPrefillTokens.length,
        },
      );
      const expectedGroupedDecodeDispatchFamilies = expectedLlamaBlockPipelineDispatchFamilies(
        hostDevice,
        groupedTinyLlamaShape.layers,
        groupedDecodeTokens.length,
        executeOutputLogits,
        {
          contextLength: groupedTinyLlamaShape.contextLength,
          hiddenSize: groupedTinyLlamaShape.hiddenSize,
          slidingWindow: packedTinyLlamaProof.slidingWindow ?? 0,
          splitTerminalLogits: groupedTinyLlamaShape.hiddenSize >= 64,
          startPosition: groupedPrefillTokens.length,
        },
      );
      const expectedGroupedCommandCount = expectedGroupedPrefillCommandCount + expectedGroupedDecodeCommandCount;
      const expectedGroupedBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? expectedGroupedCommandCount : 0;
      const expectedGroupedFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : expectedGroupedCommandCount;
      const expectedGroupedUsesGpuStorageBuffers = canUseGpuStorageBuffers(hostDevice, 6);
      const expectedGroupedGenericDispatchCount =
        expectedGroupedPrefillDispatchFamilies.generic + expectedGroupedDecodeDispatchFamilies.generic;
      const expectedGroupedScalarDispatchCount =
        expectedGroupedPrefillDispatchFamilies.scalar + expectedGroupedDecodeDispatchFamilies.scalar;
      const expectedGroupedWindowDispatchCount =
        expectedGroupedPrefillDispatchFamilies.window + expectedGroupedDecodeDispatchFamilies.window;
      const groupedProfile = groupedPipelineSession.runtimeProfile();
      if (
        groupedProfile.callCount !== 2 ||
        groupedProfile.noOutputCallCount !== 1 ||
        groupedProfile.logitsOutputCallCount !== 1 ||
        groupedProfile.executorBackendDispatchCount !== expectedGroupedBackendDispatchCount ||
        groupedProfile.executorFallbackOpCount !== expectedGroupedFallbackOpCount ||
        groupedProfile.executorCommandCount !== expectedGroupedCommandCount ||
        groupedProfile.executorCommandOpCount !== expectedGroupedCommandCount ||
        groupedProfile.executorGenericDispatchCount !== expectedGroupedGenericDispatchCount ||
        groupedProfile.executorScalarDispatchCount !== expectedGroupedScalarDispatchCount ||
        groupedProfile.executorWindowDispatchCount !== expectedGroupedWindowDispatchCount ||
        groupedProfile.lastStatus !== ok ||
        groupedProfile.lastResultOutputLength !== groupedTinyLlamaShape.vocabSize
      ) {
        throw new Error(`unexpected host-only WebGPU grouped LLaMA pipeline profile: ${JSON.stringify(groupedProfile)}`);
      }
      recordBrowserLlamaProfileEvidence(packedTinyLlamaProof.label, groupedProfile);
      if (
        groupedPipelineExecutorCalls.length !== 2 ||
        groupedPipelineExecutorCalls[0].stageCount !== groupedTinyLlamaShape.layers ||
        groupedPipelineExecutorCalls[0].commandCount !== expectedGroupedPrefillCommandCount ||
        groupedPipelineExecutorCalls[0].deviceMode !== expectedLlamaExecutorDeviceMode(hostDevice) ||
        groupedPipelineExecutorCalls[0].genericDispatchCount !== expectedGroupedPrefillDispatchFamilies.generic ||
        groupedPipelineExecutorCalls[0].maxStorageBuffersPerShaderStage !== hostDevice.maxStorageBuffersPerShaderStage() ||
        groupedPipelineExecutorCalls[0].outputPolicy !== executeOutputNone ||
        groupedPipelineExecutorCalls[0].requiredStorageBufferCount !== 6 ||
        groupedPipelineExecutorCalls[0].scalarDispatchCount !== expectedGroupedPrefillDispatchFamilies.scalar ||
        groupedPipelineExecutorCalls[0].ffnSize !== groupedTinyLlamaShape.ffnSize ||
        groupedPipelineExecutorCalls[0].usesGpuStorageBuffers !== expectedGroupedUsesGpuStorageBuffers ||
        groupedPipelineExecutorCalls[0].windowDispatchCount !== expectedGroupedPrefillDispatchFamilies.window ||
        groupedPipelineExecutorCalls[1].stageCount !== groupedTinyLlamaShape.layers ||
        groupedPipelineExecutorCalls[1].commandCount !== expectedGroupedDecodeCommandCount ||
        groupedPipelineExecutorCalls[1].deviceMode !== expectedLlamaExecutorDeviceMode(hostDevice) ||
        groupedPipelineExecutorCalls[1].genericDispatchCount !== expectedGroupedDecodeDispatchFamilies.generic ||
        groupedPipelineExecutorCalls[1].maxStorageBuffersPerShaderStage !== hostDevice.maxStorageBuffersPerShaderStage() ||
        groupedPipelineExecutorCalls[1].outputPolicy !== executeOutputLogits ||
        groupedPipelineExecutorCalls[1].outputLength !== groupedTinyLlamaShape.vocabSize ||
        groupedPipelineExecutorCalls[1].requiredStorageBufferCount !== 6 ||
        groupedPipelineExecutorCalls[1].scalarDispatchCount !== expectedGroupedDecodeDispatchFamilies.scalar ||
        groupedPipelineExecutorCalls[1].ffnSize !== groupedTinyLlamaShape.ffnSize ||
        groupedPipelineExecutorCalls[1].hiddenSize !== groupedTinyLlamaShape.hiddenSize ||
        groupedPipelineExecutorCalls[1].usesGpuStorageBuffers !== expectedGroupedUsesGpuStorageBuffers ||
        groupedPipelineExecutorCalls[1].windowDispatchCount !== expectedGroupedDecodeDispatchFamilies.window
      ) {
        throw new Error("host-only WebGPU grouped LLaMA block pipeline evidence mismatch");
      }
      const groupedGreedyCallStart = groupedPipelineExecutorCalls.length;
      markBrowserSmokeStage(`tiny-llama-webgpu-greedy-${packedTinyLlamaProof.label}`);
      const groupedGreedySession = groupedPipelineProgram.bindHostResources();
      const groupedGreedyHandle = groupedGreedySession.handle;
      try {
        const groupedGreedyCaches = groupedGreedySession.bindings.kvCache.map((layer, index) => {
          const kElements = Math.floor(layer.k.byteLength / Float32Array.BYTES_PER_ELEMENT);
          const vElements = Math.floor(layer.v.byteLength / Float32Array.BYTES_PER_ELEMENT);
          hostDevice.writeFloat32(layer.k, Array(kElements).fill(-8300 - index * 100));
          hostDevice.writeFloat32(layer.v, Array(vElements).fill(-8400 - index * 100));
          return { k: layer.k, kElements, v: layer.v, vElements };
        });
        hostDevice.writeFloat32(groupedGreedySession.bindings.output, Array(groupedTinyLlamaShape.vocabSize).fill(-8360));
        const groupedGreedyPrompt = [groupedPrefillTokens[0] ?? 0];
        const expectedGroupedGreedyCaches = groupedGreedyCaches.map((cache, index) => ({
          k: Array(cache.kElements).fill(-8300 - index * 100),
          v: Array(cache.vElements).fill(-8400 - index * 100),
        }));
        const runExpectedGroupedGreedyToken = (token, position) => {
          let inputHidden = null;
          const tokenLayers = [];
          for (let layer = 0; layer < groupedTinyLlamaShape.layers; layer += 1) {
            const expected = tinyLlamaBlockProjectionStep(groupedModelResourceBytes, expectedGroupedGreedyCaches[layer], token, position, {
              attentionHeadSize: groupedTinyLlamaShape.attentionHeadSize,
              ...(inputHidden === null ? {} : { inputHidden }),
              epsilon: packedTinyLlamaProof.epsilon,
              layer,
              ropeBase: packedTinyLlamaProof.ropeBase,
              ropeHighFreqFactor: expectedGroupedRopeHighFreqFactor,
              ropeKind: expectedGroupedRopeKind,
              ropeLowFreqFactor: expectedGroupedRopeLowFreqFactor,
              ropeOriginalContextLength: expectedGroupedRopeOriginalContextLength,
              ropeScale: expectedGroupedRopeScale,
              slidingWindow: packedTinyLlamaProof.slidingWindow ?? null,
              useRope: expectedGroupedUseRopeLayers[layer] !== 0,
            });
            inputHidden = expected.blockHidden;
            tokenLayers.push(expected);
          }
          return tokenLayers;
        };
        const expectedGroupedGreedy = [];
        const groupedGreedyFirstLayers = runExpectedGroupedGreedyToken(groupedGreedyPrompt[0], 0);
        expectedGroupedGreedy.push(groupedGreedyFirstLayers);
        const expectedGroupedGreedyFirst = argmaxOf(groupedGreedyFirstLayers.at(-1).logits);
        const groupedGreedySecondLayers = runExpectedGroupedGreedyToken(expectedGroupedGreedyFirst.token, 1);
        expectedGroupedGreedy.push(groupedGreedySecondLayers);
        const expectedGroupedGreedySecond = argmaxOf(groupedGreedySecondLayers.at(-1).logits);
        const groupedGreedy = await groupedGreedySession.generateTokensArgmax(groupedGreedyPrompt, 2, { includeLogits: true });
        expectStatus(groupedGreedy.status, ok, `host-only WebGPU grouped LLaMA ${packedTinyLlamaProof.label} ergonomic greedy generation`);
        if (
          !(groupedGreedy.tokens instanceof Uint32Array) ||
          groupedGreedy.tokensGenerated !== 2 ||
          groupedGreedy.tokens[0] !== expectedGroupedGreedyFirst.token ||
          groupedGreedy.tokens[1] !== expectedGroupedGreedySecond.token ||
          groupedGreedy.lastToken !== expectedGroupedGreedySecond.token ||
          Math.abs(groupedGreedy.lastLogit - expectedGroupedGreedySecond.logit) > groupedTolerance ||
          groupedGreedy.position !== 2 ||
          sessionPositionWith(groupedHostExports, groupedGreedyHandle) !== 2 ||
          !(groupedGreedy.logits instanceof Float32Array)
        ) {
          throw new Error(`host-only WebGPU grouped LLaMA ${packedTinyLlamaProof.label} ergonomic greedy generation returned unexpected tokens`);
        }
        expectClose(Array.from(groupedGreedy.logits), groupedGreedySecondLayers.at(-1).logits, groupedTolerance);
        expectClose(
          Array.from(await hostDevice.readFloat32(groupedGreedySession.bindings.output, groupedTinyLlamaShape.vocabSize)),
          groupedGreedySecondLayers.at(-1).logits,
          groupedTolerance,
        );
        for (let layer = 0; layer < groupedTinyLlamaShape.layers; layer += 1) {
          for (let tokenIndex = 0; tokenIndex < expectedGroupedGreedy.length; tokenIndex += 1) {
            const expected = expectedGroupedGreedy[tokenIndex][layer];
            const offset = tokenIndex * expected.k.length * Float32Array.BYTES_PER_ELEMENT;
            expectClose(Array.from(await hostDevice.readFloat32(groupedGreedyCaches[layer].k, expected.k.length, offset)), expected.k, groupedTolerance);
            expectClose(Array.from(await hostDevice.readFloat32(groupedGreedyCaches[layer].v, expected.v.length, offset)), expected.v, groupedTolerance);
          }
        }
        const expectedGroupedGreedyFirstCommandCount = expectedLlamaBlockPipelineCommandCount(
          hostDevice,
          groupedTinyLlamaShape.layers,
          1,
          executeOutputLogits,
          {
            contextLength: groupedTinyLlamaShape.contextLength,
            hiddenSize: groupedTinyLlamaShape.hiddenSize,
            slidingWindow: packedTinyLlamaProof.slidingWindow ?? 0,
            splitTerminalLogits: groupedTinyLlamaShape.hiddenSize >= 64,
            startPosition: 0,
          },
        );
        const expectedGroupedGreedySecondCommandCount = expectedLlamaBlockPipelineCommandCount(
          hostDevice,
          groupedTinyLlamaShape.layers,
          1,
          executeOutputLogits,
          {
            contextLength: groupedTinyLlamaShape.contextLength,
            hiddenSize: groupedTinyLlamaShape.hiddenSize,
            slidingWindow: packedTinyLlamaProof.slidingWindow ?? 0,
            splitTerminalLogits: groupedTinyLlamaShape.hiddenSize >= 64,
            startPosition: 1,
          },
        );
        const expectedGroupedGreedyCommandCount = expectedGroupedGreedyFirstCommandCount + expectedGroupedGreedySecondCommandCount;
        const expectedGroupedGreedyBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? expectedGroupedGreedyCommandCount : 0;
        const expectedGroupedGreedyFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : expectedGroupedGreedyCommandCount;
        const groupedGreedyProfile = groupedGreedySession.runtimeProfile();
        if (
          groupedGreedyProfile.callCount !== 2 ||
          groupedGreedyProfile.decodedCallCount !== 2 ||
          groupedGreedyProfile.logitsOutputCallCount !== 2 ||
          groupedGreedyProfile.boundOutputCallCount !== 2 ||
          groupedGreedyProfile.executorCallCount !== 2 ||
          groupedGreedyProfile.executorOkCount !== 2 ||
          groupedGreedyProfile.executorBackendDispatchCount !== expectedGroupedGreedyBackendDispatchCount ||
          groupedGreedyProfile.executorFallbackOpCount !== expectedGroupedGreedyFallbackOpCount ||
          groupedGreedyProfile.executorCommandCount !== expectedGroupedGreedyCommandCount ||
          groupedGreedyProfile.executorCommandOpCount !== expectedGroupedGreedyCommandCount ||
          groupedGreedyProfile.outputReadCount !== 2 ||
          groupedGreedyProfile.syncCount !== 2 ||
          groupedGreedyProfile.lastStatus !== ok ||
          groupedGreedyProfile.lastResultOutputLength !== groupedTinyLlamaShape.vocabSize
        ) {
          throw new Error(`unexpected host-only WebGPU grouped LLaMA ${packedTinyLlamaProof.label} ergonomic greedy profile: ${JSON.stringify(groupedGreedyProfile)}`);
        }
        recordBrowserLlamaProfileEvidence(`greedy-${packedTinyLlamaProof.label}`, groupedGreedyProfile);
        const groupedGreedyAbiProfile = sessionRuntimeProfileWith(groupedHostExports, groupedGreedyHandle);
        if (
          u64(groupedGreedyAbiProfile, 0) !== 2n ||
          u64(groupedGreedyAbiProfile, 8) !== BigInt(expectedGroupedGreedyBackendDispatchCount) ||
          u64(groupedGreedyAbiProfile, 16) !== BigInt(expectedGroupedGreedyFallbackOpCount) ||
          u64(groupedGreedyAbiProfile, 24) !== BigInt(expectedGroupedGreedyBackendDispatchCount) ||
          u64(groupedGreedyAbiProfile, 32) !== 2n ||
          u64(groupedGreedyAbiProfile, 40) !== 2n ||
          u64(groupedGreedyAbiProfile, 48) !== 2n ||
          u64(groupedGreedyAbiProfile, 56) !== 0n ||
          u64(groupedGreedyAbiProfile, 96) !== BigInt(expectedGroupedGreedyCommandCount) ||
          u64(groupedGreedyAbiProfile, 112) !== BigInt(expectedGroupedGreedyCommandCount)
        ) {
          throw new Error(`host-only WebGPU grouped LLaMA ${packedTinyLlamaProof.label} ergonomic greedy ABI profile mismatch`);
        }
        const groupedGreedyCalls = groupedPipelineExecutorCalls.slice(groupedGreedyCallStart);
        if (
          groupedGreedyCalls.length !== 2 ||
          groupedGreedyCalls[0].outputPolicy !== executeOutputLogits ||
          groupedGreedyCalls[0].tokenCount !== 1 ||
          groupedGreedyCalls[0].commandCount !== expectedGroupedGreedyFirstCommandCount ||
          groupedGreedyCalls[1].outputPolicy !== executeOutputLogits ||
          groupedGreedyCalls[1].tokenCount !== 1 ||
          groupedGreedyCalls[1].commandCount !== expectedGroupedGreedySecondCommandCount ||
          groupedGreedyCalls[1].outputLength !== groupedTinyLlamaShape.vocabSize
        ) {
          throw new Error(`host-only WebGPU grouped LLaMA ${packedTinyLlamaProof.label} ergonomic greedy executor evidence mismatch`);
        }
      } finally {
        expectStatus(groupedHostExports.zgml_session_free(groupedGreedyHandle), ok, `host-only WebGPU grouped LLaMA ${packedTinyLlamaProof.label} ergonomic greedy Session free`);
      }
      } finally {
        expectStatus(groupedHostExports.zgml_session_free(groupedPipelineSessionHandle), ok, "host-only WebGPU grouped LLaMA Session free");
        groupedPipelineProgram.tokenExecutor?.destroy?.();
      }
    }

    if (!browserFocusedLlamaProfileLabels) {
    const ropeConflictBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
      metadata: {
        num_attention_heads: "4",
        num_key_value_heads: "4",
        rope_theta: "11000",
      },
    });
    let ropeConflictRejected = false;
    try {
      WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        config: { rope_theta: 12000 },
        contextLength: mhaTinyLlamaShape.contextLength,
        safetensors: ropeConflictBytes,
      });
    } catch (err) {
      if (!String(err?.message ?? err).includes("RoPE base mismatch")) throw err;
      ropeConflictRejected = true;
    }
    if (!ropeConflictRejected) {
      throw new Error("host-only WebGPU LLaMA safetensors Program accepted conflicting RoPE base sources");
    }

    const epsilonConflictBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
      metadata: {
        num_attention_heads: "4",
        num_key_value_heads: "4",
        rms_norm_eps: "0.00002",
        rope_theta: "11000",
      },
    });
    let epsilonConflictRejected = false;
    try {
      WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        config: { rms_norm_eps: 0.00003, rope_theta: 11000 },
        contextLength: mhaTinyLlamaShape.contextLength,
        safetensors: epsilonConflictBytes,
      });
    } catch (err) {
      if (!String(err?.message ?? err).includes("RMSNorm epsilon mismatch")) throw err;
      epsilonConflictRejected = true;
    }
    if (!epsilonConflictRejected) {
      throw new Error("host-only WebGPU LLaMA safetensors Program accepted conflicting RMSNorm epsilon sources");
    }

    const intermediateConflictBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
      metadata: {
        num_attention_heads: "4",
        num_key_value_heads: "4",
      },
    });
    let intermediateConflictRejected = false;
    try {
      WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        config: {
          intermediate_size: mhaTinyLlamaShape.ffnSize + 1,
          num_attention_heads: 4,
          num_key_value_heads: 4,
        },
        contextLength: mhaTinyLlamaShape.contextLength,
        safetensors: intermediateConflictBytes,
      });
    } catch (err) {
      if (!String(err?.message ?? err).includes("FFN intermediate size mismatch")) throw err;
      intermediateConflictRejected = true;
    }
    if (!intermediateConflictRejected) {
      throw new Error("host-only WebGPU LLaMA safetensors Program accepted conflicting FFN intermediate size sources");
    }

    const metadataConfigConflictBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
      metadata: {
        config: JSON.stringify({
          hidden_size: mhaTinyLlamaShape.hiddenSize,
          intermediate_size: mhaTinyLlamaShape.ffnSize,
          num_attention_heads: 4,
          num_hidden_layers: mhaTinyLlamaShape.layers,
          num_key_value_heads: 4,
          rope_theta: 11000,
          vocab_size: mhaTinyLlamaShape.vocabSize,
        }),
      },
    });
    let metadataConfigConflictRejected = false;
    try {
      WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        config: { rope_theta: 12000 },
        contextLength: mhaTinyLlamaShape.contextLength,
        safetensors: metadataConfigConflictBytes,
      });
    } catch (err) {
      if (!String(err?.message ?? err).includes("config mismatch for rope_theta")) throw err;
      metadataConfigConflictRejected = true;
    }
    if (!metadataConfigConflictRejected) {
      throw new Error("host-only WebGPU LLaMA safetensors Program accepted conflicting embedded metadata config");
    }

	    const metadataConfigSemanticAgreementBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
	      metadata: {
	        attention_dropout: "0",
	        eos_token_id: "2",
	        hidden_act: "silu",
	        num_attention_heads: "4",
	        num_key_value_heads: "4",
	        pretraining_tp: "2",
	        qk_norm: "false",
	        torch_dtype: "float32",
	        use_cache: "true",
	      },
	    });
    const metadataConfigSemanticAgreementProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      config: {
	        attention_dropout: 0,
	        eos_token_id: 2,
	        hidden_act: "swish",
	        pretraining_tp: 2,
	        qk_norm: false,
	        torch_dtype: "torch.float32",
	        use_cache: true,
	      },
      contextLength: mhaTinyLlamaShape.contextLength,
      safetensors: metadataConfigSemanticAgreementBytes,
    });
    metadataConfigSemanticAgreementProgram.tokenExecutor?.destroy?.();

    for (const contextInferenceProof of [
      {
        label: "metadata",
        metadata: {
          max_position_embeddings: String(mhaTinyLlamaShape.contextLength),
          num_attention_heads: "4",
          num_key_value_heads: "4",
        },
        options: {},
      },
      {
        label: "config",
        metadata: {},
        options: {
          config: {
            max_position_embeddings: mhaTinyLlamaShape.contextLength,
            num_attention_heads: 4,
            num_key_value_heads: 4,
          },
        },
      },
    ]) {
      const contextInferenceBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
        metadata: contextInferenceProof.metadata,
      });
      const contextInferenceShape = WasmWebGpuLlamaResourceProgram.inferSafetensorsShape(contextInferenceBytes, contextInferenceProof.options);
      if (
        contextInferenceShape.contextLength !== mhaTinyLlamaShape.contextLength ||
        contextInferenceShape.kvCacheRequirements.kBufferByteLength !== mhaTinyLlamaShape.contextLength * mhaTinyLlamaShape.kvSize * Float32Array.BYTES_PER_ELEMENT
      ) {
        throw new Error(`host-only WebGPU LLaMA safetensors ${contextInferenceProof.label} context inference shape mismatch`);
      }
      const contextInferenceProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        safetensors: contextInferenceBytes,
        ...contextInferenceProof.options,
      });
      if (
        contextInferenceProgram.contextLength !== mhaTinyLlamaShape.contextLength ||
        contextInferenceProgram.kvRequirements.kBufferByteLength !== mhaTinyLlamaShape.contextLength * mhaTinyLlamaShape.kvSize * Float32Array.BYTES_PER_ELEMENT
      ) {
        throw new Error(`host-only WebGPU LLaMA safetensors ${contextInferenceProof.label} inferred Program context mismatch`);
      }
      contextInferenceProgram.tokenExecutor?.destroy?.();
    }

    const explicitContextOverrideBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs);
    const explicitContextOverrideProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      config: {
        max_position_embeddings: mhaTinyLlamaShape.contextLength,
        num_attention_heads: 4,
        num_key_value_heads: 4,
      },
      contextLength: mhaTinyLlamaShape.contextLength - 1,
      safetensors: explicitContextOverrideBytes,
    });
    if (
      explicitContextOverrideProgram.contextLength !== mhaTinyLlamaShape.contextLength - 1 ||
      explicitContextOverrideProgram.kvRequirements.kBufferByteLength !== (mhaTinyLlamaShape.contextLength - 1) * mhaTinyLlamaShape.kvSize * Float32Array.BYTES_PER_ELEMENT
    ) {
      throw new Error("host-only WebGPU LLaMA safetensors explicit context override mismatch");
    }
    explicitContextOverrideProgram.tokenExecutor?.destroy?.();

    const inferredContextMismatchBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
      metadata: {
        max_position_embeddings: String(mhaTinyLlamaShape.contextLength),
        num_attention_heads: "4",
        num_key_value_heads: "4",
      },
    });
    let inferredContextMismatchRejected = false;
    try {
      WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        config: {
          max_position_embeddings: mhaTinyLlamaShape.contextLength + 1,
          num_attention_heads: 4,
          num_key_value_heads: 4,
        },
        safetensors: inferredContextMismatchBytes,
      });
    } catch (err) {
      if (!String(err?.message ?? err).includes("maximum position embeddings mismatch")) throw err;
      inferredContextMismatchRejected = true;
    }
    if (!inferredContextMismatchRejected) {
      throw new Error("host-only WebGPU LLaMA safetensors accepted conflicting inferred context sources");
    }

    const customLmHeadBiasTensor = "custom.lm_head.bias";
    const customLmHeadBiasBytes = tinyLlamaModelResourceFileBytes([
      ...tinyLlamaTensorSpecs,
      [customLmHeadBiasTensor, [tinyLlamaVocabSize]],
    ], {
      metadata: {
        max_position_embeddings: "4",
      },
    });
    const customLmHeadBiasProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      lmHeadBiasTensor: customLmHeadBiasTensor,
      program,
      safetensors: customLmHeadBiasBytes,
      allowMockFallback: true,
    });
    if (!customLmHeadBiasProgram.requiredModelResources.some((entry) => entry.role === `llama.weight.${customLmHeadBiasTensor}`)) {
      throw new Error("host-only WebGPU LLaMA safetensors Program did not retain custom lm_head bias role");
    }
    if (
      customLmHeadBiasProgram.tokenExecutor.blocks[0].executor.lmHeadBiasRole !== customLmHeadBiasTensor ||
      customLmHeadBiasProgram.tokenExecutor.finalLogitsExecutor.lmHeadBiasRole !== customLmHeadBiasTensor
    ) {
      throw new Error("host-only WebGPU LLaMA safetensors Program did not derive executor custom lm_head bias role");
    }
    const customLmHeadBiasSession = customLmHeadBiasProgram.bindHostResources();
    try {
      if (!customLmHeadBiasSession.modelResources.some((entry) => entry.role === `llama.weight.${customLmHeadBiasTensor}`)) {
        throw new Error("host-only WebGPU LLaMA safetensors default bind did not materialize custom lm_head bias resource");
      }
      await expectTinyLlamaModelResourceBytes(
        hostDevice,
        customLmHeadBiasSession.modelResources,
        customLmHeadBiasBytes,
        "host-only WebGPU LLaMA custom lm_head bias default safetensors",
      );
      const customLmHeadBiasHostExports = hostRuntime.wrapExports(exportsRef);
      const customLmHeadBiasK = customLmHeadBiasSession.bindings.kvCache[0].k;
      const customLmHeadBiasV = customLmHeadBiasSession.bindings.kvCache[0].v;
      const customLmHeadBiasKElements = Math.floor(customLmHeadBiasK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const customLmHeadBiasVElements = Math.floor(customLmHeadBiasV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(customLmHeadBiasK, Array(customLmHeadBiasKElements).fill(0));
      hostDevice.writeFloat32(customLmHeadBiasV, Array(customLmHeadBiasVElements).fill(0));
      hostDevice.writeFloat32(customLmHeadBiasSession.bindings.output, Array(tinyLlamaVocabSize).fill(-7170));
      const customLmHeadBiasTokens = keep(alloc(4), 4);
      writeU32Array(customLmHeadBiasTokens, [2]);
      const customLmHeadBiasResult = tokenExecuteWith(customLmHeadBiasHostExports, customLmHeadBiasSession.handle, customLmHeadBiasTokens, 1, executeOutputLogits, 0, 0);
      expectStatus(customLmHeadBiasResult.status, ok, "host-only WebGPU LLaMA custom lm_head bias execute");
      const customLmHeadBiasBase = tinyLlamaBlockProjectionStep(
        customLmHeadBiasBytes,
        {
          k: Array(customLmHeadBiasKElements).fill(0),
          v: Array(customLmHeadBiasVElements).fill(0),
        },
        2,
        0,
        { modelResourceOptions: { lmHeadBiasTensor: customLmHeadBiasTensor } },
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(customLmHeadBiasSession.bindings.output, tinyLlamaVocabSize)),
        customLmHeadBiasBase.logits,
      );
    } finally {
      customLmHeadBiasProgram.unregister(customLmHeadBiasSession);
      customLmHeadBiasSession.dispose();
      customLmHeadBiasProgram.tokenExecutor?.destroy?.();
    }

    const customEmbeddingTensor = "custom.embed_tokens.weight";
    const customNormTensor = "custom.norm.weight";
    const customLmHeadTensor = "custom.lm_head.weight";
    const customRootTensorSpecs = tinyLlamaTensorSpecs.map(([name, shape]) => {
      if (name === "model.embed_tokens.weight") return [customEmbeddingTensor, shape];
      if (name === "model.norm.weight") return [customNormTensor, shape];
      if (name === "lm_head.weight") return [customLmHeadTensor, shape];
      return [name, shape];
    });
    const customRootBytes = tinyLlamaModelResourceFileBytes(customRootTensorSpecs, {
      metadata: {
        max_position_embeddings: "4",
      },
    });
    const customRootProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      embeddingTensor: customEmbeddingTensor,
      normTensor: customNormTensor,
      lmHeadTensor: customLmHeadTensor,
      program,
      safetensors: customRootBytes,
      allowMockFallback: true,
    });
    for (const customRootTensor of [customEmbeddingTensor, customNormTensor, customLmHeadTensor]) {
      if (!customRootProgram.requiredModelResources.some((entry) => entry.role === `llama.weight.${customRootTensor}`)) {
        throw new Error(`host-only WebGPU LLaMA safetensors Program did not retain custom root role ${customRootTensor}`);
      }
    }
    if (
      customRootProgram.tokenExecutor.blocks[0].executor.embeddingRole !== customEmbeddingTensor ||
      customRootProgram.tokenExecutor.blocks[0].executor.normRole !== customNormTensor ||
      customRootProgram.tokenExecutor.blocks[0].executor.lmHeadRole !== customLmHeadTensor ||
      customRootProgram.tokenExecutor.finalLogitsExecutor.normRole !== customNormTensor ||
      customRootProgram.tokenExecutor.finalLogitsExecutor.lmHeadRole !== customLmHeadTensor
    ) {
      throw new Error("host-only WebGPU LLaMA safetensors Program did not derive executor custom root roles");
    }
    const customRootSession = customRootProgram.bindHostResources();
    try {
      for (const customRootTensor of [customEmbeddingTensor, customNormTensor, customLmHeadTensor]) {
        if (!customRootSession.modelResources.some((entry) => entry.role === `llama.weight.${customRootTensor}`)) {
          throw new Error(`host-only WebGPU LLaMA safetensors default bind did not materialize custom root resource ${customRootTensor}`);
        }
      }
      await expectTinyLlamaModelResourceBytes(
        hostDevice,
        customRootSession.modelResources,
        customRootBytes,
        "host-only WebGPU LLaMA custom root default safetensors",
      );
      const customRootHostExports = hostRuntime.wrapExports(exportsRef);
      const customRootK = customRootSession.bindings.kvCache[0].k;
      const customRootV = customRootSession.bindings.kvCache[0].v;
      const customRootKElements = Math.floor(customRootK.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const customRootVElements = Math.floor(customRootV.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(customRootK, Array(customRootKElements).fill(0));
      hostDevice.writeFloat32(customRootV, Array(customRootVElements).fill(0));
      hostDevice.writeFloat32(customRootSession.bindings.output, Array(tinyLlamaVocabSize).fill(-8180));
      const customRootTokens = keep(alloc(4), 4);
      writeU32Array(customRootTokens, [2]);
      const customRootResult = tokenExecuteWith(customRootHostExports, customRootSession.handle, customRootTokens, 1, executeOutputLogits, 0, 0);
      expectStatus(customRootResult.status, ok, "host-only WebGPU LLaMA custom root execute");
      const customRootExpected = tinyLlamaBlockProjectionStep(
        customRootBytes,
        {
          k: Array(customRootKElements).fill(0),
          v: Array(customRootVElements).fill(0),
        },
        2,
        0,
        {
          modelResourceOptions: {
            embeddingTensor: customEmbeddingTensor,
            normTensor: customNormTensor,
            lmHeadTensor: customLmHeadTensor,
          },
        },
      );
      expectClose(
        Array.from(await hostDevice.readFloat32(customRootSession.bindings.output, tinyLlamaVocabSize)),
        customRootExpected.logits,
      );
    } finally {
      customRootProgram.unregister(customRootSession);
      customRootSession.dispose();
      customRootProgram.tokenExecutor?.destroy?.();
    }

    for (const metadataConfigSemanticConflictProof of [
      {
        config: { model_type: "mistral" },
        label: "model type",
        message: "model type mismatch",
        metadata: { model_type: "llama" },
      },
      {
        config: { model_type: "llama" },
        label: "general architecture",
        message: "model type mismatch",
        metadata: { "general.architecture": "qwen2" },
      },
      {
        config: { architectures: ["MistralForCausalLM"] },
        label: "architectures",
        message: "architectures mismatch",
        metadata: { architectures: ["LlamaForCausalLM"] },
      },
      {
        config: { torch_dtype: "bfloat16" },
        label: "torch dtype",
        message: "torch dtype mismatch",
        metadata: { torch_dtype: "float32" },
      },
      {
        config: { use_cache: false },
        label: "use_cache",
        message: "use_cache mismatch",
        metadata: { use_cache: "true" },
      },
      {
        config: { eos_token_id: [3] },
        label: "eos token id",
        message: "eos_token_id mismatch",
        metadata: { eos_token_id: "2" },
      },
      {
        config: { attention_dropout: 0.25 },
        label: "attention dropout",
        message: "attention_dropout mismatch",
        metadata: { attention_dropout: "0" },
      },
      {
        config: { pretraining_tp: 3 },
        label: "pretraining tensor parallelism",
        message: "pretraining tensor parallelism mismatch",
        metadata: { pretraining_tp: "2" },
      },
	      {
	        config: { no_rope_layer_interval: 2 },
	        label: "no-RoPE layer interval",
	        message: "no_rope_layer_interval mismatch",
	        metadata: { no_rope_layer_interval: "4" },
	      },
	      {
	        config: { qk_norm: false },
	        label: "Q/K projection normalization",
	        message: "Q/K projection normalization mismatch",
	        metadata: { qk_norm: "true" },
	      },
	    ]) {
      const metadataConfigSemanticConflictBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
        metadata: {
          num_attention_heads: "4",
          num_key_value_heads: "4",
          ...(metadataConfigSemanticConflictProof.metadata ?? {}),
        },
      });
      let metadataConfigSemanticConflictRejected = false;
      try {
        WasmWebGpuLlamaResourceProgram.fromSafetensors({
          device: hostDevice,
          resourceBridge: hostResourceBridge,
          sessionBindingBridge,
          runtime: hostRuntime,
          program,
          config: metadataConfigSemanticConflictProof.config,
          contextLength: mhaTinyLlamaShape.contextLength,
          safetensors: metadataConfigSemanticConflictBytes,
        });
      } catch (err) {
        if (!String(err?.message ?? err).includes(metadataConfigSemanticConflictProof.message)) throw err;
        metadataConfigSemanticConflictRejected = true;
      }
      if (!metadataConfigSemanticConflictRejected) {
        throw new Error(`host-only WebGPU LLaMA safetensors Program accepted conflicting direct metadata/config ${metadataConfigSemanticConflictProof.label}`);
      }
    }

    for (const contextEnvelopeConflictProof of [
      {
        config: {},
        label: "short metadata max context",
        message: "exceeds maximum position embeddings",
        metadata: {
          max_position_embeddings: "2",
        },
      },
      {
        config: { max_position_embeddings: 16 },
        label: "metadata/config max context",
        message: "maximum position embeddings mismatch",
        metadata: {
          max_position_embeddings: "32",
        },
      },
      {
        config: {},
        label: "metadata original context beyond max context",
        message: "original max position embeddings",
        metadata: {
          max_position_embeddings: "16",
          original_max_position_embeddings: "32",
        },
      },
    ]) {
      const contextEnvelopeConflictBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
        metadata: {
          num_attention_heads: "4",
          num_key_value_heads: "4",
          ...(contextEnvelopeConflictProof.metadata ?? {}),
        },
      });
      let contextEnvelopeConflictRejected = false;
      try {
        WasmWebGpuLlamaResourceProgram.fromSafetensors({
          device: hostDevice,
          resourceBridge: hostResourceBridge,
          sessionBindingBridge,
          runtime: hostRuntime,
          program,
          config: contextEnvelopeConflictProof.config,
          contextLength: mhaTinyLlamaShape.contextLength,
          safetensors: contextEnvelopeConflictBytes,
        });
      } catch (err) {
        if (!String(err?.message ?? err).includes(contextEnvelopeConflictProof.message)) throw err;
        contextEnvelopeConflictRejected = true;
      }
      if (!contextEnvelopeConflictRejected) {
        throw new Error(`host-only WebGPU LLaMA safetensors Program accepted conflicting ${contextEnvelopeConflictProof.label}`);
      }
    }

    for (const slidingWindowConflictProof of [
      {
        config: { sliding_window: mhaTinyLlamaShape.contextLength },
        label: "metadata/config sliding window",
        message: "sliding window mismatch",
        metadata: {
          sliding_window: String(mhaTinyLlamaShape.contextLength + 1),
        },
      },
      {
        config: { sliding_window: mhaTinyLlamaShape.contextLength },
        label: "option/config sliding window",
        message: "sliding window mismatch",
        options: {
          slidingWindow: mhaTinyLlamaShape.contextLength + 1,
        },
      },
      {
        config: {},
        label: "metadata sliding window enabled without window",
        message: "sliding window attention",
        metadata: {
          use_sliding_window: "true",
        },
      },
    ]) {
      const slidingWindowConflictBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
        metadata: {
          num_attention_heads: "4",
          num_key_value_heads: "4",
          ...(slidingWindowConflictProof.metadata ?? {}),
        },
      });
      let slidingWindowConflictRejected = false;
      try {
        WasmWebGpuLlamaResourceProgram.fromSafetensors({
          device: hostDevice,
          resourceBridge: hostResourceBridge,
          sessionBindingBridge,
          runtime: hostRuntime,
          program,
          config: slidingWindowConflictProof.config,
          contextLength: mhaTinyLlamaShape.contextLength,
          safetensors: slidingWindowConflictBytes,
          ...(slidingWindowConflictProof.options ?? {}),
        });
      } catch (err) {
        if (!String(err?.message ?? err).includes(slidingWindowConflictProof.message)) throw err;
        slidingWindowConflictRejected = true;
      }
      if (!slidingWindowConflictRejected) {
        throw new Error(`host-only WebGPU LLaMA safetensors Program accepted conflicting ${slidingWindowConflictProof.label}`);
      }
    }

    for (const ropeScalingConflictProof of [
      {
        config: {
          rope_scaling: { type: "linear", factor: 2 },
        },
        label: "option rope kind",
        message: "RoPE kind mismatch",
        options: {
          ropeKind: "llama3",
        },
      },
      {
        config: {
          rope_scaling: {
            factor: 8,
            low_freq_factor: 1,
            high_freq_factor: 4,
            original_max_position_embeddings: 16,
            rope_type: "llama3",
          },
        },
        label: "option Llama 3 high frequency factor",
        message: "RoPE high frequency factor mismatch",
        options: {
          ropeHighFreqFactor: 5,
        },
      },
      {
        config: {
          rope_scaling: {
            factor: 8,
            low_freq_factor: 1,
            high_freq_factor: 4,
            original_max_position_embeddings: 16,
            rope_type: "llama3",
          },
        },
        label: "metadata rope kind",
        message: "RoPE kind mismatch",
        metadata: {
          rope_kind: "linear",
        },
      },
      {
        config: {
          rope_scaling: {
            factor: 8,
            low_freq_factor: 1,
            high_freq_factor: 4,
            original_max_position_embeddings: 16,
            rope_type: "llama3",
          },
        },
        label: "metadata Llama 3 original context length",
        message: "RoPE original context length mismatch",
        metadata: {
          rope_original_context_length: "32",
        },
      },
    ]) {
      const ropeScalingConflictBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
        metadata: {
          num_attention_heads: "4",
          num_key_value_heads: "4",
          ...(ropeScalingConflictProof.metadata ?? {}),
        },
      });
      let ropeScalingConflictRejected = false;
      try {
        WasmWebGpuLlamaResourceProgram.fromSafetensors({
          device: hostDevice,
          resourceBridge: hostResourceBridge,
          sessionBindingBridge,
          runtime: hostRuntime,
          program,
          config: ropeScalingConflictProof.config,
          contextLength: mhaTinyLlamaShape.contextLength,
          safetensors: ropeScalingConflictBytes,
          ...(ropeScalingConflictProof.options ?? {}),
        });
      } catch (err) {
        if (!String(err?.message ?? err).includes(ropeScalingConflictProof.message)) throw err;
        ropeScalingConflictRejected = true;
      }
      if (!ropeScalingConflictRejected) {
        throw new Error(`host-only WebGPU LLaMA safetensors Program accepted conflicting ${ropeScalingConflictProof.label}`);
      }
    }

    for (const sourceShapeConflictProof of [
      {
        label: "metadata hidden size",
        message: "hidden size mismatch",
        metadata: {
          hidden_size: String(mhaTinyLlamaShape.hiddenSize + 1),
          num_attention_heads: "4",
          num_key_value_heads: "4",
        },
      },
      {
        label: "metadata vocab size",
        message: "vocab size mismatch",
        metadata: {
          num_attention_heads: "4",
          num_key_value_heads: "4",
          vocab_size: String(mhaTinyLlamaShape.vocabSize + 1),
        },
      },
      {
        label: "metadata layer count",
        message: "layer count mismatch",
        metadata: {
          num_attention_heads: "4",
          num_hidden_layers: String(mhaTinyLlamaShape.layers + 1),
          num_key_value_heads: "4",
        },
      },
      {
        label: "option vocab size",
        message: "vocab size mismatch",
        options: {
          vocabSize: mhaTinyLlamaShape.vocabSize + 1,
        },
      },
      {
        label: "option layer count",
        message: "layer count mismatch",
        options: {
          numHiddenLayers: mhaTinyLlamaShape.layers + 1,
        },
      },
    ]) {
      const sourceShapeConflictBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
        metadata: sourceShapeConflictProof.metadata ?? {
          num_attention_heads: "4",
          num_key_value_heads: "4",
        },
      });
      let sourceShapeConflictRejected = false;
      try {
        WasmWebGpuLlamaResourceProgram.fromSafetensors({
          device: hostDevice,
          resourceBridge: hostResourceBridge,
          sessionBindingBridge,
          runtime: hostRuntime,
          program,
          contextLength: mhaTinyLlamaShape.contextLength,
          safetensors: sourceShapeConflictBytes,
          ...(sourceShapeConflictProof.options ?? {}),
        });
      } catch (err) {
        if (!String(err?.message ?? err).includes(sourceShapeConflictProof.message)) throw err;
        sourceShapeConflictRejected = true;
      }
      if (!sourceShapeConflictRejected) {
        throw new Error(`host-only WebGPU LLaMA safetensors Program accepted conflicting ${sourceShapeConflictProof.label}`);
      }
    }

    const badDownTinyLlamaSpecs = mhaTinyLlamaTwoLayerTensorSpecs.map(([name, shape]) =>
      name === "model.layers.1.mlp.down_proj.weight"
        ? [name, [mhaTinyLlamaShape.hiddenSize, mhaTinyLlamaShape.ffnSize + 1]]
        : [name, shape]
    );
    const badDownBytes = tinyLlamaModelResourceFileBytes(badDownTinyLlamaSpecs, {
      metadata: {
        num_attention_heads: "4",
        num_key_value_heads: "4",
      },
    });
    let badDownRejected = false;
    try {
      WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        contextLength: mhaTinyLlamaShape.contextLength,
        safetensors: badDownBytes,
      });
    } catch (err) {
      if (!String(err?.message ?? err).includes("down projection shape mismatch")) throw err;
      badDownRejected = true;
    }
    if (!badDownRejected) {
      throw new Error("host-only WebGPU LLaMA safetensors Program accepted malformed FFN down projection shape");
    }

    const unsupportedDtypeHeader = JSON.parse(tinyLlamaSafetensorsHeader(mhaTinyLlamaTwoLayerTensorSpecs));
    unsupportedDtypeHeader.__metadata__ = {
      num_attention_heads: "4",
      num_key_value_heads: "4",
    };
    unsupportedDtypeHeader["model.layers.0.self_attn.q_proj.weight"] = {
      ...unsupportedDtypeHeader["model.layers.0.self_attn.q_proj.weight"],
      dtype: "I8",
    };
    let unsupportedDtypeRejected = false;
    try {
      WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        contextLength: mhaTinyLlamaShape.contextLength,
        program,
        safetensors: unsupportedDtypeHeader,
      });
    } catch (err) {
      if (!String(err?.message ?? err).includes("unsupported LLaMA safetensors tensor dtype")) throw err;
      unsupportedDtypeRejected = true;
    }
    if (!unsupportedDtypeRejected) {
      throw new Error("host-only WebGPU LLaMA safetensors Program accepted unsupported I8 tensor dtype");
    }

    const unsupportedConfigBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
      metadata: {
        num_attention_heads: "4",
        num_key_value_heads: "4",
      },
    });
    for (const unsupportedConfigProof of [
      { config: { model_type: "gpt_neox" }, label: "model type", message: "model type" },
      { config: { architectures: ["GPTNeoXForCausalLM"] }, label: "architecture", message: "architectures" },
      { config: { torch_dtype: "float8_e4m3fn" }, label: "torch dtype", message: "torch dtype" },
      { config: { use_cache: "sometimes" }, label: "use_cache", message: "use_cache" },
      { config: { eos_token_id: [-1] }, label: "eos token id", message: "eos_token_id" },
      { config: { eos_token_id: [mhaTinyLlamaShape.vocabSize] }, label: "out-of-vocab eos token id", message: "token id exceeds vocab size" },
      { config: { attention_dropout: 2 }, label: "attention dropout", message: "attention_dropout" },
	      { config: { hidden_act: "gelu" }, label: "hidden activation", message: "hidden activation" },
	      { config: { qk_norm: true }, label: "Q/K projection normalization without tensors", message: "Q projection norm" },
	      { config: { q_norm: false, k_norm: true }, label: "split Q/K projection normalization", message: "Q/K projection normalization mismatch" },
	      { config: { model_type: "smollm3", architectures: ["SmolLM3ForCausalLM"], no_rope_layers: [1] }, label: "SmolLM3 short no-RoPE layer list", message: "no_rope_layers" },
	      { config: { use_sliding_window: true }, label: "sliding window enabled without window", message: "sliding window attention" },
      { config: { rope_scaling: { factor: 2, type: "dynamic" } }, label: "dynamic rope scaling", message: "rope scaling" },
      { config: { rope_scaling: { factor: 8, low_freq_factor: 4, high_freq_factor: 1, original_max_position_embeddings: 16, rope_type: "llama3" } }, label: "invalid Llama 3 rope scaling", message: "rope scaling" },
      { config: { partial_rotary_factor: 0.5 }, label: "partial rotary", message: "partial rotary factor" },
      { config: { max_position_embeddings: 2 }, label: "short max context", message: "exceeds config maximum position embeddings" },
    ]) {
      let unsupportedConfigRejected = false;
      try {
        WasmWebGpuLlamaResourceProgram.fromSafetensors({
          device: hostDevice,
          resourceBridge: hostResourceBridge,
          sessionBindingBridge,
          runtime: hostRuntime,
          program,
          config: unsupportedConfigProof.config,
          contextLength: mhaTinyLlamaShape.contextLength,
          safetensors: unsupportedConfigBytes,
        });
      } catch (err) {
        if (!String(err?.message ?? err).includes(unsupportedConfigProof.message)) throw err;
        unsupportedConfigRejected = true;
      }
      if (!unsupportedConfigRejected) {
        throw new Error(`host-only WebGPU LLaMA safetensors Program accepted unsupported config ${unsupportedConfigProof.label}`);
      }
    }

    {
      const qkNormTensorConflictBytes = tinyLlamaModelResourceFileBytes(qkNormGroupedTinyLlamaTwoLayerTensorSpecs);
      let qkNormTensorConflictRejected = false;
      try {
        WasmWebGpuLlamaResourceProgram.fromSafetensors({
          device: hostDevice,
          resourceBridge: hostResourceBridge,
          sessionBindingBridge,
          runtime: hostRuntime,
          program,
          config: {
            architectures: ["Qwen3ForCausalLM"],
            head_dim: groupedTinyLlamaShape.attentionHeadSize,
            hidden_size: groupedTinyLlamaShape.hiddenSize,
            hidden_act: "silu",
            intermediate_size: groupedTinyLlamaShape.ffnSize,
            max_position_embeddings: groupedTinyLlamaShape.contextLength,
            model_type: "qwen3",
            num_attention_heads: 4,
            num_hidden_layers: groupedTinyLlamaShape.layers,
            num_key_value_heads: 2,
            qk_norm: false,
            rms_norm_eps: 0.000082,
            rope_theta: 17500,
            tie_word_embeddings: false,
            vocab_size: groupedTinyLlamaShape.vocabSize,
          },
          contextLength: groupedTinyLlamaShape.contextLength,
          safetensors: qkNormTensorConflictBytes,
        });
      } catch (err) {
        if (!String(err?.message ?? err).includes("Q/K projection normalization mismatch")) throw err;
        qkNormTensorConflictRejected = true;
      }
      if (!qkNormTensorConflictRejected) {
        throw new Error("host-only WebGPU LLaMA safetensors Program accepted q_norm/k_norm tensors with disabled Q/K projection normalization");
      }
    }

    for (const biasTensorProof of [
      {
        config: { attention_bias: true },
        label: "attention bias enabled without tensors",
        message: "attention bias is enabled",
        specs: mhaTinyLlamaTwoLayerTensorSpecs,
      },
      {
        config: { mlp_bias: true },
        label: "MLP bias enabled without tensors",
        message: "mlp bias is enabled",
        specs: mhaTinyLlamaTwoLayerTensorSpecs,
      },
      {
        config: { attention_bias: false },
        label: "disabled attention bias with tensor",
        message: "attention bias tensor conflicts with disabled bias",
        specs: [
          ...mhaTinyLlamaTwoLayerTensorSpecs,
          ["model.layers.0.self_attn.q_proj.bias", [mhaTinyLlamaShape.hiddenSize]],
        ],
      },
      {
        config: {},
        label: "partial MLP bias tensor set",
        message: "mlp bias tensor set is incomplete",
        specs: [
          ...mhaTinyLlamaTwoLayerTensorSpecs,
          ["model.layers.0.mlp.gate_proj.bias", [mhaTinyLlamaShape.ffnSize]],
        ],
      },
    ]) {
      const biasTensorBytes = tinyLlamaModelResourceFileBytes(biasTensorProof.specs, {
        metadata: {
          num_attention_heads: "4",
          num_key_value_heads: "4",
        },
      });
      let biasTensorRejected = false;
      try {
        WasmWebGpuLlamaResourceProgram.fromSafetensors({
          device: hostDevice,
          resourceBridge: hostResourceBridge,
          sessionBindingBridge,
          runtime: hostRuntime,
          program,
          config: biasTensorProof.config,
          contextLength: mhaTinyLlamaShape.contextLength,
          safetensors: biasTensorBytes,
        });
      } catch (err) {
        if (!String(err?.message ?? err).toLowerCase().includes(biasTensorProof.message.toLowerCase())) throw err;
        biasTensorRejected = true;
      }
      if (!biasTensorRejected) {
        throw new Error(`host-only WebGPU LLaMA safetensors Program accepted invalid ${biasTensorProof.label}`);
      }
    }

    for (const unsupportedMetadataProof of [
      { label: "model type", message: "metadata model type", metadata: { model_type: "gpt_neox" } },
      { label: "architecture", message: "metadata architectures", metadata: { architectures: ["GPTNeoXForCausalLM"] } },
      { label: "torch dtype", message: "metadata torch dtype", metadata: { torch_dtype: "float8_e4m3fn" } },
      { label: "use_cache", message: "use_cache", metadata: { use_cache: "sometimes" } },
      { label: "eos token id", message: "eos_token_id", metadata: { eos_token_id: "-1" } },
	      { label: "attention dropout", message: "attention_dropout", metadata: { attention_dropout: "2" } },
	      { label: "hidden activation", message: "metadata hidden activation", metadata: { hidden_act: "gelu" } },
	      { label: "Q/K projection normalization without tensors", message: "Q projection norm", metadata: { "llama.attention.qk_norm": "true" } },
	      { label: "SmolLM3 malformed no-RoPE layer list", message: "no_rope_layers", metadata: { model_type: "smollm3", architectures: ["SmolLM3ForCausalLM"], no_rope_layers: "[1, nope]" } },
      { label: "partial rotary", message: "metadata partial rotary factor", metadata: { partial_rotary_factor: "0.5" } },
      { label: "partial rotary dimension count", message: "partial rotary dimension count", metadata: { "qwen2.rope.dimension_count": "1" } },
      { label: "attention key length", message: "attention key/value length mismatch", metadata: { "qwen2.attention.key_length": "1" } },
      { label: "tokenizer token count", message: "vocab size mismatch", metadata: { "tokenizer.ggml.tokens": JSON.stringify(Array.from({ length: mhaTinyLlamaShape.vocabSize + 1 }, (_unused, token) => `bad-token-${token}`)) } },
      { label: "tokenizer score count", message: "vocab size mismatch", metadata: { "tokenizer.ggml.scores": JSON.stringify(Array.from({ length: mhaTinyLlamaShape.vocabSize + 1 }, (_unused, token) => -0.01 * token)) } },
      { label: "tokenizer token type count", message: "vocab size mismatch", metadata: { "tokenizer.ggml.token_type": JSON.stringify(Array.from({ length: mhaTinyLlamaShape.vocabSize + 1 }, () => 1)) } },
      { label: "out-of-vocab metadata eos token id", message: "token id exceeds vocab size", metadata: { "tokenizer.ggml.eos_token_id": String(mhaTinyLlamaShape.vocabSize) } },
    ]) {
      const unsupportedMetadataBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
        metadata: {
          num_attention_heads: "4",
          num_key_value_heads: "4",
          ...unsupportedMetadataProof.metadata,
        },
      });
      let unsupportedMetadataRejected = false;
      try {
        WasmWebGpuLlamaResourceProgram.fromSafetensors({
          device: hostDevice,
          resourceBridge: hostResourceBridge,
          sessionBindingBridge,
          runtime: hostRuntime,
          program,
          contextLength: mhaTinyLlamaShape.contextLength,
          safetensors: unsupportedMetadataBytes,
        });
      } catch (err) {
        if (!String(err?.message ?? err).includes(unsupportedMetadataProof.message)) throw err;
        unsupportedMetadataRejected = true;
      }
      if (!unsupportedMetadataRejected) {
        throw new Error(`host-only WebGPU LLaMA safetensors Program accepted unsupported metadata ${unsupportedMetadataProof.label}`);
      }
    }

    for (const unsupportedTensorProof of [
      {
        label: "norm bias tensor",
        message: "norm bias tensor",
        spec: ["model.layers.0.input_layernorm.bias", [mhaTinyLlamaShape.hiddenSize]],
      },
	      {
	        label: "malformed LM-head bias tensor",
	        message: "lm_head bias tensor",
	        spec: ["lm_head.bias", [mhaTinyLlamaShape.vocabSize + 1]],
	      },
	      {
	        label: "Q projection norm tensor",
	        message: "Q/K projection normalization tensor mismatch",
	        spec: ["model.layers.0.self_attn.q_norm.weight", [mhaTinyLlamaShape.attentionHeadSize]],
	      },
	    ]) {
      const unsupportedTensorBytes = tinyLlamaModelResourceFileBytes([
        ...mhaTinyLlamaTwoLayerTensorSpecs,
        unsupportedTensorProof.spec,
      ], {
        metadata: {
          num_attention_heads: "4",
          num_key_value_heads: "4",
        },
      });
      let unsupportedTensorRejected = false;
      try {
        WasmWebGpuLlamaResourceProgram.fromSafetensors({
          device: hostDevice,
          resourceBridge: hostResourceBridge,
          sessionBindingBridge,
          runtime: hostRuntime,
          contextLength: mhaTinyLlamaShape.contextLength,
          program,
          safetensors: unsupportedTensorBytes,
        });
      } catch (err) {
        if (!String(err?.message ?? err).includes(unsupportedTensorProof.message)) throw err;
        unsupportedTensorRejected = true;
      }
      if (!unsupportedTensorRejected) {
        throw new Error(`host-only WebGPU LLaMA safetensors Program accepted unsupported ${unsupportedTensorProof.label}`);
      }
    }

    const rotaryAuxTensorSpecs = [
      ...mhaTinyLlamaTwoLayerTensorSpecs,
      ["model.rotary_emb.inv_freq", [mhaTinyLlamaShape.attentionHeadSize / 2]],
      ["model.layers.0.self_attn.rotary_emb.inv_freq", [mhaTinyLlamaShape.attentionHeadSize / 2]],
    ];
    const rotaryAuxBytes = tinyLlamaModelResourceFileBytes(rotaryAuxTensorSpecs, {
      metadata: {
        num_attention_heads: "4",
        num_key_value_heads: "4",
        rope_theta: "11000",
      },
    });
    const rotaryAuxProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      contextLength: mhaTinyLlamaShape.contextLength,
      program,
      safetensors: rotaryAuxBytes,
      allowMockFallback: true,
    });
    if (
      rotaryAuxProgram.requiredModelResources.length !== mhaTinyLlamaTwoLayerTensorSpecs.length ||
      rotaryAuxProgram.requiredModelResources.some((entry) => entry.role.includes("inv_freq"))
    ) {
      throw new Error("host-only WebGPU LLaMA safetensors Program retained rotary inv_freq in executable manifest");
    }
    const rotaryAuxSession = rotaryAuxProgram.bindHostResources();
    try {
      if (
        rotaryAuxSession.modelResources.length !== mhaTinyLlamaTwoLayerTensorSpecs.length ||
        rotaryAuxSession.modelResources.some((entry) => entry.role.includes("inv_freq"))
      ) {
        throw new Error("host-only WebGPU LLaMA safetensors Session retained rotary inv_freq as a model resource");
      }
    } finally {
      rotaryAuxProgram.unregister(rotaryAuxSession);
      rotaryAuxSession.dispose();
      rotaryAuxProgram.tokenExecutor?.destroy?.();
    }

    for (const scaledRotaryAuxProof of [
      {
        config: {
          rope_scaling: { type: "linear", factor: 2 },
          rope_theta: 11000,
        },
        label: "linear",
        options: {
          ropeBase: 11000,
          ropeKind: "linear",
          ropeScale: 2,
        },
      },
      {
        config: {
          rope_scaling: {
            factor: 8,
            low_freq_factor: 1,
            high_freq_factor: 4,
            original_max_position_embeddings: 16,
            rope_type: "llama3",
          },
          rope_theta: 13500,
        },
        label: "Llama 3",
        options: {
          ropeBase: 13500,
          ropeHighFreqFactor: 4,
          ropeKind: "llama3",
          ropeLowFreqFactor: 1,
          ropeOriginalContextLength: 16,
          ropeScale: 8,
        },
      },
    ]) {
      const scaledRotaryAuxBytes = tinyLlamaModelResourceFileBytes(rotaryAuxTensorSpecs, {
        ...scaledRotaryAuxProof.options,
        metadata: {
          num_attention_heads: "4",
          num_key_value_heads: "4",
        },
      });
      const scaledRotaryAuxProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        config: scaledRotaryAuxProof.config,
        contextLength: mhaTinyLlamaShape.contextLength,
        program,
        safetensors: scaledRotaryAuxBytes,
        allowMockFallback: true,
      });
      if (
        scaledRotaryAuxProgram.requiredModelResources.length !== mhaTinyLlamaTwoLayerTensorSpecs.length ||
        scaledRotaryAuxProgram.requiredModelResources.some((entry) => entry.role.includes("inv_freq"))
      ) {
        throw new Error(`host-only WebGPU LLaMA safetensors Program retained ${scaledRotaryAuxProof.label} rotary inv_freq in executable manifest`);
      }
      scaledRotaryAuxProgram.tokenExecutor?.destroy?.();
    }

    const mismatchedScaledRotaryAuxBytes = tinyLlamaModelResourceFileBytes(rotaryAuxTensorSpecs, {
      metadata: {
        num_attention_heads: "4",
        num_key_value_heads: "4",
        rope_theta: "11000",
      },
    });
    let mismatchedScaledRotaryAuxRejected = false;
    try {
      WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        config: {
          rope_scaling: { type: "linear", factor: 2 },
          rope_theta: 11000,
        },
        contextLength: mhaTinyLlamaShape.contextLength,
        program,
        safetensors: mismatchedScaledRotaryAuxBytes,
      });
    } catch (err) {
      if (!String(err?.message ?? err).includes("rotary inv_freq value mismatch")) throw err;
      mismatchedScaledRotaryAuxRejected = true;
    }
    if (!mismatchedScaledRotaryAuxRejected) {
      throw new Error("host-only WebGPU LLaMA safetensors Program accepted mismatched scaled rotary inv_freq values");
    }

    const malformedRotaryAuxBytes = tinyLlamaModelResourceFileBytes([
      ...mhaTinyLlamaTwoLayerTensorSpecs,
      ["model.rotary_emb.inv_freq", [mhaTinyLlamaShape.attentionHeadSize]],
    ], {
      metadata: {
        num_attention_heads: "4",
        num_key_value_heads: "4",
      },
    });
    let malformedRotaryAuxRejected = false;
    try {
      WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        contextLength: mhaTinyLlamaShape.contextLength,
        program,
        safetensors: malformedRotaryAuxBytes,
      });
    } catch (err) {
      if (!String(err?.message ?? err).includes("rotary inv_freq length mismatch")) throw err;
      malformedRotaryAuxRejected = true;
    }
    if (!malformedRotaryAuxRejected) {
      throw new Error("host-only WebGPU LLaMA safetensors Program accepted malformed rotary inv_freq");
    }

    for (const unsupportedExtraTensorProof of [
      {
        label: "unsupported rotary cache tensor",
        message: "unsupported LLaMA safetensors tensor",
        spec: ["model.rotary_emb.cos_cached", [mhaTinyLlamaShape.hiddenSize]],
      },
      {
        label: "quantization sidecar tensor",
        message: "unsupported LLaMA safetensors layer tensor",
        spec: ["model.layers.0.self_attn.q_proj.weight_scale", [mhaTinyLlamaShape.hiddenSize]],
      },
    ]) {
      const unsupportedExtraTensorBytes = tinyLlamaModelResourceFileBytes([
        ...mhaTinyLlamaTwoLayerTensorSpecs,
        unsupportedExtraTensorProof.spec,
      ], {
        metadata: {
          num_attention_heads: "4",
          num_key_value_heads: "4",
        },
      });
      let unsupportedExtraTensorRejected = false;
      try {
        WasmWebGpuLlamaResourceProgram.fromSafetensors({
          device: hostDevice,
          resourceBridge: hostResourceBridge,
          sessionBindingBridge,
          runtime: hostRuntime,
          contextLength: mhaTinyLlamaShape.contextLength,
          program,
          safetensors: unsupportedExtraTensorBytes,
        });
      } catch (err) {
        if (!String(err?.message ?? err).includes(unsupportedExtraTensorProof.message)) throw err;
        unsupportedExtraTensorRejected = true;
      }
      if (!unsupportedExtraTensorRejected) {
        throw new Error(`host-only WebGPU LLaMA safetensors Program accepted unsupported ${unsupportedExtraTensorProof.label}`);
      }
    }

    for (const dataOffsetProof of [
      {
        label: "missing data_offsets",
        message: "missing data_offsets",
        mutate(header) {
          delete header["lm_head.weight"].data_offsets;
        },
      },
      {
        label: "zero-length data_offsets",
        message: "byteLength 0 does not match",
        mutate(header) {
          header["lm_head.weight"].data_offsets = [0, 0];
        },
      },
      {
        label: "overlapping data_offsets",
        message: "data_offsets overlap",
        mutate(header) {
          header["lm_head.weight"].data_offsets = [...header["model.embed_tokens.weight"].data_offsets];
        },
      },
      {
        label: "out-of-bounds data_offsets",
        message: "out of bounds",
        mutate(header, payloadByteLength) {
          const [start, end] = header["lm_head.weight"].data_offsets;
          const byteLength = end - start;
          header["lm_head.weight"].data_offsets = [payloadByteLength, payloadByteLength + byteLength];
        },
      },
    ]) {
      const corruptedDataOffsetBytes = mutateSafetensorsHeaderBytes(
        tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
          metadata: {
            num_attention_heads: "4",
            num_key_value_heads: "4",
          },
        }),
        dataOffsetProof.mutate,
      );
      let dataOffsetRejected = false;
      try {
        WasmWebGpuLlamaResourceProgram.fromSafetensors({
          device: hostDevice,
          resourceBridge: hostResourceBridge,
          sessionBindingBridge,
          runtime: hostRuntime,
          contextLength: mhaTinyLlamaShape.contextLength,
          program,
          safetensors: corruptedDataOffsetBytes,
        });
      } catch (err) {
        if (!String(err?.message ?? err).includes(dataOffsetProof.message)) throw err;
        dataOffsetRejected = true;
      }
      if (!dataOffsetRejected) {
        throw new Error(`host-only WebGPU LLaMA safetensors Program accepted ${dataOffsetProof.label}`);
      }
    }

    const tiedConflictBytes = tinyLlamaModelResourceFileBytes(mhaTinyLlamaTwoLayerTensorSpecs, {
      metadata: {
        num_attention_heads: "4",
        num_key_value_heads: "4",
        tie_word_embeddings: "true",
      },
    });
    let tiedConflictRejected = false;
    try {
      WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        config: { tie_word_embeddings: false },
        contextLength: mhaTinyLlamaShape.contextLength,
        safetensors: tiedConflictBytes,
      });
    } catch (err) {
      if (!String(err?.message ?? err).includes("tied LM head mismatch")) throw err;
      tiedConflictRejected = true;
    }
    if (!tiedConflictRejected) {
      throw new Error("host-only WebGPU LLaMA safetensors Program accepted conflicting tied-LM-head sources");
    }

    const tiedPipelineExecutorCalls = [];
    const tiedModelResourceBytes = tinyLlamaModelResourceFileBytes(tinyLlamaTwoLayerTiedTensorSpecs);
    const tiedModelResourceOptions = { allowTiedLmHead: true };
    const tiedModelResourceRequirements = tinyLlamaModelResourceSpecs(tiedModelResourceBytes, tiedModelResourceOptions);
    const tiedRawModelResourceRequirements = tinyLlamaModelResourceSpecs(tiedModelResourceBytes);
    if (
      tiedRawModelResourceRequirements.some((entry) => entry.tensor === "lm_head.weight") ||
      !tiedModelResourceRequirements.some((entry) => entry.tensor === "lm_head.weight")
    ) {
      throw new Error("host-only WebGPU tiny LLaMA tied lm_head safetensors requirements were not derived");
    }
    let untiedMissingLmHeadRejected = false;
    try {
      WasmWebGpuLlamaResourceProgram.fromSafetensors({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        config: { tie_word_embeddings: false },
        contextLength: 4,
        safetensors: tiedModelResourceBytes,
      });
    } catch (err) {
      if (!String(err?.message ?? err).includes("lm_head tensor")) throw err;
      untiedMissingLmHeadRejected = true;
    }
    if (!untiedMissingLmHeadRejected) {
      throw new Error("host-only WebGPU tiny LLaMA accepted missing lm_head despite untied config");
    }
	    const tiedPipelineProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
	      device: hostDevice,
	      resourceBridge: hostResourceBridge,
	      sessionBindingBridge,
	      runtime: hostRuntime,
      program,
      config: { tie_word_embeddings: true },
      contextLength: 4,
      model: compatibleModel,
      safetensors: tiedModelResourceBytes,
      allowMockFallback: true,
      executorOptions: {
        record(call) {
          recordBrowserLlamaPipelineCallEvidence(tiedPipelineExecutorCalls, "tied-lm-head-pipeline", call);
	        },
	      },
	    });
	    let nativeTiedModel = 0;
	    let nativeTiedCompatibleModel = 0;
	    let nativeTiedProgramHandle = 0;
	    let nativeTiedProgram = null;
	    let nativeTiedSession = null;
	    let nativeTiedSessionHandle = 0;
	    const nativeTiedExecutorCalls = [];
	    try {
	      nativeTiedModel = createModel(tinyLlama2LayerKind, 0, 0);
	      nativeTiedCompatibleModel = createModel(tinyLlama2LayerKind, 0, 0);
	      nativeTiedProgramHandle = compileProgram(nativeTiedModel, 4, backendWebGpu);
	      const nativeTiedKvRequirements = keep(alloc(llamaKvCacheRequirementsSize), llamaKvCacheRequirementsSize);
	      zero(nativeTiedKvRequirements, llamaKvCacheRequirementsSize);
	      check(exportsRef.zgml_llama_program_get_kv_cache_requirements(nativeTiedProgramHandle, nativeTiedKvRequirements));
	      nativeTiedProgram = WasmWebGpuLlamaResourceProgram.fromSafetensors({
	        device: hostDevice,
	        resourceBridge: hostResourceBridge,
	        sessionBindingBridge,
	        runtime: hostRuntime,
	        program: nativeTiedProgramHandle,
	        config: { tie_word_embeddings: true },
	        contextLength: 4,
	        model: nativeTiedCompatibleModel,
	        safetensors: tiedModelResourceBytes,
	        allowMockFallback: true,
	        executorOptions: {
	          record(call) {
	            recordBrowserLlamaPipelineCallEvidence(nativeTiedExecutorCalls, "native-tied-lm-head-pipeline", call);
	          },
	        },
	      });
	      if (
	        nativeTiedProgram.kvRequirements.layers !== 2 ||
	        nativeTiedProgram.kvRequirements.kBufferByteLength !== u32(nativeTiedKvRequirements, 20) ||
	        nativeTiedProgram.kvRequirements.vBufferByteLength !== u32(nativeTiedKvRequirements, 24) ||
	        nativeTiedProgram.requiredModelResources.length !== tiedModelResourceRequirements.length
	      ) {
	        throw new Error("native-backed WebGPU tiny LLaMA tied lm_head Program did not match native K/V requirements and derived model manifest");
	      }
	      nativeTiedSession = nativeTiedProgram.bind();
	      nativeTiedSessionHandle = nativeTiedSession.handle;
	      if (nativeTiedSession.nativeSession !== true || !hostRuntime.hasClaimed(nativeTiedSessionHandle)) {
	        throw new Error("native-backed WebGPU tiny LLaMA tied lm_head Session was not registered for raw ABI execution");
	      }
	      if (
	        nativeTiedSession.modelHandle !== nativeTiedCompatibleModel ||
	        nativeTiedSession.modelResources.length !== tiedModelResourceRequirements.length
	      ) {
	        throw new Error("native-backed WebGPU tiny LLaMA tied lm_head bind did not preserve derived model resources");
	      }
	      const nativeTiedEmbeddingResource = nativeTiedSession.modelResources.find((entry) => entry.role === "llama.weight.model.embed_tokens.weight");
	      const nativeTiedLmHeadResource = nativeTiedSession.modelResources.find((entry) => entry.role === "llama.weight.lm_head.weight");
	      if (
	        !nativeTiedEmbeddingResource ||
	        !nativeTiedLmHeadResource ||
	        nativeTiedEmbeddingResource.dataByteOffset !== nativeTiedLmHeadResource.dataByteOffset ||
	        nativeTiedEmbeddingResource.byteLength !== nativeTiedLmHeadResource.byteLength
	      ) {
	        throw new Error("native-backed WebGPU tiny LLaMA tied lm_head Session did not alias embedding bytes");
	      }
	      expectHostResourceTableRoles(
	        nativeTiedSession.table,
	        ["llama.output", "llama.k.0", "llama.v.0", "llama.k.1", "llama.v.1"],
	        "native-backed WebGPU tiny LLaMA tied lm_head",
	      );
	      const nativeTiedLayer0K = nativeTiedSession.bindings.kvCache[0].k;
	      const nativeTiedLayer0V = nativeTiedSession.bindings.kvCache[0].v;
	      const nativeTiedLayer1K = nativeTiedSession.bindings.kvCache[1].k;
	      const nativeTiedLayer1V = nativeTiedSession.bindings.kvCache[1].v;
	      const nativeTiedLayer0KElements = Math.floor(nativeTiedLayer0K.byteLength / Float32Array.BYTES_PER_ELEMENT);
	      const nativeTiedLayer0VElements = Math.floor(nativeTiedLayer0V.byteLength / Float32Array.BYTES_PER_ELEMENT);
	      const nativeTiedLayer1KElements = Math.floor(nativeTiedLayer1K.byteLength / Float32Array.BYTES_PER_ELEMENT);
	      const nativeTiedLayer1VElements = Math.floor(nativeTiedLayer1V.byteLength / Float32Array.BYTES_PER_ELEMENT);
	      const nativeTiedExpectedLayer0Cache = {
	        k: Array(nativeTiedLayer0KElements).fill(-3310),
	        v: Array(nativeTiedLayer0VElements).fill(-3320),
	      };
	      const nativeTiedExpectedLayer1Cache = {
	        k: Array(nativeTiedLayer1KElements).fill(-4310),
	        v: Array(nativeTiedLayer1VElements).fill(-4320),
	      };
	      hostDevice.writeFloat32(nativeTiedLayer0K, nativeTiedExpectedLayer0Cache.k);
	      hostDevice.writeFloat32(nativeTiedLayer0V, nativeTiedExpectedLayer0Cache.v);
	      hostDevice.writeFloat32(nativeTiedLayer1K, nativeTiedExpectedLayer1Cache.k);
	      hostDevice.writeFloat32(nativeTiedLayer1V, nativeTiedExpectedLayer1Cache.v);
	      hostDevice.writeFloat32(nativeTiedSession.bindings.output, Array(tinyLlamaVocabSize).fill(-6150));
	      const nativeTiedTokens = keep(alloc(4), 4);
	      writeU32Array(nativeTiedTokens, [2]);
	      const nativeTiedResult = tokenExecute(nativeTiedSessionHandle, nativeTiedTokens, 1, executeOutputLogits, 0, 0);
	      expectStatus(nativeTiedResult.status, ok, "native-backed WebGPU tiny LLaMA tied lm_head raw execute");
	      if (nativeTiedResult.outputLen !== tinyLlamaVocabSize || sessionPosition(nativeTiedSessionHandle) !== 1) {
	        throw new Error("native-backed WebGPU tiny LLaMA tied lm_head raw execute did not advance");
	      }
	      const nativeTiedInspection = inspectSession(nativeTiedSessionHandle);
	      if (
	        u32(nativeTiedInspection, 0) !== tinyLlama2LayerKind ||
	        u32(nativeTiedInspection, 4) !== backendWebGpu ||
	        u32(nativeTiedInspection, 8) !== bufferStorageExternalResource ||
	        u32(nativeTiedInspection, 12) !== bufferStorageExternalResource ||
	        u64(nativeTiedInspection, 16) !== 1n ||
	        u64(nativeTiedInspection, 24) !== 4n ||
	        u64(nativeTiedInspection, 64) !== 5n ||
	        u64(nativeTiedInspection, 72) === 0n
	      ) {
	        throw new Error("native-backed WebGPU tiny LLaMA tied lm_head raw inspect mismatch");
	      }
	      const nativeTiedExpectedLayer0 = tinyLlamaBlockProjectionStep(tiedModelResourceBytes, nativeTiedExpectedLayer0Cache, 2, 0, {
	        layer: 0,
	        modelResourceOptions: tiedModelResourceOptions,
	      });
	      const nativeTiedExpectedLayer1 = tinyLlamaBlockProjectionStep(tiedModelResourceBytes, nativeTiedExpectedLayer1Cache, 2, 0, {
	        inputHidden: nativeTiedExpectedLayer0.blockHidden,
	        layer: 1,
	        modelResourceOptions: tiedModelResourceOptions,
	      });
	      expectClose(
	        Array.from(await hostDevice.readFloat32(nativeTiedSession.bindings.output, tinyLlamaVocabSize)),
	        nativeTiedExpectedLayer1.logits,
	      );
	      expectClose(Array.from(await hostDevice.readFloat32(nativeTiedLayer0K, nativeTiedExpectedLayer0.k.length, 0)), nativeTiedExpectedLayer0.k);
	      expectClose(Array.from(await hostDevice.readFloat32(nativeTiedLayer0V, nativeTiedExpectedLayer0.v.length, 0)), nativeTiedExpectedLayer0.v);
	      expectClose(Array.from(await hostDevice.readFloat32(nativeTiedLayer1K, nativeTiedExpectedLayer1.k.length, 0)), nativeTiedExpectedLayer1.k);
	      expectClose(Array.from(await hostDevice.readFloat32(nativeTiedLayer1V, nativeTiedExpectedLayer1.v.length, 0)), nativeTiedExpectedLayer1.v);
	      const expectedNativeTiedCommandCount = expectedLlamaBlockPipelineCommandCount(hostDevice, 2, 1, executeOutputLogits);
	      const expectedNativeTiedBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? expectedNativeTiedCommandCount : 0;
	      const expectedNativeTiedFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : expectedNativeTiedCommandCount;
	      const nativeTiedProfile = nativeTiedSession.runtimeProfile();
	      if (
	        nativeTiedProfile.executorCallCount !== 1 ||
	        nativeTiedProfile.executorOkCount !== 1 ||
	        nativeTiedProfile.executorBackendDispatchCount !== expectedNativeTiedBackendDispatchCount ||
	        nativeTiedProfile.executorFallbackOpCount !== expectedNativeTiedFallbackOpCount ||
	        nativeTiedProfile.executorCommandCount !== expectedNativeTiedCommandCount ||
	        nativeTiedProfile.executorCommandOpCount !== expectedNativeTiedCommandCount ||
	        nativeTiedProfile.lastStatus !== ok ||
	        nativeTiedProfile.lastResultOutputLength !== tinyLlamaVocabSize
	      ) {
	        throw new Error(`unexpected native-backed WebGPU tiny LLaMA tied lm_head profile: ${JSON.stringify(nativeTiedProfile)}`);
	      }
	      const nativeTiedAbiProfile = sessionRuntimeProfile(nativeTiedSessionHandle);
	      if (
	        u64(nativeTiedAbiProfile, 0) !== 1n ||
	        u64(nativeTiedAbiProfile, 8) !== BigInt(expectedNativeTiedBackendDispatchCount) ||
	        u64(nativeTiedAbiProfile, 16) !== BigInt(expectedNativeTiedFallbackOpCount) ||
	        u64(nativeTiedAbiProfile, 24) !== BigInt(expectedNativeTiedBackendDispatchCount) ||
	        u64(nativeTiedAbiProfile, 40) !== 1n ||
	        u64(nativeTiedAbiProfile, 48) !== 1n ||
	        u64(nativeTiedAbiProfile, 56) !== 0n ||
	        u64(nativeTiedAbiProfile, 96) !== BigInt(expectedNativeTiedCommandCount) ||
	        u64(nativeTiedAbiProfile, 112) !== BigInt(expectedNativeTiedCommandCount)
	      ) {
	        throw new Error("native-backed WebGPU tiny LLaMA tied lm_head ABI profile mismatch");
	      }
	      if (
	        nativeTiedExecutorCalls.length !== 1 ||
	        nativeTiedExecutorCalls[0].stageCount !== 2 ||
	        nativeTiedExecutorCalls[0].commandCount !== expectedNativeTiedCommandCount ||
	        nativeTiedExecutorCalls[0].status !== ok
	      ) {
	        throw new Error("native-backed WebGPU tiny LLaMA tied lm_head pipeline evidence mismatch");
	      }
	    } finally {
	      if (nativeTiedSessionHandle !== 0) exportsRef.zgml_session_free(nativeTiedSessionHandle);
	      if (nativeTiedSession && nativeTiedSession.ownedResources.length !== 0) {
	        throw new Error("native-backed WebGPU tiny LLaMA tied lm_head Session-owned resources were not released");
	      }
	      nativeTiedProgram?.tokenExecutor?.destroy?.();
	      if (nativeTiedProgramHandle !== 0) exportsRef.zgml_program_free(nativeTiedProgramHandle);
	      if (nativeTiedCompatibleModel !== 0) exportsRef.zgml_model_free(nativeTiedCompatibleModel);
	      if (nativeTiedModel !== 0) exportsRef.zgml_model_free(nativeTiedModel);
	    }
	    const tiedPipelineSession = tiedPipelineProgram.bindHostResources();
	    const tiedPipelineSessionHandle = tiedPipelineSession.handle;
    const tiedHostExports = hostRuntime.wrapExports(exportsRef);
    try {
      const embeddingResource = tiedPipelineSession.modelResources.find((entry) => entry.role === "llama.weight.model.embed_tokens.weight");
      const lmHeadResource = tiedPipelineSession.modelResources.find((entry) => entry.role === "llama.weight.lm_head.weight");
      if (
        !embeddingResource ||
        !lmHeadResource ||
        embeddingResource.dataByteOffset !== lmHeadResource.dataByteOffset ||
        embeddingResource.byteLength !== lmHeadResource.byteLength
      ) {
        throw new Error("host-only WebGPU tiny LLaMA tied lm_head Session did not alias embedding bytes");
      }
      const tiedLayer0K = tiedPipelineSession.bindings.kvCache[0].k;
      const tiedLayer0V = tiedPipelineSession.bindings.kvCache[0].v;
      const tiedLayer1K = tiedPipelineSession.bindings.kvCache[1].k;
      const tiedLayer1V = tiedPipelineSession.bindings.kvCache[1].v;
      const tiedLayer0KElements = Math.floor(tiedLayer0K.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const tiedLayer0VElements = Math.floor(tiedLayer0V.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const tiedLayer1KElements = Math.floor(tiedLayer1K.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const tiedLayer1VElements = Math.floor(tiedLayer1V.byteLength / Float32Array.BYTES_PER_ELEMENT);
      hostDevice.writeFloat32(tiedLayer0K, Array(tiedLayer0KElements).fill(-3110));
      hostDevice.writeFloat32(tiedLayer0V, Array(tiedLayer0VElements).fill(-3120));
      hostDevice.writeFloat32(tiedLayer1K, Array(tiedLayer1KElements).fill(-4110));
      hostDevice.writeFloat32(tiedLayer1V, Array(tiedLayer1VElements).fill(-4120));
      hostDevice.writeFloat32(tiedPipelineSession.bindings.output, Array(tinyLlamaVocabSize).fill(-5150));
      const tiedTokens = keep(alloc(4), 4);
      writeU32Array(tiedTokens, [2]);
      const tiedResult = tokenExecuteWith(tiedHostExports, tiedPipelineSessionHandle, tiedTokens, 1, executeOutputLogits, 0, 0);
      expectStatus(tiedResult.status, ok, "host-only WebGPU tiny LLaMA tied lm_head block pipeline execute");
      if (tiedResult.outputLen !== tinyLlamaVocabSize || sessionPositionWith(tiedHostExports, tiedPipelineSessionHandle) !== 1) {
        throw new Error("host-only WebGPU tiny LLaMA tied lm_head block pipeline did not emit bound logits and advance");
      }
      const tiedExpectedLayer0Cache = {
        k: Array(tiedLayer0KElements).fill(-3110),
        v: Array(tiedLayer0VElements).fill(-3120),
      };
      const tiedExpectedLayer1Cache = {
        k: Array(tiedLayer1KElements).fill(-4110),
        v: Array(tiedLayer1VElements).fill(-4120),
      };
      const tiedExpectedLayer0 = tinyLlamaBlockProjectionStep(tiedModelResourceBytes, tiedExpectedLayer0Cache, 2, 0, {
        layer: 0,
        modelResourceOptions: tiedModelResourceOptions,
      });
      const tiedExpectedLayer1 = tinyLlamaBlockProjectionStep(tiedModelResourceBytes, tiedExpectedLayer1Cache, 2, 0, {
        inputHidden: tiedExpectedLayer0.blockHidden,
        layer: 1,
        modelResourceOptions: tiedModelResourceOptions,
      });
      expectClose(
        Array.from(await hostDevice.readFloat32(tiedPipelineSession.bindings.output, tinyLlamaVocabSize)),
        tiedExpectedLayer1.logits,
      );
      expectClose(Array.from(await hostDevice.readFloat32(tiedLayer0K, tiedExpectedLayer0.k.length, 0)), tiedExpectedLayer0.k);
      expectClose(Array.from(await hostDevice.readFloat32(tiedLayer0V, tiedExpectedLayer0.v.length, 0)), tiedExpectedLayer0.v);
      expectClose(Array.from(await hostDevice.readFloat32(tiedLayer1K, tiedExpectedLayer1.k.length, 0)), tiedExpectedLayer1.k);
      expectClose(Array.from(await hostDevice.readFloat32(tiedLayer1V, tiedExpectedLayer1.v.length, 0)), tiedExpectedLayer1.v);
      const expectedTiedBackendDispatchCount = canUseGpuStorageBuffers(hostDevice, 6) ? 2 : 0;
      const expectedTiedFallbackOpCount = canUseGpuStorageBuffers(hostDevice, 6) ? 0 : 2;
      const tiedProfile = tiedPipelineSession.runtimeProfile();
      if (
        tiedProfile.executorCallCount !== 1 ||
        tiedProfile.executorOkCount !== 1 ||
        tiedProfile.executorBackendDispatchCount !== expectedTiedBackendDispatchCount ||
        tiedProfile.executorFallbackOpCount !== expectedTiedFallbackOpCount ||
        tiedProfile.executorCommandCount !== 2 ||
        tiedProfile.executorCommandOpCount !== 2 ||
        tiedProfile.lastStatus !== ok ||
        tiedProfile.lastResultOutputLength !== tinyLlamaVocabSize
      ) {
        throw new Error(`unexpected host-only WebGPU tiny LLaMA tied lm_head pipeline profile: ${JSON.stringify(tiedProfile)}`);
      }
      if (
        tiedPipelineExecutorCalls.length !== 1 ||
        tiedPipelineExecutorCalls[0].stageCount !== 2 ||
        tiedPipelineExecutorCalls[0].commandCount !== 2 ||
        tiedPipelineExecutorCalls[0].status !== ok
      ) {
        throw new Error("host-only WebGPU tiny LLaMA tied lm_head block pipeline evidence mismatch");
      }
      recordBrowserLlamaProfileEvidence("tied-lm-head-pipeline", tiedProfile);
    } finally {
      expectStatus(tiedHostExports.zgml_session_free(tiedPipelineSessionHandle), ok, "host-only WebGPU tiny LLaMA tied lm_head Session free");
      tiedPipelineProgram.tokenExecutor?.destroy?.();
    }

    const expectRejectedHostLlamaTokenExecutor = (executor, slug, label, options = {}) => {
      const outputPolicy = options.outputPolicy ?? executeOutputNone;
      const expectedOutputKind = outputPolicy === executeOutputLogits ? "bound-logits" : "none";
      const invalidProgram = new WasmWebGpuLlamaResourceProgram({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        vocabSize: tinyLlamaVocabSize,
        contextLength: 4,
        kvCacheRequirements: {
          layers: u32(kvRequirements, 8),
          kBufferByteLength: u32(kvRequirements, 20),
          vBufferByteLength: u32(kvRequirements, 24),
        },
        tokenExecutor: executor,
      });
      const invalidSession = invalidProgram.bind(rawLlamaResources);
      const invalidSessionHandle = invalidSession.handle;
      try {
        const invalidTokens = keep(alloc(4), 4);
        writeU32Array(invalidTokens, [0]);
        const invalidExecute = tokenExecute(invalidSessionHandle, invalidTokens, 1, outputPolicy, 0, 0);
        expectStatus(invalidExecute.status, invalidArgument, `WebGPU tiny LLaMA host-token executor ${label} execute`);
        if (invalidExecute.outputLen !== 0 || sessionPosition(invalidSessionHandle) !== 0) {
          throw new Error(`WebGPU tiny LLaMA ${slug} executor mutated state`);
        }
        if (executor.calls !== 1 || invalidSession.tokenExecutionAttempts !== 1) {
          throw new Error(`WebGPU tiny LLaMA ${slug} executor was not called exactly once`);
        }
        expectLastLlamaTokenExecution(invalidSession, {
          tokens: [0],
          outputPolicy,
        }, `WebGPU tiny LLaMA ${slug} executor`);
        const invalidStepParams = invalidSession.stepParamsHistory.at(-1);
        if (
          invalidSession.stepParamsHistory.length !== 1 ||
          !invalidStepParams ||
          invalidStepParams.startPosition !== 0 ||
          invalidStepParams.endPosition !== 1 ||
          invalidStepParams.outputKind !== expectedOutputKind
        ) {
          throw new Error(`WebGPU tiny LLaMA ${slug} executor did not retain rejected StepParams evidence`);
        }
        const invalidInspection = inspectSession(invalidSessionHandle);
        if (u64(invalidInspection, 16) !== 0n) {
          throw new Error(`WebGPU tiny LLaMA ${slug} executor inspect position mutated`);
        }
        expectRejectedHostLlamaExecutorProfile(
          invalidSession,
          invalidSessionHandle,
          `WebGPU tiny LLaMA host-token executor ${label}`,
          { outputLen: 0, outputPolicy },
        );
      } finally {
        exportsRef.zgml_session_free(invalidSessionHandle);
        if (!invalidSession.freed || hostRuntime.lookup(invalidSessionHandle) !== null) {
          throw new Error(`WebGPU tiny LLaMA ${slug} host resource Session was not released through native free`);
        }
        if (hostRuntime.hasClaimed(invalidSessionHandle)) {
          throw new Error(`WebGPU tiny LLaMA ${slug} host resource Session claim was not released through native free`);
        }
      }
    };
    expectRejectedHostLlamaTokenExecutor(createInvalidHostLlamaCounterExecutor(), "invalid-counter", "invalid counter");
    expectRejectedHostLlamaTokenExecutor(createInvalidHostLlamaStatusExecutor(), "invalid-status", "invalid status");
    expectRejectedHostLlamaTokenExecutor(createMissingHostLlamaTableEvidenceExecutor(), "missing-table-evidence", "missing table evidence");
    expectRejectedHostLlamaTokenExecutor(createIncoherentHostLlamaCounterExecutor(), "incoherent-counter", "incoherent counter");
    expectRejectedHostLlamaTokenExecutor(createMissingCommandOpHostLlamaCounterExecutor(), "missing-command-op", "missing command op");
    expectRejectedHostLlamaTokenExecutor(createFailedWorkHostLlamaExecutor(), "failed-work", "failed work");
    expectRejectedHostLlamaTokenExecutor(createZeroWorkHostLlamaExecutor(), "zero-work", "zero work");
    expectRejectedHostLlamaTokenExecutor(createMissingGpuStorageEvidenceHostLlamaExecutor(), "missing-gpu-storage-evidence", "missing GPU storage evidence");
    expectRejectedHostLlamaTokenExecutor(createMissingGpuStorageEvidenceHostLlamaExecutor(false), "false-gpu-storage-evidence", "false GPU storage evidence");
    expectRejectedHostLlamaTokenExecutor(createForgedGpuStorageHostLlamaExecutor(), "forged-gpu-storage", "forged GPU storage evidence");
    expectRejectedHostLlamaTokenExecutor(createPartialHostLlamaOutputExecutor(), "partial-output", "partial logits output", {
      outputPolicy: executeOutputLogits,
    });

    const invalidTableHashExecutor = createInvalidHostLlamaTableHashExecutor();
    const invalidTableHashProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      tokenExecutor: invalidTableHashExecutor,
    });
    const invalidTableHashSession = invalidTableHashProgram.bind(rawLlamaResources);
    const invalidTableHashSessionHandle = invalidTableHashSession.handle;
    try {
      const invalidTableHashTokens = keep(alloc(4), 4);
      writeU32Array(invalidTableHashTokens, [0]);
      const invalidTableHashExecute = tokenExecute(invalidTableHashSessionHandle, invalidTableHashTokens, 1, executeOutputNone, 0, 0);
      expectStatus(invalidTableHashExecute.status, invalidArgument, "WebGPU tiny LLaMA host-token executor invalid table-hash execute");
      if (invalidTableHashExecute.outputLen !== 0 || sessionPosition(invalidTableHashSessionHandle) !== 0) {
        throw new Error("WebGPU tiny LLaMA invalid-table-hash executor mutated state");
      }
      if (invalidTableHashExecutor.calls !== 1 || invalidTableHashSession.tokenExecutionAttempts !== 1) {
        throw new Error("WebGPU tiny LLaMA invalid-table-hash executor was not called exactly once");
      }
      expectLastLlamaTokenExecution(invalidTableHashSession, {
        tokens: [0],
        outputPolicy: executeOutputNone,
      }, "WebGPU tiny LLaMA invalid-table-hash executor");
      const invalidTableHashStepParams = invalidTableHashSession.stepParamsHistory.at(-1);
      if (
        invalidTableHashSession.stepParamsHistory.length !== 1 ||
        !invalidTableHashStepParams ||
        invalidTableHashStepParams.startPosition !== 0 ||
        invalidTableHashStepParams.endPosition !== 1 ||
        invalidTableHashStepParams.outputKind !== "none"
      ) {
        throw new Error("WebGPU tiny LLaMA invalid-table-hash executor did not retain rejected StepParams evidence");
      }
      const invalidTableHashInspection = inspectSession(invalidTableHashSessionHandle);
      if (u64(invalidTableHashInspection, 16) !== 0n) {
        throw new Error("WebGPU tiny LLaMA invalid-table-hash executor inspect position mutated");
      }
      expectRejectedHostLlamaExecutorProfile(
        invalidTableHashSession,
        invalidTableHashSessionHandle,
        "WebGPU tiny LLaMA host-token executor invalid table hash",
      );
    } finally {
      exportsRef.zgml_session_free(invalidTableHashSessionHandle);
      if (!invalidTableHashSession.freed || hostRuntime.lookup(invalidTableHashSessionHandle) !== null) {
        throw new Error("WebGPU tiny LLaMA invalid-table-hash host resource Session was not released through native free");
      }
      if (hostRuntime.hasClaimed(invalidTableHashSessionHandle)) {
        throw new Error("WebGPU tiny LLaMA invalid-table-hash host resource Session claim was not released through native free");
      }
    }

    const invalidModelHandleExecutor = createInvalidHostLlamaModelHandleExecutor();
    const invalidModelHandleProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      tokenExecutor: invalidModelHandleExecutor,
    });
    const invalidModelHandleSession = invalidModelHandleProgram.bind(rawLlamaResources);
    const invalidModelHandleSessionHandle = invalidModelHandleSession.handle;
    try {
      const invalidModelHandleTokens = keep(alloc(4), 4);
      writeU32Array(invalidModelHandleTokens, [0]);
      const invalidModelHandleExecute = tokenExecute(invalidModelHandleSessionHandle, invalidModelHandleTokens, 1, executeOutputNone, 0, 0);
      expectStatus(invalidModelHandleExecute.status, invalidArgument, "WebGPU tiny LLaMA host-token executor invalid model-handle execute");
      if (invalidModelHandleExecute.outputLen !== 0 || sessionPosition(invalidModelHandleSessionHandle) !== 0) {
        throw new Error("WebGPU tiny LLaMA invalid-model-handle executor mutated state");
      }
      if (invalidModelHandleExecutor.calls !== 1 || invalidModelHandleSession.tokenExecutionAttempts !== 1) {
        throw new Error("WebGPU tiny LLaMA invalid-model-handle executor was not called exactly once");
      }
      expectLastLlamaTokenExecution(invalidModelHandleSession, {
        tokens: [0],
        outputPolicy: executeOutputNone,
      }, "WebGPU tiny LLaMA invalid-model-handle executor");
      const invalidModelHandleStepParams = invalidModelHandleSession.stepParamsHistory.at(-1);
      if (
        invalidModelHandleSession.stepParamsHistory.length !== 1 ||
        !invalidModelHandleStepParams ||
        invalidModelHandleStepParams.startPosition !== 0 ||
        invalidModelHandleStepParams.endPosition !== 1 ||
        invalidModelHandleStepParams.outputKind !== "none"
      ) {
        throw new Error("WebGPU tiny LLaMA invalid-model-handle executor did not retain rejected StepParams evidence");
      }
      const invalidModelHandleInspection = inspectSession(invalidModelHandleSessionHandle);
      if (u64(invalidModelHandleInspection, 16) !== 0n) {
        throw new Error("WebGPU tiny LLaMA invalid-model-handle executor inspect position mutated");
      }
      expectRejectedHostLlamaExecutorProfile(
        invalidModelHandleSession,
        invalidModelHandleSessionHandle,
        "WebGPU tiny LLaMA host-token executor invalid model handle",
      );
    } finally {
      exportsRef.zgml_session_free(invalidModelHandleSessionHandle);
      if (!invalidModelHandleSession.freed || hostRuntime.lookup(invalidModelHandleSessionHandle) !== null) {
        throw new Error("WebGPU tiny LLaMA invalid-model-handle host resource Session was not released through native free");
      }
      if (hostRuntime.hasClaimed(invalidModelHandleSessionHandle)) {
        throw new Error("WebGPU tiny LLaMA invalid-model-handle host resource Session claim was not released through native free");
      }
    }

    const missingModelEvidenceExecutor = createMissingHostLlamaModelEvidenceExecutor();
    const missingModelEvidenceProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      tokenExecutor: missingModelEvidenceExecutor,
    });
    const missingModelEvidenceSession = missingModelEvidenceProgram.bind({
      ...rawLlamaResources,
      modelResources: llamaModelResources,
    });
    const missingModelEvidenceSessionHandle = missingModelEvidenceSession.handle;
    try {
      const missingModelEvidenceTokens = keep(alloc(4), 4);
      writeU32Array(missingModelEvidenceTokens, [0]);
      const missingModelEvidenceExecute = tokenExecute(missingModelEvidenceSessionHandle, missingModelEvidenceTokens, 1, executeOutputNone, 0, 0);
      expectStatus(missingModelEvidenceExecute.status, invalidArgument, "WebGPU tiny LLaMA host-token executor missing model-resource evidence execute");
      if (missingModelEvidenceExecute.outputLen !== 0 || sessionPosition(missingModelEvidenceSessionHandle) !== 0) {
        throw new Error("WebGPU tiny LLaMA missing-model-evidence executor mutated state");
      }
      if (missingModelEvidenceExecutor.calls !== 1 || missingModelEvidenceSession.tokenExecutionAttempts !== 1) {
        throw new Error("WebGPU tiny LLaMA missing-model-evidence executor was not called exactly once");
      }
      expectLastLlamaTokenExecution(missingModelEvidenceSession, {
        tokens: [0],
        outputPolicy: executeOutputNone,
      }, "WebGPU tiny LLaMA missing-model-evidence executor");
      const missingModelEvidenceStepParams = missingModelEvidenceSession.stepParamsHistory.at(-1);
      if (
        missingModelEvidenceSession.stepParamsHistory.length !== 1 ||
        !missingModelEvidenceStepParams ||
        missingModelEvidenceStepParams.startPosition !== 0 ||
        missingModelEvidenceStepParams.endPosition !== 1 ||
        missingModelEvidenceStepParams.outputKind !== "none"
      ) {
        throw new Error("WebGPU tiny LLaMA missing-model-evidence executor did not retain rejected StepParams evidence");
      }
      const missingModelEvidenceInspection = inspectSession(missingModelEvidenceSessionHandle);
      if (u64(missingModelEvidenceInspection, 16) !== 0n) {
        throw new Error("WebGPU tiny LLaMA missing-model-evidence executor inspect position mutated");
      }
      expectRejectedHostLlamaExecutorProfile(
        missingModelEvidenceSession,
        missingModelEvidenceSessionHandle,
        "WebGPU tiny LLaMA host-token executor missing model-resource evidence",
      );
    } finally {
      exportsRef.zgml_session_free(missingModelEvidenceSessionHandle);
      if (!missingModelEvidenceSession.freed || hostRuntime.lookup(missingModelEvidenceSessionHandle) !== null) {
        throw new Error("WebGPU tiny LLaMA missing-model-evidence host resource Session was not released through native free");
      }
      if (hostRuntime.hasClaimed(missingModelEvidenceSessionHandle)) {
        throw new Error("WebGPU tiny LLaMA missing-model-evidence host resource Session claim was not released through native free");
      }
    }

    for (const implicitModelEvidenceProof of [
      { kind: "undefined", label: "undefined-result" },
      { kind: "integer-ok", label: "integer-ok-result" },
    ]) {
      const implicitModelEvidenceExecutor = createImplicitHostLlamaModelEvidenceExecutor(implicitModelEvidenceProof.kind);
      const implicitModelEvidenceProgram = new WasmWebGpuLlamaResourceProgram({
        device: hostDevice,
        resourceBridge: hostResourceBridge,
        sessionBindingBridge,
        runtime: hostRuntime,
        program,
        vocabSize: tinyLlamaVocabSize,
        contextLength: 4,
        kvCacheRequirements: {
          layers: u32(kvRequirements, 8),
          kBufferByteLength: u32(kvRequirements, 20),
          vBufferByteLength: u32(kvRequirements, 24),
        },
        tokenExecutor: implicitModelEvidenceExecutor,
      });
      const implicitModelEvidenceSession = implicitModelEvidenceProgram.bind({
        ...rawLlamaResources,
        modelResources: llamaModelResources,
      });
      const implicitModelEvidenceSessionHandle = implicitModelEvidenceSession.handle;
      try {
        const implicitModelEvidenceTokens = keep(alloc(4), 4);
        writeU32Array(implicitModelEvidenceTokens, [0]);
        const implicitModelEvidenceExecute = tokenExecute(implicitModelEvidenceSessionHandle, implicitModelEvidenceTokens, 1, executeOutputNone, 0, 0);
        expectStatus(
          implicitModelEvidenceExecute.status,
          invalidArgument,
          `WebGPU tiny LLaMA host-token executor ${implicitModelEvidenceProof.label} missing model-resource evidence execute`,
        );
        if (implicitModelEvidenceExecute.outputLen !== 0 || sessionPosition(implicitModelEvidenceSessionHandle) !== 0) {
          throw new Error(`WebGPU tiny LLaMA ${implicitModelEvidenceProof.label} missing-model-evidence executor mutated state`);
        }
        if (implicitModelEvidenceExecutor.calls !== 1 || implicitModelEvidenceSession.tokenExecutionAttempts !== 1) {
          throw new Error(`WebGPU tiny LLaMA ${implicitModelEvidenceProof.label} missing-model-evidence executor was not called exactly once`);
        }
        expectLastLlamaTokenExecution(implicitModelEvidenceSession, {
          tokens: [0],
          outputPolicy: executeOutputNone,
        }, `WebGPU tiny LLaMA ${implicitModelEvidenceProof.label} missing-model-evidence executor`);
        const implicitModelEvidenceStepParams = implicitModelEvidenceSession.stepParamsHistory.at(-1);
        if (
          implicitModelEvidenceSession.stepParamsHistory.length !== 1 ||
          !implicitModelEvidenceStepParams ||
          implicitModelEvidenceStepParams.startPosition !== 0 ||
          implicitModelEvidenceStepParams.endPosition !== 1 ||
          implicitModelEvidenceStepParams.outputKind !== "none"
        ) {
          throw new Error(`WebGPU tiny LLaMA ${implicitModelEvidenceProof.label} missing-model-evidence executor did not retain rejected StepParams evidence`);
        }
        const implicitModelEvidenceInspection = inspectSession(implicitModelEvidenceSessionHandle);
        if (u64(implicitModelEvidenceInspection, 16) !== 0n) {
          throw new Error(`WebGPU tiny LLaMA ${implicitModelEvidenceProof.label} missing-model-evidence executor inspect position mutated`);
        }
        expectRejectedHostLlamaExecutorProfile(
          implicitModelEvidenceSession,
          implicitModelEvidenceSessionHandle,
          `WebGPU tiny LLaMA host-token executor ${implicitModelEvidenceProof.label} missing model-resource evidence`,
        );
      } finally {
        exportsRef.zgml_session_free(implicitModelEvidenceSessionHandle);
        if (!implicitModelEvidenceSession.freed || hostRuntime.lookup(implicitModelEvidenceSessionHandle) !== null) {
          throw new Error(`WebGPU tiny LLaMA ${implicitModelEvidenceProof.label} missing-model-evidence host resource Session was not released through native free`);
        }
        if (hostRuntime.hasClaimed(implicitModelEvidenceSessionHandle)) {
          throw new Error(`WebGPU tiny LLaMA ${implicitModelEvidenceProof.label} missing-model-evidence host resource Session claim was not released through native free`);
        }
      }
    }

    const invalidModelTableHashExecutor = createInvalidHostLlamaModelTableHashExecutor();
    const invalidModelTableHashProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      tokenExecutor: invalidModelTableHashExecutor,
    });
    const invalidModelTableHashSession = invalidModelTableHashProgram.bind({
      ...rawLlamaResources,
      modelResources: llamaModelResources,
    });
    const invalidModelTableHashSessionHandle = invalidModelTableHashSession.handle;
    try {
      const invalidModelTableHashTokens = keep(alloc(4), 4);
      writeU32Array(invalidModelTableHashTokens, [0]);
      const invalidModelTableHashExecute = tokenExecute(invalidModelTableHashSessionHandle, invalidModelTableHashTokens, 1, executeOutputNone, 0, 0);
      expectStatus(invalidModelTableHashExecute.status, invalidArgument, "WebGPU tiny LLaMA host-token executor invalid model-table-hash execute");
      if (invalidModelTableHashExecute.outputLen !== 0 || sessionPosition(invalidModelTableHashSessionHandle) !== 0) {
        throw new Error("WebGPU tiny LLaMA invalid-model-table-hash executor mutated state");
      }
      if (invalidModelTableHashExecutor.calls !== 1 || invalidModelTableHashSession.tokenExecutionAttempts !== 1) {
        throw new Error("WebGPU tiny LLaMA invalid-model-table-hash executor was not called exactly once");
      }
      expectLastLlamaTokenExecution(invalidModelTableHashSession, {
        tokens: [0],
        outputPolicy: executeOutputNone,
      }, "WebGPU tiny LLaMA invalid-model-table-hash executor");
      const invalidModelTableHashStepParams = invalidModelTableHashSession.stepParamsHistory.at(-1);
      if (
        invalidModelTableHashSession.stepParamsHistory.length !== 1 ||
        !invalidModelTableHashStepParams ||
        invalidModelTableHashStepParams.startPosition !== 0 ||
        invalidModelTableHashStepParams.endPosition !== 1 ||
        invalidModelTableHashStepParams.outputKind !== "none"
      ) {
        throw new Error("WebGPU tiny LLaMA invalid-model-table-hash executor did not retain rejected StepParams evidence");
      }
      const invalidModelTableHashInspection = inspectSession(invalidModelTableHashSessionHandle);
      if (u64(invalidModelTableHashInspection, 16) !== 0n) {
        throw new Error("WebGPU tiny LLaMA invalid-model-table-hash executor inspect position mutated");
      }
      expectRejectedHostLlamaExecutorProfile(
        invalidModelTableHashSession,
        invalidModelTableHashSessionHandle,
        "WebGPU tiny LLaMA host-token executor invalid model table hash",
      );
    } finally {
      exportsRef.zgml_session_free(invalidModelTableHashSessionHandle);
      if (!invalidModelTableHashSession.freed || hostRuntime.lookup(invalidModelTableHashSessionHandle) !== null) {
        throw new Error("WebGPU tiny LLaMA invalid-model-table-hash host resource Session was not released through native free");
      }
      if (hostRuntime.hasClaimed(invalidModelTableHashSessionHandle)) {
        throw new Error("WebGPU tiny LLaMA invalid-model-table-hash host resource Session claim was not released through native free");
      }
    }

    const invalidModelManifestHashExecutor = createInvalidHostLlamaModelManifestHashExecutor();
    const invalidModelManifestHashProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      tokenExecutor: invalidModelManifestHashExecutor,
    });
    const invalidModelManifestHashSession = invalidModelManifestHashProgram.bind({
      ...rawLlamaResources,
      modelResources: llamaModelResources,
    });
    const invalidModelManifestHashSessionHandle = invalidModelManifestHashSession.handle;
    try {
      const invalidModelManifestHashTokens = keep(alloc(4), 4);
      writeU32Array(invalidModelManifestHashTokens, [0]);
      const invalidModelManifestHashExecute = tokenExecute(invalidModelManifestHashSessionHandle, invalidModelManifestHashTokens, 1, executeOutputNone, 0, 0);
      expectStatus(invalidModelManifestHashExecute.status, invalidArgument, "WebGPU tiny LLaMA host-token executor invalid model-manifest-hash execute");
      if (invalidModelManifestHashExecute.outputLen !== 0 || sessionPosition(invalidModelManifestHashSessionHandle) !== 0) {
        throw new Error("WebGPU tiny LLaMA invalid-model-manifest-hash executor mutated state");
      }
      if (invalidModelManifestHashExecutor.calls !== 1 || invalidModelManifestHashSession.tokenExecutionAttempts !== 1) {
        throw new Error("WebGPU tiny LLaMA invalid-model-manifest-hash executor was not called exactly once");
      }
      expectLastLlamaTokenExecution(invalidModelManifestHashSession, {
        tokens: [0],
        outputPolicy: executeOutputNone,
      }, "WebGPU tiny LLaMA invalid-model-manifest-hash executor");
      const invalidModelManifestHashStepParams = invalidModelManifestHashSession.stepParamsHistory.at(-1);
      if (
        invalidModelManifestHashSession.stepParamsHistory.length !== 1 ||
        !invalidModelManifestHashStepParams ||
        invalidModelManifestHashStepParams.startPosition !== 0 ||
        invalidModelManifestHashStepParams.endPosition !== 1 ||
        invalidModelManifestHashStepParams.outputKind !== "none"
      ) {
        throw new Error("WebGPU tiny LLaMA invalid-model-manifest-hash executor did not retain rejected StepParams evidence");
      }
      const invalidModelManifestHashInspection = inspectSession(invalidModelManifestHashSessionHandle);
      if (u64(invalidModelManifestHashInspection, 16) !== 0n) {
        throw new Error("WebGPU tiny LLaMA invalid-model-manifest-hash executor inspect position mutated");
      }
      expectRejectedHostLlamaExecutorProfile(
        invalidModelManifestHashSession,
        invalidModelManifestHashSessionHandle,
        "WebGPU tiny LLaMA host-token executor invalid model manifest hash",
      );
    } finally {
      exportsRef.zgml_session_free(invalidModelManifestHashSessionHandle);
      if (!invalidModelManifestHashSession.freed || hostRuntime.lookup(invalidModelManifestHashSessionHandle) !== null) {
        throw new Error("WebGPU tiny LLaMA invalid-model-manifest-hash host resource Session was not released through native free");
      }
      if (hostRuntime.hasClaimed(invalidModelManifestHashSessionHandle)) {
        throw new Error("WebGPU tiny LLaMA invalid-model-manifest-hash host resource Session claim was not released through native free");
      }
    }

    const missingModelPackEvidenceExecutor = createMissingHostLlamaModelPackEvidenceExecutor();
    const missingModelPackEvidenceProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      tokenExecutor: missingModelPackEvidenceExecutor,
    });
    const missingModelPackEvidenceSession = missingModelPackEvidenceProgram.bind({
      ...rawLlamaResources,
      modelResources: llamaModelResources,
    });
    const missingModelPackEvidenceSessionHandle = missingModelPackEvidenceSession.handle;
    try {
      const missingModelPackEvidenceOwnedResourceCount = missingModelPackEvidenceSession.ownedResources.length;
      const missingModelPackEvidenceDeviceResourceCount = hostDevice.ownedResources.length;
      const missingModelPackEvidenceTokens = keep(alloc(4), 4);
      writeU32Array(missingModelPackEvidenceTokens, [0]);
      const missingModelPackEvidenceExecute = tokenExecute(missingModelPackEvidenceSessionHandle, missingModelPackEvidenceTokens, 1, executeOutputNone, 0, 0);
      expectStatus(missingModelPackEvidenceExecute.status, invalidArgument, "WebGPU tiny LLaMA host-token executor missing model-pack evidence execute");
      if (missingModelPackEvidenceExecute.outputLen !== 0 || sessionPosition(missingModelPackEvidenceSessionHandle) !== 0) {
        throw new Error("WebGPU tiny LLaMA missing-model-pack-evidence executor mutated state");
      }
      if (
        missingModelPackEvidenceExecutor.calls !== 1 ||
        missingModelPackEvidenceSession.tokenExecutionAttempts !== 1 ||
        missingModelPackEvidenceSession.modelPacks.length !== 0 ||
        missingModelPackEvidenceSession.ownedResources.length !== missingModelPackEvidenceOwnedResourceCount ||
        hostDevice.ownedResources.length !== missingModelPackEvidenceDeviceResourceCount ||
        missingModelPackEvidenceExecutor.rejectedResource?.destroyed !== true
      ) {
        throw new Error("WebGPU tiny LLaMA missing-model-pack-evidence executor did not roll back rejected pack/resource evidence");
      }
      expectLastLlamaTokenExecution(missingModelPackEvidenceSession, {
        tokens: [0],
        outputPolicy: executeOutputNone,
      }, "WebGPU tiny LLaMA missing-model-pack-evidence executor");
      const missingModelPackEvidenceStepParams = missingModelPackEvidenceSession.stepParamsHistory.at(-1);
      if (
        missingModelPackEvidenceSession.stepParamsHistory.length !== 1 ||
        !missingModelPackEvidenceStepParams ||
        missingModelPackEvidenceStepParams.startPosition !== 0 ||
        missingModelPackEvidenceStepParams.endPosition !== 1 ||
        missingModelPackEvidenceStepParams.outputKind !== "none"
      ) {
        throw new Error("WebGPU tiny LLaMA missing-model-pack-evidence executor did not retain rejected StepParams evidence");
      }
      const missingModelPackEvidenceInspection = inspectSession(missingModelPackEvidenceSessionHandle);
      if (u64(missingModelPackEvidenceInspection, 16) !== 0n) {
        throw new Error("WebGPU tiny LLaMA missing-model-pack-evidence executor inspect position mutated");
      }
      expectRejectedHostLlamaExecutorProfile(
        missingModelPackEvidenceSession,
        missingModelPackEvidenceSessionHandle,
        "WebGPU tiny LLaMA host-token executor missing model pack evidence",
      );
    } finally {
      exportsRef.zgml_session_free(missingModelPackEvidenceSessionHandle);
      if (!missingModelPackEvidenceSession.freed || hostRuntime.lookup(missingModelPackEvidenceSessionHandle) !== null) {
        throw new Error("WebGPU tiny LLaMA missing-model-pack-evidence host resource Session was not released through native free");
      }
      if (hostRuntime.hasClaimed(missingModelPackEvidenceSessionHandle)) {
        throw new Error("WebGPU tiny LLaMA missing-model-pack-evidence host resource Session claim was not released through native free");
      }
    }

    const invalidModelPackHashExecutor = createInvalidHostLlamaModelPackHashExecutor();
    const invalidModelPackHashProgram = new WasmWebGpuLlamaResourceProgram({
      device: hostDevice,
      resourceBridge: hostResourceBridge,
      sessionBindingBridge,
      runtime: hostRuntime,
      program,
      vocabSize: tinyLlamaVocabSize,
      contextLength: 4,
      kvCacheRequirements: {
        layers: u32(kvRequirements, 8),
        kBufferByteLength: u32(kvRequirements, 20),
        vBufferByteLength: u32(kvRequirements, 24),
      },
      tokenExecutor: invalidModelPackHashExecutor,
    });
    const invalidModelPackHashSession = invalidModelPackHashProgram.bind({
      ...rawLlamaResources,
      modelResources: llamaModelResources,
    });
    const invalidModelPackHashSessionHandle = invalidModelPackHashSession.handle;
    try {
      const invalidModelPackHashTokens = keep(alloc(4), 4);
      writeU32Array(invalidModelPackHashTokens, [0]);
      const invalidModelPackHashExecute = tokenExecute(invalidModelPackHashSessionHandle, invalidModelPackHashTokens, 1, executeOutputNone, 0, 0);
      expectStatus(invalidModelPackHashExecute.status, invalidArgument, "WebGPU tiny LLaMA host-token executor invalid model-pack-hash execute");
      if (invalidModelPackHashExecute.outputLen !== 0 || sessionPosition(invalidModelPackHashSessionHandle) !== 0) {
        throw new Error("WebGPU tiny LLaMA invalid-model-pack-hash executor mutated state");
      }
      if (invalidModelPackHashExecutor.calls !== 1 || invalidModelPackHashSession.tokenExecutionAttempts !== 1) {
        throw new Error("WebGPU tiny LLaMA invalid-model-pack-hash executor was not called exactly once");
      }
      expectLastLlamaTokenExecution(invalidModelPackHashSession, {
        tokens: [0],
        outputPolicy: executeOutputNone,
      }, "WebGPU tiny LLaMA invalid-model-pack-hash executor");
      const invalidModelPackHashStepParams = invalidModelPackHashSession.stepParamsHistory.at(-1);
      if (
        invalidModelPackHashSession.stepParamsHistory.length !== 1 ||
        !invalidModelPackHashStepParams ||
        invalidModelPackHashStepParams.startPosition !== 0 ||
        invalidModelPackHashStepParams.endPosition !== 1 ||
        invalidModelPackHashStepParams.outputKind !== "none"
      ) {
        throw new Error("WebGPU tiny LLaMA invalid-model-pack-hash executor did not retain rejected StepParams evidence");
      }
      const invalidModelPackHashInspection = inspectSession(invalidModelPackHashSessionHandle);
      if (u64(invalidModelPackHashInspection, 16) !== 0n) {
        throw new Error("WebGPU tiny LLaMA invalid-model-pack-hash executor inspect position mutated");
      }
      expectRejectedHostLlamaExecutorProfile(
        invalidModelPackHashSession,
        invalidModelPackHashSessionHandle,
        "WebGPU tiny LLaMA host-token executor invalid model pack hash",
      );
    } finally {
      exportsRef.zgml_session_free(invalidModelPackHashSessionHandle);
      if (!invalidModelPackHashSession.freed || hostRuntime.lookup(invalidModelPackHashSessionHandle) !== null) {
        throw new Error("WebGPU tiny LLaMA invalid-model-pack-hash host resource Session was not released through native free");
      }
      if (hostRuntime.hasClaimed(invalidModelPackHashSessionHandle)) {
        throw new Error("WebGPU tiny LLaMA invalid-model-pack-hash host resource Session claim was not released through native free");
      }
    }
    }

    append("zgml browser wasm ffi webgpu resource-probe tiny llama smoke ok");
  } finally {
    if (program !== 0) exportsRef.zgml_program_free(program);
    if (compatibleModel !== 0) exportsRef.zgml_model_free(compatibleModel);
    if (model !== 0) exportsRef.zgml_model_free(model);
  }
}

function inspectProgram(program) {
  const inspection = keep(alloc(programInspectionSize), programInspectionSize);
  check(exportsRef.zgml_program_inspect(program, inspection));
  return inspection;
}

function programCapabilities(program) {
  const inspection = inspectProgram(program);
  const canExecute = u64(inspection, 8) !== 0n;
  const canBindExternalResources = u64(inspection, 16) !== 0n;
  const mode = canExecute ? "executable" : (canBindExternalResources ? "resource-probe" : "compile-only");
  const dispatchPlanSupported = u64(inspection, 200) !== 0n;
  const opCount = u64(inspection, 160);
  const coveredOpCount = u64(inspection, 208);
  const firstUnsupported = u64(inspection, 216);
  return {
    backend: Number(u64(inspection, 0)),
    mode,
    canExecute,
    canBindExternalResources,
    hasFullDispatchPlan: dispatchPlanSupported && coveredOpCount === opCount && firstUnsupported === 0xffffffffffffffffn,
    backendDispatchCount: Number(u64(inspection, 192)),
    dispatchPlanSupported,
    dispatchPlanCoveredOpCount: Number(coveredOpCount),
    dispatchPlanFirstUnsupportedOp: firstUnsupported === 0xffffffffffffffffn ? null : Number(firstUnsupported),
    dispatchPlanProjectionCount: Number(u64(inspection, 224)),
    dispatchPlanRowCount: Number(u64(inspection, 232)),
    dispatchPlanAttentionCount: Number(u64(inspection, 240)),
    dispatchPlanMovementCount: Number(u64(inspection, 248)),
    dispatchPlanElementwiseCount: Number(u64(inspection, 256)),
    dispatchPlanRopeCount: Number(u64(inspection, 264)),
    dispatchPlanQuantizedProjectionCount: Number(u64(inspection, 272)),
  };
}

function programRequirements(program) {
  const requirements = keep(alloc(programRequirementsSize), programRequirementsSize);
  check(exportsRef.zgml_program_get_requirements(program, requirements));
  return requirements;
}

function requirementInputLen(requirements) {
  return u32(requirements, 16);
}

function requirementOutputLen(requirements) {
  return u32(requirements, 24);
}

function requirementWeightsLen(requirements) {
  return u32(requirements, 28);
}

function requirementBiasLen(requirements) {
  return u32(requirements, 36);
}

function requirementLogitsLen(requirements) {
  return u32(requirements, 52);
}

function requirementOutputByteLen(requirements) {
  return u32(requirements, 56);
}

function requirementContextLen(requirements) {
  return u32(requirements, 60);
}

function requirementBatch(requirements) {
  return u32(requirements, 64);
}

function requirementMaxTokenWindow(requirements) {
  return u32(requirements, 68);
}

function programRuntimeProfile(program) {
  const profile = keep(alloc(runtimeProfileSize), runtimeProfileSize);
  check(exportsRef.zgml_program_runtime_profile(program, profile));
  return profile;
}

function sessionRuntimeProfile(session) {
  return sessionRuntimeProfileWith(exportsRef, session);
}

function sessionRuntimeProfileWith(exportsLike, session) {
  const profile = keep(alloc(runtimeProfileSize), runtimeProfileSize);
  check(exportsLike.zgml_session_runtime_profile(session, profile));
  return profile;
}

function expectNoRuntimeWork(session, label) {
  const profile = sessionRuntimeProfile(session);
  if (
    u64(profile, 0) !== 0n ||
    u64(profile, 8) !== 0n ||
    u64(profile, 16) !== 0n ||
    u64(profile, 32) !== 0n ||
    u64(profile, 40) !== 0n
  ) {
    throw new Error(`${label} recorded runtime work`);
  }
}

function expectHostTinyLinearProfile(profile, expected, label) {
  const fields = [
    "callCount",
    "backendDispatchCount",
    "noOutputCallCount",
    "outputElementCount",
    "outputReadCount",
    "syncCount",
    "lastOutputLength",
  ];
  for (const field of fields) {
    if (profile[field] !== expected[field]) {
      throw new Error(`${label} host profile ${field}: expected ${expected[field]}, got ${profile[field]}`);
    }
  }
}

function expectHostLlamaResourceProfile(profile, expected, label) {
  const fields = [
    "callCount",
    "decodedCallCount",
    "logitsOutputCallCount",
    "noOutputCallCount",
    "callerOutputCallCount",
    "boundOutputCallCount",
    "requestedOutputElementCount",
    "resultWriteCount",
    "validationFailureCount",
    "invalidArgumentCount",
    "unsupportedCallCount",
    "lastStatus",
    "lastOutputPolicy",
    "lastOutputLength",
    "lastResultOutputLength",
  ];
  for (const field of fields) {
    if (profile[field] !== expected[field]) {
      throw new Error(`${label} host profile ${field}: expected ${expected[field]}, got ${profile[field]}`);
    }
  }
}

function hostLlamaKvWindowValues(kind, layer, stepParams, length) {
  const base = (kind === "k" ? 1000 : 2000) + layer * 100 + stepParams.startPosition * 10;
  return Array.from({ length }, (_unused, i) => base + i);
}

async function expectHostLlamaKvWindow(session, stepParams, label) {
  const kvCache = session.bindings.kvCache ?? [];
  for (let layer = 0; layer < kvCache.length; layer += 1) {
    for (const kind of ["k", "v"]) {
      const descriptor = kvCache[layer][kind];
      const elementCount = Math.floor(descriptor.byteLength / Float32Array.BYTES_PER_ELEMENT);
      const stride = Math.floor(elementCount / stepParams.contextLength);
      const offset = stepParams.startPosition * stride * Float32Array.BYTES_PER_ELEMENT;
      const length = stepParams.tokenCount * stride;
      const actual = Array.from(await session.device.readFloat32(descriptor, length, offset));
      const expected = hostLlamaKvWindowValues(kind, layer, stepParams, length);
      expectClose(actual, expected, `${label} ${kind}[${layer}]`);
    }
  }
}

function createHostLlamaTokenExecutor(label) {
  const calls = [];
  const scratchSnapshots = [];
  let executor;
  executor = new WasmWebGpuLlamaTokenWindowPatternExecutor({
    record(call) {
      calls.push(call);
      scratchSnapshots.push({
        kv: executor.kvPatternScratch,
        logits: executor.logitsPatternScratch,
      });
    },
  });
  executor.calls = calls;
  executor.scratchSnapshots = scratchSnapshots;
  executor.expect = function expect(calls, expectedLabel = label) {
    if (this.calls.length !== calls.length) {
      throw new Error(`${expectedLabel} executor call count: expected ${calls.length}, got ${this.calls.length}`);
    }
    for (let i = 0; i < calls.length; i += 1) {
      const actual = this.calls[i];
      const expected = calls[i];
      const expectedCommandCount = expected.outputKind === "bound-logits" ? 3 : 2;
      const expectedModelHandle = expected.modelHandle ?? 0;
      const expectedModelResourceRoles = expected.modelResourceRoles ?? [];
      const expectedTableRoles = expected.tableRoles ?? ["llama.output", "llama.k.0", "llama.v.0"];
      if (
        actual.backendDispatchCount + actual.fallbackOpCount !== actual.commandCount ||
        actual.commandCount !== expectedCommandCount ||
        actual.endPosition !== expected.position + expected.tokens.length ||
        actual.kvWriteCount !== 2 ||
        actual.logitsLength !== (expected.outputPolicy === executeOutputLogits ? tinyLlamaVocabSize : 0) ||
        actual.modelBindingKind !== (expectedModelHandle === 0 ? "program-default" : "compatible-model") ||
        actual.modelHandle !== expectedModelHandle ||
        (expectedModelResourceRoles.length === 0 ? actual.modelManifestHash !== 0n : actual.modelManifestHash === 0n) ||
        (expectedModelResourceRoles.length === 0 ? actual.modelTableDescriptorHash !== 0n : actual.modelTableDescriptorHash === 0n) ||
        (expectedModelResourceRoles.length === 0 ? actual.modelTableHash !== 0n : actual.modelTableHash === 0n) ||
        !Array.isArray(actual.modelResourceRoles) ||
        actual.modelResourceRoles.length !== expectedModelResourceRoles.length ||
        actual.modelResourceRoles.some((role, j) => role !== expectedModelResourceRoles[j]) ||
        actual.outputKind !== expected.outputKind ||
        actual.outputPolicy !== expected.outputPolicy ||
        actual.position !== expected.position ||
        actual.tableDescriptorHash === 0n ||
        actual.tableHash === 0n ||
        !Array.isArray(actual.tableRoles) ||
        actual.tableRoles.length !== expectedTableRoles.length ||
        actual.tableRoles.some((role, j) => role !== expectedTableRoles[j]) ||
        actual.tokenCount !== expected.tokens.length ||
        actual.tokens.length !== expected.tokens.length
      ) {
        throw new Error(`${expectedLabel} executor call ${i} metadata mismatch`);
      }
      for (let j = 0; j < expected.tokens.length; j += 1) {
        if (actual.tokens[j] !== expected.tokens[j]) {
          throw new Error(`${expectedLabel} executor call ${i} token ${j}: expected ${expected.tokens[j]}, got ${actual.tokens[j]}`);
        }
      }
    }
  };
  executor.expectPatternScratchReuse = function expectPatternScratchReuse(vocabSize, expectedLabel = label) {
    if (this.scratchSnapshots.length !== this.calls.length || this.scratchSnapshots.length === 0) {
      throw new Error(`${expectedLabel} fallback scratch evidence mismatch`);
    }
    const firstKvScratch = this.scratchSnapshots[0].kv;
    if (!(firstKvScratch instanceof Float32Array) || firstKvScratch.length === 0) {
      throw new Error(`${expectedLabel} K/V fallback scratch was not retained`);
    }
    for (let i = 1; i < this.scratchSnapshots.length; i += 1) {
      if (this.scratchSnapshots[i].kv !== firstKvScratch) {
        throw new Error(`${expectedLabel} K/V fallback scratch was not reused`);
      }
    }
    const logitsSnapshots = [];
    for (let i = 0; i < this.calls.length; i += 1) {
      if (this.calls[i].outputPolicy === executeOutputLogits) {
        logitsSnapshots.push(this.scratchSnapshots[i].logits);
      }
    }
    if (logitsSnapshots.length === 0) return;
    const firstLogitsScratch = logitsSnapshots[0];
    if (!(firstLogitsScratch instanceof Float32Array) || firstLogitsScratch.length !== vocabSize) {
      throw new Error(`${expectedLabel} logits fallback scratch was not retained`);
    }
    for (let i = 1; i < logitsSnapshots.length; i += 1) {
      if (logitsSnapshots[i] !== firstLogitsScratch) {
        throw new Error(`${expectedLabel} logits fallback scratch was not reused`);
      }
    }
  };
  return executor;
}

function createInvalidHostLlamaCounterExecutor() {
  return {
    calls: 0,
    executeTokensSync() {
      this.calls += 1;
      return {
        status: ok,
        outputLength: 0,
        backendDispatchCount: -1,
        fallbackOpCount: 0,
        commandCount: 0,
        commandOpCount: 0,
      };
    },
  };
}

function createInvalidHostLlamaStatusExecutor() {
  return {
    calls: 0,
    executeTokensSync(context) {
      this.calls += 1;
      return {
        status: context.runtime.status.unclaimed,
        outputLength: 0,
        backendDispatchCount: 0,
        fallbackOpCount: 0,
        commandCount: 0,
        commandOpCount: 0,
        tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
        tableHash: context.session.table?.hash ?? 0n,
      };
    },
  };
}

function createMissingHostLlamaTableEvidenceExecutor() {
  return {
    calls: 0,
    executeTokensSync() {
      this.calls += 1;
      return {
        status: ok,
        outputLength: 0,
        backendDispatchCount: 0,
        fallbackOpCount: 0,
        commandCount: 0,
        commandOpCount: 0,
      };
    },
  };
}

function createIncoherentHostLlamaCounterExecutor() {
  return {
    calls: 0,
    executeTokensSync(context) {
      this.calls += 1;
      return {
        status: ok,
        outputLength: 0,
        backendDispatchCount: 1,
        backendOpCount: 1,
        fallbackOpCount: 1,
        commandCount: 1,
        commandOpCount: 1,
        tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
        tableHash: context.session.table?.hash ?? 0n,
      };
    },
  };
}

function createMissingCommandOpHostLlamaCounterExecutor() {
  return {
    calls: 0,
    executeTokensSync(context) {
      this.calls += 1;
      return {
        status: ok,
        outputLength: 0,
        backendDispatchCount: 1,
        backendOpCount: 1,
        fallbackOpCount: 0,
        commandCount: 1,
        commandOpCount: 0,
        tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
        tableHash: context.session.table?.hash ?? 0n,
      };
    },
  };
}

function createFailedWorkHostLlamaExecutor() {
  return {
    calls: 0,
    executeTokensSync(context) {
      this.calls += 1;
      return {
        status: context.runtime.status.unsupported,
        outputLength: 0,
        backendDispatchCount: 1,
        backendOpCount: 1,
        fallbackOpCount: 0,
        commandCount: 1,
        commandOpCount: 1,
        tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
        tableHash: context.session.table?.hash ?? 0n,
      };
    },
  };
}

function createPartialHostLlamaOutputExecutor() {
  return {
    calls: 0,
    executeTokensSync(context) {
      this.calls += 1;
      return {
        status: ok,
        outputLength: context.session.vocabSize - 1,
        backendDispatchCount: 0,
        backendOpCount: 0,
        fallbackOpCount: 0,
        commandCount: 0,
        commandOpCount: 0,
        tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
        tableHash: context.session.table?.hash ?? 0n,
      };
    },
  };
}

function createZeroWorkHostLlamaExecutor() {
  return {
    calls: 0,
    executeTokensSync(context) {
      this.calls += 1;
      return {
        status: ok,
        outputLength: 0,
        backendDispatchCount: 0,
        backendOpCount: 0,
        fallbackOpCount: 0,
        commandCount: 0,
        commandOpCount: 0,
        tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
        tableHash: context.session.table?.hash ?? 0n,
      };
    },
  };
}

function createForgedGpuStorageHostLlamaExecutor() {
  return {
    calls: 0,
    requiredGpuStorageBufferCount() {
      return 0x7fffffff;
    },
    executeTokensSync(context) {
      this.calls += 1;
      return {
        status: ok,
        outputLength: 0,
        backendDispatchCount: 1,
        backendOpCount: 1,
        fallbackOpCount: 0,
        commandCount: 1,
        commandOpCount: 1,
        tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
        tableHash: context.session.table?.hash ?? 0n,
        usesGpuStorageBuffers: true,
      };
    },
  };
}

function createMissingGpuStorageEvidenceHostLlamaExecutor(usesGpuStorageBuffers = undefined) {
  return {
    calls: 0,
    requiredGpuStorageBufferCount() {
      return 6;
    },
    executeTokensSync(context) {
      this.calls += 1;
      const result = {
        status: ok,
        outputLength: 0,
        backendDispatchCount: 1,
        backendOpCount: 1,
        fallbackOpCount: 0,
        commandCount: 1,
        commandOpCount: 1,
        tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
        tableHash: context.session.table?.hash ?? 0n,
      };
      if (usesGpuStorageBuffers !== undefined) {
        result.usesGpuStorageBuffers = usesGpuStorageBuffers;
      }
      return result;
    },
  };
}

function createInvalidHostLlamaTableHashExecutor() {
  return {
    calls: 0,
    executeTokensSync(context) {
      this.calls += 1;
      return {
        status: ok,
        outputLength: 0,
        backendDispatchCount: 0,
        fallbackOpCount: 0,
        commandCount: 0,
        commandOpCount: 0,
        tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
        tableHash: (context.session.table?.hash ?? 0n) ^ 1n,
      };
    },
  };
}

function createInvalidHostLlamaModelHandleExecutor() {
  return {
    calls: 0,
    executeTokensSync(context) {
      this.calls += 1;
      return {
        status: ok,
        outputLength: 0,
        backendDispatchCount: 0,
        fallbackOpCount: 0,
        commandCount: 0,
        commandOpCount: 0,
        modelHandle: (context.session.modelHandle + 1) >>> 0,
        tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
        tableHash: context.session.table?.hash ?? 0n,
      };
    },
  };
}

function createMissingHostLlamaModelEvidenceExecutor() {
  return {
    calls: 0,
    executeTokensSync(context) {
      this.calls += 1;
      return {
        status: ok,
        outputLength: 0,
        backendDispatchCount: 0,
        fallbackOpCount: 0,
        commandCount: 0,
        commandOpCount: 0,
        modelHandle: context.session.modelHandle,
        tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
        tableHash: context.session.table?.hash ?? 0n,
      };
    },
  };
}

function createImplicitHostLlamaModelEvidenceExecutor(kind) {
  return {
    calls: 0,
    executeTokensSync() {
      this.calls += 1;
      if (kind === "undefined") return undefined;
      if (kind === "integer-ok") return ok;
      throw new Error(`unsupported implicit model evidence proof kind: ${kind}`);
    },
  };
}

function createInvalidHostLlamaModelTableHashExecutor() {
  return {
    calls: 0,
    executeTokensSync(context) {
      this.calls += 1;
      return {
        status: ok,
        outputLength: 0,
        backendDispatchCount: 0,
        fallbackOpCount: 0,
        commandCount: 0,
        commandOpCount: 0,
        modelHandle: context.session.modelHandle,
        modelTableDescriptorHash: context.session.modelTable?.descriptorHash ?? 0n,
        modelTableHash: (context.session.modelTable?.hash ?? 0n) ^ 1n,
        tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
        tableHash: context.session.table?.hash ?? 0n,
      };
    },
  };
}

function createInvalidHostLlamaModelManifestHashExecutor() {
  return {
    calls: 0,
    executeTokensSync(context) {
      this.calls += 1;
      return {
        status: ok,
        outputLength: 0,
        backendDispatchCount: 0,
        fallbackOpCount: 0,
        commandCount: 0,
        commandOpCount: 0,
        modelHandle: context.session.modelHandle,
        modelManifestHash: (context.session.modelManifest?.hash ?? 0n) ^ 1n,
        modelTableDescriptorHash: context.session.modelTable?.descriptorHash ?? 0n,
        modelTableHash: context.session.modelTable?.hash ?? 0n,
        tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
        tableHash: context.session.table?.hash ?? 0n,
      };
    },
  };
}

function createMissingHostLlamaModelPackEvidenceExecutor() {
  return {
    calls: 0,
    rejectedResource: null,
    executeTokensSync(context) {
      this.calls += 1;
      this.rejectedResource = {
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
      context.session.ownedResources.push(this.rejectedResource);
      context.session.device?.ownedResources?.push(this.rejectedResource);
      context.session.modelPacks.push({
        descriptorHash: 0x456n,
        layoutHash: 0x123n,
      });
      return {
        status: ok,
        outputLength: 0,
        backendDispatchCount: 0,
        fallbackOpCount: 0,
        commandCount: 0,
        commandOpCount: 0,
        modelHandle: context.session.modelHandle,
        modelManifestHash: context.session.modelManifest?.hash ?? 0n,
        modelTableDescriptorHash: context.session.modelTable?.descriptorHash ?? 0n,
        modelTableHash: context.session.modelTable?.hash ?? 0n,
        tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
        tableHash: context.session.table?.hash ?? 0n,
      };
    },
  };
}

function createInvalidHostLlamaModelPackHashExecutor() {
  return {
    calls: 0,
    executeTokensSync(context) {
      this.calls += 1;
      return {
        status: ok,
        outputLength: 0,
        backendDispatchCount: 0,
        fallbackOpCount: 0,
        commandCount: 0,
        commandOpCount: 0,
        modelHandle: context.session.modelHandle,
        modelManifestHash: context.session.modelManifest?.hash ?? 0n,
        modelPackLayoutHash: 1n,
        modelTableDescriptorHash: context.session.modelTable?.descriptorHash ?? 0n,
        modelTableHash: context.session.modelTable?.hash ?? 0n,
        tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
        tableHash: context.session.table?.hash ?? 0n,
      };
    },
  };
}

function expectRejectedHostLlamaExecutorProfile(session, sessionHandle, label, options = {}) {
  const outputPolicy = options.outputPolicy ?? executeOutputNone;
  const outputLen = options.outputLen ?? 0;
  const expectedNoOutputCallCount = outputPolicy === executeOutputNone ? 1 : 0;
  const expectedLogitsOutputCallCount = outputPolicy === executeOutputLogits ? 1 : 0;
  const expectedBoundOutputCallCount = expectedLogitsOutputCallCount === 1 && outputLen === 0 ? 1 : 0;
  const expectedCallerOutputCallCount = expectedLogitsOutputCallCount === 1 && outputLen !== 0 ? 1 : 0;
  const profile = session.runtimeProfile();
  if (
    profile.callCount !== 1 ||
    profile.decodedCallCount !== 1 ||
    profile.noOutputCallCount !== expectedNoOutputCallCount ||
    profile.logitsOutputCallCount !== expectedLogitsOutputCallCount ||
    profile.boundOutputCallCount !== expectedBoundOutputCallCount ||
    profile.callerOutputCallCount !== expectedCallerOutputCallCount ||
    profile.requestedOutputElementCount !== outputLen ||
    profile.resultWriteCount !== 1 ||
    profile.validationFailureCount !== 0 ||
    profile.invalidArgumentCount !== 0 ||
    profile.unsupportedCallCount !== 0 ||
    profile.executorCallCount !== 1 ||
    profile.executorOkCount !== 0 ||
    profile.executorFailureCount !== 1 ||
    profile.executorBackendOpCount !== 0 ||
    profile.executorBackendDispatchCount !== 0 ||
    profile.executorFallbackOpCount !== 0 ||
    profile.executorCommandCount !== 0 ||
    profile.executorCommandOpCount !== 0 ||
    profile.lastStatus !== invalidArgument ||
    profile.lastOutputPolicy !== outputPolicy ||
    profile.lastOutputLength !== outputLen ||
    profile.lastResultOutputLength !== 0
  ) {
    throw new Error(`${label} rejected-executor profile mismatch: ${JSON.stringify(profile)}`);
  }
  const abiProfile = sessionRuntimeProfile(sessionHandle);
  if (
    u64(abiProfile, 0) !== 0n ||
    u64(abiProfile, 8) !== 0n ||
    u64(abiProfile, 16) !== 0n ||
    u64(abiProfile, 24) !== 0n ||
    u64(abiProfile, 40) !== 1n ||
    u64(abiProfile, 48) !== 0n ||
    u64(abiProfile, 56) !== 1n ||
    u64(abiProfile, 96) !== 0n ||
    u64(abiProfile, 112) !== 0n
  ) {
    throw new Error(`${label} rejected-executor ABI runtime profile mismatch`);
  }
  resetSessionRuntimeProfile(sessionHandle);
  const resetProfile = session.runtimeProfile();
  if (resetProfile.executorCallCount !== 0 || resetProfile.executorFailureCount !== 0 || resetProfile.lastStatus !== null) {
    throw new Error(`${label} rejected-executor profile reset mismatch: ${JSON.stringify(resetProfile)}`);
  }
  const resetAbiProfile = sessionRuntimeProfile(sessionHandle);
  if (u64(resetAbiProfile, 0) !== 0n || u64(resetAbiProfile, 40) !== 0n || u64(resetAbiProfile, 56) !== 0n) {
    throw new Error(`${label} rejected-executor ABI profile reset mismatch`);
  }
}

function expectedHostLlamaResourceProfile(vocabSize) {
  return {
    callCount: 10,
    decodedCallCount: 10,
    logitsOutputCallCount: 8,
    noOutputCallCount: 2,
    callerOutputCallCount: 4,
    boundOutputCallCount: 4,
    requestedOutputElementCount: 4 * vocabSize + 2,
    resultWriteCount: 5,
    validationFailureCount: 3,
    invalidArgumentCount: 0,
    unsupportedCallCount: 7,
    lastStatus: unsupported,
    lastOutputPolicy: executeOutputLogits,
    lastOutputLength: 0,
    lastResultOutputLength: 0,
  };
}

function emptyHostLlamaResourceProfile() {
  return {
    callCount: 0,
    decodedCallCount: 0,
    logitsOutputCallCount: 0,
    noOutputCallCount: 0,
    callerOutputCallCount: 0,
    boundOutputCallCount: 0,
    requestedOutputElementCount: 0,
    resultWriteCount: 0,
    validationFailureCount: 0,
    invalidArgumentCount: 0,
    unsupportedCallCount: 0,
    lastStatus: null,
    lastOutputPolicy: null,
    lastOutputLength: 0,
    lastResultOutputLength: 0,
  };
}

function expectUnsupportedResourceSelection(session, hostSession, label) {
  const tokenZero = keep(alloc(4), 4);
  writeU32(tokenZero, 0, 0);
  const historyStart = hostSession.tokenExecutionHistory.length;

  const boundArgmax = tokenArgmax(session);
  expectStatus(boundArgmax.status, unsupported, `${label} bound argmax`);
  const boundSample = tokenSample(session);
  expectStatus(boundSample.status, unsupported, `${label} bound sample`);
  const executeArgmax = tokenExecuteArgmax(session, tokenZero, 1);
  expectStatus(executeArgmax.status, unsupported, `${label} execute argmax`);
  expectLlamaTokenExecutionRecord(hostSession.tokenExecutionHistory[historyStart], {
    tokens: [0],
    outputPolicy: executeOutputLogits,
  }, `${label} execute argmax`);
  const executeSample = tokenExecuteSample(session, tokenZero, 1);
  expectStatus(executeSample.status, unsupported, `${label} execute sample`);
  expectLlamaTokenExecutionRecord(hostSession.tokenExecutionHistory[historyStart + 1], {
    tokens: [0],
    outputPolicy: executeOutputLogits,
  }, `${label} execute sample`);

  const generatedArgmax = keep(alloc(8), 8);
  writeU32(generatedArgmax, 0, 1234);
  writeU32(generatedArgmax, 4, 5678);
  const generateArgmax = tokenGenerateArgmax(session, tokenZero, 1, generatedArgmax, 2);
  expectStatus(generateArgmax.status, unsupported, `${label} generate argmax`);
  expectLlamaTokenExecutionRecord(hostSession.tokenExecutionHistory[historyStart + 2], {
    tokens: [0],
    outputPolicy: executeOutputLogits,
  }, `${label} generate argmax`);
  if (u32(generatedArgmax, 0) !== 1234 || u32(generatedArgmax, 4) !== 5678) {
    throw new Error(`${label} unsupported generate argmax mutated output tokens`);
  }

  const generatedSample = keep(alloc(8), 8);
  writeU32(generatedSample, 0, 4321);
  writeU32(generatedSample, 4, 8765);
  const generateSample = tokenGenerateSample(session, tokenZero, 1, generatedSample, 2);
  expectStatus(generateSample.status, unsupported, `${label} generate sample`);
  expectLlamaTokenExecutionRecord(hostSession.tokenExecutionHistory[historyStart + 3], {
    tokens: [0],
    outputPolicy: executeOutputLogits,
  }, `${label} generate sample`);
  if (u32(generatedSample, 0) !== 4321 || u32(generatedSample, 4) !== 8765) {
    throw new Error(`${label} unsupported generate sample mutated output tokens`);
  }

  if (sessionPosition(session) !== 0) {
    throw new Error(`${label} unsupported selection/generation mutated position`);
  }
}

function expectLlamaTokenExecutionRecord(actual, expected, label) {
  if (!actual) throw new Error(`${label} did not record a token execution descriptor`);
  const expectedTokens = expected.tokens ?? [];
  if (
    actual.tokensLen !== expectedTokens.length ||
    actual.outputPolicy !== expected.outputPolicy ||
    actual.outputPtr !== (expected.outputPtr ?? 0) ||
    actual.outputLen !== (expected.outputLen ?? 0)
  ) {
    throw new Error(`${label} recorded wrong token execution descriptor: ${JSON.stringify(actual)}`);
  }
  for (let i = 0; i < expectedTokens.length; i += 1) {
    if (actual.tokens[i] !== expectedTokens[i]) {
      throw new Error(`${label} token[${i}] expected ${expectedTokens[i]}, got ${actual.tokens[i]}`);
    }
  }
}

function expectLastLlamaTokenExecution(session, expected, label) {
  expectLlamaTokenExecutionRecord(session.lastTokenExecution, expected, label);
}

function inspectSession(session) {
  return inspectSessionWith(exportsRef, session);
}

function inspectSessionWith(exportsLike, session) {
  const out = keep(alloc(sessionInspectionSize), sessionInspectionSize);
  zero(out, sessionInspectionSize);
  check(exportsLike.zgml_session_inspect(session, out));
  return out;
}

function resetSessionRuntimeProfile(session) {
  check(exportsRef.zgml_session_reset_runtime_profile(session));
}

function commandCategoryTotal(inspection) {
  return u64(inspection, 104) +
    u64(inspection, 112) +
    u64(inspection, 120) +
    u64(inspection, 128) +
    u64(inspection, 136) +
    u64(inspection, 144) +
    u64(inspection, 152);
}

function expectNoDispatchPlan(inspection, label) {
  if (
    u64(inspection, 192) !== 0n ||
    u64(inspection, 200) !== 0n ||
    u64(inspection, 208) !== 0n ||
    u64(inspection, 216) !== 0xffffffffffffffffn
  ) {
    throw new Error(`expected empty dispatch plan for ${label}`);
  }
}

function expectNoDispatchPlanCapabilities(capabilities, label) {
  if (
    capabilities.backendDispatchCount !== 0 ||
    capabilities.dispatchPlanSupported ||
    capabilities.dispatchPlanCoveredOpCount !== 0 ||
    capabilities.dispatchPlanFirstUnsupportedOp !== null ||
    capabilities.dispatchPlanProjectionCount !== 0 ||
    capabilities.dispatchPlanRowCount !== 0 ||
    capabilities.dispatchPlanAttentionCount !== 0 ||
    capabilities.dispatchPlanMovementCount !== 0 ||
    capabilities.dispatchPlanElementwiseCount !== 0 ||
    capabilities.dispatchPlanRopeCount !== 0 ||
    capabilities.dispatchPlanQuantizedProjectionCount !== 0
  ) {
    throw new Error(`expected empty dispatch-plan capabilities for ${label}`);
  }
}

function bindSession(program, desc = 0) {
  const out = keep(alloc(handleSize), handleSize);
  check(exportsRef.zgml_session_bind(program, desc, out));
  const handle = u32(out);
  if (handle === 0) throw new Error("session handle was not returned");
  return handle;
}

function bindBufferSession(program, desc) {
  const out = keep(alloc(handleSize), handleSize);
  check(exportsRef.zgml_session_bind_buffers(program, desc, out));
  const handle = u32(out);
  if (handle === 0) throw new Error("buffer-backed session handle was not returned");
  return handle;
}

function bindModelSession(program, model, desc = 0) {
  const out = keep(alloc(handleSize), handleSize);
  check(exportsRef.zgml_session_bind_model(program, model, desc, out));
  const handle = u32(out);
  if (handle === 0) throw new Error("model-backed session handle was not returned");
  return handle;
}

function bindModelBufferSession(program, model, desc) {
  const out = keep(alloc(handleSize), handleSize);
  check(exportsRef.zgml_session_bind_model_buffers(program, model, desc, out));
  const handle = u32(out);
  if (handle === 0) throw new Error("model-and-buffer-backed session handle was not returned");
  return handle;
}

function bindLlamaBufferSession(program, desc = 0) {
  const out = keep(alloc(handleSize), handleSize);
  check(exportsRef.zgml_llama_session_bind_buffers(program, desc, out));
  const handle = u32(out);
  if (handle === 0) throw new Error("llama buffer-backed session handle was not returned");
  return handle;
}

function bindLlamaModelBufferSession(program, model, desc = 0) {
  const out = keep(alloc(handleSize), handleSize);
  check(exportsRef.zgml_llama_session_bind_model_buffers(program, model, desc, out));
  const handle = u32(out);
  if (handle === 0) throw new Error("llama model-and-buffer-backed session handle was not returned");
  return handle;
}

function sessionPosition(session) {
  return sessionPositionWith(exportsRef, session);
}

function sessionPositionWith(exportsLike, session) {
  const out = keep(alloc(4), 4);
  check(exportsLike.zgml_session_position(session, out));
  return u32(out);
}

function sessionReset(session) {
  check(exportsRef.zgml_session_reset(session));
}

function tokenStep(session, token, outputPtr, outputLen) {
  const desc = keep(alloc(tokenStepDescSize), tokenStepDescSize);
  writeU32(desc, 0, token);
  writeU32(desc, 4, outputPtr);
  writeU32(desc, 8, outputLen);
  const result = keep(alloc(stepResultSize), stepResultSize);
  const status = exportsRef.zgml_session_step_token(session, desc, result);
  return { status, outputLen: u32(result) };
}

function tokenAdvance(session, token) {
  const desc = keep(alloc(tokenAdvanceDescSize), tokenAdvanceDescSize);
  writeU32(desc, 0, token);
  return exportsRef.zgml_session_advance_token(session, desc);
}

function tokenAdvanceTokens(session, tokensPtr, tokensLen) {
  const desc = keep(alloc(tokenAdvanceTokensDescSize), tokenAdvanceTokensDescSize);
  writeU32(desc, 0, tokensPtr);
  writeU32(desc, 4, tokensLen);
  return exportsRef.zgml_session_advance_tokens(session, desc);
}

function tokenPrefill(session, tokensPtr, tokensLen, outputPtr, outputLen) {
  const desc = keep(alloc(tokenPrefillDescSize), tokenPrefillDescSize);
  writeU32(desc, 0, tokensPtr);
  writeU32(desc, 4, tokensLen);
  writeU32(desc, 8, outputPtr);
  writeU32(desc, 12, outputLen);
  const result = keep(alloc(stepResultSize), stepResultSize);
  const status = exportsRef.zgml_session_prefill_tokens(session, desc, result);
  return { status, outputLen: u32(result) };
}

function tokenExecute(session, tokensPtr, tokensLen, outputPolicy, outputPtr, outputLen) {
  return tokenExecuteWith(exportsRef, session, tokensPtr, tokensLen, outputPolicy, outputPtr, outputLen);
}

function tokenExecuteWith(exportsLike, session, tokensPtr, tokensLen, outputPolicy, outputPtr, outputLen) {
  const desc = keep(alloc(tokenExecuteDescSize), tokenExecuteDescSize);
  writeU32(desc, 0, tokensPtr);
  writeU32(desc, 4, tokensLen);
  writeU32(desc, 8, outputPolicy);
  writeU32(desc, 12, 0);
  writeU32(desc, 16, outputPtr);
  writeU32(desc, 20, outputLen);
  const result = keep(alloc(stepResultSize), stepResultSize);
  writeU32(result, 0, 99);
  const status = exportsLike.zgml_session_execute_tokens(session, desc, result);
  return { status, outputLen: u32(result) };
}

function tokenArgmax(session, logitsPtr = 0, logitsLen = 0) {
  let desc = 0;
  if (logitsPtr !== 0 || logitsLen !== 0) {
    desc = keep(alloc(tokenArgmaxDescSize), tokenArgmaxDescSize);
    writeU32(desc, 0, logitsPtr);
    writeU32(desc, 4, logitsLen);
    writeU32(desc, 8, 0);
  }
  const result = keep(alloc(tokenArgmaxResultSize), tokenArgmaxResultSize);
  const status = exportsRef.zgml_session_argmax_token(session, desc, result);
  return {
    status,
    token: u32(result),
    logit: view().getFloat32(result + 4, true),
  };
}

function tokenExecuteArgmax(session, tokensPtr, tokensLen) {
  const desc = keep(alloc(tokenExecuteArgmaxDescSize), tokenExecuteArgmaxDescSize);
  writeU32(desc, 0, tokensPtr);
  writeU32(desc, 4, tokensLen);
  writeU32(desc, 8, 0);
  const result = keep(alloc(tokenArgmaxResultSize), tokenArgmaxResultSize);
  const status = exportsRef.zgml_session_execute_argmax_tokens(session, desc, result);
  return {
    status,
    token: u32(result),
    logit: view().getFloat32(result + 4, true),
  };
}

function tokenGenerateArgmax(session, tokensPtr, tokensLen, outputTokensPtr, outputTokensLen) {
  const desc = keep(alloc(tokenGenerateArgmaxDescSize), tokenGenerateArgmaxDescSize);
  writeU32(desc, 0, tokensPtr);
  writeU32(desc, 4, tokensLen);
  writeU32(desc, 8, outputTokensPtr);
  writeU32(desc, 12, outputTokensLen);
  writeU32(desc, 16, 0);
  const result = keep(alloc(tokenGenerateArgmaxResultSize), tokenGenerateArgmaxResultSize);
  const status = exportsRef.zgml_session_generate_argmax_tokens(session, desc, result);
  return {
    status,
    tokensGenerated: u32(result),
    lastToken: u32(result, 4),
    lastLogit: view().getFloat32(result + 8, true),
  };
}

function tokenSample(session, logitsPtr = 0, logitsLen = 0, topK = 1, seed = 123, temperature = 1.0) {
  const desc = keep(alloc(tokenSampleDescSize), tokenSampleDescSize);
  writeU32(desc, 0, logitsPtr);
  writeU32(desc, 4, logitsLen);
  writeU32(desc, 8, topK);
  writeU32(desc, 12, seed);
  view().setFloat32(desc + 16, temperature, true);
  writeU32(desc, 20, 0);
  const result = keep(alloc(tokenSampleResultSize), tokenSampleResultSize);
  const status = exportsRef.zgml_session_sample_token(session, desc, result);
  return {
    status,
    token: u32(result),
    logit: view().getFloat32(result + 4, true),
  };
}

function tokenExecuteSample(session, tokensPtr, tokensLen, topK = 1, seed = 123, temperature = 1.0) {
  const desc = keep(alloc(tokenExecuteSampleDescSize), tokenExecuteSampleDescSize);
  writeU32(desc, 0, tokensPtr);
  writeU32(desc, 4, tokensLen);
  writeU32(desc, 8, topK);
  writeU32(desc, 12, seed);
  view().setFloat32(desc + 16, temperature, true);
  writeU32(desc, 20, 0);
  const result = keep(alloc(tokenSampleResultSize), tokenSampleResultSize);
  const status = exportsRef.zgml_session_execute_sample_tokens(session, desc, result);
  return {
    status,
    token: u32(result),
    logit: view().getFloat32(result + 4, true),
  };
}

function tokenGenerateSample(session, tokensPtr, tokensLen, outputTokensPtr, outputTokensLen, topK = 1, seed = 123, temperature = 1.0) {
  const desc = keep(alloc(tokenGenerateSampleDescSize), tokenGenerateSampleDescSize);
  writeU32(desc, 0, tokensPtr);
  writeU32(desc, 4, tokensLen);
  writeU32(desc, 8, outputTokensPtr);
  writeU32(desc, 12, outputTokensLen);
  writeU32(desc, 16, topK);
  writeU32(desc, 20, seed);
  view().setFloat32(desc + 24, temperature, true);
  writeU32(desc, 28, 0);
  const result = keep(alloc(tokenGenerateSampleResultSize), tokenGenerateSampleResultSize);
  const status = exportsRef.zgml_session_generate_sample_tokens(session, desc, result);
  return {
    status,
    tokensGenerated: u32(result),
    lastToken: u32(result, 4),
    lastLogit: view().getFloat32(result + 8, true),
  };
}

function expectArgmax(result, logitsPtr, logitsLen) {
  check(result.status);
  let token = 0;
  let logit = readF32Array(logitsPtr, 1)[0];
  for (let i = 1; i < logitsLen; i += 1) {
    const value = readF32Array(logitsPtr + i * 4, 1)[0];
    if (value > logit) {
      token = i;
      logit = value;
    }
  }
  if (result.token !== token || result.logit !== logit) {
    throw new Error(`unexpected argmax: got token=${result.token} logit=${result.logit}, expected token=${token} logit=${logit}`);
  }
}

function runTinyLinearSmoke() {
  let model = 0;
  let program = 0;
  let session = 0;
  let boundSession = 0;
  let wrappedSession = 0;
  try {
    model = createModel(tinyLinearKind, 2, 3);
    const modelInspection = inspectModel(model);
    if (
      u32(modelInspection, 0) !== tinyLinearKind ||
      u64(modelInspection, 8) !== 2n ||
      u64(modelInspection, 16) !== 3n ||
      u64(modelInspection, 24) !== 0n
    ) {
      throw new Error("unexpected tiny linear model inspection");
    }

    program = compileProgram(model, 4);
    const compatibility = programModelCompatibility(program, model);
    if (
      u32(compatibility, 0) !== tinyLinearKind ||
      u32(compatibility, 4) !== tinyLinearKind ||
      u64(compatibility, 8) !== 0n
    ) {
      throw new Error("unexpected tiny linear model compatibility");
    }
    const inspection = inspectProgram(program);
    if (u64(inspection, 0) !== BigInt(backendCpu) || u64(inspection, 8) !== 1n || u64(inspection, 16) !== 0n || u64(inspection, 24) !== 5n || u64(inspection, 32) !== 68n || u64(inspection, 40) !== 1n || u64(inspection, 48) !== 0n || u64(inspection, 56) === 0n || u64(inspection, 64) === 0n || u64(inspection, 160) === 0n || u64(inspection, 168) !== 17n || u64(inspection, 176) !== 0xffffffffn || u64(inspection, 184) !== 0xffffffffn) {
      throw new Error("expected compiled program shape evidence");
    }
    if (commandCategoryTotal(inspection) === 0n) {
      throw new Error("expected tiny linear command category evidence");
    }
    expectNoDispatchPlan(inspection, "CPU tiny linear");
    const capabilities = programCapabilities(program);
    if (
      capabilities.backend !== backendCpu ||
      capabilities.mode !== "executable" ||
      !capabilities.canExecute ||
      capabilities.canBindExternalResources ||
      capabilities.hasFullDispatchPlan
    ) {
      throw new Error("unexpected tiny linear capabilities");
    }
    expectNoDispatchPlanCapabilities(capabilities, "CPU tiny linear");
    expectPortableDeviceBufferApisUnsupported(program, "portable CPU tiny linear");
    const requirements = programRequirements(program);
    if (
      u32(requirements, 0) !== tinyLinearKind ||
      u32(requirements, 4) !== 4 ||
      u32(requirements, 8) !== 4 ||
      requirementInputLen(requirements) !== 2 ||
      requirementOutputLen(requirements) !== 3 ||
      requirementWeightsLen(requirements) !== 6 ||
      requirementBiasLen(requirements) !== 3 ||
      requirementOutputByteLen(requirements) !== 12
    ) {
      throw new Error("unexpected tiny linear requirements");
    }
    const factoryOutputBuffer = createProgramBuffer(program, programBufferOutput);
    if (bufferSize(factoryOutputBuffer) !== requirementOutputByteLen(requirements)) {
      throw new Error("unexpected tiny linear factory output size");
    }
    const factoryOutputInspection = inspectBuffer(factoryOutputBuffer);
    if (
      u32(factoryOutputInspection, 0) !== bufferStorageHost ||
      u64(factoryOutputInspection, 16) !== BigInt(requirementOutputByteLen(requirements)) ||
      u32(factoryOutputInspection, 4) !== 0 ||
      u32(factoryOutputInspection, 8) !== 0 ||
      u64(factoryOutputInspection, 24) !== 0n ||
      u64(factoryOutputInspection, 40) !== 0n
    ) {
      throw new Error("unexpected tiny linear host buffer inspection");
    }

    const weights = keep(alloc(6 * 4), 6 * 4);
    writeF32Array(weights, [1, 2, 3, 4, 5, 6]);
    const bias = keep(alloc(3 * 4), 3 * 4);
    writeF32Array(bias, [0.5, -0.5, 1.0]);

    const bindDesc = keep(alloc(bindDescSize), bindDescSize);
    zero(bindDesc, bindDescSize);
    writeU32(bindDesc, 0, weights);
    writeU32(bindDesc, 4, 6);
    writeU32(bindDesc, 8, bias);
    writeU32(bindDesc, 12, 3);
    session = bindSession(program, bindDesc);

    const input = keep(alloc(2 * 4), 2 * 4);
    writeF32Array(input, [1, 2]);
    const output = keep(alloc(3 * 4), 3 * 4);
    const stepDesc = keep(alloc(stepDescSize), stepDescSize);
    writeU32(stepDesc, 0, input);
    writeU32(stepDesc, 4, 2);
    writeU32(stepDesc, 8, output);
    writeU32(stepDesc, 12, 3);

    const result = keep(alloc(stepResultSize), stepResultSize);
    check(exportsRef.zgml_session_step(session, stepDesc, result));
    if (u32(result) !== 3) throw new Error(`expected 3 outputs, got ${u32(result)}`);
    const values = readF32Array(output, 3);
    expectClose(values, [9.5, 11.5, 16.0]);

    const programProfile = programRuntimeProfile(program);
    if (u64(programProfile, 0) !== 0n || u64(programProfile, 40) !== 0n || u64(programProfile, 96) === 0n) {
      throw new Error("expected cold tiny linear Program runtime profile with command evidence");
    }
    const sessionProfile = sessionRuntimeProfile(session);
    if (u64(sessionProfile, 0) !== 1n || u64(sessionProfile, 40) !== 1n || u64(sessionProfile, 96) === 0n) {
      throw new Error("expected hot tiny linear Session runtime profile with command evidence");
    }
    resetSessionRuntimeProfile(session);
    const resetProfile = sessionRuntimeProfile(session);
    if (u64(resetProfile, 0) !== 0n || u64(resetProfile, 40) !== 0n || u64(resetProfile, 96) === 0n) {
      throw new Error("expected reset tiny linear Session runtime profile to preserve command evidence");
    }

    const boundWeightsBuffer = createProgramBuffer(program, programBufferWeights);
    const boundInputBuffer = createProgramBuffer(program, programBufferInput);
    const boundOutputBuffer = createBuffer(4 * 4);
    if (bufferSize(boundWeightsBuffer) !== 6 * 4 || bufferSize(boundInputBuffer) !== 2 * 4 || bufferSize(boundOutputBuffer) !== 4 * 4) {
      throw new Error("unexpected Program-sized native buffer size");
    }

    const boundWeights = keep(alloc(6 * 4), 6 * 4);
    writeF32Array(boundWeights, [1, 0, 0, 1, 1, 1]);
    bufferWrite(boundWeightsBuffer, 0, boundWeights, 6 * 4);
    const boundInput = keep(alloc(2 * 4), 2 * 4);
    writeF32Array(boundInput, [2, 3]);
    bufferWrite(boundInputBuffer, 0, boundInput, 2 * 4);
    const boundOutput = keep(alloc(4 * 4), 4 * 4);
    writeF32Array(boundOutput, [-999, -999, -999, -999]);
    bufferWrite(boundOutputBuffer, 0, boundOutput, 4 * 4);
    const boundBindDesc = keep(alloc(bindDescSize), bindDescSize);
    zero(boundBindDesc, bindDescSize);
    writeU32(boundBindDesc, 0, boundWeightsBuffer);
    writeU32(boundBindDesc, 4, 6);
    writeU32(boundBindDesc, 16, boundInputBuffer);
    writeU32(boundBindDesc, 20, 2);
    writeU32(boundBindDesc, 24, boundOutputBuffer);
    writeU32(boundBindDesc, 28, 3);
    boundSession = bindBufferSession(program, boundBindDesc);
    const boundSessionInspection = inspectSession(boundSession);
    if (
      u32(boundSessionInspection, 0) !== tinyLinearKind ||
      u32(boundSessionInspection, 4) !== backendCpu ||
      u32(boundSessionInspection, 8) !== bufferStorageHost ||
      u32(boundSessionInspection, 12) !== 0 ||
      u64(boundSessionInspection, 16) !== 0n ||
      u64(boundSessionInspection, 24) !== 0n ||
      u64(boundSessionInspection, 32) !== 2n ||
      u64(boundSessionInspection, 40) !== 1n ||
      u64(boundSessionInspection, 48) !== 1n ||
      u64(boundSessionInspection, 56) !== 4n ||
      u64(boundSessionInspection, 64) !== 0n ||
      u64(boundSessionInspection, 72) === 0n
    ) {
      throw new Error("unexpected tiny linear session inspection");
    }

    check(exportsRef.zgml_session_step(boundSession, 0, result));
    if (u32(result) !== 3) throw new Error(`expected 3 bound outputs, got ${u32(result)}`);
    bufferRead(boundOutputBuffer, 0, boundOutput, 4 * 4);
    expectClose(readF32Array(boundOutput, 3), [5, 3, 3]);
    if (readF32Array(boundOutput, 4)[3] !== -999) throw new Error("bound tiny linear wrote past output");

    writeF32Array(boundInput, [4, 5]);
    bufferWrite(boundInputBuffer, 0, boundInput, 2 * 4);
    writeF32Array(boundOutput, [-111, -111, -111, -111]);
    bufferWrite(boundOutputBuffer, 0, boundOutput, 4 * 4);
    check(exportsRef.zgml_session_step(boundSession, 0, result));
    bufferRead(boundOutputBuffer, 0, boundOutput, 4 * 4);
    expectClose(readF32Array(boundOutput, 3), [9, 5, 5]);
    if (readF32Array(boundOutput, 4)[3] !== -111) throw new Error("second bound tiny linear wrote past output");

    const wrappedWeights = keep(alloc(6 * 4), 6 * 4);
    writeF32Array(wrappedWeights, [1, 0, 0, 1, 1, 1]);
    const wrappedInput = keep(alloc(2 * 4), 2 * 4);
    writeF32Array(wrappedInput, [2, 3]);
    const wrappedOutput = keep(alloc(4 * 4), 4 * 4);
    writeF32Array(wrappedOutput, [-999, -999, -999, -777]);
    const wrappedWeightsBuffer = wrapBuffer(wrappedWeights, 6 * 4);
    const wrappedInputBuffer = wrapBuffer(wrappedInput, 2 * 4);
    const wrappedOutputBuffer = wrapBuffer(wrappedOutput, 4 * 4);
    const wrappedBindDesc = keep(alloc(bindDescSize), bindDescSize);
    zero(wrappedBindDesc, bindDescSize);
    writeU32(wrappedBindDesc, 0, wrappedWeightsBuffer);
    writeU32(wrappedBindDesc, 4, 6);
    writeU32(wrappedBindDesc, 16, wrappedInputBuffer);
    writeU32(wrappedBindDesc, 20, 2);
    writeU32(wrappedBindDesc, 24, wrappedOutputBuffer);
    writeU32(wrappedBindDesc, 28, 3);
    wrappedSession = bindBufferSession(program, wrappedBindDesc);

    check(exportsRef.zgml_session_step(wrappedSession, 0, result));
    expectClose(readF32Array(wrappedOutput, 3), [5, 3, 3]);
    if (readF32Array(wrappedOutput, 4)[3] !== -777) throw new Error("wrapped tiny linear wrote past output");
    writeF32Array(wrappedInput, [4, 5]);
    writeF32Array(wrappedOutput, [-111, -111, -111, -222]);
    check(exportsRef.zgml_session_step(wrappedSession, 0, result));
    expectClose(readF32Array(wrappedOutput, 3), [9, 5, 5]);
    if (readF32Array(wrappedOutput, 4)[3] !== -222) throw new Error("second wrapped tiny linear wrote past output");

    const resourceWeightsBuffer = wrapResourceBuffer(backendWebGpu, 1, 16, 6 * 4);
    const resourceOutputBuffer = wrapResourceBuffer(backendWebGpu, 2, 0, 3 * 4);
    if (bufferSize(resourceWeightsBuffer) !== 6 * 4) throw new Error("unexpected resource buffer size");
    const resourceWeightsInspection = inspectBuffer(resourceWeightsBuffer);
    if (
      u32(resourceWeightsInspection, 0) !== bufferStorageExternalResource ||
      u32(resourceWeightsInspection, 4) !== backendWebGpu ||
      u32(resourceWeightsInspection, 8) !== resourceAccessReadWrite ||
      u64(resourceWeightsInspection, 16) !== 24n ||
      u64(resourceWeightsInspection, 24) !== 1n ||
      u64(resourceWeightsInspection, 32) !== 16n ||
      u64(resourceWeightsInspection, 40) !== 40n
    ) {
      throw new Error("unexpected tiny linear resource buffer inspection");
    }
    const resourceOutputInspection = inspectBuffer(resourceOutputBuffer);
    if (
      u32(resourceOutputInspection, 0) !== bufferStorageExternalResource ||
      u32(resourceOutputInspection, 4) !== backendWebGpu ||
      u64(resourceOutputInspection, 16) !== 12n ||
      u64(resourceOutputInspection, 24) !== 2n ||
      u64(resourceOutputInspection, 32) !== 0n ||
      u64(resourceOutputInspection, 40) !== 12n
    ) {
      throw new Error("unexpected tiny linear resource output inspection");
    }
    expectStatus(exportsRef.zgml_buffer_write(resourceWeightsBuffer, 0, wrappedWeights, 4), unsupported, "resource buffer write");
    expectStatus(exportsRef.zgml_buffer_read(resourceWeightsBuffer, 0, wrappedWeights, 4), unsupported, "resource buffer read");
    const resourceBindDesc = keep(alloc(bindDescSize), bindDescSize);
    zero(resourceBindDesc, bindDescSize);
    writeU32(resourceBindDesc, 0, resourceWeightsBuffer);
    writeU32(resourceBindDesc, 4, 6);
    writeU32(resourceBindDesc, 16, wrappedInputBuffer);
    writeU32(resourceBindDesc, 20, 2);
    writeU32(resourceBindDesc, 24, resourceOutputBuffer);
    writeU32(resourceBindDesc, 28, 3);
    const resourceOut = keep(alloc(handleSize), handleSize);
    writeU32(resourceOut, 0, 0xffffffff);
    expectStatus(exportsRef.zgml_session_bind_buffers(program, resourceBindDesc, resourceOut), unsupported, "host-only resource binding");
    if (u32(resourceOut) !== 0) throw new Error("resource bind returned a session handle");

    append(`zgml browser wasm ffi smoke ok: [${values.join(", ")}]`);
  } finally {
    if (wrappedSession !== 0) exportsRef.zgml_session_free(wrappedSession);
    if (boundSession !== 0) exportsRef.zgml_session_free(boundSession);
    if (session !== 0) exportsRef.zgml_session_free(session);
    if (program !== 0) exportsRef.zgml_program_free(program);
    if (model !== 0) exportsRef.zgml_model_free(model);
  }
}

function runTinyLlamaSmoke() {
  let model = 0;
  let compatibleModel = 0;
  let program = 0;
  const sessions = [];
  try {
    model = createModel(tinyLlamaKind, 0, 0);
    const modelInspection = inspectModel(model);
    if (
      u32(modelInspection, 0) !== tinyLlamaKind ||
      u64(modelInspection, 24) !== 8n ||
      u64(modelInspection, 32) !== 8n ||
      u64(modelInspection, 40) !== 4n ||
      u64(modelInspection, 48) !== 1n ||
      u64(modelInspection, 56) !== 1n ||
      u64(modelInspection, 64) !== 1n ||
      u64(modelInspection, 72) !== 8n ||
      Math.abs(f64(modelInspection, 80) - 10000) > 1e-5 ||
      Math.abs(f64(modelInspection, 88) - 1e-6) > 1e-12 ||
      u64(modelInspection, 96) !== 0n
    ) {
      throw new Error("unexpected tiny LLaMA model inspection");
    }

    program = compileProgram(model, 4);
    const compatibility = programModelCompatibility(program, model);
    if (
      u32(compatibility, 0) !== tinyLlamaKind ||
      u32(compatibility, 4) !== tinyLlamaKind ||
      u64(compatibility, 8) !== 1n
    ) {
      throw new Error("unexpected tiny LLaMA model compatibility");
    }

    const inspection = inspectProgram(program);
    if (u64(inspection, 0) !== BigInt(backendCpu) || u64(inspection, 8) !== 1n || u64(inspection, 16) !== 0n || u64(inspection, 24) === 0n || u64(inspection, 32) === 0n || u64(inspection, 56) === 0n || u64(inspection, 64) === 0n || u64(inspection, 72) === 0n || u64(inspection, 96) === 0n || u64(inspection, 160) === 0n || u64(inspection, 168) === 0n || u64(inspection, 176) !== 3n || u64(inspection, 184) !== 4n) {
      throw new Error("expected tiny LLaMA executable shape evidence");
    }
    if (u64(inspection, 120) === 0n && u64(inspection, 128) === 0n) {
      throw new Error("expected tiny LLaMA command category evidence");
    }
    expectNoDispatchPlan(inspection, "CPU tiny LLaMA");
    const capabilities = programCapabilities(program);
    if (
      capabilities.backend !== backendCpu ||
      capabilities.mode !== "executable" ||
      !capabilities.canExecute ||
      capabilities.canBindExternalResources ||
      capabilities.hasFullDispatchPlan
    ) {
      throw new Error("unexpected tiny LLaMA capabilities");
    }
    expectNoDispatchPlanCapabilities(capabilities, "CPU tiny LLaMA");

    const llamaInspection = keep(alloc(llamaProgramInspectionSize), llamaProgramInspectionSize);
    check(exportsRef.zgml_llama_program_inspect(program, llamaInspection));
    if (u64(llamaInspection, 0) !== 8n || u64(llamaInspection, 16) !== 4n || u64(llamaInspection, 24) !== 1n || u64(llamaInspection, 96) === 0n) {
      throw new Error("unexpected tiny LLaMA inspection");
    }
    if (u64(llamaInspection, 96) !== u64(inspection, 72) || u64(llamaInspection, 104) !== u64(inspection, 80) || u64(llamaInspection, 112) !== u64(inspection, 88)) {
      throw new Error("tiny LLaMA semantic and executable patch evidence diverged");
    }
    const requirements = programRequirements(program);
    if (
      u32(requirements, 0) !== tinyLlamaKind ||
      u32(requirements, 4) !== 4 ||
      u32(requirements, 8) !== 4 ||
      requirementOutputLen(requirements) !== tinyLlamaVocabSize ||
      requirementLogitsLen(requirements) !== tinyLlamaVocabSize ||
      requirementOutputByteLen(requirements) !== tinyLlamaVocabSize * 4 ||
      requirementContextLen(requirements) !== 4 ||
      requirementBatch(requirements) !== 1 ||
      requirementMaxTokenWindow(requirements) !== 4
    ) {
      throw new Error("unexpected tiny LLaMA requirements");
    }
    const kvRequirements = keep(alloc(llamaKvCacheRequirementsSize), llamaKvCacheRequirementsSize);
    zero(kvRequirements, llamaKvCacheRequirementsSize);
    check(exportsRef.zgml_llama_program_get_kv_cache_requirements(program, kvRequirements));
    if (
      u32(kvRequirements, 0) !== tinyLlamaKind ||
      u32(kvRequirements, 4) !== 4 ||
      u32(kvRequirements, 8) !== 1 ||
      u32(kvRequirements, 16) !== 4 ||
      u32(kvRequirements, 20) !== 64 ||
      u32(kvRequirements, 24) !== 64 ||
      u32(kvRequirements, 28) !== 64
    ) {
      throw new Error("unexpected tiny LLaMA KV cache requirements");
    }
    const factoryOutputBuffer = createProgramBuffer(program, programBufferOutput);
    if (bufferSize(factoryOutputBuffer) !== requirementOutputByteLen(requirements)) {
      throw new Error("unexpected tiny LLaMA factory output size");
    }

    const a = bindSession(program);
    const b = bindSession(program);
    const prefilled = bindSession(program);
    const advanced = bindSession(program);
    const bulkAdvanced = bindSession(program);
    const bulkExpected = bindSession(program);
    const executed = bindSession(program);
    const sessionInspection = inspectSession(a);
    if (
      u32(sessionInspection, 0) !== tinyLlamaKind ||
      u32(sessionInspection, 4) !== backendCpu ||
      u32(sessionInspection, 8) !== bufferStorageHost ||
      u32(sessionInspection, 12) !== bufferStorageHost ||
      u64(sessionInspection, 16) !== 0n ||
      u64(sessionInspection, 24) !== 4n ||
      u64(sessionInspection, 32) === 0n ||
      u64(sessionInspection, 40) === 0n ||
      u64(sessionInspection, 48) === 0n ||
      u64(sessionInspection, 56) === 0n ||
      u64(sessionInspection, 64) !== 0n ||
      u64(sessionInspection, 72) === 0n
    ) {
      throw new Error("unexpected tiny LLaMA session inspection");
    }
    const boundOutputBuffer = createBuffer((tinyLlamaVocabSize + 1) * 4);
    const boundOutput = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(boundOutput, Array(tinyLlamaVocabSize + 1).fill(-666));
    bufferWrite(boundOutputBuffer, 0, boundOutput, (tinyLlamaVocabSize + 1) * 4);
    const boundBindDesc = keep(alloc(bindDescSize), bindDescSize);
    zero(boundBindDesc, bindDescSize);
    writeU32(boundBindDesc, 24, boundOutputBuffer);
    writeU32(boundBindDesc, 28, tinyLlamaVocabSize);
    const bound = bindBufferSession(program, boundBindDesc);
    compatibleModel = createModel(tinyLlamaKind, 0, 0);
    const resourceOutputBuffer = wrapResourceBuffer(backendWebGpu, 33, 0, tinyLlamaVocabSize * 4);
    const resourceBindDesc = keep(alloc(bindDescSize), bindDescSize);
    zero(resourceBindDesc, bindDescSize);
    writeU32(resourceBindDesc, 24, resourceOutputBuffer);
    writeU32(resourceBindDesc, 28, tinyLlamaVocabSize);
    const resourceOut = keep(alloc(handleSize), handleSize);
    writeU32(resourceOut, 0, 0xffffffff);
    expectStatus(exportsRef.zgml_session_bind_buffers(program, resourceBindDesc, resourceOut), unsupported, "host-only tiny LLaMA resource output binding");
    if (u32(resourceOut) !== 0) throw new Error("resource-output tiny LLaMA bind returned a session handle");
    writeU32(resourceOut, 0, 0xffffffff);
    expectStatus(exportsRef.zgml_session_bind_model_buffers(program, compatibleModel, resourceBindDesc, resourceOut), unsupported, "host-only model-bound tiny LLaMA resource output binding");
    if (u32(resourceOut) !== 0) throw new Error("model-bound resource-output tiny LLaMA bind returned a session handle");

    const cacheByteLen = u32(kvRequirements, 20);
    const kCacheHost = createBuffer(cacheByteLen);
    const vCacheHost = createBuffer(cacheByteLen);
    const kCacheHostHandles = keep(alloc(handleSize), handleSize);
    const vCacheHostHandles = keep(alloc(handleSize), handleSize);
    writeU32(kCacheHostHandles, 0, kCacheHost);
    writeU32(vCacheHostHandles, 0, vCacheHost);
    const hostKvCacheDesc = keep(alloc(llamaKvCacheBindDescSize), llamaKvCacheBindDescSize);
    zero(hostKvCacheDesc, llamaKvCacheBindDescSize);
    writeU32(hostKvCacheDesc, 0, kCacheHostHandles);
    writeU32(hostKvCacheDesc, 4, vCacheHostHandles);
    writeU32(hostKvCacheDesc, 8, 1);
    const llamaHostKvBindDesc = keep(alloc(llamaBufferBindDescSize), llamaBufferBindDescSize);
    zero(llamaHostKvBindDesc, llamaBufferBindDescSize);
    writeU32(llamaHostKvBindDesc, 8, hostKvCacheDesc);
    const hostKv = bindLlamaBufferSession(program, llamaHostKvBindDesc);
    const hostKvLogits = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(hostKvLogits, Array(tinyLlamaVocabSize + 1).fill(-4545));
    const hostKvStep = tokenStep(hostKv, 1, hostKvLogits, tinyLlamaVocabSize + 1);
    check(hostKvStep.status);
    if (hostKvStep.outputLen !== tinyLlamaVocabSize || readF32Array(hostKvLogits + tinyLlamaVocabSize * 4, 1)[0] !== -4545) {
      throw new Error("unexpected host-KV tiny LLaMA step");
    }

    const kCacheResource = wrapResourceBuffer(backendWebGpu, 44, 0, cacheByteLen);
    const vCacheResource = wrapResourceBuffer(backendWebGpu, 45, 0, cacheByteLen);
    const kCacheHandles = keep(alloc(handleSize), handleSize);
    const vCacheHandles = keep(alloc(handleSize), handleSize);
    writeU32(kCacheHandles, 0, kCacheResource);
    writeU32(vCacheHandles, 0, vCacheResource);
    const kvCacheDesc = keep(alloc(llamaKvCacheBindDescSize), llamaKvCacheBindDescSize);
    zero(kvCacheDesc, llamaKvCacheBindDescSize);
    writeU32(kvCacheDesc, 0, kCacheHandles);
    writeU32(kvCacheDesc, 4, vCacheHandles);
    writeU32(kvCacheDesc, 8, 1);
    const llamaResourceBindDesc = keep(alloc(llamaBufferBindDescSize), llamaBufferBindDescSize);
    zero(llamaResourceBindDesc, llamaBufferBindDescSize);
    writeU32(llamaResourceBindDesc, 8, kvCacheDesc);
    writeU32(resourceOut, 0, 0xffffffff);
    expectStatus(exportsRef.zgml_llama_session_bind_buffers(program, llamaResourceBindDesc, resourceOut), unsupported, "host-only tiny LLaMA KV resource binding");
    if (u32(resourceOut) !== 0) throw new Error("KV-resource tiny LLaMA bind returned a session handle");
    writeU32(resourceOut, 0, 0xffffffff);
    expectStatus(exportsRef.zgml_llama_session_bind_model_buffers(program, compatibleModel, llamaResourceBindDesc, resourceOut), unsupported, "host-only model-bound tiny LLaMA KV resource binding");
    if (u32(resourceOut) !== 0) throw new Error("model-bound KV-resource tiny LLaMA bind returned a session handle");

    const modelBound = bindModelSession(program, compatibleModel);
    const modelBoundOutputBuffer = createBuffer((tinyLlamaVocabSize + 1) * 4);
    const modelBoundOutput = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(modelBoundOutput, Array(tinyLlamaVocabSize + 1).fill(-1212));
    bufferWrite(modelBoundOutputBuffer, 0, modelBoundOutput, (tinyLlamaVocabSize + 1) * 4);
    const modelBoundBindDesc = keep(alloc(bindDescSize), bindDescSize);
    zero(modelBoundBindDesc, bindDescSize);
    writeU32(modelBoundBindDesc, 24, modelBoundOutputBuffer);
    writeU32(modelBoundBindDesc, 28, tinyLlamaVocabSize);
    const modelBoundWithBuffer = bindModelBufferSession(program, compatibleModel, modelBoundBindDesc);
    sessions.push(a, b, prefilled, advanced, bulkAdvanced, bulkExpected, executed, bound, hostKv, modelBound, modelBoundWithBuffer);

    const tooSmall = keep(alloc((tinyLlamaVocabSize - 1) * 4), (tinyLlamaVocabSize - 1) * 4);
    expectStatus(tokenStep(a, 0, tooSmall, tinyLlamaVocabSize - 1).status, shapeMismatch, "too-small tiny LLaMA step");
    if (sessionPosition(a) !== 0) throw new Error("too-small tiny LLaMA step mutated position");

    const logitsA = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(logitsA, Array(tinyLlamaVocabSize + 1).fill(-999));
    const stepA = tokenStep(a, 0, logitsA, tinyLlamaVocabSize + 1);
    check(stepA.status);
    if (stepA.outputLen !== tinyLlamaVocabSize || sessionPosition(a) !== 1 || sessionPosition(b) !== 0) {
      throw new Error("unexpected tiny LLaMA step state");
    }
    const modelBoundLogits = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(modelBoundLogits, Array(tinyLlamaVocabSize + 1).fill(-3434));
    const modelBoundStep = tokenStep(modelBound, 0, modelBoundLogits, tinyLlamaVocabSize + 1);
    check(modelBoundStep.status);
    if (modelBoundStep.outputLen !== tinyLlamaVocabSize || sessionPosition(modelBound) !== 1) {
      throw new Error("unexpected model-bound tiny LLaMA step state");
    }
    expectClose(readF32Array(modelBoundLogits, tinyLlamaVocabSize), readF32Array(logitsA, tinyLlamaVocabSize));
    if (readF32Array(modelBoundLogits + tinyLlamaVocabSize * 4, 1)[0] !== -3434) {
      throw new Error("model-bound tiny LLaMA step wrote past vocab logits");
    }
    const modelBoundBufferStep = tokenStep(modelBoundWithBuffer, 0, 0, 0);
    check(modelBoundBufferStep.status);
    if (modelBoundBufferStep.outputLen !== tinyLlamaVocabSize || sessionPosition(modelBoundWithBuffer) !== 1) {
      throw new Error("unexpected model-bound buffer tiny LLaMA step state");
    }
    bufferRead(modelBoundOutputBuffer, 0, modelBoundOutput, (tinyLlamaVocabSize + 1) * 4);
    expectClose(readF32Array(modelBoundOutput, tinyLlamaVocabSize), readF32Array(logitsA, tinyLlamaVocabSize));
    if (readF32Array(modelBoundOutput + tinyLlamaVocabSize * 4, 1)[0] !== -1212) {
      throw new Error("model-bound buffer tiny LLaMA step wrote past vocab logits");
    }
    expectArgmax(tokenArgmax(a, logitsA, tinyLlamaVocabSize), logitsA, tinyLlamaVocabSize);
    expectArgmax(tokenSample(a, logitsA, tinyLlamaVocabSize), logitsA, tinyLlamaVocabSize);
    expectStatus(tokenArgmax(b).status, invalidArgument, "unbound tiny LLaMA argmax");
    expectStatus(tokenSample(b).status, invalidArgument, "unbound tiny LLaMA sample");

    sessionReset(a);
    if (sessionPosition(a) !== 0) throw new Error("tiny LLaMA reset did not clear position");
    const logitsReset = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(logitsReset, Array(tinyLlamaVocabSize + 1).fill(-888));
    const stepReset = tokenStep(a, 0, logitsReset, tinyLlamaVocabSize + 1);
    check(stepReset.status);
    expectClose(readF32Array(logitsReset, tinyLlamaVocabSize), readF32Array(logitsA, tinyLlamaVocabSize));
    if (readF32Array(logitsReset + tinyLlamaVocabSize * 4, 1)[0] !== -888) {
      throw new Error("reset tiny LLaMA step wrote past vocab logits");
    }

    const stepBound = tokenStep(bound, 0, 0, 0);
    check(stepBound.status);
    bufferRead(boundOutputBuffer, 0, boundOutput, (tinyLlamaVocabSize + 1) * 4);
    expectClose(readF32Array(boundOutput, tinyLlamaVocabSize), readF32Array(logitsA, tinyLlamaVocabSize));
    expectArgmax(tokenArgmax(bound), boundOutput, tinyLlamaVocabSize);
    expectArgmax(tokenSample(bound), boundOutput, tinyLlamaVocabSize);
    if (readF32Array(boundOutput + tinyLlamaVocabSize * 4, 1)[0] !== -666) {
      throw new Error("bound-output tiny LLaMA step wrote past vocab logits");
    }
    sessionReset(bound);
    writeF32Array(boundOutput, Array(tinyLlamaVocabSize + 1).fill(-222));
    bufferWrite(boundOutputBuffer, 0, boundOutput, (tinyLlamaVocabSize + 1) * 4);
    const tokenZero = keep(alloc(4), 4);
    writeU32Array(tokenZero, [0]);
    const executeArgmax = tokenExecuteArgmax(bound, tokenZero, 1);
    expectArgmax(executeArgmax, logitsA, tinyLlamaVocabSize);
    if (sessionPosition(bound) !== 1) throw new Error("one-call tiny LLaMA argmax did not advance position");
    bufferRead(boundOutputBuffer, 0, boundOutput, (tinyLlamaVocabSize + 1) * 4);
    if (readF32Array(boundOutput + tinyLlamaVocabSize * 4, 1)[0] !== -222) {
      throw new Error("one-call tiny LLaMA argmax wrote past vocab logits");
    }
    sessionReset(bound);
    writeF32Array(boundOutput, Array(tinyLlamaVocabSize + 1).fill(-111));
    bufferWrite(boundOutputBuffer, 0, boundOutput, (tinyLlamaVocabSize + 1) * 4);
    const executeSample = tokenExecuteSample(bound, tokenZero, 1);
    expectArgmax(executeSample, logitsA, tinyLlamaVocabSize);
    if (sessionPosition(bound) !== 1) throw new Error("one-call tiny LLaMA sample did not advance position");
    bufferRead(boundOutputBuffer, 0, boundOutput, (tinyLlamaVocabSize + 1) * 4);
    if (readF32Array(boundOutput + tinyLlamaVocabSize * 4, 1)[0] !== -111) {
      throw new Error("one-call tiny LLaMA sample wrote past vocab logits");
    }
    sessionReset(bound);
    const expectedFirst = tokenExecuteSample(bound, tokenZero, 1, 1, 123, 1.0);
    check(expectedFirst.status);
    const expectedSecondToken = keep(alloc(4), 4);
    writeU32Array(expectedSecondToken, [expectedFirst.token]);
    const expectedSecond = tokenExecuteSample(bound, expectedSecondToken, 1, 1, 124, 1.0);
    check(expectedSecond.status);
    sessionReset(bound);
    const generatedTokens = keep(alloc(2 * 4), 2 * 4);
    writeU32Array(generatedTokens, [777, 777]);
    const generated = tokenGenerateSample(bound, tokenZero, 1, generatedTokens, 2, 1, 123, 1.0);
    check(generated.status);
    if (
      generated.tokensGenerated !== 2 ||
      u32(generatedTokens, 0) !== expectedFirst.token ||
      u32(generatedTokens, 4) !== expectedSecond.token ||
      generated.lastToken !== expectedSecond.token ||
      generated.lastLogit !== expectedSecond.logit ||
      sessionPosition(bound) !== 2
    ) {
      throw new Error("unexpected generated tiny LLaMA sample");
    }
    sessionReset(bound);
    const greedyGeneratedTokens = keep(alloc(2 * 4), 2 * 4);
    writeU32Array(greedyGeneratedTokens, [777, 777]);
    const greedyGenerated = tokenGenerateArgmax(bound, tokenZero, 1, greedyGeneratedTokens, 2);
    check(greedyGenerated.status);
    if (
      greedyGenerated.tokensGenerated !== 2 ||
      u32(greedyGeneratedTokens, 0) !== expectedFirst.token ||
      u32(greedyGeneratedTokens, 4) !== expectedSecond.token ||
      greedyGenerated.lastToken !== expectedSecond.token ||
      greedyGenerated.lastLogit !== expectedSecond.logit ||
      sessionPosition(bound) !== 2
    ) {
      throw new Error("unexpected generated tiny LLaMA argmax");
    }

    const logitsB = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(logitsB, Array(tinyLlamaVocabSize + 1).fill(-777));
    check(tokenStep(b, 0, logitsB, tinyLlamaVocabSize + 1).status);
    expectClose(readF32Array(logitsB, tinyLlamaVocabSize), readF32Array(logitsA, tinyLlamaVocabSize));
    if (readF32Array(logitsA + tinyLlamaVocabSize * 4, 1)[0] !== -999 || readF32Array(logitsB + tinyLlamaVocabSize * 4, 1)[0] !== -777) {
      throw new Error("tiny LLaMA step wrote past vocab logits");
    }

    const tokens = keep(alloc(2 * 4), 2 * 4);
    writeU32Array(tokens, [0, 1]);
    expectStatus(tokenPrefill(prefilled, tokens, 2, tooSmall, tinyLlamaVocabSize - 1).status, shapeMismatch, "too-small tiny LLaMA prefill");
    if (sessionPosition(prefilled) !== 0) throw new Error("too-small tiny LLaMA prefill mutated position");

    const prefillLogits = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(prefillLogits, Array(tinyLlamaVocabSize + 1).fill(-222));
    const prefill = tokenPrefill(prefilled, tokens, 2, prefillLogits, tinyLlamaVocabSize + 1);
    check(prefill.status);
    if (prefill.outputLen !== tinyLlamaVocabSize || sessionPosition(prefilled) !== 2) {
      throw new Error("unexpected tiny LLaMA prefill state");
    }
    if (readF32Array(prefillLogits + tinyLlamaVocabSize * 4, 1)[0] !== -222) {
      throw new Error("tiny LLaMA prefill wrote past vocab logits");
    }

    check(tokenAdvance(advanced, 0));
    const advancedLogits = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(advancedLogits, Array(tinyLlamaVocabSize + 1).fill(-444));
    check(tokenStep(advanced, 1, advancedLogits, tinyLlamaVocabSize + 1).status);
    expectClose(readF32Array(advancedLogits, tinyLlamaVocabSize), readF32Array(prefillLogits, tinyLlamaVocabSize));
    if (readF32Array(advancedLogits + tinyLlamaVocabSize * 4, 1)[0] !== -444) {
      throw new Error("tiny LLaMA advanced step wrote past vocab logits");
    }

    const tooManyTokens = keep(alloc(5 * 4), 5 * 4);
    writeU32Array(tooManyTokens, [0, 1, 2, 3, 4]);
    expectStatus(tokenAdvanceTokens(bulkAdvanced, tooManyTokens, 5), shapeMismatch, "too-long tiny LLaMA bulk advance");
    if (sessionPosition(bulkAdvanced) !== 0) throw new Error("too-long tiny LLaMA bulk advance mutated position");
    const tooManyExecuteLogits = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(tooManyExecuteLogits, Array(tinyLlamaVocabSize + 1).fill(-909));
    const tooManyExecute = tokenExecute(executed, tooManyTokens, 5, executeOutputLogits, tooManyExecuteLogits, tinyLlamaVocabSize + 1);
    expectStatus(tooManyExecute.status, shapeMismatch, "too-long tiny LLaMA execute");
    if (tooManyExecute.outputLen !== 0) throw new Error("too-long tiny LLaMA execute did not clear result");
    if (sessionPosition(executed) !== 0) throw new Error("too-long tiny LLaMA execute mutated position");
    if (readF32Array(tooManyExecuteLogits + tinyLlamaVocabSize * 4, 1)[0] !== -909) {
      throw new Error("too-long tiny LLaMA execute wrote past vocab logits");
    }

    check(tokenAdvanceTokens(bulkAdvanced, tokens, 2));
    if (sessionPosition(bulkAdvanced) !== 2) throw new Error("tiny LLaMA bulk advance did not update position");
    const bulkLogits = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(bulkLogits, Array(tinyLlamaVocabSize + 1).fill(-123));
    check(tokenStep(bulkAdvanced, 2, bulkLogits, tinyLlamaVocabSize + 1).status);

    const expectedTokens = keep(alloc(3 * 4), 3 * 4);
    writeU32Array(expectedTokens, [0, 1, 2]);
    const expectedLogits = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(expectedLogits, Array(tinyLlamaVocabSize + 1).fill(-321));
    check(tokenPrefill(bulkExpected, expectedTokens, 3, expectedLogits, tinyLlamaVocabSize + 1).status);
    expectClose(readF32Array(bulkLogits, tinyLlamaVocabSize), readF32Array(expectedLogits, tinyLlamaVocabSize));
    if (readF32Array(bulkLogits + tinyLlamaVocabSize * 4, 1)[0] !== -123 || readF32Array(expectedLogits + tinyLlamaVocabSize * 4, 1)[0] !== -321) {
      throw new Error("tiny LLaMA bulk advance comparison wrote past vocab logits");
    }

    expectStatus(tokenExecute(executed, tokens, 2, executeOutputLogits, tooSmall, tinyLlamaVocabSize - 1).status, shapeMismatch, "too-small tiny LLaMA execute");
    if (sessionPosition(executed) !== 0) throw new Error("too-small tiny LLaMA execute mutated position");

    const executePrefillLogits = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(executePrefillLogits, Array(tinyLlamaVocabSize + 1).fill(-654));
    const executePrefill = tokenExecute(executed, tokens, 2, executeOutputLogits, executePrefillLogits, tinyLlamaVocabSize + 1);
    check(executePrefill.status);
    if (executePrefill.outputLen !== tinyLlamaVocabSize || sessionPosition(executed) !== 2) {
      throw new Error("unexpected tiny LLaMA execute prefill state");
    }
    expectClose(readF32Array(executePrefillLogits, tinyLlamaVocabSize), readF32Array(prefillLogits, tinyLlamaVocabSize));
    if (readF32Array(executePrefillLogits + tinyLlamaVocabSize * 4, 1)[0] !== -654) {
      throw new Error("tiny LLaMA execute prefill wrote past vocab logits");
    }

    sessionReset(executed);
    const executeAdvance = tokenExecute(executed, tokens, 2, executeOutputNone, 0, 0);
    check(executeAdvance.status);
    if (executeAdvance.outputLen !== 0 || sessionPosition(executed) !== 2) {
      throw new Error("unexpected tiny LLaMA no-output execute state");
    }
    const tokenTwo = keep(alloc(4), 4);
    writeU32Array(tokenTwo, [2]);
    const executeNextLogits = keep(alloc((tinyLlamaVocabSize + 1) * 4), (tinyLlamaVocabSize + 1) * 4);
    writeF32Array(executeNextLogits, Array(tinyLlamaVocabSize + 1).fill(-876));
    check(tokenExecute(executed, tokenTwo, 1, executeOutputLogits, executeNextLogits, tinyLlamaVocabSize + 1).status);
    expectClose(readF32Array(executeNextLogits, tinyLlamaVocabSize), readF32Array(expectedLogits, tinyLlamaVocabSize));
    if (readF32Array(executeNextLogits + tinyLlamaVocabSize * 4, 1)[0] !== -876) {
      throw new Error("tiny LLaMA execute next-token wrote past vocab logits");
    }

    sessionReset(bound);
    writeF32Array(boundOutput, Array(tinyLlamaVocabSize + 1).fill(-555));
    bufferWrite(boundOutputBuffer, 0, boundOutput, (tinyLlamaVocabSize + 1) * 4);
    const boundPrefill = tokenPrefill(bound, tokens, 2, 0, 0);
    check(boundPrefill.status);
    bufferRead(boundOutputBuffer, 0, boundOutput, (tinyLlamaVocabSize + 1) * 4);
    expectClose(readF32Array(boundOutput, tinyLlamaVocabSize), readF32Array(advancedLogits, tinyLlamaVocabSize));
    if (readF32Array(boundOutput + tinyLlamaVocabSize * 4, 1)[0] !== -555) {
      throw new Error("bound-output tiny LLaMA prefill wrote past vocab logits");
    }

    append(`zgml browser wasm ffi tiny llama smoke ok: ${tinyLlamaVocabSize} logits`);
  } finally {
    for (let i = sessions.length - 1; i >= 0; i -= 1) exportsRef.zgml_session_free(sessions[i]);
    if (program !== 0) exportsRef.zgml_program_free(program);
    if (compatibleModel !== 0) exportsRef.zgml_model_free(compatibleModel);
    if (model !== 0) exportsRef.zgml_model_free(model);
  }
}

async function run() {
  markBrowserSmokeStage("fetch-wasm");
  const response = await fetch("../../zig-out/bin/zgml_c.wasm");
  if (!response.ok) throw new Error(`failed to fetch wasm: ${response.status}`);
  const imports = { wasi_snapshot_preview1: makeWasiShim(), zgml_host: makeZgmlHostImports() };
  markBrowserSmokeStage("instantiate-wasm");
  const { instance } = await WebAssembly.instantiateStreaming(response, imports);
  exportsRef = instance.exports;
  markBrowserSmokeStage("init-abi");
  initAbiStructSizes();
  markBrowserSmokeStage("request-webgpu");
  const browserGpuInfo = await requestOptionalBrowserGpuDeviceInfo();
  markBrowserSmokeStage("init-host-runtime");
  hostDevice = new WasmWebGpuDevice({ device: browserGpuInfo.device, registry: hostResources });
  hostRuntime = new WasmWebGpuSessionRuntime({
    view,
    ok,
    unsupported,
  });
  recordBrowserWebGpuEvidence(browserGpuInfo, hostDevice);

  try {
    markBrowserSmokeStage("host-resource-descriptor-immutability");
    expectHostResourceDescriptorImmutability();
    markBrowserSmokeStage("executor-owned-resource-cleanup");
    expectExecutorOwnedResourceCleanupForgetsHostResources();
    markBrowserSmokeStage("imported-llama-binding-descriptor-snapshot");
    expectImportedLlamaBindingDescriptorsAreSnapshotted();
    markBrowserSmokeStage("llama-layer-role-defaults");
    expectLlamaLayerRoleDefaults();
    markBrowserSmokeStage("llama-projection-gpu-bind-group-cache");
    expectLlamaProjectionGpuBindGroupCache();
    markBrowserSmokeStage("llama-pattern-params-buffer-reuse");
    expectLlamaPatternParamsBufferReuse();
    markBrowserSmokeStage("llama-token-window-adapter-reuse");
    expectLlamaTokenWindowExecutorReusesAdapterContext();
    markBrowserSmokeStage("webgpu-readback-staging-reuse");
    await expectWasmWebGpuDeviceReusesReadbackStaging();
    markBrowserSmokeStage("llama-device-topk-one-uses-argmax");
    await expectLlamaDeviceTopKOneUsesArgmaxSelector();
    markBrowserSmokeStage("llama-device-selection-malformed-evidence");
    await expectLlamaDeviceSelectionRejectsMalformedSelectorEvidence();
    markBrowserSmokeStage("llama-full-logits-sampler-scratch-reuse");
    await expectLlamaFullLogitsSamplerReusesScratch();
    markBrowserSmokeStage("llama-full-logits-readback-scratch-reuse");
    await expectLlamaFullLogitsReadbackReusesScratch();
    markBrowserSmokeStage("llama-generation-private-selection-scratch");
    await expectLlamaGenerationUsesPrivateSelectionScratch();
    markBrowserSmokeStage("llama-generation-borrowed-hot-token-window");
    await expectLlamaGenerationBorrowsHotTokenWindow();
    markBrowserSmokeStage("llama-scalar-step-token-scratch");
    await expectLlamaScalarStepsBorrowTokenScratch();
    markBrowserSmokeStage("llama-advance-no-output-internal");
    await expectLlamaAdvanceTokensUsesNoOutputInternal();
    markBrowserSmokeStage("llama-executor-outcome-scratch");
    expectLlamaExecutorOutcomeScratchClearsInvalidResults();
    markBrowserSmokeStage("llama-validation-tokens-len-window");
    expectLlamaValidationUsesTokensLenWindow();
    markBrowserSmokeStage("llama-diagnostic-history-bounded");
    expectLlamaDiagnosticHistoryIsBounded();
    markBrowserSmokeStage("llama-device-argmax-selector-reuse");
    await expectLlamaDeviceArgmaxSelectorResourceReuseAndValidation();
    markBrowserSmokeStage("llama-device-topk-selector-reuse");
    await expectLlamaDeviceTopKSelectorResourceReuse();
    markBrowserSmokeStage("runtime-info");
    checkRuntimeInfo();
    markBrowserSmokeStage("supported-checkpoint-catalog");
    checkSupportedCheckpointCatalog();
    markBrowserSmokeStage("safetensors-header-probe");
    checkSafetensorsHeaderProbe();
    markBrowserSmokeStage("tiny-llama-safetensors-data-load");
    runTinyLlamaSafetensorsDataLoadSmoke();
    markBrowserSmokeStage("tiny-linear");
    runTinyLinearSmoke();
    markBrowserSmokeStage("tiny-linear-webgpu");
    await runTinyLinearWebGpuCompileOnlySmoke();
    markBrowserSmokeStage("tiny-llama");
    runTinyLlamaSmoke();
    markBrowserSmokeStage("tiny-llama-webgpu");
    await runTinyLlamaWebGpuCompileOnlySmoke();
    markBrowserSmokeStage("append-llama-profile-evidence");
    appendBrowserLlamaProfileEvidence();
    document.documentElement.dataset.zgmlWasmSmoke = "passed";
  } finally {
    for (let i = buffers.length - 1; i >= 0; i -= 1) {
      exportsRef.zgml_buffer_free(buffers[i]);
    }
    for (let i = allocations.length - 1; i >= 0; i -= 1) {
      const [ptr, size] = allocations[i];
      free(ptr, size);
    }
    hostDevice.destroy();
  }
}

run().catch((error) => {
  logEl.textContent = `failed: ${error.stack || error.message || error}`;
  document.documentElement.dataset.zgmlWasmSmoke = "failed";
  document.documentElement.dataset.zgmlWasmSmokeError = String(error && error.stack ? error.stack : error);
});
