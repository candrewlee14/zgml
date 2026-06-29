export const wasmWebgpuInterop = Object.freeze({
  deviceHandle: Symbol.for("zgml.wasm.webgpu.deviceHandle"),
  handle: Symbol.for("zgml.wasm.webgpu.resourceHandle"),
  byteLength: Symbol.for("zgml.wasm.webgpu.byteLength"),
  byteOffset: Symbol.for("zgml.wasm.webgpu.byteOffset"),
  placement: Symbol.for("zgml.wasm.webgpu.placement"),
  access: Symbol.for("zgml.wasm.webgpu.access"),
  usage: Symbol.for("zgml.wasm.webgpu.usage"),
  importSource: Symbol.for("zgml.wasm.webgpu.importSource"),
});

const fnv64Offset = 0xcbf29ce484222325n;
const fnv64Prime = 0x100000001b3n;
const u64Mask = 0xffffffffffffffffn;
const mockStorage = Symbol("zgml.wasm.webgpu.mockStorage");
const gpuShadowStorage = new WeakMap();
const hostResourceDeviceHandles = new WeakMap();
const wasmWebGpuResourceDevices = new WeakMap();
const destroyedWasmWebGpuResources = new WeakSet();
let nextWasmWebGpuDeviceHandle = 0x01000000;
const llamaBlockScalarWindowMaxContextLength = 64;
const defaultLlamaDiagnosticHistoryLimit = 128;
const zgmlModelTinyLinearKind = 1;
const zgmlModelTinyLlamaKind = 2;
const zgmlModelSmollm135mKind = 3;
const zgmlModelTinyLlama2LayerKind = 4;
const zgmlBackendWebGpu = 3;
const zgmlBufferStorageNone = 0;
const zgmlBufferStorageExternalResource = 2;

export const webgpuBufferUsage = Object.freeze({
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
});

function webgpuUsageBits() {
  const usage = typeof globalThis.GPUBufferUsage === "object" ? globalThis.GPUBufferUsage : {};
  return {
    COPY_SRC: usage.COPY_SRC ?? webgpuBufferUsage.COPY_SRC,
    COPY_DST: usage.COPY_DST ?? webgpuBufferUsage.COPY_DST,
    STORAGE: usage.STORAGE ?? webgpuBufferUsage.STORAGE,
  };
}

export function externalResourceAccessFlags(access = "readwrite") {
  if (Number.isInteger(access) && access > 0 && (access & ~3) === 0) return access;
  if (access === "read") return 1;
  if (access === "write") return 2;
  if (access === "readwrite" || access === "read_write") return 3;
  if (Array.isArray(access)) {
    let flags = 0;
    for (const part of access) flags |= externalResourceAccessFlags(part);
    if (flags === 0 || (flags & ~3) !== 0) throw new Error(`invalid resource access: ${access}`);
    return flags;
  }
  if (access && typeof access === "object") {
    let flags = 0;
    if (access.read) flags |= 1;
    if (access.write) flags |= 2;
    if (flags === 0) throw new Error("resource access must include read or write");
    return flags;
  }
  throw new Error(`invalid resource access: ${access}`);
}

export function externalResourcePlacementId(placement = "webgpu") {
  if (Number.isInteger(placement) && placement >= 0 && placement <= 0xffffffff) return placement;
  if (placement === "auto") return 0;
  if (placement === "cpu") return 1;
  if (placement === "metal") return 2;
  if (placement === "webgpu" || placement === "wgpu") return 3;
  throw new Error(`invalid resource placement: ${placement}`);
}

export function webgpuRequiredUsageFlags(access = "readwrite", options = {}) {
  const bits = webgpuUsageBits();
  const flags = externalResourceAccessFlags(access);
  let required = bits.STORAGE;
  if (options.hostTransfer !== false) {
    if ((flags & 1) !== 0) required |= bits.COPY_DST;
    if ((flags & 2) !== 0) required |= bits.COPY_SRC;
  }
  return required;
}

function llamaProofHeadGeometry(shape, label) {
  if (!shape || typeof shape !== "object") throw new Error(`${label} needs a shape object`);
  const attentionHeadSize = shape.attentionHeadSize ?? shape.headSize;
  const qSize = shape.qSize ?? shape.hiddenSize;
  const kvSize = shape.kvSize ?? shape.hiddenSize;
  if (!Number.isSafeInteger(attentionHeadSize) || attentionHeadSize <= 0) {
    throw new Error(`${label} needs a positive attention head size`);
  }
  if (!Number.isSafeInteger(qSize) || !Number.isSafeInteger(kvSize) || qSize <= 0 || kvSize <= 0) {
    throw new Error(`${label} needs positive Q/KV sizes`);
  }
  const numAttentionHeads = qSize / attentionHeadSize;
  const numKeyValueHeads = kvSize / attentionHeadSize;
  if (!Number.isInteger(numAttentionHeads) || !Number.isInteger(numKeyValueHeads)) {
    throw new Error(`${label} head geometry is not integral`);
  }
  return { numAttentionHeads, numKeyValueHeads };
}

function llamaProofCommonConfig(shape, label) {
  const { numAttentionHeads, numKeyValueHeads } = llamaProofHeadGeometry(shape, label);
  if (!Number.isSafeInteger(shape.hiddenSize) || !Number.isSafeInteger(shape.ffnSize) || !Number.isSafeInteger(shape.layers) || !Number.isSafeInteger(shape.vocabSize)) {
    throw new Error(`${label} needs hidden, FFN, layer, and vocab sizes`);
  }
  return {
    attention_dropout: 0,
    hidden_size: shape.hiddenSize,
    hidden_act: "silu",
    hidden_dropout: 0,
    intermediate_size: shape.ffnSize,
    max_position_embeddings: 16,
    num_attention_heads: numAttentionHeads,
    num_hidden_layers: shape.layers,
    num_key_value_heads: numKeyValueHeads,
    partial_rotary_factor: 1,
    vocab_size: shape.vocabSize,
  };
}

export function mistralSlidingWindowProofConfig(shape, overrides = {}) {
  return {
    ...llamaProofCommonConfig(shape, "Mistral sliding-window proof config"),
    architectures: ["MistralForCausalLM"],
    attention_bias: false,
    bos_token_id: 1,
    eos_token_id: [2, 3],
    mlp_bias: false,
    model_type: "mistral",
    pad_token_id: 0,
    pretraining_tp: 1,
    rms_norm_eps: 0.00006,
    rope_scaling: null,
    rope_theta: 15000,
    sliding_window: 2,
    torch_dtype: "float32",
    use_sliding_window: true,
    use_cache: true,
    ...overrides,
  };
}

export function llama3RopeProofConfig(shape, overrides = {}) {
  return {
    ...llamaProofCommonConfig(shape, "Llama 3 RoPE proof config"),
    architectures: ["LlamaForCausalLM"],
    attention_bias: false,
    mlp_bias: false,
    model_type: "llama",
    pretraining_tp: 1,
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
    ...overrides,
  };
}

export function smollm3NopeProofConfig(shape, overrides = {}) {
  return {
    ...llamaProofCommonConfig(shape, "SmolLM3 NoPE proof config"),
    architectures: ["SmolLM3ForCausalLM"],
    attention_bias: false,
    bos_token_id: 1,
    eos_token_id: [2, 3],
    mlp_bias: false,
    model_type: "smollm3",
    no_rope_layer_interval: 2,
    pad_token_id: 0,
    pretraining_tp: 2,
    rms_norm_eps: 0.000085,
    rope_scaling: null,
    rope_theta: 18000,
    sliding_window: null,
    torch_dtype: "bfloat16",
    tie_word_embeddings: false,
    use_sliding_window: false,
    use_cache: true,
    ...overrides,
  };
}

export function biasedQwen2ProofConfig(shape, overrides = {}) {
  return {
    ...llamaProofCommonConfig(shape, "biased Qwen2 proof config"),
    architectures: ["Qwen2ForCausalLM"],
    attention_bias: true,
    bos_token_id: 1,
    eos_token_id: [2, 3],
    head_dim: shape.attentionHeadSize ?? shape.headSize,
    mlp_bias: true,
    model_type: "qwen2",
    pad_token_id: 0,
    pretraining_tp: 1,
    rms_norm_eps: 0.000075,
    rope_scaling: null,
    rope_theta: 16500,
    sliding_window: null,
    torch_dtype: "float32",
    tie_word_embeddings: false,
    use_sliding_window: false,
    use_cache: true,
    ...overrides,
  };
}

export function qwen3QkNormProofConfig(shape, overrides = {}) {
  return {
    ...llamaProofCommonConfig(shape, "Qwen3 Q/K-norm proof config"),
    architectures: ["Qwen3ForCausalLM"],
    attention_bias: false,
    bos_token_id: 1,
    eos_token_id: [2, 3],
    head_dim: shape.attentionHeadSize ?? shape.headSize,
    mlp_bias: false,
    model_type: "qwen3",
    pad_token_id: 0,
    pretraining_tp: 1,
    qk_norm: true,
    rms_norm_eps: 0.000082,
    rope_scaling: null,
    rope_theta: 17500,
    sliding_window: null,
    torch_dtype: "bfloat16",
    tie_word_embeddings: false,
    use_sliding_window: false,
    use_cache: true,
    ...overrides,
  };
}

function destroyWasmWebGpuResource(resource) {
  if (!resource || (typeof resource !== "object" && typeof resource !== "function")) return;
  if (destroyedWasmWebGpuResources.has(resource)) return;
  const destroy = resource.destroy;
  if (typeof destroy !== "function") return;
  destroyedWasmWebGpuResources.add(resource);
  destroy.call(resource);
}

function isDestroyedWasmWebGpuResource(resource) {
  if (!resource || (typeof resource !== "object" && typeof resource !== "function")) return false;
  return destroyedWasmWebGpuResources.has(resource) || resource.destroyed === true;
}

function assertLiveWasmWebGpuResource(resource, label) {
  if (isDestroyedWasmWebGpuResource(resource)) {
    throw new Error(`${label} was already destroyed`);
  }
}

function forgetWasmWebGpuDeviceResource(device, resource) {
  if (!resource || (typeof resource !== "object" && typeof resource !== "function")) return;
  if (device && Array.isArray(device.ownedResources)) {
    for (let i = device.ownedResources.length - 1; i >= 0; i -= 1) {
      if (device.ownedResources[i] === resource) device.ownedResources.splice(i, 1);
    }
  }
  device?.registry?.unregister?.(resource);
  hostResourceDeviceHandles.delete(resource);
  wasmWebGpuResourceDevices.delete(resource);
  gpuShadowStorage.delete(resource);
}

function destroyWasmWebGpuDeviceResource(device, resource) {
  const owner = device ?? wasmWebGpuResourceDevices.get(resource);
  destroyWasmWebGpuResource(resource);
  forgetWasmWebGpuDeviceResource(owner, resource);
}

function destroyTrackedWasmWebGpuResource(resource) {
  destroyWasmWebGpuDeviceResource(null, resource);
}

function readField(source, keys) {
  if (!source || typeof source !== "object") return undefined;
  for (const key of keys) {
    const value = source[key];
    if (value !== undefined) return value;
  }
  return undefined;
}

function readNumber(source, keys) {
  const value = readField(source, keys);
  return typeof value === "number" ? value : undefined;
}

function readString(source, keys) {
  const value = readField(source, keys);
  return typeof value === "string" ? value : undefined;
}

export function validateWebGpuResourceUsage(resource, access = "readwrite", options = {}) {
  if (options.validateUsage === false) return;
  const placement = externalResourcePlacementId(options.placement ?? readField(resource, [wasmWebgpuInterop.placement, "placement", "backend", "device"]) ?? "webgpu");
  if (placement !== externalResourcePlacementId("webgpu")) return;
  const usage = readNumber(resource, [wasmWebgpuInterop.usage, "usage"]);
  if (usage === undefined || usage === 0) return;
  const required = options.requiredUsage ?? webgpuRequiredUsageFlags(access, options);
  if ((usage & required) !== required) {
    throw new Error(`WebGPU resource usage ${usage} does not include required flags ${required}`);
  }
}

function descriptorEvidence(descriptor) {
  const placement = externalResourcePlacementId(descriptor.placement ?? "webgpu");
  const access = externalResourceAccessFlags(descriptor.access ?? "readwrite");
  const handle = descriptor.handle;
  const byteOffset = descriptor.byteOffset ?? 0;
  const byteLength = descriptor.byteLength ?? descriptor.byteLen;
  const deviceHandle = assertDeviceHandle(descriptor.deviceHandle);
  assertHandle(handle);
  assertRange(byteOffset, byteLength);
  validateHostResourceViewRange(descriptor.resource, byteOffset, byteLength, "host resource table descriptor");
  assertU32(byteOffset, "host resource table byteOffset");
  assertU32(byteLength, "host resource table byteLength");
  return {
    access,
    byteLength,
    byteOffset,
    deviceHandle,
    handle,
    placement,
  };
}

function normalizeResourceTableRole(role) {
  if (role === undefined || role === null || role === "") return "";
  if (typeof role !== "string") {
    throw new Error(`host resource table role must be a string: ${role}`);
  }
  return role;
}

function resourceTableEntryEvidence(entry) {
  const descriptor = entry && typeof entry === "object" && entry.descriptor && typeof entry.descriptor === "object"
    ? entry.descriptor
    : entry;
  const role = normalizeResourceTableRole(entry && typeof entry === "object" ? entry.role ?? entry.kind : "");
  return Object.freeze({
    ...descriptorEvidence(descriptor),
    role,
  });
}

function hashByte(hash, byte) {
  return ((hash ^ BigInt(byte & 0xff)) * fnv64Prime) & u64Mask;
}

function hashU32(hash, value) {
  assertU32(value, "hash field");
  let next = hash;
  next = hashByte(next, value);
  next = hashByte(next, value >>> 8);
  next = hashByte(next, value >>> 16);
  next = hashByte(next, value >>> 24);
  return next;
}

function hashNonNegativeInteger(hash, value, label) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`invalid ${label}: ${value}`);
  }
  const wide = BigInt(value);
  let next = hashU32(hash, Number(wide & 0xffffffffn));
  next = hashU32(next, Number((wide >> 32n) & 0xffffffffn));
  return next;
}

function hashUint64(hash, value, label) {
  let wide;
  if (typeof value === "bigint") {
    wide = value;
  } else if (Number.isSafeInteger(value)) {
    wide = BigInt(value);
  } else {
    throw new Error(`invalid ${label}: ${value}`);
  }
  if (wide < 0n || wide > u64Mask) {
    throw new Error(`invalid ${label}: ${value}`);
  }
  let next = hashU32(hash, Number(wide & 0xffffffffn));
  next = hashU32(next, Number((wide >> 32n) & 0xffffffffn));
  return next;
}

function hashOptionalUint64(hash, value, label) {
  if (value === undefined || value === null) return hashU32(hash, 0);
  return hashUint64(hashU32(hash, 1), value, label);
}

function hashString(hash, value) {
  let next = hashU32(hash, value.length);
  for (let i = 0; i < value.length; i += 1) {
    next = hashU32(next, value.charCodeAt(i));
  }
  return next;
}

function hashResourceDescriptorFields(hash, evidence) {
  let next = hash;
  next = hashU32(next, evidence.placement);
  next = hashU32(next, evidence.access);
  next = hashU32(next, evidence.deviceHandle);
  next = hashU32(next, evidence.handle);
  next = hashU32(next, evidence.byteOffset);
  next = hashU32(next, evidence.byteLength);
  return next;
}

export function hostResourceDescriptorHash(descriptor) {
  const evidence = descriptorEvidence(descriptor);
  return hashResourceDescriptorFields(fnv64Offset, evidence);
}

function snapshotHostResourceDescriptor(descriptor) {
  return Object.freeze({
    ...descriptorEvidence(descriptor),
    resource: descriptor.resource,
  });
}

export function describeHostResourceTable(descriptors, options = {}) {
  if (!Array.isArray(descriptors)) {
    throw new Error("host resource table descriptors must be an array");
  }
  const requireSingleDevice = options.requireSingleDevice !== false;
  const requireUniqueRoles = options.requireUniqueRoles !== false;
  const entries = descriptors.map(resourceTableEntryEvidence);
  const roles = new Set();
  let deviceHandle = 0;
  let byteLength = 0;
  let hash = hashU32(fnv64Offset, entries.length);
  let descriptorHash = hashU32(fnv64Offset, entries.length);
  for (const entry of entries) {
    if (requireUniqueRoles && entry.role !== "") {
      if (roles.has(entry.role)) {
        throw new Error(`host resource table has duplicate role: ${entry.role}`);
      }
      roles.add(entry.role);
    }
    byteLength += entry.byteLength;
    if (entry.placement === externalResourcePlacementId("webgpu")) {
      if (requireSingleDevice && entry.deviceHandle === 0) {
        throw new Error("WebGPU host resource table needs deviceHandle provenance");
      }
      if (entry.deviceHandle !== 0) {
        if (deviceHandle === 0) {
          deviceHandle = entry.deviceHandle;
        } else if (requireSingleDevice && deviceHandle !== entry.deviceHandle) {
          throw new Error("WebGPU host resource table mixes device handles");
        }
      }
    }
    descriptorHash = hashResourceDescriptorFields(descriptorHash, entry);
    hash = hashString(hash, entry.role);
    hash = hashResourceDescriptorFields(hash, entry);
  }
  return Object.freeze({
    byteLength,
    descriptorHash,
    descriptors: Object.freeze(entries),
    deviceHandle,
    hash,
    resourceCount: entries.length,
  });
}

function hashBindingRequirementShape(hash, shape, label) {
  const dims = shape === undefined || shape === null ? [] : shape;
  if (!Array.isArray(dims)) throw new Error(`${label} shape must be an array`);
  let next = hashU32(hash, dims.length);
  for (const dim of dims) next = hashNonNegativeInteger(next, dim, `${label} shape dimension`);
  return next;
}

function hashBindingRequirementList(hash, section, entries) {
  let next = hashString(hash, section);
  next = hashU32(next, entries.length);
  for (const entry of entries) {
    const role = entry.role ?? "";
    next = hashString(next, role);
    next = hashString(next, entry.dtype ?? "f32");
    next = hashNonNegativeInteger(next, entry.elementCount ?? 0, `${role || section} elementCount`);
    next = hashNonNegativeInteger(next, entry.byteLength ?? 0, `${role || section} byteLength`);
    next = hashU32(next, externalResourceAccessFlags(entry.access ?? "readwrite"));
    next = hashBindingRequirementShape(next, entry.shape ?? [], role || section);
  }
  return next;
}

function describeProgramBindingRequirements(requirements) {
  const persistent = requirements.persistent ?? [];
  const stepInputs = requirements.stepInputs ?? [];
  const stepOutputs = requirements.stepOutputs ?? [];
  let hash = hashString(fnv64Offset, "zgml-wasm-program-binding-requirements-v1");
  hash = hashBindingRequirementList(hash, "persistent", persistent);
  hash = hashBindingRequirementList(hash, "step-input", stepInputs);
  hash = hashBindingRequirementList(hash, "step-output", stepOutputs);
  return Object.freeze({
    bindingRequirementHash: hash,
    persistentRequirementCount: persistent.length,
    stepInputRequirementCount: stepInputs.length,
    stepOutputRequirementCount: stepOutputs.length,
  });
}

function modelResourceManifestEntry(resource) {
  const role = normalizeResourceTableRole(resource?.role);
  const dtype = normalizeLlamaModelResourceDtype(resource?.dtype ?? "f32");
  const shape = normalizeLlamaModelResourceShape(resource?.shape)?.shape ?? [];
  const elementCount = explicitLlamaModelResourceElementCount(resource) ?? (shape.length === 0 ? undefined : normalizeLlamaModelResourceShape(shape).elementCount);
  const byteLength = resource?.byteLength ?? resource?.descriptor?.byteLength;
  if (!Number.isSafeInteger(byteLength) || byteLength <= 0) {
    throw new Error(`invalid LLaMA model resource manifest byteLength for ${role}: ${byteLength}`);
  }
  if (!Number.isSafeInteger(elementCount) || elementCount <= 0) {
    throw new Error(`invalid LLaMA model resource manifest elementCount for ${role}: ${elementCount}`);
  }
  return Object.freeze({
    byteLength,
    dataByteLength: resource?.dataByteLength,
    dataByteOffset: resource?.dataByteOffset,
    dtype,
    elementCount,
    role,
    safetensorsShardIndex: resource?.safetensorsShardIndex,
    safetensorsShardName: resource?.safetensorsShardName,
    shape: Object.freeze(shape),
  });
}

function hashOptionalNonNegativeInteger(hash, value, label) {
  if (value === undefined || value === null) return hashU32(hash, 0);
  return hashNonNegativeInteger(hashU32(hash, 1), value, label);
}

export function describeLlamaModelResourceManifest(resources) {
  if (!Array.isArray(resources)) {
    throw new Error("LLaMA model resource manifest expects an array");
  }
  const entries = resources.map(modelResourceManifestEntry);
  let hash = hashU32(fnv64Offset, entries.length);
  for (const entry of entries) {
    hash = hashString(hash, entry.role);
    hash = hashString(hash, entry.dtype);
    hash = hashNonNegativeInteger(hash, entry.byteLength, `${entry.role} byteLength`);
    hash = hashNonNegativeInteger(hash, entry.elementCount, `${entry.role} elementCount`);
    hash = hashOptionalNonNegativeInteger(hash, entry.dataByteOffset, `${entry.role} dataByteOffset`);
    hash = hashOptionalNonNegativeInteger(hash, entry.dataByteLength, `${entry.role} dataByteLength`);
    hash = hashOptionalNonNegativeInteger(hash, entry.safetensorsShardIndex, `${entry.role} safetensorsShardIndex`);
    hash = hashString(hash, entry.safetensorsShardName ?? "");
    hash = hashU32(hash, entry.shape.length);
    for (const dim of entry.shape) {
      hash = hashNonNegativeInteger(hash, dim, `${entry.role} shape dimension`);
    }
  }
  return Object.freeze({
    entries: Object.freeze(entries),
    hash,
    resourceCount: entries.length,
  });
}

function normalizeSource(source) {
  const factory = readField(source, [wasmWebgpuInterop.importSource]);
  if (typeof factory === "function") return factory.call(source);
  return source;
}

function resourceWithView(resourceOrDescriptor) {
  const descriptor = resourceOrDescriptor && typeof resourceOrDescriptor === "object" && "resource" in resourceOrDescriptor
    ? resourceOrDescriptor
    : null;
  return {
    byteLength: descriptor?.byteLength,
    byteOffset: descriptor?.byteOffset ?? 0,
    resource: descriptor ? descriptor.resource : resourceOrDescriptor,
  };
}

function gpuBufferResource(device, resourceOrDescriptor, label) {
  const { resource } = resourceWithView(resourceOrDescriptor);
  if (
    !device ||
    typeof device !== "object" ||
    typeof device.createBuffer !== "function" ||
    !resource ||
    typeof resource !== "object" ||
    typeof resource.destroy !== "function" ||
    resource[mockStorage] instanceof ArrayBuffer
  ) {
    throw new Error(`${label} is not a browser GPUBuffer`);
  }
  assertLiveWasmWebGpuResource(resource, label);
  return resource;
}

function gpuBufferView(device, resourceOrDescriptor, label) {
  const resolved = resourceWithView(resourceOrDescriptor);
  const resource = gpuBufferResource(device, resourceOrDescriptor, label);
  const byteOffset = resolved.byteOffset;
  const byteLength = resolved.byteLength ?? resource.size;
  assertRange(byteOffset, byteLength);
  if (Number.isSafeInteger(resource.size) && byteOffset + byteLength > resource.size) {
    throw new Error(`${label} view is out of range`);
  }
  return { byteLength, byteOffset, resource };
}

function assertViewRange(resolved, byteLength, label, byteOffset = 0) {
  const available = resolved.byteLength ?? resolved.resource?.size;
  if (Number.isSafeInteger(available) && byteOffset + byteLength > available) {
    throw new Error(`${label} view is out of range`);
  }
}

function mockFloat32View(resourceOrDescriptor, length, label, byteOffset = 0) {
  const resolved = resourceWithView(resourceOrDescriptor);
  const resource = resolved.resource;
  const offset = resolved.byteOffset + byteOffset;
  const storage = resource?.[mockStorage] ?? gpuShadowStorage.get(resource);
  if (!resource || typeof resource !== "object" || !(storage instanceof ArrayBuffer)) {
    throw new Error(`${label} is not a mock/shadowed WebGPU resource`);
  }
  assertLiveWasmWebGpuResource(resource, label);
  if (offset % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error(`${label} byteOffset must be Float32-aligned: ${offset}`);
  }
  const byteLength = length * Float32Array.BYTES_PER_ELEMENT;
  assertViewRange(resolved, byteLength, label, byteOffset);
  if (offset + byteLength > storage.byteLength) {
    throw new Error(`${label} view is out of range`);
  }
  return new Float32Array(storage, offset, length);
}

function mockByteView(resourceOrDescriptor, byteLength, label, byteOffset = 0) {
  const resolved = resourceWithView(resourceOrDescriptor);
  const resource = resolved.resource;
  const offset = resolved.byteOffset + byteOffset;
  const storage = resource?.[mockStorage] ?? gpuShadowStorage.get(resource);
  if (!resource || typeof resource !== "object" || !(storage instanceof ArrayBuffer)) {
    throw new Error(`${label} is not a mock/shadowed WebGPU resource`);
  }
  assertLiveWasmWebGpuResource(resource, label);
  assertViewRange(resolved, byteLength, label, byteOffset);
  if (offset + byteLength > storage.byteLength) {
    throw new Error(`${label} view is out of range`);
  }
  return new Uint8Array(storage, offset, byteLength);
}

function mockReadFloat32(resourceOrDescriptor, length, label, byteOffset = 0) {
  const view = mockFloat32View(resourceOrDescriptor, length, label, byteOffset);
  const out = new Float32Array(length);
  out.set(view);
  return out;
}

function mockReadBytes(resourceOrDescriptor, byteLength, label, byteOffset = 0) {
  const view = mockByteView(resourceOrDescriptor, byteLength, label, byteOffset);
  const out = new Uint8Array(byteLength);
  out.set(view);
  return out;
}

function gpuMapReadFlag() {
  return typeof globalThis.GPUMapMode === "object" && Number.isInteger(globalThis.GPUMapMode.READ)
    ? globalThis.GPUMapMode.READ
    : 1;
}

function gpuBufferUsageFlags() {
  const bits = webgpuUsageBits();
  return {
    copySrc: bits.COPY_SRC,
    copyDst: bits.COPY_DST,
    storage: bits.STORAGE,
    uniform: typeof globalThis.GPUBufferUsage === "object" ? globalThis.GPUBufferUsage.UNIFORM ?? webgpuBufferUsage.UNIFORM : webgpuBufferUsage.UNIFORM,
    mapRead: typeof globalThis.GPUBufferUsage === "object" ? globalThis.GPUBufferUsage.MAP_READ ?? webgpuBufferUsage.MAP_READ : webgpuBufferUsage.MAP_READ,
  };
}

function growGpuByteCapacity(byteLength) {
  if (!Number.isSafeInteger(byteLength) || byteLength <= 0) {
    throw new Error(`invalid WebGPU buffer byteLength: ${byteLength}`);
  }
  let capacity = 4;
  while (capacity < byteLength) {
    if (capacity > Math.floor(Number.MAX_SAFE_INTEGER / 2)) return byteLength;
    capacity *= 2;
  }
  return capacity;
}

function storageBufferBinding(device, resourceOrDescriptor, label, byteLength) {
  const view = storageBufferView(resourceOrDescriptor, label, byteLength, storageBufferAlignment(device));
  const resource = gpuBufferResource(device, resourceOrDescriptor, label);
  return {
    buffer: resource,
    offset: view.byteOffset,
    size: view.byteLength,
  };
}

function storageBufferAlignment(deviceOrWrapper) {
  const alignment = deviceOrWrapper instanceof WasmWebGpuDevice
    ? deviceOrWrapper.storageBufferOffsetAlignment()
    : deviceOrWrapper?.limits?.minStorageBufferOffsetAlignment;
  return Number.isSafeInteger(alignment) && alignment > 0 ? alignment : 256;
}

function storageBufferView(resourceOrDescriptor, label, byteLength, alignment) {
  const resolved = resourceWithView(resourceOrDescriptor);
  const resource = resolved.resource;
  if (!resource || typeof resource !== "object") {
    throw new Error(`${label} is not a WebGPU storage resource`);
  }
  assertLiveWasmWebGpuResource(resource, label);
  const available = resolved.byteLength ?? resource.size;
  const size = byteLength ?? available;
  assertRange(resolved.byteOffset, size);
  if (Number.isSafeInteger(available) && size > available) {
    throw new Error(`${label} binding is out of range`);
  }
  if (Number.isSafeInteger(resource.size) && resolved.byteOffset + size > resource.size) {
    throw new Error(`${label} binding is out of range`);
  }
  if (resolved.byteOffset % alignment !== 0) {
    throw new Error(`${label} byteOffset ${resolved.byteOffset} does not satisfy WebGPU storage-buffer alignment ${alignment}`);
  }
  return {
    byteLength: size,
    byteOffset: resolved.byteOffset,
    resource,
  };
}

function storageResourceDescriptor(device, resource, options, label, byteLength) {
  const descriptor = device.descriptor(resource, {
    ...options,
    byteLength,
  });
  storageBufferView(descriptor, label, byteLength, storageBufferAlignment(device));
  return descriptor;
}

function validateResourceAccess(descriptor, requiredAccess, label) {
  const required = externalResourceAccessFlags(requiredAccess);
  const actual = externalResourceAccessFlags(descriptor.access ?? "readwrite");
  if ((required & 1) !== 0 && (actual & 1) === 0) {
    throw new Error(`${label} needs readable resource access`);
  }
  if ((required & 2) !== 0 && (actual & 2) === 0) {
    throw new Error(`${label} needs writable resource access`);
  }
}

function validateStorageResourceBinding(device, binding, label, byteLength, requiredAccess = "readwrite") {
  storageBufferView(binding.descriptor, label, byteLength, storageBufferAlignment(device));
  validateResourceAccess(binding.descriptor, requiredAccess, label);
  const placement = externalResourcePlacementId(binding.descriptor.placement ?? "webgpu");
  if (placement === externalResourcePlacementId("webgpu")) {
    const descriptorDevice = assertDeviceHandle(binding.descriptor.deviceHandle);
    if (descriptorDevice === 0) {
      throw new Error(`${label} needs WebGPU device provenance`);
    }
    if (descriptorDevice !== device.deviceHandle) {
      throw new Error(`${label} device mismatch: ${descriptorDevice} != ${device.deviceHandle}`);
    }
  }
  return Object.freeze({
    ...binding,
    descriptor: snapshotHostResourceDescriptor(binding.descriptor),
  });
}

function assertHandle(handle) {
  if (!Number.isSafeInteger(handle) || handle <= 0 || handle > 0xffffffff) {
    throw new Error(`invalid host resource handle: ${handle}`);
  }
}

function assertDeviceHandle(handle) {
  if (handle === undefined || handle === null) return 0;
  assertHandle(handle);
  return handle;
}

function allocateWasmWebGpuDeviceHandle() {
  const handle = nextWasmWebGpuDeviceHandle;
  nextWasmWebGpuDeviceHandle += 1;
  assertHandle(nextWasmWebGpuDeviceHandle);
  return handle;
}

function assertU32(value, label) {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffffffff) {
    throw new Error(`invalid ${label}: ${value}`);
  }
}

function assertRange(byteOffset, byteLength) {
  if (!Number.isSafeInteger(byteOffset) || byteOffset < 0) {
    throw new Error(`invalid host resource byteOffset: ${byteOffset}`);
  }
  if (!Number.isSafeInteger(byteLength) || byteLength <= 0) {
    throw new Error(`invalid host resource byteLength: ${byteLength}`);
  }
}

function hostResourceBackingByteLength(resource) {
  if (!resource || typeof resource !== "object") return undefined;
  const storage = resource[mockStorage] ?? gpuShadowStorage.get(resource);
  if (storage instanceof ArrayBuffer) return storage.byteLength;
  const size = readNumber(resource, ["size"]);
  if (size !== undefined) return size;
  return readNumber(resource, [wasmWebgpuInterop.byteLength, "byteLength", "byteLen"]);
}

function validateHostResourceViewRange(resource, byteOffset, byteLength, label) {
  if (resource === undefined || resource === null) return;
  assertLiveWasmWebGpuResource(resource, label);
  const backingByteLength = hostResourceBackingByteLength(resource);
  if (backingByteLength === undefined) return;
  assertRange(0, backingByteLength);
  if (byteOffset + byteLength > backingByteLength) {
    throw new Error(`${label} view is out of range`);
  }
}

export function writeExternalResourceDesc(view, ptr, descriptor, options = {}) {
  if (!(view instanceof DataView)) throw new Error("external resource descriptor needs a DataView");
  const placement = externalResourcePlacementId(options.placement ?? descriptor.placement ?? "webgpu");
  const access = externalResourceAccessFlags(options.access ?? descriptor.access ?? "readwrite");
  const handle = options.handle ?? descriptor.handle;
  const byteOffset = options.byteOffset ?? descriptor.byteOffset ?? 0;
  const byteLength = options.byteLength ?? options.byteLen ?? descriptor.byteLength ?? descriptor.byteLen;
  assertHandle(handle);
  assertRange(byteOffset, byteLength);
  assertU32(byteOffset, "external resource byteOffset");
  assertU32(byteLength, "external resource byteLength");
  view.setUint32(ptr, placement, true);
  view.setUint32(ptr + 4, access, true);
  view.setUint32(ptr + 8, handle, true);
  view.setUint32(ptr + 12, byteOffset, true);
  view.setUint32(ptr + 16, byteLength, true);
}

function readOption(value) {
  return typeof value === "function" ? value() : value;
}

export class WasmExternalResourceBridge {
  constructor(options = {}) {
    this.exports = options.exports;
    this.alloc = options.alloc;
    this.keep = options.keep ?? ((ptr) => ptr);
    this.view = options.view;
    this.zero = options.zero ?? null;
    this.check = options.check ?? ((code) => {
      if (code !== 0) throw new Error(`zgml status ${code}`);
    });
    this.externalResourceDescSize = options.externalResourceDescSize;
    this.handleSize = options.handleSize ?? 4;
    this.trackBuffer = options.trackBuffer ?? (() => {});
  }

  wrapResourceBuffer(descriptor, options = {}) {
    const exportsRef = readOption(this.exports);
    const alloc = this.alloc;
    const view = this.view;
    const descSize = readOption(this.externalResourceDescSize);
    const handleSize = readOption(this.handleSize);
    if (!exportsRef || typeof exportsRef.zgml_buffer_wrap_resource !== "function") {
      throw new Error("zgml_buffer_wrap_resource export is unavailable");
    }
    if (typeof alloc !== "function") throw new Error("Wasm resource bridge needs an alloc function");
    if (typeof view !== "function") throw new Error("Wasm resource bridge needs a view function");
    assertU32(descSize, "external resource descriptor size");
    assertU32(handleSize, "Wasm handle size");
    const desc = this.keep(alloc(descSize), descSize);
    if (this.zero) {
      this.zero(desc, descSize);
    } else {
      new Uint8Array(view().buffer, desc, descSize).fill(0);
    }
    writeExternalResourceDesc(view(), desc, descriptor, options);
    const out = this.keep(alloc(handleSize), handleSize);
    this.check(exportsRef.zgml_buffer_wrap_resource(desc, out));
    const handle = view().getUint32(out, true);
    if (handle === 0) throw new Error("resource buffer handle was not returned");
    this.trackBuffer(handle);
    return handle;
  }

  wrapHostResourceBuffer(device, resource, options = {}) {
    const descriptor = device.descriptor(resource, options);
    if (
      options.validateStorageView !== false &&
      externalResourcePlacementId(descriptor.placement) === externalResourcePlacementId("webgpu")
    ) {
      storageBufferView(
        descriptor,
        options.label ?? "WebGPU host resource",
        descriptor.byteLength,
        storageBufferAlignment(device),
      );
    }
    const buffer = this.wrapResourceBuffer(descriptor);
    return { buffer, descriptor };
  }
}

function bufferHandle(value, label) {
  const handle = typeof value === "object" && value !== null ? value.buffer ?? value.handle : value;
  if (!Number.isSafeInteger(handle) || handle <= 0 || handle > 0xffffffff) {
    throw new Error(`invalid ${label} buffer handle: ${handle}`);
  }
  return handle;
}

function elementLength(value, explicit, label) {
  const len = explicit ?? (typeof value === "object" && value !== null ? value.len ?? value.length ?? value.elementCount : undefined);
  if (!Number.isSafeInteger(len) || len < 0 || len > 0xffffffff) {
    throw new Error(`invalid ${label} element length: ${len}`);
  }
  return len;
}

export class WasmSessionBindingBridge {
  constructor(options = {}) {
    this.exports = options.exports;
    this.alloc = options.alloc;
    this.keep = options.keep ?? ((ptr) => ptr);
    this.view = options.view;
    this.zero = options.zero ?? null;
	    this.check = options.check ?? ((code) => {
	      if (code !== 0) throw new Error(`zgml status ${code}`);
	    });
	    this.bindDescSize = options.bindDescSize;
	    this.programModelCompatibilitySize = options.programModelCompatibilitySize;
	    this.programRequirementsSize = options.programRequirementsSize;
	    this.llamaKvCacheBindDescSize = options.llamaKvCacheBindDescSize;
	    this.llamaBufferBindDescSize = options.llamaBufferBindDescSize;
	    this.llamaKvCacheRequirementsSize = options.llamaKvCacheRequirementsSize;
	    this.handleSize = options.handleSize ?? 4;
  }

  allocStruct(sizeOption, label) {
    const alloc = this.alloc;
    const view = this.view;
    const size = readOption(sizeOption);
    if (typeof alloc !== "function") throw new Error("Wasm session bridge needs an alloc function");
    if (typeof view !== "function") throw new Error("Wasm session bridge needs a view function");
    assertU32(size, label);
    const ptr = this.keep(alloc(size), size);
    if (this.zero) {
      this.zero(ptr, size);
    } else {
      new Uint8Array(view().buffer, ptr, size).fill(0);
    }
    return ptr;
  }

  allocHandleOut() {
    return this.allocStruct(this.handleSize, "Wasm handle size");
  }

  writeU32(ptr, offset, value, label) {
    assertU32(value, label);
    this.view().setUint32(ptr + offset, value, true);
  }

  readHandle(ptr) {
    return this.view().getUint32(ptr, true);
  }

  readUsize(ptr, offset, byteLength, label) {
    if (byteLength === 4) return this.view().getUint32(ptr + offset, true);
    if (byteLength === 8) {
      const value = this.view().getBigUint64(ptr + offset, true);
      if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
        throw new Error(`${label} is too large for JavaScript: ${value}`);
      }
      return Number(value);
    }
    throw new Error(`unsupported ${label} ABI size: ${byteLength}`);
  }

	  programRequirements(program) {
	    const exportsRef = readOption(this.exports);
	    const fn = exportsRef?.zgml_program_get_requirements;
	    if (typeof fn !== "function") throw new Error("Program requirements export is unavailable");
    const size = readOption(this.programRequirementsSize);
    const ptr = this.allocStruct(size, "Program requirements size");
    this.check(fn(program, ptr));
    const usizeBytes = size >= 128 ? 8 : 4;
    const field = (index, label) => this.readUsize(ptr, 16 + index * usizeBytes, usizeBytes, label);
    return {
      modelKind: this.view().getUint32(ptr, true),
      scalarBytes: this.view().getUint32(ptr + 4, true),
      tokenIdBytes: this.view().getUint32(ptr + 8, true),
      inputLen: field(0, "Program input length"),
      inputByteLength: field(1, "Program input byte length"),
      outputLen: field(2, "Program output length"),
      weightsLen: field(3, "Program weights length"),
      weightsByteLength: field(4, "Program weights byte length"),
      biasLen: field(5, "Program bias length"),
      biasByteLength: field(6, "Program bias byte length"),
      parameterLen: field(7, "Program parameter length"),
      parameterByteLength: field(8, "Program parameter byte length"),
      logitsLen: field(9, "Program logits length"),
      outputByteLength: field(10, "Program output byte length"),
      contextLength: field(11, "Program context length"),
      batch: field(12, "Program batch"),
      maxTokenWindow: field(13, "Program max token window"),
    };
  }

  programModelCompatibility(program, model) {
    const exportsRef = readOption(this.exports);
    const fn = exportsRef?.zgml_program_check_model_compatibility;
    if (typeof fn !== "function") throw new Error("Program/model compatibility export is unavailable");
    const ptr = this.allocStruct(this.programModelCompatibilitySize, "Program/model compatibility size");
    this.check(fn(program, model, ptr));
    const size = readOption(this.programModelCompatibilitySize);
    const compatible = size >= 16
      ? this.view().getBigUint64(ptr + 8, true) !== 0n
      : this.view().getUint32(ptr + 8, true) !== 0;
    return {
      programModelKind: this.view().getUint32(ptr, true),
      modelKind: this.view().getUint32(ptr + 4, true),
      compatible,
    };
  }

	  llamaKvCacheRequirements(program) {
	    const exportsRef = readOption(this.exports);
	    const fn = exportsRef?.zgml_llama_program_get_kv_cache_requirements;
    if (typeof fn !== "function") throw new Error("LLaMA KV cache requirements export is unavailable");
    const size = readOption(this.llamaKvCacheRequirementsSize);
    const ptr = this.allocStruct(size, "LLaMA KV cache requirements size");
    this.check(fn(program, ptr));
    const usizeBytes = size >= 48 ? 8 : 4;
    const contextOffset = 16;
    const requirements = normalizeLlamaKvRequirements({
      layers: this.view().getUint32(ptr + 8, true),
      kBufferByteLength: this.readUsize(ptr, contextOffset + usizeBytes, usizeBytes, "LLaMA K cache byte length"),
      vBufferByteLength: this.readUsize(ptr, contextOffset + usizeBytes * 2, usizeBytes, "LLaMA V cache byte length"),
    });
    return {
      ...requirements,
      contextLength: this.readUsize(ptr, contextOffset, usizeBytes, "LLaMA context length"),
      modelKind: this.view().getUint32(ptr, true),
      scalarBytes: this.view().getUint32(ptr + 4, true),
    };
  }

  bufferBindDesc(bindings = {}) {
    const desc = this.allocStruct(this.bindDescSize, "buffer bind descriptor size");
    const roles = [
      ["weights", 0],
      ["bias", 8],
      ["input", 16],
      ["output", 24],
    ];
    for (const [role, offset] of roles) {
      const value = bindings[role];
      if (value === undefined || value === null) continue;
      this.writeU32(desc, offset, bufferHandle(value, role), `${role} buffer handle`);
      this.writeU32(desc, offset + 4, elementLength(value, bindings[`${role}Len`], role), `${role} element length`);
    }
    return desc;
  }

  llamaBufferBindDesc(bindings = {}) {
    const desc = this.allocStruct(this.llamaBufferBindDescSize, "LLaMA buffer bind descriptor size");
    if (bindings.output !== undefined && bindings.output !== null) {
      this.writeU32(desc, 0, bufferHandle(bindings.output, "LLaMA output"), "LLaMA output buffer handle");
      this.writeU32(desc, 4, elementLength(bindings.output, bindings.outputLen, "LLaMA output"), "LLaMA output element length");
    }
    const kvEntries = bindings.kvCache ?? bindings.kv ?? [];
    if (kvEntries.length > 0) {
      const handleSize = readOption(this.handleSize);
      assertU32(handleSize, "Wasm handle size");
      const kHandles = this.allocStruct(kvEntries.length * handleSize, "LLaMA K cache handle array size");
      const vHandles = this.allocStruct(kvEntries.length * handleSize, "LLaMA V cache handle array size");
      for (let i = 0; i < kvEntries.length; i += 1) {
        const entry = kvEntries[i];
        this.writeU32(kHandles, i * handleSize, bufferHandle(entry.k, `LLaMA K cache ${i}`), `LLaMA K cache ${i} buffer handle`);
        this.writeU32(vHandles, i * handleSize, bufferHandle(entry.v, `LLaMA V cache ${i}`), `LLaMA V cache ${i} buffer handle`);
      }
      const kvDesc = this.allocStruct(this.llamaKvCacheBindDescSize, "LLaMA KV cache bind descriptor size");
      this.writeU32(kvDesc, 0, kHandles, "LLaMA K cache handles pointer");
      this.writeU32(kvDesc, 4, vHandles, "LLaMA V cache handles pointer");
      this.writeU32(kvDesc, 8, kvEntries.length, "LLaMA KV cache layer count");
      this.writeU32(desc, 8, kvDesc, "LLaMA KV cache descriptor pointer");
    }
    return desc;
  }

  bindBufferSession(program, desc, options = {}) {
    const exportsRef = readOption(this.exports);
    const out = this.allocHandleOut();
    const model = options.model ?? 0;
    const fn = model !== 0 ? exportsRef.zgml_session_bind_model_buffers : exportsRef.zgml_session_bind_buffers;
    if (typeof fn !== "function") throw new Error("buffer Session bind export is unavailable");
    this.check(model !== 0 ? fn(program, model, desc, out) : fn(program, desc, out));
    const session = this.readHandle(out);
    if (session === 0) throw new Error("buffer-backed session handle was not returned");
    return session;
  }

  bindLlamaBufferSession(program, desc, options = {}) {
    const exportsRef = readOption(this.exports);
    const out = this.allocHandleOut();
    const model = options.model ?? 0;
    const fn = model !== 0 ? exportsRef.zgml_llama_session_bind_model_buffers : exportsRef.zgml_llama_session_bind_buffers;
    if (typeof fn !== "function") throw new Error("LLaMA buffer Session bind export is unavailable");
    this.check(model !== 0 ? fn(program, model, desc, out) : fn(program, desc, out));
    const session = this.readHandle(out);
    if (session === 0) throw new Error("LLaMA buffer-backed session handle was not returned");
    return session;
  }
}

export class HostResourceRegistry {
  constructor(options = {}) {
    this.nextHandle = options.firstHandle ?? 0x1000;
    assertHandle(this.nextHandle);
    this.resourceToHandle = new WeakMap();
    this.handleToResource = new Map();
  }

  descriptor(source, options = {}) {
    const normalized = normalizeSource(source);
    if (!normalized || typeof normalized !== "object") {
      throw new Error("host WebGPU resource must be an object");
    }
    assertLiveWasmWebGpuResource(normalized, "host WebGPU resource");
    const byteOffset = options.byteOffset ?? readNumber(normalized, [wasmWebgpuInterop.byteOffset, "byteOffset"]) ?? 0;
    const byteLength =
      options.byteLength ??
      options.byteLen ??
      readNumber(normalized, [wasmWebgpuInterop.byteLength, "byteLength", "byteLen", "size"]);
    assertRange(byteOffset, byteLength);
    validateHostResourceViewRange(normalized, byteOffset, byteLength, "host resource descriptor");
    const placement = options.placement ?? readString(normalized, [wasmWebgpuInterop.placement, "placement", "backend", "device"]) ?? "webgpu";
    const placementId = externalResourcePlacementId(placement);
    const access = options.access ?? readField(normalized, [wasmWebgpuInterop.access, "access"]) ?? "readwrite";
    validateWebGpuResourceUsage(normalized, access, { ...options, placement });
    const annotatedDeviceHandle = readNumber(normalized, [wasmWebgpuInterop.deviceHandle, "deviceHandle"]);
    const registeredDeviceHandle = hostResourceDeviceHandles.get(normalized);
    if (
      annotatedDeviceHandle !== undefined &&
      registeredDeviceHandle !== undefined &&
      annotatedDeviceHandle !== registeredDeviceHandle
    ) {
      throw new Error(`host resource device mismatch: ${annotatedDeviceHandle} != ${registeredDeviceHandle}`);
    }
    const sourceDeviceHandle = annotatedDeviceHandle ?? registeredDeviceHandle;
    const deviceHandle = assertDeviceHandle(options.deviceHandle ?? sourceDeviceHandle);
    if (
      placementId === externalResourcePlacementId("webgpu") &&
      deviceHandle !== 0 &&
      sourceDeviceHandle === undefined &&
      options.requireDeviceProvenance !== false
    ) {
      throw new Error("host WebGPU resource needs same-device provenance; import it through WasmWebGpuDevice.importBuffer first");
    }
    if (sourceDeviceHandle !== undefined && deviceHandle !== sourceDeviceHandle) {
      throw new Error(`host resource device mismatch: ${sourceDeviceHandle} != ${deviceHandle}`);
    }
    const handle = this.handleFor(normalized);
    return Object.freeze({
      access,
      byteLength,
      byteOffset,
      deviceHandle,
      handle,
      placement,
      resource: normalized,
    });
  }

  handleFor(resource) {
    if (!resource || typeof resource !== "object") {
      throw new Error("host WebGPU resource must be an object");
    }
    assertLiveWasmWebGpuResource(resource, "host WebGPU resource");
    const existing = this.resourceToHandle.get(resource);
    if (existing !== undefined) return existing;
    const requested = readNumber(resource, [wasmWebgpuInterop.handle, "resourceHandle", "handle"]);
    const handle = requested ?? this.nextHandle;
    assertHandle(handle);
    const previousResource = this.handleToResource.get(handle);
    if (previousResource !== undefined && previousResource !== resource) {
      throw new Error(`host resource handle collision: ${handle}`);
    }
    this.resourceToHandle.set(resource, handle);
    this.handleToResource.set(handle, resource);
    if (handle >= this.nextHandle) this.nextHandle = handle + 1;
    assertHandle(this.nextHandle);
    return handle;
  }

	  resourceFor(handle) {
	    assertHandle(handle);
	    return this.handleToResource.get(handle) ?? null;
	  }

	  unregister(resourceOrHandle) {
	    if (typeof resourceOrHandle === "number") {
	      assertHandle(resourceOrHandle);
	      const resource = this.handleToResource.get(resourceOrHandle);
	      if (resource !== undefined) this.resourceToHandle.delete(resource);
	      return this.handleToResource.delete(resourceOrHandle);
	    }
	    if (!resourceOrHandle || typeof resourceOrHandle !== "object") return false;
	    const handle = this.resourceToHandle.get(resourceOrHandle);
	    const removedResource = this.resourceToHandle.delete(resourceOrHandle);
	    if (handle !== undefined) {
	      const mapped = this.handleToResource.get(handle);
	      if (mapped === resourceOrHandle) return this.handleToResource.delete(handle) || removedResource;
	    }
	    return removedResource;
	  }

	  has(resourceOrHandle) {
	    if (typeof resourceOrHandle === "number") return this.handleToResource.has(resourceOrHandle);
	    if (!resourceOrHandle || typeof resourceOrHandle !== "object") return false;
    return this.resourceToHandle.has(resourceOrHandle);
  }

  clear() {
    this.handleToResource.clear();
    this.resourceToHandle = new WeakMap();
  }
}

export class WasmWebGpuDevice {
  constructor(options = {}) {
    this.device = options.device ?? null;
    this.deviceHandle = assertDeviceHandle(options.deviceHandle ?? options.handle) || allocateWasmWebGpuDeviceHandle();
    this.placement = options.placement ?? "webgpu";
    this.registry = options.registry ?? new HostResourceRegistry();
    this.ownedResources = [];
    this.readbackStaging = null;
  }

  get mode() {
    return this.canCreateGpuBuffers() ? "gpu-buffer" : "mock";
  }

  canCreateGpuBuffers() {
    return (
      this.device !== null &&
      typeof this.device === "object" &&
      typeof this.device.createBuffer === "function" &&
      typeof globalThis.GPUBufferUsage === "object"
    );
  }

  storageBufferOffsetAlignment() {
    const alignment = this.device?.limits?.minStorageBufferOffsetAlignment;
    return Number.isSafeInteger(alignment) && alignment > 0 ? alignment : 256;
  }

  maxStorageBuffersPerShaderStage() {
    const limit = this.device?.limits?.maxStorageBuffersPerShaderStage;
    return Number.isSafeInteger(limit) && limit > 0 ? limit : 8;
  }

  canBindStorageBuffers(count) {
    if (!Number.isSafeInteger(count) || count < 0) {
      throw new Error(`invalid WebGPU storage-buffer binding count: ${count}`);
    }
    return this.canCreateGpuBuffers() && count <= this.maxStorageBuffersPerShaderStage();
  }

  createBuffer(options = {}) {
    const label = options.label ?? "zgml.wasm.webgpu.resource";
    const byteLength = options.byteLength ?? options.byteLen ?? options.size;
    assertRange(0, byteLength);
    const usage = options.usage ?? webgpuRequiredUsageFlags("readwrite");
    if (this.canCreateGpuBuffers()) {
      const buffer = this.device.createBuffer({
        label,
        size: byteLength,
        usage,
      });
      hostResourceDeviceHandles.set(buffer, this.deviceHandle);
      wasmWebGpuResourceDevices.set(buffer, this);
      gpuShadowStorage.set(buffer, new ArrayBuffer(byteLength));
      this.ownedResources.push(buffer);
      return buffer;
    }
    const buffer = {
      [wasmWebgpuInterop.deviceHandle]: this.deviceHandle,
      [mockStorage]: new ArrayBuffer(byteLength),
      label,
      deviceHandle: this.deviceHandle,
      destroyed: false,
      destroy() {
        this.destroyed = true;
      },
      mapState: "unmapped",
      size: byteLength,
      usage,
    };
    hostResourceDeviceHandles.set(buffer, this.deviceHandle);
    wasmWebGpuResourceDevices.set(buffer, this);
    this.ownedResources.push(buffer);
    return buffer;
  }

  importBuffer(resource, options = {}) {
    const normalized = normalizeSource(resource);
    if (!normalized || typeof normalized !== "object") {
      throw new Error("imported WebGPU resource must be an object");
    }
    assertLiveWasmWebGpuResource(normalized, options.label ?? "imported WebGPU resource");
    const annotatedDeviceHandle = readNumber(normalized, [wasmWebgpuInterop.deviceHandle, "deviceHandle"]);
    const registeredDeviceHandle = hostResourceDeviceHandles.get(normalized);
    const assertedDeviceHandle = assertDeviceHandle(options.deviceHandle ?? annotatedDeviceHandle ?? registeredDeviceHandle ?? this.deviceHandle);
    if (assertedDeviceHandle !== this.deviceHandle) {
      throw new Error(`imported WebGPU resource device mismatch: ${assertedDeviceHandle} != ${this.deviceHandle}`);
    }
    if (annotatedDeviceHandle !== undefined && annotatedDeviceHandle !== this.deviceHandle) {
      throw new Error(`imported WebGPU resource device mismatch: ${annotatedDeviceHandle} != ${this.deviceHandle}`);
    }
    if (registeredDeviceHandle !== undefined && registeredDeviceHandle !== this.deviceHandle) {
      throw new Error(`imported WebGPU resource device mismatch: ${registeredDeviceHandle} != ${this.deviceHandle}`);
    }
    const byteOffset = options.byteOffset ?? readNumber(normalized, [wasmWebgpuInterop.byteOffset, "byteOffset"]) ?? 0;
    const byteLength =
      options.byteLength ??
      options.byteLen ??
      readNumber(normalized, [wasmWebgpuInterop.byteLength, "byteLength", "byteLen", "size"]);
    assertRange(byteOffset, byteLength);
    validateHostResourceViewRange(normalized, byteOffset, byteLength, "imported WebGPU resource");
    validateWebGpuResourceUsage(normalized, options.access ?? readField(normalized, [wasmWebgpuInterop.access, "access"]) ?? "readwrite", {
      ...options,
      placement: this.placement,
    });
    if (this.canCreateGpuBuffers()) {
      gpuBufferResource(this.device, normalized, options.label ?? "imported WebGPU resource");
    } else if (!(normalized[mockStorage] instanceof ArrayBuffer) && !(gpuShadowStorage.get(normalized) instanceof ArrayBuffer)) {
      gpuShadowStorage.set(normalized, new ArrayBuffer(byteLength));
    }
    hostResourceDeviceHandles.set(normalized, this.deviceHandle);
    wasmWebGpuResourceDevices.set(normalized, this);
    return normalized;
  }

  descriptor(source, options = {}) {
    return this.registry.descriptor(source, {
      deviceHandle: this.deviceHandle,
      placement: this.placement,
      ...options,
    });
  }

  resourceTable(descriptors, options = {}) {
    return describeHostResourceTable(descriptors, options);
  }

  resourceFor(handle) {
    return this.registry.resourceFor(handle);
  }

  writeFloat32(resourceOrDescriptor, values, byteOffset = 0, elementLength = null) {
    const data = values instanceof Float32Array ? values : new Float32Array(values);
    const length = elementLength === null ? data.length : elementLength;
    if (!Number.isSafeInteger(length) || length < 0 || length > data.length) {
      throw new Error(`invalid WebGPU Float32 write length: ${length}`);
    }
    const byteLength = length * Float32Array.BYTES_PER_ELEMENT;
    if (!this.canCreateGpuBuffers()) {
      const target = mockFloat32View(resourceOrDescriptor, length, "mock WebGPU write target", byteOffset);
      for (let i = 0; i < length; i += 1) target[i] = data[i];
      return;
    }
    const view = gpuBufferView(this.device, resourceOrDescriptor, "WebGPU write target");
    assertViewRange(view, byteLength, "WebGPU write target", byteOffset);
    this.device.queue.writeBuffer(view.resource, view.byteOffset + byteOffset, data.buffer, data.byteOffset, byteLength);
    const shadow = gpuShadowStorage.get(view.resource);
    if (shadow instanceof ArrayBuffer) {
      new Uint8Array(shadow, view.byteOffset + byteOffset, byteLength).set(new Uint8Array(data.buffer, data.byteOffset, byteLength));
    }
  }

  writeBytes(resourceOrDescriptor, values, byteOffset = 0) {
    const data = bytesView(values, "WebGPU byte write data");
    if (!this.canCreateGpuBuffers()) {
      mockByteView(resourceOrDescriptor, data.byteLength, "mock WebGPU byte write target", byteOffset).set(data);
      return;
    }
    const view = gpuBufferView(this.device, resourceOrDescriptor, "WebGPU byte write target");
    assertViewRange(view, data.byteLength, "WebGPU byte write target", byteOffset);
    this.device.queue.writeBuffer(view.resource, view.byteOffset + byteOffset, data.buffer, data.byteOffset, data.byteLength);
    const shadow = gpuShadowStorage.get(view.resource);
    if (shadow instanceof ArrayBuffer) {
      new Uint8Array(shadow, view.byteOffset + byteOffset, data.byteLength).set(data);
    }
  }

  readFloat32Sync(resourceOrDescriptor, length, byteOffset = 0) {
    if (this.canCreateGpuBuffers()) {
      throw new Error("synchronous WebGPU readback is only available for mock resources");
    }
    return mockReadFloat32(resourceOrDescriptor, length, "mock WebGPU read source", byteOffset);
  }

  readFloat32IntoSync(resourceOrDescriptor, target, length = target?.length, byteOffset = 0) {
    if (this.canCreateGpuBuffers()) {
      throw new Error("synchronous WebGPU readback is only available for mock resources");
    }
    if (!(target instanceof Float32Array)) {
      throw new Error("mock WebGPU read target must be a Float32Array");
    }
    if (!Number.isSafeInteger(length) || length < 0 || length > target.length) {
      throw new Error(`invalid mock WebGPU read target length: ${length}`);
    }
    target.set(mockFloat32View(resourceOrDescriptor, length, "mock WebGPU read source", byteOffset).subarray(0, length));
    return target;
  }

  readBytesSync(resourceOrDescriptor, byteLength, byteOffset = 0) {
    if (this.canCreateGpuBuffers()) {
      throw new Error("synchronous WebGPU readback is only available for mock resources");
    }
    return mockReadBytes(resourceOrDescriptor, byteLength, "mock WebGPU byte read source", byteOffset);
  }

  readBytesIntoSync(resourceOrDescriptor, target, byteLength = target?.byteLength, byteOffset = 0) {
    if (this.canCreateGpuBuffers()) {
      throw new Error("synchronous WebGPU readback is only available for mock resources");
    }
    if (!(target instanceof Uint8Array)) {
      throw new Error("mock WebGPU byte read target must be a Uint8Array");
    }
    if (!Number.isSafeInteger(byteLength) || byteLength < 0 || byteLength > target.byteLength) {
      throw new Error(`invalid mock WebGPU byte read target length: ${byteLength}`);
    }
    if (byteLength === 0) return target;
    target.set(mockByteView(resourceOrDescriptor, byteLength, "mock WebGPU byte read source", byteOffset).subarray(0, byteLength));
    return target;
  }

  async readFloat32(resourceOrDescriptor, length, byteOffset = 0) {
    if (!this.canCreateGpuBuffers()) return this.readFloat32Sync(resourceOrDescriptor, length, byteOffset);
    const out = new Float32Array(length);
    return this.readFloat32Into(resourceOrDescriptor, out, length, byteOffset);
  }

  async readFloat32Into(resourceOrDescriptor, target, length = target?.length, byteOffset = 0) {
    if (!(target instanceof Float32Array)) {
      throw new Error("WebGPU read target must be a Float32Array");
    }
    if (!Number.isSafeInteger(length) || length < 0 || length > target.length) {
      throw new Error(`invalid WebGPU read target length: ${length}`);
    }
    if (!this.canCreateGpuBuffers()) return this.readFloat32IntoSync(resourceOrDescriptor, target, length, byteOffset);
    if (length === 0) return target;
    const source = gpuBufferView(this.device, resourceOrDescriptor, "WebGPU read source");
    const byteLength = length * Float32Array.BYTES_PER_ELEMENT;
    assertViewRange(source, byteLength, "WebGPU read source", byteOffset);
    const staging = this.acquireReadbackStaging(byteLength, "zgml.wasm.webgpu.readback");
    const readback = staging.buffer;
    let mapped = false;
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source.resource, source.byteOffset + byteOffset, readback, 0, byteLength);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(gpuMapReadFlag());
      mapped = true;
      const mappedRange = readback.getMappedRange();
      target.set(new Float32Array(mappedRange, 0, length).subarray(0, length));
      const shadow = gpuShadowStorage.get(source.resource);
      if (shadow instanceof ArrayBuffer) {
        new Uint8Array(shadow, source.byteOffset + byteOffset, byteLength).set(new Uint8Array(target.buffer, target.byteOffset, byteLength));
      }
      return target;
    } finally {
      if (mapped) readback.unmap();
      this.releaseReadbackStaging(staging, !mapped);
    }
  }

  async readBytes(resourceOrDescriptor, byteLength, byteOffset = 0) {
    const out = new Uint8Array(byteLength);
    return this.readBytesInto(resourceOrDescriptor, out, byteLength, byteOffset);
  }

  async readBytesInto(resourceOrDescriptor, target, byteLength = target?.byteLength, byteOffset = 0) {
    if (!(target instanceof Uint8Array)) {
      throw new Error("WebGPU byte read target must be a Uint8Array");
    }
    if (!Number.isSafeInteger(byteLength) || byteLength < 0 || byteLength > target.byteLength) {
      throw new Error(`invalid WebGPU byte read target length: ${byteLength}`);
    }
    if (!this.canCreateGpuBuffers()) return this.readBytesIntoSync(resourceOrDescriptor, target, byteLength, byteOffset);
    if (byteLength === 0) return target;
    const source = gpuBufferView(this.device, resourceOrDescriptor, "WebGPU byte read source");
    assertViewRange(source, byteLength, "WebGPU byte read source", byteOffset);
    const staging = this.acquireReadbackStaging(byteLength, "zgml.wasm.webgpu.byte-readback");
    const readback = staging.buffer;
    let mapped = false;
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source.resource, source.byteOffset + byteOffset, readback, 0, byteLength);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(gpuMapReadFlag());
      mapped = true;
      const mappedRange = readback.getMappedRange();
      target.set(new Uint8Array(mappedRange, 0, byteLength));
      const shadow = gpuShadowStorage.get(source.resource);
      if (shadow instanceof ArrayBuffer) {
        new Uint8Array(shadow, source.byteOffset + byteOffset, byteLength).set(target.subarray(0, byteLength));
      }
      return target;
    } finally {
      if (mapped) readback.unmap();
      this.releaseReadbackStaging(staging, !mapped);
    }
  }

  acquireReadbackStaging(byteLength, label) {
    assertRange(0, byteLength);
    const usage = gpuBufferUsageFlags();
    const create = () => this.createBuffer({
      byteLength,
      label,
      usage: usage.copyDst | usage.mapRead,
    });
    if (this.readbackStaging?.inUse === true) {
      return {
        buffer: create(),
        cached: false,
      };
    }
    if (this.readbackStaging === null || this.readbackStaging.byteLength < byteLength) {
      destroyWasmWebGpuDeviceResource(this, this.readbackStaging?.buffer);
      this.readbackStaging = {
        buffer: create(),
        byteLength,
        inUse: false,
      };
    }
    this.readbackStaging.inUse = true;
    return {
      buffer: this.readbackStaging.buffer,
      cached: true,
    };
  }

  releaseReadbackStaging(staging, discard = false) {
    if (staging.cached) {
      if (this.readbackStaging?.buffer === staging.buffer) {
        if (discard) {
          destroyWasmWebGpuDeviceResource(this, this.readbackStaging.buffer);
          this.readbackStaging = null;
        } else {
          this.readbackStaging.inUse = false;
        }
      }
      return;
    }
    destroyWasmWebGpuDeviceResource(this, staging.buffer);
  }

  destroy() {
    destroyWasmWebGpuDeviceResource(this, this.readbackStaging?.buffer);
    this.readbackStaging = null;
    const resources = this.ownedResources.splice(0);
    for (const resource of resources) destroyWasmWebGpuDeviceResource(this, resource);
    this.ownedResources = [];
    this.registry.clear();
  }
}

export class WasmWebGpuTinyLinearExecutor {
  constructor(options = {}) {
    this.device = options.device;
    if (!(this.device instanceof WasmWebGpuDevice)) {
      throw new Error("WasmWebGpuTinyLinearExecutor needs a WasmWebGpuDevice");
    }
    this.pipeline = null;
    this.paramsBuffer = null;
  }

  ensureGpuDevice() {
    if (!this.device.canCreateGpuBuffers()) {
      throw new Error("browser WebGPU device is unavailable");
    }
    return this.device.device;
  }

  ensurePipeline() {
    const gpuDevice = this.ensureGpuDevice();
    if (this.pipeline) return this.pipeline;
    const shader = gpuDevice.createShaderModule({
      label: "zgml.wasm.webgpu.tiny-linear",
      code: `
struct Params {
  rows: u32,
  cols: u32,
};

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) {
    return;
  }
  var acc = bias[row];
  for (var col = 0u; col < params.cols; col = col + 1u) {
    acc = acc + weights[row * params.cols + col] * input[col];
  }
  output[row] = acc;
}
`,
    });
    this.pipeline = gpuDevice.createComputePipeline({
      label: "zgml.wasm.webgpu.tiny-linear",
      layout: "auto",
      compute: {
        module: shader,
        entryPoint: "main",
      },
    });
    const usage = gpuBufferUsageFlags();
    this.paramsBuffer = gpuDevice.createBuffer({
      label: "zgml.wasm.webgpu.tiny-linear.params",
      size: 8,
      usage: usage.uniform | usage.copyDst,
    });
    return this.pipeline;
  }

  submit(options = {}) {
    const rows = options.rows;
    const cols = options.cols;
    if (!Number.isSafeInteger(rows) || rows <= 0 || !Number.isSafeInteger(cols) || cols <= 0) {
      throw new Error(`invalid tiny-linear shape: rows=${rows} cols=${cols}`);
    }
    if (!this.device.canCreateGpuBuffers()) {
      return this.submitMock(options);
    }
    const gpuDevice = this.ensureGpuDevice();
    const pipeline = this.ensurePipeline();
    const weights = storageBufferBinding(gpuDevice, options.weights, "tiny-linear weights", rows * cols * Float32Array.BYTES_PER_ELEMENT);
    const bias = storageBufferBinding(gpuDevice, options.bias, "tiny-linear bias", rows * Float32Array.BYTES_PER_ELEMENT);
    const input = storageBufferBinding(gpuDevice, options.input, "tiny-linear input", cols * Float32Array.BYTES_PER_ELEMENT);
    const output = storageBufferBinding(gpuDevice, options.output, "tiny-linear output", rows * Float32Array.BYTES_PER_ELEMENT);
    const params = new Uint32Array([rows, cols]);
    gpuDevice.queue.writeBuffer(this.paramsBuffer, 0, params.buffer, params.byteOffset, params.byteLength);
    const bindGroup = gpuDevice.createBindGroup({
      label: "zgml.wasm.webgpu.tiny-linear",
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: weights },
        { binding: 1, resource: bias },
        { binding: 2, resource: input },
        { binding: 3, resource: output },
        { binding: 4, resource: { buffer: this.paramsBuffer } },
      ],
    });
    const encoder = gpuDevice.createCommandEncoder();
    const pass = encoder.beginComputePass({ label: "zgml.wasm.webgpu.tiny-linear" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(rows / 64));
    pass.end();
    gpuDevice.queue.submit([encoder.finish()]);
    return gpuDevice;
  }

  submitMock(options = {}) {
    const rows = options.rows;
    const cols = options.cols;
    const weights = mockFloat32View(options.weights, rows * cols, "mock tiny-linear weights");
    const bias = mockFloat32View(options.bias, rows, "mock tiny-linear bias");
    const input = mockFloat32View(options.input, cols, "mock tiny-linear input");
    const output = mockFloat32View(options.output, rows, "mock tiny-linear output");
    for (let row = 0; row < rows; row += 1) {
      let acc = bias[row];
      for (let col = 0; col < cols; col += 1) {
        acc += weights[row * cols + col] * input[col];
      }
      output[row] = acc;
    }
    return null;
  }

  async execute(options = {}) {
    const gpuDevice = this.submit(options);
    if (!gpuDevice) return;
    if (typeof gpuDevice.queue.onSubmittedWorkDone === "function") {
      await gpuDevice.queue.onSubmittedWorkDone();
    }
  }

  destroy() {
    if (this.paramsBuffer && typeof this.paramsBuffer.destroy === "function") {
      this.paramsBuffer.destroy();
    }
    this.paramsBuffer = null;
    this.pipeline = null;
  }
}

function tinyLinearElementCount(kind, rows, cols) {
  switch (kind) {
    case "weights": return rows * cols;
    case "bias": return rows;
    case "input": return cols;
    case "output": return rows;
    default: throw new Error(`unknown tiny-linear buffer kind: ${kind}`);
  }
}

function tinyLinearAccess(kind) {
  return kind === "output" ? "write" : "read";
}

function resourceBinding(value, label) {
  if (
    !value ||
    typeof value !== "object" ||
    !Number.isSafeInteger(value.buffer) ||
    value.buffer <= 0 ||
    !value.descriptor ||
    typeof value.descriptor !== "object"
  ) {
    throw new Error(`${label} needs a wrapped Wasm resource buffer`);
  }
  return value;
}

function isResourceBinding(value) {
  return !!(
    value &&
    typeof value === "object" &&
    Number.isSafeInteger(value.buffer) &&
    value.buffer > 0 &&
    value.descriptor &&
    typeof value.descriptor === "object"
  );
}

function assertFloat32ByteLength(byteLength, label) {
  assertRange(0, byteLength);
  if (byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error(`${label} byteLength must be a Float32 multiple: ${byteLength}`);
  }
}

function normalizeLlamaModelResourceRole(role, index) {
  const normalized = normalizeResourceTableRole(role);
  return normalized === "" ? `llama.weight.${index}` : normalized;
}

function normalizeLlamaModelResourceEntries(value) {
  if (value === undefined || value === null) return [];
  if (Array.isArray(value)) return value;
  if (typeof value === "object") {
    return Object.entries(value).map(([role, resource]) => {
      const explicitEntry = resource && typeof resource === "object" && (
        "resource" in resource ||
        "gpuBuffer" in resource ||
        "byteLength" in resource ||
        "byteLen" in resource ||
        "access" in resource ||
        "role" in resource ||
        "name" in resource ||
        "tensor" in resource ||
        "kind" in resource ||
        "shape" in resource ||
        "dtype" in resource ||
        "elementCount" in resource ||
        "elements" in resource
      );
      if (explicitEntry && !isResourceBinding(resource)) {
        return { ...resource, role: resource.role ?? role };
      }
      return { role, resource };
    });
  }
  throw new Error("LLaMA model resources must be an array or object map");
}

function normalizeLlamaModelResourceDtype(dtype = "f32") {
  if (dtype === undefined || dtype === null || dtype === "") return "f32";
  if (typeof dtype !== "string") {
    throw new Error(`LLaMA model resource dtype must be a string: ${dtype}`);
  }
  switch (dtype.toLowerCase()) {
    case "f64":
    case "float64":
      return "f64";
    case "f32":
    case "float32":
      return "f32";
    case "f16":
    case "float16":
      return "f16";
    case "bf16":
    case "bfloat16":
      return "bf16";
    case "i64":
    case "int64":
      return "i64";
    case "u64":
    case "uint64":
      return "u64";
    case "i32":
    case "int32":
      return "i32";
    case "u32":
    case "uint32":
      return "u32";
    case "i16":
    case "int16":
      return "i16";
    case "u16":
    case "uint16":
      return "u16";
    case "i8":
    case "int8":
      return "i8";
    case "u8":
    case "uint8":
      return "u8";
    case "bool":
    case "boolean":
      return "bool";
    default:
      return dtype.toLowerCase();
  }
}

function llamaModelResourceScalarBytes(dtype = "f32") {
  switch (normalizeLlamaModelResourceDtype(dtype)) {
    case "f64":
    case "i64":
    case "u64":
      return 8;
    case "f32":
    case "i32":
    case "u32":
      return 4;
    case "f16":
    case "bf16":
    case "i16":
    case "u16":
      return 2;
    case "i8":
    case "u8":
    case "bool":
      return 1;
    default:
      throw new Error(`unsupported LLaMA model resource dtype for derived byteLength: ${dtype}`);
  }
}

function shouldMaterializeSafetensorsFloatDtype(dtype, options = {}) {
  const sourceDtype = normalizeLlamaModelResourceDtype(dtype);
  if (sourceDtype !== "f16" && sourceDtype !== "bf16") return false;
  const requested = options.materializeFloatDtypes ?? options.materializeFloatDtype ?? options.materializeDtype ?? false;
  if (requested === false || requested === undefined || requested === null) return false;
  const targetDtype = requested === true ? "f32" : normalizeLlamaModelResourceDtype(String(requested));
  if (targetDtype !== "f32") {
    throw new Error(`unsupported safetensors materialized dtype: ${requested}`);
  }
  return true;
}

function float16BitsToNumber(bits) {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) {
    return sign * Math.pow(2, -14) * (fraction / 0x400);
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : NaN;
  }
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 0x400);
}

const bfloatScratchU32 = new Uint32Array(1);
const bfloatScratchF32 = new Float32Array(bfloatScratchU32.buffer);

function bfloat16BitsToNumber(bits) {
  bfloatScratchU32[0] = (bits & 0xffff) << 16;
  return bfloatScratchF32[0];
}

function decodeSafetensorsFloat32Values(bytes, byteOffset, byteLength, sourceDtype, elementCount, label) {
  const dtype = normalizeLlamaModelResourceDtype(sourceDtype);
  if (!Number.isSafeInteger(elementCount) || elementCount <= 0) {
    throw new Error(`${label} has invalid elementCount for dtype materialization: ${elementCount}`);
  }
  const expectedByteLength = elementCount * llamaModelResourceScalarBytes(dtype);
  if (byteLength !== expectedByteLength) {
    throw new Error(`${label} source byteLength ${byteLength} does not match ${dtype} elementCount ${elementCount}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset + byteOffset, byteLength);
  const out = new Float32Array(elementCount);
  switch (dtype) {
    case "f32":
      for (let i = 0; i < elementCount; i += 1) {
        out[i] = view.getFloat32(i * 4, true);
      }
      return out;
    case "f16":
      for (let i = 0; i < elementCount; i += 1) {
        out[i] = float16BitsToNumber(view.getUint16(i * 2, true));
      }
      return out;
    case "bf16":
      for (let i = 0; i < elementCount; i += 1) {
        out[i] = bfloat16BitsToNumber(view.getUint16(i * 2, true));
      }
      return out;
    default:
      throw new Error(`${label} cannot materialize non-floating safetensors dtype ${dtype}`);
  }
}

function normalizeLlamaModelResourceShape(shape) {
  if (shape === undefined || shape === null) return undefined;
  if (!Array.isArray(shape)) {
    throw new Error("LLaMA model resource shape must be an array");
  }
  let elementCount = 1;
  const normalized = shape.map((dim) => {
    if (!Number.isSafeInteger(dim) || dim <= 0) {
      throw new Error(`invalid LLaMA model resource shape dimension: ${dim}`);
    }
    elementCount *= dim;
    if (!Number.isSafeInteger(elementCount)) {
      throw new Error("LLaMA model resource shape is too large");
    }
    return dim;
  });
  return { elementCount, shape: normalized };
}

function explicitLlamaModelResourceElementCount(entry) {
  const elementCount = entry?.elementCount ?? entry?.elements;
  if (elementCount === undefined) return undefined;
  if (!Number.isSafeInteger(elementCount) || elementCount <= 0) {
    throw new Error(`invalid LLaMA model resource element count: ${elementCount}`);
  }
  return elementCount;
}

function llamaModelResourceElementCount(entry) {
  const explicit = explicitLlamaModelResourceElementCount(entry);
  if (explicit !== undefined) return explicit;
  return normalizeLlamaModelResourceShape(entry?.shape)?.elementCount;
}

function llamaModelResourceByteLength(entry, source) {
  const explicit = entry?.byteLength ?? entry?.byteLen ?? entry?.size;
  if (explicit !== undefined) {
    assertRange(0, explicit);
    return explicit;
  }
  const elementCount = llamaModelResourceElementCount(entry);
  if (elementCount !== undefined) {
    const byteLength = elementCount * llamaModelResourceScalarBytes(entry?.dtype ?? "f32");
    if (!Number.isSafeInteger(byteLength)) {
      throw new Error("LLaMA model resource byteLength is too large");
    }
    assertRange(0, byteLength);
    return byteLength;
  }
  const sourceByteLength = source?.descriptor?.byteLength ?? source?.byteLength ?? source?.byteLen ?? source?.size;
  if (sourceByteLength !== undefined) {
    assertRange(0, sourceByteLength);
    return sourceByteLength;
  }
  throw new Error("LLaMA model resource needs byteLength, shape, elementCount, or sized resource");
}

function normalizeLlamaModelResourceRequirement(entry, index) {
  if (typeof entry === "string") {
    return { role: normalizeLlamaModelResourceRole(entry, index) };
  }
  if (!entry || typeof entry !== "object") {
    throw new Error(`invalid LLaMA model resource requirement: ${entry}`);
  }
  const role = normalizeLlamaModelResourceRole(entry.role ?? entry.name ?? entry.tensor ?? entry.kind, index);
  const hasDtype = entry.dtype !== undefined && entry.dtype !== null && entry.dtype !== "";
  const dtype = hasDtype ? normalizeLlamaModelResourceDtype(entry.dtype) : undefined;
  const shapeInfo = normalizeLlamaModelResourceShape(entry.shape);
  const elementCount = explicitLlamaModelResourceElementCount(entry) ?? shapeInfo?.elementCount;
  let byteLength = entry.byteLength ?? entry.byteLen ?? entry.size;
  if (byteLength !== undefined) {
    assertRange(0, byteLength);
  } else if (elementCount !== undefined && dtype !== undefined) {
    byteLength = elementCount * llamaModelResourceScalarBytes(dtype);
    if (!Number.isSafeInteger(byteLength)) {
      throw new Error(`LLaMA model resource requirement ${role} byteLength is too large`);
    }
    assertRange(0, byteLength);
  }
  const dataByteOffset = entry.dataByteOffset;
  if (dataByteOffset !== undefined && (!Number.isSafeInteger(dataByteOffset) || dataByteOffset < 0)) {
    throw new Error(`invalid LLaMA model resource requirement dataByteOffset for ${role}: ${dataByteOffset}`);
  }
  const dataByteLength = entry.dataByteLength;
  if (dataByteLength !== undefined && (!Number.isSafeInteger(dataByteLength) || dataByteLength < 0)) {
    throw new Error(`invalid LLaMA model resource requirement dataByteLength for ${role}: ${dataByteLength}`);
  }
  return {
    byteLength,
    dataByteLength,
    dataByteOffset,
    dtype,
    elementCount,
    role,
    shape: shapeInfo?.shape,
  };
}

function normalizeLlamaModelResourceRequirementFromEntry(entry, index, options = {}) {
  if (!entry || typeof entry !== "object") {
    return normalizeLlamaModelResourceRequirement(entry, index);
  }
  const role = normalizeLlamaModelResourceRole(entry.role ?? entry.name ?? entry.tensor ?? entry.kind, index);
  const dtype = normalizeLlamaModelResourceDtype(entry.dtype ?? options.dtype ?? "f32");
  const shapeInfo = normalizeLlamaModelResourceShape(entry.shape);
  const elementCount = explicitLlamaModelResourceElementCount(entry) ?? shapeInfo?.elementCount;
  const source = isResourceBinding(entry)
    ? entry
    : (entry.resource ?? entry.gpuBuffer ?? entry.buffer ?? entry.externalResource ?? entry);
  const byteLength = llamaModelResourceByteLength({ ...entry, dtype, elementCount }, source);
  return normalizeLlamaModelResourceRequirement({
    byteLength,
    dataByteLength: entry.dataByteLength,
    dataByteOffset: entry.dataByteOffset,
    dtype,
    elementCount,
    role,
    shape: shapeInfo?.shape,
  }, index);
}

function normalizeLlamaModelResourceRequirements(value) {
  if (value === undefined || value === null) return [];
  const source = value && typeof value === "object" && Array.isArray(value.entries) ? value.entries : value;
  const requirements = normalizeLlamaModelResourceEntries(source).map((entry, index) => normalizeLlamaModelResourceRequirement(entry, index));
  const seen = new Set();
  for (const requirement of requirements) {
    if (seen.has(requirement.role)) {
      throw new Error(`duplicate required LLaMA model resource role: ${requirement.role}`);
    }
    seen.add(requirement.role);
  }
  return requirements;
}

function llamaModelResourceRequirementInspection(entry) {
  const shape = Array.isArray(entry.shape) ? Object.freeze(entry.shape.slice()) : undefined;
  return Object.freeze({
    byteLength: entry.byteLength,
    dataByteLength: entry.dataByteLength,
    dataByteOffset: entry.dataByteOffset,
    dtype: entry.dtype,
    elementCount: entry.elementCount,
    role: entry.role,
    shape,
  });
}

function llamaResourceTableInspection(table) {
  if (table === null || table === undefined) return null;
  return Object.freeze({
    byteLength: table.byteLength,
    descriptorHash: table.descriptorHash,
    deviceHandle: table.deviceHandle,
    hash: table.hash,
    resourceCount: table.resourceCount,
    roles: Object.freeze((table.descriptors ?? []).map((entry) => entry.role)),
  });
}

function normalizeLlamaModelResourceRequirementsFromResourceMap(value, options = {}) {
  const requirements = normalizeLlamaModelResourceEntries(value).map((entry, index) => (
    normalizeLlamaModelResourceRequirementFromEntry(entry, index, options)
  ));
  const seen = new Set();
  for (const requirement of requirements) {
    if (seen.has(requirement.role)) {
      throw new Error(`duplicate required LLaMA model resource role: ${requirement.role}`);
    }
    seen.add(requirement.role);
  }
  return requirements;
}

function llamaModelResourceShapeText(shape) {
  return Array.isArray(shape) ? `[${shape.join(",")}]` : "unspecified";
}

function sameLlamaModelResourceShape(actual, expected) {
  return (
    Array.isArray(actual) &&
    actual.length === expected.length &&
    actual.every((dim, index) => dim === expected[index])
  );
}

function validateLlamaModelResourceRequirements(resources, requirements, options = {}) {
  if (!Array.isArray(requirements)) return;
  const byRole = new Map();
  for (const resource of resources) {
    if (byRole.has(resource.role)) {
      throw new Error(`duplicate LLaMA model resource role: ${resource.role}`);
    }
    byRole.set(resource.role, resource);
  }
  const requiredRoles = new Set(requirements.map((requirement) => requirement.role));
  for (const requirement of requirements) {
    const actual = byRole.get(requirement.role);
    if (!actual) {
      throw new Error(`missing required LLaMA model resource: ${requirement.role}`);
    }
    if (requirement.dtype !== undefined && normalizeLlamaModelResourceDtype(actual.dtype) !== requirement.dtype) {
      throw new Error(`LLaMA model resource ${requirement.role} dtype mismatch: ${actual.dtype} != ${requirement.dtype}`);
    }
    if (requirement.shape !== undefined && !sameLlamaModelResourceShape(actual.shape, requirement.shape)) {
      throw new Error(`LLaMA model resource ${requirement.role} shape mismatch: ${llamaModelResourceShapeText(actual.shape)} != ${llamaModelResourceShapeText(requirement.shape)}`);
    }
    if (requirement.elementCount !== undefined && actual.elementCount !== requirement.elementCount) {
      throw new Error(`LLaMA model resource ${requirement.role} element count mismatch: ${actual.elementCount} != ${requirement.elementCount}`);
    }
    if (requirement.byteLength !== undefined && actual.byteLength !== requirement.byteLength) {
      throw new Error(`LLaMA model resource ${requirement.role} byteLength mismatch: ${actual.byteLength} != ${requirement.byteLength}`);
    }
    if (requirement.dataByteOffset !== undefined && actual.dataByteOffset !== requirement.dataByteOffset) {
      throw new Error(`LLaMA model resource ${requirement.role} dataByteOffset mismatch: ${actual.dataByteOffset} != ${requirement.dataByteOffset}`);
    }
    if (requirement.dataByteLength !== undefined && actual.dataByteLength !== requirement.dataByteLength) {
      throw new Error(`LLaMA model resource ${requirement.role} dataByteLength mismatch: ${actual.dataByteLength} != ${requirement.dataByteLength}`);
    }
  }
  if (options.allowExtra === false) {
    for (const resource of resources) {
      if (!requiredRoles.has(resource.role)) {
        throw new Error(`unexpected LLaMA model resource: ${resource.role}`);
      }
    }
  }
}

function bytesView(value, label) {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  throw new Error(`${label} must be a Uint8Array, ArrayBuffer, typed array, or DataView`);
}

function isSafetensorsByteSource(value) {
  return value instanceof ArrayBuffer || ArrayBuffer.isView(value);
}

function safetensorsShardPayload(value) {
  if (isSafetensorsByteSource(value) || typeof value === "string") return value;
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value.safetensors ?? value.safetensorsData ?? value.data ?? value.bytes ?? value.source;
  }
  return undefined;
}

function isSafetensorsShardSource(value) {
  const payload = safetensorsShardPayload(value);
  return isSafetensorsByteSource(payload) || typeof payload === "string";
}

function isSafetensorsShardSourceList(value) {
  return Array.isArray(value) && value.length > 0 && value.every(isSafetensorsShardSource);
}

function safetensorsShardSourceList(source) {
  if (isSafetensorsShardSourceList(source)) return source;
  if (source && typeof source === "object" && isSafetensorsShardSourceList(source.shards)) return source.shards;
  return null;
}

function safetensorsShardName(source, index) {
  if (source && typeof source === "object" && !Array.isArray(source) && !isSafetensorsByteSource(source)) {
    const name = source.name ?? source.filename ?? source.fileName ?? source.path ?? source.file;
    if (name !== undefined && name !== null && name !== "") {
      if (typeof name !== "string") {
        throw new Error(`safetensors shard ${index} name must be a string`);
      }
      return name;
    }
  }
  return `shard.${index}`;
}

function jsonObjectFromOptionalValue(value, label) {
  if (value === undefined || value === null) return null;
  let parsed = value;
  if (typeof value === "string") {
    parsed = JSON.parse(value);
  } else if (isSafetensorsByteSource(value)) {
    const bytes = bytesView(value, label);
    parsed = JSON.parse(new TextDecoder().decode(bytes));
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON object`);
  }
  return parsed;
}

function safetensorsIndexObject(source, options = {}) {
  const indexSource =
    options.safetensorsIndex ??
    options.index ??
    source?.safetensorsIndex ??
    source?.index ??
    source?.indexJson ??
    source?.weight_map ??
    source?.weightMap;
  const parsed = jsonObjectFromOptionalValue(indexSource, "safetensors index");
  if (parsed === null) return null;
  if (parsed.weight_map !== undefined || parsed.weightMap !== undefined) return parsed;
  return { weight_map: parsed };
}

function safetensorsIndexWeightMap(source, options = {}) {
  const index = safetensorsIndexObject(source, options);
  if (index === null) return null;
  const weightMap = index.weight_map ?? index.weightMap;
  if (weightMap === undefined || weightMap === null) {
    throw new Error("safetensors index is missing weight_map");
  }
  if (typeof weightMap !== "object" || Array.isArray(weightMap)) {
    throw new Error("safetensors index weight_map must be an object");
  }
  const normalized = new Map();
  for (const [tensor, shardName] of Object.entries(weightMap)) {
    if (typeof shardName !== "string" || shardName.length === 0) {
      throw new Error(`safetensors index weight_map entry for ${tensor} must name a shard`);
    }
    normalized.set(tensor, shardName);
  }
  return normalized;
}

function safetensorsIndexTotalSize(source, options = {}) {
  const index = safetensorsIndexObject(source, options);
  if (index === null) return null;
  const metadata = index.metadata ?? index.__metadata__;
  if (metadata === undefined || metadata === null) return null;
  if (typeof metadata !== "object" || Array.isArray(metadata)) {
    throw new Error("safetensors index metadata must be an object");
  }
  const totalSize = metadata.total_size ?? metadata.totalSize;
  if (totalSize === undefined || totalSize === null || totalSize === "") return null;
  return nonNegativeIntegerFromMetadataValue(totalSize, "safetensors index metadata total_size");
}

function safetensorsTensorSourceByteLength(name, tensor) {
  if (!tensor || typeof tensor !== "object") {
    throw new Error(`safetensors tensor ${name} metadata must be an object`);
  }
  if (tensor.dtype === undefined || tensor.dtype === null) {
    throw new Error(`safetensors tensor ${name} is missing dtype`);
  }
  if (tensor.shape === undefined || tensor.shape === null) {
    throw new Error(`safetensors tensor ${name} is missing shape`);
  }
  const shapeInfo = normalizeLlamaModelResourceShape(tensor.shape);
  const sourceDtype = normalizeLlamaModelResourceDtype(tensor.dtype);
  const byteLength = shapeInfo.elementCount * llamaModelResourceScalarBytes(sourceDtype);
  if (!Number.isSafeInteger(byteLength)) {
    throw new Error(`safetensors tensor ${name} byteLength is too large`);
  }
  return byteLength;
}

function parseSingleSafetensorsHeaderSource(source, options = {}) {
  if (typeof source === "string") return JSON.parse(source);
  if (source && typeof source === "object" && !ArrayBuffer.isView(source) && !(source instanceof ArrayBuffer) && !Array.isArray(source)) {
    return source;
  }
  const bytes = bytesView(source, "safetensors data");
  if (options.headerOnly === true) {
    return JSON.parse(new TextDecoder().decode(bytes));
  }
  if (bytes.byteLength < 8) {
    throw new Error("safetensors data is missing the 8-byte header length");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = view.getBigUint64(0, true);
  if (headerLength > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`safetensors header is too large: ${headerLength}`);
  }
  const headerByteLength = Number(headerLength);
  if (8 + headerByteLength > bytes.byteLength) {
    throw new Error("safetensors data is truncated before the complete header");
  }
  return JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headerByteLength)));
}

function mergeSafetensorsMetadata(target, source, label) {
  if (source === undefined || source === null) return;
  if (typeof source !== "object" || Array.isArray(source)) {
    throw new Error(`${label} __metadata__ must be an object when present`);
  }
  for (const [key, value] of Object.entries(source)) {
    if (Object.prototype.hasOwnProperty.call(target, key) && JSON.stringify(target[key]) !== JSON.stringify(value)) {
      throw new Error(`safetensors shard metadata mismatch for ${key}: ${JSON.stringify(target[key])} != ${JSON.stringify(value)}`);
    }
    target[key] = value;
  }
}

function parseSafetensorsHeaderSource(source, options = {}) {
  const shards = safetensorsShardSourceList(source);
  if (shards === null) return parseSingleSafetensorsHeaderSource(source, options);
  const merged = {};
  const metadata = {};
  let hasMetadata = false;
  const shardNames = new Set();
  const weightMap = safetensorsIndexWeightMap(source, options);
  const expectedTotalSize = safetensorsIndexTotalSize(source, options);
  let actualTotalSize = 0;
  for (let shardIndex = 0; shardIndex < shards.length; shardIndex += 1) {
    const shard = shards[shardIndex];
    const header = parseSingleSafetensorsHeaderSource(safetensorsShardPayload(shard), options);
    const shardName = safetensorsShardName(shard, shardIndex);
    if (shardNames.has(shardName)) {
      throw new Error(`duplicate safetensors shard name: ${shardName}`);
    }
    shardNames.add(shardName);
    if (header.__metadata__ !== undefined) {
      mergeSafetensorsMetadata(metadata, header.__metadata__, `safetensors ${shardName}`);
      hasMetadata = true;
    }
    for (const [name, tensor] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      if (Object.prototype.hasOwnProperty.call(merged, name)) {
        throw new Error(`duplicate safetensors tensor across shards: ${name}`);
      }
      if (weightMap !== null) {
        const mappedShard = weightMap.get(name);
        if (mappedShard === undefined) {
          throw new Error(`safetensors index weight_map is missing tensor: ${name}`);
        }
        if (mappedShard !== shardName) {
          throw new Error(`safetensors index weight_map mismatch for ${name}: ${mappedShard} != ${shardName}`);
        }
      }
      actualTotalSize += safetensorsTensorSourceByteLength(name, tensor);
      if (!Number.isSafeInteger(actualTotalSize)) {
        throw new Error("safetensors index total tensor byteLength is too large");
      }
      merged[name] = {
        ...tensor,
        __zgml_safetensors_shard_index: shardIndex,
        __zgml_safetensors_shard_name: shardName,
      };
    }
  }
  if (weightMap !== null) {
    for (const [tensor, shardName] of weightMap.entries()) {
      if (!Object.prototype.hasOwnProperty.call(merged, tensor)) {
        throw new Error(`safetensors index weight_map references missing tensor: ${tensor}`);
      }
      if (!shardNames.has(shardName)) {
        throw new Error(`safetensors index weight_map references missing shard: ${shardName}`);
      }
    }
  }
  if (expectedTotalSize !== null && actualTotalSize !== expectedTotalSize) {
    throw new Error(`safetensors index metadata total_size mismatch: ${actualTotalSize} != ${expectedTotalSize}`);
  }
  if (hasMetadata) merged.__metadata__ = metadata;
  return merged;
}

function safetensorsDataSection(source) {
  const bytes = bytesView(source, "safetensors data");
  if (bytes.byteLength < 8) {
    throw new Error("safetensors data is missing the 8-byte header length");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = view.getBigUint64(0, true);
  if (headerLength > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`safetensors header is too large: ${headerLength}`);
  }
  const dataStart = 8 + Number(headerLength);
  if (dataStart > bytes.byteLength) {
    throw new Error("safetensors data is truncated before the complete header");
  }
  return { bytes, dataStart };
}

function safetensorsDataSections(source) {
  const shards = safetensorsShardSourceList(source);
  if (shards === null) return [safetensorsDataSection(source)];
  return shards.map((shard) => safetensorsDataSection(safetensorsShardPayload(shard)));
}

function validateSafetensorsDataOffsetLayout(source, specs) {
  const hasBytePayload = hasSafetensorsBytePayload(source);
  const sections = hasBytePayload ? safetensorsDataSections(source) : [];
  const byShard = new Map();
  for (const spec of specs) {
    if (!Number.isSafeInteger(spec.dataByteOffset) || !Number.isSafeInteger(spec.dataByteLength)) continue;
    const name = spec.tensor ?? spec.name ?? "unknown";
    if (spec.dataByteLength === 0) continue;
    if (hasBytePayload) {
      const section = sections[spec.safetensorsShardIndex ?? 0];
      if (!section) {
        throw new Error(`safetensors tensor ${name} references missing shard ${spec.safetensorsShardIndex ?? 0}`);
      }
      if (section.dataStart + spec.dataByteOffset + spec.dataByteLength > section.bytes.byteLength) {
        throw new Error(`safetensors tensor ${name} data range is out of bounds`);
      }
    }
    const shardIndex = spec.safetensorsShardIndex ?? 0;
    let ranges = byShard.get(shardIndex);
    if (!ranges) {
      ranges = [];
      byShard.set(shardIndex, ranges);
    }
    ranges.push({
      end: spec.dataByteOffset + spec.dataByteLength,
      name,
      start: spec.dataByteOffset,
    });
  }
  for (const ranges of byShard.values()) {
    ranges.sort((a, b) => a.start - b.start || a.end - b.end || a.name.localeCompare(b.name));
    for (let index = 1; index < ranges.length; index += 1) {
      const prev = ranges[index - 1];
      const current = ranges[index];
      if (prev.end > current.start) {
        throw new Error(`safetensors tensor data_offsets overlap: ${prev.name} and ${current.name}`);
      }
    }
  }
}

function normalizeSafetensorsDataOffsets(value, name) {
  if (value === undefined || value === null) return null;
  if (!Array.isArray(value) || value.length !== 2) {
    throw new Error(`safetensors tensor ${name} data_offsets must be [start, end]`);
  }
  const [start, end] = value;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start) {
    throw new Error(`invalid safetensors tensor ${name} data_offsets: ${value}`);
  }
  return { byteLength: end - start, byteOffset: start };
}

export function safetensorsModelResourceSpecs(source, options = {}) {
  const header = parseSafetensorsHeaderSource(source, options);
  const hasBytePayload = hasSafetensorsBytePayload(source);
  const rolePrefix = options.rolePrefix ?? "llama.weight.";
  const embeddingTensor = options.embeddingTensor ?? "model.embed_tokens.weight";
  const lmHeadTensor = options.lmHeadTensor ?? "lm_head.weight";
  const specs = [];
  for (const [name, tensor] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (!tensor || typeof tensor !== "object") {
      throw new Error(`safetensors tensor ${name} metadata must be an object`);
    }
    if (tensor.dtype === undefined || tensor.dtype === null) {
      throw new Error(`safetensors tensor ${name} is missing dtype`);
    }
    if (tensor.shape === undefined || tensor.shape === null) {
      throw new Error(`safetensors tensor ${name} is missing shape`);
    }
    const shapeInfo = normalizeLlamaModelResourceShape(tensor.shape);
    const sourceDtype = normalizeLlamaModelResourceDtype(tensor.dtype);
    const sourceByteLength = shapeInfo.elementCount * llamaModelResourceScalarBytes(sourceDtype);
    if (!Number.isSafeInteger(sourceByteLength)) {
      throw new Error(`safetensors tensor ${name} byteLength is too large`);
    }
    const offsets = normalizeSafetensorsDataOffsets(tensor.data_offsets ?? tensor.dataOffsets, name);
    if (hasBytePayload && offsets === null) {
      throw new Error(`safetensors tensor ${name} is missing data_offsets`);
    }
    if (offsets && (hasBytePayload || offsets.byteLength > 0) && offsets.byteLength !== sourceByteLength) {
      throw new Error(`safetensors tensor ${name} byteLength ${offsets.byteLength} does not match shape/dtype byteLength ${sourceByteLength}`);
    }
    const materializeFloatDtype = shouldMaterializeSafetensorsFloatDtype(sourceDtype, options);
    const dtype = materializeFloatDtype ? "f32" : sourceDtype;
    const byteLength = shapeInfo.elementCount * llamaModelResourceScalarBytes(dtype);
    if (!Number.isSafeInteger(byteLength)) {
      throw new Error(`safetensors tensor ${name} materialized byteLength is too large`);
    }
    const spec = {
      byteLength,
      dtype,
      name,
      role: `${rolePrefix}${name}`,
      safetensorsShardIndex: tensor.__zgml_safetensors_shard_index,
      safetensorsShardName: tensor.__zgml_safetensors_shard_name,
      shape: shapeInfo.shape,
      sourceByteLength,
      sourceDtype,
      tensor: name,
    };
    if (offsets) {
      spec.dataByteOffset = offsets.byteOffset;
      spec.dataByteLength = offsets.byteLength;
    }
    specs.push(spec);
  }
  validateSafetensorsDataOffsetLayout(source, specs);
  if (options.allowTiedLmHead === true || options.tiedLmHead === true || options.tieLmHead === true) {
    const hasLmHead = specs.some((spec) => spec.tensor === lmHeadTensor);
    if (!hasLmHead) {
      const embedding = specs.find((spec) => spec.tensor === embeddingTensor);
      if (!embedding) {
        throw new Error(`tied LLaMA lm_head needs embedding tensor ${embeddingTensor}`);
      }
      specs.push({
        ...embedding,
        name: lmHeadTensor,
        role: `${rolePrefix}${lmHeadTensor}`,
        tensor: lmHeadTensor,
      });
    }
  }
  if (specs.length === 0) {
    throw new Error("safetensors header contains no tensor resources");
  }
  return specs;
}

function findSafetensorsTensorSpec(specs, tensorName) {
  return specs.find((spec) => spec.tensor === tensorName || spec.name === tensorName) ?? null;
}

function safetensorsMatrixShape(spec, label) {
  if (!spec) {
    throw new Error(`${label} is missing`);
  }
  if (!Array.isArray(spec.shape) || spec.shape.length !== 2) {
    throw new Error(`${label} must be a matrix`);
  }
  const [rows, cols] = spec.shape;
  if (!Number.isSafeInteger(rows) || rows <= 0 || !Number.isSafeInteger(cols) || cols <= 0) {
    throw new Error(`${label} has invalid shape: ${spec.shape}`);
  }
  return { rows, cols };
}

function safetensorsVectorLength(spec, label) {
  if (!spec) {
    throw new Error(`${label} is missing`);
  }
  if (!Array.isArray(spec.shape) || spec.shape.length !== 1) {
    throw new Error(`${label} must be a vector`);
  }
  const [length] = spec.shape;
  if (!Number.isSafeInteger(length) || length <= 0) {
    throw new Error(`${label} has invalid shape: ${spec.shape}`);
  }
  return length;
}

function inferLlamaLayerIndicesFromSafetensorsSpecs(specs) {
  const layers = new Set();
  for (const spec of specs) {
    const match = /^model\.layers\.(\d+)\./.exec(spec.tensor ?? spec.name ?? "");
    if (match) layers.add(Number(match[1]));
  }
  if (layers.size === 0) {
    throw new Error("LLaMA safetensors resource Program needs at least one layer tensor");
  }
  const sorted = Array.from(layers).sort((a, b) => a - b);
  for (let index = 0; index < sorted.length; index += 1) {
    if (sorted[index] !== index) {
      throw new Error(`LLaMA safetensors layer indices must be contiguous from 0, found ${sorted.join(",")}`);
    }
  }
  return sorted;
}

function isExecutableLlamaSafetensorsModelTensorName(name, options = {}) {
  if (name === (options.embeddingTensor ?? "model.embed_tokens.weight")) return true;
  if (name === (options.lmHeadTensor ?? "lm_head.weight")) return true;
  if (name === (options.lmHeadBiasTensor ?? "lm_head.bias")) return true;
  if (name === (options.normTensor ?? "model.norm.weight")) return true;
  const match = /^model\.layers\.\d+\.(.+)$/.exec(name);
  if (!match) return false;
  switch (match[1]) {
    case "self_attn.q_proj.weight":
    case "self_attn.k_proj.weight":
    case "self_attn.v_proj.weight":
    case "self_attn.o_proj.weight":
    case "self_attn.q_proj.bias":
    case "self_attn.k_proj.bias":
    case "self_attn.v_proj.bias":
    case "self_attn.o_proj.bias":
      return true;
    case "self_attn.q_norm.weight":
    case "self_attn.k_norm.weight":
      return options.qkProjectionNorm === true;
    case "mlp.gate_proj.weight":
    case "mlp.up_proj.weight":
    case "mlp.down_proj.weight":
    case "mlp.gate_proj.bias":
    case "mlp.up_proj.bias":
    case "mlp.down_proj.bias":
    case "input_layernorm.weight":
    case "post_attention_layernorm.weight":
      return true;
    default:
      return false;
  }
}

function isLlamaSafetensorsRotaryInvFreqTensorName(name) {
  return (
    name === "model.rotary_emb.inv_freq" ||
    /^model\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq$/.test(name)
  );
}

function isSupportedLlamaSafetensorsTensorName(name, options = {}) {
  return (
    isExecutableLlamaSafetensorsModelTensorName(name, options) ||
    isLlamaSafetensorsRotaryInvFreqTensorName(name)
  );
}

function executableLlamaSafetensorsModelResourceSpecs(specs, options = {}) {
  return specs.filter((spec) => {
    const name = spec.tensor ?? spec.name ?? "";
    return typeof name === "string" && isExecutableLlamaSafetensorsModelTensorName(name, options);
  });
}

function rejectUnsupportedLlamaSafetensorsTensorNames(specs, options = {}) {
  for (const spec of specs) {
    const name = spec.tensor ?? spec.name ?? "";
    if (typeof name !== "string" || name === "") continue;
    if (!name.endsWith(".bias")) continue;
    if (/^model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.bias$/.test(name)) {
      continue;
    }
    if (/^model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)\.bias$/.test(name)) {
      continue;
    }
    if (name === (options.lmHeadBiasTensor ?? "lm_head.bias")) {
      continue;
    }
    if (
      name === "model.norm.bias" ||
      /^model\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm)\.bias$/.test(name)
    ) {
      throw new Error(`unsupported LLaMA safetensors norm bias tensor: ${name}`);
    }
    throw new Error(`unsupported LLaMA safetensors bias tensor: ${name}`);
  }
  for (const spec of specs) {
    const name = spec.tensor ?? spec.name ?? "";
    if (typeof name !== "string" || name === "") continue;
    if (isSupportedLlamaSafetensorsTensorName(name, options)) continue;
    if (/^model\.layers\.\d+\./.test(name)) {
      throw new Error(`unsupported LLaMA safetensors layer tensor: ${name}`);
    }
    throw new Error(`unsupported LLaMA safetensors tensor: ${name}`);
  }
}

function rejectUnsupportedLlamaSafetensorsExecutorDtypes(specs) {
  for (const spec of specs) {
    const dtype = normalizeLlamaModelResourceDtype(spec.dtype);
    if (dtype === "f32") continue;
    const name = spec.tensor ?? spec.name ?? spec.role ?? "unknown";
    const sourceDtype = spec.sourceDtype === undefined ? dtype : normalizeLlamaModelResourceDtype(spec.sourceDtype);
    throw new Error(`unsupported LLaMA safetensors tensor dtype for executable proof: ${name} ${sourceDtype}`);
  }
}

function hasSafetensorsBytePayload(source) {
  if (isSafetensorsByteSource(source)) return true;
  const shards = safetensorsShardSourceList(source);
  return shards !== null && shards.every((shard) => isSafetensorsByteSource(safetensorsShardPayload(shard)));
}

function validateLlamaSafetensorsRotaryInvFreqValues(source, specs, options = {}) {
  const attentionHeadSize = options.attentionHeadSize;
  const ropeBase = options.ropeBase;
  const ropeScaling = options.ropeScaling ?? {
    ropeHighFreqFactor: options.ropeHighFreqFactor,
    ropeKind: options.ropeKind,
    ropeLowFreqFactor: options.ropeLowFreqFactor,
    ropeOriginalContextLength: options.ropeOriginalContextLength,
    ropeScale: options.ropeScale,
  };
  if (!Number.isSafeInteger(attentionHeadSize) || attentionHeadSize <= 0 || attentionHeadSize % 2 !== 0) {
    throw new Error(`invalid LLaMA rotary inv_freq head size: ${attentionHeadSize}`);
  }
  if (typeof ropeBase !== "number" || !Number.isFinite(ropeBase) || ropeBase <= 0) {
    throw new Error(`invalid LLaMA rotary inv_freq RoPE base: ${ropeBase}`);
  }
  const expectedLength = attentionHeadSize / 2;
  const layerIndices = new Set(options.layerIndices ?? []);
  const rotarySpecs = specs.filter((spec) => isLlamaSafetensorsRotaryInvFreqTensorName(spec.tensor ?? spec.name ?? ""));
  for (const spec of rotarySpecs) {
    const name = spec.tensor ?? spec.name ?? "";
    const length = safetensorsVectorLength(spec, `LLaMA safetensors rotary inv_freq tensor ${name}`);
    if (length !== expectedLength) {
      throw new Error(`LLaMA safetensors rotary inv_freq length mismatch: ${name} ${length} != ${expectedLength}`);
    }
    const layerMatch = /^model\.layers\.(\d+)\./.exec(name);
    if (layerMatch && !layerIndices.has(Number(layerMatch[1]))) {
      throw new Error(`LLaMA safetensors rotary inv_freq layer is not executable: ${name}`);
    }
  }
  if (rotarySpecs.length === 0 || !hasSafetensorsBytePayload(source)) return;

  const sections = safetensorsDataSections(source);
  for (const spec of rotarySpecs) {
    const name = spec.tensor ?? spec.name ?? "";
    if ((spec.dataByteLength ?? 0) === 0) continue;
    if (!Number.isSafeInteger(spec.dataByteOffset) || !Number.isSafeInteger(spec.dataByteLength)) {
      throw new Error(`LLaMA safetensors rotary inv_freq tensor ${name} is missing data offsets`);
    }
    const shardIndex = spec.safetensorsShardIndex ?? 0;
    const section = sections[shardIndex];
    if (!section) {
      throw new Error(`LLaMA safetensors rotary inv_freq tensor ${name} references missing shard ${shardIndex}`);
    }
    const sourceDtype = normalizeLlamaModelResourceDtype(spec.sourceDtype ?? spec.dtype);
    const start = section.dataStart + spec.dataByteOffset;
    const end = start + spec.dataByteLength;
    if (end > section.bytes.byteLength) {
      throw new Error(`LLaMA safetensors rotary inv_freq tensor ${name} data range is out of bounds`);
    }
    const values = decodeSafetensorsFloat32Values(
      section.bytes,
      start,
      spec.dataByteLength,
      sourceDtype,
      expectedLength,
      `LLaMA safetensors rotary inv_freq tensor ${name}`,
    );
    for (let i = 0; i < values.length; i += 1) {
      const expected = Math.fround(llamaRopeInvFrequency(i, attentionHeadSize, ropeBase, ropeScaling));
      const tolerance = Math.max(1e-3, Math.abs(expected) * 1e-3);
      if (Math.abs(values[i] - expected) > tolerance) {
        throw new Error(`LLaMA safetensors rotary inv_freq value mismatch: ${name}[${i}] ${values[i]} != ${expected}`);
      }
    }
  }
}

function safetensorsMetadataObject(header) {
  const metadata = header?.__metadata__;
  if (metadata === undefined || metadata === null) return {};
  if (typeof metadata !== "object" || Array.isArray(metadata)) {
    throw new Error("safetensors __metadata__ must be an object when present");
  }
  return metadata;
}

function jsonObjectFromConfigValue(value, label) {
  if (value === undefined || value === null) return {};
  let parsed = value;
  if (typeof value === "string") {
    parsed = JSON.parse(value);
  } else if (typeof ArrayBuffer !== "undefined" && value instanceof ArrayBuffer) {
    parsed = JSON.parse(new TextDecoder().decode(new Uint8Array(value)));
  } else if (typeof ArrayBuffer !== "undefined" && ArrayBuffer.isView(value)) {
    parsed = JSON.parse(new TextDecoder().decode(new Uint8Array(value.buffer, value.byteOffset, value.byteLength)));
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON object`);
  }
  return parsed;
}

function configObjectFromFirstSource(source, names, label) {
  for (const name of names) {
    if (Object.prototype.hasOwnProperty.call(source, name)) {
      return jsonObjectFromConfigValue(source[name], `${label} ${name}`);
    }
  }
  return {};
}

function configValuesEquivalent(a, b) {
  if (Object.is(a, b)) return true;
  if (
    (typeof a === "number" || typeof a === "string") &&
    (typeof b === "number" || typeof b === "string")
  ) {
    const aNumber = Number(a);
    const bNumber = Number(b);
    if (Number.isFinite(aNumber) && Number.isFinite(bNumber) && Math.abs(aNumber - bNumber) <= 1e-9) return true;
  }
  if (
    (typeof a === "boolean" || typeof a === "string" || typeof a === "number") &&
    (typeof b === "boolean" || typeof b === "string" || typeof b === "number")
  ) {
    try {
      const aBool = booleanFromMetadataValue(a, "config value");
      const bBool = booleanFromMetadataValue(b, "config value");
      if (aBool === bBool) return true;
    } catch {
      // Fall through to structural comparison.
    }
  }
  return JSON.stringify(a) === JSON.stringify(b);
}

function mergeLlamaSafetensorsConfigObjects(metadataConfig, optionConfig) {
  const merged = { ...metadataConfig };
  for (const [key, value] of Object.entries(optionConfig)) {
    if (Object.prototype.hasOwnProperty.call(merged, key) && !configValuesEquivalent(merged[key], value)) {
      throw new Error(`LLaMA safetensors config mismatch for ${key}: metadata=${JSON.stringify(merged[key])} != option=${JSON.stringify(value)}`);
    }
    merged[key] = value;
  }
  return merged;
}

function llamaConfigObjectFromOptions(options, metadata = {}) {
  const optionConfig = configObjectFromFirstSource(
    options,
    ["config", "modelConfig", "configJson", "configJSON", "hfConfig", "huggingFaceConfig"],
    "LLaMA safetensors",
  );
  const metadataConfig = configObjectFromFirstSource(
    metadata,
    ["config", "model_config", "modelConfig", "config_json", "configJson", "configJSON", "hf_config", "hfConfig", "huggingface_config", "huggingFaceConfig"],
    "LLaMA safetensors metadata",
  );
  return mergeLlamaSafetensorsConfigObjects(metadataConfig, optionConfig);
}

function positiveIntegerFromMetadataValue(value, label) {
  if (value === undefined || value === null || value === "") return null;
  let parsed = value;
  if (typeof parsed === "string") {
    const trimmed = parsed.trim();
    if (!/^[0-9]+$/.test(trimmed)) {
      throw new Error(`${label} must be a positive integer, got ${JSON.stringify(value)}`);
    }
    parsed = Number(trimmed);
  }
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer, got ${value}`);
  }
  return parsed;
}

function positiveFiniteNumberFromMetadataValue(value, label) {
  if (value === undefined || value === null || value === "") return null;
  let parsed = value;
  if (typeof parsed === "string") {
    const trimmed = parsed.trim();
    if (trimmed.length === 0) return null;
    parsed = Number(trimmed);
  }
  if (typeof parsed !== "number" || !Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive finite number, got ${value}`);
  }
  return parsed;
}

function nonNegativeIntegerFromMetadataValue(value, label) {
  if (value === undefined || value === null || value === "") return null;
  let parsed = value;
  if (typeof parsed === "string") {
    const trimmed = parsed.trim();
    if (!/^[0-9]+$/.test(trimmed)) {
      throw new Error(`${label} must be a non-negative integer, got ${JSON.stringify(value)}`);
    }
    parsed = Number(trimmed);
  }
  if (!Number.isSafeInteger(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative integer, got ${value}`);
  }
  return parsed;
}

function probabilityFromMetadataValue(value, label) {
  if (value === undefined || value === null || value === "") return null;
  let parsed = value;
  if (typeof parsed === "string") {
    const trimmed = parsed.trim();
    if (trimmed.length === 0) return null;
    parsed = Number(trimmed);
  }
  if (typeof parsed !== "number" || !Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
    throw new Error(`${label} must be a probability in [0, 1], got ${JSON.stringify(value)}`);
  }
  return parsed;
}

function booleanFromMetadataValue(value, label) {
  if (value === undefined || value === null || value === "") return null;
  if (typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (value === 0) return false;
    if (value === 1) return true;
  }
  if (typeof value === "string") {
    const trimmed = value.trim().toLowerCase();
    if (trimmed === "true" || trimmed === "1" || trimmed === "yes") return true;
    if (trimmed === "false" || trimmed === "0" || trimmed === "no") return false;
  }
  throw new Error(`${label} must be a boolean, got ${JSON.stringify(value)}`);
}

function firstPositiveIntegerFromObject(source, names, label) {
  for (const name of names) {
    if (Object.prototype.hasOwnProperty.call(source, name)) {
      return positiveIntegerFromMetadataValue(source[name], `${label} ${name}`);
    }
  }
  return null;
}

function firstPositiveFiniteNumberFromObject(source, names, label) {
  for (const name of names) {
    if (Object.prototype.hasOwnProperty.call(source, name)) {
      return positiveFiniteNumberFromMetadataValue(source[name], `${label} ${name}`);
    }
  }
  return null;
}

function firstBooleanFromObject(source, names, label) {
  for (const name of names) {
    if (Object.prototype.hasOwnProperty.call(source, name)) {
      return booleanFromMetadataValue(source[name], `${label} ${name}`);
    }
  }
  return null;
}

function optionalBooleanFromObjectNames(source, names, label) {
  const values = [];
  for (const name of names) {
    if (Object.prototype.hasOwnProperty.call(source, name)) {
      values.push([name, booleanFromMetadataValue(source[name], `${label} ${name}`)]);
    }
  }
  return requireMatchingOptionalBooleans(label, values);
}

function resolveLlamaRequireGpuDispatch(source = {}, defaultValue = false) {
  const value = optionalBooleanFromObjectNames(source, [
    "requireGpuDispatch",
    "requireGpuExecution",
    "requireWebGpuDispatch",
    "requireWebGpuExecution",
    "requireGpu",
    "noFallback",
    "disallowFallback",
  ], "LLaMA resource execution");
  if (value !== null) return value;
  if (Object.prototype.hasOwnProperty.call(source, "allowFallback")) {
    return !booleanFromMetadataValue(source.allowFallback, "LLaMA resource execution allowFallback");
  }
  if (Object.prototype.hasOwnProperty.call(source, "allowMockFallback")) {
    return !booleanFromMetadataValue(source.allowMockFallback, "LLaMA resource execution allowMockFallback");
  }
  return defaultValue;
}

function firstStringFromObject(source, names, label) {
  for (const name of names) {
    if (Object.prototype.hasOwnProperty.call(source, name)) {
      const value = source[name];
      if (value === undefined || value === null || value === "") return null;
      if (typeof value !== "string") {
        throw new Error(`${label} ${name} must be a string, got ${JSON.stringify(value)}`);
      }
      return value;
    }
  }
  return null;
}

function firstStringArrayFromObject(source, names, label) {
  for (const name of names) {
    if (!Object.prototype.hasOwnProperty.call(source, name)) continue;
    const value = source[name];
    if (value === undefined || value === null || value === "") return null;
    const values = Array.isArray(value) ? value : [value];
    return values.map((entry, index) => {
      if (typeof entry !== "string" || entry.trim() === "") {
        throw new Error(`${label} ${name}[${index}] must be a string, got ${JSON.stringify(entry)}`);
      }
      return entry;
    });
  }
  return null;
}

function tokenizerMetadataArrayFromValue(value, label) {
  if (value === undefined || value === null || value === "") return null;
  let parsed = value;
  if (typeof parsed === "string") {
    const trimmed = parsed.trim();
    if (trimmed.length === 0) return null;
    try {
      parsed = JSON.parse(trimmed);
    } catch (_err) {
      throw new Error(`${label} must be a JSON array when encoded as a string`);
    }
  } else if (typeof ArrayBuffer !== "undefined" && parsed instanceof ArrayBuffer) {
    parsed = JSON.parse(new TextDecoder().decode(new Uint8Array(parsed)));
  } else if (typeof ArrayBuffer !== "undefined" && ArrayBuffer.isView(parsed)) {
    parsed = JSON.parse(new TextDecoder().decode(new Uint8Array(parsed.buffer, parsed.byteOffset, parsed.byteLength)));
  }
  if (!Array.isArray(parsed)) {
    throw new Error(`${label} must be an array`);
  }
  return parsed;
}

function finiteNumberFromMetadataValue(value, label) {
  if (value === undefined || value === null || value === "") return null;
  let parsed = value;
  if (typeof parsed === "string") {
    const trimmed = parsed.trim();
    if (trimmed.length === 0) return null;
    parsed = Number(trimmed);
  }
  if (typeof parsed !== "number" || !Number.isFinite(parsed)) {
    throw new Error(`${label} must be a finite number, got ${JSON.stringify(value)}`);
  }
  return parsed;
}

function tokenizerTokenArrayLengthFromValue(value, label) {
  const parsed = tokenizerMetadataArrayFromValue(value, label);
  if (parsed === null) return null;
  for (let index = 0; index < parsed.length; index += 1) {
    if (typeof parsed[index] !== "string") {
      throw new Error(`${label}[${index}] must be a string`);
    }
  }
  return parsed.length;
}

function tokenizerFiniteNumberArrayLengthFromValue(value, label) {
  const parsed = tokenizerMetadataArrayFromValue(value, label);
  if (parsed === null) return null;
  for (let index = 0; index < parsed.length; index += 1) {
    finiteNumberFromMetadataValue(parsed[index], `${label}[${index}]`);
  }
  return parsed.length;
}

function tokenizerNonNegativeIntegerArrayLengthFromValue(value, label) {
  const parsed = tokenizerMetadataArrayFromValue(value, label);
  if (parsed === null) return null;
  for (let index = 0; index < parsed.length; index += 1) {
    nonNegativeIntegerFromMetadataValue(parsed[index], `${label}[${index}]`);
  }
  return parsed.length;
}

function firstTokenizerTokenArrayLengthFromObject(source, names, label) {
  for (const name of names) {
    if (!Object.prototype.hasOwnProperty.call(source, name)) continue;
    return tokenizerTokenArrayLengthFromValue(source[name], `${label} ${name}`);
  }
  return null;
}

function firstTokenizerFiniteNumberArrayLengthFromObject(source, names, label) {
  for (const name of names) {
    if (!Object.prototype.hasOwnProperty.call(source, name)) continue;
    return tokenizerFiniteNumberArrayLengthFromValue(source[name], `${label} ${name}`);
  }
  return null;
}

function firstTokenizerNonNegativeIntegerArrayLengthFromObject(source, names, label) {
  for (const name of names) {
    if (!Object.prototype.hasOwnProperty.call(source, name)) continue;
    return tokenizerNonNegativeIntegerArrayLengthFromValue(source[name], `${label} ${name}`);
  }
  return null;
}

function firstConfigValueFromObject(source, names) {
  for (const name of names) {
    if (Object.prototype.hasOwnProperty.call(source, name)) return source[name];
  }
  return undefined;
}

function disabledConfigFeatureValue(value) {
  if (value === undefined || value === null || value === "") return true;
  if (typeof value === "boolean") return value === false;
  if (typeof value === "number") return value === 0;
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    return normalized === "" || normalized === "0" || normalized === "false" || normalized === "none" || normalized === "null";
  }
  return false;
}

function validateConfigTokenIds(source, names, label) {
  for (const name of names) {
    if (!Object.prototype.hasOwnProperty.call(source, name)) continue;
    const value = source[name];
    if (value === undefined || value === null || value === "") return;
    const values = Array.isArray(value) ? value : [value];
    if (values.length === 0) {
      throw new Error(`${label} ${name} must contain at least one token id`);
    }
    for (let index = 0; index < values.length; index += 1) {
      const suffix = Array.isArray(value) ? `[${index}]` : "";
      nonNegativeIntegerFromMetadataValue(values[index], `${label} ${name}${suffix}`);
    }
    return;
  }
}

function firstTokenIdsFromObject(source, names, label) {
  for (const name of names) {
    if (!Object.prototype.hasOwnProperty.call(source, name)) continue;
    const value = source[name];
    if (value === undefined || value === null || value === "") return null;
    const values = Array.isArray(value) ? value : [value];
    if (values.length === 0) {
      throw new Error(`${label} ${name} must contain at least one token id`);
    }
    return values.map((entry, index) => {
      const suffix = Array.isArray(value) ? `[${index}]` : "";
      return nonNegativeIntegerFromMetadataValue(entry, `${label} ${name}${suffix}`);
    });
  }
  return null;
}

function validateConfigProbabilities(source, names, label) {
  for (const name of names) {
    if (!Object.prototype.hasOwnProperty.call(source, name)) continue;
    probabilityFromMetadataValue(source[name], `${label} ${name}`);
  }
}

function firstProbabilityFromObject(source, names, label) {
  for (const name of names) {
    if (!Object.prototype.hasOwnProperty.call(source, name)) continue;
    return probabilityFromMetadataValue(source[name], `${label} ${name}`);
  }
  return null;
}

function normalizeLlamaConfigFamilyName(value) {
  return String(value).trim().toLowerCase().replaceAll("-", "_");
}

function isSupportedLlamaConfigModelType(value) {
  const normalized = normalizeLlamaConfigFamilyName(value);
  return normalized === "llama" || normalized === "mistral" || normalized === "qwen2" || normalized === "qwen3" || normalized === "smollm3";
}

function isSupportedLlamaConfigArchitecture(value) {
  const normalized = normalizeLlamaConfigFamilyName(value);
  return (
    normalized === "llamaforcausallm" ||
    normalized === "llama_for_causal_lm" ||
    normalized === "mistralforcausallm" ||
    normalized === "mistral_for_causal_lm" ||
    normalized === "qwen2forcausallm" ||
    normalized === "qwen2_for_causal_lm" ||
    normalized === "qwen3forcausallm" ||
    normalized === "qwen3_for_causal_lm" ||
    normalized === "smollm3forcausallm" ||
    normalized === "smollm3_for_causal_lm"
  );
}

const LLAMA_GGUF_METADATA_PREFIXES = Object.freeze(["llama", "mistral", "qwen2", "qwen3", "smollm3"]);

function llamaGgufMetadataNames(...suffixes) {
  return suffixes.flatMap((suffix) => LLAMA_GGUF_METADATA_PREFIXES.map((prefix) => `${prefix}.${suffix}`));
}

const LLAMA_SAFETENSORS_METADATA_MODEL_TYPE_NAMES = [
  "model_type",
  "modelType",
  "general.architecture",
];

function normalizeZgmlModelKind(value, label = "zgml model kind") {
  if (value === undefined || value === null) return null;
  if (!Number.isSafeInteger(value) || value <= 0 || value > 0xffffffff) {
    throw new Error(`invalid ${label}: ${value}`);
  }
  return value;
}

function llamaConfigFamilyFromSources(metadata, config) {
  const modelType = firstStringFromObject(
    config,
    ["model_type", "modelType"],
    "LLaMA safetensors config",
  ) ?? firstStringFromObject(
    metadata,
    LLAMA_SAFETENSORS_METADATA_MODEL_TYPE_NAMES,
    "LLaMA safetensors metadata",
  );
  if (modelType !== null) return normalizeLlamaConfigFamilyName(modelType);
  const architectures = firstStringArrayFromObject(
    config,
    ["architectures", "architecture"],
    "LLaMA safetensors config",
  ) ?? firstStringArrayFromObject(
    metadata,
    ["architectures", "architecture"],
    "LLaMA safetensors metadata",
  );
  const architecture = architectures?.[0];
  return architecture === undefined ? null : normalizeLlamaConfigFamilyName(architecture);
}

function inferZgmlModelKindFromLlamaSafetensors(options, metadata, config, layerCount) {
  const explicit = normalizeZgmlModelKind(
    options.modelKind ?? options.model_kind ?? options.kind,
    "LLaMA safetensors model kind",
  );
  if (explicit !== null) return explicit;
  const family = llamaConfigFamilyFromSources(metadata, config);
  if (family === "smollm3" || family === "smollm3forcausallm" || family === "smollm3_for_causal_lm") {
    return zgmlModelSmollm135mKind;
  }
  if (layerCount === 2) return zgmlModelTinyLlama2LayerKind;
  return zgmlModelTinyLlamaKind;
}

const LLAMA_QK_PROJECTION_NORM_OPTION_NAMES = [
  "qkNorm",
  "qk_norm",
  "useQkNorm",
  "use_qk_norm",
  "qkLayerNorm",
  "qkLayernorm",
  "qk_layer_norm",
  "qk_layernorm",
  "queryKeyNorm",
  "query_key_norm",
  "queryKeyLayerNorm",
  "query_key_layer_norm",
  "query_key_layernorm",
  "qNorm",
  "q_norm",
  "qLayerNorm",
  "q_layer_norm",
  "q_layernorm",
  "queryNorm",
  "query_norm",
  "queryLayerNorm",
  "query_layer_norm",
  "query_layernorm",
  "kNorm",
  "k_norm",
  "kLayerNorm",
  "k_layer_norm",
  "k_layernorm",
  "keyNorm",
  "key_norm",
  "keyLayerNorm",
  "key_layer_norm",
  "key_layernorm",
];

const LLAMA_QK_PROJECTION_NORM_METADATA_NAMES = [
  ...LLAMA_QK_PROJECTION_NORM_OPTION_NAMES,
  ...llamaGgufMetadataNames(
    "attention.qk_norm",
    "attention.query_key_norm",
    "attention.query_norm",
    "attention.key_norm",
  ),
  "zgml.qk_norm",
];

function normalizeLlamaHiddenActivation(value) {
  if (value === undefined || value === null || value === "") return null;
  const normalized = String(value).trim().toLowerCase();
  if (normalized === "silu" || normalized === "swish") return "silu";
  return normalized;
}

function normalizeLlamaTorchDtype(value) {
  if (value === undefined || value === null || value === "") return null;
  let normalized = String(value).trim().toLowerCase();
  if (normalized.startsWith("torch.")) normalized = normalized.slice("torch.".length);
  normalized = normalized.replaceAll("-", "_");
  if (normalized === "auto") return null;
  if (normalized === "float" || normalized === "float32" || normalized === "f32") return "f32";
  if (normalized === "float16" || normalized === "half" || normalized === "f16") return "f16";
  if (normalized === "bfloat16" || normalized === "bf16") return "bf16";
  return normalized;
}

function optionalPositiveFiniteNumberFromObject(source, names, label) {
  const value = firstConfigValueFromObject(source, names);
  return value === undefined ? null : positiveFiniteNumberFromMetadataValue(value, label);
}

function optionalPositiveIntegerFromObject(source, names, label) {
  const value = firstConfigValueFromObject(source, names);
  return value === undefined ? null : positiveIntegerFromMetadataValue(value, label);
}

function optionalNonNegativeIntegerFromObject(source, names, label) {
  const value = firstConfigValueFromObject(source, names);
  if (value === undefined || disabledConfigFeatureValue(value)) return null;
  return nonNegativeIntegerFromMetadataValue(value, label);
}

function useRopeLayerFlagFromValue(value, label) {
  const parsed = booleanFromMetadataValue(value, label);
  if (parsed === null) {
    throw new Error(`${label} must be 0/1 or boolean, got ${JSON.stringify(value)}`);
  }
  return parsed ? 1 : 0;
}

function useRopeLayerFlagsFromValue(value, layerCount, label) {
  if (value === undefined || value === null || value === "") return null;
  let values = value;
  if (typeof values === "string") {
    const trimmed = values.trim();
    if (disabledConfigFeatureValue(trimmed)) return null;
    if (trimmed.startsWith("[")) {
      try {
        values = JSON.parse(trimmed);
      } catch (_err) {
        throw new Error(`${label} must be a JSON array or array-like value`);
      }
    } else {
      values = trimmed.split(",").map((entry) => entry.trim()).filter((entry) => entry.length !== 0);
    }
  }
  if (!Array.isArray(values)) {
    throw new Error(`${label} must be an array`);
  }
  if (values.length < layerCount) {
    throw new Error(`${label} must contain at least ${layerCount} entries`);
  }
  return values.slice(0, layerCount).map((entry, index) => useRopeLayerFlagFromValue(entry, `${label}[${index}]`));
}

function optionalUseRopeLayerFlagsFromObject(source, names, layerCount, label) {
  const value = firstConfigValueFromObject(source, names);
  return useRopeLayerFlagsFromValue(value, layerCount, label);
}

function useRopeLayerFlagsFromInterval(interval, layerCount) {
  return Array.from({ length: layerCount }, (_unused, layer) => ((layer + 1) % interval === 0 ? 0 : 1));
}

function parseRopeScalingConfigValue(value) {
  if (disabledConfigFeatureValue(value)) {
    return {
      highFreqFactor: 1,
      kind: "none",
      lowFreqFactor: 1,
      originalContextLength: 1,
      scale: 1,
    };
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const allowedKeys = new Set([
    "factor",
    "highFreqFactor",
    "high_freq_factor",
    "lowFreqFactor",
    "low_freq_factor",
    "name",
    "originalMaxPositionEmbeddings",
    "original_max_position_embeddings",
    "ropeType",
    "rope_type",
    "type",
  ]);
  for (const key of Object.keys(value)) {
    if (!allowedKeys.has(key)) return false;
  }
  const typeValue = firstConfigValueFromObject(value, ["rope_type", "ropeType", "type", "name"]);
  let ropeType = null;
  if (typeValue !== undefined && typeValue !== null && typeValue !== "") {
    if (typeof typeValue !== "string") return false;
    ropeType = normalizeLlamaConfigFamilyName(typeValue);
  }
  const factor = optionalPositiveFiniteNumberFromObject(
    value,
    ["factor"],
    "LLaMA safetensors config rope scaling factor",
  );
  const lowFreqFactor = optionalPositiveFiniteNumberFromObject(
    value,
    ["low_freq_factor", "lowFreqFactor"],
    "LLaMA safetensors config rope scaling low frequency factor",
  );
  const highFreqFactor = optionalPositiveFiniteNumberFromObject(
    value,
    ["high_freq_factor", "highFreqFactor"],
    "LLaMA safetensors config rope scaling high frequency factor",
  );
  const originalContextLength = optionalPositiveIntegerFromObject(
    value,
    ["original_max_position_embeddings", "originalMaxPositionEmbeddings"],
    "LLaMA safetensors config rope scaling original max position embeddings",
  );
  if (ropeType === null || ropeType === "default" || ropeType === "none") {
    if (factor !== null && Math.abs(factor - 1) > 1e-6) return false;
    if (lowFreqFactor !== null && Math.abs(lowFreqFactor - 1) > 1e-6) return false;
    if (highFreqFactor !== null && Math.abs(highFreqFactor - 1) > 1e-6) return false;
    return {
      highFreqFactor: 1,
      kind: "none",
      lowFreqFactor: 1,
      originalContextLength: originalContextLength ?? 1,
      scale: 1,
    };
  }
  if (ropeType === "linear") {
    if (lowFreqFactor !== null && Math.abs(lowFreqFactor - 1) > 1e-6) return false;
    if (highFreqFactor !== null && Math.abs(highFreqFactor - 1) > 1e-6) return false;
    return {
      highFreqFactor: 1,
      kind: "linear",
      lowFreqFactor: 1,
      originalContextLength: originalContextLength ?? 1,
      scale: factor ?? 1,
    };
  }
  if (ropeType === "llama3" || ropeType === "llama_3") {
    if (factor === null || lowFreqFactor === null || highFreqFactor === null || originalContextLength === null) return false;
    if (highFreqFactor <= lowFreqFactor) return false;
    return {
      highFreqFactor,
      kind: "llama3",
      lowFreqFactor,
      originalContextLength,
      scale: factor,
    };
  }
  if (ropeType === "dynamic" && (factor === null || Math.abs(factor - 1) <= 1e-6)) {
    if (lowFreqFactor !== null && Math.abs(lowFreqFactor - 1) > 1e-6) return false;
    if (highFreqFactor !== null && Math.abs(highFreqFactor - 1) > 1e-6) return false;
    return {
      highFreqFactor: 1,
      kind: "none",
      lowFreqFactor: 1,
      originalContextLength: originalContextLength ?? 1,
      scale: 1,
    };
  }
  return false;
}

function supportedRopeScalingConfigValue(value) {
  return parseRopeScalingConfigValue(value) !== false;
}

function ropeScaleFromRopeScalingConfigValue(value) {
  const parsed = parseRopeScalingConfigValue(value);
  if (parsed === true || parsed === false) return null;
  return parsed.scale;
}

function normalizeLlamaRopeKind(value) {
  if (value === undefined || value === null || value === "") return "none";
  const normalized = normalizeLlamaConfigFamilyName(value);
  if (normalized === "default" || normalized === "none") return "none";
  if (normalized === "linear") return "linear";
  if (normalized === "llama3" || normalized === "llama_3") return "llama3";
  throw new Error(`unsupported LLaMA RoPE kind: ${value}`);
}

function llamaRopeKindId(kind) {
  switch (normalizeLlamaRopeKind(kind)) {
    case "none":
      return 0;
    case "linear":
      return 1;
    case "llama3":
      return 2;
    default:
      throw new Error(`unsupported LLaMA RoPE kind: ${kind}`);
  }
}

function isSupportedLlamaConfigTorchDtype(value) {
  let normalized = String(value).trim().toLowerCase();
  if (normalized.startsWith("torch.")) normalized = normalized.slice("torch.".length);
  normalized = normalized.replaceAll("-", "_");
  return (
    normalized === "auto" ||
    normalized === "float" ||
    normalized === "float32" ||
    normalized === "f32" ||
    normalized === "float16" ||
    normalized === "half" ||
    normalized === "f16" ||
    normalized === "bfloat16" ||
    normalized === "bf16"
  );
}

function rejectUnsupportedLlamaSafetensorsConfig(config, contextLength) {
  const modelType = firstStringFromObject(
    config,
    ["model_type", "modelType"],
    "LLaMA safetensors config",
  );
  if (modelType !== null && !isSupportedLlamaConfigModelType(modelType)) {
    throw new Error(`unsupported LLaMA safetensors config model type: ${modelType}`);
  }

  const architectures = firstStringArrayFromObject(
    config,
    ["architectures", "architecture"],
    "LLaMA safetensors config",
  );
  if (architectures !== null && architectures.length > 0 && !architectures.some(isSupportedLlamaConfigArchitecture)) {
    throw new Error(`unsupported LLaMA safetensors config architectures: ${architectures.join(",")}`);
  }

  const hiddenAct = firstStringFromObject(
    config,
    ["hidden_act", "hiddenActivation", "activation_function", "activationFunction"],
    "LLaMA safetensors config",
  );
  if (hiddenAct !== null) {
    const normalized = hiddenAct.trim().toLowerCase();
    if (normalized !== "silu" && normalized !== "swish") {
      throw new Error(`unsupported LLaMA safetensors config hidden activation: ${hiddenAct}`);
    }
  }

  const torchDtype = firstStringFromObject(
    config,
    ["torch_dtype", "torchDtype"],
    "LLaMA safetensors config",
  );
  if (torchDtype !== null && !isSupportedLlamaConfigTorchDtype(torchDtype)) {
    throw new Error(`unsupported LLaMA safetensors config torch dtype: ${torchDtype}`);
  }

  firstBooleanFromObject(
    config,
    ["use_cache", "useCache"],
    "LLaMA safetensors config",
  );
  validateConfigTokenIds(config, ["bos_token_id", "bosTokenId"], "LLaMA safetensors config");
  validateConfigTokenIds(config, ["eos_token_id", "eosTokenId"], "LLaMA safetensors config");
  validateConfigTokenIds(config, ["pad_token_id", "padTokenId"], "LLaMA safetensors config");
  validateConfigProbabilities(
    config,
    ["attention_dropout", "attentionDropout", "attn_pdrop", "attnPdrop"],
    "LLaMA safetensors config",
  );
  validateConfigProbabilities(
    config,
    ["hidden_dropout", "hiddenDropout", "hidden_dropout_prob", "hiddenDropoutProb"],
    "LLaMA safetensors config",
  );

  const attentionBias = firstBooleanFromObject(
    config,
    ["attention_bias", "attentionBias", "qkv_bias", "qkvBias"],
    "LLaMA safetensors config",
  );

  const mlpBias = firstBooleanFromObject(
    config,
    ["mlp_bias", "mlpBias", "ffn_bias", "ffnBias"],
    "LLaMA safetensors config",
  );

  const useSlidingWindow = firstBooleanFromObject(
    config,
    ["use_sliding_window", "useSlidingWindow"],
    "LLaMA safetensors config",
  );
  const slidingWindowValue = firstConfigValueFromObject(config, ["sliding_window", "slidingWindow"]);
  const slidingWindow = disabledConfigFeatureValue(slidingWindowValue)
    ? null
    : positiveIntegerFromMetadataValue(slidingWindowValue, "LLaMA safetensors config sliding window");
  if (useSlidingWindow === true && slidingWindow === null) {
    throw new Error("unsupported LLaMA safetensors config sliding window attention");
  }

  const ropeScaling = firstConfigValueFromObject(config, ["rope_scaling", "ropeScaling"]);
  if (!supportedRopeScalingConfigValue(ropeScaling)) {
    throw new Error("unsupported LLaMA safetensors config rope scaling");
  }

  const partialRotaryFactor = firstPositiveFiniteNumberFromObject(
    config,
    ["partial_rotary_factor", "partialRotaryFactor"],
    "LLaMA safetensors config",
  );
  if (partialRotaryFactor !== null && Math.abs(partialRotaryFactor - 1) > 1e-6) {
    throw new Error(`unsupported LLaMA safetensors config partial rotary factor: ${partialRotaryFactor}`);
  }

  firstPositiveIntegerFromObject(
    config,
    ["pretraining_tp", "pretrainingTensorParallelism", "pretrainingTensorParallelSize"],
    "LLaMA safetensors config",
  );

  const maxPositionEmbeddings = firstPositiveIntegerFromObject(
    config,
    ["max_position_embeddings", "maxPositionEmbeddings", "seq_length", "seqLength", "n_ctx", "context_length"],
    "LLaMA safetensors config",
  );
  if (maxPositionEmbeddings !== null && contextLength > maxPositionEmbeddings) {
    throw new Error(`LLaMA safetensors context length ${contextLength} exceeds config maximum position embeddings ${maxPositionEmbeddings}`);
  }
}

function rejectUnsupportedLlamaSafetensorsMetadataSemantics(metadata) {
  const modelType = firstStringFromObject(
    metadata,
    LLAMA_SAFETENSORS_METADATA_MODEL_TYPE_NAMES,
    "LLaMA safetensors metadata",
  );
  if (modelType !== null && !isSupportedLlamaConfigModelType(modelType)) {
    throw new Error(`unsupported LLaMA safetensors metadata model type: ${modelType}`);
  }

  const architectures = firstStringArrayFromObject(
    metadata,
    ["architectures", "architecture"],
    "LLaMA safetensors metadata",
  );
  if (architectures !== null && architectures.length > 0 && !architectures.some(isSupportedLlamaConfigArchitecture)) {
    throw new Error(`unsupported LLaMA safetensors metadata architectures: ${architectures.join(",")}`);
  }

  const hiddenAct = firstStringFromObject(
    metadata,
    ["hidden_act", "hiddenActivation", "activation_function", "activationFunction", ...llamaGgufMetadataNames("feed_forward.activation"), "zgml.hidden_act"],
    "LLaMA safetensors metadata",
  );
  if (hiddenAct !== null) {
    const normalized = hiddenAct.trim().toLowerCase();
    if (normalized !== "silu" && normalized !== "swish") {
      throw new Error(`unsupported LLaMA safetensors metadata hidden activation: ${hiddenAct}`);
    }
  }

  const torchDtype = firstStringFromObject(
    metadata,
    ["torch_dtype", "torchDtype"],
    "LLaMA safetensors metadata",
  );
  if (torchDtype !== null && !isSupportedLlamaConfigTorchDtype(torchDtype)) {
    throw new Error(`unsupported LLaMA safetensors metadata torch dtype: ${torchDtype}`);
  }

  firstBooleanFromObject(
    metadata,
    ["use_cache", "useCache"],
    "LLaMA safetensors metadata",
  );
  validateConfigTokenIds(metadata, ["bos_token_id", "bosTokenId", "tokenizer.bos_token_id", "tokenizer.ggml.bos_token_id"], "LLaMA safetensors metadata");
  validateConfigTokenIds(metadata, ["eos_token_id", "eosTokenId", "tokenizer.eos_token_id", "tokenizer.ggml.eos_token_id"], "LLaMA safetensors metadata");
  validateConfigTokenIds(metadata, ["pad_token_id", "padTokenId", "tokenizer.pad_token_id", "tokenizer.ggml.pad_token_id"], "LLaMA safetensors metadata");
  validateConfigProbabilities(
    metadata,
    ["attention_dropout", "attentionDropout", "attn_pdrop", "attnPdrop"],
    "LLaMA safetensors metadata",
  );
  validateConfigProbabilities(
    metadata,
    ["hidden_dropout", "hiddenDropout", "hidden_dropout_prob", "hiddenDropoutProb"],
    "LLaMA safetensors metadata",
  );

  const attentionBias = firstBooleanFromObject(
    metadata,
    ["attention_bias", "attentionBias", "qkv_bias", "qkvBias", ...llamaGgufMetadataNames("attention.bias"), "zgml.attention_bias"],
    "LLaMA safetensors metadata",
  );

  const mlpBias = firstBooleanFromObject(
    metadata,
    ["mlp_bias", "mlpBias", "ffn_bias", "ffnBias", ...llamaGgufMetadataNames("feed_forward.bias"), "zgml.mlp_bias"],
    "LLaMA safetensors metadata",
  );

  const partialRotaryFactor = firstPositiveFiniteNumberFromObject(
    metadata,
    ["partial_rotary_factor", "partialRotaryFactor"],
    "LLaMA safetensors metadata",
  );
  if (partialRotaryFactor !== null && Math.abs(partialRotaryFactor - 1) > 1e-6) {
    throw new Error(`unsupported LLaMA safetensors metadata partial rotary factor: ${partialRotaryFactor}`);
  }

  firstPositiveIntegerFromObject(
    metadata,
    ["pretraining_tp", "pretrainingTensorParallelism", "pretrainingTensorParallelSize"],
    "LLaMA safetensors metadata",
  );
}

function validateLlamaSafetensorsMetadataConfigSemantics(metadata, config) {
  const metadataModelType = firstStringFromObject(
    metadata,
    LLAMA_SAFETENSORS_METADATA_MODEL_TYPE_NAMES,
    "LLaMA safetensors metadata",
  );
  const configModelType = firstStringFromObject(
    config,
    ["model_type", "modelType"],
    "LLaMA safetensors config",
  );
  requireMatchingOptionalStrings("LLaMA safetensors model type", [
    ["metadata", metadataModelType === null ? null : normalizeLlamaConfigFamilyName(metadataModelType)],
    ["config", configModelType === null ? null : normalizeLlamaConfigFamilyName(configModelType)],
  ]);

  const metadataArchitectures = firstStringArrayFromObject(
    metadata,
    ["architectures", "architecture"],
    "LLaMA safetensors metadata",
  );
  const configArchitectures = firstStringArrayFromObject(
    config,
    ["architectures", "architecture"],
    "LLaMA safetensors config",
  );
  requireMatchingOptionalStringArrays("LLaMA safetensors architectures", [
    ["metadata", metadataArchitectures === null ? null : metadataArchitectures.map(normalizeLlamaConfigFamilyName).sort()],
    ["config", configArchitectures === null ? null : configArchitectures.map(normalizeLlamaConfigFamilyName).sort()],
  ]);

  const metadataHiddenAct = firstStringFromObject(
    metadata,
    ["hidden_act", "hiddenActivation", "activation_function", "activationFunction", ...llamaGgufMetadataNames("feed_forward.activation"), "zgml.hidden_act"],
    "LLaMA safetensors metadata",
  );
  const configHiddenAct = firstStringFromObject(
    config,
    ["hidden_act", "hiddenActivation", "activation_function", "activationFunction"],
    "LLaMA safetensors config",
  );
  requireMatchingOptionalStrings("LLaMA safetensors hidden activation", [
    ["metadata", normalizeLlamaHiddenActivation(metadataHiddenAct)],
    ["config", normalizeLlamaHiddenActivation(configHiddenAct)],
  ]);

  const metadataTorchDtype = firstStringFromObject(
    metadata,
    ["torch_dtype", "torchDtype"],
    "LLaMA safetensors metadata",
  );
  const configTorchDtype = firstStringFromObject(
    config,
    ["torch_dtype", "torchDtype"],
    "LLaMA safetensors config",
  );
  requireMatchingOptionalStrings("LLaMA safetensors torch dtype", [
    ["metadata", normalizeLlamaTorchDtype(metadataTorchDtype)],
    ["config", normalizeLlamaTorchDtype(configTorchDtype)],
  ]);

  const metadataUseCache = firstBooleanFromObject(
    metadata,
    ["use_cache", "useCache"],
    "LLaMA safetensors metadata",
  );
  const configUseCache = firstBooleanFromObject(
    config,
    ["use_cache", "useCache"],
    "LLaMA safetensors config",
  );
  requireMatchingOptionalBooleans("LLaMA safetensors use_cache", [
    ["metadata", metadataUseCache],
    ["config", configUseCache],
  ]);

  const metadataPretrainingTensorParallelism = firstPositiveIntegerFromObject(
    metadata,
    ["pretraining_tp", "pretrainingTensorParallelism", "pretrainingTensorParallelSize"],
    "LLaMA safetensors metadata",
  );
  const configPretrainingTensorParallelism = firstPositiveIntegerFromObject(
    config,
    ["pretraining_tp", "pretrainingTensorParallelism", "pretrainingTensorParallelSize"],
    "LLaMA safetensors config",
  );
  requireMatchingOptionalIntegers("LLaMA safetensors pretraining tensor parallelism", [
    ["metadata", metadataPretrainingTensorParallelism],
    ["config", configPretrainingTensorParallelism],
  ]);

  for (const tokenIdProof of [
    {
      label: "bos_token_id",
      metadataNames: ["bos_token_id", "bosTokenId", "tokenizer.bos_token_id", "tokenizer.ggml.bos_token_id"],
      configNames: ["bos_token_id", "bosTokenId"],
    },
    {
      label: "eos_token_id",
      metadataNames: ["eos_token_id", "eosTokenId", "tokenizer.eos_token_id", "tokenizer.ggml.eos_token_id"],
      configNames: ["eos_token_id", "eosTokenId"],
    },
    {
      label: "pad_token_id",
      metadataNames: ["pad_token_id", "padTokenId", "tokenizer.pad_token_id", "tokenizer.ggml.pad_token_id"],
      configNames: ["pad_token_id", "padTokenId"],
    },
  ]) {
    requireMatchingOptionalIntegerArrays(`LLaMA safetensors ${tokenIdProof.label}`, [
      ["metadata", firstTokenIdsFromObject(metadata, tokenIdProof.metadataNames, "LLaMA safetensors metadata")],
      ["config", firstTokenIdsFromObject(config, tokenIdProof.configNames, "LLaMA safetensors config")],
    ]);
  }

  for (const probabilityProof of [
    {
      label: "attention_dropout",
      names: ["attention_dropout", "attentionDropout", "attn_pdrop", "attnPdrop"],
    },
    {
      label: "hidden_dropout",
      names: ["hidden_dropout", "hiddenDropout", "hidden_dropout_prob", "hiddenDropoutProb"],
    },
  ]) {
    requireMatchingOptionalNumbers(`LLaMA safetensors ${probabilityProof.label}`, [
      ["metadata", firstProbabilityFromObject(metadata, probabilityProof.names, "LLaMA safetensors metadata")],
      ["config", firstProbabilityFromObject(config, probabilityProof.names, "LLaMA safetensors config")],
    ]);
  }

  const metadataAttentionBias = firstBooleanFromObject(
    metadata,
    ["attention_bias", "attentionBias", "qkv_bias", "qkvBias", ...llamaGgufMetadataNames("attention.bias"), "zgml.attention_bias"],
    "LLaMA safetensors metadata",
  );
  const configAttentionBias = firstBooleanFromObject(
    config,
    ["attention_bias", "attentionBias", "qkv_bias", "qkvBias"],
    "LLaMA safetensors config",
  );
  requireMatchingOptionalBooleans("LLaMA safetensors attention bias", [
    ["metadata", metadataAttentionBias],
    ["config", configAttentionBias],
  ]);

  const metadataMlpBias = firstBooleanFromObject(
    metadata,
    ["mlp_bias", "mlpBias", "ffn_bias", "ffnBias", ...llamaGgufMetadataNames("feed_forward.bias"), "zgml.mlp_bias"],
    "LLaMA safetensors metadata",
  );
  const configMlpBias = firstBooleanFromObject(
    config,
    ["mlp_bias", "mlpBias", "ffn_bias", "ffnBias"],
    "LLaMA safetensors config",
  );
  requireMatchingOptionalBooleans("LLaMA safetensors MLP bias", [
    ["metadata", metadataMlpBias],
    ["config", configMlpBias],
  ]);
}

const LLAMA_SAFETENSORS_CONTEXT_LENGTH_OPTION_NAMES = [
  "contextLength",
  "context_length",
  "maxPositionEmbeddings",
  "max_position_embeddings",
  "seqLength",
  "seq_length",
  "nCtx",
  "n_ctx",
];

const LLAMA_SAFETENSORS_METADATA_MAX_POSITION_NAMES = [
  "max_position_embeddings",
  "maxPositionEmbeddings",
  "seq_length",
  "seqLength",
  "n_ctx",
  "nCtx",
  "context_length",
  "contextLength",
  ...llamaGgufMetadataNames("context_length"),
  "zgml.max_position_embeddings",
];

const LLAMA_SAFETENSORS_CONFIG_MAX_POSITION_NAMES = [
  "max_position_embeddings",
  "maxPositionEmbeddings",
  "seq_length",
  "seqLength",
  "n_ctx",
  "nCtx",
  "context_length",
  "contextLength",
];

function resolveLlamaSafetensorsMaxPositionEmbeddings(metadata, config) {
  const metadataMaxPositionEmbeddings = firstPositiveIntegerFromObject(
    metadata,
    LLAMA_SAFETENSORS_METADATA_MAX_POSITION_NAMES,
    "LLaMA safetensors metadata",
  );
  const configMaxPositionEmbeddings = firstPositiveIntegerFromObject(
    config,
    LLAMA_SAFETENSORS_CONFIG_MAX_POSITION_NAMES,
    "LLaMA safetensors config",
  );
  return requireMatchingOptionalIntegers("LLaMA safetensors maximum position embeddings", [
    ["metadata", metadataMaxPositionEmbeddings],
    ["config", configMaxPositionEmbeddings],
  ]);
}

function resolveLlamaSafetensorsContextLength(options, metadata, config) {
  const explicitContextLength = firstPositiveIntegerFromObject(
    options,
    LLAMA_SAFETENSORS_CONTEXT_LENGTH_OPTION_NAMES,
    "LLaMA safetensors Program context length option",
  );
  if (explicitContextLength !== null) return explicitContextLength;

  const maxPositionEmbeddings = resolveLlamaSafetensorsMaxPositionEmbeddings(metadata, config);
  if (maxPositionEmbeddings !== null) return maxPositionEmbeddings;

  throw new Error("invalid LLaMA safetensors Program context length: missing contextLength or max_position_embeddings");
}

function validateLlamaSafetensorsContextEnvelope(options, metadata, config, contextLength) {
  const maxPositionEmbeddings = resolveLlamaSafetensorsMaxPositionEmbeddings(metadata, config);
  if (maxPositionEmbeddings !== null && contextLength > maxPositionEmbeddings) {
    throw new Error(`LLaMA safetensors context length ${contextLength} exceeds maximum position embeddings ${maxPositionEmbeddings}`);
  }
  const originalMaxPositionEmbeddings = firstPositiveIntegerFromObject(
    metadata,
    ["original_max_position_embeddings", "originalMaxPositionEmbeddings", ...llamaGgufMetadataNames("rope.original_context_length"), "zgml.rope_original_context_length"],
    "LLaMA safetensors metadata",
  );
  if (originalMaxPositionEmbeddings !== null && maxPositionEmbeddings !== null && originalMaxPositionEmbeddings > maxPositionEmbeddings) {
    throw new Error(`LLaMA safetensors original max position embeddings ${originalMaxPositionEmbeddings} exceeds maximum position embeddings ${maxPositionEmbeddings}`);
  }
}

function resolveLlamaSafetensorsQkProjectionNorm(options, metadata, config) {
  const qkProjectionNorm = requireMatchingOptionalBooleans("LLaMA safetensors Q/K projection normalization", [
    [
      "option",
      optionalBooleanFromObjectNames(
        options,
        LLAMA_QK_PROJECTION_NORM_OPTION_NAMES,
        "LLaMA safetensors option Q/K projection normalization",
      ),
    ],
    [
      "metadata",
      optionalBooleanFromObjectNames(
        metadata,
        LLAMA_QK_PROJECTION_NORM_METADATA_NAMES,
        "LLaMA safetensors metadata Q/K projection normalization",
      ),
    ],
    [
      "config",
      optionalBooleanFromObjectNames(
        config,
        LLAMA_QK_PROJECTION_NORM_OPTION_NAMES,
        "LLaMA safetensors config Q/K projection normalization",
      ),
    ],
  ]);
  return qkProjectionNorm ?? false;
}

function inferLlamaSafetensorsQkProjectionNormFromTensors(specs) {
  const qNormLayers = new Set();
  const kNormLayers = new Set();
  for (const spec of specs) {
    const name = spec.tensor ?? spec.name ?? "";
    if (typeof name !== "string") continue;
    const match = /^model\.layers\.(\d+)\.self_attn\.(q_norm|k_norm)\.weight$/.exec(name);
    if (!match) continue;
    const layer = Number(match[1]);
    if (match[2] === "q_norm") {
      qNormLayers.add(layer);
    } else {
      kNormLayers.add(layer);
    }
  }
  if (qNormLayers.size === 0 && kNormLayers.size === 0) return false;
  for (const layer of qNormLayers) {
    if (!kNormLayers.has(layer)) {
      throw new Error(`LLaMA safetensors Q/K projection normalization tensor mismatch: missing k_norm for layer ${layer}`);
    }
  }
  for (const layer of kNormLayers) {
    if (!qNormLayers.has(layer)) {
      throw new Error(`LLaMA safetensors Q/K projection normalization tensor mismatch: missing q_norm for layer ${layer}`);
    }
  }
  return true;
}

function resolveLlamaSafetensorsQkProjectionNormWithTensors(options, metadata, config, specs) {
  const semanticHint = resolveLlamaSafetensorsQkProjectionNorm(options, metadata, config);
  const hasSemanticHint = hasNonNullOwnProperty(options, LLAMA_QK_PROJECTION_NORM_OPTION_NAMES) ||
    hasNonNullOwnProperty(metadata, LLAMA_QK_PROJECTION_NORM_METADATA_NAMES) ||
    hasNonNullOwnProperty(config, LLAMA_QK_PROJECTION_NORM_OPTION_NAMES);
  const tensorHint = inferLlamaSafetensorsQkProjectionNormFromTensors(specs);
  if (hasSemanticHint && semanticHint === false && tensorHint) {
    throw new Error("LLaMA safetensors Q/K projection normalization mismatch: q_norm/k_norm tensors are present but the semantic hint is false");
  }
  return hasSemanticHint ? semanticHint : tensorHint;
}

function optionalSlidingWindowFromObject(source, names, label) {
  const value = firstConfigValueFromObject(source, names);
  if (value === undefined || disabledConfigFeatureValue(value)) return null;
  return positiveIntegerFromMetadataValue(value, `${label} sliding window`);
}

function normalizeLlamaSlidingWindowFromOptions(options, label) {
  const useSlidingWindow = firstBooleanFromObject(
    options,
    ["useSlidingWindow", "use_sliding_window"],
    label,
  );
  const slidingWindow = optionalSlidingWindowFromObject(
    options,
    ["slidingWindow", "sliding_window", "attentionSlidingWindow", "attention_sliding_window"],
    label,
  );
  if (useSlidingWindow === true && slidingWindow === null) {
    throw new Error(`${label} sliding window attention needs a sliding window`);
  }
  return useSlidingWindow === false ? null : slidingWindow;
}

function normalizeLlamaUseRopeLayerFromOptions(options, layer, label) {
  const directUseRope = firstBooleanFromObject(
    options,
    ["useRope", "use_rope"],
    label,
  );
  const useRopeLayers = optionalUseRopeLayerFlagsFromObject(
    options,
    ["useRopeLayers", "use_rope_layers", "noRopeLayers", "no_rope_layers"],
    layer + 1,
    `${label} no_rope_layers`,
  );
  const noRopeLayerInterval = optionalNonNegativeIntegerFromObject(
    options,
    ["noRopeLayerInterval", "no_rope_layer_interval"],
    `${label} no_rope_layer_interval`,
  );
  const intervalUseRopeLayers = noRopeLayerInterval === null
    ? null
    : useRopeLayerFlagsFromInterval(noRopeLayerInterval, layer + 1);
  const useRope = requireMatchingOptionalIntegers(`${label} use_rope`, [
    ["direct", directUseRope === null ? null : (directUseRope ? 1 : 0)],
    ["no_rope_layers", useRopeLayers === null ? null : useRopeLayers[layer]],
    ["no_rope_layer_interval", intervalUseRopeLayers === null ? null : intervalUseRopeLayers[layer]],
  ]);
  return useRope === null ? true : useRope !== 0;
}

function llamaAttentionStartForSeq(seq, slidingWindow) {
  if (!Number.isSafeInteger(seq) || seq <= 0) return 0;
  return Number.isSafeInteger(slidingWindow) && slidingWindow > 0 && seq > slidingWindow
    ? seq - slidingWindow
    : 0;
}

function validateLlamaSafetensorsSlidingWindow(options, metadata, config, contextLength) {
  const explicitUseSlidingWindow = firstBooleanFromObject(
    options,
    ["useSlidingWindow", "use_sliding_window"],
    "LLaMA safetensors option",
  );
  const metadataUseSlidingWindow = firstBooleanFromObject(
    metadata,
    ["use_sliding_window", "useSlidingWindow", ...llamaGgufMetadataNames("attention.use_sliding_window"), "zgml.use_sliding_window"],
    "LLaMA safetensors metadata",
  );
  const configUseSlidingWindow = firstBooleanFromObject(
    config,
    ["use_sliding_window", "useSlidingWindow"],
    "LLaMA safetensors config",
  );
  const useSlidingWindow = requireMatchingOptionalBooleans("LLaMA safetensors sliding window enabled", [
    ["option", explicitUseSlidingWindow],
    ["metadata", metadataUseSlidingWindow],
    ["config", configUseSlidingWindow],
  ]);

  const explicitSlidingWindow = optionalSlidingWindowFromObject(
    options,
    ["slidingWindow", "sliding_window", "attentionSlidingWindow", "attention_sliding_window"],
    "LLaMA safetensors option",
  );
  const metadataSlidingWindow = optionalSlidingWindowFromObject(
    metadata,
    ["sliding_window", "slidingWindow", "attention_sliding_window", "attentionSlidingWindow", ...llamaGgufMetadataNames("attention.sliding_window"), "zgml.sliding_window"],
    "LLaMA safetensors metadata",
  );
  const configSlidingWindow = optionalSlidingWindowFromObject(
    config,
    ["sliding_window", "slidingWindow"],
    "LLaMA safetensors config",
  );
  const slidingWindow = requireMatchingOptionalIntegers("LLaMA safetensors sliding window", [
    ["option", explicitSlidingWindow],
    ["metadata", metadataSlidingWindow],
    ["config", configSlidingWindow],
  ]);
  if (useSlidingWindow === true && slidingWindow === null) {
    throw new Error("unsupported LLaMA safetensors sliding window attention");
  }
  return useSlidingWindow === false ? null : slidingWindow;
}

function resolveLlamaSafetensorsUseRopeLayers(options, metadata, config, layerCount) {
  const noRopeLayerInterval = requireMatchingOptionalIntegers("LLaMA safetensors no_rope_layer_interval", [
    [
      "option",
      optionalNonNegativeIntegerFromObject(
        options,
        ["noRopeLayerInterval", "no_rope_layer_interval"],
        "LLaMA safetensors option no_rope_layer_interval",
      ),
    ],
    [
      "metadata",
      optionalNonNegativeIntegerFromObject(
        metadata,
        [
          "no_rope_layer_interval",
          "noRopeLayerInterval",
          ...llamaGgufMetadataNames("rope.no_rope_layer_interval"),
          "zgml.no_rope_layer_interval",
        ],
        "LLaMA safetensors metadata no_rope_layer_interval",
      ),
    ],
    [
      "config",
      optionalNonNegativeIntegerFromObject(
        config,
        ["no_rope_layer_interval", "noRopeLayerInterval"],
        "LLaMA safetensors config no_rope_layer_interval",
      ),
    ],
  ]);
  const intervalUseRopeLayers = noRopeLayerInterval === null
    ? null
    : useRopeLayerFlagsFromInterval(noRopeLayerInterval, layerCount);
  return requireMatchingOptionalIntegerArrays("LLaMA safetensors no_rope_layers", [
    [
      "option",
      optionalUseRopeLayerFlagsFromObject(
        options,
        ["noRopeLayers", "no_rope_layers", "useRopeLayers", "use_rope_layers"],
        layerCount,
        "LLaMA safetensors option no_rope_layers",
      ),
    ],
    [
      "metadata",
      optionalUseRopeLayerFlagsFromObject(
        metadata,
        [
          "no_rope_layers",
          "noRopeLayers",
          "use_rope_layers",
          "useRopeLayers",
          ...llamaGgufMetadataNames("rope.no_rope_layers"),
          "zgml.no_rope_layers",
        ],
        layerCount,
        "LLaMA safetensors metadata no_rope_layers",
      ),
    ],
    [
      "config",
      optionalUseRopeLayerFlagsFromObject(
        config,
        ["no_rope_layers", "noRopeLayers", "use_rope_layers", "useRopeLayers"],
        layerCount,
        "LLaMA safetensors config no_rope_layers",
      ),
    ],
    ["no_rope_layer_interval", intervalUseRopeLayers],
  ]) ?? Array(layerCount).fill(1);
}

function resolveLlamaSafetensorsBiasHint(options, metadata, config, kind) {
  const optionNames = kind === "attention"
    ? ["attentionBias", "attention_bias", "qkvBias", "qkv_bias"]
    : ["mlpBias", "mlp_bias", "ffnBias", "ffn_bias"];
  const metadataNames = kind === "attention"
    ? ["attention_bias", "attentionBias", "qkv_bias", "qkvBias", ...llamaGgufMetadataNames("attention.bias"), "zgml.attention_bias"]
    : ["mlp_bias", "mlpBias", "ffn_bias", "ffnBias", ...llamaGgufMetadataNames("feed_forward.bias"), "zgml.mlp_bias"];
  const configNames = kind === "attention"
    ? ["attention_bias", "attentionBias", "qkv_bias", "qkvBias"]
    : ["mlp_bias", "mlpBias", "ffn_bias", "ffnBias"];
  return requireMatchingOptionalBooleans(`LLaMA safetensors ${kind} bias`, [
    ["option", firstBooleanFromObject(options, optionNames, "LLaMA safetensors option")],
    ["metadata", firstBooleanFromObject(metadata, metadataNames, "LLaMA safetensors metadata")],
    ["config", firstBooleanFromObject(config, configNames, "LLaMA safetensors config")],
  ]);
}

function llamaLayerBiasTensorNames(layer, kind) {
  if (kind === "attention") {
    return [
      `model.layers.${layer}.self_attn.q_proj.bias`,
      `model.layers.${layer}.self_attn.k_proj.bias`,
      `model.layers.${layer}.self_attn.v_proj.bias`,
      `model.layers.${layer}.self_attn.o_proj.bias`,
    ];
  }
  return [
    `model.layers.${layer}.mlp.gate_proj.bias`,
    `model.layers.${layer}.mlp.up_proj.bias`,
    `model.layers.${layer}.mlp.down_proj.bias`,
  ];
}

function validateOptionalSafetensorsVector(spec, expectedLength, label) {
  if (spec === null) return false;
  const length = safetensorsVectorLength(spec, label);
  if (length !== expectedLength) {
    throw new Error(`${label} length mismatch: ${length} != ${expectedLength}`);
  }
  return true;
}

function validateLlamaSafetensorsBiasTensorGroup(specs, layer, kind, expectedLengths, biasHint) {
  const names = llamaLayerBiasTensorNames(layer, kind);
  let present = 0;
  for (let index = 0; index < names.length; index += 1) {
    const spec = findSafetensorsTensorSpec(specs, names[index]);
    if (validateOptionalSafetensorsVector(spec, expectedLengths[index], `LLaMA safetensors layer ${layer} ${kind} bias tensor ${names[index]}`)) {
      present += 1;
    }
  }
  if (biasHint === false && present > 0) {
    throw new Error(`LLaMA safetensors ${kind} bias tensor conflicts with disabled bias`);
  }
  if (biasHint === true && present !== names.length) {
    throw new Error(`LLaMA safetensors ${kind} bias is enabled but tensor set is incomplete`);
  }
  if (present > 0 && present !== names.length) {
    throw new Error(`LLaMA safetensors ${kind} bias tensor set is incomplete`);
  }
  return present === names.length;
}

function normalizeOptionalLayerIndexList(value, label) {
  if (value === undefined || value === null) return null;
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array`);
  }
  const seen = new Set();
  const layers = value.map((entry, index) => {
    if (!Number.isSafeInteger(entry) || entry < 0) {
      throw new Error(`${label}[${index}] must be a non-negative integer`);
    }
    if (seen.has(entry)) {
      throw new Error(`${label} contains duplicate layer ${entry}`);
    }
    seen.add(entry);
    return entry;
  });
  return Object.freeze(layers);
}

function requireMatchingOptionalIntegers(label, values) {
  let expected = null;
  for (const [source, value] of values) {
    if (value === null) continue;
    if (expected === null) {
      expected = { source, value };
      continue;
    }
    if (expected.value !== value) {
      throw new Error(`${label} mismatch: ${expected.source}=${expected.value} != ${source}=${value}`);
    }
  }
  return expected?.value ?? null;
}

function requireMatchingOptionalNumbers(label, values, tolerance = 1e-6) {
  let expected = null;
  for (const [source, value] of values) {
    if (value === null) continue;
    if (expected === null) {
      expected = { source, value };
      continue;
    }
    if (Math.abs(expected.value - value) > tolerance) {
      throw new Error(`${label} mismatch: ${expected.source}=${expected.value} != ${source}=${value}`);
    }
  }
  return expected?.value ?? null;
}

function requireMatchingOptionalBooleans(label, values) {
  let expected = null;
  for (const [source, value] of values) {
    if (value === null) continue;
    if (expected === null) {
      expected = { source, value };
      continue;
    }
    if (expected.value !== value) {
      throw new Error(`${label} mismatch: ${expected.source}=${expected.value} != ${source}=${value}`);
    }
  }
  return expected?.value ?? null;
}

function requireMatchingOptionalStrings(label, values) {
  let expected = null;
  for (const [source, value] of values) {
    if (value === null) continue;
    if (expected === null) {
      expected = { source, value };
      continue;
    }
    if (expected.value !== value) {
      throw new Error(`${label} mismatch: ${expected.source}=${expected.value} != ${source}=${value}`);
    }
  }
  return expected?.value ?? null;
}

function requireMatchingOptionalStringArrays(label, values) {
  let expected = null;
  for (const [source, value] of values) {
    if (value === null) continue;
    if (expected === null) {
      expected = { source, value };
      continue;
    }
    if (expected.value.length !== value.length || expected.value.some((entry, index) => entry !== value[index])) {
      throw new Error(`${label} mismatch: ${expected.source}=${expected.value.join(",")} != ${source}=${value.join(",")}`);
    }
  }
  return expected?.value ?? null;
}

function requireMatchingOptionalIntegerArrays(label, values) {
  let expected = null;
  for (const [source, value] of values) {
    if (value === null) continue;
    if (expected === null) {
      expected = { source, value };
      continue;
    }
    if (expected.value.length !== value.length || expected.value.some((entry, index) => entry !== value[index])) {
      throw new Error(`${label} mismatch: ${expected.source}=${expected.value.join(",")} != ${source}=${value.join(",")}`);
    }
  }
  return expected?.value ?? null;
}

function resolveLlamaSafetensorsAttentionHeadSize(options, metadata, config, shape) {
  const explicitHeadSize = firstPositiveIntegerFromObject(
    options,
    ["attentionHeadSize", "headSize", "attention_head_size", "head_size"],
    "LLaMA safetensors option",
  );
  const metadataHeadSize = firstPositiveIntegerFromObject(
    metadata,
    [
      "attention_head_size",
      "head_size",
      ...llamaGgufMetadataNames("attention.head_size", "attention.head_size_kv"),
      "zgml.attention_head_size",
    ],
    "LLaMA safetensors metadata",
  );
  const configHeadSize = firstPositiveIntegerFromObject(
    config,
    ["attention_head_size", "head_size", "head_dim", "llama.attention.head_size", "llama.attention.head_size_kv"],
    "LLaMA safetensors config",
  );
  const sourceHeadSize = requireMatchingOptionalIntegers("LLaMA safetensors attention head size", [
    ["option", explicitHeadSize],
    ["metadata", metadataHeadSize],
    ["config", configHeadSize],
  ]);

  const explicitKeyLength = firstPositiveIntegerFromObject(
    options,
    ["attentionKeyLength", "attention_key_length", "keyLength", "key_length"],
    "LLaMA safetensors option",
  );
  const metadataKeyLength = firstPositiveIntegerFromObject(
    metadata,
    ["attention_key_length", "key_length", ...llamaGgufMetadataNames("attention.key_length"), "zgml.attention_key_length"],
    "LLaMA safetensors metadata",
  );
  const configKeyLength = firstPositiveIntegerFromObject(
    config,
    ["attention_key_length", "attentionKeyLength", "key_length", "keyLength"],
    "LLaMA safetensors config",
  );
  const sourceKeyLength = requireMatchingOptionalIntegers("LLaMA safetensors attention key length", [
    ["option", explicitKeyLength],
    ["metadata", metadataKeyLength],
    ["config", configKeyLength],
  ]);

  const explicitValueLength = firstPositiveIntegerFromObject(
    options,
    ["attentionValueLength", "attention_value_length", "valueLength", "value_length"],
    "LLaMA safetensors option",
  );
  const metadataValueLength = firstPositiveIntegerFromObject(
    metadata,
    ["attention_value_length", "value_length", ...llamaGgufMetadataNames("attention.value_length"), "zgml.attention_value_length"],
    "LLaMA safetensors metadata",
  );
  const configValueLength = firstPositiveIntegerFromObject(
    config,
    ["attention_value_length", "attentionValueLength", "value_length", "valueLength"],
    "LLaMA safetensors config",
  );
  const sourceValueLength = requireMatchingOptionalIntegers("LLaMA safetensors attention value length", [
    ["option", explicitValueLength],
    ["metadata", metadataValueLength],
    ["config", configValueLength],
  ]);
  const sourceKeyValueLength = requireMatchingOptionalIntegers("LLaMA safetensors attention key/value length", [
    ["key", sourceKeyLength],
    ["value", sourceValueLength],
  ]);

  const explicitQueryHeads = firstPositiveIntegerFromObject(
    options,
    ["numAttentionHeads", "attentionHeads", "nHeads", "n_heads", "num_attention_heads"],
    "LLaMA safetensors option",
  );
  const metadataQueryHeads = firstPositiveIntegerFromObject(
    metadata,
    [
      "num_attention_heads",
      "n_heads",
      ...llamaGgufMetadataNames("attention.head_count"),
      "zgml.num_attention_heads",
    ],
    "LLaMA safetensors metadata",
  );
  const configQueryHeads = firstPositiveIntegerFromObject(
    config,
    ["num_attention_heads", "attention_heads", "n_heads", "llama.attention.head_count"],
    "LLaMA safetensors config",
  );
  const sourceQueryHeads = requireMatchingOptionalIntegers("LLaMA safetensors attention head-count", [
    ["option", explicitQueryHeads],
    ["metadata", metadataQueryHeads],
    ["config", configQueryHeads],
  ]);

  const explicitKvHeads = firstPositiveIntegerFromObject(
    options,
    ["numKeyValueHeads", "keyValueHeads", "nKvHeads", "n_kv_heads", "num_key_value_heads"],
    "LLaMA safetensors option",
  );
  const metadataKvHeads = firstPositiveIntegerFromObject(
    metadata,
    [
      "num_key_value_heads",
      "n_kv_heads",
      ...llamaGgufMetadataNames("attention.head_count_kv"),
      "zgml.num_key_value_heads",
    ],
    "LLaMA safetensors metadata",
  );
  const configKvHeads = firstPositiveIntegerFromObject(
    config,
    ["num_key_value_heads", "key_value_heads", "n_kv_heads", "llama.attention.head_count_kv"],
    "LLaMA safetensors config",
  );
  const sourceKvHeads = requireMatchingOptionalIntegers("LLaMA safetensors K/V head-count", [
    ["option", explicitKvHeads],
    ["metadata", metadataKvHeads],
    ["config", configKvHeads],
  ]);

  const queryHeads = sourceQueryHeads;
  const kvHeads = sourceKvHeads;
  let headSize = sourceHeadSize;
  if (headSize === null && queryHeads !== null) {
    if (shape.qRows % queryHeads !== 0) {
      throw new Error(`LLaMA safetensors Q rows do not divide attention head count: ${shape.qRows} / ${queryHeads}`);
    }
    headSize = shape.qRows / queryHeads;
  }
  if (headSize === null && kvHeads !== null) {
    if (shape.kRows % kvHeads !== 0) {
      throw new Error(`LLaMA safetensors K rows do not divide K/V head count: ${shape.kRows} / ${kvHeads}`);
    }
    headSize = shape.kRows / kvHeads;
  }
  if (headSize === null && sourceKeyValueLength !== null) {
    headSize = sourceKeyValueLength;
  }
  if (headSize === null && shape.kRows === shape.hiddenSize && shape.qRows === shape.hiddenSize) {
    headSize = shape.hiddenSize;
  }
  if (headSize === null) {
    throw new Error("LLaMA safetensors grouped attention needs attentionHeadSize or attention head-count metadata");
  }
  if (headSize % 2 !== 0) {
    throw new Error(`LLaMA safetensors attention head size must be even for RoPE: ${headSize}`);
  }
  if (shape.qRows % headSize !== 0 || shape.kRows % headSize !== 0 || shape.vRows % headSize !== 0) {
    throw new Error(`LLaMA safetensors attention rows must be multiples of head size: q=${shape.qRows} k=${shape.kRows} v=${shape.vRows} head=${headSize}`);
  }
  const queryHeadCount = shape.qRows / headSize;
  const kvHeadCount = shape.kRows / headSize;
  if (shape.vRows / headSize !== kvHeadCount) {
    throw new Error(`LLaMA safetensors K/V head count mismatch: ${kvHeadCount} != ${shape.vRows / headSize}`);
  }
  if (queryHeads !== null && queryHeads !== queryHeadCount) {
    throw new Error(`LLaMA safetensors attention head-count metadata mismatch: ${queryHeads} != ${queryHeadCount}`);
  }
  if (kvHeads !== null && kvHeads !== kvHeadCount) {
    throw new Error(`LLaMA safetensors K/V head-count metadata mismatch: ${kvHeads} != ${kvHeadCount}`);
  }
  if (sourceKeyValueLength !== null && sourceKeyValueLength !== headSize) {
    throw new Error(`LLaMA safetensors attention key/value length mismatch: ${sourceKeyValueLength} != ${headSize}`);
  }
  if (queryHeadCount % kvHeadCount !== 0) {
    throw new Error(`LLaMA safetensors query/KV heads must group evenly: ${queryHeadCount} / ${kvHeadCount}`);
  }
  return {
    attentionHeadSize: headSize,
    kvHeadCount,
    queryHeadCount,
    queryHeadsPerKvHead: queryHeadCount / kvHeadCount,
  };
}

function resolveLlamaSafetensorsRopeBase(options, metadata, config) {
  const explicitRopeBase = firstPositiveFiniteNumberFromObject(
    options,
    ["ropeBase", "ropeTheta", "rope_base", "rope_theta", "rotaryBase", "rotaryTheta"],
    "LLaMA safetensors option",
  );
  const metadataRopeBase = firstPositiveFiniteNumberFromObject(
    metadata,
    ["rope_base", "rope_theta", ...llamaGgufMetadataNames("rope.freq_base", "rope.theta"), "zgml.rope_base"],
    "LLaMA safetensors metadata",
  );
  const configRopeBase = firstPositiveFiniteNumberFromObject(
    config,
    ["rope_theta", "ropeTheta", "rope_base", "ropeBase", "rotary_emb_base", "rotaryEmbBase"],
    "LLaMA safetensors config",
  );
  return requireMatchingOptionalNumbers("LLaMA safetensors RoPE base", [
    ["option", explicitRopeBase],
    ["metadata", metadataRopeBase],
    ["config", configRopeBase],
  ]) ?? 10000;
}

function resolveLlamaSafetensorsRopeDimensionCount(options, metadata, config) {
  const explicitRopeDimensionCount = firstPositiveIntegerFromObject(
    options,
    ["ropeDimensionCount", "rope_dimension_count", "rotaryDimensionCount", "rotary_dimension_count", "rotaryDim", "rotary_dim"],
    "LLaMA safetensors option",
  );
  const metadataRopeDimensionCount = firstPositiveIntegerFromObject(
    metadata,
    ["rope_dimension_count", "rotary_dimension_count", "rotary_dim", ...llamaGgufMetadataNames("rope.dimension_count"), "zgml.rope_dimension_count"],
    "LLaMA safetensors metadata",
  );
  const configRopeDimensionCount = firstPositiveIntegerFromObject(
    config,
    ["rope_dimension_count", "ropeDimensionCount", "rotary_dimension_count", "rotaryDimensionCount", "rotary_dim", "rotaryDim"],
    "LLaMA safetensors config",
  );
  return requireMatchingOptionalIntegers("LLaMA safetensors RoPE dimension count", [
    ["option", explicitRopeDimensionCount],
    ["metadata", metadataRopeDimensionCount],
    ["config", configRopeDimensionCount],
  ]);
}

function validateLlamaSafetensorsRopeDimensionCount(options, metadata, config, attentionHeadSize) {
  const ropeDimensionCount = resolveLlamaSafetensorsRopeDimensionCount(options, metadata, config);
  if (ropeDimensionCount !== null && ropeDimensionCount !== attentionHeadSize) {
    throw new Error(`unsupported LLaMA safetensors partial rotary dimension count: ${ropeDimensionCount} != attention head size ${attentionHeadSize}`);
  }
}

function resolveLlamaSafetensorsRopeScale(options, metadata, config) {
  const explicitRopeScale = firstPositiveFiniteNumberFromObject(
    options,
    ["ropeScale", "rope_scale", "ropeScalingFactor", "rope_scaling_factor", "linearRopeScale", "linear_rope_scale"],
    "LLaMA safetensors option",
  );
  const metadataRopeScale = firstPositiveFiniteNumberFromObject(
    metadata,
    ["rope_scale", "rope_scaling_factor", ...llamaGgufMetadataNames("rope.scale"), "zgml.rope_scale"],
    "LLaMA safetensors metadata",
  );
  const ropeScaling = firstConfigValueFromObject(config, ["rope_scaling", "ropeScaling"]);
  const configRopeScale =
    ropeScaling === undefined || disabledConfigFeatureValue(ropeScaling)
      ? null
      : ropeScaleFromRopeScalingConfigValue(ropeScaling);
  return requireMatchingOptionalNumbers("LLaMA safetensors RoPE scale", [
    ["option", explicitRopeScale],
    ["metadata", metadataRopeScale],
    ["config", configRopeScale],
  ]) ?? 1;
}

function resolveLlamaSafetensorsRopeScaling(options, metadata, config) {
  const ropeScaling = firstConfigValueFromObject(config, ["rope_scaling", "ropeScaling"]);
  const hasConfigRopeScaling = ropeScaling !== undefined;
  const ropeScalingObject = typeof ropeScaling === "object" && ropeScaling !== null && !Array.isArray(ropeScaling)
    ? ropeScaling
    : null;
  const parsedConfig = ropeScaling === undefined
    ? {
        highFreqFactor: 1,
        kind: "none",
        lowFreqFactor: 1,
        originalContextLength: 1,
        scale: 1,
      }
    : parseRopeScalingConfigValue(ropeScaling);
  if (parsedConfig === false) {
    throw new Error("unsupported LLaMA safetensors config rope scaling");
  }
  const ropeScale = resolveLlamaSafetensorsRopeScale(options, metadata, config);
  const explicitKind = firstStringFromObject(
    options,
    ["ropeKind", "ropeType", "rope_kind", "rope_type", "ropeScalingKind", "rope_scaling_kind"],
    "LLaMA safetensors option",
  );
  const metadataKind = firstStringFromObject(
    metadata,
    ["rope_kind", "rope_type", "rope_scaling_kind", "rope_scaling_type", ...llamaGgufMetadataNames("rope.scaling_type"), "zgml.rope_kind"],
    "LLaMA safetensors metadata",
  );
  let ropeKind = requireMatchingOptionalStrings("LLaMA safetensors RoPE kind", [
    ["option", explicitKind === null ? null : normalizeLlamaRopeKind(explicitKind)],
    ["metadata", metadataKind === null ? null : normalizeLlamaRopeKind(metadataKind)],
    ["config", hasConfigRopeScaling ? parsedConfig.kind : null],
  ]) ?? parsedConfig.kind;
  if (ropeKind === "none" && Math.abs(ropeScale - 1) > 1e-6) ropeKind = "linear";
  const explicitLowFreqFactor = firstPositiveFiniteNumberFromObject(
    options,
    ["ropeLowFreqFactor", "rope_low_freq_factor"],
    "LLaMA safetensors option",
  );
  const metadataLowFreqFactor = firstPositiveFiniteNumberFromObject(
    metadata,
    ["rope_low_freq_factor", "ropeLowFreqFactor", ...llamaGgufMetadataNames("rope.low_freq_factor"), "zgml.rope_low_freq_factor"],
    "LLaMA safetensors metadata",
  );
  const explicitHighFreqFactor = firstPositiveFiniteNumberFromObject(
    options,
    ["ropeHighFreqFactor", "rope_high_freq_factor"],
    "LLaMA safetensors option",
  );
  const metadataHighFreqFactor = firstPositiveFiniteNumberFromObject(
    metadata,
    ["rope_high_freq_factor", "ropeHighFreqFactor", ...llamaGgufMetadataNames("rope.high_freq_factor"), "zgml.rope_high_freq_factor"],
    "LLaMA safetensors metadata",
  );
  const explicitOriginalContextLength = firstPositiveIntegerFromObject(
    options,
    ["ropeOriginalContextLength", "rope_original_context_length", "originalMaxPositionEmbeddings", "original_max_position_embeddings"],
    "LLaMA safetensors option",
  );
  const metadataOriginalContextLength = firstPositiveIntegerFromObject(
    metadata,
    ["rope_original_context_length", "ropeOriginalContextLength", "original_max_position_embeddings", ...llamaGgufMetadataNames("rope.original_context_length"), "zgml.rope_original_context_length"],
    "LLaMA safetensors metadata",
  );
  const configLowFreqFactor = ropeScalingObject === null
    ? null
    : optionalPositiveFiniteNumberFromObject(
        ropeScalingObject,
        ["low_freq_factor", "lowFreqFactor"],
        "LLaMA safetensors config rope scaling low frequency factor",
      );
  const configHighFreqFactor = ropeScalingObject === null
    ? null
    : optionalPositiveFiniteNumberFromObject(
        ropeScalingObject,
        ["high_freq_factor", "highFreqFactor"],
        "LLaMA safetensors config rope scaling high frequency factor",
      );
  const configOriginalContextLength = ropeScalingObject === null
    ? null
    : optionalPositiveIntegerFromObject(
        ropeScalingObject,
        ["original_max_position_embeddings", "originalMaxPositionEmbeddings"],
        "LLaMA safetensors config rope scaling original max position embeddings",
      );
  const lowFreqFactor = requireMatchingOptionalNumbers("LLaMA safetensors RoPE low frequency factor", [
    ["option", explicitLowFreqFactor],
    ["metadata", metadataLowFreqFactor],
    ["config", configLowFreqFactor],
  ]) ?? parsedConfig.lowFreqFactor ?? 1;
  const highFreqFactor = requireMatchingOptionalNumbers("LLaMA safetensors RoPE high frequency factor", [
    ["option", explicitHighFreqFactor],
    ["metadata", metadataHighFreqFactor],
    ["config", configHighFreqFactor],
  ]) ?? parsedConfig.highFreqFactor ?? 1;
  const originalContextLength = requireMatchingOptionalIntegers("LLaMA safetensors RoPE original context length", [
    ["option", explicitOriginalContextLength],
    ["metadata", metadataOriginalContextLength],
    ["config", configOriginalContextLength],
  ]) ?? parsedConfig.originalContextLength ?? 1;
  if (ropeKind === "llama3" && highFreqFactor <= lowFreqFactor) {
    throw new Error(`invalid LLaMA 3 RoPE frequency factors: low=${lowFreqFactor} high=${highFreqFactor}`);
  }
  return {
    ropeHighFreqFactor: highFreqFactor,
    ropeKind,
    ropeKindId: llamaRopeKindId(ropeKind),
    ropeLowFreqFactor: lowFreqFactor,
    ropeOriginalContextLength: originalContextLength,
    ropeScale,
  };
}

function resolveLlamaSafetensorsRmsNormEpsilon(options, metadata, config) {
  const explicitEpsilon = firstPositiveFiniteNumberFromObject(
    options,
    ["epsilon", "rmsNormEpsilon", "rmsNormEps", "rms_norm_epsilon", "rms_norm_eps", "normEpsilon", "norm_eps"],
    "LLaMA safetensors option",
  );
  const metadataEpsilon = firstPositiveFiniteNumberFromObject(
    metadata,
    ["rms_norm_eps", "rms_norm_epsilon", "norm_eps", ...llamaGgufMetadataNames("attention.layer_norm_rms_epsilon"), "zgml.rms_norm_eps"],
    "LLaMA safetensors metadata",
  );
  const configEpsilon = firstPositiveFiniteNumberFromObject(
    config,
    ["rms_norm_eps", "rmsNormEps", "rms_norm_epsilon", "rmsNormEpsilon", "norm_eps", "layer_norm_epsilon"],
    "LLaMA safetensors config",
  );
  return requireMatchingOptionalNumbers("LLaMA safetensors RMSNorm epsilon", [
    ["option", explicitEpsilon],
    ["metadata", metadataEpsilon],
    ["config", configEpsilon],
  ]) ?? 1e-5;
}

function resolveLlamaSafetensorsTiedLmHead(options, metadata, config) {
  const explicitTiedLmHead = firstBooleanFromObject(
    options,
    ["allowTiedLmHead", "tiedLmHead", "tieLmHead", "tie_word_embeddings", "tieWordEmbeddings"],
    "LLaMA safetensors option",
  );
  const metadataTiedLmHead = firstBooleanFromObject(
    metadata,
    ["tie_word_embeddings", "tied_lm_head", ...llamaGgufMetadataNames("tie_word_embeddings"), "zgml.tied_lm_head"],
    "LLaMA safetensors metadata",
  );
  const configTiedLmHead = firstBooleanFromObject(
    config,
    ["tie_word_embeddings", "tieWordEmbeddings", "tied_lm_head", "tiedLmHead"],
    "LLaMA safetensors config",
  );
  return requireMatchingOptionalBooleans("LLaMA safetensors tied LM head", [
    ["option", explicitTiedLmHead],
    ["metadata", metadataTiedLmHead],
    ["config", configTiedLmHead],
  ]) ?? true;
}

function resolveLlamaSafetensorsIntermediateSize(options, metadata, config, tensorIntermediateSize) {
  const explicitIntermediateSize = firstPositiveIntegerFromObject(
    options,
    ["intermediateSize", "intermediate_size", "ffnSize", "ffn_size"],
    "LLaMA safetensors option",
  );
  const metadataIntermediateSize = firstPositiveIntegerFromObject(
    metadata,
    ["intermediate_size", "ffn_size", ...llamaGgufMetadataNames("feed_forward_length"), "zgml.intermediate_size"],
    "LLaMA safetensors metadata",
  );
  const configIntermediateSize = firstPositiveIntegerFromObject(
    config,
    ["intermediate_size", "intermediateSize", "ffn_dim", "ffnSize", "ffn_size"],
    "LLaMA safetensors config",
  );
  return requireMatchingOptionalIntegers("LLaMA safetensors FFN intermediate size", [
    ["tensor", tensorIntermediateSize],
    ["option", explicitIntermediateSize],
    ["metadata", metadataIntermediateSize],
    ["config", configIntermediateSize],
  ]);
}

function resolveLlamaSafetensorsHiddenSize(options, metadata, config, tensorHiddenSize) {
  const explicitHiddenSize = firstPositiveIntegerFromObject(
    options,
    ["hiddenSize", "hidden", "hidden_size", "dModel", "d_model", "nEmbd", "n_embd"],
    "LLaMA safetensors option",
  );
  const metadataHiddenSize = firstPositiveIntegerFromObject(
    metadata,
    ["hidden_size", "hidden", "d_model", "n_embd", ...llamaGgufMetadataNames("embedding_length"), "zgml.hidden_size"],
    "LLaMA safetensors metadata",
  );
  const configHiddenSize = firstPositiveIntegerFromObject(
    config,
    ["hidden_size", "hiddenSize", "d_model", "n_embd"],
    "LLaMA safetensors config",
  );
  return requireMatchingOptionalIntegers("LLaMA safetensors hidden size", [
    ["tensor", tensorHiddenSize],
    ["option", explicitHiddenSize],
    ["metadata", metadataHiddenSize],
    ["config", configHiddenSize],
  ]);
}

function resolveLlamaSafetensorsVocabSize(options, metadata, config, tensorVocabSize) {
  const explicitVocabSize = firstPositiveIntegerFromObject(
    options,
    ["vocabSize", "vocab_size", "vocabularySize", "vocabulary_size"],
    "LLaMA safetensors option",
  );
  const metadataVocabSize = firstPositiveIntegerFromObject(
    metadata,
    ["vocab_size", "vocabulary_size", ...llamaGgufMetadataNames("vocab_size"), "tokenizer.ggml.vocab_size", "zgml.vocab_size"],
    "LLaMA safetensors metadata",
  );
  const configVocabSize = firstPositiveIntegerFromObject(
    config,
    ["vocab_size", "vocabSize", "vocabulary_size"],
    "LLaMA safetensors config",
  );
  const metadataTokenizerTokenCount = firstTokenizerTokenArrayLengthFromObject(
    metadata,
    ["tokenizer.ggml.tokens", "tokenizer.tokens"],
    "LLaMA safetensors metadata",
  );
  const metadataTokenizerScoreCount = firstTokenizerFiniteNumberArrayLengthFromObject(
    metadata,
    ["tokenizer.ggml.scores", "tokenizer.scores"],
    "LLaMA safetensors metadata",
  );
  const metadataTokenizerTokenTypeCount = firstTokenizerNonNegativeIntegerArrayLengthFromObject(
    metadata,
    ["tokenizer.ggml.token_type", "tokenizer.ggml.token_types", "tokenizer.token_type", "tokenizer.token_types"],
    "LLaMA safetensors metadata",
  );
  return requireMatchingOptionalIntegers("LLaMA safetensors vocab size", [
    ["tensor", tensorVocabSize],
    ["option", explicitVocabSize],
    ["metadata", metadataVocabSize],
    ["tokenizer tokens", metadataTokenizerTokenCount],
    ["tokenizer scores", metadataTokenizerScoreCount],
    ["tokenizer token types", metadataTokenizerTokenTypeCount],
    ["config", configVocabSize],
  ]);
}

function validateLlamaSafetensorsTokenIdsWithinVocab(source, names, label, vocabSize) {
  const tokenIds = firstTokenIdsFromObject(source, names, label);
  if (tokenIds === null) return;
  for (let index = 0; index < tokenIds.length; index += 1) {
    if (tokenIds[index] >= vocabSize) {
      const suffix = tokenIds.length > 1 ? `[${index}]` : "";
      throw new Error(`${label}${suffix} token id exceeds vocab size: ${tokenIds[index]} >= ${vocabSize}`);
    }
  }
}

function validateLlamaSafetensorsSpecialTokenIdsWithinVocab(metadata, config, vocabSize) {
  for (const tokenIdProof of [
    {
      label: "bos_token_id",
      metadataNames: ["bos_token_id", "bosTokenId", "tokenizer.bos_token_id", "tokenizer.ggml.bos_token_id"],
      configNames: ["bos_token_id", "bosTokenId"],
    },
    {
      label: "eos_token_id",
      metadataNames: ["eos_token_id", "eosTokenId", "tokenizer.eos_token_id", "tokenizer.ggml.eos_token_id"],
      configNames: ["eos_token_id", "eosTokenId"],
    },
    {
      label: "pad_token_id",
      metadataNames: ["pad_token_id", "padTokenId", "tokenizer.pad_token_id", "tokenizer.ggml.pad_token_id"],
      configNames: ["pad_token_id", "padTokenId"],
    },
  ]) {
    validateLlamaSafetensorsTokenIdsWithinVocab(
      metadata,
      tokenIdProof.metadataNames,
      `LLaMA safetensors metadata ${tokenIdProof.label}`,
      vocabSize,
    );
    validateLlamaSafetensorsTokenIdsWithinVocab(
      config,
      tokenIdProof.configNames,
      `LLaMA safetensors config ${tokenIdProof.label}`,
      vocabSize,
    );
  }
}

function resolveLlamaSafetensorsLayerCount(options, metadata, config, tensorLayerCount) {
  const explicitLayerCount = firstPositiveIntegerFromObject(
    options,
    ["numHiddenLayers", "num_hidden_layers", "nLayers", "n_layers", "nLayer", "n_layer"],
    "LLaMA safetensors option",
  );
  const metadataLayerCount = firstPositiveIntegerFromObject(
    metadata,
    ["num_hidden_layers", "n_layers", "n_layer", ...llamaGgufMetadataNames("block_count"), "zgml.num_hidden_layers"],
    "LLaMA safetensors metadata",
  );
  const configLayerCount = firstPositiveIntegerFromObject(
    config,
    ["num_hidden_layers", "numHiddenLayers", "n_layers", "n_layer"],
    "LLaMA safetensors config",
  );
  return requireMatchingOptionalIntegers("LLaMA safetensors layer count", [
    ["tensor", tensorLayerCount],
    ["option", explicitLayerCount],
    ["metadata", metadataLayerCount],
    ["config", configLayerCount],
  ]);
}

function inferLlamaResourceProgramShapeFromSafetensors(source, options = {}) {
  const modelResourceOptions = {
    materializeFloatDtypes: true,
    ...(options.modelResourceOptions ?? {}),
  };
  const header = parseSafetensorsHeaderSource(source, modelResourceOptions);
  const metadata = safetensorsMetadataObject(header);
  const config = llamaConfigObjectFromOptions(options, metadata);
  const contextLength = resolveLlamaSafetensorsContextLength(options, metadata, config);
  rejectUnsupportedLlamaSafetensorsConfig(config, contextLength);
  rejectUnsupportedLlamaSafetensorsMetadataSemantics(metadata);
  validateLlamaSafetensorsMetadataConfigSemantics(metadata, config);
  validateLlamaSafetensorsContextEnvelope(options, metadata, config, contextLength);
  const qkProjectionNormHint = resolveLlamaSafetensorsQkProjectionNorm(options, metadata, config);
  const slidingWindow = validateLlamaSafetensorsSlidingWindow(options, metadata, config, contextLength);
  const attentionBiasHint = resolveLlamaSafetensorsBiasHint(options, metadata, config, "attention");
  const mlpBiasHint = resolveLlamaSafetensorsBiasHint(options, metadata, config, "mlp");
  const tiedLmHead = resolveLlamaSafetensorsTiedLmHead(
    {
      ...(modelResourceOptions ?? {}),
      ...options,
    },
    metadata,
    config,
  );
  const resolvedModelResourceOptions = {
    ...modelResourceOptions,
    allowTiedLmHead: tiedLmHead,
    embeddingTensor: options.embeddingTensor,
    lmHeadBiasTensor: options.lmHeadBiasTensor,
    lmHeadTensor: options.lmHeadTensor,
    normTensor: options.normTensor,
    qkProjectionNorm: qkProjectionNormHint,
  };
  const specs = safetensorsModelResourceSpecs(source, resolvedModelResourceOptions);
  const qkProjectionNorm = resolveLlamaSafetensorsQkProjectionNormWithTensors(options, metadata, config, specs);
  resolvedModelResourceOptions.qkProjectionNorm = qkProjectionNorm;
  rejectUnsupportedLlamaSafetensorsTensorNames(specs, resolvedModelResourceOptions);
  rejectUnsupportedLlamaSafetensorsExecutorDtypes(specs);
  const modelResourceSpecs = executableLlamaSafetensorsModelResourceSpecs(specs, resolvedModelResourceOptions);
  const embedding = safetensorsMatrixShape(
    findSafetensorsTensorSpec(modelResourceSpecs, options.embeddingTensor ?? "model.embed_tokens.weight"),
    "LLaMA safetensors embedding tensor",
  );
  const lmHead = safetensorsMatrixShape(
    findSafetensorsTensorSpec(modelResourceSpecs, options.lmHeadTensor ?? "lm_head.weight"),
    "LLaMA safetensors lm_head tensor",
  );
  if (embedding.cols !== lmHead.cols) {
    throw new Error(`LLaMA safetensors hidden size mismatch: ${embedding.cols} != ${lmHead.cols}`);
  }
  const hiddenSize = resolveLlamaSafetensorsHiddenSize(options, metadata, config, embedding.cols);
  const vocabSize = resolveLlamaSafetensorsVocabSize(options, metadata, config, lmHead.rows);
  validateLlamaSafetensorsSpecialTokenIdsWithinVocab(metadata, config, vocabSize);
  validateOptionalSafetensorsVector(
    findSafetensorsTensorSpec(modelResourceSpecs, options.lmHeadBiasTensor ?? "lm_head.bias"),
    vocabSize,
    "LLaMA safetensors lm_head bias tensor",
  );
  const finalNormLength = safetensorsVectorLength(
    findSafetensorsTensorSpec(modelResourceSpecs, options.normTensor ?? "model.norm.weight"),
    "LLaMA safetensors final norm tensor",
  );
  if (finalNormLength !== hiddenSize) {
    throw new Error(`LLaMA safetensors final norm hidden size mismatch: ${finalNormLength} != ${hiddenSize}`);
  }
  const layerIndices = inferLlamaLayerIndicesFromSafetensorsSpecs(modelResourceSpecs);
  const layerCount = resolveLlamaSafetensorsLayerCount(options, metadata, config, layerIndices.length);
  const modelKind = inferZgmlModelKindFromLlamaSafetensors(options, metadata, config, layerCount);
  const useRopeLayers = resolveLlamaSafetensorsUseRopeLayers(options, metadata, config, layerCount);
  const attentionBiasLayers = [];
  const mlpBiasLayers = [];
  let qRows = null;
  let kRows = null;
  let vRows = null;
  let ffnSize = null;
  for (const layer of layerIndices) {
    const qProj = safetensorsMatrixShape(
      findSafetensorsTensorSpec(modelResourceSpecs, `model.layers.${layer}.self_attn.q_proj.weight`),
      `LLaMA safetensors layer ${layer} Q projection`,
    );
    const kProj = safetensorsMatrixShape(
      findSafetensorsTensorSpec(modelResourceSpecs, `model.layers.${layer}.self_attn.k_proj.weight`),
      `LLaMA safetensors layer ${layer} K projection`,
    );
    const vProj = safetensorsMatrixShape(
      findSafetensorsTensorSpec(modelResourceSpecs, `model.layers.${layer}.self_attn.v_proj.weight`),
      `LLaMA safetensors layer ${layer} V projection`,
    );
    const oProj = safetensorsMatrixShape(
      findSafetensorsTensorSpec(modelResourceSpecs, `model.layers.${layer}.self_attn.o_proj.weight`),
      `LLaMA safetensors layer ${layer} O projection`,
    );
    const gateProj = safetensorsMatrixShape(
      findSafetensorsTensorSpec(modelResourceSpecs, `model.layers.${layer}.mlp.gate_proj.weight`),
      `LLaMA safetensors layer ${layer} gate projection`,
    );
    const upProj = safetensorsMatrixShape(
      findSafetensorsTensorSpec(modelResourceSpecs, `model.layers.${layer}.mlp.up_proj.weight`),
      `LLaMA safetensors layer ${layer} up projection`,
    );
    const downProj = safetensorsMatrixShape(
      findSafetensorsTensorSpec(modelResourceSpecs, `model.layers.${layer}.mlp.down_proj.weight`),
      `LLaMA safetensors layer ${layer} down projection`,
    );
    const inputNormLength = safetensorsVectorLength(
      findSafetensorsTensorSpec(modelResourceSpecs, `model.layers.${layer}.input_layernorm.weight`),
      `LLaMA safetensors layer ${layer} input norm`,
    );
    const postNormLength = safetensorsVectorLength(
      findSafetensorsTensorSpec(modelResourceSpecs, `model.layers.${layer}.post_attention_layernorm.weight`),
      `LLaMA safetensors layer ${layer} post-attention norm`,
    );
    if (qProj.cols !== hiddenSize || kProj.cols !== hiddenSize || vProj.cols !== hiddenSize) {
      throw new Error(`LLaMA safetensors layer ${layer} hidden size mismatch`);
    }
    if (oProj.rows !== hiddenSize || oProj.cols !== qProj.rows) {
      throw new Error(`LLaMA safetensors layer ${layer} O projection shape mismatch: ${oProj.rows}x${oProj.cols}`);
    }
    if (inputNormLength !== hiddenSize || postNormLength !== hiddenSize) {
      throw new Error(`LLaMA safetensors layer ${layer} norm hidden size mismatch`);
    }
    if (gateProj.cols !== hiddenSize || upProj.cols !== hiddenSize) {
      throw new Error(`LLaMA safetensors layer ${layer} FFN input shape mismatch: ${gateProj.rows}x${gateProj.cols}/${upProj.rows}x${upProj.cols}`);
    }
    if (gateProj.rows !== upProj.rows) {
      throw new Error(`LLaMA safetensors layer ${layer} gate/up intermediate size mismatch: ${gateProj.rows} != ${upProj.rows}`);
    }
    if (downProj.rows !== hiddenSize || downProj.cols !== gateProj.rows) {
      throw new Error(`LLaMA safetensors layer ${layer} down projection shape mismatch: ${downProj.rows}x${downProj.cols}`);
    }
    if (validateLlamaSafetensorsBiasTensorGroup(
      modelResourceSpecs,
      layer,
      "attention",
      [qProj.rows, kProj.rows, vProj.rows, oProj.rows],
      attentionBiasHint,
    )) {
      attentionBiasLayers.push(layer);
    }
    if (validateLlamaSafetensorsBiasTensorGroup(
      modelResourceSpecs,
      layer,
      "mlp",
      [gateProj.rows, upProj.rows, downProj.rows],
      mlpBiasHint,
    )) {
      mlpBiasLayers.push(layer);
    }
    if (qRows === null) qRows = qProj.rows;
    if (kRows === null) kRows = kProj.rows;
    if (vRows === null) vRows = vProj.rows;
    if (ffnSize === null) ffnSize = gateProj.rows;
    if (qRows !== qProj.rows || kRows !== kProj.rows || vRows !== vProj.rows) {
      throw new Error("LLaMA safetensors Program requires uniform Q/K/V row widths across layers");
    }
    if (ffnSize !== gateProj.rows) {
      throw new Error("LLaMA safetensors Program requires uniform FFN intermediate size across layers");
    }
  }
  ffnSize = resolveLlamaSafetensorsIntermediateSize(options, metadata, config, ffnSize);
  const attentionShape = resolveLlamaSafetensorsAttentionHeadSize(options, metadata, config, {
    hiddenSize,
    kRows,
    qRows,
    vRows,
  });
  if (qkProjectionNorm) {
    for (const layer of layerIndices) {
      const qNormLength = safetensorsVectorLength(
        findSafetensorsTensorSpec(modelResourceSpecs, `model.layers.${layer}.self_attn.q_norm.weight`),
        `LLaMA safetensors layer ${layer} Q projection norm`,
      );
      const kNormLength = safetensorsVectorLength(
        findSafetensorsTensorSpec(modelResourceSpecs, `model.layers.${layer}.self_attn.k_norm.weight`),
        `LLaMA safetensors layer ${layer} K projection norm`,
      );
      if (qNormLength !== attentionShape.attentionHeadSize) {
        throw new Error(`LLaMA safetensors layer ${layer} Q projection norm length mismatch: ${qNormLength} != ${attentionShape.attentionHeadSize}`);
      }
      if (kNormLength !== attentionShape.attentionHeadSize) {
        throw new Error(`LLaMA safetensors layer ${layer} K projection norm length mismatch: ${kNormLength} != ${attentionShape.attentionHeadSize}`);
      }
    }
  }
  validateLlamaSafetensorsRopeDimensionCount(options, metadata, config, attentionShape.attentionHeadSize);
  const ropeBase = resolveLlamaSafetensorsRopeBase(options, metadata, config);
  const ropeScaling = resolveLlamaSafetensorsRopeScaling(options, metadata, config);
  validateLlamaSafetensorsRotaryInvFreqValues(source, specs, {
    attentionHeadSize: attentionShape.attentionHeadSize,
    layerIndices,
    ropeBase,
    ropeScaling,
  });
  const epsilon = resolveLlamaSafetensorsRmsNormEpsilon(options, metadata, config);
  const kBufferByteLength = contextLength * kRows * Float32Array.BYTES_PER_ELEMENT;
  const vBufferByteLength = contextLength * vRows * Float32Array.BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(kBufferByteLength) || !Number.isSafeInteger(vBufferByteLength)) {
    throw new Error("LLaMA safetensors Program K/V cache byteLength is too large");
  }
  return {
    ...attentionShape,
    contextLength,
    attentionBiasLayers: Object.freeze(attentionBiasLayers),
    hiddenSize,
    ffnSize,
    kvCacheRequirements: {
      layers: layerCount,
      kBufferByteLength,
      vBufferByteLength,
    },
    layerIndices,
    modelKind,
    modelResourceRequirements: normalizeLlamaModelResourceRequirements(modelResourceSpecs),
    epsilon,
    mlpBiasLayers: Object.freeze(mlpBiasLayers),
    ropeBase,
    ...ropeScaling,
    qkProjectionNorm,
    slidingWindow,
    tiedLmHead,
    useRopeLayers,
    vocabSize,
  };
}

function normalizeLlamaKvRequirements(value) {
  if (value === undefined || value === null) return null;
  const layers = value.layers ?? value.layerCount ?? value.len ?? value.length;
  const kBufferByteLength = value.kBufferByteLength ?? value.kByteLength ?? value.bufferByteLength;
  const vBufferByteLength = value.vBufferByteLength ?? value.vByteLength ?? value.bufferByteLength;
  if (!Number.isSafeInteger(layers) || layers <= 0) {
    throw new Error(`invalid LLaMA KV cache layer count: ${layers}`);
  }
  if (!Number.isSafeInteger(kBufferByteLength) || kBufferByteLength <= 0) {
    throw new Error(`invalid LLaMA K cache byte length: ${kBufferByteLength}`);
  }
  if (!Number.isSafeInteger(vBufferByteLength) || vBufferByteLength <= 0) {
    throw new Error(`invalid LLaMA V cache byte length: ${vBufferByteLength}`);
  }
  return { layers, kBufferByteLength, vBufferByteLength };
}

function formatLlamaKvRequirements(value) {
  const requirements = normalizeLlamaKvRequirements(value);
  return `layers=${requirements.layers} kBytes=${requirements.kBufferByteLength} vBytes=${requirements.vBufferByteLength}`;
}

function llamaKvRequirementsEqual(left, right) {
  return (
    left.layers === right.layers &&
    left.kBufferByteLength === right.kBufferByteLength &&
    left.vBufferByteLength === right.vBufferByteLength
  );
}

function assertLlamaKvRequirementsMatch(actual, expected, label) {
  const actualRequirements = normalizeLlamaKvRequirements(actual);
  const expectedRequirements = normalizeLlamaKvRequirements(expected);
  if (!llamaKvRequirementsEqual(actualRequirements, expectedRequirements)) {
    throw new Error(
      `${label} K/V requirements mismatch: ` +
      `resource Program ${formatLlamaKvRequirements(actualRequirements)} != ` +
      `native executable ${formatLlamaKvRequirements(expectedRequirements)}`,
    );
  }
}

function assertLlamaNativeProgramRequirementsMatch(program, nativeRequirements, label) {
  const modelKind = nativeRequirements?.modelKind;
  const outputLen = nativeRequirements?.outputLen;
  const outputByteLength = nativeRequirements?.outputByteLength;
  const contextLength = nativeRequirements?.contextLength;
  const scalarBytes = nativeRequirements?.scalarBytes;
  const tokenIdBytes = nativeRequirements?.tokenIdBytes;
  const batch = nativeRequirements?.batch;
  const maxTokenWindow = nativeRequirements?.maxTokenWindow;
  if (!Number.isSafeInteger(modelKind) || modelKind <= 0) {
    throw new Error(`${label} native executable has invalid model kind: ${modelKind}`);
  }
  if (scalarBytes !== Float32Array.BYTES_PER_ELEMENT) {
    throw new Error(`${label} scalar width mismatch: native executable scalar bytes=${scalarBytes}`);
  }
  if (tokenIdBytes !== Uint32Array.BYTES_PER_ELEMENT) {
    throw new Error(`${label} token-id width mismatch: native executable token bytes=${tokenIdBytes}`);
  }
  if (!Number.isSafeInteger(outputLen) || outputLen <= 0) {
    throw new Error(`${label} native executable has invalid output length: ${outputLen}`);
  }
  if (outputLen < program.vocabSize) {
    throw new Error(
      `${label} output requirements mismatch: ` +
      `resource Program vocab=${program.vocabSize} exceeds native executable output=${outputLen}`,
    );
  }
  if (outputByteLength !== outputLen * Float32Array.BYTES_PER_ELEMENT) {
    throw new Error(
      `${label} output byte requirements mismatch: ` +
      `native executable output bytes=${outputByteLength} for output=${outputLen}`,
    );
  }
  if (!Number.isSafeInteger(contextLength) || contextLength <= 0) {
    throw new Error(`${label} native executable has invalid context length: ${contextLength}`);
  }
  if (contextLength !== program.contextLength) {
    throw new Error(
      `${label} context requirements mismatch: ` +
      `resource Program context=${program.contextLength} != native executable context=${contextLength}`,
    );
  }
  if (batch !== 1) {
    throw new Error(`${label} batch requirements mismatch: native executable batch=${batch}`);
  }
  if (!Number.isSafeInteger(maxTokenWindow) || maxTokenWindow <= 0 || maxTokenWindow > contextLength) {
    throw new Error(
      `${label} token-window requirements mismatch: ` +
      `native executable maxTokenWindow=${maxTokenWindow} context=${contextLength}`,
    );
  }
}

function assertLlamaNativeModelCompatibility(compatibility, label) {
  if (compatibility?.compatible !== true) {
    throw new Error(
      `${label} model compatibility mismatch: ` +
      `native executable model=${compatibility?.programModelKind} does not accept bound model=${compatibility?.modelKind}`,
    );
  }
}

function assertLlamaNativeOutputLength(outputLen, nativeRequirements, label) {
  const nativeOutputLen = nativeRequirements?.outputLen;
  if (!Number.isSafeInteger(outputLen) || outputLen <= 0) {
    throw new Error(`invalid ${label} output length: ${outputLen}`);
  }
  if (Number.isSafeInteger(nativeOutputLen) && outputLen < nativeOutputLen) {
    throw new Error(
      `${label} output requirements mismatch: ` +
      `bound output length=${outputLen} is smaller than native executable output=${nativeOutputLen}`,
    );
  }
}

function hasExplicitLlamaTokenExecutor(options) {
  return (
    hasLlamaExecutorOption(options, "tokenExecutor") ||
    hasLlamaExecutorOption(options, "executorPreset") ||
    hasLlamaExecutorOption(options, "executionPreset") ||
    hasLlamaExecutorOption(options, "execution") ||
    hasLlamaExecutorOption(options, "executor")
  );
}

export class WasmWebGpuTinyLinearProgram {
  constructor(options = {}) {
    this.device = options.device;
    this.resourceBridge = options.resourceBridge;
    this.sessionBindingBridge = options.sessionBindingBridge;
    this.runtime = options.runtime ?? null;
    this.programHandle = options.programHandle ?? options.program;
    this.rows = options.rows;
    this.cols = options.cols;
    this.freeSession = options.freeSession ?? (() => {});
    if (!(this.device instanceof WasmWebGpuDevice)) {
      throw new Error("WasmWebGpuTinyLinearProgram needs a WasmWebGpuDevice");
    }
    if (!(this.resourceBridge instanceof WasmExternalResourceBridge)) {
      throw new Error("WasmWebGpuTinyLinearProgram needs a WasmExternalResourceBridge");
    }
    if (!(this.sessionBindingBridge instanceof WasmSessionBindingBridge)) {
      throw new Error("WasmWebGpuTinyLinearProgram needs a WasmSessionBindingBridge");
    }
    if (this.runtime !== null && !(this.runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("WasmWebGpuTinyLinearProgram runtime must be a WasmWebGpuSessionRuntime");
    }
    assertHandle(this.programHandle);
    if (!Number.isSafeInteger(this.rows) || this.rows <= 0 || !Number.isSafeInteger(this.cols) || this.cols <= 0) {
      throw new Error(`invalid tiny-linear Program shape: rows=${this.rows} cols=${this.cols}`);
    }
    this.executor = new WasmWebGpuTinyLinearExecutor({ device: this.device });
  }

  bindingRequirements() {
    return describeProgramBindingRequirements({
      persistent: [
        {
          role: "tiny-linear.weights",
          access: "read",
          elementCount: this.rows * this.cols,
          byteLength: this.rows * this.cols * Float32Array.BYTES_PER_ELEMENT,
          shape: [this.rows, this.cols],
        },
        {
          role: "tiny-linear.bias",
          access: "read",
          elementCount: this.rows,
          byteLength: this.rows * Float32Array.BYTES_PER_ELEMENT,
          shape: [this.rows],
        },
      ],
      stepInputs: [
        {
          role: "tiny-linear.input",
          access: "read",
          elementCount: this.cols,
          byteLength: this.cols * Float32Array.BYTES_PER_ELEMENT,
          shape: [this.cols],
        },
      ],
      stepOutputs: [
        {
          role: "tiny-linear.output",
          access: "write",
          elementCount: this.rows,
          byteLength: this.rows * Float32Array.BYTES_PER_ELEMENT,
          shape: [this.rows],
        },
      ],
    });
  }

  inspect() {
    return Object.freeze({
      kind: "tiny-linear-resource-program",
      rows: this.rows,
      cols: this.cols,
      inputElementCount: this.cols,
      outputElementCount: this.rows,
      weightsElementCount: this.rows * this.cols,
      biasElementCount: this.rows,
      inputByteLength: this.cols * Float32Array.BYTES_PER_ELEMENT,
      outputByteLength: this.rows * Float32Array.BYTES_PER_ELEMENT,
      weightsByteLength: this.rows * this.cols * Float32Array.BYTES_PER_ELEMENT,
      biasByteLength: this.rows * Float32Array.BYTES_PER_ELEMENT,
      ...this.bindingRequirements(),
    });
  }

  createBuffer(kind, options = {}) {
    const elementCount = tinyLinearElementCount(kind, this.rows, this.cols);
    return this.device.createBuffer({
      label: options.label ?? `zgml.wasm.webgpu.tiny-linear.${kind}`,
      byteLength: elementCount * Float32Array.BYTES_PER_ELEMENT,
      usage: options.usage ?? webgpuRequiredUsageFlags("readwrite"),
    });
  }

  wrapBuffer(kind, resource, options = {}) {
    const elementCount = tinyLinearElementCount(kind, this.rows, this.cols);
    const byteLength = elementCount * Float32Array.BYTES_PER_ELEMENT;
    const descriptor = storageResourceDescriptor(this.device, resource, {
      access: tinyLinearAccess(kind),
      byteLength,
      ...options,
    }, `tiny-linear ${kind}`, byteLength);
    const buffer = this.resourceBridge.wrapResourceBuffer(descriptor);
    return {
      kind,
      elementCount,
      resource: descriptor.resource,
      buffer,
      descriptor,
    };
  }

  createResourceBuffer(kind, options = {}) {
    const elementCount = tinyLinearElementCount(kind, this.rows, this.cols);
    const byteLength = elementCount * Float32Array.BYTES_PER_ELEMENT;
    const makeResource = options.resource ?? options.externalResource;
    const resource = typeof makeResource === "function"
      ? makeResource({ kind, elementCount, byteLength, program: this })
      : (makeResource ?? options.buffer ?? this.createBuffer(kind, options));
    if (isResourceBinding(resource)) return this.normalizeResourceBinding(kind, resource);
    return this.wrapBuffer(kind, resource, options);
  }

  createWeightsBuffer(options = {}) {
    return this.createResourceBuffer("weights", options);
  }

  createBiasBuffer(options = {}) {
    return this.createResourceBuffer("bias", options);
  }

  createInputBuffer(options = {}) {
    return this.createResourceBuffer("input", options);
  }

  createOutputBuffer(options = {}) {
    return this.createResourceBuffer("output", options);
  }

  normalizeResourceBinding(kind, value) {
    const binding = resourceBinding(value, `tiny-linear ${kind}`);
    const expectedElementCount = tinyLinearElementCount(kind, this.rows, this.cols);
    if (binding.kind !== undefined && binding.kind !== kind) {
      throw new Error(`tiny-linear ${kind} binding has wrong kind: ${binding.kind}`);
    }
    if (binding.elementCount !== undefined && binding.elementCount !== expectedElementCount) {
      throw new Error(`tiny-linear ${kind} binding has wrong element count: ${binding.elementCount} != ${expectedElementCount}`);
    }
    return validateStorageResourceBinding(
      this.device,
      binding,
      `tiny-linear ${kind}`,
      expectedElementCount * Float32Array.BYTES_PER_ELEMENT,
      tinyLinearAccess(kind),
    );
  }

  normalizeBuffer(kind, value) {
    if (value === undefined || value === null) return this.createResourceBuffer(kind);
    if (isResourceBinding(value)) return this.normalizeResourceBinding(kind, value);
    return this.wrapBuffer(kind, value);
  }

  bind(resources = {}) {
    const bindings = {
      weights: this.normalizeBuffer("weights", resources.weights),
      bias: this.normalizeBuffer("bias", resources.bias),
      input: this.normalizeBuffer("input", resources.input),
      output: this.normalizeBuffer("output", resources.output),
    };
    const table = this.device.resourceTable([
      { role: "tiny-linear.weights", descriptor: bindings.weights.descriptor },
      { role: "tiny-linear.bias", descriptor: bindings.bias.descriptor },
      { role: "tiny-linear.input", descriptor: bindings.input.descriptor },
      { role: "tiny-linear.output", descriptor: bindings.output.descriptor },
    ]);
    const bindDesc = this.sessionBindingBridge.bufferBindDesc({
      weights: bindings.weights.buffer,
      weightsLen: bindings.weights.elementCount,
      bias: bindings.bias.buffer,
      biasLen: bindings.bias.elementCount,
      input: bindings.input.buffer,
      inputLen: bindings.input.elementCount,
      output: bindings.output.buffer,
      outputLen: bindings.output.elementCount,
    });
    const sessionHandle = this.sessionBindingBridge.bindBufferSession(this.programHandle, bindDesc);
    const session = new WasmWebGpuTinyLinearSession({
      program: this,
      sessionHandle,
      bindings,
      table,
    });
    if (this.runtime) {
      try {
        this.runtime.register(session);
      } catch (err) {
        const exportsRef = readOption(this.sessionBindingBridge.exports);
        if (exportsRef && typeof exportsRef.zgml_session_free === "function") {
          exportsRef.zgml_session_free(sessionHandle);
        }
        session.dispose();
        throw err;
      }
    }
    return session;
  }

  unregister(sessionOrHandle) {
    if (this.runtime) this.runtime.unregister(sessionOrHandle);
  }

  destroy() {
    this.executor.destroy();
  }
}

export class WasmWebGpuLlamaResourceProgram {
  static inferSafetensorsShape(source, options = {}) {
    return inferLlamaResourceProgramShapeFromSafetensors(source, options);
  }

  static fromSafetensors(options = {}) {
    const source = options.safetensors ?? options.safetensorsData ?? options.modelResources ?? options.weights;
    if (source === undefined || source === null) {
      throw new Error("WasmWebGpuLlamaResourceProgram.fromSafetensors needs safetensors bytes");
    }
    const modelResourceOptions = {
      materializeFloatDtypes: true,
      ...(options.modelResourceOptions ?? {}),
    };
    const shape = inferLlamaResourceProgramShapeFromSafetensors(source, {
      attentionHeadSize:
        options.attentionHeadSize ??
        options.headSize ??
        options.executorOptions?.attentionHeadSize ??
        options.executorOptions?.headSize ??
        options.tokenExecutorOptions?.attentionHeadSize ??
        options.tokenExecutorOptions?.headSize,
      ropeBase:
        options.ropeBase ??
        options.ropeTheta ??
        options.executorOptions?.ropeBase ??
        options.executorOptions?.ropeTheta ??
        options.tokenExecutorOptions?.ropeBase ??
        options.tokenExecutorOptions?.ropeTheta,
      ropeScale:
        options.ropeScale ??
        options.rope_scale ??
        options.ropeScalingFactor ??
        options.rope_scaling_factor ??
        options.executorOptions?.ropeScale ??
        options.executorOptions?.rope_scale ??
        options.executorOptions?.ropeScalingFactor ??
        options.executorOptions?.rope_scaling_factor ??
        options.tokenExecutorOptions?.ropeScale ??
        options.tokenExecutorOptions?.rope_scale ??
        options.tokenExecutorOptions?.ropeScalingFactor ??
        options.tokenExecutorOptions?.rope_scaling_factor,
      ropeKind:
        options.ropeKind ??
        options.ropeType ??
        options.rope_kind ??
        options.rope_type ??
        options.executorOptions?.ropeKind ??
        options.executorOptions?.ropeType ??
        options.executorOptions?.rope_kind ??
        options.executorOptions?.rope_type ??
        options.tokenExecutorOptions?.ropeKind ??
        options.tokenExecutorOptions?.ropeType ??
        options.tokenExecutorOptions?.rope_kind ??
        options.tokenExecutorOptions?.rope_type,
      ropeLowFreqFactor:
        options.ropeLowFreqFactor ??
        options.rope_low_freq_factor ??
        options.executorOptions?.ropeLowFreqFactor ??
        options.executorOptions?.rope_low_freq_factor ??
        options.tokenExecutorOptions?.ropeLowFreqFactor ??
        options.tokenExecutorOptions?.rope_low_freq_factor,
      ropeHighFreqFactor:
        options.ropeHighFreqFactor ??
        options.rope_high_freq_factor ??
        options.executorOptions?.ropeHighFreqFactor ??
        options.executorOptions?.rope_high_freq_factor ??
        options.tokenExecutorOptions?.ropeHighFreqFactor ??
        options.tokenExecutorOptions?.rope_high_freq_factor,
      ropeOriginalContextLength:
        options.ropeOriginalContextLength ??
        options.rope_original_context_length ??
        options.executorOptions?.ropeOriginalContextLength ??
        options.executorOptions?.rope_original_context_length ??
        options.tokenExecutorOptions?.ropeOriginalContextLength ??
        options.tokenExecutorOptions?.rope_original_context_length,
      epsilon:
        options.epsilon ??
        options.rmsNormEpsilon ??
        options.rmsNormEps ??
        options.executorOptions?.epsilon ??
        options.executorOptions?.rmsNormEpsilon ??
        options.executorOptions?.rmsNormEps ??
        options.tokenExecutorOptions?.epsilon ??
        options.tokenExecutorOptions?.rmsNormEpsilon ??
        options.tokenExecutorOptions?.rmsNormEps,
      intermediateSize:
        options.intermediateSize ??
        options.intermediate_size ??
        options.ffnSize ??
        options.ffn_size ??
        options.executorOptions?.intermediateSize ??
        options.executorOptions?.intermediate_size ??
        options.executorOptions?.ffnSize ??
        options.executorOptions?.ffn_size ??
        options.tokenExecutorOptions?.intermediateSize ??
        options.tokenExecutorOptions?.intermediate_size ??
        options.tokenExecutorOptions?.ffnSize ??
        options.tokenExecutorOptions?.ffn_size,
      slidingWindow:
        options.slidingWindow ??
        options.sliding_window ??
        options.attentionSlidingWindow ??
        options.attention_sliding_window ??
        options.executorOptions?.slidingWindow ??
        options.executorOptions?.sliding_window ??
        options.executorOptions?.attentionSlidingWindow ??
        options.executorOptions?.attention_sliding_window ??
        options.tokenExecutorOptions?.slidingWindow ??
        options.tokenExecutorOptions?.sliding_window ??
        options.tokenExecutorOptions?.attentionSlidingWindow ??
        options.tokenExecutorOptions?.attention_sliding_window,
      useSlidingWindow:
        options.useSlidingWindow ??
        options.use_sliding_window ??
        options.executorOptions?.useSlidingWindow ??
        options.executorOptions?.use_sliding_window ??
        options.tokenExecutorOptions?.useSlidingWindow ??
        options.tokenExecutorOptions?.use_sliding_window,
      noRopeLayerInterval:
        options.noRopeLayerInterval ??
        options.no_rope_layer_interval ??
        options.executorOptions?.noRopeLayerInterval ??
        options.executorOptions?.no_rope_layer_interval ??
        options.tokenExecutorOptions?.noRopeLayerInterval ??
        options.tokenExecutorOptions?.no_rope_layer_interval,
      noRopeLayers:
        options.noRopeLayers ??
        options.no_rope_layers ??
        options.useRopeLayers ??
        options.use_rope_layers ??
        options.executorOptions?.noRopeLayers ??
        options.executorOptions?.no_rope_layers ??
        options.executorOptions?.useRopeLayers ??
        options.executorOptions?.use_rope_layers ??
        options.tokenExecutorOptions?.noRopeLayers ??
        options.tokenExecutorOptions?.no_rope_layers ??
        options.tokenExecutorOptions?.useRopeLayers ??
        options.tokenExecutorOptions?.use_rope_layers,
      contextLength:
        options.contextLength ??
        options.context_length ??
        options.maxPositionEmbeddings ??
        options.max_position_embeddings ??
        options.seqLength ??
        options.seq_length ??
        options.nCtx ??
        options.n_ctx,
      hiddenSize: options.hiddenSize ?? options.hidden ?? options.hidden_size,
      vocabSize: options.vocabSize ?? options.vocab_size,
      numHiddenLayers: options.numHiddenLayers ?? options.num_hidden_layers ?? options.nLayers ?? options.n_layers ?? options.nLayer ?? options.n_layer,
      embeddingTensor: options.embeddingTensor,
      lmHeadBiasTensor: options.lmHeadBiasTensor,
      lmHeadTensor: options.lmHeadTensor,
      modelKind: options.modelKind ?? options.model_kind ?? options.kind,
      normTensor: options.normTensor,
      modelResourceOptions,
      qkProjectionNorm:
        options.qkProjectionNorm ??
        options.qk_projection_norm ??
        options.qk_norm ??
        options.executorOptions?.qkProjectionNorm ??
        options.executorOptions?.qk_projection_norm ??
        options.executorOptions?.qk_norm ??
        options.tokenExecutorOptions?.qkProjectionNorm ??
        options.tokenExecutorOptions?.qk_projection_norm ??
        options.tokenExecutorOptions?.qk_norm,
      safetensorsIndex: options.safetensorsIndex ?? options.index,
      allowTiedLmHead: options.allowTiedLmHead ?? options.tiedLmHead ?? options.tieLmHead ?? options.tie_word_embeddings ?? options.tieWordEmbeddings,
      numAttentionHeads: options.numAttentionHeads ?? options.attentionHeads ?? options.nHeads ?? options.n_heads,
      numKeyValueHeads: options.numKeyValueHeads ?? options.keyValueHeads ?? options.nKvHeads ?? options.n_kv_heads,
      config: options.config ?? options.modelConfig ?? options.configJson ?? options.configJSON ?? options.hfConfig ?? options.huggingFaceConfig,
    });
    const hasExplicitExecutor = hasExplicitLlamaTokenExecutor(options);
    return new WasmWebGpuLlamaResourceProgram({
      ...options,
      allowExtraModelResources: options.allowExtraModelResources ?? false,
      defaultModel: options.defaultModel ?? options.model,
      defaultModelResourceOptions: options.defaultModelResourceOptions ?? {
        ...modelResourceOptions,
        allowTiedLmHead: shape.tiedLmHead,
        embeddingTensor: options.embeddingTensor,
        lmHeadBiasTensor: options.lmHeadBiasTensor,
        lmHeadTensor: options.lmHeadTensor,
        normTensor: options.normTensor,
        qkProjectionNorm: shape.qkProjectionNorm,
      },
      defaultModelResources: options.defaultModelResources ?? (isSafetensorsShardSourceList(source) ? {
        safetensors: source,
        ...(options.safetensorsIndex !== undefined ? { safetensorsIndex: options.safetensorsIndex } : {}),
        ...(options.index !== undefined ? { index: options.index } : {}),
      } : source),
      requireGpuDispatch: resolveLlamaRequireGpuDispatch(options, true),
      ...(hasExplicitExecutor ? {} : { executorPreset: "tiny-llama-block-pipeline" }),
      attentionHeadSize: options.attentionHeadSize ?? options.headSize ?? shape.attentionHeadSize,
      attentionBiasLayers: options.attentionBiasLayers ?? options.attention_bias_layers ?? shape.attentionBiasLayers,
      embeddingRole: options.embeddingRole ?? options.embedding_role ?? options.embeddingTensor,
      contextLength:
        options.contextLength ??
        options.context_length ??
        options.maxPositionEmbeddings ??
        options.max_position_embeddings ??
        options.seqLength ??
        options.seq_length ??
        options.nCtx ??
        options.n_ctx ??
        shape.contextLength,
      hiddenSize: shape.hiddenSize,
      kvCacheRequirements: options.kvCacheRequirements ?? options.kvRequirements ?? shape.kvCacheRequirements,
      modelKind: options.modelKind ?? options.model_kind ?? options.kind ?? shape.modelKind,
      requiredModelResources:
        options.requiredModelResources ??
        options.requiredModelResourceManifest ??
        options.modelResourceRequirements ??
        options.modelResourceManifest ??
        shape.modelResourceRequirements,
      epsilon: options.epsilon ?? options.rmsNormEpsilon ?? options.rmsNormEps ?? shape.epsilon,
      ffnSize: options.ffnSize ?? options.ffn_size ?? options.intermediateSize ?? options.intermediate_size ?? shape.ffnSize,
      ropeBase: options.ropeBase ?? options.ropeTheta ?? shape.ropeBase,
      ropeHighFreqFactor: options.ropeHighFreqFactor ?? options.rope_high_freq_factor ?? shape.ropeHighFreqFactor,
      ropeKind: options.ropeKind ?? options.ropeType ?? options.rope_kind ?? options.rope_type ?? shape.ropeKind,
      ropeLowFreqFactor: options.ropeLowFreqFactor ?? options.rope_low_freq_factor ?? shape.ropeLowFreqFactor,
      ropeOriginalContextLength: options.ropeOriginalContextLength ?? options.rope_original_context_length ?? shape.ropeOriginalContextLength,
      ropeScale: options.ropeScale ?? options.rope_scale ?? options.ropeScalingFactor ?? options.rope_scaling_factor ?? shape.ropeScale,
      slidingWindow: options.slidingWindow ?? options.sliding_window ?? options.attentionSlidingWindow ?? options.attention_sliding_window ?? shape.slidingWindow,
      useRopeLayers: options.useRopeLayers ?? options.use_rope_layers ?? options.noRopeLayers ?? options.no_rope_layers ?? shape.useRopeLayers,
      lmHeadBiasRole: options.lmHeadBiasRole ?? options.lm_head_bias_role ?? options.lmHeadBiasTensor,
      lmHeadRole: options.lmHeadRole ?? options.lm_head_role ?? options.lmHeadTensor,
      mlpBiasLayers: options.mlpBiasLayers ?? options.mlp_bias_layers ?? shape.mlpBiasLayers,
      normRole: options.normRole ?? options.norm_role ?? options.normTensor,
      qkProjectionNorm: options.qkProjectionNorm ?? options.qk_projection_norm ?? options.qk_norm ?? shape.qkProjectionNorm,
      vocabSize: shape.vocabSize,
    });
  }

  constructor(options = {}) {
    this.device = options.device;
    this.resourceBridge = options.resourceBridge ?? null;
    this.sessionBindingBridge = options.sessionBindingBridge;
    this.runtime = options.runtime ?? null;
    this.programHandle = options.programHandle ?? options.program;
    this.freeSession = options.freeSession ?? ((sessionHandle) => {
      const exportsRef = readOption(this.sessionBindingBridge.exports);
      if (exportsRef && typeof exportsRef.zgml_session_free === "function") {
        return exportsRef.zgml_session_free(sessionHandle);
      }
      return undefined;
    });
    this.vocabSize = options.vocabSize;
    this.modelKind = normalizeZgmlModelKind(
      options.modelKind ?? options.model_kind ?? options.kind ?? zgmlModelTinyLlamaKind,
      "LLaMA resource Program model kind",
    );
    this.contextLength = options.contextLength;
    this.hiddenSize = options.hiddenSize ?? options.hidden ?? null;
    this.ffnSize = options.ffnSize ?? options.ffn_size ?? options.intermediateSize ?? options.intermediate_size ?? null;
    this.attentionBiasLayers = normalizeOptionalLayerIndexList(
      options.attentionBiasLayers ?? options.attention_bias_layers,
      "LLaMA resource Program attention bias layers",
    );
    this.mlpBiasLayers = normalizeOptionalLayerIndexList(
      options.mlpBiasLayers ?? options.mlp_bias_layers,
      "LLaMA resource Program MLP bias layers",
    );
    this.kvRequirements = normalizeLlamaKvRequirements(options.kvCacheRequirements ?? options.kvRequirements);
    this.tokenExecutor = resolveLlamaTokenExecutor(options, this.kvRequirements);
    this.requireGpuDispatch = resolveLlamaRequireGpuDispatch(options);
    this.requiredModelResources = normalizeLlamaModelResourceRequirements(
      options.requiredModelResources ??
      options.requiredModelResourceManifest ??
      options.modelResourceRequirements ??
      options.modelResourceManifest,
    );
    this.allowExtraModelResources = options.allowExtraModelResources ?? true;
    this.defaultModelHandle = options.defaultModel ?? 0;
    this.defaultModelResourceSource = options.defaultModelResources ?? options.defaultWeights ?? null;
    this.defaultModelResourceOptions = options.defaultModelResourceOptions ?? null;
    if (this.requiredModelResources.length === 0 && this.defaultModelResourceSource !== null && this.defaultModelResourceSource !== undefined) {
      const defaultRequirements = this.modelResourceRequirementsFromValue(this.defaultModelResourceSource, {
        ...(this.defaultModelResourceOptions ?? {}),
        resourceMapManifest: true,
      });
      if (Array.isArray(defaultRequirements)) this.requiredModelResources = defaultRequirements;
    }
    if (!(this.device instanceof WasmWebGpuDevice)) {
      throw new Error("WasmWebGpuLlamaResourceProgram needs a WasmWebGpuDevice");
    }
    if (this.resourceBridge !== null && !(this.resourceBridge instanceof WasmExternalResourceBridge)) {
      throw new Error("WasmWebGpuLlamaResourceProgram resourceBridge must be a WasmExternalResourceBridge");
    }
    if (!(this.sessionBindingBridge instanceof WasmSessionBindingBridge)) {
      throw new Error("WasmWebGpuLlamaResourceProgram needs a WasmSessionBindingBridge");
    }
    if (this.runtime !== null && !(this.runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("WasmWebGpuLlamaResourceProgram runtime must be a WasmWebGpuSessionRuntime");
    }
    assertHandle(this.programHandle);
    if (!Number.isSafeInteger(this.vocabSize) || this.vocabSize <= 0) {
      throw new Error(`invalid LLaMA vocab size: ${this.vocabSize}`);
    }
    if (!Number.isSafeInteger(this.contextLength) || this.contextLength <= 0) {
      throw new Error(`invalid LLaMA context length: ${this.contextLength}`);
    }
    if (this.hiddenSize !== null && (!Number.isSafeInteger(this.hiddenSize) || this.hiddenSize <= 0)) {
      throw new Error(`invalid LLaMA hidden size: ${this.hiddenSize}`);
    }
    if (this.ffnSize !== null && (!Number.isSafeInteger(this.ffnSize) || this.ffnSize <= 0)) {
      throw new Error(`invalid LLaMA FFN intermediate size: ${this.ffnSize}`);
    }
    if (typeof this.allowExtraModelResources !== "boolean") {
      throw new Error("WasmWebGpuLlamaResourceProgram allowExtraModelResources must be a boolean");
    }
    if (
      this.tokenExecutor !== null &&
      typeof this.tokenExecutor !== "function" &&
      typeof this.tokenExecutor.executeTokensSync !== "function"
    ) {
      throw new Error("WasmWebGpuLlamaResourceProgram tokenExecutor must be a function or expose executeTokensSync");
    }
  }

  bindingRequirements() {
    const kvRequirements = this.kvRequirements;
    const persistent = this.requiredModelResources.map((entry) => ({
      access: "read",
      byteLength: entry.byteLength,
      dtype: entry.dtype,
      elementCount: entry.elementCount,
      role: entry.role,
      shape: entry.shape,
    }));
    const layers = kvRequirements?.layers ?? 0;
    const kByteLength = kvRequirements?.kBufferByteLength ?? 0;
    const vByteLength = kvRequirements?.vBufferByteLength ?? 0;
    for (let layer = 0; layer < layers; layer += 1) {
      persistent.push({
        access: "readwrite",
        byteLength: kByteLength,
        elementCount: Math.floor(kByteLength / Float32Array.BYTES_PER_ELEMENT),
        role: `llama.k.${layer}`,
        shape: [Math.floor(kByteLength / Float32Array.BYTES_PER_ELEMENT)],
      });
      persistent.push({
        access: "readwrite",
        byteLength: vByteLength,
        elementCount: Math.floor(vByteLength / Float32Array.BYTES_PER_ELEMENT),
        role: `llama.v.${layer}`,
        shape: [Math.floor(vByteLength / Float32Array.BYTES_PER_ELEMENT)],
      });
    }
    return describeProgramBindingRequirements({
      persistent,
      stepInputs: [
        {
          access: "read",
          byteLength: this.contextLength * Uint32Array.BYTES_PER_ELEMENT,
          dtype: "u32",
          elementCount: this.contextLength,
          role: "llama.tokens",
          shape: [this.contextLength],
        },
      ],
      stepOutputs: [
        {
          access: "write",
          byteLength: this.outputByteLength(),
          elementCount: this.vocabSize,
          role: "llama.output",
          shape: [this.vocabSize],
        },
      ],
    });
  }

  inspect() {
    const requiredGpuStorageBufferCount = llamaExecutorRequiredGpuStorageBufferCount(this.tokenExecutor);
    const maxStorageBuffersPerShaderStage = this.device.maxStorageBuffersPerShaderStage();
    const canCreateGpuBuffers = this.device.canCreateGpuBuffers();
    const canBindRequiredGpuStorageBuffers =
      requiredGpuStorageBufferCount !== null &&
      this.device.canBindStorageBuffers(requiredGpuStorageBufferCount);
    const kvRequirements = this.kvRequirements;
    return Object.freeze({
      allowExtraModelResources: this.allowExtraModelResources,
      attentionBiasLayers: this.attentionBiasLayers === null ? null : Object.freeze(this.attentionBiasLayers.slice()),
      canBindRequiredGpuStorageBuffers,
      canCreateGpuBuffers,
      contextLength: this.contextLength,
      defaultModelHandle: this.defaultModelHandle,
      ...llamaExecutionCoverageInspection(this.tokenExecutor),
      executionMode: this.tokenExecutor === null ? "resource-probe" : "proof-executor",
      ffnSize: this.ffnSize,
      gpuDispatchRequired: this.requireGpuDispatch,
      hasDefaultModelResources: this.defaultModelResourceSource !== null && this.defaultModelResourceSource !== undefined,
      hiddenSize: this.hiddenSize,
      kind: "llama-resource-program",
      kvCache: Object.freeze({
        kBufferByteLength: kvRequirements?.kBufferByteLength ?? 0,
        layers: kvRequirements?.layers ?? 0,
        vBufferByteLength: kvRequirements?.vBufferByteLength ?? 0,
      }),
      maxStorageBuffersPerShaderStage,
      modelKind: this.modelKind,
      mlpBiasLayers: this.mlpBiasLayers === null ? null : Object.freeze(this.mlpBiasLayers.slice()),
      outputByteLength: this.outputByteLength(),
      requiredGpuStorageBufferCount,
      requiredModelResourceCount: this.requiredModelResources.length,
      requiredModelResourceRoles: Object.freeze(this.requiredModelResources.map((entry) => entry.role)),
      requiredModelResources: Object.freeze(
        this.requiredModelResources.map((entry) => llamaModelResourceRequirementInspection(entry)),
      ),
      terminalActivationByteLength: this.hiddenSize === null ? null : this.activationByteLength(),
      tokenExecutorKind: llamaTokenExecutorInspectionKind(this.tokenExecutor),
      vocabSize: this.vocabSize,
      ...this.bindingRequirements(),
    });
  }

  outputByteLength() {
    return this.vocabSize * Float32Array.BYTES_PER_ELEMENT;
  }

  activationByteLength(options = {}) {
    const hiddenSize = options.hiddenSize ?? options.hidden ?? this.hiddenSize;
    const contextLength = options.contextLength ?? this.contextLength;
    if (!Number.isSafeInteger(hiddenSize) || hiddenSize <= 0) {
      throw new Error(`invalid LLaMA activation hidden size: ${hiddenSize}`);
    }
    if (!Number.isSafeInteger(contextLength) || contextLength <= 0) {
      throw new Error(`invalid LLaMA activation context length: ${contextLength}`);
    }
    const elementCount = hiddenSize * contextLength;
    if (!Number.isSafeInteger(elementCount)) {
      throw new Error("LLaMA activation buffer element count is too large");
    }
    const byteLength = elementCount * Float32Array.BYTES_PER_ELEMENT;
    if (!Number.isSafeInteger(byteLength)) {
      throw new Error("LLaMA activation buffer byteLength is too large");
    }
    return byteLength;
  }

  requireResourceBridge() {
    if (!(this.resourceBridge instanceof WasmExternalResourceBridge)) {
      throw new Error("WasmWebGpuLlamaResourceProgram needs a resourceBridge to create or wrap resources");
    }
    return this.resourceBridge;
  }

  requireKvRequirements() {
    if (!this.kvRequirements) {
      throw new Error("WasmWebGpuLlamaResourceProgram needs kvCacheRequirements to create or wrap raw K/V resources");
    }
    return this.kvRequirements;
  }

  createRawBuffer(kind, byteLength, options = {}) {
    return this.device.createBuffer({
      label: options.label ?? `zgml.wasm.webgpu.llama.${kind}`,
      byteLength,
      usage: options.usage ?? webgpuRequiredUsageFlags("readwrite"),
    });
  }

  wrapBuffer(kind, resource, options = {}) {
    const byteLength = options.byteLength;
    assertFloat32ByteLength(byteLength, `LLaMA ${kind} resource`);
    const access = options.access ?? (kind === "output" ? "write" : "readwrite");
    const descriptor = storageResourceDescriptor(this.device, resource, {
      ...options,
      access,
      byteLength,
    }, `LLaMA ${kind}`, byteLength);
    const buffer = this.requireResourceBridge().wrapResourceBuffer(descriptor);
    return {
      kind,
      elementCount: Math.floor(byteLength / Float32Array.BYTES_PER_ELEMENT),
      resource: descriptor.resource,
      buffer,
      descriptor,
    };
  }

  wrapModelResource(role, resource, options = {}) {
    const dtype = normalizeLlamaModelResourceDtype(options.dtype ?? "f32");
    const shapeInfo = normalizeLlamaModelResourceShape(options.shape);
    const elementCount = explicitLlamaModelResourceElementCount(options) ?? shapeInfo?.elementCount;
    const byteLength = llamaModelResourceByteLength({ ...options, dtype, elementCount }, resource);
    const descriptor = storageResourceDescriptor(this.device, resource, {
      ...options,
      access: options.access ?? "read",
      byteLength,
    }, `LLaMA model resource ${role}`, byteLength);
    validateResourceAccess(descriptor, "read", `LLaMA model resource ${role}`);
    const buffer = this.requireResourceBridge().wrapResourceBuffer(descriptor);
    return {
      role,
      byteLength,
      dataByteLength: options.dataByteLength,
      dataByteOffset: options.dataByteOffset,
      dtype,
      elementCount,
      resource: descriptor.resource,
      buffer,
      descriptor,
      safetensorsShardIndex: options.safetensorsShardIndex,
      safetensorsShardName: options.safetensorsShardName,
      shape: shapeInfo?.shape,
      sourceByteLength: options.sourceByteLength,
      sourceDtype: options.sourceDtype === undefined ? undefined : normalizeLlamaModelResourceDtype(options.sourceDtype),
    };
  }

  normalizeModelResourceEntry(entry, index) {
    const role = normalizeLlamaModelResourceRole(entry?.role ?? entry?.name ?? entry?.tensor ?? entry?.kind, index);
    const dtype = normalizeLlamaModelResourceDtype(entry?.dtype ?? "f32");
    const shapeInfo = normalizeLlamaModelResourceShape(entry?.shape);
    const elementCount = explicitLlamaModelResourceElementCount(entry) ?? shapeInfo?.elementCount;
    const source = isResourceBinding(entry)
      ? entry
      : (entry?.resource ?? entry?.gpuBuffer ?? entry?.buffer ?? entry);
    const byteLength = llamaModelResourceByteLength({ ...entry, dtype, elementCount }, source);
    if (isResourceBinding(source)) {
      const binding = validateStorageResourceBinding(
        this.device,
        resourceBinding(source, `LLaMA model resource ${role}`),
        `LLaMA model resource ${role}`,
        byteLength,
        "read",
      );
      return {
        role,
        byteLength,
        dataByteLength: entry?.dataByteLength,
        dataByteOffset: entry?.dataByteOffset,
        dtype,
        elementCount,
        resource: binding.descriptor.resource,
        buffer: binding.buffer,
        descriptor: binding.descriptor,
        safetensorsShardIndex: entry?.safetensorsShardIndex,
        safetensorsShardName: entry?.safetensorsShardName,
        shape: shapeInfo?.shape,
        sourceByteLength: entry?.sourceByteLength,
        sourceDtype: entry?.sourceDtype === undefined ? undefined : normalizeLlamaModelResourceDtype(entry.sourceDtype),
      };
    }
    return this.wrapModelResource(role, source, {
      access: entry?.access ?? "read",
      byteLength,
      dataByteLength: entry?.dataByteLength,
      dataByteOffset: entry?.dataByteOffset,
      dtype,
      elementCount,
      shape: shapeInfo?.shape,
      sourceByteLength: entry?.sourceByteLength,
      sourceDtype: entry?.sourceDtype,
      safetensorsShardIndex: entry?.safetensorsShardIndex,
      safetensorsShardName: entry?.safetensorsShardName,
    });
  }

  normalizeModelResources(value) {
    const resources = normalizeLlamaModelResourceEntries(value).map((entry, index) => this.normalizeModelResourceEntry(entry, index));
    const seen = new Set();
    for (const entry of resources) {
      if (seen.has(entry.role)) {
        throw new Error(`duplicate LLaMA model resource role: ${entry.role}`);
      }
      seen.add(entry.role);
    }
    return resources;
  }

  createModelResources(specs, options = {}) {
    const entries = normalizeLlamaModelResourceEntries(specs);
    const roles = entries.map((entry, index) => normalizeLlamaModelResourceRole(entry?.role ?? entry?.name ?? entry?.tensor ?? entry?.kind, index));
    const seenRoles = new Set();
    for (const role of roles) {
      if (seenRoles.has(role)) {
        throw new Error(`duplicate LLaMA model resource role: ${role}`);
      }
      seenRoles.add(role);
    }
    return entries.map((entry, index) => {
      const role = normalizeLlamaModelResourceRole(entry?.role ?? entry?.name ?? entry?.tensor ?? entry?.kind, index);
      const dtype = normalizeLlamaModelResourceDtype(entry?.dtype ?? options.dtype ?? "f32");
      const shapeInfo = normalizeLlamaModelResourceShape(entry?.shape);
      const elementCount = explicitLlamaModelResourceElementCount(entry) ?? shapeInfo?.elementCount;
      const source = isResourceBinding(entry)
        ? entry
        : (entry?.resource ?? entry?.gpuBuffer ?? entry?.buffer ?? entry?.externalResource);
      const byteLength = llamaModelResourceByteLength({ ...entry, dtype, elementCount }, source);
      const makeResource = source ?? options.resource ?? options.externalResource;
      const resource = typeof makeResource === "function"
        ? makeResource({
          byteLength,
          dataByteLength: entry?.dataByteLength,
          dataByteOffset: entry?.dataByteOffset,
          dtype,
          elementCount,
          index,
          program: this,
          role,
          safetensorsShardIndex: entry?.safetensorsShardIndex,
          safetensorsShardName: entry?.safetensorsShardName,
          shape: shapeInfo?.shape,
          sourceByteLength: entry?.sourceByteLength,
          sourceDtype: entry?.sourceDtype,
        })
        : (makeResource ?? this.createRawBuffer(`model.${role}`, byteLength, {
          ...options,
          ...entry,
          access: entry?.access ?? options.access ?? "read",
          label: entry?.label ?? options.label ?? `zgml.wasm.webgpu.llama.model.${role}`,
        }));
      return this.normalizeModelResourceEntry({
        ...entry,
        access: entry?.access ?? options.access ?? "read",
        byteLength,
        dataByteLength: entry?.dataByteLength,
        dataByteOffset: entry?.dataByteOffset,
        dtype,
        elementCount,
        resource,
        role,
        safetensorsShardIndex: entry?.safetensorsShardIndex,
        safetensorsShardName: entry?.safetensorsShardName,
        shape: shapeInfo?.shape,
        sourceByteLength: entry?.sourceByteLength,
        sourceDtype: entry?.sourceDtype,
      }, index);
    });
  }

  createModelResourcesFromSafetensors(source, options = {}) {
    const specs = safetensorsModelResourceSpecs(source, {
      materializeFloatDtypes: true,
      ...options,
    });
    const sections = safetensorsDataSections(source);
    const resources = this.createModelResources(executableLlamaSafetensorsModelResourceSpecs(specs, options), options);
    for (const resource of resources) {
      if (!Number.isSafeInteger(resource.dataByteOffset) || !Number.isSafeInteger(resource.dataByteLength)) {
        throw new Error(`LLaMA model resource ${resource.role} is missing safetensors data offsets`);
      }
      const shardIndex = resource.safetensorsShardIndex ?? 0;
      const section = sections[shardIndex];
      if (!section) {
        throw new Error(`LLaMA model resource ${resource.role} references missing safetensors shard ${shardIndex}`);
      }
      const sourceDtype = normalizeLlamaModelResourceDtype(resource.sourceDtype ?? resource.dtype);
      const sourceByteLength = resource.sourceByteLength ?? resource.dataByteLength;
      if (resource.dataByteLength !== sourceByteLength) {
        throw new Error(`LLaMA model resource ${resource.role} source byteLength ${sourceByteLength} does not match safetensors data length ${resource.dataByteLength}`);
      }
      const start = section.dataStart + resource.dataByteOffset;
      const end = start + sourceByteLength;
      if (end > section.bytes.byteLength) {
        throw new Error(`LLaMA model resource ${resource.role} safetensors data range is out of bounds`);
      }
      if (normalizeLlamaModelResourceDtype(resource.dtype) === "f32" && sourceDtype !== "f32") {
        this.device.writeFloat32(
          resource.descriptor,
          decodeSafetensorsFloat32Values(
            section.bytes,
            start,
            sourceByteLength,
            sourceDtype,
            resource.elementCount,
            `LLaMA model resource ${resource.role}`,
          ),
        );
      } else {
        if (sourceByteLength !== resource.byteLength) {
          throw new Error(`LLaMA model resource ${resource.role} byteLength ${resource.byteLength} does not match safetensors data length ${sourceByteLength}`);
        }
        this.device.writeBytes(resource.descriptor, section.bytes.subarray(start, end));
      }
    }
    return resources;
  }

  normalizeModelResourcesFromValue(value, options = {}) {
    if (value === undefined || value === null) return [];
    if (isSafetensorsByteSource(value) || isSafetensorsShardSourceList(value)) {
      return this.createModelResourcesFromSafetensors(value, options);
    }
    if (value && typeof value === "object") {
      const safetensors = value.safetensors ?? value.safetensorsData ?? value.shards;
      if (safetensors !== undefined && safetensors !== null) {
        return this.createModelResourcesFromSafetensors(safetensors, {
          ...value,
          ...options,
        });
      }
    }
    return this.normalizeModelResources(value);
  }

  modelResourceRequirementsFromValue(value, options = {}) {
    options = options ?? {};
    if (value === undefined || value === null) return null;
    if (isSafetensorsByteSource(value) || isSafetensorsShardSourceList(value)) {
      const specs = safetensorsModelResourceSpecs(value, {
        materializeFloatDtypes: true,
        ...options,
      });
      return normalizeLlamaModelResourceRequirements(executableLlamaSafetensorsModelResourceSpecs(specs, options));
    }
    if (value && typeof value === "object") {
      const safetensors = value.safetensors ?? value.safetensorsData ?? value.shards;
      if (safetensors !== undefined && safetensors !== null) {
        const specs = safetensorsModelResourceSpecs(safetensors, {
          materializeFloatDtypes: true,
          ...value,
          ...options,
        });
        return normalizeLlamaModelResourceRequirements(executableLlamaSafetensorsModelResourceSpecs(specs, {
          ...value,
          ...options,
        }));
      }
    }
    if (options.resourceMapManifest === true || options.allowResourceMapManifest === true) {
      return normalizeLlamaModelResourceRequirementsFromResourceMap(value, options);
    }
    return null;
  }

  createOutputBuffer(options = {}) {
    const byteLength = options.byteLength ?? this.outputByteLength();
    const makeResource = options.resource ?? options.externalResource;
    const resource = typeof makeResource === "function"
      ? makeResource({ kind: "output", byteLength, program: this })
      : (makeResource ?? options.buffer ?? this.createRawBuffer("output", byteLength, { ...options, access: "write" }));
    if (isResourceBinding(resource)) {
      return validateStorageResourceBinding(this.device, resourceBinding(resource, "LLaMA output"), "LLaMA output", byteLength, "write");
    }
    return this.wrapBuffer("output", resource, {
      ...options,
      access: options.access ?? "write",
      byteLength,
    });
  }

  createActivationBuffer(options = {}) {
    const byteLength = options.byteLength ?? this.activationByteLength(options);
    const makeResource = options.resource ?? options.externalResource;
    const resource = typeof makeResource === "function"
      ? makeResource({ kind: "activation", byteLength, program: this })
      : (makeResource ?? options.buffer ?? this.createRawBuffer("activation", byteLength, { ...options, access: "readwrite" }));
    if (isResourceBinding(resource)) {
      return validateStorageResourceBinding(
        this.device,
        resourceBinding(resource, "LLaMA activation"),
        "LLaMA activation",
        byteLength,
        "readwrite",
      );
    }
    return this.wrapBuffer("activation", resource, {
      ...options,
      access: options.access ?? "readwrite",
      byteLength,
    });
  }

  createKvCache(options = {}) {
    const requirements = this.requireKvRequirements();
    const makeResource = options.resource ?? options.externalResource;
    const makeBuffer = (kind, layer, byteLength) => {
      const resource = typeof makeResource === "function"
        ? makeResource({ kind, layer, byteLength, requirements, program: this })
        : this.createRawBuffer(`kv-${kind}-${layer}`, byteLength, { ...options, access: "readwrite" });
      if (isResourceBinding(resource)) {
        return validateStorageResourceBinding(
          this.device,
          resourceBinding(resource, `LLaMA ${kind.toUpperCase()} cache ${layer}`),
          `LLaMA ${kind.toUpperCase()} cache ${layer}`,
          byteLength,
          "readwrite",
        );
      }
      return this.wrapBuffer(`kv-${kind}`, resource, {
        ...options,
        access: options.access ?? "readwrite",
        byteLength,
      });
    };
    const kvCache = [];
    for (let layer = 0; layer < requirements.layers; layer += 1) {
      kvCache.push({
        k: makeBuffer("k", layer, requirements.kBufferByteLength),
        v: makeBuffer("v", layer, requirements.vBufferByteLength),
      });
    }
    return kvCache;
  }

  normalizeOutput(value, options = {}) {
    const outputLen = options.outputLen ?? options.outputLength ?? this.vocabSize;
    if (!Number.isSafeInteger(outputLen) || outputLen < this.vocabSize) {
      throw new Error(`invalid LLaMA output length: ${outputLen}`);
    }
    const byteLength = outputLen * Float32Array.BYTES_PER_ELEMENT;
    if (value === undefined || value === null) return this.createOutputBuffer({ byteLength });
    if (isResourceBinding(value)) {
      return validateStorageResourceBinding(this.device, resourceBinding(value, "LLaMA output"), "LLaMA output", byteLength, "write");
    }
    return this.wrapBuffer("output", value, {
      access: "write",
      byteLength,
    });
  }

  normalizeActivation(value, options = {}) {
    if (value === undefined || value === null) return null;
    const descriptorByteLength = value?.descriptor?.byteLength;
    const byteLength = value.byteLength ?? value.byteLen ?? value.size ?? descriptorByteLength;
    const label = options.label ?? "LLaMA activation";
    const access = options.access ?? "readwrite";
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0) {
      throw new Error(`invalid ${label} byteLength: ${byteLength}`);
    }
    assertFloat32ByteLength(byteLength, label);
    if (isResourceBinding(value)) {
      return validateStorageResourceBinding(
        this.device,
        resourceBinding(value, label),
        label,
        byteLength,
        access,
      );
    }
    return this.wrapBuffer(options.kind ?? "activation", value, {
      access,
      byteLength,
    });
  }

  normalizeKvEntry(value, kind, layer) {
    const requirements = this.requireKvRequirements();
    const byteLength = kind === "k" ? requirements.kBufferByteLength : requirements.vBufferByteLength;
    if (isResourceBinding(value)) {
      return validateStorageResourceBinding(
        this.device,
        resourceBinding(value, `LLaMA ${kind.toUpperCase()} cache ${layer}`),
        `LLaMA ${kind.toUpperCase()} cache ${layer}`,
        byteLength,
        "readwrite",
      );
    }
    return this.wrapBuffer(`kv-${kind}`, value, {
      access: "readwrite",
      byteLength,
    });
  }

  normalizeKvCache(value) {
    const entries = value ?? this.createKvCache();
    if (!Array.isArray(entries) || entries.length === 0) {
      throw new Error("LLaMA resource binding needs at least one K/V cache pair");
    }
    return entries.map((entry, i) => ({
      k: this.normalizeKvEntry(entry.k, "k", i),
      v: this.normalizeKvEntry(entry.v, "v", i),
    }));
  }

  prepareSessionResources(resources = {}, options = {}) {
    const output = this.normalizeOutput(resources.output, {
      outputLen: resources.outputLen ?? options.outputLen ?? options.outputLength,
    });
    const kvCache = this.normalizeKvCache(resources.kvCache ?? resources.kv);
    const activation = this.normalizeActivation(resources.activationOutput ?? resources.hiddenOutput ?? resources.activation ?? resources.hidden ?? resources.hiddenState, {
      access: "readwrite",
      kind: "activation",
      label: "LLaMA activation",
    });
    const activationInput = this.normalizeActivation(resources.activationInput ?? resources.hiddenInput, {
      access: "read",
      kind: "activation-input",
      label: "LLaMA activation input",
    });
    const modelResourceSource = options.modelResources ?? options.weights ?? resources.modelResources ?? resources.weights ?? resources.safetensors ?? resources.safetensorsData ?? this.defaultModelResourceSource;
    const modelResourceOptions = options.modelResourceOptions ?? resources.modelResourceOptions ?? this.defaultModelResourceOptions;
    const modelResources = this.normalizeModelResourcesFromValue(modelResourceSource, modelResourceOptions);
    const requiredModelResources = this.requiredModelResources.length === 0
      ? (this.modelResourceRequirementsFromValue(modelResourceSource, modelResourceOptions) ?? this.requiredModelResources)
      : this.requiredModelResources;
    validateLlamaModelResourceRequirements(modelResources, requiredModelResources, {
      allowExtra: this.allowExtraModelResources,
    });
    const modelManifest = modelResources.length === 0 ? null : describeLlamaModelResourceManifest(modelResources);
    const modelHandle = options.model ?? resources.model ?? this.defaultModelHandle;
    assertU32(modelHandle, "LLaMA model handle");
    const resourceDescriptors = [
      { role: "llama.output", descriptor: output.descriptor },
      ...kvCache.flatMap((entry, layer) => [
        { role: `llama.k.${layer}`, descriptor: entry.k.descriptor },
        { role: `llama.v.${layer}`, descriptor: entry.v.descriptor },
      ]),
    ];
    if (activation !== null) {
      resourceDescriptors.push({ role: "llama.activation", descriptor: activation.descriptor });
    }
    if (activationInput !== null) {
      resourceDescriptors.push({ role: "llama.activation.input", descriptor: activationInput.descriptor });
    }
    return {
      activation,
      activationInput,
      kvCache,
      modelHandle,
      modelManifest,
      modelResources,
      output,
      resourceDescriptors,
    };
  }

	  validateNativeExecutableRequirements(options = {}, modelHandle = 0) {
	    if (
	      options.validateNativeRequirements === false ||
	      options.validateNativeExecutableRequirements === false
    ) return null;
    const nativeProgramRequirements =
      options.nativeProgramRequirements ??
      options.nativeRequirements ??
      options.nativeExecutableRequirements ??
      this.sessionBindingBridge.programRequirements(this.programHandle);
    assertLlamaNativeProgramRequirementsMatch(this, nativeProgramRequirements, "native-backed LLaMA");
    const nativeKvRequirements = (
      options.validateNativeKvRequirements === false ||
      options.validateNativeKvCacheRequirements === false
    )
      ? null
      : (
        options.nativeKvRequirements ??
        options.nativeKvCacheRequirements ??
        options.nativeLlamaKvRequirements ??
        this.sessionBindingBridge.llamaKvCacheRequirements(this.programHandle)
      );
    if (nativeKvRequirements !== null) {
      assertLlamaKvRequirementsMatch(this.requireKvRequirements(), nativeKvRequirements, "native-backed LLaMA");
    }
    const nativeModelCompatibility = (
      modelHandle === 0 ||
      options.validateNativeModelCompatibility === false ||
      options.validateModelCompatibility === false
    )
      ? null
      : (
        options.nativeModelCompatibility ??
        options.modelCompatibility ??
        this.sessionBindingBridge.programModelCompatibility(this.programHandle, modelHandle)
      );
    if (nativeModelCompatibility !== null) {
      assertLlamaNativeModelCompatibility(nativeModelCompatibility, "native-backed LLaMA");
    }
    return {
      modelCompatibility: nativeModelCompatibility,
      programRequirements: nativeProgramRequirements,
      kvRequirements: nativeKvRequirements,
    };
  }

  createResourceSession(sessionHandle, prepared, options = {}) {
    const modelResources = prepared.modelResources;
    const session = new WasmWebGpuLlamaResourceSession({
      sessionHandle,
      bindings: {
        output: prepared.output.descriptor,
        activation: prepared.activation?.descriptor ?? null,
        activationInput: prepared.activationInput?.descriptor ?? null,
        kvCache: prepared.kvCache.map((entry) => ({
          k: entry.k.descriptor,
          v: entry.v.descriptor,
        })),
        modelResources: modelResources.map((entry) => entry.descriptor),
      },
      device: this.device,
      table: this.device.resourceTable(prepared.resourceDescriptors),
      modelTable: modelResources.length === 0
        ? null
        : this.device.resourceTable(modelResources.map((entry) => ({ role: entry.role, descriptor: entry.descriptor }))),
      modelManifest: prepared.modelManifest,
      modelResources: modelResources.map((entry) => ({
        byteLength: entry.byteLength,
        dataByteLength: entry.dataByteLength,
        dataByteOffset: entry.dataByteOffset,
        dtype: entry.dtype,
        elementCount: entry.elementCount,
        role: entry.role,
        descriptor: entry.descriptor,
        shape: entry.shape,
        sourceByteLength: entry.sourceByteLength,
        sourceDtype: entry.sourceDtype,
        safetensorsShardIndex: entry.safetensorsShardIndex,
        safetensorsShardName: entry.safetensorsShardName,
      })),
      programInspection: this.inspect(),
      diagnosticHistoryLimit:
        options.diagnosticHistoryLimit ??
        options.historyLimit ??
        options.maxDiagnosticHistoryEntries ??
        options.maxHistoryEntries,
      modelHandle: prepared.modelHandle,
      modelKind: this.modelKind,
      nativeSession: options.nativeSession ?? true,
      requireGpuDispatch: resolveLlamaRequireGpuDispatch(options, this.requireGpuDispatch),
	      freeNativeSession: this.freeSession,
	      maxTokenWindow: options.maxTokenWindow ?? this.contextLength,
	      runtime: this.runtime,
	      tokenExecutor: this.tokenExecutor,
	      vocabSize: this.vocabSize,
      contextLength: this.contextLength,
    });
    try {
      this.tokenExecutor?.prepareSession?.(session);
    } catch (err) {
      session.releaseLocal();
      throw err;
    }
    return session;
  }

	  bind(resources = {}, options = {}) {
	    const modelHandle = options.model ?? resources.model ?? this.defaultModelHandle;
	    assertU32(modelHandle, "LLaMA model handle");
	    const nativeRequirements = this.validateNativeExecutableRequirements(options, modelHandle);
	    const outputLen =
	      resources.outputLen ??
	      options.outputLen ??
      options.outputLength ??
      nativeRequirements?.programRequirements?.outputLen ??
      this.vocabSize;
	    if (nativeRequirements !== null) {
	      assertLlamaNativeOutputLength(outputLen, nativeRequirements.programRequirements, "native-backed LLaMA");
	    }
	    const bindOptions = {
	      ...options,
	      maxTokenWindow: nativeRequirements?.programRequirements?.maxTokenWindow,
	      model: modelHandle,
	      outputLen,
	    };
	    const prepared = this.prepareSessionResources(resources, bindOptions);
    const bindDesc = this.sessionBindingBridge.llamaBufferBindDesc({
      output: prepared.output.buffer,
      outputLen,
      kvCache: prepared.kvCache.map((entry) => ({
        k: entry.k.buffer,
        v: entry.v.buffer,
      })),
    });
    const sessionHandle = this.sessionBindingBridge.bindLlamaBufferSession(this.programHandle, bindDesc, {
      model: prepared.modelHandle,
    });
    let session;
    try {
      session = this.createResourceSession(sessionHandle, prepared, { ...options, nativeSession: true });
    } catch (err) {
      const exportsRef = readOption(this.sessionBindingBridge.exports);
      if (exportsRef && typeof exportsRef.zgml_session_free === "function") {
        exportsRef.zgml_session_free(sessionHandle);
      }
      throw err;
    }
    if (this.runtime) {
      try {
        this.runtime.register(session);
      } catch (err) {
        session.releaseLocal();
        const exportsRef = readOption(this.sessionBindingBridge.exports);
        if (exportsRef && typeof exportsRef.zgml_session_free === "function") {
          exportsRef.zgml_session_free(sessionHandle);
        }
        throw err;
      }
    }
    return session;
  }

  bindHostResources(resources = {}, options = {}) {
    if (!(this.runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("WasmWebGpuLlamaResourceProgram needs a runtime to bind host-only resource Sessions");
    }
    const prepared = this.prepareSessionResources(resources, options);
    const sessionHandle = options.sessionHandle ?? this.runtime.allocateHostSessionHandle();
    const session = this.createResourceSession(sessionHandle, prepared, { ...options, nativeSession: false });
    try {
      this.runtime.register(session);
    } catch (err) {
      session.releaseLocal();
      throw err;
    }
    return session;
  }

  unregister(sessionOrHandle) {
    if (this.runtime) this.runtime.unregister(sessionOrHandle);
  }
}

export class WasmWebGpuSessionRuntime {
  constructor(options = {}) {
    this.view = options.view;
    this.status = {
      ok: options.ok ?? 0,
      invalidArgument: options.invalidArgument ?? 1,
      outOfMemory: options.outOfMemory ?? 2,
      shapeMismatch: options.shapeMismatch ?? 3,
      compileFailed: options.compileFailed ?? 4,
      unsupported: options.unsupported ?? 5,
      unclaimed: options.unclaimed ?? -1,
    };
    this.sessions = new Map();
    this.claimedSessionHandles = new Set();
    this.nextHostSessionHandle = options.nextHostSessionHandle ?? 0x70000000;
  }

  allocateHostSessionHandle() {
    let handle = this.nextHostSessionHandle;
    for (let attempts = 0; attempts < 0x100000; attempts += 1) {
      if (handle > 0xffffffff) handle = 0x70000000;
      assertHandle(handle);
      if (!this.sessions.has(handle) && !this.claimedSessionHandles.has(handle)) {
        this.nextHostSessionHandle = handle + 1;
        return handle;
      }
      handle += 1;
    }
    throw new Error("could not allocate a host-only WebGPU Session handle");
  }

  register(session) {
    if (!(session instanceof WasmWebGpuTinyLinearSession) && !(session instanceof WasmWebGpuLlamaResourceSession)) {
      throw new Error("WasmWebGpuSessionRuntime can only register WebGPU host Sessions");
    }
    const handle = session.handle;
    if (this.sessions.has(handle) || this.claimedSessionHandles.has(handle)) {
      throw new Error(`WebGPU host Session handle is already registered: ${handle}`);
    }
    this.sessions.set(handle, session);
    this.claimedSessionHandles.add(handle);
    return session;
  }

  unregister(sessionOrHandle) {
    const handle = sessionOrHandle instanceof WasmWebGpuTinyLinearSession || sessionOrHandle instanceof WasmWebGpuLlamaResourceSession
      ? sessionOrHandle.sessionHandle
      : sessionOrHandle;
    if (handle) {
      this.sessions.delete(handle);
      this.claimedSessionHandles.delete(handle);
    }
  }

  hasClaimed(sessionHandle) {
    return this.claimedSessionHandles.has(sessionHandle);
  }

  lookup(sessionHandle) {
    const session = this.sessions.get(sessionHandle);
    if (!session || session.freed) return null;
    return session;
  }

  memoryView() {
    const view = typeof this.view === "function" ? this.view() : this.view;
    if (!(view instanceof DataView)) {
      throw new Error("WasmWebGpuSessionRuntime needs a DataView or view callback");
    }
    return view;
  }

  writeStepResult(resultPtr, outputLen) {
    if (resultPtr === undefined || resultPtr === null || resultPtr === 0) return;
    assertU32(resultPtr, "step result pointer");
    assertU32(outputLen, "step output length");
    this.memoryView().setUint32(resultPtr, outputLen, true);
  }

  stepSync(sessionHandle, descPtr = 0, resultPtr = 0) {
    const session = this.lookup(sessionHandle);
    if (!session) return this.status.unclaimed;
    if (typeof session.stepSync !== "function") return this.status.unsupported;
    if (descPtr !== 0) return this.status.unsupported;
    session.stepSync({ outputLength: session.program.rows });
    this.writeStepResult(resultPtr, session.program.rows);
    return this.status.ok;
  }

  async step(sessionHandle, descPtr = 0, resultPtr = 0) {
    const session = this.lookup(sessionHandle);
    if (!session) return this.status.unclaimed;
    if (typeof session.step !== "function") return this.status.unsupported;
    if (descPtr !== 0) return this.status.unsupported;
    await session.step({ outputLength: session.program.rows });
    this.writeStepResult(resultPtr, session.program.rows);
    return this.status.ok;
  }

  stepNoOutputSync(sessionHandle, descPtr = 0, resultPtr = 0) {
    const session = this.lookup(sessionHandle);
    if (!session) return this.status.unclaimed;
    if (typeof session.stepSync !== "function") return this.status.unsupported;
    if (descPtr !== 0) return this.status.unsupported;
    session.stepSync({ outputLength: 0 });
    this.writeStepResult(resultPtr, 0);
    return this.status.ok;
  }

  async stepNoOutput(sessionHandle, descPtr = 0, resultPtr = 0) {
    const session = this.lookup(sessionHandle);
    if (!session) return this.status.unclaimed;
    if (typeof session.step !== "function") return this.status.unsupported;
    if (descPtr !== 0) return this.status.unsupported;
    await session.step({ outputLength: 0 });
    this.writeStepResult(resultPtr, 0);
    return this.status.ok;
  }

  executeTokensSync(sessionHandle, descPtr = 0, resultPtr = 0) {
    const session = this.lookup(sessionHandle);
    if (!session) return this.status.unclaimed;
    if (typeof session.executeTokensSync !== "function") return this.status.unsupported;
    return session.executeTokensSync(descPtr, resultPtr, this);
  }

  positionSync(sessionHandle, outPositionPtr = 0) {
    const session = this.lookup(sessionHandle);
    if (!session) return this.status.unclaimed;
    if (!Number.isSafeInteger(outPositionPtr) || outPositionPtr === 0) return this.status.invalidArgument;
    const position = Number.isSafeInteger(session.position) ? session.position : 0;
    assertU32(position, "host Session position");
    this.memoryView().setUint32(outPositionPtr, position, true);
    return this.status.ok;
  }

  runtimeProfileSync(sessionHandle, outProfilePtr = 0) {
    const session = this.lookup(sessionHandle);
    if (!session) return this.status.unclaimed;
    if (session instanceof WasmWebGpuLlamaResourceSession && session.tokenExecutor === null) return this.status.unclaimed;
    if (typeof session.writeRuntimeProfile !== "function") return this.status.unclaimed;
    if (!Number.isSafeInteger(outProfilePtr) || outProfilePtr === 0) return this.status.invalidArgument;
    session.writeRuntimeProfile(this.memoryView(), outProfilePtr);
    return this.status.ok;
  }

  inspectSync(sessionHandle, outInspectionPtr = 0) {
    const session = this.lookup(sessionHandle);
    if (!session) return this.status.unclaimed;
    if (typeof session.writeInspection !== "function") return this.status.unclaimed;
    if (!Number.isSafeInteger(outInspectionPtr) || outInspectionPtr === 0) return this.status.invalidArgument;
    const view = this.memoryView();
    if (outInspectionPtr < 0 || outInspectionPtr + 80 > view.byteLength) return this.status.invalidArgument;
    session.writeInspection(view, outInspectionPtr);
    return this.status.ok;
  }

  resetRuntimeProfileSync(sessionHandle) {
    const session = this.lookup(sessionHandle);
    if (!session) return this.status.unclaimed;
    if (session instanceof WasmWebGpuLlamaResourceSession && session.tokenExecutor === null) return this.status.unclaimed;
    if (typeof session.writeRuntimeProfile !== "function" || typeof session.resetRuntimeProfile !== "function") {
      return this.status.unclaimed;
    }
    session.resetRuntimeProfile();
    return this.status.ok;
  }

  resetSync(sessionHandle) {
    const session = this.lookup(sessionHandle);
    if (!session) return this.status.unclaimed;
    if (typeof session.reset !== "function") return this.status.unsupported;
    session.reset();
    return this.status.ok;
  }

  releaseHostSession(sessionHandle) {
    if (!this.hasClaimed(sessionHandle)) return this.status.unclaimed;
    const session = this.sessions.get(sessionHandle);
    if (session instanceof WasmWebGpuLlamaResourceSession) {
      session.releaseLocal();
    } else if (session) {
      session.freed = true;
      session.sessionHandle = 0;
    }
    this.unregister(sessionHandle);
    return this.status.ok;
  }

  free(sessionHandle, exportsRef) {
    if (!this.hasClaimed(sessionHandle)) return exportsRef.zgml_session_free(sessionHandle);
    const session = this.lookup(sessionHandle);
    const nativeSession = session === null ? true : session.nativeSession !== false;
    this.releaseHostSession(sessionHandle);
    if (!nativeSession) return this.status.ok;
    return exportsRef.zgml_session_free(sessionHandle);
  }

  imports() {
    const runtime = this;
    return {
      zgml_wasm_host_session_step(sessionHandle, descPtr = 0, resultPtr = 0) {
        return runtime.stepSync(sessionHandle, descPtr, resultPtr);
      },
      zgml_wasm_host_session_step_no_output(sessionHandle, descPtr = 0, resultPtr = 0) {
        return runtime.stepNoOutputSync(sessionHandle, descPtr, resultPtr);
      },
      zgml_wasm_host_session_execute_tokens(sessionHandle, descPtr = 0, resultPtr = 0) {
        return runtime.executeTokensSync(sessionHandle, descPtr, resultPtr);
      },
      zgml_wasm_host_session_position(sessionHandle, outPositionPtr = 0) {
        return runtime.positionSync(sessionHandle, outPositionPtr);
      },
      zgml_wasm_host_session_runtime_profile(sessionHandle, outProfilePtr = 0) {
        return runtime.runtimeProfileSync(sessionHandle, outProfilePtr);
      },
      zgml_wasm_host_session_reset(sessionHandle) {
        return runtime.resetSync(sessionHandle);
      },
      zgml_wasm_host_session_reset_runtime_profile(sessionHandle) {
        return runtime.resetRuntimeProfileSync(sessionHandle);
      },
      zgml_wasm_host_session_free(sessionHandle) {
        return runtime.releaseHostSession(sessionHandle);
      },
    };
  }

  wrapExports(exportsRef) {
    if (
      !exportsRef ||
      typeof exportsRef.zgml_session_step !== "function" ||
      typeof exportsRef.zgml_session_step_no_output !== "function" ||
      typeof exportsRef.zgml_session_execute_tokens !== "function" ||
      typeof exportsRef.zgml_session_position !== "function" ||
      typeof exportsRef.zgml_session_inspect !== "function" ||
      typeof exportsRef.zgml_session_runtime_profile !== "function" ||
      typeof exportsRef.zgml_session_reset !== "function" ||
      typeof exportsRef.zgml_session_reset_runtime_profile !== "function" ||
      typeof exportsRef.zgml_session_free !== "function"
    ) {
      throw new Error("WasmWebGpuSessionRuntime needs zgml session lifecycle exports");
    }
    const runtime = this;
    return new Proxy(Object.create(null), {
      get(_target, property) {
        if (property === "zgml_session_step") {
          return function zgmlSessionStep(sessionHandle, descPtr = 0, resultPtr = 0) {
            if (runtime.hasClaimed(sessionHandle)) return runtime.stepSync(sessionHandle, descPtr, resultPtr);
            return exportsRef.zgml_session_step(sessionHandle, descPtr, resultPtr);
          };
        }
        if (property === "zgml_session_step_no_output") {
          return function zgmlSessionStepNoOutput(sessionHandle, descPtr = 0, resultPtr = 0) {
            if (runtime.hasClaimed(sessionHandle)) return runtime.stepNoOutputSync(sessionHandle, descPtr, resultPtr);
            return exportsRef.zgml_session_step_no_output(sessionHandle, descPtr, resultPtr);
          };
        }
        if (property === "zgml_session_execute_tokens") {
          return function zgmlSessionExecuteTokens(sessionHandle, descPtr = 0, resultPtr = 0) {
            if (runtime.hasClaimed(sessionHandle)) return runtime.executeTokensSync(sessionHandle, descPtr, resultPtr);
            return exportsRef.zgml_session_execute_tokens(sessionHandle, descPtr, resultPtr);
          };
        }
        if (property === "zgml_session_position") {
          return function zgmlSessionPosition(sessionHandle, outPositionPtr = 0) {
            if (runtime.hasClaimed(sessionHandle)) return runtime.positionSync(sessionHandle, outPositionPtr);
            return exportsRef.zgml_session_position(sessionHandle, outPositionPtr);
          };
        }
        if (property === "zgml_session_runtime_profile") {
          return function zgmlSessionRuntimeProfile(sessionHandle, outProfilePtr = 0) {
            if (runtime.hasClaimed(sessionHandle)) {
              const status = runtime.runtimeProfileSync(sessionHandle, outProfilePtr);
              if (status !== runtime.status.unclaimed) return status;
            }
            return exportsRef.zgml_session_runtime_profile(sessionHandle, outProfilePtr);
          };
        }
        if (property === "zgml_session_inspect") {
          return function zgmlSessionInspect(sessionHandle, outInspectionPtr = 0) {
            if (runtime.hasClaimed(sessionHandle)) {
              const status = runtime.inspectSync(sessionHandle, outInspectionPtr);
              if (status !== runtime.status.unclaimed) return status;
            }
            return exportsRef.zgml_session_inspect(sessionHandle, outInspectionPtr);
          };
        }
        if (property === "zgml_session_reset") {
          return function zgmlSessionReset(sessionHandle) {
            if (runtime.hasClaimed(sessionHandle)) {
              const status = runtime.resetSync(sessionHandle);
              if (status !== runtime.status.unclaimed) return status;
            }
            return exportsRef.zgml_session_reset(sessionHandle);
          };
        }
        if (property === "zgml_session_reset_runtime_profile") {
          return function zgmlSessionResetRuntimeProfile(sessionHandle) {
            if (runtime.hasClaimed(sessionHandle)) {
              const status = runtime.resetRuntimeProfileSync(sessionHandle);
              if (status !== runtime.status.unclaimed) return status;
            }
            return exportsRef.zgml_session_reset_runtime_profile(sessionHandle);
          };
        }
        if (property === "zgml_session_free") {
          return function zgmlSessionFree(sessionHandle) {
            return runtime.free(sessionHandle, exportsRef);
          };
        }
        return exportsRef[property];
      },
    });
  }
}

function decodeTokenExecuteDesc(view, descPtr) {
  assertU32(descPtr, "token execute descriptor pointer");
  const tokensPtr = view.getUint32(descPtr, true);
  const tokensLen = view.getUint32(descPtr + 4, true);
  const outputPolicy = view.getUint32(descPtr + 8, true);
  const outputPtr = view.getUint32(descPtr + 16, true);
  const outputLen = view.getUint32(descPtr + 20, true);
  const tokens = [];
  if (tokensLen !== 0) {
    assertU32(tokensPtr, "token execute tokens pointer");
    for (let i = 0; i < tokensLen; i += 1) {
      tokens.push(view.getUint32(tokensPtr + i * 4, true));
    }
  }
  return {
    descPtr,
    tokensPtr,
    tokensLen,
    tokens,
    outputPolicy,
    outputPtr,
    outputLen,
  };
}

function validateLlamaTokenExecution(execution, session, runtime) {
  if (execution.tokensLen === 0 || execution.tokensLen > session.maxTokenWindow) {
    return runtime.status.shapeMismatch;
  }
  if (session.position + execution.tokensLen > session.contextLength) {
    return runtime.status.shapeMismatch;
  }
  const tokens = execution.tokens;
  if (tokens === null || tokens === undefined || !Number.isSafeInteger(tokens.length) || tokens.length < execution.tokensLen) {
    return runtime.status.shapeMismatch;
  }
  for (let i = 0; i < execution.tokensLen; i += 1) {
    const token = tokens[i];
    if (token >= session.vocabSize) return runtime.status.shapeMismatch;
  }
  switch (execution.outputPolicy) {
    case 0:
      return execution.outputPtr === 0 && execution.outputLen === 0
        ? runtime.status.ok
        : runtime.status.shapeMismatch;
    case 1:
      if (execution.outputPtr === 0 && execution.outputLen === 0) return runtime.status.ok;
      return execution.outputPtr !== 0 && execution.outputLen >= session.vocabSize
        ? runtime.status.ok
        : runtime.status.shapeMismatch;
    default:
      return runtime.status.invalidArgument;
  }
}

function llamaOutputKind(execution) {
  if (execution.outputPolicy === 0) return "none";
  return execution.outputPtr === 0 && execution.outputLen === 0 ? "bound-logits" : "caller-logits";
}

function writeLlamaStepParams(target, session, execution) {
  const startPosition = session.position;
  const tokenCount = execution.tokensLen;
  const endPosition = startPosition + tokenCount;
  target.kind = "llama-token-window";
  target.startPosition = startPosition;
  target.endPosition = endPosition;
  target.tokenCount = tokenCount;
  target.contextLength = session.contextLength;
  target.outputPolicy = execution.outputPolicy;
  target.outputKind = llamaOutputKind(execution);
  target.logitsLength = execution.outputPolicy === 1 ? session.vocabSize : 0;
  target.requestedOutputLength = execution.outputLen;
  return target;
}

function createLlamaStepParams(session, execution) {
  return Object.freeze(writeLlamaStepParams({}, session, execution));
}

export function llamaKvCacheWindow(session, stepParams) {
  const kvCache = session.bindings?.kvCache ?? [];
  const contextLength = stepParams.contextLength;
  if (!Number.isSafeInteger(contextLength) || contextLength <= 0) {
    throw new Error(`invalid LLaMA StepParams context length: ${contextLength}`);
  }
  return kvCache.flatMap((entry, layer) => {
    return ["k", "v"].map((kind) => {
      const descriptor = entry[kind];
      const elementCount = Math.floor(descriptor.byteLength / Float32Array.BYTES_PER_ELEMENT);
      if (elementCount % contextLength !== 0) {
        throw new Error(`LLaMA ${kind.toUpperCase()} cache ${layer} element count is not divisible by context length`);
      }
      const stride = elementCount / contextLength;
      const elementOffset = stepParams.startPosition * stride;
      const elementLength = stepParams.tokenCount * stride;
      return {
        byteOffset: elementOffset * Float32Array.BYTES_PER_ELEMENT,
        byteLength: elementLength * Float32Array.BYTES_PER_ELEMENT,
        descriptor,
        elementLength,
        elementOffset,
        kind,
        layer,
        stride,
      };
    });
  });
}

function writeLlamaKvCacheWindowSlot(target, descriptor, kind, layer, contextLength, stepParams) {
  const elementCount = Math.floor(descriptor.byteLength / Float32Array.BYTES_PER_ELEMENT);
  if (elementCount % contextLength !== 0) {
    throw new Error(`LLaMA ${kind.toUpperCase()} cache ${layer} element count is not divisible by context length`);
  }
  const stride = elementCount / contextLength;
  const elementOffset = stepParams.startPosition * stride;
  const elementLength = stepParams.tokenCount * stride;
  target.byteOffset = elementOffset * Float32Array.BYTES_PER_ELEMENT;
  target.byteLength = elementLength * Float32Array.BYTES_PER_ELEMENT;
  target.descriptor = descriptor;
  target.elementLength = elementLength;
  target.elementOffset = elementOffset;
  target.kind = kind;
  target.layer = layer;
  target.stride = stride;
  return target;
}

function createLlamaKvCacheWindowSlotScratch() {
  return {
    byteLength: 0,
    byteOffset: 0,
    descriptor: null,
    elementLength: 0,
    elementOffset: 0,
    kind: "",
    layer: 0,
    stride: 0,
  };
}

function writeLlamaKvCacheWindowSlots(target, session, stepParams) {
  const kvCache = session.bindings?.kvCache ?? [];
  const contextLength = stepParams.contextLength;
  if (!Number.isSafeInteger(contextLength) || contextLength <= 0) {
    throw new Error(`invalid LLaMA StepParams context length: ${contextLength}`);
  }
  const slotCount = kvCache.length * 2;
  target.length = slotCount;
  let index = 0;
  for (let layer = 0; layer < kvCache.length; layer += 1) {
    const entry = kvCache[layer];
    for (let kindIndex = 0; kindIndex < 2; kindIndex += 1) {
      const kind = kindIndex === 0 ? "k" : "v";
      const descriptor = entry[kind];
      let slot = target[index];
      if (!slot) {
        slot = createLlamaKvCacheWindowSlotScratch();
        target[index] = slot;
      }
      writeLlamaKvCacheWindowSlot(slot, descriptor, kind, layer, contextLength, stepParams);
      index += 1;
    }
  }
  return target;
}

function prepareLlamaKvCacheSlot(session, descriptor, kind, layer, label) {
  const contextLength = session.contextLength;
  if (!Number.isSafeInteger(contextLength) || contextLength <= 0) {
    throw new Error(`invalid ${label} context length: ${contextLength}`);
  }
  if (!descriptor || typeof descriptor !== "object") {
    throw new Error(`${label} needs bound ${kind.toUpperCase()} cache resource for layer ${layer}`);
  }
  if (!Number.isSafeInteger(descriptor.byteLength) || descriptor.byteLength <= 0) {
    throw new Error(`${label} ${kind.toUpperCase()} cache ${layer} has invalid byteLength: ${descriptor.byteLength}`);
  }
  if (descriptor.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error(`${label} ${kind.toUpperCase()} cache ${layer} byteLength must be a Float32 multiple`);
  }
  const elementCount = descriptor.byteLength / Float32Array.BYTES_PER_ELEMENT;
  if (elementCount % contextLength !== 0) {
    throw new Error(`${label} ${kind.toUpperCase()} cache ${layer} element count is not divisible by context length`);
  }
  return Object.freeze({
    contextLength,
    descriptor,
    elementCount,
    kind,
    layer,
    stride: elementCount / contextLength,
  });
}

function prepareLlamaKvCacheLayer(session, layer, label) {
  const normalizedLayer = normalizeLlamaLayerIndex(layer, `${label} layer`);
  const entry = session.bindings?.kvCache?.[normalizedLayer];
  if (!entry || typeof entry !== "object") {
    throw new Error(`${label} needs bound K/V cache resources for layer ${normalizedLayer}`);
  }
  return Object.freeze({
    k: prepareLlamaKvCacheSlot(session, entry.k, "k", normalizedLayer, label),
    layer: normalizedLayer,
    v: prepareLlamaKvCacheSlot(session, entry.v, "v", normalizedLayer, label),
  });
}

function createLlamaPreparedKvCacheWindowScratch() {
  return {
    byteLength: 0,
    byteOffset: 0,
    descriptor: null,
    elementLength: 0,
    elementOffset: 0,
    kind: "",
    layer: 0,
    stride: 0,
  };
}

function createLlamaPreparedKvCacheLayerWindowScratch() {
  return {
    k: createLlamaPreparedKvCacheWindowScratch(),
    v: createLlamaPreparedKvCacheWindowScratch(),
  };
}

function writeLlamaPreparedKvCacheWindow(target, slot, stepParams) {
  if (stepParams.contextLength !== slot.contextLength) {
    throw new Error(`LLaMA ${slot.kind.toUpperCase()} cache ${slot.layer} context length changed after bind`);
  }
  const elementOffset = stepParams.startPosition * slot.stride;
  const elementLength = stepParams.tokenCount * slot.stride;
  target.byteOffset = elementOffset * Float32Array.BYTES_PER_ELEMENT;
  target.byteLength = elementLength * Float32Array.BYTES_PER_ELEMENT;
  target.descriptor = slot.descriptor;
  target.elementLength = elementLength;
  target.elementOffset = elementOffset;
  target.kind = slot.kind;
  target.layer = slot.layer;
  target.stride = slot.stride;
  return target;
}

function llamaPreparedKvCacheWindow(slot, stepParams, target = {}) {
  return writeLlamaPreparedKvCacheWindow(target, slot, stepParams);
}

function writeLlamaPreparedKvCacheLayerWindow(target, kvLayer, stepParams) {
  writeLlamaPreparedKvCacheWindow(target.k, kvLayer.k, stepParams);
  writeLlamaPreparedKvCacheWindow(target.v, kvLayer.v, stepParams);
  return target;
}

function prepareLlamaActivationSlot(session, projection, label, bindingName = "activation") {
  const descriptor = session.bindings?.[bindingName];
  if (descriptor === undefined || descriptor === null) return null;
  if (!descriptor || typeof descriptor !== "object") {
    throw new Error(`${label} activation binding must be a resource descriptor`);
  }
  if (!Number.isSafeInteger(descriptor.byteLength) || descriptor.byteLength <= 0) {
    throw new Error(`${label} activation has invalid byteLength: ${descriptor.byteLength}`);
  }
  if (descriptor.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error(`${label} activation byteLength must be a Float32 multiple`);
  }
  const elementCount = descriptor.byteLength / Float32Array.BYTES_PER_ELEMENT;
  const expectedElementCount = session.contextLength * projection.hiddenSize;
  if (!Number.isSafeInteger(expectedElementCount) || elementCount !== expectedElementCount) {
    throw new Error(`${label} activation element count mismatch: ${elementCount} != ${expectedElementCount}`);
  }
  return Object.freeze({
    contextLength: session.contextLength,
    descriptor,
    elementCount,
    hiddenSize: projection.hiddenSize,
  });
}

function sameLlamaActivationView(left, right) {
  return !!(
    left &&
    right &&
    left.placement === right.placement &&
    left.deviceHandle === right.deviceHandle &&
    left.handle === right.handle &&
    (left.byteOffset ?? 0) === (right.byteOffset ?? 0)
  );
}

function prepareLlamaBlockActivationSlots(session, projection, label) {
  const output = prepareLlamaActivationSlot(session, projection, label, "activation");
  const input = prepareLlamaActivationSlot(session, projection, label, "activationInput");
  if (output !== null && input !== null && sameLlamaActivationView(output.descriptor, input.descriptor)) {
    throw new Error(`${label} activation input and output must use separate resource views`);
  }
  return Object.freeze({ input, output });
}

function writeLlamaActivationToken(session, slot, position, values, label, elementLength = null) {
  if (slot == null) return 0;
  if (!Number.isSafeInteger(position) || position < 0 || position >= slot.contextLength) {
    throw new Error(`${label} activation position out of range: ${position}`);
  }
  const data = values instanceof Float32Array ? values : new Float32Array(values);
  const length = elementLength === null ? data.length : elementLength;
  if (!Number.isSafeInteger(length) || length !== slot.hiddenSize || length > data.length) {
    throw new Error(`${label} activation write length ${length} is incompatible with hidden size ${slot.hiddenSize}`);
  }
  session.device.writeFloat32(
    slot.descriptor,
    data,
    position * slot.hiddenSize * Float32Array.BYTES_PER_ELEMENT,
    slot.hiddenSize,
  );
  return 1;
}

function writeLlamaKvCacheSlots(session, slots, valuesForSlot) {
  if (typeof valuesForSlot !== "function") {
    throw new Error("writeLlamaKvCacheWindow needs a values callback");
  }
  for (const slot of slots) {
    const values = valuesForSlot(slot);
    const data = values instanceof Float32Array ? values : new Float32Array(values);
    if (data.length !== slot.elementLength) {
      throw new Error(`LLaMA ${slot.kind.toUpperCase()} cache ${slot.layer} write length ${data.length} != ${slot.elementLength}`);
    }
    session.device.writeFloat32(slot.descriptor, data, slot.byteOffset);
  }
  return slots.length;
}

export function writeLlamaKvCacheWindow(session, stepParams, valuesForSlot) {
  return writeLlamaKvCacheSlots(session, llamaKvCacheWindow(session, stepParams), valuesForSlot);
}

export function writeLlamaLogits(context, values, elementLength = null) {
  const { execution, runtime, session } = context;
  if (execution.outputPolicy !== 1) {
    throw new Error("writeLlamaLogits requires logits output policy");
  }
  const data = values instanceof Float32Array ? values : new Float32Array(values);
  const length = elementLength === null ? data.length : elementLength;
  if (!Number.isSafeInteger(length) || length < session.vocabSize || length > data.length) {
    throw new Error(`LLaMA logits write length ${length} is incompatible with vocab size ${session.vocabSize}`);
  }
  if (execution.outputPtr !== 0) {
    const view = runtime.memoryView();
    for (let i = 0; i < session.vocabSize; i += 1) {
      view.setFloat32(execution.outputPtr + i * Float32Array.BYTES_PER_ELEMENT, data[i], true);
    }
  } else {
    session.device.writeFloat32(session.bindings.output, data, 0, session.vocabSize);
  }
  return session.vocabSize;
}

function writeLlamaTokenWindowAdapterContext(target, context, kvCacheWindow) {
  target.bindings = context.bindings;
  target.device = context.device;
  target.execution = context.execution;
  target.kvCacheWindow = kvCacheWindow;
  target.modelBindingKind = context.modelBindingKind;
  target.modelHandle = context.modelHandle;
  target.modelManifest = context.modelManifest;
  target.modelResources = context.modelResources;
  target.modelTable = context.modelTable;
  target.resultScratch = context.resultScratch;
  target.runtime = context.runtime;
  target.session = context.session;
  target.stepParams = context.stepParams;
  target.table = context.table;
  return target;
}

export class WasmWebGpuLlamaTokenWindowExecutor {
  constructor(options = {}) {
    const opts = typeof options === "function" ? { execute: options } : options;
    this.execute = opts.execute ?? opts.executeTokensSync ?? opts.onExecute ?? null;
    if (typeof this.execute !== "function") {
      throw new Error("WasmWebGpuLlamaTokenWindowExecutor needs an execute callback");
    }
    this.currentContext = null;
    this.kvCacheWindowScratch = [];
    this.contextScratch = {
      bindings: null,
      device: null,
      execution: null,
      kvCacheWindow: this.kvCacheWindowScratch,
      modelBindingKind: null,
      modelHandle: 0,
      modelManifest: null,
      modelResources: null,
      modelTable: null,
      resultScratch: null,
      runtime: null,
      session: null,
      stepParams: null,
      table: null,
      writeKvCacheWindow: (valuesForSlot) => this.writeKvCacheWindow(valuesForSlot),
      writeLogits: (values) => this.writeLogits(values),
    };
  }

  executeTokensSync(context) {
    this.currentContext = context;
    return this.execute(writeLlamaTokenWindowAdapterContext(
      this.contextScratch,
      context,
      writeLlamaKvCacheWindowSlots(this.kvCacheWindowScratch, context.session, context.stepParams),
    ));
  }

  writeKvCacheWindow(valuesForSlot) {
    const context = this.currentContext;
    if (context === null) {
      throw new Error("WasmWebGpuLlamaTokenWindowExecutor has no active token context");
    }
    return writeLlamaKvCacheSlots(context.session, this.kvCacheWindowScratch, valuesForSlot);
  }

  writeLogits(values) {
    const context = this.currentContext;
    if (context === null) {
      throw new Error("WasmWebGpuLlamaTokenWindowExecutor has no active token context");
    }
    return writeLlamaLogits(context, values);
  }
}

function llamaTokenSum(execution) {
  let sum = 0;
  const tokens = execution.tokens;
  const length = execution.tokensLen ?? tokens.length;
  for (let i = 0; i < length; i += 1) sum += tokens[i];
  return sum;
}

export function llamaTokenWindowLogitPattern(context, length = context.session.vocabSize) {
  const base = llamaTokenSum(context.execution) * 10 + context.stepParams.startPosition;
  return Array.from({ length }, (_unused, i) => base + i);
}

export function llamaTokenWindowKvPattern(slot, stepParams, length = slot.elementLength) {
  const base = (slot.kind === "k" ? 1000 : 2000) + slot.layer * 100 + stepParams.startPosition * 10;
  return Array.from({ length }, (_unused, i) => base + i);
}

function normalizeLlamaWeightResourceRole(role) {
  const normalized = normalizeResourceTableRole(role);
  return normalized.startsWith("llama.weight.") ? normalized : `llama.weight.${normalized}`;
}

function llamaModelResourceByRole(session, role, label) {
  const normalized = normalizeLlamaWeightResourceRole(role);
  const resource = session.modelResources?.find((entry) => entry.role === normalized);
  if (!resource) {
    throw new Error(`${label} needs LLaMA model resource ${normalized}`);
  }
  return resource;
}

function optionalLlamaModelResourceByRole(session, role) {
  const normalized = normalizeLlamaWeightResourceRole(role);
  return session.modelResources?.find((entry) => entry.role === normalized) ?? null;
}

function requireF32MatrixModelResource(session, role, label) {
  const resource = llamaModelResourceByRole(session, role, label);
  if (normalizeLlamaModelResourceDtype(resource.dtype) !== "f32") {
    throw new Error(`${label} needs f32 model resource ${resource.role}`);
  }
  if (!Array.isArray(resource.shape) || resource.shape.length !== 2) {
    throw new Error(`${label} needs rank-2 model resource ${resource.role}`);
  }
  const [rows, cols] = resource.shape;
  if (!Number.isSafeInteger(rows) || rows <= 0 || !Number.isSafeInteger(cols) || cols <= 0) {
    throw new Error(`${label} has invalid model resource shape ${resource.role}`);
  }
  const expectedElements = rows * cols;
  const expectedByteLength = expectedElements * Float32Array.BYTES_PER_ELEMENT;
  if (
    resource.elementCount !== expectedElements ||
    resource.byteLength !== expectedByteLength
  ) {
    throw new Error(`${label} model resource ${resource.role} shape metadata does not match byteLength`);
  }
  return {
    ...resource,
    cols,
    rows,
  };
}

function requireF32VectorModelResource(session, role, label) {
  const resource = llamaModelResourceByRole(session, role, label);
  if (normalizeLlamaModelResourceDtype(resource.dtype) !== "f32") {
    throw new Error(`${label} needs f32 model resource ${resource.role}`);
  }
  if (!Array.isArray(resource.shape) || resource.shape.length !== 1) {
    throw new Error(`${label} needs rank-1 model resource ${resource.role}`);
  }
  const [length] = resource.shape;
  if (!Number.isSafeInteger(length) || length <= 0) {
    throw new Error(`${label} has invalid model resource shape ${resource.role}`);
  }
  if (
    resource.elementCount !== length ||
    resource.byteLength !== length * Float32Array.BYTES_PER_ELEMENT
  ) {
    throw new Error(`${label} model resource ${resource.role} shape metadata does not match byteLength`);
  }
  return {
    ...resource,
    length,
  };
}

function optionalF32VectorModelResource(session, role, label) {
  const resource = optionalLlamaModelResourceByRole(session, role);
  if (resource === null) return null;
  if (normalizeLlamaModelResourceDtype(resource.dtype) !== "f32") {
    throw new Error(`${label} needs f32 model resource ${resource.role}`);
  }
  if (!Array.isArray(resource.shape) || resource.shape.length !== 1) {
    throw new Error(`${label} needs rank-1 model resource ${resource.role}`);
  }
  const [length] = resource.shape;
  if (!Number.isSafeInteger(length) || length <= 0) {
    throw new Error(`${label} has invalid model resource shape ${resource.role}`);
  }
  if (
    resource.elementCount !== length ||
    resource.byteLength !== length * Float32Array.BYTES_PER_ELEMENT
  ) {
    throw new Error(`${label} model resource ${resource.role} shape metadata does not match byteLength`);
  }
  return {
    ...resource,
    length,
  };
}

function prepareLlamaProjectionSession(cache, session, create) {
  let projection = cache.get(session);
  if (!projection) {
    projection = create(session);
    cache.set(session, projection);
  }
  return projection;
}

function sameProjectionGpuBindGroupDeps(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false;
  for (let i = 0; i < left.length; i += 1) {
    if (left[i] !== right[i]) return false;
  }
  return true;
}

function projectionGpuBindGroup(projection, key, device, pipeline, deps, create) {
  const gpuDevice = device.device;
  const bindGroups = projection.gpuBindGroups ?? new Map();
  projection.gpuBindGroups = bindGroups;
  const cached = bindGroups.get(key);
  if (
    cached &&
    cached.device === gpuDevice &&
    cached.pipeline === pipeline &&
    sameProjectionGpuBindGroupDeps(cached.deps, deps)
  ) {
    return cached.bindGroup;
  }
  const bindGroup = create(gpuDevice);
  bindGroups.set(key, {
    bindGroup,
    deps: deps.slice(),
    device: gpuDevice,
    pipeline,
  });
  return bindGroup;
}

function describeLlamaProjectionModelPack(session, projection, fields, offsets, label, descriptor, byteLength, elementCount) {
  const descriptorHash = hostResourceDescriptorHash(descriptor);
  let sourceDescriptorHash = hashString(hashU32(fnv64Offset, fields.length), label);
  let layoutHash = hashString(hashU32(fnv64Offset, fields.length), label);
  layoutHash = hashNonNegativeInteger(layoutHash, byteLength, `${label} model pack byteLength`);
  layoutHash = hashNonNegativeInteger(layoutHash, elementCount, `${label} model pack elementCount`);
  layoutHash = hashUint64(layoutHash, descriptorHash, `${label} model pack descriptor hash`);
  layoutHash = hashOptionalUint64(layoutHash, session.modelManifest?.hash, `${label} model manifest hash`);
  layoutHash = hashOptionalUint64(layoutHash, session.modelTable?.hash, `${label} model table hash`);
  layoutHash = hashOptionalUint64(layoutHash, session.modelTable?.descriptorHash, `${label} model table descriptor hash`);
  const entries = fields.map((field) => {
    const resource = projection[field];
    const sourceHash = hostResourceDescriptorHash(resource.descriptor);
    sourceDescriptorHash = hashString(sourceDescriptorHash, field);
    sourceDescriptorHash = hashString(sourceDescriptorHash, resource.role ?? "");
    sourceDescriptorHash = hashUint64(sourceDescriptorHash, sourceHash, `${label} ${field} source descriptor hash`);
    layoutHash = hashString(layoutHash, field);
    layoutHash = hashString(layoutHash, resource.role ?? "");
    layoutHash = hashString(layoutHash, normalizeLlamaModelResourceDtype(resource.dtype ?? "f32"));
    layoutHash = hashNonNegativeInteger(layoutHash, offsets[field], `${label} ${field} offset`);
    layoutHash = hashNonNegativeInteger(layoutHash, resource.byteLength, `${label} ${field} byteLength`);
    layoutHash = hashNonNegativeInteger(layoutHash, resource.elementCount, `${label} ${field} elementCount`);
    layoutHash = hashUint64(layoutHash, sourceHash, `${label} ${field} descriptor hash`);
    layoutHash = hashU32(layoutHash, Array.isArray(resource.shape) ? resource.shape.length : 0);
    if (Array.isArray(resource.shape)) {
      for (const dim of resource.shape) {
        layoutHash = hashNonNegativeInteger(layoutHash, dim, `${label} ${field} shape dimension`);
      }
    }
    return Object.freeze({
      byteLength: resource.byteLength,
      descriptorHash: sourceHash,
      dtype: normalizeLlamaModelResourceDtype(resource.dtype ?? "f32"),
      elementCount: resource.elementCount,
      field,
      offset: offsets[field],
      role: resource.role ?? "",
      shape: Object.freeze(Array.isArray(resource.shape) ? resource.shape.slice() : []),
    });
  });
  return Object.freeze({
    byteLength,
    descriptorHash,
    elementCount,
    entries: Object.freeze(entries),
    fieldCount: entries.length,
    layoutHash,
    modelManifestHash: session.modelManifest?.hash ?? 0n,
    modelTableDescriptorHash: session.modelTable?.descriptorHash ?? 0n,
    modelTableHash: session.modelTable?.hash ?? 0n,
    sourceDescriptorHash,
  });
}

function registerLlamaSessionModelPack(session, pack) {
  if (!Array.isArray(session.modelPacks)) return;
  session.modelPacks.push(pack);
}

function registerLlamaSessionOwnedResource(session, resource, fallbackOwner = null) {
  if (Array.isArray(session.ownedResources)) {
    session.ownedResources.push(resource);
    return;
  }
  fallbackOwner?.ownedResources?.push(resource);
}

function destroyLlamaSessionOwnedResources(session) {
  const resources = Array.isArray(session.ownedResources) ? session.ownedResources : [];
  for (let i = resources.length - 1; i >= 0; i -= 1) {
    destroyWasmWebGpuDeviceResource(session.device, resources[i]);
  }
  session.ownedResources = [];
}

function rollbackLlamaSessionOwnedResources(session, count) {
  if (!Array.isArray(session.ownedResources)) return;
  if (!Number.isSafeInteger(count) || count < 0) return;
  if (session.ownedResources.length <= count) return;
  const resources = session.ownedResources.splice(count);
  for (let i = resources.length - 1; i >= 0; i -= 1) {
    destroyWasmWebGpuDeviceResource(session.device, resources[i]);
  }
}

function llamaModelPackRecordEvidence(pack) {
  if (!pack) return {};
  return {
    modelPackByteLength: pack.byteLength,
    modelPackDescriptorHash: pack.descriptorHash ?? 0n,
    modelPackElementCount: pack.elementCount,
    modelPackFieldCount: pack.fieldCount ?? pack.entries?.length ?? 0,
    modelPackLayoutHash: pack.layoutHash ?? 0n,
    modelPackModelManifestHash: pack.modelManifestHash ?? 0n,
    modelPackModelTableDescriptorHash: pack.modelTableDescriptorHash ?? 0n,
    modelPackModelTableHash: pack.modelTableHash ?? 0n,
    modelPackSourceDescriptorHash: pack.sourceDescriptorHash ?? 0n,
  };
}

function writeLlamaModelPackResultEvidence(target, pack) {
  if (!pack) return target;
  target.modelPackDescriptorHash = pack.descriptorHash ?? undefined;
  target.modelPackLayoutHash = pack.layoutHash ?? undefined;
  return target;
}

function packLlamaProjectionModelResources(session, projection, fields, label, owner = null) {
  const offsets = {};
  let elementCount = 0;
  for (const field of fields) {
    const resource = projection[field];
    if (!resource || typeof resource !== "object") {
      throw new Error(`${label} model pack is missing resource ${field}`);
    }
    if (!Number.isSafeInteger(resource.elementCount) || resource.elementCount <= 0) {
      throw new Error(`${label} model pack resource ${field} has invalid elementCount`);
    }
    if (elementCount > 0xffffffff - resource.elementCount) {
      throw new Error(`${label} model pack is too large for u32 offsets`);
    }
    offsets[field] = elementCount;
    elementCount += resource.elementCount;
  }
  const byteLength = elementCount * Float32Array.BYTES_PER_ELEMENT;
  const resource = session.device.createBuffer({
    byteLength,
    label: `zgml.wasm.webgpu.${label.toLowerCase().replaceAll(" ", "-")}.model-pack`,
    usage: webgpuRequiredUsageFlags("read"),
  });
  registerLlamaSessionOwnedResource(session, resource, owner);
  const descriptor = storageResourceDescriptor(
    session.device,
    resource,
    { access: "read", byteLength },
    `${label} model pack`,
    byteLength,
  );
  const packed = new Float32Array(elementCount);
  for (const field of fields) {
    const source = projection[field];
    packed.set(
      mockFloat32View(source.descriptor, source.elementCount, `${label} model pack source ${field}`),
      offsets[field],
    );
  }
  session.device.writeFloat32(descriptor, packed);
  const evidence = describeLlamaProjectionModelPack(session, projection, fields, offsets, label, descriptor, byteLength, elementCount);
  const pack = Object.freeze({
    byteLength,
    descriptor,
    elementCount,
    ...evidence,
    fields: evidence.entries,
    offsets: Object.freeze(offsets),
    resource,
  });
  registerLlamaSessionModelPack(session, pack);
  return pack;
}

function appendPresentLlamaModelPackFields(target, projection, fields) {
  for (const field of fields) {
    if (projection[field] !== null && projection[field] !== undefined) target.push(field);
  }
  return target;
}

function createLlamaProjectionDummyStorage(session, label, access, owner = null) {
  const byteLength = Float32Array.BYTES_PER_ELEMENT;
  const resource = session.device.createBuffer({
    byteLength,
    label,
    usage: webgpuRequiredUsageFlags(access),
  });
  registerLlamaSessionOwnedResource(session, resource, owner);
  return {
    descriptor: storageResourceDescriptor(
      session.device,
      resource,
      { access, byteLength },
      label,
      byteLength,
    ),
    resource,
  };
}

function llamaEmbeddingProjectionResources(session, options = {}) {
  const embedding = requireF32MatrixModelResource(
    session,
    options.embeddingRole ?? "model.embed_tokens.weight",
    "LLaMA embedding projection",
  );
  const lmHead = requireF32MatrixModelResource(
    session,
    options.lmHeadRole ?? "lm_head.weight",
    "LLaMA embedding projection",
  );
  const lmHeadBias = optionalF32VectorModelResource(
    session,
    options.lmHeadBiasRole ?? "lm_head.bias",
    "LLaMA embedding projection",
  );
  if (embedding.rows < session.vocabSize || lmHead.rows < session.vocabSize) {
    throw new Error("LLaMA embedding projection resources must cover the Session vocab size");
  }
  if (embedding.cols !== lmHead.cols) {
    throw new Error(`LLaMA embedding projection hidden size mismatch: ${embedding.cols} != ${lmHead.cols}`);
  }
  if (lmHeadBias !== null && lmHeadBias.length < session.vocabSize) {
    throw new Error(`LLaMA embedding projection lm_head bias length mismatch: ${lmHeadBias.length} != ${session.vocabSize}`);
  }
  return {
    embedding,
    hiddenSize: embedding.cols,
    lmHead,
    lmHeadBias,
    vocabSize: session.vocabSize,
  };
}

function llamaRmsNormProjectionResources(session, options = {}) {
  const projection = llamaEmbeddingProjectionResources(session, options);
  const norm = requireF32VectorModelResource(
    session,
    options.normRole ?? "model.norm.weight",
    "LLaMA RMSNorm projection",
  );
  if (norm.length !== projection.hiddenSize) {
    throw new Error(`LLaMA RMSNorm projection norm length mismatch: ${norm.length} != ${projection.hiddenSize}`);
  }
  const epsilon = positiveFiniteNumberFromMetadataValue(
    options.epsilon ?? options.rmsNormEpsilon ?? options.rmsNormEps ?? options.rms_norm_eps ?? 1e-5,
    "LLaMA RMSNorm projection epsilon",
  );
  return {
    ...projection,
    epsilon,
    norm,
  };
}

function llamaActivationLogitsResources(session, options = {}) {
  const norm = requireF32VectorModelResource(
    session,
    options.normRole ?? "model.norm.weight",
    "LLaMA activation logits",
  );
  const lmHead = requireF32MatrixModelResource(
    session,
    options.lmHeadRole ?? "lm_head.weight",
    "LLaMA activation logits",
  );
  const lmHeadBias = optionalF32VectorModelResource(
    session,
    options.lmHeadBiasRole ?? "lm_head.bias",
    "LLaMA activation logits",
  );
  if (lmHead.rows < session.vocabSize) {
    throw new Error("LLaMA activation logits lm_head must cover the Session vocab size");
  }
  if (lmHead.cols !== norm.length) {
    throw new Error(`LLaMA activation logits hidden size mismatch: ${lmHead.cols} != ${norm.length}`);
  }
  if (lmHeadBias !== null && lmHeadBias.length < session.vocabSize) {
    throw new Error(`LLaMA activation logits lm_head bias length mismatch: ${lmHeadBias.length} != ${session.vocabSize}`);
  }
  const projection = {
    contextLength: session.contextLength,
    hiddenSize: norm.length,
  };
  const activation = prepareLlamaActivationSlot(
    session,
    projection,
    "LLaMA activation logits",
    "activationInput",
  );
  if (activation === null) {
    throw new Error("LLaMA activation logits needs an activation input resource");
  }
  const epsilon = positiveFiniteNumberFromMetadataValue(
    options.epsilon ?? options.rmsNormEpsilon ?? options.rmsNormEps ?? options.rms_norm_eps ?? 1e-5,
    "LLaMA activation logits epsilon",
  );
  return {
    activation,
    epsilon,
    hiddenSize: norm.length,
    lmHead,
    lmHeadBias,
    norm,
    vocabSize: session.vocabSize,
  };
}

function llamaKvProjectionResources(session, options = {}) {
  const layer = normalizeLlamaLayerIndex(options.layer ?? 0, "LLaMA K/V projection layer");
  const embedding = requireF32MatrixModelResource(
    session,
    options.embeddingRole ?? "model.embed_tokens.weight",
    "LLaMA K/V projection",
  );
  const inputNorm = requireF32VectorModelResource(
    session,
    options.inputNormRole ?? llamaLayerRole(layer, "input_layernorm.weight"),
    "LLaMA K/V projection",
  );
  const kProj = requireF32MatrixModelResource(
    session,
    options.kProjRole ?? llamaLayerRole(layer, "self_attn.k_proj.weight"),
    "LLaMA K/V projection",
  );
  const vProj = requireF32MatrixModelResource(
    session,
    options.vProjRole ?? llamaLayerRole(layer, "self_attn.v_proj.weight"),
    "LLaMA K/V projection",
  );
  const qkProjectionNorm = options.qkProjectionNorm === true || options.qk_projection_norm === true || options.qk_norm === true;
  const kNorm = qkProjectionNorm
    ? requireF32VectorModelResource(
      session,
      options.kNormRole ?? llamaLayerRole(layer, "self_attn.k_norm.weight"),
      "LLaMA K/V projection",
    )
    : null;
  if (embedding.rows < session.vocabSize) {
    throw new Error("LLaMA K/V projection embedding resource must cover the Session vocab size");
  }
  const hiddenSize = embedding.cols;
  if (inputNorm.length !== hiddenSize) {
    throw new Error(`LLaMA K/V projection norm length mismatch: ${inputNorm.length} != ${hiddenSize}`);
  }
  if (kProj.cols !== hiddenSize || vProj.cols !== hiddenSize) {
    throw new Error(`LLaMA K/V projection hidden size mismatch: ${kProj.cols}/${vProj.cols} != ${hiddenSize}`);
  }
  const explicitHeadSize = options.attentionHeadSize ?? options.headSize ?? null;
  const attentionHeadSize = explicitHeadSize === null
    ? (kProj.rows === hiddenSize ? hiddenSize : null)
    : explicitHeadSize;
  if (qkProjectionNorm) {
    if (!Number.isSafeInteger(attentionHeadSize) || attentionHeadSize <= 0) {
      throw new Error("LLaMA K/V projection Q/K norm needs an explicit positive attentionHeadSize when K rows are narrower than hidden");
    }
    if (kProj.rows % attentionHeadSize !== 0) {
      throw new Error(`LLaMA K/V projection K rows must be a multiple of head size: k=${kProj.rows} head=${attentionHeadSize}`);
    }
    if (kNorm.length !== attentionHeadSize) {
      throw new Error(`LLaMA K/V projection K norm length mismatch: ${kNorm.length} != ${attentionHeadSize}`);
    }
  }
  const epsilon = positiveFiniteNumberFromMetadataValue(
    options.epsilon ?? options.rmsNormEpsilon ?? options.rmsNormEps ?? options.rms_norm_eps ?? 1e-5,
    "LLaMA K/V projection epsilon",
  );
  return {
    embedding,
    epsilon,
    hiddenSize,
    inputNorm,
    attentionHeadSize,
    kProj,
    kNorm,
    layer,
    qkProjectionNorm,
    vProj,
  };
}

function llamaAttentionProjectionResources(session, options = {}) {
  const projection = llamaKvProjectionResources(session, options);
  const qProj = requireF32MatrixModelResource(
    session,
    options.qProjRole ?? llamaLayerRole(projection.layer, "self_attn.q_proj.weight"),
    "LLaMA attention projection",
  );
  const qNorm = projection.qkProjectionNorm
    ? requireF32VectorModelResource(
      session,
      options.qNormRole ?? llamaLayerRole(projection.layer, "self_attn.q_norm.weight"),
      "LLaMA attention projection",
    )
    : null;
  const oProj = requireF32MatrixModelResource(
    session,
    options.oProjRole ?? llamaLayerRole(projection.layer, "self_attn.o_proj.weight"),
    "LLaMA attention projection",
  );
  const norm = requireF32VectorModelResource(
    session,
    options.normRole ?? "model.norm.weight",
    "LLaMA attention projection",
  );
  const lmHead = requireF32MatrixModelResource(
    session,
    options.lmHeadRole ?? "lm_head.weight",
    "LLaMA attention projection",
  );
  const lmHeadBias = optionalF32VectorModelResource(
    session,
    options.lmHeadBiasRole ?? "lm_head.bias",
    "LLaMA attention projection",
  );
  const qBias = optionalF32VectorModelResource(
    session,
    options.qBiasRole ?? llamaLayerRole(projection.layer, "self_attn.q_proj.bias"),
    "LLaMA attention projection",
  );
  const kBias = optionalF32VectorModelResource(
    session,
    options.kBiasRole ?? llamaLayerRole(projection.layer, "self_attn.k_proj.bias"),
    "LLaMA attention projection",
  );
  const vBias = optionalF32VectorModelResource(
    session,
    options.vBiasRole ?? llamaLayerRole(projection.layer, "self_attn.v_proj.bias"),
    "LLaMA attention projection",
  );
  const oBias = optionalF32VectorModelResource(
    session,
    options.oBiasRole ?? llamaLayerRole(projection.layer, "self_attn.o_proj.bias"),
    "LLaMA attention projection",
  );
  if (qProj.cols !== projection.hiddenSize) {
    throw new Error(`LLaMA attention projection Q shape mismatch: ${qProj.rows}x${qProj.cols}`);
  }
  if (qProj.rows !== projection.hiddenSize) {
    throw new Error(`LLaMA attention projection proof currently requires Q rows to match hidden size: ${qProj.rows} != ${projection.hiddenSize}`);
  }
  if (projection.kProj.rows !== projection.vProj.rows) {
    throw new Error(`LLaMA attention projection K/V row mismatch: ${projection.kProj.rows} != ${projection.vProj.rows}`);
  }
  const explicitHeadSize = options.attentionHeadSize ?? options.headSize ?? null;
  const attentionHeadSize = explicitHeadSize === null
    ? (projection.kProj.rows === projection.hiddenSize ? projection.hiddenSize : null)
    : explicitHeadSize;
  if (!Number.isSafeInteger(attentionHeadSize) || attentionHeadSize <= 0) {
    throw new Error("LLaMA grouped attention projection needs an explicit positive attentionHeadSize when K/V rows are narrower than hidden");
  }
  if (attentionHeadSize % 2 !== 0) {
    throw new Error(`LLaMA attention head size must be even for RoPE: ${attentionHeadSize}`);
  }
  if (qProj.rows % attentionHeadSize !== 0 || projection.kProj.rows % attentionHeadSize !== 0) {
    throw new Error(`LLaMA attention projection rows must be multiples of head size: q=${qProj.rows} kv=${projection.kProj.rows} head=${attentionHeadSize}`);
  }
  if (projection.qkProjectionNorm && qNorm.length !== attentionHeadSize) {
    throw new Error(`LLaMA attention projection Q norm length mismatch: ${qNorm.length} != ${attentionHeadSize}`);
  }
  const queryHeadCount = qProj.rows / attentionHeadSize;
  const kvHeadCount = projection.kProj.rows / attentionHeadSize;
  if (kvHeadCount <= 0 || queryHeadCount % kvHeadCount !== 0) {
    throw new Error(`LLaMA attention projection query/KV heads must group evenly: q=${queryHeadCount} kv=${kvHeadCount}`);
  }
  if (oProj.rows !== projection.hiddenSize || oProj.cols !== qProj.rows) {
    throw new Error(`LLaMA attention projection O shape mismatch: ${oProj.rows}x${oProj.cols}`);
  }
  if (norm.length !== projection.hiddenSize) {
    throw new Error(`LLaMA attention projection final norm length mismatch: ${norm.length} != ${projection.hiddenSize}`);
  }
  if (lmHead.rows < session.vocabSize || lmHead.cols !== projection.hiddenSize) {
    throw new Error(`LLaMA attention projection lm_head shape mismatch: ${lmHead.rows}x${lmHead.cols}`);
  }
  if (lmHeadBias !== null && lmHeadBias.length < session.vocabSize) {
    throw new Error(`LLaMA attention projection lm_head bias length mismatch: ${lmHeadBias.length} != ${session.vocabSize}`);
  }
  if (qBias !== null && qBias.length !== qProj.rows) {
    throw new Error(`LLaMA attention projection Q bias length mismatch: ${qBias.length} != ${qProj.rows}`);
  }
  if (kBias !== null && kBias.length !== projection.kProj.rows) {
    throw new Error(`LLaMA attention projection K bias length mismatch: ${kBias.length} != ${projection.kProj.rows}`);
  }
  if (vBias !== null && vBias.length !== projection.vProj.rows) {
    throw new Error(`LLaMA attention projection V bias length mismatch: ${vBias.length} != ${projection.vProj.rows}`);
  }
  if (oBias !== null && oBias.length !== projection.hiddenSize) {
    throw new Error(`LLaMA attention projection O bias length mismatch: ${oBias.length} != ${projection.hiddenSize}`);
  }
  const ropeBase = positiveFiniteNumberFromMetadataValue(options.ropeBase ?? options.ropeTheta ?? 10000, "LLaMA attention projection RoPE base");
  const ropeScale = positiveFiniteNumberFromMetadataValue(
    options.ropeScale ?? options.rope_scale ?? options.ropeScalingFactor ?? options.rope_scaling_factor ?? 1,
    "LLaMA attention projection RoPE scale",
  );
  const explicitRopeKind = options.ropeKind ?? options.ropeType ?? options.rope_kind ?? options.rope_type;
  const ropeKind = normalizeLlamaRopeKind(explicitRopeKind ?? (Math.abs(ropeScale - 1) > 1e-6 ? "linear" : "none"));
  const ropeLowFreqFactor = positiveFiniteNumberFromMetadataValue(
    options.ropeLowFreqFactor ?? options.rope_low_freq_factor ?? 1,
    "LLaMA attention projection RoPE low frequency factor",
  );
  const ropeHighFreqFactor = positiveFiniteNumberFromMetadataValue(
    options.ropeHighFreqFactor ?? options.rope_high_freq_factor ?? 4,
    "LLaMA attention projection RoPE high frequency factor",
  );
  const ropeOriginalContextLength = positiveIntegerFromMetadataValue(
    options.ropeOriginalContextLength ?? options.rope_original_context_length ?? 8192,
    "LLaMA attention projection RoPE original context length",
  );
  if (ropeKind === "llama3" && ropeHighFreqFactor <= ropeLowFreqFactor) {
    throw new Error(`invalid LLaMA attention projection RoPE frequency factors: low=${ropeLowFreqFactor} high=${ropeHighFreqFactor}`);
  }
  const slidingWindow = normalizeLlamaSlidingWindowFromOptions(
    options,
    "LLaMA attention projection",
  );
  const useRope = normalizeLlamaUseRopeLayerFromOptions(
    options,
    projection.layer,
    "LLaMA attention projection",
  );
  return {
    ...projection,
    lmHead,
    lmHeadBias,
    norm,
    oProj,
    qBias,
    kBias,
    vBias,
    oBias,
    biasMask:
      (qBias === null ? 0 : 1) |
      (kBias === null ? 0 : 2) |
      (vBias === null ? 0 : 4) |
      (oBias === null ? 0 : 8) |
      (lmHeadBias === null ? 0 : 128),
    attentionHeadSize,
    kvHeadCount,
    qNorm,
    queryHeadCount,
    queryHeadsPerKvHead: queryHeadCount / kvHeadCount,
    qProj,
    ropeBase,
    ropeHighFreqFactor,
    ropeKind,
    ropeKindId: llamaRopeKindId(ropeKind),
    ropeLowFreqFactor,
    ropeOriginalContextLength,
    ropeScale,
    slidingWindow,
    useRope,
    vocabSize: session.vocabSize,
  };
}

function llamaBlockProjectionResources(session, options = {}) {
  const projection = llamaAttentionProjectionResources(session, options);
  const postNorm = requireF32VectorModelResource(
    session,
    options.postNormRole ?? llamaLayerRole(projection.layer, "post_attention_layernorm.weight"),
    "LLaMA block projection",
  );
  const gateProj = requireF32MatrixModelResource(
    session,
    options.gateProjRole ?? llamaLayerRole(projection.layer, "mlp.gate_proj.weight"),
    "LLaMA block projection",
  );
  const upProj = requireF32MatrixModelResource(
    session,
    options.upProjRole ?? llamaLayerRole(projection.layer, "mlp.up_proj.weight"),
    "LLaMA block projection",
  );
  const downProj = requireF32MatrixModelResource(
    session,
    options.downProjRole ?? llamaLayerRole(projection.layer, "mlp.down_proj.weight"),
    "LLaMA block projection",
  );
  const gateBias = optionalF32VectorModelResource(
    session,
    options.gateBiasRole ?? llamaLayerRole(projection.layer, "mlp.gate_proj.bias"),
    "LLaMA block projection",
  );
  const upBias = optionalF32VectorModelResource(
    session,
    options.upBiasRole ?? llamaLayerRole(projection.layer, "mlp.up_proj.bias"),
    "LLaMA block projection",
  );
  const downBias = optionalF32VectorModelResource(
    session,
    options.downBiasRole ?? llamaLayerRole(projection.layer, "mlp.down_proj.bias"),
    "LLaMA block projection",
  );
  if (postNorm.length !== projection.hiddenSize) {
    throw new Error(`LLaMA block projection post-attention norm length mismatch: ${postNorm.length} != ${projection.hiddenSize}`);
  }
  if (gateProj.cols !== projection.hiddenSize || upProj.cols !== projection.hiddenSize) {
    throw new Error(`LLaMA block projection FFN input shape mismatch: ${gateProj.rows}x${gateProj.cols}/${upProj.rows}x${upProj.cols}`);
  }
  if (gateProj.rows !== upProj.rows) {
    throw new Error(`LLaMA block projection gate/up row mismatch: ${gateProj.rows} != ${upProj.rows}`);
  }
  if (downProj.rows !== projection.hiddenSize || downProj.cols !== gateProj.rows) {
    throw new Error(`LLaMA block projection down shape mismatch: ${downProj.rows}x${downProj.cols}`);
  }
  if (gateBias !== null && gateBias.length !== gateProj.rows) {
    throw new Error(`LLaMA block projection gate bias length mismatch: ${gateBias.length} != ${gateProj.rows}`);
  }
  if (upBias !== null && upBias.length !== upProj.rows) {
    throw new Error(`LLaMA block projection up bias length mismatch: ${upBias.length} != ${upProj.rows}`);
  }
  if (downBias !== null && downBias.length !== downProj.rows) {
    throw new Error(`LLaMA block projection down bias length mismatch: ${downBias.length} != ${downProj.rows}`);
  }
  return {
    ...projection,
    biasMask:
      projection.biasMask |
      (gateBias === null ? 0 : 16) |
      (upBias === null ? 0 : 32) |
      (downBias === null ? 0 : 64),
    downProj,
    downBias,
    ffnSize: gateProj.rows,
    gateProj,
    gateBias,
    postNorm,
    upProj,
    upBias,
  };
}

function llamaActiveTokenCount(execution) {
  return execution.tokensLen ?? execution.tokens.length;
}

function llamaLastToken(execution) {
  const length = llamaActiveTokenCount(execution);
  return execution.tokens[length - 1];
}

function llamaExecutionTokensArray(execution) {
  const tokens = execution.tokens ?? [];
  const length = llamaActiveTokenCount(execution);
  const out = new Array(length);
  for (let i = 0; i < length; i += 1) out[i] = tokens[i];
  return out;
}

function normalizeLlamaLayerIndex(layer, label = "LLaMA layer") {
  if (!Number.isSafeInteger(layer) || layer < 0) {
    throw new Error(`invalid ${label}: ${layer}`);
  }
  return layer;
}

function llamaLayerRole(layer, suffix) {
  return `model.layers.${normalizeLlamaLayerIndex(layer)}.${suffix}`;
}

function prepareLlamaProjectionKvLayer(session, projection, label, options = {}) {
  const kvLayer = prepareLlamaKvCacheLayer(session, projection.layer, label);
  if (kvLayer.k.stride !== kvLayer.v.stride) {
    throw new Error(`${label} K/V stride mismatch: ${kvLayer.k.stride} != ${kvLayer.v.stride}`);
  }
  if (projection.kProj.rows !== kvLayer.k.stride || projection.vProj.rows !== kvLayer.v.stride) {
    throw new Error(`${label} rows must match cache stride: ${projection.kProj.rows}/${projection.vProj.rows} != ${kvLayer.k.stride}`);
  }
  return kvLayer;
}

function rmsNormVectorInto(target, values, norm, epsilon, length = values.length) {
  let sumSquares = Math.fround(0);
  for (let i = 0; i < length; i += 1) {
    sumSquares = Math.fround(sumSquares + Math.fround(values[i] * values[i]));
  }
  const invRms = Math.fround(1 / Math.sqrt(Math.fround(sumSquares / length + epsilon)));
  for (let i = 0; i < length; i += 1) {
    target[i] = Math.fround(Math.fround(values[i] * norm[i]) * invRms);
  }
  return target;
}

function rmsNormVector(values, norm, epsilon) {
  return rmsNormVectorInto(new Float32Array(values.length), values, norm, epsilon);
}

function rmsNormVectorByHeadInto(target, values, norm, epsilon, headSize, length = values.length, valueOffset = 0, targetOffset = valueOffset) {
  if (!Number.isSafeInteger(headSize) || headSize <= 0 || length % headSize !== 0) {
    throw new Error(`invalid per-head RMSNorm shape: length=${length} head=${headSize}`);
  }
  for (let headOffset = 0; headOffset < length; headOffset += headSize) {
    let sumSquares = Math.fround(0);
    for (let i = 0; i < headSize; i += 1) {
      const value = values[valueOffset + headOffset + i];
      sumSquares = Math.fround(sumSquares + Math.fround(value * value));
    }
    const invRms = Math.fround(1 / Math.sqrt(Math.fround(sumSquares / headSize + epsilon)));
    for (let i = 0; i < headSize; i += 1) {
      const value = values[valueOffset + headOffset + i];
      target[targetOffset + headOffset + i] = Math.fround(Math.fround(value * norm[i]) * invRms);
    }
  }
  return target;
}

function matVecInto(target, weight, rows, cols, input) {
  for (let row = 0; row < rows; row += 1) {
    let acc = Math.fround(0);
    const weightOffset = row * cols;
    for (let col = 0; col < cols; col += 1) {
      acc = Math.fround(acc + Math.fround(weight[weightOffset + col] * input[col]));
    }
    target[row] = acc;
  }
  return target;
}

function matVec(weight, rows, cols, input) {
  return matVecInto(new Float32Array(rows), weight, rows, cols, input);
}

function addOptionalVectorBiasInto(target, biasResource, length, label) {
  if (biasResource === null || biasResource === undefined) return target;
  const bias = mockFloat32View(
    biasResource.descriptor,
    biasResource.elementCount,
    label,
  );
  for (let i = 0; i < length; i += 1) {
    target[i] = Math.fround(target[i] + bias[i]);
  }
  return target;
}

function llamaRopeInvFrequency(pair, headSize, base = 10000, scaling = {}) {
  const invFrequency = 1 / Math.pow(base, (2 * pair) / headSize);
  const kind = normalizeLlamaRopeKind(scaling.ropeKind ?? scaling.kind ?? (Math.abs((scaling.ropeScale ?? scaling.scale ?? 1) - 1) > 1e-6 ? "linear" : "none"));
  if (kind === "linear") return invFrequency / (scaling.ropeScale ?? scaling.scale ?? 1);
  if (kind === "llama3") {
    const factor = scaling.ropeScale ?? scaling.scale ?? 1;
    const lowFreqFactor = scaling.ropeLowFreqFactor ?? scaling.lowFreqFactor ?? 1;
    const highFreqFactor = scaling.ropeHighFreqFactor ?? scaling.highFreqFactor ?? 4;
    const originalContextLength = scaling.ropeOriginalContextLength ?? scaling.originalContextLength ?? 8192;
    const wavelen = (2 * Math.PI) / invFrequency;
    const lowFreqWavelen = originalContextLength / lowFreqFactor;
    const highFreqWavelen = originalContextLength / highFreqFactor;
    if (wavelen > lowFreqWavelen) return invFrequency / factor;
    if (wavelen < highFreqWavelen) return invFrequency;
    const smoothFactor = (originalContextLength / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
    return Math.fround(Math.fround(1 - smoothFactor) * Math.fround(invFrequency / factor) + Math.fround(smoothFactor * invFrequency));
  }
  return invFrequency;
}

function ropeVectorInto(target, values, position, base = 10000, scaling = {}, length = values.length) {
  const half = Math.floor(length / 2);
  for (let i = 0; i < half; i += 1) {
    const freq = position * llamaRopeInvFrequency(i, length, base, scaling);
    const cos = Math.fround(Math.cos(freq));
    const sin = Math.fround(Math.sin(freq));
    const lo = values[i];
    const hi = values[i + half];
    target[i] = Math.fround(Math.fround(lo * cos) - Math.fround(hi * sin));
    target[i + half] = Math.fround(Math.fround(hi * cos) + Math.fround(lo * sin));
  }
  if (length % 2 !== 0) target[length - 1] = Math.fround(0);
  return target;
}

function ropeVector(values, position, base = 10000, scaling = {}) {
  return ropeVectorInto(new Float32Array(values.length), values, position, base, scaling);
}

function ropeVectorByHeadInto(target, values, position, headSize, base = 10000, scaling = {}, length = values.length) {
  if (headSize === length) return ropeVectorInto(target, values, position, base, scaling, length);
  const half = Math.floor(headSize / 2);
  for (let headOffset = 0; headOffset < length; headOffset += headSize) {
    for (let i = 0; i < half; i += 1) {
      const freq = position * llamaRopeInvFrequency(i, headSize, base, scaling);
      const cos = Math.fround(Math.cos(freq));
      const sin = Math.fround(Math.sin(freq));
      const loIndex = headOffset + i;
      const hiIndex = headOffset + i + half;
      const lo = values[loIndex];
      const hi = values[hiIndex];
      target[loIndex] = Math.fround(Math.fround(lo * cos) - Math.fround(hi * sin));
      target[hiIndex] = Math.fround(Math.fround(hi * cos) + Math.fround(lo * sin));
    }
  }
  return target;
}

function ropeVectorByHead(values, position, headSize, base = 10000, scaling = {}) {
  return ropeVectorByHeadInto(new Float32Array(values.length), values, position, headSize, base, scaling);
}

function silu(value) {
  return Math.fround(value / Math.fround(1 + Math.exp(-value)));
}

function llamaProjectionInputHiddenInto(target, context, projection, token, position, label) {
  if (projection.activationInput != null) {
    const activation = mockFloat32View(
      projection.activationInput.descriptor,
      projection.activationInput.elementCount,
      `${label} activation input`,
    );
    const offset = position * projection.hiddenSize;
    for (let col = 0; col < projection.hiddenSize; col += 1) {
      target[col] = activation[offset + col];
    }
    return target;
  }
  const embedding = mockFloat32View(
    projection.embedding.descriptor,
    projection.embedding.elementCount,
    `${label} embedding`,
  );
  const tokenOffset = token * projection.hiddenSize;
  for (let col = 0; col < projection.hiddenSize; col += 1) {
    target[col] = embedding[tokenOffset + col];
  }
  return target;
}

function llamaProjectionInputHidden(context, projection, token, position, label) {
  return llamaProjectionInputHiddenInto(new Float32Array(projection.hiddenSize), context, projection, token, position, label);
}

function llamaEmbeddingProjectionMockLogitsInto(target, context, projection) {
  const token = llamaLastToken(context.execution);
  const embedding = mockFloat32View(
    projection.embedding.descriptor,
    projection.embedding.elementCount,
    "mock LLaMA embedding projection embedding",
  );
  const lmHead = mockFloat32View(
    projection.lmHead.descriptor,
    projection.lmHead.elementCount,
    "mock LLaMA embedding projection lm_head",
  );
  const tokenOffset = token * projection.hiddenSize;
  for (let row = 0; row < projection.vocabSize; row += 1) {
    let acc = Math.fround(0);
    const lmOffset = row * projection.hiddenSize;
    for (let col = 0; col < projection.hiddenSize; col += 1) {
      acc = Math.fround(acc + Math.fround(lmHead[lmOffset + col] * embedding[tokenOffset + col]));
    }
    target[row] = acc;
  }
  addOptionalVectorBiasInto(target, projection.lmHeadBias, projection.vocabSize, "mock LLaMA embedding projection lm_head bias");
  return target;
}

function llamaRmsNormProjectionMockLogitsInto(target, context, projection) {
  const token = llamaLastToken(context.execution);
  const embedding = mockFloat32View(
    projection.embedding.descriptor,
    projection.embedding.elementCount,
    "mock LLaMA RMSNorm projection embedding",
  );
  const norm = mockFloat32View(
    projection.norm.descriptor,
    projection.norm.elementCount,
    "mock LLaMA RMSNorm projection norm",
  );
  const lmHead = mockFloat32View(
    projection.lmHead.descriptor,
    projection.lmHead.elementCount,
    "mock LLaMA RMSNorm projection lm_head",
  );
  const tokenOffset = token * projection.hiddenSize;
  let sumSquares = Math.fround(0);
  for (let col = 0; col < projection.hiddenSize; col += 1) {
    const value = embedding[tokenOffset + col];
    sumSquares = Math.fround(sumSquares + Math.fround(value * value));
  }
  const invRms = Math.fround(1 / Math.sqrt(Math.fround(sumSquares / projection.hiddenSize + projection.epsilon)));
  for (let row = 0; row < projection.vocabSize; row += 1) {
    let acc = Math.fround(0);
    const lmOffset = row * projection.hiddenSize;
    for (let col = 0; col < projection.hiddenSize; col += 1) {
      const hidden = Math.fround(Math.fround(embedding[tokenOffset + col] * norm[col]) * invRms);
      acc = Math.fround(acc + Math.fround(lmHead[lmOffset + col] * hidden));
    }
    target[row] = acc;
  }
  addOptionalVectorBiasInto(target, projection.lmHeadBias, projection.vocabSize, "mock LLaMA RMSNorm projection lm_head bias");
  return target;
}

function llamaActivationLogitsMockLogitsInto(target, context, projection) {
  const position = context.stepParams.endPosition - 1;
  if (!Number.isSafeInteger(position) || position < 0 || position >= context.session.contextLength) {
    throw new Error(`mock LLaMA activation logits position out of range: ${position}`);
  }
  const activation = mockFloat32View(
    projection.activation.descriptor,
    projection.activation.elementCount,
    "mock LLaMA activation logits activation",
  );
  const norm = mockFloat32View(
    projection.norm.descriptor,
    projection.norm.elementCount,
    "mock LLaMA activation logits norm",
  );
  const lmHead = mockFloat32View(
    projection.lmHead.descriptor,
    projection.lmHead.elementCount,
    "mock LLaMA activation logits lm_head",
  );
  const activationOffset = position * projection.hiddenSize;
  let sumSquares = Math.fround(0);
  for (let col = 0; col < projection.hiddenSize; col += 1) {
    const value = activation[activationOffset + col];
    sumSquares = Math.fround(sumSquares + Math.fround(value * value));
  }
  const invRms = Math.fround(1 / Math.sqrt(Math.fround(sumSquares / projection.hiddenSize + projection.epsilon)));
  for (let row = 0; row < projection.vocabSize; row += 1) {
    let acc = Math.fround(0);
    const lmOffset = row * projection.hiddenSize;
    for (let col = 0; col < projection.hiddenSize; col += 1) {
      const hidden = Math.fround(Math.fround(activation[activationOffset + col] * norm[col]) * invRms);
      acc = Math.fround(acc + Math.fround(lmHead[lmOffset + col] * hidden));
    }
    target[row] = acc;
  }
  return target;
}

function ensureFloat32ScratchCapacity(value, length) {
  if (value instanceof Float32Array && value.length >= length) return value;
  return new Float32Array(length);
}

function createLlamaAttentionProjectionMockScratch() {
  return {
    afterAttention: new Float32Array(0),
    attention: new Float32Array(0),
    attnProjected: new Float32Array(0),
    finalHidden: new Float32Array(0),
    hidden: new Float32Array(0),
    k: new Float32Array(0),
    logits: new Float32Array(0),
    normed: new Float32Array(0),
    q: new Float32Array(0),
    result: {
      afterAttention: null,
      afterAttentionLength: 0,
      attention: null,
      attentionLength: 0,
      k: null,
      kLength: 0,
      logits: null,
      logitsLength: 0,
      v: null,
      vLength: 0,
    },
    scores: new Float32Array(0),
    v: new Float32Array(0),
  };
}

function prepareLlamaAttentionProjectionMockScratch(scratch, projection, seq, attentionStart, outputLogits) {
  scratch.hidden = ensureFloat32ScratchCapacity(scratch.hidden, projection.hiddenSize);
  scratch.normed = ensureFloat32ScratchCapacity(scratch.normed, projection.hiddenSize);
  scratch.q = ensureFloat32ScratchCapacity(scratch.q, projection.qProj.rows);
  scratch.k = ensureFloat32ScratchCapacity(scratch.k, projection.kProj.rows);
  scratch.v = ensureFloat32ScratchCapacity(scratch.v, projection.vProj.rows);
  scratch.attention = ensureFloat32ScratchCapacity(scratch.attention, projection.qProj.rows);
  scratch.scores = ensureFloat32ScratchCapacity(scratch.scores, Math.max(0, seq - attentionStart));
  scratch.attnProjected = ensureFloat32ScratchCapacity(scratch.attnProjected, projection.hiddenSize);
  scratch.afterAttention = ensureFloat32ScratchCapacity(scratch.afterAttention, projection.hiddenSize);
  if (outputLogits) {
    scratch.finalHidden = ensureFloat32ScratchCapacity(scratch.finalHidden, projection.hiddenSize);
    scratch.logits = ensureFloat32ScratchCapacity(scratch.logits, projection.vocabSize);
  }
  return scratch;
}

function createLlamaBlockProjectionMockScratch() {
  return {
    attention: createLlamaAttentionProjectionMockScratch(),
    attentionOptions: {
      outputLogits: false,
      position: 0,
      seq: 0,
      token: 0,
    },
    blockHidden: new Float32Array(0),
    ffn: new Float32Array(0),
    ffnInput: new Float32Array(0),
    finalHidden: new Float32Array(0),
    gate: new Float32Array(0),
    logits: new Float32Array(0),
    normedFfn: new Float32Array(0),
    result: {
      blockHidden: null,
      blockHiddenLength: 0,
      k: null,
      kLength: 0,
      logits: null,
      logitsLength: 0,
      v: null,
      vLength: 0,
    },
    up: new Float32Array(0),
  };
}

function prepareLlamaBlockProjectionMockScratch(scratch, projection, outputLogits) {
  scratch.normedFfn = ensureFloat32ScratchCapacity(scratch.normedFfn, projection.hiddenSize);
  scratch.gate = ensureFloat32ScratchCapacity(scratch.gate, projection.ffnSize);
  scratch.up = ensureFloat32ScratchCapacity(scratch.up, projection.ffnSize);
  scratch.ffnInput = ensureFloat32ScratchCapacity(scratch.ffnInput, projection.ffnSize);
  scratch.ffn = ensureFloat32ScratchCapacity(scratch.ffn, projection.hiddenSize);
  scratch.blockHidden = ensureFloat32ScratchCapacity(scratch.blockHidden, projection.hiddenSize);
  if (outputLogits) {
    scratch.finalHidden = ensureFloat32ScratchCapacity(scratch.finalHidden, projection.hiddenSize);
    scratch.logits = ensureFloat32ScratchCapacity(scratch.logits, projection.vocabSize);
  }
  return scratch;
}

function llamaKvProjectionMockValuesInto(target, context, projection, slot) {
  const weightResource = slot.kind === "k" ? projection.kProj : projection.vProj;
  const embedding = mockFloat32View(
    projection.embedding.descriptor,
    projection.embedding.elementCount,
    "mock LLaMA K/V projection embedding",
  );
  const norm = mockFloat32View(
    projection.inputNorm.descriptor,
    projection.inputNorm.elementCount,
    "mock LLaMA K/V projection input norm",
  );
  const weight = mockFloat32View(
    weightResource.descriptor,
    weightResource.elementCount,
    `mock LLaMA ${slot.kind.toUpperCase()} projection weight`,
  );
  const kNorm = slot.kind === "k" && projection.qkProjectionNorm
    ? mockFloat32View(
      projection.kNorm.descriptor,
      projection.kNorm.elementCount,
      "mock LLaMA K projection norm",
    )
    : null;
  const tokenCount = context.stepParams.tokenCount;
  for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
    const token = context.execution.tokens[tokenIndex];
    const embeddingOffset = token * projection.hiddenSize;
    let sumSquares = Math.fround(0);
    for (let col = 0; col < projection.hiddenSize; col += 1) {
      const value = embedding[embeddingOffset + col];
      sumSquares = Math.fround(sumSquares + Math.fround(value * value));
    }
    const invRms = Math.fround(1 / Math.sqrt(Math.fround(sumSquares / projection.hiddenSize + projection.epsilon)));
    for (let row = 0; row < slot.stride; row += 1) {
      let acc = Math.fround(0);
      const weightOffset = row * projection.hiddenSize;
      for (let col = 0; col < projection.hiddenSize; col += 1) {
        const hidden = Math.fround(Math.fround(embedding[embeddingOffset + col] * norm[col]) * invRms);
        acc = Math.fround(acc + Math.fround(weight[weightOffset + col] * hidden));
      }
      target[tokenIndex * slot.stride + row] = acc;
    }
    if (kNorm !== null) {
      rmsNormVectorByHeadInto(
        target,
        target,
        kNorm,
        projection.epsilon,
        projection.attentionHeadSize,
        slot.stride,
        tokenIndex * slot.stride,
      );
    }
  }
  return target;
}

function llamaAttentionProjectionMockInto(scratch, context, projection, slots, options = {}) {
  const token = options.token ?? llamaLastToken(context.execution);
  const position = options.position ?? context.stepParams.startPosition;
  const seq = options.seq ?? context.stepParams.endPosition;
  const attentionStart = options.attentionStart ?? llamaAttentionStartForSeq(
    seq,
    options.slidingWindow ?? projection.slidingWindow,
  );
  const outputLogits = options.outputLogits ?? true;
  prepareLlamaAttentionProjectionMockScratch(scratch, projection, seq, attentionStart, outputLogits);
  const inputNorm = mockFloat32View(
    projection.inputNorm.descriptor,
    projection.inputNorm.elementCount,
    "mock LLaMA attention projection input norm",
  );
  const qProj = mockFloat32View(
    projection.qProj.descriptor,
    projection.qProj.elementCount,
    "mock LLaMA Q projection weight",
  );
  const kProj = mockFloat32View(
    projection.kProj.descriptor,
    projection.kProj.elementCount,
    "mock LLaMA K projection weight",
  );
  const vProj = mockFloat32View(
    projection.vProj.descriptor,
    projection.vProj.elementCount,
    "mock LLaMA V projection weight",
  );
  const oProj = mockFloat32View(
    projection.oProj.descriptor,
    projection.oProj.elementCount,
    "mock LLaMA O projection weight",
  );
  const finalNorm = mockFloat32View(
    projection.norm.descriptor,
    projection.norm.elementCount,
    "mock LLaMA attention projection final norm",
  );
  const lmHead = mockFloat32View(
    projection.lmHead.descriptor,
    projection.lmHead.elementCount,
    "mock LLaMA attention projection lm_head",
  );
  const kCache = mockFloat32View(
    slots.k.descriptor,
    Math.floor(slots.k.descriptor.byteLength / Float32Array.BYTES_PER_ELEMENT),
    "mock LLaMA attention K cache",
  );
  const vCache = mockFloat32View(
    slots.v.descriptor,
    Math.floor(slots.v.descriptor.byteLength / Float32Array.BYTES_PER_ELEMENT),
    "mock LLaMA attention V cache",
  );
  const hidden = llamaProjectionInputHiddenInto(scratch.hidden, context, projection, token, position, "mock LLaMA attention projection");
  const normed = rmsNormVectorInto(scratch.normed, hidden, inputNorm, projection.epsilon, projection.hiddenSize);
  const q = matVecInto(scratch.q, qProj, projection.qProj.rows, projection.qProj.cols, normed);
  addOptionalVectorBiasInto(q, projection.qBias, projection.qProj.rows, "mock LLaMA Q projection bias");
  const k = matVecInto(scratch.k, kProj, projection.kProj.rows, projection.kProj.cols, normed);
  addOptionalVectorBiasInto(k, projection.kBias, projection.kProj.rows, "mock LLaMA K projection bias");
  if (projection.qkProjectionNorm) {
    const qNorm = mockFloat32View(
      projection.qNorm.descriptor,
      projection.qNorm.elementCount,
      "mock LLaMA Q projection norm",
    );
    const kNorm = mockFloat32View(
      projection.kNorm.descriptor,
      projection.kNorm.elementCount,
      "mock LLaMA K projection norm",
    );
    rmsNormVectorByHeadInto(q, q, qNorm, projection.epsilon, projection.attentionHeadSize, projection.qProj.rows);
    rmsNormVectorByHeadInto(k, k, kNorm, projection.epsilon, projection.attentionHeadSize, projection.kProj.rows);
  }
  if (projection.useRope) {
    ropeVectorByHeadInto(q, q, position, projection.attentionHeadSize, projection.ropeBase, projection, projection.qProj.rows);
    ropeVectorByHeadInto(k, k, position, projection.attentionHeadSize, projection.ropeBase, projection, projection.kProj.rows);
  }
  const v = matVecInto(scratch.v, vProj, projection.vProj.rows, projection.vProj.cols, normed);
  addOptionalVectorBiasInto(v, projection.vBias, projection.vProj.rows, "mock LLaMA V projection bias");
  const kCacheBase = position * slots.k.stride;
  for (let row = 0; row < projection.kProj.rows; row += 1) {
    kCache[kCacheBase + row] = k[row];
  }
  const vCacheBase = position * slots.v.stride;
  for (let row = 0; row < projection.vProj.rows; row += 1) {
    vCache[vCacheBase + row] = v[row];
  }

  const scale = Math.fround(1 / Math.sqrt(projection.attentionHeadSize));
  const attention = scratch.attention;
  const scores = scratch.scores;
  attention.fill(0, 0, projection.qProj.rows);
  for (let queryHead = 0; queryHead < projection.queryHeadCount; queryHead += 1) {
    const kvHead = Math.floor(queryHead / projection.queryHeadsPerKvHead);
    const qOffset = queryHead * projection.attentionHeadSize;
    const kvOffset = kvHead * projection.attentionHeadSize;
    let maxScore = -Infinity;
    for (let s = attentionStart; s < seq; s += 1) {
      let dot = Math.fround(0);
      const kOffset = s * slots.k.stride + kvOffset;
      for (let row = 0; row < projection.attentionHeadSize; row += 1) {
        dot = Math.fround(dot + Math.fround(q[qOffset + row] * kCache[kOffset + row]));
      }
      const score = Math.fround(dot * scale);
      scores[s - attentionStart] = score;
      if (score > maxScore) maxScore = score;
    }
    let denom = Math.fround(0);
    for (let s = attentionStart; s < seq; s += 1) {
      const weight = Math.fround(Math.exp(Math.fround(scores[s - attentionStart] - maxScore)));
      denom = Math.fround(denom + weight);
      const vOffset = s * slots.v.stride + kvOffset;
      for (let row = 0; row < projection.attentionHeadSize; row += 1) {
        attention[qOffset + row] = Math.fround(attention[qOffset + row] + Math.fround(weight * vCache[vOffset + row]));
      }
    }
    const invDenom = denom > 0 ? Math.fround(1 / denom) : Math.fround(0);
    for (let row = 0; row < projection.attentionHeadSize; row += 1) {
      attention[qOffset + row] = Math.fround(attention[qOffset + row] * invDenom);
    }
  }
  const attnProjected = matVecInto(scratch.attnProjected, oProj, projection.oProj.rows, projection.oProj.cols, attention);
  addOptionalVectorBiasInto(attnProjected, projection.oBias, projection.oProj.rows, "mock LLaMA O projection bias");
  const afterAttention = scratch.afterAttention;
  for (let row = 0; row < projection.hiddenSize; row += 1) {
    afterAttention[row] = Math.fround(hidden[row] + attnProjected[row]);
  }
  let logitsLength = 0;
  if (outputLogits) {
    const finalHidden = rmsNormVectorInto(scratch.finalHidden, afterAttention, finalNorm, projection.epsilon, projection.hiddenSize);
    matVecInto(scratch.logits, lmHead, projection.vocabSize, projection.hiddenSize, finalHidden);
    addOptionalVectorBiasInto(scratch.logits, projection.lmHeadBias, projection.vocabSize, "mock LLaMA attention projection lm_head bias");
    logitsLength = projection.vocabSize;
  }
  const result = scratch.result;
  result.afterAttention = afterAttention;
  result.afterAttentionLength = projection.hiddenSize;
  result.attention = attention;
  result.attentionLength = projection.qProj.rows;
  result.k = k;
  result.kLength = projection.kProj.rows;
  result.logits = scratch.logits;
  result.logitsLength = logitsLength;
  result.v = v;
  result.vLength = projection.vProj.rows;
  return result;
}

function llamaAttentionProjectionMock(context, projection, slots, options = {}) {
  return llamaAttentionProjectionMockInto(createLlamaAttentionProjectionMockScratch(), context, projection, slots, options);
}

function llamaBlockProjectionMockInto(scratch, context, projection, slots, options = {}) {
  const outputLogits = options.outputLogits ?? true;
  prepareLlamaBlockProjectionMockScratch(scratch, projection, outputLogits);
  const attentionOptions = scratch.attentionOptions;
  attentionOptions.outputLogits = false;
  attentionOptions.position = options.position ?? context.stepParams.startPosition;
  attentionOptions.seq = options.seq ?? context.stepParams.endPosition;
  attentionOptions.token = options.token ?? llamaLastToken(context.execution);
  const attention = llamaAttentionProjectionMockInto(scratch.attention, context, projection, slots, attentionOptions);
  const postNorm = mockFloat32View(
    projection.postNorm.descriptor,
    projection.postNorm.elementCount,
    "mock LLaMA block projection post-attention norm",
  );
  const gateProj = mockFloat32View(
    projection.gateProj.descriptor,
    projection.gateProj.elementCount,
    "mock LLaMA block projection gate weight",
  );
  const upProj = mockFloat32View(
    projection.upProj.descriptor,
    projection.upProj.elementCount,
    "mock LLaMA block projection up weight",
  );
  const downProj = mockFloat32View(
    projection.downProj.descriptor,
    projection.downProj.elementCount,
    "mock LLaMA block projection down weight",
  );
  const finalNorm = mockFloat32View(
    projection.norm.descriptor,
    projection.norm.elementCount,
    "mock LLaMA block projection final norm",
  );
  const lmHead = mockFloat32View(
    projection.lmHead.descriptor,
    projection.lmHead.elementCount,
    "mock LLaMA block projection lm_head",
  );
  const normedFfn = rmsNormVectorInto(scratch.normedFfn, attention.afterAttention, postNorm, projection.epsilon, projection.hiddenSize);
  const gate = matVecInto(scratch.gate, gateProj, projection.gateProj.rows, projection.gateProj.cols, normedFfn);
  addOptionalVectorBiasInto(gate, projection.gateBias, projection.gateProj.rows, "mock LLaMA gate projection bias");
  const up = matVecInto(scratch.up, upProj, projection.upProj.rows, projection.upProj.cols, normedFfn);
  addOptionalVectorBiasInto(up, projection.upBias, projection.upProj.rows, "mock LLaMA up projection bias");
  const ffnInput = scratch.ffnInput;
  for (let row = 0; row < projection.ffnSize; row += 1) {
    ffnInput[row] = Math.fround(silu(gate[row]) * up[row]);
  }
  const ffn = matVecInto(scratch.ffn, downProj, projection.downProj.rows, projection.downProj.cols, ffnInput);
  addOptionalVectorBiasInto(ffn, projection.downBias, projection.downProj.rows, "mock LLaMA down projection bias");
  const blockHidden = scratch.blockHidden;
  for (let row = 0; row < projection.hiddenSize; row += 1) {
    blockHidden[row] = Math.fround(attention.afterAttention[row] + ffn[row]);
  }
  let logitsLength = 0;
  if (outputLogits) {
    const finalHidden = rmsNormVectorInto(scratch.finalHidden, blockHidden, finalNorm, projection.epsilon, projection.hiddenSize);
    matVecInto(scratch.logits, lmHead, projection.vocabSize, projection.hiddenSize, finalHidden);
    addOptionalVectorBiasInto(scratch.logits, projection.lmHeadBias, projection.vocabSize, "mock LLaMA block projection lm_head bias");
    logitsLength = projection.vocabSize;
  }
  const result = scratch.result;
  result.blockHidden = blockHidden;
  result.blockHiddenLength = projection.hiddenSize;
  result.k = attention.k;
  result.kLength = attention.kLength;
  result.logits = scratch.logits;
  result.logitsLength = logitsLength;
  result.v = attention.v;
  result.vLength = attention.vLength;
  return result;
}

function llamaBlockProjectionMock(context, projection, slots, options = {}) {
  return llamaBlockProjectionMockInto(createLlamaBlockProjectionMockScratch(), context, projection, slots, options);
}

function writeLlamaTokenWindowPatternKvEvidence(target, backendDispatchCount, commandCount, fallbackOpCount, kvWriteCount) {
  target.backendDispatchCount = backendDispatchCount;
  target.commandCount = commandCount;
  target.fallbackOpCount = fallbackOpCount;
  target.kvWriteCount = kvWriteCount;
  return target;
}

function writeIncrementingFloat32Pattern(target, base, length) {
  for (let i = 0; i < length; i += 1) target[i] = base + i;
  return target;
}

export class WasmWebGpuLlamaTokenWindowPatternExecutor {
  constructor(options = {}) {
    this.record = typeof options.record === "function" ? options.record : null;
    this.pipeline = null;
    this.pipelineDevice = null;
    this.bindGroups = [];
    this.paramsBuffers = [];
    this.paramsBytes = new ArrayBuffer(32);
    this.paramsView = new DataView(this.paramsBytes);
    this.kvEvidenceScratch = {
      backendDispatchCount: 0,
      commandCount: 0,
      fallbackOpCount: 0,
      kvWriteCount: 0,
    };
    this.logitsPatternScratch = new Float32Array(0);
    this.kvPatternScratch = new Float32Array(0);
  }

  executeTokensSync(context) {
    const evidence = this.writeKvCache(context);
    let kvWriteCount = evidence.kvWriteCount;
    let backendDispatchCount = evidence.backendDispatchCount;
    let fallbackOpCount = evidence.fallbackOpCount;
    let commandCount = evidence.commandCount;
    if (context.execution.outputPolicy === 1) {
      if (context.execution.outputPtr !== 0 || !context.session.device.canCreateGpuBuffers()) {
        this.writeFallbackLogits(context);
        if (context.execution.outputPtr === 0) {
          fallbackOpCount += 1;
          commandCount += 1;
        }
      } else {
        const logitsDispatchCount = this.submitGpuPatternWrite(
          context.session.device,
          context.session.bindings.output,
          context.session.vocabSize,
          0,
          context.session.vocabSize * Float32Array.BYTES_PER_ELEMENT,
          llamaTokenSum(context.execution) * 10 + context.stepParams.startPosition,
          "LLaMA logits",
        );
        backendDispatchCount += logitsDispatchCount;
        commandCount += logitsDispatchCount;
      }
    }
    this.record?.({
      backendDispatchCount,
      commandCount,
      endPosition: context.stepParams.endPosition,
      fallbackOpCount,
      kvWriteCount,
      logitsLength: context.stepParams.logitsLength,
      modelBindingKind: context.session.modelBindingKind,
      modelHandle: context.session.modelHandle,
      modelManifestHash: context.session.modelManifest?.hash ?? 0n,
      modelResourceRoles: context.session.modelResources?.map((entry) => entry.role) ?? [],
      modelTableDescriptorHash: context.session.modelTable?.descriptorHash ?? 0n,
      modelTableHash: context.session.modelTable?.hash ?? 0n,
      outputKind: context.stepParams.outputKind,
      outputPolicy: context.execution.outputPolicy,
      position: context.stepParams.startPosition,
      tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
      tableHash: context.session.table?.hash ?? 0n,
      tableRoles: context.session.table?.descriptors?.map((entry) => entry.role) ?? [],
      tokenCount: context.stepParams.tokenCount,
      tokens: llamaExecutionTokensArray(context.execution),
    }, context);
    return writeLlamaTokenExecutorResult(
      context.resultScratch ?? {},
      context,
      context.runtime.status.ok,
      context.execution.outputPolicy === 1 ? context.session.vocabSize : 0,
      backendDispatchCount,
      fallbackOpCount,
      commandCount,
    );
  }

  writeKvCache(context) {
    if (!context.session.device.canCreateGpuBuffers()) {
      const kvWriteCount = this.writeFallbackKvCache(context);
      return writeLlamaTokenWindowPatternKvEvidence(this.kvEvidenceScratch, 0, kvWriteCount, kvWriteCount, kvWriteCount);
    }
    const backendDispatchCount = this.submitGpuKvPatternWrites(context.session.device, context.session, context.stepParams);
    return writeLlamaTokenWindowPatternKvEvidence(
      this.kvEvidenceScratch,
      backendDispatchCount,
      backendDispatchCount,
      0,
      backendDispatchCount,
    );
  }

  writeFallbackLogits(context) {
    const length = context.session.vocabSize;
    this.logitsPatternScratch = ensureFloat32ScratchCapacity(this.logitsPatternScratch, length);
    writeIncrementingFloat32Pattern(
      this.logitsPatternScratch,
      llamaTokenSum(context.execution) * 10 + context.stepParams.startPosition,
      length,
    );
    writeLlamaLogits(context, this.logitsPatternScratch, length);
  }

  writeFallbackKvCache(context) {
    const kvCache = context.session.bindings?.kvCache ?? [];
    const stepParams = context.stepParams;
    const contextLength = stepParams.contextLength;
    if (!Number.isSafeInteger(contextLength) || contextLength <= 0) {
      throw new Error(`invalid LLaMA StepParams context length: ${contextLength}`);
    }
    let kvWriteCount = 0;
    for (let layer = 0; layer < kvCache.length; layer += 1) {
      const entry = kvCache[layer];
      kvWriteCount += this.writeFallbackKvCacheSlot(context.session, entry?.k, "k", layer, stepParams);
      kvWriteCount += this.writeFallbackKvCacheSlot(context.session, entry?.v, "v", layer, stepParams);
    }
    return kvWriteCount;
  }

  writeFallbackKvCacheSlot(session, descriptor, kind, layer, stepParams) {
    if (!descriptor || typeof descriptor !== "object") {
      throw new Error(`LLaMA ${kind.toUpperCase()} cache ${layer} binding is missing`);
    }
    if (!Number.isSafeInteger(descriptor.byteLength) || descriptor.byteLength <= 0) {
      throw new Error(`LLaMA ${kind.toUpperCase()} cache ${layer} has invalid byteLength: ${descriptor.byteLength}`);
    }
    if (descriptor.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error(`LLaMA ${kind.toUpperCase()} cache ${layer} byteLength must be a Float32 multiple`);
    }
    const elementCount = descriptor.byteLength / Float32Array.BYTES_PER_ELEMENT;
    if (elementCount % stepParams.contextLength !== 0) {
      throw new Error(`LLaMA ${kind.toUpperCase()} cache ${layer} element count is not divisible by context length`);
    }
    const stride = elementCount / stepParams.contextLength;
    const elementLength = stepParams.tokenCount * stride;
    const byteOffset = stepParams.startPosition * stride * Float32Array.BYTES_PER_ELEMENT;
    this.kvPatternScratch = ensureFloat32ScratchCapacity(this.kvPatternScratch, elementLength);
    writeIncrementingFloat32Pattern(
      this.kvPatternScratch,
      (kind === "k" ? 1000 : 2000) + layer * 100 + stepParams.startPosition * 10,
      elementLength,
    );
    session.device.writeFloat32(descriptor, this.kvPatternScratch, byteOffset, elementLength);
    return 1;
  }

  ensureGpuPipeline(device) {
    if (!device.canCreateGpuBuffers()) {
      throw new Error("browser WebGPU device is unavailable");
    }
    if (this.pipeline && this.pipelineDevice === device.device) return this.pipeline;
    if (this.pipelineDevice !== null && this.pipelineDevice !== device.device) {
      this.bindGroups = [];
    }
    const gpuDevice = device.device;
    const shader = gpuDevice.createShaderModule({
      label: "zgml.wasm.webgpu.llama-token-window-pattern",
      code: `
struct Params {
  len: u32,
  offset: u32,
  base: f32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) {
    return;
  }
  output[params.offset + i] = params.base + f32(i);
}
`,
    });
    this.pipeline = gpuDevice.createComputePipeline({
      label: "zgml.wasm.webgpu.llama-token-window-pattern",
      layout: "auto",
      compute: {
        module: shader,
        entryPoint: "main",
      },
    });
    this.pipelineDevice = gpuDevice;
    return this.pipeline;
  }

  ensureParamsBuffer(device, index) {
    const existing = this.paramsBuffers[index];
    if (existing && existing.device === device.device) return existing.buffer;
    if (existing?.buffer && typeof existing.buffer.destroy === "function") {
      existing.buffer.destroy();
    }
    const gpuDevice = device.device;
    const usage = gpuBufferUsageFlags();
    const buffer = gpuDevice.createBuffer({
      label: "zgml.wasm.webgpu.llama-token-window-pattern.params",
      size: this.paramsBytes.byteLength,
      usage: usage.uniform | usage.copyDst,
    });
    this.paramsBuffers[index] = {
      buffer,
      device: gpuDevice,
    };
    return buffer;
  }

  writeParamsBuffer(device, index, elementLength, elementOffset, base) {
    const buffer = this.ensureParamsBuffer(device, index);
    const view = this.paramsView;
    view.setUint32(0, elementLength, true);
    view.setUint32(4, elementOffset, true);
    view.setFloat32(8, base, true);
    view.setUint32(12, 0, true);
    device.device.queue.writeBuffer(buffer, 0, this.paramsBytes);
    return buffer;
  }

  dispatchGpuPatternWrite(device, pipeline, pass, index, descriptor, bindingByteLength, elementLength, elementOffset, base, label) {
    if (!Number.isSafeInteger(elementOffset) || elementOffset < 0) {
      throw new Error(`invalid LLaMA token-window pattern element offset: ${elementOffset}`);
    }
    if (!Number.isSafeInteger(elementLength) || elementLength < 0) {
      throw new Error(`invalid LLaMA token-window pattern element length: ${elementLength}`);
    }
    if (!Number.isSafeInteger(bindingByteLength) || bindingByteLength < elementLength * Float32Array.BYTES_PER_ELEMENT) {
      throw new Error(`invalid LLaMA token-window pattern binding byte length: ${bindingByteLength}`);
    }
    const bindingElements = Math.floor(bindingByteLength / Float32Array.BYTES_PER_ELEMENT);
    if (elementOffset + elementLength > bindingElements) {
      throw new Error(`LLaMA token-window pattern write is out of range: ${elementOffset} + ${elementLength} > ${bindingElements}`);
    }
    const output = storageBufferBinding(device.device, descriptor, label, bindingByteLength);
    const params = this.writeParamsBuffer(device, index, elementLength, elementOffset, base);
    pass.setBindGroup(0, this.ensurePatternBindGroup(device, pipeline, index, output, params, label));
    pass.dispatchWorkgroups(Math.ceil(elementLength / 64));
    return 1;
  }

  ensurePatternBindGroup(device, pipeline, index, output, params, label) {
    const gpuDevice = device.device;
    const existing = this.bindGroups[index];
    if (
      existing &&
      existing.device === gpuDevice &&
      existing.outputBuffer === output.buffer &&
      existing.outputOffset === output.offset &&
      existing.outputSize === output.size &&
      existing.paramsBuffer === params
    ) {
      return existing.bindGroup;
    }
    const bindGroup = gpuDevice.createBindGroup({
      label: `zgml.wasm.webgpu.llama-token-window-pattern.${label}`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: output },
        { binding: 1, resource: { buffer: params } },
      ],
    });
    this.bindGroups[index] = {
      bindGroup,
      device: gpuDevice,
      outputBuffer: output.buffer,
      outputOffset: output.offset,
      outputSize: output.size,
      paramsBuffer: params,
    };
    return bindGroup;
  }

  submitGpuKvPatternWrites(device, session, stepParams) {
    const kvCache = session.bindings?.kvCache ?? [];
    if (kvCache.length === 0) return 0;
    const contextLength = stepParams.contextLength;
    if (!Number.isSafeInteger(contextLength) || contextLength <= 0) {
      throw new Error(`invalid LLaMA StepParams context length: ${contextLength}`);
    }
    const gpuDevice = device.device;
    const pipeline = this.ensureGpuPipeline(device);
    const encoder = gpuDevice.createCommandEncoder({ label: "zgml.wasm.webgpu.llama-token-window-pattern" });
    const pass = encoder.beginComputePass({ label: "zgml.wasm.webgpu.llama-token-window-pattern" });
    pass.setPipeline(pipeline);
    let dispatchCount = 0;
    for (let layer = 0; layer < kvCache.length; layer += 1) {
      const entry = kvCache[layer];
      for (let kindIndex = 0; kindIndex < 2; kindIndex += 1) {
        const kind = kindIndex === 0 ? "k" : "v";
        const descriptor = entry?.[kind];
        if (!descriptor || typeof descriptor !== "object") {
          throw new Error(`LLaMA ${kind.toUpperCase()} cache ${layer} needs a bound resource descriptor`);
        }
        const elementCount = Math.floor(descriptor.byteLength / Float32Array.BYTES_PER_ELEMENT);
        if (elementCount % contextLength !== 0) {
          throw new Error(`LLaMA ${kind.toUpperCase()} cache ${layer} element count is not divisible by context length`);
        }
        const stride = elementCount / contextLength;
        const elementOffset = stepParams.startPosition * stride;
        const elementLength = stepParams.tokenCount * stride;
        dispatchCount += this.dispatchGpuPatternWrite(
          device,
          pipeline,
          pass,
          dispatchCount,
          descriptor,
          descriptor.byteLength,
          elementLength,
          elementOffset,
          (kind === "k" ? 1000 : 2000) + layer * 100 + stepParams.startPosition * 10,
          `LLaMA ${kind.toUpperCase()} cache ${layer}`,
        );
      }
    }
    pass.end();
    gpuDevice.queue.submit([encoder.finish()]);
    return dispatchCount;
  }

  submitGpuPatternWrite(device, descriptor, elementLength, elementOffset, bindingByteLength, base, label) {
    const gpuDevice = device.device;
    const pipeline = this.ensureGpuPipeline(device);
    const encoder = gpuDevice.createCommandEncoder({ label: "zgml.wasm.webgpu.llama-token-window-pattern" });
    const pass = encoder.beginComputePass({ label: "zgml.wasm.webgpu.llama-token-window-pattern" });
    pass.setPipeline(pipeline);
    const dispatchCount = this.dispatchGpuPatternWrite(
      device,
      pipeline,
      pass,
      0,
      descriptor,
      bindingByteLength,
      elementLength,
      elementOffset,
      base,
      label,
    );
    pass.end();
    gpuDevice.queue.submit([encoder.finish()]);
    return dispatchCount;
  }

  submitGpuPatternWrites(device, writes) {
    if (writes.length === 0) return 0;
    const gpuDevice = device.device;
    const pipeline = this.ensureGpuPipeline(device);
    const encoder = gpuDevice.createCommandEncoder({ label: "zgml.wasm.webgpu.llama-token-window-pattern" });
    const pass = encoder.beginComputePass({ label: "zgml.wasm.webgpu.llama-token-window-pattern" });
    pass.setPipeline(pipeline);
    let dispatchCount = 0;
    for (let index = 0; index < writes.length; index += 1) {
      const write = writes[index];
      const elementOffset = write.elementOffset ?? 0;
      const bindingByteLength = write.bindingByteLength ?? ((elementOffset + write.elementLength) * Float32Array.BYTES_PER_ELEMENT);
      dispatchCount += this.dispatchGpuPatternWrite(
        device,
        pipeline,
        pass,
        index,
        write.descriptor,
        bindingByteLength,
        write.elementLength,
        elementOffset,
        write.base,
        write.label,
      );
    }
    pass.end();
    gpuDevice.queue.submit([encoder.finish()]);
    return dispatchCount;
  }

  destroy() {
    for (const entry of this.paramsBuffers.splice(0)) {
      if (entry?.buffer && typeof entry.buffer.destroy === "function") entry.buffer.destroy();
    }
    this.bindGroups = [];
    this.logitsPatternScratch = new Float32Array(0);
    this.kvPatternScratch = new Float32Array(0);
    this.pipeline = null;
    this.pipelineDevice = null;
  }
}

export class WasmWebGpuLlamaEmbeddingProjectionExecutor {
  constructor(options = {}) {
    this.embeddingRole = options.embeddingRole ?? "model.embed_tokens.weight";
    this.lmHeadRole = options.lmHeadRole ?? "lm_head.weight";
    this.lmHeadBiasRole = options.lmHeadBiasRole ?? "lm_head.bias";
    this.record = typeof options.record === "function" ? options.record : null;
    this.pipeline = null;
    this.pipelineDevice = null;
    this.paramsBuffer = null;
    this.paramsWords = new Uint32Array(4);
    this.preparedSessions = new WeakMap();
    this.mockLogitsScratch = new Float32Array(0);
    this.bindGroupDeps = [null, null, null];
  }

  prepareSession(session) {
    return prepareLlamaProjectionSession(this.preparedSessions, session, () => llamaEmbeddingProjectionResources(session, {
      embeddingRole: this.embeddingRole,
      lmHeadBiasRole: this.lmHeadBiasRole,
      lmHeadRole: this.lmHeadRole,
    }));
  }

  projectionForSession(session) {
    return this.preparedSessions.get(session) ?? this.prepareSession(session);
  }

  requiredGpuStorageBufferCount() {
    return 3;
  }

  executeTokensSync(context) {
    const projection = this.projectionForSession(context.session);
    let backendDispatchCount = 0;
    let fallbackOpCount = 0;
    let commandCount = 0;
    let status = context.runtime.status.ok;
    let outputLength = 0;
    if (context.execution.outputPolicy === 1) {
      outputLength = context.session.vocabSize;
      if (context.session.device.canCreateGpuBuffers()) {
        if (context.execution.outputPtr !== 0) {
          status = context.runtime.status.unsupported;
          outputLength = 0;
        } else {
          backendDispatchCount = this.submitGpuProjection(context, projection);
          commandCount = backendDispatchCount;
        }
      } else {
        this.mockLogitsScratch = ensureFloat32ScratchCapacity(this.mockLogitsScratch, projection.vocabSize);
        writeLlamaLogits(
          context,
          llamaEmbeddingProjectionMockLogitsInto(this.mockLogitsScratch, context, projection),
          projection.vocabSize,
        );
        fallbackOpCount = 1;
        commandCount = 1;
      }
    }
    this.record?.({
      backendDispatchCount,
      commandCount,
      embeddingRole: projection.embedding.role,
      endPosition: context.stepParams.endPosition,
      fallbackOpCount,
      hiddenSize: projection.hiddenSize,
      lmHeadBiasRole: projection.lmHeadBias?.role ?? null,
      lmHeadRole: projection.lmHead.role,
      modelBindingKind: context.session.modelBindingKind,
      modelHandle: context.session.modelHandle,
      modelManifestHash: context.session.modelManifest?.hash ?? 0n,
      modelTableDescriptorHash: context.session.modelTable?.descriptorHash ?? 0n,
      modelTableHash: context.session.modelTable?.hash ?? 0n,
      outputKind: context.stepParams.outputKind,
      outputLength,
      outputPolicy: context.execution.outputPolicy,
      position: context.stepParams.startPosition,
      status,
      tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
      tableHash: context.session.table?.hash ?? 0n,
      token: llamaLastToken(context.execution),
      tokenCount: context.stepParams.tokenCount,
      tokens: llamaExecutionTokensArray(context.execution),
      vocabSize: projection.vocabSize,
    }, context);
    return writeLlamaTokenExecutorResult(
      context.resultScratch ?? {},
      context,
      status,
      outputLength,
      backendDispatchCount,
      fallbackOpCount,
      commandCount,
    );
  }

  ensureGpuPipeline(device) {
    if (!device.canCreateGpuBuffers()) {
      throw new Error("browser WebGPU device is unavailable");
    }
    if (this.pipeline && this.pipelineDevice === device.device) return this.pipeline;
    if (this.paramsBuffer && typeof this.paramsBuffer.destroy === "function") {
      this.paramsBuffer.destroy();
    }
    const gpuDevice = device.device;
    const shader = gpuDevice.createShaderModule({
      label: "zgml.wasm.webgpu.llama-embedding-projection",
      code: `
struct Params {
  token: u32,
  vocab: u32,
  hidden: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> embeddings: array<f32>;
@group(0) @binding(1) var<storage, read> lm_head: array<f32>;
@group(0) @binding(2) var<storage, read> lm_head_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.vocab) {
    return;
  }
  var acc = 0.0;
  let embedding_offset = params.token * params.hidden;
  let lm_head_offset = row * params.hidden;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    acc = acc + lm_head[lm_head_offset + col] * embeddings[embedding_offset + col];
  }
  if (params._pad != 0u) {
    acc = acc + lm_head_bias[row];
  }
  logits[row] = acc;
}
`,
    });
    this.pipeline = gpuDevice.createComputePipeline({
      label: "zgml.wasm.webgpu.llama-embedding-projection",
      layout: "auto",
      compute: {
        module: shader,
        entryPoint: "main",
      },
    });
    const usage = gpuBufferUsageFlags();
    this.paramsBuffer = gpuDevice.createBuffer({
      label: "zgml.wasm.webgpu.llama-embedding-projection.params",
      size: 16,
      usage: usage.uniform | usage.copyDst,
    });
    this.pipelineDevice = gpuDevice;
    return this.pipeline;
  }

  submitGpuProjection(context, projection) {
    const device = context.session.device;
    const gpuDevice = device.device;
    const pipeline = this.ensureGpuPipeline(device);
    const params = this.paramsWords;
    params[0] = llamaLastToken(context.execution);
    params[1] = projection.vocabSize;
    params[2] = projection.hiddenSize;
    params[3] = projection.lmHeadBias === null ? 0 : 1;
    gpuDevice.queue.writeBuffer(this.paramsBuffer, 0, params.buffer, params.byteOffset, params.byteLength);
    const bindGroupDeps = this.bindGroupDeps;
    bindGroupDeps[0] = this.paramsBuffer;
    bindGroupDeps[1] = context.session.bindings.output;
    bindGroupDeps[2] = projection.lmHeadBias?.descriptor ?? projection.lmHead.descriptor;
    const bindGroup = projectionGpuBindGroup(
      projection,
      "embedding",
      device,
      pipeline,
      bindGroupDeps,
      (targetDevice) => targetDevice.createBindGroup({
        label: "zgml.wasm.webgpu.llama-embedding-projection",
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: storageBufferBinding(
              targetDevice,
              projection.embedding.descriptor,
              "LLaMA embedding projection embeddings",
              projection.embedding.byteLength,
            ),
          },
          {
            binding: 1,
            resource: storageBufferBinding(
              targetDevice,
              projection.lmHead.descriptor,
              "LLaMA embedding projection lm_head",
              projection.lmHead.byteLength,
            ),
          },
          {
            binding: 2,
            resource: storageBufferBinding(
              targetDevice,
              projection.lmHeadBias?.descriptor ?? projection.lmHead.descriptor,
              projection.lmHeadBias === null ? "LLaMA embedding projection lm_head bias dummy" : "LLaMA embedding projection lm_head bias",
              projection.lmHeadBias?.byteLength ?? projection.lmHead.byteLength,
            ),
          },
          {
            binding: 3,
            resource: storageBufferBinding(
              targetDevice,
              context.session.bindings.output,
              "LLaMA embedding projection logits",
              projection.vocabSize * Float32Array.BYTES_PER_ELEMENT,
            ),
          },
          { binding: 4, resource: { buffer: this.paramsBuffer } },
        ],
      }),
    );
    const encoder = gpuDevice.createCommandEncoder({ label: "zgml.wasm.webgpu.llama-embedding-projection" });
    const pass = encoder.beginComputePass({ label: "zgml.wasm.webgpu.llama-embedding-projection" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(projection.vocabSize / 64));
    pass.end();
    gpuDevice.queue.submit([encoder.finish()]);
    return 1;
  }

  destroy() {
    if (this.paramsBuffer && typeof this.paramsBuffer.destroy === "function") {
      this.paramsBuffer.destroy();
    }
    this.paramsBuffer = null;
    this.pipeline = null;
    this.pipelineDevice = null;
    this.preparedSessions = new WeakMap();
    this.mockLogitsScratch = new Float32Array(0);
    this.bindGroupDeps.fill(null);
  }
}

export class WasmWebGpuLlamaRmsNormProjectionExecutor {
  constructor(options = {}) {
    this.embeddingRole = options.embeddingRole ?? "model.embed_tokens.weight";
    this.lmHeadRole = options.lmHeadRole ?? "lm_head.weight";
    this.lmHeadBiasRole = options.lmHeadBiasRole ?? "lm_head.bias";
    this.normRole = options.normRole ?? "model.norm.weight";
    this.epsilon = options.epsilon ?? options.rmsNormEpsilon ?? options.rmsNormEps ?? options.rms_norm_eps ?? 1e-5;
    this.record = typeof options.record === "function" ? options.record : null;
    this.pipeline = null;
    this.pipelineDevice = null;
    this.paramsBuffer = null;
    this.paramsBytes = new ArrayBuffer(32);
    this.paramsView = new DataView(this.paramsBytes);
    this.preparedSessions = new WeakMap();
    this.mockLogitsScratch = new Float32Array(0);
    this.bindGroupDeps = [null, null, null];
  }

  prepareSession(session) {
    return prepareLlamaProjectionSession(this.preparedSessions, session, () => llamaRmsNormProjectionResources(session, {
      embeddingRole: this.embeddingRole,
      epsilon: this.epsilon,
      lmHeadBiasRole: this.lmHeadBiasRole,
      lmHeadRole: this.lmHeadRole,
      normRole: this.normRole,
    }));
  }

  projectionForSession(session) {
    return this.preparedSessions.get(session) ?? this.prepareSession(session);
  }

  requiredGpuStorageBufferCount() {
    return 3;
  }

  executeTokensSync(context) {
    const projection = this.projectionForSession(context.session);
    let backendDispatchCount = 0;
    let fallbackOpCount = 0;
    let commandCount = 0;
    let status = context.runtime.status.ok;
    let outputLength = 0;
    if (context.execution.outputPolicy === 1) {
      outputLength = context.session.vocabSize;
      if (context.session.device.canCreateGpuBuffers()) {
        if (context.execution.outputPtr !== 0) {
          status = context.runtime.status.unsupported;
          outputLength = 0;
        } else {
          backendDispatchCount = this.submitGpuProjection(context, projection);
          commandCount = backendDispatchCount;
        }
      } else {
        this.mockLogitsScratch = ensureFloat32ScratchCapacity(this.mockLogitsScratch, projection.vocabSize);
        writeLlamaLogits(
          context,
          llamaRmsNormProjectionMockLogitsInto(this.mockLogitsScratch, context, projection),
          projection.vocabSize,
        );
        fallbackOpCount = 1;
        commandCount = 1;
      }
    }
    this.record?.({
      backendDispatchCount,
      commandCount,
      embeddingRole: projection.embedding.role,
      endPosition: context.stepParams.endPosition,
      epsilon: projection.epsilon,
      fallbackOpCount,
      hiddenSize: projection.hiddenSize,
      lmHeadBiasRole: projection.lmHeadBias?.role ?? null,
      lmHeadRole: projection.lmHead.role,
      modelBindingKind: context.session.modelBindingKind,
      modelHandle: context.session.modelHandle,
      modelManifestHash: context.session.modelManifest?.hash ?? 0n,
      modelTableDescriptorHash: context.session.modelTable?.descriptorHash ?? 0n,
      modelTableHash: context.session.modelTable?.hash ?? 0n,
      normRole: projection.norm.role,
      outputKind: context.stepParams.outputKind,
      outputLength,
      outputPolicy: context.execution.outputPolicy,
      position: context.stepParams.startPosition,
      status,
      tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
      tableHash: context.session.table?.hash ?? 0n,
      token: llamaLastToken(context.execution),
      tokenCount: context.stepParams.tokenCount,
      tokens: llamaExecutionTokensArray(context.execution),
      vocabSize: projection.vocabSize,
    }, context);
    return writeLlamaTokenExecutorResult(
      context.resultScratch ?? {},
      context,
      status,
      outputLength,
      backendDispatchCount,
      fallbackOpCount,
      commandCount,
    );
  }

  ensureGpuPipeline(device) {
    if (!device.canCreateGpuBuffers()) {
      throw new Error("browser WebGPU device is unavailable");
    }
    if (this.pipeline && this.pipelineDevice === device.device) return this.pipeline;
    if (this.paramsBuffer && typeof this.paramsBuffer.destroy === "function") {
      this.paramsBuffer.destroy();
    }
    const gpuDevice = device.device;
    const shader = gpuDevice.createShaderModule({
      label: "zgml.wasm.webgpu.llama-rmsnorm-projection",
      code: `
struct Params {
  token: u32,
  vocab: u32,
  hidden: u32,
  epsilon: f32,
  bias_mask: u32,
};

@group(0) @binding(0) var<storage, read> embeddings: array<f32>;
@group(0) @binding(1) var<storage, read> norm: array<f32>;
@group(0) @binding(2) var<storage, read> lm_head: array<f32>;
@group(0) @binding(3) var<storage, read> lm_head_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn inv_rms() -> f32 {
  let embedding_offset = params.token * params.hidden;
  var sum = 0.0;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    let value = embeddings[embedding_offset + col];
    sum = sum + value * value;
  }
  return inverseSqrt(sum / f32(params.hidden) + params.epsilon);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.vocab) {
    return;
  }
  let embedding_offset = params.token * params.hidden;
  let lm_head_offset = row * params.hidden;
  let scale = inv_rms();
  var acc = 0.0;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    let hidden = embeddings[embedding_offset + col] * norm[col] * scale;
    acc = acc + lm_head[lm_head_offset + col] * hidden;
  }
  if ((params.bias_mask & 1u) != 0u) {
    acc = acc + lm_head_bias[row];
  }
  logits[row] = acc;
}
`,
    });
    this.pipeline = gpuDevice.createComputePipeline({
      label: "zgml.wasm.webgpu.llama-rmsnorm-projection",
      layout: "auto",
      compute: {
        module: shader,
        entryPoint: "main",
      },
    });
    const usage = gpuBufferUsageFlags();
    this.paramsBuffer = gpuDevice.createBuffer({
      label: "zgml.wasm.webgpu.llama-rmsnorm-projection.params",
      size: 32,
      usage: usage.uniform | usage.copyDst,
    });
    this.pipelineDevice = gpuDevice;
    return this.pipeline;
  }

  submitGpuProjection(context, projection) {
    const device = context.session.device;
    const gpuDevice = device.device;
    const pipeline = this.ensureGpuPipeline(device);
    const params = this.paramsBytes;
    const view = this.paramsView;
    view.setUint32(0, llamaLastToken(context.execution), true);
    view.setUint32(4, projection.vocabSize, true);
    view.setUint32(8, projection.hiddenSize, true);
    view.setFloat32(12, projection.epsilon, true);
    view.setUint32(16, projection.lmHeadBias === null ? 0 : 1, true);
    gpuDevice.queue.writeBuffer(this.paramsBuffer, 0, params);
    const bindGroupDeps = this.bindGroupDeps;
    bindGroupDeps[0] = this.paramsBuffer;
    bindGroupDeps[1] = context.session.bindings.output;
    bindGroupDeps[2] = projection.lmHeadBias?.descriptor ?? projection.lmHead.descriptor;
    const bindGroup = projectionGpuBindGroup(
      projection,
      "rmsnorm",
      device,
      pipeline,
      bindGroupDeps,
      (targetDevice) => targetDevice.createBindGroup({
        label: "zgml.wasm.webgpu.llama-rmsnorm-projection",
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: storageBufferBinding(
              targetDevice,
              projection.embedding.descriptor,
              "LLaMA RMSNorm projection embeddings",
              projection.embedding.byteLength,
            ),
          },
          {
            binding: 1,
            resource: storageBufferBinding(
              targetDevice,
              projection.norm.descriptor,
              "LLaMA RMSNorm projection norm",
              projection.norm.byteLength,
            ),
          },
          {
            binding: 2,
            resource: storageBufferBinding(
              targetDevice,
              projection.lmHead.descriptor,
              "LLaMA RMSNorm projection lm_head",
              projection.lmHead.byteLength,
            ),
          },
          {
            binding: 3,
            resource: storageBufferBinding(
              targetDevice,
              projection.lmHeadBias?.descriptor ?? projection.lmHead.descriptor,
              projection.lmHeadBias === null ? "LLaMA RMSNorm projection lm_head bias dummy" : "LLaMA RMSNorm projection lm_head bias",
              projection.lmHeadBias?.byteLength ?? projection.lmHead.byteLength,
            ),
          },
          {
            binding: 4,
            resource: storageBufferBinding(
              targetDevice,
              context.session.bindings.output,
              "LLaMA RMSNorm projection logits",
              projection.vocabSize * Float32Array.BYTES_PER_ELEMENT,
            ),
          },
          { binding: 5, resource: { buffer: this.paramsBuffer } },
        ],
      }),
    );
    const encoder = gpuDevice.createCommandEncoder({ label: "zgml.wasm.webgpu.llama-rmsnorm-projection" });
    const pass = encoder.beginComputePass({ label: "zgml.wasm.webgpu.llama-rmsnorm-projection" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(projection.vocabSize / 64));
    pass.end();
    gpuDevice.queue.submit([encoder.finish()]);
    return 1;
  }

  destroy() {
    if (this.paramsBuffer && typeof this.paramsBuffer.destroy === "function") {
      this.paramsBuffer.destroy();
    }
    this.paramsBuffer = null;
    this.pipeline = null;
    this.pipelineDevice = null;
    this.preparedSessions = new WeakMap();
    this.mockLogitsScratch = new Float32Array(0);
    this.bindGroupDeps.fill(null);
  }
}

export class WasmWebGpuLlamaActivationLogitsExecutor {
  constructor(options = {}) {
    this.lmHeadRole = options.lmHeadRole ?? "lm_head.weight";
    this.lmHeadBiasRole = options.lmHeadBiasRole ?? "lm_head.bias";
    this.normRole = options.normRole ?? "model.norm.weight";
    this.epsilon = options.epsilon ?? options.rmsNormEpsilon ?? options.rmsNormEps ?? options.rms_norm_eps ?? 1e-5;
    this.record = typeof options.record === "function" ? options.record : null;
    this.pipeline = null;
    this.pipelineDevice = null;
    this.paramsBuffer = null;
    this.paramsBytes = new ArrayBuffer(32);
    this.paramsView = new DataView(this.paramsBytes);
    this.preparedSessions = new WeakMap();
    this.mockLogitsScratch = new Float32Array(0);
    this.bindGroupDeps = [null, null, null, null, null];
  }

  prepareSession(session) {
    return prepareLlamaProjectionSession(this.preparedSessions, session, () => llamaActivationLogitsResources(session, {
      epsilon: this.epsilon,
      lmHeadBiasRole: this.lmHeadBiasRole,
      lmHeadRole: this.lmHeadRole,
      normRole: this.normRole,
    }));
  }

  projectionForSession(session) {
    return this.preparedSessions.get(session) ?? this.prepareSession(session);
  }

  requiredGpuStorageBufferCount() {
    return 5;
  }

  executeTokensSync(context) {
    const projection = this.projectionForSession(context.session);
    let backendDispatchCount = 0;
    let fallbackOpCount = 0;
    let commandCount = 0;
    let status = context.runtime.status.ok;
    let outputLength = 0;
    if (context.execution.outputPolicy === 1) {
      outputLength = context.session.vocabSize;
      if (context.stepParams.tokenCount <= 0) {
        status = context.runtime.status.unsupported;
        outputLength = 0;
      } else if (context.session.device.canCreateGpuBuffers()) {
        if (context.execution.outputPtr !== 0) {
          status = context.runtime.status.unsupported;
          outputLength = 0;
        } else {
          backendDispatchCount = this.submitGpuProjection(context, projection);
          commandCount = backendDispatchCount;
        }
      } else {
        this.mockLogitsScratch = ensureFloat32ScratchCapacity(this.mockLogitsScratch, projection.vocabSize);
        writeLlamaLogits(
          context,
          llamaActivationLogitsMockLogitsInto(this.mockLogitsScratch, context, projection),
          projection.vocabSize,
        );
        fallbackOpCount = 1;
        commandCount = 1;
      }
    }
    this.record?.({
      activationInputByteLength: projection.activation.descriptor.byteLength,
      backendDispatchCount,
      commandCount,
      endPosition: context.stepParams.endPosition,
      epsilon: projection.epsilon,
      fallbackOpCount,
      hiddenSize: projection.hiddenSize,
      lmHeadBiasRole: projection.lmHeadBias?.role ?? null,
      lmHeadRole: projection.lmHead.role,
      modelBindingKind: context.session.modelBindingKind,
      modelHandle: context.session.modelHandle,
      modelManifestHash: context.session.modelManifest?.hash ?? 0n,
      modelTableDescriptorHash: context.session.modelTable?.descriptorHash ?? 0n,
      modelTableHash: context.session.modelTable?.hash ?? 0n,
      normRole: projection.norm.role,
      outputKind: context.stepParams.outputKind,
      outputLength,
      outputPolicy: context.execution.outputPolicy,
      position: context.stepParams.startPosition,
      status,
      tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
      tableHash: context.session.table?.hash ?? 0n,
      token: llamaLastToken(context.execution),
      tokenCount: context.stepParams.tokenCount,
      tokens: llamaExecutionTokensArray(context.execution),
      vocabSize: projection.vocabSize,
    }, context);
    return writeLlamaTokenExecutorResult(
      context.resultScratch ?? {},
      context,
      status,
      outputLength,
      backendDispatchCount,
      fallbackOpCount,
      commandCount,
    );
  }

  ensureGpuPipeline(device) {
    if (!device.canCreateGpuBuffers()) {
      throw new Error("browser WebGPU device is unavailable");
    }
    if (this.pipeline && this.pipelineDevice === device.device) return this.pipeline;
    if (this.paramsBuffer && typeof this.paramsBuffer.destroy === "function") {
      this.paramsBuffer.destroy();
    }
    const gpuDevice = device.device;
    const shader = gpuDevice.createShaderModule({
      label: "zgml.wasm.webgpu.llama-activation-logits",
      code: `
struct Params {
  position: u32,
  vocab: u32,
  hidden: u32,
  epsilon: f32,
  bias_mask: u32,
};

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> norm: array<f32>;
@group(0) @binding(2) var<storage, read> lm_head: array<f32>;
@group(0) @binding(3) var<storage, read> lm_head_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn hidden_value(col: u32) -> f32 {
  return activation[params.position * params.hidden + col];
}

fn inv_rms() -> f32 {
  var sum = 0.0;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    let value = hidden_value(col);
    sum = sum + value * value;
  }
  return inverseSqrt(sum / f32(params.hidden) + params.epsilon);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.vocab) {
    return;
  }
  let lm_head_offset = row * params.hidden;
  let scale = inv_rms();
  var acc = 0.0;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    let hidden = hidden_value(col) * norm[col] * scale;
    acc = acc + lm_head[lm_head_offset + col] * hidden;
  }
  if ((params.bias_mask & 1u) != 0u) {
    acc = acc + lm_head_bias[row];
  }
  logits[row] = acc;
}
`,
    });
    this.pipeline = gpuDevice.createComputePipeline({
      label: "zgml.wasm.webgpu.llama-activation-logits",
      layout: "auto",
      compute: {
        module: shader,
        entryPoint: "main",
      },
    });
    const usage = gpuBufferUsageFlags();
    this.paramsBuffer = gpuDevice.createBuffer({
      label: "zgml.wasm.webgpu.llama-activation-logits.params",
      size: 32,
      usage: usage.uniform | usage.copyDst,
    });
    this.pipelineDevice = gpuDevice;
    return this.pipeline;
  }

  submitGpuProjection(context, projection) {
    const device = context.session.device;
    const gpuDevice = device.device;
    const pipeline = this.ensureGpuPipeline(device);
    const params = this.paramsBytes;
    const view = this.paramsView;
    view.setUint32(0, context.stepParams.endPosition - 1, true);
    view.setUint32(4, projection.vocabSize, true);
    view.setUint32(8, projection.hiddenSize, true);
    view.setFloat32(12, projection.epsilon, true);
    view.setUint32(16, projection.lmHeadBias === null ? 0 : 1, true);
    gpuDevice.queue.writeBuffer(this.paramsBuffer, 0, params);
    const bindGroupDeps = this.bindGroupDeps;
    bindGroupDeps[0] = this.paramsBuffer;
    bindGroupDeps[1] = projection.activation.descriptor;
    bindGroupDeps[2] = context.session.bindings.output;
    bindGroupDeps[3] = projection.lmHeadBias?.descriptor ?? projection.lmHead.descriptor;
    const bindGroup = projectionGpuBindGroup(
      projection,
      "activation-logits",
      device,
      pipeline,
      bindGroupDeps,
      (targetDevice) => targetDevice.createBindGroup({
        label: "zgml.wasm.webgpu.llama-activation-logits",
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: storageBufferBinding(
              targetDevice,
              projection.activation.descriptor,
              "LLaMA activation logits hidden",
              projection.activation.descriptor.byteLength,
            ),
          },
          {
            binding: 1,
            resource: storageBufferBinding(
              targetDevice,
              projection.norm.descriptor,
              "LLaMA activation logits norm",
              projection.norm.byteLength,
            ),
          },
          {
            binding: 2,
            resource: storageBufferBinding(
              targetDevice,
              projection.lmHead.descriptor,
              "LLaMA activation logits lm_head",
              projection.lmHead.byteLength,
            ),
          },
          {
            binding: 3,
            resource: storageBufferBinding(
              targetDevice,
              projection.lmHeadBias?.descriptor ?? projection.lmHead.descriptor,
              projection.lmHeadBias === null ? "LLaMA activation logits lm_head bias dummy" : "LLaMA activation logits lm_head bias",
              projection.lmHeadBias?.byteLength ?? projection.lmHead.byteLength,
            ),
          },
          {
            binding: 4,
            resource: storageBufferBinding(
              targetDevice,
              context.session.bindings.output,
              "LLaMA activation logits output",
              projection.vocabSize * Float32Array.BYTES_PER_ELEMENT,
            ),
          },
          { binding: 5, resource: { buffer: this.paramsBuffer } },
        ],
      }),
    );
    const encoder = gpuDevice.createCommandEncoder({ label: "zgml.wasm.webgpu.llama-activation-logits" });
    const pass = encoder.beginComputePass({ label: "zgml.wasm.webgpu.llama-activation-logits" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(projection.vocabSize / 64));
    pass.end();
    gpuDevice.queue.submit([encoder.finish()]);
    return 1;
  }

  destroy() {
    if (this.paramsBuffer && typeof this.paramsBuffer.destroy === "function") {
      this.paramsBuffer.destroy();
    }
    this.paramsBuffer = null;
    this.pipeline = null;
    this.pipelineDevice = null;
    this.preparedSessions = new WeakMap();
    this.mockLogitsScratch = new Float32Array(0);
    this.bindGroupDeps.fill(null);
  }
}

export class WasmWebGpuLlamaKvProjectionExecutor {
  constructor(options = {}) {
    this.layer = normalizeLlamaLayerIndex(options.layer ?? 0, "LLaMA K/V projection layer");
    this.embeddingRole = options.embeddingRole ?? "model.embed_tokens.weight";
    this.inputNormRole = options.inputNormRole ?? llamaLayerRole(this.layer, "input_layernorm.weight");
    this.kProjRole = options.kProjRole ?? llamaLayerRole(this.layer, "self_attn.k_proj.weight");
    this.vProjRole = options.vProjRole ?? llamaLayerRole(this.layer, "self_attn.v_proj.weight");
    this.kNormRole = options.kNormRole ?? llamaLayerRole(this.layer, "self_attn.k_norm.weight");
    this.qkProjectionNorm = options.qkProjectionNorm === true || options.qk_projection_norm === true || options.qk_norm === true;
    this.attentionHeadSize = options.attentionHeadSize ?? options.headSize ?? null;
    this.epsilon = options.epsilon ?? 1e-5;
    this.record = typeof options.record === "function" ? options.record : null;
    this.pipeline = null;
    this.pipelineDevice = null;
    this.paramsBuffer = null;
    this.tokensBuffer = null;
    this.tokensBufferByteLength = 0;
    this.paramsBytes = new ArrayBuffer(36);
    this.paramsView = new DataView(this.paramsBytes);
    this.preparedSessions = new WeakMap();
    this.mockKValuesScratch = new Float32Array(0);
    this.mockVValuesScratch = new Float32Array(0);
    this.kvWindowScratch = createLlamaPreparedKvCacheLayerWindowScratch();
    this.bindGroupDeps = [null, null, null, null];
  }

  prepareSession(session) {
    return prepareLlamaProjectionSession(this.preparedSessions, session, () => {
      const projection = llamaKvProjectionResources(session, {
        embeddingRole: this.embeddingRole,
        epsilon: this.epsilon,
        attentionHeadSize: this.attentionHeadSize,
        inputNormRole: this.inputNormRole,
        kProjRole: this.kProjRole,
        kNormRole: this.kNormRole,
        layer: this.layer,
        qkProjectionNorm: this.qkProjectionNorm,
        vProjRole: this.vProjRole,
      });
      return {
        ...projection,
        kvLayer: prepareLlamaProjectionKvLayer(session, projection, "LLaMA K/V projection"),
        tokenUpload: new Uint32Array(session.contextLength),
      };
    });
  }

  projectionForSession(session) {
    return this.preparedSessions.get(session) ?? this.prepareSession(session);
  }

  requiredGpuStorageBufferCount() {
    return 8;
  }

  canUseGpuProjection(device) {
    return device.canCreateGpuBuffers() && device.canBindStorageBuffers(this.requiredGpuStorageBufferCount());
  }

  executeTokensSync(context) {
    const projection = this.projectionForSession(context.session);
    const slots = this.kvProjectionSlots(context, projection);
    let backendDispatchCount = 0;
    let fallbackOpCount = 0;
    let commandCount = 0;
    let status = context.runtime.status.ok;
    if (context.execution.outputPolicy === 1) {
      status = context.runtime.status.unsupported;
    } else if (this.canUseGpuProjection(context.session.device)) {
      backendDispatchCount = this.submitGpuProjection(context, projection, slots);
      commandCount = backendDispatchCount;
    } else {
      this.mockKValuesScratch = ensureFloat32ScratchCapacity(this.mockKValuesScratch, slots.k.elementLength);
      this.mockVValuesScratch = ensureFloat32ScratchCapacity(this.mockVValuesScratch, slots.v.elementLength);
      context.session.device.writeFloat32(
        slots.k.descriptor,
        llamaKvProjectionMockValuesInto(this.mockKValuesScratch, context, projection, slots.k),
        slots.k.byteOffset,
        slots.k.elementLength,
      );
      context.session.device.writeFloat32(
        slots.v.descriptor,
        llamaKvProjectionMockValuesInto(this.mockVValuesScratch, context, projection, slots.v),
        slots.v.byteOffset,
        slots.v.elementLength,
      );
      fallbackOpCount = 1;
      commandCount = 1;
    }
    this.record?.({
      backendDispatchCount,
      commandCount,
      embeddingRole: projection.embedding.role,
      endPosition: context.stepParams.endPosition,
      epsilon: projection.epsilon,
      fallbackOpCount,
      hiddenSize: projection.hiddenSize,
      inputNormRole: projection.inputNorm.role,
      attentionHeadSize: projection.attentionHeadSize,
      kNormRole: projection.kNorm?.role ?? null,
      kProjRole: projection.kProj.role,
      qkProjectionNorm: projection.qkProjectionNorm,
      kvStride: slots.k.stride,
      kvWriteCount: status === context.runtime.status.ok ? 2 * context.stepParams.tokenCount : 0,
      layer: projection.layer,
      modelBindingKind: context.session.modelBindingKind,
      modelHandle: context.session.modelHandle,
      modelManifestHash: context.session.modelManifest?.hash ?? 0n,
      modelTableDescriptorHash: context.session.modelTable?.descriptorHash ?? 0n,
      modelTableHash: context.session.modelTable?.hash ?? 0n,
      outputKind: context.stepParams.outputKind,
      outputLength: 0,
      outputPolicy: context.execution.outputPolicy,
      position: context.stepParams.startPosition,
      status,
      tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
      tableHash: context.session.table?.hash ?? 0n,
      tokenCount: context.stepParams.tokenCount,
      tokens: llamaExecutionTokensArray(context.execution),
      vProjRole: projection.vProj.role,
    }, context);
    return writeLlamaTokenExecutorResult(
      context.resultScratch ?? {},
      context,
      status,
      0,
      backendDispatchCount,
      fallbackOpCount,
      commandCount,
    );
  }

  kvProjectionSlots(context, projection) {
    const kvLayer = projection.kvLayer ?? prepareLlamaProjectionKvLayer(context.session, projection, "LLaMA K/V projection");
    const slots = writeLlamaPreparedKvCacheLayerWindow(this.kvWindowScratch, kvLayer, context.stepParams);
    const k = slots.k;
    const v = slots.v;
    if (
      k.elementLength !== context.stepParams.tokenCount * k.stride ||
      v.elementLength !== context.stepParams.tokenCount * v.stride
    ) {
      throw new Error("LLaMA K/V projection cache window length does not match StepParams");
    }
    return slots;
  }

  ensureGpuPipeline(device) {
    if (!device.canCreateGpuBuffers()) {
      throw new Error("browser WebGPU device is unavailable");
    }
    if (this.pipeline && this.pipelineDevice === device.device) return this.pipeline;
    if (this.paramsBuffer && typeof this.paramsBuffer.destroy === "function") {
      this.paramsBuffer.destroy();
    }
    if (this.tokensBuffer && typeof this.tokensBuffer.destroy === "function") {
      this.tokensBuffer.destroy();
    }
    this.tokensBuffer = null;
    this.tokensBufferByteLength = 0;
    const gpuDevice = device.device;
    const shader = gpuDevice.createShaderModule({
      label: "zgml.wasm.webgpu.llama-kv-projection",
      code: `
struct Params {
  start_position: u32,
  token_count: u32,
  hidden: u32,
  stride: u32,
  context_length: u32,
  epsilon: f32,
  _pad0: u32,
  head_size: u32,
  qk_norm: u32,
};

@group(0) @binding(0) var<storage, read> embeddings: array<f32>;
@group(0) @binding(1) var<storage, read> input_norm: array<f32>;
@group(0) @binding(2) var<storage, read> k_proj: array<f32>;
@group(0) @binding(3) var<storage, read> v_proj: array<f32>;
@group(0) @binding(4) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(5) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<storage, read> tokens: array<u32>;
@group(0) @binding(8) var<storage, read> k_norm: array<f32>;

fn inv_rms(token: u32) -> f32 {
  let embedding_offset = token * params.hidden;
  var sum = 0.0;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    let value = embeddings[embedding_offset + col];
    sum = sum + value * value;
  }
  return inverseSqrt(sum / f32(params.hidden) + params.epsilon);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = params.token_count * params.stride;
  if (i >= total) {
    return;
  }
  let token_index = i / params.stride;
  let row = i % params.stride;
  let cache_position = params.start_position + token_index;
  if (cache_position >= params.context_length) {
    return;
  }
  let token = tokens[token_index];
  let embedding_offset = token * params.hidden;
  let scale = inv_rms(token);
  var k_acc = 0.0;
  var v_acc = 0.0;
  let weight_offset = row * params.hidden;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    let hidden = embeddings[embedding_offset + col] * input_norm[col] * scale;
    k_acc = k_acc + k_proj[weight_offset + col] * hidden;
    v_acc = v_acc + v_proj[weight_offset + col] * hidden;
  }
  if (params.qk_norm != 0u) {
    let local = row % params.head_size;
    let head_base = row - local;
    var sum = 0.0;
    for (var head_col = 0u; head_col < params.head_size; head_col = head_col + 1u) {
      var head_acc = 0.0;
      let head_weight_offset = (head_base + head_col) * params.hidden;
      for (var col = 0u; col < params.hidden; col = col + 1u) {
        let hidden = embeddings[embedding_offset + col] * input_norm[col] * scale;
        head_acc = head_acc + k_proj[head_weight_offset + col] * hidden;
      }
      sum = sum + head_acc * head_acc;
    }
    let inv_rms = inverseSqrt(sum / f32(params.head_size) + params.epsilon);
    k_acc = k_acc * k_norm[local] * inv_rms;
  }
  let cache_offset = cache_position * params.stride + row;
  k_cache[cache_offset] = k_acc;
  v_cache[cache_offset] = v_acc;
}
`,
    });
    this.pipeline = gpuDevice.createComputePipeline({
      label: "zgml.wasm.webgpu.llama-kv-projection",
      layout: "auto",
      compute: {
        module: shader,
        entryPoint: "main",
      },
    });
    const usage = gpuBufferUsageFlags();
    this.paramsBuffer = gpuDevice.createBuffer({
      label: "zgml.wasm.webgpu.llama-kv-projection.params",
      size: this.paramsBytes.byteLength,
      usage: usage.uniform | usage.copyDst,
    });
    this.pipelineDevice = gpuDevice;
    return this.pipeline;
  }

  ensureTokensBuffer(device, byteLength) {
    if (
      this.tokensBuffer &&
      this.pipelineDevice === device.device &&
      this.tokensBufferByteLength >= byteLength
    ) {
      return this.tokensBuffer;
    }
    if (this.tokensBuffer && typeof this.tokensBuffer.destroy === "function") {
      this.tokensBuffer.destroy();
    }
    const usage = gpuBufferUsageFlags();
    const capacity = growGpuByteCapacity(byteLength);
    this.tokensBuffer = device.device.createBuffer({
      label: "zgml.wasm.webgpu.llama-kv-projection.tokens",
      size: capacity,
      usage: usage.storage | usage.copyDst,
    });
    this.tokensBufferByteLength = capacity;
    return this.tokensBuffer;
  }

  submitGpuProjection(context, projection, slots) {
    const device = context.session.device;
    const gpuDevice = device.device;
    const pipeline = this.ensureGpuPipeline(device);
    const params = this.paramsBytes;
    const view = this.paramsView;
    view.setUint32(0, context.stepParams.startPosition, true);
    view.setUint32(4, context.stepParams.tokenCount, true);
    view.setUint32(8, projection.hiddenSize, true);
    view.setUint32(12, slots.k.stride, true);
    view.setUint32(16, context.stepParams.contextLength, true);
    view.setFloat32(20, projection.epsilon, true);
    view.setUint32(24, 0, true);
    view.setUint32(28, projection.attentionHeadSize ?? 0, true);
    view.setUint32(32, projection.qkProjectionNorm ? 1 : 0, true);
    gpuDevice.queue.writeBuffer(this.paramsBuffer, 0, params);
    const tokensByteLength = context.stepParams.tokenCount * Uint32Array.BYTES_PER_ELEMENT;
    let tokenUpload = projection.tokenUpload;
    if (!tokenUpload || tokenUpload.length < context.stepParams.tokenCount) {
      tokenUpload = new Uint32Array(growGpuByteCapacity(tokensByteLength) / Uint32Array.BYTES_PER_ELEMENT);
      projection.tokenUpload = tokenUpload;
    }
    for (let i = 0; i < context.stepParams.tokenCount; i += 1) {
      tokenUpload[i] = context.execution.tokens[i];
    }
    const tokensBuffer = this.ensureTokensBuffer(device, tokensByteLength);
    gpuDevice.queue.writeBuffer(tokensBuffer, 0, tokenUpload.buffer, tokenUpload.byteOffset, tokensByteLength);
    const bindGroupDeps = this.bindGroupDeps;
    bindGroupDeps[0] = this.paramsBuffer;
    bindGroupDeps[1] = tokensBuffer;
    bindGroupDeps[2] = slots.k.descriptor;
    bindGroupDeps[3] = slots.v.descriptor;
    bindGroupDeps[4] = projection.kNorm?.descriptor ?? projection.kProj.descriptor;
    const bindGroup = projectionGpuBindGroup(
      projection,
      "kv",
      device,
      pipeline,
      bindGroupDeps,
      (targetDevice) => targetDevice.createBindGroup({
        label: "zgml.wasm.webgpu.llama-kv-projection",
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: storageBufferBinding(
              targetDevice,
              projection.embedding.descriptor,
              "LLaMA K/V projection embeddings",
              projection.embedding.byteLength,
            ),
          },
          {
            binding: 1,
            resource: storageBufferBinding(
              targetDevice,
              projection.inputNorm.descriptor,
              "LLaMA K/V projection input norm",
              projection.inputNorm.byteLength,
            ),
          },
          {
            binding: 2,
            resource: storageBufferBinding(
              targetDevice,
              projection.kProj.descriptor,
              "LLaMA K projection weight",
              projection.kProj.byteLength,
            ),
          },
          {
            binding: 3,
            resource: storageBufferBinding(
              targetDevice,
              projection.vProj.descriptor,
              "LLaMA V projection weight",
              projection.vProj.byteLength,
            ),
          },
          {
            binding: 4,
            resource: storageBufferBinding(
              targetDevice,
              slots.k.descriptor,
              "LLaMA K cache",
              slots.k.descriptor.byteLength,
            ),
          },
          {
            binding: 5,
            resource: storageBufferBinding(
              targetDevice,
              slots.v.descriptor,
              "LLaMA V cache",
              slots.v.descriptor.byteLength,
            ),
          },
          { binding: 6, resource: { buffer: this.paramsBuffer } },
          { binding: 7, resource: { buffer: tokensBuffer } },
          {
            binding: 8,
            resource: storageBufferBinding(
              targetDevice,
              projection.kNorm?.descriptor ?? projection.kProj.descriptor,
              projection.kNorm === null ? "LLaMA K projection norm dummy" : "LLaMA K projection norm",
              projection.kNorm?.byteLength ?? projection.kProj.byteLength,
            ),
          },
        ],
      }),
    );
    const encoder = gpuDevice.createCommandEncoder({ label: "zgml.wasm.webgpu.llama-kv-projection" });
    const pass = encoder.beginComputePass({ label: "zgml.wasm.webgpu.llama-kv-projection" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil((context.stepParams.tokenCount * slots.k.stride) / 64));
    pass.end();
    gpuDevice.queue.submit([encoder.finish()]);
    return 1;
  }

  destroy() {
    if (this.paramsBuffer && typeof this.paramsBuffer.destroy === "function") {
      this.paramsBuffer.destroy();
    }
    if (this.tokensBuffer && typeof this.tokensBuffer.destroy === "function") {
      this.tokensBuffer.destroy();
    }
    this.paramsBuffer = null;
    this.tokensBuffer = null;
    this.tokensBufferByteLength = 0;
    this.mockKValuesScratch = new Float32Array(0);
    this.mockVValuesScratch = new Float32Array(0);
    this.pipeline = null;
    this.pipelineDevice = null;
    this.preparedSessions = new WeakMap();
    this.bindGroupDeps.fill(null);
  }
}

export class WasmWebGpuLlamaAttentionProjectionExecutor {
  constructor(options = {}) {
    this.layer = normalizeLlamaLayerIndex(options.layer ?? 0, "LLaMA attention projection layer");
    this.embeddingRole = options.embeddingRole ?? "model.embed_tokens.weight";
    this.inputNormRole = options.inputNormRole ?? llamaLayerRole(this.layer, "input_layernorm.weight");
    this.qProjRole = options.qProjRole ?? llamaLayerRole(this.layer, "self_attn.q_proj.weight");
    this.kProjRole = options.kProjRole ?? llamaLayerRole(this.layer, "self_attn.k_proj.weight");
    this.vProjRole = options.vProjRole ?? llamaLayerRole(this.layer, "self_attn.v_proj.weight");
    this.oProjRole = options.oProjRole ?? llamaLayerRole(this.layer, "self_attn.o_proj.weight");
    this.qNormRole = options.qNormRole ?? llamaLayerRole(this.layer, "self_attn.q_norm.weight");
    this.kNormRole = options.kNormRole ?? llamaLayerRole(this.layer, "self_attn.k_norm.weight");
    this.qkProjectionNorm = options.qkProjectionNorm === true || options.qk_projection_norm === true || options.qk_norm === true;
    this.qBiasRole = options.qBiasRole ?? llamaLayerRole(this.layer, "self_attn.q_proj.bias");
    this.kBiasRole = options.kBiasRole ?? llamaLayerRole(this.layer, "self_attn.k_proj.bias");
    this.vBiasRole = options.vBiasRole ?? llamaLayerRole(this.layer, "self_attn.v_proj.bias");
    this.oBiasRole = options.oBiasRole ?? llamaLayerRole(this.layer, "self_attn.o_proj.bias");
    this.normRole = options.normRole ?? "model.norm.weight";
    this.lmHeadRole = options.lmHeadRole ?? "lm_head.weight";
    this.lmHeadBiasRole = options.lmHeadBiasRole ?? "lm_head.bias";
    this.epsilon = options.epsilon ?? 1e-5;
    this.attentionHeadSize = options.attentionHeadSize ?? options.headSize ?? null;
    this.ropeBase = options.ropeBase ?? options.ropeTheta ?? 10000;
    this.ropeScale = options.ropeScale ?? options.rope_scale ?? options.ropeScalingFactor ?? options.rope_scaling_factor ?? 1;
    this.ropeKind = normalizeLlamaRopeKind(options.ropeKind ?? options.ropeType ?? options.rope_kind ?? options.rope_type ?? (Math.abs(this.ropeScale - 1) > 1e-6 ? "linear" : "none"));
    this.ropeLowFreqFactor = options.ropeLowFreqFactor ?? options.rope_low_freq_factor ?? 1;
    this.ropeHighFreqFactor = options.ropeHighFreqFactor ?? options.rope_high_freq_factor ?? 4;
    this.ropeOriginalContextLength = options.ropeOriginalContextLength ?? options.rope_original_context_length ?? 8192;
    this.useRope = normalizeLlamaUseRopeLayerFromOptions(
      options,
      this.layer,
      "LLaMA attention projection",
    );
    this.slidingWindow = normalizeLlamaSlidingWindowFromOptions(
      options,
      "LLaMA attention projection",
    );
    this.record = typeof options.record === "function" ? options.record : null;
    this.pipeline = null;
    this.pipelineDevice = null;
    this.paramsBuffer = null;
    this.paramsBytes = new ArrayBuffer(160);
    this.paramsView = new DataView(this.paramsBytes);
    this.preparedSessions = new WeakMap();
    this.ownedResources = [];
    this.kvWindowScratch = createLlamaPreparedKvCacheLayerWindowScratch();
    this.mockProjectionScratch = createLlamaAttentionProjectionMockScratch();
    this.projectionStepOptionsScratch = {
      outputLogits: false,
      outputPolicy: 0,
      position: 0,
      seq: 0,
      token: 0,
    };
    this.bindGroupDeps = [null, null, null, null, null];
  }

  prepareSession(session) {
    return prepareLlamaProjectionSession(this.preparedSessions, session, () => {
      const projection = llamaAttentionProjectionResources(session, {
        embeddingRole: this.embeddingRole,
        epsilon: this.epsilon,
        attentionHeadSize: this.attentionHeadSize,
        ropeBase: this.ropeBase,
        ropeHighFreqFactor: this.ropeHighFreqFactor,
        ropeKind: this.ropeKind,
        ropeLowFreqFactor: this.ropeLowFreqFactor,
        ropeOriginalContextLength: this.ropeOriginalContextLength,
        ropeScale: this.ropeScale,
        useRope: this.useRope,
        slidingWindow: this.slidingWindow,
        inputNormRole: this.inputNormRole,
        kProjRole: this.kProjRole,
        kBiasRole: this.kBiasRole,
        kNormRole: this.kNormRole,
        layer: this.layer,
        lmHeadBiasRole: this.lmHeadBiasRole,
        lmHeadRole: this.lmHeadRole,
        normRole: this.normRole,
        oProjRole: this.oProjRole,
        oBiasRole: this.oBiasRole,
        qProjRole: this.qProjRole,
        qBiasRole: this.qBiasRole,
        qNormRole: this.qNormRole,
        qkProjectionNorm: this.qkProjectionNorm,
        vProjRole: this.vProjRole,
        vBiasRole: this.vBiasRole,
      });
      return {
        ...projection,
        kvLayer: prepareLlamaProjectionKvLayer(session, projection, "LLaMA attention projection"),
        modelPack: packLlamaProjectionModelResources(session, projection, appendPresentLlamaModelPackFields([
          "embedding",
          "inputNorm",
          ...(projection.qkProjectionNorm ? ["qNorm", "kNorm"] : []),
          "qProj",
          "kProj",
          "vProj",
          "oProj",
          "norm",
          "lmHead",
        ], projection, ["qBias", "kBias", "vBias", "oBias", "lmHeadBias"]), "LLaMA attention projection", this),
      };
    });
  }

  projectionForSession(session) {
    return this.preparedSessions.get(session) ?? this.prepareSession(session);
  }

  requiredGpuStorageBufferCount() {
    return 4;
  }

  canUseGpuProjection(device) {
    return device.canCreateGpuBuffers() && device.canBindStorageBuffers(this.requiredGpuStorageBufferCount());
  }

  writeGpuFallbackAttentionResult(context, slots, result, position) {
    if (!context.session.device.canCreateGpuBuffers()) return;
    context.session.device.writeFloat32(
      slots.k.descriptor,
      result.k,
      position * slots.k.stride * Float32Array.BYTES_PER_ELEMENT,
      result.kLength ?? slots.k.stride,
    );
    context.session.device.writeFloat32(
      slots.v.descriptor,
      result.v,
      position * slots.v.stride * Float32Array.BYTES_PER_ELEMENT,
      result.vLength ?? slots.v.stride,
    );
  }

  executeTokensSync(context) {
    const projection = this.projectionForSession(context.session);
    const slots = this.attentionSlots(context, projection);
    let backendDispatchCount = 0;
    let fallbackOpCount = 0;
    let commandCount = 0;
    let status = context.runtime.status.ok;
    let outputLength = context.execution.outputPolicy === 1 ? context.session.vocabSize : 0;
    const tokens = context.execution.tokens;
    const tokenCount = llamaActiveTokenCount(context.execution);
    if (context.stepParams.tokenCount !== tokenCount || tokenCount === 0 || tokens.length < tokenCount) {
      status = context.runtime.status.unsupported;
      outputLength = 0;
    } else if (this.canUseGpuProjection(context.session.device)) {
      if (context.execution.outputPolicy === 1 && context.execution.outputPtr !== 0) {
        status = context.runtime.status.unsupported;
        outputLength = 0;
      } else {
        for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
          const position = context.stepParams.startPosition + tokenIndex;
          const stepOutputPolicy = tokenIndex === tokenCount - 1 ? context.execution.outputPolicy : 0;
          backendDispatchCount += this.submitGpuProjection(
            context,
            projection,
            slots,
            this.writeProjectionStepOptions(
              this.projectionStepOptionsScratch,
              stepOutputPolicy,
              position,
              tokens[tokenIndex],
            ),
          );
        }
        commandCount = backendDispatchCount;
      }
    } else {
      let result = null;
      for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
        const position = context.stepParams.startPosition + tokenIndex;
        const stepOutputPolicy = tokenIndex === tokenCount - 1 ? context.execution.outputPolicy : 0;
        result = llamaAttentionProjectionMockInto(
          this.mockProjectionScratch,
          context,
          projection,
          slots,
          this.writeProjectionStepOptions(
            this.projectionStepOptionsScratch,
            stepOutputPolicy,
            position,
            tokens[tokenIndex],
          ),
        );
        this.writeGpuFallbackAttentionResult(context, slots, result, position);
        fallbackOpCount += 1;
        commandCount += 1;
      }
      if (context.execution.outputPolicy === 1 && result !== null) {
        writeLlamaLogits(context, result.logits, result.logitsLength);
      }
    }
    this.record?.({
      backendDispatchCount,
      commandCount,
      embeddingRole: projection.embedding.role,
      endPosition: context.stepParams.endPosition,
      epsilon: projection.epsilon,
      attentionHeadSize: projection.attentionHeadSize,
      biasMask: projection.biasMask,
      fallbackOpCount,
      hiddenSize: projection.hiddenSize,
      inputNormRole: projection.inputNorm.role,
      queryHeadCount: projection.queryHeadCount,
      kvHeadCount: projection.kvHeadCount,
      kNormRole: projection.kNorm?.role ?? null,
      kProjRole: projection.kProj.role,
      kvStride: slots.k.stride,
      kvWriteCount: status === context.runtime.status.ok ? 2 * context.stepParams.tokenCount : 0,
      layer: projection.layer,
      lmHeadRole: projection.lmHead.role,
      lmHeadBiasRole: projection.lmHeadBias?.role ?? null,
      modelBindingKind: context.session.modelBindingKind,
      modelHandle: context.session.modelHandle,
      modelManifestHash: context.session.modelManifest?.hash ?? 0n,
      ...llamaModelPackRecordEvidence(projection.modelPack),
      modelTableDescriptorHash: context.session.modelTable?.descriptorHash ?? 0n,
      modelTableHash: context.session.modelTable?.hash ?? 0n,
      normRole: projection.norm.role,
      oProjRole: projection.oProj.role,
      outputKind: context.stepParams.outputKind,
      outputLength,
      outputPolicy: context.execution.outputPolicy,
      position: context.stepParams.startPosition,
      qNormRole: projection.qNorm?.role ?? null,
      qkProjectionNorm: projection.qkProjectionNorm,
      qProjRole: projection.qProj.role,
      ropeBase: projection.ropeBase,
      ropeHighFreqFactor: projection.ropeHighFreqFactor,
      ropeKind: projection.ropeKind,
      ropeKindId: projection.ropeKindId,
      ropeLowFreqFactor: projection.ropeLowFreqFactor,
      ropeOriginalContextLength: projection.ropeOriginalContextLength,
      ropeScale: projection.ropeScale,
      slidingWindow: projection.slidingWindow,
      status,
      tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
      tableHash: context.session.table?.hash ?? 0n,
      token: llamaLastToken(context.execution),
      tokens: llamaExecutionTokensArray(context.execution),
      tokenCount: context.stepParams.tokenCount,
      useRope: projection.useRope,
      vProjRole: projection.vProj.role,
      vocabSize: projection.vocabSize,
    }, context);
    return writeLlamaModelPackResultEvidence(
      writeLlamaTokenExecutorResult(
        context.resultScratch ?? {},
        context,
        status,
        outputLength,
        backendDispatchCount,
        fallbackOpCount,
        commandCount,
      ),
      projection.modelPack,
    );
  }

  attentionSlots(context, projection) {
    const kvLayer = projection.kvLayer ?? prepareLlamaProjectionKvLayer(context.session, projection, "LLaMA attention projection");
    return writeLlamaPreparedKvCacheLayerWindow(this.kvWindowScratch, kvLayer, context.stepParams);
  }

  writeProjectionStepOptions(target, outputPolicy, position, token) {
    target.outputLogits = outputPolicy === 1;
    target.outputPolicy = outputPolicy;
    target.position = position;
    target.seq = position + 1;
    target.token = token;
    return target;
  }

  ensureGpuPipeline(device) {
    if (!device.canCreateGpuBuffers()) {
      throw new Error("browser WebGPU device is unavailable");
    }
    if (this.pipeline && this.pipelineDevice === device.device) return this.pipeline;
    if (this.paramsBuffer && typeof this.paramsBuffer.destroy === "function") {
      this.paramsBuffer.destroy();
    }
    const gpuDevice = device.device;
    const shader = gpuDevice.createShaderModule({
      label: "zgml.wasm.webgpu.llama-attention-projection",
      code: `
struct Params {
  position: u32,
  seq: u32,
  hidden: u32,
  stride: u32,
  vocab: u32,
  context_length: u32,
  token: u32,
  output_policy: u32,
  epsilon: f32,
  q_stride: u32,
  head_size: u32,
  query_heads_per_kv: u32,
  embedding_offset: u32,
  input_norm_offset: u32,
  q_proj_offset: u32,
  k_proj_offset: u32,
  v_proj_offset: u32,
  o_proj_offset: u32,
  final_norm_offset: u32,
  lm_head_offset: u32,
  rope_base: f32,
  rope_scale: f32,
  rope_kind: u32,
  rope_low_freq_factor: f32,
  rope_high_freq_factor: f32,
  rope_original_context_length: f32,
  sliding_window: u32,
  q_bias_offset: u32,
  k_bias_offset: u32,
  v_bias_offset: u32,
  o_bias_offset: u32,
	  lm_head_bias_offset: u32,
	  bias_mask: u32,
	  use_rope: u32,
	  q_norm_offset: u32,
	  k_norm_offset: u32,
	  qk_norm: u32,
	};

@group(0) @binding(0) var<storage, read> model: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn embedding_value(col: u32) -> f32 {
  return model[params.embedding_offset + params.token * params.hidden + col];
}

fn input_rms_scale() -> f32 {
  var sum = 0.0;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    let value = embedding_value(col);
    sum = sum + value * value;
  }
  return inverseSqrt(sum / f32(params.hidden) + params.epsilon);
}

fn normed_input(col: u32) -> f32 {
  return embedding_value(col) * model[params.input_norm_offset + col] * input_rms_scale();
}

fn project_q_raw(row: u32) -> f32 {
  var acc = 0.0;
  let base = row * params.hidden;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    acc = acc + model[params.q_proj_offset + base + col] * normed_input(col);
  }
  if ((params.bias_mask & 1u) != 0u) {
    acc = acc + model[params.q_bias_offset + row];
  }
  return acc;
}

fn project_k_raw(row: u32) -> f32 {
  var acc = 0.0;
  let base = row * params.hidden;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    acc = acc + model[params.k_proj_offset + base + col] * normed_input(col);
  }
  if ((params.bias_mask & 2u) != 0u) {
    acc = acc + model[params.k_bias_offset + row];
  }
  return acc;
}

fn q_norm_scale(row: u32) -> f32 {
  if (params.qk_norm == 0u) {
    return 1.0;
  }
  let local = row % params.head_size;
  let base = row - local;
  var sum = 0.0;
  for (var col = 0u; col < params.head_size; col = col + 1u) {
    let value = project_q_raw(base + col);
    sum = sum + value * value;
  }
  return inverseSqrt(sum / f32(params.head_size) + params.epsilon);
}

fn k_norm_scale(row: u32) -> f32 {
  if (params.qk_norm == 0u) {
    return 1.0;
  }
  let local = row % params.head_size;
  let base = row - local;
  var sum = 0.0;
  for (var col = 0u; col < params.head_size; col = col + 1u) {
    let value = project_k_raw(base + col);
    sum = sum + value * value;
  }
  return inverseSqrt(sum / f32(params.head_size) + params.epsilon);
}

fn project_q(row: u32) -> f32 {
  let value = project_q_raw(row);
  if (params.qk_norm == 0u) {
    return value;
  }
  return value * model[params.q_norm_offset + row % params.head_size] * q_norm_scale(row);
}

fn project_k(row: u32) -> f32 {
  let value = project_k_raw(row);
  if (params.qk_norm == 0u) {
    return value;
  }
  return value * model[params.k_norm_offset + row % params.head_size] * k_norm_scale(row);
}

fn project_v(row: u32) -> f32 {
  var acc = 0.0;
  let base = row * params.hidden;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    acc = acc + model[params.v_proj_offset + base + col] * normed_input(col);
  }
  if ((params.bias_mask & 4u) != 0u) {
    acc = acc + model[params.v_bias_offset + row];
  }
  return acc;
}

fn rope_inv_frequency(pair: u32) -> f32 {
  let inv_freq = 1.0 / pow(params.rope_base, f32(pair * 2u) / f32(params.head_size));
  if (params.rope_kind == 1u) {
    return inv_freq / params.rope_scale;
  }
  if (params.rope_kind == 2u) {
    let wavelen = 6.283185307179586 / inv_freq;
    let low_freq_wavelen = params.rope_original_context_length / params.rope_low_freq_factor;
    let high_freq_wavelen = params.rope_original_context_length / params.rope_high_freq_factor;
    if (wavelen > low_freq_wavelen) {
      return inv_freq / params.rope_scale;
    }
    if (wavelen < high_freq_wavelen) {
      return inv_freq;
    }
    let smooth_factor = (params.rope_original_context_length / wavelen - params.rope_low_freq_factor) / (params.rope_high_freq_factor - params.rope_low_freq_factor);
    return (1.0 - smooth_factor) * inv_freq / params.rope_scale + smooth_factor * inv_freq;
  }
  return inv_freq;
}

fn rope_component(lo: f32, hi: f32, pair: u32, high: bool) -> f32 {
  let freq = f32(params.position) * rope_inv_frequency(pair);
  let c = cos(freq);
  let s = sin(freq);
  if (high) {
    return hi * c + lo * s;
  }
  return lo * c - hi * s;
}

	fn rotated_q(row: u32) -> f32 {
	  if (params.use_rope == 0u) {
	    return project_q(row);
	  }
	  let local = row % params.head_size;
  let base = row - local;
  let half = params.head_size / 2u;
  if (local < half) {
    return rope_component(project_q(row), project_q(row + half), local, false);
  }
  let pair = local - half;
  return rope_component(project_q(base + pair), project_q(row), pair, true);
}

	fn rotated_current_k(row: u32) -> f32 {
	  if (params.use_rope == 0u) {
	    return project_k(row);
	  }
	  let local = row % params.head_size;
  let base = row - local;
  let half = params.head_size / 2u;
  if (local < half) {
    return rope_component(project_k(row), project_k(row + half), local, false);
  }
  let pair = local - half;
  return rope_component(project_k(base + pair), project_k(row), pair, true);
}

fn cache_k(position: u32, row: u32) -> f32 {
  if (position == params.position) {
    return rotated_current_k(row);
  }
  return k_cache[position * params.stride + row];
}

fn cache_v(position: u32, row: u32) -> f32 {
  if (position == params.position) {
    return project_v(row);
  }
  return v_cache[position * params.stride + row];
}

fn attention_start(seq: u32) -> u32 {
  if (params.sliding_window == 0u || seq <= params.sliding_window) {
    return 0u;
  }
  return seq - params.sliding_window;
}

fn attention_value(row: u32) -> f32 {
  let q_head = row / params.head_size;
  let kv_head = q_head / params.query_heads_per_kv;
  let q_base = q_head * params.head_size;
  let kv_base = kv_head * params.head_size;
  let local = row - q_base;
  let scale = inverseSqrt(f32(params.head_size));
  let start = attention_start(params.seq);
  var max_score = -3.4028234663852886e38;
  for (var pos = start; pos < params.seq; pos = pos + 1u) {
    var dot = 0.0;
    for (var col = 0u; col < params.head_size; col = col + 1u) {
      dot = dot + rotated_q(q_base + col) * cache_k(pos, kv_base + col);
    }
    let score = dot * scale;
    max_score = max(max_score, score);
  }
  var denom = 0.0;
  var acc = 0.0;
  for (var pos = start; pos < params.seq; pos = pos + 1u) {
    var dot = 0.0;
    for (var col = 0u; col < params.head_size; col = col + 1u) {
      dot = dot + rotated_q(q_base + col) * cache_k(pos, kv_base + col);
    }
    let weight = exp(dot * scale - max_score);
    denom = denom + weight;
    acc = acc + weight * cache_v(pos, kv_base + local);
  }
  if (denom <= 0.0) {
    return 0.0;
  }
  return acc / denom;
}

fn attention_projected(row: u32) -> f32 {
  var acc = 0.0;
  let base = row * params.q_stride;
  for (var col = 0u; col < params.q_stride; col = col + 1u) {
    acc = acc + model[params.o_proj_offset + base + col] * attention_value(col);
  }
  if ((params.bias_mask & 8u) != 0u) {
    acc = acc + model[params.o_bias_offset + row];
  }
  return acc;
}

fn after_attention(row: u32) -> f32 {
  return embedding_value(row) + attention_projected(row);
}

fn final_rms_scale() -> f32 {
  var sum = 0.0;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    let value = after_attention(col);
    sum = sum + value * value;
  }
  return inverseSqrt(sum / f32(params.hidden) + params.epsilon);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row < params.stride) {
    let cache_offset = params.position * params.stride + row;
    k_cache[cache_offset] = rotated_current_k(row);
    v_cache[cache_offset] = project_v(row);
  }
  if (params.output_policy == 1u && row < params.vocab) {
    let scale = final_rms_scale();
    let lm_offset = row * params.hidden;
    var acc = 0.0;
      for (var col = 0u; col < params.hidden; col = col + 1u) {
        let hidden = after_attention(col) * model[params.final_norm_offset + col] * scale;
        acc = acc + model[params.lm_head_offset + lm_offset + col] * hidden;
      }
      if ((params.bias_mask & 128u) != 0u) {
        acc = acc + model[params.lm_head_bias_offset + row];
      }
    logits[row] = acc;
  }
}
`,
    });
    this.pipeline = gpuDevice.createComputePipeline({
      label: "zgml.wasm.webgpu.llama-attention-projection",
      layout: "auto",
      compute: {
        module: shader,
        entryPoint: "main",
      },
    });
    const usage = gpuBufferUsageFlags();
    this.paramsBuffer = gpuDevice.createBuffer({
      label: "zgml.wasm.webgpu.llama-attention-projection.params",
      size: 160,
      usage: usage.uniform | usage.copyDst,
    });
    this.pipelineDevice = gpuDevice;
    return this.pipeline;
  }

  submitGpuProjection(context, projection, slots, options = {}) {
    const device = context.session.device;
    const gpuDevice = device.device;
    const pipeline = this.ensureGpuPipeline(device);
    const position = options.position ?? context.stepParams.startPosition;
    const seq = options.seq ?? context.stepParams.endPosition;
    const token = options.token ?? llamaLastToken(context.execution);
    const outputPolicy = options.outputPolicy ?? context.execution.outputPolicy;
    const params = this.paramsBytes;
    const view = this.paramsView;
    view.setUint32(0, position, true);
    view.setUint32(4, seq, true);
    view.setUint32(8, projection.hiddenSize, true);
    view.setUint32(12, slots.k.stride, true);
    view.setUint32(16, projection.vocabSize, true);
    view.setUint32(20, context.stepParams.contextLength, true);
    view.setUint32(24, token, true);
    view.setUint32(28, outputPolicy, true);
    view.setFloat32(32, projection.epsilon, true);
    view.setUint32(36, projection.qProj.rows, true);
    view.setUint32(40, projection.attentionHeadSize, true);
    view.setUint32(44, projection.queryHeadsPerKvHead, true);
    view.setUint32(48, projection.modelPack.offsets.embedding, true);
    view.setUint32(52, projection.modelPack.offsets.inputNorm, true);
    view.setUint32(56, projection.modelPack.offsets.qProj, true);
    view.setUint32(60, projection.modelPack.offsets.kProj, true);
    view.setUint32(64, projection.modelPack.offsets.vProj, true);
    view.setUint32(68, projection.modelPack.offsets.oProj, true);
    view.setUint32(72, projection.modelPack.offsets.norm, true);
    view.setUint32(76, projection.modelPack.offsets.lmHead, true);
    view.setFloat32(80, projection.ropeBase, true);
    view.setFloat32(84, projection.ropeScale, true);
    view.setUint32(88, projection.ropeKindId, true);
    view.setFloat32(92, projection.ropeLowFreqFactor, true);
    view.setFloat32(96, projection.ropeHighFreqFactor, true);
    view.setFloat32(100, projection.ropeOriginalContextLength, true);
    view.setUint32(104, projection.slidingWindow ?? 0, true);
    view.setUint32(108, projection.modelPack.offsets.qBias ?? 0, true);
    view.setUint32(112, projection.modelPack.offsets.kBias ?? 0, true);
    view.setUint32(116, projection.modelPack.offsets.vBias ?? 0, true);
    view.setUint32(120, projection.modelPack.offsets.oBias ?? 0, true);
	    view.setUint32(124, projection.modelPack.offsets.lmHeadBias ?? 0, true);
	    view.setUint32(128, projection.biasMask ?? 0, true);
	    view.setUint32(132, projection.useRope ? 1 : 0, true);
	    view.setUint32(136, projection.modelPack.offsets.qNorm ?? 0, true);
	    view.setUint32(140, projection.modelPack.offsets.kNorm ?? 0, true);
	    view.setUint32(144, projection.qkProjectionNorm ? 1 : 0, true);
    gpuDevice.queue.writeBuffer(this.paramsBuffer, 0, params);
    const bindGroupDeps = this.bindGroupDeps;
    bindGroupDeps[0] = this.paramsBuffer;
    bindGroupDeps[1] = projection.modelPack.descriptor;
    bindGroupDeps[2] = context.session.bindings.output;
    bindGroupDeps[3] = slots.k.descriptor;
    bindGroupDeps[4] = slots.v.descriptor;
    const bindGroup = projectionGpuBindGroup(
      projection,
      "attention",
      device,
      pipeline,
      bindGroupDeps,
      (targetDevice) => targetDevice.createBindGroup({
        label: "zgml.wasm.webgpu.llama-attention-projection",
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: storageBufferBinding(
              targetDevice,
              projection.modelPack.descriptor,
              "LLaMA attention projection model pack",
              projection.modelPack.byteLength,
            ),
          },
          {
            binding: 1,
            resource: storageBufferBinding(
              targetDevice,
              slots.k.descriptor,
              "LLaMA attention K cache",
              slots.k.descriptor.byteLength,
            ),
          },
          {
            binding: 2,
            resource: storageBufferBinding(
              targetDevice,
              slots.v.descriptor,
              "LLaMA attention V cache",
              slots.v.descriptor.byteLength,
            ),
          },
          {
            binding: 3,
            resource: storageBufferBinding(
              targetDevice,
              context.session.bindings.output,
              "LLaMA attention projection logits",
              projection.vocabSize * Float32Array.BYTES_PER_ELEMENT,
            ),
          },
          { binding: 4, resource: { buffer: this.paramsBuffer } },
        ],
      }),
    );
    const encoder = gpuDevice.createCommandEncoder({ label: "zgml.wasm.webgpu.llama-attention-projection" });
    const pass = encoder.beginComputePass({ label: "zgml.wasm.webgpu.llama-attention-projection" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(Math.max(projection.vocabSize, slots.k.stride, projection.qProj.rows) / 64));
    pass.end();
    gpuDevice.queue.submit([encoder.finish()]);
    return 1;
  }

  destroy() {
    if (this.paramsBuffer && typeof this.paramsBuffer.destroy === "function") {
      this.paramsBuffer.destroy();
    }
    for (let i = this.ownedResources.length - 1; i >= 0; i -= 1) {
      const resource = this.ownedResources[i];
      destroyTrackedWasmWebGpuResource(resource);
    }
    this.ownedResources = [];
    this.paramsBuffer = null;
    this.pipeline = null;
    this.pipelineDevice = null;
    this.preparedSessions = new WeakMap();
    this.mockProjectionScratch = createLlamaAttentionProjectionMockScratch();
    this.bindGroupDeps.fill(null);
  }
}

export class WasmWebGpuLlamaBlockProjectionExecutor extends WasmWebGpuLlamaAttentionProjectionExecutor {
  constructor(options = {}) {
    super(options);
    this.postNormRole = options.postNormRole ?? llamaLayerRole(this.layer, "post_attention_layernorm.weight");
    this.gateProjRole = options.gateProjRole ?? llamaLayerRole(this.layer, "mlp.gate_proj.weight");
    this.upProjRole = options.upProjRole ?? llamaLayerRole(this.layer, "mlp.up_proj.weight");
    this.downProjRole = options.downProjRole ?? llamaLayerRole(this.layer, "mlp.down_proj.weight");
    this.gateBiasRole = options.gateBiasRole ?? llamaLayerRole(this.layer, "mlp.gate_proj.bias");
    this.upBiasRole = options.upBiasRole ?? llamaLayerRole(this.layer, "mlp.up_proj.bias");
    this.downBiasRole = options.downBiasRole ?? llamaLayerRole(this.layer, "mlp.down_proj.bias");
    this.paramsBytes = new ArrayBuffer(192);
    this.paramsView = new DataView(this.paramsBytes);
    this.scalarProjectionMinHiddenSize = options.scalarProjectionMinHiddenSize ?? 64;
    if (
      !Number.isSafeInteger(this.scalarProjectionMinHiddenSize) ||
      this.scalarProjectionMinHiddenSize <= 0
    ) {
      throw new Error(`invalid LLaMA block scalarProjectionMinHiddenSize: ${this.scalarProjectionMinHiddenSize}`);
    }
    this.scalarPipeline = null;
    this.scalarPipelineDevice = null;
    this.windowScalarPipeline = null;
    this.windowScalarPipelineDevice = null;
    this.tokensBuffer = null;
    this.tokensBufferByteLength = 0;
    this.bindGroupDeps = [null, null, null, null, null, null, null];
    this.mockBlockProjectionScratch = createLlamaBlockProjectionMockScratch();
    this.prefixContextScratch = {
      execution: {
        outputLen: 0,
        outputPolicy: 0,
        outputPtr: 0,
        tokens: null,
        tokensLen: 0,
      },
      runtime: null,
      session: null,
      stepParams: {
        contextLength: 0,
        endPosition: 0,
        kind: "llama-token-window",
        logitsLength: 0,
        outputKind: "none",
        outputPolicy: 0,
        requestedOutputLength: 0,
        startPosition: 0,
        tokenCount: 0,
      },
    };
  }

  resourceOptions() {
    return {
      attentionHeadSize: this.attentionHeadSize,
      downBiasRole: this.downBiasRole,
      downProjRole: this.downProjRole,
      embeddingRole: this.embeddingRole,
      epsilon: this.epsilon,
      gateBiasRole: this.gateBiasRole,
      gateProjRole: this.gateProjRole,
      inputNormRole: this.inputNormRole,
      kBiasRole: this.kBiasRole,
      kNormRole: this.kNormRole,
      kProjRole: this.kProjRole,
      layer: this.layer,
      lmHeadBiasRole: this.lmHeadBiasRole,
      lmHeadRole: this.lmHeadRole,
      normRole: this.normRole,
      oBiasRole: this.oBiasRole,
      oProjRole: this.oProjRole,
      postNormRole: this.postNormRole,
      qBiasRole: this.qBiasRole,
      qNormRole: this.qNormRole,
      qkProjectionNorm: this.qkProjectionNorm,
      qProjRole: this.qProjRole,
      ropeBase: this.ropeBase,
      ropeHighFreqFactor: this.ropeHighFreqFactor,
      ropeKind: this.ropeKind,
      ropeLowFreqFactor: this.ropeLowFreqFactor,
      ropeOriginalContextLength: this.ropeOriginalContextLength,
      ropeScale: this.ropeScale,
      useRope: this.useRope,
      slidingWindow: this.slidingWindow,
      upBiasRole: this.upBiasRole,
      upProjRole: this.upProjRole,
      vBiasRole: this.vBiasRole,
      vProjRole: this.vProjRole,
    };
  }

  requiredGpuStorageBufferCount() {
    return 6;
  }

  requiredWindowGpuStorageBufferCount() {
    return 6;
  }

  prepareSession(session) {
    return prepareLlamaProjectionSession(this.preparedSessions, session, () => {
      const projection = llamaBlockProjectionResources(session, this.resourceOptions());
      const activationSlots = prepareLlamaBlockActivationSlots(session, projection, "LLaMA block projection");
      return {
        ...projection,
        activation: activationSlots.output,
        activationDummy: createLlamaProjectionDummyStorage(
          session,
          "zgml.wasm.webgpu.llama-block-projection.activation-dummy",
          "readwrite",
          this,
        ),
        activationInput: activationSlots.input,
        activationInputDummy: createLlamaProjectionDummyStorage(
          session,
          "zgml.wasm.webgpu.llama-block-projection.activation-input-dummy",
          "read",
          this,
        ),
        kvLayer: prepareLlamaProjectionKvLayer(session, projection, "LLaMA block projection"),
        modelPack: packLlamaProjectionModelResources(session, projection, appendPresentLlamaModelPackFields([
          "embedding",
          "inputNorm",
          ...(projection.qkProjectionNorm ? ["qNorm", "kNorm"] : []),
          "qProj",
          "kProj",
          "vProj",
          "oProj",
          "postNorm",
          "gateProj",
          "upProj",
          "downProj",
          "norm",
          "lmHead",
        ], projection, ["qBias", "kBias", "vBias", "oBias", "gateBias", "upBias", "downBias", "lmHeadBias"]), "LLaMA block projection", this),
        tokenUpload: new Uint32Array(session.contextLength),
      };
    });
  }

  executeTokensSync(context) {
    const projection = this.projectionForSession(context.session);
    const slots = this.attentionSlots(context, projection);
    let backendDispatchCount = 0;
    let fallbackOpCount = 0;
    let commandCount = 0;
    let scalarDispatchCount = 0;
    let windowDispatchCount = 0;
    let genericDispatchCount = 0;
    let activationWriteCount = 0;
    let status = context.runtime.status.ok;
    let outputLength = context.execution.outputPolicy === 1 ? context.session.vocabSize : 0;
    const tokens = context.execution.tokens;
    const tokenCount = llamaActiveTokenCount(context.execution);
    if (context.stepParams.tokenCount !== tokenCount || tokenCount === 0 || tokens.length < tokenCount) {
      status = context.runtime.status.unsupported;
      outputLength = 0;
    } else if (this.canUseGpuProjection(context.session.device)) {
      if (context.execution.outputPolicy === 1 && context.execution.outputPtr !== 0) {
        status = context.runtime.status.unsupported;
        outputLength = 0;
      } else if (this.canUseScalarGpuWindowProjection(context, projection, slots)) {
        backendDispatchCount += this.submitGpuWindowProjection(context, projection, slots);
        windowDispatchCount += 1;
        if (projection.activation != null) activationWriteCount += tokenCount;
        commandCount = backendDispatchCount;
      } else if (this.canUseScalarGpuLogitsPrefixWindowProjection(context, projection, slots)) {
        const prefixTokenCount = tokenCount - 1;
        const prefixContext = this.writePrefixNoOutputContext(this.prefixContextScratch, context, prefixTokenCount);
        backendDispatchCount += this.submitGpuWindowProjection(prefixContext, projection, slots);
        windowDispatchCount += 1;
        const useScalarTerminalProjection = this.canUseScalarGpuProjection(context, projection, slots);
        if (projection.activation != null) activationWriteCount += prefixTokenCount;
        const position = context.stepParams.startPosition + prefixTokenCount;
        backendDispatchCount += this.submitGpuProjection(
          context,
          projection,
          slots,
          this.writeProjectionStepOptions(
            this.projectionStepOptionsScratch,
            context.execution.outputPolicy,
            position,
            tokens[prefixTokenCount],
          ),
        );
        if (useScalarTerminalProjection) {
          scalarDispatchCount += 1;
        } else {
          genericDispatchCount += 1;
        }
        if (projection.activation != null) activationWriteCount += 1;
        commandCount = backendDispatchCount;
      } else {
        const useScalarTokenProjection = this.canUseScalarGpuProjection(context, projection, slots);
        for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
          const position = context.stepParams.startPosition + tokenIndex;
          const stepOutputPolicy = tokenIndex === tokenCount - 1 ? context.execution.outputPolicy : 0;
          backendDispatchCount += this.submitGpuProjection(
            context,
            projection,
            slots,
            this.writeProjectionStepOptions(
              this.projectionStepOptionsScratch,
              stepOutputPolicy,
              position,
              tokens[tokenIndex],
            ),
          );
          if (useScalarTokenProjection) {
            scalarDispatchCount += 1;
          } else {
            genericDispatchCount += 1;
          }
          if (projection.activation != null) activationWriteCount += 1;
        }
        commandCount = backendDispatchCount;
      }
    } else {
      let result = null;
      for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
        const position = context.stepParams.startPosition + tokenIndex;
        const stepOutputPolicy = tokenIndex === tokenCount - 1 ? context.execution.outputPolicy : 0;
        result = llamaBlockProjectionMockInto(
          this.mockBlockProjectionScratch,
          context,
          projection,
          slots,
          this.writeProjectionStepOptions(
            this.projectionStepOptionsScratch,
            stepOutputPolicy,
            position,
            tokens[tokenIndex],
          ),
        );
        this.writeGpuFallbackAttentionResult(context, slots, result, position);
        activationWriteCount += writeLlamaActivationToken(
          context.session,
          projection.activation,
          position,
          result.blockHidden,
          "LLaMA block projection",
          result.blockHiddenLength,
        );
        fallbackOpCount += 1;
        commandCount += 1;
      }
      if (context.execution.outputPolicy === 1 && result !== null) {
        writeLlamaLogits(context, result.logits, result.logitsLength);
      }
    }
    const deviceEvidence = llamaExecutorDeviceEvidence(context.session, 6);
    this.record?.({
      activationByteLength: projection.activation?.descriptor.byteLength ?? 0,
      activationInputByteLength: projection.activationInput?.descriptor.byteLength ?? 0,
      activationInputRole: projection.activationInput == null ? null : "llama.activation.input",
      activationRole: projection.activation == null ? null : "llama.activation",
      activationWriteCount,
      attentionHeadSize: projection.attentionHeadSize,
      backendDispatchCount,
      biasMask: projection.biasMask,
      commandCount,
      ...deviceEvidence,
      downProjRole: projection.downProj.role,
      embeddingRole: projection.embedding.role,
      endPosition: context.stepParams.endPosition,
      epsilon: projection.epsilon,
      fallbackOpCount,
      ffnSize: projection.ffnSize,
      gateProjRole: projection.gateProj.role,
      hiddenSize: projection.hiddenSize,
      inputNormRole: projection.inputNorm.role,
      kNormRole: projection.kNorm?.role ?? null,
      kProjRole: projection.kProj.role,
      kvHeadCount: projection.kvHeadCount,
      kvStride: slots.k.stride,
      kvWriteCount: status === context.runtime.status.ok ? 2 * context.stepParams.tokenCount : 0,
      layer: projection.layer,
      lmHeadBiasRole: projection.lmHeadBias?.role ?? null,
      lmHeadRole: projection.lmHead.role,
      modelBindingKind: context.session.modelBindingKind,
      modelHandle: context.session.modelHandle,
      modelManifestHash: context.session.modelManifest?.hash ?? 0n,
      ...llamaModelPackRecordEvidence(projection.modelPack),
      modelTableDescriptorHash: context.session.modelTable?.descriptorHash ?? 0n,
      modelTableHash: context.session.modelTable?.hash ?? 0n,
      normRole: projection.norm.role,
      oProjRole: projection.oProj.role,
      outputKind: context.stepParams.outputKind,
      outputLength,
      outputPolicy: context.execution.outputPolicy,
      position: context.stepParams.startPosition,
      postNormRole: projection.postNorm.role,
      qNormRole: projection.qNorm?.role ?? null,
      qkProjectionNorm: projection.qkProjectionNorm,
      qProjRole: projection.qProj.role,
      queryHeadCount: projection.queryHeadCount,
      ropeBase: projection.ropeBase,
      ropeHighFreqFactor: projection.ropeHighFreqFactor,
      ropeKind: projection.ropeKind,
      ropeKindId: projection.ropeKindId,
      ropeLowFreqFactor: projection.ropeLowFreqFactor,
      ropeOriginalContextLength: projection.ropeOriginalContextLength,
      ropeScale: projection.ropeScale,
      genericDispatchCount,
      scalarDispatchCount,
      slidingWindow: projection.slidingWindow,
      status,
      windowDispatchCount,
      tableDescriptorHash: context.session.table?.descriptorHash ?? 0n,
      tableHash: context.session.table?.hash ?? 0n,
      token: llamaLastToken(context.execution),
      tokens: llamaExecutionTokensArray(context.execution),
      tokenCount: context.stepParams.tokenCount,
      useRope: projection.useRope,
      upProjRole: projection.upProj.role,
      vProjRole: projection.vProj.role,
      vocabSize: projection.vocabSize,
    }, context);
    const result = writeLlamaModelPackResultEvidence(
      writeLlamaTokenExecutorResult(
        context.resultScratch ?? {},
        context,
        status,
        outputLength,
        backendDispatchCount,
        fallbackOpCount,
        commandCount,
      ),
      projection.modelPack,
    );
    result.genericDispatchCount = genericDispatchCount;
    result.scalarDispatchCount = scalarDispatchCount;
    result.usesGpuStorageBuffers = deviceEvidence.usesGpuStorageBuffers;
    result.windowDispatchCount = windowDispatchCount;
    return result;
  }

  canUseScalarGpuProjection(context, projection, slots) {
    return (
      projection.hiddenSize >= this.scalarProjectionMinHiddenSize &&
      projection.hiddenSize <= 256 &&
      projection.ffnSize <= 256 &&
      projection.qProj.rows <= 256 &&
      slots.k.stride <= 256 &&
      this.scalarWindowScoreSlotCount(context.stepParams, projection) <= llamaBlockScalarWindowMaxContextLength
    );
  }

  scalarWindowScoreSlotCount(stepParams, projection, tokenCount = stepParams.tokenCount) {
    const endPosition = stepParams.startPosition + tokenCount;
    const activeSeq = Math.min(stepParams.contextLength, endPosition);
    const slidingWindow = projection.slidingWindow ?? 0;
    return slidingWindow > 0 ? Math.min(activeSeq, slidingWindow) : activeSeq;
  }

  canUseScalarGpuWindowProjection(context, projection, slots) {
    return (
      context.execution.outputPolicy === 0 &&
      context.stepParams.tokenCount > 1 &&
      context.stepParams.tokenCount === llamaActiveTokenCount(context.execution) &&
      context.execution.tokens.length >= context.stepParams.tokenCount &&
      context.session.device.canBindStorageBuffers(this.requiredWindowGpuStorageBufferCount()) &&
      projection.hiddenSize <= 256 &&
      projection.ffnSize <= 256 &&
      projection.qProj.rows <= 256 &&
      slots.k.stride <= 256 &&
      this.scalarWindowScoreSlotCount(context.stepParams, projection) <= llamaBlockScalarWindowMaxContextLength
    );
  }

  canUseScalarGpuLogitsPrefixWindowProjection(context, projection, slots) {
    const prefixTokenCount = context.stepParams.tokenCount - 1;
    return (
      context.execution.outputPolicy === 1 &&
      context.execution.outputPtr === 0 &&
      context.stepParams.tokenCount > 2 &&
      context.stepParams.tokenCount === llamaActiveTokenCount(context.execution) &&
      context.execution.tokens.length >= context.stepParams.tokenCount &&
      context.session.device.canBindStorageBuffers(this.requiredWindowGpuStorageBufferCount()) &&
      projection.hiddenSize <= 256 &&
      projection.ffnSize <= 256 &&
      projection.qProj.rows <= 256 &&
      slots.k.stride <= 256 &&
      this.scalarWindowScoreSlotCount(context.stepParams, projection, prefixTokenCount) <= llamaBlockScalarWindowMaxContextLength
    );
  }

  prefixNoOutputContext(context, tokenCount) {
    return this.writePrefixNoOutputContext(this.prefixContextScratch, context, tokenCount);
  }

  writePrefixNoOutputContext(target, context, tokenCount) {
    target.runtime = context.runtime;
    target.session = context.session;
    const execution = target.execution;
    execution.descPtr = context.execution.descPtr ?? 0;
    execution.outputLen = 0;
    execution.outputPolicy = 0;
    execution.outputPtr = 0;
    execution.tokens = context.execution.tokens;
    execution.tokensLen = tokenCount;
    execution.tokensPtr = context.execution.tokensPtr ?? 0;
    const stepParams = target.stepParams;
    stepParams.contextLength = context.stepParams.contextLength;
    stepParams.endPosition = context.stepParams.startPosition + tokenCount;
    stepParams.kind = context.stepParams.kind ?? "llama-token-window";
    stepParams.logitsLength = 0;
    stepParams.outputKind = "none";
    stepParams.outputPolicy = 0;
    stepParams.requestedOutputLength = 0;
    stepParams.startPosition = context.stepParams.startPosition;
    stepParams.tokenCount = tokenCount;
    return target;
  }

  ensureScalarGpuPipeline(device) {
    if (!device.canCreateGpuBuffers()) {
      throw new Error("browser WebGPU device is unavailable");
    }
    if (this.scalarPipeline && this.scalarPipelineDevice === device.device) return this.scalarPipeline;
    const gpuDevice = device.device;
    const shader = gpuDevice.createShaderModule({
      label: "zgml.wasm.webgpu.llama-block-projection.scalar",
      code: `
struct Params {
  position: u32,
  seq: u32,
  hidden: u32,
  stride: u32,
  vocab: u32,
  context_length: u32,
  token: u32,
  output_policy: u32,
  ffn: u32,
  epsilon: f32,
  activation_policy: u32,
  input_activation_policy: u32,
  q_stride: u32,
  head_size: u32,
  query_heads_per_kv: u32,
  rope_base: f32,
  rope_scale: f32,
  embedding_offset: u32,
  input_norm_offset: u32,
  q_proj_offset: u32,
  k_proj_offset: u32,
  v_proj_offset: u32,
  o_proj_offset: u32,
  post_norm_offset: u32,
  gate_proj_offset: u32,
  up_proj_offset: u32,
  down_proj_offset: u32,
  final_norm_offset: u32,
  lm_head_offset: u32,
  rope_kind: u32,
  rope_low_freq_factor: f32,
  rope_high_freq_factor: f32,
  rope_original_context_length: f32,
  sliding_window: u32,
  q_bias_offset: u32,
  k_bias_offset: u32,
  v_bias_offset: u32,
  o_bias_offset: u32,
  gate_bias_offset: u32,
  up_bias_offset: u32,
	  down_bias_offset: u32,
	  lm_head_bias_offset: u32,
	  bias_mask: u32,
	  use_rope: u32,
	  q_norm_offset: u32,
	  k_norm_offset: u32,
	  qk_norm: u32,
	};

@group(0) @binding(0) var<storage, read> model: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;
@group(0) @binding(4) var<storage, read_write> activation: array<f32>;
@group(0) @binding(5) var<storage, read> activation_input: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

fn rope_inv_frequency(pair: u32) -> f32 {
  let inv_freq = 1.0 / pow(params.rope_base, f32(pair * 2u) / f32(params.head_size));
  if (params.rope_kind == 1u) {
    return inv_freq / params.rope_scale;
  }
  if (params.rope_kind == 2u) {
    let wavelen = 6.283185307179586 / inv_freq;
    let low_freq_wavelen = params.rope_original_context_length / params.rope_low_freq_factor;
    let high_freq_wavelen = params.rope_original_context_length / params.rope_high_freq_factor;
    if (wavelen > low_freq_wavelen) {
      return inv_freq / params.rope_scale;
    }
    if (wavelen < high_freq_wavelen) {
      return inv_freq;
    }
    let smooth_factor = (params.rope_original_context_length / wavelen - params.rope_low_freq_factor) / (params.rope_high_freq_factor - params.rope_low_freq_factor);
    return (1.0 - smooth_factor) * inv_freq / params.rope_scale + smooth_factor * inv_freq;
  }
  return inv_freq;
}

fn rope_component(lo: f32, hi: f32, pair: u32, high: bool) -> f32 {
  let freq = f32(params.position) * rope_inv_frequency(pair);
  let c = cos(freq);
  let s = sin(freq);
  if (high) {
    return hi * c + lo * s;
  }
  return lo * c - hi * s;
}

fn silu_value(value: f32) -> f32 {
  return value / (1.0 + exp(-value));
}

fn attention_start(seq: u32) -> u32 {
  if (params.sliding_window == 0u || seq <= params.sliding_window) {
    return 0u;
  }
  return seq - params.sliding_window;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) {
    return;
  }
  let max_score_slots = select(params.seq, min(params.seq, params.sliding_window), params.sliding_window != 0u);
  if (
    params.hidden > 256u ||
    params.stride > 256u ||
    params.q_stride > 256u ||
    params.ffn > 256u ||
    params.seq > params.context_length ||
    max_score_slots > ${llamaBlockScalarWindowMaxContextLength}u
  ) {
    return;
  }

  var residual: array<f32, 256>;
  var input: array<f32, 256>;
  var sum = 0.0;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    let value = select(
      model[params.embedding_offset + params.token * params.hidden + col],
      activation_input[params.position * params.hidden + col],
      params.input_activation_policy == 1u,
    );
    residual[col] = value;
    sum = sum + value * value;
  }
  let inv_input_rms = inverseSqrt(sum / f32(params.hidden) + params.epsilon);
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    input[col] = residual[col] * model[params.input_norm_offset + col] * inv_input_rms;
  }

  var q: array<f32, 256>;
  var k: array<f32, 256>;
  var v: array<f32, 256>;
  for (var row = 0u; row < params.q_stride; row = row + 1u) {
    var acc = 0.0;
    let base = row * params.hidden;
    for (var col = 0u; col < params.hidden; col = col + 1u) {
      acc = acc + model[params.q_proj_offset + base + col] * input[col];
    }
    if ((params.bias_mask & 1u) != 0u) {
      acc = acc + model[params.q_bias_offset + row];
    }
    q[row] = acc;
  }
  for (var row = 0u; row < params.stride; row = row + 1u) {
    var k_acc = 0.0;
    var v_acc = 0.0;
    let base = row * params.hidden;
    for (var col = 0u; col < params.hidden; col = col + 1u) {
      k_acc = k_acc + model[params.k_proj_offset + base + col] * input[col];
      v_acc = v_acc + model[params.v_proj_offset + base + col] * input[col];
    }
    if ((params.bias_mask & 2u) != 0u) {
      k_acc = k_acc + model[params.k_bias_offset + row];
    }
    if ((params.bias_mask & 4u) != 0u) {
      v_acc = v_acc + model[params.v_bias_offset + row];
    }
    k[row] = k_acc;
    v[row] = v_acc;
  }

	  if (params.qk_norm != 0u) {
	    for (var head = 0u; head < params.q_stride; head = head + params.head_size) {
	      var sum = 0.0;
	      for (var col = 0u; col < params.head_size; col = col + 1u) {
	        let value = q[head + col];
	        sum = sum + value * value;
	      }
	      let inv_rms = inverseSqrt(sum / f32(params.head_size) + params.epsilon);
	      for (var col = 0u; col < params.head_size; col = col + 1u) {
	        q[head + col] = q[head + col] * model[params.q_norm_offset + col] * inv_rms;
	      }
	    }
	    for (var head = 0u; head < params.stride; head = head + params.head_size) {
	      var sum = 0.0;
	      for (var col = 0u; col < params.head_size; col = col + 1u) {
	        let value = k[head + col];
	        sum = sum + value * value;
	      }
	      let inv_rms = inverseSqrt(sum / f32(params.head_size) + params.epsilon);
	      for (var col = 0u; col < params.head_size; col = col + 1u) {
	        k[head + col] = k[head + col] * model[params.k_norm_offset + col] * inv_rms;
	      }
	    }
	  }

	  if (params.use_rope != 0u) {
	    let half = params.head_size / 2u;
	    for (var head = 0u; head < params.q_stride; head = head + params.head_size) {
	      for (var pair = 0u; pair < half; pair = pair + 1u) {
	        let lo = q[head + pair];
	        let hi = q[head + pair + half];
	        q[head + pair] = rope_component(lo, hi, pair, false);
	        q[head + pair + half] = rope_component(lo, hi, pair, true);
	      }
	    }
	    for (var head = 0u; head < params.stride; head = head + params.head_size) {
	      for (var pair = 0u; pair < half; pair = pair + 1u) {
	        let lo = k[head + pair];
	        let hi = k[head + pair + half];
	        k[head + pair] = rope_component(lo, hi, pair, false);
	        k[head + pair + half] = rope_component(lo, hi, pair, true);
	      }
	    }
	  }
  for (var row = 0u; row < params.stride; row = row + 1u) {
    let cache_offset = params.position * params.stride + row;
    k_cache[cache_offset] = k[row];
    v_cache[cache_offset] = v[row];
  }
  if (params.activation_policy == 0u && params.output_policy == 0u) {
    return;
  }

  var attention: array<f32, 256>;
  var scores: array<f32, ${llamaBlockScalarWindowMaxContextLength}>;
  let scale = inverseSqrt(f32(params.head_size));
  let start = attention_start(params.seq);
  for (var row = 0u; row < params.q_stride; row = row + 1u) {
    let q_head = row / params.head_size;
    let kv_head = q_head / params.query_heads_per_kv;
    let q_base = q_head * params.head_size;
    let kv_base = kv_head * params.head_size;
    let local = row - q_base;
    var max_score = -3.4028234663852886e38;
    for (var pos = start; pos < params.seq; pos = pos + 1u) {
      var dot = 0.0;
      for (var col = 0u; col < params.head_size; col = col + 1u) {
        let k_value = select(
          k_cache[pos * params.stride + kv_base + col],
          k[kv_base + col],
          pos == params.position,
        );
        dot = dot + q[q_base + col] * k_value;
      }
      let score = dot * scale;
      scores[pos - start] = score;
      max_score = max(max_score, score);
    }
    var denom = 0.0;
    var acc = 0.0;
    for (var pos = start; pos < params.seq; pos = pos + 1u) {
      let weight = exp(scores[pos - start] - max_score);
      let v_value = select(
        v_cache[pos * params.stride + kv_base + local],
        v[kv_base + local],
        pos == params.position,
      );
      denom = denom + weight;
      acc = acc + weight * v_value;
    }
    attention[row] = select(0.0, acc / denom, denom > 0.0);
  }

  var after_attention: array<f32, 256>;
  for (var row = 0u; row < params.hidden; row = row + 1u) {
    var acc = 0.0;
    let base = row * params.q_stride;
    for (var col = 0u; col < params.q_stride; col = col + 1u) {
      acc = acc + model[params.o_proj_offset + base + col] * attention[col];
    }
    if ((params.bias_mask & 8u) != 0u) {
      acc = acc + model[params.o_bias_offset + row];
    }
    after_attention[row] = residual[row] + acc;
  }

  var post_sum = 0.0;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    post_sum = post_sum + after_attention[col] * after_attention[col];
  }
  let inv_post_rms = inverseSqrt(post_sum / f32(params.hidden) + params.epsilon);
  var normed_ffn: array<f32, 256>;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    normed_ffn[col] = after_attention[col] * model[params.post_norm_offset + col] * inv_post_rms;
  }

  var ffn_hidden: array<f32, 256>;
  for (var row = 0u; row < params.ffn; row = row + 1u) {
    var gate_acc = 0.0;
    var up_acc = 0.0;
    let base = row * params.hidden;
    for (var col = 0u; col < params.hidden; col = col + 1u) {
      gate_acc = gate_acc + model[params.gate_proj_offset + base + col] * normed_ffn[col];
      up_acc = up_acc + model[params.up_proj_offset + base + col] * normed_ffn[col];
    }
    if ((params.bias_mask & 16u) != 0u) {
      gate_acc = gate_acc + model[params.gate_bias_offset + row];
    }
    if ((params.bias_mask & 32u) != 0u) {
      up_acc = up_acc + model[params.up_bias_offset + row];
    }
    ffn_hidden[row] = silu_value(gate_acc) * up_acc;
  }

  var block_hidden_values: array<f32, 256>;
  for (var row = 0u; row < params.hidden; row = row + 1u) {
    var acc = 0.0;
    let base = row * params.ffn;
    for (var col = 0u; col < params.ffn; col = col + 1u) {
      acc = acc + model[params.down_proj_offset + base + col] * ffn_hidden[col];
    }
    if ((params.bias_mask & 64u) != 0u) {
      acc = acc + model[params.down_bias_offset + row];
    }
    let hidden = after_attention[row] + acc;
    block_hidden_values[row] = hidden;
    if (params.activation_policy == 1u) {
      activation[params.position * params.hidden + row] = hidden;
    }
  }

  if (params.output_policy == 1u) {
    var final_sum = 0.0;
    for (var col = 0u; col < params.hidden; col = col + 1u) {
      final_sum = final_sum + block_hidden_values[col] * block_hidden_values[col];
    }
    let inv_final_rms = inverseSqrt(final_sum / f32(params.hidden) + params.epsilon);
    for (var row = 0u; row < params.vocab; row = row + 1u) {
      var acc = 0.0;
      let base = row * params.hidden;
      for (var col = 0u; col < params.hidden; col = col + 1u) {
        let hidden = block_hidden_values[col] * model[params.final_norm_offset + col] * inv_final_rms;
        acc = acc + model[params.lm_head_offset + base + col] * hidden;
      }
      if ((params.bias_mask & 128u) != 0u) {
        acc = acc + model[params.lm_head_bias_offset + row];
      }
      logits[row] = acc;
    }
  }
}
`,
    });
    this.scalarPipeline = gpuDevice.createComputePipeline({
      label: "zgml.wasm.webgpu.llama-block-projection.scalar",
      layout: "auto",
      compute: {
        module: shader,
        entryPoint: "main",
      },
    });
    const usage = gpuBufferUsageFlags();
    if (!this.paramsBuffer) {
      this.paramsBuffer = gpuDevice.createBuffer({
        label: "zgml.wasm.webgpu.llama-block-projection.params",
        size: 192,
        usage: usage.uniform | usage.copyDst,
      });
    }
    this.scalarPipelineDevice = gpuDevice;
    return this.scalarPipeline;
  }

  ensureWindowScalarGpuPipeline(device) {
    if (!device.canCreateGpuBuffers()) {
      throw new Error("browser WebGPU device is unavailable");
    }
    if (this.windowScalarPipeline && this.windowScalarPipelineDevice === device.device) return this.windowScalarPipeline;
    const gpuDevice = device.device;
    const shader = gpuDevice.createShaderModule({
      label: "zgml.wasm.webgpu.llama-block-projection.scalar-window",
      code: `
struct Params {
  start_position: u32,
  token_count: u32,
  hidden: u32,
  stride: u32,
  vocab: u32,
  context_length: u32,
  _pad0: u32,
  _pad1: u32,
  ffn: u32,
  epsilon: f32,
  activation_policy: u32,
  input_activation_policy: u32,
  q_stride: u32,
  head_size: u32,
  query_heads_per_kv: u32,
  rope_base: f32,
  rope_scale: f32,
  embedding_offset: u32,
  input_norm_offset: u32,
  q_proj_offset: u32,
  k_proj_offset: u32,
  v_proj_offset: u32,
  o_proj_offset: u32,
  post_norm_offset: u32,
  gate_proj_offset: u32,
  up_proj_offset: u32,
  down_proj_offset: u32,
  final_norm_offset: u32,
  lm_head_offset: u32,
  rope_kind: u32,
  rope_low_freq_factor: f32,
  rope_high_freq_factor: f32,
  rope_original_context_length: f32,
  sliding_window: u32,
  q_bias_offset: u32,
  k_bias_offset: u32,
  v_bias_offset: u32,
  o_bias_offset: u32,
  gate_bias_offset: u32,
  up_bias_offset: u32,
	  down_bias_offset: u32,
	  lm_head_bias_offset: u32,
	  bias_mask: u32,
	  use_rope: u32,
	  q_norm_offset: u32,
	  k_norm_offset: u32,
	  qk_norm: u32,
	};

@group(0) @binding(0) var<storage, read> model: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> activation: array<f32>;
@group(0) @binding(4) var<storage, read> activation_input: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read> tokens: array<u32>;

fn rope_inv_frequency(pair: u32) -> f32 {
  let inv_freq = 1.0 / pow(params.rope_base, f32(pair * 2u) / f32(params.head_size));
  if (params.rope_kind == 1u) {
    return inv_freq / params.rope_scale;
  }
  if (params.rope_kind == 2u) {
    let wavelen = 6.283185307179586 / inv_freq;
    let low_freq_wavelen = params.rope_original_context_length / params.rope_low_freq_factor;
    let high_freq_wavelen = params.rope_original_context_length / params.rope_high_freq_factor;
    if (wavelen > low_freq_wavelen) {
      return inv_freq / params.rope_scale;
    }
    if (wavelen < high_freq_wavelen) {
      return inv_freq;
    }
    let smooth_factor = (params.rope_original_context_length / wavelen - params.rope_low_freq_factor) / (params.rope_high_freq_factor - params.rope_low_freq_factor);
    return (1.0 - smooth_factor) * inv_freq / params.rope_scale + smooth_factor * inv_freq;
  }
  return inv_freq;
}

fn rope_component(lo: f32, hi: f32, pair: u32, high: bool, position: u32) -> f32 {
  let freq = f32(position) * rope_inv_frequency(pair);
  let c = cos(freq);
  let s = sin(freq);
  if (high) {
    return hi * c + lo * s;
  }
  return lo * c - hi * s;
}

fn silu_value(value: f32) -> f32 {
  return value / (1.0 + exp(-value));
}

fn attention_start(seq: u32) -> u32 {
  if (params.sliding_window == 0u || seq <= params.sliding_window) {
    return 0u;
  }
  return seq - params.sliding_window;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) {
    return;
  }
  let max_seq = min(params.start_position + params.token_count, params.context_length);
  let max_score_slots = select(max_seq, min(max_seq, params.sliding_window), params.sliding_window != 0u);
  if (
    params.hidden > 256u ||
    params.stride > 256u ||
    params.q_stride > 256u ||
    params.ffn > 256u ||
    max_score_slots > ${llamaBlockScalarWindowMaxContextLength}u
  ) {
    return;
  }

  for (var token_index = 0u; token_index < params.token_count; token_index = token_index + 1u) {
    let position = params.start_position + token_index;
    if (position >= params.context_length) {
      continue;
    }
    let seq = position + 1u;
    let token = tokens[token_index];

    var residual: array<f32, 256>;
    var input: array<f32, 256>;
    var sum = 0.0;
    for (var col = 0u; col < params.hidden; col = col + 1u) {
      let value = select(
        model[params.embedding_offset + token * params.hidden + col],
        activation_input[position * params.hidden + col],
        params.input_activation_policy == 1u,
      );
      residual[col] = value;
      sum = sum + value * value;
    }
    let inv_input_rms = inverseSqrt(sum / f32(params.hidden) + params.epsilon);
    for (var col = 0u; col < params.hidden; col = col + 1u) {
      input[col] = residual[col] * model[params.input_norm_offset + col] * inv_input_rms;
    }

    var q: array<f32, 256>;
    var k: array<f32, 256>;
    var v: array<f32, 256>;
    for (var row = 0u; row < params.q_stride; row = row + 1u) {
      var acc = 0.0;
      let base = row * params.hidden;
      for (var col = 0u; col < params.hidden; col = col + 1u) {
        acc = acc + model[params.q_proj_offset + base + col] * input[col];
      }
      if ((params.bias_mask & 1u) != 0u) {
        acc = acc + model[params.q_bias_offset + row];
      }
      q[row] = acc;
    }
    for (var row = 0u; row < params.stride; row = row + 1u) {
      var k_acc = 0.0;
      var v_acc = 0.0;
      let base = row * params.hidden;
      for (var col = 0u; col < params.hidden; col = col + 1u) {
        k_acc = k_acc + model[params.k_proj_offset + base + col] * input[col];
        v_acc = v_acc + model[params.v_proj_offset + base + col] * input[col];
      }
      if ((params.bias_mask & 2u) != 0u) {
        k_acc = k_acc + model[params.k_bias_offset + row];
      }
      if ((params.bias_mask & 4u) != 0u) {
        v_acc = v_acc + model[params.v_bias_offset + row];
      }
      k[row] = k_acc;
      v[row] = v_acc;
    }

	    if (params.qk_norm != 0u) {
	      for (var head = 0u; head < params.q_stride; head = head + params.head_size) {
	        var sum = 0.0;
	        for (var col = 0u; col < params.head_size; col = col + 1u) {
	          let value = q[head + col];
	          sum = sum + value * value;
	        }
	        let inv_rms = inverseSqrt(sum / f32(params.head_size) + params.epsilon);
	        for (var col = 0u; col < params.head_size; col = col + 1u) {
	          q[head + col] = q[head + col] * model[params.q_norm_offset + col] * inv_rms;
	        }
	      }
	      for (var head = 0u; head < params.stride; head = head + params.head_size) {
	        var sum = 0.0;
	        for (var col = 0u; col < params.head_size; col = col + 1u) {
	          let value = k[head + col];
	          sum = sum + value * value;
	        }
	        let inv_rms = inverseSqrt(sum / f32(params.head_size) + params.epsilon);
	        for (var col = 0u; col < params.head_size; col = col + 1u) {
	          k[head + col] = k[head + col] * model[params.k_norm_offset + col] * inv_rms;
	        }
	      }
	    }

	    if (params.use_rope != 0u) {
	      let half = params.head_size / 2u;
	      for (var head = 0u; head < params.q_stride; head = head + params.head_size) {
	        for (var pair = 0u; pair < half; pair = pair + 1u) {
	          let lo = q[head + pair];
	          let hi = q[head + pair + half];
	          q[head + pair] = rope_component(lo, hi, pair, false, position);
	          q[head + pair + half] = rope_component(lo, hi, pair, true, position);
	        }
	      }
	      for (var head = 0u; head < params.stride; head = head + params.head_size) {
	        for (var pair = 0u; pair < half; pair = pair + 1u) {
	          let lo = k[head + pair];
	          let hi = k[head + pair + half];
	          k[head + pair] = rope_component(lo, hi, pair, false, position);
	          k[head + pair + half] = rope_component(lo, hi, pair, true, position);
	        }
	      }
	    }
    for (var row = 0u; row < params.stride; row = row + 1u) {
      let cache_offset = position * params.stride + row;
      k_cache[cache_offset] = k[row];
      v_cache[cache_offset] = v[row];
    }
    if (params.activation_policy == 0u) {
      continue;
    }

    var attention: array<f32, 256>;
    var scores: array<f32, ${llamaBlockScalarWindowMaxContextLength}>;
    let scale = inverseSqrt(f32(params.head_size));
    let start = attention_start(seq);
    for (var row = 0u; row < params.q_stride; row = row + 1u) {
      let q_head = row / params.head_size;
      let kv_head = q_head / params.query_heads_per_kv;
      let q_base = q_head * params.head_size;
      let kv_base = kv_head * params.head_size;
      let local = row - q_base;
      var max_score = -3.4028234663852886e38;
      for (var pos = start; pos < seq; pos = pos + 1u) {
        var dot = 0.0;
        for (var col = 0u; col < params.head_size; col = col + 1u) {
          let k_value = select(
            k_cache[pos * params.stride + kv_base + col],
            k[kv_base + col],
            pos == position,
          );
          dot = dot + q[q_base + col] * k_value;
        }
        let score = dot * scale;
        scores[pos - start] = score;
        max_score = max(max_score, score);
      }
      var denom = 0.0;
      var acc = 0.0;
      for (var pos = start; pos < seq; pos = pos + 1u) {
        let weight = exp(scores[pos - start] - max_score);
        let v_value = select(
          v_cache[pos * params.stride + kv_base + local],
          v[kv_base + local],
          pos == position,
        );
        denom = denom + weight;
        acc = acc + weight * v_value;
      }
      attention[row] = select(0.0, acc / denom, denom > 0.0);
    }

    var after_attention: array<f32, 256>;
    for (var row = 0u; row < params.hidden; row = row + 1u) {
      var acc = 0.0;
      let base = row * params.q_stride;
      for (var col = 0u; col < params.q_stride; col = col + 1u) {
        acc = acc + model[params.o_proj_offset + base + col] * attention[col];
      }
      if ((params.bias_mask & 8u) != 0u) {
        acc = acc + model[params.o_bias_offset + row];
      }
      after_attention[row] = residual[row] + acc;
    }

    var post_sum = 0.0;
    for (var col = 0u; col < params.hidden; col = col + 1u) {
      post_sum = post_sum + after_attention[col] * after_attention[col];
    }
    let inv_post_rms = inverseSqrt(post_sum / f32(params.hidden) + params.epsilon);
    var normed_ffn: array<f32, 256>;
    for (var col = 0u; col < params.hidden; col = col + 1u) {
      normed_ffn[col] = after_attention[col] * model[params.post_norm_offset + col] * inv_post_rms;
    }

    var ffn_hidden: array<f32, 256>;
    for (var row = 0u; row < params.ffn; row = row + 1u) {
      var gate_acc = 0.0;
      var up_acc = 0.0;
      let base = row * params.hidden;
      for (var col = 0u; col < params.hidden; col = col + 1u) {
        gate_acc = gate_acc + model[params.gate_proj_offset + base + col] * normed_ffn[col];
        up_acc = up_acc + model[params.up_proj_offset + base + col] * normed_ffn[col];
      }
      if ((params.bias_mask & 16u) != 0u) {
        gate_acc = gate_acc + model[params.gate_bias_offset + row];
      }
      if ((params.bias_mask & 32u) != 0u) {
        up_acc = up_acc + model[params.up_bias_offset + row];
      }
      ffn_hidden[row] = silu_value(gate_acc) * up_acc;
    }

    for (var row = 0u; row < params.hidden; row = row + 1u) {
      var acc = 0.0;
      let base = row * params.ffn;
      for (var col = 0u; col < params.ffn; col = col + 1u) {
        acc = acc + model[params.down_proj_offset + base + col] * ffn_hidden[col];
      }
      if ((params.bias_mask & 64u) != 0u) {
        acc = acc + model[params.down_bias_offset + row];
      }
      activation[position * params.hidden + row] = after_attention[row] + acc;
    }
  }
}
`,
    });
    this.windowScalarPipeline = gpuDevice.createComputePipeline({
      label: "zgml.wasm.webgpu.llama-block-projection.scalar-window",
      layout: "auto",
      compute: {
        module: shader,
        entryPoint: "main",
      },
    });
    const usage = gpuBufferUsageFlags();
    if (!this.paramsBuffer) {
      this.paramsBuffer = gpuDevice.createBuffer({
        label: "zgml.wasm.webgpu.llama-block-projection.params",
        size: 192,
        usage: usage.uniform | usage.copyDst,
      });
    }
    this.windowScalarPipelineDevice = gpuDevice;
    return this.windowScalarPipeline;
  }

  ensureGpuPipeline(device) {
    if (!device.canCreateGpuBuffers()) {
      throw new Error("browser WebGPU device is unavailable");
    }
    if (this.pipeline && this.pipelineDevice === device.device) return this.pipeline;
    if (this.paramsBuffer && typeof this.paramsBuffer.destroy === "function") {
      this.paramsBuffer.destroy();
    }
    const gpuDevice = device.device;
    const shader = gpuDevice.createShaderModule({
      label: "zgml.wasm.webgpu.llama-block-projection",
      code: `
struct Params {
  position: u32,
  seq: u32,
  hidden: u32,
  stride: u32,
  vocab: u32,
  context_length: u32,
  token: u32,
  output_policy: u32,
  ffn: u32,
  epsilon: f32,
  activation_policy: u32,
  input_activation_policy: u32,
  q_stride: u32,
  head_size: u32,
  query_heads_per_kv: u32,
  rope_base: f32,
  rope_scale: f32,
  embedding_offset: u32,
  input_norm_offset: u32,
  q_proj_offset: u32,
  k_proj_offset: u32,
  v_proj_offset: u32,
  o_proj_offset: u32,
  post_norm_offset: u32,
  gate_proj_offset: u32,
  up_proj_offset: u32,
  down_proj_offset: u32,
  final_norm_offset: u32,
  lm_head_offset: u32,
  rope_kind: u32,
  rope_low_freq_factor: f32,
  rope_high_freq_factor: f32,
  rope_original_context_length: f32,
  sliding_window: u32,
  q_bias_offset: u32,
  k_bias_offset: u32,
  v_bias_offset: u32,
  o_bias_offset: u32,
  gate_bias_offset: u32,
  up_bias_offset: u32,
	  down_bias_offset: u32,
	  lm_head_bias_offset: u32,
	  bias_mask: u32,
	  use_rope: u32,
	  q_norm_offset: u32,
	  k_norm_offset: u32,
	  qk_norm: u32,
	};

@group(0) @binding(0) var<storage, read> model: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;
@group(0) @binding(4) var<storage, read_write> activation: array<f32>;
@group(0) @binding(5) var<storage, read> activation_input: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

fn embedding_value(col: u32) -> f32 {
  if (params.input_activation_policy == 1u) {
    return activation_input[params.position * params.hidden + col];
  }
  return model[params.embedding_offset + params.token * params.hidden + col];
}

fn input_rms_scale() -> f32 {
  var sum = 0.0;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    let value = embedding_value(col);
    sum = sum + value * value;
  }
  return inverseSqrt(sum / f32(params.hidden) + params.epsilon);
}

fn normed_input(col: u32) -> f32 {
  return embedding_value(col) * model[params.input_norm_offset + col] * input_rms_scale();
}

fn project(weight_offset: u32, row: u32) -> f32 {
  var acc = 0.0;
  let base = row * params.hidden;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    let hidden = normed_input(col);
    acc = acc + model[weight_offset + base + col] * hidden;
  }
  return acc;
}

fn project_biased(weight_offset: u32, bias_offset: u32, mask: u32, row: u32) -> f32 {
  var acc = project(weight_offset, row);
  if ((params.bias_mask & mask) != 0u) {
    acc = acc + model[bias_offset + row];
  }
  return acc;
}

fn q_projected(row: u32) -> f32 {
  let value = project_biased(params.q_proj_offset, params.q_bias_offset, 1u, row);
  if (params.qk_norm == 0u) {
    return value;
  }
  let local = row % params.head_size;
  let base = row - local;
  var sum = 0.0;
  for (var col = 0u; col < params.head_size; col = col + 1u) {
    let head_value = project_biased(params.q_proj_offset, params.q_bias_offset, 1u, base + col);
    sum = sum + head_value * head_value;
  }
  let inv_rms = inverseSqrt(sum / f32(params.head_size) + params.epsilon);
  return value * model[params.q_norm_offset + local] * inv_rms;
}

fn k_projected(row: u32) -> f32 {
  let value = project_biased(params.k_proj_offset, params.k_bias_offset, 2u, row);
  if (params.qk_norm == 0u) {
    return value;
  }
  let local = row % params.head_size;
  let base = row - local;
  var sum = 0.0;
  for (var col = 0u; col < params.head_size; col = col + 1u) {
    let head_value = project_biased(params.k_proj_offset, params.k_bias_offset, 2u, base + col);
    sum = sum + head_value * head_value;
  }
  let inv_rms = inverseSqrt(sum / f32(params.head_size) + params.epsilon);
  return value * model[params.k_norm_offset + local] * inv_rms;
}

fn rope_inv_frequency(pair: u32) -> f32 {
  let inv_freq = 1.0 / pow(params.rope_base, f32(pair * 2u) / f32(params.head_size));
  if (params.rope_kind == 1u) {
    return inv_freq / params.rope_scale;
  }
  if (params.rope_kind == 2u) {
    let wavelen = 6.283185307179586 / inv_freq;
    let low_freq_wavelen = params.rope_original_context_length / params.rope_low_freq_factor;
    let high_freq_wavelen = params.rope_original_context_length / params.rope_high_freq_factor;
    if (wavelen > low_freq_wavelen) {
      return inv_freq / params.rope_scale;
    }
    if (wavelen < high_freq_wavelen) {
      return inv_freq;
    }
    let smooth_factor = (params.rope_original_context_length / wavelen - params.rope_low_freq_factor) / (params.rope_high_freq_factor - params.rope_low_freq_factor);
    return (1.0 - smooth_factor) * inv_freq / params.rope_scale + smooth_factor * inv_freq;
  }
  return inv_freq;
}

fn rope_component(lo: f32, hi: f32, pair: u32, high: bool) -> f32 {
  let freq = f32(params.position) * rope_inv_frequency(pair);
  let c = cos(freq);
  let s = sin(freq);
  if (high) {
    return hi * c + lo * s;
  }
  return lo * c - hi * s;
}

	fn rotated_q(row: u32) -> f32 {
	  if (params.use_rope == 0u) {
	    return q_projected(row);
	  }
	  let local = row % params.head_size;
  let base = row - local;
  let half = params.head_size / 2u;
  if (local < half) {
    return rope_component(q_projected(row), q_projected(row + half), local, false);
  }
  let pair = local - half;
  return rope_component(q_projected(base + pair), q_projected(row), pair, true);
}

	fn rotated_current_k(row: u32) -> f32 {
	  if (params.use_rope == 0u) {
	    return k_projected(row);
	  }
	  let local = row % params.head_size;
  let base = row - local;
  let half = params.head_size / 2u;
  if (local < half) {
    return rope_component(k_projected(row), k_projected(row + half), local, false);
  }
  let pair = local - half;
  return rope_component(k_projected(base + pair), k_projected(row), pair, true);
}

fn current_v(row: u32) -> f32 {
  return project_biased(params.v_proj_offset, params.v_bias_offset, 4u, row);
}

fn cache_k(position: u32, row: u32) -> f32 {
  if (position == params.position) {
    return rotated_current_k(row);
  }
  return k_cache[position * params.stride + row];
}

fn cache_v(position: u32, row: u32) -> f32 {
  if (position == params.position) {
    return current_v(row);
  }
  return v_cache[position * params.stride + row];
}

fn attention_start(seq: u32) -> u32 {
  if (params.sliding_window == 0u || seq <= params.sliding_window) {
    return 0u;
  }
  return seq - params.sliding_window;
}

fn attention_value(row: u32) -> f32 {
  let q_head = row / params.head_size;
  let kv_head = q_head / params.query_heads_per_kv;
  let q_base = q_head * params.head_size;
  let kv_base = kv_head * params.head_size;
  let local = row - q_base;
  let scale = inverseSqrt(f32(params.head_size));
  let start = attention_start(params.seq);
  var max_score = -3.4028234663852886e38;
  for (var pos = start; pos < params.seq; pos = pos + 1u) {
    var dot = 0.0;
    for (var col = 0u; col < params.head_size; col = col + 1u) {
      dot = dot + rotated_q(q_base + col) * cache_k(pos, kv_base + col);
    }
    let score = dot * scale;
    max_score = max(max_score, score);
  }
  var denom = 0.0;
  var acc = 0.0;
  for (var pos = start; pos < params.seq; pos = pos + 1u) {
    var dot = 0.0;
    for (var col = 0u; col < params.head_size; col = col + 1u) {
      dot = dot + rotated_q(q_base + col) * cache_k(pos, kv_base + col);
    }
    let weight = exp(dot * scale - max_score);
    denom = denom + weight;
    acc = acc + weight * cache_v(pos, kv_base + local);
  }
  if (denom <= 0.0) {
    return 0.0;
  }
  return acc / denom;
}

fn attention_projected(row: u32) -> f32 {
  var acc = 0.0;
  let base = row * params.q_stride;
  for (var col = 0u; col < params.q_stride; col = col + 1u) {
    acc = acc + model[params.o_proj_offset + base + col] * attention_value(col);
  }
  if ((params.bias_mask & 8u) != 0u) {
    acc = acc + model[params.o_bias_offset + row];
  }
  return acc;
}

fn after_attention(row: u32) -> f32 {
  return embedding_value(row) + attention_projected(row);
}

fn post_rms_scale() -> f32 {
  var sum = 0.0;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    let value = after_attention(col);
    sum = sum + value * value;
  }
  return inverseSqrt(sum / f32(params.hidden) + params.epsilon);
}

fn normed_ffn(col: u32) -> f32 {
  return after_attention(col) * model[params.post_norm_offset + col] * post_rms_scale();
}

fn gate_value(row: u32) -> f32 {
  var acc = 0.0;
  let base = row * params.hidden;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    acc = acc + model[params.gate_proj_offset + base + col] * normed_ffn(col);
  }
  if ((params.bias_mask & 16u) != 0u) {
    acc = acc + model[params.gate_bias_offset + row];
  }
  return acc;
}

fn up_value(row: u32) -> f32 {
  var acc = 0.0;
  let base = row * params.hidden;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    acc = acc + model[params.up_proj_offset + base + col] * normed_ffn(col);
  }
  if ((params.bias_mask & 32u) != 0u) {
    acc = acc + model[params.up_bias_offset + row];
  }
  return acc;
}

fn silu_value(value: f32) -> f32 {
  return value / (1.0 + exp(-value));
}

fn ffn_value(row: u32) -> f32 {
  var acc = 0.0;
  let base = row * params.ffn;
  for (var col = 0u; col < params.ffn; col = col + 1u) {
    let hidden = silu_value(gate_value(col)) * up_value(col);
    acc = acc + model[params.down_proj_offset + base + col] * hidden;
  }
  if ((params.bias_mask & 64u) != 0u) {
    acc = acc + model[params.down_bias_offset + row];
  }
  return acc;
}

fn block_hidden(row: u32) -> f32 {
  return after_attention(row) + ffn_value(row);
}

fn final_rms_scale() -> f32 {
  var sum = 0.0;
  for (var col = 0u; col < params.hidden; col = col + 1u) {
    let value = block_hidden(col);
    sum = sum + value * value;
  }
  return inverseSqrt(sum / f32(params.hidden) + params.epsilon);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row < params.stride) {
    let cache_offset = params.position * params.stride + row;
    k_cache[cache_offset] = rotated_current_k(row);
    v_cache[cache_offset] = current_v(row);
  }
  if (params.activation_policy == 1u && row < params.hidden) {
    activation[params.position * params.hidden + row] = block_hidden(row);
  }
  if (params.output_policy == 1u && row < params.vocab) {
    let scale = final_rms_scale();
    let lm_offset = row * params.hidden;
    var acc = 0.0;
    for (var col = 0u; col < params.hidden; col = col + 1u) {
      let hidden = block_hidden(col) * model[params.final_norm_offset + col] * scale;
      acc = acc + model[params.lm_head_offset + lm_offset + col] * hidden;
    }
    if ((params.bias_mask & 128u) != 0u) {
      acc = acc + model[params.lm_head_bias_offset + row];
    }
    logits[row] = acc;
  }
}
`,
    });
    this.pipeline = gpuDevice.createComputePipeline({
      label: "zgml.wasm.webgpu.llama-block-projection",
      layout: "auto",
      compute: {
        module: shader,
        entryPoint: "main",
      },
    });
    const usage = gpuBufferUsageFlags();
    this.paramsBuffer = gpuDevice.createBuffer({
      label: "zgml.wasm.webgpu.llama-block-projection.params",
      size: 192,
      usage: usage.uniform | usage.copyDst,
    });
    this.pipelineDevice = gpuDevice;
    return this.pipeline;
  }

  ensureWindowTokensBuffer(device, byteLength) {
    if (
      this.tokensBuffer &&
      this.windowScalarPipelineDevice === device.device &&
      this.tokensBufferByteLength >= byteLength
    ) {
      return this.tokensBuffer;
    }
    if (this.tokensBuffer && typeof this.tokensBuffer.destroy === "function") {
      this.tokensBuffer.destroy();
    }
    const usage = gpuBufferUsageFlags();
    const capacity = growGpuByteCapacity(byteLength);
    this.tokensBuffer = device.device.createBuffer({
      label: "zgml.wasm.webgpu.llama-block-projection.tokens",
      size: capacity,
      usage: usage.storage | usage.copyDst,
    });
    this.tokensBufferByteLength = capacity;
    return this.tokensBuffer;
  }

  submitGpuWindowProjection(context, projection, slots) {
    const device = context.session.device;
    const gpuDevice = device.device;
    const pipeline = this.ensureWindowScalarGpuPipeline(device);
    const params = this.paramsBytes;
    const view = this.paramsView;
    view.setUint32(0, context.stepParams.startPosition, true);
    view.setUint32(4, context.stepParams.tokenCount, true);
    view.setUint32(8, projection.hiddenSize, true);
    view.setUint32(12, slots.k.stride, true);
    view.setUint32(16, projection.vocabSize, true);
    view.setUint32(20, context.stepParams.contextLength, true);
    view.setUint32(24, 0, true);
    view.setUint32(28, 0, true);
    view.setUint32(32, projection.ffnSize, true);
    view.setFloat32(36, projection.epsilon, true);
    view.setUint32(40, projection.activation == null ? 0 : 1, true);
    view.setUint32(44, projection.activationInput == null ? 0 : 1, true);
    view.setUint32(48, projection.qProj.rows, true);
    view.setUint32(52, projection.attentionHeadSize, true);
    view.setUint32(56, projection.queryHeadsPerKvHead, true);
    view.setFloat32(60, projection.ropeBase, true);
    view.setFloat32(64, projection.ropeScale, true);
    view.setUint32(68, projection.modelPack.offsets.embedding, true);
    view.setUint32(72, projection.modelPack.offsets.inputNorm, true);
    view.setUint32(76, projection.modelPack.offsets.qProj, true);
    view.setUint32(80, projection.modelPack.offsets.kProj, true);
    view.setUint32(84, projection.modelPack.offsets.vProj, true);
    view.setUint32(88, projection.modelPack.offsets.oProj, true);
    view.setUint32(92, projection.modelPack.offsets.postNorm, true);
    view.setUint32(96, projection.modelPack.offsets.gateProj, true);
    view.setUint32(100, projection.modelPack.offsets.upProj, true);
    view.setUint32(104, projection.modelPack.offsets.downProj, true);
    view.setUint32(108, projection.modelPack.offsets.norm, true);
    view.setUint32(112, projection.modelPack.offsets.lmHead, true);
    view.setUint32(116, projection.ropeKindId, true);
    view.setFloat32(120, projection.ropeLowFreqFactor, true);
    view.setFloat32(124, projection.ropeHighFreqFactor, true);
    view.setFloat32(128, projection.ropeOriginalContextLength, true);
    view.setUint32(132, projection.slidingWindow ?? 0, true);
    view.setUint32(136, projection.modelPack.offsets.qBias ?? 0, true);
    view.setUint32(140, projection.modelPack.offsets.kBias ?? 0, true);
    view.setUint32(144, projection.modelPack.offsets.vBias ?? 0, true);
    view.setUint32(148, projection.modelPack.offsets.oBias ?? 0, true);
    view.setUint32(152, projection.modelPack.offsets.gateBias ?? 0, true);
    view.setUint32(156, projection.modelPack.offsets.upBias ?? 0, true);
    view.setUint32(160, projection.modelPack.offsets.downBias ?? 0, true);
	    view.setUint32(164, projection.modelPack.offsets.lmHeadBias ?? 0, true);
	    view.setUint32(168, projection.biasMask ?? 0, true);
	    view.setUint32(172, projection.useRope ? 1 : 0, true);
	    view.setUint32(176, projection.modelPack.offsets.qNorm ?? 0, true);
	    view.setUint32(180, projection.modelPack.offsets.kNorm ?? 0, true);
	    view.setUint32(184, projection.qkProjectionNorm ? 1 : 0, true);
	    gpuDevice.queue.writeBuffer(this.paramsBuffer, 0, params);

    const tokensByteLength = context.stepParams.tokenCount * Uint32Array.BYTES_PER_ELEMENT;
    let tokenUpload = projection.tokenUpload;
    if (!tokenUpload || tokenUpload.length < context.stepParams.tokenCount) {
      tokenUpload = new Uint32Array(growGpuByteCapacity(tokensByteLength) / Uint32Array.BYTES_PER_ELEMENT);
      projection.tokenUpload = tokenUpload;
    }
    for (let i = 0; i < context.stepParams.tokenCount; i += 1) {
      tokenUpload[i] = context.execution.tokens[i];
    }
    const tokensBuffer = this.ensureWindowTokensBuffer(device, tokensByteLength);
    gpuDevice.queue.writeBuffer(tokensBuffer, 0, tokenUpload.buffer, tokenUpload.byteOffset, tokensByteLength);

    const activationDescriptor = projection.activation?.descriptor ?? projection.activationDummy.descriptor;
    const activationByteLength = projection.activation?.descriptor.byteLength ?? projection.activationDummy.descriptor.byteLength;
    const activationInputDescriptor = projection.activationInput?.descriptor ?? projection.activationInputDummy.descriptor;
    const activationInputByteLength = projection.activationInput?.descriptor.byteLength ?? projection.activationInputDummy.descriptor.byteLength;
    const bindGroupDeps = this.bindGroupDeps;
    bindGroupDeps[0] = this.paramsBuffer;
    bindGroupDeps[1] = tokensBuffer;
    bindGroupDeps[2] = projection.modelPack.descriptor;
    bindGroupDeps[3] = slots.k.descriptor;
    bindGroupDeps[4] = slots.v.descriptor;
    bindGroupDeps[5] = activationDescriptor;
    bindGroupDeps[6] = activationInputDescriptor;
    const bindGroup = projectionGpuBindGroup(
      projection,
      "block-window",
      device,
      pipeline,
      bindGroupDeps,
      (targetDevice) => targetDevice.createBindGroup({
        label: "zgml.wasm.webgpu.llama-block-projection.scalar-window",
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: storageBufferBinding(
              targetDevice,
              projection.modelPack.descriptor,
              "LLaMA block projection model pack",
              projection.modelPack.byteLength,
            ),
          },
          {
            binding: 1,
            resource: storageBufferBinding(
              targetDevice,
              slots.k.descriptor,
              "LLaMA block K cache",
              slots.k.descriptor.byteLength,
            ),
          },
          {
            binding: 2,
            resource: storageBufferBinding(
              targetDevice,
              slots.v.descriptor,
              "LLaMA block V cache",
              slots.v.descriptor.byteLength,
            ),
          },
          {
            binding: 3,
            resource: storageBufferBinding(
              targetDevice,
              activationDescriptor,
              projection.activation == null ? "LLaMA block activation dummy" : "LLaMA block activation",
              activationByteLength,
            ),
          },
          {
            binding: 4,
            resource: storageBufferBinding(
              targetDevice,
              activationInputDescriptor,
              projection.activationInput == null ? "LLaMA block activation input dummy" : "LLaMA block activation input",
              activationInputByteLength,
            ),
          },
          { binding: 5, resource: { buffer: this.paramsBuffer } },
          { binding: 6, resource: { buffer: tokensBuffer } },
        ],
      }),
    );
    const encoder = gpuDevice.createCommandEncoder({ label: "zgml.wasm.webgpu.llama-block-projection.scalar-window" });
    const pass = encoder.beginComputePass({ label: "zgml.wasm.webgpu.llama-block-projection.scalar-window" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    gpuDevice.queue.submit([encoder.finish()]);
    return 1;
  }

  submitGpuProjection(context, projection, slots, options = {}) {
    const device = context.session.device;
    const gpuDevice = device.device;
    const useScalarProjection = this.canUseScalarGpuProjection(context, projection, slots);
    const pipeline = useScalarProjection
      ? this.ensureScalarGpuPipeline(device)
      : this.ensureGpuPipeline(device);
    const position = options.position ?? context.stepParams.startPosition;
    const seq = options.seq ?? context.stepParams.endPosition;
    const token = options.token ?? llamaLastToken(context.execution);
    const outputPolicy = options.outputPolicy ?? context.execution.outputPolicy;
    const params = this.paramsBytes;
    const view = this.paramsView;
    view.setUint32(0, position, true);
    view.setUint32(4, seq, true);
    view.setUint32(8, projection.hiddenSize, true);
    view.setUint32(12, slots.k.stride, true);
    view.setUint32(16, projection.vocabSize, true);
    view.setUint32(20, context.stepParams.contextLength, true);
    view.setUint32(24, token, true);
    view.setUint32(28, outputPolicy, true);
    view.setUint32(32, projection.ffnSize, true);
    view.setFloat32(36, projection.epsilon, true);
    view.setUint32(40, projection.activation == null ? 0 : 1, true);
    view.setUint32(44, projection.activationInput == null ? 0 : 1, true);
    view.setUint32(48, projection.qProj.rows, true);
    view.setUint32(52, projection.attentionHeadSize, true);
    view.setUint32(56, projection.queryHeadsPerKvHead, true);
    view.setFloat32(60, projection.ropeBase, true);
    view.setFloat32(64, projection.ropeScale, true);
    view.setUint32(68, projection.modelPack.offsets.embedding, true);
    view.setUint32(72, projection.modelPack.offsets.inputNorm, true);
    view.setUint32(76, projection.modelPack.offsets.qProj, true);
    view.setUint32(80, projection.modelPack.offsets.kProj, true);
    view.setUint32(84, projection.modelPack.offsets.vProj, true);
    view.setUint32(88, projection.modelPack.offsets.oProj, true);
    view.setUint32(92, projection.modelPack.offsets.postNorm, true);
    view.setUint32(96, projection.modelPack.offsets.gateProj, true);
    view.setUint32(100, projection.modelPack.offsets.upProj, true);
    view.setUint32(104, projection.modelPack.offsets.downProj, true);
    view.setUint32(108, projection.modelPack.offsets.norm, true);
    view.setUint32(112, projection.modelPack.offsets.lmHead, true);
    view.setUint32(116, projection.ropeKindId, true);
    view.setFloat32(120, projection.ropeLowFreqFactor, true);
    view.setFloat32(124, projection.ropeHighFreqFactor, true);
    view.setFloat32(128, projection.ropeOriginalContextLength, true);
    view.setUint32(132, projection.slidingWindow ?? 0, true);
    view.setUint32(136, projection.modelPack.offsets.qBias ?? 0, true);
    view.setUint32(140, projection.modelPack.offsets.kBias ?? 0, true);
    view.setUint32(144, projection.modelPack.offsets.vBias ?? 0, true);
    view.setUint32(148, projection.modelPack.offsets.oBias ?? 0, true);
    view.setUint32(152, projection.modelPack.offsets.gateBias ?? 0, true);
    view.setUint32(156, projection.modelPack.offsets.upBias ?? 0, true);
    view.setUint32(160, projection.modelPack.offsets.downBias ?? 0, true);
	    view.setUint32(164, projection.modelPack.offsets.lmHeadBias ?? 0, true);
	    view.setUint32(168, projection.biasMask ?? 0, true);
	    view.setUint32(172, projection.useRope ? 1 : 0, true);
	    view.setUint32(176, projection.modelPack.offsets.qNorm ?? 0, true);
	    view.setUint32(180, projection.modelPack.offsets.kNorm ?? 0, true);
	    view.setUint32(184, projection.qkProjectionNorm ? 1 : 0, true);
	    gpuDevice.queue.writeBuffer(this.paramsBuffer, 0, params);
    const activationDescriptor = projection.activation?.descriptor ?? projection.activationDummy.descriptor;
    const activationByteLength = projection.activation?.descriptor.byteLength ?? projection.activationDummy.descriptor.byteLength;
    const activationInputDescriptor = projection.activationInput?.descriptor ?? projection.activationInputDummy.descriptor;
    const activationInputByteLength = projection.activationInput?.descriptor.byteLength ?? projection.activationInputDummy.descriptor.byteLength;
    const bindGroupDeps = this.bindGroupDeps;
    bindGroupDeps[0] = this.paramsBuffer;
    bindGroupDeps[1] = projection.modelPack.descriptor;
    bindGroupDeps[2] = context.session.bindings.output;
    bindGroupDeps[3] = slots.k.descriptor;
    bindGroupDeps[4] = slots.v.descriptor;
    bindGroupDeps[5] = activationDescriptor;
    bindGroupDeps[6] = activationInputDescriptor;
    const bindGroup = projectionGpuBindGroup(
      projection,
      "block",
      device,
      pipeline,
      bindGroupDeps,
      (targetDevice) => targetDevice.createBindGroup({
        label: "zgml.wasm.webgpu.llama-block-projection",
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: storageBufferBinding(
              targetDevice,
              projection.modelPack.descriptor,
              "LLaMA block projection model pack",
              projection.modelPack.byteLength,
            ),
          },
          {
            binding: 1,
            resource: storageBufferBinding(
              targetDevice,
              slots.k.descriptor,
              "LLaMA block K cache",
              slots.k.descriptor.byteLength,
            ),
          },
          {
            binding: 2,
            resource: storageBufferBinding(
              targetDevice,
              slots.v.descriptor,
              "LLaMA block V cache",
              slots.v.descriptor.byteLength,
            ),
          },
          {
            binding: 3,
            resource: storageBufferBinding(
              targetDevice,
              context.session.bindings.output,
              "LLaMA block logits",
              projection.vocabSize * Float32Array.BYTES_PER_ELEMENT,
            ),
          },
          {
            binding: 4,
            resource: storageBufferBinding(
              targetDevice,
              activationDescriptor,
              projection.activation == null ? "LLaMA block activation dummy" : "LLaMA block activation",
              activationByteLength,
            ),
          },
          {
            binding: 5,
            resource: storageBufferBinding(
              targetDevice,
              activationInputDescriptor,
              projection.activationInput == null ? "LLaMA block activation input dummy" : "LLaMA block activation input",
              activationInputByteLength,
            ),
          },
          { binding: 6, resource: { buffer: this.paramsBuffer } },
        ],
      }),
    );
    const encoder = gpuDevice.createCommandEncoder({ label: "zgml.wasm.webgpu.llama-block-projection" });
    const pass = encoder.beginComputePass({ label: "zgml.wasm.webgpu.llama-block-projection" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      useScalarProjection
        ? 1
        : Math.ceil(Math.max(projection.vocabSize, slots.k.stride, projection.hiddenSize, projection.qProj.rows) / 64),
    );
    pass.end();
    gpuDevice.queue.submit([encoder.finish()]);
    return 1;
  }

  destroy() {
    super.destroy();
    if (this.tokensBuffer && typeof this.tokensBuffer.destroy === "function") {
      this.tokensBuffer.destroy();
    }
    this.tokensBuffer = null;
    this.tokensBufferByteLength = 0;
    this.scalarPipeline = null;
    this.scalarPipelineDevice = null;
    this.windowScalarPipeline = null;
    this.windowScalarPipelineDevice = null;
    this.mockBlockProjectionScratch = createLlamaBlockProjectionMockScratch();
  }
}

function normalizeLlamaBlockPipelineSpecs(options = {}) {
  const source = Array.isArray(options.blocks)
    ? options.blocks
    : Array.isArray(options.layers)
      ? options.layers
      : Number.isSafeInteger(options.layerCount)
        ? Array.from({ length: options.layerCount }, (_unused, layer) => layer)
        : [options.layer ?? 0];
  if (!Array.isArray(source) || source.length === 0) {
    throw new Error("LLaMA block pipeline needs at least one block");
  }
  const sharedBlockOptions = {
    ...(options.blockOptions ?? {}),
  };
  if (options.attentionHeadSize !== undefined && options.attentionHeadSize !== null && sharedBlockOptions.attentionHeadSize === undefined) {
    sharedBlockOptions.attentionHeadSize = options.attentionHeadSize;
  }
  if (options.headSize !== undefined && options.headSize !== null && sharedBlockOptions.headSize === undefined) {
    sharedBlockOptions.headSize = options.headSize;
  }
  if (options.epsilon !== undefined && options.epsilon !== null && sharedBlockOptions.epsilon === undefined) {
    sharedBlockOptions.epsilon = options.epsilon;
  }
  if (options.rmsNormEpsilon !== undefined && options.rmsNormEpsilon !== null && sharedBlockOptions.rmsNormEpsilon === undefined) {
    sharedBlockOptions.rmsNormEpsilon = options.rmsNormEpsilon;
  }
  if (options.rmsNormEps !== undefined && options.rmsNormEps !== null && sharedBlockOptions.rmsNormEps === undefined) {
    sharedBlockOptions.rmsNormEps = options.rmsNormEps;
  }
  if (options.ropeBase !== undefined && options.ropeBase !== null && sharedBlockOptions.ropeBase === undefined) {
    sharedBlockOptions.ropeBase = options.ropeBase;
  }
  if (options.ropeTheta !== undefined && options.ropeTheta !== null && sharedBlockOptions.ropeTheta === undefined) {
    sharedBlockOptions.ropeTheta = options.ropeTheta;
  }
  if (options.ropeScale !== undefined && options.ropeScale !== null && sharedBlockOptions.ropeScale === undefined) {
    sharedBlockOptions.ropeScale = options.ropeScale;
  }
  if (options.rope_scale !== undefined && options.rope_scale !== null && sharedBlockOptions.rope_scale === undefined) {
    sharedBlockOptions.rope_scale = options.rope_scale;
  }
  if (options.ropeScalingFactor !== undefined && options.ropeScalingFactor !== null && sharedBlockOptions.ropeScalingFactor === undefined) {
    sharedBlockOptions.ropeScalingFactor = options.ropeScalingFactor;
  }
  if (options.rope_scaling_factor !== undefined && options.rope_scaling_factor !== null && sharedBlockOptions.rope_scaling_factor === undefined) {
    sharedBlockOptions.rope_scaling_factor = options.rope_scaling_factor;
  }
  for (const key of [
    "ropeKind",
    "ropeType",
    "rope_kind",
    "rope_type",
    "ropeLowFreqFactor",
    "rope_low_freq_factor",
    "ropeHighFreqFactor",
    "rope_high_freq_factor",
    "ropeOriginalContextLength",
    "rope_original_context_length",
    "slidingWindow",
    "sliding_window",
    "attentionSlidingWindow",
    "attention_sliding_window",
    "useSlidingWindow",
    "use_sliding_window",
    "useRopeLayers",
    "use_rope_layers",
    "noRopeLayers",
    "no_rope_layers",
    "noRopeLayerInterval",
    "no_rope_layer_interval",
    "qkProjectionNorm",
    "qk_projection_norm",
    "qk_norm",
    "embeddingRole",
    "embedding_role",
    "lmHeadRole",
    "lm_head_role",
    "lmHeadBiasRole",
    "lm_head_bias_role",
    "normRole",
    "norm_role",
    "qNormRole",
    "kNormRole",
  ]) {
    if (options[key] !== undefined && options[key] !== null && sharedBlockOptions[key] === undefined) {
      sharedBlockOptions[key] = options[key];
    }
  }
  return source.map((spec) => {
    if (spec instanceof WasmWebGpuLlamaBlockProjectionExecutor) {
      return { executor: spec, owned: false };
    }
    const blockOptions = {
      ...sharedBlockOptions,
      ...(typeof spec === "object" && spec !== null ? spec : {}),
    };
    if (Number.isSafeInteger(spec)) blockOptions.layer = spec;
    return {
      executor: new WasmWebGpuLlamaBlockProjectionExecutor(blockOptions),
      owned: true,
    };
  });
}

function createLlamaActivationWorkspace(session, hiddenSize, index) {
  const elementCount = session.contextLength * hiddenSize;
  if (!Number.isSafeInteger(elementCount) || elementCount <= 0) {
    throw new Error(`invalid LLaMA block pipeline activation workspace element count: ${elementCount}`);
  }
  const byteLength = elementCount * Float32Array.BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(byteLength)) {
    throw new Error("LLaMA block pipeline activation workspace byteLength is too large");
  }
  const resource = session.device.createBuffer({
    label: `zgml.wasm.webgpu.llama.block-pipeline.activation.${index}`,
    byteLength,
    usage: webgpuRequiredUsageFlags("readwrite"),
  });
  return {
    descriptor: storageResourceDescriptor(
      session.device,
      resource,
      { access: "readwrite", byteLength },
      `LLaMA block pipeline activation workspace ${index}`,
      byteLength,
    ),
    resource,
  };
}

function llamaPipelineStageSession(baseSession, activationInput, activationOutput) {
  return {
    ...baseSession,
    bindings: Object.freeze({
      ...baseSession.bindings,
      activation: activationOutput?.descriptor ?? null,
      activationInput: activationInput?.descriptor ?? null,
    }),
  };
}

function createLlamaPipelineExecutionScratch() {
  return {
    descPtr: 0,
    outputLen: 0,
    outputPolicy: 0,
    outputPtr: 0,
    tokens: null,
    tokensLen: 0,
    tokensPtr: 0,
  };
}

function createLlamaPipelineStepParamsScratch() {
  return {
    contextLength: 0,
    endPosition: 0,
    kind: "llama-token-window",
    logitsLength: 0,
    outputKind: "none",
    outputPolicy: 0,
    requestedOutputLength: 0,
    startPosition: 0,
    tokenCount: 0,
  };
}

function createLlamaPipelineContextScratch() {
  return {
    bindings: null,
    device: null,
    execution: createLlamaPipelineExecutionScratch(),
    modelBindingKind: null,
    modelHandle: 0,
    modelManifest: null,
    modelResources: null,
    modelTable: null,
    resultScratch: null,
    runtime: null,
    session: null,
    stepParams: createLlamaPipelineStepParamsScratch(),
    table: null,
  };
}

function writeLlamaPipelineStageStepParams(target, stepParams, outputPolicy) {
  target.contextLength = stepParams.contextLength;
  target.endPosition = stepParams.endPosition;
  target.kind = stepParams.kind ?? "llama-token-window";
  target.startPosition = stepParams.startPosition;
  target.tokenCount = stepParams.tokenCount;
  target.outputPolicy = outputPolicy;
  if (outputPolicy === stepParams.outputPolicy) {
    target.logitsLength = stepParams.logitsLength;
    target.outputKind = stepParams.outputKind;
    target.requestedOutputLength = stepParams.requestedOutputLength;
  } else {
    target.logitsLength = 0;
    target.outputKind = "none";
    target.requestedOutputLength = 0;
  }
  return target;
}

function writeLlamaPipelineStageExecution(target, execution, outputPolicy) {
  target.descPtr = execution.descPtr ?? 0;
  target.outputPolicy = outputPolicy;
  target.tokens = execution.tokens;
  target.tokensLen = execution.tokensLen;
  target.tokensPtr = execution.tokensPtr ?? 0;
  if (outputPolicy === execution.outputPolicy) {
    target.outputLen = execution.outputLen;
    target.outputPtr = execution.outputPtr;
  } else {
    target.outputLen = 0;
    target.outputPtr = 0;
  }
  return target;
}

function writeLlamaPipelineStageContext(target, context, stageSession, outputPolicy, resultScratch = null) {
  target.bindings = stageSession.bindings;
  target.device = stageSession.device;
  target.execution = writeLlamaPipelineStageExecution(target.execution, context.execution, outputPolicy);
  target.modelBindingKind = stageSession.modelBindingKind;
  target.modelHandle = stageSession.modelHandle;
  target.modelManifest = stageSession.modelManifest;
  target.modelResources = stageSession.modelResources;
  target.modelTable = stageSession.modelTable;
  target.resultScratch = resultScratch;
  target.runtime = context.runtime;
  target.session = stageSession;
  target.stepParams = writeLlamaPipelineStageStepParams(target.stepParams, context.stepParams, outputPolicy);
  target.table = stageSession.table;
  return target;
}

function writeLlamaPipelineFinalLogitsContext(target, context, finalLogitsSession, resultScratch = null) {
  target.bindings = finalLogitsSession.bindings;
  target.device = finalLogitsSession.device;
  target.execution = context.execution;
  target.modelBindingKind = finalLogitsSession.modelBindingKind;
  target.modelHandle = finalLogitsSession.modelHandle;
  target.modelManifest = finalLogitsSession.modelManifest;
  target.modelResources = finalLogitsSession.modelResources;
  target.modelTable = finalLogitsSession.modelTable;
  target.resultScratch = resultScratch;
  target.runtime = context.runtime;
  target.session = finalLogitsSession;
  target.stepParams = context.stepParams;
  target.table = finalLogitsSession.table;
  return target;
}

export class WasmWebGpuLlamaBlockPipelineExecutor {
  constructor(options = {}) {
    this.blocks = normalizeLlamaBlockPipelineSpecs(options);
    const finalBlockOptions = this.blocks.at(-1).executor.resourceOptions();
    this.finalLogitsExecutor = new WasmWebGpuLlamaActivationLogitsExecutor(finalBlockOptions);
    this.record = typeof options.record === "function" ? options.record : null;
    this.splitTerminalLogits = options.splitTerminalLogits ?? true;
    this.splitTerminalLogitsMinHiddenSize = options.splitTerminalLogitsMinHiddenSize ?? 64;
    if (typeof this.splitTerminalLogits !== "boolean") {
      throw new Error("LLaMA block pipeline splitTerminalLogits must be a boolean");
    }
    if (
      !Number.isSafeInteger(this.splitTerminalLogitsMinHiddenSize) ||
      this.splitTerminalLogitsMinHiddenSize <= 0
    ) {
      throw new Error(`invalid LLaMA block pipeline splitTerminalLogitsMinHiddenSize: ${this.splitTerminalLogitsMinHiddenSize}`);
    }
    this.preparedSessions = new WeakMap();
    this.ownedResources = [];
    this.stageContextScratch = createLlamaPipelineContextScratch();
    this.stageResultScratch = {};
    this.finalLogitsContextScratch = createLlamaPipelineContextScratch();
    this.finalLogitsResultScratch = {};
  }

  requiredGpuStorageBufferCount() {
    return 6;
  }

  prepareSession(session) {
    return prepareLlamaProjectionSession(this.preparedSessions, session, () => {
      const projections = this.blocks.map((entry) => llamaBlockProjectionResources(session, entry.executor.resourceOptions()));
      const hiddenSize = projections[0].hiddenSize;
      const ffnSize = projections[0].ffnSize;
      for (const projection of projections) {
        if (projection.hiddenSize !== hiddenSize) {
          throw new Error(`LLaMA block pipeline hidden size mismatch: ${projection.hiddenSize} != ${hiddenSize}`);
        }
        if (projection.ffnSize !== ffnSize) {
          throw new Error(`LLaMA block pipeline FFN intermediate size mismatch: ${projection.ffnSize} != ${ffnSize}`);
        }
      }
      const workspaceCount = Math.min(2, Math.max(0, this.blocks.length - 1));
      const workspaces = Array.from({ length: workspaceCount }, (_unused, index) => {
        const workspace = createLlamaActivationWorkspace(session, hiddenSize, index);
        registerLlamaSessionOwnedResource(session, workspace.resource, this);
        return workspace;
      });
      const splitTerminalLogits = this.splitTerminalLogits && hiddenSize >= this.splitTerminalLogitsMinHiddenSize;
      const terminalLogitsOutput = splitTerminalLogits
        ? (
            session.bindings.activation === null
              ? createLlamaActivationWorkspace(session, hiddenSize, "terminal-logits")
              : { descriptor: session.bindings.activation, resource: null }
          )
        : null;
      if (terminalLogitsOutput?.resource) {
        registerLlamaSessionOwnedResource(session, terminalLogitsOutput.resource, this);
      }
      const finalLogitsSession = terminalLogitsOutput === null
        ? null
        : llamaPipelineStageSession(session, terminalLogitsOutput, null);
      if (finalLogitsSession !== null) this.finalLogitsExecutor.prepareSession(finalLogitsSession);
      const lastStage = this.blocks.length - 1;
      const stages = this.blocks.map((entry, index) => {
        const input = index === 0
          ? (session.bindings.activationInput === null ? null : { descriptor: session.bindings.activationInput })
          : workspaces[(index - 1) % workspaces.length];
        const output = index === lastStage
          ? (session.bindings.activation === null ? null : { descriptor: session.bindings.activation })
          : workspaces[index % workspaces.length];
        const splitOutput = index === lastStage ? terminalLogitsOutput : null;
        const stageSession = llamaPipelineStageSession(session, input, output);
        entry.executor.prepareSession(stageSession);
        const splitSession = splitOutput === null ? null : llamaPipelineStageSession(session, input, splitOutput);
        if (splitSession !== null) entry.executor.prepareSession(splitSession);
        return Object.freeze({
          executor: entry.executor,
          index,
          inputRole: input === null ? null : (index === 0 ? "llama.activation.input" : `llama.activation.workspace.${(index - 1) % workspaces.length}`),
          layer: entry.executor.layer,
          outputRole: output === null ? null : (index === lastStage ? "llama.activation" : `llama.activation.workspace.${index % workspaces.length}`),
          session: stageSession,
          splitOutputRole: splitOutput === null ? null : "llama.activation.terminal-logits",
          splitSession,
        });
      });
      return Object.freeze({
        finalLogitsExecutor: this.finalLogitsExecutor,
        finalLogitsSession,
        ffnSize,
        hiddenSize,
        splitTerminalLogits,
        stages: Object.freeze(stages),
        terminalLogitsWorkspace: terminalLogitsOutput !== null && terminalLogitsOutput.resource !== null,
        workspaceCount,
      });
    });
  }

  executeTokensSync(context) {
    const pipeline = this.prepareSession(context.session);
    let backendDispatchCount = 0;
    let commandCount = 0;
    let fallbackOpCount = 0;
    let genericDispatchCount = 0;
    let outputLength = 0;
    let scalarDispatchCount = 0;
    let status = context.runtime.status.ok;
    let windowDispatchCount = 0;
    const useSplitTerminalLogits = context.execution.outputPolicy === 1 && pipeline.splitTerminalLogits;
    for (const stage of pipeline.stages) {
      const isLast = stage.index === pipeline.stages.length - 1;
      const stageSession = isLast && useSplitTerminalLogits ? stage.splitSession : stage.session;
      const stageOutputPolicy = isLast && useSplitTerminalLogits
        ? 0
        : (isLast ? context.execution.outputPolicy : 0);
      const stageContext = writeLlamaPipelineStageContext(
        this.stageContextScratch,
        context,
        stageSession,
        stageOutputPolicy,
        this.stageResultScratch,
      );
      const result = stage.executor.executeTokensSync(stageContext);
      backendDispatchCount += result.backendDispatchCount;
      commandCount += result.commandCount;
      fallbackOpCount += result.fallbackOpCount;
      genericDispatchCount += result.genericDispatchCount ?? 0;
      scalarDispatchCount += result.scalarDispatchCount ?? 0;
      windowDispatchCount += result.windowDispatchCount ?? 0;
      if (isLast) outputLength = result.outputLength;
      if (result.status !== context.runtime.status.ok) {
        status = result.status;
        outputLength = 0;
        break;
      }
    }
    let finalLogitsCommandCount = 0;
    if (status === context.runtime.status.ok && useSplitTerminalLogits) {
      const finalContext = writeLlamaPipelineFinalLogitsContext(
        this.finalLogitsContextScratch,
        context,
        pipeline.finalLogitsSession,
        this.finalLogitsResultScratch,
      );
      const result = pipeline.finalLogitsExecutor.executeTokensSync(finalContext);
      backendDispatchCount += result.backendDispatchCount;
      commandCount += result.commandCount;
      finalLogitsCommandCount = result.commandCount;
      fallbackOpCount += result.fallbackOpCount;
      outputLength = result.outputLength;
      if (result.status !== context.runtime.status.ok) {
        status = result.status;
        outputLength = 0;
      }
    }
    const deviceEvidence = llamaExecutorDeviceEvidence(context.session, 6);
    this.record?.({
      backendDispatchCount,
      commandCount,
      ...deviceEvidence,
      fallbackOpCount,
      finalLogitsCommandCount,
      ffnSize: pipeline.ffnSize,
      genericDispatchCount,
      hiddenSize: pipeline.hiddenSize,
      layers: pipeline.stages.map((stage) => stage.layer),
      outputLength,
      outputPolicy: context.execution.outputPolicy,
      scalarDispatchCount,
      splitTerminalLogits: useSplitTerminalLogits,
      stageCount: pipeline.stages.length,
      status,
      terminalLogitsWorkspace: pipeline.terminalLogitsWorkspace,
      tokenCount: context.stepParams.tokenCount,
      windowDispatchCount,
      workspaceCount: pipeline.workspaceCount,
    }, context);
    const result = writeLlamaTokenExecutorResult(
      context.resultScratch ?? {},
      context,
      status,
      outputLength,
      backendDispatchCount,
      fallbackOpCount,
      commandCount,
    );
    result.genericDispatchCount = genericDispatchCount;
    result.scalarDispatchCount = scalarDispatchCount;
    result.usesGpuStorageBuffers = deviceEvidence.usesGpuStorageBuffers;
    result.windowDispatchCount = windowDispatchCount;
    return result;
  }

  destroy() {
    for (const entry of this.blocks) {
      if (entry.owned) entry.executor.destroy?.();
    }
    this.finalLogitsExecutor.destroy();
    for (const resource of this.ownedResources) {
      destroyTrackedWasmWebGpuResource(resource);
    }
    this.ownedResources = [];
    this.preparedSessions = new WeakMap();
  }
}

function createLlamaResourceProfile() {
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
    executorCallCount: 0,
    executorOkCount: 0,
    executorFailureCount: 0,
    executorBackendOpCount: 0,
    executorBackendDispatchCount: 0,
    executorFallbackOpCount: 0,
    executorCommandCount: 0,
    executorCommandOpCount: 0,
    executorGenericDispatchCount: 0,
    executorScalarDispatchCount: 0,
    executorWindowDispatchCount: 0,
    selectionCallCount: 0,
    selectionBackendDispatchCount: 0,
    selectionFallbackOpCount: 0,
    selectionResultReadCount: 0,
    selectionSyncCount: 0,
    outputReadCount: 0,
    syncCount: 0,
    lastStatus: null,
    lastOutputPolicy: null,
    lastOutputLength: 0,
    lastResultOutputLength: 0,
  };
}

function llamaTokenWindowLengthOption(options = {}) {
  return options.tokensLen ??
    options.tokenLength ??
    options.activeTokenLength ??
    options.activeTokenCount;
}

function normalizeLlamaTokenWindowLength(value, containerLength, label) {
  const length = value === undefined || value === null ? containerLength : value;
  if (!Number.isSafeInteger(length) || length <= 0) {
    throw new Error(`${label} active length must be a positive integer`);
  }
  if (!Number.isSafeInteger(containerLength) || length > containerLength) {
    throw new Error(`${label} active length ${length} exceeds token container length ${containerLength}`);
  }
  return length;
}

function validateLlamaTokenValue(token, index, label) {
  if (!Number.isSafeInteger(token) || token < 0 || token > 0xffffffff) {
    throw new Error(`${label}[${index}] is not a u32 token: ${token}`);
  }
  return token;
}

function normalizeLlamaTokenWindow(tokens, label = "LLaMA tokens", options = {}) {
  const borrow = options.borrow === true &&
    (Array.isArray(tokens) || ArrayBuffer.isView(tokens)) &&
    Number.isSafeInteger(tokens.length);
  const source = Number.isInteger(tokens) ? [tokens] : (borrow ? tokens : Array.from(tokens ?? []));
  const tokensLen = normalizeLlamaTokenWindowLength(
    llamaTokenWindowLengthOption(options),
    source.length,
    label,
  );
  if (borrow) {
    for (let index = 0; index < tokensLen; index += 1) {
      validateLlamaTokenValue(source[index], index, label);
    }
    return { tokens: source, tokensLen };
  }
  const owned = new Array(tokensLen);
  for (let index = 0; index < tokensLen; index += 1) {
    owned[index] = validateLlamaTokenValue(source[index], index, label);
  }
  return { tokens: owned, tokensLen };
}

function normalizeLlamaTokenList(tokens, label = "LLaMA tokens", options = {}) {
  return normalizeLlamaTokenWindow(tokens, label, options).tokens;
}

function llamaPromptTokenWindowLengthOption(options = {}) {
  return options.promptTokensLen ??
    options.promptTokenLength ??
    options.promptActiveTokenLength ??
    options.promptActiveTokenCount ??
    llamaTokenWindowLengthOption(options);
}

function normalizeLlamaPromptTokenWindow(promptTokens, options = {}) {
  return normalizeLlamaTokenWindow(promptTokens, "LLaMA prompt tokens", {
    borrow: true,
    tokensLen: llamaPromptTokenWindowLengthOption(options),
  });
}

function normalizeLlamaGenerationCount(value, label = "LLaMA generated token count") {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return value;
}

function normalizeLlamaOutputTokenBuffer(outputTokens, label = "LLaMA output token buffer") {
  if (
    outputTokens === undefined ||
    outputTokens === null ||
    (!Array.isArray(outputTokens) && !ArrayBuffer.isView(outputTokens))
  ) {
    throw new Error(`${label} must be an Array or typed array`);
  }
  if (!Number.isSafeInteger(outputTokens.length) || outputTokens.length <= 0) {
    throw new Error(`${label} must contain at least one slot`);
  }
  return outputTokens;
}

function llamaGenerationWindowTokenCount(promptTokenCount, outputTokenCount) {
  return promptTokenCount + Math.max(0, outputTokenCount - 1);
}

function llamaGenerationWindowFits(session, promptTokenCount, outputTokenCount) {
  if (promptTokenCount > session.maxTokenWindow) return false;
  const tokenCount = llamaGenerationWindowTokenCount(promptTokenCount, outputTokenCount);
  return session.position + tokenCount <= session.contextLength;
}

function llamaGenerationShapeMismatchResult(session, runtime) {
  return {
    lastLogit: 0,
    lastToken: 0,
    position: session.position,
    status: runtime.status.shapeMismatch,
    tokensGenerated: 0,
  };
}

function llamaGenerationUnsupportedResult(session, runtime) {
  return {
    lastLogit: 0,
    lastToken: 0,
    position: session.position,
    status: runtime.status.unsupported,
    tokensGenerated: 0,
  };
}

const llamaTopKMax = 256;

function normalizeLlamaSampleTopK(value, label = "LLaMA sample topK") {
  const topK = value === undefined || value === null ? 40 : value;
  if (!Number.isSafeInteger(topK) || topK <= 0 || topK > llamaTopKMax) {
    throw new Error(`${label} must be an integer in [1, ${llamaTopKMax}]`);
  }
  return topK;
}

function normalizeLlamaSampleSeed(value, label = "LLaMA sample seed") {
  const seed = value === undefined || value === null ? 0 : value;
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error(`${label} must be a u32 integer`);
  }
  return seed >>> 0;
}

function normalizeLlamaSampleTemperature(value, label = "LLaMA sample temperature") {
  const temperature = value === undefined || value === null ? 1.0 : Number(value);
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new Error(`${label} must be finite and positive`);
  }
  return temperature;
}

function normalizeLlamaSampleOptions(options = {}) {
  return {
    seed: normalizeLlamaSampleSeed(options.seed),
    temperature: normalizeLlamaSampleTemperature(options.temperature),
    topK: normalizeLlamaSampleTopK(options.topK ?? options.top_k),
  };
}

function llamaRandomUnit(seed) {
  let x = (seed === 0 ? 0x9e3779b9 : seed) >>> 0;
  x = (x ^ ((x << 13) >>> 0)) >>> 0;
  x = (x ^ (x >>> 17)) >>> 0;
  x = (x ^ ((x << 5) >>> 0)) >>> 0;
  return x / 4294967296.0;
}

function writeLlamaArgmaxSelection(target, logits, label = "LLaMA logits") {
  if (logits === undefined || logits === null || !Number.isSafeInteger(logits.length) || logits.length <= 0) {
    throw new Error(`${label} must contain at least one value`);
  }
  let token = 0;
  let logit = Number(logits[0]);
  for (let i = 1; i < logits.length; i += 1) {
    const value = Number(logits[i]);
    if (value > logit) {
      token = i;
      logit = value;
    }
  }
  target.logit = logit;
  target.token = token;
  return target;
}

function selectLlamaArgmax(logits, label = "LLaMA logits") {
  const selected = writeLlamaArgmaxSelection({}, logits, label);
  return Object.freeze({ token: selected.token, logit: selected.logit });
}

class WasmWebGpuLlamaDeviceArgmaxSelector {
  constructor() {
    this.pipeline = null;
    this.pipelineDevice = null;
	    this.paramsBuffer = null;
	    this.resultResource = null;
	    this.resultDescriptor = null;
	    this.resultDevice = null;
	    this.paramsBytes = new ArrayBuffer(16);
    this.paramsView = new DataView(this.paramsBytes);
    this.resultBytes = new Uint8Array(8);
    this.resultView = new DataView(this.resultBytes.buffer);
    this.gpuBindGroups = new Map();
    this.bindGroupDeps = [null, null, null];
  }

  ensureGpuPipeline(device) {
    if (!(device instanceof WasmWebGpuDevice) || !device.canCreateGpuBuffers()) {
      throw new Error("LLaMA device argmax requires a browser WebGPU device");
    }
    const gpuDevice = device.device;
    if (this.pipeline && this.pipelineDevice === gpuDevice) return this.pipeline;
    this.destroy();
    const shader = gpuDevice.createShaderModule({
      label: "zgml.wasm.webgpu.llama-device-argmax",
      code: `
struct Params {
  vocab: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> best_values: array<f32, 256>;
var<workgroup> best_tokens: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
  let lane = local_id.x;
  var best_token = 0u;
  var best_value = -3.4028234663852886e38;
  var i = lane;
  while (i < params.vocab) {
    let value = logits[i];
    if (value > best_value || (value == best_value && i < best_token)) {
      best_value = value;
      best_token = i;
    }
    i = i + 256u;
  }
  best_values[lane] = best_value;
  best_tokens[lane] = best_token;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (lane < stride) {
      let other_value = best_values[lane + stride];
      let other_token = best_tokens[lane + stride];
      let self_value = best_values[lane];
      let self_token = best_tokens[lane];
      if (other_value > self_value || (other_value == self_value && other_token < self_token)) {
        best_values[lane] = other_value;
        best_tokens[lane] = other_token;
      }
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride = stride / 2u;
  }
  if (lane == 0u) {
    result[0] = best_tokens[0];
    result[1] = bitcast<u32>(best_values[0]);
  }
}
`,
    });
    this.pipeline = gpuDevice.createComputePipeline({
      label: "zgml.wasm.webgpu.llama-device-argmax",
      layout: "auto",
      compute: {
        module: shader,
        entryPoint: "main",
      },
    });
    const usage = gpuBufferUsageFlags();
    this.paramsBuffer = gpuDevice.createBuffer({
      label: "zgml.wasm.webgpu.llama-device-argmax.params",
      size: 16,
      usage: usage.uniform | usage.copyDst,
    });
	    this.resultResource = device.createBuffer({
	      label: "zgml.wasm.webgpu.llama-device-argmax.result",
	      byteLength: 8,
	      usage: webgpuRequiredUsageFlags("readwrite"),
	    });
	    this.resultDevice = device;
	    this.resultDescriptor = storageResourceDescriptor(
      device,
      this.resultResource,
      { access: "readwrite", byteLength: 8 },
      "LLaMA device argmax result",
      8,
    );
    this.pipelineDevice = gpuDevice;
    return this.pipeline;
  }

  async selectInto(target, session) {
    const device = session.device;
    const gpuDevice = device.device;
    const pipeline = this.ensureGpuPipeline(device);
    this.paramsView.setUint32(0, session.vocabSize, true);
    gpuDevice.queue.writeBuffer(this.paramsBuffer, 0, this.paramsBytes);
    const bindGroupDeps = this.bindGroupDeps;
    bindGroupDeps[0] = this.paramsBuffer;
    bindGroupDeps[1] = this.resultDescriptor;
    bindGroupDeps[2] = session.bindings.output;
    const bindGroup = projectionGpuBindGroup(
      this,
      "argmax",
      device,
      pipeline,
      bindGroupDeps,
      (targetDevice) => targetDevice.createBindGroup({
        label: "zgml.wasm.webgpu.llama-device-argmax",
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: storageBufferBinding(
              targetDevice,
              session.bindings.output,
              "LLaMA device argmax logits",
              session.vocabSize * Float32Array.BYTES_PER_ELEMENT,
            ),
          },
          {
            binding: 1,
            resource: storageBufferBinding(
              targetDevice,
              this.resultDescriptor,
              "LLaMA device argmax result",
              8,
            ),
          },
          { binding: 2, resource: { buffer: this.paramsBuffer } },
        ],
      }),
    );
    const encoder = gpuDevice.createCommandEncoder({ label: "zgml.wasm.webgpu.llama-device-argmax" });
    const pass = encoder.beginComputePass({ label: "zgml.wasm.webgpu.llama-device-argmax" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    gpuDevice.queue.submit([encoder.finish()]);
    await device.readBytesInto(this.resultDescriptor, this.resultBytes, 8);
    const token = this.resultView.getUint32(0, true);
    const logit = this.resultView.getFloat32(4, true);
    if (token >= session.vocabSize || !Number.isFinite(logit)) {
      throw new Error("LLaMA device argmax returned invalid candidate");
    }
    target.backendDispatchCount = 1;
    target.logit = logit;
    target.resultReadCount = 1;
    target.syncCount = 1;
    target.token = token;
    return target;
  }

  async select(session) {
    const selected = await this.selectInto({}, session);
    return Object.freeze({
      backendDispatchCount: selected.backendDispatchCount,
      logit: selected.logit,
      resultReadCount: selected.resultReadCount,
      syncCount: selected.syncCount,
      token: selected.token,
    });
  }

  destroy() {
	    this.gpuBindGroups = new Map();
	    if (this.paramsBuffer && typeof this.paramsBuffer.destroy === "function") this.paramsBuffer.destroy();
	    destroyWasmWebGpuDeviceResource(this.resultDevice, this.resultResource);
	    this.paramsBuffer = null;
	    this.resultDevice = null;
	    this.resultDescriptor = null;
	    this.resultResource = null;
    this.pipeline = null;
    this.pipelineDevice = null;
    this.bindGroupDeps.fill(null);
  }
}

class WasmWebGpuLlamaDeviceTopKSelector {
  constructor() {
    this.pipeline = null;
    this.pipelineDevice = null;
    this.paramsBuffer = null;
    this.paramsBytes = new ArrayBuffer(16);
    this.paramsView = new DataView(this.paramsBytes);
    this.resultBytes = null;
    this.resultByteCapacity = 0;
    this.resultView = null;
	    this.resultCapacity = 0;
	    this.resultDescriptor = null;
	    this.resultResource = null;
	    this.resultDevice = null;
	    this.gpuBindGroups = new Map();
    this.bindGroupDeps = [null, null, null];
  }

  ensureGpuPipeline(device) {
    if (!(device instanceof WasmWebGpuDevice) || !device.canCreateGpuBuffers()) {
      throw new Error("LLaMA device top-k sampling requires a browser WebGPU device");
    }
    const gpuDevice = device.device;
    if (this.pipeline && this.pipelineDevice === gpuDevice) return this.pipeline;
    this.destroy();
    const shader = gpuDevice.createShaderModule({
      label: "zgml.wasm.webgpu.llama-device-topk",
      code: `
struct Params {
  vocab: u32,
  top_k: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> best_values: array<f32, 256>;
var<workgroup> best_tokens: array<u32, 256>;
var<workgroup> selected_tokens: array<u32, 256>;

fn already_selected(token: u32, count: u32) -> bool {
  var i = 0u;
  loop {
    if (i >= count) {
      break;
    }
    if (selected_tokens[i] == token) {
      return true;
    }
    i = i + 1u;
  }
  return false;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
  let lane = local_id.x;
  var slot = 0u;
  loop {
    if (slot >= params.top_k) {
      break;
    }
    var best_token = 0xffffffffu;
    var best_value = -3.4028234663852886e38;
    var i = lane;
    while (i < params.vocab) {
      if (!already_selected(i, slot)) {
        let value = logits[i];
        if (value > best_value || (value == best_value && i < best_token)) {
          best_value = value;
          best_token = i;
        }
      }
      i = i + 256u;
    }
    best_values[lane] = best_value;
    best_tokens[lane] = best_token;
    workgroupBarrier();
    var stride = 128u;
    loop {
      if (lane < stride) {
        let other_value = best_values[lane + stride];
        let other_token = best_tokens[lane + stride];
        let self_value = best_values[lane];
        let self_token = best_tokens[lane];
        if (
          other_token != 0xffffffffu &&
          (
            self_token == 0xffffffffu ||
            other_value > self_value ||
            (other_value == self_value && other_token < self_token)
          )
        ) {
          best_values[lane] = other_value;
          best_tokens[lane] = other_token;
        }
      }
      workgroupBarrier();
      if (stride == 1u) {
        break;
      }
      stride = stride / 2u;
    }
    if (lane == 0u) {
      selected_tokens[slot] = best_tokens[0];
      let out = slot * 2u;
      result[out] = best_tokens[0];
      result[out + 1u] = bitcast<u32>(best_values[0]);
    }
    workgroupBarrier();
    slot = slot + 1u;
  }
}
`,
    });
    this.pipeline = gpuDevice.createComputePipeline({
      label: "zgml.wasm.webgpu.llama-device-topk",
      layout: "auto",
      compute: {
        module: shader,
        entryPoint: "main",
      },
    });
    const usage = gpuBufferUsageFlags();
    this.paramsBuffer = gpuDevice.createBuffer({
      label: "zgml.wasm.webgpu.llama-device-topk.params",
      size: 16,
      usage: usage.uniform | usage.copyDst,
    });
    this.pipelineDevice = gpuDevice;
    return this.pipeline;
  }

	  ensureResultResource(device, topK) {
	    if (this.resultDescriptor !== null && this.resultCapacity >= topK) return this.resultDescriptor;
	    destroyWasmWebGpuDeviceResource(this.resultDevice, this.resultResource);
	    this.gpuBindGroups = new Map();
	    const byteLength = topK * 2 * Uint32Array.BYTES_PER_ELEMENT;
	    this.resultResource = device.createBuffer({
	      label: "zgml.wasm.webgpu.llama-device-topk.result",
	      byteLength,
	      usage: webgpuRequiredUsageFlags("readwrite"),
	    });
	    this.resultDevice = device;
	    this.resultDescriptor = storageResourceDescriptor(
      device,
      this.resultResource,
      { access: "readwrite", byteLength },
      "LLaMA device top-k result",
      byteLength,
    );
    this.resultCapacity = topK;
    return this.resultDescriptor;
  }

  ensureResultBytes(byteLength) {
    if (this.resultBytes instanceof Uint8Array && this.resultByteCapacity >= byteLength) return this.resultBytes;
    this.resultBytes = new Uint8Array(byteLength);
    this.resultByteCapacity = byteLength;
    this.resultView = new DataView(this.resultBytes.buffer);
    return this.resultBytes;
  }

  async selectInto(target, session, sampleOptions) {
    const device = session.device;
    const gpuDevice = device.device;
    const effectiveK = Math.min(sampleOptions.topK, session.vocabSize);
    if (!Number.isSafeInteger(effectiveK) || effectiveK <= 0 || effectiveK > llamaTopKMax) {
      throw new Error(`invalid LLaMA device top-k count: ${effectiveK}`);
    }
    const pipeline = this.ensureGpuPipeline(device);
    const resultDescriptor = this.ensureResultResource(device, effectiveK);
    this.paramsView.setUint32(0, session.vocabSize, true);
    this.paramsView.setUint32(4, effectiveK, true);
    gpuDevice.queue.writeBuffer(this.paramsBuffer, 0, this.paramsBytes);
    const bindGroupDeps = this.bindGroupDeps;
    bindGroupDeps[0] = this.paramsBuffer;
    bindGroupDeps[1] = resultDescriptor;
    bindGroupDeps[2] = session.bindings.output;
    const bindGroup = projectionGpuBindGroup(
      this,
      "topk",
      device,
      pipeline,
      bindGroupDeps,
      (targetDevice) => targetDevice.createBindGroup({
        label: "zgml.wasm.webgpu.llama-device-topk",
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: storageBufferBinding(
              targetDevice,
              session.bindings.output,
              "LLaMA device top-k logits",
              session.vocabSize * Float32Array.BYTES_PER_ELEMENT,
            ),
          },
          {
            binding: 1,
            resource: storageBufferBinding(
              targetDevice,
              resultDescriptor,
              "LLaMA device top-k result",
              effectiveK * 2 * Uint32Array.BYTES_PER_ELEMENT,
            ),
          },
          { binding: 2, resource: { buffer: this.paramsBuffer } },
        ],
      }),
    );
    const encoder = gpuDevice.createCommandEncoder({ label: "zgml.wasm.webgpu.llama-device-topk" });
    const pass = encoder.beginComputePass({ label: "zgml.wasm.webgpu.llama-device-topk" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    gpuDevice.queue.submit([encoder.finish()]);
    const byteLength = effectiveK * 2 * Uint32Array.BYTES_PER_ELEMENT;
    const bytes = this.ensureResultBytes(byteLength);
    await device.readBytesInto(resultDescriptor, bytes, byteLength);
    writeLlamaTopKSampleDataViewSelection(
      target,
      this.resultView,
      effectiveK,
      sampleOptions,
      session.vocabSize,
      "LLaMA device top-k candidates",
    );
    target.backendDispatchCount = 1;
    target.resultReadCount = 1;
    target.syncCount = 1;
    return target;
  }

  async select(session, sampleOptions) {
    const selected = await this.selectInto({}, session, sampleOptions);
    return Object.freeze({
      backendDispatchCount: selected.backendDispatchCount,
      logit: selected.logit,
      resultReadCount: selected.resultReadCount,
      syncCount: selected.syncCount,
      token: selected.token,
    });
  }

  destroy() {
	    this.gpuBindGroups = new Map();
	    if (this.paramsBuffer && typeof this.paramsBuffer.destroy === "function") this.paramsBuffer.destroy();
	    destroyWasmWebGpuDeviceResource(this.resultDevice, this.resultResource);
	    this.paramsBuffer = null;
	    this.pipeline = null;
	    this.pipelineDevice = null;
    this.resultCapacity = 0;
    this.resultByteCapacity = 0;
	    this.resultBytes = null;
	    this.resultDescriptor = null;
	    this.resultDevice = null;
	    this.resultResource = null;
    this.resultView = null;
    this.bindGroupDeps.fill(null);
  }
}

function insertLlamaSampleCandidate(candidateTokens, candidateLogits, candidateLimit, candidateCount, token, logit) {
  let i = candidateCount;
  while (i > 0 && logit > candidateLogits[i - 1]) {
    if (i < candidateLimit) {
      candidateTokens[i] = candidateTokens[i - 1];
      candidateLogits[i] = candidateLogits[i - 1];
    }
    i -= 1;
  }
  if (i < candidateLimit) {
    candidateTokens[i] = token;
    candidateLogits[i] = logit;
  }
  return candidateCount < candidateLimit ? candidateCount + 1 : candidateCount;
}

function writeLlamaTopKSampleCandidateSelection(target, candidateTokens, candidateLogits, candidateCount, sampleOptions, label = "LLaMA top-k candidates") {
  if (!Array.isArray(candidateTokens) || !Array.isArray(candidateLogits) || !Number.isSafeInteger(candidateCount) || candidateCount <= 0) {
    throw new Error(`${label} must contain at least one candidate`);
  }
  if (candidateCount === 1) {
    target.logit = candidateLogits[0];
    target.token = candidateTokens[0];
    return target;
  }
  const maxLogit = candidateLogits[0];
  const invTemperature = 1.0 / sampleOptions.temperature;
  let total = 0;
  for (let i = 0; i < candidateCount; i += 1) {
    total += Math.exp((candidateLogits[i] - maxLogit) * invTemperature);
  }
  let threshold = llamaRandomUnit(sampleOptions.seed) * total;
  let fallbackToken = candidateTokens[0];
  let fallbackLogit = candidateLogits[0];
  for (let i = 0; i < candidateCount; i += 1) {
    const token = candidateTokens[i];
    const logit = candidateLogits[i];
    fallbackToken = token;
    fallbackLogit = logit;
    const weight = Math.exp((logit - maxLogit) * invTemperature);
    if (threshold < weight) {
      target.logit = logit;
      target.token = token;
      return target;
    }
    threshold -= weight;
  }
  target.logit = fallbackLogit;
  target.token = fallbackToken;
  return target;
}

function selectLlamaTopKSampleCandidates(candidateTokens, candidateLogits, candidateCount, sampleOptions, label = "LLaMA top-k candidates") {
  const selected = writeLlamaTopKSampleCandidateSelection({}, candidateTokens, candidateLogits, candidateCount, sampleOptions, label);
  return Object.freeze({ token: selected.token, logit: selected.logit });
}

function writeLlamaTopKSampleDataViewSelection(target, view, count, sampleOptions, vocabSize, label = "LLaMA top-k candidates") {
  if (!(view instanceof DataView) || !Number.isSafeInteger(count) || count <= 0 || view.byteLength < count * 8) {
    throw new Error(`${label} must contain at least one candidate`);
  }
  const firstToken = view.getUint32(0, true);
  const firstLogit = view.getFloat32(4, true);
  if (firstToken >= vocabSize || !Number.isFinite(firstLogit)) {
    throw new Error(`${label} returned invalid candidate`);
  }
  if (count === 1) {
    target.logit = firstLogit;
    target.token = firstToken;
    return target;
  }
  const invTemperature = 1.0 / sampleOptions.temperature;
  let total = 0;
  for (let i = 0; i < count; i += 1) {
    const token = view.getUint32(i * 8, true);
    const logit = view.getFloat32(i * 8 + 4, true);
    if (token >= vocabSize || !Number.isFinite(logit)) {
      throw new Error(`${label} returned invalid candidate`);
    }
    total += Math.exp((logit - firstLogit) * invTemperature);
  }
  let threshold = llamaRandomUnit(sampleOptions.seed) * total;
  let fallbackToken = firstToken;
  let fallbackLogit = firstLogit;
  for (let i = 0; i < count; i += 1) {
    const token = view.getUint32(i * 8, true);
    const logit = view.getFloat32(i * 8 + 4, true);
    if (token >= vocabSize || !Number.isFinite(logit)) {
      throw new Error(`${label} returned invalid candidate`);
    }
    fallbackToken = token;
    fallbackLogit = logit;
    const weight = Math.exp((logit - firstLogit) * invTemperature);
    if (threshold < weight) {
      target.logit = logit;
      target.token = token;
      return target;
    }
    threshold -= weight;
  }
  target.logit = fallbackLogit;
  target.token = fallbackToken;
  return target;
}

function selectLlamaTopKSampleDataView(view, count, sampleOptions, vocabSize, label = "LLaMA top-k candidates") {
  const selected = writeLlamaTopKSampleDataViewSelection({}, view, count, sampleOptions, vocabSize, label);
  return Object.freeze({ logit: selected.logit, token: selected.token });
}

function writeLlamaTopKSampleSelection(target, logits, options = {}, label = "LLaMA logits", scratch = null, normalizedSampleOptions = false) {
  if (logits === undefined || logits === null || !Number.isSafeInteger(logits.length) || logits.length <= 0) {
    throw new Error(`${label} must contain at least one value`);
  }
  const sampleOptions = normalizedSampleOptions === true ? options : normalizeLlamaSampleOptions(options);
  const effectiveK = Math.min(sampleOptions.topK, logits.length);
  const candidateTokens = scratch !== null && Array.isArray(scratch.candidateTokens) && scratch.candidateTokens.length >= effectiveK
    ? scratch.candidateTokens
    : new Array(effectiveK);
  const candidateLogits = scratch !== null && Array.isArray(scratch.candidateLogits) && scratch.candidateLogits.length >= effectiveK
    ? scratch.candidateLogits
    : new Array(effectiveK);
  if (scratch !== null) {
    scratch.candidateTokens = candidateTokens;
    scratch.candidateLogits = candidateLogits;
  }
  let candidateCount = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const logit = Number(logits[i]);
    if (!Number.isFinite(logit)) {
      throw new Error(`${label}[${i}] is not finite`);
    }
    if (candidateCount < effectiveK) {
      candidateCount = insertLlamaSampleCandidate(candidateTokens, candidateLogits, effectiveK, candidateCount, i, logit);
    } else if (logit > candidateLogits[candidateCount - 1]) {
      insertLlamaSampleCandidate(candidateTokens, candidateLogits, effectiveK, candidateCount, i, logit);
    }
  }
  return writeLlamaTopKSampleCandidateSelection(target, candidateTokens, candidateLogits, candidateCount, sampleOptions);
}

function selectLlamaTopKSample(logits, options = {}, label = "LLaMA logits", scratch = null) {
  const selected = writeLlamaTopKSampleSelection({}, logits, options, label, scratch);
  return Object.freeze({ logit: selected.logit, token: selected.token });
}

function normalizeLlamaSessionOutputPolicy(value, label = "LLaMA output policy") {
  if (value === undefined || value === null || value === true) return 1;
  if (value === false) return 0;
  if (value === 0 || value === 1) return value;
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase().replaceAll("_", "-").replaceAll(" ", "-");
    if (normalized === "none" || normalized === "no-output" || normalized === "advance") return 0;
    if (normalized === "logits" || normalized === "bound-logits" || normalized === "output") return 1;
  }
  throw new Error(`unsupported ${label}: ${value}`);
}

function llamaExecutionFromTokenList(tokens, options = {}) {
  const outputPolicy = normalizeLlamaSessionOutputPolicy(
    options.outputPolicy ?? options.output ?? options.outputs ?? options.logits,
  );
  const tokenWindow = normalizeLlamaTokenWindow(tokens, "LLaMA tokens", options);
  return writeLlamaExecutionFromTokenList({}, tokenWindow.tokens, outputPolicy, tokenWindow.tokensLen);
}

function writeLlamaExecutionFromTokenList(target, tokens, outputPolicy, tokensLen = tokens.length) {
  target.descPtr = 0;
  target.outputLen = 0;
  target.outputPolicy = outputPolicy;
  target.outputPtr = 0;
  target.tokens = tokens;
  target.tokensLen = tokensLen;
  target.tokensPtr = 0;
  return target;
}

function resetLlamaSelectionResult(target, status) {
  target.outputLen = 0;
  return resetLlamaSelectionFields(target, status);
}

function resetLlamaSelectionFields(target, status) {
  target.backendDispatchCount = 0;
  target.logit = 0;
  target.logits = undefined;
  target.position = 0;
  target.resultReadCount = 0;
  target.status = status;
  target.syncCount = 0;
  target.token = 0;
  return target;
}

function writeLlamaExecutionOutcome(target, outputLen, status) {
  target.outputLen = outputLen;
  target.status = status;
  return target;
}

function writeNormalizedLlamaDeviceSelection(target, selected, session) {
  if (selected === undefined || selected === null) selected = target;
  if (typeof selected !== "object") throw new Error("LLaMA device selection result must be an object");
  const token = selected.token;
  const logit = selected.logit;
  const backendDispatchCount = selected.backendDispatchCount ?? 0;
  const resultReadCount = selected.resultReadCount ?? 0;
  const syncCount = selected.syncCount ?? 0;
  if (!Number.isSafeInteger(token) || token < 0 || token >= session.vocabSize) {
    throw new Error("LLaMA device selection returned an invalid token");
  }
  if (typeof logit !== "number" || !Number.isFinite(logit)) {
    throw new Error("LLaMA device selection returned an invalid logit");
  }
  if (!Number.isSafeInteger(backendDispatchCount) || backendDispatchCount < 0) {
    throw new Error("LLaMA device selection returned an invalid backend dispatch count");
  }
  if (!Number.isSafeInteger(resultReadCount) || resultReadCount < 0) {
    throw new Error("LLaMA device selection returned an invalid result read count");
  }
  if (!Number.isSafeInteger(syncCount) || syncCount < 0) {
    throw new Error("LLaMA device selection returned an invalid sync count");
  }
  if (backendDispatchCount === 0) {
    throw new Error("LLaMA device selection reported no backend dispatch");
  }
  if (backendDispatchCount > 0) {
    if (!(session.device instanceof WasmWebGpuDevice) || !session.device.canCreateGpuBuffers()) {
      throw new Error("LLaMA device selection reported GPU dispatch without a WebGPU device");
    }
    if (resultReadCount <= 0 || syncCount <= 0) {
      throw new Error("LLaMA device selection reported GPU dispatch without result readback");
    }
  }
  target.backendDispatchCount = backendDispatchCount;
  target.logit = logit;
  target.resultReadCount = resultReadCount;
  target.syncCount = syncCount;
  target.token = token;
  return target;
}

function canUseLlamaDeviceSelection(session, options = {}) {
  return (
    session.device &&
    typeof session.device.canCreateGpuBuffers === "function" &&
    session.device.canCreateGpuBuffers() &&
    options.gpu !== false &&
    options.device !== "cpu"
  );
}

function canUseLlamaRequiredGpuDeviceSelection(session, options = {}) {
  return (
    session.device instanceof WasmWebGpuDevice &&
    session.device.canCreateGpuBuffers() &&
    options.gpu !== false &&
    options.device !== "cpu"
  );
}

function shouldRejectRequiredGpuDeviceSelection(session, options = {}, logits = undefined) {
  return (
    session.requireGpuDispatch === true &&
    (
      (logits !== undefined && logits !== null) ||
      !canUseLlamaRequiredGpuDeviceSelection(session, options)
    )
  );
}

function shouldRejectRequiredGpuHostSelection(session) {
  return session.requireGpuDispatch === true;
}

function writeLlamaRequiredGpuSelectionUnsupported(target, session, runtime) {
  target.logit = 0;
  target.position = session.position;
  target.status = runtime.status.unsupported;
  target.token = 0;
  return target;
}

function defaultLlamaExecutorOutputLength(session, execution, status, runtime) {
  return status === runtime.status.ok && execution.outputPolicy === 1 ? session.vocabSize : 0;
}

function llamaExecutorDeviceEvidence(session, requiredStorageBufferCount) {
  const device = session?.device;
  const maxStorageBuffersPerShaderStage = device instanceof WasmWebGpuDevice
    ? device.maxStorageBuffersPerShaderStage()
    : 0;
  const canCreateGpuBuffers = device instanceof WasmWebGpuDevice && device.canCreateGpuBuffers();
  return {
    deviceMode: device instanceof WasmWebGpuDevice ? device.mode : "unknown",
    maxStorageBuffersPerShaderStage,
    requiredStorageBufferCount,
    usesGpuStorageBuffers:
      canCreateGpuBuffers &&
      Number.isSafeInteger(requiredStorageBufferCount) &&
      requiredStorageBufferCount <= maxStorageBuffersPerShaderStage,
  };
}

function writeLlamaTokenExecutorResult(
  target,
  context,
  status,
  outputLength,
  backendDispatchCount,
  fallbackOpCount,
  commandCount,
  commandOpCount = commandCount,
  backendOpCount = backendDispatchCount,
  genericDispatchCount = 0,
  scalarDispatchCount = 0,
  windowDispatchCount = 0,
) {
  target.backendDispatchCount = backendDispatchCount;
  target.backendOpCount = backendOpCount;
  target.commandCount = commandCount;
  target.commandOpCount = commandOpCount;
  target.fallbackOpCount = fallbackOpCount;
  target.genericDispatchCount = genericDispatchCount;
  target.modelHandle = context.session.modelHandle;
  target.modelManifestHash = context.session.modelManifest?.hash ?? undefined;
  const modelPacks = Array.isArray(context.session.modelPacks) ? context.session.modelPacks : [];
  const latestModelPack = modelPacks.at(-1);
  target.modelPackDescriptorHash = latestModelPack?.descriptorHash ?? undefined;
  target.modelPackLayoutHash = latestModelPack?.layoutHash ?? undefined;
  target.modelTableDescriptorHash = context.session.modelTable?.descriptorHash ?? undefined;
  target.modelTableHash = context.session.modelTable?.hash ?? undefined;
  target.outputLength = outputLength;
  target.scalarDispatchCount = scalarDispatchCount;
  target.status = status;
  target.tableDescriptorHash = context.session.table?.descriptorHash ?? 0n;
  target.tableHash = context.session.table?.hash ?? 0n;
  target.usesGpuStorageBuffers =
    backendDispatchCount > 0 &&
    fallbackOpCount === 0 &&
    context.session.device instanceof WasmWebGpuDevice &&
    context.session.device.canCreateGpuBuffers();
  target.windowDispatchCount = windowDispatchCount;
  return target;
}

function writeLlamaNormalizedExecutorResult(
  target,
  status,
  outputLength,
  backendOpCount = 0,
  backendDispatchCount = 0,
  fallbackOpCount = 0,
  commandCount = 0,
  commandOpCount = 0,
  genericDispatchCount = 0,
  scalarDispatchCount = 0,
  windowDispatchCount = 0,
  usesGpuStorageBuffers = false,
) {
  target.backendDispatchCount = backendDispatchCount;
  target.backendOpCount = backendOpCount;
  target.commandCount = commandCount;
  target.commandOpCount = commandOpCount;
  target.fallbackOpCount = fallbackOpCount;
  target.genericDispatchCount = genericDispatchCount;
  target.outputLength = outputLength;
  target.scalarDispatchCount = scalarDispatchCount;
  target.status = status;
  target.usesGpuStorageBuffers = usesGpuStorageBuffers === true;
  target.windowDispatchCount = windowDispatchCount;
  return target;
}

function normalizeLlamaExecutorCounter(result, name, defaultValue = 0) {
  const value = result[name] ?? defaultValue;
  if (!Number.isSafeInteger(value) || value < 0) return null;
  return value;
}

function validateLlamaExecutorCounterCoherence(status, runtime, counters) {
  if (status !== runtime.status.ok) return true;
  if (counters.commandCount !== counters.backendDispatchCount + counters.fallbackOpCount) return false;
  if (counters.backendOpCount < counters.backendDispatchCount) return false;
  if (counters.commandOpCount < counters.commandCount) return false;
  return true;
}

function validateLlamaExecutorWorkEvidence(status, runtime, session, counters) {
  if (status !== runtime.status.ok || !requiresLlamaExecutorTableEvidence(session)) return true;
  return counters.commandCount > 0;
}

function validateLlamaExecutorRequiredGpuDispatch(status, runtime, session, counters, usesGpuStorageBuffers) {
  if (status !== runtime.status.ok || session.requireGpuDispatch !== true) return true;
  return (
    counters.backendDispatchCount > 0 &&
    counters.fallbackOpCount === 0 &&
    usesGpuStorageBuffers === true
  );
}

function validateLlamaExecutorGpuStorageEvidence(status, runtime, session, counters, usesGpuStorageBuffers) {
  if (status !== runtime.status.ok) return true;
  if (counters.backendDispatchCount > 0 && usesGpuStorageBuffers !== true) return false;
  if (usesGpuStorageBuffers !== true) return true;
  if (counters.backendDispatchCount <= 0 || counters.fallbackOpCount !== 0) return false;
  const device = session.device;
  if (!(device instanceof WasmWebGpuDevice) || !device.canCreateGpuBuffers()) return false;
  const requiredStorageBufferCount = llamaExecutorRequiredGpuStorageBufferCount(session.tokenExecutor);
  if (requiredStorageBufferCount !== null && !device.canBindStorageBuffers(requiredStorageBufferCount)) return false;
  return true;
}

function validateLlamaExecutorFailureEvidence(status, runtime, outputLength, counters) {
  if (status === runtime.status.ok) return true;
  return (
    outputLength === 0 &&
    counters.backendDispatchCount === 0 &&
    counters.backendOpCount === 0 &&
    counters.fallbackOpCount === 0 &&
    counters.commandCount === 0 &&
    counters.commandOpCount === 0 &&
    counters.genericDispatchCount === 0 &&
    counters.scalarDispatchCount === 0 &&
    counters.windowDispatchCount === 0
  );
}

function validateLlamaExecutorOutputLength(status, runtime, session, execution, outputLength) {
  if (status !== runtime.status.ok) return true;
  if (execution.outputPolicy === 0) return outputLength === 0;
  if (execution.outputPolicy === 1) return outputLength === session.vocabSize;
  return false;
}

function normalizeLlamaExecutorHash(value) {
  if (value === undefined || value === null) return undefined;
  if (typeof value === "bigint" && value >= 0n && value <= u64Mask) return value;
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  return null;
}

function normalizeLlamaExecutorHandle(value) {
  if (value === undefined || value === null) return undefined;
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffffffff) return null;
  return value;
}

function normalizeLlamaExecutorStatusValue(value, runtime) {
  if (value === undefined || value === null) return runtime.status.ok;
  if (!Number.isSafeInteger(value)) return null;
  if (
    value === runtime.status.ok ||
    value === runtime.status.invalidArgument ||
    value === runtime.status.outOfMemory ||
    value === runtime.status.shapeMismatch ||
    value === runtime.status.compileFailed ||
    value === runtime.status.unsupported
  ) {
    return value;
  }
  return null;
}

function normalizeLlamaExecutorStatus(result, runtime) {
  return normalizeLlamaExecutorStatusValue(result.status, runtime);
}

function validateLlamaExecutorModelHandle(result, session) {
  const modelHandle = normalizeLlamaExecutorHandle(result.modelHandle ?? result.bindingModelHandle);
  if (modelHandle === null) return false;
  if (modelHandle !== undefined && modelHandle !== session.modelHandle) return false;
  return true;
}

function hasNonNullOwnProperty(value, names) {
  if (!value || typeof value !== "object") return false;
  return names.some((name) => Object.prototype.hasOwnProperty.call(value, name) && value[name] !== undefined && value[name] !== null);
}

function requiresLlamaExecutorModelResourceEvidence(session) {
  return Array.isArray(session.modelResources) && session.modelResources.length > 0;
}

function requiresLlamaExecutorTableEvidence(session) {
  return session.table !== undefined && session.table !== null;
}

function validateRequiredLlamaExecutorModelResourceEvidence(result, session) {
  if (!requiresLlamaExecutorModelResourceEvidence(session)) return true;
  if (!session.modelTable || !session.modelManifest) return false;
  return (
    hasNonNullOwnProperty(result, ["modelTableHash", "modelResourceTableHash", "weightTableHash"]) &&
    hasNonNullOwnProperty(result, ["modelManifestHash", "modelResourceManifestHash", "weightManifestHash", "modelTensorHash", "modelTensorMetadataHash"])
  );
}

function validateRequiredLlamaExecutorTableEvidence(result, session, status, runtime) {
  if (status !== runtime.status.ok || !requiresLlamaExecutorTableEvidence(session)) return true;
  return hasNonNullOwnProperty(result, ["tableHash", "bindingTableHash"]);
}

function validateLlamaExecutorModelTableHash(result, session) {
  const tableHash = normalizeLlamaExecutorHash(result.modelTableHash ?? result.modelResourceTableHash ?? result.weightTableHash);
  if (tableHash === null) return false;
  if (tableHash !== undefined) {
    if (!session.modelTable || tableHash !== session.modelTable.hash) return false;
  }
  const descriptorHash = normalizeLlamaExecutorHash(result.modelTableDescriptorHash ?? result.modelResourceDescriptorHash ?? result.weightDescriptorHash);
  if (descriptorHash === null) return false;
  if (descriptorHash !== undefined) {
    if (!session.modelTable || descriptorHash !== session.modelTable.descriptorHash) return false;
  }
  return true;
}

function validateLlamaExecutorModelManifestHash(result, session) {
  const manifestHash = normalizeLlamaExecutorHash(
    result.modelManifestHash ??
    result.modelResourceManifestHash ??
    result.weightManifestHash ??
    result.modelTensorHash ??
    result.modelTensorMetadataHash,
  );
  if (manifestHash === null) return false;
  if (manifestHash !== undefined) {
    if (!session.modelManifest || manifestHash !== session.modelManifest.hash) return false;
  }
  return true;
}

function validateLlamaExecutorModelPackHash(result, session) {
  const layoutHash = normalizeLlamaExecutorHash(
    result.modelPackLayoutHash ??
    result.modelPackHash ??
    result.weightPackHash,
  );
  if (layoutHash === null) return false;
  const descriptorHash = normalizeLlamaExecutorHash(
    result.modelPackDescriptorHash ??
    result.modelPackResourceHash ??
    result.weightPackDescriptorHash,
  );
  if (descriptorHash === null) return false;
  const packs = Array.isArray(session.modelPacks) ? session.modelPacks : [];
  if (layoutHash === undefined && descriptorHash === undefined) return packs.length === 0;
  for (const pack of packs) {
    if (layoutHash !== undefined && pack.layoutHash !== layoutHash) continue;
    if (descriptorHash !== undefined && pack.descriptorHash !== descriptorHash) continue;
    return true;
  }
  return false;
}

function validateLlamaExecutorTableHash(result, session) {
  const tableHash = normalizeLlamaExecutorHash(result.tableHash ?? result.bindingTableHash);
  if (tableHash === null) return false;
  if (tableHash !== undefined) {
    if (!session.table || tableHash !== session.table.hash) return false;
  }
  const descriptorHash = normalizeLlamaExecutorHash(result.tableDescriptorHash ?? result.descriptorHash);
  if (descriptorHash === null) return false;
  if (descriptorHash !== undefined) {
    if (!session.table || descriptorHash !== session.table.descriptorHash) return false;
  }
  return true;
}

function normalizeLlamaTokenExecutorResult(result, session, execution, runtime, target = {}) {
  const needsExplicitEvidence =
    requiresLlamaExecutorModelResourceEvidence(session) ||
    requiresLlamaExecutorTableEvidence(session);
  if (result === undefined || result === null) {
    if (needsExplicitEvidence) {
      return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
    }
    return writeLlamaNormalizedExecutorResult(
      target,
      runtime.status.ok,
      defaultLlamaExecutorOutputLength(session, execution, runtime.status.ok, runtime),
    );
  }
  if (Number.isInteger(result)) {
    const status = normalizeLlamaExecutorStatusValue(result, runtime);
    if (status === null || (status === runtime.status.ok && needsExplicitEvidence)) {
      return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
    }
    return writeLlamaNormalizedExecutorResult(
      target,
      status,
      defaultLlamaExecutorOutputLength(session, execution, status, runtime),
    );
  }
  if (typeof result !== "object") {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  const status = normalizeLlamaExecutorStatus(result, runtime);
  if (status === null) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  if (!validateRequiredLlamaExecutorModelResourceEvidence(result, session)) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  if (!validateRequiredLlamaExecutorTableEvidence(result, session, status, runtime)) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  if (!validateLlamaExecutorModelHandle(result, session)) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  if (!validateLlamaExecutorModelTableHash(result, session)) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  if (!validateLlamaExecutorModelManifestHash(result, session)) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  if (!validateLlamaExecutorModelPackHash(result, session)) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  if (!validateLlamaExecutorTableHash(result, session)) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  const outputLength = result.outputLength ?? result.outputLen ?? defaultLlamaExecutorOutputLength(session, execution, status, runtime);
  if (!Number.isSafeInteger(outputLength) || outputLength < 0 || outputLength > session.vocabSize) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  if (!validateLlamaExecutorOutputLength(status, runtime, session, execution, outputLength)) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  const backendDispatchCount = normalizeLlamaExecutorCounter(result, "backendDispatchCount");
  const backendOpCount = normalizeLlamaExecutorCounter(result, "backendOpCount", backendDispatchCount ?? 0);
  const fallbackOpCount = normalizeLlamaExecutorCounter(result, "fallbackOpCount");
  const commandCount = normalizeLlamaExecutorCounter(result, "commandCount", (backendDispatchCount ?? 0) + (fallbackOpCount ?? 0));
  const commandOpCount = normalizeLlamaExecutorCounter(result, "commandOpCount", commandCount ?? 0);
  const genericDispatchCount = normalizeLlamaExecutorCounter(result, "genericDispatchCount");
  const scalarDispatchCount = normalizeLlamaExecutorCounter(result, "scalarDispatchCount");
  const windowDispatchCount = normalizeLlamaExecutorCounter(result, "windowDispatchCount");
  const usesGpuStorageBuffers = result.usesGpuStorageBuffers === true;
  if (
    backendDispatchCount === null ||
    backendOpCount === null ||
    fallbackOpCount === null ||
    commandCount === null ||
    commandOpCount === null ||
    genericDispatchCount === null ||
    scalarDispatchCount === null ||
    windowDispatchCount === null
  ) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  if (!validateLlamaExecutorCounterCoherence(status, runtime, {
    backendDispatchCount,
    backendOpCount,
    commandCount,
    commandOpCount,
    fallbackOpCount,
    genericDispatchCount,
    scalarDispatchCount,
    windowDispatchCount,
  })) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  if (!validateLlamaExecutorFailureEvidence(status, runtime, outputLength, {
    backendDispatchCount,
    backendOpCount,
    commandCount,
    commandOpCount,
    fallbackOpCount,
  })) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  if (!validateLlamaExecutorWorkEvidence(status, runtime, session, {
    commandCount,
  })) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  if (!validateLlamaExecutorGpuStorageEvidence(status, runtime, session, {
    backendDispatchCount,
    fallbackOpCount,
  }, usesGpuStorageBuffers)) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.invalidArgument, 0);
  }
  if (!validateLlamaExecutorRequiredGpuDispatch(status, runtime, session, {
    backendDispatchCount,
    fallbackOpCount,
  }, usesGpuStorageBuffers)) {
    return writeLlamaNormalizedExecutorResult(target, runtime.status.unsupported, 0);
  }
  return writeLlamaNormalizedExecutorResult(
    target,
    status,
    outputLength,
    backendOpCount,
    backendDispatchCount,
    fallbackOpCount,
    commandCount,
    commandOpCount,
    genericDispatchCount,
    scalarDispatchCount,
    windowDispatchCount,
    usesGpuStorageBuffers,
  );
}

function llamaTokenExecutorInspectionKind(executor) {
  if (executor === null || executor === undefined) return null;
  if (executor instanceof WasmWebGpuLlamaBlockPipelineExecutor) return "tiny-llama-block-pipeline";
  if (executor instanceof WasmWebGpuLlamaBlockProjectionExecutor) return "llama-block-projection";
  if (executor instanceof WasmWebGpuLlamaAttentionProjectionExecutor) return "llama-attention-projection";
  if (executor instanceof WasmWebGpuLlamaKvProjectionExecutor) return "llama-kv-projection";
  if (executor instanceof WasmWebGpuLlamaActivationLogitsExecutor) return "llama-activation-logits";
  if (executor instanceof WasmWebGpuLlamaRmsNormProjectionExecutor) return "llama-rmsnorm-projection";
  if (executor instanceof WasmWebGpuLlamaEmbeddingProjectionExecutor) return "llama-embedding-projection";
  if (executor instanceof WasmWebGpuLlamaTokenWindowPatternExecutor) return "llama-token-window-pattern";
  if (executor instanceof WasmWebGpuLlamaTokenWindowExecutor) return "llama-token-window";
  if (typeof executor === "function") return "function";
  return executor.constructor?.name ?? "custom";
}

function isLlamaBoundedProofExecutor(executor) {
  return executor instanceof WasmWebGpuLlamaBlockPipelineExecutor ||
    executor instanceof WasmWebGpuLlamaBlockProjectionExecutor ||
    executor instanceof WasmWebGpuLlamaAttentionProjectionExecutor ||
    executor instanceof WasmWebGpuLlamaKvProjectionExecutor ||
    executor instanceof WasmWebGpuLlamaActivationLogitsExecutor ||
    executor instanceof WasmWebGpuLlamaRmsNormProjectionExecutor ||
    executor instanceof WasmWebGpuLlamaEmbeddingProjectionExecutor ||
    executor instanceof WasmWebGpuLlamaTokenWindowPatternExecutor ||
    executor instanceof WasmWebGpuLlamaTokenWindowExecutor;
}

function llamaExecutionCoverageInspection(executor) {
  if (executor === null || executor === undefined) {
    return Object.freeze({
      boundedProofExecutionSupported: false,
      executionCoverage: "resource-probe",
      executionCoverageReason: "no-token-executor",
      fullDefaultExecutionSupported: false,
      tokenExecutionSupported: false,
    });
  }
  if (isLlamaBoundedProofExecutor(executor)) {
    return Object.freeze({
      boundedProofExecutionSupported: true,
      executionCoverage: "bounded-proof",
      executionCoverageReason: "llama-family-bounded-proof-executor",
      fullDefaultExecutionSupported: false,
      tokenExecutionSupported: true,
    });
  }
  return Object.freeze({
    boundedProofExecutionSupported: false,
    executionCoverage: "custom-executor",
    executionCoverageReason: "custom-token-executor",
    fullDefaultExecutionSupported: false,
    tokenExecutionSupported: true,
  });
}

function llamaExecutorRequiredGpuStorageBufferCount(executor) {
  if (executor === null || executor === undefined) return null;
  const value = typeof executor.requiredGpuStorageBufferCount === "function"
    ? executor.requiredGpuStorageBufferCount()
    : executor.requiredGpuStorageBufferCount;
  if (value === undefined || value === null) return null;
  if (!Number.isSafeInteger(value) || value < 0) return null;
  return value;
}

function requiredGpuDispatchPreflightStatus(executor, session, runtime) {
  if (session.requireGpuDispatch !== true) return runtime.status.ok;
  const device = session.device;
  if (!(device instanceof WasmWebGpuDevice) || !device.canCreateGpuBuffers()) {
    return runtime.status.unsupported;
  }
  const requiredStorageBufferCount = llamaExecutorRequiredGpuStorageBufferCount(executor);
  if (
    requiredStorageBufferCount !== null &&
    !device.canBindStorageBuffers(requiredStorageBufferCount)
  ) {
    return runtime.status.unsupported;
  }
  return runtime.status.ok;
}

function writeLlamaTokenExecutorContext(context, session, execution, runtime, stepParams) {
  context.bindings = session.bindings;
  context.device = session.device;
  context.execution = execution;
  context.modelBindingKind = session.modelBindingKind;
  context.modelHandle = session.modelHandle;
  context.modelManifest = session.modelManifest;
  context.modelResources = session.modelResources;
  context.modelTable = session.modelTable;
  context.resultScratch = session.tokenExecutorResultScratch ?? null;
  context.runtime = runtime;
  context.session = session;
  context.stepParams = stepParams;
  context.table = session.table;
  return context;
}

function callLlamaTokenExecutor(executor, session, execution, runtime, stepParams, contextScratch = null, outcomeScratch = null) {
  const context = writeLlamaTokenExecutorContext(contextScratch ?? {}, session, execution, runtime, stepParams);
  const raw = typeof executor === "function"
    ? executor(context)
    : executor.executeTokensSync(context);
  return normalizeLlamaTokenExecutorResult(raw, session, execution, runtime, outcomeScratch ?? {});
}

function writeRuntimeProfileU64(view, ptr, offset, value) {
  const bigint = typeof value === "bigint" ? value : BigInt(value);
  view.setBigUint64(ptr + offset, bigint, true);
}

function zeroRuntimeProfile(view, ptr) {
  for (let offset = 0; offset <= 160; offset += 8) {
    view.setBigUint64(ptr + offset, 0n, true);
  }
}

function writeHostRuntimeProfile(view, ptr, fields = {}) {
  zeroRuntimeProfile(view, ptr);
  writeRuntimeProfileU64(view, ptr, 0, fields.callCount ?? 0);
  writeRuntimeProfileU64(view, ptr, 8, fields.backendOpCount ?? 0);
  writeRuntimeProfileU64(view, ptr, 16, fields.fallbackOpCount ?? 0);
  writeRuntimeProfileU64(view, ptr, 24, fields.backendDispatchCount ?? 0);
  writeRuntimeProfileU64(view, ptr, 32, fields.syncCount ?? 0);
  writeRuntimeProfileU64(view, ptr, 40, fields.runtimePatchCallCount ?? 0);
  writeRuntimeProfileU64(view, ptr, 48, fields.runtimePatchChangedCount ?? 0);
  writeRuntimeProfileU64(view, ptr, 56, fields.runtimePatchInvalidCount ?? 0);
  writeRuntimeProfileU64(view, ptr, 96, fields.commandCount ?? 0);
  writeRuntimeProfileU64(view, ptr, 112, fields.commandOpCount ?? 0);
}

function writeHostSessionInspection(view, ptr, fields = {}) {
  new Uint8Array(view.buffer, ptr, 80).fill(0);
  view.setUint32(ptr, fields.modelKind ?? 0, true);
  view.setUint32(ptr + 4, fields.backend ?? zgmlBackendWebGpu, true);
  view.setUint32(ptr + 8, fields.outputStorage ?? zgmlBufferStorageNone, true);
  view.setUint32(ptr + 12, fields.kvCacheStorage ?? zgmlBufferStorageNone, true);
  writeRuntimeProfileU64(view, ptr, 16, fields.position ?? 0);
  writeRuntimeProfileU64(view, ptr, 24, fields.contextLength ?? 0);
  writeRuntimeProfileU64(view, ptr, 32, fields.persistentBindingCount ?? 0);
  writeRuntimeProfileU64(view, ptr, 40, fields.stepInputCount ?? 0);
  writeRuntimeProfileU64(view, ptr, 48, fields.stepOutputCount ?? 0);
  writeRuntimeProfileU64(view, ptr, 56, fields.hostBindingCount ?? 0);
  writeRuntimeProfileU64(view, ptr, 64, fields.resourceBindingCount ?? 0);
  writeRuntimeProfileU64(view, ptr, 72, fields.bindingShapeHash ?? 0);
}

function freezeLlamaResourceBindings(bindings = {}) {
  return Object.freeze({
    output: bindings.output,
    activation: bindings.activation ?? null,
    activationInput: bindings.activationInput ?? null,
    kvCache: Object.freeze((bindings.kvCache ?? []).map((entry) => Object.freeze({
      k: entry.k,
      v: entry.v,
    }))),
    modelResources: Object.freeze((bindings.modelResources ?? []).slice()),
  });
}

function freezeLlamaSessionModelResources(resources = []) {
  return Object.freeze(resources.map((entry) => Object.freeze({
    ...entry,
    shape: Object.freeze(Array.isArray(entry.shape) ? entry.shape.slice() : []),
  })));
}

function normalizeLlamaTokenExecutorPresetName(value) {
  if (value === undefined || value === null || value === false) return null;
  if (typeof value !== "string") {
    throw new Error("WasmWebGpuLlamaResourceProgram executor preset must be a string");
  }
  const normalized = value.trim().toLowerCase().replaceAll("_", "-").replaceAll(" ", "-");
  if (
    normalized === "tiny-llama-block-pipeline" ||
    normalized === "llama-block-pipeline" ||
    normalized === "block-pipeline" ||
    normalized === "tiny-llama"
  ) {
    return "tiny-llama-block-pipeline";
  }
  throw new Error(`unsupported LLaMA token executor preset: ${value}`);
}

function hasLlamaExecutorOption(options, key) {
  return Object.prototype.hasOwnProperty.call(options, key) && options[key] !== undefined && options[key] !== null;
}

function defaultLlamaPresetLayerCount(kvRequirements) {
  const layers = kvRequirements?.layers;
  return Number.isSafeInteger(layers) && layers > 0 ? layers : 1;
}

function createLlamaTokenExecutorPreset(preset, options, kvRequirements) {
  const executorOptions = {
    ...(options.tokenExecutorOptions ?? {}),
    ...(options.executorOptions ?? {}),
  };
  if (!hasLlamaExecutorOption(executorOptions, "attentionHeadSize") && hasLlamaExecutorOption(options, "attentionHeadSize")) {
    executorOptions.attentionHeadSize = options.attentionHeadSize;
  }
  if (!hasLlamaExecutorOption(executorOptions, "headSize") && hasLlamaExecutorOption(options, "headSize")) {
    executorOptions.headSize = options.headSize;
  }
  if (!hasLlamaExecutorOption(executorOptions, "epsilon") && hasLlamaExecutorOption(options, "epsilon")) {
    executorOptions.epsilon = options.epsilon;
  }
  if (!hasLlamaExecutorOption(executorOptions, "rmsNormEpsilon") && hasLlamaExecutorOption(options, "rmsNormEpsilon")) {
    executorOptions.rmsNormEpsilon = options.rmsNormEpsilon;
  }
  if (!hasLlamaExecutorOption(executorOptions, "rmsNormEps") && hasLlamaExecutorOption(options, "rmsNormEps")) {
    executorOptions.rmsNormEps = options.rmsNormEps;
  }
  if (!hasLlamaExecutorOption(executorOptions, "ropeBase") && hasLlamaExecutorOption(options, "ropeBase")) {
    executorOptions.ropeBase = options.ropeBase;
  }
  if (!hasLlamaExecutorOption(executorOptions, "ropeTheta") && hasLlamaExecutorOption(options, "ropeTheta")) {
    executorOptions.ropeTheta = options.ropeTheta;
  }
  if (!hasLlamaExecutorOption(executorOptions, "ropeScale") && hasLlamaExecutorOption(options, "ropeScale")) {
    executorOptions.ropeScale = options.ropeScale;
  }
  if (!hasLlamaExecutorOption(executorOptions, "rope_scale") && hasLlamaExecutorOption(options, "rope_scale")) {
    executorOptions.rope_scale = options.rope_scale;
  }
  if (!hasLlamaExecutorOption(executorOptions, "ropeScalingFactor") && hasLlamaExecutorOption(options, "ropeScalingFactor")) {
    executorOptions.ropeScalingFactor = options.ropeScalingFactor;
  }
  if (!hasLlamaExecutorOption(executorOptions, "rope_scaling_factor") && hasLlamaExecutorOption(options, "rope_scaling_factor")) {
    executorOptions.rope_scaling_factor = options.rope_scaling_factor;
  }
  for (const key of [
    "ropeKind",
    "ropeType",
    "rope_kind",
    "rope_type",
    "ropeLowFreqFactor",
    "rope_low_freq_factor",
    "ropeHighFreqFactor",
    "rope_high_freq_factor",
    "ropeOriginalContextLength",
    "rope_original_context_length",
    "slidingWindow",
    "sliding_window",
    "attentionSlidingWindow",
    "attention_sliding_window",
    "useSlidingWindow",
    "use_sliding_window",
    "useRopeLayers",
    "use_rope_layers",
    "noRopeLayers",
    "no_rope_layers",
    "noRopeLayerInterval",
    "no_rope_layer_interval",
    "qkProjectionNorm",
    "qk_projection_norm",
    "qk_norm",
    "embeddingRole",
    "embedding_role",
    "lmHeadRole",
    "lm_head_role",
    "lmHeadBiasRole",
    "lm_head_bias_role",
    "normRole",
    "norm_role",
    "qNormRole",
    "kNormRole",
  ]) {
    if (!hasLlamaExecutorOption(executorOptions, key) && hasLlamaExecutorOption(options, key)) {
      executorOptions[key] = options[key];
    }
  }
  if (!hasLlamaExecutorOption(executorOptions, "record") && typeof options.record === "function") {
    executorOptions.record = options.record;
  }
  if (
    !hasLlamaExecutorOption(executorOptions, "blocks") &&
    !hasLlamaExecutorOption(executorOptions, "layers") &&
    !hasLlamaExecutorOption(executorOptions, "layerCount") &&
    !hasLlamaExecutorOption(executorOptions, "layer")
  ) {
    if (hasLlamaExecutorOption(options, "blocks")) executorOptions.blocks = options.blocks;
    else if (hasLlamaExecutorOption(options, "layers")) executorOptions.layers = options.layers;
    else if (hasLlamaExecutorOption(options, "layerCount")) executorOptions.layerCount = options.layerCount;
    else if (hasLlamaExecutorOption(options, "layer")) executorOptions.layer = options.layer;
    else executorOptions.layerCount = defaultLlamaPresetLayerCount(kvRequirements);
  }
  if (preset === "tiny-llama-block-pipeline") {
    return new WasmWebGpuLlamaBlockPipelineExecutor(executorOptions);
  }
  throw new Error(`unsupported LLaMA token executor preset: ${preset}`);
}

function resolveLlamaTokenExecutor(options, kvRequirements) {
  const executorFromOption = typeof options.executor === "string" ? null : (options.executor ?? null);
  const explicitExecutor = options.tokenExecutor ?? executorFromOption;
  const preset = normalizeLlamaTokenExecutorPresetName(
    options.executorPreset ?? options.executionPreset ?? options.execution ?? (typeof options.executor === "string" ? options.executor : null),
  );
  if (explicitExecutor !== null && preset !== null) {
    throw new Error("WasmWebGpuLlamaResourceProgram cannot combine tokenExecutor with executorPreset");
  }
  return explicitExecutor ?? (preset === null ? null : createLlamaTokenExecutorPreset(preset, options, kvRequirements));
}

function writeLlamaGenerationCallOptions(target, copiedKeys, options) {
  for (let i = 0; i < copiedKeys.length; i += 1) {
    delete target[copiedKeys[i]];
  }
  copiedKeys.length = 0;
  for (const key in options) {
    if (!Object.prototype.hasOwnProperty.call(options, key)) continue;
    target[key] = options[key];
    copiedKeys.push(key);
  }
  delete target.tokensLen;
  delete target.tokenLength;
  delete target.activeTokenLength;
  delete target.activeTokenCount;
  delete target.promptTokensLen;
  delete target.promptTokenLength;
  delete target.promptActiveTokenLength;
  delete target.promptActiveTokenCount;
  target.borrowTokens = true;
  target.includeLogits = false;
  target.recordHistory = false;
  target.reuseExecutorContext = true;
  target.reuseExecution = true;
  target.reuseStepParams = true;
  return target;
}

function writeLlamaGenerationArgmaxCallOptions(session, options) {
  const target = writeLlamaGenerationCallOptions(
    session.generationCallOptionsScratch,
    session.generationCallOptionKeysScratch,
    options,
  );
  target.seed = undefined;
  target.temperature = undefined;
  target.topK = undefined;
  return target;
}

function writeLlamaGenerationSampleCallOptions(session, options, sampleOptions) {
  const target = writeLlamaGenerationCallOptions(
    session.generationCallOptionsScratch,
    session.generationCallOptionKeysScratch,
    options,
  );
  target.seed = sampleOptions.seed;
  target.temperature = sampleOptions.temperature;
  target.topK = sampleOptions.topK;
  return target;
}

function normalizeLlamaDiagnosticHistoryLimit(value) {
  const limit = value ?? defaultLlamaDiagnosticHistoryLimit;
  if (!Number.isSafeInteger(limit) || limit < 0 || limit > 0xffffffff) {
    throw new Error(`invalid LLaMA diagnostic history limit: ${limit}`);
  }
  return limit;
}

function recordLlamaDiagnosticHistory(history, entry, limit) {
  if (limit === 0) return;
  if (history.length < limit) {
    history.push(entry);
    return;
  }
  if (limit === 1) {
    history[0] = entry;
    return;
  }
  history.copyWithin(0, 1);
  history[limit - 1] = entry;
}

function snapshotLlamaTokenExecutionForHistory(execution) {
  const tokensLen = execution.tokensLen;
  const tokens = new Array(tokensLen);
  for (let i = 0; i < tokensLen; i += 1) tokens[i] = execution.tokens[i];
  return Object.freeze({
    descPtr: execution.descPtr,
    outputLen: execution.outputLen,
    outputPolicy: execution.outputPolicy,
    outputPtr: execution.outputPtr,
    tokens: Object.freeze(tokens),
    tokensLen,
    tokensPtr: execution.tokensPtr,
  });
}

function snapshotLlamaStepParamsForHistory(stepParams) {
  return Object.freeze({
    contextLength: stepParams.contextLength,
    endPosition: stepParams.endPosition,
    kind: stepParams.kind,
    logitsLength: stepParams.logitsLength,
    outputKind: stepParams.outputKind,
    outputPolicy: stepParams.outputPolicy,
    requestedOutputLength: stepParams.requestedOutputLength,
    startPosition: stepParams.startPosition,
    tokenCount: stepParams.tokenCount,
  });
}

export class WasmWebGpuLlamaResourceSession {
  constructor(options = {}) {
    this.sessionHandle = options.sessionHandle ?? options.handle;
    this.bindings = freezeLlamaResourceBindings(options.bindings ?? {});
    this.device = options.device ?? null;
    this.table = options.table ?? null;
    this.modelTable = options.modelTable ?? null;
    this.modelManifest = options.modelManifest ?? null;
    this.modelResources = freezeLlamaSessionModelResources(options.modelResources ?? []);
    this.programInspection = options.programInspection ?? null;
    this.modelPacks = [];
    this.ownedResources = [];
    this.modelHandle = options.modelHandle ?? 0;
    this.modelBindingKind = this.modelHandle === 0 ? "program-default" : "compatible-model";
    this.modelKind = normalizeZgmlModelKind(
      options.modelKind ?? options.model_kind ?? options.kind ?? zgmlModelTinyLlamaKind,
      "LLaMA resource Session model kind",
    );
    this.nativeSession = options.nativeSession ?? true;
    this.freeNativeSession = options.freeNativeSession ?? null;
    this.runtime = options.runtime ?? null;
    this.requireGpuDispatch = resolveLlamaRequireGpuDispatch(options);
    this.tokenExecutor = options.tokenExecutor ?? null;
    this.deviceArgmaxSelector = options.deviceArgmaxSelector ?? new WasmWebGpuLlamaDeviceArgmaxSelector();
    this.ownsDeviceArgmaxSelector = options.deviceArgmaxSelector === undefined || options.deviceArgmaxSelector === null;
    this.deviceTopKSelector = options.deviceTopKSelector ?? new WasmWebGpuLlamaDeviceTopKSelector();
    this.ownsDeviceTopKSelector = options.deviceTopKSelector === undefined || options.deviceTopKSelector === null;
    this.fullLogitsSampleScratch = {
      candidateLogits: null,
      candidateTokens: null,
    };
    this.fullLogitsOutputScratch = null;
    this.generationSelectionScratch = resetLlamaSelectionResult({}, 0);
    this.generationContinuationScratch = [0];
    this.generationCallOptionsScratch = {};
    this.generationCallOptionKeysScratch = [];
    this.scalarTokenScratch = [0];
    this.scalarBorrowTokensInternal = {
      borrowTokens: true,
      recordHistory: false,
      reuseExecution: true,
      reuseExecutorContext: true,
      reuseStepParams: true,
    };
    this.noOutputExecuteInternal = {
      borrowTokens: true,
      outputPolicy: 0,
      readLogits: false,
      recordHistory: false,
      reuseExecution: true,
      reuseExecutorContext: true,
      reuseStepParams: true,
    };
	    this.vocabSize = options.vocabSize;
	    this.contextLength = options.contextLength;
	    this.maxTokenWindow = options.maxTokenWindow ?? this.contextLength;
	    this.position = options.position ?? 0;
    this.diagnosticHistoryLimit = normalizeLlamaDiagnosticHistoryLimit(
      options.diagnosticHistoryLimit ??
      options.historyLimit ??
      options.maxDiagnosticHistoryEntries ??
      options.maxHistoryEntries,
    );
    this.tokenExecutionAttempts = 0;
    this.lastTokenExecuteDescPtr = 0;
    this.lastTokenExecution = null;
    this.lastStepParams = null;
    this.tokenExecutionHistory = [];
    this.stepParamsHistory = [];
    this.tokenExecutionScratch = {
      descPtr: 0,
      outputLen: 0,
      outputPolicy: 0,
      outputPtr: 0,
      tokens: null,
      tokensLen: 0,
      tokensPtr: 0,
    };
    this.stepParamsScratch = {
      contextLength: 0,
      endPosition: 0,
      kind: "llama-token-window",
      logitsLength: 0,
      outputKind: 0,
      outputPolicy: 0,
      requestedOutputLength: 0,
      startPosition: 0,
      tokenCount: 0,
    };
    this.tokenExecutorContextScratch = {
      bindings: null,
      device: null,
      execution: null,
      modelBindingKind: "",
      modelHandle: 0,
      modelManifest: null,
      modelResources: null,
      modelTable: null,
      resultScratch: null,
      runtime: null,
      session: null,
      stepParams: null,
      table: null,
    };
    this.tokenExecutorResultScratch = {};
    this.tokenExecutorOutcomeScratch = {
      backendDispatchCount: 0,
      backendOpCount: 0,
      commandCount: 0,
      commandOpCount: 0,
      fallbackOpCount: 0,
      outputLength: 0,
      status: 0,
    };
    this.profile = createLlamaResourceProfile();
    this.freed = false;
    assertHandle(this.sessionHandle);
    if (!Number.isSafeInteger(this.vocabSize) || this.vocabSize <= 0) {
      throw new Error(`invalid LLaMA vocab size: ${this.vocabSize}`);
    }
	    if (!Number.isSafeInteger(this.contextLength) || this.contextLength <= 0) {
	      throw new Error(`invalid LLaMA context length: ${this.contextLength}`);
	    }
	    if (
	      !Number.isSafeInteger(this.maxTokenWindow) ||
	      this.maxTokenWindow <= 0 ||
	      this.maxTokenWindow > this.contextLength
	    ) {
	      throw new Error(`invalid LLaMA max token window: ${this.maxTokenWindow}`);
	    }
	    if (!Number.isSafeInteger(this.position) || this.position < 0 || this.position > this.contextLength) {
	      throw new Error(`invalid LLaMA position: ${this.position}`);
    }
    assertU32(this.modelHandle, "LLaMA resource Session model handle");
  }

  get handle() {
    if (this.freed) throw new Error("LLaMA WebGPU resource Session was already freed");
    return this.sessionHandle;
  }

  recordResult(resultPtr, outputLength, status, runtime) {
    runtime.writeStepResult(resultPtr, outputLength);
    if (resultPtr !== undefined && resultPtr !== null && resultPtr !== 0) {
      this.profile.resultWriteCount += 1;
    }
    this.profile.lastStatus = status;
    this.profile.lastResultOutputLength = outputLength;
    return status;
  }

  recordDecodedExecution(execution) {
    this.profile.decodedCallCount += 1;
    this.profile.lastOutputPolicy = execution.outputPolicy;
    this.profile.lastOutputLength = execution.outputLen;
    if (execution.outputPolicy === 0) {
      this.profile.noOutputCallCount += 1;
    } else if (execution.outputPolicy === 1) {
      this.profile.logitsOutputCallCount += 1;
      this.profile.requestedOutputElementCount += execution.outputLen;
      if (execution.outputPtr === 0 && execution.outputLen === 0) {
        this.profile.boundOutputCallCount += 1;
      } else {
        this.profile.callerOutputCallCount += 1;
      }
    }
  }

  beginTokenExecution(descPtr = 0) {
    this.profile.callCount += 1;
    this.tokenExecutionAttempts += 1;
    this.lastTokenExecuteDescPtr = descPtr;
    this.lastTokenExecution = null;
    this.lastStepParams = null;
  }

  executeTokensSync(descPtr = 0, resultPtr = 0, runtime) {
    if (this.freed) return runtime.status.unsupported;
    this.beginTokenExecution(descPtr);
    if (descPtr === 0) {
      this.profile.invalidArgumentCount += 1;
      return this.recordResult(resultPtr, 0, runtime.status.invalidArgument, runtime);
    }
    try {
      this.lastTokenExecution = decodeTokenExecuteDesc(runtime.memoryView(), descPtr);
    } catch {
      this.profile.invalidArgumentCount += 1;
      return this.recordResult(resultPtr, 0, runtime.status.invalidArgument, runtime);
    }
    return this.executeDecodedTokensSync(this.lastTokenExecution, resultPtr, runtime).status;
  }

  executeDecodedTokensSync(execution, resultPtr = 0, runtime = this.runtime, options = {}) {
    return this.executeDecodedTokensSyncInto({}, execution, resultPtr, runtime, options);
  }

  executeDecodedTokensSyncInto(target, execution, resultPtr = 0, runtime = this.runtime, options = {}) {
    if (!(runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("LLaMA WebGPU resource Session needs a runtime to execute tokens");
    }
    const recordHistory = options.recordHistory !== false;
    this.lastTokenExecution = execution;
    this.recordDecodedExecution(this.lastTokenExecution);
    const validation = validateLlamaTokenExecution(this.lastTokenExecution, this, runtime);
    if (recordHistory && this.diagnosticHistoryLimit !== 0) {
      recordLlamaDiagnosticHistory(
        this.tokenExecutionHistory,
        snapshotLlamaTokenExecutionForHistory(this.lastTokenExecution),
        this.diagnosticHistoryLimit,
      );
    }
    if (validation !== runtime.status.ok) {
      if (validation === runtime.status.invalidArgument) {
        this.profile.invalidArgumentCount += 1;
      } else {
        this.profile.validationFailureCount += 1;
      }
      return writeLlamaExecutionOutcome(target, 0, this.recordResult(resultPtr, 0, validation, runtime));
    }
    this.lastStepParams = options.reuseStepParams === true
      ? writeLlamaStepParams(this.stepParamsScratch, this, this.lastTokenExecution)
      : createLlamaStepParams(this, this.lastTokenExecution);
    if (recordHistory && this.diagnosticHistoryLimit !== 0) {
      recordLlamaDiagnosticHistory(
        this.stepParamsHistory,
        snapshotLlamaStepParamsForHistory(this.lastStepParams),
        this.diagnosticHistoryLimit,
      );
    }
    if (this.tokenExecutor !== null) {
      const gpuPreflightStatus = requiredGpuDispatchPreflightStatus(this.tokenExecutor, this, runtime);
      if (gpuPreflightStatus !== runtime.status.ok) {
        this.profile.unsupportedCallCount += 1;
        return writeLlamaExecutionOutcome(target, 0, this.recordResult(resultPtr, 0, gpuPreflightStatus, runtime));
      }
      this.profile.executorCallCount += 1;
      let outcome;
      const modelPackCountBeforeExecutor = Array.isArray(this.modelPacks) ? this.modelPacks.length : 0;
      const ownedResourceCountBeforeExecutor = Array.isArray(this.ownedResources) ? this.ownedResources.length : 0;
      try {
        outcome = callLlamaTokenExecutor(
          this.tokenExecutor,
          this,
          this.lastTokenExecution,
          runtime,
          this.lastStepParams,
          options.reuseExecutorContext === true ? this.tokenExecutorContextScratch : null,
          options.reuseExecutorContext === true ? this.tokenExecutorOutcomeScratch : null,
        );
      } catch {
        if (Array.isArray(this.modelPacks) && this.modelPacks.length > modelPackCountBeforeExecutor) {
          this.modelPacks.length = modelPackCountBeforeExecutor;
        }
        rollbackLlamaSessionOwnedResources(this, ownedResourceCountBeforeExecutor);
        this.profile.executorFailureCount += 1;
        return writeLlamaExecutionOutcome(target, 0, this.recordResult(resultPtr, 0, runtime.status.invalidArgument, runtime));
      }
      if (outcome.status === runtime.status.ok) {
        this.profile.executorOkCount += 1;
        this.profile.executorBackendOpCount += outcome.backendOpCount;
        this.profile.executorBackendDispatchCount += outcome.backendDispatchCount;
        this.profile.executorFallbackOpCount += outcome.fallbackOpCount;
        this.profile.executorCommandCount += outcome.commandCount;
        this.profile.executorCommandOpCount += outcome.commandOpCount;
        this.profile.executorGenericDispatchCount += outcome.genericDispatchCount;
        this.profile.executorScalarDispatchCount += outcome.scalarDispatchCount;
        this.profile.executorWindowDispatchCount += outcome.windowDispatchCount;
        this.position = this.lastStepParams.endPosition;
      } else {
        if (Array.isArray(this.modelPacks) && this.modelPacks.length > modelPackCountBeforeExecutor) {
          this.modelPacks.length = modelPackCountBeforeExecutor;
        }
        rollbackLlamaSessionOwnedResources(this, ownedResourceCountBeforeExecutor);
        this.profile.executorFailureCount += 1;
      }
      const outputLen = outcome.status === runtime.status.ok ? outcome.outputLength : 0;
      return writeLlamaExecutionOutcome(target, outputLen, this.recordResult(resultPtr, outputLen, outcome.status, runtime));
    }
    this.profile.unsupportedCallCount += 1;
    return writeLlamaExecutionOutcome(target, 0, this.recordResult(resultPtr, 0, runtime.status.unsupported, runtime));
  }

  executeTokensSyncFromList(tokens, options = {}, forcedOutputPolicy = null) {
    return this.executeTokensSyncFromListInto({}, tokens, options, forcedOutputPolicy);
  }

  executeTokensSyncFromListInto(target, tokens, options = {}, forcedOutputPolicy = null, internal = null) {
    if (this.freed) {
      const runtime = options.runtime ?? this.runtime;
      return writeLlamaExecutionOutcome(target, 0, runtime?.status?.unsupported ?? 5);
    }
    const runtime = options.runtime ?? this.runtime;
    if (!(runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("LLaMA WebGPU resource Session needs a runtime to execute tokens");
    }
    const tokenWindow = normalizeLlamaTokenWindow(tokens, "LLaMA tokens", {
      borrow: options.borrowTokens === true || internal?.borrowTokens === true,
      tokensLen: options.tokensLen ?? internal?.tokensLen,
      tokenLength: options.tokenLength ?? internal?.tokenLength,
      activeTokenLength: options.activeTokenLength ?? internal?.activeTokenLength,
      activeTokenCount: options.activeTokenCount ?? internal?.activeTokenCount,
    });
    const outputPolicy = forcedOutputPolicy === null || forcedOutputPolicy === undefined
      ? normalizeLlamaSessionOutputPolicy(options.outputPolicy ?? options.output ?? options.outputs ?? options.logits)
      : normalizeLlamaSessionOutputPolicy(forcedOutputPolicy);
    const reuseExecution = options.reuseExecution === true || internal?.reuseExecution === true;
    const execution = reuseExecution
      ? writeLlamaExecutionFromTokenList(this.tokenExecutionScratch, tokenWindow.tokens, outputPolicy, tokenWindow.tokensLen)
      : writeLlamaExecutionFromTokenList({}, tokenWindow.tokens, outputPolicy, tokenWindow.tokensLen);
    this.beginTokenExecution(0);
    const executeOptions = internal?.recordHistory === false ||
      internal?.reuseExecutorContext === true ||
      internal?.reuseStepParams === true
      ? internal
      : options;
    return this.executeDecodedTokensSyncInto(target, execution, 0, runtime, executeOptions);
  }

  async executeTokens(tokens, options = {}, internal = null) {
    const runtime = options.runtime ?? this.runtime;
    if (!(runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("LLaMA WebGPU resource Session needs a runtime to execute tokens");
    }
    const outputPolicy = internal?.outputPolicy === undefined
      ? normalizeLlamaSessionOutputPolicy(options.outputPolicy ?? options.output ?? options.outputs ?? options.logits)
      : normalizeLlamaSessionOutputPolicy(internal.outputPolicy);
    const outcome = this.executeTokensSyncFromListInto({}, tokens, options, outputPolicy, internal);
    const result = {
      outputLen: outcome.outputLen,
      position: this.position,
      status: outcome.status,
    };
    const readLogits = internal?.readLogits ?? options.readLogits;
    if (outcome.status === runtime.status.ok && outputPolicy === 1 && readLogits !== false) {
      result.logits = await this.output();
    }
    return result;
  }

  scalarTokenList(token) {
    this.scalarTokenScratch[0] = token;
    return this.scalarTokenScratch;
  }

  async step(token, options = {}) {
    return this.executeTokens(this.scalarTokenList(token), options, this.scalarBorrowTokensInternal);
  }

  async prefill(tokens, options = {}) {
    return this.executeTokens(tokens, options);
  }

  async advanceTokens(tokens, options = {}) {
    return this.executeTokens(tokens, options, this.noOutputExecuteInternal);
  }

  async argmaxInto(target, options = {}) {
    const runtime = options.runtime ?? this.runtime;
    if (!(runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("LLaMA WebGPU resource Session needs a runtime to select tokens");
    }
    resetLlamaSelectionFields(target, runtime.status.ok);
    if (shouldRejectRequiredGpuHostSelection(this)) {
      return writeLlamaRequiredGpuSelectionUnsupported(target, this, runtime);
    }
    const logits = options.logits ?? await this.outputForSelection(options.includeLogits === true);
    writeLlamaArgmaxSelection(target, logits);
    target.position = this.position;
    if (options.includeLogits === true) target.logits = logits;
    return target;
  }

  async argmax(options = {}) {
    const selected = await this.argmaxInto({}, options);
    const result = {
      logit: selected.logit,
      position: selected.position,
      status: selected.status,
      token: selected.token,
    };
    if (selected.logits !== undefined) result.logits = selected.logits;
    return result;
  }

  async argmaxDeviceInto(target, options = {}, ignoreLogits = false) {
    const runtime = options.runtime ?? this.runtime;
    if (!(runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("LLaMA WebGPU resource Session needs a runtime to select tokens");
    }
    resetLlamaSelectionFields(target, runtime.status.ok);
    let selected;
    let logits = ignoreLogits === true ? undefined : options.logits;
    if (shouldRejectRequiredGpuDeviceSelection(this, options, logits)) {
      return writeLlamaRequiredGpuSelectionUnsupported(target, this, runtime);
    }
    this.profile.selectionCallCount += 1;
    if (logits !== undefined && logits !== null) {
      writeLlamaArgmaxSelection(target, logits);
      this.profile.selectionFallbackOpCount += 1;
    } else if (canUseLlamaDeviceSelection(this, options)) {
      try {
        const rawSelected = typeof this.deviceArgmaxSelector.selectInto === "function"
          ? await this.deviceArgmaxSelector.selectInto(target, this)
          : await this.deviceArgmaxSelector.select(this);
        selected = writeNormalizedLlamaDeviceSelection(target, rawSelected, this);
      } catch {
        resetLlamaSelectionFields(target, runtime.status.invalidArgument);
        target.position = this.position;
        return target;
      }
      this.profile.selectionBackendDispatchCount += selected.backendDispatchCount;
      this.profile.selectionResultReadCount += selected.resultReadCount;
      this.profile.selectionSyncCount += selected.syncCount;
    } else {
      logits = options.includeLogits === true ? await this.output() : await this.outputScratch();
      writeLlamaArgmaxSelection(target, logits);
      this.profile.selectionFallbackOpCount += 1;
    }
    target.position = this.position;
    target.status = runtime.status.ok;
    target.backendDispatchCount = selected?.backendDispatchCount ?? target.backendDispatchCount ?? 0;
    target.resultReadCount = selected?.resultReadCount ?? target.resultReadCount ?? 0;
    target.syncCount = selected?.syncCount ?? target.syncCount ?? 0;
    if (options.includeLogits === true) {
      if (logits === undefined || logits === null) logits = await this.output();
      target.logits = logits;
    }
    return target;
  }

  async argmaxDevice(options = {}, internal = {}) {
    const selected = await this.argmaxDeviceInto({}, options, internal.ignoreLogits === true);
    const result = {
      backendDispatchCount: selected.backendDispatchCount,
      logit: selected.logit,
      position: selected.position,
      resultReadCount: selected.resultReadCount,
      status: selected.status,
      syncCount: selected.syncCount,
      token: selected.token,
    };
    if (selected.logits !== undefined) result.logits = selected.logits;
    return result;
  }

  async executeTokensArgmax(tokens, options = {}, internal = null) {
    const selected = await this.executeTokensArgmaxInto({}, tokens, options, internal);
    const result = {
      logit: selected.logit,
      outputLen: selected.outputLen,
      position: selected.position,
      status: selected.status,
      token: selected.token,
    };
    if (selected.logits !== undefined) result.logits = selected.logits;
    return result;
  }

  async executeTokensArgmaxInto(target, tokens, options = {}, internal = null) {
    const runtime = options.runtime ?? this.runtime;
    if (!(runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("LLaMA WebGPU resource Session needs a runtime to select tokens");
    }
    resetLlamaSelectionResult(target, runtime.status.ok);
    if (shouldRejectRequiredGpuHostSelection(this)) {
      return writeLlamaRequiredGpuSelectionUnsupported(target, this, runtime);
    }
    this.executeTokensSyncFromListInto(target, tokens, options, 1, internal);
    target.position = this.position;
    if (target.status !== runtime.status.ok) {
      return target;
    }
    const logits = await this.outputForSelection(options.includeLogits === true);
    writeLlamaArgmaxSelection(target, logits);
    if (options.includeLogits === true) target.logits = logits;
    return target;
  }

  async executeTokensArgmaxDevice(tokens, options = {}, internal = null) {
    const selected = await this.executeTokensArgmaxDeviceInto({}, tokens, options, internal);
    const result = {
      backendDispatchCount: selected.backendDispatchCount,
      logit: selected.logit,
      outputLen: selected.outputLen,
      position: selected.position,
      resultReadCount: selected.resultReadCount,
      status: selected.status,
      syncCount: selected.syncCount,
      token: selected.token,
    };
    if (selected.logits !== undefined) result.logits = selected.logits;
    return result;
  }

  async executeTokensArgmaxDeviceInto(target, tokens, options = {}, internal = null) {
    const runtime = options.runtime ?? this.runtime;
    if (!(runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("LLaMA WebGPU resource Session needs a runtime to select tokens");
    }
    resetLlamaSelectionResult(target, runtime.status.ok);
    if (shouldRejectRequiredGpuDeviceSelection(this, options)) {
      return writeLlamaRequiredGpuSelectionUnsupported(target, this, runtime);
    }
    this.executeTokensSyncFromListInto(target, tokens, options, 1, internal);
    target.position = this.position;
    if (target.status !== runtime.status.ok) {
      return target;
    }
    const outputLen = target.outputLen;
    await this.argmaxDeviceInto(target, options, true);
    target.outputLen = outputLen;
    return target;
  }

  async stepArgmax(token, options = {}) {
    return this.executeTokensArgmax(this.scalarTokenList(token), options, this.scalarBorrowTokensInternal);
  }

  async stepArgmaxDevice(token, options = {}) {
    return this.executeTokensArgmaxDevice(this.scalarTokenList(token), options, this.scalarBorrowTokensInternal);
  }

  async sampleInto(target, options = {}, normalizedSampleOptions = false) {
    const runtime = options.runtime ?? this.runtime;
    if (!(runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("LLaMA WebGPU resource Session needs a runtime to select tokens");
    }
    resetLlamaSelectionFields(target, runtime.status.ok);
    if (shouldRejectRequiredGpuHostSelection(this)) {
      return writeLlamaRequiredGpuSelectionUnsupported(target, this, runtime);
    }
    const logits = options.logits ?? await this.outputForSelection(options.includeLogits === true);
    try {
      writeLlamaTopKSampleSelection(
        target,
        logits,
        options,
        "LLaMA logits",
        this.fullLogitsSampleScratch,
        normalizedSampleOptions === true,
      );
    } catch {
      target.position = this.position;
      target.status = runtime.status.invalidArgument;
      return target;
    }
    target.position = this.position;
    if (options.includeLogits === true) target.logits = logits;
    return target;
  }

  async sample(options = {}) {
    const selected = await this.sampleInto({}, options);
    const result = {
      logit: selected.logit,
      position: selected.position,
      status: selected.status,
      token: selected.token,
    };
    if (selected.logits !== undefined) result.logits = selected.logits;
    return result;
  }

  async sampleDeviceInto(target, options = {}, ignoreLogits = false, normalizedSampleOptions = false) {
    const runtime = options.runtime ?? this.runtime;
    if (!(runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("LLaMA WebGPU resource Session needs a runtime to select tokens");
    }
    resetLlamaSelectionFields(target, runtime.status.ok);
    let sampleOptions;
    try {
      sampleOptions = normalizedSampleOptions === true ? options : normalizeLlamaSampleOptions(options);
    } catch {
      target.position = this.position;
      target.status = runtime.status.invalidArgument;
      return target;
    }
    let selected;
    let logits = ignoreLogits === true ? undefined : options.logits;
    if (shouldRejectRequiredGpuDeviceSelection(this, options, logits)) {
      return writeLlamaRequiredGpuSelectionUnsupported(target, this, runtime);
    }
    this.profile.selectionCallCount += 1;
    if (logits !== undefined && logits !== null) {
      try {
        writeLlamaTopKSampleSelection(target, logits, sampleOptions, "LLaMA logits", this.fullLogitsSampleScratch, true);
      } catch {
        target.position = this.position;
        target.status = runtime.status.invalidArgument;
        return target;
      }
      this.profile.selectionFallbackOpCount += 1;
    } else if (canUseLlamaDeviceSelection(this, options)) {
      try {
        const rawSelected = sampleOptions.topK === 1
          ? (typeof this.deviceArgmaxSelector.selectInto === "function"
            ? await this.deviceArgmaxSelector.selectInto(target, this)
            : await this.deviceArgmaxSelector.select(this))
          : (typeof this.deviceTopKSelector.selectInto === "function"
            ? await this.deviceTopKSelector.selectInto(target, this, sampleOptions)
            : await this.deviceTopKSelector.select(this, sampleOptions));
        selected = writeNormalizedLlamaDeviceSelection(target, rawSelected, this);
      } catch {
        resetLlamaSelectionFields(target, runtime.status.invalidArgument);
        target.position = this.position;
        return target;
      }
      this.profile.selectionBackendDispatchCount += selected.backendDispatchCount;
      this.profile.selectionResultReadCount += selected.resultReadCount;
      this.profile.selectionSyncCount += selected.syncCount;
    } else {
      logits = options.includeLogits === true ? await this.output() : await this.outputScratch();
      try {
        writeLlamaTopKSampleSelection(target, logits, sampleOptions, "LLaMA logits", this.fullLogitsSampleScratch, true);
      } catch {
        target.position = this.position;
        target.status = runtime.status.invalidArgument;
        return target;
      }
      this.profile.selectionFallbackOpCount += 1;
    }
    target.position = this.position;
    target.status = runtime.status.ok;
    target.backendDispatchCount = selected?.backendDispatchCount ?? target.backendDispatchCount ?? 0;
    target.resultReadCount = selected?.resultReadCount ?? target.resultReadCount ?? 0;
    target.syncCount = selected?.syncCount ?? target.syncCount ?? 0;
    if (options.includeLogits === true) {
      if (logits === undefined || logits === null) logits = await this.output();
      target.logits = logits;
    }
    return target;
  }

  async sampleDevice(options = {}, internal = {}) {
    const selected = await this.sampleDeviceInto(
      {},
      options,
      internal.ignoreLogits === true,
      internal.normalizedSampleOptions === true,
    );
    const result = {
      backendDispatchCount: selected.backendDispatchCount,
      logit: selected.logit,
      position: selected.position,
      resultReadCount: selected.resultReadCount,
      status: selected.status,
      syncCount: selected.syncCount,
      token: selected.token,
    };
    if (selected.logits !== undefined) result.logits = selected.logits;
    return result;
  }

  async executeTokensSample(tokens, options = {}, internal = null) {
    const selected = await this.executeTokensSampleInto({}, tokens, options, false, internal);
    const result = {
      logit: selected.logit,
      outputLen: selected.outputLen,
      position: selected.position,
      status: selected.status,
      token: selected.token,
    };
    if (selected.logits !== undefined) result.logits = selected.logits;
    return result;
  }

  async executeTokensSampleInto(target, tokens, options = {}, normalizedSampleOptions = false, internal = null) {
    const runtime = options.runtime ?? this.runtime;
    if (!(runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("LLaMA WebGPU resource Session needs a runtime to select tokens");
    }
    resetLlamaSelectionResult(target, runtime.status.ok);
    if (shouldRejectRequiredGpuHostSelection(this)) {
      return writeLlamaRequiredGpuSelectionUnsupported(target, this, runtime);
    }
    this.executeTokensSyncFromListInto(target, tokens, options, 1, internal);
    target.position = this.position;
    if (target.status !== runtime.status.ok) {
      return target;
    }
    const logits = await this.outputForSelection(options.includeLogits === true);
    try {
      writeLlamaTopKSampleSelection(
        target,
        logits,
        options,
        "LLaMA logits",
        this.fullLogitsSampleScratch,
        normalizedSampleOptions === true,
      );
    } catch {
      target.logit = 0;
      target.status = runtime.status.invalidArgument;
      target.token = 0;
      return target;
    }
    if (options.includeLogits === true) target.logits = logits;
    return target;
  }

  async stepSample(token, options = {}) {
    return this.executeTokensSample(this.scalarTokenList(token), options, this.scalarBorrowTokensInternal);
  }

  async executeTokensSampleDevice(tokens, options = {}, internal = null) {
    const selected = await this.executeTokensSampleDeviceInto({}, tokens, options, false, internal);
    const result = {
      backendDispatchCount: selected.backendDispatchCount,
      logit: selected.logit,
      outputLen: selected.outputLen,
      position: selected.position,
      resultReadCount: selected.resultReadCount,
      status: selected.status,
      syncCount: selected.syncCount,
      token: selected.token,
    };
    if (selected.logits !== undefined) result.logits = selected.logits;
    return result;
  }

  async executeTokensSampleDeviceInto(target, tokens, options = {}, normalizedSampleOptions = false, internal = null) {
    const runtime = options.runtime ?? this.runtime;
    if (!(runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("LLaMA WebGPU resource Session needs a runtime to select tokens");
    }
    resetLlamaSelectionResult(target, runtime.status.ok);
    if (shouldRejectRequiredGpuDeviceSelection(this, options)) {
      return writeLlamaRequiredGpuSelectionUnsupported(target, this, runtime);
    }
    this.executeTokensSyncFromListInto(target, tokens, options, 1, internal);
    target.position = this.position;
    if (target.status !== runtime.status.ok) {
      return target;
    }
    const outputLen = target.outputLen;
    await this.sampleDeviceInto(target, options, true, normalizedSampleOptions === true);
    target.outputLen = outputLen;
    return target;
  }

  async stepSampleDevice(token, options = {}) {
    return this.executeTokensSampleDevice(this.scalarTokenList(token), options, this.scalarBorrowTokensInternal);
  }

  async generateTokensArgmaxInto(promptTokens, outputTokens, options = {}) {
    const runtime = options.runtime ?? this.runtime;
    if (!(runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("LLaMA WebGPU resource Session needs a runtime to generate tokens");
    }
    const out = normalizeLlamaOutputTokenBuffer(outputTokens);
    const promptWindow = normalizeLlamaPromptTokenWindow(promptTokens, options);
    let nextTokens = promptWindow.tokens;
    let nextTokensLen = promptWindow.tokensLen;
    if (!llamaGenerationWindowFits(this, nextTokensLen, out.length)) {
      return llamaGenerationShapeMismatchResult(this, runtime);
    }
    if (shouldRejectRequiredGpuHostSelection(this)) {
      return llamaGenerationUnsupportedResult(this, runtime);
    }
    let tokensGenerated = 0;
    let lastToken = 0;
    let lastLogit = 0;
    let status = runtime.status.ok;
    let lastLogits;
    const continuationTokens = this.generationContinuationScratch;
    const callOptions = writeLlamaGenerationArgmaxCallOptions(this, options);
    for (let i = 0; i < out.length; i += 1) {
      callOptions.includeLogits = options.includeLogits === true && i === out.length - 1;
      callOptions.tokensLen = nextTokensLen;
      const selected = await this.executeTokensArgmaxInto(this.generationSelectionScratch, nextTokens, callOptions);
      status = selected.status;
      if (status !== runtime.status.ok) {
        break;
      }
      out[i] = selected.token;
      tokensGenerated += 1;
      lastToken = selected.token;
      lastLogit = selected.logit;
      if (selected.logits !== undefined) lastLogits = selected.logits;
      continuationTokens[0] = selected.token;
      nextTokens = continuationTokens;
      nextTokensLen = 1;
    }
    const result = {
      lastLogit,
      lastToken,
      position: this.position,
      status,
      tokensGenerated,
    };
    if (lastLogits !== undefined) result.logits = lastLogits;
    return result;
  }

  async generateTokensArgmax(promptTokens, maxTokensOrOptions = 1, maybeOptions = {}) {
    const options = typeof maxTokensOrOptions === "object" && maxTokensOrOptions !== null
      ? maxTokensOrOptions
      : maybeOptions;
    const maxTokens = normalizeLlamaGenerationCount(
      typeof maxTokensOrOptions === "object" && maxTokensOrOptions !== null
        ? (options.maxTokens ?? options.tokens ?? options.count)
        : maxTokensOrOptions,
    );
    const tokens = new Uint32Array(maxTokens);
    const result = await this.generateTokensArgmaxInto(promptTokens, tokens, options);
    return {
      ...result,
      tokens,
    };
  }

  async generateTokensArgmaxDeviceInto(promptTokens, outputTokens, options = {}) {
    const runtime = options.runtime ?? this.runtime;
    if (!(runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("LLaMA WebGPU resource Session needs a runtime to generate tokens");
    }
    const out = normalizeLlamaOutputTokenBuffer(outputTokens);
    const promptWindow = normalizeLlamaPromptTokenWindow(promptTokens, options);
    let nextTokens = promptWindow.tokens;
    let nextTokensLen = promptWindow.tokensLen;
    if (!llamaGenerationWindowFits(this, nextTokensLen, out.length)) {
      return llamaGenerationShapeMismatchResult(this, runtime);
    }
    let tokensGenerated = 0;
    let lastToken = 0;
    let lastLogit = 0;
    let status = runtime.status.ok;
    let lastLogits;
    const continuationTokens = this.generationContinuationScratch;
    const callOptions = writeLlamaGenerationArgmaxCallOptions(this, options);
    for (let i = 0; i < out.length; i += 1) {
      callOptions.includeLogits = options.includeLogits === true && i === out.length - 1;
      callOptions.tokensLen = nextTokensLen;
      const selected = await this.executeTokensArgmaxDeviceInto(this.generationSelectionScratch, nextTokens, callOptions);
      status = selected.status;
      if (status !== runtime.status.ok) {
        break;
      }
      out[i] = selected.token;
      tokensGenerated += 1;
      lastToken = selected.token;
      lastLogit = selected.logit;
      if (selected.logits !== undefined) lastLogits = selected.logits;
      continuationTokens[0] = selected.token;
      nextTokens = continuationTokens;
      nextTokensLen = 1;
    }
    const result = {
      lastLogit,
      lastToken,
      position: this.position,
      status,
      tokensGenerated,
    };
    if (lastLogits !== undefined) result.logits = lastLogits;
    return result;
  }

  async generateTokensArgmaxDevice(promptTokens, maxTokensOrOptions = 1, maybeOptions = {}) {
    const options = typeof maxTokensOrOptions === "object" && maxTokensOrOptions !== null
      ? maxTokensOrOptions
      : maybeOptions;
    const maxTokens = normalizeLlamaGenerationCount(
      typeof maxTokensOrOptions === "object" && maxTokensOrOptions !== null
        ? (options.maxTokens ?? options.tokens ?? options.count)
        : maxTokensOrOptions,
    );
    const tokens = new Uint32Array(maxTokens);
    const result = await this.generateTokensArgmaxDeviceInto(promptTokens, tokens, options);
    return {
      ...result,
      tokens,
    };
  }

  async generateTokensSampleInto(promptTokens, outputTokens, options = {}) {
    const runtime = options.runtime ?? this.runtime;
    if (!(runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("LLaMA WebGPU resource Session needs a runtime to generate tokens");
    }
    const out = normalizeLlamaOutputTokenBuffer(outputTokens);
    let sampleOptions;
    try {
      sampleOptions = normalizeLlamaSampleOptions(options);
    } catch {
      return {
        lastLogit: 0,
        lastToken: 0,
        position: this.position,
        status: runtime.status.invalidArgument,
        tokensGenerated: 0,
      };
    }
    const promptWindow = normalizeLlamaPromptTokenWindow(promptTokens, options);
    let nextTokens = promptWindow.tokens;
    let nextTokensLen = promptWindow.tokensLen;
    if (!llamaGenerationWindowFits(this, nextTokensLen, out.length)) {
      return llamaGenerationShapeMismatchResult(this, runtime);
    }
    if (shouldRejectRequiredGpuHostSelection(this)) {
      return llamaGenerationUnsupportedResult(this, runtime);
    }
    let tokensGenerated = 0;
    let lastToken = 0;
    let lastLogit = 0;
    let status = runtime.status.ok;
    let lastLogits;
    const continuationTokens = this.generationContinuationScratch;
    const callOptions = writeLlamaGenerationSampleCallOptions(this, options, sampleOptions);
    for (let i = 0; i < out.length; i += 1) {
      callOptions.includeLogits = options.includeLogits === true && i === out.length - 1;
      callOptions.tokensLen = nextTokensLen;
      callOptions.seed = (sampleOptions.seed + i) >>> 0;
      const selected = await this.executeTokensSampleInto(this.generationSelectionScratch, nextTokens, callOptions, true);
      status = selected.status;
      if (status !== runtime.status.ok) {
        break;
      }
      out[i] = selected.token;
      tokensGenerated += 1;
      lastToken = selected.token;
      lastLogit = selected.logit;
      if (selected.logits !== undefined) lastLogits = selected.logits;
      continuationTokens[0] = selected.token;
      nextTokens = continuationTokens;
      nextTokensLen = 1;
    }
    const result = {
      lastLogit,
      lastToken,
      position: this.position,
      status,
      tokensGenerated,
    };
    if (lastLogits !== undefined) result.logits = lastLogits;
    return result;
  }

  async generateTokensSample(promptTokens, maxTokensOrOptions = 1, maybeOptions = {}) {
    const options = typeof maxTokensOrOptions === "object" && maxTokensOrOptions !== null
      ? maxTokensOrOptions
      : maybeOptions;
    const maxTokens = normalizeLlamaGenerationCount(
      typeof maxTokensOrOptions === "object" && maxTokensOrOptions !== null
        ? (options.maxTokens ?? options.tokens ?? options.count)
        : maxTokensOrOptions,
    );
    const tokens = new Uint32Array(maxTokens);
    const result = await this.generateTokensSampleInto(promptTokens, tokens, options);
    return {
      ...result,
      tokens,
    };
  }

  async generateTokensSampleDeviceInto(promptTokens, outputTokens, options = {}) {
    const runtime = options.runtime ?? this.runtime;
    if (!(runtime instanceof WasmWebGpuSessionRuntime)) {
      throw new Error("LLaMA WebGPU resource Session needs a runtime to generate tokens");
    }
    const out = normalizeLlamaOutputTokenBuffer(outputTokens);
    let sampleOptions;
    try {
      sampleOptions = normalizeLlamaSampleOptions(options);
    } catch {
      return {
        lastLogit: 0,
        lastToken: 0,
        position: this.position,
        status: runtime.status.invalidArgument,
        tokensGenerated: 0,
      };
    }
    const promptWindow = normalizeLlamaPromptTokenWindow(promptTokens, options);
    let nextTokens = promptWindow.tokens;
    let nextTokensLen = promptWindow.tokensLen;
    if (!llamaGenerationWindowFits(this, nextTokensLen, out.length)) {
      return llamaGenerationShapeMismatchResult(this, runtime);
    }
    let tokensGenerated = 0;
    let lastToken = 0;
    let lastLogit = 0;
    let status = runtime.status.ok;
    let lastLogits;
    const continuationTokens = this.generationContinuationScratch;
    const callOptions = writeLlamaGenerationSampleCallOptions(this, options, sampleOptions);
    for (let i = 0; i < out.length; i += 1) {
      callOptions.includeLogits = options.includeLogits === true && i === out.length - 1;
      callOptions.tokensLen = nextTokensLen;
      callOptions.seed = (sampleOptions.seed + i) >>> 0;
      const selected = await this.executeTokensSampleDeviceInto(this.generationSelectionScratch, nextTokens, callOptions, true);
      status = selected.status;
      if (status !== runtime.status.ok) {
        break;
      }
      out[i] = selected.token;
      tokensGenerated += 1;
      lastToken = selected.token;
      lastLogit = selected.logit;
      if (selected.logits !== undefined) lastLogits = selected.logits;
      continuationTokens[0] = selected.token;
      nextTokens = continuationTokens;
      nextTokensLen = 1;
    }
    const result = {
      lastLogit,
      lastToken,
      position: this.position,
      status,
      tokensGenerated,
    };
    if (lastLogits !== undefined) result.logits = lastLogits;
    return result;
  }

  async generateTokensSampleDevice(promptTokens, maxTokensOrOptions = 1, maybeOptions = {}) {
    const options = typeof maxTokensOrOptions === "object" && maxTokensOrOptions !== null
      ? maxTokensOrOptions
      : maybeOptions;
    const maxTokens = normalizeLlamaGenerationCount(
      typeof maxTokensOrOptions === "object" && maxTokensOrOptions !== null
        ? (options.maxTokens ?? options.tokens ?? options.count)
        : maxTokensOrOptions,
    );
    const tokens = new Uint32Array(maxTokens);
    const result = await this.generateTokensSampleDeviceInto(promptTokens, tokens, options);
    return {
      ...result,
      tokens,
    };
  }

  async output() {
    if (this.freed) throw new Error("LLaMA WebGPU resource Session was already freed");
    const out = new Float32Array(this.vocabSize);
    await this.readOutputInto(out);
    return out;
  }

  async outputInto(target) {
    if (this.freed) throw new Error("LLaMA WebGPU resource Session was already freed");
    if (!(target instanceof Float32Array) || target.length < this.vocabSize) {
      throw new Error(`LLaMA output target must be a Float32Array with at least ${this.vocabSize} elements`);
    }
    await this.readOutputInto(target);
    return target;
  }

  async outputScratch() {
    if (this.freed) throw new Error("LLaMA WebGPU resource Session was already freed");
    if (!(this.fullLogitsOutputScratch instanceof Float32Array) || this.fullLogitsOutputScratch.length !== this.vocabSize) {
      this.fullLogitsOutputScratch = new Float32Array(this.vocabSize);
    }
    await this.readOutputInto(this.fullLogitsOutputScratch);
    return this.fullLogitsOutputScratch;
  }

  async outputForSelection(includeLogits) {
    return includeLogits ? this.output() : this.outputScratch();
  }

  async readOutputInto(target) {
    if (this.device && typeof this.device.readFloat32Into === "function") {
      await this.device.readFloat32Into(this.bindings.output, target, this.vocabSize);
    } else if (this.device && typeof this.device.readFloat32 === "function") {
      target.set((await this.device.readFloat32(this.bindings.output, this.vocabSize)).subarray(0, this.vocabSize));
    } else {
      throw new Error("LLaMA WebGPU resource Session device cannot read Float32 output");
    }
    this.profile.outputReadCount += 1;
    this.profile.syncCount += 1;
    return target;
  }

  runtimeProfile() {
    return { ...this.profile };
  }

  inspect() {
    if (this.freed) throw new Error("LLaMA WebGPU resource Session was already freed");
    const requiredGpuStorageBufferCount = llamaExecutorRequiredGpuStorageBufferCount(this.tokenExecutor);
    return Object.freeze({
      contextLength: this.contextLength,
      ...llamaExecutionCoverageInspection(this.tokenExecutor),
      executionMode: this.tokenExecutor === null ? "resource-probe" : "proof-executor",
      handle: this.sessionHandle,
      kind: "llama-resource-session",
      kvCacheLayers: this.bindings.kvCache.length,
      maxTokenWindow: this.maxTokenWindow,
      modelBindingKind: this.modelBindingKind,
      modelHandle: this.modelHandle,
      modelKind: this.modelKind,
      modelManifestHash: this.modelManifest?.hash ?? 0n,
      modelResourceCount: this.modelResources.length,
      modelResourceRoles: Object.freeze(this.modelResources.map((entry) => entry.role)),
      modelResources: Object.freeze(this.modelResources.map((entry) => llamaModelResourceRequirementInspection(entry))),
      modelTable: llamaResourceTableInspection(this.modelTable),
      nativeSession: this.nativeSession,
      outputByteLength: this.bindings.output?.byteLength ?? 0,
      position: this.position,
      program: this.programInspection,
      requiredGpuStorageBufferCount,
      requireGpuDispatch: this.requireGpuDispatch,
      table: llamaResourceTableInspection(this.table),
      tokenExecutorKind: llamaTokenExecutorInspectionKind(this.tokenExecutor),
      vocabSize: this.vocabSize,
    });
  }

  writeRuntimeProfile(view, ptr) {
    if (this.tokenExecutor === null) throw new Error("LLaMA resource-probe Session has no host runtime profile");
    writeHostRuntimeProfile(view, ptr, {
      callCount: this.profile.executorOkCount,
      backendOpCount: this.profile.executorBackendOpCount,
      fallbackOpCount: this.profile.executorFallbackOpCount,
      backendDispatchCount: this.profile.executorBackendDispatchCount,
      syncCount: this.profile.syncCount,
      runtimePatchCallCount: this.profile.executorCallCount,
      runtimePatchChangedCount: this.profile.executorOkCount,
      runtimePatchInvalidCount: this.profile.executorFailureCount,
      commandCount: this.profile.executorCommandCount,
      commandOpCount: this.profile.executorCommandOpCount,
    });
  }

  resetRuntimeProfile() {
    this.profile = createLlamaResourceProfile();
  }

  writeInspection(view, ptr) {
    const kvLayerCount = this.bindings.kvCache.length;
    const tableResourceCount = this.table?.resourceCount ?? (
      (this.bindings.output ? 1 : 0) +
      kvLayerCount * 2 +
      (this.bindings.activation ? 1 : 0) +
      (this.bindings.activationInput ? 1 : 0)
    );
    writeHostSessionInspection(view, ptr, {
      modelKind: this.modelKind,
      backend: zgmlBackendWebGpu,
      outputStorage: this.bindings.output ? zgmlBufferStorageExternalResource : zgmlBufferStorageNone,
      kvCacheStorage: kvLayerCount === 0 ? zgmlBufferStorageNone : zgmlBufferStorageExternalResource,
      position: this.position,
      contextLength: this.contextLength,
      persistentBindingCount: tableResourceCount + this.modelResources.length,
      stepInputCount: 1,
      stepOutputCount: this.bindings.output ? 1 : 0,
      hostBindingCount: 0,
      resourceBindingCount: tableResourceCount,
      bindingShapeHash: this.table?.hash ?? 0n,
    });
  }

  reset() {
    if (this.freed) throw new Error("LLaMA WebGPU resource Session was already freed");
    this.position = 0;
  }

  releaseLocal() {
    if (this.freed) return;
    destroyLlamaSessionOwnedResources(this);
    if (this.ownsDeviceArgmaxSelector) this.deviceArgmaxSelector.destroy();
    if (this.ownsDeviceTopKSelector) this.deviceTopKSelector.destroy();
    this.fullLogitsSampleScratch.candidateLogits = null;
    this.fullLogitsSampleScratch.candidateTokens = null;
    this.fullLogitsOutputScratch = null;
    this.modelPacks = [];
    this.freed = true;
    this.sessionHandle = 0;
  }

  free() {
    if (this.freed) return;
    const handle = this.sessionHandle;
    if (this.nativeSession !== false && typeof this.freeNativeSession === "function") {
      this.freeNativeSession(handle);
    }
    this.releaseLocal();
    if (this.runtime instanceof WasmWebGpuSessionRuntime) {
      this.runtime.unregister(handle);
    }
  }

  dispose() {
    this.free();
  }

  [Symbol.dispose]() {
    this.free();
  }
}

export class WasmWebGpuTinyLinearSession {
  constructor(options = {}) {
    this.program = options.program;
    this.sessionHandle = options.sessionHandle;
    this.bindings = options.bindings;
    this.table = options.table;
    this.profile = {
      callCount: 0,
      backendDispatchCount: 0,
      noOutputCallCount: 0,
      outputElementCount: 0,
      outputReadCount: 0,
      syncCount: 0,
      lastOutputLength: 0,
    };
    this.freed = false;
  }

  get handle() {
    if (this.freed) throw new Error("tiny-linear WebGPU Session was already freed");
    return this.sessionHandle;
  }

  recordExecution(outputLength) {
    assertU32(outputLength, "tiny-linear host output length");
    this.profile.callCount += 1;
    this.profile.backendDispatchCount += 1;
    this.profile.lastOutputLength = outputLength;
    if (outputLength === 0) {
      this.profile.noOutputCallCount += 1;
    } else {
      this.profile.outputElementCount += outputLength;
    }
  }

  async step(options = {}) {
    if (this.freed) throw new Error("tiny-linear WebGPU Session was already freed");
    await this.program.executor.execute({
      weights: this.bindings.weights.descriptor,
      bias: this.bindings.bias.descriptor,
      input: this.bindings.input.descriptor,
      output: this.bindings.output.descriptor,
      rows: this.program.rows,
      cols: this.program.cols,
    });
    this.recordExecution(options.outputLength ?? this.program.rows);
  }

  stepSync(options = {}) {
    if (this.freed) throw new Error("tiny-linear WebGPU Session was already freed");
    this.program.executor.submit({
      weights: this.bindings.weights.descriptor,
      bias: this.bindings.bias.descriptor,
      input: this.bindings.input.descriptor,
      output: this.bindings.output.descriptor,
      rows: this.program.rows,
      cols: this.program.cols,
    });
    this.recordExecution(options.outputLength ?? this.program.rows);
  }

  async output() {
    if (this.freed) throw new Error("tiny-linear WebGPU Session was already freed");
    const out = await this.program.device.readFloat32(this.bindings.output.descriptor, this.program.rows);
    this.profile.outputReadCount += 1;
    this.profile.syncCount += 1;
    return out;
  }

  runtimeProfile() {
    return { ...this.profile };
  }

  writeRuntimeProfile(view, ptr) {
    writeHostRuntimeProfile(view, ptr, {
      callCount: this.profile.callCount,
      backendOpCount: this.profile.backendDispatchCount,
      backendDispatchCount: this.profile.backendDispatchCount,
      syncCount: this.profile.syncCount,
      commandCount: this.profile.backendDispatchCount,
      commandOpCount: this.profile.backendDispatchCount,
    });
  }

  resetRuntimeProfile() {
    this.profile = {
      callCount: 0,
      backendDispatchCount: 0,
      noOutputCallCount: 0,
      outputElementCount: 0,
      outputReadCount: 0,
      syncCount: 0,
      lastOutputLength: 0,
    };
  }

  writeInspection(view, ptr) {
    writeHostSessionInspection(view, ptr, {
      modelKind: zgmlModelTinyLinearKind,
      backend: zgmlBackendWebGpu,
      outputStorage: zgmlBufferStorageExternalResource,
      kvCacheStorage: zgmlBufferStorageNone,
      position: 0,
      contextLength: 0,
      persistentBindingCount: 2,
      stepInputCount: 1,
      stepOutputCount: 1,
      hostBindingCount: 0,
      resourceBindingCount: this.table?.resourceCount ?? 4,
      bindingShapeHash: this.table?.hash ?? 0n,
    });
  }

  reset() {
    if (this.freed) throw new Error("tiny-linear WebGPU Session was already freed");
  }

  free() {
    if (this.freed) return;
    const handle = this.sessionHandle;
    this.program.freeSession(handle);
    if (!this.freed) {
      this.program.unregister(handle);
      this.freed = true;
      this.sessionHandle = 0;
    }
  }

  dispose() {
    this.free();
  }

  [Symbol.dispose]() {
    this.free();
  }
}

export async function requestBrowserWebGpuDevice() {
  if (!globalThis.navigator || !navigator.gpu) return null;
  const adapter = await withBrowserWebGpuRequestTimeout(navigator.gpu.requestAdapter(), 30_000, "requestAdapter");
  if (!adapter) return null;
  return withBrowserWebGpuRequestTimeout(adapter.requestDevice(), 30_000, "requestDevice");
}

function withBrowserWebGpuRequestTimeout(promise, timeoutMs, label) {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) return promise;
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

function browserWebGpuLimitSnapshot(source) {
  const limits = source?.limits;
  const names = [
    "maxBindGroups",
    "maxBindingsPerBindGroup",
    "maxBufferSize",
    "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupSizeX",
    "maxComputeWorkgroupSizeY",
    "maxComputeWorkgroupSizeZ",
    "maxComputeWorkgroupsPerDimension",
    "maxStorageBufferBindingSize",
    "maxStorageBuffersPerShaderStage",
    "minStorageBufferOffsetAlignment",
  ];
  const snapshot = {};
  for (const name of names) {
    const value = limits?.[name];
    if (Number.isFinite(value)) snapshot[name] = value;
  }
  return Object.freeze(snapshot);
}

function browserWebGpuFeatureSnapshot(source) {
  const features = source?.features;
  if (!features || typeof features[Symbol.iterator] !== "function") return Object.freeze([]);
  return Object.freeze(Array.from(features).map((feature) => String(feature)).sort());
}

export async function requestBrowserWebGpuDeviceInfo(options = {}) {
  if (!globalThis.navigator) {
    return Object.freeze({
      adapterFeatures: Object.freeze([]),
      adapterLimits: Object.freeze({}),
      available: false,
      device: null,
      deviceFeatures: Object.freeze([]),
      deviceLimits: Object.freeze({}),
      reason: "navigator unavailable",
    });
  }
  if (!navigator.gpu) {
    return Object.freeze({
      adapterFeatures: Object.freeze([]),
      adapterLimits: Object.freeze({}),
      available: false,
      device: null,
      deviceFeatures: Object.freeze([]),
      deviceLimits: Object.freeze({}),
      reason: "navigator.gpu unavailable",
    });
  }
  const timeoutMs = options.timeoutMs ?? 30_000;
  let adapter;
  try {
    adapter = await withBrowserWebGpuRequestTimeout(navigator.gpu.requestAdapter(options.adapterOptions ?? {}), timeoutMs, "requestAdapter");
  } catch (err) {
    return Object.freeze({
      adapterFeatures: Object.freeze([]),
      adapterLimits: Object.freeze({}),
      available: false,
      device: null,
      deviceFeatures: Object.freeze([]),
      deviceLimits: Object.freeze({}),
      reason: `requestAdapter failed: ${err && err.message ? err.message : err}`,
    });
  }
  if (!adapter) {
    return Object.freeze({
      adapterFeatures: Object.freeze([]),
      adapterLimits: Object.freeze({}),
      available: false,
      device: null,
      deviceFeatures: Object.freeze([]),
      deviceLimits: Object.freeze({}),
      reason: "no WebGPU adapter",
    });
  }
  let device;
  try {
    device = await withBrowserWebGpuRequestTimeout(adapter.requestDevice(options.deviceDescriptor ?? {}), timeoutMs, "requestDevice");
  } catch (err) {
    return Object.freeze({
      adapterFeatures: browserWebGpuFeatureSnapshot(adapter),
      adapterLimits: browserWebGpuLimitSnapshot(adapter),
      available: false,
      device: null,
      deviceFeatures: Object.freeze([]),
      deviceLimits: Object.freeze({}),
      reason: `requestDevice failed: ${err && err.message ? err.message : err}`,
    });
  }
  return Object.freeze({
    adapterFeatures: browserWebGpuFeatureSnapshot(adapter),
    adapterLimits: browserWebGpuLimitSnapshot(adapter),
    available: true,
    device,
    deviceFeatures: browserWebGpuFeatureSnapshot(device),
    deviceLimits: browserWebGpuLimitSnapshot(device),
    reason: "ok",
  });
}
