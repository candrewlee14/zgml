# WebGPU And Wasm Runtime Shape

This note pins down the WebGPU/Wasm lane for the executable-stencil runtime
plan. It is intentionally an implementation contract, not a broad compiler
proposal.

The target is the same lifecycle as native Zig, C, Node, and Bun:

```text
load/create Model
inspect Model
compile Program
bind Session
write StepParams
execute step
inspect evidence
free handles
```

The browser/runtime host should see handles and typed-array views. It should
not see graph internals, backend op structs, scheduler policy, or per-kernel
dispatch machinery.

## Non-Goals

- no tensor-op FFI across JS/Wasm for hot execution
- no WebGPU command-buffer reuse assumption
- no backend-specific public API in `zgml`
- no tokenizer/generation facade until it matches the loaded model family
- no graph walking, shape inference, or allocation in the decode hot path

## Handle Model

The Wasm-facing ABI should mirror the C handle model, with integer handles
instead of host pointers:

```c
uint32_t zgml_model_load_path(uint32_t desc_ptr, uint32_t out_model_ptr);
uint32_t zgml_model_load_safetensors_data(uint32_t desc_ptr, uint32_t out_model_ptr);
uint32_t zgml_model_inspect(uint32_t model, uint32_t out_inspection_ptr);
uint32_t zgml_program_compile(uint32_t model, uint32_t desc_ptr, uint32_t out_program_ptr);
uint32_t zgml_program_check_model_compatibility(uint32_t program, uint32_t model, uint32_t out_compatibility_ptr);
uint32_t zgml_session_bind(uint32_t program, uint32_t desc_ptr, uint32_t out_session_ptr);
uint32_t zgml_session_bind_model(uint32_t program, uint32_t model, uint32_t desc_ptr, uint32_t out_session_ptr);
uint32_t zgml_session_bind_model_buffers(uint32_t program, uint32_t model, uint32_t desc_ptr, uint32_t out_session_ptr);
uint32_t zgml_session_execute_tokens(uint32_t session, uint32_t desc_ptr, uint32_t result_ptr);
uint32_t zgml_session_step_token(uint32_t session, uint32_t desc_ptr, uint32_t result_ptr);
uint32_t zgml_session_prefill_tokens(uint32_t session, uint32_t desc_ptr, uint32_t result_ptr);
void zgml_session_free(uint32_t session);
void zgml_program_free(uint32_t program);
void zgml_model_free(uint32_t model);
```

The exact spelling can differ, but the semantics should not:

- `Model` owns model metadata and loaded checkpoint storage. Browser, Bun,
  Node, and Wasm hosts can load from a path when they have one, or pass complete
  safetensors bytes through `zgml_model_load_safetensors_data` when checkpoint
  acquisition naturally lives in host memory.
- `Model.inspect` exposes the model kind and fixed shape envelope so hosts can
  preflight compatible compile/bind choices before building a Program.
- `Program` owns architecture shape, shape envelope, backend plan, command
  shape, patch table, and evidence hash.
- `Program.checkModelCompatibility` answers whether a model handle can be bound
  behind the compiled shape without cloning or allocating Session state.
- `Session` owns persistent mutable runtime state: weights, KV cache, scratch
  arenas, output bindings, optimizer state when training exists, and backend
  resource bindings.
- `StepParams` owns small per-call values: token id, token window, position,
  attention length, RNG seed, batch slot, and output policy.
  `zgml_session_execute_tokens` is the current concrete token-window
  StepParams descriptor for the Wasm/C handle lane; the step/prefill functions
  remain compatibility conveniences.

For Wasm CPU, all descriptors and typed data live in linear memory. For WebGPU,
the Wasm module remains the host/runtime glue, while the WebGPU backend owns GPU
resources behind the same `Program` and `Session` handles.

The internal backend I/O descriptor now has two shapes: host memory and
external resources. Current CPU/Metal/stencil execution accepts host memory and
rejects resource bindings unless the backend advertises resource support. The
C/Node/Bun/Node-WASI/browser handle surface can now name host storage with
`zgml_buffer_create`/`zgml_buffer_wrap` and the second shape with
`zgml_buffer_wrap_resource`; `zgml_buffer_inspect` reports whether a handle is
host storage or an external-resource view, including placement, access flags,
handle, byte offset, view length, and backing resource length. Current
host-only execution rejects resource views with `unsupported`.
`zgml_session_inspect` reports the actual bound Session shape after bind:
backend, position/context, output and K/V storage kind, persistent/step I/O
counts, host-vs-resource binding counts, and an opaque binding-shape hash. The
hash includes resource placement/handle/offset/length/access for external
resources but excludes raw host pointer addresses. That ABI surface is backed by
the native `DeviceInference.Session.inspect()` contract, so adapters do not
invent their own view of backend resource bindings. For claimed Wasm/WebGPU host
Sessions, inspection mirrors the host-owned position so `zgml_session_inspect`
and `zgml_session_position` report the same token-window state after opt-in host
execution. `zgml_program_inspect` also
reports whether the compiled backend can execute real outputs and whether it
supports external-resource bindings, so a WebGPU host can distinguish
compile-only stencil evidence from GPUBuffer-backed executable Sessions before
bind. It also reports planned op count, buffer count, total buffer elements,
byte length, initial upload count, and qweight count so hosts can size and
audit resource tables from the compiled Program instead of from model-family
constants. A
native runtime patch envelope also records the max cache write position and max
attention sequence length that StepParams updates may touch; future WebGPU
patch/update buffers should be sized from that Program evidence rather than
from wrapper-side context math. Those bounds describe the compiled executable
graph's patch table; LLaMA-specific inspection separately reports the public
context envelope that current sessions enforce before execution. The
C/Node/Bun/Wasm inspection surfaces expose those bounds; `UINT64_MAX` means
unknown/unrepresentable and JS wrappers return `null` for that sentinel. The
same `zgml_program_inspect` ABI v5 surface now exposes native WebGPU
dispatch-plan evidence for executable Programs: backend dispatch count, full
coverage flag, covered op count, first unsupported op, and dispatch-family
counts. Node, Bun, Node/WASI, and browser Wasm adapters decode those fields;
the Node/WASI and browser Wasm smokes also assert that CPU and compile-only
WebGPU capability helpers see an empty dispatch plan. Wasm and compile-only
WebGPU paths keep them empty, while native wgpu tiny-linear execution asserts
full coverage before running. The fields are
translated from the generic backend `inspectExecutionPlan` hook carried by
native `DeviceInference.Program.inspect()`, so future WebGPU hosts do not need
backend-private adapter knowledge to audit the lowered Program. The
resource-capable backend regression proves
`DeviceInference` already forwards persistent, input, and output resource
descriptors through a backend `configure_bindings` hook during Session bind,
before persistent upload or step execution. That is the future WebGPU/wgpu bind
group/resource-table seam. `DeviceInference.Program` rejects mismatched
placements, unreadable persistent/input resources, and unwritable output
resources before the backend hook is called, and applies the same rule to
dynamic per-call input/output overrides before runtime-window patching and
explicit persistent re-upload before upload; the
regression then proves upload/execute keep valid descriptors intact without
converting them to host pointers. The lower LLaMA executable bind path
and public C/Node/Bun/Node-WASI/browser LLaMA bind
descriptors can also route host K/V cache buffers through persistent cache
bindings today, or opaque K/V cache resources through the same shape for future
resource-capable backends. Node and Bun expose that resource shape through
`program.createOutputBuffer({ resource })` and
`program.createKvCache({ resource })`, while C/Wasm callers can still build the
descriptor directly from handles. Those public requirements are queried with
`zgml_llama_program_get_kv_cache_requirements` and sized from the executable
KV-cache layout. Single- and multi-KV-head programs use packed
`[d_head, contextLength * nKvHeads]` per-layer K/V buffers. The future WebGPU
backend should be the first real backend to make the resource branch executable
for JS-owned `GPUBuffer` tables.
For the portable Wasm host path, `WasmWebGpuLlamaResourceProgram` mirrors the
Node/Bun slot shape: it derives logits and K/V byte sizes from the compiled
Program, can allocate Program-sized browser `GPUBuffer`/mock resources through
`createOutputBuffer()` and `createKvCache()`, and accepts either raw host
resources or pre-wrapped resource bindings before writing the same C ABI bind
descriptors.
`WasmWebGpuTinyLinearProgram` now follows the same slot rule for the browser
host execution slice: role-specific helpers create wrapped
weights/bias/input/output bindings, and `bind(...)` accepts omitted slots, raw
host resources, or pre-wrapped Program slots.

## Memory Model

Wasm CPU path:

```text
ProgramStencil handle -> compact command table in linear memory
Session buffers       -> linear-memory offsets and lengths
StepParams            -> small descriptor written by JS or Wasm
execute               -> one Wasm call per prefill chunk or decode token
```

WebGPU path:

```text
ProgramStencil -> cached shader modules, pipelines, bind-group layouts,
                  dispatch sequence, patch slots, evidence hash
Session        -> GPU buffers, bind groups where legal, staging buffers,
                  KV cache, persistent weight resources, patchable params
StepParams     -> small uniform/storage update, queue.writeBuffer, or mapped
                  patch arena update
execute        -> fast command encoding for the cached dispatch sequence
```

The reusable unit in WebGPU is not a command buffer. It is the compiled shape:
pipelines, layouts, compatible bind groups, buffer layout, dispatch sequence,
and patch table. Each step may still encode a fresh command buffer because that
is the WebGPU execution model.

## JS Shape

The ergonomic wrapper should look like the Node/Bun FFI wrappers:

```ts
const model = await zgml.loadModel("model.gguf");
const modelInfo = await model.inspect();
const program = await model.compile({
  backend: "webgpu",
  contextLength: 4096,
  batch: 1,
});
const capabilities = await program.capabilities();
if (!capabilities.canBindExternalResources) {
  throw new Error("compiled Program cannot bind WebGPU resources");
}

const logits = program.createOutputBuffer({
  resource: ({ byteLength }) => NativeBuffer.externalResource({
    placement: "webgpu",
    handle: gpuBufferTable.lookup("logits"),
    byteLength,
    access: "write",
  }),
});
const logitsInfo = logits.inspect();
const kvCache = program.createKvCache({
  resource: ({ byteLength, kind, layer }) => NativeBuffer.externalResource({
    placement: "webgpu",
    handle: gpuBufferTable.lookup(`${kind}:${layer}`),
    byteLength,
    access: "readwrite",
  }),
});
const session = await program.bind({ output: logits, kvCache });

await session.prefill(promptTokens);
await session.step(nextToken);
```

`canBindExternalResources` is not the same as `canExecute`: current public
LLaMA-family WebGPU Programs can expose an inspectable resource layout while
still returning `unsupported` for token execution. A host must key execution on
Program evidence, not on the backend enum alone.

Async belongs at the JS boundary because WebGPU queue work and browser resource
creation are async-shaped. The core runtime contract remains synchronous inside
the compiled Program/Session step: update small slots, execute the cached shape,
return or expose output.

## Backend Contract

A WebGPU backend is acceptable only if it consumes the same `ProgramStencil`
shape as CPU and Metal. It may lower that shape into WebGPU-specific resources,
but it must not invent a parallel execution story.

Compile may:

- infer shapes
- allocate arenas
- build command streams
- select kernels
- create shader modules and pipelines
- create bind-group layouts
- allocate persistent GPU buffers
- produce evidence and hashes

Bind may:

- allocate session-owned GPU buffers
- upload persistent weights or adapter weights
- allocate KV cache and scratch resources
- create bind groups where stable
- validate checkpoint compatibility

Step may:

- update a bounded patch table
- write a small uniform/storage StepParams buffer
- encode the cached dispatch sequence
- submit GPU work
- copy logits to the caller output when requested

Step must not:

- allocate per-token heap structures in the runtime
- walk graph nodes to discover runtime slots
- infer shapes
- choose backend placement
- rebuild pipelines
- rebuild the Program command stream
- silently fall back without evidence

## Evidence

The benchmark/runtime profile fields used by Metal should also exist for
WebGPU:

- backend ops
- placed ops
- fallback ops
- dispatches per call
- commands per call
- command-kind counts
- command-attempt counts
- runtime patch call count
- runtime patch changed count
- runtime patch-hole shape
- runtime patch stencil hash
- schedule-region failures
- sync/readback counts

The verifier should be able to reject a WebGPU parity claim if any selected row
uses hidden CPU fallback, changes command shape unexpectedly, or drops runtime
patch evidence.

## First Implementation Slice

The first useful WebGPU/Wasm slice should be deliberately narrow:

1. Compile a tiny linear Program to Wasm CPU with the same handle lifecycle as
   the C ABI.
2. Run a browser or Node Wasm smoke that creates model/program/session handles,
   steps into a caller-owned output buffer, and frees every handle.
3. Add a WebGPU compile-only inspection path for the tiny linear and fixed
   tiny LLaMA ProgramStencil shapes.
4. Add a WebGPU execute path only after compile evidence proves the command
   shape is the same executable program shape used by CPU/Metal.

That order keeps the portable ABI honest before chasing GPU speed.

Current groundwork:

- `zig build ffi-wasm` builds `zig-out/bin/zgml_c.wasm`, a no-entry wasm32-wasi
  module exporting the existing `zgml_*` handle functions and linear memory.
- `zig build ffi-wasm-smoke` instantiates that module through Node/WASI and runs
  the tiny linear plus fixed tiny LLaMA handle lifecycles with caller-owned
  linear-memory buffers. The tiny linear lane now covers both per-step
  descriptors, Program/Session runtime-profile snapshots and reset, and
  Session-bound linear-memory input/output buffers, so a host can mutate bound
  memory and call `zgml_session_step` with no step descriptor. The fixed tiny
  LLaMA lane also resets a Session in place and replays the first token to prove
  reusable KV-cache lifecycle control crosses the Wasm ABI, and binds a
  linear-memory logits buffer once so token step and prompt prefill can omit
  per-call output pointers. It also binds a compatible model handle as Session
  state, both with caller-owned logits and with a native `zgml_buffer` logits
  output. Bound Session steps execute through the backend's configured I/O table
  by default, while per-call logits overrides keep using the explicit dynamic
  descriptor path. It also exercises bulk no-output prompt advancement,
  direct generic-execute rejection before state/output mutation when a token
  window exceeds the compiled context envelope, including the cache handoff from
  executable prefill into the following decode step. It also compiles tiny
  linear and fixed tiny LLaMA with `ZGML_BACKEND_WEBGPU`, inspects executable
  shape evidence, and proves host-memory binding returns unsupported rather than
  falling back to CPU execution. C/Node/Bun plus the Node/WASI and browser Wasm
  smokes now also bind a tiny-linear opaque external-resource Session probe and
  prove `step` and no-output advance remain unsupported with zero Session
  runtime work. The fixed tiny LLaMA lane also inspects semantic shape evidence
  and binds an
  external-resource logits output plus K/V cache Session probe whose inspection
  reports WebGPU-shaped resources while token execution remains unsupported.
  Executable inspection reports stable command-category counts, so the smoke
  can prove the WebGPU compile-only path produced a real command shape instead
  of only a backend label. The Node/WASI smoke also derives a compact
  `programCapabilities` view from `zgml_program_inspect` for CPU execution,
  WebGPU resource probes, and compile-only LLaMA WebGPU, proving portable hosts
  can distinguish `mode: "executable" | "resource-probe" | "compile-only"` and
  `canExecute` from `canBindExternalResources` without branching on backend
  names.
  It also wraps caller-owned linear-memory buffers with `zgml_buffer_wrap`,
  binds them as Session resources, and proves direct linear-memory updates are
  visible to execution without making the buffer handle own that memory. It
  also wraps fake external-resource descriptors with
  `zgml_buffer_wrap_resource`, proves those handles are size-queryable, and
  proves host read/write plus CPU Session binding return unsupported rather than
  treating a resource handle as a host pointer. The fixed tiny LLaMA lane also
  queries executable-layout KV-cache requirements, binds host `zgml_buffer` K/V
  cache storage and executes through it, then binds a fake external-resource
  output through normal and compatible-model buffer binding and proves host-only
  execution rejects both with unsupported while clearing the output handle. It
  also writes explicit read/write external-resource access flags into the Wasm
  descriptors, so the browser-facing ABI already matches the validation a real
  WebGPU backend will need. It also allocates tiny linear persistent/input and
  output/logits host buffers through the generic Program buffer factory,
  proving Wasm callers can derive exact `zgml_buffer` sizes from compiled
  Program handles instead of duplicating shape math in JavaScript. Raw
  `zgml_buffer_create` remains in the smoke only where an intentionally
  oversized sentinel buffer proves execution does not write past the Program's
  declared output/logits span. This
  Node/WASI smoke is also part of `zig build check`, keeping the portable
  C-handle Wasm artifact on the normal local validation path. It now decodes
  runtime feature bits into named booleans and proves the portable Wasm artifact
  exposes the model selector, Wasm-export, device-buffer factory/import,
  model-path probe, supported-checkpoint catalog, safetensors-header probe,
  safetensors data-probe, safetensors data-load, compile-only WebGPU, and
  dispatch-plan inspection feature vocabulary while keeping
  `nativeWgpuExecution` false. The Node/WASI smoke now also writes a complete
  all-zero exact tiny LLaMA safetensors checkpoint into a preopened directory,
  probes it with `zgml_model_probe_path`, loads it with `zgml_model_load_path`,
  also probes and loads the same bytes with
  `zgml_model_probe_safetensors_data` and
  `zgml_model_load_safetensors_data`, compiles it, binds it, and steps a
  token, so the portable Wasm artifact has both filesystem-backed
  checkpoint-load evidence where WASI preopens are available and pathless
  byte-probe/load evidence for browser-shaped hosts. The catalog remains
  pathless and useful to Wasm hosts as an honest "what can this artifact load?"
  query; the safetensors-header probe goes one step further and lets browser
  hosts match a fetched metadata header against the same compiled-in registry
  without tensor bytes or a filesystem path. The full-data probe gives hosts
  the same native selector over an already-owned complete blob, and the
  full-data load then turns that compatible safetensors blob into a model
  handle without creating a virtual path. The same pathless data-probe lane now
  also selects the SmolLM-135M envelope from host-owned safetensors header
  bytes, while full-data load evidence remains limited to complete checkpoint
  bytes. The same smokes now call
  `zgml_abi_struct_size(kind)` to derive every linear-memory descriptor,
  inspection, requirements, and profile allocation from the exported C layout,
  so browser/Node Wasm hosts no longer duplicate target-specific `sizeof`
  knowledge in JavaScript. The Node/WASI and browser Wasm smokes also call the
  device-buffer factory, device-handle query, and import exports on CPU and
  compile-only WebGPU Programs and assert clean unsupported status with cleared
  handles, so those feature bits do not imply fake GPU execution.
- `zig build ffi-wasm-browser-smoke` runs the same browser page under
  Chrome/Chromium through a dependency-free CDP runner. It is a portable gate:
  it passes in mock mode when the browser has no WebGPU adapter, while still
  recording the reason and adapter-limit evidence in the page dataset/log.
  `zig build check` now includes this non-required browser smoke so portable
  browser/Wasm host-resource regressions are part of the default local/subagent
  gate.
- `zig build ffi-wasm-browser-gpu-smoke` runs that same page with Chrome WebGPU
  flags, requires real `GPUBuffer` mode, and asserts the adapter can bind the
  six-storage-buffer LLaMA block-pipeline proof. It also requires aggregate
  LLaMA profile evidence with executor backend dispatches separated from
  device-selection dispatches, scalar/window block dispatch-family counts, and
  zero fallback ops, so a real-GPU pass cannot hide a mock LLaMA executor or
  the wrong block shader family. This target is intentionally not part of
  `zig build check` because it depends on local browser/GPU availability and is
  slower than the Node/WASI gate; it has a 30-minute budget for the current
  exhaustive matrix. On the local M5 Pro/Metal adapter it passes in about 21
  minutes with 63 LLaMA profile labels, 211 backend dispatches, 207 executor
  dispatches, 4 device-selection dispatches, 98/98/0/0/0 storage-mode calls, and
  zero fallback ops before the structural-QKNorm and structural-bias matrix
  widening. The current no-adapter browser mock gate records 72 profile labels,
  122/0/122/0/0 storage-mode calls, 43 output reads, and 482 explicit mock
  fallback ops. For
  focused debugging, the underlying
  `examples/wasm_ffi/browser_smoke_runner.mjs` accepts
  `--llama-profile-label=<label>` to skip unrelated LLaMA profile producers and
  run only the selected packed-family proof plus its greedy variant when
  requested, with the runner rejecting unrelated focused labels; build targets
  intentionally omit that filter. LLaMA resource Programs and Sessions also
  accept `requireGpuDispatch`/`requireGpuExecution`/`noFallback` aliases.
  `WasmWebGpuLlamaResourceProgram.fromSafetensors(...)` now applies that strict
  policy by default for checkpoint-created LLaMA Programs; proof and no-adapter
  mock lanes must opt back into fallback with `allowFallback` or
  `allowMockFallback`. The Node/WASI and browser smokes bind that strict
  default safetensors Program and exercise raw token execution, proving a
  no-adapter path returns `unsupported` without executor calls or output/K/V
  mutation, while a capable `GPUBuffer` path must report backend dispatch with
  zero fallback. That mode preflights the built-in executor
  storage-buffer requirement and returns `unsupported` before mock fallback can
  write logits or K/V resources when a real `GPUBuffer` dispatch is unavailable.
  Successful executor results with
  backend dispatches must explicitly claim `usesGpuStorageBuffers`, and that
  claim must match the bound `WasmWebGpuDevice` capability and the executor's
  required storage-buffer count, preventing custom host executors from forging
  real-GPU profile evidence with counters alone or by omitting the storage-mode
  claim. The device-selection helpers honor
  the same contract: direct host-logits argmax/sampling, `gpu:false` direct
  sampling, execute-and-select calls, and device generation calls reject before
  token execution, selection dispatch, generated-token writes, profile counters,
  or resource mutation. Non-device full-logits argmax/sample,
  execute-and-select, and generate-and-select helpers reject under the same
  mode instead of quietly doing CPU selection after a resource execution.
  Node/WASI and browser smokes prove the no-mutation branch and keep the
  real-dispatch branch valid for capable adapters.
- `zig build check -Duse-metal=false -Duse-blas=false` is the explicit
  CPU/Wasm validation lane for machines whose local Metal stack is unavailable
  or wedged. It disables Metal imports, links, conformance cases, and benchmark
  variants instead of letting CPU evidence masquerade as Metal evidence.
- `examples/wasm_ffi/browser.html` runs the same lifecycle in a browser with a
  tiny browser-side WASI shim, proving the handle module can cross the browser
  JS boundary before browser WebGPU execution exists. The browser smoke now covers
  Program/Session runtime-profile snapshots and reset, decoded runtime feature
  bits, pathless exact tiny LLaMA safetensors byte probe/load, SmolLM-135M
  safetensors data-probe selection from host-owned header bytes, Session-bound
  linear-memory step buffers, non-owning wrapped
  linear-memory buffer bindings, external-resource handle rejection,
  a shared `WasmWebGpuDevice` / host-resource registry that maps
  GPUBuffer-like objects to stable opaque WebGPU resource handles for
  `zgml_buffer_wrap_resource`, a shared `WasmExternalResourceBridge` that writes
  the ABI v5 external-resource descriptor into Wasm linear memory before
  wrapping it as a native `zgml_buffer`, a shared `WasmSessionBindingBridge`
  that writes tiny-linear and LLaMA resource `Session` bind descriptors through
  the same ABI layout, including actual browser `GPUBuffer` objects when
  `navigator.gpu` is available, a narrow browser host Program/Session plus a
  `zgml_host` import bridge. The raw exported Wasm `zgml_session_step` and
  `zgml_session_step_no_output` functions now call that bridge for registered
  tiny-linear `zgml_session` handles. In browser `GPUBuffer` mode, the bridge
  enqueues real WebGPU compute; in portable mock mode, it executes the same
  tiny-linear resource table against deterministic mock buffers so Node/WASI and
  headless browser smokes can still prove the host import boundary. Both modes
  write `zgml_step_result`-shaped output lengths, keep a small host-side
  dispatch/readback profile, and prove no-output execution by changing the bound
  input, returning `output_len = 0`, recording no implicit readback, and then
  reading the changed host output resource back explicitly. The bridge can
  now return an internal "unclaimed" status distinct from public `unsupported`,
  so unregistered handles can continue through the normal Wasm path while
  claimed resource sessions can stop at the host boundary. Raw exported
  `zgml_session_execute_tokens` and its
  `step`/`prefill`/`advance`/generation conveniences also call the same bridge
  for WebGPU LLaMA-family sessions before native token execution. The
  tiny-linear browser host Program now uses the same resource-slot convention as
  the Node/Bun and LLaMA wrappers: `createWeightsBuffer()`,
  `createBiasBuffer()`, `createInputBuffer()`, and `createOutputBuffer()`
  return wrapped slots, while `bind(...)` still accepts raw host resources.
  The
  Node/WASI and browser smokes now use a shared
  `WasmWebGpuLlamaResourceProgram` helper to bind WebGPU-labeled LLaMA logits
  and K/V resources, deriving output and cache sizes from the compiled Program
  and accepting raw browser resources or already-wrapped resource bindings.
  The helper registers the claimed Wasm session handle in that host runtime and
  preserves the host resource-table evidence in one place. They
  prove token step and no-output advance reach the host token bridge with the
  decoded `zgml_token_execute_desc` token window, output policy, output pointer,
  and output length intact. The host bridge now preserves the same public
  validation boundary for claimed resource sessions: over-context token
  windows, out-of-vocab token IDs, and too-small logits outputs return
  `shape_mismatch`, while valid requests for unimplemented execution return
  `unsupported`. Execute-and-select plus
  generation setup also reach the same bridge, append normal logits-producing
  token-window records to the host Session history, and still receive `unsupported` with
  untouched position, caller logits, output-token buffers, and native runtime
  counters because browser/Wasm LLaMA kernels are not implemented yet. Raw
  exported `zgml_session_reset` and `zgml_session_free` also call the bridge for
  claimed browser GPU sessions before replay or destruction. The wrapped-exports
  facade remains a compatibility helper and now claims host-only
  `zgml_session_inspect` as well, writing the same ABI inspection struct with
  host-owned position, resource storage, resource counts, and binding-table
  hash. It also claims tiny-linear
  `zgml_session_step` / `zgml_session_step_no_output` plus
  `zgml_session_execute_tokens` / `zgml_session_reset` for registered LLaMA
  resource sessions, so proxy users hit the same synchronous host bridge as raw
  Wasm imports instead of receiving a Promise where the C ABI expects a status
  integer.
  Freeing a claimed host Session now also removes its claim from the host
  runtime, so a later Wasm handle that reuses the same numeric value can fall
  through the normal native path instead of being misrouted to a stale bridge.
  This sits next to lazy import-source
  descriptors, role-specific access flags, and a
  `document.documentElement.dataset.zgmlWasmGpuResources` marker for whether
  the smoke used `GPUBuffer` or mock resources. The browser smoke now also
  publishes why that mode was selected (`navigator.gpu` unavailable, no
  adapter, device-request failure, or `ok`) plus adapter/device evidence for
  the execution boundary: max storage buffers per shader stage, storage-buffer
  offset alignment, and whether the six-binding LLaMA block-pipeline proof can
  claim real backend dispatch on that adapter. That keeps "mock because this
  browser has no WebGPU device" distinct from "mock because the proof executor
  exceeds the adapter binding envelope",
  LLaMA session reset/replay, Session-bound logits output, compatible
  model-handle Session binding, Program-derived LLaMA KV-cache requirements,
  host-buffer KV cache execution, LLaMA output-resource binding rejection on
  host-only backends, bulk no-output prompt advancement, executable
  prefill-to-decode cache handoff, plus the same tiny-linear and LLaMA WebGPU
  compile-only inspection lanes and derived Program capability decisions. It
  also binds WebGPU-labeled logits and K/V cache resources with an independent
  compatible LLaMA model handle, inspects that model-bound resource Session,
  and proves unsupported execution leaves position and caller output unchanged.
- `ZGML_BACKEND_WEBGPU` exists in the C/Node/Bun/Wasm handle ABI as a
  WebGPU-shaped target. Builds without native wgpu, explicit LLaMA WebGPU
  opt-out builds, and portable Wasm/browser LLaMA Programs use the compile-only
  stencil backend and WebGPU command policy to produce command/runtime-patch
  evidence without allocating GPU resources or executing kernels/tokens.
  `zgml_program_inspect` reports that distinction directly with
  `backend = ZGML_BACKEND_WEBGPU` and `execution_supported = 0`. Tiny-linear
  default builds also report `external_resources_supported = 1` for an opaque
  resource-session probe: C/Node/Bun and Node/WASI/browser Wasm callers can
  bind external-resource weights/bias/input/output, inspect the WebGPU-shaped
  Session, and still receive `unsupported` on step and no-output advance with
  zero Session runtime work.
  Native C/Node/Bun libraries built with
  `-Duse-wgpu=true` instead make tiny-linear WebGPU host bindings execute
  through the real wgpu-native backend and report `execution_supported = 1`;
  the backend regression now also executes through registered same-device
  WGPUBuffer resources configured at Session bind time. Public opaque
  caller-provided raw resource handles still reject until GPU resource import has
  a safe shared-device/provenance contract. Node/WASI and browser Wasm now
  exercise the portable half of that contract with `WasmWebGpuDevice`: JS host
  resources get stable opaque handles, byte ranges, placement, access flags,
  same-device provenance, a host-side resource-table hash, and WebGPU usage
  validation before the shared `WasmExternalResourceBridge` writes the ABI
  descriptor and enters `zgml_buffer_wrap_resource`; the shared
  `WasmSessionBindingBridge` then writes the resource-backed tiny-linear and
  LLaMA bind descriptors before calling the Session bind exports. Node/WASI
  uses the same device wrapper with deterministic mock resources and now runs
  the claimed tiny-linear host Session through raw exported Wasm
  `zgml_session_step` / `zgml_session_step_no_output`; the browser smoke
  requests a WebGPU device and registers real `GPUBuffer`s when available, or
  runs the same host bridge in mock mode when no adapter is available. Both
  smokes reject GPUBuffer-like resources whose usage flags
  cannot satisfy the requested access, resources that claim the wrong device
  token, unregistered caller-owned resources with no same-device provenance, and
  mixed-device resource tables, while allowing explicitly imported caller-owned
  resources and storage-only resources when the caller opts out of host-transfer
  requirements. Imported caller-owned resources are registered for descriptor
  evidence without transferring destruction ownership to zgml.
  Descriptor creation and host resource-table evidence also reject byte ranges
  that exceed a known mock buffer or browser `GPUBuffer` backing size, so
  impossible views cannot become hashed resource-table evidence before bind.
  The browser smoke now also proves the first real shared-device browser
  execution slice outside the Wasm ABI when `GPUBuffer` is available: a
  tiny-linear host Program binds those registered resources through the Wasm
  resource Session bridge, registers the Wasm `zgml_session` handle in a host
  runtime, then the smoke's main raw exported
  `exportsRef.zgml_session_step(session, desc, result)` call uploads
  weights/input, dispatches a WebGPU compute pass, writes `output_len`, reads
  the output `GPUBuffer`, and matches the expected values. Host resource views
  now honor `byteOffset` and `byteLength` for writes, readback copies, and
  storage-buffer bind-group entries; the smokes bind Program-sized views inside
  larger aligned backing resources to prove the shared-device table is a real
  byte-range contract rather than only a whole-buffer handle table. The direct
  host-resource bridge and the shared tiny-linear/LLaMA Program resource
  wrappers also reject unaligned storage-buffer offsets and oversized output/KV
  views before creating a Wasm resource buffer, and pre-wrapped Program slots
  reject wrong-device WebGPU provenance before binding. Wrapper-created
  mock/`GPUBuffer` resources also carry owner-device provenance and reject
  cross-device descriptor wrapping, so mock execution and real `GPUBuffer`
  execution enforce the same malformed-view boundary. In mock mode, the same
  call path computes against deterministic host resources and records the same
  bridge/profile evidence without claiming GPU execution. The same
  raw-export surface also proves synchronous CPU/Wasm fallback, no-output result
  writes with a changed host output resource, raw `zgml_session_free` host
  cleanup, and stale-handle fallback/rejection.
  The resulting ABI Sessions are still compile-only/resource probes with
  unsupported execution and zero runtime work; this is browser resource-table
  execution evidence, not a general Wasm `zgml_session_step` WebGPU executor.
  Fixed tiny LLaMA reports the same
  external-resource capability for logits output plus K/V cache Session
  inspection; host-memory bind and token execution still return `unsupported`.
  Memory-layout and command-category counts remain smoke-tested by native,
  Node, Bun, Node/WASI, and browser lanes.
- The native `DeviceInference` execution API also enforces that distinction:
  WebGPU compile-only resource Sessions can be bound and inspected, but
  `executeStep` returns unsupported before runtime patching or dispatch
  counters change. The opt-in tiny-linear wgpu executor is the first path where
  WebGPU inspection evidence and real GPU execution are the same Program shape.
- The executable resource-capable regression now records command-stream-shaped
  evidence too: the test backend builds the real `ProgramStencil` command shape
  at compile, exposes it through immutable Program inspection, preserves it
  across Program and Session profile reset, receives the full persistent/input/
  output resource table through `configure_bindings` at Session bind time, and
  records command attempts/dispatches for resource-backed execution. CPU,
  Metal, and stencil runtime bindings now also own cached copies of the complete
  persistent/input/output table, so the backend seam is shaped like the bind
  group/resource-table cache the real WebGPU executor will need rather than
  only a per-step host descriptor list. A real WebGPU backend therefore has to
  satisfy both the bind-time resource-table contract and the command-plan
  contract, not merely pass opaque resource handles through an op-counting shim.
- That distinction now exists in backend capabilities too:
  `webgpu_compile_only` and `webgpu_resource_plan` describe today's
  non-executing ProgramStencil/resource-layout lane, while executable `webgpu`
  capability is used only by a backend that actually allocates GPU resources,
  encodes dispatches, and computes outputs. The opt-in native wgpu executor is
  the first deliberately narrow backend to claim that capability, currently for
  tiny-linear, standalone dense f32 matmul, standalone qmatmul with
  Session-bound runtime qweights, fused qmatmul-add, qmatmul-elementwise add/mul
  sidecars, standalone elementwise, standalone fused elementwise, standalone repeat,
  repeat-to-fused-elementwise activation chains, standalone LayerNorm,
  standalone RMSNorm, standalone row softmax,
  standalone reduce sum/max, and standalone RoPE, slice-assign, and bounded
  attention command shapes, plus bounded multi-dispatch Programs made from
  already-supported exact WebGPU shapes. It intentionally does not advertise
  full LLaMA-family WebGPU execution, JS-owned GPUBuffer interop, or broader
  attention/movement families until the executor can compile and dispatch those
  programs for real. It also has a backend-specific Program support check now:
  standalone dense f32 matmul passes through its own
  executable shader, standalone qmatmul passes through its own executable shader
  with packed int8 qweights plus f32 scales that can be replaced as Session
  state, fused qmatmul-add accepts a two-op `DeviceProgram` and skips the
  intermediate output buffer, qmatmul-elementwise add/mul sidecars do the same
  for a secondary tensor while keeping qweights Session-bound, dense matmul-elementwise add/mul sidecars plus
  dense matmul-fused-elementwise, dense matvec-fused-elementwise, and
  qmatvec-fused-elementwise activation sidecars compile as single dispatches
  with the projection value available as a virtual secondary operand, repeat
  feeding fused elementwise compiles as one
  activation-chain dispatch with the
  repeated value treated as a virtual secondary operand, and
  standalone
  add/mul/neg/abs/sgn/step/relu/sqrt/sqr/recip/exp/log/gelu elementwise programs pass through
  their own executable shader. Standalone LayerNorm and standalone RMSNorm use
  one 256-thread workgroup per row to reduce and scale
  the row in one dispatch; standalone softmax uses the same row-workgroup shape
  for stable max/sum reductions and normalization; standalone reduce sum/max
  uses one workgroup per output element; standalone repeat expands strided
  repeated source coordinates into the destination span; standalone RoPE rotates
  packed low/high pairs from a packed cos/sin table; standalone slice-assign copies strided
  rows/columns into the current destination offset; standalone attention uses
  one 256-thread workgroup per query for `seq_kv <= 256` and `d_head <= 256`,
  and switches to a streaming softmax shader for longer key/value windows up to
  `seq_kv <= 2048` at the same `d_head <= 256` bound.
- The native build has an optional wgpu-native dependency seam now:
  `-Duse-wgpu=true` enables target-specific wgpu-native include/link setup, and
  `zig build wgpu-link-smoke -Duse-wgpu=true` proves the headers and exported
  `wgpuCreateInstance` symbol link. That seam now feeds the first
  tiny-linear/matmul execution slice instead of reviving stale backend code.
  `zig build wgpu-check -Duse-wgpu=true` is the one-shot optional gate over
  native WebGPU linkage, generic/tiny-linear execution, and the public LLaMA
  WebGPU smokes.
- The first native executor prototype now exists too:
  `zig build wgpu-exec-smoke -Duse-wgpu=true` compiles `src/backend/wgpu.zig`,
  creates a wgpu-native device, lowers host-staged tiny-linear, standalone dense
  f32 matmul, fused dense matmul-elementwise add/mul sidecars,
  fused dense matmul-fused-elementwise activation chains,
  fused dense matvec-elementwise add/mul, fused dense matvec-slice
  cache store, fused dense matvec-RoPE cache store, fused dense
  matvec-fused-elementwise activation chains, standalone qmatmul with
  Session-bound runtime qweights, fused qmatmul-add, fused qmatmul-elementwise
  add/mul, fused qmatvec-elementwise add/mul, fused qmatvec-fused-elementwise activation chains, fused
  qmatvec-slice cache store, fused qmatvec-RoPE cache store,
  standalone elementwise, standalone fused elementwise, standalone repeat,
  repeat-to-fused-elementwise activation chains, standalone LayerNorm,
  standalone RMSNorm,
  standalone softmax, standalone reduce sum/max, standalone RoPE, and
  standalone slice-assign plus bounded
  standalone attention
  `DeviceProgram`s through the same
  `ProgramStencil` shape, binds Session-owned GPU buffers, dispatches WGSL
  compute kernels, reads the outputs back, and checks numeric results plus
  Session profile evidence. Patchable wgpu params are now Session-owned too:
  runtime-window patching refreshes the bound Session's uniform buffer from its
  own mutable op tape, so multiple live bindings can share one compiled program
  without leaking cache/slice positions through a global params buffer. The C
  ABI now uses this executor for
  `ZGML_BACKEND_WEBGPU` tiny-linear Programs when the native library is built
  with `-Duse-wgpu=true` and reports that
  build-level capability with `ZGML_FEATURE_NATIVE_WGPU_EXECUTION`; default
  builds, Wasm builds, and LLaMA-family WebGPU Programs still use the
  compile-only/resource-probe stencil lane. The native backend, C ABI, Node, and
  Bun smokes now also prove
  host-buffer weights are uploaded as persistent Session state at bind time:
  mutating the bound host weight buffer after bind does not affect later
  tiny-linear WebGPU steps, while mutating the bound input buffer does. The
  native executor binds host-backed program buffers at WebGPU-valid aligned
  offset zero and keeps tensor-view offsets in the cached shader params/patch
  table; same-device imported `WGPUBuffer` views with unaligned storage-buffer
  offsets or oversized byte ranges reject before they enter a Session binding
  table. The
  native executor also exposes program-owned WebGPU device buffers through
  `zgml_program_create_device_buffer(program, kind, ZGML_BACKEND_WEBGPU, ...)`;
  Node and Bun surface the same path through
  `program.createBuffer(kind, { placement: "webgpu" })` for weights, bias,
  input, and output; role-specific helpers remain aliases. Those buffers bind
  as external-resource Session state and execute through the registered
  resource table. The C ABI also exposes
  `zgml_program_get_device_handle(...)` and
  `zgml_program_import_device_buffer(...)`; Node and Bun surface those as
  `program.deviceHandle("webgpu")` and
  `program.importDeviceBuffer(kind, { deviceHandle, bufferHandle })`, with
  optional `byteLength` for exact-view validation. Node and Bun now also expose
  `program.device("webgpu")`, a small provenance object whose `handle` is the
  same program device token and whose `createBuffer(kind)` /
  `importBuffer(kind, source)` methods route through the same ABI. For
  role-specific JS code, `ProgramDevice` also exposes `createWeightsBuffer()`,
  `createBiasBuffer()`, `createInputBuffer()`, `createOutputBuffer()`, and
  `createKvCache()` so a caller can stay on the chosen same-device object after
  selecting WebGPU. For zgml-created device `NativeBuffer` objects, Node and
  Bun can import through
  either the Program or ProgramDevice object and derive the device token, raw
  buffer handle, byte offset, and view length from buffer inspection. Native
  Node/Bun host packages can also pass GPUBuffer-like objects through the
  exported `webgpuInterop` symbols for `deviceHandle`, `bufferHandle`,
  `byteLength`, `byteOffset`, and `placement`, or an
  `webgpuInterop.importSource` method that returns that descriptor lazily;
  `ProgramDevice.importBuffer(...)` fills in the same-device token by default.
  The LLaMA wrappers also keep `program.createKvCache({ placement: "webgpu" })`
  as a compatible shorthand; `device.createKvCache()` is the preferred
  same-device shape because output/KV allocation and imports all hang off the
  same provenance object.
  Default
  compile-only Node/Bun smokes assert that this ProgramDevice path, device
  buffer creation, and raw device import all throw `unsupported` instead of
  manufacturing a fake same-device lane.
  C callers or native JS integrations that create valid same-device
  `WGPUBuffer` objects can import them after zgml validates the program device
  token, byte range, storage-buffer binding alignment, and usage flags before
  the buffer can enter the Session binding table. A zero C `byte_len`, or
  omitted JS `byteLength`, uses the compiled Program's required byte length for
  that buffer kind. The Node and Bun smokes reject fake external-resource object
  imports at the wrapper, reject wrong-device, unaligned, and oversized raw
  imports, then import
  zgml-created device buffers plus symbol-backed host objects, free the
  original source buffer handles, and execute through the imported handles.
  Fake or wrong-device opaque handles still reject. Browser `GPUBuffer` objects
  do not expose raw `WGPUBuffer` handles, so the Wasm/browser lane now uses a
  `WasmWebGpuDevice` host-resource wrapper for opaque external-resource binding
  evidence. That wrapper can hold actual browser `GPUBuffer` objects and now
  validates storage/transfer usage flags, same-device provenance, and
  resource-table hashes before descriptor wrapping. Its host resource
  descriptors, resource tables, model manifests, and bound LLaMA Session
  resource evidence are frozen when created, so cache keys and table hashes
  cannot be silently invalidated by later JS mutation. Tiny-linear browser host
  Program/Session execution now resolves those descriptors into actual
  `GPUBuffer` bindings and dispatches WebGPU compute from the raw exported Wasm
  Session step/no-output functions through the module's `zgml_host` imports
  when a browser adapter exists. The same helper owns deterministic mock buffer
  storage for Node/WASI and headless browser runs, so those smokes can prove
  the Wasm host bridge, resource descriptors, profile counters, and cleanup
  without pretending mock resources are GPU execution. The claimed host Session
  exposes JS-side profile counters for dispatches, no-output calls, explicit
  output reads, and readback syncs, so the browser smoke can prove no-output
  execution dispatched without an implicit output readback.
  Browser `zgml_session_free` now routes claimed handles through the same host
  runtime before native Wasm handle destruction. The same host runtime now also
  has a LLaMA resource-session token bridge for `zgml_session_execute_tokens`;
  it records host-side token counters for decoded calls, logits-output versus
  no-output policies, caller-output versus bound-output requests, validation
  rejections, result writes, and valid-but-unsupported execution attempts, with
  both Node-WASI and browser smokes asserting the ten-call profile and reset
  behavior. The wrapped-exports compatibility facade now routes claimed
  tiny-linear `zgml_session_step` / `zgml_session_step_no_output` calls through
  the synchronous host execution bridge and claimed LLaMA
  `zgml_session_execute_tokens` calls through the token bridge. Tiny-linear
  wrapped calls return normal numeric status codes and write
  `zgml_step_result.output_len`; LLaMA resource calls claim WebGPU-labeled
  logits/KV resource sessions and return `unsupported` after decoding the
  token-window descriptor, without mutating native Session state until real
  browser LLaMA kernels can consume that resource table. General LLaMA-family browser WebGPU
  execution still remains future work rather than a default public support
  claim.
  LLaMA-family WebGPU Programs deliberately
  remain on the
  compile-only/resource-probe lane: native wgpu now has a private fixed tiny
  LLaMA decode/prefill probe whose logits, prefill cache side effects, cache
  refresh, no-output prompt advancement, decode-after-prefill behavior, and
  registered same-device resource-bound logits/KV execution match CPU under
  `-Duse-wgpu=true`. The proof now keeps decode and prefill on one retained
  WebGPU device and imports the same K/V buffers into the prefill Program, so
  prefill can populate GPU-resident cache resources that decode later consumes
  without CPU cache staging. It also covers a small GQA shape with multiple
  query heads over packed multi-KV-head cache resources and a two-layer shape
  with per-layer K/V resource-table handoff, plus an 8-token prefill over a
  16-token context before decode. The same private helper now proves no-output
  prompt advancement leaves the logits resource untouched, records zero runtime
  syncs/fallback ops, and still hands GPU-resident K/V resources to decode. A
  private quantized tiny LLaMA decode proof also covers WebGPU qweight command
  evidence, external-resource logits/KV bindings, and Session-owned runtime
  qweight rebinding. The private LLaMA WebGPU tests also inspect the compiled
  wgpu dispatch plan from the program-owned `ProgramStencil`: decode and
  prefill must cover every op, report projection/attention dispatch families,
  report quantized projection families for the quantized proof, and record hot
  runtime dispatch counts that match the compiled dispatch-plan count. A
  combined private proof now exercises those checks with GQA, two layers, a
  16-token context, an 8-token same-device resource prefill, and decode from
  the GPU-resident K/V cache. A wider-head private proof now also exercises a
  LLaMA-style `d_head = 128` resource prefill/cache-handoff/decode path over
  the same 16-token envelope, so the no-fallback attention proof is not limited
  to toy head widths. That combined resource-bound
  prefill/cache-refresh/decode path also runs under a failing allocator after
  Session bind/configure, proving no Zig heap allocations or resizes in the hot
  path while still matching CPU logits. The private resource gate now also
  rejects bad K/V access modes before execution: decode caches must be readable,
  and prefill cache outputs must be writable. Native builds with
  `-Duse-wgpu=true` now enable the public LLaMA WebGPU executor by default,
  while `-Dexperimental-llama-wgpu-execution=false` preserves the older
  resource-probe-only LLaMA lane. The public smoke can exercise the normal LLaMA
  facade against the native wgpu executor through
  `zig build llama-wgpu-experimental-smoke -Duse-wgpu=true`; that target runs the Zig facade
  one-head proof, the GQA/MQA/MHA plus tied-LM-head public resource-handoff
  matrix, a private native multilayer MQA dispatch/resource proof, a realistic
  `d_head = 128` public resource-handoff proof, a combined
  realistic `d_head = 128` plus GQA plus two-layer public resource-handoff
  proof, a reduced compiled-context proof where an 8-token WebGPU Program uses
  8-token-sized same-device K/V resources inside a 16-token model envelope, a public
  resource-bound quantized projection proof with qweight and quantized-dispatch
  evidence, and a public resource-bound over-context guard proof, plus C, Node,
  and Bun public FFI
  smokes. It proves public prefill, no-output prompt advancement,
  decode-after-cache-handoff parity, same-device resource-bound logits/K/V
  execution, and over-context rejection before extra runtime patching, dispatch,
  sync, fallback, or logits-resource mutation through the same handle APIs.
  `zig build wgpu-check -Duse-wgpu=true` runs that LLaMA smoke together with the
  native WebGPU link and generic executor smokes.
  C, Node, and
  Bun allocate Program-owned WebGPU output and per-layer K/V buffers, bind them
  as Session resources, execute decode/prefill/decode-after-prefill, and read
  logits back through the public buffer API. They now also import those
  same-device output and K/V handles, bind the imported handles as Session
  resources, execute decode, and read logits through the imported output handle.
  They now also prove a no-output
  prompt advance on a resource-bound Session leaves the logits resource
  untouched, records real backend work with zero sync/fallback, and then decodes
  from the GPU-resident K/V resources to match CPU. The public Zig facade and
  the same public FFI paths now also prove resource-bound argmax/sample
  execution plus greedy and top-k=1 sampled generation parity; token selection
  reads the bound logits resource into session-owned staging while no-output
  advancement still avoids logits readback. The C ABI unit path plus Node/Bun
  smokes also fill the compiled context through resource-bound no-output
  execution, reject the next step as
  `shape_mismatch`, and prove unchanged Session position, logits resource, and
  runtime counters. C/Node/Bun executable decode smokes now also compare hot
  Session `backend_dispatch_count` against the compiled Program dispatch count
  for both normal and same-device resource-bound decode, tightening the
  no-fake-fallback evidence at the public handle boundary. The runtime-info handshake
  now separates `nativeWgpuExecution` from
  `experimentalLlamaWgpuExecution`, and the public C/Node/Bun smokes assert that
  the LLaMA `.webgpu` `execution_supported` bit matches that compiled-in native
  LLaMA execution lane rather than merely the presence of the native wgpu
  executor. Builds without native wgpu, explicit opt-out builds, Node/WASI, and
  browser Wasm still keep LLaMA-family WebGPU as compile-only/resource-probe
  evidence.
  C, Node, Bun, Node/WASI, and
  browser Wasm resource-probe token step and no-output advance also now return
  `unsupported` with untouched caller logits, unchanged Session position, and
  zero Session runtime calls, backend ops, fallback ops, syncs, or runtime patch
  attempts. Node-WASI and browser Wasm additionally prove those resource-probe
  token calls are claimed by the host WebGPU token bridge and decoded as
  logits-producing step or no-output advance descriptors before returning
  unsupported by default, with malformed token-window/output shapes and
  out-of-vocab token IDs rejected as shape-mismatch first, so future GPU kernels
  plug into the same exported token-window
  path rather than a side adapter. Node-WASI and browser Wasm also prove bound
  argmax/sample, execute-and-select, and greedy/top-k=1 generation reject with unchanged output
  token buffers and unchanged position, so public adapters cannot silently fall
  back while LLaMA-family
  WebGPU is gated. Terminal
  resource outputs that alias reused compiled workspace buffers now execute into
  the internal buffer and GPU-copy the terminal slice into the external resource
  instead of rebinding the workspace storage itself. The executor also proves
  `download_outputs = false` is a real no-readback path: it dispatches the
  cached command shape, leaves host output memory untouched, and does not
  increment the runtime sync counter. C exposes that path as
  `zgml_session_step_no_output`, and Node/Bun expose it as
  `TinyLinearSession.advance()`. The native backend regression also proves
  standalone dense f32 matmul can run through configured Session bindings with
  host A/B/output buffers, no-readback execution, and registered same-device
  WGPUBuffer resources for A/B/output; dense matmul-elementwise add/mul and
  dense matmul-fused-elementwise sidecars have the same
  host/resource/no-readback proof while covering the matmul and sidecar as one
  backend dispatch. Standalone qmatmul has the same
  configured host/resource/no-readback proof for input/output bindings, plus an
  independent-Session runtime qweight upload proof. Fused qmatmul-add has the
  same configured host/resource/no-readback proof for input/bias/output
  bindings and records two backend ops per dispatch. Qmatmul-elementwise add/mul
  has the same configured host/resource/no-readback proof for
  input/secondary/output bindings, keeps qweights Session-bound, and records
  two backend ops per dispatch. Standalone elementwise has the same
  configured host/resource/no-readback proof for src0/src1/output bindings.
  Standalone RMSNorm, standalone softmax, and standalone reduce sum/max have
  the same proof for src/output bindings; standalone RoPE has the same proof
  for src/cos-sin/output bindings; standalone slice-assign has the same proof
  for strided src/output bindings; bounded standalone attention has the same
  proof for q/k/v/mask/output bindings, including the longer streaming path
  above the 256-key fast-path ceiling. The native backend also proves bounded
  multi-dispatch execution over one Session buffer table: supported op tapes can
  lower to multiple exact WebGPU dispatches, each with its own pipeline,
  params buffer, and bind group, encoded into one command buffer with actual
  backend dispatch counts recorded in the runtime profile. A
  `FailingAllocator` regression now compiles, binds, configures, and uploads
  before forcing future Zig allocations/resizes to fail; patching and configured
  execution with output download, no-output dispatch, and a later output
  download leave allocator counters unchanged. The native wgpu resource-table
  configurator now enforces role access locally too: persistent and step-input
  external resources must be readable, and step-output external resources must
  be writable before any bind groups can be built.
- The native C, Node, Bun, Node/WASI, and browser Wasm lanes now also prove
  tiny-linear WebGPU resource probes can bind external-resource
  weights/bias/input/output, inspect the bound resource table, reject step and
  no-output execution, and leave Session runtime counters at zero. Those lanes
  also prove model-bound WebGPU resource probes: a Program compiled from one
  compatible LLaMA model can bind an independent compatible model plus
  external-resource logits and K/V cache buffers, inspect that Session, and
  still receive unsupported step and no-output advance without mutating logits,
  advancing position, or recording runtime work. They now also reject invalid
  public resource descriptors before a Session is returned: read-only logits,
  write-only K/V caches, and read-only K/V caches all fail the resource-probe
  bind. The shared Wasm host wrapper enforces the same role-access rules for
  tiny-linear and LLaMA pre-wrapped WebGPU resource bindings before writing
  another bind descriptor, so browser and Node/WASI callers cannot smuggle an
  invalid resource table through the ergonomic API. Imported bindings also
  snapshot their validated descriptors before Session binding, so caller-side
  descriptor mutation cannot rewrite logits/K/V byte ranges, access flags, or
  table hashes after bind preparation. This caught
  and fixed a C descriptor lifetime bug where K/V resource slices pointed at
  optional payload copies instead of the parsed descriptor storage. C, Node,
  and Bun also prove the public LLaMA-family WebGPU same-device lane remains
  closed: C device-handle/device-buffer/import calls and Node/Bun
  ProgramDevice/device-buffer/import helpers all reject as unsupported while
  the inspectable external-resource Session probe remains available.
- Node/WASI and browser Wasm also expose the next honest plug-in point for real
  browser LLaMA kernels: `WasmWebGpuLlamaResourceProgram` can now be constructed
  with an explicit token executor. The default remains resource-probe
  `unsupported`, but the opt-in executor receives the already-decoded and
  already-validated token-window descriptor plus first-class `stepParams`
  evidence, bound logits/KV resources, device, model-binding identity, and
  resource-table evidence. The same `WasmWebGpuLlamaResourceProgram` can now
  also create and bind a read-only, tensor-shaped model-resource table:
  `safetensorsModelResourceSpecs` derives stable weight roles, `shape`/`dtype`,
  byte lengths, and safetensors data offsets from checkpoint bytes or header
  metadata, while duplicate roles are rejected before host buffers are created.
  `createModelResourcesFromSafetensors` then allocates the model resources and
  uploads each exact checkpoint tensor slice into its bound buffer. Node/WASI
  and browser smokes bind every tensor in the tiny LLaMA checkpoint resource
  table and read those initialized bytes back, so future browser kernels have a
  validated, role-named place for full model weights instead of adding an
  adapter-only side channel. A LLaMA resource Program can also declare
  `requiredModelResources` and set `allowExtraModelResources: false`, making bind
  reject missing, wrong-dtype, wrong-shape, wrong-byte-length, or unexpected
  weight resources before a native Session handle is created. For byte-backed
  safetensors sources, the strict required manifest can be derived at bind time.
  Program default arbitrary resource maps can also derive a strict manifest at
  Program construction, so later binds reject extra roles without requiring a
  separate hand-written requirement manifest; one-off uncataloged maps still
  reject when unexpected weights are forbidden. The model-resource
  Session now also carries a separate tensor-manifest hash over role, dtype,
  shape, element count, byte length, and safetensors data offsets, so executor
  evidence can distinguish "same resource table" from "same tensor contract."
  `stepParams` carries the start/end position, token count, context length,
  output policy, output kind, requested output length, and logits length that
  future kernels patch into their command stream. The exported
  `WasmWebGpuLlamaTokenWindowExecutor` helper supplies reusable `writeLogits`,
  `llamaKvCacheWindow`, and `writeKvCacheWindow` utilities to an executor
  callback while reusing its augmented callback context and K/V window slot
  scratch across synchronous calls, so browser kernels share one logits/KV
  window contract instead of duplicating descriptor math in adapters. The named
  `WasmWebGpuLlamaTokenWindowPatternExecutor` is the current proof executor:
  caller-output logits still write through Wasm linear memory, while bound
  logits and K/V cache windows dispatch deterministic WebGPU compute against
  browser `GPUBuffer` resources when an adapter is available, with the same
  deterministic mock fallback used by Node/WASI and headless browser smokes.
  Its GPU path binds full storage buffers and writes the cache-window element
  offset through a tiny params buffer, avoiding unaligned storage-buffer
  bindings for nonzero positions. Smokes prove caller-output logits, no-output
  advancement, bound-output logits, and position-sliced K/V cache writes can
  return `ok`, update the host-owned Session position observed through
  `zgml_session_position`, mirror that position through
  `zgml_session_inspect`, and verify bound logits/KV side effects through
  explicit async readback in browser `GPUBuffer` mode as well as mock mode.
  `WasmWebGpuLlamaEmbeddingProjectionExecutor` is the first weight-consuming
  browser/Wasm probe: it reads the bound `model.embed_tokens.weight` and
  `lm_head.weight` model resources, accepts a token window, and computes
  embedding-to-logits projection for the final token into the bound logits
  resource. It dispatches real WGSL when browser `GPUBuffer`s are available and
  otherwise reports one explicit mock fallback operation.
  `WasmWebGpuLlamaRmsNormProjectionExecutor` extends the same proof to consume
  `model.norm.weight`, apply final RMSNorm to the selected final-token
  embedding, and project the normalized hidden state through `lm_head.weight`
  into the bound logits resource, again using browser WGSL in `GPUBuffer` mode
  and one explicit fallback op in mock mode. Node/WASI and browser smokes prove
  both one-token and multi-token final-token logits for these projection probes.
  `WasmWebGpuLlamaKvProjectionExecutor`
  is the first attention-side browser/Wasm proof: it consumes
  `model.layers.0.input_layernorm.weight`,
  `model.layers.0.self_attn.k_proj.weight`, and
  `model.layers.0.self_attn.v_proj.weight`, accepts a token window, applies
  input RMSNorm to each token embedding, and writes the projected K/V rows into
  the bound cache window while leaving logits untouched for no-output
  execution. Browser `GPUBuffer` mode fuses the K/V window write into one WGSL
  dispatch; mock mode reports one explicit fallback op. Node/WASI and browser
  smokes prove both one-token and two-token K/V cache windows. Its default
  model-resource roles are derived from the executor `layer` option, so
  `layer: n` naturally targets `model.layers.n.*` weights while explicit role
  overrides still win; the tiny checkpoint smokes still execute layer 0 only.
  `WasmWebGpuLlamaAttentionProjectionExecutor` is the first
  cache-consuming attention proof: it accepts a token window, writes each
  token's K/V cache row, reads prior rows through the growing attention window,
  applies Q/K RoPE, attention softmax, `o_proj`, residual add, final RMSNorm,
  and `lm_head`, and writes logits for the final token when requested. Browser
  `GPUBuffer` mode executes that as one WGSL dispatch per token; mock mode
  reports one explicit fallback op per token. Its Q/K/V/O default roles follow
  the same layer-derived contract. Node/WASI and browser smokes cover both
  single-token cache handoff and two-token attention prefill.
  `WasmWebGpuLlamaBlockProjectionExecutor`
  extends that cache-consuming proof through the post-attention RMSNorm and
  SwiGLU FFN (`gate_proj`, `up_proj`, `down_proj`), adds the FFN residual, then
  applies final RMSNorm and `lm_head` logits while preserving the same K/V
  cache handoff. It now accepts a token window for that one-layer proof by
  executing one browser WGSL dispatch or one explicit mock fallback per token,
  writing each token's K/V cache row, optionally writing each token's
  `blockHidden` vector into a bound `llama.activation` resource, and emitting
  bound logits for the final token only. `WasmWebGpuLlamaResourceProgram`
  exposes `createActivationBuffer({ hiddenSize })` for that f32
  `[contextLength, hiddenSize]` activation surface; factory-created safetensors
  Programs retain the inferred hidden size, so callers can create that activation
  surface without restating `hiddenSize`. Callers can bind it as
  `activation`/`activationOutput` for writes and `activationInput` for reads.
  Bind preparation rejects malformed activation byte lengths and aliased
  activation input/output views before execution, keeping the path
  ping-pong-ready for later layer chaining. With `activationInput` bound, the
  block executor consumes the previous hidden row instead of
  `embed_tokens[token]`; without it, the one-layer proof keeps the embedding
  input path. `WasmWebGpuLlamaBlockPipelineExecutor` composes those block
  executors into the first browser/Wasm activation pipeline: each token window
  runs stage-by-stage, intermediate stages write Session-owned activation
  workspaces, later stages consume those rows through `activationInput`, and
  only the terminal stage emits logits. The native-bound smoke still
  deliberately repeats the one available compiled tiny-checkpoint layer to
  prove ABI-level ping-pong orchestration without pretending broader native
  checkpoint support exists.
  `WasmWebGpuLlamaResourceProgram.bindHostResources(...)` now creates
  runtime-owned host-only LLaMA resource Sessions, with generated JS-side
  handles intercepted by `WasmWebGpuSessionRuntime.wrapExports(...)` instead of
  being passed to native pointer-taking exports. Node/WASI and browser smokes
  use that path to bind two- and three-layer f32 safetensors resource Sessions
  with real layer-indexed tensors plus separate per-layer K/V resources, then run
  the `tiny-llama-block-pipeline` proof executor against deterministic reference
  math. A second non-tiny-width shape (`vocab=10`, `hidden=6`, `ffn=12`,
  `contextLength=3`, two full-width attention layers) now runs the same
  no-output-prefill plus bound-logits decode path, proving the browser/Wasm
  safetensors factory and block pipeline are not hardwired to the original
  `[8, 4, 8]` tiny checkpoint dimensions. The safetensors factory now also
  derives grouped head geometry, FFN intermediate size, RoPE base, RMSNorm
  epsilon, and tied-LM-head policy from `__metadata__` fields, config JSON
  embedded in `__metadata__`, HF-style config objects/JSON strings, or explicit
  options, and rejects conflicting sources before binding. It also rejects
  malformed final-norm, per-layer norm,
  attention O-projection, and MLP `gate_proj`/`up_proj`/`down_proj` tensor
  shapes during Program creation, before any Session bind.
  HF-style config objects and direct metadata aliases also reject unsupported
  semantics at Program creation: non-LLaMA-family identity hints, malformed
	  dtype/cache/dropout/token-id hints, non-SiLU activations,
	  enabled attention/MLP bias without complete matching projection-bias tensors,
	  disabled attention/MLP bias that conflicts with present bias tensors,
	  enabled Q/K projection normalization without complete matching norm tensors,
	  enabled sliding-window attention without a concrete window,
	  conflicting option/metadata/config sliding-window sources,
	  dynamic/YARN/LongRoPE-style RoPE scaling, partial rotary embeddings,
	  malformed SmolLM3-style no-RoPE layer bitmaps, tensor-parallel pretraining
	  shards, and `contextLength` values beyond the config maximum position
	  embedding.
	  Direct metadata and config values that describe the same supported semantic
	  hint must also agree for model identity, architectures, concrete torch dtype,
		  cache usage, pretraining tensor-parallelism hints, tokenizer special-token
		  ids that fit the resolved vocabulary, dropout probabilities, Q/K projection-normalization flags,
		  no-RoPE layer interval, and no-RoPE layer bitmap;
	  `auto` dtype remains a non-claim and `silu`/`swish` remain equivalent.
	  Supported no-op config fields such as
		  `hidden_act = silu/swish`, `attention_bias = false`, `mlp_bias = false`,
		  positive `pretraining_tp` over already materialized full tensors,
		  disabled Q/K projection normalization, enabled Qwen3-style Q/K projection
		  normalization with complete per-head `q_norm`/`k_norm` vectors, and
		  structural Q/K-norm inference from those tensors when config metadata is
		  silent,
		  disabled sliding windows, bounded sliding windows, sliding windows that cover the compiled context,
  absent/default RoPE scaling, linear RoPE scaling, Llama 3 RoPE wavelength
	  scaling, interval-derived SmolLM3 no-RoPE layer bitmaps, and full rotary
	  embeddings, including GGUF-style `rope.dimension_count` and
	  `attention.key_length`/`attention.value_length` values that equal
	  the inferred head size, plus GGUF `tokenizer.ggml.tokens`,
	  `tokenizer.ggml.scores`, and `tokenizer.ggml.token_type` lists whose length
	  matches the inferred vocabulary, are accepted in the positive MHA/GQA/MQA/Mistral/Qwen2/Qwen3/SmolLM3
	  family proofs.
  F16 and BF16 safetensors weight tensors are accepted at this boundary by
  materializing them once into executor-compatible f32 model resources during
  Session resource creation; the hot browser/Wasm block executor still consumes
  the same f32 resource shape instead of adding dtype branches.
  The proof executor
  uses those Program-carried values for packed grouped attention: the Node/WASI
  and real browser `GPUBuffer` smokes run two-layer MHA (`kv=8`, four K/V
  heads), GQA (`kv=4`, two K/V heads), and MQA (`kv=2`, one K/V head) shapes
  with `vocab=6`, `hidden=8`, `headSize=2`, `ffn=8`, `contextLength=3`,
  non-default RoPE bases, and non-default RMSNorm epsilons through no-output
  prefill, decode-after-prefill, bound logits, and packed per-layer K/V cache
  parity without manually wiring the executor. The matrix now covers
  metadata-derived MHA, GGUF-style and architecture-prefixed metadata-only GQA,
  config-object-derived GQA, config-JSON-derived MQA,
  factor-8 Llama 3 RoPE GQA, metadata-embedded-config GQA, a 17-token
  no-output prefill plus three-token logits decode over a 32-token
  config-derived GQA context, a
  metadata-embedded Mistral-family GQA config whose `sliding_window` is shorter
  than the compiled context and changes the decode logits reference window, a biased Qwen2-style
	  GQA config with complete attention/MLP projection-bias tensors plus optional
	  terminal `lm_head.bias`, and a no-bias Qwen2-style
	  GQA config whose `head_dim` matches the inferred attention head size, all
	  with derived `intermediate_size`, plus a SmolLM3 GQA identity whose
	  `no_rope_layer_interval` disables RoPE on the second proof layer, while accepting declared
	  LLaMA-family identity fields such as
	  `model_type = llama/mistral/qwen2/smollm3` and
	  `architectures = [LlamaForCausalLM/MistralForCausalLM/Qwen2ForCausalLM/SmolLM3ForCausalLM]`, plus positive
  `pretraining_tp` as a training-time partitioning hint over already materialized full tensors,
  and well-formed inference-only HF hints such as `use_cache`, `torch_dtype`,
  dropout probabilities, tokenizer special-token ids, and no-op/default,
  factor-2 linear, and factor-8 Llama 3 `rope_scaling` objects, proving the
  browser/Wasm block pipeline is
  no longer limited to one
  full-hidden attention head, full-hidden K/V cache stride, hard-coded
  `rope_theta=10000`, hard-coded `rms_norm_eps=1e-5`, or blind tied-LM-head
  synthesis, including the single-K/V-head case, while still avoiding a false
  claim that raw safetensors tensor shapes alone identify arbitrary LLaMA head
  geometry. The safetensors factory defaults to
  that proof executor when the caller does not explicitly provide a token
  executor or preset, and the executor infers contiguous layer plans from
  `kvCacheRequirements.layers`; the three-layer smoke proves no-output prefill,
  decode-after-prefill, and the two-workspace ping-pong handoff shape.
  `WasmWebGpuLlamaResourceProgram.fromSafetensors(...)` now lets that proof
  Program be created from checkpoint bytes plus either an explicit
  `contextLength` or matching metadata/HF-config maximum-position fields; an
  explicit smaller execution context still wins over a wider checkpoint
  envelope, while conflicting metadata/config maxima reject before Program
  creation. It derives
  `vocabSize`, hidden size, FFN intermediate size, layer count, per-layer K/V
  cache byte lengths, the strict model-resource requirement manifest, default
  strict no-extra validation, the default model-resource source, and an optional
  default model handle for later Session binding. For factory-created checkpoint
  Programs, missing
  `lm_head.weight` can be treated as a tied LM head by adding a read-only
  `llama.weight.lm_head.weight` resource populated from
  `model.embed_tokens.weight`, keeping executor roles unchanged while matching
  tied-head checkpoints; if config or metadata explicitly says embeddings are
  not tied, missing `lm_head.weight` rejects before binding instead. Custom
  root and terminal tensor names supplied as `embeddingTensor`, `normTensor`,
  `lmHeadTensor`, and `lmHeadBiasTensor` are also carried from cold shape
  validation into the default model-resource manifest, bind-time safetensors
  materialization, and the default executor role map. Node/WASI and browser
  smokes execute renamed embedding, final norm, LM head, and optional terminal
  bias checkpoints through the default factory path, so a default Session cannot
  silently drop or ignore a validated root/logits tensor equivalent. The
  browser/Wasm resource Program now exposes a frozen pre-bind `inspect()`
  snapshot with model shape, output/K/V/activation byte requirements, strict
  model-resource roles, default model/resource presence, executor kind,
  GPU-storage-buffer requirements, current adapter feasibility, and the same
  binding-requirement hash/count contract as native Program inspection, so JS
  hosts can audit the executable resource contract before creating a Session.
  Bound
  resource Sessions expose a matching frozen JS `inspect()` snapshot containing
  that Program contract plus current position, max token window, model binding
  kind, logits/K/V table roles and hashes, model-resource table/manifest hashes,
  and executor kind, making post-bind Session evidence line up with the
  pre-bind Program audit.
	  Node/WASI and browser smokes now also exercise that default factory through
	  native-backed `bind(...)` and raw exported `zgml_session_execute_tokens`,
	  proving default-created K/V and model resources plus an ABI-sized
	  default-created output slot do not require the host-only facade to execute
			  the two-layer block pipeline. That public path now covers the tiny
					  two-layer checkpoint, materialized F16/BF16 variants, tied-LM-head
						  and named two-shard checkpoint derivation, plus compatible GQA-family
						  proofs for config-derived GQA, long-context GQA, Llama 3 RoPE
						  GQA, metadata-embedded-config GQA, bounded sliding-window
						  Mistral, biased/structural-biased/no-bias Qwen2 GQA, Qwen3 Q/K projection norm, and SmolLM3
						  NoPE no-output-prefill plus decode-after-prefill. Native-backed
						  safetensors `bind(...)` now preflights the checkpoint-derived
						  executable envelope against the compiled C ABI requirements before
						  creating output, K/V, or model resources: native Program kind must
						  be valid, any bound native model handle must be compatible with
						  that Program, scalar and token-id widths must match the ABI,
						  native output length must cover the proof vocab, native output
						  byte length must match f32 logits, native batch must stay at
						  one, native context length must match the Program context, the
						  max-token-window envelope must fit that context, and K/V
						  layers/bytes must match. The host resource Session now carries
						  that max-token-window envelope and rejects wider token-window
						  execution before executor dispatch. When the native output
						  envelope is wider than the proof vocab, `bind()` default-creates the
						  ABI-sized output slot without an explicit `outputLen`. Node-WASI and
						  browser smokes prove representative MHA/MQA checkpoints reject as
						  explicit K/V requirement mismatches, malformed native ABI
						  envelope fields reject as preflight mismatches, oversized proof
						  vocabularies reject as output-envelope mismatches, and
						  incompatible native model handles reject before bind, without
						  claiming a Session.
  The binding helper also accepts safetensors checkpoint bytes directly as
  `modelResources`, including explicit byte-backed shard lists and HF-style
  `weight_map` indexes, derives the strict model-resource requirement manifest
  from those bytes when none was declared, rejects duplicate tensor names,
  index/shard mismatches, or mismatched index `metadata.total_size` across
  shards, preserves each tensor's shard index/name in the model-resource
  manifest, expands the checkpoint into read-only
  model-resource buffers during Session bind, and can default-create the output
  and K/V resources. Terminal
  `llama.activation` is optional for this preset because intermediate stage
  handoff uses Session-owned activation workspaces released with the bound
  Session; callers bind activation only when they need final hidden rows. The
  no-executor constructor resource
  probe still returns unsupported; the safetensors factory only removes manual
  per-layer `tokenExecutor` wiring and preset-string knowledge for the current
  proof executor. Program and Session `inspect()` snapshots expose this as
  `executionCoverage = "bounded-proof"` with
  `fullDefaultExecutionSupported = false`; no-executor Programs report
  `executionCoverage = "resource-probe"`, and custom token executors report
  `custom-executor`. Its attention and FFN default roles also derive from
  `layer`.
  Native C ABI multi-layer checkpoint binding now has a tiny two-layer LLaMA
  safetensors catalog/load/execute/KV-bind proof across C, Node, Bun, Node-WASI,
				  and browser Wasm, plus native-backed tied-LM-head, named two-shard, and
				  compatible GQA-family proofs and host-only three-layer, wider-shape, realistic
	  one-layer `d_head=128`, metadata/config-derived MHA/GQA/MQA,
		  GGUF-style and architecture-prefixed metadata-only GQA,
		  config-derived intermediate-size, non-default-RoPE/RMSNorm-epsilon, and
			  config-derived tied-LM-head, bounded sliding-window Mistral-family,
			  biased/structural-biased/no-bias Qwen2-style, Qwen3-QKNorm, and SmolLM3-NoPE
	  Node-WASI/browser no-output-prefill and
  decode-after-prefill pipeline proofs, while default full LLaMA-family browser
  execution remains future work. Node/WASI and
  browser smokes compare single-token cache handoff plus two-token prefill
  logits/K/V cache-window values, activation-output rows, activation-input
  logits/K/V rows, the two-stage repeated-layer ABI pipeline output, the
  host-only two-layer `[0, 1]` pipeline output, and the host-only three-layer
  `[0, 1, 2]` no-output-prefill plus decode output, and the host-only wider
  `[vocab=10, hidden=6, ffn=12]` two-layer output, the host-only
  realistic-head `[vocab=5, hidden=128, headSize=128, ffn=32]`
  one-layer no-output-prefill plus bound-logits decode output, and the
  packed-head
		  `[vocab=6, hidden=8, headSize=2, ffn=8]` MHA/GQA/MQA/Mistral/Qwen2/SmolLM3-NoPE
  non-default-RoPE/RMSNorm-epsilon outputs against the f32 checkpoint tensor
	  bytes and ABI profile evidence, plus a tied-LM-head
	  two-layer checkpoint whose terminal logits reuse the embedding bytes through
	  the derived `lm_head` resource through both host-only and native-backed raw
	  ABI execution, plus a named two-shard root+layer0/layer1
	  checkpoint with an index `weight_map` plus `metadata.total_size` whose Program
	  shape, model-resource upload, shard provenance, bound-logits decode, and
	  per-layer K/V cache writes match the monolithic checkpoint reference through
	  both host-only and native-backed raw ABI execution. The host-only two-layer proof no
  longer asks callers to separately pass the weight manifest, `vocabSize`, K/V
  requirements, output, K/V buffers, terminal activation, per-layer executor
  callbacks, a preset string, or repeated bind-time model handle: the Program
  derives the shape and strict model manifest from checkpoint bytes, defaults
  the resource slots, infers layers from the derived K/V requirements, carries
  the model identity, and creates the current proof executor. Default
  block-pipeline call evidence also records `deviceMode`, required/max
  storage-buffer counts, and whether the call used real `GPUBuffer` storage
  resources or explicit mock fallback work, so mock execution, adapter-limit
  failure, and real browser GPU dispatch stay separate evidence states. The resulting
  resource Session also exposes the intended async JS control shape:
  `advanceTokens(tokens)` performs no-output advancement, `step(token)` defaults
  to bound logits, and `argmax()`, `sample({ topK, seed, temperature })`,
  `executeTokensArgmax(tokens)`, `executeTokensSample(tokens, ...)`,
  `stepArgmax(token)`, `stepSample(token, ...)`,
  `generateTokensArgmax(prompt, maxTokens)`,
  `generateTokensArgmaxInto(prompt, outputTokens)`,
  `generateTokensSample(prompt, maxTokens, ...)`, and
  `generateTokensSampleInto(prompt, outputTokens, ...)` mirror the native
  selection vocabulary. The explicit full-logits helpers still read the bound
  logits resource, so output-read/sync counters expose host selection cost
  rather than hiding it as GPU work; GPU-required Sessions reject those helpers
  so `noFallback` cannot silently become CPU selection. Browser/Wasm Sessions now also expose
  `argmaxDevice()`, `executeTokensArgmaxDevice(tokens)`,
  `stepArgmaxDevice(token)`, `generateTokensArgmaxDevice(prompt, maxTokens)`,
  `generateTokensArgmaxDeviceInto(prompt, outputTokens)`,
  `sampleDevice({ topK, seed, temperature })`,
  `executeTokensSampleDevice(tokens, ...)`, `stepSampleDevice(token, ...)`,
  `generateTokensSampleDevice(prompt, maxTokens, ...)`, and
  `generateTokensSampleDeviceInto(prompt, outputTokens, ...)`: real
  `GPUBuffer` mode runs cached one-dispatch WGSL reductions over bound logits
  and reads back only `{token, logit}` for argmax or the top-k
  `{token, logit}` pairs for sampling, with top-k=1 device sampling sharing the
  argmax selector instead of running the heavier top-k shader. The argmax
  selector validates GPU results before returning them and reuses its params
  upload view, params buffer, result buffer, and bind group across repeated
  selections. The top-k selector has the same reuse contract for same-or-smaller
  `topK` requests, rebuilding only when the result capacity grows, then samples
  directly from the readback view instead of allocating candidate and weight
  arrays. The shared full-logits/mock top-k sampler uses the same two-pass
  scalar thresholding shape over compact primitive token/logit arrays instead
  of allocating per-selection candidate objects or a weight array, and
  resource Sessions reuse those scratch arrays across same-or-smaller `topK`
  full-logits selections while growing capacity only when `topK` grows. Hot
  generation uses private `executeTokens*Into` helpers and one Session-owned
  selection/result scratch object, so the continuation loop no longer allocates
  public execute-and-select wrapper results per generated token; public helpers
  still return fresh result objects at API boundaries. The same mutable
  call-options object reaches sync and decoded execution, with logits output
  forced by a scalar override rather than a per-token spread/clone.
  Full-logits readback now follows the same ownership rule: `WasmWebGpuDevice`
  can read f32 data into caller-owned storage, LLaMA resource Sessions reuse
  one internal logits scratch buffer for selection-only output reads, public
  `includeLogits` still returns a fresh logits array, and `outputInto` fills a
  caller-provided `Float32Array`. Execute-and-device-select helpers ignore
  stale caller `logits` after execution and select from the freshly written
  bound output. Browser `GPUBuffer` readback supports caller-owned byte reads
  with `readBytesInto`, uses one device-owned staging buffer for sequential
  f32/byte reads, grows it only for larger reads while forgetting the replaced
  staging resource from the device owner list, destroys it with the device, and
  uses a temporary staging buffer for overlapping reads while releasing that
  temporary owner reference as soon as the overlapping read settles. Device
  teardown also unregisters remaining device-owned resources, so stale handles
  cannot resolve after whole-device cleanup. Wrapper-created resources retain
  owner-device provenance for cleanup, so executor fallback-owned resources are
  also unregistered when destroyed through resource-only cleanup paths. LLaMA
  resource Sessions mirror tiny-linear deterministic lifecycle cleanup: direct
  JS `free()` / `dispose()` / `Symbol.dispose` releases native-backed Sessions
  through the Wasm export, tears down host-owned resources, unregisters runtime
  lookup entries, and clears host Session claims; Node/WASI and browser smokes
  prove stale direct-freed native and direct-disposed host-only handles are
  unclaimed at the host callback boundary. Device argmax and
  top-k selectors read their result records into selector-owned byte scratch;
  selector-owned GPU result buffers are forgotten from the device owner list and
  unregistered from the host-resource handle table on selector dispose, and
  top-k does the same for the old GPU result buffer when result capacity grows.
  Destroyed selector result objects cannot be re-described or reused for later
  WebGPU reads/writes after their handles are unregistered.
  Node/WASI and
  browser fake-GPU smokes prove the staging lifecycle, selector result-scratch
  reuse, and map/unmap balance without depending on local browser adapter
  health. Mock mode records explicit fallback selection.
  Node/WASI and browser smokes prove these helper paths over
  the same two-layer block-pipeline checkpoint, seeded top-k sampling, generated
  tokens, K/V side effects, position evidence, bound-output logits,
  output-read/sync counters, small device-selection read counters, and ABI
  profile counters as the raw Wasm token descriptor path. The generation helpers
  now also preflight the full
  generation window before dispatch: prompt tokens must fit the Session's max
  token-window envelope, prompt tokens plus generated continuation must fit the
  compiled context, and over-context or over-window greedy/sampled generation
  returns `shapeMismatch` without changing output tokens, position, K/V/logits
  resources, or profile counters. After the prompt step, those hot generation
  loops reuse one Session-owned single-token continuation window and one
  Session-owned call-options scratch object, borrow prevalidated token windows,
  including caller-owned prompt token buffers with an explicit active length,
  into the token bridge, reuse one mutable execution record, one mutable
  StepParams object, and one mutable token-executor context, and skip diagnostic history
  arrays for that internal lane instead of allocating fresh token, execution,
  StepParams, context, and options objects per generated token or generation
  call. Public scalar decode/select wrappers
  (`advance`, `step`, `stepArgmax`, `stepArgmaxDevice`, `stepSample`, and
  `stepSampleDevice`) also borrow a Session-owned one-token scratch window into
  the token bridge; native Node/Bun scalar select helpers use the same shape
  plus reusable scalar option scratch instead of allocating wrapper `[token]`
  arrays or per-call option spreads, and native `advance(token)` now uses the C
  ABI scalar no-output call directly. The browser/Wasm path
  reuses the Session-owned execution record, StepParams, executor context, and
  normalized executor-outcome record, and skips diagnostic histories without
  mutating caller options, so repeated host-driven decode avoids wrapper
  one-token array, bridge-record, and executor-result wrapper allocation before
  execution.
  `advanceTokens` uses a Session-owned no-output execution override instead of
  spreading caller options, preserving caller-owned option objects while forcing
  logits output/readback off and reusing the same hot bridge scratch for prompt
  advancement.
  Public JS token execution and generation entry points also accept borrowed
  capacity-sized token buffers plus explicit active-length options across the
  native Node/Bun and portable Node-WASI/browser wrappers. Those wrappers thread
  that active count into the C/Wasm token descriptors and shared execution
  scratch, and the hot generation path strips prompt-only length metadata from
  continuations. Token execution validation indexes exactly the active
  `tokensLen` window instead of iterating the whole token container, so borrowed
  scratch buffers can be larger than the active token window without checking
  stale capacity tail values or allocating an iterator. Standalone embedding,
  RMSNorm, activation-logits, K/V, attention, and block projection executors now
  use that same active token window for last-token lookup, fallback loops, GPU
  submission loops, terminal-output decisions, and diagnostic token records;
  Node, Bun, Node/WASI, and browser smokes poison the inactive tail and iterator
  for public token execution, generation prompt buffers, and projection seed
  calls to prove the inactive capacity is ignored.
  The block pipeline executor also reuses stage/final context scratch while
  composing layer executors, including its no-output execution, StepParams, and
  result records for non-terminal stages, final-logits projection, and the
  top-level Session executor result before normalization; standalone K/V
  projection now writes into the Session executor-result scratch too; and the
  K/V, attention, and block projection executors also write K/V cache
  slot-window views into executor-owned scratch. Standalone attention and block
  projection mock fallback also reuse executor-owned math/result scratch across
  cache-handoff decode and prefill, pass exact active lengths into K/V,
  activation, and logits writes, and compute final logits only for the terminal
  token that requests them. Ordinary token execution now records diagnostic
  histories through a bounded Session retention window, with
  `diagnosticHistoryLimit: 0` available when callers only want the latest
  decoded execution and StepParams evidence. Retained history entries are
  immutable active-window snapshots, so caller token-buffer mutation and reused
  StepParams scratch cannot corrupt earlier evidence. Hot generation preserves
  caller options, final-only logits, and per-token sample seed progression. The smokes
  also
  assert Program-carried hidden/vocab/layer-count and `ffnSize` evidence and
  reject conflicting tensor/option/metadata/config shape sources, conflicting
  metadata-embedded and explicit config values, conflicting direct
  metadata/config semantic hints, conflicting sliding-window sources,
  conflicting RoPE scaling kind/frequency-factor sources, plus malformed MLP
  down-projection shapes before bind. The
  unsupported config/direct-metadata matrix rejects non-LLaMA-family `model_type`/`architectures`,
		  malformed dtype/cache/dropout/token-id hints, GELU,
		  incomplete or disabled-conflicting attention/MLP bias,
		  conflicting Q/K projection normalization or enabled Q/K projection
		  normalization without complete matching vectors,
		  enabled sliding-window attention without a concrete window, dynamic/YARN-style RoPE
	  scaling, malformed Llama 3 frequency factors, partial rotary factor,
	  mismatched GGUF `rope.dimension_count` hints, mismatched GGUF attention
	  key/value length hints, out-of-vocabulary special token ids, mismatched GGUF tokenizer
	  token/score/type counts, malformed or
	  too-short SmolLM3-style no-RoPE layer bitmaps, plus too-short or conflicting
	  config/direct-metadata max-position, sliding-window, no-RoPE interval, and
	  no-RoPE bitmap envelopes before bind; Qwen2-style configs with complete
	  attention/MLP bias and SmolLM3 configs with interval-derived NoPE layers now
	  run through Node-WASI/browser reference-output proofs,
  and the safetensors factory also validates optional `lm_head.bias` as a
  vocab-length vector while rejecting unsupported integer/quantized tensor
	  dtypes plus norm, malformed/missing Q/K projection-norm, rotary cache, and per-layer sidecar tensors before deriving
  a model manifest. RoPE
  `rotary_emb.inv_freq` tensors are accepted only as structural auxiliaries:
  shape is checked against the inferred head size, payload values are checked
  against the resolved RoPE base plus linear/Llama 3 scaling schedule when
  checkpoint bytes are available, and they are excluded from the executable
  model-resource manifest. That prevents the
  browser/Wasm proof executor from silently running the wrong model family.
  They now
	  emit tiny two-layer checkpoints as F16 and BF16, verify the bound model
	  resources carry f32 executor dtype with source dtype/data-length evidence, and
	  compare block-pipeline logits/K/V output against dtype-decoded reference math
	  through both host-only resource execution and the public native-backed
	  `bind(...)` plus raw `zgml_session_execute_tokens` path.
  The
  projection proof executors now prepare and cache their model-resource tensors
  during Session bind instead of searching and validating the model-resource
  manifest inside each token step; asking a layer-1 block executor to bind the
  one-layer tiny checkpoint rejects at bind time with the missing layer-1 role.
  The same bind preparation now caches the target layer's K/V cache descriptors
  and stride, so token steps patch only the runtime byte window; an oversized
  K-cache descriptor whose element count does not divide the context length is
  rejected at bind instead of being discovered during execution. Browser
  `GPUBuffer` projection submits also reuse static bind groups for the prepared
  Session resources, K/V cache descriptors where applicable, and executor-owned
  params/tokens upload buffers, so the hot path updates buffer contents instead
  of allocating fresh JS upload views between submissions. Projection and
  device-selection GPU submits also reuse executor-owned bind-group dependency
  scratch instead of allocating a small key array per submit. The block executor
  now also has a 64-token bounded scalar token-window kernel for no-output
  prefill stages: real browser `GPUBuffer` execution batches multi-token
  no-output block stages into one dispatch per layer while preserving
  sequential prompt K/V visibility inside the shader, and mock fallback keeps
  one explicit proof op per token. The scalar and scalar-window shaders now
  store scores by local attention-window offset, so long-context sliding-window
  Mistral-style proofs can keep the scalar decode and batched prefill paths
  when the active attention span fits the fixed scratch envelope.
  Logits-producing token windows also split terminal block stages into a
  batched no-output prefix plus one final logits dispatch, so prompt-style
  decode windows no longer submit every prefix token separately.
	  The required-GPU browser gate now requires 72 LLaMA labels, adding
	  strict-default safetensors execution evidence,
	  standalone Mistral sliding-window, Llama 3 RoPE, SmolLM3-NoPE block,
	  and structural Qwen3 Q/K-norm checkpoint inference
	  plus GGUF-style and architecture-prefixed metadata-only GQA
	  proofs to the prior standalone Qwen3-QKNorm block, standalone biased-Qwen2
	  block, standalone biased-Qwen2 attention, SmolLM3-NoPE attention, Llama 3 RoPE attention, and
	  zero-fallback real-`GPUBuffer`
	  evidence set; the last local M5 Pro/Metal required-GPU pass before this
	  matrix widening recorded
	  211 backend dispatches, 207 executor dispatches, 4 device-selection
	  dispatches, 98 real `GPUBuffer` storage-mode calls, 63 labels, and zero
	  fallback ops in about 21 minutes, while the current no-adapter mock evidence reports
	  71 labels, 122 mock storage-mode calls, 43 output reads, and 482
	  explicit fallback ops. Node/WASI and
  browser smokes exercise this with a fake WebGPU device and the real
  required-GPU gate so the cache contract, including K/V, activation-output,
  activation-input descriptor swaps, the long sliding-window scalar decode and
  batched-prefill paths, scalar/window block dispatch-family evidence, and the
  GPU-vs-mock command-count split, is covered. Block executor results/records,
  block-pipeline records, JS runtime profiles, and browser runner summaries
  expose scalar, window, and generic dispatch-family counts, so tests and the
  real-GPU gate can assert the selected shader path directly instead of
  inferring it only from aggregate command totals.
  Device-selection results now pass through the same no-fake-evidence gate as
  token-window executor results: Session normalization rejects injected
  argmax/top-k selectors that return out-of-vocab tokens, non-finite logits,
  invalid work counters, zero-work successes, or GPU dispatch evidence without
  a `WasmWebGpuDevice`-owned path.
  Logits-producing prefix windows also reuse the block executor's
  borrowed no-output prefix context and terminal dispatch options scratch, so
  that split path avoids token slices and per-dispatch wrapper allocation.
  Attention token windows reuse the same executor-owned step-options scratch
  across per-token GPU submissions, so non-batched attention proof windows avoid
  per-token dispatch wrapper allocation too. The block pipeline executor reuses
  stage and final-logits context scratch, including mutable no-output execution,
  StepParams, result records, and K/V slot-window records, so layer chaining
  avoids per-stage wrapper allocation. Standalone K/V projection mock fallback
  also writes K/V values through executor-owned Float32 scratch arrays, growing
  only when a larger token window is first seen. Embedding, RMSNorm, and
  activation-logits projection mock fallback also write bound logits through
  executor-owned Float32 scratch instead of allocating a fresh vocab buffer per
  fallback call. Attention and block projection mock fallback now reuse
  executor-owned hidden, norm, Q/K/V, score, attention, FFN, activation, result,
  and final-logits scratch while keeping capacity-sized buffers behind exact
  active write lengths.
  The
  browser runner now attaches before the smoke page loads,
  bounds CDP setup/evaluation calls, and emits stage/profile console
  breadcrumbs, so a wedged browser WebGPU adapter request reports the last
  reached stage rather than hanging the validation gate. The fake-device gate
  also asserts standalone K/V projection and block-window params uploads,
  token uploads, token GPU buffers, and bind groups are reused across
  same-capacity token windows, and that token upload/GPU buffers grow into
  reusable capacity buckets instead of reallocating on every slightly larger
  window. It also proves a long-context sliding-window block prefill keeps the
  one-dispatch batched path locally, so that command-shape claim is not only
  covered by the real-adapter browser gate. The token-window
  pattern proof executor also reuses
  slot-indexed bind groups, params GPU buffers, one JS params upload view, and
  one K/V write evidence record across submissions; the same fake WebGPU smokes
  assert GPU objects are created once per write slot while uploads still occur
  on each submit. It also encodes
  validated writes directly into the compute pass instead of allocating a
  per-submit JS command list, and its GPU K/V-cache path walks the bound
  Session K/V entries directly instead of staging token-window/write arrays.
  Its mock fallback path writes deterministic logits and K/V values through
  executor-owned Float32 scratch instead of allocating per-call pattern arrays
  or a K/V slot-window list.
  Bound-logits GPU writes use the same direct single-write submit instead of a
  one-element staging array.
  The same host executions
  expose ABI-visible work through
  `zgml_session_runtime_profile`: browser `GPUBuffer` pattern writes report
  backend dispatches, mock-resource pattern writes report fallback work, and
  command counts reflect the two K/V writes plus optional bound-logits write per
  token-window execution rather than a generic callback count. The smokes also
  prove malformed executor evidence is rejected: invalid public status codes,
  invalid work counters, mismatched executor-returned table hashes, or
  missing resource-table identity on successful table-bound results, or
  mismatched executor-returned model handles return `invalid_argument`, do not
  advance the host-owned Session position, preserve the rejected StepParams for
  diagnosis, and report ABI invalid runtime-patch evidence without fabricating
  backend, fallback, or command work. The internal `unclaimed` routing sentinel
  is not accepted as an executor status. Successful executor results must also
  keep work counters coherent: command count equals backend dispatch count plus
  explicit fallback op count, and op counters cannot be smaller than their
  corresponding command/dispatch counts; Node-WASI and browser smokes now
  explicitly reject an executor that claims a command while under-reporting
  command-op evidence. Non-`ok` executor results must also report zero output
  length and zero backend/fallback/command work; a failed result that still
  claims work is rejected as malformed evidence rather than hidden in the
  failure path. Table-bound successful results must
  report nonzero command work; an `ok` result with a validated table but no
  backend or fallback command evidence is rejected as a fake success.
  Successful logits-output executor results must report exactly the Session
  vocab length; partial logits evidence rejects before position advancement. A
  model-bound success smoke proves the
  compatible model handle and model-resource roles are visible to the executor
  and accepted only when the result reports the same model identity and
  model-resource table hash. The
  smokes reset that host evidence through
  `zgml_session_reset_runtime_profile` and still reject over-context requests
  before executor dispatch. This keeps the exported
  token-window ABI stable while making the future GPUBuffer executor a real
  host bridge rather than another adapter layer. Browser host execution now
  prepares a Session-owned read-only model pack for the LLaMA attention/block
  proof executors and patches model-tensor element offsets through params.
  That pack carries a derived layout/source-descriptor hash over field order,
  roles, offsets, dtype/shape/byte lengths, original model-resource descriptors,
  and Session model-table/manifest hashes; once an executor registers a pack on
  the bound Session, the host bridge requires a matching pack hash before
  accepting the result, and rejected calls roll back any pack registrations they
  created. `zgml_session_free` releases those Session-owned
  derived pack/dummy resources plus block-pipeline activation workspaces through
  the host runtime, so repeated browser binds do not leak packed weight or
  handoff buffers until executor shutdown. Attention
  binds four storage buffers, while block and block-pipeline execution bind six,
	  so real browser `GPUBuffer` adapters with the WebGPU minimum storage-buffer
	  envelope can claim backend dispatch instead of falling back
  only because the proof used too many read-only model bindings. The browser
  page now also exports aggregate LLaMA resource-session profile evidence:
  profile count, total backend dispatch count, executor backend dispatch count,
  device-selection backend dispatch count, fallback-op count, command count,
  scalar/window/generic dispatch-family counts, block-pipeline storage-mode
  call counts split into real `GPUBuffer`, mock, adapter-limited, and unknown
  states, max required storage-buffer count, output reads, small device-selection
  result reads, syncs, and profile labels.
  The required-GPU browser runner refuses real-`GPUBuffer` runs that report no
  LLaMA executor backend dispatches, any LLaMA fallback ops, mismatched
  total-vs-executor-plus-selection dispatch evidence, no scalar/window block
  dispatch-family evidence, no direct GPU-storage block-pipeline call evidence,
  any mock, adapter-limited, or unknown storage-mode block-pipeline calls,
  no six-storage-buffer requirement marker, missing representative
  storage-mode labels across tiny, sharded, dtype-materialized, wider,
  realistic-head, packed-family, and tied-head default pipeline paths, no
  device-selection readback evidence, or any
  missing current LLaMA label across host-token and model-token hooks,
  embedding/RMSNorm/KV/attention/block projection probes,
  block-pipeline and ergonomic two-layer execution, full-logits generation,
  device-argmax generation, device-sampled generation, grouped-family greedy
  generation, sharded, F16/BF16
	  materialized-checkpoint, tied-LM-head, multi-layer, wider, realistic-head,
	  individual metadata-MHA, GGUF-style metadata-GQA, architecture-prefixed metadata-GQA, config-GQA, long-context-GQA, config-JSON-MQA,
	  Llama3-RoPE-GQA, metadata-config-GQA, Mistral-GQA, biased-Qwen2-GQA,
	  structural-biased-Qwen2-GQA, no-bias Qwen2-GQA, Qwen3-QKNorm-GQA,
	  structural-Qwen3-QKNorm-GQA,
	  standalone K/V-QKNorm,
	  standalone attention-QKNorm, standalone sliding-window attention,
	  standalone Llama3-RoPE attention, standalone SmolLM3-NoPE attention,
	  standalone biased-Qwen2 attention, standalone biased-Qwen2 block,
	  standalone Qwen3-QKNorm block, standalone sliding-window block,
	  standalone Llama3-RoPE block, standalone SmolLM3-NoPE block,
	  SmolLM3-NoPE-GQA, and realistic-`d_head=128` GQA
	  profile labels, including the long sliding-window Mistral-GQA profile. The same
  gate rejects duplicate LLaMA profile labels or a
  profile count that does not match the label list,
  making the no-fake-fallback contract visible outside the page assertions.
  The realistic-head proof now avoids the old pathological fused shader shape:
  wide block-pipeline stages use bounded scalar/window GPU block kernels for a
  66-token long sliding-window prefill plus decode, then a separate
  activation-logits dispatch, so the `[hidden=128, headSize=128]` browser path
  emits checked logits in real `GPUBuffer` mode instead of proving only K/V
  side effects. The packed MHA/GQA/MQA/Mistral/Qwen2/Qwen3-QKNorm/SmolLM3-NoPE
  default safetensors Programs now also run ergonomic greedy generation in
  Node/WASI and browser smokes, comparing generated tokens, terminal logits,
  K/V writes, ABI profiles, and executor evidence to reference math.
  family proofs now assert the same scalar/window/generic dispatch-family split
  through their Session profiles and executor call records, not only aggregate
  command counts.
  Wrapper-created LLaMA resources still use CPU shadows and bidirectional copy
  usage so mock fallback, sentinel uploads, and explicit readback are legal
  WebGPU operations. It is not full transformer execution or default public
  LLaMA-family browser WebGPU support yet; checkpoint-family browser execution
	  beyond the current individually required grouped-attention, long-context GQA,
	  Qwen3-QKNorm GQA, SmolLM3-NoPE GQA, realistic `d_head=128` GQA, and realistic-head
	  proofs remains the next support claim.
- Session inspection now exposes a binding-shape hash. Node/WASI and browser
  smokes prove it is nonzero for host and WebGPU-resource sessions and stable
  across equivalent model-bound resource sessions, giving the future WebGPU
  executor a cheap way to evidence the exact bound resource table shape without
  exposing backend internals. The JS host bridge now also hashes the
  WebGPU-like resource descriptor table before binding, including placement,
  access, resource handle, byte range, and same-device provenance, so browser
  hosts can reject malformed shared-device tables before entering the Wasm ABI.
  That host table now keeps two pieces of evidence: a physical descriptor hash
  over handles/ranges/access and a semantic table hash that also includes the
  Program role names such as `tiny-linear.weights`, `llama.output`,
  `llama.k.0`, and `llama.v.0`. Node/WASI and browser smokes prove swapped
  roles keep the physical descriptor hash but change the semantic table hash,
  distinct UTF-16 role code units do not collapse through low-byte truncation,
  duplicate non-empty roles reject before binding, and the LLaMA token executor
  receives those roles with the StepParams window.
  A separate LLaMA model-resource table uses the same physical and semantic
  hash split for read-only weight resources; Node/WASI and browser smokes derive
  the table from safetensors bytes, bind all tiny LLaMA checkpoint tensors, and
  verify each role keeps dtype, shape, element-count, byte-length, checkpoint
  data-offset evidence, exact initialized bytes, and a tensor-manifest hash.
  Byte-backed safetensors also reject missing, zero-length-for-nonempty,
  overlapping, or out-of-bounds `data_offsets` before the executable
  model-resource manifest is derived, including named byte-backed shard records.
  Strict model-resource requirements reject missing tensors, dtype/shape/byte
  length mismatches, and extra weights before native Session creation, so later
  kernels can trust the bound model layout instead of revalidating checkpoint
  schema at execution time.
	  The proof executor returns the physical and semantic logits/KV table hashes it
	  used, plus the model handle, model-resource table hash, and model manifest
	  hash for model-bound/model-resource sessions; the host bridge requires and
	  validates those identities against the bound Session before accepting an `ok`
	  result. The smokes reject omitted or implicit-success model-resource evidence,
	  omitted registered-pack evidence, and mismatched model-table, model-manifest,
	  or model-pack evidence without position advancement, and verify rejected
	  pack registrations do not linger on the Session.
- `zgml_wasm_alloc` and `zgml_wasm_free` let JS hosts reserve descriptor and
  tensor scratch memory without guessing where Zig's heap begins.
  `zgml_abi_struct_size(kind)` gives those hosts target-native struct byte
  lengths before allocation, including the 32-bit Wasm layout differences from
  native Node/Bun FFI.
- The native dynamic-library FFI step remains separate as `zig build ffi-c`;
  Wasm does not try to use the native dynamic-library artifact shape.
- Single-threaded targets compile graph and fused execution through serial
  thread-pool paths, and quantized GEMV avoids pthread imports when pthreads are
  unavailable.
- GGUF and safetensors loaders use checked `usize` casts for file sizes and
  metadata counts so wasm32 builds fail cleanly on impossible sizes instead of
  failing at compile time.

## Performance Thesis

This can be elegant and fast if the hot path is one host call per token or
prefill chunk, plus bounded StepParams writes and cached WebGPU resource usage.
It will not beat native Metal by default, but it can be the best browser lane if
it keeps buffers GPU-resident, avoids CPU/GPU readback except requested logits,
and keeps dispatch count low through semantic commands rather than per-op graph
execution.
