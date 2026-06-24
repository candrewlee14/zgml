# C ABI Prototype

`src/c_api.zig` is the first FFI slice for the executable-stencil runtime plan.
It intentionally exposes opaque handles instead of tensor or graph internals.
The C declaration surface lives in `include/zgml.h`.

Build it with:

```sh
zig build ffi-c
```

Build the exported Wasm version of the same handle surface with:

```sh
zig build ffi-wasm
```

That writes `zig-out/bin/zgml_c.wasm`. It is a no-entry Wasm module with
exported `zgml_*` functions and exported linear memory, intended as groundwork
for the future Wasm/Node/browser handle lane rather than a complete WebGPU
backend. Wasm hosts must provide the `zgml_host` imports
`zgml_wasm_host_session_step`, `zgml_wasm_host_session_step_no_output`,
`zgml_wasm_host_session_execute_tokens`, and
`zgml_wasm_host_session_free`. Returning the internal unclaimed sentinel from
the shared JS host runtime lets exported calls continue through normal Wasm
execution; returning public `ZGML_UNSUPPORTED` means the host claimed the
session shape but has no executable kernel for that call.

Run the Wasm smoke with Node/WASI:

```sh
zig build ffi-wasm-smoke
```

The smoke instantiates `zgml_c.wasm`, allocates descriptors in exported linear
memory with `zgml_wasm_alloc`, creates a tiny linear model, compiles a Program,
binds a Session, steps into a caller-owned output buffer, binds a native
`zgml_buffer` as Session output storage, inspects executable evidence, and frees
all handles. It also runs the fixed tiny LLaMA handle lane:
semantic/executable inspection, independent sessions, token step, no-output
advance, prompt prefill, too-small output failures that do not mutate position,
caller-owned logits buffers, and compatible model-handle binding with and
without a native `zgml_buffer` logits output. It also writes a complete
all-zero exact tiny LLaMA safetensors checkpoint into a Node/WASI preopened
directory, probes it through `zgml_model_probe_path`, loads it through
`zgml_model_load_path`, also loads the same checkpoint bytes through
`zgml_model_load_safetensors_data`, compiles the loaded model, binds it, and
steps a token through the portable Wasm C ABI.

`zig build check` also runs this Node/WASI Wasm smoke, so the portable handle
surface is part of the normal local gate rather than only a separate example
command.

The same smoke has a browser page at `examples/wasm_ffi/browser.html`. Serve
the repository root, open that page, and it will fetch `zig-out/bin/zgml_c.wasm`
with a small browser-side WASI shim. The page records its machine-readable
result in `document.documentElement.dataset.zgmlWasmSmoke`.

For a repeatable browser run, use:

```sh
zig build ffi-wasm-browser-smoke
```

That launches Chrome/Chromium through a dependency-free CDP runner and passes in
portable mock mode when no browser WebGPU adapter is available. To require real
browser `GPUBuffer` execution evidence, use:

```sh
zig build ffi-wasm-browser-gpu-smoke
```

That target enables Chrome WebGPU flags and fails unless the page runs in real
`GPUBuffer` mode with enough storage-buffer bindings for the LLaMA block
pipeline proof. It also reads page-exported LLaMA profile counters and requires
backend dispatches with zero fallback ops in real-GPU mode.

Run the C smoke executable with:

```sh
zig build ffi-c-smoke
```

The current artifact is a dynamic library named `zgml_c`. The normal local gate
also runs the ABI smoke tests:

```sh
zig build check
```

The C smoke intentionally touches the current `include/zgml.h` surface at
compile/link time, including feature flags, model/backend ids, Wasm helpers,
Program requirements/buffer-factory/profile calls, unified token execution, and
native argmax/sampling/generation helpers. A stale public header should fail
there before an embedder discovers the mismatch.

## Runtime Info

FFI hosts should check the ABI before passing descriptor structs across the
boundary:

```c
zgml_runtime_info info = {0};
zgml_get_runtime_info(&info);
```

`info.abi_version` must match `ZGML_ABI_VERSION`; the current header/runtime ABI
version is 6. The struct also reports the native `size_t` width, pointer width,
token-id width, and a compact feature bitset. The current feature bits cover
native buffer handles, the
compatible-checkpoint model selector, runtime profiles, WebGPU compile-only
inspection, Wasm exports, native buffer I/O, native greedy argmax over
LLaMA-family logits, one-call execute-and-argmax, native greedy generation, and
bounded deterministic top-k sampling with one-call execute-and-sample plus
multi-token generate-and-sample helpers. It also covers cold Program
requirements metadata, Program-sized host buffer factories for weights, bias,
input, and output/logits buffers, WebGPU program-owned device-buffer factories,
same-device device-buffer import handles, and compatible model-handle Session
binding for LLaMA-family Programs. The bitset also distinguishes zgml-owned host
buffer allocation, non-owning external host-buffer wrapping, opaque
external-resource buffer views with read/write access flags, LLaMA-family
KV-cache resource binding descriptors, model-handle inspection before compile,
Program/model compatibility preflight before dynamic Session binding, direct
native tiny-MLP Program/Session support, native module Program support for
rank-1 and batched rank-2 traced Sequential graphs over the current supported
op subset, and explicit persistent Session upload for buffer-backed
tiny-linear, tiny-MLP, and module Programs. It also advertises
`ZGML_FEATURE_NATIVE_MODULE_ACTIVATION_CHAIN` when the runtime understands the
packed activation-chain module descriptor. C embedders build those module Programs with
`zgml_module_program_compile` and the public `ZGML_MODULE_OP_*` vocabulary:
Linear, activation, Softmax, LogSoftmax, LayerNorm, RMSNorm, Embedding, Reshape,
BroadcastTo, bounded Narrow, Slice, activation chains, pooling, Conv2d, Add,
dim-aware reductions, feature-affine, diagonal, and ArgMax/ArgMin reductions.
Activation ops use
`ZGML_MODULE_ACTIVATION_*`, currently ReLU, GELU, SiLU, Sigmoid, Exp, Log, Neg,
Recip, Abs, Sqrt, Square, Sgn, Step, and Tanh. For
`ZGML_MODULE_OP_ACTIVATION_CHAIN`, `flags` is the activation count and
`activation`/`a`/`b`/`c` carry up to four activation IDs. Program inspection
feature bits distinguish external-resource capability, cold memory layout,
static op/buffer shape evidence, and the bounded runtime patch envelope. Session
inspection also carries an opaque binding-shape hash so hosts can prove a bound
resource/host table stayed stable without exposing backend internals or host
pointers. ABI v6 also exposes Program dispatch-plan inspection fields:
`backend_dispatch_count`, `dispatch_plan_supported`,
`dispatch_plan_covered_op_count`, first unsupported op, and dispatch-family
counts for projection, row, attention, movement, elementwise, RoPE, and
quantized projection work. These fields are populated only for real native
WebGPU executable Programs today; compile-only and non-WebGPU paths leave them
empty with `UINT64_MAX` as the "no unsupported op reported" sentinel. Native
builds compiled with `-Duse-wgpu=true` also set
`ZGML_FEATURE_NATIVE_WGPU_EXECUTION`; that bit is a build-level capability hint,
while `zgml_program_inspect(...).execution_supported` and
`dispatch_plan_supported` remain the per-Program truth for a selected
model/backend. Native builds compiled with `-Duse-wgpu=true` now enable the
LLaMA-family WebGPU execution lane by default and set
`ZGML_FEATURE_EXPERIMENTAL_LLAMA_WGPU_EXECUTION`; passing
`-Dexperimental-llama-wgpu-execution=false` forces the older resource-probe-only
mode. The feature bit means this artifact includes that native LLaMA execution
lane, while still leaving per-Program inspection as the execution truth.
`ZGML_FEATURE_ABI_STRUCT_SIZE` announces
`zgml_abi_struct_size(kind)`, which returns the target-native `sizeof` for
public descriptor, inspection, requirements, and profile structs, or zero for
an unknown kind. Node and Bun expose this as `abiStructSize(...)` and
`abiStructSizes()`, and Node/WASI plus browser Wasm smokes derive their
linear-memory descriptor allocations from it instead of duplicating C layout
math in JavaScript.
`ZGML_FEATURE_SAFETENSORS_DATA_LOAD` announces
`zgml_model_load_safetensors_data(...)`, the pathless checkpoint load lane for
hosts that already own complete safetensors bytes.
`ZGML_FEATURE_SAFETENSORS_DATA_PROBE` announces
`zgml_model_probe_safetensors_data(...)`, the matching metadata-only preflight
lane over a complete safetensors byte blob.
Node, Bun, Node/WASI, and browser smokes all exercise this handshake before
relying on the handle API.
Node and Bun expose the raw `featureFlags` plus decoded
`runtimeInfo().features.*` booleans for the ABI vocabulary, including
`features.modelAuto`, `features.wasmExports`, the device-buffer factory/import
bits, `features.abiStructSize`, `features.modelPathProbe`,
`features.supportedCheckpoints`, `features.safetensorsHeaderProbe`,
`features.safetensorsDataLoad`, `features.safetensorsDataProbe`,
`features.nativeTinyMlp`, `features.sessionPersistentUpload`,
`features.nativeWgpuExecution` for the
build-level wgpu hint, and `features.experimentalLlamaWgpuExecution` for the
compiled-in native LLaMA WebGPU execution lane. The Node/WASI and browser Wasm smokes
decode and assert the same feature vocabulary before running their handle
lifecycle checks. The C
ABI tests plus Node/WASI and browser Wasm smokes also call the device-buffer
factory, device-handle, and import exports and prove compile-only/portable paths
clear output handles and report `ZGML_UNSUPPORTED` until a real shared WebGPU
device exists. Node and Bun smokes prove the higher-level
`program.device("webgpu")` object, device buffer factory, and raw import helpers
also throw `unsupported` on compile-only WebGPU Programs. Program capabilities
still remain the per-Program execution truth. Node, Bun, Node/WASI, and browser
Wasm helpers derive the same compact execution mode vocabulary from Program
inspection: `"executable"`, `"resource-probe"`, or `"compile-only"`.

## Handle Lifecycle

The C-facing lifecycle mirrors the planned runtime vocabulary:

```text
zgml_model_create
  -> zgml_model_inspect
  -> zgml_program_compile
  -> zgml_session_bind
  -> zgml_session_step
  -> free session/program/model
```

Real checkpoint-backed model handles use the same lifecycle after loading:

```text
zgml_model_load_path
  -> zgml_model_inspect
  -> zgml_program_compile
  -> zgml_session_bind
  -> zgml_session_execute_tokens
  -> zgml_session_reset / zgml_session_position as needed
  -> free session/program/model
```

The handles are opaque:

```c
typedef struct zgml_model zgml_model;
typedef struct zgml_program zgml_program;
typedef struct zgml_session zgml_session;
typedef struct zgml_buffer zgml_buffer;
```

Model handles can be inspected before compiling a Program:

```c
zgml_model_inspection model_info = {0};
zgml_model_inspect(model, &model_info);
```

`zgml_model_inspection` exposes the model kind and fixed shape envelope. Tiny
linear handles report input/output lengths. LLaMA-family handles report vocab
size, max sequence length, hidden size, layer/head counts, feed-forward width,
RoPE base, RMS norm epsilon, and whether the LM head is tied. This gives C,
Wasm, Node, and Bun hosts a cheap compatibility preflight before asking zgml to
compile an executable stencil. It is metadata only: it does not traverse the
graph, allocate runtime state, or execute tokens.

Ownership is explicit. Callers free each successful handle with:

```c
zgml_session_free(session);
zgml_program_free(program);
zgml_model_free(model);
```

Native-owned host buffers use the same explicit ownership style:

```c
zgml_buffer *buffer = NULL;
zgml_buffer_create(&(zgml_buffer_desc){ .byte_len = 4096 }, &buffer);
size_t bytes = zgml_buffer_size(buffer);
zgml_buffer_write(buffer, 0, src, byte_len);
zgml_buffer_read(buffer, 0, dst, byte_len);
zgml_buffer_free(buffer);
```

Callers that already own stable host memory can wrap it without transferring
ownership:

```c
float logits[49152];
zgml_buffer *view = NULL;
zgml_buffer_wrap(logits, sizeof(logits), &view);
/* bind view as Session output */
zgml_buffer_free(view); /* frees only the handle, not logits */
```

`zgml_buffer` has two intentionally separate shapes today. `zgml_buffer_create`
and `zgml_buffer_wrap` produce host-memory handles: the first owns storage, and
the second is a non-owning view over caller-owned host memory or Wasm linear
memory. `zgml_buffer_wrap_resource` produces an opaque external-resource view
with a backend placement, read/write access flags, caller-provided handle, byte
offset, and byte length. `zgml_buffer_size` works for both shapes. `zgml_buffer_data`,
`zgml_buffer_write`, and `zgml_buffer_read` are host-only and return
`ZGML_UNSUPPORTED` for resource views. `zgml_session_bind_buffers` can receive
either shape, but current CPU/Metal/stencil execution rejects resource views
with `ZGML_UNSUPPORTED` until a backend advertises real external-resource
support. LLaMA-family buffer binding now preserves output and KV-cache resource
descriptors through the Program/Session bind path instead of first converting
them to host pointers; host-only backends still reject them. Resource-capable
backends receive the complete persistent/input/output table at Session bind
time, validate that input/persistent resources are readable and output resources
are writable before upload/execute, and can build backend resource tables before
the hot step path. Normal Session execution now runs against that configured
bound I/O table; per-call output overrides and cache side-effect-only paths use
an explicit dynamic I/O route so descriptor mutation stays visible without
making every token step rebuild the table. This is the ABI seam for future
WebGPU/wgpu/Node resource binding; it is not yet zero-copy GPU execution.
`zgml_buffer_inspect` exposes which storage shape a handle has, plus external
resource placement, access flags, handle, view offset, view byte length, and
backing resource byte length, so embedders can preflight resource-backed
bindings before trying to execute them.

```c
zgml_buffer *gpu_view = NULL;
zgml_buffer_wrap_resource(&(zgml_external_resource_desc){
    .placement = ZGML_BACKEND_WEBGPU,
    .access_flags = ZGML_RESOURCE_ACCESS_WRITE,
    .handle = gpu_buffer_id,
    .byte_offset = 0,
    .byte_len = logits_bytes,
}, &gpu_view);
zgml_buffer_inspection gpu_info = {0};
zgml_buffer_inspect(gpu_view, &gpu_info);
/* CPU/Metal currently reject binding this view for execution. */
zgml_buffer_free(gpu_view);
```

LLaMA-family programs have a LLaMA-specific buffer descriptor for binding
persistent logits output and one K/V cache resource pair per layer:

```c
zgml_buffer *k_layers[LAYERS] = { ... };
zgml_buffer *v_layers[LAYERS] = { ... };

zgml_llama_kv_cache_bind_desc kv = {
    .k = k_layers,
    .v = v_layers,
    .len = LAYERS,
};
zgml_llama_buffer_bind_desc bind = {
    .output = logits_view,       /* optional */
    .output_len = vocab_size,
    .kv_cache = &kv,             /* optional */
};
zgml_llama_session_bind_buffers(program, &bind, &session);
```

The KV buffers may be host `zgml_buffer` storage or opaque external-resource
views, but all K/V entries in one descriptor must use the same storage kind. The
C/Wasm ABI exposes the required per-layer size with
`zgml_llama_program_get_kv_cache_requirements`, and Node/Bun expose the same
Program query as `program.kvCacheRequirements()`. Its byte lengths are derived
from the compiled Program's current executable KV layout: each layer uses packed
`[d_head, contextLength * nKvHeads]` K and V buffers.
Host KV buffers execute on CPU/Wasm today; CPU/Metal host execution still
rejects external-resource KV views with `ZGML_UNSUPPORTED`.

Model, Program, and Session handles retain their parent handles internally:
freeing a `zgml_model` after compiling a Program, or freeing a `zgml_program`
after binding a Session, releases the caller's reference without invalidating
the child handle. Each successful public handle should still be freed exactly
once. Program handles can be reused after a session is freed, so callers can
compile once and bind a new compatible weight set without rebuilding the
executable shape.

For tiny linear programs, `zgml_bind_desc` can also bind caller-owned step I/O:
`input/input_len` and `output/output_len` are optional. If they are present,
the Session keeps those host buffers as its step input/output bindings and
`zgml_session_step(session, NULL, &result)` executes against the bound buffers.
If they are omitted, callers keep using `zgml_step_desc` to provide input and
output buffers for each call. This is the first small ABI proof of the target
shape where a Session owns the runtime binding table and a hot step only updates
small params before executing.
`zgml_session_step_no_output(session, desc, &result)` is the matching
side-effect-only tiny-linear step: it accepts the same optional input descriptor,
rejects output pointers, executes the bound Program/Session without downloading
outputs, and reports `result.output_len = 0`. Node and Bun expose this as
`session.advance(input?)`.
For `zgml_session_bind_buffers` sessions, buffer-backed persistent state can be
refreshed after bind without rebuilding the Session:
`zgml_session_upload_persistent(session)` uploads every persistent binding, while
`zgml_session_upload_persistent_range(session, first, len)` uploads a contiguous
persistent binding range. Mutating a host `zgml_buffer` does not affect the
compiled runtime until one of those upload calls succeeds and a later step
executes. Node and Bun expose the same contract as `session.uploadParameters()`
and `session.uploadParameterRange(first, len)`.
`zgml_session_inspect(session, &info)` reports that bound runtime shape: model
kind, backend, position/context, output and K/V storage kind, persistent/step
binding counts, host-vs-resource binding counts, and an opaque binding-shape
hash. Host pointer addresses are not part of the hash; external-resource
placement, handle, byte offset, byte length, and access flags are. Node and Bun
expose the same information as `session.inspect()`. Wasm WebGPU host-only
Sessions use the wrapped export facade to fill the same ABI struct from
host-owned resource metadata.

`zgml_session_reset(session)` resets reusable session state. It is a no-op for
tiny linear sessions and rewinds LLaMA-family position/KV state while preserving
already-bound executable decode/prefill sessions when their cache bindings can
be refreshed. The same Session handle can replay a prompt or start a new
generation without recompiling or rebinding in the normal path. In Wasm WebGPU
hosts, native-backed resource Sessions route reset through the host bridge, and
host-only Sessions use the wrapped export facade to preserve the same replay
semantics.

For LLaMA-family programs, `zgml_bind_desc.output/output_len` can bind a
caller-owned logits buffer to the Session. When that buffer is present,
`zgml_session_step_token` and `zgml_session_prefill_tokens` may omit their
per-call output pointer and write into the bound logits buffer. Passing an
explicit output buffer per call remains supported.

LLaMA-family Programs can also bind a compatible model handle as the Session's
persistent weight/KV source:

```c
zgml_program_model_compatibility compat = {0};
zgml_program_check_model_compatibility(program, compatible_model, &compat);
if (!compat.compatible) {
    /* choose a different model or compile a new Program */
}

zgml_session_bind_model(program, compatible_model, NULL, &session);
zgml_session_bind_model_buffers(program, compatible_model, &buffer_desc, &session);
```

This is the first honest FFI shape for dynamic LLaMA weights: the checkpoint is
a structured model handle, not a flat `float *` blob. The compiled Program shape
is reused, while Session binding clones the compatible model's runtime state and
uploads its persistent resources.
`zgml_program_check_model_compatibility` uses the same model-kind rule as
Session binding but does not allocate or clone runtime state, so C/Wasm/JS hosts
can reject incompatible handles without treating bind failure as the query.

`zgml_session_execute_tokens(session, desc, result)` is the unified LLaMA-family
token-window primitive. `desc.tokens/tokens_len` is the runtime token window as
compact `uint32_t` token ids, and `desc.output_policy` selects whether the call
should write final-token logits or just advance Session state:

```c
const uint32_t *tokens = ...;
zgml_token_execute_desc desc = {
    .tokens = tokens,
    .tokens_len = tokens_len,
    .output_policy = ZGML_EXECUTE_OUTPUT_LOGITS, // or ZGML_EXECUTE_OUTPUT_NONE
    .output = logits,
    .output_len = logits_len,
};
zgml_session_execute_tokens(session, &desc, &result);
```

`ZGML_EXECUTE_OUTPUT_NONE` rejects output pointers and advances the token window
without returning logits. `ZGML_EXECUTE_OUTPUT_LOGITS` writes final-token logits
into `desc.output`, or into the Session-bound logits buffer when the output
pointer is omitted. This mirrors native Zig `LlamaExecuteParams` and is the
StepParams-shaped ABI entry point; the older `zgml_session_step_token`,
`zgml_session_advance_token`,
`zgml_session_advance_tokens`, and `zgml_session_prefill_tokens` calls remain as
convenience forms and internally route through the same execute-token handler.
Single-token convenience descriptors also use `uint32_t` token ids. Native Zig
converts these compact ABI ids to its internal `usize` token representation at
the boundary without heap allocation.
Prompt KV-cache side effects are synchronized into the Session before decode
resumes, so a later token execution sees the advanced prompt state rather than a
separate prefill-only cache copy.

Compiled programs can be inspected before binding:

```c
zgml_program_inspection inspection = {0};
zgml_program_inspect(program, &inspection);
```

The inspection fields expose compact executable evidence: selected backend id,
whether session binding/execution is currently supported for that compiled
backend, whether external-resource bindings are supported before bind, planned
op count, buffer count, total buffer elements, byte length, initial upload
count, qweight count, command count, command hash, runtime patch-hole shape,
runtime patch envelope bounds, pre-bind binding requirement hash/counts, and
stable command-category counts for op, row, projection, attention, movement,
elementwise, and RoPE command families. The binding requirement fields describe
the persistent tensors, per-step inputs, and per-step outputs a Session must
bind for this compiled Program; Session inspection separately reports the
actual host/resource bindings after bind. The
envelope fields report max cache write position and max attention sequence
length; `UINT64_MAX` means the bound is unknown/unrepresentable, and the JS
wrappers expose that sentinel as `null`. For
LLaMA-family handles this evidence is produced during `zgml_program_compile` by
a cold decode-shape stencil compile, then read back by `zgml_program_inspect`;
it does not execute tokens. Tiny linear programs expose the same compact
executable evidence for the smallest tensor graph.

`zgml_compile_desc` selects the compile envelope for compiled programs:

```c
zgml_compile_desc desc = {
    .backend = ZGML_BACKEND_CPU, // AUTO, CPU, METAL, or WEBGPU
    .context_len = 1024,         // 0 means the model maximum
    .batch = 1,                  // 0 means the current default, also 1
};
zgml_program_compile(model, &desc, &program);
```

`ZGML_BACKEND_AUTO` is the stable default and currently maps to the CPU
executable path. `ZGML_BACKEND_CPU` requests the CPU runtime explicitly.
`ZGML_BACKEND_METAL` requests the Metal runtime when the C library was built for
macOS with the Metal backend linked. `ZGML_BACKEND_WEBGPU` is a WebGPU-shaped
inspection target when the native wgpu executor is not compiled in. When the
native C library is built with `-Duse-wgpu=true`, tiny-linear, direct tiny-MLP,
traced module, and supported LLaMA-family WebGPU Programs use the wgpu-native
executor and report `execution_supported = 1`. Tiny-linear and LLaMA-family
paths can bind host or same-device resources where their Program inspections
advertise it; direct tiny-MLP Programs currently execute with host-bound buffers
and do not advertise external-resource binding, while traced module Programs
expose the generic external-resource binding shape and leave execution support
to per-Program inspection.
Passing `-Dexperimental-llama-wgpu-execution=false` keeps LLaMA-family WebGPU
Programs on the compile-only/resource-probe lane while leaving tiny-linear,
direct tiny-MLP, and traced module WebGPU execution available.
`ZGML_FEATURE_NATIVE_WGPU_EXECUTION` tells a host whether the native library was
built with the wgpu executor available; `zgml_program_inspect` is the
authoritative way to distinguish each compiled Program.
Unknown backend IDs fail with `ZGML_INVALID_ARGUMENT`; lanes that do not support
the requested backend fail with `ZGML_UNSUPPORTED`. For LLaMA-family programs,
`context_len`
becomes the compiled Session context envelope: executable `execute`, `step`,
`advance`, and `prefill` calls fail before mutating state if they would exceed
it. Batched LLaMA decode is not exposed yet, so `batch` values other than `0`
or `1` fail with `ZGML_UNSUPPORTED`.

## Current Scope

This is deliberately a small prototype, not the final dynamic model-loading
LLaMA ABI. The ABI has two tiny smoke lanes, one fixed real-checkpoint lane, and
a first compiled-in compatible-checkpoint selector:

- `ZGML_MODEL_TINY_LINEAR` constructs a small linear `ComputeGraph`, compiles it
  through an explicit `DeviceInference.Program` on the CPU backend, and records
  the persistent weight/bias buffers as session-bound inputs without uploading
  them as compile-time initial data. `zgml_session_bind` creates a
  `DeviceInference.Session` and uploads those weights once through the backend
  persistent-binding path. Callers may either pass input/output through each
  `zgml_step_desc` or bind caller-owned input/output buffers once in
  `zgml_bind_desc`; `zgml_session_step` then updates only the per-step input
  source and executes through `StepParams`.
- `ZGML_MODEL_TINY_LLAMA` constructs a fixed tiny LLaMA model/program/session
  lifecycle and exposes token stepping through `zgml_session_step_token`, which
  writes logits into a caller-owned output buffer. `zgml_program_compile` owns a
  persistent decode `DeviceInference.Program` for the selected backend, and
  sessions bind independent KV-cache state, lifecycle, and evidence behind that
  executable shape. f32 decode parameters are Program-shaped but Session-bound:
  compile includes their buffers in the executable shape while bind uploads the
  runtime session's parameter storage through the backend persistent-binding
  path.
  `zgml_session_bind_model` and `zgml_session_bind_model_buffers` can bind a
  second compatible tiny LLaMA model handle behind the same compiled Program,
  proving the FFI surface can reuse a compiled architecture while varying the
  Session's model state.
  `zgml_bind_desc.output` can bind the caller's logits buffer once, so token
  step and prompt prefill can execute without repeating the output pointer in
  every descriptor.
  `zgml_session_reset` clears a bound LLaMA session back to position zero while
  keeping the Program handle, Session ownership, and already-bound executable
  sessions intact when their runtime cache bindings refresh cleanly.
  `zgml_session_execute_tokens` is the unified token-window execution call: it
  supports one-token decode, prompt prefill with logits, and no-output prompt
  advancement through one descriptor with an explicit output policy.
  `zgml_session_advance_token` updates the token sequence without downloading
  logits, for prompt tokens whose probabilities are not needed.
  `zgml_session_advance_tokens` does the same for a prompt token window and
  proves the prefill executable cache updates hand off to the decode
  executable before the next token step.
  `zgml_session_prefill_tokens` runs a prompt token window through the
  persistent executable prefill Program and writes final-token logits into a
  caller-owned output buffer. It also exposes `zgml_llama_program_inspect` for
  pre-bind LLaMA architecture, context, and semantic runtime-patch shape evidence,
  `zgml_llama_program_get_kv_cache_requirements` for executable-layout K/V cache
  buffer sizing, and `zgml_program_inspect` for decode executable command-shape
  evidence. This is still a fixed tiny model, not the GGUF/SmolLM ABI.
- `ZGML_MODEL_SMOLLM_135M` is the first real checkpoint-backed lane. It is
  created with `zgml_model_load_path`, loads a local SmolLM-135M GGUF or
  safetensors file into the fixed SmolLM config used by the benchmark, then uses
  the same compile, semantic inspect, executable inspect, bind, position, and
  token-step/advance C handles. `zgml_program_compile` owns a persistent decode
  `DeviceInference.Program` for the selected backend, and sessions bind
  independent KV-cache state, lifecycle, and evidence behind that executable
  shape. On CPU and Metal, compatible direct quantized GGUF qweight payloads
  are shape-validated at compile time and uploaded into session runtime state at
  bind time, so independent quantized sessions can share one compiled
  executable Program.
- `ZGML_MODEL_AUTO` is the first public compatible-checkpoint selector. It is
  valid for `zgml_model_load_path` and inspects GGUF metadata or safetensors
  tensor-shape metadata before loading. Today it walks the compiled-in LLaMA
  family registry, accepts only checkpoints that exactly match one executable
  family, currently the fixed tiny LLaMA smoke envelope and SmolLM-135M, and
  returns `ZGML_UNSUPPORTED` for other shapes. The final ABI should broaden this
  into a general compatible model and backend selector by adding families to
  that registry rather than open-coding one checkpoint path.
  `zgml_supported_checkpoint_count()` and
  `zgml_supported_checkpoint_inspect(index, out)` expose that compiled-in
  registry as `zgml_model_inspection` records, so C, Node, Bun, and Wasm hosts
  can list the loadable checkpoint envelopes before probing a path. Today the
  catalog has two exact entries, fixed tiny LLaMA and SmolLM-135M, and invalid
  indexes return
  `ZGML_INVALID_ARGUMENT` after clearing the output inspection struct.
  `zgml_model_probe_safetensors_header(...)` accepts a safetensors JSON header
  directly and returns the matching `zgml_model_inspection` without a file path,
  model handle, or tensor data. This gives browser/Wasm hosts the same
  compatibility preflight as `zgml_model_probe_path(...)` after they fetch only
  a safetensors header range; it is still an exact-envelope selector, not a
  promise to load arbitrary safetensors families.
  `zgml_model_probe_safetensors_data(...)` accepts the full safetensors byte
  slice and returns the same inspection without loading weights or allocating a
  model handle, so C/Wasm/JS hosts do not need to parse the safetensors length
  prefix themselves just to preflight a byte blob.
  `zgml_model_load_safetensors_data(...)` takes the full safetensors byte slice
  and returns a real model handle through the same exact-envelope selector.
  This is the pathless browser/Wasm and Bun/Node-shaped load lane: callers can
  keep checkpoint acquisition in JS or another host, pass bytes across the ABI
  once, and avoid manufacturing a filesystem path just to enter zgml.
  The C ABI also exposes prompt prefill through `zgml_session_prefill_tokens`,
  using the same persistent executable prefill Program shape as the benchmark.

Each FFI `zgml_session` owns its host-side persistent, input, and output buffers.
That keeps JS/C buffers as session state rather than hidden global program
state. Internally, `DeviceInference.Session` also owns a backend runtime handle.
On CPU, that runtime handle owns an independent buffer table and mutable
`ProgramStencil`, so compatible sessions can keep distinct persistent weights
behind one compiled program. Metal exposes the same handle shape and per-session
buffers/stencil. Bound Metal execution now constructs an explicit `RuntimeView`
for upload, patching, scheduling, command lowering, fallback execution, kernel
buffer binding, and downloads. Lower Metal GPU helpers receive that view
directly instead of reading a hidden active runtime view from the compiled
encoder, and Metal command buffers are now owned by a per-execution encoder
context instead of the backend singleton. DeviceInference profile inspection and
reset now call backend-owned snapshot/reset hooks. C, Node, and Bun callers can
read/reset Program and Session runtime profiles, and Metal serializes default
Program plus bound Session profile windows behind their owning profile locks.

That means the smoke path now exercises graph lowering into backend
`DeviceProgram` and the same internal `ProgramStencil` kind used by executable
backends instead of doing hand-written C-ABI math. LLaMA decode compilation now
happens at `zgml_program_compile`; prompt prefill still compiles fixed-window
Programs on demand because the prompt chunk length is part of that Program
shape.

The current slice proves:

- opaque handle ownership
- ABI/runtime-info handshake for C, Node, Bun, Node/WASI, and browser Wasm
  hosts before descriptor structs are driven across the FFI boundary
- opaque native host buffer ownership through `zgml_buffer_create`,
  non-owning external host-buffer views through `zgml_buffer_wrap`,
  opaque external-resource views through `zgml_buffer_wrap_resource`,
  `zgml_buffer_write`, `zgml_buffer_read`, `zgml_buffer_size`,
  `zgml_buffer_inspect`, and
  `zgml_buffer_free`; resource views are size-queryable but host read/write/data
  access and current host-only execution return `ZGML_UNSUPPORTED`; inspection
  distinguishes host storage from external-resource views and surfaces resource
  placement/access/offset metadata for future resource-capable WebGPU/wgpu
  backends
- handle-backed Session binding through `zgml_session_bind_buffers`, so FFI
  hosts can bind zgml-owned or explicitly wrapped caller-owned weights, inputs,
  and outputs without exposing graph tensors; traced module Programs accept
  rank-1 and batched rank-2 host buffers and can carry WebGPU external-resource binding shape for
  resource-probe evidence, while LLaMA-family output resources are routed
  through the executable bind path and rejected by current host-only backends
  rather than treated as host memory
- Session inspection through `zgml_session_inspect`, so C, Node, Bun,
  Node/WASI, and browser Wasm callers can verify the bound Program/Session
  runtime shape, including position/context, output/KV storage kind, and
  host-vs-resource binding counts plus an opaque binding-shape hash, without
  exposing backend internals
- LLaMA-specific handle-backed resource binding through
  `zgml_llama_session_bind_buffers` and
  `zgml_llama_session_bind_model_buffers`, so FFI hosts can name persistent
  logits outputs and per-layer K/V cache external resources without overloading
  the generic tensor buffer descriptor; Node, Bun, Node/WASI, and browser Wasm
  smokes prove host-only backends reject those KV resource descriptors cleanly
- compatible model-handle Session binding through `zgml_session_bind_model` and
  `zgml_session_bind_model_buffers`, so LLaMA-family FFI callers can compile a
  Program from one model and bind a fresh compatible model handle as Session
  state without flattening checkpoint weights into an artificial C array
- cold Program requirements through `zgml_program_get_requirements`, so FFI
  hosts can allocate edge buffers from the compiled Program's declared
  input/output/logits/context envelope instead of duplicating model constants
- cold Program capability decisions derived from `zgml_program_inspect`, so
  Node, Bun, Node/WASI, and browser Wasm hosts can distinguish executable
  Programs, inspectable external-resource layouts, and compile-only Programs
  without branching on a backend enum alone
- LLaMA-specific KV-cache requirements through
  `zgml_llama_program_get_kv_cache_requirements`, so C, Node, Bun, Node/WASI,
  and browser Wasm callers can size per-layer K/V host or resource buffers from the
  Program handle instead of reimplementing LLaMA head/context math
- Program-sized native buffers through `zgml_program_create_buffer`, so FFI
  hosts can allocate correctly sized weights, bias, input, and logits/output
  `zgml_buffer` handles directly from the compiled Program contract;
  `zgml_program_create_output_buffer` remains a compatibility shorthand for the
  output/logits role
- native greedy argmax through `zgml_session_argmax_token`, including both
  caller-provided logits and Session-bound logits buffers
- one-call greedy execution through `zgml_session_execute_argmax_tokens`, so
  FFI hosts can execute into bound logits and receive only the selected token
- native greedy token generation through
  `zgml_session_generate_argmax_tokens`, so FFI hosts can pass prompt token IDs
  and an output token-id buffer while native code loops over execute/argmax
  steps without copying logits across the boundary
- bounded deterministic top-k sampling through `zgml_session_sample_token`,
  including both caller-provided logits and Session-bound logits buffers, backed
  by the same bounded sampler used by the Zig-native LLaMA helpers
- one-call sampled execution through `zgml_session_execute_sample_tokens`, so
  FFI hosts can execute into bound logits and receive only the sampled token
- native sampled token generation through
  `zgml_session_generate_sample_tokens`, so FFI hosts can pass prompt token IDs
  and an output token-id buffer while native code loops over execute/sample
  steps without copying logits across the boundary. The sampler treats `seed`
  as a base seed and increments it by generated-token index inside the native
  loop, using the shared bounded sampler implementation rather than a parallel
  C-ABI-only algorithm.
- model -> program -> session -> step structure
- persistent bound weights uploaded without executing the program
- f32 LLaMA executable decode parameters uploaded as Session bindings rather
  than compile-time initial data
- CPU and Metal direct quantized LLaMA executable decode qweights uploaded as
  Session bindings after compile-time shape validation
- sequential rebinding of compatible weight sets through one compiled program
- fixed tiny LLaMA model/program/session handles
- explicit `zgml_compile_desc` backend/context/batch parsing, selected-context
  inspection, and clean invalid-backend/envelope failures
- WebGPU inspection through the same model/program handle lifecycle. Default
  tiny-linear, direct tiny-MLP, traced module, and LLaMA-family Programs report
  WebGPU-shaped compile-only/resource-probe evidence with
  `execution_supported = 0`.
  Tiny-linear also has an opaque external-resource Session probe through
  `zgml_session_bind_buffers`, so embedders can inspect WebGPU-shaped bindings
  before execution exists; fixed tiny LLaMA can likewise bind an
  external-resource logits output plus external-resource K/V cache for Session
  inspection. Native builds compiled with `-Duse-wgpu=true` make tiny-linear,
  direct tiny-MLP, and traced module WebGPU host-memory binds executable through
  the opt-in wgpu-native backend. Direct tiny-MLP Programs still do not claim
  same-device/external-resource binding, while traced module Programs expose the
  generic resource-bind shape;
  LLaMA-family WebGPU execution depends on the experimental build flag and
  per-Program inspection.
- pre-bind fixed tiny LLaMA semantic program inspection
- pre-bind fixed tiny LLaMA decode executable program inspection
- fixed tiny LLaMA executable selected-backend decode binding with CPU
  descriptor coverage and multi-token sequence correctness
- fixed SmolLM-135M GGUF/safetensors loading through `zgml_model_load_path`
- registry-backed compatible GGUF/safetensors auto-selection for exact fixed
  tiny LLaMA and SmolLM-135M envelopes through `ZGML_MODEL_AUTO`
- SmolLM-135M compile/semantic-inspect/executable-inspect/bind/token-step
  handles when a local checkpoint is available
- unified LLaMA token-window execution through `zgml_session_execute_tokens`
  with both final-logits and no-output policies
- fixed tiny LLaMA and SmolLM prompt prefill through
  `zgml_session_prefill_tokens`, backed by a persistent executable prefill
  Program, caller-owned logits buffers, and Session-synchronized KV-cache
  side effects before decode resumes
- caller-owned LLaMA logits buffers via `zgml_session_step_token`, wired as the
  executable session's per-step output binding rather than a post-step copy
- no-output LLaMA token advancement via `zgml_session_advance_token`
- no-output bulk LLaMA prompt advancement via `zgml_session_advance_tokens`,
  with regression coverage that bulk prompt advancement plus decode matches
  a single executable prefill over the same token window
- independent LLaMA session positions behind one compiled program handle
- backend runtime handles owned by sessions
- per-step input/output parameters
- pre-bind executable and LLaMA semantic program inspection evidence
- shape validation and numeric status errors
- graph-built `DeviceInference` execution
- explicit Metal runtime views down to kernel buffer binding helpers
- per-execution Metal command buffers and transient command-profile counters
- backend-owned profile inspection/reset hooks for DeviceInference
- synchronized Metal Program/Session profile windows
- C, Node, and Bun runtime-profile snapshots and resets for Program/Session
  handles
- a real C header and C smoke executable
- a dynamic library target suitable for Node/Bun FFI experiments
- an exported Wasm module target for the same handle surface
- a Node/WASI smoke over the Wasm tiny linear, `zgml_buffer`, and tiny LLaMA
  handle lifecycle, including compatible model-handle Session binding and the
  shared WebGPU LLaMA resource-session token bridge, including the
  wrapped-export `zgml_session_execute_tokens` compatibility path
- a browser smoke page over the same Wasm tiny linear, `zgml_buffer`, and tiny
  LLaMA lifecycle, including compatible model-handle Session binding and the
  WebGPU LLaMA resource-session token bridge that claims browser host resources
  before returning unsupported without mutating native state, plus host-side
  counters for decoded token calls, output policies, validation rejections, and
  result writes, including the wrapped-export `zgml_session_execute_tokens`
  compatibility path

`zgml_program_inspect` is the generic executable command-shape inspection
surface. For LLaMA-family programs it returns cached decode command/runtime
patch evidence produced by `zgml_program_compile`, including backend/execution
support, external-resource bind support, memory-layout counts, runtime patch
envelope bounds, and command-category counts that FFI hosts can assert without
knowing the internal command enum. The patch envelope is the compiled
executable graph's patch-table bound. `zgml_llama_program_inspect` still reports
the public LLaMA compile context, and that context determines the executable
KV-cache tensor span plus runtime patch envelope.
For native WebGPU Programs that are actually executable, ABI v6 also reports the
lowered backend dispatch plan through the same inspection struct, so hosts can
assert full op coverage, no unsupported op, dispatch count, and dispatch-family
mix before binding buffers. Current public LLaMA WebGPU remains compile-only, so
those dispatch-plan fields deliberately stay empty there. The ABI translates the
generic native `DeviceInference.Program.inspect().execution_plan` evidence; it
does not reach around the Program abstraction to ask a specific backend for
private details.
`zgml_llama_program_inspect` remains the semantic LLaMA architecture/context inspection surface, while
`zgml_llama_program_get_kv_cache_requirements` is the concrete buffer-sizing
surface for external K/V resources. ABI v6 also exposes
`zgml_model_probe_path`: given a `zgml_model_load_desc`, it opens GGUF or
safetensors metadata, selects the compatible compiled-in model envelope, and
returns `zgml_model_inspection` without loading weights or allocating a model
handle. `zgml_model_load_safetensors_data` is the matching pathless load
function: it reads the header from a complete safetensors byte slice, selects a
compatible compiled-in envelope, loads the tensor payload, and returns a model
handle. Today these selectors accept only compatible compiled-in LLaMA families
such as the exact tiny LLaMA smoke envelope and SmolLM-135M; unsupported shapes
return `ZGML_UNSUPPORTED` instead of pretending the library can run them. The
next ABI milestone is broadening that selector into a general LLaMA-family ABI.

## Bun / Node Shape

The JS wrappers hide raw FFI details. The concrete Bun FFI adapter now lives in
`src/ts/adapters/bun_ffi_runtime.ts`, selected through the TS-owned
`src/ts/adapters/bun_concrete_runtime.ts` loader policy; the package `zgml/bun` subpath resolves through the TS-built
`dist/bun_native.cjs` bridge under Bun, with a smoke program in
`examples/bun_ffi/smoke.ts`:

```sh
zig build ffi-bun-smoke
```

That target expects Bun to be installed and builds the dynamic C ABI library
before running the smoke.

The package root and `zgml/node` subpath resolve through the TS-built
`dist/node.cjs` native bridge when package artifacts are present. That bridge
loads the `tsdown`-emitted concrete Node FFI runtime at
`dist/adapters/node_ffi_runtime.cjs` without falling back through
`js/generated/**`, after checking it against the TS native API contract; the
implementation source lives at `src/ts/adapters/node_ffi_runtime.ts`, and
`tsdown.config.mjs` emits its JS artifact while excluding that internal concrete
adapter from public declaration bundling. Checked Node host loading/path policy
lives in `src/ts/adapters/node_host_runtime.ts`, checked ABI struct registration
lives in `src/ts/adapters/node_abi_structs.ts`, checked native status/error and
handle extraction policy lives in `src/ts/adapters/node_status.ts`, checked
	generic TinyLinear/TinyMlp/Program/Session public class surface lives once in
	`src/ts/runtime/generic_family_surface.ts`, with Node native model-handle adapter glue
	in `src/ts/adapters/node_generic_family_surface.ts`, checked NativeBuffer public
class surface lives in `src/ts/adapters/node_native_buffer_surface.ts`, checked
LLaMA token-step/execute/selection ABI calls live in
`src/ts/adapters/node_llama_token_ops.ts`, checked shared LLaMA
model/program/session public class surface lives in
`src/ts/runtime/llama_family_surface.ts`, with no adapter compatibility
re-export path, checked LLaMA KV-cache/session bind
ABI packing lives in `src/ts/adapters/node_llama_session_bind_ops.ts`, while
shared LLaMA Session bind output/KV-cache validation and descriptor fields live
in `src/ts/runtime/llama_session_bind_desc.ts`. Checked shared LLaMA KV-cache
allocation/resource policy lives in `src/ts/runtime/llama_kv_cache.ts`, with
Node object/program-device adapter glue in `src/ts/adapters/node_llama_kv_cache_ops.ts`, checked
Program buffer/device ABI policy lives in
`src/ts/adapters/node_program_buffer_ops.ts`, checked
generic Program bind ABI policy lives in
`src/ts/adapters/node_program_bind_ops.ts`, checked
generic Session upload/step ABI policy lives in
`src/ts/adapters/node_session_ops.ts`, checked
module-program compile and bind descriptor ABI policy lives in
`src/ts/adapters/node_module_program_ops.ts`, checked model path/safetensors
load-probe and supported-checkpoint catalog policy lives in
`src/ts/adapters/node_model_source_ops.ts`,
checked native inspection/compile descriptor/runtime-profile ABI calls live in
`src/ts/adapters/node_inspection_ops.ts`,
checked native symbol binding lives
in `src/ts/adapters/node_symbols.ts`, and the obsolete `js/node.cjs`
compatibility shim has been removed. Public TypeScript metadata resolves
through the `tsdown`-emitted `dist/public_api.d.cts` contract from
`src/ts/public_api.ts`, not through root declaration shims or a second
manually mirrored API contract. The Bun
bridge follows the same contract shape while loading the TS-owned
`src/ts/adapters/bun_ffi_runtime.ts` adapter without a legacy `js/bun.ts`
runtime shim. Bun LLaMA token-step/execute/selection ABI policy lives in
`src/ts/adapters/bun_llama_token_ops.ts`, matching the Node-side split while
keeping Bun-specific FFI packing out of the concrete wrapper. Bun Program
buffer/device ABI policy lives in `src/ts/adapters/bun_program_buffer_ops.ts`,
so Program-owned host/device buffers share the same TS-owned adapter boundary.
`src/ts/adapters/bun_module_program_ops.ts` owns Bun module Program compile ABI
policy, so traced-module descriptor packing stays out of the concrete FFI
wrapper.
`src/ts/adapters/bun_program_bind_ops.ts` owns Bun generic Program bind ABI
policy, so host/native session binding and owned-buffer cleanup stay out of the
concrete FFI wrapper.
`src/ts/adapters/bun_llama_session_bind_ops.ts` owns Bun LLaMA KV-cache/session
bind ABI packing, keeping persistent LLaMA state binding out of the concrete FFI
wrapper while sharing descriptor validation through
`src/ts/runtime/llama_session_bind_desc.ts`. `src/ts/runtime/llama_kv_cache.ts`
owns shared LLaMA KV-cache allocation/resource policy, with Bun adapter glue in
`src/ts/adapters/bun_llama_kv_cache_ops.ts`, keeping resource-backed cache
construction out of the concrete FFI wrapper. `src/ts/adapters/bun_native_buffer_surface.ts` owns
the Bun `NativeBuffer` public surface, keeping buffer methods out of the
	concrete FFI wrapper. `src/ts/adapters/bun_generic_family_surface.ts` owns Bun
	native model-handle adapter glue over the shared
	`src/ts/runtime/generic_family_surface.ts`, keeping generic handle classes out
	of the concrete FFI wrapper and out of host-specific duplicate copies.
	`src/ts/adapters/bun_model_source_ops.ts` owns Bun model path,
safetensors load/probe, and supported-checkpoint catalog ABI policy.
Node can still load native-free
`dist/bun.cjs` for package-spine checks. The
wrapper uses `koffi` to call the same opaque C handles and currently proves tiny
linear, fixed tiny LLaMA, fixed SmolLM-135M, and optional compatible-checkpoint
LLaMA model/program/session lifecycles:

```sh
npm --prefix examples/node_ffi install
zig build ffi-node-smoke
```

The compiled-in supported-checkpoint catalog now advertises the exact tiny
LLaMA smoke envelope plus SmolLM-135M. Node, Bun, Node/WASI, and browser Wasm
smokes prove both catalog rows and both safetensors-header probes. The C ABI
tests, Node/Bun smokes, and Node/WASI Wasm smoke also generate a complete
all-zero exact tiny LLaMA safetensors checkpoint, load it through the path
selectors, compile it, bind a CPU session, and step it to prove the tiny
catalog row is an executable load path rather than metadata-only. The C ABI,
Node/WASI, and browser Wasm smokes also pass the same bytes through
`zgml_model_probe_safetensors_data` and
`zgml_model_load_safetensors_data`, proving the pathless byte-probe ABI can
preflight the same envelope before the byte-load ABI creates and executes the
same model handle. The JS smokes cover root
`probeModel(bytes)` and fixed-family `TinyLlama.probe(bytes)` metadata-only
preflight over those same safetensors bytes, then cover
`TinyLlama.load(path)`, `loadModel(path)`,
`loadModel(path, { kind: "tiny-llama" })`, root `loadModel(bytes)`, root
`loadSafetensorsData(bytes)`, and fixed-family
`TinyLlama.loadSafetensorsData(bytes)` helpers. When
`ZGML_SMOLLM_MODEL`, `ZGML_SMOLLM_GGUF`, or `ZGML_SMOLLM_SAFETENSORS` points at
a local compatible checkpoint, the same Node smoke first runs
`probeModel(path)` / `SmolLM135M.probe(path)` over `zgml_model_probe_path`, then
runs root `loadModel(path)` through the `ZGML_MODEL_AUTO` selector and
`loadModel(path, { kind: "smollm-135m" })` through the fixed compiled-in family
selector. The `TinyLlama.load(path)`, `Llama.load(path)`, and
`SmolLM135M.load(path)` helpers remain aliases over the same native selectors.
For `.safetensors` paths, the Node and Bun wrappers now read only the
safetensors length prefix plus JSON header and route through
`probeSafetensorsHeader(...)`, so `probeModel(path)` stays metadata-only instead
of reading checkpoint tensor bytes.

The high-level shape is:

```ts
import { TinyLinear, runtimeInfo } from "zgml/bun";

const runtime = runtimeInfo();
const model = TinyLinear.create({ inputLen: 2, outputLen: 3 });
const program = model.compile({ backend: "cpu" });
const inspection = program.inspect();
const capabilities = program.capabilities();
if (!capabilities.canExecute) throw new Error("program is inspection-only");
const session = program.bind({
  weights: new Float32Array([1, 2, 3, 4, 5, 6]),
  bias: new Float32Array([0.5, -0.5, 1.0]),
});

const output = session.step(new Float32Array([1, 2]));
const profile = session.runtimeProfile();
session.resetRuntimeProfile();

session.free();
program.free();
model.free();
```

If callers reuse stable edge buffers up front, the JS call shape stays thin:

```ts
const weights = program.createBuffer("weights");
const bias = program.createBuffer("bias");
const input = program.createBuffer("input");
const output = program.createBuffer("output");
weights.writeFloat32([1, 2, 3, 4, 5, 6]);
bias.writeFloat32([0.5, -0.5, 1.0]);
input.writeFloat32([1, 2]);
const session = program.bind({
  weights,
  bias,
  input,
  output,
});

session.step(); // wrapper reuses input/output edge buffers
input.writeFloat32([3, 4]);
session.step(); // same Program/Session handle, new input contents
```

For tiny tensor programs, Node and Bun expose the generic
`program.createBuffer(kind, options)` factory for `"weights"`, `"bias"`,
`"input"`, and `"output"`. The role-specific aliases
`program.createWeightsBuffer()`, `program.createBiasBuffer()`, and
`program.createInputBuffer()` remain available. Host defaults call
`zgml_program_create_buffer` so JS does not duplicate byte math; like
`createOutputBuffer`, each factory also accepts `{ resource }` to build an
opaque external-resource view from the compiled Program's requirement record.

Program objects also expose a compact capability view over
`zgml_program_inspect`: `program.capabilities()`, `program.canExecute()`,
`program.canBindExternalResources()`, `program.hasFullDispatchPlan()`, and
`program.executionMode()`. The capability object carries
`mode: "executable" | "resource-probe" | "compile-only"` plus the decoded
dispatch-plan detail fields:
backend dispatch count, covered op count, first unsupported op, and
dispatch-family counts. These helpers are intentionally evidence-derived
rather than backend-name derived. A WebGPU-shaped resource probe can report
`canBindExternalResources() === true` while `canExecute() === false`; callers
should treat that as an inspectable binding/resource-layout state, not as an
executable inference path.

At the raw C boundary, callers can pass `zgml_session_step(session, NULL,
&result)` when all step I/O is already bound and no per-call input/output
descriptor is needed. The Node, Bun, and Wasm lanes exercise that shape through
native `zgml_buffer` handles. Typed arrays are still convenient per-call edge
buffers, but persistent foreign storage should use native `zgml_buffer` /
`NativeBuffer` handles through `zgml_session_bind_buffers` when the caller wants
zgml-owned lifetime instead of a long-lived JS pointer contract.
For output-free execution, `zgml_session_step_no_output(session, NULL, &result)`
keeps the same bound input table but skips host output download.

The same wrapper also exposes the fixed tiny LLaMA smoke:

```ts
import { TinyLlama } from "zgml/bun";

const model = TinyLlama.create();
const program = model.compile({ backend: "cpu" });
const inspection = program.inspect();
const executable = program.inspectExecutable();
const prefillSession = program.bind();
const promptLogits = prefillSession.prefill([0, 1], new Float32Array(program.vocabSize));
const logitsBuffer = program.createOutputBuffer();
const kvCache = program.createKvCache();
const decodeSession = program.bind({ output: logitsBuffer, kvCache });
decodeSession.advanceTokens([0, 1]);
const next = decodeSession.stepArgmax(2); // logits stay in the native buffer

const compatible = TinyLlama.create();
const rebound = program.bind({ model: compatible, output: logitsBuffer });
```

Node and Bun `session.step(token)` use the scalar `zgml_session_step_token`
entry point; token-window calls such as `prefill(...)` and `advanceTokens(...)`
use `zgml_session_execute_tokens`.

`program.createKvCache()` is a small Node/Bun convenience over
`program.kvCacheRequirements()`: by default it allocates correctly sized
per-layer host `NativeBuffer` K/V storage and returns the same `{ k, v }` shape
accepted by `program.bind({ kvCache })`. With executable WebGPU Programs,
`program.createKvCache({ placement: "webgpu" })` allocates one Program-owned
same-device K/V buffer per layer. It can also build resource-backed views from
the same executable Program sizing contract:

```ts
const resourceKv = program.createKvCache({
  resource: ({ byteLength, kind, layer }) => NativeBuffer.externalResource({
    placement: "webgpu",
    handle: gpuBufferTable.lookup(`${kind}:${layer}`),
    byteLength,
    access: "readwrite",
  }),
});

const resourceLogits = program.createOutputBuffer({
  resource: ({ byteLength }) => NativeBuffer.externalResource({
    placement: "webgpu",
    handle: gpuBufferTable.lookup("logits"),
    byteLength,
    access: "write",
  }),
});

const logitsInfo = resourceLogits.inspect();
// { storage: "external-resource", placement: "webgpu", access: { write: true }, ... }
```

`NativeBuffer.externalResource(...)` can also be passed as a LLaMA-family
output buffer through `program.createOutputBuffer({ resource })` or as K/V
storage through `program.createKvCache({ resource })`. Today the Node and Bun
smokes assert that CPU/host-only binding returns native `unsupported` for both
normal and compatible-model sessions, but the descriptors now reach the
executable bind path instead of being treated as host pointers.

And the first real checkpoint-backed lane:

```ts
import { SmolLM135M } from "zgml/bun";

const model = SmolLM135M.load("data/smollm/SmolLM-135M.Q8_0.gguf");
const program = model.compile({ backend: "cpu" });
const inspection = program.inspect();
const executable = program.inspectExecutable();
const prefillSession = program.bind();
const promptLogits = prefillSession.prefill([0, 1], new Float32Array(program.vocabSize));
const decodeSession = program.bind();
decodeSession.advanceTokens([0, 1]);
const logits = decodeSession.step(2, new Float32Array(program.vocabSize));
```

Node and Bun also accept `{ backend: "webgpu" }` for tiny-linear, supported
traced module graphs, direct tiny-MLP handles, and fixed tiny LLaMA inspection.
Dynamic-library builds without
native wgpu expose compile-only shape evidence with `backend: "webgpu"` and
backend-reported `executionSupported: false`. Tiny-linear reports
`externalResourcesSupported: true` for an opaque resource-session probe:
`program.bind({ weights, bias, input, output })` can bind
`NativeBuffer.externalResource(...)` handles and `session.inspect()` will report
the WebGPU-shaped resource bindings, while execution still returns
`unsupported`. Direct tiny-MLP Programs report `externalResourcesSupported:
false` until a real same-device binding lane exists; traced module Programs
report the generic resource-bind shape through Program inspection and
capabilities. The C, Node, Bun, Node/WASI, and browser Wasm smokes also prove the
unsupported resource-probe `step` and no-output advance
leave caller logits
untouched, keep Session position unchanged, and leave the bound Session runtime
profile at zero calls, backend ops, fallback ops, syncs, and runtime patch
attempts. Native libraries built with `-Duse-wgpu=true` instead make
tiny-linear, direct tiny-MLP, traced module host-bound, and supported fixed tiny
LLaMA WebGPU bindings executable and report `executionSupported: true`; passing
`-Dexperimental-llama-wgpu-execution=false` keeps the LLaMA lane on the
resource-probe path. The native wgpu backend now also has a registered
resource-table execution proof for buffers owned by the same backend/device.
Fake or wrong-device opaque resource handles still reject before Session
binding. The public C ABI exposes the owned-device path as
`zgml_program_create_device_buffer(program, kind, ZGML_BACKEND_WEBGPU, ...)`;
Node and Bun expose the same path as
`program.createBuffer(kind, { placement: "webgpu" })` for weights, bias, input,
and output; the role-specific buffer helpers remain aliases. In that executable native build,
C, Node, and Bun smokes prove those device buffers bind as external-resource
Session state, execute through the WebGPU command shape, and can advance
without a Session output readback. The C ABI also exposes
`zgml_program_get_device_handle(...)` plus
`zgml_program_import_device_buffer(...)`; Node and Bun expose those as
`program.deviceHandle("webgpu")` and
`program.importDeviceBuffer(kind, { deviceHandle, bufferHandle })`, with
optional `byteLength` for explicit view-size validation. The preferred
Node/Bun shape is now `const device = program.device("webgpu")`, whose
`device.handle` carries the program's same-device token and whose
`device.createBuffer(kind)` / `device.importBuffer(kind, source)` methods route
through the same ABI. The same device object also exposes role-specific helpers
like `createWeightsBuffer()`, `createInputBuffer()`, `createOutputBuffer()`, and
`createKvCache()`, so LLaMA callers can allocate logits and per-layer K/V
storage without leaving the same-device provenance object. When the source is a
zgml-created device `NativeBuffer`,
Node and Bun can also call either `device.importBuffer(kind, nativeBuffer)` or
`program.importDeviceBuffer(kind, nativeBuffer)` and let the wrapper inspect the
buffer object for the raw handle, placement, offset, and byte length across
weights, bias, input, output, and K/V roles. Native Node/Bun integrations can also
pass GPUBuffer-like host objects through the exported `webgpuInterop` symbols:
`webgpuInterop.bufferHandle`, `byteLength`, `byteOffset`, `placement`, and
`deviceHandle`, or `webgpuInterop.importSource` when the descriptor should be
produced lazily. `ProgramDevice.importBuffer(...)` supplies the same-device
token by default; direct `program.importDeviceBuffer(...)` still requires an
explicit device token or symbol.
C callers or native JS integrations that create valid same-device `WGPUBuffer`
objects can import them after zgml validates the program device token, byte
range, storage-buffer binding alignment, and usage flags. Passing C
`byte_len = 0`, or omitting JS `byteLength`, uses the compiled Program's
required byte length for that buffer kind. The Node and Bun smokes also prove
wrapper-level fake external-resource object imports reject, wrong-device,
unaligned, and oversized raw imports reject, and imported same-device buffers
can bind and execute by importing buffers created by zgml, even after the
original source `NativeBuffer` handles are freed. Browser `GPUBuffer` objects do
not expose raw `WGPUBuffer` handles, so browser interop still needs the separate
Wasm/WebGPU shared-device runtime rather than this native FFI descriptor shape.
The portable Node/WASI and browser smokes do exercise the front half of that
browser contract through `examples/wasm_ffi/host_resources.mjs`: a shared
`WasmWebGpuDevice` wrapper creates GPUBuffer-like JS resources, gives them
stable opaque handles, byte ranges, placement, and access flags, and a shared
`WasmExternalResourceBridge` writes the ABI v6 external-resource descriptor into
linear memory before calling `zgml_buffer_wrap_resource`. A shared
`WasmSessionBindingBridge` then writes the resource-backed tiny-linear and LLaMA
Session bind descriptors before calling the same bind exports used by C hosts.
For LLaMA-family resource probes, `WasmWebGpuLlamaResourceProgram` now gives the
portable Wasm path the same Program-owned slot shape as Node/Bun: it derives
logits and K/V cache byte sizes from compiled Program metadata, creates
Program-sized output/KV resources, and accepts either raw browser resources or
already-wrapped resource bindings before writing the bind descriptors.
The tiny-linear portable host Program follows the same convention for browser
execution: role-specific helpers create wrapped weights/bias/input/output slots,
and `bind(...)` can consume those slots or raw browser resources before calling
the ABI bind exports.
The browser smoke prefers actual browser
`GPUBuffer` objects when `navigator.gpu` is available and records whether it
used `GPUBuffer` or mock resources in
`document.documentElement.dataset.zgmlWasmGpuResources`, while execution remains
compile-only/unsupported with zero runtime work.
These smokes also prove host `NativeBuffer` weights are persistent
Session state: mutating the host weight buffer after bind does not affect later
WebGPU steps, while mutating the bound input buffer does. Default fixed tiny
LLaMA still reports `externalResourcesSupported: true` for a resource-session
probe: `program.bind({ output, kvCache })` can bind WebGPU-labeled
external-resource logits and K/V cache buffers, `session.inspect()` reports
external-resource output and K/V storage, and token execution returns
`unsupported` without advancing position, mutating caller logits, or recording
Session runtime work. C, Node, Bun, Node/WASI, and browser Wasm also reject
invalid public resource descriptors before a resource Session is returned:
read-only logits, write-only K/V caches, and read-only K/V caches all report
unsupported.

With `-Duse-wgpu=true`, the public fixed tiny LLaMA lane now exposes the
same-device resource executor through the normal handle API by default. C,
Node, and Bun allocate Program-owned WebGPU logits plus
per-layer K/V buffers (`ZGML_PROGRAM_BUFFER_LLAMA_K_CACHE` /
`ZGML_PROGRAM_BUFFER_LLAMA_V_CACHE`, or JS `"kv-k"` / `"kv-v"`), bind them as
Session output/cache resources, execute decode, prefill, and
decode-after-prefill, then read logits back through `zgml_buffer_read` or
`NativeBuffer.readFloat32` / `NativeBuffer.readFloat32Into` and compare against
CPU. Node and Bun also expose the same generic C buffer I/O as
`NativeBuffer.fromBytes`, `writeBytes`, `readBytes`, and `readBytesInto`, so
non-f32 FFI buffers can use caller-owned byte storage without tensor views.
Single-token executable
decode also checks the hot Session backend dispatch count against the compiled
Program dispatch count for both normal and same-device resource-bound Sessions,
so the public FFI lane proves exact dispatch-plan execution rather than only
nonzero backend work. The optional `zig build wgpu-check -Duse-wgpu=true`
target runs the native WebGPU link, generic executor, and public LLaMA WebGPU
smokes together. The broader LLaMA-family
WebGPU claim remains gated until dispatch-plan and shared-resource coverage
widens across the supported public surface; browser `GPUBuffer` execution is
also still future work.

And the first compatible-checkpoint lane, backed by `ZGML_MODEL_AUTO`:

```ts
import { loadModel, probeModel, probeSafetensorsData, probeSafetensorsHeader, supportedCheckpointModels } from "zgml/bun";

const supported = supportedCheckpointModels();
if (!supported.some((model) => model.modelKind === "tiny-llama")) {
  throw new Error("this zgml build cannot load exact tiny LLaMA checkpoints");
}
if (!supported.some((model) => model.modelKind === "smollm-135m")) {
  throw new Error("this zgml build cannot load SmolLM checkpoints");
}
const probe = probeModel("data/smollm/SmolLM-135M.Q8_0.gguf");
if (probe.modelKind !== "smollm-135m") throw new Error("unsupported checkpoint");
const headerProbe = probeSafetensorsHeader(safetensorsHeaderJson);
if (headerProbe.modelKind !== "smollm-135m") throw new Error("unsupported safetensors header");
const byteProbe = probeSafetensorsData(safetensorsBytes);
if (byteProbe.modelKind !== "tiny-llama") throw new Error("unsupported safetensors bytes");
const model = loadModel("data/smollm/SmolLM-135M.Q8_0.gguf");
const modelInfo = model.inspect();
const program = model.compile();
const inspection = program.inspect();
const session = program.bind();
const logits = session.step(0, new Float32Array(session.vocabSize));
```

The Bun wrapper also implements `dispose()` and `Symbol.dispose` as aliases for
`free()`, and LLaMA-family programs/sessions expose `vocabSize` from native
program inspection instead of requiring callers to hard-code logits sizes. In
both host packages, fixed tiny LLaMA, generic LLaMA, and fixed SmolLM-135M now
share one LLaMA-family wrapper implementation, so fixed checkpoints are typed
load/bind specializations over the same Program/Session methods rather than
separate copied lifecycles. The Node wrapper uses the same handle lifecycle and
deterministic cleanup for tiny linear, fixed tiny LLaMA, fixed SmolLM-135M, and
generic compatible-checkpoint `probeModel(path)`, `probeModel(bytes)`,
`loadModel(path)`, `loadModel(bytes)`, and `Llama.load(path)`. Both wrappers validate
compile-envelope numbers before FFI:
`contextLength` and `batch` must be non-negative safe integers, while
valid-but-unsupported envelopes such as LLaMA `batch: 2` still report native
`unsupported`. The Node wrapper is intentionally smaller than the Bun wrapper
while the ABI surface stabilizes. A production Node package should likely move
from FFI to a Node-API addon once the ABI and packaging story are stable:

```ts
const model = await zgml.loadModel("model.gguf");
const program = model.compile({ backend: "metal", contextLength: 4096 });
const logitsBuffer = zgml.NativeBuffer.create(program.vocabSize * 4);
const session = program.bind({ output: logitsBuffer });
const next = session.stepArgmax(token).token;
session.readOutputInto(new Float32Array(program.vocabSize));
const generatedTokens = new Uint32Array(128);
const generated = session.generateTokensArgmaxInto(promptTokens, generatedTokens).tokens;
```

For LLaMA-family sessions where the caller only wants native token selection or
generation, Node and Bun also accept `program.bind({ output: "native" })`. The
wrapper creates a Program-sized native logits buffer, binds it once, and frees
that wrapper-owned buffer when the Session is freed. `session.readOutputInto(...)`
copies those native-bound logits into caller-owned `Float32Array` storage when a
host-side logits view is actually needed. Passing an explicit `NativeBuffer`
keeps ownership with the caller.

In that shape, Node/Bun owns object lifetimes and typed-array edge views, while
zgml owns the compiled program, native session buffers, runtime patch table, and
backend execution. Greedy generation can avoid copying a full logits vector back
to JS on every token, and the `generateTokens*Into(...)` methods let hot loops
reuse caller-owned token output arrays.

Bun can still load the C functions directly:

```ts
import { dlopen, FFIType, suffix } from "bun:ffi";

const lib = dlopen(`./zig-out/lib/libzgml_c.${suffix}`, {
  zgml_model_create: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },
  zgml_model_load_path: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },
  zgml_model_probe_path: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },
  zgml_model_inspect: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },
  zgml_program_check_model_compatibility: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },
  zgml_program_compile: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },
  zgml_program_inspect: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },
  zgml_llama_program_inspect: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },
  zgml_session_bind: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },
  zgml_session_upload_persistent: {
    args: [FFIType.ptr],
    returns: FFIType.i32,
  },
  zgml_session_upload_persistent_range: {
    args: [FFIType.ptr, FFIType.u64, FFIType.u64],
    returns: FFIType.i32,
  },
  zgml_session_step: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },
  zgml_session_step_token: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },
  zgml_session_advance_token: {
    args: [FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },
  zgml_session_prefill_tokens: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.i32,
  },
});
```
