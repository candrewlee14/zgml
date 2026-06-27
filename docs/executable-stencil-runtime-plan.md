# TS-First Executable Runtime Plan

This is the north-star plan for turning zgml into a tiny TS-first ML library
whose differentiator is an inspectable native executable runtime.

The thesis is:

```text
compile once
bind persistent state
update small runtime parameters
execute at native speed
prove the shape with evidence
```

zgml should not be only a tensor library that happens to have stencils. It
should be a compiled Program runtime with a clean tensor, neural-network, and
LLM frontend on top. A stencil is the internal executable artifact produced by
that runtime, not a separate public lane.

The product should feel like a brilliant, small PyTorch-like replacement for
JS/TS. TypeScript is the authored library. `tsdown` emits the package. Zig sits
underneath as the native runtime/kernel language. The executable Program runtime
is the differentiator, not a reason to make everyday tensor, neural-network,
optimizer, loss, or training code feel alien.

The important correction is this: there is no TS/Zig frontend synchronization
problem to solve. That would be the wrong architecture. Write the library in
TypeScript, emit it with `tsdown`, and keep Zig focused on the implementation
substrate: kernels, buffers, executable Programs, Sessions, backend hooks, and
ABI-stable native handles. Cross-language alignment should happen at runtime
contracts and evidence, not by re-authoring the product API twice.
The executable contract now names that explicitly: `productSourceOfTruth` is
`ts-api-zig-core`, and native product policy is `required-core`.

Current goal, stated without the old sync trap:

```text
write the product library once in TypeScript
emit Node, Bun, browser, types, and subpaths with tsdown
use Zig as the native tensor-storage/runtime/kernel/ABI substrate
surface native speed through Program -> Session -> StepParams
never make JS/TS features wait on a mirrored Zig frontend
```

The strongest form of the rule is:

```text
do not synchronize frontends
author the frontend once
make the boundary smaller
make the boundary executable
```

If a future design needs a "sync pass" to keep TS and Zig behavior coherent, it
is probably the wrong design. The fix is not a better sync system; the fix is to
move product semantics into TS, lower hot work through Program/Session, and
reduce the Zig contract to stable handles, buffers, kernels, ABI records,
inspection evidence, and backend execution.

This means the product architecture is not "keep TS and Zig APIs in sync." The
product architecture is "author the JS/TS library once, then compile or bind the
hot parts to native execution." Files such as `src/nn.zig`, `src/loss.zig`,
`src/optim.zig`, and `src/train.zig` may continue to exist where they are useful
as native tests, model internals, C ABI substrate, or low-level Zig examples, but
they are not a second product frontend. They should not define the JS/TS package
roadmap, and new PyTorch-like API surface should not be blocked on adding a
matching Zig API. If a native helper duplicates TS product policy, the desired
direction is to either move that policy into `src/ts/**` or lower the TS feature
through a stable Program/Session/ABI contract.
Native helper modules can keep substrate conveniences, fixtures, and ABI
evidence, but they are not allowed to become product-policy owners. Their
manifest role is `native-helper-substrate`, with `product_policy = "required-core"`.

The Zig root module exposes `native_substrate_manifest` for the same reason the
TS package exposes `frontendManifest`: it makes the boundary executable. Zig's
role is `core-kernel-runtime`, the product language is TypeScript, the
product source of truth is `ts-api-zig-core`, the product semantics owner is
`src/ts/** + src/**/*.zig`, the package fan-out is `tsdown`, native product policy is
`required-core`, and native alignment is `zig-core-contract-tested`:
JS/TS API -> Zig C ABI -> Program/Session kernels, runtime evidence, and backend behavior tests.
Handwritten frontend mirrors are not allowed.
The same boundary now has a static numeric drift guard: `npm run
check:ts-source-architecture` compares the TS ABI descriptor maps in
`src/ts/runtime/abi.ts` with the native constants in `src/c_api.zig` for model
kinds, backend ids, buffer/storage/access ids, Program buffer kinds, ABI struct
kinds, module op ids, activation ids, module flags, runtime feature bits, and
the C ABI version. That keeps `KernelPlan -> native descriptor -> Program`
lowering from becoming a parallel, silently divergent compiler contract.

The short rule is:

```text
TS is the product language.
tsdown is the artifact fan-out.
Zig is the native execution substrate.
Contracts and tests keep the boundary honest.
```

Read "native execution substrate" strongly: Zig is also the long-term tensor
storage and single-op execution substrate for tensor-sized eager work.

This is closer to PyTorch and NumPy than to a JS library with optional native
plugins. Python does not own PyTorch tensor execution; it owns the user-facing
semantics and calls into ATen/CUDA. Python does not own NumPy array math; it owns
the ergonomic shell over native ndarray storage and kernels. zgml should follow
that split for JS/TS: TypeScript owns the package API, type-level safety,
autograd policy, module composition, optimizer/training ergonomics, and evidence
records, while Zig owns tensor storage, executable programs, kernels, backend
dispatch, ABI handles, and eventually the default large-tensor eager execution
path. "TS-first" means authored and typed in TS, not that substantial tensor
math should run in JS by default forever.

Architecture decision: the product library is written in TypeScript, then
emitted with `tsdown`. We do not keep a Zig frontend and TS frontend in sync.
There is one JS/TS product source tree: `src/ts/**`. `tsdown` fans it out into
runnable package artifacts, declarations, source maps, and subpaths. Zig is the
native runtime, ABI, and kernel substrate; it may expose idiomatic low-level
native handles, but it is not a mirrored product frontend. When a TS feature
needs speed, it lowers to or binds a native Program/Session; when it is
frontend policy, it stays in TS.

## Why This Exists

PyTorch wins if the goal is "all of PyTorch, but in another language." ZML wins
if the goal is a Zig frontend over a broad OpenXLA/PJRT accelerator stack.
llama.cpp wins if the goal is a mature local LLM app/runtime today.

zgml's credible lane is smaller and sharper:

```text
a tiny TS-first ML library with a PyTorch-simple frontend
and an inspectable native executable-program runtime underneath,
where programs are bindable, updatable, fast, and easy to expose
through C, Node, Bun, Wasm, and browser bindings.
```

The long-term identity is:

```text
PyTorch   = universal Python ML ecosystem
tinygrad  = small Python tensor/JIT/autograd framework
ggml      = tiny native tensor/runtime substrate
llama.cpp = mature local LLM runtime/app stack
ZML       = Zig + MLIR/OpenXLA/PJRT production compiler stack
zgml      = tiny native executable-stencil runtime with PyTorch-like ergonomics
```

## Product North Star

zgml should feel like one library, not a runtime plus bindings. The frontend is
the primary interface; the executable runtime is what makes that interface
portable, inspectable, and fast.

The ideal user story is:

```text
write ordinary model code
train or debug it eagerly
ask whether it can compile
compile the stable work
bind model state
run hot loops through a tiny executable handle
```

The first-contact API should be the tiny executable handle:

```ts
const fast = zgml.native(model, { inputShape: [2] as const });
const y = fast.forward(input);
const alsoY = fast.call(input);
const out = fast.into(new Float32Array(2), input);
const proof = fast.explain();
fast.dispose();
```

That handle is intentionally not a second runtime abstraction. It owns a
`Program` and bound `Session`, exposes them for evidence and advanced control,
and gives ordinary inference users the short path they actually want.
The handle now also exposes first-contact proof methods directly:
`explain()`, `preflight()`, `compileSupport()`, `inputShape()`,
`outputShape()`, `kernelPlan()`, and `compilerSignatures()`. That keeps the
happy path tiny while still making the executable artifact inspectable without
forcing users to know the lower-level `Program` API on day one.
The same handle is now deliberately module-shaped: it exposes `forward`,
`call`, and `__call__`, so native inference feels like ordinary `nn` code while
still owning a real `Program` and `Session` underneath. The `nn` namespace also
provides `nn.native(model, opts)` / `nn.inference(model, opts)`, and module
instances expose `model.native(opts)`, all as TS-authored sugar over the same
native Program/Session binding path. That is the desired split: ergonomic JS/TS
API, Zig-owned execution core, no mirrored Zig product frontend.

The refined compiler shape is:

```text
Tensor/nn frontend
  -> lazy typed tensor/UOp IR
  -> scheduler/kernelizer
  -> KernelPlan
  -> ProgramStencil
  -> Program
  -> Session
  -> execute/step with evidence
```

This is the tinygrad lesson folded into zgml's native runtime shape. Ordinary
model code should create a lazy graph that can be scheduled into a small number
of kernels. The Program/Session layer then turns that scheduled work into an
explicit, bindable, inspectable artifact for Zig, C, Node, Bun, Wasm, and
browser hosts.

The "PyTorch-like replacement" goal is scoped but serious:

- JS/TS users should get the model vocabulary from Node, Bun, browser, and FFI
  hosts: typed tensors, modules, optimizers, training helpers, state snapshots,
  host-side eager execution for small work, and native Program/Session handles
  for the paths zgml can honestly compile.
- Zig users should get the runtime substrate and carefully chosen native APIs:
  buffers, kernels, Programs, Sessions, C ABI, backend hooks, and low-level
  execution handles. They should not get a second product frontend that must be
  kept synchronized with TS.
- FFI users should not have to understand tensor internals. They should pass
  typed arrays or opaque native buffers into compiled Programs and keep hot
  loops allocation-free.
- Runtime users should be able to prove which backend path ran, which model
  shape was compiled, what state was bound, and why an unsupported graph did not
  silently fall back.

The "better than PyTorch" goal is not to clone Python dynamism. It is to keep
the eager experience pleasant while making stable model code more knowable:
Zig APIs should move toward `comptime` shape/layout/module contracts where the
caller has static information, TS APIs should preserve literal shape evidence
and branded tensor/module state where that remains ergonomic, and compiled
Programs should provide runtime evidence for everything that could not be
proven statically. Public Tensor shape preflights (`hasShape` /
`requireShape`, plus `Tensor.hasShape` / `Tensor.requireShape`) are part of
that contract: callers can narrow literal Tensor shapes or fail before compile,
bind, or FFI handoff. Eager execution is the reference/debug mode; compiled
execution is the performance claim. Silent eager fallback is not allowed.

Current checked progress:

- Program/Session performance substrate: ~85%. The Program/Session shape,
  runtime patching, C/Node/Bun/Wasm handles, portable LLaMA profile coverage,
  native Metal execution, a scorecard-run optional native wgpu validation gate,
  and ggml benchmark gates are real enough that the substrate is past "architecture".
  This is not a headline throughput-complete score: accepted SmolLM evidence is
  still materially behind llama.cpp/ggml, and the semantic Q8 target remains a
  diagnostic until it grows real tile-parallel throughput.
  Compile-capable lazy graphs can now lower through the host adapter into a
  native Program with preserved KernelPlan evidence. Lazy
  Linear+GELU and
  `matmul -> add(bias) -> activation` now collapse to one native Program
  dispatch and are benchmarked for both ReLU and GELU while preserving the
  original three-op Tensor IR evidence. Parameterless activation chains also
  have decision-grade one-dispatch evidence for arithmetic
  `relu -> square -> sqrt`, sign-style `neg -> abs -> step` chains, and
  nonlinear `tanh -> sigmoid -> tanh` chains.
  LayerNorm/RMSNorm descriptors can carry post-affine activations,
  and the norm-GELU MLP module benchmarks now prove
  `Linear -> LayerNorm(+affine)+GELU -> Linear` as a three-dispatch Program
  path for scalar and batched inputs. Full-model F16 SmolLM FFN blocks now
  form dense projection-pair commands for `matmul -> silu -> matmul -> mul`,
  with Metal unary sidecar lowering and focused ReleaseFast evidence at
  prompt/decode command pressure `242/212`, `30` dense FFN pair commands, and
  zero fallback ops. A batched `RMSNorm+GELU -> Linear`
  Program benchmark now proves the RMSNorm side of that descriptor path on a
  profitable workload. Common classifier/token-head `LogSoftmax` tails now
  lower to a native row op on reference CPU, Metal, and WGPU instead of
  replaying ten composite Tensor Program IR sub-ops, while an explicit
  capability-disabled test still proves the composite fallback. Module
  `min(dim)` now lowers through a first-class `reduceMin` descriptor and a
  direct backend reduce-min command on reference CPU, Metal, and WGPU instead
  of exposing or executing its old `neg -> max -> neg` decomposition.
  batched rank-3 last-axis reductions now also compile and execute
  as native Program kernels for `sum`/`mean`/`prod`/`max`/`min`/`argmax`/`argmin`,
  so `[B, T, C]` reductions over the feature axis stay on the native Program
  path instead of falling back through compatibility lanes. The
  optional native WebGPU validation gate now also runs LLaMA execution proofs
  for runtime quantized-weight rebinding, resource-bound decode/prefill
  handoff, long-prompt prefill, GQA long-prompt prefill, and realistic
  head-width resource handoff instead of merely compiling those paths. The
  frontier benchmark gate now also includes the real SmolLM prompt geometry
  (`m=128 n=576 k=576`) for projection-chain, grouped projection-chain,
  grouped projection-row-chain, and single projection-row-chain paths, with
  command-shape evidence proving the row-chain lowering covers the intended
  five-op chains. The full-model Q8 prompt gate now separates the usable
  command-fusion path from the slower single-dispatch experiment: the
  two-dispatch projection-row-chain command path keeps the fast tiled
  qmatmul column parallelism while reducing command shape from `241->181`,
  whereas the single-dispatch candidate remains a diagnostic for the needed
  tiled row-chain kernel. The frontier benchmark now exposes that
  single-dispatch tiled candidate directly as
  `projection_row_chain_single_dispatch` prompt/full-prefill/SmolLM-prompt
  lanes, with correctness, command-shape, and runtime-dispatch evidence. Current
  evidence keeps it diagnostic: it can help smaller prompt tiles, but
  full-prefill and SmolLM-prompt shapes remain effectively neutral or slower
  than the split command path. A focused Metal tail-parallelization experiment
  confirmed that the single-dispatch trap is the loss of MxN tile parallelism:
  row-tile threadgroups must loop over N tiles without cross-threadgroup row
  reduction. The q8 prompt gate now records this as
  `single_dispatch_trap=serial_n_tile_loop_without_cross_threadgroup_row_reduce`,
  and the benchmark JSON carries `qmatmul_row_chain_tiled_*` counters so the
  gate prints actual row groups, N tiles, serial tile loops, and elementwise
  spills for the candidate. It also reports the ABI-preserving scratch
  precondition for the two-phase design: the fused `rmsnorm.dst` buffer can hold
  the per-row/per-N-tile partial sums when `two_phase_scratch=ready`. The gate
  now makes the real next target explicit:
  `row_chain_next=semantic_sublayer_or_two_phase_tile_parallel_row_chain`.
  A guarded Metal prototype now exists behind
  `fuse_projection_row_chain_two_phase_candidate`: it dispatches a tiled
  qmatmul/elementwise partial-sum kernel across `(row_tile, n_tile)` and then a
  finalize kernel that reduces those partials into the RMS scale. This preserves
  the MxN tile parallelism that the one-dispatch experiment lost while keeping
  the old split command path as the default until full-model ReleaseFast evidence
  says otherwise. The q8 prompt candidate gate now measures this as a fourth
  lane and prints `two_phase_count`/`two_phase_selected` so selection is proven
  rather than inferred from a flag. A June 23, 2026 one-attempt probe selected
  the two-phase path for all 60 full-model prompt row-chain commands
  (`two_phase_count=60`, `two_phase_selected=yes`) and landed at `0.86x` versus default
  while the simpler command path landed at `1.02x`. That is useful
  evidence, not a default path: the prototype fixed the old single-dispatch
  `0.20x` trap materially, but the extra partial/finalize work still loses to
  the simpler two-dispatch command lowering. The frontier gate now also has an
  x7 anchored-region microscope that selects the actual region-command two-phase
  path (`two_phase_count=7`) with zero observed diff for full-prefill and
  SmolLM-prompt shapes, while labeling it diagnostic because it still lands
  below the staged baseline in that isolated geometry.
  The frontier row-chain runtime profile now also prints
  `qmatmul_row_chain_tiled_spilled_elementwise`, so spill-free synthetic
  row-chain measurements and full-model residual-liveness measurements no
  longer get blurred together. The single-chain SmolLM-prompt microscope is a
  spill-free diagnostic for raw kernel shape, while the full-model Q8 prompt
  gate still reports `spills=30` because the residual stream remains live.
  A fresh qrow-region gate reports `qmatmul_row_chain_tiled_spilled_elementwise=0`
  for both x7 full-prefill and SmolLM-prompt rows, but the spill-free two-phase
  region still lands at `0.80x` and `0.61x` respectively, so spill removal alone
  is not the missing default-performance move.
  That split keeps the next move honest: a larger semantic command has to
  either consume that residual use inside the command or beat the staged path
  while materializing it.
  The single-dispatch tiled kernel remains a diagnostic for the dispatch-only
  trap, not the design center for the next performance pass.
  The frontier and q8 prompt candidate gates now rebuild benchmark binaries with `-Doptimize=ReleaseFast` before any no-rebuild rerun evidence.
  That sharper ReleaseFast microscope found the row-chain candidate can look
  ready in isolated frontier lanes (`projection_row_chain_candidate=ready`),
  but the full-model Q8 prompt gate still keeps `single_throughput=off`:
  command fusion holds at `0.98x` to `1.02x` while the single-dispatch candidate
  remains around `0.19x` to `0.21x`. That makes the next move clearer, not
  fuzzier: the command path is structurally right, but the remaining full-model
  throughput gap still needs a materially different throughput kernel or larger
  semantic sublayer. The gate treats structural/no-fallback evidence as the
  hard pass condition and reports command throughput readiness separately, so a
  noisy command median cannot promote or erase the measured command shape.
  The Q8 prompt gate now also prints the existing quantized
  `projection_pair_fused_elementwise_chain` counters as `projection_pair`,
  `projection_pair_dispatch`, `semantic_pair_path`, and
  `semantic_pair_target`. The first version of that gate accidentally omitted
  `--metal-decode-region` and exposed source drift from the accepted artifact:
  Q8 no-repeat FFN shapes were falling through as `projection_pair=0` and
  `projection_chain=90` because only the dense path recognized the four-op
  pair shape. The corrected gate now uses the same Metal decode-region shape as
  the accepted substrate artifact, and the source restores the quantized
  four-op pair path: the raw Q8 prompt comparison baseline has
  `projection_pair=30`, `projection_chain=60`, and `commands_per_call=241`,
  while the row-chain command path must keep `projection_pair>=30`, lower
  projection chains to `0`, and reduce command shape toward `181` before it can
  claim the full-model semantic row-chain path. The ggml comparison's Q8 prompt
  lane now uses that command path by default, so formal artifacts can prove the
  cleaner command shape without mutating the F16 lane.
  The Q8 prompt probe now separates generic projection groups from
  attention/cache projection groups: full-model Q8 reports
  `projection_group=0->0`, `projection_cache_group=30->30`, and
  `decode_projection_cache_group=30`. That means the attention/cache projection
  grouping lane is already live in the full model; the remaining Q8 prompt gap
  is the `projection_chain=60` row-chain/semantic-sublayer lane, while the
  frontier x7 qproj region remains the isolated executable proof for general
  qmatmul-plus-sidecar projection groups. The Q8 prompt gate now makes that
  distinction explicit with
  `qproj_group_full_model_target=frontier_only_not_full_model_sibling_region`
  and `qproj_frontiers=60`, so a strong qproj frontier run does not get
  mistaken for a missed full-model sibling-group scheduling opportunity.
  The next semantic-sublayer move should therefore build on the already-live
  pair-fused FFN path, or deliberately supersede it with a larger FFN sublayer
  command, before spending more time on the shallow row-chain tail.
  The useful next move is therefore not blindly promoting the shallow two-kernel variant;
  it is either a larger semantic sublayer that removes surrounding work
  or a materially different row-chain throughput kernel.
  A June 23, 2026 qmatvec/decode row-chain command probe confirmed that decode
  has the same dispatch-only trap in sharper form: forcing Q8 decode through
  projection-row-chain command shape reduced commands from `211` to `151`, but
  collapsed throughput to about `14 tok/s` versus the ordinary region decode's
  roughly `159 tok/s` in the same short smoke. That disqualifies shallow
  qmatvec row-chain command fusion as a decode default; decode needs either the
  ordinary staged projection-chain path, a larger semantic sublayer, or a
  genuinely different qmatvec throughput kernel.
  The June 22, 2026 full required-GPU browser smoke also refreshed the wider
  browser/WebGPU proof after the matrix expansion: it passed in real
  `GPUBuffer` mode with `maxStorageBuffers=8`, `storageAlignment=256`,
  `canBindBlockPipeline=true`, 74 LLaMA profile labels, 253 backend dispatches,
  249 executor dispatches, 4 device-selection dispatches, 127/127/0/0/0
  storage-mode calls, 41 output reads, 4 selection reads, 45 syncs, and zero
  fallback ops in about 27.5 minutes. The required-GPU browser runner's full
  LLaMA profile and storage-mode matrices are now scorecard-checked source
  contracts, the cheap focused browser smoke has checked dispatch-family and
  selection-read summary fields, and the goal scorecard now attempts a focused
  required-GPU browser LLaMA run for `gguf-smollm3-nope-gqa-pipeline` when
  Chrome/WebGPU is locally available. A June 23, 2026 follow-up added the
  matching required-GPU representative family lane for SmolLM3 GGUF NoPE/GQA,
  long-sliding-window Mistral, and Qwen3 Q/K-norm. On the local Chrome/WebGPU
  path it passed in real `GPUBuffer` mode with `llamaProfiles=6`,
  `llamaBackendDispatches=24`, `llamaFamilyDispatches=22/0/2`,
  `llamaStorageCalls=14/14/0/0/0`, `llamaFallbackOps=0`, and
  `llamaProfileLabels=6` in about 285 seconds. The remaining substrate jump is
  not another compatibility lane; it is a real tiled quantized row-chain
  throughput kernel, a larger semantic sublayer, or broad full/default browser
  execution beyond these bounded family proofs.
- zgml frontend replacement feel: ~85%. The TS-owned product frontend now has
  typed and runtime evidence for `Tensor`, `nn.Module`, `nn.Linear`, containers,
  `data` loaders/samplers, `loss`, `optim`, schedulers, `train`, state dicts,
  checkpoints, eager debugging, eager/autograd `einsum` with ellipsis and
  broadcast semantics plus literal-equation shape inference, compile-aware lazy
  parameter slots plus `matmul`/`mm`/parameterized `add`/`mul` lowering evidence,
  direct `affine(scale, bias)` lowering, parameterized `add`/`mul` activation-tail
  folding, and natural `mul(scale) -> add(bias)` coalescing, with optional
  activation tails, to the native feature-affine Program,
  along with a
  benchmarked allocation-free lazy Linear+GELU,
  `matmul -> add -> relu/gelu`, Conv2d+ReLU, MLP,
  reduced MLP, classifier log-softmax, classifier softmax-reduction, transformer FFN,
  normalized transformer classifier, and token-head Session paths,
  `lazyGraph.compile()` and compatibility `torch.compile.compile(lazyGraph)`
  Program construction paths through Node/Bun
  adapters, direct and natural chained lazy affine execution through one native Program dispatch,
  eval-mode `BatchNorm1d` lowering through a derived native affine Program while
  training-mode BatchNorm remains honestly stateful/eager,
  rank-3 `nn.Linear` eager/compiled parity by flattening leading dimensions
  into the native Linear row contract and restoring the output shape,
  native Program lowering for rank-2 `diagonal`, rank-1/rank-2/rank-3 PyTorch-style
  `repeat`/`tile` lowering through the native module Program ABI,
  native Program lowering for `argmax(dim)` and `argmin(dim)`,
  rank-3 `reshape`/`flatten`/`squeeze`/`unsqueeze`, rank-3 `broadcastTo`/`expand`,
  and rank-3 `narrow`/`select`/`slice`
  lowering through the native module Program ABI,
  rank-3 `transpose` plus single-swap and cycle `permute`
  lowering through the native module Program ABI,
  rank-3 last-axis `sum`/`mean`/`prod`/`max`/`min`/`argmax`/`argmin` Program lowering, and
  compile/bind/session hooks through package and type smokes. A June 27, 2026
  pass made the training loop follow the same split: `compile.compileForTraining`
  produces native compiled trainers for the supported Sequential Linear/ReLU
  classifier shape and the direct Linear/MSE/plain-SGD regression shape, while
  `train.fit(trainer, loader, { epochs })` keeps the JS/TS side ergonomic and
  each batch's forward/backward/update goes through Zig FFI. The Node and Bun
  classifier smokes check the AdamW path learns, and the Node and Bun linear
  smokes check the compiled native MSE/SGD path learns, with native
  `TrainFitEvidence` reported in both cases. The root-level
  product/API diet now has a checked internal non-breaking taxonomy:
  `src/ts/public_surface.ts` names the stable root namespaces users should learn
  first, names `zgml` as the stable friendly root value, separates advanced
  runtime/evidence namespaces, and keeps compatibility slices such as `torch`
  explicit while preserving existing exports without adding another public
  package subpath. Native eager execution policy is still open work. The first
  normal-module route now exists on Node and Bun: eligible `nn.Linear.forward`
  calls inside `zgml.noGrad(...)` route through the native eager linear hook
  while grad-enabled training keeps the TS/autograd graph path. That Linear
  boundary now preserves rectangular nested-array shape too, so subclassed
  `nn.Module` models can accept plain batched production inputs and still enter
  the native eager Linear lane instead of flattening them into a single feature
  vector. The public TS API now mirrors that runtime path for literal and typed
  nested arrays: `TensorLikeShape`, `Linear.forward(...)`, and
  `Sequential.forward(...)` infer vector/batch output shapes from plain JS array
  inputs instead of requiring users to pre-wrap every production batch as a
  `Tensor`. Supported
  `nn.Sequential` inference also now keeps the ergonomic JS/TS
  `model.forward(input)` call shape while transparently caching a native
  Program/Session in `noGrad`; the package smoke poisons the JS layer
  `forward()` methods and still proves the Linear/ReLU/Linear output, then
  mutates weights and proves the cached Program rebinds fresh packed parameters
  instead of serving stale native state. The same automatic native Program path
  also accepts the PyTorch-like plain vector input shape for Linear-headed
  Sequential models, so `noGrad(() => model.forward([x0, x1]))` reaches the
  Zig Program/Session path without forcing users to pre-wrap the vector as a
  Tensor. Rectangular nested arrays now preserve their batch shape too, so
  `noGrad(() => model.forward([[x0, x1], [y0, y1]]))` follows the same native
  Program route with a `[batch, features]` tensor boundary.
  The new Node/Bun-selectable `NATIVE_EAGER_GAP_JSON` microscope measures the first targets directly:
  `linear_batched` eager TS tensor execution,
  `lazy_matmul_add_gelu_batched` eager fused matmul work, and
  `lazy_matmul_add_relu_batched` / `lazy_matmul_add_silu_batched` /
  `lazy_matmul_add_sigmoid_batched` / `lazy_matmul_add_tanh_batched` eager fused
  matmul work plus scalar `elementwise_mul_batched` /
  `reduce_sum_scalar_batched`, `softmax_batched` / `log_softmax_batched` row
  tails, and `conv2d_batched` image-kernel work versus the relevant Zig-backed
  native eager or allocation-free compiled `prepare/executeInto` path for the
  same shape. It now also reports
  `nativeEagerIntoMs` for `linear_batched`, backed by the stateless
  `zgml_eager_linear_f32` C ABI and surfaced on Node and Bun as
  `zgml.nativeEager.linearInto`. It also reports `nativeEagerIntoMs` for
  `lazy_matmul_add_gelu_batched` through the fused
  `zgml_eager_linear_activation_f32` C ABI, surfaced as
  `zgml.nativeEager.linearActivationInto` /
  `zgml.native_eager.linear_activation_into`, so the first caller-owned
  native eager epilogue path covers `matmul -> add(bias) -> GELU` directly
  instead of only proving plain Linear.
  Native eager matmul is also now surfaced as `zgml_eager_matmul_f32` and
  `zgml.nativeEager.matmulInto`, with Node/Bun `Tensor.matmul` dispatching
  through that Zig path automatically when gradients are disabled.
  Scalar RHS elementwise and scalar reductions are now covered the same way:
  `zgml.nativeEager.elementwiseInto` / `zgml.nativeEager.reduceInto` call
  `zgml_eager_elementwise_f32` / `zgml_eager_reduce_f32` directly, and normal
  no-grad `Tensor.mul(2)` / `Tensor.sum()` calls use those Zig ABI hooks for
  large tensors while grad-enabled training keeps the TS/autograd path.
  The same ordinary Tensor eager lane now covers more PyTorch-like control
  primitives without changing the frontend shape: primitive comparisons
  (`eq`/`ne`/`lt`/`le`/`gt`/`ge`) lower through the Zig elementwise ABI for
  scalar and same-shape operands, `zgml.nativeEager.clampInto` exposes the
  single-pass `zgml_eager_clamp_f32` ABI, and Node no-grad `Tensor.clamp()`
  uses that hook where the benchmark proves it is at least parity. Bun keeps
  high-level clamp on its faster JS loop for now while still exposing the raw
  Zig clamp ABI for explicit measurement. `zgml.nativeEager.whereInto` /
  no-grad `Tensor.where()` use a dedicated `zgml_eager_where_f32` ABI for
  result-shaped conditions plus scalar or same-shape values. General broadcast
  and autograd cases stay on the TS reference path until they have an equally
  honest native contract.
  The same microscope now also covers `matmul -> add(bias) -> ReLU`,
  `matmul -> add(bias) -> SiLU`, `matmul -> add(bias) -> Sigmoid`, and
  `matmul -> add(bias) -> Tanh`, proving the activation hook for common
  production, recurrent/classical, and LLaMA-style epilogues instead of only
  GELU-shaped transformer work.
  The same microscope reports `nativeEagerModuleForwardMs`,
  `nativeEagerModuleSpeedup`, and `nativeEagerModuleMaxAbsDiff` for
  both `zgml.noGrad(() => linearModel.forward(input))` and
  `zgml.noGrad(() => linearGeluModel.forward(input))`, plus
  `zgml.noGrad(() => linearReluModel.forward(input))` and
  `zgml.noGrad(() => linearSiluModel.forward(input))`,
  `zgml.noGrad(() => linearSigmoidModel.forward(input))`, and
  `zgml.noGrad(() => linearTanhModel.forward(input))`, proving ordinary
  `nn.Linear`, adjacent `nn.Sequential(Linear, GELU)`, and adjacent
  `nn.Sequential(Linear, ReLU)` / `nn.Sequential(Linear, SiLU)` /
  `nn.Sequential(Linear, Sigmoid)` / `nn.Sequential(Linear, Tanh)` module
  surfaces can take the native eager lane without users calling the low-level
  primitive directly. Exact two-layer `Sequential(Linear, Activation)` no-grad
  tensor inputs now prefer that fused native eager lane before falling through
  to the broader cached Program route, so cheap activations such as ReLU, SiLU,
  Sigmoid, and Tanh do not pay the heavier Program/session path for ordinary
  eager inference.
  The native eager ABI now also exposes row-wise `softmaxInto` and
  `logSoftmaxInto`; Node and Bun package smokes prove both caller-owned-output
  helpers plus ordinary `zgml.noGrad(() => nn.Softmax/LogSoftmax.forward(x))`
  module calls route through the Zig row kernel for last-axis inference.
  Normal no-grad `Tensor.softmax()` / `Tensor.logSoftmax()` calls now use the
  same native eager row-softmax hook directly, while grad-enabled calls keep the
  TS/autograd path.
  The same no-grad native eager policy now covers `nn.Conv2d.forward` for
  inference: Node/Bun expose `zgml.nativeEager.conv2dInto`, ordinary
  `zgml.noGrad(() => conv2dModel.forward(input))` routes through the Zig
  Conv2d kernel, and the native eager microscope compares direct native eager,
  normal module forward, and compiled Program execution on the same
  `conv2d_batched` workload.
  That same no-grad native eager policy now covers parameterless pooling:
  Node/Bun expose `zgml.nativeEager.pool2dInto`, ordinary
  `zgml.noGrad(() => nn.MaxPool2d/AvgPool2d.forward(input))` routes through
  the Zig pool2d kernel, and grad-enabled training keeps the TS path with
  max-index/count bookkeeping for backward correctness.
  The direct Zig C ABI implementations now have unpadded, unit-dilation
  conv2d/pool2d fast paths for the caller-owned-output lane. This keeps the
  JS/TS surface ergonomic while making the hot inference loops native by
  default for the common CNN case; optimized native eager microscope runs prove
  direct `conv2dInto` and `pool2dInto` rows above the native-eager floor with
  zero measured diff against the TS reference.
  The native eager microscope now carries those rows as decision-grade evidence
  as well: the expected row set is `row_coverage=20/20` after adding direct
  `matmul_batched`, `elementwise_mul_batched`, `reduce_sum_scalar_batched`,
  `elementwise_lt_batched`, `clamp_batched`, `where_batched`,
  standalone `activation_relu_batched` / `activation_sigmoid_batched` /
  `activation_silu_batched`, native eager `conv2d_batched`, and native eager
  `max_pool2d_batched` /
  `avg_pool2d_batched`; fresh Node/Bun short runs show zero measured module
  diff across the native eager rows, with per-row speedup floors recorded where
  the direct low-level ABI is evidence rather than the preferred high-level
  route. The standalone activation rows now prove the promoted path too:
  `zgml_eager_activation_f32` uses a vectorized Zig helper for ReLU/Sigmoid/SiLU
  sized tensors, and normal no-grad high-level `Tensor.relu()` /
  `Tensor.sigmoid()` / `Tensor.silu()` calls route through
  `nativeEager.activationInto` once the runtime threshold is met. Fresh Node/Bun
  runs show the public Tensor path
  above floor with zero measured diff, while grad-enabled activation calls and
  small tensors stay on the TS/autograd path.
  The native eager adapter policy now lives in
  `src/ts/adapters/native_eager_surface.ts`: Node and Bun share tensor coercion,
  shape inference, output validation, public aliases, and activation mapping,
  while the concrete runtimes only provide the host-specific ABI calls and
  status checks. That keeps the first native eager lane extensible without
  duplicating user-facing semantics in each FFI adapter.
  Native eager microscope runs now write ignored
  `bench-results/native-eager/native-eager-*.json` artifacts, and
  `bench:status` reports both the latest native eager artifact and a
  `native-eager-runtime-results:` summary with separate Node and Bun
  row coverage, missing-row list, minimum module/native-into speedups, and max
  module diff. That keeps the normal-module native eager proof close to
  PyTorch/ggml evidence instead of hidden in transient console output, and it
  prevents an older Node or Bun artifact from looking complete after the native
  eager surface grows.
  These C ABI paths use the same shared native matmul substrate as compiled
  Program execution, then apply bias and optional activation into the
  caller-owned output buffer.
  Treat the reported
  `nativeProgramSpeedup` rows as executable targets for the first native eager
  `Linear`/`matmul` and fused `matmul -> add -> GELU` storage slices, not as a
  claim that eager already runs natively.
The public-surface taxonomy now separates the small checked first-contact
surface (`simple`, `zgml`, `tensor`, `nn`, `loss`, `optim`, `train`, `data`,
`checkpoint`, `lazy`, `compile`, and `compile.compileForInference`) from the
wider inspectable Program/Session/runtime evidence surface, so users learn the
brilliant path before the deployment controls. The remaining frontend jump is
native lowering, breadth, and first-contact simplicity, not proof that
`nn.Linear`, training, state dicts, data loaders, model math primitives, or
compile hooks exist.
The root runtime now also exports `simple`, a frozen native-backed first-contact
subset containing `Tensor`, `tensor`, `nn`, `F`, `data`, `loss`, `optim`,
`train`, `checkpoint`, `lazy`, `compile`, `compileForInference`, and grad-mode
helpers. The `zgml/simple` subpath owns the matching manifest, so examples can
opt into the small surface without hiding the advanced runtime SDK from users
who need it.
The frontend workflow evidence now uses that surface directly for the previously
open Conv2d and token-classifier gaps: `examples/node_training/train_conv2d.cjs`
trains, checkpoints, restores, and compiles a small Conv2d+ReLU feature model,
while `examples/node_training/train_token_classifier.cjs` trains, checkpoints,
restores, and compiles an `Embedding -> Linear -> LogSoftmax` classifier with
allocation-free output. These are examples, not new performance claims, but
they make the PyTorch-like first-contact workflow more concrete.

PyTorch remains an important comparison target and useful compatibility
vocabulary, but it is not the identity of the library. The primary product
surface is zgml: `Tensor`, `nn`, `F`, `loss`, `optim`, `train`, lazy graphs,
compiled Programs, Sessions, and model-source helpers. Any `torch` namespace
should be treated as a compatibility alias/lane and should not own roadmap
language, benchmark names, or the default mental model. The PyTorch performance
target is now explicit: `bench:pytorch` is an evidence command that always
prints the worst `zgml_vs_pytorch` ratio, while `bench:pytorch:parity` is the
hard parity gate (`BENCH_PYTORCH_REQUIRE_PARITY=1`). That gate is now expected
to be the proof target for checked CPU workloads. The current checked set is
green for linear, MLP, fused GELU matmul, RMSNorm+SiLU FFN, and pooling lanes.
The root Node/Bun native package now exposes `zgml` as the canonical friendly
namespace and keeps `torch` as the compatibility alias to the same frozen object;
`README.md` leads with `import { zgml } from "zgml"`, while runtime smokes assert
`adapter.zgml === adapter.torch` so this naming correction does not create a
second frontend implementation.
The shared adapter compile namespace now follows the same rule in diagnostics:
unsupported compile paths report canonical `compile.*` errors, not
`torch.compile.*`, even though the compatibility alias can still call the same
functions.
The adapter also reuses the canonical `compileSupportRejectionReason` helper
instead of carrying a local rejection parser, shrinking one more duplicated
piece of compile namespace behavior while the host-specific native compiler
hooks remain injected.
The canonical tutorial `examples/quickstart/zgml-first.cjs` now keeps the same
first-contact story small: train, checkpoint, restore,
`zgml.native(...)`, run allocation-free `into(...)`, and print a
plain proof line. The assertion-heavy contract smoke remains
`examples/node_training/quickstart.cjs`; it exercises `compileSupport()`,
`explain()`, `preflight()`, `kernelPlan()`, and `compilerSignatures()` for drift
protection without making those internals the tutorial's first screen.
The top-level README and its typechecked quickstart smoke now teach that same
friendly handle first, leaving raw `Program`/`Session` binding as the explicit
advanced path.
Exploratory focused lanes still track softer micro-workloads such as
classifier `LogSoftmax` tails and token-head paths, but those are microscopes
for ranking optimization opportunities rather than promoted hard-gate misses.
The report treats microbenchmark noise as part of the contract by printing the
selected attempt and noisy-attempt count.
Both commands build the native C ABI with `-Doptimize=ReleaseFast` first; PyTorch
comparisons must not silently measure a stale Debug dylib.
The PyTorch comparison must also match the zgml workload shape exactly. The
current corrected evidence compares the lazy RMS/SiLU/FFN case as 64->128->64
instead of a smaller 64->64->64 PyTorch proxy; on June 19, 2026 this moved the
honest worst observed ratio to about `0.40x`, with the FFN case at about `0.54x`.
A later June 19, 2026 pass fused the dense `matmul -> repeat(bias) -> add`
linear-bias command path into one dense projection dispatch. Current evidence:
`linear_batched` improved to about `0.71x` of PyTorch, `lazy_mlp_batched`
reached about `1.03x`, the FFN case moved to about `0.71x`, and the worst
observed PyTorch ratio moved to about `0.58x` (`lazy_matmul_add_gelu_batched`).
A follow-up pass fused the trailing GELU in the same dense projection command
for the lazy `matmul -> add(bias) -> gelu` path, reducing that runtime profile
from two commands to one and moving the observed worst PyTorch ratio to about
`0.62x`.
The same dense projection activation path now handles SiLU for FFN blocks,
dropping the lazy RMS/SiLU/FFN runtime profile from four commands to three and
moving that observed PyTorch ratio to about `0.79x`.
A later full-model F16 SmolLM pass extended that same idea to the real FFN
shape where the gate projection is followed by a standalone unary SiLU before
the up-projection product. The dense projection-pair command now accepts
`matmul -> silu -> matmul -> mul` directly, and the Metal sidecar kernels can
apply unary activation post-ops without materializing an extra dispatch. Focused
ReleaseFast evidence on June 22, 2026 moved F16 SmolLM prompt/decode command
pressure to `242/212`, with `30` dense FFN pair commands, `60` dense projection
chains, and zero fallback ops. This is the preferred iteration pattern:
discover shape misses with `npm run dev:zig:filter -- "<test name>"`, the
narrow incremental Zig loops, and debug microscopes, then promote only the
ReleaseFast full-model evidence.
A focused PyTorch-parity pass then specialized the dense bias and bias+activation
post-op loops so they branch once per fused command instead of inside each
vector chunk, and made the PyTorch GELU comparison explicitly use
`approximate="tanh"` to match zgml's documented GELU semantics. Latest
June 19, 2026 evidence: `linear_batched` is near parity at about `0.94x`,
`lazy_matmul_add_gelu_batched` beats PyTorch at about `1.68x`,
`lazy_mlp_batched` beats PyTorch at about `1.59x`, and the remaining worst
case is `lazy_rms_silu_ffn_batched` at about `0.86x`.
A subsequent row-chain executor pass taught the CPU reference tape to execute
`rmsnorm -> repeat(weight) -> mul` as one scaled RMSNorm pass instead of three
materialized row-chain entries. At that June 19, 2026 checkpoint,
`linear_batched` was the only remaining miss at about `0.94x`, while
`lazy_matmul_add_gelu_batched` is about `1.69x`, `lazy_mlp_batched` is about
`1.43x`, and `lazy_rms_silu_ffn_batched` now reaches about `1.01x`.
A direct-session CPU fast-path pass then tightened the direct `executeInto`
path and made it allocation-free with lighter profile bookkeeping. That pass
improved the contract and some lightweight samples, but did not yet make the
hard parity gate green: one lightweight run matched `linear_batched` at about
`1.01x` while missing `lazy_mlp_batched` and `lazy_rms_silu_ffn_batched`; a
required three-attempt run still missed with `linear_batched` around `0.94x`
even though
`lazy_matmul_add_gelu_batched`, pooling lanes, and most fused module lanes beat
PyTorch. That confirmed the right next move was more native/public hot-path
headroom, not lowering the parity floor or treating a lucky sample as success.
The direct-session fix did land an important runtime contract improvement:
borrowed NativeBuffer parameters no longer bypass explicit persistent upload on
the direct linear fast path.
The next pass removed duplicate output-buffer validation from the generic
`executeInto` hot path while preserving structured StepParams diagnostics for
invalid output buffers. That was useful hot-path work, but a fresh June 23,
2026 local rerun exposed that the no-rebuild hard parity shortcut could measure
a stale non-ReleaseFast native dylib after other Zig targets ran. The hard
`bench:pytorch:parity:run` command now rebuilds the native C ABI with
`-Doptimize=ReleaseFast` before comparing. With that correction, the current
hard PyTorch rerun against PyTorch `2.12.1` passes on the selected attempt:
`linear_batched` is `1.04x`, `lazy_matmul_add_gelu_batched` is `2.75x`,
`lazy_mlp_batched` is `1.61x`, `lazy_rms_silu_ffn_batched` is `1.63x`,
`max_pool2d_batched` is `7.02x`, and `avg_pool2d_batched` is `4.45x`.
Treat this as current evidence for the checked CPU lanes, with one important
caveat: `linear_batched` remains a thin/noisy margin lane (`ratio_median=0.97x`)
and still deserves more stable headroom.
The hard parity command now bootstraps upstream PyTorch into the repo-local
`.venv` with `uv` (`BENCH_PYTORCH_INSTALL=1`) before comparing, so the PyTorch
gate is no longer a soft local-environment skip when the reference package has
not been installed yet. The lightweight `bench:pytorch` evidence command still
uses the active Python environment and reports a skip instead of unexpectedly
downloading PyTorch.
The hard parity gate also treats microbenchmark noise as part of the benchmark
contract. `BENCH_PYTORCH_ATTEMPTS` defaults to `3` for required parity and `1`
for lightweight evidence, the report prints the selected `attempt=x/y` plus
`noisy=n`, and the gate only passes when at least one measured attempt clears
the requested per-workload ratio floor.
For iteration speed, the benchmark probes now have opt-in narrow lanes:
`BENCH_MODULE_PROGRAM_KEYS=<key>` runs only selected module Program specs, and
`BENCH_PYTORCH_KEYS=<key>` compares only selected PyTorch comparison keys while
passing the same filter to the child module bench. The default PyTorch
comparison set also filters its child module bench to the PyTorch comparison
lanes, so an unrelated full module Program floor cannot mask the PyTorch result;
the full `bench:module-program` command remains the separate release proof for
all module Program lanes. The default `bench:pytorch`, `bench:pytorch:parity`,
and `bench:module-program` commands still run their owned evidence sets; the
filters are for microscope work, not release claims. The
named `bench:pytorch:gaps` and `bench:pytorch:gaps:run` scripts keep the former
PyTorch soft spots as a one-command loop (`linear_batched` and
`log_softmax_classifier_batched`) so kernel work can still iterate without
repeatedly typing benchmark key filters. They are now microscopes, not current
hard-gate blockers: the June 23, 2026 hard parity run has no PyTorch lane below
parity in the required set. That shifts the next performance question away from
PyTorch catch-up and back to the Q8/ggml substrate frontier.
The PyTorch comparison output now also prints `ratio_range` and `ratio_median`
for each selected lane across repeated attempts, because these microsecond CPU
lanes are noisy enough that a single lucky attempt can make a miss look like
parity. A June 22, 2026 three-attempt gap rerun reported
`ratio_range=linear_batched:0.28-0.86x,log_softmax_classifier_batched:0.39-0.91x`
and `ratio_median=linear_batched:0.73x,log_softmax_classifier_batched:0.82x`;
those median numbers are the better guide for the next tiny-kernel pass.
The PyTorch checker now has an explicit median-parity mode
(`BENCH_PYTORCH_REQUIRE_MEDIAN_PARITY=1`) for hard claims. The default current-gap
rerun stays fast, while `dev:perf:pytorch:gaps:steady:run` and
`bench:pytorch:steady` use a 150ms timing window, three attempts, and exact
`prepared_execute_into_ms` zgml timings. The default steady gate now covers the
six current compiled hot-path lanes together:
`linear_batched`, `lazy_matmul_add_gelu_batched`,
`lazy_rms_silu_ffn_batched`, `rms_gelu_linear_batched`,
`log_softmax_classifier_batched`, and `lazy_token_head_batched`. That makes the
PyTorch evidence broader instead of easier: strong fused Program wins and the
known classifier-tail/linear noise stay in the same artifact. A hard
median-parity claim must opt into
`BENCH_PYTORCH_REQUIRE_PARITY=1 BENCH_PYTORCH_REQUIRE_MEDIAN_PARITY=1`; the
current steady evidence remains intentionally non-fatal because
`log_softmax_classifier_batched` still has noisy sub-parity reruns.
A June 27, 2026 three-attempt, 80ms-window eleven-lane broad artifact now
carries that claim across the broader PyTorch scoreboard with fresh native code
and PyTorch `2.12.1`: `lane_pass=11/11`, `median_lane_pass=11/11`, and
`first_contact_inference=11/11`, with
`ratio_median=linear_batched:1.24x,lazy_matmul_add_gelu_batched:2.84x,lazy_mlp_batched:1.80x,lazy_rms_silu_ffn_batched:1.67x,lazy_conv2d_relu_batched:1.19x,max_pool2d_batched:8.70x,avg_pool2d_batched:4.98x,rms_gelu_linear_batched:2.73x,softmax_classifier_batched:1.16x,log_softmax_classifier_batched:1.43x,lazy_token_head_batched:1.65x`.
All eleven broad lanes now prove the friendly `zgml.compileInference` /
`compile.compileForInference` handle reaches the same allocation-free prepared
path, including named-parameter lazy graphs that bind through explicit Program
bindings. That includes the lazy `Conv2d -> ReLU` path, whose core loop is now
owned by the Zig reference backend rather than by JS/TS. This makes the scoped
"PyTorch-like compiled inference" claim stronger without pretending that broad
eager PyTorch replacement is done.
A June 24, 2026 three-attempt, 150ms-window six-lane focus artifact now passes
both selected-attempt and median parity against PyTorch `2.12.1` with fresh
native code:
`ratio_median=linear_batched:1.21x,lazy_matmul_add_gelu_batched:2.83x,lazy_rms_silu_ffn_batched:1.71x,rms_gelu_linear_batched:3.58x,log_softmax_classifier_batched:1.05x,lazy_token_head_batched:1.22x`.
The selected attempt's worst lane was `log_softmax_classifier_batched:1.05x`,
so the current CPU Program/Session evidence is no longer merely a narrow
linear/log-softmax microscope. It is still not a broad PyTorch replacement
claim, but it does raise the checked compiled-hot-path CPU evidence from
"plausible" to "currently passing on the six promoted focus lanes." A rejected
follow-up native shortcut for `Embedding -> Linear -> LogSoftmax` is also useful
negative evidence: a scalar per-token direct path regressed the token-head lane,
so the next token-head win should come from a real fused/backend row kernel or
better dispatch amortization, not from replacing BLAS-backed scheduled work with
another hand-rolled loop.
The PyTorch comparison gate now also refuses stale native evidence by default:
before timing it checks the loaded `zig-out/lib/libzgml_c.*` timestamp against
`build.zig` and Zig/Metal/C-header sources, prints `native=fresh` in accepted
reports, and throws with an explicit rebuild instruction when the library is
older than source. `BENCH_PYTORCH_ALLOW_STALE_NATIVE=1` is reserved for
intentional diagnostics and prints `native=stale`, not a normal evidence claim.
This matters because a stale no-rebuild microscope can make a near-parity
kernel look several times slower than it is.
The PyTorch checker now also writes an ignored JSON artifact by default under
`bench-results/pytorch/` (`schema: "zgml.pytorch-comparison.v1"`) and prints a
machine-readable `PYTORCH_COMPARISON_JSON` line with the artifact path, selected
attempt, native freshness, parity status, lane pass counts, lane miss lists, and
worst ratio. Set
`BENCH_PYTORCH_WRITE_ARTIFACT=0` only for throwaway local diagnostics. This
keeps PyTorch competitiveness evidence closer to the ggml artifact model:
checked claims should have preserved keys, attempts, medians, timing windows,
Torch version, machine metadata, and native freshness, not just a console line
or a dated sentence in this plan. `bench:status` reads the newest eleven-lane broad
artifact back out as the selected `pytorch-results:` line when one exists, with
status, median status, worst ratio, selected/median lane pass counts,
selected/median miss lists, selected attempt, native freshness, Torch version,
timing metric, active keys, and ratio medians. When a narrower microscope is
newer than that selected broad artifact, it prints `pytorch-latest-results:` so
freshness stays visible without letting a one-lane probe hijack the PyTorch
replacement scoreboard. That latest line now includes status, median status,
worst ratio, first-contact inference coverage, attempts, native freshness,
timing metric, key count, and ratio medians; a fresh one-lane microscope can
sharpen the next optimization target without pretending to prove broad PyTorch
parity. The selected PyTorch line also prints `first_contact_inference=N/M`
when the artifact carries that field, so public-handle reachability is visible
beside the raw prepared Program timing. The same status readback also prints
the most recent six-lane focus artifact as `pytorch-focus-results:` when it
differs. This keeps three different facts visible at once: the selected broad
PyTorch-like replacement sample, the freshest local experiment, and the
current-hot-path focus sample.
The same status readback now prints a compact `perf-next:` line that turns the
current artifacts into an iteration target: weakest full-model ggml lane and
distance-to-90%, PyTorch median misses, Q8 prompt semantic readiness, and the
qsemantic frontier target. The qsemantic field includes candidate-vs-default
ratios, not only raw speedups, so a tiled diagnostic stays visibly diagnostic
when it is faster than the staged baseline but slower than the current semantic
command path. That keeps the next performance step evidence-led instead of
plan-led: start with the line, run the focused microscope it points at, then
only promote to full ggml/PyTorch gates after the bottleneck moves.
`perf-next` keeps two Q8 full-model signals separate: `full_model=` remains the
accepted long-run p128/g200 artifact used for baseline/trend evidence, while
`q8_current=` reads the latest required-pass p128/g40 smoke so the promoted
semantic command shape can steer local iteration. That prevents the next-action
hint from pointing at stale `projection_chain:60` pressure after a newer
semantic-default smoke has already reduced the Q8 prompt command shape to 151
commands; the current smoke target is the semantic FFN sublayer throughput
kernel.
The `dev:perf:next` router follows that split: while `q8_current` points at
`semantic_ffn_sublayer_throughput_kernel`, the default next lane is the
qsemantic throughput microscope until fresh qsemantic throughput is above the
current default path. Once that happens, the router advances to the Q8 prompt
viability gate so the microscope win has to prove itself at the model level
before any ggml smoke or promotion claim. When the Q8 prompt readback already
selects a `semantic_bridge_candidate`, the router treats
`next=steady_semantic_bridge_candidate` as a model-level proof request: it runs
the Q8 prompt viability gate in steady paired-default mode even when
`BENCH_NEXT_PERF_STEADY` is not set. That proof is strict for bridge candidates:
median parity is not enough if the worst paired attempt is still below `1.0x`.
In that case `perf-next:` reports `next=semantic_bridge_throughput_kernel` and
the router returns to the semantic throughput microscope instead of repeating
the same Q8 proof. If the latest bridge artifact reports single-dispatch
refusals concentrated on `dim`, `perf-next:` names the sharper target as
`next=semantic_width_parallel_kernel`; that means the bridge recognizer is
ready, but the row-serial semantic kernel is too narrow or too slow for the
full hidden width, so `dev:perf:next` routes to the exact qsemantic bridge
microscope for the `576 x 1536 x 576` shape. Use the ggml smoke after the
frontier bottleneck moves or when explicitly checking model-level promotion.
The absorbed input-bridge microscope now also separates the old "one dispatch"
temptation from the real throughput target. A fresh three-attempt run adds a
`direct_serial` lane to the `qsemantic input bridge` artifact: it proves the
existing direct bridge kernel is correct and dispatches once, but it remains
row-owned/serial (`row_serial_dot_ops=2985984`,
`total_row_serial_dot_ops=382205952`) and is slower than the decomposed absorbed
path (`direct_serial` selected `1.42x`, median `1.23x`, versus absorbed
selected `3.09x`, median `2.56x`). That makes the next implementation target
more precise: not "reduce dispatches at any cost", but
`semantic_with_input_width_parallel_kernel`, carrying the width-parallel
semantic FFN work into the residual input-bridge shape. The direct row-serial
bridge is a named diagnostic opt-in policy rather than a raw-policy default:
turning on input-bridge recognition should preserve the fastest physical
lowering unless a benchmark explicitly asks for the one-dispatch microscope.
`perf-next:` now carries that fact as a compact
`qsemantic_input_bridge=...:next=semantic_with_input_width_parallel_kernel`
field, and `dev:perf:next` routes to the exact qsemantic input-bridge
microscope when that target is present. The generic
`semantic_width_parallel_kernel` bridge microscope remains the fallback when no
input-bridge artifact exists; once direct-serial versus absorbed evidence is
available, the next loop should work on the input-bridge kernel directly. The
frontier microscopes now also report `qmatmul_row_chain_width_parallel_count`
and lane totals, so the benchmark artifact proves which physical row-chain tail
used the width-partitioned Metal kernel instead of inferring it from generic
tiled row-chain counters. The input-bridge artifact also records
`semantic_ffn_with_input_decomposed_extra_dispatches`: the retained absorbed
path reports `4`, meaning the current correct production lowering is still four
dispatches away from the intended single executable input-bridge artifact.
The input-bridge artifact and `bench:status` readback now also carry the
semantic-width scratch fields (`semantic_width_scratch=*`) for the absorbed
lane. That keeps the exact bridge and input-bridge microscopes comparable:
fresh artifacts show whether the width-parallel tail has backend-owned scratch
capacity, whether runtime execution used it, and how large the down-partial
scratch is relative to the final output. A fresh one-attempt input-bridge
microscope proves the readback on the retained decomposed path with
`candidates:1`, `down_partial_bytes:14155776`, `runtime_capacity:9216`,
`runtime_uses:628`, and `runtime_bytes:9216`.
The qsemantic-throughput and qsemantic input-bridge readbacks follow the same
stability rule as the other noisy perf lanes:
`qsemantic-throughput-results:` and `qsemantic-input-bridge-results:` prefer the
latest three-attempt artifact, while `qsemantic-throughput-latest-results:` and
`qsemantic-input-bridge-latest-results:` report newer one-attempt probes when
they exist. That keeps quick kernel iteration visible without letting a lucky or
unlucky single attempt replace the selected semantic frontier proof. The checked
input-bridge gate now adds a collapse floor for steady runs too:
`BENCH_FRONTIER_ATTEMPTS>=3` requires best absorbed speedup to stay at or above
`2.45x` by default, configurable with
`BENCH_QSEMANTIC_INPUT_BRIDGE_STEADY_ABSORBED_FLOOR`.
The PyTorch comparison microscope also accepts exploratory lanes such as
`rms_gelu_linear_batched`, `softmax_classifier_batched`,
`log_softmax_classifier_batched`, and `lazy_token_head_batched`, so optimization
work can rank PyTorch gaps without promoting those gaps into the hard parity
gate. The `rms_gelu_linear_batched` lane has already graduated from "known
miss" to proof of the intended loop: the focused benchmark exposed a PyTorch
gap, then a direct CPU Session fast path for
`RMSNorm(weight)+GELU -> Linear(weight,bias)` moved that lane from about `0.45x`
to about `2.63x` versus PyTorch while keeping the public Program plan
unchanged. A later June 22, 2026 direct CPU `Linear -> LogSoftmax` classifier
tail pass did the same for the tiny row-tail gap without adding a public ABI
op: the Program plan still reports `linear|log-softmax`, but CPU
`executeInto` recognizes that exact shape and runs small direct linear plus
in-place stable row log-softmax. A fresh ReleaseFast pre-change run measured
`log_softmax_classifier_batched` at `0.61x` (`zgml:0.0159ms`,
`pytorch:0.0098ms`). After extending the small direct linear threshold to the
actual `M=128,N=32,K=64` gap shape, the post-change three-attempt microscope
found a passing attempt at `3.97x` (`zgml:0.0151ms`, `pytorch:0.0599ms`) with
median `1.57x`; the same pass kept `linear_batched` at a `2.23x` selected ratio
with median `0.81x`. That is useful progress, but still noisy enough that the
ratio range remains the honest contract rather than a permanent universal
parity claim. A June 23, 2026 small-direct CPU pass then made that gap less
pathological: the direct `Linear` fast path now uses a 16-wide vector kernel for
aligned small output rows while preserving the old 8-wide path for narrower
rows, and the direct `Linear -> LogSoftmax` tail now vectorizes row
max/sum/subtract work. The later hard parity rerun moved this from "best attempt
can clear parity" to "required PyTorch parity is green"; further tiny-kernel
work is still useful, but it is no longer the main plan blocker.
A later microscope pass split the tiny CPU policy more carefully: plain batched
direct `Linear` now lets the BLAS-backed path handle sufficiently batched small
dense shapes (`M>=64,N<=64,K<=128`), while the fused `Linear -> LogSoftmax`
tail keeps the hand-written small direct kernel because that tail regressed when
the linear portion was forced through BLAS. In the focused PyTorch microscope
this moved `linear_batched` from a noisy below-parity median to a measured
`1.05x` median against PyTorch `2.12.1`, without changing the public Program
plan or promoting the exploratory log-softmax lane into the hard parity set.
The repo now exposes that inner loop directly:

```text
npm run dev:zig:test        # incremental Zig tests while editing kernels/runtime
npm run dev:zig:test:watch  # Zig 0.16 watch mode with incremental rebuilds
npm run dev:zig:quick       # faster native unit loop without optional Metal/BLAS linking
npm run dev:zig:quick:watch # watched version of the fast native unit loop
npm run dev:zig:quick:webui # watched fast unit loop with Zig build Web UI for compile-latency diagnosis
npm run dev:zig:public      # narrow public API Zig test loop
npm run dev:zig:internal    # narrow internal runtime Zig test loop
npm run dev:zig:conformance # narrow backend conformance Zig test loop
npm run dev:zig:c-api       # narrow C ABI Zig test loop
npm run dev:zig:ffi         # incremental native FFI dylib build
npm run dev:zig:ffi:watch   # watched native FFI dylib build
npm run dev:zig:bench       # explicit ReleaseFast incremental benchmark-binary build while editing hot code
npm run dev:zig:bench:watch # watched ReleaseFast incremental benchmark-binary build loop
npm run dev:zig:bench:webui # watched ReleaseFast benchmark build with Zig build Web UI
npm run dev:zig:time-report # one-off Zig compile-time report when build latency itself is the bottleneck
npm run dev:zig:metal-row-chain       # focused incremental Metal row-chain kernel test
npm run dev:zig:metal-row-chain:watch # watched focused Metal row-chain kernel test
npm run dev:wasm:browser-llama        # incremental Wasm C ABI build, then focused browser LLaMA proof rerun
npm run dev:wasm:browser-llama-families # incremental Wasm build, then representative SmolLM3/Mistral/Qwen3 browser proof
npm run dev:wasm:browser-gpu-llama-families # incremental Wasm build, then require real-GPU representative family proof
npm run dev:perf:module-program       # incremental ReleaseFast native rebuild plus focused Program/Session bench
npm run dev:perf:module-program:run   # rerun focused Program/Session bench without rebuilding artifacts
npm run dev:perf:native-eager-gap     # incremental eager-vs-compiled native gap microscope
npm run dev:perf:native-eager-gap:run # no-rebuild eager-vs-compiled native gap microscope
npm run dev:perf:native-eager-gap:bun # incremental Bun native eager gap microscope
npm run dev:perf:native-eager-gap:bun:run # no-rebuild Bun native eager gap microscope
npm run dev:perf:pytorch:focus        # incremental ReleaseFast native rebuild plus focused PyTorch comparison
npm run dev:perf:pytorch:focus:native # native-only focused PyTorch loop after dist exists
npm run dev:perf:pytorch:focus:run    # rerun focused PyTorch comparison without rebuilding artifacts
npm run dev:perf:pytorch:focus:steady:native # native-only 150ms-window focused PyTorch comparison
npm run dev:perf:pytorch:focus:steady:run # no-rebuild 150ms-window focused PyTorch comparison
npm run dev:perf:pytorch:broad        # incremental eleven-lane PyTorch replacement evidence loop
npm run dev:perf:pytorch:broad:native # native-only eleven-lane PyTorch loop after dist exists
npm run dev:perf:pytorch:broad:run    # rerun eleven-lane PyTorch comparison without rebuilding artifacts
npm run dev:perf:pytorch:broad:steady:native # fresh-native 150ms-window eleven-lane PyTorch replacement evidence
npm run dev:perf:pytorch:broad:steady:run # no-rebuild 150ms-window eleven-lane PyTorch replacement evidence
npm run dev:perf:pytorch:gaps:native   # native-only current-gap loop after dist exists
npm run dev:perf:pytorch:gaps:steady:native # fresh-native 150ms-window current-gap evidence
npm run dev:perf:pytorch:linear:native # native-only ReleaseFast microscope for the linear_batched PyTorch miss
npm run dev:perf:pytorch:linear:run    # no-rebuild rerun for the linear_batched PyTorch miss
npm run dev:perf:pytorch:logsoftmax:native # native-only ReleaseFast microscope for the logSoftmax classifier miss
npm run dev:perf:pytorch:logsoftmax:run    # no-rebuild rerun for the logSoftmax classifier miss
npm run dev:perf:pytorch:logsoftmax:steady:native # native-only 150ms-window logSoftmax classifier microscope
npm run dev:perf:pytorch:logsoftmax:steady:run    # no-rebuild 150ms-window logSoftmax classifier microscope
npm run dev:perf:competitive        # incremental PyTorch + native eager + qsemantic + full-model Q8 prompt + cheap ggml competitiveness loop
npm run dev:perf:competitive:run    # no-rebuild rerun of selected competitiveness lanes
npm run dev:perf:next               # read perf-next and run the smallest current bottleneck microscope
npm run dev:perf:next:build         # rebuild needed artifacts first, then run that microscope
npm run dev:perf:next:steady        # rerun that microscope with steadier attempts/windows
npm run dev:perf:next:qsemantic-bridge # rebuild and run the exact 576x1536x576 bridge microscope
npm run dev:perf:next:qsemantic-bridge:run # no-rebuild exact bridge microscope
npm run dev:perf:promotion          # qsemantic throughput + Q8 prompt viable + ggml smoke promotion path
npm run dev:perf:promotion:run      # no-rebuild rerun of the promotion path
npm run dev:perf:promotion:full     # escalate promotion to the baseline-gated p128/g200/r3 ggml artifact
npm run dev:perf:promotion:parity   # same full artifact, but require the hard 90% ggml parity gate
npm run dev:perf:competitive:native-eager # incremental native-eager-only competitiveness loop
npm run dev:perf:competitive:qsemantic # incremental qsemantic-only competitiveness loop
npm run dev:perf:competitive:q8-prompt # incremental full-model Q8 prompt competitiveness loop
npm run bench:module-program:focus
npm run bench:module-program:focus:run # rerun focused module benches without rebuilding artifacts
npm run bench:pytorch:parity:run       # rebuild ReleaseFast native, then rerun hard PyTorch parity
npm run bench:pytorch:steady           # rebuild and measure steady PyTorch current-gap evidence
npm run bench:pytorch:focus
npm run bench:pytorch:focus:run        # rerun focused PyTorch comparison without rebuilding artifacts
npm run bench:pytorch:gaps             # rebuild and measure current PyTorch soft spots
npm run bench:pytorch:gaps:run         # rerun current PyTorch soft spots without rebuilding artifacts
npm run dev:perf:pytorch:gaps:steady:run # no-rebuild steady PyTorch current-gap evidence
npm run bench:competitive           # promoted PyTorch/frontier/ggml competitiveness gate
npm run bench:frontier:gate            # rebuild ReleaseFast and measure scheduler/kernelizer frontier evidence
npm run bench:frontier:gate:run        # rerun frontier evidence without rebuilding artifacts
npm run bench:frontier:qproj           # rebuild ReleaseFast and gate only qproj projection-chain/group labels
npm run bench:frontier:qproj:run       # rerun only qproj frontier evidence without rebuilding artifacts
npm run dev:perf:frontier:qproj        # incremental checked qproj projection-chain/group microscope
npm run dev:perf:frontier:qproj:run    # no-rebuild checked qproj projection-chain/group microscope
npm run bench:frontier:qsemantic       # rebuild ReleaseFast and gate the Q8 semantic FFN/residual/norm frontier
npm run bench:frontier:qsemantic:run   # rerun only the Q8 semantic frontier without rebuilding artifacts
npm run dev:perf:frontier:qsemantic    # incremental checked Q8 semantic FFN/residual/norm microscope
npm run dev:perf:frontier:qsemantic:run # no-rebuild checked Q8 semantic frontier microscope
npm run dev:perf:frontier:qsemantic:steady # incremental three-attempt Q8 semantic microscope
npm run dev:perf:frontier:qsemantic:steady:run # no-rebuild three-attempt Q8 semantic microscope
npm run dev:perf:frontier:qsemantic:full:raw # incremental raw full-prefill qsemantic microscope
npm run dev:perf:frontier:qsemantic:full:raw:run # no-rebuild raw full-prefill qsemantic microscope
npm run dev:perf:frontier:qsemantic:target:raw # incremental target-only qsemantic kernel microscope
npm run dev:perf:frontier:qsemantic:target:raw:run # no-rebuild target-only qsemantic kernel microscope
npm run dev:perf:frontier:qsemantic:throughput # incremental checked throughput-candidate-only qsemantic microscope
npm run dev:perf:frontier:qsemantic:throughput:run # no-rebuild checked throughput-candidate-only qsemantic microscope
npm run dev:perf:frontier:qsemantic:throughput:raw # incremental throughput-candidate-only qsemantic kernel microscope
npm run dev:perf:frontier:qsemantic:throughput:raw:run # no-rebuild throughput-candidate-only qsemantic kernel microscope
npm run dev:perf:frontier:qsemantic:bridge # incremental exact 9-op semantic bridge microscope
npm run dev:perf:frontier:qsemantic:bridge:run # no-rebuild exact 9-op semantic bridge microscope
npm run dev:perf:frontier:qsemantic:input-bridge # incremental exact 14-op semantic input-bridge microscope
npm run dev:perf:frontier:qsemantic:input-bridge:run # no-rebuild exact 14-op semantic input-bridge microscope
npm run dev:perf:frontier:qsemantic:smollm:raw # incremental raw SmolLM qsemantic microscope
npm run dev:perf:frontier:qsemantic:smollm:raw:run # no-rebuild raw SmolLM qsemantic microscope
npm run bench:frontier:row-chain       # rebuild ReleaseFast and run only row-chain frontier labels
npm run bench:frontier:row-chain:run   # rerun only row-chain frontier labels without rebuilding artifacts
npm run bench:frontier:row-chain-region      # rebuild ReleaseFast and run only x7 row-chain region labels
npm run bench:frontier:row-chain-region:run  # rerun only x7 row-chain region labels without rebuilding artifacts
npm run dev:perf:frontier:row-chain-region      # incremental checked x7 row-chain region microscope
npm run dev:perf:frontier:row-chain-region:run  # rerun checked x7 row-chain region microscope without rebuilding artifacts
npm run bench:stencil:shape            # rebuild ReleaseFast and prove current source stencil shape/hashes
npm run bench:stencil:shape:run        # rerun current source stencil shape/hashes without rebuilding artifacts
npm run bench:q8-prompt-candidate      # rebuild ReleaseFast and measure full-model Q8 prompt candidate evidence
npm run bench:q8-prompt-candidate:run  # rerun Q8 prompt candidate evidence without rebuilding artifacts
npm run bench:q8-prompt-viable         # rebuild ReleaseFast and measure only command/two-phase Q8 prompt viable lanes
npm run bench:q8-prompt-viable:run     # rerun only command/two-phase Q8 prompt viable lanes without rebuilding artifacts
npm run dev:perf:q8-prompt:viable      # incremental command/two-phase Q8 prompt microscope, skipping the known-bad single-dispatch lane
npm run dev:perf:q8-prompt:viable:run  # no-rebuild command/two-phase Q8 prompt microscope rerun
npm run bench:ggml:parity:run          # rerun hard ggml parity; bench script rebuilds ReleaseFast by default
npm run dev:perf:ggml:q8-command-smoke # cheap ggml smoke with explicit Q8 projection-row-chain command path
npm run dev:perf:ggml:q8-command-smoke:run # rerun explicit Q8 command-path ggml smoke without rebuilding artifacts
npm run smoke:portable-ffi:browser-llama:run # rerun the focused browser LLaMA proof after Wasm artifacts already exist
npm run smoke:portable-ffi:browser-llama-families:run # rerun representative SmolLM3/Mistral/Qwen3 browser proof after Wasm artifacts exist
npm run smoke:portable-ffi:browser-gpu-llama-families:run # rerun representative family proof and require real GPUBuffer mode
zig build -Doptimize=ReleaseFast bench-build && ./zig-out/bin/bench-llama-smollm ignored 128 1 1 --stencil-only --debug-row-chain
```

These are not substitutes for `bench:pytorch:parity`, `bench:ggml:parity`, or
`check:goal-scorecard`; they are the tight microscope loop for forming and
discarding performance hypotheses quickly. The full gates remain the release
proof. The practical workflow is: use Zig `-fincremental`/`--watch` while
changing kernel/runtime code, keep benchmark binaries explicit `ReleaseFast`,
use `--webui` when rebuild latency needs inspection, reserve `--time-report`
for one-off compile-time diagnosis because it forces a full rebuild, build the
native/package artifacts once, use the focused `:run` reruns to check noisy
microscope lanes quickly, and let hard parity reruns rebuild ReleaseFast native
before claiming a PyTorch comparison result. Then run the full evidence gate
before claiming a new SOTA/simple/perf state.
The promotion runner is the bridge between those two worlds. It first runs the
checked qsemantic throughput frontier and the full-model Q8 prompt viable lane
with steady attempts, then runs a ggml smoke artifact by default. When a kernel
change survives that path, `dev:perf:promotion:full` escalates to the
baseline-gated p128/g200/r3 ggml artifact, and `dev:perf:promotion:parity`
turns the same path into the hard 90% parity claim. This keeps "candidate is
ready" from lingering as an unpromoted microscope result after the full-model
artifact should have moved.
The default edit loop should stay fast: start with `npm run dev:zig:quick` for
no-Metal/no-BLAS unit coverage, narrow failures with
`npm run dev:zig:filter -- "<test substring>"`, keep `npm run dev:zig:ffi` for
native ABI work, and only escalate to `dev:zig:bench`, focused PyTorch
microscopes, and `check:goal-scorecard` after the local hypothesis is shaped.
Watch variants of those commands are the preferred long-running loop when
iterating on a kernel or runtime contract.
For the remaining Q8 prompt frontier, the default all-lane
`bench:q8-prompt-candidate` gate is still the release proof, but the fast
kernel loop is now `dev:perf:q8-prompt:viable`: it sets
`BENCH_Q8_PROMPT_LANES=command,two_phase,semantic` and defaults to one attempt, so the
known-bad single-dispatch diagnostic is skipped while the command and two-phase
paths plus the semantic-throughput candidate still prove structure, fallback,
dispatch shape, decode fast-path shape, and throughput. The full-model gate now
accepts either the older pair-fused `projection_pair>=30` shape or the larger
semantic-command shape where projection pairs are consumed and at least 30
semantic row-chain commands remain. A June 24, 2026 fresh-native Q8 viable run
selected the semantic full-model lane structurally while keeping throughput
honest: `command_structural=ready`, `two_phase_structural=ready`,
`semantic_structural=ready`, `semantic_structural_selected=yes`,
`semantic_throughput_ready=off`,
`command_command=241->151`, `command_speedup=1.00-1.03x`,
`semantic_command=241->151`, `semantic_projection_pair=30->0`,
`semantic_projection_row_chain=0->30`, `two_phase_speedup=0.95-0.98x`,
`semantic_speedup=0.99-1.00x` after the tiled finalize pass, and zero fallback. That is evidence that the
larger semantic command is wired into the full model, but the stricter steady
readiness gate correctly keeps semantic throughput diagnostic until median
throughput clears parity. It is not yet a release promotion signal; repeated
all-lane candidate evidence still has to prove the semantic path is stable
before replacing the default. The Q8 prompt
candidate checker now writes an ignored JSON artifact by default under
`bench-results/q8-prompt/` (`schema: "zgml.q8-prompt-candidate.v1"`) and prints
a machine-readable `Q8_PROMPT_CANDIDATE_JSON` line. Each measured lane now
stores best/median/worst speedup stats and the attempt indices that produced
them, so a selected lucky attempt cannot be mistaken for stable throughput.
`bench:status` reads the latest artifact back as `q8-prompt-results:` with
status, semantic throughput, semantic structural selection, semantic throughput
readiness, lane speedups, median/worst throughput, command shape, active lanes,
and source freshness when the artifact carries it. It now prefers the newest
steady viable artifact (`attempts >= 3` with command, two-phase, and semantic
lanes) for the selected `q8-prompt-results:` broad scoreboard. The active
semantic bottleneck also has `q8-prompt-semantic-steady-results:`; when present,
`perf-next:` uses that newest steady semantic artifact as pressure evidence and
still prints `q8-prompt-latest-results:` when a newer quick probe exists. Set
`BENCH_Q8_PROMPT_WRITE_ARTIFACT=0` only for throwaway local
diagnostics. New Q8 and frontier artifacts also stamp generic benchmark-binary
metadata: git commit, dirty state, build mode, binary mtime, newest native
source path/mtime, and a fresh/stale label. This makes `BENCH_BUILD_ZGML=0`
iteration fast without making stale binaries invisible.
A June 27, 2026 no-rebuild `dev:perf:q8-prompt:semantic-steady:run` refreshed
that pressure artifact after the next-perf routing fix: three semantic-only
attempts held `semantic_median=1.15x`, `semantic_worst=1.11x`, and
`semantic_throughput_ready=yes`, with the same `semantic_dispatch=242->242`
and `semantic_command=151->121` shape. The proof is strong enough to keep the
semantic candidate promoted as the full-model default, but it also confirms the
remaining work is still the width-parallel kernel/storage model rather than
more command-policy selection.
The qsemantic frontier selector now mirrors that stability rule: `bench:status`
prefers the newest three-attempt qsemantic artifact for `frontier-results:` and
`perf-next:`, while printing `frontier-latest-results:` when a newer one-attempt
microscope exists. The freshness line now includes the newest microscope's
target status, throughput-candidate status, candidate-vs-default ratios,
target-vs-default ratios, and attempt count, so kernel iteration can see the
fresh local result without letting a single attempt rewrite the steady
`frontier-results:` or `perf-next:` target. That matches the qsemantic warning
below: single-attempt probes are useful for fast iteration, but not stable
enough to steer the whole next-action line.
A three-attempt no-rebuild proof remains useful as a noisier iteration lens,
while the fresh-native one-attempt proof now reports
`command_structural=ready`, `single_structural=skipped`,
`two_phase_structural=ready`,
`attempt=skipped median_attempt=skipped noisy=skipped`,
`command_command=241->151`, `command_speedup=1.00-1.01x`,
`command_throughput=ready`, `command_projection_row_chain=0->30`,
`command_projection_row_chain_dispatch=0->60`, `two_phase_count=60`,
`two_phase_speedup=0.97-0.99x`, `two_phase_tiled_work=60`,
`finalize_tile_groups=4320`, `finalize_elements=4423680`,
`semantic_speedup=0.96-1.00x`, `semantic_median_speedup=0.99x`,
`semantic_throughput_ready=off`, and zero fallback. That
is the intended iteration lens before spending time on the full all-lane gate:
it proves the command shape and confirms that the two-dispatch
projection-row-chain command path is model-level viable at the structural level,
while keeping the semantic path below release promotion until repeated all-lane
candidate evidence proves stable median throughput.
The fuller three-attempt all-lane Q8 scorecard keeps that line honest:
`command_median_speedup=1.00x`, `command_worst_speedup=1.00x`,
`single_throughput=off`, `dispatch=242->182`,
`single_dispatch_trap=serial_n_tile_loop_without_cross_threadgroup_row_reduce`,
and `reason=dispatch_reduction_without_tiled_throughput`.
The per-attempt progress line now reports `active_lane`, `dispatch`, and
`commands` for the measured command/two-phase lane when the known-bad
single-dispatch diagnostic is skipped, so the fast loop no longer prints
`242->n/a`/`241->n/a` while the useful lane evidence is present.
The next focused microscope is `qsemantic`, which models the larger Q8 prompt
shape the row-chain work keeps pointing at: projection-pair activation plus the
following projection, residual add, RMSNorm, repeat-scale, and multiply. The
subagent and local audits agreed that small row-chain tweaks are exhausted as a
primary lever; the credible next win is a semantic FFN sublayer kernel that
crosses the current `projection_pair_fused_elementwise_chain` and following
projection/residual/norm boundary. Current local evidence is intentionally
framed as a target gap: the full-prefill row is roughly `1.8x` faster than
staged with zero diff, the SmolLM prompt row is roughly parity with zero diff,
the current runtime shape is still `2` commands covering `9` ops and saving `7`
staged dispatches, and the current runtime still needs `3` backend dispatches.
The opt-in target compiler policy now recognizes the same work as one
`semantic_ffn_sublayer` command covering `9` ops and saving `8` staged
dispatches. Metal now executes that exact command as a bounded one-dispatch
diagnostic kernel. The target path is zero-diff, and the target dispatches `1`
backend kernel. The default throughput lane remains the faster
projection-pair product, down-projection residual add, then RMSNorm scale path.
The final missing performance move is a semantic FFN sublayer throughput
kernel, concretely replacing the scalar diagnostic with a
true tiled/vectorized `semantic_ffn_sublayer_throughput_kernel`; this is not
proof that full-model Q8 has been solved.
The current fresh qsemantic microscope makes the shape of the miss precise:
the one-dispatch target has `target_vs_default=full:0.28x,smollm:0.24x`, while
the mixed throughput candidate is still below default at
`candidate_vs_default=full:0.91x,smollm:0.93x`. The target exposes
`target_tile_groups=full:192,smollm:216`, and the mixed candidate exposes
`candidate_tile_groups=full:64,smollm:72`, so `perf-next` now reports the
3.00x `tile_gap` directly. The blocker is still the serial dot-loop work inside
the semantic row groups.
The source-fresh rebuilt qsemantic pass on June 24, 2026 confirmed that this is
not a promotion-ready candidate: the throughput candidate moved opposite
directions across the two prompt shapes
(`candidate_vs_default=full:1.20x,smollm:0.91x`), while the one-dispatch target
remained much slower (`target_vs_default=full:0.36x,smollm:0.22x`). Treat that
as proof that the current candidate is useful diagnostic scaffolding, not a
default path. The next move still has to change the semantic FFN sublayer
kernel's work partitioning or grow the semantic command to remove more
surrounding work; merely selecting the existing throughput candidate would
improve one frontier row and regress the SmolLM-shaped row.
A later June 24, 2026 pass tightened that diagnostic lane: the named
`promptSemanticFfnSublayerThroughputCandidate()` now exercises the actual
one-dispatch semantic FFN sublayer kernel instead of the older three-dispatch
two-phase row-chain tail. The fresh rebuilt qsemantic throughput artifact
passes shape/profile evidence with `runtime_backend_dispatches=1`,
`semantic_ffn_sublayer_count=1`, and semantic tile groups
`full:192,smollm:216`, with zero row-chain tiled tail. A retained
`SEMANTIC_FFN_THREADS=512` tuning pass then moved the three-attempt focused
throughput artifact to `full_prefill=0.95x` and `smollm_prompt=0.83x`. That is
still not a promotion, but it is a better frontier because it proves and tunes
the exact kernel that must be optimized, rather than a faster fallback tail
wearing the throughput-candidate name.
A follow-up shape-selective probe tried using the two-phase tiled tail only for
the `512`-wide full-prefill geometry while keeping the `576`-wide SmolLM shape
on the default tail. It was not kept: a three-attempt no-rebuild qsemantic run
selected `candidate_vs_default=full:0.93x,smollm:1.00x`, so the policy avoided
the SmolLM-specific tiled-tail regression but still failed to improve both
geometries. Do not spend the next pass on static output-width gating; the useful
move remains a better work partition or a larger semantic command.
Evidence tag: semantic FFN sublayer throughput kernel; Metal now executes that exact command as a bounded one-dispatch diagnostic kernel; the default throughput lane remains the faster projection-pair product, down-projection residual add, then RMSNorm scale path; target dispatches `1` backend kernel.
It also names the hard performance fact directly: the structurally useful
two-dispatch command path is dispatch-neutral in the full model
(`command_dispatch=242->242`, `command_dispatch_reduced=no`), and the two-phase
prototype is also dispatch-neutral at the model level
(`two_phase_dispatch=242->242`, `two_phase_dispatch_reduced=no`) while adding
partial/finalize work. The single-dispatch diagnostic can reduce actual
dispatch count, but its tiled kernel is too slow. So the next target is not
"more command lowering" by itself; it is
`reduce_actual_dispatch_or_larger_semantic_sublayer`: either a real
throughput-preserving dispatch reduction or a larger semantic sublayer that
removes surrounding work instead of merely renaming it.
The obvious shared `TILE` retune is not that move. A focused June 24, 2026
experiment rejected both sides: `TILE=16` made one isolated qrow-region ratio
look less bad but cut full-model Q8 prompt throughput roughly in half, while
`TILE=64` made the two-phase qrow-region path numerically wrong
(`max_abs_diff` around `0.54`). Keep the shared Metal matmul tile at `32`
for normal matmul. Row-chain tiled kernels now use a separate
`ROW_CHAIN_TILE=32`, which is behavior-preserving today and lets future
row-chain-only tile experiments happen without perturbing the normal
qmatmul/matmul path.
The two-phase partial kernel now also drops the unused scale buffer binding;
the finalize kernel still owns scale application, while the partial pass binds
only weight, input, residual, elementwise output, partial scratch, and params.
The direct `Linear -> LogSoftmax` CPU tail now shares the batched-linear BLAS
preference for plain dense projection, while the fused classifier tail keeps a
native row log-softmax path with a fast vector exp approximation and a measured
`N=32` specialization for the current classifier shape. On the focused PyTorch
gap microscope this reduced the native module hot path to roughly
`0.013ms` for `linear_batched` and `0.039ms` for
`log_softmax_classifier_batched` under the 8ms minimum timing window. That is
good module-vs-eager evidence, but the PyTorch comparison remains a real soft
spot: the stabilized three-attempt rerun reported
`ratio_median=linear_batched:0.30x,log_softmax_classifier_batched:0.22x`.
The public Program plan still reports `linear|log-softmax`; the shortcut is an
implementation detail of the CPU Session hot path.
The PyTorch comparison now consumes exact `ZGML_MODULE_BENCH_JSON` module-bench
timings (`hot_execute_into_ms`) before falling back to the human summary line,
because rounding sub-`0.05ms` hot-path timings to four decimals can move
the reported PyTorch ratio by enough to obscure small real kernel changes. Both
the zgml module bench and the embedded PyTorch comparison now also run each
tiny lane until an 8ms minimum timing window is reached, so sub-0.01ms paths are
not judged from a single too-short iteration batch.
Generic Sessions now also expose `prepareExecuteInto(output, { input })`, which
validates and binds the caller-owned buffers once, then returns a tiny runner
for repeated hot-loop execution. The module bench emits
`prepared_execute_into_ms` beside the normal `hot_execute_into_ms`; the focused
June 23, 2026 microscope showed this is useful API shape but not the main
PyTorch fix by itself (`linear_batched` moved only from about `0.0108ms` to
`0.0104ms`, and `log_softmax_classifier_batched` from about `0.0433ms` to
`0.0428ms`). That evidence points the next PyTorch catch-up work below the StepParams facade:
native kernel shape, FFI call granularity, and larger fused
Programs matter more than further TS object parsing polish on these lanes.
The public `compile.compileForInference(model, { inputShape })` helper now wraps
the same Program/Session path into a frozen handle with `forward`, `call`,
`__call__`, `into`, `prepareInto`, `dispose`, and `free`, so the simple
inference API and the allocation-conscious hot path are the same object.
For module targets, that helper binds with `Program.bindModule(model)`. For lazy
graph targets, the same helper can now accept explicit Program bindings as its
third argument and bind with `Program.bind(bindings)`, so named-parameter lazy
graphs can use the same first-contact handle instead of forcing users down to
manual `compile -> bind -> Session` ceremony.
The module-shaped native inference sugar now sits one level above this helper:
`nn.native(model, { inputShape, backend })`, `nn.inference(...)`, and
`model.native(...)` compile and bind through the same Program/Session substrate
but keep the first-contact JS/TS experience centered on ordinary model objects.
The prepared runner now reaches one layer lower than the facade when an adapter
can help: Node and Bun Session ops expose `prepareStepSession`, so
`prepareExecuteInto` can precompute direct FFI lengths/records and, on Bun,
reuse ABI words and result storage instead of rebuilding them every call. The
focused Node microscope moved to about `0.0093ms -> 0.0089ms` for
`linear_batched` and `0.0314ms -> 0.0292ms` for
`log_softmax_classifier_batched`; Bun's no-rebuild dist smoke also passes. This
is still incremental, but it is the right direction for host FFI granularity.
The PyTorch comparison now names its zgml timing source with
`zgml_timing=prepared_execute_into_ms` and consumes that exact JSON field by
default, because repeated inference should compare against the prepared hot
runner rather than the more defensive one-shot `executeInto` facade. The
focused three-attempt rerun still shows the true remaining gap:
`ratio_median=linear_batched:0.31x,log_softmax_classifier_batched:0.23x`, with
best selected-attempt timings around `linear_batched=zgml:0.0090ms
pytorch:0.0035ms` and `log_softmax_classifier_batched=zgml:0.0301ms
pytorch:0.0072ms`. That confirms prepared host calls help but do not change the
next target: the dense/log-softmax native kernels and dispatch granularity must
get better.
The next native cuts did exactly that for the tracked classifier shape. Earlier
experiments tested a forced native small-direct linear path before the row
log-softmax pass; the current measured path now prefers the same BLAS-backed
batched linear policy as `linear_batched`, then runs the measured `N=32` row
log-softmax specialization for the fused direct `Linear -> LogSoftmax` tail.
Under a 50ms timing window this moved `log_softmax_classifier_batched` prepared
time from roughly `0.0310ms` to the `0.008-0.010ms` range. The focused
three-attempt PyTorch rerun stayed noisy but moved the tracked soft spot from far
behind to roughly parity:
`ratio_median=linear_batched:1.30x,log_softmax_classifier_batched:0.95x`, with
the selected attempt at `linear_batched=zgml:0.0021ms pytorch:0.0027ms` and
`log_softmax_classifier_batched=zgml:0.0077ms pytorch:0.0076ms`. This does not
prove broad PyTorch superiority, but it does prove the right kind of
improvement: exact-shape native execution beats wrapper work for these micro
gaps.
The latest fresh-native three-attempt PyTorch gap microscope keeps the honest
line: `linear_batched` is ahead of PyTorch (`ratio_median=linear_batched:1.25x`)
while `log_softmax_classifier_batched` remains a real soft spot around
`0.91x` median (`zgml:0.0078ms`, `pytorch:0.0071ms`). The next useful cut is
therefore the row log-softmax tail itself, not another host wrapper tweak.
Two tempting row-tail shortcuts were measured and rejected: a bind-time scratch
buffer plus macOS `vvexpf` batch for the `N=32` shifted exponentials produced
no stable median improvement, and a range-reduced no-division fast-log
approximation passed the focused correctness tolerance but did not improve the
fresh-native PyTorch gap. The next attempt should avoid those isolated math
swaps and instead change the whole row-tail shape: fewer row passes, a better
native classifier-tail kernel, or a backend path that amortizes the row
normalization differently.
The first whole-tail cut keeps the public two-op Program shape but removes a
private row pass for the tracked `M=128,N=32,K=64` classifier: the direct CPU
path now runs the BLAS-backed linear without bias, then folds the bias vector
into the `N=32` row log-softmax normalization itself. The focused native module
bench with a 100ms timing window measured
`log_softmax_classifier_batched` at `prepared_execute_into_ms=0.00759ms`, but
the fresh-native three-attempt PyTorch microscope remains a miss:
`ratio_median=linear_batched:1.24x,log_softmax_classifier_batched:0.90x`, with
the selected attempt at `log_softmax_classifier_batched=zgml:0.0076ms
pytorch:0.0071ms`. This is a useful simplicity/perf direction because it
removes a row pass without changing the frontend contract, but the remaining
PyTorch gap is still real.
The next amortization idea was also measured and rejected: a bind-time scratch
buffer plus macOS `vvlogf` over the 128 per-row denominators compiled and passed
the focused correctness filter, but the 150ms local classifier microscope moved
the prepared path to `0.00765ms`, worse than the simpler fused-bias row path.
The June 24, 2026 fast-log polynomial shortcut was also rejected: dropping the
`y9 / 9` term from `fastLogPositiveApprox` passed the focused direct
log-softmax row correctness tests with a `2e-5` standalone log guard, but the
fresh-native three-attempt PyTorch microscope stayed unchanged at
`ratio_median=log_softmax_classifier_batched:0.91x` and
`zgml:0.0076ms pytorch:0.0069ms`.
Two follow-up probes on June 24, 2026 were also rejected. Widening the fused
small direct `N=32` classifier path to the tracked `M=128,K=64` shape with an
extra 16-row unroll lost to the existing BLAS-backed linear plus fused-bias row
log-softmax path: the fresh-native three-attempt microscope selected
`zgml_vs_pytorch=0.91x` with `ratio_median=0.90x`. Adding a private
prepared-direct C ABI entry to skip repeated direct-call length arguments also
failed to move the measured prepared path, selecting `0.92x` with
`ratio_median=0.91x`. Do not spend the next pass on larger small-direct row
unrolls or prepared ABI bypasses for this lane.
The post-BLAS row tail was checked too: unrolling
`logSoftmaxRowsInPlaceBias32` in four-row chunks compiled and ran, but the
fresh-native three-attempt microscope regressed the selected attempt from the
baseline `0.94x` sample to `0.92x`, with `ratio_median=0.91x`. That rules out
simple row-loop unrolling as the missing PyTorch-parity move for this lane.
That reinforces the current rule for this soft spot: isolated vForce swaps,
shorter scalar approximations, wider small-direct row unrolls, post-BLAS row
tail unrolls, and prepared ABI bypasses are not the missing move; the next credible
improvement needs a more substantial classifier-tail kernel or a backend path
that changes the row-normalization shape. A steadier 150ms-window,
three-attempt PyTorch rerun of only the
log-softmax classifier lane confirms the target: `ratio_median=0.89x` with the
selected attempt at `zgml:0.0077ms` and `pytorch:0.0069ms`.
The broad PyTorch replacement loop is now named explicitly as
`dev:perf:pytorch:broad*`. An earlier ten-lane sample showed nine lanes ahead
of PyTorch while `log_softmax_classifier_batched` remained the only miss
(`0.93x` one-attempt; steady focused median around `0.92x-0.97x` depending on
the local timing sample). `bench:status` now promotes broad steady evidence to
the selected `pytorch-results:` scoreboard when present, while narrower focus
and one-lane microscopes remain visible as diagnostic lines. A June 24, 2026
policy experiment that forced the
`N=32` path through the small direct linear+bias kernel before the row
log-softmax was also rejected: the module bench regressed
`prepared_execute_into_ms` to about `0.0112ms`, and the three-attempt PyTorch
comparison still missed median parity. A follow-up 4th-degree vector-exp
experiment for the `N=32` row tail also preserved correctness but failed to move
the steady target: the module bench stayed around `prepared_execute_into_ms =
0.0098ms`, and the three-attempt PyTorch lane still missed median parity
(`ratio_median=0.94x`). A June 24 follow-up kept the public
`linear|log-softmax` Program plan but replaced the private row-tail path with a
fused small-direct classifier kernel: it accumulates the `N=32` linear bias and
weights into two 16-wide column vectors, computes the stable row max/exp sum,
and writes log-softmax output directly. That removes the separate linear-output
row pass while preserving the allocation-free direct Session path and
C ABI coverage for the `M=128,N=32,K=64` comparison shape. Later row-tiled helpers
improved weight reuse for the same direct path without changing the public
Program evidence, and the current helper removes the private 4/2/1-row
duplicate normalization bodies while trying the hot `M=128` path as 8-row
tiles first. The final focused 150ms module microscope for that helper measured
`prepared_execute_into_ms=0.00766ms`, a local improvement from the previous
`0.00950ms` sample, but this remains a focused module-path improvement rather
than a PyTorch parity claim. The latest fresh-native three-attempt gap artifact after the
four-row tile was narrower but still honest:
`ratio_median=linear_batched:1.22x,log_softmax_classifier_batched:0.95x`, with
the selected miss at `log_softmax_classifier_batched=zgml:0.0073ms
pytorch:0.0069ms`. Do not claim PyTorch broad parity yet; keep the fused
small-direct row kernel because it improves the module prepared path, but the
next move still needs a stronger classifier-tail kernel or a path that
amortizes row normalization differently.
The PyTorch comparison workload now mirrors the zgml module-bench data scales
for the tracked lanes instead of timing nearby-but-different tensors. The
linear, MLP, RMS/SiLU FFN, softmax classifier, log-softmax classifier, and token
head comparisons now use the same `values(length, scale)` shapes as their zgml
counterparts before ratios are computed. That makes the comparison less likely
to hide value-distribution-dependent softmax/log-softmax timing. Under the
corrected comparison, an earlier fresh-native broad sample remained 9/10:
`ratio_median=linear_batched:1.23x,lazy_matmul_add_gelu_batched:2.81x,lazy_mlp_batched:1.76x,lazy_rms_silu_ffn_batched:1.63x,max_pool2d_batched:8.57x,avg_pool2d_batched:5.34x,rms_gelu_linear_batched:2.94x,softmax_classifier_batched:1.20x,log_softmax_classifier_batched:0.91x,lazy_token_head_batched:1.08x`.
The refreshed six-lane focus sample from that stage was also honest rather than green:
`lane_pass=5/6`, `ratio_median=linear_batched:1.23x,lazy_matmul_add_gelu_batched:2.83x,lazy_rms_silu_ffn_batched:1.63x,rms_gelu_linear_batched:2.95x,log_softmax_classifier_batched:0.91x,lazy_token_head_batched:1.09x`,
with the best selected log-softmax attempt at
`log_softmax_classifier_batched=zgml:0.0076ms pytorch:0.0074ms`. The attempted
`K`-loop unroll for the fused four-row small-direct classifier-tail kernel did
not move the corrected median and was not kept. The next PyTorch catch-up move
is still a stronger classifier-tail kernel or a path that changes
row-normalization amortization, not a timing-harness artifact. The current
eleven-lane broad artifact supersedes this stage after the direct projection
tail and Conv2d path work.
A fresh exact-shape direct-tail recheck then let the fused native
`M=128,N=32,K=64` classifier helper cover the focused PyTorch comparison shape
instead of forcing that case through the BLAS-plus-row-normalization path. The
150ms three-attempt microscope moved the selected ratio only modestly
(`0.91x -> 0.93x`, median `0.92x`) with zgml prepared times around
`0.00755ms`; the lane remains a miss, but the direct path is no longer worse
than the fallback on the local ReleaseFast evidence. Keep this as a small
classifier-tail policy correction, not a parity claim.
The next focused pass moved the winning fix one layer lower: the CPU execution
tape now recognizes `matmul -> repeat(bias) -> add(bias) -> logsoftmax` as a
single dense projection command and runs the classifier tail directly into the
final output with vectorized bias+log-softmax normalization. The public
KernelPlan remains `linear|logSoftmax`, but Program inspection for the focused
classifier shape now reports one projection command instead of a projection
command plus a row op. Fresh-native 150ms evidence against PyTorch `2.12.1`
flipped the former soft spot from a miss to a stable win:
`log_softmax_classifier_batched` selected at `1.48x` with
`ratio_median=1.43x`, and the six-lane focused comparison now passes
`lane_pass=6/6`, `median_lane_pass=6/6`, with worst selected lane
`linear_batched:1.24x`. The eleven-lane replacement comparison now includes a
lazy `Conv2d -> ReLU` lane and is green on fresh-native three-attempt 80ms
evidence: `lane_pass=11/11`, `median_lane_pass=11/11`, worst selected lane
`softmax_classifier_batched:1.16x`, and
`ratio_median=linear_batched:1.24x,lazy_matmul_add_gelu_batched:2.84x,lazy_mlp_batched:1.80x,lazy_rms_silu_ffn_batched:1.67x,lazy_conv2d_relu_batched:1.19x,max_pool2d_batched:8.70x,avg_pool2d_batched:4.98x,rms_gelu_linear_batched:2.73x,softmax_classifier_batched:1.16x,log_softmax_classifier_batched:1.43x,lazy_token_head_batched:1.65x`.
The current PyTorch next item is therefore `none`; the next performance frontier
moves back to Q8/ggml semantic throughput.
The model-free stencil-only debug microscope now also prints both decode and
prompt row-chain/projection-chain diagnostics before enforcing its p128 stencil
hash contract, so a stale decode hash no longer hides the prompt-side frontier
shape needed for Q8 projection-chain work.
The source-current stencil shape gate parses those `ZGML_STENCIL_JSON` rows and
checks prompt/decode command counts, covered ops, patch holes, projection-chain
shape, cache-group shape, and nonzero runtime/command stencil hashes without
waiting for a hard ggml artifact to pass on a quiet machine.
Focused module Program benchmarks also build the native C ABI in ReleaseFast
first, so microscope results do not silently compare against a stale Debug
dylib. `scripts/native_freshness.cjs` owns the shared freshness policy for both
the module Program and PyTorch comparison benches: direct `:run` reruns check
`zig-out/lib/libzgml_c.*` against native sources, print `native=fresh` in
benchmark output, and require bench-specific `*_ALLOW_STALE_NATIVE=1` flags for
explicitly stale diagnostics.
Frontier, Q8 prompt candidate, and ggml comparison rebuild commands now
also force `-Doptimize=ReleaseFast`; their `:run` variants are explicitly
artifact reruns after that benchmark-grade build, except `bench:ggml:parity:run`
which deliberately leaves `BENCH_BUILD_ZGML` at the `bench_vs_ggml.sh` default
so the hard ggml rerun rebuilds ReleaseFast unless the caller explicitly opts
into reuse with `BENCH_BUILD_ZGML=0`. The frontier binary now also
accepts `BENCH_FRONTIER_FILTER=<label-substring>` for microscope loops; the
named row-chain scripts use `BENCH_FRONTIER_FILTER=qrow` so the remaining tiled
row-chain kernel work can skip unrelated elementwise/matmul/norm frontier
families while preserving the full unfiltered release gate. The x7 region
scripts use `BENCH_FRONTIER_FILTER="qrow region"` to hit the same anchored
region-command shape that full-model prompt scheduling selects, which keeps the
two-phase kernel loop tight while still proving the real region path. The frontier
checker now also has a focused `qrow region` branch, so
`dev:perf:frontier:row-chain-region` can validate the exact two-phase row-chain
dispatch/profile shape and correctness without requiring unrelated frontier
labels; its speed output is diagnostic until the two-phase kernel earns a real
throughput floor, and repeated attempts choose the best minimum throughput across
full-prefill and SmolLM-prompt rows while still failing only on correctness or
profile drift. The frontier
qproj scripts use `BENCH_FRONTIER_FILTER=qproj` so projection-chain/group work
can check the current Q8 prompt frontier without paying for unrelated rows,
(norms, row-chain candidates, and matmuls). The matching
`dev:perf:frontier:qproj` loop adds the incremental ReleaseFast rebuild, so
quantized projection-chain/layout changes can iterate against the checked
qproj gate before escalating to the full frontier or ggml gates; it is the
checked qproj gate in the fast dev loop. That qproj dev loop defaults to three
attempts rather than one because the prompt-tile lane is noisy enough that a
single roughly-three-second sample can fall below the `0.95x` floor while the
x7 executable region proof remains healthy; three attempts keeps iteration fast
without turning noise into a false architecture signal. The frontier
qsemantic scripts use `BENCH_FRONTIER_FILTER=qsemantic` and default the
checked dev loop to one attempt because the profile assertions are the value:
they prove the current 3-dispatch throughput boundary and the opt-in one-command semantic
target (`target_shape_commands=1`,
`target_semantic_ffn_sublayers=1`,
`target_runtime_backend_dispatches=1`) without requiring the full Q8 prompt
artifact loop on every kernel edit. It also prints
`target_throughput_status=diagnostic_needs_throughput_kernel` when the
one-dispatch semantic target is structurally correct but slower than the
current throughput path, so dispatch-count evidence cannot be mistaken for a
throughput promotion signal. The qsemantic microscope now also measures the
middle policy shape, `semantic pair_row_chain_single_dispatch`: it preserves the
same two-command semantic shape and, after whole-command execution was enabled
for projection row-chain command streams, now reports
`single_dispatch_row_chain_dispatch_reduced=yes`,
`single_dispatch_row_chain_blocker=none`, and
`single_dispatch_throughput_status=dispatch_reduced_but_throughput_diagnostic`.
The default pair-row-chain path now records the down-projection row-chain command
as attempted and dispatched (`runtime_projection_row_chain_attempts=1`,
`runtime_projection_row_chain_dispatches=2`) while keeping
`runtime_backend_dispatches=3`; the single-dispatch diagnostic lights up the
tiled leaf (`qmatmul_row_chain_tiled_count=1`) and drops to
`runtime_backend_dispatches=2`. The qsemantic gate now prints the tiled-work
shape directly: full-prefill single-dispatch/two-phase row-chain paths both
report `qmatmul_row_chain_tiled_row_tile_groups=4`,
`qmatmul_row_chain_tiled_n_tiles=16`,
`qmatmul_row_chain_tiled_serial_tile_loops=64`,
`qmatmul_row_chain_tiled_partial_slots=2048`, and
`qmatmul_row_chain_tiled_scratch_capacity=65536`, with
`qmatmul_row_chain_tiled_finalize_tile_groups=64` and
`qmatmul_row_chain_tiled_finalize_elements=65536`; SmolLM prompt reports
`qmatmul_row_chain_tiled_row_tile_groups=4`,
`qmatmul_row_chain_tiled_n_tiles=18`,
`qmatmul_row_chain_tiled_serial_tile_loops=72`,
`qmatmul_row_chain_tiled_partial_slots=2304`, and
`qmatmul_row_chain_tiled_scratch_capacity=73728`, with
`qmatmul_row_chain_tiled_finalize_tile_groups=72` and
`qmatmul_row_chain_tiled_finalize_elements=73728`. That is real
executable-stencil progress, but it is not a default promotion yet because the
tiled leaf still loses throughput on the SmolLM prompt shape. The same gate now
also exposes why the one-command semantic target is not the missing throughput
kernel yet: full-prefill reports `target_semantic_rows=128`,
`target_semantic_hidden=512`, and
`target_semantic_row_serial_dot_ops=786432`, while SmolLM prompt reports
`target_semantic_rows=128`, `target_semantic_hidden=576`, and
`target_semantic_row_serial_dot_ops=995328`. The qsemantic gate now also prints
the total trapped row-serial work:
`target_semantic_total_row_serial_dot_ops=100663296` for full-prefill and
`target_semantic_total_row_serial_dot_ops=127401984` for the SmolLM prompt
shape. The command shape is ideal, but the implementation still performs that
row-serial dot workload inside one threadgroup per row. A later throughput
candidate pass split the semantic FFN kernel width away from the scalar
row-chain diagnostic width: `QMATMUL_ROW_CHAIN_THREADS` stays `256`, while the
semantic throughput kernel now uses `SEMANTIC_FFN_THREADS=512`. The rejected
`SEMANTIC_FFN_THREADS=128` probe made the one-dispatch semantic kernel slower
(`full_prefill:0.59x`, `smollm_prompt:0.44x`). The retained `512`-thread
version keeps the one-dispatch shape and improves the focused three-attempt
qsemantic throughput artifact to `full_prefill=0.95x` and
`smollm_prompt=0.83x`, with `semantic_ffn_sublayer_count=1`,
`semantic_tile_groups=192/216`, and zero row-chain tiled tail. This is still
diagnostic, not a default promotion: it shows that width helps, but the next
real win still needs to change the semantic kernel's work partitioning rather
than only retune threadgroup size. In the full-model Q8 prompt gate this same
lane still sits in the `semantic_speedup=0.95-0.98x` diagnostic band; the next
real win still needs to change the semantic kernel's work partitioning before
promotion is honest. Put plainly: the next real win still needs to change the semantic kernel's work partitioning.
A later fresh no-build throughput artifact made the work-partitioning problem
explicit: full-prefill can hit `width_lane_utilization_x1000=1000` and
`thread_lane_utilization_x1000=1000`, while the 576-wide SmolLM prompt shape
only reaches `width_lane_utilization_x1000=562` and
`thread_lane_utilization_x1000=611`. The serial math gap is only
`serial_gap=1.13x`, but the fixed-thread lane footprint is now split into
`width_slot_gap=2.00x` and `thread_slot_gap=1.80x`; the next semantic
throughput pass should therefore be a 576-aware work-partitioning/vectorization
change, not another blind
`SEMANTIC_FFN_THREADS` probe. The status router names that
shape `semantic_width_parallel_kernel` when fresh throughput evidence shows the
SmolLM width/thread utilization gap; until that kernel exists, the route still
runs the qsemantic throughput microscope. A 576-only
`SEMANTIC_FFN_THREADS=256` narrow
kernel probe reduced the measured thread-slot footprint
(`thread_slot_gap` roughly `1.80x -> 1.30x`) but made throughput worse
(`smollm_prompt=0.67x`). A follow-up 384-thread mid-width probe preserved more
parallelism and improved lane utilization to about `777/1000`, but still lost
throughput (`smollm_prompt=0.69x`). A 768-thread exact-width probe raised the
SmolLM width utilization to `750/1000` and total lane utilization to about
`800/1000`, but still failed to beat the default (`smollm_prompt=0.99x`) and
regressed full-prefill to `full_prefill=1.12x`; width fit alone is not the
missing architecture. The retained implementation stayed on the 512-thread
kernel until the partitioning change also preserved enough parallelism. The
retained fix is a four-way unroll of the semantic gate/up and down inner dot
loops inside the 512-thread kernel. That keeps the one-dispatch shape, preserves
the `thread_lane_utilization_x1000=1000/611` evidence for full-prefill/SmolLM,
and moves the focused three-attempt throughput gate to
`full_prefill=1.99x`, `smollm_prompt=1.04x`, `gate=ready`. A follow-up
three-attempt Q8 prompt viable run refreshes the accepted steady prompt evidence:
`semantic_speedup=1.18x`, `semantic_median=1.01x`, `semantic_worst=0.99x`,
`semantic=promoted`, while keeping the 151-command shape and zero fallbacks. The
qsemantic gate now reports
`target_vs_default=full_prefill:...x,smollm_prompt:...x` so the dispatch
frontier can be separated from the throughput frontier. The throughput-only
qsemantic artifact also records selected, median, and worst speedups for
full-prefill and SmolLM prompt, matching the Q8 prompt promotion discipline:
one lucky semantic attempt is evidence for a hypothesis, not a promotion signal.
reduction is always interpreted against the current throughput path; a fresh
no-rebuild run still prints both ratios, and the target remains diagnostic
until both geometries beat the default path across repeated attempts.
It also reports the tiled work shape the real throughput kernel must expose:
full-prefill has `target_semantic_tile_parallel_groups=192` and SmolLM prompt
has `target_semantic_tile_parallel_groups=216`, corresponding to four row
groups and 16/18 hidden-output tiles at the 32-wide tile shape. The latest
qsemantic microscope also prints the row-serial gap per tile-parallel group:
`target_semantic_row_serial_dot_ops_per_tile_parallel_group=4096` and
`target_semantic_total_row_serial_dot_ops_per_tile_parallel_group=524288` for
full-prefill, versus
`target_semantic_row_serial_dot_ops_per_tile_parallel_group=4608` and
`target_semantic_total_row_serial_dot_ops_per_tile_parallel_group=589824` for
the SmolLM prompt geometry. Those are the concrete work-shape numbers the
next throughput kernel has to collapse, not just descriptive counters.
The qsemantic microscope now promotes that separation into the main semantic
command path. `promptProjectionRowChainCommand()` recognizes the FFN/residual
norm as one `semantic command` with `shape_commands=1`,
`shape_semantic_ffn_sublayers=1`, `shape_covered_ops=9`, and
`shape_saved_dispatches=8`, but deliberately lowers through the current
parallel pieces, so it reports `runtime_backend_dispatches=3`,
`runtime_semantic_ffn_dispatches=3`, and
`semantic_ffn_sublayer_count=0` instead of using the row-serial semantic
kernel. The gate prints
`semantic_command_status=preserves_default_work_shape` when the one-command
semantic lowering keeps the current default work shape. This is the useful
near-term architecture: one semantic library command is now the promoted path,
while the one-dispatch semantic target remains an explicit diagnostic until the
true tiled semantic kernel exists.
At that stage, the named `promptSemanticFfnSublayerThroughputCandidate()`
policy kept the same one-command semantic shape but swapped the
down-projection/residual/RMSNorm tail to the existing two-phase tiled row-chain
leaf. The qsemantic gate reported that as `throughput_candidate_status=...` with
`throughput_candidate_vs_default=full_prefill:...x,smollm_prompt:...x`,
`full_prefill_throughput_candidate=...x` and
`smollm_prompt_throughput_candidate=...x`; fresh no-rebuild runs now show the
candidate preserving correctness and tiled profile shape, with single-attempt
throughput noisy enough to move between `mixed_tiled_tail_diagnostic` and
`ready`. A three-attempt qsemantic rerun found the candidate beating SmolLM
prompt on the best evidence attempt while still missing full-prefill, so it is
not yet an automatic promotion signal: it needs repeated attempts and the full
Q8 prompt candidate gate before it can replace the default semantic command
path. It does, however, give the next Metal pass a checked semantic-command lane
instead of only separate row-chain experiments.
The qsemantic microscope now also prints
`throughput_candidate_vs_two_phase=...`, which separates semantic-command
overhead from the tiled row-chain tail itself. The latest three-attempt
source-current rerun after tiling the two-phase finalize pass showed the
throughput candidate slightly ahead of the standalone two-phase tail
(`full_prefill:1.02x`, `smollm_prompt:1.02x`) and roughly even with the default
semantic command path (`full_prefill:0.98x`, `smollm_prompt:1.01x`). That is
real movement, but still not a promotion: the next implementation work remains
a true semantic throughput kernel or a row-chain tail that clears the steady
full-model gate, not wrapper overhead around the semantic command.
A June 24, 2026 tiled-finalize pass changed the two-phase row-chain finalize
kernel from one threadgroup per row tile to a `(row_tile, col_tile)` grid. The
fresh qsemantic steady gate stayed correct and moved the mixed tiled-tail
candidate from below-default diagnostic evidence toward parity with the current
default; the fresh steady full-model Q8 prompt viable gate moved semantic
median throughput to about `0.99x` with best evidence near `1.00x`, while still
leaving `semantic_throughput_ready=off`. Keep this kernel shape, but do not
promote it by default until a steady full-model run clears the readiness floor.
A later source-current qsemantic steady run selected a stronger mixed tiled-tail
attempt and now reports `throughput_candidate_status=ready` with
`candidate_vs_default=full:1.02x,smollm:1.13x` and
`candidate_vs_two_phase=full:1.60x,smollm:1.05x`. The status line also exposes
the semantic target decomposition: full-prefill is `target_shape=4x16x16`, and
SmolLM prompt is `target_shape=4x18x18`. The current candidate still only
exposes `candidate_tiles=full:64,smollm:72` against target
`target_tiles=full:192,smollm:216`, so the next Metal pass remains the true
semantic FFN throughput kernel that collapses the 3.00x tile gap and the
`4096`/`4608` serial dot ops per target tile group, not a default policy flip
based only on frontier evidence.
A full Q8 prompt follow-up confirmed why that distinction matters. A
one-attempt viable run reported `semantic_throughput=ready` at
`semantic_speedup=1.03x`, but the three-attempt steady run stayed diagnostic:
`semantic_best=1.33x`, `semantic_median=1.01x`, and
`semantic_worst=0.77x`, with zero fallback and the expected
`semantic_command=241->151`. The next-performance selector should therefore
hand off from qsemantic to the Q8 prompt gate once frontier says
`candidate=ready`, but the default policy still needs either steadier full-model
semantic evidence or the true semantic FFN throughput kernel before promotion.
The Q8 prompt steady path now supports paired default baselines and
`dev:perf:next:steady` enables them for this lane. With paired baselines, the
selected three-attempt Q8 prompt evidence is less ambiguous:
`pair_defaults=yes`, `baseline_noise=1.10x`, `semantic_best=1.01x`,
`semantic_median=0.97x`, and `semantic_worst=0.95x`. That narrows the diagnosis:
the semantic path is structurally right and roughly parity, but it is not hiding
a stable throughput win behind the earlier unpaired measurement swing.
The Metal runtime now checks requested output byte spans, not just output buffer
ids, before forcing semantic and row-chain intermediate materialization. That is
the right ABI-level precision for arena-backed buffers, but the follow-up
three-attempt Q8 prompt semantic gate still stayed diagnostic:
`semantic_best=1.11x`, `semantic_median=1.00x`, `semantic_worst=0.92x`, and
`spills=30`. So span-aware output reads are useful cleanup, not the missing
throughput move. Keep the next target on a larger semantic FFN throughput kernel
or on absorbing the live residual use, rather than promoting the current
semantic tail path.
The row-chain spill profile now separates total spills from Program-output
driven spills. A source-current one-attempt Q8 prompt semantic probe reported
`semantic_spills=30` and `semantic_output_spills=0`, confirming that the
remaining materialization is caused by command/liveness structure rather than
logits or KV-cache output bindings. That makes the next perf move sharper:
absorb the residual user into a larger semantic command or replace the
two-dispatch tail with a true throughput kernel; do not spend another pass on
output binding policy. The Q8 prompt artifact/status path now exposes
`semantic_bridges`, derived from
`program_command_shape_projection_row_chain_semantic_residual_bridges`, so that
residual-bridge opportunity is measured directly before the next semantic
command is designed.
The latest selected three-attempt Q8 prompt artifact now records the promotion
state directly: `status=promoted-default`, `semantic=promoted`,
`default_policy=semantic-promoted`, `command_command=151->151`, zero fallback,
and the same semantic row-chain pressure (`semantic_spills=30`,
`semantic_spill_input=17280`, `semantic_output_spills=0`, plus
`semantic_bridges` readback for residual-bridge shape). Treat this as the
semantic command becoming the Metal scheduled Q8 prompt default, not as ggml
parity. The next bottleneck is the remaining model-width spill/work shape: absorb
the residual user into a larger semantic command if bridge evidence exists, or
replace the two-dispatch tail with a true throughput kernel.
The first bridge-absorbing command now exists as a measured candidate, not a
default promotion: `semantic_ffn_sublayer_with_input_row_chain` covers the 14-op
`projection_row_chain + semantic_ffn_sublayer` window. A focused Q8 prompt probe
proved the desired command shape with zero fallback
(`semantic_command=151->121`, `semantic_projection_row_chain=30->0`,
`semantic_bridges=30->0`), while also proving that the conservative encoder is
not the final performance answer (`semantic_speedup=0.96x` in that one-attempt
probe). Keep the default on the 151-command semantic-promoted path until the
bridge command owns a true throughput kernel.
The qsemantic input-bridge microscope now separates two meanings of
"absorbed". The guarded one-dispatch
`qmatmul_semantic_ffn_input_bridge_f32` path is still useful as a shape proof,
but it is row-owned and serial over too much work: the focused profile exposes
`128` row threadgroups and `2,985,984` row-serial dot ops per row threadgroup.
It must not be preferred for the SmolLM `576 x 1536 x 576` bridge shape until it
is rewritten as a real tiled/width-parallel kernel. The retained input-bridge
candidate therefore uses the decomposed width-parallel path: input row-chain
plus the streamed SIMD-width semantic tail. A fresh three-attempt focused
artifact reports `absorbed=2.69x`, median `2.56x`, worst `2.52x`,
`runtime_dispatches=5`, `semantic_with_input_dispatches=5`,
`decomposed=1`, `decomposed_dispatches=5`, `decomposed_extra_dispatches=4`,
`decomposed_row_chain=2`, `decomposed_pair=1`, `decomposed_tail=2`,
`direct=0`, and `max_abs_diff=0.000001`.
The full-model semantic throughput candidate now deliberately fences off the
row-owned direct bridge while preserving the 121-command shape and zero
fallback. A fresh three-attempt Q8 semantic steady probe reports best
`semantic_speedup=1.12x`, median `0.99x`, worst `0.99x`,
`semantic_command=151->121`, `semantic_absorbed=30`,
`semantic_absorbed_dispatch=150`, `semantic_absorbed_split=5.00`,
`semantic_direct=0`, and `semantic_fallback_dispatch=90`. That is a real
structural improvement but still not a model-level throughput promotion:
dispatch remains `242->242`. The next model-level win must either make the
14-op input bridge a true tiled/width-parallel one-dispatch kernel, or otherwise
reduce the five-dispatch absorbed bridge without returning to row-serial work.
A measured hybrid lane tried that second shape as
`input_product + width_parallel_tail`: it reduced the absorbed input bridge from
five dispatches to three, but the product bridge was still row-owned and dropped
the focused Q8 absorbed result to `0.70x`. Do not add that kernel to the
production encoder unless the input product side is rewritten with real
width/K parallelism; fewer dispatches alone is not a valid promotion criterion.
After the one-dispatch semantic throughput kernel became a real measured lane,
the full-model default was kept on `promptProjectionRowChainCommand()` while
`--metal-prompt-semantic-throughput-candidate` remains the explicit diagnostic
flag. Do not couple those paths again until the one-dispatch kernel beats the
semantic command default across repeated Q8 prompt and ggml evidence.
The row-chain spill profile also tracks spilled input width. A source-current
semantic quick probe reported `semantic_spill_input=17280` for `30` spills,
so the average spilling row-chain is `K=576`. That identifies the materialized
chain as the model-width attention/output-projection residual feeding the next
FFN, not the `K=1536` FFN-down residual tail. The larger semantic command should
therefore bridge attention-output residual plus following FFN gate/up inputs, or
otherwise make that model-width residual materialization cheap enough to keep.
A June 24, 2026 shape-gated policy experiment tried limiting the two-phase
semantic throughput tail to the 512-wide full-prefill shape so the 576-wide
SmolLM prompt shape would preserve the semantic command but skip the tiled tail.
That fixed the SmolLM candidate-vs-default ratio to about `1.00x`, but the
three-attempt qsemantic microscope selected a full-prefill miss around `0.90x`.
That experiment was rejected; use
`dev:perf:frontier:qsemantic:steady{,:run}` before trusting any one-attempt
qsemantic promotion signal.
A June 24, 2026 tail-loop micro-kernel experiment also rejected the tempting
local cleanup of replacing the final row-chain scale loop's flattened
`i / N`/`i % N` indexing with nested row/column loops. It preserved correctness,
but the three-attempt full Q8 prompt viable gate still left the semantic lane
diagnostic (`semantic_median_speedup=0.87x`, noisy attempts `2`) while the
qsemantic microscope stayed mixed (`full_prefill:0.97x`,
`smollm_prompt:1.03x`). Treat that as another sign that the next win must change
the work shape of the semantic FFN/down/residual/norm kernel, not polish the
existing tiled row-chain tail.
A June 24, 2026 quant-scale-index micro-kernel experiment tried replacing hot
Metal dequant expressions such as `w_idx / block_size` with a uniform
`block_size == 32 ? w_idx >> 5 : w_idx / block_size` helper across quantized
matmul, row-chain, pair, and semantic kernels. It compiled and preserved
correctness in the qsemantic microscope, but it did not improve the target:
the fresh one-attempt qsemantic lane moved the throughput candidate to
`full_prefill:0.91x,smollm_prompt:0.94x` versus default, while the
one-dispatch semantic target remained around `0.28x/0.23x` versus default. The
change was reverted. Do not retry scale-index arithmetic as the next Q8
semantic move; the missing win is still tile-parallel semantic work or a faster
row-chain leaf.
A narrower follow-up hoisted the `block_size == 32` branch only in the plain
`qmatmul_f32` and `qmatmul_elementwise_f32` tile-load loops, so the Q8
projection-chain path could use `w_idx >> 5` without the generic helper branch.
The three-attempt qproj microscope still did not improve: the temporary run
reported `projection_chain_full_prefill=0.99x`,
`projection_chain_smollm_prompt=1.02x`, and region proof at
`1.29x/1.24x`, while a post-revert source-current run returned to
`projection_chain_full_prefill=1.00x`,
`projection_chain_smollm_prompt=1.01x`, and region proof at `1.39x/1.30x`.
That variant was reverted too. Treat scale-index specialization as exhausted
for now; the qproj lane needs better quantized projection-chain structure, not
a shift-vs-div rewrite.
A semantic-only follow-up tried the same idea inside the one-dispatch
`qmatmul_semantic_ffn_sublayer_f32` kernel, but kept it narrower than the broad
qproj rewrite: the encoder now requires the gate/up/down qweight block sizes to
be `32`, and the semantic shader uses `w_idx >> 5` for those scale lookups. It
compiled, preserved correctness, and turned the qsemantic throughput lane from
diagnostic into a real win:
`full_prefill=2.67x:median:2.66x:worst:2.65x` and
`smollm_prompt=2.60x:median:2.31x:worst:1.63x`. The full-model Q8 bridge path
also improved to `semantic_best=1.16x`, `semantic_median=1.00x`, and
`semantic_worst=0.99x`; the strict bridge gate correctly keeps it diagnostic
until worst-case non-regression clears. Treat broad scale-index rewriting as
closed, but keep the semantic block-32 specialization as a retained kernel
improvement.
A row-chain tiled follow-up tried to apply the same `w_idx >> 5` spelling to
the two-phase tiled row-chain kernels under a block-size-32 encoder guard. It
compiled but worsened paired Q8 semantic bridge evidence
(`semantic_median=0.97x`, `semantic_worst=0.94x`), so that path is rejected too.
Instead, the Q8 candidate artifact now records semantic single-dispatch
diagnostics. A quick semantic-lane probe showed
`semantic_single_dispatch_attempts=30`,
`semantic_single_dispatch_output_read_refusals=0`, and
`semantic_single_dispatch_block_size_refusals=0`; the full bridge path is not
blocked by requested intermediate outputs or Q8 block-size guards. The next
useful target is the compatibility/shape refusal that prevents the one-dispatch
semantic FFN kernel from replacing the two-phase row-chain tail in the full
bridge lane.
The follow-up refusal histogram made that specific: all `30` attempts refused
on `dim`, because the full SmolLM FFN hidden width exceeds the current
row-serial semantic kernel cap. The Q8 prompt artifact/status path now records
the refused shape directly:
`semantic_single_dispatch_dim_refusal_shape=k:576,h:1536,o:576,cap:1024`.
Temporarily raising `SEMANTIC_FFN_MAX_DIM` from `1024` to `2048` proved the
recognizer/encoder can select the single-dispatch path
(`semantic_dispatch=242->182`, `semantic_single_dispatch_refusals=0`), but
throughput fell to `0.62x`. That cap bump is rejected: the needed work is a
tile-parallel semantic FFN/residual/norm throughput kernel for the
`576 x 1536 x 576` FFN shape, not enabling the larger row-serial
single-dispatch kernel by default. A checked exact-shape frontier microscope
now exists for that target:
`npm run dev:perf:frontier:qsemantic:bridge{,:run}` writes
`frontier-qsemantic-bridge-*.json` and feeds `bench:status`; the raw terminal
microscope remains `npm run dev:perf:frontier:qsemantic:bridge:raw{,:run}`.
The same conclusion held after a narrower June 25, 2026 row-serial tuning
probe: reducing `SEMANTIC_FFN_THREADS` from `512` to `256` while widening the
non-input product scratch to `SEMANTIC_FFN_MAX_HIDDEN` compiled and preserved
bridge correctness, but the exact bridge microscope selected the one-dispatch
path at only about `1.00x`. That is below both the retained three-attempt bridge
scoreboard (`1.87x` selected, `1.97x` median) and the latest source-fresh
one-attempt bridge evidence (`2.23x`). Do not retry threadgroup-width tuning as
the SmolLM bridge fix; the missing implementation is still a tile/width-parallel
semantic FFN kernel with explicit staged product/partial work.
Its first
ReleaseFast runs measured
`qsemantic bridge-ffn m=128 h=1536 k=576 o=576 semantic throughput_candidate`
in the `1.82x-2.51x` range over staged execution with `max_abs_diff=0.000000`,
`shape_semantic_ffn_sublayers=1`, and `runtime_backend_dispatches=3`. This is
kernel-shape evidence, not a full-model promotion. The checked bridge artifact
now also prints a `width_target=` tuple for the future kernel:
`rows:128,hidden:1536,input:576,output:576,row_groups:4,hidden_tiles:48,output_tiles:18`,
plus product/output element counts and gate-up/down/total dot-work. A refreshed
bridge artifact now makes the scratch contract explicit too: the obvious fully
staged width-parallel down path needs `down_partial_elements=3538944`
(`down_partial_bytes=14155776`), which is `48.00x` the output-sized
`73728`-element buffer for this shape. The retained product buffer is only
`product_elements=196608` (`786432` bytes). The Metal runtime now allocates
and profiles a reusable backend-owned down-partial scratch buffer for semantic
commands with `hidden > SEMANTIC_FFN_MAX_DIM`; the refreshed bridge gate asserts
`semantic_width_scratch=candidates:1,bytes:14155776,...,down_partial_to_output:48.00`.
The runtime now keeps that future width target separate from the scratch it
actually allocates, and makes allocation follow the executable command policy:
ordinary semantic-command policy reserves no semantic-width scratch, while the
throughput-candidate bridge path reserves a precisely-sized
`runtime_capacity:9216` byte surface instead of the full `14155776` byte
down-partial target. The checked bridge artifact reports `runtime_uses`
cumulatively across benchmark repetitions and `runtime_bytes:9216`, where
`9216` bytes is the per-dispatch partial surface for the `2304` row-chain
partial slots.
The first kept width-parallel slice is a streamed SIMD-width partial kernel:
`qmatmul_row_chain_width_partials_f32` keeps the existing pair-plus-finalize
shape, but splits the down-projection K loop across four simdgroup width lanes
inside each row/output tile. A global down-partial prototype was rejected after
measuring only `1.17x` on the bridge lane; it paid too much memory traffic for
the `14.2 MB` partial surface. The kept kernel preserves the small
`runtime_bytes:9216` scratch contract and a fresh three-attempt bridge artifact
selected `2.40x` over staged execution with `max_abs_diff=0.000002`,
`runtime_backend_dispatches=3`, `semantic_pair_dispatches=1`, and
`semantic_tail_dispatches=2`.
Do not simply widen `ROW_CHAIN_WIDTH_LANES` to `8`: the resulting
`qmatmul_row_chain_width_partials_f32` prototype needed `49152` bytes of
threadgroup memory and Metal rejected pipeline creation against the `32768` byte
limit. The next width-kernel move must reduce per-threadgroup storage, split the
tile differently, or fuse dispatches without growing the tile-local arrays past
the hardware limit.
A narrower `ROW_CHAIN_WIDTH_LANES=2` / 256-thread probe compiled but also lost:
the three-attempt exact bridge gate selected `2.21x` and the input-bridge gate
selected `2.46x`, below the refreshed four-lane `2.40x` and `2.63x` evidence.
The current four-lane shape is therefore the local sweet spot until the kernel
changes its storage model or fuses a dispatch.
A June 27, 2026 semantic input-bridge probe tried reusing that same four-lane
width-partial row-chain for the model-width input projection (`K=576`) inside
the 14-op bridge command instead of reserving it only for the hidden-width
semantic tail (`K=1536`). It compiled and preserved correctness, but the
three-attempt input-bridge gate regressed to `absorbed=2.65x` versus the
retained `2.73x` evidence, with the same five-dispatch bridge shape
(`row_chain=2,pair=1,tail=2`). That path is rejected too: simply swapping the
first row-chain partial kernel to the SIMD-width spelling does not reduce
dispatches and does not improve the bridge. The next input-bridge move still
needs a true fused or differently staged width-parallel input kernel, not the
existing tail kernel applied to the model-width projection. The focused
input-bridge gate now fails steady attempts below the `2.45x` absorbed collapse
floor so severe correct-but-weaker probes are caught automatically.
The row-serial semantic down loops now use an eight-hidden-value unroll instead
of the older four-wide accumulation in both the plain semantic FFN and direct
input-bridge kernels. A fresh three-attempt input-bridge microscope kept
correctness (`max_abs_diff=0.000001`) and selected `absorbed=2.73x`
(`median=2.77x`, `worst=2.73x`) while the one-dispatch direct diagnostic moved
to `direct_serial=1.38x` (`median=1.38x`). This is useful cleanup for the
diagnostic path, not the missing architectural move: the direct bridge still
reports `2,985,984` row-serial dot ops per row threadgroup, so the next target
remains `semantic_with_input_width_parallel_kernel`.
A finalize-path probe then tested replacing
`qmatmul_row_chain_tiled_finalize_tiles_f32` with the coarser row-tile
`qmatmul_row_chain_tiled_finalize_f32` so the RMS reduction would be computed
once per row tile instead of once per output tile. It regressed the exact bridge
lane to `2.13x` versus the retained `~2.31x` evidence and was reverted; the
restored per-output-tile finalize path produced fresh checked bridge evidence at
`3.04x`. Do not rechase this finalize spelling until a new design proves both
bridge and full-model Q8 prompt stability.
The broader `dev:perf:competitive` runner now wraps the PyTorch, native eager,
qsemantic, full-model Q8 prompt viable, and cheap ggml smoke lanes behind
`BENCH_COMPETITIVE_LANES`, so a kernel edit can run only
`BENCH_COMPETITIVE_LANES=native_eager npm run dev:perf:competitive` for the
native eager bridge, `BENCH_COMPETITIVE_LANES=qsemantic npm run
dev:perf:competitive` for a fresh frontier artifact,
`BENCH_COMPETITIVE_LANES=q8_prompt npm run dev:perf:competitive` for the
full-model Q8 prompt lane, or the matching `:run` commands after artifacts are
already fresh. That keeps the daily competitiveness loop explicit without
forcing every local native-eager or qsemantic edit to pay the PyTorch,
full-model Q8, and llama.cpp smoke cost. Its default PyTorch lane is now the
eleven-lane broad replacement set with three attempts and 150ms timing windows,
and its default native eager lane keeps the normal-module native bridge visible
in the same product scoreboard.
For tight semantic kernel work, the raw variant scripts set
`BENCH_QSEMANTIC_VARIANTS=target` or
`BENCH_QSEMANTIC_VARIANTS=throughput_candidate` so the frontier harness times
only the staged baseline plus the selected semantic target. Use target-only for
one-dispatch row-serial diagnostics and throughput-candidate-only for the mixed
tiled-tail path that should feed the next Q8 throughput kernel.
The qsemantic checker writes ignored JSON artifacts under
`bench-results/frontier/frontier-qsemantic-*.json`, emits a
`FRONTIER_BENCH_JSON` summary line, and `bench:status` reads the latest artifact
back as `frontier-results:`. That makes the semantic frontier durable without
promoting it to a throughput claim: the artifact records whether the selected
attempt is still diagnostic, what the throughput-candidate status is, and the
next target before any model-level speed claim is made. The status readback also
prints target and candidate tile-group counts so the next kernel edit is pointed
at reducing serial row-dot work, not merely reducing dispatch count.
The next implementation target remains the
`semantic_ffn_sublayer_throughput_kernel` or a faster tiled row-chain leaf, not
another command policy toggle.
`dev:perf:next{,:run}` now preserves that priority: when full Q8 prompt status
names `semantic_width_parallel_kernel`, it routes to the exact qsemantic bridge
microscope even if the adjacent input-bridge artifact also advertises
`semantic_with_input_width_parallel_kernel`. Input-bridge still has an explicit
lane, but it no longer steals the next-perf loop from the full-model prompt
blocker.
Use it when changing projection-pair, row-chain, residual, RMSNorm, or
semantic-sublayer scheduling, then escalate to `dev:perf:q8-prompt:viable` and
the full Q8 prompt candidate gate before making a model-level speed claim. The
frontier
microscope uses `BENCH_FRONTIER_ATTEMPTS`
(default `5`) so the row-chain and qproj kernel work can absorb local timing
noise without falling back to the much slower full scorecard.
The full goal scorecard is still the release proof, but Q8 work now also has a
focused commit-boundary loop: `npm run check:goal-scorecard:q8` runs only the
static, frontier, and Q8 prompt candidate scorecard sections via
`ZGML_SCORECARD_CHECKS=static,frontier,q8`. That keeps the hot path on the
current Q8 substrate frontier and avoids spending a kernel-iteration turn in the
long portable Wasm/browser evidence tail unless the change actually touches it.
The qproj group microscope now also prints command-shape and runtime-command
profile evidence. Current local evidence is deliberately two-lane and honest:
the compact x4 grouped shape is present (`shape_commands=1`,
`shape_projection_groups=1`, `shape_covered_ops=8`,
`shape_saved_dispatches=7`), but `runtime=off` with zero
`runtime_projection_group_dispatches` and zero
`runtime_projection_cache_group_dispatches` because it is below the Metal
executable-region threshold. The matching x7 qproj region microscope uses the
same threshold as the runtime schedule and now proves named projection-group
execution above the focused `1.00x` speed floor:
`projection_group_region_full_prefill`, `projection_group_region_smollm_prompt`,
`runtime=command`, `shape_commands=2`, `shape_projection_groups=2`,
`shape_covered_ops=14`, `shape_saved_dispatches=12`, and
`runtime_projection_group_dispatches=2` with zero cache-group dispatches. Count
x4 as a scheduler-shape diagnostic and x7 as the executable-command proof.
The ggml script now also accepts per-format `ZGML_F16_EXTRA_ARGS` and
`ZGML_Q8_EXTRA_ARGS`, so Q8 prompt paths can be measured against llama.cpp
without mutating the F16 evidence lane. Its Q8 comparison lane defaults to the
two-dispatch projection-row-chain command path because that is the structurally
clean `181`-command prompt shape; the older
`--metal-prompt-projection-row-chain-command-candidate` flag remains an alias.
The `dev:perf:ggml:q8-command-smoke` loop keeps this Q8 command path explicit
for cheap artifact-producing microscope runs, not as a replacement for the hard
ggml parity gate. That loop now runs `scripts/check_ggml_q8_command_smoke.cjs`,
which requires the Q8 prompt lane to be
`metal scheduled prefill projection-row-chain command`, keeps the `181`-command
shape, proves `120` projection-row-chain dispatches plus the existing projection
pair/cache-group commands, and prints a compact `ggml q8 command smoke: pass`
line with the accepted artifact path.
The frontier benchmark gate keeps the same floors but now evaluates them across
its repeated noisy attempts instead of requiring every independent microbench
lane to pass in one lucky attempt. If no single attempt clears all floors but
each floor and profile invariant is proven by the attempt set, it prints
`frontier bench aggregate: pass across ... noisy attempts`; if any floor is not
proven, the gate still fails and prints per-attempt diagnostics.
The PyTorch parity set is no longer only dense/transformer-shaped CPU work: it
also compares batched `max_pool2d` and `avg_pool2d` against upstream
`torch.nn.functional`, so the parity gate covers common compiled tensor kernels
outside matmul/linear dispatches.
The June 19, 2026 PyTorch parity push found the remaining FFN miss was not
RMSNorm or SiLU but the `128x128 @ 128x64` down projection. The fix routes that
larger contiguous dense projection through a column-major BLAS view of the same
row-major buffers (`C^T = B^T A^T`) and keeps the small hand projection lane for
`K <= 64`. Current hard-gate evidence against PyTorch `2.12.1` is green after
forcing the hard rerun through a ReleaseFast native rebuild, while the caveat
remains that `linear_batched` is the thinnest/noisiest lane. The right next
move is more stable margin for the small batched linear hot path, not lowering
the floor or hiding that lane. A June 23, 2026 exact-shape
`M=128,N=32,K=64` unrolled direct-linear kernel was tested and rejected: it made
the module hot path slower (`hot_execute_into` around `0.0032ms` instead of the
existing `~0.0024-0.0025ms`) and did not give stable PyTorch median margin.
Do not repeat that as the next linear fix; the next small-linear move needs
better structure than a one-off K-loop unroll.
The next focused PyTorch pass used the new microscope loop to close the
exploratory `rms_gelu_linear_batched` miss: current evidence is
`rms_gelu_linear_batched=zgml:0.0382ms pytorch:0.1005ms
zgml_vs_pytorch=2.63x`, with the module hot path reporting
`hot_execute_into=0.0369ms`.
A later microscope pass found that the small direct linear kernel needed a
shape-specific implementation rather than a blanket BLAS escape hatch. Sending
the PyTorch comparison shape `M=128,N=32,K=64` through BLAS could make one noisy
sample look better, but it regressed the focused module hot path. The current
shape is better served by a native small-direct kernel that uses 16-wide column
vectors when aligned and keeps the old 8-wide path for narrower rows.
Current focused exploratory evidence after the ReleaseFast rebuild keeps
`rms_gelu_linear_batched` comfortably ahead of PyTorch, keeps
`lazy_token_head_batched` around parity, and closes the previous
`log_softmax_classifier_batched` softness with a direct CPU
`Linear -> LogSoftmax` execute path whose public Program plan remains
`linear|log-softmax`. A previous BLAS-backed direct `Linear -> LogSoftmax`
fusion attempt made that lane slower, so the lesson is not "add every shortcut";
it is "only keep the shortcut when the exact shape has ReleaseFast evidence and
the public operation story stays simple."
This was rechecked against the BLAS-backed direct `Linear -> LogSoftmax`
variant: a CPU-only direct Session path that ran BLAS for `Linear` and then
in-place row `LogSoftmax` still regressed the focused module hot path to about
`0.0095ms`, so it was reverted. A fixed-width native log-softmax micro-kernel
for the exact `cols == 32` gap was also tried and rejected on June 22, 2026.
The current June 23, 2026 version keeps the simpler public Program plan and
uses a measured native row log-softmax implementation instead: vector max, fast
vector exp sum, vector subtract, scalar tails for general widths, and an exact
`N=32` specialization for the classifier shape that PyTorch comparison tracks.
The larger lesson remains: keep measured kernels that improve the exact hot
lane while preserving coverage, and allow a narrow shape specialization only
when focused ReleaseFast evidence improves both the module hot path and the
PyTorch comparison gate without changing the public operation story.

The JS/TS face has one source of truth: TypeScript. The answer to "how do we
keep these in sync?" is: we do not. Do not build a sync system. Build one TS
library. The product API for JS, Node, Bun, browser, and the future JS FFI
package should be written in `src/ts/**`, checked by `tsc`, and emitted by
`tsdown` into `dist/**` declarations, CJS artifacts, source maps, and package
subpaths. Do not hand-maintain a parallel JS implementation, declaration twins,
generated checkout artifacts, or a Zig-shaped product frontend that every JS/TS
feature must mirror before it is useful.

The practical version is:

```text
TS owns: Tensor ergonomics, nn modules, autograd policy, loss, optim, train,
         checkpoint/state, compile support, package subpaths, JS FFI facade,
         and policy for when execution is eager, compiled, or unsupported.
Zig owns: kernels, allocators, buffers, native tensor storage, Program handles,
          Session handles, backend dispatch, ABI structs, C/Wasm exports, and
          backend-specific execution. For tensor-sized eager work, that native
          tensor storage should become the default execution lane rather than a
          special compiled-only escape hatch.
Shared: executable contracts, evidence records, signatures, tests, benchmarks.
```

When the same idea appears in both TS and Zig, it is not automatically a sync
target. First ask which side owns the product decision. If it is API policy or
ergonomics, TS owns it and Zig should expose only the primitive needed to run it
fast. If it is execution machinery, Zig owns it and TS should call it through a
small, typed adapter. The only thing that must stay aligned by design is the
contract at the Program/Session/ABI boundary.

Operational rule: if a change starts to feel like keeping two frontends in
sync, delete the premise. There should be no manually synchronized TS/Zig
product API. Write the library behavior once in TS, emit Node/Bun/browser/JS
FFI artifacts with `tsdown`, and make native work appear as a smaller runtime
contract underneath it. Zig code that duplicates `Tensor`, `nn`, `loss`,
`optim`, or `train` product policy is migration residue unless it is clearly a
native test fixture, native-only example, kernel/runtime primitive, or ABI
surface. The desired fix is to move policy upward into `src/ts/**` and lower
hot paths downward into Program/Session/ABI, not to add another synchronization
guard.

Zig sits below that TS product API as the native runtime/kernel/ABI substrate:
kernels, buffers, compiled Programs, Sessions, backend hooks, and low-level
native handles. If a feature is ordinary library policy, it belongs in TS. If it
needs native speed, the TS surface should compile, bind, or call into
Program/Session/runtime primitives. If it cannot compile yet, keep it
eager-correct and honest about compile support instead of creating another API
copy. Over time, eager tensor operations should also move from JS array math to
native-backed storage and native single-op kernels where the data size or
operation cost justifies crossing the boundary. Host-specific differences belong
at the adapter edge; tensor, module, compiler, optimizer, loss, and training
policy belong in `src/ts/**`.
The ideal package has a single editable product language, not many source-level
frontends pretending to be one API.

Pinned decision: do not build a sync system. Build one TS library; the future JS FFI package should be written in the same TS product surface; keep a single editable product language, not many source-level frontends pretending to be one API.

The rule is stronger than "prefer TS." The package API is TS-authored, while
the core runtime is Zig-owned. `tsdown` is the fan-out mechanism for Node, Bun, browser-safe
frontend entries, declarations, smoke artifacts, and future JS FFI shims. Zig
compatibility means stable native contracts, ABI records, kernels, buffers, and
Program/Session behavior. It does not mean every tensor, module, optimizer,
loss, or training helper gets re-authored in Zig before it is considered real.
If two things need to stay aligned, align the executable contract and prove it
with tests; do not create a second frontend and then spend architecture budget
keeping both frontends synchronized.
The runtime manifests should encode this as data, not vibes:
`productSourceOfTruth: "ts-api-zig-core"` on the TS frontend/native contract and
`.product_source_of_truth = "ts-api-zig-core"` plus `.native_product_policy =
"required-core"` on the Zig native substrate.
Public TS product namespaces should not repeat those fields by hand. They use
the shared `tsProductManifestPolicy(...)` helper in
`src/ts/internal/product_manifest.ts`, making product ownership a deep Module:
one implementation, small manifest Interface, and package/type/smoke tests at
the namespace seams.

Verification should enforce that boundary. It should compare behavior, public
types, package exports, runtime evidence, and native binding contracts. It
should reject checked-in generated JS or compatibility shims rather than
byte-compare copied source files. It should also guard the build fan-out itself:
the package metadata policy should derive runtime entries, declaration globs,
npm subpaths, package root fields, and file-boundary rules from `src/ts/**`;
`tsdown.config.mjs` should consume that policy to emit artifacts, keeping only
narrow host/runtime exceptions such as the separate Node FFI runtime entry.

Implementation rule: authored package behavior belongs in `src/ts/**`. `dist/**`
is an emitted artifact, and host entries such as Node, Bun, browser, and future
JS FFI should be generated from the TS product tree by `tsdown`. Host adapters
may supply runtime-specific native calls, symbol loading, resource import, and
ABI packing, but they should not duplicate tensor/module/loss/optim/training
policy. When a change feels like "add the same method to every host," first
move the method into the TS-owned shared facade and let the package build fan it
out.

Handwritten JS/CJS/MJS examples are allowed only as consumer harnesses over the
package, dist artifacts, or wasm binary. They are not a hidden implementation
lane. If an example starts owning tensor, module, optimizer, training, compile,
or runtime policy, that behavior belongs in TS and should be emitted through
`tsdown`.

This creates three compatible faces without requiring source-level mirroring:

```text
JS/TS product:   Tensor, Module, Optimizer, stateDict, trace, compile
Zig native API:  Program, Session, buffers, kernels, ABI, backend hooks
runtime core:    Program, Session, StepParams, NativeBuffer, evidence
```

Every new product feature should land first in the TS product surface unless it
is purely a native kernel/runtime primitive. A frontend feature that cannot
compile yet is still valuable if it is eager-correct, differentiable when
appropriate, serializable through the common state interfaces, and honest about
compile support. A runtime feature is not done until it appears behind the
simple module/tensor surface or a deliberate low-level FFI handle.

## Library Shape Contract

The ideal zgml is a small PyTorch-like replacement for JS/TS, plus a native
execution core that PyTorch does not try to be. These are one product, not two
tracks.

The current user-facing replacement claim is tracked in
`docs/frontend-capability-matrix.md`. That file is the compact source for what
is eager, autograd-capable, shape-typed, native-Program-backed, packaged,
browser/Wasm-ready, or still only partial. The plan should stay aspirational;
the matrix should stay falsifiable.

The public library should be judged by this ladder:

```text
1. Can I express the model plainly?
2. Can I run it eagerly and debug the values?
3. Can I train small versions with autograd, loss, and optimizers?
4. Can I snapshot and restore the model state with stable names?
5. Can I ask whether this model can compile?
6. Can I compile the hot path into Program -> Session -> step/execute?
7. Can I prove what ran, where it ran, and what was bound?
```

The answer should be "yes" first in JS/TS, because that is the highest-leverage
package surface for Node, Bun, browser, and JS FFI users. Zig should remain an
excellent native API and implementation language, but not a second source of
truth that every product feature must mirror before it is useful. A feature
that only exists as a C handle is an execution primitive, not a complete
user-facing library feature. A feature that only exists in eager tensors is
still useful, but it should report compile support honestly and leave a clear
lowering path.

This implies a few non-negotiables:

- `Tensor`, autograd, `nn`, `loss`, `optim`, `train`, state dictionaries, and
  compile-support metadata stay first-class.
- TS is the reference product implementation for the JS package. Zig is the
  native runtime/kernel implementation and an idiomatic native-facing API, but
  it should not force package work into source synchronization.
- The runtime core should make deployment and FFI excellent without leaking
  backend vocabulary into ordinary model code.
- The graph/IR stays small; PyTorch-like breadth comes from composed frontend
  APIs, not a giant primitive set.
- Performance claims belong to compiled Programs and Sessions, with no hidden
  fallback, no surprise hot-loop allocation, and inspectable evidence.

## Frontend Product Contract

The ideal library has two faces:

```text
friendly ML frontend:
  Tensor, autograd, nn, optim, losses, data helpers, model modules

execution core:
  Program, Session, StepParams, backend bindings, runtime patch table
```

The frontend should feel familiar to people who know PyTorch, tinygrad, or
simple NumPy-style tensor code:

- tensor construction, indexing, broadcasting, math, reductions, matmul, views,
  dim-aware reductions/softmax, dtype/device movement, and serialization should
  be boring and discoverable
- autograd should make ordinary training code possible without exposing graph
  internals
- `nn` should own modules, parameters, `Linear`, embeddings, normalization,
  activations, containers, and simple model composition
- `optim` should own optimizer state and parameter updates
- losses and training helpers should exist as curated frontend APIs and examples
  rather than hidden scripts
- JS/TS should expose the same object model where practical: `Tensor`, `Module`,
  `Optimizer`, `Program`, `Session`, `NativeBuffer`, and typed-array edges

The runtime plan should never be used as a justification for deleting useful ML
surface such as `nn.Linear`, optimizers, losses, or basic training loops. Those
belong above the executable runtime. If an API is too ad hoc, reshape it into a
small module/frontend primitive; do not erase the capability.

The PyTorch-replacement bar is explicit:

- ordinary Zig and JS/TS users should be able to write small models with
  `Tensor`, `nn`, `loss`, `optim`, and `train` without thinking about C handles,
  backend descriptors, or stencil internals
- eager execution should be correct, differentiable for the curated frontend
  ops, and useful for tests, small training jobs, and model shaping; its default
  storage/execution should become native-backed for tensor-sized work, with JS
  array execution retained only as an explicit correctness, bootstrap, or tiny
  scalar lane
- compiled execution should be opt-in and obvious: `model.compile(...)` returns
  a `Program`, `program.bind(...)` returns a `Session`, and hot calls are
  `step`/`execute`
- native performance work must live under this simple surface rather than
  replacing it with backend-specific vocabulary

The library should grow by a small surface ladder, not by exposing backend
machinery upward:

```text
Tensor ops -> autograd -> nn.Module -> optimizer/loss/train -> compile Program
```

TS is the reference package API. It should own the friendly module vocabulary,
state dictionaries, optimizer snapshots, compile-support metadata, and typed
runtime evidence for JS users. Zig should expose the native runtime
idiomatically, but package development should not wait on a mirrored Zig
surface. If a TS feature cannot compile yet, it should
still be honest about whether it is a standalone Program, a composable
module-graph layer, or an eager frontend-only operation. The default answer for
Node, Bun, browser, and future JS FFI ergonomics is still: write the library in
TS, emit it with `tsdown`, and keep only the hot execution substrate native.

The JS/TS frontend should not stay split across hand-maintained `.cjs`,
`.ts`, and `.d.ts` copies. Trying to keep those files in sync is the wrong
problem. The target shape is a typed TS source tree as the single frontend
source of truth. Host-specific code should be limited to small adapters for FFI,
filesystem/path loading, and native buffer interop; tensor math, autograd,
`nn`, `loss`, `optim`, `train`, Program/Session facades, evidence records, and
type contracts should live once in TS. The package emitter is `tsdown`: the
architecture requirement is authored TS plus `tsdown`-emitted package artifacts,
not synchronized runtime and declaration files.

Practically, this means no new shared frontend policy should be authored in
CJS. More importantly, migration should stop optimizing for "CJS calls generated
TS helpers" as the final form. CJS is temporary adapter/bootstrap glue until the
package build emits the needed Node/Bun/browser entrypoints from TS. The
package-facing API should resolve through `dist/**` artifacts emitted by
`tsdown`. Public API types are TS-owned too: the root type contract is the
emitted package artifact `dist/public_api.d.cts`, not a source-tree `.d.ts` that
consumers import directly and not a root compatibility shim. Generated
source-checkout JS is no longer committed or required for tests; the editable
source of truth is `src/ts/**`, and `tsdown` owns the package/runtime artifacts
under `dist/**`.

This should be treated as a product constraint, not only a migration preference:
if a proposed change adds a second authored JS/CJS declaration or a Zig-gated
frontend mirror, the change is probably architectural debt. Add the behavior to
TS, emit it, and prove the emitted package artifact.

For this reason, "keep it in sync" is not a goal for the product surface. The
goal is stronger and simpler: **make sync unnecessary**. The architecture guard
should reject new authored JS/CJS policy, direct package imports from generated
checkout artifacts, and frontend behavior that can only be used after a Zig
mirror exists. Verification should prove that the TS source builds, the emitted
package shape works, and the native boundary contracts agree; it should not
reward parallel frontend implementations.
The repository now enforces that direction for LLaMA token-window scratch
planning: `scripts/check_ts_source_architecture.cjs` fails if `js/shared_session_facade.cjs`
starts hand-passing the session scratch records to raw option helpers again
instead of calling the TS-authored session facade methods. The same architecture gate
also rejects reintroduced hand-written CJS Session method policy for the
generic and LLaMA `step`/`execute`/`prefill`/argmax/sample/generation surfaces
that now live in `src/ts/runtime/session_facade.ts`.

The execution core should stay unusual and strict:

- compile stable work into `Program`
- bind mutable state into `Session`
- update hot values through `StepParams`
- execute with no hidden fallback and no surprise allocation
- expose evidence that the fast path is real

The user-facing rule is simple:

```text
write model code like a tiny PyTorch
ship/run model code like an executable runtime
```

The same explicit-preflight rule now applies at the hot-call boundary:
`Session.requireCanExecuteStepParams(...)` / `require_can_execute_step_params(...)`
and `stepParams.requireCanExecuteStepParamsCompatibility(...)` return the
frozen StepParams compatibility evidence when a call is executable, and throw a
reasoned `not-executable-step-params` diagnostic otherwise. Stronger assertions
such as `requireHotStepParams(...)` remain available for allocation-free and
readback-free hot paths, but plain executable preflight is first-class so
callers do not have to overload "hot" when they only need "will run without
fallback."
`zgml/step_params` also exposes standalone compatibility evidence predicates
and require/assert helpers; this matters because rejected StepParams records are
still useful frozen evidence after host, FFI, or serialization hops, even when
the next assertion correctly refuses to execute them.

The execution-plan boundary has the same descriptive/assertive split:
`Program.executionPlan(...)` and `Session.executionPlan(...)` expose frozen
evidence even when a module or StepParams object is rejected, while
`Program.requireExecutionPlan(...)` / `require_execution_plan(...)` and
`Session.requireExecutionPlan(...)` / `require_execution_plan(...)` return that
same evidence only when the plan is executable. This lets user code say "show me
why" and production code say "prove this will execute" without inventing
host-specific checks. The same TS-owned policy is also available as standalone
evidence helpers on the `program`, `session`, and `inspection` subpaths:
`acceptsProgramExecutionPlan(...)`, `requireProgramExecutionPlan(...)`,
`matchesProgramExecutionPlanSignature(...)`, and their Session equivalents.
That makes frozen execution-plan evidence useful after it crosses Node, Bun,
FFI, or serialized boundaries, without asking callers to rebuild the original
Program or Session facade.

The cold compile boundary follows the same assertion style at both public
levels: `compile.requireCompileSupport(...)` is the generic root helper, while
`nn.requireCompileSupport(...)`, `require_compile_support(...)`, and module
instance `requireCompileSupport(...)` / `require_compile_support(...)` expose
the same checked evidence through the friendly model vocabulary. Unsupported
modules must throw before `compile(...)` or `bind(...)` can imply a native path.
For callers that want a cold gate before compiling,
`requireCompilePlan(...)` / `require_compile_plan(...)` and the matching
`assertCompilePlan(...)` aliases return supported preflight evidence and throw
with the same reason vocabulary when unsupported; module-owned preflight returns
the frozen `zgml.nn.compile-explanation` artifact.
The frozen explanation is also first-class evidence: `acceptsModuleCompilePlan`,
`requireModuleCompilePlan`, assertion aliases, and
`matchesModuleCompilePlanSignature` are exported through `zgml/compile`,
`zgml/nn`, and `zgml/inspection`, so serialized or FFI-crossed plans can be
checked without re-running compile analysis. Root `compile.analyze(...)`
records now carry signed `zgml.compile.analysis` evidence too, with
`isCompileAnalysis`, `requireCompileAnalysis`, assertion aliases, and signature
matchers exported through `zgml/compile`, so cold trace/artifact analysis can
cross package or log boundaries without becoming an anonymous blob.

The bind boundary follows the same shape: `Program.bindingPlan(...)` exposes
frozen host/native/rejected binding evidence, while `Program.requireBindingPlan(...)`
and `require_binding_plan(...)` return that same accepted plan or throw before
`bind(...)` can allocate buffers or create a Session. The intended runtime
ladder is now explicit at each cold/hot transition: require compile support,
require an accepted binding plan, then require executable or hot StepParams.
The friendly `nn` surface mirrors this for module state:
`nn.bindingPlan(...)` distinguishes ModuleBindings produced by
`nn.bindParameters(...)` / `nn.placeParameters(...)` from raw ProgramBindings,
and `nn.requireBindingPlan(...)` / `require_binding_plan(...)` rejects raw or
unsupported module binding evidence before callers hand it to a Program.
Standalone binding-plan predicates and assertions follow the same shape on
`program`, `nn`, and `inspection`: `acceptsProgramBindingPlan(...)`,
`requireProgramBindingPlan(...)`, `acceptsModuleBindingPlan(...)`,
`requireModuleBindingPlan(...)`, plus signature matchers. A binding plan can
therefore be inspected, serialized, passed across a host boundary, and checked
again before Session creation.
Runtime-profile evidence follows the same JS/TS plus FFI-friendly naming rule:
Program and Session objects expose both camel and snake forms for profile
snapshots, signature matching, resets, and no-fallback/no-sync/hot assertions.
The alias methods delegate to the same evidence helpers; they are not a second
policy path.

Product gate for future work:

- If a runtime primitive is useful to application authors, expose it through the
  small `Tensor`/`nn`/`Program` vocabulary rather than leaving it as a C-only
  shape.
- If a frontend primitive is useful for ML work, keep it even before it has a
  compiled lowering, but require eager correctness, state/autograd behavior when
  applicable, and honest `compileSupport()` metadata.
- If a feature can compile, prove the whole ladder: eager API, trace/support
  metadata, Program inspection, Session binding, execution/profile evidence,
  and JS/TS/C ABI behavior where the feature is public.
- Prefer one sharp API that spans Zig, Node, Bun, Wasm, and browser targets over
  many backend-shaped escape hatches.

High-bang architecture moves from here:

- Build the **Lazy Tensor IR** and **Kernelizer** as the next deep compiler
  Modules. Ordinary Tensor and `nn` code should lower into one normalized graph;
  the Kernelizer should own scheduling, fusion, memory/buffer layout, and kernel
  command shape. That gives callers leverage through one small compile
  Interface and maintainers locality for performance work. The Zig compiler now
  has a shared internal `TensorProgramIr` Module before
  `LoweredDeviceProgram`; it materializes owned `IrValue` shape/layout
  evidence, an `IrOp` table, and flat input-edge lists, threads that evidence
  through `Program.inspect()`, and lowers ordinary node ops through normalized
  `IrOp` / `ValueId` inputs and outputs. Supported fusions now lower from
  normalized IR evidence too: elementwise-chain fusion owns an IR sub-tape,
  layer-norm fusion lowers from `IrOp` attrs, and log-softmax fusion owns a
  range of normalized sub-`IrOp`s. JS/TS module Tensor Program IR now carries a
  frozen value table, graph input/output value IDs, and per-op input/output edge
  IDs into KernelPlan evidence too, so the frontend compiler is moving from
  trace-shaped descriptors toward op/shape/value evidence. Keep deepening that
  seam until frontend module tracing and the Kernelizer consume the same shared
  native op/shape/value IR instead of carrying parallel trace/kernel-lowering
  representations.
- Build the **Trace-to-Program Compiler** as the next deep module after
  `compileSupport()`. Normalized traces should lower through Lazy Tensor IR and
  the Kernelizer instead of accumulating special-case model matchers. A layer
  should be "compile-capable" when its eager op has shape propagation, state
  metadata, a lowering rule, and a scheduler/kernelizer story.
- Collapse duplicated Node/Bun frontend semantics into a **Shared JS/TS
  Frontend Core** with thin host adapters. `Tensor`, autograd, modules, losses,
  optimizers, state dictionaries, traces, and support metadata should not drift
  between runtimes.
- Add a small **Tensor Placement Interface** so eager host tensors, compiled
  `Program`s, and `NativeBuffer`/device-backed execution feel like one object
  model. Users should opt into placement and evidence, not hand-author backend
  binding records for normal model work. The first JS/TS host-buffer rung is in
  place: `Tensor.toNativeBuffer()`, `Tensor.place(program, kind)`, and
  `Tensor.fromNativeBuffer(...)`. The next rung is now source-level placement
  evidence: `Tensor.nativePlacement(...)` / `Tensor.native_placement(...)`
  returns a frozen no-allocation host/program placement descriptor with shape,
  byte length, buffer kind, and a stable signature, and the package/type smokes
  pin that normal model code can inspect placement without building backend
  binding records by hand.

Current frontend slice:

- The public root exports a first-class `loss` namespace next to `nn`, `data`,
  `optim`, `train`, and `llm`.
- The TS-owned tensor and graph surface exposes PyTorch-familiar construction
  helpers (`fromSlice`, `zeros`, `ones`, `full`, `arange`, `linspace`, `rand`,
  `randn`, `scalar`, `param`) so JS/TS users can write normal tensor code
  before opting into executable `Program`/`Session` paths.
- The TS-owned `zgml.nn` namespace includes value-owned `Sequential`
  composition, `Embedding`, feature-axis `LayerNorm` and `RmsNorm`, plus
  `gelu`, `relu`, and
  source-compatible `siluActivation`, `sigmoidActivation`, and `stepActivation`
  modules, small unary elementwise modules (`exp`, `log`, `neg`, `recip`,
  `abs`, `sqrt`, `square`/`sqr`, `sgn`/`sign`), plus composable `softmax(dim)`
  and `logSoftmax(dim)` modules and dim-preserving reduction modules
  `sum(dim)`, `mean(dim)`, `max(dim)`, `min(dim)`, `argmax(dim)`, and `argmin(dim)`, plus metadata-shaped `identity`,
  `reshape`, `view`, `flatten`, `transpose`, `permute`, `broadcastTo`, and `expand`
  modules, so users can build small transformer-shaped models without growing
  the primitive IR. Native helper modules may keep matching parameter metadata
  only as runtime-substrate evidence;
  `Sequential` preserves stable prefixed names such as `0.weight` and `2.bias`
  so state inspection does not require pointer archaeology. JS/TS modules and
  the root `nn` namespace also expose frozen `parameterNames()`,
  `parameterInfos()`, and `parameterInfo(nameOrIndex)` snapshots, giving
  callers the eager state-name, shape, layout, and scalar-count contract before
  the same names become Program and Session packed-state evidence. `nn.stateDict(...)`
  now snapshots those named parameters into owned name/data/layout entries, and
  `nn.loadStateDict(...)` restores them with strict name, shape, and explicit
  layout checks. JS/TS `stateDict(prefix)`, `loadStateDict(..., { prefix })`,
  and `checkpoint.create/restore({ prefix })` now share the same prefix contract
  for composed module snapshots. Checkpoint metadata is JSON-cloned and frozen
  as part of the snapshot evidence, and `checkpoint.inspect(snapshot)` returns
  a frozen payload-free summary of model/optimizer names, shapes, layouts, and scalar counts, with
  `checkpoint.modelParameterInfo(...)` and `checkpoint.optimizerEntryInfo(...)`
  lookup helpers for name-or-index access. Native helper `loadStateDict(...)`
  validation may mirror the same executable state contract for low-level tests,
  but package policy lives in TS. Native helper modules also expose honest
  `compileSupport()` / `canCompile()` reporting; f32 vector linear and
  embedding helpers now compile into owned native Program/Session wrappers over the existing
  `DeviceInference` executable path, and supported multi-layer `Sequential`
  graphs trace their `forward` pass into the same `ModuleProgram` wrapper with
  parameters bound as persistent Session state. Unsupported module ops still
  fail at compile time instead of falling back silently. Zig `compileSupport()`
  now carries scalar length evidence for length-aware module paths:
  `nn.Linear` reports input/output edge shapes plus input, output, weights,
  bias, and total parameter counts;
	  single-layer `Sequential` inherits those fields; and multi-layer `Sequential`
	  propagates edge shapes and lengths through shape-preserving glue, reports
	  aggregate parameter/bias/weight counts, and rejects obvious layer-shape
	  mismatches before compile. Zig also exposes
	  `nn.compileSupportForInputShape(T, alloc, &module, shape)`, which runs the cold
	  module trace and returns exact input/output/parameter evidence before backend
	  Program construction for shape-specialized paths such as metadata shape glue,
	  reductions, and `Flatten -> Linear` heads. Shape-specialized support and
	  retained `ModuleProgram` support now also carry normalized Tensor Program IR
	  hashes, so Zig module compatibility can reject same-parameter modules with
	  different op tapes before Session binding. Native Zig `ModuleProgram` and `ModuleSession` now
	  expose retained post-compile `compileSupport()` / `canCompile()` evidence plus
  the matching persistent-state identity, scalar, and shape contract after compile/bind through
  `parameterTensorCount()`, `parameterTensorName(index)`,
  `parameterTensorLayout(index)`, `parameterTensorLen(index)`,
  `parameterTensorShape(index)`, `parameterLen()`, `parameterIndex(name)`, and iterable
  `parameterInfo(index)` / `parameterInfoByName(name)` / `parameterInfos()` descriptors, so
  callers can verify the bound parameter surface from the executable handle
  instead of only from pre-compile support metadata. The same Zig wrapper now
  exposes cold binding requirement counts plus a Program requirement hash, and
  bound Sessions expose the actual persistent/input/output counts, host/resource
  counts, and binding-shape hash from runtime inspection. Bound Zig module
  Sessions now expose `uploadParameters()`, `uploadParameter(index)`,
  `uploadParameterByName(name)`, and `uploadParameterRange(first, len)` over the
  executable persistent-binding upload path, so callers can refresh only changed
  module state without re-binding a Session. It also exposes
  structured `moduleCompatibility(...)` evidence plus
  `checkModuleCompatibility(...)`, `acceptsModule(...)`, `bindModule(...)`, and
  `bindModuleTensors(...)`, so a compiled module Program can preflight and bind
  a second compatible module's named parameter state through the same executable
  shape before uploading persistent Session bindings. The compatibility record
  carries the first mismatch code plus parameter name, index, shape, and layout
  evidence, keeping Zig's native preflight closer to the typed JS/TS
  `Program.moduleCompatibility(...)` surface. It now compares the candidate
  module's current `compileSupport()` evidence before inspecting the named
  parameter table, so native-path, model-kind, layer-count, edge-shape, and
  scalar-count mismatches reject before Session binding as architecture
  mismatches rather than late parameter upload failures.
- Trainable tensors now keep a durable parameter-gradient buffer separate from
  graph-built gradient expressions. `ComputeGraph` restores those buffers on
  teardown, so `nn.Linear`, `nn.Sequential`, optimizers, and future JS/TS
  bindings can keep weights outside a one-shot graph arena.
- JS/TS parameters now expose `requiresGrad` / `requires_grad` aliases and
  parameter-info evidence, and optimizers plus gradient clipping skip frozen
  parameters even when stale gradient buffers exist. That keeps transfer-learning
  style freezing simple without hiding parameters from state dictionaries or
  checkpoints.
- `zgml.loss` owns `meanSquaredError`, `mse`, `crossEntropy`, and
  `classTargets` as primitive-composed loss helpers.
- `zgml.optim` now exposes both coupled-weight-decay `Adam` and decoupled
  `AdamW`, plus `SGD`, so ordinary PyTorch-like training examples do not have
  to pretend `AdamW` is the only adaptive optimizer. Zig and JS/TS optimizers
  expose `stateDict()` / `loadStateDict()` snapshots for SGD momentum and Adam
  moment/timestep state, using the same named, shape-aware entry convention as
  module state. They now validate optimizer snapshot kind, parameter count,
  expected state names, duplicate names, tensor shape/layout, data length, and step before
  mutating moment buffers or timestep state, so failed restores are
  all-or-nothing like module state restores. JS/TS optimizers also expose frozen
  `config()` snapshots plus `setLearningRate(...)` / `set_lr(...)`, so simple
  schedulers can update learning rate without poking mutable fields directly.
  Optimizers now accept small PyTorch-style parameter groups with per-group
  `lr` and `weightDecay` / `weight_decay`, and report frozen group evidence in
  `config()`, so transfer-learning workflows can tune head/backbone rates
  without a policy-heavy trainer. They also expose `addParamGroup(...)` /
  `add_param_group(...)`, so later unfreeze phases can append newly trainable
  parameter groups while preserving the same optimizer/state-dict shape. Group
  registration rejects duplicate parameter objects at construction time and
  during later appends, so a parameter cannot be silently updated twice by the
  same optimizer.
  The TS-owned optimizer namespace now includes small `StepLR`-style and
  exponential learning-rate schedulers (`stepLR` / `step_lr` and
  `exponentialLR` / `exponential_lr`) with frozen config/state snapshots, so
  training loops can schedule optimizer state without adopting a policy-heavy
  trainer. Checkpoints now include scheduler snapshots alongside model and
  optimizer state, with validation-only restore support so resuming a training
  run is all-or-nothing across weights, optimizer moments, and learning-rate
  progress.
- `zgml.train` owns tiny loop helpers, currently `backward`, `step`,
  `lossStep`, `fit`, read-only gradient monitoring via `gradNorm` /
  `grad_norm`, `clipGradNorm` / `clip_grad_norm_`, and `clipGradValue` /
  `clip_grad_value_`, so
  examples and future bindings can express the common loss/optimizer workflow
  without exposing graph internals or inventing a policy-heavy trainer.
  `train.fit(model, loader, { optimizer, loss })` is now the documented
  beginner path, while the older optimizer-first `fit(...)` and explicit
  `fitModule(...)` / `fitClassifier(...)` forms remain available for manual or
  compatibility-oriented loops. The public type smoke pins
  `TrainModelFitOptions`, and the package smoke proves the model-first path
  returns the same signed fit evidence as the explicit module helper.
  `zgml.data` now owns the matching tiny data rung: `tensorDataset` /
  `tensor_dataset` plus iterable `batches` / `batch` / `dataLoader` helpers
  that stack leading-dimension tensor rows into frozen batch evidence for
  `train.fit` without introducing a policy-heavy trainer.
  Tensor/module/optimizer/namespace zero-grad surfaces accept
  `setToNone` / `set_to_none`, matching PyTorch's useful memory/semantics knob
  while keeping the default behavior as zero-fill.
- Node and Bun now expose a small `Tensor`, `nn`/`loss`/`optim`/`train`
  frontend facade. `Tensor` carries shape metadata over typed-array storage,
  exposes PyTorch-familiar shape introspection (`size(dim?)`, `dim()`,
  `numel()`, `ndim`),
  basic arithmetic, scalar-exponent `pow`, elementwise `maximum`/`minimum`, broadcasted `where`, `maskedFill` / `masked_fill`, and elementwise comparisons (`eq`/`ne`/`lt`/`le`/`gt`/`ge`)
  with scalar-number tensor-like inputs, trailing-dimension
  broadcasting, row-major `matmul`/`mm`, vector `dot`, matrix `trace`/`diagonal`, rank-3 `bmm`, reductions, dim-aware
  `sumDim`/`prodDim`/`meanDim`/`maxDim` plus eager `minDim`/`argmaxDim`/`argminDim` and mask reductions `anyDim`/`allDim`,
  PyTorch-like reduction aliases `sum(dim)`, `prod(dim)`, `mean(dim)`, `max(dim)`, eager `min(dim)`, `variance(dim?)` / `var(dim?)`, `std(dim?)`, `norm(dim?, p?)`, shape-preserving `cumsum(dim)`, stable `logsumexp(dim?)` / `logSumExp(dim?)`, mask-valued `any(dim)` / `all(dim)`, and eager index-valued `argmax(dim)` / `argmin(dim)`,
  dim-aware `softmaxDim`/`logSoftmaxDim`, activations, tensor-level `clamp`/`clip`, view helpers (`reshape`,
  `view`, `reshape_as`, `view_as`, `flatten`, `squeeze`, `unsqueeze`, `transpose`, `permute`, `T`, `broadcastTo`, `expand`, `expand_as`, `repeat`, `tile`, `flip`, `roll`, `select`,
  `narrow`, `slice`, `split`, `chunk`, `unbind`),
  rank-aware indexing/debug helpers and numerical masks/transforms
  (`get`, `set`, `toArray`, `tolist`, `numpy`, `nelement`, `ndimension`,
  `elementSize`, `element_size`, `nbytes`, `typeAs`, `type_as`, `stride`, `strides`,
  `isContiguous`, `contiguous`, `storageOffset`, `storage_offset`,
  `isCpu`, `is_cpu`, `isFloatingPoint`, `is_floating_point`, `isLeaf`,
  `is_leaf`, `inspect`, `allclose`, `equal`, `isclose`, `isnan`, `isinf`, `isfinite`,
  `floor`, `ceil`, `round`, `trunc`, `rsqrt`, `log1p`, `expm1`, `negative`, `reciprocal`, `sin`, `cos`, `tan`), PyTorch-style gradient aliases
  (`requires_grad`, `requires_grad_`, `requiresGrad_`, `zero_grad`), tensor-context factories
  (`new_empty`, `newEmpty`, `new_zeros`, `newZeros`, `new_ones`, `newOnes`, `new_full`, `newFull`), differentiable `clone`, `detach`,
  in-place `detach_`, root grad-mode helpers (`isGradEnabled`,
  `setGradEnabled`, `noGrad`, `inferenceMode`, `enableGrad`) that suppress
  tensor/module/loss autograd edges while preserving explicit parameter
  construction,
  indexed selection/gather/take/argsort/sort/topk/scatter-add (`indexSelect`, `index_select`, `gather`, `take`, `argsort`, `sort`, `topk`, `scatterAdd`, `scatter_add`), tensor joins (`cat`, `concat`, `concatenate`, `stack`, `vstack`, `hstack`) and eager/autograd `einsum` for explicit-output contractions, repeated-label traces, ellipsis batch axes, implicit ellipsis reductions, broadcast dimensions, and TS literal-equation shape inference, compile-aware lazy parameter slots with `matmul`/`mm`/parameterized-add lowering evidence, JSON serialization, deterministic seeded
  randomness (`manualSeed` / `manual_seed`, `initialSeed`, `seededRng`, and
  per-call `seed` options), and
  PyTorch-familiar construction helpers (`zeros`, `ones`, `full`, `rand`,
  `randn`, `linspace`, `arange`, `scalar`, `parameter`, `param`) on both the
  module and `Tensor` class; eager batched
  `nn.linear`, `nn.embedding`, `nn.dropout`, `nn.sequential`,
  `gelu`/`relu`/`silu`/`sigmoid` activations, unary elementwise modules
  (`exp`, `log`, `neg`, `recip`, `abs`, `sqrt`, `square`/`sqr`, `sign`/`sgn`, `step`), traceable shape modules (`reshape`, `view`,
  `squeeze`, `unsqueeze`, `transpose`, `permute`,
  `flatten`, `broadcastTo`, `expand`, `narrow`, `select`, `slice`), `softmax`, `logsumexp`, `layerNorm`, `rmsNorm`,
  eager/stateful `BatchNorm1d` with PyTorch-style `nn.functional.batch_norm1d`,
  `loss.mse`, `loss.crossEntropy`, SGD, Adam, AdamW, `train.step`,
  `train.lossStep`, `train.clipGradNorm`, and `train.clipGradValue` can run
  host-side model checks and small training loops while still accepting typed
  arrays at the edges. JS/TS `Dropout` is eager/autograd-capable, obeys
  `train()` / `eval()` mode, lowers deterministic no-op Dropout (eval mode or
  `p = 0`) as an identity native module Program, uses zero native dispatches
  when that no-op is the whole compiled graph, and reports structured
  unsupported native compile evidence for training-mode stochastic Dropout.
  Zig `nn.Dropout(T)` now mirrors that shape as composed tensor frontend code:
  training mode builds a mask multiply with normal autograd, eval mode or `p=0`
  reports shape-preserving native support, pure deterministic no-op Dropout
  Programs execute with zero backend dispatches, and `Sequential.train(...)` /
  `eval()` cascades the mode to child modules.
  Modules expose
  read-only `Sequential.length` / `at()` / iteration,
  PyTorch-style `children()` / `modules()` and `namedChildren()` /
  `namedModules()` traversal, matching root `nn.children(...)` /
  `nn.modules(...)` / `nn.namedChildren(...)` / `nn.namedModules(...)`
  helpers, `namedParameters`, `requiresGrad_` module
  freezing plus root `nn.requiresGrad(...)` / `nn.requiresGrad_(...)`, `zeroGrad`,
  shape/layout-aware `stateDict`/`loadStateDict`, PyTorch-style class aliases
  such as `nn.Linear`, `nn.ReLU`, `nn.GELU`, `nn.SiLU`, `nn.Sigmoid`,
  `nn.Flatten`, `nn.Dropout`, and `nn.Identity`, and honest `compileSupport()` /
  `canCompile()` reporting
  so callers can discover whether a model has a native Program path before
  calling `compile()`. The state-dict load path still accepts legacy bare
  arrays, but canonical snapshots carry `{ shape, layout, data }` entries so
  Zig, Node, and Bun can converge on one model-state Interface without treating
  JS row-major Linear weights as interchangeable with Zig feature-major Linear
  weights. The shared JS/TS load path now validates unexpected names, missing
  names, shape, layout, and data length before copying any parameter bytes, so
  failed restores are all-or-nothing across Node and Bun too. Module snapshots,
  module entries, optimizer entry names, and their shape arrays are frozen, so shape evidence
	  cannot be mutated underneath later restore or checkpoint calls. Optimizers expose
	  matching state snapshots, so small training loops
	  can checkpoint momentum/Adam state without adopting a policy-heavy trainer.
	  The shared emitted Node/Bun package smoke now proves strict prefixed
	  `loadStateDict(...)`, `validateOnly` non-mutation, optimizer state restore,
	  checkpoint inspection, and full model/optimizer checkpoint restore through the
	  public package surface.
	  The shared JSON-safe `checkpoint` namespace composes model and optimizer
  state into a single `{ format, version, metadata, model, optimizer }`
  snapshot and restores it strictly across Node and Bun. Combined checkpoint
  restore now preflights both model and optimizer targets before mutating either
  one, so a bad optimizer snapshot cannot leave model weights half-restored.
  The package root now exposes the Node
  FFI wrapper through the TS-built `dist/node.cjs` runtime plus the emitted
  `dist/public_api.d.cts` declaration surface, so JS/TS users get
  a real `zgml` import path instead of an example-private module. The package
  also has a public TypeScript declaration smoke behind `npm run typecheck`,
  proving literal tensor-shape evidence and the shaped `Embedding.compile`
  contract stay type-checked instead of only runtime-tested. That smoke now also
  proves explicit `Session.stepTensor(...)` / `readOutputTensor(...)` shapes,
  LLaMA `stepTensor(...)` / `executeTensor(...)` / `prefillTensor(...)` shapes,
  output-tensor-derived shapes, and `Tensor.fromNativeBuffer(...)` placement
  readback shapes survive the public declaration surface, plus typed Program
  compile-evidence helpers, module compatibility, `acceptsModule(...)`,
	  `bindModule(...)` Session binding, and composed
	  `Sequential([Linear, ReLU, Linear])` trace/IR/kernel-plan evidence for the
	  real module Program path. It also proves LLaMA Program requirements,
	  capabilities, KV-cache requirements/layout, output-shape, runtime-profile,
	  and Session inspection/layout evidence remain typed read-only snapshots.
	  The shared package runtime smoke also proves built-in TinyLlama tensor
	  helpers (`stepTensor`, `executeTensor`, `prefillTensor`, and native-output
	  `readOutputTensor`) can fill caller-owned `Float32Array` carriers while
	  returning shaped Tensor views over that same storage, and now pins
	  TinyLlama Program capabilities, diagnostics, and model-compatibility
	  evidence as frozen runtime proof surfaces. Generic Session and
	  TinyLlama `readOutputTensor` now also accept larger reusable carriers with
	  an explicit active `length`, returning a shaped subarray view over the
	  caller-owned storage while leaving inactive capacity untouched.
	  The npm package now
  uses an explicit runtime/build-source allowlist, so internal benchmark,
  planning, and smoke-harness artifacts stay out of the published library while
  shared JS adapters, the C header, C API docs, and
  native Zig build sources remain available. Its published `npm test`,
  `smoke:node`, and `smoke:bun` scripts now use tiny TS-authored package
  smokes emitted to `dist/smokes/**` instead of repo-only harness paths, while
  `smoke:adapters` and `test:adapters` name the Node+Bun package proof
  explicitly. `npm run test:repo` keeps the heavier source-checkout validation
  path. Both host smokes consume the TS-authored native API contract before
  running behavior checks, so the published proof cannot drift by runtime while
  the host wrappers only select the adapter. Those
  included smokes prove the
  PyTorch-like nested-array tensor constructor path with rectangular-shape
  validation plus shape-validated nested Linear/Embedding/norm parameters and
  direct Program/Session tensor inputs/outputs; the compiled linear `Program`/`Session` path, including `canCompile()` /
  `compileSupport()` evidence, Program/Session tensor shapes, buffer layout,
  retained compiler evidence, compatible-module preflight, signed Program
  capability evidence with matching `canExecute()` / external-resource /
  dispatch-plan / execution-mode helpers, `bindModule(...)` state replacement,
  incompatible-module rejection, module-bound `stepTensor` / carrier
  `stepTensor` / `stepInto` / `execute` / `executeInto` /
  `executeTensor` output values, no-output execution, unbound-readback
  rejection, output-bound `readOutputInto` / `readOutputTensor` carrier
  readback, inactive carrier-capacity preservation, Tensor/native storage
  round-trip through `Tensor.toNativeBuffer()`, `Tensor.place(...)`, and
  `Tensor.fromNativeBuffer(...)`, role-specific Program `NativeBuffer`
  factories, retained input/output `NativeBuffer` Session execution,
  caller-owned `NativeBuffer.readFloat32Into(...)` readback,
  NativeBuffer-backed persistent weight/bias refresh through `uploadParameterByName(...)`,
  `uploadParameterRange(...)`, `uploadParameters()`, and `uploadParameter(...)`,
  fully-qualified parameter-name validation, frozen parameter inspection, and
  matching Session call-profile counters; and the composed module
  compiler path through `Sequential([Linear, ReLU, Linear])`, including
  `device-program` compile support, Trace, Tensor Program IR, KernelPlan
  evidence with fused Linear+activation, module compatibility, `bindModule`,
  and eager-vs-compiled output parity. The same shared smoke also proves the
  composed `Sequential([Linear, LayerNorm, GELU, Linear])` path with nested
  tensor input, nested Linear parameter matrices, packed mixed Linear/LayerNorm
  parameter layout, compatibility preflight, direct Program bind/step with nested
  array input plus wrong-rank input and exact-sized shape-bearing output
  rejection, Session
  inspection, retained output-shape readback, and eager/compiled parity. The
  same shared smoke also proves the
  PyTorch-like eager `nn`/`loss`/`optim`/`train` path through an MSE + SGD
  update with module and optimizer state snapshots. The generated tarball can run
  `npm run build:native` plus those smokes against its own `zig-out` library.
  The root Zig build keeps the rich repo validation in full checkouts
  while treating absent repo-only smoke/benchmark files as skipped in the
  published source package, so `zig build check` and
  `npm run smoke:native-wgpu` / `zig build wgpu-check -Duse-wgpu=true` work in
  both places without publishing the heavyweight harnesses. Node and Bun also report a direct
  `ZGML_C_DYLIB` / `zig build ffi-c` recovery hint when the native library is
  missing. A tiny host autograd tape now covers the existing
  Tensor arithmetic with broadcast-gradient reduction, matmul, reductions,
  dim-aware reductions, dim-aware softmax/logSoftmax, activations, embeddings,
  normalization modules, MSE, and cross-entropy classification paths. `Tensor`
  reports its eager `dtype` / `device` as `f32` / `cpu` and exposes honest
  `to`, `cpu`, `float`, and `float32` helpers: CPU/f32 movement is a no-op
  unless `copy: true` asks for a differentiable clone, while non-CPU movement
  rejects with guidance toward explicit Program placement instead of pretending
  to move storage implicitly. `Tensor`
  also owns the first small placement bridge into native execution:
  `toNativeBuffer()` creates an owned host `NativeBuffer`,
  `place(program, kind)` creates and fills the correct Program buffer, and
  `Tensor.fromNativeBuffer(...)` reads shaped outputs back into eager tensors.
  `program.bufferLayout()` exposes the primary Program slot map those helpers
  target, while `program.bufferSlotNames()` and `program.bufferSlot(nameOrKind)`
  expose direct frozen slot lookup so callers can inspect placement before creating buffers, and
  `Tensor.place(...)` rejects mismatched primary slot lengths and retained
  input/output edge shapes before allocating a Program buffer.
  `module.placeParameters(program, options)` now applies the same placement
  vocabulary to module state: it derives the Program's retained input shape when
  the caller omits one, preflights module compatibility against the Program's
  retained compile evidence, packs eager parameters through `bindParameters()`,
  preflights the Program weights/bias slot lengths, fills Program-owned native
  weights/bias buffers, and returns normal
  `ModuleBindings` for Session binding. The root
  `nn.placeParameters(module, program, options)` helper exposes the same
  compatibility-first behavior for generic module values. The public type smoke proves shaped
  Embedding Programs can use both forms without restating `inputShape`, and the
  shared package smoke executes that placed-buffer path. The public type smoke
  also imports the published `zgml/node` and `zgml/bun` subpaths and proves they
  expose the same Program, Session, Tensor, and KernelPlan descriptor evidence
  types as the root package entrypoint. Program buffer resource factories receive
  the same primary slot evidence (`kind`, `role`, element/byte lengths, and
  layout), so host/device resource binding code can implement the inspected
  contract directly. The shared package smoke now exercises generic
  `createInputBuffer({ resource })` and `createOutputBuffer({ resource })`
  callbacks too, proving their frozen slot evidence can create caller-owned
  edge buffers that bind and execute through a Session. `program.inputShape()` and `program.outputShape()` expose
  the compiled tensor edge shapes directly, so callers do not have to scrape
  `compileEvidence()` just to allocate shaped eager I/O. `program.trace()`,
  `program.compilerSignatures()`, `program.tensorProgramIr()`, `program.kernelPlan()`,
  `program.memoryLayout()`, `program.shapeConstraints()`, and `program.parameterLayout()` similarly expose
  the retained compiler ladder, module schedule, value-storage layout, shape
  specialization, and named packed module state layout when a Program was compiled from frontend module
  code. `program.parameterNames()`, `program.parameterInfos()`, and
  `program.parameterInfo(nameOrIndex)` expose the same named packed-state
  evidence without making callers scan layout arrays before binding. The shared package smoke now asserts these Program and Session shape,
  buffer-layout, runtime inspection, capability, requirement, signed profile, IR,
  kernel-plan, parameter-layout, and compiler-signature evidence objects are
  frozen, so callers cannot mutate the proof surfaces they later use for
  placement, capability, or compatibility checks. LLaMA-family Programs mirror this for cache residency
  through `kvCacheLayout()` and slot-rich `createKvCache({ resource })`
  callbacks for per-layer K/V buffers; the package smoke now creates a host
  `LlamaKvCache`, checks it against the inspected K/V requirements/layout, and
  verifies deterministic cleanup. It also exercises the `resource` callback path
  and freezes the per-slot requirement/layout evidence passed to resource
  factories. Direct `Program.bind(...)` calls also
  preflight supplied weights, bias, input, and output against that layout before
	  entering native binding. `ModuleBindings` produced by
	  `module.bindParameters(...)` or `module.placeParameters(...)` now carry
	  binding-time module evidence with frozen copied shape options, with TS
	  declarations branding those
	  module-produced bind objects separately from raw `ProgramBindings`, so direct
  `program.bind(module.bindParameters(...))` or reuse of placed parameter
  buffers rejects architecture, shape, kernel-plan, elided-op, or packed-layout
  mismatches before Session creation, while hand-authored
  `{ weights, bias }` bindings remain the raw FFI-shaped escape hatch checked by
  slot dtype, length, and retained edge shape.
  Host-backed direct binds accept eager `Tensor`
  values for weights, bias, input, output, and explicit
  `Session.step(..., outputTensor)` overrides. NativeBuffer-backed persistent
  weights/bias can also be mixed with ordinary host tensor or typed-array step
  input/output, so moving model state into Program buffers does not force every
  edge buffer into the same binding mode. Conversely, when a direct bind needs
  buffer-backed output/input, host weights and bias are placed into
  Session-owned Program buffers during bind instead of making callers pre-place
	  every persistent slot by hand. Generic compiled Sessions expose
	  `bufferLayout()`, `bufferSlotNames()`, `bufferSlot(nameOrKind)`,
	  `inputShape()`, and `outputShape()` too, with
	  `outputShape()` reflecting a bound output Tensor's retained shape when
	  present, so hot-loop callers can inspect the bound tensor edge contract
	  without keeping a Program reference
	  beside the Session. `Session.stepContract()` returns that same frozen
	  StepParams contract in one place, including input/output shapes, scalar
	  lengths, bound buffer kinds, whether missing input is valid, whether output
	  readback is available, the default output ownership/readback path, whether
	  no-output execution advances state, and a stable `signature` for comparing
	  hot-loop contracts, including input/output byte lengths and slot names/roles
	  for buffer sizing plus `defaultReadbackRequired` and
	  `defaultAllocationFree` / `defaultHotPath` plus default output
	  effect/ownership/return ownership.
	  `Session.matchesStepContractSignature(signature)` compares that cached
	  contract directly.
	  `Session.acceptsStepParams(params)` reuses the same input/output
	  validation as `execute(...)` without executing or bumping call counters.
	  `Session.canExecuteStepParams(params)` is the `canExecute()`-shaped alias
	  for that same non-executing accepted check.
	  `Session.acceptsAllocationFreeStepParams(params)` is the same non-executing
	  guard with the additional requirement that the accepted call shape avoids
	  runtime output allocation for already-normalized inputs.
	  `Session.acceptsRuntimeOutputAllocationFreeStepParams(params)` names that
	  exact predicate while keeping the older compatibility alias available.
	  `Session.acceptsNoReadbackStepParams(params)` is the matching
	  non-executing guard for accepted call shapes that avoid output readback.
	  `Session.acceptsReadbackFreeStepParams(params)` names that same positive
	  evidence predicate while keeping the older compatibility alias available.
	  `Session.acceptsHotStepParams(params)` combines those two hot-loop
	  predicates: accepted, allocation-free, and no-readback.
	  `Session.requireAllocationFreeStepParams(params)`,
	  `Session.requireRuntimeOutputAllocationFreeStepParams(params)`,
	  `Session.requireNoReadbackStepParams(params)`, and
	  `Session.requireReadbackFreeStepParams(params)` are assertion-shaped
	  preflight guards for individual hot-loop facets: they return frozen
	  compatibility evidence on success and throw structured StepParams
	  diagnostics before execution on mismatch.
	  `Session.requireHotStepParams(params)` is the assertion-shaped deployment
	  guard for that same predicate: it returns the frozen compatibility evidence
	  when the call shape is hot-loop safe and throws a structured StepParams
	  diagnostic before execution when the call would allocate, read back, or
	  reject.
	  `Session.matchesStepParamsSignature(params, signature)` compares the
	  non-executing preflight shape against a cached `stepParamsSignature`.
	  `Session.matchesStepParamsCompatibility(params, compatibility)` compares
	  against a frozen preflight record without making callers unwrap the raw
	  signature string.
	  These preflight helpers do not execute or bump call counters. The shared
	  emitted Node/Bun package smoke now proves both accepted and rejected
	  StepParams compatibility evidence through this public surface: records and
	  diagnostics are frozen, `preflightStepParams(...)` preserves signatures,
	  accepted and rejected signatures are matchable, `acceptsStepParams(...)`
	  follows the evidence, and `requireHotStepParams(...)` rejects before
	  execution on invalid input shapes.
	  `Session.preflightStepParams(params)` is the friendly evidence verb, while
	  `Session.stepParamsCompatibility(params)` remains the compatibility alias.
	  Both accept unknown boundary values and return the frozen accepted flag,
	  typed `canExecute` alias, typed accepted/rejected status, contract kind,
	  contract signature, stable
	  StepParams call-shape signature,
	  contract position (`null` for generic Sessions), per-call `stateEffect` and `allocationFree`
	  evidence, typed `inputSource` / `outputTarget` / `outputEffect` routing
	  evidence, planned input/output element types, shapes, shape signatures,
	  element lengths, and byte lengths, `inputOwnership`, `outputOwnership`, `outputReturnOwnership`,
	  `readsInput`, `writesOutput`, `readbackRequired`, `readbackFree`,
	  `runtimeOutputAllocationFree`, top-level `hotPath`, `hotPathStatus`,
	  `hotPathBlockers`, `rejectionCode`, and
	  typed rejection diagnostics with structured expected/actual length, shape,
	  and context fields when available for the same check.
	  NativeBuffer-backed persistent state can be refreshed
	  after bind with `Session.uploadParameters()`,
	  `Session.uploadParameter(index)`, `Session.uploadParameterByName(name)`, or
	  `Session.uploadParameterRange(first, len)`, making weight updates explicit
	  without rebuilding the Session; named JS/TS refreshes use retained
	  `Session.parameterLayout()` evidence, plus `Session.parameterNames()`,
	  `Session.parameterInfos()`, and `Session.parameterInfo(nameOrIndex)`
	  lookup helpers, to refresh the containing persistent weights/bias binding.
	  They also expose `SessionStepParams` through
	  `execute({ input, output })`, `stepInto(output, input)`,
	  explicit `SessionExecuteIntoParams` for `executeInto(output, { input })`,
	  `executeTensor({ input, shape | output })`,
	  and `stepTensor(input, { shape | output })`,
  which wraps the compiled output as a shaped eager `Tensor` using an explicit
	  shape, a supplied or Session-bound output Tensor's shape, or the Program's
	  retained output-shape evidence by default, plus `readOutputInto(...)` for
	  caller-owned readback from a Session-bound host or native output buffer and
	  `readOutputTensor({ shape | output })` when the caller wants shaped eager
	  Tensor readback into fresh storage or a caller-provided Tensor/`Float32Array`
	  carrier. Readback targets are length and f32-alignment checked before
	  native reads. Generic JS/TS Sessions also expose signed
	  `sessionCallProfile()` snapshots plus
	  `session_call_profile()`,
	  `matchesSessionCallProfileSignature(...)` /
	  `matches_session_call_profile_signature(...)`, and
	  `resetSessionCallProfile()` / `reset_session_call_profile()`, keeping
	  frontend helper-family, reset, and parameter-upload counters separate from native backend
	  `runtimeProfile()` evidence. `zgml/session` and `zgml/inspection` also
	  expose standalone call-profile predicates, assertions, and signature
	  matchers, so captured call-profile evidence remains useful after crossing
	  host, FFI, or serialization boundaries. The shared emitted Node/Bun package smoke now
	  round-trips compiled Program `runtimeProfile()` evidence plus bound Session
	  `sessionCallProfile()` and `runtimeProfile()` signatures through the public
	  package surface before proving hot StepParams compatibility. LLaMA-family JS/TS Sessions now expose the
	  same signed/resettable call-profile surface for the shared
	  Program/Session helper family, so native LLM sessions and generic compiled
	  sessions can be inspected with the same frontend evidence vocabulary. That
	  profile also includes LLaMA prefill helper counters, keeping prompt
	  ingestion visible beside step/execute/readback evidence for inference
	  loops.
  `program.bindModule(module, options)` is
  the Program-centered convenience path: it places module parameters, binds the
  Session, and makes the returned Session own those temporary parameter buffers.
  `program.moduleCompatibility(...)` and `program.acceptsModule(...)` expose the
  same evidence preflight used by direct `module.placeParameters(...)`: a
	  module's current `compileSupport()` evidence is compared with the Program's
	  retained `compileEvidence()` before placement, so module-aware binding and
	  direct placement both reject architecture, shape, kernel-plan, elided-op, or
	  packed-parameter-layout mismatches before native buffers are created.
	  The shared emitted Node/Bun package smoke now proves this preflight at the
	  public package boundary: compatible modules return frozen accepted evidence,
	  incompatible modules return frozen diagnostics, `acceptsModule(...)` follows
	  that evidence, and `bindModule(...)` rejects before creating a Session.
	  The Node and Bun smokes prove placed tensor weights/input/output, placed
  module parameters, and one-call module binding can bind compiled Programs and
  round-trip through `NativeBuffer`. Module parameters
  share those Tensor
  gradient buffers with the existing optimizers, so
  `train.backward(loss)`, `train.step(opt, { loss })`,
  `train.lossStep(opt, () => loss)`, norm clipping, and value clipping work for small
  PyTorch-like JS/TS regression and classification loops. The exact
  single-linear case compiles and binds through the native tiny-linear
  Program/Session path via `bindParameters()`, while multi-layer
  `nn.Sequential` graphs, including
  `nn.sequential([linear, relu|gelu|silu|sigmoid, linear])`, now lower through the
  native traced module Program path with the same eager parameters packed as
  persistent Session state. SiLU is now a first-class native tensor op in the
  Program substrate, which tightens the RMSNorm -> Linear+SiLU -> Linear FFN
  proof to a 5.00x module Program benchmark floor instead of paying for an
  expanded sigmoid graph. Direct tiny-MLP handles still exist as a low-level
  native Program/Session compatibility path, but `nn.Sequential` no longer
  routes ordinary MLPs through a hidden tiny-MLP matcher. Traced MLP/module Programs also
  compile for WebGPU: native `-Duse-wgpu=true` builds execute with host-bound
  buffers, while portable and non-wgpu builds expose compile-only WebGPU
  command-shape evidence. Direct tiny-MLP same-device/external-resource binding
  is still not a public support claim; traced module Programs expose the generic
  external-resource binding shape, with actual execution governed by Program
  capabilities.
  JS/TS standalone and Sequential native module Programs now cover Linear,
  supported activations, small unary elementwise modules, dim-aware rank-2
  Softmax/LogSoftmax, dim-aware rank-1/rank-2 `sum`/`mean`/`prod`/`max` reductions,
  eager/trace-visible `min`, `argmax`, and `argmin` reductions,
  LayerNorm, and RMSNorm. Batch-axis Softmax/LogSoftmax and batch-axis rank-2
  reductions lower through the same materialized transpose movement path as
  rank-2 transpose, then reuse the existing row kernel before transposing dense
  output back to the caller's shape. Batched
  `Linear.compile({ .input_shape = &.{features, batch} })` in Zig's
  feature-major layout plus JS/TS `Linear.compile({ inputShape: [batch,
  features] })` and supported `Sequential.compile({ inputShape })` graphs lower
  through that same path, bind eager parameters as persistent Session state, and
  execute with either typed-array or host `NativeBuffer` weights/input/output edges. The
	  C ABI also accepts their WebGPU external-resource binding shape for
	  resource-probe evidence, while execution still requires an actually executable
	  backend. Generic module Programs with no learnable parameters, such as
	  `nn.logSoftmax()`, use the same Program buffer API: the shared package smoke
	  now compiles one, proves the zero-persistent-state buffer layout, binds
	  typed-array and Program-owned input/output buffers, and matches eager output.
	  The public TypeScript smoke also proves this path: `ProgramBindings.weights`
	  is optional for zero-parameter Programs, and `createInputBuffer()` /
	  `createOutputBuffer()` can feed a bound Session with default retained-shape
	  tensor readback or explicit typed tensor readback.
	  Consecutive supported activation modules now lower as one bounded
	  activation-chain module descriptor. The KernelPlan records that descriptor as
	  an `activation-chain` scheduled op with the original activation IR value
	  edges retained in `fusedValueEdges`, and the C ABI advertises support through
	  `ZGML_FEATURE_NATIVE_MODULE_ACTIVATION_CHAIN`, so old native libraries cannot
	  be mistaken for runtimes that understand the packed descriptor.
	  Runtime preflight remains strict: the same package smoke proves weighted
	  Programs still reject omitted weights, and zero-parameter Programs reject
	  accidental non-empty weights before native binding.
	  The longer Node/Bun native smokes keep the WebGPU version honest with
	  device-buffer execution and zero-sync profile evidence when native wgpu is
	  enabled. The JS/TS wrapper now
  exports those compiled handles as the generic `Program` and `Session` names;
  `TinyLinearProgram` and `TinyLinearSession` remain compatibility aliases, but
  the public mental model is no longer tied to a single toy model family. The
  public smokes also prove composed classification heads
  `Sequential([Linear, LogSoftmax])` and token heads
	  `Sequential([Embedding, Linear, LogSoftmax])` lower through the same module
	  Program path and match eager execution, which is the smallest useful compiler
	  shape for PyTorch-like JS/TS model code. The shared package smoke now also
	  covers `LayerNorm` and `RMSNorm` module Programs, including path-stable
	  packed parameter layout, Program-shaped parameter placement, Session
	  inspection, retained tensor-shape readback, and eager/compiled parity for
	  batched feature-normalization inputs. JS/TS
	  Embedding module Programs now compile for explicit 1-D token
  windows (`inputShape: [tokens]`), bind the embedding table as persistent
  Session state, accept number-array and `Uint32Array`/`Int32Array` token inputs
  through `step`/bound input buffers, and reject unshaped embedding compilation
  honestly because the output length is otherwise not static. Pure shape
  modules now participate in that same audit path: `nn.Identity`/`nn.identity`,
  `nn.Flatten`/`nn.flatten`, `nn.reshape`, `nn.view`,
  `nn.squeeze`, and `nn.unsqueeze` run eagerly, propagate
  normalized trace shapes, and lower through the native module Program ABI for
  rank-1/rank-2/rank-3 shapes, so common `Flatten -> Linear` heads and tensor-rank
  glue compile instead of hitting an artificial frontend wall. The trace and
  Tensor Program IR keep each user-authored shape op, while the kernel plan
  coalesces consecutive reshape-equivalent shape ops, including identity, into one native reshape
  descriptor. Deterministic no-op shape/Dropout glue before a real op is now
  recorded as kernel-plan `elidedOps` and skipped by native descriptor emission,
  so trace/IR auditability does not force an extra runtime command. That is the
  same evidence shape used by pure terminal metadata-only shape/Dropout graphs:
  they compile as zero-dispatch native module Programs rather than synthetic
  reshape commands, with the skipped chain retained in `elidedOps`. That is the
  first small scheduler/kernelizer behavior for frontend-only model glue rather
  than another pattern matcher. `nn.narrow` now
  joins that path for rank-1 and rank-2 ranges. Contiguous spans lower as views,
  while batched feature-axis ranges materialize dense Program output through the
  same stride-aware movement seam used by transpose. `nn.select` lowers through
  the same native ABI as a one-element `narrow` span while preserving
  rank-dropping frontend evidence, including batched feature-axis selection.
	  `nn.slice` lowers through a native slice ABI for rank-1/rank-2/rank-3 ranges,
	  including positive stepped slices that materialize dense Program output when
	  the view stride is not contiguous.
	  Unsupported view-shaped ops still keep frozen Tensor
  Program IR evidence, so the rejection is owned by the Kernelizer instead of
  erasing the normalized compiler input. The trace compiler now defers
  higher-rank shape/view rank decisions for reshape/view/flatten/squeeze/
  unsqueeze/broadcast/expand/narrow/select/slice/transpose/permute to the
  Kernelizer, so unsupported shape/view ops report op-specific
  `kernelizer:unsupported-view` diagnostics instead of being hidden behind a
  coarse IR rank rejection. `nn.transpose` now joins the native
  module Program path for
  rank-2/rank-3 axis swaps: it runs eagerly, carries trace plus Tensor Program IR
  evidence, lowers through a `transpose` kernel-plan entry, and materializes the
  transposed view into a dense output buffer through the existing stride-aware
  movement kernel. Rank-3 `permute` also lowers through a short transpose chain,
  including non-envelope batched inputs. Rank-4+ transpose/permute and zero-copy strided output bindings
  remain future compiler work. `nn.broadcastTo` and `nn.expand` also lower
  through the native module Program ABI for rank-1/rank-2/rank-3 shapes as materialized
  dense repeat ops. Rank-3 `reshape`, `view`, `flatten`, `squeeze`, and
  `unsqueeze` now use the same descriptor ABI through the spare `reserved`
  dimension slot without changing the C struct layout.
  Rank-3 `nn.narrow`, `nn.select`, and `nn.slice` use the existing
  axis/start/length/step descriptor lane and materialize dense Program output
  across batch, middle, and feature axes, including non-envelope batched inputs.
  `nn.diagonal` now lowers through a native materialized
  stride-view descriptor for rank-2 matrices. `nn.repeat` and `nn.tile` now use that same ABI lane for
  rank-1/rank-2/rank-3 positive-multiple tiled repeats, including non-envelope
  batched rank-3 inputs, giving FFI callers a concrete
  output-buffer contract while leaving zero-copy broadcast/tile views and
  rank-4+ row-major tile lowering as future compiler work. Other multi-layer
  JS/TS graphs still reject explicitly until a general tensor/program compiler
  exists.
- Package smoke evidence now covers that shape/view family from the public
  Node/Bun product runtime, not only internal compiler helpers:
  rank-3 `nn.Linear` modules,
  `broadcastTo`, `expand`, `diagonal`, rank-1/rank-2/rank-3 `repeat`/`tile`
  including batched rank-3 inputs,
  row/feature-axis `narrow`, rank-2/rank-3 `select`, contiguous and stepped `slice`,
  plus terminal zero-dispatch
  rank-2/rank-3 `flatten`, `squeeze`, and `unsqueeze` all prove frozen compile evidence,
  kernel-plan dispatch/elision counts, and eager/compiled output parity; the
  same package smoke also proves batched rank-3 `transpose` plus rank-3 `permute`
  single-swap and cycle lowering through frozen Tensor Program IR, KernelPlan
  evidence, and eager/compiled parity.
- The shallowest PyTorch-like frontend seam is JS/TS module compilation beyond
  recognized native patterns. The live frontend Adapter now starts in
  `src/ts/shared_frontend.ts`: it wires the TS-authored module/compiler/runtime
  surface before preserving the single-`Linear` compatibility path and lowering
  multi-layer graphs through the supported traced module Program subset first. The
  record-shaped compiler seam now lives behind internal compiler Modules:
  `src/ts/runtime/tensor_program_ir.ts` owns Tensor Program IR construction,
  freezing, normalized attrs, value-layout evidence, and canonical signatures
  through the tsdown-emitted `dist/runtime/tensor_program_ir.cjs`,
  `src/ts/runtime/kernel_plan.ts` owns KernelPlan lowering, freezing, public
  projection, layout/signature policy, and descriptor-ready native op scheduling,
  emitted to
  `dist/runtime/kernel_plan.cjs`,
  and KernelPlan is public through `zgml/kernel_plan`, `zgml/compile`, and
  `zgml/runtime/kernel_plan`, with package and dist smokes proving the
  top-level facade is tsdown-owned rather than a handwritten compatibility lane,
  `src/ts/runtime/compiler_signatures.ts` owns canonical compiler-signature derivation
  through the tsdown-emitted `dist/runtime/compiler_signatures.cjs`
  and evidence completion shared by the trace compiler, module compiler,
  Program facade, and compatibility checker,
	  `src/ts/runtime/trace_compiler.ts` owns compile diagnostics, Trace-to-IR
	  orchestration, and public compile-evidence projection through the
	  tsdown-emitted `dist/runtime/trace_compiler.cjs`,
  `src/ts/runtime/module_compiler_policy.ts` owns Sequential constructor
  validation, nested module flattening, trace shape normalization, tiny-linear
  compatibility detection, module trace shape inference, trace-record
  construction, support-analysis orchestration, support-detail projection, and
  parameter trace projection through the tsdown-emitted
  `dist/runtime/module_compiler_policy.cjs`,
  the obsolete `js/shared_module_compiler.cjs` compatibility shim has been
  removed, and `src/ts/shared_frontend.ts` re-exports compiler policy,
  TS-authored parameter-packing helpers, and module facade policy,
  `src/ts/runtime/module_facade.ts` owns root `nn.trace`/`nn.compileSupport`/
  `nn.compile`/`nn.bindParameters`/`nn.placeParameters` facade policy through
  the tsdown-emitted `dist/runtime/module_facade.cjs`,
  `src/ts/runtime/module_bindings.ts` owns ModuleBinding metadata annotation,
  metadata lookup, `nn.placeParameters(...)` placement policy, and Program.bind
  compatibility validation through the tsdown-emitted `dist/runtime/module_bindings.cjs`, and
  `src/ts/runtime/module_compatibility.ts` owns Program/module compatibility
  through the tsdown-emitted `dist/runtime/module_compatibility.cjs`
  diagnostics and comparison outside the tensor/module facade. Node and Bun
  now instantiate one shared `TraceModuleCompiler` from that module instead of
  manually passing constructor tables through every adapter call. It then shares
  generated compiler policy, support metadata, unsupported composable-graph
  analysis, and parameter packing across Node and Bun. `Sequential.trace()` emits a normalized op tape with path-stable
  parameter metadata, and optional `trace({ inputShape })` propagates tensor
  shapes while rejecting impossible wiring. The root `nn.trace(module, ...)`,
  `nn.compileSupport(module, ...)`, `nn.compilerSignatures(module, ...)`,
  `nn.tensorProgramIr(module, ...)`, `nn.kernelPlan(module, ...)`,
  `nn.memoryLayout(module, ...)`, `nn.bufferLayout(module, ...)`, `nn.inputShape(module, ...)`,
  `nn.outputShape(module, ...)`, `nn.shapeConstraints(module, ...)`,
  `nn.parameterLayout(module, ...)`, `nn.canCompile(module, ...)`,
  `nn.compile(module, ...)`, and `nn.bindParameters(module, ...)` helpers now expose the same
  audit-to-Program vocabulary for JS/TS modules, so callers do not have to know
  whether a module is standalone or Sequential before inspecting, comparing
  compiler artifacts, sizing packed state, compiling, or binding it. The
  `zgml/compile` namespace now mirrors the same inspection vocabulary for
  `compilerSignatures`, Tensor Program IR, KernelPlan, buffer/memory layouts,
  input/output shapes, shape constraints, and parameter layout, with snake-case
  aliases plus `compileExplanation`/`compilePlan` aliases where the module
  namespace exposes them, so low-level callers can stay on the neutral compile
  facade instead of reaching through `nn`. It also exposes
  `requireCompileSupport` / `require_compile_support` and
  `requireCompilePlan` / `require_compile_plan` as explicit cold preflight
  gates: callers get structured supported evidence or, for module-owned
  preflight, the full frozen compile explanation back; unsupported targets throw
  a reasoned error before callers attempt to compile. The same namespace, plus
  `zgml/nn` and `zgml/inspection`, exposes
  standalone Module compile-plan predicates/assertions/signature checks for
  already-produced `zgml.nn.compile-explanation` evidence.
  JS/TS eager Tensor operation policy now lives in TS-authored `src/ts/core/**`
  modules re-exported by `src/ts/shared_frontend.ts`: tensor data
  conversion/gradient helpers, scalar/core Tensor methods, eager tensor
  math/autograd method implementations, Tensor construction, parameter creation,
  factory constructors, Tensor facade wiring, view helpers, joins/indexing,
  parameter views, activation scalar/derivative math, and grad-mode helpers
  share one Interface across Node and Bun. The obsolete
  `js/shared_tensor_ops.cjs` re-export shim has been removed. Tensor
  placement/readback through compiled Programs now lives separately in
	  `src/ts/runtime/tensor_placement.ts`, emitted to
	  `dist/runtime/tensor_placement.cjs`, while raw NativeBuffer edge policy now
	  lives in `src/ts/runtime/native_buffer.ts`, emitted to
	  `dist/runtime/native_buffer.cjs`. Eager JS/TS `nn.Shape` modules now
	  live in `src/ts/nn/shape_module.ts`, emitted to
	  `dist/nn/shape_module.cjs`; `src/ts/shared_frontend.ts` injects the
		  placement hook, and the TS-source smoke pins construction,
	  forward dispatch, no-parameter state behavior, compile support, bind, place,
	  compile, and validation errors. Stateless eager JS/TS
	  `nn` activation, softmax/logSoftmax, reduction, and Dropout modules now live in
	  `src/ts/nn/parameterless_modules.ts`, emitted to
	  `dist/nn/parameterless_modules.cjs`; `src/ts/shared_frontend.ts` injects
		  the placement hook, and the TS-source
	  smoke pins tensor/host forward dispatch, dropout training/eval behavior,
	  dropout autograd mask routing, empty parameter binding, placement,
	  compile support, unsupported native dropout rejection, no-parameter state
	  behavior, and validation errors. Shared no-parameter module traversal,
	  parameter metadata, state-dict/load-state, zero-grad, requires-grad, and
	  train/eval policy now lives in `src/ts/nn/parameterless_state.ts`, emitted
	  to `dist/nn/parameterless_state.cjs`, so shape/view modules and stateless
	  activation/dropout-style modules no longer carry separate empty-parameter
	  policy copies. The TS architecture guard pins that shared helper so adding
	  another parameterless module does not create another source-level surface
	  to reconcile. Shared `nn` module traversal helpers now live in
	  `src/ts/nn/module_tree.ts`, emitted to `dist/nn/module_tree.cjs`, so leaf
	  modules and Sequential use one traversal-entry vocabulary instead of
	  copying tiny `children`/`modules`/`named*` helpers across module files.
	  Shared stateful module lifecycle/state methods now live in
	  `src/ts/nn/stateful_module_state.ts`, emitted to
	  `dist/nn/stateful_module_state.cjs`, so Linear, Embedding, Sequential,
	  LayerNorm, and RMSNorm keep one parameter metadata, zero-grad,
	  requires-grad, train/eval, state-dict, and load-state surface while their
	  module files stay focused on forward math, parameter layout, and compile
	  behavior.
	  Shared single-module compile/bind/place methods now live in
	  `src/ts/nn/single_module_compile.ts`, emitted to
	  `dist/nn/single_module_compile.cjs`, so Shape, stateless activation/
	  reduction/dropout modules, Embedding, LayerNorm, and RMSNorm use one
	  generic Program compile surface while preserving their own diagnostics and
	  parameter binding differences. Linear and Sequential keep their specialized
	  tiny-linear and composition paths because those are real frontend behavior,
	  not generic ceremony.
	  Shared sequential Program compile methods now live in
	  `src/ts/nn/sequential_program_compile.ts`, emitted to
	  `dist/nn/sequential_program_compile.cjs`, so Linear and Sequential share
	  sequential analysis, support evidence, placement, device-program compile
	  delegation, and tiny-linear compile-evidence attachment while keeping their
	  genuinely different parameter binding paths local.
	  Eager JS/TS `nn.Linear` now lives in
		  `src/ts/nn/linear_module.ts`, emitted to `dist/nn/linear_module.cjs`;
		  `src/ts/shared_frontend.ts` injects the placement hook, and the TS-source smoke pins parameter initialization,
	  vector/batch forward math, state/metadata, binding/placement, tiny-linear
	  compile evidence, device-program fallback compile delegation, and validation
	  errors. Eager JS/TS `nn.Embedding` now lives in
	  `src/ts/nn/embedding_module.ts`, emitted to
	  `dist/nn/embedding_module.cjs`; `src/ts/shared_frontend.ts` injects the
	  placement hook, and the TS-source smoke
	  pins indexed gather output, repeated-index gradient accumulation,
	  state/metadata, binding/placement, input-shape compile diagnostics,
	  device-program compile delegation, and validation errors. Eager JS/TS
	  `nn.Sequential` now lives in `src/ts/nn/sequential_module.ts`, emitted to
	  `dist/nn/sequential_module.cjs`; `src/ts/shared_frontend.ts` injects the
	  placement hook, and the TS-source smoke
	  pins ordered forward composition, nested traversal, child training-mode
	  propagation, parameter naming/metadata, trace, bind/place, tiny-linear
	  delegation, device-program compile delegation, and validation errors. Eager
		  JS/TS feature-axis `nn.LayerNorm` and `nn.RMSNorm` now live in
		  `src/ts/nn/feature_norm_module.ts`, emitted to
		  `dist/nn/feature_norm_module.cjs`; `src/ts/shared_frontend.ts` injects the
		  placement hook, and the TS-source smoke
		  pins forward math, Tensor autograd, affine/no-affine state and metadata,
		  bind/place, device-program compile delegation, and validation errors. Shared
		  `nn` namespace construction now lives in `src/ts/nn/namespace.ts`, emitted
		  to `dist/nn/namespace.cjs`; `src/ts/shared_frontend.ts` wires the
		  scalar helpers, and the TS-source smoke pins PyTorch-style
		  constructor aliases, scalar activation aliases, shape/norm factories, and
		  compile/state facades across Node and Bun.
	  Module `stateDict`/`loadStateDict`, parameter metadata, compile-support
	  metadata, `zeroGrad`, `requiresGrad` policy, optimizer config parsing, and
	  optimizer snapshot validation have moved to `src/ts/train/module_state.ts`,
	  emitted to `dist/train/module_state.cjs`. Stateful module classes now
	  delegate `parameterNames` / `parameterInfos` / `parameterInfo`
	  shape-layout evidence to that shared helper instead of each carrying a
	  local copy of the metadata policy.
	  The training surface is split by ownership instead of routed through one
	  broad hub. Checkpoint composition plus optimizer/scheduler namespace wiring
	  live in `src/ts/train/training.ts`, emitted to `dist/train/training.cjs`,
	  while public `loss`, `train`, `optim`, and shared frontend facades import
	  simple factories directly from their owning helper modules. Checkpoint JSON
	  serialization helpers now live in
	  `src/ts/train/checkpoint_serialization.ts`, emitted to
	  `dist/train/checkpoint_serialization.cjs`, so plain-record validation,
	  JSON-safe metadata, tensor-entry normalization, scheduler snapshot parsing,
	  and checkpoint inspection records are shared instead of embedded in the broad
	  training namespace factory. Optimizer update classes now live in
	  `src/ts/train/optimizer_classes.ts`, emitted to
	  `dist/train/optimizer_classes.cjs`, so SGD/Adam/AdamW math,
	  parameter-group validation, config signatures, state snapshots, and
	  strict/validation-only restore are shared instead of embedded in the broad
	  training namespace factory. Loss and tiny training helpers now live in
	  `src/ts/train/loss_train_helpers.ts`, emitted to
	  `dist/train/loss_train_helpers.cjs`, so MSE/CrossEntropy losses,
	  gradient clipping/norms, `train.step`, `train.lossStep`, and `train.fit`
	  evidence share one TS-authored frontend contract instead of living in the
	  broad training namespace factory. `zgml/train` also exposes standalone
	  TrainStep/TrainFit/TrainFitStep evidence predicates, assertions, require
	  helpers, and signature matchers, so training-loop evidence can cross
	  callbacks, package boundaries, or logs without losing its proof shape.
	  Optimizer config/state and LR-scheduler snapshots now share standalone
	  predicates, assertions, require helpers, and signature matchers from
	  `src/ts/train/optimizer_snapshot.ts`, re-exported through `zgml/optim`
	  and the `optim` namespace, so PyTorch-style state surfaces stay pleasant
	  while still carrying explicit proof records.
	  Optimizer learning-rate scheduler
	  state/signature helpers now live in `src/ts/train/scheduler_state.ts`,
	  emitted to `dist/train/scheduler_state.cjs`, so `StepLR` and
	  `ExponentialLR` share snapshot signatures, `step_size` compatibility, and
	  validation-only restore behavior instead of burying that policy inside the
	  namespace factory. Module
		  training-mode normalization and recursive child propagation have moved to
		  `src/ts/train/module_mode.ts`, emitted to
		  `dist/train/module_mode.cjs`, while Node and Bun adapters keep only host
	  imports and concrete constructor wiring. Program facade evidence/capability/lifecycle
	  policy now lives in `src/ts/runtime/program_facade_policy.ts`, emitted to
	  `dist/runtime/program_facade_policy.cjs`; the TS-source smoke pins
	  requirements projection, buffer-layout derivation, trace/IR/KernelPlan views,
	  model/module compatibility predicates, capability/runtime-profile signature
	  matching, reset, and idempotent free/dispose behavior. Program-owned buffer
	  factory routing now lives in `src/ts/runtime/program_buffer_factory.ts`,
	  emitted to `dist/runtime/program_buffer_factory.cjs`; the TS-source
		  smoke pins host/device/resource buffer creation, named buffer validation,
		  LLaMA KV buffer routing, delegated resource-backed KV-cache construction, and
		  callback/error validation. `Program.bindModule(...)` routing now lives in
		  `src/ts/runtime/program_module_binding.ts`, emitted to
		  `dist/runtime/program_module_binding.cjs`; the TS-source smoke pins
		  compatibility preflight, module parameter placement, owned-buffer
		  deduplication, cleanup-on-failure, and validation errors. `ProgramDevice`
		  policy now lives in `src/ts/runtime/program_device.ts`, emitted to
		  `dist/runtime/program_device.cjs`; the TS-source smoke pins placement
		  liveness, role-specific device buffers, same-device import defaults,
		  KV-cache buffer routing, custom placements, and validation errors. Program
		  buffer sizing/signature policy now lives in
		  `src/ts/runtime/program_sizing.ts`, emitted to
		  `dist/runtime/program_sizing.cjs`; the TS-source smoke pins byte
		  fallback sizing, explicit byte overrides, frozen snapshots, and signature
		  matching, plus the shared Program sizing accessor table used by both
		  generic and LLaMA-family Program facades. Generic Program parameter projection and
		  accessor table now live in
		  `src/ts/runtime/program_parameters.ts`, emitted to
		  `dist/runtime/program_parameters.cjs`; the TS-source smoke pins
		  frozen names/infos arrays, lookup by name/index, misses, accessor-table
		  routing, and validation errors. Program edge-shape evidence selection and
		  shape accessor tables now live in
		  `src/ts/runtime/program_shapes.ts`, emitted to
		  `dist/runtime/program_shapes.cjs`; the TS-source smoke pins compiled
		  shape evidence preference, descriptor fallback, frozen snapshots, and
		  LLaMA vocab-sized output shapes, accessor-table routing, and validation
		  errors. Program resource accessors now live in
		  `src/ts/runtime/program_resources.ts`, emitted to
		  `dist/runtime/program_resources.cjs`; the TS-source smoke pins
		  create-buffer routing, role buffers, device handles, imports, liveness, and
		  validation errors. Program layout accessors now live in
		  `src/ts/runtime/program_layout.ts`, emitted to
		  `dist/runtime/program_layout.cjs`; the TS-source smoke pins cached
		  buffer layout, slot names/lookups, labeled slot errors, KV-cache
		  requirements/layout, and liveness. Generic Program policy accessors now
		  live in `src/ts/runtime/program_policy_accessors.ts`, emitted to
		  `dist/runtime/program_policy_accessors.cjs`; the TS-source smoke
		  pins compile-evidence normalization, compiler artifact forwarding,
		  compatibility/capability/runtime-profile forwarding, bind liveness, and
		  lifecycle calls. LLaMA-family Program policy forwarding lives there too;
		  the TS-source smoke pins cached inspection, vocab size, model/capability
		  forwarding, KV-cache creation, executable inspection, bind handoff,
		  runtime profiles, lifecycle, and validation errors. The generic and
		  LLaMA-family Program facade factories now live in
		  `src/ts/runtime/program_facade.ts`, emitted to
		  `dist/runtime/program_facade.cjs`; the TS-source smoke pins factory
		  assembly for generic and LLaMA-family Programs, and
		  `src/ts/shared_frontend.ts` re-exports the TS-authored Program helpers.
	  Generic Program edge-shape and
  Session tensor-output shape/readback helpers now live in
  `src/ts/runtime/session_tensor.ts` emitted as
  `dist/runtime/session_tensor.cjs`. Shared StepParams compatibility
  evidence, frozen diagnostics, ownership/effect classification, hot-path
  signature construction, validation-error shaping, StepParams object
  validation, token-window fit checks, logits-output validation,
  no-inline-output guards, reusable token scratch-option normalization, and
  StepParams predicate/signature matching now live in
  `src/ts/runtime/step_params.ts`, emitted as
  `dist/runtime/step_params.cjs`; generic and LLaMA-family Session
  facades now import that TS-authored helper for their StepParams contract
  vocabulary. Session buffer-sizing signatures and generic/LLaMA step-contract
  record construction now live in `src/ts/runtime/session_contract.ts`, emitted
  as `dist/runtime/session_contract.cjs`; Session facades still read live
  adapter state in CJS, but cold evidence construction is TS-authored. Frontend
  Session call-profile counters, signature construction, matching, and reset
  now live in `src/ts/runtime/session_profile.ts`, emitted as
  `dist/runtime/session_profile.cjs`, so generic and LLaMA-family
  Sessions share one TS-authored profile record. Generic Session parameter
  names/infos/info lookup, upload-by-name validation, and persistent
  binding-index selection now live in `src/ts/runtime/session_parameters.ts`,
  emitted as `dist/runtime/session_parameters.cjs`, while adapter-local
  FFI still performs the actual upload call. Generic and LLaMA-family Session
  slot lookup, KV-cache layout presence checks, shape/length/byte-length
  projection, and bound-buffer kind classification now live in
  `src/ts/runtime/session_layout.ts`, emitted as
  `dist/runtime/session_layout.cjs`; that module also owns LLaMA default
  Session buffer-layout construction, reusable Session scratch allocation, and
  NativeBuffer-output binding checks plus host/native bound input/output
  selectors. LLaMA-family `Program.bind(...)` route selection, output
  normalization, inspection handoff, and owned-output cleanup now live in
  `src/ts/runtime/session_binding.ts`, emitted as
  `dist/runtime/session_binding.cjs`. CJS facades still own live Session
  handles, FFI callbacks, and execution calls, but the frontend/runtime policy
  source of truth is TS and CJS should shrink toward generated adapter glue.
  LLaMA and generic Session cleanup/owned-resource release policy now lives in
  `src/ts/runtime/session_lifecycle.ts`, emitted as
  `dist/runtime/session_lifecycle.cjs`; adapters supply the native
  handle-free callback and null-handle sentinel.
  Generic Session explicit input/output preparation,
  execute/executeTensor/executeInto plan routing, no-output result validation,
  step output target/readback routing, read-output tensor target planning, and
  LLaMA execute/executeTensor/executeInto method planning, LLaMA stepTensor
  output validation, LLaMA execute-token output/readback routing, LLaMA
  read-output tensor target planning, and generic plus LLaMA StepParams
  compatibility construction plus LLaMA prefillTensor/generation/sample,
  token-selection/generation token-window planning, and native-output
  preconditions for token selection/generation now live in
  `src/ts/runtime/session_values.ts`, emitted as
  `dist/runtime/session_values.cjs`; facades supply host-value callbacks
  and keep the native step call. NativeBuffer
  shape algebra now has its own TS source module,
  `src/ts/core/shape.ts`, emitted into `dist/core/shape.cjs`: tensor
  shape validation, factory/view shape normalization, row-major strides,
  broadcast and reduction plans, dim insertion/selection, and trace index/slice
  bounds share one TS-authored package artifact across eager Tensor operations,
  live module trace adaptation, and the JS/TS Trace-to-Program Compiler instead
  of being reimplemented in each layer. Tensor data normalization now follows
  the same pattern: `src/ts/core/tensor_data.ts` emits
  `dist/core/tensor_data.cjs`, and `createTensorDataHelpers` owns raw
  f32 conversion, nested rectangular input checks, byte views, scalar tensor
  construction, gradient accumulation, and shaped length validation from the TS
  source path. Scalar activation math and eager grad-mode state now live in
  `src/ts/core/activation.ts` and `src/ts/core/grad_mode.ts`, emitted into
  `dist/core/`, so GELU/SiLU/Sigmoid numerics and
  `noGrad`/`enableGrad`/`inferenceMode` state restoration are shared through
  TS-authored checkout artifacts too. LLaMA token/window/sample/result helpers now live
  in `src/ts/core/token.ts`, emitted as `dist/core/token.cjs`, so token
  validation, token windows, sampling defaults, generated-token output checks,
  and ABI token-selection/generation result shaping are TS-sourced as well.
  NativeBuffer
	  construction, wrapping, read/write/readback, device-import, and deterministic
	  free/dispose policy now live behind one shared facade too, so Node and Bun
	  adapters pack only runtime-specific FFI calls while exposing the same buffer
	  object semantics. Generic tensor Session low-level native step/no-output
	  callbacks, upload/reset/profile hooks, and owned-buffer cleanup live in
	  `js/shared_session_facade.cjs`; generic Session
	  bound host/native input-output interpretation, explicit input/output
	  preparation, core step/advance execution, StepParams compatibility,
	  step/stepTensor/stepInto and execute/executeTensor/executeInto/advance
	  orchestration plus LLaMA executeTokens/execute/executeTensor/executeInto/
	  advanceTokens, step/stepTensor/stepInto/advance, and
	  prefill/prefillTensor/prefillInto, argmax/sample token selection, and
	  native token-window generation orchestration now live in
	  `src/ts/runtime/session_facade.ts`, while Program edge-shape and Session
	  tensor-output wrapping/readback helpers live in `src/ts/runtime/session_tensor.ts`,
	  and shared Session StepParams facade predicates live in
	  `src/ts/runtime/session_facade.ts` alongside live-session
	  handle/inspection/position accessors, Session layout/shape/size accessors,
	  live-guarded Session call-profile facade wrappers, buffer-sizing facade assembly/signature matching, and
	  runtime-profile read/match/reset wrappers plus generic Session parameter
	  inspection/upload facade methods, readOutputInto/readOutputTensor facade
	  wrappers, LLaMA scalar-token and token-window scratch facade helpers, and
	  reset/free/dispose lifecycle facade
	  wrappers,
	  so Node and Bun cannot drift on the placement/readback contract while exposing
	  the same adapter-local FFI handles.
  The shared core also owns the generic
  Program edge-shape helpers used by both adapters, plus module Program
  descriptor projection and op-word packing from KernelPlan evidence, keeping
  `inputShape()`, `outputShape()`, and Session buffer-layout derivation tied to
  one frontend contract. Tensor placement and direct Program bind policy now
  live in `src/ts/runtime/tensor_placement.ts`, emitted to
  `dist/runtime/tensor_placement.cjs`: eager CPU/f32 movement, `Tensor.place(...)`
  slot checks, direct `Program.bind(...)` slot validation, host/native-buffer
  binding preparation, temporary weights/bias Program buffers, retained bound
  tensor shape, logical NativeBuffer bind-field selection, packed Program
  weight/bias length helpers, and cleanup of owned bind buffers share one
  Interface across Node and Bun. Compile-envelope
  validation for backend, context length, and batch is shared there too, leaving
  each adapter to wrap the shared descriptor words in its host-specific C
  pointer/object shape. LLaMA bind
  option normalization, including
  `output: "native"` / `true` output-buffer ownership, also lives in the shared
  session facade helper while adapters create the actual host buffers and native
  descriptors. TinyLLaMA and generic LLaMA-family Program bind lifecycle policy
  is shared there too: Program inspection, buffer layout, KV-cache layout, bind
  option normalization, compatible-model/KV-cache/persistent-output bind
  decision policy, owned-output cleanup on native bind failure, and concrete
  Session factory handoff. Fixed-family LLaMA Program subclasses only choose
  their concrete Session constructor. Direct generic `Program.bind(...)` preflight now uses
  the same shared slot validator too: required weights, optional bias/input,
  larger caller-owned outputs, host tensor lengths, and `NativeBuffer`
  byte-length checks all share one JS/TS rule, and the matching host/native
  binding preparation now shares one placement Module policy for temporary
  weights/bias device buffers, retained bound tensors, output-shape retention,
  and cleanup after placement failures. Node and Bun keep only host-specific FFI descriptor
  wrappers, native session-bind calls, and `Session` construction. Generic
	  `Program.bindModule(...)` policy is shared as
	  well: module compatibility, parameter placement, owned `NativeBuffer`
	  deduplication, and cleanup-on-failure now live in
	  `src/ts/runtime/program_module_binding.ts` while
	  Node and Bun only assert Program liveness and call their FFI-backed
  `Program.bind(...)`. Program capability derivation is shared as well:
  `mode`, stable capability `signature`, `matchesCapabilitySignature(...)`,
  `canExecute()`, external-resource support, full-dispatch-plan checks, and
  runtime diagnostics are derived from inspection evidence in one frontend
  helper rather than separately in each adapter. Runtime ABI feature decoding
  plus the required feature-mask, ABI-version, and token-width compatibility
  handshake, runtime snapshot shaping, ABI struct-size catalog iteration, and
  supported-checkpoint catalog iteration are shared too, so Node and Bun cannot
  drift on wrapper capability policy or runtime evidence while loading the same
  native library. Runtime facade and host adapter catalog calls now share
  `src/ts/runtime/model_source_catalog.ts` for count validation and frozen
  iteration, leaving Node and Bun to supply only their ABI-specific inspection
  call shape. Numeric ABI vocabulary such as model kinds, backend ids,
  buffer storage ids, Program buffer kinds, and ABI struct-kind ids now comes
  from `src/ts/runtime/abi.ts`, emitted to `dist/runtime/abi.cjs`
  and re-exported by the shared frontend
  facade, along with runtime ABI facade helpers and compile-envelope
  normalization, default-envelope detection, and descriptor record projection
  for backend, context length, and batch. That leaves the adapters
  to own host-specific FFI calls rather than duplicated descriptor policy while
  keeping ABI churn and runtime descriptor validation out of tensor/module
  frontend code. Buffer, model, Program, Session, requirement, compatibility,
  and runtime profile evidence decoding now come from an internal
  `src/ts/runtime/inspection.ts` emitted to `dist/runtime/inspection.cjs`
  and re-exported by the shared frontend facade,
  keeping native ABI field maps, frozen diagnostic shaping, and capability
  derivation in one place while host adapters only fetch records/words from
  their FFI layer. Compatibility booleans and capability aliases live with that
  same evidence decoder, so `canExecute()`-style helpers cannot drift from the
  `Program.inspect()` / `Program.capabilities()` records they interpret.
	  Program buffer layout projection, buffer callback slot evidence, resource
	  slot validation, byte-length checks, and LLaMA KV-cache layout/slot
	  policy now come from `src/ts/runtime/program_buffers.ts`, emitted to
  `dist/runtime/program_buffers.cjs`,
  re-exported by the shared frontend facade, keeping Program/Session placement
  contracts aligned while host adapters keep native buffer construction and FFI
	  descriptor packing. NativeBuffer edge policy now comes from
	  `src/ts/runtime/native_buffer.ts`, emitted to
	  `dist/runtime/native_buffer.cjs` and re-exported by that facade: WebGPU import
  source normalization, external-resource option validation, create/wrap
  byte-length validation, frozen NativeBuffer byte-range evidence for
  write/readback calls, standalone byte-range predicates/assertions/signature
  matching, signed external-resource and device-import evidence with matching
  predicates/assertions/signature matchers, caller-owned readback target
  validation, Session-owned buffer retain/dedup rules, and deterministic
  free/dispose behavior stay shared while
  adapters supply concrete native
  callbacks. The
	  compatible-checkpoint load-kind aliases, native model-kind mapping,
	  safetensors path/byte source classification, path header-prefix parsing,
	  source-to-probe/load routing, load-kind dispatch, and safetensors header/data
	  byte normalization now live in `src/ts/runtime/model_source.ts`, emitted to
	  `dist/runtime/model_source.cjs` and re-exported by the shared frontend
	  facade, so Node and Bun cannot diverge on
	  path/byte model selectors while still owning their host-specific filesystem
	  calls, model construction, and descriptor packing. Program model-handle
	  acceptance and liveness policy for bind/model-compatibility preflights now
	  live in that same TS-authored Module. Model facade lifecycle
	  policy and LLaMA-family create/load/byte-load/probe/header-probe routing now
	  live there too, while adapters still provide their runtime-specific class
	  predicates, native handle extraction, native source callbacks, and concrete
	  Model / Program constructors.
  Module Program descriptor projection from KernelPlan evidence, descriptor
  validation, and op-word packing now live in an internal
  `src/ts/runtime/module_program_desc.ts` Module descriptor packer re-exported by the shared frontend
  facade, so Node and Bun consume one native descriptor layout while keeping
  their host-specific pointer/object packing local.
  External-resource access flag parsing, external-resource option validation,
  and inspection decoding also share one policy, keeping resource binding
  semantics aligned while each adapter keeps its own descriptor packing.
  Program device-buffer import-source normalization and device/buffer/byte-range
	  validation now produce signed TS-authored evidence on the same principle,
	  and `NativeBuffer` same-device import provenance now uses the same
	  TS-authored native-buffer rule: Node
  and Bun pack different FFI descriptors but accept and reject the same public
  import shapes.
  NativeBuffer live-handle validation, typed-array wrapping, create-size
  validation, write byte-range validation, default full-buffer readback
  allocation, caller-owned readback target/range/alignment validation, and
  Session-owned buffer lifetime deduplication now share that edge policy too, so
  byte and f32 buffer I/O errors stay aligned across both adapters. Program buffer
  resource callback slot shaping, factory
  validation, and callback-produced buffer byte-length checks now come from
  `src/ts/runtime/program_buffers.ts`, so callbacks for weights, bias, input, and output
  receive the same `kind`, element/byte length, layout, and raw slot evidence
  and fail with the same edge errors regardless of whether the host adapter is
  Node or Bun. LLaMA KV-cache resource factories now use the same shared rule
  for both per-layer `createKvCache({ resource })` callbacks and role-level
  `kv-k` / `kv-v` Program buffer callbacks, including resource factory
  validation, slot evidence, and byte-length checks. LLaMA token-window
  normalization, uint32 token validation, generated-token output validation,
  sampling defaults, and native token-selection/generation result shaping come
  from `src/ts/core/token.ts` through the generated
  `dist/core/token.cjs` artifact re-exported by the shared frontend
  facade. LLaMA Session policy now lives in
  `js/shared_session_facade.cjs`: Program bind decision policy, StepParams
  compatibility callbacks, generated StepParams facade predicates, output-buffer
  preconditions, NativeBuffer-output requirements, scalar/no-output scratch
  option shaping, token-window `executeTokens` output-selection/readback,
  tensor-output wrappers for step/execute/prefill, bound-output
  `readOutputInto` / `readOutputTensor` readback, default buffer-layout
  projection, one-token scratch allocation, reusable token-window option
  scratch, output shape, KV-cache layout errors, argmax/sample/generation
  preconditions, sampling-option normalization, `maxTokens` checks, generated
  token output validation, lifecycle inspection, reset/profile, and
  free/dispose follow one Interface. Adapters only pack already normalized
  callback inputs into their host-specific descriptors, keeping Node and Bun
  aligned on StepParams-like public token semantics while adapters only own FFI
  packing.
	  Generic and LLaMA-family Program facade helpers now share compile-evidence
	  accessors, module-IR/kernel-plan/shape/parameter-layout extraction, compatible
	  result booleans, and capability aliases too. Program facade wrapper, evidence,
	  capability, device/buffer factory, and lifecycle policy now comes from
	  `src/ts/runtime/program_*.ts` modules re-exported by `src/ts/shared_frontend.ts`;
			  compile-evidence accessors come from `src/ts/runtime/trace_compiler.ts`,
		  emitted to `dist/runtime/trace_compiler.cjs`;
	  module trace/compiler policy comes from
	  `src/ts/runtime/module_compiler_policy.ts`; the obsolete
	  `js/shared_module_compiler.cjs` shim has been removed, while root module facade policy comes from
		  `src/ts/runtime/module_facade.ts`; eager JS/TS `nn` module classes and namespace
		  construction come from `src/ts/nn/**`, emitted to `dist/nn/**`, with
		  `src/ts/shared_frontend.ts` injecting TS-authored placement policy and scalar
		  helper wiring; module compatibility comparisons come from
	  `src/ts/runtime/module_compatibility.ts`;
  Tensor Program IR construction and signature evidence come from
  `src/ts/runtime/tensor_program_ir.ts`, emitted to
  `dist/runtime/tensor_program_ir.cjs`; and KernelPlan lowering/projection/signature
  policy is imported from the TS `runtime/kernel_plan` module, leaving adapters to fetch native inspection
  objects rather than reinterpret them differently. Buffer
  inspection, model inspection, Program inspection, Session inspection, Program
  requirements, LLaMA KV-cache requirements, Program/model compatibility, and
  runtime profile decoding now share field maps in `src/ts/runtime/inspection.ts`,
  so storage/resource
  metadata, model envelopes, Program capabilities, diagnostics, binding
  requirements, dispatch-plan evidence, Session binding-shape evidence,
  compatibility preflights, and Program/Session profile snapshots expose the
  same patch, command, dispatch, sync, fallback, and stencil-hash evidence in
  Node, Bun, and the published TS declarations.
  `compileSupport(...)` now carries a frozen copy of that normalized trace,
  including shape-specialized input/output evidence for compiled module
  Programs, so callers can inspect capability, lowering shape, and parameter
  names in one pre-compile audit call. The published TS `Tensor` Interface now
  preserves literal factory, reshape/view, placement-readback, and
  from-native-buffer shapes where callers pass literal tuples, giving eager
  Tensor code a lightweight static bridge into those shape-specialized compile
  queries without changing runtime semantics. Batched
  compile remains shape specialization at `compile` / `compileSupport` /
  `bindParameters` time, so eager modules stay reusable while executable
  Programs get the static shape evidence they need. Unsupported composable
  graphs therefore have a concrete compiler input artifact plus frozen
  structured diagnostics such as `missing-input-shape`, `shape-mismatch`,
  `unsupported-rank`, `unsupported-view`, and `unsupported-op` instead of only a
  string reason. Each diagnostic now carries a `stage`
  (`trace`, `ir`, `kernelizer`, or `support`) so callers can distinguish shape
  propagation failures, normalized-IR formation failures, native-lowering gaps,
  and layer capability gaps without parsing prose. The public JS/TS diagnostic
  type is now a closed discriminated Interface, so those stage/code pairs expose
  typed shape, op, or layer evidence instead of an arbitrary field bag. The
  public trace, Tensor Program IR, KernelPlan, fused shape-op evidence, and
  Program/module compatibility diagnostics now use closed JS/TS vocabularies as
  well: unsupported modules can still appear as an honest `unknown` trace op,
  eval-mode `BatchNorm1d` preserves typed trace/IR evidence and lowers through
  a native affine KernelPlan with derived running-stat scale/offset bindings,
  while training-mode `BatchNorm1d` still honestly stops at a
  `kernelizer:unsupported-op` native-lowering diagnostic. Fixed native-subset `Conv2d`,
  `MaxPool2d(2)`, and `AvgPool2d(2)` shapes now go further and compile to
  named native Program kernels. Unsupported Conv/Pool shapes still stop with
  structured diagnostics. Executable KernelPlans can only claim the
  runtime's named op/kernel set, and compatibility preflights expose typed mismatch codes from the
  compatibility Module rather than arbitrary diagnostic records. Those records
  are discriminated by their op/kernel/code
  tags too: `linear` trace evidence carries linear dimensions, activation
  KernelPlan entries can only name activation kernels, `shape-chain` is the only
  KernelPlan entry with fused shape-op metadata, parameter layouts only name
  parameter-bearing ops, and length/shape compatibility mismatches carry the
  matching preflight fields. Shape-module trace records also freeze their
  op-local target-shape arrays, so retained trace evidence cannot be mutated
  underneath `compileSupport()` or `Program.compileEvidence()`. Explicit shape mismatches
  return that diagnostic and bypass legacy single-linear compatibility, so
  `canCompile()` cannot claim support for a Program shape the trace compiler
  would reject. If trace lowering succeeds and native kernel lowering fails,
	  unsupported support keeps the frozen Tensor Program IR plus top-level
	  input/output shapes and scalar lengths as partial compiler evidence for
	  kernelizer and support-layer rejections instead of collapsing the result back to a
	  trace-only rejection.
  Supported native module Programs now expose frozen
  `ir` and backend-neutral `kernelPlan` evidence
  from `compileSupport()`, giving the first explicit Trace -> Tensor Program IR
  -> native module kernelizer seam; descriptor emission now consumes the shared
  KernelPlan-derived module Program descriptor rather than bypassing it in each
  Adapter. The JS/TS Tensor Program IR now
  exposes the same essential dataflow vocabulary as the Zig `TensorProgramIr`
  Module: frozen values, a graph input value, a graph output value, and per-op
  input/output value IDs. Those values now carry frozen `f32` dtype,
	  scalar-byte width, rank, strides, dense row-major storage-layout, and
	  storage-offset evidence too, making the JS/TS IR value table closer to native
	  typed `IrValue` shape/layout evidence. KernelPlan scheduled ops and
	  scheduler signatures preserve that scalar evidence plus each op's
	  `nativeDispatchCount`, `nativeDescriptorCount`, `nativeKernels`, and the
	  descriptor payload that will be packed for the native module Program, so typed
	  IR does not collapse back to shape-only or aggregate-only evidence at the
	  Kernelizer boundary. The package smoke now proves same-shape `narrow` graphs
	  with different descriptor offsets produce different IR and KernelPlan
	  signatures before module binding. Backend-neutral and WebGPU `softmax(0)`
	  / `logSoftmax(0)` evidence still keeps the explicit
	  transpose + row-softmax + transpose descriptors, while CPU compile planning
	  now emits the direct native axis descriptor after proving the native Program
	  substrate has CPU/reference `inner` geometry for strided/grouped `softmax`
	  and `logSoftmax` execution through the C ABI. Metal/WebGPU capability plus
	  dispatch-plan tests keep accelerator evidence row-only until matching
	  kernels exist. Module
	  `min(dim)` now has the same single-op KernelPlan contract as sum/mean/max
	  on the native reduction axis, with batch-axis lowering kept to
	  transpose + min + transpose instead of transpose + neg + max + neg +
	  transpose. It also
	  proves zero-dispatch elided shape chains carry empty public
	  descriptor-signature arrays as intentional evidence, while still deriving the
	  same canonical compiler signatures. Fused KernelPlan entries now also retain
	  frozen per-fused-op value-edge evidence, and the KernelPlan signature includes
	  those internal fused edges rather than only the aggregate scheduled edge.
	  Bounded activation chains use that same evidence shape: one native descriptor
	  and dispatch, with the original activation IR value edges preserved for
	  inspection and compatibility checks. IR ops now
  carry normalized `attrs` records rather than extending trace records, and the
  published TS Interface names a standalone `ModuleTensorProgramIrOpKind`
  instead of deriving IR op identity from `ModuleTraceOp`. The JS/TS Kernelizer
  helper vocabulary now consumes IR ops at this seam instead of trace-specific
  fields such as top-level activation names or parameter lists. Program/module
  compatibility now compares a canonical Tensor Program IR signature before the
  KernelPlan signature, so changed value layout, dataflow edges, or normalized
  attrs fail as `ir-mismatch` before backend scheduling differences. KernelPlan ops retain those value edges, and compatibility
  signatures include them, so a changed dataflow edge is visible even when op
  names and shapes still match. Consecutive reshape-equivalent shape
  ops remain visible in trace/IR evidence but lower as one `shape-chain`
  reshape kernel-plan entry, and deterministic no-op glue before a real op is
  exposed in `elidedOps` while being omitted from native dispatch descriptors,
  while pure terminal metadata-only shape/Dropout chains lower to zero native
  dispatches with the skipped chain preserved in `elidedOps`,
  so the compiler can optimize metadata-only view glue without erasing frontend
  auditability. Kernel plans also carry exact
  `shapeConstraints` for the compiled input/output ranks, shapes, and scalar
  counts, making shape-specialized scheduling explicit evidence rather than an
  inference from scattered fields. They also carry `memoryLayout`, a frozen
  typed value-storage table with step-input, persistent, scratch, and
  step-output storage classes, per-value producer/consumer lifetime evidence,
  global and buffer-local scalar/byte offsets, byte lengths, scratch size, and
  total arena size derived from IR values. Scratch offsets are conservatively
  reuse-aware when lifetimes do not overlap,
  `bufferLayout`, a frozen
  description of step input/output and persistent weights/bias slots with
	  scalar counts and byte ranges, plus `parameterLayout`, a frozen map from
	  trace-stable parameter names to their packed `weights` / `bias` binding
	  offsets, so callers can audit value memory, buffer, and parameter state layout
	  before binding a Session instead of inferring it from total lengths. The same
	  compile-support and Program compile-evidence records now carry a frozen
	  `compilerSignatures` record for IR, KernelPlan, memory-layout,
	  parameter-layout, and buffer-layout signatures, plus the flat `irSignature`,
	  `kernelPlanSignature`, `memoryLayoutSignature`, `parameterLayoutSignature`,
	  and `bufferLayoutSignature` compatibility aliases produced by the compiler
	  Modules. Canonical signature derivation now lives in
	  `src/ts/runtime/compiler_signatures.ts`, consumed through
	  `dist/runtime/compiler_signatures.cjs`, so the trace compiler, root
	  `nn.compilerSignatures(...)`, Program evidence helpers, and
	  Program/module compatibility preflights share one signature policy instead
	  of recomputing private compatibility keys. The shared Program facade now normalizes attached module compile
	  evidence at the Program seam, so artifact-rich older records regain the
	  canonical frozen `compilerSignatures` record and flat aliases before
	  `compileEvidence()`, `compilerSignatures()`, or compatibility preflights see
	  them. `Program.moduleCompatibility(...)` treats that nested
	  `compilerSignatures` record as canonical, with flat-field fallback for older
	  evidence, and compares top-level input/output shape evidence before
	  signature checks so same-scalar-count rank changes remain readable
	  compatibility diagnostics. Program compile evidence is now signed evidence
	  too: `programCompileEvidenceSignature`, `isProgramCompileEvidence`,
	  `requireProgramCompileEvidence`, assertion aliases, and signature matchers
	  are exported through `zgml/program` and `zgml/compile`, so callers can
	  validate retained compile proof after package, log, or FFI handoff before
	  trusting it for compatibility or binding. Callers can therefore
	  compare compiler evidence without rebuilding private compatibility keys from
		  nested records, while mismatches still land at the intended IR, KernelPlan,
		  memory-layout, parameter-layout, or buffer-layout diagnostic stage with the
		  signature kind plus program/module signature values preserved as typed evidence.
		  Empty parameter layouts for parameterless Programs now remain canonical
		  empty-string signatures instead of being mistaken for missing signature
		  evidence during public-only derivation. Runtime
	  `Program` objects now also expose `bufferLayout()` from native requirements
  plus `trace()`, `compilerSignatures()`, `tensorProgramIr()`, `kernelPlan()`, `memoryLayout()`, `shapeConstraints()`,
  and `parameterLayout()` from retained module compiler evidence, giving
  tiny-linear, traced-module, and LLaMA-family hosts the same primary
  input/output/weights/bias slot contract before buffer creation or placement
  plus the frontend trace, normalized Tensor Program IR, module schedule, shape
  specialization, and named packed-parameter map where those exist.
  Compiled module `Program`
  objects retain that same public-safe compiler evidence through
  `compileEvidence()` and direct signature/trace/IR/kernel-plan helpers, while the single
  `nn.Linear` tiny-linear compatibility path now retains matching normalized IR
  and KernelPlan evidence, including flattened nested-`Sequential` parameter
  paths, without leaving the tiny-linear fast execution path. Module
  compatibility uses that evidence too, so a same-shape bare Linear cannot
  masquerade as a nested Linear with different stable parameter names.
  Shape-specialized Programs now also use their retained input-shape evidence
  as the default module-analysis shape for later `moduleCompatibility(...)`,
  `acceptsModule(...)`, and `bindModule(...)`, so batched module Programs and
  shaped Embedding Programs can bind their source modules without callers
  restating the compile shape.
  Parameterless unary
  elementwise chains such as
  `nn.neg() -> nn.exp() -> nn.log() -> nn.abs() -> nn.sqrt() -> nn.square() -> nn.sign() -> nn.step() -> nn.recip()`
  now prove that the kernel-plan seam is not limited to model-family matchers.
  The KernelPlan also fuses supported `Linear -> activation` pairs into one
  native linear descriptor with fused schedule and value-edge evidence, and the
  Program command planner carries the resulting dense `matmul + fused_elementwise`
  pair as one projection command, preserving the original IR ops while reducing common MLP
  command depth. Metal exact command execution now encodes dense elementwise and
  fused-elementwise sidecars as one dispatch when the sidecar fits the kernel
  envelope.
  `Program.inspect()` and
  `Program.capabilities()` now expose frozen runtime evidence records, including
  stable capability signatures, `matchesCapabilitySignature(...)`, plus matching
  frozen diagnostics for compile-only/resource-probe Programs and incomplete
  dispatch plans, with Node/Bun sharing one classifier derived from the existing
  native dispatch-plan inspection fields. `Program.requirements()`
  and `Program.modelCompatibility(...)` are frozen evidence records too,
  runtime-profile snapshots carry stable signatures plus
  `matchesRuntimeProfileSignature(...)` for cache/log comparison; `zgml/inspection`
  exposes standalone runtime-profile predicates/assertions/signature matchers
  for already-captured profile evidence, and
  model, buffer, Session, LLaMA Program, KV-cache requirement, runtime
  ABI/capability/layout, and supported-checkpoint catalog snapshots follow the
  same rule. The published TS Interface marks those evidence records read-only
  instead of mutable result bags. Eager Tensor shape arrays and `size()`
	  snapshots are frozen shape evidence too. Program `inputLen()` /
	  `outputLen()` and `inputByteLength()` / `outputByteLength()` return the
	  direct compiled scalar and byte edge lengths, `weightsLen()` /
	  `weightsByteLength()` and `biasLen()` / `biasByteLength()` return
	  persistent slot sizing, `parameterLen()` / `parameterByteLength()` return
	  aggregate persistent parameter sizing, `bufferSizing()` returns the same
	  primary sizing evidence as a signed frozen summary,
	  `matchesBufferSizingSignature(...)` compares cached sizing signatures,
	  `Program.requirements()`
	  carries matching scalar/byte evidence for input, output, weights, bias,
  and aggregate persistent parameters, Program
	  `inputShape()` / `outputShape()` arrays are frozen compiled-edge evidence,
	  and generic Sessions expose the same direct length helpers while
	  `outputShape()` can narrow to frozen bound-output Tensor evidence. The
	  public type smoke now proves Program, Session, and LLaMA-family sizing
	  helpers and `bufferSizing()` summaries are usable from normal TS imports
	  and remain readonly evidence records. It also pins everyday tensor readback
	  ergonomics such as elementwise comparison tensors, scalar `item()` / `valueOf()` /
	  `Symbol.toPrimitive`, nested-array `toArray()` / `tolist()`,
	  copied `numpy()` views, named `TensorJSON` serialization, and
	  shape-preserving `cpu()` / `float()` /
	  `float32()` / `to()` / `clone()` / `detach()` helpers at the declaration
	  boundary, while the shared package smoke proves JSON round-trip arrays do
	  not alias tensor storage. The same TS smoke
	  now pins module traversal, fluent `train()` / `eval()` /
	  `requiresGrad_(...)`, `zeroGrad`, and namespace state-dict helpers as part
	  of the ordinary PyTorch-like module surface, including concrete target
	  preservation for chainable module helpers. It also pins the direct
	  `optim.stateDict(...)` / `optim.loadStateDict(...)` / `optim.zeroGrad(...)`
	  / `optim.config(...)` / `optim.setLearningRate(...)`
	  and `train.zeroGrad(...)` / `train.backward(...)` / `train.step(...)` /
	  `train.lossStep(...)` / `train.clipGradNorm(...)` /
	  `train.clipGradValue(...)`
	  namespace helpers so small training loops are declaration-stable without
	  going through checkpoint wrappers. The root `loss` namespace now exposes
	  overloads that preserve the runtime distinction between raw host-data losses
	  (`number`) and differentiable Tensor losses (`Tensor`), with the public type
	  smoke pinning MSE, cross-entropy, and class-target helpers. It also covers the root
	  `nn.compilerSignatures(...)`, `nn.memoryLayout(...)`, `nn.bufferLayout(...)`,
	  `nn.inputShape(...)`, `nn.outputShape(...)`,
	  `nn.shapeConstraints(...)`, `nn.parameterLayout(...)`, and
	  `nn.canCompile(...)` compiler-inspection facade forms. Generic and
	  LLaMA-family Program/Session signature predicates for capabilities,
	  runtime profiles, Session call profiles, and StepParams contracts are also
	  pinned at the declaration boundary.
	  Program buffer resource-factory slots and LLaMA KV-cache resource slots
  are frozen edge evidence too, so placement callbacks cannot rewrite the
  inspected buffer contract they receive. Runtime profile snapshots are frozen
  read-only evidence even though reset calls mutate backend counters behind the
  Program/Session seam. Those diagnostics now have a closed
  discriminated JS/TS Interface, so `execution-unavailable` exposes
  execution-mode evidence and dispatch-plan diagnostics expose op-coverage
  evidence through type narrowing.
  JS/TS declarations now name the binding object
  `ModuleBindings` at the module seam and `ProgramBindings` at the Program seam.
	  `ModuleBindings` is branded read-only evidence produced by module helpers,
	  snapshots shape options, and is preserved through object spread, while
	  `ProgramBindings` remains the raw
  bind-object escape hatch and `TinyLinearWeights` stays a compatibility alias. The next deepening
  move is to keep pulling duplicated host
	  frontend code into shared internal Modules with thin runtime adapters, as the
		  ABI vocabulary and compile-envelope policy, eager Tensor operation policy,
		  eager `nn` module/namespace policy, runtime evidence decoding, Program buffer contracts, NativeBuffer edge
	  policy, Tensor placement and direct Program bind policy, trace compiler evidence,
	  Tensor Program IR construction/signature evidence, KernelPlan
  lowering/projection/signature policy, module Program descriptor layout,
  token-window semantics, and model-source routing have started to do.
  The next compiler deepening move is to align the
  now-normalized JS/TS `ModuleTensorProgramIr` vocabulary with the native
  `TensorProgramIr` / `Kernelizer` contract so the executable tensor Program
  compiler becomes one conceptual Interface across Zig and JS/TS.
- The BLAS matmul adapter now rejects stride layouts it cannot faithfully encode
  and falls back to the internal tiled kernel, preserving correctness for thin
  transposed gradients while keeping BLAS on clean dense cases.
- `Tensor.addBias` now uses direct broadcasted `add` instead of materializing a
  repeated bias buffer, shrinking dense layer graphs while preserving automatic
  broadcast-gradient reduction. `DeviceInference` lowers one-sided broadcasted
  binary ops into explicit `repeat` plus elementwise ops when building a
  `DeviceProgram`, so `nn.linear` stays a tiny PyTorch-like frontend expression
  while compiled classifier/token-head Programs still satisfy backend buffer
  bounds.

## Core Vocabulary

The public vocabulary should be boring and familiar:

- **Program**: a compiled, parameterized ML executable for a fixed architecture
  and shape envelope.
- **Session**: persistent bound runtime state for a Program, such as weights, KV
  cache buffers, optimizer buffers, and output buffers.
- **StepParams**: small per-call values such as token id, position, attention
  window, RNG seed, batch slot, or learning-rate scalar.
- **execute / step**: run the compiled Program for the current Session and
  StepParams.

The internal vocabulary can stay more mechanical:

- **KernelPlan**: the scheduled kernel/command list plus memory layout, buffer
  layout, and shape constraints produced by the Kernelizer.
- **ProgramStencil**: the static executable skeleton: schedule, memory layout,
  backend command shape, patch slots, and evidence hash.
- **RuntimeBindings**: backend-owned mutable bindings derived from a
  ProgramStencil and a Session.
- **RuntimeUpdate**: the small update applied before a step.
- **PatchTable**: the exact offsets, slots, or backend binding records updated
  by RuntimeUpdate.

Use "patch" internally when the engine writes small deltas into known slots.
Avoid making "patch and replay" the main public phrasing. Public users should
mostly see compile, bind, update, step, and execute.

## Mental Model

The closest mental model is a GPU pipeline or database prepared statement.

```text
GPU:
  shader + pipeline layout -> pipeline
  buffers/textures         -> bound resources
  uniforms/push constants  -> small per-dispatch updates
  dispatch                 -> execution

zgml:
  model architecture       -> Program
  weights/cache/buffers    -> Session
  token/window/seed/etc    -> StepParams
  step                     -> execution
```

Weights are dynamic. They are persistent Session bindings, not hot per-token
updates. A compatible Program should be able to bind different checkpoints,
adapter weights, KV caches, optimizer states, or external tensor buffers without
rebuilding the program shape.

## Architecture Shape

The ideal shape is:

```text
Tensor/model frontend
  -> Lazy Tensor IR
  -> Kernelizer
  -> KernelPlan
       scheduled kernels/commands
       memory layout
       buffer layout
       shape constraints
  -> Program construction
  -> ProgramStencil
       backend command shape
       patch table
       evidence hash
  -> Session binding
       weights
       KV cache
       optimizer state
       output buffers
  -> StepParams update
  -> execute/step
```

Cold work is allowed to be rich:

- shape inference
- graph lowering
- schedule construction
- memory planning
- kernel selection
- backend compilation
- evidence generation
- artifact verification

Hot work must be tiny:

- no graph walking
- no shape inference
- no backend decision-making
- no allocations
- no debug metadata scans
- no rebinding large buffers when a small slot update is enough

The hot path should reduce to:

```text
write StepParams into known slots
submit the cached backend command shape
return or expose output
```

## Stencil Is Not A Fake Backend

The current stencil-only path is useful, but the future design should avoid a
mental split between "stencil path" and "real path."

The stencil is not the architecture. The architecture is ordinary tensor code
lowered through Lazy Tensor IR, scheduled by the Kernelizer, and materialized as
a ProgramStencil.

The target is:

```text
one ProgramStencil
  -> inspect/hash for tests and benchmark evidence
  -> bind/execute for real CPU, Metal, wgpu, or Wasm work
  -> expose as an opaque handle through C/Node
```

Inspection is a view over the real executable program, not a parallel
implementation. If a stencil artifact passes but the real backend uses a
different shape, the architecture has failed.

## Public API Sketch

The root API should stay PyTorch-simple:

```ts
const y = x.mm(w).gelu();
await y.backward();
```

The common training surface should stay equally plain:

```ts
const model = zgml.nn.sequential([
  zgml.nn.linear(inDim, hiddenDim),
  zgml.nn.layerNorm([hiddenDim]),
  zgml.nn.gelu(),
  zgml.nn.linear(hiddenDim, outDim),
]);

const logits = model.forward(x);
const objective = zgml.loss.crossEntropy(logits, target);
zgml.train.step(zgml.optim.adamW(model, { lr: 3e-4 }), { loss: objective });
```

Users should only drop to explicit compilation when they want the runtime
contract:

```ts
const program = await model.compile({ backend: "webgpu", batch: 1 })
const session = await program.bind({ weights: model.parameters() })
const y = await session.execute({ input: x })
```

Zig can keep native helper modules for low-level tests, kernels, ABI examples,
and runtime substrate work, but those helpers are not the product frontend. If
an example starts to read like ordinary PyTorch-style user code, it belongs in
TypeScript and should reach native speed through `Program`/`Session`.

The LLM/program API should make compiled execution explicit:

```zig
const model = try zgml.llm.LlamaSession.load(allocator, .{
    .path = "model.gguf",
});
defer model.deinit();

const program = try model.compile(.{
    .context = 4096,
    .batch = 1,
});
defer program.deinit();

var session = try program.bind(.{
    .weights = model.weights(),
});
defer session.deinit();

const logits = try session.step(.{
    .token = token,
    .position = position,
    .attention_window = .{ .position = position, .len = 1 },
});
```

The exact names can change, but the layers should remain:

```text
compile Program
bind Session
update StepParams
execute step
```

## C And Node FFI Shape

The FFI boundary should expose opaque handles, not tensor internals.

```c
typedef struct zgml_model zgml_model;
typedef struct zgml_program zgml_program;
typedef struct zgml_session zgml_session;
typedef struct zgml_buffer zgml_buffer;

zgml_status zgml_model_load(const zgml_model_desc*, zgml_model** out);
zgml_status zgml_model_load_safetensors_data(const zgml_safetensors_data_load_desc*, zgml_model** out);
zgml_status zgml_program_compile(zgml_model*, const zgml_compile_desc*, zgml_program** out);
zgml_status zgml_program_get_requirements(zgml_program*, zgml_program_requirements* out);
zgml_status zgml_llama_program_get_kv_cache_requirements(zgml_program*, zgml_llama_kv_cache_requirements* out);
zgml_status zgml_program_create_buffer(zgml_program*, uint32_t kind, zgml_buffer** out);
zgml_status zgml_program_create_device_buffer(zgml_program*, uint32_t kind, uint32_t placement, zgml_buffer** out);
zgml_status zgml_program_get_device_handle(zgml_program*, uint32_t placement, uintptr_t* out);
zgml_status zgml_program_import_device_buffer(zgml_program*, uint32_t kind, const zgml_device_buffer_import_desc*, zgml_buffer** out);
zgml_status zgml_program_create_output_buffer(zgml_program*, zgml_buffer** out);
zgml_status zgml_session_bind(zgml_program*, const zgml_bind_desc*, zgml_session** out);
zgml_status zgml_session_bind_buffers(zgml_program*, const zgml_buffer_bind_desc*, zgml_session** out);
zgml_status zgml_session_step(zgml_session*, const zgml_step_desc*, zgml_step_result* out);
zgml_status zgml_session_step_no_output(zgml_session*, const zgml_step_desc*, zgml_step_result* out);
zgml_status zgml_session_argmax_token(zgml_session*, const zgml_token_argmax_desc*, zgml_token_argmax_result* out);
zgml_status zgml_session_execute_argmax_tokens(zgml_session*, const zgml_token_execute_argmax_desc*, zgml_token_argmax_result* out);
zgml_status zgml_session_generate_argmax_tokens(zgml_session*, const zgml_token_generate_argmax_desc*, zgml_token_generate_argmax_result* out);
zgml_status zgml_session_sample_token(zgml_session*, const zgml_token_sample_desc*, zgml_token_sample_result* out);
zgml_status zgml_session_execute_sample_tokens(zgml_session*, const zgml_token_execute_sample_desc*, zgml_token_sample_result* out);
zgml_status zgml_session_generate_sample_tokens(zgml_session*, const zgml_token_generate_sample_desc*, zgml_token_generate_sample_result* out);
zgml_status zgml_buffer_create(const zgml_buffer_desc*, zgml_buffer** out);
zgml_status zgml_buffer_write(zgml_buffer*, size_t byte_offset, const void*, size_t byte_len);
zgml_status zgml_buffer_read(zgml_buffer*, size_t byte_offset, void*, size_t byte_len);
void zgml_buffer_free(zgml_buffer*);
void zgml_model_free(zgml_model*);
void zgml_program_free(zgml_program*);
void zgml_session_free(zgml_session*);
```

The Node wrapper should feel like:

```ts
const model = await zgml.loadModel("model.gguf")
const program = await model.compile({ context: 4096, batch: 1 })
const logitsBuffer = program.createOutputBuffer()
const kvCache = program.createKvCache()
// Future GPU-resident output shape:
// const logitsBuffer = program.createOutputBuffer({
//   resource: ({ byteLength }) =>
//     zgml.NativeBuffer.externalResource({
//       placement: "webgpu",
//       handle: gpuBufferTable.lookup("logits"),
//       byteLength,
//       access: "write",
//     })
// })
// Future GPU-resident cache shape:
// const kvCache = program.createKvCache({
//   resource: ({ byteLength, kind, layer }) =>
//     zgml.NativeBuffer.externalResource({
//       placement: "webgpu",
//       handle: gpuBufferTable.lookup(`${kind}:${layer}`),
//       byteLength,
//       access: "readwrite",
//     })
// })
const session = await program.bind({ output: logitsBuffer, kvCache })
// Or let the Node/Bun wrapper allocate and own Program-sized native logits
// storage when all you need is native token selection/generation:
const simpleSession = await program.bind({ output: "native" })
const next = session.stepArgmax(token).token
const greedy = session.generateTokensArgmax(promptTokens, 128).tokens
const greedyOut = new Uint32Array(128)
session.generateTokensArgmaxInto(promptTokens, greedyOut)
const sampled = session.stepSample(token, { topK: 40, temperature: 0.8, seed }).token
const generated = session.generateTokensSample(promptTokens, 128, { topK: 40, temperature: 0.8, seed }).tokens
```

Node should not own graph execution details. It should call into native handles
that own compiled programs, sessions, buffers, and backend placement. JS typed
arrays are useful edge-copy views, not the representation of a long-lived
executable binding.

## Backend Targets

Each backend should consume the same ProgramStencil idea, but lower it into its
own efficient cached representation.

### CPU

```text
ProgramStencil -> function tape + memory arena + buffer pointer table
StepParams     -> small slot writes
execute        -> run cached function tape
```

### Metal

```text
ProgramStencil -> cached pipelines, buffer bindings, dispatch sequence
StepParams     -> patch small buffers or binding slots
execute        -> submit cached command shape with minimal encoding overhead
```

Metal is the first serious local performance lane. It should stay benchmarked
against llama.cpp/ggml on Apple Silicon.

### wgpu / WebGPU

wgpu command buffers are not the right reusable unit. The reusable unit is:

```text
cached pipelines
cached bind group layouts
cached bind groups and buffers where legal
cached dispatch sequence
small patch/update buffer
fast command encoding per step
```

This can be elegant and performant if dispatch count stays low, buffers stay GPU
resident, and StepParams updates avoid CPU/GPU readbacks.

The implementation contract for the browser lane lives in
`docs/wgpu-wasm-runtime.md`. It mirrors the C/Node/Bun handle lifecycle while
spelling out the WebGPU-specific cached pipeline, bind-group, patch-table, and
evidence requirements.

### Wasm

Wasm should use compact tables and linear-memory offsets:

```text
ProgramStencil handle -> wasm memory object/id
Session buffers       -> wasm memory offsets
StepParams            -> writes into linear memory
execute               -> one wasm call where possible
```

Wasm is the portable/browser lane. Top-tier browser performance likely needs a
WebGPU backend, with Wasm as the host/runtime glue and CPU fallback.

## Implementation Plan

### Phase 1: Name The Core Object

Create a first-class internal `ProgramStencil` or `CompiledProgram` type that
represents the reusable executable skeleton.

Acceptance:

- `DeviceProgram` and `RuntimeBindings` terminology either maps clearly onto the
  new type or is renamed in small, mechanically safe steps.
- Existing stencil hashes are emitted from the same object that real execution
  consumes.
- Stencil-only tests become inspection tests over a real program shape.
- No public API bloat.

Current slice:

- `ProgramStencil.inspect()` reports the copied op count, buffer count, total
  buffer elements, runtime patch shape, bounded StepParams envelope, and
  command-stream shape derived from the same executable op tape.
- CPU, Metal, and stencil backend runtime profiles retain the command-stream
  shape as cold compile evidence without adding hot-path work.
- Backends expose a persistent-binding upload primitive so `Session` state can
  be copied into a compiled program without executing a step.
- `DeviceInference` now exposes explicit `Program.compile(...).bind(...)` types
  while preserving the old one-call convenience wrapper.
- Persistent tensors participate in the compiled buffer shape but are excluded
  from compile-time initial uploads; `Session` binding uploads them once, and
  hot execution stays to StepParams plus per-step I/O.
- `DeviceInference.Program.inspect()` exposes IR shape, runtime patch, and
  command-stream shape evidence captured at backend compile time plus the
  graph-lowered runtime patch envelope, giving callers one compiled-program
  inspection surface instead of a separate stencil-only helper. That cold
  inspection snapshot no longer depends on mutable backend runtime-profile
  counters.

Frontend slice:

- Persistent module-owned parameters now have durable gradient buffers that are
  restored after graph teardown, which is the ownership shape required for a
  PyTorch-like TS package frontend over Zig native runtime bindings.

### Phase 2: Split Cold Evidence From Hot Execution

Separate rich inspection metadata from hot runtime state.

Acceptance:

- Runtime step does not walk evidence/debug structures.
- Runtime update touches only a bounded patch/update table.
- Benchmark artifacts still include schedule shape, fallback placement, dispatch
  counts, runtime update counts, and stencil hashes.
- `zig build check` remains the local gate. On macOS, `-Duse-metal=false`
  provides an explicit CPU/Wasm validation lane when the local Metal driver is
  unavailable or wedged; it must skip Metal execution rather than silently
  claiming Metal evidence.

Current slice:

- `DeviceInference.Session.executeStep` patches the bound backend runtime handle
  and executes already-bound per-step I/O; it does not allocate, rebuild I/O
  lists, or inspect graph/evidence metadata.
- A LLaMA executable decode regression test binds a CPU Program/Session, then
  switches the stored allocator to fail on the next allocation while running
  `stepInto`, `advance`, and another `stepInto`. This proves token input
  patching, runtime patch-table updates, caller-owned logits, no-output advance,
  and CPU executable dispatch stay allocation-free after bind.
- A matching LLaMA executable prefill regression binds a fixed-window CPU
  Program/Session, then forces later allocations to fail while running
  `prefillInto` and no-output `advanceTokens`. This proves prompt-window
  StepParams, caller-owned logits, KV-cache output synchronization, and
  executable prefill dispatch stay allocation-free after bind.
- A model-agnostic `DeviceInference` CPU regression now compiles and binds a
  dynamic slice-assign program with an injected failing allocator, then proves
  `RuntimeWindow` patching, CPU executable dispatch, normal output download,
  and no-output advancement all stay allocation-free after binding.
- `DeviceInference.Session` now exposes reset/add runtime-profile hooks through
  bound backend runtime handles. CPU, stencil, and Metal backends keep those
  counters on `RuntimeBindings`, so two Sessions sharing one Program can report
  independent patch/execute windows while Program inspection remains cold shape
  evidence.
- `DeviceInference.Session.inspect()` now reports the selected backend,
  bound output storage kind, persistent/step I/O counts, and host-vs-resource
  binding counts from the actual bound runtime state. LLaMA, C, Node, Bun, and
  Wasm inspection surfaces translate this native Session contract instead of
  re-deriving binding shape in adapters.
- `DeviceInference.Session` now owns its persistent, input, and output
  `ProgramIO` tables through a cold `SessionBindingLayout` module. That layout
  owns validation, backend configure calls, output-selection views, binding
  counts, and binding-shape hashing, so Session hot execution can keep using
  preconfigured backend state while FFI and LLaMA callers still get the same
  mutable binding slots for deliberate dynamic output rebinding.
- `DeviceInference.Program.inspect()` now exposes a cold pre-bind binding
  requirement manifest: a stable hash plus persistent, per-step input, and
  per-step output counts. The C ABI, Node, Bun, and browser/Wasm resource
  Program surfaces report the same fields, so hosts can inspect the
  model/program contract before creating a Session. This is distinct from
  `Session.inspect().binding_shape_hash`, which proves the actual bound
  host/resource slots after binding.
- `DeviceInference` now rejects undersized per-step input/output bindings before
  backend configuration. Persistent Session state can still bind deliberate
  Program-compatible spans such as context-sized KV caches, but step I/O must
  cover the exact tensor extent so hot execution cannot silently truncate caller
  inputs or outputs.
- The C ABI plus Node, Bun, Node/WASI, and browser Wasm smoke paths now expose
  or exercise Program and Session runtime-profile snapshot/reset calls, so FFI
  hosts can distinguish cold Program evidence from bound Session patch/execute
  counters without peeking into backend internals.
- The C ABI plus Node, Bun, Node/WASI, and browser Wasm smoke paths now expose
  model-handle inspection before compile. Hosts can read the model kind and
  shape envelope for tiny linear and LLaMA-family handles, including LLaMA
  vocab/context/width/head/feed-forward and compatibility scalars, before
  choosing a compile envelope or binding a compatible model.
- The same handle lanes now expose Program/model compatibility preflight before
  dynamic Session binding. `zgml_program_check_model_compatibility` and
  JS `program.acceptsModel(model)` report whether a Program can bind a model
  handle without allocating or cloning runtime state, so callers do not need to
  use bind failure as their compatibility query.

### Phase 3: Make LLaMA Use Program -> Session -> Step

Move LLaMA inference toward the explicit object lifecycle:

```text
compile LLaMA Program
bind weights and KV cache as Session state
step with token/window params
```

Acceptance:

- Different compatible weight bindings can share the same program shape where
  practical.
- RuntimeWindow is expressed as StepParams or an internal equivalent.
- KV-cache write position and attention length remain derived from the runtime
  update, not rescanned by callers.
- Existing SmolLM stencil evidence remains stable or changes with an explained
  baseline update.

Current first slice:

- `zgml.llm` now exposes `LlamaModel`, `LlamaProgram`, and `LlamaSession` type
  factories alongside the compatibility `LlamaSession.init/load/step` path.
- The public LLM facade supports `model.compile(.{}).bind(.{}).step(.{ .token
  = ... })` without exposing graph, loader, backend, or quantization internals.
- `LlamaCompileOptions`, `LlamaBindOptions`, `LlamaStepParams`,
  `LlamaExecuteParams`, and `LlamaExecuteIntoParams` establish the public
  vocabulary for context/batch, explicit session binding, single-token params,
  token-window/output-policy execution params, and caller-owned output-buffer
  execution params. `LlamaStepParams` is deliberately scalar-token-only; token
  windows live in execute/prefill params. `LlamaBindOptions` deliberately has no reset switch today:
  each bind creates a fresh independent runtime Session, and replay/reset lives
  on the Session itself.
- `LlamaCompileOptions.backend` now selects `.auto`, `.cpu`, `.metal`, or
  `.webgpu`. `.auto` is the stable CPU executable path today; explicit `.metal`
  owns a Metal backend instance inside the compiled Program when the build
  links the Metal backend. Native builds with `-Duse-wgpu=true` now let explicit
  `.webgpu` LLaMA-family Programs execute through the same public
  Program/Session facade against the real wgpu executor; builds without native
  wgpu, portable Wasm/browser builds, or explicit
  `-Dexperimental-llama-wgpu-execution=false` builds keep `.webgpu` on the
  compile-only/resource-probe lane.
- `LlamaCompileOptions.context_len` is now enforced by executable sessions:
  `execute`, `step`, `advance`, and `prefill` fail before mutating state when
  they would exceed the compiled Program context envelope. The C ABI regression
  covers the generic `zgml_session_execute_tokens` path directly, including a
  cleared result struct and untouched caller logits buffer on rejection. `batch`
  is kept honest at `1` until the LLaMA executable path supports true batched
  decode.
- `LlamaModel.compile(...)` now eagerly builds the persistent decode
  `DeviceInference.Program` for the selected backend. Public
  `LlamaProgram.bind(...)` uses executable Program/Session bindings rather than
  the host inference fallback, so Zig, C, and Bun all exercise the same
  compiled-program path.
- For executable backends, `LlamaProgram.inspectExecutable()` now reads command
  and runtime-patch evidence from that selected decode `DeviceInference.Program`
  rather than from a separate stencil-only compile. The regression compares the
  public inspection surface to the compiled Program's cold runtime-profile
  shape. WebGPU remains the explicit compile-only inspection lane until it has a
  real execution backend. Native Zig inspection now also carries
  `execution_mode: .executable | .resource_probe | .compile_only`, derived from
  the same execution and external-resource evidence as the JS/Wasm capability
  helpers, so embedders do not have to duplicate that boolean interpretation.
- `LlamaProgram.bind()` now forks an independent runtime session from the model:
  persistent f32 params are copied, direct quantized GGUF weights and param maps
  are deep-cloned, and each bound session owns separate KV caches, position,
  plans, and runtime evidence. CPU executable bindings now upload compatible
  quantized qweight payloads into runtime state, so quantized decode can share a
  compiled Program shape across independent bound sessions.
- Bound LLaMA sessions expose `stepInto` and `prefillInto`, so embedders can
  reuse caller-owned logits buffers while the simple slice-returning
  `step`/`prefill` API remains available. Those per-call caller-owned output
  paths now replace the full backend `ProgramIO` output binding for the call,
  rather than only rewriting a host pointer field, so future resource-bound
  Session outputs can still be temporarily overridden by an explicit host
  logits buffer without leaving stale resource metadata attached.
- LLaMA executable Program binding now has an internal output-resource option
  for resource-capable backends. A resource-bound default output deliberately
  rejects slice-returning `step`/`prefill` calls, because those APIs promise
  fresh host logits; explicit `stepInto`/`prefillInto` calls can still override
  the resource binding with a caller-owned host buffer for one call, and
  no-output advancement can keep using the resource-bound Session without a
  host readback.
- LLaMA executable binding now exposes KV-cache host-buffer and external-resource
  descriptors through the public C, Node, Bun, Node/WASI, and browser Wasm handle
  lanes. The JS shape is `kvCache: { k: NativeBuffer[], v: NativeBuffer[] }`,
  validated from the compiled Program before crossing FFI. KV byte lengths are
  derived from the compiled context envelope rather than the model maximum
  context. Host `zgml_buffer` KV caches execute on CPU/Wasm today; opaque
  external-resource views still reject on host-only backends while preserving the
  future resource-capable wgpu/WebGPU bind shape.
- Bound LLaMA sessions also accept token-window execution params through
  `execute(.{ .tokens = ..., .output = .logits | .none })` and
  `executeInto(...)`, so native Zig and the C/JS/Wasm handle lanes now share the
  same StepParams-shaped mental model while retaining the older `step`,
  `prefill`, and `advanceTokens` conveniences.
- The public executable token-window path now uses one fixed prefill envelope
  per Program: `min(default_prefill_chunk, context_len)` is compiled during
  `LlamaModel.compile(...)` and bound during `LlamaProgram.bind(...)`. Full
  token chunks execute through that prefill Program, short windows and tails
  execute through the already-bound decode Program, and a failing-allocator
  regression proves first full-window prefill, shorter logits execution, and
  no-output continuation do not allocate after Session bind. `Session.reset()`
  now preserves those bound executable sessions, rewinds model/KV state, and
  refreshes runtime cache bindings so reset/replay stays allocation-free too.
- Bound LLaMA sessions expose native Zig greedy helpers too:
  `stepArgmax(...)` and `generateArgmaxInto(output_tokens, prompt_tokens)`.
  They run through the same Program/Session execution path, keep logits inside
  the session, and fill caller-owned token slices for hot generation loops.
- Native Zig sessions also expose bounded top-k sampling through
  `stepSample(...)` and
  `generateSampleInto(output_tokens, prompt_tokens, options)`. Sampling
  validates `top_k`, temperature, and context capacity before mutating Session
  state, uses a fixed stack top-k buffer, and treats `seed` as a base seed for
  generated token offsets. Native Zig and the C ABI now share the same bounded
  sampler implementation, so FFI token selection cannot drift from the
  Zig-native convenience path.
- The native Zig conveniences now route non-empty token windows through that
  same internal executor, preserving `prefill(&.{})` compatibility while making
  `execute` the real implementation center for decode, prefill, caller-owned
  logits, and no-output advancement.
- `LlamaProgram.bindExecutableDecode()` binds the public LLM facade to the
  persistent decode `DeviceInference.Program` path. The fixed tiny LLaMA C ABI
  uses this executable decode binding for `zgml_session_step_token`.
- LLaMA programs expose `inspect()` with architecture, compile-envelope, and
  semantic runtime-patch shape evidence, giving embedders a pre-bind program
  description without exposing graph internals.
- `LlamaInferenceSession.compileDeviceDecodeProgram(...)` now builds a
  persistent decode `DeviceInference.Program` that can bind independent runtime
  sessions. The regression test proves two executable sessions can share one
  compiled decode program, keep independent positions/runtime state, and match
  host decode logits.
- For f32 LLaMA decode, model parameters are now Program-shaped but
  Session-bound: compile includes their buffers in the executable shape while
  skipping initial uploads, and bind maps those compile-time tensor slots to the
  runtime session's parameter storage. The regression test also proves a
  modified runtime `out_proj` binding changes executable logits without
  recompiling the program.
- LLaMA prefill now has the same internal persistent executable lifecycle:
  `compileDevicePrefillProgram(chunk)` builds a reusable fixed-window Program,
  `bind` attaches a runtime session's matching prefill plan, weights, KV cache,
  and output buffer, and `prefill`/`prefillInto` run prompt windows through
  StepParams. The regression test proves independent prefill sessions,
  caller-owned logits buffers, and modified f32 runtime weight bindings behind
  one compiled prefill Program.
- LLaMA executable prefill now treats KV cache as Session state rather than a
  private prefill-only backend copy: cache tensors are bound as persistent
  session resources, prefill executions download their cache side effects into
  the Session, and the already-bound decode runtime refreshes only the
  persistent KV-cache binding range after a position-zero prefill. The low-level
  regression proves the decode runtime handle and bound weights are preserved
  while its cache view is refreshed; the C/Node/Bun/Wasm smokes prove bulk
  no-output prompt advancement followed by decode matches one executable prefill
  over the same token window.
- LLaMA executable Program binding now also accepts caller-owned host K/V cache
  buffers in addition to KV-cache resource options for resource-capable backends.
  The regression proves host K/V buffers replace internal persistent cache
  bindings and execute through CPU/Wasm, while external K/V resources replace
  host persistent cache bindings for resource-capable backends, prefill cache
  side-effect outputs can target those same resources, and mismatched resource
  placement rejects during bind. The same shape is public through
  C/Node/Bun/Node-WASI/browser Wasm LLaMA-specific bind descriptors, while
  current host-only backends still reject external resource views as unsupported.
- The SmolLM benchmark device-decode and stencil-decode rows now use that
  persistent decode program path, so benchmark evidence is tied to a reusable
  Program object instead of a one-shot device wrapper.
- The SmolLM benchmark device-prefill and stencil-prefill rows now use the
  persistent prefill Program path too, so prompt evidence is also tied to a
  reusable Program object instead of the one-shot device wrapper.
- This is still copy-on-bind for public LLaMA session binding. CPU and Metal
  direct quantized qweights are now Program-shaped and Session-bound for
  executable decode, and the C/JS ABI has a first compiled-in
  checkpoint-compatible selector for GGUF and safetensors. The internal backend
  `ProgramIO` descriptor and C/JS/Wasm `zgml_buffer` handle can now distinguish
  host-memory bindings from opaque external-resource views. Public KV buffer
  requirements are now executable-layout sized rather than wrapper-math sized:
  single- and multi-KV-head programs use the packed context span
  `[d_head, context_len * n_kv_heads]`. Host KV buffers can be caller-owned
  Session state on CPU/Wasm. The core `LlamaBlock.forwardCachedMasked` derives
  its per-head KV slab stride from the cache tensor shape instead of
  `max_seq_len`, and the higher-level LLaMA plan/session allocation now threads
  context-sized caches through Program compile and Session bind. Public
  multi-KV-head requirements are therefore packed-context sized.
  External-resource KV remains future-backend work, and zero-copy adapters plus
  arbitrary dynamic model-family selection remain future work.

### Phase 4: C ABI Handles

Add an intentionally tiny C ABI around model/program/session/buffer handles.

Acceptance:

- The ABI exposes no Zig allocator details, no graph internals, and no backend
  structs.
- Handles have explicit ownership and error reporting.
- A C smoke test can load or construct a tiny model, compile, bind, step, and
  free handles.
- The ABI is optional and does not distort the Zig-native API.

Current first slice:

- `src/c_api.zig` exposes opaque `zgml_model`, `zgml_program`, and
  `zgml_session` handles.
- `include/zgml.h` exposes the C declarations for non-Zig consumers.
- `zgml_get_runtime_info` gives embedders an ABI handshake before they pass
  descriptor structs across the FFI boundary. It reports `ZGML_ABI_VERSION`,
  native `size_t` and pointer widths, compact `uint32_t` token ids, and feature
  bits for buffer handles, compatible-checkpoint selection, runtime profiles,
  WebGPU compile-only inspection, Wasm exports, native buffer I/O, native
  greedy argmax over LLaMA-family logits, one-call execute-and-argmax, native
  generate-and-argmax, bounded top-k sampling, one-call execute-and-sample,
  Program requirements/output buffers, native generate-and-sample,
  program-owned WebGPU device-buffer factories, same-device device-buffer import
  handles, compatible model-handle Session binding, non-owning external
  host-buffer wrapping, and opaque external-resource buffer views with
  read/write access flags,
  model-handle inspection, Program/model compatibility preflight, and
  host/external-resource buffer inspection, metadata-only checkpoint path probe,
  and compiled-in supported checkpoint catalog inspection.
  The ABI also exposes pathless safetensors-header probing, full safetensors
  byte probing, and full safetensors byte loading, so browser/Wasm hosts can
  preflight an exact checkpoint envelope from metadata bytes or a complete
  owned checkpoint blob, then create a model handle from that same blob, before
  they have a filesystem path.
  `zgml_abi_struct_size(kind)` now
  gives C/Node/Bun/Wasm hosts target-native `sizeof` values for public
  descriptor, inspection, requirements, and profile structs, with Node/Bun
  wrapper helpers and Node/WASI/browser Wasm smokes deriving descriptor
  allocation sizes from that exported layout.
- The C ABI now also exposes an opaque `zgml_buffer` handle with
  create/size/read/write/free calls, non-owning caller-memory wrapping through
  `zgml_buffer_wrap`, opaque resource views through
  `zgml_buffer_wrap_resource`, storage/resource metadata through
  `zgml_buffer_inspect`, and direct Session binding through
  `zgml_session_bind_buffers`. Host buffers support host data/read/write.
  Resource views are size-queryable but host data/read/write and current
  CPU/Metal/stencil execution return unsupported. Buffer inspection exposes
  host vs external-resource storage, placement, access flags, caller handle,
  byte offset, view length, and backing resource length. Resource-capable
  backends validate that persistent/input resources are readable and output
  resources are writable before upload/execute. This gives C, Node, Bun, and
  Wasm-style embedders explicit buffer handles without exposing tensor internals
  or backend structs, while keeping the WebGPU/wgpu resource-binding seam honest
  before real GPU execution exists.
- The C ABI now also exposes program-owned WebGPU device-buffer creation, raw
  same-device handle discovery, and imported device-buffer views through
  `zgml_program_create_device_buffer`, `zgml_program_get_device_handle`, and
  `zgml_program_import_device_buffer`. Node and Bun expose the same narrow raw
  handle lane as `program.deviceHandle("webgpu")` and
  `program.importDeviceBuffer(kind, { deviceHandle, bufferHandle })`, with an
  optional `byteLength` when the host wants exact-view validation instead of the
  compiled Program's required size. Node and Bun now also expose
  `program.device("webgpu")`, a small same-device provenance object whose
  `handle` matches the Program's raw device token and whose
  `createBuffer(kind)` / `importBuffer(kind, source)` methods route through the
  same C ABI. For zgml-created device buffers, Node and Bun accept imports
  through either object and derive the device token plus raw buffer handle from
  the inspected buffer object for weights, bias, input, and output roles.
  Smokes prove zgml-created same-device buffers can be imported back through
  that public path, bound as Session resources, and executed even after the
  original source buffer handles are freed. They also prove wrapper-level fake
  external-resource object rejection plus wrong-device, unaligned, and
  oversized raw imported views before Session binding; browser `GPUBuffer`
  interop remains a separate shared-device host contract.
- Sessions now expose `zgml_session_inspect`, surfaced as `session.inspect()`
  in Node and Bun, so embedders can verify model kind, backend, position,
  context length, output/KV storage kind, and host-vs-resource binding counts
  for the actual bound runtime state. Claimed Wasm/WebGPU host Sessions mirror
  their host-owned token position through this inspection path as well as
  `zgml_session_position`. This keeps Session binding evidence next to Program
  and Buffer inspection instead of relying on wrapper-side guesses.
- `zgml_session_argmax_token` is the first native post-logits helper. It can
  read caller-provided logits or the Session-bound logits buffer and return the
  greedy token plus logit without copying the full vocabulary vector back across
  C, Node, Bun, or Wasm.
- `zgml_session_execute_argmax_tokens` combines token-window execution with
  native greedy selection. It requires a bound logits buffer, writes logits into
  that Session-owned binding, and returns only the selected token/logit. Node
  and Bun expose that as `session.stepArgmax(token)` /
  `session.executeTokensArgmax(tokens)`.
- `zgml_session_generate_argmax_tokens` is the native greedy generation loop.
  It consumes caller-owned `uint32_t` token IDs, writes generated token IDs into
  a caller-owned output buffer, and keeps logits inside the bound Session
  output buffer.
- `zig build ffi-c` builds the dynamic `zgml_c` library.
- `zig build ffi-c-smoke` compiles and runs a C executable against the ABI.
  The smoke now also compile-touches the current header surface: model kinds,
  backend ids, execute policies, runtime feature flags, Wasm helpers,
  requirements/output-buffer/profile calls, unified token execution, native
  argmax/sampling/generation helpers, session reset/position, and buffer-data
  access must all be declared in `include/zgml.h` and link from the dynamic
  library.
- `zig build check` runs the C ABI smoke tests.
- The first smoke lane is tiny linear inference, which builds a `ComputeGraph`
  and compiles it through `DeviceInference` on the CPU backend during
  `zgml_program_compile`.
- The C ABI also exposes a fixed tiny LLaMA smoke lane:
  `ZGML_MODEL_TINY_LLAMA -> zgml_program_compile -> zgml_session_bind ->
  zgml_session_step_token`. It proves token stepping, prompt prefill,
  caller-owned logits buffers, independent executable decode/prefill session
  positions, and pre-bind semantic program inspection plus decode executable
  command-shape inspection across C handles, but it is not yet the GGUF/SmolLM
  executable-stencil ABI.
- The C ABI now has a first real-checkpoint LLaMA lane:
  `ZGML_MODEL_SMOLLM_135M -> zgml_model_load_path -> zgml_program_compile ->
  zgml_llama_program_inspect -> zgml_program_inspect -> zgml_session_bind ->
  zgml_session_step_token`.
  This loads a local SmolLM-135M GGUF or safetensors file into the fixed config
  used by the benchmark. `zgml_program_compile` owns a persistent selected-backend
  decode `DeviceInference.Program` for executable token stepping, independent
  KV-cache state, lifecycle, and evidence. The C ABI also exposes
  `zgml_session_execute_tokens`, a token-window execution primitive with an
  explicit output policy for final-token logits or no-output prompt
  advancement. It is the StepParams-shaped C entry point used by the JS and
  Wasm lanes, and the older convenience calls now route through that same
  handler:
  `zgml_session_advance_token`, a no-output token update for prompt tokens whose
  logits do not need to cross the FFI boundary,
  `zgml_session_advance_tokens`, a bulk no-output prompt advancement call that
  keeps prefill cache side effects visible to later decode, and
  `zgml_session_prefill_tokens`, an executable prompt-prefill call that writes
  final-token logits into a caller-owned output buffer. It is not yet the final
  compatible-checkpoint executable-stencil ABI.
- `ZGML_MODEL_AUTO` is the first compatible-checkpoint selector for FFI callers:
  `zgml_model_load_path` inspects GGUF metadata or safetensors tensor-shape
  metadata, accepts configs that match a compiled-in executable family, and
  currently maps exact fixed tiny LLaMA and SmolLM-135M GGUF/safetensors
  checkpoints onto their existing compiled lanes. Unsupported shapes fail before
  loading.
  `zgml_supported_checkpoint_count` and `zgml_supported_checkpoint_inspect`
  expose that same compiled-in family registry as `zgml_model_inspection`
  records, so hosts can list the checkpoint envelopes this artifact can load
  before they probe a file path.
  `zgml_model_probe_safetensors_header` now runs the same registry match over a
  safetensors JSON header slice, closing the path-only probe gap for browser and
  Wasm hosts while keeping unsupported shapes rejected before loading.
  `zgml_model_load_safetensors_data` is the matching pathless load lane for
  hosts that already own checkpoint bytes. It selects the exact compiled-in
  LLaMA-family envelope from the safetensors header, loads the tensor payload
  into a real model handle, and preserves the same `ZGML_UNSUPPORTED` behavior
  for shapes outside this artifact's supported checkpoint catalog. C ABI tests
  now also prove the full-data probe descriptor selects the SmolLM-135M envelope
  from host-owned safetensors header bytes without allocating or loading a dummy
  full checkpoint payload.
- `zgml_compile_desc` now exposes backend, context length, and batch envelope
  fields. Unknown backend IDs or invalid context lengths fail with
  `ZGML_INVALID_ARGUMENT`; unsupported model/backend or batch combinations fail
  with `ZGML_UNSUPPORTED`.
- On macOS the C ABI dynamic library links the Metal shim/frameworks, so
  Node/Bun callers can request Metal through the same opaque
  `zgml_program_compile` handle path. The default remains CPU-backed `AUTO`
  until benchmark evidence justifies changing automatic placement.
- For f32 LLaMA executable decode, session binding uploads model parameters
  through the same persistent-binding path as tiny linear. On CPU and Metal,
  direct quantized GGUF SmolLM qweight payloads are now uploaded into session
  runtime state at bind time after compile-time shape validation, so compatible
  quantized sessions can share one compiled executable Program.
- LLaMA prefill also has a persistent Program/Session path now, the
  benchmark/stencil evidence uses it, and C/Node/Bun/Wasm callers can reach it
  through `zgml_session_prefill_tokens` / `session.prefill(tokens, output)`.
- LLaMA prompt advancement also has a bulk no-output FFI path through
  `zgml_session_advance_tokens` / `session.advanceTokens(tokens)`. Position-zero
  bulk advancement uses the executable prefill Program, synchronizes Session KV
  cache side effects, and refreshes only the existing decode runtime's
  persistent KV-cache binding range before the next token step.
- `zgml_program_compile` now owns a real `DeviceInference.Program`; persistent
  weight/bias tensors are excluded from compile-time initial uploads.
- LLaMA-family token windows use compact `uint32_t` token ids at the C ABI
  boundary. Native Zig converts them to internal `usize` token slices on the
  stack before dispatch, so Node, Bun, and Wasm callers do not need BigInt
  token arrays or pointer-sized token buffers.
- `zgml_session_bind` creates a `DeviceInference.Session` and uploads weight and
  bias buffers through the persistent binding path without executing the
  program. For tiny linear programs, `zgml_bind_desc` can now also bind
  caller-owned input/output buffers as Session step I/O, so
  `zgml_session_step(session, NULL, &result)` executes against already-bound
  host buffers. The per-step descriptor path remains available for callers that
  want to provide input/output buffers per call. `zgml_session_upload_persistent`
  and `zgml_session_upload_persistent_range` expose the same persistent-binding
  refresh point through C, Node, and Bun for tiny-linear, tiny-MLP, and traced
  module Programs, so buffer-backed weights can change behind an already-bound
  Session without rebuilding it. `zgml_session_reset` resets reusable Session state; it is a no-op for tiny linear and rewinds LLaMA-family
  position/KV state while preserving bound executable decode/prefill sessions
  when their cache bindings refresh cleanly. For LLaMA-family programs,
  `zgml_bind_desc.output` can now bind a caller-owned logits buffer once,
  allowing token step and prompt prefill calls to omit per-call output pointers
  while still supporting explicit per-call output buffers. LLaMA-family FFI
  callers can also use `zgml_session_bind_model` or
  `zgml_session_bind_model_buffers` to bind a second compatible model handle as
  the Session's persistent state behind the same compiled Program shape. Node
  and Bun expose that as `program.bind({ model, output })`.
- `DeviceInference.Session` now executes through the backend's configured
  Session I/O table by default. Backends receive the full persistent/input/output
  binding table at bind/reconfigure time and can cache host descriptors,
  resource tables, or future bind groups outside the hot token loop. Intentional
  per-call descriptor changes use an explicit dynamic-I/O step path; LLaMA
  `stepInto`/`prefillInto` and prefill cache side-effect-only advancement use
  that path, while normal bound token/prefill execution uses the configured
  table. LLaMA prefill reconfigures the backend table after installing its
  logits-plus-KV-cache output side-effect descriptors, so decode cache refresh
  still observes prefilled KV state without rebuilding the session.
- `zgml_program_inspect` exposes compact executable shape evidence for FFI
  hosts before session binding. For executable LLaMA-family backends,
  `zgml_program_compile` eagerly compiles the selected decode
  `DeviceInference.Program` and `zgml_program_inspect` reads that Program's cold
  command/runtime-patch shape without executing a token. Builds without native
  wgpu, portable Wasm/browser builds, and explicit LLaMA WebGPU opt-out builds
  remain compile-only/resource-probe and use WebGPU-labeled stencil evidence;
  native `-Duse-wgpu=true` builds now execute supported LLaMA-family WebGPU
  Programs through the real wgpu executor. The inspection also reports the selected backend,
  whether execution is supported for that backend, whether external-resource
  bindings are supported before bind, planned op count, buffer count, total
  buffer elements, byte length, initial upload count, qweight count, and stable
  command-category counts for op, row, projection, attention, movement,
  elementwise, and RoPE command families.
- `zgml_program_get_requirements` exposes cold Program buffer and envelope
  metadata: scalar width, token-id width, input/output/weight/bias/logits
  lengths, output byte length, context length, batch, and maximum token-window
  envelope. C, Node, Bun, Node/WASI, and browser Wasm smokes use it to allocate
  or verify edge buffers from the Program handle instead of duplicating
  model-family constants in callers.
- `zgml_llama_program_get_kv_cache_requirements` exposes LLaMA-family per-layer
  K/V cache resource sizes from the compiled Program handle. C, Node, Bun,
  Node/WASI, and browser Wasm smokes use it to size executable-layout cache
  buffers without duplicating head/context/stride math in host wrappers.
- Node/WASI and browser Wasm now share a host token-window bridge for
  WebGPU-labeled LLaMA resource Sessions. The exported
  `zgml_session_execute_tokens` path and its token step, prefill, no-output
  advance, execute-and-select, and generation conveniences call a `zgml_host`
  import when the Session belongs to the host runtime. That runtime
  distinguishes "unclaimed" from public `unsupported`, so normal Wasm fallback
  remains possible while claimed resource Sessions stop at the WebGPU host
  boundary. The default LLaMA bridge deliberately returns `unsupported` without
  mutating native position, caller logits, output-token buffers, or runtime
  counters. The shared host wrapper now also has an explicit token-executor hook
  at that same boundary: after token-window validation, the opt-in executor
  receives the decoded descriptor, explicit `stepParams` window evidence, and
  bound resource table and can return `ok`, write logits, and advance the
  host-owned position reported by `zgml_session_position` and
  `zgml_session_inspect`. `stepParams` names the start/end position, token
  count, context length, output policy, output kind, requested output length,
  and logits length that future kernels patch into their command stream. The
  reusable `WasmWebGpuLlamaTokenWindowExecutor` helper provides `writeLogits`,
  `llamaKvCacheWindow`, and `writeKvCacheWindow` utilities to executor
  callbacks while reusing its augmented callback context and K/V window slot
  scratch across synchronous calls, so browser kernels share one logits/KV
  window contract instead of duplicating descriptor math in adapters. The named
  `WasmWebGpuLlamaTokenWindowPatternExecutor` builds on that contract as a
  deterministic proof executor: caller-owned logits still write through Wasm
  linear memory, while bound logits and K/V cache windows dispatch real WebGPU
  compute against `GPUBuffer` resources when a browser adapter exists, with a
  mock-resource fallback for Node/WASI and headless browsers. The GPU path
  binds whole storage buffers and patches the element offset in a tiny params
  buffer, so nonzero K/V cache windows do not violate WebGPU storage-buffer
  alignment rules. Node/WASI and browser smokes prove caller-output, no-output,
  and bound-output hook executions plus position-sliced K/V cache writes; the
  browser smoke verifies bound logits/KV side effects through explicit async
  readback for real `GPUBuffer` resources when an adapter is available as well
  as mock resources. The same
  executions expose ABI-visible host work through
  `zgml_session_runtime_profile`: real browser `GPUBuffer` pattern writes count
  as backend dispatches, mock-resource pattern writes count as fallback work,
  and command counts report the actual two K/V writes plus the optional
  bound-logits write per token-window execution instead of just counting
  successful callbacks. Node/WASI and browser smokes also prove malformed
  executor evidence is rejected: negative or non-integer work counters convert
  the call to `invalid_argument`, leave the host-owned position unchanged, keep
  the rejected StepParams visible for diagnosis, and report the failure through
  the ABI invalid runtime-patch counter without inventing backend, fallback, or
  command work. Host evidence reset through
  `zgml_session_reset_runtime_profile` clears those counters, and
  over-context windows still reject before executor dispatch. This is the
  correct Program/Session/StepParams insertion point and resource-binding proof
  for future browser/Wasm LLaMA kernels, not a fake default execution claim or
  full model-math implementation.
- `zgml_program_create_buffer` allocates correctly-sized host `zgml_buffer`
  handles for the Program's weights, bias, input, and output/logits roles.
  `zgml_program_create_output_buffer` remains the output/logits compatibility
  shorthand. These are host handles, not backend device-buffer APIs, but they
  keep FFI edge allocation tied to the compiled Program contract.
- `zgml_session_generate_sample_tokens` is the first native token generation
  loop. It consumes caller-owned `uint32_t` token IDs, writes generated token
  IDs into a caller-owned output buffer, and keeps logits inside the bound
  Session output buffer. The sampler treats `seed` as a base seed and
  increments it by generated-token index. Its bounded top-k selection is shared
  with the Zig-native LLaMA helpers. It intentionally does not claim tokenizer
  ownership.
- `zgml_llama_program_inspect` exposes fixed tiny LLaMA and fixed SmolLM-135M
  architecture, compile-envelope, and semantic runtime-patch shape evidence for
  FFI hosts.
- FFI sessions now own their host-side persistent, input, and output buffers, so
  a C/JS caller can compile once, free a session, and bind a new compatible
  weight set without mutating the compiled graph tensors.
- C ABI Model and Program handles now retain parents across child lifetimes:
  freeing a Model after compiling a Program, or freeing a Program after binding
  a Session, releases the caller's reference without invalidating the child.
  Regression tests cover parent-before-child free order for tiny-linear
  Sessions and LLaMA Programs.
- `DeviceInference.Session` now binds a backend `RuntimeHandle`, so session
  lifetime owns backend mutable state rather than only host-side `ProgramIO`
  descriptors.
- CPU runtime bindings clone the compiled program's base buffer image and
  mutable `ProgramStencil`, so multiple live CPU sessions can keep independent
  weights, runtime patches, and outputs behind one compiled program.
- The CPU backend accepts an allocator at construction time. This keeps the
  default page-allocator path simple while giving tests a precise way to prove
  compile/bind may allocate and bound step execution does not.
- The stencil backend also clones mutable patch state per runtime binding, which
  keeps inspection evidence separate from runtime mutation. Its runtime profile
  counters are now binding-local as well, so compile-only inspection does not
  share hot execution counters across bound sessions. Shape-only stencil
  execution now records command attempt/dispatch counters from the same
  compiled command shape instead of only incrementing `call_count`; it still
  does not compute outputs or claim GPU execution.
- Metal exposes the same runtime-binding seam and allocates per-session Metal
  buffers plus a mutable stencil. Bound execution now constructs an explicit
  `RuntimeView`; upload, patch, scheduling, command-region checks, fallback
  execution, exact command lowering, lower GPU kernel buffer binding, and output
  download consume that view instead of reaching through the compiled program's
  default buffers. The view also selects runtime-owned quantized qweight buffers
  and reference fallback qweights when a session uploads compatible qweights,
  otherwise it falls back to the compiled program's base qweight table.
- Metal command buffers, active command-kind state, and transient
  command-profile counters now live in a per-execution encoder context. The
  compiled program receives only the merged profile delta after the step.
- The native wgpu executor now follows the same runtime-binding ownership rule:
  compiled programs share device, pipeline, shader, base qweight, and immutable
  shape state, while each bound Session owns its GPU buffers, bind group, mutable
  `ProgramStencil`, runtime qweights, readback staging, profile counters, and
  patchable params uniform buffer. Runtime-window patching refreshes the
  Session-owned params buffer from that Session's patched op tape, and the
  regression test proves two live wgpu bindings can patch slice/cache positions
  independently behind one compiled program.
- DeviceInference profile inspection, reset, and accumulation now call explicit
  backend-owned profile hooks instead of reading or mutating the raw profile
  pointer. Metal serializes default Program and bound Session profile windows
  behind their owning profile locks, so bound execution no longer pollutes the
  shared compiled Program's accumulated counters.
- FFI handle paths expose those runtime-profile windows as explicit
  Program/Session snapshot/reset calls. This keeps C, Node, and Bun Adapters
  thin while preserving the same evidence split available inside Zig.
- The FFI handle path therefore constructs real graph-lowered `ProgramStencil`
  runtime state instead of a parallel math loop, hand-authored backend tape, or
  fake compile handle whose real compilation happens at bind time.
- The C smoke and C ABI tests prove `zgml_buffer` storage can back tiny-linear
  Session weights/input/output binding through `zgml_session_bind_buffers`,
  keeping buffer ownership explicit while preserving the compact
  Program/Session API.
- Remaining backend concurrency work is narrower: the backend Interface now
  exposes explicit profile snapshot/reset hooks instead of a raw profile pointer,
  while future async/session-local evidence should avoid shared accumulated
  program state where possible.

### Phase 5: Node Binding Prototype

Build the first Node FFI wrapper around the C ABI.

Acceptance:

- JS sees model/program/session objects, not raw tensor execution internals.
- A Node smoke test can run a tiny compiled program.
- The wrapper has deterministic cleanup.
- The design supports future async execution without forcing async into the core
  Zig runtime.

Current first slice:

- `src/ts/adapters/bun_ffi_runtime.ts` wraps the C ABI as Bun
  model/program/session objects, selected by the TS-owned
  `src/ts/adapters/bun_concrete_runtime.ts` loader policy.
  The Bun package smoke now first bundles the published `zgml/bun` bridge with
  `bun build --target=bun`, then runs the shared package smoke, so the shipped
  Bun entrypoint has a syntax/module-resolution probe in addition to
  runtime execution. The Bun and Node fixed tiny LLaMA, generic compatible-checkpoint LLaMA, and
  fixed SmolLM-135M facades now share one LLaMA-family Program/Session wrapper
  implementation per host package, with the fixed families reduced to typed
  load/bind specializations rather than copied lifecycle code.
- Bun compile calls accept `{ backend, contextLength, batch }` and encode that
  into `zgml_compile_desc` while keeping raw C struct details hidden from JS
  callers. Node and Bun both reject non-integer or negative `contextLength` and
  `batch` values before FFI, while valid-but-unsupported envelopes such as
  LLaMA `batch: 2` still fail through the native `ZGML_UNSUPPORTED` ABI path.
- The root package entrypoint now re-exports the Node FFI wrapper through
  TS-built `dist/node.cjs`, with
  `index.d.ts` describing the stable
  JS/TS-facing tensor, `nn`, optimizer, Program, Session, and buffer shapes.
  The Bun FFI wrapper is now TS-owned under
  `src/ts/adapters/bun_ffi_runtime.ts` and is exported through `zgml/bun`, so
  examples are consumers instead of package internals. Shared native-library
  load-info policy lives in `src/ts/adapters/native.ts`, while
  `src/ts/adapters/bun_host_runtime.ts` owns the Bun host shell that feeds
  environment/path hooks into that shared resolver. Shared native status error
  formatting/check policy lives in `src/ts/runtime/native_status.ts`; Bun handle
  extraction policy lives in `src/ts/adapters/bun_status.ts`, and Bun
  ABI word-writing helpers live in `src/ts/adapters/bun_abi_words.ts`, so
  descriptor-packing modules share one checked pointer/size slot writer. Bun
  native inspection/compile descriptor/runtime-profile ABI policy lives in
  `src/ts/adapters/bun_inspection_ops.ts`. Bun generic Session upload ABI
  policy lives in `src/ts/adapters/bun_session_ops.ts`, while shared Session
  step I/O descriptor/result policy lives in `src/ts/runtime/session_step_io.ts`. Bun LLaMA
  token-step/execute/selection ABI policy lives in
	  `src/ts/adapters/bun_llama_token_ops.ts`. Bun Program buffer/device ABI
	  policy lives in `src/ts/adapters/bun_program_buffer_ops.ts`. Bun LLaMA
	  KV-cache/session bind ABI policy lives in
	  `src/ts/adapters/bun_llama_session_bind_ops.ts`. Shared Adapter
	  `NativeBuffer` public surface lives in
	  `src/ts/adapters/native_buffer_surface.ts`.
	  Shared generic model/program/session surface lives in
	  `src/ts/runtime/generic_family_surface.ts`, with Bun native model-handle
	  adapter glue in `src/ts/adapters/bun_generic_family_surface.ts`.
	  Bun model path/safetensors
	  load-probe and supported-checkpoint catalog policy lives in
	  `src/ts/adapters/bun_model_source_ops.ts`.
  The wrapper uses `koffi` to expose tiny linear, fixed tiny LLaMA, fixed
  SmolLM-135M, and generic compatible-checkpoint `loadModel(path)` /
  `Llama.load(path)` model/program/session objects over the same opaque C
  handles, with deterministic cleanup, caller-owned typed-array edge I/O,
  native `NativeBuffer` handles for persistent Session bindings,
  `Uint32Array` token windows over the compact C ABI, and a load-time
  `zgml_get_runtime_info` compatibility check. The wrappers preserve the raw
  feature bitset and also expose decoded `runtimeInfo().features.*` booleans,
  including the build-level `nativeWgpuExecution` hint and the compiled-in
  `experimentalLlamaWgpuExecution` native LLaMA WebGPU execution lane and
  `modelPathProbe` for metadata-only checkpoint preflight plus
  `supportedCheckpoints` for the compiled-in checkpoint catalog,
  `safetensorsHeaderProbe` for pathless safetensors-header preflight,
  `safetensorsDataProbe` for native full-byte safetensors preflight, and
  `safetensorsDataLoad` for full byte-backed safetensors loading. Node and Bun
  expose
  `probeModel(path)`, `Llama.probe(path)`, and `SmolLM135M.probe(path)` over
  `zgml_model_probe_path`, so JS can ask which compiled-in envelope a GGUF or
  safetensors path matches before loading weights. They also expose
  `supportedCheckpointModels()` over
  `zgml_supported_checkpoint_count` /
  `zgml_supported_checkpoint_inspect`, so JS hosts can discover the available
  checkpoint envelopes without a file path. The current catalog covers the
  exact fixed tiny LLaMA smoke envelope plus SmolLM-135M; it deliberately does
  not claim arbitrary LLaMA-family compatibility yet. They also expose
  `probeSafetensorsHeader(header)` plus fixed-family aliases, giving browser-ish
  JS hosts a metadata-only exact-envelope query before any weight bytes are
  loaded, root `probeModel(bytes)` plus fixed-family byte-probe aliases, and
  both root `loadModel(bytes)` and `loadSafetensorsData(bytes)` plus
  fixed-family load aliases, giving Node/Bun hosts a pathless `Uint8Array`
  preflight/load path backed by the same native model handle lifecycle. Their
  smokes now prove that byte-probe lane selects SmolLM-135M from host-owned
  safetensors header bytes through root, explicit data-probe, and fixed-family
  aliases without manufacturing a dummy full checkpoint payload.
  For `.safetensors` paths, wrapper `probeModel(path)` now reads only
  the safetensors header prefix/JSON and routes through that same header probe,
  so path preflight remains metadata-only in native JS hosts too. C ABI tests
  plus Node/Bun and Node/WASI Wasm smokes also generate a complete all-zero
  exact tiny LLaMA safetensors file and run it through load selectors, compile,
  bind, and CPU step, proving the tiny catalog row is an executable checkpoint
  path, not just a probe. Loaded model objects expose `model.inspect()` over
  `zgml_model_inspect`, so JS can preflight the model family and envelope before
  compiling a Program. Program objects expose
  `modelCompatibility(model)` and `acceptsModel(model)` over
  `zgml_program_check_model_compatibility`, so JS can preflight dynamic model
  rebinding before crossing into Session bind.
- `zig build ffi-node-smoke` runs the Node tiny-linear and tiny-LLaMA smoke,
  including tiny-linear WebGPU compile-only inspection/bind rejection,
  after the C dynamic library is built. When `ZGML_SMOLLM_MODEL`,
  `ZGML_SMOLLM_GGUF`, or `ZGML_SMOLLM_SAFETENSORS` points at a local compatible
  checkpoint, the Node smoke also runs the
  `ZGML_MODEL_AUTO -> loadModel(path) -> compile -> inspect -> bind -> step`
  lane plus the fixed
  `loadModel(path, { kind: "smollm-135m" }) -> compile -> inspect -> bind ->
  step/prefill` lane. It also probes the same checkpoint path through
  `probeModel(path)` and `SmolLM135M.probe(path)` before load, proving the
  selector can inspect a compatible envelope without allocating a model handle
  or reading weights. The older `Llama.load(path)` and
  `SmolLM135M.load(path)` family helpers remain aliases over the same native
  selectors. It expects
  `npm --prefix examples/node_ffi install` to have installed the prototype
  dependency.
- `examples/bun_ffi/smoke.ts` runs tiny linear inference and fixed tiny LLaMA
  token stepping plus prompt prefill through the dynamic `zgml_c` library. When
  `ZGML_SMOLLM_MODEL`, `ZGML_SMOLLM_GGUF`, or `ZGML_SMOLLM_SAFETENSORS` points
  at a local checkpoint, it also runs the fixed
  SmolLM-135M load/compile/semantic-inspect/executable-inspect/bind/step/prefill
  path and the generic `loadModel(...)` auto-selected compatible checkpoint
  path, after first probing the same path through the metadata-only
  `probeModel(...)` selector.
  Each native handle is freed deterministically.
  `zig build ffi-bun-smoke` builds the dynamic C ABI library first and then
  runs that Bun smoke for environments with Bun installed.
- The Node and Bun wrappers preserve the Session-bound output path through
  `NativeBuffer`: when tiny-linear or LLaMA-family native output buffers are
  bound once at `program.bind(...)`, step/token execution omits the per-call
  output pointer unless the caller explicitly overrides it. Typed arrays remain
  convenient per-call edge buffers. LLaMA-family wrappers also accept
  `program.bind({ output: "native" })`, which creates a Program-sized native
  logits buffer through the compiled Program contract, binds it as Session
  output, lets native selection helpers reuse it without logits readback, exposes
  `session.readOutputInto(...)` for caller-owned logits readback when needed and
  `session.readOutputTensor(...)` for shaped eager logits readback, and frees
  that wrapper-owned buffer with the Session.
  LLaMA-family Sessions also expose `stepTensor(...)`, `executeTensor(...)`,
  and `prefillTensor(...)` as Tensor-fronted logits conveniences over the same
  token execution ABI, leaving `stepInto` / `executeInto` / `prefillInto` as
  the caller-owned hot-loop output path. Caller-owned logits buffers reject
  before native execution when they cannot hold the full vocab-length output.
  LLaMA-family Sessions retain and expose `bufferLayout()` and
  `kvCacheLayout()` from their Program, and LLaMA-family Programs and Sessions
  expose `inputLen()` as zero for the absent f32 tensor input slot plus
  `outputLen()` / `outputShape()` as the logits edge, so callers can use the
  same slot/shape vocabulary for generic tensor Programs and token Programs.
  `TinyLlamaSession.stepContract()` exposes the token StepParams contract in one
  frozen snapshot too: token-vs-window support, logits length/shape, context
  position, remaining context capacity, bound-output kind, default output/readback
  path, readback availability, logits byte length, output slot name/role,
  no-output state advancement, `defaultReadbackRequired`,
  `defaultAllocationFree`, `defaultHotPath`, default output
  effect/ownership/return ownership, and a stable `signature` that excludes the
  moving position.
  `TinyLlamaSession.matchesStepContractSignature(signature)` compares that
  cached contract directly.
  `TinyLlamaSession.acceptsStepParams(params)`
  reuses the token validation path as a non-executing guard before a hot loop.
  `TinyLlamaSession.acceptsAllocationFreeStepParams(params)` adds the same
  no-runtime-output-allocation predicate over that validation path.
  `TinyLlamaSession.acceptsRuntimeOutputAllocationFreeStepParams(params)` names
  that exact predicate while keeping the older compatibility alias available.
  `TinyLlamaSession.acceptsNoReadbackStepParams(params)` adds the matching
  no-readback hot-loop predicate over that same validation path.
  `TinyLlamaSession.acceptsReadbackFreeStepParams(params)` names that same
  positive evidence predicate while keeping the older compatibility alias
  available.
  `TinyLlamaSession.acceptsHotStepParams(params)` combines those two hot-loop
  predicates: accepted, allocation-free, and no-readback.
  `TinyLlamaSession.requireAllocationFreeStepParams(params)`,
  `TinyLlamaSession.requireRuntimeOutputAllocationFreeStepParams(params)`,
  `TinyLlamaSession.requireNoReadbackStepParams(params)`, and
  `TinyLlamaSession.requireReadbackFreeStepParams(params)` mirror the generic
  Session assertion-shaped preflight guards for individual hot-loop facets.
  `TinyLlamaSession.matchesStepParamsSignature(params, signature)` compares a
  token StepParams shape against a cached `stepParamsSignature` without
  executing.
  `TinyLlamaSession.matchesStepParamsCompatibility(params, compatibility)`
  compares against a frozen preflight record without making callers unwrap the
  raw signature string.
  These token preflight helpers do not execute or bump call counters.
  `TinyLlamaSession.preflightStepParams(params)` is the friendly evidence verb,
  while `TinyLlamaSession.stepParamsCompatibility(params)` remains the
  compatibility alias. Both accept unknown boundary values and return the frozen
  accepted flag, typed accepted/rejected status, contract kind, stable contract
  signature, stable StepParams call-shape signature, current contract position,
  per-call `stateEffect` and `allocationFree`
  evidence, typed `inputSource` / `outputTarget` / `outputEffect` routing
  evidence, planned input/output element types, shapes, shape signatures,
  element lengths, and byte lengths, `inputOwnership`, `outputOwnership`, `outputReturnOwnership`,
  `readsInput`, `writesOutput`, `readbackRequired`, `readbackFree`,
  `runtimeOutputAllocationFree`, top-level `hotPath`, `hotPathStatus`,
  `hotPathBlockers`, `rejectionCode`, and typed
  rejection diagnostics for the same check.
  The public type smoke and package runtime smoke now prove LLaMA Program and
  Session inspection, buffer-layout, direct input/output length helpers,
  output-shape, and runtime-profile evidence directly: token Programs
  intentionally have no tensor input slot (`inputLen` is zero), while logits
  remain the shaped output edge.
  `NativeBuffer.readFloat32Into(...)` now fills caller-owned logits readback
  storage without allocating a fresh result array, and
  `NativeBuffer.readBytesInto(...)` does the same for generic byte-oriented FFI
  buffers, matching the browser/Wasm ownership rule for caller-owned readback.
  Both wrappers expose `NativeBuffer` for
  zgml-owned storage, `NativeBuffer.fromBytes(...)` / `NativeBuffer.fromFloat32(...)`
  for owned host buffers, `NativeBuffer.wrap*` for explicit non-owning typed-array
  views, `NativeBuffer.externalResource(...)` for opaque future device-resource
  views, and call `zgml_session_bind_buffers` for native handle binding. Tiny
  tensor programs expose a Program-sized `createBuffer(kind, options)` factory
  for weights, bias, input, and output, backed by the generic C Program buffer
  factory, including resource/device factories, so JS callers do not duplicate
  persistent/input/output byte math. The older role-specific helpers remain
  aliases. JS Program objects also expose `capabilities()`, `canExecute()`,
  `canBindExternalResources()`, `hasFullDispatchPlan()`, and
  `executionMode()` as thin views over executable Program inspection. The
  capability object also exposes `mode: "executable" | "resource-probe" |
  "compile-only"` plus decoded dispatch-plan detail fields, so hosts can
  distinguish real execution from a WebGPU-shaped resource probe without
  guessing from backend names.
  `NativeBuffer.inspect()` reports host vs external-resource storage
  plus placement/access/offset metadata, while `session.inspect()` reports the
  actual bound runtime storage/count shape. The
  Node, Bun, Node/WASI, and browser Wasm smokes assert host Session bindings and
  both host and fake WebGPU resource Buffer shapes. Their
  smokes now prove host `NativeBuffer` KV caches can back LLaMA execution,
  `program.createOutputBuffer({ resource })` can produce resource-backed output
  descriptors from the compiled Program output contract,
  `program.createKvCache({ resource })` can produce resource-backed K/V
  descriptors from the same compiled Program sizing contract, tiny-linear
  WebGPU can bind opaque resource-backed Sessions for inspection while still
  rejecting step execution, LLaMA output resource binding reaches the native
  ABI, and resource views return unsupported on host-only backends for both
  normal and model-rebound sessions.
		  Node and Bun now share the `ProgramDevice` facade policy too:
		  `src/ts/runtime/program_device.ts` owns placement liveness, frozen
		  ProgramDevice info evidence, role-specific device-buffer helpers,
		  same-device import defaults, and KV-cache device-buffer routing, while each adapter only supplies native callbacks and
		  its host-specific `LlamaKvCache` constructor. Program-owned buffer factories
		  now come from `src/ts/runtime/program_buffer_factory.ts`: output/input/weights/bias/KV
		  buffer creation, resource callback validation, Program requirement sizing, and
		  host-vs-placement-vs-resource selection are shared typed Program facade policy,
		  while Node and Bun retain only the native buffer allocation/import calls.
		  Generic `Program` facade policy is shared as well:
	  executable evidence/capability/lifecycle policy comes from
	  `src/ts/runtime/program_facade_policy.ts`, `bindModule(...)` routing
	  comes from `src/ts/runtime/program_module_binding.ts`, and `ProgramDevice`
	  policy comes from `src/ts/runtime/program_device.ts`; retained compile-evidence
	  access, edge-shape accessors,
	  requirements-to-layout projection, trace/IR/KernelPlan views, model/module
	  compatibility acceptance, execution-mode capability booleans, runtime-profile
	  snapshot/reset, deterministic free/dispose, and Program wrapper policy are
	  now TS-authored and imported through package-built artifacts. LLaMA-family Program bind/free/dispose
  and fixed-family Session
  factory handoff share the same frontend policy shape.
	  Model inspection, compile-to-Program lifecycle, deterministic model
	  free/dispose, and LLaMA-family model create/load, byte-load, path/header
	  probe, and compile routing now come from `src/ts/runtime/model_source.ts`,
	  emitted to `dist/runtime/model_source.cjs`, with adapters retaining only native
	  handle callbacks plus concrete Model and Program constructors.
	  Generic tensor Session call policy now follows that same shape from
	  `js/shared_session_facade.cjs`: bound-input/output selection, no-output
	  advancement, inspection, shape/layout accessors, persistent-parameter upload,
	  reset/profile, and deterministic owned-buffer cleanup are shared frontend
	  semantics, while Program edge-shape and Session tensor-output wrapping/readback
	  helpers live in `src/ts/runtime/session_tensor.ts` and StepParams facade
	  predicates/signature matching plus call-profile facade wrappers live in
	  `src/ts/runtime/session_facade.ts`, which also owns shared Session
	  buffer-sizing facade assembly/signature matching and runtime-profile
	  read/match/reset wrappers plus generic Session parameter inspection/upload
	  facade methods.
	  Node and Bun keep only the
	  native `session_step` and lifecycle symbol packing.
  LLaMA-family Session lifecycle policy follows the same Module seam in
  `js/shared_session_facade.cjs`: position, inspection, reset, runtime-profile
  snapshot/reset, and deterministic free/dispose are shared, while Node and Bun
  supply only native handle callbacks and their null-handle value.
  The portable Node/WASI and browser Wasm LLaMA resource helper now mirrors the
  same Program-owned slot model: `WasmWebGpuLlamaResourceProgram` derives logits
  and K/V byte sizes from compiled Program metadata, creates Program-sized
  resources through `createOutputBuffer()` / `createKvCache()`, and accepts raw
  browser `GPUBuffer`/mock resources or already-wrapped resource bindings before
  emitting the ABI bind descriptors.
  The portable Wasm tiny-linear host Program now uses the same resource-slot
  shape: role-specific helpers create wrapped weights/bias/input/output slots,
  `bind(...)` accepts omitted slots, raw browser resources, or pre-wrapped
  slots, and Node/WASI plus browser smokes prove the bound ABI Session shape.
  Single-token JS `session.step(token)` calls now route through the scalar
  `zgml_session_step_token` ABI instead of allocating a wrapper-side one-token
  array, while multi-token prefill/advance continues to use
  `zgml_session_execute_tokens`. Node and Bun sessions now also expose
  `session.execute({ token | tokens, output })` as the public
  StepParams-shaped alias over that same token execution ABI, while retaining
  `executeTokens(...)` for callers that want the raw token-window name. Their
  caller-owned logits API now has explicit output-first names too:
  `session.stepInto(out, token)`, `session.prefillInto(out, tokens)`, and
  `session.executeInto(out, { token | tokens })`, which forward to the same
  native step/token execution ABI instead of copying from an internal logits
  buffer after execution. They also expose `session.stepArgmax(token)`,
  `session.generateTokensArgmax(tokens, maxTokens)`,
  `session.generateTokensArgmaxInto(tokens, outputTokens)`,
  `session.stepSample(token, ...)`, and
  `session.generateTokensSample(tokens, maxTokens, ...)`, plus
  `session.generateTokensSampleInto(tokens, outputTokens, ...)`, backed by
  native bound-logits helpers so generation can return token IDs without
  copying vocabulary logits over FFI or allocating a new token array in each JS
  generation loop. The returned selection/generation records are frozen,
  read-only snapshots while their token buffers remain the explicit data edge.
  C, Node/WASI, and browser Wasm smokes also exercise
  `zgml_session_step(session, NULL, &result)` for all-bound tiny-linear I/O.
  Node and Bun smokes now also prove fixed tiny LLaMA Program runtime profiles
  stay cold while bound Session profiles record hot patch/execute calls and can
  be reset without losing command-shape evidence.
- The Bun wrapper exposes deterministic `free`/`dispose`/`Symbol.dispose`
  cleanup and its smoke test proves compile-once, sequential rebind with a new
  persistent weight set from JS, JS-reused tiny-linear input/output typed-array
  edge buffers, native-buffer tiny-linear weights/input/output bindings,
  per-call output overrides over native-bound tiny-linear output buffers,
  native-buffer LLaMA logits bindings, native greedy argmax over both caller and
  bound logits, per-call logits overrides over native-bound LLaMA output
  buffers, resource-output LLaMA binding rejection on host-only backends,
  rejection of unbound no-readback token helpers, no-readback `stepArgmax`,
  fixed tiny LLaMA semantic/executable program inspection,
  fixed SmolLM-135M loading, auto-selected compatible
  checkpoint loading, JS-side rejection of malformed compile envelopes,
  native rejection of unsupported batch envelopes, independent LLaMA session
  positions, session reset/replay, no-output token advancement, prompt prefill,
  bulk no-output prompt advancement, over-context execute/prefill rejection
  before state or output mutation, prefill-to-decode cache handoff,
  caller-owned logits buffers, Session-bound logits output, and the tiny-linear
  plus fixed tiny LLaMA WebGPU compile-only inspection lanes.
  Executable inspection includes backend,
  execution support, runtime patch shape, and command-category count evidence
  so JS tests can prove they received a real compiled command shape without
  depending on internal command enum values. For executable decode/prefill sessions,
  `stepInto`/`prefillInto`, `zgml_session_execute_tokens`,
  `zgml_session_step_token`, and `zgml_session_prefill_tokens` bind the caller
  buffer as that execution's output target instead of copying from an internal
  logits buffer after execution. The JS wrappers now implement scalar
  `step(token)` through `zgml_session_step_token`, and use the unified
  execute-token ABI for `prefill` and no-output advancement with plain number
  arrays or `Uint32Array` token windows rather than BigInt token buffers.
  Native Node/Bun `advanceTokens(...)` now also reuses a Session-owned
  no-output options scratch instead of spreading caller options, preserving the
  caller object while forcing logits output/readback off for prompt
  advancement. Caller-output helpers now follow the same rule:
  `execute({ token, output })`, `executeInto(out, ...)`, and
  `prefillInto(out, ...)` reuse Session-owned scalar/output options scratch
  while leaving caller params/options unmodified.
- Bun and Node remain FFI lanes over the same C handle surface; a future
  production Node adapter may still move to Node-API once the handle surface
  stabilizes.
- `ZGML_BACKEND_WEBGPU` now maps to a WebGPU compile-only stencil target for
  default tiny-linear, direct tiny-MLP, traced module, and LLaMA-family
  programs. Zig, C, Node, Bun, Node/WASI, and the browser smoke can compile
  tiny-linear, supported traced module graphs, direct tiny-MLP handles, and
  fixed tiny LLaMA with that backend, inspect executable shape evidence, verify
  `backend = WEBGPU` plus command-category evidence, and observe whether the
  compiled Program is executable through
  `execution_supported`. Default tiny-linear builds expose a WebGPU
  external-resource Session probe: resource binding succeeds through the real
  Program/Session handle path and inspection reports external-resource
  storage/counts, but `step` and no-output advance still return unsupported and
  record zero bound Session runtime work. Default direct tiny-MLP WebGPU
  Programs are compile-only and do not claim external-resource binding; traced
  module Programs expose the generic external-resource binding shape while still
  reporting resource-probe versus executable mode.
  When the native C library is built with `-Duse-wgpu=true`, tiny-linear,
  direct tiny-MLP, and traced module WebGPU instead report
  `execution_supported = 1`, bind host tensors, and execute through the real
  wgpu-native backend. Tiny-linear additionally rejects fake
  external-resource bindings, exposes program-owned device buffers, and imports
  same-device device-buffer handles through C/Node/Bun. Fixed tiny LLaMA exposes
  the matching
  resource-session probe for
  external-resource logits plus K/V cache binding and inspection, while
  host-memory WebGPU binding and token execution remain unsupported before
  cloning runtime Session state or silently falling back to CPU.
  Default-build C ABI tests also call the device-buffer factory, device-handle,
  and import exports on the compile-only WebGPU Program and prove they return
  unsupported with cleared outputs, so the public API surface cannot be mistaken
  for an executable same-device resource lane.
  LLaMA-family programs also expose semantic shape evidence.
- `zig build ffi-wasm` now builds `zig-out/bin/zgml_c.wasm`, a no-entry Wasm
  module exporting the same `zgml_*` handle functions and linear memory. This
  is not the WebGPU backend yet, but it proves the C handle surface can be
  shaped as a Wasm module instead of only a native dynamic library.
- `zig build ffi-wasm-smoke` runs a Node/WASI smoke over that module:
  check runtime ABI/feature metadata, allocate linear-memory descriptors, create
  tiny linear Model, inspect the model envelope, compile Program, inspect
  executable evidence, bind Session,
  step into a caller-owned output buffer, inspect cold Program versus hot
  Session runtime profiles, reset the Session runtime profile while preserving
  command evidence, then bind native
  `zgml_buffer` weights/input/output storage through `zgml_session_bind_buffers`
  and execute with no step descriptor before freeing handles. It also wraps
  caller-owned linear-memory weights/input/output storage through
  `zgml_buffer_wrap`, binds those handles, mutates the input/output memory
  directly, and proves the wrapped handle does not own the linear-memory region.
  It also runs the
  fixed tiny LLaMA
  handle lifecycle with
  model-envelope, semantic/executable inspection, independent sessions,
  too-small output failure
  checks, token step, reset/replay, no-output advance, prompt prefill, and
  caller-owned logits buffers plus native-buffer Session-bound logits output.
  Before that synthetic-model lane, the smoke writes a complete all-zero exact
  tiny LLaMA safetensors checkpoint into a Node/WASI preopened directory, probes
  it through `zgml_model_probe_path`, rejects the wrong fixed family, loads it
  through both `ZGML_MODEL_AUTO` and `ZGML_MODEL_TINY_LLAMA`, compiles the
  loaded model, binds a CPU Session, and steps a token with zero logits and an
  unchanged sentinel. The same smoke also passes those complete safetensors
  bytes directly through `zgml_model_probe_safetensors_data` and
  `zgml_model_load_safetensors_data`, proving the pathless Wasm/Node-shaped
  selector can preflight, create, and execute the same tiny LLaMA model handle
  without relying on a preopened directory. It also passes SmolLM-135M
  safetensors header bytes through `zgml_model_probe_safetensors_data`, proving
  the same pathless full-data descriptor can select the SmolLM envelope from
  host-owned bytes without manufacturing a dummy full checkpoint payload.
  The Wasm smokes allocate exact Program-sized weights/input/output/logits
  handles through the generic `zgml_program_create_buffer(program, kind, ...)`
  factory; raw `zgml_buffer_create` is kept only for intentionally oversized
  sentinel buffers that prove output/logits writes respect the Program-declared
  span.
  It also binds a compatible tiny LLaMA model handle as Session state with both
  caller-owned logits and native-buffer logits output, so the portable Wasm
  handle lane now exercises the same compiled-Program/dynamic-Session model
  binding shape as C, Node, and Bun. It also binds host `zgml_buffer` KV caches
  as caller-owned Session state and executes a token through them. The same
  smoke binds fake external-resource outputs for normal and compatible-model
  tiny LLaMA sessions and proves host-only Wasm execution rejects both without
  returning a Session handle.
  Both caller-logits and native-bound logits are checked with
  `zgml_session_argmax_token`, and native-bound token windows are checked with
  `zgml_session_execute_argmax_tokens`, proving the common greedy loop can avoid
  a full logits readback and a second FFI call.
  It also covers
  `zgml_session_execute_tokens` with logits output, over-context rejection
  before state/output mutation, no-output prompt advancement, and the
  prefill-to-decode cache handoff. The smoke also covers
  the WebGPU compile-only tiny-linear, direct tiny-MLP, traced module, and
  fixed tiny LLaMA lanes.
  Tiny-linear now exposes an opaque resource-session probe whose bind can be
  inspected while step and no-output advance return unsupported without recording
  Session runtime work; fixed tiny LLaMA now exposes the same inspectable
  resource-session probe for WebGPU-labeled logits and K/V cache resources
  while token execution still returns unsupported. Tiny-MLP and traced module
  WebGPU smokes prove default compile-only rejection and native `-Duse-wgpu=true`
  host-buffer execution; direct tiny-MLP still does not claim external-resource
  binding, while traced module Programs expose the generic resource-bind shape.
  The Node/WASI smoke also
  binds those WebGPU-labeled resources together with an independent compatible
  model handle, proving the portable model-bound Session path reaches the same
  resource-session shape without mutating state. Executable inspection checks
  include command-category totals and derived Program capability decisions, so
  the Wasm smoke proves a concrete command shape and the same
  `executionMode` / `canExecute` / `canBindExternalResources` distinction as
  Node/Bun, not only a nonzero hash.
  This smoke is now included in `zig build check`, so the portable Wasm handle
  artifact is validated by the normal local gate as well as its explicit build
  target.
- `examples/wasm_ffi/browser.html` runs the same tiny linear plus fixed tiny
  LLaMA lifecycle in a browser by fetching `zig-out/bin/zgml_c.wasm` and using a
  small browser-side WASI shim for the no-file smoke path. The page exposes a
  DOM pass/fail marker at `document.documentElement.dataset.zgmlWasmSmoke` and
  also verifies runtime ABI/feature metadata, decoded runtime feature bits
  including model-auto, Wasm-export, device-buffer factory/import, dispatch-plan
  inspection, ABI struct-size introspection, and portable non-native-wgpu state,
  plus exported struct-size-derived descriptor allocation,
  unsupported/cleared-handle device-buffer factory, device-handle, and import
  probes,
  pathless exact tiny LLaMA safetensors byte preflight/load through
  `zgml_model_probe_safetensors_data` and
  `zgml_model_load_safetensors_data`, pathless SmolLM-135M safetensors
  data-probe selection from host-owned header bytes,
  tiny-linear Program/Session runtime-profile snapshots and reset,
  Session-bound native `zgml_buffer` weights/input/output storage for tiny
  linear, LLaMA session reset/replay,
  native-buffer Session-bound logits output, compatible model-handle Session
  binding, LLaMA output-resource binding rejection on host-only backends, the
  WebGPU compile-only tiny-linear and LLaMA inspection paths, tiny-linear
  WebGPU external-resource Session binding with unsupported step/no-output and
  zero runtime work, LLaMA WebGPU resource-probe step/advance plus
  argmax/sample/generation rejection with unchanged Session state, native
  greedy argmax over caller and bound logits, native
  greedy generated token IDs, native
  bounded sampling, and native sampled generated token IDs,
  one-call execute-and-argmax over bound logits, executable command-category
  evidence, a shared `WasmWebGpuDevice` host wrapper that maps GPUBuffer-like
  objects to opaque WebGPU resource handles for portable Wasm resource
  descriptors, validates WebGPU usage flags plus same-device provenance
  against requested resource access, hashes host-side resource tables before
  binding,
  a shared `WasmExternalResourceBridge` that writes those descriptors into Wasm
  linear memory before `zgml_buffer_wrap_resource`, a shared
  `WasmSessionBindingBridge` that writes the resource-backed tiny-linear and
  LLaMA Session bind descriptors, a shared
  `WasmWebGpuLlamaResourceProgram` that binds WebGPU-labeled LLaMA logits/KV
  resources from Program-derived output/KV sizes, accepts raw host resources or
  pre-wrapped resource bindings, registers the claimed host token bridge
  session in one step, and keeps host-side token profile counters for decoded
  calls, logits-output calls, no-output calls, validation rejections, result
  writes, and
  valid-but-unsupported execution attempts, plus a named deterministic
  token-window pattern executor that can drive bound-logits and K/V side
  effects through real browser `GPUBuffer` compute when an adapter is available
  while remaining a resource-binding proof rather than LLaMA model math,
  actual browser `GPUBuffer` registration when `navigator.gpu` is available, a
  narrow browser host Program/Session plus a `zgml_host` import bridge. The raw
  exported Wasm
  `zgml_session_step` and `zgml_session_step_no_output` functions now try that
  host bridge for registered tiny-linear WebGPU `zgml_session` handles. Browser
  `GPUBuffer` mode enqueues real WebGPU compute; portable mock mode executes
  the same resource table against deterministic host buffers so Node/WASI and
  headless browser smokes prove the import/session boundary too. Both modes
  write `zgml_step_result`-shaped output lengths and prove no-output execution
  by changing the bound input, returning `output_len = 0`, and then reading the
  changed host output resource back. They fall back synchronously to normal Wasm
  execution or unsupported resource-probe status for unregistered handles. The
  raw exported
  `zgml_session_free` also calls the host bridge so claimed browser GPU
  sessions are cleaned up before the Wasm handle is destroyed. The
  wrapped-exports facade remains a compatibility helper, and now claims
  tiny-linear `zgml_session_step` / `zgml_session_step_no_output` plus
  `zgml_session_execute_tokens` for registered LLaMA resource sessions, so
  proxy users hit the same synchronous host bridge as raw Wasm imports instead
  of receiving a Promise where the C ABI expects a status integer.
  Freeing a claimed host Session now also removes its claim from the host
  runtime, so a later Wasm handle that reuses the same numeric value can fall
  through the normal native path instead of being misrouted to a stale bridge.
  This sits alongside model-bound WebGPU resource-session inspection for logits
  and K/V cache handles.
- `zig build ffi-wasm-browser-smoke` drives that browser page through
  Chrome/Chromium and CDP without Playwright/Puppeteer. It is the portable
  browser gate: mock mode is accepted when no WebGPU adapter exists, but the
  page log and dataset explain that boundary. `zig build check` now includes
  this non-required browser smoke so portable browser/Wasm host-resource
  regressions are part of the default local/subagent gate.
- `zig build ffi-wasm-browser-llama-focused-smoke` runs the same browser page
  with `--llama-profile-label=gguf-smollm3-nope-gqa-pipeline`, giving the goal
  scorecard a narrower SmolLM3 GGUF LLaMA-family proof that checks focused
  browser execution without requiring the full browser profile matrix on every
  scorecard run.
- `zig build ffi-wasm-browser-llama-family-focused-smoke` runs a compact
  representative browser family proof over SmolLM3 GGUF NoPE/GQA,
  long-sliding-window Mistral, and Qwen3 Q/K-norm labels. The runner expands
  each label to its normal and greedy generation profile, so this six-profile
  proof exercises broader checkpoint-family execution without paying the
  full required-GPU matrix cost on every iteration.
- `zig build ffi-wasm-browser-gpu-focused-smoke` runs that same focused browser
  LLaMA proof with Chrome WebGPU flags and `--require-gpu`. The goal scorecard
  attempts this lane after the portable focused browser smoke: on machines with
  usable Chrome/WebGPU it must report `mode=gpu-buffer`, `canBindBlockPipeline=true`,
  two focused LLaMA profile labels, backend dispatch evidence, and zero fallback
  ops; on machines without that local browser/GPU path it records an explicit
  skip note instead of pretending mock storage is real GPU execution.
- `zig build ffi-wasm-browser-gpu-llama-family-focused-smoke` is the matching
  required-GPU representative family lane. It runs the compact SmolLM3 GGUF
  NoPE/GQA, long-sliding-window Mistral, and Qwen3 Q/K-norm family proof in
  real `GPUBuffer` mode, with the same normal plus greedy expansion as the
  portable representative family smoke. This is still a bounded proof lane, not
  a claim that full/default browser LLaMA-family execution is complete.
- `zig build ffi-wasm-browser-gpu-smoke` runs the same page with Chrome WebGPU
  flags, requires real `GPUBuffer` mode, and fails if the adapter cannot bind
  the six-storage-buffer LLaMA block-pipeline proof. It is an exhaustive
  opt-in gate with a 30-minute budget, not part of default `zig build check`.
  The June 22, 2026 local M5 Pro/Metal adapter pass took about 27.5 minutes
  and recorded `maxStorageBuffers=8`, `storageAlignment=256`,
  `canBindBlockPipeline=true`, 74 LLaMA profile labels, 253 backend dispatches,
  249 executor dispatches, 4 device-selection dispatches, 127/127/0/0/0
  storage-mode calls, 41 output reads, 4 selection reads, 45 syncs, and zero
  fallback ops. The no-adapter browser mock gate still records the same
  74-profile-label matrix, storage-mode call splits, output reads, and explicit
  mock fallback work when a real browser GPU path is unavailable.
  The Node/WASI portable Wasm smoke now keeps a matching packed-family LLaMA
  profile ledger and fails on missing or duplicate normal/native/greedy labels
  before printing `zgml wasm ffi webgpu LLaMA packed proof ok: labels=...`.
  For debugging one packed-family browser path without redefining the proof
  gate, `examples/wasm_ffi/browser_smoke_runner.mjs` accepts
  `--llama-profile-label=<label>`; the filtered page now skips unrelated
  LLaMA profile producers and runs only the selected packed proof plus its
  greedy variant when requested. The runner rejects focused results that report
  unrelated LLaMA profile labels, while the normal build targets omit that
  filter and still require the full matrix.
- Core source portability for that Wasm artifact now avoids pthread imports on
  targets without pthreads, compiles graph/fused threading paths to serial
  implementations on single-threaded targets, and uses explicit checked casts
  for GGUF/safetensors file sizes and counts on wasm32.

### Phase 6: Backend Step Quality

Make backend execution match the architecture's performance promise.

Acceptance:

- Metal path caches the right backend objects and avoids hot-path allocations.
- CPU path has a clear function-tape baseline.
- wgpu/WebGPU design and compile-only inspection evidence land before execution.
- Every abstraction change is benchmark-gated against current artifacts.

Current slice:

- The CPU compiled Program path now builds a small command-shaped reference
  `ExecutionTape` during compile from the `ProgramCommand` stream owned by the
  `ProgramStencil`, so command evidence and CPU reference execution share one
  cold command-lowering result. Commands group cached entries; each entry stores
  the op index plus the already-selected runner for that op kind. Runtime still
  reads the mutable patched `ProgramStencil` op payloads, so
  `RuntimeWindow`/StepParams updates affect execution without re-selecting op
  behavior or allocating in the hot path. Regression tests prove the tape
  observes patched op fields, compiled CPU Programs own the tape, the tape
  command count matches the inspected command stream, and bound CPU execution
  records command attempt/dispatch profile counters.
- Metal compile now builds the owned `ProgramStencil` before backend scheduling
  and derives its cached `ExecutionPlan`/region command plans from
  `ProgramStencil.ops` using the same command policy as stencil inspection.
  Metal's scheduled lowering therefore plans against the same mutable op tape
  that runtime window updates patch before execution, and the region-batching
  regression checks the cached region command plan against the
  ProgramStencil-owned command stream for a whole-program region.
- `DeviceInference.Session.executeStep` now refuses non-executing backends
  before applying runtime-window patches or dispatching. A WebGPU compile-only
  resource-session regression still proves Program/Session inspection and
  external-resource binding shape, but `executeStep` returns unsupported with
  zero patch/dispatch profile counters. This keeps WebGPU compile-only evidence
  useful without letting a direct Zig caller mistake shape execution for real
  GPU output.
- The current WebGPU path now uses explicit `webgpu_compile_only` and
  `webgpu_resource_plan` capability shapes instead of starting from the future
  executable `webgpu` capability and clearing execution later. This keeps
  compile-only resource planning distinct from the real wgpu/WebGPU backend
  capability that still needs to own GPU resources and execution. The opt-in
  native wgpu executor also narrows its advertised capabilities to the ops and
  bounded dispatch plans it can actually compile and dispatch, so full
  LLaMA-family WebGPU execution, JS-owned GPUBuffer interop, and broader
  attention/movement families cannot be inferred from the future WebGPU
  resource-plan shape. Its backend-level `supportsProgram` check is
  dispatch-plan-aware too: standalone dense
  f32 matmul now passes through its own executable shader, standalone qmatmul
  passes through its own executable shader with packed int8 qweights plus f32
  scales that can be replaced as Session state, fused qmatmul-add accepts a
  two-op `DeviceProgram` and skips the intermediate output buffer, dense
  matmul-elementwise add/mul sidecars plus dense matmul-fused-elementwise,
  matvec-fused-elementwise, and qmatvec-fused-elementwise activation sidecars
  now compile as single dispatches with the projection value available as a
  virtual secondary operand, repeat feeding fused elementwise now compiles as
  one activation-chain dispatch with the repeated value treated as a virtual
  secondary operand, and standalone
  add/mul/neg/abs/sgn/step/relu/sqrt/sqr/recip/exp/log/gelu elementwise programs pass through
  their own executable shader, and standalone fused elementwise chains pass
  through their own executable shader for up to eight steps and six distinct
  secondary buffers. Standalone LayerNorm, standalone RMSNorm, and standalone
  softmax now pass through their own row-workgroup executable shaders, and
  standalone reduce sum/max passes through its own row-workgroup executable
  shader. Standalone repeat now passes through its own strided movement shader.
  Standalone RoPE now passes through its own packed low/high pair
  executable shader, and standalone slice-assign passes through its own strided
  movement shader. Bounded standalone attention now passes through a fast
  one-workgroup-per-query executable shader for `seq_kv <= 256` and
  `d_head <= 256`, plus a streaming softmax shader for longer key/value
  windows up to `seq_kv <= 2048` at the same `d_head <= 256` bound. The wgpu
  backend now also has an internal dispatch-plan
  inspection helper that reports full op coverage, covered op count, backend
  dispatch count, first unsupported op, and dispatch-family counts from either a
  raw `DeviceProgram` or the compiled program's owned `ProgramStencil`. ABI v6
  now exposes the same dispatch-plan evidence through `zgml_program_inspect` for
  real native WebGPU executable Programs, and the Node, Bun, Node/WASI, and
  browser Wasm wrappers/smokes decode those fields. Node/WASI and browser Wasm
  now also assert the decoded capability fields stay empty for CPU and
  compile-only WebGPU paths, so hosts can distinguish command-shape evidence
  from a lowered executable backend dispatch plan. That evidence now flows
  through the generic backend `inspectExecutionPlan` hook and native
  `DeviceInference.Program.inspect()` rather than through a C-ABI-only wgpu
  adapter, so Zig-native, C, and JS/Wasm hosts see the same compiled Program
  evidence surface.
- The high-level LLaMA facade now treats non-executing resource Sessions as an
  inspection/capability state rather than as hard-coded WebGPU behavior:
  resource-probe binding is allowed only when executable inspection reports
  `execution_supported = false`, `external_resources_supported = true`, and the
  caller provides output plus K/V resource bindings. A future real WebGPU
  backend can therefore become executable by changing backend evidence, not by
  fighting enum-specific adapter logic.
- Native `ProgramStencil.inspect()` and `DeviceInference.Program.inspect()` now
  expose the bounded runtime patch envelope: max cache write position and max
  attention sequence length. Focused tests prove those inspected bounds match
  the same windows that hot patch execution accepts or rejects. The same
  envelope is now exported through `zgml_program_inspect` and surfaced by the
  C, Node, Bun, Node/WASI, and browser Wasm smokes so embedders can size future
  StepParams/update buffers from compiled Program evidence.
- Metal now has an allocator seam through `MetalBackend.initWithAllocator(...)`
  while preserving the default `MetalBackend.init()` page-allocator path.
- A bound Metal executable regression compiles and binds with an injected
  failing allocator, closes that allocator after bind, then patches and executes
  the same runtime with output download and no-output advancement. The test
  proves the bound Metal hot path performs no Zig heap allocations or resizes
  after Session binding.
- Metal now has LLaMA-shaped decode and fixed-window prefill regressions that
  compile and bind through the persistent Program/Session path, close the same
  injected allocator after binding, then run logits-output and no-output steps.
  The tests prove transformer-shaped Metal runtime-window patching, backend
  dispatch, caller-owned logits buffers, prompt cache side-effect download, and
  no-output advancement perform no Zig heap allocations or resizes after
  Session binding.
- `backend.ProgramIO` now has an explicit external-resource branch next to the
  host-memory branch. `DeviceInference.TensorBinding` can express that future
  resource shape for persistent, input, and output bindings; `DeviceProgram`
  rejects resource initial uploads unless the backend capability says they are
  supported; and current host-only CPU/Metal session binding fails before
  upload/execute rather than treating a device handle like a host pointer. The
  same distinction is now visible through `zgml_buffer_wrap_resource` in the C,
  Node, Bun, Node/WASI, and browser Wasm lanes, and `zgml_buffer_inspect` makes
  the host/resource split and resource metadata observable before bind.
  `DeviceInference.Program.inspect()` and `zgml_program_inspect` now report
  whether the compiled backend computes real outputs separately from whether it
  can accept external-resource Session bindings. WebGPU stencil evidence
  therefore reports compile-only shape with `execution_supported = false`,
  while the resource-capable test backend reports the same WebGPU-shaped
  resource path as executable. JS/Wasm hosts do not need to use bind failure as
  a capability query. They also expose planned op count, buffer count, total
  buffer elements, byte length, initial upload count, qweight count, and runtime
  patch envelope bounds as cold memory-layout and StepParams evidence for
  future WebGPU/wgpu resource planning.
  Resource-capable Programs only accept resource descriptors whose placement
  matches the backend device and whose access flags allow their role:
  persistent/input resources must be readable and outputs must be writable.
  `DeviceInference.Program` now rejects mismatched placement or bad access
  before the backend hook is called, and the same validation runs for dynamic
  per-call input/output overrides before runtime-window patching and for
  explicit persistent re-upload before upload. A
  resource-capable test backend then proves persistent, input, and output
  external-resource descriptors flow through a backend-owned
  `configure_bindings` hook at Session bind time, before persistent upload,
  runtime patch, or execute. That gives future WebGPU/wgpu execution a place to
  build bind groups/resource tables outside the hot step path. The regression
  then proves persistent upload, runtime patch, and execute receive those
  descriptors without host-pointer conversion, and that mismatched placements or
  wrong access modes reject before configure/upload. It now builds the same
  `ProgramStencil` command shape during compile, exposes that shape through
  immutable cold Program inspection, preserves it across Program and Session
  profile reset, and records command-shaped attempts/dispatches for the
  resource-backed step instead of a generic dispatch counter. That preserves the
  same evidence split real WebGPU/wgpu execution must satisfy. A LLaMA decode
  regression also proves caller-owned host KV buffers can replace internal cache
  bindings,
  resource-bound default logits cannot be observed through a stale host slice,
  and explicit host overrides plus no-output advancement preserve the original
  resource binding.
  The LLaMA facade and C/JS/Wasm buffer-binding lanes can now carry output and
  K/V cache resource descriptors to that executable bind point, while current
  host-only backends still return unsupported. This is the resource-binding
  path WebGPU needs; it is not a claim that CPU or Metal accept arbitrary
  external resources yet.
- The C ABI LLaMA K/V cache descriptor parser now keeps resource and host cache
  binding slices pointed at the parsed descriptor storage instead of optional
  payload copies, so both K and V resources survive through synchronous
  Program/Session binding. C, Node, Bun, Node/WASI, and browser Wasm smokes now
  prove a WebGPU resource-probe LLaMA Program can compile from one compatible
  model handle, bind an independent compatible model with external-resource
  logits and K/V cache buffers, inspect the model-bound resource Session shape,
  and still reject execution without advancing position until real WebGPU
  execution lands. The C, Node, Bun, Node/WASI, and browser Wasm lanes now also
  assert those rejected public LLaMA WebGPU token steps leave caller logits
  untouched and record zero Session calls, backend ops, fallback ops, syncs, and
  runtime patch attempts, making unsupported resource probes no-fake-fallback
  evidence rather than only an error-code check.
- Session inspection now includes an opaque binding-shape hash across persistent,
  step-input, and step-output bindings. The hash records section, buffer index,
  byte range, storage kind, and external-resource placement/handle/offset/length/
  access, while deliberately excluding host pointer addresses. DeviceInference,
  LLaMA, C, Node, Bun, Node/WASI, and browser Wasm smokes prove the hash is
  nonzero for bound sessions, stable across profile reset, and identical for
  equivalent model-bound WebGPU resource sessions. This gives the future
  WebGPU/wgpu executor a stronger resource-table evidence contract without
  exposing backend internals or changing hot execution. Node/WASI and browser
  Wasm now also hash the JS-side WebGPU-like resource descriptor table before
  binding, including placement, access, handle, byte range, and same-device
  provenance, and reject resources or tables that claim mixed device tokens.
- The opt-in native WebGPU executor now exposes the first public owned-device
  buffer lane through
  `zgml_program_create_device_buffer(program, kind, ZGML_BACKEND_WEBGPU, ...)`.
  Tiny-linear C, Node, and Bun smokes allocate program-sized WebGPU
  weights/bias/input/output buffers, upload through `zgml_buffer_write`, bind
  them as external-resource Session state, execute the registered resource
  table, and prove `step_no_output`/`advance()` dispatches without incrementing
  the Session sync counter. The C ABI also exposes the first imported-buffer
  provenance path: `zgml_program_get_device_handle(...)` returns the compiled
  Program's WebGPU device token, and
  `zgml_program_import_device_buffer(...)` imports valid same-device
  `WGPUBuffer` handles after byte-size/usage validation; a zero C `byte_len`
  or omitted JS `byteLength` uses the compiled Program's required buffer size.
  Fake or wrong-device handles, unaligned storage-buffer offsets, and oversized
  views still reject before Session binding, and JS/browser `GPUBuffer` interop
  remains future work. In native `-Duse-wgpu=true` builds, the same public
  device-buffer lane now also reaches the fixed tiny LLaMA
  Program: C, Node, and Bun allocate same-device logits plus per-layer K/V
  buffers, bind them as Session output/cache resources, execute decode,
  prefill, and decode-after-prefill through the normal handle APIs, read logits
  back through `zgml_buffer_read` / `NativeBuffer.readFloat32`, and compare
  against CPU. They also import those same-device logits and K/V handles through
  the public device-buffer import API, bind the imported handles as Session
  resources, execute decode, and read the logits back through the imported
  output handle. Those same public FFI smokes now also prove a resource-bound
  no-output prompt advance leaves the logits resource untouched, records real
  backend work with zero sync/fallback, and then decodes from the GPU-resident
  K/V resources to match CPU. The same C, Node, and Bun public paths now also
  execute resource-bound `executeTokensArgmax`, `executeTokensSample`,
  `generateTokensArgmax`, and top-k=1 `generateTokensSample` against CPU
  token/logit parity by reading the bound logits resource into session-owned
  staging only when token selection needs it. The C ABI unit path plus Node/Bun
  public smokes now also fill the compiled context through that resource-bound
  no-output path,
  reject the next JS/C step as `shape_mismatch`, preserve Session position and
  logits resources, and record no extra runtime patching, dispatch, sync, or
  fallback work. The C ABI unit path also rejects resource-bound greedy and
  sampled generation after a full context before mutating output-token buffers
  or recording additional runtime work. The public Zig LLaMA facade now also covers a
  stronger experimental attention-family resource-handoff matrix: GQA
  (`n_heads = 4`, `n_kv_heads = 2`), MQA (`n_heads = 4`, `n_kv_heads = 1`),
  and MHA
  (`n_heads = n_kv_heads = 2`) shapes run through same-device resource prefill,
  untouched resource logits for no-output advancement, GPU-resident K/V reuse
  for decode, and zero sync/fallback profile evidence against CPU parity. The
  GQA and MQA public shapes are two-layer, 16-token-context, 8-token-prefill
  proofs; the MHA public shape also covers a three-layer K/V resource table,
  and the matrix now includes a tied LM head shape where the terminal logits
  projection reuses the token embedding instead of a separate output weight. The
  same Zig facade proof now also drives `generateArgmaxInto` and top-k=1
  `generateSampleInto` through the resource-bound logits output and compares the
  selected tokens/logits against CPU.
  The experimental native smoke now also includes a true multilayer MQA executor
  proof (`n_heads = 4`, `n_kv_heads = 1`) through the private native WebGPU
  resource-handoff helper, closing the gap between public facade MQA coverage
  and native dispatch-family evidence without adding another ABI model kind.
  The same public helper now also covers a reduced compiled-context envelope:
  a 16-token model compiled as an 8-token WebGPU Program binds 8-token-sized
  same-device K/V resources, pre-fills within that smaller envelope, and decodes
  from those Program-sized resources against CPU parity. A focused public
  resource-bound guard now fills that smaller envelope with no-output GPU work,
  rejects the next bound-output step plus resource-bound greedy and sampled
  generation with `SequenceTooLong`, leaves logits/generated-token outputs and
  Session position untouched, and records no extra runtime patching, dispatches,
  syncs, or fallback work.
  Native public LLaMA-family `.webgpu` execution now defaults on for
  `-Duse-wgpu=true` builds, while browser `GPUBuffer` interop and broader
  checkpoint-family coverage remain future work.
- The opt-in native WebGPU executor now has the same style of hot-path
  allocation proof as Metal for its tiny-linear configured runtime. A
  `FailingAllocator` regression compiles, binds, configures, and uploads first,
  then forces future Zig allocations/resizes to fail while it patches,
  dispatches with output download, dispatches with no output readback, and
  dispatches again. The allocator counters stay unchanged, proving the
  configured wgpu step path does not allocate Zig heap memory after Session
  binding.

## Performance Contract

This architecture is allowed only if it preserves or improves performance.

Rules:

- Program creation may allocate and analyze.
- Session binding may allocate persistent buffers and backend resources.
- Step execution must not allocate.
- Step execution must not infer shapes.
- Step execution must not scan graph ops to discover runtime slots.
- Step execution must not mix inspection metadata into the hot loop.
- Backend fallbacks must be visible in benchmark artifacts.
- Dispatch count and command shape must remain evidence-gated.

For LLaMA/SmolLM, `docs/perf-targets.md` remains the measured parity contract.
The executable-stencil plan is not a substitute for ggml/llama.cpp evidence.

## Simplicity Contract

This plan should reduce conceptual count, not add another layer for its own
sake. Simplicity means the user writes less ceremonial ML code, not that the
library has fewer useful ML capabilities.

Prefer:

- a PyTorch-like tensor/nn/optim/loss frontend for ordinary model work
- one central Program object over separate fake/real paths
- explicit Session binding over hidden global model state
- StepParams over ad hoc runtime arguments
- compact patch/update tables over per-op dynamic behavior
- small public APIs over exposed backend machinery
- compile/bind/execute as an opt-in power path for deployment, hot loops, FFI,
  and resource residency

Avoid:

- public "stencil-only inference"
- removing `Tensor`, autograd, `nn`, `optim`, losses, or training helpers just
  because they are not the execution core
- forcing Zig or JS/TS users to think in C descriptors, buffer slots, or backend
  handles for normal model code
- runtime graph walking during generation
- backend-specific concepts leaking into `zgml` root
- tokenizer/generation paths that pretend to support model families they do not
  actually support
- broad compiler abstractions before the LLaMA executable path is proven

## TS Source-Of-Truth Package Architecture

The current package shape is useful as a proof, but not the final architecture:

- `index.d.ts` used to be hand-authored declaration truth.
- `js/node.cjs` used to be hand-authored Node runtime truth.
- `js/bun.ts` used to be a separate Bun runtime/adaptor copy with embedded types.
- shared `.cjs` helper files reduce duplication but still tempt the library to
  maintain a second implementation beside the TS source.

That is the wrong long-term shape for a brilliant PyTorch-like TS library with
a native Zig execution substrate. The frontend should be written once in TS.
Generated JS, declarations, and package bundles are build output, not peer
source files to keep in sync:

```text
src/ts/core/*.ts        Tensor, shape, dtype, autograd, math
src/ts/nn/*.ts          Module, Linear, Embedding, Sequential, activations, norms
src/ts/train.ts         public train namespace and manifest
src/ts/train/*.ts       loss, optim, checkpoint, train internals
src/ts/inspection.ts    public inspection/evidence namespace and manifest
src/ts/model_source.ts  public model-source routing namespace and manifest
src/ts/native_buffer.ts public NativeBuffer policy namespace and manifest
src/ts/program_device.ts public ProgramDevice policy namespace and manifest
src/ts/runtime/*.ts     Program, Session, StepParams, evidence records
src/ts/adapters/node.ts Node FFI/dlopen/filesystem adapter only
src/ts/adapters/bun.ts  Bun FFI adapter only
src/ts/adapters/web.ts  browser/wasm/wgpu adapter only
```

Migration priority is now whole-frontend ownership, not endless helper
extraction and not TS/Zig frontend parity. The next useful unit is not "move
one predicate from CJS to TS" or "mirror one TS API in Zig"; it is "make the
package spine TS-authored and emitted":

```text
src/ts/index.ts           public frontend namespace and exports
src/ts/node.ts            Node entrypoint that imports shared TS frontend
src/ts/bun.ts             Bun entrypoint that imports shared TS frontend
src/ts/adapters/native.ts narrow host callback interfaces for native execution
src/ts/adapters/node.ts   koffi/path/filesystem loading and ABI packing
src/ts/adapters/bun.ts    bun:ffi loading and ABI packing
```

The artifact ladder should be:

```text
src/ts/**     authored implementation and public types
dist/**       tsdown-emitted package entrypoints, declarations, maps
js/*.cjs      removed; package/source-checkout runtime should use TS-built dist artifacts
js/generated  removed; source-checkout smokes use TS-built dist artifacts
```

The architectural rule is:

```text
TS owns frontend behavior and public types.
Adapters own only host I/O, FFI packing, and native handle lifetimes.
Zig owns hot kernels, executable programs, and native backend execution.
```

So `tsdown` or an equivalent bundler is the right next package move now that
the entrypoint shape exists. The first bundler pass should emit from
`src/ts/index.ts`, `src/ts/node.ts`, and `src/ts/bun.ts`; it should not merely
wrap the current CJS facades.
The final package should be installable from `dist/**` plus the native library;
checked-in generated JS is no longer part of the source-of-truth story.
`js/generated/**` has been removed, and the TS architecture guard rejects it
if it grows back. The package/type/smoke gates prove the `src/ts/** -> tsdown ->
dist/**` path instead of proving checkout artifact provenance.
The package builder now sorts TS entries by path depth before batching, so
root public facades such as `zgml/inspection`, `zgml/model_source`,
`zgml/native_buffer`, and `zgml/program_device` keep stable
`dist/<subpath>.d.cts` declaration targets instead of losing their names to
same-basename internal chunks.
Friendly package subpaths are product contracts, not synchronization shims.
`zgml/inspection`, `zgml/model_source`, `zgml/native_buffer`,
`zgml/program_device`, and `zgml/step_params` own their public TS manifests in
their top-level `src/ts/*.ts` files and explicitly choose the runtime helpers
they expose. The matching `zgml/runtime/*` subpaths keep runtime-owned
manifests for low-level evidence and backend policy. That is the desired
pattern for TS-first package work: write the public surface once in TS, keep the
runtime boundary named, and let `tsdown` emit the package artifacts.
The npm artifact now enforces that boundary: `package.json` no longer includes
the local `js/**` tree, and `scripts/check_package_artifact.cjs` rejects
packaged `js/generated/**`, local smoke harness files, public package scripts
that route back through repo-only `js/` validation, and a public declaration for
the internal concrete Node FFI adapter. Shipping consumers get `dist/**`,
including `dist/smokes/dist_package_smoke.cjs`,
`dist/smokes/frontend_package_smoke.cjs`,
`dist/smokes/node_package_smoke.cjs`, and
`dist/smokes/bun_package_smoke.cjs`, plus the native/source files required to
build the package artifacts.
That pass has started: `npm run build:package` now calls `tsdown` directly;
`scripts/package_metadata_policy.cjs` derives the package entry graph,
declaration exclusions, npm subpaths, and package file boundary from
`src/ts/**`; `scripts/apply_package_metadata_policy.cjs` is the CLI that applies
that policy to `package.json`, and `tsdown.config.mjs` consumes the same policy
to emit the
TS-authored package spine into `dist/**` as CJS, declarations, and source maps.
The Node concrete runtime is now part of that package artifact too:
`src/ts/adapters/node_ffi_runtime.ts` emits to
`dist/adapters/node_ffi_runtime.cjs`, while its internal adapter entry is
excluded from public declaration bundling because the public Node API is exposed
through `dist/node.d.cts`. The shared
`src/ts/adapters/concrete_runtime_loader.ts` owns native contract assertion,
candidate fallback execution, and host runtime manifest construction; Node and
Bun concrete loader modules now provide only host policy and reuse that shared
Interface. `npm run typecheck:dist` verifies the generated declarations, and
`npm run smoke:dist` now executes the TS-authored
`src/ts/smokes/dist_package_smoke.ts` artifact from `dist/smokes/**` to verify
the bundled runtime entrypoints. `zgml/frontend`,
`zgml/node` runtime, the top-level friendly TS subpaths, adapter evidence, and
deep wildcard package subpaths now resolve to `dist/**` CJS artifacts, so the
TS-bundled frontend surface and Node native bridge are package-facing. The
frontend package smoke also lives at
`src/ts/smokes/frontend_package_smoke.ts` and runs from
`dist/smokes/frontend_package_smoke.cjs`, so package-subpath proof follows the
same TS-owned `tsdown` path instead of a repo-only `js/` harness. The package
root now resolves directly to that TS-built Node bridge through `dist/node.cjs`;
`index.js` is no longer a shipped root runtime bridge.
The Node package export now resolves through TS-built `dist/node.cjs`, whose
native bridge loads local sibling/package-built `tsdown` concrete runtime
artifacts such as `dist/adapters/node_ffi_runtime.cjs`; it no longer falls back
through `js/generated/**`, and the obsolete `js/node.cjs` compatibility shim has
been removed.
The Bun package export
now uses the Bun condition to resolve to TS-built
`dist/bun_native.cjs`, which loads the concrete
`src/ts/adapters/bun_ffi_runtime.ts` FFI adapter behind the same TS native API
contract while Node can still load native-free `dist/bun.cjs` for package-spine
smokes. `src/ts/adapters/native.ts` owns native-library extension/path/load-info
policy and missing-library diagnostics; `src/ts/adapters/bun_host_runtime.ts`
owns the Bun host shell that supplies import-meta/path/existence hooks.
`src/ts/runtime/native_status.ts` owns shared native status error formatting/check
policy; `src/ts/adapters/bun_status.ts` owns Bun native handle extraction
policy. `src/ts/adapters/bun_inspection_ops.ts`
owns Bun native inspection/compile descriptor/runtime-profile ABI policy.
`src/ts/adapters/bun_session_ops.ts` owns Bun generic Session upload ABI
policy, while `src/ts/runtime/session_step_io.ts` owns shared Session step I/O
descriptor/result policy. `src/ts/adapters/bun_llama_token_ops.ts` owns Bun LLaMA
token-step/execute/selection ABI policy. `src/ts/adapters/bun_program_buffer_ops.ts`
owns Bun Program buffer/device ABI policy.
`src/ts/adapters/bun_module_program_ops.ts` owns Bun module Program compile ABI
policy.
`src/ts/adapters/bun_program_bind_ops.ts` owns Bun generic Program bind ABI
policy.
`src/ts/adapters/bun_llama_session_bind_ops.ts` owns Bun LLaMA
KV-cache/session bind ABI packing. `src/ts/runtime/llama_session_bind_desc.ts`
owns shared LLaMA Session bind output/KV-cache validation and descriptor fields.
`src/ts/runtime/llama_kv_cache.ts` owns shared LLaMA KV-cache allocation/resource policy, with Bun adapter glue in
`src/ts/adapters/bun_llama_kv_cache_ops.ts`.
`src/ts/adapters/bun_llama_kv_cache_surface.ts` owns the Bun LLaMA
KV-cache class lifecycle.
`src/ts/adapters/native_buffer_surface.ts` owns the shared Adapter
`NativeBuffer` public surface for Node and Bun. `src/ts/runtime/generic_family_surface.ts` owns the shared
generic model/program/session public surface, while
`src/ts/adapters/generic_family_model_handle_surface.ts` owns shared
generic model-create/read-handle adapter policy and
`src/ts/adapters/bun_generic_family_surface.ts` owns Bun descriptor adapter
glue. `src/ts/adapters/bun_model_source_ops.ts`
owns Bun model path/safetensors load-probe ABI policy and delegates shared
supported-checkpoint catalog count validation/iteration to
`src/ts/runtime/model_source_catalog.ts`.
The legacy `js/bun.ts` compatibility shim has been removed, so Bun no longer
has a separate authored runtime copy outside `src/ts`.
`zgml/node` and `zgml/bun` now publish their own tiny declaration wrappers over
the TS-owned public surface, so the host subpaths no longer depend on root
declaration fallback. Checked Node host runtime loading now lives in
`src/ts/adapters/node_host_runtime.ts`, shared native-library load-info policy
now lives in `src/ts/adapters/native.ts`, checked ABI struct registration now
lives in `src/ts/adapters/node_abi_structs.ts`, shared native status error
formatting/check policy now lives in `src/ts/runtime/native_status.ts`, checked Node
handle extraction policy now lives in `src/ts/adapters/node_status.ts`, and
checked generic TinyLinear/TinyMlp/Program/Session public class surface now
lives once in `src/ts/runtime/generic_family_surface.ts`, while
`src/ts/adapters/generic_family_model_handle_surface.ts` owns shared
generic model-create/read-handle adapter policy and
`src/ts/adapters/node_generic_family_surface.ts` owns Node descriptor adapter
glue,
checked shared Adapter NativeBuffer public class surface now lives in
`src/ts/adapters/native_buffer_surface.ts`, checked shared Adapter Tensor public
class surface now lives in `src/ts/adapters/tensor_surface.ts`, checked
shared Tensor root helper/coercion policy now lives in
`src/ts/core/tensor_root.ts`
with no adapter compatibility re-export path, checked shared Tensor
inspection/metadata policy now lives in `src/ts/core/tensor_metadata.ts`
with no adapter compatibility re-export path, and
checked LLaMA token-step/execute/selection ABI calls now live in
`src/ts/adapters/node_llama_token_ops.ts`, checked shared LLaMA
model/program/session public class surface now lives in
`src/ts/runtime/llama_family_surface.ts`, with no adapter compatibility
re-export path, checked LLaMA KV-cache/session bind
ABI packing now lives in `src/ts/adapters/node_llama_session_bind_ops.ts`,
shared LLaMA Session bind descriptor policy now lives in
`src/ts/runtime/llama_session_bind_desc.ts`, and checked shared LLaMA KV-cache
allocation/resource policy now lives in
`src/ts/runtime/llama_kv_cache.ts`, with Node object/program-device adapter
glue in `src/ts/adapters/node_llama_kv_cache_ops.ts`,
checked Program buffer/device ABI policy now lives in
`src/ts/adapters/node_program_buffer_ops.ts`,
checked generic Program bind ABI policy now lives in
`src/ts/adapters/node_program_bind_ops.ts`,
checked generic Session upload ABI policy now lives in
`src/ts/adapters/node_session_ops.ts`, with shared Session step I/O
descriptor/result policy in `src/ts/runtime/session_step_io.ts`,
checked module-program compile and bind descriptor ABI policy now lives in
`src/ts/adapters/node_module_program_ops.ts`,
checked model path/safetensors load-probe ABI policy now lives in
`src/ts/adapters/node_model_source_ops.ts`, with shared supported-checkpoint
catalog count validation/iteration in `src/ts/runtime/model_source_catalog.ts`,
checked native inspection/compile descriptor/runtime-profile ABI calls now live
in `src/ts/adapters/node_inspection_ops.ts`, with shared compile-envelope
defaults/projection coming from `src/ts/runtime/abi.ts`,
while checked native symbol binding
now lives in `src/ts/adapters/node_symbols.ts`, so generating that public
declaration surface from implementation types and continuing to shrink the
temporary unchecked Node FFI body into typed TS modules are the remaining
package-surface cleanups.

This spine has started: `src/ts/index.ts` now re-exports the TS-authored
frontend/runtime modules behind one frontend manifest, generated
`src/ts/node.ts`, generated native-free `src/ts/bun.ts`, and generated
Bun-native `src/ts/bun_native.ts` re-export that shared frontend, and
`src/ts/adapters/native.ts` defines adapter evidence that makes the intended
ownership explicit: TS owns frontend policy; host adapters own native loading.
`src/ts/adapters/native.ts` also owns native-library extension, filename/path
resolution, shared load-info construction, and missing-library diagnostics, while `src/ts/adapters/node.ts`
and `src/ts/adapters/bun.ts` own the generated host adapter evidence and
host-specific adapter exports. The concrete Node and Bun adapters now call that
generated policy instead of carrying duplicate resolver logic. The native
adapter helper also owns the source-tree `koffi` fallback path used by the Node
adapter, keeping the actual `require()` host-specific while moving path policy into TS. `npm run
test:ts-source` proves the `tsdown` dist frontend, friendly subpath, adapter
evidence, and wildcard package artifacts plus the transitional generated
`dist/node.cjs` and `dist/bun.cjs` adapter-spine artifacts. The package also exposes
`zgml/frontend` as a native-free TS-authored subpath backed by the `dist`
frontend artifact, with runtime and type smokes proving consumers can import
shared frontend/runtime policy without loading the Node FFI wrapper. Generated
module subpaths are now package exports too: `zgml/tensor`, `zgml/nn`,
`zgml/compile`, `zgml/inspection`, `zgml/model_source`, `zgml/program`, `zgml/session`, `zgml/step_params`, `zgml/checkpoint`,
`zgml/loss`, `zgml/optim`, `zgml/train`, `zgml/native_buffer`, `zgml/program_device`, `zgml/adapters/*`, `zgml/core/*`, `zgml/runtime/*`,
`zgml/nn/*`, and `zgml/train/*` resolve to TS-authored dist artifacts, so consumers can
import tensor, nn, compile, inspection evidence, model-source routing, Program, Session, StepParams, NativeBuffer/ProgramDevice policy, checkpoint, loss, optimizer, train,
shape/runtime/module, and training
building blocks without reaching into `js/generated/**` or loading host FFI
glue. Adapter evidence is exposed the same way through `zgml/adapters/native`,
`zgml/adapters/node`, and `zgml/adapters/bun`, so host ownership/path policy can
be imported separately from the concrete FFI wrappers. `zgml/node` now resolves
to the TS-built `dist/node.cjs` entrypoint; the root package entrypoint prefers
that same artifact when present. The bridge loads the package-emitted
`dist/adapters/node_ffi_runtime.cjs` through the typed
`src/ts/adapters/node_concrete_runtime.ts` loader, which owns the native API
contract check and no longer falls back through `js/generated/**`, then re-exports the native
Tensor/Program/Session/model surface with identity parity. The implementation
source is `src/ts/adapters/node_ffi_runtime.ts`, and its checked Node host
runtime shell lives in `src/ts/adapters/node_host_runtime.ts`, while shared
native-library load-info policy lives in `src/ts/adapters/native.ts`; checked ABI struct
registration lives in `src/ts/adapters/node_abi_structs.ts`, shared native
status error formatting/check policy lives in `src/ts/runtime/native_status.ts`,
checked Node handle extraction policy lives in
`src/ts/adapters/node_status.ts`, checked shared Tensor root
construction/factory/helper/coercion policy lives in
`src/ts/core/tensor_root.ts` and is covered by the TS source smoke,
checked shared Tensor inspection/metadata policy lives in
`src/ts/core/tensor_metadata.ts` and is covered by the TS source smoke,
checked shared Node/Bun Tensor grad-state policy lives in
`src/ts/core/tensor_grad_state.ts` and is covered by the TS source smoke,
core Tensor flexible indexing/assignment policy lives in
`src/ts/core/tensor_index.ts` and is covered by the TS source smoke,
basic Tensor length/rank/copy/iterator surface lives in
`src/ts/core/tensor_core.ts` and is guarded against host-adapter drift,
Tensor alias spellings now delegate to checked core/index/metadata helpers
instead of host-local `this.*` wrappers,
Tensor static class surface policy now delegates through
`src/ts/core/tensor_static_surface.ts`, keeping factory, JSON/native-buffer,
root math, and join statics shared while Node/Bun preserve their host-specific
overload syntax,
Tensor instance math methods now delegate through
`src/ts/core/tensor_math.ts` via a checked surface helper, keeping eager math
implementation and PyTorch-like method forwarding in one TS-owned place,
Tensor view/shape methods now delegate through
`src/ts/core/tensor_view.ts` via a checked surface helper, keeping clone/detach,
reshape/view, broadcast/expand, slicing, and `T` forwarding in one TS-owned place,
checked LLaMA token-step/execute/selection ABI calls live in
`src/ts/adapters/node_llama_token_ops.ts`, checked LLaMA
KV-cache/session bind ABI packing lives in
`src/ts/adapters/node_llama_session_bind_ops.ts`, shared LLaMA Session bind
descriptor policy lives in `src/ts/runtime/llama_session_bind_desc.ts`, and
LLaMA KV-cache allocation/resource policy lives in
`src/ts/runtime/llama_kv_cache.ts` plus Node adapter glue in
`src/ts/adapters/node_llama_kv_cache_ops.ts`,
checked Program buffer/device ABI policy lives in
`src/ts/adapters/node_program_buffer_ops.ts`,
checked generic Program bind ABI policy lives in
`src/ts/adapters/node_program_bind_ops.ts`,
checked generic Session upload ABI policy lives in
`src/ts/adapters/node_session_ops.ts`, with shared Session step I/O
descriptor/result policy in `src/ts/runtime/session_step_io.ts`,
checked module-program compile and bind descriptor ABI policy lives in
`src/ts/adapters/node_module_program_ops.ts`, checked model path/safetensors
load-probe ABI policy lives in `src/ts/adapters/node_model_source_ops.ts`,
shared supported-checkpoint catalog count validation/iteration lives in
`src/ts/runtime/model_source_catalog.ts`, checked native inspection/compile
descriptor/runtime-profile ABI calls live in
`src/ts/adapters/node_inspection_ops.ts`, and native symbol binding lives in
`src/ts/adapters/node_symbols.ts`, and the PyTorch-compatible `torch` namespace
assembly now lives in typed `src/ts/adapters/frontend_namespace_surface.ts`
instead of being hand-copied by Node and Bun concrete runtimes; its remaining
cleanup is replacing the temporary unchecked CommonJS body with typed TS
modules. `zgml/bun` now
resolves to the TS-built Bun native bridge under Bun; that bridge loads
`src/ts/adapters/bun_ffi_runtime.ts`, checks it against the same contract, and
re-exports the native surface with identity parity.
`examples/types/package-spine-smoke.ts` now imports the dist-backed
`zgml/frontend`, friendly subpath, adapter evidence, and wildcard package
declarations directly instead of reaching into `../../js/generated/**`, so
`npm run typecheck` exercises the package declaration spine instead of only the
root public declaration. `scripts/check_ts_source_architecture.cjs` rejects public type
examples that import repo-only `js/` artifacts.
`scripts/package_metadata_policy.cjs` now treats `src/ts/**` as the
package-entry source of truth, and `tsdown.config.mjs` consumes its entry map
instead of walking the tree independently, so adding a TS
frontend/runtime/adapter module is a source-tree operation rather than an
npm-script or bundler-policy edit. `scripts/check_ts_source_architecture.cjs`
now typechecks `src/ts/**` with `tsc --noEmit` and enforces architecture guards
without byte-comparing against `js/generated`; `dist/**` is the runtime artifact
contract. The source
smoke has started consuming `dist/**` directly through the TS-authored
`src/ts/smokes/ts_source_smoke.ts`, emitted by `tsdown` as
`dist/smokes/ts_source_smoke.cjs`; generated checkout JS has been removed
instead of kept as a compatibility lane. The same gate
currently checks that `package.json` keeps the TS-authored frontend,
adapter evidence, `tensor`/`nn`/`program`/`session`/`step_params`/`checkpoint`/`loss`/`optim`, and `core`/`runtime`/`nn`/`train` module subpaths
package-facing. `npm run test:ts-source` runs that check after the TS-source
smoke, so source type errors or package-surface drift fail package
validation.
`src/ts/runtime/native_api_contract.ts` now owns the native bridge contract
contract: the required PyTorch-like eager Tensor/native API exports, the
required TS package-spine exports, and comparison helpers used by
`src/ts/smokes/native_bridge_contract_smoke.ts`, emitted by `tsdown` as
`dist/smokes/native_bridge_contract_smoke.cjs`. That smoke pins package/native contract identity: live
`zgml`, the legacy Node adapter, `dist/node.cjs`, and `zgml/node` must expose
the same concrete native API, with root and `zgml/node` resolving through the
TS-built bridge after `npm run build:package`. `zgml/runtime/native_api_contract` is
package-facing through the wildcard runtime export, so future migration slices
can update the contract in TS and prove it through source, dist, and
package-subpath smokes. This keeps future export-map flips honest: native bridge
cannot move to `dist/**` by accident unless `Tensor`, `nn`, `Program`,
`Session`, model loading, runtime info, and native resource helpers are present
and identity-checked against the concrete native adapter. The generated
package-spine contract also declares its generator, TS-contract origin, and
`handwrittenRootExportList: false`, so the root export list is fan-out evidence
rather than a second source to keep synchronized.
The Node and Bun package smokes also consume the same TS contract, so adapter
drift such as a missing model alias fails in the host smoke that would publish
the broken surface.

The package build should then emit:

```text
dist/index.cjs       common frontend runtime
dist/index.d.cts     emitted declarations
dist/node.cjs        Node adapter
dist/bun.cjs         native-free Bun adapter evidence
dist/bun_native.cjs  Bun native bridge
dist/**/*.cjs        exported subpath artifacts
dist/**/*.d.cts      exported subpath declarations
```

Migration slices:

1. Introduce `src/ts/` and move the pure shared helpers first
   (shape algebra, tensor data normalization, inspection freezing, token
   sampling helpers). Keep current package exports green. This has started:
   `src/ts/core/shape.ts` is now the strict TS source of truth for shape algebra,
   `tsconfig.source.json` typechecks authored TS without emitting checkout JS,
   and `npm run test:ts-source` runs the tsdown-emitted
   `dist/smokes/ts_source_smoke.cjs` to prove package-artifact helper behavior. The first
   Session tensor shape/readback policy has moved into
   `src/ts/runtime/session_tensor.ts`, which imports `normalizeFactoryShape`
   from the TS-authored shape helper, and Tensor Program IR evidence construction
   has moved to `src/ts/runtime/tensor_program_ir.ts`, emitted to
   `dist/runtime/tensor_program_ir.cjs` and importing
   `shapeScalarCount` and `rowMajorStrides` from TS-authored shape helpers.
   Tensor placement has moved to `src/ts/runtime/tensor_placement.ts`, emitted
   to `dist/runtime/tensor_placement.cjs`, and uses
   `sameShape` and `createShapedF32Helpers` in tensor placement and Program bind validation paths, so TS-owned helpers cover
   both readback shape checks and binding shape checks. `src/ts/runtime/kernel_plan.ts`
   now owns KernelPlan shape constraints, slice/narrow/select metadata,
   memory-layout evidence, public projection, and native-op scheduling on the TS
   source path.
   `src/ts/runtime/session_facade_composition.ts` and
   `src/ts/runtime/session_step_io.ts` now use TS-authored shape helpers for
   StepParams input/output shape evidence, keeping the central
   Program -> Session -> StepParams validation path on the TS source rail.
   Module compiler policy now lives in
   `src/ts/runtime/module_compiler_policy.ts`, emitted to
   `dist/runtime/module_compiler_policy.cjs`; the obsolete
   `js/shared_module_compiler.cjs` shim has been removed.
   `src/ts/shared_frontend.ts` now re-exports the TS-authored shape and eager
   tensor helpers, so adapter startup and public helper wiring no longer
   depend on hand-written CJS shape/tensor re-export modules. The hand-written
   `js/shared_shape.cjs` and obsolete `js/shared_tensor_ops.cjs` shims have been
   removed; the TS-source smoke now checks emitted package shape behavior and
   eager Tensor policy directly. `src/ts/core/tensor_data.ts`,
   `src/ts/core/tensor_core.ts`, `src/ts/core/tensor_factory.ts`,
   `src/ts/core/tensor_facade.ts`, `src/ts/core/tensor_join.ts`,
   `src/ts/core/tensor_view.ts`, `src/ts/core/tensor_math.ts`,
   `src/ts/core/tensor_index.ts`, `src/ts/core/activation.ts`, and
   `src/ts/core/grad_mode.ts` own the corresponding eager Tensor, activation,
   and grad-mode helpers emitted under `dist/core/**`.
   `src/ts/core/token.ts` now owns token validation, token-window normalization,
   sample-option normalization, generated-token output validation, and ABI
   token selection/generation result shaping; `src/ts/shared_frontend.ts`,
   TS session/runtime facades, and package smokes import the TS-owned helper
   through tsdown-emitted artifacts, and the hand-written `js/shared_token.cjs`
   module has been removed.
2. Move `Tensor` and eager tensor ops into TS, emit declarations from the
   implementation, and delete matching hand-written declaration sections.
3. Move `nn`, `loss`, `optim`, `train`, and checkpoint helpers into TS. Keep
   Node/Bun adapters as FFI/resource shims rather than frontend copies. This has
   started for `nn`: `src/ts/nn/shape_module.ts` now owns the parameterless shape
   module factory for reshape/view/flatten/squeeze/unsqueeze/transpose/permute/broadcast/expand/narrow/select/slice,
   with `src/ts/shared_frontend.ts` injecting the placement hook for
   that constructor. `permute` is trace/Tensor Program IR visible; rank-2/rank-3
   identity/swap permutations and rank-3 cycle permutations
   lower through existing reshape/transpose native kernels, with a cycle
   represented as a short transpose chain instead of a new ABI op. `src/ts/nn/parameterless_modules.ts` now owns stateless
   activation, softmax/logSoftmax, reduction, and Dropout module factories the same way.
   `src/ts/nn/linear_module.ts` now owns the trainable `nn.Linear` factory,
   including tiny-linear compile evidence and device-program fallback delegation
   through `src/ts/nn/sequential_program_compile.ts`.
   `src/ts/nn/embedding_module.ts` now owns the trainable `nn.Embedding` factory,
   including indexed gather gradients and input-shape compile diagnostics.
	   `src/ts/nn/sequential_module.ts` now owns the compositional `nn.Sequential`
	   factory, including traversal, child training propagation, and compile/bind
		   delegation through the same sequential Program compile helper.
		   `src/ts/nn/feature_norm_module.ts` now owns the trainable
		   feature-axis `nn.LayerNorm` / `nn.RMSNorm` factory, including forward math,
		   Tensor autograd, affine/no-affine state, binding/placement, and compile
		   delegation. `src/ts/nn/single_module_compile.ts` now owns the shared
		   generic single-module compile/bind/place surface used by Shape,
		   stateless modules, Embedding, and feature-axis norms, so adding another
		   simple module does not add another local Program compile method copy.
		   `src/ts/nn/namespace.ts` now owns the shared PyTorch-style `nn`
		   namespace constructor table and helper facade aliases, while
		   `src/ts/nn/namespace_methods.ts` owns the shared module prototype
		   compiler-inspection aliases and PyTorch-style snake-case methods.
4. Move Program/Session JS facades into TS while keeping Zig/C ABI structs as
   the native boundary. Evidence records should be declared by implementation
   types, not duplicated in `index.d.ts`. This has started for Program policy:
	   `src/ts/runtime/program_facade_policy.ts` now owns requirements projection,
	   evidence views, compatibility/capability predicates, runtime-profile access,
		   and lifecycle free/dispose policy. `src/ts/runtime/program_buffer_factory.ts`
		   now owns Program-owned host/device/resource buffer factory routing and
			   LLaMA KV-cache buffer routing. `src/ts/runtime/program_module_binding.ts`
			   now owns `Program.bindModule(...)` compatibility preflight, parameter
			   placement, owned-buffer deduplication, and cleanup-on-failure policy.
			   `src/ts/runtime/program_device.ts` now owns `ProgramDevice` placement,
			   frozen info evidence, same-device imports, role-specific device buffers, and KV-cache routing.
			   `src/ts/runtime/program_sizing.ts` now owns Program buffer sizing,
			   byte-length fallback policy, frozen sizing snapshots, and public sizing
			   signatures plus the shared sizing accessor table used by Program
			   facades.
			   `src/ts/runtime/program_parameters.ts` now owns Generic Program
			   parameter name/info projection, accessor-table routing, and validation.
			   `src/ts/runtime/program_shapes.ts` now owns Generic Program
			   compiled-shape preference, descriptor shape fallback, LLaMA
			   vocab-sized output shape projection, and shape accessor-table routing.
			   `src/ts/runtime/program_resources.ts` now owns Program create-buffer,
			   role-buffer, device-handle, device wrapper, and import accessors.
			   `src/ts/runtime/program_layout.ts` now owns Program buffer-layout
			   caching, slot-name/slot lookup, and LLaMA KV-cache layout accessors.
			   `src/ts/runtime/program_policy_accessors.ts` now owns Generic and
			   LLaMA-family Program object-method to typed-policy forwarding for
			   compile evidence, compiler artifacts, compatibility, capabilities,
			   cached LLaMA inspection, KV-cache creation, executable inspection,
			   bind handoff, runtime profiles, lifecycle, and `bindModule(...)`.
			   `src/ts/runtime/program_facade.ts` now owns the generic and
			   LLaMA-family Program facade factory assembly; the obsolete
			   `js/shared_program_facade.cjs` re-export shim has been removed.
			   `src/ts/runtime/step_params.ts` now owns shared StepParams
			   compatibility-result construction, diagnostics, validation-error
			   metadata, ownership/effect classification, hot-path signature
			   policy, object validation, output-buffer checks, token-window fit
			   checks, no-inline-output guards, token scratch-option helpers, and
			   compatibility predicate/signature matching;
			   generic and LLaMA-family Session facades consume that TS-authored
			   policy through the tsdown-emitted runtime artifact.
			   `src/ts/runtime/session_facade.ts` now owns the shared
			   Session-level StepParams facade predicates and signature-match
			   methods, live-session handle/inspection/position accessors,
			   Session layout/shape/size accessors, live-guarded Session
			   call-profile facade wrappers, and
			   Session buffer-sizing facade assembly/signature matching plus
			   runtime-profile read/match/reset wrappers and generic Session
			   parameter inspection/upload facade methods,
			   generic Session bound host/native input-output interpretation,
			   explicit input/output preparation, core step/advance execution,
			   StepParams compatibility,
			   readOutputInto/readOutputTensor facade wrappers, generic
			   Session step/stepTensor/stepInto and
			   execute/executeTensor/executeInto/advance
			   orchestration, LLaMA executeTokens/execute/executeTensor/
			   executeInto/advanceTokens orchestration, LLaMA step/stepTensor/stepInto/advance
			   orchestration, LLaMA prefill/prefillTensor/prefillInto
			   orchestration, LLaMA argmax/sample token selection and native
			   token-window generation orchestration, LLaMA scalar-token and
			   token-window scratch facade helpers, plus reset/free/dispose
			   lifecycle facade wrappers;
			   `js/shared_session_facade.cjs` supplies only the live contract,
			   compatibility callbacks, sizing callbacks, native runtime-profile
			   callbacks, native parameter-upload callbacks, resource-cleanup
			   callbacks, and adapter liveness check.
			   `src/ts/runtime/session_contract.ts` now owns shared Session
			   buffer-sizing signatures and generic/LLaMA step-contract record
			   construction; the Session facade imports the TS-authored helper after
			   reading live adapter state.
			   `src/ts/runtime/session_profile.ts` now owns frontend Session
			   call-profile counters, signatures, signature matching, and reset
			   policy for both generic and LLaMA-family Session facades.
			   `src/ts/runtime/session_parameters.ts` now owns generic Session
			   parameter metadata projection, upload-by-name validation, and
			   persistent binding-index selection; the Session facade keeps only
			   the native upload call and profile bump.
			   `src/ts/runtime/session_layout.ts` now owns generic/LLaMA Session
			   slot lookup, KV-cache layout presence checks, shape/length/byte
			   projection, bound-buffer kind classification, LLaMA default buffer
			   layout construction, Session scratch allocation, and
			   NativeBuffer-output binding checks plus host/native bound
			   input/output selectors.
			   `src/ts/runtime/session_binding.ts` now owns LLaMA-family
			   `Program.bind(...)` route selection, output normalization,
			   inspection/layout handoff, and owned-output cleanup; the adapter
			   supplies only native callbacks and concrete Session construction.
			   `src/ts/runtime/session_lifecycle.ts` now owns LLaMA and generic
			   Session cleanup/owned-resource release policy; the adapter supplies
			   the native handle-free callback and null-handle sentinel.
			   `src/ts/runtime/session_values.ts` now owns generic Session
			   explicit input/output preparation, execute/executeTensor/executeInto
			   plan routing, no-output result validation, step output
			   target/readback routing, read-output tensor target planning, and
			   LLaMA execute/executeTensor/executeInto method planning, LLaMA
			   stepTensor output validation, LLaMA execute-token output/readback
			   routing, LLaMA read-output tensor target planning, and generic plus
			   LLaMA StepParams compatibility construction plus LLaMA
			   prefillTensor/generation/sample, token-selection/generation
			   token-window planning, and native-output preconditions for token
			   selection/generation; the adapter supplies
			   host-value preparation callbacks and keeps native execution
			   orchestration.
5. Create the TS package spine: `src/ts/index.ts`, `src/ts/node.ts`,
   `src/ts/bun.ts`, and narrow adapter callback interfaces. This is the move
   that makes Node/Bun share one frontend implementation instead of sharing
   fragments through CJS. This has started with generated and smoke-tested TS
   package bridges for Node and Bun. The concrete Node FFI runtime now lives at
   `src/ts/adapters/node_ffi_runtime.ts`, emits package artifacts such as
   `dist/adapters/node_ffi_runtime.cjs`, and is loaded by `dist/node.cjs`
   without falling through to `js/generated/**` or a legacy root Node shim. The obsolete `js/node.cjs` compatibility entrypoint has been
   removed. The TS-authored native bridge contract smoke now proves root,
   `dist/node.cjs`, and `zgml/node` bridge directly to the `tsdown`-emitted
   concrete runtime.
   `src/ts/adapters/concrete_runtime_loader.ts` owns the shared typed concrete
   runtime contract assertion, fallback loop, and diagnostics; `src/ts/adapters/node_concrete_runtime.ts`
   now owns only the Node load order, bridge label, and manifest before
   `dist/node.cjs` re-exports the checked surface.
   The concrete runtime delegates checked host
   loading shell policy to `src/ts/adapters/node_host_runtime.ts`, delegates
   shared native-library load-info construction to `src/ts/adapters/native.ts`, delegates
   checked ABI struct registration to `src/ts/adapters/node_abi_structs.ts`,
   delegates shared native status error formatting/check policy to
   `src/ts/runtime/native_status.ts`, delegates checked Node handle extraction policy
   to `src/ts/adapters/node_status.ts`, imports the TS-owned WebGPU interop symbol
   table from `src/ts/adapters/webgpu_interop.ts`, imports shared Node/Bun
   adapter ABI model-kind and buffer-storage aliases from
   `src/ts/runtime/native_abi_constants.ts` for shared native ABI model-kind
   and buffer-storage alias policy,
   delegates checked generic TinyLinear/TinyMlp/Program/Session public class
   surface to `src/ts/runtime/generic_family_surface.ts` through
   `src/ts/adapters/node_generic_family_surface.ts`,
   delegates checked shared Adapter NativeBuffer public class surface to
   `src/ts/adapters/native_buffer_surface.ts`,
   delegates checked shared Tensor root construction/factory/helper/coercion
   policy to `src/ts/core/tensor_root.ts`,
   delegates checked shared Tensor inspection/metadata policy to
   `src/ts/core/tensor_metadata.ts`,
   delegates checked shared Node/Bun Tensor grad-state policy to
   `src/ts/core/tensor_grad_state.ts`,
   delegates checked shared Node/Bun Tensor static class surface policy to
   `src/ts/core/tensor_static_surface.ts`,
   delegates checked shared Node/Bun Tensor instance math surface policy to
   `src/ts/core/tensor_math.ts`,
   delegates checked shared Node/Bun Tensor view/shape surface policy to
   `src/ts/core/tensor_view.ts`,
   delegates checked LLaMA token-step/execute/selection ABI calls to
   `src/ts/adapters/node_llama_token_ops.ts`,
   delegates checked shared LLaMA model/program/session public class surface to
   `src/ts/runtime/llama_family_surface.ts`, with no adapter compatibility
   re-export path,
   delegates checked LLaMA KV-cache/session bind ABI packing to
   `src/ts/adapters/node_llama_session_bind_ops.ts`,
   delegates checked shared LLaMA KV-cache allocation/resource policy to
   `src/ts/runtime/llama_kv_cache.ts`, with Node adapter glue in
   `src/ts/adapters/node_llama_kv_cache_ops.ts`,
   delegates checked Program buffer/device ABI policy to
   `src/ts/adapters/node_program_buffer_ops.ts`,
   delegates checked generic Program bind ABI policy to
   `src/ts/adapters/node_program_bind_ops.ts`,
   delegates checked generic Session upload ABI policy to
   `src/ts/adapters/node_session_ops.ts` and shared Session step I/O
   descriptor/result policy to `src/ts/runtime/session_step_io.ts`,
   delegates checked module-program compile and bind descriptor ABI policy to
   `src/ts/adapters/node_module_program_ops.ts`,
   delegates checked model path/safetensors load-probe and
   supported-checkpoint catalog policy to
   `src/ts/adapters/node_model_source_ops.ts`,
   delegates checked native inspection/compile descriptor/runtime-profile ABI
   calls to `src/ts/adapters/node_inspection_ops.ts`, with shared
   compile-envelope defaults/projection in `src/ts/runtime/abi.ts`,
   delegates checked native symbol binding to `src/ts/adapters/node_symbols.ts`; the concrete Bun FFI runtime lives at
   `src/ts/adapters/bun_ffi_runtime.ts`, selected by
   `src/ts/adapters/bun_concrete_runtime.ts` instead of the legacy `js/bun.ts`
   compatibility shim. The Bun loader delegates the same native API contract
   assertion and fallback diagnostics to `src/ts/adapters/concrete_runtime_loader.ts`,
   keeping host-specific behavior to candidate selection and bridge labels. Bun
   ABI word-writing helpers live in
   `src/ts/adapters/bun_abi_words.ts`, so descriptor-packing modules share one
   checked pointer/size slot writer. Bun host runtime shell policy lives in
   `src/ts/adapters/bun_host_runtime.ts`, shared native-library load-info policy
   lives in `src/ts/adapters/native.ts`, shared native status error
   formatting/check policy lives in `src/ts/runtime/native_status.ts`, and Bun handle
   extraction policy lives in `src/ts/adapters/bun_status.ts`.
   Bun native inspection/compile descriptor/runtime-profile ABI policy lives in
   `src/ts/adapters/bun_inspection_ops.ts`, with shared compile-envelope
   defaults/projection in `src/ts/runtime/abi.ts`. Shared LLaMA Program inspection
   projection policy lives in `src/ts/runtime/llama_program_inspection.ts`,
   with no adapter compatibility re-export path.
   Bun generic Session upload ABI policy lives in
   `src/ts/adapters/bun_session_ops.ts`, while shared Session step I/O
   descriptor/result policy lives in `src/ts/runtime/session_step_io.ts`.
   Bun LLaMA token-step/execute/selection
   ABI policy lives in `src/ts/adapters/bun_llama_token_ops.ts`. Bun Program
   buffer/device ABI policy lives in
   `src/ts/adapters/bun_program_buffer_ops.ts`. Bun module Program compile ABI
   policy lives in `src/ts/adapters/bun_module_program_ops.ts`. Shared retained
   Program compile-evidence attachment policy lives in
   `src/ts/runtime/module_program_evidence.ts`, with no adapter compatibility
   re-export path. Bun generic
   Program bind ABI policy lives in `src/ts/adapters/bun_program_bind_ops.ts`. Bun LLaMA
   KV-cache/session bind ABI policy lives in `src/ts/adapters/bun_llama_session_bind_ops.ts`.
   Shared LLaMA session output/KV-cache validation policy now lives in
   `src/ts/runtime/llama_session_bind_desc.ts`, so Node and Bun only differ
   in ABI descriptor packing, not in what resource shapes they accept. Shared
   adapter index/tensor input normalization now lives in
   `src/ts/core/index_values.ts`, so Node and Bun accept the same tensor,
   JS-array, and numeric-typed-array inputs for embedding/class-index paths. Bun
   shared LLaMA KV-cache allocation/resource policy lives in
   `src/ts/runtime/llama_kv_cache.ts`, with Bun adapter glue in
   `src/ts/adapters/bun_llama_kv_cache_ops.ts`. Bun LLaMA KV-cache class
   lifecycle lives in `src/ts/adapters/bun_llama_kv_cache_surface.ts`.
   Shared Adapter `NativeBuffer` public surface lives in
   `src/ts/adapters/native_buffer_surface.ts`. Shared
   generic model/program/session surface lives in
   `src/ts/runtime/generic_family_surface.ts`; shared generic
   model-create/read-handle adapter policy lives in
   `src/ts/adapters/generic_family_model_handle_surface.ts`, with Bun
   descriptor adapter glue in `src/ts/adapters/bun_generic_family_surface.ts`. Bun
   model path/safetensors load-probe ABI policy lives in
   `src/ts/adapters/bun_model_source_ops.ts`; shared supported-checkpoint
   catalog count validation/iteration lives in
   `src/ts/runtime/model_source_catalog.ts`.
   `bun_ffi_runtime.ts` is selected by the TS-owned
   `src/ts/adapters/bun_concrete_runtime.ts` loader policy and now participates
   in the normal strict source build: `scripts/check_ts_source_architecture.cjs`
   requires `tsconfig.source.json` to have no authored-source exclusions,
   forbids `@ts-nocheck` in that file, requires raw Bun host intrinsics to live
   in checked `src/ts/adapters/bun_ffi_intrinsics.ts`, and requires
   `smoke:bun` to keep proving the emitted Bun package runtime plus
   `dist/bun_native.cjs` bundling. The concrete runtime uses the same CJS-safe
   `pathToFileURL(__filename).href` module URL pattern as the Bun native
   wrapper. Bun is no longer a strict-source exception; remaining FFI cleanup is
   concentrated in the separately emitted Node concrete body and any large
   factory surfaces that still require narrow boundary casts.
   `src/ts/runtime/runtime_load_candidates.ts` now owns the shared
   package-artifact candidate policy used by concrete Node/Bun runtime loading
   and shared frontend loading: local sibling/package-built `dist/*.cjs`
   artifacts first, with no `js/generated/**` fallback and no fallthrough to
   repo-only shared CJS shims. Adapter paths consume that runtime policy through
   narrow loader helpers. The flat shared frontend aggregator now lives at
   `src/ts/shared_frontend.ts` and emits to `dist/shared_frontend.cjs` plus
   `dist/shared_frontend.d.cts`; it re-exports the TS-authored Tensor/nn/train/
   runtime helper surface and keeps the temporary module-hook wrappers in one
   typed place. `scripts/check_ts_source_architecture.cjs` guards that aggregator against
   reaching back into repo-only shared CJS shims,
   and `scripts/check_package_artifact.cjs` requires the dist artifact in the
   package. Program/Session composition now lives at
   `src/ts/runtime/session_facade_composition.ts`, emitted to
   `dist/runtime/session_facade_composition.cjs`; it assembles the generic
   `Program -> Session -> StepParams` helper set, parameter upload facade,
   readback/execution/lifecycle wrappers, plus the LLaMA bind-routing, token
   scratch, token selection/generation, readback, prefill, step/execute,
   runtime-profile, and lifecycle composition. The obsolete source-checkout
   `js/shared_session_facade.cjs` and `js/shared_frontend.cjs` compatibility shims
   have been removed, and `scripts/check_ts_source_architecture.cjs` prevents them
   from growing back into second implementations. The shared Adapter Tensor
   public class surface now lives in typed `src/ts/adapters/tensor_surface.ts`
   and is bound by concrete Node and Bun Adapters through a small helper-surface
   seam. Adapter Tensor runtime helper assembly now lives in typed
   `src/ts/adapters/tensor_runtime_surface.ts`, so tensor data/core/math,
   metadata, grad-state, Session tensor wrapping, and host-surface helper
   composition are built once while Node and Bun supply only host callbacks such
   as dtype/device policy, grad-mode policy, NativeBuffer checks, and f32 buffer
   conversion. The
   concrete Node runtime export object now flows through typed
   `src/ts/adapters/public_runtime_exports_surface.ts`, which validates it
   against the TS-owned native API contract at load time. Safetensors file-header IO
   now lives in typed `src/ts/adapters/safetensors_file_header.ts`, leaving
   Node and Bun concrete Adapters to provide only host open/read/close
   callbacks while the shared header parsing semantics stay in
   `src/ts/runtime/model_source.ts`.
   Adapter model-source load-kind routing now lives in typed
   `src/ts/adapters/model_source_facade.ts`, so Node and Bun concrete Adapters
   supply constructors and native handles without owning the path/data routing
   table. Adapter model-source facade forwarding now lives in typed
   `src/ts/adapters/model_source_surface.ts`, so Node and Bun concrete
   Adapters share one late-bound model-source facade seam while each host keeps
   its own uninitialized-surface message. Adapter NativeBuffer instance guards now live in typed
   `src/ts/adapters/native_buffer_instance_surface.ts`, so Node and Bun
   concrete Adapters late-bind the current `NativeBuffer` class without owning
   separate instance-check seams. Adapter index-values forwarding now lives in
   typed `src/ts/adapters/index_values_surface.ts`, so Node and Bun share the
   same Tensor/index input normalization seam. Adapter Module compiler
   forwarding now lives in typed `src/ts/adapters/module_compiler_surface.ts`,
   so Node and Bun share the same late-bound `TraceModuleCompiler` forwarding
   seam. Node model-source Buffer conversion now lives in typed
   `src/ts/adapters/node_model_source_bytes.ts`, so the concrete Adapter does
   not own string/view-to-native-byte-source policy inline. Adapter model handle
   bind/compatibility policy now lives in typed
   `src/ts/adapters/model_handle_policy.ts`, so the Node and Bun concrete
   Adapters supply only model classes and host null-handle policy without owning
   the shared helper factory, LLaMA-bindable/generic-compatible class lists, or
   handle extraction policy inline. Adapter module/optimizer state helper
   assembly now lives in typed
   `src/ts/adapters/module_state_surface.ts`, so parameter creation, module
   state dictionaries, optimizer state snapshots, zero-grad, requires-grad, and
   state restore helpers are composed once from the TS-owned shared frontend while
   hosts supply only Tensor/f32 layout knobs. Adapter `nn` module class assembly
   now lives in typed
   `src/ts/adapters/frontend_module_surface.ts`, so Node and Bun concrete
   Adapters share one Adapter Module for `Linear`, `Embedding`, `Sequential`,
   stateless modules, feature norms, shape modules, trace compiler wiring, and
   module-facade helper construction instead of reassembling product module
   policy in each host runtime. Adapter module Program
   construction now lives in typed
   `src/ts/adapters/program_factory_surface.ts`, so the Node and Bun concrete
   Adapters late-bind the current `Program` constructor without owning module
   compile result construction inline. Adapter Program buffer native bridging
   now lives in typed `src/ts/adapters/program_buffer_native_bridge.ts`, so the
   Node and Bun concrete Adapters supply the current `NativeBuffer` class
   without owning `fromNativeHandle` or device-import extraction inline.
   Adapter NativeBuffer facade assembly now lives in typed
   `src/ts/adapters/native_buffer_facade_surface.ts`, so allocation, wrapping,
   external-resource, IO, inspection, free, and ProgramDevice-handle callbacks
   are composed once while Node and Bun supply only native syscalls and
   null-handle policy.
   Adapter Program buffer factory forwarding now lives in typed
   `src/ts/adapters/program_buffer_factory_surface.ts`, so output, named buffer,
   KV-cache buffer, and LLaMA KV-cache forwarding stay out of concrete runtime
   bodies. Adapter Program/runtime facade assembly now lives in typed
   `src/ts/adapters/program_runtime_surface.ts`, so `ProgramDevice`,
   Program buffer factories, Program/model facade policy, generic Program
   facades, LLaMA model-family facades, and LLaMA Program facades are composed
   once from TS-owned shared frontend helpers while Node and Bun supply only
   native callbacks, null-handle policy, and KV-cache construction differences.
   Adapter LLaMA Session facade assembly now lives in typed
   `src/ts/adapters/llama_session_facade_surface.ts`, so token execution,
   prefill, generation, output readback, runtime-profile, lifecycle, and
   sampling facade helpers are composed once while Node and Bun supply only
   native token-window callbacks and null-handle policy.
   Adapter LLaMA family surface assembly now lives in typed
   `src/ts/adapters/llama_family_surface.ts`, so concrete runtimes pass one
   grouped LLaMA Session facade into the runtime-owned LLaMA Model/Program/
   Session public class surface instead of re-threading every Session method.
   Adapter Session/binding facade assembly now lives in typed
   `src/ts/adapters/session_runtime_surface.ts`, so generic Session lifecycle,
   step/upload helpers, Program bind validation/preparation, owned buffer
   lifetime policy, module binding, and native-buffer bind fields are composed
   once for Node and Bun from TS-owned shared frontend helpers.
   Node LLaMA KV-cache
   factory policy now lives in typed
   `src/ts/adapters/node_llama_kv_cache_factory_surface.ts`, so the concrete
   Adapter late-binds `NativeBuffer` and `LlamaKvCache` without owning
   allocation wrapper construction inline. Adapter ProgramDevice construction
   now lives in typed `src/ts/adapters/program_device_factory_surface.ts`, so
   Node and Bun generic/LLaMA Program facades share one late-bound construction
   callback instead of owning inline `new ProgramDevice(...)` callbacks. Program
   bind Session construction now lives in typed
   `src/ts/adapters/session_factory_surface.ts`: the Node and Bun concrete
   Adapters supply late-bound `Session` constructors, while host-vs-native
   prepared bind construction policy stays in TS source.
   The remaining work is to keep shrinking the temporary separately emitted Node
   concrete FFI body into
   smaller typed TS modules while keeping Bun in the strict source build.
6. Keep `tsdown` or an equivalent package build as the only JS/TS artifact
   fan-out from the TS package spine. It must emit CJS/ESM where needed, `.d.ts`,
   source maps, and subpath exports from authored TS; it must never become a
   second source tree to keep in sync. This is now the package contract:
   `npm run build:package` calls `tsdown` directly;
   `scripts/package_metadata_policy.cjs` derives the package entry map,
   declaration globs, subpaths, root fields, and file boundary from `src/ts/**`;
   the package ships that policy script beside `tsdown.config.mjs`, and the
   policy script exports a `zgml-package-metadata-policy` manifest so package
   fan-out is auditable instead of implicit;
   `tsdown.config.mjs` consumes that policy and emits runnable JS and
   declarations into `dist/**`; `npm run typecheck:dist` and
   TS-authored `npm run smoke:dist` verify those artifacts. `zgml/frontend`, `zgml/node`
   runtime, top-level friendly TS subpaths, adapter evidence, and deep wildcard
   package subpaths now use those dist artifacts. The package root also resolves
   directly to the TS-built Node bridge at `dist/node.cjs`, and Bun resolves
   `zgml/bun` to `dist/bun_native.cjs` under the Bun condition. Root,
   `zgml/node`, and `zgml/bun` type metadata point directly at the TS-owned
   `dist/public_api.d.cts` artifact instead of relying on root declaration
   fallback. The native-free Bun root wrapper is generated from the same
   package-spine generator as the Node and Bun-native wrappers, so `zgml/bun`
   has no hand-maintained root export list in either its default frontend path
   or its Bun-condition native path. The Zig substrate manifest has matching
   named commands: `npm run generate:native-substrate-manifest` and
   `npm run check:native-substrate-manifest` regenerate or verify
   `src/native_substrate_manifest.zig` from `src/ts/frontend_manifest.ts`, and
   the architecture check runs that verification to keep Zig aligned through
   generated substrate metadata instead of a mirrored frontend. A
   dedicated `smoke:native-contract` gate, authored at
   `src/ts/smokes/native_bridge_contract_smoke.ts`, now guards the Node boundary by requiring
   root, the concrete Node adapter, `dist/node.cjs`, and `zgml/node` to expose
   the same TS-declared native API. `scripts/check_ts_source_architecture.cjs` also
   rejects regrowing the legacy root `js/node.cjs` fallback, requires the TS
   Node bridge to use the typed concrete runtime
   loader, requires Node and Bun concrete runtime loaders to share
   `src/ts/adapters/concrete_runtime_loader.ts` for contract fallback loops,
   host require/assert helpers, load candidates, and manifest construction,
   requires Node and Bun host runtimes to share
   `src/ts/adapters/native.ts` native-library load-info policy instead of
   duplicating suffix/path/missing-library checks, requires Node and Bun status
   helpers to share `src/ts/runtime/native_status.ts` instead of duplicating native
   status error formatting, requires Node and Bun module Program helpers to share
   retained compile-evidence attachment policy through
   `src/ts/runtime/module_program_evidence.ts`, requires Node and Bun
   inspection helpers to share LLaMA Program inspection projection through
   `src/ts/runtime/llama_program_inspection.ts`, rejects the TS shared frontend aggregator if it depends on
   repo-only shared CJS shims, rejects TS session facade composition if it
   depends on repo-only CJS shims, rejects regrowing the removed shared CJS
   compatibility shims, requires concrete FFI runtimes to use the typed shared
   frontend loader policy, requires Node and Bun concrete runtimes to consume
   `tensor_surface.ts` instead of defining the Tensor public class in their
   concrete Adapters, requires `node_ffi_runtime.ts` to export through
   `public_runtime_exports_surface.ts` instead of owning a raw unchecked native API
   object literal, requires Node and Bun concrete runtimes to consume
   `frontend_namespace_surface.ts` instead of separately assembling public
   `nn`/`loss`/`train`/`data`/`optim`/`checkpoint` namespaces, requires both
   concrete runtimes to consume `createAdapterTorchNamespace` instead of owning
   separate `torch` compatibility alias objects, requires
   `node_ffi_runtime.ts` and `bun_ffi_runtime.ts` to consume
   `safetensors_file_header.ts` instead of owning inline safetensors
   `fs`/byte-reader callbacks, requires `node_ffi_runtime.ts` and
   `bun_ffi_runtime.ts` to consume `model_source_facade.ts` instead of owning
   model-source load-kind routing maps, requires `node_ffi_runtime.ts` to consume
   `node_model_source_bytes.ts` instead of owning inline Buffer conversion
   callbacks, requires `node_ffi_runtime.ts` and `bun_ffi_runtime.ts` to consume
   `model_handle_policy.ts` instead of owning inline model handle
   bind/compatibility policy, requires `node_ffi_runtime.ts` and
   `bun_ffi_runtime.ts` to consume
   `program_factory_surface.ts` instead of owning inline module Program
   construction policy, requires `node_ffi_runtime.ts` and `bun_ffi_runtime.ts`
   to consume `program_buffer_native_bridge.ts` instead of owning inline Program
   buffer native bridge callbacks, requires `node_ffi_runtime.ts` and
   `bun_ffi_runtime.ts` to consume `program_buffer_factory_surface.ts` instead
   of owning inline Program buffer forwarding methods, requires `node_ffi_runtime.ts` to consume
   `node_llama_kv_cache_factory_surface.ts` instead of owning inline LLaMA
   KV-cache factory callbacks, requires `node_ffi_runtime.ts` and
   `bun_ffi_runtime.ts` to consume
   `program_device_factory_surface.ts` instead of owning inline
   ProgramDevice construction callbacks, requires `node_ffi_runtime.ts` and
   `bun_ffi_runtime.ts` to consume
   `session_factory_surface.ts` instead of owning inline Program bind Session
   construction policy, requires `smoke:dist` to
   execute the generated `dist/smokes/dist_package_smoke.cjs` artifact,
   requires `smoke:frontend` to execute
   `dist/smokes/frontend_package_smoke.cjs`, and requires the package to ship
   `dist/shared_frontend.cjs` and
   `dist/runtime/session_facade_composition.cjs`; the Bun package smoke imports `zgml/bun` and
   proves the Bun bridge satisfies the same contract. The root, `zgml/node`, and
   `zgml/bun` type metadata now point directly at the `tsdown`-emitted
   `dist/public_api.d.cts` declaration artifact, with no root `.d.ts`
   compatibility shim lane. Package and source-artifact guards reject a return
   to source-tree public declarations or root declaration re-exports as
   canonical package types.
7. Keep the public declaration surface generated from implementation types.
   This is now the enforced package shape: the large public contract lives in
   `src/ts/public_api.ts`, `tsdown` emits `dist/public_api.d.cts`, and the
   package metadata points at that emitted artifact. It carries a type-only
   `PublicApiContractManifest` so package type tests can prove the root
   declaration surface is TS-owned without inventing a runtime export. Future declaration work
   should refine implementation-owned types, not re-create a hand-maintained
   root `.d.ts` body or a root declaration re-export.
8. Keep source verification TS-first and package-artifact-backed.
   `scripts/check_ts_source_architecture.cjs` now runs `tsc --noEmit` over the TS
   source tree and architecture guards against legacy repo-only JS paths during
   `npm run test:ts-source`. It rejects any new authored JS/CJS/MJS runtime
   files under `src/ts/**` and rejects hand-maintained product `.d.ts` files
   except for the tiny ambient declarations required for Node/Bun host interop.
   It also rejects `js/generated/**` as a checked-in
   checkout artifact tree; TS is the implementation source of truth, and
   `tsdown` emits the package/runtime artifacts without a wrapper build script
   or generated-source parity lane. The package artifact has
   crossed that line already: `scripts/check_package_artifact.cjs` proves
   `npm pack --dry-run` ships `dist/**` and rejects `js/generated/**` plus local
   smoke harness files. The old repo-only smoke harnesses
   `js/package_smoke.cjs`, `js/package_smoke_bun.ts`,
   `js/native_bridge_contract_smoke.cjs`, `js/ts_source_smoke.cjs`, and
   `js/shared_package_smoke.cjs` have been removed; the source architecture guard
   now requires future smoke coverage to live under `src/ts/smokes` and run
   from `dist/smokes/**`. The public Node/Bun FFI examples have started crossing
   it too: Node loads the built `dist/node.cjs` adapter, Bun imports `zgml/bun`,
   and `scripts/check_ts_source_architecture.cjs` rejects regressions to repo-only
   `../../js/**` example imports. Example assertions should prove behavior,
   frozen evidence, structural signatures, and non-mutating caller APIs; they
   should not require package builds to preserve private object identity or
   scratch-routing internals.

Definition of done:

- no hand-written public `.d.ts` file for frontend/runtime APIs
- no authored JS/CJS/MJS files under `src/ts/**`, and no product declarations
  besides narrow ambient host-interop declarations
- no checked-in generated JS needed for package source-of-truth semantics
- package consumers and source-checkout smokes run from TS-authored `dist/**`
  artifacts emitted by `tsdown`
- no duplicated Node/Bun frontend implementation
- no CJS-authored shared frontend/session/program policy
- host adapters contain only host-specific loading, FFI, and resource interop
- public type smoke imports package-facing declarations rather than repo-only
  generated artifacts
- public Node/Bun FFI examples import package/dist surfaces rather than
  repo-only `js/**` implementation files
- TS Node native bridge loads tsdown-emitted concrete runtime artifacts, and
  the obsolete legacy root `js/node.cjs` shim is removed
- TS shared frontend and session facade composition are package-built from
  `src/ts/**`; obsolete shared CJS compatibility shims are removed
- Node, Bun, browser, and future Wasm entrypoints share the same TS frontend
  semantics
- the existing `npm test && npm run test:adapters` gate remains green throughout

## Open Questions

- Should the public name be `Program`, `CompiledProgram`, or something else?
- Should the internal executable object be called `ProgramStencil`, or should
  "stencil" stay only in evidence/benchmark terminology?
- How much shape polymorphism should one Program allow before recompilation is
  required?
- Can compatible quantized and unquantized checkpoints share a program shape, or
  do kernel/layout differences require distinct Programs?
- Which C ABI surface should ship first: tiny tensor programs, LLaMA inference,
  or both?
- Should Node support only native addons first, or should the ABI be designed
  from day one to mirror the Wasm handle model?

## Near-Term Next Move

The LLaMA Program/Session lifecycle target and the first honest real WebGPU/wgpu
execution slices have largely landed. The next high-bang product move is to make
the PyTorch-like Frontend feel less like a set of historical adapters:

- make **TypeScript the only authored product frontend**: tensor, `nn`, `loss`,
  `optim`, `train`, model state, tracing, compile policy, and JS FFI ergonomics
  should be implemented once in `src/ts/**` and emitted with `tsdown`
- keep converging the **Trace-to-Program Compiler** with the native executable
  runtime through one lowering contract; the contract is the IR/ABI/Program
  behavior, not mirrored Zig and JS/TS module APIs
- keep collapsing Node/Bun semantics into the **Shared JS/TS Frontend Core**
  with runtime adapters owning only FFI/loading differences
- keep growing the **Tensor Placement Interface** so eager tensors,
  `ModuleBindings`, `ProgramBindings`, `NativeBuffer`, and future device buffers
  feel like one object model
- generate or mechanically validate native-boundary artifacts wherever possible:
  ABI constants, native API manifests, package exports, host adapter required
  symbols, and smoke expectations should be contract-checked rather than
  synchronized by hand

Do not spend product budget making Zig `nn` match TS `nn` one method at a time.
Zig should grow when the runtime needs a kernel, buffer primitive, C ABI record,
backend hook, or low-level native handle. TS should grow when the library needs
PyTorch-like user vocabulary. The two meet at the executable contract.

The current backend/runtime evidence that this frontend should sit on is:

```text
current real state:
  Program owns command shape, kernels, schedule evidence, and patch table
  Module kernel plans expose exact shape constraints and named packed-parameter
  binding layout for Session state
  LLaMA weights, KV cache, logits/output buffers, and adapter state bind as
  Session-owned persistent/runtime resources
  StepParams carries token/window/output policy without graph access
  C/Node/Bun/Wasm handles can compile once, bind compatible sessions, step into
  caller-provided buffers, and inspect evidence without exposing tensors
  Builds without native wgpu and portable Wasm/browser LLaMA WebGPU remain
  compile-only/resource-layout evidence; the optional `-Duse-wgpu=true` build now proves wgpu-native
  headers/native linkage via `zig build wgpu-link-smoke -Duse-wgpu=true`, and
  `npm run smoke:native-wgpu` wraps
  `zig build wgpu-check -Duse-wgpu=true`, which runs the complete optional
  native WebGPU validation suite: link smoke, generic/tiny-linear executor
  smoke, and public LLaMA WebGPU smokes.
  The `src/backend/wgpu.zig` executor proves host-staged tiny-linear,
  standalone dense f32 matmul, dense matmul-elementwise add/mul sidecars,
  dense matmul-fused-elementwise sidecars,
  standalone qmatmul with Session-bound runtime
  qweights, fused qmatmul-add, qmatmul-elementwise add/mul sidecars,
  standalone elementwise, standalone fused elementwise, standalone RMSNorm,
  standalone softmax, standalone reduce sum/max, standalone RoPE, and
  standalone slice-assign plus bounded standalone attention Programs can
  allocate wgpu resources, dispatch compute kernels, read outputs back, report
  `ZGML_FEATURE_NATIVE_WGPU_EXECUTION` at runtime-info level, and report bound
  Session profile evidence through the C ABI tests and
  `zig build wgpu-exec-smoke -Duse-wgpu=true`
  The executor can now also compile and execute bounded multi-dispatch Programs
  made from already-supported exact WebGPU shapes: it lowers the DeviceProgram
  op tape into per-dispatch `ProgramShape`s, allocates one pipeline/params
  buffer/bind group per dispatch, encodes the sequence into one command buffer,
  and reports actual WebGPU dispatch counts rather than assuming command-shape
  counts always match backend dispatches
  WebGPU bind groups now bind whole backend-owned program buffers at aligned
  offset zero for host-backed Session state; tensor-view offsets remain in the
  cached shader params/patch table. Same-device imported `WGPUBuffer` views
  must use a WebGPU-valid storage-buffer binding offset before entering the
  Session resource table.
  C/Node/Bun opt-in wgpu smokes prove host-buffer weights are uploaded as
  persistent Session state at bind time: changing the host weight buffer after
  bind does not change later steps, while changing the host input buffer does
  The wgpu executor also proves no-readback execution: `download_outputs=false`
  dispatches without touching host output memory or incrementing sync counters,
  and C/Node/Bun expose that as tiny-linear `step_no_output` / `advance`
  The opt-in wgpu backend now also has the first real configured resource-table
  execution proof: registered same-device WGPUBuffer resources can back
  tiny-linear weights, bias, input, and output bindings; fake/unregistered
  opaque handles still reject before bind-group execution
  Standalone dense f32 matmul also runs through configured Session bindings with
  host A/B/output buffers, no-readback dispatch, and registered same-device
  WGPUBuffer A/B/output resources
  Standalone qmatmul also runs through configured Session bindings with host
  input/output buffers, no-readback dispatch, registered same-device WGPUBuffer
  input/output resources, and independent Session-bound qweight uploads
  Fused qmatmul-add also runs through configured Session bindings with host
  input/bias/output buffers, no-readback dispatch, registered same-device
  WGPUBuffer input/bias/output resources, and independent Session-bound qweight
  uploads while preserving two-op command/profile evidence
  Qmatmul-elementwise add/mul sidecars also run through configured Session
  bindings with host input/secondary/output buffers, no-readback dispatch,
  registered same-device WGPUBuffer input/secondary/output resources, and
  independent Session-bound qweight uploads while preserving two-op
  command/profile evidence
  Standalone elementwise also runs through configured Session bindings with host
  src0/src1/output buffers, no-readback dispatch, and registered same-device
  WGPUBuffer src0/src1/output resources
  Standalone RMSNorm also runs through configured Session bindings with host
  src/output buffers, no-readback dispatch, and registered same-device WGPUBuffer
  src/output resources
  Standalone softmax also runs through configured Session bindings with host
  src/output buffers, no-readback dispatch, and registered same-device WGPUBuffer
  src/output resources
  Standalone reduce sum/max also runs through configured Session bindings with
  host src/output buffers, no-readback dispatch, and registered same-device
  WGPUBuffer src/output resources
  Standalone RoPE also runs through configured Session bindings with host
  src/cos-sin/output buffers, no-readback dispatch, and registered same-device
  WGPUBuffer src/cos-sin/output resources
  Standalone slice-assign also runs through configured Session bindings with
  host strided src/output buffers, no-readback dispatch, and registered
  same-device WGPUBuffer src/output resources
  Bounded standalone attention also runs through configured Session bindings
  with host q/k/v/mask/output buffers, no-readback dispatch, and registered
  same-device WGPUBuffer q/k/v/mask/output resources
  `zgml_program_create_device_buffer` turns that proof into a public
  zgml-owned WebGPU device-buffer lane for C/Node/Bun tiny-linear programs
  `zgml_program_import_device_buffer` adds the first public same-device import
  lane for valid WGPUBuffer handles validated by program device token, size,
  storage-buffer alignment, and usage; C/Node/Bun smokes reject wrong-device,
  unaligned, and oversized views
  `zig build wgpu-exec-smoke -Duse-wgpu=true` now also proves the configured
  tiny-linear wgpu hot path performs no Zig heap allocations/resizes after
  Session bind/configure
  A private native LLaMA WebGPU probe now lowers the fixed tiny LLaMA decode
  and fixed-window prefill tapes far enough to prove CPU-parity logits,
  no-output decode advancement, prefill cache side-effect readback, explicit
  decode cache refresh, no-output prompt advancement,
  decode-after-prefill runtime-window behavior, and registered same-device
  resource-bound logits/KV execution for decode and prefill under
  `zig build test -Duse-wgpu=true -- --test-filter "native WebGPU tiny llama"`.
  The fixed tiny LLaMA resource proof now also keeps decode and prefill on the
  same retained WebGPU device, imports the same registered K/V buffers into the
  prefill Program, prefill-writes those buffers on GPU, then decodes from the
  decode Program's bound resource table without CPU cache staging. That same
  private gate now includes a small GQA shape (`n_heads = 4`, `n_kv_heads = 2`)
  proving multi-head attention over packed multi-KV-head cache resources matches
  CPU for prefill and decode-after-prefill, plus a two-layer shape proving the
  per-layer K/V resource table survives prefill-to-decode handoff across more
  than one cache pair. A longer-context private proof now compiles an 8-token
  fixed-window prefill over a 16-token context, writes same-device K/V resources
  on GPU, and decodes from that GPU-resident cache with zero fallback. The same
  helper also proves the no-output prompt path: prefill advances tokens with
  logits omitted, leaves the logits resource untouched, records zero runtime
  syncs/fallback ops, and decode still consumes the GPU-resident K/V resources
  to match CPU. A wider-head private proof now compiles and executes a
  LLaMA-style `d_head = 128` resource prefill/cache-handoff/decode path over
  the same 16-token envelope, so the attention proof is no longer limited to
  toy head widths. A private quantized tiny LLaMA decode proof now quantizes
  the in-memory weights, compiles a WebGPU Program with nonzero qweight
  evidence, matches CPU quantized logits within the qmatmul tolerance through
  both host and external-resource logits/KV bindings, then binds a second
  runtime with zeroed qweight payloads and proves WebGPU follows that
  Session-owned qweight state without fallback. These private LLaMA WebGPU
  proofs now also assert the
  compiled program has full wgpu dispatch-plan coverage, includes projection and
  attention dispatch families, includes quantized projection families for the
  quantized proof, and records runtime `backend_dispatch_count` equal to the
  compiled dispatch-plan count for executed decode/prefill steps. A combined
  private shape now puts those gates together: two LLaMA layers, GQA
  (`n_heads = 4`, `n_kv_heads = 2`), a 16-token context, an 8-token GPU-resource
  prefill, same-device K/V handoff, decode from the GPU-resident cache, and the
  same full dispatch-plan/runtime-dispatch-count evidence. The same combined
  resource-handoff shape now also has a failing-allocator proof: after Program
  compile, Session bind/configure, resource registration, and persistent upload,
  the resource-bound prefill hot path, decode cache refresh, and resource-bound
  decode hot path perform no Zig heap allocations or resizes while still
  matching CPU logits. The native and LLaMA-specific resource gates now also
  prove bad K/V cache access modes reject before Session binding can dispatch:
  decode rejects write-only cache resources, and prefill rejects read-only cache
  resources because it must write cache side effects.
  The resource proof also covers aliased terminal outputs: when a terminal
  logits tensor shares compiled workspace storage with intermediates, wgpu now
  executes into the internal buffer and GPU-copies the terminal slice into the
  external resource instead of rebinding the workspace buffer itself.
  Native public LLaMA-family `.webgpu` execution now defaults on with
  `-Duse-wgpu=true`, while explicit opt-out and portable browser/Wasm builds
  keep the resource-probe evidence lane.

next target:
  grow the bounded multi-dispatch wgpu executor from C-level same-device
  WGPUBuffer import
  toward browser/Node GPUBuffer interop without silently falling back to CPU
  keep consuming the same ProgramStencil command shape as CPU/Metal/stencil
  keep configuring the complete persistent/input/output binding table into
  backend resource tables at Session bind time
  preserve execution_supported = true only for the real executor
  widen the experimental LLaMA WebGPU parity proof into full LLaMA-family
  no-fallback and broader shared-resource handoff coverage before making
  LLaMA-family WebGPU execution a default public support claim
```

The backend seam now points in that direction: `configure_bindings` receives the
complete persistent/input/output table, CPU/Metal/stencil runtime bindings own a
copy of that full table, and normal Session execution uses the configured table
by default. That gives wgpu/WebGPU a place to build bind groups/resource tables
outside the hot step path.

The first native wgpu prerequisite is also now explicit: `build.zig.zon`
declares the local `vendor/wgpu-native` package, `build.zig` has an optional
`use_wgpu` setting, and `src/backend/wgpu_probe.zig` compile-links against the
target-specific wgpu-native artifact without importing the old stale backend.
The next native executor step has landed too:
`src/backend/wgpu.zig` consumes the existing `DeviceProgram`/`ProgramStencil`
shape for deliberately narrow tiny-linear, standalone dense f32 matmul,
fused dense matmul-elementwise add/mul sidecars,
fused dense matmul-fused-elementwise activation chains,
fused dense matvec-elementwise add/mul, fused dense matvec-fused-elementwise
activation chains, fused dense matvec-slice cache store,
fused dense matvec-RoPE cache store, standalone qmatmul with Session-bound
runtime qweights, fused qmatmul-add, fused qmatmul-elementwise add/mul, fused
qmatvec-elementwise add/mul, fused
qmatvec-fused-elementwise activation chains, fused qmatvec-slice cache store,
fused qmatvec-RoPE cache store, standalone
elementwise, standalone fused elementwise, standalone repeat,
repeat-to-fused-elementwise
activation chains, standalone LayerNorm, standalone RMSNorm,
standalone softmax, standalone reduce sum/max, standalone RoPE, standalone
slice-assign, and bounded standalone attention
programs, owns wgpu
device/pipeline/buffer resources at compile/bind
time, executes WGSL compute kernels, downloads host-bound output when requested,
executes no-output steps without backend readback sync, and records Session
profile evidence. It now also has the first real bounded multi-dispatch
executor: a supported `DeviceProgram` may lower to multiple exact WebGPU
dispatches over the same runtime buffer table, with one pipeline, params buffer,
and bind group per dispatch and a single encoded command buffer for the
sequence. Runtime-window patch values now live in Session-owned wgpu params
buffers instead of a compiled-program global uniform buffer, so compiled
programs can be reused by multiple live sessions without cross-session patch
leakage. It also now resolves the
configured Session binding table into a real bind group for registered
same-device WGPUBuffer resources, proving resource-backed tiny-linear
weights/bias/input/output, standalone matmul A/B/output, standalone qmatmul
input/output with independent Session-bound qweight uploads, standalone
qmatmul-add input/bias/output with the same qweight state, fused
qmatmul-elementwise input/secondary/output add/mul sidecars with the same
qweight state, fused dense
matmul-elementwise input/weights/secondary/output add/mul sidecars,
fused dense
matmul-fused-elementwise input/weights/secondary/output activation sidecars,
fused dense
matvec-elementwise input/secondary/output for add/mul sidecars, fused dense
matvec-fused-elementwise input/weights/secondary/output activation sidecars,
fused dense
matvec-slice input/weights/cache-output with runtime-window cache-position
patching, fused dense matvec-RoPE input/weights/cos-sin/cache-output with the
same patching, fused
qmatvec-elementwise input/secondary/output for add/mul sidecars, fused
qmatvec-fused-elementwise input/secondary/output activation sidecars, fused
qmatvec-slice input/cache-output with runtime-window cache-position patching,
fused qmatvec-RoPE input/cos-sin/cache-output with the same patching,
standalone elementwise src0/src1/output, standalone fused elementwise
src/secondary/output, repeat-to-fused-elementwise src/repeated-source/secondary/output,
standalone repeat src/output,
standalone LayerNorm src/output, standalone RMSNorm src/output, and
standalone softmax src/output, standalone reduce src/output, and standalone
RoPE src/cos-sin/output, standalone slice-assign src/output, and bounded
standalone attention q/k/v/mask/output execution without CPU fallback while
continuing to reject fake/unregistered opaque handles. The
registered-resource proof now has public owned-buffer and C import surfaces:
  `zgml_program_create_device_buffer(program, kind, ZGML_BACKEND_WEBGPU, ...)`
  allocates same-device buffers that C/Node/Bun bind and execute as
  external-resource Session state, while
  `zgml_program_get_device_handle` plus `zgml_program_import_device_buffer`
  lets C/Node/Bun callers import valid same-device WGPUBuffer handles after
  device-token, size, storage-buffer alignment, and usage validation. Node/Bun
  smokes now exercise the same path through `program.device("webgpu")`, whose
  role-specific helpers allocate weights/bias/input/output and LLaMA logits/KV
  resources directly from the same provenance object. Program-level
  `createKvCache({ placement: "webgpu" })` remains as a compatible shorthand,
  but the future JS shared-device object shape now dispatches through the
  evidence-gated ABI without a manual per-layer resource callback or bouncing
  back to Program-level placement options. Native Node/Bun hosts can now
  pass GPUBuffer-like objects through exported `webgpuInterop` symbols for
  device/buffer handles, byte ranges, placement, or a lazy import-source
  descriptor; `src/ts/adapters/webgpu_interop.ts` owns that exported
  `Symbol.for("zgml.webgpu.*")` vocabulary for both concrete Node and Bun
  runtimes, and `ProgramDevice` fills in the same-device token by default while
  direct Program imports still require explicit provenance. The package TS
  declarations now type that symbol vocabulary as a closed unique-symbol
  Interface, so symbol-keyed same-device imports get compile-time validation
  instead of collapsing to a generic `Record<string, symbol>`. Default compile-only
  Node/Bun smokes also prove the
  ProgramDevice object, device-buffer creation, and raw device import throw
  unsupported instead of creating fake same-device resources. C/Node/Bun smokes
  also prove public LLaMA-family WebGPU Programs cannot use that same-device
  lane yet: device-handle lookup, ProgramDevice creation, device-buffer
  creation, and raw import throw/return unsupported, preserving the compile-only
  resource-probe public gate. Those public smokes also reject wrong-device,
  unaligned, and oversized imported views before binding. This is not yet a
  general public
  `ZGML_BACKEND_WEBGPU` executor because LLaMA-family execution and browser
  GPUBuffer execution still need a Wasm/WebGPU shared-device runtime beyond
  native FFI raw handles and portable opaque host-resource descriptors, but the C
  ABI now routes tiny-linear WebGPU compilation to it when built with
  `-Duse-wgpu=true`. Runtime qweight
  rebinding now exists for standalone qmatmul, fused qmatmul-add, and fused
  qmatmul-elementwise add/mul sidecars, and bounded
  multi-dispatch lowering now exists for short Programs made from supported
  dispatch shapes, and a private fixed tiny LLaMA probe now proves native
  decode/prefill/no-output prompt/cache-handoff parity against CPU, including
  same-device imported K/V resource handoff from prefill to decode, no-output
  resource prefill with zero sync/fallback profile evidence, a tiny GQA
  resource-handoff shape, a two-layer resource-table handoff shape, and
  an 8-token prefill over a 16-token context plus resource-bound quantized
  decode with Session-owned runtime qweight rebinding. It now also covers a
  classic `d_head = 128` resource handoff, plus the combined case of GQA plus
  two layers plus 8-token same-device resource prefill over a 16-token context
  before decode. That private probe audits the compiled wgpu dispatch plan:
  every op in the compiled `ProgramStencil` is covered, the expected
  projection/attention and quantized projection families appear, and hot runtime
  dispatch counters match the compiled dispatch count. The combined
  resource-bound prefill/cache-refresh/decode path also now runs under a
  failing allocator after binding, proving no Zig heap allocations or resizes in
  the hot path. Native and LLaMA-specific tests also reject bad K/V resource
  access modes before execution, covering write-only decode caches and
  read-only prefill cache outputs. The public LLaMA facade now enables native
  WebGPU execution by default when the artifact is built with `-Duse-wgpu=true`,
  and the public Zig facade now covers the GQA/MQA/MHA plus tied-LM-head same-device resource
  prefill handoff matrix, a realistic `d_head = 128` resource-handoff shape,
  the combined realistic `d_head = 128` plus GQA plus two-layer resource-handoff
  shape, resource-bound quantized projection dispatch/logits parity through the
  normal `LlamaModel`/`Program`/`Session` path, and the reduced compiled-context
  envelope described above:
  `zig build llama-wgpu-experimental-smoke -Duse-wgpu=true` now runs the public Zig facade plus
  C, Node, and Bun FFI smokes. Together they compile the normal
  `model.compile(.{ .backend = .webgpu }).bind(...).step(...)` path against the
  real wgpu executor, check CPU-parity logits, and cover public prefill,
  no-output prompt advancement, decode-after-cache-handoff, and same-device
  resource-bound logits/KV cache execution through the same Program / Session
  APIs that callers would use. The Zig facade plus C/Node/Bun additionally
  run under the one-shot `npm run smoke:native-wgpu` /
  `zig build wgpu-check -Duse-wgpu=true` optional validation gate. They
  prove resource-bound argmax/sample execution and greedy/top-k=1 generation
  parity, with logits read back from the bound output resource only for host
  token selection. The Zig facade, C ABI unit path, and Node/Bun
  smokes also prove a resource-bound over-context step rejects before additional
  runtime patching, dispatch, sync, fallback, or logits-resource mutation.
  C/Node/Bun now also assert that single-token executable decode, including
  same-device resource-bound decode, records a hot runtime backend dispatch
  count exactly equal to the compiled executable dispatch count, rather than
  merely proving that some GPU dispatch occurred.
  That optional gate now also runs the native WebGPU LLaMA execution proofs for
  runtime quantized-weight rebinding, resource-bound decode/prefill handoff,
  long-prompt prefill, GQA long-prompt prefill, and a realistic head-width
  resource-handoff shape, so `-Duse-wgpu=true` validates real same-device
  execution breadth instead of only compiling those paths in the full test
  suite.
  Runtime-info now exposes that native lane separately:
  `ZGML_FEATURE_NATIVE_WGPU_EXECUTION` means the native wgpu executor exists,
  while `ZGML_FEATURE_EXPERIMENTAL_LLAMA_WGPU_EXECUTION` means the
  native LLaMA WebGPU execution lane was compiled in. It defaults on with
  `-Duse-wgpu=true` and can still be forced off with
  `-Dexperimental-llama-wgpu-execution=false`. C/Node/Bun smokes assert the
  latter bit agrees with LLaMA `.webgpu` `execution_supported`, so embedders can
  distinguish a real native executor from compile-only resource-probe evidence.
  LLaMA-family WebGPU execution still needs broader checkpoint-family coverage
  and browser/JS `GPUBuffer` shared-device interop before the whole plan is
  complete. Public
  C/Node/Bun/Node-WASI/browser Wasm resource-probe token step and no-output
  advance now return unsupported with untouched caller logits, unchanged
  Session position, and zero Session runtime work; Node-WASI/browser Wasm also
  prove those claimed token calls are decoded by the host WebGPU bridge as
  concrete `zgml_token_execute_desc` token-window descriptors, with
  over-context windows, out-of-vocab token IDs, and undersized logits outputs
  still rejected as shape-mismatch before valid requests return unsupported.
  Host-side profile counters now distinguish the ten Node-WASI/browser resource-token attempts
  in the smokes: logits-output versus no-output policies, caller-output versus
  bound-output requests, shape rejections, result-pointer writes, and
  valid-but-unsupported execution attempts. Execute-and-select and generation
  setup append the same logits-producing descriptor records to host Session
  history, while bound argmax/sample plus greedy/top-k=1 generation reject with
  unchanged output token buffers and unchanged position. The host wrapper also
  proves an explicit token-executor hook at the same bridge: valid
  caller-output, no-output, and bound-output token windows can return `ok`,
  receive explicit `stepParams` window evidence plus model-binding identity,
  create and bind a read-only tensor-shaped model-resource table for future
  weight buffers, write logits, mutate position-sliced K/V cache resources
  through the reusable `WasmWebGpuLlamaTokenWindowPatternExecutor`, verify those
  bound resource side effects through browser async readback, and advance the
  host-owned position surfaced through `zgml_session_position` and
  `zgml_session_inspect`; the same host work is visible through
  `zgml_session_runtime_profile` with backend dispatch, fallback-op, and command
  counts derived from the executor result rather than from callback count alone,
  and clearable through `zgml_session_reset_runtime_profile`; malformed executor
  statuses, malformed executor counters, mismatched executor-returned
  resource-table hashes, missing resource-table identity on successful
  table-bound results, or mismatched executor-returned model handles reject as
  `invalid_argument` without position advancement or invented work, and the
  internal `unclaimed` routing sentinel cannot leak back as an executor status.
  The same WebGPU host bridge now claims raw exported `zgml_session_reset` for
  native-backed Wasm resource Sessions and the wrapped-exports facade claims
  reset for host-only resource Sessions, so replay rewinds host-owned position
  without bypassing the public C ABI lifecycle. Node-WASI and browser smokes
  prove reset plus first-token replay on the two-layer safetensors block
  pipeline, including terminal logits, layer K/V row-zero writes, position
  rewind, command-count evidence, and zero fake native fallback.
  The wrapped facade also claims host-only `zgml_session_inspect`, writes the
  fixed ABI inspection struct from the host resource Session, and proves execute,
  reset, and replay positions through the same resource storage, binding-count,
  and binding-table-hash fields as native-backed WebGPU Sessions.
  Successful executor results must also keep work counters coherent: command
  count equals backend dispatch count plus explicit fallback op count, and op
  counters cannot be smaller than their corresponding command/dispatch counts;
  Node-WASI and browser smokes now explicitly reject an executor that claims a
  command while under-reporting command-op evidence.
  Non-`ok` executor results must also report zero output length and zero
  backend/fallback/command work; a failed result that still claims work is
  rejected as malformed evidence rather than hidden in the failure path.
  Table-bound successful results must report nonzero command work; an `ok`
  result with a validated table but no backend or fallback command evidence is
  rejected as a fake success.
  Successful logits-output executor results must report exactly the Session
  vocab length; partial logits evidence rejects before position advancement.
  A model-bound/model-resource success smoke
  proves the compatible model handle and every tiny LLaMA checkpoint
  weight-resource role reach the executor with dtype, shape, element-count, and
  byte-length evidence and are accepted only when the result reports the same
  model and model-resource table identities, while an over-context request is
  still rejected before executor dispatch. In browser
  `GPUBuffer` mode the pattern executor uses real compute dispatches for bound
  logits and K/V cache writes and uses an explicit element-offset params slot
  for nonzero K/V windows; in mock mode those same logical pattern writes report
  as fallback work. A separate
  `WasmWebGpuLlamaEmbeddingProjectionExecutor` now consumes actual
  `model.embed_tokens.weight` and `lm_head.weight` model-resource tensors,
  accepts a token window, computes embedding-to-logits projection for the final
  token into the bound logits resource, dispatches real WGSL for browser
  `GPUBuffer` resources, and reports a single explicit fallback op in mock
  mode. `WasmWebGpuLlamaRmsNormProjectionExecutor` now consumes
  `model.norm.weight` too, applies final RMSNorm over the selected final-token
  embedding, and projects the normalized hidden state through `lm_head.weight`
  into the bound logits resource with the same browser-WGSL or
  explicit-mock-fallback split. Node/WASI and browser smokes prove both
  one-token and multi-token final-token logits for these projection probes.
  `WasmWebGpuLlamaKvProjectionExecutor` now adds
  the first attention-side browser/Wasm tensor proof: it consumes the
  first-layer input RMSNorm plus K/V projection weights, accepts a token window,
  writes the projected K/V rows into the bound cache window for no-output
  execution, leaves bound logits untouched, and uses one fused browser WGSL
  dispatch or one explicit mock fallback op. Node/WASI and browser smokes prove
  both one-token and two-token K/V cache windows. Its default model-resource
  roles are now derived from the executor `layer` option, so `layer: n`
  naturally targets `model.layers.n.*` weights while explicit role overrides
  still win; the tiny checkpoint smokes still execute layer 0 only.
  `WasmWebGpuLlamaAttentionProjectionExecutor` now proves the
  next cache-consuming step: it accepts a token window, writes each token's K/V
  cache row, reads prior rows through the growing attention window, applies Q/K
  RoPE, attention softmax, `o_proj`, residual add, final RMSNorm, and
  `lm_head`, and writes logits for the final token when requested. Node/WASI and
  browser smokes cover both single-token cache handoff and two-token attention
  prefill, and assert the same layer-derived Q/K/V/O default-role contract.
  `WasmWebGpuLlamaBlockProjectionExecutor`
  extends that proof through post-attention RMSNorm, SwiGLU FFN
  (`gate_proj`, `up_proj`, `down_proj`), the FFN residual, final RMSNorm, and
  `lm_head` logits while preserving the same K/V cache handoff. It now accepts
  a token window for that one-layer proof by executing one browser WGSL dispatch
  or one explicit mock fallback per token, writing each token's K/V cache row,
  optionally writing each token's `blockHidden` vector into a bound
  `llama.activation` resource, and emitting bound logits for the final token
  only. `WasmWebGpuLlamaResourceProgram` now exposes
  `createActivationBuffer({ hiddenSize })` for that f32
  `[contextLength, hiddenSize]` activation surface; factory-created safetensors
  Programs retain the inferred hidden size, so callers can create that
  activation surface without restating `hiddenSize`. Callers can bind it as
  `activation`/`activationOutput` for writes and `activationInput` for reads.
  Block-executor bind preparation rejects malformed activation byte lengths and
  aliased activation input/output views before execution, keeping the path
  ping-pong-ready for later layer chaining. With `activationInput` bound, the
  block executor consumes the previous hidden row instead of
  `embed_tokens[token]`; without it, the one-layer proof keeps the embedding
  input path. `WasmWebGpuLlamaBlockPipelineExecutor` composes those block
  executors into the first browser/Wasm activation pipeline: each token window
  runs stage-by-stage, intermediate stages write Session-owned activation
  workspaces, later stages consume those rows through `activationInput`, and
  only the terminal stage emits logits. The native-bound smoke still
  deliberately repeats the one available compiled tiny-checkpoint layer to
  prove ABI-level ping-pong orchestration without overclaiming broader native
  checkpoint support. `WasmWebGpuLlamaResourceProgram.bindHostResources(...)`
  now creates runtime-owned host-only LLaMA resource Sessions, with generated
  JS-side handles intercepted by `WasmWebGpuSessionRuntime.wrapExports(...)`
  instead of being passed to native pointer-taking exports. Node/WASI and
  browser smokes use that path to bind two- and three-layer f32 safetensors
  resource Sessions with real layer-indexed tensors plus separate per-layer K/V
  resources, then run the `tiny-llama-block-pipeline` proof executor against
  deterministic reference math. A second non-tiny-width shape
  (`vocab=10`, `hidden=6`, `ffn=12`, `contextLength=3`, two full-width
  attention layers) now runs the same no-output-prefill plus bound-logits
  decode path, proving the browser/Wasm safetensors factory and block pipeline
  are not hardwired to the original `[8, 4, 8]` tiny checkpoint dimensions.
  The safetensors factory now also derives grouped head geometry, FFN
  intermediate size, RoPE base, RMSNorm epsilon, and tied-LM-head policy from
  `__metadata__` fields, config JSON embedded in `__metadata__`, HF-style config
  objects/JSON strings, or explicit options, and rejects conflicting sources
  before binding. It also rejects malformed final-norm,
  per-layer norm, attention O-projection, and MLP `gate_proj`/`up_proj`/
  `down_proj` tensor shapes during Program creation, before any Session bind.
  HF-style config objects and direct metadata aliases also now reject
  unsupported semantics at Program creation: non-LLaMA-family identity hints,
	  malformed dtype/cache/dropout/token-id hints, non-SiLU activations,
	  enabled attention/MLP bias without complete matching projection-bias tensors,
	  disabled attention/MLP bias that conflicts with present bias tensors,
	  enabled Q/K projection normalization without complete matching norm tensors,
		  enabled sliding-window attention without a concrete
	  window, conflicting option/metadata/config sliding-window sources,
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
  F16 and BF16 safetensors weight tensors are now accepted at this boundary by
  materializing them once into executor-compatible f32 model resources during
  Session resource creation; the hot block executor still runs the same f32
  resource shape rather than growing dtype branches.
  The proof executor uses those Program-carried values for packed grouped
  attention: the Node/WASI and real browser `GPUBuffer` smokes run two-layer MHA
  (`kv=8`, four K/V heads), GQA (`kv=4`, two K/V heads), and MQA (`kv=2`, one
  K/V head) shapes with `vocab=6`, `hidden=8`, `headSize=2`, `ffn=8`,
  `contextLength=3`, non-default RoPE bases, and non-default RMSNorm epsilons
  through no-output prefill, decode-after-prefill, bound logits, and packed
  per-layer K/V cache parity without manually wiring the executor. The matrix
  now covers metadata-derived MHA, GGUF-style and architecture-prefixed
  metadata-only GQA,
  config-object-derived GQA,
  a 17-token no-output prefill plus three-token logits decode over a 32-token
  config-derived GQA context,
  config-JSON-derived MQA, a factor-8 Llama 3 RoPE GQA config,
  metadata-embedded-config GQA, a metadata-embedded Mistral-family GQA config
  whose `sliding_window` is shorter than the compiled context and changes the
  decode logits reference window,
	  a biased Qwen2-style GQA config with complete attention/MLP projection-bias tensors
	  plus optional terminal `lm_head.bias`, structural Qwen2 attention/MLP
	  bias inference from complete projection-bias tensors when config metadata
	  is silent,
	  and a no-bias Qwen2-style GQA config whose `head_dim` matches the inferred
	  attention head size, a Qwen3-style GQA config whose Q/K projection norms
	  are applied per head before RoPE, plus a SmolLM3 GQA identity whose
	  `no_rope_layer_interval` disables RoPE on the second proof layer, all with
	  derived `intermediate_size`, while accepting declared LLaMA-family identity fields such as
	  `model_type = llama/mistral/qwen2/qwen3/smollm3`,
	  `architectures = [LlamaForCausalLM/MistralForCausalLM/Qwen2ForCausalLM/Qwen3ForCausalLM/SmolLM3ForCausalLM]`, and
  positive `pretraining_tp` as a training-time partitioning hint over already
  materialized full tensors, plus well-formed inference-only HF hints such as
  `use_cache`, `torch_dtype`, dropout probabilities, and tokenizer special-token
  ids, plus no-op/default, factor-2 linear, and factor-8 Llama 3
  `rope_scaling` objects, proving the
  browser/Wasm block pipeline is no longer
  limited to one full-hidden attention head, full-hidden K/V cache stride,
  hard-coded `rope_theta=10000`, hard-coded `rms_norm_eps=1e-5`, or blind
  tied-LM-head synthesis, including the single-K/V-head case, while still
  avoiding a false claim that raw safetensors tensor shapes alone identify
  arbitrary LLaMA head geometry.
  The safetensors factory defaults to that proof
  executor when the caller does not explicitly provide a token executor or
  preset, and the executor infers contiguous layer plans from
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
  browser/Wasm resource Program now also exposes a frozen pre-bind `inspect()`
  snapshot with model shape, output/K/V/activation byte requirements, strict
  model-resource roles, default model/resource presence, executor kind,
  explicit execution coverage (`resource-probe`, `bounded-proof`, or
  `custom-executor`), the current `fullDefaultExecutionSupported` claim,
  GPU-storage-buffer requirements, current adapter feasibility, and the same
  binding-requirement hash/count evidence as native Program inspection. That
  gives embedders the same Program-first audit point before Session binding
  across C, Node, Bun, and browser/Wasm. Bound browser/Wasm
  resource Sessions now expose a matching frozen JS `inspect()` snapshot with
  the Program contract, current position, max token window, model binding kind,
  logits/K/V table roles and hashes, model-resource table/manifest hashes, and
  the same execution coverage fields, so hosts can prove the Session they are
  stepping is still attached to the audited Program/resource shape without
  mistaking a bounded proof executor for full/default LLaMA-family support.
  The binding helper now also accepts safetensors checkpoint bytes directly as
  `modelResources`, including explicit byte-backed shard lists and HF-style
  `weight_map` indexes, derives the strict model-resource requirement manifest
  from those bytes when none was declared, rejects duplicate tensor names,
  index/shard mismatches, or mismatched index `metadata.total_size` across
  shards, preserves each tensor's shard index/name in the model-resource
  manifest, and expands them into read-only
  model-resource buffers during Session bind; output and K/V resources can be
  default-created by the Program. Terminal `llama.activation` is optional for
  the pipeline preset because intermediate stage handoff uses Session-owned
  activation workspaces released with the bound Session; callers bind activation
  only when they want the final hidden rows surfaced. This keeps the old
  no-executor constructor resource probe
  unsupported, while letting the safetensors factory construct the proof
  executor without requiring callers to manually wire a per-layer
  `tokenExecutor` or remember the preset string.
	  The same default safetensors factory now also runs through the public
	  native-backed `bind(...)` path, not only `bindHostResources(...)`: Node-WASI
	  and browser smokes compile a matching two-layer WebGPU C Program, let
	  `fromSafetensors(...)` default-create K/V and model resources, default-create
	  an ABI-sized output slot when the native executable envelope is wider than
	  the proof-family logits, enter raw exported
	  `zgml_session_execute_tokens`, and verify logits, K/V side effects,
	  `zgml_session_inspect`, ABI runtime profile counters, executor command
		  evidence, and Session-owned cleanup. The public native-backed default path
			  now covers the tiny two-layer checkpoint, materialized F16/BF16 variants,
				  tied-LM-head and named two-shard checkpoint derivation, plus compatible
				  GQA-family variants: config-derived GQA, long-context GQA, Llama 3
				  RoPE GQA, metadata-embedded-config GQA, bounded sliding-window
				  Mistral, biased/structural-biased/no-bias Qwen2 GQA, Qwen3 Q/K projection norm, and SmolLM3 NoPE
				  no-output-prefill plus decode-after-prefill proofs. Native-backed
				  safetensors `bind(...)` now also preflights the checkpoint-derived
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
  This gives browser/Wasm
  LLaMA its first
  block-level layer-chaining input/output primitive, an ABI orchestration
  proof, and a real host-resource multi-layer dataflow proof. Native C ABI
  multi-layer checkpoint binding now has a tiny two-layer LLaMA safetensors
  catalog/load/execute/KV-bind proof across C, Node, Bun, Node-WASI, and browser
				Wasm, plus native-backed tied-LM-head, named two-shard, and compatible
				GQA-family proofs and host-only three-layer, wider-shape, realistic one-layer
	  `d_head=128`, metadata/config-derived MHA/GQA/MQA,
	  GGUF-style and architecture-prefixed metadata-only GQA,
	  config-derived intermediate-size, non-default RoPE/RMSNorm-epsilon,
		  config-derived tied-LM-head, Mistral-family, biased/structural-biased/no-bias Qwen2-style,
		  explicit and structurally inferred Qwen3-style Q/K projection norm,
		  and SmolLM3-NoPE
	  Node-WASI/browser no-output-prefill and decode-after-prefill pipeline proofs.
  Default full LLaMA-family browser execution remains future work before broad
  support can be claimed. Node/WASI
  and browser smokes compare single-token cache handoff plus two-token prefill
  logits/K/V cache values, activation-output rows, activation-input logits/K/V
  rows, the two-stage repeated-layer ABI pipeline output, the host-only
  two-layer `[0, 1]` pipeline output, and the host-only three-layer `[0, 1, 2]`
  no-output-prefill plus decode output, and the host-only wider
  `[vocab=10, hidden=6, ffn=12]` two-layer output, the host-only
  realistic-head `[vocab=5, hidden=128, headSize=128, ffn=32]`
  no-output-prefill plus checked bound-logits decode proof, and the packed-head
		  `[vocab=6, hidden=8, headSize=2, ffn=8]` MHA/GQA/MQA/Mistral/Qwen2/Qwen3-QKNorm/SmolLM3-NoPE
  non-default-RoPE/RMSNorm-epsilon outputs against deterministic f32
	  safetensors payload bytes, plus a tied-LM-head two-layer
	  checkpoint whose terminal logits reuse the embedding bytes through the
	  derived `lm_head` resource through both host-only and native-backed raw ABI
		  execution, plus a named two-shard root+layer0/layer1
	  checkpoint with an index `weight_map` plus `metadata.total_size` whose Program
	  shape, model-resource upload, shard provenance, bound-logits decode, and
	  per-layer K/V cache writes match the monolithic checkpoint reference through
	  both host-only and native-backed raw ABI execution. The
  host-only two-layer proof no longer asks callers to
  separately pass the weight manifest, `vocabSize`, K/V requirements, output,
  K/V buffers, terminal activation, per-layer executor callbacks, or a preset
  string, or repeated bind-time model handle: the Program derives the shape and
  strict model manifest from checkpoint bytes, defaults the resource slots,
  infers layers from the derived K/V requirements, carries the model identity,
  and creates the current proof executor. The smokes also verify the StepParams,
  model-resource, ABI profile
  evidence, layer-derived FFN default roles, optional `lm_head.bias`
  vocab-vector validation, and default block-pipeline call evidence for
  `deviceMode`, required/max storage-buffer counts, and whether the call used
  real `GPUBuffer` storage resources or explicit mock fallback work. This makes
  mock execution, adapter-limit failure, and real browser GPU dispatch distinct
  evidence states instead of inferring them only from aggregate counters.
  They also verify cold rejection of unsupported integer/quantized
  tensor dtypes before executor resource creation, executable Q/K projection-norm
  model resources when enabled, and cold rejection of unsupported norm,
  malformed/missing Q/K projection-norm, rotary cache, and per-layer sidecar tensors when checkpoint
  tensors would otherwise be ignored by the proof executor. RoPE
  `rotary_emb.inv_freq` tensors are accepted only as
  structural auxiliaries: their shape is checked against the inferred head size,
  payload values are checked against the resolved RoPE base plus linear/Llama 3
  scaling schedule when checkpoint bytes are available, and they are excluded
  from the executable model-resource manifest.
  They also assert Program-carried hidden/vocab/layer-count and `ffnSize`
  evidence and reject conflicting tensor/option/metadata/config shape sources,
  conflicting metadata-embedded and explicit config values, conflicting direct
  metadata/config semantic hints, conflicting sliding-window sources,
  conflicting RoPE scaling kind/frequency-factor sources, plus malformed MLP
  down-projection shapes before bind.
  The unsupported config/direct-metadata matrix rejects non-LLaMA-family `model_type`/`architectures`,
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
	  run through Node-WASI/browser reference-output proofs, preventing the
	  browser/Wasm proof executor from silently running the wrong model family.
	  The same Node-WASI/browser proof now emits tiny two-layer checkpoints as F16
	  and BF16, verifies the bound model resources carry f32 executor dtype with
	  source dtype/data-length evidence, and compares block-pipeline logits/K/V
	  output against dtype-decoded reference math through both host-only resource
	  execution and the public native-backed `bind(...)` plus raw
	  `zgml_session_execute_tokens` path.
  These projection proof
  executors now prepare and cache their
  model-resource tensors during Session bind instead of searching and validating
  the weight manifest inside each token step; Node/WASI and browser smokes prove
  a layer-1 block executor bound to the one-layer tiny checkpoint rejects during
  bind with the missing layer-1 role. They also prepare the target layer's K/V
  cache descriptors and stride at bind time, leaving token steps to compute only
  the runtime byte window; an oversized K-cache descriptor whose element count
  cannot be divided by context length rejects during bind. Browser `GPUBuffer`
  projection submits now reuse static bind groups for prepared Session
  resources, K/V cache descriptors where applicable, plus executor-owned
  params/tokens upload buffers, patching only buffer contents between
  submissions instead of allocating fresh JS upload views. Projection and
  device-selection GPU submits also reuse executor-owned bind-group dependency
  scratch instead of allocating a small key array per submit. The block executor
  now also has a 64-token bounded scalar token-window kernel for no-output
  prefill stages: on real browser `GPUBuffer` runs, multi-token no-output block
  stages execute as one dispatch per layer while preserving sequential prompt
  K/V visibility inside the shader; mock fallback remains one explicit proof op
  per token. The scalar and scalar-window shaders now index score scratch by
  local attention-window offset instead of absolute context position, so
  long-context sliding-window Mistral-style proofs can keep the scalar decode
  and batched prefill paths when the active attention span fits the fixed
  scratch envelope. Logits-producing token
  windows also split terminal block stages into a batched no-output prefix plus
  one final logits dispatch, so prompt-style decode windows no longer submit
  every prefix token separately. The
	  required-GPU browser gate now requires 74 LLaMA profile labels plus
	  scalar/window block dispatch-family evidence, adding
	  strict-default safetensors execution evidence,
	  standalone Mistral sliding-window, Llama 3 RoPE, SmolLM3-NoPE block,
	  and structural Qwen3 Q/K-norm checkpoint inference
	  plus GGUF-style and architecture-prefixed metadata-only GQA
	  proofs to the prior standalone Qwen3-QKNorm block, standalone biased-Qwen2
	  block, standalone biased-Qwen2 attention, SmolLM3-NoPE attention, Llama 3 RoPE attention, and
	  zero-fallback real-`GPUBuffer`
	  evidence set; the June 22, 2026 local M5 Pro/Metal required-GPU pass
	  recorded 253 backend dispatches, 249 executor dispatches, 4
	  device-selection dispatches, 127 real `GPUBuffer` storage-mode calls, 74
	  labels, 41 output reads, 4 selection reads, 45 syncs, and zero fallback
	  ops in about 27.5 minutes. The no-adapter mock evidence covers the same
	  profile ledger with explicit mock fallback work. Node/WASI and
  browser smokes prove that dependency-exact cache contract, including K/V,
  activation-output, activation-input descriptor swaps, and the GPU-vs-mock
  command-count split, with a fake WebGPU device and the real required-GPU
  gate. The fake-device gate also asserts standalone K/V projection and
  block-window params uploads, token uploads, token GPU buffers, and bind groups
  are reused across same-capacity token windows, and that token upload/GPU
  buffers grow into reusable capacity buckets instead of reallocating on every
  slightly larger window. The same fake-GPU command-shape check now proves a
  long-context sliding-window block prefill stays on one batched backend
  dispatch and a long-context sliding-window block decode uses the bounded
  scalar GPU pipeline rather than falling back to the generic projection shader
  or per-token block submissions. The block executor result/record,
  block-pipeline record, JS runtime profile, and browser runner summary now
  expose scalar, window, and generic dispatch-family counts, so these choices
  are visible as proof evidence instead of being inferred only from aggregate
  command totals.
  Logits-producing prefix windows also reuse the block executor's borrowed
  no-output prefix context and terminal dispatch options scratch, so that
  split path does not slice tokens or allocate wrapper objects around the
  prefix and final-token dispatches. Attention token windows reuse the same
  executor-owned step-options scratch across their per-token GPU submissions,
  so non-batched attention proof windows avoid per-token dispatch wrapper
  allocation too. The block pipeline executor now reuses executor-owned
  stage and final-logits context scratch, including mutable no-output
  execution, StepParams, result records, and K/V slot-window records, so layer
  chaining does not allocate per-stage wrapper objects before dispatch.
  Standalone K/V projection mock fallback also writes K/V values through
  executor-owned Float32 scratch arrays, growing only when a larger token window
  is first seen. Embedding, RMSNorm, and activation-logits projection mock
  fallback also write bound logits through executor-owned Float32 scratch
  instead of allocating a fresh vocab buffer per fallback call. Attention and
  block projection mock fallback now also reuse executor-owned hidden, norm,
  Q/K/V, score, attention, FFN, activation, result, and final-logits scratch,
  pass exact active lengths into K/V, activation, and logits writes, and compute
  final logits only for the terminal token that actually requests them. The
  token-window
  pattern proof executor now also reuses
  slot-indexed bind groups, params GPU buffers, one JS params upload view, and
  one K/V write evidence record across submissions; the fake WebGPU smokes
  assert GPU objects are created once per write slot while uploads still occur
  on each submit, and the executor now encodes validated writes directly into
  the compute pass instead of allocating a per-submit JS command list. Its GPU
  K/V-cache path also walks bound Session K/V entries directly instead of
  staging token-window/write arrays, and bound-logits GPU writes use a direct
  single-write submit instead of a one-element staging array. Its mock fallback
  path now mirrors that shape by writing deterministic logits and K/V values
  through executor-owned Float32 scratch instead of allocating per-call pattern
  arrays or a K/V slot-window list. It still
  does not claim default full LLaMA-family
  transformer execution across layers, model families, or browser shapes.
  Those public probes also reject bad access
  descriptors before Session binding succeeds: read-only
  logits, write-only K/V caches, and read-only K/V caches cannot create a
  resource Session. The shared Wasm host wrapper now enforces the same
  role-access invariant for tiny-linear and LLaMA pre-wrapped WebGPU resource
  bindings before writing another bind descriptor, so the ergonomic
  Node-WASI/browser API cannot accept a resource table the future executor would
  have to reject. The real native wgpu executor now enforces the same
  role-access invariant at resource-table configuration time for generic
  Programs too: persistent and step-input resources must be readable, while
  step-output resources must be writable before bind groups are built. The
  current public surface therefore cannot silently fall back or accept an
  invalid resource table while that broader gate is still missing. Imported
  resource bindings now snapshot their validated descriptors before becoming
  Session bindings, so caller-side descriptor mutation after bind preparation
  cannot rewrite K/V, logits, or table-hash evidence.
  Node/Bun default smokes still exercise the compile-only resource probe library
  build; native Node/Bun GPUBuffer-like import is now a raw-handle
  descriptor contract over the same C ABI, and Node-WASI/browser Wasm now share
  a `WasmWebGpuDevice` host wrapper plus `WasmExternalResourceBridge` and
  `WasmSessionBindingBridge` for GPUBuffer-like opaque resource descriptors,
  plus `WasmWebGpuLlamaResourceProgram` for the LLaMA logits/KV resource
  session and host-token bridge registration, with storage/transfer usage
  validation, same-device provenance checks, immutable host resource
  descriptors, immutable resource tables/model manifests/bound Session resource
  evidence, host resource-table hashing, and
  browser smoke using actual `GPUBuffer`s when available. Node/WASI and
  browser smokes now split that JS-side table evidence into a physical
  descriptor hash and a semantic table hash that includes Program roles such as
  `tiny-linear.weights`, `llama.output`, `llama.k.0`, and `llama.v.0`; swapping
  roles keeps the physical descriptor hash but changes the semantic table hash,
  distinct UTF-16 role code units do not collapse through low-byte truncation,
  duplicate non-empty roles reject before binding, and the LLaMA token-executor
  hook receives those roles with its StepParams window. A separate
  model-resource table applies the same physical and
  semantic hash split to read-only weight resources; the JS host can derive
  roles, `shape`/`dtype`, byte lengths, and checkpoint data offsets from
  safetensors bytes, rejects duplicate weight roles before creating resources,
  rejects missing, zero-length-for-nonempty, overlapping, or out-of-bounds
  byte-backed `data_offsets` before deriving the executable model-resource
  manifest, including named byte-backed shard records,
  uploads each exact checkpoint tensor slice into its bound buffer, and the
  smokes bind all tiny LLaMA checkpoint tensors with byte readback so real
  browser kernels get a validated place to bind initialized model tensors. That
  resource Program can now declare `requiredModelResources` and opt into strict
  `allowExtraModelResources: false` binding, so missing tensors, dtype/shape
  mismatches, byte-length mismatches, or unexpected weights reject before a
  native Session handle is created. Byte-backed safetensors sources can now
  derive that strict requirement manifest during bind; Program default
  arbitrary model-resource maps can also derive a strict manifest at Program
  construction, so later binds reject extra roles without requiring a separate
  hand-written manifest. One-off uncataloged resource maps without a Program
  default or explicit manifest still reject when extra weights are forbidden.
  The
  model-resource Session also carries a
  separate tensor-manifest hash over role, dtype, shape, element count, byte
  length, and safetensors offsets. The proof executor returns logits/KV table
  hashes plus model-handle, model-resource-table, and model-manifest evidence for
  model-bound sessions, and the host bridge requires and validates those
  identities against the bound Session before accepting an `ok` result; omitted,
  implicit-success, or mismatched model-table/model-manifest evidence rejects
  without position advancement.
  Node/WASI and
  headless browser mock mode now also execute the claimed tiny-linear host
  Session through the raw exported Wasm
  `exportsRef.zgml_session_step(session, desc, result)` /
  `zgml_session_step_no_output` path, proving the host import bridge, resource
  descriptors, result writes, no-output profile counters, and cleanup in the
  normal portable smoke gate. The same smokes cover the wrapped-exports facade
  for those claimed tiny-linear handles and assert that
  `zgml_session_step` / `zgml_session_step_no_output` still return synchronous
  numeric status codes while writing the expected `zgml_step_result` length.
  When browser `GPUBuffer`s are available, the same
  call path uploads weights/input, dispatches WebGPU compute, writes
  `output_len`, records host-side dispatch/readback profile counters, then
  reads the output buffer explicitly and matches CPU-visible expected values.
  Host resource views now honor `byteOffset` and `byteLength` during writes,
  readback copies, and WebGPU storage-buffer binding; the portable smokes bind
  Program-sized views inside larger aligned backing resources so the
  shared-device table proves byte-range semantics rather than whole-buffer-only
  handles. The direct host-resource bridge and the tiny-linear/LLaMA Program
  resource wrappers reject unaligned storage-buffer offsets and oversized
  output/KV views before creating a Wasm resource buffer, and pre-wrapped
  Program slots reject wrong-device WebGPU provenance before binding.
  Descriptor creation and host resource-table evidence now also reject
  impossible byte ranges whenever the JS host can see the backing mock buffer or
  browser `GPUBuffer` size, so a future browser executor cannot inherit an
  oversized view that was already blessed by ABI descriptor hashing.
  Wrapper-created mock/`GPUBuffer` resources also carry owner-device provenance
  and reject cross-device descriptor wrapping, while caller-owned browser
  resources must be explicitly imported through the `WasmWebGpuDevice` wrapper
  before descriptor hashing or ABI wrapping can assign the same-device token.
  Node-WASI and browser smokes now reject unregistered caller-owned resources,
  then accept the same object after import without taking ownership. This keeps
  mock host execution and real browser `GPUBuffer` execution under one
  malformed-view boundary.
  The same raw-export path proves synchronous CPU/Wasm fallback for unclaimed
  sessions, no-output result writes with a changed host output resource, no
  implicit readback before the explicit output read, and exported
  `zgml_session_free` host cleanup for claimed browser GPU sessions, including
  claim-set cleanup so stale host handles do not intercept future reused Wasm
  handles. The browser host now also treats adapter storage-buffer limits as
  part of the executable evidence: attention and block proof executors prepare a
  Session-owned read-only model pack, patch model-tensor element offsets through
  params, and bind only that pack plus K/V, logits, and activation runtime
  resources. The pack carries a derived layout/source-descriptor hash over its
  field order, roles, offsets, dtype/shape/byte lengths, original model-resource
  descriptors, and Session model-table/manifest hashes; attention and block
  executor results must report a pack hash registered on that Session once a
  pack exists, or the host bridge rejects the result as invalid without position
  advancement and rolls back pack registrations plus Session-owned resources
  created by the rejected call. The Session owns those derived pack/dummy
  resources plus block-pipeline activation workspaces, and rejected calls destroy
  newly registered resources before restoring the previous Session and device
  ownership lists; `zgml_session_free` releases the accepted resources through
  the host runtime and forgets their long-lived device-owner references, so
  repeated browser binds do not leak packed weight or handoff buffers until
  executor shutdown. Attention now fits a
	  four-storage-buffer shape and block/pipeline execution fits a
	  six-storage-buffer shape, so normal browser `GPUBuffer` adapters with the
	  WebGPU minimum storage-buffer limit can claim real dispatch instead of falling
	  back only because the proof was over-bound. Browser smoke
  output now exposes the selected resource mode, WebGPU availability/reason,
  storage-buffer offset alignment, max storage buffers per shader stage, and a
  `canBindBlockPipeline` marker, so "no adapter", "device request failed",
  "adapter too small", and real `GPUBuffer` dispatch are separate pieces of
  evidence rather than one quiet mock fallback. Browser startup now also uses a
  single page load, bounded CDP setup/evaluation calls, and stage/profile
  console breadcrumbs, including per-family packed LLaMA stages, so a wedged
  browser WebGPU adapter request or long-running proof fails with the last
  reached stage instead of parking the validation gate. The same page can be
  focused on one packed-family LLaMA profile label through the runner's
  `--llama-profile-label=<label>` query bridge, which skips the unrelated
  LLaMA profile matrix and preserves the exhaustive default gates while giving
  browser GPU interop work a short feedback loop. A filtered
  `greedy-qwen3-qknorm-gqa-pipeline` mock run now reports only the selected
  pipeline and greedy labels, not the previous common 39-label sweep, and the
  runner rejects any unrelated focused labels. It
  also has an API-level no-fake-fallback switch: LLaMA resource Programs and
  Sessions accept `requireGpuDispatch`/`requireGpuExecution`/`noFallback`
  aliases. `WasmWebGpuLlamaResourceProgram.fromSafetensors(...)` now applies
  that strict policy by default for checkpoint-created LLaMA Programs; proof
  and no-adapter mock lanes must opt back into fallback with `allowFallback` or
  `allowMockFallback`. Node/WASI and browser smokes bind
  that strict default safetensors Program and exercise raw token execution,
  proving a no-adapter path returns `unsupported` without executor calls or
  output/K/V mutation, while a capable `GPUBuffer` path must report backend
  dispatch with zero fallback. Strict mode preflights the built-in
  executor's storage-buffer requirement before invoking it, and rejects with
  `unsupported` rather than running a mock fallback when no real `GPUBuffer`
  dispatch is possible. Successful executor results
  with backend dispatches must explicitly claim `usesGpuStorageBuffers`, and
  that claim is validated against the bound `WasmWebGpuDevice` and the
  executor's required storage-buffer count, so a custom host executor cannot
  inflate real-GPU profile evidence with counters alone or by omitting the
  storage-mode claim. The explicit device-selection
  helpers apply the same rule: direct host-logits argmax/sampling, `gpu:false`
  direct sampling, execute-and-select calls, and device generation calls reject
  before token execution, device selection, generated-token writes, profile
  work, or resource mutation. Non-device full-logits argmax/sample,
  execute-and-select, and generate-and-select helpers now reject the same way on
  GPU-required Sessions instead of sneaking through CPU selection after a
  resource execution. Node/WASI and browser smokes prove this
  GPU-required mode keeps logits and K/V resources unchanged, records no
  executor calls, selection calls, or fallback ops, and still allows real
  six-storage-buffer dispatch when the adapter can support it. It
  also exports
  aggregate LLaMA resource-session profile evidence from the browser page:
  profile count,
  total backend dispatch count, executor backend dispatch count,
  device-selection backend dispatch count, fallback-op count, command count,
  scalar/window/generic dispatch-family counts, block-pipeline storage-mode
  call counts split into real `GPUBuffer`, mock, adapter-limited, and unknown
  states, max required storage-buffer count, output reads, small device-selection
  result reads, syncs, and profile labels. The
  required-GPU browser runner now rejects any real-`GPUBuffer` pass that lacks
  LLaMA profile evidence, records zero executor backend dispatches, records any
  LLaMA fallback ops, has mismatched total-vs-executor-plus-selection dispatch
	  evidence, lacks scalar/window block dispatch-family evidence, lacks
	  direct GPU-storage block-pipeline call evidence, reports any mock,
	  adapter-limited, or unknown storage-mode block-pipeline calls, lacks the
	  six-storage-buffer requirement marker, omits representative storage-mode
	  labels across tiny, sharded, dtype-materialized, wider, realistic-head,
	  packed-family, and tied-head default pipeline paths, lacks
	  device-selection readback evidence, or omits any of the
	  74 current LLaMA labels: host-token/model-token hooks, strict-default
	  safetensors execution, embedding/RMSNorm/KV/
	  attention/block projection probes, block-pipeline and ergonomic two-layer
	  execution, full-logits generation, device-argmax generation, device-sampled
	  generation, grouped-family greedy generation, sharded, multi-layer, F16/BF16 materialized-checkpoint,
	  tied-LM-head, wider, realistic-head, individual metadata-MHA,
	  GGUF-style metadata-GQA, architecture-prefixed metadata-GQA, config-GQA,
	  long-context-GQA, config-JSON-MQA, Llama3-RoPE-GQA, metadata-config-GQA,
	  Mistral-GQA, long sliding-window Mistral-GQA, biased-Qwen2-GQA,
	  structural-biased-Qwen2-GQA, no-bias Qwen2-GQA, Qwen3-QKNorm-GQA,
	  structural-Qwen3-QKNorm-GQA,
	  standalone K/V-QKNorm, standalone attention-QKNorm,
	  standalone sliding-window attention, standalone Llama3-RoPE attention,
	  standalone SmolLM3-NoPE attention, standalone biased-Qwen2 attention,
	  standalone biased-Qwen2 block, standalone Qwen3-QKNorm block,
	  standalone sliding-window block, standalone Llama3-RoPE block,
	  standalone SmolLM3-NoPE block,
	  SmolLM3-NoPE-GQA, GGUF-metadata SmolLM3-NoPE-GQA, and
	  realistic-`d_head=128` GQA labels. It also rejects duplicate labels or a
  profile count that does not match the label list.
  Wrapper-created
  LLaMA resources still carry CPU shadows and bidirectional copy usage so mock
  fallback, sentinel initialization, and explicit readback all remain legal and
  visible. Factory-created safetensors resource Sessions now also expose the
  ergonomic JS shape the browser API should keep: `advanceTokens(tokens)` runs a
  no-output token window, `step(token)` defaults to bound logits, and the
  returned result includes an explicit async readback of the Session-bound
  logits only when logits are requested. The same resource Session now mirrors
  the native selection subset with `argmax()`, `argmaxDevice()`,
  `sample({ topK, seed, temperature })`, `executeTokensArgmax(tokens)`,
  `executeTokensArgmaxDevice(tokens)`, `executeTokensSample(tokens, ...)`,
  `executeTokensSampleDevice(tokens, ...)`, `stepArgmax(token)`,
  `stepArgmaxDevice(token)`, `stepSample(token, ...)`,
  `stepSampleDevice(token, ...)`,
  `generateTokensArgmax(prompt, maxTokens)`,
  `generateTokensArgmaxDevice(prompt, maxTokens)`,
  `generateTokensArgmaxInto(prompt, outputTokens)`,
  `generateTokensArgmaxDeviceInto(prompt, outputTokens)`,
  `generateTokensSample(prompt, maxTokens, ...)`, and
  `generateTokensSampleDevice(prompt, maxTokens, ...)`,
  `generateTokensSampleInto(prompt, outputTokens, ...)`, and
  `generateTokensSampleDeviceInto(prompt, outputTokens, ...)`: full-logits
  selection remains usable and counted as output-read/sync work, while the
  explicit device paths run cached browser WGSL reductions over bound logits in
  real `GPUBuffer` mode and read back only `{token, logit}` for argmax or the
  top-k `{token, logit}` pairs for sampling; top-k=1 device sampling shares
  the argmax selector instead of running the heavier top-k shader. Node/WASI
  and browser smokes now also run ergonomic greedy generation through every
  packed grouped-attention default safetensors Program, comparing generated
  tokens, terminal logits, K/V writes, ABI profiles, and executor evidence to
  the reference math for MHA/GQA/MQA/Mistral/Qwen2/Qwen3-QKNorm/SmolLM3-NoPE
  family shapes. The argmax
  selector now validates GPU results before returning them, and Session-level
  device-selection normalization rejects injected selectors that return
  out-of-vocab tokens, non-finite logits, invalid work counters, zero-work
  successes, or GPU dispatch evidence without a `WasmWebGpuDevice`-owned path.
  The argmax selector has fake-device
  evidence that its params upload view, params buffer, result buffer, and bind
  group are reused across repeated selections. The top-k selector has matching
  evidence for same-or-smaller `topK` requests, with only result capacity growth
  rebuilding the result resource and bind group; after GPU readback it samples
  directly from the `{token, logit}` view rather than allocating candidate and
  weight arrays. The shared full-logits/mock top-k sampler now also avoids a
  per-selection candidate-object list and weight array by keeping compact
  primitive token/logit arrays, using the same two-pass scalar thresholding
  shape, and reusing Session-owned scratch arrays across same-or-smaller
  `topK` selections. Node/WASI and browser smokes prove scratch reuse across
  direct sampling, sample-device CPU/mock fallback, and execute-and-sample,
  while GPU-required sessions reject those fallback routes before execution,
  generation, or selection counters move. Capacity growth allocates only when
  `topK` grows. Full-logits
  selection readback now also has a caller-owned path: `WasmWebGpuDevice`
  exposes `readFloat32Into`, LLaMA resource Sessions reuse one Session-owned
  logits scratch buffer for internal argmax/sample and execute-and-select
  fallback reads, `includeLogits` still returns fresh public logits, and
  `outputInto` fills caller-provided storage. Node/WASI and browser smokes
  prove that stale caller `logits` cannot override freshly executed bound
  output in execute-and-device-select paths. Browser `GPUBuffer` readback also
  exposes `readBytesInto`, reuses a device-owned staging buffer across
  sequential f32 and byte reads, growing it only when a larger read is
  requested and destroying it with the device; overlapping reads get a
  temporary staging buffer instead of corrupting the cached one, and that
  temporary buffer is released from the device owner list as soon as the
  overlapping read settles. Device teardown unregisters remaining device-owned
  resources as well as destroying them, so stale handles do not survive a
  whole-device cleanup. Wrapper-created resources now carry an owner-device
  map too, so executor fallback-owned cleanup can unregister host-resource
  handles even when it only sees the resource object. LLaMA resource Sessions
  now mirror tiny-linear deterministic lifecycle cleanup: direct JS
  `free()` / `dispose()` / `Symbol.dispose` releases native-backed Sessions
  through the Wasm export, tears down host-owned resources, unregisters runtime
  lookup entries, and clears host Session claims; Node/WASI and browser smokes
  prove stale direct-freed native and direct-disposed host-only handles are
  unclaimed at the host callback boundary. Device argmax and top-k selectors read their tiny
  result records into selector-owned byte scratch, forget selector-owned GPU
  result buffers from the device owner list and host-resource handle table on
  dispose, and top-k does the same for the old GPU result buffer when result
  capacity grows, with top-k growing byte scratch only when result capacity
  grows. Those destroyed selector result objects cannot be rewrapped into fresh
  host descriptors or used as WebGPU write/read targets after their handles are
  unregistered, so cleanup is enforced by both handle lookup and object-lifetime
  validation.
  Node/WASI and browser fake-GPU smokes prove same-capacity reuse, exact growth,
  map/unmap balance, and cleanup.
  Node/WASI and browser smokes prove the helper paths against the same two-layer
  block-pipeline reference math, seeded top-k sampling, generated tokens, final
  logits, position updates, K/V resource side effects, output-read/sync
  counters, small device-selection read counters, and ABI runtime profile
  evidence as the raw token descriptor path. The same
  helper path now preflights the whole generation window before executing:
  prompt length must fit the Session's max token-window envelope, prompt length
  plus generated continuation must fit the compiled context, and over-context
  or over-window greedy/sampled generation returns `shapeMismatch` with zero
  generated tokens, unchanged position, untouched output/K/V resources, and
  zero profile calls/dispatches/fallbacks/syncs. Hot generation loops reuse one
  Session-owned single-token continuation window and one Session-owned
  call-options scratch object after the prompt step, borrow prevalidated token
  windows, including caller-owned prompt token buffers with an explicit active
  length, into the token bridge, reuse one mutable execution record, one mutable
  StepParams object, and one mutable token-executor context, and skip diagnostic history
  arrays for that internal lane instead of allocating fresh token, execution,
  StepParams, context, and options objects for each generated token or generation
  call. Public scalar decode/select
  wrappers (`advance`, `step`, `stepArgmax`, `stepArgmaxDevice`, `stepSample`,
  and `stepSampleDevice`) now also borrow a Session-owned one-token scratch
  window into the token bridge; native Node/Bun scalar select helpers use the
  same shape plus reusable scalar option scratch instead of allocating wrapper
  `[token]` arrays or per-call option spreads, and native `advance(token)` now
  uses the C ABI scalar no-output call directly. The
  browser/Wasm path reuses the Session-owned execution record, StepParams,
  executor context, and normalized executor-outcome record, and skips diagnostic
  histories without mutating caller options, so repeated host-driven decode does
  not allocate wrapper one-token arrays, bridge records, or executor-result
  wrappers before execution. `advanceTokens` uses a
  Session-owned no-output execution override instead of spreading caller
  options, preserving caller-owned option objects while forcing logits
  output/readback off and reusing the same hot bridge scratch for prompt
  advancement. The generation helpers now also
  bypass the public execute-and-select wrappers in that hot loop: they fill a
  single Session-owned private selection/result scratch object through
  `executeTokens*Into` helpers, while the public one-shot helpers still return
  fresh ergonomic result objects. Those private helpers force logits output with
  a scalar output-policy override and pass the same mutable call-options object
  through sync and decoded execution, avoiding per-token option-wrapper spreads.
  Public JS token execution and generation entry points now also accept borrowed
  capacity-sized token buffers plus explicit active-length options across the
  native Node/Bun and portable Node-WASI/browser wrappers. Those wrappers thread
  that active count into the C/Wasm token descriptors and shared execution
  scratch, and the hot generation path strips prompt-only length metadata from
  continuations. The token execution validator now indexes exactly the active
  `tokensLen` window instead of iterating the whole token container, so borrowed
  scratch buffers can be larger than the active window without checking stale
  capacity tail values or allocating an iterator. Standalone embedding, RMSNorm,
  activation-logits, K/V, attention, and block projection executors now use the
  same active token window for last-token lookup, fallback loops, GPU submission
  loops, terminal-output decisions, and diagnostic token records; Node, Bun,
  Node/WASI, and browser smokes poison the inactive tail and iterator for public
  token execution, generation prompt buffers, and projection seed calls to prove
  the inactive capacity is ignored.
  The block pipeline executor also reuses stage/final context scratch while
  composing layer executors, including its no-output execution, StepParams, and
  result records for non-terminal stages, final-logits projection, and the
  top-level Session executor result before normalization; standalone K/V
  projection now writes into the Session executor-result scratch too; and the
  K/V, attention, and block projection executors also write K/V cache
  slot-window views into executor-owned scratch. Standalone attention and block
  projection mock fallback now reuse executor-owned math/result scratch across
  cache-handoff decode and prefill, with capacity-sized buffers guarded by
  exact active write lengths. Ordinary token execution now keeps diagnostic
  histories behind a bounded Session retention window, and
  `diagnosticHistoryLimit: 0` preserves the latest decoded execution and
  StepParams evidence while retaining no history arrays. Retained diagnostic
  entries are immutable active-window snapshots, so caller-owned token buffers
  and reused StepParams scratch cannot rewrite earlier evidence. Node/WASI and browser
  smokes prove those bounded diagnostics plus the hot generation and pipeline
  dispatch objects are reused without mutating caller options while final-only
  logits and sample seed progression are preserved. The realistic-head
  `[hidden=128, headSize=128]` browser proof now emits checked bound logits
  after a 66-token long sliding-window prefill instead of only K/V side
  effects: wide block-pipeline proof stages use the bounded scalar/window GPU
  block kernels for proof-scale shapes, then a separate activation-logits
  dispatch, preserving real `GPUBuffer` dispatch and zero fallback evidence
  without the old recursively recomputed fused shader. The packed
  MHA/GQA/MQA/Mistral/Qwen2/Qwen3-QKNorm/SmolLM3-NoPE family proofs now assert
  scalar/window/generic dispatch-family splits through Session profiles and
  executor call records too, so broader checkpoint-family coverage has the same
  fast-path evidence as the focused realistic-head proof. This closes
  the wrapper-only browser execution gap for tiny-linear
  WebGPU and the minimum-limit gap for the tiny LLaMA attention/block proofs,
  but full/default LLaMA-family browser WebGPU execution remains the next real
  leap before broad support can be claimed.

The tiny C/Node/Bun/Wasm prototype has therefore stopped being only ABI
evidence. It is already exercising the local-LLM engine shape users would
embed:

```text
Program -> Session -> StepParams -> execute
```

The packaging direction is now explicitly TS-first: authored source under
`src/ts/**` is guarded as TypeScript source only, with package artifacts
generated by the build rather than mirrored by hand. The Bun/native adapter is
now included in the normal `tsconfig.source.json` strict build rather than
living behind a source-build exclusion. Shared facade contracts describe
structural runtime truth at composition seams: readonly shape evidence,
adapter-neutral callback slots, richer KV-cache records, and one-time casts for
large factory-produced surfaces. This makes `npm run build:ts-source` prove the
Node, Bun, and shared frontend authored TS surfaces together before tsdown emits
runtime/declaration artifacts.

The remaining leap is not more lifecycle naming. It is making full/default
LLaMA-family browser WebGPU execution and checkpoint-family coverage beyond the
	current materialized F16/BF16, tied-LM-head, individually required
	MHA/GQA/MQA/Mistral/Qwen2/Qwen3-QKNorm/SmolLM3-NoPE grouped-attention, native-backed
	long-context GQA, bounded sliding-window Mistral, realistic `d_head=128` GQA,
	and realistic-head proofs compute real outputs
while preserving the same evidence, no-allocation hot path, and
no-fake-fallback contract.

June 27, 2026 follow-up: the default ggml smoke now measures the promoted
Q8 semantic throughput prefill lane instead of the older projection-row-chain
command lane. The latest accepted smoke records Q8 prompt at `4119.59 tok/s`,
`42.0%` of llama.cpp in the quick local run, `242` dispatches, `121` commands,
and zero fallback on `metal scheduled prefill semantic throughput candidate`.
The explicit Q8 projection-row-chain command smoke remains available and writes
to `bench-results/q8-command-smoke/`, so it can keep proving the legacy command
shape without taking over the headline `ggml-smoke-results` status. The next
real performance target is therefore not selecting the semantic lane; it is
making that semantic-with-input command faster, especially the width-parallel
kernel/storage model behind `semantic_width_parallel_kernel`.
The status loop now exposes that active default bottleneck directly:
`semantic_with_input=30`, `semantic_with_input_dispatch=150`, and
`semantic_with_input_extra=120` in `ggml-smoke-results`, while `q8_current`
targets `semantic_ffn_sublayer_with_input_row_chain:150` and routes the fallback
current-smoke case to `semantic_with_input_width_parallel_kernel`. That keeps
the next-perf loop from falling back to vague command-pressure inspection after
the default lane has already moved to the 121-command semantic path.
A follow-up exact no-build microscope run kept the 9-op bridge healthy at
`bridge_ffn=2.89x` with `runtime_backend_dispatches=3`, `width_parallel=1`,
and `runtime_bytes=9216`. The exact 14-op input-bridge run stayed correct and
selected `absorbed=2.51x`, but still reported the five-dispatch decomposition
(`row_chain=2,pair=1,tail=2`, `decomposed_extra=4`, `spilled_input=576`).
Those are the local numbers the next semantic-with-input kernel needs to beat;
the rejected fewer-dispatch shape is not enough unless it also improves
throughput.
