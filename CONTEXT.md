# zgml Context

zgml is a TS-first ML library with a native Zig runtime substrate. Its public
package Interface is meant to feel like a compact PyTorch replacement for
JS/TS: tensor ops, graph execution, canonical `nn` layers, optimizers, training
helpers, and a narrow `llm` facade. The product library is authored once in
`src/ts/**` and emitted with `tsdown` for Node, Bun, browser-safe frontend
subpaths, declarations, and package artifacts. Zig owns kernels, buffers,
Programs, Sessions, ABI boundaries, backend hooks, and low-level native APIs;
it is not a second package frontend that must be kept in sync with TypeScript.
Contract compatibility matters at the native boundary, but source-level product
sync is architectural debt: align ABI/runtime behavior with tests, not by
mirroring tensor/module/training APIs in two languages.
Short form: TS is the ergonomic product API, `tsdown` is the artifact fan-out,
Zig is the required core runtime, and contracts/tests keep that boundary honest.
Runtime manifests make the rule explicit: `productSourceOfTruth` is
`ts-api-zig-core`, native product policy is `required-core`, and Zig helper
modules are runtime substrate rather than a second package frontend.
Public TS product namespace manifests inherit those facts through
`src/ts/internal/product_manifest.ts`, so `tensor`, `nn`, `loss`, `optim`,
`train`, `compile`, `program`, `session`, and other package subpaths do not
hand-copy ownership policy.
Handwritten JS/CJS/MJS examples may exist as consumer harnesses over package,
dist, or wasm artifacts, but they are not a product implementation lane.

## Domain Terms

- **Primitive IR**: the small graph operation set in `src/op.zig`. User-facing
  tensor sugar should decompose into these primitives unless a new primitive
  materially improves expressiveness or performance.
- **Tensor Interface**: the chainable API exposed by `Tensor(T)` and
  `ComputeGraph(T)`. Tensor methods may be rich; the primitive IR should remain
  small.
- **Lazy Tensor IR**: the normalized tensor/UOp graph produced by ordinary
  Tensor and `nn` code before backend lowering. Eager execution remains the
  reference/debug mode, but compiled execution should come from this lazy graph
  rather than from one-off model matchers. The Zig compile path now has a shared
  internal `TensorProgramIr` Module in `src/tensor_program_ir.zig` that
  materializes an owned `IrValue` shape/layout table, an `IrOp` table, and flat
  input-edge lists before `DeviceProgram` lowering. Ordinary node ops now lower
  through normalized `IrOp` / `ValueId` inputs and outputs. Supported fusions
  now lower through normalized IR evidence too: elementwise-chain fusion owns
  an IR sub-tape, layer-norm fusion lowers from `IrOp` attrs, and log-softmax
  fusion owns a range of normalized sub-`IrOp`s. Unsupported fusion families
  still reject before backend lowering instead of preserving source payloads as
  hidden execution adapters. JS/TS module traces now retain the same kind of
  value/edge evidence in their public Tensor Program IR: a frozen value table,
  input/output value IDs, dense row-major storage-layout evidence for each
  value, and per-op input/output edge IDs that carry into KernelPlan evidence.
  The next locality win is to make the frontend
  Trace-to-Program Compiler and Kernelizer consume one shared native Module, so
  `ComputeGraph` fusion details and JS descriptor details stop leaking past the
  normalized compile Interface.
- **Kernelizer**: the deep compiler Module that schedules Lazy Tensor IR into
  a small list of executable kernels or backend commands. Elementwise fusion,
  views, reductions, matmul epilogues, broadcast materialization, and recognized
  module traces should concentrate here for locality and leverage. The Zig
  backend now has an explicit `Kernelizer` Interface that produces `KernelPlan`
  artifacts and backend execution plans; raw command-stream and execution-plan
  construction are internal to that Module.
- **KernelPlan**: the scheduled command/kernel list plus memory and buffer
  layout produced by the Kernelizer. It is the cold artifact that becomes a
  `ProgramStencil`; it should be inspectable and backend-independent enough for
  CPU, Metal, WGPU, Wasm, and stencil evidence to agree on shape.
- **PyTorch-like Frontend**: the public TS-first `Tensor`, `nn`, `loss`,
  `optim`, and `train` Interface for ordinary package users, with native
  Program/Session/runtime handles where speed or deployment needs them. It should be eager, differentiable,
  discoverable, and boring to use; executable `Program` support deepens this
  frontend, but backend/stencil vocabulary should not leak upward.
- **StepParams executable preflight**: the `Session.step` / `execute` hot-call
  gate. `Session.requireCanExecuteStepParams(...)` and
  `stepParams.requireCanExecuteStepParamsCompatibility(...)` return frozen
  compatibility evidence for executable calls or throw a structured
  `not-executable-step-params` diagnostic; stronger hot/allocation/readback
  assertions stay available for stricter performance policies.
  `zgml/step_params` also exposes standalone compatibility evidence predicates,
  assertions, and require helpers, so accepted and rejected StepParams evidence
  can be validated after crossing host, FFI, or serialization boundaries before
  stricter executable/hot-path assertions run.
- **Execution plan preflight**: the checked Program/Session runtime gate.
  `Program.executionPlan(...)` and `Session.executionPlan(...)` remain
  descriptive frozen evidence, including rejected module or StepParams plans.
  `Program.requireExecutionPlan(...)` / `require_execution_plan(...)` and
  `Session.requireExecutionPlan(...)` / `require_execution_plan(...)` assert the
  plan is executable before binding, stepping, or handing a call to host code.
  The TS-owned `program`, `session`, and `inspection` package subpaths also
  expose standalone execution-plan evidence predicates and assertions
  (`accepts*ExecutionPlan`, `require*ExecutionPlan`, and signature matchers) so
  Node, Bun, FFI, and serialized evidence flows can validate a frozen plan
  without rebuilding it through the original object facade.
- **Program binding preflight**: the bind-time gate between compiled Program
  evidence and live Session state. `Program.bindingPlan(...)` exposes frozen
  host/native/rejected binding evidence, while `Program.requireBindingPlan(...)`
  / `require_binding_plan(...)` returns the accepted plan or throws before
  buffer ownership and Session creation begin. The friendly `nn` namespace also
  exposes `nn.requireBindingPlan(...)` / `require_binding_plan(...)` so
  ModuleBindings can be asserted before they reach a Program. The TS-owned
  `program`, `nn`, and `inspection` package subpaths expose standalone
  binding-plan predicates/assertions and signature matchers too, so
  ProgramBindingPlan and ModuleBindingPlan evidence can be checked after it
  crosses host, FFI, or serialization boundaries.
- **Runtime profile evidence aliases**: Program and Session runtime-profile
  snapshots, signature matching, resets, and no-fallback/no-sync/hot assertions
  expose camel and snake names from the same TS-owned helpers. The inspection
  namespace also exposes standalone runtime-profile predicates/assertions and
  signature matchers, so hot-path profile evidence can be validated after
  crossing host, FFI, or serialization boundaries without a live Program or
  Session facade.
- **Tensor shape preflight**: root `hasShape(...)` / `requireShape(...)` and
  static `Tensor.hasShape(...)` / `Tensor.requireShape(...)` are TS-owned
  shape gates over the shared tensor root surface. `hasShape` narrows existing
  Tensor values when a literal shape is proven, while `requireShape` returns a
  `Tensor<S>` or throws before compile, bind, or FFI handoff. This keeps
  compile-time shape evidence and runtime shape checks in one public vocabulary.
- **Session call-profile aliases**: Session helper-family counters expose
  camel and snake names (`sessionCallProfile` / `session_call_profile`,
  `matchesSessionCallProfileSignature` /
  `matches_session_call_profile_signature`, `resetSessionCallProfile` /
  `reset_session_call_profile`) from the shared TS facade, so generic and LLaMA
  sessions share one evidence vocabulary without hand-maintained policy copies.
  The `zgml/session` and `zgml/inspection` subpaths also expose standalone
  call-profile predicates, assertions, and signature matchers, so frozen
  helper-family evidence can be validated after crossing host, FFI, or
  serialization boundaries without a live Session facade.
- **Module compile preflight**: the cold gate before compile/bind work claims a
  native path. `compile.requireCompileSupport(...)` remains the generic root
  helper, and `nn.requireCompileSupport(...)` plus module instance
  `requireCompileSupport(...)` expose the same checked support evidence through
  the PyTorch-like module vocabulary. `requireCompilePlan(...)` /
  `require_compile_plan(...)` and `assertCompilePlan(...)` expose the richer
  frozen `zgml.nn.compile-explanation` gate before compile/bind, throwing with
  the same unsupported reason vocabulary when the graph cannot take a native
  path. `zgml/compile`, `zgml/nn`, and `zgml/inspection` also expose standalone
  Module compile-plan predicates/assertions/signature matchers, so frozen
  explanation evidence can cross package, serialization, or FFI boundaries
  without being recomputed.
- **Trace-to-Program Compiler**: the lowering module that should turn the
  normalized Zig or JS/TS module/tensor trace into Lazy Tensor IR, then through
  the Kernelizer into a backend `DeviceProgram`/`ProgramStencil` shape. It is
  the desired deep seam after `compileSupport()`: adding a frontend op should
  teach one compiler path how to lower and schedule the op, not add one more
  hand-matched model pattern. JS/TS `nn.Sequential` now uses this path first for
  multi-layer module graphs; direct tiny model handles remain compatibility
  adapters rather than the normal frontend shape. The public JS/TS trace,
  Tensor Program IR, KernelPlan, and module-compatibility diagnostic Interfaces
  name the current op/kernel/code vocabularies and discriminate records by those
  tags; Tensor Program IR and KernelPlan evidence also expose value IDs,
  storage-layout evidence, and op-edge IDs, and Tensor Program IR ops now carry
  normalized `attrs` instead of extending trace records. The published TS Interface now names
  `ModuleTensorProgramIrOpKind` independently from `ModuleTraceOpKind`, so
  callers can branch on compiler evidence without accepting arbitrary string
  records, optional-field probes, or trace-record inheritance. Module
  compatibility now compares a canonical Tensor Program IR signature before
  KernelPlan signatures, making normalized value/op/layout evidence load-bearing
  for `Program.acceptsModule(...)` and `Program.bindModule(...)` instead of only
	  inspectable. Tensor Program IR construction, freezing, normalized attrs, value
	  layout evidence, and canonical signatures now live in
	  `src/ts/runtime/tensor_program_ir.ts`, emitted to
	  `dist/runtime/tensor_program_ir.cjs`. KernelPlan lowering, freezing,
	  public projection, layout/signature policy, and descriptor-ready native op
	  scheduling now live in `src/ts/runtime/kernel_plan.ts`, while native
	  module descriptor shape, descriptor signatures, kernel-name evidence, and
	  ABI-packable descriptor field validation live in
	  `src/ts/runtime/native_kernel_contract.ts`; both are emitted under
	  `dist/runtime/**`. The obsolete
	  `js/shared_kernel_plan.cjs` source-checkout compatibility shim has been
	  removed so KernelPlan policy has a single TS-authored source.
	  Compiler-signature derivation and evidence completion now live in
	  `src/ts/runtime/compiler_signatures.ts`, emitted to
	  `dist/runtime/compiler_signatures.cjs`, keeping trace compiler, Program facade,
	  root `nn.compilerSignatures(...)`, and compatibility preflight policy on the
	  same canonical record,
	  JS/TS public-safe compiler diagnostic construction/freezing now lives in
	  `src/ts/runtime/compile_diagnostics.ts`, emitted to
	  `dist/runtime/compile_diagnostics.cjs`; Trace-to-IR orchestration and
	  compile-evidence projection live in `src/ts/runtime/trace_compiler.ts`,
	  emitted to `dist/runtime/trace_compiler.cjs`; Sequential constructor
	  validation, nested module flattening, trace shape normalization, tiny-linear
	  compatibility detection, module trace shape inference, trace-record
	  construction, support-analysis orchestration, support-detail projection, and parameter trace projection now live in
	  `src/ts/runtime/module_compiler_policy.ts`, emitted to
	  `dist/runtime/module_compiler_policy.cjs`; the obsolete
	  `js/shared_module_compiler.cjs` compatibility shim has been removed, and
	  `src/ts/shared_frontend.ts` re-exports the compiler policy,
	  parameter-packing helpers, and module facade policy; root
	  `nn.trace`/`nn.compileSupport`/`nn.compile`/`nn.bindParameters`/
	  `nn.placeParameters` facade policy now lives in
	  `src/ts/runtime/module_facade.ts`, emitted to
	  `dist/runtime/module_facade.cjs`; ModuleBinding metadata annotation,
	  Sequential parameter packing, `nn.placeParameters(...)` placement policy, and
	  Program.bind compatibility validation now live in
	  `src/ts/runtime/module_bindings.ts`, emitted to
	  `dist/runtime/module_bindings.cjs`. Program/module compatibility comparison
	  lives in `src/ts/runtime/module_compatibility.ts`, emitted to
	  `dist/runtime/module_compatibility.cjs`, behind
	  the shared frontend facade.
	  Eager JS/TS `nn` module classes and namespace vocabulary are now TS-authored
	  under `src/ts/nn/**`; the obsolete `shared_nn_modules.cjs` hook-injection
	  shim has been removed, and `src/ts/shared_frontend.ts` owns the small
	  placement and scalar wrapper injection around TS-authored `nn` helpers.
	  The remaining locality move
  is to make this JS/TS compiler seam share more of the native
  `TensorProgramIr` / `Kernelizer` contract instead of maintaining a parallel
  JavaScript lowering vocabulary.
- **Shared JS/TS Frontend Core**: the reference product implementation of `Tensor`,
  autograd, `nn`, `loss`, `optim`, `train`, state dictionaries, trace metadata,
  and compile-support analysis shared by Node and Bun. Host adapters should
  only own runtime loading and native FFI differences. The intended end state is
  TS-authored package entrypoints (`src/ts/index.ts`, `src/ts/node.ts`,
  `src/ts/bun.ts`, and `src/ts/browser.ts`) emitted through `tsdown` or an
  equivalent. TypeScript is the only authored JS/TS host implementation;
  CJS/JS files are package output, adapter loading glue, or temporary migration
  provenance, not a second source tree to keep in sync. Public declarations and
  runnable JS are emitted from the TS implementation. Zig is the native
  kernel/runtime substrate and may expose its own native-friendly API, but it is
  not a second package frontend that TS must mechanically mirror. The
  package-facing boundary is `dist/**`. The architecture question is no longer
  "how do we keep TS, generated JS, declarations, and Zig frontend surfaces in
  sync?" The answer is to remove that synchronization requirement: author the
  product library once in TS, let `tsdown` emit package artifacts, and use Zig as
  the native execution substrate reached through Program/Session/runtime
  primitives. The target is one product language and many runtime targets, not
  multiple frontend source trees that need reconciliation. `tsdown.config.mjs`
  is part of that contract now: it must discover package runtime entries from
  `src/ts/**`, emit declarations from the same TS product tree, and keep
  host-specific exceptions narrow instead of hand-listing another frontend.
  The typed root contract now lives in `src/ts/public_api.ts`, emits
  `dist/public_api.d.cts`, and the package metadata for root, `zgml/node`, and
  `zgml/bun` points directly at that emitted declaration surface. Root
  declaration shims (`index.d.ts`, `node.d.ts`, `bun.d.ts`) are not part of the
  package architecture; the package-artifact guard rejects regressions back to
  root declaration re-exports or source-tree declaration exports.
  `PublicApiContractManifest` is a type-only marker in that contract, not a
  runtime value, and proves the root declarations are TS-owned `tsdown` output.
  The TS source
  architecture guard also rejects any new authored JS/CJS/MJS runtime files
  under `src/ts/**` and allows only narrow ambient host-interop `.d.ts`
  declarations there; product declarations must come from TS implementation
  types and `tsdown` output. Package root runtime resolution now points directly at the TS-built `dist/node.cjs`
  artifact instead of a shipped `index.js` bridge. The first TS package spine
  now exists and is smoke-tested through the `tsdown` `dist/index.cjs`
  frontend artifact plus TS-authored Node/Bun adapter-spine modules; those
  entrypoints expose
  one TS frontend manifest and adapter evidence that says host adapters own
  native loading, not frontend policy. Native library extension, filename/path
  resolution, load-info construction, and missing-library diagnostics now live in
  `src/ts/adapters/native.ts`, emitted to `dist/adapters/native.cjs`.
  `src/ts/adapters/node.ts` and `src/ts/adapters/bun.ts` now own generated
  host-adapter evidence and adapter re-export shape, so generated
  `src/ts/node.ts`, generated native-free `src/ts/bun.ts`, and generated
  Bun-native `src/ts/bun_native.ts` are package-spine aggregators over shared
  frontend plus host adapter/runtime modules. Both legacy Node and Bun adapters call that
  TS-authored adapter policy. The native adapter module owns the Node source-tree `koffi`
  fallback path, while the Node adapter keeps the host `require()` call.
  Concrete native runtime load/contract fallback policy is shared in
  `src/ts/adapters/concrete_runtime_loader.ts`; Node and Bun concrete loaders
  now only contribute host-specific runtime kind, bridge labels, and policy
  literals while the shared concrete-runtime host surface owns candidate
  lookup, require/assert choreography, and manifest construction.
  `examples/types/package-spine-smoke.ts` now typechecks the dist-backed
  frontend, friendly subpaths, adapter evidence, and wildcard package
  declarations directly through package imports rather than
  `../../js/generated/**`, so package declarations are becoming load-bearing
  instead of only emitted artifacts. `scripts/check_ts_source_architecture.cjs` rejects
  public type examples that import repo-only `js/` artifacts. The package now
  exposes `zgml/frontend` as a native-free
  TS-authored frontend subpath backed by `dist/index.cjs` and
  `dist/index.d.cts`, so callers can import shared frontend/runtime
  policy without loading the legacy Node FFI entrypoint. It also exposes
  `zgml/browser` as a TS-authored browser-safe frontend subpath backed by
  `dist/browser.cjs` and `dist/browser.d.cts`; its `browserManifest` proves
  `nativeLoader: false`, `adapterRole: "browser-safe-frontend"`, and
  `runtimePath: "Program -> Session -> StepParams"` so browser packaging is a
  real entrypoint, not an alias to the Node bridge. It also exposes
  `zgml/tensor`, `zgml/nn`, `zgml/compile`, `zgml/inspection`, `zgml/model_source`, `zgml/program`, `zgml/session`,
  `zgml/step_params`, `zgml/native_buffer`, `zgml/program_device`, `zgml/checkpoint`, `zgml/loss`, `zgml/optim`,
  `zgml/train`,
  `zgml/adapters/*`, `zgml/core/*`, `zgml/runtime/*`, `zgml/nn/*`, and
  `zgml/train/*` through `dist/**` package artifacts, so serious TS
  users can import the tensor, nn, compile, inspection evidence, model-source routing, Program, Session, StepParams, NativeBuffer/ProgramDevice policy, checkpoint, loss, optimizer, train, shape/runtime/module
  building blocks directly from package exports instead of reaching into
  `js/generated/**`. Adapter
  evidence subpaths remain native-free dist artifacts, separating host ownership
  evidence/path policy from the concrete FFI wrappers. The package root and
  `zgml/node` now resolve through TS-built `dist/node.cjs` when the package
  artifact is present; that bridge now loads local sibling/package-built
  `tsdown` concrete runtime artifacts such as
  `dist/adapters/node_ffi_runtime.cjs` and no longer falls back through
  `js/generated/**` or a legacy root Node shim.
  `scripts/check_ts_source_architecture.cjs` guards that boundary so the TS Node
  bridge uses `src/ts/adapters/node_concrete_runtime.ts` to load
  `tsdown`-emitted concrete runtime artifacts instead of reviving the old root
  shim path or a generated-source compatibility lane. That typed loader owns
  the native API contract assertion and shared host runtime manifest shape
  before the bridge re-exports the Tensor/Program/Session/model/runtime surface
  with identity parity. Node and Bun concrete loaders now declare only host
  policy (`runtime`, `bridge`, runtime kind, and policy owner) and delegate
  candidate lookup, require fallback handling, assertion, and manifest
  construction to `src/ts/adapters/concrete_runtime_loader.ts`. The
  implementation source lives at
  `src/ts/adapters/node_ffi_runtime.ts`; `tsdown.config.mjs` emits its JS
  artifact while excluding that internal concrete adapter from public
  declaration bundling. Checked Node host runtime loading lives in
  `src/ts/adapters/node_host_runtime.ts`, while shared native-library load-info
  policy lives in `src/ts/adapters/native.ts`; checked ABI struct registration
  lives in `src/ts/adapters/node_abi_structs.ts`; shared native status error
  formatting/check policy lives in `src/ts/runtime/native_status.ts`, while checked
  Node handle extraction policy lives in `src/ts/adapters/node_status.ts`;
  shared Node/Bun native ABI model-kind and buffer-storage alias policy lives
  in `src/ts/runtime/native_abi_constants.ts`, with no adapter compatibility
  re-export path;
	  checked generic TinyLinear/TinyMlp/Program/Session public class surface lives
	  once in `src/ts/runtime/generic_family_surface.ts`, with shared generic
	  model-create/read-handle adapter policy in
	  `src/ts/adapters/generic_family_model_handle_surface.ts` and Node native
	  descriptor adapter glue in `src/ts/adapters/node_generic_family_surface.ts`;
  checked shared Adapter Tensor public class surface lives in
  `src/ts/adapters/tensor_surface.ts`, keeping Node and Bun on one facade-bound
  Tensor constructor/method surface while `src/ts/core/**` owns the actual
  Tensor semantics. checked shared Adapter Tensor runtime helper assembly lives
  in typed `src/ts/adapters/tensor_runtime_surface.ts`, so tensor
  data/core/math, metadata, grad-state, Session tensor wrapping, and host-surface
  helper composition are built once while Node and Bun supply only host
  callbacks such as dtype/device policy, grad-mode policy, NativeBuffer checks,
  and f32 buffer conversion.
  checked shared Adapter NativeBuffer public class surface lives in
  `src/ts/adapters/native_buffer_surface.ts`, keeping Node and Bun on one
  facade-bound NativeBuffer constructor/method surface;
  checked LLaMA token-step/execute/selection ABI calls live in
  `src/ts/adapters/node_llama_token_ops.ts`; shared LLaMA token step output
  allocation/native-readback policy lives in
  `src/ts/runtime/llama_token_output.ts`, and token sample ABI field
  projection lives in `src/ts/core/token.ts`, so Node and Bun do not separately
  decide when to allocate host logits, read from a bound native output, or map
  normalized sampling policy into native fields. There is no adapter
  compatibility re-export path;
	  checked shared LLaMA model/program/session public class surface lives in
	  `src/ts/runtime/llama_family_surface.ts`, with no adapter compatibility
	  re-export path;
  checked LLaMA KV-cache/session bind ABI packing lives in
  `src/ts/adapters/node_llama_session_bind_ops.ts`; shared LLaMA Session bind
  output/KV-cache validation and descriptor fields live in
  `src/ts/runtime/llama_session_bind_desc.ts`, with no adapter compatibility
  re-export path;
	  checked shared LLaMA KV-cache allocation/resource policy lives in
	  `src/ts/runtime/llama_kv_cache.ts`, with Node object/program-device
	  adapter glue in `src/ts/adapters/node_llama_kv_cache_ops.ts` and no
	  adapter compatibility re-export path;
  checked Program buffer/device ABI policy lives in
  `src/ts/adapters/node_program_buffer_ops.ts`; shared Program device-buffer
  import descriptor vocabulary lives in `src/ts/runtime/program_buffer_desc.ts`,
  so Node emits canonical records while Bun packs the same fields into ABI
  words. There is no adapter compatibility re-export path;
  checked generic Program bind ABI policy lives in
  `src/ts/adapters/node_program_bind_ops.ts`; shared Node/Bun generic Program
  bind descriptor vocabulary lives in
  `src/ts/runtime/program_bind_desc.ts`, so host and native-buffer bind
  records share one TS-owned field/length mapping while Node emits object
  records and Bun only packs those fields into ABI words. There is no adapter
  compatibility re-export path;
  the same runtime Module owns output-only bind fields used by LLaMA
  native-output binding so LLaMA adapters do not rebuild generic Program bind
  descriptor records by hand;
  checked generic Session upload ABI policy lives in
  `src/ts/adapters/node_session_ops.ts`, while shared Session step I/O
  fields, descriptor, and result policy live once in
  `src/ts/runtime/session_step_io.ts`;
  checked module-program compile and bind descriptor ABI policy lives in
  `src/ts/adapters/node_module_program_ops.ts`;
  checked model path/safetensors load-probe ABI policy lives in
  `src/ts/adapters/node_model_source_ops.ts`; shared model-source descriptor
  vocabulary lives in `src/ts/runtime/model_source_desc.ts`, so Node emits
  canonical object records while Bun packs the same fields into ABI words, with
  no adapter compatibility re-export path;
  shared supported-checkpoint catalog count validation/iteration lives in
  `src/ts/runtime/model_source_catalog.ts`;
  checked native inspection/compile descriptor/runtime-profile ABI calls live in
  `src/ts/adapters/node_inspection_ops.ts`; shared compile descriptor
  fields/default-envelope policy lives in `src/ts/runtime/abi.ts`, so Node
  emits canonical compile records while Bun packs the same fields into ABI words;
  checked native symbol binding lives in `src/ts/adapters/node_symbols.ts`;
  the package root resolves directly to `dist/node.cjs`; the obsolete
  `js/node.cjs` compatibility shim has been removed. Shared concrete native
  runtime contract loading/fallback diagnostics live in
  `src/ts/adapters/concrete_runtime_loader.ts`, whose host surface now also
  owns load-candidate lookup, host require/assert helpers, and manifest
  construction so Node and Bun loaders no longer duplicate the native API
  contract loop. `zgml/bun` now
  uses the Bun export condition to resolve to TS-built `dist/bun_native.cjs`,
  which loads the concrete Bun FFI runtime from
  `src/ts/adapters/bun_ffi_runtime.ts` behind the same TS native API contract
  while Node keeps `dist/bun.cjs` as a native-free frontend artifact. Checked
  Bun ABI word-writing helpers live in `src/ts/adapters/bun_abi_words.ts`, so
  descriptor-packing modules share one pointer/size slot writer instead of
  hand-rolling `BigUint64Array` helpers.
  Checked
  Bun host runtime loading lives in `src/ts/adapters/bun_host_runtime.ts`,
  while shared native-library load-info policy lives in `src/ts/adapters/native.ts`,
  keeping suffix/path/missing-library resolution out of the concrete Bun FFI
  wrapper and out of duplicate host implementations.
  Shared native status error formatting/check policy lives in
  `src/ts/runtime/native_status.ts`, while checked Bun handle extraction policy lives
  in `src/ts/adapters/bun_status.ts`; checked Bun native inspection/compile
  descriptor/runtime-profile ABI policy lives in
  `src/ts/adapters/bun_inspection_ops.ts`; shared compile descriptor
  fields/default-envelope policy lives in `src/ts/runtime/abi.ts`; shared
  LLaMA Program inspection projection policy lives in
  `src/ts/runtime/llama_program_inspection.ts`, with no adapter compatibility
  re-export path;
  checked Bun generic Session upload ABI policy lives in
  `src/ts/adapters/bun_session_ops.ts`, while shared Session step I/O
  fields, descriptor, and result policy live once in
  `src/ts/runtime/session_step_io.ts`;
  checked Bun LLaMA token-step/execute/selection ABI policy lives in
  `src/ts/adapters/bun_llama_token_ops.ts`, with shared LLaMA token step output
  allocation/native-readback policy in `src/ts/runtime/llama_token_output.ts`
  and shared token sample ABI field projection in `src/ts/core/token.ts`;
  checked Bun Program buffer/device
  ABI policy lives in `src/ts/adapters/bun_program_buffer_ops.ts`, with shared
  Program device-buffer import descriptor vocabulary in
  `src/ts/runtime/program_buffer_desc.ts`; checked Bun
  module Program compile ABI policy lives in
  `src/ts/adapters/bun_module_program_ops.ts`; shared retained Program
  compile-evidence attachment policy lives in
  `src/ts/runtime/module_program_evidence.ts`, with no adapter compatibility
  re-export path; checked Bun
  generic Program bind ABI policy lives in
  `src/ts/adapters/bun_program_bind_ops.ts`; checked Bun
	  LLaMA KV-cache/session bind ABI packing lives in
	  `src/ts/adapters/bun_llama_session_bind_ops.ts`, which packs native-output
	  binding from shared Program output-bind fields and shared
	  `src/ts/runtime/llama_session_bind_desc.ts` LLaMA Session bind descriptor
	  fields rather than re-owning validation policy or the generic Program bind
	  descriptor mapping; checked Bun
		  shared LLaMA KV-cache allocation/resource policy lives in
		  `src/ts/runtime/llama_kv_cache.ts`, with Bun adapter glue in
		  `src/ts/adapters/bun_llama_kv_cache_ops.ts`; checked Bun LLaMA
		  KV-cache class lifecycle lives in
		  `src/ts/adapters/bun_llama_kv_cache_surface.ts`; checked shared Adapter
		  `NativeBuffer` public surface lives in
		  `src/ts/adapters/native_buffer_surface.ts`;
		  checked shared generic model-create/read-handle adapter policy lives in
		  `src/ts/adapters/generic_family_model_handle_surface.ts`, with Bun
		  native descriptor adapter glue in
		  `src/ts/adapters/bun_generic_family_surface.ts` over
		  `src/ts/runtime/generic_family_surface.ts`; generic tiny-model
		  native descriptor inference now lives once in
		  `src/ts/runtime/generic_model_desc.ts`; that module also owns
		  generic tiny-model Program requirement evidence and byte sizing, so
		  Node passes the canonical record to `zgml_model_create`, Bun only
		  serializes that same record into ABI words, and the shared generic
		  Session surface asks for named requirements instead of rebuilding
		  buffer-layout arithmetic in a constructor. There is no adapter
		  compatibility re-export path; shared model-source descriptor
		  vocabulary lives in `src/ts/runtime/model_source_desc.ts`, so Bun packs
		  TS-owned path/safetensors/tiny-model fields while Node emits canonical
		  object records, with no adapter compatibility re-export path; checked
		  Bun model path/safetensors load-probe ABI policy lives in
  `src/ts/adapters/bun_model_source_ops.ts`, with shared supported-checkpoint
  catalog count validation/iteration in `src/ts/runtime/model_source_catalog.ts`. The
  legacy `js/bun.ts` path has been removed instead of remaining as a separate
  Bun source copy. The large public package declaration
  body now lives in `src/ts/public_api.ts`, emits
  `dist/public_api.d.cts`, which is the canonical package `types` target for
  the root, Node, and Bun entrypoints. `scripts/check_ts_source_architecture.cjs` rejects a return to
  `src/ts/public_api.d.ts` or source-tree declaration exports, so public types
  are built by the same `tsdown` path as package runtime artifacts.
  `scripts/check_ts_source_architecture.cjs` now typechecks `src/ts/**` with
  `tsc --noEmit` and enforces architecture guards directly; it rejects
  `js/generated/**` as a checked-in checkout artifact tree, because generated
  checkout JS must not become a second source-of-truth contract. The npm
  artifact has already moved to package-first semantics:
  `scripts/check_package_artifact.cjs` runs `npm pack --dry-run`, requires the
  `dist/**` package/runtime artifacts, and rejects packaged `js/generated/**`
  plus local smoke harness files. It also requires the TS-authored package
  smokes emitted to `dist/smokes/**` and rejects public package scripts that
  route back through repo-only `js/` validation. Package consumers and smokes
  run from `dist/**` emitted directly from `src/ts/**`. `npm run build:package`
  now calls `tsdown` directly; `tsdown.config.mjs` discovers
  `src/ts/**/*.ts` entries and emits them into `dist/**` CJS and declaration
  artifacts without a wrapper build script or generated-source parity lane.
  The npm package ships `scripts/package_metadata_policy.cjs` beside
  `tsdown.config.mjs`, because that config consumes the policy; the policy
  module now carries a `zgml-package-metadata-policy` manifest proving TS-only
  `tsdown` fan-out rather than a hidden package source tree.
  `npm run typecheck:dist` proves those generated declarations are consumable,
  and `npm run smoke:dist` now runs the TS-authored
  `src/ts/smokes/dist_package_smoke.ts` artifact from `dist/smokes/**` to prove
  the dist runtime entrypoints load without a repo-only `js/` smoke harness.
  `zgml/frontend`, `zgml/node` runtime, top-level friendly TS subpaths, adapter
  evidence, and deep wildcard package subpaths now resolve to `dist/**` CJS
  artifacts, making the TS-bundled frontend and Node native bridge real package
  surfaces without hand-maintained npm-script entry lists.
  The Node and Bun FFI examples now consume package-built artifacts instead of
  importing repo-only `js/` implementation files: the Node example loads the
  built `dist/node.cjs` adapter, the Bun example imports `zgml/bun`, and
  `scripts/check_ts_source_architecture.cjs` rejects regressions back to
  repo-only JS or `src/ts/**` product internals in public JS/CJS/MJS examples.
  Their deep runtime checks now assert frozen structural
  evidence, correct outputs, and non-mutating caller options rather than relying
  on private object-identity or scratch-routing details from the legacy shim.
  A package-build probe also made the next Bun cleanup concrete:
  `src/ts/adapters/bun_ffi_runtime.ts` can be emitted by `tsdown`, and
  `src/ts/runtime/runtime_load_candidates.ts` now owns the shared
  package-artifact candidate policy for concrete runtimes and shared frontend
  loading: prefer package-built/local sibling `dist/*.cjs` artifacts and never
  fall back through `js/generated/**` or repo-only shared CJS shims. The
  Node/Bun adapter loaders consume that runtime-owned policy.
  The Bun FFI type-safety gap has been closed at the source-build boundary:
  `tsconfig.source.json` includes `src/ts/**/*.ts` without exclusions, so
  `src/ts/adapters/bun_ffi_runtime.ts` participates in strict source checking.
  Raw Bun host intrinsics (`bun:ffi` loading, pointer creation, and pointer
  readback) stay isolated in the checked
  `src/ts/adapters/bun_ffi_intrinsics.ts` wrapper, while the concrete runtime
  uses the CJS-safe `pathToFileURL(__filename).href` module URL pattern. `npm
  run smoke:bun` still proves the emitted Bun package runtime plus
  `dist/bun_native.cjs` bundling, but Bun is no longer a strict-source
  exception.
  The flat shared frontend aggregation has now moved into
  `src/ts/shared_frontend.ts`, which emits `dist/shared_frontend.cjs` and
  `dist/shared_frontend.d.cts`; it re-exports the TS-authored Tensor/nn/train/
  runtime helper surface and preserves the small hook-injection wrappers needed
  during migration. `scripts/check_ts_source_architecture.cjs` rejects that TS aggregator
  if it reaches back into repo-only shared CJS shims, and
  `scripts/check_package_artifact.cjs` requires the dist artifact in the npm
  package. Session facade composition has now moved into TS-owned package
  artifacts, and the obsolete source-checkout `js/shared_session_facade.cjs`
  shim has been removed. The Program/Session
  composition lives in
  `src/ts/runtime/session_facade_composition.ts`, emitted to
  `dist/runtime/session_facade_composition.cjs` and package-built as
  `dist/runtime/session_facade_composition.cjs`; it assembles the generic
  `Program -> Session -> StepParams` helpers, parameter upload facade,
  readback/execution/lifecycle wrappers, LLaMA bind-routing policy, LLaMA token
  scratch helpers, token selection/generation facade methods, readback, prefill,
  step/execute, runtime-profile, and lifecycle composition. The
  generated-source guard rejects repo-only JS imports from that TS composition
  module, rejects regrowing shared compatibility shims, and the
  package-artifact guard requires its `dist/**` output.
  `src/ts/runtime/native_api_contract.ts` now owns the root/native parity
  contract: required PyTorch-like eager Tensor/native exports, required TS
  package-spine exports, and comparison helpers.
  `src/ts/smokes/root_native_parity_smoke.ts`, emitted by `tsdown` as
  `dist/smokes/root_native_parity_smoke.cjs`, consumes that TS contract to make
  package/native parity executable: live `zgml`, the concrete Node adapter,
  `dist/node.cjs`, and `zgml/node` must expose the same eager Tensor/native API,
  with root and `zgml/node` resolving through the TS-built bridge after
  `npm run build:package`.
  `zgml/runtime/native_api_contract` is package-facing through the dist runtime
  wildcard, so future migration slices update contract truth in TS rather than
  in a CJS smoke. That prevents an early export-map flip from silently dropping
  the PyTorch-like `Tensor`/`nn`/`Program`/`Session` and model/runtime helpers.
  Node model-source byte conversion lives in typed
  `src/ts/adapters/node_model_source_bytes.ts`, keeping Buffer-backed string
  and `Uint8Array` view conversion policy out of the unchecked concrete
  Adapter while preserving the native FFI byte-pointer shape.
  Adapter safetensors file-header IO policy lives in typed
  `src/ts/adapters/safetensors_file_header.ts`, keeping host open/read/close
  callbacks out of Node and Bun concrete Adapters while preserving the shared
  header exact-read and little-endian length decoding policy.
  Adapter NativeBuffer instance guard policy lives in typed
  `src/ts/adapters/native_buffer_instance_surface.ts`, keeping Node/Bun
  concrete Adapters on one late-bound `NativeBuffer` class check while letting
  each host supply its own uninitialized-surface message.
  Adapter index-values forwarding policy lives in typed
  `src/ts/adapters/index_values_surface.ts`, keeping Node and Bun on one
  Tensor/index input normalization seam backed by `src/ts/core/index_values.ts`.
  Adapter Module compiler forwarding policy lives in typed
  `src/ts/adapters/module_compiler_surface.ts`, keeping Node and Bun on one
  late-bound `TraceModuleCompiler` forwarding seam while each host supplies its
  own uninitialized-surface message.
  Adapter model-source facade forwarding policy lives in typed
  `src/ts/adapters/model_source_surface.ts`, keeping Node and Bun on one
  late-bound model-source facade seam while each host supplies its own
  uninitialized-surface message.
  Adapter model handle bind/compatibility policy lives in typed
  `src/ts/adapters/model_handle_policy.ts`, keeping the LLaMA-bindable and
  generic-compatible class lists, handle extraction policy, and shared
  model-handle helper assembly out of the Node and Bun concrete Adapters while
  preserving the shared model-handle helper contract.
  Adapter module/optimizer state helper assembly lives in typed
  `src/ts/adapters/module_state_surface.ts`, so parameter creation, module
  state dictionaries, optimizer state snapshots, zero-grad, requires-grad, and
  state restore helpers are composed once from the TS-owned shared frontend while
  hosts supply only Tensor/f32 layout knobs.
  Adapter `nn` module class assembly lives in typed
  `src/ts/adapters/frontend_module_surface.ts`, keeping `Linear`, `Embedding`,
  `Sequential`, stateless modules, feature norms, shape modules, trace compiler
  wiring, and module-facade helper construction in one shared Adapter Module
  instead of repeating host-specific product module policy.
  Adapter native eager public policy lives in typed
  `src/ts/adapters/native_eager_surface.ts`, so Node and Bun share tensor
  coercion, shape inference, output validation, activation mapping, and public
  aliases for `nativeEager`, while concrete adapters supply only ABI calls and
  status checking.
  Adapter module Program construction lives in typed
  `src/ts/adapters/program_factory_surface.ts`: Node and Bun concrete
  Adapters late-bind the current `Program` class, while the module compile
  `handle`/`desc`/compile-evidence construction stays in TS source.
  Adapter Program buffer native bridging lives in typed
  `src/ts/adapters/program_buffer_native_bridge.ts`, keeping
  `NativeBuffer.fromNativeHandle(...)` and `NativeBuffer.deviceImportOptions(...)`
  policy out of the Node and Bun concrete Adapters while preserving the native
  buffer handle/import shape. Adapter NativeBuffer facade assembly lives in typed
  `src/ts/adapters/native_buffer_facade_surface.ts`, so allocation, wrapping,
  external-resource, IO, inspection, free, and ProgramDevice-handle callbacks
  are composed once while Node and Bun supply only native syscalls and
  null-handle policy. Program buffer factory forwarding lives in typed
  `src/ts/adapters/program_buffer_factory_surface.ts`, keeping output, named
  buffer, KV-cache buffer, and LLaMA KV-cache forwarding out of concrete
  runtime bodies while preserving the shared Program facade callbacks.
  Adapter Program/runtime facade assembly lives in typed
  `src/ts/adapters/program_runtime_surface.ts`, so `ProgramDevice`,
  Program buffer factories, Program/model facade policy, generic Program
  facades, LLaMA model-family facades, and LLaMA Program facades are composed
  once from TS-owned shared frontend helpers while Node and Bun supply only
  native callbacks, null-handle policy, and KV-cache construction differences.
  Adapter LLaMA Session facade assembly lives in typed
  `src/ts/adapters/llama_session_facade_surface.ts`, so token execution,
  prefill, generation, output readback, runtime-profile, lifecycle, and
  sampling facade helpers are composed once while Node and Bun supply only
  native token-window callbacks and null-handle policy.
  Adapter LLaMA family surface assembly lives in typed
  `src/ts/adapters/llama_family_surface.ts`, so concrete runtimes pass one
  grouped LLaMA Session facade into the runtime-owned LLaMA Model/Program/
  Session public class surface instead of re-threading every Session method.
  Adapter Session/binding facade assembly lives in typed
  `src/ts/adapters/session_runtime_surface.ts`, so generic Session lifecycle,
  step/upload helpers, Program bind validation/preparation, owned buffer
  lifetime policy, module binding, and native-buffer bind fields are composed
  once for Node and Bun from TS-owned shared frontend helpers.
  Node LLaMA KV-cache factory policy lives in typed
  `src/ts/adapters/node_llama_kv_cache_factory_surface.ts`, keeping
  `NativeBuffer.create(...)` and `new LlamaKvCache(k, v)` construction out of
  the unchecked concrete Adapter while preserving the shared runtime KV-cache
  allocation/resource policy.
  Adapter ProgramDevice construction lives in typed
  `src/ts/adapters/program_device_factory_surface.ts`, keeping
  `new ProgramDevice(handle, placement)` out of the Node and Bun concrete
  Adapters while preserving the shared generic/LLaMA Program facade resource
  accessors.
  Program bind Session construction policy lives in typed
  `src/ts/adapters/session_factory_surface.ts`: the Node and Bun concrete
  Adapters late-bind the current `Session` class, but host-vs-native prepared
  bind construction and parameter-layout threading stay in TS source.
  Its generated package-spine artifact now exports provenance evidence too:
  `src/ts/runtime/native_package_spine_contract.ts` is generated from the TS
  native contract, declares `source: "generated-ts"`, and proves there is no
  hand-maintained root export list to keep synchronized. The same generator now
  owns the native-free Bun frontend root as well as the Node and Bun-native root
  wrappers.
  The Zig native substrate manifest has the same named maintenance seam:
  `npm run generate:native-substrate-manifest` and
  `npm run check:native-substrate-manifest` regenerate or verify
  `src/native_substrate_manifest.zig` from `src/ts/frontend_manifest.ts`, and
  `check:ts-source-architecture` runs that check so Zig substrate metadata
  cannot drift from TS product policy.
  The Node and Bun package smokes now consume that same TS contract too from
  `dist/smokes/**`, prove nested-array tensor construction, eager
  `Linear`/MSE/SGD training, frozen `stateDict` evidence, and native API export
  parity, and fail before a runtime-specific export gap can hide behind shared
  behavior tests. Bun's public smoke also bundles `dist/bun_native.cjs` with
  `bun build --target=bun`.
  The remaining package-spine work is to keep shrinking temporary unchecked
  concrete FFI adapter bodies into typed TS modules. The checked adapter islands
  currently cover host loading/path policy, ABI struct registration, native
  status/error and handle extraction policy, the generic
  TinyLinear/TinyMlp/Program/Session public class surface, the NativeBuffer
  public class surface, shared Adapter generic family model-handle creation
  surface, shared Adapter NativeBuffer instance guard surface,
  shared Adapter Tensor public class surface, the shared Adapter public runtime
  export contract surface, shared Adapter safetensors file-header IO policy, shared
  Adapter frontend namespace assembly surface, shared Adapter model-source facade
  load-kind routing, shared Tensor root construction/factory/helper/coercion policy, smoke-covered shared Tensor
  inspection/metadata policy, smoke-covered shared Node/Bun Tensor grad-state
  policy, core Tensor flexible indexing/assignment policy, LLaMA
  token-step/execute/selection ABI calls, the LLaMA model/program/session public class surface, LLaMA
  KV-cache/session bind ABI policy, shared Adapter model-source public facade forwarding surface, module-program compile and
  bind descriptor ABI policy, shared Adapter Module compiler forwarding surface,
  shared runtime Program bind descriptor
  vocabulary, generic Program bind policy, generic Session
  upload policy, shared Session step I/O policy, Program buffer/device policy, Node Program
  buffer factory forwarding surface, shared Adapter index-values surface,
  LLaMA KV-cache allocation/resource policy,
  Node and Bun LLaMA KV-cache class lifecycle, model path/safetensors load-probe and
  supported-checkpoint catalog policy, native
  inspection/compile descriptor/runtime-profile ABI calls, and native symbol
  binding.
  it also checks that `package.json` keeps the TS-authored frontend, adapter
  evidence, `tensor`/`nn`/`compile`/`program`/`session`/`step_params`/`checkpoint`/`loss`/`optim`, and `core`/`runtime`/`nn`/`train` module subpaths package-facing.
  The same gate rejects regrowing `js/shared_session_facade.cjs`, forcing callers
  through the TS-authored session facade helpers instead of re-growing a
  parallel CJS implementation, and rejects reintroduced hand-written CJS Session
  method policy for the
  generic and LLaMA `step`/`execute`/`prefill`/token-selection surfaces now
  owned by `src/ts/runtime/session_facade.ts`.
  `npm run test:ts-source` runs this architecture gate after the TS smoke. The live package now enters through
  tsdown-built CJS/ESM artifacts, so the next architectural move is to keep
  moving real Node/Bun adapter wiring into checked TS modules behind that spine.
  JS/TS eager Tensor operation policy now lives in
  `src/ts/core/**` modules re-exported by
  `src/ts/shared_frontend.ts`: tensor data conversion/gradient helpers,
  scalar/core Tensor methods, eager tensor math/autograd method implementations,
  Tensor construction, parameter creation, factory/view/broadcasting/indexing/slicing/join
  helpers, activation scalar/derivative math, grad-mode helpers, and the public
  Tensor facade share one Interface across Node and Bun. The obsolete
  `js/shared_tensor_ops.cjs` re-export shim has been removed. Tensor placement and direct Program bind edge policy now live in
  `src/ts/runtime/tensor_placement.ts`, emitted to
  `dist/runtime/tensor_placement.cjs`: eager CPU/f32 movement,
  `Tensor.place(...)` slot checks, direct `Program.bind(...)` slot validation,
  host/native-buffer binding preparation, temporary weights/bias buffer
  placement, logical NativeBuffer bind-field selection, packed Program
	  weight/bias length helpers, and cleanup of owned bind buffers share one
	  Interface across Node and Bun. Eager JS/TS `nn.Shape` modules now live in
	  `src/ts/nn/shape_module.ts`, emitted to `dist/nn/shape_module.cjs`,
	  so reshape/view/flatten/squeeze/unsqueeze/transpose/broadcast/expand/narrow/select/slice
	  construction, parameterless state, compile support, binding, placement, and
	  compile delegation are TS-authored while Node/Bun keep their existing factory
	  surface. The trace compiler now lets higher-rank shape/view ops reach the
	  Kernelizer for exact support decisions, so rank-3 `permute` preserves frozen
	  Tensor Program IR and reports `kernelizer:unsupported-view` instead of a
	  coarse IR rank rejection. Stateless eager JS/TS `nn` activation, softmax/logSoftmax,
	  reduction, and Dropout modules now live in `src/ts/nn/parameterless_modules.ts`, emitted to
	  `dist/nn/parameterless_modules.cjs`, so tensor/host forward dispatch,
	  stochastic dropout masks, dropout autograd routing, empty parameter binding,
	  placement, compile support, unsupported native dropout rejection, and
	  no-parameter state policy are TS-authored too. Shared parameterless module
	  traversal, parameter metadata, state-dict, load-state, zero-grad,
	  requires-grad, and train/eval policy now lives in
	  `src/ts/nn/parameterless_state.ts`, emitted to
	  `dist/nn/parameterless_state.cjs`; `nn.Shape` and the stateless modules
	  delegate there instead of carrying duplicate empty-parameter surfaces.
	  Shared `nn` module traversal helpers now live in `src/ts/nn/module_tree.ts`,
	  emitted to `dist/nn/module_tree.cjs`, so leaf module `children`/`modules`/
	  `namedChildren`/`namedModules` evidence and Sequential traversal entries
	  no longer grow local helper copies in every module file.
	  Shared stateful module lifecycle/state methods now live in
	  `src/ts/nn/stateful_module_state.ts`, emitted to
	  `dist/nn/stateful_module_state.cjs`; `nn.Linear`, `nn.Embedding`,
	  `nn.Sequential`, `nn.LayerNorm`, and `nn.RMSNorm` delegate parameter
	  metadata access, zero-grad, requires-grad, train/eval, state dict, and
	  load-state methods there instead of repeating the same public module
	  surface in each class.
	  Shared single-module Program compile methods now live in
	  `src/ts/nn/single_module_compile.ts`, emitted to
	  `dist/nn/single_module_compile.cjs`; `nn.Shape`, stateless activation/
	  reduction/dropout modules, `nn.Embedding`, and feature-axis norm modules
	  delegate generic `compileSupport`/`canCompile`/`bindParameters`/
	  `placeParameters`/`compile` mechanics there while keeping only their
	  domain-specific forward math, diagnostics, and parameter binding
	  differences. `nn.Linear` and `nn.Sequential` retain their specialized
	  tiny-linear and composition paths instead of being forced through the
	  generic helper.
	  Shared sequential Program compile methods now live in
	  `src/ts/nn/sequential_program_compile.ts`, emitted to
	  `dist/nn/sequential_program_compile.cjs`; `nn.Linear` and `nn.Sequential`
	  delegate sequential analysis, support evidence, placement, device-program
	  compile delegation, and tiny-linear evidence attachment there while keeping
	  direct Linear parameter binding and Sequential packed-parameter binding as
	  explicit local differences.
	  Eager JS/TS `nn.Linear` now
	  lives in `src/ts/nn/linear_module.ts`, emitted to
	  `dist/nn/linear_module.cjs`, so parameter initialization, vector/batch
	  forward math, state/metadata, binding/placement, tiny-linear compile evidence,
	  and device-program fallback compile delegation are TS-authored. Eager JS/TS
	  `nn.Embedding` now lives in `src/ts/nn/embedding_module.ts`, emitted to
	  `dist/nn/embedding_module.cjs`, so indexed gather output, repeated-index
	  gradient accumulation, state/metadata, binding/placement, input-shape compile
	  diagnostics, and device-program compile delegation are TS-authored. Eager
	  JS/TS `nn.Sequential` now lives in `src/ts/nn/sequential_module.ts`,
	  emitted to `dist/nn/sequential_module.cjs`, so ordered forward
	  composition, nested traversal, child training-mode propagation, parameter
	  naming/metadata, trace, bind/place, tiny-linear delegation, and device-program
	  compile delegation are TS-authored. Eager JS/TS feature-axis `nn.LayerNorm`
	  and `nn.RMSNorm` now live in `src/ts/nn/feature_norm_module.ts`, emitted to
	  `dist/nn/feature_norm_module.cjs`, so forward math, Tensor autograd,
	  affine/no-affine state and metadata, bind/place, and device-program compile
	  delegation are TS-authored. Module training-mode normalization and recursive
	  child propagation now live in `src/ts/train/module_mode.ts`, emitted to
	  `dist/train/module_mode.cjs`. Module state and parameter metadata
  helpers now live in `src/ts/train/module_state.ts`, emitted to
  `dist/train/module_state.cjs`: `stateDict`/`loadStateDict`, `zeroGrad`,
  `requiresGrad` policy, compile-support metadata, shared `parameterNames` /
  `parameterInfos` / `parameterInfo` shape-layout evidence, and optimizer state snapshot
  validation share one TS-authored implementation across Node and Bun. The
  stateful `nn.Linear`, `nn.Embedding`, `nn.Sequential`, `nn.LayerNorm`, and
  `nn.RMSNorm` classes delegate parameter names and metadata there instead of
  re-creating shape/layout records in each module. The training surface is
  split by ownership instead of routed through one broad hub: checkpoint
  orchestration and optimizer/scheduler namespace composition live in
  `src/ts/train/training.ts`, emitted to `dist/train/training.cjs`, while
  public `loss`, `train`, `optim`, and shared frontend facades import the
  simpler helper factories directly from their owning modules. Checkpoint
		  JSON serialization policy now lives in
		  `src/ts/train/checkpoint_serialization.ts`, emitted to
		  `dist/train/checkpoint_serialization.cjs`, so plain-record validation,
		  JSON-safe metadata freezing, tensor-entry normalization, scheduler
		  snapshot parsing, and checkpoint inspection records share one TS-authored
		  implementation. Optimizer update classes now live in
		  `src/ts/train/optimizer_classes.ts`, emitted to
		  `dist/train/optimizer_classes.cjs`, so SGD/Adam/AdamW math, parameter-group
		  validation, config signatures, state snapshots, and strict/validate-only
		  restore share one implementation while `training.ts` only re-exports and
		  wires namespaces. Loss and tiny training helpers now live in
		  `src/ts/train/loss_train_helpers.ts`, emitted to
		  `dist/train/loss_train_helpers.cjs`, so MSE/CrossEntropy losses, gradient
		  clipping/norms, `train.step`, `train.lossStep`, and `train.fit` keep loop
		  mechanics together. Train evidence signatures, constructors, predicates,
		  assertions, require helpers, and signature matchers now live in
		  `src/ts/train/evidence.ts`, emitted to `dist/train/evidence.cjs`, so
		  training-loop evidence can cross callbacks, package boundaries, or logs
		  without making the loop Module re-own proof-shape policy. Optimizer
		  config/state and LR-scheduler snapshots now
		  share standalone predicates, assertions, require helpers, and signature
		  matchers from `src/ts/train/optimizer_snapshot.ts`, re-exported through
		  `zgml/optim` and the `optim` namespace, so PyTorch-style state surfaces
		  remain simple while still carrying explicit proof records. Optimizer
		  learning-rate scheduler state/signature policy now lives in
		  `src/ts/train/scheduler_state.ts`, emitted to
		  `dist/train/scheduler_state.cjs`, so `StepLR`/`ExponentialLR` snapshots,
		  snake-case restore aliases, and validation-only scheduler restore share one
		  TS-authored implementation. Shared
		  `nn` namespace construction now lives in `src/ts/nn/namespace.ts`,
		  emitted to `dist/nn/namespace.cjs`: PyTorch-style constructor
	  aliases, scalar activation aliases, shape/norm factories, and compile/state facades
	  share one Interface across Node and Bun. Shared module prototype method
	  installation now lives in `src/ts/nn/namespace_methods.ts`, emitted to
	  `dist/nn/namespace_methods.cjs`, so compiler-inspection aliases and
	  PyTorch-style snake-case module methods stay out of the namespace
	  constructor table. NativeBuffer facade and
  public edge policy for live-handle validation, construction, create-size
  validation, typed-array wrapping, external-resource wrapping, byte-range
  writes, default readback allocation, caller-owned readback target/range
  validation, deterministic free/dispose, zgml-created device-buffer import
  provenance, signed external-resource/device-import/byte-range evidence with
  standalone predicates/assertions/signature matchers, and Session-owned buffer
  lifetime deduplication is also shared here so Node and Bun adapters only pack
  runtime-specific FFI calls. Program
	  facade evidence, capability, and lifecycle policy now live in
	  `src/ts/runtime/program_facade_policy.ts`, emitted to
	  `dist/runtime/program_facade_policy.cjs`:
	  requirements-to-buffer-layout projection, trace/IR/KernelPlan views,
	  model/module compatibility acceptance, executable capability booleans,
	  runtime-profile snapshot/reset, and deterministic free/dispose share typed
		  source. Program-owned buffer factory routing now lives in
		  `src/ts/runtime/program_buffer_factory.ts`, emitted to
		  `dist/runtime/program_buffer_factory.cjs`: output/input/weights/bias/KV
		  buffer creation, resource callback validation, Program requirement sizing,
		  host-vs-placement-vs-resource selection, and KV-cache buffer factory routing
		  share typed source. `Program.bindModule(...)` routing now lives in
		  `src/ts/runtime/program_module_binding.ts`, emitted to
		  `dist/runtime/program_module_binding.cjs`: compatibility preflight,
		  parameter placement, owned-buffer deduplication, and cleanup-on-failure share
		  typed source. `ProgramDevice` placement/device-handle liveness,
		  frozen ProgramDevice info evidence, role-specific device-buffer helpers,
		  same-device import defaults, and
		  KV-cache device-buffer routing now live in `src/ts/runtime/program_device.ts`,
		  emitted to `dist/runtime/program_device.cjs`. Program buffer sizing,
		  byte-length fallback policy, frozen sizing snapshots, and public sizing
		  signatures plus the shared Program sizing accessor table now live in
		  `src/ts/runtime/program_sizing.ts`, emitted to
		  `dist/runtime/program_sizing.cjs`. Generic Program parameter
		  introspection (`parameterNames`, `parameterInfos`, `parameterInfo`) and
		  the shared parameter accessor table now
		  lives in `src/ts/runtime/program_parameters.ts`, emitted to
		  `dist/runtime/program_parameters.cjs`. Program edge-shape evidence
		  selection and generic/LLaMA shape accessor tables now live in `src/ts/runtime/program_shapes.ts`, emitted to
		  `dist/runtime/program_shapes.cjs`: generic Programs prefer compiled
		  shape constraints before descriptor fallbacks, while LLaMA-family Programs
		  expose vocab-sized output shapes from inspection. Program resource accessors
		  (`createBuffer`, role buffer helpers, device handles, and device-buffer
		  imports) now live in `src/ts/runtime/program_resources.ts`, emitted to
		  `dist/runtime/program_resources.cjs`. Program buffer layout caching,
		  slot-name/slot lookup, and LLaMA KV-cache layout access now live in
		  `src/ts/runtime/program_layout.ts`, emitted to
		  `dist/runtime/program_layout.cjs`. Generic Program policy accessors
		  for compile evidence, compiler artifacts, compatibility, capabilities,
		  runtime profiles, lifecycle, and `bindModule(...)` now live in
		  `src/ts/runtime/program_policy_accessors.ts`, emitted to
		  `dist/runtime/program_policy_accessors.cjs`; LLaMA-family Program
		  policy accessors for cached inspection, vocab size, KV-cache creation,
		  executable inspection, runtime profile, lifecycle, and Session bind handoff
		  live there too. The generic and LLaMA-family Program facade factories now
		  live in `src/ts/runtime/program_facade.ts`, emitted to
		  `dist/runtime/program_facade.cjs`; the obsolete
		  `js/shared_program_facade.cjs` compatibility shim has been removed, and
		  `src/ts/shared_frontend.ts` re-exports the TS-authored Program helpers.
		  Program wrapper policy follows one frontend policy for
  generic tensor Programs and LLaMA-family Programs. Node and Bun adapters keep
  only the actual native buffer/device calls and host-specific `LlamaKvCache`
  construction. Fixed-family LLaMA subclasses only specialize concrete Session
  construction. Model facade policy is shared too, but now lives with model
  source policy in `src/ts/runtime/model_source.ts`, emitted to
  `dist/runtime/model_source.cjs`: model inspection,
  compile-to-Program lifecycle, deterministic free/dispose, LLaMA-family public
  create/load/byte-load/probe/header-probe routing, and family-specific compile
  routing follow one Interface across Node and Bun, while adapters supply only
  native create/load/probe handle callbacks and concrete Model / Program
  constructors. Generic tensor Session call policy now
  lives in `src/ts/runtime/session_facade.ts` and
  `src/ts/runtime/session_facade_composition.ts`: bound-input/output selection,
  no-output advancement checks, Session inspection, shape/layout accessors,
  persistent-parameter upload hooks, reset/profile, and deterministic
  owned-buffer cleanup follow one Interface, while adapters retain native
  `session_step` and lifecycle symbol packing. Generic Program edge-shape and
  Session tensor-output wrapping/readback helpers now live in the internal
  `src/ts/runtime/session_tensor.ts` emitted to
  `dist/runtime/session_tensor.cjs`: `programInputShape`,
  `programOutputShape`, shaped output Tensor construction, output target
  selection, and host/NativeBuffer `readOutputInto` validation share one
  Interface across Node and Bun. Shared StepParams compatibility evidence,
  diagnostic freezing, ownership/effect classification, hot-path signatures,
  validation-error shaping, StepParams object validation, token-window fit
  checks, logits-output buffer validation, no-inline-output guards, and
  reusable token scratch-option normalization plus StepParams predicate/signature
  matching helpers plus assertion-shaped allocation-free, no-readback, and
  hot-path preflight guards now live in `src/ts/runtime/step_params.ts`,
  emitted to `dist/runtime/step_params.cjs`, so generic and LLaMA-family
  Sessions consume one TS-authored StepParams contract vocabulary instead of
  hand-maintaining it in the adapter facade. Generic and LLaMA-family Session
  facade StepParams predicates, assertion guards, and signature-match methods now live in
  `src/ts/runtime/session_facade.ts`, emitted to
  `dist/runtime/session_facade.cjs`, alongside shared live-session
  handle/inspection/position accessors, Session layout/shape/size accessors,
  live-guarded Session call-profile facade wrappers, and Session buffer-sizing facade
  assembly/signature matching plus runtime-profile facade read/match/reset
  wrappers, generic Session parameter inspection/upload facade methods,
  generic Session bound host/native input-output interpretation, explicit
  input/output preparation, core step/advance execution, and generic StepParams
  compatibility,
  readOutputInto/readOutputTensor facade wrappers, generic Session
  step/stepTensor/stepInto plus execute/executeTensor/executeInto/advance
  orchestration, LLaMA executeTokens/execute/executeTensor/executeInto/
  advanceTokens orchestration, LLaMA step/stepTensor/stepInto/advance orchestration,
  LLaMA prefill/prefillTensor/prefillInto orchestration, LLaMA scalar-token
  and token-window scratch facade helpers, LLaMA argmax/sample token
  selection plus native token-window generation orchestration, and
  reset/free/dispose lifecycle facade wrappers, while adapters supply only the
  concrete contract/compatibility callbacks and adapter
  liveness/native-profile/upload/resource-cleanup callbacks. Session buffer-sizing signatures
  and generic/LLaMA step-contract records now live in
  `src/ts/runtime/session_contract.ts`, emitted to
  `dist/runtime/session_contract.cjs`, so cold Session evidence and hot
  StepParams signatures share one TS-authored vocabulary. Frontend Session call
  profile counters, signatures, matching, and reset now live in
  `src/ts/runtime/session_profile.ts`, emitted to
  `dist/runtime/session_profile.cjs`, so generic and LLaMA-family
  Sessions share one TS-authored call-profile record. Generic Session parameter
  name/info projection, upload-by-name validation, and persistent binding-index
  selection now live in `src/ts/runtime/session_parameters.ts`, emitted to
  `dist/runtime/session_parameters.cjs`; native upload calls remain in
  the adapter facade. Generic and LLaMA-family Session buffer slot lookup,
  KV-cache layout presence checks, shape/length/byte-length projection, and
  bound-buffer kind classification now live in
  `src/ts/runtime/session_layout.ts`, emitted to
  `dist/runtime/session_layout.cjs`; that module also owns LLaMA default
  Session buffer-layout construction, reusable Session scratch allocation, and
  NativeBuffer-output binding checks plus host/native bound input/output
  selectors. LLaMA-family `Program.bind(...)` route selection, output
  normalization, inspection handoff, and owned-output cleanup now live in
  `src/ts/runtime/session_binding.ts`, emitted to
  `dist/runtime/session_binding.cjs`. Adapters still own live Session
  handles, FFI callbacks, and execution, but API policy should move to typed TS
  source first and treat CJS as generated/adapter glue. LLaMA and generic
  Session cleanup/owned-resource release policy now lives in
  `src/ts/runtime/session_lifecycle.ts`, emitted to
  `dist/runtime/session_lifecycle.cjs`; adapters supply the native
  handle-free callback and null-handle sentinel. Generic Session explicit
  input/output preparation, execute/executeTensor/executeInto plan routing,
  no-output result validation, step output target/readback routing, read-output
  tensor target planning, LLaMA execute/executeTensor/executeInto method
  planning, LLaMA stepTensor output validation, LLaMA execute-token
  output/readback routing, LLaMA read-output tensor target planning, and
  LLaMA prefillTensor/generation/sample planning, LLaMA token-selection and
  generation token-window planning, native-output preconditions for token
  selection/generation, plus generic and LLaMA StepParams compatibility
  construction now live in
  `src/ts/runtime/session_values.ts`, emitted to
  `dist/runtime/session_values.cjs`; the facade supplies host-value
  preparation callbacks and keeps the native step call. LLaMA-family Session policy now
  lives in the same TS session facade modules for:
  bound-output `readOutputInto` / `readOutputTensor`, token-window `executeTokens`
  output-selection/readback, tensor-output wrappers for step/execute/prefill,
  argmax/sample/generation preconditions, sample-option normalization,
  max-token checks, generated-token output validation, default buffer layout,
  one-token scratch, reusable option scratch, output-shape, KV-cache layout,
  position/inspection, reset/profile, and deterministic free/dispose follow one
  Interface. Token-window scalars and normalized token-selection/generation
  result shapes now come from `src/ts/core/token.ts` through the generated
  `dist/core/token.cjs` artifact, while token execution and selection
  descriptor packing remain adapter-local. Direct
  generic `Program.bind(...)` validation, host/native-buffer binding
  preparation, temporary weights/bias buffer placement, bound tensor retention,
  and owned-buffer cleanup policy come from `src/ts/runtime/tensor_placement.ts`,
  emitted to `dist/runtime/tensor_placement.cjs`;
  adapters own only the actual native session-bind call and `Session`
  construction. Generic
	  `Program.bindModule(...)` module compatibility, parameter placement,
	  owned-buffer deduplication, and cleanup-on-failure policy now lives in
	  `src/ts/runtime/program_module_binding.ts`;
  adapters assert handle liveness and delegate to their existing FFI-backed
  `Program.bind(...)`.
	  Safetensors path/byte source classification, path header-prefix parsing,
	  source-to-probe/load routing, load-kind dispatch, header/data byte
	  normalization, model-handle acceptance, liveness policy for bind and
	  compatibility preflights, model facade lifecycle, and LLaMA-family model
	  routing now live in `src/ts/runtime/model_source.ts`, emitted to
	  `dist/runtime/model_source.cjs` and the friendly `zgml/model_source`
	  package subpath behind the `shared_frontend` facade,
	  while adapters keep their host-specific
	  filesystem calls, model constructors, class predicates, handle extraction,
	  and native descriptor packing. Supported-checkpoint count validation and
	  catalog iteration now live in `src/ts/runtime/model_source_catalog.ts`,
	  so the runtime facade and host adapters share the same model-source catalog
	  policy while adapters only provide ABI-specific inspection calls. Runtime ABI snapshot shaping,
	  ABI struct-size catalog iteration, and compatibility handshake policy are
	  shared here too, so adapters cannot drift on runtime evidence while
	  loading the same native library. The numeric ABI vocabulary, runtime feature handshake, runtime ABI
  facade helpers, compile-envelope normalization, default-envelope detection,
  and descriptor record projection now live in the internal
  `src/ts/runtime/abi.ts`, emitted to `dist/runtime/abi.cjs`,
  behind the `shared_frontend` facade, so ABI churn
  and runtime descriptor validation have locality without widening the Node/Bun
  Adapter Interface.
  Buffer, model, Program, Session, requirement, compatibility, and runtime
  profile evidence decoding now live in `src/ts/runtime/inspection.ts`, emitted
  to `dist/runtime/inspection.cjs`,
  Module behind the same facade, so native ABI field maps, frozen diagnostic
  shaping, compatibility booleans, and capability derivation have locality
  while Node and Bun still import one shared frontend Interface. Program buffer
	  layout projection, buffer callback slot evidence, resource slot validation,
	  byte-length checks, and LLaMA KV-cache layout/slot policy now
	  live in `src/ts/runtime/program_buffers.ts`, emitted to
	  `dist/runtime/program_buffers.cjs`, behind that
  facade, keeping Program/Session placement contracts aligned while host
  adapters keep native buffer construction and FFI descriptor packing.
	  NativeBuffer edge policy now lives in
	  `src/ts/runtime/native_buffer.ts`, emitted to
	  `dist/runtime/native_buffer.cjs`, behind the same facade: WebGPU import
  source normalization, external-resource option validation, create/wrap
  byte-length validation, frozen NativeBuffer byte-range evidence for
  write/readback calls, standalone byte-range predicates/assertions/signature
  matching, caller-owned readback target validation, Session-owned buffer
  retain/dedup rules, and deterministic free/dispose behavior stay shared while
  adapters supply concrete native
  callbacks. Shared shape algebra now lives in `src/ts/core/shape.ts`, with
  package-ready JS/declarations emitted under `dist/core/`: scalar
  counts, tensor shape validation, factory/view shape normalization, row-major
  strides, broadcasting plans, dim normalization, reduction plans, and trace
  index/slice bounds share one generated artifact across eager Tensor
  operations, live module trace adaptation, and the JS/TS Trace-to-Program
  Compiler. Tensor data normalization now lives in
  `src/ts/core/tensor_data.ts`, emitted as `dist/core/tensor_data.cjs`,
  so raw f32 conversion, nested rectangular input checks, byte views, scalar
  tensor construction, gradient accumulation, and shaped length validation are
  sourced from TS while Node/Bun keep consuming the generated package artifact.
	  Tensor core policy now lives in `src/ts/core/tensor_core.ts`, emitted as
	  `dist/core/tensor_core.cjs`, so length/rank and iterator/copy surface,
	  scalar extraction, shape/size metadata, `allclose`/`equal`, zeroing gradients,
	  PyTorch-style aliases such as `ndimension`/`nelement`/`zero_grad`,
	  topological `backward`, and JSON serialization share one TS-authored
	  implementation. The Tensor info/core method surface delegates through
	  that same file, so host adapters no longer own parallel length/rank,
	  dtype/device, metadata, requires-grad, scalar, stride, and iterator
	  forwarding tables. In-place detach and JSON serialization forwarding are
	  covered by that same checked surface.
  Tensor factory policy now lives in `src/ts/core/tensor_factory.ts`, emitted as
  `dist/core/tensor_factory.cjs`, so `full`, `zeros`, `ones`, `scalar`,
  deterministic `rand`/`randn` hooks, `linspace`, and `arange` share one
  TS-authored implementation.
  Tensor facade policy now lives in `src/ts/core/tensor_facade.ts`, emitted as
  `dist/core/tensor_facade.cjs`, so Tensor initialization, parameter
  construction, native-buffer conversion, placement delegation, factory/join
  delegation, and `nn` shape-module normalization share one TS-authored
  implementation. The Tensor native edge method surface delegates through that
  same file, so host adapters no longer own parallel `toNativeBuffer`/`place`
  forwarding.
  Tensor static class surface policy now lives in
  `src/ts/core/tensor_static_surface.ts`, emitted as
  `dist/core/tensor_static_surface.cjs`, so `Tensor.fromJSON`,
  factory aliases, root math statics, and `Tensor.cat`/`Tensor.stack` stay
  Node/Bun-identical while each host keeps its own public overload spelling.
	  Tensor join policy now lives in `src/ts/core/tensor_join.ts`, emitted as
	  `dist/core/tensor_join.cjs`, so `cat`/`stack` shape checks, row-major
	  output mapping, and backward gradient routing share one TS-authored
	  implementation.
	  Tensor indexing policy now lives in `src/ts/core/tensor_index.ts`, emitted
	  as `dist/core/tensor_index.cjs`, so flat index lookup, nested-array
	  projection, and flexible `Tensor.set` array/varargs assignment parsing
	  plus the `tolist` alias share one TS-authored implementation. The Tensor
	  indexing method surface delegates through that same file, so host adapters
	  no longer own parallel `get`/`set`/`toArray`/`tolist` forwarding.
		  Tensor view policy now lives in `src/ts/core/tensor_view.ts`, emitted as
	  `dist/core/tensor_view.cjs`, so `clone`/`detach`, reshape/view,
	  broadcast/expand, flatten/squeeze/unsqueeze, transpose, select/narrow/slice,
	  row-major gather maps, and backward routing share one TS-authored
	  implementation. The Tensor view/shape method surface delegates through
	  that same file, so host adapters no longer own parallel view wrapper tables.
	  Tensor math policy now lives in `src/ts/core/tensor_math.ts`, emitted as
	  `dist/core/tensor_math.cjs`, so eager arithmetic, unary activations,
	  matmul, reductions, softmax/logSoftmax, clamp/clip, broadcasting gradients,
	  and reduction gradients share one TS-authored implementation. The Tensor
	  instance math method surface delegates through the same TS file, so Node/Bun
	  no longer own parallel wrapper tables for math forwarding.
	  Tensor indexing policy now lives in `src/ts/core/tensor_index.ts`, emitted as
	  `dist/core/tensor_index.cjs`, so `Tensor.get`, `Tensor.set`,
	  `toArray`, flat row-major index computation, negative coordinate handling, and
  index error messages share one TS-authored implementation.
  Scalar activation math and eager grad-mode helpers now follow the same pattern
  through `src/ts/core/activation.ts` and `src/ts/core/grad_mode.ts`, preserving
  GELU/SiLU/Sigmoid numerics and `noGrad`/`enableGrad` state restoration in
  generated TS artifacts. LLaMA token/window/sample/result helpers now live in
  `src/ts/core/token.ts`, emitted as `dist/core/token.cjs`, so uint32
  token validation, token windows, sampling defaults, output-token validation,
  and ABI token-result shaping are sourced from TS too.
  Tensor Program IR
	  construction and signature policy now live in
	  `src/ts/runtime/tensor_program_ir.ts`, emitted to
	  `dist/runtime/tensor_program_ir.cjs` behind the same facade. KernelPlan
	  lowering, freezing, public projection, layout/signature policy, and
	  descriptor-ready native op scheduling now live in
	  `src/ts/runtime/kernel_plan.ts`, emitted to
	  `dist/runtime/kernel_plan.cjs`. Compile diagnostic construction/freezing now
	  lives in `src/ts/runtime/compile_diagnostics.ts`, emitted to
	  `dist/runtime/compile_diagnostics.cjs`, so Trace-to-IR and Kernelizer
	  diagnostics share one proof-shape policy. Trace compiler orchestration and
	  evidence projection live in `src/ts/runtime/trace_compiler.ts`, emitted to
	  `dist/runtime/trace_compiler.cjs`: record-shaped Trace-to-IR orchestration,
	  frozen compiler evidence, and public compile evidence projection share one
	  Interface. Sequential
	  module compiler policy and support-analysis orchestration now live in
	  `src/ts/runtime/module_compiler_policy.ts`; Sequential parameter packing lives in
	  `src/ts/runtime/module_bindings.ts`; root `nn` facade policy lives in
	  `src/ts/runtime/module_facade.ts`, emitted to
	  `dist/runtime/module_facade.cjs`. Program/module
		  compatibility comparison now lives in `src/ts/runtime/module_compatibility.ts`,
		  Program facade evidence/capability/lifecycle policy now lives in
				  `src/ts/runtime/program_facade_policy.ts`, Program-owned buffer factory routing lives in
				  `src/ts/runtime/program_buffer_factory.ts`, `Program.bindModule(...)`
				  routing lives in `src/ts/runtime/program_module_binding.ts`,
				  `ProgramDevice` info evidence and policy live in `src/ts/runtime/program_device.ts`,
				  Program sizing/signature policy and sizing accessors live in
				  `src/ts/runtime/program_sizing.ts`,
				  Program parameter projection and accessors live in
				  `src/ts/runtime/program_parameters.ts`,
				  Program edge-shape selection and shape accessors live in
				  `src/ts/runtime/program_shapes.ts`,
				  Program resource accessors live in
				  `src/ts/runtime/program_resources.ts`,
				  Program layout accessors live in
				  `src/ts/runtime/program_layout.ts`,
				  Generic and LLaMA-family Program policy accessors live in
				  `src/ts/runtime/program_policy_accessors.ts`,
				  and Program facade factory assembly lives in
				  `src/ts/runtime/program_facade.ts`, so Node and Bun keep host-specific Program
	  construction without owning compatibility diagnostic, capability, or wrapper
	  policy.
  Module Program descriptor projection from KernelPlan evidence,
  descriptor validation, and op-word packing now live in the internal
	  `src/ts/runtime/module_program_desc.ts` Module descriptor packer, which
	  imports the shared NativeModuleOpDesc contract instead of retyping the
	  native op record and delegates descriptor-field validation to the native
	  kernel contract. Adapters do not own compile policy or module descriptor
	  layout rules when packing native descriptors. Generic Program edge-shape and Session tensor-output
	  wrapping/readback helpers now live in `src/ts/runtime/session_tensor.ts`, while
	  compile-envelope validation for backend, context length, batch,
	  default-envelope detection, compile descriptor fields, and Node record
	  projection now comes from `src/ts/runtime/abi.ts`.
  LLaMA bind option normalization, including `output: "native"` / `true`
  output-buffer ownership, lives in the shared session facade helper while
  adapters still create the actual host buffers and native descriptors. The
  same helper now owns the TinyLLaMA and generic LLaMA-family Program bind
  lifecycle: inspection, buffer layout, KV-cache layout, bind option
  normalization, compatible-model/KV-cache/persistent-output bind decision
  policy, owned-output cleanup on native bind failure, and Session factory
  handoff. Node and Bun adapters keep only native descriptor packing and FFI
  symbol calls for each selected bind shape.
  `nn.identity`, `nn.reshape`, `nn.flatten`, `nn.transpose`, `nn.broadcastTo`, `nn.expand`,
  `nn.narrow`, `nn.select`, and step-1 `nn.slice` are traceable and lower for
  rank-1 / rank-2 native module Programs where their shape contract has a
  dense caller-visible output. Rank-2 transpose, broadcast/expand, and batched
  feature-axis narrow/select/slice lower as materialized dense movement ops so
  compiled FFI callers get an explicit output-buffer contract. Deterministic
  no-op shape/Dropout glue before a real op remains visible in trace/IR evidence
  but is recorded as kernel-plan `elidedOps` rather than emitted as a native
  dispatch. Terminal metadata-only shape/Dropout chains with no real native
  work compile as zero-dispatch Programs and retain the skipped chain in
  `elidedOps`. Rank-2
  batch-axis softmax/log-softmax and `sum`/`mean`/`max` reductions reuse that
  movement seam to normalize strided reductions into existing row kernels and
  then materialize the caller-shaped output. The runtime
  `ShapeModule` implementation is shared between Node and Bun through the
  shared frontend core rather than duplicated in each host adapter. Stateless activation and
  softmax/log-softmax/reduction module runtime behavior is shared the same way, as are
  `Dropout`, `Linear`, `Embedding`, `Sequential`, module `train()` / `eval()`
  mode, and feature-axis `LayerNorm`/`RMSNorm` eager/autograd/state/trace/compile
  semantics. `Dropout` is eager/autograd-capable; deterministic no-op Dropout
  (eval mode or `p = 0`) lowers through the native module Program path without
  a dispatch when it is the whole compiled graph, while training-mode stochastic
  Dropout reports structured unsupported compile evidence. JS/TS
  PyTorch-style constructor aliases such as `nn.ReLU`, `nn.Flatten`,
  `nn.Dropout`, and `nn.Identity` are thin names over the same shared module
  classes as the factory helpers, so trace, compile support, state, and placement
  behavior stay identical.
- **Tensor Placement Interface**: the future public tensor/device movement
  vocabulary that lets users move from eager host tensors to `Program` and
  `NativeBuffer` execution without learning backend binding records. It should
  feel like part of the PyTorch-like Frontend, while still preserving explicit
  runtime evidence. JS/TS now has the first host-buffer rung:
  eager tensors report `dtype: "f32"` and `device: "cpu"`, `Tensor.to(...)`
  only acknowledges honest CPU/f32 movement or differentiable host copies, and
  non-CPU movement goes through `Tensor.toNativeBuffer()`,
  `Tensor.place(program, kind)`, and `Tensor.fromNativeBuffer(...)`. The
	  `src/ts/runtime/tensor_placement.ts` Module, emitted to
	  `dist/runtime/tensor_placement.cjs`, owns the shared placement/bind
	  validation policy behind that public vocabulary, while
	  `src/ts/runtime/native_buffer.ts`, emitted to
	  `dist/runtime/native_buffer.cjs` and exposed through `zgml/native_buffer`,
	  owns raw buffer construction and byte/read/write edge policy. The public
  `zgml/program_device` subpath exposes the TS-owned ProgramDevice placement
  factory from `src/ts/program_device.ts`, emitted to `dist/program_device.cjs`.
  TS Interface also exposes a closed `webgpuInterop` symbol vocabulary for
  same-device imports. `src/ts/adapters/webgpu_interop.ts` owns those exported
  `Symbol.for("zgml.webgpu.*")` identities for both concrete Node and Bun
  runtimes, so host GPUBuffer-like objects can carry typed placement,
  device-handle, buffer-handle, byte-range, and lazy import-source provenance
  without widening the ordinary tensor surface.
- **Semantic Copy-and-Patch**: zgml's compiled hot-path strategy for LLaMA
  inference. A fixed executable program shape is compiled once, then each
  RuntimeWindow patches token, KV-cache, and valid-attention spans without
  rebuilding the graph or schedule. It is copy-and-patch-JIT shaped, but the
  stencil is a tensor/device program shape rather than generated machine code.
  Semantic LLM recognizers are optimization passes inside the compiler/runtime,
  not a substitute for the general Lazy Tensor IR and Kernelizer path.
- **DeviceProgram**: the backend-facing static program produced from graph
  tensors. It owns the command and buffer shape that backend Adapters compile.
- **ProgramStencil**: the backend-internal executable skeleton derived from a
  `KernelPlan`/`DeviceProgram`. It owns the copied op or command tape, buffer
  shape, bounded runtime update table, command-stream shape evidence, and
  inspection evidence that CPU, Metal, WGPU, Wasm, and stencil backends consume.
  It is the compiled artifact for real Programs, not a separate "stencil path."
- **RuntimeWindow**: the token position and length passed to compiled execution.
  Backend bindings derive KV-cache write positions and valid attention lengths
  from this window.
- **RuntimeBindings**: backend-owned mutable bindings derived from a
  `ProgramStencil`. They patch a `RuntimeWindow` into compiled execution
  without exposing op scanning or derived KV-cache fields to callers. CPU and
  stencil bindings now clone mutable buffer/stencil state per session. Metal has
  the same handle shape and per-session buffers/stencil, and bound execution
  passes an explicit `RuntimeView` through upload, scheduling, command lowering,
  fallback, kernel buffer binding, and host/device transfer.
- **Device Inference Program/Session/StepParams**: the internal lifecycle for
  compiled graph execution. `Program` owns the backend compiled handle, `Session`
  owns persistent bindings plus per-step input/output bindings, and `StepParams`
  carries the runtime window plus output policy for one execution step.
  `Program.compile` keeps bindable tensor/buffer shape while excluding
  persistent tensors from compile-time initial uploads. Persistent bindings can
  be uploaded without executing the program; they are not hot StepParams.
  `Program.inspect()` exposes the compiled Lazy Tensor IR, runtime patch, and
  command-stream shape evidence that callers can read without graph access.
  `Session` owns a backend runtime handle; lower Metal kernel helpers now receive the
  `RuntimeView` directly instead of reading hidden compiled-encoder state.
  Metal command buffers and active command-kind profiling are owned by a
  per-execution encoder context. Normal profile reset/snapshot calls are now
  backend-owned. Metal keeps default Program and bound Session profile windows
  behind their owning locks, and its allocator seam lets tests prove bound
  hot-path execution performs no Zig heap work after Session binding, including
  transformer-shaped decode and fixed-window prefill Program/Session paths.
  JS/TS Program runtime diagnostics are a closed discriminated Interface, so
  callers can branch on `diagnostic.code` and get typed execution-mode or
  dispatch-plan evidence instead of parsing strings or probing optional fields.
- **LLaMA Session**: the owner of model weights, KV caches, host input patching,
  and private plan construction for LLaMA inference.
- **LLaMA Model/Program/Session Facade**: the public `zgml.llm` lifecycle that
  names model loading, program compilation, session binding, and token step
  params without exposing graph or backend internals. `LlamaProgram.bind()`
  now creates an independent runtime session by copying persistent model
  weights, preserving direct quantized GGUF bindings, and allocating separate
  KV caches, position, plans, and profile state. Deeper zero-copy compatible
  checkpoint/KV rebinding remains an internal runtime task. Bound sessions can
  also `stepInto`/`prefillInto` caller-provided logits buffers, matching the
  output ownership shape needed by C, Node, Bun, and Wasm hosts.
- **FFI Token Window**: C, Node, Bun, and Wasm pass LLaMA token windows as
  compact `uint32_t` / `Uint32Array` ids. The C boundary converts them to native
  Zig `usize` slices on the stack before dispatching to the internal LLaMA
  Session, keeping host bindings simple without adding hot-path heap work.
- **FFI Buffer Handle**: `zgml_buffer` is an opaque native-owned host buffer for
  C and Wasm-style embedders. It exposes create/data/size/free ownership without
  exposing tensor internals. It can back existing Session bind/step descriptors,
  but it is not yet a backend device-buffer or zero-copy external-resource API.
- **Benchmark Artifact**: the JSON evidence emitted by `scripts/bench_vs_ggml.sh`.
  It is the contract for ggml/llama.cpp parity, structural dispatch gates, and
  preflight blocker classification.

## Current Shape

The public root exports `Tensor`, `ComputeGraph`, `nn`, `loss`, `optim`,
`train`, and `llm`.
Implementation-only loaders, backends, profiler data, and benchmark helpers live
behind `zgml_internal` or build scripts.

Inference performance work should deepen the Lazy Tensor IR -> Kernelizer ->
ProgramStencil path through the PyTorch-like Frontend. Semantic LLM
copy-and-patch remains important for token windows, KV-cache writes, RoPE, and
quantized projection layouts, but it should appear as compiler/runtime
recognizers on top of the general path rather than as a parallel model-specific
architecture.
The Zig path has started this migration with `TensorProgramIr ->
DeviceOpLowerer -> LoweredDeviceProgram -> DeviceProgram -> Kernelizer`, and
`Program.inspect()` now carries owned value/op/shape evidence from that path.
`DeviceOpLowerer` now consumes `TensorProgramIr`'s normalized `IrOp` /
`ValueId` form for ordinary node ops plus elementwise-chain, layer-norm, and
log-softmax fusions. `TensorProgramIr` now lives behind the curated internal
root as a shared compiler Module instead of a private `DeviceInference`
implementation detail. The next move is to route frontend module tracing and
kernel scheduling through that same Interface.
Training ergonomics should grow through PyTorch-familiar `nn`, `loss`, `optim`,
and `train` facades rather than by expanding the primitive IR or leaking backend
details. JS/TS is the reference product surface for that vocabulary: eager
execution must stay useful, recognized native patterns should compile through
honest Program/Session paths, and the next deepening move is a
Trace-to-Program Compiler plus Kernelizer backed by a Shared JS/TS Frontend
Core and a fuller Tensor Placement Interface. Module compile diagnostics are a
closed discriminated JS/TS Interface too, so `compileSupport()` callers can
branch on trace, IR, Kernelizer, and support-stage failures with typed evidence
instead of accepting arbitrary diagnostic records. Trace/IR ops, Tensor Program
IR values, KernelPlan ops, KernelPlan kernel names, fused and elided shape-op evidence, and
`program.moduleCompatibility(...)` diagnostics are closed vocabularies as well,
and their TS Interfaces are discriminated by op/kernel/code so Program binding
preflights are now typed evidence rather than stringly inspection. Public
  trace evidence also freezes op-local target-shape arrays, and JS/TS Tensor
Program IR freezes value dtype, scalar-byte width, shapes, strides, dense
row-major storage-layout evidence, per-op value-edge arrays, and normalized
attrs, so shape-module records are immutable all the way through nested shape
metadata without leaking trace-only parameter lists or trace-derived op typing
into the IR Interface. KernelPlan ops preserve the same scalar type/width
evidence and include it in canonical scheduler signatures, so compatibility does
not collapse typed IR back into shape-only scheduling evidence.
Supported JS/TS module compile support and Program compile evidence now also
carry a frozen `compilerSignatures` record for IR, KernelPlan, memory-layout,
parameter-layout, and buffer-layout signatures, plus the older flat signature
fields as compatibility aliases. The shared Program facade normalizes attached
module compile evidence at the Program seam, deriving missing signatures from
retained IR/KernelPlan artifacts and freezing the canonical record before
`compileEvidence()` or compatibility helpers read it. Compatibility preflights
treat that nested record as canonical, with flat-field fallback for older
evidence, so dataflow/layout
drift fails at the intended `ir-mismatch`, `kernel-plan-mismatch`,
`memory-layout-mismatch`, `parameter-layout-mismatch`, or
`buffer-layout-mismatch` stage without callers reconstructing private
compatibility keys. Those diagnostics carry the signature
kind plus program/module signature values, making the compiler evidence
comparison itself part of the typed Interface. Program compile evidence itself
is now a signed proof record too: `programCompileEvidenceSignature`,
`isProgramCompileEvidence`, `requireProgramCompileEvidence`, assertion aliases,
and signature matchers are exported through `zgml/program` and `zgml/compile`
so retained compile evidence can cross logs, package seams, or FFI boundaries
without becoming an anonymous mutable object. Root
`nn.compilerSignatures(...)`, `nn.tensorProgramIr(...)`, `nn.kernelPlan(...)`,
`nn.memoryLayout(...)`, `nn.bufferLayout(...)`, `nn.inputShape(...)`, `nn.outputShape(...)`,
`nn.shapeConstraints(...)`, and `nn.parameterLayout(...)` expose the same
pre-compile compiler evidence that runtime `Program` objects retain after
compile, so callers do not need to peel open `compileSupport()` or
`compileEvidence()` for common artifact-identity, edge-shape, and layout
queries. The public `zgml/compile` namespace also exposes
`requireCompileSupport` / `require_compile_support` / `assertCompileSupport`
plus `requireCompilePlan` / `require_compile_plan` / `assertCompilePlan`, so
cold compile preflight returns supported evidence, module-owned preflight can
return a full compile explanation, and unsupported targets throw a reasoned
error before native compile/bind. Root `compile.analyze(...)` now returns signed
`zgml.compile.analysis` evidence with standalone predicates, require/assert
helpers, and signature matchers, so trace/artifact analysis can cross package
or log boundaries without losing its proof shape.
error before callers try to build a Program.
The standalone Module compile-plan predicates/assertions/signature matchers keep
that frozen explanation reusable as evidence after package or FFI handoff.
Module Program descriptor projection now comes from that same frozen
KernelPlan evidence in the shared frontend core, keeping Node and Bun Adapter
locality at FFI packing.
Program
inspection, requirements, execution-capability, and model-compatibility records
are frozen runtime evidence too, as are model, buffer, Session, LLaMA Program,
and KV-cache requirement snapshots; the published TS Interface exposes them as
read-only records rather than mutable result bags. Program and Session
`inputShape()` / `outputShape()` arrays are frozen compiled-edge evidence too.
Eager Tensor shape arrays and `size()` snapshots are frozen shape evidence too,
so caller mutation cannot rewrite the shape bridge into compile support. Runtime
ABI/capability/layout snapshots, supported-checkpoint catalog snapshots, and
runtime profile snapshots are frozen read-only evidence as well; reset calls
mutate backend counters, not previously returned profile records.
Program buffer resource-factory slots and LLaMA KV-cache resource slots are
frozen edge evidence too, so placement callbacks cannot rewrite the inspected
buffer contract they were handed.
The published TS Tensor
Interface also preserves literal shape evidence for factory, reshape/view,
placement-readback, and from-native-buffer paths, giving callers a lightweight
compile-time bridge from eager Tensor code toward shape-specialized
`compileSupport({ inputShape })` and Program binding.
Authored package source is now intentionally TS-first: `src/ts/**` rejects
hand-authored JS/CJS/MJS/CTS/MTS files and only permits the narrow host ambient
`.d.ts` files needed to probe Node/Bun sources before packaging. Shared TS
facade contracts are being simplified around readonly shapes and structural
adapter callbacks so the Bun/native FFI runtime now joins the same strict
source build instead of living as a separately mirrored surface. `npm run
build:ts-source` includes `src/ts/adapters/bun_ffi_runtime.ts`; tsdown remains
responsible for emitted runtime and declaration artifacts.
