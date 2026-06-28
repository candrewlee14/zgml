# Frontend Capability Matrix

This is the trust matrix for zgml as a small PyTorch-like replacement for
JS/TS over native Programs. It is intentionally concrete: each row names what a
user can express, how it runs today, and what evidence should move before the
library claims broader replacement status.

Legend:

- `yes`: supported as a first-class path.
- `partial`: supported for important shapes or workflows, with honest limits.
- `no`: not a current claim.

## First-Contact Surface

The root package intentionally separates the small surface users should learn
first from the wider inspectable runtime surface:

```text
simple, zgml, F
tensor, nn, loss, optim, train, data, checkpoint, lazy, compile
zgml.native(...)
```

`Program`, `Session`, native buffers, model-source helpers, and runtime
inspection stay public and stable where they are needed, but they are the
advanced control/evidence lane rather than the first tutorial vocabulary.
`F` is the small functional root value for common neural-network/loss operations
that read better as functions than modules.
`simple` is the checked first-contact runtime subset for examples that should
avoid advanced deployment/evidence names while still using the native-backed
root package.
`zgml.nativeCore()` is the typed runtime proof for the boundary: the friendly
API is TypeScript, tensor runtime and hot-path execution are Zig through the C
ABI, module Programs compile through the Zig module compiler with native
Program-inspection evidence, and supported fixed-shape training uses Zig FFI
kernels.
The shared `frontendManifest` names that native core explicitly with
`moduleCompilerCore: "zig-module-program"`,
`nativeCompileEvidence: "native-program-inspection"`, and
`programInspectionCore: "zig-program-inspection"`, and the generated Zig
`native_substrate_manifest` carries the same fields so the native root cannot
drift into a mirrored frontend by accident.

## User-Facing Capabilities

| Capability | Eager | Autograd | TS Shape Safety | Native Program | Notes |
| --- | --- | --- | --- | --- | --- |
| Tensor factories and metadata | yes | n/a | yes | partial | Tensors expose `dtype: "f32"` and `device: "cpu"`; non-CPU placement is explicit through Program buffers. |
| Elementwise tensor math | yes | yes | partial | partial | Common unary/binary ops are eager/autograd-capable; supported lazy chains lower through native activation/fused-elementwise descriptors. |
| Broadcasting | yes | partial | partial | partial | Trailing-dimension broadcasting and broadcast-gradient reduction exist; symmetric no-grad last-dimension bias broadcasts route through native eager for large tensors. |
| Views and movement ops | yes | partial | yes | partial | Reshape/flatten/squeeze/unsqueeze, transpose/permute, repeat/tile, narrow/select/slice, gather/take/scatter-style helpers exist; native lowering is shape-bounded. |
| Reductions | yes | partial | partial | partial | Dim-aware reductions and rank-3 last-axis native lowering exist for common model shapes. |
| Matmul and linear algebra | yes | yes | yes | yes | `matmul`/`mm`, `nn.linear`, fused bias/activation paths, and selected batched shapes have PyTorch comparison evidence. |
| `einsum` | yes | partial | partial | no | Eager/autograd `einsum` with ellipsis and literal-equation shape inference exists; native lowering is not a broad claim. |
| Modules and containers | yes | yes | yes | partial | `Module`, `Sequential`, `ModuleList`, `ModuleDict`, `ParameterList`, `ParameterDict`, traversal, named parameters, and buffers exist. |
| Common neural layers | yes | partial | yes | partial | Linear, Embedding, Conv2d, pooling, activations, normalization, Dropout, Softmax/LogSoftmax, and shape modules exist; `F.linear`, `F.conv2d`, `F.max_pool2d`, and `F.avgPool2d` cover the common PyTorch-like functional forms; native lowering is strongest for dense/module Program paths. |
| Losses | yes | yes | partial | partial | MSE, BCE variants, cross entropy, NLL, L1, Huber/SmoothL1 exist for training; native lowering depends on the compiled module shape. |
| Optimizers and schedulers | yes | n/a | partial | n/a | SGD, Adam, AdamW, RMSprop, Adagrad, parameter groups, state snapshots, and common schedulers exist in TS. |
| Training helpers | yes | yes | partial | partial | Manual loops, `fitModule`, evaluate, predict, classifier helpers, train/eval mode, and evidence records exist; large-scale training is not the current performance claim. |
| State dicts and checkpoints | yes | n/a | partial | n/a | Module, optimizer, and scheduler state dicts plus JSON-safe checkpoint save/load exist with preflight restore checks and README/example import/export recipes. |
| Program compile support | partial | n/a | yes | yes | Compile support is explicit and inspectable; unsupported graphs should report why instead of silently falling back. |
| Program/Session hot path | n/a | n/a | yes | yes | `zgml.native` is the friendly first-contact handle; `zgml.forInference` remains the explicit readable alias and `compile.forInference` is the namespace form. `zgml.run(model, input, options)` and `zgml.runInto(output, model, input, options)` are one-shot conveniences over the same native Program/Session path, with hot loops still using the reusable handle. Module targets bind through `Program.bindModule`, lazy graph targets can pass explicit Program bindings, and `compile -> Program -> bind -> Session -> step/executeInto` remains available for inspection and deployment control. |
| Node and Bun package use | yes | yes | yes | yes | Root, Node, Bun, adapter, runtime, and type smokes cover the emitted package surface. |
| Browser/Wasm/WebGPU surface | partial | n/a | partial | partial | Browser/Wasm and WebGPU proofs exist for bounded runtime paths; broad frontend WebGPU eager execution is not a current claim. |
| Safetensors/model-source interop | partial | n/a | partial | partial | Model-source helpers support safetensors header/data probing and LLaMA-family loading paths; ordinary TS module weights use state-dict/checkpoint save/load recipes. |

## Replacement Gaps

These are the user-visible gaps that matter most before calling zgml a broad
PyTorch replacement:

- `dtype` and `device` are honest but narrow. The public tensor story is
  effectively f32 CPU eager plus explicit native Program placement.
  Program-backed `Tensor.nativePlacement(...)` carries the Program
  compile-evidence signature and native compiler/inspection provenance, so
  placement is inspectable boundary evidence rather than only a buffer helper.
- Autograd coverage is broad enough for small model workflows, but the project
  still needs deeper operation-by-operation numerical-gradient coverage beyond
  the checked family-level runtime evidence in `docs/frontend-autograd-coverage.md`.
- Native eager tensor storage is not the default. Large tensor performance
  claims should continue to go through compiled Programs/Sessions. The
  `dev:perf:native-eager-gap{,:run,:bun,:bun:run}` microscope now measures the first
  `linear_batched` and `lazy_matmul_add_gelu_batched` targets, plus the
  adjacent `lazy_matmul_add_relu_batched` and
  `lazy_matmul_add_silu_batched`, `lazy_matmul_add_sigmoid_batched`, and
  `lazy_matmul_add_tanh_batched` production-activation targets, row-wise
  scalar `elementwise_mul_batched` / `elementwise_div_batched` /
  `elementwise_minimum_batched` / `elementwise_maximum_batched` and unary
  `elementwise_neg_batched` / `elementwise_abs_batched` /
  `elementwise_sqrt_batched` / `elementwise_reciprocal_batched` /
  `elementwise_rsqrt_batched` / `elementwise_exp_batched` /
  `elementwise_log_batched` / `elementwise_isfinite_batched` /
  bias-style `elementwise_add_row_broadcast_batched` /
  `elementwise_sub_lhs_row_broadcast_batched` / `dot_batched` /
  `reduce_sum_scalar_batched`,
  `softmax_batched` / `log_softmax_batched`, `conv2d_batched`, and
  `max_pool2d_batched` / `avg_pool2d_batched`, as eager TS tensor execution
  versus the relevant Zig-backed native eager or compiled allocation-free path;
  the first native eager storage slices have executable baselines.
  Node and Bun also expose the first stateless native eager primitive:
  `zgml.nativeEager.linearInto`, backed by the `zgml_eager_linear_f32` C ABI,
  for caller-owned f32 `Linear` output. That primitive routes through the shared
  native matmul substrate instead of a JS or ABI-local matmul loop. The advanced
  `weightLayout: "out-in"` option is backed by
  `zgml_eager_linear_transposed_weights_f32`, so PyTorch-shaped `F.linear`
  weights do not need a TS-side transpose before reaching Zig. On Node and Bun,
  eligible `nn.Linear.forward` calls inside `zgml.noGrad(...)` now route through
  that native eager hook as the normal module path; grad-enabled training keeps
  the TS/autograd graph path. The same caller-owned-output shape now covers
  vector dot through `zgml.nativeEager.dotInto`, backed by
  `zgml_eager_dot_f32`; Node no-grad `Tensor.dot(...)` uses that direct Zig path
  for large vectors instead of materializing an intermediate product. Rank-3
  batched matmul is now covered by `zgml.nativeEager.bmmInto`, backed by
  `zgml_eager_bmm_f32`, so no-grad `Tensor.bmm(...)` crosses FFI once for the
  whole batch instead of looping through per-batch matmul calls in TS. The
  native eager gap microscope records
  `nativeEagerModuleForwardMs`, `nativeEagerModuleSpeedup`, and
  `nativeEagerModuleMaxAbsDiff` for that no-grad module lane. Node and Bun now
  also expose `zgml.nativeEager.linearActivationInto` /
  `zgml.native_eager.linear_activation_into`, backed by
  `zgml_eager_linear_activation_f32`, so the `lazy_matmul_add_gelu_batched`
  microscope has a caller-owned native eager path for
  `matmul -> add(bias) -> GELU` instead of only a compiled Program baseline.
  The same fused path is used by eligible no-grad
  `nn.Sequential(Linear, GELU)`, `nn.Sequential(Linear, ReLU)`, and
  `nn.Sequential(Linear, SiLU)`, `nn.Sequential(Linear, Sigmoid)`, and
  `nn.Sequential(Linear, Tanh)` module calls, and the microscope records
  non-null fused-module `nativeEagerModuleForwardMs` rows on Node and Bun.
  Node and Bun also expose `zgml.nativeEager.conv2dInto`, backed by
  `zgml_eager_conv2d_f32`; eligible no-grad `nn.Conv2d.forward` calls now
  route through that Zig kernel, and the microscope includes the same
  `conv2d_batched` workload as direct native eager, normal module forward, and
  compiled Program execution evidence. The PyTorch-like functional surface
  `nn.functional.conv2d` / `F.conv2d` delegates through that same module path,
  so no-grad functional Conv2d can reach the Zig kernel without teaching users
  a lower-level native eager primitive.
  They now expose `zgml.nativeEager.pool2dInto`, backed by
  `zgml_eager_pool2d_f32`, for max/avg pooling inference. Eligible no-grad
  `nn.MaxPool2d.forward` and `nn.AvgPool2d.forward` calls use the Zig kernel;
  grad-enabled training stays on the TS path so backward still has max-index
  and count bookkeeping. The functional aliases
  `nn.functional.max_pool2d`, `nn.functional.maxPool2d`,
  `nn.functional.avg_pool2d`, and `nn.functional.avgPool2d` reuse that module
  boundary and are covered by emitted package smokes and type examples.
  `bench:status` selects the latest ignored native-eager artifact and also
  prints a per-runtime `native-eager-runtime-results:` line with row coverage
  and missing rows, so Node and Bun native eager proof cannot be accidentally
  collapsed into whichever runtime ran last or an older partial artifact.
- Safetensors/model-source interop is strong for runtime paths, while ordinary
  TS module weights now have explicit state-dict/checkpoint save/load recipes;
  broader third-party weight-format adapters remain future work.
- The root public API is still wider than the ideal first-contact surface; users
  should learn the checked first-contact namespaces first and reach for advanced
  runtime evidence only when inspecting or deploying. `zgml` is the canonical
  friendly namespace; `torch` remains a compatibility alias for PyTorch-shaped
  habits.

## Evidence Bar

Raising the frontend replacement score should require one of these:

- a type smoke or package smoke proving a user-visible API;
- a focused runtime or module Program benchmark proving native lowering;
- an autograd test proving backward behavior for a public op;
- a README/tutorial example that exercises the full user workflow;
- a runnable example script when the workflow is better shown as a smoke;
- a scorecard check that keeps the capability from drifting.

Workflow-level replacement confidence is tracked in
`docs/frontend-replacement-confidence.md`. That matrix is deliberately narrower
than this capability table: it asks whether complete user workflows such as
linear regression, classifiers, Conv2d feature models, token heads, and
checkpointed compiled inference have executable evidence.

Raising the performance substrate score should require benchmark artifacts, not
prose. The relevant proof is still `bench:status`, PyTorch comparison artifacts,
frontier/q8 artifacts, and ggml/llama.cpp artifacts.
