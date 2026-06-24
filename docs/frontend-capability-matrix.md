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
zgml
F
tensor, nn, loss, optim, train, data, checkpoint, lazy, compile
compile.compileForInference(...)
```

`Program`, `Session`, native buffers, model-source helpers, and runtime
inspection stay public and stable where they are needed, but they are the
advanced control/evidence lane rather than the first tutorial vocabulary.
`F` is the small functional root value for common neural-network/loss operations
that read better as functions than modules.

## User-Facing Capabilities

| Capability | Eager | Autograd | TS Shape Safety | Native Program | Notes |
| --- | --- | --- | --- | --- | --- |
| Tensor factories and metadata | yes | n/a | yes | partial | Tensors expose `dtype: "f32"` and `device: "cpu"`; non-CPU placement is explicit through Program buffers. |
| Elementwise tensor math | yes | yes | partial | partial | Common unary/binary ops are eager/autograd-capable; supported lazy chains lower through native activation/fused-elementwise descriptors. |
| Broadcasting | yes | partial | partial | partial | Trailing-dimension broadcasting and broadcast-gradient reduction exist; broad PyTorch-style broadcast coverage should keep expanding through tests. |
| Views and movement ops | yes | partial | yes | partial | Reshape/flatten/squeeze/unsqueeze, transpose/permute, repeat/tile, narrow/select/slice, gather/take/scatter-style helpers exist; native lowering is shape-bounded. |
| Reductions | yes | partial | partial | partial | Dim-aware reductions and rank-3 last-axis native lowering exist for common model shapes. |
| Matmul and linear algebra | yes | yes | yes | yes | `matmul`/`mm`, `nn.linear`, fused bias/activation paths, and selected batched shapes have PyTorch comparison evidence. |
| `einsum` | yes | partial | partial | no | Eager/autograd `einsum` with ellipsis and literal-equation shape inference exists; native lowering is not a broad claim. |
| Modules and containers | yes | yes | yes | partial | `Module`, `Sequential`, `ModuleList`, `ModuleDict`, `ParameterList`, `ParameterDict`, traversal, named parameters, and buffers exist. |
| Common neural layers | yes | partial | yes | partial | Linear, Embedding, Conv2d, pooling, activations, normalization, Dropout, Softmax/LogSoftmax, and shape modules exist; native lowering is strongest for dense/module Program paths. |
| Losses | yes | yes | partial | partial | MSE, BCE variants, cross entropy, NLL, L1, Huber/SmoothL1 exist for training; native lowering depends on the compiled module shape. |
| Optimizers and schedulers | yes | n/a | partial | n/a | SGD, Adam, AdamW, RMSprop, Adagrad, parameter groups, state snapshots, and common schedulers exist in TS. |
| Training helpers | yes | yes | partial | partial | Manual loops, `fitModule`, evaluate, predict, classifier helpers, train/eval mode, and evidence records exist; large-scale training is not the current performance claim. |
| State dicts and checkpoints | yes | n/a | partial | n/a | Module, optimizer, and scheduler state dicts plus JSON-safe checkpoint save/load exist with preflight restore checks and README/example import/export recipes. |
| Program compile support | partial | n/a | yes | yes | Compile support is explicit and inspectable; unsupported graphs should report why instead of silently falling back. |
| Program/Session hot path | n/a | n/a | yes | yes | `compile.compileForInference` is the friendly handle; explicit `compile -> Program -> bind -> Session -> step/executeInto` remains available for inspection and deployment control. |
| Node and Bun package use | yes | yes | yes | yes | Root, Node, Bun, adapter, runtime, and type smokes cover the emitted package surface. |
| Browser/Wasm/WebGPU surface | partial | n/a | partial | partial | Browser/Wasm and WebGPU proofs exist for bounded runtime paths; broad frontend WebGPU eager execution is not a current claim. |
| Safetensors/model-source interop | partial | n/a | partial | partial | Model-source helpers support safetensors header/data probing and LLaMA-family loading paths; ordinary TS module weights use state-dict/checkpoint save/load recipes. |

## Replacement Gaps

These are the user-visible gaps that matter most before calling zgml a broad
PyTorch replacement:

- `dtype` and `device` are honest but narrow. The public tensor story is
  effectively f32 CPU eager plus explicit native Program placement.
- Autograd coverage is broad enough for small model workflows, but the project
  still needs deeper operation-by-operation numerical-gradient coverage beyond
  the checked family-level runtime evidence in `docs/frontend-autograd-coverage.md`.
- Native eager tensor storage is not the default. Large tensor performance
  claims should continue to go through compiled Programs/Sessions.
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
- a scorecard check that keeps the capability from drifting.

Workflow-level replacement confidence is tracked in
`docs/frontend-replacement-confidence.md`. That matrix is deliberately narrower
than this capability table: it asks whether complete user workflows such as
linear regression, classifiers, Conv2d feature models, token heads, and
checkpointed compiled inference have executable evidence.

Raising the performance substrate score should require benchmark artifacts, not
prose. The relevant proof is still `bench:status`, PyTorch comparison artifacts,
frontier/q8 artifacts, and ggml/llama.cpp artifacts.
