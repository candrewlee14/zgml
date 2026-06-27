# Frontend Replacement Confidence

This file keeps the PyTorch-like replacement claim concrete. It names the
small set of workflows a JS/TS user should expect to work before zgml claims a
broader replacement feel, and ties each workflow to checked evidence instead of
roadmap prose.

The confidence bar is intentionally higher than "the symbol exists":

```text
user workflow -> eager/autograd proof -> checkpoint or state proof -> compiled inference proof
```

Compiled inference does not mean every eager training op lowers natively. It
means the stable inference shape for that workflow can expose
`compileInference`, `compileForInference`, `compileSupport`, `kernelPlan`, or a Program/Session
proof without silent fallback.

The single command for this claim is:

```sh
npm run smoke:frontend-confidence
```

It rebuilds the package, runs the canonical zgml-first training and inference
examples, runs the package smoke that contains the Conv2d and token-head native
Program assertions, and checks this table still points at the runtime evidence.
For a fast static guard while editing docs or smoke labels, use
`npm run check:frontend-confidence`.

| Workflow | User Story | Current Evidence | Remaining Gap |
| --- | --- | --- | --- |
| Linear regression | Train a small regression model with `nn.Linear`, `MSELoss`, `optim`, scheduler state, and checkpoints. | `examples/node_training/train_linear.cjs`; compiled native `Linear + MSE + SGD` training; `train.fitModule`; checkpoint JSON round-trip; `src/ts/smokes/package_smoke_core.ts` fit/checkpoint assertions. | Native eager tensor storage is not yet the default for large tensors. |
| MLP classifier/regressor | Use subclassed `nn.Module` or `Sequential`, train with AdamW, checkpoint, restore, run batched production inputs, and compile the trained model. | `examples/node_training/train_mlp.cjs`; README quickstart; batched plain-array production input parity; typed plain-array `Linear`/`Sequential` forward shape inference; native no-grad `Tensor.matmul` hook; compiled/restored output parity against eager. | Broader TS shape-polymorphism polish for non-literal dynamic arrays. |
| MNIST MLP benchmark smoke | Train a standard MNIST `784 -> 128 -> 10` classifier with cross entropy, assert held-out loss/accuracy, checkpoint restore, and compile a trained single-image inference Program. | `examples/node_training/train_mnist_mlp.cjs`; `smoke:training:mnist`; `bench:mnist:pytorch`; bounded cached IDX download; PyTorch-matched loss/accuracy/logit parity; `compile.trainingStep` native `Linear -> ReLU -> Linear` Adam/AdamW lane at 5.8x faster than the JS training loop in the latest local ReleaseFast artifact; eager/restored/compiled logits parity. | Native training is now real and faster than JS, but the first MLP step compiler remains behind PyTorch on full training throughput. |
| Classifier with cross entropy | Train a classifier with `CrossEntropyLoss`, `fitClassifier`, evaluation/prediction helpers, checkpoint restore, and allocation-free compiled inference. | `examples/node_training/train_classifier.cjs`; README classifier section; `Linear -> LogSoftmax` package smoke Program evidence. | Broader native lowering for larger classifier heads and device/dtype choices. |
| Conv2d feature model | Run Conv2d forward/backward, state dicts, and fixed native-subset compiled Conv2d/Conv2d+ReLU paths. | `examples/node_training/train_conv2d.cjs`; `src/ts/smokes/package_smoke_core.ts` Conv2d eager, gradient, stateDict, compileSupport, and compiled-output assertions; `docs/frontend-autograd-coverage.md` Conv/pool row. | Wider native Conv2d shapes. |
| Embedding/token classifier | Express token models with `Embedding`, `Linear`, `LogSoftmax`, compile support, native Program binding, and eager/compiled parity. | `examples/node_training/train_token_classifier.cjs`; `src/ts/smokes/package_smoke_core.ts` Embedding/Linear/LogSoftmax token-head Program evidence; README token-head notes; checkpoint restore and allocation-free compiled logits. | Broader third-party weight-format import adapters. |
| Checkpointed compiled inference | Train or restore a model, create a JSON-safe checkpoint, compile with the friendly handle, inspect proof, and run into caller-owned output. | README quickstart; `examples/quickstart/zgml-first.cjs`; `examples/node_training/quickstart.cjs` contract smoke; `zgml.native` / `compile.compileForInference` package smoke with `forward`, `stepTensor`, `into`, and `prepareInto`. | Browser/Wasm/WebGPU compiled frontend adapter remains partial. |

## Product Rule

New replacement claims should attach to one of these workflows or add a new
workflow row with comparable evidence. API breadth alone is not enough; each
claim needs a user-visible path, a smoke or example, and an honest native
compile/support story where performance is part of the claim.

`zgml` remains the canonical first-contact namespace, and `simple` is the
checked small runtime subset for examples that should avoid advanced deployment
vocabulary. `torch` compatibility smokes are useful evidence that
PyTorch-shaped habits work, but they do not define the product identity.
