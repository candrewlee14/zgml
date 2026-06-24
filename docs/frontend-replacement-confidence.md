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
`compileForInference`, `compileSupport`, `kernelPlan`, or a Program/Session
proof without silent fallback.

| Workflow | User Story | Current Evidence | Remaining Gap |
| --- | --- | --- | --- |
| Linear regression | Train a small regression model with `nn.Linear`, `MSELoss`, `optim`, scheduler state, and checkpoints. | `examples/node_training/train_linear.cjs`; `train.fitModule`; checkpoint JSON round-trip; `src/ts/smokes/package_smoke_core.ts` fit/checkpoint assertions. | Native eager tensor storage is not yet the default for large tensors. |
| MLP classifier/regressor | Use subclassed `nn.Module` or `Sequential`, train with AdamW, checkpoint, restore, and compile the trained model. | `examples/node_training/train_mlp.cjs`; README quickstart; compiled/restored output parity against eager. | More shape-polished examples for batched production inputs. |
| Classifier with cross entropy | Train a classifier with `CrossEntropyLoss`, `fitClassifier`, evaluation/prediction helpers, checkpoint restore, and allocation-free compiled inference. | `examples/node_training/train_classifier.cjs`; README classifier section; `Linear -> LogSoftmax` package smoke Program evidence. | Broader native lowering for larger classifier heads and device/dtype choices. |
| Conv2d feature model | Run Conv2d forward/backward, state dicts, and fixed native-subset compiled Conv2d/Conv2d+ReLU paths. | `src/ts/smokes/package_smoke_core.ts` Conv2d eager, gradient, stateDict, compileSupport, and compiled-output assertions; `docs/frontend-autograd-coverage.md` Conv/pool row. | End-to-end public Conv2d training example and wider native Conv2d shapes. |
| Embedding/token classifier | Express token models with `Embedding`, `Linear`, `LogSoftmax`, compile support, native Program binding, and eager/compiled parity. | `src/ts/smokes/package_smoke_core.ts` Embedding/Linear/LogSoftmax token-head Program evidence; README token-head notes; `examples/node_training/torch_quickstart.cjs` compatibility smoke. | Canonical zgml-first token classifier example and broader weight-format import adapters. |
| Checkpointed compiled inference | Train or restore a model, create a JSON-safe checkpoint, compile with the friendly handle, inspect proof, and run into caller-owned output. | README quickstart; `examples/node_training/quickstart.cjs`; `compile.compileForInference` package smoke with `forward`, `stepTensor`, `into`, and `prepareInto`. | Browser/Wasm/WebGPU compiled frontend adapter remains partial. |

## Product Rule

New replacement claims should attach to one of these workflows or add a new
workflow row with comparable evidence. API breadth alone is not enough; each
claim needs a user-visible path, a smoke or example, and an honest native
compile/support story where performance is part of the claim.

`zgml` remains the canonical first-contact namespace. `torch` compatibility
smokes are useful evidence that PyTorch-shaped habits work, but they do not
define the product identity.
