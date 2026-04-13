# zgml
Tensor library for machine learning, inspired by [ggml](https://github.com/ggerganov/ggml).

## Design

`zgml` uses a small graph IR on purpose.

- `src/op.zig` is the primitive operation set used by the computation graph.
- User-facing tensor methods in `src/tensor.zig` are free to be richer than that IR.
- Higher-level operations should usually be expressed as compositions of primitive ops.
- Fusion and graph rewrites happen after graph construction as optimization passes.

That split keeps the core easy to reason about:

- the graph stays small and stable,
- backward rules stay attached to a compact primitive set,
- user-facing APIs can still be ergonomic,
- performance work can happen in fusion/rewrite passes without bloating the IR.

## Contributor Notes

When adding a new tensor feature, prefer this order:

1. Check whether it can be expressed as a composition of existing primitives.
2. If yes, add it in `src/tensor.zig` and let fusion optimize it later.
3. Only add a new `Op` when it is a true graph primitive that materially simplifies the IR or unlocks behavior composition cannot express cleanly.

In other words: do not add a new `Op` just because there is a dedicated eager kernel for it. Eager helpers and graph primitives are different layers.
