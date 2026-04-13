# zgml

A tensor library for machine learning in Zig. Automatic differentiation, comptime shape checking, and auto-fused kernels.

```zig
var g = ComputeGraph(f32).init(allocator);
defer g.deinit();

const x = try g.param(&.{4, 3});   // learnable [4, 3] tensor
const w = try g.param(&.{4, 4});   // learnable [4, 4] weight

const out = x.mm(w).gelu().layerNorm(&.{1}, 1e-5);
const loss = out.sumAll();

try g.buildForward(loss);
try g.buildBackward(false);
_ = loss.setGrad(1);
g.compute();
// x.grad and w.grad now contain gradients
```

## Features

**Small primitive IR.** A compact set of structural, elementwise, reduction, indexing, and matmul ops. Everything else (softmax, layerNorm, ReLU, sub, div, ...) decomposes into primitives. Backward rules come for free through the chain rule.

**Zero-noise API.** Tensors carry their allocator. No `alloc` parameter on ops:
```zig
// Clean chaining, no allocator threading
const activated = hidden.mm(w1).addBias(b1).gelu();
const ff_out = activated.mm(w2).addBias(b2);
```

**Comptime shape tracking.** `Shaped(f32, .{784, 128})` catches dimension mismatches at compile time:
```zig
const W = try Shaped(f32, .{4, 3}).init(alloc);
const x = try Shaped(f32, .{3, 2}).init(alloc);
const y = W.matMul(false, x, false);  // type is Shaped(f32, .{3, 3})
// const bad = W.matMul(false, W, false);  // compile error: inner dims don't match
```

**Auto-fusion.** `fusionPass()` detects chains of elementwise ops and executes them as comptime-generated, branch-free kernels:
```zig
try g.buildForward(loss);
try g.fusionPass();  // neg+add+mul chains → single-pass kernels
g.compute();
```

**Explicit fusion.** `map`/`map2` for hand-written fused kernels:
```zig
const sq_err = try pred.map2(target, struct {
    fn f(p: f32, t: f32) f32 {
        const d = p - t;
        return d * d;
    }
}.f);
```

## Architecture

```
src/
  tensor.zig          Tensor struct, shape/view metadata, ownership bookkeeping
  tensor/
    api.zig           Lazy graph-building ops (add, mul, softmax, ...)
    forward.zig       SIMD forward compute implementations
    backward.zig      Reverse-mode autodiff rules
    fused.zig         Comptime-generated fused kernels
  graph.zig           ComputeGraph, arena allocator, graph build, fusion pass
  shaped.zig          Compile-time shape-tracked tensor wrapper
  op.zig              Primitive operation enum (17 ops)
  loss.zig            Loss functions (MSE, cross-entropy)
  models/
    linear.zig        Linear regression
    poly.zig          Polynomial regression
    transformer.zig   GPT-2 style transformer block
  optim/
    sgd.zig           SGD with momentum
```

## Design

The graph IR is deliberately small:

- `op.zig` defines the primitive operations and is the source of truth for the current IR surface
- `tensor/api.zig` provides the user-facing API, which is richer — higher-level ops decompose into primitives
- Fusion and graph rewrites happen after graph construction as optimization passes

This keeps the core easy to reason about: the graph stays small, backward rules stay attached to a compact primitive set, user-facing APIs are ergonomic, and performance work happens in fusion passes without bloating the IR.

Tensor runtime state is split into two broad parts:

- hot shape/view metadata (`ne`, `strides`, `storage_offset`, `op`, `data`)
- colder bookkeeping (`role`, data/index ownership, auxiliary index payloads)

That split keeps the common execution path simple while still making aliasing and cleanup rules explicit.

Graph construction uses a visited set while walking parent links, so shared subgraphs are recorded once instead of being rediscovered via repeated linear scans.

### Primitives

The primitive IR is organized into a few stable categories:

- structural/view ops
- elementwise unary and binary ops
- reductions and repeat/broadcast helpers
- indexing and scatter/gather ops
- matrix multiplication

For the exact current primitive set, see `src/op.zig`.

### Sugar (decomposed into primitives)

`sub`, `div`, `sqr`, `relu`, `mean`, `softmax`, `logSoftmax`, `layerNorm`, `scale`, `addBias`

## Building

```bash
zig build test           # run all tests
zig build bench          # run benchmarks (ReleaseFast)
zig build -Duse-blas     # enable BLAS for matmul
```

## Contributing

When adding a new tensor feature:

1. Check whether it can be expressed as a composition of existing primitives
2. If yes, add it in `tensor/api.zig` and let fusion optimize it later
3. Only add a new `Op` when it is a true graph primitive that materially simplifies the IR or unlocks behavior composition cannot express cleanly
