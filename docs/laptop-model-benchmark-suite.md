# Laptop Model Benchmark Suite

zgml needs a standing suite of real laptop-grade models, not only synthetic
kernels, tiny training examples, and one SmolLM checkpoint. The suite is meant
to answer four questions:

- Can zgml load and execute the model family at all?
- Does it produce reference-comparable outputs?
- Is the hot path faster or at least competitive with the obvious native
  reference?
- Does the API still feel simple from JS/TS while Zig owns the execution work?

The manifest lives at `benchmarks/laptop-model-suite.json`. It is deliberately
broader than the implementation today. `zgmlStatus` is the contract:

- `ready`: executable today with checked artifacts.
- `next`: close enough that implementation work should make it a benchmark.
- `target`: architecture is in scope, but more loader/op/runtime work is needed.
- `unsupported-target`: important for the product shape, but not executable yet.

## Required Shape

The model suite should cover these domains:

| Domain | First required target | Why |
| --- | --- | --- |
| LLM | SmolLM-135M, then SmolLM2/Qwen3 laptop models | Proves executable-stencil generation beyond toy shapes |
| Text encoder | DistilBERT/SST-2 | Proves bidirectional attention and classifier workflows |
| Vision | MobileNetV3 Small | Proves real CNN/image workloads |
| Multimodal encoder | CLIP ViT-B/32 | Proves paired text/image embedding workflows |
| Diffusion | TAESD, then SD 1.5 UNet step | Proves image decoder/UNet kernels without starting at the hardest case |
| Voice | Whisper tiny.en, then base.en | Proves audio preprocessing plus encoder-decoder loops |

## Gates

PR gates should stay cheap. Full laptop model gates should be manual or nightly
until they are stable and fast.

| Tier | Expected cadence | Requirement |
| --- | --- | --- |
| `pr` | routine PR evidence | fast, already downloaded, deterministic |
| `laptop_core` | release/manual benchmark | real laptop model, reference comparison, memory metric |
| `frontier` | exploratory | defines missing runtime/compiler work |

## Current Reality

The only ready model in this suite today is the existing SmolLM-135M GGUF path.
That is not enough to claim broad AI-library competitiveness. The next useful
step is to graduate at least two larger LLMs and one non-LLM family:

1. SmolLM2-360M or SmolLM2-1.7B GGUF against llama.cpp.
2. Qwen3-0.6B GGUF against llama.cpp.
3. DistilBERT or MobileNetV3 against PyTorch/ONNX Runtime.

Diffusion and voice should stay explicit unsupported targets until the graph
import, op coverage, and host preprocessing stories are real.

## Benchmark Output Contract

Each ready model should write an ignored JSON artifact under `bench-results/`
with:

- model id, model source, local path, format, and checksum when available
- backend, host, CPU/GPU details, thread count, and memory peak
- reference runtime and version
- cold load time, compile time, first-token/first-output time, and hot-path time
- task metric: token/s, images/s, audio seconds/s, or denoise-step ms
- correctness metric: max absolute diff, top-1 agreement, transcript agreement,
  cosine similarity, or task-level convergence

The suite is not a marketing list. If a model is unsupported, the manifest says
so. The point is to make the missing work obvious and measurable.
