# Laptop Model Benchmark Suite

zgml needs a standing suite of real laptop-grade models benched primarily
against PyTorch, not only synthetic kernels, tiny training examples, and one
SmolLM checkpoint. The suite is meant to answer four questions:

- Can zgml load and execute the model family at all?
- Does it produce reference-comparable outputs?
- Is the hot path faster or at least competitive with PyTorch?
- Does the API still feel simple from JS/TS while Zig owns the execution work?

PyTorch is the primary reference for every model in the suite. Secondary
references such as llama.cpp, ONNX Runtime, whisper.cpp, or diffusion-specific
runtimes are useful diagnostics, but they do not make a target suite-ready by
themselves. For LLMs that means PyTorch-compatible weights are the promotion
path; GGUF/llama.cpp comparisons remain secondary systems evidence.

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

No larger model target is PyTorch-suite-ready today. The existing SmolLM-135M
GGUF path is executable and useful, but it is only benchmarked against
llama.cpp, so it is not enough to claim PyTorch-replacement competitiveness.
The next useful step is to graduate at least two larger LLMs and one non-LLM
family against PyTorch:

1. SmolLM2-360M or SmolLM2-1.7B from PyTorch-compatible weights against
   PyTorch Transformers.
2. Qwen3-0.6B from PyTorch-compatible weights against PyTorch Transformers.
3. DistilBERT or MobileNetV3 against PyTorch, with ONNX Runtime allowed only as
   a secondary diagnostic.

Diffusion and voice should stay explicit unsupported targets until the graph
import, op coverage, and host preprocessing stories are real.

## First PyTorch LLM Baseline

`npm run bench:laptop:llm:pytorch:run` records the first PyTorch-primary laptop
LLM artifact for `llm.smollm2_360m.instruct.q8_0`.

Latest local result:

| Model | PyTorch prefill | PyTorch decode | PyTorch load | zgml prefill | zgml decode | zgml load | zgml compile | zgml |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SmolLM2-360M-Instruct | 378.86 tok/s | 68.84 tok/s | 238.71 ms | 81.26 tok/s | 80.51 tok/s | 2272.05 ms | 1978.65 ms | executable: `smollm2-360m`, generated IDs match PyTorch |

This is now an honest execution comparison, not a probe-only placeholder. The
native model path recognizes the PyTorch-compatible BF16 safetensors checkpoint
from its Hugging Face `config.json`, loads it into the compile-time specialized
LLaMA execution path, compiles a CPU session, and generates matching token IDs
for the benchmark prompt. Load/compile latency and prompt prefill throughput are
still the obvious next optimization targets.

## Benchmark Output Contract

Each ready model should write an ignored JSON artifact under `bench-results/`
with:

- model id, model source, local path, format, and checksum when available
- backend, host, CPU/GPU details, thread count, and memory peak
- PyTorch version and optional secondary reference runtime/version
- cold load time, compile time, first-token/first-output time, and hot-path time
- task metric: token/s, images/s, audio seconds/s, or denoise-step ms
- correctness metric: max absolute diff, top-1 agreement, transcript agreement,
  cosine similarity, or task-level convergence

The suite is not a marketing list. If a model is unsupported, the manifest says
so. If a model only has llama.cpp or ONNX Runtime evidence, the manifest says
that too. The point is to make the missing PyTorch-competitive work obvious and
measurable.
