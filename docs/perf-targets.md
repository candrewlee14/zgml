# Apple-First GGML Parity Targets

This is the checked-in performance contract for the GGML parity push. zgml does
not claim parity until it is within 10% of llama.cpp on the same Apple Silicon
machine for both prompt/prefill and decode.

## Current SmolLM-135M Baseline

Measured on Apple Silicon using `scripts/bench_vs_ggml.sh 128 200 3`.

| Engine | Format/backend | Prompt/prefill | Decode |
| --- | --- | ---: | ---: |
| zgml | safetensors, current session path | ~986 tok/s | ~208 tok/s |
| llama.cpp | GGUF Q8_0, Metal | ~9,800 tok/s | ~237 tok/s |

Interpretation:

- Decode is close enough to be worth tightening immediately: roughly 12-13% behind llama.cpp.
- Prefill is the real parity blocker: roughly 10x behind llama.cpp.
- A recent device profile showed roughly 1,500 CPU-side ops per token in the SmolLM decode program, so backend lowering and fused execution must become visible in profiles, not hidden behind aggregate tok/s.

## Dispatch/Fusion Findings

Current decode planning exposes two useful lowering shapes:

- `qmatvec-rope-attention`: 30 regions, 240/1714 ops. If lowered alone it creates 30 backend islands and 60 CPU/GPU transitions, so it is too small to be a good Metal boundary.
- `decode-layer` / `prefill-layer` stage windows: 30 regions, roughly one transformer layer each. In decode this covers 1713/1714 ops; in prefill it gives the Metal backend a reusable whole-layer lowering target instead of ad hoc qmatmul clusters.

Implication: do not chase one-dispatch-per-op execution. Metal work should target
large region/layer lowerings that keep the decode step on device across the
transformer block, then return to CPU only at explicit boundaries.

Current prefill work has a first real device-only path:

| Path | Prompt/prefill | Runtime placement | Dispatches |
| --- | ---: | --- | ---: |
| Current GGUF session path | ~0.9k tok/s | CPU/Accelerate quant path | n/a |
| Experimental Metal device prefill Q8_0 | ~2.7k tok/s | 100% backend | ~542/call |

Measured with
`./zig-out/bin/bench-llama-smollm data/smollm/SmolLM-135M.Q8_0.gguf 128 1 3 --metal-prefill-device`.
This is roughly 3x the default GGUF path while eliminating fallback for the
prefill graph, but it is still not parity. QMatmul batching, producer-sidecar
fusion, movement groups, attention output-store chains, and RoPE-store grouping
now remove about 1,112 dispatches/call. The next target is reusable layer-stage
lowering that cuts dispatch count by at least another order of magnitude without
adding model special cases to the public API.

The stage planning abstraction now lives in `src/backend/program.zig` as a pure
`StagePolicy`: a named anchored region plus a backend pattern id. Metal uses it
for `decode-layer` and `prefill-layer` schedules with seven projection anchors.
This is a structural cleanup, not a claimed speedup by itself; the current
SmolLM Metal prefill smoke is roughly `542` dispatches/call after command-stream
lowering, RoPE-store grouping, and projection sidecar kernels.

The next layer of the same abstraction is `StageCommand`: a pure, model-agnostic
lowering view over the ops inside a stage. Today it names `row_chain` commands
for RMSNorm + repeated scale + multiply and `rope_chain` commands for RoPE plus
the cache write that consumes it. On SmolLM prefill this currently finds 61
row-chains and 90 RoPE/cache chains, raising the pure command view to 212
estimated saved dispatches already covered by Metal fused kernels.

Projection batching now has the same pure legality layer:
`ProjectionGroupPolicy` finds independent qmatvec/qmatmul anchors, proves when
they can be hoisted into one command, and marks compatible qmatmul cache-store
sidecars. Metal consumes that shared selection plus the dependency and sidecar
predicates for its batched projection kernels, so future movement, RoPE/KV, and
attention commands can follow the same path before adding more backend-specific
pattern code.

That legality layer now reasons in terms of buffer spans instead of whole-buffer
touches. This keeps the command stream pure while making future fusion more
precise for shared workspaces, KV caches, and head-sliced tensors: two ops can
share a buffer and still batch if their read/write spans do not overlap.

The first unified command stream now combines those pure stage commands and
projection groups into a single ordered view. On SmolLM prefill it currently
emits 542 commands for 1,654 ops, including 61 row chains, 30 movement groups,
60 attention output-store chains, 120 projection sidecar chains, and 30
projection groups covering 90 anchors plus 30 sidecars. Additional
producer-sidecar commands cover 30 RoPE-store groups, 30 RoPE-attention-store
chains, and 60 RoPE-attention-store groups.
Metal can consume this stream for those command classes while still falling
back to existing local lowering for commands that are not first-class yet. The
stream also has first-class contiguous batch commands for RoPE,
movement/slice-assign, and attention; SmolLM prefill does not currently expose
those as contiguous runs, but decode and future lowering passes can share the
same command shape. Movement/slice-assign now also has an indexed
`movement_group` command for non-contiguous independent copies that share
source/destination buffers; SmolLM prefill currently finds 30 such groups
covering 60 copy ops.
It can also represent non-contiguous attention groups when all attention inputs
are already available; the current SmolLM prefill trace still reports `0`, which
means attention batching must include the producer movement/slice work rather
than only grouping ready attention ops.
It also has a non-contiguous `elementwise_batch` command backed by pure
dependency checks; this is a reusable command-stream primitive, though the
current SmolLM prefill trace reports `0` such batches because its elementwise ops
are dependency-adjacent rather than independent.

QMatmul side effects are now represented as `projection_chain` commands when a
single projection feeds an immediate elementwise, fused-elementwise, or cache
write sidecar. This lifts Metal's local pair fusers into the same pure command
stream and gives the next batching pass a clear target: teach projection batch
kernels to carry those sidecars without losing the multi-anchor dispatch win.

Attention output assembly now follows the same rule: the cached LLaMA forward
uses a scratch concat buffer that is fully overwritten by per-head stores instead
of emitting a fake zero-fill dependency. The command stream can therefore attach
270 delayed attention output stores to their producer attention ops. Dynamic
slice-store patching now only applies to KV-cache writes, so static
`slice_assign_rows` output stores keep their row offsets after refresh. Metal's
runtime command stream now matches the pure prefill target at ~542
dispatches/call with no backend fallback; the next step is coarser layer command
realization, not more local pair fusers.

The current SmolLM Metal prefill command stream is now materially smaller:
`rope_store_group` batches the remaining RoPE-to-KV-store chains and elides the
intermediate scratch write when the scratch is only live until the store. The
profile reports roughly 542 dispatches/call, 100% backend placement, and no
fallback for the device prefill path. This is still not parity, but it confirms
the durable rule: producer-sidecar lowerings should prove side-effect legality
in the pure planner, then let Metal write the final side effect directly instead
of materializing scratch tensors.

The next projection target should follow that rule rather than growing
model-specific kernels. SmolLM prefill has 210 qmatmul anchors and 150 immediate
compatible projection sidecars: 30 slice stores, 90 simple elementwise tails, and
30 fused-elementwise tails. Today only the slice-store sidecars ride with
batched projection groups; the 120 elementwise/fused tails remain
`projection_chain` commands. The long-term fix is a reusable
producer-sidecar-group abstraction with explicit scratch liveness and optional
primary-output elision, so qmatmul batches can write final elementwise/fused
outputs directly without racing on reused scratch buffers.

The first slice of that direction lets qmatmul projection groups carry simple
elementwise sidecars in the same batched Metal kernel. This is a reusable
producer-sidecar capability, but it does not reduce the current SmolLM dispatch
count because the remaining elementwise/fused projection tails are
dependency-adjacent `projection_chain` commands rather than independent
projection-group members. That is an important boundary: further prefill wins
come from stage-level lowering of producer-consumer subgraphs such as FFN
gate/up/SwiGLU/down and attention assembly, not from endlessly expanding local
pair fusers.

The second slice makes that producer-sidecar rule cheaper in memory and visible
in profiles: qmatmul sidecar kernels now receive a pure planner liveness bit and
can skip writing the primary scratch output when only the fused sidecar observes
it. Current SmolLM prefill reports 90 primary-elidable projection sidecars and
60 that still require the primary output. The dispatch count is unchanged, but
projection groups and projection chains now share the same dead-scratch elision
rule as RoPE-store groups.

## Acceptance Thresholds

SmolLM-135M:

- `pp128`: zgml >= 90% of llama.cpp Metal F16/Q8_0 on the same machine.
- `tg200`: zgml >= 90% of llama.cpp Metal F16/Q8_0 on the same machine.
- Default release build: `zig build -Doptimize=ReleaseFast` passes without WGPU installed.
- WGPU targets build only when requested with `-Duse-wgpu=true`.
- Benchmarks record fallback counts or runtime profile data so CPU fallback cannot masquerade as GPU parity.

1B-class target:

- `pp512`: zgml >= 90% of llama.cpp for equivalent F16 and quantized formats.
- `tg128`: zgml >= 90% of llama.cpp for equivalent F16 and quantized formats.
- Memory use <= 115% of llama.cpp for equivalent quant formats.

## Milestone Gate

After a parity milestone is locked, performance changes should fail CI or release
checks if they regress by more than 5% against the recorded target machine
baseline, unless the regression is explicitly accepted in the benchmark report.
