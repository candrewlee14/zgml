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
| Experimental Metal device prefill Q8_0 | ~2.6-2.7k tok/s | 100% backend | ~1,322/call |

Measured with
`./zig-out/bin/bench-llama-smollm data/smollm/SmolLM-135M.Q8_0.gguf 128 1 3 --metal-prefill-device`.
This roughly doubles `pp128` while eliminating fallback for the prefill graph,
but it is still not parity. QMatmul batching plus qmatmul sidecar fusion
removed about 180 dispatches/call. The next target is reusable layer-stage
lowering that cuts dispatch count by at least an order of magnitude without
adding model special cases to the public API.

The stage planning abstraction now lives in `src/backend/program.zig` as a pure
`StagePolicy`: a named anchored region plus a backend pattern id. Metal uses it
for `decode-layer` and `prefill-layer` schedules with seven projection anchors.
This is a structural cleanup, not a claimed speedup by itself; the current
SmolLM Metal prefill smoke remains at roughly `1,322` dispatches/call.

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
