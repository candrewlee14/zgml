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
