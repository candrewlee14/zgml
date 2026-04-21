#!/usr/bin/env bash
# bench_vs_ggml.sh — Benchmark zgml vs ggml (llama.cpp) on SmolLM-135M
#
# Prerequisites:
#   brew install llama.cpp
#   python3 scripts/download_smollm.py
#   python3 -c "from huggingface_hub import hf_hub_download; \
#     hf_hub_download('mradermacher/SmolLM-135M-GGUF','SmolLM-135M.f16.gguf',local_dir='data/smollm'); \
#     hf_hub_download('mradermacher/SmolLM-135M-GGUF','SmolLM-135M.Q8_0.gguf',local_dir='data/smollm')"
#   zig build -Doptimize=ReleaseFast
#
# Usage:
#   ./scripts/bench_vs_ggml.sh [prompt_tokens] [gen_tokens] [repetitions]

set -euo pipefail

PROMPT=${1:-128}
GEN=${2:-200}
REPS=${3:-3}

SAFETENSORS="data/smollm/model.safetensors"
GGUF_F16="data/smollm/SmolLM-135M.f16.gguf"
GGUF_Q8="data/smollm/SmolLM-135M.Q8_0.gguf"
ZGML_BIN="./zig-out/bin/bench-llama-smollm"

echo "=============================================="
echo " zgml vs ggml (llama.cpp) — SmolLM-135M"
echo "=============================================="
echo " prompt=$PROMPT  gen=$GEN  reps=$REPS"
echo " date: $(date)"
echo " machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
echo "=============================================="

# --- Preflight checks ---
for f in "$SAFETENSORS" "$GGUF_F16" "$GGUF_Q8"; do
    [ -f "$f" ] || { echo "Missing: $f — see prerequisites above"; exit 1; }
done
command -v llama-bench >/dev/null || { echo "llama-bench not found — brew install llama.cpp"; exit 1; }
[ -x "$ZGML_BIN" ] || { echo "Missing: $ZGML_BIN — run: zig build -Doptimize=ReleaseFast"; exit 1; }

echo ""
echo ">>> zgml (CPU + WebGPU, Accelerate BLAS, f32 + int8)"
echo "----------------------------------------------"
"$ZGML_BIN" "$SAFETENSORS" "$PROMPT" "$GEN" "$REPS"

echo ""
echo ">>> ggml / llama.cpp — Metal GPU (F16)"
echo "----------------------------------------------"
llama-bench -m "$GGUF_F16" -p "$PROMPT" -n "$GEN" -r "$REPS" -o md 2>&1 | grep -E '^\|'

echo ""
echo ">>> ggml / llama.cpp — Metal GPU (Q8_0)"
echo "----------------------------------------------"
llama-bench -m "$GGUF_Q8" -p "$PROMPT" -n "$GEN" -r "$REPS" -o md 2>&1 | grep -E '^\|'

echo ""
echo ">>> ggml / llama.cpp — CPU only (F16, -ngl 0)"
echo "----------------------------------------------"
llama-bench -m "$GGUF_F16" -p "$PROMPT" -n "$GEN" -r "$REPS" -ngl 0 -o md 2>&1 | grep -E '^\|'

echo ""
echo ">>> ggml / llama.cpp — CPU only (Q8_0, -ngl 0)"
echo "----------------------------------------------"
llama-bench -m "$GGUF_Q8" -p "$PROMPT" -n "$GEN" -r "$REPS" -ngl 0 -o md 2>&1 | grep -E '^\|'

echo ""
echo "=============================================="
echo " Done."
echo "=============================================="
