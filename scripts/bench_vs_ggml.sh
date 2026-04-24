#!/usr/bin/env bash
# Benchmark zgml vs ggml/llama.cpp on SmolLM-135M.
#
# Prerequisites:
#   brew install llama.cpp
#   python3 -c "from huggingface_hub import hf_hub_download; \
#     hf_hub_download('mradermacher/SmolLM-135M-GGUF','SmolLM-135M.f16.gguf',local_dir='data/smollm'); \
#     hf_hub_download('mradermacher/SmolLM-135M-GGUF','SmolLM-135M.Q8_0.gguf',local_dir='data/smollm')"
#   zig build -Doptimize=ReleaseFast
#
# Usage:
#   ./scripts/bench_vs_ggml.sh [prompt_tokens] [gen_tokens] [repetitions]
#
# Artifacts:
#   bench-results/smollm-<timestamp>-p<PROMPT>-g<GEN>-r<REPS>.md
#   bench-results/smollm-<timestamp>-p<PROMPT>-g<GEN>-r<REPS>.json

set -euo pipefail

PROMPT=${1:-128}
GEN=${2:-200}
REPS=${3:-3}

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

GGUF_F16="data/smollm/SmolLM-135M.f16.gguf"
GGUF_Q8="data/smollm/SmolLM-135M.Q8_0.gguf"
ZGML_MODEL="${ZGML_MODEL:-$GGUF_Q8}"
ZGML_BIN="./zig-out/bin/bench-llama-smollm"
OUT_DIR="${OUT_DIR:-bench-results}"
STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
BASE="$OUT_DIR/smollm-${STAMP}-p${PROMPT}-g${GEN}-r${REPS}"
MD_OUT="${BASE}.md"
JSON_OUT="${BASE}.json"

mkdir -p "$OUT_DIR"

for f in "$ZGML_MODEL" "$GGUF_F16" "$GGUF_Q8"; do
    [ -f "$f" ] || { echo "Missing: $f. See prerequisites in this script."; exit 1; }
done
command -v llama-bench >/dev/null || { echo "llama-bench not found. Run: brew install llama.cpp"; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found."; exit 1; }
[ -x "$ZGML_BIN" ] || { echo "Missing: $ZGML_BIN. Run: zig build -Doptimize=ReleaseFast"; exit 1; }

DATE_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
MACHINE="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
ZIG_VERSION="$(zig version 2>/dev/null || true)"
ZGML_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
ZGML_STATUS="$(git status --short 2>/dev/null || true)"
LLAMA_BENCH_PATH="$(command -v llama-bench)"
LLAMA_BREW_VERSION="$(brew list --versions llama.cpp 2>/dev/null || true)"
GGML_BREW_VERSION="$(brew list --versions ggml 2>/dev/null || true)"

echo "zgml vs ggml/llama.cpp - SmolLM-135M"
echo "prompt=$PROMPT gen=$GEN reps=$REPS"
echo "date=$DATE_UTC"
echo "machine=$MACHINE"
echo

echo "Running zgml benchmark..."
ZGML_OUT="$("$ZGML_BIN" "$ZGML_MODEL" "$PROMPT" "$GEN" "$REPS" 2>&1)"

echo "Running llama.cpp Metal F16 benchmark..."
GGML_F16_OUT="$(llama-bench -m "$GGUF_F16" -p "$PROMPT" -n "$GEN" -r "$REPS" -o md 2>&1 | grep -E '^\|')"

echo "Running llama.cpp Metal Q8_0 benchmark..."
GGML_Q8_OUT="$(llama-bench -m "$GGUF_Q8" -p "$PROMPT" -n "$GEN" -r "$REPS" -o md 2>&1 | grep -E '^\|')"

echo "Running llama.cpp CPU F16 benchmark..."
GGML_CPU_F16_OUT="$(llama-bench -m "$GGUF_F16" -p "$PROMPT" -n "$GEN" -r "$REPS" -ngl 0 -o md 2>&1 | grep -E '^\|')"

echo "Running llama.cpp CPU Q8_0 benchmark..."
GGML_CPU_Q8_OUT="$(llama-bench -m "$GGUF_Q8" -p "$PROMPT" -n "$GEN" -r "$REPS" -ngl 0 -o md 2>&1 | grep -E '^\|')"

cat > "$MD_OUT" <<EOF
# zgml vs ggml/llama.cpp - SmolLM-135M

- date_utc: \`$DATE_UTC\`
- machine: \`$MACHINE\`
- prompt_tokens: \`$PROMPT\`
- gen_tokens: \`$GEN\`
- repetitions: \`$REPS\`
- zgml_model: \`$ZGML_MODEL\`
- llama_cpp_f16_model: \`$GGUF_F16\`
- llama_cpp_q8_model: \`$GGUF_Q8\`
- zgml_commit: \`$ZGML_COMMIT\`
- zig_version: \`$ZIG_VERSION\`
- llama_bench: \`$LLAMA_BENCH_PATH\`
- llama.cpp_brew: \`$LLAMA_BREW_VERSION\`
- ggml_brew: \`$GGML_BREW_VERSION\`

## zgml

\`\`\`text
$ZGML_OUT
\`\`\`

## llama.cpp Metal F16

$GGML_F16_OUT

## llama.cpp Metal Q8_0

$GGML_Q8_OUT

## llama.cpp CPU F16

$GGML_CPU_F16_OUT

## llama.cpp CPU Q8_0

$GGML_CPU_Q8_OUT

## Worktree

\`\`\`text
$ZGML_STATUS
\`\`\`
EOF

export DATE_UTC MACHINE PROMPT GEN REPS ZGML_MODEL GGUF_F16 GGUF_Q8 ZGML_COMMIT ZGML_STATUS ZIG_VERSION
export LLAMA_BENCH_PATH LLAMA_BREW_VERSION GGML_BREW_VERSION
export ZGML_OUT GGML_F16_OUT GGML_Q8_OUT GGML_CPU_F16_OUT GGML_CPU_Q8_OUT
python3 - <<'PY' > "$JSON_OUT"
import json
import os

data = {
    "benchmark": "smollm-135m",
    "date_utc": os.environ["DATE_UTC"],
    "machine": os.environ["MACHINE"],
    "prompt_tokens": int(os.environ["PROMPT"]),
    "gen_tokens": int(os.environ["GEN"]),
    "repetitions": int(os.environ["REPS"]),
    "metadata": {
        "zgml_commit": os.environ["ZGML_COMMIT"],
        "zgml_status": os.environ["ZGML_STATUS"],
        "zig_version": os.environ["ZIG_VERSION"],
        "zgml_model": os.environ["ZGML_MODEL"],
        "llama_cpp_f16_model": os.environ["GGUF_F16"],
        "llama_cpp_q8_model": os.environ["GGUF_Q8"],
        "llama_bench_path": os.environ["LLAMA_BENCH_PATH"],
        "llama_cpp_brew": os.environ["LLAMA_BREW_VERSION"],
        "ggml_brew": os.environ["GGML_BREW_VERSION"],
    },
    "outputs": {
        "zgml": os.environ["ZGML_OUT"],
        "llama_cpp_metal_f16": os.environ["GGML_F16_OUT"],
        "llama_cpp_metal_q8_0": os.environ["GGML_Q8_OUT"],
        "llama_cpp_cpu_f16": os.environ["GGML_CPU_F16_OUT"],
        "llama_cpp_cpu_q8_0": os.environ["GGML_CPU_Q8_OUT"],
    },
}
print(json.dumps(data, indent=2))
PY

cat "$MD_OUT"
echo
echo "Wrote:"
echo "  $MD_OUT"
echo "  $JSON_OUT"
