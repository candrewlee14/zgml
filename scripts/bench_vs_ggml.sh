#!/usr/bin/env bash
# Benchmark zgml vs ggml/llama.cpp on SmolLM-135M.
#
# Prerequisites:
#   brew install llama.cpp
#   curl and python3
#   zig build -Doptimize=ReleaseFast
#
# Usage:
#   ./scripts/bench_vs_ggml.sh [prompt_tokens] [gen_tokens] [repetitions]
#
# Optional:
#   ZGML_EXTRA_ARGS="" ./scripts/bench_vs_ggml.sh 128 200 3
#   ZGML_EXTRA_ARGS="--metal-prefill-device --metal-region --profile-timing" ./scripts/bench_vs_ggml.sh 128 200 3
#   BENCH_AUTO_DOWNLOAD=0 ./scripts/bench_vs_ggml.sh 128 200 3
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
ZGML_DEFAULT_EXTRA_ARGS="${ZGML_DEFAULT_EXTRA_ARGS:---metal-prefill-device --metal-region}"
if [ "${ZGML_EXTRA_ARGS+x}" != "x" ]; then
    ZGML_EXTRA_ARGS="$ZGML_DEFAULT_EXTRA_ARGS"
fi
HF_GGUF_REPO="${HF_GGUF_REPO:-mradermacher/SmolLM-135M-GGUF}"
BENCH_AUTO_DOWNLOAD="${BENCH_AUTO_DOWNLOAD:-1}"
ZGML_BIN="./zig-out/bin/bench-llama-smollm"
OUT_DIR="${OUT_DIR:-bench-results}"
STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
BASE="$OUT_DIR/smollm-${STAMP}-p${PROMPT}-g${GEN}-r${REPS}"
MD_OUT="${BASE}.md"
JSON_OUT="${BASE}.json"

mkdir -p "$OUT_DIR"

download_known_model() {
    local path="$1"
    local filename="$2"
    [ -f "$path" ] && return 0
    if [ "$BENCH_AUTO_DOWNLOAD" != "1" ]; then
        echo "Missing: $path. Set BENCH_AUTO_DOWNLOAD=1 or download the model manually."
        exit 1
    fi
    command -v curl >/dev/null || { echo "curl not found; cannot download missing model: $path"; exit 1; }
    mkdir -p "$(dirname "$path")"
    local url="https://huggingface.co/${HF_GGUF_REPO}/resolve/main/${filename}"
    echo "Downloading missing model: $path"
    curl --fail --location --continue-at - --output "$path" "$url"
}

download_known_model "$GGUF_F16" "SmolLM-135M.f16.gguf"
download_known_model "$GGUF_Q8" "SmolLM-135M.Q8_0.gguf"
if [ ! -f "$ZGML_MODEL" ]; then
    echo "Missing custom ZGML_MODEL: $ZGML_MODEL"
    echo "Auto-download currently knows only: $GGUF_F16 and $GGUF_Q8"
    exit 1
fi
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
if [ -n "$ZGML_EXTRA_ARGS" ]; then
    ZGML_EXTRA_ARGV=()
    read -r -a ZGML_EXTRA_ARGV <<< "$ZGML_EXTRA_ARGS"
    ZGML_OUT="$("$ZGML_BIN" "$ZGML_MODEL" "$PROMPT" "$GEN" "$REPS" "${ZGML_EXTRA_ARGV[@]}" 2>&1)"
else
    ZGML_OUT="$("$ZGML_BIN" "$ZGML_MODEL" "$PROMPT" "$GEN" "$REPS" 2>&1)"
fi

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
- zgml_default_extra_args: \`$ZGML_DEFAULT_EXTRA_ARGS\`
- zgml_extra_args: \`$ZGML_EXTRA_ARGS\`
- hf_gguf_repo: \`$HF_GGUF_REPO\`
- bench_auto_download: \`$BENCH_AUTO_DOWNLOAD\`
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

export DATE_UTC MACHINE PROMPT GEN REPS ZGML_MODEL ZGML_DEFAULT_EXTRA_ARGS ZGML_EXTRA_ARGS HF_GGUF_REPO BENCH_AUTO_DOWNLOAD GGUF_F16 GGUF_Q8 ZGML_COMMIT ZGML_STATUS ZIG_VERSION
export LLAMA_BENCH_PATH LLAMA_BREW_VERSION GGML_BREW_VERSION
export ZGML_OUT GGML_F16_OUT GGML_Q8_OUT GGML_CPU_F16_OUT GGML_CPU_Q8_OUT
python3 - <<'PY' > "$JSON_OUT"
import json
import os
import re

def parse_float(text):
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    return float(m.group(1)) if m else None

def parse_zgml(text):
    rows = {}
    line_re = re.compile(
        r"^\s*(?P<label>[^:]+):\s+prompt\s+(?P<prompt>[0-9.]+|[-—]+)\s+tok/s.*?"
        r"decode\s+(?P<decode>[0-9.]+|[-—]+)\s+tok/s",
    )
    for line in text.splitlines():
        m = line_re.search(line)
        if not m:
            continue
        label = " ".join(m.group("label").split())
        rows[label] = {
            "prompt_tok_s": parse_float(m.group("prompt")),
            "decode_tok_s": parse_float(m.group("decode")),
        }
    return rows

def parse_llama_bench_md(text):
    rows = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        test_index = None
        test_name = None
        for i, cell in enumerate(cells):
            if re.fullmatch(r"(pp|tg)\d+", cell):
                test_index = i
                test_name = cell
                break
        if test_index is None:
            continue
        tok_s = None
        for cell in cells[test_index + 1:]:
            tok_s = parse_float(cell)
            if tok_s is not None:
                break
        if tok_s is not None:
            rows[test_name] = {"tok_s": tok_s, "row": cells}
    return rows

def first_metric(rows, prefix):
    for key, value in rows.items():
        if key.startswith(prefix):
            return value.get("tok_s")
    return None

def parity(zgml_rows, llama_rows):
    pp = first_metric(llama_rows, "pp")
    tg = first_metric(llama_rows, "tg")
    out = {}
    for label, row in zgml_rows.items():
        ratios = {}
        if pp and row.get("prompt_tok_s"):
            ratios["prompt"] = row["prompt_tok_s"] / pp
        if tg and row.get("decode_tok_s"):
            ratios["decode"] = row["decode_tok_s"] / tg
        if ratios:
            out[label] = ratios
    return out

zgml = parse_zgml(os.environ["ZGML_OUT"])
llama_metal_f16 = parse_llama_bench_md(os.environ["GGML_F16_OUT"])
llama_metal_q8 = parse_llama_bench_md(os.environ["GGML_Q8_OUT"])
llama_cpu_f16 = parse_llama_bench_md(os.environ["GGML_CPU_F16_OUT"])
llama_cpu_q8 = parse_llama_bench_md(os.environ["GGML_CPU_Q8_OUT"])

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
        "zgml_default_extra_args": os.environ["ZGML_DEFAULT_EXTRA_ARGS"],
        "zgml_extra_args": os.environ["ZGML_EXTRA_ARGS"],
        "hf_gguf_repo": os.environ["HF_GGUF_REPO"],
        "bench_auto_download": os.environ["BENCH_AUTO_DOWNLOAD"],
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
    "parsed": {
        "zgml": zgml,
        "llama_cpp_metal_f16": llama_metal_f16,
        "llama_cpp_metal_q8_0": llama_metal_q8,
        "llama_cpp_cpu_f16": llama_cpu_f16,
        "llama_cpp_cpu_q8_0": llama_cpu_q8,
        "parity_vs_llama_cpp_metal_f16": parity(zgml, llama_metal_f16),
        "parity_vs_llama_cpp_metal_q8_0": parity(zgml, llama_metal_q8),
    },
}
print(json.dumps(data, indent=2))
PY

cat "$MD_OUT"
echo
echo "Wrote:"
echo "  $MD_OUT"
echo "  $JSON_OUT"
