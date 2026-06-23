#!/usr/bin/env bash
# Benchmark zgml vs ggml/llama.cpp on SmolLM-135M.
# Requires llama-bench, curl, python3, and a ReleaseFast build.
# Usage: ./scripts/bench_vs_ggml.sh [prompt_tokens] [gen_tokens] [repetitions]
# Useful env: ZGML_EXTRA_ARGS, ZGML_F16_EXTRA_ARGS, ZGML_Q8_EXTRA_ARGS, BENCH_AUTO_DOWNLOAD=0,
# BENCH_BASELINE_JSON=<artifact.json>, BENCH_REQUIRE_PARITY=1,
# BENCH_ALLOW_QUARANTINED=1 for local smoke loops that should write artifacts
# without failing the shell command on known parity/perf misses,
# BENCH_ZGML_SAMPLES=1 for a quick smoke, BENCH_BUILD_ZGML=0 to reuse
# an already-built zig-out/bin/bench-llama-smollm.
# Artifact: bench-results/smollm-<timestamp>-p<PROMPT>-g<GEN>-r<REPS>.json

set -euo pipefail

PROMPT=${1:-128}
GEN=${2:-200}
REPS=${3:-3}

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$ROOT/.zig-cache/pycache}"

GGUF_F16="data/smollm/SmolLM-135M.f16.gguf"
GGUF_Q8="data/smollm/SmolLM-135M.Q8_0.gguf"
ZGML_F16_MODEL="${ZGML_F16_MODEL:-$GGUF_F16}"
ZGML_Q8_MODEL="${ZGML_Q8_MODEL:-${ZGML_MODEL:-$GGUF_Q8}}"
ZGML_DEFAULT_EXTRA_ARGS="${ZGML_DEFAULT_EXTRA_ARGS:---metal-prefill-device --metal-decode-region --gate-only}"
ZGML_EXTRA_ARGS="${ZGML_EXTRA_ARGS-$ZGML_DEFAULT_EXTRA_ARGS}"
ZGML_F16_EXTRA_ARGS="${ZGML_F16_EXTRA_ARGS:-}"
ZGML_Q8_EXTRA_ARGS="${ZGML_Q8_EXTRA_ARGS:---metal-prompt-projection-row-chain-command}"
HF_GGUF_REPO="${HF_GGUF_REPO:-mradermacher/SmolLM-135M-GGUF}"
BENCH_AUTO_DOWNLOAD="${BENCH_AUTO_DOWNLOAD:-1}"
BENCH_BASELINE_JSON="${BENCH_BASELINE_JSON:-}"
BENCH_REQUIRE_PARITY="${BENCH_REQUIRE_PARITY:-0}"
BENCH_ALLOW_QUARANTINED="${BENCH_ALLOW_QUARANTINED:-0}"
BENCH_ZGML_SAMPLES="${BENCH_ZGML_SAMPLES:-3}"
BENCH_BUILD_ZGML="${BENCH_BUILD_ZGML:-1}"
if [ "$BENCH_ALLOW_QUARANTINED" != "0" ] && [ "$BENCH_ALLOW_QUARANTINED" != "1" ]; then
    echo "BENCH_ALLOW_QUARANTINED must be 0 or 1"
    exit 1
fi
case "$BENCH_ZGML_SAMPLES" in
    ''|*[!0-9]*) echo "BENCH_ZGML_SAMPLES must be a positive integer"; exit 1 ;;
esac
if [ "$BENCH_ZGML_SAMPLES" -lt 1 ]; then
    echo "BENCH_ZGML_SAMPLES must be a positive integer"
    exit 1
fi
if { [ -n "$BENCH_BASELINE_JSON" ] || [ "$BENCH_REQUIRE_PARITY" = "1" ]; } && [ "$BENCH_ZGML_SAMPLES" -lt 3 ]; then
    echo "BENCH_ZGML_SAMPLES must be >= 3 for parity or baseline gates; set it to 1 only for smoke runs."
    exit 1
fi
ZGML_BIN="./zig-out/bin/bench-llama-smollm"
OUT_DIR="${OUT_DIR:-bench-results}"
STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
BASE="$OUT_DIR/smollm-${STAMP}-p${PROMPT}-g${GEN}-r${REPS}"
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
if [ ! -f "$ZGML_F16_MODEL" ]; then
    echo "Missing custom ZGML_F16_MODEL: $ZGML_F16_MODEL"
    echo "Auto-download currently knows only: $GGUF_F16 and $GGUF_Q8"
    exit 1
fi
if [ ! -f "$ZGML_Q8_MODEL" ]; then
    echo "Missing custom ZGML_Q8_MODEL/ZGML_MODEL: $ZGML_Q8_MODEL"
    echo "Auto-download currently knows only: $GGUF_F16 and $GGUF_Q8"
    exit 1
fi
command -v llama-bench >/dev/null || { echo "llama-bench not found. Run: brew install llama.cpp"; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found."; exit 1; }
if [ "$BENCH_BUILD_ZGML" = "1" ]; then
    zig build -Doptimize=ReleaseFast bench-build >/dev/null
elif [ "$BENCH_BUILD_ZGML" != "0" ]; then
    echo "BENCH_BUILD_ZGML must be 0 or 1"
    exit 1
fi
[ -x "$ZGML_BIN" ] || { echo "Missing: $ZGML_BIN. Run: zig build -Doptimize=ReleaseFast"; exit 1; }

DATE_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
MACHINE="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
ZIG_VERSION="$(zig version 2>/dev/null || true)"
ZGML_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
ZGML_STATUS="$(git status --short 2>/dev/null || true)"
LLAMA_BENCH_PATH="$(command -v llama-bench)"
LLAMA_BREW_VERSION="$(brew list --versions llama.cpp 2>/dev/null || true)"
GGML_BREW_VERSION="$(brew list --versions ggml 2>/dev/null || true)"
LLAMA_CPP_F16_MODEL="$GGUF_F16"
LLAMA_CPP_Q8_MODEL="$GGUF_Q8"

echo "zgml vs ggml/llama.cpp - SmolLM-135M"
echo "prompt=$PROMPT gen=$GEN reps=$REPS"
echo "date=$DATE_UTC"
echo "machine=$MACHINE"
echo

write_preflight_failure_artifact() {
    local label="$1"
    local model="$2"
    local raw="$3"
    local status="$4"
    local reason="$5"
    PREFLIGHT_LABEL="$label" PREFLIGHT_MODEL="$model" PREFLIGHT_RAW="$raw" PREFLIGHT_STATUS="$status" PREFLIGHT_REASON="$reason" \
    python3 - "$JSON_OUT" <<'PY'
import os
import sys
from scripts.bench_contract import write_preflight_failure_artifact as write_artifact

write_artifact(
    sys.argv[1],
    os.environ,
    os.environ["PREFLIGHT_LABEL"],
    os.environ["PREFLIGHT_MODEL"],
    os.environ["PREFLIGHT_RAW"],
    os.environ["PREFLIGHT_STATUS"],
    os.environ["PREFLIGHT_REASON"],
)
PY
    python3 scripts/verify_bench_artifact.py "$JSON_OUT"
    echo
    echo "Wrote preflight failure artifact:"
    echo "  $JSON_OUT"
}

run_zgml() {
    local model="$1"
    local format="$2"
    case "$format" in
        f16)
            if [ "${#ZGML_F16_EXTRA_ARGV[@]}" -gt 0 ]; then
                "$ZGML_BIN" "$model" "$PROMPT" "$GEN" "$REPS" "${ZGML_F16_EXTRA_ARGV[@]}" 2>&1
            else
                "$ZGML_BIN" "$model" "$PROMPT" "$GEN" "$REPS" 2>&1
            fi
            ;;
        q8)
            if [ "${#ZGML_Q8_EXTRA_ARGV[@]}" -gt 0 ]; then
                "$ZGML_BIN" "$model" "$PROMPT" "$GEN" "$REPS" "${ZGML_Q8_EXTRA_ARGV[@]}" 2>&1
            else
                "$ZGML_BIN" "$model" "$PROMPT" "$GEN" "$REPS" 2>&1
            fi
            ;;
        *)
            echo "unknown zgml benchmark format: $format" >&2
            return 1
            ;;
    esac
}

run_zgml_samples() {
    local model="$1"
    local format="$2"
    local i
    for ((i = 0; i < BENCH_ZGML_SAMPLES; i++)); do
        run_zgml "$model" "$format"
    done
}

run_llama_rows() {
    local label="$1"
    local model="$2"
    local raw rows status
    set +e
    raw="$(llama-bench -m "$model" -p "$PROMPT" -n "$GEN" -r "$REPS" -o md 2>&1)"
    status=$?
    set -e
    rows="$(printf '%s\n' "$raw" | grep -E '^\|' || true)"
    if [ "$status" -ne 0 ] || [ -z "$rows" ]; then
        printf '%s\n' "$raw" >&2
        echo "llama-bench $label failed with status $status; check model compatibility and Metal access." >&2
        exit 1
    fi
    printf '%s\n' "$rows"
}

preflight_llama_rows() {
    local label="$1"
    local model="$2"
    local devices raw rows status diag failure_raw reason
    devices="$(llama-bench --list-devices 2>&1)"
    if ! llama_device_list_has_metal "$devices" && ! llama_device_list_loaded_mtl_backend "$devices"; then
        printf '%s\n' "$devices" >&2
        write_preflight_failure_artifact "$label" "$model" "$devices" 0 "llama-bench $label preflight did not list a usable Metal/MTL device"
        echo "llama-bench $label preflight did not list a usable Metal/MTL device; cannot run GGML parity gate." >&2
        exit 1
    fi
    set +e
    raw="$(llama-bench -m "$model" -p 1 -n 1 -r 1 -o md 2>&1)"
    status=$?
    set -e
    rows="$(printf '%s\n' "$raw" | grep -E '^\|' || true)"
    if [ "$status" -ne 0 ] || [ -z "$rows" ]; then
        printf '%s\n' "$raw" >&2
        diag="$(llama_preflight_diagnostics "$model")"
        failure_raw="$(printf '%s\n\n%s\n\nFiltered verbose diagnostics:\n%s' "$devices" "$raw" "$diag")"
        reason="llama-bench $label preflight failed after loading Metal/MTL backend"
        if printf '%s\n' "$failure_raw" | grep -qiE 'ggml_metal_init: error|ggml_backend_metal_device_init_backend|failed to create command queue|failed to initialize .*backend'; then
            reason="llama-bench $label Metal/MTL context preflight failed"
        fi
        write_preflight_failure_artifact "$label" "$model" "$failure_raw" "$status" "$reason"
        echo "$reason; cannot run GGML parity gate." >&2
        exit 1
    fi
    if ! llama_rows_have_metal_reference "$rows"; then
        printf '%s\n' "$rows" >&2
        write_preflight_failure_artifact "$label" "$model" "$(printf '%s\n\n%s' "$devices" "$rows")" 0 "llama-bench $label preflight did not report a Metal/MTL device result"
        echo "llama-bench $label preflight did not report a Metal/MTL device result; cannot run GGML parity gate." >&2
        exit 1
    fi
}

llama_preflight_diagnostics() {
    local model="$1"
    local raw status
    set +e
    raw="$(llama-bench -v -m "$model" -p 1 -n 1 -r 1 -o md 2>&1)"
    status=$?
    set -e
    printf '%s\n' "$raw" | grep -E '^(load_backend:|llama_prepare_model_devices:|ggml_metal_|ggml_backend_metal_|llama_init_from_model:|llama_bench: error:)' || true
    printf 'verbose_status=%s\n' "$status"
    printf 'explicit_blas_smoke:\n'
    set +e
    raw="$(llama-bench -m "$model" -p 1 -n 1 -r 1 -dev BLAS -o md 2>&1)"
    status=$?
    set -e
    printf '%s\n' "$raw" | grep -E '^(load_backend:|\|)' || true
    printf 'blas_status=%s\n' "$status"
}

llama_device_list_has_metal() {
    python3 - "$1" <<'PY'
import re
import sys

in_devices = False
for line in sys.argv[1].splitlines():
    if line.strip() == "Available devices:":
        in_devices = True
        continue
    if in_devices and re.search(r"\b(Metal|MTL)\b", line, re.IGNORECASE):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

llama_device_list_loaded_mtl_backend() {
    python3 - "$1" <<'PY'
import re
import sys

for line in sys.argv[1].splitlines():
    if re.search(r"loaded\s+MTL\s+backend", line, re.IGNORECASE):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

llama_rows_have_metal_reference() {
    python3 - "$1" <<'PY'
import re
import sys

rows = sys.argv[1].splitlines()
header = None
for line in rows:
    if not line.startswith("|"):
        continue
    cells = [c.strip() for c in line.strip("|").split("|")]
    if not cells or all(set(c) <= {"-", ":"} for c in cells):
        continue
    if header is None:
        normalized = [re.sub(r"\s+", " ", c.lower()) for c in cells]
        try:
            backend_i = normalized.index("backend")
        except ValueError:
            continue
        dev_i = normalized.index("dev") if "dev" in normalized else None
        header = (backend_i, dev_i)
        continue
    backend_i, dev_i = header
    backend = cells[backend_i] if backend_i < len(cells) else ""
    dev = cells[dev_i] if dev_i is not None and dev_i < len(cells) else ""
    reference = dev or backend
    if re.search(r"Metal|MTL", reference, re.IGNORECASE):
        raise SystemExit(0)

raise SystemExit(1)
PY
}

ZGML_EXTRA_ARGV=()
ZGML_F16_EXTRA_ARGV=()
ZGML_Q8_EXTRA_ARGV=()
if [ -n "$ZGML_EXTRA_ARGS" ]; then
    read -r -a ZGML_EXTRA_ARGV <<< "$ZGML_EXTRA_ARGS"
fi
ZGML_F16_EXTRA_ARGV=("${ZGML_EXTRA_ARGV[@]}")
ZGML_Q8_EXTRA_ARGV=("${ZGML_EXTRA_ARGV[@]}")
if [ -n "$ZGML_F16_EXTRA_ARGS" ]; then
    ZGML_FORMAT_EXTRA_ARGV=()
    read -r -a ZGML_FORMAT_EXTRA_ARGV <<< "$ZGML_F16_EXTRA_ARGS"
    ZGML_F16_EXTRA_ARGV+=("${ZGML_FORMAT_EXTRA_ARGV[@]}")
fi
if [ -n "$ZGML_Q8_EXTRA_ARGS" ]; then
    ZGML_FORMAT_EXTRA_ARGV=()
    read -r -a ZGML_FORMAT_EXTRA_ARGV <<< "$ZGML_Q8_EXTRA_ARGS"
    ZGML_Q8_EXTRA_ARGV+=("${ZGML_FORMAT_EXTRA_ARGV[@]}")
fi

export DATE_UTC MACHINE PROMPT GEN REPS ZGML_F16_MODEL ZGML_Q8_MODEL ZGML_DEFAULT_EXTRA_ARGS ZGML_EXTRA_ARGS ZGML_F16_EXTRA_ARGS ZGML_Q8_EXTRA_ARGS HF_GGUF_REPO BENCH_AUTO_DOWNLOAD BENCH_ZGML_SAMPLES ZGML_COMMIT ZGML_STATUS ZIG_VERSION
export LLAMA_BENCH_PATH LLAMA_BREW_VERSION GGML_BREW_VERSION BENCH_BASELINE_JSON BENCH_REQUIRE_PARITY
export LLAMA_CPP_F16_MODEL LLAMA_CPP_Q8_MODEL

echo "Preflighting llama.cpp Metal reference..."
preflight_llama_rows F16 "$GGUF_F16"
preflight_llama_rows Q8_0 "$GGUF_Q8"

echo "Running zgml F16 benchmark..."
ZGML_F16_OUT="$(run_zgml_samples "$ZGML_F16_MODEL" f16)"

echo "Running zgml Q8_0 benchmark..."
ZGML_Q8_OUT="$(run_zgml_samples "$ZGML_Q8_MODEL" q8)"

echo "Running llama.cpp F16 benchmark..."
GGML_F16_OUT="$(run_llama_rows F16 "$GGUF_F16")"

echo "Running llama.cpp Q8_0 benchmark..."
GGML_Q8_OUT="$(run_llama_rows Q8_0 "$GGUF_Q8")"

export ZGML_F16_OUT ZGML_Q8_OUT GGML_F16_OUT GGML_Q8_OUT
set +e
GATE_OUT="$(
python3 - "$JSON_OUT" <<'PY'
import json, os, re, sys
from scripts.bench_contract import (
    BASELINE_COUNTER_CEILING,
    BASELINE_FLOOR,
    PARITY_FLOOR,
    GATE_DECODE_LABEL,
    NATIVE_EXECUTION_COMMAND_SHAPES,
    PROMPT_LABEL,
    baseline_counter_fields,
    baseline_mismatches,
    command_metric,
    command_shape_metrics,
    counter_value,
    gate_evidence,
    native_aux_metrics,
    native_command_budget,
    native_command_dispatch_shape,
    native_command_pressure,
    native_lane_evidence,
    native_sidecar_pressure,
    require_gate_lanes,
    require_stored_or_current_native_evidence,
    run_metadata_from_env,
    semantic_expectations,
)

def num(s):
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    return float(m.group(1)) if m else None

def parse_zgml(s):
    samples = {}
    prefix = "ZGML_BENCH_JSON "
    for line in s.splitlines():
        if not line.startswith(prefix):
            continue
        row = json.loads(line[len(prefix):])
        label = " ".join(row.pop("label").split())
        samples.setdefault(label, []).append(row)
    if not samples:
        raise SystemExit("zgml benchmark output missing ZGML_BENCH_JSON rows; rebuild bench-llama-smollm")
    def sample_tok_s(row):
        return row.get("decode_tok_s") if row.get("decode_tok_s") is not None else row.get("prompt_tok_s")
    out = {}
    for label, rows in samples.items():
        rows.sort(key=lambda r: sample_tok_s(r) or 0.0)
        selected = dict(rows[len(rows) // 2])
        vals = [sample_tok_s(r) for r in rows if sample_tok_s(r) is not None]
        if vals:
            mid = vals[len(vals) // 2]
            selected.update({
                "sample_count": len(vals),
                "sample_min_tok_s": vals[0],
                "sample_median_tok_s": mid,
                "sample_max_tok_s": vals[-1],
                "sample_range_pct": None if mid == 0 else (vals[-1] - vals[0]) * 100.0 / mid,
            })
        out[label] = selected
    return out

def parse_llama(s):
    out = {}
    header = None
    for line in s.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or all(set(c) <= {"-", ":"} for c in cells):
            continue
        if header is None:
            normalized = [re.sub(r"\s+", " ", c.lower()) for c in cells]
            try:
                header = (
                    normalized.index("backend"),
                    normalized.index("dev") if "dev" in normalized else None,
                    normalized.index("test"),
                    next(i for i, h in enumerate(normalized) if h in ("t/s", "tok/s", "tokens/s")),
                )
            except (ValueError, StopIteration):
                continue
            continue
        backend_i, dev_i, test_i, tok_i = header
        required = [backend_i, test_i, tok_i]
        if dev_i is not None:
            required.append(dev_i)
        if max(required) >= len(cells):
            continue
        test = cells[test_i]
        if re.fullmatch(r"(pp|tg)\d+", test):
            tok_s = num(cells[tok_i])
            if tok_s is not None:
                row = {"tok_s": tok_s, "backend": cells[backend_i]}
                if dev_i is not None:
                    row["dev"] = cells[dev_i]
                out[test] = row
    if header is None:
        raise SystemExit("llama-bench markdown missing backend/test/t/s columns")
    return out

def llama_row_is_metal(row):
    if not row:
        return False
    dev = row.get("dev")
    reference = dev if dev else row.get("backend", "")
    return bool(re.search(r"Metal|MTL", reference, re.IGNORECASE))

def parse():
    return {
        "zgml_f16": parse_zgml(os.environ["ZGML_F16_OUT"]),
        "zgml_q8_0": parse_zgml(os.environ["ZGML_Q8_OUT"]),
        "llama_cpp_metal_f16": parse_llama(os.environ["GGML_F16_OUT"]),
        "llama_cpp_metal_q8_0": parse_llama(os.environ["GGML_Q8_OUT"]),
    }

def select_lane(rows, field, label):
    row = rows.get(label)
    if row is None:
        return None
    value = row.get(field)
    if value is None:
        return None
    found = {"label": label, "tok_s": value}
    for k, v in row.items():
        if k not in ("prompt_tok_s", "decode_tok_s"):
            found[k] = v
    return found

def phase(gate_row, llama_row):
    llama_tok_s = llama_row.get("tok_s") if llama_row else None
    z = gate_row.get("tok_s") if gate_row else None
    p = z / llama_tok_s if z and llama_tok_s else None
    return {"zgml": gate_row, "llama": llama_row, "llama_tok_s": llama_tok_s, "parity": p}

def annotate_profile_windows(gate, gen, reps):
    expected = {"prompt": reps, "decode": gen * reps}
    for lane in gate.values():
        for phase_name, expected_calls in expected.items():
            row = lane.get(phase_name)
            if not isinstance(row, dict) or "profile_calls" not in row:
                continue
            row["expected_profile_calls"] = expected_calls
            row["profile_calls_match"] = row["profile_calls"] == expected_calls

def annotate_command_pressure(gate):
    for fmt in ("f16", "q8_0"):
        for phase_name in ("prompt", "decode"):
            row = gate.get(fmt, {}).get(phase_name)
            if not isinstance(row, dict):
                continue
            pressure = native_command_pressure(NATIVE_EXECUTION_COMMAND_SHAPES[fmt][phase_name])
            if pressure is not None:
                row["command_pressure_top3_per_call"] = pressure["top"]
                row["command_pressure_total_per_call"] = pressure["total"]
                row["command_pressure_top3_ratio"] = pressure["ratio"]
            row["projection_sidecar_pressure_per_call"] = native_sidecar_pressure(native_aux_metrics(fmt, phase_name))

def summarize(parsed, prompt, gen, reps):
    pp, tg = str(prompt), str(gen)
    f16_prompt_label = PROMPT_LABEL
    q8_prompt_label = PROMPT_LABEL
    f16_extra = os.environ.get("ZGML_F16_EXTRA_ARGS", "")
    q8_extra = os.environ.get("ZGML_Q8_EXTRA_ARGS", "")
    def has_extra_arg(raw, flag):
        return flag in raw.split()
    if has_extra_arg(f16_extra, "--metal-prompt-projection-row-chain-command"):
        f16_prompt_label = "metal scheduled prefill projection-row-chain command"
    elif has_extra_arg(f16_extra, "--metal-prompt-projection-row-chain-command-candidate"):
        f16_prompt_label = "metal scheduled prefill projection-row-chain command candidate"
    if has_extra_arg(q8_extra, "--metal-prompt-projection-row-chain-command"):
        q8_prompt_label = "metal scheduled prefill projection-row-chain command"
    elif has_extra_arg(q8_extra, "--metal-prompt-projection-row-chain-command-candidate"):
        q8_prompt_label = "metal scheduled prefill projection-row-chain command candidate"
    gate = {}
    for fmt, key, prompt_label in (("f16", "zgml_f16", f16_prompt_label), ("q8_0", "zgml_q8_0", q8_prompt_label)):
        gate[fmt] = {
            "prompt": select_lane(parsed[key], "prompt_tok_s", prompt_label),
            "decode": select_lane(parsed[key], "decode_tok_s", GATE_DECODE_LABEL),
        }
    annotate_profile_windows(gate, gen, reps)
    annotate_command_pressure(gate)
    s = {"gate_zgml": gate}
    s["parity_gate_vs_llama_cpp_metal_f16"] = {
        "prompt": phase(gate["f16"]["prompt"], parsed["llama_cpp_metal_f16"].get("pp" + pp)),
        "decode": phase(gate["f16"]["decode"], parsed["llama_cpp_metal_f16"].get("tg" + tg)),
    }
    s["parity_gate_vs_llama_cpp_metal_q8_0"] = {
        "prompt": phase(gate["q8_0"]["prompt"], parsed["llama_cpp_metal_q8_0"].get("pp" + pp)),
        "decode": phase(gate["q8_0"]["decode"], parsed["llama_cpp_metal_q8_0"].get("tg" + tg)),
    }
    return s

def load_baseline(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def tok(v, suffix=""):
    return "n/a" if v is None else f"{v:.3f}{suffix}"

def fallback_text(row):
    pct = row.get("fallback_pct")
    if pct is not None:
        return tok(pct, "%")
    ops = row.get("fallback_ops")
    return "n/a" if ops is None else f"{ops:g} ops"

def fallback_compare(base_row, cur_row):
    if base_row.get("fallback_pct") is not None and cur_row.get("fallback_pct") is not None:
        return f"{tok(base_row.get('fallback_pct'), '%')} -> {tok(cur_row.get('fallback_pct'), '%')}"
    return f"{fallback_text(base_row)} -> {fallback_text(cur_row)}"

parsed = parse()
summary = summarize(parsed, int(os.environ["PROMPT"]), int(os.environ["GEN"]), int(os.environ["REPS"]))
data = {
    "benchmark": "smollm-135m",
    "date_utc": os.environ["DATE_UTC"],
    "machine": os.environ["MACHINE"],
    "prompt_tokens": int(os.environ["PROMPT"]),
    "gen_tokens": int(os.environ["GEN"]),
    "repetitions": int(os.environ["REPS"]),
    "metadata": run_metadata_from_env(os.environ),
    "outputs": {
        "zgml_f16": os.environ["ZGML_F16_OUT"],
        "zgml_q8_0": os.environ["ZGML_Q8_OUT"],
        "llama_cpp_metal_f16": os.environ["GGML_F16_OUT"],
        "llama_cpp_metal_q8_0": os.environ["GGML_Q8_OUT"],
    },
    "lanes": parsed,
    "summary": summary,
}

print("Perf gate: GGML parity")
print("| Target | zgml gate lane | llama.cpp | llama.cpp backend | llama.cpp dev | Parity | Fallback | Gate |")
print("| --- | ---: | ---: | --- | --- | ---: | ---: | --- |")
parity_rows = []
parity_ok = True
reference_backend_rows = []
reference_backend_ok = True
for label, target, phase_name in (
    ("Metal F16 prompt", "llama_cpp_metal_f16", "prompt"),
    ("Metal F16 decode", "llama_cpp_metal_f16", "decode"),
    ("Metal Q8_0 prompt", "llama_cpp_metal_q8_0", "prompt"),
    ("Metal Q8_0 decode", "llama_cpp_metal_q8_0", "decode"),
):
    row = summary["parity_gate_vs_" + target][phase_name]
    z, p = row["zgml"] or {}, row["parity"]
    backend = ((row.get("llama") or {}).get("backend") or "")
    dev = ((row.get("llama") or {}).get("dev") or "")
    is_metal = llama_row_is_metal(row.get("llama"))
    reference_backend_ok = reference_backend_ok and is_metal
    reference_backend_rows.append({"label": label, "backend": backend, "dev": dev, "passed": is_metal})
    fmt = "f16" if "f16" in target else "q8_0"
    evidence = native_lane_evidence(z, fmt, phase_name, int(os.environ["PROMPT"]))
    passed = p is not None and p >= PARITY_FLOOR and is_metal and evidence["passed"]
    parity_ok = parity_ok and passed
    parity_rows.append({
        "label": label,
        "target": target,
        "phase": phase_name,
        "zgml": row["zgml"],
        "llama_tok_s": row["llama_tok_s"],
        "llama_backend": backend,
        "llama_dev": dev,
        "llama_backend_is_metal": is_metal,
        "zgml_no_fallback": evidence["no_fallback"],
        "zgml_profile_window_ok": evidence["profile_calls_match"],
        "zgml_native_evidence_ok": evidence["passed"],
        "parity": p,
        "floor": PARITY_FLOOR,
        "passed": passed,
    })
    print(f"| {label} | {z.get('label', 'n/a')} {tok(z.get('tok_s'), ' tok/s')} | {tok(row['llama_tok_s'], ' tok/s')} | {backend or 'n/a'} | {dev or 'n/a'} | {tok(None if p is None else p * 100, '%')} | {fallback_text(z)} | {'pass' if passed else 'miss'} |")

print("\nPerf gate: native execution evidence")
print("| Format | Phase | Lane | Profile calls | Schedule | Command plans | Dispatch roof | ProgramCommand shape | Gate |")
print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
native_execution_rows = []
native_execution_ok = True
for fmt in ("f16", "q8_0"):
    for phase_name in ("prompt", "decode"):
        row = summary["gate_zgml"][fmt][phase_name] or {}
        expected = row.get("expected_profile_calls")
        command_shape = NATIVE_EXECUTION_COMMAND_SHAPES[fmt][phase_name]
        command_dispatch_shape = native_command_dispatch_shape(fmt, phase_name)
        budget = native_command_budget(fmt, phase_name)
        prompt_tokens = int(os.environ["PROMPT"])
        aux_metrics = native_aux_metrics(fmt, phase_name)
        evidence = native_lane_evidence(row, fmt, phase_name, prompt_tokens)
        native_execution_ok = native_execution_ok and evidence["passed"]
        native_execution_rows.append(evidence | {
            "format": fmt,
            "phase": phase_name,
            "zgml": row,
            "dispatch_budget": budget,
            "program_command_shape": command_shape,
            "program_command_dispatch_shape": command_dispatch_shape,
            "program_command_shape_metrics": command_shape_metrics(command_shape, command_dispatch_shape),
        })
        calls = f"{row.get('profile_calls', 'n/a')}/{expected or 'n/a'}"
        patch_calls = row.get("runtime_patch_calls", "n/a")
        schedule = f"{patch_calls} runtime patch calls"
        if "runtime_patch_stencil_hash" in row:
            schedule += f", stencil {row.get('runtime_patch_stencil_hash')}"
        if phase_name == "decode":
            patched = row.get("runtime_patch_changed", "n/a")
            schedule += f", {patched} changed"
        commands = f"{tok(row.get('region_command_plan_cached_per_call'), '/call')} cached, {row.get('dynamic_region_command_plans', 'n/a')} dynamic"
        roof = f"{row.get('dispatches_per_call', 'n/a')}/{budget['dispatches_per_call']} dispatch, {row.get('commands_per_call', 'n/a')}/{budget['commands_per_call']} command"
        semantic = semantic_expectations(row, phase_name, prompt_tokens) or {}
        structural = ", ".join(
            f"{row.get(command_metric('dispatches', kind), 0)}/{count} dispatched {kind}"
            for kind, count in command_dispatch_shape.items()
        )
        structural += f"; semantic {row.get('semantic_stage_count', 'n/a')}/{semantic.get('semantic_stage_count', 'n/a')} stages, {row.get('semantic_token_count', 'n/a')}/{semantic.get('semantic_token_count', 'n/a')} tokens"
        structural += f"; patch holes {row.get('runtime_patch_holes', 'n/a')}/{semantic.get('runtime_patch_holes', 'n/a')}"
        structural += f" ({row.get('runtime_patch_cache_write_pos_holes', 'n/a')}/{semantic.get('runtime_patch_cache_write_pos_holes', 'n/a')} cache, {row.get('runtime_patch_attention_seq_kv_holes', 'n/a')}/{semantic.get('runtime_patch_attention_seq_kv_holes', 'n/a')} attention)"
        structural += "; aux " + ", ".join(f"{row.get(key, 0)}/{value} {key}" for key, value in aux_metrics.items())
        if evidence.get("extra_program_command_metrics"):
            structural += "; extra " + ", ".join(evidence["extra_program_command_metrics"])
        if evidence.get("extra_aux_metrics"):
            structural += "; extra aux " + ", ".join(evidence["extra_aux_metrics"])
        print(f"| {fmt} | {phase_name} | {row.get('label', 'n/a')} | {calls} | {schedule} | {commands} | {roof} | {structural} | {'pass' if evidence['passed'] else 'miss'} |")

baseline_rows = []
baseline_ok = True
if os.environ["BENCH_BASELINE_JSON"]:
    baseline_ok = False
    base_data = load_baseline(os.environ["BENCH_BASELINE_JSON"])
    mismatches = baseline_mismatches(base_data, data)
    print("\nPerf gate: baseline regression")
    if mismatches:
        baseline_rows.append({"compatible": False, "mismatches": mismatches, "passed": False})
        print("Baseline artifact is incompatible with this run; refusing regression comparison.")
    else:
        try:
            base = require_gate_lanes(base_data, "baseline")
            cur = require_gate_lanes(data, "current")
        except ValueError as err:
            baseline_rows.append({"compatible": False, "schema_error": str(err), "passed": False})
            print(f"Baseline artifact has incompatible gate schema: {err}")
        else:
            native_evidence_note = None
            native_evidence_ok = True
            try:
                require_stored_or_current_native_evidence(base_data, base, "baseline")
            except ValueError as err:
                native_evidence_ok = False
                native_evidence_note = str(err)
            print("| Lane | baseline | current | Allowed floor | Fallback | Gate |")
            print("| --- | ---: | ---: | ---: | ---: | --- |")
            baseline_ok = True
            reference_drift_rows = []
            for fmt in ("f16", "q8_0"):
                for ph in ("prompt", "decode"):
                    base_row = base[fmt][ph]
                    cur_row = cur[fmt][ph]
                    b = base_row.get("tok_s")
                    c = cur_row.get("tok_s")
                    floor = b * BASELINE_FLOOR if b is not None else None
                    base_evidence = gate_evidence(base_row)
                    cur_evidence = gate_evidence(cur_row)
                    fallback_ok = (not base_evidence["no_fallback"]) or cur_evidence["no_fallback"]
                    counter_rows = []
                    counters_ok = True
                    for metric in baseline_counter_fields(base_row):
                        bv = counter_value(base_row, metric, False)
                        if bv is None:
                            continue
                        current_missing_as_zero = metric.startswith("program_command_")
                        cv = counter_value(cur_row, metric, current_missing_as_zero)
                        ceiling = bv * BASELINE_COUNTER_CEILING
                        ok = cv is not None and cv <= ceiling
                        counters_ok = counters_ok and ok
                        counter_rows.append({
                            "metric": metric,
                            "baseline": bv,
                            "current": cv,
                            "ceiling": ceiling,
                            "ceiling_ratio": BASELINE_COUNTER_CEILING,
                            "passed": ok,
                        })
                    passed = c is not None and floor is not None and c >= floor and fallback_ok and counters_ok and cur_evidence["profile_calls_match"] and native_evidence_ok
                    baseline_ok = baseline_ok and passed
                    baseline_rows.append({
                        "format": fmt,
                        "phase": ph,
                        "baseline_tok_s": b,
                        "current_tok_s": c,
                        "floor_tok_s": floor,
                        "floor_ratio": BASELINE_FLOOR,
                        "baseline_no_fallback": base_evidence["no_fallback"],
                        "current_no_fallback": cur_evidence["no_fallback"],
                        "fallback_regression_passed": fallback_ok,
                        "current_profile_window_ok": cur_evidence["profile_calls_match"],
                        "baseline_native_evidence_ok": native_evidence_ok,
                        "counter_rows": counter_rows,
                        "baseline_native_evidence_note": native_evidence_note,
                        "passed": passed,
                    })
                    fallback = fallback_compare(base_row, cur_row)
                    evidence_gate = "pass" if fallback_ok and counters_ok and cur_evidence["profile_calls_match"] and native_evidence_ok else "miss"
                    print(f"| {fmt} {ph} | {tok(b, ' tok/s')} | {tok(c, ' tok/s')} | {tok(floor, ' tok/s')} | {fallback}; evidence {evidence_gate} | {'pass' if passed else 'miss'} |")
                    base_ref = (base_data.get("summary", {}).get(f"parity_gate_vs_llama_cpp_metal_{fmt}", {}).get(ph, {}) or {}).get("llama_tok_s")
                    cur_ref = (data.get("summary", {}).get(f"parity_gate_vs_llama_cpp_metal_{fmt}", {}).get(ph, {}) or {}).get("llama_tok_s")
                    if base_ref and cur_ref:
                        reference_drift_rows.append((fmt, ph, base_ref, cur_ref, cur_ref / base_ref))
            slow_reference_rows = [row for row in reference_drift_rows if row[4] < BASELINE_FLOOR]
            if slow_reference_rows:
                print("\nReference drift diagnostic: llama.cpp reference is also below the baseline floor; the artifact remains quarantined, but this points to system/load drift rather than ProgramCommand structural regression.")
                print("| Lane | baseline llama.cpp | current llama.cpp | Ratio |")
                print("| --- | ---: | ---: | ---: |")
                for fmt, ph, base_ref, cur_ref, ratio in slow_reference_rows:
                    print(f"| {fmt} {ph} | {tok(base_ref, ' tok/s')} | {tok(cur_ref, ' tok/s')} | {tok(ratio * 100, '%')} |")

data["gates"] = {
    "parity": {
        "required": os.environ["BENCH_REQUIRE_PARITY"] == "1",
        "floor": PARITY_FLOOR,
        "passed": parity_ok,
        "rows": parity_rows,
    },
    "reference_backend": {
        "passed": reference_backend_ok,
        "rows": reference_backend_rows,
    },
    "baseline": {
        "baseline_json": os.environ["BENCH_BASELINE_JSON"] or None,
        "floor_ratio": BASELINE_FLOOR,
        "counter_ceiling_ratio": BASELINE_COUNTER_CEILING,
        "passed": baseline_ok,
        "rows": baseline_rows,
    },
    "native_execution": {
        "passed": native_execution_ok,
        "rows": native_execution_rows,
    },
}
data["gates"]["overall_pass"] = parity_ok and reference_backend_ok and baseline_ok and native_execution_ok
data["gates"]["required_pass"] = reference_backend_ok and native_execution_ok and (not data["gates"]["parity"]["required"] or parity_ok) and baseline_ok
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

if not data["gates"]["required_pass"]:
    raise SystemExit(1)
PY
)"
GATE_STATUS=$?
set -e

printf '%s\n' "$GATE_OUT"
echo
if [ "$GATE_STATUS" -eq 0 ]; then
    echo "Wrote accepted artifact:"
    echo "  $JSON_OUT"
else
    FAILED_DIR="$OUT_DIR/failed"
    FAILED_OUT="$FAILED_DIR/$(basename "$JSON_OUT")"
    mkdir -p "$FAILED_DIR"
    if [ -f "$JSON_OUT" ]; then
        mv -f "$JSON_OUT" "$FAILED_OUT"
        echo "Quarantined failed artifact:"
        echo "  $FAILED_OUT"
    else
        echo "No failed artifact was written."
    fi
fi
if [ "$GATE_STATUS" -ne 0 ] && [ "$BENCH_ALLOW_QUARANTINED" = "1" ]; then
    echo "BENCH_ALLOW_QUARANTINED=1: retaining quarantined artifact but returning success for local iteration."
    exit 0
fi
exit "$GATE_STATUS"
