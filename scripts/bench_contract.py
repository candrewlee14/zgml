"""Shared SmolLM benchmark gate contract."""

import hashlib
import json
import os

PARITY_FLOOR = 0.90
BASELINE_FLOOR = 0.95
BASELINE_COUNTER_CEILING = 1.05


PROMPT_LABEL = "metal scheduled prefill"
GATE_DECODE_LABEL = "metal region decode"

REQUIRED_METADATA = (
    "zgml_f16_model",
    "zgml_q8_model",
    "zgml_default_extra_args",
    "zgml_extra_args",
    "zig_version",
    "llama_bench_path",
    "llama_brew_version",
    "ggml_brew_version",
    "llama_cpp_f16_model",
    "llama_cpp_q8_model",
    "zgml_f16_sha256",
    "zgml_q8_sha256",
    "llama_cpp_f16_sha256",
    "llama_cpp_q8_sha256",
)

RUN_ARTIFACT_FIELDS = ("benchmark", "date_utc", "machine", "prompt_tokens", "gen_tokens", "repetitions", "metadata")
PREFLIGHT_ARTIFACT_FIELDS = RUN_ARTIFACT_FIELDS + ("gates",)
COMPACT_BASELINE_FIELDS = RUN_ARTIFACT_FIELDS + ("summary",)
STENCIL_ARTIFACT_FIELDS = ("benchmark", "prompt_tokens", "backend_capabilities", "rows")
PREFLIGHT_FAILURE_KINDS = ("metal_device_unavailable", "metal_context_init_failed", "metal_result_missing", "llama_bench_failed")
BLOCKED_GATE_NAMES = ("parity", "reference_backend", "baseline", "native_execution")
COMPACT_SUMMARY_FIELDS = ("gate_zgml",)

METADATA_ENV_KEYS = (
    "ZGML_COMMIT",
    "ZGML_STATUS",
    "ZIG_VERSION",
    "ZGML_F16_MODEL",
    "ZGML_Q8_MODEL",
    "ZGML_DEFAULT_EXTRA_ARGS",
    "ZGML_EXTRA_ARGS",
    "ZGML_F16_EXTRA_ARGS",
    "ZGML_Q8_EXTRA_ARGS",
    "HF_GGUF_REPO",
    "BENCH_AUTO_DOWNLOAD",
    "BENCH_BASELINE_JSON",
    "BENCH_REQUIRE_PARITY",
    "BENCH_ZGML_SAMPLES",
    "LLAMA_BENCH_PATH",
    "LLAMA_BREW_VERSION",
    "GGML_BREW_VERSION",
    "LLAMA_CPP_F16_MODEL",
    "LLAMA_CPP_Q8_MODEL",
)

BASELINE_COUNTER_FIELDS = (
    "backend_ops",
    "fallback_ops",
    "placed_ops",
    "dispatches_per_call",
    "commands_per_call",
    "sync_waits",
    "syncs_per_call",
    "schedule_region_failed_ops",
    "schedule_region_failed_ops_per_call",
    "dynamic_region_command_plans",
    "dynamic_region_command_plans_per_call",
    "projection_chain_qmatvec_elementwise_per_call",
    "projection_chain_qmatvec_fused_elementwise_per_call",
    "projection_chain_qmatmul_elementwise_per_call",
    "projection_chain_qmatmul_fused_elementwise_per_call",
)

PROJECTION_CHAIN_SPLIT_FIELDS = (
    "program_command_shape_projection_chains",
    "program_command_shape_dense_projection_chains",
    "program_command_shape_quantized_projection_chains",
)

NATIVE_EXECUTION_COMMAND_SHAPES = {
    "f16": {
        "prompt": {
            "op": 1,
            "row_chain": 61,
            "rope_store_group": 30,
            "rope_attention_store_group": 30,
            "dense_projection_pair_fused_elementwise_chain": 30,
            "dense_projection_chain": 60,
            "dense_projection_cache_group": 30,
        },
        "decode": {
            "op": 1,
            "row_chain": 61,
            "rope_attention_store_group": 30,
            "dense_projection_pair_fused_elementwise_chain": 30,
            "dense_projection_chain": 60,
            "dense_projection_cache_group": 30,
        },
    },
    "q8_0": {
        "prompt": {
            "row_chain": 1,
            "rope_store_group": 30,
            "rope_attention_store_group": 30,
            "projection_pair_fused_elementwise_chain": 30,
            "projection_row_chain": 60,
            "projection_cache_group": 30,
        },
        "decode": {
            "row_chain": 61,
            "rope_attention_store_group": 30,
            "projection_pair_fused_elementwise_chain": 30,
            "projection_chain": 60,
            "projection_cache_group": 30,
        },
    },
}

NATIVE_EXECUTION_COMMAND_DISPATCH_SHAPES = {
    "q8_0": {
        "prompt": {
            "row_chain": 1,
            "rope_store_group": 30,
            "rope_attention_store_group": 30,
            "projection_pair_fused_elementwise_chain": 30,
            "projection_row_chain": 120,
            "projection_cache_group": 30,
        },
    },
}

NATIVE_EXECUTION_EXTRA_DISPATCHES = {
    "f16": {"prompt": 0, "decode": 0},
    "q8_0": {"prompt": 1, "decode": 1},
}

NATIVE_EXECUTION_AUX_METRICS = {
    "f16": {
        "prompt": {"program_op_command_matmul_per_call": 1},
        "decode": {"program_op_command_matmul_per_call": 1},
    },
    "q8_0": {
        "prompt": {},
        "decode": {"projection_chain_qmatvec_elementwise_per_call": 60},
    },
}

SEMANTIC_SHAPE_INPUTS = (
    "semantic_layers",
    "semantic_heads",
    "semantic_kv_heads",
    "semantic_layer_stage_count",
    "semantic_terminal_stage_count",
)


def llama_semantic_shape(layers, heads, kv_heads, layer_stage_count, terminal_stage_count):
    return {
        "semantic_layers": layers,
        "semantic_heads": heads,
        "semantic_kv_heads": kv_heads,
        "semantic_layer_stage_count": layer_stage_count,
        "semantic_terminal_stage_count": terminal_stage_count,
        "semantic_stage_count": layers * layer_stage_count + terminal_stage_count,
        "semantic_runtime_patch_cache_write_pos_holes": layers * 2 * kv_heads,
        "semantic_runtime_patch_attention_seq_kv_holes": layers * heads,
        "semantic_runtime_patch_holes": layers * (2 * kv_heads + heads),
    }


SMOLLM_SEMANTIC_SHAPE = llama_semantic_shape(30, 9, 3, 7, 2)
SMOLLM_STENCIL_HASHES = {
    ("decode", 1): 14405191909906507341,
    ("prompt", 128): 17558208047327870709,
}
SMOLLM_STENCIL_COMMAND_SHAPES = {
    ("decode", 1): {
        "program_command_shape_commands": 181,
        "program_command_shape_covered_ops": 1654,
        "program_command_shape_estimated_saved_dispatches": 1473,
        "program_command_shape_row_chains": 30,
        "program_command_shape_projection_row_chains": 0,
        "program_command_shape_dense_projection_row_chains": 31,
        "program_command_shape_projection_chains": 29,
        "program_command_shape_dense_projection_chains": 29,
        "program_command_shape_quantized_projection_chains": 0,
        "program_command_shape_projection_chain_sidecars": 29,
        "program_command_shape_projection_chain_row_chain_frontiers": 29,
        "program_command_shape_projection_groups": 0,
        "program_command_shape_projection_anchors": 0,
        "program_command_shape_projection_sidecars": 0,
        "program_command_shape_projection_cache_groups": 30,
        "program_command_shape_projection_cache_anchors": 90,
        "program_command_shape_projection_cache_sidecars": 270,
        "program_command_shape_max_projection_span_ops": 25,
        "program_command_shape_stencil_hash": 8226090819949847953,
    },
    ("prompt", 128): {
        "program_command_shape_commands": 211,
        "program_command_shape_covered_ops": 1654,
        "program_command_shape_estimated_saved_dispatches": 1443,
        "program_command_shape_row_chains": 30,
        "program_command_shape_projection_row_chains": 0,
        "program_command_shape_dense_projection_row_chains": 31,
        "program_command_shape_projection_chains": 29,
        "program_command_shape_dense_projection_chains": 29,
        "program_command_shape_quantized_projection_chains": 0,
        "program_command_shape_projection_chain_sidecars": 29,
        "program_command_shape_projection_chain_row_chain_frontiers": 29,
        "program_command_shape_projection_groups": 0,
        "program_command_shape_projection_anchors": 0,
        "program_command_shape_projection_sidecars": 0,
        "program_command_shape_projection_cache_groups": 30,
        "program_command_shape_projection_cache_anchors": 90,
        "program_command_shape_projection_cache_sidecars": 90,
        "program_command_shape_max_projection_span_ops": 25,
        "program_command_shape_stencil_hash": 11864361040785585704,
    },
}

SEMANTIC_ARTIFACT_FIELDS = tuple(SMOLLM_SEMANTIC_SHAPE) + ("semantic_token_count",)
RUNTIME_PATCH_ARTIFACT_FIELDS = (
    "runtime_patch_holes",
    "runtime_patch_cache_write_pos_holes",
    "runtime_patch_attention_seq_kv_holes",
    "runtime_patch_stencil_hash",
)
COMMAND_SHAPE_ARTIFACT_FIELDS = tuple(next(iter(SMOLLM_STENCIL_COMMAND_SHAPES.values())).keys())
STENCIL_ROW_FIELDS = (
    "label",
    "phase",
    "profile_calls",
) + SEMANTIC_ARTIFACT_FIELDS + RUNTIME_PATCH_ARTIFACT_FIELDS + COMMAND_SHAPE_ARTIFACT_FIELDS


def smollm_semantic_shape_passed(row):
    return all(row.get(field) == value for field, value in SMOLLM_SEMANTIC_SHAPE.items())


def smollm_stencil_evidence(row, phase, prompt_tokens):
    token_count = 1 if phase == "decode" else prompt_tokens
    semantic = semantic_expectations(row, phase, prompt_tokens)
    expected_hash = SMOLLM_STENCIL_HASHES.get((phase, token_count))
    expected_command_shape = SMOLLM_STENCIL_COMMAND_SHAPES.get((phase, token_count))
    checks = {
        "smollm_semantic_shape_passed": semantic is not None and smollm_semantic_shape_passed(row),
        "semantic_token_shape_passed": semantic is not None and row.get("semantic_token_count") == token_count,
        "runtime_patch_holes_match": semantic is not None and row.get("runtime_patch_holes") == semantic["runtime_patch_holes"],
        "runtime_patch_shape_match": semantic is not None
        and row.get("runtime_patch_cache_write_pos_holes") == semantic["runtime_patch_cache_write_pos_holes"]
        and row.get("runtime_patch_attention_seq_kv_holes") == semantic["runtime_patch_attention_seq_kv_holes"],
        "runtime_patch_stencil_hash_exact": expected_hash is not None and row.get("runtime_patch_stencil_hash") == expected_hash,
        "program_command_shape_exact": expected_command_shape is not None and all(row.get(key) == value for key, value in expected_command_shape.items()),
        "profile_calls_zero": row.get("profile_calls") == 0,
    }
    checks["passed"] = all(checks.values())
    return checks


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_metadata_from_env(env=os.environ):
    metadata = {key.lower(): env.get(key, "") for key in METADATA_ENV_KEYS}
    for hash_key, path_key in (
        ("zgml_f16_sha256", "zgml_f16_model"),
        ("zgml_q8_sha256", "zgml_q8_model"),
        ("llama_cpp_f16_sha256", "llama_cpp_f16_model"),
        ("llama_cpp_q8_sha256", "llama_cpp_q8_model"),
    ):
        path = metadata.get(path_key)
        if path and os.path.exists(path):
            metadata[hash_key] = file_sha256(path)
    return metadata


def preflight_failure_kind(reason):
    lowered = reason.lower()
    if "did not list a usable metal/mtl device" in lowered:
        return "metal_device_unavailable"
    if "metal/mtl context preflight failed" in lowered:
        return "metal_context_init_failed"
    if "did not report a metal/mtl device result" in lowered:
        return "metal_result_missing"
    return "llama_bench_failed"


def preflight_failure_artifact(env, label, model, raw_output, status, reason):
    return {
        "benchmark": "smollm-135m",
        "date_utc": env["DATE_UTC"],
        "machine": env["MACHINE"],
        "prompt_tokens": int(env["PROMPT"]),
        "gen_tokens": int(env["GEN"]),
        "repetitions": int(env["REPS"]),
        "metadata": run_metadata_from_env(env),
        "gates": {
            "preflight": {
                "passed": False,
                "label": label,
                "model": model,
                "status": int(status),
                "reason": reason,
                "failure_kind": preflight_failure_kind(reason),
                "raw_output": raw_output,
            },
            "parity": {"required": env.get("BENCH_REQUIRE_PARITY") == "1", "floor": PARITY_FLOOR, "passed": False, "rows": []},
            "reference_backend": {"passed": False, "rows": [{"label": label, "passed": False}]},
            "baseline": {"baseline_json": env.get("BENCH_BASELINE_JSON") or None, "floor_ratio": BASELINE_FLOOR, "counter_ceiling_ratio": BASELINE_COUNTER_CEILING, "passed": False, "rows": []},
            "native_execution": {"passed": False, "rows": []},
            "overall_pass": False,
            "required_pass": False,
        },
    }


def write_preflight_failure_artifact(path, env, label, model, raw_output, status, reason):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preflight_failure_artifact(env, label, model, raw_output, status, reason), f, indent=2)
        f.write("\n")


def baseline_mismatches(base, cur):
    mismatches = []
    for key in ("machine", "prompt_tokens", "gen_tokens", "repetitions"):
        if base.get(key) != cur.get(key):
            mismatches.append({"field": key, "baseline": base.get(key), "current": cur.get(key)})
    base_meta, cur_meta = base.get("metadata") or {}, cur.get("metadata") or {}
    for key in REQUIRED_METADATA:
        if base_meta.get(key) != cur_meta.get(key):
            mismatches.append({"field": "metadata." + key, "baseline": base_meta.get(key), "current": cur_meta.get(key)})
    return mismatches


def require_gate_lanes(data, label):
    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{label} is missing summary")
    lanes = summary.get("gate_zgml")
    if not isinstance(lanes, dict):
        raise ValueError(f"{label} is missing summary.gate_zgml")
    for fmt in ("f16", "q8_0"):
        lane = lanes.get(fmt)
        if not isinstance(lane, dict):
            raise ValueError(f"{label} is missing summary.gate_zgml.{fmt}")
        for phase in ("prompt", "decode"):
            row = lane.get(phase)
            if not isinstance(row, dict) or row.get("tok_s") is None:
                raise ValueError(f"{label} is missing summary.gate_zgml.{fmt}.{phase}.tok_s")
            if "fallback_ops" not in row:
                raise ValueError(f"{label} is missing summary.gate_zgml.{fmt}.{phase}.fallback_ops")
            if label in ("baseline", "current") and row.get("sample_count", 0) < 3:
                raise ValueError(f"{label} summary.gate_zgml.{fmt}.{phase} was not sampled at least three times")
            if label == "baseline" and row.get("profile_calls_match") is not True:
                raise ValueError(f"{label} summary.gate_zgml.{fmt}.{phase} is missing matched profile-window evidence")
    return lanes


def baseline_counter_fields(row):
    base_fields = tuple(key for key in BASELINE_COUNTER_FIELDS if key == "placed_ops" or key in row)
    program_command_fields = sorted(
        key
        for key in row
        if key.endswith("_per_call")
        and (
            key.startswith("program_command_attempts_")
            or key.startswith("program_command_encoded_")
            or key.startswith("program_command_dispatches_")
            or key.startswith("program_command_refused_")
        )
    )
    return base_fields + tuple(program_command_fields)


def counter_value(row, metric, missing_as_zero):
    if metric == "placed_ops":
        backend = row.get("backend_ops")
        fallback = row.get("fallback_ops")
        if backend is None or fallback is None:
            return None
        return backend + fallback
    return row.get(metric, 0 if missing_as_zero else None)


def command_metric(action, kind):
    return f"program_command_{action}_{kind}_per_call"


def command_budget_with_extra_dispatch(shape, extra_dispatches=0, dispatch_shape=None):
    command_total = sum(shape.values())
    dispatch_total = sum((dispatch_shape or shape).values())
    return {"dispatches_per_call": dispatch_total + extra_dispatches, "commands_per_call": command_total}


def native_command_dispatch_shape(fmt, phase):
    return NATIVE_EXECUTION_COMMAND_DISPATCH_SHAPES.get(fmt, {}).get(phase, NATIVE_EXECUTION_COMMAND_SHAPES[fmt][phase])


def command_shape_metrics(shape, dispatch_shape=None):
    dispatch_shape = shape if dispatch_shape is None else dispatch_shape
    return {
        **{command_metric("encoded", kind): count for kind, count in shape.items()},
        **{command_metric("dispatches", kind): count for kind, count in dispatch_shape.items()},
    }


def command_attempt_metrics(shape):
    return {
        metric: expected
        for kind, count in shape.items()
        if kind != "op"
        for metric, expected in (
            (command_metric("attempts", kind), count),
            (command_metric("refused", kind), 0),
        )
    }


def extra_program_command_metrics(row, expected):
    prefixes = ("program_command_encoded_", "program_command_dispatches_")
    return sorted(
        key
        for key, value in row.items()
        if key.endswith("_per_call") and key.startswith(prefixes) and key not in expected and (value or 0) != 0
    )


def extra_program_command_attempt_metrics(row, expected):
    prefixes = ("program_command_attempts_", "program_command_refused_")
    return sorted(
        key
        for key, value in row.items()
        if key.endswith("_per_call") and key.startswith(prefixes) and key not in expected and (value or 0) != 0
    )


def extra_aux_metrics(row, expected):
    prefixes = ("program_op_command_", "projection_chain_qmatvec_", "projection_chain_qmatmul_")
    return sorted(
        key
        for key, value in row.items()
        if key.endswith("_per_call") and key.startswith(prefixes) and key not in expected and (value or 0) != 0
    )


def native_command_budget(fmt, phase):
    return command_budget_with_extra_dispatch(
        NATIVE_EXECUTION_COMMAND_SHAPES[fmt][phase],
        NATIVE_EXECUTION_EXTRA_DISPATCHES[fmt][phase],
        native_command_dispatch_shape(fmt, phase),
    )


def native_aux_metrics(fmt, phase):
    return NATIVE_EXECUTION_AUX_METRICS[fmt][phase]


def native_command_pressure(shape, limit=3):
    counts = sorted(shape.values(), reverse=True)
    total = sum(counts)
    if total <= 0:
        return None
    top = sum(counts[:limit])
    return {
        "top": top,
        "total": total,
        "ratio": top / total,
    }


def native_sidecar_pressure(aux_metrics):
    values = [
        value
        for key, value in aux_metrics.items()
        if key.startswith("projection_chain_") and key.endswith("_per_call")
    ]
    return sum(values)


def native_lane_fields(fmt, phase):
    fields = {
        "backend_ops",
        "commands_per_call",
        "dispatches_per_call",
        "dynamic_region_command_plans",
        "dynamic_region_command_plans_per_call",
        "expected_profile_calls",
        "fallback_ops",
        "profile_calls",
        "profile_calls_match",
        "region_command_plan_cached_per_call",
        "runtime_patch_calls",
        "runtime_patch_invalid",
        "sample_count",
        "sample_max_tok_s",
        "sample_median_tok_s",
        "sample_min_tok_s",
        "schedule_region_failed_ops",
        "schedule_region_failed_ops_per_call",
        "sync_waits",
        "syncs_per_call",
        "tok_s",
        "command_pressure_top3_per_call",
        "command_pressure_total_per_call",
        "command_pressure_top3_ratio",
        "projection_sidecar_pressure_per_call",
    } | set(SEMANTIC_ARTIFACT_FIELDS) | set(RUNTIME_PATCH_ARTIFACT_FIELDS)
    fields |= set(command_shape_metrics(NATIVE_EXECUTION_COMMAND_SHAPES[fmt][phase]))
    fields |= set(command_shape_metrics(NATIVE_EXECUTION_COMMAND_SHAPES[fmt][phase], native_command_dispatch_shape(fmt, phase)))
    fields |= set(command_attempt_metrics(NATIVE_EXECUTION_COMMAND_SHAPES[fmt][phase]))
    fields |= set(native_aux_metrics(fmt, phase))
    fields |= set(PROJECTION_CHAIN_SPLIT_FIELDS)
    if phase == "decode":
        fields.add("runtime_patch_changed")
    return fields


def semantic_program_shape(row):
    values = {field: row.get(field) for field in SEMANTIC_SHAPE_INPUTS}
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values.values()):
        return None
    if any(values[field] <= 0 for field in ("semantic_layers", "semantic_heads", "semantic_kv_heads", "semantic_layer_stage_count")):
        return None
    if values["semantic_terminal_stage_count"] < 0 or values["semantic_kv_heads"] > values["semantic_heads"]:
        return None

    expected = llama_semantic_shape(
        values["semantic_layers"],
        values["semantic_heads"],
        values["semantic_kv_heads"],
        values["semantic_layer_stage_count"],
        values["semantic_terminal_stage_count"],
    )
    if not all(row.get(field) == value for field, value in expected.items()):
        return None
    return {
        "semantic_stage_count": expected["semantic_stage_count"],
        "runtime_patch_holes": expected["semantic_runtime_patch_holes"],
        "runtime_patch_cache_write_pos_holes": expected["semantic_runtime_patch_cache_write_pos_holes"],
        "runtime_patch_attention_seq_kv_holes": expected["semantic_runtime_patch_attention_seq_kv_holes"],
    }


def semantic_expectations(row, phase, prompt_tokens=None):
    shape = semantic_program_shape(row)
    if shape is None:
        return None
    return {
        "semantic_stage_count": shape["semantic_stage_count"],
        "semantic_token_count": 1 if phase == "decode" else prompt_tokens,
        "runtime_patch_holes": shape["runtime_patch_holes"],
        "runtime_patch_cache_write_pos_holes": shape["runtime_patch_cache_write_pos_holes"],
        "runtime_patch_attention_seq_kv_holes": shape["runtime_patch_attention_seq_kv_holes"],
    }


def gate_evidence(
    row,
    phase=None,
    budget=None,
    command_shape=None,
    prompt_tokens=None,
    require_runtime_patch_holes=False,
    require_runtime_patch_stencil_hash=False,
    aux_metrics=None,
    command_dispatch_shape=None,
):
    checks = {
        "profile_calls_match": row.get("profile_calls_match") is True,
        "no_fallback": row.get("fallback_ops") == 0,
        "runtime_patch_valid": row.get("runtime_patch_invalid", 0) == 0,
    }
    if phase is not None:
        checks["runtime_patch_calls_match"] = row.get("runtime_patch_calls") == row.get("expected_profile_calls")
        checks["runtime_patch_changed_match"] = phase != "decode" or row.get("runtime_patch_changed") == row.get("expected_profile_calls")
    if budget is not None:
        checks["cached_command_plans"] = row.get("dynamic_region_command_plans") == 0 and (row.get("region_command_plan_cached_per_call") or 0) > 0
        checks["no_schedule_region_failures"] = row.get("schedule_region_failed_ops") == 0 and row.get("schedule_region_failed_ops_per_call") == 0
        checks["dispatch_budget_passed"] = all(row.get(key) is not None and row.get(key) <= value for key, value in budget.items())
    if command_shape is not None:
        expected = command_shape_metrics(command_shape, command_dispatch_shape)
        extras = extra_program_command_metrics(row, expected)
        expected_attempts = command_attempt_metrics(command_shape)
        attempt_extras = extra_program_command_attempt_metrics(row, expected_attempts)
        checks["program_command_shape_passed"] = all(row.get(key) == value for key, value in expected.items()) and not extras
        checks["extra_program_command_metrics"] = extras
        checks["program_command_attempt_shape_passed"] = all(row.get(key) == value for key, value in expected_attempts.items()) and not attempt_extras
        checks["extra_program_command_attempt_metrics"] = attempt_extras
        semantic = semantic_expectations(row, phase, prompt_tokens)
        checks["semantic_program_shape_passed"] = semantic is not None
        checks["smollm_semantic_shape_passed"] = semantic is not None and smollm_semantic_shape_passed(row)
        checks["semantic_stage_shape_passed"] = semantic is not None and row.get("semantic_stage_count") == semantic["semantic_stage_count"]
        expected_tokens = None if semantic is None else semantic["semantic_token_count"]
        checks["semantic_token_shape_passed"] = expected_tokens is not None and row.get("semantic_token_count") == expected_tokens
    if aux_metrics is not None:
        aux_extras = extra_aux_metrics(row, aux_metrics)
        checks["aux_metric_shape_passed"] = all(row.get(key) == value for key, value in aux_metrics.items()) and not aux_extras
        checks["extra_aux_metrics"] = aux_extras
        if aux_metrics:
            checks["projection_sidecar_pressure_passed"] = row.get("projection_sidecar_pressure_per_call") == native_sidecar_pressure(aux_metrics)
        else:
            checks["projection_sidecar_pressure_passed"] = row.get("projection_sidecar_pressure_per_call") == 0
    if command_shape:
        pressure = native_command_pressure(command_shape)
        checks["command_pressure_shape_passed"] = pressure is not None
        if pressure is not None:
            checks["command_pressure_top3_passed"] = row.get("command_pressure_top3_per_call") == pressure["top"]
            checks["command_pressure_total_passed"] = row.get("command_pressure_total_per_call") == pressure["total"]
            checks["command_pressure_ratio_passed"] = row.get("command_pressure_top3_ratio") == pressure["ratio"]
    if require_runtime_patch_holes or "runtime_patch_holes" in row:
        holes = row.get("runtime_patch_holes")
        semantic = semantic_expectations(row, phase, prompt_tokens)
        expected_holes = None if semantic is None else semantic["runtime_patch_holes"]
        checks["runtime_patch_holes_match"] = (
            expected_holes is not None
            and holes == expected_holes
            and row.get("semantic_runtime_patch_holes") == expected_holes
        )
        checks["runtime_patch_shape_match"] = (
            expected_holes is not None
            and row.get("runtime_patch_cache_write_pos_holes") == semantic["runtime_patch_cache_write_pos_holes"]
            and row.get("runtime_patch_attention_seq_kv_holes") == semantic["runtime_patch_attention_seq_kv_holes"]
        )
    if require_runtime_patch_stencil_hash or "runtime_patch_stencil_hash" in row:
        patch_hash = row.get("runtime_patch_stencil_hash")
        checks["runtime_patch_stencil_hash_valid"] = isinstance(patch_hash, int) and patch_hash > 0
    checks["passed"] = all(value for key, value in checks.items() if key not in ("extra_program_command_metrics", "extra_program_command_attempt_metrics", "extra_aux_metrics"))
    return checks


def native_lane_evidence(row, fmt, phase, prompt_tokens, require_runtime_patch_stencil_hash=True):
    return gate_evidence(
        row,
        phase,
        native_command_budget(fmt, phase),
        NATIVE_EXECUTION_COMMAND_SHAPES[fmt][phase],
        prompt_tokens,
        require_runtime_patch_holes=True,
        require_runtime_patch_stencil_hash=require_runtime_patch_stencil_hash,
        aux_metrics=native_aux_metrics(fmt, phase),
        command_dispatch_shape=native_command_dispatch_shape(fmt, phase),
    )


def failed_evidence_checks(evidence):
    non_checks = ("passed", "extra_program_command_metrics", "extra_program_command_attempt_metrics", "extra_aux_metrics")
    return sorted(key for key, value in evidence.items() if key not in non_checks and not value)


def require_native_evidence(lanes, label, prompt_tokens=None):
    for fmt in ("f16", "q8_0"):
        for phase in ("prompt", "decode"):
            evidence = native_lane_evidence(lanes[fmt][phase], fmt, phase, prompt_tokens)
            if not evidence["passed"]:
                failed = failed_evidence_checks(evidence)
                raise ValueError(f"{label} summary.gate_zgml.{fmt}.{phase} failed native evidence: {', '.join(failed)}")


def stored_native_evidence_is_current(native):
    rows = native.get("rows") or []
    expected_lanes = {(fmt, phase) for fmt in ("f16", "q8_0") for phase in ("prompt", "decode")}
    seen_lanes = {(row.get("format"), row.get("phase")) for row in rows}
    required_checks = (
        "passed",
        "profile_calls_match",
        "no_fallback",
        "runtime_patch_valid",
        "runtime_patch_calls_match",
        "runtime_patch_changed_match",
        "cached_command_plans",
        "no_schedule_region_failures",
        "dispatch_budget_passed",
        "program_command_shape_passed",
        "program_command_attempt_shape_passed",
        "aux_metric_shape_passed",
        "semantic_program_shape_passed",
        "smollm_semantic_shape_passed",
        "semantic_stage_shape_passed",
        "semantic_token_shape_passed",
        "runtime_patch_holes_match",
        "runtime_patch_shape_match",
        "runtime_patch_stencil_hash_valid",
        "command_pressure_shape_passed",
        "command_pressure_top3_passed",
        "command_pressure_total_passed",
        "command_pressure_ratio_passed",
        "projection_sidecar_pressure_passed",
    )
    return (
        native.get("passed") is True
        and seen_lanes == expected_lanes
        and all(
            all(row.get(check) is True for check in required_checks)
            and not row.get("extra_program_command_metrics")
            and not row.get("extra_program_command_attempt_metrics")
            and not row.get("extra_aux_metrics")
            for row in rows
        )
    )


def require_stored_or_current_native_evidence(data, lanes, label):
    native = ((data.get("gates") or {}).get("native_execution") or {})
    if stored_native_evidence_is_current(native):
        return
    require_native_evidence(lanes, label, data.get("prompt_tokens"))
