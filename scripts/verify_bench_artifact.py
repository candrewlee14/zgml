#!/usr/bin/env python3
"""Verify compact SmolLM benchmark artifacts without running Metal benches."""

import argparse
import json
import sys

from bench_contract import (
    BLOCKED_GATE_NAMES,
    COMPACT_BASELINE_FIELDS,
    COMPACT_SUMMARY_FIELDS,
    BASELINE_COUNTER_CEILING,
    BASELINE_FLOOR,
    NATIVE_EXECUTION_COMMAND_SHAPES,
    PARITY_FLOOR,
    PREFLIGHT_ARTIFACT_FIELDS,
    PREFLIGHT_FAILURE_KINDS,
    REQUIRED_METADATA,
    RUN_ARTIFACT_FIELDS,
    SMOLLM_STENCIL_COMMAND_SHAPES,
    SMOLLM_STENCIL_HASHES,
    STENCIL_ARTIFACT_FIELDS,
    STENCIL_ROW_FIELDS,
    baseline_counter_fields,
    command_attempt_metrics,
    command_shape_metrics,
    failed_evidence_checks,
    gate_evidence,
    llama_semantic_shape,
    native_aux_metrics,
    native_command_dispatch_shape,
    native_command_pressure,
    native_command_budget,
    native_lane_fields,
    native_lane_evidence,
    native_sidecar_pressure,
    preflight_failure_artifact,
    semantic_expectations,
    smollm_stencil_evidence,
    stored_native_evidence_is_current,
)

FULL_RUN_ARTIFACT_FIELDS = RUN_ARTIFACT_FIELDS + ("outputs", "lanes", "summary", "gates")
FULL_RUN_OUTPUT_FIELDS = ("zgml_f16", "zgml_q8_0", "llama_cpp_metal_f16", "llama_cpp_metal_q8_0")
FULL_RUN_GATE_FIELDS = ("parity", "reference_backend", "baseline", "native_execution", "overall_pass", "required_pass")
EXPECTED_PARITY_ROWS = (
    ("Metal F16 prompt", "llama_cpp_metal_f16", "f16", "prompt"),
    ("Metal F16 decode", "llama_cpp_metal_f16", "f16", "decode"),
    ("Metal Q8_0 prompt", "llama_cpp_metal_q8_0", "q8_0", "prompt"),
    ("Metal Q8_0 decode", "llama_cpp_metal_q8_0", "q8_0", "decode"),
)
LAPTOP_LLM_PYTORCH_FIELDS = (
    "schema",
    "createdAt",
    "command",
    "suiteId",
    "modelId",
    "prompt",
    "platform",
    "status",
    "pytorch",
    "zgml",
)


def require(errors, cond, message):
    if not cond:
        errors.append(message)


def require_exact_fields(errors, obj, expected, label):
    actual = set(obj)
    expected = set(expected)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    require(errors, not missing and not extra, f"{label} mismatch; missing: {', '.join(missing) or 'none'}; unexpected: {', '.join(extra) or 'none'}")


def is_int_metric(value):
    return isinstance(value, int) and not isinstance(value, bool)


def semantic_patch_fixture(phase="prompt", prompt_tokens=128, expected_calls=3):
    shape = llama_semantic_shape(30, 9, 3, 7, 2)
    semantic = semantic_expectations(shape, phase, prompt_tokens)
    return {
        **shape,
        "profile_calls_match": True,
        "fallback_ops": 0,
        "dynamic_region_command_plans": 0,
        "region_command_plan_cached_per_call": 1,
        "schedule_region_failed_ops": 0,
        "schedule_region_failed_ops_per_call": 0,
        "dispatches_per_call": 0,
        "commands_per_call": 0,
        "expected_profile_calls": expected_calls,
        "semantic_token_count": semantic["semantic_token_count"],
        "runtime_patch_holes": semantic["runtime_patch_holes"],
        "runtime_patch_cache_write_pos_holes": semantic["runtime_patch_cache_write_pos_holes"],
        "runtime_patch_attention_seq_kv_holes": semantic["runtime_patch_attention_seq_kv_holes"],
        "runtime_patch_calls": expected_calls,
    }


def valid_native_lane(fmt, phase, prompt_tokens=128, gen_tokens=200, repetitions=3):
    expected_calls = repetitions if phase == "prompt" else gen_tokens * repetitions
    token_count = 1 if phase == "decode" else prompt_tokens
    row = {
        **semantic_patch_fixture(phase, prompt_tokens, expected_calls),
        "tok_s": 1.0,
        "sample_count": 3,
        "sample_min_tok_s": 0.9,
        "sample_median_tok_s": 1.0,
        "sample_max_tok_s": 1.1,
        "profile_calls": expected_calls,
        "backend_ops": 1,
        "sync_waits": 1,
        "syncs_per_call": 1,
        "dynamic_region_command_plans_per_call": 0,
        "runtime_patch_invalid": 0,
        "runtime_patch_stencil_hash": SMOLLM_STENCIL_HASHES[(phase, token_count)],
    }
    if phase == "decode":
        row["runtime_patch_changed"] = expected_calls
    row.update(native_command_budget(fmt, phase))
    row.update(command_shape_metrics(NATIVE_EXECUTION_COMMAND_SHAPES[fmt][phase], native_command_dispatch_shape(fmt, phase)))
    row.update(command_attempt_metrics(NATIVE_EXECUTION_COMMAND_SHAPES[fmt][phase]))
    aux_metrics = native_aux_metrics(fmt, phase)
    pressure = native_command_pressure(NATIVE_EXECUTION_COMMAND_SHAPES[fmt][phase])
    row.update(aux_metrics)
    if pressure is not None:
        row["command_pressure_top3_per_call"] = pressure["top"]
        row["command_pressure_total_per_call"] = pressure["total"]
        row["command_pressure_top3_ratio"] = pressure["ratio"]
    row["projection_sidecar_pressure_per_call"] = native_sidecar_pressure(aux_metrics)
    return row


def metric(value):
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "?"


def compact_baseline_status_line(path, data):
    lanes = lane_rows(data, path)

    def row(fmt, phase):
        lane = lanes.get(fmt, {}).get(phase, {})
        return lane if isinstance(lane, dict) else {}

    f16_prompt = row("f16", "prompt")
    f16_decode = row("f16", "decode")
    q8_prompt = row("q8_0", "prompt")
    q8_decode = row("q8_0", "decode")
    fallback_ops = sum((lane.get("fallback_ops", 0) for lane in (f16_prompt, f16_decode, q8_prompt, q8_decode)), 0)
    patch_holes = f16_prompt.get("runtime_patch_holes", "?")
    return (
        f"{path}: pass structural-baseline (not parity evidence): "
        f"F16 pp={metric(f16_prompt.get('tok_s'))} tok/s tg={metric(f16_decode.get('tok_s'))} tok/s; "
        f"Q8_0 pp={metric(q8_prompt.get('tok_s'))} tok/s tg={metric(q8_decode.get('tok_s'))} tok/s; "
        f"dispatch pp={f16_prompt.get('dispatches_per_call', '?')}/{q8_prompt.get('dispatches_per_call', '?')} "
        f"tg={f16_decode.get('dispatches_per_call', '?')}/{q8_decode.get('dispatches_per_call', '?')}; "
        f"fallback_ops={fallback_ops}; patch_holes={patch_holes}"
    )


def stencil_status_line(path, data):
    rows = {
        row.get("phase"): row
        for row in data.get("rows", [])
        if isinstance(row, dict)
    }
    prompt = rows.get("prompt", {})
    decode = rows.get("decode", {})
    prompt_split = projection_chain_split(prompt)
    decode_split = projection_chain_split(decode)
    return (
        f"{path}: pass offline-stencil (shape/hash only, no kernel execution): "
        f"prompt_hash={prompt.get('runtime_patch_stencil_hash', '?')} "
        f"decode_hash={decode.get('runtime_patch_stencil_hash', '?')}; "
        f"patch_holes={prompt.get('runtime_patch_holes', decode.get('runtime_patch_holes', '?'))}; "
        f"shape pp={prompt.get('program_command_shape_commands', '?')}/{prompt.get('program_command_shape_covered_ops', '?')} "
        f"saved={prompt.get('program_command_shape_estimated_saved_dispatches', '?')} "
        f"tg={decode.get('program_command_shape_commands', '?')}/{decode.get('program_command_shape_covered_ops', '?')} "
        f"saved={decode.get('program_command_shape_estimated_saved_dispatches', '?')}; "
        f"frontier projection_chain={prompt.get('program_command_shape_projection_chains', '?')}/{decode.get('program_command_shape_projection_chains', '?')} "
        f"chain_split={prompt_split}/{decode_split} "
        f"row_frontier={prompt.get('program_command_shape_projection_chain_row_chain_frontiers', '?')}/{decode.get('program_command_shape_projection_chain_row_chain_frontiers', '?')} "
        f"projection_row_chain={prompt.get('program_command_shape_projection_row_chains', '?')}/{decode.get('program_command_shape_projection_row_chains', '?')} "
        f"dense_projection_row_chain={prompt.get('program_command_shape_dense_projection_row_chains', '?')}/{decode.get('program_command_shape_dense_projection_row_chains', '?')}"
    )


def projection_chain_split(row):
    split = projection_chain_split_metrics(row)
    total = row.get("program_command_shape_projection_chains", "?")
    if split is not None:
        return f"dense:{split['dense']}+quantized:{split['quantized']}"
    return f"total:{total}"


def projection_chain_split_metrics(row):
    if not isinstance(row, dict):
        return None
    dense = row.get("program_command_shape_dense_projection_chains")
    quantized = row.get("program_command_shape_quantized_projection_chains")
    if is_int_metric(dense) or is_int_metric(quantized):
        return {
            "dense": dense if is_int_metric(dense) else 0,
            "quantized": quantized if is_int_metric(quantized) else 0,
            "source": "explicit",
        }
    dense = row.get("program_command_encoded_dense_projection_chain_per_call")
    quantized = row.get("program_command_encoded_projection_chain_per_call")
    if is_int_metric(dense) or is_int_metric(quantized):
        return {
            "dense": dense if is_int_metric(dense) else 0,
            "quantized": quantized if is_int_metric(quantized) else 0,
            "source": "command-shape",
        }
    return None


def preflight_status_line(path, data):
    preflight = (data.get("gates") or {}).get("preflight") or {}
    return (
        f"{path}: pass preflight-blocker (not parity evidence): "
        f"failure_kind={preflight.get('failure_kind', '?')}; "
        f"required_pass={data.get('gates', {}).get('required_pass', '?')}"
    )


def parity_pct(data, label):
    rows = (((data.get("gates") or {}).get("parity") or {}).get("rows") or [])
    for row in rows:
        if row.get("label") == label and isinstance(row.get("parity"), (int, float)):
            return row["parity"] * 100
    return None


def full_run_status_line(path, data):
    lanes = lane_rows(data, path)

    def row(fmt, phase):
        lane = lanes.get(fmt, {}).get(phase, {})
        return lane if isinstance(lane, dict) else {}

    f16_prompt = row("f16", "prompt")
    f16_decode = row("f16", "decode")
    q8_prompt = row("q8_0", "prompt")
    q8_decode = row("q8_0", "decode")
    fallback_ops = sum((lane.get("fallback_ops", 0) for lane in (f16_prompt, f16_decode, q8_prompt, q8_decode)), 0)
    prompt_hashes = f"{f16_prompt.get('runtime_patch_stencil_hash', '?')}/{q8_prompt.get('runtime_patch_stencil_hash', '?')}"
    decode_hashes = f"{f16_decode.get('runtime_patch_stencil_hash', '?')}/{q8_decode.get('runtime_patch_stencil_hash', '?')}"
    prompt_chain_split = f"{projection_chain_split(f16_prompt)}/{projection_chain_split(q8_prompt)}"
    decode_chain_split = f"{projection_chain_split(f16_decode)}/{projection_chain_split(q8_decode)}"
    gates = data.get("gates") or {}
    parity = gates.get("parity") or {}
    parity_state = "parity pass" if parity.get("passed") is True else "parity miss"
    if parity.get("required") is True:
        parity_state += ", parity required"
    else:
        parity_state += ", parity not required"
    return (
        f"{path}: pass full-run evidence ({parity_state}): "
        f"required_pass={gates.get('required_pass', '?')}; overall_pass={gates.get('overall_pass', '?')}; "
        f"F16 pp={metric(f16_prompt.get('tok_s'))} tok/s ({metric(parity_pct(data, 'Metal F16 prompt'))}%) "
        f"tg={metric(f16_decode.get('tok_s'))} tok/s ({metric(parity_pct(data, 'Metal F16 decode'))}%); "
        f"Q8_0 pp={metric(q8_prompt.get('tok_s'))} tok/s ({metric(parity_pct(data, 'Metal Q8_0 prompt'))}%) "
        f"tg={metric(q8_decode.get('tok_s'))} tok/s ({metric(parity_pct(data, 'Metal Q8_0 decode'))}%); "
        f"dispatch pp={f16_prompt.get('dispatches_per_call', '?')}/{q8_prompt.get('dispatches_per_call', '?')} "
        f"tg={f16_decode.get('dispatches_per_call', '?')}/{q8_decode.get('dispatches_per_call', '?')}; "
        f"chain_split pp={prompt_chain_split} tg={decode_chain_split}; "
        f"stencil_hash pp={prompt_hashes} tg={decode_hashes}; "
        f"fallback_ops={fallback_ops}"
    )


def is_full_run_artifact(data):
    return (
        data.get("benchmark") == "smollm-135m"
        and isinstance(data.get("gates"), dict)
        and isinstance(data.get("lanes"), dict)
        and isinstance(data.get("outputs"), dict)
        and isinstance(data.get("summary"), dict)
        and not isinstance((data.get("gates") or {}).get("preflight"), dict)
    )


def status_line_for_data(path, data):
    if data.get("benchmark") == "smollm-135m-stencil":
        return stencil_status_line(path, data)
    if isinstance((data.get("gates") or {}).get("preflight"), dict):
        return preflight_status_line(path, data)
    if is_full_run_artifact(data):
        return full_run_status_line(path, data)
    return compact_baseline_status_line(path, data)


def status_line(path):
    with open(path, "r", encoding="utf-8") as f:
        return status_line_for_data(path, json.load(f))


def stored_native_rows(**overrides):
    true_checks = (
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
    rows = []
    for fmt in ("f16", "q8_0"):
        for phase in ("prompt", "decode"):
            row = {"format": fmt, "phase": phase}
            row.update({check: True for check in true_checks})
            row.update({
                "extra_aux_metrics": [],
                "extra_program_command_metrics": [],
                "extra_program_command_attempt_metrics": [],
            })
            row.update(overrides)
            rows.append(row)
    return rows


def verify_contract_helpers():
    errors = []
    preflight = preflight_failure_artifact(
        {
            "DATE_UTC": "2026-06-02T00:00:00Z",
            "MACHINE": "test-machine",
            "PROMPT": "128",
            "GEN": "200",
            "REPS": "3",
            "BENCH_REQUIRE_PARITY": "1",
            "BENCH_BASELINE_JSON": "baseline.json",
        },
        "F16",
        "model.gguf",
        "raw failure",
        1,
        "preflight failed",
    )
    require(errors, preflight["gates"]["preflight"]["passed"] is False, "preflight artifact must record failed preflight")
    require(errors, preflight["gates"]["required_pass"] is False, "preflight artifact must fail required gate")
    require(errors, preflight["gates"]["parity"]["required"] is True, "preflight artifact must preserve parity-required flag")
    require(errors, preflight["gates"]["preflight"]["failure_kind"] == "llama_bench_failed", "preflight artifact must classify generic preflight failure")
    require(errors, preflight["gates"]["preflight"]["raw_output"] == "raw failure", "preflight artifact must preserve raw output")

    no_device_preflight = preflight_failure_artifact(
        {
            "DATE_UTC": "2026-06-02T00:00:00Z",
            "MACHINE": "test-machine",
            "PROMPT": "128",
            "GEN": "200",
            "REPS": "3",
        },
        "F16",
        "model.gguf",
        "Available devices:\n  BLAS: Accelerate",
        0,
        "llama-bench F16 preflight did not list a usable Metal/MTL device",
    )
    require(errors, no_device_preflight["gates"]["preflight"]["failure_kind"] == "metal_device_unavailable", "preflight artifact must classify missing Metal device")

    metal_context_preflight = preflight_failure_artifact(
        {
            "DATE_UTC": "2026-06-02T00:00:00Z",
            "MACHINE": "test-machine",
            "PROMPT": "128",
            "GEN": "200",
            "REPS": "3",
        },
        "F16",
        "model.gguf",
        "ggml_metal_init: error: failed to create command queue",
        1,
        "llama-bench F16 Metal/MTL context preflight failed",
    )
    require(errors, metal_context_preflight["gates"]["preflight"]["failure_kind"] == "metal_context_init_failed", "preflight artifact must classify Metal context init failure")
    smoke_preflight = preflight_failure_artifact(
        {
            "DATE_UTC": "2026-06-02T00:00:00Z",
            "MACHINE": "test-machine",
            "PROMPT": "128",
            "GEN": "1",
            "REPS": "1",
        },
        "F16",
        "model.gguf",
        "ggml_metal_init: error: failed to create command queue",
        1,
        "llama-bench F16 Metal/MTL context preflight failed",
    )
    smoke_preflight["metadata"].update({
        "zgml_f16_sha256": "test",
        "zgml_q8_sha256": "test",
        "llama_cpp_f16_sha256": "test",
        "llama_cpp_q8_sha256": "test",
    })
    require(errors, not verify_preflight_artifact(smoke_preflight, "self-test smoke preflight"), "preflight verifier must accept smoke-sized preflight artifacts")

    shape = llama_semantic_shape(30, 9, 3, 7, 2)
    row = semantic_patch_fixture()
    semantic = semantic_expectations(row, "prompt", 128)
    require(errors, gate_evidence(row, "prompt", {"dispatches_per_call": 0, "commands_per_call": 0}, {}, 128)["passed"], "contract helper must accept exact prompt token shape")
    require(errors, gate_evidence(row, "prompt", {"dispatches_per_call": 0, "commands_per_call": 0}, {}, 128, require_runtime_patch_holes=True)["runtime_patch_holes_match"], "contract helper must accept exact patch-hole evidence")
    missing_patch_row = dict(row, runtime_patch_holes=0)
    require(errors, not gate_evidence(missing_patch_row, "prompt", {"dispatches_per_call": 0, "commands_per_call": 0}, {}, 128, require_runtime_patch_holes=True)["runtime_patch_holes_match"], "contract helper must reject missing patch holes when required")
    wrong_patch_kind_row = dict(row, runtime_patch_cache_write_pos_holes=semantic["runtime_patch_attention_seq_kv_holes"], runtime_patch_attention_seq_kv_holes=semantic["runtime_patch_cache_write_pos_holes"])
    require(errors, not gate_evidence(wrong_patch_kind_row, "prompt", {"dispatches_per_call": 0, "commands_per_call": 0}, {}, 128, require_runtime_patch_holes=True)["runtime_patch_shape_match"], "contract helper must reject wrong runtime patch split")
    hash_row = dict(row, runtime_patch_stencil_hash=123)
    require(errors, gate_evidence(hash_row, "prompt", {"dispatches_per_call": 0, "commands_per_call": 0}, {}, 128, require_runtime_patch_holes=True, require_runtime_patch_stencil_hash=True)["runtime_patch_stencil_hash_valid"], "contract helper must accept positive runtime patch stencil hash")
    bad_hash_row = dict(row, runtime_patch_stencil_hash=0)
    require(errors, not gate_evidence(bad_hash_row, "prompt", {"dispatches_per_call": 0, "commands_per_call": 0}, {}, 128, require_runtime_patch_holes=True, require_runtime_patch_stencil_hash=True)["runtime_patch_stencil_hash_valid"], "contract helper must reject zero runtime patch stencil hash when present")
    require(errors, not gate_evidence(row, "prompt", {"dispatches_per_call": 0, "commands_per_call": 0}, {}, 128, require_runtime_patch_holes=True, require_runtime_patch_stencil_hash=True)["runtime_patch_stencil_hash_valid"], "contract helper must reject missing runtime patch stencil hash when required")
    require(errors, not gate_evidence(row, "prompt", {"dispatches_per_call": 0, "commands_per_call": 0}, {})["semantic_token_shape_passed"], "contract helper must reject prompt evidence without prompt width")
    require(errors, not gate_evidence(row, "prompt", {"dispatches_per_call": 0, "commands_per_call": 0}, {}, 64)["semantic_token_shape_passed"], "contract helper must reject wrong prompt token shape")
    wrong_semantic_shape = dict(row, semantic_heads=8)
    require(errors, not gate_evidence(wrong_semantic_shape, "prompt", {"dispatches_per_call": 0, "commands_per_call": 0}, {}, 128, require_runtime_patch_holes=True)["semantic_program_shape_passed"], "contract helper must reject inconsistent semantic shape")
    wrong_but_consistent_shape = llama_semantic_shape(24, 8, 4, 7, 2)
    wrong_but_consistent_row = dict(row, **wrong_but_consistent_shape)
    wrong_but_consistent_semantic = semantic_expectations(wrong_but_consistent_row, "prompt", 128)
    wrong_but_consistent_row.update({
        "runtime_patch_holes": wrong_but_consistent_semantic["runtime_patch_holes"],
        "runtime_patch_cache_write_pos_holes": wrong_but_consistent_semantic["runtime_patch_cache_write_pos_holes"],
        "runtime_patch_attention_seq_kv_holes": wrong_but_consistent_semantic["runtime_patch_attention_seq_kv_holes"],
    })
    wrong_but_consistent_evidence = gate_evidence(wrong_but_consistent_row, "prompt", {"dispatches_per_call": 0, "commands_per_call": 0}, {}, 128, require_runtime_patch_holes=True)
    require(errors, wrong_but_consistent_evidence["semantic_program_shape_passed"], "contract helper must still recognize self-consistent non-SmolLM shape")
    require(errors, not wrong_but_consistent_evidence["smollm_semantic_shape_passed"], "contract helper must reject non-SmolLM semantic shape")
    require(errors, not wrong_but_consistent_evidence["passed"], "native evidence must fail for non-SmolLM semantic shape")
    require(errors, not stored_native_evidence_is_current({"passed": True, "rows": [{"passed": True}]}), "stored native evidence without semantic checks must be stale")
    require(
        errors,
        not stored_native_evidence_is_current({"passed": True, "rows": [{"format": "f16", "phase": "prompt", "passed": True, "semantic_stage_shape_passed": True, "semantic_token_shape_passed": True}]}),
        "stored native evidence must cover every native lane",
    )
    shallow_rows = [
        {"format": fmt, "phase": phase, "passed": True, "semantic_stage_shape_passed": True, "semantic_token_shape_passed": True, "runtime_patch_holes_match": True, "runtime_patch_shape_match": True}
        for fmt in ("f16", "q8_0")
        for phase in ("prompt", "decode")
    ]
    require(errors, not stored_native_evidence_is_current({"passed": True, "rows": shallow_rows}), "stored native evidence without strict shape checks must be stale")
    current_rows = stored_native_rows()
    require(errors, stored_native_evidence_is_current({"passed": True, "rows": current_rows}), "stored native evidence with all strict native checks must be current")
    missing_patch_rows = [dict(row, runtime_patch_holes_match=False) for row in current_rows]
    require(errors, not stored_native_evidence_is_current({"passed": True, "rows": missing_patch_rows}), "stored native evidence without patch-hole checks must be stale")
    missing_patch_kind_rows = [dict(row, runtime_patch_shape_match=False) for row in current_rows]
    require(errors, not stored_native_evidence_is_current({"passed": True, "rows": missing_patch_kind_rows}), "stored native evidence without runtime patch split checks must be stale")
    missing_hash_rows = [dict(row, runtime_patch_stencil_hash_valid=False) for row in current_rows]
    require(errors, not stored_native_evidence_is_current({"passed": True, "rows": missing_hash_rows}), "stored native evidence without runtime patch stencil hash checks must be stale")
    missing_pressure_rows = [dict(row, command_pressure_top3_passed=False) for row in current_rows]
    require(errors, not stored_native_evidence_is_current({"passed": True, "rows": missing_pressure_rows}), "stored native evidence without command-pressure checks must be stale")
    missing_sidecar_pressure_rows = [dict(row, projection_sidecar_pressure_passed=False) for row in current_rows]
    require(errors, not stored_native_evidence_is_current({"passed": True, "rows": missing_sidecar_pressure_rows}), "stored native evidence without sidecar-pressure checks must be stale")
    valid_stencil = {
        **shape,
        "semantic_token_count": 128,
        "profile_calls": 0,
        "runtime_patch_holes": semantic["runtime_patch_holes"],
        "runtime_patch_cache_write_pos_holes": semantic["runtime_patch_cache_write_pos_holes"],
        "runtime_patch_attention_seq_kv_holes": semantic["runtime_patch_attention_seq_kv_holes"],
        "runtime_patch_stencil_hash": 17558208047327870709,
        **SMOLLM_STENCIL_COMMAND_SHAPES[("prompt", 128)],
    }
    require(errors, smollm_stencil_evidence(valid_stencil, "prompt", 128)["passed"], "stencil evidence must accept checked SmolLM prompt hash")
    wrong_stencil = dict(valid_stencil, runtime_patch_stencil_hash=123)
    require(errors, not smollm_stencil_evidence(wrong_stencil, "prompt", 128)["runtime_patch_stencil_hash_exact"], "stencil evidence must reject wrong SmolLM hash")
    extra_command_rows = [dict(row, extra_program_command_metrics=["program_command_dispatches_unexpected_per_call"]) for row in current_rows]
    require(errors, not stored_native_evidence_is_current({"passed": True, "rows": extra_command_rows}), "stored native evidence with extra command metrics must be stale")
    extra_attempt_rows = [dict(row, extra_program_command_attempt_metrics=["program_command_attempts_unexpected_per_call"]) for row in current_rows]
    require(errors, not stored_native_evidence_is_current({"passed": True, "rows": extra_attempt_rows}), "stored native evidence with extra command-attempt metrics must be stale")
    extra_aux_rows = [dict(row, extra_aux_metrics=["program_op_command_unexpected_per_call"]) for row in current_rows]
    require(errors, not stored_native_evidence_is_current({"passed": True, "rows": extra_aux_rows}), "stored native evidence with extra aux metrics must be stale")
    compact_counter_row = {"backend_ops": 3, "fallback_ops": 0, "dispatches_per_call": 2, "program_command_encoded_op_per_call": 1}
    compact_counter_fields = baseline_counter_fields(compact_counter_row)
    require(errors, "placed_ops" in compact_counter_fields, "baseline counters must include synthetic placed_ops")
    require(errors, "sync_waits" not in compact_counter_fields, "baseline counters must not include absent optional counters")
    require(errors, "program_command_encoded_op_per_call" in compact_counter_fields, "baseline counters must include present ProgramCommand counters")
    shape_counter_row = {
        **compact_counter_row,
        "runtime_patch_holes": semantic["runtime_patch_holes"],
        "runtime_patch_cache_write_pos_holes": semantic["runtime_patch_cache_write_pos_holes"],
        "runtime_patch_attention_seq_kv_holes": semantic["runtime_patch_attention_seq_kv_holes"],
        "semantic_runtime_patch_holes": semantic["runtime_patch_holes"],
        "semantic_runtime_patch_cache_write_pos_holes": semantic["runtime_patch_cache_write_pos_holes"],
        "semantic_runtime_patch_attention_seq_kv_holes": semantic["runtime_patch_attention_seq_kv_holes"],
        "semantic_stage_count": semantic["semantic_stage_count"],
        "semantic_token_count": semantic["semantic_token_count"],
    }
    shape_counter_fields = baseline_counter_fields(shape_counter_row)
    for field in ("runtime_patch_holes", "runtime_patch_cache_write_pos_holes", "runtime_patch_attention_seq_kv_holes", "semantic_runtime_patch_holes", "semantic_runtime_patch_cache_write_pos_holes", "semantic_runtime_patch_attention_seq_kv_holes", "semantic_stage_count", "semantic_token_count"):
        require(errors, field not in shape_counter_fields, f"{field} is shape evidence, not a baseline lower-is-better counter")

    bench_data = {"prompt_tokens": 128, "gen_tokens": 200, "repetitions": 3}
    valid_prompt_lane = valid_native_lane("f16", "prompt")
    require(errors, not verify_lane(bench_data, "self-test", "f16", "prompt", valid_prompt_lane), "compact verifier must accept exact patch-hole lane evidence")
    split_prompt_lane = dict(
        valid_prompt_lane,
        program_command_shape_projection_chains=60,
        program_command_shape_dense_projection_chains=60,
        program_command_shape_quantized_projection_chains=0,
    )
    require(errors, not verify_lane(bench_data, "self-test", "f16", "prompt", split_prompt_lane), "compact verifier must accept projection-chain split that sums to total")
    bad_split_prompt_lane = dict(
        valid_prompt_lane,
        program_command_shape_projection_chains=60,
        program_command_shape_dense_projection_chains=59,
        program_command_shape_quantized_projection_chains=0,
    )
    require(errors, any("projection-chain split" in err for err in verify_lane(bench_data, "self-test", "f16", "prompt", bad_split_prompt_lane)), "compact verifier must reject projection-chain split that does not sum to total")
    bool_split_prompt_lane = dict(
        valid_prompt_lane,
        program_command_shape_projection_chains=60,
        program_command_shape_dense_projection_chains=True,
        program_command_shape_quantized_projection_chains=0,
    )
    require(errors, any("dense_projection_chains must be an integer" in err for err in verify_lane(bench_data, "self-test", "f16", "prompt", bool_split_prompt_lane)), "compact verifier must reject boolean projection-chain split metrics")
    inferred_split_prompt_lane = dict(
        valid_prompt_lane,
        program_command_shape_projection_chains=60,
    )
    require(errors, not verify_lane(bench_data, "self-test", "f16", "prompt", inferred_split_prompt_lane), "compact verifier must accept command-shape projection-chain split that sums to total")
    bad_inferred_split_prompt_lane = dict(
        inferred_split_prompt_lane,
        program_command_encoded_dense_projection_chain_per_call=59,
    )
    require(errors, any("projection-chain split" in err for err in verify_lane(bench_data, "self-test", "f16", "prompt", bad_inferred_split_prompt_lane)), "compact verifier must reject command-shape projection-chain split that does not sum to total")
    stale_prompt_lane = dict(valid_prompt_lane)
    stale_prompt_lane.pop("runtime_patch_holes")
    require(errors, any("runtime_patch_holes" in err for err in verify_lane(bench_data, "self-test", "f16", "prompt", stale_prompt_lane)), "compact verifier must reject lanes missing patch-hole evidence")
    stale_hash_lane = dict(valid_prompt_lane)
    stale_hash_lane.pop("runtime_patch_stencil_hash")
    require(errors, any("runtime_patch_stencil_hash_valid" in err for err in verify_lane(bench_data, "self-test", "f16", "prompt", stale_hash_lane)), "compact verifier must reject lanes missing runtime patch stencil hash")
    zero_hash_lane = dict(valid_prompt_lane, runtime_patch_stencil_hash=0)
    require(errors, any("runtime_patch_stencil_hash_valid" in err for err in verify_lane(bench_data, "self-test", "f16", "prompt", zero_hash_lane)), "compact verifier must reject zero runtime patch stencil hash")
    invalid_patch_lane = dict(valid_prompt_lane, runtime_patch_invalid=1)
    require(errors, any("runtime_patch_invalid" in err for err in verify_lane(bench_data, "self-test", "f16", "prompt", invalid_patch_lane)), "compact verifier must reject invalid runtime patch evidence")
    stale_patch_call_lane = dict(valid_prompt_lane)
    stale_patch_call_lane.pop("runtime_patch_calls")
    require(errors, any("runtime_patch_calls" in err for err in verify_lane(bench_data, "self-test", "f16", "prompt", stale_patch_call_lane)), "compact verifier must reject lanes missing runtime patch call evidence")
    valid_decode_lane = valid_native_lane("f16", "decode")
    require(errors, not verify_lane(bench_data, "self-test", "f16", "decode", valid_decode_lane), "compact verifier must accept explicit zero invalid-patch decode evidence")
    compact_status = status_line_for_data("self-test-baseline", {
        "summary": {
            "gate_zgml": {
                "f16": {
                    "prompt": valid_native_lane("f16", "prompt"),
                    "decode": valid_native_lane("f16", "decode"),
                },
                "q8_0": {
                    "prompt": valid_native_lane("q8_0", "prompt"),
                    "decode": valid_native_lane("q8_0", "decode"),
                },
            },
        },
    })
    require(errors, "structural-baseline" in compact_status and "not parity evidence" in compact_status, "status line must label compact baselines as structural, not parity")

    full_lanes = {
        fmt: {
            phase: valid_native_lane(fmt, phase)
            for phase in ("prompt", "decode")
        }
        for fmt in ("f16", "q8_0")
    }

    def native_row(fmt, phase):
        row = full_lanes[fmt][phase]
        evidence = native_lane_evidence(row, fmt, phase, 128)
        out = {
            "format": fmt,
            "phase": phase,
            "zgml": row,
            "dispatch_budget": native_command_budget(fmt, phase),
            "program_command_shape": NATIVE_EXECUTION_COMMAND_SHAPES[fmt][phase],
            "program_command_shape_metrics": command_shape_metrics(NATIVE_EXECUTION_COMMAND_SHAPES[fmt][phase], native_command_dispatch_shape(fmt, phase)),
        }
        out.update(evidence)
        return out

    def parity_row(label, target, fmt, phase):
        row = full_lanes[fmt][phase]
        evidence = native_lane_evidence(row, fmt, phase, 128)
        llama_tok_s = row["tok_s"] * 2
        return {
            "label": label,
            "target": target,
            "phase": phase,
            "zgml": row,
            "llama_tok_s": llama_tok_s,
            "llama_backend": "BLAS,MTL",
            "llama_dev": "Metal",
            "llama_backend_is_metal": True,
            "zgml_no_fallback": evidence["no_fallback"],
            "zgml_profile_window_ok": evidence["profile_calls_match"],
            "zgml_native_evidence_ok": evidence["passed"],
            "parity": row["tok_s"] / llama_tok_s,
            "floor": PARITY_FLOOR,
            "passed": False,
        }

    full_run = {
        "benchmark": "smollm-135m",
        "date_utc": "2026-06-02T00:00:00Z",
        "machine": "test-machine",
        "prompt_tokens": 128,
        "gen_tokens": 200,
        "repetitions": 3,
        "metadata": {
            key: "test"
            for key in REQUIRED_METADATA
        },
        "outputs": {
            "zgml_f16": "zgml-f16.txt",
            "zgml_q8_0": "zgml-q8.txt",
            "llama_cpp_metal_f16": "llama-f16.txt",
            "llama_cpp_metal_q8_0": "llama-q8.txt",
        },
        "lanes": {},
        "summary": {"gate_zgml": full_lanes},
        "gates": {
            "parity": {
                "required": False,
                "floor": PARITY_FLOOR,
                "passed": False,
                "rows": [
                    parity_row(label, target, fmt, phase)
                    for label, target, fmt, phase in EXPECTED_PARITY_ROWS
                ],
            },
            "reference_backend": {
                "passed": True,
                "rows": [
                    {"label": label, "backend": "BLAS,MTL", "dev": "Metal", "passed": True}
                    for label, _, _, _ in EXPECTED_PARITY_ROWS
                ],
            },
            "baseline": {
                "baseline_json": None,
                "floor_ratio": BASELINE_FLOOR,
                "counter_ceiling_ratio": BASELINE_COUNTER_CEILING,
                "passed": True,
                "rows": [],
            },
            "native_execution": {
                "passed": True,
                "rows": [
                    native_row(fmt, phase)
                    for fmt in ("f16", "q8_0")
                    for phase in ("prompt", "decode")
                ],
            },
            "overall_pass": False,
            "required_pass": True,
        },
    }
    require(errors, not verify_full_run_artifact(full_run, "self-test-full-run"), "full-run verifier must accept structurally valid parity-miss evidence")
    full_status = status_line_for_data("self-test-full-run", full_run)
    require(errors, "full-run evidence" in full_status and "parity miss" in full_status and "required_pass=True" in full_status, "status line must label full run artifacts without claiming parity")

    valid_decode_stencil = {
        **valid_stencil,
        "phase": "decode",
        "semantic_token_count": 1,
        "runtime_patch_stencil_hash": SMOLLM_STENCIL_HASHES[("decode", 1)],
        **SMOLLM_STENCIL_COMMAND_SHAPES[("decode", 1)],
    }
    stencil_status = status_line_for_data("self-test-stencil", {
        "benchmark": "smollm-135m-stencil",
        "rows": [
            dict(valid_stencil, phase="prompt"),
            valid_decode_stencil,
        ],
    })
    require(errors, "offline-stencil" in stencil_status and "no kernel execution" in stencil_status, "status line must label stencil artifacts as offline shape/hash evidence")
    require(errors, "shape pp=211/1654 saved=1443 tg=181/1654 saved=1473" in stencil_status, "status line must expose offline command-shape compression evidence")
    require(errors, "chain_split=dense:29+quantized:0/dense:29+quantized:0" in stencil_status, "status line must expose projection-chain dense/quantized split evidence")
    require(errors, "row_frontier=29/29" in stencil_status, "status line must expose projection-chain row frontier evidence")
    require(errors, "dense_projection_row_chain=31/31" in stencil_status, "status line must expose direct projection-row-chain evidence")
    return errors


def lane_rows(data, path):
    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{path}: missing object summary")
    lanes = summary.get("gate_zgml")
    if not isinstance(lanes, dict):
        raise ValueError(f"{path}: missing object summary.gate_zgml")
    return lanes


def verify_lane(data, path, fmt, phase, row, strict_fields=True, evidence_override=None):
    errors = []
    prefix = f"{path}: summary.gate_zgml.{fmt}.{phase}"
    allowed_fields = native_lane_fields(fmt, phase)

    require(errors, isinstance(row, dict), f"{prefix} must be an object")
    if errors:
        return errors
    if strict_fields:
        extra_fields = sorted(set(row) - allowed_fields)
        require(errors, not extra_fields, f"{prefix} has unexpected fields: {', '.join(extra_fields)}")

    require(errors, isinstance(row.get("tok_s"), (int, float)) and row["tok_s"] > 0, f"{prefix}.tok_s must be positive")
    require(errors, row.get("sample_count", 0) >= 3, f"{prefix}.sample_count must be >= 3")
    sample_min = row.get("sample_min_tok_s")
    sample_median = row.get("sample_median_tok_s")
    sample_max = row.get("sample_max_tok_s")
    require(
        errors,
        all(isinstance(v, (int, float)) and v > 0 for v in (sample_min, sample_median, sample_max)),
        f"{prefix} must include positive sample_min/median/max tok/s",
    )
    if all(isinstance(v, (int, float)) for v in (sample_min, sample_median, sample_max)):
        require(errors, sample_min <= sample_median <= sample_max, f"{prefix} sample tok/s values must be ordered")
        require(
            errors,
            abs(row["tok_s"] - sample_median) <= max(1e-9, abs(row["tok_s"]) * 1e-9),
            f"{prefix}.tok_s must match sample_median_tok_s",
        )
    require(errors, row.get("runtime_patch_invalid", 0) == 0, f"{prefix}.runtime_patch_invalid must be 0")
    for key in ("backend_ops", "sync_waits", "syncs_per_call"):
        require(errors, isinstance(row.get(key), (int, float)) and row[key] >= 0, f"{prefix}.{key} must be non-negative")

    repetitions = data.get("repetitions")
    gen_tokens = data.get("gen_tokens")
    expected_calls = repetitions if phase == "prompt" else gen_tokens * repetitions
    require(errors, row.get("expected_profile_calls") == expected_calls, f"{prefix}.expected_profile_calls must be {expected_calls}")
    require(errors, row.get("profile_calls") == expected_calls, f"{prefix}.profile_calls must be {expected_calls}")
    require(errors, row.get("runtime_patch_calls") == expected_calls, f"{prefix}.runtime_patch_calls must be {expected_calls}")
    if phase == "decode":
        require(errors, row.get("runtime_patch_changed") == expected_calls, f"{prefix}.runtime_patch_changed must be {expected_calls}")

    evidence = evidence_override if isinstance(evidence_override, dict) else native_lane_evidence(row, fmt, phase, data.get("prompt_tokens"))
    for check in failed_evidence_checks(evidence):
        errors.append(f"{prefix} failed {check}")
    has_dense_split = "program_command_shape_dense_projection_chains" in row
    has_quantized_split = "program_command_shape_quantized_projection_chains" in row
    if has_dense_split or has_quantized_split:
        dense = row.get("program_command_shape_dense_projection_chains")
        quantized = row.get("program_command_shape_quantized_projection_chains")
        total = row.get("program_command_shape_projection_chains")
        require(errors, is_int_metric(dense), f"{prefix}.program_command_shape_dense_projection_chains must be an integer when projection-chain split is present")
        require(errors, is_int_metric(quantized), f"{prefix}.program_command_shape_quantized_projection_chains must be an integer when projection-chain split is present")
        require(errors, is_int_metric(dense) and is_int_metric(quantized) and dense + quantized == total, f"{prefix} projection-chain split must sum to program_command_shape_projection_chains")
    elif "program_command_shape_projection_chains" in row:
        split = projection_chain_split_metrics(row)
        if split is not None:
            total = row.get("program_command_shape_projection_chains")
            require(errors, split["dense"] + split["quantized"] == total, f"{prefix} projection-chain split must sum to program_command_shape_projection_chains")
    token_count = 1 if phase == "decode" else data.get("prompt_tokens")
    expected_hash = SMOLLM_STENCIL_HASHES.get((phase, token_count))
    if expected_hash is not None:
        require(
            errors,
            row.get("runtime_patch_stencil_hash") == expected_hash,
            f"{prefix}.runtime_patch_stencil_hash must be {expected_hash}",
        )
    return errors


def verify_preflight_artifact(data, path):
    errors = []
    require_exact_fields(errors, data, PREFLIGHT_ARTIFACT_FIELDS, f"{path}: preflight artifact top-level")
    require(errors, data.get("benchmark") == "smollm-135m", f"{path}: benchmark must be smollm-135m")
    require(errors, isinstance(data.get("date_utc"), str) and "T" in data["date_utc"] and data["date_utc"].endswith("Z"), f"{path}: date_utc must be a UTC timestamp")
    require(errors, isinstance(data.get("machine"), str) and data["machine"], f"{path}: machine must be non-empty")
    require(errors, isinstance(data.get("prompt_tokens"), int) and data["prompt_tokens"] > 0, f"{path}: prompt_tokens must be a positive integer")
    require(errors, isinstance(data.get("gen_tokens"), int) and data["gen_tokens"] > 0, f"{path}: gen_tokens must be a positive integer")
    require(errors, isinstance(data.get("repetitions"), int) and data["repetitions"] > 0, f"{path}: repetitions must be a positive integer")
    metadata = data.get("metadata")
    require(errors, isinstance(metadata, dict), f"{path}: metadata must be an object")
    if isinstance(metadata, dict):
        for key in REQUIRED_METADATA:
            require(errors, key in metadata, f"{path}: metadata.{key} is required")

    gates = data.get("gates")
    require(errors, isinstance(gates, dict), f"{path}: gates must be an object")
    if not isinstance(gates, dict):
        return errors
    preflight = gates.get("preflight")
    require(errors, isinstance(preflight, dict), f"{path}: gates.preflight must be an object")
    if isinstance(preflight, dict):
        require(errors, preflight.get("passed") is False, f"{path}: gates.preflight.passed must be false")
        require(errors, isinstance(preflight.get("label"), str) and preflight["label"], f"{path}: gates.preflight.label must be non-empty")
        require(errors, isinstance(preflight.get("model"), str) and preflight["model"], f"{path}: gates.preflight.model must be non-empty")
        require(errors, isinstance(preflight.get("status"), int), f"{path}: gates.preflight.status must be an integer")
        require(errors, isinstance(preflight.get("reason"), str) and preflight["reason"], f"{path}: gates.preflight.reason must be non-empty")
        require(errors, preflight.get("failure_kind") in PREFLIGHT_FAILURE_KINDS, f"{path}: gates.preflight.failure_kind must be known")
        require(errors, isinstance(preflight.get("raw_output"), str) and preflight["raw_output"], f"{path}: gates.preflight.raw_output must be non-empty")
    require(errors, gates.get("required_pass") is False, f"{path}: gates.required_pass must be false")
    require(errors, gates.get("overall_pass") is False, f"{path}: gates.overall_pass must be false")
    for key in BLOCKED_GATE_NAMES:
        gate = gates.get(key)
        require(errors, isinstance(gate, dict), f"{path}: gates.{key} must be an object")
        if isinstance(gate, dict):
            require(errors, gate.get("passed") is False, f"{path}: gates.{key}.passed must be false")
    return errors


def verify_stencil_artifact(data, path):
    errors = []
    require_exact_fields(errors, data, STENCIL_ARTIFACT_FIELDS, f"{path}: stencil artifact top-level")
    require(errors, data.get("benchmark") == "smollm-135m-stencil", f"{path}: benchmark must be smollm-135m-stencil")
    require(errors, data.get("prompt_tokens") == 128, f"{path}: prompt_tokens must be 128")
    require(errors, data.get("backend_capabilities") == "metal", f"{path}: backend_capabilities must be metal")
    rows = data.get("rows")
    require(errors, isinstance(rows, list), f"{path}: rows must be a list")
    if not isinstance(rows, list):
        return errors

    expected_phases = {"decode", "prompt"}
    seen_phases = set()
    for i, row in enumerate(rows):
        prefix = f"{path}: rows[{i}]"
        require(errors, isinstance(row, dict), f"{prefix} must be an object")
        if not isinstance(row, dict):
            continue
        extra = sorted(set(row) - set(STENCIL_ROW_FIELDS))
        require(errors, not extra, f"{prefix} has unexpected fields: {', '.join(extra)}")
        phase = row.get("phase")
        require(errors, phase in expected_phases, f"{prefix}.phase must be prompt or decode")
        if phase in expected_phases:
            seen_phases.add(phase)
            evidence = smollm_stencil_evidence(row, phase, data.get("prompt_tokens"))
            for check in failed_evidence_checks(evidence):
                errors.append(f"{prefix} failed {check}")
    require(errors, seen_phases == expected_phases, f"{path}: stencil rows must cover decode and prompt")
    return errors


def close_enough(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return abs(a - b) <= max(1e-9, abs(b) * 1e-9)


def verify_full_run_artifact(data, path):
    errors = []
    require_exact_fields(errors, data, FULL_RUN_ARTIFACT_FIELDS, f"{path}: full run top-level")
    require(errors, data.get("benchmark") == "smollm-135m", f"{path}: benchmark must be smollm-135m")
    require(errors, isinstance(data.get("date_utc"), str) and "T" in data["date_utc"] and data["date_utc"].endswith("Z"), f"{path}: date_utc must be a UTC timestamp")
    require(errors, isinstance(data.get("machine"), str) and data["machine"], f"{path}: machine must be non-empty")
    require(errors, isinstance(data.get("prompt_tokens"), int) and data["prompt_tokens"] > 0, f"{path}: prompt_tokens must be a positive integer")
    require(errors, isinstance(data.get("gen_tokens"), int) and data["gen_tokens"] > 0, f"{path}: gen_tokens must be a positive integer")
    require(errors, isinstance(data.get("repetitions"), int) and data["repetitions"] > 0, f"{path}: repetitions must be a positive integer")

    metadata = data.get("metadata")
    require(errors, isinstance(metadata, dict), f"{path}: metadata must be an object")
    if isinstance(metadata, dict):
        for key in REQUIRED_METADATA:
            require(errors, key in metadata, f"{path}: metadata.{key} is required")

    outputs = data.get("outputs")
    require(errors, isinstance(outputs, dict), f"{path}: outputs must be an object")
    if isinstance(outputs, dict):
        require_exact_fields(errors, outputs, FULL_RUN_OUTPUT_FIELDS, f"{path}: outputs")
        for key in FULL_RUN_OUTPUT_FIELDS:
            require(errors, isinstance(outputs.get(key), str) and outputs[key], f"{path}: outputs.{key} must be a non-empty path")

    try:
        lanes = lane_rows(data, path)
    except ValueError as err:
        errors.append(str(err))
        lanes = {}

    stored_native_rows = {}
    native_gate = ((data.get("gates") or {}).get("native_execution") or {})
    for row in native_gate.get("rows") or []:
        if isinstance(row, dict):
            stored_native_rows[(row.get("format"), row.get("phase"))] = row

    lane_evidence = {}
    for fmt in ("f16", "q8_0"):
        fmt_lanes = lanes.get(fmt)
        if not isinstance(fmt_lanes, dict):
            errors.append(f"{path}: summary.gate_zgml.{fmt} must be an object")
            continue
        for phase in ("prompt", "decode"):
            row = fmt_lanes.get(phase)
            stored_evidence = stored_native_rows.get((fmt, phase))
            errors.extend(verify_lane(data, path, fmt, phase, row, strict_fields=False, evidence_override=stored_evidence))
            if isinstance(row, dict):
                lane_evidence[(fmt, phase)] = stored_evidence if isinstance(stored_evidence, dict) else native_lane_evidence(row, fmt, phase, data.get("prompt_tokens"))

    gates = data.get("gates")
    require(errors, isinstance(gates, dict), f"{path}: gates must be an object")
    if not isinstance(gates, dict):
        return errors
    require_exact_fields(errors, gates, FULL_RUN_GATE_FIELDS, f"{path}: gates")

    native = gates.get("native_execution")
    require(errors, isinstance(native, dict), f"{path}: gates.native_execution must be an object")
    if isinstance(native, dict):
        require(errors, stored_native_evidence_is_current(native), f"{path}: gates.native_execution must carry current native evidence rows")
        require(errors, native.get("passed") is True, f"{path}: gates.native_execution.passed must be true for an accepted full run artifact")

    parity = gates.get("parity")
    require(errors, isinstance(parity, dict), f"{path}: gates.parity must be an object")
    parity_rows_ok = False
    parity_passed = False
    if isinstance(parity, dict):
        require(errors, parity.get("required") in (True, False), f"{path}: gates.parity.required must be boolean")
        require(errors, close_enough(parity.get("floor"), PARITY_FLOOR), f"{path}: gates.parity.floor must be {PARITY_FLOOR}")
        rows = parity.get("rows")
        require(errors, isinstance(rows, list), f"{path}: gates.parity.rows must be a list")
        if isinstance(rows, list):
            row_map = {(row.get("target"), row.get("phase")): row for row in rows if isinstance(row, dict)}
            parity_rows_ok = len(rows) == len(EXPECTED_PARITY_ROWS)
            for label, target, fmt, phase in EXPECTED_PARITY_ROWS:
                row = row_map.get((target, phase))
                require(errors, isinstance(row, dict), f"{path}: gates.parity.rows missing {label}")
                if not isinstance(row, dict):
                    parity_rows_ok = False
                    continue
                require(errors, row.get("label") == label, f"{path}: {label} parity row has wrong label")
                require(errors, row.get("target") == target, f"{path}: {label} parity row has wrong target")
                require(errors, row.get("phase") == phase, f"{path}: {label} parity row has wrong phase")
                zgml = row.get("zgml")
                require(errors, isinstance(zgml, dict), f"{path}: {label}.zgml must be an object")
                llama_tok_s = row.get("llama_tok_s")
                computed_parity = None
                if isinstance(zgml, dict) and isinstance(zgml.get("tok_s"), (int, float)) and isinstance(llama_tok_s, (int, float)) and llama_tok_s > 0:
                    computed_parity = zgml["tok_s"] / llama_tok_s
                    require(errors, close_enough(row.get("parity"), computed_parity), f"{path}: {label}.parity must equal zgml/llama throughput")
                else:
                    require(errors, False, f"{path}: {label} must carry positive zgml and llama throughput")
                evidence = lane_evidence.get((fmt, phase), {})
                require(errors, row.get("zgml_no_fallback") == evidence.get("no_fallback"), f"{path}: {label}.zgml_no_fallback must match native evidence")
                require(errors, row.get("zgml_profile_window_ok") == evidence.get("profile_calls_match"), f"{path}: {label}.zgml_profile_window_ok must match native evidence")
                require(errors, row.get("zgml_native_evidence_ok") == evidence.get("passed"), f"{path}: {label}.zgml_native_evidence_ok must match native evidence")
                expected_passed = (
                    computed_parity is not None
                    and computed_parity >= PARITY_FLOOR
                    and row.get("llama_backend_is_metal") is True
                    and evidence.get("passed") is True
                )
                require(errors, row.get("passed") is expected_passed, f"{path}: {label}.passed must match parity/reference/native evidence")
                parity_rows_ok = parity_rows_ok and row.get("passed") is expected_passed
            parity_passed = parity_rows_ok and all(row.get("passed") is True for row in rows if isinstance(row, dict))
            require(errors, parity.get("passed") is parity_passed, f"{path}: gates.parity.passed must match parity rows")

    reference = gates.get("reference_backend")
    require(errors, isinstance(reference, dict), f"{path}: gates.reference_backend must be an object")
    reference_passed = False
    if isinstance(reference, dict):
        rows = reference.get("rows")
        require(errors, isinstance(rows, list), f"{path}: gates.reference_backend.rows must be a list")
        if isinstance(rows, list):
            require(errors, len(rows) == len(EXPECTED_PARITY_ROWS), f"{path}: gates.reference_backend.rows must cover every parity row")
            reference_passed = all(isinstance(row, dict) and row.get("passed") is True for row in rows)
            require(errors, reference.get("passed") is reference_passed, f"{path}: gates.reference_backend.passed must match rows")

    baseline = gates.get("baseline")
    require(errors, isinstance(baseline, dict), f"{path}: gates.baseline must be an object")
    baseline_passed = False
    if isinstance(baseline, dict):
        require(errors, close_enough(baseline.get("floor_ratio"), BASELINE_FLOOR), f"{path}: gates.baseline.floor_ratio must be {BASELINE_FLOOR}")
        require(errors, close_enough(baseline.get("counter_ceiling_ratio"), BASELINE_COUNTER_CEILING), f"{path}: gates.baseline.counter_ceiling_ratio must be {BASELINE_COUNTER_CEILING}")
        rows = baseline.get("rows")
        require(errors, isinstance(rows, list), f"{path}: gates.baseline.rows must be a list")
        if isinstance(rows, list):
            baseline_passed = baseline.get("baseline_json") is None or all(isinstance(row, dict) and row.get("passed") is True for row in rows)
            require(errors, baseline.get("passed") is baseline_passed, f"{path}: gates.baseline.passed must match baseline rows")

    native_passed = isinstance(native, dict) and native.get("passed") is True
    expected_overall = parity_passed and reference_passed and baseline_passed and native_passed
    expected_required = reference_passed and native_passed and (not (isinstance(parity, dict) and parity.get("required") is True) or parity_passed) and baseline_passed
    require(errors, gates.get("overall_pass") is expected_overall, f"{path}: gates.overall_pass must match component gates")
    require(errors, gates.get("required_pass") is expected_required, f"{path}: gates.required_pass must match required component gates")
    return errors


def positive_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def verify_laptop_llm_pytorch_artifact(data, path):
    errors = []
    require_exact_fields(errors, data, LAPTOP_LLM_PYTORCH_FIELDS, f"{path}: laptop LLM PyTorch artifact top-level")
    require(errors, data.get("schema") == "zgml.laptop-llm-pytorch-comparison.v1", f"{path}: schema must be zgml.laptop-llm-pytorch-comparison.v1")
    require(errors, data.get("command") == "scripts/check_laptop_llm_pytorch_comparison.cjs", f"{path}: command must be the laptop LLM comparison script")
    require(errors, data.get("suiteId") == "llm.smollm2_360m.instruct.q8_0", f"{path}: suiteId must be llm.smollm2_360m.instruct.q8_0")
    require(errors, data.get("modelId") == "HuggingFaceTB/SmolLM2-360M-Instruct", f"{path}: modelId must be HuggingFaceTB/SmolLM2-360M-Instruct")
    require(errors, isinstance(data.get("createdAt"), str) and "T" in data["createdAt"] and data["createdAt"].endswith("Z"), f"{path}: createdAt must be a UTC timestamp")
    require(errors, isinstance(data.get("prompt"), str) and data["prompt"], f"{path}: prompt must be non-empty")

    status = data.get("status")
    require(errors, isinstance(status, dict), f"{path}: status must be an object")
    if isinstance(status, dict):
        require(errors, status.get("pytorchReady") is True, f"{path}: status.pytorchReady must be true")
        require(errors, status.get("zgmlProbeReady") is True, f"{path}: status.zgmlProbeReady must be true")
        require(errors, status.get("zgmlReady") is True, f"{path}: status.zgmlReady must be true")
        require(errors, status.get("comparisonReady") is True, f"{path}: status.comparisonReady must be true")

    pytorch = data.get("pytorch")
    zgml = data.get("zgml")
    require(errors, isinstance(pytorch, dict), f"{path}: pytorch must be an object")
    require(errors, isinstance(zgml, dict), f"{path}: zgml must be an object")
    if not isinstance(pytorch, dict) or not isinstance(zgml, dict):
        return errors

    require(errors, pytorch.get("modelId") == data.get("modelId"), f"{path}: pytorch.modelId must match modelId")
    require(errors, isinstance(pytorch.get("modelPath"), str) and pytorch["modelPath"].endswith(".safetensors"), f"{path}: pytorch.modelPath must be a safetensors path")
    require(errors, isinstance(pytorch.get("promptTokens"), int) and pytorch["promptTokens"] > 0, f"{path}: pytorch.promptTokens must be positive")
    prompt_ids = pytorch.get("promptTokenIds")
    require(errors, isinstance(prompt_ids, list) and len(prompt_ids) == pytorch.get("promptTokens"), f"{path}: pytorch.promptTokenIds must match promptTokens")
    if isinstance(prompt_ids, list):
        require(errors, all(isinstance(token, int) and token >= 0 for token in prompt_ids), f"{path}: pytorch.promptTokenIds must be non-negative integers")
    require(errors, isinstance(pytorch.get("decodeTokens"), int) and pytorch["decodeTokens"] > 0, f"{path}: pytorch.decodeTokens must be positive")
    require(errors, isinstance(pytorch.get("prefillIters"), int) and pytorch["prefillIters"] > 0, f"{path}: pytorch.prefillIters must be positive")
    for key in ("loadMs", "prefillMs", "prefillTokS", "decodeMsPerToken", "decodeTokS"):
        require(errors, positive_number(pytorch.get(key)), f"{path}: pytorch.{key} must be positive")
    require(errors, positive_number(pytorch.get("rssLoadDeltaBytes")), f"{path}: pytorch.rssLoadDeltaBytes must be positive")

    require(errors, zgml.get("supported") is True, f"{path}: zgml.supported must be true")
    require(errors, zgml.get("probeReady") is True, f"{path}: zgml.probeReady must be true")
    require(errors, zgml.get("executableReady") is True, f"{path}: zgml.executableReady must be true")
    require(errors, zgml.get("stage") == "execute", f"{path}: zgml.stage must be execute")
    require(errors, isinstance(zgml.get("contextLength"), int) and zgml["contextLength"] >= pytorch.get("promptTokens", 0), f"{path}: zgml.contextLength must cover the prompt")
    for key in ("probeMs", "loadMs", "compileMs", "bindMs", "prefillMs", "prefillTokS", "decodeMsPerToken", "decodeTokS"):
        require(errors, positive_number(zgml.get(key)), f"{path}: zgml.{key} must be positive")
    probe = zgml.get("probe")
    require(errors, isinstance(probe, dict), f"{path}: zgml.probe must be an object")
    if isinstance(probe, dict):
        require(errors, probe.get("modelKind") == "smollm2-360m", f"{path}: zgml.probe.modelKind must be smollm2-360m")
        require(errors, probe.get("vocabSize") == 49152, f"{path}: zgml.probe.vocabSize must be 49152")
    require(errors, isinstance(zgml.get("firstToken"), int), f"{path}: zgml.firstToken must be an integer")
    pytorch_tokens = pytorch.get("generatedTokenIds")
    zgml_tokens = zgml.get("generatedTokenIds")
    require(errors, isinstance(pytorch_tokens, list) and len(pytorch_tokens) == pytorch.get("decodeTokens"), f"{path}: pytorch.generatedTokenIds must match decodeTokens")
    require(errors, isinstance(zgml_tokens, list) and len(zgml_tokens) == pytorch.get("decodeTokens"), f"{path}: zgml.generatedTokenIds must match decodeTokens")
    require(errors, pytorch_tokens == zgml_tokens, f"{path}: generated token IDs must match PyTorch")
    return errors


def verify_artifact(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("schema") == "zgml.laptop-llm-pytorch-comparison.v1":
        return verify_laptop_llm_pytorch_artifact(data, path)

    if data.get("benchmark") == "smollm-135m-stencil":
        return verify_stencil_artifact(data, path)

    if isinstance((data.get("gates") or {}).get("preflight"), dict):
        return verify_preflight_artifact(data, path)

    if is_full_run_artifact(data):
        return verify_full_run_artifact(data, path)

    errors = []
    require_exact_fields(errors, data, COMPACT_BASELINE_FIELDS, f"{path}: compact baseline top-level")
    require(errors, data.get("benchmark") == "smollm-135m", f"{path}: benchmark must be smollm-135m")
    require(errors, isinstance(data.get("date_utc"), str) and "T" in data["date_utc"] and data["date_utc"].endswith("Z"), f"{path}: date_utc must be a UTC timestamp")
    require(errors, data.get("machine") == "Apple M5 Pro", f"{path}: machine must be Apple M5 Pro")
    require(errors, data.get("prompt_tokens") == 128, f"{path}: prompt_tokens must be 128")
    require(errors, data.get("gen_tokens") == 200, f"{path}: gen_tokens must be 200")
    require(errors, data.get("repetitions") == 3, f"{path}: repetitions must be 3")
    metadata = data.get("metadata")
    require(errors, isinstance(metadata, dict), f"{path}: metadata must be an object")
    if isinstance(metadata, dict):
        for key in REQUIRED_METADATA:
            require(errors, key in metadata, f"{path}: metadata.{key} is required")
        extra_metadata = sorted(set(metadata) - set(REQUIRED_METADATA))
        require(errors, not extra_metadata, f"{path}: compact baseline has unexpected metadata fields: {', '.join(extra_metadata)}")
    summary = data.get("summary")
    if isinstance(summary, dict):
        extra_summary = sorted(set(summary) - set(COMPACT_SUMMARY_FIELDS))
        require(errors, not extra_summary, f"{path}: compact baseline has unexpected summary fields: {', '.join(extra_summary)}")

    try:
        lanes = lane_rows(data, path)
    except ValueError as err:
        errors.append(str(err))
        return errors

    for fmt in ("f16", "q8_0"):
        fmt_lanes = lanes.get(fmt)
        if not isinstance(fmt_lanes, dict):
            errors.append(f"{path}: summary.gate_zgml.{fmt} must be an object")
            continue
        for phase in ("prompt", "decode"):
            errors.extend(verify_lane(data, path, fmt, phase, fmt_lanes.get(phase)))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="print concise artifact evidence status for passing artifacts")
    parser.add_argument("artifacts", nargs="+", help="benchmark JSON artifact(s)")
    args = parser.parse_args()

    ok = True
    contract_errors = verify_contract_helpers()
    if contract_errors:
        ok = False
        print("bench_contract self-test: FAIL")
        for error in contract_errors:
            print(f"  - {error}")
    for path in args.artifacts:
        errors = verify_artifact(path)
        if errors:
            ok = False
            print(f"{path}: FAIL")
            for error in errors:
                print(f"  - {error}")
        else:
            print(status_line(path) if args.status else f"{path}: pass")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
