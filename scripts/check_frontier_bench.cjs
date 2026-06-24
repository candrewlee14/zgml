"use strict";

const { spawnSync } = require("node:child_process");
const { mkdirSync, writeFileSync } = require("node:fs");
const { join, resolve } = require("node:path");
const os = require("node:os");
const { benchmarkBinaryMetadata } = require("./native_freshness.cjs");

const root = resolve(__dirname, "..");
const frontierBinary = "./zig-out/bin/bench-frontier";

const maxAttempts = positiveInt(process.env.BENCH_FRONTIER_ATTEMPTS || "5", "BENCH_FRONTIER_ATTEMPTS");
const largeChainSpeedupFloor = 2.95;
const decodeishTokSFloor = 400_000;
const projectionChainTileSpeedupFloor = 0.95;
const projectionChainFullPrefillSpeedupFloor = 0.90;
const projectionChainFullPrefillCandidateSpeedupFloor = 1.00;
const projectionChainMaxAbsDiffCeil = 0.002;
const projectionGroupCandidateSpeedupFloor = 1.05;
const projectionGroupRegionSpeedupFloor = 1.00;
const projectionSmollmPromptSpeedupFloor = 0.90;
const projectionRowChainDefaultSpeedupFloor = 1.10;
const projectionRowChainMaxAbsDiffCeil = 0.02;
const projectionRowChainLowering = "prompt_split_tiled_qmatmul_plus_rmsnorm";
const projectionRowChainDiagnosticKernel = "single_dispatch_tiled_candidate";
const projectionRowChainNextTarget = "semantic_sublayer_or_two_phase_tile_parallel_row_chain";
const semanticCommandStatusToken = "semantic_command_status=preserves_default_work_shape";
const build = process.env.BENCH_FRONTIER_BUILD ?? "1";
const frontierFilter = process.env.BENCH_FRONTIER_FILTER ?? "";
const qsemanticVariants = (process.env.BENCH_QSEMANTIC_VARIANTS ?? "")
  .split(",")
  .map((part) => part.trim())
  .filter(Boolean);
const writeArtifact = process.env.BENCH_FRONTIER_WRITE_ARTIFACT !== "0";
const artifactDir = process.env.BENCH_FRONTIER_ARTIFACT_DIR || join("bench-results", "frontier");

function positiveInt(value, label) {
  const n = Number(value);
  if (!Number.isSafeInteger(n) || n <= 0) {
    throw new Error(`${label} must be a positive integer, got ${value}`);
  }
  return n;
}

function timestampForArtifact(date = new Date()) {
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function roundMetric(value) {
  return Number.isFinite(value) ? Number(value.toFixed(6)) : null;
}

function runBench() {
  const command = build === "1" ? "zig" : build === "0" ? "./zig-out/bin/bench-frontier" : null;
  const args = build === "1" ? ["build", "-Doptimize=ReleaseFast", "bench-frontier"] : build === "0" ? [] : null;
  if (command === null || args === null) {
    throw new Error(`BENCH_FRONTIER_BUILD must be 0 or 1, got ${build}`);
  }
  const result = spawnSync(command, args, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  const output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
  if (result.status !== 0) {
    const err = new Error(`frontier bench command failed with status ${result.status ?? 1}`);
    err.output = output;
    throw err;
  }
  return output;
}

let cachedMetricOutput = null;
let cachedMetricRows = null;

function frontierMetricRows(output) {
  if (cachedMetricOutput === output && cachedMetricRows !== null) return cachedMetricRows;
  const prefix = "ZGML_FRONTIER_METRIC_JSON ";
  const rows = [];
  for (const line of output.split(/\r?\n/)) {
    if (!line.startsWith(prefix)) continue;
    const raw = line.slice(prefix.length);
    try {
      const row = JSON.parse(raw);
      if (row && typeof row === "object" && typeof row.label === "string") {
        rows.push(row);
      }
    } catch {
      // Keep the human-readable text parser as the compatibility fallback.
    }
  }
  cachedMetricOutput = output;
  cachedMetricRows = rows;
  return rows;
}

function jsonMetric(output, label, key) {
  for (const row of frontierMetricRows(output)) {
    if (row.label !== label || !(key in row)) continue;
    const value = Number(row[key]);
    if (Number.isFinite(value)) return value;
    throw new Error(`frontier bench invalid JSON metric ${label} ${key}: ${row[key]}`);
  }
  return null;
}

function metric(output, label, key) {
  const structured = jsonMetric(output, label, key);
  if (structured !== null) return structured;
  const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = output.match(new RegExp(`${escaped}[^\\n]*${key}=\\s*([0-9]+(?:\\.[0-9]+)?)`));
  if (!match) {
    throw new Error(`frontier bench missing ${label} ${key}`);
  }
  const value = Number(match[1]);
  if (!Number.isFinite(value)) {
    throw new Error(`frontier bench invalid ${label} ${key}: ${match[1]}`);
  }
  return value;
}

function p50(output, label) {
  return metric(output, label, "p50");
}

function throughput(output, label, key) {
  return metric(output, label, key);
}

function hasMetric(output, label, key) {
  if (jsonMetric(output, label, key) !== null) return true;
  const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`${escaped}[^\\n]*${key}=`).test(output);
}

function isFocusedQprojFilter(filter) {
  return /\bqproj\b|projection_chain|projection_group/.test(filter) && !/\bqrow\b|projection_row_chain/.test(filter);
}

function isFocusedQrowRegionFilter(filter) {
  return /\bqrow\b/.test(filter) && /\bregion\b|two_phase_group/.test(filter);
}

function isFocusedSemanticFilter(filter) {
  return /\bqsemantic\b|semantic pair_row_chain/.test(filter);
}

function isQsemanticThroughputOnly() {
  return qsemanticVariants.length === 1 && qsemanticVariants[0] === "throughput_candidate";
}

function scoreFocusedQproj(output, attempt) {
  const projectionChainLabel = "qproj prompt m=32 n=512 k=512 projection_chain";
  const projectionChainFullPrefillLabel = "qproj full-prefill m=128 n=512 k=512 projection_chain";
  const projectionChainSmollmPromptLabel = "qproj smollm-prompt m=128 n=576 k=576 projection_chain";
  const projectionGroupFullPrefillLabel = "qproj group full-prefill x4 m=128 n=512 k=512 projection_group";
  const projectionGroupSmollmPromptLabel = "qproj group smollm-prompt x4 m=128 n=576 k=576 projection_group";
  const projectionGroupRegionFullPrefillLabel = "qproj region full-prefill x7 m=128 n=512 k=512 projection_group";
  const projectionGroupRegionSmollmPromptLabel = "qproj region smollm-prompt x7 m=128 n=576 k=576 projection_group";
  const projectionGroupFullPrefillProfileLabel = `${projectionGroupFullPrefillLabel} dispatch_profile`;
  const projectionGroupSmollmPromptProfileLabel = `${projectionGroupSmollmPromptLabel} dispatch_profile`;
  const projectionGroupRegionFullPrefillProfileLabel = `${projectionGroupRegionFullPrefillLabel} dispatch_profile`;
  const projectionGroupRegionSmollmPromptProfileLabel = `${projectionGroupRegionSmollmPromptLabel} dispatch_profile`;

  const projectionChainSpeedup = metric(output, projectionChainLabel, "speedup");
  const projectionChainMaxAbsDiff = metric(output, projectionChainLabel, "max_abs_diff");
  const projectionChainFullPrefillSpeedup = metric(output, projectionChainFullPrefillLabel, "speedup");
  const projectionChainFullPrefillMaxAbsDiff = metric(output, projectionChainFullPrefillLabel, "max_abs_diff");
  const projectionChainSmollmPromptSpeedup = metric(output, projectionChainSmollmPromptLabel, "speedup");
  const projectionChainSmollmPromptMaxAbsDiff = metric(output, projectionChainSmollmPromptLabel, "max_abs_diff");
  const projectionGroupFullPrefillSpeedup = metric(output, projectionGroupFullPrefillLabel, "speedup");
  const projectionGroupFullPrefillMaxAbsDiff = metric(output, projectionGroupFullPrefillLabel, "max_abs_diff");
  const projectionGroupSmollmPromptSpeedup = metric(output, projectionGroupSmollmPromptLabel, "speedup");
  const projectionGroupSmollmPromptMaxAbsDiff = metric(output, projectionGroupSmollmPromptLabel, "max_abs_diff");
  const projectionGroupRegionFullPrefillSpeedup = metric(output, projectionGroupRegionFullPrefillLabel, "speedup");
  const projectionGroupRegionFullPrefillMaxAbsDiff = metric(output, projectionGroupRegionFullPrefillLabel, "max_abs_diff");
  const projectionGroupRegionSmollmPromptSpeedup = metric(output, projectionGroupRegionSmollmPromptLabel, "speedup");
  const projectionGroupRegionSmollmPromptMaxAbsDiff = metric(output, projectionGroupRegionSmollmPromptLabel, "max_abs_diff");
  const projectionGroupFullPrefillShapeCommands = metric(output, projectionGroupFullPrefillProfileLabel, "shape_commands");
  const projectionGroupFullPrefillShapeGroups = metric(output, projectionGroupFullPrefillProfileLabel, "shape_projection_groups");
  const projectionGroupFullPrefillShapeCoveredOps = metric(output, projectionGroupFullPrefillProfileLabel, "shape_covered_ops");
  const projectionGroupFullPrefillShapeSavedDispatches = metric(output, projectionGroupFullPrefillProfileLabel, "shape_saved_dispatches");
  const projectionGroupFullPrefillRuntimeDispatches = metric(output, projectionGroupFullPrefillProfileLabel, "runtime_projection_group_dispatches");
  const projectionGroupFullPrefillRuntimeCacheDispatches = metric(output, projectionGroupFullPrefillProfileLabel, "runtime_projection_cache_group_dispatches");
  const projectionGroupSmollmPromptShapeCommands = metric(output, projectionGroupSmollmPromptProfileLabel, "shape_commands");
  const projectionGroupSmollmPromptShapeGroups = metric(output, projectionGroupSmollmPromptProfileLabel, "shape_projection_groups");
  const projectionGroupSmollmPromptShapeCoveredOps = metric(output, projectionGroupSmollmPromptProfileLabel, "shape_covered_ops");
  const projectionGroupSmollmPromptShapeSavedDispatches = metric(output, projectionGroupSmollmPromptProfileLabel, "shape_saved_dispatches");
  const projectionGroupSmollmPromptRuntimeDispatches = metric(output, projectionGroupSmollmPromptProfileLabel, "runtime_projection_group_dispatches");
  const projectionGroupSmollmPromptRuntimeCacheDispatches = metric(output, projectionGroupSmollmPromptProfileLabel, "runtime_projection_cache_group_dispatches");
  const projectionGroupRegionFullPrefillShapeCommands = metric(output, projectionGroupRegionFullPrefillProfileLabel, "shape_commands");
  const projectionGroupRegionFullPrefillShapeGroups = metric(output, projectionGroupRegionFullPrefillProfileLabel, "shape_projection_groups");
  const projectionGroupRegionFullPrefillShapeCoveredOps = metric(output, projectionGroupRegionFullPrefillProfileLabel, "shape_covered_ops");
  const projectionGroupRegionFullPrefillShapeSavedDispatches = metric(output, projectionGroupRegionFullPrefillProfileLabel, "shape_saved_dispatches");
  const projectionGroupRegionFullPrefillRuntimeDispatches = metric(output, projectionGroupRegionFullPrefillProfileLabel, "runtime_projection_group_dispatches");
  const projectionGroupRegionFullPrefillRuntimeCacheDispatches = metric(output, projectionGroupRegionFullPrefillProfileLabel, "runtime_projection_cache_group_dispatches");
  const projectionGroupRegionSmollmPromptShapeCommands = metric(output, projectionGroupRegionSmollmPromptProfileLabel, "shape_commands");
  const projectionGroupRegionSmollmPromptShapeGroups = metric(output, projectionGroupRegionSmollmPromptProfileLabel, "shape_projection_groups");
  const projectionGroupRegionSmollmPromptShapeCoveredOps = metric(output, projectionGroupRegionSmollmPromptProfileLabel, "shape_covered_ops");
  const projectionGroupRegionSmollmPromptShapeSavedDispatches = metric(output, projectionGroupRegionSmollmPromptProfileLabel, "shape_saved_dispatches");
  const projectionGroupRegionSmollmPromptRuntimeDispatches = metric(output, projectionGroupRegionSmollmPromptProfileLabel, "runtime_projection_group_dispatches");
  const projectionGroupRegionSmollmPromptRuntimeCacheDispatches = metric(output, projectionGroupRegionSmollmPromptProfileLabel, "runtime_projection_cache_group_dispatches");

  const failures = [];
  if (projectionChainSpeedup < projectionChainTileSpeedupFloor) failures.push(`projection_chain prompt tile ${projectionChainSpeedup.toFixed(2)}x < ${projectionChainTileSpeedupFloor.toFixed(2)}x`);
  if (projectionChainMaxAbsDiff > projectionChainMaxAbsDiffCeil) failures.push(`projection_chain prompt max_abs_diff ${projectionChainMaxAbsDiff.toFixed(6)} > ${projectionChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionChainFullPrefillSpeedup < projectionChainFullPrefillSpeedupFloor) failures.push(`projection_chain full-prefill ${projectionChainFullPrefillSpeedup.toFixed(2)}x < ${projectionChainFullPrefillSpeedupFloor.toFixed(2)}x`);
  if (projectionChainFullPrefillMaxAbsDiff > projectionChainMaxAbsDiffCeil) failures.push(`projection_chain full-prefill max_abs_diff ${projectionChainFullPrefillMaxAbsDiff.toFixed(6)} > ${projectionChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionChainSmollmPromptSpeedup < projectionSmollmPromptSpeedupFloor) failures.push(`projection_chain smollm-prompt ${projectionChainSmollmPromptSpeedup.toFixed(2)}x < ${projectionSmollmPromptSpeedupFloor.toFixed(2)}x`);
  if (projectionChainSmollmPromptMaxAbsDiff > projectionChainMaxAbsDiffCeil) failures.push(`projection_chain smollm-prompt max_abs_diff ${projectionChainSmollmPromptMaxAbsDiff.toFixed(6)} > ${projectionChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionGroupFullPrefillMaxAbsDiff > projectionChainMaxAbsDiffCeil) failures.push(`projection_group full-prefill max_abs_diff ${projectionGroupFullPrefillMaxAbsDiff.toFixed(6)} > ${projectionChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionGroupSmollmPromptSpeedup < projectionSmollmPromptSpeedupFloor) failures.push(`projection_group smollm-prompt ${projectionGroupSmollmPromptSpeedup.toFixed(2)}x < ${projectionSmollmPromptSpeedupFloor.toFixed(2)}x`);
  if (projectionGroupSmollmPromptMaxAbsDiff > projectionChainMaxAbsDiffCeil) failures.push(`projection_group smollm-prompt max_abs_diff ${projectionGroupSmollmPromptMaxAbsDiff.toFixed(6)} > ${projectionChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionGroupFullPrefillShapeCommands !== 1 || projectionGroupFullPrefillShapeGroups !== 1 || projectionGroupFullPrefillShapeCoveredOps !== 8 || projectionGroupFullPrefillShapeSavedDispatches !== 7) failures.push("projection_group full-prefill shape profile must stay shape_commands=1 shape_projection_groups=1 shape_covered_ops=8 shape_saved_dispatches=7");
  if (projectionGroupSmollmPromptShapeCommands !== 1 || projectionGroupSmollmPromptShapeGroups !== 1 || projectionGroupSmollmPromptShapeCoveredOps !== 8 || projectionGroupSmollmPromptShapeSavedDispatches !== 7) failures.push("projection_group smollm-prompt shape profile must stay shape_commands=1 shape_projection_groups=1 shape_covered_ops=8 shape_saved_dispatches=7");
  if (projectionGroupRegionFullPrefillSpeedup < projectionGroupRegionSpeedupFloor) failures.push(`projection_group region full-prefill ${projectionGroupRegionFullPrefillSpeedup.toFixed(2)}x < ${projectionGroupRegionSpeedupFloor.toFixed(2)}x`);
  if (projectionGroupRegionFullPrefillMaxAbsDiff > projectionChainMaxAbsDiffCeil) failures.push(`projection_group region full-prefill max_abs_diff ${projectionGroupRegionFullPrefillMaxAbsDiff.toFixed(6)} > ${projectionChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionGroupRegionSmollmPromptSpeedup < projectionGroupRegionSpeedupFloor) failures.push(`projection_group region smollm-prompt ${projectionGroupRegionSmollmPromptSpeedup.toFixed(2)}x < ${projectionGroupRegionSpeedupFloor.toFixed(2)}x`);
  if (projectionGroupRegionSmollmPromptMaxAbsDiff > projectionChainMaxAbsDiffCeil) failures.push(`projection_group region smollm-prompt max_abs_diff ${projectionGroupRegionSmollmPromptMaxAbsDiff.toFixed(6)} > ${projectionChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionGroupRegionFullPrefillShapeCommands !== 2 || projectionGroupRegionFullPrefillShapeGroups !== 2 || projectionGroupRegionFullPrefillShapeCoveredOps !== 14 || projectionGroupRegionFullPrefillShapeSavedDispatches !== 12 || projectionGroupRegionFullPrefillRuntimeDispatches !== 2 || projectionGroupRegionFullPrefillRuntimeCacheDispatches !== 0) failures.push("projection_group region full-prefill runtime profile must stay shape_commands=2 shape_projection_groups=2 shape_covered_ops=14 shape_saved_dispatches=12 runtime_projection_group_dispatches=2 runtime_projection_cache_group_dispatches=0");
  if (projectionGroupRegionSmollmPromptShapeCommands !== 2 || projectionGroupRegionSmollmPromptShapeGroups !== 2 || projectionGroupRegionSmollmPromptShapeCoveredOps !== 14 || projectionGroupRegionSmollmPromptShapeSavedDispatches !== 12 || projectionGroupRegionSmollmPromptRuntimeDispatches !== 2 || projectionGroupRegionSmollmPromptRuntimeCacheDispatches !== 0) failures.push("projection_group region smollm-prompt runtime profile must stay shape_commands=2 shape_projection_groups=2 shape_covered_ops=14 shape_saved_dispatches=12 runtime_projection_group_dispatches=2 runtime_projection_cache_group_dispatches=0");

  const line = [
    `frontier qproj gate: ${failures.length === 0 ? "pass" : "fail"}`,
    `attempt=${attempt}/${maxAttempts}`,
    `projection_chain_prompt=${projectionChainSpeedup.toFixed(2)}x floor=${projectionChainTileSpeedupFloor.toFixed(2)} max_abs_diff=${projectionChainMaxAbsDiff.toFixed(6)} diff_ceil=${projectionChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_chain_full_prefill=${projectionChainFullPrefillSpeedup.toFixed(2)}x floor=${projectionChainFullPrefillSpeedupFloor.toFixed(2)} candidate=${projectionChainFullPrefillSpeedup >= projectionChainFullPrefillCandidateSpeedupFloor ? "ready" : "off"} max_abs_diff=${projectionChainFullPrefillMaxAbsDiff.toFixed(6)}`,
    `projection_chain_smollm_prompt=${projectionChainSmollmPromptSpeedup.toFixed(2)}x floor=${projectionSmollmPromptSpeedupFloor.toFixed(2)} max_abs_diff=${projectionChainSmollmPromptMaxAbsDiff.toFixed(6)}`,
    `projection_group_full_prefill=${projectionGroupFullPrefillSpeedup.toFixed(2)}x candidate=${projectionGroupFullPrefillSpeedup >= projectionGroupCandidateSpeedupFloor ? "ready" : "off"} candidate_floor=${projectionGroupCandidateSpeedupFloor.toFixed(2)} max_abs_diff=${projectionGroupFullPrefillMaxAbsDiff.toFixed(6)} shape_commands=${projectionGroupFullPrefillShapeCommands} shape_projection_groups=${projectionGroupFullPrefillShapeGroups} shape_covered_ops=${projectionGroupFullPrefillShapeCoveredOps} shape_saved_dispatches=${projectionGroupFullPrefillShapeSavedDispatches} runtime=${projectionGroupFullPrefillRuntimeDispatches + projectionGroupFullPrefillRuntimeCacheDispatches > 0 ? "command" : "off"} runtime_projection_group_dispatches=${projectionGroupFullPrefillRuntimeDispatches} runtime_projection_cache_group_dispatches=${projectionGroupFullPrefillRuntimeCacheDispatches}`,
    `projection_group_smollm_prompt=${projectionGroupSmollmPromptSpeedup.toFixed(2)}x floor=${projectionSmollmPromptSpeedupFloor.toFixed(2)} max_abs_diff=${projectionGroupSmollmPromptMaxAbsDiff.toFixed(6)} shape_commands=${projectionGroupSmollmPromptShapeCommands} shape_projection_groups=${projectionGroupSmollmPromptShapeGroups} shape_covered_ops=${projectionGroupSmollmPromptShapeCoveredOps} shape_saved_dispatches=${projectionGroupSmollmPromptShapeSavedDispatches} runtime=${projectionGroupSmollmPromptRuntimeDispatches + projectionGroupSmollmPromptRuntimeCacheDispatches > 0 ? "command" : "off"} runtime_projection_group_dispatches=${projectionGroupSmollmPromptRuntimeDispatches} runtime_projection_cache_group_dispatches=${projectionGroupSmollmPromptRuntimeCacheDispatches}`,
    `projection_group_region_full_prefill=${projectionGroupRegionFullPrefillSpeedup.toFixed(2)}x floor=${projectionGroupRegionSpeedupFloor.toFixed(2)} max_abs_diff=${projectionGroupRegionFullPrefillMaxAbsDiff.toFixed(6)} shape_commands=${projectionGroupRegionFullPrefillShapeCommands} shape_projection_groups=${projectionGroupRegionFullPrefillShapeGroups} shape_covered_ops=${projectionGroupRegionFullPrefillShapeCoveredOps} shape_saved_dispatches=${projectionGroupRegionFullPrefillShapeSavedDispatches} runtime=${projectionGroupRegionFullPrefillRuntimeDispatches > 0 ? "command" : "off"} runtime_projection_group_dispatches=${projectionGroupRegionFullPrefillRuntimeDispatches} runtime_projection_cache_group_dispatches=${projectionGroupRegionFullPrefillRuntimeCacheDispatches}`,
    `projection_group_region_smollm_prompt=${projectionGroupRegionSmollmPromptSpeedup.toFixed(2)}x floor=${projectionGroupRegionSpeedupFloor.toFixed(2)} max_abs_diff=${projectionGroupRegionSmollmPromptMaxAbsDiff.toFixed(6)} shape_commands=${projectionGroupRegionSmollmPromptShapeCommands} shape_projection_groups=${projectionGroupRegionSmollmPromptShapeGroups} shape_covered_ops=${projectionGroupRegionSmollmPromptShapeCoveredOps} shape_saved_dispatches=${projectionGroupRegionSmollmPromptShapeSavedDispatches} runtime=${projectionGroupRegionSmollmPromptRuntimeDispatches > 0 ? "command" : "off"} runtime_projection_group_dispatches=${projectionGroupRegionSmollmPromptRuntimeDispatches} runtime_projection_cache_group_dispatches=${projectionGroupRegionSmollmPromptRuntimeCacheDispatches}`,
  ].join("; ");

  return {
    attempt,
    projectionChainSpeedup,
    projectionChainMaxAbsDiff,
    projectionChainFullPrefillSpeedup,
    projectionChainFullPrefillMaxAbsDiff,
    projectionChainSmollmPromptSpeedup,
    projectionChainSmollmPromptMaxAbsDiff,
    projectionGroupFullPrefillSpeedup,
    projectionGroupFullPrefillMaxAbsDiff,
    projectionGroupFullPrefillShapeCommands,
    projectionGroupFullPrefillShapeGroups,
    projectionGroupFullPrefillShapeCoveredOps,
    projectionGroupFullPrefillShapeSavedDispatches,
    projectionGroupFullPrefillRuntimeDispatches,
    projectionGroupFullPrefillRuntimeCacheDispatches,
    projectionGroupSmollmPromptSpeedup,
    projectionGroupSmollmPromptMaxAbsDiff,
    projectionGroupSmollmPromptShapeCommands,
    projectionGroupSmollmPromptShapeGroups,
    projectionGroupSmollmPromptShapeCoveredOps,
    projectionGroupSmollmPromptShapeSavedDispatches,
    projectionGroupSmollmPromptRuntimeDispatches,
    projectionGroupSmollmPromptRuntimeCacheDispatches,
    projectionGroupRegionFullPrefillSpeedup,
    projectionGroupRegionFullPrefillMaxAbsDiff,
    projectionGroupRegionSmollmPromptSpeedup,
    projectionGroupRegionSmollmPromptMaxAbsDiff,
    projectionGroupRegionFullPrefillShapeCommands,
    projectionGroupRegionFullPrefillShapeGroups,
    projectionGroupRegionFullPrefillShapeCoveredOps,
    projectionGroupRegionFullPrefillShapeSavedDispatches,
    projectionGroupRegionFullPrefillRuntimeDispatches,
    projectionGroupRegionFullPrefillRuntimeCacheDispatches,
    projectionGroupRegionSmollmPromptShapeCommands,
    projectionGroupRegionSmollmPromptShapeGroups,
    projectionGroupRegionSmollmPromptShapeCoveredOps,
    projectionGroupRegionSmollmPromptShapeSavedDispatches,
    projectionGroupRegionSmollmPromptRuntimeDispatches,
    projectionGroupRegionSmollmPromptRuntimeCacheDispatches,
    failures,
    line,
  };
}

function focusedQprojMargin(current) {
  return Math.min(
    current.projectionChainSpeedup / projectionChainTileSpeedupFloor,
    projectionChainMaxAbsDiffCeil / Math.max(current.projectionChainMaxAbsDiff, Number.EPSILON),
    current.projectionChainFullPrefillSpeedup / projectionChainFullPrefillSpeedupFloor,
    projectionChainMaxAbsDiffCeil / Math.max(current.projectionChainFullPrefillMaxAbsDiff, Number.EPSILON),
    current.projectionChainSmollmPromptSpeedup / projectionSmollmPromptSpeedupFloor,
    projectionChainMaxAbsDiffCeil / Math.max(current.projectionChainSmollmPromptMaxAbsDiff, Number.EPSILON),
    projectionChainMaxAbsDiffCeil / Math.max(current.projectionGroupFullPrefillMaxAbsDiff, Number.EPSILON),
    current.projectionGroupFullPrefillShapeCommands === 1 ? 1 : 0,
    current.projectionGroupFullPrefillShapeGroups === 1 ? 1 : 0,
    current.projectionGroupFullPrefillShapeCoveredOps === 8 ? 1 : 0,
    current.projectionGroupFullPrefillShapeSavedDispatches === 7 ? 1 : 0,
    current.projectionGroupSmollmPromptSpeedup / projectionSmollmPromptSpeedupFloor,
    projectionChainMaxAbsDiffCeil / Math.max(current.projectionGroupSmollmPromptMaxAbsDiff, Number.EPSILON),
    current.projectionGroupSmollmPromptShapeCommands === 1 ? 1 : 0,
    current.projectionGroupSmollmPromptShapeGroups === 1 ? 1 : 0,
    current.projectionGroupSmollmPromptShapeCoveredOps === 8 ? 1 : 0,
    current.projectionGroupSmollmPromptShapeSavedDispatches === 7 ? 1 : 0,
    current.projectionGroupRegionFullPrefillSpeedup / projectionGroupRegionSpeedupFloor,
    projectionChainMaxAbsDiffCeil / Math.max(current.projectionGroupRegionFullPrefillMaxAbsDiff, Number.EPSILON),
    current.projectionGroupRegionFullPrefillShapeCommands === 2 ? 1 : 0,
    current.projectionGroupRegionFullPrefillShapeGroups === 2 ? 1 : 0,
    current.projectionGroupRegionFullPrefillShapeCoveredOps === 14 ? 1 : 0,
    current.projectionGroupRegionFullPrefillShapeSavedDispatches === 12 ? 1 : 0,
    current.projectionGroupRegionFullPrefillRuntimeDispatches === 2 ? 1 : 0,
    current.projectionGroupRegionFullPrefillRuntimeCacheDispatches === 0 ? 1 : 0,
    current.projectionGroupRegionSmollmPromptSpeedup / projectionGroupRegionSpeedupFloor,
    projectionChainMaxAbsDiffCeil / Math.max(current.projectionGroupRegionSmollmPromptMaxAbsDiff, Number.EPSILON),
    current.projectionGroupRegionSmollmPromptShapeCommands === 2 ? 1 : 0,
    current.projectionGroupRegionSmollmPromptShapeGroups === 2 ? 1 : 0,
    current.projectionGroupRegionSmollmPromptShapeCoveredOps === 14 ? 1 : 0,
    current.projectionGroupRegionSmollmPromptShapeSavedDispatches === 12 ? 1 : 0,
    current.projectionGroupRegionSmollmPromptRuntimeDispatches === 2 ? 1 : 0,
    current.projectionGroupRegionSmollmPromptRuntimeCacheDispatches === 0 ? 1 : 0,
  );
}

function chooseBestFocusedQproj(attempts) {
  return attempts.reduce((acc, current) => {
    if (!acc) return current;
    return focusedQprojMargin(current) > focusedQprojMargin(acc) ? current : acc;
  }, null);
}

function selectedQprojAttemptSummary(attempt) {
  return {
    attempt: attempt.attempt,
    projectionChain: {
      prompt: {
        speedup: roundMetric(attempt.projectionChainSpeedup),
        maxAbsDiff: roundMetric(attempt.projectionChainMaxAbsDiff),
      },
      fullPrefill: {
        speedup: roundMetric(attempt.projectionChainFullPrefillSpeedup),
        maxAbsDiff: roundMetric(attempt.projectionChainFullPrefillMaxAbsDiff),
        candidateReady: attempt.projectionChainFullPrefillSpeedup >= projectionChainFullPrefillCandidateSpeedupFloor,
      },
      smollmPrompt: {
        speedup: roundMetric(attempt.projectionChainSmollmPromptSpeedup),
        maxAbsDiff: roundMetric(attempt.projectionChainSmollmPromptMaxAbsDiff),
      },
    },
    projectionGroup: {
      fullPrefill: {
        speedup: roundMetric(attempt.projectionGroupFullPrefillSpeedup),
        maxAbsDiff: roundMetric(attempt.projectionGroupFullPrefillMaxAbsDiff),
        shapeCommands: attempt.projectionGroupFullPrefillShapeCommands,
        shapeProjectionGroups: attempt.projectionGroupFullPrefillShapeGroups,
        shapeCoveredOps: attempt.projectionGroupFullPrefillShapeCoveredOps,
        shapeSavedDispatches: attempt.projectionGroupFullPrefillShapeSavedDispatches,
        runtimeProjectionGroupDispatches: attempt.projectionGroupFullPrefillRuntimeDispatches,
        runtimeProjectionCacheGroupDispatches: attempt.projectionGroupFullPrefillRuntimeCacheDispatches,
      },
      smollmPrompt: {
        speedup: roundMetric(attempt.projectionGroupSmollmPromptSpeedup),
        maxAbsDiff: roundMetric(attempt.projectionGroupSmollmPromptMaxAbsDiff),
        shapeCommands: attempt.projectionGroupSmollmPromptShapeCommands,
        shapeProjectionGroups: attempt.projectionGroupSmollmPromptShapeGroups,
        shapeCoveredOps: attempt.projectionGroupSmollmPromptShapeCoveredOps,
        shapeSavedDispatches: attempt.projectionGroupSmollmPromptShapeSavedDispatches,
        runtimeProjectionGroupDispatches: attempt.projectionGroupSmollmPromptRuntimeDispatches,
        runtimeProjectionCacheGroupDispatches: attempt.projectionGroupSmollmPromptRuntimeCacheDispatches,
      },
    },
    projectionGroupRegion: {
      fullPrefill: {
        speedup: roundMetric(attempt.projectionGroupRegionFullPrefillSpeedup),
        maxAbsDiff: roundMetric(attempt.projectionGroupRegionFullPrefillMaxAbsDiff),
        shapeCommands: attempt.projectionGroupRegionFullPrefillShapeCommands,
        shapeProjectionGroups: attempt.projectionGroupRegionFullPrefillShapeGroups,
        shapeCoveredOps: attempt.projectionGroupRegionFullPrefillShapeCoveredOps,
        shapeSavedDispatches: attempt.projectionGroupRegionFullPrefillShapeSavedDispatches,
        runtimeProjectionGroupDispatches: attempt.projectionGroupRegionFullPrefillRuntimeDispatches,
        runtimeProjectionCacheGroupDispatches: attempt.projectionGroupRegionFullPrefillRuntimeCacheDispatches,
      },
      smollmPrompt: {
        speedup: roundMetric(attempt.projectionGroupRegionSmollmPromptSpeedup),
        maxAbsDiff: roundMetric(attempt.projectionGroupRegionSmollmPromptMaxAbsDiff),
        shapeCommands: attempt.projectionGroupRegionSmollmPromptShapeCommands,
        shapeProjectionGroups: attempt.projectionGroupRegionSmollmPromptShapeGroups,
        shapeCoveredOps: attempt.projectionGroupRegionSmollmPromptShapeCoveredOps,
        shapeSavedDispatches: attempt.projectionGroupRegionSmollmPromptShapeSavedDispatches,
        runtimeProjectionGroupDispatches: attempt.projectionGroupRegionSmollmPromptRuntimeDispatches,
        runtimeProjectionCacheGroupDispatches: attempt.projectionGroupRegionSmollmPromptRuntimeCacheDispatches,
      },
    },
  };
}

function writeFocusedQprojArtifact(best, attempts, aggregate, line) {
  if (!writeArtifact) return null;
  mkdirSync(artifactDir, { recursive: true });
  const artifactPath = join(artifactDir, `frontier-qproj-${timestampForArtifact()}-${process.pid}.json`);
  const artifact = {
    schema: "zgml.frontier-qproj.v1",
    createdAt: new Date().toISOString(),
    command: {
      argv: process.argv,
      cwd: process.cwd(),
      build,
      frontierFilter,
    },
    platform: {
      node: process.version,
      platform: process.platform,
      arch: process.arch,
      cpus: os.cpus().length,
    },
    source: benchmarkBinaryMetadata({ root, binary: frontierBinary, build }),
    config: {
      maxAttempts,
      build,
      frontierFilter,
      projectionChainTileSpeedupFloor,
      projectionChainFullPrefillSpeedupFloor,
      projectionSmollmPromptSpeedupFloor,
      projectionGroupRegionSpeedupFloor,
      projectionChainMaxAbsDiffCeil,
    },
    kind: "qproj",
    status: aggregate.length === 0 ? "pass" : "fail",
    selectedAttempt: best.attempt,
    attempts: attempts.length,
    aggregateFailures: aggregate,
    selected: selectedQprojAttemptSummary(best),
    attemptSummaries: attempts.map(selectedQprojAttemptSummary),
    next: "semantic_sublayer_or_quantized_projection_chain",
    line,
  };
  writeFileSync(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
  return resolve(artifactPath);
}

function aggregateFocusedQprojFailures(attempts) {
  const failures = [];
  const speedAtLeast = (field, floor, label) => {
    const value = bestMax(attempts, field);
    if (!Number.isFinite(value) || value < floor) failures.push(`${label} best ${Number.isFinite(value) ? value.toFixed(2) : "n/a"}x < ${floor.toFixed(2)}x`);
  };
  const diffAtMost = (field, ceil, label) => {
    const value = bestMin(attempts, field);
    if (!Number.isFinite(value) || value > ceil) failures.push(`${label} best max_abs_diff ${Number.isFinite(value) ? value.toFixed(6) : "n/a"} > ${ceil.toFixed(6)}`);
  };
  speedAtLeast("projectionChainSpeedup", projectionChainTileSpeedupFloor, "projection_chain prompt tile");
  diffAtMost("projectionChainMaxAbsDiff", projectionChainMaxAbsDiffCeil, "projection_chain prompt");
  speedAtLeast("projectionChainFullPrefillSpeedup", projectionChainFullPrefillSpeedupFloor, "projection_chain full-prefill");
  diffAtMost("projectionChainFullPrefillMaxAbsDiff", projectionChainMaxAbsDiffCeil, "projection_chain full-prefill");
  speedAtLeast("projectionChainSmollmPromptSpeedup", projectionSmollmPromptSpeedupFloor, "projection_chain smollm-prompt");
  diffAtMost("projectionChainSmollmPromptMaxAbsDiff", projectionChainMaxAbsDiffCeil, "projection_chain smollm-prompt");
  diffAtMost("projectionGroupFullPrefillMaxAbsDiff", projectionChainMaxAbsDiffCeil, "projection_group full-prefill");
  if (!anyEquals(attempts, [
    ["projectionGroupFullPrefillShapeCommands", 1],
    ["projectionGroupFullPrefillShapeGroups", 1],
    ["projectionGroupFullPrefillShapeCoveredOps", 8],
    ["projectionGroupFullPrefillShapeSavedDispatches", 7],
  ])) failures.push("projection_group full-prefill shape profile did not match in any attempt");
  speedAtLeast("projectionGroupSmollmPromptSpeedup", projectionSmollmPromptSpeedupFloor, "projection_group smollm-prompt");
  diffAtMost("projectionGroupSmollmPromptMaxAbsDiff", projectionChainMaxAbsDiffCeil, "projection_group smollm-prompt");
  if (!anyEquals(attempts, [
    ["projectionGroupSmollmPromptShapeCommands", 1],
    ["projectionGroupSmollmPromptShapeGroups", 1],
    ["projectionGroupSmollmPromptShapeCoveredOps", 8],
    ["projectionGroupSmollmPromptShapeSavedDispatches", 7],
  ])) failures.push("projection_group smollm-prompt shape profile did not match in any attempt");
  speedAtLeast("projectionGroupRegionFullPrefillSpeedup", projectionGroupRegionSpeedupFloor, "projection_group region full-prefill");
  diffAtMost("projectionGroupRegionFullPrefillMaxAbsDiff", projectionChainMaxAbsDiffCeil, "projection_group region full-prefill");
  if (!anyEquals(attempts, [
    ["projectionGroupRegionFullPrefillShapeCommands", 2],
    ["projectionGroupRegionFullPrefillShapeGroups", 2],
    ["projectionGroupRegionFullPrefillShapeCoveredOps", 14],
    ["projectionGroupRegionFullPrefillShapeSavedDispatches", 12],
    ["projectionGroupRegionFullPrefillRuntimeDispatches", 2],
    ["projectionGroupRegionFullPrefillRuntimeCacheDispatches", 0],
  ])) failures.push("projection_group region full-prefill runtime profile did not match in any attempt");
  speedAtLeast("projectionGroupRegionSmollmPromptSpeedup", projectionGroupRegionSpeedupFloor, "projection_group region smollm-prompt");
  diffAtMost("projectionGroupRegionSmollmPromptMaxAbsDiff", projectionChainMaxAbsDiffCeil, "projection_group region smollm-prompt");
  if (!anyEquals(attempts, [
    ["projectionGroupRegionSmollmPromptShapeCommands", 2],
    ["projectionGroupRegionSmollmPromptShapeGroups", 2],
    ["projectionGroupRegionSmollmPromptShapeCoveredOps", 14],
    ["projectionGroupRegionSmollmPromptShapeSavedDispatches", 12],
    ["projectionGroupRegionSmollmPromptRuntimeDispatches", 2],
    ["projectionGroupRegionSmollmPromptRuntimeCacheDispatches", 0],
  ])) failures.push("projection_group region smollm-prompt runtime profile did not match in any attempt");
  return failures;
}

function runFocusedQprojGate() {
  const attempts = [];
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      attempts.push(scoreFocusedQproj(runBench(), attempt));
    } catch (err) {
      process.stderr.write(`${err.output ?? err.message}\n`);
      process.exit(1);
    }
  }
  const passing = attempts.filter((current) => current.failures.length === 0);
  const best = chooseBestFocusedQproj(passing.length > 0 ? passing : attempts);
  const aggregate = aggregateFocusedQprojFailures(attempts);
  const line = aggregate.length === 0 ? best.line.replace("frontier qproj gate: fail", "frontier qproj gate: pass") : best.line;
  const artifactPath = writeFocusedQprojArtifact(best, attempts, aggregate, line);
  if (artifactPath) {
    process.stdout.write(`FRONTIER_BENCH_JSON ${JSON.stringify({
      artifact: artifactPath,
      status: aggregate.length === 0 ? "pass" : "fail",
      kind: "qproj",
      selectedAttempt: best.attempt,
      attempts: attempts.length,
      projectionChainFullPrefill: `${best.projectionChainFullPrefillSpeedup.toFixed(2)}x`,
      projectionChainSmollmPrompt: `${best.projectionChainSmollmPromptSpeedup.toFixed(2)}x`,
      projectionGroupRegionFullPrefill: `${best.projectionGroupRegionFullPrefillSpeedup.toFixed(2)}x`,
      projectionGroupRegionSmollmPrompt: `${best.projectionGroupRegionSmollmPromptSpeedup.toFixed(2)}x`,
      next: "semantic_sublayer_or_quantized_projection_chain",
    })}\n`);
  }
  process.stdout.write(`${line}\n`);
  if (aggregate.length === 0 && passing.length === 0) {
    process.stdout.write(`frontier qproj aggregate: pass across ${attempts.length} noisy attempts\n`);
  }
  if (best.attempt > 1) {
    process.stdout.write(`frontier qproj retries: ${best.attempt - 1} noisy attempt(s) below best evidence\n`);
  }
  if (aggregate.length !== 0) {
    process.stderr.write(`${aggregate.join("; ")}\n`);
    process.exit(1);
  }
  process.exit(0);
}

function scoreFocusedQrowRegion(output, attempt) {
  const fullPrefillLabel = "qrow region full-prefill x7 m=128 n=512 k=512 projection_row_chain_two_phase_group";
  const smollmPromptLabel = "qrow region smollm-prompt x7 m=128 n=576 k=576 projection_row_chain_two_phase_group";
  const fullPrefillProfileLabel = `${fullPrefillLabel} dispatch_profile`;
  const smollmPromptProfileLabel = `${smollmPromptLabel} dispatch_profile`;

  const fullPrefillSpeedup = metric(output, fullPrefillLabel, "speedup");
  const fullPrefillMaxAbsDiff = metric(output, fullPrefillLabel, "max_abs_diff");
  const fullPrefillShapeCommands = metric(output, fullPrefillProfileLabel, "shape_commands");
  const fullPrefillShapeRowChains = metric(output, fullPrefillProfileLabel, "shape_projection_row_chains");
  const fullPrefillShapeCoveredOps = metric(output, fullPrefillProfileLabel, "shape_covered_ops");
  const fullPrefillShapeSavedDispatches = metric(output, fullPrefillProfileLabel, "shape_saved_dispatches");
  const fullPrefillRuntimeCommandDispatches = metric(output, fullPrefillProfileLabel, "runtime_command_dispatches");
  const fullPrefillTwoPhaseCount = metric(output, fullPrefillProfileLabel, "qmatmul_row_chain_tiled_two_phase_count");
  const fullPrefillSpilledElementwise = metric(output, fullPrefillProfileLabel, "qmatmul_row_chain_tiled_spilled_elementwise");
  const fullPrefillSpilledInput = metric(output, fullPrefillProfileLabel, "qmatmul_row_chain_tiled_spilled_input");
  const fullPrefillOutputSpills = metric(output, fullPrefillProfileLabel, "qmatmul_row_chain_tiled_output_spills");

  const smollmPromptSpeedup = metric(output, smollmPromptLabel, "speedup");
  const smollmPromptMaxAbsDiff = metric(output, smollmPromptLabel, "max_abs_diff");
  const smollmPromptShapeCommands = metric(output, smollmPromptProfileLabel, "shape_commands");
  const smollmPromptShapeRowChains = metric(output, smollmPromptProfileLabel, "shape_projection_row_chains");
  const smollmPromptShapeCoveredOps = metric(output, smollmPromptProfileLabel, "shape_covered_ops");
  const smollmPromptShapeSavedDispatches = metric(output, smollmPromptProfileLabel, "shape_saved_dispatches");
  const smollmPromptRuntimeCommandDispatches = metric(output, smollmPromptProfileLabel, "runtime_command_dispatches");
  const smollmPromptTwoPhaseCount = metric(output, smollmPromptProfileLabel, "qmatmul_row_chain_tiled_two_phase_count");
  const smollmPromptSpilledElementwise = metric(output, smollmPromptProfileLabel, "qmatmul_row_chain_tiled_spilled_elementwise");
  const smollmPromptSpilledInput = metric(output, smollmPromptProfileLabel, "qmatmul_row_chain_tiled_spilled_input");
  const smollmPromptOutputSpills = metric(output, smollmPromptProfileLabel, "qmatmul_row_chain_tiled_output_spills");

  const failures = [];
  if (fullPrefillMaxAbsDiff > projectionRowChainMaxAbsDiffCeil) failures.push(`qrow region full-prefill max_abs_diff ${fullPrefillMaxAbsDiff.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  if (smollmPromptMaxAbsDiff > projectionRowChainMaxAbsDiffCeil) failures.push(`qrow region smollm-prompt max_abs_diff ${smollmPromptMaxAbsDiff.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  if (fullPrefillShapeCommands !== 7 || fullPrefillShapeRowChains !== 7 || fullPrefillShapeCoveredOps !== 35 || fullPrefillShapeSavedDispatches !== 28 || fullPrefillRuntimeCommandDispatches !== 14 || fullPrefillTwoPhaseCount !== 7) {
    failures.push("qrow region full-prefill profile must stay shape_commands=7 shape_projection_row_chains=7 shape_covered_ops=35 shape_saved_dispatches=28 runtime_command_dispatches=14 two_phase_count=7");
  }
  if (smollmPromptShapeCommands !== 7 || smollmPromptShapeRowChains !== 7 || smollmPromptShapeCoveredOps !== 35 || smollmPromptShapeSavedDispatches !== 28 || smollmPromptRuntimeCommandDispatches !== 14 || smollmPromptTwoPhaseCount !== 7) {
    failures.push("qrow region smollm-prompt profile must stay shape_commands=7 shape_projection_row_chains=7 shape_covered_ops=35 shape_saved_dispatches=28 runtime_command_dispatches=14 two_phase_count=7");
  }

  const line = [
    `frontier qrow region gate: ${failures.length === 0 ? "pass" : "fail"}`,
    `attempt=${attempt}/${maxAttempts}`,
    `full_prefill=${fullPrefillSpeedup.toFixed(2)}x diagnostic_floor=not-yet max_abs_diff=${fullPrefillMaxAbsDiff.toFixed(6)} diff_ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)} shape_commands=${fullPrefillShapeCommands} runtime_command_dispatches=${fullPrefillRuntimeCommandDispatches} two_phase_count=${fullPrefillTwoPhaseCount} qmatmul_row_chain_tiled_spilled_elementwise=${fullPrefillSpilledElementwise} qmatmul_row_chain_tiled_spilled_input=${fullPrefillSpilledInput} qmatmul_row_chain_tiled_output_spills=${fullPrefillOutputSpills}`,
    `smollm_prompt=${smollmPromptSpeedup.toFixed(2)}x diagnostic_floor=not-yet max_abs_diff=${smollmPromptMaxAbsDiff.toFixed(6)} diff_ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)} shape_commands=${smollmPromptShapeCommands} runtime_command_dispatches=${smollmPromptRuntimeCommandDispatches} two_phase_count=${smollmPromptTwoPhaseCount} qmatmul_row_chain_tiled_spilled_elementwise=${smollmPromptSpilledElementwise} qmatmul_row_chain_tiled_spilled_input=${smollmPromptSpilledInput} qmatmul_row_chain_tiled_output_spills=${smollmPromptOutputSpills}`,
    `next=${projectionRowChainNextTarget}`,
  ].join("; ");

  return {
    attempt,
    fullPrefillSpeedup,
    fullPrefillMaxAbsDiff,
    fullPrefillShapeCommands,
    fullPrefillShapeRowChains,
    fullPrefillShapeCoveredOps,
    fullPrefillShapeSavedDispatches,
    fullPrefillRuntimeCommandDispatches,
    fullPrefillTwoPhaseCount,
    fullPrefillSpilledElementwise,
    fullPrefillSpilledInput,
    fullPrefillOutputSpills,
    smollmPromptSpeedup,
    smollmPromptMaxAbsDiff,
    smollmPromptShapeCommands,
    smollmPromptShapeRowChains,
    smollmPromptShapeCoveredOps,
    smollmPromptShapeSavedDispatches,
    smollmPromptRuntimeCommandDispatches,
    smollmPromptTwoPhaseCount,
    smollmPromptSpilledElementwise,
    smollmPromptSpilledInput,
    smollmPromptOutputSpills,
    failures,
    line,
  };
}

function focusedQrowRegionMargin(current) {
  const throughputScore = Math.min(current.fullPrefillSpeedup, current.smollmPromptSpeedup);
  return Math.min(
    projectionRowChainMaxAbsDiffCeil / Math.max(current.fullPrefillMaxAbsDiff, Number.EPSILON),
    projectionRowChainMaxAbsDiffCeil / Math.max(current.smollmPromptMaxAbsDiff, Number.EPSILON),
    current.fullPrefillShapeCommands === 7 ? 1 : 0,
    current.fullPrefillTwoPhaseCount === 7 ? 1 : 0,
    current.smollmPromptShapeCommands === 7 ? 1 : 0,
    current.smollmPromptTwoPhaseCount === 7 ? 1 : 0,
  ) + throughputScore;
}

function chooseBestFocusedQrowRegion(attempts) {
  return attempts.reduce((acc, current) => {
    if (!acc) return current;
    return focusedQrowRegionMargin(current) > focusedQrowRegionMargin(acc) ? current : acc;
  }, null);
}

function aggregateFocusedQrowRegionFailures(attempts) {
  const failures = [];
  const diffAtMost = (field, ceil, label) => {
    const value = bestMin(attempts, field);
    if (!Number.isFinite(value) || value > ceil) failures.push(`${label} best max_abs_diff ${Number.isFinite(value) ? value.toFixed(6) : "n/a"} > ${ceil.toFixed(6)}`);
  };
  const exactProfile = (fields, label) => {
    if (!anyEquals(attempts, fields)) failures.push(`${label} profile did not match in any attempt`);
  };
  diffAtMost("fullPrefillMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "qrow region full-prefill");
  diffAtMost("smollmPromptMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "qrow region smollm-prompt");
  exactProfile([
    ["fullPrefillShapeCommands", 7],
    ["fullPrefillShapeRowChains", 7],
    ["fullPrefillShapeCoveredOps", 35],
    ["fullPrefillShapeSavedDispatches", 28],
    ["fullPrefillRuntimeCommandDispatches", 14],
    ["fullPrefillTwoPhaseCount", 7],
  ], "qrow region full-prefill");
  exactProfile([
    ["smollmPromptShapeCommands", 7],
    ["smollmPromptShapeRowChains", 7],
    ["smollmPromptShapeCoveredOps", 35],
    ["smollmPromptShapeSavedDispatches", 28],
    ["smollmPromptRuntimeCommandDispatches", 14],
    ["smollmPromptTwoPhaseCount", 7],
  ], "qrow region smollm-prompt");
  return failures;
}

function runFocusedQrowRegionGate() {
  const attempts = [];
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      attempts.push(scoreFocusedQrowRegion(runBench(), attempt));
    } catch (err) {
      process.stderr.write(`${err.output ?? err.message}\n`);
      process.exit(1);
    }
  }
  const passing = attempts.filter((current) => current.failures.length === 0);
  const best = chooseBestFocusedQrowRegion(passing.length > 0 ? passing : attempts);
  const aggregate = aggregateFocusedQrowRegionFailures(attempts);
  const line = aggregate.length === 0 ? best.line.replace("frontier qrow region gate: fail", "frontier qrow region gate: pass") : best.line;
  process.stdout.write(`${line}\n`);
  if (aggregate.length === 0 && passing.length === 0) {
    process.stdout.write(`frontier qrow region aggregate: pass across ${attempts.length} noisy attempts\n`);
  }
  if (best.attempt > 1) {
    process.stdout.write(`frontier qrow region retries: ${best.attempt - 1} noisy attempt(s) below best evidence\n`);
  }
  if (aggregate.length !== 0) {
    process.stderr.write(`${aggregate.join("; ")}\n`);
    process.exit(1);
  }
  process.exit(0);
}

function scoreFocusedSemantic(output, attempt) {
  const fullPrefillLabel = "qsemantic full-prefill m=128 n=512 k=512 semantic command";
  const fullPrefillTwoPhaseLabel = "qsemantic full-prefill m=128 n=512 k=512 semantic pair_row_chain_two_phase";
  const fullPrefillSingleDispatchLabel = "qsemantic full-prefill m=128 n=512 k=512 semantic pair_row_chain_single_dispatch";
  const fullPrefillThroughputCandidateLabel = "qsemantic full-prefill m=128 n=512 k=512 semantic throughput_candidate";
  const smollmPromptLabel = "qsemantic smollm-prompt m=128 n=576 k=576 semantic command";
  const smollmPromptTwoPhaseLabel = "qsemantic smollm-prompt m=128 n=576 k=576 semantic pair_row_chain_two_phase";
  const smollmPromptSingleDispatchLabel = "qsemantic smollm-prompt m=128 n=576 k=576 semantic pair_row_chain_single_dispatch";
  const smollmPromptThroughputCandidateLabel = "qsemantic smollm-prompt m=128 n=576 k=576 semantic throughput_candidate";
  const fullPrefillProfileLabel = `${fullPrefillLabel} dispatch_profile`;
  const fullPrefillTwoPhaseProfileLabel = `${fullPrefillTwoPhaseLabel} dispatch_profile`;
  const fullPrefillSingleDispatchProfileLabel = `${fullPrefillSingleDispatchLabel} dispatch_profile`;
  const fullPrefillThroughputCandidateProfileLabel = `${fullPrefillThroughputCandidateLabel} dispatch_profile`;
  const fullPrefillTargetProfileLabel = "qsemantic full-prefill m=128 n=512 k=512 semantic target dispatch_profile";
  const smollmPromptProfileLabel = `${smollmPromptLabel} dispatch_profile`;
  const smollmPromptTwoPhaseProfileLabel = `${smollmPromptTwoPhaseLabel} dispatch_profile`;
  const smollmPromptSingleDispatchProfileLabel = `${smollmPromptSingleDispatchLabel} dispatch_profile`;
  const smollmPromptThroughputCandidateProfileLabel = `${smollmPromptThroughputCandidateLabel} dispatch_profile`;
  const smollmPromptTargetProfileLabel = "qsemantic smollm-prompt m=128 n=576 k=576 semantic target dispatch_profile";

  const fullPrefillSpeedup = metric(output, fullPrefillLabel, "speedup");
  const fullPrefillMaxAbsDiff = metric(output, fullPrefillLabel, "max_abs_diff");
  const fullPrefillShapeCommands = metric(output, fullPrefillProfileLabel, "shape_commands");
  const fullPrefillShapeSemantic = metric(output, fullPrefillProfileLabel, "shape_semantic_ffn_sublayers");
  const fullPrefillShapeRowChains = metric(output, fullPrefillProfileLabel, "shape_projection_row_chains");
  const fullPrefillShapeCoveredOps = metric(output, fullPrefillProfileLabel, "shape_covered_ops");
  const fullPrefillShapeSavedDispatches = metric(output, fullPrefillProfileLabel, "shape_saved_dispatches");
  const fullPrefillRuntimeDispatches = metric(output, fullPrefillProfileLabel, "runtime_backend_dispatches");
  const fullPrefillRuntimeRowChainDispatches = metric(output, fullPrefillProfileLabel, "runtime_projection_row_chain_dispatches");
  const fullPrefillRuntimeRowChainAttempts = metric(output, fullPrefillProfileLabel, "runtime_projection_row_chain_attempts");
  const fullPrefillRuntimeRowChainRefused = metric(output, fullPrefillProfileLabel, "runtime_projection_row_chain_refused");
  const fullPrefillRuntimeSemanticDispatches = metric(output, fullPrefillProfileLabel, "runtime_semantic_ffn_dispatches");
  const fullPrefillRuntimeRowChainTiled = metric(output, fullPrefillProfileLabel, "qmatmul_row_chain_tiled_count");
  const fullPrefillSemanticRowSerialCount = metric(output, fullPrefillProfileLabel, "semantic_ffn_sublayer_count");
  const fullPrefillTargetDispatches = metric(output, fullPrefillProfileLabel, "semantic_target_dispatches");
  const fullPrefillTargetShapeCommands = metric(output, fullPrefillTargetProfileLabel, "shape_commands");
  const fullPrefillTargetShapeSemantic = metric(output, fullPrefillTargetProfileLabel, "shape_semantic_ffn_sublayers");
  const fullPrefillTargetShapeCoveredOps = metric(output, fullPrefillTargetProfileLabel, "shape_covered_ops");
  const fullPrefillTargetShapeSavedDispatches = metric(output, fullPrefillTargetProfileLabel, "shape_saved_dispatches");
  const fullPrefillTargetSpeedup = metric(output, "qsemantic full-prefill m=128 n=512 k=512 semantic target", "speedup");
  const fullPrefillTargetMaxAbsDiff = metric(output, "qsemantic full-prefill m=128 n=512 k=512 semantic target", "max_abs_diff");
  const fullPrefillTargetRuntimeDispatches = metric(output, fullPrefillTargetProfileLabel, "runtime_backend_dispatches");
  const fullPrefillTargetSemanticCount = metric(output, fullPrefillTargetProfileLabel, "semantic_ffn_sublayer_count");
  const fullPrefillTargetSemanticRows = metric(output, fullPrefillTargetProfileLabel, "semantic_ffn_sublayer_rows");
  const fullPrefillTargetSemanticHidden = metric(output, fullPrefillTargetProfileLabel, "semantic_ffn_sublayer_hidden");
  const fullPrefillTargetSemanticInput = metric(output, fullPrefillTargetProfileLabel, "semantic_ffn_sublayer_input");
  const fullPrefillTargetSemanticOutput = metric(output, fullPrefillTargetProfileLabel, "semantic_ffn_sublayer_output");
  const fullPrefillTargetSemanticRowSerialDotOps = metric(output, fullPrefillTargetProfileLabel, "semantic_ffn_sublayer_row_serial_dot_ops");
  const fullPrefillTargetSemanticTotalRowSerialDotOps = metric(output, fullPrefillTargetProfileLabel, "semantic_ffn_sublayer_total_row_serial_dot_ops");
  const fullPrefillTargetSemanticTileRowGroups = metric(output, fullPrefillTargetProfileLabel, "semantic_ffn_sublayer_tile_row_groups");
  const fullPrefillTargetSemanticTileHiddenTiles = metric(output, fullPrefillTargetProfileLabel, "semantic_ffn_sublayer_tile_hidden_tiles");
  const fullPrefillTargetSemanticTileOutputTiles = metric(output, fullPrefillTargetProfileLabel, "semantic_ffn_sublayer_tile_output_tiles");
  const fullPrefillTargetSemanticTileParallelGroups = metric(output, fullPrefillTargetProfileLabel, "semantic_ffn_sublayer_tile_parallel_groups");
  const fullPrefillTargetSemanticRowSerialDotOpsPerTileGroup = metric(output, fullPrefillTargetProfileLabel, "semantic_ffn_sublayer_row_serial_dot_ops_per_tile_parallel_group");
  const fullPrefillTargetSemanticTotalRowSerialDotOpsPerTileGroup = metric(output, fullPrefillTargetProfileLabel, "semantic_ffn_sublayer_total_row_serial_dot_ops_per_tile_parallel_group");

  const fullPrefillTwoPhaseSpeedup = metric(output, fullPrefillTwoPhaseLabel, "speedup");
  const fullPrefillTwoPhaseMaxAbsDiff = metric(output, fullPrefillTwoPhaseLabel, "max_abs_diff");
  const fullPrefillTwoPhaseRuntimeDispatches = metric(output, fullPrefillTwoPhaseProfileLabel, "runtime_backend_dispatches");
  const fullPrefillTwoPhaseRuntimeRowChainTiled = metric(output, fullPrefillTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_count");
  const fullPrefillTwoPhaseRowTileGroups = metric(output, fullPrefillTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_row_tile_groups");
  const fullPrefillTwoPhaseNTiles = metric(output, fullPrefillTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_n_tiles");
  const fullPrefillTwoPhaseSerialLoops = metric(output, fullPrefillTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_serial_tile_loops");
  const fullPrefillTwoPhasePartialSlots = metric(output, fullPrefillTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_partial_slots");
  const fullPrefillTwoPhaseScratchCapacity = metric(output, fullPrefillTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_scratch_capacity");
  const fullPrefillTwoPhaseRuntimeRowChainTwoPhase = metric(output, fullPrefillTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_two_phase_count");
  const fullPrefillTwoPhaseFinalizeTileGroups = metric(output, fullPrefillTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_finalize_tile_groups");
  const fullPrefillTwoPhaseFinalizeElements = metric(output, fullPrefillTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_finalize_elements");
  const fullPrefillTwoPhaseSpilledInput = metric(output, fullPrefillTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_spilled_input");
  const fullPrefillTwoPhaseOutputSpills = metric(output, fullPrefillTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_output_spills");
  const fullPrefillSingleDispatchSpeedup = metric(output, fullPrefillSingleDispatchLabel, "speedup");
  const fullPrefillSingleDispatchMaxAbsDiff = metric(output, fullPrefillSingleDispatchLabel, "max_abs_diff");
  const fullPrefillSingleDispatchShapeCommands = metric(output, fullPrefillSingleDispatchProfileLabel, "shape_commands");
  const fullPrefillSingleDispatchShapeRowChains = metric(output, fullPrefillSingleDispatchProfileLabel, "shape_projection_row_chains");
  const fullPrefillSingleDispatchShapeCoveredOps = metric(output, fullPrefillSingleDispatchProfileLabel, "shape_covered_ops");
  const fullPrefillSingleDispatchRuntimeDispatches = metric(output, fullPrefillSingleDispatchProfileLabel, "runtime_backend_dispatches");
  const fullPrefillSingleDispatchRuntimeRowChainDispatches = metric(output, fullPrefillSingleDispatchProfileLabel, "runtime_projection_row_chain_dispatches");
  const fullPrefillSingleDispatchRuntimeRowChainAttempts = metric(output, fullPrefillSingleDispatchProfileLabel, "runtime_projection_row_chain_attempts");
  const fullPrefillSingleDispatchRuntimeRowChainRefused = metric(output, fullPrefillSingleDispatchProfileLabel, "runtime_projection_row_chain_refused");
  const fullPrefillSingleDispatchRuntimeRowChainTiled = metric(output, fullPrefillSingleDispatchProfileLabel, "qmatmul_row_chain_tiled_count");
  const fullPrefillSingleDispatchRowTileGroups = metric(output, fullPrefillSingleDispatchProfileLabel, "qmatmul_row_chain_tiled_row_tile_groups");
  const fullPrefillSingleDispatchNTiles = metric(output, fullPrefillSingleDispatchProfileLabel, "qmatmul_row_chain_tiled_n_tiles");
  const fullPrefillSingleDispatchSerialLoops = metric(output, fullPrefillSingleDispatchProfileLabel, "qmatmul_row_chain_tiled_serial_tile_loops");
  const fullPrefillSingleDispatchPartialSlots = metric(output, fullPrefillSingleDispatchProfileLabel, "qmatmul_row_chain_tiled_partial_slots");
  const fullPrefillSingleDispatchScratchCapacity = metric(output, fullPrefillSingleDispatchProfileLabel, "qmatmul_row_chain_tiled_scratch_capacity");
  const fullPrefillThroughputCandidateSpeedup = metric(output, fullPrefillThroughputCandidateLabel, "speedup");
  const fullPrefillThroughputCandidateMaxAbsDiff = metric(output, fullPrefillThroughputCandidateLabel, "max_abs_diff");
  const fullPrefillThroughputCandidateShapeCommands = metric(output, fullPrefillThroughputCandidateProfileLabel, "shape_commands");
  const fullPrefillThroughputCandidateShapeSemantic = metric(output, fullPrefillThroughputCandidateProfileLabel, "shape_semantic_ffn_sublayers");
  const fullPrefillThroughputCandidateRuntimeDispatches = metric(output, fullPrefillThroughputCandidateProfileLabel, "runtime_backend_dispatches");
  const fullPrefillThroughputCandidateRuntimeSemanticDispatches = metric(output, fullPrefillThroughputCandidateProfileLabel, "runtime_semantic_ffn_dispatches");
  const fullPrefillThroughputCandidateRuntimeRowChainTiled = metric(output, fullPrefillThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_count");
  const fullPrefillThroughputCandidateRowTileGroups = metric(output, fullPrefillThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_row_tile_groups");
  const fullPrefillThroughputCandidateNTiles = metric(output, fullPrefillThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_n_tiles");
  const fullPrefillThroughputCandidateTwoPhase = metric(output, fullPrefillThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_two_phase_count");
  const fullPrefillThroughputCandidateFinalizeTileGroups = metric(output, fullPrefillThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_finalize_tile_groups");
  const fullPrefillThroughputCandidateFinalizeElements = metric(output, fullPrefillThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_finalize_elements");
  const fullPrefillThroughputCandidateSpilledInput = metric(output, fullPrefillThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_spilled_input");
  const fullPrefillThroughputCandidateOutputSpills = metric(output, fullPrefillThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_output_spills");
  const smollmPromptSpeedup = metric(output, smollmPromptLabel, "speedup");
  const smollmPromptMaxAbsDiff = metric(output, smollmPromptLabel, "max_abs_diff");
  const smollmPromptShapeCommands = metric(output, smollmPromptProfileLabel, "shape_commands");
  const smollmPromptShapeSemantic = metric(output, smollmPromptProfileLabel, "shape_semantic_ffn_sublayers");
  const smollmPromptShapeRowChains = metric(output, smollmPromptProfileLabel, "shape_projection_row_chains");
  const smollmPromptShapeCoveredOps = metric(output, smollmPromptProfileLabel, "shape_covered_ops");
  const smollmPromptShapeSavedDispatches = metric(output, smollmPromptProfileLabel, "shape_saved_dispatches");
  const smollmPromptRuntimeDispatches = metric(output, smollmPromptProfileLabel, "runtime_backend_dispatches");
  const smollmPromptRuntimeRowChainDispatches = metric(output, smollmPromptProfileLabel, "runtime_projection_row_chain_dispatches");
  const smollmPromptRuntimeRowChainAttempts = metric(output, smollmPromptProfileLabel, "runtime_projection_row_chain_attempts");
  const smollmPromptRuntimeRowChainRefused = metric(output, smollmPromptProfileLabel, "runtime_projection_row_chain_refused");
  const smollmPromptRuntimeSemanticDispatches = metric(output, smollmPromptProfileLabel, "runtime_semantic_ffn_dispatches");
  const smollmPromptRuntimeRowChainTiled = metric(output, smollmPromptProfileLabel, "qmatmul_row_chain_tiled_count");
  const smollmPromptSemanticRowSerialCount = metric(output, smollmPromptProfileLabel, "semantic_ffn_sublayer_count");
  const smollmPromptTargetDispatches = metric(output, smollmPromptProfileLabel, "semantic_target_dispatches");
  const smollmPromptTargetShapeCommands = metric(output, smollmPromptTargetProfileLabel, "shape_commands");
  const smollmPromptTargetShapeSemantic = metric(output, smollmPromptTargetProfileLabel, "shape_semantic_ffn_sublayers");
  const smollmPromptTargetShapeCoveredOps = metric(output, smollmPromptTargetProfileLabel, "shape_covered_ops");
  const smollmPromptTargetShapeSavedDispatches = metric(output, smollmPromptTargetProfileLabel, "shape_saved_dispatches");
  const smollmPromptTargetSpeedup = metric(output, "qsemantic smollm-prompt m=128 n=576 k=576 semantic target", "speedup");
  const smollmPromptTargetMaxAbsDiff = metric(output, "qsemantic smollm-prompt m=128 n=576 k=576 semantic target", "max_abs_diff");
  const smollmPromptTargetRuntimeDispatches = metric(output, smollmPromptTargetProfileLabel, "runtime_backend_dispatches");
  const smollmPromptTargetSemanticCount = metric(output, smollmPromptTargetProfileLabel, "semantic_ffn_sublayer_count");
  const smollmPromptTargetSemanticRows = metric(output, smollmPromptTargetProfileLabel, "semantic_ffn_sublayer_rows");
  const smollmPromptTargetSemanticHidden = metric(output, smollmPromptTargetProfileLabel, "semantic_ffn_sublayer_hidden");
  const smollmPromptTargetSemanticInput = metric(output, smollmPromptTargetProfileLabel, "semantic_ffn_sublayer_input");
  const smollmPromptTargetSemanticOutput = metric(output, smollmPromptTargetProfileLabel, "semantic_ffn_sublayer_output");
  const smollmPromptTargetSemanticRowSerialDotOps = metric(output, smollmPromptTargetProfileLabel, "semantic_ffn_sublayer_row_serial_dot_ops");
  const smollmPromptTargetSemanticTotalRowSerialDotOps = metric(output, smollmPromptTargetProfileLabel, "semantic_ffn_sublayer_total_row_serial_dot_ops");
  const smollmPromptTargetSemanticTileRowGroups = metric(output, smollmPromptTargetProfileLabel, "semantic_ffn_sublayer_tile_row_groups");
  const smollmPromptTargetSemanticTileHiddenTiles = metric(output, smollmPromptTargetProfileLabel, "semantic_ffn_sublayer_tile_hidden_tiles");
  const smollmPromptTargetSemanticTileOutputTiles = metric(output, smollmPromptTargetProfileLabel, "semantic_ffn_sublayer_tile_output_tiles");
  const smollmPromptTargetSemanticTileParallelGroups = metric(output, smollmPromptTargetProfileLabel, "semantic_ffn_sublayer_tile_parallel_groups");
  const smollmPromptTargetSemanticRowSerialDotOpsPerTileGroup = metric(output, smollmPromptTargetProfileLabel, "semantic_ffn_sublayer_row_serial_dot_ops_per_tile_parallel_group");
  const smollmPromptTargetSemanticTotalRowSerialDotOpsPerTileGroup = metric(output, smollmPromptTargetProfileLabel, "semantic_ffn_sublayer_total_row_serial_dot_ops_per_tile_parallel_group");

  const smollmPromptTwoPhaseSpeedup = metric(output, smollmPromptTwoPhaseLabel, "speedup");
  const smollmPromptTwoPhaseMaxAbsDiff = metric(output, smollmPromptTwoPhaseLabel, "max_abs_diff");
  const smollmPromptTwoPhaseRuntimeDispatches = metric(output, smollmPromptTwoPhaseProfileLabel, "runtime_backend_dispatches");
  const smollmPromptTwoPhaseRuntimeRowChainTiled = metric(output, smollmPromptTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_count");
  const smollmPromptTwoPhaseRowTileGroups = metric(output, smollmPromptTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_row_tile_groups");
  const smollmPromptTwoPhaseNTiles = metric(output, smollmPromptTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_n_tiles");
  const smollmPromptTwoPhaseSerialLoops = metric(output, smollmPromptTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_serial_tile_loops");
  const smollmPromptTwoPhasePartialSlots = metric(output, smollmPromptTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_partial_slots");
  const smollmPromptTwoPhaseScratchCapacity = metric(output, smollmPromptTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_scratch_capacity");
  const smollmPromptTwoPhaseRuntimeRowChainTwoPhase = metric(output, smollmPromptTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_two_phase_count");
  const smollmPromptTwoPhaseFinalizeTileGroups = metric(output, smollmPromptTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_finalize_tile_groups");
  const smollmPromptTwoPhaseFinalizeElements = metric(output, smollmPromptTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_finalize_elements");
  const smollmPromptTwoPhaseSpilledInput = metric(output, smollmPromptTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_spilled_input");
  const smollmPromptTwoPhaseOutputSpills = metric(output, smollmPromptTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_output_spills");
  const smollmPromptSingleDispatchSpeedup = metric(output, smollmPromptSingleDispatchLabel, "speedup");
  const smollmPromptSingleDispatchMaxAbsDiff = metric(output, smollmPromptSingleDispatchLabel, "max_abs_diff");
  const smollmPromptSingleDispatchShapeCommands = metric(output, smollmPromptSingleDispatchProfileLabel, "shape_commands");
  const smollmPromptSingleDispatchShapeRowChains = metric(output, smollmPromptSingleDispatchProfileLabel, "shape_projection_row_chains");
  const smollmPromptSingleDispatchShapeCoveredOps = metric(output, smollmPromptSingleDispatchProfileLabel, "shape_covered_ops");
  const smollmPromptSingleDispatchRuntimeDispatches = metric(output, smollmPromptSingleDispatchProfileLabel, "runtime_backend_dispatches");
  const smollmPromptSingleDispatchRuntimeRowChainDispatches = metric(output, smollmPromptSingleDispatchProfileLabel, "runtime_projection_row_chain_dispatches");
  const smollmPromptSingleDispatchRuntimeRowChainAttempts = metric(output, smollmPromptSingleDispatchProfileLabel, "runtime_projection_row_chain_attempts");
  const smollmPromptSingleDispatchRuntimeRowChainRefused = metric(output, smollmPromptSingleDispatchProfileLabel, "runtime_projection_row_chain_refused");
  const smollmPromptSingleDispatchRuntimeRowChainTiled = metric(output, smollmPromptSingleDispatchProfileLabel, "qmatmul_row_chain_tiled_count");
  const smollmPromptSingleDispatchRowTileGroups = metric(output, smollmPromptSingleDispatchProfileLabel, "qmatmul_row_chain_tiled_row_tile_groups");
  const smollmPromptSingleDispatchNTiles = metric(output, smollmPromptSingleDispatchProfileLabel, "qmatmul_row_chain_tiled_n_tiles");
  const smollmPromptSingleDispatchSerialLoops = metric(output, smollmPromptSingleDispatchProfileLabel, "qmatmul_row_chain_tiled_serial_tile_loops");
  const smollmPromptSingleDispatchPartialSlots = metric(output, smollmPromptSingleDispatchProfileLabel, "qmatmul_row_chain_tiled_partial_slots");
  const smollmPromptSingleDispatchScratchCapacity = metric(output, smollmPromptSingleDispatchProfileLabel, "qmatmul_row_chain_tiled_scratch_capacity");
  const smollmPromptThroughputCandidateSpeedup = metric(output, smollmPromptThroughputCandidateLabel, "speedup");
  const smollmPromptThroughputCandidateMaxAbsDiff = metric(output, smollmPromptThroughputCandidateLabel, "max_abs_diff");
  const smollmPromptThroughputCandidateShapeCommands = metric(output, smollmPromptThroughputCandidateProfileLabel, "shape_commands");
  const smollmPromptThroughputCandidateShapeSemantic = metric(output, smollmPromptThroughputCandidateProfileLabel, "shape_semantic_ffn_sublayers");
  const smollmPromptThroughputCandidateRuntimeDispatches = metric(output, smollmPromptThroughputCandidateProfileLabel, "runtime_backend_dispatches");
  const smollmPromptThroughputCandidateRuntimeSemanticDispatches = metric(output, smollmPromptThroughputCandidateProfileLabel, "runtime_semantic_ffn_dispatches");
  const smollmPromptThroughputCandidateRuntimeRowChainTiled = metric(output, smollmPromptThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_count");
  const smollmPromptThroughputCandidateRowTileGroups = metric(output, smollmPromptThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_row_tile_groups");
  const smollmPromptThroughputCandidateNTiles = metric(output, smollmPromptThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_n_tiles");
  const smollmPromptThroughputCandidateTwoPhase = metric(output, smollmPromptThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_two_phase_count");
  const smollmPromptThroughputCandidateFinalizeTileGroups = metric(output, smollmPromptThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_finalize_tile_groups");
  const smollmPromptThroughputCandidateFinalizeElements = metric(output, smollmPromptThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_finalize_elements");
  const smollmPromptThroughputCandidateSpilledInput = metric(output, smollmPromptThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_spilled_input");
  const smollmPromptThroughputCandidateOutputSpills = metric(output, smollmPromptThroughputCandidateProfileLabel, "qmatmul_row_chain_tiled_output_spills");
  const targetThroughputStatus = smollmPromptTargetSpeedup >= smollmPromptSpeedup && fullPrefillTargetSpeedup >= fullPrefillSpeedup
    ? "ready"
    : "diagnostic_needs_throughput_kernel";
  const throughputCandidateStatus = smollmPromptThroughputCandidateSpeedup >= smollmPromptSpeedup && fullPrefillThroughputCandidateSpeedup >= fullPrefillSpeedup
    ? "ready"
    : "mixed_tiled_tail_diagnostic";
  const fullPrefillThroughputCandidateVsDefault = fullPrefillThroughputCandidateSpeedup / fullPrefillSpeedup;
  const smollmPromptThroughputCandidateVsDefault = smollmPromptThroughputCandidateSpeedup / smollmPromptSpeedup;
  const fullPrefillThroughputCandidateVsTwoPhase = fullPrefillThroughputCandidateSpeedup / fullPrefillTwoPhaseSpeedup;
  const smollmPromptThroughputCandidateVsTwoPhase = smollmPromptThroughputCandidateSpeedup / smollmPromptTwoPhaseSpeedup;
  const fullPrefillTargetVsDefault = fullPrefillTargetSpeedup / fullPrefillSpeedup;
  const smollmPromptTargetVsDefault = smollmPromptTargetSpeedup / smollmPromptSpeedup;
  const semanticCommandStatus =
    fullPrefillShapeCommands === 1 &&
    fullPrefillShapeSemantic === 1 &&
    fullPrefillRuntimeDispatches === 3 &&
    fullPrefillRuntimeSemanticDispatches === 3 &&
    fullPrefillSemanticRowSerialCount === 0 &&
    smollmPromptShapeCommands === 1 &&
    smollmPromptShapeSemantic === 1 &&
    smollmPromptRuntimeDispatches === 3 &&
    smollmPromptRuntimeSemanticDispatches === 3 &&
    smollmPromptSemanticRowSerialCount === 0
      ? "preserves_default_work_shape"
      : "diagnostic";
  const singleDispatchReduced = fullPrefillSingleDispatchRuntimeDispatches < fullPrefillRuntimeDispatches && smollmPromptSingleDispatchRuntimeDispatches < smollmPromptRuntimeDispatches;
  const singleDispatchBlocker = singleDispatchReduced ? "none" : "metal_row_chain_leaf_encoder_declined_semantic_shape";
  const singleDispatchThroughputStatus = singleDispatchReduced && fullPrefillSingleDispatchSpeedup >= fullPrefillSpeedup && smollmPromptSingleDispatchSpeedup >= smollmPromptSpeedup
    ? "ready"
    : "dispatch_reduced_but_throughput_diagnostic";

  const failures = [];
  for (const [label, value] of [
    ["semantic full-prefill", fullPrefillMaxAbsDiff],
    ["semantic full-prefill two-phase", fullPrefillTwoPhaseMaxAbsDiff],
    ["semantic full-prefill single-dispatch row-chain", fullPrefillSingleDispatchMaxAbsDiff],
    ["semantic full-prefill throughput candidate", fullPrefillThroughputCandidateMaxAbsDiff],
    ["semantic full-prefill target", fullPrefillTargetMaxAbsDiff],
    ["semantic smollm-prompt", smollmPromptMaxAbsDiff],
    ["semantic smollm-prompt two-phase", smollmPromptTwoPhaseMaxAbsDiff],
    ["semantic smollm-prompt single-dispatch row-chain", smollmPromptSingleDispatchMaxAbsDiff],
    ["semantic smollm-prompt throughput candidate", smollmPromptThroughputCandidateMaxAbsDiff],
    ["semantic smollm-prompt target", smollmPromptTargetMaxAbsDiff],
  ]) {
    if (value > projectionRowChainMaxAbsDiffCeil) failures.push(`${label} max_abs_diff ${value.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  }
  if (fullPrefillShapeCommands !== 1 || fullPrefillShapeSemantic !== 1 || fullPrefillShapeRowChains !== 0 || fullPrefillShapeCoveredOps !== 9 || fullPrefillShapeSavedDispatches !== 8 || fullPrefillRuntimeDispatches !== 3 || fullPrefillRuntimeSemanticDispatches !== 3 || fullPrefillRuntimeRowChainDispatches !== 0 || fullPrefillRuntimeRowChainAttempts !== 0 || fullPrefillRuntimeRowChainRefused !== 0 || fullPrefillRuntimeRowChainTiled !== 0 || fullPrefillSemanticRowSerialCount !== 0 || fullPrefillTargetDispatches !== 1) {
    failures.push("semantic full-prefill command profile must stay shape_commands=1 shape_semantic_ffn_sublayers=1 shape_projection_row_chains=0 shape_covered_ops=9 shape_saved_dispatches=8 runtime_backend_dispatches=3 runtime_semantic_ffn_dispatches=3 runtime_projection_row_chain_dispatches=0 runtime_projection_row_chain_attempts=0 runtime_projection_row_chain_refused=0 qmatmul_row_chain_tiled_count=0 semantic_ffn_sublayer_count=0 semantic_target_dispatches=1");
  }
  if (fullPrefillTargetShapeCommands !== 1 || fullPrefillTargetShapeSemantic !== 1 || fullPrefillTargetShapeCoveredOps !== 9 || fullPrefillTargetShapeSavedDispatches !== 8 || fullPrefillTargetRuntimeDispatches !== 1) {
    failures.push("semantic full-prefill target profile must stay shape_commands=1 shape_semantic_ffn_sublayers=1 shape_covered_ops=9 shape_saved_dispatches=8 runtime_backend_dispatches=1");
  }
  if (fullPrefillTargetSemanticCount !== 1 || fullPrefillTargetSemanticRows !== 128 || fullPrefillTargetSemanticHidden !== 512 || fullPrefillTargetSemanticInput !== 512 || fullPrefillTargetSemanticOutput !== 512 || fullPrefillTargetSemanticRowSerialDotOps !== 786432 || fullPrefillTargetSemanticTotalRowSerialDotOps !== 100663296) {
    failures.push("semantic full-prefill target must expose count=1 rows=128 hidden=512 input=512 output=512 row_serial_dot_ops=786432 total_row_serial_dot_ops=100663296 until the throughput kernel replaces the row-serial diagnostic");
  }
  if (fullPrefillTargetSemanticTileRowGroups !== 4 || fullPrefillTargetSemanticTileHiddenTiles !== 16 || fullPrefillTargetSemanticTileOutputTiles !== 16 || fullPrefillTargetSemanticTileParallelGroups !== 192) {
    failures.push("semantic full-prefill target must expose tiled throughput shape row_groups=4 hidden_tiles=16 output_tiles=16 tile_parallel_groups=192");
  }
  if (fullPrefillTargetSemanticRowSerialDotOpsPerTileGroup !== 4096 || fullPrefillTargetSemanticTotalRowSerialDotOpsPerTileGroup !== 524288) {
    failures.push("semantic full-prefill target must expose row-serial throughput gap row_serial_dot_ops_per_tile_parallel_group=4096 total_row_serial_dot_ops_per_tile_parallel_group=524288");
  }
  if (fullPrefillSingleDispatchShapeCommands !== 2 || fullPrefillSingleDispatchShapeRowChains !== 1 || fullPrefillSingleDispatchShapeCoveredOps !== 9 || fullPrefillSingleDispatchRuntimeDispatches !== 2 || fullPrefillSingleDispatchRuntimeRowChainDispatches !== 1 || fullPrefillSingleDispatchRuntimeRowChainAttempts !== 1 || fullPrefillSingleDispatchRuntimeRowChainRefused !== 0 || fullPrefillSingleDispatchRuntimeRowChainTiled !== 1) {
    failures.push("semantic full-prefill single-dispatch row-chain diagnostic must stay shape_commands=2 shape_projection_row_chains=1 shape_covered_ops=9 runtime_backend_dispatches=2 runtime_projection_row_chain_dispatches=1 runtime_projection_row_chain_attempts=1 runtime_projection_row_chain_refused=0 qmatmul_row_chain_tiled_count=1 until throughput is ready");
  }
  if (fullPrefillTwoPhaseRowTileGroups !== 4 || fullPrefillTwoPhaseNTiles !== 16 || fullPrefillTwoPhaseSerialLoops !== 64 || fullPrefillTwoPhasePartialSlots !== 2048 || fullPrefillTwoPhaseScratchCapacity !== 65536 || fullPrefillSingleDispatchRowTileGroups !== 4 || fullPrefillSingleDispatchNTiles !== 16 || fullPrefillSingleDispatchSerialLoops !== 64 || fullPrefillSingleDispatchPartialSlots !== 2048 || fullPrefillSingleDispatchScratchCapacity !== 65536) {
    failures.push("semantic full-prefill tiled row-chain diagnostics must expose row_tile_groups=4 n_tiles=16 serial_tile_loops=64 partial_slots=2048 scratch_capacity=65536 for both two-phase and single-dispatch paths");
  }
  if (fullPrefillThroughputCandidateShapeCommands !== 1 || fullPrefillThroughputCandidateShapeSemantic !== 1 || fullPrefillThroughputCandidateRuntimeDispatches !== 3 || fullPrefillThroughputCandidateRuntimeSemanticDispatches !== 3 || fullPrefillThroughputCandidateRuntimeRowChainTiled !== 1 || fullPrefillThroughputCandidateRowTileGroups !== 4 || fullPrefillThroughputCandidateNTiles !== 16 || fullPrefillThroughputCandidateTwoPhase !== 1) {
    failures.push("semantic full-prefill throughput candidate must preserve semantic command shape while using one two-phase tiled row-chain tail");
  }
  if (smollmPromptShapeCommands !== 1 || smollmPromptShapeSemantic !== 1 || smollmPromptShapeRowChains !== 0 || smollmPromptShapeCoveredOps !== 9 || smollmPromptShapeSavedDispatches !== 8 || smollmPromptRuntimeDispatches !== 3 || smollmPromptRuntimeSemanticDispatches !== 3 || smollmPromptRuntimeRowChainDispatches !== 0 || smollmPromptRuntimeRowChainAttempts !== 0 || smollmPromptRuntimeRowChainRefused !== 0 || smollmPromptRuntimeRowChainTiled !== 0 || smollmPromptSemanticRowSerialCount !== 0 || smollmPromptTargetDispatches !== 1) {
    failures.push("semantic smollm-prompt command profile must stay shape_commands=1 shape_semantic_ffn_sublayers=1 shape_projection_row_chains=0 shape_covered_ops=9 shape_saved_dispatches=8 runtime_backend_dispatches=3 runtime_semantic_ffn_dispatches=3 runtime_projection_row_chain_dispatches=0 runtime_projection_row_chain_attempts=0 runtime_projection_row_chain_refused=0 qmatmul_row_chain_tiled_count=0 semantic_ffn_sublayer_count=0 semantic_target_dispatches=1");
  }
  if (smollmPromptTargetShapeCommands !== 1 || smollmPromptTargetShapeSemantic !== 1 || smollmPromptTargetShapeCoveredOps !== 9 || smollmPromptTargetShapeSavedDispatches !== 8 || smollmPromptTargetRuntimeDispatches !== 1) {
    failures.push("semantic smollm-prompt target profile must stay shape_commands=1 shape_semantic_ffn_sublayers=1 shape_covered_ops=9 shape_saved_dispatches=8 runtime_backend_dispatches=1");
  }
  if (smollmPromptTargetSemanticCount !== 1 || smollmPromptTargetSemanticRows !== 128 || smollmPromptTargetSemanticHidden !== 576 || smollmPromptTargetSemanticInput !== 576 || smollmPromptTargetSemanticOutput !== 576 || smollmPromptTargetSemanticRowSerialDotOps !== 995328 || smollmPromptTargetSemanticTotalRowSerialDotOps !== 127401984) {
    failures.push("semantic smollm-prompt target must expose count=1 rows=128 hidden=576 input=576 output=576 row_serial_dot_ops=995328 total_row_serial_dot_ops=127401984 until the throughput kernel replaces the row-serial diagnostic");
  }
  if (smollmPromptTargetSemanticTileRowGroups !== 4 || smollmPromptTargetSemanticTileHiddenTiles !== 18 || smollmPromptTargetSemanticTileOutputTiles !== 18 || smollmPromptTargetSemanticTileParallelGroups !== 216) {
    failures.push("semantic smollm-prompt target must expose tiled throughput shape row_groups=4 hidden_tiles=18 output_tiles=18 tile_parallel_groups=216");
  }
  if (smollmPromptTargetSemanticRowSerialDotOpsPerTileGroup !== 4608 || smollmPromptTargetSemanticTotalRowSerialDotOpsPerTileGroup !== 589824) {
    failures.push("semantic smollm-prompt target must expose row-serial throughput gap row_serial_dot_ops_per_tile_parallel_group=4608 total_row_serial_dot_ops_per_tile_parallel_group=589824");
  }
  if (smollmPromptSingleDispatchShapeCommands !== 2 || smollmPromptSingleDispatchShapeRowChains !== 1 || smollmPromptSingleDispatchShapeCoveredOps !== 9 || smollmPromptSingleDispatchRuntimeDispatches !== 2 || smollmPromptSingleDispatchRuntimeRowChainDispatches !== 1 || smollmPromptSingleDispatchRuntimeRowChainAttempts !== 1 || smollmPromptSingleDispatchRuntimeRowChainRefused !== 0 || smollmPromptSingleDispatchRuntimeRowChainTiled !== 1) {
    failures.push("semantic smollm-prompt single-dispatch row-chain diagnostic must stay shape_commands=2 shape_projection_row_chains=1 shape_covered_ops=9 runtime_backend_dispatches=2 runtime_projection_row_chain_dispatches=1 runtime_projection_row_chain_attempts=1 runtime_projection_row_chain_refused=0 qmatmul_row_chain_tiled_count=1 until throughput is ready");
  }
  if (smollmPromptTwoPhaseRowTileGroups !== 4 || smollmPromptTwoPhaseNTiles !== 18 || smollmPromptTwoPhaseSerialLoops !== 72 || smollmPromptTwoPhasePartialSlots !== 2304 || smollmPromptTwoPhaseScratchCapacity !== 73728 || smollmPromptSingleDispatchRowTileGroups !== 4 || smollmPromptSingleDispatchNTiles !== 18 || smollmPromptSingleDispatchSerialLoops !== 72 || smollmPromptSingleDispatchPartialSlots !== 2304 || smollmPromptSingleDispatchScratchCapacity !== 73728) {
    failures.push("semantic smollm-prompt tiled row-chain diagnostics must expose row_tile_groups=4 n_tiles=18 serial_tile_loops=72 partial_slots=2304 scratch_capacity=73728 for both two-phase and single-dispatch paths");
  }
  if (smollmPromptThroughputCandidateShapeCommands !== 1 || smollmPromptThroughputCandidateShapeSemantic !== 1 || smollmPromptThroughputCandidateRuntimeDispatches !== 3 || smollmPromptThroughputCandidateRuntimeSemanticDispatches !== 3 || smollmPromptThroughputCandidateRuntimeRowChainTiled !== 1 || smollmPromptThroughputCandidateRowTileGroups !== 4 || smollmPromptThroughputCandidateNTiles !== 18 || smollmPromptThroughputCandidateTwoPhase !== 1) {
    failures.push("semantic smollm-prompt throughput candidate must preserve semantic command shape while using one two-phase tiled row-chain tail");
  }

  const line = [
    `frontier qsemantic gate: ${failures.length === 0 ? "pass" : "fail"}`,
    `attempt=${attempt}/${maxAttempts}`,
    `target_throughput_status=${targetThroughputStatus}`,
    `throughput_candidate_status=${throughputCandidateStatus}`,
    `throughput_candidate_vs_default=full_prefill:${fullPrefillThroughputCandidateVsDefault.toFixed(2)}x,smollm_prompt:${smollmPromptThroughputCandidateVsDefault.toFixed(2)}x`,
    `throughput_candidate_vs_two_phase=full_prefill:${fullPrefillThroughputCandidateVsTwoPhase.toFixed(2)}x,smollm_prompt:${smollmPromptThroughputCandidateVsTwoPhase.toFixed(2)}x`,
    `target_vs_default=full_prefill:${fullPrefillTargetVsDefault.toFixed(2)}x,smollm_prompt:${smollmPromptTargetVsDefault.toFixed(2)}x`,
    semanticCommandStatus === "preserves_default_work_shape"
      ? semanticCommandStatusToken
      : `semantic_command_status=${semanticCommandStatus}`,
    `single_dispatch_row_chain_dispatch_reduced=${singleDispatchReduced ? "yes" : "no"}`,
    `single_dispatch_row_chain_blocker=${singleDispatchBlocker}`,
    `single_dispatch_throughput_status=${singleDispatchThroughputStatus}`,
    `full_prefill=${fullPrefillSpeedup.toFixed(2)}x max_abs_diff=${fullPrefillMaxAbsDiff.toFixed(6)} shape_commands=${fullPrefillShapeCommands} shape_semantic_ffn_sublayers=${fullPrefillShapeSemantic} shape_covered_ops=${fullPrefillShapeCoveredOps} runtime_backend_dispatches=${fullPrefillRuntimeDispatches} runtime_semantic_ffn_dispatches=${fullPrefillRuntimeSemanticDispatches} runtime_projection_row_chain_dispatches=${fullPrefillRuntimeRowChainDispatches} runtime_projection_row_chain_attempts=${fullPrefillRuntimeRowChainAttempts} runtime_projection_row_chain_refused=${fullPrefillRuntimeRowChainRefused} qmatmul_row_chain_tiled_count=${fullPrefillRuntimeRowChainTiled} semantic_ffn_sublayer_count=${fullPrefillSemanticRowSerialCount} semantic_target_dispatches=${fullPrefillTargetDispatches} target_speedup=${fullPrefillTargetSpeedup.toFixed(2)}x target_max_abs_diff=${fullPrefillTargetMaxAbsDiff.toFixed(6)} target_shape_commands=${fullPrefillTargetShapeCommands} target_semantic_ffn_sublayers=${fullPrefillTargetShapeSemantic} target_runtime_backend_dispatches=${fullPrefillTargetRuntimeDispatches} target_semantic_rows=${fullPrefillTargetSemanticRows} target_semantic_hidden=${fullPrefillTargetSemanticHidden} target_semantic_row_serial_dot_ops=${fullPrefillTargetSemanticRowSerialDotOps} target_semantic_total_row_serial_dot_ops=${fullPrefillTargetSemanticTotalRowSerialDotOps} target_semantic_tile_parallel_groups=${fullPrefillTargetSemanticTileParallelGroups} target_semantic_row_serial_dot_ops_per_tile_parallel_group=${fullPrefillTargetSemanticRowSerialDotOpsPerTileGroup} target_semantic_total_row_serial_dot_ops_per_tile_parallel_group=${fullPrefillTargetSemanticTotalRowSerialDotOpsPerTileGroup}`,
    `full_prefill_two_phase=${fullPrefillTwoPhaseSpeedup.toFixed(2)}x max_abs_diff=${fullPrefillTwoPhaseMaxAbsDiff.toFixed(6)} runtime_backend_dispatches=${fullPrefillTwoPhaseRuntimeDispatches} qmatmul_row_chain_tiled_count=${fullPrefillTwoPhaseRuntimeRowChainTiled} qmatmul_row_chain_tiled_row_tile_groups=${fullPrefillTwoPhaseRowTileGroups} qmatmul_row_chain_tiled_n_tiles=${fullPrefillTwoPhaseNTiles} qmatmul_row_chain_tiled_serial_tile_loops=${fullPrefillTwoPhaseSerialLoops} qmatmul_row_chain_tiled_partial_slots=${fullPrefillTwoPhasePartialSlots} qmatmul_row_chain_tiled_scratch_capacity=${fullPrefillTwoPhaseScratchCapacity} qmatmul_row_chain_tiled_two_phase_count=${fullPrefillTwoPhaseRuntimeRowChainTwoPhase} qmatmul_row_chain_tiled_finalize_tile_groups=${fullPrefillTwoPhaseFinalizeTileGroups} qmatmul_row_chain_tiled_finalize_elements=${fullPrefillTwoPhaseFinalizeElements} qmatmul_row_chain_tiled_spilled_input=${fullPrefillTwoPhaseSpilledInput} qmatmul_row_chain_tiled_output_spills=${fullPrefillTwoPhaseOutputSpills}`,
    `full_prefill_single_dispatch=${fullPrefillSingleDispatchSpeedup.toFixed(2)}x max_abs_diff=${fullPrefillSingleDispatchMaxAbsDiff.toFixed(6)} runtime_backend_dispatches=${fullPrefillSingleDispatchRuntimeDispatches} runtime_projection_row_chain_dispatches=${fullPrefillSingleDispatchRuntimeRowChainDispatches} runtime_projection_row_chain_attempts=${fullPrefillSingleDispatchRuntimeRowChainAttempts} runtime_projection_row_chain_refused=${fullPrefillSingleDispatchRuntimeRowChainRefused} qmatmul_row_chain_tiled_count=${fullPrefillSingleDispatchRuntimeRowChainTiled} qmatmul_row_chain_tiled_row_tile_groups=${fullPrefillSingleDispatchRowTileGroups} qmatmul_row_chain_tiled_n_tiles=${fullPrefillSingleDispatchNTiles} qmatmul_row_chain_tiled_serial_tile_loops=${fullPrefillSingleDispatchSerialLoops} qmatmul_row_chain_tiled_partial_slots=${fullPrefillSingleDispatchPartialSlots} qmatmul_row_chain_tiled_scratch_capacity=${fullPrefillSingleDispatchScratchCapacity}`,
    `full_prefill_throughput_candidate=${fullPrefillThroughputCandidateSpeedup.toFixed(2)}x max_abs_diff=${fullPrefillThroughputCandidateMaxAbsDiff.toFixed(6)} shape_commands=${fullPrefillThroughputCandidateShapeCommands} shape_semantic_ffn_sublayers=${fullPrefillThroughputCandidateShapeSemantic} runtime_backend_dispatches=${fullPrefillThroughputCandidateRuntimeDispatches} runtime_semantic_ffn_dispatches=${fullPrefillThroughputCandidateRuntimeSemanticDispatches} qmatmul_row_chain_tiled_count=${fullPrefillThroughputCandidateRuntimeRowChainTiled} qmatmul_row_chain_tiled_row_tile_groups=${fullPrefillThroughputCandidateRowTileGroups} qmatmul_row_chain_tiled_n_tiles=${fullPrefillThroughputCandidateNTiles} qmatmul_row_chain_tiled_two_phase_count=${fullPrefillThroughputCandidateTwoPhase} qmatmul_row_chain_tiled_finalize_tile_groups=${fullPrefillThroughputCandidateFinalizeTileGroups} qmatmul_row_chain_tiled_finalize_elements=${fullPrefillThroughputCandidateFinalizeElements} qmatmul_row_chain_tiled_spilled_input=${fullPrefillThroughputCandidateSpilledInput} qmatmul_row_chain_tiled_output_spills=${fullPrefillThroughputCandidateOutputSpills}`,
    `smollm_prompt=${smollmPromptSpeedup.toFixed(2)}x max_abs_diff=${smollmPromptMaxAbsDiff.toFixed(6)} shape_commands=${smollmPromptShapeCommands} shape_semantic_ffn_sublayers=${smollmPromptShapeSemantic} shape_covered_ops=${smollmPromptShapeCoveredOps} runtime_backend_dispatches=${smollmPromptRuntimeDispatches} runtime_semantic_ffn_dispatches=${smollmPromptRuntimeSemanticDispatches} runtime_projection_row_chain_dispatches=${smollmPromptRuntimeRowChainDispatches} runtime_projection_row_chain_attempts=${smollmPromptRuntimeRowChainAttempts} runtime_projection_row_chain_refused=${smollmPromptRuntimeRowChainRefused} qmatmul_row_chain_tiled_count=${smollmPromptRuntimeRowChainTiled} semantic_ffn_sublayer_count=${smollmPromptSemanticRowSerialCount} semantic_target_dispatches=${smollmPromptTargetDispatches} target_speedup=${smollmPromptTargetSpeedup.toFixed(2)}x target_max_abs_diff=${smollmPromptTargetMaxAbsDiff.toFixed(6)} target_shape_commands=${smollmPromptTargetShapeCommands} target_semantic_ffn_sublayers=${smollmPromptTargetShapeSemantic} target_runtime_backend_dispatches=${smollmPromptTargetRuntimeDispatches} target_semantic_rows=${smollmPromptTargetSemanticRows} target_semantic_hidden=${smollmPromptTargetSemanticHidden} target_semantic_row_serial_dot_ops=${smollmPromptTargetSemanticRowSerialDotOps} target_semantic_total_row_serial_dot_ops=${smollmPromptTargetSemanticTotalRowSerialDotOps} target_semantic_tile_parallel_groups=${smollmPromptTargetSemanticTileParallelGroups} target_semantic_row_serial_dot_ops_per_tile_parallel_group=${smollmPromptTargetSemanticRowSerialDotOpsPerTileGroup} target_semantic_total_row_serial_dot_ops_per_tile_parallel_group=${smollmPromptTargetSemanticTotalRowSerialDotOpsPerTileGroup}`,
    `smollm_prompt_two_phase=${smollmPromptTwoPhaseSpeedup.toFixed(2)}x max_abs_diff=${smollmPromptTwoPhaseMaxAbsDiff.toFixed(6)} runtime_backend_dispatches=${smollmPromptTwoPhaseRuntimeDispatches} qmatmul_row_chain_tiled_count=${smollmPromptTwoPhaseRuntimeRowChainTiled} qmatmul_row_chain_tiled_row_tile_groups=${smollmPromptTwoPhaseRowTileGroups} qmatmul_row_chain_tiled_n_tiles=${smollmPromptTwoPhaseNTiles} qmatmul_row_chain_tiled_serial_tile_loops=${smollmPromptTwoPhaseSerialLoops} qmatmul_row_chain_tiled_partial_slots=${smollmPromptTwoPhasePartialSlots} qmatmul_row_chain_tiled_scratch_capacity=${smollmPromptTwoPhaseScratchCapacity} qmatmul_row_chain_tiled_two_phase_count=${smollmPromptTwoPhaseRuntimeRowChainTwoPhase} qmatmul_row_chain_tiled_finalize_tile_groups=${smollmPromptTwoPhaseFinalizeTileGroups} qmatmul_row_chain_tiled_finalize_elements=${smollmPromptTwoPhaseFinalizeElements} qmatmul_row_chain_tiled_spilled_input=${smollmPromptTwoPhaseSpilledInput} qmatmul_row_chain_tiled_output_spills=${smollmPromptTwoPhaseOutputSpills}`,
    `smollm_prompt_single_dispatch=${smollmPromptSingleDispatchSpeedup.toFixed(2)}x max_abs_diff=${smollmPromptSingleDispatchMaxAbsDiff.toFixed(6)} runtime_backend_dispatches=${smollmPromptSingleDispatchRuntimeDispatches} runtime_projection_row_chain_dispatches=${smollmPromptSingleDispatchRuntimeRowChainDispatches} runtime_projection_row_chain_attempts=${smollmPromptSingleDispatchRuntimeRowChainAttempts} runtime_projection_row_chain_refused=${smollmPromptSingleDispatchRuntimeRowChainRefused} qmatmul_row_chain_tiled_count=${smollmPromptSingleDispatchRuntimeRowChainTiled} qmatmul_row_chain_tiled_row_tile_groups=${smollmPromptSingleDispatchRowTileGroups} qmatmul_row_chain_tiled_n_tiles=${smollmPromptSingleDispatchNTiles} qmatmul_row_chain_tiled_serial_tile_loops=${smollmPromptSingleDispatchSerialLoops} qmatmul_row_chain_tiled_partial_slots=${smollmPromptSingleDispatchPartialSlots} qmatmul_row_chain_tiled_scratch_capacity=${smollmPromptSingleDispatchScratchCapacity}`,
    `smollm_prompt_throughput_candidate=${smollmPromptThroughputCandidateSpeedup.toFixed(2)}x max_abs_diff=${smollmPromptThroughputCandidateMaxAbsDiff.toFixed(6)} shape_commands=${smollmPromptThroughputCandidateShapeCommands} shape_semantic_ffn_sublayers=${smollmPromptThroughputCandidateShapeSemantic} runtime_backend_dispatches=${smollmPromptThroughputCandidateRuntimeDispatches} runtime_semantic_ffn_dispatches=${smollmPromptThroughputCandidateRuntimeSemanticDispatches} qmatmul_row_chain_tiled_count=${smollmPromptThroughputCandidateRuntimeRowChainTiled} qmatmul_row_chain_tiled_row_tile_groups=${smollmPromptThroughputCandidateRowTileGroups} qmatmul_row_chain_tiled_n_tiles=${smollmPromptThroughputCandidateNTiles} qmatmul_row_chain_tiled_two_phase_count=${smollmPromptThroughputCandidateTwoPhase} qmatmul_row_chain_tiled_finalize_tile_groups=${smollmPromptThroughputCandidateFinalizeTileGroups} qmatmul_row_chain_tiled_finalize_elements=${smollmPromptThroughputCandidateFinalizeElements} qmatmul_row_chain_tiled_spilled_input=${smollmPromptThroughputCandidateSpilledInput} qmatmul_row_chain_tiled_output_spills=${smollmPromptThroughputCandidateOutputSpills}`,
    `next=semantic_ffn_sublayer_throughput_kernel`,
  ].join("; ");

  return {
    attempt,
    fullPrefillSpeedup,
    fullPrefillMaxAbsDiff,
    fullPrefillShapeCommands,
    fullPrefillShapeRowChains,
    fullPrefillShapeCoveredOps,
    fullPrefillShapeSavedDispatches,
    fullPrefillRuntimeDispatches,
    fullPrefillRuntimeRowChainDispatches,
    fullPrefillRuntimeRowChainAttempts,
    fullPrefillRuntimeRowChainRefused,
    fullPrefillRuntimeRowChainTiled,
    fullPrefillTargetDispatches,
    fullPrefillTargetShapeCommands,
    fullPrefillTargetShapeSemantic,
    fullPrefillTargetShapeCoveredOps,
    fullPrefillTargetShapeSavedDispatches,
    fullPrefillTargetSpeedup,
    fullPrefillTargetMaxAbsDiff,
    fullPrefillTargetRuntimeDispatches,
    fullPrefillTargetSemanticTileParallelGroups,
    fullPrefillTargetSemanticTileRowGroups,
    fullPrefillTargetSemanticTileHiddenTiles,
    fullPrefillTargetSemanticTileOutputTiles,
    fullPrefillTargetSemanticRowSerialDotOpsPerTileGroup,
    fullPrefillTargetSemanticTotalRowSerialDotOpsPerTileGroup,
    fullPrefillThroughputCandidateSpeedup,
    fullPrefillThroughputCandidateMaxAbsDiff,
    fullPrefillThroughputCandidateShapeCommands,
    fullPrefillThroughputCandidateShapeSemantic,
    fullPrefillThroughputCandidateRuntimeDispatches,
    fullPrefillThroughputCandidateRuntimeSemanticDispatches,
    fullPrefillThroughputCandidateRuntimeRowChainTiled,
    fullPrefillThroughputCandidateRowTileGroups,
    fullPrefillThroughputCandidateNTiles,
    fullPrefillThroughputCandidateTwoPhase,
    fullPrefillThroughputCandidateFinalizeTileGroups,
    fullPrefillThroughputCandidateFinalizeElements,
    fullPrefillThroughputCandidateVsTwoPhase,
    fullPrefillTwoPhaseSpeedup,
    fullPrefillTwoPhaseMaxAbsDiff,
    fullPrefillTwoPhaseRuntimeDispatches,
    fullPrefillTwoPhaseRuntimeRowChainTiled,
    fullPrefillTwoPhaseRuntimeRowChainTwoPhase,
    fullPrefillSingleDispatchSpeedup,
    fullPrefillSingleDispatchMaxAbsDiff,
    fullPrefillSingleDispatchShapeCommands,
    fullPrefillSingleDispatchShapeRowChains,
    fullPrefillSingleDispatchShapeCoveredOps,
    fullPrefillSingleDispatchRuntimeDispatches,
    fullPrefillSingleDispatchRuntimeRowChainDispatches,
    fullPrefillSingleDispatchRuntimeRowChainAttempts,
    fullPrefillSingleDispatchRuntimeRowChainRefused,
    fullPrefillSingleDispatchRuntimeRowChainTiled,
    singleDispatchThroughputStatus,
    semanticCommandStatus,
    fullPrefillShapeSemantic,
    fullPrefillRuntimeSemanticDispatches,
    fullPrefillSemanticRowSerialCount,
    smollmPromptSpeedup,
    smollmPromptMaxAbsDiff,
    smollmPromptShapeCommands,
    smollmPromptShapeRowChains,
    smollmPromptShapeCoveredOps,
    smollmPromptShapeSavedDispatches,
    smollmPromptRuntimeDispatches,
    smollmPromptRuntimeRowChainDispatches,
    smollmPromptRuntimeRowChainAttempts,
    smollmPromptRuntimeRowChainRefused,
    smollmPromptRuntimeRowChainTiled,
    smollmPromptTargetDispatches,
    smollmPromptTargetShapeCommands,
    smollmPromptTargetShapeSemantic,
    smollmPromptTargetShapeCoveredOps,
    smollmPromptTargetShapeSavedDispatches,
    smollmPromptTargetSpeedup,
    smollmPromptTargetMaxAbsDiff,
    smollmPromptTargetRuntimeDispatches,
    smollmPromptTargetSemanticTileParallelGroups,
    smollmPromptTargetSemanticTileRowGroups,
    smollmPromptTargetSemanticTileHiddenTiles,
    smollmPromptTargetSemanticTileOutputTiles,
    smollmPromptTargetSemanticRowSerialDotOpsPerTileGroup,
    smollmPromptTargetSemanticTotalRowSerialDotOpsPerTileGroup,
    targetThroughputStatus,
    throughputCandidateStatus,
    smollmPromptThroughputCandidateSpeedup,
    smollmPromptThroughputCandidateMaxAbsDiff,
    smollmPromptThroughputCandidateShapeCommands,
    smollmPromptThroughputCandidateShapeSemantic,
    smollmPromptThroughputCandidateRuntimeDispatches,
    smollmPromptThroughputCandidateRuntimeSemanticDispatches,
    smollmPromptThroughputCandidateRuntimeRowChainTiled,
    smollmPromptThroughputCandidateRowTileGroups,
    smollmPromptThroughputCandidateNTiles,
    smollmPromptThroughputCandidateTwoPhase,
    smollmPromptThroughputCandidateFinalizeTileGroups,
    smollmPromptThroughputCandidateFinalizeElements,
    smollmPromptThroughputCandidateVsTwoPhase,
    smollmPromptTwoPhaseSpeedup,
    smollmPromptTwoPhaseMaxAbsDiff,
    smollmPromptTwoPhaseRuntimeDispatches,
    smollmPromptTwoPhaseRuntimeRowChainTiled,
    smollmPromptTwoPhaseRuntimeRowChainTwoPhase,
    smollmPromptSingleDispatchSpeedup,
    smollmPromptSingleDispatchMaxAbsDiff,
    smollmPromptSingleDispatchShapeCommands,
    smollmPromptSingleDispatchShapeRowChains,
    smollmPromptSingleDispatchShapeCoveredOps,
    smollmPromptSingleDispatchRuntimeDispatches,
    smollmPromptSingleDispatchRuntimeRowChainDispatches,
    smollmPromptSingleDispatchRuntimeRowChainAttempts,
    smollmPromptSingleDispatchRuntimeRowChainRefused,
    smollmPromptSingleDispatchRuntimeRowChainTiled,
    smollmPromptShapeSemantic,
    smollmPromptRuntimeSemanticDispatches,
    smollmPromptSemanticRowSerialCount,
    singleDispatchReduced,
    singleDispatchBlocker,
    failures,
    line,
  };
}

function focusedSemanticMargin(current) {
  return Math.min(
    projectionRowChainMaxAbsDiffCeil / Math.max(current.fullPrefillMaxAbsDiff, Number.EPSILON),
    projectionRowChainMaxAbsDiffCeil / Math.max(current.fullPrefillTwoPhaseMaxAbsDiff, Number.EPSILON),
    projectionRowChainMaxAbsDiffCeil / Math.max(current.fullPrefillSingleDispatchMaxAbsDiff, Number.EPSILON),
    projectionRowChainMaxAbsDiffCeil / Math.max(current.fullPrefillTargetMaxAbsDiff, Number.EPSILON),
    projectionRowChainMaxAbsDiffCeil / Math.max(current.smollmPromptMaxAbsDiff, Number.EPSILON),
    projectionRowChainMaxAbsDiffCeil / Math.max(current.smollmPromptTwoPhaseMaxAbsDiff, Number.EPSILON),
    projectionRowChainMaxAbsDiffCeil / Math.max(current.smollmPromptSingleDispatchMaxAbsDiff, Number.EPSILON),
    projectionRowChainMaxAbsDiffCeil / Math.max(current.smollmPromptTargetMaxAbsDiff, Number.EPSILON),
    current.fullPrefillRuntimeDispatches === 3 ? 1 : 0,
    current.fullPrefillRuntimeSemanticDispatches === 3 ? 1 : 0,
    current.fullPrefillSemanticRowSerialCount === 0 ? 1 : 0,
    current.fullPrefillSingleDispatchRuntimeDispatches === 2 ? 1 : 0,
    current.fullPrefillTargetRuntimeDispatches === 1 ? 1 : 0,
    current.smollmPromptRuntimeDispatches === 3 ? 1 : 0,
    current.smollmPromptRuntimeSemanticDispatches === 3 ? 1 : 0,
    current.smollmPromptSemanticRowSerialCount === 0 ? 1 : 0,
    current.smollmPromptSingleDispatchRuntimeDispatches === 2 ? 1 : 0,
    current.smollmPromptTargetRuntimeDispatches === 1 ? 1 : 0,
  ) + Math.min(current.fullPrefillSpeedup, current.smollmPromptSpeedup);
}

function selectedSemanticAttemptSummary(attempt) {
  const fullPrefillThroughputCandidateTileGroups =
    attempt.fullPrefillThroughputCandidateRowTileGroups * attempt.fullPrefillThroughputCandidateNTiles;
  const smollmPromptThroughputCandidateTileGroups =
    attempt.smollmPromptThroughputCandidateRowTileGroups * attempt.smollmPromptThroughputCandidateNTiles;
  return {
    attempt: attempt.attempt,
    failures: attempt.failures,
    targetThroughputStatus: attempt.targetThroughputStatus,
    throughputCandidateStatus: attempt.throughputCandidateStatus,
    semanticCommandStatus: attempt.semanticCommandStatus,
    singleDispatchThroughputStatus: attempt.singleDispatchThroughputStatus,
    singleDispatchRowChainDispatchReduced: attempt.singleDispatchReduced,
    fullPrefill: {
      speedup: roundMetric(attempt.fullPrefillSpeedup),
      maxAbsDiff: roundMetric(attempt.fullPrefillMaxAbsDiff),
      runtimeDispatches: attempt.fullPrefillRuntimeDispatches,
      semanticRuntimeDispatches: attempt.fullPrefillRuntimeSemanticDispatches,
      targetSpeedup: roundMetric(attempt.fullPrefillTargetSpeedup),
      targetRuntimeDispatches: attempt.fullPrefillTargetRuntimeDispatches,
      targetTileParallelGroups: attempt.fullPrefillTargetSemanticTileParallelGroups,
      targetTileRowGroups: attempt.fullPrefillTargetSemanticTileRowGroups,
      targetTileHiddenTiles: attempt.fullPrefillTargetSemanticTileHiddenTiles,
      targetTileOutputTiles: attempt.fullPrefillTargetSemanticTileOutputTiles,
      targetRowSerialDotOpsPerTileGroup: attempt.fullPrefillTargetSemanticRowSerialDotOpsPerTileGroup,
      targetTotalRowSerialDotOpsPerTileGroup: attempt.fullPrefillTargetSemanticTotalRowSerialDotOpsPerTileGroup,
      throughputCandidateSpeedup: roundMetric(attempt.fullPrefillThroughputCandidateSpeedup),
      throughputCandidateVsDefault: roundMetric(attempt.fullPrefillThroughputCandidateSpeedup / attempt.fullPrefillSpeedup),
      throughputCandidateVsTwoPhase: roundMetric(attempt.fullPrefillThroughputCandidateVsTwoPhase),
      throughputCandidateRuntimeDispatches: attempt.fullPrefillThroughputCandidateRuntimeDispatches,
      throughputCandidateTileParallelGroups: fullPrefillThroughputCandidateTileGroups,
      throughputCandidateFinalizeTileGroups: attempt.fullPrefillThroughputCandidateFinalizeTileGroups,
      throughputCandidateFinalizeElements: attempt.fullPrefillThroughputCandidateFinalizeElements,
      tileParallelGroupGap: roundMetric(
        attempt.fullPrefillTargetSemanticTileParallelGroups / fullPrefillThroughputCandidateTileGroups,
      ),
    },
    smollmPrompt: {
      speedup: roundMetric(attempt.smollmPromptSpeedup),
      maxAbsDiff: roundMetric(attempt.smollmPromptMaxAbsDiff),
      runtimeDispatches: attempt.smollmPromptRuntimeDispatches,
      semanticRuntimeDispatches: attempt.smollmPromptRuntimeSemanticDispatches,
      targetSpeedup: roundMetric(attempt.smollmPromptTargetSpeedup),
      targetRuntimeDispatches: attempt.smollmPromptTargetRuntimeDispatches,
      targetTileParallelGroups: attempt.smollmPromptTargetSemanticTileParallelGroups,
      targetTileRowGroups: attempt.smollmPromptTargetSemanticTileRowGroups,
      targetTileHiddenTiles: attempt.smollmPromptTargetSemanticTileHiddenTiles,
      targetTileOutputTiles: attempt.smollmPromptTargetSemanticTileOutputTiles,
      targetRowSerialDotOpsPerTileGroup: attempt.smollmPromptTargetSemanticRowSerialDotOpsPerTileGroup,
      targetTotalRowSerialDotOpsPerTileGroup: attempt.smollmPromptTargetSemanticTotalRowSerialDotOpsPerTileGroup,
      throughputCandidateSpeedup: roundMetric(attempt.smollmPromptThroughputCandidateSpeedup),
      throughputCandidateVsDefault: roundMetric(attempt.smollmPromptThroughputCandidateSpeedup / attempt.smollmPromptSpeedup),
      throughputCandidateVsTwoPhase: roundMetric(attempt.smollmPromptThroughputCandidateVsTwoPhase),
      throughputCandidateRuntimeDispatches: attempt.smollmPromptThroughputCandidateRuntimeDispatches,
      throughputCandidateTileParallelGroups: smollmPromptThroughputCandidateTileGroups,
      throughputCandidateFinalizeTileGroups: attempt.smollmPromptThroughputCandidateFinalizeTileGroups,
      throughputCandidateFinalizeElements: attempt.smollmPromptThroughputCandidateFinalizeElements,
      tileParallelGroupGap: roundMetric(
        attempt.smollmPromptTargetSemanticTileParallelGroups / smollmPromptThroughputCandidateTileGroups,
      ),
    },
  };
}

function writeFocusedSemanticArtifact(best, attempts, aggregate, line) {
  if (!writeArtifact) return null;
  mkdirSync(artifactDir, { recursive: true });
  const artifactPath = join(artifactDir, `frontier-qsemantic-${timestampForArtifact()}-${process.pid}.json`);
  const bestSummary = selectedSemanticAttemptSummary(best);
  const artifact = {
    schema: "zgml.frontier-qsemantic.v1",
    createdAt: new Date().toISOString(),
    command: {
      argv: process.argv,
      cwd: process.cwd(),
      build,
      frontierFilter,
    },
    platform: {
      node: process.version,
      platform: process.platform,
      arch: process.arch,
      cpus: os.cpus().length,
    },
    source: benchmarkBinaryMetadata({ root, binary: frontierBinary, build }),
    config: {
      maxAttempts,
      build,
      frontierFilter,
      projectionRowChainMaxAbsDiffCeil,
    },
    kind: "qsemantic",
    status: aggregate.length === 0 ? "pass" : "fail",
    selectedAttempt: best.attempt,
    attempts: attempts.length,
    aggregateFailures: aggregate,
    targetThroughputStatus: best.targetThroughputStatus,
    throughputCandidateStatus: best.throughputCandidateStatus,
    semanticCommandStatus: best.semanticCommandStatus,
    singleDispatchThroughputStatus: best.singleDispatchThroughputStatus,
    singleDispatchRowChainDispatchReduced: best.singleDispatchReduced,
    fullPrefill: bestSummary.fullPrefill,
    smollmPrompt: bestSummary.smollmPrompt,
    attemptSummaries: attempts.map(selectedSemanticAttemptSummary),
    next: "semantic_ffn_sublayer_throughput_kernel",
    line,
  };
  writeFileSync(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
  return resolve(artifactPath);
}

function scoreFocusedSemanticThroughputCandidate(output, attempt) {
  const fullPrefillLabel = "qsemantic full-prefill m=128 n=512 k=512 semantic throughput_candidate";
  const smollmPromptLabel = "qsemantic smollm-prompt m=128 n=576 k=576 semantic throughput_candidate";
  const fullPrefillProfileLabel = `${fullPrefillLabel} dispatch_profile`;
  const smollmPromptProfileLabel = `${smollmPromptLabel} dispatch_profile`;

  const fullPrefillSpeedup = metric(output, fullPrefillLabel, "speedup");
  const fullPrefillMaxAbsDiff = metric(output, fullPrefillLabel, "max_abs_diff");
  const fullPrefillShapeCommands = metric(output, fullPrefillProfileLabel, "shape_commands");
  const fullPrefillShapeSemantic = metric(output, fullPrefillProfileLabel, "shape_semantic_ffn_sublayers");
  const fullPrefillShapeCoveredOps = metric(output, fullPrefillProfileLabel, "shape_covered_ops");
  const fullPrefillRuntimeDispatches = metric(output, fullPrefillProfileLabel, "runtime_backend_dispatches");
  const fullPrefillRuntimeSemanticDispatches = metric(output, fullPrefillProfileLabel, "runtime_semantic_ffn_dispatches");
  const fullPrefillRuntimeRowChainTiled = metric(output, fullPrefillProfileLabel, "qmatmul_row_chain_tiled_count");
  const fullPrefillRowTileGroups = metric(output, fullPrefillProfileLabel, "qmatmul_row_chain_tiled_row_tile_groups");
  const fullPrefillNTiles = metric(output, fullPrefillProfileLabel, "qmatmul_row_chain_tiled_n_tiles");
  const fullPrefillTwoPhase = metric(output, fullPrefillProfileLabel, "qmatmul_row_chain_tiled_two_phase_count");
  const fullPrefillSpilledInput = metric(output, fullPrefillProfileLabel, "qmatmul_row_chain_tiled_spilled_input");
  const fullPrefillOutputSpills = metric(output, fullPrefillProfileLabel, "qmatmul_row_chain_tiled_output_spills");

  const smollmPromptSpeedup = metric(output, smollmPromptLabel, "speedup");
  const smollmPromptMaxAbsDiff = metric(output, smollmPromptLabel, "max_abs_diff");
  const smollmPromptShapeCommands = metric(output, smollmPromptProfileLabel, "shape_commands");
  const smollmPromptShapeSemantic = metric(output, smollmPromptProfileLabel, "shape_semantic_ffn_sublayers");
  const smollmPromptShapeCoveredOps = metric(output, smollmPromptProfileLabel, "shape_covered_ops");
  const smollmPromptRuntimeDispatches = metric(output, smollmPromptProfileLabel, "runtime_backend_dispatches");
  const smollmPromptRuntimeSemanticDispatches = metric(output, smollmPromptProfileLabel, "runtime_semantic_ffn_dispatches");
  const smollmPromptRuntimeRowChainTiled = metric(output, smollmPromptProfileLabel, "qmatmul_row_chain_tiled_count");
  const smollmPromptRowTileGroups = metric(output, smollmPromptProfileLabel, "qmatmul_row_chain_tiled_row_tile_groups");
  const smollmPromptNTiles = metric(output, smollmPromptProfileLabel, "qmatmul_row_chain_tiled_n_tiles");
  const smollmPromptTwoPhase = metric(output, smollmPromptProfileLabel, "qmatmul_row_chain_tiled_two_phase_count");
  const smollmPromptSpilledInput = metric(output, smollmPromptProfileLabel, "qmatmul_row_chain_tiled_spilled_input");
  const smollmPromptOutputSpills = metric(output, smollmPromptProfileLabel, "qmatmul_row_chain_tiled_output_spills");

  const failures = [];
  if (fullPrefillMaxAbsDiff > projectionRowChainMaxAbsDiffCeil) failures.push(`semantic throughput full-prefill max_abs_diff ${fullPrefillMaxAbsDiff.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  if (smollmPromptMaxAbsDiff > projectionRowChainMaxAbsDiffCeil) failures.push(`semantic throughput smollm-prompt max_abs_diff ${smollmPromptMaxAbsDiff.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  if (fullPrefillShapeCommands !== 1 || fullPrefillShapeSemantic !== 1 || fullPrefillShapeCoveredOps !== 9 || fullPrefillRuntimeDispatches !== 3 || fullPrefillRuntimeSemanticDispatches !== 3 || fullPrefillRuntimeRowChainTiled !== 1 || fullPrefillRowTileGroups !== 4 || fullPrefillNTiles !== 16 || fullPrefillTwoPhase !== 1) {
    failures.push("semantic throughput full-prefill profile must stay shape_commands=1 shape_semantic_ffn_sublayers=1 shape_covered_ops=9 runtime_backend_dispatches=3 runtime_semantic_ffn_dispatches=3 qmatmul_row_chain_tiled_count=1 row_tile_groups=4 n_tiles=16 two_phase_count=1");
  }
  if (smollmPromptShapeCommands !== 1 || smollmPromptShapeSemantic !== 1 || smollmPromptShapeCoveredOps !== 9 || smollmPromptRuntimeDispatches !== 3 || smollmPromptRuntimeSemanticDispatches !== 3 || smollmPromptRuntimeRowChainTiled !== 1 || smollmPromptRowTileGroups !== 4 || smollmPromptNTiles !== 18 || smollmPromptTwoPhase !== 1) {
    failures.push("semantic throughput smollm-prompt profile must stay shape_commands=1 shape_semantic_ffn_sublayers=1 shape_covered_ops=9 runtime_backend_dispatches=3 runtime_semantic_ffn_dispatches=3 qmatmul_row_chain_tiled_count=1 row_tile_groups=4 n_tiles=18 two_phase_count=1");
  }

  const line = [
    `frontier qsemantic throughput gate: ${failures.length === 0 ? "pass" : "fail"}`,
    `attempt=${attempt}/${maxAttempts}`,
    `full_prefill=${fullPrefillSpeedup.toFixed(2)}x diagnostic_floor=not-yet max_abs_diff=${fullPrefillMaxAbsDiff.toFixed(6)} runtime_backend_dispatches=${fullPrefillRuntimeDispatches} qmatmul_row_chain_tiled_count=${fullPrefillRuntimeRowChainTiled} tile_groups=${fullPrefillRowTileGroups * fullPrefillNTiles} qmatmul_row_chain_tiled_spilled_input=${fullPrefillSpilledInput} qmatmul_row_chain_tiled_output_spills=${fullPrefillOutputSpills}`,
    `smollm_prompt=${smollmPromptSpeedup.toFixed(2)}x diagnostic_floor=not-yet max_abs_diff=${smollmPromptMaxAbsDiff.toFixed(6)} runtime_backend_dispatches=${smollmPromptRuntimeDispatches} qmatmul_row_chain_tiled_count=${smollmPromptRuntimeRowChainTiled} tile_groups=${smollmPromptRowTileGroups * smollmPromptNTiles} qmatmul_row_chain_tiled_spilled_input=${smollmPromptSpilledInput} qmatmul_row_chain_tiled_output_spills=${smollmPromptOutputSpills}`,
    "next=semantic_ffn_sublayer_throughput_kernel",
  ].join("; ");

  return {
    attempt,
    fullPrefillSpeedup,
    fullPrefillMaxAbsDiff,
    fullPrefillShapeCommands,
    fullPrefillShapeSemantic,
    fullPrefillShapeCoveredOps,
    fullPrefillRuntimeDispatches,
    fullPrefillRuntimeSemanticDispatches,
    fullPrefillRuntimeRowChainTiled,
    fullPrefillRowTileGroups,
    fullPrefillNTiles,
    fullPrefillTwoPhase,
    fullPrefillSpilledInput,
    fullPrefillOutputSpills,
    smollmPromptSpeedup,
    smollmPromptMaxAbsDiff,
    smollmPromptShapeCommands,
    smollmPromptShapeSemantic,
    smollmPromptShapeCoveredOps,
    smollmPromptRuntimeDispatches,
    smollmPromptRuntimeSemanticDispatches,
    smollmPromptRuntimeRowChainTiled,
    smollmPromptRowTileGroups,
    smollmPromptNTiles,
    smollmPromptTwoPhase,
    smollmPromptSpilledInput,
    smollmPromptOutputSpills,
    failures,
    line,
  };
}

function focusedSemanticThroughputMargin(current) {
  return Math.min(
    projectionRowChainMaxAbsDiffCeil / Math.max(current.fullPrefillMaxAbsDiff, Number.EPSILON),
    projectionRowChainMaxAbsDiffCeil / Math.max(current.smollmPromptMaxAbsDiff, Number.EPSILON),
    current.fullPrefillShapeCommands === 1 ? 1 : 0,
    current.fullPrefillRuntimeDispatches === 3 ? 1 : 0,
    current.fullPrefillRuntimeRowChainTiled === 1 ? 1 : 0,
    current.smollmPromptShapeCommands === 1 ? 1 : 0,
    current.smollmPromptRuntimeDispatches === 3 ? 1 : 0,
    current.smollmPromptRuntimeRowChainTiled === 1 ? 1 : 0,
  ) + Math.min(current.fullPrefillSpeedup, current.smollmPromptSpeedup);
}

function selectedSemanticThroughputAttemptSummary(attempt) {
  return {
    attempt: attempt.attempt,
    failures: attempt.failures,
    fullPrefill: {
      speedup: roundMetric(attempt.fullPrefillSpeedup),
      maxAbsDiff: roundMetric(attempt.fullPrefillMaxAbsDiff),
      runtimeDispatches: attempt.fullPrefillRuntimeDispatches,
      semanticRuntimeDispatches: attempt.fullPrefillRuntimeSemanticDispatches,
      qmatmulRowChainTiledCount: attempt.fullPrefillRuntimeRowChainTiled,
      tileGroups: attempt.fullPrefillRowTileGroups * attempt.fullPrefillNTiles,
      spilledInput: attempt.fullPrefillSpilledInput,
      outputSpills: attempt.fullPrefillOutputSpills,
    },
    smollmPrompt: {
      speedup: roundMetric(attempt.smollmPromptSpeedup),
      maxAbsDiff: roundMetric(attempt.smollmPromptMaxAbsDiff),
      runtimeDispatches: attempt.smollmPromptRuntimeDispatches,
      semanticRuntimeDispatches: attempt.smollmPromptRuntimeSemanticDispatches,
      qmatmulRowChainTiledCount: attempt.smollmPromptRuntimeRowChainTiled,
      tileGroups: attempt.smollmPromptRowTileGroups * attempt.smollmPromptNTiles,
      spilledInput: attempt.smollmPromptSpilledInput,
      outputSpills: attempt.smollmPromptOutputSpills,
    },
  };
}

function writeFocusedSemanticThroughputArtifact(best, attempts, aggregate, line) {
  if (!writeArtifact) return null;
  mkdirSync(artifactDir, { recursive: true });
  const artifactPath = join(artifactDir, `frontier-qsemantic-throughput-${timestampForArtifact()}-${process.pid}.json`);
  const bestSummary = selectedSemanticThroughputAttemptSummary(best);
  const artifact = {
    schema: "zgml.frontier-qsemantic-throughput.v1",
    createdAt: new Date().toISOString(),
    command: {
      argv: process.argv,
      cwd: process.cwd(),
      build,
      frontierFilter,
      qsemanticVariants,
    },
    platform: {
      node: process.version,
      platform: process.platform,
      arch: process.arch,
      cpus: os.cpus().length,
    },
    source: benchmarkBinaryMetadata({ root, binary: frontierBinary, build }),
    config: {
      maxAttempts,
      build,
      frontierFilter,
      qsemanticVariants,
      projectionRowChainMaxAbsDiffCeil,
    },
    kind: "qsemantic-throughput",
    status: aggregate.length === 0 ? "pass" : "fail",
    selectedAttempt: best.attempt,
    attempts: attempts.length,
    aggregateFailures: aggregate,
    fullPrefill: bestSummary.fullPrefill,
    smollmPrompt: bestSummary.smollmPrompt,
    attemptSummaries: attempts.map(selectedSemanticThroughputAttemptSummary),
    next: "semantic_ffn_sublayer_throughput_kernel",
    line,
  };
  writeFileSync(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
  return resolve(artifactPath);
}

function runFocusedSemanticThroughputGate() {
  const attempts = [];
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      attempts.push(scoreFocusedSemanticThroughputCandidate(runBench(), attempt));
    } catch (err) {
      process.stderr.write(`${err.output ?? err.message}\n`);
      process.exit(1);
    }
  }
  const passing = attempts.filter((current) => current.failures.length === 0);
  const best = (passing.length > 0 ? passing : attempts).reduce((acc, current) => {
    if (!acc) return current;
    return focusedSemanticThroughputMargin(current) > focusedSemanticThroughputMargin(acc) ? current : acc;
  }, null);
  const aggregate = [];
  const diffAtMost = (field, ceil, label) => {
    const value = bestMin(attempts, field);
    if (!Number.isFinite(value) || value > ceil) aggregate.push(`${label} best max_abs_diff ${Number.isFinite(value) ? value.toFixed(6) : "n/a"} > ${ceil.toFixed(6)}`);
  };
  diffAtMost("fullPrefillMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "semantic throughput full-prefill");
  diffAtMost("smollmPromptMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "semantic throughput smollm-prompt");
  if (!anyEquals(attempts, [
    ["fullPrefillShapeCommands", 1],
    ["fullPrefillShapeSemantic", 1],
    ["fullPrefillShapeCoveredOps", 9],
    ["fullPrefillRuntimeDispatches", 3],
    ["fullPrefillRuntimeSemanticDispatches", 3],
    ["fullPrefillRuntimeRowChainTiled", 1],
    ["fullPrefillRowTileGroups", 4],
    ["fullPrefillNTiles", 16],
    ["fullPrefillTwoPhase", 1],
  ])) aggregate.push("semantic throughput full-prefill profile did not match in any attempt");
  if (!anyEquals(attempts, [
    ["smollmPromptShapeCommands", 1],
    ["smollmPromptShapeSemantic", 1],
    ["smollmPromptShapeCoveredOps", 9],
    ["smollmPromptRuntimeDispatches", 3],
    ["smollmPromptRuntimeSemanticDispatches", 3],
    ["smollmPromptRuntimeRowChainTiled", 1],
    ["smollmPromptRowTileGroups", 4],
    ["smollmPromptNTiles", 18],
    ["smollmPromptTwoPhase", 1],
  ])) aggregate.push("semantic throughput smollm-prompt profile did not match in any attempt");

  const line = aggregate.length === 0 ? best.line.replace("frontier qsemantic throughput gate: fail", "frontier qsemantic throughput gate: pass") : best.line;
  const artifactPath = writeFocusedSemanticThroughputArtifact(best, attempts, aggregate, line);
  if (artifactPath) {
    process.stdout.write(`FRONTIER_BENCH_JSON ${JSON.stringify({
      artifact: artifactPath,
      status: aggregate.length === 0 ? "pass" : "fail",
      kind: "qsemantic-throughput",
      selectedAttempt: best.attempt,
      attempts: attempts.length,
      next: "semantic_ffn_sublayer_throughput_kernel",
    })}\n`);
  }
  process.stdout.write(`${line}\n`);
  if (aggregate.length === 0 && passing.length === 0) {
    process.stdout.write(`frontier qsemantic throughput aggregate: pass across ${attempts.length} noisy attempts\n`);
  }
  if (best.attempt > 1) {
    process.stdout.write(`frontier qsemantic throughput retries: ${best.attempt - 1} noisy attempt(s) below best evidence\n`);
  }
  if (aggregate.length !== 0) {
    process.stderr.write(`${aggregate.join("; ")}\n`);
    process.exit(1);
  }
  process.exit(0);
}

function runFocusedSemanticGate() {
  if (isQsemanticThroughputOnly()) {
    runFocusedSemanticThroughputGate();
  }
  const attempts = [];
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      attempts.push(scoreFocusedSemantic(runBench(), attempt));
    } catch (err) {
      process.stderr.write(`${err.output ?? err.message}\n`);
      process.exit(1);
    }
  }
  const passing = attempts.filter((current) => current.failures.length === 0);
  const best = (passing.length > 0 ? passing : attempts).reduce((acc, current) => {
    if (!acc) return current;
    return focusedSemanticMargin(current) > focusedSemanticMargin(acc) ? current : acc;
  }, null);
  const aggregate = [];
  const diffAtMost = (field, ceil, label) => {
    const value = bestMin(attempts, field);
    if (!Number.isFinite(value) || value > ceil) aggregate.push(`${label} best max_abs_diff ${Number.isFinite(value) ? value.toFixed(6) : "n/a"} > ${ceil.toFixed(6)}`);
  };
  diffAtMost("fullPrefillMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "semantic full-prefill");
  diffAtMost("fullPrefillTwoPhaseMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "semantic full-prefill two-phase");
  diffAtMost("fullPrefillSingleDispatchMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "semantic full-prefill single-dispatch row-chain");
  diffAtMost("fullPrefillTargetMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "semantic full-prefill target");
  diffAtMost("smollmPromptMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "semantic smollm-prompt");
  diffAtMost("smollmPromptTwoPhaseMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "semantic smollm-prompt two-phase");
  diffAtMost("smollmPromptSingleDispatchMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "semantic smollm-prompt single-dispatch row-chain");
  diffAtMost("smollmPromptTargetMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "semantic smollm-prompt target");
  if (!anyEquals(attempts, [
    ["fullPrefillShapeCommands", 1],
    ["fullPrefillShapeSemantic", 1],
    ["fullPrefillShapeRowChains", 0],
    ["fullPrefillShapeCoveredOps", 9],
    ["fullPrefillShapeSavedDispatches", 8],
    ["fullPrefillRuntimeDispatches", 3],
    ["fullPrefillRuntimeSemanticDispatches", 3],
    ["fullPrefillRuntimeRowChainDispatches", 0],
    ["fullPrefillRuntimeRowChainAttempts", 0],
    ["fullPrefillRuntimeRowChainRefused", 0],
    ["fullPrefillRuntimeRowChainTiled", 0],
    ["fullPrefillSemanticRowSerialCount", 0],
    ["fullPrefillTargetDispatches", 1],
  ])) aggregate.push("semantic full-prefill command profile did not match in any attempt");
  if (!anyEquals(attempts, [
    ["fullPrefillTargetShapeCommands", 1],
    ["fullPrefillTargetShapeSemantic", 1],
    ["fullPrefillTargetShapeCoveredOps", 9],
    ["fullPrefillTargetShapeSavedDispatches", 8],
    ["fullPrefillTargetRuntimeDispatches", 1],
  ])) aggregate.push("semantic full-prefill target profile did not match in any attempt");
  if (!anyEquals(attempts, [
    ["fullPrefillSingleDispatchShapeCommands", 2],
    ["fullPrefillSingleDispatchShapeRowChains", 1],
    ["fullPrefillSingleDispatchShapeCoveredOps", 9],
    ["fullPrefillSingleDispatchRuntimeDispatches", 2],
    ["fullPrefillSingleDispatchRuntimeRowChainDispatches", 1],
    ["fullPrefillSingleDispatchRuntimeRowChainAttempts", 1],
    ["fullPrefillSingleDispatchRuntimeRowChainRefused", 0],
    ["fullPrefillSingleDispatchRuntimeRowChainTiled", 1],
  ])) aggregate.push("semantic full-prefill single-dispatch row-chain diagnostic did not match in any attempt");
  if (!anyEquals(attempts, [
    ["smollmPromptShapeCommands", 1],
    ["smollmPromptShapeSemantic", 1],
    ["smollmPromptShapeRowChains", 0],
    ["smollmPromptShapeCoveredOps", 9],
    ["smollmPromptShapeSavedDispatches", 8],
    ["smollmPromptRuntimeDispatches", 3],
    ["smollmPromptRuntimeSemanticDispatches", 3],
    ["smollmPromptRuntimeRowChainDispatches", 0],
    ["smollmPromptRuntimeRowChainAttempts", 0],
    ["smollmPromptRuntimeRowChainRefused", 0],
    ["smollmPromptRuntimeRowChainTiled", 0],
    ["smollmPromptSemanticRowSerialCount", 0],
    ["smollmPromptTargetDispatches", 1],
  ])) aggregate.push("semantic smollm-prompt command profile did not match in any attempt");
  if (!anyEquals(attempts, [
    ["smollmPromptTargetShapeCommands", 1],
    ["smollmPromptTargetShapeSemantic", 1],
    ["smollmPromptTargetShapeCoveredOps", 9],
    ["smollmPromptTargetShapeSavedDispatches", 8],
    ["smollmPromptTargetRuntimeDispatches", 1],
  ])) aggregate.push("semantic smollm-prompt target profile did not match in any attempt");
  if (!anyEquals(attempts, [
    ["smollmPromptSingleDispatchShapeCommands", 2],
    ["smollmPromptSingleDispatchShapeRowChains", 1],
    ["smollmPromptSingleDispatchShapeCoveredOps", 9],
    ["smollmPromptSingleDispatchRuntimeDispatches", 2],
    ["smollmPromptSingleDispatchRuntimeRowChainDispatches", 1],
    ["smollmPromptSingleDispatchRuntimeRowChainAttempts", 1],
    ["smollmPromptSingleDispatchRuntimeRowChainRefused", 0],
    ["smollmPromptSingleDispatchRuntimeRowChainTiled", 1],
  ])) aggregate.push("semantic smollm-prompt single-dispatch row-chain diagnostic did not match in any attempt");

  const line = aggregate.length === 0 ? best.line.replace("frontier qsemantic gate: fail", "frontier qsemantic gate: pass") : best.line;
  const artifactPath = writeFocusedSemanticArtifact(best, attempts, aggregate, line);
  if (artifactPath) {
    process.stdout.write(`FRONTIER_BENCH_JSON ${JSON.stringify({
      artifact: artifactPath,
      status: aggregate.length === 0 ? "pass" : "fail",
      kind: "qsemantic",
      selectedAttempt: best.attempt,
      attempts: attempts.length,
      targetThroughputStatus: best.targetThroughputStatus,
      throughputCandidateStatus: best.throughputCandidateStatus,
      next: "semantic_ffn_sublayer_throughput_kernel",
    })}\n`);
  }
  process.stdout.write(`${line}\n`);
  if (aggregate.length === 0 && passing.length === 0) {
    process.stdout.write(`frontier qsemantic aggregate: pass across ${attempts.length} noisy attempts\n`);
  }
  if (best.attempt > 1) {
    process.stdout.write(`frontier qsemantic retries: ${best.attempt - 1} noisy attempt(s) below best evidence\n`);
  }
  if (aggregate.length !== 0) {
    process.stderr.write(`${aggregate.join("; ")}\n`);
    process.exit(1);
  }
  process.exit(0);
}

function score(output, attempt) {
  const smallStaged = p50(output, "chain n=4096 staged");
  const smallFused = p50(output, "chain n=4096 one-pass");
  const largeStaged = p50(output, "chain n=262144 staged");
  const largeFused = p50(output, "chain n=262144 one-pass");
  const decodeTokS = throughput(output, "rmsnorm-attn-logits token", "tokens");
  const projectionChainLabel = "qproj prompt m=32 n=512 k=512 projection_chain";
  const projectionChainFullPrefillLabel = "qproj full-prefill m=128 n=512 k=512 projection_chain";
  const projectionChainSmollmPromptLabel = "qproj smollm-prompt m=128 n=576 k=576 projection_chain";
  const projectionGroupFullPrefillLabel = "qproj group full-prefill x4 m=128 n=512 k=512 projection_group";
  const projectionGroupSmollmPromptLabel = "qproj group smollm-prompt x4 m=128 n=576 k=576 projection_group";
  const projectionRowChainGroupFullPrefillLabel = "qrow group full-prefill x4 m=128 n=512 k=512 projection_row_chain_group";
  const projectionRowChainGroupSmollmPromptLabel = "qrow group smollm-prompt x4 m=128 n=576 k=576 projection_row_chain_group";
  const projectionRowChainRegionFullPrefillLabel = "qrow region full-prefill x7 m=128 n=512 k=512 projection_row_chain_two_phase_group";
  const projectionRowChainRegionSmollmPromptLabel = "qrow region smollm-prompt x7 m=128 n=576 k=576 projection_row_chain_two_phase_group";
  const projectionDecodeLabel = "qrow decode m=1 n=512 k=512 projection_row_chain";
  const projectionPromptLabel = "qrow prompt m=32 n=512 k=512 projection_row_chain";
  const projectionFullPrefillLabel = "qrow full-prefill m=128 n=512 k=512 projection_row_chain";
  const projectionSmollmPromptLabel = "qrow smollm-prompt m=128 n=576 k=576 projection_row_chain";
  const projectionPromptSingleDispatchLabel = "qrow prompt m=32 n=512 k=512 projection_row_chain_single_dispatch";
  const projectionFullPrefillSingleDispatchLabel = "qrow full-prefill m=128 n=512 k=512 projection_row_chain_single_dispatch";
  const projectionSmollmPromptSingleDispatchLabel = "qrow smollm-prompt m=128 n=576 k=576 projection_row_chain_single_dispatch";
  const projectionPromptTwoPhaseLabel = "qrow prompt m=32 n=512 k=512 projection_row_chain_two_phase";
  const projectionFullPrefillTwoPhaseLabel = "qrow full-prefill m=128 n=512 k=512 projection_row_chain_two_phase";
  const projectionSmollmPromptTwoPhaseLabel = "qrow smollm-prompt m=128 n=576 k=576 projection_row_chain_two_phase";
  const projectionRowChainGroupFullPrefillProfileLabel = `${projectionRowChainGroupFullPrefillLabel} dispatch_profile`;
  const projectionRowChainGroupSmollmPromptProfileLabel = `${projectionRowChainGroupSmollmPromptLabel} dispatch_profile`;
  const projectionRowChainRegionFullPrefillProfileLabel = `${projectionRowChainRegionFullPrefillLabel} dispatch_profile`;
  const projectionRowChainRegionSmollmPromptProfileLabel = `${projectionRowChainRegionSmollmPromptLabel} dispatch_profile`;
  const projectionDecodeProfileLabel = `${projectionDecodeLabel} dispatch_profile`;
  const projectionPromptProfileLabel = `${projectionPromptLabel} dispatch_profile`;
  const projectionFullPrefillProfileLabel = `${projectionFullPrefillLabel} dispatch_profile`;
  const projectionSmollmPromptProfileLabel = `${projectionSmollmPromptLabel} dispatch_profile`;
  const projectionPromptSingleDispatchProfileLabel = `${projectionPromptSingleDispatchLabel} dispatch_profile`;
  const projectionFullPrefillSingleDispatchProfileLabel = `${projectionFullPrefillSingleDispatchLabel} dispatch_profile`;
  const projectionSmollmPromptSingleDispatchProfileLabel = `${projectionSmollmPromptSingleDispatchLabel} dispatch_profile`;
  const projectionPromptTwoPhaseProfileLabel = `${projectionPromptTwoPhaseLabel} dispatch_profile`;
  const projectionFullPrefillTwoPhaseProfileLabel = `${projectionFullPrefillTwoPhaseLabel} dispatch_profile`;
  const projectionSmollmPromptTwoPhaseProfileLabel = `${projectionSmollmPromptTwoPhaseLabel} dispatch_profile`;

  const smallSpeedup = smallStaged / smallFused;
  const largeSpeedup = largeStaged / largeFused;
  const projectionChainSpeedup = metric(output, projectionChainLabel, "speedup");
  const projectionChainMaxAbsDiff = metric(output, projectionChainLabel, "max_abs_diff");
  const projectionChainFullPrefillSpeedup = metric(output, projectionChainFullPrefillLabel, "speedup");
  const projectionChainFullPrefillMaxAbsDiff = metric(output, projectionChainFullPrefillLabel, "max_abs_diff");
  const projectionChainSmollmPromptSpeedup = metric(output, projectionChainSmollmPromptLabel, "speedup");
  const projectionChainSmollmPromptMaxAbsDiff = metric(output, projectionChainSmollmPromptLabel, "max_abs_diff");
  const projectionGroupFullPrefillSpeedup = metric(output, projectionGroupFullPrefillLabel, "speedup");
  const projectionGroupFullPrefillMaxAbsDiff = metric(output, projectionGroupFullPrefillLabel, "max_abs_diff");
  const projectionGroupSmollmPromptSpeedup = metric(output, projectionGroupSmollmPromptLabel, "speedup");
  const projectionGroupSmollmPromptMaxAbsDiff = metric(output, projectionGroupSmollmPromptLabel, "max_abs_diff");
  const projectionRowChainGroupFullPrefillSpeedup = metric(output, projectionRowChainGroupFullPrefillLabel, "speedup");
  const projectionRowChainGroupFullPrefillMaxAbsDiff = metric(output, projectionRowChainGroupFullPrefillLabel, "max_abs_diff");
  const projectionRowChainGroupSmollmPromptSpeedup = metric(output, projectionRowChainGroupSmollmPromptLabel, "speedup");
  const projectionRowChainGroupSmollmPromptMaxAbsDiff = metric(output, projectionRowChainGroupSmollmPromptLabel, "max_abs_diff");
  const projectionRowChainRegionFullPrefillSpeedup = metric(output, projectionRowChainRegionFullPrefillLabel, "speedup");
  const projectionRowChainRegionFullPrefillMaxAbsDiff = metric(output, projectionRowChainRegionFullPrefillLabel, "max_abs_diff");
  const projectionRowChainRegionSmollmPromptSpeedup = metric(output, projectionRowChainRegionSmollmPromptLabel, "speedup");
  const projectionRowChainRegionSmollmPromptMaxAbsDiff = metric(output, projectionRowChainRegionSmollmPromptLabel, "max_abs_diff");
  const projectionDecodeSpeedup = hasMetric(output, projectionDecodeLabel, "speedup") ? metric(output, projectionDecodeLabel, "speedup") : null;
  const projectionDecodeMaxAbsDiff = hasMetric(output, projectionDecodeLabel, "max_abs_diff") ? metric(output, projectionDecodeLabel, "max_abs_diff") : null;
  const projectionPromptSpeedup = hasMetric(output, projectionPromptLabel, "speedup") ? metric(output, projectionPromptLabel, "speedup") : null;
  const projectionPromptMaxAbsDiff = hasMetric(output, projectionPromptLabel, "max_abs_diff") ? metric(output, projectionPromptLabel, "max_abs_diff") : null;
  const projectionFullPrefillSpeedup = hasMetric(output, projectionFullPrefillLabel, "speedup") ? metric(output, projectionFullPrefillLabel, "speedup") : null;
  const projectionFullPrefillMaxAbsDiff = hasMetric(output, projectionFullPrefillLabel, "max_abs_diff") ? metric(output, projectionFullPrefillLabel, "max_abs_diff") : null;
  const projectionSmollmPromptSpeedup = metric(output, projectionSmollmPromptLabel, "speedup");
  const projectionSmollmPromptMaxAbsDiff = metric(output, projectionSmollmPromptLabel, "max_abs_diff");
  const projectionPromptSingleDispatchSpeedup = metric(output, projectionPromptSingleDispatchLabel, "speedup");
  const projectionPromptSingleDispatchMaxAbsDiff = metric(output, projectionPromptSingleDispatchLabel, "max_abs_diff");
  const projectionFullPrefillSingleDispatchSpeedup = metric(output, projectionFullPrefillSingleDispatchLabel, "speedup");
  const projectionFullPrefillSingleDispatchMaxAbsDiff = metric(output, projectionFullPrefillSingleDispatchLabel, "max_abs_diff");
  const projectionSmollmPromptSingleDispatchSpeedup = metric(output, projectionSmollmPromptSingleDispatchLabel, "speedup");
  const projectionSmollmPromptSingleDispatchMaxAbsDiff = metric(output, projectionSmollmPromptSingleDispatchLabel, "max_abs_diff");
  const projectionPromptTwoPhaseSpeedup = metric(output, projectionPromptTwoPhaseLabel, "speedup");
  const projectionPromptTwoPhaseMaxAbsDiff = metric(output, projectionPromptTwoPhaseLabel, "max_abs_diff");
  const projectionFullPrefillTwoPhaseSpeedup = metric(output, projectionFullPrefillTwoPhaseLabel, "speedup");
  const projectionFullPrefillTwoPhaseMaxAbsDiff = metric(output, projectionFullPrefillTwoPhaseLabel, "max_abs_diff");
  const projectionSmollmPromptTwoPhaseSpeedup = metric(output, projectionSmollmPromptTwoPhaseLabel, "speedup");
  const projectionSmollmPromptTwoPhaseMaxAbsDiff = metric(output, projectionSmollmPromptTwoPhaseLabel, "max_abs_diff");
  const projectionRowChainGroupFullPrefillShapeCommands = metric(output, projectionRowChainGroupFullPrefillProfileLabel, "shape_commands");
  const projectionRowChainGroupFullPrefillShapeRowChains = metric(output, projectionRowChainGroupFullPrefillProfileLabel, "shape_projection_row_chains");
  const projectionRowChainGroupFullPrefillShapeCoveredOps = metric(output, projectionRowChainGroupFullPrefillProfileLabel, "shape_covered_ops");
  const projectionRowChainGroupFullPrefillShapeSavedDispatches = metric(output, projectionRowChainGroupFullPrefillProfileLabel, "shape_saved_dispatches");
  const projectionRowChainGroupFullPrefillRuntimeCommandDispatches = metric(output, projectionRowChainGroupFullPrefillProfileLabel, "runtime_command_dispatches");
  const projectionRowChainGroupSmollmPromptShapeCommands = metric(output, projectionRowChainGroupSmollmPromptProfileLabel, "shape_commands");
  const projectionRowChainGroupSmollmPromptShapeRowChains = metric(output, projectionRowChainGroupSmollmPromptProfileLabel, "shape_projection_row_chains");
  const projectionRowChainGroupSmollmPromptShapeCoveredOps = metric(output, projectionRowChainGroupSmollmPromptProfileLabel, "shape_covered_ops");
  const projectionRowChainGroupSmollmPromptShapeSavedDispatches = metric(output, projectionRowChainGroupSmollmPromptProfileLabel, "shape_saved_dispatches");
  const projectionRowChainGroupSmollmPromptRuntimeCommandDispatches = metric(output, projectionRowChainGroupSmollmPromptProfileLabel, "runtime_command_dispatches");
  const projectionRowChainRegionFullPrefillShapeCommands = metric(output, projectionRowChainRegionFullPrefillProfileLabel, "shape_commands");
  const projectionRowChainRegionFullPrefillShapeRowChains = metric(output, projectionRowChainRegionFullPrefillProfileLabel, "shape_projection_row_chains");
  const projectionRowChainRegionFullPrefillShapeCoveredOps = metric(output, projectionRowChainRegionFullPrefillProfileLabel, "shape_covered_ops");
  const projectionRowChainRegionFullPrefillShapeSavedDispatches = metric(output, projectionRowChainRegionFullPrefillProfileLabel, "shape_saved_dispatches");
  const projectionRowChainRegionFullPrefillRuntimeCommandDispatches = metric(output, projectionRowChainRegionFullPrefillProfileLabel, "runtime_command_dispatches");
  const projectionRowChainRegionFullPrefillRuntimeCount = metric(output, projectionRowChainRegionFullPrefillProfileLabel, "qmatmul_row_chain_tiled_two_phase_count");
  const projectionRowChainRegionSmollmPromptShapeCommands = metric(output, projectionRowChainRegionSmollmPromptProfileLabel, "shape_commands");
  const projectionRowChainRegionSmollmPromptShapeRowChains = metric(output, projectionRowChainRegionSmollmPromptProfileLabel, "shape_projection_row_chains");
  const projectionRowChainRegionSmollmPromptShapeCoveredOps = metric(output, projectionRowChainRegionSmollmPromptProfileLabel, "shape_covered_ops");
  const projectionRowChainRegionSmollmPromptShapeSavedDispatches = metric(output, projectionRowChainRegionSmollmPromptProfileLabel, "shape_saved_dispatches");
  const projectionRowChainRegionSmollmPromptRuntimeCommandDispatches = metric(output, projectionRowChainRegionSmollmPromptProfileLabel, "runtime_command_dispatches");
  const projectionRowChainRegionSmollmPromptRuntimeCount = metric(output, projectionRowChainRegionSmollmPromptProfileLabel, "qmatmul_row_chain_tiled_two_phase_count");
  const projectionDecodeShapeCommands = metric(output, projectionDecodeProfileLabel, "shape_commands");
  const projectionDecodeShapeRowChains = metric(output, projectionDecodeProfileLabel, "shape_projection_row_chains");
  const projectionDecodeShapeCoveredOps = metric(output, projectionDecodeProfileLabel, "shape_covered_ops");
  const projectionDecodeShapeSavedDispatches = metric(output, projectionDecodeProfileLabel, "shape_saved_dispatches");
  const projectionDecodeRuntimeCommandDispatches = metric(output, projectionDecodeProfileLabel, "runtime_command_dispatches");
  const projectionPromptShapeCommands = metric(output, projectionPromptProfileLabel, "shape_commands");
  const projectionPromptShapeRowChains = metric(output, projectionPromptProfileLabel, "shape_projection_row_chains");
  const projectionPromptShapeCoveredOps = metric(output, projectionPromptProfileLabel, "shape_covered_ops");
  const projectionPromptShapeSavedDispatches = metric(output, projectionPromptProfileLabel, "shape_saved_dispatches");
  const projectionPromptRuntimeCommandDispatches = metric(output, projectionPromptProfileLabel, "runtime_command_dispatches");
  const projectionPromptSingleDispatchShapeCommands = metric(output, projectionPromptSingleDispatchProfileLabel, "shape_commands");
  const projectionPromptSingleDispatchShapeRowChains = metric(output, projectionPromptSingleDispatchProfileLabel, "shape_projection_row_chains");
  const projectionPromptSingleDispatchShapeCoveredOps = metric(output, projectionPromptSingleDispatchProfileLabel, "shape_covered_ops");
  const projectionPromptSingleDispatchShapeSavedDispatches = metric(output, projectionPromptSingleDispatchProfileLabel, "shape_saved_dispatches");
  const projectionPromptSingleDispatchRuntimeCommandDispatches = metric(output, projectionPromptSingleDispatchProfileLabel, "runtime_command_dispatches");
  const projectionPromptTwoPhaseShapeCommands = metric(output, projectionPromptTwoPhaseProfileLabel, "shape_commands");
  const projectionPromptTwoPhaseShapeRowChains = metric(output, projectionPromptTwoPhaseProfileLabel, "shape_projection_row_chains");
  const projectionPromptTwoPhaseShapeCoveredOps = metric(output, projectionPromptTwoPhaseProfileLabel, "shape_covered_ops");
  const projectionPromptTwoPhaseShapeSavedDispatches = metric(output, projectionPromptTwoPhaseProfileLabel, "shape_saved_dispatches");
  const projectionPromptTwoPhaseRuntimeCommandDispatches = metric(output, projectionPromptTwoPhaseProfileLabel, "runtime_command_dispatches");
  const projectionPromptTwoPhaseRuntimeCount = metric(output, projectionPromptTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_two_phase_count");
  const projectionFullPrefillShapeCommands = metric(output, projectionFullPrefillProfileLabel, "shape_commands");
  const projectionFullPrefillShapeRowChains = metric(output, projectionFullPrefillProfileLabel, "shape_projection_row_chains");
  const projectionFullPrefillShapeCoveredOps = metric(output, projectionFullPrefillProfileLabel, "shape_covered_ops");
  const projectionFullPrefillShapeSavedDispatches = metric(output, projectionFullPrefillProfileLabel, "shape_saved_dispatches");
  const projectionFullPrefillRuntimeCommandDispatches = metric(output, projectionFullPrefillProfileLabel, "runtime_command_dispatches");
  const projectionFullPrefillSingleDispatchRuntimeCommandDispatches = metric(output, projectionFullPrefillSingleDispatchProfileLabel, "runtime_command_dispatches");
  const projectionFullPrefillTwoPhaseRuntimeCommandDispatches = metric(output, projectionFullPrefillTwoPhaseProfileLabel, "runtime_command_dispatches");
  const projectionFullPrefillTwoPhaseRuntimeCount = metric(output, projectionFullPrefillTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_two_phase_count");
  const projectionSmollmPromptShapeCommands = metric(output, projectionSmollmPromptProfileLabel, "shape_commands");
  const projectionSmollmPromptShapeRowChains = metric(output, projectionSmollmPromptProfileLabel, "shape_projection_row_chains");
  const projectionSmollmPromptShapeCoveredOps = metric(output, projectionSmollmPromptProfileLabel, "shape_covered_ops");
  const projectionSmollmPromptShapeSavedDispatches = metric(output, projectionSmollmPromptProfileLabel, "shape_saved_dispatches");
  const projectionSmollmPromptRuntimeCommandDispatches = metric(output, projectionSmollmPromptProfileLabel, "runtime_command_dispatches");
  const projectionSmollmPromptSingleDispatchRuntimeCommandDispatches = metric(output, projectionSmollmPromptSingleDispatchProfileLabel, "runtime_command_dispatches");
  const projectionSmollmPromptTwoPhaseRuntimeCommandDispatches = metric(output, projectionSmollmPromptTwoPhaseProfileLabel, "runtime_command_dispatches");
  const projectionSmollmPromptTwoPhaseRuntimeCount = metric(output, projectionSmollmPromptTwoPhaseProfileLabel, "qmatmul_row_chain_tiled_two_phase_count");
  const projectionCandidateReady =
    projectionDecodeSpeedup !== null &&
    projectionPromptSpeedup !== null &&
    projectionDecodeMaxAbsDiff !== null &&
    projectionPromptMaxAbsDiff !== null &&
    projectionDecodeSpeedup >= projectionRowChainDefaultSpeedupFloor &&
    projectionPromptSpeedup >= projectionRowChainDefaultSpeedupFloor &&
    projectionDecodeMaxAbsDiff <= projectionRowChainMaxAbsDiffCeil &&
    projectionPromptMaxAbsDiff <= projectionRowChainMaxAbsDiffCeil;
  const projectionPromptCandidateReady =
    projectionPromptSpeedup !== null &&
    projectionPromptMaxAbsDiff !== null &&
    projectionPromptSpeedup >= projectionRowChainDefaultSpeedupFloor &&
    projectionPromptMaxAbsDiff <= projectionRowChainMaxAbsDiffCeil;
  const projectionDefaultDecision = "off";
  const projectionDefaultReason = projectionCandidateReady
    ? "full_model_gate_required"
    : "speedup_or_correctness_below_default_floor";
  const failures = [];
  if (smallSpeedup < 2.0) failures.push(`small chain fusion ${smallSpeedup.toFixed(2)}x < 2.00x`);
  if (largeSpeedup < largeChainSpeedupFloor) failures.push(`large chain fusion ${largeSpeedup.toFixed(2)}x < ${largeChainSpeedupFloor.toFixed(2)}x`);
  if (decodeTokS < decodeishTokSFloor) failures.push(`decode-ish token frontier ${decodeTokS.toFixed(2)} tok/s < ${decodeishTokSFloor} tok/s`);
  if (projectionChainSpeedup < projectionChainTileSpeedupFloor) failures.push(`projection_chain prompt tile ${projectionChainSpeedup.toFixed(2)}x < ${projectionChainTileSpeedupFloor.toFixed(2)}x`);
  if (projectionChainMaxAbsDiff > projectionChainMaxAbsDiffCeil) failures.push(`projection_chain prompt max_abs_diff ${projectionChainMaxAbsDiff.toFixed(6)} > ${projectionChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionChainFullPrefillSpeedup < projectionChainFullPrefillSpeedupFloor) failures.push(`projection_chain full-prefill ${projectionChainFullPrefillSpeedup.toFixed(2)}x < ${projectionChainFullPrefillSpeedupFloor.toFixed(2)}x`);
  if (projectionChainFullPrefillMaxAbsDiff > projectionChainMaxAbsDiffCeil) failures.push(`projection_chain full-prefill max_abs_diff ${projectionChainFullPrefillMaxAbsDiff.toFixed(6)} > ${projectionChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionChainSmollmPromptSpeedup < projectionSmollmPromptSpeedupFloor) failures.push(`projection_chain smollm-prompt ${projectionChainSmollmPromptSpeedup.toFixed(2)}x < ${projectionSmollmPromptSpeedupFloor.toFixed(2)}x`);
  if (projectionChainSmollmPromptMaxAbsDiff > projectionChainMaxAbsDiffCeil) failures.push(`projection_chain smollm-prompt max_abs_diff ${projectionChainSmollmPromptMaxAbsDiff.toFixed(6)} > ${projectionChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionGroupFullPrefillMaxAbsDiff > projectionChainMaxAbsDiffCeil) failures.push(`projection_group full-prefill max_abs_diff ${projectionGroupFullPrefillMaxAbsDiff.toFixed(6)} > ${projectionChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionGroupSmollmPromptSpeedup < projectionSmollmPromptSpeedupFloor) failures.push(`projection_group smollm-prompt ${projectionGroupSmollmPromptSpeedup.toFixed(2)}x < ${projectionSmollmPromptSpeedupFloor.toFixed(2)}x`);
  if (projectionGroupSmollmPromptMaxAbsDiff > projectionChainMaxAbsDiffCeil) failures.push(`projection_group smollm-prompt max_abs_diff ${projectionGroupSmollmPromptMaxAbsDiff.toFixed(6)} > ${projectionChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionRowChainGroupFullPrefillMaxAbsDiff > projectionRowChainMaxAbsDiffCeil) failures.push(`projection_row_chain_group full-prefill max_abs_diff ${projectionRowChainGroupFullPrefillMaxAbsDiff.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionRowChainGroupSmollmPromptSpeedup < projectionSmollmPromptSpeedupFloor) failures.push(`projection_row_chain_group smollm-prompt ${projectionRowChainGroupSmollmPromptSpeedup.toFixed(2)}x < ${projectionSmollmPromptSpeedupFloor.toFixed(2)}x`);
  if (projectionRowChainGroupSmollmPromptMaxAbsDiff > projectionRowChainMaxAbsDiffCeil) failures.push(`projection_row_chain_group smollm-prompt max_abs_diff ${projectionRowChainGroupSmollmPromptMaxAbsDiff.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionRowChainRegionFullPrefillMaxAbsDiff > projectionRowChainMaxAbsDiffCeil) failures.push(`projection_row_chain_two_phase_group full-prefill region max_abs_diff ${projectionRowChainRegionFullPrefillMaxAbsDiff.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionRowChainRegionSmollmPromptMaxAbsDiff > projectionRowChainMaxAbsDiffCeil) failures.push(`projection_row_chain_two_phase_group smollm-prompt region max_abs_diff ${projectionRowChainRegionSmollmPromptMaxAbsDiff.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionPromptMaxAbsDiff !== null && projectionPromptMaxAbsDiff > projectionRowChainMaxAbsDiffCeil) failures.push(`projection_row_chain prompt max_abs_diff ${projectionPromptMaxAbsDiff.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionFullPrefillMaxAbsDiff !== null && projectionFullPrefillMaxAbsDiff > projectionRowChainMaxAbsDiffCeil) failures.push(`projection_row_chain full-prefill max_abs_diff ${projectionFullPrefillMaxAbsDiff.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionSmollmPromptMaxAbsDiff > projectionRowChainMaxAbsDiffCeil) failures.push(`projection_row_chain smollm-prompt max_abs_diff ${projectionSmollmPromptMaxAbsDiff.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionRowChainGroupFullPrefillShapeCommands !== 4 || projectionRowChainGroupFullPrefillShapeRowChains !== 4 || projectionRowChainGroupFullPrefillShapeCoveredOps !== 20 || projectionRowChainGroupFullPrefillShapeSavedDispatches !== 16) failures.push("projection_row_chain group full-prefill shape profile must stay shape_commands=4 shape_projection_row_chains=4 shape_covered_ops=20 shape_saved_dispatches=16");
  if (projectionRowChainGroupSmollmPromptShapeCommands !== 4 || projectionRowChainGroupSmollmPromptShapeRowChains !== 4 || projectionRowChainGroupSmollmPromptShapeCoveredOps !== 20 || projectionRowChainGroupSmollmPromptShapeSavedDispatches !== 16) failures.push("projection_row_chain group smollm-prompt shape profile must stay shape_commands=4 shape_projection_row_chains=4 shape_covered_ops=20 shape_saved_dispatches=16");
  if (projectionRowChainRegionFullPrefillShapeCommands !== 7 || projectionRowChainRegionFullPrefillShapeRowChains !== 7 || projectionRowChainRegionFullPrefillShapeCoveredOps !== 35 || projectionRowChainRegionFullPrefillShapeSavedDispatches !== 28) failures.push("projection_row_chain two-phase group full-prefill region shape profile must stay shape_commands=7 shape_projection_row_chains=7 shape_covered_ops=35 shape_saved_dispatches=28");
  if (projectionRowChainRegionSmollmPromptShapeCommands !== 7 || projectionRowChainRegionSmollmPromptShapeRowChains !== 7 || projectionRowChainRegionSmollmPromptShapeCoveredOps !== 35 || projectionRowChainRegionSmollmPromptShapeSavedDispatches !== 28) failures.push("projection_row_chain two-phase group smollm-prompt region shape profile must stay shape_commands=7 shape_projection_row_chains=7 shape_covered_ops=35 shape_saved_dispatches=28");
  if (projectionDecodeShapeCommands !== 1 || projectionDecodeShapeRowChains !== 1 || projectionDecodeShapeCoveredOps !== 5 || projectionDecodeShapeSavedDispatches !== 4) failures.push("projection_row_chain decode shape profile must stay shape_commands=1 shape_projection_row_chains=1 shape_covered_ops=5 shape_saved_dispatches=4");
  if (projectionPromptShapeCommands !== 1 || projectionPromptShapeRowChains !== 1 || projectionPromptShapeCoveredOps !== 5 || projectionPromptShapeSavedDispatches !== 4) failures.push("projection_row_chain prompt shape profile must stay shape_commands=1 shape_projection_row_chains=1 shape_covered_ops=5 shape_saved_dispatches=4");
  if (projectionPromptSingleDispatchShapeCommands !== 1 || projectionPromptSingleDispatchShapeRowChains !== 1 || projectionPromptSingleDispatchShapeCoveredOps !== 5 || projectionPromptSingleDispatchShapeSavedDispatches !== 4) failures.push("projection_row_chain_single_dispatch prompt shape profile must stay shape_commands=1 shape_projection_row_chains=1 shape_covered_ops=5 shape_saved_dispatches=4");
  if (projectionPromptTwoPhaseShapeCommands !== 1 || projectionPromptTwoPhaseShapeRowChains !== 1 || projectionPromptTwoPhaseShapeCoveredOps !== 5 || projectionPromptTwoPhaseShapeSavedDispatches !== 4) failures.push("projection_row_chain_two_phase prompt shape profile must stay shape_commands=1 shape_projection_row_chains=1 shape_covered_ops=5 shape_saved_dispatches=4");
  if (projectionFullPrefillShapeCommands !== 1 || projectionFullPrefillShapeRowChains !== 1 || projectionFullPrefillShapeCoveredOps !== 5 || projectionFullPrefillShapeSavedDispatches !== 4) failures.push("projection_row_chain full-prefill shape profile must stay shape_commands=1 shape_projection_row_chains=1 shape_covered_ops=5 shape_saved_dispatches=4");
  if (projectionSmollmPromptShapeCommands !== 1 || projectionSmollmPromptShapeRowChains !== 1 || projectionSmollmPromptShapeCoveredOps !== 5 || projectionSmollmPromptShapeSavedDispatches !== 4) failures.push("projection_row_chain smollm-prompt shape profile must stay shape_commands=1 shape_projection_row_chains=1 shape_covered_ops=5 shape_saved_dispatches=4");
  if (projectionRowChainGroupFullPrefillRuntimeCommandDispatches !== 8) failures.push("projection_row_chain group full-prefill runtime profile must stay at 8 command dispatches");
  if (projectionRowChainGroupSmollmPromptRuntimeCommandDispatches !== 8) failures.push("projection_row_chain group smollm-prompt runtime profile must stay at 8 command dispatches");
  if (projectionRowChainRegionFullPrefillRuntimeCommandDispatches !== 14 || projectionRowChainRegionFullPrefillRuntimeCount !== 7) failures.push("projection_row_chain two-phase group full-prefill region runtime profile must stay at 14 command dispatches with two_phase_count=7");
  if (projectionRowChainRegionSmollmPromptRuntimeCommandDispatches !== 14 || projectionRowChainRegionSmollmPromptRuntimeCount !== 7) failures.push("projection_row_chain two-phase group smollm-prompt region runtime profile must stay at 14 command dispatches with two_phase_count=7");
  if (projectionDecodeRuntimeCommandDispatches !== 1) failures.push("projection_row_chain decode runtime profile must stay at 1 command dispatch");
  if (projectionPromptRuntimeCommandDispatches !== 2) failures.push("projection_row_chain prompt runtime profile must stay at 2 command dispatches");
  if (projectionPromptSingleDispatchRuntimeCommandDispatches !== 1) failures.push("projection_row_chain_single_dispatch prompt runtime profile must stay at 1 command dispatch");
  if (projectionFullPrefillRuntimeCommandDispatches !== 2) failures.push("projection_row_chain full-prefill runtime profile must stay at 2 command dispatches");
  if (projectionFullPrefillSingleDispatchRuntimeCommandDispatches !== 1) failures.push("projection_row_chain_single_dispatch full-prefill runtime profile must stay at 1 command dispatch");
  if (projectionSmollmPromptRuntimeCommandDispatches !== 2) failures.push("projection_row_chain smollm-prompt runtime profile must stay at 2 command dispatches");
  if (projectionSmollmPromptSingleDispatchRuntimeCommandDispatches !== 1) failures.push("projection_row_chain_single_dispatch smollm-prompt runtime profile must stay at 1 command dispatch");

  const line = [
    `frontier bench gate: ${failures.length === 0 ? "pass" : "fail"}`,
    `attempt=${attempt}/${maxAttempts}`,
    `small_chain=${smallSpeedup.toFixed(2)}x floor=2.00x`,
    `large_chain=${largeSpeedup.toFixed(2)}x floor=${largeChainSpeedupFloor.toFixed(2)}x`,
    `decodeish=${decodeTokS.toFixed(2)} tok/s floor=${decodeishTokSFloor}`,
    `projection_chain_prompt=${projectionChainSpeedup.toFixed(2)}x observed max_abs_diff=${projectionChainMaxAbsDiff.toFixed(6)} floor=${projectionChainTileSpeedupFloor.toFixed(2)} diff_ceil=${projectionChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_chain_full_prefill=${projectionChainFullPrefillSpeedup.toFixed(2)}x observed max_abs_diff=${projectionChainFullPrefillMaxAbsDiff.toFixed(6)} candidate=${projectionChainFullPrefillSpeedup >= projectionChainFullPrefillCandidateSpeedupFloor ? "ready" : "off"} floor=${projectionChainFullPrefillSpeedupFloor.toFixed(2)} candidate_floor=${projectionChainFullPrefillCandidateSpeedupFloor.toFixed(2)} diff_ceil=${projectionChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_chain_smollm_prompt=${projectionChainSmollmPromptSpeedup.toFixed(2)}x observed max_abs_diff=${projectionChainSmollmPromptMaxAbsDiff.toFixed(6)} floor=${projectionSmollmPromptSpeedupFloor.toFixed(2)} diff_ceil=${projectionChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_group_full_prefill=${projectionGroupFullPrefillSpeedup.toFixed(2)}x observed max_abs_diff=${projectionGroupFullPrefillMaxAbsDiff.toFixed(6)} candidate=${projectionGroupFullPrefillSpeedup >= projectionGroupCandidateSpeedupFloor ? "ready" : "off"} floor=${projectionGroupCandidateSpeedupFloor.toFixed(2)} diff_ceil=${projectionChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_group_smollm_prompt=${projectionGroupSmollmPromptSpeedup.toFixed(2)}x observed max_abs_diff=${projectionGroupSmollmPromptMaxAbsDiff.toFixed(6)} floor=${projectionSmollmPromptSpeedupFloor.toFixed(2)} diff_ceil=${projectionChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_group_full_prefill=${projectionRowChainGroupFullPrefillSpeedup.toFixed(2)}x observed max_abs_diff=${projectionRowChainGroupFullPrefillMaxAbsDiff.toFixed(6)} shape_commands=${projectionRowChainGroupFullPrefillShapeCommands} shape_projection_row_chains=${projectionRowChainGroupFullPrefillShapeRowChains} shape_covered_ops=${projectionRowChainGroupFullPrefillShapeCoveredOps} shape_saved_dispatches=${projectionRowChainGroupFullPrefillShapeSavedDispatches} runtime_command_dispatches=${projectionRowChainGroupFullPrefillRuntimeCommandDispatches} candidate=${projectionRowChainGroupFullPrefillSpeedup >= projectionRowChainDefaultSpeedupFloor ? "ready" : "off"} ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_group_smollm_prompt=${projectionRowChainGroupSmollmPromptSpeedup.toFixed(2)}x observed max_abs_diff=${projectionRowChainGroupSmollmPromptMaxAbsDiff.toFixed(6)} shape_commands=${projectionRowChainGroupSmollmPromptShapeCommands} shape_projection_row_chains=${projectionRowChainGroupSmollmPromptShapeRowChains} shape_covered_ops=${projectionRowChainGroupSmollmPromptShapeCoveredOps} shape_saved_dispatches=${projectionRowChainGroupSmollmPromptShapeSavedDispatches} runtime_command_dispatches=${projectionRowChainGroupSmollmPromptRuntimeCommandDispatches} floor=${projectionSmollmPromptSpeedupFloor.toFixed(2)} ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_two_phase_region_full_prefill=${projectionRowChainRegionFullPrefillSpeedup.toFixed(2)}x observed max_abs_diff=${projectionRowChainRegionFullPrefillMaxAbsDiff.toFixed(6)} shape_commands=${projectionRowChainRegionFullPrefillShapeCommands} shape_projection_row_chains=${projectionRowChainRegionFullPrefillShapeRowChains} shape_covered_ops=${projectionRowChainRegionFullPrefillShapeCoveredOps} shape_saved_dispatches=${projectionRowChainRegionFullPrefillShapeSavedDispatches} runtime_command_dispatches=${projectionRowChainRegionFullPrefillRuntimeCommandDispatches} two_phase_count=${projectionRowChainRegionFullPrefillRuntimeCount} selected=${projectionRowChainRegionFullPrefillRuntimeCount === 7 ? "yes" : "off"} candidate=diagnostic ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_two_phase_region_smollm_prompt=${projectionRowChainRegionSmollmPromptSpeedup.toFixed(2)}x observed max_abs_diff=${projectionRowChainRegionSmollmPromptMaxAbsDiff.toFixed(6)} shape_commands=${projectionRowChainRegionSmollmPromptShapeCommands} shape_projection_row_chains=${projectionRowChainRegionSmollmPromptShapeRowChains} shape_covered_ops=${projectionRowChainRegionSmollmPromptShapeCoveredOps} shape_saved_dispatches=${projectionRowChainRegionSmollmPromptShapeSavedDispatches} runtime_command_dispatches=${projectionRowChainRegionSmollmPromptRuntimeCommandDispatches} two_phase_count=${projectionRowChainRegionSmollmPromptRuntimeCount} selected=${projectionRowChainRegionSmollmPromptRuntimeCount === 7 ? "yes" : "off"} candidate=diagnostic ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_decode=${projectionDecodeSpeedup === null ? "unavailable" : `${projectionDecodeSpeedup.toFixed(2)}x observed max_abs_diff=${projectionDecodeMaxAbsDiff.toFixed(6)} shape_commands=${projectionDecodeShapeCommands} shape_projection_row_chains=${projectionDecodeShapeRowChains} shape_covered_ops=${projectionDecodeShapeCoveredOps} shape_saved_dispatches=${projectionDecodeShapeSavedDispatches} runtime_command_dispatches=${projectionDecodeRuntimeCommandDispatches} ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`}`,
    `projection_row_chain_prompt=${projectionPromptSpeedup === null ? "unavailable" : `${projectionPromptSpeedup.toFixed(2)}x observed max_abs_diff=${projectionPromptMaxAbsDiff.toFixed(6)} shape_commands=${projectionPromptShapeCommands} shape_projection_row_chains=${projectionPromptShapeRowChains} shape_covered_ops=${projectionPromptShapeCoveredOps} shape_saved_dispatches=${projectionPromptShapeSavedDispatches} runtime_command_dispatches=${projectionPromptRuntimeCommandDispatches} ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`}`,
    `projection_row_chain_single_dispatch_prompt=${projectionPromptSingleDispatchSpeedup.toFixed(2)}x observed max_abs_diff=${projectionPromptSingleDispatchMaxAbsDiff.toFixed(6)} shape_commands=${projectionPromptSingleDispatchShapeCommands} shape_projection_row_chains=${projectionPromptSingleDispatchShapeRowChains} shape_covered_ops=${projectionPromptSingleDispatchShapeCoveredOps} shape_saved_dispatches=${projectionPromptSingleDispatchShapeSavedDispatches} runtime_command_dispatches=${projectionPromptSingleDispatchRuntimeCommandDispatches} ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_two_phase_prompt=${projectionPromptTwoPhaseSpeedup.toFixed(2)}x observed max_abs_diff=${projectionPromptTwoPhaseMaxAbsDiff.toFixed(6)} shape_commands=${projectionPromptTwoPhaseShapeCommands} shape_projection_row_chains=${projectionPromptTwoPhaseShapeRowChains} shape_covered_ops=${projectionPromptTwoPhaseShapeCoveredOps} shape_saved_dispatches=${projectionPromptTwoPhaseShapeSavedDispatches} runtime_command_dispatches=${projectionPromptTwoPhaseRuntimeCommandDispatches} two_phase_count=${projectionPromptTwoPhaseRuntimeCount} selected=${projectionPromptTwoPhaseRuntimeCount > 0 ? "yes" : "off"} ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_full_prefill=${projectionFullPrefillSpeedup === null ? "unavailable" : `${projectionFullPrefillSpeedup.toFixed(2)}x observed max_abs_diff=${projectionFullPrefillMaxAbsDiff.toFixed(6)} shape_commands=${projectionFullPrefillShapeCommands} shape_projection_row_chains=${projectionFullPrefillShapeRowChains} shape_covered_ops=${projectionFullPrefillShapeCoveredOps} shape_saved_dispatches=${projectionFullPrefillShapeSavedDispatches} runtime_command_dispatches=${projectionFullPrefillRuntimeCommandDispatches} candidate=${projectionFullPrefillSpeedup >= projectionRowChainDefaultSpeedupFloor ? "ready" : "off"} ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`}`,
    `projection_row_chain_single_dispatch_full_prefill=${projectionFullPrefillSingleDispatchSpeedup.toFixed(2)}x observed max_abs_diff=${projectionFullPrefillSingleDispatchMaxAbsDiff.toFixed(6)} runtime_command_dispatches=${projectionFullPrefillSingleDispatchRuntimeCommandDispatches} candidate=${projectionFullPrefillSingleDispatchSpeedup >= projectionRowChainDefaultSpeedupFloor ? "ready" : "off"} ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_two_phase_full_prefill=${projectionFullPrefillTwoPhaseSpeedup.toFixed(2)}x observed max_abs_diff=${projectionFullPrefillTwoPhaseMaxAbsDiff.toFixed(6)} runtime_command_dispatches=${projectionFullPrefillTwoPhaseRuntimeCommandDispatches} two_phase_count=${projectionFullPrefillTwoPhaseRuntimeCount} selected=${projectionFullPrefillTwoPhaseRuntimeCount > 0 ? "yes" : "off"} candidate=${projectionFullPrefillTwoPhaseSpeedup >= projectionRowChainDefaultSpeedupFloor ? "ready" : "off"} ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_smollm_prompt=${projectionSmollmPromptSpeedup.toFixed(2)}x observed max_abs_diff=${projectionSmollmPromptMaxAbsDiff.toFixed(6)} shape_commands=${projectionSmollmPromptShapeCommands} shape_projection_row_chains=${projectionSmollmPromptShapeRowChains} shape_covered_ops=${projectionSmollmPromptShapeCoveredOps} shape_saved_dispatches=${projectionSmollmPromptShapeSavedDispatches} runtime_command_dispatches=${projectionSmollmPromptRuntimeCommandDispatches} diagnostic=off ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_single_dispatch_smollm_prompt=${projectionSmollmPromptSingleDispatchSpeedup.toFixed(2)}x observed max_abs_diff=${projectionSmollmPromptSingleDispatchMaxAbsDiff.toFixed(6)} runtime_command_dispatches=${projectionSmollmPromptSingleDispatchRuntimeCommandDispatches} diagnostic=off ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_two_phase_smollm_prompt=${projectionSmollmPromptTwoPhaseSpeedup.toFixed(2)}x observed max_abs_diff=${projectionSmollmPromptTwoPhaseMaxAbsDiff.toFixed(6)} runtime_command_dispatches=${projectionSmollmPromptTwoPhaseRuntimeCommandDispatches} two_phase_count=${projectionSmollmPromptTwoPhaseRuntimeCount} selected=${projectionSmollmPromptTwoPhaseRuntimeCount > 0 ? "yes" : "off"} diagnostic=off ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_prompt_candidate=${projectionPromptCandidateReady ? "ready" : "off"} floor=${projectionRowChainDefaultSpeedupFloor.toFixed(2)} diff_ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_candidate=${projectionCandidateReady ? "ready" : "off"} floor=${projectionRowChainDefaultSpeedupFloor.toFixed(2)} diff_ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_lowering=${projectionRowChainLowering}`,
    `projection_row_chain_diagnostic_kernel=${projectionRowChainDiagnosticKernel}`,
    `projection_row_chain_next=${projectionRowChainNextTarget}`,
    `projection_row_chain_default=${projectionDefaultDecision} floor=${projectionRowChainDefaultSpeedupFloor.toFixed(2)} reason=${projectionDefaultReason}`,
  ].join("; ");

  return {
    attempt,
    smallSpeedup,
    largeSpeedup,
    decodeTokS,
    projectionChainSpeedup,
    projectionChainMaxAbsDiff,
    projectionChainFullPrefillSpeedup,
    projectionChainFullPrefillMaxAbsDiff,
    projectionChainSmollmPromptSpeedup,
    projectionChainSmollmPromptMaxAbsDiff,
    projectionGroupFullPrefillSpeedup,
    projectionGroupFullPrefillMaxAbsDiff,
    projectionGroupSmollmPromptSpeedup,
    projectionGroupSmollmPromptMaxAbsDiff,
    projectionRowChainGroupFullPrefillSpeedup,
    projectionRowChainGroupFullPrefillMaxAbsDiff,
    projectionRowChainGroupSmollmPromptSpeedup,
    projectionRowChainGroupSmollmPromptMaxAbsDiff,
    projectionRowChainRegionFullPrefillSpeedup,
    projectionRowChainRegionFullPrefillMaxAbsDiff,
    projectionRowChainRegionSmollmPromptSpeedup,
    projectionRowChainRegionSmollmPromptMaxAbsDiff,
    projectionDecodeSpeedup,
    projectionDecodeMaxAbsDiff,
    projectionPromptSpeedup,
    projectionPromptMaxAbsDiff,
    projectionFullPrefillSpeedup,
    projectionFullPrefillMaxAbsDiff,
    projectionSmollmPromptSpeedup,
    projectionSmollmPromptMaxAbsDiff,
    projectionPromptSingleDispatchSpeedup,
    projectionPromptSingleDispatchMaxAbsDiff,
    projectionFullPrefillSingleDispatchSpeedup,
    projectionFullPrefillSingleDispatchMaxAbsDiff,
    projectionSmollmPromptSingleDispatchSpeedup,
    projectionSmollmPromptSingleDispatchMaxAbsDiff,
    projectionPromptTwoPhaseSpeedup,
    projectionPromptTwoPhaseMaxAbsDiff,
    projectionFullPrefillTwoPhaseSpeedup,
    projectionFullPrefillTwoPhaseMaxAbsDiff,
    projectionSmollmPromptTwoPhaseSpeedup,
    projectionSmollmPromptTwoPhaseMaxAbsDiff,
    projectionRowChainGroupFullPrefillShapeCommands,
    projectionRowChainGroupFullPrefillShapeRowChains,
    projectionRowChainGroupFullPrefillShapeCoveredOps,
    projectionRowChainGroupFullPrefillShapeSavedDispatches,
    projectionRowChainGroupFullPrefillRuntimeCommandDispatches,
    projectionRowChainGroupSmollmPromptShapeCommands,
    projectionRowChainGroupSmollmPromptShapeRowChains,
    projectionRowChainGroupSmollmPromptShapeCoveredOps,
    projectionRowChainGroupSmollmPromptShapeSavedDispatches,
    projectionRowChainGroupSmollmPromptRuntimeCommandDispatches,
    projectionRowChainRegionFullPrefillShapeCommands,
    projectionRowChainRegionFullPrefillShapeRowChains,
    projectionRowChainRegionFullPrefillShapeCoveredOps,
    projectionRowChainRegionFullPrefillShapeSavedDispatches,
    projectionRowChainRegionFullPrefillRuntimeCommandDispatches,
    projectionRowChainRegionFullPrefillRuntimeCount,
    projectionRowChainRegionSmollmPromptShapeCommands,
    projectionRowChainRegionSmollmPromptShapeRowChains,
    projectionRowChainRegionSmollmPromptShapeCoveredOps,
    projectionRowChainRegionSmollmPromptShapeSavedDispatches,
    projectionRowChainRegionSmollmPromptRuntimeCommandDispatches,
    projectionRowChainRegionSmollmPromptRuntimeCount,
    projectionDecodeShapeCommands,
    projectionDecodeShapeRowChains,
    projectionDecodeShapeCoveredOps,
    projectionDecodeShapeSavedDispatches,
    projectionDecodeRuntimeCommandDispatches,
    projectionPromptShapeCommands,
    projectionPromptShapeRowChains,
    projectionPromptShapeCoveredOps,
    projectionPromptShapeSavedDispatches,
    projectionPromptRuntimeCommandDispatches,
    projectionPromptSingleDispatchShapeCommands,
    projectionPromptSingleDispatchShapeRowChains,
    projectionPromptSingleDispatchShapeCoveredOps,
    projectionPromptSingleDispatchShapeSavedDispatches,
    projectionPromptSingleDispatchRuntimeCommandDispatches,
    projectionPromptTwoPhaseShapeCommands,
    projectionPromptTwoPhaseShapeRowChains,
    projectionPromptTwoPhaseShapeCoveredOps,
    projectionPromptTwoPhaseShapeSavedDispatches,
    projectionPromptTwoPhaseRuntimeCommandDispatches,
    projectionPromptTwoPhaseRuntimeCount,
    projectionFullPrefillShapeCommands,
    projectionFullPrefillShapeRowChains,
    projectionFullPrefillShapeCoveredOps,
    projectionFullPrefillShapeSavedDispatches,
    projectionFullPrefillRuntimeCommandDispatches,
    projectionFullPrefillSingleDispatchRuntimeCommandDispatches,
    projectionFullPrefillTwoPhaseRuntimeCommandDispatches,
    projectionFullPrefillTwoPhaseRuntimeCount,
    projectionSmollmPromptShapeCommands,
    projectionSmollmPromptShapeRowChains,
    projectionSmollmPromptShapeCoveredOps,
    projectionSmollmPromptShapeSavedDispatches,
    projectionSmollmPromptRuntimeCommandDispatches,
    projectionSmollmPromptSingleDispatchRuntimeCommandDispatches,
    projectionSmollmPromptTwoPhaseRuntimeCommandDispatches,
    projectionSmollmPromptTwoPhaseRuntimeCount,
    projectionDefaultDecision,
    projectionDefaultReason,
    failures,
    line,
  };
}

function scoreMargin(current) {
  const projectionPromptMargin = current.projectionPromptMaxAbsDiff === null
    ? Infinity
    : projectionRowChainMaxAbsDiffCeil / Math.max(current.projectionPromptMaxAbsDiff, Number.EPSILON);
  const projectionPromptSpeedupMargin = current.projectionPromptSpeedup === null
    ? 0
    : current.projectionPromptSpeedup / projectionRowChainDefaultSpeedupFloor;
  const projectionFullPrefillMargin = current.projectionFullPrefillMaxAbsDiff === null
    ? Infinity
    : projectionRowChainMaxAbsDiffCeil / Math.max(current.projectionFullPrefillMaxAbsDiff, Number.EPSILON);
  return Math.min(
    current.smallSpeedup / 2.0,
    current.largeSpeedup / largeChainSpeedupFloor,
    current.decodeTokS / decodeishTokSFloor,
    current.projectionChainSpeedup / projectionChainTileSpeedupFloor,
    projectionChainMaxAbsDiffCeil / Math.max(current.projectionChainMaxAbsDiff, Number.EPSILON),
    current.projectionChainFullPrefillSpeedup / projectionChainFullPrefillSpeedupFloor,
    projectionChainMaxAbsDiffCeil / Math.max(current.projectionChainFullPrefillMaxAbsDiff, Number.EPSILON),
    current.projectionChainSmollmPromptSpeedup / projectionSmollmPromptSpeedupFloor,
    projectionChainMaxAbsDiffCeil / Math.max(current.projectionChainSmollmPromptMaxAbsDiff, Number.EPSILON),
    projectionChainMaxAbsDiffCeil / Math.max(current.projectionGroupFullPrefillMaxAbsDiff, Number.EPSILON),
    current.projectionGroupSmollmPromptSpeedup / projectionSmollmPromptSpeedupFloor,
    projectionChainMaxAbsDiffCeil / Math.max(current.projectionGroupSmollmPromptMaxAbsDiff, Number.EPSILON),
    projectionRowChainMaxAbsDiffCeil / Math.max(current.projectionRowChainGroupFullPrefillMaxAbsDiff, Number.EPSILON),
    current.projectionRowChainGroupSmollmPromptSpeedup / projectionSmollmPromptSpeedupFloor,
    projectionRowChainMaxAbsDiffCeil / Math.max(current.projectionRowChainGroupSmollmPromptMaxAbsDiff, Number.EPSILON),
    projectionRowChainMaxAbsDiffCeil / Math.max(current.projectionRowChainRegionFullPrefillMaxAbsDiff, Number.EPSILON),
    projectionRowChainMaxAbsDiffCeil / Math.max(current.projectionRowChainRegionSmollmPromptMaxAbsDiff, Number.EPSILON),
    current.projectionRowChainRegionFullPrefillRuntimeCount === 7 ? 1 : 0,
    current.projectionRowChainRegionSmollmPromptRuntimeCount === 7 ? 1 : 0,
    projectionRowChainMaxAbsDiffCeil / Math.max(current.projectionSmollmPromptMaxAbsDiff, Number.EPSILON),
    projectionPromptMargin,
    projectionFullPrefillMargin,
    projectionPromptSpeedupMargin,
  );
}

function chooseBest(attempts) {
  return attempts.reduce((acc, current) => {
    if (!acc) return current;
    return scoreMargin(current) > scoreMargin(acc) ? current : acc;
  }, null);
}

function bestMax(attempts, field) {
  return Math.max(...attempts.map((attempt) => attempt[field]).filter((value) => Number.isFinite(value)));
}

function bestMin(attempts, field) {
  return Math.min(...attempts.map((attempt) => attempt[field]).filter((value) => Number.isFinite(value)));
}

function anyEquals(attempts, fields) {
  return attempts.some((attempt) => fields.every(([field, expected]) => attempt[field] === expected));
}

function aggregateFailures(attempts) {
  const failures = [];
  const speedAtLeast = (field, floor, label) => {
    const value = bestMax(attempts, field);
    if (!Number.isFinite(value) || value < floor) failures.push(`${label} best ${Number.isFinite(value) ? value.toFixed(2) : "n/a"}x < ${floor.toFixed(2)}x`);
  };
  const diffAtMost = (field, ceil, label) => {
    const value = bestMin(attempts, field);
    if (!Number.isFinite(value) || value > ceil) failures.push(`${label} best max_abs_diff ${Number.isFinite(value) ? value.toFixed(6) : "n/a"} > ${ceil.toFixed(6)}`);
  };
  const exactProfile = (fields, label) => {
    if (!anyEquals(attempts, fields)) failures.push(`${label} profile did not match in any attempt`);
  };

  speedAtLeast("smallSpeedup", 2.0, "small chain fusion");
  speedAtLeast("largeSpeedup", largeChainSpeedupFloor, "large chain fusion");
  speedAtLeast("decodeTokS", decodeishTokSFloor, "decode-ish token frontier");
  speedAtLeast("projectionChainSpeedup", projectionChainTileSpeedupFloor, "projection_chain prompt tile");
  diffAtMost("projectionChainMaxAbsDiff", projectionChainMaxAbsDiffCeil, "projection_chain prompt");
  speedAtLeast("projectionChainFullPrefillSpeedup", projectionChainFullPrefillSpeedupFloor, "projection_chain full-prefill");
  diffAtMost("projectionChainFullPrefillMaxAbsDiff", projectionChainMaxAbsDiffCeil, "projection_chain full-prefill");
  speedAtLeast("projectionChainSmollmPromptSpeedup", projectionSmollmPromptSpeedupFloor, "projection_chain smollm-prompt");
  diffAtMost("projectionChainSmollmPromptMaxAbsDiff", projectionChainMaxAbsDiffCeil, "projection_chain smollm-prompt");
  diffAtMost("projectionGroupFullPrefillMaxAbsDiff", projectionChainMaxAbsDiffCeil, "projection_group full-prefill");
  speedAtLeast("projectionGroupSmollmPromptSpeedup", projectionSmollmPromptSpeedupFloor, "projection_group smollm-prompt");
  diffAtMost("projectionGroupSmollmPromptMaxAbsDiff", projectionChainMaxAbsDiffCeil, "projection_group smollm-prompt");
  diffAtMost("projectionRowChainGroupFullPrefillMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "projection_row_chain_group full-prefill");
  speedAtLeast("projectionRowChainGroupSmollmPromptSpeedup", projectionSmollmPromptSpeedupFloor, "projection_row_chain_group smollm-prompt");
  diffAtMost("projectionRowChainGroupSmollmPromptMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "projection_row_chain_group smollm-prompt");
  diffAtMost("projectionRowChainRegionFullPrefillMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "projection_row_chain_two_phase_group full-prefill region");
  diffAtMost("projectionRowChainRegionSmollmPromptMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "projection_row_chain_two_phase_group smollm-prompt region");
  diffAtMost("projectionPromptMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "projection_row_chain prompt");
  diffAtMost("projectionFullPrefillMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "projection_row_chain full-prefill");
  diffAtMost("projectionSmollmPromptMaxAbsDiff", projectionRowChainMaxAbsDiffCeil, "projection_row_chain smollm-prompt");

  exactProfile([
    ["projectionRowChainGroupFullPrefillShapeCommands", 4],
    ["projectionRowChainGroupFullPrefillShapeRowChains", 4],
    ["projectionRowChainGroupFullPrefillShapeCoveredOps", 20],
    ["projectionRowChainGroupFullPrefillShapeSavedDispatches", 16],
    ["projectionRowChainGroupFullPrefillRuntimeCommandDispatches", 8],
  ], "projection_row_chain group full-prefill");
  exactProfile([
    ["projectionRowChainGroupSmollmPromptShapeCommands", 4],
    ["projectionRowChainGroupSmollmPromptShapeRowChains", 4],
    ["projectionRowChainGroupSmollmPromptShapeCoveredOps", 20],
    ["projectionRowChainGroupSmollmPromptShapeSavedDispatches", 16],
    ["projectionRowChainGroupSmollmPromptRuntimeCommandDispatches", 8],
  ], "projection_row_chain group smollm-prompt");
  exactProfile([
    ["projectionRowChainRegionFullPrefillShapeCommands", 7],
    ["projectionRowChainRegionFullPrefillShapeRowChains", 7],
    ["projectionRowChainRegionFullPrefillShapeCoveredOps", 35],
    ["projectionRowChainRegionFullPrefillShapeSavedDispatches", 28],
    ["projectionRowChainRegionFullPrefillRuntimeCommandDispatches", 14],
    ["projectionRowChainRegionFullPrefillRuntimeCount", 7],
  ], "projection_row_chain two-phase group full-prefill region");
  exactProfile([
    ["projectionRowChainRegionSmollmPromptShapeCommands", 7],
    ["projectionRowChainRegionSmollmPromptShapeRowChains", 7],
    ["projectionRowChainRegionSmollmPromptShapeCoveredOps", 35],
    ["projectionRowChainRegionSmollmPromptShapeSavedDispatches", 28],
    ["projectionRowChainRegionSmollmPromptRuntimeCommandDispatches", 14],
    ["projectionRowChainRegionSmollmPromptRuntimeCount", 7],
  ], "projection_row_chain two-phase group smollm-prompt region");
  for (const [label, prefix] of [
    ["projection_row_chain decode", "projectionDecode"],
    ["projection_row_chain prompt", "projectionPrompt"],
    ["projection_row_chain full-prefill", "projectionFullPrefill"],
    ["projection_row_chain smollm-prompt", "projectionSmollmPrompt"],
  ]) {
    exactProfile([
      [`${prefix}ShapeCommands`, 1],
      [`${prefix}ShapeRowChains`, 1],
      [`${prefix}ShapeCoveredOps`, 5],
      [`${prefix}ShapeSavedDispatches`, 4],
      [`${prefix}RuntimeCommandDispatches`, prefix === "projectionDecode" ? 1 : 2],
    ], label);
  }
  exactProfile([
    ["projectionPromptSingleDispatchShapeCommands", 1],
    ["projectionPromptSingleDispatchShapeRowChains", 1],
    ["projectionPromptSingleDispatchShapeCoveredOps", 5],
    ["projectionPromptSingleDispatchShapeSavedDispatches", 4],
    ["projectionPromptSingleDispatchRuntimeCommandDispatches", 1],
  ], "projection_row_chain_single_dispatch prompt");
  exactProfile([["projectionFullPrefillSingleDispatchRuntimeCommandDispatches", 1]], "projection_row_chain_single_dispatch full-prefill");
  exactProfile([["projectionSmollmPromptSingleDispatchRuntimeCommandDispatches", 1]], "projection_row_chain_single_dispatch smollm-prompt");
  exactProfile([
    ["projectionPromptTwoPhaseShapeCommands", 1],
    ["projectionPromptTwoPhaseShapeRowChains", 1],
    ["projectionPromptTwoPhaseShapeCoveredOps", 5],
    ["projectionPromptTwoPhaseShapeSavedDispatches", 4],
  ], "projection_row_chain_two_phase prompt");

  return failures;
}

function attemptDiagnostic(current) {
  const decode = current.projectionDecodeSpeedup === null ? "n/a" : current.projectionDecodeSpeedup.toFixed(2);
  const prompt = current.projectionPromptSpeedup === null ? "n/a" : current.projectionPromptSpeedup.toFixed(2);
  const full = current.projectionFullPrefillSpeedup === null ? "n/a" : current.projectionFullPrefillSpeedup.toFixed(2);
  return [
    `attempt=${current.attempt}/${maxAttempts}`,
    `margin=${scoreMargin(current).toFixed(2)}x`,
    `projection_chain_prompt=${current.projectionChainSpeedup.toFixed(2)}x`,
    `projection_chain_full=${current.projectionChainFullPrefillSpeedup.toFixed(2)}x`,
    `projection_row_chain_decode=${decode}x`,
    `projection_row_chain_prompt=${prompt}x`,
    `projection_row_chain_full=${full}x`,
    `two_phase_region_full=${current.projectionRowChainRegionFullPrefillSpeedup.toFixed(2)}x/${current.projectionRowChainRegionFullPrefillRuntimeCount}`,
    `two_phase_region_smollm=${current.projectionRowChainRegionSmollmPromptSpeedup.toFixed(2)}x/${current.projectionRowChainRegionSmollmPromptRuntimeCount}`,
    `single_dispatch_prompt=${current.projectionPromptSingleDispatchSpeedup.toFixed(2)}x`,
    `single_dispatch_full=${current.projectionFullPrefillSingleDispatchSpeedup.toFixed(2)}x`,
    `single_dispatch_smollm=${current.projectionSmollmPromptSingleDispatchSpeedup.toFixed(2)}x`,
    `failures=${current.failures.length}`,
  ].join(" ");
}

if (isFocusedQprojFilter(frontierFilter)) {
  runFocusedQprojGate();
}

if (isFocusedQrowRegionFilter(frontierFilter)) {
  runFocusedQrowRegionGate();
}

if (isFocusedSemanticFilter(frontierFilter)) {
  runFocusedSemanticGate();
}

const attempts = [];
for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
  try {
    const current = score(runBench(), attempt);
    attempts.push(current);
  } catch (err) {
    process.stderr.write(`${err.output ?? err.message}\n`);
    process.exit(1);
  }
}

const passing = attempts.filter((current) => current.failures.length === 0);
const best = chooseBest(passing.length > 0 ? passing : attempts);
const aggregate = aggregateFailures(attempts);

const line = aggregate.length === 0 ? best.line.replace("frontier bench gate: fail", "frontier bench gate: pass") : best.line;
process.stdout.write(`${line}\n`);
if (aggregate.length === 0 && passing.length === 0) {
  process.stdout.write(`frontier bench aggregate: pass across ${attempts.length} noisy attempts\n`);
}
if (best.attempt > 1) {
  process.stdout.write(`frontier bench retries: ${best.attempt - 1} noisy attempt(s) below best evidence\n`);
}
if (aggregate.length !== 0) {
  process.stderr.write(`frontier bench attempt diagnostics:\n${attempts.map(attemptDiagnostic).join("\n")}\n`);
  process.stderr.write(`${aggregate.join("; ")}\n`);
  process.exit(1);
}
process.exit(0);
