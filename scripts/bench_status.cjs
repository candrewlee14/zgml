"use strict";

const { spawnSync } = require("node:child_process");
const { existsSync, readFileSync, readdirSync } = require("node:fs");
const { join } = require("node:path");

const trendGateEnabled = process.argv.includes("--trend-gate");
const substrateGateEnabled = process.argv.includes("--substrate-gate");
const trendGateFloors = {
  prompt: 0.80,
  decode: 0.90,
};
const baselineGateFloor = 1.00;
const parityTarget = 0.90;

const baselineArtifacts = [
  "benchmarks/baselines/smollm-m5pro-p128-g200-r3.json",
  "benchmarks/baselines/smollm-stencil-p128.json",
];
const fullRunArtifactPattern = /^smollm-\d{8}T\d{6}Z(?:-\d+)?-p128-g200-r3\.json$/;
const ggmlSmokeArtifactPattern = /^smollm-\d{8}T\d{6}Z(?:-\d+)?-p128-g40-r1\.json$/;
const pytorchArtifactPattern = /^pytorch-\d{8}T\d{6}Z-\d+\.json$/;
const q8PromptArtifactPattern = /^q8-prompt-\d{8}T\d{6}Z-\d+\.json$/;
const frontierArtifactPattern = /^frontier-qsemantic-\d{8}T\d{6}Z-\d+\.json$/;
const qsemanticThroughputArtifactPattern = /^frontier-qsemantic-throughput-\d{8}T\d{6}Z-\d+\.json$/;
const qprojFrontierArtifactPattern = /^frontier-qproj-\d{8}T\d{6}Z-\d+\.json$/;
const pytorchFocusKeys = [
  "linear_batched",
  "lazy_matmul_add_gelu_batched",
  "lazy_rms_silu_ffn_batched",
  "rms_gelu_linear_batched",
  "log_softmax_classifier_batched",
  "lazy_token_head_batched",
];
const pytorchBroadKeys = [
  "linear_batched",
  "lazy_matmul_add_gelu_batched",
  "lazy_mlp_batched",
  "lazy_rms_silu_ffn_batched",
  "max_pool2d_batched",
  "avg_pool2d_batched",
  "rms_gelu_linear_batched",
  "softmax_classifier_batched",
  "log_softmax_classifier_batched",
  "lazy_token_head_batched",
];

function latestFullRunArtifact() {
  const artifacts = fullRunArtifacts();
  return artifacts.at(-1) ?? null;
}

function isAcceptedFullRunArtifact(path) {
  try {
    const data = JSON.parse(readFileSync(path, "utf8"));
    if (!(data?.benchmark === "smollm-135m" &&
      data?.lanes &&
      data?.outputs &&
      data?.summary &&
      data?.gates &&
      data.gates.required_pass === true &&
      !data.gates.preflight)) {
      return false;
    }
    const baseline = baselineEvidence(path);
    return !!baseline &&
      baseline.rows.every((row) => row.ok) &&
      baseline.structuralRows.every((row) => row.ok);
  } catch {
    return false;
  }
}

function fullRunArtifacts() {
  const dir = "bench-results";
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((name) => fullRunArtifactPattern.test(name))
    .sort()
    .map((name) => join(dir, name))
    .filter(isAcceptedFullRunArtifact);
}

function quarantinedFullRunArtifacts() {
  const dir = join("bench-results", "failed");
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((name) => fullRunArtifactPattern.test(name))
    .sort()
    .map((name) => join(dir, name));
}

function ggmlSmokeArtifacts() {
  const dirs = ["bench-results", join("bench-results", "failed")];
  const paths = [];
  for (const dir of dirs) {
    if (!existsSync(dir)) continue;
    for (const name of readdirSync(dir)) {
      if (ggmlSmokeArtifactPattern.test(name)) paths.push(join(dir, name));
    }
  }
  return paths.sort((left, right) => compactName(left).localeCompare(compactName(right)));
}

function latestGgmlSmokeArtifact() {
  return ggmlSmokeArtifacts().at(-1) ?? null;
}

function pytorchComparisonArtifacts() {
  const dir = join("bench-results", "pytorch");
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((name) => pytorchArtifactPattern.test(name))
    .sort()
    .map((name) => join(dir, name))
    .filter((path) => {
      try {
        const data = readJson(path);
        return data?.schema === "zgml.pytorch-comparison.v1" &&
          data?.config &&
          data?.worst &&
          data?.ratioStats &&
          Array.isArray(data?.config?.activeComparisonKeys);
      } catch {
        return false;
      }
    });
}

function latestPytorchComparisonArtifact() {
  return latestPytorchFocusArtifact() ?? latestPytorchBroadArtifact() ?? pytorchComparisonArtifacts().at(-1) ?? null;
}

function latestRawPytorchComparisonArtifact() {
  return pytorchComparisonArtifacts().at(-1) ?? null;
}

function isPytorchFocusArtifact(path) {
  return isPytorchKeySetArtifact(path, pytorchFocusKeys);
}

function isPytorchBroadArtifact(path) {
  return isPytorchKeySetArtifact(path, pytorchBroadKeys);
}

function isPytorchSteadyArtifact(path) {
  try {
    const data = readJson(path);
    return Number.isInteger(data?.config?.attempts) &&
      data.config.attempts >= 3 &&
      Number(data?.config?.minTimingMs) >= 150 &&
      Number(data?.config?.moduleProgramMinTimingMs) >= 150;
  } catch {
    return false;
  }
}

function isPytorchKeySetArtifact(path, expectedKeys) {
  try {
    const data = readJson(path);
    const keys = data?.config?.activeComparisonKeys;
    return Array.isArray(keys) &&
      keys.length === expectedKeys.length &&
      expectedKeys.every((key, index) => keys[index] === key);
  } catch {
    return false;
  }
}

function latestPytorchFocusArtifact() {
  const artifacts = pytorchComparisonArtifacts().filter(isPytorchFocusArtifact);
  return artifacts.filter(isPytorchSteadyArtifact).at(-1) ?? artifacts.at(-1) ?? null;
}

function latestPytorchBroadArtifact() {
  const artifacts = pytorchComparisonArtifacts().filter(isPytorchBroadArtifact);
  return artifacts.filter(isPytorchSteadyArtifact).at(-1) ?? artifacts.at(-1) ?? null;
}

function q8PromptCandidateArtifacts() {
  const dir = join("bench-results", "q8-prompt");
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((name) => q8PromptArtifactPattern.test(name))
    .sort()
    .map((name) => join(dir, name))
    .filter((path) => {
      try {
        const data = readJson(path);
        return data?.schema === "zgml.q8-prompt-candidate.v1" &&
          data?.config &&
          data?.lanes &&
          data?.structural;
      } catch {
        return false;
      }
    });
}

function isQ8PromptSteadyArtifact(path) {
  try {
    const data = readJson(path);
    const attempts = Number(data?.config?.attempts);
    const lanes = data?.config?.measuredLanes;
    return Number.isInteger(attempts) &&
      attempts >= 3 &&
      Array.isArray(lanes) &&
      ["command", "two_phase", "semantic"].every((lane) => lanes.includes(lane));
  } catch {
    return false;
  }
}

function latestQ8PromptCandidateArtifact() {
  const artifacts = q8PromptCandidateArtifacts();
  return artifacts.filter(isQ8PromptSteadyArtifact).at(-1) ?? artifacts.at(-1) ?? null;
}

function latestRawQ8PromptCandidateArtifact() {
  return q8PromptCandidateArtifacts().at(-1) ?? null;
}

function q8PromptFreshnessStatusLine(selectedPath, rawPath) {
  if (!selectedPath || !rawPath || selectedPath === rawPath) return null;
  let latest = "summary=unreadable";
  try {
    const data = readJson(rawPath);
    const semanticThroughput = typeof data?.throughput?.semantic === "string" ? data.throughput.semantic : "unknown";
    const semanticStats = q8LaneSpeedupStats(data, "semantic", "semanticSpeedup");
    const semanticSpeedup = formatRatio(data?.lanes?.semantic?.speedup);
    const semanticMedian = formatRatio(semanticStats?.median);
    const semanticWorst = formatRatio(semanticStats?.worst);
    const semanticSpills = data?.lanes?.semantic?.tiledSpills ?? "n/a";
    const semanticSpillInput = data?.lanes?.semantic?.tiledSpillInput ?? "n/a";
    const semanticSpillK = Number(data?.lanes?.semantic?.tiledSpills) > 0
      ? Number(data?.lanes?.semantic?.tiledSpillInput) / Number(data.lanes.semantic.tiledSpills)
      : NaN;
    const semanticOutputSpills = data?.lanes?.semantic?.tiledOutputSpills ?? "n/a";
    const attempts = Number.isInteger(data?.config?.attempts) ? data.config.attempts : "n/a";
    const lanes = Array.isArray(data?.config?.measuredLanes) ? data.config.measuredLanes.join(",") : "unknown";
    latest = `semantic=${semanticThroughput} semantic_speedup=${semanticSpeedup} semantic_median=${semanticMedian} semantic_worst=${semanticWorst} semantic_spills=${semanticSpills} semantic_spill_input=${semanticSpillInput} semantic_spill_k=${formatNumber(semanticSpillK, 0)} semantic_output_spills=${semanticOutputSpills} attempts=${attempts} lanes=${lanes}`;
  } catch {
    latest = "summary=unreadable";
  }
  return `q8-prompt-latest-results: newest=${compactName(rawPath)} selected=${compactName(selectedPath)} reason=prefer_steady_attempts ${latest}`;
}

function frontierArtifacts() {
  const dir = join("bench-results", "frontier");
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((name) => frontierArtifactPattern.test(name))
    .sort()
    .map((name) => join(dir, name))
    .filter((path) => {
      try {
        const data = readJson(path);
        return data?.schema === "zgml.frontier-qsemantic.v1" &&
          data?.kind === "qsemantic" &&
          data?.config &&
          data?.fullPrefill &&
          data?.smollmPrompt;
      } catch {
        return false;
      }
    });
}

function latestFrontierArtifact() {
  const artifacts = frontierArtifacts();
  return artifacts.filter(isFrontierSteadyArtifact).at(-1) ?? artifacts.at(-1) ?? null;
}

function latestRawFrontierArtifact() {
  return frontierArtifacts().at(-1) ?? null;
}

function qsemanticThroughputArtifacts() {
  const dir = join("bench-results", "frontier");
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((name) => qsemanticThroughputArtifactPattern.test(name))
    .sort()
    .map((name) => join(dir, name))
    .filter((path) => {
      try {
        const data = readJson(path);
        return data?.schema === "zgml.frontier-qsemantic-throughput.v1" &&
          data?.kind === "qsemantic-throughput" &&
          data?.fullPrefill &&
          data?.smollmPrompt;
      } catch {
        return false;
      }
    });
}

function latestQsemanticThroughputArtifact() {
  return qsemanticThroughputArtifacts().at(-1) ?? null;
}

function isFrontierSteadyArtifact(path) {
  try {
    const data = readJson(path);
    return Number.isInteger(data?.attempts) && data.attempts >= 3;
  } catch {
    return false;
  }
}

function frontierFreshnessStatusLine(selectedPath, rawPath) {
  if (!selectedPath || !rawPath || selectedPath === rawPath) return null;
  let latest = "summary=unreadable";
  try {
    const data = readJson(rawPath);
    const target = typeof data?.targetThroughputStatus === "string" ? data.targetThroughputStatus : "unknown";
    const throughputCandidate = typeof data?.throughputCandidateStatus === "string" ? data.throughputCandidateStatus : "unknown";
    const fullPrefillCandidateVsDefault = formatRatio(data?.fullPrefill?.throughputCandidateVsDefault);
    const smollmPromptCandidateVsDefault = formatRatio(data?.smollmPrompt?.throughputCandidateVsDefault);
    const fullPrefillTargetVsDefault = formatRatio(data?.fullPrefill?.targetSpeedup && data?.fullPrefill?.speedup ? data.fullPrefill.targetSpeedup / data.fullPrefill.speedup : null);
    const smollmPromptTargetVsDefault = formatRatio(data?.smollmPrompt?.targetSpeedup && data?.smollmPrompt?.speedup ? data.smollmPrompt.targetSpeedup / data.smollmPrompt.speedup : null);
    const attempts = Number.isInteger(data?.attempts) ? data.attempts : "n/a";
    const source = typeof data?.source?.label === "string" ? data.source.label : "unknown";
    const fullTargetTileGroups = data?.fullPrefill?.targetTileParallelGroups ?? "n/a";
    const smollmTargetTileGroups = data?.smollmPrompt?.targetTileParallelGroups ?? "n/a";
    const fullTargetTileShape = qsemanticTargetTileShape(data?.fullPrefill);
    const smollmTargetTileShape = qsemanticTargetTileShape(data?.smollmPrompt);
    const fullCandidateTileGroups = data?.fullPrefill?.throughputCandidateTileParallelGroups ?? "n/a";
    const smollmCandidateTileGroups = data?.smollmPrompt?.throughputCandidateTileParallelGroups ?? "n/a";
    const fullCandidateFinalizeTileGroups = data?.fullPrefill?.throughputCandidateFinalizeTileGroups ?? "n/a";
    const smollmCandidateFinalizeTileGroups = data?.smollmPrompt?.throughputCandidateFinalizeTileGroups ?? "n/a";
    const fullCandidateFinalizeElements = data?.fullPrefill?.throughputCandidateFinalizeElements ?? "n/a";
    const smollmCandidateFinalizeElements = data?.smollmPrompt?.throughputCandidateFinalizeElements ?? "n/a";
    const fullTileGap = formatRatio(qsemanticTileGap(data?.fullPrefill));
    const smollmTileGap = formatRatio(qsemanticTileGap(data?.smollmPrompt));
    const fullTargetSerialPerTile = data?.fullPrefill?.targetRowSerialDotOpsPerTileGroup ?? "n/a";
    const smollmTargetSerialPerTile = data?.smollmPrompt?.targetRowSerialDotOpsPerTileGroup ?? "n/a";
    latest = `target=${target} throughput_candidate=${throughputCandidate} candidate_vs_default=full:${fullPrefillCandidateVsDefault},smollm:${smollmPromptCandidateVsDefault} target_vs_default=full:${fullPrefillTargetVsDefault},smollm:${smollmPromptTargetVsDefault} target_tile_groups=full:${fullTargetTileGroups},smollm:${smollmTargetTileGroups} target_tile_shape=full:${fullTargetTileShape},smollm:${smollmTargetTileShape} candidate_tile_groups=full:${fullCandidateTileGroups},smollm:${smollmCandidateTileGroups} candidate_finalize_groups=full:${fullCandidateFinalizeTileGroups},smollm:${smollmCandidateFinalizeTileGroups} candidate_finalize_elements=full:${fullCandidateFinalizeElements},smollm:${smollmCandidateFinalizeElements} tile_gap=full:${fullTileGap},smollm:${smollmTileGap} target_serial_per_tile=full:${fullTargetSerialPerTile},smollm:${smollmTargetSerialPerTile} attempts=${attempts} source=${source}`;
  } catch {
    // Keep the freshness signal even if the newest artifact cannot be read.
  }
  return `frontier-latest-results: newest=${compactName(rawPath)} selected=${compactName(selectedPath)} reason=prefer_steady_attempts ${latest}`;
}

function qsemanticThroughputStatusLine(path) {
  if (!path) {
    return "qsemantic-throughput-results: no local throughput-only qsemantic artifact found; run npm run dev:perf:next:qsemantic-throughput:run";
  }
  let data;
  try {
    data = readJson(path);
  } catch {
    return `qsemantic-throughput-results: latest=${compactName(path)} unreadable`;
  }
  const status = typeof data?.status === "string" ? data.status : "unknown";
  const selectedAttempt = Number.isInteger(data?.selectedAttempt) ? data.selectedAttempt : "n/a";
  const attempts = Number.isInteger(data?.attempts) ? data.attempts : "n/a";
  const fullPrefillSpeedup = Number(data?.fullPrefill?.speedup);
  const smollmPromptSpeedup = Number(data?.smollmPrompt?.speedup);
  const fullPrefillMedian = formatRatio(data?.speedupStats?.fullPrefill?.median);
  const smollmPromptMedian = formatRatio(data?.speedupStats?.smollmPrompt?.median);
  const fullPrefillWorst = formatRatio(data?.speedupStats?.fullPrefill?.worst);
  const smollmPromptWorst = formatRatio(data?.speedupStats?.smollmPrompt?.worst);
  const fullPrefill = formatRatio(data?.fullPrefill?.speedup);
  const smollmPrompt = formatRatio(data?.smollmPrompt?.speedup);
  const gate = Number.isFinite(fullPrefillSpeedup) && Number.isFinite(smollmPromptSpeedup) && fullPrefillSpeedup >= 1 && smollmPromptSpeedup >= 1
    ? "ready"
    : "below_default";
  const bottleneck = Number.isFinite(fullPrefillSpeedup) && Number.isFinite(smollmPromptSpeedup)
    ? fullPrefillSpeedup <= smollmPromptSpeedup
      ? "full_prefill"
      : "smollm_prompt"
    : "unknown";
  const fullDispatches = data?.fullPrefill?.runtimeDispatches ?? "n/a";
  const smollmDispatches = data?.smollmPrompt?.runtimeDispatches ?? "n/a";
  const fullSemanticCount = data?.fullPrefill?.semanticFfnSublayerCount ?? "n/a";
  const smollmSemanticCount = data?.smollmPrompt?.semanticFfnSublayerCount ?? "n/a";
  const fullSemanticTileGroups = data?.fullPrefill?.semanticTileParallelGroups ?? data?.fullPrefill?.tileGroups ?? "n/a";
  const smollmSemanticTileGroups = data?.smollmPrompt?.semanticTileParallelGroups ?? data?.smollmPrompt?.tileGroups ?? "n/a";
  const fullSemanticRowSerial = data?.fullPrefill?.semanticRowSerialDotOpsPerTileGroup ?? "n/a";
  const smollmSemanticRowSerial = data?.smollmPrompt?.semanticRowSerialDotOpsPerTileGroup ?? "n/a";
  const fullSemanticTotalRowSerial = data?.fullPrefill?.semanticTotalRowSerialDotOpsPerTileGroup ?? "n/a";
  const smollmSemanticTotalRowSerial = data?.smollmPrompt?.semanticTotalRowSerialDotOpsPerTileGroup ?? "n/a";
  const semanticSerialGap = Number(fullSemanticRowSerial) > 0 && Number.isFinite(Number(smollmSemanticRowSerial))
    ? formatRatio(Number(smollmSemanticRowSerial) / Number(fullSemanticRowSerial))
    : "n/a";
  const fullWidthLaneSlots = data?.fullPrefill?.semanticWidthLaneSlots ?? "n/a";
  const smollmWidthLaneSlots = data?.smollmPrompt?.semanticWidthLaneSlots ?? "n/a";
  const fullWidthLaneUtilization = data?.fullPrefill?.semanticWidthLaneUtilizationX1000 ?? "n/a";
  const smollmWidthLaneUtilization = data?.smollmPrompt?.semanticWidthLaneUtilizationX1000 ?? "n/a";
  const semanticWidthSlotGap = Number(fullWidthLaneSlots) > 0 && Number.isFinite(Number(smollmWidthLaneSlots))
    ? formatRatio(Number(smollmWidthLaneSlots) / Number(fullWidthLaneSlots))
    : "n/a";
  const fullThreadLaneSlots = data?.fullPrefill?.semanticThreadLaneSlots ?? "n/a";
  const smollmThreadLaneSlots = data?.smollmPrompt?.semanticThreadLaneSlots ?? "n/a";
  const fullThreadLaneUtilization = data?.fullPrefill?.semanticThreadLaneUtilizationX1000 ?? "n/a";
  const smollmThreadLaneUtilization = data?.smollmPrompt?.semanticThreadLaneUtilizationX1000 ?? "n/a";
  const semanticThreadSlotGap = Number(fullThreadLaneSlots) > 0 && Number.isFinite(Number(smollmThreadLaneSlots))
    ? formatRatio(Number(smollmThreadLaneSlots) / Number(fullThreadLaneSlots))
    : "n/a";
  const fullSpilledInput = data?.fullPrefill?.spilledInput ?? "n/a";
  const smollmSpilledInput = data?.smollmPrompt?.spilledInput ?? "n/a";
  const fullOutputSpills = data?.fullPrefill?.outputSpills ?? "n/a";
  const smollmOutputSpills = data?.smollmPrompt?.outputSpills ?? "n/a";
  const next = typeof data?.next === "string" ? data.next : "unknown";
  const source = typeof data?.source?.label === "string" ? data.source.label : "unknown";
  return `qsemantic-throughput-results: latest=${compactName(path)} status=${status} gate=${gate} bottleneck=${bottleneck} serial_gap=${semanticSerialGap} width_slot_gap=${semanticWidthSlotGap} thread_slot_gap=${semanticThreadSlotGap} attempt=${selectedAttempt}/${attempts} full_prefill=${fullPrefill}:median:${fullPrefillMedian}:worst:${fullPrefillWorst}:dispatches:${fullDispatches}:semantic_count:${fullSemanticCount}:semantic_tile_groups:${fullSemanticTileGroups}:row_serial_per_group:${fullSemanticRowSerial}:total_row_serial_per_group:${fullSemanticTotalRowSerial}:width_lane_slots:${fullWidthLaneSlots}:width_lane_utilization_x1000:${fullWidthLaneUtilization}:thread_lane_slots:${fullThreadLaneSlots}:thread_lane_utilization_x1000:${fullThreadLaneUtilization}:spilled_input:${fullSpilledInput}:output_spills:${fullOutputSpills} smollm_prompt=${smollmPrompt}:median:${smollmPromptMedian}:worst:${smollmPromptWorst}:dispatches:${smollmDispatches}:semantic_count:${smollmSemanticCount}:semantic_tile_groups:${smollmSemanticTileGroups}:row_serial_per_group:${smollmSemanticRowSerial}:total_row_serial_per_group:${smollmSemanticTotalRowSerial}:width_lane_slots:${smollmWidthLaneSlots}:width_lane_utilization_x1000:${smollmWidthLaneUtilization}:thread_lane_slots:${smollmThreadLaneSlots}:thread_lane_utilization_x1000:${smollmThreadLaneUtilization}:spilled_input:${smollmSpilledInput}:output_spills:${smollmOutputSpills} next=${next} source=${source}`;
}

function ggmlSmokeStatusLine(path) {
  if (!path) {
    return "ggml-smoke-results: no local p128/g40/r1 smoke artifact found; run npm run dev:perf:next:run after Q8 promotion";
  }
  let data;
  try {
    data = readJson(path);
  } catch {
    return `ggml-smoke-results: latest=${compactName(path)} unreadable`;
  }
  const q8Prompt = data?.summary?.gate_zgml?.q8_0?.prompt;
  const q8Decode = data?.summary?.gate_zgml?.q8_0?.decode;
  const q8PromptParity = data?.summary?.parity_gate_vs_llama_cpp_metal_q8_0?.prompt;
  const q8DecodeParity = data?.summary?.parity_gate_vs_llama_cpp_metal_q8_0?.decode;
  const required = data?.gates?.required_pass === true ? "required-pass" : "diagnostic";
  const promptParity = formatPct(q8PromptParity?.parity);
  const decodeParity = formatPct(q8DecodeParity?.parity);
  const promptTokS = Number(q8Prompt?.tok_s);
  const decodeTokS = Number(q8Decode?.tok_s);
  const promptCommands = q8Prompt?.commands_per_call ?? "n/a";
  const decodeCommands = q8Decode?.commands_per_call ?? "n/a";
  const promptDispatches = q8Prompt?.dispatches_per_call ?? "n/a";
  const decodeDispatches = q8Decode?.dispatches_per_call ?? "n/a";
  const fallback = q8Prompt?.fallback_ops ?? "n/a";
  const semantic = q8Prompt?.program_command_encoded_semantic_ffn_sublayer_per_call ?? 0;
  const rowChains = q8Prompt?.program_command_encoded_projection_row_chain_per_call ?? 0;
  const bridges = q8Prompt?.program_command_shape_projection_row_chain_semantic_residual_bridges ?? 0;
  const cacheGroups = q8Prompt?.program_command_encoded_projection_cache_group_per_call ?? 0;
  const promptLabel = typeof q8Prompt?.label === "string" ? q8Prompt.label.replace(/\s+/g, "_") : "unknown";
  return `ggml-smoke-results: latest=${compactName(path)} status=${required} q8_prompt=${Number.isFinite(promptTokS) ? promptTokS.toFixed(2) : "n/a"}tok/s:${promptParity}:dispatch=${promptDispatches}:commands=${promptCommands}:semantic_ffn=${semantic}:projection_row_chain=${rowChains}:semantic_bridges=${bridges}:cache_groups=${cacheGroups}:fallback=${fallback}:lane=${promptLabel} q8_decode=${Number.isFinite(decodeTokS) ? decodeTokS.toFixed(2) : "n/a"}tok/s:${decodeParity}:dispatch=${decodeDispatches}:commands=${decodeCommands}`;
}

function qprojFrontierArtifacts() {
  const dir = join("bench-results", "frontier");
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((name) => qprojFrontierArtifactPattern.test(name))
    .sort()
    .map((name) => join(dir, name))
    .filter((path) => {
      try {
        const data = readJson(path);
        return data?.schema === "zgml.frontier-qproj.v1" &&
          data?.kind === "qproj" &&
          data?.selected?.projectionChain &&
          data?.selected?.projectionGroupRegion;
      } catch {
        return false;
      }
    });
}

function latestQprojFrontierArtifact() {
  return qprojFrontierArtifacts().at(-1) ?? null;
}

const trendMetrics = [
  ["F16 pp", "prompt", "zgml_f16", "metal scheduled prefill", "prompt_tok_s"],
  ["F16 tg", "decode", "zgml_f16", "metal region decode", "decode_tok_s"],
  ["Q8 pp", "prompt", "zgml_q8_0", "metal scheduled prefill", "prompt_tok_s"],
  ["Q8 tg", "decode", "zgml_q8_0", "metal region decode", "decode_tok_s"],
];

function metricValue(data, lane, row, key) {
  const value = data?.lanes?.[lane]?.[row]?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function compactName(path) {
  return path.split("/").at(-1)?.replace(/^smollm-/, "").replace(/\.json$/, "") ?? path;
}

function formatPct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatMultiplier(value) {
  return `${value.toFixed(1)}x`;
}

function formatRatio(value) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)}x` : "n/a";
}

function formatNumber(value, digits = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

function qsemanticTileGap(lane) {
  const stored = Number(lane?.tileParallelGroupGap);
  if (Number.isFinite(stored)) return stored;
  const target = Number(lane?.targetTileParallelGroups);
  const candidate = Number(lane?.throughputCandidateTileParallelGroups);
  return Number.isFinite(target) && Number.isFinite(candidate) && candidate !== 0 ? target / candidate : null;
}

function qsemanticTargetTileShape(lane) {
  const rows = lane?.targetTileRowGroups;
  const hidden = lane?.targetTileHiddenTiles;
  const output = lane?.targetTileOutputTiles;
  if (rows == null || hidden == null || output == null) return "n/a";
  return `${rows}x${hidden}x${output}`;
}

function formatMs(value) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(4)}ms` : "n/a";
}

function formatUs(value) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 1000).toFixed(2)}us` : "n/a";
}

function attemptRatioStats(data, field) {
  if (!Array.isArray(data?.attempts)) return null;
  const values = data.attempts
    .map((row) => Number(row?.[field]))
    .filter((value) => Number.isFinite(value))
    .sort((left, right) => left - right);
  if (values.length === 0) return null;
  return {
    best: values[values.length - 1],
    median: values[Math.floor(values.length / 2)],
    worst: values[0],
  };
}

function q8LaneSpeedupStats(data, lane, attemptField) {
  const stats = data?.lanes?.[lane]?.speedupStats;
  if (stats && typeof stats === "object") {
    const best = Number(stats.best);
    const median = Number(stats.median);
    const worst = Number(stats.worst);
    if (Number.isFinite(best) || Number.isFinite(median) || Number.isFinite(worst)) {
      return {
        best: Number.isFinite(best) ? best : null,
        median: Number.isFinite(median) ? median : null,
        worst: Number.isFinite(worst) ? worst : null,
      };
    }
  }
  return attemptRatioStats(data, attemptField);
}

function ratioPassSummary(entries, minRatio) {
  if (!entries || typeof entries !== "object") return null;
  const rows = Object.entries(entries).flatMap(([key, value]) => {
    const ratio = Number(value?.zgmlVsPytorch ?? value?.median);
    return Number.isFinite(ratio) ? [{ key, ratio }] : [];
  });
  if (rows.length === 0 || !Number.isFinite(minRatio)) return null;
  const misses = rows.filter((row) => row.ratio < minRatio).map((row) => row.key);
  return {
    pass: rows.length - misses.length,
    total: rows.length,
    misses,
  };
}

function trendEvidence(latestPath) {
  const runs = fullRunArtifacts();
  if (!latestPath || runs.length === 0) return null;
  let latestData;
  try {
    latestData = JSON.parse(readFileSync(latestPath, "utf8"));
  } catch {
    return null;
  }
  const rows = [];
  for (const [label, phase, lane, row, key] of trendMetrics) {
    const latestValue = metricValue(latestData, lane, row, key);
    let best = null;
    for (const path of runs) {
      const data = JSON.parse(readFileSync(path, "utf8"));
      const value = metricValue(data, lane, row, key);
      if (value === null) continue;
      if (best === null || value > best.value) best = { path, value };
    }
    if (latestValue === null || best === null) {
      rows.push({ label, phase, ok: false, reason: "missing" });
      continue;
    }
    rows.push({
      label,
      phase,
      latest: latestValue,
      best: best.value,
      ratio: latestValue / best.value,
      bestPath: best.path,
      floor: trendGateFloors[phase],
      ok: latestValue / best.value >= trendGateFloors[phase],
    });
  }
  return { latestPath, runs, rows };
}

function pytorchComparisonStatusLine(path) {
  if (!path) {
    return "pytorch-results: no local PyTorch comparison artifact found; run npm run bench:pytorch for CPU parity evidence";
  }
  let data;
  try {
    data = readJson(path);
  } catch {
    return `pytorch-results: latest=${compactName(path)} unreadable`;
  }
  const keys = Array.isArray(data?.config?.activeComparisonKeys) ? data.config.activeComparisonKeys.join(",") : "unknown";
  const medians = data?.ratioStats && typeof data.ratioStats === "object"
    ? Object.entries(data.ratioStats).map(([key, stats]) => `${key}:${formatRatio(stats?.median)}`).join(",")
    : "missing";
  const status = data?.comparisonReady === true ? "pass" : "miss";
  const medianStatus = data?.medianParityReady === true ? "pass" : "miss";
  const worst = data?.worst?.key ? `${data.worst.key}:${formatRatio(data.worst.ratio)}` : "missing";
  const worstGap = data?.worst?.key
    ? `${data.worst.key}:zgml=${formatMs(data.worst.zgmlMs)},pytorch=${formatMs(data.worst.pytorchMs)},delta=${formatUs(Number(data.worst.zgmlMs) - Number(data.worst.pytorchMs))}`
    : "missing";
  const minRatio = Number(data?.config?.minRatio ?? 1);
  const selectedFallback = ratioPassSummary(data?.selectedRatios, minRatio);
  const medianFallback = ratioPassSummary(data?.ratioStats, minRatio);
  const selectedLanePass = Number.isInteger(data?.lanePass?.selected) && Number.isInteger(data?.lanePass?.selectedTotal)
    ? `${data.lanePass.selected}/${data.lanePass.selectedTotal}`
    : selectedFallback ? `${selectedFallback.pass}/${selectedFallback.total}` : "n/a";
  const medianLanePass = Number.isInteger(data?.lanePass?.median) && Number.isInteger(data?.lanePass?.medianTotal)
    ? `${data.lanePass.median}/${data.lanePass.medianTotal}`
    : medianFallback ? `${medianFallback.pass}/${medianFallback.total}` : "n/a";
  const laneMiss = Array.isArray(data?.lanePass?.selectedMisses) && data.lanePass.selectedMisses.length !== 0
    ? data.lanePass.selectedMisses.join(",")
    : selectedFallback && selectedFallback.misses.length !== 0 ? selectedFallback.misses.join(",") : "none";
  const medianLaneMiss = Array.isArray(data?.lanePass?.medianMisses) && data.lanePass.medianMisses.length !== 0
    ? data.lanePass.medianMisses.join(",")
    : medianFallback && medianFallback.misses.length !== 0 ? medianFallback.misses.join(",") : "none";
  const native = typeof data?.native?.label === "string" ? data.native.label : "unknown";
  const torch = typeof data?.pytorchVersion === "string" ? data.pytorchVersion : "unknown";
  const selectedAttempt = Number.isInteger(data?.selectedAttempt) ? data.selectedAttempt : "n/a";
  const attempts = Number.isInteger(data?.config?.attempts) ? data.config.attempts : "n/a";
  const timing = typeof data?.config?.zgmlTimingMetric === "string" ? data.config.zgmlTimingMetric : "unknown";
  return `pytorch-results: latest=${compactName(path)} status=${status} median=${medianStatus} worst=${worst} gap=${worstGap} lane_pass=${selectedLanePass} median_lane_pass=${medianLanePass} lane_miss=${laneMiss} median_lane_miss=${medianLaneMiss} attempt=${selectedAttempt}/${attempts} native=${native} torch=${torch} timing=${timing} keys=${keys} ratio_median=${medians}`;
}

function pytorchFocusStatusLine(path, latestPath) {
  if (!path || path === latestPath) return null;
  return pytorchComparisonStatusLine(path).replace("pytorch-results: latest=", "pytorch-focus-results: latest=");
}

function pytorchBroadStatusLine(path, latestPath) {
  if (!path || path === latestPath) return null;
  return pytorchComparisonStatusLine(path).replace("pytorch-results: latest=", "pytorch-broad-results: latest=");
}

function pytorchFreshnessStatusLine(selectedPath, rawPath) {
  if (!selectedPath || !rawPath || selectedPath === rawPath) return null;
  let data;
  try {
    data = readJson(rawPath);
  } catch {
    return `pytorch-latest-results: newest=${compactName(rawPath)} selected=${compactName(selectedPath)} reason=prefer_focus_keyset unreadable`;
  }
  const status = data?.comparisonReady === true ? "pass" : "miss";
  const medianStatus = data?.medianParityReady === true ? "pass" : "miss";
  const worst = data?.worst?.key ? `${data.worst.key}:${formatRatio(data.worst.ratio)}` : "missing";
  const worstGap = data?.worst?.key
    ? `${data.worst.key}:zgml=${formatMs(data.worst.zgmlMs)},pytorch=${formatMs(data.worst.pytorchMs)},delta=${formatUs(Number(data.worst.zgmlMs) - Number(data.worst.pytorchMs))}`
    : "missing";
  const medians = data?.ratioStats && typeof data.ratioStats === "object"
    ? Object.entries(data.ratioStats).map(([key, stats]) => `${key}:${formatRatio(stats?.median)}`).join(",")
    : "missing";
  const attempts = Number.isInteger(data?.config?.attempts) ? data.config.attempts : "n/a";
  const native = typeof data?.native?.label === "string" ? data.native.label : "unknown";
  const timing = typeof data?.config?.zgmlTimingMetric === "string" ? data.config.zgmlTimingMetric : "unknown";
  const keyCount = Array.isArray(data?.config?.activeComparisonKeys) ? data.config.activeComparisonKeys.length : "n/a";
  return `pytorch-latest-results: newest=${compactName(rawPath)} selected=${compactName(selectedPath)} reason=prefer_focus_keyset status=${status} median=${medianStatus} worst=${worst} gap=${worstGap} attempts=${attempts} native=${native} timing=${timing} keys=${keyCount} ratio_median=${medians}`;
}

function q8PromptCandidateStatusLine(path) {
  if (!path) {
    return "q8-prompt-results: no local Q8 prompt candidate artifact found; run npm run bench:q8-prompt-viable for full-model Q8 evidence";
  }
  let data;
  try {
    data = readJson(path);
  } catch {
    return `q8-prompt-results: latest=${compactName(path)} unreadable`;
  }
  const lanes = Array.isArray(data?.config?.measuredLanes) ? data.config.measuredLanes.join(",") : "unknown";
  const status = typeof data?.status === "string" ? data.status : "unknown";
  const commandSpeedup = formatRatio(data?.lanes?.command?.speedup);
  const twoPhaseSpeedup = formatRatio(data?.lanes?.twoPhase?.speedup);
  const semanticSpeedup = formatRatio(data?.lanes?.semantic?.speedup);
  const twoPhaseStats = q8LaneSpeedupStats(data, "twoPhase", "twoPhaseSpeedup");
  const semanticStats = q8LaneSpeedupStats(data, "semantic", "semanticSpeedup");
  const semanticThroughput = typeof data?.throughput?.semantic === "string" ? data.throughput.semantic : "unknown";
  const semanticStructuralSelected = (data?.lanes?.semantic?.structuralSelected ?? data?.lanes?.semantic?.selected) === true ? "yes" : "off";
  const semanticThroughputReady = (data?.lanes?.semantic?.throughputReady ?? semanticThroughput === "ready") === true ? "yes" : "off";
  const semanticShape = `${data?.lanes?.semantic?.projectionPairs ?? "n/a"}->${data?.lanes?.semantic?.projectionRowChains ?? "n/a"}`;
  const commandShape = `${data?.lanes?.command?.commands ?? "n/a"}`;
  const semanticSpills = data?.lanes?.semantic?.tiledSpills ?? "n/a";
  const semanticSpillInput = data?.lanes?.semantic?.tiledSpillInput ?? "n/a";
  const semanticSpillK = Number(data?.lanes?.semantic?.tiledSpills) > 0
    ? Number(data?.lanes?.semantic?.tiledSpillInput) / Number(data.lanes.semantic.tiledSpills)
    : NaN;
  const semanticOutputSpills = data?.lanes?.semantic?.tiledOutputSpills ?? "n/a";
  const attempts = Number.isInteger(data?.config?.attempts) ? data.config.attempts : "n/a";
  const source = typeof data?.source?.label === "string" ? data.source.label : "unknown";
  const pairDefaults = data?.config?.pairDefaults === true ? "yes" : "no";
  const baselineNoise = formatRatio(data?.baselineNoise?.maxOverMin);
  const defaultPolicies = Array.isArray(data?.attempts)
    ? [...new Set(data.attempts.map((row) => row?.defaultPromptPolicy).filter((value) => typeof value === "string"))].join(",") || "unknown"
    : "unknown";
  return `q8-prompt-results: latest=${compactName(path)} status=${status} semantic=${semanticThroughput} semantic_structural_selected=${semanticStructuralSelected} semantic_throughput_ready=${semanticThroughputReady} command_speedup=${commandSpeedup} two_phase_speedup=${twoPhaseSpeedup} two_phase_median=${formatRatio(twoPhaseStats?.median)} two_phase_worst=${formatRatio(twoPhaseStats?.worst)} semantic_speedup=${semanticSpeedup} semantic_median=${formatRatio(semanticStats?.median)} semantic_worst=${formatRatio(semanticStats?.worst)} command_commands=${commandShape} semantic_pair_to_row=${semanticShape} semantic_spills=${semanticSpills} semantic_spill_input=${semanticSpillInput} semantic_spill_k=${formatNumber(semanticSpillK, 0)} semantic_output_spills=${semanticOutputSpills} attempts=${attempts} lanes=${lanes} pair_defaults=${pairDefaults} default_policy=${defaultPolicies} baseline_noise=${baselineNoise} source=${source}`;
}

function frontierStatusLine(path, pressurePath = path) {
  if (!path) {
    return "frontier-results: no local qsemantic artifact found; run npm run bench:frontier:qsemantic for Q8 semantic frontier evidence";
  }
  let data;
  try {
    data = readJson(path);
  } catch {
    return `frontier-results: latest=${compactName(path)} unreadable`;
  }
  let pressureData = data;
  if (pressurePath && pressurePath !== path) {
    try {
      pressureData = readJson(pressurePath);
    } catch {
      pressureData = data;
    }
  }
  const status = typeof data?.status === "string" ? data.status : "unknown";
  const target = typeof data?.targetThroughputStatus === "string" ? data.targetThroughputStatus : "unknown";
  const throughputCandidate = typeof data?.throughputCandidateStatus === "string" ? data.throughputCandidateStatus : "unknown";
  const semanticCommand = typeof data?.semanticCommandStatus === "string" ? data.semanticCommandStatus : "unknown";
  const singleDispatch = typeof data?.singleDispatchThroughputStatus === "string" ? data.singleDispatchThroughputStatus : "unknown";
  const fullPrefill = formatRatio(data?.fullPrefill?.speedup);
  const fullPrefillCandidate = formatRatio(data?.fullPrefill?.throughputCandidateSpeedup);
  const fullPrefillCandidateVsTwoPhase = formatRatio(data?.fullPrefill?.throughputCandidateVsTwoPhase);
  const smollmPrompt = formatRatio(data?.smollmPrompt?.speedup);
  const smollmPromptCandidate = formatRatio(data?.smollmPrompt?.throughputCandidateSpeedup);
  const smollmPromptCandidateVsTwoPhase = formatRatio(data?.smollmPrompt?.throughputCandidateVsTwoPhase);
  const fullTargetTileGroups = data?.fullPrefill?.targetTileParallelGroups ?? "n/a";
  const smollmTargetTileGroups = data?.smollmPrompt?.targetTileParallelGroups ?? "n/a";
  const fullTargetTileShape = qsemanticTargetTileShape(data?.fullPrefill);
  const smollmTargetTileShape = qsemanticTargetTileShape(data?.smollmPrompt);
  const fullCandidateTileGroups = data?.fullPrefill?.throughputCandidateTileParallelGroups ?? "n/a";
  const smollmCandidateTileGroups = data?.smollmPrompt?.throughputCandidateTileParallelGroups ?? "n/a";
  const fullCandidateFinalizeTileGroups = data?.fullPrefill?.throughputCandidateFinalizeTileGroups ?? "n/a";
  const smollmCandidateFinalizeTileGroups = data?.smollmPrompt?.throughputCandidateFinalizeTileGroups ?? "n/a";
  const fullCandidateFinalizeElements = data?.fullPrefill?.throughputCandidateFinalizeElements ?? "n/a";
  const smollmCandidateFinalizeElements = data?.smollmPrompt?.throughputCandidateFinalizeElements ?? "n/a";
  const fullTileGap = formatRatio(qsemanticTileGap(data?.fullPrefill));
  const smollmTileGap = formatRatio(qsemanticTileGap(data?.smollmPrompt));
  const fullTargetSerialPerTile =
    data?.fullPrefill?.targetRowSerialDotOpsPerTileGroup ??
    pressureData?.fullPrefill?.targetRowSerialDotOpsPerTileGroup ??
    "n/a";
  const smollmTargetSerialPerTile =
    data?.smollmPrompt?.targetRowSerialDotOpsPerTileGroup ??
    pressureData?.smollmPrompt?.targetRowSerialDotOpsPerTileGroup ??
    "n/a";
  const selectedAttempt = Number.isInteger(data?.selectedAttempt) ? data.selectedAttempt : "n/a";
  const attempts = Number.isInteger(data?.attempts) ? data.attempts : "n/a";
  const next = typeof data?.next === "string" ? data.next : "unknown";
  const source = typeof data?.source?.label === "string" ? data.source.label : "unknown";
  return `frontier-results: latest=${compactName(path)} status=${status} kind=qsemantic target=${target} throughput_candidate=${throughputCandidate} semantic_command=${semanticCommand} single_dispatch=${singleDispatch} attempt=${selectedAttempt}/${attempts} full_prefill=${fullPrefill} full_prefill_candidate=${fullPrefillCandidate} smollm_prompt=${smollmPrompt} smollm_prompt_candidate=${smollmPromptCandidate} vs_two_phase=full:${fullPrefillCandidateVsTwoPhase},smollm:${smollmPromptCandidateVsTwoPhase} target_tile_groups=full:${fullTargetTileGroups},smollm:${smollmTargetTileGroups} target_tile_shape=full:${fullTargetTileShape},smollm:${smollmTargetTileShape} candidate_tile_groups=full:${fullCandidateTileGroups},smollm:${smollmCandidateTileGroups} candidate_finalize_groups=full:${fullCandidateFinalizeTileGroups},smollm:${smollmCandidateFinalizeTileGroups} candidate_finalize_elements=full:${fullCandidateFinalizeElements},smollm:${smollmCandidateFinalizeElements} tile_gap=full:${fullTileGap},smollm:${smollmTileGap} target_serial_per_tile=full:${fullTargetSerialPerTile},smollm:${smollmTargetSerialPerTile} next=${next} source=${source}`;
}

function qprojFrontierStatusLine(path) {
  if (!path) {
    return "qproj-results: no local qproj artifact found; run npm run bench:frontier:qproj for projection-chain frontier evidence";
  }
  let data;
  try {
    data = readJson(path);
  } catch {
    return `qproj-results: latest=${compactName(path)} unreadable`;
  }
  const status = typeof data?.status === "string" ? data.status : "unknown";
  const selectedAttempt = Number.isInteger(data?.selectedAttempt) ? data.selectedAttempt : "n/a";
  const attempts = Number.isInteger(data?.attempts) ? data.attempts : "n/a";
  const chainFull = formatRatio(data?.selected?.projectionChain?.fullPrefill?.speedup);
  const chainSmollm = formatRatio(data?.selected?.projectionChain?.smollmPrompt?.speedup);
  const chainReady = data?.selected?.projectionChain?.fullPrefill?.candidateReady === true ? "ready" : "off";
  const groupRegionFull = formatRatio(data?.selected?.projectionGroupRegion?.fullPrefill?.speedup);
  const groupRegionSmollm = formatRatio(data?.selected?.projectionGroupRegion?.smollmPrompt?.speedup);
  const groupRegionFullDispatches = data?.selected?.projectionGroupRegion?.fullPrefill?.runtimeProjectionGroupDispatches ?? "n/a";
  const groupRegionSmollmDispatches = data?.selected?.projectionGroupRegion?.smollmPrompt?.runtimeProjectionGroupDispatches ?? "n/a";
  const next = typeof data?.next === "string" ? data.next : "unknown";
  const source = typeof data?.source?.label === "string" ? data.source.label : "unknown";
  return `qproj-results: latest=${compactName(path)} status=${status} attempt=${selectedAttempt}/${attempts} projection_chain=full:${chainFull},smollm:${chainSmollm},candidate:${chainReady} projection_group_region=full:${groupRegionFull}:dispatches:${groupRegionFullDispatches},smollm:${groupRegionSmollm}:dispatches:${groupRegionSmollmDispatches} next=${next} source=${source}`;
}

function pytorchNextTarget(path) {
  if (!path) return "pytorch=missing_artifact";
  try {
    const data = readJson(path);
    const minRatio = Number(data?.config?.minRatio ?? 1);
    const misses = [];
    if (data?.ratioStats && typeof data.ratioStats === "object") {
      for (const [key, stats] of Object.entries(data.ratioStats)) {
        const ratio = Number(stats?.median);
        if (Number.isFinite(ratio) && ratio < minRatio) misses.push({ key, ratio });
      }
    }
    misses.sort((a, b) => a.ratio - b.ratio || a.key.localeCompare(b.key));
    if (misses.length === 0) return "pytorch=none";
    return `pytorch=${misses.map((row) => `${row.key}:${formatRatio(row.ratio)}`).join(",")}`;
  } catch {
    return "pytorch=unreadable_artifact";
  }
}

function q8PromptNextTarget(path, pressurePath = path) {
  if (!path) return "q8_prompt=missing_artifact";
  try {
    const data = readJson(path);
    let pressureData = data;
    let hasFreshPressure = false;
    if (pressurePath && pressurePath !== path) {
      try {
        pressureData = readJson(pressurePath);
        hasFreshPressure = true;
      } catch {
        pressureData = data;
        hasFreshPressure = false;
      }
    }
    const semanticSpeedup = Number(data?.lanes?.semantic?.speedup);
    const semanticStats = q8LaneSpeedupStats(data, "semantic", "semanticSpeedup");
    const semanticThroughput = typeof data?.throughput?.semantic === "string" ? data.throughput.semantic : "unknown";
    const status = typeof data?.status === "string" ? data.status : "unknown";
    const pressureStatus = typeof pressureData?.status === "string" ? pressureData.status : "unknown";
    const pressureSemanticThroughput = typeof pressureData?.throughput?.semantic === "string" ? pressureData.throughput.semantic : "unknown";
    const freshStats = q8LaneSpeedupStats(pressureData, "semantic", "semanticSpeedup");
    const freshSpills = pressureData?.lanes?.semantic?.tiledSpills ?? "n/a";
    const freshSpillInput = pressureData?.lanes?.semantic?.tiledSpillInput ?? "n/a";
    const freshSpillK = Number(pressureData?.lanes?.semantic?.tiledSpills) > 0
      ? Number(pressureData?.lanes?.semantic?.tiledSpillInput) / Number(pressureData.lanes.semantic.tiledSpills)
      : NaN;
    const freshOutputSpills = pressureData?.lanes?.semantic?.tiledOutputSpills ?? "n/a";
    const fresh = hasFreshPressure
      ? `:fresh=best:${formatRatio(pressureData?.lanes?.semantic?.speedup)},median:${formatRatio(freshStats?.median)},worst:${formatRatio(freshStats?.worst)},spills:${freshSpills},spill_k:${formatNumber(freshSpillK, 0)},spill_input:${freshSpillInput},output_spills:${freshOutputSpills}`
      : "";
    if (pressureStatus === "promoted-default" || pressureSemanticThroughput === "promoted") {
      return `q8_prompt=promoted_semantic_default:commands=${pressureData?.lanes?.semantic?.commands ?? "n/a"}:row_chains=${pressureData?.lanes?.semantic?.projectionRowChains ?? "n/a"}:spills=${freshSpills}:spill_input=${freshSpillInput}:output_spills=${freshOutputSpills}`;
    }
    if (semanticThroughput === "ready" && Number.isFinite(semanticSpeedup) && semanticSpeedup >= 1) {
      return `q8_prompt=promote_semantic_candidate:${formatRatio(semanticSpeedup)}:median=${formatRatio(semanticStats?.median)}${fresh}`;
    }
    return `q8_prompt=semantic_throughput_kernel:${status}:${semanticThroughput}:best=${formatRatio(semanticSpeedup)}:median=${formatRatio(semanticStats?.median)}:worst=${formatRatio(semanticStats?.worst)}${fresh}`;
  } catch {
    return "q8_prompt=unreadable_artifact";
  }
}

function frontierNextTargetLine(path, pressurePath = path) {
  if (!path) return "frontier=missing_artifact";
  try {
    const data = readJson(path);
    let pressureData = data;
    let hasFreshPressure = false;
    if (pressurePath && pressurePath !== path) {
      try {
        pressureData = readJson(pressurePath);
        hasFreshPressure = true;
      } catch {
        pressureData = data;
        hasFreshPressure = false;
      }
    }
    const next = typeof data?.next === "string" ? data.next : "unknown";
    const smollmDefault = Number(data?.smollmPrompt?.speedup);
    const smollmCandidateValue = Number(data?.smollmPrompt?.throughputCandidateSpeedup);
    const fullDefault = Number(data?.fullPrefill?.speedup);
    const fullCandidateValue = Number(data?.fullPrefill?.throughputCandidateSpeedup);
    const storedSmollmVsDefault = Number(data?.smollmPrompt?.throughputCandidateVsDefault);
    const storedFullVsDefault = Number(data?.fullPrefill?.throughputCandidateVsDefault);
    const storedSmollmVsTwoPhase = Number(data?.smollmPrompt?.throughputCandidateVsTwoPhase);
    const storedFullVsTwoPhase = Number(data?.fullPrefill?.throughputCandidateVsTwoPhase);
    const smollmCandidate = formatRatio(smollmCandidateValue);
    const fullCandidate = formatRatio(fullCandidateValue);
    const smollmVsDefault = Number.isFinite(storedSmollmVsDefault)
      ? formatRatio(storedSmollmVsDefault)
      : Number.isFinite(smollmCandidateValue) && Number.isFinite(smollmDefault) && smollmDefault !== 0
      ? formatRatio(smollmCandidateValue / smollmDefault)
      : "n/a";
    const fullVsDefault = Number.isFinite(storedFullVsDefault)
      ? formatRatio(storedFullVsDefault)
      : Number.isFinite(fullCandidateValue) && Number.isFinite(fullDefault) && fullDefault !== 0
      ? formatRatio(fullCandidateValue / fullDefault)
      : "n/a";
    const throughputCandidate = typeof data?.throughputCandidateStatus === "string" ? data.throughputCandidateStatus : "unknown";
    const fullTargetTileGroups = data?.fullPrefill?.targetTileParallelGroups ?? "n/a";
    const smollmTargetTileGroups = data?.smollmPrompt?.targetTileParallelGroups ?? "n/a";
    const fullTargetTileShape = qsemanticTargetTileShape(data?.fullPrefill);
    const smollmTargetTileShape = qsemanticTargetTileShape(data?.smollmPrompt);
    const fullCandidateTileGroups = data?.fullPrefill?.throughputCandidateTileParallelGroups ?? "n/a";
    const smollmCandidateTileGroups = data?.smollmPrompt?.throughputCandidateTileParallelGroups ?? "n/a";
    const fullCandidateFinalizeTileGroups = data?.fullPrefill?.throughputCandidateFinalizeTileGroups ?? "n/a";
    const smollmCandidateFinalizeTileGroups = data?.smollmPrompt?.throughputCandidateFinalizeTileGroups ?? "n/a";
    const fullCandidateFinalizeElements = data?.fullPrefill?.throughputCandidateFinalizeElements ?? "n/a";
    const smollmCandidateFinalizeElements = data?.smollmPrompt?.throughputCandidateFinalizeElements ?? "n/a";
    const fullTileGap = formatRatio(qsemanticTileGap(data?.fullPrefill));
    const smollmTileGap = formatRatio(qsemanticTileGap(data?.smollmPrompt));
    const fullTargetSerialPerTile =
      data?.fullPrefill?.targetRowSerialDotOpsPerTileGroup ??
      pressureData?.fullPrefill?.targetRowSerialDotOpsPerTileGroup ??
      "n/a";
    const smollmTargetSerialPerTile =
      data?.smollmPrompt?.targetRowSerialDotOpsPerTileGroup ??
      pressureData?.smollmPrompt?.targetRowSerialDotOpsPerTileGroup ??
      "n/a";
    const freshThroughputGate =
      pressureData?.kind === "qsemantic-throughput"
        ? Number(pressureData?.smollmPrompt?.speedup) >= 1 && Number(pressureData?.fullPrefill?.speedup) >= 1
          ? "ready"
          : "below_default"
        : "n/a";
    const freshWidth =
      pressureData?.kind === "qsemantic-throughput"
        ? `,width_util=smollm:${pressureData?.smollmPrompt?.semanticWidthLaneUtilizationX1000 ?? "n/a"},full:${pressureData?.fullPrefill?.semanticWidthLaneUtilizationX1000 ?? "n/a"}`
        : "";
    const freshStats =
      pressureData?.kind === "qsemantic-throughput"
        ? `,median=smollm:${formatRatio(pressureData?.speedupStats?.smollmPrompt?.median)},full:${formatRatio(pressureData?.speedupStats?.fullPrefill?.median)},worst=smollm:${formatRatio(pressureData?.speedupStats?.smollmPrompt?.worst)},full:${formatRatio(pressureData?.speedupStats?.fullPrefill?.worst)}`
        : "";
    const fresh = hasFreshPressure
      ? pressureData?.kind === "qsemantic-throughput"
        ? `:fresh=source:${typeof pressureData?.source?.label === "string" ? pressureData.source.label : "unknown"},throughput=smollm:${formatRatio(pressureData?.smollmPrompt?.speedup)},full:${formatRatio(pressureData?.fullPrefill?.speedup)},gate=${freshThroughputGate},spilled_input=smollm:${pressureData?.smollmPrompt?.spilledInput ?? "n/a"},full:${pressureData?.fullPrefill?.spilledInput ?? "n/a"}${freshWidth}${freshStats}`
        : `:fresh=source:${typeof pressureData?.source?.label === "string" ? pressureData.source.label : "unknown"},vs_default=smollm:${formatRatio(pressureData?.smollmPrompt?.throughputCandidateVsDefault)},full:${formatRatio(pressureData?.fullPrefill?.throughputCandidateVsDefault)}`
      : "";
    return `frontier=${next}:candidate=${throughputCandidate}:smollm=${smollmCandidate}:full=${fullCandidate}:vs_default=smollm:${smollmVsDefault},full:${fullVsDefault}:vs_two_phase=smollm:${formatRatio(storedSmollmVsTwoPhase)},full:${formatRatio(storedFullVsTwoPhase)}:target_tiles=smollm:${smollmTargetTileGroups},full:${fullTargetTileGroups}:target_shape=smollm:${smollmTargetTileShape},full:${fullTargetTileShape}:candidate_tiles=smollm:${smollmCandidateTileGroups},full:${fullCandidateTileGroups}:candidate_finalize_groups=smollm:${smollmCandidateFinalizeTileGroups},full:${fullCandidateFinalizeTileGroups}:candidate_finalize_elements=smollm:${smollmCandidateFinalizeElements},full:${fullCandidateFinalizeElements}:tile_gap=smollm:${smollmTileGap},full:${fullTileGap}:target_serial_per_tile=smollm:${smollmTargetSerialPerTile},full:${fullTargetSerialPerTile}${fresh}`;
  } catch {
    return "frontier=unreadable_artifact";
  }
}

function qprojNextTargetLine(path) {
  if (!path) return "qproj=missing_artifact";
  try {
    const data = readJson(path);
    const chainFull = Number(data?.selected?.projectionChain?.fullPrefill?.speedup);
    const chainSmollm = Number(data?.selected?.projectionChain?.smollmPrompt?.speedup);
    const chainReady = data?.selected?.projectionChain?.fullPrefill?.candidateReady === true ? "ready" : "off";
    const regionFull = Number(data?.selected?.projectionGroupRegion?.fullPrefill?.speedup);
    const regionSmollm = Number(data?.selected?.projectionGroupRegion?.smollmPrompt?.speedup);
    const status = typeof data?.status === "string" ? data.status : "unknown";
    const next = typeof data?.next === "string" ? data.next : "unknown";
    return `qproj=${status}:chain=full:${formatRatio(chainFull)},smollm:${formatRatio(chainSmollm)},candidate:${chainReady}:region=full:${formatRatio(regionFull)},smollm:${formatRatio(regionSmollm)}:next=${next}`;
  } catch {
    return "qproj=unreadable_artifact";
  }
}

function fullModelNextTarget(latestPath) {
  if (!latestPath) return "full_model=missing_artifact";
  try {
    const data = readJson(latestPath);
    const rows = selectedLaneRows(data);
    let weakest = null;
    for (const [fmt, phase, row] of rows) {
      if (!row || typeof row !== "object") continue;
      const parityRow = parityRowFor(data, fmt, phase);
      if (!parityRow || typeof parityRow.parity !== "number" || !Number.isFinite(parityRow.parity) || parityRow.parity <= 0) continue;
      const candidate = {
        label: `${fmt}/${phase}`,
        parity: parityRow.parity,
        dispatches: row.dispatches_per_call,
        commands: row.commands_per_call,
        pressureTarget: pressureReductionTarget(row),
      };
      if (!weakest || candidate.parity < weakest.parity) weakest = candidate;
    }
    if (!weakest) return "full_model=missing_parity";
    return `full_model=${weakest.label}:${formatPct(weakest.parity)}:to90=${formatMultiplier(parityTarget / weakest.parity)}:dispatch=${weakest.dispatches}:commands=${weakest.commands}:target=${weakest.pressureTarget}:next=${frontierNextTarget(weakest)}`;
  } catch {
    return "full_model=unreadable_artifact";
  }
}

function perfNextStatusLine({ latestPath, pytorchPath, q8Path, rawQ8Path, frontierPath, rawFrontierPath }) {
  const qprojPath = latestQprojFrontierArtifact();
  return [
    "perf-next:",
    fullModelNextTarget(latestPath),
    pytorchNextTarget(pytorchPath),
    q8PromptNextTarget(q8Path, rawQ8Path),
    qprojNextTargetLine(qprojPath),
    frontierNextTargetLine(frontierPath, rawFrontierPath),
  ].join(" ");
}

function trendStatusLine(latestPath) {
  const trend = trendEvidence(latestPath);
  if (!trend) return null;
  const parts = trend.rows.map((row) => {
    if (row.reason) return `${row.label}=n/a`;
    return `${row.label}=${row.latest.toFixed(2)}/${row.best.toFixed(2)} ${formatPct(row.ratio)} best=${compactName(row.bestPath)}`;
  });
  return `bench trend: ${trend.runs.length} full p128/g200/r3 runs; latest=${compactName(latestPath)}; ${parts.join("; ")}`;
}

function trendGateLine(latestPath) {
  const trend = trendEvidence(latestPath);
  if (!trend) return null;
  const failed = trend.rows.filter((row) => !row.ok);
  const parts = trend.rows.map((row) => {
    if (row.reason) return `${row.label}=missing`;
    return `${row.label}=${formatPct(row.ratio)} floor=${formatPct(row.floor)}`;
  });
  return {
    passed: failed.length === 0,
    line: `bench trend gate: ${failed.length === 0 ? "pass" : "fail"}; latest=${compactName(latestPath)}; ${parts.join("; ")}`,
  };
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function matchesExpected(value, expected) {
  return Array.isArray(expected) ? expected.includes(value) : value === expected;
}

function selectedLaneRows(data) {
  const gate = data?.summary?.gate_zgml;
  return [
    ["f16", "prompt", gate?.f16?.prompt],
    ["f16", "decode", gate?.f16?.decode],
    ["q8_0", "prompt", gate?.q8_0?.prompt],
    ["q8_0", "decode", gate?.q8_0?.decode],
  ];
}

const baselineComparisonMetrics = [
  ["F16 pp", "f16", "prompt"],
  ["F16 tg", "f16", "decode"],
  ["Q8 pp", "q8_0", "prompt"],
  ["Q8 tg", "q8_0", "decode"],
];

const substrateLaneShape = {
  "f16/prompt": { dispatches: 242, commands: 242, cachedCommandPlansPerCall: 31 },
  "f16/decode": { dispatches: 212, commands: 212, cachedCommandPlansPerCall: 31 },
  "q8_0/prompt": { dispatches: 242, commands: [181, 241], cachedCommandPlansPerCall: 30 },
  "q8_0/decode": { dispatches: 212, commands: 211, cachedCommandPlansPerCall: 30 },
};

const substrateParityFloors = {
  "f16/prompt": 0.32,
  "f16/decode": 0.31,
  "q8_0/prompt": 0.27,
  "q8_0/decode": 0.32,
};

const substratePatchShape = {
  holes: 450,
  cacheWritePosHoles: 180,
  attentionSeqKvHoles: 270,
};

const substrateRuntimePatchStencilHashes = {
  prompt: "17558208047327870709",
  decode: "14405191909906507341",
};

function rawRuntimePatchStencilHash(path, fmt, phase) {
  let source;
  try {
    source = readFileSync(path, "utf8");
  } catch {
    return null;
  }
  const gateIndex = source.indexOf('"gate_zgml"');
  if (gateIndex === -1) return null;
  const fmtIndex = source.indexOf(`"${fmt}"`, gateIndex);
  if (fmtIndex === -1) return null;
  const phaseIndex = source.indexOf(`"${phase}"`, fmtIndex);
  if (phaseIndex === -1) return null;
  const nextPhase = phase === "prompt" ? source.indexOf('"decode"', phaseIndex + 1) : -1;
  const searchEnd = nextPhase === -1 ? source.indexOf(`"${fmt === "f16" ? "q8_0" : "outputs"}"`, phaseIndex + 1) : nextPhase;
  const slice = source.slice(phaseIndex, searchEnd === -1 ? undefined : searchEnd);
  const match = slice.match(/"runtime_patch_stencil_hash"\s*:\s*([0-9]+)/);
  return match?.[1] ?? null;
}

function laneMetricFromSummary(data, fmt, phase, key) {
  const row = data?.summary?.gate_zgml?.[fmt]?.[phase];
  const value = row?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function parityRowFor(data, fmt, phase) {
  const target = fmt === "f16" ? "llama_cpp_metal_f16" : "llama_cpp_metal_q8_0";
  const rows = data?.gates?.parity?.rows;
  if (!Array.isArray(rows)) return null;
  return rows.find((row) => row?.target === target && row?.phase === phase) ?? null;
}

function commandBreakdown(row, limit = 3) {
  if (!row || typeof row !== "object") return "missing";
  const entries = Object.entries(row)
    .filter(([key, value]) =>
      key.startsWith("program_command_dispatches_") &&
      key.endsWith("_per_call") &&
      typeof value === "number" &&
      Number.isFinite(value) &&
      value > 0
    )
    .map(([key, value]) => [
      key
        .replace(/^program_command_dispatches_/, "")
        .replace(/_per_call$/, ""),
      value,
    ])
    .sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])))
    .slice(0, limit);
  return entries.length ? entries.map(([name, count]) => `${name}:${count}`).join(",") : "none";
}

function commandPressure(row, limit = 3) {
  if (!row || typeof row !== "object") return "missing";
  const commands = row.commands_per_call;
  if (!(typeof commands === "number" && Number.isFinite(commands) && commands > 0)) return "missing";
  const topTotal = Object.entries(row)
    .filter(([key, value]) =>
      key.startsWith("program_command_dispatches_") &&
      key.endsWith("_per_call") &&
      typeof value === "number" &&
      Number.isFinite(value) &&
      value > 0
    )
    .map(([, value]) => value)
    .sort((a, b) => b - a)
    .slice(0, limit)
    .reduce((sum, value) => sum + value, 0);
  return `${topTotal}/${commands} ${formatPct(topTotal / commands)}`;
}

function projectionChainSidecars(row) {
  if (!row || typeof row !== "object") return "missing";
  const entries = [
    ["qmatvec_elementwise", row.projection_chain_qmatvec_elementwise_per_call],
    ["qmatmul_elementwise", row.projection_chain_qmatmul_elementwise_per_call],
    ["projection_row_chain", row.program_command_encoded_projection_row_chain_per_call],
  ].filter(([, value]) => typeof value === "number" && Number.isFinite(value) && value > 0);
  return entries.length ? entries.map(([name, count]) => `${name}:${count}`).join(",") : "none";
}

function projectionChainShape(row) {
  if (!row || typeof row !== "object") return "missing";
  const total = row.program_command_shape_projection_chains;
  const split = projectionChainSplit(row);
  const parts = [];
  if (split) parts.push(`dense:${split.dense}`, `quantized:${split.quantized}`);
  if (typeof total === "number" && Number.isFinite(total)) parts.push(`total:${total}`);
  return parts.length ? parts.join(",") : "missing";
}

function projectionChainSplit(row) {
  if (!row || typeof row !== "object") return null;
  const dense = row.program_command_shape_dense_projection_chains;
  const quantized = row.program_command_shape_quantized_projection_chains;
  if (Number.isInteger(dense) || Number.isInteger(quantized)) {
    return {
      dense: Number.isInteger(dense) ? dense : 0,
      quantized: Number.isInteger(quantized) ? quantized : 0,
    };
  }
  const encodedDense = row.program_command_encoded_dense_projection_chain_per_call;
  const encodedQuantized = row.program_command_encoded_projection_chain_per_call;
  if (Number.isInteger(encodedDense) || Number.isInteger(encodedQuantized)) {
    return {
      dense: Number.isInteger(encodedDense) ? encodedDense : 0,
      quantized: Number.isInteger(encodedQuantized) ? encodedQuantized : 0,
    };
  }
  return null;
}

function pressureReductionTarget(row) {
  if (!row || typeof row !== "object") return "missing";
  const targets = Object.entries(row)
    .filter(([key, value]) =>
      key.startsWith("program_command_dispatches_") &&
      key.endsWith("_per_call") &&
      typeof value === "number" &&
      Number.isFinite(value) &&
      value > 0
    )
    .map(([key, value]) => [
      key
        .replace(/^program_command_dispatches_/, "")
        .replace(/_per_call$/, ""),
      value,
    ])
    .filter(([name]) => name.includes("projection_chain") || name.includes("projection_row_chain"))
    .sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));
  return targets.length ? `${targets[0][0]}:${targets[0][1]}` : "none";
}

function frontierNextTarget(weakest) {
  if (!weakest) return "none";
  if (weakest.label === "q8_0/prompt" && weakest.pressureTarget === "projection_chain:60") {
    return "semantic_sublayer_or_quantized_projection_chain";
  }
  if (weakest.label === "q8_0/prompt" && weakest.pressureTarget.startsWith("projection_row_chain:")) {
    return "semantic_sublayer_or_two_phase_tile_parallel_row_chain";
  }
  if (weakest.label === "f16/decode" && weakest.pressureTarget === "dense_projection_chain:60") {
    return "semantic_sublayer_or_dense_projection_chain";
  }
  return "inspect_command_pressure";
}

function rejectedFrontierShortcut(weakest) {
  if (!weakest) return "none";
  if (weakest.label === "q8_0/prompt" && weakest.pressureTarget === "projection_chain:60") {
    return "projection_row_chain_default_off_needs_model_speedup";
  }
  if (weakest.label === "q8_0/prompt" && weakest.pressureTarget.startsWith("projection_row_chain:")) {
    return "default_semantic_row_chain_needs_throughput_kernel";
  }
  return "none";
}

function baselineEvidence(latestPath) {
  if (!latestPath) return null;
  let baselineData;
  let latestData;
  try {
    baselineData = readJson(baselineArtifacts[0]);
    latestData = readJson(latestPath);
  } catch {
    return null;
  }
  const rows = baselineComparisonMetrics.map(([label, fmt, phase]) => {
    const latestValue = laneMetricFromSummary(latestData, fmt, phase, "tok_s");
    const baselineValue = laneMetricFromSummary(baselineData, fmt, phase, "tok_s");
    if (latestValue === null || baselineValue === null || baselineValue === 0) {
      return { label, fmt, phase, ok: false, reason: "missing" };
    }
    return {
      label,
      fmt,
      phase,
      latest: latestValue,
      baseline: baselineValue,
      ratio: latestValue / baselineValue,
      floor: baselineGateFloor,
      ok: latestValue / baselineValue >= baselineGateFloor,
    };
  });
  const structuralRows = baselineComparisonMetrics.map(([label, fmt, phase]) => {
    const latestDispatch = laneMetricFromSummary(latestData, fmt, phase, "dispatches_per_call");
    const baselineDispatch = laneMetricFromSummary(baselineData, fmt, phase, "dispatches_per_call");
    const latestFallback = laneMetricFromSummary(latestData, fmt, phase, "fallback_ops");
    const baselineFallback = laneMetricFromSummary(baselineData, fmt, phase, "fallback_ops");
    if (latestDispatch === null || baselineDispatch === null || latestFallback === null || baselineFallback === null) {
      return { label, fmt, phase, ok: false, reason: "missing" };
    }
    return {
      label,
      fmt,
      phase,
      latestDispatch,
      baselineDispatch,
      latestFallback,
      baselineFallback,
      ok: latestDispatch <= baselineDispatch && latestFallback === 0 && baselineFallback === 0,
    };
  });
  return { latestPath, baselinePath: baselineArtifacts[0], rows, structuralRows };
}

function baselineDeltaLine(latestPath) {
  const evidence = baselineEvidence(latestPath);
  if (!evidence) return null;
  const parts = baselineComparisonMetrics.map(([label, fmt, phase]) => {
    const row = evidence.rows.find((candidate) => candidate.fmt === fmt && candidate.phase === phase);
    if (!row || row.reason) return `${label}=n/a`;
    return `${label}=${row.latest.toFixed(2)}/${row.baseline.toFixed(2)} ${formatPct(row.ratio)}`;
  });
  const structuralParts = baselineComparisonMetrics.map(([label, fmt, phase]) => {
    const row = evidence.structuralRows.find((candidate) => candidate.fmt === fmt && candidate.phase === phase);
    if (!row || row.reason) return `${label}=n/a`;
    return `${label}=dispatch ${row.latestDispatch}/${row.baselineDispatch} fallback ${row.latestFallback}/${row.baselineFallback}`;
  });
  return `bench baseline delta: latest=${compactName(evidence.latestPath)}; baseline=${compactName(evidence.baselinePath)}; ${parts.join("; ")}; ${structuralParts.join("; ")}`;
}

function referenceTokS(data, fmt, phase) {
  const key = fmt === "f16" ? "parity_gate_vs_llama_cpp_metal_f16" : "parity_gate_vs_llama_cpp_metal_q8_0";
  const value = data?.summary?.[key]?.[phase]?.llama_tok_s;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function failedReferenceDriftLine(failedPath) {
  if (!failedPath) return null;
  let baselineData;
  let failedData;
  try {
    baselineData = readJson(baselineArtifacts[0]);
    failedData = readJson(failedPath);
  } catch {
    return null;
  }
  const rows = baselineComparisonMetrics.map(([label, fmt, phase]) => {
    const baseline = referenceTokS(baselineData, fmt, phase);
    const failed = referenceTokS(failedData, fmt, phase);
    if (baseline === null || failed === null || baseline === 0) return { label, reason: "missing" };
    return {
      label,
      baseline,
      failed,
      ratio: failed / baseline,
      ok: failed / baseline >= baselineGateFloor,
    };
  });
  const slow = rows.filter((row) => row.reason || !row.ok);
  if (slow.length === 0) {
    return `bench-results: latest_failed_reference=ok latest_failed=${compactName(failedPath)}`;
  }
  const parts = slow.map((row) => {
    if (row.reason) return `${row.label}=missing`;
    return `${row.label}=${row.failed.toFixed(2)}/${row.baseline.toFixed(2)} ${formatPct(row.ratio)} floor=${formatPct(baselineGateFloor)}`;
  });
  return `bench-results: latest_failed_reference_drift=${compactName(failedPath)}; ${parts.join("; ")}`;
}

function failedArtifactLaneLine(failedPath) {
  if (!failedPath) return null;
  let failedData;
  try {
    failedData = readJson(failedPath);
  } catch {
    return null;
  }
  const rows = selectedLaneRows(failedData);
  const parts = rows.map(([fmt, phase, row]) => {
    const label = `${fmt}/${phase}`;
    if (!row || typeof row !== "object") return `${label}=missing`;
    const parityRow = parityRowFor(failedData, fmt, phase);
    const parity = parityRow && typeof parityRow.parity === "number" && Number.isFinite(parityRow.parity)
      ? ` ggml=${formatPct(parityRow.parity)}`
      : " ggml=missing";
    return `${label}=${Number(row.tok_s).toFixed(2)} tok/s${parity} dispatch=${row.dispatches_per_call} commands=${row.commands_per_call} fallback=${row.fallback_ops}`;
  });
  return `bench-results: latest_failed_zgml=${compactName(failedPath)}; ${parts.join("; ")}`;
}

function failedAcceptedDeltaLine(failedPath, acceptedPath) {
  if (!failedPath || !acceptedPath) return null;
  let failedData;
  let acceptedData;
  try {
    failedData = readJson(failedPath);
    acceptedData = readJson(acceptedPath);
  } catch {
    return null;
  }
  const parts = baselineComparisonMetrics.map(([label, fmt, phase]) => {
    const failed = laneMetricFromSummary(failedData, fmt, phase, "tok_s");
    const accepted = laneMetricFromSummary(acceptedData, fmt, phase, "tok_s");
    if (failed === null || accepted === null || accepted === 0) return `${label}=missing`;
    return `${label}=${formatPct(failed / accepted)} failed=${failed.toFixed(2)} accepted=${accepted.toFixed(2)}`;
  });
  return `bench-results: latest_failed_vs_accepted=${compactName(failedPath)}; accepted=${compactName(acceptedPath)}; ${parts.join("; ")}`;
}

function baselineGateLine(latestPath) {
  const evidence = baselineEvidence(latestPath);
  if (!evidence) return null;
  const failed = [
    ...evidence.rows.filter((row) => !row.ok),
    ...evidence.structuralRows.filter((row) => !row.ok),
  ];
  const speedParts = evidence.rows.map((row) => {
    if (row.reason) return `${row.label}=missing`;
    return `${row.label}=${formatPct(row.ratio)} floor=${formatPct(row.floor)}`;
  });
  const structuralParts = evidence.structuralRows.map((row) => {
    if (row.reason) return `${row.label}=missing`;
    return `${row.label}=dispatch ${row.latestDispatch}/${row.baselineDispatch} fallback ${row.latestFallback}/${row.baselineFallback}`;
  });
  return {
    passed: failed.length === 0,
    line: `bench baseline gate: ${failed.length === 0 ? "pass" : "fail"}; latest=${compactName(evidence.latestPath)}; baseline=${compactName(evidence.baselinePath)}; ${speedParts.join("; ")}; ${structuralParts.join("; ")}`,
  };
}

function substrateGateLine(latestPath) {
  if (!latestPath) {
    return {
      passed: false,
      line: "bench substrate gate: fail; no local p128/g200/r3 full-run artifact found; run npm run bench:ggml",
    };
  }
  let data;
  try {
    data = readJson(latestPath);
  } catch (error) {
    return {
      passed: false,
      line: `bench substrate gate: fail; latest=${compactName(latestPath)}; could not read artifact: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
  const rows = selectedLaneRows(data);
  const failures = [];
  if (data?.benchmark !== "smollm-135m" || data?.prompt_tokens !== 128 || data?.gen_tokens !== 200 || data?.repetitions !== 3) {
    failures.push("wrong benchmark shape");
  }
  if (data?.gates?.required_pass !== true) failures.push("required_pass is not true");
  if (data?.gates?.native_execution?.passed !== true) failures.push("native_execution gate is not true");
  for (const [fmt, phase, row] of rows) {
    const label = `${fmt}/${phase}`;
    if (!row || typeof row !== "object") {
      failures.push(`${label} missing selected lane`);
      continue;
    }
    const shape = substrateLaneShape[label];
    const parityFloor = substrateParityFloors[label];
    const expectedStencilHash = substrateRuntimePatchStencilHashes[phase];
    const rawStencilHash = rawRuntimePatchStencilHash(latestPath, fmt, phase);
    if (!shape) failures.push(`${label} missing substrate lane shape expectation`);
    if (row.fallback_ops !== 0) failures.push(`${label} fallback_ops=${row.fallback_ops}`);
    if (row.dynamic_region_command_plans !== 0) failures.push(`${label} dynamic_region_command_plans=${row.dynamic_region_command_plans}`);
    if (row.dynamic_region_command_plans_per_call !== 0) failures.push(`${label} dynamic_region_command_plans_per_call=${row.dynamic_region_command_plans_per_call}`);
    if (row.schedule_region_failed_ops !== 0) failures.push(`${label} schedule_region_failed_ops=${row.schedule_region_failed_ops}`);
    if (row.schedule_region_failed_ops_per_call !== 0) failures.push(`${label} schedule_region_failed_ops_per_call=${row.schedule_region_failed_ops_per_call}`);
    if ((row.runtime_patch_invalid ?? 0) !== 0) failures.push(`${label} runtime_patch_invalid=${row.runtime_patch_invalid}`);
    if ((row.runtime_patch_changed ?? 0) !== (phase === "decode" ? row.expected_profile_calls : 0)) failures.push(`${label} runtime_patch_changed=${row.runtime_patch_changed ?? 0} expected=${phase === "decode" ? row.expected_profile_calls : 0}`);
    if (row.profile_calls_match !== true) failures.push(`${label} profile_calls_match is not true`);
    if (!(typeof row.runtime_patch_stencil_hash === "number" && row.runtime_patch_stencil_hash > 0)) failures.push(`${label} missing runtime_patch_stencil_hash`);
    if (rawStencilHash !== expectedStencilHash) failures.push(`${label} runtime_patch_stencil_hash=${rawStencilHash ?? "missing"} expected=${expectedStencilHash}`);
    const parityRow = parityRowFor(data, fmt, phase);
    if (!parityRow) failures.push(`${label} missing ggml parity row`);
    else {
      if (parityRow.llama_backend_is_metal !== true) failures.push(`${label} ggml reference backend is not Metal`);
      if (!(typeof parityRow.parity === "number" && Number.isFinite(parityRow.parity) && parityRow.parity > 0)) failures.push(`${label} missing ggml parity ratio`);
      else if (typeof parityFloor === "number" && parityRow.parity < parityFloor) failures.push(`${label} ggml=${formatPct(parityRow.parity)} below floor=${formatPct(parityFloor)}`);
      if (!(typeof parityRow.llama_tok_s === "number" && Number.isFinite(parityRow.llama_tok_s) && parityRow.llama_tok_s > 0)) failures.push(`${label} missing ggml reference tok/s`);
    }
    if (shape && row.dispatches_per_call !== shape.dispatches) failures.push(`${label} dispatches_per_call=${row.dispatches_per_call}`);
    if (shape && !matchesExpected(row.commands_per_call, shape.commands)) failures.push(`${label} commands_per_call=${row.commands_per_call}`);
    if (shape && row.region_command_plan_cached_per_call !== shape.cachedCommandPlansPerCall) failures.push(`${label} region_command_plan_cached_per_call=${row.region_command_plan_cached_per_call}`);
    if (row.syncs_per_call !== 1) failures.push(`${label} syncs_per_call=${row.syncs_per_call}`);
    if (row.runtime_patch_holes !== substratePatchShape.holes) failures.push(`${label} runtime_patch_holes=${row.runtime_patch_holes}`);
    if (row.runtime_patch_cache_write_pos_holes !== substratePatchShape.cacheWritePosHoles) failures.push(`${label} runtime_patch_cache_write_pos_holes=${row.runtime_patch_cache_write_pos_holes}`);
    if (row.runtime_patch_attention_seq_kv_holes !== substratePatchShape.attentionSeqKvHoles) failures.push(`${label} runtime_patch_attention_seq_kv_holes=${row.runtime_patch_attention_seq_kv_holes}`);
    if (row.semantic_runtime_patch_holes !== substratePatchShape.holes) failures.push(`${label} semantic_runtime_patch_holes=${row.semantic_runtime_patch_holes}`);
    if (row.semantic_runtime_patch_cache_write_pos_holes !== substratePatchShape.cacheWritePosHoles) failures.push(`${label} semantic_runtime_patch_cache_write_pos_holes=${row.semantic_runtime_patch_cache_write_pos_holes}`);
    if (row.semantic_runtime_patch_attention_seq_kv_holes !== substratePatchShape.attentionSeqKvHoles) failures.push(`${label} semantic_runtime_patch_attention_seq_kv_holes=${row.semantic_runtime_patch_attention_seq_kv_holes}`);
    if (row.runtime_patch_calls !== row.expected_profile_calls) failures.push(`${label} runtime_patch_calls=${row.runtime_patch_calls} expected_profile_calls=${row.expected_profile_calls}`);
  }
  const trendGate = trendGateLine(latestPath);
  if (!trendGate) failures.push("missing trend evidence");
  else if (!trendGate.passed) failures.push("trend floor failed");
  const baselineGate = baselineGateLine(latestPath);
  if (!baselineGate) failures.push("missing baseline evidence");
  else if (!baselineGate.passed) failures.push("baseline floor failed");
  const laneSummary = rows.map(([fmt, phase, row]) => {
    if (!row || typeof row !== "object") return `${fmt}/${phase}=missing`;
    const rawStencilHash = rawRuntimePatchStencilHash(latestPath, fmt, phase);
    const expectedStencilHash = substrateRuntimePatchStencilHashes[phase];
    const parityRow = parityRowFor(data, fmt, phase);
    const parityFloor = substrateParityFloors[`${fmt}/${phase}`];
    const floorText = typeof parityFloor === "number" ? ` ggml_floor=${formatPct(parityFloor)}` : "";
    const ggml = parityRow && typeof parityRow.parity === "number" && Number.isFinite(parityRow.parity)
      ? ` ggml=${formatPct(parityRow.parity)} to90=${formatMultiplier(parityTarget / parityRow.parity)} ref=${Number(parityRow.llama_tok_s).toFixed(2)}`
      : " ggml=missing";
    return `${fmt}/${phase}=${Number(row.tok_s).toFixed(2)} tok/s${ggml} dispatch=${row.dispatches_per_call} commands=${row.commands_per_call} cached=${row.region_command_plan_cached_per_call} dynamic=${row.dynamic_region_command_plans}/${row.dynamic_region_command_plans_per_call} schedule_fail=${row.schedule_region_failed_ops}/${row.schedule_region_failed_ops_per_call} sync=${row.syncs_per_call} patch_holes=${row.runtime_patch_holes} patch_calls=${row.runtime_patch_calls}/${row.expected_profile_calls} patch_changed=${row.runtime_patch_changed ?? 0} stencil=${rawStencilHash ?? "missing"}/${expectedStencilHash} fallback=${row.fallback_ops}${floorText}`;
  }).join("; ");
  const weakest = rows.reduce((best, [fmt, phase, row]) => {
    if (!row || typeof row !== "object") return best;
    const parityRow = parityRowFor(data, fmt, phase);
    if (!parityRow || typeof parityRow.parity !== "number" || !Number.isFinite(parityRow.parity) || parityRow.parity <= 0) return best;
    const current = {
      label: `${fmt}/${phase}`,
      parity: parityRow.parity,
      to90: parityTarget / parityRow.parity,
      dispatches: row.dispatches_per_call,
      commands: row.commands_per_call,
      topCommands: commandBreakdown(row),
      commandPressure: commandPressure(row),
      sidecarChains: projectionChainSidecars(row),
      chainShape: projectionChainShape(row),
      pressureTarget: pressureReductionTarget(row),
    };
    return !best || current.parity < best.parity ? current : best;
  }, null);
  const q8DecodeSidecars = projectionChainSidecars(rows.find(([fmt, phase]) => fmt === "q8_0" && phase === "decode")?.[2]);
  const frontierSummary = weakest
    ? `; frontier weakest=${weakest.label} ggml=${formatPct(weakest.parity)} to90=${formatMultiplier(weakest.to90)} dispatch=${weakest.dispatches} commands=${weakest.commands} top=${weakest.topCommands} pressure=${weakest.commandPressure} target=${weakest.pressureTarget} chain_shape=${weakest.chainShape} next=${frontierNextTarget(weakest)} rejected=${rejectedFrontierShortcut(weakest)} sidecars=${weakest.sidecarChains} q8_decode_sidecars=${q8DecodeSidecars}`
    : "";
  return {
    passed: failures.length === 0,
    line: `bench substrate gate: ${failures.length === 0 ? "pass" : "fail"}; latest=${compactName(latestPath)}; ${laneSummary}${frontierSummary}${failures.length ? `; failures=${failures.join(", ")}` : ""}`,
  };
}

const artifacts = [...baselineArtifacts];
const latest = latestFullRunArtifact();
if (latest) artifacts.push(latest);

const result = spawnSync("python3", ["scripts/verify_bench_artifact.py", "--status", ...artifacts], {
  encoding: "utf8",
  stdio: ["ignore", "pipe", "pipe"],
});

if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
const latestPytorch = latestPytorchComparisonArtifact();
const latestRawPytorch = latestRawPytorchComparisonArtifact();
process.stdout.write(`${pytorchComparisonStatusLine(latestPytorch)}\n`);
const broadPytorch = pytorchBroadStatusLine(latestPytorchBroadArtifact(), latestPytorch);
if (broadPytorch) process.stdout.write(`${broadPytorch}\n`);
const focusPytorch = pytorchFocusStatusLine(latestPytorchFocusArtifact(), latestPytorch);
if (focusPytorch) process.stdout.write(`${focusPytorch}\n`);
const pytorchFreshness = pytorchFreshnessStatusLine(latestPytorch, latestRawPytorch);
if (pytorchFreshness) process.stdout.write(`${pytorchFreshness}\n`);
const latestQ8Prompt = latestQ8PromptCandidateArtifact();
const latestRawQ8Prompt = latestRawQ8PromptCandidateArtifact();
const latestFrontier = latestFrontierArtifact();
const latestRawFrontier = latestRawFrontierArtifact();
const latestQsemanticThroughput = latestQsemanticThroughputArtifact();
const latestQprojFrontier = latestQprojFrontierArtifact();
const latestGgmlSmoke = latestGgmlSmokeArtifact();
process.stdout.write(`${q8PromptCandidateStatusLine(latestQ8Prompt)}\n`);
const q8PromptFreshness = q8PromptFreshnessStatusLine(latestQ8Prompt, latestRawQ8Prompt);
if (q8PromptFreshness) process.stdout.write(`${q8PromptFreshness}\n`);
process.stdout.write(`${qprojFrontierStatusLine(latestQprojFrontier)}\n`);
process.stdout.write(`${frontierStatusLine(latestFrontier, latestRawFrontier)}\n`);
const frontierFreshness = frontierFreshnessStatusLine(latestFrontier, latestRawFrontier);
if (frontierFreshness) process.stdout.write(`${frontierFreshness}\n`);
process.stdout.write(`${qsemanticThroughputStatusLine(latestQsemanticThroughput)}\n`);
process.stdout.write(`${ggmlSmokeStatusLine(latestGgmlSmoke)}\n`);
process.stdout.write(`${perfNextStatusLine({ latestPath: latest, pytorchPath: latestPytorch, q8Path: latestQ8Prompt, rawQ8Path: latestRawQ8Prompt, frontierPath: latestFrontier, rawFrontierPath: latestQsemanticThroughput ?? latestRawFrontier })}\n`);
const quarantined = quarantinedFullRunArtifacts();
if (quarantined.length > 0) {
  process.stdout.write(`bench-results: ${quarantined.length} quarantined p128/g200/r3 artifact(s) ignored for accepted evidence; latest_failed=${compactName(quarantined.at(-1))}\n`);
  const referenceDrift = failedReferenceDriftLine(quarantined.at(-1));
  if (referenceDrift) process.stdout.write(`${referenceDrift}\n`);
  const failedLanes = failedArtifactLaneLine(quarantined.at(-1));
  if (failedLanes) process.stdout.write(`${failedLanes}\n`);
  const failedAcceptedDelta = failedAcceptedDeltaLine(quarantined.at(-1), latest);
  if (failedAcceptedDelta) process.stdout.write(`${failedAcceptedDelta}\n`);
}
if (!latest) {
  process.stdout.write("bench-results: no local p128/g200/r3 full-run artifact found; run npm run bench:ggml for parity evidence\n");
} else {
  const baselineDelta = baselineDeltaLine(latest);
  if (baselineDelta) process.stdout.write(`${baselineDelta}\n`);
  const baselineGate = substrateGateEnabled ? baselineGateLine(latest) : null;
  if (baselineGate) {
    process.stdout.write(`${baselineGate.line}\n`);
    if (!baselineGate.passed) process.exit(1);
  }
  const trend = trendStatusLine(latest);
  if (trend) process.stdout.write(`${trend}\n`);
  const gate = trendGateEnabled ? trendGateLine(latest) : null;
  if (gate) {
    process.stdout.write(`${gate.line}\n`);
    if (!gate.passed) process.exit(1);
  }
}
const substrateGate = substrateGateEnabled ? substrateGateLine(latest) : null;
if (substrateGate) {
  process.stdout.write(`${substrateGate.line}\n`);
  if (!substrateGate.passed) process.exit(1);
}
process.exit(result.status ?? 1);
