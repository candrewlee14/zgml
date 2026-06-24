"use strict";

const { existsSync, mkdirSync, writeFileSync } = require("node:fs");
const { spawnSync } = require("node:child_process");
const { join, resolve } = require("node:path");
const os = require("node:os");

const root = resolve(__dirname, "..");
const model = process.env.ZGML_Q8_MODEL || process.env.ZGML_MODEL || "data/smollm/SmolLM-135M.Q8_0.gguf";
const promptTokens = process.env.BENCH_CANDIDATE_PROMPT || "128";
const genTokens = process.env.BENCH_CANDIDATE_GEN || "40";
const repetitions = process.env.BENCH_CANDIDATE_REPS || "1";
const build = process.env.BENCH_BUILD_ZGML ?? "1";
const binary = "./zig-out/bin/bench-llama-smollm";
const writeArtifact = process.env.BENCH_Q8_PROMPT_WRITE_ARTIFACT !== "0";
const artifactDir = process.env.BENCH_Q8_PROMPT_ARTIFACT_DIR || join("bench-results", "q8-prompt");
const speedupFloor = Number(process.env.BENCH_CANDIDATE_SPEEDUP_FLOOR || "1.05");
const commandSpeedupFloor = Number(process.env.BENCH_COMMAND_CANDIDATE_SPEEDUP_FLOOR || "0.95");
const semanticSpeedupFloor = Number(process.env.BENCH_SEMANTIC_CANDIDATE_SPEEDUP_FLOOR || "1.00");
const attempts = positiveInt(process.env.BENCH_CANDIDATE_ATTEMPTS || "3", "BENCH_CANDIDATE_ATTEMPTS");
const laneMode = process.env.BENCH_Q8_PROMPT_LANES || "all";
const rowChainLowering = "default_projection_chain_plus_row_chain_candidate_single_dispatch_tiled_row_chain";
const commandLowering = "default_projection_chain_plus_row_chain_command_two_dispatch";
const twoPhaseLowering = "default_projection_chain_plus_row_chain_candidate_two_phase_tiled_row_chain";
const semanticLowering = "semantic_ffn_sublayer_command_plus_two_phase_tiled_row_chain_tail";
const requiredNextTarget = "semantic_sublayer_or_two_phase_tile_parallel_row_chain";
const singleDispatchTrap = "serial_n_tile_loop_without_cross_threadgroup_row_reduce";
const viableNextTarget = "semantic_sublayer_or_two_phase_tile_parallel_row_chain";
const semanticPairTarget = "projection_pair_fused_elementwise_or_larger_ffn_sublayer";
const qprojGroupTarget = "frontier_only_not_full_model_sibling_region";
const decodeLowering = "staged_qmatvec_projection_chain_plus_row_chain";
const decodeRowChainDefault = "off";
const decodeNextTarget = "larger_semantic_sublayer_or_qmatvec_throughput_kernel";
const defaultCommandFloor = Number(process.env.BENCH_Q8_PROMPT_DEFAULT_COMMAND_FLOOR || "241");
const candidateCommandCeil = Number(process.env.BENCH_Q8_PROMPT_COMMAND_CEIL || "181");
const candidateProjectionRowChainFloor = Number(process.env.BENCH_Q8_PROMPT_PROJECTION_ROW_CHAIN_FLOOR || "60");
const semanticProjectionRowChainFloor = Number(process.env.BENCH_Q8_PROMPT_SEMANTIC_PROJECTION_ROW_CHAIN_FLOOR || "30");
const defaultProjectionChainFloor = Number(process.env.BENCH_Q8_PROMPT_DEFAULT_PROJECTION_CHAIN_FLOOR || "60");
const candidateProjectionChainCeil = Number(process.env.BENCH_Q8_PROMPT_PROJECTION_CHAIN_CEIL || "0");
const defaultProjectionPairFloor = Number(process.env.BENCH_Q8_PROMPT_DEFAULT_PROJECTION_PAIR_FLOOR || "30");
const candidateProjectionPairFloor = Number(process.env.BENCH_Q8_PROMPT_PROJECTION_PAIR_FLOOR || "30");
const decodeCommandCeil = Number(process.env.BENCH_Q8_DECODE_COMMAND_CEIL || "211");
const decodeProjectionPairFloor = Number(process.env.BENCH_Q8_DECODE_PROJECTION_PAIR_FLOOR || "30");
const decodeProjectionChainFloor = Number(process.env.BENCH_Q8_DECODE_PROJECTION_CHAIN_FLOOR || "60");

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: root,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    ...options,
  });
  const output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
  if (result.status !== 0) {
    const err = new Error(`${command} ${args.join(" ")} failed with status ${result.status ?? 1}`);
    err.output = output;
    throw err;
  }
  return output;
}

function progress(message) {
  process.stderr.write(`[q8-prompt] ${message}\n`);
}

function parseRows(output) {
  const prefix = "ZGML_BENCH_JSON ";
  const rows = [];
  for (const line of output.split(/\r?\n/)) {
    if (!line.startsWith(prefix)) continue;
    rows.push(JSON.parse(line.slice(prefix.length)));
  }
  return rows;
}

function rowFor(output, label) {
  const rows = parseRows(output).filter((row) => String(row.label ?? "").split(/\s+/).join(" ") === label);
  if (rows.length === 0) throw new Error(`missing ZGML_BENCH_JSON row: ${label}`);
  rows.sort((left, right) => Number(left.prompt_tok_s ?? 0) - Number(right.prompt_tok_s ?? 0));
  return rows[Math.floor(rows.length / 2)];
}

function number(row, key) {
  const value = Number(row[key]);
  return Number.isFinite(value) ? value : null;
}

function format(value, digits = 2) {
  return value === null || value === undefined || !Number.isFinite(value) ? "n/a" : value.toFixed(digits);
}

function timestampForArtifact(date = new Date()) {
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function roundMetric(value) {
  return Number.isFinite(value) ? Number(value.toFixed(6)) : null;
}

function positiveInt(value, label) {
  const n = Number(value);
  if (!Number.isSafeInteger(n) || n <= 0) {
    console.error(`${label} must be a positive integer, got ${value}`);
    process.exit(1);
  }
  return n;
}

function parseLanes(value) {
  const names = new Set(
    String(value)
      .split(",")
      .map((part) => part.trim().toLowerCase().replace(/-/g, "_"))
      .filter(Boolean),
  );
  if (names.size === 0 || names.has("all")) return new Set(["command", "single", "two_phase", "semantic"]);
  for (const name of names) {
    if (name !== "command" && name !== "single" && name !== "two_phase" && name !== "semantic") {
      console.error(`BENCH_Q8_PROMPT_LANES contains unsupported lane ${name}; use all, command, single, two_phase, semantic`);
      process.exit(1);
    }
  }
  return names;
}

const measuredLanes = parseLanes(laneMode);
const measureCommand = measuredLanes.has("command");
const measureSingle = measuredLanes.has("single");
const measureTwoPhase = measuredLanes.has("two_phase");
const measureSemantic = measuredLanes.has("semantic");

function emptyLane(index) {
  return {
    index,
    tokS: null,
    speedup: null,
    dispatches: null,
    commands: null,
    projectionRowChains: 0,
    projectionChains: 0,
    projectionPairs: 0,
    projectionPairDispatches: 0,
    projectionRowChainDispatches: 0,
    projectionRowChainDispatchSplit: null,
    projectionRowChainDispatchExcess: 0,
    tiledCount: 0,
    tiledRowTileGroups: 0,
    tiledNTiles: 0,
    tiledSerialLoops: 0,
    tiledPartialSlots: 0,
    tiledScratchCapacity: 0,
    tiledSpills: 0,
    tiledTwoPhaseCount: 0,
    fallback: 0,
  };
}

function projectionRowChainSplit(rowChains, dispatches) {
  return rowChains > 0 ? dispatches / rowChains : null;
}

function readProjectionLane(row, defaultTokS, index) {
  const tokS = number(row, "prompt_tok_s");
  const dispatches = number(row, "dispatches_per_call");
  const commands = number(row, "commands_per_call");
  const projectionRowChains = number(row, "program_command_encoded_projection_row_chain_per_call") ?? 0;
  const projectionChains = number(row, "program_command_encoded_projection_chain_per_call") ?? 0;
  const projectionPairs = number(row, "program_command_encoded_projection_pair_fused_elementwise_chain_per_call") ?? 0;
  const projectionPairDispatches = number(row, "program_command_dispatches_projection_pair_fused_elementwise_chain_per_call") ?? 0;
  const projectionGroups = number(row, "program_command_encoded_projection_group_per_call") ?? 0;
  const projectionGroupDispatches = number(row, "program_command_dispatches_projection_group_per_call") ?? 0;
  const projectionCacheGroups = number(row, "program_command_encoded_projection_cache_group_per_call") ?? 0;
  const projectionCacheGroupDispatches = number(row, "program_command_dispatches_projection_cache_group_per_call") ?? 0;
  const projectionRowChainDispatches = number(row, "program_command_dispatches_projection_row_chain_per_call") ?? 0;
  return {
    index,
    tokS,
    speedup: defaultTokS && tokS ? tokS / defaultTokS : null,
    dispatches,
    commands,
    projectionRowChains,
    projectionChains,
    projectionPairs,
    projectionPairDispatches,
    projectionGroups,
    projectionGroupDispatches,
    projectionCacheGroups,
    projectionCacheGroupDispatches,
    projectionRowChainDispatches,
    projectionRowChainDispatchSplit: projectionRowChainSplit(projectionRowChains, projectionRowChainDispatches),
    projectionRowChainDispatchExcess: Math.max(0, projectionRowChainDispatches - projectionRowChains),
    tiledCount: number(row, "qmatmul_row_chain_tiled_count_per_call") ?? 0,
    tiledRowTileGroups: number(row, "qmatmul_row_chain_tiled_row_tile_groups_per_call") ?? 0,
    tiledNTiles: number(row, "qmatmul_row_chain_tiled_n_tiles_per_call") ?? 0,
    tiledSerialLoops: number(row, "qmatmul_row_chain_tiled_serial_tile_loops_per_call") ?? 0,
    tiledPartialSlots: number(row, "qmatmul_row_chain_tiled_partial_slots_per_call") ?? 0,
    tiledScratchCapacity: number(row, "qmatmul_row_chain_tiled_scratch_capacity_per_call") ?? 0,
    tiledSpills: number(row, "qmatmul_row_chain_tiled_spilled_elementwise_per_call") ?? 0,
    tiledTwoPhaseCount: number(row, "qmatmul_row_chain_tiled_two_phase_count_per_call") ?? 0,
    fallback: number(row, "fallback_ops") ?? 0,
  };
}

function keepsProjectionPairOrSemanticCommand(lane) {
  return lane.projectionPairs >= candidateProjectionPairFloor ||
    (lane.projectionPairs === 0 && lane.projectionRowChains >= semanticProjectionRowChainFloor);
}

if (!existsSync(resolve(root, model))) {
  console.error(`q8 prompt candidate gate: missing model ${model}`);
  console.error("Set ZGML_Q8_MODEL/ZGML_MODEL or run with BENCH_AUTO_DOWNLOAD through scripts/bench_vs_ggml.sh first.");
  process.exit(1);
}

if (build === "1") {
  progress("build bench binary");
  run("zig", ["build", "-Doptimize=ReleaseFast", "bench-build"]);
  progress("built bench binary");
} else if (build !== "0") {
  console.error("BENCH_BUILD_ZGML must be 0 or 1");
  process.exit(1);
}

const baseArgs = [model, promptTokens, genTokens, repetitions, "--metal-prefill-device", "--metal-decode-region", "--gate-only"];
progress(`attempts=${attempts} lanes=${[...measuredLanes].join(",")} model=${model} prompt=${promptTokens} gen=${genTokens} reps=${repetitions}`);

function measureAttempt(index) {
  progress(`attempt ${index}/${attempts} default`);
  const defaultOutput = run(binary, baseArgs);
  const commandOutput = measureCommand
    ? (progress(`attempt ${index}/${attempts} command`), run(binary, [...baseArgs, "--metal-prompt-projection-row-chain-command"]))
    : null;
  const candidateOutput = measureSingle
    ? (progress(`attempt ${index}/${attempts} single-dispatch-candidate`), run(binary, [...baseArgs, "--metal-prompt-projection-row-chain-candidate"]))
    : null;
  const twoPhaseOutput = measureTwoPhase
    ? (progress(`attempt ${index}/${attempts} two-phase-candidate`), run(binary, [...baseArgs, "--metal-prompt-projection-row-chain-two-phase-candidate"]))
    : null;
  const semanticOutput = measureSemantic
    ? (progress(`attempt ${index}/${attempts} semantic-throughput-candidate`), run(binary, [...baseArgs, "--metal-prompt-semantic-throughput-candidate"]))
    : null;
  const defaultRow = rowFor(defaultOutput, "metal scheduled prefill");
  const defaultDecodeRow = rowFor(defaultOutput, "metal region decode");

  const defaultTokS = number(defaultRow, "prompt_tok_s");
  const commandLane = measureCommand
    ? readProjectionLane(rowFor(commandOutput, "metal scheduled prefill projection-row-chain command"), defaultTokS, index)
    : emptyLane(index);
  const singleLane = measureSingle
    ? readProjectionLane(rowFor(candidateOutput, "metal scheduled prefill projection-row-chain candidate"), defaultTokS, index)
    : emptyLane(index);
  const twoPhaseLane = measureTwoPhase
    ? readProjectionLane(rowFor(twoPhaseOutput, "metal scheduled prefill projection-row-chain two-phase candidate"), defaultTokS, index)
    : emptyLane(index);
  const semanticLane = measureSemantic
    ? readProjectionLane(rowFor(semanticOutput, "metal scheduled prefill semantic throughput candidate"), defaultTokS, index)
    : emptyLane(index);
  const defaultDispatches = number(defaultRow, "dispatches_per_call");
  const defaultCommands = number(defaultRow, "commands_per_call");
  const defaultProjectionRowChains = number(defaultRow, "program_command_encoded_projection_row_chain_per_call") ?? 0;
  const defaultProjectionChains = number(defaultRow, "program_command_encoded_projection_chain_per_call") ?? 0;
  const defaultProjectionPairs = number(defaultRow, "program_command_encoded_projection_pair_fused_elementwise_chain_per_call") ?? 0;
  const defaultProjectionGroups = number(defaultRow, "program_command_encoded_projection_group_per_call") ?? 0;
  const defaultProjectionChainRowChainFrontiers = number(defaultRow, "program_command_shape_projection_chain_row_chain_frontiers") ?? 0;
  const defaultProjectionCacheGroups = number(defaultRow, "program_command_encoded_projection_cache_group_per_call") ?? 0;
  const defaultDecodeCommands = number(defaultDecodeRow, "commands_per_call");
  const defaultDecodeProjectionChains = number(defaultDecodeRow, "program_command_encoded_projection_chain_per_call") ?? 0;
  const defaultDecodeProjectionPairs = number(defaultDecodeRow, "program_command_encoded_projection_pair_fused_elementwise_chain_per_call") ?? 0;
  const defaultDecodeProjectionGroups = number(defaultDecodeRow, "program_command_encoded_projection_group_per_call") ?? 0;
  const defaultDecodeProjectionCacheGroups = number(defaultDecodeRow, "program_command_encoded_projection_cache_group_per_call") ?? 0;
  const defaultDecodeFallback = number(defaultDecodeRow, "fallback_ops") ?? 0;
  const defaultProjectionPairDispatches = number(defaultRow, "program_command_dispatches_projection_pair_fused_elementwise_chain_per_call") ?? 0;
  const defaultProjectionRowChainDispatches = number(defaultRow, "program_command_dispatches_projection_row_chain_per_call") ?? 0;
  const tiledEvidenceLane = measureSingle ? singleLane : twoPhaseLane;
  const twoPhaseScratchReady = tiledEvidenceLane.tiledPartialSlots > 0 && tiledEvidenceLane.tiledScratchCapacity >= tiledEvidenceLane.tiledPartialSlots;
  const defaultProjectionRowChainDispatchSplit = defaultProjectionRowChains > 0 ? defaultProjectionRowChainDispatches / defaultProjectionRowChains : null;
  const defaultProjectionRowChainDispatchExcess = Math.max(0, defaultProjectionRowChainDispatches - defaultProjectionRowChains);
  const defaultFallback = number(defaultRow, "fallback_ops") ?? 0;
  const defaultFastPathReady =
    defaultCommands !== null &&
    defaultCommands >= defaultCommandFloor &&
    defaultProjectionChains >= defaultProjectionChainFloor &&
    defaultProjectionPairs >= defaultProjectionPairFloor &&
    defaultProjectionCacheGroups >= defaultProjectionPairFloor &&
    defaultProjectionRowChains === 0;
  const defaultDecodeFastPathReady =
    defaultDecodeCommands !== null &&
    defaultDecodeCommands <= decodeCommandCeil &&
    defaultDecodeProjectionChains >= decodeProjectionChainFloor &&
    defaultDecodeProjectionPairs >= decodeProjectionPairFloor &&
    defaultDecodeProjectionCacheGroups >= decodeProjectionPairFloor &&
    defaultDecodeFallback === 0;
  const commandSemanticReady =
    !measureCommand ||
    (commandLane.commands !== null &&
      commandLane.commands <= candidateCommandCeil &&
      commandLane.projectionChains <= candidateProjectionChainCeil &&
      keepsProjectionPairOrSemanticCommand(commandLane) &&
      commandLane.projectionCacheGroups >= candidateProjectionPairFloor &&
      commandLane.projectionRowChains >= semanticProjectionRowChainFloor);
  const commandDispatchShapeReady =
    !measureCommand ||
    (defaultDispatches !== null &&
      commandLane.dispatches !== null &&
      commandLane.dispatches <= defaultDispatches &&
      commandLane.projectionRowChainDispatches >= commandLane.projectionRowChains &&
      commandLane.projectionRowChainDispatchSplit !== null &&
      commandLane.projectionRowChainDispatchSplit <= 2.0);
  const candidateSemanticReady =
    !measureSingle ||
    (singleLane.commands !== null &&
      singleLane.commands <= candidateCommandCeil &&
      singleLane.projectionChains <= candidateProjectionChainCeil &&
      singleLane.projectionPairs >= candidateProjectionPairFloor &&
      singleLane.projectionCacheGroups >= candidateProjectionPairFloor &&
      singleLane.projectionRowChains >= candidateProjectionRowChainFloor);
  const candidateMatchesCommandShape =
    !measureSingle ||
    (singleLane.commands !== null &&
      singleLane.commands <= candidateCommandCeil &&
      singleLane.projectionChains <= candidateProjectionChainCeil &&
      singleLane.projectionPairs >= candidateProjectionPairFloor &&
      singleLane.projectionCacheGroups >= candidateProjectionPairFloor &&
      singleLane.projectionRowChains >= candidateProjectionRowChainFloor);
  const candidateDispatchShapeReady =
    !measureSingle ||
    (defaultDispatches !== null &&
      singleLane.dispatches !== null &&
      singleLane.dispatches < defaultDispatches &&
      singleLane.projectionRowChainDispatches <= singleLane.projectionRowChains);
  const twoPhaseSemanticReady =
    !measureTwoPhase ||
    (twoPhaseLane.commands !== null &&
      twoPhaseLane.commands <= candidateCommandCeil &&
      twoPhaseLane.projectionChains <= candidateProjectionChainCeil &&
      twoPhaseLane.projectionPairs >= candidateProjectionPairFloor &&
      twoPhaseLane.projectionCacheGroups >= candidateProjectionPairFloor &&
      twoPhaseLane.projectionRowChains >= candidateProjectionRowChainFloor);
  const twoPhaseDispatchShapeReady =
    !measureTwoPhase ||
    (defaultDispatches !== null &&
      twoPhaseLane.dispatches !== null &&
      twoPhaseLane.dispatches <= defaultDispatches &&
      twoPhaseLane.projectionRowChainDispatchSplit !== null &&
      twoPhaseLane.projectionRowChainDispatchSplit <= 2.0);
  const twoPhaseStructuralReady = defaultFastPathReady && twoPhaseSemanticReady && twoPhaseDispatchShapeReady && twoPhaseLane.fallback === 0;
  const semanticSemanticReady =
    !measureSemantic ||
    (semanticLane.commands !== null &&
      semanticLane.commands <= candidateCommandCeil &&
      semanticLane.projectionChains <= candidateProjectionChainCeil &&
      keepsProjectionPairOrSemanticCommand(semanticLane) &&
      semanticLane.projectionCacheGroups >= candidateProjectionPairFloor &&
      semanticLane.projectionRowChains >= semanticProjectionRowChainFloor);
  const semanticDispatchShapeReady =
    !measureSemantic ||
    (defaultDispatches !== null &&
      semanticLane.dispatches !== null &&
      semanticLane.dispatches <= defaultDispatches &&
      semanticLane.projectionRowChainDispatchSplit !== null &&
      semanticLane.projectionRowChainDispatchSplit <= 2.0);
  const semanticStructuralReady = defaultFastPathReady && semanticSemanticReady && semanticDispatchShapeReady && semanticLane.fallback === 0;
  const fallbackOk = defaultFallback === 0 && commandLane.fallback === 0 && singleLane.fallback === 0 && twoPhaseLane.fallback === 0 && semanticLane.fallback === 0;
  const commandStructuralReady = defaultFastPathReady && defaultDecodeFastPathReady && commandSemanticReady && commandDispatchShapeReady && fallbackOk;
  const structuralReady = defaultFastPathReady && defaultDecodeFastPathReady && commandStructuralReady && candidateSemanticReady && candidateMatchesCommandShape && candidateDispatchShapeReady && fallbackOk;
  const commandThroughputReady = commandLane.speedup !== null && commandLane.speedup >= commandSpeedupFloor;
  const throughputReady = singleLane.speedup !== null && singleLane.speedup >= speedupFloor;
  const progressLane = measureSingle
      ? { name: "single", dispatches: singleLane.dispatches, commands: singleLane.commands, fallback: singleLane.fallback }
      : measureTwoPhase
        ? { name: "two_phase", dispatches: twoPhaseLane.dispatches, commands: twoPhaseLane.commands, fallback: twoPhaseLane.fallback }
        : measureSemantic
          ? { name: "semantic", dispatches: semanticLane.dispatches, commands: semanticLane.commands, fallback: semanticLane.fallback }
          : { name: "command", dispatches: commandLane.dispatches, commands: commandLane.commands, fallback: commandLane.fallback };
  progress(
    `attempt ${index}/${attempts} result ` +
      `command=${format(commandLane.speedup)}x single=${format(singleLane.speedup)}x ` +
      `two_phase=${format(twoPhaseLane.speedup)}x semantic=${format(semanticLane.speedup)}x ` +
      `active_lane=${progressLane.name} ` +
      `dispatch=${format(defaultDispatches, 0)}->${format(progressLane.dispatches, 0)} ` +
      `commands=${format(defaultCommands, 0)}->${format(progressLane.commands, 0)} ` +
      `fallback=${format(defaultFallback, 0)}->${format(progressLane.fallback, 0)}`,
  );
  return {
    index,
    defaultTokS,
    commandTokS: commandLane.tokS,
    candidateTokS: singleLane.tokS,
    twoPhaseTokS: twoPhaseLane.tokS,
    semanticTokS: semanticLane.tokS,
    commandSpeedup: commandLane.speedup,
    speedup: singleLane.speedup,
    twoPhaseSpeedup: twoPhaseLane.speedup,
    semanticSpeedup: semanticLane.speedup,
    defaultDispatches,
    commandDispatches: commandLane.dispatches,
    candidateDispatches: singleLane.dispatches,
    twoPhaseDispatches: twoPhaseLane.dispatches,
    semanticDispatches: semanticLane.dispatches,
    defaultCommands,
    commandCommands: commandLane.commands,
    candidateCommands: singleLane.commands,
    twoPhaseCommands: twoPhaseLane.commands,
    semanticCommands: semanticLane.commands,
    defaultProjectionRowChains,
    commandProjectionRowChains: commandLane.projectionRowChains,
    candidateProjectionRowChains: singleLane.projectionRowChains,
    twoPhaseProjectionRowChains: twoPhaseLane.projectionRowChains,
    semanticProjectionRowChains: semanticLane.projectionRowChains,
    defaultProjectionChains,
    commandProjectionChains: commandLane.projectionChains,
    candidateProjectionChains: singleLane.projectionChains,
    twoPhaseProjectionChains: twoPhaseLane.projectionChains,
    semanticProjectionChains: semanticLane.projectionChains,
    defaultProjectionPairs,
    commandProjectionPairs: commandLane.projectionPairs,
    candidateProjectionPairs: singleLane.projectionPairs,
    twoPhaseProjectionPairs: twoPhaseLane.projectionPairs,
    semanticProjectionPairs: semanticLane.projectionPairs,
    defaultProjectionGroups,
    defaultProjectionChainRowChainFrontiers,
    commandProjectionGroups: commandLane.projectionGroups,
    candidateProjectionGroups: singleLane.projectionGroups,
    twoPhaseProjectionGroups: twoPhaseLane.projectionGroups,
    semanticProjectionGroups: semanticLane.projectionGroups,
    defaultProjectionCacheGroups,
    commandProjectionCacheGroups: commandLane.projectionCacheGroups,
    candidateProjectionCacheGroups: singleLane.projectionCacheGroups,
    twoPhaseProjectionCacheGroups: twoPhaseLane.projectionCacheGroups,
    semanticProjectionCacheGroups: semanticLane.projectionCacheGroups,
    defaultDecodeCommands,
    defaultDecodeProjectionChains,
    defaultDecodeProjectionPairs,
    defaultDecodeProjectionGroups,
    defaultDecodeProjectionCacheGroups,
    defaultDecodeFallback,
    defaultDecodeFastPathReady,
    defaultProjectionPairDispatches,
    commandProjectionPairDispatches: commandLane.projectionPairDispatches,
    candidateProjectionPairDispatches: singleLane.projectionPairDispatches,
    twoPhaseProjectionPairDispatches: twoPhaseLane.projectionPairDispatches,
    semanticProjectionPairDispatches: semanticLane.projectionPairDispatches,
    defaultProjectionGroupDispatches: number(defaultRow, "program_command_dispatches_projection_group_per_call") ?? 0,
    commandProjectionGroupDispatches: commandLane.projectionGroupDispatches,
    candidateProjectionGroupDispatches: singleLane.projectionGroupDispatches,
    twoPhaseProjectionGroupDispatches: twoPhaseLane.projectionGroupDispatches,
    semanticProjectionGroupDispatches: semanticLane.projectionGroupDispatches,
    defaultProjectionCacheGroupDispatches: number(defaultRow, "program_command_dispatches_projection_cache_group_per_call") ?? 0,
    commandProjectionCacheGroupDispatches: commandLane.projectionCacheGroupDispatches,
    candidateProjectionCacheGroupDispatches: singleLane.projectionCacheGroupDispatches,
    twoPhaseProjectionCacheGroupDispatches: twoPhaseLane.projectionCacheGroupDispatches,
    semanticProjectionCacheGroupDispatches: semanticLane.projectionCacheGroupDispatches,
    defaultProjectionRowChainDispatches,
    commandProjectionRowChainDispatches: commandLane.projectionRowChainDispatches,
    candidateProjectionRowChainDispatches: singleLane.projectionRowChainDispatches,
    twoPhaseProjectionRowChainDispatches: twoPhaseLane.projectionRowChainDispatches,
    semanticProjectionRowChainDispatches: semanticLane.projectionRowChainDispatches,
    candidateTiledCount: singleLane.tiledCount,
    candidateTiledRowTileGroups: singleLane.tiledRowTileGroups,
    candidateTiledNTiles: singleLane.tiledNTiles,
    candidateTiledSerialLoops: singleLane.tiledSerialLoops,
    candidateTiledPartialSlots: singleLane.tiledPartialSlots,
    candidateTiledScratchCapacity: singleLane.tiledScratchCapacity,
    candidateTiledSpills: singleLane.tiledSpills,
    twoPhaseTiledCount: twoPhaseLane.tiledCount,
    twoPhaseTiledRowTileGroups: twoPhaseLane.tiledRowTileGroups,
    twoPhaseTiledNTiles: twoPhaseLane.tiledNTiles,
    twoPhaseTiledSerialLoops: twoPhaseLane.tiledSerialLoops,
    twoPhaseTiledPartialSlots: twoPhaseLane.tiledPartialSlots,
    twoPhaseTiledScratchCapacity: twoPhaseLane.tiledScratchCapacity,
    twoPhaseTiledSpills: twoPhaseLane.tiledSpills,
    twoPhaseTiledTwoPhaseCount: twoPhaseLane.tiledTwoPhaseCount,
    semanticTiledCount: semanticLane.tiledCount,
    semanticTiledRowTileGroups: semanticLane.tiledRowTileGroups,
    semanticTiledNTiles: semanticLane.tiledNTiles,
    semanticTiledSerialLoops: semanticLane.tiledSerialLoops,
    semanticTiledPartialSlots: semanticLane.tiledPartialSlots,
    semanticTiledScratchCapacity: semanticLane.tiledScratchCapacity,
    semanticTiledSpills: semanticLane.tiledSpills,
    semanticTiledTwoPhaseCount: semanticLane.tiledTwoPhaseCount,
    twoPhaseScratchReady,
    defaultProjectionRowChainDispatchSplit,
    commandProjectionRowChainDispatchSplit: commandLane.projectionRowChainDispatchSplit,
    candidateProjectionRowChainDispatchSplit: singleLane.projectionRowChainDispatchSplit,
    defaultProjectionRowChainDispatchExcess,
    commandProjectionRowChainDispatchExcess: commandLane.projectionRowChainDispatchExcess,
    candidateProjectionRowChainDispatchExcess: singleLane.projectionRowChainDispatchExcess,
    defaultFallback,
    commandFallback: commandLane.fallback,
    candidateFallback: singleLane.fallback,
    twoPhaseFallback: twoPhaseLane.fallback,
    semanticFallback: semanticLane.fallback,
    commandStructuralReady,
    commandThroughputReady,
    candidateMatchesCommandShape,
    candidateDispatchShapeReady,
    twoPhaseProjectionRowChainDispatchSplit: twoPhaseLane.projectionRowChainDispatchSplit,
    twoPhaseSemanticReady,
    twoPhaseDispatchShapeReady,
    twoPhaseStructuralReady,
    semanticProjectionRowChainDispatchSplit: semanticLane.projectionRowChainDispatchSplit,
    semanticSemanticReady,
    semanticDispatchShapeReady,
    semanticStructuralReady,
    structuralReady,
    throughputReady,
  };
}

const attemptRows = [];
for (let i = 0; i < attempts; i += 1) {
  attemptRows.push(measureAttempt(i + 1));
}

const structuralReady = attemptRows.every((row) => row.structuralReady);
const commandStructuralReady = attemptRows.every((row) => row.commandStructuralReady);
const twoPhaseStructuralReady = attemptRows.every((row) => row.twoPhaseStructuralReady);
const semanticStructuralReady = attemptRows.every((row) => row.semanticStructuralReady);
const ranked = [...attemptRows].sort((left, right) => Number(right.speedup ?? -Infinity) - Number(left.speedup ?? -Infinity));
const rankedAscending = [...attemptRows].sort((left, right) => Number(left.speedup ?? Infinity) - Number(right.speedup ?? Infinity));
const commandRanked = [...attemptRows].sort((left, right) => Number(right.commandSpeedup ?? -Infinity) - Number(left.commandSpeedup ?? -Infinity));
const commandRankedAscending = [...attemptRows].sort((left, right) => Number(left.commandSpeedup ?? Infinity) - Number(right.commandSpeedup ?? Infinity));
const twoPhaseRanked = [...attemptRows].sort((left, right) => Number(right.twoPhaseSpeedup ?? -Infinity) - Number(left.twoPhaseSpeedup ?? -Infinity));
const twoPhaseRankedAscending = [...attemptRows].sort((left, right) => Number(left.twoPhaseSpeedup ?? Infinity) - Number(right.twoPhaseSpeedup ?? Infinity));
const semanticRanked = [...attemptRows].sort((left, right) => Number(right.semanticSpeedup ?? -Infinity) - Number(left.semanticSpeedup ?? -Infinity));
const semanticRankedAscending = [...attemptRows].sort((left, right) => Number(left.semanticSpeedup ?? Infinity) - Number(right.semanticSpeedup ?? Infinity));
const best = ranked[0];
const median = rankedAscending[Math.floor(rankedAscending.length / 2)];
const worst = rankedAscending[0];
const commandBest = commandRanked[0];
const commandMedian = commandRankedAscending[Math.floor(commandRankedAscending.length / 2)];
const commandWorst = commandRankedAscending[0];
const twoPhaseBest = twoPhaseRanked[0];
const twoPhaseMedian = twoPhaseRankedAscending[Math.floor(twoPhaseRankedAscending.length / 2)];
const twoPhaseWorst = twoPhaseRankedAscending[0];
const semanticBest = semanticRanked[0];
const semanticMedian = semanticRankedAscending[Math.floor(semanticRankedAscending.length / 2)];
const semanticWorst = semanticRankedAscending[0];
const throughputReady = structuralReady && median.speedup !== null && median.speedup >= speedupFloor;
const commandThroughputReady = commandStructuralReady && commandMedian.commandSpeedup !== null && commandMedian.commandSpeedup >= commandSpeedupFloor;
const commandReady = commandStructuralReady && commandThroughputReady;
const candidateReady = structuralReady && throughputReady;
const noisyAttempts = attemptRows.filter((row) => row.speedup === null || row.speedup < speedupFloor).length;
const commandNoisyAttempts = attemptRows.filter((row) => row.commandSpeedup === null || row.commandSpeedup < commandSpeedupFloor).length;
const twoPhaseNoisyAttempts = attemptRows.filter((row) => row.twoPhaseSpeedup === null || row.twoPhaseSpeedup < commandSpeedupFloor).length;
const semanticNoisyAttempts = attemptRows.filter((row) => row.semanticSpeedup === null || row.semanticSpeedup < commandSpeedupFloor).length;
const dispatchOnlyTrap =
  median.speedup !== null &&
  median.speedup < speedupFloor &&
  ((median.defaultDispatches !== null && median.candidateDispatches !== null && median.candidateDispatches < median.defaultDispatches) ||
    (median.defaultProjectionRowChainDispatchSplit !== null &&
      median.candidateProjectionRowChainDispatchSplit !== null &&
      median.candidateProjectionRowChainDispatchSplit < median.defaultProjectionRowChainDispatchSplit) ||
    median.candidateProjectionRowChainDispatchExcess < median.defaultProjectionRowChainDispatchExcess);
const semanticPairActive =
  Math.max(
    best.defaultProjectionPairs,
    best.commandProjectionPairs,
    best.candidateProjectionPairs,
    best.twoPhaseProjectionPairs,
  ) > 0;
const reason = candidateReady
  ? "projection_row_chain_candidate_meets_structure_and_speed"
  : !measureSingle
    ? "single_dispatch_lane_skipped"
    : dispatchOnlyTrap
    ? "dispatch_reduction_without_tiled_throughput"
    : !structuralReady
      ? "projection_row_chain_candidate_failed_structure_or_fallback"
      : "projection_row_chain_candidate_needs_throughput_kernel";
const commandStructuralStatus = measureCommand ? (commandStructuralReady ? "ready" : "off") : "skipped";
const commandThroughputStatus = measureCommand ? (commandThroughputReady ? "ready" : "off") : "skipped";
const singleStructuralStatus = measureSingle ? (structuralReady ? "ready" : "off") : "skipped";
const singleThroughputStatus = measureSingle ? (throughputReady ? "ready" : "off") : "skipped";
const twoPhaseStructuralStatus = measureTwoPhase ? (twoPhaseStructuralReady ? "ready" : "off") : "skipped";
const semanticStructuralStatus = measureSemantic ? (semanticStructuralReady ? "ready" : "off") : "skipped";
const semanticThroughputStatus = measureSemantic
  ? semanticStructuralReady &&
      semanticMedian.semanticSpeedup !== null &&
      semanticMedian.semanticSpeedup >= semanticSpeedupFloor &&
      semanticWorst.semanticSpeedup !== null &&
      semanticWorst.semanticSpeedup >= commandSpeedupFloor
    ? "ready"
    : "diagnostic"
  : "skipped";
const semanticStructuralSelected = semanticBest.semanticTiledTwoPhaseCount > 0;
const semanticThroughputReady = semanticThroughputStatus === "ready";
const twoPhaseStructuralSelected = twoPhaseBest.twoPhaseTiledTwoPhaseCount > 0;
const commandDispatchReduced =
  commandBest.defaultDispatches !== null && commandBest.commandDispatches !== null && commandBest.commandDispatches < commandBest.defaultDispatches;
const twoPhaseDispatchReduced =
  twoPhaseBest.defaultDispatches !== null && twoPhaseBest.twoPhaseDispatches !== null && twoPhaseBest.twoPhaseDispatches < twoPhaseBest.defaultDispatches;
const semanticDispatchReduced =
  semanticBest.defaultDispatches !== null && semanticBest.semanticDispatches !== null && semanticBest.semanticDispatches < semanticBest.defaultDispatches;
const singleDispatchReduced =
  best.defaultDispatches !== null && best.candidateDispatches !== null && best.candidateDispatches < best.defaultDispatches;
const dispatchRealityTarget = "reduce_actual_dispatch_or_larger_semantic_sublayer";
const fullModelQprojTarget =
  commandBest.defaultProjectionGroups === 0 && commandBest.defaultProjectionChainRowChainFrontiers >= defaultProjectionChainFloor
    ? qprojGroupTarget
    : "inspect_projection_group_scheduler";
const singleAttemptSummary = measureSingle
  ? `attempt=${best.index}/${attempts} median_attempt=${median.index}/${attempts} noisy=${noisyAttempts}`
  : "attempt=skipped median_attempt=skipped noisy=skipped";
const gateStatus = commandReady ? "command-ready" : candidateReady ? "ready" : "structural";

function laneArtifact(row, prefix) {
  const speedupKey = prefix === "candidate" ? "speedup" : `${prefix}Speedup`;
  return {
    tokS: roundMetric(row[`${prefix}TokS`]),
    speedup: roundMetric(row[speedupKey]),
    dispatches: row[`${prefix}Dispatches`],
    commands: row[`${prefix}Commands`],
    projectionChains: row[`${prefix}ProjectionChains`],
    projectionPairs: row[`${prefix}ProjectionPairs`],
    projectionRowChains: row[`${prefix}ProjectionRowChains`],
    projectionRowChainDispatches: row[`${prefix}ProjectionRowChainDispatches`],
    fallback: row[`${prefix}Fallback`],
  };
}

function laneSpeedupStats(bestRow, medianRow, worstRow, prefix) {
  const speedupKey = prefix === "candidate" ? "speedup" : `${prefix}Speedup`;
  return {
    best: roundMetric(bestRow[speedupKey]),
    median: roundMetric(medianRow[speedupKey]),
    worst: roundMetric(worstRow[speedupKey]),
    bestAttempt: bestRow.index,
    medianAttempt: medianRow.index,
    worstAttempt: worstRow.index,
  };
}

let artifactPath = null;
if (writeArtifact) {
  const resolvedArtifactDir = resolve(root, artifactDir);
  mkdirSync(resolvedArtifactDir, { recursive: true });
  artifactPath = join(resolvedArtifactDir, `q8-prompt-${timestampForArtifact()}-${process.pid}.json`);
  const artifact = {
    schema: "zgml.q8-prompt-candidate.v1",
    createdAt: new Date().toISOString(),
    command: "scripts/check_q8_prompt_candidate.cjs",
    nodeVersion: process.version,
    platform: {
      type: os.type(),
      platform: os.platform(),
      arch: os.arch(),
      release: os.release(),
      cpus: os.cpus().length,
    },
    config: {
      model,
      promptTokens,
      genTokens,
      repetitions,
      attempts,
      laneMode,
      measuredLanes: [...measuredLanes],
      speedupFloor,
      commandSpeedupFloor,
      semanticSpeedupFloor,
      build,
      lowerings: {
        command: commandLowering,
        twoPhase: twoPhaseLowering,
        semantic: semanticLowering,
        rowChain: rowChainLowering,
      },
    },
    status: gateStatus,
    structural: {
      command: commandStructuralStatus,
      single: singleStructuralStatus,
      twoPhase: twoPhaseStructuralStatus,
      semantic: semanticStructuralStatus,
    },
    throughput: {
      command: commandThroughputStatus,
      single: singleThroughputStatus,
      semantic: semanticThroughputStatus,
    },
    reason,
    selectedAttempts: {
      command: commandBest.index,
      single: best.index,
      twoPhase: twoPhaseBest.index,
      semantic: semanticBest.index,
    },
    noisyAttempts: {
      command: commandNoisyAttempts,
      single: noisyAttempts,
      twoPhase: twoPhaseNoisyAttempts,
      semantic: semanticNoisyAttempts,
    },
    lanes: {
      command: {
        ...laneArtifact(commandBest, "command"),
        speedupStats: laneSpeedupStats(commandBest, commandMedian, commandWorst, "command"),
      },
      single: {
        ...laneArtifact(best, "candidate"),
        speedupStats: laneSpeedupStats(best, median, worst, "candidate"),
      },
      twoPhase: {
        ...laneArtifact(twoPhaseBest, "twoPhase"),
        speedupStats: laneSpeedupStats(twoPhaseBest, twoPhaseMedian, twoPhaseWorst, "twoPhase"),
        selected: twoPhaseStructuralSelected,
        structuralSelected: twoPhaseStructuralSelected,
        tiledTwoPhaseCount: twoPhaseBest.twoPhaseTiledTwoPhaseCount,
        tiledWork: twoPhaseBest.twoPhaseTiledCount,
        tiledSpills: twoPhaseBest.twoPhaseTiledSpills,
      },
      semantic: {
        ...laneArtifact(semanticBest, "semantic"),
        speedupStats: laneSpeedupStats(semanticBest, semanticMedian, semanticWorst, "semantic"),
        selected: semanticStructuralSelected,
        structuralSelected: semanticStructuralSelected,
        throughputReady: semanticThroughputReady,
        tiledTwoPhaseCount: semanticBest.semanticTiledTwoPhaseCount,
        tiledWork: semanticBest.semanticTiledCount,
        tiledSpills: semanticBest.semanticTiledSpills,
      },
    },
    attempts: attemptRows.map((row) => ({
      index: row.index,
      defaultTokS: roundMetric(row.defaultTokS),
      commandSpeedup: roundMetric(row.commandSpeedup),
      singleSpeedup: roundMetric(row.speedup),
      twoPhaseSpeedup: roundMetric(row.twoPhaseSpeedup),
      semanticSpeedup: roundMetric(row.semanticSpeedup),
      commandCommands: row.commandCommands,
      twoPhaseCommands: row.twoPhaseCommands,
      semanticCommands: row.semanticCommands,
      commandFallback: row.commandFallback,
      twoPhaseFallback: row.twoPhaseFallback,
      semanticFallback: row.semanticFallback,
    })),
  };
  writeFileSync(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
  console.log(`Q8_PROMPT_CANDIDATE_JSON ${JSON.stringify({
    artifact: artifactPath,
    status: gateStatus,
    semantic: semanticThroughputStatus,
    semanticSelected: semanticStructuralSelected,
    semanticStructuralSelected,
    semanticThroughputReady,
    commandSpeedup: roundMetric(commandBest.commandSpeedup),
    commandMedianSpeedup: roundMetric(commandMedian.commandSpeedup),
    semanticSpeedup: roundMetric(semanticBest.semanticSpeedup),
    semanticMedianSpeedup: roundMetric(semanticMedian.semanticSpeedup),
    attempts,
  })}`);
}

console.log(
  `q8 prompt semantic row-chain gate: ${gateStatus}; ` +
    `command_structural=${commandStructuralStatus} command_throughput=${commandThroughputStatus} ` +
    `single_structural=${singleStructuralStatus} single_throughput=${singleThroughputStatus} ` +
    `two_phase_structural=${twoPhaseStructuralStatus} semantic_structural=${semanticStructuralStatus} semantic_throughput=${semanticThroughputStatus} reason=${reason}; ` +
    `${singleAttemptSummary}; ` +
    `command_attempt=${commandBest.index}/${attempts} command_median_attempt=${commandMedian.index}/${attempts} command_noisy=${commandNoisyAttempts}; ` +
    `command_default=${format(commandBest.defaultTokS)} tok/s command_candidate=${format(commandBest.commandTokS)} tok/s command_speedup=${format(commandBest.commandSpeedup)}x command_floor=${format(commandSpeedupFloor)}x; ` +
    `command_median_speedup=${format(commandMedian.commandSpeedup)}x command_worst_speedup=${format(commandWorst.commandSpeedup)}x command_best_speedup=${format(commandBest.commandSpeedup)}x; ` +
    `command_dispatch=${format(commandBest.defaultDispatches, 0)}->${format(commandBest.commandDispatches, 0)} command_command=${format(commandBest.defaultCommands, 0)}->${format(commandBest.commandCommands, 0)} ` +
    `command_dispatch_reduced=${commandDispatchReduced ? "yes" : "no"} command_runtime_target=${dispatchRealityTarget} ` +
    `command_projection_chain=${format(commandBest.defaultProjectionChains, 0)}->${format(commandBest.commandProjectionChains, 0)} ` +
    `command_projection_pair=${format(commandBest.defaultProjectionPairs, 0)}->${format(commandBest.commandProjectionPairs, 0)} ` +
    `command_projection_pair_dispatch=${format(commandBest.defaultProjectionPairDispatches, 0)}->${format(commandBest.commandProjectionPairDispatches, 0)} ` +
    `command_projection_group=${format(commandBest.defaultProjectionGroups, 0)}->${format(commandBest.commandProjectionGroups, 0)} ` +
    `command_projection_group_dispatch=${format(commandBest.defaultProjectionGroupDispatches, 0)}->${format(commandBest.commandProjectionGroupDispatches, 0)} ` +
    `qproj_group_full_model_target=${fullModelQprojTarget} qproj_frontiers=${format(commandBest.defaultProjectionChainRowChainFrontiers, 0)} ` +
    `command_projection_cache_group=${format(commandBest.defaultProjectionCacheGroups, 0)}->${format(commandBest.commandProjectionCacheGroups, 0)} ` +
    `command_projection_cache_group_dispatch=${format(commandBest.defaultProjectionCacheGroupDispatches, 0)}->${format(commandBest.commandProjectionCacheGroupDispatches, 0)} ` +
    `decode_command=${format(commandBest.defaultDecodeCommands, 0)} decode_projection_chain=${format(commandBest.defaultDecodeProjectionChains, 0)} ` +
    `decode_projection_pair=${format(commandBest.defaultDecodeProjectionPairs, 0)} decode_projection_group=${format(commandBest.defaultDecodeProjectionGroups, 0)} ` +
    `decode_projection_cache_group=${format(commandBest.defaultDecodeProjectionCacheGroups, 0)} decode_fallback=${format(commandBest.defaultDecodeFallback, 0)} ` +
    `decode_fast_path=${commandBest.defaultDecodeFastPathReady ? "ready" : "off"} decode_lowering=${decodeLowering} ` +
    `decode_row_chain_default=${decodeRowChainDefault} decode_next=${decodeNextTarget} ` +
    `command_projection_row_chain=${format(commandBest.defaultProjectionRowChains, 0)}->${format(commandBest.commandProjectionRowChains, 0)} ` +
    `command_projection_row_chain_dispatch=${format(commandBest.defaultProjectionRowChainDispatches, 0)}->${format(commandBest.commandProjectionRowChainDispatches, 0)} ` +
    `command_split=${format(commandBest.defaultProjectionRowChainDispatchSplit)}->${format(commandBest.commandProjectionRowChainDispatchSplit)} ` +
    `command_excess_dispatch=${format(commandBest.defaultProjectionRowChainDispatchExcess, 0)}->${format(commandBest.commandProjectionRowChainDispatchExcess, 0)} command_target=0 ` +
    `command_fallback=${format(commandBest.defaultFallback, 0)}->${format(commandBest.commandFallback, 0)} command_lowering=${commandLowering}; ` +
    `two_phase_attempt=${twoPhaseBest.index}/${attempts} two_phase_median_attempt=${twoPhaseMedian.index}/${attempts} two_phase_noisy=${twoPhaseNoisyAttempts}; ` +
    `two_phase_default=${format(twoPhaseBest.defaultTokS)} tok/s two_phase_candidate=${format(twoPhaseBest.twoPhaseTokS)} tok/s two_phase_speedup=${format(twoPhaseBest.twoPhaseSpeedup)}x; ` +
    `two_phase_median_speedup=${format(twoPhaseMedian.twoPhaseSpeedup)}x two_phase_worst_speedup=${format(twoPhaseWorst.twoPhaseSpeedup)}x two_phase_best_speedup=${format(twoPhaseBest.twoPhaseSpeedup)}x; ` +
    `two_phase_dispatch=${format(twoPhaseBest.defaultDispatches, 0)}->${format(twoPhaseBest.twoPhaseDispatches, 0)} two_phase_command=${format(twoPhaseBest.defaultCommands, 0)}->${format(twoPhaseBest.twoPhaseCommands, 0)} ` +
    `two_phase_dispatch_reduced=${twoPhaseDispatchReduced ? "yes" : "no"} two_phase_runtime_target=${dispatchRealityTarget} ` +
    `two_phase_count=${format(twoPhaseBest.twoPhaseTiledTwoPhaseCount, 0)} two_phase_selected=${twoPhaseBest.twoPhaseTiledTwoPhaseCount > 0 ? "yes" : "off"} ` +
    `two_phase_tiled_work=${format(twoPhaseBest.twoPhaseTiledCount, 0)} chains row_groups=${format(twoPhaseBest.twoPhaseTiledRowTileGroups, 0)} n_tiles=${format(twoPhaseBest.twoPhaseTiledNTiles, 0)} serial_tile_loops=${format(twoPhaseBest.twoPhaseTiledSerialLoops, 0)} partial_slots=${format(twoPhaseBest.twoPhaseTiledPartialSlots, 0)} scratch_capacity=${format(twoPhaseBest.twoPhaseTiledScratchCapacity, 0)} spills=${format(twoPhaseBest.twoPhaseTiledSpills, 0)} ` +
    `two_phase_projection_chain=${format(twoPhaseBest.defaultProjectionChains, 0)}->${format(twoPhaseBest.twoPhaseProjectionChains, 0)} ` +
    `two_phase_projection_pair=${format(twoPhaseBest.defaultProjectionPairs, 0)}->${format(twoPhaseBest.twoPhaseProjectionPairs, 0)} ` +
    `two_phase_projection_pair_dispatch=${format(twoPhaseBest.defaultProjectionPairDispatches, 0)}->${format(twoPhaseBest.twoPhaseProjectionPairDispatches, 0)} ` +
    `two_phase_projection_group=${format(twoPhaseBest.defaultProjectionGroups, 0)}->${format(twoPhaseBest.twoPhaseProjectionGroups, 0)} ` +
    `two_phase_projection_group_dispatch=${format(twoPhaseBest.defaultProjectionGroupDispatches, 0)}->${format(twoPhaseBest.twoPhaseProjectionGroupDispatches, 0)} ` +
    `two_phase_projection_cache_group=${format(twoPhaseBest.defaultProjectionCacheGroups, 0)}->${format(twoPhaseBest.twoPhaseProjectionCacheGroups, 0)} ` +
    `two_phase_projection_cache_group_dispatch=${format(twoPhaseBest.defaultProjectionCacheGroupDispatches, 0)}->${format(twoPhaseBest.twoPhaseProjectionCacheGroupDispatches, 0)} ` +
    `two_phase_projection_row_chain=${format(twoPhaseBest.defaultProjectionRowChains, 0)}->${format(twoPhaseBest.twoPhaseProjectionRowChains, 0)} ` +
    `two_phase_projection_row_chain_dispatch=${format(twoPhaseBest.defaultProjectionRowChainDispatches, 0)}->${format(twoPhaseBest.twoPhaseProjectionRowChainDispatches, 0)} ` +
    `two_phase_split=${format(twoPhaseBest.defaultProjectionRowChainDispatchSplit)}->${format(twoPhaseBest.twoPhaseProjectionRowChainDispatchSplit)} ` +
    `two_phase_fallback=${format(twoPhaseBest.defaultFallback, 0)}->${format(twoPhaseBest.twoPhaseFallback, 0)} two_phase_lowering=${twoPhaseLowering}; ` +
    `semantic_attempt=${semanticBest.index}/${attempts} semantic_median_attempt=${semanticMedian.index}/${attempts} semantic_noisy=${semanticNoisyAttempts}; ` +
    `semantic_default=${format(semanticBest.defaultTokS)} tok/s semantic_candidate=${format(semanticBest.semanticTokS)} tok/s semantic_speedup=${format(semanticBest.semanticSpeedup)}x; ` +
    `semantic_median_speedup=${format(semanticMedian.semanticSpeedup)}x semantic_worst_speedup=${format(semanticWorst.semanticSpeedup)}x semantic_best_speedup=${format(semanticBest.semanticSpeedup)}x; ` +
    `semantic_dispatch=${format(semanticBest.defaultDispatches, 0)}->${format(semanticBest.semanticDispatches, 0)} semantic_command=${format(semanticBest.defaultCommands, 0)}->${format(semanticBest.semanticCommands, 0)} ` +
    `semantic_dispatch_reduced=${semanticDispatchReduced ? "yes" : "no"} semantic_runtime_target=${dispatchRealityTarget} ` +
    `semantic_count=${format(semanticBest.semanticTiledTwoPhaseCount, 0)} semantic_structural_selected=${semanticStructuralSelected ? "yes" : "off"} semantic_throughput_ready=${semanticThroughputReady ? "yes" : "off"} semantic_selected=${semanticStructuralSelected ? "yes" : "off"} ` +
    `semantic_tiled_work=${format(semanticBest.semanticTiledCount, 0)} chains row_groups=${format(semanticBest.semanticTiledRowTileGroups, 0)} n_tiles=${format(semanticBest.semanticTiledNTiles, 0)} serial_tile_loops=${format(semanticBest.semanticTiledSerialLoops, 0)} partial_slots=${format(semanticBest.semanticTiledPartialSlots, 0)} scratch_capacity=${format(semanticBest.semanticTiledScratchCapacity, 0)} spills=${format(semanticBest.semanticTiledSpills, 0)} ` +
    `semantic_projection_chain=${format(semanticBest.defaultProjectionChains, 0)}->${format(semanticBest.semanticProjectionChains, 0)} ` +
    `semantic_projection_pair=${format(semanticBest.defaultProjectionPairs, 0)}->${format(semanticBest.semanticProjectionPairs, 0)} ` +
    `semantic_projection_pair_dispatch=${format(semanticBest.defaultProjectionPairDispatches, 0)}->${format(semanticBest.semanticProjectionPairDispatches, 0)} ` +
    `semantic_projection_group=${format(semanticBest.defaultProjectionGroups, 0)}->${format(semanticBest.semanticProjectionGroups, 0)} ` +
    `semantic_projection_group_dispatch=${format(semanticBest.defaultProjectionGroupDispatches, 0)}->${format(semanticBest.semanticProjectionGroupDispatches, 0)} ` +
    `semantic_projection_cache_group=${format(semanticBest.defaultProjectionCacheGroups, 0)}->${format(semanticBest.semanticProjectionCacheGroups, 0)} ` +
    `semantic_projection_cache_group_dispatch=${format(semanticBest.defaultProjectionCacheGroupDispatches, 0)}->${format(semanticBest.semanticProjectionCacheGroupDispatches, 0)} ` +
    `semantic_projection_row_chain=${format(semanticBest.defaultProjectionRowChains, 0)}->${format(semanticBest.semanticProjectionRowChains, 0)} ` +
    `semantic_projection_row_chain_dispatch=${format(semanticBest.defaultProjectionRowChainDispatches, 0)}->${format(semanticBest.semanticProjectionRowChainDispatches, 0)} ` +
    `semantic_split=${format(semanticBest.defaultProjectionRowChainDispatchSplit)}->${format(semanticBest.semanticProjectionRowChainDispatchSplit)} ` +
    `semantic_fallback=${format(semanticBest.defaultFallback, 0)}->${format(semanticBest.semanticFallback, 0)} semantic_lowering=${semanticLowering}; ` +
    `default=${format(best.defaultTokS)} tok/s candidate=${format(best.candidateTokS)} tok/s speedup=${format(best.speedup)}x floor=${format(speedupFloor)}x; ` +
    `median_speedup=${format(median.speedup)}x worst_speedup=${format(worst.speedup)}x best_speedup=${format(best.speedup)}x; ` +
    `dispatch=${format(best.defaultDispatches, 0)}->${format(best.candidateDispatches, 0)} command=${format(best.defaultCommands, 0)}->${format(best.candidateCommands, 0)} ` +
    `dispatch_reduced=${singleDispatchReduced ? "yes" : "no"} runtime_target=${dispatchRealityTarget} ` +
    `projection_chain=${format(best.defaultProjectionChains, 0)}->${format(best.candidateProjectionChains, 0)} ` +
    `projection_pair=${format(best.defaultProjectionPairs, 0)}->${format(best.candidateProjectionPairs, 0)} ` +
    `projection_pair_dispatch=${format(best.defaultProjectionPairDispatches, 0)}->${format(best.candidateProjectionPairDispatches, 0)} ` +
    `projection_group=${format(best.defaultProjectionGroups, 0)}->${format(best.candidateProjectionGroups, 0)} ` +
    `projection_group_dispatch=${format(best.defaultProjectionGroupDispatches, 0)}->${format(best.candidateProjectionGroupDispatches, 0)} ` +
    `projection_cache_group=${format(best.defaultProjectionCacheGroups, 0)}->${format(best.candidateProjectionCacheGroups, 0)} ` +
    `projection_cache_group_dispatch=${format(best.defaultProjectionCacheGroupDispatches, 0)}->${format(best.candidateProjectionCacheGroupDispatches, 0)} ` +
    `semantic_pair_path=${semanticPairActive ? "active" : "absent"} semantic_pair_target=${semanticPairTarget} ` +
    `projection_row_chain=${format(best.defaultProjectionRowChains, 0)}->${format(best.candidateProjectionRowChains, 0)} ` +
    `projection_row_chain_dispatch=${format(best.defaultProjectionRowChainDispatches, 0)}->${format(best.candidateProjectionRowChainDispatches, 0)} ` +
    `split=${format(best.defaultProjectionRowChainDispatchSplit)}->${format(best.candidateProjectionRowChainDispatchSplit)} ` +
    `excess_dispatch=${format(best.defaultProjectionRowChainDispatchExcess, 0)}->${format(best.candidateProjectionRowChainDispatchExcess, 0)} target=0 ` +
    `tiled_work=${format(best.candidateTiledCount, 0)} chains row_groups=${format(best.candidateTiledRowTileGroups, 0)} n_tiles=${format(best.candidateTiledNTiles, 0)} serial_tile_loops=${format(best.candidateTiledSerialLoops, 0)} partial_slots=${format(best.candidateTiledPartialSlots, 0)} scratch_capacity=${format(best.candidateTiledScratchCapacity, 0)} two_phase_scratch=${best.twoPhaseScratchReady ? "ready" : "off"} spills=${format(best.candidateTiledSpills, 0)} ` +
    `dispatch_only_trap=${dispatchOnlyTrap ? "yes" : "no"} ` +
    `fallback=${format(best.defaultFallback, 0)}->${format(best.candidateFallback, 0)} ` +
    `row_chain_lowering=${rowChainLowering} row_chain_next=${requiredNextTarget} ` +
    `single_dispatch_trap=${dispatchOnlyTrap ? singleDispatchTrap : "none"} viable_next=${viableNextTarget} ` +
    `next=${requiredNextTarget}`,
);

if (!commandStructuralReady || !structuralReady || !twoPhaseStructuralReady || !semanticStructuralReady) {
  process.exit(1);
}
