"use strict";

const { existsSync } = require("node:fs");
const { spawnSync } = require("node:child_process");
const { resolve } = require("node:path");

const root = resolve(__dirname, "..");
const model = process.env.ZGML_Q8_MODEL || process.env.ZGML_MODEL || "data/smollm/SmolLM-135M.Q8_0.gguf";
const promptTokens = process.env.BENCH_CANDIDATE_PROMPT || "128";
const genTokens = process.env.BENCH_CANDIDATE_GEN || "40";
const repetitions = process.env.BENCH_CANDIDATE_REPS || "1";
const build = process.env.BENCH_BUILD_ZGML ?? "1";
const binary = "./zig-out/bin/bench-llama-smollm";
const speedupFloor = Number(process.env.BENCH_CANDIDATE_SPEEDUP_FLOOR || "1.05");
const commandSpeedupFloor = Number(process.env.BENCH_COMMAND_CANDIDATE_SPEEDUP_FLOOR || "0.95");
const attempts = positiveInt(process.env.BENCH_CANDIDATE_ATTEMPTS || "3", "BENCH_CANDIDATE_ATTEMPTS");
const rowChainLowering = "default_projection_chain_plus_row_chain_candidate_single_dispatch_tiled_row_chain";
const commandLowering = "default_projection_chain_plus_row_chain_command_two_dispatch";
const requiredNextTarget = "single_dispatch_tiled_qmatmul_row_chain_throughput";
const defaultCommandFloor = Number(process.env.BENCH_Q8_PROMPT_DEFAULT_COMMAND_FLOOR || "301");
const candidateCommandCeil = Number(process.env.BENCH_Q8_PROMPT_COMMAND_CEIL || "241");
const candidateProjectionRowChainFloor = Number(process.env.BENCH_Q8_PROMPT_PROJECTION_ROW_CHAIN_FLOOR || "60");
const defaultProjectionChainFloor = Number(process.env.BENCH_Q8_PROMPT_DEFAULT_PROJECTION_CHAIN_FLOOR || "90");
const candidateProjectionChainCeil = Number(process.env.BENCH_Q8_PROMPT_PROJECTION_CHAIN_CEIL || "30");

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

function positiveInt(value, label) {
  const n = Number(value);
  if (!Number.isSafeInteger(n) || n <= 0) {
    console.error(`${label} must be a positive integer, got ${value}`);
    process.exit(1);
  }
  return n;
}

if (!existsSync(resolve(root, model))) {
  console.error(`q8 prompt candidate gate: missing model ${model}`);
  console.error("Set ZGML_Q8_MODEL/ZGML_MODEL or run with BENCH_AUTO_DOWNLOAD through scripts/bench_vs_ggml.sh first.");
  process.exit(1);
}

if (build === "1") {
  run("zig", ["build", "bench-build"]);
} else if (build !== "0") {
  console.error("BENCH_BUILD_ZGML must be 0 or 1");
  process.exit(1);
}

const baseArgs = [model, promptTokens, genTokens, repetitions, "--metal-prefill-device", "--gate-only"];

function measureAttempt(index) {
  const defaultOutput = run(binary, baseArgs);
  const commandOutput = run(binary, [...baseArgs, "--metal-prompt-projection-row-chain-command-candidate"]);
  const candidateOutput = run(binary, [...baseArgs, "--metal-prompt-projection-row-chain-candidate"]);
  const defaultRow = rowFor(defaultOutput, "metal scheduled prefill");
  const commandRow = rowFor(commandOutput, "metal scheduled prefill projection-row-chain command candidate");
  const candidateRow = rowFor(candidateOutput, "metal scheduled prefill projection-row-chain candidate");

  const defaultTokS = number(defaultRow, "prompt_tok_s");
  const commandTokS = number(commandRow, "prompt_tok_s");
  const candidateTokS = number(candidateRow, "prompt_tok_s");
  const commandSpeedup = defaultTokS && commandTokS ? commandTokS / defaultTokS : null;
  const speedup = defaultTokS && candidateTokS ? candidateTokS / defaultTokS : null;
  const defaultDispatches = number(defaultRow, "dispatches_per_call");
  const commandDispatches = number(commandRow, "dispatches_per_call");
  const candidateDispatches = number(candidateRow, "dispatches_per_call");
  const defaultCommands = number(defaultRow, "commands_per_call");
  const commandCommands = number(commandRow, "commands_per_call");
  const candidateCommands = number(candidateRow, "commands_per_call");
  const defaultProjectionRowChains = number(defaultRow, "program_command_encoded_projection_row_chain_per_call") ?? 0;
  const commandProjectionRowChains = number(commandRow, "program_command_encoded_projection_row_chain_per_call") ?? 0;
  const candidateProjectionRowChains = number(candidateRow, "program_command_encoded_projection_row_chain_per_call") ?? 0;
  const defaultProjectionChains = number(defaultRow, "program_command_encoded_projection_chain_per_call") ?? 0;
  const commandProjectionChains = number(commandRow, "program_command_encoded_projection_chain_per_call") ?? 0;
  const candidateProjectionChains = number(candidateRow, "program_command_encoded_projection_chain_per_call") ?? 0;
  const defaultProjectionRowChainDispatches = number(defaultRow, "program_command_dispatches_projection_row_chain_per_call") ?? 0;
  const commandProjectionRowChainDispatches = number(commandRow, "program_command_dispatches_projection_row_chain_per_call") ?? 0;
  const candidateProjectionRowChainDispatches = number(candidateRow, "program_command_dispatches_projection_row_chain_per_call") ?? 0;
  const defaultProjectionRowChainDispatchSplit = defaultProjectionRowChains > 0 ? defaultProjectionRowChainDispatches / defaultProjectionRowChains : null;
  const commandProjectionRowChainDispatchSplit = commandProjectionRowChains > 0 ? commandProjectionRowChainDispatches / commandProjectionRowChains : null;
  const candidateProjectionRowChainDispatchSplit = candidateProjectionRowChains > 0 ? candidateProjectionRowChainDispatches / candidateProjectionRowChains : null;
  const defaultProjectionRowChainDispatchExcess = Math.max(0, defaultProjectionRowChainDispatches - defaultProjectionRowChains);
  const commandProjectionRowChainDispatchExcess = Math.max(0, commandProjectionRowChainDispatches - commandProjectionRowChains);
  const candidateProjectionRowChainDispatchExcess = Math.max(0, candidateProjectionRowChainDispatches - candidateProjectionRowChains);
  const commandFallback = number(commandRow, "fallback_ops") ?? 0;
  const candidateFallback = number(candidateRow, "fallback_ops") ?? 0;
  const defaultFallback = number(defaultRow, "fallback_ops") ?? 0;
  const defaultFastPathReady =
    defaultCommands !== null &&
    defaultCommands >= defaultCommandFloor &&
    defaultProjectionChains >= defaultProjectionChainFloor &&
    defaultProjectionRowChains === 0;
  const commandSemanticReady =
    commandCommands !== null &&
    commandCommands <= candidateCommandCeil &&
    commandProjectionChains <= candidateProjectionChainCeil &&
    commandProjectionRowChains >= candidateProjectionRowChainFloor;
  const commandDispatchShapeReady =
    defaultDispatches !== null &&
    commandDispatches !== null &&
    commandDispatches <= defaultDispatches &&
    commandProjectionRowChainDispatches >= commandProjectionRowChains &&
    commandProjectionRowChainDispatchSplit !== null &&
    commandProjectionRowChainDispatchSplit <= 2.0;
  const candidateSemanticReady =
    candidateCommands !== null &&
    candidateCommands <= candidateCommandCeil &&
    candidateProjectionChains <= candidateProjectionChainCeil &&
    candidateProjectionRowChains >= candidateProjectionRowChainFloor;
  const candidateMatchesCommandShape =
    candidateCommands !== null &&
    candidateCommands <= candidateCommandCeil &&
    candidateProjectionChains <= candidateProjectionChainCeil &&
    candidateProjectionRowChains >= candidateProjectionRowChainFloor;
  const candidateDispatchShapeReady =
    defaultDispatches !== null &&
    candidateDispatches !== null &&
    candidateDispatches < defaultDispatches &&
    candidateProjectionRowChainDispatches <= candidateProjectionRowChains;
  const fallbackOk = defaultFallback === 0 && commandFallback === 0 && candidateFallback === 0;
  const commandStructuralReady = defaultFastPathReady && commandSemanticReady && commandDispatchShapeReady && fallbackOk;
  const structuralReady = defaultFastPathReady && commandStructuralReady && candidateSemanticReady && candidateMatchesCommandShape && candidateDispatchShapeReady && fallbackOk;
  const commandThroughputReady = commandSpeedup !== null && commandSpeedup >= commandSpeedupFloor;
  const throughputReady = speedup !== null && speedup >= speedupFloor;
  return {
    index,
    defaultTokS,
    commandTokS,
    candidateTokS,
    commandSpeedup,
    speedup,
    defaultDispatches,
    commandDispatches,
    candidateDispatches,
    defaultCommands,
    commandCommands,
    candidateCommands,
    defaultProjectionRowChains,
    commandProjectionRowChains,
    candidateProjectionRowChains,
    defaultProjectionChains,
    commandProjectionChains,
    candidateProjectionChains,
    defaultProjectionRowChainDispatches,
    commandProjectionRowChainDispatches,
    candidateProjectionRowChainDispatches,
    defaultProjectionRowChainDispatchSplit,
    commandProjectionRowChainDispatchSplit,
    candidateProjectionRowChainDispatchSplit,
    defaultProjectionRowChainDispatchExcess,
    commandProjectionRowChainDispatchExcess,
    candidateProjectionRowChainDispatchExcess,
    defaultFallback,
    commandFallback,
    candidateFallback,
    commandStructuralReady,
    commandThroughputReady,
    candidateMatchesCommandShape,
    candidateDispatchShapeReady,
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
const ranked = [...attemptRows].sort((left, right) => Number(right.speedup ?? -Infinity) - Number(left.speedup ?? -Infinity));
const rankedAscending = [...attemptRows].sort((left, right) => Number(left.speedup ?? Infinity) - Number(right.speedup ?? Infinity));
const commandRanked = [...attemptRows].sort((left, right) => Number(right.commandSpeedup ?? -Infinity) - Number(left.commandSpeedup ?? -Infinity));
const commandRankedAscending = [...attemptRows].sort((left, right) => Number(left.commandSpeedup ?? Infinity) - Number(right.commandSpeedup ?? Infinity));
const best = ranked[0];
const median = rankedAscending[Math.floor(rankedAscending.length / 2)];
const worst = rankedAscending[0];
const commandBest = commandRanked[0];
const commandMedian = commandRankedAscending[Math.floor(commandRankedAscending.length / 2)];
const commandWorst = commandRankedAscending[0];
const throughputReady = structuralReady && median.speedup !== null && median.speedup >= speedupFloor;
const commandThroughputReady = commandStructuralReady && commandMedian.commandSpeedup !== null && commandMedian.commandSpeedup >= commandSpeedupFloor;
const commandReady = commandStructuralReady && commandThroughputReady;
const candidateReady = structuralReady && throughputReady;
const noisyAttempts = attemptRows.filter((row) => row.speedup === null || row.speedup < speedupFloor).length;
const commandNoisyAttempts = attemptRows.filter((row) => row.commandSpeedup === null || row.commandSpeedup < commandSpeedupFloor).length;
const dispatchOnlyTrap =
  median.speedup !== null &&
  median.speedup < speedupFloor &&
  ((median.defaultDispatches !== null && median.candidateDispatches !== null && median.candidateDispatches < median.defaultDispatches) ||
    (median.defaultProjectionRowChainDispatchSplit !== null &&
      median.candidateProjectionRowChainDispatchSplit !== null &&
      median.candidateProjectionRowChainDispatchSplit < median.defaultProjectionRowChainDispatchSplit) ||
    median.candidateProjectionRowChainDispatchExcess < median.defaultProjectionRowChainDispatchExcess);
const reason = candidateReady
  ? "projection_row_chain_candidate_meets_structure_and_speed"
  : dispatchOnlyTrap
    ? "dispatch_reduction_without_tiled_throughput"
    : !structuralReady
      ? "projection_row_chain_candidate_failed_structure_or_fallback"
      : "projection_row_chain_candidate_needs_throughput_kernel";

console.log(
  `q8 prompt semantic row-chain gate: ${commandReady ? "command-ready" : candidateReady ? "ready" : "structural"}; ` +
    `command_structural=${commandStructuralReady ? "ready" : "off"} command_throughput=${commandThroughputReady ? "ready" : "off"} ` +
    `single_structural=${structuralReady ? "ready" : "off"} single_throughput=${throughputReady ? "ready" : "off"} reason=${reason}; ` +
    `attempt=${best.index}/${attempts} median_attempt=${median.index}/${attempts} noisy=${noisyAttempts}; ` +
    `command_attempt=${commandBest.index}/${attempts} command_median_attempt=${commandMedian.index}/${attempts} command_noisy=${commandNoisyAttempts}; ` +
    `command_default=${format(commandBest.defaultTokS)} tok/s command_candidate=${format(commandBest.commandTokS)} tok/s command_speedup=${format(commandBest.commandSpeedup)}x command_floor=${format(commandSpeedupFloor)}x; ` +
    `command_median_speedup=${format(commandMedian.commandSpeedup)}x command_worst_speedup=${format(commandWorst.commandSpeedup)}x command_best_speedup=${format(commandBest.commandSpeedup)}x; ` +
    `command_dispatch=${format(commandBest.defaultDispatches, 0)}->${format(commandBest.commandDispatches, 0)} command_command=${format(commandBest.defaultCommands, 0)}->${format(commandBest.commandCommands, 0)} ` +
    `command_projection_chain=${format(commandBest.defaultProjectionChains, 0)}->${format(commandBest.commandProjectionChains, 0)} ` +
    `command_projection_row_chain=${format(commandBest.defaultProjectionRowChains, 0)}->${format(commandBest.commandProjectionRowChains, 0)} ` +
    `command_projection_row_chain_dispatch=${format(commandBest.defaultProjectionRowChainDispatches, 0)}->${format(commandBest.commandProjectionRowChainDispatches, 0)} ` +
    `command_split=${format(commandBest.defaultProjectionRowChainDispatchSplit)}->${format(commandBest.commandProjectionRowChainDispatchSplit)} ` +
    `command_excess_dispatch=${format(commandBest.defaultProjectionRowChainDispatchExcess, 0)}->${format(commandBest.commandProjectionRowChainDispatchExcess, 0)} command_target=0 ` +
    `command_fallback=${format(commandBest.defaultFallback, 0)}->${format(commandBest.commandFallback, 0)} command_lowering=${commandLowering}; ` +
    `default=${format(best.defaultTokS)} tok/s candidate=${format(best.candidateTokS)} tok/s speedup=${format(best.speedup)}x floor=${format(speedupFloor)}x; ` +
    `median_speedup=${format(median.speedup)}x worst_speedup=${format(worst.speedup)}x best_speedup=${format(best.speedup)}x; ` +
    `dispatch=${format(best.defaultDispatches, 0)}->${format(best.candidateDispatches, 0)} command=${format(best.defaultCommands, 0)}->${format(best.candidateCommands, 0)} ` +
    `projection_chain=${format(best.defaultProjectionChains, 0)}->${format(best.candidateProjectionChains, 0)} ` +
    `projection_row_chain=${format(best.defaultProjectionRowChains, 0)}->${format(best.candidateProjectionRowChains, 0)} ` +
    `projection_row_chain_dispatch=${format(best.defaultProjectionRowChainDispatches, 0)}->${format(best.candidateProjectionRowChainDispatches, 0)} ` +
    `split=${format(best.defaultProjectionRowChainDispatchSplit)}->${format(best.candidateProjectionRowChainDispatchSplit)} ` +
    `excess_dispatch=${format(best.defaultProjectionRowChainDispatchExcess, 0)}->${format(best.candidateProjectionRowChainDispatchExcess, 0)} target=0 ` +
    `dispatch_only_trap=${dispatchOnlyTrap ? "yes" : "no"} ` +
    `fallback=${format(best.defaultFallback, 0)}->${format(best.candidateFallback, 0)} ` +
    `row_chain_lowering=${rowChainLowering} row_chain_next=${requiredNextTarget} ` +
    `next=${requiredNextTarget}`,
);

if (!commandReady || !structuralReady) {
  process.exit(1);
}
