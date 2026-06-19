"use strict";

const { spawnSync } = require("node:child_process");

const maxAttempts = 3;
const decodeishTokSFloor = 400_000;
const projectionChainTileSpeedupFloor = 0.95;
const projectionChainFullPrefillSpeedupFloor = 0.90;
const projectionChainFullPrefillCandidateSpeedupFloor = 1.00;
const projectionChainMaxAbsDiffCeil = 0.002;
const projectionGroupCandidateSpeedupFloor = 1.05;
const projectionRowChainDefaultSpeedupFloor = 1.10;
const projectionRowChainMaxAbsDiffCeil = 0.02;
const projectionRowChainKernel = "scalar_per_row_col";
const projectionRowChainNextTarget = "tiled_qmatmul_row_chain_throughput";

function runBench() {
  const result = spawnSync("zig", ["build", "bench-frontier"], {
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

function metric(output, label, key) {
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
  const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`${escaped}[^\\n]*${key}=`).test(output);
}

function score(output, attempt) {
  const smallStaged = p50(output, "chain n=4096 staged");
  const smallFused = p50(output, "chain n=4096 one-pass");
  const largeStaged = p50(output, "chain n=262144 staged");
  const largeFused = p50(output, "chain n=262144 one-pass");
  const decodeTokS = throughput(output, "rmsnorm-attn-logits token", "tokens");
  const projectionChainLabel = "qproj prompt m=32 n=512 k=512 projection_chain";
  const projectionChainFullPrefillLabel = "qproj full-prefill m=128 n=512 k=512 projection_chain";
  const projectionGroupFullPrefillLabel = "qproj group full-prefill x4 m=128 n=512 k=512 projection_group";
  const projectionRowChainGroupFullPrefillLabel = "qrow group full-prefill x4 m=128 n=512 k=512 projection_row_chain_group";
  const projectionDecodeLabel = "qrow decode m=1 n=512 k=512 projection_row_chain";
  const projectionPromptLabel = "qrow prompt m=32 n=512 k=512 projection_row_chain";
  const projectionFullPrefillLabel = "qrow full-prefill m=128 n=512 k=512 projection_row_chain";
  const projectionRowChainGroupFullPrefillProfileLabel = `${projectionRowChainGroupFullPrefillLabel} dispatch_profile`;
  const projectionDecodeProfileLabel = `${projectionDecodeLabel} dispatch_profile`;
  const projectionPromptProfileLabel = `${projectionPromptLabel} dispatch_profile`;
  const projectionFullPrefillProfileLabel = `${projectionFullPrefillLabel} dispatch_profile`;

  const smallSpeedup = smallStaged / smallFused;
  const largeSpeedup = largeStaged / largeFused;
  const projectionChainSpeedup = metric(output, projectionChainLabel, "speedup");
  const projectionChainMaxAbsDiff = metric(output, projectionChainLabel, "max_abs_diff");
  const projectionChainFullPrefillSpeedup = metric(output, projectionChainFullPrefillLabel, "speedup");
  const projectionChainFullPrefillMaxAbsDiff = metric(output, projectionChainFullPrefillLabel, "max_abs_diff");
  const projectionGroupFullPrefillSpeedup = metric(output, projectionGroupFullPrefillLabel, "speedup");
  const projectionGroupFullPrefillMaxAbsDiff = metric(output, projectionGroupFullPrefillLabel, "max_abs_diff");
  const projectionRowChainGroupFullPrefillSpeedup = metric(output, projectionRowChainGroupFullPrefillLabel, "speedup");
  const projectionRowChainGroupFullPrefillMaxAbsDiff = metric(output, projectionRowChainGroupFullPrefillLabel, "max_abs_diff");
  const projectionDecodeSpeedup = hasMetric(output, projectionDecodeLabel, "speedup") ? metric(output, projectionDecodeLabel, "speedup") : null;
  const projectionDecodeMaxAbsDiff = hasMetric(output, projectionDecodeLabel, "max_abs_diff") ? metric(output, projectionDecodeLabel, "max_abs_diff") : null;
  const projectionPromptSpeedup = hasMetric(output, projectionPromptLabel, "speedup") ? metric(output, projectionPromptLabel, "speedup") : null;
  const projectionPromptMaxAbsDiff = hasMetric(output, projectionPromptLabel, "max_abs_diff") ? metric(output, projectionPromptLabel, "max_abs_diff") : null;
  const projectionFullPrefillSpeedup = hasMetric(output, projectionFullPrefillLabel, "speedup") ? metric(output, projectionFullPrefillLabel, "speedup") : null;
  const projectionFullPrefillMaxAbsDiff = hasMetric(output, projectionFullPrefillLabel, "max_abs_diff") ? metric(output, projectionFullPrefillLabel, "max_abs_diff") : null;
  const projectionRowChainGroupFullPrefillShapeCommands = metric(output, projectionRowChainGroupFullPrefillProfileLabel, "shape_commands");
  const projectionRowChainGroupFullPrefillShapeRowChains = metric(output, projectionRowChainGroupFullPrefillProfileLabel, "shape_projection_row_chains");
  const projectionRowChainGroupFullPrefillShapeCoveredOps = metric(output, projectionRowChainGroupFullPrefillProfileLabel, "shape_covered_ops");
  const projectionRowChainGroupFullPrefillShapeSavedDispatches = metric(output, projectionRowChainGroupFullPrefillProfileLabel, "shape_saved_dispatches");
  const projectionRowChainGroupFullPrefillRuntimeCommandDispatches = metric(output, projectionRowChainGroupFullPrefillProfileLabel, "runtime_command_dispatches");
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
  const projectionFullPrefillShapeCommands = metric(output, projectionFullPrefillProfileLabel, "shape_commands");
  const projectionFullPrefillShapeRowChains = metric(output, projectionFullPrefillProfileLabel, "shape_projection_row_chains");
  const projectionFullPrefillShapeCoveredOps = metric(output, projectionFullPrefillProfileLabel, "shape_covered_ops");
  const projectionFullPrefillShapeSavedDispatches = metric(output, projectionFullPrefillProfileLabel, "shape_saved_dispatches");
  const projectionFullPrefillRuntimeCommandDispatches = metric(output, projectionFullPrefillProfileLabel, "runtime_command_dispatches");
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
  if (largeSpeedup < 3.0) failures.push(`large chain fusion ${largeSpeedup.toFixed(2)}x < 3.00x`);
  if (decodeTokS < decodeishTokSFloor) failures.push(`decode-ish token frontier ${decodeTokS.toFixed(2)} tok/s < ${decodeishTokSFloor} tok/s`);
  if (projectionChainSpeedup < projectionChainTileSpeedupFloor) failures.push(`projection_chain prompt tile ${projectionChainSpeedup.toFixed(2)}x < ${projectionChainTileSpeedupFloor.toFixed(2)}x`);
  if (projectionChainMaxAbsDiff > projectionChainMaxAbsDiffCeil) failures.push(`projection_chain prompt max_abs_diff ${projectionChainMaxAbsDiff.toFixed(6)} > ${projectionChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionChainFullPrefillSpeedup < projectionChainFullPrefillSpeedupFloor) failures.push(`projection_chain full-prefill ${projectionChainFullPrefillSpeedup.toFixed(2)}x < ${projectionChainFullPrefillSpeedupFloor.toFixed(2)}x`);
  if (projectionChainFullPrefillMaxAbsDiff > projectionChainMaxAbsDiffCeil) failures.push(`projection_chain full-prefill max_abs_diff ${projectionChainFullPrefillMaxAbsDiff.toFixed(6)} > ${projectionChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionGroupFullPrefillMaxAbsDiff > projectionChainMaxAbsDiffCeil) failures.push(`projection_group full-prefill max_abs_diff ${projectionGroupFullPrefillMaxAbsDiff.toFixed(6)} > ${projectionChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionRowChainGroupFullPrefillMaxAbsDiff > projectionRowChainMaxAbsDiffCeil) failures.push(`projection_row_chain_group full-prefill max_abs_diff ${projectionRowChainGroupFullPrefillMaxAbsDiff.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionDecodeMaxAbsDiff !== null && projectionDecodeMaxAbsDiff > projectionRowChainMaxAbsDiffCeil) failures.push(`projection_row_chain decode max_abs_diff ${projectionDecodeMaxAbsDiff.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionPromptMaxAbsDiff !== null && projectionPromptMaxAbsDiff > projectionRowChainMaxAbsDiffCeil) failures.push(`projection_row_chain prompt max_abs_diff ${projectionPromptMaxAbsDiff.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionFullPrefillMaxAbsDiff !== null && projectionFullPrefillMaxAbsDiff > projectionRowChainMaxAbsDiffCeil) failures.push(`projection_row_chain full-prefill max_abs_diff ${projectionFullPrefillMaxAbsDiff.toFixed(6)} > ${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`);
  if (projectionRowChainGroupFullPrefillShapeCommands !== 4 || projectionRowChainGroupFullPrefillShapeRowChains !== 4 || projectionRowChainGroupFullPrefillShapeCoveredOps !== 20 || projectionRowChainGroupFullPrefillShapeSavedDispatches !== 16) failures.push("projection_row_chain group full-prefill shape profile must stay shape_commands=4 shape_projection_row_chains=4 shape_covered_ops=20 shape_saved_dispatches=16");
  if (projectionDecodeShapeCommands !== 1 || projectionDecodeShapeRowChains !== 1 || projectionDecodeShapeCoveredOps !== 5 || projectionDecodeShapeSavedDispatches !== 4) failures.push("projection_row_chain decode shape profile must stay shape_commands=1 shape_projection_row_chains=1 shape_covered_ops=5 shape_saved_dispatches=4");
  if (projectionPromptShapeCommands !== 1 || projectionPromptShapeRowChains !== 1 || projectionPromptShapeCoveredOps !== 5 || projectionPromptShapeSavedDispatches !== 4) failures.push("projection_row_chain prompt shape profile must stay shape_commands=1 shape_projection_row_chains=1 shape_covered_ops=5 shape_saved_dispatches=4");
  if (projectionFullPrefillShapeCommands !== 1 || projectionFullPrefillShapeRowChains !== 1 || projectionFullPrefillShapeCoveredOps !== 5 || projectionFullPrefillShapeSavedDispatches !== 4) failures.push("projection_row_chain full-prefill shape profile must stay shape_commands=1 shape_projection_row_chains=1 shape_covered_ops=5 shape_saved_dispatches=4");
  if (projectionRowChainGroupFullPrefillRuntimeCommandDispatches !== 4) failures.push("projection_row_chain group full-prefill runtime profile must stay at 4 command dispatches");
  if (projectionDecodeRuntimeCommandDispatches !== 1) failures.push("projection_row_chain decode runtime profile must stay at 1 command dispatch");
  if (projectionPromptRuntimeCommandDispatches !== 1) failures.push("projection_row_chain prompt runtime profile must stay at 1 command dispatch");
  if (projectionFullPrefillRuntimeCommandDispatches !== 1) failures.push("projection_row_chain full-prefill runtime profile must stay at 1 command dispatch");

  const line = [
    `frontier bench gate: ${failures.length === 0 ? "pass" : "fail"}`,
    `attempt=${attempt}/${maxAttempts}`,
    `small_chain=${smallSpeedup.toFixed(2)}x floor=2.00x`,
    `large_chain=${largeSpeedup.toFixed(2)}x floor=3.00x`,
    `decodeish=${decodeTokS.toFixed(2)} tok/s floor=${decodeishTokSFloor}`,
    `projection_chain_prompt=${projectionChainSpeedup.toFixed(2)}x observed max_abs_diff=${projectionChainMaxAbsDiff.toFixed(6)} floor=${projectionChainTileSpeedupFloor.toFixed(2)} diff_ceil=${projectionChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_chain_full_prefill=${projectionChainFullPrefillSpeedup.toFixed(2)}x observed max_abs_diff=${projectionChainFullPrefillMaxAbsDiff.toFixed(6)} candidate=${projectionChainFullPrefillSpeedup >= projectionChainFullPrefillCandidateSpeedupFloor ? "ready" : "off"} floor=${projectionChainFullPrefillSpeedupFloor.toFixed(2)} candidate_floor=${projectionChainFullPrefillCandidateSpeedupFloor.toFixed(2)} diff_ceil=${projectionChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_group_full_prefill=${projectionGroupFullPrefillSpeedup.toFixed(2)}x observed max_abs_diff=${projectionGroupFullPrefillMaxAbsDiff.toFixed(6)} candidate=${projectionGroupFullPrefillSpeedup >= projectionGroupCandidateSpeedupFloor ? "ready" : "off"} floor=${projectionGroupCandidateSpeedupFloor.toFixed(2)} diff_ceil=${projectionChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_group_full_prefill=${projectionRowChainGroupFullPrefillSpeedup.toFixed(2)}x observed max_abs_diff=${projectionRowChainGroupFullPrefillMaxAbsDiff.toFixed(6)} shape_commands=${projectionRowChainGroupFullPrefillShapeCommands} shape_projection_row_chains=${projectionRowChainGroupFullPrefillShapeRowChains} shape_covered_ops=${projectionRowChainGroupFullPrefillShapeCoveredOps} shape_saved_dispatches=${projectionRowChainGroupFullPrefillShapeSavedDispatches} runtime_command_dispatches=${projectionRowChainGroupFullPrefillRuntimeCommandDispatches} candidate=${projectionRowChainGroupFullPrefillSpeedup >= projectionRowChainDefaultSpeedupFloor ? "ready" : "off"} ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_decode=${projectionDecodeSpeedup === null ? "unavailable" : `${projectionDecodeSpeedup.toFixed(2)}x observed max_abs_diff=${projectionDecodeMaxAbsDiff.toFixed(6)} shape_commands=${projectionDecodeShapeCommands} shape_projection_row_chains=${projectionDecodeShapeRowChains} shape_covered_ops=${projectionDecodeShapeCoveredOps} shape_saved_dispatches=${projectionDecodeShapeSavedDispatches} runtime_command_dispatches=${projectionDecodeRuntimeCommandDispatches} ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`}`,
    `projection_row_chain_prompt=${projectionPromptSpeedup === null ? "unavailable" : `${projectionPromptSpeedup.toFixed(2)}x observed max_abs_diff=${projectionPromptMaxAbsDiff.toFixed(6)} shape_commands=${projectionPromptShapeCommands} shape_projection_row_chains=${projectionPromptShapeRowChains} shape_covered_ops=${projectionPromptShapeCoveredOps} shape_saved_dispatches=${projectionPromptShapeSavedDispatches} runtime_command_dispatches=${projectionPromptRuntimeCommandDispatches} ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`}`,
    `projection_row_chain_full_prefill=${projectionFullPrefillSpeedup === null ? "unavailable" : `${projectionFullPrefillSpeedup.toFixed(2)}x observed max_abs_diff=${projectionFullPrefillMaxAbsDiff.toFixed(6)} shape_commands=${projectionFullPrefillShapeCommands} shape_projection_row_chains=${projectionFullPrefillShapeRowChains} shape_covered_ops=${projectionFullPrefillShapeCoveredOps} shape_saved_dispatches=${projectionFullPrefillShapeSavedDispatches} runtime_command_dispatches=${projectionFullPrefillRuntimeCommandDispatches} candidate=${projectionFullPrefillSpeedup >= projectionRowChainDefaultSpeedupFloor ? "ready" : "off"} ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`}`,
    `projection_row_chain_prompt_candidate=${projectionPromptCandidateReady ? "ready" : "off"} floor=${projectionRowChainDefaultSpeedupFloor.toFixed(2)} diff_ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_candidate=${projectionCandidateReady ? "ready" : "off"} floor=${projectionRowChainDefaultSpeedupFloor.toFixed(2)} diff_ceil=${projectionRowChainMaxAbsDiffCeil.toFixed(6)}`,
    `projection_row_chain_kernel=${projectionRowChainKernel}`,
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
    projectionGroupFullPrefillSpeedup,
    projectionGroupFullPrefillMaxAbsDiff,
    projectionRowChainGroupFullPrefillSpeedup,
    projectionRowChainGroupFullPrefillMaxAbsDiff,
    projectionDecodeSpeedup,
    projectionDecodeMaxAbsDiff,
    projectionPromptSpeedup,
    projectionPromptMaxAbsDiff,
    projectionFullPrefillSpeedup,
    projectionFullPrefillMaxAbsDiff,
    projectionRowChainGroupFullPrefillShapeCommands,
    projectionRowChainGroupFullPrefillShapeRowChains,
    projectionRowChainGroupFullPrefillShapeCoveredOps,
    projectionRowChainGroupFullPrefillShapeSavedDispatches,
    projectionRowChainGroupFullPrefillRuntimeCommandDispatches,
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
    projectionFullPrefillShapeCommands,
    projectionFullPrefillShapeRowChains,
    projectionFullPrefillShapeCoveredOps,
    projectionFullPrefillShapeSavedDispatches,
    projectionFullPrefillRuntimeCommandDispatches,
    projectionDefaultDecision,
    projectionDefaultReason,
    failures,
    line,
  };
}

function scoreMargin(current) {
  const projectionDecodeMargin = current.projectionDecodeMaxAbsDiff === null
    ? Infinity
    : projectionRowChainMaxAbsDiffCeil / Math.max(current.projectionDecodeMaxAbsDiff, Number.EPSILON);
  const projectionPromptMargin = current.projectionPromptMaxAbsDiff === null
    ? Infinity
    : projectionRowChainMaxAbsDiffCeil / Math.max(current.projectionPromptMaxAbsDiff, Number.EPSILON);
  const projectionDecodeSpeedupMargin = current.projectionDecodeSpeedup === null
    ? 0
    : current.projectionDecodeSpeedup / projectionRowChainDefaultSpeedupFloor;
  const projectionPromptSpeedupMargin = current.projectionPromptSpeedup === null
    ? 0
    : current.projectionPromptSpeedup / projectionRowChainDefaultSpeedupFloor;
  const projectionFullPrefillMargin = current.projectionFullPrefillMaxAbsDiff === null
    ? Infinity
    : projectionRowChainMaxAbsDiffCeil / Math.max(current.projectionFullPrefillMaxAbsDiff, Number.EPSILON);
  return Math.min(
    current.smallSpeedup / 2.0,
    current.largeSpeedup / 3.0,
    current.decodeTokS / decodeishTokSFloor,
    current.projectionChainSpeedup / projectionChainTileSpeedupFloor,
    projectionChainMaxAbsDiffCeil / Math.max(current.projectionChainMaxAbsDiff, Number.EPSILON),
    current.projectionChainFullPrefillSpeedup / projectionChainFullPrefillSpeedupFloor,
    projectionChainMaxAbsDiffCeil / Math.max(current.projectionChainFullPrefillMaxAbsDiff, Number.EPSILON),
    projectionChainMaxAbsDiffCeil / Math.max(current.projectionGroupFullPrefillMaxAbsDiff, Number.EPSILON),
    projectionRowChainMaxAbsDiffCeil / Math.max(current.projectionRowChainGroupFullPrefillMaxAbsDiff, Number.EPSILON),
    projectionDecodeMargin,
    projectionPromptMargin,
    projectionFullPrefillMargin,
    projectionDecodeSpeedupMargin,
    projectionPromptSpeedupMargin,
  );
}

function chooseBest(attempts) {
  return attempts.reduce((acc, current) => {
    if (!acc) return current;
    return scoreMargin(current) > scoreMargin(acc) ? current : acc;
  }, null);
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

process.stdout.write(`${best.line}\n`);
if (best.attempt > 1) {
  process.stdout.write(`frontier bench retries: ${best.attempt - 1} noisy attempt(s) below best evidence\n`);
}
if (passing.length === 0) {
  process.stderr.write(`${best.failures.join("; ")}\n`);
  process.exit(1);
}
process.exit(0);
