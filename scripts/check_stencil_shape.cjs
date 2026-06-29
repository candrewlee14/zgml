"use strict";

const { spawnSync } = require("node:child_process");

const build = process.env.BENCH_STENCIL_BUILD ?? "1";
const binary = "./zig-out/bin/bench-llama-smollm";

function run(command, args) {
  const result = spawnSync(command, args, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
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
  const prefix = "ZGML_STENCIL_JSON ";
  return output
    .split(/\r?\n/)
    .filter((line) => line.startsWith(prefix))
    .map((line) => {
      const raw = line.slice(prefix.length);
      const row = JSON.parse(raw);
      row.__raw = raw;
      return row;
    });
}

function rowFor(rows, phase) {
  const row = rows.find((candidate) => candidate.phase === phase);
  if (!row) throw new Error(`stencil shape gate missing ${phase} row`);
  return row;
}

function requireEqual(row, key, expected, errors) {
  const actual = row[key];
  if (actual !== expected) errors.push(`${row.phase} ${key}=${actual} expected ${expected}`);
}

function requirePositive(row, key, errors) {
  const actual = Number(row[key]);
  if (!Number.isFinite(actual) || actual <= 0) errors.push(`${row.phase} ${key}=${row[key]} expected positive`);
}

function rawNumber(row, key) {
  const match = String(row.__raw ?? "").match(new RegExp(`"${key}"\\s*:\\s*([0-9]+)`));
  return match ? match[1] : String(row[key] ?? "missing");
}

if (build === "1") {
  run("zig", ["build", "-Doptimize=ReleaseFast", "bench-build"]);
} else if (build !== "0") {
  console.error(`BENCH_STENCIL_BUILD must be 0 or 1, got ${build}`);
  process.exit(1);
}

let output;
try {
  output = run(binary, ["ignored", "128", "1", "1", "--stencil-only", "--debug-row-chain"]);
} catch (err) {
  process.stderr.write(`${err.output ?? err.message}\n`);
  process.exit(1);
}

const rows = parseRows(output);
const errors = [];
const prompt = rowFor(rows, "prompt");
const decode = rowFor(rows, "decode");

for (const row of [prompt, decode]) {
  requireEqual(row, "semantic_stage_count", 212, errors);
  requireEqual(row, "semantic_layers", 30, errors);
  requireEqual(row, "semantic_heads", 9, errors);
  requireEqual(row, "semantic_kv_heads", 3, errors);
  requireEqual(row, "semantic_runtime_patch_holes", 450, errors);
  requireEqual(row, "semantic_runtime_patch_cache_write_pos_holes", 180, errors);
  requireEqual(row, "semantic_runtime_patch_attention_seq_kv_holes", 270, errors);
  requireEqual(row, "runtime_patch_holes", 450, errors);
  requireEqual(row, "runtime_patch_cache_write_pos_holes", 180, errors);
  requireEqual(row, "runtime_patch_attention_seq_kv_holes", 270, errors);
  requireEqual(row, "program_command_shape_row_chains", 61, errors);
  requireEqual(row, "program_command_shape_projection_chains", 60, errors);
  requireEqual(row, "program_command_shape_dense_projection_chains", 60, errors);
  requireEqual(row, "program_command_shape_quantized_projection_chains", 0, errors);
  requireEqual(row, "program_command_shape_projection_chain_sidecars", 60, errors);
  requireEqual(row, "program_command_shape_projection_chain_row_chain_frontiers", 60, errors);
  requireEqual(row, "program_command_shape_projection_row_chain_semantic_residual_bridges", 0, errors);
  requireEqual(row, "program_command_shape_projection_cache_groups", 30, errors);
  requirePositive(row, "runtime_patch_stencil_hash", errors);
  requirePositive(row, "program_command_shape_stencil_hash", errors);
}

requireEqual(prompt, "semantic_token_count", 128, errors);
requireEqual(prompt, "program_command_shape_commands", 242, errors);
requireEqual(prompt, "program_command_shape_covered_ops", 1594, errors);
requireEqual(prompt, "program_command_shape_estimated_saved_dispatches", 1352, errors);
requireEqual(prompt, "program_command_shape_projection_cache_anchors", 90, errors);
requireEqual(prompt, "program_command_shape_projection_cache_sidecars", 90, errors);

requireEqual(decode, "semantic_token_count", 1, errors);
requireEqual(decode, "program_command_shape_commands", 212, errors);
requireEqual(decode, "program_command_shape_covered_ops", 1594, errors);
requireEqual(decode, "program_command_shape_estimated_saved_dispatches", 1382, errors);
requireEqual(decode, "program_command_shape_projection_cache_anchors", 90, errors);
requireEqual(decode, "program_command_shape_projection_cache_sidecars", 270, errors);

const line = [
  "stencil shape gate: pass",
  `prompt_commands=${prompt.program_command_shape_commands}`,
  `decode_commands=${decode.program_command_shape_commands}`,
  `covered_ops=${prompt.program_command_shape_covered_ops}/${decode.program_command_shape_covered_ops}`,
  `prompt_hash=${rawNumber(prompt, "runtime_patch_stencil_hash")}/${rawNumber(prompt, "program_command_shape_stencil_hash")}`,
  `decode_hash=${rawNumber(decode, "runtime_patch_stencil_hash")}/${rawNumber(decode, "program_command_shape_stencil_hash")}`,
  `projection_chains=${prompt.program_command_shape_projection_chains}/${decode.program_command_shape_projection_chains}`,
  `semantic_residual_bridges=${prompt.program_command_shape_projection_row_chain_semantic_residual_bridges}/${decode.program_command_shape_projection_row_chain_semantic_residual_bridges}`,
  `cache_groups=${prompt.program_command_shape_projection_cache_groups}/${decode.program_command_shape_projection_cache_groups}`,
].join("; ");

if (errors.length !== 0) {
  process.stderr.write(`${line.replace("pass", "fail")}\n${errors.join("\n")}\n`);
  process.exit(1);
}

process.stdout.write(`${line}\n`);
