#!/usr/bin/env node
"use strict";

const { spawnSync } = require("node:child_process");

const prompt = process.env.BENCH_GGML_PROMPT || "128";
const gen = process.env.BENCH_GGML_GEN || "40";
const reps = process.env.BENCH_GGML_REPS || "1";
const parityFloor = Number(process.env.BENCH_GGML_Q8_COMMAND_PROMPT_FLOOR || "27");

const env = {
  ...process.env,
  BENCH_BUILD_ZGML: process.env.BENCH_BUILD_ZGML ?? "0",
  BENCH_ALLOW_QUARANTINED: process.env.BENCH_ALLOW_QUARANTINED ?? "1",
  BENCH_ZGML_SAMPLES: process.env.BENCH_ZGML_SAMPLES ?? "1",
  ZGML_Q8_EXTRA_ARGS: "--metal-prompt-projection-row-chain-command",
};

const result = spawnSync("./scripts/bench_vs_ggml.sh", [prompt, gen, reps], {
  cwd: process.cwd(),
  encoding: "utf8",
  env,
  stdio: ["ignore", "pipe", "pipe"],
});

const output = `${result.stdout || ""}${result.stderr || ""}`;
if (result.status !== 0) {
  process.stderr.write(output);
  process.exit(result.status ?? 1);
}

function requireText(needle, label) {
  if (!output.includes(needle)) {
    process.stderr.write(output);
    process.stderr.write(`\nmissing ${label}: ${needle}\n`);
    process.exit(1);
  }
}

requireText("Metal Q8_0 prompt | metal scheduled prefill projection-row-chain command", "Q8 command parity lane");
requireText("| q8_0 | prompt | metal scheduled prefill projection-row-chain command |", "Q8 command native evidence lane");
requireText("242/242 dispatch, 151/151 command", "Q8 command shape");
requireText("90/90 dispatched semantic_ffn_sublayer", "Q8 semantic FFN dispatch proof");
requireText("60/60 dispatched projection_row_chain", "Q8 projection row-chain dispatch proof");
requireText("30/30 dispatched projection_cache_group", "Q8 projection cache-group proof");
requireText("Fallback", "fallback column");
requireText("| pass |", "native evidence pass");

const q8PromptMatch = output.match(/\| Metal Q8_0 prompt \| metal scheduled prefill projection-row-chain command [^|]+ \| [^|]+ \| [^|]+ \| [^|]+ \| ([0-9.]+)% \| 0\.000% \| (?:miss|pass) \|/);
if (!q8PromptMatch) {
  process.stderr.write(output);
  process.stderr.write("\nmissing Q8 command prompt parity row with zero fallback\n");
  process.exit(1);
}

const q8PromptPct = Number(q8PromptMatch[1]);
if (!Number.isFinite(q8PromptPct) || q8PromptPct < parityFloor) {
  process.stderr.write(output);
  process.stderr.write(`\nQ8 command prompt parity ${q8PromptPct.toFixed(3)}% below ${parityFloor.toFixed(3)}%\n`);
  process.exit(1);
}

const artifact = output.match(/Wrote accepted artifact:\s*\n\s*(\S+)/)?.[1] ?? "n/a";
process.stdout.write(
  `ggml q8 command smoke: pass q8_prompt=${q8PromptPct.toFixed(3)}% floor=${parityFloor.toFixed(3)} commands=151 semantic_ffn=90 projection_row_chain=60 projection_cache_group=30 artifact=${artifact}\n`,
);
