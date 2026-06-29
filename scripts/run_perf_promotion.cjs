"use strict";

const { spawnSync } = require("node:child_process");

const baselineArtifact = "benchmarks/baselines/smollm-m5pro-p128-g200-r3.json";

function envWithDefaults(defaults) {
  return { ...process.env, ...Object.fromEntries(Object.entries(defaults).filter(([key]) => process.env[key] === undefined)) };
}

function run(label, command, args, env = process.env) {
  console.log(`[promotion] ${label}: ${[command, ...args].join(" ")}`);
  const result = spawnSync(command, args, { stdio: "inherit", env });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    const signal = result.signal ? ` signal=${result.signal}` : "";
    throw new Error(`${label} failed with status ${result.status}${signal}`);
  }
}

function truthy(value) {
  return value === "1" || value === "true" || value === "yes";
}

function main() {
  const shouldBuild = process.env.BENCH_PROMOTION_BUILD !== "0";
  const steady = process.env.BENCH_PROMOTION_STEADY !== "0";
  const hardParity = truthy(process.env.BENCH_PROMOTION_REQUIRE_PARITY);
  const full = truthy(process.env.BENCH_PROMOTION_FULL) || hardParity;
  const attempts = steady ? "3" : "1";
  console.log(`[promotion] build=${shouldBuild ? "yes" : "no"} steady=${steady ? "yes" : "no"} ggml=${full ? "full" : "smoke"} parity=${hardParity ? "required" : "not-required"}`);

  if (shouldBuild) {
    run("build benchmark artifacts", "zig", ["build", "-Doptimize=ReleaseFast", "bench-build", "-fincremental", "--summary", "failures"]);
  }

  run(
    "qsemantic throughput frontier",
    process.execPath,
    ["scripts/check_frontier_bench.cjs"],
    envWithDefaults({
      BENCH_FRONTIER_BUILD: "0",
      BENCH_FRONTIER_ATTEMPTS: attempts,
      BENCH_QSEMANTIC_VARIANTS: "throughput_candidate",
      BENCH_FRONTIER_FILTER: "qsemantic",
    }),
  );

  run(
    "q8 prompt viable full-model",
    process.execPath,
    ["scripts/check_q8_prompt_candidate.cjs"],
    envWithDefaults({
      BENCH_BUILD_ZGML: "0",
      BENCH_CANDIDATE_ATTEMPTS: attempts,
      BENCH_Q8_PROMPT_LANES: "command,two_phase,semantic",
      BENCH_Q8_PROMPT_PAIR_DEFAULTS: steady ? "1" : "0",
    }),
  );

  run(
    full ? "ggml baseline-gated full run" : "ggml smoke",
    "./scripts/bench_vs_ggml.sh",
    full
      ? ["128", "200", "3"]
      : [
          process.env.BENCH_GGML_PROMPT ?? "128",
          process.env.BENCH_GGML_GEN ?? "40",
          process.env.BENCH_GGML_REPS ?? "1",
        ],
    envWithDefaults({
      BENCH_BUILD_ZGML: "0",
      BENCH_BASELINE_JSON: full || hardParity ? baselineArtifact : "",
      BENCH_REQUIRE_PARITY: hardParity ? "1" : "0",
      BENCH_ALLOW_QUARANTINED: full ? "0" : "1",
      BENCH_ZGML_SAMPLES: full ? "3" : "1",
    }),
  );
}

try {
  main();
} catch (error) {
  console.error(`[promotion] ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
}
