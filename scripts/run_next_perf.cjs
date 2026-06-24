"use strict";

const { spawnSync } = require("node:child_process");

function runCaptured(command, args, env = process.env) {
  const result = spawnSync(command, args, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    env,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    const err = new Error(`${command} ${args.join(" ")} failed with status ${result.status ?? 1}`);
    err.output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
    throw err;
  }
  return result.stdout ?? "";
}

function runInherited(label, command, args, env = process.env) {
  console.log(`[next-perf] ${label}: ${[command, ...args].join(" ")}`);
  const result = spawnSync(command, args, { stdio: "inherit", env });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    const signal = result.signal ? ` signal=${result.signal}` : "";
    throw new Error(`${label} failed with status ${result.status}${signal}`);
  }
}

function envWithDefaults(defaults) {
  return { ...process.env, ...Object.fromEntries(Object.entries(defaults).filter(([key]) => process.env[key] === undefined)) };
}

function perfNextLine() {
  const output = runCaptured(process.execPath, ["scripts/bench_status.cjs"]);
  const line = output.split(/\r?\n/).find((row) => row.startsWith("perf-next:"));
  if (!line) throw new Error("bench_status did not print a perf-next line");
  return line;
}

function chooseLane(line) {
  const forced = String(process.env.BENCH_NEXT_PERF_LANE ?? "").trim();
  if (forced) return forced;
  const steady = process.env.BENCH_NEXT_PERF_STEADY === "1";
  if (
    !steady &&
    /frontier=semantic_ffn_sublayer_throughput_kernel:candidate=ready/.test(line)
  ) return "qsemantic_throughput";
  if (
    steady &&
    /frontier=semantic_ffn_sublayer_throughput_kernel:candidate=ready/.test(line) &&
    /q8_prompt=semantic_throughput_kernel/.test(line)
  ) return "q8_prompt";
  if (/frontier=semantic_ffn_sublayer_throughput_kernel/.test(line)) return "qsemantic";
  if (/q8_prompt=semantic_throughput_kernel/.test(line)) return "q8_prompt";
  if (/pytorch=(?!none\b)[^ ]+/.test(line)) return "pytorch";
  if (/full_model=/.test(line)) return "qsemantic";
  return "status";
}

function validateLane(lane) {
  const known = new Set(["status", "pytorch", "qsemantic", "qsemantic_throughput", "qproj", "q8_prompt", "ggml"]);
  if (!known.has(lane)) throw new Error(`unknown BENCH_NEXT_PERF_LANE: ${lane}`);
}

function main() {
  const line = perfNextLine();
  const lane = chooseLane(line);
  validateLane(lane);
  const shouldBuild = process.env.BENCH_NEXT_PERF_BUILD === "1";
  const steady = process.env.BENCH_NEXT_PERF_STEADY === "1";
  console.log(`[next-perf] ${line}`);
  console.log(`[next-perf] lane=${lane} build=${shouldBuild ? "yes" : "no"} steady=${steady ? "yes" : "no"}`);

  if (lane === "status") return;

  if (shouldBuild && (lane === "qsemantic" || lane === "qsemantic_throughput" || lane === "qproj" || lane === "q8_prompt" || lane === "ggml")) {
    runInherited("build benchmark artifacts", "zig", ["build", "-Doptimize=ReleaseFast", "bench-build", "-fincremental", "--summary", "failures"]);
  }
  if (shouldBuild && lane === "pytorch") {
    runInherited("build native ffi", "zig", ["build", "ffi-c", "-Doptimize=ReleaseFast", "-fincremental", "--summary", "failures"]);
  }

  if (lane === "qsemantic") {
    runInherited(
      "qsemantic frontier",
      process.execPath,
      ["scripts/check_frontier_bench.cjs"],
      envWithDefaults({
        BENCH_FRONTIER_BUILD: "0",
        BENCH_FRONTIER_ATTEMPTS: steady ? "3" : "1",
        BENCH_FRONTIER_FILTER: "qsemantic",
      }),
    );
    return;
  }

  if (lane === "qsemantic_throughput") {
    runInherited(
      "qsemantic throughput frontier",
      process.execPath,
      ["scripts/check_frontier_bench.cjs"],
      envWithDefaults({
        BENCH_FRONTIER_BUILD: "0",
        BENCH_FRONTIER_ATTEMPTS: steady ? "3" : "1",
        BENCH_QSEMANTIC_VARIANTS: "throughput_candidate",
        BENCH_FRONTIER_FILTER: "qsemantic",
      }),
    );
    return;
  }

  if (lane === "qproj") {
    runInherited(
      "qproj frontier",
      process.execPath,
      ["scripts/check_frontier_bench.cjs"],
      envWithDefaults({
        BENCH_FRONTIER_BUILD: "0",
        BENCH_FRONTIER_ATTEMPTS: steady ? "3" : "1",
        BENCH_FRONTIER_FILTER: "qproj",
      }),
    );
    return;
  }

  if (lane === "q8_prompt") {
    runInherited(
      "q8 prompt viable full-model",
      process.execPath,
      ["scripts/check_q8_prompt_candidate.cjs"],
      envWithDefaults({
        BENCH_BUILD_ZGML: "0",
        BENCH_CANDIDATE_ATTEMPTS: steady ? "3" : "1",
        BENCH_Q8_PROMPT_LANES: "command,two_phase,semantic",
        BENCH_Q8_PROMPT_PAIR_DEFAULTS: steady ? "1" : "0",
      }),
    );
    return;
  }

  if (lane === "pytorch") {
    runInherited(
      "pytorch current gap",
      process.execPath,
      ["scripts/check_pytorch_comparison.cjs"],
      envWithDefaults({
        BENCH_PYTORCH_INSTALL: "1",
        BENCH_PYTORCH_ATTEMPTS: steady ? "3" : "1",
        BENCH_PYTORCH_MIN_TIMING_MS: steady ? "150" : "8",
        BENCH_MODULE_PROGRAM_MIN_TIMING_MS: steady ? "150" : "8",
        BENCH_PYTORCH_KEYS: "linear_batched,log_softmax_classifier_batched",
      }),
    );
    return;
  }

  if (lane === "ggml") {
    runInherited(
      "ggml smoke",
      "./scripts/bench_vs_ggml.sh",
      [
        process.env.BENCH_GGML_PROMPT ?? "128",
        process.env.BENCH_GGML_GEN ?? "40",
        process.env.BENCH_GGML_REPS ?? "1",
      ],
      envWithDefaults({
        BENCH_BUILD_ZGML: "0",
        BENCH_ALLOW_QUARANTINED: "1",
        BENCH_ZGML_SAMPLES: "1",
      }),
    );
  }
}

try {
  main();
} catch (error) {
  if (error && typeof error === "object" && "output" in error) process.stderr.write(String(error.output));
  console.error(`[next-perf] ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
}
