"use strict";

const { spawnSync } = require("node:child_process");

const knownLanes = new Set(["pytorch", "native_eager", "qsemantic", "q8_prompt", "ggml"]);
const broadPytorchKeys = "linear_batched,lazy_matmul_add_gelu_batched,lazy_mlp_batched,lazy_rms_silu_ffn_batched,lazy_conv2d_relu_batched,max_pool2d_batched,avg_pool2d_batched,rms_gelu_linear_batched,softmax_classifier_batched,log_softmax_classifier_batched,lazy_token_head_batched";

function parseLanes(value) {
  const raw = String(value ?? "pytorch,native_eager,qsemantic,q8_prompt,ggml").trim();
  if (raw.length === 0) return ["pytorch", "native_eager", "qsemantic", "q8_prompt", "ggml"];
  const lanes = raw.split(",").map((part) => part.trim()).filter(Boolean);
  for (const lane of lanes) {
    if (!knownLanes.has(lane)) {
      throw new Error(`unknown BENCH_COMPETITIVE_LANES entry: ${lane}`);
    }
  }
  return [...new Set(lanes)];
}

function envWithDefaults(defaults) {
  return { ...process.env, ...Object.fromEntries(Object.entries(defaults).filter(([key]) => process.env[key] === undefined)) };
}

function run(label, command, args, env = process.env) {
  console.log(`[competitive] ${label}: ${[command, ...args].join(" ")}`);
  const result = spawnSync(command, args, { stdio: "inherit", env });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    const signal = result.signal ? ` signal=${result.signal}` : "";
    throw new Error(`${label} failed with status ${result.status}${signal}`);
  }
}

function main() {
  const lanes = parseLanes(process.env.BENCH_COMPETITIVE_LANES);
  const shouldBuild = process.env.BENCH_COMPETITIVE_BUILD !== "0";
  const needsNative = lanes.includes("pytorch") || lanes.includes("native_eager");
  const needsBench = lanes.includes("qsemantic") || lanes.includes("q8_prompt") || lanes.includes("ggml");

  console.log(`[competitive] lanes=${lanes.join(",")} build=${shouldBuild ? "yes" : "no"}`);

  if (shouldBuild && needsNative) {
    run("build native ffi", "zig", ["build", "ffi-c", "-Doptimize=ReleaseFast", "-fincremental", "--summary", "failures"]);
    run("build package", "npm", ["run", "build:package"]);
  }
  if (shouldBuild && needsBench) {
    run("build benchmark artifacts", "zig", ["build", "-Doptimize=ReleaseFast", "bench-build", "-fincremental", "--summary", "failures"]);
  }

  if (lanes.includes("pytorch")) {
    run(
      "pytorch steady broad replacement",
      process.execPath,
      ["scripts/check_pytorch_comparison.cjs"],
      envWithDefaults({
        BENCH_PYTORCH_INSTALL: "1",
        BENCH_PYTORCH_ATTEMPTS: "3",
        BENCH_PYTORCH_MIN_TIMING_MS: "150",
        BENCH_MODULE_PROGRAM_MIN_TIMING_MS: "150",
        BENCH_PYTORCH_KEYS: broadPytorchKeys,
      }),
    );
  }

  if (lanes.includes("native_eager")) {
    run(
      "native eager replacement gap",
      process.execPath,
      ["scripts/check_native_eager_gap.cjs"],
      envWithDefaults({
        BENCH_NATIVE_EAGER_RUNTIME: "node",
      }),
    );
  }

  if (lanes.includes("qsemantic")) {
    run(
      "qsemantic frontier",
      process.execPath,
      ["scripts/check_frontier_bench.cjs"],
      envWithDefaults({
        BENCH_FRONTIER_BUILD: "0",
        BENCH_FRONTIER_ATTEMPTS: "1",
        BENCH_FRONTIER_FILTER: "qsemantic",
      }),
    );
  }

  if (lanes.includes("q8_prompt")) {
    run(
      "q8 prompt viable full-model",
      process.execPath,
      ["scripts/check_q8_prompt_candidate.cjs"],
      envWithDefaults({
        BENCH_BUILD_ZGML: "0",
        BENCH_CANDIDATE_ATTEMPTS: "1",
        BENCH_Q8_PROMPT_LANES: "command,two_phase,semantic",
      }),
    );
  }

  if (lanes.includes("ggml")) {
    run(
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
  console.error(`[competitive] ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
}
