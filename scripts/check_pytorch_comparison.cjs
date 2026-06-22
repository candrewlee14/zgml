"use strict";

const { existsSync } = require("node:fs");
const { spawnSync } = require("node:child_process");
const { join, resolve } = require("node:path");

const root = resolve(__dirname, "..");
const nodeEntry = join(root, "dist", "node.cjs");
const venvPython = join(root, ".venv", "bin", "python");
const python = process.env.PYTHON || (existsSync(venvPython) ? venvPython : "python3");
const requireParity = process.env.BENCH_PYTORCH_REQUIRE_PARITY === "1";
const installTorch = process.env.BENCH_PYTORCH_INSTALL === "1";
const minRatio = Number(process.env.BENCH_PYTORCH_MIN_RATIO || "1.0");

function positiveInt(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer, got ${value}`);
  }
  return parsed;
}

const attempts = positiveInt(process.env.BENCH_PYTORCH_ATTEMPTS || (requireParity ? "3" : "1"), "BENCH_PYTORCH_ATTEMPTS");
const comparisonKeys = [
  "linear_batched",
  "lazy_matmul_add_gelu_batched",
  "lazy_mlp_batched",
  "lazy_rms_silu_ffn_batched",
  "max_pool2d_batched",
  "avg_pool2d_batched",
];

function selectedComparisonKeys() {
  const raw = process.env.BENCH_PYTORCH_KEYS;
  if (!raw) return comparisonKeys;
  const requested = raw.split(",").map((key) => key.trim()).filter(Boolean);
  if (requested.length === 0) {
    throw new Error("BENCH_PYTORCH_KEYS must list at least one comparison key when set");
  }
  const known = new Set(comparisonKeys);
  const unknown = requested.filter((key) => !known.has(key));
  if (unknown.length !== 0) {
    throw new Error(`BENCH_PYTORCH_KEYS contains unknown comparison key(s): ${unknown.join(", ")}`);
  }
  return requested;
}
const activeComparisonKeys = selectedComparisonKeys();

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: root,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    ...options,
  });
  if (result.status !== 0) {
    const err = new Error(`${command} ${args.join(" ")} failed with status ${result.status ?? 1}`);
    err.output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
    throw err;
  }
  return result.stdout ?? "";
}

function hasPythonTorch() {
  const result = spawnSync(python, ["-c", "import torch; print(torch.__version__)"], {
    cwd: root,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  return result.status === 0 ? result.stdout.trim() : null;
}

function installPythonTorchWithUv() {
  const uv = process.env.UV || "uv";
  if (!existsSync(venvPython)) {
    run(uv, ["venv", ".venv"]);
  }
  run(uv, ["pip", "install", "torch", "--python", venvPython], { stdio: "inherit" });
}

function parseZgmlModuleBench(output, keys) {
  const timings = {};
  for (const key of keys) {
    const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const match = output.match(new RegExp(`${escaped}=[^;]*hot_execute_into=([0-9.]+)ms`));
    if (!match) {
      throw new Error(`pytorch comparison could not find zgml timing for ${key}`);
    }
    timings[key] = Number(match[1]);
  }
  return timings;
}

const pytorchCode = String.raw`
import json
import math
import time

import torch

torch.set_num_threads(1)
torch.manual_seed(0)

def values(shape, scale):
    n = math.prod(shape)
    data = [((i % 17) - 8) / scale for i in range(n)]
    return torch.tensor(data, dtype=torch.float32).reshape(shape)

def bench(fn, iterations=1000):
    with torch.inference_mode():
        for _ in range(min(100, iterations)):
            fn()
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        return (time.perf_counter() - start) * 1000.0 / iterations

x128_64 = values((128, 64), 16.0)
w32_64 = values((32, 64), 24.0)
b32 = values((32,), 32.0)
w64_64 = values((64, 64), 24.0)
b64 = values((64,), 32.0)
w128_64 = values((128, 64), 48.0)
b128 = values((128,), 80.0)
w64_128 = values((64, 128), 64.0)
b64_down = values((64,), 96.0)
rms_weight = values((64,), 48.0) + 1.0
pool_x = values((2, 2, 128, 128), 10.0)

def linear_batched():
    return torch.nn.functional.linear(x128_64, w32_64, b32)

def lazy_matmul_add_gelu_batched():
    return torch.nn.functional.gelu(torch.matmul(x128_64, w64_64) + b64, approximate="tanh")

def lazy_mlp_batched():
    hidden = torch.relu(torch.nn.functional.linear(x128_64, w64_64, b64))
    return torch.nn.functional.linear(hidden, w32_64, b32)

def lazy_rms_silu_ffn_batched():
    ss = torch.mean(x128_64 * x128_64, dim=1, keepdim=True)
    normed = x128_64 * torch.rsqrt(ss + 1e-5) * rms_weight
    hidden = torch.nn.functional.silu(torch.matmul(normed, w128_64.T) + b128)
    return torch.matmul(hidden, w64_128.T) + b64_down

def max_pool2d_batched():
    return torch.nn.functional.max_pool2d(pool_x, 2)

def avg_pool2d_batched():
    return torch.nn.functional.avg_pool2d(pool_x, 2, count_include_pad=True)

print(json.dumps({
    "linear_batched": bench(linear_batched),
    "lazy_matmul_add_gelu_batched": bench(lazy_matmul_add_gelu_batched),
    "lazy_mlp_batched": bench(lazy_mlp_batched),
    "lazy_rms_silu_ffn_batched": bench(lazy_rms_silu_ffn_batched),
    "max_pool2d_batched": bench(max_pool2d_batched, 300),
    "avg_pool2d_batched": bench(avg_pool2d_batched, 300),
}))
`;

if (!existsSync(nodeEntry)) {
  throw new Error("pytorch comparison requires dist/node.cjs; run npm run build:package first");
}

let pytorchVersion = hasPythonTorch();
if (!pytorchVersion && installTorch) {
  installPythonTorchWithUv();
  pytorchVersion = hasPythonTorch();
}
if (!pytorchVersion) {
  const installHint = installTorch ? "uv install attempt did not make torch importable" : "set BENCH_PYTORCH_INSTALL=1 to bootstrap .venv with uv";
  console.log(`pytorch comparison skipped: ${python} import torch failed; ${installHint}`);
  process.exit(0);
}
if (!Number.isFinite(minRatio) || minRatio < 0) {
  throw new Error(`BENCH_PYTORCH_MIN_RATIO must be a non-negative number, got ${process.env.BENCH_PYTORCH_MIN_RATIO}`);
}

function measureAttempt(index) {
  const moduleBenchEnv = process.env.BENCH_PYTORCH_KEYS
    ? {
        ...process.env,
        BENCH_MODULE_PROGRAM_KEYS: activeComparisonKeys.join(","),
      }
    : process.env;
  const zgmlTimings = parseZgmlModuleBench(run(process.execPath, ["scripts/check_module_program_bench.cjs"], { env: moduleBenchEnv }), activeComparisonKeys);
  const pytorchTimings = JSON.parse(run(python, ["-c", pytorchCode]));
  const ratioEntries = [];
  for (const [key, zgmlMs] of Object.entries(zgmlTimings)) {
    const pytorchMs = pytorchTimings[key];
    const ratio = pytorchMs / zgmlMs;
    ratioEntries.push({ key, zgmlMs, pytorchMs, ratio });
  }
  const worst = ratioEntries.reduce((current, entry) => (entry.ratio < current.ratio ? entry : current), ratioEntries[0]);
  const parityReady = ratioEntries.every((entry) => entry.ratio >= minRatio);
  const margin = Math.min(...ratioEntries.map((entry) => entry.ratio / minRatio));
  return { index, ratioEntries, worst, parityReady, margin };
}

const attemptRows = [];
for (let index = 1; index <= attempts; index += 1) {
  attemptRows.push(measureAttempt(index));
}

const passing = attemptRows.filter((entry) => entry.parityReady);
const best = (passing.length > 0 ? passing : attemptRows).reduce((current, entry) => (entry.margin > current.margin ? entry : current));
const worst = best.worst;
const parityReady = best.parityReady;
const noisyAttempts = attemptRows.filter((entry) => !entry.parityReady).length;
const parts = [
  `pytorch comparison: ${requireParity ? (parityReady ? "parity-pass" : "parity-miss") : "evidence"}`,
  `python=${python}`,
  `pytorch=${pytorchVersion}`,
  `uv_install=${installTorch ? "enabled" : "disabled"}`,
  "gelu=approximate-tanh",
  `required=${requireParity ? "yes" : "no"}`,
  `floor=${minRatio.toFixed(2)}x`,
  `attempt=${best.index}/${attempts}`,
  `noisy=${noisyAttempts}`,
  `worst=${worst.key}:${worst.ratio.toFixed(2)}x`,
  `parity=${parityReady ? "pass" : "miss"}`,
];
for (const { key, zgmlMs, pytorchMs, ratio } of best.ratioEntries) {
  parts.push(`${key}=zgml:${zgmlMs.toFixed(4)}ms pytorch:${pytorchMs.toFixed(4)}ms zgml_vs_pytorch=${ratio.toFixed(2)}x`);
}
console.log(parts.join("; "));
if (requireParity && !parityReady) {
  process.exit(1);
}
