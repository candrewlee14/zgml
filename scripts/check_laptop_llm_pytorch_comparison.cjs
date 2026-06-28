"use strict";

const { existsSync, mkdirSync, writeFileSync } = require("node:fs");
const { spawnSync } = require("node:child_process");
const { join, resolve } = require("node:path");
const os = require("node:os");

const root = resolve(__dirname, "..");
const nodeEntry = join(root, "dist", "node.cjs");
const venvPython = join(root, ".venv", "bin", "python");
const python = process.env.PYTHON || (existsSync(venvPython) ? venvPython : "python3");
const artifactDir = process.env.BENCH_LAPTOP_MODEL_ARTIFACT_DIR || join(root, "bench-results", "laptop-models");
const writeArtifact = process.env.BENCH_LAPTOP_MODEL_WRITE_ARTIFACT !== "0";
const modelId = process.env.BENCH_LAPTOP_MODEL_ID || "HuggingFaceTB/SmolLM2-360M-Instruct";
const suiteId = process.env.BENCH_LAPTOP_SUITE_ID || "llm.smollm2_360m.instruct.q8_0";
const prompt = process.env.BENCH_LAPTOP_PROMPT || "Explain why small fast local AI runtimes matter in two concise sentences.";
const decodeTokens = positiveInt(process.env.BENCH_LAPTOP_DECODE_TOKENS || "16", "BENCH_LAPTOP_DECODE_TOKENS");
const prefillIters = positiveInt(process.env.BENCH_LAPTOP_PREFILL_ITERS || "5", "BENCH_LAPTOP_PREFILL_ITERS");
const torchThreads = positiveInt(process.env.BENCH_LAPTOP_PYTORCH_THREADS || "1", "BENCH_LAPTOP_PYTORCH_THREADS");
const zgmlBackends = (process.env.BENCH_LAPTOP_ZGML_BACKENDS || (process.platform === "darwin" ? "cpu,metal" : "cpu"))
  .split(",")
  .map((backend) => backend.trim())
  .filter(Boolean);

function positiveInt(value, name) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive safe integer, got ${value}`);
  }
  return parsed;
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: root,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    ...options,
  });
  if (result.status !== 0) {
    const error = new Error(`${command} ${args.join(" ")} failed with status ${result.status ?? 1}`);
    error.output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
    throw error;
  }
  return result.stdout ?? "";
}

function timestampForArtifact(date = new Date()) {
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function round(value) {
  return Number.isFinite(value) ? Number(value.toFixed(6)) : null;
}

function artifactJson(value) {
  return JSON.stringify(value, (_key, inner) => typeof inner === "bigint" ? `${inner}` : inner, 2);
}

const pytorchCode = String.raw`
import json
import os
import platform
import time

import psutil
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = os.environ["BENCH_LAPTOP_MODEL_ID"]
prompt = os.environ["BENCH_LAPTOP_PROMPT"]
decode_tokens = int(os.environ["BENCH_LAPTOP_DECODE_TOKENS"])
prefill_iters = int(os.environ["BENCH_LAPTOP_PREFILL_ITERS"])
torch_threads = int(os.environ["BENCH_LAPTOP_PYTORCH_THREADS"])

torch.set_num_threads(torch_threads)
torch.set_num_interop_threads(1)
torch.manual_seed(0)

process = psutil.Process()
rss_before = process.memory_info().rss
download_start = time.perf_counter()
snapshot_dir = snapshot_download(model_id, allow_patterns=[
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
])
download_ms = (time.perf_counter() - download_start) * 1000.0
model_path = os.path.join(snapshot_dir, "model.safetensors")

load_start = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
model = AutoModelForCausalLM.from_pretrained(
    snapshot_dir,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()
load_ms = (time.perf_counter() - load_start) * 1000.0
rss_after_load = process.memory_info().rss

inputs = tokenizer(prompt, return_tensors="pt")
prompt_tokens = int(inputs["input_ids"].shape[1])
prompt_token_ids = [int(x) for x in inputs["input_ids"][0].tolist()]

with torch.inference_mode():
    warm = model(**inputs, use_cache=True)

prefill_start = time.perf_counter()
with torch.inference_mode():
    for _ in range(prefill_iters):
        prefill = model(**inputs, use_cache=True)
prefill_ms = (time.perf_counter() - prefill_start) * 1000.0 / prefill_iters

with torch.inference_mode():
    out = model(**inputs, use_cache=True)
    past = out.past_key_values
    next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    decode_start = time.perf_counter()
    generated = []
    for _ in range(decode_tokens):
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(int(next_id.item()))
    decode_ms = (time.perf_counter() - decode_start) * 1000.0 / decode_tokens

rss_after_run = process.memory_info().rss
print(json.dumps({
    "modelId": model_id,
    "snapshotDir": snapshot_dir,
    "modelPath": model_path,
    "torchVersion": torch.__version__,
    "transformersVersion": __import__("transformers").__version__,
    "threads": torch.get_num_threads(),
    "platform": platform.platform(),
    "downloadMs": download_ms,
    "loadMs": load_ms,
    "promptTokens": prompt_tokens,
    "promptTokenIds": prompt_token_ids,
    "prefillIters": prefill_iters,
    "prefillMs": prefill_ms,
    "prefillTokS": prompt_tokens / (prefill_ms / 1000.0),
    "decodeTokens": decode_tokens,
    "decodeMsPerToken": decode_ms,
    "decodeTokS": 1000.0 / decode_ms,
    "logitsShape": list(out.logits.shape),
    "generatedTokenIds": generated,
    "rssBeforeBytes": rss_before,
    "rssAfterLoadBytes": rss_after_load,
    "rssAfterRunBytes": rss_after_run,
    "rssLoadDeltaBytes": rss_after_load - rss_before,
    "rssRunDeltaBytes": rss_after_run - rss_before,
}))
`;

function runPytorch() {
  const output = run(python, ["-c", pytorchCode], {
    env: {
      ...process.env,
      BENCH_LAPTOP_MODEL_ID: modelId,
      BENCH_LAPTOP_PROMPT: prompt,
      BENCH_LAPTOP_DECODE_TOKENS: String(decodeTokens),
      BENCH_LAPTOP_PREFILL_ITERS: String(prefillIters),
      BENCH_LAPTOP_PYTORCH_THREADS: String(torchThreads),
    },
  });
  const lines = output.trim().split(/\r?\n/);
  const jsonLine = lines.reverse().find((line) => line.trim().startsWith("{"));
  if (!jsonLine) throw new Error(`PyTorch benchmark did not print JSON: ${output}`);
  return JSON.parse(jsonLine);
}

function runZgmlBackend(zgml, model, probe, probeMs, loadMs, backend, promptTokenIds) {
  let program = null;
  let session = null;
  try {
    const contextLength = Math.max(32, promptTokenIds.length + decodeTokens + 1);
    const compileStarted = performance.now();
    program = model.compile({ backend, contextLength });
    const compileMs = performance.now() - compileStarted;
    const executable = typeof program.inspectExecutable === "function" ? program.inspectExecutable() : null;
    const bindStarted = performance.now();
    session = program.bind({ output: true });
    const bindMs = performance.now() - bindStarted;
    const inspection = typeof session.inspect === "function" ? session.inspect() : null;
    const warm = session.execute_tokens_argmax(promptTokenIds, { tokensLen: promptTokenIds.length });
    session.reset();

    const prefillStarted = performance.now();
    let prefill = warm;
    for (let i = 0; i < prefillIters; i += 1) {
      prefill = session.execute_tokens_argmax(promptTokenIds, { tokensLen: promptTokenIds.length });
      session.reset();
    }
    const prefillMs = (performance.now() - prefillStarted) / prefillIters;

    let next = prefill.token;
    const generated = [];
    session.execute_tokens_argmax(promptTokenIds, { tokensLen: promptTokenIds.length });
    const decodeStarted = performance.now();
    for (let i = 0; i < decodeTokens; i += 1) {
      const result = session.execute_tokens_argmax([next], { tokensLen: 1 });
      next = result.token;
      generated.push(next);
    }
    const decodeMsPerToken = (performance.now() - decodeStarted) / decodeTokens;

    return {
      supported: true,
      probeReady: true,
      executableReady: true,
      stage: "execute",
      backend,
      probeMs,
      loadMs,
      compileMs,
      bindMs,
      contextLength,
      prefillMs,
      prefillTokS: promptTokenIds.length / (prefillMs / 1000.0),
      decodeMsPerToken,
      decodeTokS: 1000.0 / decodeMsPerToken,
      firstToken: prefill.token,
      generatedTokenIds: generated,
      executable,
      inspection,
      probe,
    };
  } catch (error) {
    return {
      supported: false,
      probeReady: true,
      executableReady: false,
      stage: program == null ? "compile" : session == null ? "bind" : "execute",
      backend,
      probeMs,
      loadMs,
      probe,
      error: error && error.message ? error.message : String(error),
    };
  } finally {
    if (session && typeof session.dispose === "function") session.dispose();
    if (program && typeof program.dispose === "function") program.dispose();
  }
}

function choosePrimaryZgmlResult(results) {
  const ready = results.filter((result) => result.executableReady);
  if (ready.length === 0) return results[0] || { supported: false, probeReady: false, executableReady: false, stage: "none", error: "no zgml backends requested" };
  return ready.reduce((best, result) => {
    if (!Number.isFinite(best.decodeTokS)) return result;
    if (!Number.isFinite(result.decodeTokS)) return best;
    return result.decodeTokS > best.decodeTokS ? result : best;
  }, ready[0]);
}

function runZgmlProbe(modelPath, promptTokenIds) {
  if (!existsSync(nodeEntry)) {
    return { supported: false, stage: "load-runtime", error: "dist/node.cjs missing; run npm run build:package" };
  }
  let model = null;
  try {
    const zgml = require(nodeEntry);
    const started = performance.now();
    const probe = zgml.probeModel(modelPath, { modelKind: "auto" });
    const probeMs = performance.now() - started;
    const loadStarted = performance.now();
    try {
      model = zgml.loadModel(modelPath, { modelKind: "auto" });
      const loadMs = performance.now() - loadStarted;
      const backends = zgmlBackends.map((backend) => runZgmlBackend(zgml, model, probe, probeMs, loadMs, backend, promptTokenIds));
      const primary = choosePrimaryZgmlResult(backends);
      return { ...primary, backends };
    } catch (error) {
      return {
        supported: false,
        probeReady: true,
        executableReady: false,
        stage: model == null ? "load" : "backend",
        probeMs,
        probe,
        error: error && error.message ? error.message : String(error),
      };
    } finally {
      if (model && typeof model.dispose === "function") model.dispose();
    }
  } catch (error) {
    return {
      supported: false,
      probeReady: false,
      executableReady: false,
      stage: "probe",
      error: error && error.message ? error.message : String(error),
    };
  }
}

const pytorch = runPytorch();
const zgml = runZgmlProbe(pytorch.modelPath, pytorch.promptTokenIds);
const comparisonReady = zgml.executableReady === true;
const artifact = {
  schema: "zgml.laptop-llm-pytorch-comparison.v1",
  createdAt: new Date().toISOString(),
  command: "scripts/check_laptop_llm_pytorch_comparison.cjs",
  suiteId,
  modelId,
  prompt,
  platform: {
    type: os.type(),
    platform: os.platform(),
    arch: os.arch(),
    release: os.release(),
    cpus: os.cpus().length,
  },
  status: {
    pytorchReady: true,
    zgmlProbeReady: zgml.probeReady === true,
    zgmlReady: zgml.executableReady === true,
    comparisonReady,
  },
  pytorch,
  zgml,
};

let artifactPath = null;
if (writeArtifact) {
  mkdirSync(artifactDir, { recursive: true });
  artifactPath = join(artifactDir, `laptop-llm-pytorch-${timestampForArtifact()}-${process.pid}.json`);
  writeFileSync(artifactPath, `${artifactJson(artifact)}\n`);
}

console.log([
  "laptop llm pytorch comparison:",
  comparisonReady ? "pass" : zgml.probeReady ? "probe-only" : "unsupported",
  `suite=${suiteId}`,
  `model=${modelId}`,
  `torch=${pytorch.torchVersion}`,
  `threads=${pytorch.threads}`,
  `prompt_tokens=${pytorch.promptTokens}`,
  `pytorch_prefill=${round(pytorch.prefillTokS)}tok/s`,
  `pytorch_decode=${round(pytorch.decodeTokS)}tok/s`,
  `pytorch_load=${round(pytorch.loadMs)}ms`,
  `pytorch_rss_load_delta=${round(pytorch.rssLoadDeltaBytes / 1024 / 1024)}MiB`,
  `zgml=${zgml.executableReady ? "supported" : zgml.probeReady ? "probe-only" : "unsupported"}`,
  `zgml_stage=${zgml.stage}`,
  `zgml_backend=${zgml.backend || "none"}`,
  `zgml_model=${zgml.probe && zgml.probe.modelKind ? zgml.probe.modelKind : "none"}`,
  `zgml_prefill=${round(zgml.prefillTokS)}tok/s`,
  `zgml_decode=${round(zgml.decodeTokS)}tok/s`,
  `zgml_load=${round(zgml.loadMs)}ms`,
  `zgml_compile=${round(zgml.compileMs)}ms`,
  `zgml_backends=${Array.isArray(zgml.backends) ? zgml.backends.map((result) => `${result.backend}:${result.stage}:${round(result.prefillTokS)}/${round(result.decodeTokS)}`).join(",") : "none"}`,
  `zgml_error=${zgml.error ? JSON.stringify(zgml.error) : "none"}`,
  `artifact=${artifactPath ? artifactPath.replace(`${root}/`, "") : "disabled"}`,
].join(" "));
