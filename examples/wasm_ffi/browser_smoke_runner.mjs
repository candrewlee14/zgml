import { spawn } from "node:child_process";
import { createReadStream } from "node:fs";
import { mkdtemp, rm, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { platform, tmpdir } from "node:os";
import { basename, extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { createServer as createTcpServer } from "node:net";

const repoRoot = resolve(fileURLToPath(new URL("../..", import.meta.url)));

function parseArgs(argv) {
  const options = {
    chrome: process.env.CHROME_PATH ?? process.env.CHROME ?? "",
    enableUnsafeWebGpu: false,
    llamaProfileLabels: [],
    requireGpu: false,
    showLog: false,
    timeoutMs: 120_000,
  };
  for (const arg of argv) {
    if (arg === "--enable-unsafe-webgpu") {
      options.enableUnsafeWebGpu = true;
    } else if (arg === "--require-gpu") {
      options.requireGpu = true;
    } else if (arg === "--show-log") {
      options.showLog = true;
    } else if (arg.startsWith("--llama-profile-label=")) {
      const labels = arg.slice("--llama-profile-label=".length).split(",").map((label) => label.trim()).filter(Boolean);
      if (labels.length === 0) throw new Error("empty --llama-profile-label");
      options.llamaProfileLabels.push(...labels);
    } else if (arg.startsWith("--timeout-ms=")) {
      options.timeoutMs = Number(arg.slice("--timeout-ms=".length));
    } else if (arg.startsWith("--chrome=")) {
      options.chrome = arg.slice("--chrome=".length);
    } else {
      throw new Error(`unknown argument: ${arg}`);
    }
  }
  if (!Number.isSafeInteger(options.timeoutMs) || options.timeoutMs <= 0) {
    throw new Error(`invalid --timeout-ms: ${options.timeoutMs}`);
  }
  options.llamaProfileLabels = [...new Set(options.llamaProfileLabels)];
  return options;
}

function focusedLlamaProfileLabelSet(labels) {
  const focused = new Set();
  for (const label of labels) {
    const baseLabel = label.startsWith("greedy-") ? label.slice("greedy-".length) : label;
    focused.add(baseLabel);
    focused.add(`greedy-${baseLabel}`);
  }
  return focused;
}

async function fileExists(path) {
  try {
    const info = await stat(path);
    return info.isFile();
  } catch {
    return false;
  }
}

async function findChrome(explicit) {
  const candidates = [
    explicit,
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (await fileExists(candidate)) return candidate;
  }
  throw new Error("Chrome/Chromium not found; set CHROME_PATH or pass --chrome=/path/to/chrome");
}

function mimeType(path) {
  switch (extname(path)) {
    case ".html": return "text/html; charset=utf-8";
    case ".js":
    case ".mjs": return "text/javascript; charset=utf-8";
    case ".wasm": return "application/wasm";
    case ".json": return "application/json; charset=utf-8";
    case ".css": return "text/css; charset=utf-8";
    default: return "application/octet-stream";
  }
}

function isInsideRoot(path) {
  return path === repoRoot || path.startsWith(`${repoRoot}${sep}`);
}

async function startStaticServer() {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? "/", "http://127.0.0.1");
      let pathname = decodeURIComponent(url.pathname);
      if (pathname === "/") pathname = "/examples/wasm_ffi/browser.html";
      const fullPath = resolve(repoRoot, pathname.slice(1));
      if (!isInsideRoot(fullPath)) {
        res.writeHead(403);
        res.end("forbidden");
        return;
      }
      const info = await stat(fullPath);
      if (!info.isFile()) {
        res.writeHead(404);
        res.end("not found");
        return;
      }
      res.writeHead(200, {
        "Content-Length": info.size,
        "Content-Type": mimeType(fullPath),
      });
      createReadStream(fullPath).pipe(res);
    } catch {
      res.writeHead(404);
      res.end("not found");
    }
  });
  await new Promise((resolveServer, rejectServer) => {
    server.once("error", rejectServer);
    server.listen(0, "127.0.0.1", () => resolveServer());
  });
  const address = server.address();
  return { port: address.port, server };
}

async function freePort() {
  const server = createTcpServer();
  await new Promise((resolveListen, rejectListen) => {
    server.once("error", rejectListen);
    server.listen(0, "127.0.0.1", () => resolveListen());
  });
  const { port } = server.address();
  await new Promise((resolveClose) => server.close(resolveClose));
  return port;
}

async function findPageTarget(debugPort) {
  const targets = await fetch(`http://127.0.0.1:${debugPort}/json/list`).then((res) => res.json());
  const target = targets.find((candidate) => candidate.type === "page" && candidate.webSocketDebuggerUrl);
  if (target) return target;
  return fetch(`http://127.0.0.1:${debugPort}/json/new?${encodeURIComponent("about:blank")}`, { method: "PUT" }).then((res) => res.json());
}

async function waitForDevTools(port, timeoutMs, chromeStatus = null) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    if (chromeStatus?.exited === true) {
      throw new Error(
        `Chrome/Chromium not usable: exited before DevTools started ` +
        `(exit=${chromeStatus.exitCode ?? "null"} signal=${chromeStatus.signalCode ?? "null"})`,
      );
    }
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (response.ok) return response.json();
    } catch (err) {
      lastError = err;
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 100));
  }
  throw new Error(`Chrome DevTools did not start: ${lastError?.message ?? "timeout"}`);
}

function cdpSession(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let nextId = 1;
  const pending = new Map();
  const events = [];
  ws.addEventListener("message", (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) {
      const request = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) request.reject(new Error(JSON.stringify(msg.error)));
      else request.resolve(msg.result ?? {});
      return;
    }
    if (msg.method === "Runtime.consoleAPICalled") {
      events.push(`console:${msg.params.type}:${(msg.params.args ?? []).map((arg) => arg.value ?? arg.description ?? "").join(" ")}`);
    } else if (msg.method === "Runtime.exceptionThrown") {
      events.push(`exception:${msg.params.exceptionDetails?.text ?? ""}`);
    }
  });
  const opened = new Promise((resolveOpen, rejectOpen) => {
    ws.addEventListener("open", resolveOpen, { once: true });
    ws.addEventListener("error", rejectOpen, { once: true });
  });
  return {
    events,
    async open() {
      await opened;
    },
    async send(method, params = {}, timeoutMs = 0) {
      const id = nextId;
      nextId += 1;
      ws.send(JSON.stringify({ id, method, params }));
      return new Promise((resolveSend, rejectSend) => {
        let timer;
        const resolve = (value) => {
          if (timer) clearTimeout(timer);
          resolveSend(value);
        };
        const reject = (err) => {
          if (timer) clearTimeout(timer);
          rejectSend(err);
        };
        if (timeoutMs > 0) {
          timer = setTimeout(() => {
            pending.delete(id);
            reject(new Error(`Chrome DevTools ${method} timed out after ${timeoutMs}ms`));
          }, timeoutMs);
        }
        pending.set(id, { resolve, reject });
      });
    },
    async close() {
      for (const [id, request] of pending) {
        pending.delete(id);
        request.reject(new Error(`Chrome DevTools request ${id} canceled during session close`));
      }
      if (ws.readyState === WebSocket.CLOSED) return;
      await Promise.race([
        new Promise((resolveClose) => {
          ws.addEventListener("close", resolveClose, { once: true });
          ws.close();
        }),
        new Promise((resolveTimeout) => setTimeout(resolveTimeout, 1_000)),
      ]);
    },
  };
}

async function closeServer(server) {
  const closed = new Promise((resolveClose) => server.close(resolveClose));
  server.closeIdleConnections?.();
  server.closeAllConnections?.();
  await Promise.race([
    closed,
    new Promise((resolveTimeout) => setTimeout(resolveTimeout, 1_000)),
  ]);
}

async function stopChrome(chrome) {
  if (!chrome || chrome.exitCode !== null) return;
  chrome.kill("SIGTERM");
  const exited = await Promise.race([
    new Promise((resolveExit) => chrome.once("exit", resolveExit)),
    new Promise((resolveTimeout) => setTimeout(resolveTimeout, 2_000, "timeout")),
  ]);
  if (exited === "timeout" && chrome.exitCode === null) chrome.kill("SIGKILL");
  chrome.stdout?.destroy();
  chrome.stderr?.destroy();
}

function chromeArgs(options, userDataDir, debugPort) {
  const args = [
    "--headless=new",
    "--remote-debugging-address=127.0.0.1",
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${userDataDir}`,
    "--no-first-run",
    "--no-default-browser-check",
    "about:blank",
  ];
  if (options.enableUnsafeWebGpu) {
    args.splice(1, 0, "--enable-unsafe-webgpu");
    if (platform() === "darwin") {
      args.splice(2, 0, "--use-angle=metal");
    } else {
      args.splice(2, 0, "--enable-features=Vulkan,UseSkiaRenderer");
    }
  } else {
    args.splice(1, 0, "--disable-gpu");
  }
  return args;
}

async function runBrowserSmoke(options) {
  const chromePath = await findChrome(options.chrome);
  const userDataDir = await mkdtemp(join(tmpdir(), "zgml-browser-smoke-"));
  const { port: httpPort, server } = await startStaticServer();
  const debugPort = await freePort();
  const chrome = spawn(chromePath, chromeArgs(options, userDataDir, debugPort), {
    stdio: ["ignore", "pipe", "pipe"],
  });
  const chromeStatus = { exitCode: null, exited: false, signalCode: null };
  chrome.once("exit", (code, signal) => {
    chromeStatus.exitCode = code;
    chromeStatus.exited = true;
    chromeStatus.signalCode = signal;
  });
  const chromeLog = [];
  chrome.stdout.on("data", (chunk) => chromeLog.push(String(chunk)));
  chrome.stderr.on("data", (chunk) => chromeLog.push(String(chunk)));
  let session;
  try {
    await waitForDevTools(debugPort, 15_000, chromeStatus);
    const page = new URL(`http://127.0.0.1:${httpPort}/examples/wasm_ffi/browser.html`);
    for (const label of options.llamaProfileLabels) page.searchParams.append("llamaProfileLabel", label);
    const pageUrl = String(page);
    const target = await findPageTarget(debugPort);
    session = cdpSession(target.webSocketDebuggerUrl);
    await session.open();
    const setupTimeoutMs = Math.min(options.timeoutMs, 60_000);
    await session.send("Runtime.enable", {}, setupTimeoutMs);
    await session.send("Page.enable", {}, setupTimeoutMs);
    await session.send("Page.navigate", { url: pageUrl }, setupTimeoutMs);
    let value = {};
    const start = Date.now();
    while (Date.now() - start < options.timeoutMs) {
      const remainingMs = options.timeoutMs - (Date.now() - start);
      const result = await session.send("Runtime.evaluate", {
        expression: `JSON.stringify({
          state: document.documentElement.dataset.zgmlWasmSmoke || "",
          stage: document.documentElement.dataset.zgmlWasmSmokeStage || "",
          mode: document.documentElement.dataset.zgmlWasmGpuResources || "",
          available: document.documentElement.dataset.zgmlWasmGpuAvailable || "",
          reason: document.documentElement.dataset.zgmlWasmGpuReason || "",
          maxStorageBuffers: document.documentElement.dataset.zgmlWasmGpuMaxStorageBuffers || "",
          storageAlignment: document.documentElement.dataset.zgmlWasmGpuStorageAlignment || "",
          canBindBlockPipeline: document.documentElement.dataset.zgmlWasmGpuCanBindBlockPipeline || "",
          llamaProfileCount: document.documentElement.dataset.zgmlWasmLlamaProfileCount || "",
          llamaBackendDispatches: document.documentElement.dataset.zgmlWasmLlamaBackendDispatches || "",
          llamaExecutorBackendDispatches: document.documentElement.dataset.zgmlWasmLlamaExecutorBackendDispatches || "",
          llamaSelectionBackendDispatches: document.documentElement.dataset.zgmlWasmLlamaSelectionBackendDispatches || "",
          llamaFallbackOps: document.documentElement.dataset.zgmlWasmLlamaFallbackOps || "",
          llamaCommandCount: document.documentElement.dataset.zgmlWasmLlamaCommandCount || "",
          llamaGenericDispatches: document.documentElement.dataset.zgmlWasmLlamaGenericDispatches || "",
          llamaScalarDispatches: document.documentElement.dataset.zgmlWasmLlamaScalarDispatches || "",
          llamaWindowDispatches: document.documentElement.dataset.zgmlWasmLlamaWindowDispatches || "",
          llamaStorageCallCount: document.documentElement.dataset.zgmlWasmLlamaStorageCallCount || "",
          llamaGpuStorageCalls: document.documentElement.dataset.zgmlWasmLlamaGpuStorageCalls || "",
          llamaMockStorageCalls: document.documentElement.dataset.zgmlWasmLlamaMockStorageCalls || "",
          llamaAdapterLimitStorageCalls: document.documentElement.dataset.zgmlWasmLlamaAdapterLimitStorageCalls || "",
          llamaUnknownStorageCalls: document.documentElement.dataset.zgmlWasmLlamaUnknownStorageCalls || "",
          llamaMaxRequiredStorageBuffers: document.documentElement.dataset.zgmlWasmLlamaMaxRequiredStorageBuffers || "",
          llamaStorageLabels: document.documentElement.dataset.zgmlWasmLlamaStorageLabels || "",
          llamaOutputReads: document.documentElement.dataset.zgmlWasmLlamaOutputReads || "",
          llamaSelectionReads: document.documentElement.dataset.zgmlWasmLlamaSelectionReads || "",
          llamaSyncs: document.documentElement.dataset.zgmlWasmLlamaSyncs || "",
          llamaProfileLabels: document.documentElement.dataset.zgmlWasmLlamaProfileLabels || "",
          llamaLastProfileLabel: document.documentElement.dataset.zgmlWasmLlamaLastProfileLabel || "",
          lastMessage: document.documentElement.dataset.zgmlWasmSmokeLastMessage || "",
          error: document.documentElement.dataset.zgmlWasmSmokeError || "",
          text: document.getElementById("log")?.textContent || ""
        })`,
        returnByValue: true,
      }, remainingMs);
      value = JSON.parse(result.result.value);
      if (value.state === "passed" || value.state === "failed") {
        value.elapsedMs = Date.now() - start;
        break;
      }
      await new Promise((resolveDelay) => setTimeout(resolveDelay, 500));
    }
    if (!value.elapsedMs) value.elapsedMs = Date.now() - start;
    await session.send("Target.closeTarget", { targetId: target.id }).catch(() => {});
    if (value.state !== "passed") {
      throw new Error(`browser smoke did not pass: ${JSON.stringify({ ...value, events: session.events }, null, 2)}`);
    }
    if (options.requireGpu && value.mode !== "gpu-buffer") {
      throw new Error(`browser smoke passed without real GPUBuffer mode: ${JSON.stringify(value, null, 2)}`);
    }
    if (options.requireGpu && value.canBindBlockPipeline !== "true") {
      throw new Error(`browser GPUBuffer smoke cannot bind LLaMA block pipeline: ${JSON.stringify(value, null, 2)}`);
    }
    const llamaProfileCount = Number(value.llamaProfileCount || 0);
    const llamaBackendDispatches = Number(value.llamaBackendDispatches || 0);
    const llamaExecutorBackendDispatches = Number(value.llamaExecutorBackendDispatches || 0);
    const llamaSelectionBackendDispatches = Number(value.llamaSelectionBackendDispatches || 0);
    const llamaFallbackOps = Number(value.llamaFallbackOps || 0);
    const llamaGenericDispatches = Number(value.llamaGenericDispatches || 0);
    const llamaScalarDispatches = Number(value.llamaScalarDispatches || 0);
    const llamaWindowDispatches = Number(value.llamaWindowDispatches || 0);
    const llamaStorageCallCount = Number(value.llamaStorageCallCount || 0);
    const llamaGpuStorageCalls = Number(value.llamaGpuStorageCalls || 0);
    const llamaMockStorageCalls = Number(value.llamaMockStorageCalls || 0);
    const llamaAdapterLimitStorageCalls = Number(value.llamaAdapterLimitStorageCalls || 0);
    const llamaUnknownStorageCalls = Number(value.llamaUnknownStorageCalls || 0);
    const llamaMaxRequiredStorageBuffers = Number(value.llamaMaxRequiredStorageBuffers || 0);
    const labelList = String(value.llamaProfileLabels || "").split(",").filter(Boolean);
    const storageLabelList = String(value.llamaStorageLabels || "").split(",").filter(Boolean);
    const labels = new Set(labelList);
    const storageLabels = new Set(storageLabelList);
    const focusedLlamaLabels = options.llamaProfileLabels.length !== 0;
    if (options.requireGpu && llamaProfileCount <= 0) {
      throw new Error(`browser GPUBuffer smoke did not report LLaMA profile evidence: ${JSON.stringify(value, null, 2)}`);
    }
    if (options.requireGpu && llamaBackendDispatches <= 0) {
      throw new Error(`browser GPUBuffer smoke did not report LLaMA backend dispatches: ${JSON.stringify(value, null, 2)}`);
    }
    if (options.requireGpu && llamaExecutorBackendDispatches <= 0) {
      throw new Error(`browser GPUBuffer smoke did not report LLaMA executor backend dispatches: ${JSON.stringify(value, null, 2)}`);
    }
    if ((options.requireGpu || llamaProfileCount > 0) && llamaBackendDispatches !== llamaExecutorBackendDispatches + llamaSelectionBackendDispatches) {
      throw new Error(`browser smoke LLaMA aggregate dispatch evidence does not match executor+selection dispatches: ${JSON.stringify(value, null, 2)}`);
    }
    if (options.requireGpu && llamaFallbackOps !== 0) {
      throw new Error(`browser GPUBuffer smoke reported LLaMA fallback ops: ${JSON.stringify(value, null, 2)}`);
    }
    if (options.requireGpu && !focusedLlamaLabels && llamaScalarDispatches <= 0) {
      throw new Error(`browser GPUBuffer smoke did not report LLaMA scalar block dispatch evidence: ${JSON.stringify(value, null, 2)}`);
    }
    if (options.requireGpu && !focusedLlamaLabels && llamaWindowDispatches <= 0) {
      throw new Error(`browser GPUBuffer smoke did not report LLaMA window block dispatch evidence: ${JSON.stringify(value, null, 2)}`);
    }
    if (options.requireGpu && !focusedLlamaLabels && llamaStorageCallCount <= 0) {
      throw new Error(`browser GPUBuffer smoke did not report LLaMA storage-mode call evidence: ${JSON.stringify(value, null, 2)}`);
    }
    if (options.requireGpu && !focusedLlamaLabels && llamaGpuStorageCalls <= 0) {
      throw new Error(`browser GPUBuffer smoke did not report any GPU-storage block-pipeline calls: ${JSON.stringify(value, null, 2)}`);
    }
    if (options.requireGpu && llamaMockStorageCalls !== 0) {
      throw new Error(`browser GPUBuffer smoke reported mock-storage block-pipeline calls: ${JSON.stringify(value, null, 2)}`);
    }
    if (options.requireGpu && llamaAdapterLimitStorageCalls !== 0) {
      throw new Error(`browser GPUBuffer smoke reported adapter-limited block-pipeline calls: ${JSON.stringify(value, null, 2)}`);
    }
    if (options.requireGpu && llamaUnknownStorageCalls !== 0) {
      throw new Error(`browser GPUBuffer smoke reported unknown storage-mode block-pipeline calls: ${JSON.stringify(value, null, 2)}`);
    }
    if ((options.requireGpu || llamaStorageCallCount > 0) && llamaStorageCallCount !== llamaGpuStorageCalls + llamaMockStorageCalls + llamaAdapterLimitStorageCalls + llamaUnknownStorageCalls) {
      throw new Error(`browser smoke LLaMA storage-mode evidence does not balance: ${JSON.stringify(value, null, 2)}`);
    }
    if ((options.requireGpu || llamaStorageCallCount > 0) && llamaStorageCallCount !== storageLabelList.length) {
      throw new Error(`browser smoke LLaMA storage-mode call count does not match labels: ${JSON.stringify(value, null, 2)}`);
    }
    if (!focusedLlamaLabels && (options.requireGpu || llamaStorageCallCount > 0)) {
      const evidenceName = options.requireGpu ? "browser GPUBuffer smoke" : "browser smoke";
      const requiredStorageLabels = [
        "block-pipeline",
        "two-layer-pipeline",
        ...(options.requireGpu ? ["strict-default-two-layer"] : []),
        "native-default-two-layer",
        "sharded-two-layer",
        "native-sharded-two-layer",
        "f16-materialized-pipeline",
        "native-f16-materialized-pipeline",
        "bf16-materialized-pipeline",
        "native-bf16-materialized-pipeline",
        "three-layer-pipeline",
        "wider-pipeline",
        "realistic-head-pipeline",
        "metadata-mha-pipeline",
        "gguf-metadata-gqa-pipeline",
        "native-gguf-metadata-gqa-pipeline",
        "qwen2-gguf-metadata-gqa-pipeline",
        "native-qwen2-gguf-metadata-gqa-pipeline",
        "config-gqa-pipeline",
        "native-config-gqa-pipeline",
        "long-context-gqa-pipeline",
        "native-long-context-gqa-pipeline",
        "config-json-mqa-pipeline",
        "llama3-rope-gqa-pipeline",
        "native-llama3-rope-gqa-pipeline",
        "metadata-config-gqa-pipeline",
        "native-metadata-config-gqa-pipeline",
        "mistral-gqa-pipeline",
        "native-mistral-gqa-pipeline",
        "long-sliding-window-mistral-pipeline",
        "biased-qwen2-gqa-pipeline",
        "native-biased-qwen2-gqa-pipeline",
        "structural-biased-qwen2-gqa-pipeline",
        "native-structural-biased-qwen2-gqa-pipeline",
        "qwen2-gqa-pipeline",
        "native-qwen2-gqa-pipeline",
        "qwen3-qknorm-gqa-pipeline",
        "native-qwen3-qknorm-gqa-pipeline",
        "structural-qwen3-qknorm-gqa-pipeline",
        "native-structural-qwen3-qknorm-gqa-pipeline",
        "smollm3-nope-gqa-pipeline",
        "native-smollm3-nope-gqa-pipeline",
        "realistic-gqa-pipeline",
        "tied-lm-head-pipeline",
        "native-tied-lm-head-pipeline",
      ];
      const missingStorageLabels = requiredStorageLabels.filter((label) => !storageLabels.has(label));
      if (missingStorageLabels.length !== 0) {
        throw new Error(`${evidenceName} missing LLaMA storage-mode labels ${missingStorageLabels.join(", ")}: ${JSON.stringify(value, null, 2)}`);
      }
    }
    if (options.requireGpu && (!focusedLlamaLabels || llamaStorageCallCount > 0) && llamaMaxRequiredStorageBuffers < 6) {
      throw new Error(`browser GPUBuffer smoke did not report the six-storage-buffer block-pipeline requirement: ${JSON.stringify(value, null, 2)}`);
    }
    if (options.requireGpu || llamaProfileCount > 0) {
      const evidenceName = options.requireGpu ? "browser GPUBuffer smoke" : "browser smoke";
      if (labelList.length !== labels.size) {
        const duplicateLabels = labelList.filter((label, index) => labelList.indexOf(label) !== index);
        throw new Error(`${evidenceName} reported duplicate LLaMA profile labels ${[...new Set(duplicateLabels)].join(", ")}: ${JSON.stringify(value, null, 2)}`);
      }
      if (llamaProfileCount !== labelList.length) {
        throw new Error(`${evidenceName} LLaMA profile count does not match profile labels: ${JSON.stringify(value, null, 2)}`);
      }
      const focusedRequiredLabels = focusedLlamaLabels ? focusedLlamaProfileLabelSet(options.llamaProfileLabels) : null;
      if (focusedRequiredLabels) {
        const unexpectedLabels = labelList.filter((label) => !focusedRequiredLabels.has(label));
        if (unexpectedLabels.length !== 0) {
          throw new Error(`${evidenceName} focused LLaMA profile run reported unrelated labels ${unexpectedLabels.join(", ")}: ${JSON.stringify(value, null, 2)}`);
        }
      }
      const requiredLabels = focusedRequiredLabels ? [...focusedRequiredLabels] : [
        "host-token",
        "model-host-token",
        "embedding-projection",
        "embedding-projection-window",
        "rmsnorm-projection",
        "rmsnorm-projection-window",
        "kv-projection",
        "kv-projection-qknorm",
        "kv-projection-window",
        "attention-projection",
        "attention-projection-qknorm",
        "attention-projection-sliding-window",
        "attention-projection-llama3-rope",
        "attention-projection-smollm3-nope",
        "attention-projection-biased-qwen2",
        "attention-projection-prefill",
        "block-projection",
        "block-projection-biased-qwen2",
        "block-projection-qknorm-qwen3",
        "block-projection-sliding-window",
        "block-projection-llama3-rope",
        "block-projection-smollm3-nope",
        "block-projection-prefill",
        "block-pipeline",
        "two-layer-pipeline",
        "strict-default-two-layer",
        "ergonomic-two-layer",
        "ergonomic-greedy",
        "device-greedy",
        "ergonomic-sampled",
        "device-sampled",
        "sharded-two-layer",
        "f16-materialized-pipeline",
        "bf16-materialized-pipeline",
        "three-layer-pipeline",
        "wider-pipeline",
        "realistic-head-pipeline",
        "metadata-mha-pipeline",
        "gguf-metadata-gqa-pipeline",
        "qwen2-gguf-metadata-gqa-pipeline",
        "config-gqa-pipeline",
        "long-context-gqa-pipeline",
        "config-json-mqa-pipeline",
        "llama3-rope-gqa-pipeline",
        "metadata-config-gqa-pipeline",
        "mistral-gqa-pipeline",
        "long-sliding-window-mistral-pipeline",
        "biased-qwen2-gqa-pipeline",
        "structural-biased-qwen2-gqa-pipeline",
        "qwen2-gqa-pipeline",
        "qwen3-qknorm-gqa-pipeline",
        "structural-qwen3-qknorm-gqa-pipeline",
        "smollm3-nope-gqa-pipeline",
        "realistic-gqa-pipeline",
        "greedy-metadata-mha-pipeline",
        "greedy-gguf-metadata-gqa-pipeline",
        "greedy-qwen2-gguf-metadata-gqa-pipeline",
        "greedy-config-gqa-pipeline",
        "greedy-long-context-gqa-pipeline",
        "greedy-config-json-mqa-pipeline",
        "greedy-llama3-rope-gqa-pipeline",
        "greedy-metadata-config-gqa-pipeline",
        "greedy-mistral-gqa-pipeline",
        "greedy-long-sliding-window-mistral-pipeline",
        "greedy-biased-qwen2-gqa-pipeline",
        "greedy-structural-biased-qwen2-gqa-pipeline",
        "greedy-qwen2-gqa-pipeline",
        "greedy-qwen3-qknorm-gqa-pipeline",
        "greedy-structural-qwen3-qknorm-gqa-pipeline",
        "greedy-smollm3-nope-gqa-pipeline",
        "greedy-realistic-gqa-pipeline",
        "tied-lm-head-pipeline",
      ];
      const missingLabels = requiredLabels.filter((label) => !labels.has(label));
      if (missingLabels.length !== 0) {
        throw new Error(`${evidenceName} missing LLaMA profile labels ${missingLabels.join(", ")}: ${JSON.stringify(value, null, 2)}`);
      }
    }
    if (options.requireGpu && !focusedLlamaLabels) {
      if (Number(value.llamaSelectionReads || 0) <= 0) {
        throw new Error(`browser GPUBuffer smoke did not report LLaMA device-selection readback evidence: ${JSON.stringify(value, null, 2)}`);
      }
    }
    if (options.showLog) process.stdout.write(value.text);
    console.log(
      `zgml browser wasm smoke ok: mode=${value.mode} available=${value.available}` +
      ` reason=${value.reason || "ok"} maxStorageBuffers=${value.maxStorageBuffers}` +
      ` storageAlignment=${value.storageAlignment} canBindBlockPipeline=${value.canBindBlockPipeline}` +
      ` llamaProfiles=${value.llamaProfileCount || "0"}` +
      ` llamaBackendDispatches=${value.llamaBackendDispatches || "0"}` +
      ` llamaExecutorDispatches=${value.llamaExecutorBackendDispatches || "0"}` +
      ` llamaSelectionDispatches=${value.llamaSelectionBackendDispatches || "0"}` +
      ` llamaFallbackOps=${value.llamaFallbackOps || "0"}` +
      ` llamaCommandCount=${value.llamaCommandCount || "0"}` +
      ` llamaFamilyDispatches=${value.llamaGenericDispatches || "0"}/${value.llamaScalarDispatches || "0"}/${value.llamaWindowDispatches || "0"}` +
      ` llamaStorageCalls=${value.llamaStorageCallCount || "0"}/${value.llamaGpuStorageCalls || "0"}/${value.llamaMockStorageCalls || "0"}/${value.llamaAdapterLimitStorageCalls || "0"}/${value.llamaUnknownStorageCalls || "0"}` +
      ` llamaOutputReads=${value.llamaOutputReads || "0"}` +
      ` llamaSelectionReads=${value.llamaSelectionReads || "0"}` +
      ` llamaSyncs=${value.llamaSyncs || "0"}` +
      ` llamaProfileLabels=${String(value.llamaProfileLabels || "").split(",").filter(Boolean).length}` +
      ` elapsedMs=${value.elapsedMs}`,
    );
    return value;
  } catch (err) {
    if (chromeLog.length !== 0) {
      const tail = chromeLog.join("").split(/\r?\n/).filter(Boolean).slice(-20).join("\n");
      if (tail) console.error(`Chrome log tail:\n${tail}`);
    }
    if (session?.events.length) {
      console.error(`Chrome event tail:\n${session.events.slice(-40).join("\n")}`);
    }
    throw err;
  } finally {
    await session?.close();
    await stopChrome(chrome);
    await closeServer(server);
    await rm(userDataDir, { recursive: true, force: true });
  }
}

const options = parseArgs(process.argv.slice(2));
await runBrowserSmoke(options);
