"use strict";

const { existsSync, readFileSync } = require("node:fs");
const { join, resolve } = require("node:path");
const { spawnSync } = require("node:child_process");

const root = resolve(__dirname, "..");
const docsPath = join(root, "docs", "frontend-replacement-confidence.md");
const smokePath = join(root, "src", "ts", "smokes", "package_smoke_core.ts");
const docs = readFileSync(docsPath, "utf8");
const smoke = readFileSync(smokePath, "utf8");
const skipRuntime = process.env.ZGML_FRONTEND_CONFIDENCE_STATIC === "1";

const workflows = Object.freeze([
  {
    name: "Linear regression",
    docNeedles: [
      "| Linear regression |",
      "`examples/node_training/train_linear.cjs`",
      "checkpoint JSON round-trip",
    ],
    smokeNeedles: [
      "train.fitModule",
      "checkpoint.create",
      "checkpoint.restore",
    ],
    commands: [
      ["node", ["examples/node_training/train_linear.cjs"]],
    ],
  },
  {
    name: "MLP classifier/regressor",
    docNeedles: [
      "| MLP classifier/regressor |",
      "`examples/node_training/train_mlp.cjs`",
      "compiled/restored output parity against eager",
    ],
    smokeNeedles: [
      "compileForInference",
      "compileForInference forward",
      "compileForInference into",
    ],
    commands: [
      ["node", ["examples/node_training/train_mlp.cjs"]],
    ],
  },
  {
    name: "MNIST MLP benchmark smoke",
    docNeedles: [
      "| MNIST MLP benchmark smoke |",
      "`examples/node_training/train_mnist_mlp.cjs`",
      "`smoke:training:mnist`",
      "eager/restored/compiled logits parity",
    ],
    smokeNeedles: [
      "compileForInference",
      "compileForInference forward",
      "compileForInference into",
    ],
    commands: [
      ["node", ["examples/node_training/train_mnist_mlp.cjs"]],
    ],
  },
  {
    name: "Classifier with cross entropy",
    docNeedles: [
      "| Classifier with cross entropy |",
      "`examples/node_training/train_classifier.cjs`",
      "`Linear -> LogSoftmax` package smoke Program evidence",
    ],
    smokeNeedles: [
      "expected Linear/LogSoftmax classifier Program evidence",
      "batched Linear/LogSoftmax classifier Program evidence",
      "batched log-softmax classifier eager/compiled parity",
    ],
    commands: [
      ["node", ["examples/node_training/train_classifier.cjs"]],
    ],
  },
  {
    name: "Conv2d feature model",
    docNeedles: [
      "| Conv2d feature model |",
      "`examples/node_training/train_conv2d.cjs`",
      "Conv2d eager, gradient, stateDict, compileSupport",
      "`docs/frontend-autograd-coverage.md` Conv/pool row",
    ],
    smokeNeedles: [
      "nn.Conv2d input grad",
      "nn.Conv2d stateDict weight",
      "nn.Conv2d batched compiled output",
    ],
    commands: [
      ["node", ["examples/node_training/train_conv2d.cjs"]],
      ["node", ["scripts/check_frontend_autograd_coverage.cjs"]],
    ],
  },
  {
    name: "Embedding/token classifier",
    docNeedles: [
      "| Embedding/token classifier |",
      "`examples/node_training/train_token_classifier.cjs`",
      "Embedding/Linear/LogSoftmax token-head Program evidence",
      "checkpoint restore and allocation-free compiled logits",
    ],
    smokeNeedles: [
      "expected Embedding/Linear/LogSoftmax token-head Program evidence",
      "lazyEmbeddingGraph",
      "lazyEmbeddingModuleGraph",
    ],
    commands: [
      ["node", ["examples/node_training/train_token_classifier.cjs"]],
    ],
  },
  {
    name: "Checkpointed compiled inference",
    docNeedles: [
      "| Checkpointed compiled inference |",
      "`examples/quickstart/zgml-first.cjs`",
      "`compile.compileForInference` package smoke",
    ],
    smokeNeedles: [
      "expected compileForInference handle to expose compile evidence",
      "compileForInference prepareInto",
      "compileForInference into",
    ],
    commands: [
      ["node", ["examples/quickstart/zgml-first.cjs"]],
      ["node", ["examples/node_training/quickstart.cjs"]],
    ],
  },
]);

const runtimeCommands = Object.freeze([
  ["node", ["dist/smokes/node_package_smoke.cjs"]],
  ...workflows.flatMap((workflow) => workflow.commands),
]);

function fail(message) {
  console.error(message);
  process.exitCode = 1;
}

function requireIncludes(source, path, needle, label) {
  if (!source.includes(needle)) fail(`${label}: ${path} must include ${JSON.stringify(needle)}`);
}

for (const workflow of workflows) {
  for (const needle of workflow.docNeedles) {
    requireIncludes(docs, "docs/frontend-replacement-confidence.md", needle, workflow.name);
  }
  for (const needle of workflow.smokeNeedles) {
    requireIncludes(smoke, "src/ts/smokes/package_smoke_core.ts", needle, workflow.name);
  }
}

if (process.exitCode) process.exit(process.exitCode);

if (skipRuntime) {
  console.log(`frontend confidence static ok: ${workflows.length} workflows`);
  process.exit(0);
}

if (!existsSync(join(root, "dist", "smokes", "node_package_smoke.cjs"))) {
  fail("dist package is missing; run `npm run build:package` before the frontend confidence runner");
  process.exit(process.exitCode);
}

for (const [cmd, args] of runtimeCommands) {
  const label = [cmd, ...args].join(" ");
  console.log(`[frontend-confidence] ${label}`);
  const result = spawnSync(cmd, args, {
    cwd: root,
    stdio: "inherit",
    env: process.env,
  });
  if (result.status !== 0) {
    fail(`[frontend-confidence] failed: ${label}`);
    process.exit(process.exitCode);
  }
}

console.log(`frontend confidence ok: ${workflows.length} workflows, ${runtimeCommands.length} runtime commands`);
