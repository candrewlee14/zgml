"use strict";

const { readFileSync } = require("node:fs");
const { join, resolve } = require("node:path");

const root = resolve(__dirname, "..");
const smoke = readFileSync(join(root, "src/ts/smokes/package_smoke_core.ts"), "utf8");
const coverage = readFileSync(join(root, "docs/frontend-autograd-coverage.md"), "utf8");
const matrix = readFileSync(join(root, "docs/frontend-capability-matrix.md"), "utf8");

const rows = Object.freeze([
  {
    family: "Broadcasted binary math",
    labels: [
      "Tensor.maximum broadcast",
      "where broadcast helper",
      "Tensor.maskedFill value autograd backward",
    ],
  },
  {
    family: "Movement and view-style ops",
    labels: [
      "Tensor.flip autograd backward",
      "Tensor.roll autograd backward",
      "Tensor.split autograd backward",
    ],
  },
  {
    family: "Scalar unary math",
    labels: [
      "Tensor.reciprocal autograd backward",
      "Tensor.rsqrt autograd backward",
      "Tensor.sin autograd backward",
      "Tensor.tanh autograd backward",
    ],
  },
  {
    family: "Zero-gradient integer-like ops",
    labels: [
      "Tensor.floor zero gradient",
      "Tensor.round zero gradient",
    ],
  },
  {
    family: "Reductions",
    labels: [
      "Tensor.variance autograd backward",
      "Tensor.std autograd backward",
      "Tensor.norm autograd backward",
      "Tensor.prod autograd backward",
      "Tensor.cumsum autograd backward",
    ],
  },
  {
    family: "Log-sum-exp family",
    labels: [
      "Tensor.logsumexp autograd backward",
    ],
  },
  {
    family: "Linear algebra",
    labels: [
      "Tensor.dot lhs autograd backward",
      "Tensor.dot rhs autograd backward",
      "Tensor.trace autograd backward",
      "Tensor.diagonal autograd backward",
      "Tensor.bmm lhs autograd backward",
      "Tensor.bmm rhs autograd backward",
    ],
  },
  {
    family: "Losses",
    labels: [
      "tensor crossEntropy gradient",
      "tensor nllLoss gradient",
      "tensor nllLoss sum gradient",
    ],
  },
  {
    family: "Module parameters",
    labels: [
      "module zero_grad alias clears weight grad",
      "nn.zero_grad alias clears bias grad",
      "optimizer must skip frozen parameters with stale gradients",
    ],
  },
  {
    family: "Conv/pool modules",
    labels: [
      "nn.Conv2d input grad",
      "nn.Conv2d weight grad",
      "nn.Conv2d bias grad",
      "nn.MaxPool2d input grad",
    ],
  },
]);

const errors = [];

function requireIncludes(source, path, needle) {
  if (!source.includes(needle)) errors.push(`${path} must include: ${needle}`);
}

requireIncludes(coverage, "docs/frontend-autograd-coverage.md", "# Frontend Autograd Coverage");
requireIncludes(coverage, "docs/frontend-autograd-coverage.md", "src/ts/smokes/package_smoke_core.ts");
requireIncludes(matrix, "docs/frontend-capability-matrix.md", "Autograd coverage is broad enough for small model workflows");
requireIncludes(matrix, "docs/frontend-capability-matrix.md", "an autograd test proving backward behavior for a public op");

for (const row of rows) {
  requireIncludes(coverage, "docs/frontend-autograd-coverage.md", `| ${row.family} | checked |`);
  for (const label of row.labels) {
    requireIncludes(coverage, "docs/frontend-autograd-coverage.md", label);
    requireIncludes(smoke, "src/ts/smokes/package_smoke_core.ts", label);
  }
}

if (errors.length > 0) {
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

console.log(`frontend autograd coverage ok: ${rows.length} families, ${rows.reduce((sum, row) => sum + row.labels.length, 0)} runtime assertions`);
