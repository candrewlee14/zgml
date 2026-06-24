"use strict";

const { readFileSync } = require("node:fs");
const { join, resolve } = require("node:path");

const root = resolve(__dirname, "..");
const matrixPath = "docs/frontend-capability-matrix.md";
const surfacePath = "src/ts/public_surface.ts";
const matrix = readFileSync(join(root, matrixPath), "utf8");
const publicSurface = readFileSync(join(root, surfacePath), "utf8");

const allowedStatuses = new Set(["yes", "partial", "no", "n/a"]);
const requiredCapabilities = Object.freeze([
  "Tensor factories and metadata",
  "Elementwise tensor math",
  "Broadcasting",
  "Views and movement ops",
  "Reductions",
  "Matmul and linear algebra",
  "`einsum`",
  "Modules and containers",
  "Common neural layers",
  "Losses",
  "Optimizers and schedulers",
  "Training helpers",
  "State dicts and checkpoints",
  "Program compile support",
  "Program/Session hot path",
  "Node and Bun package use",
  "Browser/Wasm/WebGPU surface",
  "Safetensors/model-source interop",
]);

const errors = [];

function exportedConstArray(source, constName) {
  const match = source.match(new RegExp(`export const ${constName} = Object\\.freeze\\(\\[([\\s\\S]*?)\\] as const\\);`));
  if (!match) {
    errors.push(`${surfacePath} must export ${constName}`);
    return [];
  }
  return [...match[1].matchAll(/"([^"]+)"/g)].map((entry) => entry[1]);
}

function requireIncludes(source, path, needle) {
  if (!source.includes(needle)) errors.push(`${path} must include: ${needle}`);
}

function firstContactBlock() {
  const match = matrix.match(/```text\n([\s\S]*?)\n```/);
  if (!match) {
    errors.push(`${matrixPath} must keep a first-contact text block`);
    return [];
  }
  return match[1]
    .split(/\r?\n/)
    .flatMap((line) => line.split(","))
    .map((part) => part.trim())
    .filter(Boolean);
}

function capabilityRows() {
  return matrix
    .split(/\r?\n/)
    .filter((line) => line.startsWith("| ") && !line.startsWith("| ---"))
    .slice(1)
    .map((line) => line.split("|").slice(1, -1).map((cell) => cell.trim()))
    .filter((cells) => cells.length === 6)
    .map(([capability, eager, autograd, shapeSafety, nativeProgram, notes]) => ({
      capability,
      statuses: { eager, autograd, shapeSafety, nativeProgram },
      notes,
    }));
}

const firstContactNamespaces = exportedConstArray(publicSurface, "firstContactRootNamespaces");
const firstContactValues = exportedConstArray(publicSurface, "firstContactRootValues");
const stableNamespaces = exportedConstArray(publicSurface, "stableRootNamespaces");
const firstContactDoc = firstContactBlock();
const docFirstContactNamespaces = firstContactDoc.filter((entry) => entry !== "zgml" && entry !== "compile.compileForInference(...)");

if (!firstContactValues.includes("zgml") || !firstContactDoc.includes("zgml")) {
  errors.push("first-contact surface must name zgml as the user-facing root value");
}
if (!firstContactDoc.includes("compile.compileForInference(...)")) {
  errors.push("first-contact surface must name compile.compileForInference(...) as the friendly runtime handle");
}
for (const name of firstContactNamespaces) {
  if (!docFirstContactNamespaces.includes(name)) errors.push(`${matrixPath} first-contact block missing namespace ${name}`);
  if (!stableNamespaces.includes(name)) errors.push(`${surfacePath} first-contact namespace ${name} must also be stable`);
}
for (const name of docFirstContactNamespaces) {
  if (!firstContactNamespaces.includes(name)) errors.push(`${matrixPath} first-contact block has stale namespace ${name}`);
}
for (const advancedName of ["program", "session", "nativeBuffer", "programDevice", "inspection", "modelSource"]) {
  if (docFirstContactNamespaces.includes(advancedName)) {
    errors.push(`${matrixPath} first-contact block must not teach advanced namespace ${advancedName}`);
  }
}

const rows = capabilityRows();
const rowNames = rows.map((row) => row.capability);
for (const required of requiredCapabilities) {
  if (!rowNames.includes(required)) errors.push(`${matrixPath} capability table missing row: ${required}`);
}
const duplicateRows = rowNames.filter((name, index) => rowNames.indexOf(name) !== index);
if (duplicateRows.length > 0) {
  errors.push(`${matrixPath} capability table has duplicate rows: ${[...new Set(duplicateRows)].join(", ")}`);
}
for (const row of rows) {
  for (const [column, status] of Object.entries(row.statuses)) {
    if (!allowedStatuses.has(status)) {
      errors.push(`${matrixPath} row ${row.capability} has invalid ${column} status ${status}`);
    }
  }
  if (row.notes.length < 24) {
    errors.push(`${matrixPath} row ${row.capability} must keep a concrete evidence note`);
  }
}

requireIncludes(matrix, matrixPath, "Raising the frontend replacement score should require one of these:");
requireIncludes(matrix, matrixPath, "a type smoke or package smoke proving a user-visible API");
requireIncludes(matrix, matrixPath, "a focused runtime or module Program benchmark proving native lowering");
requireIncludes(matrix, matrixPath, "an autograd test proving backward behavior for a public op");
requireIncludes(matrix, matrixPath, "a README/tutorial example that exercises the full user workflow");
requireIncludes(matrix, matrixPath, "a scorecard check that keeps the capability from drifting");
requireIncludes(matrix, matrixPath, "Raising the performance substrate score should require benchmark artifacts, not");

if (errors.length > 0) {
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

console.log(`frontend capability matrix ok: ${rows.length} capabilities, ${firstContactNamespaces.length} first-contact namespaces`);
