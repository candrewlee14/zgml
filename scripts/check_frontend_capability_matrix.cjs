"use strict";

const { readFileSync } = require("node:fs");
const { join, resolve } = require("node:path");

const root = resolve(__dirname, "..");
const matrixPath = "docs/frontend-capability-matrix.md";
const confidencePath = "docs/frontend-replacement-confidence.md";
const surfacePath = "src/ts/public_surface.ts";
const matrix = readFileSync(join(root, matrixPath), "utf8");
const confidence = readFileSync(join(root, confidencePath), "utf8");
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
const requiredConfidenceWorkflows = Object.freeze([
  "Linear regression",
  "MLP classifier/regressor",
  "Classifier with cross entropy",
  "Conv2d feature model",
  "Embedding/token classifier",
  "Checkpointed compiled inference",
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

function confidenceRows() {
  return confidence
    .split(/\r?\n/)
    .filter((line) => line.startsWith("| ") && !line.startsWith("| ---"))
    .slice(1)
    .map((line) => line.split("|").slice(1, -1).map((cell) => cell.trim()))
    .filter((cells) => cells.length === 4)
    .map(([workflow, userStory, evidence, remainingGap]) => ({
      workflow,
      userStory,
      evidence,
      remainingGap,
    }));
}

const firstContactNamespaces = exportedConstArray(publicSurface, "firstContactRootNamespaces");
const firstContactValues = exportedConstArray(publicSurface, "firstContactRootValues");
const stableNamespaces = exportedConstArray(publicSurface, "stableRootNamespaces");
const stableValues = exportedConstArray(publicSurface, "stableRootValues");
const firstContactDoc = firstContactBlock();
const runtimeHandle = "zgml.native(...)";
const docFirstContactValues = firstContactDoc.filter((entry) => firstContactValues.includes(entry));
const docFirstContactNamespaces = firstContactDoc.filter((entry) => !firstContactValues.includes(entry) && entry !== runtimeHandle);

if (!firstContactValues.includes("zgml") || !firstContactDoc.includes("zgml")) {
  errors.push("first-contact surface must name zgml as the user-facing root value");
}
if (!firstContactDoc.includes(runtimeHandle)) {
  errors.push(`first-contact surface must name ${runtimeHandle} as the friendly runtime handle`);
}
for (const name of firstContactValues) {
  if (!docFirstContactValues.includes(name)) errors.push(`${matrixPath} first-contact block missing root value ${name}`);
  if (!stableValues.includes(name)) errors.push(`${surfacePath} first-contact root value ${name} must also be stable`);
}
for (const name of firstContactNamespaces) {
  if (!docFirstContactNamespaces.includes(name)) errors.push(`${matrixPath} first-contact block missing namespace ${name}`);
  if (!stableNamespaces.includes(name)) errors.push(`${surfacePath} first-contact namespace ${name} must also be stable`);
}
for (const name of docFirstContactValues) {
  if (!firstContactValues.includes(name)) errors.push(`${matrixPath} first-contact block has stale root value ${name}`);
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
requireIncludes(matrix, matrixPath, "docs/frontend-replacement-confidence.md");
requireIncludes(matrix, matrixPath, "linear regression, classifiers, Conv2d feature models, token heads, and");
requireIncludes(matrix, matrixPath, "Raising the performance substrate score should require benchmark artifacts, not");
requireIncludes(matrix, matrixPath, "README/example import/export recipes");
requireIncludes(matrix, matrixPath, "ordinary TS module weights use state-dict/checkpoint save/load recipes");
requireIncludes(matrix, matrixPath, "broader third-party weight-format adapters remain future work");
requireIncludes(matrix, matrixPath, "Node and Bun also expose the first stateless native eager primitive");
requireIncludes(matrix, matrixPath, "On Node and Bun,");
requireIncludes(matrix, matrixPath, "`nn.Linear.forward` calls inside `zgml.noGrad(...)`");
requireIncludes(matrix, matrixPath, "`nativeEagerModuleForwardMs`");
requireIncludes(matrix, matrixPath, "`nativeEagerModuleSpeedup`");
requireIncludes(matrix, matrixPath, "`nativeEagerModuleMaxAbsDiff`");

const workflowRows = confidenceRows();
const workflowNames = workflowRows.map((row) => row.workflow);
for (const workflow of requiredConfidenceWorkflows) {
  if (!workflowNames.includes(workflow)) errors.push(`${confidencePath} workflow table missing row: ${workflow}`);
}
const duplicateWorkflowRows = workflowNames.filter((name, index) => workflowNames.indexOf(name) !== index);
if (duplicateWorkflowRows.length > 0) {
  errors.push(`${confidencePath} workflow table has duplicate rows: ${[...new Set(duplicateWorkflowRows)].join(", ")}`);
}
for (const row of workflowRows) {
  if (row.userStory.length < 32) errors.push(`${confidencePath} row ${row.workflow} must keep a concrete user story`);
  if (row.evidence.length < 40) errors.push(`${confidencePath} row ${row.workflow} must point at concrete executable evidence`);
  if (row.remainingGap.length < 24) errors.push(`${confidencePath} row ${row.workflow} must keep an honest remaining gap`);
}
requireIncludes(confidence, confidencePath, "user workflow -> eager/autograd proof -> checkpoint or state proof -> compiled inference proof");
requireIncludes(confidence, confidencePath, "`compileForInference`, `compileSupport`, `kernelPlan`, or a Program/Session");
requireIncludes(confidence, confidencePath, "New replacement claims should attach to one of these workflows");
requireIncludes(confidence, confidencePath, "`zgml` remains the canonical first-contact namespace");
requireIncludes(confidence, confidencePath, "`torch` compatibility");

if (errors.length > 0) {
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

console.log(`frontend capability matrix ok: ${rows.length} capabilities, ${workflowRows.length} workflows, ${firstContactNamespaces.length} first-contact namespaces, ${firstContactValues.length} first-contact root values`);
