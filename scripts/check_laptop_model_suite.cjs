"use strict";

const { readFileSync } = require("node:fs");
const { join, resolve } = require("node:path");

const root = resolve(__dirname, "..");
const manifestPath = join(root, "benchmarks", "laptop-model-suite.json");
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));

const requiredDomains = [
  "llm",
  "text-encoder",
  "vision",
  "multimodal-encoder",
  "diffusion",
  "voice",
];
const requiredTiers = ["pr", "laptop_core", "frontier"];
const allowedStatuses = new Set(["ready", "next", "target", "unsupported-target"]);
const errors = [];

function fail(message) {
  errors.push(message);
}

if (manifest.schema !== "zgml.laptop-model-suite.v1") {
  fail(`unexpected schema: ${manifest.schema}`);
}
for (const tier of requiredTiers) {
  if (!manifest.tiers || !manifest.tiers[tier]) {
    fail(`missing tier: ${tier}`);
  }
}
if (!Array.isArray(manifest.models) || manifest.models.length === 0) {
  fail("manifest.models must be a non-empty array");
}

const ids = new Set();
const domains = new Set();
const tiers = new Set();
let laptopCoreCount = 0;
let pytorchPrimaryCount = 0;

for (const model of manifest.models || []) {
  if (!model || typeof model !== "object") {
    fail("model entry must be an object");
    continue;
  }
  for (const field of ["id", "tier", "domain", "architecture", "model", "source", "format", "primaryReference", "zgmlStatus", "benchmark", "why"]) {
    if (model[field] === undefined || model[field] === null || model[field] === "") {
      fail(`${model.id || "<unknown>"} missing ${field}`);
    }
  }
  if (ids.has(model.id)) fail(`duplicate model id: ${model.id}`);
  ids.add(model.id);
  domains.add(model.domain);
  tiers.add(model.tier);
  if (!requiredTiers.includes(model.tier)) fail(`${model.id} has unknown tier: ${model.tier}`);
  if (!allowedStatuses.has(model.zgmlStatus)) fail(`${model.id} has unknown zgmlStatus: ${model.zgmlStatus}`);
  if (model.primaryReference === "PyTorch") {
    pytorchPrimaryCount += 1;
  } else {
    fail(`${model.id} primaryReference must be PyTorch`);
  }
  if (model.tier === "laptop_core") laptopCoreCount += 1;
  if (!model.source.startsWith("https://")) fail(`${model.id} source must be an https URL`);
  if (!model.benchmark || typeof model.benchmark !== "object") {
    fail(`${model.id} benchmark must be an object`);
  } else {
    for (const field of ["metric", "gate"]) {
      if (!model.benchmark[field]) fail(`${model.id} benchmark missing ${field}`);
    }
  }
}

for (const domain of requiredDomains) {
  if (!domains.has(domain)) fail(`missing required domain: ${domain}`);
}
for (const tier of requiredTiers) {
  if (!tiers.has(tier)) fail(`missing model for tier: ${tier}`);
}
if (pytorchPrimaryCount !== (manifest.models || []).length) {
  fail(`all models must use PyTorch as the primary reference, got ${pytorchPrimaryCount}/${(manifest.models || []).length}`);
}
if (laptopCoreCount < requiredDomains.length) {
  fail(`suite should have at least ${requiredDomains.length} laptop_core models, got ${laptopCoreCount}`);
}

if (errors.length !== 0) {
  console.error(`laptop model suite failed with ${errors.length} issue(s):`);
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

const byDomain = [...domains].sort().map((domain) => {
  const rows = manifest.models.filter((model) => model.domain === domain);
  const statuses = rows.reduce((acc, model) => {
    acc[model.zgmlStatus] = (acc[model.zgmlStatus] || 0) + 1;
    return acc;
  }, {});
  return `${domain}=${rows.length}(${Object.entries(statuses).map(([status, count]) => `${status}:${count}`).join(",")})`;
});

console.log([
  "zgml laptop model suite ok:",
  `models=${manifest.models.length}`,
  `pytorch_primary=${pytorchPrimaryCount}/${manifest.models.length}`,
  `laptop_core=${laptopCoreCount}`,
  byDomain.join(" "),
].join(" "));
