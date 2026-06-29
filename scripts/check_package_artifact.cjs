"use strict";

const { execFileSync } = require("node:child_process");
const { readFileSync } = require("node:fs");
const {
  expectedPackageMetadata,
  forbiddenPackageFiles,
  forbiddenPackagePrefixes,
  forbiddenPublicExportPrefixes,
  packageMetadataPolicyManifest,
  packageMetadataMatches,
  publicPackageScripts,
  requiredTsOwnedArtifacts,
  requiredBoundaryArtifacts,
  sourceContracts,
} = require("./package_metadata_policy.cjs");

const output = execFileSync("npm", ["pack", "--dry-run", "--json"], {
  encoding: "utf8",
  stdio: ["ignore", "pipe", "inherit"],
});
const [pack] = JSON.parse(output);
const files = pack.files.map((file) => file.path).sort();
const fileSet = new Set(files);
const packageJson = JSON.parse(readFileSync("package.json", "utf8"));
const expectedMetadata = expectedPackageMetadata();

const errors = [];
function publicScriptClosure() {
  const scripts = packageJson.scripts ?? {};
  const pending = [...publicPackageScripts];
  const seen = new Set();
  while (pending.length > 0) {
    const name = pending.shift();
    if (seen.has(name)) continue;
    seen.add(name);
    const command = scripts[name] ?? "";
    for (const match of command.matchAll(/\bnpm\s+run\s+([A-Za-z0-9:_-]+)/g)) {
      pending.push(match[1]);
    }
  }
  return [...seen].sort();
}

function scriptLocalFileRefs(command) {
  const refs = new Set();
  for (const match of command.matchAll(/\bnode\s+((?:scripts|dist|examples)\/[^\s&|;]+)/g)) refs.add(match[1]);
  for (const match of command.matchAll(/\bbun\s+((?:\.\/)?(?:dist|examples)\/[^\s&|;]+)/g)) refs.add(match[1].replace(/^\.\//, ""));
  for (const match of command.matchAll(/\btsc\s+-p\s+([^\s&|;]+)/g)) refs.add(match[1]);
  return [...refs].map((ref) => ref.replace(/^["']|["']$/g, ""));
}

const packageFiles = Array.isArray(packageJson.files) ? packageJson.files : [];
for (const entry of expectedMetadata.files) {
  if (!packageFiles.includes(entry)) {
    errors.push(`package.json files must include ${entry}; npm packaging is the TS product plus native substrate boundary`);
  }
}
for (const entry of packageFiles) {
  if (entry === "js/" || entry.startsWith("js/")) {
    errors.push(`package.json files must not include repo-only JS mirror trees: ${entry}`);
  }
}
const expectedExports = expectedMetadata.exports;
if (
  packageMetadataPolicyManifest.source !== "ts-package-policy" ||
  packageMetadataPolicyManifest.policyOwner !== "scripts/package_metadata_policy.cjs" ||
  packageMetadataPolicyManifest.productSourceOfTruth !== "ts-api-zig-core" ||
  packageMetadataPolicyManifest.packageFanout !== "tsdown" ||
  packageMetadataPolicyManifest.frontendSync !== "none" ||
  packageMetadataPolicyManifest.packageRuntimeRoot !== expectedMetadata.main ||
  packageMetadataPolicyManifest.packageTypesRoot !== expectedMetadata.types ||
  packageMetadataPolicyManifest.generatedArtifactRoot !== "dist/**"
) {
  errors.push("package metadata policy manifest must prove package artifacts are TS-authored tsdown fan-out");
}
if (!packageMetadataMatches(packageJson, expectedMetadata)) {
  errors.push("package.json metadata must match scripts/apply_package_metadata_policy.cjs --check; public npm subpaths are derived from src/ts package metadata policy");
}
const bunExport = packageJson.exports?.["./bun"] ?? {};
for (const condition of ["bun", "require", "import", "default"]) {
  if (bunExport[condition] !== "./dist/bun_native.cjs") {
    errors.push(`package.json ./bun ${condition} export must resolve to ./dist/bun_native.cjs so public Bun imports are native-backed and fail closed`);
  }
}
const expectedRootExport = expectedMetadata.exports["."];
const rootExport = packageJson.exports?.["."];
if (!rootExport || rootExport.types !== expectedRootExport.types || rootExport.require !== expectedRootExport.require || rootExport.default !== expectedRootExport.default) {
  errors.push(`package root export must resolve runtime imports to the tsdown-emitted ${expectedMetadata.main} artifact`);
}
for (const [subpath, exportSpec] of Object.entries(packageJson.exports ?? {})) {
  if (forbiddenPublicExportPrefixes.some((prefix) => subpath === prefix || subpath.startsWith(`${prefix}/`))) {
    errors.push(`package export ${subpath} must stay internal; product policy helpers are implementation artifacts, not public subpaths`);
  }
  if (subpath === "." || subpath.includes("*") || !exportSpec || typeof exportSpec !== "object") continue;
  const expected = expectedExports[subpath];
  if (!expected) {
    errors.push(`package export ${subpath} must be owned by scripts/package_metadata_policy.cjs`);
    continue;
  }
  for (const [field, value] of Object.entries(expected)) {
    if (exportSpec[field] !== value) {
      errors.push(`package export ${subpath} must resolve ${field} to ${value}; concrete package subpaths follow the TS-owned tsdown artifact convention`);
    }
  }
}
const exportedArtifactPaths = new Set();
for (const [subpath, exportSpec] of Object.entries(packageJson.exports ?? {})) {
  if (subpath.includes("*") || !exportSpec || typeof exportSpec !== "object") continue;
  for (const field of ["types", "require", "import", "default", "bun"]) {
    const target = exportSpec[field];
    if (typeof target === "string" && target.startsWith("./dist/")) {
      exportedArtifactPaths.add(target.slice(2));
    }
  }
}
for (const target of [...exportedArtifactPaths].sort()) {
  if (!fileSet.has(target)) {
    errors.push(`package export target is missing from npm artifact: ${target}`);
  }
}
for (const [subpath, exportSpec] of Object.entries(packageJson.exports ?? {})) {
  if (!subpath.includes("*") || !exportSpec || typeof exportSpec !== "object") continue;
  const expected = expectedExports[subpath];
  if (!expected) {
    errors.push(`package wildcard export ${subpath} must be owned by scripts/package_metadata_policy.cjs`);
    continue;
  }
  for (const [field, value] of Object.entries(expected)) {
    if (exportSpec[field] !== value) {
      errors.push(`package wildcard export ${subpath} must resolve ${field} to ${value}; wildcard package subpaths follow the TS-owned tsdown artifact convention`);
    }
  }
}
for (const path of requiredTsOwnedArtifacts()) {
  if (!fileSet.has(path)) errors.push(`missing TS-owned package artifact derived from src/ts/**: ${path}`);
}
for (const path of requiredBoundaryArtifacts) {
  if (!fileSet.has(path)) errors.push(`missing package artifact: ${path}`);
}
for (const tsconfigPath of ["tsconfig.types.json", "tsconfig.dist.json"]) {
  if (!fileSet.has(tsconfigPath)) continue;
  const config = JSON.parse(readFileSync(tsconfigPath, "utf8"));
  for (const path of Array.isArray(config.files) ? config.files : []) {
    if (!fileSet.has(path)) {
      errors.push(`shipped ${tsconfigPath} references missing package file: ${path}`);
    }
  }
}
for (const path of [
  "dist/runtime/llama_family_surface.cjs",
  "dist/runtime/llama_family_surface.cjs.map",
  "dist/runtime/llama_family_surface.d.cts",
]) {
  if (!fileSet.has(path)) {
    errors.push(`missing runtime-owned LLaMA family package artifact: ${path}`);
  }
}
for (const [path, needles] of Object.entries(sourceContracts)) {
  if (!fileSet.has(path)) {
    errors.push(`missing package source contract: ${path}`);
    continue;
  }
  const source = readFileSync(path, "utf8");
  for (const needle of needles) {
    if (!source.includes(needle)) {
      errors.push(`${path} must classify the package source boundary with: ${needle}`);
    }
  }
}
if (fileSet.has("index.js")) {
  errors.push("package must not ship index.js as a root runtime bridge; root runtime belongs to dist/node.cjs");
}
for (const path of files) {
  if (path.startsWith("js/")) {
    errors.push(`package must not ship repo-only JS mirror tree artifact: ${path}`);
  }
  if (/^src\/ts\/.*\.(?:js|cjs|mjs)$/.test(path)) {
    errors.push(`package must not ship hand-written JS under TS product source: ${path}`);
  }
  if (forbiddenPackagePrefixes.some((prefix) => path.startsWith(prefix))) {
    errors.push(`package must not ship transitional generated artifact: ${path}`);
  }
}
for (const path of forbiddenPackageFiles) {
  if (fileSet.has(path)) errors.push(`package must not ship internal test/runtime shim: ${path}`);
}
for (const path of [
  "dist/adapters/node_llama_family_surface.cjs",
  "dist/adapters/node_llama_family_surface.cjs.map",
  "dist/adapters/node_llama_family_surface.d.cts",
]) {
  if (fileSet.has(path)) {
    errors.push(`package must not ship obsolete Node-named LLaMA family artifact: ${path}`);
  }
}
for (const name of publicPackageScripts) {
  const command = packageJson.scripts?.[name] ?? "";
  if (/\bjs\//.test(command)) {
    errors.push(`package script ${name} must not depend on excluded repo-only js/ files: ${command}`);
  }
  if (/\b(test:ts-source|smoke:native-contract|smoke:dist|smoke:frontend|smoke:ts-source)\b/.test(command)) {
    errors.push(`package script ${name} must not call repo-only validation scripts: ${command}`);
  }
}
for (const name of publicScriptClosure()) {
  const command = packageJson.scripts?.[name] ?? "";
  for (const path of scriptLocalFileRefs(command)) {
    if (!fileSet.has(path)) {
      errors.push(`public package script ${name} references missing package file: ${path}`);
    }
  }
}

if (errors.length > 0) {
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

console.log(`zgml package artifact ok: ${files.length} files, dist-owned runtime`);
