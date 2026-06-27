"use strict";

const { readFileSync, writeFileSync } = require("node:fs");
const { join } = require("node:path");
const {
  expectedPackageMetadata,
  packageMetadataPolicyManifest,
  packageMetadataMatches,
} = require("./package_metadata_policy.cjs");

const root = process.cwd();

function formatPackageJson(packageJson) {
  return `${JSON.stringify(packageJson, null, 2)}\n`;
}

function main(argv = process.argv) {
  const check = argv.includes("--check");
  const packageJsonPath = join(root, "package.json");
  const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8"));
  const expected = expectedPackageMetadata();
  if (
    packageMetadataPolicyManifest.productSourceOfTruth !== "ts-api-zig-core" ||
    packageMetadataPolicyManifest.packageFanout !== "tsdown" ||
    packageMetadataPolicyManifest.packageRuntimeRoot !== expected.main ||
    packageMetadataPolicyManifest.packageTypesRoot !== expected.types ||
    packageMetadataPolicyManifest.generatedArtifactRoot !== "dist/**"
  ) {
    console.error("package metadata policy manifest must describe TS-only tsdown artifact fan-out");
    process.exit(1);
  }
  if (packageMetadataMatches(packageJson, expected)) {
    console.log("package metadata checked: TS-derived npm root fields, subpath map, and package file boundary");
    return;
  }
  packageJson.main = expected.main;
  packageJson.types = expected.types;
  packageJson.exports = expected.exports;
  packageJson.files = expected.files;
  if (check) {
    console.error("package.json main/types/exports/files must be regenerated from src/ts package metadata policy");
    process.exit(1);
  }
  writeFileSync(packageJsonPath, formatPackageJson(packageJson));
  console.log("package metadata generated: TS-derived npm root fields, subpath map, and package file boundary");
}

if (require.main === module) {
  main();
}

module.exports = {
  main,
};
