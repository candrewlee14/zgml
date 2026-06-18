"use strict";

const fs = require("node:fs");
const path = require("node:path");
const {
  frontendManifestPolicyFromSource,
} = require("./frontend_manifest_policy.cjs");

const root = path.resolve(__dirname, "..");
const manifestPath = path.join(root, "src", "ts", "frontend_manifest.ts");
const outputPath = path.join(root, "src", "native_substrate_manifest.zig");
const check = process.argv.includes("--check");

function zigString(value) {
  return JSON.stringify(value);
}

function renderManifest(source) {
  const manifest = frontendManifestPolicyFromSource(source);

  return `// Generated from src/ts/frontend_manifest.ts by scripts/generate_native_substrate_manifest.cjs.
// Do not edit by hand; TS owns product policy, Zig imports this native substrate view.

pub const native_substrate_manifest = .{
    .kind = "zgml-native-substrate",
    .role = ${zigString(manifest.nativeRole)},
    .product_language = ${zigString(manifest.productLanguage)},
    .product_source = ${zigString(manifest.productSource)},
    .product_source_of_truth = ${zigString(manifest.productSourceOfTruth)},
    .product_semantics_owner = ${zigString(manifest.productSemanticsOwner)},
    .package_fanout = ${zigString(manifest.packageFanout)},
    .frontend_sync = ${zigString(manifest.frontendSync)},
    .handwritten_frontend_mirrors = ${manifest.handwrittenFrontendMirrors},
    .native_alignment = ${zigString(manifest.nativeAlignment)},
    .native_product_policy = ${zigString(manifest.nativeProductPolicy)},
    .native_contract_boundary = ${zigString(manifest.nativeContractBoundary)},
};
`;
}

const source = fs.readFileSync(manifestPath, "utf8");
const output = renderManifest(source);

if (check) {
  const current = fs.existsSync(outputPath) ? fs.readFileSync(outputPath, "utf8") : "";
  if (current !== output) {
    console.error("src/native_substrate_manifest.zig is stale; run node scripts/generate_native_substrate_manifest.cjs");
    process.exitCode = 1;
  }
} else {
  fs.writeFileSync(outputPath, output);
}
