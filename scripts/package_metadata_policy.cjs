"use strict";

const { readFileSync, readdirSync, statSync } = require("node:fs");
const { join, relative, resolve, sep } = require("node:path");
const {
  frontendManifestPolicyFromSource,
} = require("./frontend_manifest_policy.cjs");

const root = process.cwd();
const sourceRoot = join(root, "src", "ts");
const frontendManifestPolicy = frontendManifestPolicyFromSource(readFileSync(join(root, "src", "ts", "frontend_manifest.ts"), "utf8"));
const packageMain = "dist/node.cjs";
const packageTypes = "dist/public_api.d.cts";
const packageRootRuntime = `./${packageMain}`;
const packageRootTypes = `./${packageTypes}`;

const excludedTopLevelSubpaths = new Set([
  "browser",
  "bun",
  "bun_native",
  "frontend_manifest",
  "index",
  "node",
  "public_api",
  "public_surface",
  "shared_frontend",
]);

const publicAdapterSubpaths = Object.freeze(["native", "node", "bun"]);
const publicWildcardDirectories = Object.freeze(["core", "runtime", "nn", "train"]);
const forbiddenPublicExportPrefixes = Object.freeze(["./internal"]);
const nodeFfiRuntimeEntry = "src/ts/adapters/node_ffi_runtime.ts";
const excludedPackageEntries = Object.freeze([
  nodeFfiRuntimeEntry,
  "src/ts/public_surface.ts",
]);
const noDeclarationEntries = Object.freeze([
  nodeFfiRuntimeEntry,
]);
const excludedDeclarationGlobs = Object.freeze([
  ...noDeclarationEntries.map((entry) => `!${entry}`),
  "!src/ts/smokes/**/*.ts",
]);
const dtsEntryGlobs = Object.freeze([
  "src/ts/**/*.ts",
  ...excludedDeclarationGlobs,
]);
const forbiddenPackagePrefixes = Object.freeze([
  "js/generated/",
]);
const forbiddenPackageFiles = Object.freeze([
  "index.d.ts",
  "node.d.ts",
  "bun.d.ts",
  "dist/adapters/node_ffi_runtime.d.cts",
  "js/dist_package_smoke.cjs",
  "js/frontend_package_smoke.cjs",
  "js/bun.ts",
  "js/package_smoke.cjs",
  "js/package_smoke_bun.ts",
  "js/native_bridge_contract_smoke.cjs",
  "js/ts_source_smoke.cjs",
  "js/shared_frontend.cjs",
  "js/shared_session_facade.cjs",
  "js/shared_kernel_plan.cjs",
  "js/shared_package_smoke.cjs",
  "js/node.cjs",
]);
const publicPackageScripts = Object.freeze([
  "smoke",
  "smoke:node",
  "smoke:bun",
  "smoke:training",
  "smoke:training:bun",
  "smoke:adapters",
  "test",
  "test:adapters",
]);
const expectedPackageFiles = Object.freeze([
  "README.md",
  "LICENSE",
  "dist/",
  "include/",
  "docs/c-api.md",
  "tsconfig.types.json",
  "tsconfig.dist.json",
  "tsconfig.package.json",
  "examples/types/public-api-smoke.ts",
  "examples/types/package-spine-smoke.ts",
  "examples/types/readme-quickstart-smoke.ts",
  "examples/types/dist-package-smoke.cts",
  "examples/types/shape-safety-smoke.ts",
  "examples/types/pytorch-training-smoke.ts",
  "examples/types/browser-frontend-smoke.ts",
  "examples/types/bun-training-smoke.ts",
  "examples/types/program-session-smoke.ts",
  "examples/quickstart/zgml-first.cjs",
  "examples/node_training/manual_loop.cjs",
  "examples/node_training/quickstart.cjs",
  "examples/node_training/train_linear.cjs",
  "examples/node_training/train_conv2d.cjs",
  "examples/node_training/train_mlp.cjs",
  "examples/node_training/train_token_classifier.cjs",
  "examples/node_training/train_classifier.cjs",
  "examples/node_training/torch_quickstart.cjs",
  "examples/node_program_session/run_descriptor_contract.cjs",
  "examples/node_program_session/run_linear.cjs",
  "examples/bun_training/manual_loop.ts",
  "examples/bun_training/train_linear.ts",
  "examples/bun_training/train_mlp.ts",
  "examples/bun_training/train_classifier.ts",
  "examples/bun_program_session/run_descriptor_contract.ts",
  "examples/bun_program_session/run_linear.ts",
  "scripts/frontend_manifest_policy.cjs",
  "scripts/check_frontend_capability_matrix.cjs",
  "scripts/generate_package_declarations.cjs",
  "scripts/package_metadata_policy.cjs",
  "tsdown.config.mjs",
  "build.zig",
  "build.zig.zon",
  "src/",
  "vendor/wgpu-native/",
]);
const requiredExtraBoundaryArtifacts = Object.freeze([
  "tsconfig.package.json",
  "src/ts/adapters/node_ffi_runtime.ts",
  "src/native_substrate_manifest.zig",
]);
const requiredTsdownConfigContracts = Object.freeze([
  "import { createRequire } from \"node:module\";",
  "require(\"./scripts/package_metadata_policy.cjs\")",
  "nodeFfiRuntimeEntry,",
  "packageEntryMap,",
  "packageEntryGroups,",
  "const entryMap = packageEntryMap();",
  "const entryGroups = packageEntryGroups(entryMap);",
  "entryGroups.map((group, index) => {",
  "entry: group.entry",
  "\"adapters/node_ffi_runtime\": nodeFfiRuntimeEntry",
  "dts: false",
  "clean: index === 0",
  "clean: false",
]);
const requiredDeclarationGeneratorContracts = Object.freeze([
  "require.resolve(\"typescript/bin/tsc\")",
  "--emitDeclarationOnly",
  "--declarationMap",
  "false",
  "replace(/\\.ts$/, \".d.cts\")",
  "rewriteDeclarationForCjsPackage",
  ".cjs",
  "entry.startsWith(\"src/ts/smokes/\")",
  "noDeclarationEntries",
]);
const forbiddenTsdownConfigRefs = Object.freeze([
  "function packageEntries(dir)",
  "excludedPackageEntries",
  "js/generated",
  "index.js",
  "node.js",
  "bun.js",
  ".zig-cache/ts-generated-check",
]);
const productLanguage = frontendManifestPolicy.productLanguage;
const productSource = frontendManifestPolicy.productSource;
const productSourceOfTruth = frontendManifestPolicy.productSourceOfTruth;
const productSemanticsOwner = frontendManifestPolicy.productSemanticsOwner;
const packageFanout = frontendManifestPolicy.packageFanout;
const frontendSync = frontendManifestPolicy.frontendSync;
const handwrittenFrontendMirrors = String(frontendManifestPolicy.handwrittenFrontendMirrors);
const nativeContractBoundary = frontendManifestPolicy.nativeContractBoundary;
const nativeRuntimeRole = frontendManifestPolicy.nativeRole;
const nativeAlignment = frontendManifestPolicy.nativeAlignment;
const nativeProductPolicy = frontendManifestPolicy.nativeProductPolicy;
const nativeHelperRole = "native-helper-substrate";
const packageMetadataPolicyManifest = Object.freeze({
  kind: "zgml-package-metadata-policy",
  source: "ts-package-policy",
  policyOwner: "scripts/package_metadata_policy.cjs",
  productLanguage,
  productSource,
  productSourceOfTruth,
  productSemanticsOwner,
  packageFanout,
  frontendSync,
  nativeProductPolicy,
  handwrittenFrontendMirrors: false,
  packageRuntimeRoot: packageMain,
  packageTypesRoot: packageTypes,
  generatedArtifactRoot: "dist/**",
});
const nativeHelperDomains = Object.freeze([
  ["src/nn.zig", "nn", "//! Native neural-network helpers that decompose into primitive tensor ops."],
  ["src/loss.zig", "loss", "//! Native loss helpers over Tensor primitives."],
  ["src/optim.zig", "optim", "//! Native imperative optimizer helpers over trainable Tensor parameters."],
  ["src/train.zig", "train", "//! Native training-loop helpers over ComputeGraph, Tensor losses, and optimizers."],
]);

function nativeHelperSourceContract(domain, leadingComment) {
  return Object.freeze([
    leadingComment,
    "pub const native_helper_manifest = .{",
    `.domain = "${domain}"`,
    ".product_frontend = false",
    `.product_owner = "${productSource}"`,
    `.product_policy = "${nativeProductPolicy}"`,
    `.role = "${nativeHelperRole}"`,
    `.alignment_boundary = "${nativeContractBoundary}"`,
  ]);
}

const nativeHelperSourceContracts = Object.freeze(Object.fromEntries(
  nativeHelperDomains.map(([path, domain, leadingComment]) => [
    path,
    nativeHelperSourceContract(domain, leadingComment),
  ]),
));
const sourceContracts = Object.freeze({
  "src/ts/frontend_manifest.ts": Object.freeze([
    `productLanguage: "${productLanguage}"`,
    `productSource: "${productSource}"`,
    `productSourceOfTruth: "${productSourceOfTruth}"`,
    `productSemanticsOwner: "${productSemanticsOwner}"`,
    `packageFanout: "${packageFanout}"`,
    `nativeRole: "${nativeRuntimeRole}"`,
    `nativeAlignment: "${nativeAlignment}"`,
    `nativeProductPolicy: "${nativeProductPolicy}"`,
    `frontendSync: "${frontendSync}"`,
    `handwrittenFrontendMirrors: ${handwrittenFrontendMirrors}`,
    `nativeContractBoundary: "${nativeContractBoundary}"`,
  ]),
  "src/ts/index.ts": Object.freeze([
    'export { frontendManifest } from "./frontend_manifest.js";',
  ]),
  "src/ts/public_surface.ts": Object.freeze([
    'import { tsProductManifestPolicy } from "./internal/product_manifest.js";',
    "export const firstContactRootNamespaces = Object.freeze([",
    "export const firstContactRootValues = Object.freeze([",
    "export const stableRootNamespaces = Object.freeze([",
    "export const advancedRootNamespaces = Object.freeze([",
    "export const legacyCompatibleRootNamespaces = Object.freeze([",
    "export const rootSurfacePolicy = Object.freeze({",
    "export const publicSurfaceManifest = Object.freeze({",
    'kind: "zgml-public-surface"',
    '...tsProductManifestPolicy("src/ts/public_surface.ts")',
    'rootEntry: "src/ts/index.ts"',
    "internalPackagePolicyOnly: true",
    "firstContactSurfaceIsSmall: true",
    'firstContactRuntimeHandle: "compile.compileForInference"',
    "classificationCoversRootNamespaceExports: true",
    "firstContactIsSubsetOfStableSurface: true",
    "newProductSurfaceGoesThroughStableNamespaces: true",
  ]),
  "scripts/check_frontend_capability_matrix.cjs": Object.freeze([
    "docs/frontend-capability-matrix.md",
    "src/ts/public_surface.ts",
    "firstContactRootNamespaces",
    "firstContactRootValues",
    "stableRootNamespaces",
    "allowedStatuses",
    "requiredCapabilities",
    "compile.compileForInference(...)",
    "frontend capability matrix ok:",
  ]),
  "src/ts/browser.ts": Object.freeze([
    'export * from "./index.js";',
    'import { tsProductManifestPolicy } from "./internal/product_manifest.js";',
    "export const browserManifest = Object.freeze({",
    'kind: "zgml-browser-frontend"',
    '...tsProductManifestPolicy("src/ts/browser.ts")',
    'frontendEntry: "src/ts/index.ts"',
    "nativeLoader: false",
    'adapterRole: "browser-safe-frontend"',
  ]),
  "src/ts/shared_frontend.ts": Object.freeze([
    'import { tsProductManifestPolicy } from "./internal/product_manifest.js";',
    "export const sharedFrontendManifest = Object.freeze({",
    '...tsProductManifestPolicy("src/ts/shared_frontend.ts")',
    "legacyCjsBridge: false",
  ]),
  "src/ts/internal/product_manifest.ts": Object.freeze([
    'import { frontendManifest } from "../frontend_manifest.js";',
    "export function tsProductManifestPolicy<const PolicyOwner extends string>(policyOwner: PolicyOwner)",
    "productSourceOfTruth: frontendManifest.productSourceOfTruth",
    "productSemanticsOwner: frontendManifest.productSemanticsOwner",
    "nativeProductPolicy: frontendManifest.nativeProductPolicy",
    "handwrittenFrontendMirrors: frontendManifest.handwrittenFrontendMirrors",
    "export function tsRuntimeManifestPolicy<const PolicyOwner extends string, const RuntimePath extends string>",
    "runtimePath: RuntimePath",
  ]),
  "src/ts/public_api.ts": Object.freeze([
    'type FrontendManifestContract = typeof import("./frontend_manifest.js").frontendManifest;',
    "export type PublicApiContractManifest = Readonly<{",
    'kind: "zgml-public-api-contract"',
    'source: FrontendManifestContract["source"]',
    'policyOwner: "src/ts/public_api.ts"',
    'productSourceOfTruth: FrontendManifestContract["productSourceOfTruth"]',
    'productSemanticsOwner: FrontendManifestContract["productSemanticsOwner"]',
    'packageFanout: FrontendManifestContract["packageFanout"]',
    'packageTypesRoot: "dist/public_api.d.cts"',
    'runtimePath: FrontendManifestContract["runtimePath"]',
    'nativeProductPolicy: FrontendManifestContract["nativeProductPolicy"]',
    "declarationMirror: false",
    'export type FrontendManifest = typeof import("./frontend_manifest.js").frontendManifest;',
    "export declare const frontendManifest: FrontendManifest",
  ]),
  "src/main.zig": Object.freeze([
    'const native_manifest = @import("native_substrate_manifest.zig");',
    "pub const native_substrate_manifest = native_manifest.native_substrate_manifest;",
  ]),
  "src/native_substrate_manifest.zig": Object.freeze([
    "Generated from src/ts/frontend_manifest.ts by scripts/generate_native_substrate_manifest.cjs.",
    "TS owns product policy, Zig imports this native substrate view.",
    "pub const native_substrate_manifest = .{",
    '.kind = "zgml-native-substrate"',
    `.role = "${nativeRuntimeRole}"`,
    `.product_language = "${productLanguage}"`,
    `.product_source = "${productSource}"`,
    `.product_source_of_truth = "${productSourceOfTruth}"`,
    `.product_semantics_owner = "${productSemanticsOwner}"`,
    `.package_fanout = "${packageFanout}"`,
    `.frontend_sync = "${frontendSync}"`,
    `.handwritten_frontend_mirrors = ${handwrittenFrontendMirrors}`,
    `.native_alignment = "${nativeAlignment}"`,
    `.native_product_policy = "${nativeProductPolicy}"`,
    `.native_contract_boundary = "${nativeContractBoundary}"`,
  ]),
  ...nativeHelperSourceContracts,
});
const requiredBoundaryArtifacts = Object.freeze([
  ...requiredExtraBoundaryArtifacts,
  ...Object.keys(sourceContracts),
]);

function sortObjectByKey(value) {
  return Object.fromEntries(Object.entries(value).sort(([left], [right]) => left.localeCompare(right)));
}

function concreteDistExport(stem) {
  return {
    types: `./dist/${stem}.d.cts`,
    require: `./dist/${stem}.cjs`,
    default: `./dist/${stem}.cjs`,
  };
}

function topLevelPublicSubpaths() {
  return readdirSync(sourceRoot)
    .filter((name) => name.endsWith(".ts") && !name.endsWith(".d.ts"))
    .map((name) => name.replace(/\.ts$/, ""))
    .filter((stem) => !excludedTopLevelSubpaths.has(stem))
    .sort();
}

function packageEntryDepth(entry) {
  return entry.split("/").length;
}

function packageEntryOrder(left, right) {
  return packageEntryDepth(left) - packageEntryDepth(right) || left.localeCompare(right);
}

function tsSourceEntries(dir = sourceRoot) {
  return readdirSync(dir)
    .flatMap((name) => {
      const absolute = resolve(dir, name);
      const stat = statSync(absolute);
      if (stat.isDirectory()) return tsSourceEntries(absolute);
      if (!name.endsWith(".ts") || name.endsWith(".d.ts")) return [];
      return relative(root, absolute).split(sep).join("/");
    })
    .sort(packageEntryOrder);
}

function packageRuntimeEntries() {
  const excluded = new Set(excludedPackageEntries);
  return tsSourceEntries(sourceRoot).filter((entry) => !excluded.has(entry));
}

function packageEntryMap() {
  return Object.fromEntries(packageRuntimeEntries().map((entry) => [
    relative(sourceRoot, resolve(root, entry)).split(sep).join("/").replace(/\.ts$/, ""),
    entry,
  ]));
}

const platformRuntimeRootEntries = Object.freeze(new Set([
  "browser",
  "browser_frontend_runtime",
  "bun",
  "bun_native",
  "checkpoint",
  "data",
  "frontend_manifest",
  "runtime",
]));
const libraryRootEntries = Object.freeze(new Set([
  "inspection",
  "kernel_plan",
  "lazy",
  "loss",
  "model_source",
  "native_buffer",
  "nn",
  "optim",
  "runtime",
]));
const executableRootEntries = Object.freeze(new Set([
  "program",
  "program_device",
  "public_api",
  "session",
  "step_params",
  "tensor",
  "train",
  "runtime",
]));
const facadeRootEntries = Object.freeze(new Set([
  "compile",
  "index",
  "node",
  "simple",
  "shared_frontend",
]));

function packageEntryGroupEntry(entryMap, predicate) {
  return Object.fromEntries(Object.entries(entryMap).filter(([stem]) => predicate(stem)));
}

function packageEntryGroups(entryMap = packageEntryMap()) {
  const rootStem = (stem) => stem.split("/")[0];
  const groups = [
    ["platform-runtime", (stem) => platformRuntimeRootEntries.has(rootStem(stem))],
    ["library", (stem) => libraryRootEntries.has(rootStem(stem)) || stem.startsWith("nn/") || stem.startsWith("train/")],
    ["executable", (stem) => executableRootEntries.has(rootStem(stem)) || stem.startsWith("core/")],
    ["facades", (stem) => facadeRootEntries.has(rootStem(stem)) || stem.startsWith("internal/")],
    ["adapters", (stem) => stem.startsWith("adapters/")],
    ["smokes", (stem) => stem.startsWith("smokes/")],
  ];
  const seen = new Set();
  const planned = groups.map(([name, predicate]) => {
    const entry = packageEntryGroupEntry(entryMap, (stem) => {
      if (!predicate(stem) || seen.has(stem)) return false;
      seen.add(stem);
      return true;
    });
    return { name, entry };
  }).filter((group) => Object.keys(group.entry).length > 0);
  const missed = Object.keys(entryMap).filter((stem) => !seen.has(stem));
  if (missed.length !== 0) {
    throw new Error(`packageEntryGroups missed package entries: ${missed.join(", ")}`);
  }
  return planned;
}

function distStemForTsEntry(entry) {
  return `dist/${relative(sourceRoot, resolve(root, entry)).split(sep).join("/").replace(/\.ts$/, "")}`;
}

function requiredTsOwnedArtifacts() {
  const noDeclarations = new Set(noDeclarationEntries);
  const excluded = new Set(excludedPackageEntries);
  return tsSourceEntries(sourceRoot).filter((entry) => !excluded.has(entry)).flatMap((entry) => {
    const stem = distStemForTsEntry(entry);
    const artifacts = [`${stem}.cjs`];
    if (!entry.startsWith("src/ts/smokes/") && !noDeclarations.has(entry)) {
      artifacts.push(`${stem}.d.cts`);
    }
    return artifacts;
  });
}

function expectedPackageExports() {
  const exportsMap = {
    ".": {
      types: packageRootTypes,
      require: packageRootRuntime,
      default: packageRootRuntime,
    },
    "./node": {
      types: packageRootTypes,
      require: packageRootRuntime,
      default: packageRootRuntime,
    },
    "./bun": {
      types: packageRootTypes,
      bun: "./dist/bun_native.cjs",
      require: "./dist/bun.cjs",
      import: "./dist/bun.cjs",
      default: "./dist/bun.cjs",
    },
    "./frontend": concreteDistExport("index"),
    "./browser": concreteDistExport("browser"),
  };

  for (const stem of topLevelPublicSubpaths()) {
    exportsMap[`./${stem}`] = concreteDistExport(stem);
  }
  for (const stem of publicAdapterSubpaths) {
    exportsMap[`./adapters/${stem}`] = concreteDistExport(`adapters/${stem}`);
  }
  for (const stem of publicWildcardDirectories) {
    exportsMap[`./${stem}/*`] = concreteDistExport(`${stem}/*`);
  }

  return sortObjectByKey(exportsMap);
}

function expectedPackageMetadata() {
  return {
    main: packageMain,
    types: packageTypes,
    exports: expectedPackageExports(),
    files: [...expectedPackageFiles],
  };
}

function packageMetadataMatches(packageJson, expected = expectedPackageMetadata()) {
  return packageJson.main === expected.main &&
    packageJson.types === expected.types &&
    JSON.stringify(packageJson.exports ?? {}) === JSON.stringify(expected.exports) &&
    JSON.stringify(packageJson.files ?? []) === JSON.stringify(expected.files);
}

module.exports = {
  concreteDistExport,
  distStemForTsEntry,
  dtsEntryGlobs,
  excludedPackageEntries,
  expectedPackageFiles,
  expectedPackageExports,
  expectedPackageMetadata,
  packageMetadataMatches,
  excludedDeclarationGlobs,
  forbiddenPackageFiles,
  forbiddenPackagePrefixes,
  forbiddenPublicExportPrefixes,
  noDeclarationEntries,
  nodeFfiRuntimeEntry,
  packageMetadataPolicyManifest,
  packageMain,
  packageEntryDepth,
  packageEntryGroups,
  packageEntryMap,
  packageEntryOrder,
  packageRootRuntime,
  packageRootTypes,
  packageRuntimeEntries,
  packageTypes,
  nativeContractBoundary,
  nativeHelperDomains,
  nativeHelperRole,
  nativeHelperSourceContract,
  nativeHelperSourceContracts,
  nativeProductPolicy,
  nativeRuntimeRole,
  packageFanout,
  productLanguage,
  productSource,
  productSourceOfTruth,
  publicAdapterSubpaths,
  publicPackageScripts,
  publicWildcardDirectories,
  requiredTsdownConfigContracts,
  requiredDeclarationGeneratorContracts,
  forbiddenTsdownConfigRefs,
  requiredTsOwnedArtifacts,
  requiredExtraBoundaryArtifacts,
  requiredBoundaryArtifacts,
  sourceContracts,
  tsSourceEntries,
  topLevelPublicSubpaths,
};
