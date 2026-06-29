import { createRequire } from "node:module";
import { defineConfig } from "tsdown/config";

const require = createRequire(import.meta.url);
const {
  nodeFfiRuntimeEntry,
  packageEntryGroups,
  packageEntryMap,
} = require("./scripts/package_metadata_policy.cjs");

const entryMap = packageEntryMap();
const entryGroups = packageEntryGroups(entryMap);

const baseConfig = {
  format: "cjs",
  outDir: "dist",
  sourcemap: true,
  tsconfig: "tsconfig.package.json",
  platform: "node",
  deps: {
    neverBundle: ["bun:ffi"],
  },
  failOnWarn: true,
};

const packageConfigs = entryGroups.map((group, index) => {
  return {
    ...baseConfig,
    entry: group.entry,
    dts: false,
    clean: index === 0,
  };
});

const nodeFfiRuntimeConfig = {
  ...baseConfig,
  entry: {
    "adapters/node_ffi_runtime": nodeFfiRuntimeEntry,
  },
  dts: false,
  clean: false,
};

export default defineConfig([
  ...packageConfigs,
  nodeFfiRuntimeConfig,
]);
