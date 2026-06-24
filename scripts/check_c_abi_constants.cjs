"use strict";

const { readFileSync } = require("node:fs");
const { join, resolve } = require("node:path");

const root = resolve(__dirname, "..");

function read(relativePath) {
  return readFileSync(join(root, relativePath), "utf8");
}

function screamingSnake(name) {
  return name.toUpperCase();
}

function parseZigConstants(source, prefix, type) {
  const constants = new Map();
  const pattern = new RegExp(`const ${prefix}([a-z0-9_]+): ${type} = ([^;]+);`, "g");
  for (const match of source.matchAll(pattern)) {
    constants.set(screamingSnake(match[1]), parseValue(match[2].trim()));
  }
  return constants;
}

function parseHeaderConstants(source, prefix) {
  const constants = new Map();
  const pattern = new RegExp(`${prefix}([A-Z0-9_]+)\\s*=\\s*([^,\\n]+)`, "g");
  for (const match of source.matchAll(pattern)) {
    const rawValue = match[2].trim();
    if (rawValue.startsWith(prefix)) continue;
    constants.set(match[1], parseValue(rawValue));
  }
  return constants;
}

function parseValue(raw) {
  const shift = raw.match(/^1(?:u|ul|ull)?\s*<<\s*(\d+)$/i);
  if (shift) return 2n ** BigInt(shift[1]);
  const integer = raw.match(/^\d+$/);
  if (integer) return BigInt(raw);
  throw new Error(`unsupported ABI constant expression: ${raw}`);
}

function compareConstants(errors, label, nativeConstants, headerConstants) {
  for (const [name, value] of nativeConstants) {
    if (!headerConstants.has(name)) {
      errors.push(`${label} missing public header constant ${name}`);
      continue;
    }
    const headerValue = headerConstants.get(name);
    if (headerValue !== value) {
      errors.push(`${label} ${name} drifted: Zig=${value.toString()} header=${headerValue.toString()}`);
    }
  }
}

function main() {
  const zig = read("src/c_api.zig");
  const header = read("include/zgml.h");
  const errors = [];

  compareConstants(
    errors,
    "feature flags",
    parseZigConstants(zig, "feature_", "u64"),
    parseHeaderConstants(header, "ZGML_FEATURE_"),
  );
  compareConstants(
    errors,
    "module op ids",
    parseZigConstants(zig, "module_op_", "u32"),
    parseHeaderConstants(header, "ZGML_MODULE_OP_"),
  );
  compareConstants(
    errors,
    "module activation ids",
    parseZigConstants(zig, "module_activation_", "u32"),
    parseHeaderConstants(header, "ZGML_MODULE_ACTIVATION_"),
  );

  if (errors.length > 0) {
    console.error("C ABI constant checks failed.");
    for (const error of errors) console.error(`- ${error}`);
    process.exit(1);
  }
  console.log("zgml C ABI constants ok");
}

if (require.main === module) main();
