"use strict";

const nativeApiArrayNames = Object.freeze([
  "requiredNativePackageSpinePrefixExports",
  "requiredNativePackageSpineSuffixExports",
  "requiredNativeTensorFactoryExports",
  "requiredNativeTensorViewExports",
  "requiredNativeTensorMathExports",
  "requiredNativeGradModeExports",
  "requiredNativeFrontendNamespaceExports",
  "requiredNativeExecutableRuntimeExports",
  "requiredNativeLlmRuntimeExports",
  "requiredNativeModelSourceExports",
  "requiredNativeInspectionExports",
  "requiredNativeApiExports",
  "requiredNativeApiSentinelExports",
]);

const nativeApiDomainArrayNames = Object.freeze([
  "requiredNativeTensorFactoryExports",
  "requiredNativeTensorViewExports",
  "requiredNativeTensorMathExports",
  "requiredNativeGradModeExports",
  "requiredNativeFrontendNamespaceExports",
  "requiredNativeExecutableRuntimeExports",
  "requiredNativeLlmRuntimeExports",
  "requiredNativeModelSourceExports",
  "requiredNativeInspectionExports",
]);

function stringArrayContract(source, constName) {
  const seen = new Set();
  function expand(name) {
    if (seen.has(name)) {
      throw new Error(`cycle in native API contract arrays at ${name}`);
    }
    seen.add(name);
    const match = source.match(new RegExp(`export const ${name} = Object\\.freeze\\(\\[([\\s\\S]*?)\\] as const\\);`));
    if (!match) {
      throw new Error(`could not find ${name} in native API contract source`);
    }
    const entries = [];
    for (const entry of match[1].matchAll(/"([^"]+)"|\.\.\.([A-Za-z_$][\w$]*)/g)) {
      if (entry[1]) entries.push(entry[1]);
      if (entry[2]) entries.push(...expand(entry[2]));
    }
    seen.delete(name);
    return entries;
  }
  return expand(constName);
}

function nativeApiContractArrays(source) {
  return Object.freeze(Object.fromEntries(
    nativeApiArrayNames.map((name) => [name, Object.freeze(stringArrayContract(source, name))]),
  ));
}

function nativeApiContractManifestPolicy(source) {
  const match = source.match(/tsRuntimeManifestPolicy\("([^"]+)", "([^"]+)"\)/);
  if (!match) {
    throw new Error("could not find native API contract runtime manifest policy");
  }
  return Object.freeze({
    policyOwner: match[1],
    runtimePath: match[2],
  });
}

function nativeApiDomainExports(arrays) {
  return Object.freeze(nativeApiDomainArrayNames.flatMap((name) => arrays[name] ?? []));
}

function nativePackageSpineExports(arrays, runtimeOverrides) {
  return Object.freeze([
    ...(arrays.requiredNativePackageSpinePrefixExports ?? []),
    ...runtimeOverrides,
    ...(arrays.requiredNativePackageSpineSuffixExports ?? []),
  ]);
}

function frontendNamespaceExports(source) {
  return Object.freeze([...source.matchAll(/^export \* as ([A-Za-z_$][\w$]*) from /gm)].map((entry) => entry[1]));
}

function nativeRootOverrideExports(nativeExports, frontendSource, rootAliases = []) {
  const frontendNamespaces = new Set(frontendNamespaceExports(frontendSource));
  const requiredRootAliases = new Set(rootAliases);
  return Object.freeze(nativeExports.filter((name) => frontendNamespaces.has(name) || requiredRootAliases.has(name)));
}

function normalizeNumericExpression(source, constants = {}) {
  const normalized = String(source)
    .replace(/\/\/.*$/gm, "")
    .replace(/\bn\b/g, "")
    .replace(/([0-9])n\b/g, "$1")
    .replace(/\b0x([0-9a-fA-F]+)\b/g, (_, value) => String(Number.parseInt(value, 16)))
    .trim();
  if (/^-?[0-9]+$/.test(normalized)) return Number(normalized);

  const shift = normalized.match(/^(.+?)\s*<<\s*(.+)$/);
  if (shift) {
    return normalizeNumericExpression(shift[1], constants) * (2 ** normalizeNumericExpression(shift[2], constants));
  }

  const parts = normalized.split("|").map((part) => part.trim()).filter(Boolean);
  if (parts.length > 1) {
    return parts.reduce((value, part) => value + normalizeNumericExpression(part, constants), 0);
  }

  if (Object.hasOwn(constants, normalized)) return constants[normalized];
  throw new Error(`unsupported numeric contract expression: ${source}`);
}

function tsNumericObjectContract(source, constName, seen = new Set()) {
  if (seen.has(constName)) {
    throw new Error(`cycle in TS numeric contract objects at ${constName}`);
  }
  seen.add(constName);
  const match = source.match(new RegExp(`export const ${constName}(?::[^=]+)? = Object\\.freeze\\(\\{([\\s\\S]*?)\\}\\);`));
  if (!match) {
    throw new Error(`could not find TS numeric contract object ${constName}`);
  }

  const out = {};
  for (const line of match[1].split("\n")) {
    const spread = line.match(/\.\.\.([A-Za-z_$][\w$]*)/);
    if (spread) {
      Object.assign(out, tsNumericObjectContract(source, spread[1], seen));
      continue;
    }
    const entry = line.match(/^\s*(?:"([^"]+)"|'([^']+)'|([A-Za-z_$][\w$]*))\s*:\s*([^,]+),?\s*$/);
    if (!entry) continue;
    const key = entry[1] ?? entry[2] ?? entry[3];
    out[key] = normalizeNumericExpression(entry[4], out);
  }
  seen.delete(constName);
  return Object.freeze(out);
}

function tsNumericConstContract(source, constName) {
  const match = source.match(new RegExp(`export const ${constName}\\s*=\\s*([^;]+);`));
  if (!match) {
    throw new Error(`could not find TS numeric contract const ${constName}`);
  }
  return normalizeNumericExpression(match[1]);
}

function zigNumericConstContract(source) {
  const out = {};
  const constPattern = /^const\s+([A-Za-z_$][\w$]*)\s*:\s*(?:u\d+|usize|c_int)\s*=\s*([^;]+);/gm;
  for (const match of source.matchAll(constPattern)) {
    out[match[1]] = normalizeNumericExpression(match[2], out);
  }
  return Object.freeze(out);
}

function camelToSnake(name) {
  return String(name)
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .replace(/-/g, "_")
    .toLowerCase();
}

function tsNativeDescriptorContracts(source) {
  return Object.freeze({
    expectedRuntimeAbiVersion: tsNumericConstContract(source, "expectedRuntimeAbiVersion"),
    moduleActivationIds: tsNumericObjectContract(source, "moduleActivationIds"),
    moduleOpIds: tsNumericObjectContract(source, "moduleOpIds"),
    moduleFlags: tsNumericObjectContract(source, "moduleFlags"),
    modelKinds: tsNumericObjectContract(source, "modelKinds"),
    backendIds: tsNumericObjectContract(source, "backendIds"),
    bufferStorageIds: tsNumericObjectContract(source, "bufferStorageIds"),
    resourceAccessIds: tsNumericObjectContract(source, "resourceAccessIds"),
    programBufferKinds: tsNumericObjectContract(source, "programBufferKinds"),
    abiStructKinds: tsNumericObjectContract(source, "abiStructKinds"),
    runtimeFeatureBits: tsNumericObjectContract(source, "runtimeFeatureBits"),
  });
}

module.exports = {
  camelToSnake,
  frontendNamespaceExports,
  nativeApiArrayNames,
  nativeApiContractArrays,
  nativeApiContractManifestPolicy,
  nativeApiDomainArrayNames,
  nativeApiDomainExports,
  nativePackageSpineExports,
  nativeRootOverrideExports,
  stringArrayContract,
  tsNativeDescriptorContracts,
  tsNumericConstContract,
  tsNumericObjectContract,
  zigNumericConstContract,
};
