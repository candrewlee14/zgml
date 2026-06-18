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

module.exports = {
  frontendNamespaceExports,
  nativeApiArrayNames,
  nativeApiContractArrays,
  nativeApiContractManifestPolicy,
  nativeApiDomainArrayNames,
  nativeApiDomainExports,
  nativePackageSpineExports,
  nativeRootOverrideExports,
  stringArrayContract,
};
