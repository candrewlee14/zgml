"use strict";

const requiredStringFields = Object.freeze([
  "productLanguage",
  "productSource",
  "productSourceOfTruth",
  "productSemanticsOwner",
  "packageFanout",
  "runtimePath",
  "nativeRole",
  "nativeAlignment",
  "nativeProductPolicy",
  "adapterRole",
  "frontendSync",
  "nativeContractBoundary",
  "eagerHotPathCore",
  "inferenceHotPathCore",
  "trainingHotPathCore",
  "unsupportedHotPathPolicy",
]);

const requiredBooleanFields = Object.freeze([
  "handwrittenFrontendMirrors",
]);

function stringField(source, name) {
  const match = source.match(new RegExp(`${name}:\\s*"([^"]+)"`));
  if (!match) {
    throw new Error(`src/ts/frontend_manifest.ts is missing string field ${name}`);
  }
  return match[1];
}

function booleanField(source, name) {
  const match = source.match(new RegExp(`${name}:\\s*(true|false)`));
  if (!match) {
    throw new Error(`src/ts/frontend_manifest.ts is missing boolean field ${name}`);
  }
  return match[1] === "true";
}

function frontendManifestPolicyFromSource(source) {
  return Object.freeze({
    ...Object.fromEntries(requiredStringFields.map((name) => [name, stringField(source, name)])),
    ...Object.fromEntries(requiredBooleanFields.map((name) => [name, booleanField(source, name)])),
  });
}

module.exports = {
  booleanField,
  frontendManifestPolicyFromSource,
  requiredBooleanFields,
  requiredStringFields,
  stringField,
};
