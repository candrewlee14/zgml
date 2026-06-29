"use strict";

import { frontendManifest } from "../frontend_manifest.js";

export function tsProductManifestPolicy<const PolicyOwner extends string>(policyOwner: PolicyOwner) {
  return Object.freeze({
    source: "ts",
    policyOwner,
    productSourceOfTruth: frontendManifest.productSourceOfTruth,
    productSemanticsOwner: frontendManifest.productSemanticsOwner,
    nativeProductPolicy: frontendManifest.nativeProductPolicy,
    handwrittenFrontendMirrors: frontendManifest.handwrittenFrontendMirrors,
  });
}

export function tsRuntimeManifestPolicy<const PolicyOwner extends string, const RuntimePath extends string>(
  policyOwner: PolicyOwner,
  runtimePath: RuntimePath,
) {
  return Object.freeze({
    source: "ts",
    policyOwner,
    runtimePath,
  });
}
