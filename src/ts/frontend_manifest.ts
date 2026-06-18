"use strict";

export const frontendManifest = Object.freeze({
  kind: "zgml-frontend",
  source: "ts",
  productLanguage: "typescript",
  productSource: "src/ts/**",
  productSourceOfTruth: "ts-only",
  productSemanticsOwner: "src/ts/**",
  policyOwner: "src/ts",
  packageFanout: "tsdown",
  runtimePath: "Program -> Session -> StepParams",
  nativeRole: "runtime-kernel-abi-substrate",
  nativeAlignment: "contract-tested-substrate",
  nativeProductPolicy: "forbidden",
  adapterRole: "ffi-loader",
  frontendSync: "none",
  handwrittenFrontendMirrors: false,
  nativeContractBoundary: "Program/Session/ABI contracts",
});
