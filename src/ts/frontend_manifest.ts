"use strict";

export const frontendManifest = Object.freeze({
  kind: "zgml-frontend",
  source: "ts",
  productLanguage: "typescript",
  productSource: "src/ts/**",
  productSourceOfTruth: "ts-api-zig-core",
  productSemanticsOwner: "src/ts/** + src/**/*.zig",
  policyOwner: "src/ts",
  packageFanout: "tsdown",
  runtimePath: "Program -> Session -> StepParams",
  nativeRole: "core-kernel-runtime",
  nativeAlignment: "zig-core-contract-tested",
  nativeProductPolicy: "required-core",
  adapterRole: "ffi-loader",
  frontendSync: "none",
  handwrittenFrontendMirrors: false,
  nativeContractBoundary: "JS/TS API -> Zig C ABI -> Program/Session kernels",
});
