"use strict";

import { frontendManifest as canonicalFrontendManifest } from "../frontend_manifest.js";

type AnyRecord = Record<string, any>;

function normalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(normalize);
  if (value && typeof value === "object") {
    const out: AnyRecord = {};
    for (const key of Object.keys(value as AnyRecord)) out[key] = normalize((value as AnyRecord)[key]);
    return out;
  }
  return value;
}

export function expectSame(actual: unknown, expected: unknown, label: string): void {
  const actualJson = JSON.stringify(normalize(actual));
  const expectedJson = JSON.stringify(normalize(expected));
  if (actualJson !== expectedJson) {
    throw new Error(`${label} mismatch:\nactual   ${actualJson}\nexpected ${expectedJson}`);
  }
}

export function expectFrontendManifest(surface: AnyRecord, label: string): void {
  expectSame(surface.frontendManifest, canonicalFrontendManifest, `${label} frontendManifest`);
}

export function expectTsProductManifest(manifest: AnyRecord | null | undefined, policyOwner: string, label: string): void {
  if (
    manifest?.source !== "ts" ||
    manifest.policyOwner !== policyOwner ||
    manifest.productSourceOfTruth !== canonicalFrontendManifest.productSourceOfTruth ||
    manifest.productSemanticsOwner !== canonicalFrontendManifest.productSemanticsOwner ||
    manifest.nativeProductPolicy !== canonicalFrontendManifest.nativeProductPolicy ||
    manifest.handwrittenFrontendMirrors !== canonicalFrontendManifest.handwrittenFrontendMirrors
  ) {
    throw new Error(`${label} must expose TS-owned product manifest policy`);
  }
}

export function expectNativeApiContractManifest(manifest: AnyRecord | null | undefined, label: string): void {
  if (
    manifest?.source !== canonicalFrontendManifest.source ||
    manifest.policyOwner !== "src/ts/runtime/native_api_contract.ts" ||
    manifest.productSourceOfTruth !== canonicalFrontendManifest.productSourceOfTruth ||
    manifest.productSemanticsOwner !== canonicalFrontendManifest.productSemanticsOwner ||
    manifest.nativeAlignment !== canonicalFrontendManifest.nativeAlignment ||
    manifest.nativeProductPolicy !== canonicalFrontendManifest.nativeProductPolicy ||
    manifest.handwrittenFrontendMirrors !== canonicalFrontendManifest.handwrittenFrontendMirrors
  ) {
    throw new Error(`${label} must expose the TS-authored native API contract manifest`);
  }
}

export function expectNativeAdapterEvidence(evidence: AnyRecord | null | undefined, host: string, label: string): void {
  if (
    evidence?.frontendSource !== canonicalFrontendManifest.source ||
    evidence.host !== host ||
    evidence.productSemanticsOwner !== canonicalFrontendManifest.productSemanticsOwner ||
    evidence.nativeAlignment !== canonicalFrontendManifest.nativeAlignment ||
    evidence.ownsFrontendPolicy !== canonicalFrontendManifest.handwrittenFrontendMirrors ||
    evidence.ownsNativeLoading !== true
  ) {
    throw new Error(`${label} must expose TS frontend ownership evidence`);
  }
}

export function expectConcreteRuntimeLoaderEvidence(loader: AnyRecord | null | undefined, label: string): void {
  if (
    loader?.productSemanticsOwner !== canonicalFrontendManifest.productSemanticsOwner ||
    loader.productLanguage !== canonicalFrontendManifest.productLanguage ||
    loader.nativeAlignment !== canonicalFrontendManifest.nativeAlignment ||
    loader.handwrittenFrontendMirrors !== canonicalFrontendManifest.handwrittenFrontendMirrors ||
    loader.runtimeType !== "NativeRuntime"
  ) {
    throw new Error(`${label} must expose TS-owned native loader evidence`);
  }
}

export function hotStepParamsCompatibilityForSmoke(smokeKind: string): AnyRecord {
  return Object.freeze({
    kind: "zgml.step-params.compatibility",
    accepted: true,
    canExecute: true,
    allocationFree: true,
    runtimeOutputAllocationFree: true,
    readbackRequired: false,
    readbackFree: true,
    hotPath: true,
    hotPathStatus: "hot",
    hotPathBlockers: Object.freeze([]),
    status: "accepted",
    contractKind: `${smokeKind}-smoke-contract`,
    contractSignature: `${smokeKind}-smoke-contract`,
    contractPosition: null,
    stepParamsSignature: `${smokeKind}-smoke-hot-step-params`,
    rejectionCode: null,
    stateEffect: "none",
    inputSource: "inline",
    inputOwnership: "caller",
    readsInput: true,
    outputTarget: "inline",
    outputEffect: "write",
    outputOwnership: "caller",
    outputReturnOwnership: "caller",
    writesOutput: true,
    inputElementType: "f32",
    outputElementType: "f32",
    inputElementLength: 1,
    outputElementLength: 1,
    inputShape: Object.freeze([1]),
    outputShape: Object.freeze([1]),
    inputShapeSignature: "1",
    outputShapeSignature: "1",
    inputByteLength: 4,
    outputByteLength: 4,
    diagnostics: Object.freeze([]),
  });
}

export function expectStepParamsNamespace(surface: AnyRecord, label: string, smokeKind: string): void {
  const hotStepParamsCompatibility = hotStepParamsCompatibilityForSmoke(smokeKind);
  if (
    surface.stepParamsManifest?.source !== "ts" ||
    typeof surface.stepParamsCompatibilityResult !== "function" ||
    surface.acceptsHotStepParamsCompatibility(hotStepParamsCompatibility) !== true ||
    surface.requireCanExecuteStepParamsCompatibility(hotStepParamsCompatibility) !== hotStepParamsCompatibility ||
    surface.requireHotStepParamsCompatibility(hotStepParamsCompatibility) !== hotStepParamsCompatibility ||
    surface.matchesStepParamsSignature(hotStepParamsCompatibility, hotStepParamsCompatibility.stepParamsSignature) !== true ||
    surface.matchesStepParamsCompatibility(hotStepParamsCompatibility, hotStepParamsCompatibility) !== true
  ) {
    throw new Error(`${label} must expose the TS-authored StepParams hot-path helper namespace`);
  }
}
