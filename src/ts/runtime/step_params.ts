"use strict";

import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";
import type {
  SessionStepParamsCompatibility,
  TokenSampleOptions,
  TokenWindowOptions,
} from "../public_api.js";

type AnyRecord = Record<string, any>;

export const stepParamsManifest = Object.freeze({
  kind: "zgml-step-params",
  ...tsRuntimeManifestPolicy("src/ts/step_params.ts", "Program -> Session -> StepParams"),
});

type StepContract = {
  readonly kind: string;
  readonly signature: string;
  readonly position?: number;
};

type StepParamsEvidence = {
  readonly stateEffect?: string;
  readonly allocationFree?: boolean;
  readonly inputSource?: string;
  readonly outputTarget?: string;
  readonly inputElementType?: string;
  readonly outputElementType?: string;
  readonly inputElementLength?: number;
  readonly outputElementLength?: number;
  readonly inputShape?: readonly number[];
  readonly outputShape?: readonly number[];
  readonly inputByteLength?: number;
  readonly outputByteLength?: number;
};

type StepParamsDiagnosticError = Error & {
  stepParamsDiagnosticCode?: string;
  stepParamsDiagnosticDetails?: AnyRecord;
};

type TokenWindowScratch = TokenWindowOptions & {
  tokensLen?: unknown;
  tokenLength?: unknown;
  activeTokenLength?: unknown;
  activeTokenCount?: unknown;
  output?: unknown;
};

type TokenSampleScratch = TokenSampleOptions & {
  tokensLen?: unknown;
  topK?: unknown;
  top_k?: unknown;
  temperature?: unknown;
  seed?: unknown;
};

export function shapeSignature(shape: readonly number[] | unknown): string {
  return Array.isArray(shape) ? shape.join("x") : "";
}

export function freezeStepParamsDiagnostics(diagnostics: readonly AnyRecord[]) {
  return Object.freeze(diagnostics.map((diagnostic) => {
    const frozenDiagnostic: AnyRecord = { ...diagnostic };
    if (Array.isArray(frozenDiagnostic.actualShape)) {
      frozenDiagnostic.actualShape = Object.freeze(frozenDiagnostic.actualShape.slice());
    }
    if (Array.isArray(frozenDiagnostic.expectedShape)) {
      frozenDiagnostic.expectedShape = Object.freeze(frozenDiagnostic.expectedShape.slice());
    }
    return Object.freeze(frozenDiagnostic);
  }));
}

export function frozenShapeEvidence(shape: unknown): readonly number[] {
  return Object.freeze(Array.isArray(shape) ? shape.slice() : []);
}

export function stepParamsValidationError(
  code: string,
  message: string,
  details: AnyRecord | null = null,
): StepParamsDiagnosticError {
  const err = new Error(message) as StepParamsDiagnosticError;
  err.stepParamsDiagnosticCode = code;
  if (details && typeof details === "object") {
    err.stepParamsDiagnosticDetails = { ...details };
  }
  return err;
}

export function stepParamsDiagnosticCode(error: unknown): string {
  return typeof ((error as StepParamsDiagnosticError | null)?.stepParamsDiagnosticCode) === "string"
    ? (error as StepParamsDiagnosticError).stepParamsDiagnosticCode as string
    : "invalid-step-params";
}

export function stepParamsDiagnosticDetails(error: unknown): AnyRecord | null {
  const details = (error as StepParamsDiagnosticError | null)?.stepParamsDiagnosticDetails;
  return details && typeof details === "object" ? details : null;
}

export function assertFloat32OutputBuffer(outputValues: unknown, method: string) {
  if (!(outputValues instanceof Float32Array)) {
    throw new Error(`session.${method} requires a Float32Array output buffer`);
  }
}

export function validateLogitsOutputBuffer(
  outputValues: unknown,
  logitsLen: number,
  label: string,
  options: { allowFalse?: boolean } = {},
) {
  if (outputValues === false && options.allowFalse) return;
  if (!(outputValues instanceof Float32Array)) {
    const suffix = options.allowFalse ? " or false" : "";
    throw stepParamsValidationError("invalid-output", `${label} output must be a Float32Array${suffix}`);
  }
  if (outputValues.length < logitsLen) {
    throw stepParamsValidationError(
      "invalid-output",
      `${label} output length ${outputValues.length} is smaller than logits length ${logitsLen}`,
      { actualLength: outputValues.length, expectedLength: logitsLen },
    );
  }
}

export function validateTokenWindowFitsContract(contract: AnyRecord, tokensLen: number, label: string) {
  if (tokensLen <= contract.remainingContext) return;
  throw stepParamsValidationError(
    "invalid-token-window",
    `${label} length ${tokensLen} exceeds remaining context ${contract.remainingContext}`,
    {
      actualLength: tokensLen,
      expectedLength: contract.remainingContext,
      contextLength: contract.contextLength,
      position: contract.position,
      remainingContext: contract.remainingContext,
    },
  );
}

export function validateLlamaTokenStepParams(params: unknown, method: string) {
  if (params === null || typeof params !== "object" || Array.isArray(params) || ArrayBuffer.isView(params)) {
    throw stepParamsValidationError("invalid-step-params", `session.${method} requires a StepParams object`);
  }
  const record = params as AnyRecord;
  const hasTokens = Object.prototype.hasOwnProperty.call(record, "tokens");
  const hasToken = Object.prototype.hasOwnProperty.call(record, "token");
  if (hasTokens === hasToken) {
    throw stepParamsValidationError("token-choice-mismatch", `session.${method} requires exactly one of token or tokens`);
  }
  return { hasTokens, hasToken };
}

export function validateExecuteStepParams(params: unknown, method: string) {
  if (params === undefined || params === null) return {};
  if (typeof params !== "object" || Array.isArray(params) || ArrayBuffer.isView(params)) {
    throw stepParamsValidationError("invalid-step-params", `session.${method} requires a StepParams object`);
  }
  return params as AnyRecord;
}

export function assertNoInlineOutput(params: AnyRecord, method: string) {
  if (Object.prototype.hasOwnProperty.call(params, "output")) {
    throw new Error(`session.${method} takes the output buffer as its first argument`);
  }
}

export function copyTokenWindowOptions(options: TokenWindowOptions = {}, scratch: TokenWindowScratch, output: unknown) {
  const src = options || {};
  scratch.tokensLen = src.tokensLen;
  scratch.tokenLength = src.tokenLength;
  scratch.activeTokenLength = src.activeTokenLength;
  scratch.activeTokenCount = src.activeTokenCount;
  scratch.output = output;
  return scratch;
}

export function noOutputTokenWindowOptions(options: TokenWindowOptions, scratch: TokenWindowScratch) {
  return copyTokenWindowOptions(options, scratch, false);
}

export function outputTokenWindowOptions(options: TokenWindowOptions, scratch: TokenWindowScratch, outputValues: unknown) {
  return copyTokenWindowOptions(options, scratch, outputValues);
}

export function scalarTokenExecuteOptions(options: TokenWindowOptions & { output?: unknown } = {}, scratch: TokenWindowScratch) {
  copyTokenWindowOptions(options, scratch, options ? options.output : undefined);
  scratch.tokensLen = 1;
  return scratch;
}

export function scalarTokenOutputOptions(scratch: TokenWindowScratch, outputValues: unknown) {
  scratch.tokensLen = 1;
  scratch.tokenLength = undefined;
  scratch.activeTokenLength = undefined;
  scratch.activeTokenCount = undefined;
  scratch.output = outputValues;
  return scratch;
}

export function scalarTokenSampleOptions(options: TokenSampleOptions = {}, scratch: TokenSampleScratch) {
  const src = options || {};
  scratch.tokensLen = 1;
  scratch.topK = src.topK;
  scratch.top_k = src.top_k;
  scratch.temperature = src.temperature;
  scratch.seed = src.seed;
  return scratch;
}

export function stepParamsOutputEffect(outputTarget: string): string {
  if (outputTarget === "none") return "none";
  if (outputTarget === "bound-native-readback") return "write-readback";
  if (
    outputTarget === "inline" ||
    outputTarget === "allocated" ||
    outputTarget === "bound-host"
  ) {
    return "write";
  }
  return "rejected";
}

export function stepParamsOutputOwnership(outputTarget: string): string {
  if (outputTarget === "none") return "none";
  if (outputTarget === "inline") return "caller";
  if (outputTarget === "allocated") return "runtime";
  if (outputTarget === "bound-host" || outputTarget === "bound-native-readback") return "session";
  return "rejected";
}

export function stepParamsOutputReturnOwnership(outputTarget: string): string {
  if (outputTarget === "none") return "none";
  if (outputTarget === "inline") return "caller";
  if (outputTarget === "bound-host") return "session";
  if (outputTarget === "allocated" || outputTarget === "bound-native-readback") return "runtime";
  return "rejected";
}

export function stepParamsInputOwnership(inputSource: string): string {
  if (inputSource === "none") return "none";
  if (inputSource === "inline" || inputSource === "token" || inputSource === "tokens") return "caller";
  if (inputSource === "bound-host" || inputSource === "bound-native") return "session";
  return "rejected";
}

export function stepParamsHotPathStatus(accepted: boolean, allocationFree: boolean, readbackRequired: boolean): string {
  if (!accepted) return "rejected";
  if (allocationFree && !readbackRequired) return "hot";
  if (!allocationFree && readbackRequired) return "allocates-readback";
  if (!allocationFree) return "allocates";
  return "readback";
}

export function stepParamsHotPathBlockers(
  accepted: boolean,
  runtimeOutputAllocationFree: boolean,
  readbackFree: boolean,
) {
  if (!accepted) return Object.freeze(["rejected"]);
  const blockers: string[] = [];
  if (!runtimeOutputAllocationFree) blockers.push("runtime-output-allocation");
  if (!readbackFree) blockers.push("readback");
  return Object.freeze(blockers);
}

export function stepParamsCompatibilityResult(
  contract: StepContract,
  error: unknown,
  evidence: StepParamsEvidence | null = null,
) {
  const contractPosition = typeof contract.position === "number" ? contract.position : null;
  const status = error ? "rejected" : "accepted";
  const accepted = !error;
  const stateEffect = accepted && evidence && evidence.stateEffect === "advance"
    ? "advance"
    : accepted
      ? "none"
      : "rejected";
  const allocationFree = evidence && typeof evidence.allocationFree === "boolean"
    ? evidence.allocationFree
    : false;
  const runtimeOutputAllocationFree = allocationFree;
  const inputSource = evidence && typeof evidence.inputSource === "string"
    ? evidence.inputSource
    : "rejected";
  const inputOwnership = stepParamsInputOwnership(inputSource);
  const readsInput = accepted && inputSource !== "none";
  const outputTarget = evidence && typeof evidence.outputTarget === "string"
    ? evidence.outputTarget
    : "rejected";
  const outputEffect = stepParamsOutputEffect(outputTarget);
  const outputOwnership = stepParamsOutputOwnership(outputTarget);
  const outputReturnOwnership = stepParamsOutputReturnOwnership(outputTarget);
  const writesOutput = accepted && (outputEffect === "write" || outputEffect === "write-readback");
  const readbackRequired = outputEffect === "write-readback";
  const readbackFree = accepted && !readbackRequired;
  const hotPath = runtimeOutputAllocationFree && readbackFree;
  const hotPathStatus = stepParamsHotPathStatus(accepted, allocationFree, readbackRequired);
  const hotPathBlockers = stepParamsHotPathBlockers(accepted, runtimeOutputAllocationFree, readbackFree);
  const inputElementType = evidence && typeof evidence.inputElementType === "string"
    ? evidence.inputElementType
    : "rejected";
  const outputElementType = evidence && typeof evidence.outputElementType === "string"
    ? evidence.outputElementType
    : "rejected";
  const inputElementLength = evidence && Number.isFinite(evidence.inputElementLength)
    ? Number(evidence.inputElementLength)
    : 0;
  const outputElementLength = evidence && Number.isFinite(evidence.outputElementLength)
    ? Number(evidence.outputElementLength)
    : 0;
  const inputByteLength = evidence && Number.isFinite(evidence.inputByteLength)
    ? Number(evidence.inputByteLength)
    : 0;
  const outputByteLength = evidence && Number.isFinite(evidence.outputByteLength)
    ? Number(evidence.outputByteLength)
    : 0;
  const inputShape = accepted && evidence && Array.isArray(evidence.inputShape)
    ? frozenShapeEvidence(evidence.inputShape)
    : Object.freeze([]);
  const outputShape = accepted && evidence && Array.isArray(evidence.outputShape)
    ? frozenShapeEvidence(evidence.outputShape)
    : Object.freeze([]);
  const inputShapeSignature = shapeSignature(inputShape) || "none";
  const outputShapeSignature = shapeSignature(outputShape) || "none";
  const diagnosticCode = error ? stepParamsDiagnosticCode(error) : null;
  const stepParamsSignature = [
    contract.signature,
    `position=${contractPosition === null ? "none" : contractPosition}`,
    `status=${status}`,
    `canExecute=${accepted ? 1 : 0}`,
    `stateEffect=${stateEffect}`,
    `inputSource=${inputSource}`,
    `inputOwnership=${inputOwnership}`,
    `readsInput=${readsInput ? 1 : 0}`,
    `outputTarget=${outputTarget}`,
    `outputEffect=${outputEffect}`,
    `outputOwnership=${outputOwnership}`,
    `outputReturnOwnership=${outputReturnOwnership}`,
    `writesOutput=${writesOutput ? 1 : 0}`,
    `readback=${readbackRequired ? 1 : 0}`,
    `readbackFree=${readbackFree ? 1 : 0}`,
    `allocationFree=${allocationFree ? 1 : 0}`,
    `runtimeOutputAllocationFree=${runtimeOutputAllocationFree ? 1 : 0}`,
    `hotPath=${hotPath ? 1 : 0}`,
    `hotPathStatus=${hotPathStatus}`,
    `hotPathBlockers=${hotPathBlockers.join(",") || "none"}`,
    `inputType=${inputElementType}`,
    `outputType=${outputElementType}`,
    `inputElements=${inputElementLength}`,
    `outputElements=${outputElementLength}`,
    `inputShape=${inputShapeSignature}`,
    `outputShape=${outputShapeSignature}`,
    `inputBytes=${inputByteLength}`,
    `outputBytes=${outputByteLength}`,
    diagnosticCode ? `diagnostic=${diagnosticCode}` : null,
  ].filter(Boolean).join("|");
  if (!error) {
    return Object.freeze({
      kind: "zgml.step-params.compatibility",
      accepted: true,
      canExecute: true,
      status,
      contractKind: contract.kind,
      contractSignature: contract.signature,
      contractPosition,
      stepParamsSignature,
      rejectionCode: null,
      stateEffect,
      allocationFree,
      inputSource,
      inputOwnership,
      readsInput,
      outputTarget,
      outputEffect,
      outputOwnership,
      outputReturnOwnership,
      writesOutput,
      readbackRequired,
      readbackFree,
      runtimeOutputAllocationFree,
      hotPath,
      hotPathStatus,
      hotPathBlockers,
      inputElementType,
      outputElementType,
      inputElementLength,
      outputElementLength,
      inputShape,
      outputShape,
      inputShapeSignature,
      outputShapeSignature,
      inputByteLength,
      outputByteLength,
      diagnostics: Object.freeze([]),
    });
  }
  return Object.freeze({
    kind: "zgml.step-params.compatibility",
    accepted: false,
    canExecute: false,
    status,
    contractKind: contract.kind,
    contractSignature: contract.signature,
    contractPosition,
    stepParamsSignature,
    rejectionCode: diagnosticCode,
    stateEffect: "rejected",
    allocationFree: false,
    inputSource: "rejected",
    inputOwnership: "rejected",
    readsInput: false,
    outputTarget: "rejected",
    outputEffect: "rejected",
    outputOwnership: "rejected",
    outputReturnOwnership: "rejected",
    writesOutput: false,
    readbackRequired: false,
    readbackFree: false,
    runtimeOutputAllocationFree: false,
    hotPath: false,
    hotPathStatus: "rejected",
    hotPathBlockers,
    inputElementType: "rejected",
    outputElementType: "rejected",
    inputElementLength: 0,
    outputElementLength: 0,
    inputShape,
    outputShape,
    inputShapeSignature,
    outputShapeSignature,
    inputByteLength: 0,
    outputByteLength: 0,
    diagnostics: freezeStepParamsDiagnostics([{
      code: stepParamsDiagnosticCode(error),
      message: String((error as Error | null)?.message ? (error as Error).message : error),
      ...stepParamsDiagnosticDetails(error),
    }]),
  });
}

function isRecord(value: unknown): value is AnyRecord {
  return value !== null && typeof value === "object";
}

function isFrozenArray(value: unknown): value is readonly unknown[] {
  return Array.isArray(value) && Object.isFrozen(value);
}

function isNonNegativeFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function stepParamsCompatibilityRejectionReason(compatibility: AnyRecord): string | null {
  if (compatibility.kind !== "zgml.step-params.compatibility") return "kind is not zgml.step-params.compatibility";
  if (typeof compatibility.accepted !== "boolean") return "accepted is missing";
  if (typeof compatibility.canExecute !== "boolean") return "canExecute is missing";
  if (compatibility.status !== "accepted" && compatibility.status !== "rejected") return "status is not accepted or rejected";
  if (typeof compatibility.contractKind !== "string" || compatibility.contractKind.length === 0) return "contractKind is missing";
  if (typeof compatibility.contractSignature !== "string" || compatibility.contractSignature.length === 0) return "contractSignature is missing";
  if (compatibility.contractPosition !== null && !Number.isSafeInteger(compatibility.contractPosition)) return "contractPosition is not an integer or null";
  if (typeof compatibility.stepParamsSignature !== "string" || compatibility.stepParamsSignature.length === 0) return "stepParamsSignature is missing";
  if (compatibility.rejectionCode !== null && typeof compatibility.rejectionCode !== "string") return "rejectionCode is not a string or null";
  if (typeof compatibility.stateEffect !== "string" || compatibility.stateEffect.length === 0) return "stateEffect is missing";
  if (typeof compatibility.inputSource !== "string" || compatibility.inputSource.length === 0) return "inputSource is missing";
  if (typeof compatibility.outputTarget !== "string" || compatibility.outputTarget.length === 0) return "outputTarget is missing";
  if (typeof compatibility.hotPathStatus !== "string" || compatibility.hotPathStatus.length === 0) return "hotPathStatus is missing";
  for (const field of [
    "allocationFree",
    "readsInput",
    "writesOutput",
    "readbackRequired",
    "readbackFree",
    "runtimeOutputAllocationFree",
    "hotPath",
  ]) {
    if (typeof compatibility[field] !== "boolean") return `${field} is missing`;
  }
  for (const field of [
    "inputElementLength",
    "outputElementLength",
    "inputByteLength",
    "outputByteLength",
  ]) {
    if (!isNonNegativeFiniteNumber(compatibility[field])) return `${field} is not a non-negative number`;
  }
  for (const field of ["inputShape", "outputShape", "hotPathBlockers", "diagnostics"]) {
    if (!isFrozenArray(compatibility[field])) return `${field} is not frozen evidence`;
  }
  if (compatibility.accepted !== compatibility.canExecute) return "accepted and canExecute disagree";
  if (compatibility.accepted && compatibility.status !== "accepted") return "accepted evidence must have accepted status";
  if (!compatibility.accepted && compatibility.status !== "rejected") return "rejected evidence must have rejected status";
  return null;
}

export function isStepParamsCompatibility(compatibility: unknown): compatibility is SessionStepParamsCompatibility {
  return isRecord(compatibility) && stepParamsCompatibilityRejectionReason(compatibility) === null;
}

export function requireStepParamsCompatibility(compatibility: unknown): SessionStepParamsCompatibility {
  if (isStepParamsCompatibility(compatibility)) return compatibility;
  const reason = isRecord(compatibility)
    ? stepParamsCompatibilityRejectionReason(compatibility)
    : "value is not a StepParams compatibility record";
  throw stepParamsValidationError(
    "invalid-step-params",
    `StepParamsCompatibility is not valid evidence: ${reason}`,
  );
}

export const assertStepParamsCompatibility = requireStepParamsCompatibility;
export const assert_step_params_compatibility = requireStepParamsCompatibility;

export function acceptsStepParamsCompatibility(compatibility: unknown): compatibility is SessionStepParamsCompatibility {
  return isStepParamsCompatibility(compatibility) && compatibility.accepted === true;
}

export function canExecuteStepParamsCompatibility(compatibility: unknown): compatibility is SessionStepParamsCompatibility {
  return acceptsStepParamsCompatibility(compatibility);
}

export function requireCanExecuteStepParamsCompatibility(compatibility: unknown): SessionStepParamsCompatibility {
  const checked = requireStepParamsCompatibility(compatibility);
  if (checked.accepted === true) return checked;
  throw stepParamsValidationError(
    "not-executable-step-params",
    `StepParams are not executable: status=${String(checked.status || "unknown")}`,
    {
      status: checked.status,
      rejectionCode: checked.rejectionCode || null,
      stepParamsSignature: checked.stepParamsSignature,
    },
  );
}

export function acceptsAllocationFreeStepParamsCompatibility(compatibility: unknown): compatibility is SessionStepParamsCompatibility {
  return isStepParamsCompatibility(compatibility) && compatibility.accepted === true && compatibility.allocationFree === true;
}

export function requireAllocationFreeStepParamsCompatibility(compatibility: unknown): SessionStepParamsCompatibility {
  const checked = requireStepParamsCompatibility(compatibility);
  if (checked.accepted === true && checked.allocationFree === true) return checked;
  throw stepParamsValidationError(
    "not-allocation-free-step-params",
    `StepParams are not allocation-free compatible: status=${String(checked.status || "unknown")} allocationFree=${String(checked.allocationFree === true)}`,
    {
      status: checked.status,
      allocationFree: checked.allocationFree === true,
      rejectionCode: checked.rejectionCode || null,
      stepParamsSignature: checked.stepParamsSignature,
    },
  );
}

export function acceptsRuntimeOutputAllocationFreeStepParamsCompatibility(compatibility: unknown): compatibility is SessionStepParamsCompatibility {
  return isStepParamsCompatibility(compatibility) && compatibility.accepted === true && compatibility.runtimeOutputAllocationFree === true;
}

export function requireRuntimeOutputAllocationFreeStepParamsCompatibility(compatibility: unknown): SessionStepParamsCompatibility {
  const checked = requireStepParamsCompatibility(compatibility);
  if (checked.accepted === true && checked.runtimeOutputAllocationFree === true) return checked;
  throw stepParamsValidationError(
    "not-runtime-output-allocation-free-step-params",
    `StepParams are not runtime-output-allocation-free compatible: status=${String(checked.status || "unknown")} runtimeOutputAllocationFree=${String(checked.runtimeOutputAllocationFree === true)}`,
    {
      status: checked.status,
      runtimeOutputAllocationFree: checked.runtimeOutputAllocationFree === true,
      rejectionCode: checked.rejectionCode || null,
      stepParamsSignature: checked.stepParamsSignature,
    },
  );
}

export function acceptsNoReadbackStepParamsCompatibility(compatibility: unknown): compatibility is SessionStepParamsCompatibility {
  return isStepParamsCompatibility(compatibility) && compatibility.accepted === true && compatibility.readbackRequired !== true;
}

export function requireNoReadbackStepParamsCompatibility(compatibility: unknown): SessionStepParamsCompatibility {
  const checked = requireStepParamsCompatibility(compatibility);
  if (checked.accepted === true && checked.readbackRequired !== true) return checked;
  throw stepParamsValidationError(
    "not-no-readback-step-params",
    `StepParams are not no-readback compatible: status=${String(checked.status || "unknown")} readbackRequired=${String(checked.readbackRequired === true)}`,
    {
      status: checked.status,
      readbackRequired: checked.readbackRequired === true,
      rejectionCode: checked.rejectionCode || null,
      stepParamsSignature: checked.stepParamsSignature,
    },
  );
}

export function acceptsReadbackFreeStepParamsCompatibility(compatibility: unknown): compatibility is SessionStepParamsCompatibility {
  return isStepParamsCompatibility(compatibility) && compatibility.accepted === true && compatibility.readbackFree === true;
}

export function requireReadbackFreeStepParamsCompatibility(compatibility: unknown): SessionStepParamsCompatibility {
  const checked = requireStepParamsCompatibility(compatibility);
  if (checked.accepted === true && checked.readbackFree === true) return checked;
  throw stepParamsValidationError(
    "not-readback-free-step-params",
    `StepParams are not readback-free compatible: status=${String(checked.status || "unknown")} readbackFree=${String(checked.readbackFree === true)}`,
    {
      status: checked.status,
      readbackFree: checked.readbackFree === true,
      rejectionCode: checked.rejectionCode || null,
      stepParamsSignature: checked.stepParamsSignature,
    },
  );
}

export function acceptsHotStepParamsCompatibility(compatibility: unknown): compatibility is SessionStepParamsCompatibility {
  return isStepParamsCompatibility(compatibility) && compatibility.hotPath === true;
}

export function requireHotStepParamsCompatibility(compatibility: unknown): SessionStepParamsCompatibility {
  const checked = requireStepParamsCompatibility(compatibility);
  if (checked.hotPath === true) return checked;
  const blockers = checked.hotPathBlockers.join(",");
  const rejection = typeof checked.rejectionCode === "string" && checked.rejectionCode.length > 0
    ? ` rejection=${checked.rejectionCode}`
    : "";
  throw stepParamsValidationError(
    "not-hot-step-params",
    `StepParams are not hot-path compatible: status=${String(checked.hotPathStatus || "unknown")} blockers=${blockers || "none"}${rejection}`,
    {
      status: checked.status,
      hotPathStatus: checked.hotPathStatus,
      hotPathBlockers: checked.hotPathBlockers.slice(),
      rejectionCode: checked.rejectionCode || null,
      stepParamsSignature: checked.stepParamsSignature,
    },
  );
}

export function matchesStepContractSignature(contract: unknown, signature: unknown): boolean {
  return typeof signature === "string" && signature.length > 0 && isRecord(contract) && contract.signature === signature;
}

export function matchesStepParamsSignature(compatibility: unknown, signature: unknown): boolean {
  return typeof signature === "string" &&
    signature.length > 0 &&
    isStepParamsCompatibility(compatibility) &&
    compatibility.stepParamsSignature === signature;
}

export function matchesStepParamsCompatibility(compatibility: unknown, expected: unknown): boolean {
  return isStepParamsCompatibility(expected) &&
    matchesStepParamsSignature(compatibility, expected.stepParamsSignature);
}
