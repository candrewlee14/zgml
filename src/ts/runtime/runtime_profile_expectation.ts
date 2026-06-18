"use strict";

import type {
  RuntimeProfile,
  RuntimeProfileExpectation,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
type RuntimeProfileEvidenceRecord = Readonly<UnknownRecord & {
  kind?: unknown;
  signature?: unknown;
}>;

const runtimeProfileNumberFields = Object.freeze([
  "callCount",
  "backendOpCount",
  "fallbackOpCount",
  "backendDispatchCount",
  "syncCount",
  "runtimePatchCallCount",
  "runtimePatchChangedCount",
  "runtimePatchInvalidCount",
  "runtimePatchHoles",
  "runtimePatchCacheWritePosHoles",
  "runtimePatchAttentionSeqKvHoles",
  "commandCount",
  "commandOpCount",
  "commandRowCount",
  "commandProjectionCount",
  "commandAttentionCount",
  "commandMovementCount",
  "commandElementwiseCount",
  "commandRopeCount",
] as const);

const runtimeProfileBigintFields = Object.freeze([
  "runtimePatchStencilHash",
  "commandStencilHash",
] as const);

function profileCount(profile: RuntimeProfileEvidenceRecord, field: string) {
  const value = Number(profile[field]);
  const out = Number(value ?? 0);
  if (!Number.isSafeInteger(out) || out < 0) {
    throw new Error(`runtime profile requires non-negative integer ${field}`);
  }
  return out;
}

function expectationSignature(expectation: Pick<RuntimeProfileExpectation, "fallbackOpCount" | "syncCount" | "runtimePatchInvalidCount" | "noFallback" | "noSync" | "noRuntimePatchInvalid">) {
  return [
    "runtime-profile-expectation",
    `fallbackOpCount=${expectation.fallbackOpCount}`,
    `syncCount=${expectation.syncCount}`,
    `runtimePatchInvalidCount=${expectation.runtimePatchInvalidCount}`,
    `noFallback=${expectation.noFallback ? 1 : 0}`,
    `noSync=${expectation.noSync ? 1 : 0}`,
    `noRuntimePatchInvalid=${expectation.noRuntimePatchInvalid ? 1 : 0}`,
  ].join("|");
}

function isRecord(value: unknown): value is UnknownRecord {
  return value !== null && typeof value === "object";
}

function profileFieldReason(profile: RuntimeProfileEvidenceRecord) {
  if (profile.kind !== "zgml.runtime.profile") return "kind is not zgml.runtime.profile";
  if (typeof profile.signature !== "string" || profile.signature.length === 0) return "signature is missing";
  for (const field of runtimeProfileNumberFields) {
    const value = profile[field];
    if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) return `${field} is not a non-negative integer`;
  }
  for (const field of runtimeProfileBigintFields) {
    if (typeof profile[field] !== "bigint") return `${field} is not a bigint`;
  }
  return null;
}

export function acceptsRuntimeProfile(profile: unknown): profile is RuntimeProfile {
  if (!isRecord(profile)) return false;
  return profileFieldReason(profile) === null;
}

export function requireRuntimeProfile(profile: unknown): RuntimeProfile {
  if (acceptsRuntimeProfile(profile)) return profile;
  const reason = isRecord(profile) ? profileFieldReason(profile) : "value is not a runtime profile";
  throw new Error(`RuntimeProfile is not valid evidence: ${reason}`);
}

export const assertRuntimeProfile = requireRuntimeProfile;
export const assert_runtime_profile = requireRuntimeProfile;

export function matchesRuntimeProfileSignature(profile: unknown, signature: unknown): boolean {
  return isRecord(profile) &&
    profile.kind === "zgml.runtime.profile" &&
    typeof signature === "string" &&
    signature.length > 0 &&
    profile.signature === signature;
}

export function runtimeProfileExpectation(profile: unknown): RuntimeProfileExpectation {
  requireRuntimeProfile(profile);
  if (!isRecord(profile)) {
    throw new Error("runtime profile expectation requires a runtime profile");
  }
  const record = profile as RuntimeProfileEvidenceRecord;
  const counts = {
    fallbackOpCount: profileCount(record, "fallbackOpCount"),
    syncCount: profileCount(record, "syncCount"),
    runtimePatchInvalidCount: profileCount(record, "runtimePatchInvalidCount"),
  };
  const expectation: RuntimeProfileExpectation = {
    kind: "zgml.runtime.profile-expectation",
    ...counts,
    noFallback: counts.fallbackOpCount === 0,
    noSync: counts.syncCount === 0,
    noRuntimePatchInvalid: counts.runtimePatchInvalidCount === 0,
    signature: "",
  };
  return Object.freeze({
    ...expectation,
    signature: expectationSignature(expectation),
  });
}

export function runtimeProfileHasNoFallback(profile: unknown) {
  return runtimeProfileExpectation(profile).noFallback;
}

export function runtimeProfileHasNoSync(profile: unknown) {
  return runtimeProfileExpectation(profile).noSync;
}

export function runtimeProfileHasNoInvalidRuntimePatches(profile: unknown) {
  return runtimeProfileExpectation(profile).noRuntimePatchInvalid;
}

export function requireNoFallbackRuntimeProfile(profile: unknown) {
  const expectation = runtimeProfileExpectation(profile);
  if (!expectation.noFallback) {
    throw new Error(`runtime profile expected no fallback ops, got ${expectation.fallbackOpCount}`);
  }
  return expectation;
}

export function requireNoSyncRuntimeProfile(profile: unknown) {
  const expectation = runtimeProfileExpectation(profile);
  if (!expectation.noSync) {
    throw new Error(`runtime profile expected no syncs, got ${expectation.syncCount}`);
  }
  return expectation;
}

export function requireRuntimePatchValidProfile(profile: unknown) {
  const expectation = runtimeProfileExpectation(profile);
  if (!expectation.noRuntimePatchInvalid) {
    throw new Error(`runtime profile expected no invalid runtime patches, got ${expectation.runtimePatchInvalidCount}`);
  }
  return expectation;
}

export function requireHotRuntimeProfile(profile: unknown) {
  const expectation = runtimeProfileExpectation(profile);
  const failures = [];
  if (!expectation.noFallback) failures.push(`fallbackOpCount=${expectation.fallbackOpCount}`);
  if (!expectation.noSync) failures.push(`syncCount=${expectation.syncCount}`);
  if (!expectation.noRuntimePatchInvalid) failures.push(`runtimePatchInvalidCount=${expectation.runtimePatchInvalidCount}`);
  if (failures.length > 0) {
    throw new Error(`runtime profile expected hot path evidence, got ${failures.join(", ")}`);
  }
  return expectation;
}
