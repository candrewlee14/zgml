"use strict";

import type {
  SessionCallProfile,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
type MutableSessionCallProfileRecord = Record<string, number>;
type SessionCallProfileSource = {
  frontendCallProfile?: MutableSessionCallProfileRecord;
};
type SessionCallProfileEvidenceRecord = Readonly<UnknownRecord & {
  kind?: unknown;
  signature?: unknown;
}>;

export const sessionCallProfileFields = Object.freeze([
  "stepCount",
  "stepTensorCount",
  "stepIntoCount",
  "executeCount",
  "executeTensorCount",
  "executeIntoCount",
  "prepareExecuteIntoCount",
  "prefillCount",
  "prefillTensorCount",
  "prefillIntoCount",
  "readOutputIntoCount",
  "readOutputTensorCount",
  "advanceCount",
  "resetCount",
  "uploadParametersCount",
  "uploadParameterCount",
  "uploadParameterByNameCount",
  "uploadParameterRangeCount",
]);

export function emptySessionCallProfile(): MutableSessionCallProfileRecord {
  const profile: MutableSessionCallProfileRecord = {};
  for (const field of sessionCallProfileFields) profile[field] = 0;
  return profile;
}

export function mutableSessionCallProfile(session: SessionCallProfileSource): MutableSessionCallProfileRecord {
  if (!session.frontendCallProfile) session.frontendCallProfile = emptySessionCallProfile();
  return session.frontendCallProfile;
}

export function bumpSessionCallProfile(session: SessionCallProfileSource, field: string) {
  if (!sessionCallProfileFields.includes(field)) {
    throw new Error(`unknown Session call profile field: ${field}`);
  }
  const profile = mutableSessionCallProfile(session);
  profile[field] = (profile[field] ?? 0) + 1;
}

export function sessionCallProfile(session: SessionCallProfileSource): SessionCallProfile {
  const profile: Record<string, number | string> = { ...mutableSessionCallProfile(session) };
  profile.kind = "zgml.session.call-profile";
  profile.signature = [
    "session-call-profile",
    ...sessionCallProfileFields.map((field) => `${field}=${profile[field]}`),
  ].join("|");
  return Object.freeze(profile) as SessionCallProfile;
}

function isRecord(value: unknown): value is UnknownRecord {
  return value !== null && typeof value === "object";
}

function sessionCallProfileRejectionReason(profile: SessionCallProfileEvidenceRecord) {
  if (profile.kind !== "zgml.session.call-profile") return "kind is not zgml.session.call-profile";
  if (typeof profile.signature !== "string" || profile.signature.length === 0) return "signature is missing";
  for (const field of sessionCallProfileFields) {
    const value = profile[field];
    if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) return `${field} is not a non-negative integer`;
  }
  return null;
}

export function acceptsSessionCallProfile(profile: unknown): profile is SessionCallProfile {
  return isRecord(profile) && sessionCallProfileRejectionReason(profile) === null;
}

export function requireSessionCallProfile(profile: unknown): SessionCallProfile {
  if (acceptsSessionCallProfile(profile)) return profile;
  const reason = isRecord(profile) ? sessionCallProfileRejectionReason(profile) : "value is not a Session call profile";
  throw new Error(`SessionCallProfile is not valid evidence: ${reason}`);
}

export const assertSessionCallProfile = requireSessionCallProfile;
export const assert_session_call_profile = requireSessionCallProfile;

export function matchesSessionCallProfileSignature(sessionOrProfile: unknown, signature: unknown): boolean {
  if (typeof signature !== "string" || signature.length === 0) return false;
  if (acceptsSessionCallProfile(sessionOrProfile)) return sessionOrProfile.signature === signature;
  return isRecord(sessionOrProfile) && sessionCallProfile(sessionOrProfile as SessionCallProfileSource).signature === signature;
}

export function resetSessionCallProfile(session: SessionCallProfileSource) {
  session.frontendCallProfile = emptySessionCallProfile();
}
