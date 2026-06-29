"use strict";

import type {
  ModuleKernelParameterLayout,
  ModuleKernelParameterLayoutEntry,
} from "../public_api.js";

type UnknownRecord = Readonly<Record<string, unknown>>;

export type SessionParameterLayoutEntryInput =
  | ModuleKernelParameterLayoutEntry
  | Readonly<{
    name?: unknown;
    binding?: unknown;
    readonly [key: string]: unknown;
  }>;

export type SessionParameterLayoutInput =
  | ModuleKernelParameterLayout
  | Readonly<{
    weightsLen?: unknown;
    biasLen?: unknown;
    parameters?: readonly SessionParameterLayoutEntryInput[];
  }>
  | UnknownRecord
  | null
  | undefined;

function parametersFromLayout(layout: SessionParameterLayoutInput): readonly SessionParameterLayoutEntryInput[] {
  return layout && Array.isArray(layout.parameters) ? layout.parameters : [];
}

export function sessionParameterNames(layout: SessionParameterLayoutInput): readonly string[] {
  return Object.freeze(parametersFromLayout(layout).map((param) => String(param.name)));
}

export function sessionParameterInfos(layout: SessionParameterLayoutInput): readonly SessionParameterLayoutEntryInput[] {
  return Object.freeze(parametersFromLayout(layout).slice());
}

export function sessionParameterInfo(
  layout: SessionParameterLayoutInput,
  nameOrIndex: unknown,
  label = "Session.parameterInfo",
): SessionParameterLayoutEntryInput | null {
  const parameters = parametersFromLayout(layout);
  if (parameters.length === 0) return null;
  if (typeof nameOrIndex === "number") {
    if (!Number.isInteger(nameOrIndex) || nameOrIndex < 0) {
      throw new Error(`${label} index must be a non-negative integer`);
    }
    return parameters[nameOrIndex] || null;
  }
  if (typeof nameOrIndex === "string") {
    return parameters.find((param) => param && param.name === nameOrIndex) || null;
  }
  throw new Error(`${label} requires a parameter name or index`);
}

export function sessionParameterPersistentBindingIndex(
  entry: SessionParameterLayoutEntryInput | null | undefined,
  layout: SessionParameterLayoutInput,
): number {
  if (!entry || typeof entry !== "object") {
    throw new Error("Session parameter entry is unavailable");
  }
  if (entry.binding === "weights") {
    if (layout && Number(layout.weightsLen || 0) <= 0) {
      throw new Error(`Session parameter ${entry.name} has no weights binding`);
    }
    return 0;
  }
  if (entry.binding === "bias") {
    if (layout && Number(layout.biasLen || 0) <= 0) {
      throw new Error(`Session parameter ${entry.name} has no bias binding`);
    }
    return layout && Number(layout.weightsLen || 0) > 0 ? 1 : 0;
  }
  throw new Error(`Session parameter ${entry.name} uses unsupported binding ${entry.binding}`);
}

export function requireSessionParameterName(name: unknown): string {
  if (typeof name !== "string" || name.length === 0) {
    throw new Error("Session.uploadParameterByName requires a non-empty parameter name");
  }
  return name;
}

export function sessionParameterInfoByNameForUpload(
  layout: SessionParameterLayoutInput,
  name: unknown,
): SessionParameterLayoutEntryInput {
  const parameterName = requireSessionParameterName(name);
  const entry = sessionParameterInfo(layout, parameterName);
  if (!entry) {
    throw new Error(`Session parameter ${parameterName} was not found`);
  }
  return entry;
}
