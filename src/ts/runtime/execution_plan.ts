"use strict";

import {
  requireCanExecuteStepParamsCompatibility,
} from "./step_params.js";
import { compileSupportRejectionReason } from "./compile_support.js";

import type {
  ModuleBindingPlan,
  ModuleCompileExplanation,
  ProgramBindingPlan,
  ProgramExecutionPlan,
  SessionExecutionPlan,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
type DiagnosticRecord = Readonly<{
  message?: unknown;
}>;
type DiagnosticCarrierRecord = Readonly<UnknownRecord & {
  diagnostics?: unknown;
  reason?: unknown;
}>;
type ModuleCompatibilityRejectionRecord = Readonly<DiagnosticCarrierRecord & {
  compatible?: unknown;
}>;
type ProgramExecutionPlanDiagnosticRecord = Readonly<DiagnosticCarrierRecord & {
  moduleCompatibility?: unknown;
  executionMode?: unknown;
}>;
type BindingPlanDiagnosticRecord = DiagnosticCarrierRecord;
type ModuleBindingPlanDiagnosticRecord = Readonly<BindingPlanDiagnosticRecord & {
  moduleBindings?: unknown;
}>;

function isRecord(value: unknown): value is UnknownRecord {
  return value !== null && typeof value === "object";
}

function diagnosticMessage(diagnostics: unknown) {
  if (!Array.isArray(diagnostics) || diagnostics.length === 0) return null;
  const diagnostic = diagnostics[0];
  if (!isRecord(diagnostic)) return null;
  const diagnosticRecord = diagnostic as DiagnosticRecord;
  return typeof diagnosticRecord.message === "string" && diagnosticRecord.message.length > 0
    ? diagnosticRecord.message
    : null;
}

export function programExecutionPlanRejectionReason(plan: ProgramExecutionPlanDiagnosticRecord) {
  if (isRecord(plan.moduleCompatibility)) {
    const compatibility = plan.moduleCompatibility as ModuleCompatibilityRejectionRecord;
    if (compatibility.compatible !== true) {
      return diagnosticMessage(compatibility.diagnostics) || "module is incompatible with this Program";
    }
  }
  if (typeof plan.reason === "string" && plan.reason.length > 0) return plan.reason;
  const message = diagnosticMessage(plan.diagnostics);
  if (message) return message;
  if (typeof plan.executionMode === "string" && plan.executionMode.length > 0) {
    return `Program execution mode is ${plan.executionMode}`;
  }
  return "Program is not executable";
}

export function acceptsProgramExecutionPlan(plan: unknown): plan is ProgramExecutionPlan {
  return isRecord(plan) &&
    plan.kind === "zgml.program.execution-plan" &&
    typeof plan.signature === "string" &&
    plan.signature.length > 0 &&
    plan.canExecute === true &&
    plan.acceptsModule !== false;
}

export function requireProgramExecutionPlan(plan: unknown): ProgramExecutionPlan {
  if (acceptsProgramExecutionPlan(plan)) return plan;
  const reason = isRecord(plan) ? programExecutionPlanRejectionReason(plan) : "value is not a Program execution plan";
  throw new Error(`ProgramExecutionPlan is not executable: ${reason}`);
}

export const assertProgramExecutionPlan = requireProgramExecutionPlan;
export const assert_program_execution_plan = requireProgramExecutionPlan;

export function matchesProgramExecutionPlanSignature(plan: unknown, signature: unknown): boolean {
  return isRecord(plan) &&
    plan.kind === "zgml.program.execution-plan" &&
    typeof signature === "string" &&
    signature.length > 0 &&
    plan.signature === signature;
}

export function acceptsSessionExecutionPlan(plan: unknown): plan is SessionExecutionPlan {
  return isRecord(plan) &&
    plan.kind === "zgml.session.execution-plan" &&
    typeof plan.signature === "string" &&
    plan.signature.length > 0 &&
    plan.accepted === true &&
    plan.canExecute === true;
}

export function requireSessionExecutionPlan(plan: unknown): SessionExecutionPlan {
  if (acceptsSessionExecutionPlan(plan)) return plan;
  if (isRecord(plan) && isRecord(plan.compatibility)) {
    requireCanExecuteStepParamsCompatibility(plan.compatibility);
  }
  throw new Error("SessionExecutionPlan is not executable: value is not a Session execution plan");
}

export const assertSessionExecutionPlan = requireSessionExecutionPlan;
export const assert_session_execution_plan = requireSessionExecutionPlan;

export function matchesSessionExecutionPlanSignature(plan: unknown, signature: unknown): boolean {
  return isRecord(plan) &&
    plan.kind === "zgml.session.execution-plan" &&
    typeof signature === "string" &&
    signature.length > 0 &&
    plan.signature === signature;
}

function bindingPlanReason(plan: BindingPlanDiagnosticRecord, fallback: string) {
  if (typeof plan.reason === "string" && plan.reason.length > 0) return plan.reason;
  const message = diagnosticMessage(plan.diagnostics);
  return message || fallback;
}

export function programBindingPlanRejectionReason(plan: BindingPlanDiagnosticRecord) {
  return bindingPlanReason(plan, "Program bindings are rejected");
}

export function moduleBindingPlanRejectionReason(plan: ModuleBindingPlanDiagnosticRecord) {
  if (plan.moduleBindings !== true) {
    return "bindings are not ModuleBindings from nn.bindParameters(...) or nn.placeParameters(...)";
  }
  return bindingPlanReason(plan, "ModuleBindings are not supported by the native Program compiler");
}

export function acceptsProgramBindingPlan(plan: unknown): plan is ProgramBindingPlan {
  return isRecord(plan) &&
    plan.kind === "zgml.program.binding-plan" &&
    typeof plan.signature === "string" &&
    plan.signature.length > 0 &&
    plan.accepted === true &&
    plan.canBind === true;
}

export function requireProgramBindingPlan(plan: unknown): ProgramBindingPlan {
  if (acceptsProgramBindingPlan(plan)) return plan;
  const reason = isRecord(plan)
    ? programBindingPlanRejectionReason(plan)
    : "value is not a Program binding plan";
  throw new Error(`ProgramBindingPlan is not bindable: ${reason}`);
}

export const assertProgramBindingPlan = requireProgramBindingPlan;
export const assert_program_binding_plan = requireProgramBindingPlan;

export function matchesProgramBindingPlanSignature(plan: unknown, signature: unknown): boolean {
  return isRecord(plan) &&
    plan.kind === "zgml.program.binding-plan" &&
    typeof signature === "string" &&
    signature.length > 0 &&
    plan.signature === signature;
}

export function acceptsModuleBindingPlan(plan: unknown): plan is ModuleBindingPlan {
  return isRecord(plan) &&
    plan.kind === "zgml.nn.module-bindings-plan" &&
    typeof plan.signature === "string" &&
    plan.signature.length > 0 &&
    plan.moduleBindings === true &&
    plan.supported !== false;
}

export function requireModuleBindingPlan(plan: unknown): ModuleBindingPlan {
  if (acceptsModuleBindingPlan(plan)) return plan;
  const reason = isRecord(plan)
    ? moduleBindingPlanRejectionReason(plan)
    : "value is not an nn ModuleBindings plan";
  throw new Error(`ModuleBindingPlan is not bindable: ${reason}`);
}

export const assertModuleBindingPlan = requireModuleBindingPlan;
export const assert_module_binding_plan = requireModuleBindingPlan;

export function matchesModuleBindingPlanSignature(plan: unknown, signature: unknown): boolean {
  return isRecord(plan) &&
    plan.kind === "zgml.nn.module-bindings-plan" &&
    typeof signature === "string" &&
    signature.length > 0 &&
    plan.signature === signature;
}

export function acceptsModuleCompilePlan(plan: unknown): plan is ModuleCompileExplanation {
  return isRecord(plan) &&
    plan.kind === "zgml.nn.compile-explanation" &&
    typeof plan.signature === "string" &&
    plan.signature.length > 0 &&
    plan.supported === true;
}

export function requireModuleCompilePlan(plan: unknown): ModuleCompileExplanation {
  if (acceptsModuleCompilePlan(plan)) return plan;
  const reason = isRecord(plan)
    ? compileSupportRejectionReason(plan, { fallbackReason: "Module compile plan is not supported by the native Program compiler" })
    : "value is not an nn Module compile plan";
  throw new Error(`ModuleCompilePlan is not compilable: ${reason}`);
}

export const assertModuleCompilePlan = requireModuleCompilePlan;
export const assert_module_compile_plan = requireModuleCompilePlan;

export function matchesModuleCompilePlanSignature(plan: unknown, signature: unknown): boolean {
  return isRecord(plan) &&
    plan.kind === "zgml.nn.compile-explanation" &&
    typeof signature === "string" &&
    signature.length > 0 &&
    plan.signature === signature;
}
