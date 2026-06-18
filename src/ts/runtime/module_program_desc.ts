"use strict";

import {
  kernelPlanInputShape,
  kernelPlanOutputShape,
  kernelPlanInputLen,
  kernelPlanOutputLen,
  type InternalKernelPlan,
} from "./kernel_plan.js";
import {
  nativeModuleOpDescAbiFields,
  type NativeModuleOpDesc,
} from "./native_kernel_contract.js";

type CompiledModuleSpec = {
  readonly kind?: unknown;
  readonly kernelPlan?: InternalKernelPlan;
};

export type ModuleProgramDesc = {
  readonly inputShape: readonly number[];
  readonly outputShape?: readonly number[];
  readonly inputLen?: number;
  readonly outputLen?: number;
  readonly weightsLen?: number;
  readonly biasLen?: number;
  readonly ops: readonly NativeModuleOpDesc[];
};

export type ModuleProgramPackedDesc<TInputShape = unknown, TOpWords = unknown> = Readonly<{
  inputShape: TInputShape;
  opWords: TOpWords;
  opCount: number;
}>;

export type ModuleProgramAbiDescriptorFields<TInputShape = unknown, TOpWords = unknown> = Readonly<{
  inputShape: TInputShape;
  inputRank: number;
  opWords: TOpWords;
  opCount: number;
}>;

export function moduleProgramDescFromKernelPlan(kernelPlan: InternalKernelPlan | null | undefined): Readonly<ModuleProgramDesc> {
  if (!kernelPlan || !Array.isArray(kernelPlan.nativeOps)) {
    throw new Error("module Program descriptor requires a kernelPlan with nativeOps");
  }
  const inputShape = kernelPlanInputShape(kernelPlan);
  const outputShape = kernelPlanOutputShape(kernelPlan);
  if (!inputShape || !outputShape) {
    throw new Error("module Program descriptor requires KernelPlan shape constraints");
  }
  return Object.freeze({
    inputShape: Object.freeze(inputShape.slice()),
    outputShape: Object.freeze(outputShape.slice()),
    inputLen: kernelPlanInputLen(kernelPlan),
    outputLen: kernelPlanOutputLen(kernelPlan),
    weightsLen: kernelPlan.weightsLen,
    biasLen: kernelPlan.biasLen,
    ops: Object.freeze(kernelPlan.nativeOps.map((op) => Object.freeze({ ...op }))),
  });
}

export function moduleProgramDescFromCompiledSpec(spec: CompiledModuleSpec | null | undefined): Readonly<ModuleProgramDesc> {
  if (!spec || spec.kind !== "module" || !spec.kernelPlan) {
    throw new Error("module Program compile requires a compiled module spec with a kernelPlan");
  }
  return moduleProgramDescFromKernelPlan(spec.kernelPlan);
}

function moduleProgramDescUSize(name: string, value: unknown): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${name} must be a non-negative safe integer, got ${value}`);
  }
  return value as number;
}

export function moduleProgramAbiDescriptorFields<
  TInputShape extends { readonly length: number },
  TOpWords,
>(
  packed: ModuleProgramPackedDesc<TInputShape, TOpWords>,
): ModuleProgramAbiDescriptorFields<TInputShape, TOpWords> {
  return Object.freeze({
    inputShape: packed.inputShape,
    inputRank: packed.inputShape.length,
    opWords: packed.opWords,
    opCount: packed.opCount,
  });
}

export function packModuleProgramDesc(desc: ModuleProgramDesc) {
  if (!Array.isArray(desc.inputShape) || desc.inputShape.length === 0) {
    throw new Error("module Program inputShape must be a non-empty shape array");
  }
  if (!Array.isArray(desc.ops)) {
    throw new Error("module Program ops must be an array");
  }

  const inputShape = new BigUint64Array(desc.inputShape.length);
  for (let i = 0; i < desc.inputShape.length; i += 1) {
    inputShape[i] = BigInt(moduleProgramDescUSize(`module Program inputShape[${i}]`, desc.inputShape[i]));
  }

  const opWords = desc.ops.length === 0 ? null : new BigUint64Array(desc.ops.length * 6);
  const view = opWords ? new DataView(opWords.buffer) : null;
  for (let i = 0; i < desc.ops.length; i += 1) {
    const op = nativeModuleOpDescAbiFields(desc.ops[i], `module Program ops[${i}]`);
    const base = i * 48;
    view!.setUint32(base, op.kind, true);
    view!.setUint32(base + 4, op.activation, true);
    view!.setUint32(base + 8, op.flags, true);
    view!.setUint32(base + 12, op.reserved, true);
    view!.setBigUint64(base + 16, BigInt(op.a), true);
    view!.setBigUint64(base + 24, BigInt(op.b), true);
    view!.setBigUint64(base + 32, BigInt(op.c), true);
    view!.setFloat64(base + 40, op.eps, true);
  }

  return Object.freeze({
    inputShape,
    opWords,
    opCount: desc.ops.length,
  });
}

export function createModuleProgramDescPackerHelpers() {
  return Object.freeze({
    pack: packModuleProgramDesc,
  });
}
