"use strict";

import {
  freezeCompileDiagnostics,
} from "../runtime/compile_diagnostics.js";
import type { LoadStateDictOptions } from "../public_api.js";
import { optimizerStateSnapshotSignature } from "./optimizer_snapshot.js";

type UnknownRecord = Record<string, unknown>;

export type ModuleStateTensor = {
  data: Float32Array;
  length: number;
  shape: readonly number[];
  grad?: Float32Array | null;
  requiresGrad?: boolean;
  requires_grad?: boolean;
  zeroGrad(options?: ModuleStateOptionsRecord): void;
};

type ModuleStateTensorConstructor = new (
  data: Float32Array,
  shape?: readonly number[],
  options?: ModuleStateOptionsRecord,
) => ModuleStateTensor;

type ModuleStateOptionsRecord = Readonly<Record<string, unknown>>;

export type ModuleParameter = {
  name: string;
  data: Float32Array;
  grad?: Float32Array | null;
  layout?: string;
  tensor: ModuleStateTensor;
  requiresGrad?: boolean;
  requires_grad?: boolean;
};

export type ModuleStateEntry = Readonly<{
  shape: readonly number[];
  layout: string;
  data: Float32Array;
  signature: string;
}>;

type ModuleCompileSupportExtra = UnknownRecord & {
  diagnostics?: unknown;
};

type ModuleStateDictRecord = Record<string, ModuleStateEntry>;

type OptimizerStateEntry = UnknownRecord & {
  name: string;
  shape?: readonly number[];
  layout?: string;
  data?: unknown;
};

type OptimizerStateSnapshot = UnknownRecord & {
  kind?: unknown;
  paramCount?: unknown;
  step?: unknown;
  entries?: unknown;
};

export type ModuleStateHelpersOptions = Readonly<{
  Tensor: ModuleStateTensorConstructor;
  f32WithLength: (value: unknown, length: number, label: string) => Float32Array;
  defaultLayout?: string;
  optionalParameterFields?: boolean;
}>;

export function createModuleStateHelpers(options: ModuleStateHelpersOptions) {
  const TensorClass = options.Tensor;
  const f32WithLength = options.f32WithLength;
  if (typeof TensorClass !== "function" || typeof f32WithLength !== "function") {
    throw new Error("module state helpers require Tensor and f32WithLength");
  }
  const defaultLayout = options.defaultLayout ?? "row-major";
  const optionalParameterFields = Boolean(options.optionalParameterFields);

  function isTensor(value: unknown): value is ModuleStateTensor {
    return value instanceof TensorClass;
  }

  function makeParameter(name: string, data: Float32Array, shape: readonly number[] = [data.length], layout = defaultLayout): ModuleParameter {
    const grad = new Float32Array(data.length);
    const tensor = new TensorClass(data, shape, { requiresGrad: true, grad });
    const param = {
      name,
      data,
      grad,
      layout,
      tensor,
    } as ModuleParameter;
    Object.defineProperties(param, {
      requiresGrad: {
        enumerable: true,
        get() {
          return Boolean(tensor.requiresGrad);
        },
        set(value) {
          setTensorRequiresGrad(tensor, Boolean(value));
        },
      },
      requires_grad: {
        enumerable: true,
        get() {
          return Boolean(tensor.requiresGrad);
        },
        set(value) {
          setTensorRequiresGrad(tensor, Boolean(value));
        },
      },
    });
    return param;
  }

  function parameterView(prefix: string, param: ModuleParameter): ModuleParameter {
    if (!prefix) return param;
    const view = { name: `${prefix}.${param.name}`, data: param.data, grad: param.grad, layout: param.layout, tensor: param.tensor };
    Object.defineProperties(view, {
      requiresGrad: {
        enumerable: true,
        get() {
          return parameterRequiresGrad(param);
        },
        set(value) {
          if (param.tensor) setTensorRequiresGrad(param.tensor, Boolean(value));
        },
      },
      requires_grad: {
        enumerable: true,
        get() {
          return parameterRequiresGrad(param);
        },
        set(value) {
          if (param.tensor) setTensorRequiresGrad(param.tensor, Boolean(value));
        },
      },
    });
    return view as ModuleParameter;
  }

  function moduleCompileSupport(supported: boolean, reason: unknown, extra: ModuleCompileSupportExtra = {}) {
    const normalized = { ...extra };
    if (Array.isArray(normalized.diagnostics)) {
      normalized.diagnostics = freezeCompileDiagnostics(normalized.diagnostics);
    }
    return Object.freeze({ supported, reason, ...normalized });
  }

  function composableModuleCompileSupport(reason: unknown) {
    return moduleCompileSupport(false, reason, { composable: true });
  }

  function resolveParameters(paramsOrModule: unknown): ModuleParameter[] {
    if (isTensor(paramsOrModule)) {
      if (!paramsOrModule.grad) {
        paramsOrModule.requiresGrad = true;
        paramsOrModule.grad = new Float32Array(paramsOrModule.length);
      }
      return [{ name: "tensor", data: paramsOrModule.data, grad: paramsOrModule.grad, layout: defaultLayout, tensor: paramsOrModule }];
    }
    if (Array.isArray(paramsOrModule)) return paramsOrModule as ModuleParameter[];
    const target = paramsOrModule as UnknownRecord;
    if (target && typeof target.parameters === "function") return target.parameters();
    throw new Error("expected an nn module or parameter array");
  }

  function setTensorRequiresGrad(tensor: ModuleStateTensor, enabled: boolean) {
    tensor.requiresGrad = enabled;
    tensor.requires_grad = enabled;
  }

  function parameterRequiresGrad(param: ModuleParameter) {
    return param?.tensor ? Boolean(param.tensor.requiresGrad) : param?.requiresGrad !== false;
  }

  function parameterNames(paramsOrModule: unknown, prefix = "") {
    return Object.freeze(resolveParameters(paramsOrModule).map((param) => prefix ? `${prefix}.${param.name}` : param.name));
  }

  function frozenShape(shape: ArrayLike<number>) {
    return Object.freeze(Array.from(shape));
  }

  function shapeSignature(shape: unknown) {
    if (!shape || typeof shape !== "object" || !("length" in shape) || typeof shape.length !== "number") return "null";
    return Array.from(shape as ArrayLike<number>, Number).join("x");
  }

  function dataSignature(data: unknown) {
    if (!data || typeof data !== "object" || !("length" in data) || typeof data.length !== "number") return "len=0";
    const values = data as ArrayLike<number>;
    const length = values.length;
    let sum = 0;
    for (let i = 0; i < length; i += 1) sum += Number(values[i]);
    const first = length > 0 ? Number(values[0]) : 0;
    const last = length > 0 ? Number(values[length - 1]) : 0;
    return `len=${length}:first=${first}:last=${last}:sum=${sum}`;
  }

  function tensorStateSignature(kind: string, fields: UnknownRecord) {
    return [
      kind,
      `name=${fields.name ?? "null"}`,
      `shape=${shapeSignature(fields.shape)}`,
      `layout=${fields.layout ?? "null"}`,
      dataSignature(fields.data),
    ].join("|");
  }

  function parameterInfoEntry(param: ModuleParameter, index: number, prefix = "") {
    return Object.freeze({
      name: prefix ? `${prefix}.${param.name}` : param.name,
      index,
      scalarCount: param.data.length,
      shape: frozenShape(param.tensor.shape),
      layout: param.layout ?? defaultLayout,
      requiresGrad: parameterRequiresGrad(param),
      requires_grad: parameterRequiresGrad(param),
    });
  }

  function parameterInfos(paramsOrModule: unknown, prefix = "") {
    return Object.freeze(resolveParameters(paramsOrModule).map((param, index) => parameterInfoEntry(param, index, prefix)));
  }

  function parameterInfo(paramsOrModule: unknown, nameOrIndex: unknown, prefix = "") {
    const infos = parameterInfos(paramsOrModule, prefix);
    if (typeof nameOrIndex === "number") {
      if (!Number.isInteger(nameOrIndex) || nameOrIndex < 0) {
        throw new Error("module parameterInfo index must be a non-negative integer");
      }
      return infos[nameOrIndex] ?? null;
    }
    if (typeof nameOrIndex === "string") {
      return infos.find((info) => info.name === nameOrIndex) ?? null;
    }
    throw new Error("module parameterInfo requires a parameter name or index");
  }

  function stateDictPrefix(prefixOrOptions: unknown = "") {
    if (prefixOrOptions && typeof prefixOrOptions === "object" && !Array.isArray(prefixOrOptions)) {
      const prefix = (prefixOrOptions as ModuleStateOptionsRecord).prefix;
      return prefix === undefined || prefix === null ? "" : String(prefix);
    }
    return prefixOrOptions === undefined || prefixOrOptions === null ? "" : String(prefixOrOptions);
  }

  function stateValue(source: unknown, name: string) {
    if (source instanceof Map) return source.get(name);
    if (source && typeof source === "object") {
      return Object.prototype.hasOwnProperty.call(source, name) ? (source as UnknownRecord)[name] : undefined;
    }
    throw new Error("state dict must be an object or Map");
  }

  function stateKeys(source: unknown) {
    if (source instanceof Map) return Array.from(source.keys());
    if (source && typeof source === "object") return Object.keys(source);
    throw new Error("state dict must be an object or Map");
  }

  function moduleStateEntry(param: ModuleParameter): ModuleStateEntry {
    const entry = {
      shape: frozenShape(param.tensor.shape),
      layout: param.layout ?? defaultLayout,
      data: new Float32Array(param.data),
    } as UnknownRecord & Omit<ModuleStateEntry, "signature"> & { signature?: string };
    entry.signature = tensorStateSignature("module-state-entry", entry);
    return Object.freeze(entry) as ModuleStateEntry;
  }

  function isModuleStateEntry(value: unknown): value is ModuleStateEntry {
    return value !== null && typeof value === "object" && !isTensor(value) && "shape" in value && "data" in value;
  }

  function stateShape(value: unknown): readonly number[] | null {
    if (isModuleStateEntry(value)) return Array.from(value.shape);
    if (isTensor(value)) return value.shape;
    return null;
  }

  function stateLayout(value: unknown) {
    if (isModuleStateEntry(value) && typeof value.layout === "string") return value.layout;
    return null;
  }

  function stateData(value: unknown) {
    return isModuleStateEntry(value) ? value.data : value;
  }

  function sameShape(a: readonly number[], b: readonly number[]) {
    return a.length === b.length && a.every((dim, index) => dim === b[index]);
  }

  function shapeText(shape: readonly number[]) {
    return `[${shape.join(",")}]`;
  }

  function assertStateShape(param: ModuleParameter, value: unknown) {
    const shape = stateShape(value);
    if (shape && !sameShape(shape, param.tensor.shape)) {
      throw new Error(`state dict ${param.name} shape must be ${shapeText(param.tensor.shape)}, got ${shapeText(shape)}`);
    }
  }

  function assertStateLayout(param: ModuleParameter, value: unknown) {
    const layout = stateLayout(value);
    if (layout && param.layout && layout !== param.layout) {
      throw new Error(`state dict ${param.name} layout must be ${param.layout}, got ${layout}`);
    }
  }

  function stateDict(paramsOrModule: unknown, prefixOrOptions: unknown = "") {
    const prefix = stateDictPrefix(prefixOrOptions);
    const out: ModuleStateDictRecord = {};
    for (const param of resolveParameters(paramsOrModule)) {
      const name = prefix ? `${prefix}.${param.name}` : param.name;
      out[name] = moduleStateEntry(param);
    }
    return Object.freeze(out);
  }

  function loadStateDict(paramsOrModule: unknown, source: unknown, options: ModuleStateOptionsRecord = {}) {
    const strict = options.strict !== false;
    const validateOnly = options.validateOnly === true;
    const prefix = options.prefix ?? "";
    const params = resolveParameters(paramsOrModule);
    const stateName = (param: ModuleParameter) => prefix ? `${prefix}.${param.name}` : param.name;
    const expected = new Set(params.map(stateName));
    const pending: Array<{ param: ModuleParameter; data: Float32Array }> = [];
    if (strict) {
      for (const key of stateKeys(source)) {
        if (!expected.has(key)) throw new Error(`state dict has unexpected parameter ${key}`);
      }
    }
    for (const param of params) {
      const name = stateName(param);
      const value = stateValue(source, name);
      if (value === undefined) {
        if (strict) throw new Error(`state dict is missing parameter ${name}`);
        continue;
      }
      assertStateShape(param, value);
      assertStateLayout(param, value);
      const data = f32WithLength(stateData(value), param.data.length, `state dict ${name}`);
      pending.push({ param, data });
    }
    for (const { param, data } of pending) {
      if (!validateOnly) param.data.set(data);
    }
    return paramsOrModule;
  }

  function zeroGrad(paramsOrModule: unknown, options: ModuleStateOptionsRecord = {}) {
    const setToNone = options.setToNone === true || options.set_to_none === true;
    if (isTensor(paramsOrModule)) {
      paramsOrModule.zeroGrad(options);
      return;
    }
    for (const param of resolveParameters(paramsOrModule)) {
      if (setToNone) {
        param.grad = null;
        if (param.tensor) param.tensor.grad = null;
        continue;
      }
      if (optionalParameterFields) {
        if (param.grad) param.grad.fill(0);
        if (param.tensor) param.tensor.zeroGrad(options);
      } else {
        if (!param.grad) param.grad = new Float32Array(param.data.length);
        param.grad.fill(0);
        param.tensor.zeroGrad(options);
      }
    }
  }

  function setRequiresGrad(paramsOrModule: unknown, requiresGrad = true) {
    const enabled = Boolean(requiresGrad);
    if (isTensor(paramsOrModule)) {
      setTensorRequiresGrad(paramsOrModule, enabled);
      return paramsOrModule;
    }
    for (const param of resolveParameters(paramsOrModule)) {
      if (param.tensor) setTensorRequiresGrad(param.tensor, enabled);
      if (enabled && !param.grad) param.grad = new Float32Array(param.data.length);
    }
    return paramsOrModule;
  }

  function finiteConfigNumber(config: ModuleStateOptionsRecord, name: string, fallback: number, check: (value: number) => boolean) {
    const value = Number(config[name] ?? fallback);
    if (!Number.isFinite(value) || !check(value)) {
      throw new Error(`optimizer ${name} has invalid value ${value}`);
    }
    return value;
  }

  function optimizerStateEntry(name: string, param: ModuleParameter, data: Float32Array) {
    const entry = {
      name,
      shape: frozenShape(param.tensor.shape),
      layout: param.layout ?? defaultLayout,
      data: new Float32Array(data),
    } as OptimizerStateEntry & { signature?: string };
    entry.signature = tensorStateSignature("optimizer-state-entry", entry);
    return Object.freeze(entry);
  }

  function optimizerStateDict(kind: string, params: readonly unknown[], entries: readonly unknown[], step = 0) {
    const frozenEntries = Object.freeze(entries);
    const snapshot = {
      kind,
      step,
      paramCount: params.length,
      entries: frozenEntries,
    } as OptimizerStateSnapshot & { signature?: string };
    snapshot.signature = optimizerStateSnapshotSignature(snapshot);
    return Object.freeze(snapshot);
  }

  function optimizerStateEntries(source: OptimizerStateSnapshot, expectedKind: string, expectedParamCount: number, options: LoadStateDictOptions = {}) {
    const strict = options.strict !== false;
    if (!source || typeof source !== "object") throw new Error("optimizer state dict must be an object");
    if (strict && source.kind !== expectedKind) {
      throw new Error(`optimizer state kind must be ${expectedKind}, got ${source.kind}`);
    }
    if (strict && source.paramCount !== expectedParamCount) {
      throw new Error(`optimizer state paramCount must be ${expectedParamCount}, got ${source.paramCount}`);
    }
    const entries = Array.isArray(source.entries) ? source.entries : [];
    const byName = new Map();
    for (const entry of entries) {
      if (!entry || typeof entry.name !== "string") throw new Error("optimizer state entries must have names");
      if (byName.has(entry.name)) throw new Error(`optimizer state has duplicate entry ${entry.name}`);
      byName.set(entry.name, entry);
    }
    return { byName, entries, strict };
  }

  function optimizerStepFromState(source: OptimizerStateSnapshot, fallback: number, strict: boolean) {
    if (source.step === undefined) {
      if (strict) throw new Error("optimizer state is missing step");
      return fallback;
    }
    const step = Number(source.step);
    if (!Number.isSafeInteger(step) || step < 0) throw new Error(`optimizer state step must be a non-negative safe integer, got ${source.step}`);
    return step;
  }

  function loadOptimizerTensorState(byName: Map<string, OptimizerStateEntry>, expectedNames: Set<string>, name: string, param: ModuleParameter, target: Float32Array, strict: boolean) {
    const entry = byName.get(name);
    if (!entry) {
      if (strict) throw new Error(`optimizer state is missing ${name}`);
      return null;
    }
    assertStateShape(param, entry);
    assertStateLayout(param, entry);
    const data = f32WithLength(stateData(entry), target.length, `optimizer state ${name}`);
    expectedNames.add(name);
    return { target, data };
  }

  function rejectUnexpectedOptimizerState(entries: readonly OptimizerStateEntry[], expectedNames: Set<string>) {
    for (const entry of entries) {
      if (!expectedNames.has(entry.name)) throw new Error(`optimizer state has unexpected entry ${entry.name}`);
    }
  }

  return {
    makeParameter,
    parameterView,
    parameterRequiresGrad,
    moduleCompileSupport,
    composableModuleCompileSupport,
    resolveParameters,
    parameterNames,
    parameterInfos,
    parameterInfo,
    stateKeys,
    stateData,
    assertStateShape,
    assertStateLayout,
    stateDict,
    loadStateDict,
    zeroGrad,
    setRequiresGrad,
    finiteConfigNumber,
    optimizerStateEntry,
    optimizerStateDict,
    optimizerStateEntries,
    optimizerStepFromState,
    loadOptimizerTensorState,
    rejectUnexpectedOptimizerState,
  };
}
