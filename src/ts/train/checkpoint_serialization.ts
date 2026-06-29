"use strict";

type AnyRecord = Record<string, any>;
type CheckpointTensorEntryOptions = Readonly<{
  requireDtype?: boolean;
}>;

export function createCheckpointSerializationHelpers() {
  function freezeArray<T>(values: T[]) {
    return Object.freeze(values);
  }

  function freezeRecord<T extends AnyRecord>(record: T) {
    return Object.freeze(record);
  }

  function plainRecord(value: unknown, label: string): AnyRecord {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error(`${label} must be a plain object`);
    }
    const proto = Object.getPrototypeOf(value);
    if (proto !== Object.prototype && proto !== null) {
      throw new Error(`${label} must be a plain object`);
    }
    return value as AnyRecord;
  }

  function assertJsonSafeMetadata(value: unknown, path = "checkpoint metadata", seen = new WeakSet<object>()): void {
    if (value === null) return;
    const valueType = typeof value;
    if (valueType === "string" || valueType === "boolean") return;
    if (valueType === "number") {
      if (!Number.isFinite(value)) throw new Error(`${path} number must be finite, got ${value}`);
      return;
    }
    if (valueType === "undefined" || valueType === "function" || valueType === "symbol" || valueType === "bigint") {
      throw new Error(`${path} must be JSON-safe plain data`);
    }
    if (!value || valueType !== "object") {
      throw new Error(`${path} must be JSON-safe plain data`);
    }
    if (seen.has(value)) throw new Error(`${path} must not contain cycles`);
    seen.add(value);
    if (Array.isArray(value)) {
      for (let i = 0; i < value.length; i += 1) assertJsonSafeMetadata(value[i], `${path}[${i}]`, seen);
      seen.delete(value);
      return;
    }
    const proto = Object.getPrototypeOf(value);
    if (proto !== Object.prototype && proto !== null) {
      throw new Error(`${path} must be JSON-safe plain data`);
    }
    for (const [key, item] of Object.entries(value)) {
      assertJsonSafeMetadata(item, `${path}.${key}`, seen);
    }
    seen.delete(value);
  }

  function freezeJson(value: unknown): unknown {
    if (Array.isArray(value)) {
      for (const item of value) freezeJson(item);
      return Object.freeze(value);
    }
    if (value && typeof value === "object") {
      for (const item of Object.values(value)) freezeJson(item);
      return Object.freeze(value);
    }
    return value;
  }

  function jsonMetadata(value: unknown) {
    if (value === undefined) return undefined;
    assertJsonSafeMetadata(value);
    const text = JSON.stringify(value);
    if (text === undefined) throw new Error("checkpoint metadata must be JSON-serializable");
    return freezeJson(JSON.parse(text));
  }

  function stateEntryShape(entry: AnyRecord, label: string) {
    const shape = entry && entry.shape;
    if (!Array.isArray(shape) || shape.length === 0) throw new Error(`${label} shape must be a non-empty array`);
    return freezeArray(shape.map((dim) => {
      if (!Number.isSafeInteger(dim) || dim <= 0) throw new Error(`${label} shape dimensions must be positive safe integers, got ${dim}`);
      return dim;
    }));
  }

  function stateEntryData(entry: AnyRecord, label: string) {
    const source = entry && entry.data;
    if (!source || typeof source.length !== "number") throw new Error(`${label} data must be array-like`);
    const data = Array.from(source, (value, index) => {
      if (!Number.isFinite(value)) throw new Error(`${label} data[${index}] must be finite, got ${value}`);
      return Number(value);
    });
    return freezeArray(data);
  }

  function tensorScalarCount(entry: AnyRecord) {
    if (!entry || !Array.isArray(entry.shape)) return 0;
    return entry.shape.reduce((acc: number, dim: number) => acc * dim, 1);
  }

  function shapeSignature(shape: readonly unknown[] | null | undefined) {
    return Array.isArray(shape) ? shape.map(Number).join("x") : "null";
  }

  function dataSignature(data: ArrayLike<number> | null | undefined) {
    if (!data || typeof data.length !== "number") return "len=0";
    const length = data.length;
    let sum = 0;
    for (let i = 0; i < length; i += 1) sum += Number(data[i]);
    const first = length > 0 ? Number(data[0]) : 0;
    const last = length > 0 ? Number(data[length - 1]) : 0;
    return `len=${length}:first=${first}:last=${last}:sum=${sum}`;
  }

  function checkpointTensorSignature(kind: string, fields: AnyRecord) {
    return [
      kind,
      `name=${fields.name ?? "null"}`,
      `shape=${shapeSignature(fields.shape)}`,
      `layout=${fields.layout ?? "null"}`,
      `scalars=${fields.scalarCount ?? tensorScalarCount(fields)}`,
      dataSignature(fields.data),
    ].join("|");
  }

  function checkpointTensorEntry(entry: AnyRecord, label: string, name?: string, options: CheckpointTensorEntryOptions = {}) {
    entry = plainRecord(entry, label);
    if (options.requireDtype === true && entry?.dtype !== "f32") {
      throw new Error(`${label} dtype must be f32, got ${entry?.dtype}`);
    }
    const out: AnyRecord = {
      dtype: "f32",
      shape: stateEntryShape(entry, label),
      layout: typeof entry.layout === "string" ? entry.layout : "row-major",
      data: stateEntryData(entry, label),
    };
    if (name !== undefined) out.name = name;
    out.signature = checkpointTensorSignature("checkpoint-tensor", out);
    return freezeRecord(out);
  }

  function checkpointModuleStateFromJson(source: unknown) {
    if (source === undefined) return undefined;
    const state = plainRecord(source, "checkpoint model state");
    const out: AnyRecord = {};
    for (const name of Object.keys(state)) {
      out[name] = checkpointTensorEntry(state[name], `checkpoint model ${name}`, undefined, { requireDtype: true });
    }
    return freezeRecord(out);
  }

  function checkpointOptimizerStateFromJson(source: unknown) {
    if (source === undefined) return undefined;
    const state = plainRecord(source, "checkpoint optimizer state");
    if (typeof state.kind !== "string") throw new Error("checkpoint optimizer kind must be a string");
    if (!Number.isSafeInteger(state.step) || state.step < 0) {
      throw new Error(`checkpoint optimizer step must be a non-negative safe integer, got ${state.step}`);
    }
    if (!Number.isSafeInteger(state.paramCount) || state.paramCount < 0) {
      throw new Error(`checkpoint optimizer paramCount must be a non-negative safe integer, got ${state.paramCount}`);
    }
    if (!Array.isArray(state.entries)) throw new Error("checkpoint optimizer entries must be an array");
    const names = new Set<string>();
    return freezeRecord({
      kind: state.kind,
      step: state.step,
      paramCount: state.paramCount,
      entries: freezeArray(state.entries.map((entry: AnyRecord, index: number) => {
        entry = plainRecord(entry, `checkpoint optimizer entry ${index}`);
        if (!entry || typeof entry.name !== "string") throw new Error(`checkpoint optimizer entry ${index} must have a name`);
        if (names.has(entry.name)) throw new Error(`checkpoint optimizer has duplicate entry ${entry.name}`);
        names.add(entry.name);
        return checkpointTensorEntry(entry, `checkpoint optimizer ${entry.name}`, entry.name, { requireDtype: true });
      })),
    });
  }

  function checkpointSchedulerStateFromJson(source: unknown) {
    if (source === undefined) return undefined;
    const state = plainRecord(source, "checkpoint scheduler state");
    if (state.kind !== "step-lr" && state.kind !== "exponential-lr" && state.kind !== "cosine-annealing-lr" && state.kind !== "reduce-lr-on-plateau") {
      throw new Error(`checkpoint scheduler kind must be step-lr, exponential-lr, cosine-annealing-lr, or reduce-lr-on-plateau, got ${state.kind}`);
    }
    if (!Number.isSafeInteger(state.step) || state.step < 0) {
      throw new Error(`checkpoint scheduler step must be a non-negative safe integer, got ${state.step}`);
    }
    const baseLr = Number(state.baseLr ?? state.base_lr);
    const lastLr = Number(state.lastLr ?? state.last_lr);
    if (!Number.isFinite(baseLr) || baseLr < 0) {
      throw new Error(`checkpoint scheduler baseLr must be a non-negative finite number, got ${baseLr}`);
    }
    for (const [name, value] of [["gamma", Number(state.gamma)]] as const) {
      if (!Number.isFinite(value) || value <= 0) throw new Error(`checkpoint scheduler ${name} must be a positive finite number`);
    }
    if (!Number.isFinite(lastLr) || lastLr < 0) {
      throw new Error(`checkpoint scheduler lastLr must be a non-negative finite number, got ${lastLr}`);
    }
    const stepSize = state.stepSize ?? state.step_size;
    if (stepSize !== undefined && (!Number.isSafeInteger(stepSize) || stepSize <= 0)) {
      throw new Error(`checkpoint scheduler stepSize must be a positive safe integer, got ${stepSize}`);
    }
    const tMax = state.tMax ?? state.t_max;
    if (tMax !== undefined && (!Number.isSafeInteger(tMax) || tMax <= 0)) {
      throw new Error(`checkpoint scheduler tMax must be a positive safe integer, got ${tMax}`);
    }
    const etaMin = state.etaMin ?? state.eta_min;
    if (etaMin !== undefined && (!Number.isFinite(etaMin) || etaMin < 0)) {
      throw new Error(`checkpoint scheduler etaMin must be a non-negative finite number, got ${etaMin}`);
    }
    const mode = state.mode === undefined ? undefined : String(state.mode);
    if (mode !== undefined && mode !== "min" && mode !== "max") throw new Error(`checkpoint scheduler mode must be min or max, got ${mode}`);
    const thresholdMode = state.thresholdMode ?? state.threshold_mode;
    if (thresholdMode !== undefined && thresholdMode !== "rel" && thresholdMode !== "abs") {
      throw new Error(`checkpoint scheduler thresholdMode must be rel or abs, got ${thresholdMode}`);
    }
    const plateauNumbers = [
      ["factor", state.factor, (value: number) => value > 0 && value < 1],
      ["threshold", state.threshold, (value: number) => value >= 0],
      ["minLr", state.minLr ?? state.min_lr, (value: number) => value >= 0],
      ["eps", state.eps, (value: number) => value >= 0],
      ["best", state.best, (_value: number) => true],
    ] as const;
    const plateauNumberValues = new Map<string, number>();
    for (const [name, sourceValue, valid] of plateauNumbers) {
      if (sourceValue === undefined) continue;
      const value = Number(sourceValue);
      if (!Number.isFinite(value) || !valid(value)) throw new Error(`checkpoint scheduler ${name} must be finite and valid, got ${sourceValue}`);
      plateauNumberValues.set(name, value);
    }
    const plateauIntegers = [
      ["patience", state.patience],
      ["cooldown", state.cooldown],
      ["cooldownCounter", state.cooldownCounter ?? state.cooldown_counter],
      ["badEpochs", state.badEpochs ?? state.bad_epochs],
    ] as const;
    const plateauIntegerValues = new Map<string, number>();
    for (const [name, sourceValue] of plateauIntegers) {
      if (sourceValue === undefined) continue;
      const value = Number(sourceValue);
      if (!Number.isSafeInteger(value) || value < 0) throw new Error(`checkpoint scheduler ${name} must be a non-negative safe integer, got ${sourceValue}`);
      plateauIntegerValues.set(name, value);
    }
    const out: AnyRecord = {
      kind: state.kind,
      step: state.step,
      baseLr,
      base_lr: baseLr,
      lastLr,
      last_lr: lastLr,
      gamma: state.gamma,
      optimizerKind: typeof state.optimizerKind === "string" ? state.optimizerKind : null,
    };
    if (stepSize !== undefined) {
      out.stepSize = stepSize;
      out.step_size = stepSize;
    }
    if (tMax !== undefined) {
      out.tMax = tMax;
      out.t_max = tMax;
    }
    if (etaMin !== undefined) {
      out.etaMin = etaMin;
      out.eta_min = etaMin;
    }
    if (mode !== undefined) out.mode = mode;
    if (thresholdMode !== undefined) {
      out.thresholdMode = thresholdMode;
      out.threshold_mode = thresholdMode;
    }
    for (const [name, value] of plateauNumberValues) {
      out[name] = value;
      if (name === "minLr") out.min_lr = value;
    }
    for (const [name, value] of plateauIntegerValues) {
      out[name] = value;
      if (name === "cooldownCounter") out.cooldown_counter = value;
      if (name === "badEpochs") out.bad_epochs = value;
    }
    out.signature = [
      "lr-scheduler-state",
      `kind=${out.kind}`,
      `step=${out.step}`,
      `baseLr=${out.baseLr}`,
      `lastLr=${out.lastLr}`,
      `gamma=${out.gamma}`,
      `stepSize=${out.stepSize ?? "null"}`,
      `tMax=${out.tMax ?? "null"}`,
      `etaMin=${out.etaMin ?? "null"}`,
      `mode=${out.mode ?? "null"}`,
      `factor=${out.factor ?? "null"}`,
      `patience=${out.patience ?? "null"}`,
      `threshold=${out.threshold ?? "null"}`,
      `thresholdMode=${out.thresholdMode ?? "null"}`,
      `cooldown=${out.cooldown ?? "null"}`,
      `cooldownCounter=${out.cooldownCounter ?? "null"}`,
      `minLr=${out.minLr ?? "null"}`,
      `eps=${out.eps ?? "null"}`,
      `best=${out.best ?? "null"}`,
      `badEpochs=${out.badEpochs ?? "null"}`,
      `optimizer=${out.optimizerKind ?? "null"}`,
    ].join("|");
    return freezeRecord(out);
  }

  function checkpointTensorInspection(name: string, entry: AnyRecord, index: number) {
    const inspection: AnyRecord = {
      name,
      index,
      scalarCount: tensorScalarCount(entry),
      shape: freezeArray(Array.isArray(entry?.shape) ? Array.from(entry.shape) : []),
      layout: typeof entry?.layout === "string" ? entry.layout : null,
    };
    inspection.signature = checkpointTensorSignature("checkpoint-tensor-inspection", inspection);
    return freezeRecord(inspection);
  }

  function checkpointInfo(entries: readonly AnyRecord[], nameOrIndex: unknown, label: string) {
    if (typeof nameOrIndex === "number") {
      if (!Number.isInteger(nameOrIndex) || nameOrIndex < 0) {
        throw new Error(`checkpoint ${label} index must be a non-negative integer`);
      }
      return entries[nameOrIndex] ?? null;
    }
    if (typeof nameOrIndex === "string") {
      return entries.find((entry) => entry.name === nameOrIndex) ?? null;
    }
    throw new Error(`checkpoint ${label} lookup requires a name or index`);
  }

  return Object.freeze({
    freezeArray,
    freezeRecord,
    plainRecord,
    jsonMetadata,
    checkpointTensorEntry,
    checkpointModuleStateFromJson,
    checkpointOptimizerStateFromJson,
    checkpointSchedulerStateFromJson,
    checkpointTensorInspection,
    checkpointInfo,
    tensorScalarCount,
  });
}
