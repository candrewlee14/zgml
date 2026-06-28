"use strict";

import { tsProductManifestPolicy } from "./internal/product_manifest.js";

type AnyRecord = Record<string, any>;

export type {
  CollatedDataLoader,
  DataBatchOptions,
  DataBatches,
  BatchSampler,
  DataCollateFn,
  DataCollateOptions,
  DefaultCollateOptions,
  DefaultCollateTensor,
  Dataset,
  DataLoader,
  DataNamespace,
  PublicDataNamespace,
  RandomSampler,
  Sampler,
  SequentialSampler,
  DataSplitOptions,
  TensorDataset,
  TensorDatasetBatch,
  TensorDatasetBatchShape,
  TensorDatasetMapper,
  TensorDatasetSample,
  TensorShapeTail,
} from "./public_api.js";

export type DataNamespaceOptions<TTensor> = Readonly<{
  tensor: (data: unknown, shape?: unknown, options?: unknown) => TTensor;
  stack: (tensors: readonly TTensor[], dim?: number) => TTensor;
}>;

function requireNonNegativeIndex(index: unknown, length: number, label: string) {
  if (!Number.isSafeInteger(index)) throw new Error(`${label} index must be a safe integer, got ${index}`);
  const value = index as number;
  if (value < 0 || value >= length) throw new Error(`${label} index ${value} is out of range for dataset length ${length}`);
  return value;
}

function requireDataset(value: unknown, label: string) {
  const dataset = value as AnyRecord;
  if (!dataset || typeof dataset !== "object" || typeof dataset.sample !== "function" || typeof dataset.batch !== "function") {
    throw new Error(`${label} requires a dataset with sample(index) and batch(indices)`);
  }
  datasetLength(dataset, label);
  return dataset;
}

function datasetLength(dataset: AnyRecord, label: string) {
  let rawLength: unknown;
  try {
    rawLength = dataset.length;
  } catch {
    rawLength = undefined;
  }
  const value = typeof rawLength === "number"
    ? rawLength
    : typeof dataset.__len__ === "function"
      ? dataset.__len__()
      : typeof dataset.len === "function"
        ? dataset.len()
        : undefined;
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${label} dataset length must be a non-negative safe integer, got ${value}`);
  }
  return value as number;
}

function samplerLength(sampler: AnyRecord, label: string) {
  let rawLength: unknown;
  try {
    rawLength = sampler.length;
  } catch {
    rawLength = undefined;
  }
  const value = typeof rawLength === "number"
    ? rawLength
    : typeof sampler.__len__ === "function"
      ? sampler.__len__()
      : typeof sampler.len === "function"
        ? sampler.len()
        : undefined;
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${label} sampler length must be a non-negative safe integer, got ${value}`);
  }
  return value as number;
}

function requireBatchSize(value: unknown) {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    throw new Error(`data.batches batchSize must be a positive safe integer, got ${value}`);
  }
  return value as number;
}

function inferLeadingLength(value: unknown, label: string) {
  const record = value as AnyRecord;
  if (!record || typeof record !== "object" || !Array.isArray(record.shape) || record.shape.length === 0) {
    throw new Error(`${label} requires a tensor-like value with a non-empty shape`);
  }
  const length = record.shape[0];
  if (!Number.isSafeInteger(length) || length < 0) {
    throw new Error(`${label} leading dimension must be a non-negative safe integer, got ${length}`);
  }
  return length;
}

function selectRow<TTensor>(value: unknown, index: number, label: string): TTensor {
  const record = value as AnyRecord;
  if (!record || typeof record.select !== "function") {
    throw new Error(`${label} requires tensor-like values with select(dim, index)`);
  }
  return record.select(0, index) as TTensor;
}

function createSeededRng(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function rngFromOptions(options: AnyRecord) {
  if (typeof options.rng === "function") return options.rng as () => number;
  if (options.seed !== undefined) {
    if (!Number.isSafeInteger(options.seed)) throw new Error(`data.batches seed must be a safe integer, got ${options.seed}`);
    return createSeededRng(options.seed);
  }
  return Math.random;
}

function shuffledIndices(length: number, options: AnyRecord) {
  const indices = Array.from({ length }, (_value, index) => index);
  if (!options.shuffle) return indices;
  const rng = rngFromOptions(options);
  for (let i = indices.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }
  return indices;
}

function sequentialIndices(length: number) {
  return Object.freeze(Array.from({ length }, (_value, index) => index));
}

function indexSamplerIndices(samplerValue: unknown, length: number, label: string) {
  if (!samplerValue) return shuffledIndices(length, {});
  if (typeof (samplerValue as Iterable<unknown>)[Symbol.iterator] !== "function") {
    throw new Error(`${label} sampler must be iterable`);
  }
  return iterableArray(samplerValue, `${label} sampler`).map((index) => requireNonNegativeIndex(index, length, `${label} sampler`));
}

function batchCount(length: number, batchSize: number, dropLast: boolean) {
  if (length === 0) return 0;
  return dropLast ? Math.floor(length / batchSize) : Math.ceil(length / batchSize);
}

function splitLengths(value: unknown, datasetLength: number) {
  const values = iterableArray(value, "data.randomSplit lengths");
  if (values.length === 0) throw new Error("data.randomSplit lengths must be a non-empty iterable");
  const lengths = values.map((length, index) => {
    if (!Number.isSafeInteger(length) || (length as number) < 0) {
      throw new Error(`data.randomSplit length at position ${index} must be a non-negative safe integer, got ${length}`);
    }
    return length as number;
  });
  const total = lengths.reduce((sum, length) => sum + length, 0);
  if (total !== datasetLength) {
    throw new Error(`data.randomSplit lengths must sum to dataset length ${datasetLength}, got ${total}`);
  }
  return lengths;
}

function iterableArray(value: unknown, label: string) {
  if (!value || typeof (value as Iterable<unknown>)[Symbol.iterator] !== "function") {
    throw new Error(`${label} requires an iterable`);
  }
  return Array.from(value as Iterable<unknown>);
}

function indexValues(value: unknown, length: number, label: string, allowEmpty = false) {
  const values = iterableArray(value, label);
  if (!allowEmpty && values.length === 0) throw new Error(`${label} requires a non-empty indices iterable`);
  return values.map((index) => requireNonNegativeIndex(index, length, label));
}

export function createDataNamespace<TTensor>(options: DataNamespaceOptions<TTensor>) {
  const stack = options && options.stack;
  if (typeof stack !== "function") throw new Error("data namespace requires stack");

  class DatasetClass {
    kind = "zgml.data.dataset";

    len() {
      const ownLength = (this as AnyRecord).length;
      if (Number.isSafeInteger(ownLength) && ownLength >= 0) return ownLength;
      throw new Error("data.Dataset subclasses must provide a non-negative length property/getter or override __len__");
    }

    __len__() {
      return this.len();
    }

    size() {
      return this.len();
    }

    sample(index: number): AnyRecord {
      const item = (this as AnyRecord).__getitem__;
      if (typeof item === "function" && item !== DatasetClass.prototype.__getitem__) {
        return item.call(this, index);
      }
      throw new Error("data.Dataset subclasses must implement sample(index) or __getitem__(index)");
    }

    get(index: number) {
      return this.sample(index);
    }

    at(index: number) {
      return this.sample(index);
    }

    __getitem__(index: number) {
      return this.sample(index);
    }

    batch(indices: Iterable<number>, batchIndex = 0) {
      const length = this.len();
      const rows = indexValues(indices, length, "data.Dataset.batch");
      const samples = rows.map((row) => this.sample(row));
      const first = samples[0] as AnyRecord | undefined;
      const out: AnyRecord = {
        kind: "zgml.data.batch",
        batchIndex,
        indices: Object.freeze(rows.slice()),
      };
      if (!first) return Object.freeze(out);
      if ("input" in first) out.input = stack(samples.map((sample) => (sample as AnyRecord).input as TTensor), 0);
      if ("target" in first) out.target = stack(samples.map((sample) => (sample as AnyRecord).target as TTensor), 0);
      for (const key of Object.keys(first)) {
        if (key === "kind" || key === "index" || key === "input" || key === "target") continue;
        out[key] = Object.freeze(samples.map((sample) => (sample as AnyRecord)[key]));
      }
      return Object.freeze(out);
    }

    *__iter__() {
      const length = this.len();
      for (let index = 0; index < length; index += 1) yield this.sample(index);
    }

    [Symbol.iterator]() {
      return this.__iter__();
    }
  }

  function tensorDataset(input: TTensor, target?: TTensor) {
    const inputLength = inferLeadingLength(input, "data.tensorDataset input");
    if (target !== undefined) {
      const targetLength = inferLeadingLength(target, "data.tensorDataset target");
      if (targetLength !== inputLength) {
        throw new Error(`data.tensorDataset target length ${targetLength} must match input length ${inputLength}`);
      }
    }

    function sample(index: unknown) {
      const row = requireNonNegativeIndex(index, inputLength, "data.tensorDataset.sample");
      const entry: AnyRecord = {
        kind: "zgml.data.sample",
        index: row,
        input: selectRow<TTensor>(input, row, "data.tensorDataset input"),
      };
      if (target !== undefined) entry.target = selectRow<TTensor>(target, row, "data.tensorDataset target");
      return Object.freeze(entry);
    }

    function batch(indices: Iterable<number>, batchIndex = 0) {
      const rows = indexValues(indices, inputLength, "data.tensorDataset.batch");
      const out: AnyRecord = {
        kind: "zgml.data.batch",
        batchIndex,
        indices: Object.freeze(rows.slice()),
        input: stack(rows.map((row) => selectRow<TTensor>(input, row, "data.tensorDataset input")), 0),
      };
      if (target !== undefined) {
        out.target = stack(rows.map((row) => selectRow<TTensor>(target, row, "data.tensorDataset target")), 0);
      }
      return Object.freeze(out);
    }

    function* iterator() {
      for (let index = 0; index < inputLength; index += 1) yield sample(index);
    }

    return Object.freeze({
      kind: "zgml.data.tensor-dataset",
      length: inputLength,
      input,
      target,
      len: () => inputLength,
      __len__: () => inputLength,
      size: () => inputLength,
      sample,
      get: sample,
      at: sample,
      __getitem__: sample,
      batch,
      __iter__: iterator,
      [Symbol.iterator]: iterator,
    });
  }

  class TensorDatasetClass {
    constructor(input: TTensor, target?: TTensor) {
      return tensorDataset(input, target);
    }
  }

  function defaultCollate(samplesValue: unknown, collateOptions: AnyRecord = {}) {
    const samples = iterableArray(samplesValue, "data.defaultCollate samples") as AnyRecord[];
    if (samples.length === 0) throw new Error("data.defaultCollate requires at least one sample");
    const firstHasTarget = samples[0]?.target !== undefined;
    const inputs: TTensor[] = [];
    const targets: TTensor[] = [];
    const indices: number[] = [];
    for (let offset = 0; offset < samples.length; offset += 1) {
      const sample = samples[offset];
      if (!sample || typeof sample !== "object") throw new Error(`data.defaultCollate samples[${offset}] must be a sample object`);
      if (sample.input === undefined) throw new Error(`data.defaultCollate samples[${offset}].input is required`);
      const targetPresent = sample.target !== undefined;
      if (targetPresent !== firstHasTarget) throw new Error("data.defaultCollate samples must consistently provide target");
      inputs.push(sample.input as TTensor);
      if (targetPresent) targets.push(sample.target as TTensor);
      const index = sample.index;
      indices.push(Number.isSafeInteger(index) && index >= 0 ? Number(index) : offset);
    }
    const optionIndices = collateOptions.indices ?? collateOptions.sampleIndices ?? collateOptions.sample_indices;
    const batchIndices = optionIndices === undefined
      ? Object.freeze(indices)
      : Object.freeze(indexValues(optionIndices, Number.MAX_SAFE_INTEGER, "data.defaultCollate indices", true));
    const batchIndexOption = collateOptions.batchIndex ?? collateOptions.batch_index ?? 0;
    if (!Number.isSafeInteger(batchIndexOption) || (batchIndexOption as number) < 0) {
      throw new Error(`data.defaultCollate batchIndex must be a non-negative safe integer, got ${batchIndexOption}`);
    }
    const out: AnyRecord = {
      kind: "zgml.data.batch",
      batchIndex: batchIndexOption as number,
      indices: batchIndices,
      input: stack(inputs, 0),
    };
    if (firstHasTarget) out.target = stack(targets, 0);
    return Object.freeze(out);
  }

  class SequentialSamplerClass {
    readonly dataSource: AnyRecord;
    readonly length: number;
    readonly kind = "zgml.data.sequential-sampler";

    constructor(dataSource: AnyRecord) {
      this.dataSource = requireDataset(dataSource, "data.SequentialSampler");
      this.length = datasetLength(this.dataSource, "data.SequentialSampler");
    }

    len() { return this.length; }
    __len__() { return this.length; }
    size() { return this.length; }

    *[Symbol.iterator]() {
      for (let index = 0; index < this.length; index += 1) yield index;
    }

    __iter__() {
      return this[Symbol.iterator]();
    }
  }

  class RandomSamplerClass {
    readonly dataSource: AnyRecord;
    readonly length: number;
    readonly replacement: boolean;
    readonly numSamples: number;
    readonly seed: number | undefined;
    readonly rng: (() => number) | undefined;
    readonly kind = "zgml.data.random-sampler";

    constructor(dataSource: AnyRecord, samplerOptions: AnyRecord = {}) {
      this.dataSource = requireDataset(dataSource, "data.RandomSampler");
      this.length = datasetLength(this.dataSource, "data.RandomSampler");
      this.replacement = Boolean(samplerOptions.replacement);
      const rawNumSamples = samplerOptions.numSamples ?? samplerOptions.num_samples ?? this.length;
      if (!Number.isSafeInteger(rawNumSamples) || (rawNumSamples as number) < 0) {
        throw new Error(`data.RandomSampler numSamples must be a non-negative safe integer, got ${rawNumSamples}`);
      }
      if (this.replacement && this.length === 0 && (rawNumSamples as number) > 0) {
        throw new Error("data.RandomSampler cannot sample with replacement from an empty dataset");
      }
      if (!this.replacement && (rawNumSamples as number) > this.length) {
        throw new Error(`data.RandomSampler numSamples ${rawNumSamples} cannot exceed dataset length ${this.length} without replacement`);
      }
      this.numSamples = rawNumSamples as number;
      this.seed = samplerOptions.seed;
      this.rng = samplerOptions.rng;
    }

    len() { return this.numSamples; }
    __len__() { return this.numSamples; }
    size() { return this.numSamples; }

    *[Symbol.iterator]() {
      const rng = rngFromOptions({ seed: this.seed, rng: this.rng });
      if (this.replacement) {
        for (let index = 0; index < this.numSamples; index += 1) yield Math.floor(rng() * this.length);
        return;
      }
      const order = shuffledIndices(this.length, { shuffle: true, rng });
      for (let index = 0; index < this.numSamples; index += 1) yield order[index];
    }

    __iter__() {
      return this[Symbol.iterator]();
    }
  }

  class BatchSamplerClass {
    readonly sampler: Iterable<number>;
    readonly batchSize: number;
    readonly dropLast: boolean;
    readonly length: number;
    readonly kind = "zgml.data.batch-sampler";

    constructor(sampler: Iterable<number>, batchSize: number, dropLast = false) {
      if (!sampler || typeof sampler[Symbol.iterator] !== "function") throw new Error("data.BatchSampler requires an iterable sampler");
      this.sampler = sampler;
      this.batchSize = requireBatchSize(batchSize);
      this.dropLast = Boolean(dropLast);
      this.length = batchCount(samplerLength(sampler as AnyRecord, "data.BatchSampler"), this.batchSize, this.dropLast);
    }

    len() { return this.length; }
    __len__() { return this.length; }
    size() { return this.length; }

    *[Symbol.iterator]() {
      let batch: number[] = [];
      for (const index of this.sampler) {
        batch.push(index);
        if (batch.length === this.batchSize) {
          yield Object.freeze(batch.slice());
          batch = [];
        }
      }
      if (batch.length > 0 && !this.dropLast) yield Object.freeze(batch.slice());
    }

    __iter__() {
      return this[Symbol.iterator]();
    }
  }

  function batches(dataset: AnyRecord, batchOptions: AnyRecord = {}) {
    if (!dataset || typeof dataset !== "object" || typeof dataset.batch !== "function") {
      throw new Error("data.batches requires a dataset with batch(indices)");
    }
    const length = datasetLength(dataset, "data.batches");
    const batchSize = requireBatchSize(batchOptions.batchSize ?? batchOptions.batch_size ?? length);
    const dropLast = Boolean(batchOptions.dropLast ?? batchOptions.drop_last);
    const batchSampler = batchOptions.batchSampler ?? batchOptions.batch_sampler;
    const collateFn = batchOptions.collateFn ?? batchOptions.collate_fn;
    if (collateFn !== undefined && typeof collateFn !== "function") throw new Error("data.batches collateFn must be a function");
    if (collateFn !== undefined && typeof dataset.sample !== "function") throw new Error("data.batches collateFn requires a dataset with sample(index)");
    const count = batchSampler
      ? samplerLength(batchSampler, "data.batches batchSampler")
      : batchCount(samplerLength(batchOptions.sampler as AnyRecord ?? { length }, "data.batches sampler"), batchSize, dropLast);
    function makeBatch(rows: readonly number[], batchIndex: number) {
      if (typeof collateFn === "function") {
        const samples = rows.map((row) => dataset.sample(row));
        const indices = Object.freeze(rows.slice());
        return collateFn(Object.freeze(samples.slice()), Object.freeze({
          batchIndex,
          indices,
          sampleIndices: indices,
          sample_indices: indices,
        }));
      }
      return dataset.batch(rows, batchIndex);
    }
    function batchRows() {
      if (batchSampler !== undefined) {
        if (typeof (batchSampler as Iterable<unknown>)[Symbol.iterator] !== "function") throw new Error("data.batches batchSampler must be iterable");
        return iterableArray(batchSampler, "data.batches batchSampler").map((rows) => indexValues(rows, length, "data.batches batchSampler", true));
      }
      const order = batchOptions.sampler !== undefined || batchOptions.shuffle === undefined
        ? indexSamplerIndices(batchOptions.sampler, length, "data.batches")
        : shuffledIndices(length, batchOptions);
      const rows: number[][] = [];
      for (let offset = 0; offset < order.length; offset += batchSize) {
        const rowBatch = order.slice(offset, offset + batchSize);
        if (rowBatch.length < batchSize && dropLast) break;
        if (rowBatch.length > 0) rows.push(rowBatch);
      }
      return rows;
    }
    function batchAt(index: unknown) {
      const batchIndex = requireNonNegativeIndex(index, count, "data.batches.batch");
      const rows = batchRows()[batchIndex] ?? [];
      if (rows.length === 0 || (rows.length < batchSize && dropLast)) {
        throw new Error(`data.batches batch index ${batchIndex} is out of range`);
      }
      return makeBatch(rows, batchIndex);
    }
    function* iterator() {
      let batchIndex = 0;
      for (const rows of batchRows()) {
        if (rows.length === 0) continue;
        yield makeBatch(rows, batchIndex);
        batchIndex += 1;
      }
    }
    return Object.freeze({
      kind: "zgml.data.batches",
      length,
      sampleCount: length,
      sample_count: length,
      batchSize,
      batch_size: batchSize,
      batchCount: count,
      batch_count: count,
      dropLast,
      drop_last: dropLast,
      shuffle: Boolean(batchOptions.shuffle),
      dataset,
      sampler: batchOptions.sampler ?? null,
      batchSampler: batchSampler ?? null,
      batch_sampler: batchSampler ?? null,
      collateFn: collateFn ?? null,
      collate_fn: collateFn ?? null,
      len: () => count,
      __len__: () => count,
      size: () => count,
      batch: batchAt,
      batchRows,
      batch_rows: batchRows,
      get: batchAt,
      at: batchAt,
      __getitem__: batchAt,
      __iter__: iterator,
      [Symbol.iterator]: iterator,
    });
  }

  class DataLoaderClass {
    constructor(dataset: AnyRecord, batchOptions: AnyRecord = {}) {
      return batches(dataset, batchOptions);
    }
  }

  function subsetFromIndices(dataset: AnyRecord, indices: readonly number[], splitIndex: number | null = null) {
    const length = indices.length;

    function originalIndex(index: unknown, label: string) {
      const row = requireNonNegativeIndex(index, length, label);
      return indices[row];
    }

    function sample(index: unknown) {
      return dataset.sample(originalIndex(index, "data.randomSplit.sample"));
    }

    function batch(localIndices: Iterable<number>, batchIndex = 0) {
      const rows = indexValues(localIndices, length, "data.randomSplit.batch").map((index) => originalIndex(index, "data.randomSplit.batch"));
      return dataset.batch(rows, batchIndex);
    }

    function* iterator() {
      for (let index = 0; index < length; index += 1) yield sample(index);
    }

    return Object.freeze({
      kind: "zgml.data.tensor-dataset",
      length,
      input: dataset.input,
      target: dataset.target,
      splitIndex,
      split_index: splitIndex,
      indices: Object.freeze(indices.slice()),
      len: () => length,
      __len__: () => length,
      size: () => length,
      sample,
      get: sample,
      at: sample,
      __getitem__: sample,
      batch,
      __iter__: iterator,
      [Symbol.iterator]: iterator,
    });
  }

  function subsetDataset(datasetValue: unknown, indicesValue: unknown) {
    const dataset = requireDataset(datasetValue, "data.subset");
    const indices = indexValues(indicesValue, datasetLength(dataset, "data.subset"), "data.subset", true);
    return subsetFromIndices(dataset, indices);
  }

  class SubsetClass {
    constructor(datasetValue: unknown, indicesValue: unknown) {
      return subsetDataset(datasetValue, indicesValue);
    }
  }

  function take(datasetValue: unknown, countValue: unknown) {
    const dataset = requireDataset(datasetValue, "data.take");
    if (!Number.isSafeInteger(countValue) || (countValue as number) < 0) {
      throw new Error(`data.take count must be a non-negative safe integer, got ${countValue}`);
    }
    const count = Math.min(countValue as number, datasetLength(dataset, "data.take"));
    return subsetFromIndices(dataset, Array.from({ length: count }, (_value, index) => index));
  }

  function concatDataset(datasetsValue: unknown) {
    const datasetValues = iterableArray(datasetsValue, "data.concatDataset");
    if (datasetValues.length === 0) throw new Error("data.concatDataset requires a non-empty dataset iterable");
    const datasets = Object.freeze(datasetValues.map((dataset, index) => requireDataset(dataset, `data.concatDataset datasets[${index}]`)));
    const offsets: number[] = [];
    let length = 0;
    for (const dataset of datasets) {
      offsets.push(length);
      length += datasetLength(dataset, "data.concatDataset");
      if (!Number.isSafeInteger(length)) throw new Error("data.concatDataset total length must be a safe integer");
    }

    function locate(index: unknown, label: string) {
      const row = requireNonNegativeIndex(index, length, label);
      let datasetIndex = offsets.length - 1;
      for (let i = 0; i < offsets.length; i += 1) {
        const nextOffset = offsets[i + 1] ?? length;
        if (row >= offsets[i] && row < nextOffset) {
          datasetIndex = i;
          break;
        }
      }
      return Object.freeze({
        row,
        datasetIndex,
        localIndex: row - offsets[datasetIndex],
      });
    }

    function sample(index: unknown) {
      const location = locate(index, "data.concatDataset.sample");
      const entry = datasets[location.datasetIndex].sample(location.localIndex);
      return Object.freeze({
        ...entry,
        kind: "zgml.data.sample",
        index: location.row,
        sourceIndex: location.datasetIndex,
        source_index: location.datasetIndex,
        localIndex: location.localIndex,
        local_index: location.localIndex,
      });
    }

    function batch(indices: Iterable<number>, batchIndex = 0) {
      const rows = indexValues(indices, length, "data.concatDataset.batch");
      const samples = rows.map((row) => sample(row));
      const first = samples[0] as AnyRecord;
      const out: AnyRecord = {
        kind: "zgml.data.batch",
        batchIndex,
        indices: Object.freeze(rows.slice()),
        sourceIndices: Object.freeze(samples.map((entry) => (entry as AnyRecord).sourceIndex)),
        source_indices: Object.freeze(samples.map((entry) => (entry as AnyRecord).sourceIndex)),
        localIndices: Object.freeze(samples.map((entry) => (entry as AnyRecord).localIndex)),
        local_indices: Object.freeze(samples.map((entry) => (entry as AnyRecord).localIndex)),
      };
      if ("input" in first) out.input = stack(samples.map((entry) => (entry as AnyRecord).input as TTensor), 0);
      if ("target" in first) out.target = stack(samples.map((entry) => (entry as AnyRecord).target as TTensor), 0);
      return Object.freeze(out);
    }

    function* iterator() {
      for (let index = 0; index < length; index += 1) yield sample(index);
    }

    return Object.freeze({
      kind: "zgml.data.concat-dataset",
      length,
      datasets,
      offsets: Object.freeze(offsets.slice()),
      len: () => length,
      __len__: () => length,
      size: () => length,
      sample,
      get: sample,
      at: sample,
      __getitem__: sample,
      batch,
      __iter__: iterator,
      [Symbol.iterator]: iterator,
    });
  }

  class ConcatDatasetClass {
    constructor(datasetsValue: unknown) {
      return concatDataset(datasetsValue);
    }
  }

  function mapDataset(datasetValue: unknown, mapperValue: unknown) {
    const dataset = requireDataset(datasetValue, "data.mapDataset");
    if (typeof mapperValue !== "function") throw new Error("data.mapDataset requires a mapper function");
    const mapper = mapperValue as (sample: AnyRecord, index: number) => AnyRecord;
    const length = datasetLength(dataset, "data.mapDataset");

    function mappedSample(index: unknown) {
      const row = requireNonNegativeIndex(index, length, "data.mapDataset.sample");
      const mapped = mapper(dataset.sample(row), row);
      if (!mapped || typeof mapped !== "object") throw new Error("data.mapDataset mapper must return a sample object");
      return Object.freeze({
        ...mapped,
        kind: "zgml.data.sample",
        index: row,
      });
    }

    function mappedBatch(indices: Iterable<number>, batchIndex = 0) {
      const rows = indexValues(indices, length, "data.mapDataset.batch");
      const samples = rows.map((row) => mappedSample(row));
      const first = samples[0] as AnyRecord;
      const out: AnyRecord = {
        kind: "zgml.data.batch",
        batchIndex,
        indices: Object.freeze(rows.slice()),
      };
      if ("input" in first) out.input = stack(samples.map((sample) => (sample as AnyRecord).input as TTensor), 0);
      if ("target" in first) out.target = stack(samples.map((sample) => (sample as AnyRecord).target as TTensor), 0);
      for (const key of Object.keys(first)) {
        if (key === "kind" || key === "index" || key === "input" || key === "target") continue;
        out[key] = Object.freeze(samples.map((sample) => (sample as AnyRecord)[key]));
      }
      return Object.freeze(out);
    }

    function* iterator() {
      for (let index = 0; index < length; index += 1) yield mappedSample(index);
    }

    return Object.freeze({
      kind: "zgml.data.mapped-dataset",
      length,
      source: dataset,
      len: () => length,
      __len__: () => length,
      size: () => length,
      sample: mappedSample,
      get: mappedSample,
      at: mappedSample,
      __getitem__: mappedSample,
      batch: mappedBatch,
      __iter__: iterator,
      [Symbol.iterator]: iterator,
    });
  }

  class MapDatasetClass {
    constructor(datasetValue: unknown, mapperValue: unknown) {
      return mapDataset(datasetValue, mapperValue);
    }
  }

  function randomSplit(dataset: AnyRecord, lengthsValue: unknown, splitOptions: AnyRecord = {}) {
    if (!dataset || typeof dataset !== "object" || typeof dataset.sample !== "function" || typeof dataset.batch !== "function") {
      throw new Error("data.randomSplit requires a dataset with sample(index) and batch(indices)");
    }
    const length = datasetLength(dataset, "data.randomSplit");
    const lengths = splitLengths(lengthsValue, length);
    const order = shuffledIndices(length, { ...splitOptions, shuffle: splitOptions.shuffle ?? true });
    let offset = 0;
    return Object.freeze(lengths.map((splitLength, splitIndex) => {
      const indices = order.slice(offset, offset + splitLength);
      offset += splitLength;
      return subsetFromIndices(dataset, indices, splitIndex);
    }));
  }

  return Object.freeze({
    Dataset: DatasetClass,
    TensorDataset: TensorDatasetClass,
    DataLoader: DataLoaderClass,
    SequentialSampler: SequentialSamplerClass,
    RandomSampler: RandomSamplerClass,
    BatchSampler: BatchSamplerClass,
    Subset: SubsetClass,
    ConcatDataset: ConcatDatasetClass,
    MapDataset: MapDatasetClass,
    tensorDataset,
    tensor_dataset: tensorDataset,
    defaultCollate,
    default_collate: defaultCollate,
    batches,
    batch: batches,
    dataLoader: batches,
    dataloader: batches,
    subset: subsetDataset,
    take,
    concatDataset,
    concat_dataset: concatDataset,
    mapDataset,
    map_dataset: mapDataset,
    randomSplit,
    random_split: randomSplit,
  });
}

export const dataManifest = Object.freeze({
  kind: "zgml-data",
  ...tsProductManifestPolicy("src/ts/data.ts"),
  factory: "createDataNamespace",
});
