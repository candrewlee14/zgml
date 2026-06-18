"use strict";

import {
  normalizeFactoryShape,
  type Shape,
} from "./shape.js";
import type {
  RandomIntTensorOptions,
  RandomTensorOptions,
  RandomUniformTensorOptions,
  TensorOptions,
} from "../public_api.js";

type TensorFactoryTensor = {
  data: Float32Array;
  shape: readonly number[];
  length: number;
};
type TensorConstructor = new (
  data: Float32Array,
  shape: readonly number[],
  options?: TensorOptions,
) => TensorFactoryTensor;

export type TensorFactoryHelpersOptions = Readonly<{
  Tensor?: TensorConstructor;
}>;

let defaultSeededRng: (() => number) | null = null;
let defaultSeed: number | null = null;

function normalizeSeed(seed: unknown, label = "manualSeed seed") {
  const value = Number(seed);
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffffffff) {
    throw new Error(`${label} must be a uint32, got ${seed}`);
  }
  return value >>> 0;
}

export function seededRng(seed: unknown) {
  let state = normalizeSeed(seed, "seededRng seed") || 0x6d2b79f5;
  return function rng() {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function manualSeed(seed: unknown) {
  defaultSeed = normalizeSeed(seed);
  defaultSeededRng = seededRng(defaultSeed);
  return defaultSeed;
}

export function manual_seed(seed: unknown) {
  return manualSeed(seed);
}

export function initialSeed() {
  return defaultSeed;
}

export const initial_seed = initialSeed;

function randomSource(options: RandomIntTensorOptions | RandomTensorOptions | RandomUniformTensorOptions, label: string) {
  if (options.rng !== undefined) {
    if (typeof options.rng !== "function") throw new Error(`${label} rng option must be a function`);
    return options.rng;
  }
  if (options.seed !== undefined) return seededRng(options.seed);
  return defaultSeededRng ?? Math.random;
}

export function createTensorFactoryHelpers(options: TensorFactoryHelpersOptions = {}) {
  const TensorClass = options && options.Tensor;
  if (typeof TensorClass !== "function") {
    throw new Error("tensor factory helpers require a Tensor constructor");
  }
  const TensorCtor = TensorClass as TensorConstructor;

  function full(shape: number | Shape, value: number, tensorOptions: TensorOptions = {}) {
    if (!Number.isFinite(value)) throw new Error(`full tensor value must be finite, got ${value}`);
    const spec = normalizeFactoryShape(shape);
    const data = new Float32Array(spec.length);
    data.fill(value);
    return new TensorCtor(data, spec.shape, tensorOptions);
  }

  function zeros(shape: number | Shape, tensorOptions: TensorOptions = {}) {
    const spec = normalizeFactoryShape(shape);
    return new TensorCtor(new Float32Array(spec.length), spec.shape, tensorOptions);
  }

  function empty(shape: number | Shape, tensorOptions: TensorOptions = {}) {
    return zeros(shape, tensorOptions);
  }

  function ones(shape: number | Shape, tensorOptions: TensorOptions = {}) {
    return full(shape, 1, tensorOptions);
  }

  function eye(size: number, tensorOptions: TensorOptions = {}) {
    if (!Number.isSafeInteger(size) || size <= 0) {
      throw new Error(`eye size must be a positive safe integer, got ${size}`);
    }
    const data = new Float32Array(size * size);
    for (let i = 0; i < size; i += 1) data[i * size + i] = 1;
    return new TensorCtor(data, [size, size], tensorOptions);
  }

  function scalar(value: number, tensorOptions: TensorOptions = {}) {
    if (!Number.isFinite(value)) throw new Error(`scalar tensor value must be finite, got ${value}`);
    return new TensorCtor(Float32Array.of(value), [1], tensorOptions);
  }

  function rand(shape: number | Shape, options: RandomUniformTensorOptions = {}) {
    const spec = normalizeFactoryShape(shape);
    const {
      rng: _rng,
      seed: _seed,
      ...tensorOptions
    } = options;
    const rng = randomSource(options, "rand");
    const data = new Float32Array(spec.length);
    for (let i = 0; i < data.length; i += 1) {
      const value = Number(rng());
      if (!Number.isFinite(value)) throw new Error("rand rng must return finite numbers");
      data[i] = value;
    }
    return new TensorCtor(data, spec.shape, tensorOptions);
  }

  function randn(shape: number | Shape, options: RandomTensorOptions = {}) {
    const spec = normalizeFactoryShape(shape);
    const {
      mean = 0,
      std = 1,
      rng: _rng,
      seed: _seed,
      ...tensorOptions
    } = options;
    if (!Number.isFinite(mean)) throw new Error(`randn mean must be finite, got ${mean}`);
    if (!Number.isFinite(std) || std < 0) throw new Error(`randn std must be a finite non-negative number, got ${std}`);
    const rng = randomSource(options, "randn");
    const data = new Float32Array(spec.length);
    for (let i = 0; i < data.length; i += 2) {
      const u1 = Math.max(Number.EPSILON, Math.min(1 - Number.EPSILON, Number(rng())));
      const u2 = Math.max(0, Math.min(1, Number(rng())));
      if (!Number.isFinite(u1) || !Number.isFinite(u2)) throw new Error("randn rng must return finite numbers");
      const radius = Math.sqrt(-2 * Math.log(u1));
      const angle = 2 * Math.PI * u2;
      data[i] = mean + std * radius * Math.cos(angle);
      if (i + 1 < data.length) data[i + 1] = mean + std * radius * Math.sin(angle);
    }
    return new TensorCtor(data, spec.shape, tensorOptions);
  }

  function randInt(first: number, second: number | Shape, third?: number | Shape | RandomIntTensorOptions, fourth?: RandomIntTensorOptions) {
    let low = 0;
    let high = first;
    let shape = second as number | Shape;
    let options = (third ?? {}) as RandomIntTensorOptions;
    if (typeof second === "number" && third !== undefined && !(third && typeof third === "object" && !Array.isArray(third))) {
      low = first;
      high = second;
      shape = third as number | Shape;
      options = fourth ?? {};
    }
    if (!Number.isSafeInteger(low) || !Number.isSafeInteger(high) || high <= low) {
      throw new Error(`randInt range must be safe integers with high > low, got low=${low} high=${high}`);
    }
    const spec = normalizeFactoryShape(shape, "randInt shape");
    const {
      rng: _rng,
      seed: _seed,
      ...tensorOptions
    } = options;
    const rng = randomSource(options, "randInt");
    const width = high - low;
    const data = new Float32Array(spec.length);
    for (let i = 0; i < data.length; i += 1) {
      const value = Number(rng());
      if (!Number.isFinite(value)) throw new Error("randInt rng must return finite numbers");
      data[i] = low + Math.floor(Math.max(0, Math.min(1 - Number.EPSILON, value)) * width);
    }
    return new TensorCtor(data, spec.shape, tensorOptions);
  }

  function randint(first: number, second: number | Shape, third?: number | Shape | RandomIntTensorOptions, fourth?: RandomIntTensorOptions) {
    return randInt(first, second, third, fourth);
  }

  function randPerm(size: number, options: RandomIntTensorOptions = {}) {
    if (!Number.isSafeInteger(size) || size <= 0) {
      throw new Error(`randPerm size must be a positive safe integer, got ${size}`);
    }
    const {
      rng: _rng,
      seed: _seed,
      ...tensorOptions
    } = options;
    const rng = randomSource(options, "randPerm");
    const values = Array.from({ length: size }, (_value, index) => index);
    for (let i = values.length - 1; i > 0; i -= 1) {
      const value = Number(rng());
      if (!Number.isFinite(value)) throw new Error("randPerm rng must return finite numbers");
      const j = Math.floor(Math.max(0, Math.min(1 - Number.EPSILON, value)) * (i + 1));
      const tmp = values[i];
      values[i] = values[j];
      values[j] = tmp;
    }
    return new TensorCtor(Float32Array.from(values), [size], tensorOptions);
  }

  function randperm(size: number, options: RandomIntTensorOptions = {}) {
    return randPerm(size, options);
  }

  function linspace(first: number | Shape, second: number, third: number, fourth?: TensorOptions) {
    let shape: number | Shape = first;
    let start = second as number;
    let end = third as number;
    let tensorOptions = fourth ?? {};
    if (Number.isFinite(first) && Number.isFinite(second) && Number.isSafeInteger(third)) {
      shape = [third];
      start = first as number;
      end = second as number;
      tensorOptions = fourth ?? {};
    }
    if (!Number.isFinite(start) || !Number.isFinite(end)) {
      throw new Error(`linspace start and end must be finite numbers, got ${start} and ${end}`);
    }
    const spec = normalizeFactoryShape(shape, "linspace shape");
    const data = new Float32Array(spec.length);
    const step = (end - start) / spec.length;
    for (let i = 0; i < data.length; i += 1) data[i] = start + step * i;
    return new TensorCtor(data, spec.shape, tensorOptions);
  }

  function arange(start: number, end?: number | TensorOptions, stepOrOptions?: number | TensorOptions, maybeOptions?: TensorOptions) {
    let begin = start;
    let finish = end as number | undefined;
    let step = stepOrOptions as number | undefined;
    let tensorOptions = maybeOptions ?? {};
    if (finish === undefined || (finish && typeof finish === "object")) {
      tensorOptions = (finish as unknown as TensorOptions) ?? {};
      finish = begin;
      begin = 0;
      step = 1;
    } else if (step === undefined || (step && typeof step === "object")) {
      tensorOptions = (step as unknown as TensorOptions) ?? {};
      step = 1;
    }
    if (!Number.isFinite(begin) || !Number.isFinite(finish) || !Number.isFinite(step) || step === 0) {
      throw new Error("arange start, end, and step must be finite numbers with non-zero step");
    }
    const values = [];
    if (step > 0) {
      for (let value = begin; value < finish; value += step) values.push(value);
    } else {
      for (let value = begin; value > finish; value += step) values.push(value);
    }
    if (values.length === 0) throw new Error("arange produced an empty tensor");
    return new TensorCtor(Float32Array.from(values), [values.length], tensorOptions);
  }

  return {
    full,
    empty,
    zeros,
    ones,
    eye,
    scalar,
    rand,
    randn,
    randInt,
    randint,
    randPerm,
    randperm,
    manualSeed,
    manual_seed,
    initialSeed,
    initial_seed,
    seededRng,
    linspace,
    arange,
  };
}
