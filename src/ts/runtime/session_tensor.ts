"use strict";

declare const require: (path: string) => unknown;

import type {
  SessionExecuteTensorParams,
  SessionReadOutputTensorOptions,
  SessionStepTensorOptions,
} from "../public_api.js";

type ProgramDesc = {
  readonly inputShape?: readonly number[];
  readonly outputShape?: readonly number[];
  readonly inputLen: number;
  readonly outputLen: number;
};

type TensorLike = {
  readonly data: Float32Array;
  readonly shape: readonly number[];
};

type TensorConstructor = new (values: Float32Array, shape: readonly number[], options?: Record<string, unknown>) => TensorLike;

export type SessionTensorHelpersOptions = {
  getTensorClass?: () => unknown;
};

export type TensorOutputOptions =
  | SessionStepTensorOptions
  | SessionExecuteTensorParams
  | SessionReadOutputTensorOptions
  | Readonly<{ output?: false; shape?: undefined }>;

type NativeReadableOutput = {
  readFloat32Into: (target: Float32Array, length: number, byteOffset: number) => unknown;
};

export type SessionTensorHelpers = Readonly<{
  outputShape(
    desc: Pick<ProgramDesc, "outputShape">,
    outputLen: number,
    shape: unknown,
    label?: string,
  ): readonly number[];
  outputTarget(options?: TensorOutputOptions, label?: string): unknown;
  outputTensor(values: unknown, options?: TensorOutputOptions, fallbackShape?: readonly number[]): TensorLike;
  outputTensorForSession(
    desc: unknown,
    values: unknown,
    options?: TensorOutputOptions,
    fallbackShape?: unknown,
    label?: string,
  ): TensorLike;
  readOutputInto(
    boundOutput: unknown,
    target: unknown,
    length: unknown,
    byteOffset: unknown,
    label?: string,
  ): unknown;
}>;

type ShapeHelpers = {
  normalizeFactoryShape: (shape: unknown, label: string) => { shape: readonly number[]; length: number };
};

const {
  normalizeFactoryShape,
} = require("../core/shape.js") as ShapeHelpers;

export function programInputShape(desc: ProgramDesc): readonly number[] {
  return Object.freeze(Array.isArray(desc.inputShape) ? desc.inputShape.slice() : [desc.inputLen]);
}

export function programOutputShape(desc: ProgramDesc): readonly number[] {
  return Object.freeze(Array.isArray(desc.outputShape) ? desc.outputShape.slice() : [desc.outputLen]);
}

export function sessionOutputTensorShape(
  desc: Pick<ProgramDesc, "outputShape">,
  outputLen: number,
  shape: unknown,
  label = "Session.stepTensor output shape",
): readonly number[] {
  const rawShape = shape === undefined ? (desc.outputShape ?? [outputLen]) : shape;
  const normalized = normalizeFactoryShape(rawShape, label);
  if (normalized.length !== outputLen) {
    throw new Error(`${label} has ${normalized.length} elements, expected ${outputLen}`);
  }
  return normalized.shape;
}

export function createSessionTensorHelpers(options: SessionTensorHelpersOptions): SessionTensorHelpers {
  const getTensorClass = options && options.getTensorClass;
  if (typeof getTensorClass !== "function") {
    throw new Error("Session tensor helpers require getTensorClass");
  }
  const getTensorClassFn = getTensorClass;

  function TensorClass(): TensorConstructor {
    const cls = getTensorClassFn();
    if (typeof cls !== "function") throw new Error("Session tensor helpers require a Tensor class");
    return cls as TensorConstructor;
  }

  function isTensor(value: unknown): value is TensorLike {
    return value instanceof TensorClass();
  }

  const outputShape = sessionOutputTensorShape;

  function outputTarget(options: TensorOutputOptions = {}, label = "Tensor output helpers"): unknown {
    const output = options.output;
    if (output === false) throw new Error(`${label} require logits output`);
    return isTensor(output) ? output.data : output;
  }

  function outputTensor(values: Float32Array, options: TensorOutputOptions = {}, fallbackShape?: readonly number[]): TensorLike {
    const shape = options.shape !== undefined
      ? options.shape
      : isTensor(options.output) ? options.output.shape : fallbackShape ?? [values.length];
    return new (TensorClass())(values, shape as readonly number[], options);
  }

  function outputTensorForSession(
    desc: Pick<ProgramDesc, "outputShape">,
    values: Float32Array,
    options: TensorOutputOptions = {},
    fallbackShape?: readonly number[],
    label = "Session.stepTensor output shape",
  ): TensorLike {
    const shape = options.shape !== undefined
      ? options.shape
      : isTensor(options.output) ? options.output.shape : fallbackShape;
    return new (TensorClass())(values, sessionOutputTensorShape(desc, values.length, shape, label), options);
  }

  function readOutputInto(
    boundOutput: Float32Array | NativeReadableOutput | null | undefined,
    target: Float32Array,
    length: number,
    byteOffset: number,
    label = "Session.readOutputInto",
  ): unknown {
    if (!(target instanceof Float32Array)) {
      throw new Error(`${label} requires a Float32Array target`);
    }
    if (!Number.isSafeInteger(length) || length < 0) {
      throw new Error(`invalid ${label} length: ${length}`);
    }
    if (!Number.isSafeInteger(byteOffset) || byteOffset < 0) {
      throw new Error(`invalid ${label} byteOffset: ${byteOffset}`);
    }
    if (byteOffset % Float32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error(`${label} byteOffset must be aligned to f32`);
    }
    if (target.length < length) {
      throw new Error(`${label} target is too small: ${target.length} < ${length}`);
    }
    if (boundOutput && typeof (boundOutput as NativeReadableOutput).readFloat32Into === "function") {
      return (boundOutput as NativeReadableOutput).readFloat32Into(target, length, byteOffset);
    }
    if (!boundOutput) {
      throw new Error(`${label} requires a bound output buffer`);
    }
    if (!(boundOutput instanceof Float32Array)) {
      throw new Error(`${label} requires a Float32Array or NativeBuffer bound output`);
    }
    const elementOffset = byteOffset / Float32Array.BYTES_PER_ELEMENT;
    if (elementOffset + length > boundOutput.length) {
      throw new Error(`${label} source range is out of bounds: ${elementOffset}..${elementOffset + length}`);
    }
    target.set(boundOutput.subarray(elementOffset, elementOffset + length), 0);
    return target;
  }

  return Object.freeze({
    outputShape,
    outputTarget,
    outputTensor,
    outputTensorForSession,
    readOutputInto,
  });
}
