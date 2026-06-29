"use strict";

import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";

type BindingValue = { readonly length?: number } | null;

export type ProgramBindDescriptorFields<TValue = unknown> = Readonly<{
  weights: TValue | null;
  weightsLen: number;
  bias: TValue | null;
  biasLen: number;
  input: TValue | null;
  inputLen: number;
  output: TValue | null;
  outputLen: number;
}>;

export type ProgramBindDescriptorRecord<TValue = unknown> = Readonly<{
  weights: TValue | null;
  weights_len: number;
  bias: TValue | null;
  bias_len: number;
  input: TValue | null;
  input_len: number;
  output: TValue | null;
  output_len: number;
}>;

export type ProgramBindDescriptorRecordFields<TValue = unknown> = Readonly<{
  weights?: TValue | null;
  weightsLen: number;
  bias?: TValue | null;
  biasLen: number;
  input?: TValue | null;
  inputLen: number;
  output?: TValue | null;
  outputLen: number;
}>;

export type ProgramOutputBindDescriptorFields<TValue = unknown> = ProgramBindDescriptorFields<TValue>;

function valueLength(value: BindingValue): number {
  return value && typeof value.length === "number" ? value.length : 0;
}

export function programBindDescriptorFields<TValue extends BindingValue>(
  values: Readonly<{
    weights: TValue;
    bias?: TValue | null;
    input?: TValue | null;
    output?: TValue | null;
  }>,
): ProgramBindDescriptorFields<TValue> {
  const bias = values.bias ?? null;
  const input = values.input ?? null;
  const output = values.output ?? null;
  return Object.freeze({
    weights: values.weights,
    weightsLen: valueLength(values.weights),
    bias,
    biasLen: valueLength(bias),
    input,
    inputLen: valueLength(input),
    output,
    outputLen: valueLength(output),
  });
}

export function programOutputBindDescriptorFields<TValue>(
  output: TValue | null,
  outputLen: number,
): ProgramOutputBindDescriptorFields<TValue> {
  return Object.freeze({
    weights: null,
    weightsLen: 0,
    bias: null,
    biasLen: 0,
    input: null,
    inputLen: 0,
    output,
    outputLen: output ? outputLen : 0,
  });
}

export function programBindDescriptorRecord<TValue, TRecordValue = TValue>(
  fields: ProgramBindDescriptorRecordFields<TValue>,
  valueForRecord: (value: TValue | null) => TRecordValue | null = (value) => value as TRecordValue | null,
): ProgramBindDescriptorRecord<TRecordValue> {
  return Object.freeze({
    weights: valueForRecord(fields.weights ?? null),
    weights_len: fields.weightsLen,
    bias: valueForRecord(fields.bias ?? null),
    bias_len: fields.biasLen,
    input: valueForRecord(fields.input ?? null),
    input_len: fields.inputLen,
    output: valueForRecord(fields.output ?? null),
    output_len: fields.outputLen,
  });
}

export const programBindDescriptorManifest = Object.freeze({
  kind: "zgml-program-bind-descriptor",
  ...tsRuntimeManifestPolicy("src/ts/runtime/program_bind_desc.ts", "Program -> binding descriptor -> Session"),
});
