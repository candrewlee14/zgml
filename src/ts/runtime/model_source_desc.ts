"use strict";

import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";

type ModelSourceBytes = {
  readonly byteLength: number;
  readonly length?: number;
};

export type ModelPathDescriptorFields<TBytes = unknown> = Readonly<{
  kind: number;
  path: TBytes;
  pathLen: number;
}>;

export type ModelPathDescriptorRecord<TBytes = unknown> = Readonly<{
  kind: number;
  path: TBytes;
  path_len: number;
}>;

export type SafetensorsDataDescriptorFields<TBytes = unknown> = Readonly<{
  kind: number;
  reserved: 0;
  data: TBytes;
  dataLen: number;
}>;

export type SafetensorsDataDescriptorRecord<TBytes = unknown> = Readonly<{
  kind: number;
  reserved: 0;
  data: TBytes;
  data_len: number;
}>;

export type SafetensorsHeaderDescriptorFields<TBytes = unknown> = Readonly<{
  kind: number;
  reserved: 0;
  header: TBytes;
  headerLen: number;
}>;

export type SafetensorsHeaderDescriptorRecord<TBytes = unknown> = Readonly<{
  kind: number;
  reserved: 0;
  header: TBytes;
  header_len: number;
}>;

export type TinyLlamaModelDescriptorFields = Readonly<{
  kind: number;
  inputLen: 0;
  outputLen: 0;
}>;

export type TinyLlamaModelDescriptorRecord = Readonly<{
  kind: number;
  input_len: 0;
  output_len: 0;
}>;

function sourceByteLength(bytes: ModelSourceBytes): number {
  return typeof bytes.length === "number" ? bytes.length : bytes.byteLength;
}

export function modelPathDescriptorFields<TBytes extends ModelSourceBytes>(
  kind: number,
  path: TBytes,
): ModelPathDescriptorFields<TBytes> {
  return Object.freeze({
    kind,
    path,
    pathLen: sourceByteLength(path),
  });
}

export function modelPathDescriptorRecord<TBytes>(
  fields: ModelPathDescriptorFields<TBytes>,
): ModelPathDescriptorRecord<TBytes> {
  return Object.freeze({
    kind: fields.kind,
    path: fields.path,
    path_len: fields.pathLen,
  });
}

export function safetensorsDataDescriptorFields<TBytes extends ModelSourceBytes>(
  kind: number,
  data: TBytes,
): SafetensorsDataDescriptorFields<TBytes> {
  return Object.freeze({
    kind,
    reserved: 0,
    data,
    dataLen: sourceByteLength(data),
  });
}

export function safetensorsDataDescriptorRecord<TBytes>(
  fields: SafetensorsDataDescriptorFields<TBytes>,
): SafetensorsDataDescriptorRecord<TBytes> {
  return Object.freeze({
    kind: fields.kind,
    reserved: fields.reserved,
    data: fields.data,
    data_len: fields.dataLen,
  });
}

export function safetensorsHeaderDescriptorFields<TBytes extends ModelSourceBytes>(
  kind: number,
  header: TBytes,
): SafetensorsHeaderDescriptorFields<TBytes> {
  return Object.freeze({
    kind,
    reserved: 0,
    header,
    headerLen: sourceByteLength(header),
  });
}

export function safetensorsHeaderDescriptorRecord<TBytes>(
  fields: SafetensorsHeaderDescriptorFields<TBytes>,
): SafetensorsHeaderDescriptorRecord<TBytes> {
  return Object.freeze({
    kind: fields.kind,
    reserved: fields.reserved,
    header: fields.header,
    header_len: fields.headerLen,
  });
}

export function tinyLlamaModelDescriptorFields(kind: number): TinyLlamaModelDescriptorFields {
  return Object.freeze({
    kind,
    inputLen: 0,
    outputLen: 0,
  });
}

export function tinyLlamaModelDescriptorRecord(
  fields: TinyLlamaModelDescriptorFields,
): TinyLlamaModelDescriptorRecord {
  return Object.freeze({
    kind: fields.kind,
    input_len: fields.inputLen,
    output_len: fields.outputLen,
  });
}

export const modelSourceDescriptorManifest = Object.freeze({
  kind: "zgml-model-source-descriptor",
  ...tsRuntimeManifestPolicy("src/ts/runtime/model_source_desc.ts", "ModelSource -> load descriptor -> Program"),
});
