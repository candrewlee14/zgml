"use strict";

import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";

export type LlamaProgramInspectionValueReader = (field: string, index: number) => unknown;

export type LlamaProgramInspection = Readonly<{
  kind: "zgml.llama.program.inspection";
  signature: string;
  vocabSize: number;
  maxSeqLen: number;
  contextLen: number;
  batch: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  nKvHeads: number;
  semanticStageCount: number;
  semanticTokenCount: number;
  semanticLayerStageCount: number;
  semanticTerminalStageCount: number;
  semanticRuntimePatchHoles: number;
  semanticRuntimePatchCacheWritePosHoles: number;
  semanticRuntimePatchAttentionSeqKvHoles: number;
}>;

const llamaProgramInspectionFields = Object.freeze([
  Object.freeze({ name: "vocabSize", record: "vocab_size" }),
  Object.freeze({ name: "maxSeqLen", record: "max_seq_len" }),
  Object.freeze({ name: "contextLen", record: "context_len" }),
  Object.freeze({ name: "batch", record: "batch" }),
  Object.freeze({ name: "dModel", record: "d_model" }),
  Object.freeze({ name: "nLayers", record: "n_layers" }),
  Object.freeze({ name: "nHeads", record: "n_heads" }),
  Object.freeze({ name: "nKvHeads", record: "n_kv_heads" }),
  Object.freeze({ name: "semanticStageCount", record: "semantic_stage_count" }),
  Object.freeze({ name: "semanticTokenCount", record: "semantic_token_count" }),
  Object.freeze({ name: "semanticLayerStageCount", record: "semantic_layer_stage_count" }),
  Object.freeze({ name: "semanticTerminalStageCount", record: "semantic_terminal_stage_count" }),
  Object.freeze({ name: "semanticRuntimePatchHoles", record: "semantic_runtime_patch_holes" }),
  Object.freeze({ name: "semanticRuntimePatchCacheWritePosHoles", record: "semantic_runtime_patch_cache_write_pos_holes" }),
  Object.freeze({ name: "semanticRuntimePatchAttentionSeqKvHoles", record: "semantic_runtime_patch_attention_seq_kv_holes" }),
] as const);

type LlamaProgramInspectionFieldName = typeof llamaProgramInspectionFields[number]["name"];

export function llamaProgramInspectionFromReader(read: LlamaProgramInspectionValueReader): LlamaProgramInspection {
  const inspection: Record<LlamaProgramInspectionFieldName, number> & { kind: "zgml.llama.program.inspection"; signature?: string } = {
    kind: "zgml.llama.program.inspection",
  } as Record<LlamaProgramInspectionFieldName, number> & { kind: "zgml.llama.program.inspection"; signature?: string };
  for (let index = 0; index < llamaProgramInspectionFields.length; index += 1) {
    const field = llamaProgramInspectionFields[index]!;
    inspection[field.name] = Number(read(field.record, index));
  }
  inspection.signature = [
    "llama-program-inspection",
    ...llamaProgramInspectionFields.map((field) => `${field.name}=${inspection[field.name]}`),
  ].join("|");
  return Object.freeze(inspection) as LlamaProgramInspection;
}

export function llamaProgramInspectionFromRecord(record: Record<string, unknown>): LlamaProgramInspection {
  return llamaProgramInspectionFromReader((field) => record[field]);
}

export function llamaProgramInspectionFromWords(words: ArrayLike<unknown>): LlamaProgramInspection {
  return llamaProgramInspectionFromReader((_, index) => words[index]);
}

export const llamaProgramInspectionManifest = Object.freeze({
  kind: "zgml-llama-program-inspection",
  ...tsRuntimeManifestPolicy("src/ts/runtime/llama_program_inspection.ts", "LLaMA Program -> inspection evidence"),
});
