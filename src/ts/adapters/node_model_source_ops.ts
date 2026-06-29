import {
  modelPathDescriptorFields,
  modelPathDescriptorRecord,
  safetensorsDataDescriptorFields,
  safetensorsDataDescriptorRecord,
  safetensorsHeaderDescriptorFields,
  safetensorsHeaderDescriptorRecord,
  tinyLlamaModelDescriptorFields,
  tinyLlamaModelDescriptorRecord,
} from "../runtime/model_source_desc.js";
import { supportedCheckpointModelsFromCatalog } from "../runtime/model_source_catalog.js";

type NativeHandle = unknown;
type NativeOut = [NativeHandle | null];

type ByteSource = {
  readonly byteLength: number;
};

type PathBytes = ByteSource & {
  readonly length: number;
};

type NodeModelSourceSymbols = {
  modelCreate(desc: Record<string, unknown>, out: NativeOut): number;
  modelLoadPath(desc: Record<string, unknown>, out: NativeOut): number;
  modelLoadSafetensorsData(desc: Record<string, unknown>, out: NativeOut): number;
  modelProbePath(desc: Record<string, unknown>, out: Record<string, unknown>): number;
  modelProbeSafetensorsData(desc: Record<string, unknown>, out: Record<string, unknown>): number;
  modelProbeSafetensorsHeader(desc: Record<string, unknown>, out: Record<string, unknown>): number;
  supportedCheckpointCount(): number;
  supportedCheckpointInspect(index: number, out: Record<string, unknown>): number;
};

export type NodeModelSourceOpsOptions = Readonly<{
  symbols: NodeModelSourceSymbols;
  check(code: number): void;
  handleOut(): NativeOut;
  readHandle<T = NativeHandle>(out: readonly [T | null | undefined], name: string): T;
  bytesFromString(value: string): PathBytes;
  bytesFromView(value: Uint8Array): ByteSource;
  safetensorsDataBytes(value: unknown): Uint8Array;
  safetensorsHeaderBytes(value: unknown): Uint8Array;
  normalizeLoadModelKind(options: unknown): unknown;
  modelLoadKindId(kind: unknown): number;
  modelInspectionFromAbiRecord(record: Record<string, unknown>): unknown;
  tinyLlamaKind: number;
}>;

export function createNodeModelSourceOps(options: NodeModelSourceOpsOptions) {
  const {
    symbols,
    check,
    handleOut,
    readHandle,
    bytesFromString,
    bytesFromView,
    safetensorsDataBytes,
    safetensorsHeaderBytes,
    normalizeLoadModelKind,
    modelLoadKindId,
    modelInspectionFromAbiRecord,
    tinyLlamaKind,
  } = options;

  function loadModelPath(kind: number, modelPath: string): NativeHandle {
    const path = bytesFromString(modelPath);
    const out = handleOut();
    check(symbols.modelLoadPath(modelPathDescriptorRecord(modelPathDescriptorFields(kind, path)), out));
    return readHandle(out, "model");
  }

  function loadSafetensorsDataHandle(kind: number, data: unknown): NativeHandle {
    const bytes = bytesFromView(safetensorsDataBytes(data));
    const out = handleOut();
    check(symbols.modelLoadSafetensorsData(safetensorsDataDescriptorRecord(safetensorsDataDescriptorFields(kind, bytes)), out));
    return readHandle(out, "model");
  }

  function createTinyLlamaModelHandle(): NativeHandle {
    const out = handleOut();
    check(symbols.modelCreate(tinyLlamaModelDescriptorRecord(tinyLlamaModelDescriptorFields(tinyLlamaKind)), out));
    return readHandle(out, "model");
  }

  function probeModelPath(source: string, kind: unknown): unknown {
    const path = bytesFromString(source);
    const out: Record<string, unknown> = {};
    check(symbols.modelProbePath(modelPathDescriptorRecord(modelPathDescriptorFields(modelLoadKindId(kind), path)), out));
    return modelInspectionFromAbiRecord(out);
  }

  function probeSafetensorsData(data: unknown, opts: unknown = {}): unknown {
    const bytes = bytesFromView(safetensorsDataBytes(data));
    const out: Record<string, unknown> = {};
    check(symbols.modelProbeSafetensorsData(safetensorsDataDescriptorRecord(safetensorsDataDescriptorFields(
      modelLoadKindId(normalizeLoadModelKind(opts)),
      bytes,
    )), out));
    return modelInspectionFromAbiRecord(out);
  }

  function probeSafetensorsHeader(header: unknown, opts: unknown = {}): unknown {
    const bytes = bytesFromView(safetensorsHeaderBytes(header));
    const out: Record<string, unknown> = {};
    check(symbols.modelProbeSafetensorsHeader(safetensorsHeaderDescriptorRecord(safetensorsHeaderDescriptorFields(
      modelLoadKindId(normalizeLoadModelKind(opts)),
      bytes,
    )), out));
    return modelInspectionFromAbiRecord(out);
  }

  function supportedCheckpointModels(): readonly unknown[] {
    return supportedCheckpointModelsFromCatalog(
      () => symbols.supportedCheckpointCount(),
      (i) => {
        const out: Record<string, unknown> = {};
        check(symbols.supportedCheckpointInspect(i, out));
        return modelInspectionFromAbiRecord(out);
      },
    );
  }

  return Object.freeze({
    loadModelPath,
    loadSafetensorsDataHandle,
    createTinyLlamaModelHandle,
    probeModelPath,
    probeSafetensorsData,
    probeSafetensorsHeader,
    supportedCheckpointModels,
  });
}
