import {
  setPtr,
  setUSize,
} from "./bun_abi_words.js";
import {
  modelPathDescriptorFields,
  SafetensorsDataDescriptorFields,
  safetensorsDataDescriptorFields,
  SafetensorsHeaderDescriptorFields,
  safetensorsHeaderDescriptorFields,
  TinyLlamaModelDescriptorFields,
  tinyLlamaModelDescriptorFields,
  type ModelPathDescriptorFields,
} from "../runtime/model_source_desc.js";
import { supportedCheckpointModelsFromCatalog } from "../runtime/model_source_catalog.js";

type NativeHandle = number;
type NativeOut = BigUint64Array;

type ByteSource = Uint8Array;

type BunModelSourceSymbols = Readonly<{
  modelCreate(desc: BigUint64Array, out: NativeOut): number;
  modelLoadPath(desc: BigUint64Array, out: NativeOut): number;
  modelLoadSafetensorsData(desc: BigUint64Array, out: NativeOut): number;
  modelProbePath(desc: BigUint64Array, out: BigUint64Array): number;
  modelProbeSafetensorsData(desc: BigUint64Array, out: BigUint64Array): number;
  modelProbeSafetensorsHeader(desc: BigUint64Array, out: BigUint64Array): number;
  supportedCheckpointCount(): bigint | number;
  supportedCheckpointInspect(index: bigint, out: BigUint64Array): number;
}>;

export type BunModelSourceOpsOptions = Readonly<{
  symbols: BunModelSourceSymbols;
  check(code: number): void;
  handleOut(): NativeOut;
  readHandle(out: NativeOut): NativeHandle;
  pointerFor(value: ByteSource): NativeHandle;
  pathBytes(path: string): ByteSource;
  safetensorsDataBytes(data: unknown): ByteSource;
  safetensorsHeaderBytes(header: unknown): ByteSource;
  normalizeLoadModelKind(options: unknown): unknown;
  modelLoadKindId(kind: unknown): number;
  modelInspectionFromAbiWords(words: BigUint64Array): unknown;
  tinyLlamaKind: number;
}>;

export function createBunModelSourceOps(options: BunModelSourceOpsOptions) {
  const {
    symbols,
    check,
    handleOut,
    readHandle,
    pointerFor,
    pathBytes,
    safetensorsDataBytes,
    safetensorsHeaderBytes,
    normalizeLoadModelKind,
    modelLoadKindId,
    modelInspectionFromAbiWords,
    tinyLlamaKind,
  } = options;

  function pathDesc(fields: ModelPathDescriptorFields<ByteSource>): BigUint64Array {
    const desc = new BigUint64Array(3);
    const view = new DataView(desc.buffer);
    view.setUint32(0, fields.kind, true);
    setPtr(view, 8, pointerFor(fields.path));
    setUSize(view, 16, fields.pathLen);
    return desc;
  }

  function headerDesc(fields: SafetensorsHeaderDescriptorFields<ByteSource>): BigUint64Array {
    const desc = new BigUint64Array(3);
    const view = new DataView(desc.buffer);
    view.setUint32(0, fields.kind, true);
    view.setUint32(4, fields.reserved, true);
    setPtr(view, 8, pointerFor(fields.header));
    setUSize(view, 16, fields.headerLen);
    return desc;
  }

  function dataDesc(fields: SafetensorsDataDescriptorFields<ByteSource>): BigUint64Array {
    const desc = new BigUint64Array(3);
    const view = new DataView(desc.buffer);
    view.setUint32(0, fields.kind, true);
    view.setUint32(4, fields.reserved, true);
    setPtr(view, 8, pointerFor(fields.data));
    setUSize(view, 16, fields.dataLen);
    return desc;
  }

  function tinyLlamaModelDesc(fields: TinyLlamaModelDescriptorFields): BigUint64Array {
    const buf = new BigUint64Array(3);
    const view = new DataView(buf.buffer);
    view.setUint32(0, fields.kind, true);
    setUSize(view, 8, fields.inputLen);
    setUSize(view, 16, fields.outputLen);
    return buf;
  }

  function loadSafetensorsDataHandle(kind: number, data: unknown): NativeHandle {
    const out = handleOut();
    const dataBytes = safetensorsDataBytes(data);
    const fields = safetensorsDataDescriptorFields(kind, dataBytes);
    check(symbols.modelLoadSafetensorsData(dataDesc(fields), out));
    void dataBytes;
    return readHandle(out);
  }

  function loadModelPath(kind: number, path: string): NativeHandle {
    const out = handleOut();
    const bytes = pathBytes(path);
    const fields = modelPathDescriptorFields(kind, bytes);
    check(symbols.modelLoadPath(pathDesc(fields), out));
    void bytes;
    return readHandle(out);
  }

  function createTinyLlamaModelHandle(): NativeHandle {
    const out = handleOut();
    check(symbols.modelCreate(tinyLlamaModelDesc(tinyLlamaModelDescriptorFields(tinyLlamaKind)), out));
    return readHandle(out);
  }

  function probeModelPath(path: string, kind: unknown): unknown {
    const bytes = pathBytes(path);
    const fields = modelPathDescriptorFields(modelLoadKindId(kind), bytes);
    const out = new BigUint64Array(13);
    check(symbols.modelProbePath(pathDesc(fields), out));
    void bytes;
    return modelInspectionFromAbiWords(out);
  }

  function probeSafetensorsData(data: unknown, opts: unknown = {}): unknown {
    const out = new BigUint64Array(13);
    const dataBytes = safetensorsDataBytes(data);
    const fields = safetensorsDataDescriptorFields(modelLoadKindId(normalizeLoadModelKind(opts)), dataBytes);
    check(symbols.modelProbeSafetensorsData(dataDesc(fields), out));
    void dataBytes;
    return modelInspectionFromAbiWords(out);
  }

  function probeSafetensorsHeader(header: unknown, opts: unknown = {}): unknown {
    const headerBytes = safetensorsHeaderBytes(header);
    const fields = safetensorsHeaderDescriptorFields(modelLoadKindId(normalizeLoadModelKind(opts)), headerBytes);
    const out = new BigUint64Array(13);
    check(symbols.modelProbeSafetensorsHeader(headerDesc(fields), out));
    void headerBytes;
    return modelInspectionFromAbiWords(out);
  }

  function supportedCheckpointModels(): readonly unknown[] {
    return supportedCheckpointModelsFromCatalog(
      () => symbols.supportedCheckpointCount(),
      (i) => {
        const out = new BigUint64Array(13);
        check(symbols.supportedCheckpointInspect(BigInt(i), out));
        return modelInspectionFromAbiWords(out);
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
