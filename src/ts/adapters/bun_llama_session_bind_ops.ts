import {
  llamaKvCacheBindDescriptorFields,
  llamaSessionBufferBindDescriptorFields,
  requireLlamaNativeOutput,
  type LlamaKvCacheBindDescriptorFields,
  type LlamaKvCacheRequirements,
  type LlamaSessionBufferBindDescriptorFields,
} from "../runtime/llama_session_bind_desc.js";
import {
  programOutputBindDescriptorFields,
  type ProgramOutputBindDescriptorFields,
} from "../runtime/program_bind_desc.js";
import {
  setPtr,
  setUSize,
} from "./bun_abi_words.js";
import type {
  LlamaBindOptionsRecord,
  LlamaSessionBindPolicy as SharedLlamaSessionBindPolicy,
} from "../runtime/session_binding.js";

type NativeHandle = number;
type NativeOut = BigUint64Array;

type NativeBufferLike = {
  readonly handleValue: NativeHandle;
  readonly byteLength: number;
  assertAlive?(): void;
};

type LlamaKvCacheResources = {
  readonly k: readonly unknown[];
  readonly v: readonly unknown[];
};

type BunLlamaSessionBindOptions = LlamaBindOptionsRecord & {
  readonly kvCache?: LlamaKvCacheResources | undefined;
};

type BunKvCacheBindDesc = Readonly<{
  desc: BigUint64Array;
  kHandles: BigUint64Array;
  vHandles: BigUint64Array;
}>;

type LlamaProgramInspection = {
  readonly vocabSize: number;
};

type BunLlamaSessionBindSymbols = Readonly<{
  llamaProgramGetKvCacheRequirements(handle: NativeHandle, out: BigUint64Array): number;
  llamaSessionBindBuffers(handle: NativeHandle, desc: BigUint64Array | NativeHandle, out: NativeOut): number;
  llamaSessionBindModelBuffers(
    handle: NativeHandle,
    sourceModel: NativeHandle,
    desc: BigUint64Array | NativeHandle,
    out: NativeOut,
  ): number;
  sessionBindBuffers(handle: NativeHandle, desc: BigUint64Array, out: NativeOut): number;
  sessionBindModelBuffers(handle: NativeHandle, sourceModel: NativeHandle, desc: BigUint64Array, out: NativeOut): number;
  sessionBind(handle: NativeHandle, desc: NativeHandle, out: NativeOut): number;
  sessionBindModel(handle: NativeHandle, sourceModel: NativeHandle, desc: NativeHandle, out: NativeOut): number;
}>;

type LlamaSessionBindPolicy = (
  programHandle: NativeHandle,
  options: BunLlamaSessionBindOptions,
  inspection: LlamaProgramInspection,
  policy: Partial<SharedLlamaSessionBindPolicy<NativeHandle, LlamaProgramInspection, NativeHandle>>,
) => NativeHandle;

export type BunLlamaSessionBindOpsOptions = Readonly<{
  symbols: BunLlamaSessionBindSymbols;
  check(code: number): void;
  handleOut(): NativeOut;
  readHandle(out: NativeOut): NativeHandle;
  requireNativeBuffer(value: unknown, label: string): NativeBufferLike | null;
  isNativeBuffer(value: unknown): boolean;
  modelHandleForBind(model: unknown): NativeHandle | null;
  pointerFor(value: BigUint64Array): NativeHandle;
  bindLlamaSessionHandleByPolicy: LlamaSessionBindPolicy;
  llamaKvCacheRequirementsFromAbiWords(words: BigUint64Array): LlamaKvCacheRequirements;
}>;

export function createBunLlamaSessionBindOps(options: BunLlamaSessionBindOpsOptions) {
  const {
    symbols,
    check,
    handleOut,
    readHandle,
    requireNativeBuffer,
    isNativeBuffer,
    modelHandleForBind,
    pointerFor,
    bindLlamaSessionHandleByPolicy,
    llamaKvCacheRequirementsFromAbiWords,
  } = options;

  function llamaBufferBindDesc(fields: ProgramOutputBindDescriptorFields<NativeBufferLike>): BigUint64Array {
    const buf = new BigUint64Array(8);
    const view = new DataView(buf.buffer);
    setPtr(view, 48, fields.output?.handleValue ?? 0);
    setUSize(view, 56, fields.outputLen);
    return buf;
  }

  function llamaOutputBindFields(output: unknown, vocabSize: number): ProgramOutputBindDescriptorFields<NativeBufferLike> {
    return programOutputBindDescriptorFields(requireLlamaNativeOutput(output, requireNativeBuffer), vocabSize);
  }

  function llamaKvCacheRequirements(handle: NativeHandle): LlamaKvCacheRequirements {
    const out = new BigUint64Array(6);
    check(symbols.llamaProgramGetKvCacheRequirements(handle, out));
    return llamaKvCacheRequirementsFromAbiWords(out);
  }

  function llamaKvCacheBindDesc(fields: LlamaKvCacheBindDescriptorFields<bigint> | null): BunKvCacheBindDesc | null {
    if (fields === null) return null;
    const kHandles = new BigUint64Array(fields.len);
    const vHandles = new BigUint64Array(fields.len);
    for (let i = 0; i < fields.len; i += 1) {
      kHandles[i] = fields.kHandles[i]!;
      vHandles[i] = fields.vHandles[i]!;
    }
    const desc = new BigUint64Array(3);
    const view = new DataView(desc.buffer);
    setPtr(view, 0, pointerFor(kHandles));
    setPtr(view, 8, pointerFor(vHandles));
    setUSize(view, 16, fields.len);
    return Object.freeze({ desc, kHandles, vHandles });
  }

  function llamaKvCacheBindFields(kvCache: unknown, requirements: LlamaKvCacheRequirements) {
    return llamaKvCacheBindDescriptorFields(kvCache, requirements, requireNativeBuffer, (buffer) => BigInt(buffer.handleValue));
  }

  type BunSessionBufferBindFields = LlamaSessionBufferBindDescriptorFields<
    NativeBufferLike,
    LlamaKvCacheBindDescriptorFields<bigint>
  >;

  function packLlamaSessionBufferBindDesc(fields: BunSessionBufferBindFields): {
    desc: BigUint64Array;
    kv: BunKvCacheBindDesc | null;
  } {
    const kv = llamaKvCacheBindDesc(fields.kvCache);
    const desc = new BigUint64Array(3);
    const view = new DataView(desc.buffer);
    setPtr(view, 0, fields.output?.handleValue ?? 0);
    setUSize(view, 8, fields.outputLen);
    setPtr(view, 16, kv ? pointerFor(kv.desc) : 0);
    return Object.freeze({ desc, kv });
  }

  function llamaSessionBufferBindDesc(
    bindOptions: BunLlamaSessionBindOptions,
    inspection: LlamaProgramInspection,
    requirements: LlamaKvCacheRequirements,
  ): {
    desc: BigUint64Array;
    kv: BunKvCacheBindDesc | null;
  } | null {
    const fields = llamaSessionBufferBindDescriptorFields(
      bindOptions,
      inspection,
      requirements,
      requireNativeBuffer,
      llamaKvCacheBindFields,
    );
    return fields ? packLlamaSessionBufferBindDesc(fields) : null;
  }

  function bindLlamaSessionHandle(
    programHandle: NativeHandle,
    bindOptions: BunLlamaSessionBindOptions,
    inspection: LlamaProgramInspection,
  ): NativeHandle {
    return bindLlamaSessionHandleByPolicy(programHandle, bindOptions, inspection, {
      isNativeBuffer,
      modelHandleForBind,
      hasSourceModel: (model: NativeHandle | null): model is NativeHandle => model !== null && model !== 0,
      bindKvCache: (handle: NativeHandle, optionsForBind: LlamaBindOptionsRecord, info: LlamaProgramInspection) => {
        const out = handleOut();
        const bind = llamaSessionBufferBindDesc(optionsForBind as BunLlamaSessionBindOptions, info, llamaKvCacheRequirements(handle));
        check(symbols.llamaSessionBindBuffers(handle, bind?.desc ?? 0, out));
        void bind;
        return readHandle(out);
      },
      bindModelKvCache: (
        handle: NativeHandle,
        sourceModel: NativeHandle,
        optionsForBind: LlamaBindOptionsRecord,
        info: LlamaProgramInspection,
      ) => {
        const out = handleOut();
        const bind = llamaSessionBufferBindDesc(optionsForBind as BunLlamaSessionBindOptions, info, llamaKvCacheRequirements(handle));
        check(symbols.llamaSessionBindModelBuffers(handle, sourceModel, bind?.desc ?? 0, out));
        void bind;
        return readHandle(out);
      },
      bindNativeOutput: (handle: NativeHandle, optionsForBind: LlamaBindOptionsRecord, info: LlamaProgramInspection) => {
        const out = handleOut();
        check(symbols.sessionBindBuffers(handle, llamaBufferBindDesc(llamaOutputBindFields(optionsForBind.output, info.vocabSize)), out));
        return readHandle(out);
      },
      bindModelNativeOutput: (
        handle: NativeHandle,
        sourceModel: NativeHandle,
        optionsForBind: LlamaBindOptionsRecord,
        info: LlamaProgramInspection,
      ) => {
        const out = handleOut();
        check(symbols.sessionBindModelBuffers(
          handle,
          sourceModel,
          llamaBufferBindDesc(llamaOutputBindFields(optionsForBind.output, info.vocabSize)),
          out,
        ));
        return readHandle(out);
      },
      bindPlain: (handle: NativeHandle) => {
        const out = handleOut();
        check(symbols.sessionBind(handle, 0, out));
        return readHandle(out);
      },
      bindModelPlain: (handle: NativeHandle, sourceModel: NativeHandle) => {
        const out = handleOut();
        check(symbols.sessionBindModel(handle, sourceModel, 0, out));
        return readHandle(out);
      },
    });
  }

  return Object.freeze({
    llamaKvCacheRequirements,
    bindLlamaSessionHandle,
  });
}
