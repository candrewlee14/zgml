import {
  llamaKvCacheBindDescriptorFields,
  llamaSessionBufferBindDescriptorFields,
  requireLlamaNativeOutput,
  type LlamaKvCacheBindDescriptorFields,
  type LlamaKvCacheRequirements,
} from "../runtime/llama_session_bind_desc.js";
import {
  programBindDescriptorRecord,
  programOutputBindDescriptorFields,
} from "../runtime/program_bind_desc.js";
import type {
  LlamaBindOptionsRecord,
  LlamaSessionBindPolicy as SharedLlamaSessionBindPolicy,
} from "../runtime/session_binding.js";

type NativeHandle = unknown;
type NativeOut = [NativeHandle | null];

type NativeBufferLike = {
  readonly handle: NativeHandle;
  readonly byteLength: number;
};

type LlamaProgramInspection = {
  readonly vocabSize: number;
};

type NodeLlamaSessionBindSymbols = Readonly<{
  llamaProgramGetKvCacheRequirements(handle: NativeHandle, out: Record<string, unknown>): number;
  llamaSessionBindBuffers(handle: NativeHandle, desc: Record<string, unknown> | null, out: NativeOut): number;
  llamaSessionBindModelBuffers(
    handle: NativeHandle,
    sourceModel: NativeHandle,
    desc: Record<string, unknown> | null,
    out: NativeOut,
  ): number;
  sessionBindBuffers(handle: NativeHandle, desc: Record<string, unknown> | null, out: NativeOut): number;
  sessionBindModelBuffers(
    handle: NativeHandle,
    sourceModel: NativeHandle,
    desc: Record<string, unknown> | null,
    out: NativeOut,
  ): number;
  sessionBind(handle: NativeHandle, desc: null, out: NativeOut): number;
  sessionBindModel(handle: NativeHandle, sourceModel: NativeHandle, desc: null, out: NativeOut): number;
}>;

type LlamaSessionBindPolicy = (
  programHandle: NativeHandle,
  options: LlamaBindOptionsRecord,
  inspection: LlamaProgramInspection,
  policy: Partial<SharedLlamaSessionBindPolicy<NativeHandle, LlamaProgramInspection, NativeHandle>>,
) => NativeHandle;

export type NodeLlamaSessionBindOpsOptions = Readonly<{
  symbols: NodeLlamaSessionBindSymbols;
  check(code: number): void;
  handleOut(): NativeOut;
  readHandle<T = NativeHandle>(out: readonly [T | null | undefined], name: string): T;
  requireNativeBuffer(value: unknown, name: string): NativeBufferLike | null;
  isNativeBuffer(value: unknown): boolean;
  modelHandleForBind(model: unknown): NativeHandle | null;
  bindLlamaSessionHandleByPolicy: LlamaSessionBindPolicy;
  llamaKvCacheRequirementsFromAbiRecord(record: Record<string, unknown>): LlamaKvCacheRequirements;
}>;

export function createNodeLlamaSessionBindOps(options: NodeLlamaSessionBindOpsOptions) {
  const {
    symbols,
    check,
    handleOut,
    readHandle,
    requireNativeBuffer,
    isNativeBuffer,
    modelHandleForBind,
    bindLlamaSessionHandleByPolicy,
    llamaKvCacheRequirementsFromAbiRecord,
  } = options;

  function bufferBindDescForLlamaOutput(output: unknown, vocabSize: number): Record<string, unknown> {
    const nativeOutput = requireLlamaNativeOutput(output, requireNativeBuffer);
    return programBindDescriptorRecord(
      programOutputBindDescriptorFields(nativeOutput, vocabSize),
      (value) => value ? value.handle : null,
    );
  }

  function llamaKvCacheRequirements(handle: NativeHandle): LlamaKvCacheRequirements {
    const out: Record<string, unknown> = {};
    check(symbols.llamaProgramGetKvCacheRequirements(handle, out));
    return llamaKvCacheRequirementsFromAbiRecord(out);
  }

  function llamaKvCacheBindDesc(fields: LlamaKvCacheBindDescriptorFields<NativeHandle> | null): Record<string, unknown> | null {
    if (fields === null) return null;
    return {
      k: fields.kHandles,
      v: fields.vHandles,
      len: fields.len,
    };
  }

  function llamaKvCacheBindFields(kvCache: unknown, requirements: LlamaKvCacheRequirements) {
    return llamaKvCacheBindDescriptorFields(kvCache, requirements, requireNativeBuffer, (buffer) => buffer.handle);
  }

  function llamaBufferBindDesc(
    bindOptions: LlamaBindOptionsRecord,
    inspection: LlamaProgramInspection,
    requirements: LlamaKvCacheRequirements,
  ): Record<string, unknown> | null {
    const fields = llamaSessionBufferBindDescriptorFields(
      bindOptions,
      inspection,
      requirements,
      requireNativeBuffer,
      llamaKvCacheBindFields,
    );
    if (fields === null) return null;
    return {
      output: fields.output ? fields.output.handle : null,
      output_len: fields.outputLen,
      kv_cache: llamaKvCacheBindDesc(fields.kvCache),
    };
  }

  function bindLlamaSessionHandle(
    programHandle: NativeHandle,
    bindOptions: LlamaBindOptionsRecord,
    inspection: LlamaProgramInspection,
  ): NativeHandle {
    return bindLlamaSessionHandleByPolicy(programHandle, bindOptions, inspection, {
      isNativeBuffer,
      modelHandleForBind,
      hasSourceModel: (model: NativeHandle | null): model is NativeHandle => model !== null,
      bindKvCache: (handle: NativeHandle, optionsForBind: LlamaBindOptionsRecord, info: LlamaProgramInspection) => {
        const out = handleOut();
        check(symbols.llamaSessionBindBuffers(
          handle,
          llamaBufferBindDesc(optionsForBind, info, llamaKvCacheRequirements(handle)),
          out,
        ));
        return readHandle(out, "session");
      },
      bindModelKvCache: (
        handle: NativeHandle,
        sourceModel: NativeHandle,
        optionsForBind: LlamaBindOptionsRecord,
        info: LlamaProgramInspection,
      ) => {
        const out = handleOut();
        check(symbols.llamaSessionBindModelBuffers(
          handle,
          sourceModel,
          llamaBufferBindDesc(optionsForBind, info, llamaKvCacheRequirements(handle)),
          out,
        ));
        return readHandle(out, "session");
      },
      bindNativeOutput: (handle: NativeHandle, optionsForBind: LlamaBindOptionsRecord, info: LlamaProgramInspection) => {
        const out = handleOut();
        check(symbols.sessionBindBuffers(handle, bufferBindDescForLlamaOutput(optionsForBind.output, info.vocabSize), out));
        return readHandle(out, "session");
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
          bufferBindDescForLlamaOutput(optionsForBind.output, info.vocabSize),
          out,
        ));
        return readHandle(out, "session");
      },
      bindPlain: (handle: NativeHandle) => {
        const out = handleOut();
        check(symbols.sessionBind(handle, null, out));
        return readHandle(out, "session");
      },
      bindModelPlain: (handle: NativeHandle, sourceModel: NativeHandle) => {
        const out = handleOut();
        check(symbols.sessionBindModel(handle, sourceModel, null, out));
        return readHandle(out, "session");
      },
    });
  }

  return Object.freeze({
    llamaKvCacheRequirements,
    bindLlamaSessionHandle,
  });
}
