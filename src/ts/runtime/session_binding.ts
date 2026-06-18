"use strict";

type UnknownRecord = Record<string, unknown>;

type DisposableOwnedOutput = Readonly<{
  free?: () => void;
}>;

type LlamaProgramForSessionBind = Readonly<UnknownRecord & {
  inspect: () => unknown;
  bufferLayout: () => unknown;
  kvCacheLayout: () => unknown;
}>;

export type LlamaBindOptionsRecord = Readonly<{
  output?: unknown;
  model?: unknown;
  kvCache?: unknown;
}>;

export type LlamaBindNormalization = {
  options: LlamaBindOptionsRecord;
  boundOutput: unknown;
  ownedOutput: DisposableOwnedOutput | null;
};

export type NormalizeLlamaBindOptionsDeps = {
  createProgramOutputBuffer: (programHandle: unknown) => DisposableOwnedOutput;
  isNativeBuffer: (value: unknown) => boolean;
};

export type BindLlamaProgramSessionDeps = NormalizeLlamaBindOptionsDeps & {
  bindSessionHandle: (programHandle: unknown, options: LlamaBindOptionsRecord, inspection: unknown) => unknown;
  createSession: (
    sessionHandle: unknown,
    inspection: unknown,
    boundOutput: unknown,
    ownedOutput: DisposableOwnedOutput | null,
    layout: unknown,
    kvLayout: unknown,
  ) => unknown;
};

export type LlamaSessionBindPolicy<
  THandle = unknown,
  TInspection = unknown,
  TSourceModel = THandle,
> = {
  isNativeBuffer: (value: unknown) => boolean;
  modelHandleForBind: (model: unknown) => TSourceModel | null;
  hasSourceModel: (model: TSourceModel | null) => model is TSourceModel;
  bindKvCache: (programHandle: THandle, options: LlamaBindOptionsRecord, inspection: TInspection) => unknown;
  bindModelKvCache: (programHandle: THandle, sourceModel: TSourceModel, options: LlamaBindOptionsRecord, inspection: TInspection) => unknown;
  bindNativeOutput: (programHandle: THandle, options: LlamaBindOptionsRecord, inspection: TInspection) => unknown;
  bindModelNativeOutput: (programHandle: THandle, sourceModel: TSourceModel, options: LlamaBindOptionsRecord, inspection: TInspection) => unknown;
  bindPlain: (programHandle: THandle, options: LlamaBindOptionsRecord, inspection: TInspection) => unknown;
  bindModelPlain: (programHandle: THandle, sourceModel: TSourceModel, options: LlamaBindOptionsRecord, inspection: TInspection) => unknown;
};

function requireFunction<T extends (...args: any[]) => any>(
  fn: unknown,
  message: string,
): T {
  if (typeof fn !== "function") {
    throw new Error(message);
  }
  return fn as T;
}

function requireLlamaProgram(program: unknown): LlamaProgramForSessionBind {
  const record = program as LlamaProgramForSessionBind | null;
  if (
    !record ||
    typeof record.inspect !== "function" ||
    typeof record.bufferLayout !== "function" ||
    typeof record.kvCacheLayout !== "function"
  ) {
    throw new Error("LLaMA Program.bind requires a Program with inspect, bufferLayout, and kvCacheLayout");
  }
  return record;
}

function requireBindPolicyFunction<T extends (...args: any[]) => any>(
  policy: Readonly<Record<string, unknown>> | null | undefined,
  name: string,
): T {
  return requireFunction<T>(
    policy && policy[name],
    `LLaMA Session bind policy requires ${String(name)}`,
  );
}

export function bindLlamaSessionHandleByPolicy<
  THandle = unknown,
  TInspection = unknown,
  TSourceModel = THandle,
>(
  programHandle: THandle,
  options: LlamaBindOptionsRecord = {},
  inspection: TInspection,
  policy: Partial<LlamaSessionBindPolicy<THandle, TInspection, TSourceModel>>,
) {
  type Policy = LlamaSessionBindPolicy<THandle, TInspection, TSourceModel>;
  const isNativeBuffer = requireBindPolicyFunction<Policy["isNativeBuffer"]>(policy, "isNativeBuffer");
  const modelHandleForBind = requireBindPolicyFunction<Policy["modelHandleForBind"]>(policy, "modelHandleForBind");
  const hasSourceModel = requireBindPolicyFunction<Policy["hasSourceModel"]>(policy, "hasSourceModel");
  const bindKvCache = requireBindPolicyFunction<Policy["bindKvCache"]>(policy, "bindKvCache");
  const bindModelKvCache = requireBindPolicyFunction<Policy["bindModelKvCache"]>(policy, "bindModelKvCache");
  const bindNativeOutput = requireBindPolicyFunction<Policy["bindNativeOutput"]>(policy, "bindNativeOutput");
  const bindModelNativeOutput = requireBindPolicyFunction<Policy["bindModelNativeOutput"]>(policy, "bindModelNativeOutput");
  const bindPlain = requireBindPolicyFunction<Policy["bindPlain"]>(policy, "bindPlain");
  const bindModelPlain = requireBindPolicyFunction<Policy["bindModelPlain"]>(policy, "bindModelPlain");

  const bindOptions = options || {};
  const sourceModel = modelHandleForBind(bindOptions.model);
  const hasModel = hasSourceModel(sourceModel);
  if (bindOptions.kvCache !== undefined) {
    return hasModel
      ? bindModelKvCache(programHandle, sourceModel, bindOptions, inspection)
      : bindKvCache(programHandle, bindOptions, inspection);
  }
  if (isNativeBuffer(bindOptions.output)) {
    return hasModel
      ? bindModelNativeOutput(programHandle, sourceModel, bindOptions, inspection)
      : bindNativeOutput(programHandle, bindOptions, inspection);
  }
  if (bindOptions.output !== undefined) {
    throw new Error("LLaMA persistent output binding requires a zgml NativeBuffer; pass Float32Array outputs per call");
  }
  return hasModel
    ? bindModelPlain(programHandle, sourceModel, bindOptions, inspection)
    : bindPlain(programHandle, bindOptions, inspection);
}

export function normalizeLlamaBindOptions(
  programHandle: unknown,
  options: LlamaBindOptionsRecord = {},
  deps: NormalizeLlamaBindOptionsDeps,
): LlamaBindNormalization {
  const createProgramOutputBuffer = requireFunction<NormalizeLlamaBindOptionsDeps["createProgramOutputBuffer"]>(
    deps && deps.createProgramOutputBuffer,
    "normalizeLlamaBindOptions requires createProgramOutputBuffer",
  );
  const isNativeBuffer = requireFunction<NormalizeLlamaBindOptionsDeps["isNativeBuffer"]>(
    deps && deps.isNativeBuffer,
    "normalizeLlamaBindOptions requires isNativeBuffer",
  );

  if (options.output === "native" || options.output === true) {
    const output = createProgramOutputBuffer(programHandle);
    return {
      options: { ...options, output },
      boundOutput: output,
      ownedOutput: output,
    };
  }
  return {
    options,
    boundOutput: isNativeBuffer(options.output) ? options.output : null,
    ownedOutput: null,
  };
}

export function bindLlamaProgramSession(
  programHandle: unknown,
  program: unknown,
  options: LlamaBindOptionsRecord = {},
  deps: BindLlamaProgramSessionDeps,
) {
  const llamaProgram = requireLlamaProgram(program);
  const bindSessionHandle = requireFunction<BindLlamaProgramSessionDeps["bindSessionHandle"]>(
    deps && deps.bindSessionHandle,
    "LLaMA Program.bind requires bindSessionHandle",
  );
  const createSession = requireFunction<BindLlamaProgramSessionDeps["createSession"]>(
    deps && deps.createSession,
    "LLaMA Program.bind requires createSession",
  );

  const inspection = llamaProgram.inspect();
  const layout = llamaProgram.bufferLayout();
  const kvLayout = llamaProgram.kvCacheLayout();
  const bind = normalizeLlamaBindOptions(programHandle, options, deps);
  try {
    return createSession(
      bindSessionHandle(programHandle, bind.options, inspection),
      inspection,
      bind.boundOutput,
      bind.ownedOutput,
      layout,
      kvLayout,
    );
  } catch (err) {
    if (bind.ownedOutput && typeof bind.ownedOutput.free === "function") {
      bind.ownedOutput.free();
    }
    throw err;
  }
}
