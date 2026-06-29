import {
  createRuntimeAbiFacadeHelpers,
} from "../runtime/abi.js";
import {
  createBunStatusHelpers,
  type BunHandleOut,
  type BunStatusHelpers,
  type BunStatusHelpersOptions,
} from "./bun_status.js";

type RuntimeAbiFacade = ReturnType<typeof createRuntimeAbiFacadeHelpers>;

export type BunFfiBootstrapSymbols = Readonly<{
  zgml_abi_struct_size(kind: number): bigint | number;
  zgml_get_runtime_info(outInfo: BigUint64Array): number;
  zgml_status_name(code: number): string;
}>;

export type BunFfiBootstrapOptions = Readonly<{
  symbols: BunFfiBootstrapSymbols;
  readPointer: BunStatusHelpersOptions["readPointer"];
}>;

export type BunFfiBootstrap = BunStatusHelpers & Readonly<{
  ok: 0;
  runtimeInfo: RuntimeAbiFacade["runtimeInfo"];
  abiStructSize: RuntimeAbiFacade["abiStructSize"];
  abiStructSizes: RuntimeAbiFacade["abiStructSizes"];
  assertCompatibleRuntime: RuntimeAbiFacade["assertCompatibleRuntime"];
  loadedRuntimeInfo: ReturnType<RuntimeAbiFacade["assertCompatibleRuntime"]>;
}>;

function runtimeInfoFromAbiWords(out: BunHandleOut) {
  const view = new DataView(out.buffer);
  return {
    abiVersion: view.getUint32(0, true),
    sizeTBytes: view.getUint32(4, true),
    pointerBytes: view.getUint32(8, true),
    tokenIdBytes: view.getUint32(12, true),
    featureFlags: view.getBigUint64(16, true),
  };
}

export function createBunFfiBootstrap(options: BunFfiBootstrapOptions): BunFfiBootstrap {
  const { symbols } = options;
  const status = createBunStatusHelpers({
    ok: 0,
    statusName: symbols.zgml_status_name,
    readPointer: options.readPointer,
  });
  const runtimeAbiFacade = createRuntimeAbiFacadeHelpers({
    readRuntimeInfo: () => {
      const out = new BigUint64Array(3);
      status.check(symbols.zgml_get_runtime_info(out));
      return runtimeInfoFromAbiWords(out);
    },
    readAbiStructSize: (kind) => symbols.zgml_abi_struct_size(kind),
  });

  return Object.freeze({
    ...status,
    ok: 0,
    runtimeInfo: runtimeAbiFacade.runtimeInfo,
    abiStructSize: runtimeAbiFacade.abiStructSize,
    abiStructSizes: runtimeAbiFacade.abiStructSizes,
    assertCompatibleRuntime: runtimeAbiFacade.assertCompatibleRuntime,
    loadedRuntimeInfo: runtimeAbiFacade.assertCompatibleRuntime(),
  });
}
