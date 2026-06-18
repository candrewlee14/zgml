import {
  createRuntimeAbiFacadeHelpers,
} from "../runtime/abi.js";
import {
  registerNodeAbiStructs,
} from "./node_abi_structs.js";
import {
  createNodeHostRuntime,
  type NodeHostRuntime,
  type NodeHostRuntimeOptions,
} from "./node_host_runtime.js";
import {
  bindNodeSymbols,
  type NodeNativeSymbols,
} from "./node_symbols.js";
import {
  createNodeStatusHelpers,
  type NodeStatusHelpers,
} from "./node_status.js";

type RuntimeAbiFacade = ReturnType<typeof createRuntimeAbiFacadeHelpers>;

export type NodeFfiBootstrapOptions = Readonly<{
  dirname?: NodeHostRuntimeOptions["dirname"];
}>;

export type NodeFfiBootstrap = NodeHostRuntime & NodeStatusHelpers & Readonly<{
  ok: 0;
  symbols: NodeNativeSymbols;
  runtimeInfo: RuntimeAbiFacade["runtimeInfo"];
  abiStructSize: RuntimeAbiFacade["abiStructSize"];
  abiStructSizes: RuntimeAbiFacade["abiStructSizes"];
  assertCompatibleRuntime: RuntimeAbiFacade["assertCompatibleRuntime"];
  loadedRuntimeInfo: ReturnType<RuntimeAbiFacade["assertCompatibleRuntime"]>;
}>;

export function createNodeFfiBootstrap(options: NodeFfiBootstrapOptions = {}): NodeFfiBootstrap {
  const host = createNodeHostRuntime({ dirname: options.dirname });
  registerNodeAbiStructs(host.koffi);
  const symbols = bindNodeSymbols(host.koffi.load(host.libPath));
  const status = createNodeStatusHelpers({
    ok: 0,
    statusName: (code) => String(symbols.statusName(code)),
  });
  const runtimeAbiFacade = createRuntimeAbiFacadeHelpers({
    readRuntimeInfo: () => {
      const out: Record<string, unknown> = {};
      status.check(Number(symbols.getRuntimeInfo(out)));
      return {
        abiVersion: out.abi_version,
        sizeTBytes: out.size_t_bytes,
        pointerBytes: out.pointer_bytes,
        tokenIdBytes: out.token_id_bytes,
        featureFlags: out.feature_flags,
      };
    },
    readAbiStructSize: (kind) => symbols.abiStructSize(kind),
  });
  return Object.freeze({
    ...host,
    ...status,
    ok: 0,
    symbols,
    runtimeInfo: runtimeAbiFacade.runtimeInfo,
    abiStructSize: runtimeAbiFacade.abiStructSize,
    abiStructSizes: runtimeAbiFacade.abiStructSizes,
    assertCompatibleRuntime: runtimeAbiFacade.assertCompatibleRuntime,
    loadedRuntimeInfo: runtimeAbiFacade.assertCompatibleRuntime(),
  });
}
