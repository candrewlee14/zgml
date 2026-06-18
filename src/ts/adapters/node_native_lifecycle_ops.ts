import type { NodeNativeSymbols } from "./node_symbols.js";
import type { NodeStatusHelpers } from "./node_status.js";

export type NodeNativeLifecycleSymbols = Pick<
  NodeNativeSymbols,
  | "sessionFree"
  | "programResetRuntimeProfile"
  | "programFree"
  | "modelFree"
>;

export type NodeNativeLifecycleOpsOptions = Readonly<{
  symbols: NodeNativeLifecycleSymbols;
  check: NodeStatusHelpers["check"];
}>;

export type NodeNativeLifecycleOps = Readonly<{
  sessionFree(handle: unknown): void;
  programResetRuntimeProfile(handle: unknown): void;
  programFree(handle: unknown): void;
  modelFree(handle: unknown): void;
}>;

export function createNodeNativeLifecycleOps(options: NodeNativeLifecycleOpsOptions): NodeNativeLifecycleOps {
  const { symbols, check } = options;

  return Object.freeze({
    sessionFree(handle) {
      symbols.sessionFree(handle);
    },
    programResetRuntimeProfile(handle) {
      check(Number(symbols.programResetRuntimeProfile(handle)));
    },
    programFree(handle) {
      symbols.programFree(handle);
    },
    modelFree(handle) {
      symbols.modelFree(handle);
    },
  });
}
