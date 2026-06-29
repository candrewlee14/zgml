import type { BunNativeHandle, BunStatusHelpers } from "./bun_status.js";

export type BunNativeLifecycleSymbols = Readonly<{
  zgml_session_free(handle: BunNativeHandle): void;
  zgml_program_reset_runtime_profile(handle: BunNativeHandle): number;
  zgml_program_free(handle: BunNativeHandle): void;
  zgml_model_free(handle: BunNativeHandle): void;
}>;

export type BunNativeLifecycleOpsOptions = Readonly<{
  symbols: BunNativeLifecycleSymbols;
  check: BunStatusHelpers["check"];
}>;

export type BunNativeLifecycleOps = Readonly<{
  sessionFree(handle: BunNativeHandle): void;
  programResetRuntimeProfile(handle: BunNativeHandle): void;
  programFree(handle: BunNativeHandle): void;
  modelFree(handle: BunNativeHandle): void;
}>;

export function createBunNativeLifecycleOps(options: BunNativeLifecycleOpsOptions): BunNativeLifecycleOps {
  const { symbols, check } = options;

  return Object.freeze({
    sessionFree(handle) {
      symbols.zgml_session_free(handle);
    },
    programResetRuntimeProfile(handle) {
      check(symbols.zgml_program_reset_runtime_profile(handle));
    },
    programFree(handle) {
      symbols.zgml_program_free(handle);
    },
    modelFree(handle) {
      symbols.zgml_model_free(handle);
    },
  });
}
