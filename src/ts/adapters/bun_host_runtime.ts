"use strict";

import {
  resolveNativeLibraryLoadInfo,
} from "./native.js";

declare const process: {
  cwd(): string;
  env: { ZGML_C_DYLIB?: string };
  platform: string;
};

export type BunHostRuntime = Readonly<{
  packageRoot: string;
  suffix: string;
  libPath: string;
}>;

export type BunHostRuntimeOptions = Readonly<{
  importMetaUrl: string;
  dirname: (path: string) => string;
  fileURLToPath: (url: string) => string;
  joinPath: (...parts: string[]) => string;
  exists: (path: string) => boolean;
  envLibraryPath?: string;
  cwd?: string;
  platform?: string;
}>;

export function createBunHostRuntime(options: BunHostRuntimeOptions): BunHostRuntime {
  const here = options.dirname(options.fileURLToPath(options.importMetaUrl));
  const packageRoot = options.joinPath(here, "..", "..");
  return resolveNativeLibraryLoadInfo({
    envLibraryPath: options.envLibraryPath ?? process.env.ZGML_C_DYLIB,
    moduleDir: packageRoot,
    cwd: options.cwd ?? process.cwd(),
    platform: options.platform ?? process.platform,
    joinPath: options.joinPath,
    exists: options.exists,
  });
}
