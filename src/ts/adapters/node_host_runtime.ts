"use strict";

import {
  nodeKoffiFallbackPath,
  resolveNativeLibraryLoadInfo,
} from "./native.js";

declare const require: {
  (id: string): unknown;
};
declare const process: {
  cwd(): string;
  env: { ZGML_C_DYLIB?: string };
  platform: string;
};
declare const __dirname: string;

type NodeFs = {
  existsSync(path: string): boolean;
};

type NodePath = {
  join(...parts: string[]): string;
};

type KoffiModule = {
  load(path: string): unknown;
  struct(name: string, fields: Record<string, unknown>): unknown;
  proto(name: string, result: unknown, params: readonly unknown[]): unknown;
  pointer(type: unknown): unknown;
  out(type: unknown): unknown;
};

type ModuleNotFoundError = Error & { code?: string };

export type NodeHostRuntime = Readonly<{
  fs: NodeFs;
  path: NodePath;
  koffi: KoffiModule;
  packageRoot: string;
  suffix: string;
  libPath: string;
}>;

export type NodeHostRuntimeOptions = Readonly<{
  dirname?: string;
  envLibraryPath?: string;
  cwd?: string;
  platform?: string;
}>;

function requireNodeModule<T>(id: string): T {
  return require(id) as T;
}

function isModuleNotFound(error: unknown): error is ModuleNotFoundError {
  return error instanceof Error && (error as ModuleNotFoundError).code === "MODULE_NOT_FOUND";
}

function requireKoffi(fs: NodeFs, path: NodePath, packageRoot: string): KoffiModule {
  try {
    return requireNodeModule<KoffiModule>("koffi");
  } catch (error) {
    if (!isModuleNotFound(error)) {
      throw error;
    }
    const sourceTreeKoffi = nodeKoffiFallbackPath({
      moduleDir: packageRoot,
      joinPath: path.join,
    });
    if (fs.existsSync(sourceTreeKoffi)) {
      return requireNodeModule<KoffiModule>(sourceTreeKoffi);
    }
    throw error;
  }
}

export function createNodeHostRuntime(options: NodeHostRuntimeOptions = {}): NodeHostRuntime {
  const fs = requireNodeModule<NodeFs>("node:fs");
  const path = requireNodeModule<NodePath>("node:path");
  const dirname = options.dirname ?? __dirname;
  const packageRoot = path.join(dirname, "..", "..");
  const nativeLibrary = resolveNativeLibraryLoadInfo({
    envLibraryPath: options.envLibraryPath ?? process.env.ZGML_C_DYLIB,
    moduleDir: packageRoot,
    cwd: options.cwd ?? process.cwd(),
    platform: options.platform ?? process.platform,
    joinPath: path.join,
    exists: fs.existsSync,
  });

  return Object.freeze({
    fs,
    path,
    koffi: requireKoffi(fs, path, nativeLibrary.packageRoot),
    packageRoot: nativeLibrary.packageRoot,
    suffix: nativeLibrary.suffix,
    libPath: nativeLibrary.libPath,
  });
}
