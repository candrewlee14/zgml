"use strict";

const { existsSync, readdirSync, statSync } = require("node:fs");
const { join, resolve } = require("node:path");

function nativeLibraryPath(root) {
  if (process.env.ZGML_C_DYLIB) return resolve(root, process.env.ZGML_C_DYLIB);
  const extension = process.platform === "darwin" ? "dylib" : process.platform === "win32" ? "dll" : "so";
  return join(root, "zig-out", "lib", `libzgml_c.${extension}`);
}

function newestNativeSourceMtimeMs(paths) {
  let newest = { path: "", mtimeMs: 0 };
  const visit = (path) => {
    if (!existsSync(path)) return;
    const stat = statSync(path);
    if (stat.isDirectory()) {
      for (const entry of readdirSync(path)) visit(join(path, entry));
      return;
    }
    if (!path.endsWith(".zig") && !path.endsWith(".h") && !path.endsWith(".metal")) return;
    if (stat.mtimeMs > newest.mtimeMs) newest = { path, mtimeMs: stat.mtimeMs };
  };
  for (const path of paths) visit(path);
  return newest;
}

function verifyFreshNativeLibrary(options) {
  const root = options.root;
  const label = options.label;
  const allowStaleEnv = options.allowStaleEnv;
  const allowStale = process.env[allowStaleEnv] === "1";
  const libPath = nativeLibraryPath(root);
  if (!existsSync(libPath)) {
    throw new Error(`${label} requires native library ${libPath}; run npm run build:native:release first`);
  }
  const libStat = statSync(libPath);
  const newestSource = newestNativeSourceMtimeMs([join(root, "build.zig"), join(root, "src")]);
  const stale = newestSource.mtimeMs > libStat.mtimeMs + 1;
  if (stale) {
    const message = `${label} native library is older than Zig source (${libPath} < ${newestSource.path}); run npm run build:native:release or set ${allowStaleEnv}=1 for an explicitly stale diagnostic`;
    if (!allowStale) throw new Error(message);
    console.warn(`warning: ${message}`);
  }
  return Object.freeze({
    libPath,
    stale,
    label: stale ? "stale" : "fresh",
    libMtimeMs: libStat.mtimeMs,
    newestSourcePath: newestSource.path,
    newestSourceMtimeMs: newestSource.mtimeMs,
  });
}

module.exports = {
  nativeLibraryPath,
  newestNativeSourceMtimeMs,
  verifyFreshNativeLibrary,
};
