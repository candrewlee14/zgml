"use strict";

const { mkdirSync, readdirSync, rmSync, readFileSync, writeFileSync } = require("node:fs");
const { dirname, join, relative, resolve, sep } = require("node:path");
const { spawnSync } = require("node:child_process");
const {
  noDeclarationEntries,
  tsSourceEntries,
} = require("./package_metadata_policy.cjs");

const root = process.cwd();
const tempOut = join(root, ".zig-cache", "package-declarations");
const dist = join(root, "dist");
const sourceRoot = join(root, "src", "ts");
const noDeclaration = new Set(noDeclarationEntries);

function posixPath(path) {
  return path.split(sep).join("/");
}

function declarationTargetForSource(entry) {
  const stem = relative(sourceRoot, resolve(root, entry)).replace(/\.ts$/, ".d.cts");
  return join(dist, stem);
}

function declarationSourceForSource(entry) {
  const stem = relative(sourceRoot, resolve(root, entry)).replace(/\.ts$/, ".d.ts");
  return join(tempOut, stem);
}

function rewriteDeclarationForCjsPackage(source) {
  return source.replace(/(["'][^"']+)\.js(["'])/g, "$1.cjs$2");
}

function removeExistingDeclarations(dir) {
  for (const name of readdirSync(dir, { withFileTypes: true })) {
    const absolute = join(dir, name.name);
    if (name.isDirectory()) {
      removeExistingDeclarations(absolute);
      continue;
    }
    if (name.name.endsWith(".d.cts") || name.name.endsWith(".d.cts.map")) {
      rmSync(absolute);
    }
  }
}

rmSync(tempOut, { recursive: true, force: true });
mkdirSync(tempOut, { recursive: true });

const tsc = spawnSync(
  process.execPath,
  [
    require.resolve("typescript/bin/tsc"),
    "-p",
    "tsconfig.package.json",
    "--emitDeclarationOnly",
    "--declaration",
    "--declarationMap",
    "false",
    "--outDir",
    tempOut,
    "--rootDir",
    "src/ts",
    "--pretty",
    "false",
  ],
  { cwd: root, stdio: "inherit" },
);
if (tsc.status !== 0) {
  process.exit(tsc.status ?? 1);
}

removeExistingDeclarations(dist);

let copied = 0;
for (const entry of tsSourceEntries()) {
  if (entry.startsWith("src/ts/smokes/") || noDeclaration.has(entry)) continue;
  const source = declarationSourceForSource(entry);
  const target = declarationTargetForSource(entry);
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, rewriteDeclarationForCjsPackage(readFileSync(source, "utf8")));
  copied += 1;
}

console.log(`package declarations generated: ${copied} .d.cts files from ${posixPath(relative(root, tempOut))}`);
