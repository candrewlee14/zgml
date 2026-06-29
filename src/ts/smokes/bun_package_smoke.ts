"use strict";

declare const require: (id: string) => any;

const zgml = require([["zg", "ml"].join(""), "bun"].join("/"));
const { smokePackage } = require([".", "package_smoke_core.cjs"].join("/"));

smokePackage(zgml, "zgml bun package smoke");

export {};
