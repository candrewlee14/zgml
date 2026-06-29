type AnyRecord = Record<string, any>;

export type ModuleTraversalOptions = Readonly<{
  prefix?: unknown;
  recurse?: unknown;
}>;

export function moduleTraversalOptions(prefixOrOptions: unknown = "", options: ModuleTraversalOptions = {}) {
  const source = prefixOrOptions && typeof prefixOrOptions === "object" && !Array.isArray(prefixOrOptions)
    ? prefixOrOptions as ModuleTraversalOptions
    : { prefix: prefixOrOptions };
  const prefix = source.prefix === undefined || source.prefix === null ? "" : String(source.prefix);
  const recurse = (options.recurse ?? source.recurse) !== false;
  return Object.freeze({ prefix, recurse });
}

export function moduleTraversalEntry(name: string, module: unknown) {
  return Object.freeze({ name, module });
}

export function leafChildren() {
  return Object.freeze([]);
}

export function moduleListSelf(module: AnyRecord) {
  return Object.freeze([module]);
}

export function leafNamedChildren() {
  return Object.freeze([]);
}

export function leafNamedModules(module: AnyRecord, prefix = "") {
  return Object.freeze([moduleTraversalEntry(String(prefix), module)]);
}
