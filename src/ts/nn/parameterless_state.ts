import { setModuleTraining, type ModuleTrainingTarget } from "../train/module_mode.js";
import type { LoadStateDictOptions, ZeroGradOptions } from "../public_api.js";
import {
  leafChildren,
  leafNamedChildren,
  leafNamedModules,
  moduleListSelf,
} from "./module_tree.js";

type ParameterlessStateHookOptions = Readonly<Record<string, unknown>>;
type StateKeysFn = (source: unknown) => readonly unknown[];
type SetRequiresGradFn = (module: unknown, requiresGrad: boolean) => unknown;
export type ParameterlessStateHooks = Readonly<{
  stateKeys: StateKeysFn;
  setRequiresGrad: SetRequiresGradFn;
}>;
export type ParameterlessStateModule = Readonly<{
  kind?: unknown;
}> & ModuleTrainingTarget;
type ParameterlessStateInstance = ParameterlessStateModule & ParameterlessStatePrototype;
export type ParameterlessStatePrototype = object & {
  parameters?: () => readonly unknown[];
  children?: () => readonly unknown[];
  modules?: () => readonly unknown[];
  namedChildren?: () => readonly unknown[];
  namedModules?: (prefix?: string) => readonly unknown[];
  namedParameters?: () => readonly unknown[];
  parameterNames?: () => readonly string[];
  parameterInfos?: () => readonly unknown[];
  parameterInfo?: typeof parameterInfo;
  zeroGrad?: (this: ParameterlessStateInstance, options?: ZeroGradOptions) => void;
  zero_grad?: (this: ParameterlessStateInstance, options?: ZeroGradOptions) => void;
  requiresGrad_?: (this: ParameterlessStateInstance, requiresGrad?: boolean) => unknown;
  requires_grad_?: (this: ParameterlessStateInstance, requiresGrad?: boolean) => unknown;
  train?: (this: ParameterlessStateInstance, mode?: boolean) => unknown;
  eval?: (this: ParameterlessStateInstance) => unknown;
  stateDict?: (this: ParameterlessStateInstance) => Record<string, never>;
  loadStateDict?: (this: ParameterlessStateInstance, source: unknown, options?: LoadStateDictOptions) => unknown;
};

function parameterInfo(nameOrIndex: unknown) {
  if (typeof nameOrIndex === "number" && (!Number.isInteger(nameOrIndex) || nameOrIndex < 0)) {
    throw new Error("module parameterInfo index must be a non-negative integer");
  }
  if (typeof nameOrIndex !== "number" && typeof nameOrIndex !== "string") {
    throw new Error("module parameterInfo requires a parameter name or index");
  }
  return null;
}

export function createParameterlessStateHooks(options: ParameterlessStateHookOptions, label: string) {
  const stateKeys = options && options.stateKeys;
  const setRequiresGrad = options && options.setRequiresGrad;
  if (typeof stateKeys !== "function" || typeof setRequiresGrad !== "function") {
    throw new Error(`${label} factory requires state and requires-grad hooks`);
  }
  return { stateKeys, setRequiresGrad } as ParameterlessStateHooks;
}

export function installParameterlessStateMethods(
  proto: ParameterlessStatePrototype,
  hooks: ParameterlessStateHooks,
  parameterLabel: (module: ParameterlessStateModule) => unknown,
) {
  proto.parameters = function parameters() { return []; };
  proto.children = function children() { return leafChildren(); };
  proto.modules = function modules() { return moduleListSelf(this); };
  proto.namedChildren = function namedChildren() { return leafNamedChildren(); };
  proto.namedModules = function namedModules(prefix = "") { return leafNamedModules(this, prefix); };
  proto.namedParameters = function namedParameters() { return []; };
  proto.parameterNames = function parameterNames() { return Object.freeze([]); };
  proto.parameterInfos = function parameterInfos() { return Object.freeze([]); };
  proto.parameterInfo = parameterInfo;
  proto.zeroGrad = function zeroGrad(_options: ZeroGradOptions = {}) {};
  proto.zero_grad = function zero_grad(options: ZeroGradOptions = {}) { return this.zeroGrad?.(options); };
  proto.requiresGrad_ = function requiresGrad_(requiresGrad = true) {
    hooks.setRequiresGrad(this, requiresGrad);
    return this;
  };
  proto.requires_grad_ = function requires_grad_(requiresGrad = true) {
    return this.requiresGrad_?.(requiresGrad);
  };
  proto.train = function train(mode = true) { return setModuleTraining(this, mode); };
  proto.eval = function evalModule() { return this.train?.(false); };
  proto.stateDict = function stateDict() { return {}; };
  proto.loadStateDict = function loadStateDict(source: unknown, options: LoadStateDictOptions = {}) {
    if (options.strict !== false && hooks.stateKeys(source).length !== 0) {
      throw new Error(`${parameterLabel(this)} has no parameters`);
    }
    return this;
  };
}
