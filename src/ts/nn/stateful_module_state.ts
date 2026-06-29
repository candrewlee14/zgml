import { setModuleTraining, type ModuleTrainingTarget } from "../train/module_mode.js";
import type { LoadStateDictOptions, ZeroGradOptions } from "../public_api.js";

type ModuleStateHookOptions = Readonly<Record<string, unknown>>;
type StatefulModuleStateModule = ModuleTrainingTarget;
type StatefulModuleStateTrainModule = (module: StatefulModuleStateModule, mode?: boolean) => unknown;
type StatefulModuleStateHooks = Readonly<{
  parameterNames: (module: unknown, prefix?: string) => readonly string[];
  parameterInfos: (module: unknown, prefix?: string) => readonly unknown[];
  parameterInfo: (module: unknown, nameOrIndex: unknown, prefix?: string) => unknown;
  zeroGrad: (module: unknown, options?: ZeroGradOptions) => void;
  setRequiresGrad: (module: unknown, requiresGrad: boolean) => unknown;
  stateDict: (module: unknown, prefixOrOptions?: unknown) => unknown;
  loadStateDict: (module: unknown, source: unknown, options?: LoadStateDictOptions) => unknown;
  trainModule: StatefulModuleStateTrainModule;
}>;
type StatefulModuleStateHookOverrides = Readonly<{
  trainModule?: StatefulModuleStateTrainModule;
}>;
type StatefulModuleStateInstance = StatefulModuleStateModule & {
  zeroGrad(options?: ZeroGradOptions): void;
  train(mode?: boolean): unknown;
};
export type StatefulModuleStatePrototype = object & {
  parameterNames?: (this: StatefulModuleStateInstance, prefix?: string) => readonly string[];
  parameterInfos?: (this: StatefulModuleStateInstance, prefix?: string) => readonly unknown[];
  parameterInfo?: (this: StatefulModuleStateInstance, nameOrIndex: unknown, prefix?: string) => unknown;
  zeroGrad?: (this: StatefulModuleStateInstance, options?: ZeroGradOptions) => void;
  zero_grad?: (this: StatefulModuleStateInstance, options?: ZeroGradOptions) => void;
  requiresGrad_?: (this: StatefulModuleStateInstance, requiresGrad?: boolean) => StatefulModuleStateInstance;
  requires_grad_?: (this: StatefulModuleStateInstance, requiresGrad?: boolean) => StatefulModuleStateInstance;
  train?: (this: StatefulModuleStateInstance, mode?: boolean) => unknown;
  eval?: (this: StatefulModuleStateInstance) => unknown;
  stateDict?: (this: StatefulModuleStateInstance, prefixOrOptions?: unknown) => unknown;
  loadStateDict?: (this: StatefulModuleStateInstance, source: unknown, options?: LoadStateDictOptions) => StatefulModuleStateInstance;
};

export function createStatefulModuleStateHooks(options: ModuleStateHookOptions, label: string, overrides: StatefulModuleStateHookOverrides = {}): StatefulModuleStateHooks {
  const parameterNames = options && options.parameterNames;
  const parameterInfos = options && options.parameterInfos;
  const parameterInfo = options && options.parameterInfo;
  const zeroGrad = options && options.zeroGrad;
  const setRequiresGrad = options && options.setRequiresGrad;
  const stateDict = options && options.stateDict;
  const loadStateDict = options && options.loadStateDict;
  const trainModule = overrides.trainModule ?? setModuleTraining;
  if (
    typeof parameterNames !== "function" ||
    typeof parameterInfos !== "function" ||
    typeof parameterInfo !== "function" ||
    typeof zeroGrad !== "function" ||
    typeof setRequiresGrad !== "function" ||
    typeof stateDict !== "function" ||
    typeof loadStateDict !== "function" ||
    typeof trainModule !== "function"
  ) {
    throw new Error(`${label} factory requires shared module state hooks`);
  }
  return Object.freeze({
    parameterNames,
    parameterInfos,
    parameterInfo,
    zeroGrad,
    setRequiresGrad,
    stateDict,
    loadStateDict,
    trainModule,
  } as StatefulModuleStateHooks);
}

export function installStatefulModuleStateMethods(proto: StatefulModuleStatePrototype, hooks: StatefulModuleStateHooks) {
  proto.parameterNames = function parameterNames(this: StatefulModuleStateInstance, prefix = "") {
    return hooks.parameterNames(this, prefix);
  };
  proto.parameterInfos = function parameterInfos(this: StatefulModuleStateInstance, prefix = "") {
    return hooks.parameterInfos(this, prefix);
  };
  proto.parameterInfo = function parameterInfo(this: StatefulModuleStateInstance, nameOrIndex: unknown, prefix = "") {
    return hooks.parameterInfo(this, nameOrIndex, prefix);
  };
  proto.zeroGrad = function zeroGrad(this: StatefulModuleStateInstance, options: ZeroGradOptions = {}) {
    hooks.zeroGrad(this, options);
  };
  proto.zero_grad = function zero_grad(this: StatefulModuleStateInstance, options: ZeroGradOptions = {}) {
    return this.zeroGrad(options);
  };
  proto.requiresGrad_ = function requiresGrad_(this: StatefulModuleStateInstance, requiresGrad = true) {
    hooks.setRequiresGrad(this, requiresGrad);
    return this;
  };
  proto.requires_grad_ = function requires_grad_(this: StatefulModuleStateInstance, requiresGrad = true) {
    hooks.setRequiresGrad(this, requiresGrad);
    return this;
  };
  proto.train = function train(this: StatefulModuleStateInstance, mode = true) {
    return hooks.trainModule(this, mode);
  };
  proto.eval = function evalModule(this: StatefulModuleStateInstance) {
    return this.train(false);
  };
  proto.stateDict = function stateDict(this: StatefulModuleStateInstance, prefixOrOptions: unknown = "") {
    return hooks.stateDict(this, prefixOrOptions);
  };
  proto.loadStateDict = function loadStateDict(this: StatefulModuleStateInstance, source: unknown, options: LoadStateDictOptions = {}) {
    hooks.loadStateDict(this, source, options);
    return this;
  };
}
