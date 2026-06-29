import type {
  CompileOptions,
  LoadStateDictOptions,
  ModuleStateDict,
  ModuleParameterPlacementOptions,
  ModuleTraceOptions,
  TrainEvaluateOptions,
  TrainFitOptions,
  TrainPredictOptions,
} from "../public_api.js";

export type NnModulePrototype = {
  call?: (this: NnModulePrototype, input: unknown) => unknown;
  __call__?: (this: NnModulePrototype, input: unknown) => unknown;
  forward?: (input: unknown) => unknown;
  trace?: (this: NnModulePrototype, traceOptions?: ModuleTraceOptions) => unknown;
  explain?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  compile_support?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  requireCompileSupport?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  require_compile_support?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  compileExplanation?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  compile_explanation?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  compilePlan?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  compile_plan?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  native?: (this: NnModulePrototype, compileOptions?: CompileOptions, bindOptions?: unknown) => unknown;
  inference?: (this: NnModulePrototype, compileOptions?: CompileOptions, bindOptions?: unknown) => unknown;
  forInference?: (this: NnModulePrototype, compileOptions?: CompileOptions, bindOptions?: unknown) => unknown;
  for_inference?: (this: NnModulePrototype, compileOptions?: CompileOptions, bindOptions?: unknown) => unknown;
  compileInference?: (this: NnModulePrototype, compileOptions?: CompileOptions, bindOptions?: unknown) => unknown;
  compile_inference?: (this: NnModulePrototype, compileOptions?: CompileOptions, bindOptions?: unknown) => unknown;
  preflight?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  requireCompilePlan?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  require_compile_plan?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  assertCompilePlan?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  assert_compile_plan?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  compilerSignatures?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  compiler_signatures?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  tensorProgramIr?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  tensor_program_ir?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  kernelPlan?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  kernel_plan?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  bufferLayout?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  buffer_layout?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  memoryLayout?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  memory_layout?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  inputShape?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  input_shape?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  outputShape?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  output_shape?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  shapeConstraints?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  shape_constraints?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  parameterLayout?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  parameter_layout?: (this: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  can_compile?: (this: NnModulePrototype, compileOptions?: CompileOptions) => boolean;
  namedParameters?: (prefix?: string) => unknown;
  named_parameters?: (prefix?: string) => unknown;
  getParameter?: (name: string) => unknown;
  get_parameter?: (name: string) => unknown;
  buffers?: (prefixOrOptions?: unknown, options?: unknown) => unknown;
  namedBuffers?: (prefixOrOptions?: unknown, options?: unknown) => unknown;
  named_buffers?: (prefixOrOptions?: unknown, options?: unknown) => unknown;
  getBuffer?: (name: string) => unknown;
  get_buffer?: (name: string) => unknown;
  namedChildren?: (prefix?: string) => unknown;
  named_children?: (prefix?: string) => unknown;
  namedModules?: (prefix?: string) => unknown;
  named_modules?: (prefix?: string) => unknown;
  getSubmodule?: (name: string) => unknown;
  get_submodule?: (name: string) => unknown;
  apply?: (callback: unknown) => unknown;
  to?: (target?: unknown, options?: unknown) => unknown;
  cpu?: (options?: unknown) => unknown;
  float?: (options?: unknown) => unknown;
  float32?: (options?: unknown) => unknown;
  stateDict?: (prefixOrOptions?: unknown) => unknown;
  state_dict?: (prefixOrOptions?: unknown) => unknown;
  loadStateDict?: (source: ModuleStateDict, options?: LoadStateDictOptions) => unknown;
  load_state_dict?: (source: ModuleStateDict, options?: LoadStateDictOptions) => unknown;
  requiresGrad_?: (requiresGrad?: boolean) => unknown;
  requires_grad_?: (requiresGrad?: boolean) => unknown;
  bindParameters?: (bindOptions?: CompileOptions) => unknown;
  bind_parameters?: (this: NnModulePrototype, bindOptions?: CompileOptions) => unknown;
  placeParameters?: (program: unknown, placementOptions?: ModuleParameterPlacementOptions) => unknown;
  place_parameters?: (this: NnModulePrototype, program: unknown, placementOptions?: ModuleParameterPlacementOptions) => unknown;
  fit?: (this: NnModulePrototype, batches: unknown, fitOptions?: TrainFitOptions) => unknown;
  explainTraining?: (this: NnModulePrototype, batches: unknown, fitOptions?: TrainFitOptions) => unknown;
  explain_training?: (this: NnModulePrototype, batches: unknown, fitOptions?: TrainFitOptions) => unknown;
  explainNativeTraining?: (this: NnModulePrototype, batches: unknown, fitOptions?: TrainFitOptions) => unknown;
  explain_native_training?: (this: NnModulePrototype, batches: unknown, fitOptions?: TrainFitOptions) => unknown;
  canTrainNative?: (this: NnModulePrototype, batches: unknown, fitOptions?: TrainFitOptions) => boolean;
  can_train_native?: (this: NnModulePrototype, batches: unknown, fitOptions?: TrainFitOptions) => boolean;
  fitNative?: (this: NnModulePrototype, batches: unknown, fitOptions?: TrainFitOptions) => unknown;
  fit_native?: (this: NnModulePrototype, batches: unknown, fitOptions?: TrainFitOptions) => unknown;
  evaluate?: (this: NnModulePrototype, batches: unknown, criterion: unknown, evaluateOptions?: TrainEvaluateOptions) => unknown;
  evalModule?: (this: NnModulePrototype, batches: unknown, criterion: unknown, evaluateOptions?: TrainEvaluateOptions) => unknown;
  eval_module?: (this: NnModulePrototype, batches: unknown, criterion: unknown, evaluateOptions?: TrainEvaluateOptions) => unknown;
  predict?: (this: NnModulePrototype, batches: unknown, predictOptions?: TrainPredictOptions) => unknown;
};

export type NnModulePrototypeConstructor = Readonly<{
  prototype?: object | null;
}>;

export type NnCompileEvidenceMethodHooks = Readonly<{
  traceSequentialProgram: (layers: readonly NnModulePrototype[], traceOptions?: ModuleTraceOptions) => unknown;
  compileSupportForModule: (target: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  requireCompileSupportForModule: (target: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  compilerSignaturesForModule: (target: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  tensorProgramIrForModule: (target: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  kernelPlanForModule: (target: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  bufferLayoutForModule: (target: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  memoryLayoutForModule: (target: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  inputShapeForModule: (target: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  outputShapeForModule: (target: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  shapeConstraintsForModule: (target: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  parameterLayoutForModule: (target: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  explainModule: (target: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  requireCompilePlanForModule: (target: NnModulePrototype, compileOptions?: CompileOptions) => unknown;
  canCompileModule: (target: NnModulePrototype, compileOptions?: CompileOptions) => boolean;
  nativeInferenceForModule: (target: NnModulePrototype, compileOptions?: CompileOptions, bindOptions?: unknown) => unknown;
  fitModule?: (target: NnModulePrototype, batches: unknown, fitOptions?: TrainFitOptions) => unknown;
  explainNativeModule?: (target: NnModulePrototype, batches: unknown, fitOptions?: TrainFitOptions) => unknown;
  canTrainNativeModule?: (target: NnModulePrototype, batches: unknown, fitOptions?: TrainFitOptions) => boolean;
  fitNativeModule?: (target: NnModulePrototype, batches: unknown, fitOptions?: TrainFitOptions) => unknown;
  evaluateModule?: (target: NnModulePrototype, batches: unknown, criterion: unknown, evaluateOptions?: TrainEvaluateOptions) => unknown;
  predictModule?: (target: NnModulePrototype, batches: unknown, predictOptions?: TrainPredictOptions) => unknown;
}>;

export function installNnCompileEvidenceMethods(constructors: readonly NnModulePrototypeConstructor[], options: NnCompileEvidenceMethodHooks) {
  const traceSequentialProgram = options.traceSequentialProgram;
  const compileSupportForModule = options.compileSupportForModule;
  const requireCompileSupportForModule = options.requireCompileSupportForModule;
  const compilerSignaturesForModule = options.compilerSignaturesForModule;
  const tensorProgramIrForModule = options.tensorProgramIrForModule;
  const kernelPlanForModule = options.kernelPlanForModule;
  const bufferLayoutForModule = options.bufferLayoutForModule;
  const memoryLayoutForModule = options.memoryLayoutForModule;
  const inputShapeForModule = options.inputShapeForModule;
  const outputShapeForModule = options.outputShapeForModule;
  const shapeConstraintsForModule = options.shapeConstraintsForModule;
  const parameterLayoutForModule = options.parameterLayoutForModule;
  const explainModule = options.explainModule;
  const requireCompilePlanForModule = options.requireCompilePlanForModule;
  const canCompileModule = options.canCompileModule;
  const nativeInferenceForModule = options.nativeInferenceForModule;
  const fitModule = options.fitModule;
  const explainNativeModule = options.explainNativeModule;
  const canTrainNativeModule = options.canTrainNativeModule;
  const fitNativeModule = options.fitNativeModule;
  const evaluateModule = options.evaluateModule;
  const predictModule = options.predictModule;
  if (
    typeof traceSequentialProgram !== "function" ||
    typeof compileSupportForModule !== "function" ||
    typeof requireCompileSupportForModule !== "function" ||
    typeof compilerSignaturesForModule !== "function" ||
    typeof tensorProgramIrForModule !== "function" ||
    typeof kernelPlanForModule !== "function" ||
    typeof bufferLayoutForModule !== "function" ||
    typeof memoryLayoutForModule !== "function" ||
    typeof inputShapeForModule !== "function" ||
    typeof outputShapeForModule !== "function" ||
    typeof shapeConstraintsForModule !== "function" ||
    typeof parameterLayoutForModule !== "function" ||
    typeof explainModule !== "function" ||
    typeof requireCompilePlanForModule !== "function" ||
    typeof canCompileModule !== "function" ||
    typeof nativeInferenceForModule !== "function"
  ) {
    throw new Error("nn compile evidence method installer requires trace and compile inspection helpers");
  }

  for (const ModuleClass of constructors) {
    const protoSource = ModuleClass && ModuleClass.prototype;
    if (!protoSource) continue;
    const proto = protoSource as NnModulePrototype;
    if (typeof proto.trace !== "function") {
      proto.trace = function trace(this: NnModulePrototype, traceOptions: ModuleTraceOptions = {}) {
        return traceSequentialProgram([this], traceOptions);
      };
    }
    if (typeof proto.explain !== "function") {
      proto.explain = function explain(this: NnModulePrototype, compileOptions: CompileOptions = {}) {
        return explainModule(this, compileOptions);
      };
    }
    proto.compile_support = function compile_support(this: NnModulePrototype, compileOptions: CompileOptions = {}) {
      return compileSupportForModule(this, compileOptions);
    };
    proto.requireCompileSupport = function requireCompileSupport(this: NnModulePrototype, compileOptions: CompileOptions = {}) {
      return requireCompileSupportForModule(this, compileOptions);
    };
    proto.require_compile_support = proto.requireCompileSupport;
    proto.compileExplanation = proto.explain;
    proto.compile_explanation = proto.explain;
    proto.compilePlan = proto.explain;
    proto.compile_plan = proto.explain;
    proto.preflight = proto.explain;
    proto.requireCompilePlan = function requireCompilePlan(this: NnModulePrototype, compileOptions: CompileOptions = {}) {
      return requireCompilePlanForModule(this, compileOptions);
    };
    proto.require_compile_plan = proto.requireCompilePlan;
    proto.assertCompilePlan = proto.requireCompilePlan;
    proto.assert_compile_plan = proto.requireCompilePlan;
    proto.compilerSignatures = function compilerSignatures(this: NnModulePrototype, compileOptions: CompileOptions = {}) {
      return compilerSignaturesForModule(this, compileOptions);
    };
    proto.compiler_signatures = proto.compilerSignatures;
    proto.tensorProgramIr = function tensorProgramIr(this: NnModulePrototype, compileOptions: CompileOptions = {}) {
      return tensorProgramIrForModule(this, compileOptions);
    };
    proto.tensor_program_ir = proto.tensorProgramIr;
    proto.kernelPlan = function kernelPlan(this: NnModulePrototype, compileOptions: CompileOptions = {}) {
      return kernelPlanForModule(this, compileOptions);
    };
    proto.kernel_plan = proto.kernelPlan;
    proto.bufferLayout = function bufferLayout(this: NnModulePrototype, compileOptions: CompileOptions = {}) {
      return bufferLayoutForModule(this, compileOptions);
    };
    proto.buffer_layout = proto.bufferLayout;
    proto.memoryLayout = function memoryLayout(this: NnModulePrototype, compileOptions: CompileOptions = {}) {
      return memoryLayoutForModule(this, compileOptions);
    };
    proto.memory_layout = proto.memoryLayout;
    proto.inputShape = function inputShape(this: NnModulePrototype, compileOptions: CompileOptions = {}) {
      return inputShapeForModule(this, compileOptions);
    };
    proto.input_shape = proto.inputShape;
    proto.outputShape = function outputShape(this: NnModulePrototype, compileOptions: CompileOptions = {}) {
      return outputShapeForModule(this, compileOptions);
    };
    proto.output_shape = proto.outputShape;
    proto.shapeConstraints = function shapeConstraints(this: NnModulePrototype, compileOptions: CompileOptions = {}) {
      return shapeConstraintsForModule(this, compileOptions);
    };
    proto.shape_constraints = proto.shapeConstraints;
    proto.parameterLayout = function parameterLayout(this: NnModulePrototype, compileOptions: CompileOptions = {}) {
      return parameterLayoutForModule(this, compileOptions);
    };
    proto.parameter_layout = proto.parameterLayout;
    proto.can_compile = function can_compile(this: NnModulePrototype, compileOptions: CompileOptions = {}) {
      return canCompileModule(this, compileOptions);
    };
    proto.native = function native(this: NnModulePrototype, compileOptions: CompileOptions = {}, bindOptions?: unknown) {
      return nativeInferenceForModule(this, compileOptions, bindOptions);
    };
    proto.inference = proto.native;
    proto.forInference = proto.native;
    proto.for_inference = proto.native;
    proto.compileInference = proto.native;
    proto.compile_inference = proto.native;
    proto.fit = function fit(this: NnModulePrototype, batches: unknown, fitOptions: TrainFitOptions = {}) {
      if (typeof fitModule !== "function") throw new Error("nn.Module.fit requires a train.fit runtime");
      return fitModule(this, batches, fitOptions);
    };
    proto.explainTraining = function explainTraining(this: NnModulePrototype, batches: unknown, fitOptions: TrainFitOptions = {}) {
      if (typeof explainNativeModule !== "function") throw new Error("nn.Module.explainTraining requires a train.explainNative runtime");
      return explainNativeModule(this, batches, fitOptions);
    };
    proto.explain_training = proto.explainTraining;
    proto.explainNativeTraining = proto.explainTraining;
    proto.explain_native_training = proto.explainTraining;
    proto.canTrainNative = function canTrainNative(this: NnModulePrototype, batches: unknown, fitOptions: TrainFitOptions = {}) {
      if (typeof canTrainNativeModule === "function") return canTrainNativeModule(this, batches, fitOptions);
      if (typeof explainNativeModule !== "function") throw new Error("nn.Module.canTrainNative requires a train.canTrainNative or train.explainNative runtime");
      const explanation = explainNativeModule(this, batches, fitOptions) as { supported?: unknown };
      return explanation?.supported === true;
    };
    proto.can_train_native = proto.canTrainNative;
    proto.fitNative = function fitNative(this: NnModulePrototype, batches: unknown, fitOptions: TrainFitOptions = {}) {
      if (typeof fitNativeModule !== "function") throw new Error("nn.Module.fitNative requires a train.fitNative runtime");
      return fitNativeModule(this, batches, fitOptions);
    };
    proto.fit_native = proto.fitNative;
    proto.evaluate = function evaluate(this: NnModulePrototype, batches: unknown, criterion: unknown, evaluateOptions: TrainEvaluateOptions = {}) {
      if (typeof evaluateModule !== "function") throw new Error("nn.Module.evaluate requires a train.evaluateModule runtime");
      return evaluateModule(this, batches, criterion, evaluateOptions);
    };
    proto.evalModule = proto.evaluate;
    proto.eval_module = proto.evaluate;
    proto.predict = function predict(this: NnModulePrototype, batches: unknown, predictOptions: TrainPredictOptions = {}) {
      if (typeof predictModule !== "function") throw new Error("nn.Module.predict requires a train.predictModule runtime");
      return predictModule(this, batches, predictOptions);
    };
  }
}

export function installNnPyTorchAliasMethods(constructors: readonly NnModulePrototypeConstructor[]) {
  function normalizeModulePlacement(target: unknown, options: unknown = {}) {
    const targetRecord = target && typeof target === "object" ? target as Record<string, unknown> : null;
    const optionsRecord = options && typeof options === "object" ? options as Record<string, unknown> : null;
    const device = targetRecord
      ? targetRecord.device ?? targetRecord.backend ?? targetRecord.placement
      : target;
    const dtype = targetRecord?.dtype;
    const fallbackDevice = optionsRecord?.device ?? optionsRecord?.backend ?? optionsRecord?.placement;
    const placement = device ?? fallbackDevice ?? dtype ?? optionsRecord?.dtype ?? "cpu";
    if (
      placement === "cpu" ||
      placement === "host" ||
      placement === "f32" ||
      placement === "float" ||
      placement === "float32" ||
      placement === undefined ||
      placement === null
    ) return;
    throw new Error(`nn.Module.to currently supports only cpu/f32 placement, got ${String(placement)}`);
  }

  for (const ModuleClass of constructors) {
    const protoSource = ModuleClass && ModuleClass.prototype;
    if (!protoSource) continue;
    const proto = protoSource as NnModulePrototype;
    proto.call = function call(this: NnModulePrototype, input: unknown) {
      const forward = this.forward;
      if (typeof forward !== "function") throw new Error("nn.call requires forward on module prototype");
      return forward.call(this, input);
    };
    proto.__call__ = proto.call;
    proto.named_parameters = function named_parameters(this: NnModulePrototype, prefix = "") {
      const namedParameters = this.namedParameters;
      if (typeof namedParameters !== "function") throw new Error("nn.named_parameters requires namedParameters on module prototype");
      return namedParameters.call(this, prefix);
    };
    proto.getParameter = function getParameter(this: NnModulePrototype, name: string) {
      if (typeof name !== "string") throw new Error("nn.getParameter requires a string name");
      const namedParameters = this.namedParameters;
      if (typeof namedParameters !== "function") throw new Error("nn.getParameter requires namedParameters on module prototype");
      const entries = namedParameters.call(this, "") as unknown;
      if (!Array.isArray(entries)) throw new Error("nn.getParameter requires namedParameters to return an array");
      return (entries as readonly { name?: unknown }[]).find((entry) => entry && entry.name === name) ?? null;
    };
    proto.get_parameter = proto.getParameter;
    if (typeof proto.buffers !== "function") {
      proto.buffers = function buffers() {
        return [];
      };
    }
    if (typeof proto.namedBuffers !== "function") {
      proto.namedBuffers = function namedBuffers(this: NnModulePrototype, prefixOrOptions: unknown = "", options = {}) {
        const buffers = this.buffers;
        if (typeof buffers !== "function") throw new Error("nn.namedBuffers requires buffers on module prototype");
        return buffers.call(this, prefixOrOptions, options);
      };
    }
    proto.named_buffers = function named_buffers(this: NnModulePrototype, prefixOrOptions: unknown = "", options = {}) {
      const namedBuffers = this.namedBuffers;
      if (typeof namedBuffers !== "function") throw new Error("nn.named_buffers requires namedBuffers on module prototype");
      return namedBuffers.call(this, prefixOrOptions, options);
    };
    proto.getBuffer = function getBuffer(this: NnModulePrototype, name: string) {
      if (typeof name !== "string") throw new Error("nn.getBuffer requires a string name");
      const namedBuffers = this.namedBuffers;
      if (typeof namedBuffers !== "function") throw new Error("nn.getBuffer requires namedBuffers on module prototype");
      const entries = namedBuffers.call(this, "") as unknown;
      if (!Array.isArray(entries)) throw new Error("nn.getBuffer requires namedBuffers to return an array");
      return (entries as readonly { name?: unknown }[]).find((entry) => entry && entry.name === name) ?? null;
    };
    proto.get_buffer = proto.getBuffer;
    proto.requires_grad_ = function requires_grad_(this: NnModulePrototype, requiresGrad = true) {
      const requiresGradMethod = this.requiresGrad_;
      if (typeof requiresGradMethod !== "function") throw new Error("nn.requires_grad_ requires requiresGrad_ on module prototype");
      return requiresGradMethod.call(this, requiresGrad);
    };
    proto.named_children = function named_children(this: NnModulePrototype, prefix = "") {
      const namedChildren = this.namedChildren;
      if (typeof namedChildren !== "function") throw new Error("nn.named_children requires namedChildren on module prototype");
      return namedChildren.call(this, prefix);
    };
    proto.named_modules = function named_modules(this: NnModulePrototype, prefix = "") {
      const namedModules = this.namedModules;
      if (typeof namedModules !== "function") throw new Error("nn.named_modules requires namedModules on module prototype");
      return namedModules.call(this, prefix);
    };
    proto.getSubmodule = function getSubmodule(this: NnModulePrototype, name: string) {
      if (typeof name !== "string") throw new Error("nn.getSubmodule requires a string name");
      const namedModules = this.namedModules;
      if (typeof namedModules !== "function") throw new Error("nn.getSubmodule requires namedModules on module prototype");
      const entries = namedModules.call(this, "") as unknown;
      if (!Array.isArray(entries)) throw new Error("nn.getSubmodule requires namedModules to return an array");
      const entry = (entries as readonly { name?: unknown; module?: unknown }[]).find((candidate) => candidate && candidate.name === name);
      return entry && entry.module !== undefined ? entry.module : null;
    };
    proto.get_submodule = proto.getSubmodule;
    proto.apply = function apply(this: NnModulePrototype, callback: unknown) {
      if (typeof callback !== "function") throw new Error("nn.apply requires a function");
      const namedModules = this.namedModules;
      if (typeof namedModules !== "function") throw new Error("nn.apply requires namedModules on module prototype");
      const entries = namedModules.call(this, "") as unknown;
      if (!Array.isArray(entries)) throw new Error("nn.apply requires namedModules to return an array");
      for (const entry of entries as readonly { name?: unknown; module?: unknown }[]) {
        const module = entry && entry.module !== undefined ? entry.module : entry;
        callback(module, { name: String(entry?.name ?? ""), module });
      }
      return this;
    };
    proto.to = function to(this: NnModulePrototype, target?: unknown, options?: unknown) {
      normalizeModulePlacement(target, options);
      return this;
    };
    proto.cpu = function cpu(this: NnModulePrototype, options?: unknown) {
      return this.to?.("cpu", options);
    };
    proto.float = function float(this: NnModulePrototype, options?: unknown) {
      return this.to?.("float32", options);
    };
    proto.float32 = function float32(this: NnModulePrototype, options?: unknown) {
      return this.to?.("float32", options);
    };
    proto.state_dict = function state_dict(this: NnModulePrototype, prefixOrOptions: unknown = "") {
      const stateDict = this.stateDict;
      if (typeof stateDict !== "function") throw new Error("nn.state_dict requires stateDict on module prototype");
      return stateDict.call(this, prefixOrOptions);
    };
    proto.load_state_dict = function load_state_dict(this: NnModulePrototype, source: ModuleStateDict, loadOptions: LoadStateDictOptions = {}) {
      const loadStateDict = this.loadStateDict;
      if (typeof loadStateDict !== "function") throw new Error("nn.load_state_dict requires loadStateDict on module prototype");
      return loadStateDict.call(this, source, loadOptions);
    };
    proto.bind_parameters = function bind_parameters(this: NnModulePrototype, bindOptions: CompileOptions = {}) {
      const bindParameters = this.bindParameters;
      if (typeof bindParameters !== "function") throw new Error("nn.bind_parameters requires bindParameters on module prototype");
      return bindParameters.call(this, bindOptions);
    };
    proto.place_parameters = function place_parameters(this: NnModulePrototype, program: unknown, placementOptions: ModuleParameterPlacementOptions = {}) {
      const placeParameters = this.placeParameters;
      if (typeof placeParameters !== "function") throw new Error("nn.place_parameters requires placeParameters on module prototype");
      return placeParameters.call(this, program, placementOptions);
    };
  }
}

export function installNnModulePrototypeMethods(constructors: readonly NnModulePrototypeConstructor[], options: NnCompileEvidenceMethodHooks) {
  installNnCompileEvidenceMethods(constructors, options);
  installNnPyTorchAliasMethods(constructors);
}
