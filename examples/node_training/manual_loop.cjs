"use strict";

const {
  checkpoint,
  data,
  loss,
  nn,
  optim,
  tensor,
  train,
} = require("../..");

function scalar(t) {
  return typeof t.item === "function" ? t.item() : t.data[0];
}

function assertClose(actual, expected, tolerance, label) {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${label}: expected ${expected} +/- ${tolerance}, got ${actual}`);
  }
}

class FieldParameterHead extends nn.Module {
  constructor() {
    super({ kind: "field-parameter-head" });
    this.weight = nn.Parameter("weight", [0], [1]);
    this.bias = nn.Parameter("bias", [0], [1]);
  }

  forward(input) {
    return input.mul(this.weight.tensor).add(this.bias.tensor);
  }
}

class RegisteredLinearHead extends nn.Module {
  constructor() {
    super({ kind: "registered-linear-head" });
    this.addModule("head", nn.linear(1, 1, { weights: [2], bias: [0] }));
  }

  forward(input) {
    return this.head.forward(input);
  }
}

class RegisteredParameterHead extends nn.Module {
  constructor() {
    super({ kind: "registered-parameter-head" });
    this.registerParameter("weight", nn.Parameter("weight", [0], [1]));
    this.register_parameter("bias", nn.Parameter("bias", [0], [1]));
  }

  forward(input) {
    return input.mul(this.weight.tensor).add(this.bias.tensor);
  }
}

class BufferedHead extends nn.Module {
  constructor() {
    super({ kind: "buffered-head" });
    this.offset = nn.Buffer("offset", [1], [1]);
    this.registerBuffer("scale", nn.Buffer("scale", [2], [1]));
  }

  forward(input) {
    return input.mul(tensor(this.scale.data, [1])).add(tensor(this.offset.data, [1]));
  }
}

class ModuleListHead extends nn.Module {
  constructor() {
    super({ kind: "module-list-head" });
    this.layers = new nn.ModuleList([
      nn.linear(1, 1, { weights: [2], bias: [0] }),
    ]);
  }

  forward(input) {
    return this.layers.at(0).forward(input);
  }
}

class ParameterListHead extends nn.Module {
  constructor() {
    super({ kind: "parameter-list-head" });
    this.params = nn.parameterList([
      nn.Parameter("weight", [0], [1]),
      nn.Parameter("bias", [0], [1]),
    ]);
  }

  forward(input) {
    return input.mul(this.params.at(0).tensor).add(this.params.at(1).tensor);
  }
}

class ModuleDictHead extends nn.Module {
  constructor() {
    super({ kind: "module-dict-head" });
    this.layers = new nn.ModuleDict({
      head: nn.linear(1, 1, { weights: [2], bias: [0] }),
    });
  }

  forward(input) {
    return this.layers.get("head").forward(input);
  }
}

class ParameterDictHead extends nn.Module {
  constructor() {
    super({ kind: "parameter-dict-head" });
    this.params = nn.parameterDict({
      weight: nn.Parameter("weight", [0], [1]),
      bias: nn.Parameter("bias", [0], [1]),
    });
  }

  forward(input) {
    return input.mul(this.params.get("weight").tensor).add(this.params.get("bias").tensor);
  }
}

const fieldParameterModel = new FieldParameterHead();
const fieldParameterNames = fieldParameterModel.parameterNames("fieldParam").join("|");
if (fieldParameterNames !== "fieldParam.weight|fieldParam.bias") {
  throw new Error(`field-owned nn.Parameter registration failed: ${fieldParameterNames}`);
}
const fieldParameterState = fieldParameterModel.stateDict("fieldParam");
if (!fieldParameterState["fieldParam.weight"] || !fieldParameterState["fieldParam.bias"]) {
  throw new Error(`field-owned nn.Parameter state dict failed: ${Object.keys(fieldParameterState).join("|")}`);
}
const fieldParameterOptimizer = optim.sgd(fieldParameterModel, { lr: 0.1 });
const fieldParameterLoss = loss.mse(fieldParameterModel.forward(tensor([2], [1])), tensor([3], [1]));
fieldParameterLoss.backward();
if (train.gradNorm(fieldParameterModel) <= 0) throw new Error("field-owned nn.Parameter gradients were not visible to train.gradNorm");
fieldParameterOptimizer.step();
fieldParameterOptimizer.zeroGrad();
if (train.gradNorm(fieldParameterModel) !== 0) throw new Error("field-owned nn.Parameter optimizer.zeroGrad failed");

const registeredLinear = new RegisteredLinearHead();
const registeredLinearNames = registeredLinear.parameterNames("registered").join("|");
if (registeredLinearNames !== "registered.head.weight|registered.head.bias") {
  throw new Error(`explicit registered child module traversal failed: ${registeredLinearNames}`);
}
if (registeredLinear.getSubmodule("head") !== registeredLinear.head || nn.getSubmodule(registeredLinear, "head") !== registeredLinear.head) {
  throw new Error("registered child module lookup failed");
}
if (!registeredLinear.getParameter("head.weight") || !nn.getParameter(registeredLinear, "head.bias")) {
  throw new Error("registered child parameter lookup failed");
}
const registeredLinearProgram = registeredLinear.compile({ backend: "cpu", inputShape: [1] });
const registeredLinearSession = registeredLinearProgram.bindModule(registeredLinear);
const registeredLinearInput = tensor([3], [1]);
const registeredLinearCompiled = registeredLinearSession.stepTensor(registeredLinearInput);
assertClose(scalar(registeredLinearCompiled), scalar(registeredLinear.forward(registeredLinearInput)), 1e-5, "compiled registered child module");
registeredLinearSession.free();
registeredLinearProgram.free();

const registeredParameterModel = new RegisteredParameterHead();
const registeredParameterNames = registeredParameterModel.parameterNames("registeredParam").join("|");
if (registeredParameterNames !== "registeredParam.weight|registeredParam.bias") {
  throw new Error(`explicit registered nn.Parameter traversal failed: ${registeredParameterNames}`);
}
const registeredParameterState = registeredParameterModel.stateDict("registeredParam");
if (!registeredParameterState["registeredParam.weight"] || !registeredParameterState["registeredParam.bias"]) {
  throw new Error(`explicit registered nn.Parameter state dict failed: ${Object.keys(registeredParameterState).join("|")}`);
}
const registeredParameterOptimizer = optim.sgd(registeredParameterModel, { lr: 0.1 });
const registeredParameterLoss = loss.mse(registeredParameterModel.forward(tensor([2], [1])), tensor([3], [1]));
registeredParameterLoss.backward();
if (train.gradNorm(registeredParameterModel) <= 0) throw new Error("explicit registered nn.Parameter gradients were not visible");
registeredParameterOptimizer.step();
registeredParameterOptimizer.zeroGrad();
if (train.gradNorm(registeredParameterModel) !== 0) throw new Error("explicit registered nn.Parameter optimizer.zeroGrad failed");

const bufferedModel = new BufferedHead();
const bufferedNames = bufferedModel.namedBuffers("buffered").map((entry) => entry.name).join("|");
if (bufferedNames !== "buffered.offset|buffered.scale") {
  throw new Error(`nn.Buffer traversal failed: ${bufferedNames}`);
}
if (bufferedModel.getBuffer("offset") !== bufferedModel.offset || nn.getBuffer(bufferedModel, "scale") !== bufferedModel.scale) {
  throw new Error("nn.Buffer lookup failed");
}
if (bufferedModel.parameterNames("buffered").join("|") !== "") {
  throw new Error("nn.Buffer must not appear in parameter traversal");
}
const bufferedState = bufferedModel.stateDict("buffered");
if (!bufferedState["buffered.offset"] || !bufferedState["buffered.scale"]) {
  throw new Error(`nn.Buffer state dict failed: ${Object.keys(bufferedState).join("|")}`);
}
const bufferedNamespaceState = nn.stateDict(bufferedModel, "buffered");
if (!bufferedNamespaceState["buffered.offset"] || !bufferedNamespaceState["buffered.scale"]) {
  throw new Error(`nn.stateDict buffer entries failed: ${Object.keys(bufferedNamespaceState).join("|")}`);
}
bufferedModel.offset.data[0] = 10;
bufferedModel.scale.data[0] = 20;
bufferedModel.loadStateDict(bufferedState, { prefix: "buffered", strict: true });
assertClose(bufferedModel.offset.data[0], 1, 1e-6, "field-owned nn.Buffer restore");
assertClose(bufferedModel.scale.data[0], 2, 1e-6, "registered nn.Buffer restore");
bufferedModel.offset.data[0] = 11;
bufferedModel.scale.data[0] = 21;
nn.loadStateDict(bufferedModel, bufferedNamespaceState, { prefix: "buffered", strict: true });
assertClose(bufferedModel.offset.data[0], 1, 1e-6, "namespace nn.Buffer restore");
assertClose(bufferedModel.scale.data[0], 2, 1e-6, "namespace registered nn.Buffer restore");
const bufferedNamespaceNames = nn.namedBuffers(bufferedModel, "buffered").map((entry) => entry.name).join("|");
if (bufferedNamespaceNames !== bufferedNames) {
  throw new Error(`nn.namedBuffers traversal failed: ${bufferedNamespaceNames}`);
}
const bufferedCheckpoint = checkpoint.create({ model: bufferedModel, prefix: "buffered" });
const bufferedCheckpointNames = checkpoint.inspect(bufferedCheckpoint).modelParameterNames.join("|");
if (bufferedCheckpointNames !== "buffered.offset|buffered.scale") {
  throw new Error(`checkpoint buffer state failed: ${bufferedCheckpointNames}`);
}
bufferedModel.offset.data[0] = 12;
bufferedModel.scale.data[0] = 22;
checkpoint.load(bufferedCheckpoint, { model: bufferedModel, prefix: "buffered", strict: true });
assertClose(bufferedModel.offset.data[0], 1, 1e-6, "checkpoint nn.Buffer restore");
assertClose(bufferedModel.scale.data[0], 2, 1e-6, "checkpoint registered nn.Buffer restore");

const moduleListModel = new ModuleListHead();
const moduleListNames = moduleListModel.parameterNames("moduleList").join("|");
if (moduleListNames !== "moduleList.layers.0.weight|moduleList.layers.0.bias") {
  throw new Error(`nn.ModuleList traversal failed: ${moduleListNames}`);
}
const moduleListChildren = moduleListModel.namedChildren("moduleList").map((entry) => entry.name).join("|");
if (moduleListChildren !== "moduleList.layers.0") {
  throw new Error(`nn.ModuleList named child traversal failed: ${moduleListChildren}`);
}
if (moduleListModel.getSubmodule("layers.0") !== moduleListModel.layers.at(0) || nn.getParameter(moduleListModel, "layers.0.weight") === null) {
  throw new Error("nn.ModuleList direct lookup failed");
}
const moduleListApplyNames = [];
const moduleListApplyResult = moduleListModel.apply((_module, entry) => moduleListApplyNames.push(entry.name));
if (moduleListApplyResult !== moduleListModel || moduleListApplyNames.join("|") !== "|layers.0") {
  throw new Error(`nn.Module.apply traversal failed for ModuleList: ${moduleListApplyNames.join("|")}`);
}
const moduleListProgram = moduleListModel.compile({ backend: "cpu", inputShape: [1] });
const moduleListSession = moduleListProgram.bindModule(moduleListModel);
const moduleListInput = tensor([4], [1]);
const moduleListCompiled = moduleListSession.stepTensor(moduleListInput);
assertClose(scalar(moduleListCompiled), scalar(moduleListModel.forward(moduleListInput)), 1e-5, "compiled nn.ModuleList child module");
moduleListSession.free();
moduleListProgram.free();

const parameterListModel = new ParameterListHead();
const parameterListNames = parameterListModel.parameterNames("parameterList").join("|");
if (parameterListNames !== "parameterList.params.0|parameterList.params.1") {
  throw new Error(`nn.ParameterList traversal failed: ${parameterListNames}`);
}
const parameterListState = parameterListModel.stateDict("parameterList");
if (!parameterListState["parameterList.params.0"] || !parameterListState["parameterList.params.1"]) {
  throw new Error(`nn.ParameterList state dict failed: ${Object.keys(parameterListState).join("|")}`);
}
if (parameterListModel.params.getParameter("0") !== parameterListModel.params.at(0) || nn.getParameter(parameterListModel, "params.1") === null) {
  throw new Error("nn.ParameterList direct lookup failed");
}
if (parameterListModel.params.getBuffer("0") !== null || parameterListModel.params.get_buffer("1") !== null) {
  throw new Error("nn.ParameterList buffer lookup should be explicit and empty");
}
const parameterListOptimizer = optim.sgd(parameterListModel, { lr: 0.1 });
const parameterListLoss = loss.mse(parameterListModel.forward(tensor([2], [1])), tensor([3], [1]));
parameterListLoss.backward();
if (train.gradNorm(parameterListModel) <= 0) throw new Error("nn.ParameterList gradients were not visible");
parameterListOptimizer.step();
parameterListOptimizer.zeroGrad();
if (train.gradNorm(parameterListModel) !== 0) throw new Error("nn.ParameterList optimizer.zeroGrad failed");

const moduleDictModel = new ModuleDictHead();
const moduleDictNames = moduleDictModel.parameterNames("moduleDict").join("|");
if (moduleDictNames !== "moduleDict.layers.head.weight|moduleDict.layers.head.bias") {
  throw new Error(`nn.ModuleDict traversal failed: ${moduleDictNames}`);
}
const moduleDictChildren = moduleDictModel.namedChildren("moduleDict").map((entry) => entry.name).join("|");
if (moduleDictChildren !== "moduleDict.layers.head") {
  throw new Error(`nn.ModuleDict named child traversal failed: ${moduleDictChildren}`);
}
if (moduleDictModel.getSubmodule("layers.head") !== moduleDictModel.layers.get("head") || nn.getParameter(moduleDictModel, "layers.head.bias") === null) {
  throw new Error("nn.ModuleDict direct lookup failed");
}
const moduleDictApplyNames = [];
const moduleDictApplyResult = nn.apply(moduleDictModel, (_module, entry) => moduleDictApplyNames.push(entry.name));
if (moduleDictApplyResult !== moduleDictModel || moduleDictApplyNames.join("|") !== "|layers.head") {
  throw new Error(`nn.apply traversal failed for ModuleDict: ${moduleDictApplyNames.join("|")}`);
}
const moduleDictProgram = moduleDictModel.compile({ backend: "cpu", inputShape: [1] });
const moduleDictSession = moduleDictProgram.bindModule(moduleDictModel);
const moduleDictInput = tensor([5], [1]);
const moduleDictCompiled = moduleDictSession.stepTensor(moduleDictInput);
assertClose(scalar(moduleDictCompiled), scalar(moduleDictModel.forward(moduleDictInput)), 1e-5, "compiled nn.ModuleDict child module");
moduleDictSession.free();
moduleDictProgram.free();

const parameterDictModel = new ParameterDictHead();
const parameterDictNames = parameterDictModel.parameterNames("parameterDict").join("|");
if (parameterDictNames !== "parameterDict.params.weight|parameterDict.params.bias") {
  throw new Error(`nn.ParameterDict traversal failed: ${parameterDictNames}`);
}
const parameterDictState = parameterDictModel.stateDict("parameterDict");
if (!parameterDictState["parameterDict.params.weight"] || !parameterDictState["parameterDict.params.bias"]) {
  throw new Error(`nn.ParameterDict state dict failed: ${Object.keys(parameterDictState).join("|")}`);
}
if (parameterDictModel.params.getParameter("weight") !== parameterDictModel.params.get("weight") || nn.getParameter(parameterDictModel, "params.bias") === null) {
  throw new Error("nn.ParameterDict direct lookup failed");
}
if (parameterDictModel.params.getBuffer("weight") !== null || parameterDictModel.params.get_buffer("bias") !== null) {
  throw new Error("nn.ParameterDict buffer lookup should be explicit and empty");
}
const parameterDictOptimizer = optim.sgd(parameterDictModel, { lr: 0.1 });
const parameterDictLoss = loss.mse(parameterDictModel.forward(tensor([2], [1])), tensor([3], [1]));
parameterDictLoss.backward();
if (train.gradNorm(parameterDictModel) <= 0) throw new Error("nn.ParameterDict gradients were not visible");
parameterDictOptimizer.step();
parameterDictOptimizer.zeroGrad();
if (train.gradNorm(parameterDictModel) !== 0) throw new Error("nn.ParameterDict optimizer.zeroGrad failed");

const model = nn.linear(2, 1, { weights: [0, 0], bias: [0] });
let linearApplyCount = 0;
if (model.apply(() => { linearApplyCount += 1; }) !== model || linearApplyCount !== 1) {
  throw new Error(`nn.Module.apply traversal failed for linear module: ${linearApplyCount}`);
}
if (model.getSubmodule("") !== model || model.getParameter("weight") === null || nn.getParameter(model, "bias") === null) {
  throw new Error("linear module direct lookup failed");
}
const layerList = [nn.linear(2, 2), nn.relu()];
const layerListApplyNames = [];
const layerListApplyResult = nn.apply(layerList, (_module, entry) => layerListApplyNames.push(entry.name));
if (typeof layerListApplyResult.forward !== "function" || layerListApplyNames.join("|") !== "|0|1") {
  throw new Error(`nn.apply traversal failed for raw layer list: ${layerListApplyNames.join("|")}`);
}
if (nn.getSubmodule(layerList, "0") !== layerList[0] || nn.getParameter(layerList, "0.weight") === null) {
  throw new Error("raw layer list direct lookup failed");
}
const bias = model.namedParameters().find((param) => param.name === "bias");
if (!bias) throw new Error("manual loop smoke expected a bias parameter");
bias.requiresGrad = false;

const optimizer = optim.sgd(model, { lr: 0.04, momentum: 0.2 });
const samples = data.tensorDataset(
  tensor([
    -1, -1,
    -1, 1,
    1, -1,
    1, 1,
  ], [4, 2]),
  tensor([
    -1,
    -2,
    2,
    1,
  ], [4, 1]),
);
const loader = data.dataLoader(samples, { batchSize: 1, shuffle: true, seed: 31 });
const probeInput = tensor([1, -1], [2]);
const probeTarget = tensor([2], [1]);
const before = scalar(loss.mse(model.forward(probeInput), probeTarget));

let manualSteps = 0;
let sawGradient = false;
for (let epoch = 0; epoch < 40; epoch += 1) {
  for (const batch of loader) {
    optimizer.zeroGrad();
    const objective = loss.mse(model.forward(batch.input.select(0, 0)), batch.target.select(0, 0));
    objective.backward();
    const gradNorm = train.gradNorm(model);
    if (gradNorm > 0) sawGradient = true;
    optimizer.step();
    optimizer.zeroGrad();
    manualSteps += 1;
  }
}

if (!sawGradient) throw new Error("manual loop never observed a non-zero gradient");
if (train.gradNorm(model) !== 0) throw new Error("optimizer.zeroGrad must clear gradients after a manual loop");
if (bias.data[0] !== 0 || bias.requiresGrad !== false) {
  throw new Error(`frozen bias changed during manual training: value=${bias.data[0]} requiresGrad=${bias.requiresGrad}`);
}

optimizer.zeroGrad();
const helperLoss = loss.mse(model.forward(probeInput), probeTarget);
train.backward(helperLoss);
const helperGradNorm = train.gradNorm(model);
if (!(helperGradNorm > 0)) throw new Error(`train.backward helper did not produce gradients: norm=${helperGradNorm}`);
const unclippedNorm = train.clipGradNorm(model, 1000, { eps: 1e-6 });
if (unclippedNorm !== helperGradNorm) {
  throw new Error(`train.clipGradNorm should return the unclipped norm; expected=${helperGradNorm} got=${unclippedNorm}`);
}
if (train.clipGradValue(model, 1000) !== model) throw new Error("train.clipGradValue should return its target");
optimizer.zeroGrad();

const closureLoss = train.lossStep(optimizer, () => loss.mse(model.forward(probeInput), probeTarget), {
  zeroGrad: true,
  zero_grad_options: { set_to_none: false },
  clipGradNorm: 1000,
  clipGradValue: 1000,
});
const closureEvidence = train.inspectStep();
if (
  scalar(closureLoss) < 0 ||
  !train.isTrainStepEvidence(closureEvidence) ||
  closureEvidence.clipGradNormApplied !== true ||
  closureEvidence.clipGradValueApplied !== true ||
  closureEvidence.gradientsCleared !== true
) {
  throw new Error(`unexpected train.lossStep helper evidence: ${JSON.stringify(closureEvidence)}`);
}

const evidenceLoss = loss.mse(model.forward(probeInput), probeTarget);
const stepEvidence = train.step(optimizer, { loss: evidenceLoss, evidence: true, clipGradNorm: 1000 });
if (
  !train.isTrainStepEvidence(stepEvidence) ||
  stepEvidence.optimizerKind !== "sgd" ||
  stepEvidence.hadLoss !== true ||
  stepEvidence.zeroGradApplied !== true ||
  stepEvidence.gradientsCleared !== true ||
  stepEvidence.stepAdvanced !== 1
) {
  throw new Error(`unexpected train.step evidence after manual loop: ${JSON.stringify(stepEvidence)}`);
}

const after = scalar(loss.mse(model.forward(probeInput), probeTarget));
if (!(after < before * 0.02)) {
  throw new Error(`expected manual loop to reduce held-out loss sharply; before=${before}, after=${after}`);
}

const program = model.compile({ backend: "cpu", inputShape: [2] });
const session = program.bindModule(model);
const compiled = session.stepTensor(probeInput);
assertClose(scalar(compiled), scalar(model.forward(probeInput)), 1e-5, "compiled manual-loop prediction");
session.free();
program.free();

console.log(`zgml manual training loop smoke ok: before=${before.toFixed(6)} after=${after.toFixed(6)} manualSteps=${manualSteps} compiled=${scalar(compiled).toFixed(6)}`);
