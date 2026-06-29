"use strict";

const {
  nn,
  tensor,
} = require("../..");

function assertClose(actual, expected, tolerance, label) {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${label}: expected ${expected} +/- ${tolerance}, got ${actual}`);
  }
}

function assertHotRuntimeProfile(target, label) {
  const expectation = target.requireHotRuntimeProfile();
  const profile = target.runtimeProfile();
  if (
    expectation.noFallback !== true ||
    expectation.noSync !== true ||
    expectation.noRuntimePatchInvalid !== true ||
    profile.fallbackOpCount !== 0 ||
    profile.syncCount !== 0 ||
    profile.runtimePatchInvalidCount !== 0 ||
    !target.runtimeProfileHasNoFallback() ||
    !target.runtimeProfileHasNoSync() ||
    !target.runtimeProfileHasNoInvalidRuntimePatches()
  ) {
    throw new Error(`${label} runtime profile is not hot-path clean: ${JSON.stringify({ expectation, profile })}`);
  }
}

function assertProgramEvidence(program, model, requirements) {
  const evidence = program.compileEvidence();
  const signatures = program.compilerSignatures();
  const ir = program.tensorProgramIr();
  const plan = program.kernelPlan();
  const bufferLayout = program.bufferLayout();
  const memoryLayout = program.memoryLayout();
  const parameterLayout = program.parameterLayout();
  const compatibility = program.moduleCompatibility(model);
  const executionPlan = program.executionPlan(model);

  if (
    !evidence ||
    !signatures ||
    !ir ||
    !plan ||
    !bufferLayout ||
    !memoryLayout ||
    !parameterLayout ||
    !compatibility.compatible ||
    compatibility.diagnostics.length !== 0 ||
    !executionPlan.canExecute ||
    !executionPlan.acceptsModule
  ) {
    throw new Error(`incomplete Program compile evidence: ${JSON.stringify({ evidence, signatures, compatibility, executionPlan })}`);
  }
  if (
    evidence.irSignature !== signatures.ir ||
    evidence.kernelPlanSignature !== signatures.kernelPlan ||
    evidence.memoryLayoutSignature !== signatures.memoryLayout ||
    evidence.parameterLayoutSignature !== signatures.parameterLayout ||
    evidence.bufferLayoutSignature !== signatures.bufferLayout ||
    evidence.ir.signature !== ir.signature ||
    evidence.kernelPlan.signature !== plan.signature ||
    plan.memoryLayout.signature !== memoryLayout.signature ||
    plan.parameterLayout.signature !== parameterLayout.signature ||
    plan.bufferLayout.signature !== bufferLayout.signature ||
    executionPlan.compileEvidence?.signature !== evidence.signature ||
    executionPlan.kernelPlan?.signature !== plan.signature ||
    executionPlan.memoryLayout?.signature !== memoryLayout.signature ||
    executionPlan.parameterLayout?.signature !== parameterLayout.signature ||
    executionPlan.bufferLayout.signature !== bufferLayout.signature
  ) {
    throw new Error("Program compiler signatures and execution plan must agree");
  }
  if (
    plan.inputLen !== requirements.inputLen ||
    plan.outputLen !== requirements.outputLen ||
    plan.weightsLen !== requirements.weightsLen ||
    plan.biasLen !== requirements.biasLen ||
    bufferLayout.input.elementCount !== requirements.inputLen ||
    bufferLayout.output.elementCount !== requirements.outputLen ||
    bufferLayout.weights.elementCount !== requirements.weightsLen ||
    bufferLayout.bias.elementCount !== requirements.biasLen ||
    parameterLayout.weightsLen !== requirements.weightsLen ||
    parameterLayout.biasLen !== requirements.biasLen
  ) {
    throw new Error(`Program layout evidence does not match requirements: ${JSON.stringify({ requirements, plan, bufferLayout, parameterLayout })}`);
  }
}

const model = nn.linear(2, 1, { weights: [0.5, -0.5], bias: [0] });
const eager = model.forward(tensor([1, 2], [2]));

const program = model.compile({ backend: "cpu" });
const capabilities = program.capabilities();
const requirements = program.requirements();

if (
  !capabilities.canExecute ||
  program.inputShape().join("x") !== "2" ||
  program.outputShape().join("x") !== "1" ||
  requirements.inputLen !== 2 ||
  requirements.outputLen !== 1
) {
  throw new Error(`unexpected Program evidence: ${JSON.stringify({ capabilities, requirements })}`);
}
if (!program.acceptsModule(model)) {
  throw new Error("compiled Program must accept the module it was compiled from");
}
assertProgramEvidence(program, model, requirements);
assertHotRuntimeProfile(program, "Program");

const session = program.bindModule(model);
const compiled = session.stepTensor(tensor([1, 2], [2]));
const contract = session.stepContract();
const hotParams = { input: tensor([1, 2], [2]), output: false };
const hotCompatibility = session.requireHotStepParams(hotParams);
const hotPlan = session.hotPathPlan(hotParams);
const rejectedCompatibility = session.stepParamsCompatibility({ input: tensor([1, 2, 3], [3]), output: false });

assertClose(eager.data[0], -0.5, 1e-6, "eager output");
assertClose(compiled.data[0], eager.data[0], 1e-6, "compiled output");
if (contract.inputShape.join("x") !== "2" || contract.outputShape.join("x") !== "1") {
  throw new Error(`unexpected Session step contract: ${JSON.stringify(contract)}`);
}
if (
  hotCompatibility.hotPath !== true ||
  hotCompatibility.runtimeOutputAllocationFree !== true ||
  hotCompatibility.readbackFree !== true ||
  hotCompatibility.outputEffect !== "none" ||
  !session.matchesStepContractSignature(contract.signature) ||
  !session.matchesStepParamsSignature(hotParams, hotCompatibility.stepParamsSignature) ||
  !session.matchesStepParamsCompatibility(hotParams, hotCompatibility) ||
  hotPlan.signature !== session.executionPlan(hotParams).signature ||
  hotPlan.stepParamsSignature !== hotCompatibility.stepParamsSignature ||
  hotPlan.hotPath !== true
) {
  throw new Error(`Session hot-path evidence failed: ${JSON.stringify({ hotCompatibility, hotPlan })}`);
}
if (
  rejectedCompatibility.accepted !== false ||
  rejectedCompatibility.status !== "rejected" ||
  rejectedCompatibility.hotPath !== false ||
  session.acceptsHotStepParams({ input: tensor([1, 2, 3], [3]), output: false }) !== false
) {
  throw new Error(`Session rejected StepParams evidence failed: ${JSON.stringify(rejectedCompatibility)}`);
}
assertHotRuntimeProfile(session, "Session");

session.free();
program.free();

console.log(`zgml Program/Session smoke ok: eager=${eager.data[0].toFixed(6)} compiled=${compiled.data[0].toFixed(6)}`);
