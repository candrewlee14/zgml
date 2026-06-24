import {
  nn,
  tensor,
  type NnCompilableModule,
  type ModuleKernelPlan,
} from "zgml/bun";

type DescriptorExpectation = Readonly<{
  dispatchCount: number;
  descriptorSignatures: readonly string[];
  nativeKernels: readonly string[];
  outputShape: readonly number[];
  outputLen: number;
  nativeCommandCount: number;
  persistentRequirementCount: number;
}>;

function assertArrayClose(actual: Float32Array, expected: Float32Array, tolerance: number, label: string): void {
  if (actual.length !== expected.length) {
    throw new Error(`${label}: expected length ${expected.length}, got ${actual.length}`);
  }
  for (let i = 0; i < expected.length; i += 1) {
    const actualValue = actual[i] ?? Number.NaN;
    const expectedValue = expected[i] ?? Number.NaN;
    const delta = Math.abs(actualValue - expectedValue);
    if (delta > tolerance) {
      throw new Error(`${label}[${i}]: expected ${expectedValue} +/- ${tolerance}, got ${actualValue}`);
    }
  }
}

function descriptorSignatures(plan: ModuleKernelPlan): readonly string[] {
  return plan.ops.flatMap((op) => op.nativeDescriptorSignatures ?? []);
}

function assertDescriptorContract(
  label: string,
  model: NnCompilableModule,
  inputShape: readonly number[],
  inputData: readonly number[],
  expected: DescriptorExpectation,
): void {
  const input = tensor(inputData, inputShape);
  const eager = model.forward(input);
  const support = model.compileSupport({ backend: "cpu", inputShape });
  if (!support || support.supported !== true || !support.kernelPlan) {
    throw new Error(`${label}: expected native compile support, got ${JSON.stringify(support)}`);
  }

  const plan = support.kernelPlan;
  const signatures = descriptorSignatures(plan);
  const kernels = plan.ops.flatMap((op) => op.nativeKernels ?? []);
  if (
    plan.dispatchCount !== expected.dispatchCount ||
    signatures.join("|") !== expected.descriptorSignatures.join("|") ||
    kernels.join("|") !== expected.nativeKernels.join("|") ||
    support.outputShape.join("x") !== expected.outputShape.join("x")
  ) {
    throw new Error(`${label}: TS KernelPlan descriptor evidence drifted: ${JSON.stringify({
      dispatchCount: plan.dispatchCount,
      signatures,
      kernels,
      outputShape: support.outputShape,
    })}`);
  }

  const program = model.compile({ backend: "cpu", inputShape });
  const evidence = program.compileEvidence();
  const inspection = program.inspect();
  const requirements = program.requirements();
  const compiledPlan = program.kernelPlan();
  if (
    !inspection.executionSupported ||
    inspection.backend !== "cpu" ||
    inspection.commandCount !== expected.nativeCommandCount ||
    inspection.persistentRequirementCount !== expected.persistentRequirementCount ||
    inspection.stepInputRequirementCount !== 1 ||
    inspection.stepOutputRequirementCount !== 1 ||
    inspection.runtimePatchHoles !== 0 ||
    !evidence ||
    !compiledPlan ||
    evidence.kernelPlan.signature !== plan.signature ||
    compiledPlan.signature !== plan.signature ||
    requirements.inputLen !== inputData.length ||
    requirements.outputLen !== expected.outputLen
  ) {
    program.free();
    throw new Error(`${label}: native Program inspection/compile evidence drifted: ${JSON.stringify({
      inspection,
      requirements,
      supportSignature: plan.signature,
      evidenceSignature: evidence?.kernelPlan?.signature,
      compiledSignature: compiledPlan?.signature,
    })}`);
  }

  const session = program.bindModule(model);
  const compiled = session.stepTensor(input);
  assertArrayClose(compiled.data, eager.data, 1e-5, `${label} compiled output`);
  session.free();
  program.free();
}

assertDescriptorContract(
  "Linear+GELU descriptor",
  new nn.Sequential([
    nn.linear(3, 2, {
      weights: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
      bias: [0.1, -0.2],
    }),
    nn.gelu(),
  ]),
  [3] as const,
  [1, 2, 3],
  {
    dispatchCount: 1,
    descriptorSignatures: ["1:2:2:0:3:2:0:0"],
    nativeKernels: ["linear", "gelu"],
    outputShape: [2],
    outputLen: 2,
    nativeCommandCount: 1,
    persistentRequirementCount: 2,
  },
);

assertDescriptorContract(
  "Reshape+ReLU descriptor",
  new nn.Sequential([
    nn.reshape([2, 2]),
    nn.relu(),
  ]),
  [4] as const,
  [-1, 2, -3, 4],
  {
    dispatchCount: 2,
    descriptorSignatures: ["8:0:0:0:2:2:2:0", "2:1:0:0:0:0:0:0"],
    nativeKernels: ["reshape", "relu"],
    outputShape: [2, 2],
    outputLen: 4,
    nativeCommandCount: 1,
    persistentRequirementCount: 0,
  },
);

assertDescriptorContract(
  "Reduction descriptor",
  new nn.Sequential([
    nn.reshape([2, 2]),
    nn.mean(1),
  ]),
  [4] as const,
  [1, 2, 3, 4],
  {
    dispatchCount: 2,
    descriptorSignatures: ["8:0:0:0:2:2:2:0", "13:0:0:0:0:0:0:0"],
    nativeKernels: ["reshape", "mean"],
    outputShape: [2, 1],
    outputLen: 2,
    nativeCommandCount: 3,
    persistentRequirementCount: 0,
  },
);

assertDescriptorContract(
  "Embedding classifier descriptor",
  new nn.Sequential([
    nn.embedding(4, 3, {
      weight: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
      ],
    }),
    nn.linear(3, 2, {
      weights: [
        0.2, -0.1,
        -0.1, 0.2,
        0.1, 0.1,
      ],
      bias: [0, 0],
    }),
    nn.logSoftmax(-1),
  ]),
  [2] as const,
  [0, 1],
  {
    dispatchCount: 3,
    descriptorSignatures: ["6:0:1:0:4:3:0:0", "1:0:2:0:3:2:0:0", "7:0:0:0:0:0:0:0"],
    nativeKernels: ["embedding", "linear", "log-softmax"],
    outputShape: [2, 2],
    outputLen: 4,
    nativeCommandCount: 2,
    persistentRequirementCount: 3,
  },
);

console.log("zgml bun TS/native descriptor contract smoke ok: linear+gelu, reshape+relu, reduction, embedding classifier");
