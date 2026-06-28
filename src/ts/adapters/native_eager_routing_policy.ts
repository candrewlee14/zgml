export type NativeEagerRoutingRuntime = "node" | "bun";
export type NativeEagerRoutingActivation =
  | "relu"
  | "gelu"
  | "silu"
  | "sigmoid"
  | "tanh";

export type NativeEagerRoutingPolicy = Readonly<{
  kind: "zgml.native-eager-routing-policy";
  runtime: NativeEagerRoutingRuntime;
  source: "src/ts/adapters/native_eager_routing_policy.ts";
  nativeCore: "zig-c-abi";
  tensorMath: Readonly<{
    matmul: "native";
    softmax: "native";
    elementwiseMinLength: number;
    activationMinLength: number;
    reduceMinLength: number;
    disabledActivations: readonly NativeEagerRoutingActivation[];
  }>;
  signature: string;
}>;

function policySignature(runtime: NativeEagerRoutingRuntime, tensorMath: NativeEagerRoutingPolicy["tensorMath"]) {
  return [
    "native-eager-routing-policy",
    `runtime=${runtime}`,
    `core=zig-c-abi`,
    `matmul=${tensorMath.matmul}`,
    `softmax=${tensorMath.softmax}`,
    `elementwiseMin=${tensorMath.elementwiseMinLength}`,
    `activationMin=${tensorMath.activationMinLength}`,
    `reduceMin=${tensorMath.reduceMinLength}`,
    `disabledActivations=${tensorMath.disabledActivations.join(",") || "none"}`,
  ].join("|");
}

function routingPolicy(
  runtime: NativeEagerRoutingRuntime,
  tensorMathPolicy: NativeEagerRoutingPolicy["tensorMath"],
): NativeEagerRoutingPolicy {
  const tensorMath = Object.freeze({
    ...tensorMathPolicy,
    disabledActivations: Object.freeze([...tensorMathPolicy.disabledActivations]),
  });
  return Object.freeze({
    kind: "zgml.native-eager-routing-policy",
    runtime,
    source: "src/ts/adapters/native_eager_routing_policy.ts",
    nativeCore: "zig-c-abi",
    tensorMath,
    signature: policySignature(runtime, tensorMath),
  });
}

export const nodeNativeEagerRoutingPolicy = routingPolicy("node", {
  matmul: "native",
  softmax: "native",
  elementwiseMinLength: 512,
  activationMinLength: 512,
  reduceMinLength: 512,
  disabledActivations: [],
});

export const bunNativeEagerRoutingPolicy = routingPolicy("bun", {
  matmul: "native",
  softmax: "native",
  elementwiseMinLength: 65536,
  activationMinLength: 65536,
  reduceMinLength: 512,
  disabledActivations: [],
});

export function nativeEagerRoutingActivationEnabled(
  policy: NativeEagerRoutingPolicy,
  activation: string,
) {
  return !policy.tensorMath.disabledActivations.includes(activation as NativeEagerRoutingActivation);
}
