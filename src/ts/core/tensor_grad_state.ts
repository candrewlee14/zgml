"use strict";

type TensorGradStateTarget = {
  readonly length: number;
  requiresGrad: boolean;
  requires_grad?: boolean;
  grad: Float32Array | null;
  _prev?: unknown[];
  _backward?: (grad: Float32Array | null) => void;
};

export function createTensorGradStateHelpers<TTensor extends TensorGradStateTarget>() {
  function hasRequiresGradAliasAccessor(tensor: TTensor): boolean {
    let current: object | null = tensor;
    while (current) {
      const descriptor = Object.getOwnPropertyDescriptor(current, "requires_grad");
      if (descriptor) return typeof descriptor.get === "function" || typeof descriptor.set === "function";
      current = Object.getPrototypeOf(current);
    }
    return false;
  }

  function setRequiresGradAlias(tensor: TTensor, enabled: boolean): void {
    if (!hasRequiresGradAliasAccessor(tensor)) tensor.requires_grad = enabled;
  }

  function requiresGrad(tensor: TTensor): boolean {
    return tensor.requiresGrad;
  }

  function setRequiresGrad(tensor: TTensor, value: unknown): void {
    const enabled = Boolean(value);
    tensor.requiresGrad = enabled;
    setRequiresGradAlias(tensor, enabled);
    if (tensor.requiresGrad && !tensor.grad) tensor.grad = new Float32Array(tensor.length);
  }

  function requiresGrad_(tensor: TTensor, value = true): TTensor {
    setRequiresGrad(tensor, value);
    return tensor;
  }

  function detach_(tensor: TTensor): TTensor {
    tensor.requiresGrad = false;
    setRequiresGradAlias(tensor, false);
    tensor.grad = null;
    tensor._prev = [];
    tensor._backward = () => {};
    return tensor;
  }

  return Object.freeze({
    requiresGrad,
    setRequiresGrad,
    requiresGrad_,
    detach_,
  });
}
