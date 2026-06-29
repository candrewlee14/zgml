let gradEnabled = true;

export function isGradEnabled(): boolean {
  return gradEnabled;
}

export const is_grad_enabled = isGradEnabled;

export function setGradEnabled(enabled: boolean): boolean {
  const previous = gradEnabled;
  gradEnabled = Boolean(enabled);
  return previous;
}

export const set_grad_enabled = setGradEnabled;

function withGradEnabled<T>(enabled: boolean, fn: () => T): T {
  if (typeof fn !== "function") {
    throw new Error("grad mode helper requires a callback");
  }
  const previous = setGradEnabled(enabled);
  try {
    return fn();
  } finally {
    setGradEnabled(previous);
  }
}

export function noGrad<T>(fn: () => T): T {
  return withGradEnabled(false, fn);
}

export const no_grad = noGrad;

export function inferenceMode<T>(fn: () => T): T {
  return noGrad(fn);
}

export const inference_mode = inferenceMode;

export function enableGrad<T>(fn: () => T): T {
  return withGradEnabled(true, fn);
}

export const enable_grad = enableGrad;
