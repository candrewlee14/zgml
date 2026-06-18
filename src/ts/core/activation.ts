export function geluScalar(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

export function geluDerivativeScalar(x: number): number {
  const k = Math.sqrt(2 / Math.PI);
  const u = k * (x + 0.044715 * x * x * x);
  const t = Math.tanh(u);
  const du = k * (1 + 3 * 0.044715 * x * x);
  return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du;
}

export function siluScalar(x: number): number {
  return x / (1 + Math.exp(-x));
}

export function siluDerivativeScalar(x: number): number {
  const s = 1 / (1 + Math.exp(-x));
  return s * (1 + x * (1 - s));
}

export function sigmoidScalar(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function sigmoidDerivativeScalar(x: number): number {
  const s = sigmoidScalar(x);
  return s * (1 - s);
}
