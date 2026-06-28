# Frontend Autograd Coverage

This file maps the public TS autograd claim to executable package-smoke
evidence. It is not a complete mathematical proof of every overload; it is the
checked coverage spine for the operations users naturally try first.

The authoritative runtime assertions live in
`src/ts/smokes/package_smoke_core.ts`. Keep this table synchronized with those
assertion labels, and add rows when a new public family becomes a user-facing
autograd claim.

| Public Family | Coverage Status | Runtime Smoke Evidence |
| --- | --- | --- |
| Broadcasted binary math | checked | `Tensor.maximum broadcast`, `where broadcast helper`, `Tensor.maskedFill value autograd backward` |
| Movement and view-style ops | checked | `Tensor.flip autograd backward`, `Tensor.roll autograd backward`, `Tensor.split autograd backward` |
| Scalar unary math | checked | `Tensor.reciprocal autograd backward`, `Tensor.rsqrt autograd backward`, `Tensor.sin autograd backward`, `Tensor.sin numerical gradient`, `Tensor.sigmoid numerical gradient`, `Tensor.silu numerical gradient`, `Tensor.gelu numerical gradient`, `Tensor.tanh autograd backward`, `Tensor.tanh numerical gradient` |
| Zero-gradient integer-like ops | checked | `Tensor.floor zero gradient`, `Tensor.round zero gradient` |
| Reductions | checked | `Tensor.variance autograd backward`, `Tensor.std autograd backward`, `Tensor.norm autograd backward`, `Tensor.prod autograd backward`, `Tensor.cumsum autograd backward` |
| Log-sum-exp family | checked | `Tensor.logsumexp autograd backward`, `Tensor.logsumexp numerical gradient` |
| Linear algebra | checked | `Tensor.dot lhs autograd backward`, `Tensor.dot rhs autograd backward`, `Tensor.trace autograd backward`, `Tensor.diagonal autograd backward`, `Tensor.bmm lhs autograd backward`, `Tensor.bmm rhs autograd backward` |
| Losses | checked | `tensor crossEntropy gradient`, `tensor nllLoss gradient`, `tensor nllLoss sum gradient` |
| Module parameters | checked | `module zero_grad alias clears weight grad`, `nn.zero_grad alias clears bias grad`, `optimizer must skip frozen parameters with stale gradients` |
| Conv/pool modules | checked | `nn.Conv2d input grad`, `nn.Conv2d weight grad`, `nn.Conv2d bias grad`, `nn.MaxPool2d input grad` |

## Remaining Work

- Continue adding numerical-gradient checks for nonlinear TS frontend paths that
  still have exact-value smoke checks only; `sin`, `sigmoid`, `silu`, `gelu`,
  `tanh`, and `logsumexp` now have finite-difference package-smoke coverage.
- Split broad rows such as broadcasted binary math into operation-level rows if
  a new public API depends on a subtle gradient convention.
- Add compile-support columns only when backward through compiled Programs
  becomes a product claim; today compiled Programs are the inference/hot-path
  performance lane, while eager autograd owns training.
