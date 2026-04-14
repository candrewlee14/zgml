# Quantized Inference Plan

Quantization should stay narrow and inference-only.

## Goal

- do not make every tensor quantized
- keep graph math and training in `f32`
- quantize only large static linear weights

## Type Boundary

- keep `Tensor` unchanged
- add a small frozen weight abstraction for inference-time linear layers
- prefer a union like:
  - dense `f32`
  - packed blockwise `q8`

## Integration Strategy

- `nn.linear()` stays training/full-precision
- add an inference-only linear path for packed weights
- quantize selected transformer and LM-head weights only
- keep KV cache, activations, norms, and embeddings in `f32`

## Loading Strategy

- load dense weights from checkpoint or safetensors first
- quantize selected matrices once during inference setup
- add a dedicated serialized inference artifact only later if startup cost matters

## Phases

1. tighten `src/quant.zig` around packed 2D linear weights
2. add inference-only linear dispatch
3. quantize transformer projections and FFN weights
4. optionally add cached serialized packed-weight format

This keeps quantization out of the core graph architecture and avoids dtype complexity spreading through the runtime.
