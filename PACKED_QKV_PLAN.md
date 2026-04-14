# Packed QKV Plan

The current transformer uses per-head `Q/K/V/O` projections. That is simple, but it creates too many small GEMMs.

## Goal

- keep attention code readable
- replace many tiny projection matmuls with a few large ones
- simplify loader and KV cache plumbing

## Target Layout

- `w_qkv: [3*d_model, d_model]`
- `b_qkv: [3*d_model]`
- `w_o: [d_model, d_model]`
- `b_o: [d_model]`

## Execution Shape

- compute one packed `qkv` projection
- split `q`, `k`, `v` by row views, not copies
- keep the per-head attention loop initially
- concatenate head outputs into `[d_model, seq]`
- run one final output projection

## KV Cache Layout

- one packed `k` tensor per layer: `[d_model, max_seq]`
- one packed `v` tensor per layer: `[d_model, max_seq]`
- derive head-local views by row slicing

## Phases

1. Add row-view slicing helpers
2. Pack `Q/K/V` weights and biases
3. Switch cached decode state to packed per-layer KV tensors
4. Update safetensors loader to pack full matrices directly

This keeps the model architecture close to standard decoder attention while staying much simpler than a flash-attention rewrite.
