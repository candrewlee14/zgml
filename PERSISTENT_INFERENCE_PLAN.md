# Persistent Inference Plan

This repo should keep `ComputeGraph` as the authoring and training abstraction, then add a thin persistent inference layer on top.

## Goal

- keep the current small IR and fusion architecture
- avoid per-token model rebuilds, weight copies, and KV cache memcpy churn
- separate build-time graph memory from long-lived inference state

## Target Architecture

- `ComputeGraph` remains the builder and optimizer
- `InferencePlan` becomes a frozen forward-only schedule
- `InferenceSession` owns mutable runtime state:
  - token input
  - position
  - KV cache
  - reusable workspace
- model weights live outside transient graph arenas

## Runtime Memory Split

- build memory: graph arena, fusion analysis, temporary planning
- immutable memory: weights, positional tables
- session state: KV cache, current position, bound inputs
- workspace: reusable intermediate buffers

## Simple Memory Reuse

- compute `last_use_step` for temporary nodes
- assign workspace slots with first-fit reuse
- never recycle params or KV cache
- keep alias/view nodes allocation-free

## Phases

1. Persistent GPT decode session using the current graph runtime
2. Freeze forward graphs into an index-based `InferencePlan`
3. Add workspace/liveness reuse to frozen plans
4. Add bound inputs for token/position to avoid rebuilding graph shape state

## Phase 1 — Persistent Session (done)

- `GPTDecodeSession` owns model + KV cache persistently
- per-step `ComputeGraph` created, used, and discarded
- proves the architecture supports persistent inference

## Phases 2–4 — Frozen Plan + Workspace Reuse + Bound Inputs (done)

- `InferencePlan`: frozen forward-only schedule built once from a
  `ComputeGraph` trace of `GPT.forwardCachedFrozen`
  - no per-step graph rebuild, topology sort, or fusion re-analysis
  - bound inputs patched each step: token embedding, position encoding,
    causal mask, KV-cache write positions
- `InferenceSession`: owns model, KV caches, and plan; single `step()` API
- `forwardCachedFrozen` on `TransformerBlock` and `GPT`: fixed-shape
  attention over full KV cache with explicit mask (all shapes
  position-independent)
- workspace liveness analysis: `last_use_step` per node, first-fit slot
  assignment recycles dead intermediate buffers
