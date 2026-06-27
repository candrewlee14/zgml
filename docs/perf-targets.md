# Apple-First GGML Parity Targets

This is the checked-in performance contract for the GGML parity push. zgml does
not claim parity until it is within 10% of llama.cpp on the same Apple Silicon
machine for both prompt/prefill and decode.

## Current SmolLM-135M Status

- F16 and Q8_0 prompt/prefill are backend-only, but still roughly 27-34% of
  llama.cpp Metal on SmolLM-135M. F16 prompt now uses dense projection-chain,
  dense projection-pair fused-elementwise, row-chain, and
  dense projection-cache commands for prefill-shaped projections, reducing its
  structural roof from the original one-dispatch-per-op shape to 242
  executable region commands/dispatches per prompt call. Dense projection
  row-chain fusion is intentionally disabled by default because fresh
  benchmark evidence showed it lowers dispatch count while losing an order of
  magnitude of throughput. Projection row-chain fusion is also disabled by
  default for Q8_0 for the same reason: the full SmolLM path is much faster as
  projection-chain plus row-chain commands, even though the isolated row-chain
  microbench still passes. Q8_0 tied embeddings keep the f32 lookup tensor but repack a
  direct quantized logits weight, so the final tied LM head is no longer a dense
  ProgramCommand. Current accepted gate evidence is 5171.73 F16 prompt tok/s
  and 4585.71 Q8_0 prompt tok/s from
  `bench-results/smollm-20260618T025954Z-p128-g200-r3.json`, with zero fallback,
  passing native execution evidence, and a checked Program/Session substrate
  shape: F16 prompt/decode at 242/212 dispatches and 242/212 executable region ProgramCommands,
  Q8_0 prompt/decode at 242/212 dispatches and 241/211 ProgramCommands because
  the direct quantized tied LM head is one standalone backend dispatch outside
  the ProgramCommand stream.
- Decode is not a parity claim. The benchmark gate selects only the native
  region-dispatch lane because it is backend-only, even though it is still
  slower because it pays 212 F16 or 212 Q8_0 dispatches and one backend sync per
  token. Current accepted local gate evidence is 121.00 F16 decode tok/s
  and 141.50 Q8_0 decode tok/s from
  `bench-results/smollm-20260618T025954Z-p128-g200-r3.json`, with zero fallback,
  passing native execution evidence, and a checked runtime patch shape of 450
  total holes: 180 cache-write-position holes and 270 attention-seq/KV holes.
  This is still only 34.53% F16 decode and 32.88% Q8_0 decode parity; the
  remaining work is coarser semantic
  commands produced by a stronger scheduler/kernelizer, with LLaMA-specific
  recognizers for projection/MLP chains, row chains, and attention/rope-store
  groups only where generic kernelization cannot express the fast path cleanly.
- Latest evidence lives in the JSON artifact from `scripts/bench_vs_ggml.sh`;
  `summary.gate_zgml.{f16,q8_0}` and `gates.*` are the machine-readable
  contract.
- If llama.cpp Metal preflight cannot find or execute a usable Metal/MTL
  reference before zgml runs, `bench_vs_ggml.sh` still writes a JSON artifact with
  `gates.preflight.passed=false`, `gates.preflight.failure_kind`,
  `gates.required_pass=false`, and the raw preflight output. That artifact is
  diagnostic blocker evidence, not parity evidence.
- Preflight blocker kinds distinguish no usable Metal device, Metal context
  initialization failure, missing Metal result rows, and generic llama-bench
  failure. Metal-context failures include an explicit `-dev BLAS` smoke in the
  raw output when available, proving whether the model and llama.cpp CPU/BLAS
  path still work.
- The latest accepted artifact is useful as a structural/baseline gate result,
  not a parity claim. Its current parity rows are 33.88% F16 prompt, 34.53% F16
  decode, 27.38% Q8_0 prompt, and 32.88% Q8_0 decode.
- `bench:status`, `bench:substrate`, and `bench:trend` select the latest
  accepted local full-run artifact, meaning `gates.required_pass=true` and the
  selected lanes still pass the checked M5 Pro baseline comparison. Failed or
  baseline-regressing p128/g200/r3 artifacts can remain in `bench-results/` for
  diagnosis without becoming the substrate proof; status output reports the
  quarantined count and latest failed artifact explicitly.
- Quarantined full-run diagnostics are useful for explaining rejected changes,
  but not for raising the substrate score. Recent failed artifacts include both
  reference-drift runs and the unpromoted Q8_0 projection-row-chain default
  experiment: they record zero fallback and useful shape evidence, but they
  either miss the checked M5 Pro baseline floor or prove that a lower command
  count did not preserve full-model throughput. Treat them as diagnosis, not as
  accepted baseline evidence.
- In artifacts, `gates.overall_pass` is the actual parity+baseline result.
  Without `BENCH_REQUIRE_PARITY=1` or `BENCH_BASELINE_JSON`, `gates.required_pass`
  is structural native-decode evidence rather than a throughput claim.

## Architecture Rules

- The stable public Interface stays small. Backend, GGUF, quantization, and
  benchmark internals live in the build-only `zgml_internal` Module; runtime
  profiling stays in benchmark/internal evidence, not the public root.
- Do not ship a generation CLI or tokenizer Adapter unless it matches the model
  family being served. A smaller Interface is better than a misleading one.
- `DeviceInference` lowers the caller-side DeviceProgram Interface shape, then
  keeps the compiled backend `Program`, persistent-bound `Session` buffers,
  per-step I/O, and per-call `StepParams` separate. `Program.compile` records
  the bindable tensor/buffer shape, but persistent tensors are not compile-time
  initial uploads. The C/JS session path now owns host-side persistent and
  per-step buffers and can sequentially rebind compatible weights through one
  compiled program. Backend Adapters bind a `RuntimeHandle` per session and can
  upload persistent bindings without dispatching a step. CPU/stencil runtime
  handles own independent mutable buffers/stencils; Metal runtime handles own
  per-session buffers/stencils and bound execution constructs an explicit
  `RuntimeView` for scheduling, command lowering, fallback, kernel buffer
  binding, and host/device transfer. Metal command buffers and transient
  command-profile counters are owned by a per-execution encoder context, then
  merged through the backend-owned profile path. Callers execute with semantic
  token windows; they do not scan or patch ops.
  `Program.inspect()` is the
  non-hot-path evidence surface for runtime patch and command-stream shape.
- Treat LLaMA prefill/decode as semantic copy-and-patch on top of the general
  compiled-Program path: a fixed scheduled program shape plus host-side patching
  for token embedding, masks, and RoPE ranges; compiled backends derive runtime
  bindings for cache write positions and valid attention length directly from
  the DeviceProgram/ProgramStencil. Generic op fusion is not enough by itself,
  but the first performance seam should be the scheduler/kernelizer that turns
  lazy tensor graphs into fewer kernels. Semantic LLaMA recognizers should be
  used when they buy dispatch depth, memory locality, or quantized layout wins
  that the generic kernelizer cannot express cleanly.
- `LlamaInferenceSession` owns semantic device patching for tokens, causal
  masks, RoPE, and host KV offsets. The concrete `LlamaInferencePlan` factory
  and host-graph patch implementation are private; the internal LLaMA device
  Module asks the session for `RuntimeWindow` instead of knowing plan patch
  intents. The plan trace owns the semantic stage and runtime patch-hole shape;
  device execution Adapters consume that shape instead of re-deriving it from
  configuration formulas.
- The public `zgml.llm` surface now names the planned lifecycle as
  `LlamaModel -> LlamaProgram -> LlamaSession -> StepParams` while preserving
  the direct `LlamaSession.init/load/step` convenience path. Bound sessions now
  own independent runtime state: model params are copied, CPU and Metal direct
  quantized GGUF qweights are uploaded into runtime bindings after compile-time
  shape validation, and KV caches/position/plans are separate per session.
  Executable prefill now synchronizes KV-cache side effects into the Session so
  the following decode call observes prompt state through a refreshed
  persistent KV-cache binding range instead of a prefill-only backend cache copy
  or broad weight re-upload.
  `stepInto` and `prefillInto` let callers reuse their own logits buffers for
  embedder/FFI-style output ownership.
  This is still a copy-on-bind slice, not the final zero-copy compatible
  checkpoint/KV binding model.
- Backend capability metadata declares support facts only. Command-stream
  defaults live behind the backend/program planning seam, so scheduler tuning is
  not part of the public backend Interface.
- Prefer scheduler/kernelizer-produced coarse kernels over one-dispatch-per-op
  execution and over endless local pair fusers. Prefer semantic stage/layer
  commands only when they preserve the same ProgramStencil evidence and prove a
  real throughput or dispatch-depth win.
- Row-chain Adapters may drop dead norm/repeat materialization when planner
  liveness proves only the scaled output survives. Do not fuse RMSNorm into
  projection matmuls by recomputing it per output tile; dispatch-count wins must
  not buy themselves with substantially higher math.
- Row-chain kernels run one threadgroup per row with a cooperative reduction and
  parallel scale writes; keep non-multiple-of-64 column coverage in exact Metal
  tests when changing them.
- Decode projection matvec batches use a four-column Metal variant for slice,
  elementwise, and RoPE-store sidecars, sharing input loads across adjacent
  output columns. The tiled RoPE-store path proved faster on the benchmark gate
  and retired the old one-column batched matvec kernels without changing the
  ProgramCommand Interface. It is a throughput win, not a dispatch-depth win:
  prompt remains gated at 242 F16 / 242 Q8_0 dispatches/call and decode at
  212 F16 / 212 Q8_0 dispatches/token.
- Decode-shaped dense projection chains (`M == 1` dense matmul plus add/mul
  sidecar) lower through the same four-column dense matvec Adapter instead of
  the tiled dense matmul-elementwise kernel. This preserves the
  `dense_projection_chain` ProgramCommand evidence and dispatch roof while
  raising F16 decode from the old ~20% ggml lane into the checked >=31% floor.
- Quantized prompt `projection_row_chain` is an opt-in semantic ProgramCommand,
  not a default throughput claim. The accepted default keeps the faster
  projection-chain plus row-chain split. The candidate path can lower
  prompt-sized Q8 row chains through either the existing fast tiled
  `qmatmul_elementwise_f32` dispatch plus the existing RMSNorm-scale row-chain
  dispatch under one executable command, or the experimental single-dispatch
  tiled row-chain hook. This keeps the command-frontier simplification visible
  without promoting a slower full-model path. The scalar `qmatmul_row_chain_f32`
  kernel remains available as a diagnostic candidate, but the next throughput
  target is a single-dispatch tiled row-chain kernel, not another scalar
  per-row/per-column variant. Default full-model prompt evidence must keep zero
  fallback and avoid throughput regression.
- The controlled full-model prompt hook remains:
  `--metal-prompt-projection-row-chain-candidate`. It now enables the
  opt-in single-dispatch tiled row-chain experiment. `npm run
  bench:q8-prompt-candidate` runs paired default/candidate attempts
  (`BENCH_CANDIDATE_ATTEMPTS`, default 3), requires structural readiness and
  zero fallback on every attempt, and reports command/candidate throughput
  readiness separately. Median throughput must still clear the speedup floor
  before the output is labeled throughput-ready, but a noisy command median no
  longer hides otherwise valid structural evidence. It still prints
  best/median/worst speedup plus the number of noisy attempts below floor so a
  single lucky run cannot promote the candidate. On Q8_0 SmolLM p128/g40/r1, the current
  one-dispatch candidate lowers commands from the default 241-command split to
  181 commands with `projection_row_chain` at 60 while reducing total dispatches
  from 242 to 182 and keeping fallback at zero.
  Throughput is not ready: the latest focused run reported
  `dispatch_reduction_without_tiled_throughput` because prompt speed fell to
  0.19x of the default split tiled qmatmul plus RMSNorm path. This proves the
  target shape and the trap at the same time: the next accepted kernel must keep
  one dispatch per semantic row-chain while restoring tiled qmatmul throughput.
  A focused Metal tail-parallelization experiment confirmed the deeper trap:
  the one-dispatch candidate owns only row-tile threadgroups and therefore loops
  over N tiles inside each threadgroup instead of using the MxN tile parallelism
  that makes `qmatmul_elementwise_f32` fast. The q8 prompt gate now names this as
  `single_dispatch_trap=serial_n_tile_loop_without_cross_threadgroup_row_reduce`.
  The viable next target is therefore either a larger semantic sublayer command or
  a two-phase tile-parallel row-chain, not another local serial-tail variant.
  For that viable target, `bench:q8-prompt-viable` and
  `dev:perf:q8-prompt:viable` set `BENCH_Q8_PROMPT_LANES=command,two_phase,semantic`
  and default to one attempt, so command, two-phase, and semantic-throughput
  changes can be checked without rerunning the known-bad single-dispatch diagnostic on every edit. The
  full-model gate accepts either the older pair-fused row-chain shape or the
  larger semantic-command shape that consumes the FFN projection pair and leaves
  30 semantic row-chain commands. A June 24, 2026 fresh-native viable run kept
  the command lane structurally and throughput-ready
  (`command_command=241->151`, `command_speedup=1.00-1.03x`) while the selected
  two-phase and semantic lanes proved wiring and command-shape readiness
  (`semantic_structural_selected=yes`, `semantic_projection_pair=30->0`,
  `semantic_projection_row_chain=0->30`) with zero fallback. Repeated steady
  evidence initially kept semantic throughput diagnostic when median throughput
  sat below parity. A later paired-default three-attempt run cleared the
  promotion bar (`status=promoted-default`, `semantic=promoted`,
  `default_policy=semantic-promoted`, `command_command=151->151`, zero
  fallback), so Metal scheduled Q8 prompt now defaults to the semantic FFN
  command plus two-phase tiled row-chain tail.
  The two-phase partial kernel does not bind the scale buffer anymore; scale is
  only needed by the finalize pass. This keeps the candidate ABI shape smaller
  without changing command semantics.
  Its finalize pass now has a tiled `(row_tile, col_tile)` variant, so the
  candidate no longer scales an entire prompt row tile through only one
  threadgroup. Fresh promoted-default evidence keeps the semantic command shape
  as the baseline while still reporting `semantic_spills=30`,
  `semantic_spill_input=17280`, and `semantic_output_spills=0`. Q8 prompt
  candidate artifacts and `bench:status` also expose `semantic_bridges`, derived
  from `program_command_shape_projection_row_chain_semantic_residual_bridges`,
  so the next pass can distinguish a bridgeable residual-use opportunity from
  generic row-chain materialization pressure.
- The first semantic residual-bridge command is now a candidate shape rather
  than the default: `semantic_ffn_sublayer_with_input_row_chain` recognizes the
  14-op `projection_row_chain + semantic_ffn_sublayer` pattern. A focused Q8
  prompt probe proved the structural target with zero fallback:
  `semantic_command=151->121`, `semantic_projection_row_chain=30->0`, and
  `semantic_bridges=30->0`. It is not a throughput promotion yet; the same
  one-attempt probe reported `semantic_speedup=0.96x`, so the default stays on
  the known 151-command semantic-promoted path until the bridge command gets a
  real throughput kernel instead of just reusing the existing two-piece encoder.
  The exact input-bridge microscope now serializes that limitation explicitly.
  The legacy absorbed command executed as five backend dispatches
  (`row_chain=2`, `pair=1`, `tail=2`); the source-current input-bridge kernel
  replaces that with one `semantic_ffn_sublayer_with_input_row_chain` dispatch
  for the `m=128,h=1536,k=576,o=576` microscope. The latest focused artifact
  reports `absorbed=1.17-1.57x`, `runtime_dispatches=1`,
  `semantic_with_input_dispatches=1`, `decomposed_dispatches=0`,
  `pair_dispatches=0`, `tail_dispatches=0`, and `spilled_input=0`. Treat this
  as an isolated width-parallel bridge proof, not a full-model parity claim.
  It also records the current partitioning limit explicitly:
  `direct_partition:row_serial`, `direct_rows=128`, `direct_row_threadgroups=128`, and
  `direct_per_row_threadgroup=2985984` row-serial dot ops in the focused
  bridge artifact.
  The full-model semantic throughput candidate now keeps the 14-op input-bridge
  command shape but disables the direct input-bridge kernel until it is
  partitioned. A fresh one-attempt probe recovered the semantic candidate from
  the old direct-bridge `0.59-0.60x` diagnostic to `semantic_speedup=0.94x`
  while preserving `semantic_command=151->121` and zero fallback. The q8
  artifact now exposes `semantic_direct=0`, `semantic_absorbed_dispatch=150`,
  `semantic_absorbed_split=5.00`, and `semantic_fallback_dispatch=90`; the
  remaining target is therefore the semantic width/dim kernel
  (`semantic_single_dispatch_refused_dim=30`), while the direct bridge stays as
  a focused partitioning microscope.
- The current weakest checked lane is still Q8_0 prompt, but its command
  pressure has moved from the old `projection_chain:60` baseline to the
  promoted semantic-default shape: 151 ProgramCommands, 30 semantic row-chain
  commands, and 30 remaining model-width spills. The next meaningful Q8 prompt
  move is no longer "promote the semantic candidate"; it is either a larger
  semantic command that absorbs the live model-width residual feeding the next
  FFN, or a true semantic FFN/down/residual/norm throughput kernel that removes
  the two-dispatch tail cost without reintroducing the row-serial trap. Use
  `dev:perf:frontier:qproj` only for projection-chain kernel/layout regressions;
  the promoted default's main pressure is now semantic row-chain spill/work
  shape. It rebuilds the benchmark artifacts with
  `-fincremental`, filters to qproj projection-chain/group lanes, and keeps the
  checked correctness and speed floors from `check_frontier_bench.cjs`.
- The frontier gate now measures prompt-shaped quantized projection-chain
  kernels independently of the full model:
  `qproj prompt m=32 n=512 k=512 projection_chain` tile must stay within 5% of
  staged qmatmul-plus-elementwise throughput, while the noisier
  `qproj full-prefill m=128 n=512 k=512 projection_chain` diagnostic must stay
  within 10%. The full-prefill line also reports a parity candidate signal at
  1.00x. Both must keep max absolute
  difference below 0.002.
  Frontier samples now run against an 8ms minimum timing window so the
  near-parity prompt tile gate is less likely to pass or fail on timer noise
  rather than kernel behavior.
  Observed runs are often faster, but the gate treats this as local
  non-regression evidence because Metal command timing is noisy at these sizes.
  This proves the current weak-lane command is locally justified across tile
  and full-prompt shapes while keeping the larger Q8 prompt percentage honest.
- The frontier gate also reports a four-way prompt projection-group diagnostic:
  `qproj group full-prefill x4 m=128 n=512 k=512 projection_group` compares
  four staged projection chains against the existing qmatmul batch-with-sidecars
  kernel. Correctness is enforced, but speed only marks a candidate as ready at
  1.05x because current runs show this shape is exact but not reliably faster.
  The focused qproj gate now also reports command-shape and runtime-command
  evidence for this lane. The compact x4 diagnostic still reports
  `shape_commands=1`, `shape_projection_groups=1`, `shape_covered_ops=8`, and
  `shape_saved_dispatches=7`, but its runtime evidence is `runtime=off` with
  zero `runtime_projection_group_dispatches` and zero
  `runtime_projection_cache_group_dispatches` because it sits below the Metal
  executable-region threshold. The x7 qproj region microscope is the executable
  proof: it reports `shape_commands=2`, `shape_projection_groups=2`,
  `shape_covered_ops=14`, `shape_saved_dispatches=12`, `runtime=command`, and
  `runtime_projection_group_dispatches=2`, with both region lanes required to
  clear the focused `1.00x` speed floor. This keeps projection grouping visible
  as a real planner target without pretending the smaller x4 diagnostic already
  dispatches a named executable command.
- Row-chain frontier runtime profiles now report
  `qmatmul_row_chain_tiled_spilled_elementwise`, separating spill-free synthetic
  row-chain kernel measurements from full-model Q8 prompt measurements where
  residual liveness still forces `spills=30`. Treat spill-free frontier rows as
  raw kernel-shape diagnostics; a default Q8 prompt win must either absorb the
  live residual use into a larger semantic command or beat the staged path while
  materializing it. The focused qrow-region gate now reports
  `qmatmul_row_chain_tiled_spilled_elementwise=0` for the x7 full-prefill and
  SmolLM-prompt two-phase rows, but they still land at `0.80x` and `0.61x`;
  spill removal alone is not the missing default-performance move.
- Metal output-read checks are now span-aware for semantic and row-chain
  intermediate writes, so a Program output on the same arena buffer no longer
  forces materialization unless its byte range overlaps the candidate write. A
  source-current three-attempt Q8 prompt semantic run still reported
  `semantic_median=1.00x`, `semantic_worst=0.92x`, and `spills=30`, so this
  narrows correctness/materialization policy but does not change the next perf
  target: larger semantic FFN throughput or live-residual absorption.
- The Q8 prompt gate now also reports `semantic_output_spills`. A fresh
  source-current semantic probe reported `semantic_spills=30` and
  `semantic_output_spills=0`, which rules out logits/KV-cache output bindings
  as the spill cause. The remaining spills are command/liveness pressure, so the
  next optimization should absorb the live residual user into a larger semantic
  command or make the materialized tail fast enough to win. It also reports
  `semantic_bridges` from
  `program_command_shape_projection_row_chain_semantic_residual_bridges`, making
  the residual-bridge hypothesis an artifact field instead of a guess.
- The selected three-attempt Q8 prompt artifact now exposes the same diagnosis:
  `semantic_spills=30`, `semantic_output_spills=0`,
  `semantic_median=0.98x`, `semantic_worst=0.94x`. Its two-phase lane is a
  current candidate (`two_phase_median=1.06x`, `two_phase_worst=0.99x`,
  `two_phase_output_spills=0`), but semantic FFN remains diagnostic.
- Spill input-width evidence identifies the remaining materialized chains. A
  source-current semantic quick probe reported `semantic_spill_input=17280`
  across `30` spills, i.e. average `K=576`, so the spills are model-width
  attention/output-projection residual chains feeding the next FFN, not the
  `K=1536` FFN-down tails.
- The full-model Q8 prompt probe now distinguishes general projection groups
  from attention/cache projection groups. Its Q8 evidence reports
  `projection_group=0->0`, `projection_cache_group=30->30`, and
  `decode_projection_cache_group=30`, so the attention/cache group path is
  already active in the full model. The remaining Q8 prompt work is the
  `projection_chain=60` row-chain/semantic-sublayer lane; the frontier x7 qproj
  region is the isolated executable proof for general qmatmul-plus-sidecar
  projection groups. The Q8 prompt gate also prints
  `qproj_group_full_model_target=frontier_only_not_full_model_sibling_region`
  and `qproj_frontiers=60`, making the current full-model target explicit:
  row-chain frontier pressure, not a missed sibling qproj grouping pass.
- Qsemantic target-thread experiments should not retune the row-serial semantic
  kernel width blindly. A focused `SEMANTIC_FFN_THREADS=512` rerun preserved
  correctness but stayed diagnostic (`target_vs_default` roughly
  `full_prefill:0.29x,smollm_prompt:0.26x`), so the useful target remains
  tiled/vectorized semantic dot work or a faster tiled row-chain leaf. The
  throughput-only semantic artifact now also reports fixed-thread width-lane
  utilization separately from the reduction lane: full-prefill can hit
  `width_lane_utilization_x1000=1000`, while the 576-wide SmolLM prompt shape is
  still about `562/1000` (`thread_lane_utilization_x1000=611`) with
  `width_slot_gap=2.00x` and `thread_slot_gap=1.80x`. Treat that as the next
  kernel-design target: reduce the 576-wide width-slot waste, not just the
  nominal row-serial math. Fresh throughput evidence with that utilization gap
  is reported as `semantic_width_parallel_kernel` in
  `perf-next:`, and `dev:perf:next` routes it to the exact qsemantic bridge
  microscope so width-parallel kernel edits iterate on the `576 x 1536 x 576`
  shape directly. A
  576-only 256-thread narrow-kernel probe reduced the slot footprint but hurt
  throughput (`smollm_prompt=0.67x`). A 384-thread mid-width probe improved
  utilization to about `777/1000` but still lost throughput
  (`smollm_prompt=0.69x`). A 768-thread exact-width probe improved the new
  SmolLM width-lane evidence from `562/1000` to `750/1000` and total lane
  utilization to about `800/1000`, but still stayed below default
  (`smollm_prompt=0.99x`) while regressing full-prefill (`full_prefill=1.12x`),
  so a useful fix needs better partitioning or vectorization without dropping
  too much useful parallelism. The retained unrolled
  512-thread semantic kernel is the first such win: it preserves the one-dispatch
  shape and lifts the three-attempt qsemantic throughput gate to
  `full_prefill=1.99x`, `smollm_prompt=1.04x`, `gate=ready`. The refreshed
  three-attempt Q8 prompt viable run also promotes the semantic lane with
  `semantic_speedup=1.18x`, `semantic_median=1.01x`, and
  `semantic_worst=0.99x`. A later source-current semantic-only block-32 scale
  specialization replaced hot semantic dequant divisions with `w_idx >> 5`
  under an encoder guard requiring all three semantic qweight block sizes to be
  `32`. That preserves correctness and lifts the focused qsemantic throughput
  gate to `full_prefill=2.67x:median:2.66x:worst:2.65x` and
  `smollm_prompt=2.60x:median:2.31x:worst:1.63x`. It also moves the full Q8
  bridge lane from clear regression to near-promotion
  (`semantic_best=1.16x`, `semantic_median=1.00x`, `semantic_worst=0.99x`).
  A later eight-wide semantic down-loop unroll kept the focused input-bridge
  microscope correct and selected `absorbed=2.73x` with
  `direct_serial=1.38x`; however, the direct bridge still reports
  `2,985,984` row-serial dot ops per row threadgroup, so this is diagnostic
  cleanup rather than a replacement for the width-partitioned input bridge. The
  direct bridge is therefore opt-in only on the named diagnostic policy; the
  normal input-bridge candidate keeps the faster absorbed/width-partitioned
  lowering while a future kernel removes its remaining dispatch split. The
  frontier artifacts now expose `qmatmul_row_chain_width_parallel_count` and
  lane totals, so the bridge and input-bridge gates can prove when the
  width-partitioned Metal tail actually ran rather than relying on the broader
  two-phase tiled counters. Input-bridge artifacts also expose
  `semantic_ffn_with_input_decomposed_extra_dispatches`; the retained absorbed
  lowering reports `4`, making the remaining dispatch-collapse target visible
  in both focused artifacts and `perf-next`. The
  checked input-bridge gate now treats that lane as a steady collapse guard:
  runs with at least three attempts must keep best absorbed speedup at or above
  `2.45x` by default, configurable with
  `BENCH_QSEMANTIC_INPUT_BRIDGE_STEADY_ABSORBED_FLOOR`.
  The missing win is still work partitioning, but the semantic block-32 scale
  specialization is retained as a real kernel improvement.
- A later row-chain tiled scale-index shift probe was rejected. Guarding the
  tiled row-chain encoders to block size `32` and changing their scale lookup to
  `w_idx >> 5` preserved compilation but worsened the paired Q8 semantic bridge
  sample (`semantic_median=0.97x`, `semantic_worst=0.94x`), so do not chase that
  spelling again. The Q8 candidate artifact now records semantic single-dispatch
  diagnostics instead: a quick semantic-lane probe showed
  `semantic_single_dispatch_attempts=30`,
  `semantic_single_dispatch_output_read_refusals=0`, and
  `semantic_single_dispatch_block_size_refusals=0`. That means the full bridge
  path is not blocked by requested intermediate outputs or Q8 block-size guards;
  the next useful target is the remaining compatibility/shape refusal before
  the one-dispatch semantic FFN kernel can replace the two-phase row-chain tail.
  A compact refusal histogram then identified that remaining refusal as `dim`
  for all `30` attempts. The Q8 prompt artifact/status path now records the
  refused shape directly as
  `semantic_single_dispatch_dim_refusal_shape=k:576,h:1536,o:576,cap:1024`.
  Temporarily lifting `SEMANTIC_FFN_MAX_DIM` from `1024` to `2048` selected the
  single-dispatch path and reduced dispatches `242->182`, but the full-model
  semantic bridge fell to `0.62x`. Do not promote that cap bump. The next
  performance target is a tile-parallel semantic FFN/residual/norm kernel for
  the `576 x 1536 x 576` FFN shape, without the row-serial throughput loss.
  A narrower hidden-only probe reached the same conclusion: changing the
  non-input semantic FFN scratch from `SEMANTIC_FFN_MAX_DIM` to
  `SEMANTIC_FFN_MAX_HIDDEN` made the exact `m=128,h=1536,k=576,o=576` bridge
  structurally dispatch as one semantic kernel with `max_abs_diff=0.000001`, but
  the fused kernel measured about `2.23ms` versus the existing tiled
  pair-plus-tail control around `0.88ms`. Keep the non-input semantic
  single-dispatch product scratch capped at `SEMANTIC_FFN_MAX_DIM`; only the
  input-bridge direct kernel may use `SEMANTIC_FFN_MAX_HIDDEN` until the
  replacement is genuinely tile/width parallel.
  A June 25, 2026 follow-up also tried reducing `SEMANTIC_FFN_THREADS` from
  `512` to `256` while allowing the non-input product scratch to span
  `SEMANTIC_FFN_MAX_HIDDEN`. It compiled and preserved bridge correctness, but
  the exact bridge microscope selected the one-dispatch row-serial kernel at
  only about `1.00x`, below the retained mixed pair-plus-tiled-tail evidence
  (`1.87x` steady, `2.23x` latest one-attempt). Do not retry threadgroup-width
  tuning as the bridge fix; the missing work is a real tile/width-parallel
  semantic FFN kernel, not a narrower row-serial group.
  That exact shape now has a checked frontier microscope:
  `npm run dev:perf:frontier:qsemantic:bridge{,:run}`, which writes
  `frontier-qsemantic-bridge-*.json` for `bench:status`; the raw terminal
  microscope remains
  `npm run dev:perf:frontier:qsemantic:bridge:raw{,:run}`. Its first
  ReleaseFast runs measured
  `qsemantic bridge-ffn m=128 h=1536 k=576 o=576 semantic throughput_candidate`
  in the `1.82x-2.51x` range over the staged path with `max_abs_diff=0.000000`,
  `shape_semantic_ffn_sublayers=1`, and `runtime_backend_dispatches=3`. Treat
  this as an isolated kernel-work baseline, not a full-model promotion signal.
  The bridge artifact/status line now also reports `width_target=` with the
  exact hidden-tile target: `rows:128,hidden:1536,input:576,output:576`,
  `row_groups:4`, `hidden_tiles:48`, `output_tiles:18`, plus product/output
  element counts and gate-up/down/total dot-work. It now also reports the
  staged-scratch pressure for the obvious width-parallel down design:
  `down_partial_elements=3538944`, `down_partial_bytes=14155776`, and
  `down_partial_to_output=48.00`. That is larger than the existing
  output-sized residual/norm buffers. The Metal runtime now allocates and
  profiles reusable backend-owned scratch for this contract, but keeps the
  allocation policy-aware and precise for the current mixed path: default
  semantic-command Programs reserve no semantic-width scratch, while the
  throughput-candidate tiled row-chain tail reserves only the `9216` byte partial
  surface it consumes and still records the `14155776` byte future down-partial
  target. The next semantic-width implementation needs to either consume that
  larger target, prove an equivalent accumulation strategy, or use a streamed
  hidden-tile design that avoids materializing all partials.
  A follow-up finalize-path probe tried replacing
  `qmatmul_row_chain_tiled_finalize_tiles_f32` with the coarser row-tile
  `qmatmul_row_chain_tiled_finalize_f32` to avoid repeated RMS reductions
  across output tiles. It was rejected: the exact bridge lane fell from the
  retained `~2.31x` evidence to `2.13x`. The restored tiled finalize path then
  produced a fresh checked bridge artifact at `3.04x`, so keep the per-output
  tile finalize until a replacement proves both bridge and full-model Q8 prompt
  stability.
- The frontier gate now also reports the paired row-chain diagnostic
  `qrow group full-prefill x4 m=128 n=512 k=512 projection_row_chain_group`.
  This compares four staged qmatmul+residual+RMSNorm-scale row chains against
  the current semantic `projection_row_chain` lowering and enforces the same
  row-chain correctness ceiling. It is intentionally diagnostic: it tells us
  whether a larger semantic sublayer is promising before we write a tiled
  row-wide reduction kernel, but it does not lower the full-model dispatch split
  or promote projection row-chain fusion by default.
  The gate also prints canonical scheduler shape evidence for each row-chain
  diagnostic via `ProgramCommandStreamShape.fromCommands`: a single row-chain
  command covers 5 ops and records `shape_saved_dispatches=4`, while the x4
  full-prefill group records 4 row-chain commands, 20 covered ops, and
  `shape_saved_dispatches=16`. It also executes the compiled Metal Program once
  with region ProgramCommand dispatch enabled and prints
  `runtime_command_dispatches`, proving the public runtime profile sees one
  command object lower through two backend command dispatches for each split
  row-chain diagnostic and eight backend command dispatches for the x4 group.
  This keeps the semantic scheduler win explicit
  without confusing it for the still-missing tiled qmatmul row-chain throughput
  kernel.
  The same frontier gate now reports
  `projection_row_chain_single_dispatch` prompt/full-prefill/SmolLM-prompt
  lanes for the tiled single-dispatch candidate. Those lanes are dispatch-profile
  checked and print their observed max-abs-diff, but remain diagnostic until
  they are both correct and faster at full-prefill and SmolLM-prompt promotion
  shapes instead of only improving smaller prompt tiles.
- Q8_0 tied LM-head logits are a standalone backend qmatvec dispatch outside
  the ProgramCommand stream. Accepted default Q8_0 prompt evidence is gated at
  241 ProgramCommands and 242 dispatches because the prompt path keeps
  projection-chain plus row-chain commands until row-chain fusion proves
  full-model speed. Q8_0 decode is gated at 212 dispatches and 211
  ProgramCommands. Removing the dense `op` command or fusing projection
  row-chains is a structural win only when fallback remains zero and
  dispatch-accounting stays explicit.
- Do not enable prefill projection RoPE-store sidecars by default until the
  Adapter is tiled enough to win throughput. The scalar pair-column experiment
  in `bench-results/smollm-20260601T162410Z-p128-g200-r3.json` lowered the
  old prompt roof from 242 to 212 dispatches/call but regressed prompt
  throughput, so the default planner keeps prefill RoPE/cache store as separate
  commands.
- The next decode-depth target is a semantic layer or sublayer command that
  shares one materialized normalized input across projection/cache and MLP
  projections. Do not collapse `row_chain` into projection-cache commands with a
  one-dispatch Metal kernel unless the Adapter proves it avoids per-output-tile
  RMSNorm recomputation; otherwise it is either two dispatches hidden behind one
  command name or a likely throughput regression.
- Producer-sidecar lowerings prove dependency, offset, and liveness legality in
  the pure planner, then let Metal write the final side effect directly.
- Runtime profiles are the gate evidence: every compiled-program Adapter must
  expose backend/fallback placement, dispatch counts, schedule-region lowering,
  command counts, runtime patch changes, and the bounded runtime patch-hole
  count.
- Execution-plan allocation or command-plan construction errors must fail
  compilation rather than degrade into empty schedule regions.
- A no-readback decode row is available only as a diagnostic. It is not a
  parity lane because llama.cpp also materializes logits for `llama-bench`; on
  current Q8 smoke runs it does not beat the normal native row, confirming that
  command depth, not logits download, is the first-order bottleneck.
- Experimental perf knobs stay opt-in until model benchmarks prove a default win.
- Quantized prompt projection-row-chain remains a candidate semantic
  ProgramCommand because the full-model default is performance-gated. It can
  make the command shape look better by removing projection-chain row frontiers,
  but the substrate gate must still distinguish that command-shape win from a
  throughput win. The frontier microbench also requires local speedup and
  max-absolute-difference correctness before reporting the single-kernel
  candidate as ready; it includes full-prefill and SmolLM-prompt diagnostics for
  both the current two-dispatch semantic command and the tiled single-dispatch
  candidate. Prompt-sized qmatmul row-chain commands are protected by a
  multi-tile regression that proves the safe command path routes through tiled
  `qmatmul_elementwise_f32` plus the RMSNorm-scale row pass, while
  `projection_row_chain_single_dispatch` exposes the tiled row-chain hook
  directly. The scalar kernel stays diagnostic for the qmatvec/decode-style
  path, and the prompt-sized candidate path still keeps decode qmatvec unfused
  while preserving prompt qmatmul evidence.
  Current local frontier evidence shows the single-dispatch tiled candidate is
  correct and can improve the smaller prompt tile, but full-prefill and
  SmolLM-prompt shapes remain effectively neutral or slower than the split
  command path. The active frontier is therefore still tiled row-chain
  throughput or a larger
  semantic sublayer, not command-count reduction by itself.
- Benchmark artifacts keep raw outputs, gate-selected summaries, and gate
  decisions so this document does not become a changelog.

## Acceptance Thresholds

SmolLM-135M:

- `pp128`: zgml >= 90% of llama.cpp Metal F16/Q8_0 on the same machine.
- `tg200`: zgml >= 90% of llama.cpp Metal F16/Q8_0 on the same machine.
- Default release build: `zig build -Doptimize=ReleaseFast` passes without ggml
  installed.
- Benchmarks gate only the intended Metal lanes and record fallback/profile data
  so CPU fallback cannot masquerade as GPU parity.

1B-class target:

- `pp512`: zgml >= 90% of llama.cpp for equivalent F16 and quantized formats.
- `tg128`: zgml >= 90% of llama.cpp for equivalent F16 and quantized formats.
- Memory use <= 115% of llama.cpp for equivalent quant formats.

## Gates

Use `zig build check` as the default local/subagent gate. It runs the unit tests,
builds benchmark artifacts, and verifies the checked compact baseline without a
full parity run.

The source-checkout npm workflow exposes the same proof path for JS/TS contributors:

```sh
npm run bench
npm run bench:status
npm run bench:frontier
npm run bench:ggml
npm run bench:ggml:parity
```

`npm run bench` is intentionally the cheap checked-baseline gate.
`bench:status` is the cheap evidence-reading gate: it verifies the checked
baseline artifacts and, when present, the latest local `bench-results/`
p128/g200/r3 full-run artifact without rerunning Metal. These benchmark
artifact commands are source-checkout evidence, not packaged npm runtime API.
It also prints a latest-vs-checked-baseline delta for the selected native lanes,
including throughput ratio plus dispatch/fallback shape, so a reader can tell
whether the current full run is above the checked floor without opening JSON.
The same status readback ends with a compact `perf-next:` line that ranks the
current artifact-derived bottlenecks: weakest full-model ggml lane, PyTorch
median misses when present, Q8 prompt semantic readiness, and qsemantic frontier target. The
frontier segment prints both candidate speedup and candidate-vs-current-default
speedup so a tiled diagnostic cannot look promotable merely because it beats the
staged baseline. Use that line to choose the next focused microscope before
paying for a full parity run.
For PyTorch comparison artifacts, `bench:status` prefers the newest eleven-lane
broad artifact for `pytorch-results:` and `perf-next:` when one exists. Newer
focus or one-lane microscopes still appear as `pytorch-focus-results:` or
`pytorch-latest-results:`, so diagnostics stay visible without replacing the
broader PyTorch-competitiveness scoreboard.
The current selected broad artifact is a June 27, 2026 three-attempt,
80ms-window run against PyTorch `2.12.1` with fresh native code:
`lane_pass=11/11`, `median_lane_pass=11/11`,
`first_contact_inference=11/11`, and
`ratio_median=linear_batched:1.24x,lazy_matmul_add_gelu_batched:2.84x,lazy_mlp_batched:1.80x,lazy_rms_silu_ffn_batched:1.67x,lazy_conv2d_relu_batched:1.19x,max_pool2d_batched:8.70x,avg_pool2d_batched:4.98x,rms_gelu_linear_batched:2.73x,softmax_classifier_batched:1.16x,log_softmax_classifier_batched:1.43x,lazy_token_head_batched:1.65x`.
Both module-backed lanes and named-parameter lazy-graph lanes now prove the
friendly `zgml.native` / `compile.compileForInference` handle reaches
the allocation-free prepared Program path.
The lane-selectable `dev:perf:competitive` runner uses that same eleven-lane,
three-attempt, 150ms-window PyTorch broad set and includes the native eager gap
lane by default, so the daily competitiveness loop proves both the compiled
replacement scoreboard and the no-grad native module bridge instead of only
rerunning the older two-lane current-gap microscope.
For Q8 prompt candidate artifacts, `bench:status` prefers the newest steady
viable run (`attempts >= 3` with command, two-phase, and semantic lanes) over a
newer one-attempt quick probe, and prints `q8-prompt-latest-results:` when it
does so. That keeps `q8-prompt-results:` tied to the broad all-lane scoreboard
while still making freshness visible. The active semantic bottleneck has its
own `q8-prompt-semantic-steady-results:` readback; when present, `perf-next:`
uses that newest steady semantic artifact as pressure evidence instead of a
newer one-attempt probe or an older all-lane artifact.
For qsemantic frontier artifacts, `bench:status` follows the same stability
policy: it prefers the newest artifact with at least three attempts and prints
`frontier-latest-results:` when a newer one-attempt microscope exists. That keeps
the semantic frontier useful for iteration without letting a noisy quick probe
steer `perf-next:`.
`npm run dev:perf:next` automates that choice: it reads `perf-next:` and runs
the smallest no-rebuild microscope for the current bottleneck. Use
`npm run dev:perf:next:build` when the native or benchmark artifact is stale,
`npm run dev:perf:next:steady` when a noisy lane needs repeated attempts or
longer timing windows, or set
`BENCH_NEXT_PERF_LANE=pytorch|qsemantic|qsemantic_throughput|qsemantic_bridge|qsemantic_input_bridge|qproj|q8_prompt|ggml`
to force a lane. The automatic qsemantic handoff is two-step: below-default or
missing fresh throughput evidence keeps the loop on `qsemantic_throughput`,
while fresh above-default qsemantic throughput advances the loop to `q8_prompt`
so model-level prompt viability is tested before ggml promotion. If
`perf-next:` reports `q8_prompt=semantic_bridge_candidate` with
`next=steady_semantic_bridge_candidate`, the router chooses `q8_prompt` and
forces the steady paired-default settings for that proof run; this keeps a
bridge candidate from looking promoted before its worst-case model-level
evidence is stable. Bridge readiness requires worst-case non-regression, not
only median parity. If the bridge candidate still has `worst < 1.0x`,
`perf-next:` reports `next=semantic_bridge_throughput_kernel` and
`dev:perf:next` routes back to the semantic throughput microscope.
If the latest bridge evidence reports single-dispatch refusals concentrated on
`dim`, `perf-next:` reports `next=semantic_width_parallel_kernel` instead. That
is the actionable signal from the rejected `SEMANTIC_FFN_MAX_DIM=2048` probe:
the next kernel must preserve width/tile parallelism for the larger hidden
dimension, not merely enable the row-serial single-dispatch path. The
`dev:perf:next` router sends this case to `qsemantic_bridge`, which routes
through the checked `frontier-qsemantic-bridge-*.json` artifact instead of the
broader qsemantic pair. You can still force that lane explicitly with
`BENCH_NEXT_PERF_LANE=qsemantic_bridge npm run dev:perf:next{,:run}`.
When an input-bridge artifact is available,
`qsemantic-input-bridge-results:` remains the selected steady signal and
`qsemantic-input-bridge-latest-results:` reports newer one-attempt probes. That
keeps fresh `semantic_with_input_width_parallel_kernel` iteration visible
without letting a quick probe replace the steadier `perf-next` target. The
checked gate also fails three-attempt input-bridge runs when the best absorbed
speedup falls below the collapse floor (`2.45x` by default), so
correct but weaker kernel probes do not become invisible regressions. Its
compact status line now also includes `decomposed_extra`, so a passing legacy
artifact still advertises how far it is from the intended one-dispatch
input-bridge kernel. Force that
exact microscope with
`BENCH_NEXT_PERF_LANE=qsemantic_input_bridge npm run dev:perf:next{,:run}`.
For qsemantic kernel work, `BENCH_QSEMANTIC_VARIANTS=target` limits the raw
frontier harness to the staged baseline plus the one-dispatch semantic target,
while `BENCH_QSEMANTIC_VARIANTS=throughput_candidate` limits it to the staged
baseline plus the mixed tiled-tail throughput candidate. Use the checked
`npm run dev:perf:frontier:qsemantic:throughput{,:run}` loop first for the
current throughput candidate; it writes a separate
`frontier-qsemantic-throughput-*.json` diagnostic artifact so the normal
`bench:status` qsemantic selector is not promoted by a variant-only run. Use
`npm run dev:perf:frontier:qsemantic:target:raw{,:run}` for row-serial target
diagnostics and `npm run dev:perf:frontier:qsemantic:throughput:raw{,:run}` for
raw benchmark text from the same semantic throughput-kernel loop.
`bench:ggml` builds the benchmark binaries in ReleaseFast, writes a full
`bench-results/*.json` artifact, and requires the checked M5 Pro baseline so a
new local artifact cannot improve a ggml percentage by merely running both
zgml and llama.cpp slower. `bench:ggml:parity` keeps that baseline guard and
also makes the 90% parity threshold required instead of diagnostic.
Complete full-run artifacts that fail their required gate are moved to
`bench-results/failed/` automatically; successful artifacts stay in
`bench-results/` and become eligible for `bench:status` selection.

When a macOS validation host has an unavailable or wedged Metal stack, use
`zig build check -Duse-metal=false -Duse-blas=false` for an explicit CPU/Wasm
gate. That mode is not Metal performance evidence; it exists to keep tests,
Wasm handles, benchmark artifact construction, and stencil proof checks
verifiable while Metal-specific gates remain blocked.

Use the ReleaseFast build for all parity runs:

```sh
zig build -Doptimize=ReleaseFast
zig build bench-baseline-check
python3 scripts/verify_bench_artifact.py --status benchmarks/baselines/smollm-m5pro-p128-g200-r3.json benchmarks/baselines/smollm-stencil-p128.json
BENCH_BASELINE_JSON=benchmarks/baselines/smollm-m5pro-p128-g200-r3.json ./scripts/bench_vs_ggml.sh 128 200 3
python3 scripts/verify_bench_artifact.py --status bench-results/<latest>.json
BENCH_BASELINE_JSON=benchmarks/baselines/smollm-m5pro-p128-g200-r3.json BENCH_REQUIRE_PARITY=1 ./scripts/bench_vs_ggml.sh 128 200 3
BENCH_REQUIRE_PARITY=1 BENCH_BASELINE_JSON=bench-results/<baseline>.json ./scripts/bench_vs_ggml.sh 128 200 3
BENCH_BASELINE_JSON=benchmarks/baselines/smollm-m5pro-p128-g200-r3.json ./scripts/bench_vs_ggml.sh 128 200 3
```

`bench_vs_ggml.sh` runs zgml F16 and Q8_0 lanes with
`--metal-prefill-device --metal-decode-region --gate-only` by default, so the
selected zgml rows are native backend-only prompt and decode evidence.
The cheap `dev:perf:ggml:q8-command-smoke` loop wraps the same script with
`scripts/check_ggml_q8_command_smoke.cjs`, requiring the Q8 prompt lane to be
`metal scheduled prefill projection-row-chain command`, `242/242 dispatch,
151/151 command`, `90/90 dispatched semantic_ffn_sublayer`,
`60/60 dispatched projection_row_chain`, and the existing cache-group command
evidence before it prints
`ggml q8 command smoke: pass` with the accepted artifact path.
`verify_bench_artifact.py --status` prints the evidence class for checked
artifacts: full benchmark runs are full-run evidence and explicitly report
whether parity passed or missed, compact baselines are structural
native-shape/baseline gates, and stencil artifacts are offline shape/hash gates.
Only a full run with `gates.overall_pass=true` is a parity claim.

The JSON artifact is the source of truth:

- `summary.gate_zgml` selects native backend-only lanes for parity evidence.
- `gates.reference_backend` requires the parsed llama-bench rows to be Metal
  device rows. When llama-bench emits a `dev` column, the gate checks that
  actual device rather than the broader loaded-backend list, so a BLAS/CPU lane
  cannot satisfy required Metal parity merely because the MTL backend was also
  loaded.
- `gates.native_execution` requires matched profile windows, zero fallback,
  cached region command plans, zero dynamic plans, and no increase beyond the
  dispatch/command-per-call roof implied by the current ProgramCommand shape. It
  also asserts the exact encoded/dispatch/attempt/refusal shape and rejects
  extra command counters. An apparent dispatch-depth win cannot add a new
  semantic command or remove RoPE/projection/cache command evidence unless the
  replacement is explicitly benchmarked.
- `gates.baseline` compares throughput against a compatible artifact with a 5%
  floor and applies the same 5% ceiling to lower-is-better structural counters.
  Semantic stage/token and runtime patch-hole fields are exact native execution
  evidence, not baseline counters; shrinking program shape is not a throughput
  win. Baseline and current lanes must include matched profile-window evidence;
  baseline lanes must also include at least three zgml samples. Stored native
  execution gates are accepted only when they include current semantic
  stage/token evidence; otherwise the summary rows must satisfy the strict
  native execution shape.
- `program_commands`, `program_op_commands`, and projection sidecar counters are
  emitted as artifact fields so dispatch-depth work is benchmark-gated instead
  of checked by hand.
- `ProgramStencil.inspect()` also carries cold command-stream shape evidence
  for the executable op tape. This is available to backend profiles for
  regression tests; benchmark artifact fields remain intentionally strict and
  should only grow with an explicit baseline schema update.
- `semantic_stage_count`, `semantic_token_count`, and the flat semantic program
  shape fields are emitted for device prompt/decode rows, so the benchmark gate
  proves those lanes still flow through the compiled LLaMA patch path and exact
  measured token window.
- `runtime_patch_holes` and its cache-write/attention-length split are emitted
  for compiled device rows and must match the Zig-emitted semantic stencil
  shape, including `semantic_runtime_patch_cache_write_pos_holes` and
  `semantic_runtime_patch_attention_seq_kv_holes`. The Python gate derives the
  expected stencil shape from those semantic fields, so copy-and-patch work
  cannot quietly widen, narrow, or reshuffle the runtime bindings without
  updating the semantic evidence. The checked compact baseline must include the
  same fields so offline verification gates the patch bindings even when Metal
  is unavailable locally.
- Program inspection also exposes the bounded runtime patch envelope: max cache
  write position and max attention sequence length. Those bounds are the
  StepParams validity evidence future WebGPU/wgpu update-buffer code should use;
  benchmark rows remain focused on hole counts, profile calls, and stencil
  hashes unless the artifact schema is explicitly expanded.
- `runtime_patch_stencil_hash` is the backend-emitted fingerprint of the
  ordered runtime patch holes and their fixed geometry. `DeviceInference`
  compares it against the canonical `ProgramStencil` inspection for the same
  `DeviceProgram` whenever a caller requests runtime patch evidence, so a
  backend Adapter cannot satisfy the copy-and-patch contract with the right
  counts but a different patch layout. The canonical stencil is now inspection
  evidence over the same executable skeleton that backend execution patches,
  not a detached pure-shape helper. It is shape evidence, not a lower-is-better
  perf counter. Full native execution evidence requires it, and baseline
  regression runs fail if their baseline artifact lacks current native execution
  evidence with a valid stencil hash. The compact checked-baseline verifier now
  requires a valid hash on every native zgml lane while keeping the offline
  schema check runnable on machines without Metal.
- `runtime_patch_calls` must equal the expected profile-call window for prompt
  and decode rows. This proves every compiled execution reaches the patch seam;
  decode rows additionally require `runtime_patch_changed == expected calls`.
- `runtime_patch_invalid` must be absent or zero; invalid runtime windows are
  refused before execution and cannot satisfy native execution evidence.
- zgml rows are sampled three times by default and the median parsed row is
  gated; structural counters still gate the selected row. Set
  `BENCH_ZGML_SAMPLES=1` only for quick smoke checks.
- Full benchmark artifacts attach `sample_count`, `sample_*_tok_s`, and
  `sample_range_pct` to each selected zgml lane.

`benchmarks/baselines/smollm-m5pro-p128-g200-r3.json` is a compact checked-in
baseline for the M5 Pro target machine. It keeps exact-schema compatibility
metadata plus selected zgml lane fields consumed by the regression gate and
native ProgramCommand shape verifier. It excludes raw stdout, full-run `gates`,
parity summaries, display labels, and raw duplicate counters. The checked
baseline is intentionally a conservative guard floor rather than the best
observed run, so ordinary local variance does not fail the gate. The current
compact baseline is refreshed from
`bench-results/smollm-20260604T064807Z-p128-g200-r3.json` because the older
June 1 compact artifact lacked the now-required `runtime_patch_stencil_hash`
native-evidence field. The rejected prefill RoPE-store regression from
`bench-results/smollm-20260601T162410Z-p128-g200-r3.json` still remains the
kind of throughput regression this baseline is meant to catch.

`benchmarks/baselines/smollm-stencil-p128.json` is the checked offline
copy-and-patch stencil artifact. It is generated from:

```bash
./zig-out/bin/bench-llama-smollm ignored 128 1 1 --stencil-only
```

That mode builds the SmolLM decode and prefill `DeviceProgram` shapes with Metal
planning capabilities through the internal stencil backend, then records the
exact `ProgramStencil` inspection hash without loading weights, executing
kernels, or requiring a Metal device. Decode evidence now comes through the
persistent `compileDeviceDecodeProgram(...)` path, so the offline stencil row
tracks the same Program object shape that executable decode sessions bind.

`zig build bench-baseline-check` runs
`scripts/verify_bench_artifact.py` against the checked compact baselines without
requiring Metal or `llama-bench`. `zig build check` also runs the stencil probe
binary with `--stencil-only` and fails if the generated p128 decode/prefill
hashes drift from the checked exact-stencil artifact. This is a
schema/native-shape gate, not a throughput run: the full parity run above is
still required for new performance claims. Native ProgramCommand shape budgets
live in
`scripts/bench_contract.py` and are shared by both the full parity gate and the
compact baseline verifier.

After a parity milestone is locked, release checks should fail if prompt or
decode throughput regresses by more than 5% against the recorded target-machine
baseline. Exceptions must be explicitly accepted in the benchmark report.
