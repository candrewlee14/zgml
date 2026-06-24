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
  `dev:perf:q8-prompt:viable` set `BENCH_Q8_PROMPT_LANES=command,two_phase`
  and default to one attempt, so command/two-phase changes can be checked
  without rerunning the known-bad single-dispatch diagnostic on every edit. The
  full all-lane candidate gate remains the release proof before promotion.
  The two-phase partial kernel does not bind the scale buffer anymore; scale is
  only needed by the finalize pass. This keeps the candidate ABI shape smaller
  without changing command semantics.
- The current weakest checked lane is Q8_0 prompt at roughly 30% of llama.cpp.
  Its pressure is not an obvious wrong-kernel issue: the remaining
  `projection_chain:60` work is prefill-shaped qmatmul plus add/mul sidecars,
  already lowered by the tiled `qmatmul_elementwise_f32` Adapter with primary
  output elision. The next
  meaningful Q8 prompt move should therefore be either
  a semantic sublayer command that removes real command depth, or quantized
  projection-chain layout/kernel work that improves throughput without hiding
  extra dispatches behind a new command name. Use `dev:perf:frontier:qproj`
  for that kernel/layout loop: it rebuilds the benchmark artifacts with
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
- The full-model Q8 prompt probe now distinguishes general projection groups
  from attention/cache projection groups. Its Q8 evidence reports
  `projection_group=0->0`, `projection_cache_group=30->30`, and
  `decode_projection_cache_group=30`, so the attention/cache group path is
  already active in the full model. The remaining Q8 prompt work is the
  `projection_chain=60` row-chain/semantic-sublayer lane; the frontier x7 qproj
  region is the isolated executable proof for general qmatmul-plus-sidecar
  projection groups.
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
  command dispatch for each single row-chain diagnostic and four command
  dispatches for the x4 group. This keeps the semantic scheduler win explicit
  without confusing it for the still-missing tiled qmatmul row-chain throughput
  kernel.
  The same frontier gate now reports
  `projection_row_chain_single_dispatch` prompt/full-prefill/SmolLM-prompt
  lanes for the tiled single-dispatch candidate. Those lanes are correctness-
  and dispatch-profile checked, but remain diagnostic until they beat the
  full-prefill and SmolLM-prompt promotion floor instead of only improving
  smaller prompt tiles.
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
