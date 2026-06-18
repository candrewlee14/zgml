# zgml

A tiny TS-first ML library with PyTorch-shaped ergonomics and inspectable
native executable programs.

The tensor API is PyTorch-shaped, with automatic differentiation, a small
primitive IR, and auto-fused kernels. The inference lane is sharper: compile a
model into a reusable Program, bind persistent Session state, update small
per-step parameters, and execute through Node, Bun, browser/Wasm, C, or native
Zig runtime handles.

The intended shape is one library with two temperatures. Write, debug, and train
small models with familiar `Tensor`, `nn`, `loss`, `optim`, `train`, state-dict,
and autograd vocabulary. Then compile the stable work into an explicit
`Program`/`Session` handle when the hot path needs native speed, FFI-friendly
binding, and evidence.

```ts
import { torch } from "zgml";
import type { Tensor } from "zgml";

const model = new torch.nn.Sequential(
  new torch.nn.Linear(2, 4),
  new torch.nn.ReLU(),
  new torch.nn.Linear(4, 1),
);

const optimizer = new torch.optim.AdamW(model, { lr: 3e-2, weight_decay: 1e-3 });
const scheduler = new torch.optim.lr_scheduler.StepLR(optimizer, { step_size: 20, gamma: 0.5 });
const criterion = new torch.nn.MSELoss();
const dataset = new torch.utils.data.TensorDataset(
  torch.tensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]),
  torch.tensor([0, 1, 1, 0], [4, 1]),
);
const loader = new torch.utils.data.DataLoader(dataset, { batch_size: 2, shuffle: true });

torch.train.fitModule(optimizer, model, loader, criterion, {
  epochs: 8,
  zero_grad: true,
  clip_grad_norm: 1,
  onStep() {
    scheduler.step();
  },
});

const snapshot = torch.checkpoint.create({ model, optimizer, scheduler, prefix: "xor" });
const text = torch.checkpoint.stringify(snapshot, 2);
const loaded = torch.checkpoint.parse(text);
torch.checkpoint.restore(loaded, { model, optimizer, scheduler, prefix: "xor", strict: true });
// Direct in-memory restore also works:
torch.checkpoint.restore(snapshot, { model, optimizer, scheduler, prefix: "xor", strict: true });

const program = torch.compile(model, { inputShape: [2] as const });
```

The same pieces remain available as small TS-first namespaces when you want a
more explicit import surface:

```ts
import { F, checkpoint, compile, data, load, nn, optim, save, tensor, train } from "zgml";

const model = nn.sequential([
  nn.linear(2, 4),
  nn.relu(),
  nn.linear(4, 1),
]);
const optimizer = optim.adamW(model, { lr: 3e-2, weightDecay: 1e-3 });
const scheduler = optim.stepLR(optimizer, { stepSize: 20, gamma: 0.5 });
const criterion = new nn.MSELoss();
const dataset = data.tensorDataset(
  tensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]),
  tensor([0, 1, 1, 0], [4, 1]),
);
const loader = data.dataLoader(dataset, { batchSize: 2, shuffle: true });

train.fitModule(optimizer, model, loader, criterion, {
  epochs: 8,
  zeroGrad: true,
  clipGradNorm: 1,
  onStep() {
    scheduler.step();
  },
});

const snapshot = checkpoint.create({ model, optimizer, scheduler, prefix: "xor" });
const text = save(snapshot, 2);
const loaded = load(text);
load(text, { model, optimizer, scheduler, prefix: "xor", strict: true });
const savedPath = save(snapshot, "./xor.zgml", 2);
load(savedPath, { model, optimizer, scheduler, prefix: "xor", strict: true });
const program = compile(model, { inputShape: [2] as const });
void F;
void loaded;
void program;
```

Subclassed models keep the same typed path. This is the shape zgml optimizes for:
write ordinary TS module code, train eagerly, then bind the trained state into a
compiled Program/Session boundary when the hot path is stable.

```ts
class Classifier extends torch.nn.Module<readonly [2], readonly [2]> {
  readonly graph = new torch.nn.Sequential(
    new torch.nn.Linear(2, 4),
    new torch.nn.ReLU(),
    new torch.nn.Linear(4, 2),
  );

  forward(input: Tensor<readonly [2]>): Tensor<readonly [2]>;
  forward(input: Tensor): Tensor;
  forward(input: Tensor): Tensor {
    return this.graph.forward(input);
  }
}

const classifier = new Classifier();
const criterion = new torch.nn.CrossEntropyLoss({ classes: 2 });
const classifierOptimizer = new torch.optim.AdamW(classifier, { lr: 1e-2 });
const classifierDataset = new torch.utils.data.TensorDataset(
  torch.tensor([-1, -1, 1, -1], [2, 2]),
  torch.tensor([0, 1], [2]),
);
const classifierLoader = new torch.utils.data.DataLoader(classifierDataset, { batch_size: 1 });

torch.train.fitClassifier(classifierOptimizer, classifier, classifierLoader, criterion, {
  epochs: 4,
  zero_grad: true,
});

classifier.eval();
torch.train.evaluate(classifierLoader, (batch) => {
  if (!batch.target) throw new Error("classifier eval batch requires targets");
  return criterion.forward(classifier.forward(batch.input), batch.target);
});
const predictions = torch.train.predictClassifier(classifier, classifierLoader);

const program = torch.compile.compile(classifier, { inputShape: [2] as const });
const session = program.bindModule(classifier);
const input = torch.tensor([1, -1], [2] as const);
const logits = torch.inference_mode(() => session.stepTensor(input));
const output = new Float32Array(2);
const hotParams = { input, output };
session.requireHotStepParams(hotParams);
session.hotPathPlan(hotParams);
const logitsInto = torch.inference_mode(() => session.executeInto(output, { input }));
const probabilities = torch.F.softmax(logits, -1);
const classes = torch.train.predictClasses(logits, { classes: 2 });
const classesInto = torch.train.predict_classes(logitsInto, { numClasses: 2 });
```

The composable namespace form is equivalent:

```ts
class Classifier extends nn.Module<readonly [2], readonly [2]> {
  readonly graph = nn.sequential([
    nn.linear(2, 4),
    nn.relu(),
    nn.linear(4, 2),
  ]);

  forward(input: Tensor<readonly [2]>): Tensor<readonly [2]> {
    return this.graph.forward(input);
  }
}

const classifier = new Classifier();
const namespaceCriterion = new nn.CrossEntropyLoss({ classes: 2 });
train.evaluate(loader, (batch) => namespaceCriterion.forward(classifier.forward(batch.input), batch.target));
train.predictClassifier(classifier, loader);
const namespaceProgram = classifier.compile({ inputShape: [2] as const });
const namespaceSession = namespaceProgram.bindModule(classifier);
const logits = namespaceSession.stepTensor(tensor([1, -1], [2] as const));
const classes = train.predictClasses(logits, { classes: 2 });
```

For fully explicit loops, use the same PyTorch-shaped pieces directly:
`optimizer.zeroGrad()`, `loss.backward()`, `optimizer.step()`, parameter
`requiresGrad` flags, and then `model.compile(...)` once the model is ready for
the hot path. The Node and Bun `manual_loop` smokes keep that workflow covered.

## Features

**Small primitive IR.** A compact set of structural, elementwise, reduction,
indexing, matmul, and deliberately promoted composite ops. Frontend conveniences
either lower to that set or compose it, and backward rules stay attached to the
same compact graph vocabulary.

**Zero-noise API.** Product code is authored once in TypeScript and emitted by
`tsdown`; native execution is reached through Programs, Sessions, buffers, and
adapters instead of a mirrored Zig frontend:
```ts
const activated = hidden.mm(w1).add(b1).gelu();
const ffOut = activated.mm(w2).add(b2);
```
There is no frontend sync pass by design. If a feature is product semantics,
add it in `src/ts/**`; if it needs native speed, lower or bind it through the
Program/Session/ABI contract instead of re-authoring the same API in Zig.
Graph-owned constructors use familiar names (`zeros`, `ones`, `full`,
`arange`, `linspace`, `rand`, `randn`, `scalar`, `parameter`, `param`) so
small JS/TS models can start from ordinary tensor vocabulary and only opt into
`Program`/`Session` when they need compiled execution.

**Auto-fusion.** Stable lazy tensor work can lower through the runtime compiler
into compact native kernels:
```ts
const program = model.compile({ inputShape: [2] as const });
const session = program.bindModule(model);
const y = session.stepTensor(tensor([1, 0], [2] as const));
```

**Executable LLM sessions.** LLaMA-family inference uses the explicit runtime
shape: load/create Model, compile Program, bind Session, step tokens. The C ABI
and JS/Wasm examples expose opaque model/program/session/buffer handles instead
of tensor internals. Caller-owned logits buffers use `stepInto`,
`prefillInto`, or `executeInto` for hot loops that should not allocate output
storage; `NativeBuffer.readFloat32()` / `readBytes()` allocate full-buffer
readback views by default, while `NativeBuffer.readFloat32Into` and
`readBytesInto` fill caller-owned storage for hot loops without forcing generic
FFI buffers through tensor views. Caller-owned logits buffers are checked against
the full vocab length before native execution. Tensor-first callers can use
`stepTensor`, `executeTensor`, `prefillTensor`, and `readOutputTensor` to keep
logits in the eager frontend shape, including caller-owned Tensor or
`Float32Array` carriers. LLaMA-family Programs and Sessions expose
`bufferLayout()`, `kvCacheLayout()`, and `outputShape()` as the logits/cache
edge contract, matching the generic Program/Session shape vocabulary.
`TinyLlamaSession.stepContract()` exposes the token StepParams contract too:
token-vs-window support, logits length/shape, context position, bound-output
kind, default output/readback path, remaining context capacity, readback
availability, logits byte length, output slot name/role, no-output state
advancement, `defaultReadbackRequired`, `defaultAllocationFree`,
`defaultHotPath`, default output effect/ownership/return ownership, and a stable
`signature` that excludes the moving position.
`TinyLlamaSession.matchesStepContractSignature(signature)` compares that cached
contract directly, while
`TinyLlamaSession.acceptsStepParams(params)` reuses the token validation path
as a non-executing guard before a hot loop, while
`TinyLlamaSession.acceptsAllocationFreeStepParams(params)` additionally requires
the accepted call shape to avoid runtime output allocation for
already-normalized inputs, and
`TinyLlamaSession.acceptsRuntimeOutputAllocationFreeStepParams(params)` names
that exact predicate while keeping the older compatibility alias available.
`TinyLlamaSession.acceptsNoReadbackStepParams(params)` requires the accepted
call shape to avoid default/native readback, while
`TinyLlamaSession.acceptsReadbackFreeStepParams(params)` names that same
positive evidence predicate, and
`TinyLlamaSession.acceptsHotStepParams(params)` requires both allocation-free
and no-readback evidence.
`TinyLlamaSession.matchesStepParamsSignature(params, signature)` compares a
preflighted call shape against a cached `stepParamsSignature`.
`TinyLlamaSession.matchesStepParamsCompatibility(params, compatibility)` does
the same comparison against a frozen preflight record.
These preflight helpers do not execute or bump call counters.
`TinyLlamaSession.preflightStepParams(params)` is the friendly evidence verb;
`TinyLlamaSession.stepParamsCompatibility(params)` is the compatibility alias.
Both accept unknown boundary values and return the frozen accepted flag,
typed accepted/rejected status, contract kind, stable contract signature,
stable StepParams call-shape
signature, current contract position, per-call `allocationFree` evidence, and
typed `inputSource` / `outputTarget` / `outputEffect` routing evidence, planned
input/output element types, shapes, shape signatures, element lengths, byte lengths, and
`stateEffect`, `inputOwnership`, `outputOwnership`, `outputReturnOwnership`,
`readsInput`, `writesOutput`, `readbackRequired`, `readbackFree`,
`runtimeOutputAllocationFree`, plus top-level `hotPath`, `hotPathStatus`,
`hotPathBlockers`, `rejectionCode`, and rejection diagnostics.

```ts
const info = zgml.probeModel(bytes);
const model = zgml.loadModel(bytes, { kind: info.modelKind });
const program = model.compile({ backend: "cpu", contextLength: 2048 });
const logitsShape = program.outputShape();
const session = program.bind({ output: "native" });
const next = session.stepArgmax(token).token;
const logits = session.readOutputInto(new Float32Array(program.vocabSize));
const logitsTensor = session.readOutputTensor();
const nextLogits = session.stepTensor(token);
```

Token selection and generation helpers return read-only result snapshots; caller
token/logits buffers remain the explicit mutable data edge.

**FFI-first artifacts.** `zig build ffi-c`, `zig build ffi-node-smoke`,
`zig build ffi-bun-smoke`, and `zig build ffi-wasm-smoke` prove the same handle
lifecycle across native C, Node, Bun, and Wasm. The current compatible
checkpoint catalog is intentionally exact: the tiny LLaMA smoke envelope and
SmolLM-135M, with unsupported shapes rejected before pretending they can run.

## Interface Shape

The public root exports `Tensor`, tensor factories, `nn`, `F`, `data`, `loss`,
`optim`, `train`, `checkpoint`, model-source helpers, and explicit
`Program`/`Session` handles from the TS-authored product surface.
`nn` is intentionally small: canonical ML vocabulary such as `Linear`,
`Embedding`, `LayerNorm`, `RmsNorm`, `Dropout`, `Sequential`, `gelu`/`relu`/`silu`/`sigmoid`/`step`
activation modules, unary modules such as `nn.exp()`, `nn.log()`,
`nn.neg()`, `nn.recip()`, `nn.abs()`, `nn.sqrt()`, `nn.square()`/`nn.sqr()`,
`nn.sgn()`/`nn.sign()`, tensor-level `clamp`/`clip`, stable `logsumexp`/`logSumExp`, and dim-preserving reductions `nn.sum(dim)`,
`nn.mean(dim)`, and `nn.max(dim)`, plus shape modules such as `nn.identity()`,
`nn.reshape(shape)`, `nn.view(shape)`, and `nn.flatten(...)`,
`nn.transpose()`, `nn.broadcastTo(shape)`, and `nn.expand(shape)`,
PyTorch-style JS/TS constructors such as `new nn.ReLU()`,
`new nn.GELU()`, `new nn.SiLU()`, `new nn.Sigmoid()`, `new nn.Flatten()`,
`new nn.Dropout()`, and `new nn.Identity()`, `softmax`, `linear`, and
`parameters` compose over tensor primitives instead of expanding the graph IR or
forcing new backend kernels. `loss` owns
PyTorch-familiar loss names such as `meanSquaredError`, `mse`, `classTargets`,
and `crossEntropy`. `optim` owns imperative optimizer state, including `SGD`,
`Adam`, and `AdamW`, outside the graph, so basic training is ergonomic without
adding IR or backend obligations. JS/TS optimizers expose
`stateDict()` / `loadStateDict()` for momentum and Adam moment snapshots, using
the same named, shape-aware entry convention as module state, and validate
optimizer snapshots before mutating step or moment buffers. Learning-rate
schedulers such as `stepLR` / `StepLR` and `exponentialLR` /
`ExponentialLR` attach to optimizers, expose state dictionaries, and round-trip
through checkpoints alongside model and optimizer state. `train` owns compact
loop helpers such as `backward`, `step`, `lossStep`, `fit`, `fitModule`,
`fitClassifier`, `clipGradNorm`, and `clipGradValue` without becoming a
policy-heavy trainer. Dataset batching,
per-step callbacks, module `train()` / `eval()` mode, and JSON-safe
`checkpoint.create` / `checkpoint.restore` are deliberate frontend policies
now, while metrics dashboards and larger experiment orchestration remain outside
the tiny core. Module-owned parameters now keep
durable gradient buffers across graph lifetimes, which is the ownership shape
the package should expose at every host edge. JS/TS modules and the root `nn`
namespace expose frozen `parameterNames()`, `parameterInfos()`, and
`parameterInfo(nameOrIndex)` snapshots so callers can inspect eager state names,
shapes, layouts, and scalar counts before compiling. `nn.stateDict(...)` snapshots those
named weights and `nn.loadStateDict(...)` restores them with strict name and
shape checks; JS/TS `stateDict(prefix)`, `loadStateDict(..., { prefix })`,
and `checkpoint.create/restore({ prefix })` share the same prefix contract for
composed modules. Checkpoint metadata is JSON-cloned and frozen as part of the
snapshot evidence, and `checkpoint.inspect(snapshot)` returns a frozen payload-free
summary of model/optimizer names, shapes, layouts, and scalar counts, with
`checkpoint.modelParameterInfo(...)` and `checkpoint.optimizerEntryInfo(...)`
lookup helpers for name-or-index access. Root `save(...)` and `load(...)`
provide the compact JSON text round-trip for ordinary checkpoint storage, while
Node/Bun root `save(snapshot, path)` / `load(path)` helpers write and read
checkpoint files directly;
`checkpoint.stringify(...)` and `checkpoint.parse(...)` expose the same validated
serialization policy from the checkpoint namespace for layers that do their own
file, blob, or KV I/O. Snapshots also carry optional layout strings, and restore paths
validate the full state table before mutating parameters, so state transfer can
distinguish Zig feature-major Linear weights from JS row-major Linear weights
without partially overwriting a model on error. `compileSupport()` / `canCompile()`
distinguish standalone native Programs, composable module-graph layers, and
eager-only frontend ops; f32 vector `nn.Linear` still compiles to the
exact tiny-linear Program/Session compatibility shape, while the traced module
Program path now covers fixed native-subset `Conv2d`, `MaxPool2d(2)`, and
`AvgPool2d(2)` shapes with structured diagnostics for unsupported configs. Zig `nn.Linear`
compile-support evidence includes allocation-free `inputShape()` /
`outputShape()` snapshots plus static input, output, weight, and bias scalar
lengths, so callers can allocate and validate the native binding contract before
creating a `ModuleProgram`; single-layer `Sequential` inherits the same
evidence. Multi-layer `Sequential` propagates those edge shapes and lengths
through shape-preserving glue, reports aggregate parameter scalar counts, and
rejects obvious layer-shape mismatches through `compileSupport()` before a
compile call. For module paths whose output shape depends on the exemplar input,
`nn.compileSupportForInputShape(T, alloc, &module, shape)` runs the cold trace
only and returns exact input/output/parameter evidence before constructing a
backend Program. Shape-specialized support and retained `ModuleProgram`
support also carry normalized Tensor Program IR hashes, so
`moduleCompatibility(...)` can reject same-parameter modules whose op tape
differs before Session binding.
Multi-layer
`nn.Dropout` is eager/autograd-capable, participates in module `train()` /
`eval()` mode cascades, lowers deterministic no-op Dropout (eval mode or
`p = 0`) as an identity native module Program, uses zero native dispatches when
that deterministic no-op is the whole compiled graph, and reports unsupported
native compile evidence for training-mode stochastic Dropout. `nn.Sequential` graphs, including the common
`nn.sequential([linear, relu|gelu|silu|sigmoid, linear])` MLP shape, lower through the
native traced module Program path. Native module Programs cover standalone and
Sequential `Linear`, shape-specialized `Embedding`, `Identity`, shape glue,
`relu`/`gelu`/`silu`/`sigmoid`,
`Softmax`, `LogSoftmax`, `sum`/`mean`/`max`, `Conv2d`, `MaxPool2d`,
`AvgPool2d`, `LayerNorm`, and `RMSNorm`; `Linear.compile({
input_shape: &.{features, batch} })` in Zig feature-major form,
`Linear.compile({ inputShape: [batch, features] })` in JS/TS row-major form,
`Embedding.compile({ inputShape: [tokens] })`, and supported
`Sequential.compile({ inputShape })` graphs lower through that path while
keeping the eager module and `bindParameters({ inputShape })` surface. The
class and factory forms share the same module implementations, so
`new nn.Flatten()` / `new nn.Linear(...)` compile through the same traced
Program path as `nn.flatten()` / `nn.linear(...)`. Direct
`TinyMlpModel` handles remain available for ABI compatibility, but
`nn.Sequential` no longer routes ordinary MLPs through a hidden tiny-MLP matcher.
The smoke matrix now proves common classifier shapes such as
`Sequential([Linear, LogSoftmax])` and token heads such as
`Sequential([Embedding, Linear, LogSoftmax])` compile, bind, and match eager
execution. Module
Programs can bind typed arrays or host
`NativeBuffer` weights/input/output buffers through the same Program/Session
contract; the C ABI also carries their WebGPU external-resource binding shape
for resource-probe evidence. Native Zig `ModuleProgram` and `ModuleSession`
values expose retained `compileSupport()` / `canCompile()` evidence,
`inputShape()` / `outputShape()`, `parameterTensorCount()`,
`parameterTensorName(index)`, `parameterTensorLayout(index)`,
`parameterTensorLen(index)`, `parameterTensorShape(index)`, and
`parameterLen()`, plus `parameterIndex(name)` and iterable
`parameterInfo(index)` / `parameterInfoByName(name)` / `parameterInfos()`
descriptors carrying name, layout, shape, and scalar count, plus
structured `moduleCompatibility(...)` diagnostics, `checkModuleCompatibility(...)`,
`acceptsModule(...)`, `bindModule(...)`,
`bindTensors(...)`, and `bindModuleTensors(...)`, while tensor-bound
`ModuleSession` values expose
`stepTensor(...)`, `executeTensor(...)`, and `outputTensor()`, so eager
`Tensor` buffers and persistent module state can be sized, bound, and carried
directly into explicit compiled execution without dropping to raw slices.
`ModuleProgram` also exposes binding requirement counts and a requirement hash,
while `ModuleSession` exposes the actual bound persistent/input/output counts,
host/resource counts, and binding-shape hash from the runtime Session.
`ModuleSession.uploadParameters()`, `uploadParameter(index)`,
`uploadParameterByName(name)`, and `uploadParameterRange(first, len)` refresh
persistent Session state explicitly, so hot loops can update only the module
state that changed.
`moduleCompatibility(...)` compares the candidate module's current
`compileSupport()` evidence before the parameter table, so architecture,
native-path, input/output shape, and scalar-count mismatches can be diagnosed
before binding, including same-length but different-rank boundaries such as
`[4]` versus `[2, 2]`. They
also expose
`runtimeProfile()` / `resetRuntimeProfile()` and accumulator-style
`addRuntimeProfileTo(...)`, with signed runtime-profile snapshots and
`matchesRuntimeProfileSignature(...)` keeping cold
Program evidence and hot Session work visible from the frontend API. Generic JS/TS Sessions also expose
signed `sessionCallProfile()` / `matchesSessionCallProfileSignature(...)` /
`resetSessionCallProfile()` as frozen frontend
helper-family, reset, and parameter-upload counters, separate from backend runtime-profile evidence. JS/TS exports the generic compiled handles as
`Program` and `Session`; older `TinyLinearProgram` and `TinyLinearSession`
names remain compatibility aliases for the same constructors.
The shared Sequential compiler boundary now lives outside the runtime-specific
wrappers, normalizes nested `Sequential` modules, preserves the single-linear
compatibility path, and routes multi-layer graphs through trace-to-Program
module lowering plus honest unsupported-graph reporting. Node and Bun each
instantiate one shared `TraceModuleCompiler`, so trace, support analysis, and
parameter packing use the same lowering rules instead of repeatedly threading
constructor tables through adapter-local calls. `Sequential.trace()` exposes that normalized JS/TS op
tape with parameter names, shapes, layouts, and scalar counts without copying raw
weights; `trace({ inputShape })` also propagates tensor shapes and rejects
impossible wiring, giving the traced compiler a stable input artifact.
The root `nn.trace`, `nn.compileSupport`, `nn.compilerSignatures`,
`nn.tensorProgramIr`, `nn.kernelPlan`, `nn.shapeConstraints`,
`nn.memoryLayout`, `nn.bufferLayout`, `nn.inputShape`, `nn.outputShape`,
`nn.parameterLayout`, `nn.canCompile`, `nn.compile`, and `nn.bindParameters`
helpers expose the same
audit-to-Program vocabulary for standalone and Sequential modules. Raw layer
arrays use that same product path through `nn.compile(layers)`; the standalone
`compile` namespace stays evidence-first for arrays because it has no
runtime-specific Program factory.
The shared JS/TS frontend core also owns tensor shape algebra, tensor factory
constructors, view helpers, parameter views, shape/layout-aware state
dictionaries, loss functions, tiny training helpers, optimizer config parsing,
optimizer snapshot validation, and `zeroGrad`, so Node and Bun cannot drift on
tensor, model, loss, training, or optimizer checkpoint semantics.
JS/TS declarations call the shared bind object `ModuleBindings` at the module
seam and `ProgramBindings` at the Program seam. `ModuleBindings` is branded
read-only evidence produced by module helpers, while `ProgramBindings` remains
the raw bind-object escape hatch; `TinyLinearWeights` remains a compatibility
alias for existing callers.
`compileSupport(...)` returns a frozen copy of the same normalized trace for the
support decision, including shape-specialized input/output evidence for compiled
module Programs plus structured diagnostics such as `missing-input-shape` for
unsupported paths. Explicit shape mismatches now report `shape-mismatch`
diagnostics through `compileSupport()` / `canCompile()` instead of escaping or
falling through legacy compatibility paths. Diagnostics carry a `stage`
(`trace`, `ir`, `kernelizer`, or `support`), and when trace/IR lowering succeeds
but native kernel lowering or layer support does not, unsupported support keeps
the frozen Tensor Program IR plus input/output shapes and scalar lengths
alongside the diagnostic, so callers can audit exactly which compiler stage
rejected the model before creating a native handle.
Supported module Programs also expose frozen `ir` and backend-neutral
`kernelPlan` evidence, the first small Trace -> Tensor Program IR -> native
module kernelizer seam; descriptor emission consumes the private native op plan
instead of bypassing it. IR values carry frozen `f32` dtype and scalar-byte
evidence next to shape, stride, and layout metadata, and scheduled KernelPlan
ops preserve that scalar evidence plus per-op `nativeDispatchCount` and
`nativeDescriptorCount` / `nativeKernels` / `nativeDescriptorSignatures` in
their own signatures. Fused KernelPlan entries also retain frozen per-fused-op
value-edge evidence, so public compiler evidence shows the internal dataflow
that a fused native descriptor represents. Parameterless module Programs keep an empty
parameter-layout signature as valid canonical evidence rather than treating it as
missing, and zero-dispatch elided shape chains expose empty descriptor-signature
arrays as intentional public evidence. This keeps
backend command evidence separate from descriptor emission evidence, so compiler
signatures are typed and schedule-aware instead of only shape-based. Kernel plans now include exact
`shapeConstraints` evidence for the compiled input/output ranks, shapes, and
scalar counts, a `memoryLayout` value table for typed scheduled storage with
step-input, persistent, scratch, and step-output storage classes, per-value
producer/consumer lifetime evidence, global and buffer-local scalar/byte
offsets, byte lengths, reuse-aware scratch size, and total arena size, and a
`bufferLayout` that names the step input/output and persistent weights/bias
slots with scalar counts and byte ranges, so the shape-specialized schedule,
value memory, and buffer contract are inspectable without reverse-engineering op metadata. They
also include `parameterLayout`, a public-safe map from stable parameter names to
their packed `weights` / `bias` buffer offsets, so callers can prove what
persistent state a compiled module expects before binding it.
The same compile-support and Program compile-evidence records carry a frozen
`compilerSignatures` record plus compatibility fields
(`irSignature`, `kernelPlanSignature`, `memoryLayoutSignature`,
`parameterLayoutSignature`, and `bufferLayoutSignature`) produced by the compiler modules, so callers can
compare compiler artifacts without rebuilding private compatibility keys from
nested records. `Program.moduleCompatibility(...)` treats that nested
`compilerSignatures` record as canonical, falls back to flat fields or nested
artifacts for older records, and reports the signature kind plus program/module
signature values when they differ. It also compares the public top-level
`inputShape` / `outputShape` evidence before signature checks, so shape
diagnostics stay readable even when scalar counts match.
Compiled module `Program` objects retain that same public-safe compiler evidence
through `compileEvidence()` and expose the frontend trace, normalized Tensor
Program IR, scheduled module plan, shape specialization, and named packed state
directly through `trace()`, `compilerSignatures()`, `tensorProgramIr()`,
`kernelPlan()`, `memoryLayout()`, `shapeConstraints()`, and `parameterLayout()`. The single `nn.Linear`
tiny-linear compatibility path now keeps the same normalized IR and KernelPlan
evidence, including flattened nested-`Sequential` parameter paths, while still
executing through the tiny-linear fast path; module compatibility preflights use
that evidence so same-shape modules with different parameter paths do not bind
silently. Shape-specialized Programs also use retained input-shape evidence as
the default for later `moduleCompatibility(...)`, `acceptsModule(...)`, and
`bindModule(...)`, so batched module Programs and shaped Embedding Programs can
bind their source modules without callers restating the compile shape. Runtime `Program` objects also expose
`bufferLayout()` derived from native requirements, so tiny-linear,
traced-module, and LLaMA-family callers can inspect the same primary
input/output/weights/bias buffer contract before creating or placing buffers.
`Program.inspect()` and `Program.capabilities()` now expose matching frozen
runtime diagnostics for compile-only/resource-probe Programs and incomplete
dispatch plans, plus a stable capability `signature` and
`matchesCapabilitySignature(...)` for cache/log comparison, using one shared
Node/Bun classifier over the existing native dispatch-plan evidence without
widening the C ABI.
The compiler also lowers bias-style broadcasted binary ops to explicit
`repeat` plus elementwise Program ops when a backend needs concrete buffer
spans, so `nn.linear` can remain ordinary `x @ w + b` tensor code without
materializing repeated bias tensors in the frontend.
When the native library is built with `-Duse-wgpu=true`, tiny-linear, direct
tiny-MLP, and traced module Programs can execute through WebGPU with host-bound
buffers; normal non-wgpu builds still expose WebGPU command-shape evidence
without pretending same-device resource execution exists for every MLP path.
The Node and Bun wrappers now expose the same small `Tensor`,
`nn`/`loss`/`optim`/`train` vocabulary, including shape-aware tensor arithmetic
with scalar-number tensor-like inputs, trailing-dimension broadcasting, matmul,
reductions, activations, views (`reshape`, `view`, `reshape_as`, `view_as`, `flatten`, `squeeze`, `unsqueeze`, `transpose`,
`T`, `broadcastTo`, `expand`, `expand_as`, `repeat`, `tile`, `flip`, `roll`, `select`, `narrow`, `slice`, `indexSelect`, `index_select`, `gather`, `take`, `argsort`, `sort`, `topk`, `scatterAdd`, `scatter_add`, `split`, `chunk`, `unbind`),
rank-aware indexing/debug helpers (`get`,
`set`, `toArray`, `tolist`, `numpy`, `nelement`, `ndimension`,
`elementSize`, `element_size`, `nbytes`, `stride`, `strides`, `isContiguous`,
`contiguous`, `storageOffset`, `storage_offset`, `isCpu`, `is_cpu`, `isFloatingPoint`,
`is_floating_point`, `isLeaf`, `is_leaf`, `inspect`, `allclose`, `equal`, `isclose`), PyTorch-style gradient aliases (`requires_grad`,
`requires_grad_`, `requiresGrad_`, `zero_grad`), tensor-context factories
(`new_empty`, `newEmpty`, `new_zeros`, `newZeros`, `new_ones`, `newOnes`, `new_full`, `newFull`), differentiable `clone`, `detach`, in-place
`detach_`, root grad-mode helpers (`isGradEnabled`, `setGradEnabled`,
`noGrad`, `inferenceMode`, `enableGrad`) for inference/eval scopes, tensor
factories (`zeros`, `ones`, `full`, `rand`, `randn`, `linspace`, `arange`,
  `scalar`, `parameter`, `param`), nested JS array tensor construction with
  rectangular-shape validation, shape-validated nested module parameters and
  Program/Session inputs/outputs,
  differentiable tensor joins (`cat`, `concat`, `concatenate`, `stack`, `vstack`, `hstack`), eager/autograd `einsum` with ellipsis, broadcast semantics, and literal-equation shape inference, plus compile-aware lazy parameter slots with `matmul`/`mm`/parameterized-add lowering evidence,
JSON serialization, PyTorch-like dim
reductions (`sum(dim)`, `prod(dim)`, `mean(dim)`,
`max(dim)`, eager tensor `min(dim)` / `minDim(dim)`, `variance(dim)`, `std(dim)`, `norm(dim, p)`, `cumsum(dim)`), batched `nn.linear`, `nn.embedding`, shape modules
(`nn.reshape`, `nn.view`, `nn.flatten`, `nn.squeeze`, `nn.unsqueeze`, `nn.transpose`, `nn.broadcastTo`, `nn.expand`, `nn.narrow`, `nn.select`, `nn.slice`), normalization
modules that compile with packed parameter-layout evidence and Program-shaped
parameter placement, cross-entropy, `nn.softmax`, `nn.logSoftmax`, PyTorch-style class aliases
for common modules (`nn.Linear`, `nn.Embedding`, `nn.Sequential`,
`nn.ReLU`, `nn.GELU`, `nn.SiLU`, `nn.Sigmoid`, `nn.Flatten`,
`nn.Identity`, `nn.Dropout`, `nn.Softmax`, `nn.LogSoftmax`, `nn.LayerNorm`, and
`nn.RMSNorm`), unary elementwise modules
(`nn.exp`, `nn.log`, `nn.neg`, `nn.recip`, `nn.abs`, `nn.sqrt`, `nn.square`,
`nn.sqr`, `nn.sign`, `nn.sgn`, `nn.step`), tensor-level `clamp`/`clip`, read-only `Sequential.length` / `at()` / iteration, subclassed `nn.Module` models with automatically discovered named layer and `nn.Parameter` fields, module `children()` / `modules()` and `namedChildren()` / `namedModules()` traversal, direct `getSubmodule` / `getParameter` / `getBuffer` lookups, `namedParameters`, module/namespace
`requiresGrad_` freezing, `stateDict` /
`loadStateDict`, `train()` / `eval()` mode, with shape/layout-aware
all-or-nothing entries, explicit
standalone/composable `compileSupport()`, SGD, Adam, and AdamW with optimizer
`stateDict` / `loadStateDict`, plus a JSON-safe `checkpoint` namespace and
Node/Bun root `save(snapshot, path)` / `load(path)` helpers for model+optimizer
snapshots with joint preflight before restore mutates either
target, for host-side model checks and small training loops.
`nn.identity`, `nn.reshape`, `nn.view`, `nn.flatten`, `nn.squeeze`, `nn.unsqueeze`, `nn.broadcastTo`, `nn.expand`, `nn.transpose`, bounded `nn.narrow`,
bounded `nn.select`, and bounded `nn.slice` now lower through native module
Programs for rank-1/rank-2 shapes where their output shape has a dense
caller-visible buffer contract. Consecutive reshape-equivalent shape ops are
kept in trace/IR evidence but coalesced into one native reshape descriptor by
the kernel plan, so ergonomic model glue does not pay a dispatch per view.
When that coalesced glue is a deterministic no-op before a real op, such as
eval `Dropout -> Identity -> Linear`, the kernel plan records it in
`elidedOps` and emits only the real native dispatch.
When metadata-only shape/Dropout glue is the whole compiled graph, the native
module Program emits zero dispatches and keeps the skipped shape chain in
`elidedOps`, so pure `Identity`, `View`, and eval/zero-probability `Dropout`
Programs do not run a synthetic reshape command.
Broadcast/expand and rank-2 transpose lower as materialized dense movement
ops, and batched feature-axis narrow/select/slice plus stepped rank-1/rank-2
slices use the same movement seam. Unsupported view-shaped traces
still keep frozen Tensor Program IR evidence, so callers can see what
normalized compiler input failed to lower.
The same native module Program compiler also lowers small unary elementwise
module chains through the Trace -> Tensor Program IR -> kernelPlan path, so
parameterless `Sequential` work can stay in the compiled Program lane instead
of bouncing back to eager host execution. Consecutive supported activations are
packed into one bounded activation-chain descriptor with frozen fused value-edge
evidence, matching the backend fused-elementwise path without growing the
normal `nn` surface. The KernelPlan also fuses supported
`Linear -> activation` pairs into one native linear descriptor with fused
kernel and value-edge evidence, and the native Program planner carries the
resulting dense `matmul + fused_elementwise` pair as one projection command, so
common MLPs reduce command depth without erasing the original IR ops. Metal exact command
execution now encodes dense elementwise and fused-elementwise sidecars as one
dispatch when the sidecar fits the kernel envelope, and the opt-in native wgpu
executor now does the same for dense batched matmul sidecars plus quantized
batched qmatmul add/mul sidecars.
Eager tensors report `dtype: "f32"` and `device: "cpu"` and expose honest
`to`/`cpu`/`float32` helpers; non-CPU movement is explicit through
`Tensor.place(program, kind)` and Program-owned buffers, not a hidden fallback.
JS/TS `nn.embedding` remains eager/autograd-capable and now also compiles
for explicit 1-D token windows with `inputShape: [tokens]`; compiled embedding
Sessions accept ordinary number arrays plus `Uint32Array`/`Int32Array` token
buffers at the JS edge while still binding f32 ABI buffers underneath. Without
that shape it rejects honestly because the Program output length is not static. The
package root now resolves to the TS-built CommonJS `dist/node.cjs` entrypoint, a `zgml/node`
adapter, a Bun-friendly `zgml/bun` adapter, and the emitted
`dist/public_api.d.cts` declaration surface, so Node/TS/Bun callers can import
`zgml` directly instead of reaching into `examples`.
A tiny host autograd tape covers the existing
Tensor arithmetic with broadcast-gradient reduction, matmul, reductions,
dim-aware reductions (`sumDim`, `prodDim`, `meanDim`, `maxDim`, eager `minDim`), dim-aware
`softmaxDim`/`logSoftmaxDim`, activations, embeddings, normalization modules,
MSE, and cross-entropy classification paths. Tensors also have the first
host-buffer placement bridge into native execution:
`toNativeBuffer()`, `place(program, kind)`, and
`Tensor.fromNativeBuffer(...)`, so compiled Program buffers can be filled from
and read back into shaped eager tensors without manually spelling every FFI
write/read; `program.bufferLayout()` exposes the exact primary slot map, while
`program.bufferSlotNames()` and `program.bufferSlot(nameOrKind)` expose direct
frozen slot lookup for the primary buffers those helpers target, and
`Tensor.place(...)` rejects mismatched primary slot lengths
and retained input/output edge shapes before allocating a Program buffer. `program.inputLen()` /
`program.outputLen()` and `program.inputByteLength()` / `program.outputByteLength()`
expose the compiled scalar and byte edge lengths directly,
`program.weightsLen()` / `program.weightsByteLength()` and `program.biasLen()` /
`program.biasByteLength()` expose persistent slot sizing, `program.parameterLen()` /
`program.parameterByteLength()` expose aggregate persistent parameter sizing,
`program.bufferSizing()` returns the same primary sizing evidence as a signed frozen summary,
`program.matchesBufferSizingSignature(...)` compares cached sizing signatures,
`program.requirements()` carries matching scalar/byte evidence for input, output, weights, bias, and
aggregate persistent parameters, and
`program.inputShape()` / `program.outputShape()` expose the compiled tensor edge
shapes directly, so callers do not have to scrape `compileEvidence()` just to
allocate eager I/O; `program.trace()`, `program.compilerSignatures()`,
`program.tensorProgramIr()`, `program.kernelPlan()`,
`program.memoryLayout()`, `program.shapeConstraints()`, and
`program.parameterLayout()` similarly expose the retained compiler ladder,
module schedule, value-storage layout, shape specialization, and named packed
module state layout when a Program was compiled from frontend module
code. `program.parameterNames()`, `program.parameterInfos()`, and
`program.parameterInfo(nameOrIndex)` expose the same named packed-state evidence
without making callers scan layout arrays before binding.
`module.placeParameters(program, options)` extends that bridge to
module state by using the Program's retained input shape when needed, preflighting
the module against the Program's retained compile evidence, packing eager
parameters through `bindParameters()`, checking the Program weights/bias slot
lengths, and filling Program-owned native weights/bias buffers for Session
binding; the root `nn.placeParameters(module, program, options)` helper exposes
the same compatibility-first behavior for generic module values. Program buffer resource factories receive that same
slot evidence (`kind`, `role`, element/byte lengths, and `layout`) so
host/device resource binding code can implement the inspected contract directly.
LLaMA-family Programs mirror this for cache residency through
`kvCacheLayout()` and slot-rich `createKvCache({ resource })` callbacks for
per-layer K/V buffers. Direct `Program.bind(...)` calls also preflight supplied
weights, bias, input, and output against that layout before entering native
binding. `ModuleBindings` produced by `module.bindParameters(...)` or
`module.placeParameters(...)` also carry binding-time module evidence with
frozen copied shape options, so direct `program.bind(module.bindParameters(...))`
or reuse of placed parameter buffers rejects architecture, shape, kernel-plan,
or packed-layout mismatches before Session creation. In TS, that
module-produced evidence is branded and preserved
through object spread, while hand-authored `{ weights, bias }` bindings remain
the raw FFI-shaped escape hatch and are checked by slot dtype, length, and
retained edge shape.
Host-backed direct binds accept eager `Tensor` values for weights,
bias, input, output, and explicit `Session.step(..., outputTensor)` overrides.
NativeBuffer-backed persistent weights/bias can also be mixed with ordinary
host tensor or typed-array step input/output, so moving model state into Program
buffers does not force every edge buffer into the same binding mode. Conversely,
when a direct bind needs buffer-backed output/input, host weights and bias are
placed into Session-owned Program buffers during bind instead of making callers
pre-place every persistent slot by hand. Generic compiled Sessions expose
`bufferLayout()`, `bufferSlotNames()`, `bufferSlot(nameOrKind)`, `inputShape()`,
`outputShape()`, `inputLen()`, `outputLen()`, `inputByteLength()`, and
`outputByteLength()`, weights/bias and aggregate parameter sizing helpers, and
signed `bufferSizing()` / `matchesBufferSizingSignature(...)` too, with `outputShape()`
reflecting a bound output Tensor's retained shape when present, so hot-loop callers
can inspect the bound tensor edge contract without keeping a Program reference;
`Session.stepContract()` returns that same frozen StepParams contract in one
place, including input/output shapes, scalar lengths, bound buffer kinds, and
whether missing input is valid, the default output ownership/readback path,
whether no-output execution advances state, whether output readback is
available, input/output byte lengths and slot names/roles for hot-buffer sizing,
`defaultReadbackRequired`, `defaultAllocationFree`, `defaultHotPath`,
default output effect/ownership/return ownership, plus a stable `signature` for
comparing hot-loop contracts;
`Session.matchesStepContractSignature(signature)` compares that cached contract
directly;
`Session.acceptsStepParams(params)` reuses the same
input/output validation as `execute(...)` without executing or bumping call
counters; `Session.canExecuteStepParams(params)` is the `canExecute()`-shaped
alias for the same non-executing accepted check.
`Session.acceptsAllocationFreeStepParams(params)` is the same guard
with the additional requirement that the accepted call shape avoids runtime
output allocation for already-normalized inputs,
`Session.acceptsRuntimeOutputAllocationFreeStepParams(params)` names that exact
predicate while keeping the older compatibility alias available,
`Session.acceptsNoReadbackStepParams(params)` similarly requires an
accepted call shape with no output readback,
`Session.acceptsReadbackFreeStepParams(params)` names that same positive
evidence predicate, and
`Session.acceptsHotStepParams(params)` combines the allocation-free and
no-readback guards for hot loops.
`Session.matchesStepParamsSignature(params, signature)` compares a preflighted
call shape against a cached `stepParamsSignature`, and
`Session.matchesStepParamsCompatibility(params, compatibility)` compares
against a frozen preflight record; these preflight helpers do not execute or
bump call counters.
`Session.preflightStepParams(params)` is the friendly evidence verb and
`Session.stepParamsCompatibility(params)` is the compatibility alias. Both
accept unknown boundary values and return the frozen accepted flag, contract
typed `canExecute` alias, typed accepted/rejected status, contract kind,
contract signature, stable
StepParams call-shape signature, contract
position (`null` for generic Sessions), per-call `stateEffect` and `allocationFree` evidence,
typed `inputSource` / `outputTarget` / `outputEffect` routing evidence, planned
input/output element types, shapes, shape signatures, element lengths, byte lengths, `inputOwnership`, `outputOwnership`, `outputReturnOwnership`,
`readsInput`, `writesOutput`, `readbackRequired`, `readbackFree`,
`runtimeOutputAllocationFree`, top-level `hotPath`, `hotPathStatus`,
`hotPathBlockers`, `rejectionCode`, and typed rejection diagnostics with structured
expected/actual length, shape, and context
fields when available;
output Tensor bindings and per-step output overrides may use any same-element
shape for readback while inputs still use the compiled input shape
beside the Session. NativeBuffer-backed persistent state can be refreshed after
bind with `Session.uploadParameters()`, `Session.uploadParameter(index)`, or
`Session.uploadParameterByName(name)`, or `Session.uploadParameterRange(first,
len)`, making weight updates explicit
without rebuilding the Session; named JS/TS refreshes use retained
`Session.parameterLayout()` evidence, plus `Session.parameterNames()`,
`Session.parameterInfos()`, and `Session.parameterInfo(nameOrIndex)` lookup
helpers, to refresh the containing persistent weights/bias binding. They also
expose `SessionStepParams` through `execute({ input, output })`,
`stepInto(output, input)`, `executeInto(output, { input })`,
`executeTensor({ input, shape | output })`, and `stepTensor(input, { shape | output })`,
which wraps the compiled output as a shaped eager `Tensor` using an explicit
shape, a supplied or Session-bound output Tensor's shape, or the Program's
retained output-shape evidence by default, plus `readOutputInto(...)` for
caller-owned readback from a Session-bound host or native output buffer and
`readOutputTensor({ shape | output })` when the caller wants shaped eager Tensor
readback into fresh storage or a caller-provided Tensor/`Float32Array` carrier;
readback targets are length and f32-alignment checked before native reads.
`program.bindModule(module, options)` is the concise
Program-centered form: it places module parameters, binds the Session, and owns
the temporary parameter buffers with that Session. `program.moduleCompatibility(...)`
and `program.acceptsModule(...)` expose the same compile-evidence preflight used
by direct placement and module-aware binding, so incompatible architecture,
shape, kernel plan, elided no-op plan records, or packed parameter layout fails
before native parameter buffers are created.
`loss.mse(...).backward()`,
`loss.crossEntropy(...).backward()`, and `train.step(opt, { loss })` fill the
same durable parameter gradients consumed by optimizers. Typed arrays still work
at every edge. The single-`nn.linear` case compiles into the
existing native Program/Session tiny-linear path instead of exposing C
descriptors directly.

```ts
const model = nn.sequential([nn.linear(2, 3), nn.relu(), nn.linear(3, 2)]);
const logits = model.forward(tensor([0.2, 0.8], [2]));
const objective = loss.crossEntropy(logits, [1]);
train.step(optim.adamW(model, { lr: 1e-3 }), { loss: objective });
```

The native side is intentionally lower-level. It owns executable handles,
buffers, kernels, backend dispatch, and ABI records. Higher-level model,
optimizer, loss, training, and package policy belong in TypeScript and reach
native speed by compiling or binding through `Program`/`Session`, not by
growing a second Zig-shaped product frontend.

The graph IR is deliberately small:

- `src/ts/**` is the product source of truth for tensor, `nn`, `loss`, `optim`,
  `train`, state, compile-support, and package subpath behavior.
- `tsdown` emits the runnable package artifacts, declarations, and host
  subpaths from the TS-derived package policy rather than a second hand-written
  entry graph.
- `src/op.zig`, native buffers, Programs, Sessions, kernels, backend hooks, and
  ABI structs are the execution substrate, not a second frontend to keep in
  sync.
- The Zig root exposes `native_substrate_manifest` with the same boundary:
  TypeScript is the product language, `tsdown` fans out package artifacts, and
  Zig aligns only through Program/Session/ABI contracts.
- Higher-level TS tensor and module conveniences lower to primitive/composite
  IR ops or compose them; new primitives should be rare.
- Optimizer and training policy stays host/product-level until there is a
  deliberate compiled optimizer path.
- Fusion and graph rewrites happen behind compile/run paths as optimization
  passes.
- LLM users go through model-source helpers such as `probeModel` / `loadModel`,
  then explicit model/program/session/step handles, while backend adapters,
  model-file loaders, and benchmark internals stay out of the stable surface.

This keeps the core easy to reason about: the graph stays small, backward rules stay attached to a compact primitive set, user-facing APIs are ergonomic, and performance work happens in fusion passes without bloating the IR.

Tensor runtime state is split into two broad parts:

- hot shape/view metadata (`ne`, `strides`, `storage_offset`, `op`, `data`)
- colder bookkeeping (`role`, data/index ownership, auxiliary index payloads)

That split keeps the common execution path simple while still making aliasing and cleanup rules explicit.

Graph construction uses a visited set while walking parent links, so shared subgraphs are recorded once instead of being rediscovered via repeated linear scans.

### Primitives

The primitive IR is organized into a few stable categories:

- structural/view ops
- elementwise unary and binary ops
- reductions and repeat/broadcast helpers
- indexing and scatter/gather ops
- matrix multiplication

For the exact current primitive set, see `src/op.zig`.

### Frontend Conveniences

The frontend keeps PyTorch-like names where they make model code simpler.
Some conveniences lower to existing primitive/composite IR ops, while others
compose multiple primitive ops:

`sub`, `div`, `mean`, `prod`, `variance`/`var`, `std`, `norm`, `cumsum`, `sumDim`, `prodDim`, `meanDim`, `maxDim`, eager `minDim`, `isclose`, `dot`, `trace`, `diagonal`, `bmm`,
`softmaxDim`, `logSoftmax`, `logSoftmaxDim`, `logsumexp`, `logSumExp`, `layerNorm`, `rmsNorm`,
`normalize`, `oneHot`, and `embedding`

## Building

```bash
zig build test            # run Zig/native tests
zig build check           # run local validation; full source checkouts also run repo-only smokes and benchmark-shape checks
npm test                  # run public JS/TS typecheck plus Node/Bun package and training smokes
npm run test:adapters     # typecheck, then run the one-build Node/Bun adapter gate
npm run smoke:adapters    # build once, then run Node and Bun smokes against the same dist artifact
npm run smoke:training    # run the public JS training/checkpoint smoke
npm run smoke:training:bun # run the public Bun training/checkpoint smoke
npm run smoke:node        # run Node package, training, and Program/Session smokes
npm run smoke:bun         # run Bun package, training, Program/Session, and bundle smokes
npm run check:goal-scorecard # verify Program/Session substrate and PyTorch-like surface evidence
npm run bench             # verify checked benchmark baseline artifacts
npm run bench:status      # summarize source-checkout baseline/latest ggml artifact evidence
npm run bench:ggml        # run the local ggml/llama.cpp artifact gate with baseline regression protection
zig build bench-frontier  # run decision-grade local benchmarks
zig build -Duse-blas      # enable BLAS for matmul
```

The npm source package keeps the native build inputs but excludes heavyweight
repo-only examples, benchmark artifacts, and planning docs. After installing
its dev dependencies, `npm run build:native`, `npm test`,
`npm run test:adapters`, `zig build check`, and
`zig build wgpu-check -Duse-wgpu=true` validate the included source subset.
When `zgml` is installed as a dependency, the packed-install gate proves the
shipped no-build Node and Bun smoke scripts against the emitted `dist`
artifacts. Full repository checkouts additionally exercise the excluded
long-form smoke and benchmark gates.

`npm run check:goal-scorecard` is the lightweight progress guard for the current
library goal. It requires a passing no-fallback Program/Session substrate gate
with a latest-vs-checked-baseline delta report for the selected native lanes,
and checks that the public type smokes still cover the PyTorch-like surface:
`goal progress: Program/Session substrate=77% floor=65%; PyTorch-like surface=96% floor=60%`.
Those numbers are deliberately conservative: q8 prompt execution still needs a
real tiled row-chain throughput kernel, while the PyTorch-like surface now has
runtime and type evidence for the core replacement loop and PyTorch-like
eager/autograd `einsum` semantics with typed output shapes plus lazy
parameterized `matmul`/`mm`/add compile evidence, direct `lazyGraph.compile()`
native Program construction, and benchmarked allocation-free lazy
Linear+GELU, `matmul -> add -> relu/gelu`, Conv2d+ReLU, MLP, reduced MLP, classifier log-softmax,
native `diagonal`,
rank-1/rank-2 repeat/tile Program lowering,
direct backend `min(dim)`, eval-mode `BatchNorm1d`, classifier softmax-reduction, transformer FFN, normalized transformer
classifier, and token-head Session paths rather than only API exports.
Tensor, `nn`, `optim`, `train`, `data`, loss modules, state dicts,
manual `backward`/`step` loops, train helper verbs (`backward`, `lossStep`,
`clipGradNorm`, `clipGradValue`), checkpoints, compile support, and compiled
`Program`/`Session` binding, plus optimizer parameter groups and group config
snapshots, StepLR/ExponentialLR schedulers, module `train`/`eval` mode
cascades, traversal and parameter metadata, deterministic eval Dropout compile,
seeded random, integer-random, and permutation factories, tensor reshape/view/transpose, `where`/`maskedFill`/`masked_fill`, `indexSelect`/`index_select`/`gather`/`take`/`argsort`/`sort`/`topk`/`scatterAdd`/`scatter_add`, `cat`/`concat`/`concatenate`/`stack`/`vstack`/`hstack`,
`tolist`/`numpy`/`allclose`/`equal`, PyTorch-style data aliases such as
`tensor_dataset`, `batch_size`, and `drop_last`,
JSON-safe checkpoint round-trips through `checkpoint.toJSON` / `fromJSON` /
`load`, PyTorch-style class constructors for modules/optimizers/losses,
class-style and subclassed train-then-compile MLP examples with module traversal,
field-owned `class Model extends nn.Module` layers that automatically feed
parameters, state dicts, mode cascades, and simple Program compilation, raw
field-owned `nn.Parameter` entries that automatically feed eager autograd,
optimizers, freezing, and state dicts, explicit `addModule` / `registerModule`
and `registerParameter` / `register_parameter` subclass registration for dynamic
model construction, `ModuleList` / `moduleList`, `ModuleDict` / `moduleDict`,
`ParameterList` / `parameterList`, and `ParameterDict` / `parameterDict`
containers for dynamic traversal, state, and optimizer ownership,
`nn.Buffer` / `nn.buffer` plus `registerBuffer` / `namedBuffers` for
non-trainable persistent module state,
PyTorch-style module `apply(fn)` traversal on models, containers, and raw layer
lists via `nn.apply`, plus direct object lookup with `getSubmodule`,
`getParameter`, and `getBuffer` on modules and through the `nn` namespace;
module and namespace `stateDict` / `loadStateDict` include persistent buffers
with the same strict shape/layout checks, and checkpoint model snapshots carry
the same persistent buffer state,
custom `nn.module({ graph })` and
subclassed classifier shells with automatic child parameter state and top-level
compile delegation, cross-entropy classifier losses,
grad-mode scopes, `clone`/`detach` tensor lifetime helpers, and runnable classifier
train-then-compile smokes. It also guards compile-time shape safety for eager tensor ops,
dataloaders, modules, compiled Sessions, and negative `@ts-expect-error` cases
where incompatible shapes must not narrow to happy-path types. The Node and Bun manual-loop smokes prove explicit
autograd, frozen parameters, optimizer stepping, and train-step evidence before
compiling. The MLP and classifier training smokes prove the larger workflow:
train a frontend model, bind the trained module into a compiled
Program, and verify compiled output against eager output. The classifier smoke
uses `CrossEntropyLoss`/`classTargets` and verifies the learned class logits
before and after checkpoint restore. The MLP examples use subclassed `nn.Module`
models with named assigned layers that compile as a simple sequential graph; the
classifier examples use subclassed `nn.Module` models backed by explicit
compiled graphs. They also restore the checkpoint into a
fresh module/optimizer pair and compile that restored module, which keeps
state-dict portability tied to the fast Program/Session path.

For stricter ggml/llama.cpp parity gates, including required-parity and
baseline-regression modes, see `docs/perf-targets.md`.

## Contributing

When adding a new tensor feature:

1. Check whether it can be expressed as a composition of existing primitives
2. If yes, add the product API in `src/ts/**` and let compile/fusion optimize
   it later
3. Add native/Zig work only when the feature needs a kernel, buffer/runtime
   primitive, ABI entry, or backend lowering
4. Only add a new primitive op when it materially simplifies the IR or unlocks
   behavior composition cannot express cleanly
