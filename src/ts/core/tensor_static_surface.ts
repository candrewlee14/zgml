"use strict";

type TensorOptions = Record<string, unknown>;

type TensorStaticSortResult<TTensor> = Readonly<{
  values: TTensor;
  indices: TTensor;
}>;

export type TensorStaticSurfaceOptions<TTensor> = {
  fromJSON: (value: unknown) => TTensor;
  fromNativeBuffer: (buffer: unknown, shape?: unknown, options?: TensorOptions) => TTensor;
  tensor: (data: unknown, shape?: unknown, options?: TensorOptions) => TTensor;
  parameter: (data: unknown, shape?: unknown, options?: TensorOptions) => TTensor;
  param: (data: unknown, shape?: unknown, options?: TensorOptions) => TTensor;
  scalar: (value: number, options?: TensorOptions) => TTensor;
  empty: (shape: unknown, options?: TensorOptions) => TTensor;
  emptyLike: (input: unknown, options?: TensorOptions) => TTensor;
  empty_like: (input: unknown, options?: TensorOptions) => TTensor;
  zeros: (shape: unknown, options?: TensorOptions) => TTensor;
  zerosLike: (input: unknown, options?: TensorOptions) => TTensor;
  zeros_like: (input: unknown, options?: TensorOptions) => TTensor;
  ones: (shape: unknown, options?: TensorOptions) => TTensor;
  onesLike: (input: unknown, options?: TensorOptions) => TTensor;
  ones_like: (input: unknown, options?: TensorOptions) => TTensor;
  eye: (size: number, options?: TensorOptions) => TTensor;
  full: (shape: unknown, value: number, options?: TensorOptions) => TTensor;
  fullLike: (input: unknown, value: number, options?: TensorOptions) => TTensor;
  full_like: (input: unknown, value: number, options?: TensorOptions) => TTensor;
  rand: (shape: unknown, options?: TensorOptions) => TTensor;
  randLike: (input: unknown, options?: TensorOptions) => TTensor;
  rand_like: (input: unknown, options?: TensorOptions) => TTensor;
  randn: (shape: unknown, options?: TensorOptions) => TTensor;
  randnLike: (input: unknown, options?: TensorOptions) => TTensor;
  randn_like: (input: unknown, options?: TensorOptions) => TTensor;
  randInt: (first: number, second: unknown, third?: unknown, fourth?: TensorOptions) => TTensor;
  randint: (first: number, second: unknown, third?: unknown, fourth?: TensorOptions) => TTensor;
  randPerm: (size: number, options?: TensorOptions) => TTensor;
  randperm: (size: number, options?: TensorOptions) => TTensor;
  manualSeed: (seed: number) => number;
  manual_seed: (seed: number) => number;
  initialSeed: () => number | null;
  initial_seed: () => number | null;
  seededRng: (seed: number) => () => number;
  linspace: (first: unknown, second: number, third: number, fourth?: TensorOptions) => TTensor;
  arange: (start: number, endOrOptions?: unknown, stepOrOptions?: unknown, maybeOptions?: TensorOptions) => TTensor;
  allclose: (actual: unknown, expected: unknown, options?: TensorOptions) => boolean;
  equal: (actual: unknown, expected: unknown) => boolean;
  hasShape: (input: unknown, shape: unknown) => boolean;
  requireShape: (input: unknown, shape: unknown) => TTensor;
  to: (input: unknown, target?: unknown, options?: TensorOptions) => TTensor;
  cpu: (input: unknown, options?: TensorOptions) => TTensor;
  float: (input: unknown, options?: TensorOptions) => TTensor;
  float32: (input: unknown, options?: TensorOptions) => TTensor;
  typeAs: (input: unknown, other: unknown, options?: TensorOptions) => TTensor;
  type_as: (input: unknown, other: unknown, options?: TensorOptions) => TTensor;
  clone: (input: unknown) => TTensor;
  detach: (input: unknown) => TTensor;
  reshape: (input: unknown, shape: unknown) => TTensor;
  view: (input: unknown, shape: unknown) => TTensor;
  broadcastTo: (input: unknown, shape: unknown) => TTensor;
  expand: (input: unknown, shape: unknown) => TTensor;
  repeat: (input: unknown, repeats: unknown, ...moreRepeats: unknown[]) => TTensor;
  tile: (input: unknown, repeats: unknown, ...moreRepeats: unknown[]) => TTensor;
  flatten: (input: unknown, startDim?: number, endDim?: number) => TTensor;
  squeeze: (input: unknown, dim?: number | null) => TTensor;
  unsqueeze: (input: unknown, dim: number) => TTensor;
  transpose: (input: unknown, dim0?: number, dim1?: number) => TTensor;
  permute: (input: unknown, dims: readonly number[]) => TTensor;
  flip: (input: unknown, dims: readonly number[]) => TTensor;
  roll: (input: unknown, shifts: number | readonly number[], dims?: number | readonly number[] | null) => TTensor;
  select: (input: unknown, dim: number, index: number) => TTensor;
  narrow: (input: unknown, dim: number, start: number, length: number) => TTensor;
  slice: (input: unknown, dim: number, start?: number | null, end?: number | null, step?: number) => TTensor;
  indexSelect: (input: unknown, dim: number, indices: unknown) => TTensor;
  index_select: (input: unknown, dim: number, indices: unknown) => TTensor;
  gather: (input: unknown, dim: number, index: unknown) => TTensor;
  take: (input: unknown, index: unknown) => TTensor;
  argsort: (input: unknown, dim?: number, descending?: boolean) => TTensor;
  sort: (input: unknown, dim?: number, descending?: boolean) => TensorStaticSortResult<TTensor>;
  topk: (input: unknown, k: number, dim?: number, largest?: boolean, sorted?: boolean) => TensorStaticSortResult<TTensor>;
  scatterAdd: (input: unknown, dim: number, index: unknown, src: unknown) => TTensor;
  scatter_add: (input: unknown, dim: number, index: unknown, src: unknown) => TTensor;
  split: (input: unknown, splitSizeOrSections: number | readonly number[], dim?: number) => readonly TTensor[];
  chunk: (input: unknown, chunks: number, dim?: number) => readonly TTensor[];
  unbind: (input: unknown, dim?: number) => readonly TTensor[];
  add: (input: unknown, other: unknown) => TTensor;
  sub: (input: unknown, other: unknown) => TTensor;
  mul: (input: unknown, other: unknown) => TTensor;
  div: (input: unknown, other: unknown) => TTensor;
  eq: (input: unknown, other: unknown) => TTensor;
  ne: (input: unknown, other: unknown) => TTensor;
  lt: (input: unknown, other: unknown) => TTensor;
  le: (input: unknown, other: unknown) => TTensor;
  gt: (input: unknown, other: unknown) => TTensor;
  ge: (input: unknown, other: unknown) => TTensor;
  isclose: (input: unknown, other: unknown, options?: unknown) => TTensor;
  pow: (input: unknown, exponent: number) => TTensor;
  neg: (input: unknown) => TTensor;
  negative: (input: unknown) => TTensor;
  exp: (input: unknown) => TTensor;
  expm1: (input: unknown) => TTensor;
  log: (input: unknown) => TTensor;
  log1p: (input: unknown) => TTensor;
  sqr: (input: unknown) => TTensor;
  square: (input: unknown) => TTensor;
  recip: (input: unknown) => TTensor;
  reciprocal: (input: unknown) => TTensor;
  abs: (input: unknown) => TTensor;
  sgn: (input: unknown) => TTensor;
  sign: (input: unknown) => TTensor;
  step: (input: unknown) => TTensor;
  isnan: (input: unknown) => TTensor;
  isinf: (input: unknown) => TTensor;
  isfinite: (input: unknown) => TTensor;
  floor: (input: unknown) => TTensor;
  ceil: (input: unknown) => TTensor;
  round: (input: unknown) => TTensor;
  trunc: (input: unknown) => TTensor;
  sqrt: (input: unknown) => TTensor;
  rsqrt: (input: unknown) => TTensor;
  relu: (input: unknown) => TTensor;
  gelu: (input: unknown) => TTensor;
  silu: (input: unknown) => TTensor;
  sigmoid: (input: unknown) => TTensor;
  tanh: (input: unknown) => TTensor;
  sin: (input: unknown) => TTensor;
  cos: (input: unknown) => TTensor;
  tan: (input: unknown) => TTensor;
  maximum: (input: unknown, other: unknown) => TTensor;
  minimum: (input: unknown, other: unknown) => TTensor;
  where: (condition: unknown, input: unknown, other: unknown) => TTensor;
  maskedFill: (input: unknown, mask: unknown, value: unknown) => TTensor;
  masked_fill: (input: unknown, mask: unknown, value: unknown) => TTensor;
  sum: (input: unknown, dim?: number) => TTensor;
  prod: (input: unknown, dim?: number) => TTensor;
  cumsum: (input: unknown, dim?: number) => TTensor;
  mean: (input: unknown, dim?: number) => TTensor;
  max: (input: unknown, dim?: number) => TTensor;
  min: (input: unknown, dim?: number) => TTensor;
  any: (input: unknown, dim?: number) => TTensor;
  all: (input: unknown, dim?: number) => TTensor;
  argmax: (input: unknown, dim?: number) => TTensor;
  argmin: (input: unknown, dim?: number) => TTensor;
  variance: (input: unknown, dim?: number, correction?: number) => TTensor;
  var: (input: unknown, dim?: number, correction?: number) => TTensor;
  std: (input: unknown, dim?: number, correction?: number) => TTensor;
  norm: (input: unknown, dim?: number, p?: number) => TTensor;
  softmax: (input: unknown, dim?: number) => TTensor;
  softmax_dim: (input: unknown, dim?: number) => TTensor;
  softmaxDim: (input: unknown, dim?: number) => TTensor;
  logSoftmax: (input: unknown, dim?: number) => TTensor;
  log_softmax: (input: unknown, dim?: number) => TTensor;
  log_softmax_dim: (input: unknown, dim?: number) => TTensor;
  logSoftmaxDim: (input: unknown, dim?: number) => TTensor;
  logsumexp: (input: unknown, dim?: number) => TTensor;
  logSumExp: (input: unknown, dim?: number) => TTensor;
  clamp: (input: unknown, min?: number | null, max?: number | null) => TTensor;
  clip: (input: unknown, min?: number | null, max?: number | null) => TTensor;
  matmul: (input: unknown, other: unknown, otherShape?: readonly number[]) => TTensor;
  mm: (input: unknown, other: unknown, otherShape?: readonly number[]) => TTensor;
  dot: (input: unknown, other: unknown, otherShape?: readonly number[]) => TTensor;
  trace: (input: unknown) => TTensor;
  diagonal: (input: unknown) => TTensor;
  bmm: (input: unknown, other: unknown, otherShape?: readonly number[]) => TTensor;
  cat: (tensors: readonly TTensor[], dim?: number) => TTensor;
  concat: (tensors: readonly TTensor[], dim?: number) => TTensor;
  concatenate: (tensors: readonly TTensor[], dim?: number) => TTensor;
  stack: (tensors: readonly TTensor[], dim?: number) => TTensor;
  vstack: (tensors: readonly TTensor[]) => TTensor;
  hstack: (tensors: readonly TTensor[]) => TTensor;
  einsum: (equation: string, tensors: readonly TTensor[] | TTensor, ...moreTensors: readonly TTensor[]) => TTensor;
};

export type TensorStaticSurfaceCore<TTensor> = {
  fromJSON: (value: unknown) => TTensor;
};

export type TensorStaticSurfaceFacade<TTensor> = Pick<
  TensorStaticSurfaceOptions<TTensor>,
  "fromNativeBuffer" | "tensor" | "parameter" | "param"
>;

export type TensorStaticSurfaceRootOps<TTensor> = Omit<
  TensorStaticSurfaceOptions<TTensor>,
  "fromJSON" | "fromNativeBuffer" | "tensor" | "parameter" | "param"
>;

export type TensorStaticSurfaceFromFacadeOptions<TTensor> = Readonly<{
  tensorCoreHelpers: TensorStaticSurfaceCore<TTensor>;
  tensorFacade: TensorStaticSurfaceFacade<TTensor>;
  rootOps: TensorStaticSurfaceRootOps<TTensor>;
}>;

export function createTensorStaticSurface<TTensor>(options: TensorStaticSurfaceOptions<TTensor>) {
  return Object.freeze({
    fromJSON: (value: unknown) => options.fromJSON(value),
    fromNativeBuffer: (buffer: unknown, shape?: unknown, tensorOptions: TensorOptions = {}) => (
      options.fromNativeBuffer(buffer, shape, tensorOptions)
    ),
    tensor: (data: unknown, shape?: unknown, tensorOptions?: TensorOptions) => options.tensor(data, shape, tensorOptions),
    parameter: (data: unknown, shape?: unknown, tensorOptions?: TensorOptions) => options.parameter(data, shape, tensorOptions),
    param: (data: unknown, shape?: unknown, tensorOptions?: TensorOptions) => options.param(data, shape, tensorOptions),
    scalar: (value: number, tensorOptions: TensorOptions = {}) => options.scalar(value, tensorOptions),
    empty: (shape: unknown, tensorOptions: TensorOptions = {}) => options.empty(shape, tensorOptions),
    emptyLike: (input: unknown, tensorOptions: TensorOptions = {}) => options.emptyLike(input, tensorOptions),
    empty_like: (input: unknown, tensorOptions: TensorOptions = {}) => options.empty_like(input, tensorOptions),
    zeros: (shape: unknown, tensorOptions: TensorOptions = {}) => options.zeros(shape, tensorOptions),
    zerosLike: (input: unknown, tensorOptions: TensorOptions = {}) => options.zerosLike(input, tensorOptions),
    zeros_like: (input: unknown, tensorOptions: TensorOptions = {}) => options.zeros_like(input, tensorOptions),
    ones: (shape: unknown, tensorOptions: TensorOptions = {}) => options.ones(shape, tensorOptions),
    onesLike: (input: unknown, tensorOptions: TensorOptions = {}) => options.onesLike(input, tensorOptions),
    ones_like: (input: unknown, tensorOptions: TensorOptions = {}) => options.ones_like(input, tensorOptions),
    eye: (size: number, tensorOptions: TensorOptions = {}) => options.eye(size, tensorOptions),
    full: (shape: unknown, value: number, tensorOptions: TensorOptions = {}) => options.full(shape, value, tensorOptions),
    fullLike: (input: unknown, value: number, tensorOptions: TensorOptions = {}) => options.fullLike(input, value, tensorOptions),
    full_like: (input: unknown, value: number, tensorOptions: TensorOptions = {}) => options.full_like(input, value, tensorOptions),
    rand: (shape: unknown, tensorOptions: TensorOptions = {}) => options.rand(shape, tensorOptions),
    randLike: (input: unknown, tensorOptions: TensorOptions = {}) => options.randLike(input, tensorOptions),
    rand_like: (input: unknown, tensorOptions: TensorOptions = {}) => options.rand_like(input, tensorOptions),
    randn: (shape: unknown, tensorOptions: TensorOptions = {}) => options.randn(shape, tensorOptions),
    randnLike: (input: unknown, tensorOptions: TensorOptions = {}) => options.randnLike(input, tensorOptions),
    randn_like: (input: unknown, tensorOptions: TensorOptions = {}) => options.randn_like(input, tensorOptions),
    randInt: (first: number, second: unknown, third?: unknown, fourth?: TensorOptions) => options.randInt(first, second, third, fourth),
    randint: (first: number, second: unknown, third?: unknown, fourth?: TensorOptions) => options.randint(first, second, third, fourth),
    randPerm: (size: number, tensorOptions: TensorOptions = {}) => options.randPerm(size, tensorOptions),
    randperm: (size: number, tensorOptions: TensorOptions = {}) => options.randperm(size, tensorOptions),
    manualSeed: (seed: number) => options.manualSeed(seed),
    manual_seed: (seed: number) => options.manual_seed(seed),
    initialSeed: () => options.initialSeed(),
    initial_seed: () => options.initial_seed(),
    seededRng: (seed: number) => options.seededRng(seed),
    linspace: (first: unknown, second: number, third: number, fourth?: TensorOptions) => options.linspace(first, second, third, fourth),
    arange: (start: number, endOrOptions?: unknown, stepOrOptions?: unknown, maybeOptions?: TensorOptions) => (
      options.arange(start, endOrOptions, stepOrOptions, maybeOptions)
    ),
    allclose: (actual: unknown, expected: unknown, tensorOptions?: TensorOptions) => options.allclose(actual, expected, tensorOptions),
    equal: (actual: unknown, expected: unknown) => options.equal(actual, expected),
    hasShape: (input: unknown, shape: unknown) => options.hasShape(input, shape),
    requireShape: (input: unknown, shape: unknown) => options.requireShape(input, shape),
    to: (input: unknown, target?: unknown, tensorOptions?: TensorOptions) => options.to(input, target, tensorOptions),
    cpu: (input: unknown, tensorOptions?: TensorOptions) => options.cpu(input, tensorOptions),
    float: (input: unknown, tensorOptions?: TensorOptions) => options.float(input, tensorOptions),
    float32: (input: unknown, tensorOptions?: TensorOptions) => options.float32(input, tensorOptions),
    typeAs: (input: unknown, other: unknown, tensorOptions?: TensorOptions) => options.typeAs(input, other, tensorOptions),
    type_as: (input: unknown, other: unknown, tensorOptions?: TensorOptions) => options.type_as(input, other, tensorOptions),
    clone: (input: unknown) => options.clone(input),
    detach: (input: unknown) => options.detach(input),
    reshape: (input: unknown, shape: unknown) => options.reshape(input, shape),
    view: (input: unknown, shape: unknown) => options.view(input, shape),
    broadcastTo: (input: unknown, shape: unknown) => options.broadcastTo(input, shape),
    expand: (input: unknown, shape: unknown) => options.expand(input, shape),
    repeat: (input: unknown, repeats: unknown, ...moreRepeats: unknown[]) => (
      options.repeat(input, moreRepeats.length === 0 ? repeats : [repeats, ...moreRepeats])
    ),
    tile: (input: unknown, repeats: unknown, ...moreRepeats: unknown[]) => (
      options.tile(input, moreRepeats.length === 0 ? repeats : [repeats, ...moreRepeats])
    ),
    flatten: (input: unknown, startDim?: number, endDim?: number) => options.flatten(input, startDim, endDim),
    squeeze: (input: unknown, dim?: number | null) => options.squeeze(input, dim),
    unsqueeze: (input: unknown, dim: number) => options.unsqueeze(input, dim),
    transpose: (input: unknown, dim0?: number, dim1?: number) => options.transpose(input, dim0, dim1),
    permute: (input: unknown, dims: readonly number[]) => options.permute(input, dims),
    flip: (input: unknown, dims: readonly number[]) => options.flip(input, dims),
    roll: (input: unknown, shifts: number | readonly number[], dims?: number | readonly number[] | null) => options.roll(input, shifts, dims),
    select: (input: unknown, dim: number, index: number) => options.select(input, dim, index),
    narrow: (input: unknown, dim: number, start: number, length: number) => options.narrow(input, dim, start, length),
    slice: (input: unknown, dim: number, start?: number | null, end?: number | null, step?: number) => (
      options.slice(input, dim, start, end, step)
    ),
    indexSelect: (input: unknown, dim: number, indices: unknown) => options.indexSelect(input, dim, indices),
    index_select: (input: unknown, dim: number, indices: unknown) => options.index_select(input, dim, indices),
    gather: (input: unknown, dim: number, index: unknown) => options.gather(input, dim, index),
    take: (input: unknown, index: unknown) => options.take(input, index),
    argsort: (input: unknown, dim = -1, descending = false) => options.argsort(input, dim, descending),
    sort: (input: unknown, dim = -1, descending = false) => options.sort(input, dim, descending),
    topk: (input: unknown, k: number, dim = -1, largest = true, sorted = true) => options.topk(input, k, dim, largest, sorted),
    scatterAdd: (input: unknown, dim: number, index: unknown, src: unknown) => options.scatterAdd(input, dim, index, src),
    scatter_add: (input: unknown, dim: number, index: unknown, src: unknown) => options.scatter_add(input, dim, index, src),
    split: (input: unknown, splitSizeOrSections: number | readonly number[], dim = 0) => options.split(input, splitSizeOrSections, dim),
    chunk: (input: unknown, chunks: number, dim = 0) => options.chunk(input, chunks, dim),
    unbind: (input: unknown, dim = 0) => options.unbind(input, dim),
    add: (input: unknown, other: unknown) => options.add(input, other),
    sub: (input: unknown, other: unknown) => options.sub(input, other),
    mul: (input: unknown, other: unknown) => options.mul(input, other),
    div: (input: unknown, other: unknown) => options.div(input, other),
    eq: (input: unknown, other: unknown) => options.eq(input, other),
    ne: (input: unknown, other: unknown) => options.ne(input, other),
    lt: (input: unknown, other: unknown) => options.lt(input, other),
    le: (input: unknown, other: unknown) => options.le(input, other),
    gt: (input: unknown, other: unknown) => options.gt(input, other),
    ge: (input: unknown, other: unknown) => options.ge(input, other),
    isclose: (input: unknown, other: unknown, isCloseOptions?: unknown) => options.isclose(input, other, isCloseOptions),
    pow: (input: unknown, exponent: number) => options.pow(input, exponent),
    neg: (input: unknown) => options.neg(input),
    negative: (input: unknown) => options.negative(input),
    exp: (input: unknown) => options.exp(input),
    expm1: (input: unknown) => options.expm1(input),
    log: (input: unknown) => options.log(input),
    log1p: (input: unknown) => options.log1p(input),
    sqr: (input: unknown) => options.sqr(input),
    square: (input: unknown) => options.square(input),
    recip: (input: unknown) => options.recip(input),
    reciprocal: (input: unknown) => options.reciprocal(input),
    abs: (input: unknown) => options.abs(input),
    sgn: (input: unknown) => options.sgn(input),
    sign: (input: unknown) => options.sign(input),
    step: (input: unknown) => options.step(input),
    isnan: (input: unknown) => options.isnan(input),
    isinf: (input: unknown) => options.isinf(input),
    isfinite: (input: unknown) => options.isfinite(input),
    floor: (input: unknown) => options.floor(input),
    ceil: (input: unknown) => options.ceil(input),
    round: (input: unknown) => options.round(input),
    trunc: (input: unknown) => options.trunc(input),
    sqrt: (input: unknown) => options.sqrt(input),
    rsqrt: (input: unknown) => options.rsqrt(input),
    relu: (input: unknown) => options.relu(input),
    gelu: (input: unknown) => options.gelu(input),
    silu: (input: unknown) => options.silu(input),
    sigmoid: (input: unknown) => options.sigmoid(input),
    tanh: (input: unknown) => options.tanh(input),
    sin: (input: unknown) => options.sin(input),
    cos: (input: unknown) => options.cos(input),
    tan: (input: unknown) => options.tan(input),
    maximum: (input: unknown, other: unknown) => options.maximum(input, other),
    minimum: (input: unknown, other: unknown) => options.minimum(input, other),
    where: (condition: unknown, input: unknown, other: unknown) => options.where(condition, input, other),
    maskedFill: (input: unknown, mask: unknown, value: unknown) => options.maskedFill(input, mask, value),
    masked_fill: (input: unknown, mask: unknown, value: unknown) => options.masked_fill(input, mask, value),
    sum: (input: unknown, dim?: number) => options.sum(input, dim),
    prod: (input: unknown, dim?: number) => options.prod(input, dim),
    cumsum: (input: unknown, dim?: number) => options.cumsum(input, dim),
    mean: (input: unknown, dim?: number) => options.mean(input, dim),
    max: (input: unknown, dim?: number) => options.max(input, dim),
    min: (input: unknown, dim?: number) => options.min(input, dim),
    any: (input: unknown, dim?: number) => options.any(input, dim),
    all: (input: unknown, dim?: number) => options.all(input, dim),
    argmax: (input: unknown, dim?: number) => options.argmax(input, dim),
    argmin: (input: unknown, dim?: number) => options.argmin(input, dim),
    variance: (input: unknown, dim?: number, correction?: number) => options.variance(input, dim, correction),
    var: (input: unknown, dim?: number, correction?: number) => options.var(input, dim, correction),
    std: (input: unknown, dim?: number, correction?: number) => options.std(input, dim, correction),
    norm: (input: unknown, dim?: number, p?: number) => options.norm(input, dim, p),
    softmax: (input: unknown, dim?: number) => options.softmax(input, dim),
    softmax_dim: (input: unknown, dim?: number) => options.softmax_dim(input, dim),
    softmaxDim: (input: unknown, dim?: number) => options.softmaxDim(input, dim),
    logSoftmax: (input: unknown, dim?: number) => options.logSoftmax(input, dim),
    log_softmax: (input: unknown, dim?: number) => options.log_softmax(input, dim),
    log_softmax_dim: (input: unknown, dim?: number) => options.log_softmax_dim(input, dim),
    logSoftmaxDim: (input: unknown, dim?: number) => options.logSoftmaxDim(input, dim),
    logsumexp: (input: unknown, dim?: number) => options.logsumexp(input, dim),
    logSumExp: (input: unknown, dim?: number) => options.logSumExp(input, dim),
    clamp: (input: unknown, min?: number | null, max?: number | null) => options.clamp(input, min, max),
    clip: (input: unknown, min?: number | null, max?: number | null) => options.clip(input, min, max),
    matmul: (input: unknown, other: unknown, otherShape?: readonly number[]) => options.matmul(input, other, otherShape),
    mm: (input: unknown, other: unknown, otherShape?: readonly number[]) => options.mm(input, other, otherShape),
    dot: (input: unknown, other: unknown, otherShape?: readonly number[]) => options.dot(input, other, otherShape),
    trace: (input: unknown) => options.trace(input),
    diagonal: (input: unknown) => options.diagonal(input),
    bmm: (input: unknown, other: unknown, otherShape?: readonly number[]) => options.bmm(input, other, otherShape),
    cat: (tensors: readonly TTensor[], dim = 0) => options.cat(tensors, dim),
    concat: (tensors: readonly TTensor[], dim = 0) => options.concat(tensors, dim),
    concatenate: (tensors: readonly TTensor[], dim = 0) => options.concatenate(tensors, dim),
    stack: (tensors: readonly TTensor[], dim = 0) => options.stack(tensors, dim),
    vstack: (tensors: readonly TTensor[]) => options.vstack(tensors),
    hstack: (tensors: readonly TTensor[]) => options.hstack(tensors),
    einsum: (equation: string, tensors: readonly TTensor[] | TTensor, ...moreTensors: readonly TTensor[]) => (
      options.einsum(equation, tensors, ...moreTensors)
    ),
  });
}

export function createTensorStaticSurfaceFromFacade<TTensor>(options: TensorStaticSurfaceFromFacadeOptions<TTensor>) {
  const { tensorCoreHelpers, tensorFacade, rootOps } = options;
  return createTensorStaticSurface<TTensor>({
    fromJSON: (value: unknown) => tensorCoreHelpers.fromJSON(value),
    fromNativeBuffer: (buffer: unknown, shape?: unknown, tensorOptions?: TensorOptions) => (
      tensorFacade.fromNativeBuffer(buffer, shape, tensorOptions)
    ),
    tensor: (data: unknown, shape?: unknown, tensorOptions?: TensorOptions) => (
      tensorFacade.tensor(data, shape, tensorOptions)
    ),
    parameter: (data: unknown, shape?: unknown, tensorOptions?: TensorOptions) => (
      tensorFacade.parameter(data, shape, tensorOptions)
    ),
    param: (data: unknown, shape?: unknown, tensorOptions?: TensorOptions) => (
      tensorFacade.param(data, shape, tensorOptions)
    ),
    scalar: rootOps.scalar,
    empty: rootOps.empty,
    emptyLike: rootOps.emptyLike,
    empty_like: rootOps.empty_like,
    zeros: rootOps.zeros,
    zerosLike: rootOps.zerosLike,
    zeros_like: rootOps.zeros_like,
    ones: rootOps.ones,
    onesLike: rootOps.onesLike,
    ones_like: rootOps.ones_like,
    eye: rootOps.eye,
    full: rootOps.full,
    fullLike: rootOps.fullLike,
    full_like: rootOps.full_like,
    rand: rootOps.rand,
    randLike: rootOps.randLike,
    rand_like: rootOps.rand_like,
    randn: rootOps.randn,
    randnLike: rootOps.randnLike,
    randn_like: rootOps.randn_like,
    randInt: rootOps.randInt,
    randint: rootOps.randint,
    randPerm: rootOps.randPerm,
    randperm: rootOps.randperm,
    manualSeed: rootOps.manualSeed,
    manual_seed: rootOps.manual_seed,
    initialSeed: rootOps.initialSeed,
    initial_seed: rootOps.initial_seed,
    seededRng: rootOps.seededRng,
    linspace: rootOps.linspace,
    arange: rootOps.arange,
    allclose: rootOps.allclose,
    equal: rootOps.equal,
    hasShape: rootOps.hasShape,
    requireShape: rootOps.requireShape,
    to: rootOps.to,
    cpu: rootOps.cpu,
    float: rootOps.float,
    float32: rootOps.float32,
    typeAs: rootOps.typeAs,
    type_as: rootOps.type_as,
    clone: rootOps.clone,
    detach: rootOps.detach,
    reshape: rootOps.reshape,
    view: rootOps.view,
    broadcastTo: rootOps.broadcastTo,
    expand: rootOps.expand,
    repeat: rootOps.repeat,
    tile: rootOps.tile,
    flatten: rootOps.flatten,
    squeeze: rootOps.squeeze,
    unsqueeze: rootOps.unsqueeze,
    transpose: rootOps.transpose,
    permute: rootOps.permute,
    flip: rootOps.flip,
    roll: rootOps.roll,
    select: rootOps.select,
    narrow: rootOps.narrow,
    slice: rootOps.slice,
    indexSelect: rootOps.indexSelect,
    index_select: rootOps.index_select,
    gather: rootOps.gather,
    take: rootOps.take,
    argsort: rootOps.argsort,
    sort: rootOps.sort,
    topk: rootOps.topk,
    scatterAdd: rootOps.scatterAdd,
    scatter_add: rootOps.scatter_add,
    split: rootOps.split,
    chunk: rootOps.chunk,
    unbind: rootOps.unbind,
    add: rootOps.add,
    sub: rootOps.sub,
    mul: rootOps.mul,
    div: rootOps.div,
    eq: rootOps.eq,
    ne: rootOps.ne,
    lt: rootOps.lt,
    le: rootOps.le,
    gt: rootOps.gt,
    ge: rootOps.ge,
    isclose: rootOps.isclose,
    pow: rootOps.pow,
    neg: rootOps.neg,
    negative: rootOps.negative,
    exp: rootOps.exp,
    expm1: rootOps.expm1,
    log: rootOps.log,
    log1p: rootOps.log1p,
    sqr: rootOps.sqr,
    square: rootOps.square,
    recip: rootOps.recip,
    reciprocal: rootOps.reciprocal,
    abs: rootOps.abs,
    sgn: rootOps.sgn,
    sign: rootOps.sign,
    step: rootOps.step,
    isnan: rootOps.isnan,
    isinf: rootOps.isinf,
    isfinite: rootOps.isfinite,
    floor: rootOps.floor,
    ceil: rootOps.ceil,
    round: rootOps.round,
    trunc: rootOps.trunc,
    sqrt: rootOps.sqrt,
    rsqrt: rootOps.rsqrt,
    relu: rootOps.relu,
    gelu: rootOps.gelu,
    silu: rootOps.silu,
    sigmoid: rootOps.sigmoid,
    tanh: rootOps.tanh,
    sin: rootOps.sin,
    cos: rootOps.cos,
    tan: rootOps.tan,
    maximum: rootOps.maximum,
    minimum: rootOps.minimum,
    where: rootOps.where,
    maskedFill: rootOps.maskedFill,
    masked_fill: rootOps.masked_fill,
    sum: rootOps.sum,
    prod: rootOps.prod,
    cumsum: rootOps.cumsum,
    mean: rootOps.mean,
    max: rootOps.max,
    min: rootOps.min,
    any: rootOps.any,
    all: rootOps.all,
    argmax: rootOps.argmax,
    argmin: rootOps.argmin,
    variance: rootOps.variance,
    var: (input: unknown, dim?: number, correction?: number) => rootOps.variance(input, dim, correction),
    std: rootOps.std,
    norm: rootOps.norm,
    softmax: rootOps.softmax,
    softmax_dim: rootOps.softmax_dim,
    softmaxDim: rootOps.softmaxDim,
    logSoftmax: rootOps.logSoftmax,
    log_softmax: rootOps.log_softmax,
    log_softmax_dim: rootOps.log_softmax_dim,
    logSoftmaxDim: rootOps.logSoftmaxDim,
    logsumexp: rootOps.logsumexp,
    logSumExp: rootOps.logSumExp,
    clamp: rootOps.clamp,
    clip: rootOps.clip,
    matmul: rootOps.matmul,
    mm: rootOps.mm,
    dot: rootOps.dot,
    trace: rootOps.trace,
    diagonal: rootOps.diagonal,
    bmm: rootOps.bmm,
    cat: rootOps.cat,
    concat: rootOps.concat,
    concatenate: rootOps.concatenate,
    stack: rootOps.stack,
    vstack: rootOps.vstack,
    hstack: rootOps.hstack,
    einsum: rootOps.einsum,
  });
}
