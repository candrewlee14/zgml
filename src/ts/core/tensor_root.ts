import type {
  AllCloseOptions,
  RandomIntTensorOptions,
  RandomTensorOptions,
  RandomUniformTensorOptions,
  TensorLike,
  TensorOptions,
  TensorShape,
  TensorToOptions,
  TensorToTarget,
} from "../public_api.js";
import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";
import {
  normalizeFactoryShape,
  sameShape,
} from "./shape.js";

export type TensorConstructor<TTensor> = {
  new(data: TensorLike, shape?: TensorShape, options?: TensorOptions): TTensor;
};

export type TensorFacade<TTensor> = {
  tensor(data: TensorLike, shape?: TensorShape, options?: TensorOptions): TTensor;
  parameter(data: TensorLike, shape?: TensorShape | TensorOptions, options?: TensorOptions): TTensor;
  cat(tensors: readonly TTensor[], dim?: number): TTensor;
  stack(tensors: readonly TTensor[], dim?: number): TTensor;
  einsum(equation: string, tensors: readonly TTensor[]): TTensor;
  einsum(equation: string, ...tensors: readonly TTensor[]): TTensor;
  full(shape: TensorShape, value: number, options?: TensorOptions): TTensor;
  empty(shape: TensorShape, options?: TensorOptions): TTensor;
  zeros(shape: TensorShape, options?: TensorOptions): TTensor;
  ones(shape: TensorShape, options?: TensorOptions): TTensor;
  eye(size: number, options?: TensorOptions): TTensor;
  scalar(value: number, options?: TensorOptions): TTensor;
  rand(shape: TensorShape, options?: RandomUniformTensorOptions): TTensor;
  randn(shape: TensorShape, options?: RandomTensorOptions): TTensor;
  randInt(first: number, second: number | TensorShape, third?: number | TensorShape | RandomIntTensorOptions, fourth?: RandomIntTensorOptions): TTensor;
  randint(first: number, second: number | TensorShape, third?: number | TensorShape | RandomIntTensorOptions, fourth?: RandomIntTensorOptions): TTensor;
  randPerm(size: number, options?: RandomIntTensorOptions): TTensor;
  randperm(size: number, options?: RandomIntTensorOptions): TTensor;
  manualSeed(seed: number): number;
  manual_seed(seed: number): number;
  initialSeed(): number | null;
  initial_seed(): number | null;
  seededRng(seed: number): () => number;
  linspace(first: readonly number[] | number, second: number, third: number, fourth?: TensorOptions): TTensor;
  arange(start: number, endOrOptions?: number | TensorOptions, stepOrOptions?: number | TensorOptions, maybeOptions?: TensorOptions): TTensor;
};

export type TensorRootSource<TTensor> = TTensor & {
  readonly shape: readonly number[];
  allclose(other: TensorLike, options?: AllCloseOptions): boolean;
  equal(other: TensorLike): boolean;
  to(target?: TensorToTarget | TensorToOptions, options?: TensorToOptions): TTensor;
  cpu(options?: TensorToOptions): TTensor;
  float(options?: TensorToOptions): TTensor;
  float32(options?: TensorToOptions): TTensor;
  typeAs(other: unknown, options?: TensorToOptions): TTensor;
  type_as(other: unknown, options?: TensorToOptions): TTensor;
  clone(): TTensor;
  detach(): TTensor;
  reshape(shape: TensorShape): TTensor;
  view(shape: TensorShape): TTensor;
  broadcastTo(shape: TensorShape): TTensor;
  expand(shape: TensorShape): TTensor;
  repeat(repeats: TensorShape): TTensor;
  tile(repeats: TensorShape): TTensor;
  flatten(startDim?: number, endDim?: number): TTensor;
  squeeze(dim?: number | null): TTensor;
  unsqueeze(dim: number): TTensor;
  transpose(dim0?: number, dim1?: number): TTensor;
  permute(dims: readonly number[]): TTensor;
  flip(dims: readonly number[]): TTensor;
  roll(shifts: number | readonly number[], dims?: number | readonly number[] | null): TTensor;
  select(dim: number, index: number): TTensor;
  narrow(dim: number, start: number, length: number): TTensor;
  slice(dim: number, start?: number | null, end?: number | null, step?: number): TTensor;
  indexSelect(dim: number, indices: TensorLike): TTensor;
  index_select(dim: number, indices: TensorLike): TTensor;
  gather(dim: number, index: TensorLike): TTensor;
  take(index: TensorLike): TTensor;
  argsort(dim?: number, descending?: boolean): TTensor;
  sort(dim?: number, descending?: boolean): Readonly<{ values: TTensor; indices: TTensor }>;
  topk(k: number, dim?: number, largest?: boolean, sorted?: boolean): Readonly<{ values: TTensor; indices: TTensor }>;
  scatterAdd(dim: number, index: TensorLike, src: TensorLike): TTensor;
  scatter_add(dim: number, index: TensorLike, src: TensorLike): TTensor;
  split(splitSizeOrSections: number | readonly number[], dim?: number): readonly TTensor[];
  chunk(chunks: number, dim?: number): readonly TTensor[];
  unbind(dim?: number): readonly TTensor[];
  add(other: TensorLike): TTensor;
  sub(other: TensorLike): TTensor;
  mul(other: TensorLike): TTensor;
  div(other: TensorLike): TTensor;
  eq(other: TensorLike): TTensor;
  ne(other: TensorLike): TTensor;
  lt(other: TensorLike): TTensor;
  le(other: TensorLike): TTensor;
  gt(other: TensorLike): TTensor;
  ge(other: TensorLike): TTensor;
  isclose(other: TensorLike, options?: unknown): TTensor;
  pow(exponent: number): TTensor;
  neg(): TTensor;
  negative(): TTensor;
  exp(): TTensor;
  expm1(): TTensor;
  log(): TTensor;
  log1p(): TTensor;
  sqr(): TTensor;
  square(): TTensor;
  recip(): TTensor;
  reciprocal(): TTensor;
  abs(): TTensor;
  sgn(): TTensor;
  sign(): TTensor;
  step(): TTensor;
  isnan(): TTensor;
  isinf(): TTensor;
  isfinite(): TTensor;
  floor(): TTensor;
  ceil(): TTensor;
  round(): TTensor;
  trunc(): TTensor;
  sqrt(): TTensor;
  rsqrt(): TTensor;
  relu(): TTensor;
  gelu(): TTensor;
  silu(): TTensor;
  sigmoid(): TTensor;
  tanh(): TTensor;
  sin(): TTensor;
  cos(): TTensor;
  tan(): TTensor;
  maximum(other: TensorLike): TTensor;
  minimum(other: TensorLike): TTensor;
  where(input: TensorLike, other: TensorLike): TTensor;
  maskedFill(mask: TensorLike, value: TensorLike): TTensor;
  masked_fill(mask: TensorLike, value: TensorLike): TTensor;
  sum(dim?: number): TTensor;
  prod(dim?: number): TTensor;
  cumsum(dim?: number): TTensor;
  mean(dim?: number): TTensor;
  max(dim?: number): TTensor;
  min(dim?: number): TTensor;
  any(dim?: number): TTensor;
  all(dim?: number): TTensor;
  argmax(dim?: number): TTensor;
  argmin(dim?: number): TTensor;
  variance(dim?: number, correction?: number): TTensor;
  var(dim?: number, correction?: number): TTensor;
  std(dim?: number, correction?: number): TTensor;
  norm(dim?: number, p?: number): TTensor;
  softmax(dim?: number): TTensor;
  softmax_dim(dim?: number): TTensor;
  softmaxDim(dim?: number): TTensor;
  logSoftmax(dim?: number): TTensor;
  log_softmax(dim?: number): TTensor;
  log_softmax_dim(dim?: number): TTensor;
  logSoftmaxDim(dim?: number): TTensor;
  logsumexp(dim?: number): TTensor;
  logSumExp(dim?: number): TTensor;
  clamp(min?: number | null, max?: number | null): TTensor;
  clip(min?: number | null, max?: number | null): TTensor;
  matmul(other: TensorLike, otherShape?: readonly number[]): TTensor;
  mm(other: TensorLike, otherShape?: readonly number[]): TTensor;
  dot(other: TensorLike, otherShape?: readonly number[]): TTensor;
  trace(): TTensor;
  diagonal(): TTensor;
  bmm(other: TensorLike, otherShape?: readonly number[]): TTensor;
};

export type TensorRootOpsOptions<TTensor> = Readonly<{
  readonly Tensor: TensorConstructor<TTensor>;
  readonly tensorFacade: TensorFacade<TTensor>;
}>;

export function createTensorRootOps<TTensor>(options: TensorRootOpsOptions<TTensor>) {
  const { Tensor, tensorFacade } = options;

  function tensor(data: TensorLike, shape?: TensorShape, tensorOptions?: TensorOptions): TTensor {
    return tensorFacade.tensor(data, shape, tensorOptions);
  }

  function parameter(data: TensorLike, shape?: TensorShape | TensorOptions, tensorOptions?: TensorOptions): TTensor {
    return tensorFacade.parameter(data, shape, tensorOptions);
  }

  function cat(tensors: readonly TTensor[], dim = 0): TTensor {
    return tensorFacade.cat(tensors, dim);
  }

  function concat(tensors: readonly TTensor[], dim = 0): TTensor {
    return cat(tensors, dim);
  }

  function concatenate(tensors: readonly TTensor[], dim = 0): TTensor {
    return cat(tensors, dim);
  }

  function stack(tensors: readonly TTensor[], dim = 0): TTensor {
    return tensorFacade.stack(tensors, dim);
  }

  function einsum(equation: string, operandsOrFirst: readonly TTensor[] | TTensor, ...moreOperands: readonly TTensor[]): TTensor {
    return Array.isArray(operandsOrFirst) && moreOperands.length === 0
      ? tensorFacade.einsum(equation, operandsOrFirst)
      : tensorFacade.einsum(equation, operandsOrFirst as TTensor, ...moreOperands);
  }

  function requireStackArray(tensors: readonly TTensor[], label: string): readonly TTensor[] {
    if (!Array.isArray(tensors) || tensors.length === 0) throw new Error(`${label} requires a non-empty tensor array`);
    return tensors;
  }

  function vstack(tensors: readonly TTensor[]): TTensor {
    const values = requireStackArray(tensors, "vstack");
    const prepared = values.map((value) => {
      const tensorValue = value as TensorRootSource<TTensor>;
      return tensorValue.shape.length === 1 ? tensorValue.reshape([1, tensorValue.shape[0]]) : tensorValue;
    });
    return cat(prepared, 0);
  }

  function hstack(tensors: readonly TTensor[]): TTensor {
    const values = requireStackArray(tensors, "hstack");
    const first = values[0] as TensorRootSource<TTensor>;
    return cat(values, first.shape.length <= 1 ? 0 : 1);
  }

  function full(shape: TensorShape, value: number, tensorOptions: TensorOptions = {}): TTensor {
    return tensorFacade.full(shape, value, tensorOptions);
  }

  function zeros(shape: TensorShape, tensorOptions: TensorOptions = {}): TTensor {
    return tensorFacade.zeros(shape, tensorOptions);
  }

  function empty(shape: TensorShape, tensorOptions: TensorOptions = {}): TTensor {
    return tensorFacade.empty(shape, tensorOptions);
  }

  function emptyLike(input: TensorLike, tensorOptions: TensorOptions = {}): TTensor {
    return empty(source(input).shape, tensorOptions);
  }

  function empty_like(input: TensorLike, tensorOptions: TensorOptions = {}): TTensor {
    return emptyLike(input, tensorOptions);
  }

  function zerosLike(input: TensorLike, tensorOptions: TensorOptions = {}): TTensor {
    return tensorFacade.zeros(source(input).shape, tensorOptions);
  }

  function zeros_like(input: TensorLike, tensorOptions: TensorOptions = {}): TTensor {
    return zerosLike(input, tensorOptions);
  }

  function ones(shape: TensorShape, tensorOptions: TensorOptions = {}): TTensor {
    return tensorFacade.ones(shape, tensorOptions);
  }

  function onesLike(input: TensorLike, tensorOptions: TensorOptions = {}): TTensor {
    return tensorFacade.ones(source(input).shape, tensorOptions);
  }

  function ones_like(input: TensorLike, tensorOptions: TensorOptions = {}): TTensor {
    return onesLike(input, tensorOptions);
  }

  function eye(size: number, tensorOptions: TensorOptions = {}): TTensor {
    return tensorFacade.eye(size, tensorOptions);
  }

  function scalar(value: number, tensorOptions: TensorOptions = {}): TTensor {
    return tensorFacade.scalar(value, tensorOptions);
  }

  function fullLike(input: TensorLike, value: number, tensorOptions: TensorOptions = {}): TTensor {
    return tensorFacade.full(source(input).shape, value, tensorOptions);
  }

  function full_like(input: TensorLike, value: number, tensorOptions: TensorOptions = {}): TTensor {
    return fullLike(input, value, tensorOptions);
  }

  function rand(shape: TensorShape, tensorOptions: RandomUniformTensorOptions = {}): TTensor {
    return tensorFacade.rand(shape, tensorOptions);
  }

  function randLike(input: TensorLike, tensorOptions: RandomUniformTensorOptions = {}): TTensor {
    return tensorFacade.rand(source(input).shape, tensorOptions);
  }

  function rand_like(input: TensorLike, tensorOptions: RandomUniformTensorOptions = {}): TTensor {
    return randLike(input, tensorOptions);
  }

  function randn(shape: TensorShape, tensorOptions: RandomTensorOptions = {}): TTensor {
    return tensorFacade.randn(shape, tensorOptions);
  }

  function randnLike(input: TensorLike, tensorOptions: RandomTensorOptions = {}): TTensor {
    return tensorFacade.randn(source(input).shape, tensorOptions);
  }

  function randn_like(input: TensorLike, tensorOptions: RandomTensorOptions = {}): TTensor {
    return randnLike(input, tensorOptions);
  }

  function randInt(first: number, second: number | TensorShape, third?: number | TensorShape | RandomIntTensorOptions, fourth?: RandomIntTensorOptions): TTensor {
    return tensorFacade.randInt(first, second, third, fourth);
  }

  function randint(first: number, second: number | TensorShape, third?: number | TensorShape | RandomIntTensorOptions, fourth?: RandomIntTensorOptions): TTensor {
    return tensorFacade.randint(first, second, third, fourth);
  }

  function randPerm(size: number, tensorOptions: RandomIntTensorOptions = {}): TTensor {
    return tensorFacade.randPerm(size, tensorOptions);
  }

  function randperm(size: number, tensorOptions: RandomIntTensorOptions = {}): TTensor {
    return tensorFacade.randperm(size, tensorOptions);
  }

  function manualSeed(seed: number): number {
    return tensorFacade.manualSeed(seed);
  }

  function manual_seed(seed: number): number {
    return tensorFacade.manual_seed(seed);
  }

  function initialSeed(): number | null {
    return tensorFacade.initialSeed();
  }

  function initial_seed(): number | null {
    return tensorFacade.initial_seed();
  }

  function seededRng(seed: number): () => number {
    return tensorFacade.seededRng(seed);
  }

  function linspace(first: readonly number[] | number, second: number, third: number, fourth: TensorOptions = {}): TTensor {
    return tensorFacade.linspace(first, second, third, fourth);
  }

  function arange(start: number, endOrOptions?: number | TensorOptions, stepOrOptions?: number | TensorOptions, maybeOptions: TensorOptions = {}): TTensor {
    return tensorFacade.arange(start, endOrOptions, stepOrOptions, maybeOptions);
  }

  function source(value: TensorLike): TensorRootSource<TTensor> {
    return value instanceof Tensor ? value as TensorRootSource<TTensor> : tensor(value) as TensorRootSource<TTensor>;
  }

  function allclose(actual: TensorLike, expected: TensorLike, optionsForCompare?: AllCloseOptions): boolean {
    return source(actual).allclose(expected, optionsForCompare);
  }

  function equal(actual: TensorLike, expected: TensorLike): boolean {
    return source(actual).equal(expected);
  }

  function hasShape(input: TensorLike, shape: TensorShape): boolean {
    const expectedShape = normalizeFactoryShape(shape, "hasShape shape").shape;
    return sameShape(source(input).shape, expectedShape);
  }

  function requireShape(input: TensorLike, shape: TensorShape): TTensor {
    const tensorValue = source(input);
    const expectedShape = normalizeFactoryShape(shape, "requireShape shape").shape;
    if (!sameShape(tensorValue.shape, expectedShape)) {
      throw new Error(`Tensor.requireShape expected [${expectedShape.join(", ")}], got [${tensorValue.shape.join(", ")}]`);
    }
    return tensorValue;
  }

  function to(input: TensorLike, target?: TensorToTarget | TensorToOptions, options?: TensorToOptions): TTensor {
    return source(input).to(target, options);
  }

  function cpu(input: TensorLike, options?: TensorToOptions): TTensor {
    return source(input).cpu(options);
  }

  function float(input: TensorLike, options?: TensorToOptions): TTensor {
    return source(input).float(options);
  }

  function float32(input: TensorLike, options?: TensorToOptions): TTensor {
    return source(input).float32(options);
  }

  function typeAs(input: TensorLike, other: TensorLike, options?: TensorToOptions): TTensor {
    return source(input).typeAs(source(other), options);
  }

  function type_as(input: TensorLike, other: TensorLike, options?: TensorToOptions): TTensor {
    return source(input).type_as(source(other), options);
  }

  function clone(input: TensorLike): TTensor {
    return source(input).clone();
  }

  function detach(input: TensorLike): TTensor {
    return source(input).detach();
  }

  function reshape(input: TensorLike, shape: TensorShape): TTensor {
    return source(input).reshape(shape);
  }

  function view(input: TensorLike, shape: TensorShape): TTensor {
    return source(input).view(shape);
  }

  function broadcastTo(input: TensorLike, shape: TensorShape): TTensor {
    return source(input).broadcastTo(shape);
  }

  function expand(input: TensorLike, shape: TensorShape): TTensor {
    return source(input).expand(shape);
  }

  function repeat(input: TensorLike, repeats: TensorShape): TTensor {
    return source(input).repeat(repeats);
  }

  function tile(input: TensorLike, repeats: TensorShape): TTensor {
    return source(input).tile(repeats);
  }

  function flatten(input: TensorLike, startDim?: number, endDim?: number): TTensor {
    return source(input).flatten(startDim, endDim);
  }

  function squeeze(input: TensorLike, dim?: number | null): TTensor {
    return source(input).squeeze(dim);
  }

  function unsqueeze(input: TensorLike, dim: number): TTensor {
    return source(input).unsqueeze(dim);
  }

  function transpose(input: TensorLike, dim0?: number, dim1?: number): TTensor {
    return source(input).transpose(dim0, dim1);
  }

  function permute(input: TensorLike, dims: readonly number[]): TTensor {
    return source(input).permute(dims);
  }

  function flip(input: TensorLike, dims: readonly number[]): TTensor {
    return source(input).flip(dims);
  }

  function roll(input: TensorLike, shifts: number | readonly number[], dims?: number | readonly number[] | null): TTensor {
    return source(input).roll(shifts, dims);
  }

  function select(input: TensorLike, dim: number, index: number): TTensor {
    return source(input).select(dim, index);
  }

  function narrow(input: TensorLike, dim: number, start: number, length: number): TTensor {
    return source(input).narrow(dim, start, length);
  }

  function slice(input: TensorLike, dim: number, start?: number | null, end?: number | null, step?: number): TTensor {
    return source(input).slice(dim, start, end, step);
  }

  function indexSelect(input: TensorLike, dim: number, indices: TensorLike): TTensor {
    return source(input).indexSelect(dim, indices);
  }

  function index_select(input: TensorLike, dim: number, indices: TensorLike): TTensor {
    return source(input).index_select(dim, indices);
  }

  function gather(input: TensorLike, dim: number, index: TensorLike): TTensor {
    return source(input).gather(dim, index);
  }

  function take(input: TensorLike, index: TensorLike): TTensor {
    return source(input).take(index);
  }

  function argsort(input: TensorLike, dim = -1, descending = false): TTensor {
    return source(input).argsort(dim, descending);
  }

  function sort(input: TensorLike, dim = -1, descending = false): Readonly<{ values: TTensor; indices: TTensor }> {
    return source(input).sort(dim, descending);
  }

  function topk(input: TensorLike, k: number, dim = -1, largest = true, sorted = true): Readonly<{ values: TTensor; indices: TTensor }> {
    return source(input).topk(k, dim, largest, sorted);
  }

  function scatterAdd(input: TensorLike, dim: number, index: TensorLike, src: TensorLike): TTensor {
    return source(input).scatterAdd(dim, index, src);
  }

  function scatter_add(input: TensorLike, dim: number, index: TensorLike, src: TensorLike): TTensor {
    return source(input).scatter_add(dim, index, src);
  }

  function split(input: TensorLike, splitSizeOrSections: number | readonly number[], dim = 0): readonly TTensor[] {
    return source(input).split(splitSizeOrSections, dim);
  }

  function chunk(input: TensorLike, chunks: number, dim = 0): readonly TTensor[] {
    return source(input).chunk(chunks, dim);
  }

  function unbind(input: TensorLike, dim = 0): readonly TTensor[] {
    return source(input).unbind(dim);
  }

  function add(input: TensorLike, other: TensorLike): TTensor {
    return source(input).add(other);
  }

  function sub(input: TensorLike, other: TensorLike): TTensor {
    return source(input).sub(other);
  }

  function mul(input: TensorLike, other: TensorLike): TTensor {
    return source(input).mul(other);
  }

  function div(input: TensorLike, other: TensorLike): TTensor {
    return source(input).div(other);
  }

  function eq(input: TensorLike, other: TensorLike): TTensor {
    return source(input).eq(other);
  }

  function ne(input: TensorLike, other: TensorLike): TTensor {
    return source(input).ne(other);
  }

  function lt(input: TensorLike, other: TensorLike): TTensor {
    return source(input).lt(other);
  }

  function le(input: TensorLike, other: TensorLike): TTensor {
    return source(input).le(other);
  }

  function gt(input: TensorLike, other: TensorLike): TTensor {
    return source(input).gt(other);
  }

  function ge(input: TensorLike, other: TensorLike): TTensor {
    return source(input).ge(other);
  }

  function isclose(input: TensorLike, other: TensorLike, options?: unknown): TTensor {
    return source(input).isclose(other, options);
  }

  function pow(input: TensorLike, exponent: number): TTensor {
    return source(input).pow(exponent);
  }

  function neg(input: TensorLike): TTensor {
    return source(input).neg();
  }

  function negative(input: TensorLike): TTensor {
    return source(input).negative();
  }

  function exp(input: TensorLike): TTensor {
    return source(input).exp();
  }

  function expm1(input: TensorLike): TTensor {
    return source(input).expm1();
  }

  function log(input: TensorLike): TTensor {
    return source(input).log();
  }

  function log1p(input: TensorLike): TTensor {
    return source(input).log1p();
  }

  function sqr(input: TensorLike): TTensor {
    return source(input).sqr();
  }

  function square(input: TensorLike): TTensor {
    return source(input).square();
  }

  function recip(input: TensorLike): TTensor {
    return source(input).recip();
  }

  function reciprocal(input: TensorLike): TTensor {
    return source(input).reciprocal();
  }

  function abs(input: TensorLike): TTensor {
    return source(input).abs();
  }

  function sgn(input: TensorLike): TTensor {
    return source(input).sgn();
  }

  function sign(input: TensorLike): TTensor {
    return source(input).sign();
  }

  function step(input: TensorLike): TTensor {
    return source(input).step();
  }

  function isnan(input: TensorLike): TTensor {
    return source(input).isnan();
  }

  function isinf(input: TensorLike): TTensor {
    return source(input).isinf();
  }

  function isfinite(input: TensorLike): TTensor {
    return source(input).isfinite();
  }

  function floor(input: TensorLike): TTensor {
    return source(input).floor();
  }

  function ceil(input: TensorLike): TTensor {
    return source(input).ceil();
  }

  function round(input: TensorLike): TTensor {
    return source(input).round();
  }

  function trunc(input: TensorLike): TTensor {
    return source(input).trunc();
  }

  function sqrt(input: TensorLike): TTensor {
    return source(input).sqrt();
  }

  function rsqrt(input: TensorLike): TTensor {
    return source(input).rsqrt();
  }

  function relu(input: TensorLike): TTensor {
    return source(input).relu();
  }

  function gelu(input: TensorLike): TTensor {
    return source(input).gelu();
  }

  function silu(input: TensorLike): TTensor {
    return source(input).silu();
  }

  function sigmoid(input: TensorLike): TTensor {
    return source(input).sigmoid();
  }

  function tanh(input: TensorLike): TTensor {
    return source(input).tanh();
  }

  function sin(input: TensorLike): TTensor {
    return source(input).sin();
  }

  function cos(input: TensorLike): TTensor {
    return source(input).cos();
  }

  function tan(input: TensorLike): TTensor {
    return source(input).tan();
  }

  function maximum(input: TensorLike, other: TensorLike): TTensor {
    return source(input).maximum(other);
  }

  function minimum(input: TensorLike, other: TensorLike): TTensor {
    return source(input).minimum(other);
  }

  function where(condition: TensorLike, input: TensorLike, other: TensorLike): TTensor {
    return source(condition).where(input, other);
  }

  function maskedFill(input: TensorLike, mask: TensorLike, value: TensorLike): TTensor {
    return source(input).maskedFill(mask, value);
  }

  function masked_fill(input: TensorLike, mask: TensorLike, value: TensorLike): TTensor {
    return source(input).masked_fill(mask, value);
  }

  function sum(input: TensorLike, dim?: number): TTensor {
    return source(input).sum(dim);
  }

  function prod(input: TensorLike, dim?: number): TTensor {
    return source(input).prod(dim);
  }

  function cumsum(input: TensorLike, dim?: number): TTensor {
    return source(input).cumsum(dim);
  }

  function mean(input: TensorLike, dim?: number): TTensor {
    return source(input).mean(dim);
  }

  function max(input: TensorLike, dim?: number): TTensor {
    return source(input).max(dim);
  }

  function min(input: TensorLike, dim?: number): TTensor {
    return source(input).min(dim);
  }

  function any(input: TensorLike, dim?: number): TTensor {
    return source(input).any(dim);
  }

  function all(input: TensorLike, dim?: number): TTensor {
    return source(input).all(dim);
  }

  function argmax(input: TensorLike, dim?: number): TTensor {
    return source(input).argmax(dim);
  }

  function argmin(input: TensorLike, dim?: number): TTensor {
    return source(input).argmin(dim);
  }

  function variance(input: TensorLike, dim?: number, correction?: number): TTensor {
    return source(input).variance(dim, correction);
  }

  function std(input: TensorLike, dim?: number, correction?: number): TTensor {
    return source(input).std(dim, correction);
  }

  function norm(input: TensorLike, dim?: number, p?: number): TTensor {
    return source(input).norm(dim, p);
  }

  function softmax(input: TensorLike, dim?: number): TTensor {
    const tensor = source(input);
    return dim === undefined ? tensor.softmax() : tensor.softmaxDim(dim);
  }

  function softmaxDim(input: TensorLike, dim?: number): TTensor {
    return source(input).softmaxDim(dim);
  }

  function softmax_dim(input: TensorLike, dim?: number): TTensor {
    return dim === undefined ? softmax(input) : softmaxDim(input, dim);
  }

  function logSoftmax(input: TensorLike, dim?: number): TTensor {
    const tensor = source(input);
    return dim === undefined ? tensor.logSoftmax() : tensor.logSoftmaxDim(dim);
  }

  function logSoftmaxDim(input: TensorLike, dim?: number): TTensor {
    return source(input).logSoftmaxDim(dim);
  }

  function log_softmax(input: TensorLike, dim?: number): TTensor {
    return logSoftmax(input, dim);
  }

  function log_softmax_dim(input: TensorLike, dim?: number): TTensor {
    return dim === undefined ? logSoftmax(input) : logSoftmaxDim(input, dim);
  }

  function logsumexp(input: TensorLike, dim?: number): TTensor {
    return source(input).logsumexp(dim);
  }

  function logSumExp(input: TensorLike, dim?: number): TTensor {
    return source(input).logSumExp(dim);
  }

  function clamp(input: TensorLike, min?: number | null, max?: number | null): TTensor {
    return source(input).clamp(min, max);
  }

  function clip(input: TensorLike, min?: number | null, max?: number | null): TTensor {
    return source(input).clip(min, max);
  }

  function matmul(input: TensorLike, other: TensorLike, otherShape?: readonly number[]): TTensor {
    return source(input).matmul(other, otherShape);
  }

  function mm(input: TensorLike, other: TensorLike, otherShape?: readonly number[]): TTensor {
    return source(input).mm(other, otherShape);
  }

  function dot(input: TensorLike, other: TensorLike, otherShape?: readonly number[]): TTensor {
    return source(input).dot(other, otherShape);
  }

  function trace(input: TensorLike): TTensor {
    return source(input).trace();
  }

  function diagonal(input: TensorLike): TTensor {
    return source(input).diagonal();
  }

  function bmm(input: TensorLike, other: TensorLike, otherShape?: readonly number[]): TTensor {
    return source(input).bmm(other, otherShape);
  }

  return Object.freeze({
    tensor,
    asTensor: tensor,
    as_tensor: tensor,
    asarray: tensor,
    fromNumpy: tensor,
    from_numpy: tensor,
    parameter,
    cat,
    concat,
    concatenate,
    stack,
    vstack,
    hstack,
    einsum,
    full,
    fullLike,
    full_like,
    empty,
    emptyLike,
    empty_like,
    zeros,
    zerosLike,
    zeros_like,
    ones,
    onesLike,
    ones_like,
    eye,
    scalar,
    rand,
    randLike,
    rand_like,
    randn,
    randnLike,
    randn_like,
    randInt,
    randint,
    randPerm,
    randperm,
    manualSeed,
    manual_seed,
    initialSeed,
    initial_seed,
    seededRng,
    linspace,
    arange,
    allclose,
    equal,
    hasShape,
    requireShape,
    to,
    cpu,
    float,
    float32,
    typeAs,
    type_as,
    clone,
    detach,
    reshape,
    view,
    broadcastTo,
    expand,
    repeat,
    tile,
    flatten,
    squeeze,
    unsqueeze,
    transpose,
    permute,
    flip,
    roll,
    select,
    narrow,
    slice,
    indexSelect,
    index_select,
    gather,
    take,
    argsort,
    sort,
    topk,
    scatterAdd,
    scatter_add,
    split,
    chunk,
    unbind,
    add,
    sub,
    mul,
    div,
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    isclose,
    pow,
    neg,
    negative,
    exp,
    expm1,
    log,
    log1p,
    sqr,
    square,
    recip,
    reciprocal,
    abs,
    sgn,
    sign,
    step,
    isnan,
    isinf,
    isfinite,
    floor,
    ceil,
    round,
    trunc,
    sqrt,
    rsqrt,
    relu,
    gelu,
    silu,
    sigmoid,
    tanh,
    sin,
    cos,
    tan,
    maximum,
    minimum,
    where,
    maskedFill,
    masked_fill,
    sum,
    prod,
    cumsum,
    mean,
    max,
    min,
    any,
    all,
    argmax,
    argmin,
    variance,
    std,
    norm,
    softmax,
    softmax_dim,
    softmaxDim,
    logSoftmax,
    log_softmax,
    log_softmax_dim,
    logSoftmaxDim,
    logsumexp,
    logSumExp,
    clamp,
    clip,
    matmul,
    mm,
    dot,
    trace,
    diagonal,
    bmm,
  });
}

export const tensorRootManifest = Object.freeze({
  kind: "zgml-tensor-root",
  ...tsRuntimeManifestPolicy("src/ts/core/tensor_root.ts", "Tensor root helpers -> Tensor facade"),
});
