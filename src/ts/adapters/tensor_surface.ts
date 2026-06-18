import type { TensorHostSurface } from "../core/tensor_host_surface.js";
import type { createTensorMathSurfaceHelpers } from "../core/tensor_math.js";

type AdapterCallback<Return> = {
  bivarianceHack(...args: unknown[]): Return;
}["bivarianceHack"];

type AdapterCallableSurface<TSurface> = Readonly<{
  [Key in keyof TSurface]: TSurface[Key] extends (...args: infer _Args) => infer Return
    ? AdapterCallback<Return>
    : TSurface[Key];
}>;

type AdapterTensorHostSurfaceHelpers = Pick<
  {
    [Key in keyof TensorHostSurface<Tensor>]: AdapterCallableSurface<TensorHostSurface<Tensor>[Key]>;
  },
  | "tensorFacade"
  | "tensorNativeSurfaceHelpers"
  | "tensorStaticHelpers"
  | "tensorInfoSurfaceHelpers"
  | "tensorIndexSurfaceHelpers"
  | "tensorViewSurfaceHelpers"
>;

export type AdapterTensorSurfaceHelpers = AdapterTensorHostSurfaceHelpers & Readonly<{
  tensorMathSurfaceHelpers: AdapterCallableSurface<ReturnType<typeof createTensorMathSurfaceHelpers<Tensor>>>;
}>;

let currentSurface: AdapterTensorSurfaceHelpers | null = null;

function surface(): AdapterTensorSurfaceHelpers {
  if (currentSurface === null) {
    throw new Error("Adapter Tensor surface is not initialized");
  }
  return currentSurface;
}

function requireSameShapeForInPlace(target: Tensor, result: Tensor, label: string) {
  if (target.shape.length !== result.shape.length || target.shape.some((dim, index) => dim !== result.shape[index])) {
    throw new Error(`${label} result shape [${result.shape.join(",")}] must match tensor shape [${target.shape.join(",")}]`);
  }
  return result;
}

export function setAdapterTensorSurfaceHelpers(helpers: AdapterTensorSurfaceHelpers): void {
  currentSurface = helpers;
}

export class Tensor {
  [key: string]: unknown;
  readonly data!: Float32Array;
  readonly shape!: readonly number[];
  requiresGrad!: boolean;
  grad!: Float32Array | null;
  private _prev!: Tensor[];
  private _backward!: (grad: Float32Array | null) => void;

  constructor(data: unknown, shape?: unknown, options: unknown = {}) {
    surface().tensorFacade.initialize(this, data, shape, options);
  }

  get length() {
    return surface().tensorInfoSurfaceHelpers.length(this);
  }

  get rank() {
    return surface().tensorInfoSurfaceHelpers.rank(this);
  }

  get ndim() {
    return surface().tensorInfoSurfaceHelpers.ndim(this);
  }

  get dtype() {
    return surface().tensorInfoSurfaceHelpers.dtype(this);
  }

  get device() {
    return surface().tensorInfoSurfaceHelpers.device(this);
  }

  isCpu() {
    return surface().tensorInfoSurfaceHelpers.isCpu(this);
  }

  is_cpu() {
    return surface().tensorInfoSurfaceHelpers.is_cpu(this);
  }

  isFloatingPoint() {
    return surface().tensorInfoSurfaceHelpers.isFloatingPoint(this);
  }

  is_floating_point() {
    return surface().tensorInfoSurfaceHelpers.is_floating_point(this);
  }

  get isLeaf() {
    return surface().tensorInfoSurfaceHelpers.isLeaf(this);
  }

  get is_leaf() {
    return surface().tensorInfoSurfaceHelpers.is_leaf(this);
  }

  get requires_grad(): boolean {
    return surface().tensorInfoSurfaceHelpers.requires_grad(this);
  }

  set requires_grad(value: boolean) {
    surface().tensorInfoSurfaceHelpers.setRequires_grad(this, value);
  }

  requires_grad_(value: boolean = true) {
    return surface().tensorInfoSurfaceHelpers.requires_grad_(this, value);
  }

  requiresGrad_(value: boolean = true) {
    return surface().tensorInfoSurfaceHelpers.requires_grad_(this, value);
  }

  toFloat32Array() {
    return surface().tensorInfoSurfaceHelpers.toFloat32Array(this);
  }

  numpy() {
    return surface().tensorInfoSurfaceHelpers.numpy(this);
  }

  to(target: unknown, options: unknown) {
    return surface().tensorInfoSurfaceHelpers.to(this, target, options);
  }

  cpu(options: unknown) {
    return surface().tensorInfoSurfaceHelpers.cpu(this, options);
  }

  float(options: unknown) {
    return surface().tensorInfoSurfaceHelpers.float(this, options);
  }

  float32(options: unknown) {
    return surface().tensorInfoSurfaceHelpers.float32(this, options);
  }

  typeAs(other: unknown, options: unknown) {
    return surface().tensorInfoSurfaceHelpers.typeAs(this, other, options);
  }

  type_as(other: unknown, options: unknown) {
    return surface().tensorInfoSurfaceHelpers.type_as(this, other, options);
  }

  new_empty(shape: unknown, options: unknown) {
    return surface().tensorStaticHelpers.empty(shape, options).type_as(this, undefined);
  }

  newEmpty(shape: unknown, options: unknown) {
    return this.new_empty(shape, options);
  }

  new_zeros(shape: unknown, options: unknown) {
    return surface().tensorStaticHelpers.zeros(shape, options).type_as(this, undefined);
  }

  newZeros(shape: unknown, options: unknown) {
    return this.new_zeros(shape, options);
  }

  new_ones(shape: unknown, options: unknown) {
    return surface().tensorStaticHelpers.ones(shape, options).type_as(this, undefined);
  }

  newOnes(shape: unknown, options: unknown) {
    return this.new_ones(shape, options);
  }

  new_full(shape: unknown, value: unknown, options: unknown) {
    return surface().tensorStaticHelpers.full(shape, value, options).type_as(this, undefined);
  }

  newFull(shape: unknown, value: unknown, options: unknown) {
    return this.new_full(shape, value, options);
  }

  inspect() {
    return surface().tensorInfoSurfaceHelpers.inspect(this);
  }

  toNativeBuffer(options: unknown = {}) {
    return surface().tensorNativeSurfaceHelpers.toNativeBuffer(this, options);
  }

  place(program: unknown, kind: unknown = "input", options: unknown = {}) {
    return surface().tensorNativeSurfaceHelpers.place(this, program, kind, options);
  }

  item() {
    return surface().tensorInfoSurfaceHelpers.item(this);
  }

  toNumber() {
    return surface().tensorInfoSurfaceHelpers.toNumber(this);
  }

  dim() {
    return surface().tensorInfoSurfaceHelpers.dim(this);
  }

  ndimension() {
    return surface().tensorInfoSurfaceHelpers.ndimension(this);
  }

  numel() {
    return surface().tensorInfoSurfaceHelpers.numel(this);
  }

  nelement() {
    return surface().tensorInfoSurfaceHelpers.nelement(this);
  }

  elementSize() {
    return surface().tensorInfoSurfaceHelpers.elementSize(this);
  }

  element_size() {
    return surface().tensorInfoSurfaceHelpers.element_size(this);
  }

  nbytes() {
    return surface().tensorInfoSurfaceHelpers.nbytes(this);
  }

  size(dim: unknown) {
    return surface().tensorInfoSurfaceHelpers.size(this, dim);
  }

  allclose(other: unknown, options: unknown) {
    return surface().tensorInfoSurfaceHelpers.allclose(this, other, options);
  }

  equal(other: unknown) {
    return surface().tensorInfoSurfaceHelpers.equal(this, other);
  }

  stride(dim: unknown) {
    return surface().tensorInfoSurfaceHelpers.stride(this, dim);
  }

  strides() {
    return surface().tensorInfoSurfaceHelpers.strides(this);
  }

  storageOffset() {
    return surface().tensorInfoSurfaceHelpers.storageOffset(this);
  }

  storage_offset() {
    return surface().tensorInfoSurfaceHelpers.storage_offset(this);
  }

  isContiguous() {
    return surface().tensorInfoSurfaceHelpers.isContiguous(this);
  }

  is_contiguous() {
    return surface().tensorInfoSurfaceHelpers.is_contiguous(this);
  }

  contiguous() {
    return surface().tensorInfoSurfaceHelpers.contiguous(this);
  }

  get(...indices: unknown[]) {
    return surface().tensorIndexSurfaceHelpers.get(this, indices);
  }

  set(indicesOrFirst: unknown, ...rest: unknown[]) {
    return surface().tensorIndexSurfaceHelpers.set(this, indicesOrFirst, rest);
  }

  toArray() {
    return surface().tensorIndexSurfaceHelpers.toArray(this);
  }

  tolist() {
    return surface().tensorIndexSurfaceHelpers.tolist(this);
  }

  to_list() {
    return surface().tensorIndexSurfaceHelpers.to_list(this);
  }

  valueOf() {
    return surface().tensorInfoSurfaceHelpers.valueOf(this);
  }

  [Symbol.toPrimitive]() {
    return surface().tensorInfoSurfaceHelpers.toPrimitive(this);
  }

  zeroGrad(options: unknown = {}) {
    return surface().tensorInfoSurfaceHelpers.zeroGrad(this, options);
  }

  zero_grad(options: unknown = {}) {
    return surface().tensorInfoSurfaceHelpers.zero_grad(this, options);
  }

  fill_(value: unknown) {
    return surface().tensorInfoSurfaceHelpers.fill_(this, value);
  }

  zero_() {
    return surface().tensorInfoSurfaceHelpers.zero_(this);
  }

  ones_() {
    return surface().tensorInfoSurfaceHelpers.ones_(this);
  }

  copy_(source: unknown) {
    return surface().tensorInfoSurfaceHelpers.copy_(this, source);
  }

  backward(gradient: unknown) {
    return surface().tensorInfoSurfaceHelpers.backward(this, gradient);
  }

  clone() {
    return surface().tensorViewSurfaceHelpers.clone(this);
  }

  detach() {
    return surface().tensorViewSurfaceHelpers.detach(this);
  }

  detach_() {
    return surface().tensorInfoSurfaceHelpers.detach_(this);
  }

  reshape(shape: unknown) {
    return surface().tensorViewSurfaceHelpers.reshape(this, shape);
  }

  view(shape: unknown) {
    return surface().tensorViewSurfaceHelpers.view(this, shape);
  }

  reshape_as(other: unknown) {
    if (!other || typeof other !== "object" || !Array.isArray((other as { shape?: unknown }).shape)) {
      throw new Error("Tensor.reshape_as requires a tensor-like shape source");
    }
    return this.reshape((other as { shape: readonly number[] }).shape);
  }

  view_as(other: unknown) {
    if (!other || typeof other !== "object" || !Array.isArray((other as { shape?: unknown }).shape)) {
      throw new Error("Tensor.view_as requires a tensor-like shape source");
    }
    return this.view((other as { shape: readonly number[] }).shape);
  }

  broadcastTo(shape: unknown) {
    return surface().tensorViewSurfaceHelpers.broadcastTo(this, shape);
  }

  expand(shape: unknown) {
    return surface().tensorViewSurfaceHelpers.expand(this, shape);
  }

  expand_as(other: unknown) {
    if (!other || typeof other !== "object" || !Array.isArray((other as { shape?: unknown }).shape)) {
      throw new Error("Tensor.expand_as requires a tensor-like shape source");
    }
    return this.expand((other as { shape: readonly number[] }).shape);
  }

  repeat(...repeats: unknown[]) {
    return surface().tensorViewSurfaceHelpers.repeat(this, repeats.length === 1 ? repeats[0] : repeats);
  }

  tile(...repeats: unknown[]) {
    return surface().tensorViewSurfaceHelpers.tile(this, repeats.length === 1 ? repeats[0] : repeats);
  }

  flatten(startDim: unknown = 0, endDim: unknown = -1) {
    return surface().tensorViewSurfaceHelpers.flatten(this, startDim, endDim);
  }

  squeeze(dim: unknown = null) {
    return surface().tensorViewSurfaceHelpers.squeeze(this, dim);
  }

  unsqueeze(dim: unknown) {
    return surface().tensorViewSurfaceHelpers.unsqueeze(this, dim);
  }

  transpose(dim0: unknown = 0, dim1: unknown = 1) {
    return surface().tensorViewSurfaceHelpers.transpose(this, dim0, dim1);
  }

  permute(dims: unknown) {
    return surface().tensorViewSurfaceHelpers.permute(this, dims);
  }

  flip(dims: unknown) {
    return surface().tensorViewSurfaceHelpers.flip(this, dims);
  }

  roll(shifts: unknown, dims: unknown = null) {
    return surface().tensorViewSurfaceHelpers.roll(this, shifts, dims);
  }

  select(dim: unknown, index: unknown) {
    return surface().tensorViewSurfaceHelpers.select(this, dim, index);
  }

  narrow(dim: unknown, start: unknown, length: unknown) {
    return surface().tensorViewSurfaceHelpers.narrow(this, dim, start, length);
  }

  slice(dim: unknown, start: unknown, end: unknown, step: unknown) {
    return surface().tensorViewSurfaceHelpers.slice(this, dim, start, end, step);
  }

  indexSelect(dim: unknown, indices: unknown) {
    return surface().tensorViewSurfaceHelpers.indexSelect(this, dim, indices);
  }

  index_select(dim: unknown, indices: unknown) {
    return surface().tensorViewSurfaceHelpers.index_select(this, dim, indices);
  }

  gather(dim: unknown, index: unknown) {
    return surface().tensorViewSurfaceHelpers.gather(this, dim, index);
  }

  take(index: unknown) {
    return surface().tensorViewSurfaceHelpers.take(this, index);
  }

  argsort(dim: unknown = -1, descending: unknown = false) {
    return surface().tensorViewSurfaceHelpers.argsort(this, dim, descending);
  }

  sort(dim: unknown = -1, descending: unknown = false) {
    return surface().tensorViewSurfaceHelpers.sort(this, dim, descending);
  }

  topk(k: unknown, dim: unknown = -1, largest: unknown = true, sorted: unknown = true) {
    return surface().tensorViewSurfaceHelpers.topk(this, k, dim, largest, sorted);
  }

  scatterAdd(dim: unknown, index: unknown, src: unknown) {
    return surface().tensorViewSurfaceHelpers.scatterAdd(this, dim, index, src);
  }

  scatter_add(dim: unknown, index: unknown, src: unknown) {
    return surface().tensorViewSurfaceHelpers.scatter_add(this, dim, index, src);
  }

  split(splitSizeOrSections: unknown, dim: unknown = 0) {
    return surface().tensorViewSurfaceHelpers.split(this, splitSizeOrSections, dim);
  }

  chunk(chunks: unknown, dim: unknown = 0) {
    return surface().tensorViewSurfaceHelpers.chunk(this, chunks, dim);
  }

  unbind(dim: unknown = 0) {
    return surface().tensorViewSurfaceHelpers.unbind(this, dim);
  }

  get T() {
    return surface().tensorViewSurfaceHelpers.T(this);
  }

  get mT() {
    return surface().tensorViewSurfaceHelpers.mT(this);
  }

  toJSON() {
    return surface().tensorInfoSurfaceHelpers.toJSON(this);
  }

  static fromJSON(value: unknown) {
    return surface().tensorStaticHelpers.fromJSON(value);
  }

  static fromNativeBuffer(buffer: unknown, shape: unknown, options: unknown = {}) {
    return surface().tensorStaticHelpers.fromNativeBuffer(buffer, shape, options);
  }

  static tensor(data: unknown, shape: unknown, options: unknown) {
    return surface().tensorStaticHelpers.tensor(data, shape, options);
  }

  static parameter(data: unknown, shape: unknown, options: unknown) {
    return surface().tensorStaticHelpers.parameter(data, shape, options);
  }

  static param(data: unknown, shape: unknown, options: unknown) {
    return surface().tensorStaticHelpers.param(data, shape, options);
  }

  static scalar(value: unknown, options: unknown) {
    return surface().tensorStaticHelpers.scalar(value, options);
  }

  static empty(shape: unknown, options: unknown) {
    return surface().tensorStaticHelpers.empty(shape, options);
  }

  static emptyLike(input: unknown, options: unknown) {
    return surface().tensorStaticHelpers.emptyLike(input, options);
  }

  static empty_like(input: unknown, options: unknown) {
    return surface().tensorStaticHelpers.empty_like(input, options);
  }

  static zeros(shape: unknown, options: unknown) {
    return surface().tensorStaticHelpers.zeros(shape, options);
  }

  static zerosLike(input: unknown, options: unknown) {
    return surface().tensorStaticHelpers.zerosLike(input, options);
  }

  static zeros_like(input: unknown, options: unknown) {
    return surface().tensorStaticHelpers.zeros_like(input, options);
  }

  static ones(shape: unknown, options: unknown) {
    return surface().tensorStaticHelpers.ones(shape, options);
  }

  static onesLike(input: unknown, options: unknown) {
    return surface().tensorStaticHelpers.onesLike(input, options);
  }

  static ones_like(input: unknown, options: unknown) {
    return surface().tensorStaticHelpers.ones_like(input, options);
  }

  static eye(size: unknown, options: unknown) {
    return surface().tensorStaticHelpers.eye(size, options);
  }

  static full(shape: unknown, value: unknown, options: unknown) {
    return surface().tensorStaticHelpers.full(shape, value, options);
  }

  static fullLike(input: unknown, value: unknown, options: unknown) {
    return surface().tensorStaticHelpers.fullLike(input, value, options);
  }

  static full_like(input: unknown, value: unknown, options: unknown) {
    return surface().tensorStaticHelpers.full_like(input, value, options);
  }

  static hasShape(input: unknown, shape: unknown) {
    return surface().tensorStaticHelpers.hasShape(input, shape);
  }

  static requireShape(input: unknown, shape: unknown) {
    return surface().tensorStaticHelpers.requireShape(input, shape);
  }

  static rand(shape: unknown, options: unknown) {
    return surface().tensorStaticHelpers.rand(shape, options);
  }

  static randLike(input: unknown, options: unknown) {
    return surface().tensorStaticHelpers.randLike(input, options);
  }

  static rand_like(input: unknown, options: unknown) {
    return surface().tensorStaticHelpers.rand_like(input, options);
  }

  static randn(shape: unknown, options: unknown) {
    return surface().tensorStaticHelpers.randn(shape, options);
  }

  static randnLike(input: unknown, options: unknown) {
    return surface().tensorStaticHelpers.randnLike(input, options);
  }

  static randn_like(input: unknown, options: unknown) {
    return surface().tensorStaticHelpers.randn_like(input, options);
  }

  static randInt(first: unknown, second: unknown, third: unknown, fourth: unknown) {
    return surface().tensorStaticHelpers.randInt(first as number, second, third, fourth as Record<string, unknown>);
  }

  static randint(first: unknown, second: unknown, third: unknown, fourth: unknown) {
    return surface().tensorStaticHelpers.randint(first as number, second, third, fourth as Record<string, unknown>);
  }

  static randPerm(size: unknown, options: unknown) {
    return surface().tensorStaticHelpers.randPerm(size as number, options as Record<string, unknown>);
  }

  static randperm(size: unknown, options: unknown) {
    return surface().tensorStaticHelpers.randperm(size as number, options as Record<string, unknown>);
  }

  static manualSeed(seed: unknown) {
    return surface().tensorStaticHelpers.manualSeed(seed);
  }

  static manual_seed(seed: unknown) {
    return surface().tensorStaticHelpers.manual_seed(seed);
  }

  static initialSeed() {
    return surface().tensorStaticHelpers.initialSeed();
  }

  static initial_seed() {
    return surface().tensorStaticHelpers.initial_seed();
  }

  static seededRng(seed: unknown) {
    return surface().tensorStaticHelpers.seededRng(seed);
  }

  static linspace(first: unknown, second: unknown, third: unknown, fourth: unknown) {
    return surface().tensorStaticHelpers.linspace(first, second, third, fourth);
  }

  static arange(start: unknown, end: unknown, stepOrOptions: unknown, maybeOptions: unknown) {
    return surface().tensorStaticHelpers.arange(start, end, stepOrOptions, maybeOptions);
  }

  static allclose(actual: unknown, expected: unknown, options: unknown) {
    return surface().tensorStaticHelpers.allclose(actual, expected, options);
  }

  static equal(actual: unknown, expected: unknown) {
    return surface().tensorStaticHelpers.equal(actual, expected);
  }

  static isclose(input: unknown, other: unknown, options: unknown) {
    return surface().tensorStaticHelpers.isclose(input, other, options);
  }

  static to(input: unknown, target: unknown, options: unknown) {
    return surface().tensorStaticHelpers.to(input, target, options);
  }

  static typeAs(input: unknown, other: unknown, options: unknown) {
    return surface().tensorStaticHelpers.typeAs(input, other, options);
  }

  static type_as(input: unknown, other: unknown, options: unknown) {
    return surface().tensorStaticHelpers.type_as(input, other, options);
  }

  static clone(input: unknown) {
    return surface().tensorStaticHelpers.clone(input);
  }

  static detach(input: unknown) {
    return surface().tensorStaticHelpers.detach(input);
  }

  static reshape(input: unknown, shape: unknown) {
    return surface().tensorStaticHelpers.reshape(input, shape);
  }

  static view(input: unknown, shape: unknown) {
    return surface().tensorStaticHelpers.view(input, shape);
  }

  static broadcastTo(input: unknown, shape: unknown) {
    return surface().tensorStaticHelpers.broadcastTo(input, shape);
  }

  static expand(input: unknown, shape: unknown) {
    return surface().tensorStaticHelpers.expand(input, shape);
  }

  static repeat(input: unknown, ...repeats: unknown[]) {
    return surface().tensorStaticHelpers.repeat(input, repeats.length === 1 ? repeats[0] : repeats);
  }

  static tile(input: unknown, ...repeats: unknown[]) {
    return surface().tensorStaticHelpers.tile(input, repeats.length === 1 ? repeats[0] : repeats);
  }

  static flatten(input: unknown, startDim: unknown, endDim: unknown) {
    return surface().tensorStaticHelpers.flatten(input, startDim, endDim);
  }

  static squeeze(input: unknown, dim: unknown) {
    return surface().tensorStaticHelpers.squeeze(input, dim);
  }

  static unsqueeze(input: unknown, dim: unknown) {
    return surface().tensorStaticHelpers.unsqueeze(input, dim);
  }

  static transpose(input: unknown, dim0: unknown, dim1: unknown) {
    return surface().tensorStaticHelpers.transpose(input, dim0, dim1);
  }

  static permute(input: unknown, dims: unknown) {
    return surface().tensorStaticHelpers.permute(input, dims);
  }

  static flip(input: unknown, dims: unknown) {
    return surface().tensorStaticHelpers.flip(input, dims);
  }

  static roll(input: unknown, shifts: unknown, dims: unknown = null) {
    return surface().tensorStaticHelpers.roll(input, shifts, dims);
  }

  static select(input: unknown, dim: unknown, index: unknown) {
    return surface().tensorStaticHelpers.select(input, dim, index);
  }

  static narrow(input: unknown, dim: unknown, start: unknown, length: unknown) {
    return surface().tensorStaticHelpers.narrow(input, dim, start, length);
  }

  static slice(input: unknown, dim: unknown, start: unknown, end: unknown, step: unknown) {
    return surface().tensorStaticHelpers.slice(input, dim, start, end, step);
  }

  static indexSelect(input: unknown, dim: unknown, indices: unknown) {
    return surface().tensorStaticHelpers.indexSelect(input, dim, indices);
  }

  static index_select(input: unknown, dim: unknown, indices: unknown) {
    return surface().tensorStaticHelpers.index_select(input, dim, indices);
  }

  static gather(input: unknown, dim: unknown, index: unknown) {
    return surface().tensorStaticHelpers.gather(input, dim, index);
  }

  static take(input: unknown, index: unknown) {
    return surface().tensorStaticHelpers.take(input, index);
  }

  static argsort(input: unknown, dim: unknown = -1, descending: unknown = false) {
    return surface().tensorStaticHelpers.argsort(input, dim, descending);
  }

  static sort(input: unknown, dim: unknown = -1, descending: unknown = false) {
    return surface().tensorStaticHelpers.sort(input, dim, descending);
  }

  static topk(input: unknown, k: unknown, dim: unknown = -1, largest: unknown = true, sorted: unknown = true) {
    return surface().tensorStaticHelpers.topk(input, k, dim, largest, sorted);
  }

  static scatterAdd(input: unknown, dim: unknown, index: unknown, src: unknown) {
    return surface().tensorStaticHelpers.scatterAdd(input, dim, index, src);
  }

  static scatter_add(input: unknown, dim: unknown, index: unknown, src: unknown) {
    return surface().tensorStaticHelpers.scatter_add(input, dim, index, src);
  }

  static split(input: unknown, splitSizeOrSections: unknown, dim: unknown = 0) {
    return surface().tensorStaticHelpers.split(input, splitSizeOrSections, dim);
  }

  static chunk(input: unknown, chunks: unknown, dim: unknown = 0) {
    return surface().tensorStaticHelpers.chunk(input, chunks, dim);
  }

  static unbind(input: unknown, dim: unknown = 0) {
    return surface().tensorStaticHelpers.unbind(input, dim);
  }

  static add(input: unknown, other: unknown) {
    return surface().tensorStaticHelpers.add(input, other);
  }

  static sub(input: unknown, other: unknown) {
    return surface().tensorStaticHelpers.sub(input, other);
  }

  static mul(input: unknown, other: unknown) {
    return surface().tensorStaticHelpers.mul(input, other);
  }

  static div(input: unknown, other: unknown) {
    return surface().tensorStaticHelpers.div(input, other);
  }

  static eq(input: unknown, other: unknown) {
    return surface().tensorStaticHelpers.eq(input, other);
  }

  static ne(input: unknown, other: unknown) {
    return surface().tensorStaticHelpers.ne(input, other);
  }

  static lt(input: unknown, other: unknown) {
    return surface().tensorStaticHelpers.lt(input, other);
  }

  static le(input: unknown, other: unknown) {
    return surface().tensorStaticHelpers.le(input, other);
  }

  static gt(input: unknown, other: unknown) {
    return surface().tensorStaticHelpers.gt(input, other);
  }

  static ge(input: unknown, other: unknown) {
    return surface().tensorStaticHelpers.ge(input, other);
  }

  static pow(input: unknown, exponent: unknown) {
    return surface().tensorStaticHelpers.pow(input, exponent);
  }

  static neg(input: unknown) {
    return surface().tensorStaticHelpers.neg(input);
  }

  static negative(input: unknown) {
    return surface().tensorStaticHelpers.negative(input);
  }

  static exp(input: unknown) {
    return surface().tensorStaticHelpers.exp(input);
  }

  static expm1(input: unknown) {
    return surface().tensorStaticHelpers.expm1(input);
  }

  static log(input: unknown) {
    return surface().tensorStaticHelpers.log(input);
  }

  static log1p(input: unknown) {
    return surface().tensorStaticHelpers.log1p(input);
  }

  static sqr(input: unknown) {
    return surface().tensorStaticHelpers.sqr(input);
  }

  static square(input: unknown) {
    return surface().tensorStaticHelpers.square(input);
  }

  static recip(input: unknown) {
    return surface().tensorStaticHelpers.recip(input);
  }

  static reciprocal(input: unknown) {
    return surface().tensorStaticHelpers.reciprocal(input);
  }

  static abs(input: unknown) {
    return surface().tensorStaticHelpers.abs(input);
  }

  static sgn(input: unknown) {
    return surface().tensorStaticHelpers.sgn(input);
  }

  static sign(input: unknown) {
    return surface().tensorStaticHelpers.sign(input);
  }

  static step(input: unknown) {
    return surface().tensorStaticHelpers.step(input);
  }

  static isnan(input: unknown) {
    return surface().tensorStaticHelpers.isnan(input);
  }

  static isinf(input: unknown) {
    return surface().tensorStaticHelpers.isinf(input);
  }

  static isfinite(input: unknown) {
    return surface().tensorStaticHelpers.isfinite(input);
  }

  static floor(input: unknown) {
    return surface().tensorStaticHelpers.floor(input);
  }

  static ceil(input: unknown) {
    return surface().tensorStaticHelpers.ceil(input);
  }

  static round(input: unknown) {
    return surface().tensorStaticHelpers.round(input);
  }

  static trunc(input: unknown) {
    return surface().tensorStaticHelpers.trunc(input);
  }

  static sqrt(input: unknown) {
    return surface().tensorStaticHelpers.sqrt(input);
  }

  static rsqrt(input: unknown) {
    return surface().tensorStaticHelpers.rsqrt(input);
  }

  static relu(input: unknown) {
    return surface().tensorStaticHelpers.relu(input);
  }

  static gelu(input: unknown) {
    return surface().tensorStaticHelpers.gelu(input);
  }

  static silu(input: unknown) {
    return surface().tensorStaticHelpers.silu(input);
  }

  static sigmoid(input: unknown) {
    return surface().tensorStaticHelpers.sigmoid(input);
  }

  static tanh(input: unknown) {
    return surface().tensorStaticHelpers.tanh(input);
  }

  static sin(input: unknown) {
    return surface().tensorStaticHelpers.sin(input);
  }

  static cos(input: unknown) {
    return surface().tensorStaticHelpers.cos(input);
  }

  static tan(input: unknown) {
    return surface().tensorStaticHelpers.tan(input);
  }

  static maximum(input: unknown, other: unknown) {
    return surface().tensorStaticHelpers.maximum(input, other);
  }

  static minimum(input: unknown, other: unknown) {
    return surface().tensorStaticHelpers.minimum(input, other);
  }

  static where(condition: unknown, input: unknown, other: unknown) {
    return surface().tensorStaticHelpers.where(condition, input, other);
  }

  static maskedFill(input: unknown, mask: unknown, value: unknown) {
    return surface().tensorStaticHelpers.maskedFill(input, mask, value);
  }

  static masked_fill(input: unknown, mask: unknown, value: unknown) {
    return surface().tensorStaticHelpers.masked_fill(input, mask, value);
  }

  static sum(input: unknown, dim: unknown) {
    return surface().tensorStaticHelpers.sum(input, dim);
  }

  static prod(input: unknown, dim?: unknown) {
    return surface().tensorStaticHelpers.prod(input, dim as number | undefined);
  }

  static cumsum(input: unknown, dim?: unknown) {
    return surface().tensorStaticHelpers.cumsum(input, dim as number | undefined);
  }

  static mean(input: unknown, dim: unknown) {
    return surface().tensorStaticHelpers.mean(input, dim);
  }

  static max(input: unknown, dim: unknown) {
    return surface().tensorStaticHelpers.max(input, dim);
  }

  static min(input: unknown, dim: unknown) {
    return surface().tensorStaticHelpers.min(input, dim);
  }

  static any(input: unknown, dim: unknown) {
    return surface().tensorStaticHelpers.any(input, dim);
  }

  static all(input: unknown, dim: unknown) {
    return surface().tensorStaticHelpers.all(input, dim);
  }

  static argmax(input: unknown, dim: unknown) {
    return surface().tensorStaticHelpers.argmax(input, dim);
  }

  static argmin(input: unknown, dim: unknown) {
    return surface().tensorStaticHelpers.argmin(input, dim);
  }

  static variance(input: unknown, dim?: unknown, correction?: unknown) {
    return surface().tensorStaticHelpers.variance(input, dim as number | undefined, correction as number | undefined);
  }

  static var(input: unknown, dim?: unknown, correction?: unknown) {
    return surface().tensorStaticHelpers.var(input, dim as number | undefined, correction as number | undefined);
  }

  static std(input: unknown, dim?: unknown, correction?: unknown) {
    return surface().tensorStaticHelpers.std(input, dim as number | undefined, correction as number | undefined);
  }

  static norm(input: unknown, dim?: unknown, p?: unknown) {
    return surface().tensorStaticHelpers.norm(input, dim as number | undefined, p as number | undefined);
  }

  static softmax(input: unknown, dim?: unknown) {
    return surface().tensorStaticHelpers.softmax(input, dim as number | undefined);
  }

  static softmax_dim(input: unknown, dim?: unknown) {
    return surface().tensorStaticHelpers.softmax_dim(input, dim as number | undefined);
  }

  static softmaxDim(input: unknown, dim: unknown) {
    return surface().tensorStaticHelpers.softmaxDim(input, dim);
  }

  static logSoftmax(input: unknown, dim?: unknown) {
    return surface().tensorStaticHelpers.logSoftmax(input, dim as number | undefined);
  }

  static log_softmax(input: unknown, dim?: unknown) {
    return surface().tensorStaticHelpers.log_softmax(input, dim as number | undefined);
  }

  static log_softmax_dim(input: unknown, dim?: unknown) {
    return surface().tensorStaticHelpers.log_softmax_dim(input, dim as number | undefined);
  }

  static logSoftmaxDim(input: unknown, dim: unknown) {
    return surface().tensorStaticHelpers.logSoftmaxDim(input, dim);
  }

  static logsumexp(input: unknown, dim?: unknown) {
    return surface().tensorStaticHelpers.logsumexp(input, dim as number | undefined);
  }

  static logSumExp(input: unknown, dim?: unknown) {
    return surface().tensorStaticHelpers.logSumExp(input, dim as number | undefined);
  }

  static clamp(input: unknown, min: unknown, max: unknown) {
    return surface().tensorStaticHelpers.clamp(input, min, max);
  }

  static clip(input: unknown, min: unknown, max: unknown) {
    return surface().tensorStaticHelpers.clip(input, min, max);
  }

  static matmul(input: unknown, other: unknown, otherShape: unknown) {
    return surface().tensorStaticHelpers.matmul(input, other, otherShape);
  }

  static mm(input: unknown, other: unknown, otherShape: unknown) {
    return surface().tensorStaticHelpers.mm(input, other, otherShape);
  }

  static dot(input: unknown, other: unknown, otherShape: unknown) {
    return surface().tensorStaticHelpers.dot(input, other, otherShape);
  }

  static trace(input: unknown) {
    return surface().tensorStaticHelpers.trace(input);
  }

  static diagonal(input: unknown) {
    return surface().tensorStaticHelpers.diagonal(input);
  }

  static bmm(input: unknown, other: unknown, otherShape: unknown) {
    return surface().tensorStaticHelpers.bmm(input, other, otherShape);
  }

  static cat(tensors: unknown, dim: unknown = 0) {
    return surface().tensorStaticHelpers.cat(tensors, dim);
  }

  static concat(tensors: unknown, dim: unknown = 0) {
    return surface().tensorStaticHelpers.concat(tensors, dim);
  }

  static concatenate(tensors: unknown, dim: unknown = 0) {
    return surface().tensorStaticHelpers.concatenate(tensors, dim);
  }

  static stack(tensors: unknown, dim: unknown = 0) {
    return surface().tensorStaticHelpers.stack(tensors, dim);
  }

  static vstack(tensors: unknown) {
    return surface().tensorStaticHelpers.vstack(tensors);
  }

  static hstack(tensors: unknown) {
    return surface().tensorStaticHelpers.hstack(tensors);
  }

  binary(other: unknown, op: unknown, gradLeft: unknown, gradRight: unknown, label: unknown) {
    return surface().tensorMathSurfaceHelpers.binary(this, other, op, gradLeft, gradRight, label);
  }

  add(other: unknown) {
    return surface().tensorMathSurfaceHelpers.add(this, other);
  }

  sub(other: unknown) {
    return surface().tensorMathSurfaceHelpers.sub(this, other);
  }

  mul(other: unknown) {
    return surface().tensorMathSurfaceHelpers.mul(this, other);
  }

  div(other: unknown) {
    return surface().tensorMathSurfaceHelpers.div(this, other);
  }

  add_(other: unknown) {
    return this.copy_(requireSameShapeForInPlace(this, this.add(other) as Tensor, "Tensor.add_"));
  }

  sub_(other: unknown) {
    return this.copy_(requireSameShapeForInPlace(this, this.sub(other) as Tensor, "Tensor.sub_"));
  }

  mul_(other: unknown) {
    return this.copy_(requireSameShapeForInPlace(this, this.mul(other) as Tensor, "Tensor.mul_"));
  }

  div_(other: unknown) {
    return this.copy_(requireSameShapeForInPlace(this, this.div(other) as Tensor, "Tensor.div_"));
  }

  maximum(other: unknown) {
    return surface().tensorMathSurfaceHelpers.maximum(this, other);
  }

  minimum(other: unknown) {
    return surface().tensorMathSurfaceHelpers.minimum(this, other);
  }

  where(input: unknown, other: unknown) {
    return surface().tensorMathSurfaceHelpers.where(this, input, other);
  }

  maskedFill(mask: unknown, value: unknown) {
    return surface().tensorMathSurfaceHelpers.maskedFill(this, mask, value);
  }

  masked_fill(mask: unknown, value: unknown) {
    return surface().tensorMathSurfaceHelpers.masked_fill(this, mask, value);
  }

  eq(other: unknown) {
    return surface().tensorMathSurfaceHelpers.eq(this, other);
  }

  ne(other: unknown) {
    return surface().tensorMathSurfaceHelpers.ne(this, other);
  }

  lt(other: unknown) {
    return surface().tensorMathSurfaceHelpers.lt(this, other);
  }

  le(other: unknown) {
    return surface().tensorMathSurfaceHelpers.le(this, other);
  }

  gt(other: unknown) {
    return surface().tensorMathSurfaceHelpers.gt(this, other);
  }

  ge(other: unknown) {
    return surface().tensorMathSurfaceHelpers.ge(this, other);
  }

  isclose(other: unknown, options: unknown) {
    return surface().tensorMathSurfaceHelpers.isclose(this, other, options);
  }

  sqr() {
    return surface().tensorMathSurfaceHelpers.sqr(this);
  }

  square() {
    return surface().tensorMathSurfaceHelpers.square(this);
  }

  pow(exponent: unknown) {
    return surface().tensorMathSurfaceHelpers.pow(this, exponent);
  }

  recip() {
    return surface().tensorMathSurfaceHelpers.recip(this);
  }

  reciprocal() {
    return surface().tensorMathSurfaceHelpers.reciprocal(this);
  }

  abs() {
    return surface().tensorMathSurfaceHelpers.abs(this);
  }

  sgn() {
    return surface().tensorMathSurfaceHelpers.sgn(this);
  }

  sign() {
    return surface().tensorMathSurfaceHelpers.sign(this);
  }

  step() {
    return surface().tensorMathSurfaceHelpers.step(this);
  }

  isnan() {
    return surface().tensorMathSurfaceHelpers.isnan(this);
  }

  isinf() {
    return surface().tensorMathSurfaceHelpers.isinf(this);
  }

  isfinite() {
    return surface().tensorMathSurfaceHelpers.isfinite(this);
  }

  floor() {
    return surface().tensorMathSurfaceHelpers.floor(this);
  }

  ceil() {
    return surface().tensorMathSurfaceHelpers.ceil(this);
  }

  sqrt() {
    return surface().tensorMathSurfaceHelpers.sqrt(this);
  }

  clamp(min: unknown, max: unknown) {
    return surface().tensorMathSurfaceHelpers.clamp(this, min, max);
  }

  clip(min: unknown, max: unknown) {
    return surface().tensorMathSurfaceHelpers.clip(this, min, max);
  }

  matmul(other: unknown, otherShape: unknown) {
    return surface().tensorMathSurfaceHelpers.matmul(this, other, otherShape);
  }

  mm(other: unknown, otherShape: unknown) {
    return surface().tensorMathSurfaceHelpers.mm(this, other, otherShape);
  }

  dot(other: unknown, otherShape: unknown) {
    return surface().tensorMathSurfaceHelpers.dot(this, other, otherShape);
  }

  trace() {
    return surface().tensorMathSurfaceHelpers.trace(this);
  }

  diagonal() {
    return surface().tensorMathSurfaceHelpers.diagonal(this);
  }

  bmm(other: unknown, otherShape: unknown) {
    return surface().tensorMathSurfaceHelpers.bmm(this, other, otherShape);
  }

  map(fn: unknown) {
    return surface().tensorMathSurfaceHelpers.map(this, fn);
  }

  unary(fn: unknown, derivative: unknown) {
    return surface().tensorMathSurfaceHelpers.unary(this, fn, derivative);
  }

  relu() {
    return surface().tensorMathSurfaceHelpers.relu(this);
  }

  gelu() {
    return surface().tensorMathSurfaceHelpers.gelu(this);
  }

  silu() {
    return surface().tensorMathSurfaceHelpers.silu(this);
  }

  sigmoid() {
    return surface().tensorMathSurfaceHelpers.sigmoid(this);
  }

  tanh() {
    return surface().tensorMathSurfaceHelpers.tanh(this);
  }

  sin() {
    return surface().tensorMathSurfaceHelpers.sin(this);
  }

  cos() {
    return surface().tensorMathSurfaceHelpers.cos(this);
  }

  tan() {
    return surface().tensorMathSurfaceHelpers.tan(this);
  }

  round() {
    return surface().tensorMathSurfaceHelpers.round(this);
  }

  trunc() {
    return surface().tensorMathSurfaceHelpers.trunc(this);
  }

  exp() {
    return surface().tensorMathSurfaceHelpers.exp(this);
  }

  expm1() {
    return surface().tensorMathSurfaceHelpers.expm1(this);
  }

  log() {
    return surface().tensorMathSurfaceHelpers.log(this);
  }

  log1p() {
    return surface().tensorMathSurfaceHelpers.log1p(this);
  }

  rsqrt() {
    return surface().tensorMathSurfaceHelpers.rsqrt(this);
  }

  neg() {
    return surface().tensorMathSurfaceHelpers.neg(this);
  }

  negative() {
    return surface().tensorMathSurfaceHelpers.negative(this);
  }

  sumDim(dim: unknown) {
    return surface().tensorMathSurfaceHelpers.sumDim(this, dim);
  }

  meanDim(dim: unknown) {
    return surface().tensorMathSurfaceHelpers.meanDim(this, dim);
  }

  prodDim(dim: unknown) {
    return surface().tensorMathSurfaceHelpers.prodDim(this, dim);
  }

  maxDim(dim: unknown) {
    return surface().tensorMathSurfaceHelpers.maxDim(this, dim);
  }

  minDim(dim: unknown) {
    return surface().tensorMathSurfaceHelpers.minDim(this, dim);
  }

  argmaxDim(dim: unknown) {
    return surface().tensorMathSurfaceHelpers.argmaxDim(this, dim);
  }

  argminDim(dim: unknown) {
    return surface().tensorMathSurfaceHelpers.argminDim(this, dim);
  }

  variance(dim?: unknown, correction?: unknown) {
    return surface().tensorMathSurfaceHelpers.variance(this, dim as number | undefined, correction as number | undefined);
  }

  var(dim?: unknown, correction?: unknown) {
    return surface().tensorMathSurfaceHelpers.var(this, dim as number | undefined, correction as number | undefined);
  }

  std(dim?: unknown, correction?: unknown) {
    return surface().tensorMathSurfaceHelpers.std(this, dim as number | undefined, correction as number | undefined);
  }

  norm(dim?: unknown, p?: unknown) {
    return surface().tensorMathSurfaceHelpers.norm(this, dim as number | undefined, p as number | undefined);
  }

  anyDim(dim: unknown) {
    return surface().tensorMathSurfaceHelpers.anyDim(this, dim);
  }

  allDim(dim: unknown) {
    return surface().tensorMathSurfaceHelpers.allDim(this, dim);
  }

  softmax(dim?: unknown) {
    return dim === undefined
      ? surface().tensorMathSurfaceHelpers.softmax(this)
      : surface().tensorMathSurfaceHelpers.softmaxDim(this, dim as number);
  }

  softmax_dim(dim?: unknown) {
    return dim === undefined
      ? surface().tensorMathSurfaceHelpers.softmax(this)
      : surface().tensorMathSurfaceHelpers.softmaxDim(this, dim as number);
  }

  softmaxDim(dim: unknown) {
    return surface().tensorMathSurfaceHelpers.softmaxDim(this, dim);
  }

  logSoftmax(dim?: unknown) {
    return dim === undefined
      ? surface().tensorMathSurfaceHelpers.logSoftmax(this)
      : surface().tensorMathSurfaceHelpers.logSoftmaxDim(this, dim as number);
  }

  log_softmax(dim?: unknown) {
    return dim === undefined
      ? surface().tensorMathSurfaceHelpers.logSoftmax(this)
      : surface().tensorMathSurfaceHelpers.logSoftmaxDim(this, dim as number);
  }

  log_softmax_dim(dim?: unknown) {
    return dim === undefined
      ? surface().tensorMathSurfaceHelpers.logSoftmax(this)
      : surface().tensorMathSurfaceHelpers.logSoftmaxDim(this, dim as number);
  }

  logSoftmaxDim(dim: unknown) {
    return surface().tensorMathSurfaceHelpers.logSoftmaxDim(this, dim);
  }

  logsumexp(dim?: unknown) {
    return surface().tensorMathSurfaceHelpers.logsumexp(this, dim as number | undefined);
  }

  logSumExp(dim?: unknown) {
    return surface().tensorMathSurfaceHelpers.logSumExp(this, dim as number | undefined);
  }

  sum(dim?: unknown) {
    return surface().tensorMathSurfaceHelpers.sum(this, dim);
  }

  prod(dim?: unknown) {
    return surface().tensorMathSurfaceHelpers.prod(this, dim as number | undefined);
  }

  cumsum(dim?: unknown) {
    return surface().tensorMathSurfaceHelpers.cumsum(this, dim as number | undefined);
  }

  max(dim?: unknown) {
    return surface().tensorMathSurfaceHelpers.max(this, dim);
  }

  min(dim?: unknown) {
    return surface().tensorMathSurfaceHelpers.min(this, dim);
  }

  any(dim?: unknown) {
    return surface().tensorMathSurfaceHelpers.any(this, dim);
  }

  all(dim?: unknown) {
    return surface().tensorMathSurfaceHelpers.all(this, dim);
  }

  mean(dim?: unknown) {
    return surface().tensorMathSurfaceHelpers.mean(this, dim);
  }

  argmax(dim?: unknown) {
    return surface().tensorMathSurfaceHelpers.argmax(this, dim);
  }

  argmin(dim?: unknown) {
    return surface().tensorMathSurfaceHelpers.argmin(this, dim);
  }

  meanSquaredError(target: unknown) {
    return surface().tensorMathSurfaceHelpers.meanSquaredError(this, target);
  }

  [Symbol.iterator]() {
    return surface().tensorInfoSurfaceHelpers.iterator(this);
  }
}
