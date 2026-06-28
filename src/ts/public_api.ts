type FrontendManifestContract = typeof import("./frontend_manifest.js").frontendManifest;

export type PublicApiContractManifest = Readonly<{
  kind: "zgml-public-api-contract";
  source: FrontendManifestContract["source"];
  policyOwner: "src/ts/public_api.ts";
  productSourceOfTruth: FrontendManifestContract["productSourceOfTruth"];
  productSemanticsOwner: FrontendManifestContract["productSemanticsOwner"];
  packageFanout: FrontendManifestContract["packageFanout"];
  packageTypesRoot: "dist/public_api.d.cts";
  runtimePath: FrontendManifestContract["runtimePath"];
  nativeProductPolicy: FrontendManifestContract["nativeProductPolicy"];
  declarationMirror: false;
}>;

export type TensorNestedArray = readonly (number | TensorNestedArray)[];
export type TensorData = number | Float32Array | TensorNestedArray;
export type TensorJSON = {
  dtype: "f32";
  shape: readonly number[];
  data: readonly number[];
  requiresGrad?: boolean;
  requires_grad?: boolean;
};
export type TensorShapeTuple = readonly number[];
export type TensorLike<Shape extends TensorShapeTuple = TensorShapeTuple> = TensorData | Tensor<Shape>;
export type TensorLikeShape<Input> =
  Input extends Tensor<infer Shape extends TensorShapeTuple>
    ? Shape
    : Input extends number
      ? readonly [1]
      : Input extends Float32Array
        ? readonly [number]
        : Input extends readonly []
          ? readonly [0]
          : Input extends readonly (infer Element)[]
            ? Element extends number
              ? readonly [Input["length"]]
              : Element extends readonly unknown[]
                ? readonly [Input["length"], ...TensorLikeShape<Element>]
                : TensorShapeTuple
            : TensorShapeTuple;
export type TensorSortResult<Shape extends TensorShapeTuple = TensorShapeTuple> = Readonly<{
  values: Tensor<Shape>;
  indices: Tensor<Shape>;
}>;
export type TensorTopkResult<Shape extends TensorShapeTuple = TensorShapeTuple> = TensorSortResult<Shape>;
export type TensorShape = number | TensorShapeTuple;
export type TensorRollArg = number | readonly number[];
export type TensorShapeOf<S extends TensorShape> = S extends number
  ? readonly [S]
  : S extends TensorShapeTuple
    ? S
    : TensorShapeTuple;
type TupleOfLength<N extends number, Items extends readonly unknown[] = readonly []> =
  number extends N
    ? readonly unknown[]
    : `${N}` extends `-${string}` | `${string}.${string}`
      ? readonly unknown[]
    : Items["length"] extends N
      ? Items
      : TupleOfLength<N, readonly [...Items, unknown]>;
type AddNumbers<A extends number, B extends number> =
  number extends A | B
    ? number
    : [...TupleOfLength<A>, ...TupleOfLength<B>]["length"] extends infer Sum extends number
      ? Sum
      : number;
type MultiplyNumbers<A extends number, B extends number, Acc extends readonly unknown[] = readonly []> =
  number extends A | B
    ? number
    : TupleOfLength<B> extends readonly [unknown, ...infer Rest extends readonly unknown[]]
      ? MultiplyNumbers<A, Rest["length"], readonly [...Acc, ...TupleOfLength<A>]>
      : Acc["length"];
type MultiplyThree<A extends number, B extends number, C extends number> =
  MultiplyNumbers<A, B> extends infer AB extends number
    ? MultiplyNumbers<AB, C>
    : number;
type MultiplyFour<A extends number, B extends number, C extends number, D extends number> =
  MultiplyThree<A, B, C> extends infer ABC extends number
    ? MultiplyNumbers<ABC, D>
    : number;
type DivideNumbers<A extends number, B extends number, Count extends readonly unknown[] = readonly []> =
  number extends A | B
    ? number
    : `${A}` | `${B}` extends `-${string}` | `${string}.${string}`
      ? number
    : B extends 0
      ? number
      : A extends 0
        ? Count["length"]
        : TupleOfLength<A> extends readonly [...TupleOfLength<B>, ...infer Rest extends readonly unknown[]]
          ? DivideNumbers<Rest["length"], B, readonly [...Count, unknown]>
          : number;
type ArangeLength<N extends number> =
  number extends N
    ? number
    : `${N}` extends `-${string}` | `${string}.${string}`
      ? number
      : N;
type SubtractNumbers<A extends number, B extends number> =
  number extends A | B
    ? number
    : `${A}` | `${B}` extends `-${string}` | `${string}.${string}`
      ? number
      : TupleOfLength<A> extends readonly [...TupleOfLength<B>, ...infer Rest extends readonly unknown[]]
        ? Rest["length"]
        : number;
type MinNumbers<A extends number, B extends number> =
  number extends A | B
    ? number
    : `${A}` | `${B}` extends `-${string}` | `${string}.${string}`
      ? number
      : TupleOfLength<A> extends readonly [...TupleOfLength<B>, ...readonly unknown[]]
        ? B
        : A;
export type ArangeShape<End extends number> = readonly [ArangeLength<End>];
export type ArangeRangeShape<Start extends number, End extends number> = readonly [SubtractNumbers<End, Start>];
type TensorElementCount<Shape extends TensorShapeTuple> =
  Shape extends readonly [infer A extends number]
    ? A
    : Shape extends readonly [infer A extends number, infer B extends number]
      ? MultiplyNumbers<A, B>
      : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number]
        ? MultiplyThree<A, B, C>
        : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number]
          ? MultiplyFour<A, B, C, D>
          : number;
type DivideByProduct<Total extends number, Product extends number> = DivideNumbers<Total, Product>;
type DivideByMultiply<Total extends number, A extends number, B extends number> =
  MultiplyNumbers<A, B> extends infer Product extends number ? DivideByProduct<Total, Product> : number;
type DivideByMultiplyThree<Total extends number, A extends number, B extends number, C extends number> =
  MultiplyThree<A, B, C> extends infer Product extends number ? DivideByProduct<Total, Product> : number;
type ReshapeTupleShape<SourceShape extends TensorShapeTuple, TargetShape extends TensorShapeTuple> =
  TensorElementCount<SourceShape> extends infer Total extends number
    ? TargetShape extends readonly [-1]
      ? readonly [Total]
      : TargetShape extends readonly [infer A extends number]
        ? readonly [A]
        : TargetShape extends readonly [-1, infer B extends number]
          ? readonly [DivideNumbers<Total, B>, B]
          : TargetShape extends readonly [infer A extends number, -1]
            ? readonly [A, DivideNumbers<Total, A>]
            : TargetShape extends readonly [infer A extends number, infer B extends number]
              ? readonly [A, B]
              : TargetShape extends readonly [-1, infer B extends number, infer C extends number]
                ? readonly [DivideByMultiply<Total, B, C>, B, C]
                : TargetShape extends readonly [infer A extends number, -1, infer C extends number]
                  ? readonly [A, DivideByMultiply<Total, A, C>, C]
                  : TargetShape extends readonly [infer A extends number, infer B extends number, -1]
                    ? readonly [A, B, DivideByMultiply<Total, A, B>]
                    : TargetShape extends readonly [infer A extends number, infer B extends number, infer C extends number]
                      ? readonly [A, B, C]
                      : TargetShape extends readonly [-1, infer B extends number, infer C extends number, infer D extends number]
                        ? readonly [DivideByMultiplyThree<Total, B, C, D>, B, C, D]
                        : TargetShape extends readonly [infer A extends number, -1, infer C extends number, infer D extends number]
                          ? readonly [A, DivideByMultiplyThree<Total, A, C, D>, C, D]
                          : TargetShape extends readonly [infer A extends number, infer B extends number, -1, infer D extends number]
                            ? readonly [A, B, DivideByMultiplyThree<Total, A, B, D>, D]
                            : TargetShape extends readonly [infer A extends number, infer B extends number, infer C extends number, -1]
                              ? readonly [A, B, C, DivideByMultiplyThree<Total, A, B, C>]
                              : TargetShape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number]
                                ? readonly [A, B, C, D]
                                : TensorShapeTuple
    : TensorShapeTuple;
export type ReshapeShape<SourceShape extends TensorShapeTuple, TargetShape extends TensorShape> =
  TargetShape extends number
    ? readonly [TargetShape]
    : TargetShape extends TensorShapeTuple
      ? ReshapeTupleShape<SourceShape, TargetShape>
      : TensorShapeTuple;
type BroadcastDim<Lhs extends number, Rhs extends number> =
  Lhs extends Rhs
    ? Lhs
    : Lhs extends 1
      ? Rhs
      : Rhs extends 1
        ? Lhs
        : never;
type BroadcastTuple<Lhs extends TensorShapeTuple, Rhs extends TensorShapeTuple> =
  Lhs extends readonly [infer LHead extends number, ...infer LTail extends number[]]
    ? Rhs extends readonly [infer RHead extends number, ...infer RTail extends number[]]
      ? BroadcastDim<LHead, RHead> extends infer Head extends number
        ? [BroadcastTuple<readonly [...LTail], readonly [...RTail]>] extends [infer Tail extends TensorShapeTuple]
          ? readonly [Head, ...Tail]
          : TensorShapeTuple
        : TensorShapeTuple
      : readonly []
    : Rhs extends readonly []
      ? readonly []
      : TensorShapeTuple;
type PadLeftToRank<Shape extends TensorShapeTuple, Rank extends number> =
  Shape["length"] extends Rank ? Shape : PadLeftToRank<readonly [1, ...Shape], Rank>;
type BroadcastTupleOrFallback<Lhs extends TensorShapeTuple, Rhs extends TensorShapeTuple> =
  [BroadcastTuple<Lhs, Rhs>] extends [infer Out extends TensorShapeTuple] ? Out : TensorShapeTuple;
export type BroadcastShape<Lhs extends TensorShapeTuple, Rhs extends TensorShapeTuple> =
  Lhs extends KnownTensorShapeTuple
    ? Rhs extends KnownTensorShapeTuple
      ? Lhs["length"] extends Rhs["length"]
        ? BroadcastTupleOrFallback<Lhs, Rhs>
        : Lhs["length"] extends 4
          ? BroadcastTupleOrFallback<Lhs, PadLeftToRank<Rhs, 4>>
          : Rhs["length"] extends 4
            ? BroadcastTupleOrFallback<PadLeftToRank<Lhs, 4>, Rhs>
            : Lhs["length"] extends 3
              ? BroadcastTupleOrFallback<Lhs, PadLeftToRank<Rhs, 3>>
              : Rhs["length"] extends 3
                ? BroadcastTupleOrFallback<PadLeftToRank<Lhs, 3>, Rhs>
                : Lhs["length"] extends 2
                  ? BroadcastTupleOrFallback<Lhs, PadLeftToRank<Rhs, 2>>
                  : Rhs["length"] extends 2
                    ? BroadcastTupleOrFallback<PadLeftToRank<Lhs, 2>, Rhs>
                    : TensorShapeTuple
      : TensorShapeTuple
    : TensorShapeTuple;
type RepeatTupleShape<Shape extends TensorShapeTuple, Repeats extends TensorShapeTuple> =
  Repeats extends readonly [infer R0 extends number]
    ? Shape extends readonly [infer A extends number]
      ? readonly [MultiplyNumbers<A, R0>]
      : TensorShapeTuple
    : Repeats extends readonly [infer R0 extends number, infer R1 extends number]
      ? Shape extends readonly [infer A extends number]
        ? readonly [R0, MultiplyNumbers<A, R1>]
        : Shape extends readonly [infer A extends number, infer B extends number]
          ? readonly [MultiplyNumbers<A, R0>, MultiplyNumbers<B, R1>]
          : TensorShapeTuple
      : Repeats extends readonly [infer R0 extends number, infer R1 extends number, infer R2 extends number]
        ? Shape extends readonly [infer A extends number]
          ? readonly [R0, R1, MultiplyNumbers<A, R2>]
          : Shape extends readonly [infer A extends number, infer B extends number]
            ? readonly [R0, MultiplyNumbers<A, R1>, MultiplyNumbers<B, R2>]
            : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number]
              ? readonly [MultiplyNumbers<A, R0>, MultiplyNumbers<B, R1>, MultiplyNumbers<C, R2>]
              : TensorShapeTuple
        : Repeats extends readonly [infer R0 extends number, infer R1 extends number, infer R2 extends number, infer R3 extends number]
          ? Shape extends readonly [infer A extends number]
            ? readonly [R0, R1, R2, MultiplyNumbers<A, R3>]
            : Shape extends readonly [infer A extends number, infer B extends number]
              ? readonly [R0, R1, MultiplyNumbers<A, R2>, MultiplyNumbers<B, R3>]
              : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number]
                ? readonly [R0, MultiplyNumbers<A, R1>, MultiplyNumbers<B, R2>, MultiplyNumbers<C, R3>]
                : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number]
                  ? readonly [MultiplyNumbers<A, R0>, MultiplyNumbers<B, R1>, MultiplyNumbers<C, R2>, MultiplyNumbers<D, R3>]
                  : TensorShapeTuple
          : TensorShapeTuple;
export type RepeatShape<Shape extends TensorShapeTuple, Repeats extends TensorShape> =
  Repeats extends number
    ? Shape extends readonly [infer A extends number] ? readonly [MultiplyNumbers<A, Repeats>] : TensorShapeTuple
    : Repeats extends TensorShapeTuple
      ? RepeatTupleShape<Shape, Repeats>
      : TensorShapeTuple;
export type WhereShape<ConditionShape extends TensorShapeTuple, InputShape extends TensorShapeTuple, OtherShape extends TensorShapeTuple> =
  BroadcastShape<InputShape, OtherShape> extends infer ValueShape extends TensorShapeTuple
    ? BroadcastShape<ConditionShape, ValueShape>
    : TensorShapeTuple;
export type FlattenShape<Shape extends TensorShapeTuple, StartDim extends number = 0, EndDim extends number = -1> =
  Shape extends readonly [infer A extends number]
    ? StartDim extends 0 | -1
      ? EndDim extends 0 | -1
        ? readonly [A]
        : TensorShapeTuple
      : TensorShapeTuple
    : Shape extends readonly [infer A extends number, infer B extends number]
      ? StartDim extends 0 | -2
        ? EndDim extends 1 | -1
          ? readonly [MultiplyNumbers<A, B>]
          : EndDim extends 0 | -2
            ? Shape
            : TensorShapeTuple
        : StartDim extends 1 | -1
          ? EndDim extends 1 | -1
            ? Shape
            : TensorShapeTuple
          : TensorShapeTuple
      : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number]
        ? StartDim extends 0 | -3
          ? EndDim extends 0 | -3
            ? Shape
            : EndDim extends 1 | -2
              ? readonly [MultiplyNumbers<A, B>, C]
              : EndDim extends 2 | -1
                ? readonly [MultiplyThree<A, B, C>]
                : TensorShapeTuple
          : StartDim extends 1 | -2
            ? EndDim extends 1 | -2
              ? Shape
              : EndDim extends 2 | -1
                ? readonly [A, MultiplyNumbers<B, C>]
                : TensorShapeTuple
            : StartDim extends 2 | -1
              ? EndDim extends 2 | -1
                ? Shape
                : TensorShapeTuple
              : TensorShapeTuple
      : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number]
        ? StartDim extends 0 | -4
          ? EndDim extends 0 | -4
            ? Shape
            : EndDim extends 1 | -3
              ? readonly [MultiplyNumbers<A, B>, C, D]
              : EndDim extends 2 | -2
                ? readonly [MultiplyThree<A, B, C>, D]
                : EndDim extends 3 | -1
                  ? readonly [MultiplyFour<A, B, C, D>]
                  : TensorShapeTuple
          : StartDim extends 1 | -3
            ? EndDim extends 1 | -3
              ? Shape
              : EndDim extends 2 | -2
                ? readonly [A, MultiplyNumbers<B, C>, D]
                : EndDim extends 3 | -1
                  ? readonly [A, MultiplyThree<B, C, D>]
                  : TensorShapeTuple
            : StartDim extends 2 | -2
              ? EndDim extends 2 | -2
                ? Shape
                : EndDim extends 3 | -1
                  ? readonly [A, B, MultiplyNumbers<C, D>]
                  : TensorShapeTuple
              : StartDim extends 3 | -1
                ? EndDim extends 3 | -1
                  ? Shape
                  : TensorShapeTuple
                : TensorShapeTuple
      : TensorShapeTuple;
type TensorCatAxisSize<Tensors extends readonly Tensor[], Shape extends TensorShapeTuple, Axis extends number> =
  Tensors["length"] extends 2
    ? MultiplyNumbers<TupleAt<Shape, Axis>, 2> extends infer Size extends number
      ? Size
      : number
    : number;
type KnownTensorShapeTuple =
  readonly [number] |
  readonly [number, number] |
  readonly [number, number, number] |
  readonly [number, number, number, number] |
  readonly [number, number, number, number, number];
type ReplaceAxis<Shape extends TensorShapeTuple, Axis extends number, Value extends number> =
  Shape extends readonly [infer A extends number]
    ? Axis extends 0 ? readonly [Value] : never
    : Shape extends readonly [infer A extends number, infer B extends number]
      ? Axis extends 0 ? readonly [Value, B] : Axis extends 1 ? readonly [A, Value] : never
      : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number]
        ? Axis extends 0 ? readonly [Value, B, C] : Axis extends 1 ? readonly [A, Value, C] : Axis extends 2 ? readonly [A, B, Value] : never
        : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number]
          ? Axis extends 0 ? readonly [Value, B, C, D] : Axis extends 1 ? readonly [A, Value, C, D] : Axis extends 2 ? readonly [A, B, Value, D] : Axis extends 3 ? readonly [A, B, C, Value] : never
          : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number, infer E extends number]
            ? Axis extends 0 ? readonly [Value, B, C, D, E] : Axis extends 1 ? readonly [A, Value, C, D, E] : Axis extends 2 ? readonly [A, B, Value, D, E] : Axis extends 3 ? readonly [A, B, C, Value, E] : Axis extends 4 ? readonly [A, B, C, D, Value] : never
            : never;
type InsertAxis<Shape extends TensorShapeTuple, Axis extends number, Value extends number> =
  Shape extends readonly [infer A extends number]
    ? Axis extends 0 ? readonly [Value, A] : Axis extends 1 ? readonly [A, Value] : never
    : Shape extends readonly [infer A extends number, infer B extends number]
      ? Axis extends 0 ? readonly [Value, A, B] : Axis extends 1 ? readonly [A, Value, B] : Axis extends 2 ? readonly [A, B, Value] : never
      : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number]
        ? Axis extends 0 ? readonly [Value, A, B, C] : Axis extends 1 ? readonly [A, Value, B, C] : Axis extends 2 ? readonly [A, B, Value, C] : Axis extends 3 ? readonly [A, B, C, Value] : never
        : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number]
          ? Axis extends 0 ? readonly [Value, A, B, C, D] : Axis extends 1 ? readonly [A, Value, B, C, D] : Axis extends 2 ? readonly [A, B, Value, C, D] : Axis extends 3 ? readonly [A, B, C, Value, D] : Axis extends 4 ? readonly [A, B, C, D, Value] : never
          : never;
type RemoveAxis<Shape extends TensorShapeTuple, Axis extends number> =
  Shape extends readonly [number]
    ? Axis extends 0 ? readonly [] : never
    : Shape extends readonly [infer A extends number, infer B extends number]
      ? Axis extends 0 ? readonly [B] : Axis extends 1 ? readonly [A] : never
      : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number]
        ? Axis extends 0 ? readonly [B, C] : Axis extends 1 ? readonly [A, C] : Axis extends 2 ? readonly [A, B] : never
        : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number]
          ? Axis extends 0 ? readonly [B, C, D] : Axis extends 1 ? readonly [A, C, D] : Axis extends 2 ? readonly [A, B, D] : Axis extends 3 ? readonly [A, B, C] : never
          : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number, infer E extends number]
            ? Axis extends 0 ? readonly [B, C, D, E] : Axis extends 1 ? readonly [A, C, D, E] : Axis extends 2 ? readonly [A, B, D, E] : Axis extends 3 ? readonly [A, B, C, E] : Axis extends 4 ? readonly [A, B, C, D] : never
            : never;
type ScalarFallbackShape<Shape extends TensorShapeTuple> =
  Shape extends readonly [] ? readonly [1] : Shape;
export type TensorTopkShape<Shape extends TensorShapeTuple, Dim extends number, K extends number> =
  Shape extends KnownTensorShapeTuple
    ? [NormalizeAxisForRank<Shape["length"], Dim>] extends [never]
      ? TensorShapeTuple
      : NormalizeAxisForRank<Shape["length"], Dim> extends infer Axis extends number
        ? ReplaceAxis<Shape, Axis, K> extends infer Out extends TensorShapeTuple
          ? Out
          : TensorShapeTuple
        : TensorShapeTuple
    : TensorShapeTuple;
export type TensorCatShape<Tensors extends readonly Tensor[], Dim extends number = 0> =
  Tensors extends readonly [Tensor<infer S extends TensorShapeTuple>, ...readonly Tensor[]]
    ? S extends KnownTensorShapeTuple
      ? [NormalizeAxisForRank<S["length"], Dim>] extends [never]
        ? TensorShapeTuple
        : NormalizeAxisForRank<S["length"], Dim> extends infer Axis extends number
          ? ReplaceAxis<S, Axis, TensorCatAxisSize<Tensors, S, Axis>> extends infer Out extends TensorShapeTuple
            ? Out
            : TensorShapeTuple
          : TensorShapeTuple
      : TensorShapeTuple
    : TensorShapeTuple;
export type TensorStackShape<Tensors extends readonly Tensor[], Dim extends number = 0> =
  Tensors extends readonly [Tensor<infer S extends TensorShapeTuple>, ...readonly Tensor[]]
    ? S extends KnownTensorShapeTuple
      ? AddNumbers<S["length"], 1> extends infer OutRank extends number
        ? [NormalizeAxisForRank<OutRank, Dim>] extends [never]
          ? TensorShapeTuple
          : InsertAxis<S, NormalizeAxisForRank<OutRank, Dim>, Tensors["length"]> extends infer Out extends TensorShapeTuple
            ? Out
            : TensorShapeTuple
          : TensorShapeTuple
      : TensorShapeTuple
    : TensorShapeTuple;
export type TensorVStackShape<Tensors extends readonly Tensor[]> =
  Tensors extends readonly [Tensor<infer S extends TensorShapeTuple>, ...readonly Tensor[]]
    ? S extends readonly [infer Width extends number]
      ? readonly [Tensors["length"], Width]
      : TensorCatShape<Tensors, 0>
    : TensorShapeTuple;
export type TensorHStackShape<Tensors extends readonly Tensor[]> =
  Tensors extends readonly [Tensor<infer S extends TensorShapeTuple>, ...readonly Tensor[]]
    ? S extends readonly [number]
      ? TensorCatShape<Tensors, 0>
      : TensorCatShape<Tensors, 1>
    : TensorShapeTuple;
type TensorOperandShape<Operand> = Operand extends Tensor<infer S extends TensorShapeTuple> ? S : TensorShapeTuple;
type EinsumTraceShape<Shape extends TensorShapeTuple> =
  Shape extends readonly [infer N extends number, infer N2 extends number]
    ? N2 extends N ? readonly [1] : TensorShapeTuple
    : TensorShapeTuple;
type EinsumImplicitEllipsisReductionShape<Shape extends TensorShapeTuple> =
  Shape extends readonly [number]
    ? readonly [1]
    : Shape extends readonly [infer A extends number, number]
      ? readonly [A]
      : Shape extends readonly [infer A extends number, infer B extends number, number]
        ? readonly [A, B]
        : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, number]
          ? readonly [A, B, C]
          : TensorShapeTuple;
type EinsumEllipsisMatmulShape<Lhs extends TensorShapeTuple, Rhs extends TensorShapeTuple> =
  Rhs extends readonly [infer K extends number, infer N extends number]
    ? Lhs extends readonly [infer M extends number, K]
      ? readonly [M, N]
      : Lhs extends readonly [infer B extends number, infer M extends number, K]
        ? readonly [B, M, N]
        : Lhs extends readonly [infer A extends number, infer B extends number, infer M extends number, K]
          ? readonly [A, B, M, N]
          : TensorShapeTuple
    : TensorShapeTuple;
type EinsumBatchMatmulShape<Lhs extends TensorShapeTuple, Rhs extends TensorShapeTuple> =
  Lhs extends readonly [infer LB extends number, infer M extends number, infer K extends number]
    ? Rhs extends readonly [infer RB extends number, K, infer N extends number]
      ? BroadcastDim<LB, RB> extends infer B extends number
        ? readonly [B, M, N]
        : TensorShapeTuple
      : TensorShapeTuple
    : TensorShapeTuple;
export type EinsumShape<Equation extends string, Operands extends readonly Tensor[]> =
  Operands extends readonly [infer Lhs extends Tensor, infer Rhs extends Tensor]
    ? Equation extends "ij,jk->ik"
      ? MatmulShape<TensorOperandShape<Lhs>, TensorOperandShape<Rhs>>
      : Equation extends "...ij,jk->...ik"
        ? EinsumEllipsisMatmulShape<TensorOperandShape<Lhs>, TensorOperandShape<Rhs>>
        : Equation extends "bij,bjk->bik"
          ? EinsumBatchMatmulShape<TensorOperandShape<Lhs>, TensorOperandShape<Rhs>>
          : TensorShapeTuple
    : Operands extends readonly [infer Input extends Tensor]
      ? Equation extends "ii->"
        ? EinsumTraceShape<TensorOperandShape<Input>>
        : Equation extends "...i->..."
          ? EinsumImplicitEllipsisReductionShape<TensorOperandShape<Input>>
          : TensorShapeTuple
      : TensorShapeTuple;
export type MatmulShape<Lhs extends TensorShapeTuple, Rhs extends TensorShapeTuple> =
  Lhs extends readonly [infer M extends number, infer K extends number]
    ? Rhs extends readonly [K, infer N extends number]
      ? readonly [M, N]
      : TensorShapeTuple
    : TensorShapeTuple;
export type BmmShape<Lhs extends TensorShapeTuple, Rhs extends TensorShapeTuple> =
  Lhs extends readonly [infer B extends number, infer M extends number, infer K extends number]
    ? Rhs extends readonly [B, K, infer N extends number]
      ? readonly [B, M, N]
      : TensorShapeTuple
    : TensorShapeTuple;
export type DiagonalShape<Shape extends TensorShapeTuple> =
  Shape extends readonly [infer Rows extends number, infer Cols extends number]
    ? readonly [MinNumbers<Rows, Cols>]
    : TensorShapeTuple;
export type ReductionShape<Shape extends TensorShapeTuple, Dim extends number | undefined = undefined> =
  Dim extends undefined
    ? readonly [1]
    : Shape extends readonly [number]
      ? Dim extends 0 | -1
        ? readonly [1]
        : TensorShapeTuple
      : Shape extends readonly [infer Rows extends number, infer Cols extends number]
        ? Dim extends 0 | -2
          ? readonly [1, Cols]
          : Dim extends 1 | -1
            ? readonly [Rows, 1]
            : TensorShapeTuple
        : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number]
          ? Dim extends 0 | -3
            ? readonly [1, B, C]
            : Dim extends 1 | -2
              ? readonly [A, 1, C]
              : Dim extends 2 | -1
                ? readonly [A, B, 1]
                : TensorShapeTuple
        : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number]
          ? Dim extends 0 | -4
            ? readonly [1, B, C, D]
            : Dim extends 1 | -3
              ? readonly [A, 1, C, D]
              : Dim extends 2 | -2
                ? readonly [A, B, 1, D]
                : Dim extends 3 | -1
                  ? readonly [A, B, C, 1]
                  : TensorShapeTuple
        : TensorShapeTuple;
export type LazyReductionShape<Shape extends TensorShapeTuple, Dim extends number> =
  Shape extends KnownTensorShapeTuple
    ? [NormalizeAxisForRank<Shape["length"], Dim>] extends [never]
      ? TensorShapeTuple
      : NormalizeAxisForRank<Shape["length"], Dim> extends infer Axis extends number
        ? RemoveAxis<Shape, Axis> extends infer Out extends TensorShapeTuple
          ? ScalarFallbackShape<Out>
          : TensorShapeTuple
        : TensorShapeTuple
    : TensorShapeTuple;
export type TransposeShape<Shape extends TensorShapeTuple, Dim0 extends number = 0, Dim1 extends number = 1> =
  Shape extends readonly [infer Rows extends number, infer Cols extends number]
    ? Dim0 extends Dim1
      ? Shape
      : Dim0 extends 0 | -2
        ? Dim1 extends 1 | -1
          ? readonly [Cols, Rows]
          : TensorShapeTuple
        : Dim0 extends 1 | -1
          ? Dim1 extends 0 | -2
            ? readonly [Cols, Rows]
            : TensorShapeTuple
          : TensorShapeTuple
    : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number]
      ? Dim0 extends Dim1
        ? Shape
        : Dim0 extends 0 | -3
          ? Dim1 extends 1 | -2
            ? readonly [B, A, C]
            : Dim1 extends 2 | -1
              ? readonly [C, B, A]
              : TensorShapeTuple
          : Dim0 extends 1 | -2
            ? Dim1 extends 0 | -3
              ? readonly [B, A, C]
              : Dim1 extends 2 | -1
                ? readonly [A, C, B]
                : TensorShapeTuple
            : Dim0 extends 2 | -1
              ? Dim1 extends 0 | -3
                ? readonly [C, B, A]
                : Dim1 extends 1 | -2
                  ? readonly [A, C, B]
                  : TensorShapeTuple
              : TensorShapeTuple
      : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number]
        ? Dim0 extends Dim1
          ? Shape
          : Dim0 extends 0 | -4
            ? Dim1 extends 1 | -3
              ? readonly [B, A, C, D]
              : Dim1 extends 2 | -2
                ? readonly [C, B, A, D]
                : Dim1 extends 3 | -1
                  ? readonly [D, B, C, A]
                  : TensorShapeTuple
            : Dim0 extends 1 | -3
              ? Dim1 extends 0 | -4
                ? readonly [B, A, C, D]
                : Dim1 extends 2 | -2
                  ? readonly [A, C, B, D]
                  : Dim1 extends 3 | -1
                    ? readonly [A, D, C, B]
                    : TensorShapeTuple
              : Dim0 extends 2 | -2
                ? Dim1 extends 0 | -4
                  ? readonly [C, B, A, D]
                  : Dim1 extends 1 | -3
                    ? readonly [A, C, B, D]
                    : Dim1 extends 3 | -1
                      ? readonly [A, B, D, C]
                      : TensorShapeTuple
                : Dim0 extends 3 | -1
                  ? Dim1 extends 0 | -4
                    ? readonly [D, B, C, A]
                    : Dim1 extends 1 | -3
                      ? readonly [A, D, C, B]
                      : Dim1 extends 2 | -2
                        ? readonly [A, B, D, C]
                        : TensorShapeTuple
                  : TensorShapeTuple
        : TensorShapeTuple;
type NormalizeAxisForRank<Rank extends number, Dim extends number> =
  Rank extends 1
    ? Dim extends 0 | -1 ? 0 : never
    : Rank extends 2
      ? Dim extends 0 | -2 ? 0 : Dim extends 1 | -1 ? 1 : never
      : Rank extends 3
        ? Dim extends 0 | -3 ? 0 : Dim extends 1 | -2 ? 1 : Dim extends 2 | -1 ? 2 : never
        : Rank extends 4
          ? Dim extends 0 | -4 ? 0 : Dim extends 1 | -3 ? 1 : Dim extends 2 | -2 ? 2 : Dim extends 3 | -1 ? 3 : never
          : Rank extends 5
            ? Dim extends 0 | -5 ? 0 : Dim extends 1 | -4 ? 1 : Dim extends 2 | -3 ? 2 : Dim extends 3 | -2 ? 3 : Dim extends 4 | -1 ? 4 : never
            : never;
type TupleAt<Shape extends TensorShapeTuple, Axis extends number> =
  Shape extends readonly [infer A extends number]
    ? Axis extends 0 ? A : never
    : Shape extends readonly [infer A extends number, infer B extends number]
      ? Axis extends 0 ? A : Axis extends 1 ? B : never
      : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number]
        ? Axis extends 0 ? A : Axis extends 1 ? B : Axis extends 2 ? C : never
        : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number]
          ? Axis extends 0 ? A : Axis extends 1 ? B : Axis extends 2 ? C : Axis extends 3 ? D : never
          : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number, infer E extends number]
            ? Axis extends 0 ? A : Axis extends 1 ? B : Axis extends 2 ? C : Axis extends 3 ? D : Axis extends 4 ? E : never
            : never;
type ContainsAxis<Axes extends readonly number[], Axis extends number> =
  Axis extends Axes[number] ? true : false;
type PermuteShapeInner<Shape extends TensorShapeTuple, Dims extends readonly number[], Seen extends readonly number[] = readonly []> =
  Dims extends readonly []
    ? readonly []
    : Dims extends readonly [infer Head extends number, ...infer Tail extends number[]]
      ? [NormalizeAxisForRank<Shape["length"], Head>] extends [never]
        ? never
        : NormalizeAxisForRank<Shape["length"], Head> extends infer Axis extends number
          ? ContainsAxis<Seen, Axis> extends true
            ? never
            : TupleAt<Shape, Axis> extends infer Value extends number
              ? [PermuteShapeInner<Shape, readonly [...Tail], readonly [...Seen, Axis]>] extends [infer Rest extends TensorShapeTuple]
                ? readonly [Value, ...Rest]
                : never
              : never
          : never
      : never;
export type PermuteShape<Shape extends TensorShapeTuple, Dims extends readonly number[]> =
  Shape["length"] extends Dims["length"]
    ? [PermuteShapeInner<Shape, Dims>] extends [never]
      ? TensorShapeTuple
      : PermuteShapeInner<Shape, Dims>
    : TensorShapeTuple;
export type SqueezeAllShape<Shape extends TensorShapeTuple, Out extends TensorShapeTuple = readonly []> =
  Shape extends readonly [infer Head extends number, ...infer Tail extends number[]]
    ? Head extends 1
      ? SqueezeAllShape<readonly [...Tail], Out>
      : SqueezeAllShape<readonly [...Tail], readonly [...Out, Head]>
    : Out extends readonly []
      ? readonly [1]
      : Out;
export type SqueezeShape<Shape extends TensorShapeTuple, Dim extends number | undefined = undefined> =
  Dim extends undefined
    ? SqueezeAllShape<Shape>
    : Shape extends readonly [infer A extends number]
      ? Dim extends 0 | -1
        ? A extends 1
          ? readonly [1]
          : Shape
        : TensorShapeTuple
      : Shape extends readonly [infer A extends number, infer B extends number]
        ? Dim extends 0 | -2
          ? A extends 1
            ? readonly [B]
            : Shape
          : Dim extends 1 | -1
            ? B extends 1
              ? readonly [A]
              : Shape
            : TensorShapeTuple
        : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number]
          ? Dim extends 0 | -3
            ? A extends 1
              ? readonly [B, C]
              : Shape
            : Dim extends 1 | -2
              ? B extends 1
                ? readonly [A, C]
                : Shape
              : Dim extends 2 | -1
                ? C extends 1
                  ? readonly [A, B]
                  : Shape
                : TensorShapeTuple
          : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number]
            ? Dim extends 0 | -4
              ? A extends 1
                ? readonly [B, C, D]
                : Shape
              : Dim extends 1 | -3
                ? B extends 1
                  ? readonly [A, C, D]
                  : Shape
                : Dim extends 2 | -2
                  ? C extends 1
                    ? readonly [A, B, D]
                    : Shape
                  : Dim extends 3 | -1
                    ? D extends 1
                      ? readonly [A, B, C]
                      : Shape
                    : TensorShapeTuple
            : TensorShapeTuple;
export type UnsqueezeShape<Shape extends TensorShapeTuple, Dim extends number> =
  Shape extends readonly [infer A extends number]
    ? Dim extends 0 | -2
      ? readonly [1, A]
      : Dim extends 1 | -1
        ? readonly [A, 1]
        : TensorShapeTuple
    : Shape extends readonly [infer A extends number, infer B extends number]
      ? Dim extends 0 | -3
        ? readonly [1, A, B]
        : Dim extends 1 | -2
        ? readonly [A, 1, B]
        : Dim extends 2 | -1
          ? readonly [A, B, 1]
          : TensorShapeTuple
      : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number]
        ? Dim extends 0 | -4
          ? readonly [1, A, B, C]
          : Dim extends 1 | -3
            ? readonly [A, 1, B, C]
            : Dim extends 2 | -2
              ? readonly [A, B, 1, C]
              : Dim extends 3 | -1
                ? readonly [A, B, C, 1]
                : TensorShapeTuple
        : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number]
          ? Dim extends 0 | -5
            ? readonly [1, A, B, C, D]
            : Dim extends 1 | -4
              ? readonly [A, 1, B, C, D]
              : Dim extends 2 | -3
                ? readonly [A, B, 1, C, D]
                : Dim extends 3 | -2
                  ? readonly [A, B, C, 1, D]
                  : Dim extends 4 | -1
                    ? readonly [A, B, C, D, 1]
                    : TensorShapeTuple
          : TensorShapeTuple;
export type SelectShape<Shape extends TensorShapeTuple, Dim extends number> =
  Shape extends readonly [number]
    ? Dim extends 0 | -1
      ? readonly [1]
      : TensorShapeTuple
    : Shape extends readonly [infer Rows extends number, infer Cols extends number]
      ? Dim extends 0 | -2
        ? readonly [Cols]
        : Dim extends 1 | -1
          ? readonly [Rows]
          : TensorShapeTuple
      : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number]
        ? Dim extends 0 | -3
          ? readonly [B, C]
          : Dim extends 1 | -2
            ? readonly [A, C]
            : Dim extends 2 | -1
              ? readonly [A, B]
              : TensorShapeTuple
      : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number]
        ? Dim extends 0 | -4
          ? readonly [B, C, D]
          : Dim extends 1 | -3
            ? readonly [A, C, D]
            : Dim extends 2 | -2
              ? readonly [A, B, D]
              : Dim extends 3 | -1
                ? readonly [A, B, C]
                : TensorShapeTuple
      : TensorShapeTuple;
export type NarrowShape<Shape extends TensorShapeTuple, Dim extends number, Length extends number> =
  Shape extends readonly [number]
    ? Dim extends 0 | -1
      ? readonly [Length]
      : TensorShapeTuple
    : Shape extends readonly [infer Rows extends number, infer Cols extends number]
      ? Dim extends 0 | -2
        ? readonly [Length, Cols]
        : Dim extends 1 | -1
          ? readonly [Rows, Length]
          : TensorShapeTuple
      : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number]
        ? Dim extends 0 | -3
          ? readonly [Length, B, C]
          : Dim extends 1 | -2
            ? readonly [A, Length, C]
            : Dim extends 2 | -1
              ? readonly [A, B, Length]
              : TensorShapeTuple
      : Shape extends readonly [infer A extends number, infer B extends number, infer C extends number, infer D extends number]
        ? Dim extends 0 | -4
          ? readonly [Length, B, C, D]
          : Dim extends 1 | -3
            ? readonly [A, Length, C, D]
            : Dim extends 2 | -2
              ? readonly [A, B, Length, D]
              : Dim extends 3 | -1
                ? readonly [A, B, C, Length]
                : TensorShapeTuple
      : TensorShapeTuple;
export type SliceShape<Shape extends TensorShapeTuple, Dim extends number, Start extends number, End extends number> =
  NarrowShape<Shape, Dim, SubtractNumbers<End, Start>>;
export type IndexLike = Tensor | Uint32Array | Int32Array | Float32Array | readonly number[];
type IndexSelectCount<Indices extends IndexLike> =
  Indices extends readonly number[] ? Indices["length"] : Indices extends Tensor<readonly [infer N extends number]> ? N : number;
export type TensorIndexSelectShape<Shape extends TensorShapeTuple, Dim extends number, Indices extends IndexLike> =
  Shape extends KnownTensorShapeTuple
    ? [NormalizeAxisForRank<Shape["length"], Dim>] extends [never]
      ? TensorShapeTuple
      : NormalizeAxisForRank<Shape["length"], Dim> extends infer Axis extends number
        ? ReplaceAxis<Shape, Axis, IndexSelectCount<Indices>> extends infer Out extends TensorShapeTuple
          ? Out
          : TensorShapeTuple
        : TensorShapeTuple
    : TensorShapeTuple;
export type TensorGatherShape<Index extends IndexLike> =
  Index extends Tensor<infer S extends TensorShapeTuple>
    ? S
    : Index extends readonly number[]
      ? readonly [Index["length"]]
      : readonly [number];
export type OneHotShape<InputShape extends TensorShapeTuple, NumClasses extends number> = readonly [...InputShape, NumClasses];
export type ByteLike = ArrayBufferView | ArrayBuffer | readonly number[];
export type ZgmlBackend = "auto" | "cpu" | "metal" | "webgpu";

export type TensorOptions = {
  requiresGrad?: boolean;
  grad?: Float32Array | null;
};
export type AllCloseOptions = {
  rtol?: number;
  atol?: number;
};
export type IsCloseOptions = AllCloseOptions & {
  equalNan?: boolean;
  equal_nan?: boolean;
};

export type RandomTensorOptions = TensorOptions & {
  mean?: number;
  std?: number;
  rng?: () => number;
  seed?: number;
};
export type RandomUniformTensorOptions = TensorOptions & {
  rng?: () => number;
  seed?: number;
};
export type RandomIntTensorOptions = TensorOptions & {
  rng?: () => number;
  seed?: number;
};
export type TensorNativeBufferOptions = ProgramCreateBufferOptions & {
  program?: Program;
  kind?: ProgramDeviceBufferKind;
};
export type TensorNativePlacement = Readonly<{
  kind: "zgml.tensor.native-placement";
  storage: "host" | "program";
  bufferKind: ProgramDeviceBufferKind | null;
  dtype: "f32";
  device: "cpu" | "program" | string;
  shape: readonly number[];
  length: number;
  byteLength: number;
  programCompileEvidenceSignature: string | null;
  nativeCompilerAuthority: "zig-module-program" | null;
  nativeProgramInspectionSignature: string | null;
  nativeProgramInspectionSource: "zig-program-inspection" | null;
  signature: string;
}>;
export type TensorDType = "f32";
export type TensorDTypeLike = TensorDType | "float32";
export type TensorDevice = "cpu";
export type TensorDeviceLike = TensorDevice | "host" | ZgmlBackend;
export type TensorInspection = {
  dtype: TensorDType;
  device: TensorDevice;
  shape: readonly number[];
  rank: number;
  length: number;
  strides: readonly number[];
  storageOffset: number;
  elementSize: number;
  byteLength: number;
  contiguous: boolean;
  requiresGrad: boolean;
  isLeaf: boolean;
};
export type TensorToTarget = TensorDTypeLike | TensorDeviceLike;
export type TensorToOptions = {
  dtype?: TensorDTypeLike;
  device?: TensorDeviceLike;
  backend?: ZgmlBackend;
  placement?: ZgmlBackend;
  copy?: boolean;
};
export type TensorFromNativeBufferOptions = TensorOptions & {
  length?: number;
  byteOffset?: number;
};
export type SessionStepTensorOptions = TensorOptions & {
  shape?: TensorShape;
  output?: Tensor | Float32Array;
};
export type SessionStepParams<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = {
  input?: ProgramInputBinding<InputShape>;
  output?: ProgramOutputBinding<OutputShape> | false;
};
export type SessionExecuteParams<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = SessionStepParams<InputShape, OutputShape>;
export type SessionExecuteTensorParams<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = SessionStepTensorOptions & {
  input?: ProgramInputBinding<InputShape>;
  output?: Tensor<OutputShape> | Float32Array;
};
export type SessionExecuteIntoParams<InputShape extends TensorShapeTuple = TensorShapeTuple> = {
  input?: ProgramInputBinding<InputShape>;
};
export type SessionReadOutputTensorOptions = TensorOptions & {
  shape?: TensorShape;
  output?: Tensor | Float32Array;
  length?: number;
  byteOffset?: number;
};
export type SessionCallProfile = Readonly<{
  kind: "zgml.session.call-profile";
  signature: string;
  stepCount: number;
  stepTensorCount: number;
  stepIntoCount: number;
  executeCount: number;
  executeTensorCount: number;
  executeIntoCount: number;
  prepareExecuteIntoCount: number;
  prefillCount: number;
  prefillTensorCount: number;
  prefillIntoCount: number;
  readOutputIntoCount: number;
  readOutputTensorCount: number;
  advanceCount: number;
  resetCount: number;
  uploadParametersCount: number;
  uploadParameterCount: number;
  uploadParameterByNameCount: number;
  uploadParameterRangeCount: number;
}>;
export type SessionBoundBufferKind = "none" | "host" | "native";
export type SessionDefaultOutputKind = "allocated" | "bound-host" | "bound-native-readback";
export type SessionNoOutputEffect = "advance-state";
export type SessionStepContract = Readonly<{
  kind: "session-step-contract";
  signature: string;
  scalarType: "f32";
  scalarBytes: 4;
  inputLen: number;
  outputLen: number;
  inputByteLength: number;
  outputByteLength: number;
  inputSlotName: "input";
  outputSlotName: "output";
  inputSlotRole: "step-input";
  outputSlotRole: "step-output";
  inputShape: readonly number[];
  outputShape: readonly number[];
  programOutputShape: readonly number[];
  boundInput: SessionBoundBufferKind;
  boundOutput: SessionBoundBufferKind;
  defaultOutput: SessionDefaultOutputKind;
  defaultReadbackRequired: boolean;
  defaultAllocationFree: boolean;
  defaultHotPath: boolean;
  defaultOutputEffect: SessionStepParamsOutputEffect;
  defaultOutputOwnership: SessionStepParamsOutputOwnership;
  defaultOutputReturnOwnership: SessionStepParamsOutputReturnOwnership;
  hasBoundInput: boolean;
  hasBoundOutput: boolean;
  canReadOutput: boolean;
  acceptsInlineInput: true;
  acceptsNoInput: boolean;
  requiresInput: boolean;
  acceptsInlineOutput: true;
  acceptsNoOutput: true;
  noOutputEffect: SessionNoOutputEffect;
}>;
export type SessionStepParamsDiagnosticCode =
  | "invalid-step-params"
  | "invalid-input"
  | "invalid-output"
  | "token-choice-mismatch"
  | "invalid-token"
  | "invalid-token-window"
  | "not-allocation-free-step-params"
  | "not-runtime-output-allocation-free-step-params"
  | "not-no-readback-step-params"
  | "not-readback-free-step-params"
  | "not-hot-step-params";
export type SessionStepParamsDiagnostic = Readonly<{
  code: SessionStepParamsDiagnosticCode;
  message: string;
  actualLength?: number;
  expectedLength?: number;
  actualShape?: readonly number[];
  expectedShape?: readonly number[];
  contextLength?: number;
  position?: number;
  remainingContext?: number;
}>;
export type SessionStepParamsInputSource =
  | "inline"
  | "bound-host"
  | "bound-native"
  | "none"
  | "token"
  | "tokens"
  | "rejected";
export type SessionStepParamsInputOwnership =
  | "caller"
  | "session"
  | "none"
  | "rejected";
export type SessionStepParamsOutputTarget =
  | "inline"
  | "allocated"
  | "bound-host"
  | "bound-native-readback"
  | "none"
  | "rejected";
export type SessionStepParamsOutputEffect =
  | "write"
  | "write-readback"
  | "none"
  | "rejected";
export type SessionStepParamsOutputOwnership =
  | "caller"
  | "session"
  | "runtime"
  | "none"
  | "rejected";
export type SessionStepParamsOutputReturnOwnership =
  | "caller"
  | "session"
  | "runtime"
  | "none"
  | "rejected";
export type SessionStepParamsStateEffect =
  | "advance"
  | "none"
  | "rejected";
export type SessionStepParamsHotPathStatus =
  | "hot"
  | "rejected"
  | "allocates"
  | "readback"
  | "allocates-readback";
export type SessionStepParamsHotPathBlocker =
  | "rejected"
  | "runtime-output-allocation"
  | "readback";
export type SessionStepParamsElementType =
  | "f32"
  | "token-u32"
  | "none"
  | "rejected";
export type SessionStepParamsCompatibility = Readonly<{
  kind: "zgml.step-params.compatibility";
  accepted: boolean;
  canExecute: boolean;
  status: "accepted" | "rejected";
  contractKind: "session-step-contract" | "llama-session-step-contract";
  contractSignature: string;
  contractPosition: number | null;
  stepParamsSignature: string;
  rejectionCode: SessionStepParamsDiagnosticCode | null;
  stateEffect: SessionStepParamsStateEffect;
  allocationFree: boolean;
  inputSource: SessionStepParamsInputSource;
  inputOwnership: SessionStepParamsInputOwnership;
  readsInput: boolean;
  outputTarget: SessionStepParamsOutputTarget;
  outputEffect: SessionStepParamsOutputEffect;
  outputOwnership: SessionStepParamsOutputOwnership;
  outputReturnOwnership: SessionStepParamsOutputReturnOwnership;
  writesOutput: boolean;
  readbackRequired: boolean;
  readbackFree: boolean;
  runtimeOutputAllocationFree: boolean;
  hotPath: boolean;
  hotPathStatus: SessionStepParamsHotPathStatus;
  hotPathBlockers: readonly SessionStepParamsHotPathBlocker[];
  inputElementType: SessionStepParamsElementType;
  outputElementType: SessionStepParamsElementType;
  inputElementLength: number;
  outputElementLength: number;
  inputShape: readonly number[];
  outputShape: readonly number[];
  inputShapeSignature: string;
  outputShapeSignature: string;
  inputByteLength: number;
  outputByteLength: number;
  diagnostics: readonly SessionStepParamsDiagnostic[];
}>;
export type SessionExecutionPlan<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = Readonly<{
  kind: "zgml.session.execution-plan";
  signature: string;
  contract: SessionStepContract | LlamaSessionStepContract;
  compatibility: SessionStepParamsCompatibility;
  accepted: boolean;
  canExecute: boolean;
  hotPath: boolean;
  hotPathStatus: SessionStepParamsHotPathStatus;
  hotPathBlockers: readonly SessionStepParamsHotPathBlocker[];
  contractKind: "session-step-contract" | "llama-session-step-contract";
  contractSignature: string;
  stepParamsSignature: string;
  defaultHotPath: boolean;
  defaultAllocationFree: boolean;
  defaultReadbackRequired: boolean;
  allocationFree: boolean;
  runtimeOutputAllocationFree: boolean;
  readbackFree: boolean;
  readbackRequired: boolean;
  stateEffect: SessionStepParamsStateEffect;
  inputSource: SessionStepParamsInputSource;
  inputOwnership: SessionStepParamsInputOwnership;
  readsInput: boolean;
  outputTarget: SessionStepParamsOutputTarget;
  outputEffect: SessionStepParamsOutputEffect;
  outputOwnership: SessionStepParamsOutputOwnership;
  outputReturnOwnership: SessionStepParamsOutputReturnOwnership;
  writesOutput: boolean;
  inputElementType: SessionStepParamsElementType;
  outputElementType: SessionStepParamsElementType;
  inputElementLength: number;
  outputElementLength: number;
  inputShape: InputShape;
  outputShape: OutputShape;
  inputShapeSignature: string;
  outputShapeSignature: string;
  inputByteLength: number;
  outputByteLength: number;
  rejectionCode: SessionStepParamsDiagnosticCode | null;
  diagnostics: readonly SessionStepParamsDiagnostic[];
}>;

export declare class Tensor<Shape extends TensorShapeTuple = TensorShapeTuple> {
  readonly data: Float32Array;
  readonly shape: readonly [...Shape];
  requiresGrad: boolean;
  requires_grad: boolean;
  grad: Float32Array | null;

  constructor(data: TensorLike, shape?: Shape, options?: TensorOptions);

  readonly length: number;
  readonly rank: number;
  readonly ndim: number;
  readonly T: Tensor<TransposeShape<Shape>>;
  readonly mT: Tensor<TransposeShape<Shape, -2, -1>>;
  readonly dtype: TensorDType;
  readonly device: TensorDevice;
  readonly isLeaf: boolean;
  readonly is_leaf: boolean;

  isCpu(): boolean;
  is_cpu(): boolean;
  isFloatingPoint(): boolean;
  is_floating_point(): boolean;
  requires_grad_(requiresGrad?: boolean): this;
  requiresGrad_(requiresGrad?: boolean): this;
  toFloat32Array(): Float32Array;
  numpy(): Float32Array;
  to(target?: TensorToTarget | TensorToOptions, options?: TensorToOptions): Tensor<Shape>;
  cpu(options?: TensorToOptions): Tensor<Shape>;
  float(options?: TensorToOptions): Tensor<Shape>;
  float32(options?: TensorToOptions): Tensor<Shape>;
  typeAs(other: TensorLike, options?: TensorToOptions): Tensor<Shape>;
  type_as(other: TensorLike, options?: TensorToOptions): Tensor<Shape>;
  new_empty<const S extends TensorShape>(shape: S, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
  newEmpty<const S extends TensorShape>(shape: S, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
  new_zeros<const S extends TensorShape>(shape: S, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
  newZeros<const S extends TensorShape>(shape: S, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
  new_ones<const S extends TensorShape>(shape: S, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
  newOnes<const S extends TensorShape>(shape: S, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
  new_full<const S extends TensorShape>(shape: S, value: number, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
  newFull<const S extends TensorShape>(shape: S, value: number, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
  inspect(): TensorInspection;
  toNativeBuffer(options?: TensorNativeBufferOptions): NativeBuffer;
  nativePlacement(options?: TensorNativeBufferOptions): TensorNativePlacement;
  native_placement(options?: TensorNativeBufferOptions): TensorNativePlacement;
  place(program: Program, kind?: ProgramDeviceBufferKind, options?: ProgramCreateBufferOptions): NativeBuffer;
  copyFromNativeBuffer_(buffer: NativeBuffer, options?: TensorFromNativeBufferOptions): this;
  copy_from_native_buffer_(buffer: NativeBuffer, options?: TensorFromNativeBufferOptions): this;
  toJSON(): TensorJSON;
  item(): number;
  toNumber(): number;
  dim(): number;
  ndimension(): number;
  numel(): number;
  nelement(): number;
  elementSize(): number;
  element_size(): number;
  nbytes(): number;
  size(): readonly number[];
  size(dim: number): number;
  allclose(other: TensorLike, options?: AllCloseOptions): boolean;
  equal(other: TensorLike): boolean;
  stride(): readonly number[];
  stride(dim: number): number;
  strides(): readonly number[];
  storageOffset(): number;
  storage_offset(): number;
  isContiguous(): boolean;
  is_contiguous(): boolean;
  contiguous(): this;
  get(indices: readonly number[]): number;
  get(...indices: number[]): number;
  set(indices: readonly number[], value: number): this;
  set(...indicesAndValue: number[]): this;
  toArray(): TensorNestedArray;
  tolist(): TensorNestedArray;
  to_list(): TensorNestedArray;
  valueOf(): number;
  [Symbol.toPrimitive](): number;
  zeroGrad(options?: ZeroGradOptions): void;
  zero_grad(options?: ZeroGradOptions): void;
  fill_(value: number): this;
  zero_(): this;
  ones_(): this;
  copy_(source: TensorLike): this;
  clone(): Tensor<Shape>;
  detach(): Tensor<Shape>;
  detach_(): this;
  reshape<const S extends TensorShape>(shape: S): Tensor<ReshapeShape<Shape, S>>;
  view<const S extends TensorShape>(shape: S): Tensor<ReshapeShape<Shape, S>>;
  reshape_as<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): Tensor<OtherShape>;
  reshape_as(other: TensorLike): Tensor;
  view_as<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): Tensor<OtherShape>;
  view_as(other: TensorLike): Tensor;
  broadcastTo<const S extends TensorShape>(shape: S): Tensor<TensorShapeOf<S>>;
  expand<const S extends TensorShape>(shape: S): Tensor<TensorShapeOf<S>>;
  expand_as<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): Tensor<OtherShape>;
  expand_as(other: TensorLike): Tensor;
  repeat<const Repeats extends TensorShape>(repeats: Repeats): Tensor<RepeatShape<Shape, Repeats>>;
  repeat<const Repeats extends TensorShapeTuple>(...repeats: Repeats): Tensor<RepeatShape<Shape, Repeats>>;
  repeat(...repeats: number[]): Tensor;
  tile<const Repeats extends TensorShape>(repeats: Repeats): Tensor<RepeatShape<Shape, Repeats>>;
  tile<const Repeats extends TensorShapeTuple>(...repeats: Repeats): Tensor<RepeatShape<Shape, Repeats>>;
  tile(...repeats: number[]): Tensor;
  flatten(): Tensor<FlattenShape<Shape>>;
  flatten<const StartDim extends number, const EndDim extends number>(startDim: StartDim, endDim: EndDim): Tensor<FlattenShape<Shape, StartDim, EndDim>>;
  flatten(startDim?: number, endDim?: number): Tensor;
  squeeze(): Tensor<SqueezeShape<Shape>>;
  squeeze<const Dim extends number>(dim: Dim): Tensor<SqueezeShape<Shape, Dim>>;
  squeeze(dim?: number | null): Tensor;
  unsqueeze<const Dim extends number>(dim: Dim): Tensor<UnsqueezeShape<Shape, Dim>>;
  transpose(): Tensor<TransposeShape<Shape>>;
  transpose<const Dim0 extends number, const Dim1 extends number>(dim0: Dim0, dim1: Dim1): Tensor<TransposeShape<Shape, Dim0, Dim1>>;
  permute<const Dims extends readonly number[]>(dims: Dims): Tensor<PermuteShape<Shape, Dims>>;
  permute(dims: readonly number[]): Tensor;
  flip(dims: readonly number[]): Tensor<Shape>;
  roll(shifts: TensorRollArg, dims?: TensorRollArg | null): Tensor<Shape>;
  select<const Dim extends number>(dim: Dim, index: number): Tensor<SelectShape<Shape, Dim>>;
  narrow<const Dim extends number, const Length extends number>(dim: Dim, start: number, length: Length): Tensor<NarrowShape<Shape, Dim, Length>>;
  narrow(dim: number, start: number, length: number): Tensor;
  slice<const Dim extends number, const Start extends number, const End extends number>(dim: Dim, start: Start, end: End, step?: 1): Tensor<SliceShape<Shape, Dim, Start, End>>;
  slice(dim: number, start?: number | null, end?: number | null, step?: number): Tensor;
  indexSelect<const Dim extends number, const Indices extends IndexLike>(dim: Dim, indices: Indices): Tensor<TensorIndexSelectShape<Shape, Dim, Indices>>;
  indexSelect(dim: number, indices: IndexLike): Tensor;
  index_select<const Dim extends number, const Indices extends IndexLike>(dim: Dim, indices: Indices): Tensor<TensorIndexSelectShape<Shape, Dim, Indices>>;
  index_select(dim: number, indices: IndexLike): Tensor;
  gather<const Dim extends number, const Index extends IndexLike>(dim: Dim, index: Index): Tensor<TensorGatherShape<Index>>;
  gather(dim: number, index: IndexLike): Tensor;
  take<const Index extends IndexLike>(index: Index): Tensor<TensorGatherShape<Index>>;
  take(index: IndexLike): Tensor;
  argsort(dim?: number, descending?: boolean): Tensor<Shape>;
  sort(dim?: number, descending?: boolean): TensorSortResult<Shape>;
  topk<const K extends number>(k: K): TensorTopkResult<TensorTopkShape<Shape, -1, K>>;
  topk<const K extends number, const Dim extends number>(k: K, dim: Dim, largest?: boolean, sorted?: boolean): TensorTopkResult<TensorTopkShape<Shape, Dim, K>>;
  topk(k: number, dim?: number, largest?: boolean, sorted?: boolean): TensorTopkResult;
  scatterAdd<const Dim extends number>(dim: Dim, index: IndexLike, src: TensorLike): Tensor<Shape>;
  scatterAdd(dim: number, index: IndexLike, src: TensorLike): Tensor;
  scatter_add<const Dim extends number>(dim: Dim, index: IndexLike, src: TensorLike): Tensor<Shape>;
  scatter_add(dim: number, index: IndexLike, src: TensorLike): Tensor;
  split(splitSizeOrSections: number | readonly number[], dim?: number): readonly Tensor[];
  chunk(chunks: number, dim?: number): readonly Tensor[];
  unbind(dim?: number): readonly Tensor[];
  backward(gradient?: TensorLike): void;

  add(other: number): Tensor<Shape>;
  add<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): Tensor<BroadcastShape<Shape, OtherShape>>;
  add(other: TensorLike): Tensor;
  sub(other: number): Tensor<Shape>;
  sub<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): Tensor<BroadcastShape<Shape, OtherShape>>;
  sub(other: TensorLike): Tensor;
  mul(other: number): Tensor<Shape>;
  mul<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): Tensor<BroadcastShape<Shape, OtherShape>>;
  mul(other: TensorLike): Tensor;
  div(other: number): Tensor<Shape>;
  div<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): Tensor<BroadcastShape<Shape, OtherShape>>;
  div(other: TensorLike): Tensor;
  add_(other: number): this;
  add_<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): this;
  add_(other: TensorLike): this;
  sub_(other: number): this;
  sub_<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): this;
  sub_(other: TensorLike): this;
  mul_(other: number): this;
  mul_<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): this;
  mul_(other: TensorLike): this;
  div_(other: number): this;
  div_<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): this;
  div_(other: TensorLike): this;
  maximum(other: number): Tensor<Shape>;
  maximum<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): Tensor<BroadcastShape<Shape, OtherShape>>;
  maximum(other: TensorLike): Tensor;
  minimum(other: number): Tensor<Shape>;
  minimum<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): Tensor<BroadcastShape<Shape, OtherShape>>;
  minimum(other: TensorLike): Tensor;
  where(input: number, other: number): Tensor<Shape>;
  where<const InputShape extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<InputShape>, other: Tensor<OtherShape>): Tensor<WhereShape<Shape, InputShape, OtherShape>>;
  where(input: TensorLike, other: TensorLike): Tensor;
  maskedFill(mask: TensorLike, value: number): Tensor<Shape>;
  maskedFill(mask: TensorLike, value: TensorLike): Tensor;
  masked_fill(mask: TensorLike, value: number): Tensor<Shape>;
  masked_fill(mask: TensorLike, value: TensorLike): Tensor;
  eq(other: number): Tensor<Shape>;
  eq<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): Tensor<BroadcastShape<Shape, OtherShape>>;
  eq(other: TensorLike): Tensor;
  ne(other: number): Tensor<Shape>;
  ne<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): Tensor<BroadcastShape<Shape, OtherShape>>;
  ne(other: TensorLike): Tensor;
  lt(other: number): Tensor<Shape>;
  lt<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): Tensor<BroadcastShape<Shape, OtherShape>>;
  lt(other: TensorLike): Tensor;
  le(other: number): Tensor<Shape>;
  le<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): Tensor<BroadcastShape<Shape, OtherShape>>;
  le(other: TensorLike): Tensor;
  gt(other: number): Tensor<Shape>;
  gt<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): Tensor<BroadcastShape<Shape, OtherShape>>;
  gt(other: TensorLike): Tensor;
  ge(other: number): Tensor<Shape>;
  ge<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>): Tensor<BroadcastShape<Shape, OtherShape>>;
  ge(other: TensorLike): Tensor;
  isclose(other: number, options?: IsCloseOptions): Tensor<Shape>;
  isclose<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>, options?: IsCloseOptions): Tensor<BroadcastShape<Shape, OtherShape>>;
  isclose(other: TensorLike, options?: IsCloseOptions): Tensor;
  neg(): Tensor<Shape>;
  negative(): Tensor<Shape>;
  exp(): Tensor<Shape>;
  expm1(): Tensor<Shape>;
  log(): Tensor<Shape>;
  log1p(): Tensor<Shape>;
  sqr(): Tensor<Shape>;
  square(): Tensor<Shape>;
  pow(exponent: number): Tensor<Shape>;
  recip(): Tensor<Shape>;
  reciprocal(): Tensor<Shape>;
  abs(): Tensor<Shape>;
  sgn(): Tensor<Shape>;
  sign(): Tensor<Shape>;
  step(): Tensor<Shape>;
  isnan(): Tensor<Shape>;
  isinf(): Tensor<Shape>;
  isfinite(): Tensor<Shape>;
  floor(): Tensor<Shape>;
  ceil(): Tensor<Shape>;
  sqrt(): Tensor<Shape>;
  rsqrt(): Tensor<Shape>;
  clamp(min?: number | null, max?: number | null): Tensor<Shape>;
  clip(min?: number | null, max?: number | null): Tensor<Shape>;
  matmul<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>, otherShape?: readonly number[]): Tensor<MatmulShape<Shape, OtherShape>>;
  matmul(other: TensorLike, otherShape?: readonly number[]): Tensor;
  mm<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>, otherShape?: readonly number[]): Tensor<MatmulShape<Shape, OtherShape>>;
  mm(other: TensorLike, otherShape?: readonly number[]): Tensor;
  dot<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>, otherShape?: readonly number[]): Tensor<readonly [1]>;
  dot(other: TensorLike, otherShape?: readonly number[]): Tensor<readonly [1]>;
  trace(): Tensor<readonly [1]>;
  diagonal(): Tensor<DiagonalShape<Shape>>;
  bmm<const OtherShape extends TensorShapeTuple>(other: Tensor<OtherShape>, otherShape?: readonly number[]): Tensor<BmmShape<Shape, OtherShape>>;
  bmm(other: TensorLike, otherShape?: readonly number[]): Tensor;
  sum(): Tensor<ReductionShape<Shape>>;
  sum<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  prod(): Tensor<ReductionShape<Shape>>;
  prod<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  cumsum(axis?: number): Tensor<Shape>;
  mean(): Tensor<ReductionShape<Shape>>;
  mean<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  max(): Tensor<ReductionShape<Shape>>;
  max<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  min(): Tensor<ReductionShape<Shape>>;
  min<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  any(): Tensor<ReductionShape<Shape>>;
  any<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  all(): Tensor<ReductionShape<Shape>>;
  all<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  argmax(): Tensor<ReductionShape<Shape>>;
  argmax<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  argmin(): Tensor<ReductionShape<Shape>>;
  argmin<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  variance(): Tensor<ReductionShape<Shape>>;
  variance<const Dim extends number>(axis: Dim, correction?: number): Tensor<ReductionShape<Shape, Dim>>;
  var(): Tensor<ReductionShape<Shape>>;
  var<const Dim extends number>(axis: Dim, correction?: number): Tensor<ReductionShape<Shape, Dim>>;
  std(): Tensor<ReductionShape<Shape>>;
  std<const Dim extends number>(axis: Dim, correction?: number): Tensor<ReductionShape<Shape, Dim>>;
  norm(): Tensor<ReductionShape<Shape>>;
  norm<const Dim extends number>(axis: Dim, p?: number): Tensor<ReductionShape<Shape, Dim>>;
  sumDim<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  prodDim<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  meanDim<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  maxDim<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  minDim<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  anyDim<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  allDim<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  argmaxDim<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  argminDim<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  softmax(axis?: number): Tensor<Shape>;
  softmax_dim(axis?: number): Tensor<Shape>;
  softmaxDim(axis?: number): Tensor<Shape>;
  logSoftmax(axis?: number): Tensor<Shape>;
  log_softmax(axis?: number): Tensor<Shape>;
  log_softmax_dim(axis?: number): Tensor<Shape>;
  logSoftmaxDim(axis?: number): Tensor<Shape>;
  logsumexp(): Tensor<ReductionShape<Shape>>;
  logsumexp<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  logSumExp(): Tensor<ReductionShape<Shape>>;
  logSumExp<const Dim extends number>(axis: Dim): Tensor<ReductionShape<Shape, Dim>>;
  gelu(): Tensor<Shape>;
  relu(): Tensor<Shape>;
  silu(): Tensor<Shape>;
  sigmoid(): Tensor<Shape>;
  tanh(): Tensor<Shape>;
  sin(): Tensor<Shape>;
  cos(): Tensor<Shape>;
  tan(): Tensor<Shape>;
  round(): Tensor<Shape>;
  trunc(): Tensor<Shape>;
  meanSquaredError(target: TensorLike): Tensor<readonly [1]>;

  [Symbol.iterator](): IterableIterator<number>;

  static tensor<const S extends TensorShapeTuple>(data: TensorLike, shape: S, options?: TensorOptions): Tensor<S>;
  static tensor(data: TensorLike): Tensor;
  static parameter<const S extends TensorShapeTuple>(data: TensorLike, shape: S, options?: TensorOptions): Tensor<S>;
  static parameter(data: TensorLike, options?: TensorOptions): Tensor;
  static param<const S extends TensorShapeTuple>(data: TensorLike, shape: S, options?: TensorOptions): Tensor<S>;
  static param(data: TensorLike, options?: TensorOptions): Tensor;
  static scalar(value: number, options?: TensorOptions): Tensor<readonly [1]>;
  static empty<const S extends TensorShape>(shape: S, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
  static emptyLike<const S extends TensorShapeTuple>(input: Tensor<S>, options?: TensorOptions): Tensor<S>;
  static emptyLike(input: TensorLike, options?: TensorOptions): Tensor;
  static empty_like<const S extends TensorShapeTuple>(input: Tensor<S>, options?: TensorOptions): Tensor<S>;
  static empty_like(input: TensorLike, options?: TensorOptions): Tensor;
  static zeros<const S extends TensorShape>(shape: S, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
  static zerosLike<const S extends TensorShapeTuple>(input: Tensor<S>, options?: TensorOptions): Tensor<S>;
  static zerosLike(input: TensorLike, options?: TensorOptions): Tensor;
  static zeros_like<const S extends TensorShapeTuple>(input: Tensor<S>, options?: TensorOptions): Tensor<S>;
  static zeros_like(input: TensorLike, options?: TensorOptions): Tensor;
  static ones<const S extends TensorShape>(shape: S, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
  static onesLike<const S extends TensorShapeTuple>(input: Tensor<S>, options?: TensorOptions): Tensor<S>;
  static onesLike(input: TensorLike, options?: TensorOptions): Tensor;
  static ones_like<const S extends TensorShapeTuple>(input: Tensor<S>, options?: TensorOptions): Tensor<S>;
  static ones_like(input: TensorLike, options?: TensorOptions): Tensor;
  static eye<const N extends number>(size: N, options?: TensorOptions): Tensor<readonly [N, N]>;
  static full<const S extends TensorShape>(shape: S, value: number, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
  static fullLike<const S extends TensorShapeTuple>(input: Tensor<S>, value: number, options?: TensorOptions): Tensor<S>;
  static fullLike(input: TensorLike, value: number, options?: TensorOptions): Tensor;
  static full_like<const S extends TensorShapeTuple>(input: Tensor<S>, value: number, options?: TensorOptions): Tensor<S>;
  static full_like(input: TensorLike, value: number, options?: TensorOptions): Tensor;
  static rand<const S extends TensorShape>(shape: S, options?: RandomUniformTensorOptions): Tensor<TensorShapeOf<S>>;
  static randLike<const S extends TensorShapeTuple>(input: Tensor<S>, options?: RandomUniformTensorOptions): Tensor<S>;
  static randLike(input: TensorLike, options?: RandomUniformTensorOptions): Tensor;
  static rand_like<const S extends TensorShapeTuple>(input: Tensor<S>, options?: RandomUniformTensorOptions): Tensor<S>;
  static rand_like(input: TensorLike, options?: RandomUniformTensorOptions): Tensor;
  static randn<const S extends TensorShape>(shape: S, options?: RandomTensorOptions): Tensor<TensorShapeOf<S>>;
  static randnLike<const S extends TensorShapeTuple>(input: Tensor<S>, options?: RandomTensorOptions): Tensor<S>;
  static randnLike(input: TensorLike, options?: RandomTensorOptions): Tensor;
  static randn_like<const S extends TensorShapeTuple>(input: Tensor<S>, options?: RandomTensorOptions): Tensor<S>;
  static randn_like(input: TensorLike, options?: RandomTensorOptions): Tensor;
  static randInt<const S extends TensorShape>(high: number, shape: S, options?: RandomIntTensorOptions): Tensor<TensorShapeOf<S>>;
  static randInt<const S extends TensorShape>(low: number, high: number, shape: S, options?: RandomIntTensorOptions): Tensor<TensorShapeOf<S>>;
  static randint<const S extends TensorShape>(high: number, shape: S, options?: RandomIntTensorOptions): Tensor<TensorShapeOf<S>>;
  static randint<const S extends TensorShape>(low: number, high: number, shape: S, options?: RandomIntTensorOptions): Tensor<TensorShapeOf<S>>;
  static randPerm<const N extends number>(size: N, options?: RandomIntTensorOptions): Tensor<readonly [N]>;
  static randperm<const N extends number>(size: N, options?: RandomIntTensorOptions): Tensor<readonly [N]>;
  static manualSeed(seed: number): number;
  static manual_seed(seed: number): number;
  static initialSeed(): number | null;
  static initial_seed(): number | null;
  static seededRng(seed: number): () => number;
  static linspace<const S extends TensorShapeTuple>(shape: S, start: number, end: number, options?: TensorOptions): Tensor<S>;
  static linspace<const Steps extends number>(start: number, end: number, steps: Steps, options?: TensorOptions): Tensor<readonly [Steps]>;
  static linspace(start: number, end: number, steps: number, options?: TensorOptions): Tensor;
  static arange<const End extends number>(end: End, options?: TensorOptions): Tensor<ArangeShape<End>>;
  static arange<const Start extends number, const End extends number>(start: Start, end: End, options?: TensorOptions): Tensor<ArangeRangeShape<Start, End>>;
  static arange(start: number, end: number, step: number, options?: TensorOptions): Tensor;
  static allclose(actual: TensorLike, expected: TensorLike, options?: AllCloseOptions): boolean;
  static equal(actual: TensorLike, expected: TensorLike): boolean;
  static to<const S extends TensorShapeTuple>(input: Tensor<S>, target?: TensorToTarget | TensorToOptions, options?: TensorToOptions): Tensor<S>;
  static to(input: TensorLike, target?: TensorToTarget | TensorToOptions, options?: TensorToOptions): Tensor;
  static typeAs<const S extends TensorShapeTuple>(input: Tensor<S>, other: TensorLike, options?: TensorToOptions): Tensor<S>;
  static typeAs(input: TensorLike, other: TensorLike, options?: TensorToOptions): Tensor;
  static type_as<const S extends TensorShapeTuple>(input: Tensor<S>, other: TensorLike, options?: TensorToOptions): Tensor<S>;
  static type_as(input: TensorLike, other: TensorLike, options?: TensorToOptions): Tensor;
  static clone<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static clone(input: TensorLike): Tensor;
  static detach<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static detach(input: TensorLike): Tensor;
  static reshape<const InputShape extends TensorShapeTuple, const S extends TensorShape>(input: Tensor<InputShape>, shape: S): Tensor<ReshapeShape<InputShape, S>>;
  static reshape<const S extends TensorShape>(input: TensorLike, shape: S): Tensor<TensorShapeOf<S>>;
  static view<const InputShape extends TensorShapeTuple, const S extends TensorShape>(input: Tensor<InputShape>, shape: S): Tensor<ReshapeShape<InputShape, S>>;
  static view<const S extends TensorShape>(input: TensorLike, shape: S): Tensor<TensorShapeOf<S>>;
  static broadcastTo<const S extends TensorShape>(input: TensorLike, shape: S): Tensor<TensorShapeOf<S>>;
  static expand<const S extends TensorShape>(input: TensorLike, shape: S): Tensor<TensorShapeOf<S>>;
  static repeat<const InputShape extends TensorShapeTuple, const Repeats extends TensorShape>(input: Tensor<InputShape>, repeats: Repeats): Tensor<RepeatShape<InputShape, Repeats>>;
  static repeat<const InputShape extends TensorShapeTuple, const Repeats extends TensorShapeTuple>(input: Tensor<InputShape>, ...repeats: Repeats): Tensor<RepeatShape<InputShape, Repeats>>;
  static repeat(input: TensorLike, ...repeats: number[]): Tensor;
  static tile<const InputShape extends TensorShapeTuple, const Repeats extends TensorShape>(input: Tensor<InputShape>, repeats: Repeats): Tensor<RepeatShape<InputShape, Repeats>>;
  static tile<const InputShape extends TensorShapeTuple, const Repeats extends TensorShapeTuple>(input: Tensor<InputShape>, ...repeats: Repeats): Tensor<RepeatShape<InputShape, Repeats>>;
  static tile(input: TensorLike, ...repeats: number[]): Tensor;
  static flatten<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<FlattenShape<S>>;
  static flatten<const S extends TensorShapeTuple, const StartDim extends number, const EndDim extends number>(input: Tensor<S>, startDim: StartDim, endDim: EndDim): Tensor<FlattenShape<S, StartDim, EndDim>>;
  static flatten(input: TensorLike, startDim?: number, endDim?: number): Tensor;
  static squeeze<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<SqueezeShape<S>>;
  static squeeze<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<SqueezeShape<S, Dim>>;
  static squeeze(input: TensorLike, dim?: number | null): Tensor;
  static unsqueeze<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<UnsqueezeShape<S, Dim>>;
  static unsqueeze(input: TensorLike, dim: number): Tensor;
  static transpose<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<TransposeShape<S>>;
  static transpose<const S extends TensorShapeTuple, const Dim0 extends number, const Dim1 extends number>(input: Tensor<S>, dim0: Dim0, dim1: Dim1): Tensor<TransposeShape<S, Dim0, Dim1>>;
  static transpose(input: TensorLike, dim0?: number, dim1?: number): Tensor;
  static permute<const S extends TensorShapeTuple, const Dims extends readonly number[]>(input: Tensor<S>, dims: Dims): Tensor<PermuteShape<S, Dims>>;
  static permute(input: TensorLike, dims: readonly number[]): Tensor;
  static flip<const S extends TensorShapeTuple>(input: Tensor<S>, dims: readonly number[]): Tensor<S>;
  static flip(input: TensorLike, dims: readonly number[]): Tensor;
  static roll<const S extends TensorShapeTuple>(input: Tensor<S>, shifts: TensorRollArg, dims?: TensorRollArg | null): Tensor<S>;
  static roll(input: TensorLike, shifts: TensorRollArg, dims?: TensorRollArg | null): Tensor;
  static select<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim, index: number): Tensor<SelectShape<S, Dim>>;
  static select(input: TensorLike, dim: number, index: number): Tensor;
  static narrow<const S extends TensorShapeTuple, const Dim extends number, const Length extends number>(input: Tensor<S>, dim: Dim, start: number, length: Length): Tensor<NarrowShape<S, Dim, Length>>;
  static narrow(input: TensorLike, dim: number, start: number, length: number): Tensor;
  static slice<const S extends TensorShapeTuple, const Dim extends number, const Start extends number, const End extends number>(input: Tensor<S>, dim: Dim, start: Start, end: End, step?: 1): Tensor<SliceShape<S, Dim, Start, End>>;
  static slice(input: TensorLike, dim: number, start?: number | null, end?: number | null, step?: number): Tensor;
  static indexSelect<const S extends TensorShapeTuple, const Dim extends number, const Indices extends IndexLike>(input: Tensor<S>, dim: Dim, indices: Indices): Tensor<TensorIndexSelectShape<S, Dim, Indices>>;
  static indexSelect(input: TensorLike, dim: number, indices: IndexLike): Tensor;
  static index_select<const S extends TensorShapeTuple, const Dim extends number, const Indices extends IndexLike>(input: Tensor<S>, dim: Dim, indices: Indices): Tensor<TensorIndexSelectShape<S, Dim, Indices>>;
  static index_select(input: TensorLike, dim: number, indices: IndexLike): Tensor;
  static gather<const S extends TensorShapeTuple, const Dim extends number, const Index extends IndexLike>(input: Tensor<S>, dim: Dim, index: Index): Tensor<TensorGatherShape<Index>>;
  static gather(input: TensorLike, dim: number, index: IndexLike): Tensor;
  static take<const S extends TensorShapeTuple, const Index extends IndexLike>(input: Tensor<S>, index: Index): Tensor<TensorGatherShape<Index>>;
  static take(input: TensorLike, index: IndexLike): Tensor;
  static argsort<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number, descending?: boolean): Tensor<S>;
  static argsort(input: TensorLike, dim?: number, descending?: boolean): Tensor;
  static sort<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number, descending?: boolean): TensorSortResult<S>;
  static sort(input: TensorLike, dim?: number, descending?: boolean): TensorSortResult;
  static topk<const S extends TensorShapeTuple, const K extends number>(input: Tensor<S>, k: K): TensorTopkResult<TensorTopkShape<S, -1, K>>;
  static topk<const S extends TensorShapeTuple, const K extends number, const Dim extends number>(input: Tensor<S>, k: K, dim: Dim, largest?: boolean, sorted?: boolean): TensorTopkResult<TensorTopkShape<S, Dim, K>>;
  static topk(input: TensorLike, k: number, dim?: number, largest?: boolean, sorted?: boolean): TensorTopkResult;
  static scatterAdd<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim, index: IndexLike, src: TensorLike): Tensor<S>;
  static scatterAdd(input: TensorLike, dim: number, index: IndexLike, src: TensorLike): Tensor;
  static scatter_add<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim, index: IndexLike, src: TensorLike): Tensor<S>;
  static scatter_add(input: TensorLike, dim: number, index: IndexLike, src: TensorLike): Tensor;
  static split(input: TensorLike, splitSizeOrSections: number | readonly number[], dim?: number): readonly Tensor[];
  static chunk(input: TensorLike, chunks: number, dim?: number): readonly Tensor[];
  static unbind(input: TensorLike, dim?: number): readonly Tensor[];
  static add<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
  static add<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
  static add(input: TensorLike, other: TensorLike): Tensor;
  static sub<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
  static sub<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
  static sub(input: TensorLike, other: TensorLike): Tensor;
  static mul<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
  static mul<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
  static mul(input: TensorLike, other: TensorLike): Tensor;
  static div<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
  static div<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
  static div(input: TensorLike, other: TensorLike): Tensor;
  static eq<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
  static eq<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
  static eq(input: TensorLike, other: TensorLike): Tensor;
  static ne<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
  static ne<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
  static ne(input: TensorLike, other: TensorLike): Tensor;
  static lt<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
  static lt<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
  static lt(input: TensorLike, other: TensorLike): Tensor;
  static le<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
  static le<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
  static le(input: TensorLike, other: TensorLike): Tensor;
  static gt<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
  static gt<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
  static gt(input: TensorLike, other: TensorLike): Tensor;
  static ge<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
  static ge<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
  static ge(input: TensorLike, other: TensorLike): Tensor;
  static isclose<const S extends TensorShapeTuple>(input: Tensor<S>, other: number, options?: IsCloseOptions): Tensor<S>;
  static isclose<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>, options?: IsCloseOptions): Tensor<BroadcastShape<S, OtherShape>>;
  static isclose(input: TensorLike, other: TensorLike, options?: IsCloseOptions): Tensor;
  static pow<const S extends TensorShapeTuple>(input: Tensor<S>, exponent: number): Tensor<S>;
  static pow(input: TensorLike, exponent: number): Tensor;
  static neg<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static neg(input: TensorLike): Tensor;
  static negative<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static negative(input: TensorLike): Tensor;
  static exp<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static exp(input: TensorLike): Tensor;
  static expm1<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static expm1(input: TensorLike): Tensor;
  static log<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static log(input: TensorLike): Tensor;
  static log1p<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static log1p(input: TensorLike): Tensor;
  static sqr<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static sqr(input: TensorLike): Tensor;
  static square<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static square(input: TensorLike): Tensor;
  static recip<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static recip(input: TensorLike): Tensor;
  static reciprocal<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static reciprocal(input: TensorLike): Tensor;
  static abs<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static abs(input: TensorLike): Tensor;
  static sgn<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static sgn(input: TensorLike): Tensor;
  static sign<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static sign(input: TensorLike): Tensor;
  static step<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static step(input: TensorLike): Tensor;
  static isnan<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static isnan(input: TensorLike): Tensor;
  static isinf<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static isinf(input: TensorLike): Tensor;
  static isfinite<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static isfinite(input: TensorLike): Tensor;
  static floor<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static floor(input: TensorLike): Tensor;
  static ceil<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static ceil(input: TensorLike): Tensor;
  static round<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static round(input: TensorLike): Tensor;
  static trunc<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static trunc(input: TensorLike): Tensor;
  static sqrt<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static sqrt(input: TensorLike): Tensor;
  static rsqrt<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static rsqrt(input: TensorLike): Tensor;
  static relu<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static relu(input: TensorLike): Tensor;
  static gelu<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static gelu(input: TensorLike): Tensor;
  static silu<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static silu(input: TensorLike): Tensor;
  static sigmoid<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static sigmoid(input: TensorLike): Tensor;
  static tanh<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static tanh(input: TensorLike): Tensor;
  static sin<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static sin(input: TensorLike): Tensor;
  static cos<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static cos(input: TensorLike): Tensor;
  static tan<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  static tan(input: TensorLike): Tensor;
  static maximum<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
  static maximum<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
  static maximum(input: TensorLike, other: TensorLike): Tensor;
  static minimum<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
  static minimum<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
  static minimum(input: TensorLike, other: TensorLike): Tensor;
  static where<const S extends TensorShapeTuple>(condition: Tensor<S>, input: number, other: number): Tensor<S>;
  static where<const S extends TensorShapeTuple, const InputShape extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(condition: Tensor<S>, input: Tensor<InputShape>, other: Tensor<OtherShape>): Tensor<WhereShape<S, InputShape, OtherShape>>;
  static where(condition: TensorLike, input: TensorLike, other: TensorLike): Tensor;
  static maskedFill<const S extends TensorShapeTuple>(input: Tensor<S>, mask: TensorLike, value: number): Tensor<S>;
  static maskedFill(input: TensorLike, mask: TensorLike, value: TensorLike): Tensor;
  static masked_fill<const S extends TensorShapeTuple>(input: Tensor<S>, mask: TensorLike, value: number): Tensor<S>;
  static masked_fill(input: TensorLike, mask: TensorLike, value: TensorLike): Tensor;
  static sum<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
  static sum<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
  static sum(input: TensorLike, dim?: number): Tensor;
  static prod<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
  static prod<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
  static prod(input: TensorLike, dim?: number): Tensor;
  static cumsum<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
  static cumsum(input: TensorLike, dim?: number): Tensor;
  static mean<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
  static mean<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
  static mean(input: TensorLike, dim?: number): Tensor;
  static max<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
  static max<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
  static max(input: TensorLike, dim?: number): Tensor;
  static min<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
  static min<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
  static min(input: TensorLike, dim?: number): Tensor;
  static any<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
  static any<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
  static any(input: TensorLike, dim?: number): Tensor;
  static all<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
  static all<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
  static all(input: TensorLike, dim?: number): Tensor;
  static argmax<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
  static argmax<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
  static argmax(input: TensorLike, dim?: number): Tensor;
  static argmin<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
  static argmin<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
  static argmin(input: TensorLike, dim?: number): Tensor;
  static variance<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
  static variance<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim, correction?: number): Tensor<ReductionShape<S, Dim>>;
  static variance(input: TensorLike, dim?: number, correction?: number): Tensor;
  static var<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
  static var<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim, correction?: number): Tensor<ReductionShape<S, Dim>>;
  static var(input: TensorLike, dim?: number, correction?: number): Tensor;
  static std<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
  static std<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim, correction?: number): Tensor<ReductionShape<S, Dim>>;
  static std(input: TensorLike, dim?: number, correction?: number): Tensor;
  static norm<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
  static norm<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim, p?: number): Tensor<ReductionShape<S, Dim>>;
  static norm(input: TensorLike, dim?: number, p?: number): Tensor;
  static softmax<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
  static softmax(input: TensorLike, dim?: number): Tensor;
  static softmax_dim<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
  static softmax_dim(input: TensorLike, dim?: number): Tensor;
  static softmaxDim<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
  static softmaxDim(input: TensorLike, dim?: number): Tensor;
  static logSoftmax<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
  static logSoftmax(input: TensorLike, dim?: number): Tensor;
  static log_softmax<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
  static log_softmax(input: TensorLike, dim?: number): Tensor;
  static log_softmax_dim<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
  static log_softmax_dim(input: TensorLike, dim?: number): Tensor;
  static logSoftmaxDim<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
  static logSoftmaxDim(input: TensorLike, dim?: number): Tensor;
  static logsumexp<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
  static logsumexp<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
  static logsumexp(input: TensorLike, dim?: number): Tensor;
  static logSumExp<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
  static logSumExp<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
  static logSumExp(input: TensorLike, dim?: number): Tensor;
  static clamp<const S extends TensorShapeTuple>(input: Tensor<S>, min?: number | null, max?: number | null): Tensor<S>;
  static clamp(input: TensorLike, min?: number | null, max?: number | null): Tensor;
  static clip<const S extends TensorShapeTuple>(input: Tensor<S>, min?: number | null, max?: number | null): Tensor<S>;
  static clip(input: TensorLike, min?: number | null, max?: number | null): Tensor;
  static matmul<const LhsShape extends TensorShapeTuple, const RhsShape extends TensorShapeTuple>(input: Tensor<LhsShape>, other: Tensor<RhsShape>, otherShape?: readonly number[]): Tensor<MatmulShape<LhsShape, RhsShape>>;
  static matmul(input: TensorLike, other: TensorLike, otherShape?: readonly number[]): Tensor;
  static mm<const LhsShape extends TensorShapeTuple, const RhsShape extends TensorShapeTuple>(input: Tensor<LhsShape>, other: Tensor<RhsShape>, otherShape?: readonly number[]): Tensor<MatmulShape<LhsShape, RhsShape>>;
  static mm(input: TensorLike, other: TensorLike, otherShape?: readonly number[]): Tensor;
  static dot<const LhsShape extends TensorShapeTuple, const RhsShape extends TensorShapeTuple>(input: Tensor<LhsShape>, other: Tensor<RhsShape>, otherShape?: readonly number[]): Tensor<readonly [1]>;
  static dot(input: TensorLike, other: TensorLike, otherShape?: readonly number[]): Tensor<readonly [1]>;
  static trace(input: TensorLike): Tensor<readonly [1]>;
  static diagonal<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<DiagonalShape<S>>;
  static diagonal(input: TensorLike): Tensor;
  static bmm<const LhsShape extends TensorShapeTuple, const RhsShape extends TensorShapeTuple>(input: Tensor<LhsShape>, other: Tensor<RhsShape>, otherShape?: readonly number[]): Tensor<BmmShape<LhsShape, RhsShape>>;
  static bmm(input: TensorLike, other: TensorLike, otherShape?: readonly number[]): Tensor;
  static cat<const Tensors extends readonly [Tensor, ...Tensor[]]>(tensors: Tensors): Tensor<TensorCatShape<Tensors>>;
  static cat<const Tensors extends readonly [Tensor, ...Tensor[]], const Dim extends number>(tensors: Tensors, dim: Dim): Tensor<TensorCatShape<Tensors, Dim>>;
  static cat(tensors: readonly Tensor[], dim?: number): Tensor;
  static concat<const Tensors extends readonly [Tensor, ...Tensor[]]>(tensors: Tensors): Tensor<TensorCatShape<Tensors>>;
  static concat<const Tensors extends readonly [Tensor, ...Tensor[]], const Dim extends number>(tensors: Tensors, dim: Dim): Tensor<TensorCatShape<Tensors, Dim>>;
  static concat(tensors: readonly Tensor[], dim?: number): Tensor;
  static concatenate<const Tensors extends readonly [Tensor, ...Tensor[]]>(tensors: Tensors): Tensor<TensorCatShape<Tensors>>;
  static concatenate<const Tensors extends readonly [Tensor, ...Tensor[]], const Dim extends number>(tensors: Tensors, dim: Dim): Tensor<TensorCatShape<Tensors, Dim>>;
  static concatenate(tensors: readonly Tensor[], dim?: number): Tensor;
  static stack<const Tensors extends readonly [Tensor, ...Tensor[]]>(tensors: Tensors): Tensor<TensorStackShape<Tensors>>;
  static stack<const Tensors extends readonly [Tensor, ...Tensor[]], const Dim extends number>(tensors: Tensors, dim: Dim): Tensor<TensorStackShape<Tensors, Dim>>;
  static stack(tensors: readonly Tensor[], dim?: number): Tensor;
  static vstack<const Tensors extends readonly [Tensor, ...Tensor[]]>(tensors: Tensors): Tensor<TensorVStackShape<Tensors>>;
  static vstack(tensors: readonly Tensor[]): Tensor;
  static hstack<const Tensors extends readonly [Tensor, ...Tensor[]]>(tensors: Tensors): Tensor<TensorHStackShape<Tensors>>;
  static hstack(tensors: readonly Tensor[]): Tensor;
  static einsum<const Equation extends string, const Operands extends readonly [Tensor, ...Tensor[]]>(equation: Equation, operands: Operands): Tensor<EinsumShape<Equation, Operands>>;
  static einsum<const Equation extends string, const Operands extends readonly [Tensor, ...Tensor[]]>(equation: Equation, ...operands: Operands): Tensor<EinsumShape<Equation, Operands>>;
  static einsum(equation: string, operands: readonly Tensor[]): Tensor;
  static einsum(equation: string, ...operands: readonly Tensor[]): Tensor;
  static hasShape<const S extends TensorShape>(input: Tensor, shape: S): input is Tensor<TensorShapeOf<S>>;
  static hasShape(input: TensorLike, shape: TensorShape): boolean;
  static requireShape<const S extends TensorShape>(input: TensorLike, shape: S): Tensor<TensorShapeOf<S>>;
  static fromJSON(json: TensorJSON): Tensor;
  static fromNativeBuffer<const S extends TensorShape>(buffer: NativeBuffer, shape: S, options?: TensorFromNativeBufferOptions): Tensor<TensorShapeOf<S>>;
  static fromNativeBuffer(buffer: NativeBuffer, options?: TensorFromNativeBufferOptions): Tensor;
}

export declare function tensor<const S extends TensorShapeTuple>(data: TensorLike, shape: S, options?: TensorOptions): Tensor<S>;
export declare function tensor(data: TensorLike): Tensor;
export declare const asTensor: typeof tensor;
export declare const as_tensor: typeof tensor;
export declare const asarray: typeof tensor;
export declare const fromNumpy: typeof tensor;
export declare const from_numpy: typeof tensor;
export declare function parameter<const S extends TensorShapeTuple>(data: TensorLike, shape: S, options?: TensorOptions): Tensor<S>;
export declare function parameter(data: TensorLike, options?: TensorOptions): Tensor;
export declare const param: typeof parameter;
export declare function cat<const Tensors extends readonly [Tensor, ...Tensor[]]>(tensors: Tensors): Tensor<TensorCatShape<Tensors>>;
export declare function cat<const Tensors extends readonly [Tensor, ...Tensor[]], const Dim extends number>(tensors: Tensors, dim: Dim): Tensor<TensorCatShape<Tensors, Dim>>;
export declare function cat(tensors: readonly Tensor[], dim?: number): Tensor;
export declare function concat<const Tensors extends readonly [Tensor, ...Tensor[]]>(tensors: Tensors): Tensor<TensorCatShape<Tensors>>;
export declare function concat<const Tensors extends readonly [Tensor, ...Tensor[]], const Dim extends number>(tensors: Tensors, dim: Dim): Tensor<TensorCatShape<Tensors, Dim>>;
export declare function concat(tensors: readonly Tensor[], dim?: number): Tensor;
export declare function concatenate<const Tensors extends readonly [Tensor, ...Tensor[]]>(tensors: Tensors): Tensor<TensorCatShape<Tensors>>;
export declare function concatenate<const Tensors extends readonly [Tensor, ...Tensor[]], const Dim extends number>(tensors: Tensors, dim: Dim): Tensor<TensorCatShape<Tensors, Dim>>;
export declare function concatenate(tensors: readonly Tensor[], dim?: number): Tensor;
export declare function stack<const Tensors extends readonly [Tensor, ...Tensor[]]>(tensors: Tensors): Tensor<TensorStackShape<Tensors>>;
export declare function stack<const Tensors extends readonly [Tensor, ...Tensor[]], const Dim extends number>(tensors: Tensors, dim: Dim): Tensor<TensorStackShape<Tensors, Dim>>;
export declare function stack(tensors: readonly Tensor[], dim?: number): Tensor;
export declare function vstack<const Tensors extends readonly [Tensor, ...Tensor[]]>(tensors: Tensors): Tensor<TensorVStackShape<Tensors>>;
export declare function vstack(tensors: readonly Tensor[]): Tensor;
export declare function hstack<const Tensors extends readonly [Tensor, ...Tensor[]]>(tensors: Tensors): Tensor<TensorHStackShape<Tensors>>;
export declare function hstack(tensors: readonly Tensor[]): Tensor;
export declare function einsum<const Equation extends string, const Operands extends readonly [Tensor, ...Tensor[]]>(equation: Equation, operands: Operands): Tensor<EinsumShape<Equation, Operands>>;
export declare function einsum<const Equation extends string, const Operands extends readonly [Tensor, ...Tensor[]]>(equation: Equation, ...operands: Operands): Tensor<EinsumShape<Equation, Operands>>;
export declare function einsum(equation: string, operands: readonly Tensor[]): Tensor;
export declare function einsum(equation: string, ...operands: readonly Tensor[]): Tensor;
export declare function allclose(actual: TensorLike, expected: TensorLike, options?: AllCloseOptions): boolean;
export declare function equal(actual: TensorLike, expected: TensorLike): boolean;
export declare function to<const S extends TensorShapeTuple>(input: Tensor<S>, target?: TensorToTarget | TensorToOptions, options?: TensorToOptions): Tensor<S>;
export declare function to(input: TensorLike, target?: TensorToTarget | TensorToOptions, options?: TensorToOptions): Tensor;
export declare function typeAs<const S extends TensorShapeTuple>(input: Tensor<S>, other: TensorLike, options?: TensorToOptions): Tensor<S>;
export declare function typeAs(input: TensorLike, other: TensorLike, options?: TensorToOptions): Tensor;
export declare function type_as<const S extends TensorShapeTuple>(input: Tensor<S>, other: TensorLike, options?: TensorToOptions): Tensor<S>;
export declare function type_as(input: TensorLike, other: TensorLike, options?: TensorToOptions): Tensor;
export declare function cpu<const S extends TensorShapeTuple>(input: Tensor<S>, options?: TensorToOptions): Tensor<S>;
export declare function cpu(input: TensorLike, options?: TensorToOptions): Tensor;
export declare function float<const S extends TensorShapeTuple>(input: Tensor<S>, options?: TensorToOptions): Tensor<S>;
export declare function float(input: TensorLike, options?: TensorToOptions): Tensor;
export declare function float32<const S extends TensorShapeTuple>(input: Tensor<S>, options?: TensorToOptions): Tensor<S>;
export declare function float32(input: TensorLike, options?: TensorToOptions): Tensor;
export declare function clone<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function clone(input: TensorLike): Tensor;
export declare function detach<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function detach(input: TensorLike): Tensor;
export declare function reshape<const InputShape extends TensorShapeTuple, const S extends TensorShape>(input: Tensor<InputShape>, shape: S): Tensor<ReshapeShape<InputShape, S>>;
export declare function reshape<const S extends TensorShape>(input: TensorLike, shape: S): Tensor<TensorShapeOf<S>>;
export declare function view<const InputShape extends TensorShapeTuple, const S extends TensorShape>(input: Tensor<InputShape>, shape: S): Tensor<ReshapeShape<InputShape, S>>;
export declare function view<const S extends TensorShape>(input: TensorLike, shape: S): Tensor<TensorShapeOf<S>>;
export declare function broadcastTo<const S extends TensorShape>(input: TensorLike, shape: S): Tensor<TensorShapeOf<S>>;
export declare function expand<const S extends TensorShape>(input: TensorLike, shape: S): Tensor<TensorShapeOf<S>>;
export declare function repeat<const InputShape extends TensorShapeTuple, const Repeats extends TensorShape>(input: Tensor<InputShape>, repeats: Repeats): Tensor<RepeatShape<InputShape, Repeats>>;
export declare function repeat<const InputShape extends TensorShapeTuple, const Repeats extends TensorShapeTuple>(input: Tensor<InputShape>, ...repeats: Repeats): Tensor<RepeatShape<InputShape, Repeats>>;
export declare function repeat(input: TensorLike, ...repeats: number[]): Tensor;
export declare function tile<const InputShape extends TensorShapeTuple, const Repeats extends TensorShape>(input: Tensor<InputShape>, repeats: Repeats): Tensor<RepeatShape<InputShape, Repeats>>;
export declare function tile<const InputShape extends TensorShapeTuple, const Repeats extends TensorShapeTuple>(input: Tensor<InputShape>, ...repeats: Repeats): Tensor<RepeatShape<InputShape, Repeats>>;
export declare function tile(input: TensorLike, ...repeats: number[]): Tensor;
export declare function flatten<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<FlattenShape<S>>;
export declare function flatten<const S extends TensorShapeTuple, const StartDim extends number, const EndDim extends number>(input: Tensor<S>, startDim: StartDim, endDim: EndDim): Tensor<FlattenShape<S, StartDim, EndDim>>;
export declare function flatten(input: TensorLike, startDim?: number, endDim?: number): Tensor;
export declare function squeeze<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<SqueezeShape<S>>;
export declare function squeeze<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<SqueezeShape<S, Dim>>;
export declare function squeeze(input: TensorLike, dim?: number | null): Tensor;
export declare function unsqueeze<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<UnsqueezeShape<S, Dim>>;
export declare function unsqueeze(input: TensorLike, dim: number): Tensor;
export declare function transpose<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<TransposeShape<S>>;
export declare function transpose<const S extends TensorShapeTuple, const Dim0 extends number, const Dim1 extends number>(input: Tensor<S>, dim0: Dim0, dim1: Dim1): Tensor<TransposeShape<S, Dim0, Dim1>>;
export declare function transpose(input: TensorLike, dim0?: number, dim1?: number): Tensor;
export declare function permute<const S extends TensorShapeTuple, const Dims extends readonly number[]>(input: Tensor<S>, dims: Dims): Tensor<PermuteShape<S, Dims>>;
export declare function permute(input: TensorLike, dims: readonly number[]): Tensor;
export declare function flip<const S extends TensorShapeTuple>(input: Tensor<S>, dims: readonly number[]): Tensor<S>;
export declare function flip(input: TensorLike, dims: readonly number[]): Tensor;
export declare function roll<const S extends TensorShapeTuple>(input: Tensor<S>, shifts: TensorRollArg, dims?: TensorRollArg | null): Tensor<S>;
export declare function roll(input: TensorLike, shifts: TensorRollArg, dims?: TensorRollArg | null): Tensor;
export declare function select<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim, index: number): Tensor<SelectShape<S, Dim>>;
export declare function select(input: TensorLike, dim: number, index: number): Tensor;
export declare function narrow<const S extends TensorShapeTuple, const Dim extends number, const Length extends number>(input: Tensor<S>, dim: Dim, start: number, length: Length): Tensor<NarrowShape<S, Dim, Length>>;
export declare function narrow(input: TensorLike, dim: number, start: number, length: number): Tensor;
export declare function slice<const S extends TensorShapeTuple, const Dim extends number, const Start extends number, const End extends number>(input: Tensor<S>, dim: Dim, start: Start, end: End, step?: 1): Tensor<SliceShape<S, Dim, Start, End>>;
export declare function slice(input: TensorLike, dim: number, start?: number | null, end?: number | null, step?: number): Tensor;
export declare function indexSelect<const S extends TensorShapeTuple, const Dim extends number, const Indices extends IndexLike>(input: Tensor<S>, dim: Dim, indices: Indices): Tensor<TensorIndexSelectShape<S, Dim, Indices>>;
export declare function indexSelect(input: TensorLike, dim: number, indices: IndexLike): Tensor;
export declare function index_select<const S extends TensorShapeTuple, const Dim extends number, const Indices extends IndexLike>(input: Tensor<S>, dim: Dim, indices: Indices): Tensor<TensorIndexSelectShape<S, Dim, Indices>>;
export declare function index_select(input: TensorLike, dim: number, indices: IndexLike): Tensor;
export declare function gather<const S extends TensorShapeTuple, const Dim extends number, const Index extends IndexLike>(input: Tensor<S>, dim: Dim, index: Index): Tensor<TensorGatherShape<Index>>;
export declare function gather(input: TensorLike, dim: number, index: IndexLike): Tensor;
export declare function take<const S extends TensorShapeTuple, const Index extends IndexLike>(input: Tensor<S>, index: Index): Tensor<TensorGatherShape<Index>>;
export declare function take(input: TensorLike, index: IndexLike): Tensor;
export declare function argsort<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number, descending?: boolean): Tensor<S>;
export declare function argsort(input: TensorLike, dim?: number, descending?: boolean): Tensor;
export declare function sort<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number, descending?: boolean): TensorSortResult<S>;
export declare function sort(input: TensorLike, dim?: number, descending?: boolean): TensorSortResult;
export declare function topk<const S extends TensorShapeTuple, const K extends number>(input: Tensor<S>, k: K): TensorTopkResult<TensorTopkShape<S, -1, K>>;
export declare function topk<const S extends TensorShapeTuple, const K extends number, const Dim extends number>(input: Tensor<S>, k: K, dim: Dim, largest?: boolean, sorted?: boolean): TensorTopkResult<TensorTopkShape<S, Dim, K>>;
export declare function topk(input: TensorLike, k: number, dim?: number, largest?: boolean, sorted?: boolean): TensorTopkResult;
export declare function scatterAdd<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim, index: IndexLike, src: TensorLike): Tensor<S>;
export declare function scatterAdd(input: TensorLike, dim: number, index: IndexLike, src: TensorLike): Tensor;
export declare function scatter_add<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim, index: IndexLike, src: TensorLike): Tensor<S>;
export declare function scatter_add(input: TensorLike, dim: number, index: IndexLike, src: TensorLike): Tensor;
export declare function split(input: TensorLike, splitSizeOrSections: number | readonly number[], dim?: number): readonly Tensor[];
export declare function chunk(input: TensorLike, chunks: number, dim?: number): readonly Tensor[];
export declare function unbind(input: TensorLike, dim?: number): readonly Tensor[];
export declare function add<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
export declare function add<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
export declare function add(input: TensorLike, other: TensorLike): Tensor;
export declare function sub<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
export declare function sub<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
export declare function sub(input: TensorLike, other: TensorLike): Tensor;
export declare function mul<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
export declare function mul<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
export declare function mul(input: TensorLike, other: TensorLike): Tensor;
export declare function div<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
export declare function div<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
export declare function div(input: TensorLike, other: TensorLike): Tensor;
export declare function eq<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
export declare function eq<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
export declare function eq(input: TensorLike, other: TensorLike): Tensor;
export declare function ne<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
export declare function ne<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
export declare function ne(input: TensorLike, other: TensorLike): Tensor;
export declare function lt<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
export declare function lt<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
export declare function lt(input: TensorLike, other: TensorLike): Tensor;
export declare function le<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
export declare function le<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
export declare function le(input: TensorLike, other: TensorLike): Tensor;
export declare function gt<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
export declare function gt<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
export declare function gt(input: TensorLike, other: TensorLike): Tensor;
export declare function ge<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
export declare function ge<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
export declare function ge(input: TensorLike, other: TensorLike): Tensor;
export declare function isclose<const S extends TensorShapeTuple>(input: Tensor<S>, other: number, options?: IsCloseOptions): Tensor<S>;
export declare function isclose<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>, options?: IsCloseOptions): Tensor<BroadcastShape<S, OtherShape>>;
export declare function isclose(input: TensorLike, other: TensorLike, options?: IsCloseOptions): Tensor;
export declare function pow<const S extends TensorShapeTuple>(input: Tensor<S>, exponent: number): Tensor<S>;
export declare function pow(input: TensorLike, exponent: number): Tensor;
export declare function neg<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function neg(input: TensorLike): Tensor;
export declare function negative<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function negative(input: TensorLike): Tensor;
export declare function exp<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function exp(input: TensorLike): Tensor;
export declare function expm1<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function expm1(input: TensorLike): Tensor;
export declare function log<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function log(input: TensorLike): Tensor;
export declare function log1p<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function log1p(input: TensorLike): Tensor;
export declare function sqr<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function sqr(input: TensorLike): Tensor;
export declare function square<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function square(input: TensorLike): Tensor;
export declare function recip<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function recip(input: TensorLike): Tensor;
export declare function reciprocal<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function reciprocal(input: TensorLike): Tensor;
export declare function abs<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function abs(input: TensorLike): Tensor;
export declare function sgn<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function sgn(input: TensorLike): Tensor;
export declare function sign<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function sign(input: TensorLike): Tensor;
export declare function step<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function step(input: TensorLike): Tensor;
export declare function isnan<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function isnan(input: TensorLike): Tensor;
export declare function isinf<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function isinf(input: TensorLike): Tensor;
export declare function isfinite<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function isfinite(input: TensorLike): Tensor;
export declare function floor<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function floor(input: TensorLike): Tensor;
export declare function ceil<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function ceil(input: TensorLike): Tensor;
export declare function round<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function round(input: TensorLike): Tensor;
export declare function trunc<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function trunc(input: TensorLike): Tensor;
export declare function sqrt<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function sqrt(input: TensorLike): Tensor;
export declare function rsqrt<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function rsqrt(input: TensorLike): Tensor;
export declare function relu<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function relu(input: TensorLike): Tensor;
export declare function gelu<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function gelu(input: TensorLike): Tensor;
export declare function silu<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function silu(input: TensorLike): Tensor;
export declare function sigmoid<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function sigmoid(input: TensorLike): Tensor;
export declare function tanh<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function tanh(input: TensorLike): Tensor;
export declare function sin<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function sin(input: TensorLike): Tensor;
export declare function cos<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function cos(input: TensorLike): Tensor;
export declare function tan<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
export declare function tan(input: TensorLike): Tensor;
export declare function maximum<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
export declare function maximum<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
export declare function maximum(input: TensorLike, other: TensorLike): Tensor;
export declare function minimum<const S extends TensorShapeTuple>(input: Tensor<S>, other: number): Tensor<S>;
export declare function minimum<const S extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<S>, other: Tensor<OtherShape>): Tensor<BroadcastShape<S, OtherShape>>;
export declare function minimum(input: TensorLike, other: TensorLike): Tensor;
export declare function where<const S extends TensorShapeTuple>(condition: Tensor<S>, input: number, other: number): Tensor<S>;
export declare function where<const S extends TensorShapeTuple, const InputShape extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(condition: Tensor<S>, input: Tensor<InputShape>, other: Tensor<OtherShape>): Tensor<WhereShape<S, InputShape, OtherShape>>;
export declare function where(condition: TensorLike, input: TensorLike, other: TensorLike): Tensor;
export declare function maskedFill<const S extends TensorShapeTuple>(input: Tensor<S>, mask: TensorLike, value: number): Tensor<S>;
export declare function maskedFill(input: TensorLike, mask: TensorLike, value: TensorLike): Tensor;
export declare function masked_fill<const S extends TensorShapeTuple>(input: Tensor<S>, mask: TensorLike, value: number): Tensor<S>;
export declare function masked_fill(input: TensorLike, mask: TensorLike, value: TensorLike): Tensor;
export declare function sum<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
export declare function sum<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
export declare function sum(input: TensorLike, dim?: number): Tensor;
export declare function prod<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
export declare function prod<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
export declare function prod(input: TensorLike, dim?: number): Tensor;
export declare function cumsum<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
export declare function cumsum(input: TensorLike, dim?: number): Tensor;
export declare function mean<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
export declare function mean<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
export declare function mean(input: TensorLike, dim?: number): Tensor;
export declare function max<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
export declare function max<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
export declare function max(input: TensorLike, dim?: number): Tensor;
export declare function min<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
export declare function min<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
export declare function min(input: TensorLike, dim?: number): Tensor;
export declare function any<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
export declare function any<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
export declare function any(input: TensorLike, dim?: number): Tensor;
export declare function all<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
export declare function all<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
export declare function all(input: TensorLike, dim?: number): Tensor;
export declare function argmax<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
export declare function argmax<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
export declare function argmax(input: TensorLike, dim?: number): Tensor;
export declare function argmin<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
export declare function argmin<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
export declare function argmin(input: TensorLike, dim?: number): Tensor;
export declare function variance<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
export declare function variance<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim, correction?: number): Tensor<ReductionShape<S, Dim>>;
export declare function variance(input: TensorLike, dim?: number, correction?: number): Tensor;
export declare function std<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
export declare function std<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim, correction?: number): Tensor<ReductionShape<S, Dim>>;
export declare function std(input: TensorLike, dim?: number, correction?: number): Tensor;
export declare function norm<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
export declare function norm<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim, p?: number): Tensor<ReductionShape<S, Dim>>;
export declare function norm(input: TensorLike, dim?: number, p?: number): Tensor;
export declare function softmax<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
export declare function softmax(input: TensorLike, dim?: number): Tensor;
export declare function softmax_dim<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
export declare function softmax_dim(input: TensorLike, dim?: number): Tensor;
export declare function softmaxDim<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
export declare function softmaxDim(input: TensorLike, dim?: number): Tensor;
export declare function logSoftmax<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
export declare function logSoftmax(input: TensorLike, dim?: number): Tensor;
export declare function log_softmax<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
export declare function log_softmax(input: TensorLike, dim?: number): Tensor;
export declare function log_softmax_dim<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
export declare function log_softmax_dim(input: TensorLike, dim?: number): Tensor;
export declare function logSoftmaxDim<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
export declare function logSoftmaxDim(input: TensorLike, dim?: number): Tensor;
export declare function logsumexp<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
export declare function logsumexp<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
export declare function logsumexp(input: TensorLike, dim?: number): Tensor;
export declare function logSumExp<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S>>;
export declare function logSumExp<const S extends TensorShapeTuple, const Dim extends number>(input: Tensor<S>, dim: Dim): Tensor<ReductionShape<S, Dim>>;
export declare function logSumExp(input: TensorLike, dim?: number): Tensor;
export declare function clamp<const S extends TensorShapeTuple>(input: Tensor<S>, min?: number | null, max?: number | null): Tensor<S>;
export declare function clamp(input: TensorLike, min?: number | null, max?: number | null): Tensor;
export declare function clip<const S extends TensorShapeTuple>(input: Tensor<S>, min?: number | null, max?: number | null): Tensor<S>;
export declare function clip(input: TensorLike, min?: number | null, max?: number | null): Tensor;
export declare function matmul<const LhsShape extends TensorShapeTuple, const RhsShape extends TensorShapeTuple>(input: Tensor<LhsShape>, other: Tensor<RhsShape>, otherShape?: readonly number[]): Tensor<MatmulShape<LhsShape, RhsShape>>;
export declare function matmul(input: TensorLike, other: TensorLike, otherShape?: readonly number[]): Tensor;
export declare function mm<const LhsShape extends TensorShapeTuple, const RhsShape extends TensorShapeTuple>(input: Tensor<LhsShape>, other: Tensor<RhsShape>, otherShape?: readonly number[]): Tensor<MatmulShape<LhsShape, RhsShape>>;
export declare function mm(input: TensorLike, other: TensorLike, otherShape?: readonly number[]): Tensor;
export declare function dot<const LhsShape extends TensorShapeTuple, const RhsShape extends TensorShapeTuple>(input: Tensor<LhsShape>, other: Tensor<RhsShape>, otherShape?: readonly number[]): Tensor<readonly [1]>;
export declare function dot(input: TensorLike, other: TensorLike, otherShape?: readonly number[]): Tensor<readonly [1]>;
export declare function trace(input: TensorLike): Tensor<readonly [1]>;
export declare function diagonal<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<DiagonalShape<S>>;
export declare function diagonal(input: TensorLike): Tensor;
export declare function bmm<const LhsShape extends TensorShapeTuple, const RhsShape extends TensorShapeTuple>(input: Tensor<LhsShape>, other: Tensor<RhsShape>, otherShape?: readonly number[]): Tensor<BmmShape<LhsShape, RhsShape>>;
export declare function bmm(input: TensorLike, other: TensorLike, otherShape?: readonly number[]): Tensor;
export declare function hasShape<const S extends TensorShape>(input: Tensor, shape: S): input is Tensor<TensorShapeOf<S>>;
export declare function hasShape(input: TensorLike, shape: TensorShape): boolean;
export declare function requireShape<const S extends TensorShape>(input: TensorLike, shape: S): Tensor<TensorShapeOf<S>>;
export declare function isGradEnabled(): boolean;
export declare function is_grad_enabled(): boolean;
export declare function setGradEnabled(enabled: boolean): boolean;
export declare function set_grad_enabled(enabled: boolean): boolean;
export declare function noGrad<T>(fn: () => T): T;
export declare function no_grad<T>(fn: () => T): T;
export declare function inferenceMode<T>(fn: () => T): T;
export declare function inference_mode<T>(fn: () => T): T;
export declare function enableGrad<T>(fn: () => T): T;
export declare function enable_grad<T>(fn: () => T): T;
export type GradModeNamespace = Readonly<{
  isGradEnabled: typeof isGradEnabled;
  is_grad_enabled: typeof is_grad_enabled;
  setGradEnabled: typeof setGradEnabled;
  set_grad_enabled: typeof set_grad_enabled;
  noGrad: typeof noGrad;
  no_grad: typeof no_grad;
  inferenceMode: typeof inferenceMode;
  inference_mode: typeof inference_mode;
  enableGrad: typeof enableGrad;
  enable_grad: typeof enable_grad;
}>;
export declare const gradMode: GradModeNamespace;
export declare function scalar(value: number, options?: TensorOptions): Tensor<readonly [1]>;
export declare function empty<const S extends TensorShape>(shape: S, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
export declare function emptyLike<const S extends TensorShapeTuple>(input: Tensor<S>, options?: TensorOptions): Tensor<S>;
export declare function emptyLike(input: TensorLike, options?: TensorOptions): Tensor;
export declare function empty_like<const S extends TensorShapeTuple>(input: Tensor<S>, options?: TensorOptions): Tensor<S>;
export declare function empty_like(input: TensorLike, options?: TensorOptions): Tensor;
export declare function zeros<const S extends TensorShape>(shape: S, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
export declare function zerosLike<const S extends TensorShapeTuple>(input: Tensor<S>, options?: TensorOptions): Tensor<S>;
export declare function zerosLike(input: TensorLike, options?: TensorOptions): Tensor;
export declare function zeros_like<const S extends TensorShapeTuple>(input: Tensor<S>, options?: TensorOptions): Tensor<S>;
export declare function zeros_like(input: TensorLike, options?: TensorOptions): Tensor;
export declare function ones<const S extends TensorShape>(shape: S, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
export declare function onesLike<const S extends TensorShapeTuple>(input: Tensor<S>, options?: TensorOptions): Tensor<S>;
export declare function onesLike(input: TensorLike, options?: TensorOptions): Tensor;
export declare function ones_like<const S extends TensorShapeTuple>(input: Tensor<S>, options?: TensorOptions): Tensor<S>;
export declare function ones_like(input: TensorLike, options?: TensorOptions): Tensor;
export declare function eye<const N extends number>(size: N, options?: TensorOptions): Tensor<readonly [N, N]>;
export declare function full<const S extends TensorShape>(shape: S, value: number, options?: TensorOptions): Tensor<TensorShapeOf<S>>;
export declare function fullLike<const S extends TensorShapeTuple>(input: Tensor<S>, value: number, options?: TensorOptions): Tensor<S>;
export declare function fullLike(input: TensorLike, value: number, options?: TensorOptions): Tensor;
export declare function full_like<const S extends TensorShapeTuple>(input: Tensor<S>, value: number, options?: TensorOptions): Tensor<S>;
export declare function full_like(input: TensorLike, value: number, options?: TensorOptions): Tensor;
export declare function rand<const S extends TensorShape>(shape: S, options?: RandomUniformTensorOptions): Tensor<TensorShapeOf<S>>;
export declare function randLike<const S extends TensorShapeTuple>(input: Tensor<S>, options?: RandomUniformTensorOptions): Tensor<S>;
export declare function randLike(input: TensorLike, options?: RandomUniformTensorOptions): Tensor;
export declare function rand_like<const S extends TensorShapeTuple>(input: Tensor<S>, options?: RandomUniformTensorOptions): Tensor<S>;
export declare function rand_like(input: TensorLike, options?: RandomUniformTensorOptions): Tensor;
export declare function randn<const S extends TensorShape>(shape: S, options?: RandomTensorOptions): Tensor<TensorShapeOf<S>>;
export declare function randnLike<const S extends TensorShapeTuple>(input: Tensor<S>, options?: RandomTensorOptions): Tensor<S>;
export declare function randnLike(input: TensorLike, options?: RandomTensorOptions): Tensor;
export declare function randn_like<const S extends TensorShapeTuple>(input: Tensor<S>, options?: RandomTensorOptions): Tensor<S>;
export declare function randn_like(input: TensorLike, options?: RandomTensorOptions): Tensor;
export declare function randInt<const S extends TensorShape>(high: number, shape: S, options?: RandomIntTensorOptions): Tensor<TensorShapeOf<S>>;
export declare function randInt<const S extends TensorShape>(low: number, high: number, shape: S, options?: RandomIntTensorOptions): Tensor<TensorShapeOf<S>>;
export declare function randint<const S extends TensorShape>(high: number, shape: S, options?: RandomIntTensorOptions): Tensor<TensorShapeOf<S>>;
export declare function randint<const S extends TensorShape>(low: number, high: number, shape: S, options?: RandomIntTensorOptions): Tensor<TensorShapeOf<S>>;
export declare function randPerm<const N extends number>(size: N, options?: RandomIntTensorOptions): Tensor<readonly [N]>;
export declare function randperm<const N extends number>(size: N, options?: RandomIntTensorOptions): Tensor<readonly [N]>;
export declare function manualSeed(seed: number): number;
export declare function manual_seed(seed: number): number;
export declare function initialSeed(): number | null;
export declare function initial_seed(): number | null;
export declare function seededRng(seed: number): () => number;
export type NativeEagerLinearIntoOptions = Readonly<{
  bias?: TensorLike | Float32Array | null;
  batch?: number;
  inFeatures?: number;
  in_features?: number;
  outFeatures?: number;
  out_features?: number;
  weightLayout?: "in-out" | "out-in" | "pytorch" | "torch";
  weight_layout?: "in-out" | "out-in" | "pytorch" | "torch";
}>;
export type NativeEagerLinearActivationIntoOptions = NativeEagerLinearIntoOptions & Readonly<{
  activation: "relu" | "gelu" | "silu" | "sigmoid" | "tanh";
}>;
export type NativeEagerActivationIntoOptions = Readonly<{
  activation: "relu" | "gelu" | "silu" | "sigmoid" | "tanh";
}>;
export type NativeEagerElementwiseOp =
  | "add" | "sub" | "mul" | "div"
  | "neg" | "negative"
  | "exp" | "log"
  | "expm1" | "expMinusOne"
  | "log1p" | "logOnePlus"
  | "sqr" | "square"
  | "recip" | "reciprocal"
  | "abs" | "sqrt" | "rsqrt" | "reciprocal_sqrt" | "reciprocalSqrt"
  | "maximum" | "max"
  | "minimum" | "min"
  | "eq" | "equal"
  | "ne" | "not_equal" | "notEqual"
  | "lt" | "less"
  | "le" | "less_equal" | "lessEqual"
  | "gt" | "greater"
  | "ge" | "greater_equal" | "greaterEqual"
  | "sign" | "sgn"
  | "step"
  | "floor" | "ceil" | "round" | "trunc" | "truncate"
  | "isnan" | "isNaN"
  | "isinf" | "isInf" | "is_infinite" | "isInfinite"
  | "isfinite" | "isFinite";
export type NativeEagerElementwiseIntoOptions = Readonly<{
  op: NativeEagerElementwiseOp;
  rows?: number;
  cols?: number;
  broadcast?: "lhs" | "rhs";
}>;
export type NativeEagerReduceOp = "sum" | "mean" | "max" | "min" | "prod";
export type NativeEagerReduceIntoOptions = Readonly<{
  op: NativeEagerReduceOp;
}>;
export type NativeEagerSoftmaxIntoOptions = Readonly<{
  rows?: number;
  cols?: number;
  dim?: number;
}>;
export type NativeEagerMatmulIntoOptions = Readonly<{
  rows?: number;
  shared?: number;
  cols?: number;
  lhsRows?: number;
  lhs_rows?: number;
  lhsCols?: number;
  lhs_cols?: number;
  rhsRows?: number;
  rhs_rows?: number;
  rhsCols?: number;
  rhs_cols?: number;
}>;
export type NativeEagerBmmIntoOptions = NativeEagerMatmulIntoOptions & Readonly<{
  batch?: number;
  batches?: number;
}>;
export type NativeEagerConv2dIntoOptions = Readonly<{
  bias?: TensorLike | Float32Array | null;
  batch?: number;
  inChannels?: number;
  in_channels?: number;
  height?: number;
  width?: number;
  outChannels?: number;
  out_channels?: number;
  kernelH?: number;
  kernel_h?: number;
  kernelW?: number;
  kernel_w?: number;
  strideH?: number;
  stride_h?: number;
  strideW?: number;
  stride_w?: number;
  paddingH?: number;
  padding_h?: number;
  paddingW?: number;
  padding_w?: number;
  dilationH?: number;
  dilation_h?: number;
  dilationW?: number;
  dilation_w?: number;
  outH?: number;
  out_h?: number;
  outW?: number;
  out_w?: number;
}>;
export type NativeEagerPool2dOp = "max" | "avg" | "average";
export type NativeEagerPool2dIntoOptions = Readonly<{
  op: NativeEagerPool2dOp;
  batch?: number;
  channels?: number;
  height?: number;
  width?: number;
  kernelH?: number;
  kernel_h?: number;
  kernelW?: number;
  kernel_w?: number;
  strideH?: number;
  stride_h?: number;
  strideW?: number;
  stride_w?: number;
  paddingH?: number;
  padding_h?: number;
  paddingW?: number;
  padding_w?: number;
  dilationH?: number;
  dilation_h?: number;
  dilationW?: number;
  dilation_w?: number;
  outH?: number;
  out_h?: number;
  outW?: number;
  out_w?: number;
  ceilMode?: boolean;
  ceil_mode?: boolean;
  countIncludePad?: boolean;
  count_include_pad?: boolean;
}>;
export type NativeEagerRoutingActivation = "relu" | "gelu" | "silu" | "sigmoid" | "tanh";
export type NativeEagerRoutingPolicy = Readonly<{
  kind: "zgml.native-eager-routing-policy";
  runtime: "node" | "bun";
  source: "src/ts/adapters/native_eager_routing_policy.ts";
  nativeCore: "zig-c-abi";
  tensorMath: Readonly<{
    matmul: "native";
    softmax: "native";
    elementwiseMinLength: number;
    activationMinLength: number;
    reduceMinLength: number;
    disabledActivations: readonly NativeEagerRoutingActivation[];
  }>;
  signature: string;
}>;
export type PublicNativeEagerNamespace = Readonly<{
  routingPolicy: NativeEagerRoutingPolicy;
  routing_policy: NativeEagerRoutingPolicy;
  linearInto(output: Float32Array, input: TensorLike, weights: TensorLike, options?: NativeEagerLinearIntoOptions): Float32Array;
  linear_into(output: Float32Array, input: TensorLike, weights: TensorLike, options?: NativeEagerLinearIntoOptions): Float32Array;
  linearActivationInto(output: Float32Array, input: TensorLike, weights: TensorLike, options: NativeEagerLinearActivationIntoOptions): Float32Array;
  linear_activation_into(output: Float32Array, input: TensorLike, weights: TensorLike, options: NativeEagerLinearActivationIntoOptions): Float32Array;
  activationInto(output: Float32Array, input: TensorLike, options: NativeEagerActivationIntoOptions): Float32Array;
  activation_into(output: Float32Array, input: TensorLike, options: NativeEagerActivationIntoOptions): Float32Array;
  elementwiseInto(output: Float32Array, lhs: TensorLike, rhs: TensorLike | Float32Array | number | null | undefined, options: NativeEagerElementwiseIntoOptions): Float32Array;
  elementwise_into(output: Float32Array, lhs: TensorLike, rhs: TensorLike | Float32Array | number | null | undefined, options: NativeEagerElementwiseIntoOptions): Float32Array;
  reduceInto(output: Float32Array, input: TensorLike, options: NativeEagerReduceIntoOptions): Float32Array;
  reduce_into(output: Float32Array, input: TensorLike, options: NativeEagerReduceIntoOptions): Float32Array;
  dotInto(output: Float32Array, lhs: TensorLike, rhs: TensorLike): Float32Array;
  dot_into(output: Float32Array, lhs: TensorLike, rhs: TensorLike): Float32Array;
  conv2dInto(output: Float32Array, input: TensorLike, weights: TensorLike, options: NativeEagerConv2dIntoOptions): Float32Array;
  conv2d_into(output: Float32Array, input: TensorLike, weights: TensorLike, options: NativeEagerConv2dIntoOptions): Float32Array;
  pool2dInto(output: Float32Array, input: TensorLike, options: NativeEagerPool2dIntoOptions): Float32Array;
  pool2d_into(output: Float32Array, input: TensorLike, options: NativeEagerPool2dIntoOptions): Float32Array;
  matmulInto(output: Float32Array, lhs: TensorLike, rhs: TensorLike, options?: NativeEagerMatmulIntoOptions): Float32Array;
  matmul_into(output: Float32Array, lhs: TensorLike, rhs: TensorLike, options?: NativeEagerMatmulIntoOptions): Float32Array;
  bmmInto(output: Float32Array, lhs: TensorLike, rhs: TensorLike, options?: NativeEagerBmmIntoOptions): Float32Array;
  bmm_into(output: Float32Array, lhs: TensorLike, rhs: TensorLike, options?: NativeEagerBmmIntoOptions): Float32Array;
  softmaxInto(output: Float32Array, input: TensorLike, options?: NativeEagerSoftmaxIntoOptions): Float32Array;
  softmax_into(output: Float32Array, input: TensorLike, options?: NativeEagerSoftmaxIntoOptions): Float32Array;
  logSoftmaxInto(output: Float32Array, input: TensorLike, options?: NativeEagerSoftmaxIntoOptions): Float32Array;
  log_softmax_into(output: Float32Array, input: TensorLike, options?: NativeEagerSoftmaxIntoOptions): Float32Array;
}>;
export type NativeCoreEvidence = Readonly<{
  kind: "zgml.native-core";
  host: "node" | "bun";
  productApi: "typescript";
  nativeCore: "zig-c-abi";
  boundary: Readonly<{
    userApi: "typescript";
    tensorRuntime: "zig";
    ffi: "c-abi";
    moduleCompiler: FrontendManifest["moduleCompilerCore"];
    compileEvidence: FrontendManifest["nativeCompileEvidence"];
    hotPath: "program-session";
    training: "zig-ffi-kernels";
  }>;
  productSourceOfTruth: FrontendManifest["productSourceOfTruth"];
  nativeProductPolicy: FrontendManifest["nativeProductPolicy"];
  runtimePath: "JS/TS API -> Zig C ABI -> Program/Session kernels";
  abiVersion: number;
  featureFlags: bigint;
  domains: Readonly<{
    tensorStorage: boolean;
    eagerKernels: boolean;
    programSession: boolean;
    trainingStep: boolean;
    llmSession: boolean;
    modelSource: boolean;
    webgpuInterop: boolean;
  }>;
  eagerOps: Readonly<{
    linear: boolean;
    linearActivation: boolean;
    matmul: boolean;
    bmm: boolean;
    activation: boolean;
    elementwise: boolean;
    elementwiseBroadcast: boolean;
    reduce: boolean;
    dot: boolean;
    softmax: boolean;
    conv2d: boolean;
    pool2d: boolean;
  }>;
  unsupportedHotPathPolicy: FrontendManifest["unsupportedHotPathPolicy"];
  signature: string;
}>;
export type NativeCoreFunction = () => NativeCoreEvidence;
export declare function linspace<const S extends TensorShapeTuple>(shape: S, start: number, end: number, options?: TensorOptions): Tensor<S>;
export declare function linspace<const Steps extends number>(start: number, end: number, steps: Steps, options?: TensorOptions): Tensor<readonly [Steps]>;
export declare function linspace(start: number, end: number, steps: number, options?: TensorOptions): Tensor;
export declare function arange<const End extends number>(end: End, options?: TensorOptions): Tensor<ArangeShape<End>>;
export declare function arange<const Start extends number, const End extends number>(start: Start, end: End, options?: TensorOptions): Tensor<ArangeRangeShape<Start, End>>;
export declare function arange(start: number, end: number, step: number, options?: TensorOptions): Tensor;

export type NnParameter<Shape extends TensorShapeTuple = TensorShapeTuple> = {
  name: string;
  data: Float32Array;
  grad: Float32Array | null;
  layout: string;
  tensor: Tensor<Shape>;
  requiresGrad: boolean;
  requires_grad: boolean;
};
export type NnParameterOptions<S extends TensorShape = TensorShape> = Readonly<{
  shape?: S;
  layout?: string;
}>;
export type NnBuffer<Shape extends TensorShapeTuple = TensorShapeTuple> = {
  readonly kind: "buffer";
  name: string;
  data: Float32Array;
  shape: Shape;
  layout: string;
  persistent: boolean;
  tensor?: Tensor<Shape>;
};
export type NnBufferOptions<S extends TensorShape = TensorShape> = Readonly<{
  shape?: S;
  layout?: string;
  persistent?: boolean;
}>;
export type NnInitTarget = NnParameter | Tensor;
export type KaimingInitOptions = {
  a?: number;
  negativeSlope?: number;
  negative_slope?: number;
  mode?: "fan_in" | "fan_out" | "fanIn" | "fanOut";
  nonlinearity?: "linear" | "sigmoid" | "tanh" | "relu" | "leaky_relu";
};
export type KaimingUniformOptions = KaimingInitOptions;
export type KaimingNormalOptions = KaimingInitOptions;
export type NnInitNamespace = Readonly<{
  constant_<T extends NnInitTarget>(target: T, value: number): T;
  zeros_<T extends NnInitTarget>(target: T): T;
  ones_<T extends NnInitTarget>(target: T): T;
  uniform_<T extends NnInitTarget>(target: T, min?: number, max?: number): T;
  normal_<T extends NnInitTarget>(target: T, mean?: number, std?: number): T;
  xavierUniform_<T extends NnInitTarget>(target: T, gain?: number): T;
  xavier_uniform_<T extends NnInitTarget>(target: T, gain?: number): T;
  xavierNormal_<T extends NnInitTarget>(target: T, gain?: number): T;
  xavier_normal_<T extends NnInitTarget>(target: T, gain?: number): T;
  kaimingUniform_<T extends NnInitTarget>(target: T, options?: KaimingUniformOptions): T;
  kaiming_uniform_<T extends NnInitTarget>(target: T, options?: KaimingUniformOptions): T;
  kaimingNormal_<T extends NnInitTarget>(target: T, options?: KaimingNormalOptions): T;
  kaiming_normal_<T extends NnInitTarget>(target: T, options?: KaimingNormalOptions): T;
}>;
export type ModuleParameterInfo = Readonly<{
  name: string;
  index: number;
  scalarCount: number;
  shape: readonly number[];
  layout: string;
  requiresGrad: boolean;
  requires_grad: boolean;
}>;
export type ModuleTraversalEntry = Readonly<{
  name: string;
  module: NnModule;
}>;
export type ModuleParameterTraversalOptions = Readonly<{
  prefix?: string;
  recurse?: boolean;
}>;
export type ModuleBufferTraversalOptions = Readonly<{
  prefix?: string;
  recurse?: boolean;
}>;
export type ModuleStateDictOptions = Readonly<{
  prefix?: string;
}>;

export type LossReduction = "mean" | "sum";
export type LossReductionOptions = { reduction?: LossReduction };
export type HuberLossOptions = LossReductionOptions & { delta?: number };
export type SmoothL1LossOptions = LossReductionOptions & { beta?: number };
export type BCELossOptions = LossReductionOptions & { eps?: number };
export type ClassLossOptions = LossReductionOptions & { classes?: number; numClasses?: number };

export interface MSELoss {
  readonly kind: "mse-loss";
  readonly reduction: LossReduction;
  forward(pred: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  forward(pred: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  forward(pred: TensorData, target: TensorData, options?: LossReductionOptions): number;
  call(pred: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  call(pred: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  call(pred: TensorData, target: TensorData, options?: LossReductionOptions): number;
  __call__(pred: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  __call__(pred: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  __call__(pred: TensorData, target: TensorData, options?: LossReductionOptions): number;
}

export interface L1Loss {
  readonly kind: "l1-loss";
  readonly reduction: LossReduction;
  forward(pred: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  forward(pred: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  forward(pred: TensorData, target: TensorData, options?: LossReductionOptions): number;
  call(pred: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  call(pred: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  call(pred: TensorData, target: TensorData, options?: LossReductionOptions): number;
  __call__(pred: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  __call__(pred: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  __call__(pred: TensorData, target: TensorData, options?: LossReductionOptions): number;
}

export interface HuberLoss {
  readonly kind: "huber-loss";
  readonly reduction: LossReduction;
  readonly delta: number;
  forward(pred: Tensor, target: TensorLike, options?: HuberLossOptions): Tensor<readonly [1]>;
  forward(pred: TensorData, target: Tensor, options?: HuberLossOptions): Tensor<readonly [1]>;
  forward(pred: TensorData, target: TensorData, options?: HuberLossOptions): number;
  call(pred: Tensor, target: TensorLike, options?: HuberLossOptions): Tensor<readonly [1]>;
  call(pred: TensorData, target: Tensor, options?: HuberLossOptions): Tensor<readonly [1]>;
  call(pred: TensorData, target: TensorData, options?: HuberLossOptions): number;
  __call__(pred: Tensor, target: TensorLike, options?: HuberLossOptions): Tensor<readonly [1]>;
  __call__(pred: TensorData, target: Tensor, options?: HuberLossOptions): Tensor<readonly [1]>;
  __call__(pred: TensorData, target: TensorData, options?: HuberLossOptions): number;
}

export interface SmoothL1Loss {
  readonly kind: "smooth-l1-loss";
  readonly reduction: LossReduction;
  readonly beta: number;
  forward(pred: Tensor, target: TensorLike, options?: SmoothL1LossOptions): Tensor<readonly [1]>;
  forward(pred: TensorData, target: Tensor, options?: SmoothL1LossOptions): Tensor<readonly [1]>;
  forward(pred: TensorData, target: TensorData, options?: SmoothL1LossOptions): number;
  call(pred: Tensor, target: TensorLike, options?: SmoothL1LossOptions): Tensor<readonly [1]>;
  call(pred: TensorData, target: Tensor, options?: SmoothL1LossOptions): Tensor<readonly [1]>;
  call(pred: TensorData, target: TensorData, options?: SmoothL1LossOptions): number;
  __call__(pred: Tensor, target: TensorLike, options?: SmoothL1LossOptions): Tensor<readonly [1]>;
  __call__(pred: TensorData, target: Tensor, options?: SmoothL1LossOptions): Tensor<readonly [1]>;
  __call__(pred: TensorData, target: TensorData, options?: SmoothL1LossOptions): number;
}

export interface BCELoss {
  readonly kind: "bce-loss";
  readonly reduction: LossReduction;
  readonly eps: number;
  forward(pred: Tensor, target: TensorLike, options?: BCELossOptions): Tensor<readonly [1]>;
  forward(pred: TensorData, target: Tensor, options?: BCELossOptions): Tensor<readonly [1]>;
  forward(pred: TensorData, target: TensorData, options?: BCELossOptions): number;
  call(pred: Tensor, target: TensorLike, options?: BCELossOptions): Tensor<readonly [1]>;
  call(pred: TensorData, target: Tensor, options?: BCELossOptions): Tensor<readonly [1]>;
  call(pred: TensorData, target: TensorData, options?: BCELossOptions): number;
  __call__(pred: Tensor, target: TensorLike, options?: BCELossOptions): Tensor<readonly [1]>;
  __call__(pred: TensorData, target: Tensor, options?: BCELossOptions): Tensor<readonly [1]>;
  __call__(pred: TensorData, target: TensorData, options?: BCELossOptions): number;
}

export interface BCEWithLogitsLoss {
  readonly kind: "bce-with-logits-loss";
  readonly reduction: LossReduction;
  forward(logits: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  forward(logits: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  forward(logits: TensorData, target: TensorData, options?: LossReductionOptions): number;
  call(logits: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  call(logits: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  call(logits: TensorData, target: TensorData, options?: LossReductionOptions): number;
  __call__(logits: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  __call__(logits: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  __call__(logits: TensorData, target: TensorData, options?: LossReductionOptions): number;
}

export interface CrossEntropyLoss {
  readonly kind: "cross-entropy-loss";
  readonly classes: number | null;
  readonly numClasses: number | null;
  readonly reduction: LossReduction;
  forward(logits: Tensor, targets: IndexLike, options?: ClassLossOptions): Tensor<readonly [1]>;
  forward(logits: TensorData, targets: IndexLike, options?: ClassLossOptions): number;
  call(logits: Tensor, targets: IndexLike, options?: ClassLossOptions): Tensor<readonly [1]>;
  call(logits: TensorData, targets: IndexLike, options?: ClassLossOptions): number;
  __call__(logits: Tensor, targets: IndexLike, options?: ClassLossOptions): Tensor<readonly [1]>;
  __call__(logits: TensorData, targets: IndexLike, options?: ClassLossOptions): number;
}

export interface NLLLoss {
  readonly kind: "nll-loss";
  readonly classes: number | null;
  readonly numClasses: number | null;
  readonly reduction: LossReduction;
  forward(logProbabilities: Tensor, targets: IndexLike, options?: ClassLossOptions): Tensor<readonly [1]>;
  forward(logProbabilities: TensorData, targets: IndexLike, options?: ClassLossOptions): number;
  call(logProbabilities: Tensor, targets: IndexLike, options?: ClassLossOptions): Tensor<readonly [1]>;
  call(logProbabilities: TensorData, targets: IndexLike, options?: ClassLossOptions): number;
  __call__(logProbabilities: Tensor, targets: IndexLike, options?: ClassLossOptions): Tensor<readonly [1]>;
  __call__(logProbabilities: TensorData, targets: IndexLike, options?: ClassLossOptions): number;
}

export type ModuleStateEntry = {
  shape: readonly number[];
  layout?: string;
  data: TensorLike;
};
export type ModuleStateSnapshotEntry = Readonly<{
  signature: string;
  shape: readonly number[];
  layout: string;
  data: Float32Array;
}>;
export type ModuleStateSnapshot = Readonly<Record<string, ModuleStateSnapshotEntry>>;
export type ModuleStateValue = TensorLike | ModuleStateEntry;
export type ModuleStateDict = Record<string, ModuleStateValue> | Map<string, ModuleStateValue>;
export type LoadStateDictOptions = {
  strict?: boolean;
  prefix?: string;
  validateOnly?: boolean;
};
export type ZeroGradOptions = {
  setToNone?: boolean;
  set_to_none?: boolean;
};
export type OptimizerStateKind = "sgd" | "adam" | "adamw" | "rmsprop" | "adagrad";
export type OptimizerStateEntry = ModuleStateEntry & {
  name: string;
};
export type OptimizerStateSnapshotEntry = ModuleStateSnapshotEntry & Readonly<{
  name: string;
}>;
export type OptimizerStateSnapshot<Kind extends OptimizerStateKind = OptimizerStateKind> = Readonly<{
  kind: Kind;
  signature: string;
  step: number;
  paramCount: number;
  entries: readonly OptimizerStateSnapshotEntry[];
}>;
export type OptimizerStateDict<Kind extends OptimizerStateKind = OptimizerStateKind> = {
  kind?: Kind;
  step?: number;
  paramCount?: number;
  entries?: readonly OptimizerStateEntry[];
};
export type CheckpointTensorEntry = Readonly<{
  signature: string;
  dtype: "f32";
  shape: readonly number[];
  layout: string;
  data: readonly number[];
}>;
export type CheckpointOptimizerEntry = CheckpointTensorEntry & Readonly<{
  name: string;
}>;
export type CheckpointModuleState = Readonly<Record<string, CheckpointTensorEntry>>;
export type CheckpointOptimizerState = Readonly<{
  kind: OptimizerStateKind;
  step: number;
  paramCount: number;
  entries: readonly CheckpointOptimizerEntry[];
}>;
export type CheckpointSchedulerState = Readonly<{
  kind: "step-lr" | "exponential-lr" | "cosine-annealing-lr" | "reduce-lr-on-plateau";
  signature: string;
  step: number;
  baseLr: number;
  base_lr: number;
  lastLr: number;
  last_lr: number;
  gamma: number;
  stepSize?: number;
  step_size?: number;
  tMax?: number;
  t_max?: number;
  etaMin?: number;
  eta_min?: number;
  mode?: "min" | "max";
  factor?: number;
  patience?: number;
  threshold?: number;
  thresholdMode?: "rel" | "abs";
  threshold_mode?: "rel" | "abs";
  cooldown?: number;
  cooldownCounter?: number;
  cooldown_counter?: number;
  minLr?: number;
  min_lr?: number;
  eps?: number;
  best?: number;
  badEpochs?: number;
  bad_epochs?: number;
  optimizerKind: OptimizerStateKind | null;
}>;
export type ZgmlCheckpoint = Readonly<{
  format: "zgml.checkpoint";
  version: 1;
  metadata?: unknown;
  model?: CheckpointModuleState;
  optimizer?: CheckpointOptimizerState;
  scheduler?: CheckpointSchedulerState;
}>;
export type CheckpointTensorInspection = Readonly<{
  signature: string;
  name: string;
  index: number;
  scalarCount: number;
  shape: readonly number[];
  layout: string | null;
}>;
export type CheckpointInspection = Readonly<{
  signature: string;
  format: "zgml.checkpoint";
  version: 1;
  hasMetadata: boolean;
  hasModel: boolean;
  hasOptimizer: boolean;
  hasScheduler: boolean;
  modelParameterCount: number;
  modelScalarCount: number;
  modelParameterNames: readonly string[];
  modelParameters: readonly CheckpointTensorInspection[];
  optimizerKind: OptimizerStateKind | null;
  optimizerStep: number | null;
  optimizerParamCount: number;
  optimizerEntryCount: number;
  optimizerScalarCount: number;
  optimizerEntryNames: readonly string[];
  optimizerEntries: readonly CheckpointTensorInspection[];
  schedulerKind: "step-lr" | "exponential-lr" | "cosine-annealing-lr" | "reduce-lr-on-plateau" | null;
  schedulerStep: number | null;
  schedulerBaseLr: number | null;
  schedulerLastLr: number | null;
  schedulerGamma: number | null;
  schedulerStepSize: number | null;
  schedulerTMax: number | null;
  schedulerT_max: number | null;
  schedulerEtaMin: number | null;
  schedulerEta_min: number | null;
  schedulerOptimizerKind: OptimizerStateKind | null;
}>;
export type CheckpointCreateOptions<OptimizerKind extends OptimizerStateKind = OptimizerStateKind> = {
  model?: OptimizerTarget;
  optimizer?: Optimizer<OptimizerKind>;
  scheduler?: LRScheduler<LRSchedulerStateKind, NoInfer<OptimizerKind>>;
  metadata?: unknown;
  prefix?: string;
};
export type CheckpointRestoreOptions<OptimizerKind extends OptimizerStateKind = OptimizerStateKind> = {
  model?: OptimizerTarget;
  optimizer?: Optimizer<OptimizerKind>;
  scheduler?: LRScheduler<LRSchedulerStateKind, NoInfer<OptimizerKind>>;
  strict?: boolean;
  prefix?: string;
};
export type CheckpointNamespace = Readonly<{
  create<const OptimizerKind extends OptimizerStateKind>(options?: CheckpointCreateOptions<OptimizerKind>): ZgmlCheckpoint;
  inspect(snapshot: ZgmlCheckpoint): CheckpointInspection;
  modelParameterInfo(snapshot: ZgmlCheckpoint, nameOrIndex: string | number): CheckpointTensorInspection | null;
  optimizerEntryInfo(snapshot: ZgmlCheckpoint, nameOrIndex: string | number): CheckpointTensorInspection | null;
  toJSON(snapshot: ZgmlCheckpoint): ZgmlCheckpoint;
  fromJSON(value: unknown): ZgmlCheckpoint;
  stringify(snapshot: ZgmlCheckpoint, space?: string | number): string;
  serialize(snapshot: ZgmlCheckpoint, space?: string | number): string;
  parse(text: string): ZgmlCheckpoint;
  deserialize(text: string): ZgmlCheckpoint;
  restore<const OptimizerKind extends OptimizerStateKind>(snapshot: ZgmlCheckpoint, targets?: CheckpointRestoreOptions<OptimizerKind>): CheckpointRestoreOptions<OptimizerKind>;
  load<const OptimizerKind extends OptimizerStateKind>(snapshot: ZgmlCheckpoint, targets?: CheckpointRestoreOptions<OptimizerKind>): CheckpointRestoreOptions<OptimizerKind>;
  moduleStateDict(target: OptimizerTarget, prefix?: string): CheckpointModuleState;
  module_state_dict(target: OptimizerTarget, prefix?: string): CheckpointModuleState;
  optimizerStateDict(optimizer: Optimizer): CheckpointOptimizerState;
  optimizer_state_dict(optimizer: Optimizer): CheckpointOptimizerState;
  schedulerStateDict(scheduler: LRScheduler): CheckpointSchedulerState;
  scheduler_state_dict(scheduler: LRScheduler): CheckpointSchedulerState;
}>;

export type ModuleActivationKind =
  | "gelu"
  | "relu"
  | "silu"
  | "sigmoid"
  | "tanh"
  | "exp"
  | "log"
  | "neg"
  | "recip"
  | "abs"
  | "sqrt"
  | "square"
  | "step"
  | "sgn";
export type ModuleReductionKind = "sum" | "mean" | "prod" | "max" | "min" | "argmax" | "argmin";
export type ModuleShapeOpKind =
  | "identity"
  | "diagonal"
  | "reshape"
  | "view"
  | "flatten"
  | "squeeze"
  | "unsqueeze"
  | "transpose"
  | "permute"
  | "broadcastTo"
  | "expand"
  | "repeat"
  | "tile"
  | "narrow"
  | "select"
  | "slice";
export type ModuleTraceOpKind =
  | "linear"
  | "matmul"
  | "add"
  | "mul"
  | "affine"
  | "embedding"
  | "conv2d"
  | "avgPool2d"
  | "maxPool2d"
  | "activation"
  | "softmax"
  | "logSoftmax"
  | ModuleReductionKind
  | "layerNorm"
  | "rmsNorm"
  | "batchNorm1d"
  | "dropout"
  | ModuleShapeOpKind
  | "unknown";
export type ModuleTensorProgramIrOpKind =
  | "linear"
  | "matmul"
  | "add"
  | "mul"
  | "affine"
  | "embedding"
  | "conv2d"
  | "avgPool2d"
  | "maxPool2d"
  | "activation"
  | "softmax"
  | "logSoftmax"
  | ModuleReductionKind
  | "layerNorm"
  | "rmsNorm"
  | "batchNorm1d"
  | "dropout"
  | ModuleShapeOpKind
  | "unknown";
export type ModuleKernelFusedOpKind = "identity" | "reshape" | "view" | "flatten" | "squeeze" | "unsqueeze" | "dropout";
export type ModuleKernelPlanOpKind =
  | "linear"
  | "matmul"
  | "add"
  | "mul"
  | "affine"
  | "embedding"
  | "avgPool2d"
  | "maxPool2d"
  | "activation"
  | "softmax"
  | "logSoftmax"
  | ModuleReductionKind
  | "layerNorm"
  | "rmsNorm"
  | "batchNorm1d"
  | "dropout"
  | ModuleShapeOpKind
  | "shape-chain"
  | "activation-chain";
export type ModuleKernelName =
  | "linear"
  | "matmul"
  | "add"
  | "mul"
  | "activation-chain"
  | ModuleActivationKind
  | "softmax"
  | "log-softmax"
  | ModuleReductionKind
  | "reshape"
  | "broadcast"
  | "narrow"
  | "select"
  | "slice"
  | "transpose"
  | "layer-norm"
  | "rms-norm"
  | "affine"
  | "embedding"
  | "avg-pool2d"
  | "max-pool2d";
export type ModuleCompileDiagnosticStage = "trace" | "ir" | "kernelizer" | "support";
export type ModuleShapeDiagnosticFields = Readonly<{
  opIndex?: number;
  op?: ModuleTraceOpKind | ModuleTensorProgramIrOpKind;
  path?: string;
  inputShape?: readonly number[];
  outputShape?: readonly number[];
}>;
export type ModuleMissingInputShapeDiagnostic = Readonly<{
  stage: "ir";
  code: "missing-input-shape";
  message: string;
}>;
export type ModuleShapeMismatchDiagnostic = Readonly<{
  stage: "trace";
  code: "shape-mismatch";
  message: string;
  inputShape?: readonly number[];
}>;
export type ModuleIrShapeDiagnostic = ModuleShapeDiagnosticFields & Readonly<{
  stage: "ir";
  code: "shape-unknown" | "unsupported-rank";
  message: string;
}>;
export type ModuleKernelizerDiagnostic = ModuleShapeDiagnosticFields & Readonly<{
  stage: "kernelizer";
  code: "unsupported-op" | "unsupported-view" | "unsupported-graph";
  message: string;
}>;
export type ModuleSupportDiagnostic = Readonly<{
  stage: "support";
  code: "layer-missing-compiler" | "layer-unsupported";
  message: string;
  layerIndex: number;
  moduleKind: string;
}>;
export type ModuleCompileDiagnostic =
  | ModuleMissingInputShapeDiagnostic
  | ModuleShapeMismatchDiagnostic
  | ModuleIrShapeDiagnostic
  | ModuleKernelizerDiagnostic
  | ModuleSupportDiagnostic;
export type CompileDiagnostic = import("./runtime/compile_diagnostics.js").CompileDiagnostic;

export type ModuleCompilerSignatures = Readonly<{
  ir?: string;
  kernelPlan?: string;
  memoryLayout?: string;
  parameterLayout?: string;
  bufferLayout?: string;
}>;
export type ModuleCompleteCompilerSignatures = Readonly<{
  ir: string;
  kernelPlan: string;
  memoryLayout: string;
  parameterLayout: string;
  bufferLayout: string;
}>;
export type ModuleCompileSupport<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = Readonly<{
  supported: boolean;
  reason: string | null;
  composable?: boolean;
  nativePath?: "tiny-linear" | "device-program";
  nativeCore?: "zig-tiny-linear" | "zig-module-program";
  modelKind?: "tiny-linear" | "tiny-mlp" | "module";
  layerCount?: number;
  inputShape?: InputShape;
  outputShape?: OutputShape;
  inputLen?: number;
  outputLen?: number;
  weightsLen?: number;
  biasLen?: number;
  diagnostics?: readonly ModuleCompileDiagnostic[];
  irSignature?: string;
  kernelPlanSignature?: string;
  memoryLayoutSignature?: string;
  parameterLayoutSignature?: string;
  bufferLayoutSignature?: string;
  compilerSignatures?: ModuleCompilerSignatures;
  trace?: ModuleProgramTrace;
  ir?: ModuleTensorProgramIr;
  kernelPlan?: ModuleKernelPlan;
}>;
export type ModuleCompileExplanation<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = Readonly<{
  kind: "zgml.nn.compile-explanation";
  supported: boolean;
  reason: string | null;
  nativePath: ModuleCompileSupport["nativePath"] | null;
  modelKind: ModuleCompileSupport["modelKind"] | null;
  signature: string;
  inputShape: InputShape | null;
  outputShape: OutputShape | null;
  parameterNames: readonly string[];
  parameterInfos: readonly ModuleParameterInfo[];
  parameterCount: number | null;
  parameterScalarCount: number | null;
  trace: ModuleProgramTrace | null;
  ir: ModuleTensorProgramIr | null;
  kernelPlan: ModuleKernelPlan | null;
  compilerSignatures: ModuleCompilerSignatures;
  bufferLayout: ModuleKernelBufferLayout | null;
  memoryLayout: ModuleKernelMemoryLayout | null;
  shapeConstraints: ModuleKernelShapeConstraints | null;
  parameterLayout: ModuleKernelParameterLayout | null;
  diagnostics: readonly ModuleCompileDiagnostic[];
}>;
export type CompileManifest = typeof import("./compile.js").compileManifest;
export type RawSequentialLayerListCompileDiagnostic = import("./compile.js").RawSequentialLayerListCompileDiagnostic;
export type FrontendManifest = typeof import("./frontend_manifest.js").frontendManifest;
export type CompileAnalysis = Readonly<{
  kind: "zgml.compile.analysis";
  signature: string;
  trace: ModuleProgramTrace;
  artifacts: Readonly<{
    ir: ModuleTensorProgramIr | null;
    kernelPlan: ModuleKernelPlan | null;
    diagnostic: CompileDiagnostic | null;
  }>;
}>;
export type CompileNamespace = typeof import("./compile.js");
export type InspectionNamespace = typeof import("./inspection.js");
export type ProgramNamespace = typeof import("./program.js");
export type SessionNamespace = typeof import("./session.js");
export type StepParamsNamespace = typeof import("./step_params.js");
export type NativeApiContractSignatureParts = import("./runtime/native_api_contract.js").NativeApiContractSignatureParts;
export type NativeApiContractExportKey<Owner extends object> = Extract<keyof Owner, string>;
export type NativeApiContractNamespace = typeof import("./runtime/native_api_contract.js");
export type ModuleTraceParameter = Readonly<{
  name: string;
  shape: readonly number[];
  layout: string | null;
  scalarCount: number;
}>;
export type ModuleTraceOptions = Readonly<{
  inputShape?: readonly number[];
}>;
export type ModuleTraceOpBase<Op extends ModuleTraceOpKind> = Readonly<{
  index: number;
  path: string;
  op: Op;
  inputShape?: readonly number[];
  outputShape?: readonly number[];
  parameters: readonly ModuleTraceParameter[];
}>;
export type ModuleLinearTraceOp = ModuleTraceOpBase<"linear"> & Readonly<{
  inFeatures: number;
  outFeatures: number;
  bias: boolean;
}>;
export type ModuleMatmulTraceOp = ModuleTraceOpBase<"matmul"> & Readonly<{
  inFeatures: number;
  outFeatures: number;
}>;
export type ModuleAddTraceOp = ModuleTraceOpBase<"add"> & Readonly<{
  features: number;
}>;
export type ModuleMulTraceOp = ModuleTraceOpBase<"mul"> & Readonly<{
  features: number;
}>;
export type ModuleAffineTraceOp = ModuleTraceOpBase<"affine"> & Readonly<{
  features: number;
}>;
export type ModuleEmbeddingTraceOp = ModuleTraceOpBase<"embedding"> & Readonly<{
  numEmbeddings: number;
  embeddingDim: number;
}>;
export type ModuleConv2dTraceOp = ModuleTraceOpBase<"conv2d"> & Readonly<{
  inChannels: number;
  outChannels: number;
  kernelSize: readonly [number, number];
  stride: readonly [number, number];
  padding: readonly [number, number];
  dilation: readonly [number, number];
  groups: number;
}>;
export type ModuleMaxPool2dTraceOp = ModuleTraceOpBase<"maxPool2d"> & Readonly<{
  kernelSize: readonly [number, number];
  stride: readonly [number, number];
  padding: readonly [number, number];
  dilation: readonly [number, number];
  ceilMode: boolean;
}>;
export type ModuleAvgPool2dTraceOp = ModuleTraceOpBase<"avgPool2d"> & Readonly<{
  kernelSize: readonly [number, number];
  stride: readonly [number, number];
  padding: readonly [number, number];
  ceilMode: boolean;
  countIncludePad: boolean;
}>;
export type ModuleActivationTraceOp = ModuleTraceOpBase<"activation"> & Readonly<{
  activation: ModuleActivationKind;
}>;
export type ModuleSoftmaxTraceOp = ModuleTraceOpBase<"softmax" | "logSoftmax"> & Readonly<{
  dim: number;
}>;
export type ModuleReductionTraceOp = ModuleTraceOpBase<ModuleReductionKind> & Readonly<{
  dim: number;
}>;
export type ModuleDropoutTraceOp = ModuleTraceOpBase<"dropout"> & Readonly<{
  p: number;
  training: boolean;
}>;
export type ModuleIdentityTraceOp = ModuleTraceOpBase<"identity">;
export type ModuleDiagonalTraceOp = ModuleTraceOpBase<"diagonal">;
export type ModuleShapeTargetTraceOp = ModuleTraceOpBase<"reshape" | "view" | "broadcastTo" | "expand" | "repeat" | "tile"> & Readonly<{
  shape: readonly number[];
}>;
export type ModuleFlattenTraceOp = ModuleTraceOpBase<"flatten"> & Readonly<{
  startDim: number;
  endDim: number;
}>;
export type ModuleSqueezeTraceOp = ModuleTraceOpBase<"squeeze"> & (
  | Readonly<{ squeezeAll: true }>
  | Readonly<{ squeezeAll: false; dim: number }>
);
export type ModuleUnsqueezeTraceOp = ModuleTraceOpBase<"unsqueeze"> & Readonly<{
  dim: number;
}>;
export type ModuleTransposeTraceOp = ModuleTraceOpBase<"transpose"> & Readonly<{
  dim0: number;
  dim1: number;
}>;
export type ModulePermuteTraceOp = ModuleTraceOpBase<"permute"> & Readonly<{
  dims: readonly number[];
}>;
export type ModuleNarrowTraceOp = ModuleTraceOpBase<"narrow"> & Readonly<{
  dim: number;
  start: number;
  length: number;
}>;
export type ModuleSelectTraceOp = ModuleTraceOpBase<"select"> & Readonly<{
  dim: number;
  selectIndex: number;
}>;
export type ModuleSliceTraceOp = ModuleTraceOpBase<"slice"> & Readonly<{
  dim: number;
  start: number;
  end: number | null;
  step: number;
}>;
export type ModuleFeatureNormTraceOp = ModuleTraceOpBase<"layerNorm" | "rmsNorm" | "batchNorm1d"> & Readonly<{
  features: number;
  eps: number;
  affine: boolean;
  bias: boolean;
  momentum?: number;
  trackRunningStats?: boolean;
  training?: boolean;
}>;
export type ModuleUnknownTraceOp = ModuleTraceOpBase<"unknown"> & Readonly<{
  moduleKind: string;
}>;
export type ModuleTraceOp =
  | ModuleLinearTraceOp
  | ModuleMatmulTraceOp
  | ModuleAddTraceOp
  | ModuleMulTraceOp
  | ModuleAffineTraceOp
  | ModuleEmbeddingTraceOp
  | ModuleConv2dTraceOp
  | ModuleAvgPool2dTraceOp
  | ModuleMaxPool2dTraceOp
  | ModuleActivationTraceOp
  | ModuleSoftmaxTraceOp
  | ModuleReductionTraceOp
  | ModuleDropoutTraceOp
  | ModuleIdentityTraceOp
  | ModuleDiagonalTraceOp
  | ModuleShapeTargetTraceOp
  | ModuleFlattenTraceOp
  | ModuleSqueezeTraceOp
  | ModuleUnsqueezeTraceOp
  | ModuleTransposeTraceOp
  | ModulePermuteTraceOp
  | ModuleNarrowTraceOp
  | ModuleSelectTraceOp
  | ModuleSliceTraceOp
  | ModuleFeatureNormTraceOp
  | ModuleUnknownTraceOp;
export type ModuleTensorProgramIrValueRole = "input" | "parameter" | "op-output";
export type ModuleTensorProgramIrValue = Readonly<{
  id: number;
  role: ModuleTensorProgramIrValueRole;
  shape: readonly number[];
  dtype: "f32";
  rank: number;
  scalarCount: number;
  scalarBytes: 4;
  byteLength: number;
  storageLayout: "row-major";
  strides: readonly number[];
  storageOffset: number;
  dense: boolean;
  name?: string;
  binding?: "weights" | "bias";
  layout?: string | null;
  opIndex?: number;
  op?: ModuleTensorProgramIrOpKind;
  path?: string;
}>;
export type ModuleTensorProgramIrLinearAttrs = Readonly<{
  inFeatures: number;
  outFeatures: number;
  bias: boolean;
}>;
export type ModuleTensorProgramIrAddAttrs = Readonly<{
  features: number;
  hasBias: boolean;
}>;
export type ModuleTensorProgramIrMulAttrs = Readonly<{
  features: number;
  hasWeight: boolean;
}>;
export type ModuleTensorProgramIrAffineAttrs = Readonly<{
  features: number;
  hasWeight: boolean;
  hasBias: boolean;
}>;
export type ModuleTensorProgramIrEmbeddingAttrs = Readonly<{
  numEmbeddings: number;
  embeddingDim: number;
  hasWeight: boolean;
}>;
export type ModuleTensorProgramIrConv2dAttrs = Readonly<{
  inChannels: number;
  outChannels: number;
  kernelSize: number | readonly [number, number];
  stride: number | readonly [number, number];
  padding: number | readonly [number, number];
  dilation: number | readonly [number, number];
  groups: number;
  hasWeight: boolean;
  hasBias: boolean;
}>;
export type ModuleTensorProgramIrAvgPool2dAttrs = Readonly<{
  kernelSize: number | readonly [number, number];
  stride: number | readonly [number, number];
  padding: number | readonly [number, number];
  ceilMode: boolean;
  countIncludePad: boolean;
}>;
export type ModuleTensorProgramIrMaxPool2dAttrs = Readonly<{
  kernelSize: number | readonly [number, number];
  stride: number | readonly [number, number];
  padding: number | readonly [number, number];
  dilation: number | readonly [number, number];
  ceilMode: boolean;
}>;
export type ModuleTensorProgramIrActivationAttrs = Readonly<{
  activation: ModuleActivationKind;
}>;
export type ModuleTensorProgramIrDimAttrs = Readonly<{
  dim: number;
}>;
export type ModuleTensorProgramIrDropoutAttrs = Readonly<{
  p: number;
  training: boolean;
}>;
export type ModuleTensorProgramIrShapeTargetAttrs = Readonly<{
  shape: readonly number[];
}>;
export type ModuleTensorProgramIrFlattenAttrs = Readonly<{
  startDim: number;
  endDim: number;
}>;
export type ModuleTensorProgramIrSqueezeAttrs = Readonly<{
  squeezeAll: boolean;
  dim?: number;
}>;
export type ModuleTensorProgramIrUnsqueezeAttrs = Readonly<{
  dim: number;
}>;
export type ModuleTensorProgramIrNarrowAttrs = Readonly<{
  dim: number;
  start: number;
  length: number;
}>;
export type ModuleTensorProgramIrSelectAttrs = Readonly<{
  dim: number;
  selectIndex: number;
}>;
export type ModuleTensorProgramIrSliceAttrs = Readonly<{
  dim: number;
  start: number;
  end: number | null;
  step: number;
}>;
export type ModuleTensorProgramIrTransposeAttrs = Readonly<{
  dim0: number;
  dim1: number;
}>;
export type ModuleTensorProgramIrPermuteAttrs = Readonly<{
  dims: readonly number[];
}>;
export type ModuleTensorProgramIrFeatureNormAttrs = Readonly<{
  features: number;
  eps: number;
  affine: boolean;
  bias: boolean;
  hasWeight: boolean;
  hasBias?: boolean;
  momentum?: number;
  trackRunningStats?: boolean;
  training?: boolean;
}>;
export type ModuleTensorProgramIrUnknownAttrs = Readonly<{
  moduleKind: string;
}>;
export type ModuleTensorProgramIrNoAttrs = Readonly<Record<string, never>>;
export type ModuleTensorProgramIrOpBase<Op extends ModuleTensorProgramIrOpKind, Attrs> = Readonly<{
  index: number;
  path: string;
  op: Op;
  inputShape: readonly number[];
  outputShape: readonly number[];
  inputLen: number;
  outputLen: number;
  scalarType: "f32";
  scalarBytes: 4;
  inputValueIds: readonly number[];
  outputValueId: number;
  parameterValueIds: readonly number[];
  parameterScalarCount: number;
  attrs: Attrs;
}>;
export type ModuleTensorProgramIrOp =
  | ModuleTensorProgramIrOpBase<"linear", ModuleTensorProgramIrLinearAttrs>
  | ModuleTensorProgramIrOpBase<"matmul", ModuleTensorProgramIrLinearAttrs>
  | ModuleTensorProgramIrOpBase<"add", ModuleTensorProgramIrAddAttrs>
  | ModuleTensorProgramIrOpBase<"mul", ModuleTensorProgramIrMulAttrs>
  | ModuleTensorProgramIrOpBase<"affine", ModuleTensorProgramIrAffineAttrs>
  | ModuleTensorProgramIrOpBase<"embedding", ModuleTensorProgramIrEmbeddingAttrs>
  | ModuleTensorProgramIrOpBase<"conv2d", ModuleTensorProgramIrConv2dAttrs>
  | ModuleTensorProgramIrOpBase<"avgPool2d", ModuleTensorProgramIrAvgPool2dAttrs>
  | ModuleTensorProgramIrOpBase<"maxPool2d", ModuleTensorProgramIrMaxPool2dAttrs>
  | ModuleTensorProgramIrOpBase<"activation", ModuleTensorProgramIrActivationAttrs>
  | ModuleTensorProgramIrOpBase<"softmax" | "logSoftmax", ModuleTensorProgramIrDimAttrs>
  | ModuleTensorProgramIrOpBase<ModuleReductionKind, ModuleTensorProgramIrDimAttrs>
  | ModuleTensorProgramIrOpBase<"dropout", ModuleTensorProgramIrDropoutAttrs>
  | ModuleTensorProgramIrOpBase<"identity", ModuleTensorProgramIrNoAttrs>
  | ModuleTensorProgramIrOpBase<"diagonal", ModuleTensorProgramIrNoAttrs>
  | ModuleTensorProgramIrOpBase<"reshape" | "view" | "broadcastTo" | "expand" | "repeat" | "tile", ModuleTensorProgramIrShapeTargetAttrs>
  | ModuleTensorProgramIrOpBase<"flatten", ModuleTensorProgramIrFlattenAttrs>
  | ModuleTensorProgramIrOpBase<"squeeze", ModuleTensorProgramIrSqueezeAttrs>
  | ModuleTensorProgramIrOpBase<"unsqueeze", ModuleTensorProgramIrUnsqueezeAttrs>
  | ModuleTensorProgramIrOpBase<"narrow", ModuleTensorProgramIrNarrowAttrs>
  | ModuleTensorProgramIrOpBase<"select", ModuleTensorProgramIrSelectAttrs>
  | ModuleTensorProgramIrOpBase<"slice", ModuleTensorProgramIrSliceAttrs>
  | ModuleTensorProgramIrOpBase<"transpose", ModuleTensorProgramIrTransposeAttrs>
  | ModuleTensorProgramIrOpBase<"permute", ModuleTensorProgramIrPermuteAttrs>
  | ModuleTensorProgramIrOpBase<"layerNorm" | "rmsNorm" | "batchNorm1d", ModuleTensorProgramIrFeatureNormAttrs>
  | ModuleTensorProgramIrOpBase<"unknown", ModuleTensorProgramIrUnknownAttrs>;
export type ModuleTensorProgramIr = Readonly<{
  kind: "tensor-program-ir";
  version: 1;
  signature: string;
  inputShape: readonly number[];
  outputShape: readonly number[];
  inputLen: number;
  outputLen: number;
  inputValueId: number;
  outputValueId: number;
  valueCount: number;
  opCount: number;
  parameterCount: number;
  parameterScalarCount: number;
  values: readonly ModuleTensorProgramIrValue[];
  ops: readonly ModuleTensorProgramIrOp[];
}>;
export type ModuleKernelPlanOpBase<Op extends ModuleKernelPlanOpKind, Kernel extends ModuleKernelName> = Readonly<{
  index: number;
  path: string;
  op: Op;
  kernel: Kernel;
  inputShape: readonly number[];
  outputShape: readonly number[];
  inputLen: number;
  outputLen: number;
  inputValueIds: readonly number[];
  outputValueId: number;
  parameterScalarCount: number;
  nativeDispatchCount: number;
  nativeDescriptorCount: number;
  nativeKernels: readonly ModuleKernelName[];
  nativeDescriptorSignatures: readonly string[];
  fusedValueEdges?: readonly ModuleKernelPlanFusedValueEdge[];
}>;
export type ModuleKernelPlanFusedValueEdge = Readonly<{
  opIndex: number;
  op: ModuleTensorProgramIrOpKind;
  path: string;
  inputValueIds: readonly number[];
  outputValueId: number;
}>;
export type ModuleKernelPlanLinearOp = ModuleKernelPlanOpBase<"linear", "linear"> & Readonly<{
  fusedOpCount?: number;
  fusedOps?: readonly ("linear" | "activation")[];
  fusedIndices?: readonly number[];
}>;
export type ModuleKernelPlanMatmulOp = ModuleKernelPlanOpBase<"matmul", "matmul">;
export type ModuleKernelPlanAddOp = ModuleKernelPlanOpBase<"add", "add"> & Readonly<{
  fusedOpCount?: number;
  fusedOps?: readonly ("add" | "activation")[];
  fusedIndices?: readonly number[];
}>;
export type ModuleKernelPlanMulOp = ModuleKernelPlanOpBase<"mul", "mul"> & Readonly<{
  fusedOpCount?: number;
  fusedOps?: readonly ("mul" | "activation")[];
  fusedIndices?: readonly number[];
}>;
export type ModuleKernelPlanAffineOp = ModuleKernelPlanOpBase<"affine", "affine"> & Readonly<{
  fusedOpCount?: number;
  fusedOps?: readonly ("affine" | "mul" | "add" | "activation")[];
  fusedIndices?: readonly number[];
  fusedValueEdges?: readonly ModuleKernelPlanFusedValueEdge[];
}>;
export type ModuleKernelPlanEmbeddingOp = ModuleKernelPlanOpBase<"embedding", "embedding">;
export type ModuleKernelPlanAvgPool2dOp = ModuleKernelPlanOpBase<"avgPool2d", "avg-pool2d">;
export type ModuleKernelPlanMaxPool2dOp = ModuleKernelPlanOpBase<"maxPool2d", "max-pool2d">;
export type ModuleKernelPlanActivationOp = ModuleKernelPlanOpBase<"activation", ModuleActivationKind>;
export type ModuleKernelPlanActivationChainOp = ModuleKernelPlanOpBase<"activation-chain", "activation-chain"> & Readonly<{
  fusedOpCount: number;
  fusedOps: readonly "activation"[];
  fusedIndices: readonly number[];
  fusedValueEdges: readonly ModuleKernelPlanFusedValueEdge[];
}>;
export type ModuleKernelPlanSoftmaxOp = ModuleKernelPlanOpBase<"softmax", "softmax">;
export type ModuleKernelPlanLogSoftmaxOp = ModuleKernelPlanOpBase<"logSoftmax", "log-softmax">;
export type ModuleKernelPlanReductionOp = ModuleKernelPlanOpBase<ModuleReductionKind, ModuleReductionKind>;
export type ModuleKernelPlanReshapeOp = ModuleKernelPlanOpBase<ModuleKernelFusedOpKind, "reshape">;
export type ModuleKernelPlanBroadcastOp = ModuleKernelPlanOpBase<"broadcastTo" | "expand", "broadcast">;
export type ModuleKernelPlanNarrowOp = ModuleKernelPlanOpBase<"narrow", "narrow">;
export type ModuleKernelPlanSelectOp = ModuleKernelPlanOpBase<"select", "select">;
export type ModuleKernelPlanSliceOp = ModuleKernelPlanOpBase<"slice", "slice">;
export type ModuleKernelPlanTransposeOp = ModuleKernelPlanOpBase<"transpose", "transpose">;
export type ModuleKernelPlanFeatureNormOp =
  | ModuleKernelPlanOpBase<"layerNorm", "layer-norm">
  | ModuleKernelPlanOpBase<"rmsNorm", "rms-norm">
  | ModuleKernelPlanOpBase<"batchNorm1d", "affine">;
export type ModuleKernelPlanShapeChainOp = ModuleKernelPlanOpBase<"shape-chain", "reshape"> & Readonly<{
  fusedOpCount: number;
  fusedOps: readonly ModuleKernelFusedOpKind[];
  fusedIndices: readonly number[];
  fusedValueEdges: readonly ModuleKernelPlanFusedValueEdge[];
}>;
export type ModuleKernelPlanElidedOp = ModuleKernelPlanReshapeOp | ModuleKernelPlanShapeChainOp;
export type ModuleKernelPlanOp =
  | ModuleKernelPlanLinearOp
  | ModuleKernelPlanMatmulOp
  | ModuleKernelPlanAddOp
  | ModuleKernelPlanMulOp
  | ModuleKernelPlanAffineOp
  | ModuleKernelPlanEmbeddingOp
  | ModuleKernelPlanAvgPool2dOp
  | ModuleKernelPlanMaxPool2dOp
  | ModuleKernelPlanActivationOp
  | ModuleKernelPlanActivationChainOp
  | ModuleKernelPlanSoftmaxOp
  | ModuleKernelPlanLogSoftmaxOp
  | ModuleKernelPlanReductionOp
  | ModuleKernelPlanReshapeOp
  | ModuleKernelPlanBroadcastOp
  | ModuleKernelPlanNarrowOp
  | ModuleKernelPlanSelectOp
  | ModuleKernelPlanSliceOp
  | ModuleKernelPlanTransposeOp
  | ModuleKernelPlanFeatureNormOp
  | ModuleKernelPlanShapeChainOp;
export type ModuleKernelShapeConstraints = Readonly<{
  specialization: "exact";
  inputRank: number;
  outputRank: number;
  inputShape: readonly number[];
  outputShape: readonly number[];
  inputLen: number;
  outputLen: number;
}>;
export type ModuleKernelMemoryLayoutValue = Readonly<{
  id: number;
  role: ModuleTensorProgramIrValueRole;
  storageClass: "step-input" | "step-output" | "persistent" | "scratch";
  buffer: "input" | "output" | "weights" | "bias" | "scratch";
  producerOpIndex: number | null;
  consumerOpIndices: readonly number[];
  firstUseOpIndex: number | null;
  lastUseOpIndex: number | null;
  liveStartOpIndex: number | null;
  liveEndOpIndex: number | null;
  scalarType: "f32";
  scalarBytes: 4;
  shape: readonly number[];
  rank: number;
  scalarCount: number;
  scalarOffset: number;
  bufferScalarOffset: number;
  byteOffset: number;
  bufferByteOffset: number;
  byteLength: number;
  storageLayout: "row-major";
  strides: readonly number[];
  storageOffset: number;
  dense: boolean;
  name?: string;
  binding?: "weights" | "bias";
  layout?: string | null;
  opIndex?: number;
  op?: ModuleTensorProgramIrOpKind;
  path?: string;
}>;
export type ModuleKernelMemoryLayout = Readonly<{
  signature: string;
  scalarType: "f32";
  scalarBytes: 4;
  valueCount: number;
  totalScalarCount: number;
  totalByteLength: number;
  scratchScalarCount: number;
  scratchByteLength: number;
  values: readonly ModuleKernelMemoryLayoutValue[];
}>;
export type ModuleKernelBufferRole = "step-input" | "step-output" | "persistent";
export type ModuleKernelBufferLayoutSlot = Readonly<{
  name: "input" | "output" | "weights" | "bias";
  role: ModuleKernelBufferRole;
  scalarType: "f32";
  scalarBytes: 4;
  elementOffset: number;
  elementCount: number;
  byteOffset: number;
  byteLength: number;
}>;
export type ModuleKernelBufferLayout = Readonly<{
  kind: "zgml.program.buffer-layout";
  signature: string;
  scalarType: "f32";
  scalarBytes: 4;
  slots: readonly ModuleKernelBufferLayoutSlot[];
  input: ModuleKernelBufferLayoutSlot;
  output: ModuleKernelBufferLayoutSlot;
  weights: ModuleKernelBufferLayoutSlot;
  bias: ModuleKernelBufferLayoutSlot;
}>;
export type ProgramBufferLayoutSlot = Readonly<{
  name: "input" | "output" | "weights" | "bias";
  role: ModuleKernelBufferRole;
  scalarType: "f32";
  scalarBytes: number;
  elementOffset: number;
  elementCount: number;
  byteOffset: number;
  byteLength: number;
}>;
export type ProgramBufferLayout = Readonly<{
  kind: "zgml.program.buffer-layout";
  signature: string;
  scalarType: "f32";
  scalarBytes: number;
  slots: readonly ProgramBufferLayoutSlot[];
  input: ProgramBufferLayoutSlot;
  output: ProgramBufferLayoutSlot;
  weights: ProgramBufferLayoutSlot;
  bias: ProgramBufferLayoutSlot;
}>;
export type ModuleKernelParameterBinding = "weights" | "bias";
export type ModuleKernelParameterTraceOpKind = "linear" | "embedding" | "layerNorm" | "rmsNorm";
export type ModuleKernelParameterLayoutEntry = Readonly<{
  name: string;
  binding: ModuleKernelParameterBinding;
  offset: number;
  scalarCount: number;
  shape: readonly number[];
  layout: string | null;
  opIndex: number;
  op: ModuleKernelParameterTraceOpKind;
  path: string;
}>;
export type ModuleKernelParameterLayout = Readonly<{
  signature: string;
  weightsLen: number;
  biasLen: number;
  parameters: readonly ModuleKernelParameterLayoutEntry[];
}>;
export type ModuleKernelPlan = Readonly<{
  kind: "native-module-kernel-plan";
  version: 1;
  nativePath: "device-program";
  signature: string;
  inputShape: readonly number[];
  outputShape: readonly number[];
  inputLen: number;
  outputLen: number;
  shapeConstraints: ModuleKernelShapeConstraints;
  memoryLayout: ModuleKernelMemoryLayout;
  bufferLayout: ModuleKernelBufferLayout;
  weightsLen: number;
  biasLen: number;
  parameterLayout: ModuleKernelParameterLayout;
  opCount: number;
  dispatchCount: number;
  descriptorCount: number;
  elidedOpCount: number;
  ops: readonly ModuleKernelPlanOp[];
  elidedOps: readonly ModuleKernelPlanElidedOp[];
}>;
export type ProgramCompileEvidence = Readonly<{
  signature: string;
  kind: "tiny-linear";
  nativePath: "tiny-linear";
  nativeCore: "zig-tiny-linear";
  modelKind: "tiny-linear";
  layerCount: number;
  inputLen: number;
  outputLen: number;
  weightsLen: number;
  biasLen: number;
  irSignature: string;
  kernelPlanSignature: string;
  memoryLayoutSignature: string;
  parameterLayoutSignature: string;
  bufferLayoutSignature: string;
  compilerSignatures: ModuleCompleteCompilerSignatures;
  trace: ModuleProgramTrace;
  ir: ModuleTensorProgramIr;
  kernelPlan: ModuleKernelPlan;
}> | Readonly<{
  signature: string;
  kind: "module";
  nativePath: "device-program";
  nativeCore: "zig-module-program";
  modelKind: "module";
  layerCount: number;
  inputLen?: number;
  outputLen?: number;
  weightsLen?: number;
  biasLen?: number;
  nativeRequirements?: ProgramRequirements;
  nativeRequirementsSignature?: string;
  nativeRequirementsSource?: "zig-module-program";
  nativeCompilerAuthority?: "zig-module-program";
  nativeProgramInspection?: ProgramInspection;
  nativeProgramInspectionSignature?: string;
  nativeProgramInspectionSource?: "zig-program-inspection";
  nativeExecutionSupported?: boolean;
  nativeCommandStencilHash?: bigint;
  irSignature: string;
  kernelPlanSignature: string;
  memoryLayoutSignature: string;
  parameterLayoutSignature: string;
  bufferLayoutSignature: string;
  compilerSignatures: ModuleCompleteCompilerSignatures;
  trace: ModuleProgramTrace;
  ir: ModuleTensorProgramIr;
  kernelPlan: ModuleKernelPlan;
}>;
export type ProgramModuleCompatibilityLengthField = "inputLen" | "outputLen" | "weightsLen" | "biasLen";
export type ProgramModuleCompatibilityDiagnosticCode =
  | "missing-program-evidence"
  | "module-missing-compiler"
  | "module-unsupported"
  | "module-analysis-failed"
  | "native-path-mismatch"
  | "model-kind-mismatch"
  | "layer-count-mismatch"
  | `${ProgramModuleCompatibilityLengthField}-mismatch`
  | "missing-module-kernel-plan"
  | "input-shape-mismatch"
  | "output-shape-mismatch"
  | "ir-mismatch"
  | "kernel-plan-mismatch"
  | "memory-layout-mismatch"
  | "parameter-layout-mismatch"
  | "buffer-layout-mismatch"
  | "program-kind-mismatch"
  | "trace-mismatch";
export type ProgramModuleCompatibilitySimpleDiagnosticCode =
  | "missing-program-evidence"
  | "module-missing-compiler"
  | "module-unsupported"
  | "module-analysis-failed"
  | "missing-module-kernel-plan"
  | "program-kind-mismatch"
  | "trace-mismatch";
export type ProgramModuleCompatibilitySimpleDiagnostic = Readonly<{
  code: ProgramModuleCompatibilitySimpleDiagnosticCode;
  message: string;
}>;
export type ProgramModuleCompatibilitySignatureKind = "ir" | "kernel-plan" | "memory-layout" | "parameter-layout" | "buffer-layout";
export type ProgramModuleCompatibilitySignatureDiagnostic = Readonly<{
  code: "ir-mismatch" | "kernel-plan-mismatch" | "memory-layout-mismatch" | "parameter-layout-mismatch" | "buffer-layout-mismatch";
  message: string;
  signatureKind: ProgramModuleCompatibilitySignatureKind;
  programSignature: string;
  moduleSignature: string;
}>;
export type ProgramModuleCompatibilityNativePathDiagnostic = Readonly<{
  code: "native-path-mismatch";
  message: string;
  programNativePath: ProgramCompileEvidence["nativePath"];
  moduleNativePath: ModuleCompileSupport["nativePath"];
}>;
export type ProgramModuleCompatibilityModelKindDiagnostic = Readonly<{
  code: "model-kind-mismatch";
  message: string;
  programModelKind: ProgramCompileEvidence["modelKind"] | null;
  moduleModelKind: ModuleCompileSupport["modelKind"] | null;
}>;
export type ProgramModuleCompatibilityLayerCountDiagnostic = Readonly<{
  code: "layer-count-mismatch";
  message: string;
  programLayerCount: number;
  moduleLayerCount: number;
}>;
export type ProgramModuleCompatibilityLengthDiagnostic = {
  [Field in ProgramModuleCompatibilityLengthField]: Readonly<{
    code: `${Field}-mismatch`;
    message: string;
    field: Field;
    programValue: number;
    moduleValue: number;
  }>;
}[ProgramModuleCompatibilityLengthField];
export type ProgramModuleCompatibilityShapeDiagnostic = Readonly<{
  code: "input-shape-mismatch" | "output-shape-mismatch";
  message: string;
  programShape: readonly number[];
  moduleShape: readonly number[];
}>;
export type ProgramModuleCompatibilityDiagnostic =
  | ProgramModuleCompatibilitySimpleDiagnostic
  | ProgramModuleCompatibilitySignatureDiagnostic
  | ProgramModuleCompatibilityNativePathDiagnostic
  | ProgramModuleCompatibilityModelKindDiagnostic
  | ProgramModuleCompatibilityLayerCountDiagnostic
  | ProgramModuleCompatibilityLengthDiagnostic
  | ProgramModuleCompatibilityShapeDiagnostic;
export type ProgramModuleCompatibility = Readonly<{
  kind: "zgml.program.module-compatibility";
  signature: string;
  compatible: boolean;
  reason: string | null;
  programKind: ProgramCompileEvidence["kind"] | null;
  moduleKind: ModuleCompileSupport["modelKind"] | null;
  diagnostics: readonly ProgramModuleCompatibilityDiagnostic[];
}>;
export type ModuleProgramTrace = Readonly<{
  kind: "sequential";
  normalized: true;
  layerCount: number;
  opCount: number;
  parameterCount: number;
  parameterScalarCount: number;
  shapeKnown: boolean;
  inputShape?: readonly number[];
  outputShape?: readonly number[] | null;
  ops: readonly ModuleTraceOp[];
}>;

export type NnLinearConfig = {
  weight?: TensorLike;
  weights?: TensorLike;
  bias?: false | true | TensorLike;
  biasValues?: TensorLike;
};

export type NnEmbeddingConfig = {
  weight?: TensorLike;
  weights?: TensorLike;
};

export type NnConv2dConfig = {
  stride?: number | readonly [number, number];
  padding?: number | readonly [number, number];
  dilation?: number | readonly [number, number];
  groups?: number;
  weight?: TensorLike;
  weights?: TensorLike;
  bias?: false | true | TensorLike;
  biasValues?: TensorLike;
};

export type NnMaxPool2dConfig = {
  stride?: number | readonly [number, number];
  padding?: number | readonly [number, number];
  dilation?: number | readonly [number, number];
  ceilMode?: boolean;
  ceil_mode?: boolean;
};
export type NnAvgPool2dConfig = {
  stride?: number | readonly [number, number];
  padding?: number | readonly [number, number];
  ceilMode?: boolean;
  ceil_mode?: boolean;
  countIncludePad?: boolean;
  count_include_pad?: boolean;
};

export type NnNormConfig = {
  eps?: number;
  momentum?: number;
  trackRunningStats?: boolean;
  track_running_stats?: boolean;
  runningMean?: TensorLike;
  running_mean?: TensorLike;
  runningVar?: TensorLike;
  running_var?: TensorLike;
  numBatchesTracked?: number;
  num_batches_tracked?: number;
  affine?: boolean;
  weight?: TensorLike;
  weights?: TensorLike;
  bias?: false | true | TensorLike;
  biasValues?: TensorLike;
};

export type NnDropoutConfig = {
  rng?: () => number;
  training?: boolean;
};

export type LinearForwardShape<InputShape extends TensorShapeTuple, OutFeatures extends number> =
  InputShape extends readonly [number]
    ? readonly [OutFeatures]
    : InputShape extends readonly [infer Batch extends number, number]
      ? readonly [Batch, OutFeatures]
      : TensorShapeTuple;

export interface NnModule {
  readonly training: boolean;
  call(input: TensorLike): Tensor;
  __call__(input: TensorLike): Tensor;
  forward(input: TensorLike): Tensor;
  parameters(prefix?: string): NnParameter[];
  parameters(options: ModuleParameterTraversalOptions): NnParameter[];
  parameters(prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  namedParameters(prefix?: string): NnParameter[];
  namedParameters(options: ModuleParameterTraversalOptions): NnParameter[];
  namedParameters(prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  named_parameters(prefix?: string): NnParameter[];
  named_parameters(options: ModuleParameterTraversalOptions): NnParameter[];
  named_parameters(prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  getParameter(name: string): NnParameter | null;
  get_parameter(name: string): NnParameter | null;
  buffers(prefix?: string): NnBuffer[];
  buffers(options: ModuleBufferTraversalOptions): NnBuffer[];
  buffers(prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  namedBuffers(prefix?: string): NnBuffer[];
  namedBuffers(options: ModuleBufferTraversalOptions): NnBuffer[];
  namedBuffers(prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  named_buffers(prefix?: string): NnBuffer[];
  named_buffers(options: ModuleBufferTraversalOptions): NnBuffer[];
  named_buffers(prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  getBuffer(name: string): NnBuffer | null;
  get_buffer(name: string): NnBuffer | null;
  parameterNames(prefix?: string): readonly string[];
  parameterInfos(prefix?: string): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number, prefix?: string): ModuleParameterInfo | null;
  children(): readonly NnModule[];
  modules(): readonly NnModule[];
  namedChildren(prefix?: string): readonly ModuleTraversalEntry[];
  named_children(prefix?: string): readonly ModuleTraversalEntry[];
  namedModules(prefix?: string): readonly ModuleTraversalEntry[];
  named_modules(prefix?: string): readonly ModuleTraversalEntry[];
  getSubmodule(name: string): NnModule | null;
  get_submodule(name: string): NnModule | null;
  apply(callback: (module: NnModule, entry: ModuleTraversalEntry) => void): this;
  zeroGrad(options?: ZeroGradOptions): void;
  zero_grad(options?: ZeroGradOptions): void;
  requiresGrad_(requiresGrad?: boolean): this;
  requires_grad_(requiresGrad?: boolean): this;
  train(mode?: boolean): this;
  eval(): this;
  to(target?: TensorToTarget | TensorToOptions, options?: TensorToOptions): this;
  cpu(options?: TensorToOptions): this;
  float(options?: TensorToOptions): this;
  float32(options?: TensorToOptions): this;
  stateDict(prefix?: string): ModuleStateSnapshot;
  stateDict(options: ModuleStateDictOptions): ModuleStateSnapshot;
  state_dict(prefix?: string): ModuleStateSnapshot;
  state_dict(options: ModuleStateDictOptions): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  load_state_dict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  trace(options?: ModuleTraceOptions): ModuleProgramTrace;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  compile_support(options?: CompileOptions): ModuleCompileSupport;
  requireCompileSupport(options?: CompileOptions): ModuleCompileSupport;
  require_compile_support(options?: CompileOptions): ModuleCompileSupport;
  compilerSignatures(options?: CompileOptions): ModuleCompilerSignatures | null;
  compiler_signatures(options?: CompileOptions): ModuleCompilerSignatures | null;
  tensorProgramIr(options?: CompileOptions): ModuleTensorProgramIr | null;
  tensor_program_ir(options?: CompileOptions): ModuleTensorProgramIr | null;
  kernelPlan(options?: CompileOptions): ModuleKernelPlan | null;
  kernel_plan(options?: CompileOptions): ModuleKernelPlan | null;
  bufferLayout(options?: CompileOptions): ModuleKernelBufferLayout | null;
  buffer_layout(options?: CompileOptions): ModuleKernelBufferLayout | null;
  memoryLayout(options?: CompileOptions): ModuleKernelMemoryLayout | null;
  memory_layout(options?: CompileOptions): ModuleKernelMemoryLayout | null;
  inputShape(options?: CompileOptions): readonly number[] | null;
  input_shape(options?: CompileOptions): readonly number[] | null;
  outputShape(options?: CompileOptions): readonly number[] | null;
  output_shape(options?: CompileOptions): readonly number[] | null;
  shapeConstraints(options?: CompileOptions): ModuleKernelShapeConstraints | null;
  shape_constraints(options?: CompileOptions): ModuleKernelShapeConstraints | null;
  parameterLayout(options?: CompileOptions): ModuleKernelParameterLayout | null;
  parameter_layout(options?: CompileOptions): ModuleKernelParameterLayout | null;
  explain(options?: CompileOptions): ModuleCompileExplanation;
  preflight(options?: CompileOptions): ModuleCompileExplanation;
  compileExplanation(options?: CompileOptions): ModuleCompileExplanation;
  compile_explanation(options?: CompileOptions): ModuleCompileExplanation;
  compilePlan(options?: CompileOptions): ModuleCompileExplanation;
  compile_plan(options?: CompileOptions): ModuleCompileExplanation;
  requireCompilePlan(options?: CompileOptions): ModuleCompileExplanation;
  require_compile_plan(options?: CompileOptions): ModuleCompileExplanation;
  assertCompilePlan(options?: CompileOptions): ModuleCompileExplanation;
  assert_compile_plan(options?: CompileOptions): ModuleCompileExplanation;
  canCompile(options?: CompileOptions): boolean;
  can_compile(options?: CompileOptions): boolean;
  native<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, TensorShapeTuple>;
  native(options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  inference<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, TensorShapeTuple>;
  inference(options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  forInference<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, TensorShapeTuple>;
  forInference(options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  for_inference<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, TensorShapeTuple>;
  for_inference(options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  compileInference<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, TensorShapeTuple>;
  compileInference(options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  compile_inference<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, TensorShapeTuple>;
  compile_inference(options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  forTraining(optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  for_training(optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  explainTraining(batches: Iterable<unknown>, options: TrainFitOptions & {
    optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void };
    loss?: unknown;
    criterion?: unknown;
  }): NativeTrainingExplanation;
  explain_training: NnModule["explainTraining"];
  explainNativeTraining: NnModule["explainTraining"];
  explain_native_training: NnModule["explainTraining"];
  fit(batches: Iterable<unknown>, options: TrainFitOptions & {
    optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void };
    loss?: unknown;
    criterion?: unknown;
  }): TrainFitEvidence;
  fitNative(batches: Iterable<unknown>, options: TrainFitOptions & {
    optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void };
    loss?: unknown;
    criterion?: unknown;
    requireNative?: true;
    require_native?: true;
  }): TrainFitEvidence;
  fit_native: NnModule["fitNative"];
  evaluate(batches: Iterable<unknown>, criterion: unknown, options?: TrainEvaluateOptions): TrainEvaluateEvidence;
  evalModule(batches: Iterable<unknown>, criterion: unknown, options?: TrainEvaluateOptions): TrainEvaluateEvidence;
  eval_module(batches: Iterable<unknown>, criterion: unknown, options?: TrainEvaluateOptions): TrainEvaluateEvidence;
  predict(batches: Iterable<unknown>, options?: TrainPredictOptions): TrainPredictEvidence;
}

export type NnModuleConfig<
  InputShape extends TensorShapeTuple = TensorShapeTuple,
  OutputShape extends TensorShapeTuple = TensorShapeTuple,
> = Readonly<{
  kind?: string;
  forward(input: Tensor<InputShape>): Tensor<OutputShape>;
  parameters?: readonly NnParameter[] | ((prefix?: string) => NnParameter[]);
  children?: readonly NnModule[] | (() => readonly NnModule[]);
  graph?: NnCompilableModule | (() => NnCompilableModule);
}>;

export interface CustomModule<
  InputShape extends TensorShapeTuple = TensorShapeTuple,
  OutputShape extends TensorShapeTuple = TensorShapeTuple,
> extends NnModule {
  readonly kind: string;
  readonly customModule: true;
  call(input: Tensor<InputShape>): Tensor<OutputShape>;
  call(input: TensorLike): Tensor;
  __call__(input: Tensor<InputShape>): Tensor<OutputShape>;
  __call__(input: TensorLike): Tensor;
  forward(input: Tensor<InputShape>): Tensor<OutputShape>;
  forward(input: TensorLike): Tensor;
  compile<const S extends InputShape>(options: CompileOptionsWithInputShape<S>): Program<S, OutputShape>;
  compile(options?: CompileOptions): Program<InputShape, OutputShape>;
  compileSupport<const S extends InputShape>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, OutputShape>;
  compileSupport(options?: CompileOptions): ModuleCompileSupport<InputShape, OutputShape>;
  requireCompileSupport<const S extends InputShape>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, OutputShape>;
  requireCompileSupport(options?: CompileOptions): ModuleCompileSupport<InputShape, OutputShape>;
  compilePlan<const S extends InputShape>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, OutputShape>;
  compilePlan(options?: CompileOptions): ModuleCompileExplanation<InputShape, OutputShape>;
  requireCompilePlan<const S extends InputShape>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, OutputShape>;
  requireCompilePlan(options?: CompileOptions): ModuleCompileExplanation<InputShape, OutputShape>;
  bindParameters<const S extends InputShape>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, OutputShape>;
  bindParameters(options?: CompileOptions): ModuleBindings<InputShape, OutputShape>;
}

export interface ModuleBase<
  InputShape extends TensorShapeTuple = TensorShapeTuple,
  OutputShape extends TensorShapeTuple = TensorShapeTuple,
> extends CustomModule<InputShape, OutputShape> {
  graph?: NnCompilableModule | (() => NnCompilableModule);
  addModule(name: string, module: NnModule): this;
  add_module(name: string, module: NnModule): this;
  registerModule(name: string, module: NnModule): this;
  register_module(name: string, module: NnModule): this;
  registerParameter<const S extends TensorShapeTuple>(name: string, parameter: NnParameter<S>): this;
  registerParameter(name: string, parameter: NnParameter): this;
  register_parameter<const S extends TensorShapeTuple>(name: string, parameter: NnParameter<S>): this;
  register_parameter(name: string, parameter: NnParameter): this;
  registerBuffer<const S extends TensorShapeTuple>(name: string, buffer: NnBuffer<S>): this;
  registerBuffer<const S extends TensorShape>(name: string, data: TensorLike, shape: S): this;
  registerBuffer(name: string, data: TensorLike): this;
  register_buffer<const S extends TensorShapeTuple>(name: string, buffer: NnBuffer<S>): this;
  register_buffer<const S extends TensorShape>(name: string, data: TensorLike, shape: S): this;
  register_buffer(name: string, data: TensorLike): this;
}

export interface ParameterList<Params extends readonly NnParameter[] = readonly NnParameter[]> {
  readonly kind: "parameterList";
  readonly params: Params;
  readonly length: Params["length"];
  len(): number;
  __len__(): number;
  size(): number;
  at(index: number): Params[number] | undefined;
  get(index: number): Params[number] | undefined;
  __getitem__(index: number): Params[number] | undefined;
  __setitem__(index: number, parameter: NnParameter): this;
  __delitem__(index: number): this;
  pop(index?: number): Params[number] | undefined;
  clear(): this;
  push(parameter: NnParameter): this;
  append(parameter: NnParameter): this;
  insert(index: number, parameter: NnParameter): this;
  extend(parameters: Iterable<NnParameter>): this;
  [Symbol.iterator](): IterableIterator<Params[number]>;
  parameters(prefix?: string): NnParameter[];
  parameters(options: ModuleParameterTraversalOptions): NnParameter[];
  parameters(prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  namedParameters(prefix?: string): NnParameter[];
  namedParameters(options: ModuleParameterTraversalOptions): NnParameter[];
  namedParameters(prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  named_parameters(prefix?: string): NnParameter[];
  named_parameters(options: ModuleParameterTraversalOptions): NnParameter[];
  named_parameters(prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  getParameter(name: string): NnParameter | null;
  get_parameter(name: string): NnParameter | null;
  buffers(prefix?: string): NnBuffer[];
  buffers(options: ModuleBufferTraversalOptions): NnBuffer[];
  buffers(prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  namedBuffers(prefix?: string): NnBuffer[];
  namedBuffers(options: ModuleBufferTraversalOptions): NnBuffer[];
  namedBuffers(prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  named_buffers(prefix?: string): NnBuffer[];
  named_buffers(options: ModuleBufferTraversalOptions): NnBuffer[];
  named_buffers(prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  getBuffer(name: string): NnBuffer | null;
  get_buffer(name: string): NnBuffer | null;
  parameterNames(prefix?: string): readonly string[];
  parameterInfos(prefix?: string): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number, prefix?: string): ModuleParameterInfo | null;
  zeroGrad(options?: ZeroGradOptions): void;
  zero_grad(options?: ZeroGradOptions): void;
  requiresGrad_(requiresGrad?: boolean): this;
  requires_grad_(requiresGrad?: boolean): this;
  stateDict(prefix?: string): ModuleStateSnapshot;
  stateDict(options: ModuleStateDictOptions): ModuleStateSnapshot;
  state_dict(prefix?: string): ModuleStateSnapshot;
  state_dict(options: ModuleStateDictOptions): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  load_state_dict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
}

export interface ParameterDict<Params extends Readonly<Record<string, NnParameter>> = Readonly<Record<string, NnParameter>>> {
  readonly kind: "parameterDict";
  readonly params: Params;
  readonly length: number;
  len(): number;
  __len__(): number;
  size(): number;
  get<const Name extends keyof Params & string>(name: Name): Params[Name] | undefined;
  get(name: string): NnParameter | undefined;
  __getitem__<const Name extends keyof Params & string>(name: Name): Params[Name] | undefined;
  __getitem__(name: string): NnParameter | undefined;
  __setitem__<const Name extends keyof Params & string>(name: Name, parameter: Params[Name]): this;
  __setitem__(name: string, parameter: NnParameter): this;
  set(name: string, parameter: NnParameter): this;
  update(params: Readonly<Record<string, NnParameter>>): this;
  has(name: string): boolean;
  __contains__(name: string): boolean;
  __delitem__<const Name extends keyof Params & string>(name: Name): this;
  __delitem__(name: string): this;
  pop<const Name extends keyof Params & string>(name: Name): Params[Name] | undefined;
  pop(name: string): NnParameter | undefined;
  clear(): this;
  keys(): readonly string[];
  values(): readonly NnParameter[];
  entries(): readonly (readonly [string, NnParameter])[];
  items(): readonly (readonly [string, NnParameter])[];
  [Symbol.iterator](): IterableIterator<readonly [string, NnParameter]>;
  parameters(prefix?: string): NnParameter[];
  parameters(options: ModuleParameterTraversalOptions): NnParameter[];
  parameters(prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  namedParameters(prefix?: string): NnParameter[];
  namedParameters(options: ModuleParameterTraversalOptions): NnParameter[];
  namedParameters(prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  named_parameters(prefix?: string): NnParameter[];
  named_parameters(options: ModuleParameterTraversalOptions): NnParameter[];
  named_parameters(prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  getParameter(name: string): NnParameter | null;
  get_parameter(name: string): NnParameter | null;
  buffers(prefix?: string): NnBuffer[];
  buffers(options: ModuleBufferTraversalOptions): NnBuffer[];
  buffers(prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  namedBuffers(prefix?: string): NnBuffer[];
  namedBuffers(options: ModuleBufferTraversalOptions): NnBuffer[];
  namedBuffers(prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  named_buffers(prefix?: string): NnBuffer[];
  named_buffers(options: ModuleBufferTraversalOptions): NnBuffer[];
  named_buffers(prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  getBuffer(name: string): NnBuffer | null;
  get_buffer(name: string): NnBuffer | null;
  parameterNames(prefix?: string): readonly string[];
  parameterInfos(prefix?: string): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number, prefix?: string): ModuleParameterInfo | null;
  zeroGrad(options?: ZeroGradOptions): void;
  zero_grad(options?: ZeroGradOptions): void;
  requiresGrad_(requiresGrad?: boolean): this;
  requires_grad_(requiresGrad?: boolean): this;
  stateDict(prefix?: string): ModuleStateSnapshot;
  stateDict(options: ModuleStateDictOptions): ModuleStateSnapshot;
  state_dict(prefix?: string): ModuleStateSnapshot;
  state_dict(options: ModuleStateDictOptions): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  load_state_dict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
}

export interface ModuleList<Layers extends readonly NnModule[] = readonly NnModule[]> {
  readonly kind: "moduleList";
  readonly layers: Layers;
  readonly length: Layers["length"];
  len(): number;
  __len__(): number;
  size(): number;
  at(index: number): Layers[number] | undefined;
  get(index: number): Layers[number] | undefined;
  __getitem__(index: number): Layers[number] | undefined;
  __setitem__(index: number, module: NnModule): this;
  __delitem__(index: number): this;
  pop(index?: number): Layers[number] | undefined;
  clear(): this;
  push(module: NnModule): this;
  append(module: NnModule): this;
  insert(index: number, module: NnModule): this;
  extend(modules: Iterable<NnModule>): this;
  [Symbol.iterator](): IterableIterator<Layers[number]>;
  parameters(prefix?: string): NnParameter[];
  parameters(options: ModuleParameterTraversalOptions): NnParameter[];
  parameters(prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  namedParameters(prefix?: string): NnParameter[];
  namedParameters(options: ModuleParameterTraversalOptions): NnParameter[];
  namedParameters(prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  named_parameters(prefix?: string): NnParameter[];
  named_parameters(options: ModuleParameterTraversalOptions): NnParameter[];
  named_parameters(prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  getParameter(name: string): NnParameter | null;
  get_parameter(name: string): NnParameter | null;
  buffers(prefix?: string): NnBuffer[];
  buffers(options: ModuleBufferTraversalOptions): NnBuffer[];
  buffers(prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  namedBuffers(prefix?: string): NnBuffer[];
  namedBuffers(options: ModuleBufferTraversalOptions): NnBuffer[];
  namedBuffers(prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  named_buffers(prefix?: string): NnBuffer[];
  named_buffers(options: ModuleBufferTraversalOptions): NnBuffer[];
  named_buffers(prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  getBuffer(name: string): NnBuffer | null;
  get_buffer(name: string): NnBuffer | null;
  parameterNames(prefix?: string): readonly string[];
  parameterInfos(prefix?: string): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number, prefix?: string): ModuleParameterInfo | null;
  children(): readonly NnModule[];
  modules(): readonly NnModule[];
  namedChildren(prefix?: string): readonly ModuleTraversalEntry[];
  named_children(prefix?: string): readonly ModuleTraversalEntry[];
  namedModules(prefix?: string): readonly ModuleTraversalEntry[];
  named_modules(prefix?: string): readonly ModuleTraversalEntry[];
  getSubmodule(name: string): NnModule | null;
  get_submodule(name: string): NnModule | null;
  apply(callback: (module: NnModule, entry: ModuleTraversalEntry) => void): this;
  zeroGrad(options?: ZeroGradOptions): void;
  zero_grad(options?: ZeroGradOptions): void;
  requiresGrad_(requiresGrad?: boolean): this;
  requires_grad_(requiresGrad?: boolean): this;
  train(mode?: boolean): this;
  eval(): this;
  stateDict(prefix?: string): ModuleStateSnapshot;
  stateDict(options: ModuleStateDictOptions): ModuleStateSnapshot;
  state_dict(prefix?: string): ModuleStateSnapshot;
  state_dict(options: ModuleStateDictOptions): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  load_state_dict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
}

export interface ModuleDict<Layers extends Readonly<Record<string, NnModule>> = Readonly<Record<string, NnModule>>> {
  readonly kind: "moduleDict";
  readonly layers: Layers;
  readonly length: number;
  len(): number;
  __len__(): number;
  size(): number;
  get<const Name extends keyof Layers & string>(name: Name): Layers[Name] | undefined;
  get(name: string): NnModule | undefined;
  __getitem__<const Name extends keyof Layers & string>(name: Name): Layers[Name] | undefined;
  __getitem__(name: string): NnModule | undefined;
  __setitem__<const Name extends keyof Layers & string>(name: Name, module: Layers[Name]): this;
  __setitem__(name: string, module: NnModule): this;
  set(name: string, module: NnModule): this;
  update(layers: Readonly<Record<string, NnModule>>): this;
  has(name: string): boolean;
  __contains__(name: string): boolean;
  __delitem__<const Name extends keyof Layers & string>(name: Name): this;
  __delitem__(name: string): this;
  pop<const Name extends keyof Layers & string>(name: Name): Layers[Name] | undefined;
  pop(name: string): NnModule | undefined;
  clear(): this;
  keys(): readonly string[];
  values(): readonly NnModule[];
  entries(): readonly (readonly [string, NnModule])[];
  items(): readonly (readonly [string, NnModule])[];
  [Symbol.iterator](): IterableIterator<readonly [string, NnModule]>;
  parameters(prefix?: string): NnParameter[];
  parameters(options: ModuleParameterTraversalOptions): NnParameter[];
  parameters(prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  namedParameters(prefix?: string): NnParameter[];
  namedParameters(options: ModuleParameterTraversalOptions): NnParameter[];
  namedParameters(prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  named_parameters(prefix?: string): NnParameter[];
  named_parameters(options: ModuleParameterTraversalOptions): NnParameter[];
  named_parameters(prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  getParameter(name: string): NnParameter | null;
  get_parameter(name: string): NnParameter | null;
  buffers(prefix?: string): NnBuffer[];
  buffers(options: ModuleBufferTraversalOptions): NnBuffer[];
  buffers(prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  namedBuffers(prefix?: string): NnBuffer[];
  namedBuffers(options: ModuleBufferTraversalOptions): NnBuffer[];
  namedBuffers(prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  named_buffers(prefix?: string): NnBuffer[];
  named_buffers(options: ModuleBufferTraversalOptions): NnBuffer[];
  named_buffers(prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  getBuffer(name: string): NnBuffer | null;
  get_buffer(name: string): NnBuffer | null;
  parameterNames(prefix?: string): readonly string[];
  parameterInfos(prefix?: string): readonly ModuleParameterInfo[];
  parameterInfo(nameOrIndex: string | number, prefix?: string): ModuleParameterInfo | null;
  children(): readonly NnModule[];
  modules(): readonly NnModule[];
  namedChildren(prefix?: string): readonly ModuleTraversalEntry[];
  named_children(prefix?: string): readonly ModuleTraversalEntry[];
  namedModules(prefix?: string): readonly ModuleTraversalEntry[];
  named_modules(prefix?: string): readonly ModuleTraversalEntry[];
  getSubmodule(name: string): NnModule | null;
  get_submodule(name: string): NnModule | null;
  apply(callback: (module: NnModule, entry: ModuleTraversalEntry) => void): this;
  zeroGrad(options?: ZeroGradOptions): void;
  zero_grad(options?: ZeroGradOptions): void;
  requiresGrad_(requiresGrad?: boolean): this;
  requires_grad_(requiresGrad?: boolean): this;
  train(mode?: boolean): this;
  eval(): this;
  stateDict(prefix?: string): ModuleStateSnapshot;
  stateDict(options: ModuleStateDictOptions): ModuleStateSnapshot;
  state_dict(prefix?: string): ModuleStateSnapshot;
  state_dict(options: ModuleStateDictOptions): ModuleStateSnapshot;
  loadStateDict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
  load_state_dict(source: ModuleStateDict, options?: LoadStateDictOptions): this;
}

export interface LinearModule<InFeatures extends number = number, OutFeatures extends number = number> extends NnModule {
  readonly kind: "linear";
  readonly inFeatures: InFeatures;
  readonly outFeatures: OutFeatures;
  readonly weight: Float32Array;
  readonly bias: Float32Array | null;
  call<const Input extends TensorLike>(input: Input): Tensor<LinearForwardShape<TensorLikeShape<Input>, OutFeatures>>;
  call<const InputShape extends readonly [InFeatures] | readonly [number, InFeatures]>(input: Tensor<InputShape>): Tensor<LinearForwardShape<InputShape, OutFeatures>>;
  call(input: TensorLike): Tensor;
  __call__<const Input extends TensorLike>(input: Input): Tensor<LinearForwardShape<TensorLikeShape<Input>, OutFeatures>>;
  __call__<const InputShape extends readonly [InFeatures] | readonly [number, InFeatures]>(input: Tensor<InputShape>): Tensor<LinearForwardShape<InputShape, OutFeatures>>;
  __call__(input: TensorLike): Tensor;
  forward<const Input extends TensorLike>(input: Input): Tensor<LinearForwardShape<TensorLikeShape<Input>, OutFeatures>>;
  forward<const InputShape extends readonly [InFeatures] | readonly [number, InFeatures]>(input: Tensor<InputShape>): Tensor<LinearForwardShape<InputShape, OutFeatures>>;
  forward(input: TensorLike): Tensor;
  compile<const S extends readonly [InFeatures] | readonly [number, InFeatures]>(options: CompileOptionsWithInputShape<S>): Program<S, LinearForwardShape<S, OutFeatures>>;
  compile(options?: CompileOptions): Program<readonly [InFeatures], readonly [OutFeatures]>;
  compileSupport<const S extends readonly [InFeatures] | readonly [number, InFeatures]>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, LinearForwardShape<S, OutFeatures>>;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  requireCompileSupport<const S extends readonly [InFeatures] | readonly [number, InFeatures]>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, LinearForwardShape<S, OutFeatures>>;
  requireCompileSupport(options?: CompileOptions): ModuleCompileSupport;
  compilePlan<const S extends readonly [InFeatures] | readonly [number, InFeatures]>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, LinearForwardShape<S, OutFeatures>>;
  compilePlan(options?: CompileOptions): ModuleCompileExplanation;
  requireCompilePlan<const S extends readonly [InFeatures] | readonly [number, InFeatures]>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, LinearForwardShape<S, OutFeatures>>;
  requireCompilePlan(options?: CompileOptions): ModuleCompileExplanation;
  outputShape<const S extends readonly [InFeatures] | readonly [number, InFeatures]>(options: CompileOptionsWithInputShape<S>): LinearForwardShape<S, OutFeatures> | null;
  outputShape(options?: CompileOptions): readonly number[] | null;
  output_shape<const S extends readonly [InFeatures] | readonly [number, InFeatures]>(options: CompileOptionsWithInputShape<S>): LinearForwardShape<S, OutFeatures> | null;
  output_shape(options?: CompileOptions): readonly number[] | null;
  bindParameters<const S extends readonly [InFeatures] | readonly [number, InFeatures]>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, LinearForwardShape<S, OutFeatures>>;
  bindParameters(options?: CompileOptions): ModuleBindings;
  bind_parameters<const S extends readonly [InFeatures] | readonly [number, InFeatures]>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, LinearForwardShape<S, OutFeatures>>;
  bind_parameters(options?: CompileOptions): ModuleBindings;
  placeParameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  place_parameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  place_parameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  compileForTraining(optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  compile_for_training(optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  forTraining(optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  for_training(optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  trainingStep(optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  training_step(optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
}

export interface EmbeddingModule<NumEmbeddings extends number = number, EmbeddingDim extends number = number> extends NnModule {
  readonly kind: "embedding";
  readonly numEmbeddings: NumEmbeddings;
  readonly embeddingDim: EmbeddingDim;
  call<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<EmbeddingForwardShape<S, EmbeddingDim>>;
  __call__<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<EmbeddingForwardShape<S, EmbeddingDim>>;
  forward<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<EmbeddingForwardShape<S, EmbeddingDim>>;
  compile<const S extends TensorShapeTuple>(options: EmbeddingCompileOptions<S>): Program<S, EmbeddingForwardShape<S, EmbeddingDim>>;
  compileSupport<const S extends TensorShapeTuple>(options: EmbeddingCompileOptions<S>): ModuleCompileSupport<S, EmbeddingForwardShape<S, EmbeddingDim>>;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  requireCompileSupport<const S extends TensorShapeTuple>(options: EmbeddingCompileOptions<S>): ModuleCompileSupport<S, EmbeddingForwardShape<S, EmbeddingDim>>;
  requireCompileSupport(options?: CompileOptions): ModuleCompileSupport;
  compilePlan<const S extends TensorShapeTuple>(options: EmbeddingCompileOptions<S>): ModuleCompileExplanation<S, EmbeddingForwardShape<S, EmbeddingDim>>;
  compilePlan(options?: CompileOptions): ModuleCompileExplanation;
  requireCompilePlan<const S extends TensorShapeTuple>(options: EmbeddingCompileOptions<S>): ModuleCompileExplanation<S, EmbeddingForwardShape<S, EmbeddingDim>>;
  requireCompilePlan(options?: CompileOptions): ModuleCompileExplanation;
  outputShape<const S extends TensorShapeTuple>(options: EmbeddingCompileOptions<S>): EmbeddingForwardShape<S, EmbeddingDim> | null;
  outputShape(options?: CompileOptions): readonly number[] | null;
  output_shape<const S extends TensorShapeTuple>(options: EmbeddingCompileOptions<S>): EmbeddingForwardShape<S, EmbeddingDim> | null;
  output_shape(options?: CompileOptions): readonly number[] | null;
  bindParameters<const S extends TensorShapeTuple>(options: EmbeddingCompileOptions<S>): ModuleBindings<S, EmbeddingForwardShape<S, EmbeddingDim>>;
  bind_parameters<const S extends TensorShapeTuple>(options: EmbeddingCompileOptions<S>): ModuleBindings<S, EmbeddingForwardShape<S, EmbeddingDim>>;
  placeParameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  place_parameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  place_parameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
}

export interface Conv2dModule<
  InChannels extends number = number,
  OutChannels extends number = number,
  Kernel extends number = number,
  Stride extends number = 1,
> extends NnModule {
  readonly kind: "conv2d";
  readonly inChannels: InChannels;
  readonly outChannels: OutChannels;
  readonly kernelSize: readonly [Kernel, Kernel] | readonly [number, number];
  readonly stride: readonly [Stride, Stride] | readonly [number, number];
  readonly padding: readonly [number, number];
  readonly dilation: readonly [number, number];
  readonly groups: number;
  readonly weight: Float32Array;
  readonly bias: Float32Array | null;
  call<const InputShape extends readonly [InChannels, number, number] | readonly [number, InChannels, number, number]>(input: Tensor<InputShape>): Tensor<Conv2dForwardShape<InputShape, OutChannels, Kernel, Stride>>;
  call(input: TensorLike): Tensor;
  __call__<const InputShape extends readonly [InChannels, number, number] | readonly [number, InChannels, number, number]>(input: Tensor<InputShape>): Tensor<Conv2dForwardShape<InputShape, OutChannels, Kernel, Stride>>;
  __call__(input: TensorLike): Tensor;
  forward<const InputShape extends readonly [InChannels, number, number] | readonly [number, InChannels, number, number]>(input: Tensor<InputShape>): Tensor<Conv2dForwardShape<InputShape, OutChannels, Kernel, Stride>>;
  forward(input: TensorLike): Tensor;
  compile<const S extends readonly [InChannels, number, number] | readonly [number, InChannels, number, number]>(options: CompileOptionsWithInputShape<S>): Program<S, Conv2dForwardShape<S, OutChannels, Kernel, Stride>>;
  compile(options?: CompileOptions): Program;
  canCompile(options?: CompileOptions): boolean;
  can_compile(options?: CompileOptions): boolean;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  requireCompileSupport(options?: CompileOptions): ModuleCompileSupport;
  compilePlan(options?: CompileOptions): ModuleCompileExplanation;
  requireCompilePlan(options?: CompileOptions): ModuleCompileExplanation;
  outputShape(options?: CompileOptions): readonly number[] | null;
  output_shape(options?: CompileOptions): readonly number[] | null;
  bindParameters<const S extends readonly [InChannels, number, number] | readonly [number, InChannels, number, number]>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, Conv2dForwardShape<S, OutChannels, Kernel, Stride>>;
  bindParameters(options?: CompileOptions): ModuleBindings;
  bind_parameters<const S extends readonly [InChannels, number, number] | readonly [number, InChannels, number, number]>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, Conv2dForwardShape<S, OutChannels, Kernel, Stride>>;
  bind_parameters(options?: CompileOptions): ModuleBindings;
  placeParameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  place_parameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  place_parameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
}

export interface MaxPool2dModule<Kernel extends number = number, Stride extends number = Kernel> extends NnModule {
  readonly kind: "maxPool2d";
  readonly kernelSize: readonly [Kernel, Kernel] | readonly [number, number];
  readonly stride: readonly [Stride, Stride] | readonly [number, number];
  readonly padding: readonly [number, number];
  readonly dilation: readonly [number, number];
  readonly ceilMode: boolean;
  call<const InputShape extends readonly [number, number, number] | readonly [number, number, number, number]>(input: Tensor<InputShape>): Tensor<MaxPool2dForwardShape<InputShape, Kernel, Stride>>;
  call(input: TensorLike): Tensor;
  __call__<const InputShape extends readonly [number, number, number] | readonly [number, number, number, number]>(input: Tensor<InputShape>): Tensor<MaxPool2dForwardShape<InputShape, Kernel, Stride>>;
  __call__(input: TensorLike): Tensor;
  forward<const InputShape extends readonly [number, number, number] | readonly [number, number, number, number]>(input: Tensor<InputShape>): Tensor<MaxPool2dForwardShape<InputShape, Kernel, Stride>>;
  forward(input: TensorLike): Tensor;
  compile<const S extends readonly [number, number, number] | readonly [number, number, number, number]>(options: CompileOptionsWithInputShape<S>): Program<S, MaxPool2dForwardShape<S, Kernel, Stride>>;
  compile(options?: CompileOptions): Program;
  canCompile(options?: CompileOptions): boolean;
  can_compile(options?: CompileOptions): boolean;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  requireCompileSupport(options?: CompileOptions): ModuleCompileSupport;
  compilePlan(options?: CompileOptions): ModuleCompileExplanation;
  requireCompilePlan(options?: CompileOptions): ModuleCompileExplanation;
  outputShape(options?: CompileOptions): readonly number[] | null;
  output_shape(options?: CompileOptions): readonly number[] | null;
  bindParameters<const S extends readonly [number, number, number] | readonly [number, number, number, number]>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, MaxPool2dForwardShape<S, Kernel, Stride>>;
  bindParameters(options?: CompileOptions): ModuleBindings;
  bind_parameters<const S extends readonly [number, number, number] | readonly [number, number, number, number]>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, MaxPool2dForwardShape<S, Kernel, Stride>>;
  bind_parameters(options?: CompileOptions): ModuleBindings;
  placeParameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  place_parameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  place_parameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
}

export interface AvgPool2dModule<Kernel extends number = number, Stride extends number = Kernel> extends NnModule {
  readonly kind: "avgPool2d";
  readonly kernelSize: readonly [Kernel, Kernel] | readonly [number, number];
  readonly stride: readonly [Stride, Stride] | readonly [number, number];
  readonly padding: readonly [number, number];
  readonly ceilMode: boolean;
  readonly countIncludePad: boolean;
  call<const InputShape extends readonly [number, number, number] | readonly [number, number, number, number]>(input: Tensor<InputShape>): Tensor<MaxPool2dForwardShape<InputShape, Kernel, Stride>>;
  call(input: TensorLike): Tensor;
  __call__<const InputShape extends readonly [number, number, number] | readonly [number, number, number, number]>(input: Tensor<InputShape>): Tensor<MaxPool2dForwardShape<InputShape, Kernel, Stride>>;
  __call__(input: TensorLike): Tensor;
  forward<const InputShape extends readonly [number, number, number] | readonly [number, number, number, number]>(input: Tensor<InputShape>): Tensor<MaxPool2dForwardShape<InputShape, Kernel, Stride>>;
  forward(input: TensorLike): Tensor;
  compile<const S extends readonly [number, number, number] | readonly [number, number, number, number]>(options: CompileOptionsWithInputShape<S>): Program<S, MaxPool2dForwardShape<S, Kernel, Stride>>;
  compile(options?: CompileOptions): Program;
  canCompile(options?: CompileOptions): boolean;
  can_compile(options?: CompileOptions): boolean;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  requireCompileSupport(options?: CompileOptions): ModuleCompileSupport;
  compilePlan(options?: CompileOptions): ModuleCompileExplanation;
  requireCompilePlan(options?: CompileOptions): ModuleCompileExplanation;
  outputShape(options?: CompileOptions): readonly number[] | null;
  output_shape(options?: CompileOptions): readonly number[] | null;
  bindParameters<const S extends readonly [number, number, number] | readonly [number, number, number, number]>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, MaxPool2dForwardShape<S, Kernel, Stride>>;
  bindParameters(options?: CompileOptions): ModuleBindings;
  bind_parameters<const S extends readonly [number, number, number] | readonly [number, number, number, number]>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, MaxPool2dForwardShape<S, Kernel, Stride>>;
  bind_parameters(options?: CompileOptions): ModuleBindings;
  placeParameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  place_parameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  place_parameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
}

export type CompilableActivationKind = "gelu" | "relu" | "silu" | "sigmoid" | "tanh" | "exp" | "log" | "neg" | "recip" | "abs" | "sgn" | "step" | "sqrt" | "square";
export type ActivationKind = CompilableActivationKind;

export interface ActivationModule extends NnModule {
  readonly kind: ActivationKind;
  call<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  call(input: TensorLike): Tensor;
  __call__<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  __call__(input: TensorLike): Tensor;
  forward<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  forward(input: TensorLike): Tensor;
}

export interface CompilableActivationModule extends ActivationModule {
  readonly kind: CompilableActivationKind;
  compile<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): Program<S, S>;
  compile(options?: CompileOptions): Program;
  compileSupport<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, S>;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  requireCompileSupport<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, S>;
  requireCompileSupport(options?: CompileOptions): ModuleCompileSupport;
  compilePlan<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, S>;
  compilePlan(options?: CompileOptions): ModuleCompileExplanation;
  requireCompilePlan<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, S>;
  requireCompilePlan(options?: CompileOptions): ModuleCompileExplanation;
  outputShape<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): S | null;
  outputShape(options?: CompileOptions): readonly number[] | null;
  output_shape<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): S | null;
  output_shape(options?: CompileOptions): readonly number[] | null;
  bindParameters<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, S>;
  bindParameters(options?: CompileOptions): ModuleBindings;
  bind_parameters<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, S>;
  bind_parameters(options?: CompileOptions): ModuleBindings;
  placeParameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  place_parameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  place_parameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
}

export interface SoftmaxModule extends NnModule {
  readonly kind: "softmax";
  readonly dim: number;
  call<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  call(input: TensorLike): Tensor;
  __call__<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  __call__(input: TensorLike): Tensor;
  forward<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  forward(input: TensorLike): Tensor;
  compile<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): Program<S, S>;
  compile(options?: CompileOptions): Program;
  compileSupport<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, S>;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  requireCompileSupport<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, S>;
  requireCompileSupport(options?: CompileOptions): ModuleCompileSupport;
  compilePlan<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, S>;
  compilePlan(options?: CompileOptions): ModuleCompileExplanation;
  requireCompilePlan<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, S>;
  requireCompilePlan(options?: CompileOptions): ModuleCompileExplanation;
  outputShape<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): S | null;
  outputShape(options?: CompileOptions): readonly number[] | null;
  output_shape<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): S | null;
  output_shape(options?: CompileOptions): readonly number[] | null;
  bindParameters<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, S>;
  bindParameters(options?: CompileOptions): ModuleBindings;
  bind_parameters<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, S>;
  bind_parameters(options?: CompileOptions): ModuleBindings;
  placeParameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  place_parameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  place_parameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
}

export interface LogSoftmaxModule extends NnModule {
  readonly kind: "logSoftmax";
  readonly dim: number;
  call<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  call(input: TensorLike): Tensor;
  __call__<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  __call__(input: TensorLike): Tensor;
  forward<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  forward(input: TensorLike): Tensor;
  compile<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): Program<S, S>;
  compile(options?: CompileOptions): Program;
  compileSupport<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, S>;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  requireCompileSupport<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, S>;
  requireCompileSupport(options?: CompileOptions): ModuleCompileSupport;
  compilePlan<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, S>;
  compilePlan(options?: CompileOptions): ModuleCompileExplanation;
  requireCompilePlan<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, S>;
  requireCompilePlan(options?: CompileOptions): ModuleCompileExplanation;
  outputShape<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): S | null;
  outputShape(options?: CompileOptions): readonly number[] | null;
  output_shape<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): S | null;
  output_shape(options?: CompileOptions): readonly number[] | null;
  bindParameters<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, S>;
  bindParameters(options?: CompileOptions): ModuleBindings;
  bind_parameters<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, S>;
  bind_parameters(options?: CompileOptions): ModuleBindings;
  placeParameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  place_parameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  place_parameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
}

export interface ReductionModule<Dim extends number = number> extends NnModule {
  readonly kind: ModuleReductionKind;
  readonly dim: Dim;
  call<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S, Dim>>;
  call(input: TensorLike): Tensor;
  __call__<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S, Dim>>;
  __call__(input: TensorLike): Tensor;
  forward<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ReductionShape<S, Dim>>;
  forward(input: TensorLike): Tensor;
  compile<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): Program<S, ReductionShape<S, Dim>>;
  compile(options?: CompileOptions): Program;
  compileSupport<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, ReductionShape<S, Dim>>;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  requireCompileSupport<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, ReductionShape<S, Dim>>;
  requireCompileSupport(options?: CompileOptions): ModuleCompileSupport;
  compilePlan<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ReductionShape<S, Dim>>;
  compilePlan(options?: CompileOptions): ModuleCompileExplanation;
  requireCompilePlan<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ReductionShape<S, Dim>>;
  requireCompilePlan(options?: CompileOptions): ModuleCompileExplanation;
  outputShape<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ReductionShape<S, Dim> | null;
  outputShape(options?: CompileOptions): readonly number[] | null;
  output_shape<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ReductionShape<S, Dim> | null;
  output_shape(options?: CompileOptions): readonly number[] | null;
  bindParameters<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, ReductionShape<S, Dim>>;
  bindParameters(options?: CompileOptions): ModuleBindings;
  bind_parameters<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, ReductionShape<S, Dim>>;
  bind_parameters(options?: CompileOptions): ModuleBindings;
  placeParameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  place_parameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  place_parameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
}

export interface DropoutModule extends NnModule {
  readonly kind: "dropout";
  readonly p: number;
  call<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  call(input: TensorLike): Tensor;
  __call__<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  __call__(input: TensorLike): Tensor;
  forward<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  forward(input: TensorLike): Tensor;
}

export interface FeatureNormModule extends NnModule {
  readonly kind: "layerNorm" | "rmsNorm" | "batchNorm1d";
  readonly features: number;
  readonly momentum: number;
  readonly trackRunningStats: boolean;
  readonly runningMean: NnBuffer | null;
  readonly runningVar: NnBuffer | null;
  readonly numBatchesTracked: NnBuffer | null;
  call<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  call(input: TensorLike): Tensor;
  __call__<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  __call__(input: TensorLike): Tensor;
  forward<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  forward(input: TensorLike): Tensor;
  compile<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): Program<S, S>;
  compile(options?: CompileOptions): Program;
  compileSupport<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, S>;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  requireCompileSupport<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, S>;
  requireCompileSupport(options?: CompileOptions): ModuleCompileSupport;
  compilePlan<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, S>;
  compilePlan(options?: CompileOptions): ModuleCompileExplanation;
  requireCompilePlan<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, S>;
  requireCompilePlan(options?: CompileOptions): ModuleCompileExplanation;
  outputShape<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): S | null;
  outputShape(options?: CompileOptions): readonly number[] | null;
  output_shape<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): S | null;
  output_shape(options?: CompileOptions): readonly number[] | null;
  bindParameters<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, S>;
  bindParameters(options?: CompileOptions): ModuleBindings;
  bind_parameters<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, S>;
  bind_parameters(options?: CompileOptions): ModuleBindings;
  placeParameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  place_parameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  place_parameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
}

export type ShapeModuleKind = "identity" | "reshape" | "view" | "flatten" | "squeeze" | "unsqueeze" | "transpose" | "permute" | "broadcastTo" | "expand" | "narrow" | "select" | "slice" | "diagonal" | "repeat" | "tile";
export type ShapeModuleForwardShape<
  Kind extends ShapeModuleKind,
  TargetShape extends TensorShapeTuple,
  InputShape extends TensorShapeTuple,
  Dim extends number = number,
  Dim1 extends number = number,
  Length extends number = number,
  Start extends number = number,
  End extends number = number,
  Dims extends readonly number[] = readonly number[],
> =
  [Kind] extends ["identity"]
    ? InputShape
    : [Kind] extends ["diagonal"]
      ? DiagonalShape<InputShape>
    : [Kind] extends ["repeat" | "tile"]
      ? RepeatShape<InputShape, TargetShape>
    : [Kind] extends ["reshape" | "view"]
      ? ReshapeShape<InputShape, TargetShape>
      : [Kind] extends ["broadcastTo" | "expand"]
        ? TargetShape
        : [Kind] extends ["squeeze"]
          ? SqueezeShape<InputShape, Dim>
          : [Kind] extends ["unsqueeze"]
            ? UnsqueezeShape<InputShape, Dim>
            : [Kind] extends ["transpose"]
              ? TransposeShape<InputShape, Dim, Dim1>
              : [Kind] extends ["permute"]
                ? PermuteShape<InputShape, Dims>
                : [Kind] extends ["narrow"]
                  ? NarrowShape<InputShape, Dim, Length>
                  : [Kind] extends ["select"]
                    ? SelectShape<InputShape, Dim>
                    : [Kind] extends ["slice"]
                      ? SliceShape<InputShape, Dim, Start, End>
                      : TargetShape;

export interface ShapeModule<
  TargetShape extends TensorShapeTuple = TensorShapeTuple,
  Kind extends ShapeModuleKind = ShapeModuleKind,
  Dim extends number = number,
  Dim1 extends number = number,
  Length extends number = number,
  Start extends number = number,
  End extends number = number,
  Dims extends readonly number[] = readonly number[],
> extends NnModule {
  readonly kind: Kind;
  readonly shape: readonly [...TargetShape];
  readonly dims: readonly number[];
  readonly startDim: number;
  readonly endDim: number;
  readonly squeezeAll: boolean;
  readonly dim: number;
  readonly start: number;
  readonly length: number;
  readonly index: number;
  readonly end: number | null;
  readonly step: number;
  readonly dim0: number;
  readonly dim1: number;
  call<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ShapeModuleForwardShape<Kind, TargetShape, S, Dim, Dim1, Length, Start, End, Dims>>;
  call(input: TensorLike): Tensor;
  __call__<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ShapeModuleForwardShape<Kind, TargetShape, S, Dim, Dim1, Length, Start, End, Dims>>;
  __call__(input: TensorLike): Tensor;
  forward<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<ShapeModuleForwardShape<Kind, TargetShape, S, Dim, Dim1, Length, Start, End, Dims>>;
  forward(input: TensorLike): Tensor;
  compile<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): Program<S, ShapeModuleForwardShape<Kind, TargetShape, S, Dim, Dim1, Length, Start, End, Dims>>;
  compile(options?: CompileOptions): Program;
  compileSupport<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, ShapeModuleForwardShape<Kind, TargetShape, S, Dim, Dim1, Length, Start, End, Dims>>;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  requireCompileSupport<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, ShapeModuleForwardShape<Kind, TargetShape, S, Dim, Dim1, Length, Start, End, Dims>>;
  requireCompileSupport(options?: CompileOptions): ModuleCompileSupport;
  compilePlan<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ShapeModuleForwardShape<Kind, TargetShape, S, Dim, Dim1, Length, Start, End, Dims>>;
  compilePlan(options?: CompileOptions): ModuleCompileExplanation;
  requireCompilePlan<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ShapeModuleForwardShape<Kind, TargetShape, S, Dim, Dim1, Length, Start, End, Dims>>;
  requireCompilePlan(options?: CompileOptions): ModuleCompileExplanation;
  outputShape<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ShapeModuleForwardShape<Kind, TargetShape, S, Dim, Dim1, Length, Start, End, Dims> | null;
  outputShape(options?: CompileOptions): readonly number[] | null;
  output_shape<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ShapeModuleForwardShape<Kind, TargetShape, S, Dim, Dim1, Length, Start, End, Dims> | null;
  output_shape(options?: CompileOptions): readonly number[] | null;
  bindParameters<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, ShapeModuleForwardShape<Kind, TargetShape, S, Dim, Dim1, Length, Start, End, Dims>>;
  bindParameters(options?: CompileOptions): ModuleBindings;
  bind_parameters<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, ShapeModuleForwardShape<Kind, TargetShape, S, Dim, Dim1, Length, Start, End, Dims>>;
  bind_parameters(options?: CompileOptions): ModuleBindings;
  placeParameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  place_parameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  place_parameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
}

export interface SequentialModule<Layers extends readonly NnModule[] = readonly NnModule[]> extends NnModule {
  readonly layers: readonly [...Layers];
  readonly length: number;
  len(): number;
  __len__(): number;
  size(): number;
  at(index: number): NnModule;
  get(index: number): NnModule;
  __getitem__(index: number): NnModule;
  __setitem__(index: number, module: NnModule): this;
  __delitem__(index: number): this;
  pop(index?: number): NnModule | undefined;
  clear(): this;
  append(module: NnModule): this;
  insert(index: number, module: NnModule): this;
  extend(modules: Iterable<NnModule>): this;
  [Symbol.iterator](): IterableIterator<NnModule>;
  call<const Input extends TensorLike>(input: Input): Tensor<SequentialForwardShape<Layers, TensorLikeShape<Input>>>;
  call<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<SequentialForwardShape<Layers, S>>;
  call(input: TensorLike): Tensor;
  __call__<const Input extends TensorLike>(input: Input): Tensor<SequentialForwardShape<Layers, TensorLikeShape<Input>>>;
  __call__<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<SequentialForwardShape<Layers, S>>;
  __call__(input: TensorLike): Tensor;
  forward<const Input extends TensorLike>(input: Input): Tensor<SequentialForwardShape<Layers, TensorLikeShape<Input>>>;
  forward<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<SequentialForwardShape<Layers, S>>;
  forward(input: TensorLike): Tensor;
  compile<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): Program<S, SequentialForwardShape<Layers, S>>;
  compile(options?: CompileOptions): Program;
  compileSupport<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, SequentialForwardShape<Layers, S>>;
  compileSupport(options?: CompileOptions): ModuleCompileSupport;
  requireCompileSupport<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, SequentialForwardShape<Layers, S>>;
  requireCompileSupport(options?: CompileOptions): ModuleCompileSupport;
  compilePlan<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, SequentialForwardShape<Layers, S>>;
  compilePlan(options?: CompileOptions): ModuleCompileExplanation;
  requireCompilePlan<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, SequentialForwardShape<Layers, S>>;
  requireCompilePlan(options?: CompileOptions): ModuleCompileExplanation;
  outputShape<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): SequentialForwardShape<Layers, S> | null;
  outputShape(options?: CompileOptions): readonly number[] | null;
  output_shape<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): SequentialForwardShape<Layers, S> | null;
  output_shape(options?: CompileOptions): readonly number[] | null;
  bindParameters<const S extends TensorShapeTuple>(options: CompileOptionsWithInputShape<S>): ModuleBindings<S, SequentialForwardShape<Layers, S>>;
  bindParameters(options?: CompileOptions): ModuleBindings;
  placeParameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  placeParameters(program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  compileForTraining(optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  compile_for_training(optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  forTraining(optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  for_training(optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  trainingStep(optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  training_step(optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
}

export type SequentialForwardShape<Layers extends readonly NnModule[], InputShape extends TensorShapeTuple> =
  Layers extends readonly [infer Head extends NnModule, ...infer Tail extends readonly NnModule[]]
    ? SequentialForwardShape<Tail, ModuleForwardShape<Head, InputShape>>
    : InputShape;

export type EmbeddingForwardShape<InputShape extends TensorShapeTuple, EmbeddingDim extends number> =
  readonly [...InputShape, EmbeddingDim];

type ConvSpatialDim<Dim extends number, Kernel extends number, Stride extends number> =
  Kernel extends 1
    ? Stride extends 1
      ? Dim
      : number
    : Kernel extends 2
      ? Stride extends 2
        ? DivideNumbers<Dim, 2>
        : number
      : number;
export type Conv2dForwardShape<
  InputShape extends TensorShapeTuple,
  OutChannels extends number,
  Kernel extends number = number,
  Stride extends number = 1,
> =
  InputShape extends readonly [number, number, number, number]
    ? readonly [InputShape[0], OutChannels, ConvSpatialDim<InputShape[2], Kernel, Stride>, ConvSpatialDim<InputShape[3], Kernel, Stride>]
    : InputShape extends readonly [number, infer Height extends number, infer Width extends number]
      ? readonly [OutChannels, ConvSpatialDim<Height, Kernel, Stride>, ConvSpatialDim<Width, Kernel, Stride>]
      : TensorShapeTuple;

export type MaxPool2dForwardShape<
  InputShape extends TensorShapeTuple,
  Kernel extends number = number,
  Stride extends number = Kernel,
> =
  InputShape extends readonly [number, number, number, number]
    ? readonly [InputShape[0], InputShape[1], ConvSpatialDim<InputShape[2], Kernel, Stride>, ConvSpatialDim<InputShape[3], Kernel, Stride>]
    : InputShape extends readonly [infer Channels extends number, infer Height extends number, infer Width extends number]
      ? readonly [Channels, ConvSpatialDim<Height, Kernel, Stride>, ConvSpatialDim<Width, Kernel, Stride>]
      : TensorShapeTuple;

export type ModuleForwardShape<Target extends NnModule, InputShape extends TensorShapeTuple> =
  Target extends CustomModule<infer DeclaredInputShape extends TensorShapeTuple, infer OutputShape extends TensorShapeTuple>
    ? InputShape extends DeclaredInputShape
      ? OutputShape
      : InputShape extends readonly [number, ...DeclaredInputShape]
        ? readonly [number, ...OutputShape]
        : OutputShape
    : Target extends LinearModule<infer InFeatures extends number, infer OutFeatures extends number>
    ? InputShape extends readonly [InFeatures] | readonly [number, InFeatures]
      ? LinearForwardShape<InputShape, OutFeatures>
      : TensorShapeTuple
    : Target extends EmbeddingModule<number, infer EmbeddingDim extends number>
      ? EmbeddingForwardShape<InputShape, EmbeddingDim>
    : Target extends Conv2dModule<number, infer OutChannels extends number, infer Kernel extends number, infer Stride extends number>
      ? Conv2dForwardShape<InputShape, OutChannels, Kernel, Stride>
    : Target extends AvgPool2dModule<infer Kernel extends number, infer Stride extends number>
      ? MaxPool2dForwardShape<InputShape, Kernel, Stride>
    : Target extends MaxPool2dModule<infer Kernel extends number, infer Stride extends number>
      ? MaxPool2dForwardShape<InputShape, Kernel, Stride>
    : Target extends ShapeModule<
        infer OutputShape extends TensorShapeTuple,
        infer Kind extends ShapeModuleKind,
        infer Dim extends number,
        infer Dim1 extends number,
        infer Length extends number,
        infer Start extends number,
        infer End extends number,
        infer Dims extends readonly number[]
      >
      ? ShapeModuleForwardShape<Kind, OutputShape, InputShape, Dim, Dim1, Length, Start, End, Dims>
      : Target extends ReductionModule<infer Dim extends number>
        ? ReductionShape<InputShape, Dim>
        : Target extends ActivationModule | SoftmaxModule | LogSoftmaxModule | DropoutModule | FeatureNormModule
          ? InputShape
          : Target extends SequentialModule<infer Layers extends readonly NnModule[]>
            ? SequentialForwardShape<Layers, InputShape>
            : TensorShapeTuple;

export type ModuleTargetForwardShape<Target extends NnModule | readonly NnModule[], InputShape extends TensorShapeTuple> =
  Target extends readonly NnModule[]
    ? SequentialForwardShape<Target, InputShape>
    : Target extends NnModule
      ? ModuleForwardShape<Target, InputShape>
      : TensorShapeTuple;

export type ProgramCompatibleModule<
  InputShape extends TensorShapeTuple,
  OutputShape extends TensorShapeTuple,
  Target extends EmbeddingModule | NnCompilableModule,
> = ModuleForwardShape<Target, InputShape> extends OutputShape ? Target : never;

export type TrainBatchInputShape<Batch> =
  Batch extends { input: Tensor<infer InputShape extends TensorShapeTuple> } ? InputShape : TensorShapeTuple;

export type TrainBatchTargetShape<Batch> =
  Batch extends { target?: Tensor<infer TargetShape extends TensorShapeTuple> } ? TargetShape : TensorShapeTuple;

export type TrainModuleOutputShape<Target extends NnModule, Batch> =
  ModuleForwardShape<Target, TrainBatchInputShape<Batch>>;

export type TrainSupervisedBatch<Target extends NnModule, Batch> =
  TrainModuleOutputShape<Target, Batch> extends TrainBatchTargetShape<Batch> ? Batch : never;

export type TrainClassificationTargetShape<LogitsShape extends TensorShapeTuple> =
  LogitsShape extends readonly [...infer TargetShape extends readonly number[], number]
    ? readonly [...TargetShape]
    : TensorShapeTuple;

export type TrainClassificationBatch<Target extends NnModule, Batch> =
  TrainBatchTargetShape<Batch> extends TrainClassificationTargetShape<TrainModuleOutputShape<Target, Batch>> ? Batch : never;

export type TrainSupervisedCriterion<Target extends NnModule, Batch> = {
  forward(
    prediction: Tensor<TrainModuleOutputShape<Target, Batch>>,
    target: Tensor<TrainBatchTargetShape<Batch>>,
  ): Tensor<readonly [1]>;
};

export type TrainClassificationCriterion<Target extends NnModule, Batch> = {
  forward(
    logits: Tensor<TrainModuleOutputShape<Target, Batch>>,
    targets: Tensor<TrainClassificationTargetShape<TrainModuleOutputShape<Target, Batch>>>,
  ): Tensor<readonly [1]>;
};

export type NnCompilableModule =
  | LinearModule
  | Conv2dModule
  | AvgPool2dModule
  | MaxPool2dModule
  | CompilableActivationModule
  | SoftmaxModule
  | LogSoftmaxModule
  | ReductionModule
  | FeatureNormModule
  | ShapeModule
  | SequentialModule
  | CustomModule
  | ModuleBase;

export type SGDConfig = {
  lr?: number;
  momentum?: number;
  weightDecay?: number;
  weight_decay?: number;
};

export type AdamConfig = {
  lr?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay?: number;
  weight_decay?: number;
};

export type RMSpropConfig = {
  lr?: number;
  alpha?: number;
  eps?: number;
  momentum?: number;
  weightDecay?: number;
  weight_decay?: number;
};

export type AdagradConfig = {
  lr?: number;
  lrDecay?: number;
  lr_decay?: number;
  eps?: number;
  weightDecay?: number;
  weight_decay?: number;
};

export type OptimizerParameterSource = Tensor | NnParameter[] | { parameters(): NnParameter[] };
export type OptimizerParamGroupInput = {
  params: OptimizerParameterSource;
  lr?: number;
  weightDecay?: number;
  weight_decay?: number;
};
export type OptimizerTarget = OptimizerParameterSource | readonly OptimizerParamGroupInput[];
export type OptimizerParamGroupSnapshot = Readonly<{
  index: number;
  lr: number;
  weightDecay: number;
  weight_decay: number;
  paramCount: number;
  start: number;
  end: number;
}>;

export type OptimizerConfigSnapshot<Kind extends OptimizerStateKind = OptimizerStateKind> = Readonly<{
  kind: Kind;
  signature: string;
  lr: number;
  momentum?: number;
  alpha?: number;
  lrDecay?: number;
  lr_decay?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay: number;
  weight_decay: number;
  decoupledWeightDecay?: boolean;
  step?: number;
  paramCount: number;
  paramGroups: readonly OptimizerParamGroupSnapshot[];
  param_groups: readonly OptimizerParamGroupSnapshot[];
}>;

export interface Optimizer<Kind extends OptimizerStateKind = OptimizerStateKind> {
  readonly kind: Kind;
  readonly params: NnParameter[];
  readonly paramGroups: readonly OptimizerParamGroupSnapshot[];
  readonly param_groups: readonly OptimizerParamGroupSnapshot[];
  readonly defaults: OptimizerConfigSnapshot<Kind>;
  step(): void;
  zeroGrad(options?: ZeroGradOptions): void;
  zero_grad(options?: ZeroGradOptions): void;
  setLearningRate(lr: number): this;
  set_lr(lr: number): this;
  getLearningRate(): number;
  get_lr(): number;
  addParamGroup(group: OptimizerParamGroupInput): this;
  add_param_group(group: OptimizerParamGroupInput): this;
  config(): OptimizerConfigSnapshot<Kind>;
  stateDict(): OptimizerStateSnapshot<Kind>;
  state_dict(): OptimizerStateSnapshot<Kind>;
  loadStateDict(source: OptimizerStateDict<Kind>, options?: LoadStateDictOptions): this;
  load_state_dict(source: OptimizerStateDict<Kind>, options?: LoadStateDictOptions): this;
}

export type LRSchedulerStateKind = "step-lr" | "exponential-lr" | "cosine-annealing-lr" | "reduce-lr-on-plateau";
export type LRSchedulerConfig = {
  gamma?: number;
};
export type StepLRConfig = LRSchedulerConfig & {
  stepSize?: number;
  step_size?: number;
};
export type CosineAnnealingLRConfig = LRSchedulerConfig & {
  tMax?: number;
  t_max?: number;
  etaMin?: number;
  eta_min?: number;
};
export type ReduceLROnPlateauConfig = {
  mode?: "min" | "max";
  factor?: number;
  patience?: number;
  threshold?: number;
  thresholdMode?: "rel" | "abs";
  threshold_mode?: "rel" | "abs";
  cooldown?: number;
  minLr?: number;
  min_lr?: number;
  eps?: number;
};
export type LRSchedulerStateSnapshot<
  Kind extends LRSchedulerStateKind = LRSchedulerStateKind,
  OptimizerKind extends OptimizerStateKind | null = OptimizerStateKind | null,
> = Readonly<{
  kind: Kind;
  signature: string;
  step: number;
  baseLr: number;
  base_lr: number;
  lastLr: number;
  last_lr: number;
  gamma: number;
  stepSize?: number;
  step_size?: number;
  tMax?: number;
  t_max?: number;
  etaMin?: number;
  eta_min?: number;
  mode?: "min" | "max";
  factor?: number;
  patience?: number;
  threshold?: number;
  thresholdMode?: "rel" | "abs";
  threshold_mode?: "rel" | "abs";
  cooldown?: number;
  cooldownCounter?: number;
  cooldown_counter?: number;
  minLr?: number;
  min_lr?: number;
  eps?: number;
  best?: number;
  badEpochs?: number;
  bad_epochs?: number;
  optimizerKind: OptimizerKind;
}>;
export type LRSchedulerStateDict<Kind extends LRSchedulerStateKind = LRSchedulerStateKind> = {
  kind: Kind;
  step?: number;
  baseLr?: number;
  base_lr?: number;
  lastLr?: number;
  last_lr?: number;
  gamma?: number;
  stepSize?: number;
  step_size?: number;
  tMax?: number;
  t_max?: number;
  etaMin?: number;
  eta_min?: number;
  mode?: "min" | "max";
  factor?: number;
  patience?: number;
  threshold?: number;
  thresholdMode?: "rel" | "abs";
  threshold_mode?: "rel" | "abs";
  cooldown?: number;
  cooldownCounter?: number;
  cooldown_counter?: number;
  minLr?: number;
  min_lr?: number;
  eps?: number;
  best?: number;
  badEpochs?: number;
  bad_epochs?: number;
};
export interface LRScheduler<
  Kind extends LRSchedulerStateKind = LRSchedulerStateKind,
  OptimizerKind extends OptimizerStateKind | null = OptimizerStateKind | null,
> {
  readonly baseLr: number;
  readonly base_lr: number;
  readonly lastLr: number;
  readonly last_lr: number;
  readonly stepSize: number;
  readonly step_size: number;
  readonly tMax?: number;
  readonly t_max?: number;
  readonly etaMin?: number;
  readonly eta_min?: number;
  readonly mode?: "min" | "max";
  readonly factor?: number;
  readonly patience?: number;
  readonly threshold?: number;
  readonly thresholdMode?: "rel" | "abs";
  readonly threshold_mode?: "rel" | "abs";
  readonly cooldown?: number;
  readonly cooldownCounter?: number;
  readonly cooldown_counter?: number;
  readonly minLr?: number;
  readonly min_lr?: number;
  readonly eps?: number;
  readonly best?: number;
  readonly badEpochs?: number;
  readonly bad_epochs?: number;
  step(metric?: number): number;
  getLastLr(): number;
  get_last_lr(): number;
  config(): LRSchedulerStateSnapshot<Kind, OptimizerKind>;
  stateDict(): LRSchedulerStateSnapshot<Kind, OptimizerKind>;
  state_dict(): LRSchedulerStateSnapshot<Kind, OptimizerKind>;
  loadStateDict(source: LRSchedulerStateDict<Kind>, options?: { strict?: boolean; validateOnly?: boolean }): this;
  load_state_dict(source: LRSchedulerStateDict<Kind>, options?: { strict?: boolean; validateOnly?: boolean }): this;
}

export type LRSchedulerNamespace = Readonly<{
  stepLR<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: StepLRConfig): LRScheduler<"step-lr", Kind>;
  step_lr<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: StepLRConfig): LRScheduler<"step-lr", Kind>;
  StepLR: new <const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: StepLRConfig) => LRScheduler<"step-lr", Kind>;
  exponentialLR<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: LRSchedulerConfig): LRScheduler<"exponential-lr", Kind>;
  exponential_lr<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: LRSchedulerConfig): LRScheduler<"exponential-lr", Kind>;
  ExponentialLR: new <const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: LRSchedulerConfig) => LRScheduler<"exponential-lr", Kind>;
  cosineAnnealingLR<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: CosineAnnealingLRConfig): LRScheduler<"cosine-annealing-lr", Kind>;
  cosine_annealing_lr<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: CosineAnnealingLRConfig): LRScheduler<"cosine-annealing-lr", Kind>;
  CosineAnnealingLR: new <const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: CosineAnnealingLRConfig) => LRScheduler<"cosine-annealing-lr", Kind>;
  reduceLROnPlateau<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: ReduceLROnPlateauConfig): LRScheduler<"reduce-lr-on-plateau", Kind>;
  reduce_lr_on_plateau<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: ReduceLROnPlateauConfig): LRScheduler<"reduce-lr-on-plateau", Kind>;
  ReduceLROnPlateau: new <const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: ReduceLROnPlateauConfig) => LRScheduler<"reduce-lr-on-plateau", Kind>;
}>;

export type TrainStepEvidence<Kind extends OptimizerStateKind | null = OptimizerStateKind | null> = Readonly<{
  kind: "zgml.train.step";
  signature: string;
  optimizerKind: Kind;
  parameterCount: number | null;
  beforeStep: number | null;
  afterStep: number | null;
  stepAdvanced: number | null;
  hadLoss: boolean;
  lossScalar: number | null;
  gradientProvided: boolean;
  zeroGradApplied: boolean;
  gradientsCleared: boolean | null;
  clipGradNormApplied: boolean;
  clipGradValueApplied: boolean;
  gradNormBeforeClip: number | null;
  gradNormAfterClip: number | null;
}>;

export type TrainGradientClipOptions = {
  clipGradNorm?: number;
  clip_grad_norm?: number;
  clipGradNormOptions?: { eps?: number };
  clip_grad_norm_options?: { eps?: number };
  clipGradValue?: number;
  clip_grad_value?: number;
};

export type TrainStepOptions = {
  zeroGrad?: boolean;
  zero_grad?: boolean;
  zeroGradOptions?: ZeroGradOptions;
  zero_grad_options?: ZeroGradOptions;
  loss?: Tensor;
  gradient?: TensorLike;
  inspect?: boolean;
  evidence?: boolean;
} & TrainGradientClipOptions;

export type TrainLossStepOptions = {
  zeroGrad?: boolean;
  zero_grad?: boolean;
  zeroGradOptions?: ZeroGradOptions;
  zero_grad_options?: ZeroGradOptions;
  gradient?: TensorLike;
} & TrainGradientClipOptions;

export type TrainFitStepEvidence<OptimizerKind extends OptimizerStateKind | null = OptimizerStateKind | null> = Readonly<{
  kind: "zgml.train.fit-step";
  signature: string;
  epoch: number;
  batchIndex: number;
  step: number;
  sampleIndices: readonly number[] | null;
  sample_indices: readonly number[] | null;
  loss: number;
  stepEvidence: TrainStepEvidence<OptimizerKind> | null;
}>;

export type TrainFitContext = Readonly<{
  epoch: number;
  batchIndex: number;
  step: number;
  sampleIndices: readonly number[] | null;
  sample_indices: readonly number[] | null;
}>;

export type TrainEarlyStoppingOptions = {
  patience?: number;
  minDelta?: number;
  min_delta?: number;
  mode?: "min" | "max";
};

export type TrainFitOptions<OptimizerKind extends OptimizerStateKind | null = OptimizerStateKind | null> = {
  epochs?: number;
  maxSteps?: number;
  max_steps?: number;
  zeroGrad?: boolean;
  zero_grad?: boolean;
  zeroGradOptions?: ZeroGradOptions;
  zero_grad_options?: ZeroGradOptions;
  gradient?: TensorLike;
  native?: boolean;
  autoNative?: boolean;
  auto_native?: boolean;
  requireNative?: boolean;
  require_native?: boolean;
  compile?: boolean;
  inputShape?: readonly number[];
  input_shape?: readonly number[];
  batchSize?: number;
  batch_size?: number;
  classes?: number;
  numClasses?: number;
  num_classes?: number;
  earlyStopping?: boolean | TrainEarlyStoppingOptions;
  early_stopping?: boolean | TrainEarlyStoppingOptions;
  earlyStoppingPatience?: number;
  early_stopping_patience?: number;
  earlyStoppingMinDelta?: number;
  early_stopping_min_delta?: number;
  earlyStoppingMode?: "min" | "max";
  early_stopping_mode?: "min" | "max";
  onStep?: (evidence: TrainFitStepEvidence<OptimizerKind>) => void;
  on_step?: (evidence: TrainFitStepEvidence<OptimizerKind>) => void;
} & TrainGradientClipOptions;

export type CompileTrainingOptions = Readonly<Record<string, unknown> & {
  inputShape?: readonly number[];
  input_shape?: readonly number[];
  batchSize?: number;
  batch_size?: number;
  loss?: "crossEntropy" | "cross_entropy" | "mse" | "meanSquaredError" | "mean_squared_error";
  criterion?: "crossEntropy" | "cross_entropy" | "mse" | "meanSquaredError" | "mean_squared_error";
  classes?: number;
  numClasses?: number;
  num_classes?: number;
}>;

export type CompiledTrainingStepEvidence = Readonly<{
  kind: "zgml.native-training-step";
  native: true;
  backend: string;
  loss: number;
  correct?: number;
  accuracy?: number;
  batch?: number;
  optimizerStep?: number;
}>;

export type CompiledTrainingPlan = Readonly<{
  kind: "zgml.native-training-plan";
  native: true;
  loweredBy: "zig-ffi";
  runtimePath: string;
  backend: string;
  modelKind: string;
  optimizerKind: string;
  lossKind: string;
  inputShape: readonly [number, number];
  outputShape: readonly [number, number];
  parameterCount: number;
  parameterElements: number;
  kernels: readonly string[];
  workspace: Readonly<Record<string, number>>;
}>;

export type NativeTrainingBulkFitPlan = Readonly<{
  kind: "zgml.native-training-bulk-fit-plan";
  supported: boolean;
  native: boolean;
  nativeBulk: boolean;
  native_bulk: boolean;
  backend: string | null;
  reason: string | null;
  kernel: string | null;
  batch: number | null;
  sampleCount: number | null;
  sample_count: number | null;
  trainedSampleCount: number | null;
  trained_sample_count: number | null;
  datasetSampleCount: number | null;
  dataset_sample_count: number | null;
  epochs: number | null;
  steps: number | null;
  plannedSteps: number | null;
  planned_steps: number | null;
  stoppedEarly: boolean;
  stopped_early: boolean;
  stopReason: "max-steps" | null;
  stop_reason: "max-steps" | null;
}>;

export type NativeTrainingExplanation = Readonly<{
  kind: "zgml.train.native-training-explanation";
  supported: boolean;
  native: boolean;
  nativeBulk: boolean;
  native_bulk: boolean;
  loweredBy: "zig-ffi" | null;
  runtimePath: "JS/TS module API -> Zig native training kernel";
  backend: string | null;
  reason: string | null;
  compileOptions: Readonly<Record<string, unknown>> | null;
  compile_options: Readonly<Record<string, unknown>> | null;
  plan: CompiledTrainingPlan | null;
  bulkPlan: NativeTrainingBulkFitPlan | null;
  bulk_plan: NativeTrainingBulkFitPlan | null;
  bulkKernel: string | null;
  kernels: readonly string[];
  signature: string;
}>;

export interface CompiledTrainingStep {
  readonly kind: "zgml.compiled-training-step";
  readonly native: true;
  readonly backend: string;
  readonly modelKind: string;
  readonly optimizerKind: string;
  readonly lossKind: string;
  inputShape(): readonly number[];
  outputShape(): readonly number[];
  plan(): CompiledTrainingPlan;
  compileEvidence(): CompiledTrainingPlan;
  compile_evidence(): CompiledTrainingPlan;
  step(input: TensorLike, target: TensorLike | IndexLike): CompiledTrainingStepEvidence;
  forward(input: TensorLike, target: TensorLike | IndexLike): CompiledTrainingStepEvidence;
  fitPlan?(batches: unknown, fitOptions?: TrainFitOptions): NativeTrainingBulkFitPlan | null;
  fit_plan?(batches: unknown, fitOptions?: TrainFitOptions): NativeTrainingBulkFitPlan | null;
  dispose(): void;
  free(): void;
}

export type TrainModelFitOptions<
  OptimizerKind extends OptimizerStateKind | null,
  Target extends NnModule,
  Batch,
> = TrainFitOptions<OptimizerKind> & {
  optimizer: OptimizerKind extends OptimizerStateKind
    ? Optimizer<OptimizerKind>
    : { step(): void; zeroGrad?(options?: ZeroGradOptions): void };
} & (
  | { loss: TrainSupervisedCriterion<Target, Batch>; criterion?: TrainSupervisedCriterion<Target, Batch> }
  | { criterion: TrainSupervisedCriterion<Target, Batch>; loss?: TrainSupervisedCriterion<Target, Batch> }
);

export type TrainFitEvidence<OptimizerKind extends OptimizerStateKind | null = OptimizerStateKind | null> = Readonly<{
  kind: "zgml.train.fit";
  signature: string;
  epochs: number;
  steps: number;
  stoppedEarly: boolean;
  stopReason: "max-steps" | "early-stopping" | null;
  stop_reason: "max-steps" | "early-stopping" | null;
  batchCount: number | null;
  batch_count: number | null;
  sampleCount: number | null;
  sample_count: number | null;
  losses: readonly number[];
  bestLoss: number | null;
  best_loss: number | null;
  bestStep: number | null;
  best_step: number | null;
  finalLoss: number | null;
  lastStep: TrainStepEvidence<OptimizerKind> | null;
  lastLoss: Tensor | null;
  native?: boolean;
  nativeBulk?: boolean;
  native_bulk?: boolean;
  backend?: string | null;
  compiledPlan?: CompiledTrainingPlan | null;
  compiled_plan?: CompiledTrainingPlan | null;
  bulkResult?: unknown;
  bulk_result?: unknown;
}>;

export type TrainEvaluateContext = Readonly<{
  batchIndex: number;
  step: number;
  sampleIndices: readonly number[] | null;
  sample_indices: readonly number[] | null;
}>;

export type TrainEvaluateStepEvidence = Readonly<{
  kind: "zgml.train.evaluate-step";
  signature: string;
  batchIndex: number;
  step: number;
  sampleIndices: readonly number[] | null;
  sample_indices: readonly number[] | null;
  loss: number;
}>;

export type TrainEvaluateOptions = {
  maxSteps?: number;
  max_steps?: number;
  onStep?: (evidence: TrainEvaluateStepEvidence) => void;
  on_step?: (evidence: TrainEvaluateStepEvidence) => void;
};

export type TrainEvaluateEvidence = Readonly<{
  kind: "zgml.train.evaluate";
  signature: string;
  steps: number;
  stoppedEarly: boolean;
  batchCount: number | null;
  batch_count: number | null;
  sampleCount: number | null;
  sample_count: number | null;
  losses: readonly number[];
  meanLoss: number | null;
  mean_loss: number | null;
  finalLoss: number | null;
  final_loss: number | null;
  lastLoss: Tensor | null;
}>;

export type TrainPredictContext = Readonly<{
  batchIndex: number;
  step: number;
  sampleIndices: readonly number[] | null;
  sample_indices: readonly number[] | null;
}>;

export type TrainPredictStepEvidence<TOutput extends Tensor = Tensor> = Readonly<{
  kind: "zgml.train.predict-step";
  signature: string;
  batchIndex: number;
  step: number;
  sampleIndices: readonly number[] | null;
  sample_indices: readonly number[] | null;
  output: TOutput;
}>;

export type TrainPredictOptions<TOutput extends Tensor = Tensor> = {
  maxSteps?: number;
  max_steps?: number;
  onStep?: (evidence: TrainPredictStepEvidence<TOutput>) => void;
  on_step?: (evidence: TrainPredictStepEvidence<TOutput>) => void;
};

export type TrainPredictEvidence<TOutput extends Tensor = Tensor> = Readonly<{
  kind: "zgml.train.predict";
  signature: string;
  steps: number;
  stoppedEarly: boolean;
  batchCount: number | null;
  batch_count: number | null;
  sampleCount: number | null;
  sample_count: number | null;
  outputs: readonly TOutput[];
  lastOutput: TOutput | null;
  last_output: TOutput | null;
}>;

export type TrainClassificationClassMetric = Readonly<{
  classIndex: number;
  class_index: number;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}>;

export type TrainClassificationReport = Readonly<{
  accuracy: number;
  macroPrecision: number;
  macro_precision: number;
  macroRecall: number;
  macro_recall: number;
  macroF1: number;
  macro_f1: number;
  weightedPrecision: number;
  weighted_precision: number;
  weightedRecall: number;
  weighted_recall: number;
  weightedF1: number;
  weighted_f1: number;
  support: number;
  classes: number;
  confusionMatrix: readonly (readonly number[])[];
  confusion_matrix: readonly (readonly number[])[];
  perClass: readonly TrainClassificationClassMetric[];
  per_class: readonly TrainClassificationClassMetric[];
}>;

export type DataCollateContext = Readonly<{
  batchIndex: number;
  indices: readonly number[];
  sampleIndices: readonly number[];
  sample_indices: readonly number[];
}>;
export type DataCollateFn<TSample = TensorDatasetSample, TBatch = TensorDatasetBatch> =
  (samples: readonly TSample[], context: DataCollateContext) => TBatch;
export type DefaultCollateOptions = {
  batchIndex?: number;
  batch_index?: number;
  indices?: readonly number[];
  sampleIndices?: readonly number[];
  sample_indices?: readonly number[];
};
export type DataBatchOptions<TSample = TensorDatasetSample, TBatch = TensorDatasetBatch> = {
  batchSize?: number;
  batch_size?: number;
  shuffle?: boolean;
  dropLast?: boolean;
  drop_last?: boolean;
  seed?: number;
  rng?: () => number;
  sampler?: Iterable<number>;
  batchSampler?: Iterable<Iterable<number>>;
  batch_sampler?: Iterable<Iterable<number>>;
  collateFn?: DataCollateFn<TSample, TBatch>;
  collate_fn?: DataCollateFn<TSample, TBatch>;
};
export type DataSplitOptions = {
  shuffle?: boolean;
  seed?: number;
  rng?: () => number;
};

export type TensorShapeTail<Shape extends TensorShapeTuple> =
  Shape extends readonly [number, ...infer Tail extends readonly number[]] ? readonly [...Tail] : TensorShapeTuple;
export type TensorDatasetBatchShape<Shape extends TensorShapeTuple> =
  Shape extends readonly [number, ...infer Tail extends readonly number[]] ? readonly [number, ...Tail] : TensorShapeTuple;
export type DefaultCollateTensor<TValue> =
  TValue extends Tensor<infer Shape extends TensorShapeTuple> ? Tensor<readonly [number, ...Shape]> : Tensor;

export type TensorDatasetSample<TInput = Tensor, TTarget = Tensor> = Readonly<{
  kind: "zgml.data.sample";
  index: number;
  input: TInput;
  target?: TTarget;
}>;

export type TensorDatasetBatch<TInput = Tensor, TTarget = Tensor> = Readonly<{
  kind: "zgml.data.batch";
  batchIndex: number;
  indices: readonly number[];
  input: TInput;
  target?: TTarget;
}>;

export type TensorDatasetMapper<
  TSampleInput = Tensor,
  TSampleTarget = Tensor,
  TMappedInput = Tensor,
  TMappedTarget = Tensor,
> = (sample: TensorDatasetSample<TSampleInput, TSampleTarget>, index: number) => {
  input: TMappedInput;
  target?: TMappedTarget;
  readonly [key: string]: unknown;
};

export interface Dataset<TSample = TensorDatasetSample, TBatch = TensorDatasetBatch> extends Iterable<TSample> {
  readonly kind: string;
  readonly length: number;
  len(): number;
  __len__(): number;
  size(): number;
  sample(index: number): TSample;
  get(index: number): TSample;
  at(index: number): TSample;
  __getitem__(index: number): TSample;
  batch(indices: Iterable<number>, batchIndex?: number): TBatch;
  __iter__(): IterableIterator<TSample>;
}

export interface Sampler extends Iterable<number> {
  readonly kind: string;
  readonly length: number;
  len(): number;
  __len__(): number;
  size(): number;
  __iter__(): IterableIterator<number>;
}

export interface SequentialSampler extends Sampler {
  readonly kind: "zgml.data.sequential-sampler";
  readonly dataSource: Dataset;
}

export interface RandomSampler extends Sampler {
  readonly kind: "zgml.data.random-sampler";
  readonly dataSource: Dataset;
  readonly replacement: boolean;
  readonly numSamples: number;
  readonly seed: number | undefined;
}

export interface BatchSampler extends Iterable<readonly number[]> {
  readonly kind: "zgml.data.batch-sampler";
  readonly sampler: Iterable<number>;
  readonly batchSize: number;
  readonly dropLast: boolean;
  readonly length: number;
  len(): number;
  __len__(): number;
  size(): number;
  __iter__(): IterableIterator<readonly number[]>;
}

export type TensorDataset<TSampleInput = Tensor, TSampleTarget = Tensor, TBatchInput = TSampleInput, TBatchTarget = TSampleTarget> =
  Dataset<TensorDatasetSample<TSampleInput, TSampleTarget>, TensorDatasetBatch<TBatchInput, TBatchTarget>> & Readonly<{
  kind: "zgml.data.tensor-dataset" | "zgml.data.mapped-dataset" | "zgml.data.concat-dataset";
  input?: TBatchInput;
  target?: TBatchTarget;
}>;

export type DataBatches<
  TInput = Tensor,
  TTarget = Tensor,
  TBatch = TensorDatasetBatch<TInput, TTarget>,
> = Iterable<TBatch> & Readonly<{
  kind: "zgml.data.batches";
  length: number;
  sampleCount: number;
  sample_count: number;
  batchSize: number;
  batch_size: number;
  batchCount: number;
  batch_count: number;
  dropLast: boolean;
  drop_last: boolean;
  shuffle: boolean;
  sampler: Iterable<number> | null;
  batchSampler: Iterable<Iterable<number>> | null;
  batch_sampler: Iterable<Iterable<number>> | null;
  collateFn: DataCollateFn<unknown, TBatch> | null;
  collate_fn: DataCollateFn<unknown, TBatch> | null;
  len(): number;
  __len__(): number;
  size(): number;
  batch(batchIndex: number): TBatch;
  get(batchIndex: number): TBatch;
  at(batchIndex: number): TBatch;
  __getitem__(batchIndex: number): TBatch;
  __iter__(): IterableIterator<TBatch>;
}>;
export type DataLoader<
  TInput = Tensor,
  TTarget = Tensor,
  TBatch = TensorDatasetBatch<TInput, TTarget>,
> = DataBatches<TInput, TTarget, TBatch>;
export type CollatedDataLoader<TBatch> = DataLoader<Tensor, Tensor, TBatch>;
export type DataCollateOptions<TSample, TBatch> =
  DataBatchOptions<TSample, TBatch> & (
    | { collateFn: DataCollateFn<TSample, TBatch> }
    | { collate_fn: DataCollateFn<TSample, TBatch> }
  );

export interface DataNamespace {
  Dataset: {
    new <TSample = TensorDatasetSample, TBatch = TensorDatasetBatch>(): Dataset<TSample, TBatch>;
  };
  TensorDataset: {
    new <const I extends TensorShapeTuple, const T extends TensorShapeTuple>(
      input: Tensor<I>,
      target: Tensor<T>,
    ): TensorDataset<
      Tensor<TensorShapeTail<I>>,
      Tensor<TensorShapeTail<T>>,
      Tensor<TensorDatasetBatchShape<I>>,
      Tensor<TensorDatasetBatchShape<T>>
    >;
    new <const I extends TensorShapeTuple>(
      input: Tensor<I>,
    ): TensorDataset<Tensor<TensorShapeTail<I>>, Tensor, Tensor<TensorDatasetBatchShape<I>>, Tensor>;
    new (input: Tensor, target?: Tensor): TensorDataset<Tensor, Tensor>;
  };
  DataLoader: {
    new <const TSample, const TBatch>(dataset: Dataset<TSample>, options: DataCollateOptions<TSample, TBatch>): CollatedDataLoader<TBatch>;
    new <const TSample, const TBatch extends TensorDatasetBatch>(dataset: Dataset<TSample, TBatch>, options?: DataBatchOptions<TSample, TBatch>): DataLoader<TBatch["input"], TBatch["target"]>;
    new <const TSample, const TBatch extends TensorDatasetBatch>(dataset: Dataset<TSample>, options: DataBatchOptions<TSample, TBatch>): DataLoader<TBatch["input"], TBatch["target"]>;
    new <const SI, const ST, const BI, const BT>(dataset: TensorDataset<SI, ST, BI, BT>, options?: DataBatchOptions<TensorDatasetSample<SI, ST>, TensorDatasetBatch<BI, BT>>): DataLoader<BI, BT>;
    new (dataset: Dataset, options?: DataBatchOptions): DataLoader;
  };
  SequentialSampler: {
    new (dataset: Dataset): SequentialSampler;
  };
  RandomSampler: {
    new (dataset: Dataset, options?: { replacement?: boolean; numSamples?: number; num_samples?: number; seed?: number; rng?: () => number }): RandomSampler;
  };
  BatchSampler: {
    new (sampler: Iterable<number>, batchSize: number, dropLast?: boolean): BatchSampler;
  };
  defaultCollate<const SI, const ST>(samples: readonly TensorDatasetSample<SI, ST>[], options?: DefaultCollateOptions): TensorDatasetBatch<DefaultCollateTensor<SI>, DefaultCollateTensor<ST>>;
  defaultCollate(samples: readonly TensorDatasetSample[], options?: DefaultCollateOptions): TensorDatasetBatch;
  default_collate<const SI, const ST>(samples: readonly TensorDatasetSample<SI, ST>[], options?: DefaultCollateOptions): TensorDatasetBatch<DefaultCollateTensor<SI>, DefaultCollateTensor<ST>>;
  default_collate(samples: readonly TensorDatasetSample[], options?: DefaultCollateOptions): TensorDatasetBatch;
  Subset: {
    new <const SI, const ST, const BI, const BT>(dataset: TensorDataset<SI, ST, BI, BT>, indices: Iterable<number>): TensorDataset<SI, ST, BI, BT>;
    new (dataset: TensorDataset, indices: Iterable<number>): TensorDataset;
  };
  ConcatDataset: {
    new <const SI, const ST, const BI, const BT>(
      datasets: readonly [TensorDataset<SI, ST, BI, BT>, ...TensorDataset<SI, ST, BI, BT>[]],
    ): TensorDataset<SI, ST, BI, BT>;
    new (datasets: Iterable<TensorDataset>): TensorDataset;
  };
  MapDataset: {
    new <const SI, const ST, const MI, const MT>(
      dataset: TensorDataset<SI, ST>,
      mapper: TensorDatasetMapper<SI, ST, MI, MT>,
    ): TensorDataset<MI, MT>;
    new (dataset: TensorDataset, mapper: TensorDatasetMapper): TensorDataset;
  };
  tensorDataset<const I extends TensorShapeTuple, const T extends TensorShapeTuple>(
    input: Tensor<I>,
    target: Tensor<T>,
  ): TensorDataset<
    Tensor<TensorShapeTail<I>>,
    Tensor<TensorShapeTail<T>>,
    Tensor<TensorDatasetBatchShape<I>>,
    Tensor<TensorDatasetBatchShape<T>>
  >;
  tensorDataset<const I extends TensorShapeTuple>(
    input: Tensor<I>,
  ): TensorDataset<Tensor<TensorShapeTail<I>>, Tensor, Tensor<TensorDatasetBatchShape<I>>, Tensor>;
  tensorDataset(input: Tensor, target?: Tensor): TensorDataset<Tensor, Tensor>;
  tensor_dataset<const I extends TensorShapeTuple, const T extends TensorShapeTuple>(
    input: Tensor<I>,
    target: Tensor<T>,
  ): TensorDataset<
    Tensor<TensorShapeTail<I>>,
    Tensor<TensorShapeTail<T>>,
    Tensor<TensorDatasetBatchShape<I>>,
    Tensor<TensorDatasetBatchShape<T>>
  >;
  tensor_dataset<const I extends TensorShapeTuple>(
    input: Tensor<I>,
  ): TensorDataset<Tensor<TensorShapeTail<I>>, Tensor, Tensor<TensorDatasetBatchShape<I>>, Tensor>;
  tensor_dataset(input: Tensor, target?: Tensor): TensorDataset<Tensor, Tensor>;
  defaultCollate<const SI, const ST>(samples: readonly TensorDatasetSample<SI, ST>[], options?: DefaultCollateOptions): TensorDatasetBatch<DefaultCollateTensor<SI>, DefaultCollateTensor<ST>>;
  defaultCollate(samples: readonly TensorDatasetSample[], options?: DefaultCollateOptions): TensorDatasetBatch;
  default_collate<const SI, const ST>(samples: readonly TensorDatasetSample<SI, ST>[], options?: DefaultCollateOptions): TensorDatasetBatch<DefaultCollateTensor<SI>, DefaultCollateTensor<ST>>;
  default_collate(samples: readonly TensorDatasetSample[], options?: DefaultCollateOptions): TensorDatasetBatch;
  batches<const TSample, const TBatch>(dataset: Dataset<TSample>, options: DataCollateOptions<TSample, TBatch>): CollatedDataLoader<TBatch>;
  batches<const TSample, const TBatch extends TensorDatasetBatch>(dataset: Dataset<TSample, TBatch>, options?: DataBatchOptions<TSample, TBatch>): DataBatches<TBatch["input"], TBatch["target"]>;
  batches<const TSample, const TBatch extends TensorDatasetBatch>(dataset: Dataset<TSample>, options: DataBatchOptions<TSample, TBatch>): DataBatches<TBatch["input"], TBatch["target"]>;
  batches<const SI, const ST, const BI, const BT>(dataset: TensorDataset<SI, ST, BI, BT>, options?: DataBatchOptions<TensorDatasetSample<SI, ST>, TensorDatasetBatch<BI, BT>>): DataBatches<BI, BT>;
  batches(dataset: Dataset, options?: DataBatchOptions): DataBatches;
  batch<const TSample, const TBatch>(dataset: Dataset<TSample>, options: DataCollateOptions<TSample, TBatch>): CollatedDataLoader<TBatch>;
  batch<const TSample, const TBatch extends TensorDatasetBatch>(dataset: Dataset<TSample, TBatch>, options?: DataBatchOptions<TSample, TBatch>): DataBatches<TBatch["input"], TBatch["target"]>;
  batch<const TSample, const TBatch extends TensorDatasetBatch>(dataset: Dataset<TSample>, options: DataBatchOptions<TSample, TBatch>): DataBatches<TBatch["input"], TBatch["target"]>;
  batch<const SI, const ST, const BI, const BT>(dataset: TensorDataset<SI, ST, BI, BT>, options?: DataBatchOptions<TensorDatasetSample<SI, ST>, TensorDatasetBatch<BI, BT>>): DataBatches<BI, BT>;
  batch(dataset: Dataset, options?: DataBatchOptions): DataBatches;
  dataLoader<const TSample, const TBatch>(dataset: Dataset<TSample>, options: DataCollateOptions<TSample, TBatch>): CollatedDataLoader<TBatch>;
  dataLoader<const TSample, const TBatch extends TensorDatasetBatch>(dataset: Dataset<TSample, TBatch>, options?: DataBatchOptions<TSample, TBatch>): DataLoader<TBatch["input"], TBatch["target"]>;
  dataLoader<const TSample, const TBatch extends TensorDatasetBatch>(dataset: Dataset<TSample>, options: DataBatchOptions<TSample, TBatch>): DataLoader<TBatch["input"], TBatch["target"]>;
  dataLoader<const SI, const ST, const BI, const BT>(dataset: TensorDataset<SI, ST, BI, BT>, options?: DataBatchOptions<TensorDatasetSample<SI, ST>, TensorDatasetBatch<BI, BT>>): DataLoader<BI, BT>;
  dataLoader(dataset: Dataset, options?: DataBatchOptions): DataLoader;
  dataloader<const TSample, const TBatch>(dataset: Dataset<TSample>, options: DataCollateOptions<TSample, TBatch>): CollatedDataLoader<TBatch>;
  dataloader<const TSample, const TBatch extends TensorDatasetBatch>(dataset: Dataset<TSample, TBatch>, options?: DataBatchOptions<TSample, TBatch>): DataLoader<TBatch["input"], TBatch["target"]>;
  dataloader<const TSample, const TBatch extends TensorDatasetBatch>(dataset: Dataset<TSample>, options: DataBatchOptions<TSample, TBatch>): DataLoader<TBatch["input"], TBatch["target"]>;
  dataloader<const SI, const ST, const BI, const BT>(dataset: TensorDataset<SI, ST, BI, BT>, options?: DataBatchOptions<TensorDatasetSample<SI, ST>, TensorDatasetBatch<BI, BT>>): DataLoader<BI, BT>;
  dataloader(dataset: Dataset, options?: DataBatchOptions): DataLoader;
  subset<const SI, const ST, const BI, const BT>(dataset: TensorDataset<SI, ST, BI, BT>, indices: Iterable<number>): TensorDataset<SI, ST, BI, BT>;
  subset(dataset: TensorDataset, indices: Iterable<number>): TensorDataset;
  take<const SI, const ST, const BI, const BT>(dataset: TensorDataset<SI, ST, BI, BT>, count: number): TensorDataset<SI, ST, BI, BT>;
  take(dataset: TensorDataset, count: number): TensorDataset;
  concatDataset<const SI, const ST, const BI, const BT>(
    datasets: readonly [TensorDataset<SI, ST, BI, BT>, ...TensorDataset<SI, ST, BI, BT>[]],
  ): TensorDataset<SI, ST, BI, BT>;
  concatDataset(datasets: Iterable<TensorDataset>): TensorDataset;
  concat_dataset<const SI, const ST, const BI, const BT>(
    datasets: readonly [TensorDataset<SI, ST, BI, BT>, ...TensorDataset<SI, ST, BI, BT>[]],
  ): TensorDataset<SI, ST, BI, BT>;
  concat_dataset(datasets: Iterable<TensorDataset>): TensorDataset;
  mapDataset<const SI, const ST, const MI, const MT>(
    dataset: TensorDataset<SI, ST>,
    mapper: TensorDatasetMapper<SI, ST, MI, MT>,
  ): TensorDataset<MI, MT>;
  mapDataset(dataset: TensorDataset, mapper: TensorDatasetMapper): TensorDataset;
  map_dataset<const SI, const ST, const MI, const MT>(
    dataset: TensorDataset<SI, ST>,
    mapper: TensorDatasetMapper<SI, ST, MI, MT>,
  ): TensorDataset<MI, MT>;
  map_dataset(dataset: TensorDataset, mapper: TensorDatasetMapper): TensorDataset;
  randomSplit<const SI, const ST, const BI, const BT>(
    dataset: TensorDataset<SI, ST, BI, BT>,
    lengths: readonly [number, number],
    options?: DataSplitOptions,
  ): readonly [TensorDataset<SI, ST, BI, BT>, TensorDataset<SI, ST, BI, BT>];
  randomSplit<const SI, const ST, const BI, const BT>(
    dataset: TensorDataset<SI, ST, BI, BT>,
    lengths: readonly number[],
    options?: DataSplitOptions,
  ): readonly TensorDataset<SI, ST, BI, BT>[];
  randomSplit<const SI, const ST, const BI, const BT>(
    dataset: TensorDataset<SI, ST, BI, BT>,
    lengths: Iterable<number>,
    options?: DataSplitOptions,
  ): readonly TensorDataset<SI, ST, BI, BT>[];
  randomSplit(dataset: TensorDataset, lengths: Iterable<number>, options?: DataSplitOptions): readonly TensorDataset[];
  random_split<const SI, const ST, const BI, const BT>(
    dataset: TensorDataset<SI, ST, BI, BT>,
    lengths: readonly [number, number],
    options?: DataSplitOptions,
  ): readonly [TensorDataset<SI, ST, BI, BT>, TensorDataset<SI, ST, BI, BT>];
  random_split<const SI, const ST, const BI, const BT>(
    dataset: TensorDataset<SI, ST, BI, BT>,
    lengths: readonly number[],
    options?: DataSplitOptions,
  ): readonly TensorDataset<SI, ST, BI, BT>[];
  random_split<const SI, const ST, const BI, const BT>(
    dataset: TensorDataset<SI, ST, BI, BT>,
    lengths: Iterable<number>,
    options?: DataSplitOptions,
  ): readonly TensorDataset<SI, ST, BI, BT>[];
  random_split(dataset: TensorDataset, lengths: Iterable<number>, options?: DataSplitOptions): readonly TensorDataset[];
}

export interface TrainNamespace {
  zeroGrad(target: OptimizerTarget, options?: ZeroGradOptions): void;
  gradNorm(target: OptimizerTarget): number;
  grad_norm(target: OptimizerTarget): number;
  clipGradNorm(target: OptimizerTarget, maxNorm: number, options?: { eps?: number }): number;
  clip_grad_norm_(target: OptimizerTarget, maxNorm: number, options?: { eps?: number }): number;
  clipGradValue<T extends OptimizerTarget>(target: T, clipValue: number): T;
  clip_grad_value_<T extends OptimizerTarget>(target: T, clipValue: number): T;
  accuracy(logits: TensorLike, targets: IndexLike, options?: { classes?: number; numClasses?: number }): number;
  classificationAccuracy(logits: TensorLike, targets: IndexLike, options?: { classes?: number; numClasses?: number }): number;
  classification_accuracy(logits: TensorLike, targets: IndexLike, options?: { classes?: number; numClasses?: number }): number;
  classPredictions(logits: TensorLike, options?: { classes?: number; numClasses?: number }): Uint32Array;
  class_predictions(logits: TensorLike, options?: { classes?: number; numClasses?: number }): Uint32Array;
  predictClasses(logits: TensorLike, options?: { classes?: number; numClasses?: number }): Uint32Array;
  predict_classes(logits: TensorLike, options?: { classes?: number; numClasses?: number }): Uint32Array;
  topKAccuracy(logits: TensorLike, targets: IndexLike, options?: { k?: number; topK?: number; top_k?: number; classes?: number; numClasses?: number }): number;
  top_k_accuracy(logits: TensorLike, targets: IndexLike, options?: { k?: number; topK?: number; top_k?: number; classes?: number; numClasses?: number }): number;
  confusionMatrix(logits: TensorLike, targets: IndexLike, options?: { classes?: number; numClasses?: number }): readonly (readonly number[])[];
  confusion_matrix(logits: TensorLike, targets: IndexLike, options?: { classes?: number; numClasses?: number }): readonly (readonly number[])[];
  classificationReport(logits: TensorLike, targets: IndexLike, options?: { classes?: number; numClasses?: number }): TrainClassificationReport;
  classification_report(logits: TensorLike, targets: IndexLike, options?: { classes?: number; numClasses?: number }): TrainClassificationReport;
  binaryAccuracy(predictions: TensorLike, targets: TensorLike, options?: { threshold?: number }): number;
  binary_accuracy(predictions: TensorLike, targets: TensorLike, options?: { threshold?: number }): number;
  binaryLogitsAccuracy(logits: TensorLike, targets: TensorLike, options?: { threshold?: number }): number;
  binary_logits_accuracy(logits: TensorLike, targets: TensorLike, options?: { threshold?: number }): number;
  backward(loss: Tensor, gradient?: TensorLike): void;
  step<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, options?: TrainStepOptions & { inspect?: false; evidence?: false }): void;
  step<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, options: TrainStepOptions & ({ inspect: true } | { evidence: true })): TrainStepEvidence<Kind>;
  step(optimizer: { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options?: TrainStepOptions & { inspect?: false; evidence?: false }): void;
  step(optimizer: { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: TrainStepOptions & ({ inspect: true } | { evidence: true })): TrainStepEvidence;
  lossStep(optimizer: { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, lossFn: () => Tensor, options?: TrainLossStepOptions): Tensor;
  fit<const Kind extends OptimizerStateKind, Batch>(
    optimizer: Optimizer<Kind>,
    batches: Iterable<Batch>,
    lossFn: (batch: Batch, context: TrainFitContext) => Tensor,
    options?: TrainFitOptions<Kind>,
  ): TrainFitEvidence<Kind>;
  fit<const Kind extends OptimizerStateKind, Target extends NnModule, Batch>(
    optimizer: Optimizer<Kind>,
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    criterion: TrainSupervisedCriterion<Target, Batch>,
    options?: TrainFitOptions<Kind>,
  ): TrainFitEvidence<Kind>;
  fit<Batch extends { input: TensorLike; target: TensorLike | IndexLike }>(
    compiled: CompiledTrainingStep,
    batches: Iterable<Batch>,
    options?: TrainFitOptions,
  ): TrainFitEvidence;
  fit<const Kind extends OptimizerStateKind, Target extends NnModule, Batch>(
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    options: TrainModelFitOptions<Kind, Target, Batch>,
  ): TrainFitEvidence<Kind>;
  fit<Batch>(
    optimizer: { step(): void; zeroGrad?(options?: ZeroGradOptions): void },
    batches: Iterable<Batch>,
    lossFn: (batch: Batch, context: TrainFitContext) => Tensor,
    options?: TrainFitOptions,
  ): TrainFitEvidence;
  fit<Target extends NnModule, Batch>(
    optimizer: { step(): void; zeroGrad?(options?: ZeroGradOptions): void },
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    criterion: TrainSupervisedCriterion<Target, Batch>,
    options?: TrainFitOptions,
  ): TrainFitEvidence;
  fit<Target extends NnModule, Batch>(
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    options: TrainModelFitOptions<null, Target, Batch>,
  ): TrainFitEvidence;
  explainNative<Batch extends { input: TensorLike; target: TensorLike | IndexLike }>(
    compiled: CompiledTrainingStep,
    batches: Iterable<Batch>,
    options?: TrainFitOptions,
  ): NativeTrainingExplanation;
  explainNative<const Kind extends OptimizerStateKind, Target extends NnModule, Batch>(
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    options: TrainModelFitOptions<Kind, Target, Batch>,
  ): NativeTrainingExplanation;
  explainNative<Target extends NnModule, Batch>(
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    options: TrainModelFitOptions<null, Target, Batch>,
  ): NativeTrainingExplanation;
  explainNative<const Kind extends OptimizerStateKind, Target extends NnModule, Batch>(
    optimizer: Optimizer<Kind>,
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    criterion: TrainSupervisedCriterion<Target, Batch>,
    options?: TrainFitOptions<Kind>,
  ): NativeTrainingExplanation;
  explainNative<Target extends NnModule, Batch>(
    optimizer: { step(): void; zeroGrad?(options?: ZeroGradOptions): void },
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    criterion: TrainSupervisedCriterion<Target, Batch>,
    options?: TrainFitOptions,
  ): NativeTrainingExplanation;
  explain_native: TrainNamespace["explainNative"];
  nativePlan: TrainNamespace["explainNative"];
  native_plan: TrainNamespace["explainNative"];
  fitNative<Batch extends { input: TensorLike; target: TensorLike | IndexLike }>(
    compiled: CompiledTrainingStep,
    batches: Iterable<Batch>,
    options?: TrainFitOptions & { requireNative?: true; require_native?: true },
  ): TrainFitEvidence;
  fitNative<const Kind extends OptimizerStateKind, Target extends NnModule, Batch>(
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    options: TrainModelFitOptions<Kind, Target, Batch> & { requireNative?: true; require_native?: true },
  ): TrainFitEvidence<Kind>;
  fitNative<Target extends NnModule, Batch>(
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    options: TrainModelFitOptions<null, Target, Batch> & { requireNative?: true; require_native?: true },
  ): TrainFitEvidence;
  fitNative<const Kind extends OptimizerStateKind, Target extends NnModule, Batch>(
    optimizer: Optimizer<Kind>,
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    criterion: TrainSupervisedCriterion<Target, Batch>,
    options?: TrainFitOptions<Kind> & { requireNative?: true; require_native?: true },
  ): TrainFitEvidence<Kind>;
  fitNative<Target extends NnModule, Batch>(
    optimizer: { step(): void; zeroGrad?(options?: ZeroGradOptions): void },
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    criterion: TrainSupervisedCriterion<Target, Batch>,
    options?: TrainFitOptions & { requireNative?: true; require_native?: true },
  ): TrainFitEvidence;
  fit_native: TrainNamespace["fitNative"];
  fitModule<const Kind extends OptimizerStateKind, Target extends NnModule, Batch>(
    optimizer: Optimizer<Kind>,
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    criterion: TrainSupervisedCriterion<Target, Batch>,
    options?: TrainFitOptions<Kind>,
  ): TrainFitEvidence<Kind>;
  fitModule<Target extends NnModule, Batch>(
    optimizer: { step(): void; zeroGrad?(options?: ZeroGradOptions): void },
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    criterion: TrainSupervisedCriterion<Target, Batch>,
    options?: TrainFitOptions,
  ): TrainFitEvidence;
  fit_module<const Kind extends OptimizerStateKind, Target extends NnModule, Batch>(
    optimizer: Optimizer<Kind>,
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    criterion: TrainSupervisedCriterion<Target, Batch>,
    options?: TrainFitOptions<Kind>,
  ): TrainFitEvidence<Kind>;
  fit_module<Target extends NnModule, Batch>(
    optimizer: { step(): void; zeroGrad?(options?: ZeroGradOptions): void },
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    criterion: TrainSupervisedCriterion<Target, Batch>,
    options?: TrainFitOptions,
  ): TrainFitEvidence;
  fitClassifier<const Kind extends OptimizerStateKind, Target extends NnModule, Batch>(
    optimizer: Optimizer<Kind>,
    module: Target,
    batches: Iterable<TrainClassificationBatch<Target, Batch>>,
    criterion: TrainClassificationCriterion<Target, Batch>,
    options?: TrainFitOptions<Kind>,
  ): TrainFitEvidence<Kind>;
  fitClassifier<Target extends NnModule, Batch>(
    optimizer: { step(): void; zeroGrad?(options?: ZeroGradOptions): void },
    module: Target,
    batches: Iterable<TrainClassificationBatch<Target, Batch>>,
    criterion: TrainClassificationCriterion<Target, Batch>,
    options?: TrainFitOptions,
  ): TrainFitEvidence;
  fit_classifier<const Kind extends OptimizerStateKind, Target extends NnModule, Batch>(
    optimizer: Optimizer<Kind>,
    module: Target,
    batches: Iterable<TrainClassificationBatch<Target, Batch>>,
    criterion: TrainClassificationCriterion<Target, Batch>,
    options?: TrainFitOptions<Kind>,
  ): TrainFitEvidence<Kind>;
  fit_classifier<Target extends NnModule, Batch>(
    optimizer: { step(): void; zeroGrad?(options?: ZeroGradOptions): void },
    module: Target,
    batches: Iterable<TrainClassificationBatch<Target, Batch>>,
    criterion: TrainClassificationCriterion<Target, Batch>,
    options?: TrainFitOptions,
  ): TrainFitEvidence;
  evaluate<Batch>(
    batches: Iterable<Batch>,
    lossFn: (batch: Batch, context: TrainEvaluateContext) => Tensor,
    options?: TrainEvaluateOptions,
  ): TrainEvaluateEvidence;
  evaluate_loss<Batch>(
    batches: Iterable<Batch>,
    lossFn: (batch: Batch, context: TrainEvaluateContext) => Tensor,
    options?: TrainEvaluateOptions,
  ): TrainEvaluateEvidence;
  evaluateModule<Target extends NnModule, Batch>(
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    criterion: TrainSupervisedCriterion<Target, Batch>,
    options?: TrainEvaluateOptions,
  ): TrainEvaluateEvidence;
  evaluate_module<Target extends NnModule, Batch>(
    module: Target,
    batches: Iterable<TrainSupervisedBatch<Target, Batch>>,
    criterion: TrainSupervisedCriterion<Target, Batch>,
    options?: TrainEvaluateOptions,
  ): TrainEvaluateEvidence;
  evaluateClassifier<Target extends NnModule, Batch>(
    module: Target,
    batches: Iterable<TrainClassificationBatch<Target, Batch>>,
    criterion: TrainClassificationCriterion<Target, Batch>,
    options?: TrainEvaluateOptions,
  ): TrainEvaluateEvidence;
  evaluate_classifier<Target extends NnModule, Batch>(
    module: Target,
    batches: Iterable<TrainClassificationBatch<Target, Batch>>,
    criterion: TrainClassificationCriterion<Target, Batch>,
    options?: TrainEvaluateOptions,
  ): TrainEvaluateEvidence;
  predict<Batch, TOutput extends Tensor = Tensor>(
    batches: Iterable<Batch>,
    predictFn: (batch: Batch, context: TrainPredictContext) => TOutput,
    options?: TrainPredictOptions<TOutput>,
  ): TrainPredictEvidence<TOutput>;
  predict_batches<Batch, TOutput extends Tensor = Tensor>(
    batches: Iterable<Batch>,
    predictFn: (batch: Batch, context: TrainPredictContext) => TOutput,
    options?: TrainPredictOptions<TOutput>,
  ): TrainPredictEvidence<TOutput>;
  predictModule<Target extends NnModule, Batch>(
    module: Target,
    batches: Iterable<Batch>,
    options?: TrainPredictOptions<Tensor<TrainModuleOutputShape<Target, Batch>>>,
  ): TrainPredictEvidence<Tensor<TrainModuleOutputShape<Target, Batch>>>;
  predict_module<Target extends NnModule, Batch>(
    module: Target,
    batches: Iterable<Batch>,
    options?: TrainPredictOptions<Tensor<TrainModuleOutputShape<Target, Batch>>>,
  ): TrainPredictEvidence<Tensor<TrainModuleOutputShape<Target, Batch>>>;
  predictClassifier<Target extends NnModule, Batch>(
    module: Target,
    batches: Iterable<Batch>,
    options?: TrainPredictOptions<Tensor<TrainModuleOutputShape<Target, Batch>>>,
  ): TrainPredictEvidence<Tensor<TrainModuleOutputShape<Target, Batch>>>;
  predict_classifier<Target extends NnModule, Batch>(
    module: Target,
    batches: Iterable<Batch>,
    options?: TrainPredictOptions<Tensor<TrainModuleOutputShape<Target, Batch>>>,
  ): TrainPredictEvidence<Tensor<TrainModuleOutputShape<Target, Batch>>>;
  inspectStep(): TrainStepEvidence | null;
  isTrainStepEvidence(evidence: unknown): evidence is TrainStepEvidence;
  requireTrainStepEvidence(evidence: unknown): TrainStepEvidence;
  assertTrainStepEvidence(evidence: unknown): TrainStepEvidence;
  assert_train_step_evidence(evidence: unknown): TrainStepEvidence;
  matchesTrainStepEvidenceSignature(evidence: unknown, signature: unknown): boolean;
  matches_train_step_evidence_signature(evidence: unknown, signature: unknown): boolean;
  isTrainFitStepEvidence(evidence: unknown): evidence is TrainFitStepEvidence;
  requireTrainFitStepEvidence(evidence: unknown): TrainFitStepEvidence;
  assertTrainFitStepEvidence(evidence: unknown): TrainFitStepEvidence;
  assert_train_fit_step_evidence(evidence: unknown): TrainFitStepEvidence;
  matchesTrainFitStepEvidenceSignature(evidence: unknown, signature: unknown): boolean;
  matches_train_fit_step_evidence_signature(evidence: unknown, signature: unknown): boolean;
  isTrainFitEvidence(evidence: unknown): evidence is TrainFitEvidence;
  requireTrainFitEvidence(evidence: unknown): TrainFitEvidence;
  assertTrainFitEvidence(evidence: unknown): TrainFitEvidence;
  assert_train_fit_evidence(evidence: unknown): TrainFitEvidence;
  matchesTrainFitEvidenceSignature(evidence: unknown, signature: unknown): boolean;
  matches_train_fit_evidence_signature(evidence: unknown, signature: unknown): boolean;
  isTrainEvaluateStepEvidence(evidence: unknown): evidence is TrainEvaluateStepEvidence;
  requireTrainEvaluateStepEvidence(evidence: unknown): TrainEvaluateStepEvidence;
  assertTrainEvaluateStepEvidence(evidence: unknown): TrainEvaluateStepEvidence;
  assert_train_evaluate_step_evidence(evidence: unknown): TrainEvaluateStepEvidence;
  matchesTrainEvaluateStepEvidenceSignature(evidence: unknown, signature: unknown): boolean;
  matches_train_evaluate_step_evidence_signature(evidence: unknown, signature: unknown): boolean;
  isTrainEvaluateEvidence(evidence: unknown): evidence is TrainEvaluateEvidence;
  requireTrainEvaluateEvidence(evidence: unknown): TrainEvaluateEvidence;
  assertTrainEvaluateEvidence(evidence: unknown): TrainEvaluateEvidence;
  assert_train_evaluate_evidence(evidence: unknown): TrainEvaluateEvidence;
  matchesTrainEvaluateEvidenceSignature(evidence: unknown, signature: unknown): boolean;
  matches_train_evaluate_evidence_signature(evidence: unknown, signature: unknown): boolean;
  isTrainPredictStepEvidence(evidence: unknown): evidence is TrainPredictStepEvidence;
  requireTrainPredictStepEvidence(evidence: unknown): TrainPredictStepEvidence;
  assertTrainPredictStepEvidence(evidence: unknown): TrainPredictStepEvidence;
  assert_train_predict_step_evidence(evidence: unknown): TrainPredictStepEvidence;
  matchesTrainPredictStepEvidenceSignature(evidence: unknown, signature: unknown): boolean;
  matches_train_predict_step_evidence_signature(evidence: unknown, signature: unknown): boolean;
  isTrainPredictEvidence(evidence: unknown): evidence is TrainPredictEvidence;
  requireTrainPredictEvidence(evidence: unknown): TrainPredictEvidence;
  assertTrainPredictEvidence(evidence: unknown): TrainPredictEvidence;
  assert_train_predict_evidence(evidence: unknown): TrainPredictEvidence;
  matchesTrainPredictEvidenceSignature(evidence: unknown, signature: unknown): boolean;
  matches_train_predict_evidence_signature(evidence: unknown, signature: unknown): boolean;
}

export type LazyLinearShape<InputShape extends TensorShapeTuple, OutFeatures extends number> =
  InputShape extends readonly [number]
    ? readonly [OutFeatures]
    : InputShape extends readonly [infer Batch extends number, number]
      ? readonly [Batch, OutFeatures]
      : TensorShapeTuple;
export type LazyEmbeddingShape<InputShape extends TensorShapeTuple, EmbeddingDim extends number> =
  readonly [...InputShape, EmbeddingDim];
export type LazyMatmulShape<InputShape extends TensorShapeTuple, WeightShape extends TensorShapeTuple> =
  InputShape extends readonly [infer InFeatures extends number]
    ? WeightShape extends readonly [InFeatures, infer OutFeatures extends number]
      ? readonly [OutFeatures]
      : TensorShapeTuple
    : InputShape extends readonly [infer Batch extends number, infer InFeatures extends number]
      ? WeightShape extends readonly [InFeatures, infer OutFeatures extends number]
        ? readonly [Batch, OutFeatures]
        : TensorShapeTuple
      : TensorShapeTuple;
export type LazyLinearOptions = Readonly<{
  name?: string;
  bias?: boolean;
  weightLayout?: string | null;
  biasLayout?: string | null;
}>;
export type LazyFlattenOptions = Readonly<{
  startDim?: number;
  endDim?: number;
}>;
export type LazyDropoutOptions = Readonly<{
  training?: boolean;
}>;
export type LazyNormOptions = Readonly<{
  eps?: number;
  affine?: boolean;
  bias?: boolean;
}>;
export type LazyConv2dOptions = Readonly<{
  name?: string;
  stride?: number | readonly [number, number];
  padding?: number | readonly [number, number];
  dilation?: number | readonly [number, number];
  groups?: number;
  bias?: boolean;
}>;
export type LazyPool2dOptions = Readonly<{
  kernelSize?: number | readonly [number, number];
  stride?: number | readonly [number, number];
  padding?: number | readonly [number, number];
  dilation?: number | readonly [number, number];
  ceilMode?: boolean;
  countIncludePad?: boolean;
}>;
export type LazyModuleTraceOptions<Shape extends TensorShapeTuple = TensorShapeTuple> = Readonly<{
  inputShape: Shape;
  name?: string;
}>;
export type LazyCompilerArtifacts = Readonly<{
  trace: ModuleProgramTrace;
  ir: ModuleTensorProgramIr | null;
  kernelPlan: ModuleKernelPlan | null;
  diagnostic: ModuleCompileDiagnostic | null;
}>;
export type LazyCompileSupport = Readonly<{
  supported: boolean;
  reason: string | null;
  nativePath: "device-program";
  trace: ModuleProgramTrace;
  ir: ModuleTensorProgramIr | null;
  kernelPlan: ModuleKernelPlan | null;
  diagnostic: ModuleCompileDiagnostic | null;
}>;
export interface LazyTensor<Shape extends TensorShapeTuple = TensorShapeTuple> {
  readonly shape: Shape;
  readonly name: string | null;
  linear<const OutFeatures extends number>(outFeatures: OutFeatures, options?: LazyLinearOptions): LazyTensor<LazyLinearShape<Shape, OutFeatures>>;
  matmul<const WeightShape extends TensorShapeTuple>(weight: LazyTensor<WeightShape>): LazyTensor<LazyMatmulShape<Shape, WeightShape>>;
  mm<const WeightShape extends TensorShapeTuple>(weight: LazyTensor<WeightShape>): LazyTensor<LazyMatmulShape<Shape, WeightShape>>;
  add<const BiasShape extends TensorShapeTuple>(bias: LazyTensor<BiasShape>): LazyTensor<Shape>;
  mul<const ScaleShape extends TensorShapeTuple>(scale: LazyTensor<ScaleShape>): LazyTensor<Shape>;
  affine<const ScaleShape extends TensorShapeTuple, const BiasShape extends TensorShapeTuple>(scale: LazyTensor<ScaleShape>, bias: LazyTensor<BiasShape>): LazyTensor<Shape>;
  embedding<const EmbeddingDim extends number>(numEmbeddings: number, embeddingDim: EmbeddingDim, options?: Readonly<{ name?: string }>): LazyTensor<LazyEmbeddingShape<Shape, EmbeddingDim>>;
  layerNorm(features: number, options?: LazyNormOptions): LazyTensor<Shape>;
  layer_norm(features: number, options?: LazyNormOptions): LazyTensor<Shape>;
  rmsNorm(features: number, options?: LazyNormOptions): LazyTensor<Shape>;
  rms_norm(features: number, options?: LazyNormOptions): LazyTensor<Shape>;
  conv2d(outChannels: number, kernelSize: number | readonly [number, number], options?: LazyConv2dOptions): LazyTensor<TensorShapeTuple>;
  maxPool2d(kernelSize?: number | readonly [number, number], options?: LazyPool2dOptions): LazyTensor<TensorShapeTuple>;
  max_pool2d(kernelSize?: number | readonly [number, number], options?: LazyPool2dOptions): LazyTensor<TensorShapeTuple>;
  avgPool2d(kernelSize?: number | readonly [number, number], options?: LazyPool2dOptions): LazyTensor<TensorShapeTuple>;
  avg_pool2d(kernelSize?: number | readonly [number, number], options?: LazyPool2dOptions): LazyTensor<TensorShapeTuple>;
  relu(): LazyTensor<Shape>;
  gelu(): LazyTensor<Shape>;
  silu(): LazyTensor<Shape>;
  sigmoid(): LazyTensor<Shape>;
  tanh(): LazyTensor<Shape>;
  exp(): LazyTensor<Shape>;
  log(): LazyTensor<Shape>;
  neg(): LazyTensor<Shape>;
  recip(): LazyTensor<Shape>;
  abs(): LazyTensor<Shape>;
  sqrt(): LazyTensor<Shape>;
  square(): LazyTensor<Shape>;
  sgn(): LazyTensor<Shape>;
  step(): LazyTensor<Shape>;
  dropout(p?: number, options?: LazyDropoutOptions): LazyTensor<Shape>;
  softmax(dim: number): LazyTensor<Shape>;
  logSoftmax(dim: number): LazyTensor<Shape>;
  log_softmax(dim: number): LazyTensor<Shape>;
  reshape<const TargetShape extends TensorShapeTuple>(shape: TargetShape): LazyTensor<TargetShape>;
  view<const TargetShape extends TensorShapeTuple>(shape: TargetShape): LazyTensor<TargetShape>;
  flatten(options?: LazyFlattenOptions): LazyTensor<TensorShapeTuple>;
  squeeze(): LazyTensor<SqueezeShape<Shape>>;
  squeeze<const Dim extends number>(dim: Dim): LazyTensor<SqueezeShape<Shape, Dim>>;
  squeeze(dim?: number | null): LazyTensor<TensorShapeTuple>;
  unsqueeze<const Dim extends number>(dim: Dim): LazyTensor<UnsqueezeShape<Shape, Dim>>;
  broadcastTo<const TargetShape extends TensorShape>(shape: TargetShape): LazyTensor<TensorShapeOf<TargetShape>>;
  broadcast_to<const TargetShape extends TensorShape>(shape: TargetShape): LazyTensor<TensorShapeOf<TargetShape>>;
  expand<const TargetShape extends TensorShape>(shape: TargetShape): LazyTensor<TensorShapeOf<TargetShape>>;
  narrow<const Dim extends number, const Length extends number>(dim: Dim, start: number, length: Length): LazyTensor<NarrowShape<Shape, Dim, Length>>;
  narrow(dim: number, start: number, length: number): LazyTensor<TensorShapeTuple>;
  select<const Dim extends number>(dim: Dim, index: number): LazyTensor<SelectShape<Shape, Dim>>;
  select(dim: number, index: number): LazyTensor<TensorShapeTuple>;
  slice<const Dim extends number, const Start extends number, const End extends number>(dim: Dim, start: Start, end: End, step?: 1): LazyTensor<SliceShape<Shape, Dim, Start, End>>;
  slice(dim: number, start?: number | null, end?: number | null, step?: number): LazyTensor<TensorShapeTuple>;
  transpose(): LazyTensor<TransposeShape<Shape>>;
  transpose<const Dim0 extends number, const Dim1 extends number>(dim0: Dim0, dim1: Dim1): LazyTensor<TransposeShape<Shape, Dim0, Dim1>>;
  permute<const Dims extends readonly number[]>(dims: Dims): LazyTensor<PermuteShape<Shape, Dims>>;
  permute(dims: readonly number[]): LazyTensor<TensorShapeTuple>;
  sum<const Dim extends number>(dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>>;
  mean<const Dim extends number>(dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>>;
  max<const Dim extends number>(dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>>;
  min<const Dim extends number>(dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>>;
  apply<const Target extends LazyModuleTarget>(module: Target): LazyTensor<ModuleTargetForwardShape<Target, Shape>>;
  trace(): ModuleProgramTrace;
  tensorProgramIr(): ModuleTensorProgramIr | null;
  tensor_program_ir(): ModuleTensorProgramIr | null;
  kernelPlan(options?: CompileOptions): ModuleKernelPlan | null;
  kernel_plan(options?: CompileOptions): ModuleKernelPlan | null;
  artifacts(options?: CompileOptions): LazyCompilerArtifacts;
  compileSupport(options?: CompileOptions): LazyCompileSupport;
  compile_support(options?: CompileOptions): LazyCompileSupport;
  canCompile(options?: CompileOptions): boolean;
  can_compile(options?: CompileOptions): boolean;
  requireCompileSupport(options?: CompileOptions): LazyCompileSupport;
  require_compile_support(options?: CompileOptions): LazyCompileSupport;
  compile(options?: CompileOptions): Program<TensorShapeTuple, Shape>;
}
export type LazyModuleTarget = NnModule | readonly NnModule[];
export interface LazyNamespace {
  readonly lazyManifest: Readonly<{
    kind: "zgml-lazy-tensor-frontend";
    source: FrontendManifestContract["source"];
    policyOwner: "src/ts/lazy.ts";
    productSourceOfTruth: FrontendManifestContract["productSourceOfTruth"];
    productSemanticsOwner: FrontendManifestContract["productSemanticsOwner"];
    nativeProductPolicy: FrontendManifestContract["nativeProductPolicy"];
    packageFanout: FrontendManifestContract["packageFanout"];
    runtimePath: "LazyTensor -> Trace -> TensorProgramIr -> KernelPlan -> Program";
  }>;
  LazyTensor: new <const Shape extends TensorShapeTuple>(shape: Shape, options?: Readonly<{ name?: string | null }>) => LazyTensor<Shape>;
  input<const Shape extends TensorShapeTuple>(shape: Shape, name?: string): LazyTensor<Shape>;
  parameter<const Shape extends TensorShapeTuple>(shape: Shape, name?: string, layout?: string | null): LazyTensor<Shape>;
  linear<const Shape extends TensorShapeTuple, const OutFeatures extends number>(tensor: LazyTensor<Shape>, outFeatures: OutFeatures, options?: LazyLinearOptions): LazyTensor<LazyLinearShape<Shape, OutFeatures>>;
  matmul<const Shape extends TensorShapeTuple, const WeightShape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, weight: LazyTensor<WeightShape>): LazyTensor<LazyMatmulShape<Shape, WeightShape>>;
  mm<const Shape extends TensorShapeTuple, const WeightShape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, weight: LazyTensor<WeightShape>): LazyTensor<LazyMatmulShape<Shape, WeightShape>>;
  add<const Shape extends TensorShapeTuple, const BiasShape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, bias: LazyTensor<BiasShape>): LazyTensor<Shape>;
  mul<const Shape extends TensorShapeTuple, const ScaleShape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, scale: LazyTensor<ScaleShape>): LazyTensor<Shape>;
  affine<const Shape extends TensorShapeTuple, const ScaleShape extends TensorShapeTuple, const BiasShape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, scale: LazyTensor<ScaleShape>, bias: LazyTensor<BiasShape>): LazyTensor<Shape>;
  embedding<const Shape extends TensorShapeTuple, const EmbeddingDim extends number>(tensor: LazyTensor<Shape>, numEmbeddings: number, embeddingDim: EmbeddingDim, options?: Readonly<{ name?: string }>): LazyTensor<LazyEmbeddingShape<Shape, EmbeddingDim>>;
  layerNorm<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, features: number, options?: LazyNormOptions): LazyTensor<Shape>;
  layer_norm<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, features: number, options?: LazyNormOptions): LazyTensor<Shape>;
  rmsNorm<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, features: number, options?: LazyNormOptions): LazyTensor<Shape>;
  rms_norm<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, features: number, options?: LazyNormOptions): LazyTensor<Shape>;
  conv2d<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, outChannels: number, kernelSize: number | readonly [number, number], options?: LazyConv2dOptions): LazyTensor<TensorShapeTuple>;
  maxPool2d<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, kernelSize?: number | readonly [number, number], options?: LazyPool2dOptions): LazyTensor<TensorShapeTuple>;
  max_pool2d<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, kernelSize?: number | readonly [number, number], options?: LazyPool2dOptions): LazyTensor<TensorShapeTuple>;
  avgPool2d<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, kernelSize?: number | readonly [number, number], options?: LazyPool2dOptions): LazyTensor<TensorShapeTuple>;
  avg_pool2d<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, kernelSize?: number | readonly [number, number], options?: LazyPool2dOptions): LazyTensor<TensorShapeTuple>;
  relu<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape>;
  gelu<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape>;
  silu<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape>;
  sigmoid<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape>;
  tanh<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape>;
  exp<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape>;
  log<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape>;
  neg<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape>;
  recip<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape>;
  abs<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape>;
  sqrt<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape>;
  square<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape>;
  sgn<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape>;
  step<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<Shape>;
  dropout<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, p?: number, options?: LazyDropoutOptions): LazyTensor<Shape>;
  softmax<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim: number): LazyTensor<Shape>;
  logSoftmax<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim: number): LazyTensor<Shape>;
  log_softmax<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim: number): LazyTensor<Shape>;
  reshape<const Shape extends TensorShapeTuple, const TargetShape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, shape: TargetShape): LazyTensor<TargetShape>;
  view<const Shape extends TensorShapeTuple, const TargetShape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, shape: TargetShape): LazyTensor<TargetShape>;
  flatten<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, options?: LazyFlattenOptions): LazyTensor<TensorShapeTuple>;
  squeeze<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<SqueezeShape<Shape>>;
  squeeze<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, dim: Dim): LazyTensor<SqueezeShape<Shape, Dim>>;
  squeeze<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim?: number | null): LazyTensor<TensorShapeTuple>;
  unsqueeze<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, dim: Dim): LazyTensor<UnsqueezeShape<Shape, Dim>>;
  broadcastTo<const Shape extends TensorShapeTuple, const TargetShape extends TensorShape>(tensor: LazyTensor<Shape>, shape: TargetShape): LazyTensor<TensorShapeOf<TargetShape>>;
  broadcast_to<const Shape extends TensorShapeTuple, const TargetShape extends TensorShape>(tensor: LazyTensor<Shape>, shape: TargetShape): LazyTensor<TensorShapeOf<TargetShape>>;
  expand<const Shape extends TensorShapeTuple, const TargetShape extends TensorShape>(tensor: LazyTensor<Shape>, shape: TargetShape): LazyTensor<TensorShapeOf<TargetShape>>;
  narrow<const Shape extends TensorShapeTuple, const Dim extends number, const Length extends number>(tensor: LazyTensor<Shape>, dim: Dim, start: number, length: Length): LazyTensor<NarrowShape<Shape, Dim, Length>>;
  narrow<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim: number, start: number, length: number): LazyTensor<TensorShapeTuple>;
  select<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, dim: Dim, index: number): LazyTensor<SelectShape<Shape, Dim>>;
  select<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim: number, index: number): LazyTensor<TensorShapeTuple>;
  slice<const Shape extends TensorShapeTuple, const Dim extends number, const Start extends number, const End extends number>(tensor: LazyTensor<Shape>, dim: Dim, start: Start, end: End, step?: 1): LazyTensor<SliceShape<Shape, Dim, Start, End>>;
  slice<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dim: number, start?: number | null, end?: number | null, step?: number): LazyTensor<TensorShapeTuple>;
  transpose<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>): LazyTensor<TransposeShape<Shape>>;
  transpose<const Shape extends TensorShapeTuple, const Dim0 extends number, const Dim1 extends number>(tensor: LazyTensor<Shape>, dim0: Dim0, dim1: Dim1): LazyTensor<TransposeShape<Shape, Dim0, Dim1>>;
  permute<const Shape extends TensorShapeTuple, const Dims extends readonly number[]>(tensor: LazyTensor<Shape>, dims: Dims): LazyTensor<PermuteShape<Shape, Dims>>;
  permute<const Shape extends TensorShapeTuple>(tensor: LazyTensor<Shape>, dims: readonly number[]): LazyTensor<TensorShapeTuple>;
  sum<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>>;
  mean<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>>;
  max<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>>;
  min<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>>;
  reduction<const Shape extends TensorShapeTuple, const Dim extends number>(tensor: LazyTensor<Shape>, kind: ModuleReductionKind, dim: Dim): LazyTensor<LazyReductionShape<Shape, Dim>>;
  apply<const Target extends LazyModuleTarget, const Shape extends TensorShapeTuple>(target: Target, tensor: LazyTensor<Shape>): LazyTensor<ModuleTargetForwardShape<Target, Shape>>;
  fromModule<const Target extends LazyModuleTarget, const Shape extends TensorShapeTuple>(target: Target, options: LazyModuleTraceOptions<Shape>): LazyTensor<ModuleTargetForwardShape<Target, Shape>>;
  traceModule<const Shape extends TensorShapeTuple>(target: LazyModuleTarget, options: LazyModuleTraceOptions<Shape>): ModuleProgramTrace;
  moduleArtifacts<const Shape extends TensorShapeTuple>(target: LazyModuleTarget, options: LazyModuleTraceOptions<Shape>): LazyCompilerArtifacts;
  moduleCompileSupport<const Shape extends TensorShapeTuple>(target: LazyModuleTarget, options: LazyModuleTraceOptions<Shape>): LazyCompileSupport;
  trace(tensor: LazyTensor): ModuleProgramTrace;
  artifacts(tensor: LazyTensor, options?: CompileOptions): LazyCompilerArtifacts;
  tensorProgramIr(tensor: LazyTensor): ModuleTensorProgramIr | null;
  kernelPlan(tensor: LazyTensor, options?: CompileOptions): ModuleKernelPlan | null;
  compileSupport(tensor: LazyTensor, options?: CompileOptions): LazyCompileSupport;
  canCompile(tensor: LazyTensor, options?: CompileOptions): boolean;
  can_compile(tensor: LazyTensor, options?: CompileOptions): boolean;
  requireCompileSupport(tensor: LazyTensor, options?: CompileOptions): LazyCompileSupport;
  require_compile_support(tensor: LazyTensor, options?: CompileOptions): LazyCompileSupport;
}
export type PublicLazyNamespace = Readonly<LazyNamespace>;
export type PublicDataNamespace = Readonly<DataNamespace>;
export type PublicCompileNamespace = Readonly<CompileNamespace> & {
  <const Target extends EmbeddingModule, const S extends TensorShapeTuple>(target: Target, options: EmbeddingCompileOptions<S>): Program<S, ModuleForwardShape<Target, S>>;
  <const Target extends NnCompilableModule, const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): Program<S, ModuleForwardShape<Target, S>>;
  <const Shape extends TensorShapeTuple>(target: LazyTensor<Shape>, options?: CompileOptions): Program<TensorShapeTuple, Shape>;
  (target: NnCompilableModule, options?: CompileOptions): Program;
  compileForInference<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, ModuleTargetForwardShape<Target, S>>;
  compileForInference(target: NnModule | readonly NnModule[], options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  compile_for_inference<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, ModuleTargetForwardShape<Target, S>>;
  compile_for_inference(target: NnModule | readonly NnModule[], options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  native<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, ModuleTargetForwardShape<Target, S>>;
  native(target: NnModule | readonly NnModule[], options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  inference<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, ModuleTargetForwardShape<Target, S>>;
  inference(target: NnModule | readonly NnModule[], options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  forInference<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, ModuleTargetForwardShape<Target, S>>;
  forInference(target: NnModule | readonly NnModule[], options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  for_inference<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, ModuleTargetForwardShape<Target, S>>;
  for_inference(target: NnModule | readonly NnModule[], options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  run<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, input: ProgramInputBinding<S>, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): Tensor<ModuleTargetForwardShape<Target, S>>;
  run(target: NnModule | readonly NnModule[], input: ProgramInputBinding, options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): Tensor;
  infer<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, input: ProgramInputBinding<S>, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): Tensor<ModuleTargetForwardShape<Target, S>>;
  infer(target: NnModule | readonly NnModule[], input: ProgramInputBinding, options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): Tensor;
  predict<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, input: ProgramInputBinding<S>, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): Tensor<ModuleTargetForwardShape<Target, S>>;
  predict(target: NnModule | readonly NnModule[], input: ProgramInputBinding, options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): Tensor;
  runInto<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(output: Float32Array, target: Target, input: ProgramInputBinding<S>, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): Float32Array;
  runInto(output: Float32Array, target: NnModule | readonly NnModule[], input: ProgramInputBinding, options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): Float32Array;
  run_into<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(output: Float32Array, target: Target, input: ProgramInputBinding<S>, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): Float32Array;
  run_into(output: Float32Array, target: NnModule | readonly NnModule[], input: ProgramInputBinding, options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): Float32Array;
  inferInto<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(output: Float32Array, target: Target, input: ProgramInputBinding<S>, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): Float32Array;
  inferInto(output: Float32Array, target: NnModule | readonly NnModule[], input: ProgramInputBinding, options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): Float32Array;
  infer_into<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(output: Float32Array, target: Target, input: ProgramInputBinding<S>, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): Float32Array;
  infer_into(output: Float32Array, target: NnModule | readonly NnModule[], input: ProgramInputBinding, options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): Float32Array;
  predictInto<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(output: Float32Array, target: Target, input: ProgramInputBinding<S>, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): Float32Array;
  predictInto(output: Float32Array, target: NnModule | readonly NnModule[], input: ProgramInputBinding, options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): Float32Array;
  predict_into<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(output: Float32Array, target: Target, input: ProgramInputBinding<S>, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): Float32Array;
  predict_into(output: Float32Array, target: NnModule | readonly NnModule[], input: ProgramInputBinding, options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): Float32Array;
  trainingStep(target: NnModule, optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  training_step(target: NnModule, optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  compileForTraining(target: NnModule, optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  compile_for_training(target: NnModule, optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  forTraining(target: NnModule, optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
  for_training(target: NnModule, optimizer: Optimizer | { step(): void; zeroGrad?(options?: ZeroGradOptions): void }, options: CompileTrainingOptions): CompiledTrainingStep;
};
export type PublicProgramNamespace = Readonly<ProgramNamespace>;
export type PublicSessionNamespace = Readonly<SessionNamespace>;
export declare const data: PublicDataNamespace;
export declare const compile: PublicCompileNamespace;
export declare const lazy: PublicLazyNamespace;
export declare const inspection: InspectionNamespace;
export declare const program: PublicProgramNamespace;
export declare const session: PublicSessionNamespace;
export declare const stepParams: StepParamsNamespace;
export declare const nativeApiContract: NativeApiContractNamespace;
export declare const F: NnFunctionalNamespace;
export type NnLossConstructorName =
  | "MSELoss"
  | "L1Loss"
  | "HuberLoss"
  | "SmoothL1Loss"
  | "BCELoss"
  | "BCEWithLogitsLoss"
  | "CrossEntropyLoss"
  | "NLLLoss";
export type NnLossConstructors = Pick<LossNamespace, NnLossConstructorName>;
export type NnFunctionalNamespace = Omit<LossNamespace, "mse_loss" | "l1_loss" | "huber_loss" | "smooth_l1_loss"> & {
  mse_loss(pred: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  mse_loss(pred: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  mse_loss(pred: TensorData, target: TensorData, options?: LossReductionOptions): number;
  l1_loss(pred: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  l1_loss(pred: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  l1_loss(pred: TensorData, target: TensorData, options?: LossReductionOptions): number;
  huber_loss(pred: Tensor, target: TensorLike, options?: HuberLossOptions): Tensor<readonly [1]>;
  huber_loss(pred: TensorData, target: Tensor, options?: HuberLossOptions): Tensor<readonly [1]>;
  huber_loss(pred: TensorData, target: TensorData, options?: HuberLossOptions): number;
  smooth_l1_loss(pred: Tensor, target: TensorLike, options?: SmoothL1LossOptions): Tensor<readonly [1]>;
  smooth_l1_loss(pred: TensorData, target: Tensor, options?: SmoothL1LossOptions): Tensor<readonly [1]>;
  smooth_l1_loss(pred: TensorData, target: TensorData, options?: SmoothL1LossOptions): number;
  relu<const S extends TensorShapeTuple>(input: Tensor<S>, inplace?: false): Tensor<S>;
  relu(input: TensorLike, inplace?: false): Tensor;
  gelu<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  gelu(input: TensorLike): Tensor;
  silu<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  silu(input: TensorLike): Tensor;
  sigmoid<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  sigmoid(input: TensorLike): Tensor;
  tanh<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<S>;
  tanh(input: TensorLike): Tensor;
  softmax<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
  softmax(input: TensorLike, dim?: number): Tensor;
  softmaxDim<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
  softmaxDim(input: TensorLike, dim?: number): Tensor;
  softmax_dim<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
  softmax_dim(input: TensorLike, dim?: number): Tensor;
  logSoftmax<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
  logSoftmax(input: TensorLike, dim?: number): Tensor;
  log_softmax<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
  log_softmax(input: TensorLike, dim?: number): Tensor;
  logSoftmaxDim<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
  logSoftmaxDim(input: TensorLike, dim?: number): Tensor;
  log_softmax_dim<const S extends TensorShapeTuple>(input: Tensor<S>, dim?: number): Tensor<S>;
  log_softmax_dim(input: TensorLike, dim?: number): Tensor;
  dropout<const S extends TensorShapeTuple>(input: Tensor<S>, p?: number, config?: NnDropoutConfig | boolean, inplace?: false): Tensor<S>;
  dropout(input: TensorLike, p?: number, config?: NnDropoutConfig | boolean, inplace?: false): Tensor;
  flatten<const S extends TensorShapeTuple>(input: Tensor<S>): Tensor<FlattenShape<S>>;
  flatten<const S extends TensorShapeTuple, const StartDim extends number, const EndDim extends number>(
    input: Tensor<S>,
    startDim: StartDim,
    endDim: EndDim,
  ): Tensor<FlattenShape<S, StartDim, EndDim>>;
  flatten(input: TensorLike, startDim?: number, endDim?: number): Tensor;
  linear<const InFeatures extends number, const OutFeatures extends number>(
    input: Tensor<readonly [InFeatures]>,
    weight: Tensor<readonly [OutFeatures, InFeatures]>,
    bias?: TensorLike<readonly [OutFeatures]> | null,
  ): Tensor<readonly [OutFeatures]>;
  linear<const Batch extends number, const InFeatures extends number, const OutFeatures extends number>(
    input: Tensor<readonly [Batch, InFeatures]>,
    weight: Tensor<readonly [OutFeatures, InFeatures]>,
    bias?: TensorLike<readonly [OutFeatures]> | null,
  ): Tensor<readonly [Batch, OutFeatures]>;
  linear<const Batch extends number, const Steps extends number, const InFeatures extends number, const OutFeatures extends number>(
    input: Tensor<readonly [Batch, Steps, InFeatures]>,
    weight: Tensor<readonly [OutFeatures, InFeatures]>,
    bias?: TensorLike<readonly [OutFeatures]> | null,
  ): Tensor<readonly [Batch, Steps, OutFeatures]>;
  linear<const A extends number, const B extends number, const C extends number, const InFeatures extends number, const OutFeatures extends number>(
    input: Tensor<readonly [A, B, C, InFeatures]>,
    weight: Tensor<readonly [OutFeatures, InFeatures]>,
    bias?: TensorLike<readonly [OutFeatures]> | null,
  ): Tensor<readonly [A, B, C, OutFeatures]>;
  linear(input: TensorLike, weight: TensorLike, bias?: TensorLike | null): Tensor;
  conv2d<const InChannels extends number, const OutChannels extends number, const Kernel extends number>(
    input: Tensor<readonly [InChannels, number, number]>,
    weight: Tensor<readonly [OutChannels, InChannels, Kernel, Kernel]>,
    bias?: TensorLike<readonly [OutChannels]> | null,
    stride?: Kernel | number | readonly [number, number],
    padding?: number | readonly [number, number],
    dilation?: number | readonly [number, number],
    groups?: number,
  ): Tensor<Conv2dForwardShape<readonly [InChannels, number, number], OutChannels, Kernel, Kernel>>;
  conv2d<const Batch extends number, const InChannels extends number, const OutChannels extends number, const Kernel extends number>(
    input: Tensor<readonly [Batch, InChannels, number, number]>,
    weight: Tensor<readonly [OutChannels, InChannels, Kernel, Kernel]>,
    bias?: TensorLike<readonly [OutChannels]> | null,
    stride?: Kernel | number | readonly [number, number],
    padding?: number | readonly [number, number],
    dilation?: number | readonly [number, number],
    groups?: number,
  ): Tensor<Conv2dForwardShape<readonly [Batch, InChannels, number, number], OutChannels, Kernel, Kernel>>;
  conv2d(input: TensorLike, weight: TensorLike, bias?: TensorLike | null, stride?: number | readonly [number, number], padding?: number | readonly [number, number], dilation?: number | readonly [number, number], groups?: number): Tensor;
  maxPool2d<const S extends readonly [number, number, number] | readonly [number, number, number, number], const Kernel extends number>(
    input: Tensor<S>,
    kernelSize: Kernel,
    stride?: Kernel | number | readonly [number, number] | null,
    padding?: number | readonly [number, number],
    dilation?: number | readonly [number, number],
    ceilMode?: boolean,
  ): Tensor<MaxPool2dForwardShape<S, Kernel, Kernel>>;
  maxPool2d(input: TensorLike, kernelSize: number | readonly [number, number], stride?: number | readonly [number, number] | null, padding?: number | readonly [number, number], dilation?: number | readonly [number, number], ceilMode?: boolean): Tensor;
  max_pool2d<const S extends readonly [number, number, number] | readonly [number, number, number, number], const Kernel extends number>(
    input: Tensor<S>,
    kernelSize: Kernel,
    stride?: Kernel | number | readonly [number, number] | null,
    padding?: number | readonly [number, number],
    dilation?: number | readonly [number, number],
    ceilMode?: boolean,
  ): Tensor<MaxPool2dForwardShape<S, Kernel, Kernel>>;
  max_pool2d(input: TensorLike, kernelSize: number | readonly [number, number], stride?: number | readonly [number, number] | null, padding?: number | readonly [number, number], dilation?: number | readonly [number, number], ceilMode?: boolean): Tensor;
  avgPool2d<const S extends readonly [number, number, number] | readonly [number, number, number, number], const Kernel extends number>(
    input: Tensor<S>,
    kernelSize: Kernel,
    stride?: Kernel | number | readonly [number, number] | null,
    padding?: number | readonly [number, number],
    ceilMode?: boolean,
    countIncludePad?: boolean,
  ): Tensor<MaxPool2dForwardShape<S, Kernel, Kernel>>;
  avgPool2d(input: TensorLike, kernelSize: number | readonly [number, number], stride?: number | readonly [number, number] | null, padding?: number | readonly [number, number], ceilMode?: boolean, countIncludePad?: boolean): Tensor;
  avg_pool2d<const S extends readonly [number, number, number] | readonly [number, number, number, number], const Kernel extends number>(
    input: Tensor<S>,
    kernelSize: Kernel,
    stride?: Kernel | number | readonly [number, number] | null,
    padding?: number | readonly [number, number],
    ceilMode?: boolean,
    countIncludePad?: boolean,
  ): Tensor<MaxPool2dForwardShape<S, Kernel, Kernel>>;
  avg_pool2d(input: TensorLike, kernelSize: number | readonly [number, number], stride?: number | readonly [number, number] | null, padding?: number | readonly [number, number], ceilMode?: boolean, countIncludePad?: boolean): Tensor;
  normalize<const S extends TensorShapeTuple>(input: Tensor<S>, p?: number, dim?: number, eps?: number): Tensor<S>;
  normalize(input: TensorLike, p?: number, dim?: number, eps?: number): Tensor;
  oneHot<const S extends TensorShapeTuple, const NumClasses extends number>(input: Tensor<S>, numClasses: NumClasses): Tensor<OneHotShape<S, NumClasses>>;
  oneHot<const NumClasses extends number>(input: IndexLike, numClasses: NumClasses): Tensor<readonly [number, NumClasses]>;
  oneHot(input: IndexLike, numClasses?: number): Tensor;
  one_hot<const S extends TensorShapeTuple, const NumClasses extends number>(input: Tensor<S>, numClasses: NumClasses): Tensor<OneHotShape<S, NumClasses>>;
  one_hot<const NumClasses extends number>(input: IndexLike, numClasses: NumClasses): Tensor<readonly [number, NumClasses]>;
  one_hot(input: IndexLike, numClasses?: number): Tensor;
  embedding<const S extends TensorShapeTuple, const NumEmbeddings extends number, const EmbeddingDim extends number>(
    input: Tensor<S>,
    weight: Tensor<readonly [NumEmbeddings, EmbeddingDim]>,
  ): Tensor<EmbeddingForwardShape<S, EmbeddingDim>>;
  embedding(input: TensorLike, weight: TensorLike): Tensor;
  layerNorm<const S extends TensorShapeTuple>(input: Tensor<S>, features: number | TensorShape, config?: NnNormConfig): Tensor<S>;
  layerNorm<const S extends TensorShapeTuple>(input: Tensor<S>, features: number | TensorShape, weight?: TensorLike | null, bias?: TensorLike | null, eps?: number): Tensor<S>;
  layerNorm(input: TensorLike, features: number | TensorShape, config?: NnNormConfig): Tensor;
  layerNorm(input: TensorLike, features: number | TensorShape, weight?: TensorLike | null, bias?: TensorLike | null, eps?: number): Tensor;
  layer_norm<const S extends TensorShapeTuple>(input: Tensor<S>, features: number | TensorShape, config?: NnNormConfig): Tensor<S>;
  layer_norm<const S extends TensorShapeTuple>(input: Tensor<S>, features: number | TensorShape, weight?: TensorLike | null, bias?: TensorLike | null, eps?: number): Tensor<S>;
  layer_norm(input: TensorLike, features: number | TensorShape, config?: NnNormConfig): Tensor;
  layer_norm(input: TensorLike, features: number | TensorShape, weight?: TensorLike | null, bias?: TensorLike | null, eps?: number): Tensor;
  rmsNorm<const S extends TensorShapeTuple>(input: Tensor<S>, features: number | TensorShape, config?: NnNormConfig): Tensor<S>;
  rmsNorm<const S extends TensorShapeTuple>(input: Tensor<S>, features: number | TensorShape, weight?: TensorLike | null, eps?: number): Tensor<S>;
  rmsNorm(input: TensorLike, features: number | TensorShape, config?: NnNormConfig): Tensor;
  rmsNorm(input: TensorLike, features: number | TensorShape, weight?: TensorLike | null, eps?: number): Tensor;
  rms_norm<const S extends TensorShapeTuple>(input: Tensor<S>, features: number | TensorShape, config?: NnNormConfig): Tensor<S>;
  rms_norm<const S extends TensorShapeTuple>(input: Tensor<S>, features: number | TensorShape, weight?: TensorLike | null, eps?: number): Tensor<S>;
  rms_norm(input: TensorLike, features: number | TensorShape, config?: NnNormConfig): Tensor;
  rms_norm(input: TensorLike, features: number | TensorShape, weight?: TensorLike | null, eps?: number): Tensor;
  batchNorm1d<const S extends TensorShapeTuple>(input: Tensor<S>, features: number | TensorShape, config?: NnNormConfig): Tensor<S>;
  batchNorm1d(input: TensorLike, features: number | TensorShape, config?: NnNormConfig): Tensor;
  batch_norm1d<const S extends TensorShapeTuple>(input: Tensor<S>, features: number | TensorShape, config?: NnNormConfig): Tensor<S>;
  batch_norm1d(input: TensorLike, features: number | TensorShape, config?: NnNormConfig): Tensor;
};

export type NnNamespace = Readonly<{
  functional: NnFunctionalNamespace;
  F: NnFunctionalNamespace;
  Module: new <
    const InputShape extends TensorShapeTuple = TensorShapeTuple,
    const OutputShape extends TensorShapeTuple = TensorShapeTuple,
  >(config?: { kind?: string; graph?: NnCompilableModule | (() => NnCompilableModule) }) => ModuleBase<InputShape, OutputShape>;
  ModuleList: {
    new <const Layers extends readonly NnModule[]>(layers?: readonly [...Layers]): ModuleList<Layers>;
    new(layers?: Iterable<NnModule>): ModuleList;
  };
  ModuleDict: new <const Layers extends Readonly<Record<string, NnModule>>>(layers?: Layers) => ModuleDict<Layers>;
  ParameterList: {
    new <const Params extends readonly NnParameter[]>(params?: readonly [...Params]): ParameterList<Params>;
    new(params?: Iterable<NnParameter>): ParameterList;
  };
  ParameterDict: new <const Params extends Readonly<Record<string, NnParameter>>>(params?: Params) => ParameterDict<Params>;
  Linear: new <const InFeatures extends number, const OutFeatures extends number>(inFeatures: InFeatures, outFeatures: OutFeatures, config?: NnLinearConfig) => LinearModule<InFeatures, OutFeatures>;
  Embedding: new <const NumEmbeddings extends number, const EmbeddingDim extends number>(numEmbeddings: NumEmbeddings, embeddingDim: EmbeddingDim, config?: NnEmbeddingConfig) => EmbeddingModule<NumEmbeddings, EmbeddingDim>;
  Conv2d: new <const InChannels extends number, const OutChannels extends number, const Kernel extends number>(inChannels: InChannels, outChannels: OutChannels, kernelSize: Kernel, config?: NnConv2dConfig) => Conv2dModule<InChannels, OutChannels, Kernel>;
  AvgPool2d: new <const Kernel extends number>(kernelSize: Kernel, config?: NnAvgPool2dConfig) => AvgPool2dModule<Kernel, Kernel>;
  MaxPool2d: new <const Kernel extends number>(kernelSize: Kernel, config?: NnMaxPool2dConfig) => MaxPool2dModule<Kernel, Kernel>;
  Sequential: {
    new (): SequentialModule<readonly []>;
    new <const Layers extends readonly NnModule[]>(layers: Layers): SequentialModule<Layers>;
    new <const Layers extends Readonly<Record<string, NnModule>>>(layers: Layers): SequentialModule<ReadonlyArray<Layers[keyof Layers]>>;
    new <const First extends NnModule, const Rest extends readonly NnModule[]>(layer: First, ...layers: Rest): SequentialModule<readonly [First, ...Rest]>;
  };
  Softmax: new (dim?: number) => SoftmaxModule;
  LogSoftmax: new (dim?: number) => LogSoftmaxModule;
  Reduction: new <const Dim extends number = number>(kind: ModuleReductionKind, dim?: Dim) => ReductionModule<Dim>;
  Dropout: new (p?: number, config?: NnDropoutConfig) => DropoutModule;
  Shape: new (kind: ShapeModuleKind, shapeOrStartDim?: TensorShape | number | null, endDim?: number | null, length?: number | null, step?: number) => ShapeModule;
  LayerNorm: new (features: number, config?: NnNormConfig) => FeatureNormModule;
  RMSNorm: new (features: number, config?: NnNormConfig) => FeatureNormModule;
  BatchNorm1d: new (features: number, config?: NnNormConfig) => FeatureNormModule;
  Identity: new () => ShapeModule;
  Diagonal: new () => ShapeModule<readonly [], "diagonal">;
  Repeat: new <const Repeats extends TensorShape>(repeats: Repeats) => ShapeModule<TensorShapeOf<Repeats>, "repeat">;
  Tile: new <const Repeats extends TensorShape>(repeats: Repeats) => ShapeModule<TensorShapeOf<Repeats>, "tile">;
  Flatten: new (startDim?: number, endDim?: number) => ShapeModule;
  ReLU: new () => CompilableActivationModule;
  GELU: new () => CompilableActivationModule;
  SiLU: new () => CompilableActivationModule;
  Sigmoid: new () => CompilableActivationModule;
  Tanh: new () => CompilableActivationModule;
  MSELoss: NnLossConstructors["MSELoss"];
  L1Loss: NnLossConstructors["L1Loss"];
  HuberLoss: NnLossConstructors["HuberLoss"];
  SmoothL1Loss: NnLossConstructors["SmoothL1Loss"];
  BCELoss: NnLossConstructors["BCELoss"];
  BCEWithLogitsLoss: NnLossConstructors["BCEWithLogitsLoss"];
  CrossEntropyLoss: NnLossConstructors["CrossEntropyLoss"];
  NLLLoss: NnLossConstructors["NLLLoss"];
  linear<const InFeatures extends number, const OutFeatures extends number>(inFeatures: InFeatures, outFeatures: OutFeatures, config?: NnLinearConfig): LinearModule<InFeatures, OutFeatures>;
  conv2d<const InChannels extends number, const OutChannels extends number, const Kernel extends number>(inChannels: InChannels, outChannels: OutChannels, kernelSize: Kernel, config?: NnConv2dConfig): Conv2dModule<InChannels, OutChannels, Kernel>;
  conv2d<const InChannels extends number, const OutChannels extends number>(inChannels: InChannels, outChannels: OutChannels, kernelSize: readonly [number, number], config?: NnConv2dConfig): Conv2dModule<InChannels, OutChannels>;
  avgPool2d<const Kernel extends number>(kernelSize: Kernel, config?: NnAvgPool2dConfig): AvgPool2dModule<Kernel, Kernel>;
  avgPool2d(kernelSize: readonly [number, number], config?: NnAvgPool2dConfig): AvgPool2dModule;
  avg_pool2d<const Kernel extends number>(kernelSize: Kernel, config?: NnAvgPool2dConfig): AvgPool2dModule<Kernel, Kernel>;
  avg_pool2d(kernelSize: readonly [number, number], config?: NnAvgPool2dConfig): AvgPool2dModule;
  maxPool2d<const Kernel extends number>(kernelSize: Kernel, config?: NnMaxPool2dConfig): MaxPool2dModule<Kernel, Kernel>;
  maxPool2d(kernelSize: readonly [number, number], config?: NnMaxPool2dConfig): MaxPool2dModule;
  max_pool2d<const Kernel extends number>(kernelSize: Kernel, config?: NnMaxPool2dConfig): MaxPool2dModule<Kernel, Kernel>;
  max_pool2d(kernelSize: readonly [number, number], config?: NnMaxPool2dConfig): MaxPool2dModule;
  Buffer<const S extends TensorShapeTuple>(name: string, data: Tensor<S>, options?: NnBufferOptions<S>): NnBuffer<S>;
  Buffer<const S extends TensorShape>(name: string, data: TensorLike, shape: S, options?: NnBufferOptions<S>): NnBuffer<TensorShapeOf<S>>;
  Buffer(name: string, data: TensorLike, options?: NnBufferOptions): NnBuffer;
  buffer<const S extends TensorShapeTuple>(name: string, data: Tensor<S>, options?: NnBufferOptions<S>): NnBuffer<S>;
  buffer<const S extends TensorShape>(name: string, data: TensorLike, shape: S, options?: NnBufferOptions<S>): NnBuffer<TensorShapeOf<S>>;
  buffer(name: string, data: TensorLike, options?: NnBufferOptions): NnBuffer;
  Parameter<const S extends TensorShapeTuple>(name: string, data: Tensor<S>, options?: NnParameterOptions<S>): NnParameter<S>;
  Parameter<const S extends TensorShape>(name: string, data: TensorLike, shape: S, options?: NnParameterOptions<S>): NnParameter<TensorShapeOf<S>>;
  Parameter(name: string, data: TensorLike, options?: NnParameterOptions): NnParameter;
  parameter<const S extends TensorShapeTuple>(name: string, data: Tensor<S>, options?: NnParameterOptions<S>): NnParameter<S>;
  parameter<const S extends TensorShape>(name: string, data: TensorLike, shape: S, options?: NnParameterOptions<S>): NnParameter<TensorShapeOf<S>>;
  parameter(name: string, data: TensorLike, options?: NnParameterOptions): NnParameter;
  init: NnInitNamespace;
  moduleList<const Layers extends readonly NnModule[]>(layers?: readonly [...Layers]): ModuleList<Layers>;
  moduleList(layers?: Iterable<NnModule>): ModuleList;
  module_list<const Layers extends readonly NnModule[]>(layers?: readonly [...Layers]): ModuleList<Layers>;
  module_list(layers?: Iterable<NnModule>): ModuleList;
  moduleDict<const Layers extends Readonly<Record<string, NnModule>>>(layers?: Layers): ModuleDict<Layers>;
  module_dict<const Layers extends Readonly<Record<string, NnModule>>>(layers?: Layers): ModuleDict<Layers>;
  parameterList<const Params extends readonly NnParameter[]>(params?: readonly [...Params]): ParameterList<Params>;
  parameterList(params?: Iterable<NnParameter>): ParameterList;
  parameter_list<const Params extends readonly NnParameter[]>(params?: readonly [...Params]): ParameterList<Params>;
  parameter_list(params?: Iterable<NnParameter>): ParameterList;
  parameterDict<const Params extends Readonly<Record<string, NnParameter>>>(params?: Params): ParameterDict<Params>;
  parameter_dict<const Params extends Readonly<Record<string, NnParameter>>>(params?: Params): ParameterDict<Params>;
  module<const InputShape extends TensorShapeTuple, const OutputShape extends TensorShapeTuple>(config: NnModuleConfig<InputShape, OutputShape>): CustomModule<InputShape, OutputShape>;
  module(config: NnModuleConfig): CustomModule;
  embedding<const NumEmbeddings extends number, const EmbeddingDim extends number>(numEmbeddings: NumEmbeddings, embeddingDim: EmbeddingDim, config?: NnEmbeddingConfig): EmbeddingModule<NumEmbeddings, EmbeddingDim>;
  sequential(): SequentialModule<readonly []>;
  sequential<const Layers extends readonly NnModule[]>(layers: readonly [...Layers]): SequentialModule<Layers>;
  sequential<const Layers extends Readonly<Record<string, NnModule>>>(layers: Layers): SequentialModule<ReadonlyArray<Layers[keyof Layers]>>;
  sequential<const First extends NnModule, const Rest extends readonly NnModule[]>(layer: First, ...layers: Rest): SequentialModule<readonly [First, ...Rest]>;
  gelu(): CompilableActivationModule;
  relu(): CompilableActivationModule;
  silu(): CompilableActivationModule;
  sigmoid(): CompilableActivationModule;
  tanh(): CompilableActivationModule;
  exp(): CompilableActivationModule;
  log(): CompilableActivationModule;
  neg(): CompilableActivationModule;
  recip(): CompilableActivationModule;
  abs(): CompilableActivationModule;
  sgn(): CompilableActivationModule;
  sign(): CompilableActivationModule;
  step(): CompilableActivationModule;
  sqrt(): CompilableActivationModule;
  square(): CompilableActivationModule;
  sqr(): CompilableActivationModule;
  softmax(dim?: number): SoftmaxModule;
  logSoftmax(dim?: number): LogSoftmaxModule;
  log_softmax(dim?: number): LogSoftmaxModule;
  sum<const Dim extends number = number>(dim?: Dim): ReductionModule<Dim>;
  mean<const Dim extends number = number>(dim?: Dim): ReductionModule<Dim>;
  prod<const Dim extends number = number>(dim?: Dim): ReductionModule<Dim>;
  max<const Dim extends number = number>(dim?: Dim): ReductionModule<Dim>;
  min<const Dim extends number = number>(dim?: Dim): ReductionModule<Dim>;
  argmax<const Dim extends number = number>(dim?: Dim): ReductionModule<Dim>;
  argmin<const Dim extends number = number>(dim?: Dim): ReductionModule<Dim>;
  dropout(p?: number, config?: NnDropoutConfig): DropoutModule;
  identity(): ShapeModule;
  diagonal(): ShapeModule<readonly [], "diagonal">;
  repeat<const Repeats extends TensorShape>(repeats: Repeats): ShapeModule<TensorShapeOf<Repeats>, "repeat">;
  tile<const Repeats extends TensorShape>(repeats: Repeats): ShapeModule<TensorShapeOf<Repeats>, "tile">;
  reshape<const S extends TensorShape>(shape: S): ShapeModule<TensorShapeOf<S>, "reshape">;
  view<const S extends TensorShape>(shape: S): ShapeModule<TensorShapeOf<S>, "view">;
  flatten(startDim?: number, endDim?: number): ShapeModule;
  squeeze(): ShapeModule;
  squeeze<const Dim extends number>(dim: Dim): ShapeModule<TensorShapeTuple, "squeeze", Dim>;
  squeeze(dim?: number | null): ShapeModule;
  unsqueeze<const Dim extends number>(dim: Dim): ShapeModule<TensorShapeTuple, "unsqueeze", Dim>;
  transpose<const Dim0 extends number = 0, const Dim1 extends number = 1>(dim0?: Dim0, dim1?: Dim1): ShapeModule<TensorShapeTuple, "transpose", Dim0, Dim1>;
  permute<const Dims extends readonly number[]>(dims: Dims): ShapeModule<TensorShapeTuple, "permute", number, number, number, number, number, Dims>;
  broadcastTo<const S extends TensorShape>(shape: S): ShapeModule<TensorShapeOf<S>, "broadcastTo">;
  expand<const S extends TensorShape>(shape: S): ShapeModule<TensorShapeOf<S>, "expand">;
  narrow<const Dim extends number, const Length extends number>(dim: Dim, start: number, length: Length): ShapeModule<TensorShapeTuple, "narrow", Dim, number, Length>;
  select<const Dim extends number>(dim: Dim, index: number): ShapeModule<TensorShapeTuple, "select", Dim>;
  slice<const Dim extends number, const Start extends number, const End extends number>(dim: Dim, start: Start, end: End, step?: 1): ShapeModule<TensorShapeTuple, "slice", Dim, number, number, Start, End>;
  slice(dim: number, start?: number, end?: number | null, step?: number): ShapeModule;
  layerNorm(features: number, config?: NnNormConfig): FeatureNormModule;
  layer_norm(features: number, config?: NnNormConfig): FeatureNormModule;
  rmsNorm(features: number, config?: NnNormConfig): FeatureNormModule;
  rms_norm(features: number, config?: NnNormConfig): FeatureNormModule;
  batchNorm1d(features: number, config?: NnNormConfig): FeatureNormModule;
  batch_norm1d(features: number, config?: NnNormConfig): FeatureNormModule;
  parameters(target: readonly NnModule[]): NnParameter[];
  parameters(target: OptimizerTarget): NnParameter[];
  parameters(target: readonly NnModule[], options: ModuleParameterTraversalOptions): NnParameter[];
  parameters(target: OptimizerTarget, options: ModuleParameterTraversalOptions): NnParameter[];
  parameters(target: readonly NnModule[], prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  parameters(target: OptimizerTarget, prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  namedParameters(target: readonly NnModule[], prefix?: string): NnParameter[];
  namedParameters(target: OptimizerTarget, prefix?: string): NnParameter[];
  namedParameters(target: readonly NnModule[], options: ModuleParameterTraversalOptions): NnParameter[];
  namedParameters(target: OptimizerTarget, options: ModuleParameterTraversalOptions): NnParameter[];
  namedParameters(target: readonly NnModule[], prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  namedParameters(target: OptimizerTarget, prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  named_parameters(target: readonly NnModule[], prefix?: string): NnParameter[];
  named_parameters(target: OptimizerTarget, prefix?: string): NnParameter[];
  named_parameters(target: readonly NnModule[], options: ModuleParameterTraversalOptions): NnParameter[];
  named_parameters(target: OptimizerTarget, options: ModuleParameterTraversalOptions): NnParameter[];
  named_parameters(target: readonly NnModule[], prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  named_parameters(target: OptimizerTarget, prefix: string, options: ModuleParameterTraversalOptions): NnParameter[];
  getParameter(target: readonly NnModule[], name: string): NnParameter | null;
  getParameter(target: OptimizerTarget, name: string): NnParameter | null;
  get_parameter(target: readonly NnModule[], name: string): NnParameter | null;
  get_parameter(target: OptimizerTarget, name: string): NnParameter | null;
  buffers(target: NnModule | readonly NnModule[], prefix?: string): NnBuffer[];
  buffers(target: NnModule | readonly NnModule[], options: ModuleBufferTraversalOptions): NnBuffer[];
  buffers(target: NnModule | readonly NnModule[], prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  namedBuffers(target: NnModule | readonly NnModule[], prefix?: string): NnBuffer[];
  namedBuffers(target: NnModule | readonly NnModule[], options: ModuleBufferTraversalOptions): NnBuffer[];
  namedBuffers(target: NnModule | readonly NnModule[], prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  named_buffers(target: NnModule | readonly NnModule[], prefix?: string): NnBuffer[];
  named_buffers(target: NnModule | readonly NnModule[], options: ModuleBufferTraversalOptions): NnBuffer[];
  named_buffers(target: NnModule | readonly NnModule[], prefix: string, options: ModuleBufferTraversalOptions): NnBuffer[];
  getBuffer(target: NnModule | readonly NnModule[], name: string): NnBuffer | null;
  get_buffer(target: NnModule | readonly NnModule[], name: string): NnBuffer | null;
  parameterNames(target: readonly NnModule[], prefix?: string): readonly string[];
  parameterNames(target: OptimizerTarget, prefix?: string): readonly string[];
  parameterInfos(target: readonly NnModule[], prefix?: string): readonly ModuleParameterInfo[];
  parameterInfos(target: OptimizerTarget, prefix?: string): readonly ModuleParameterInfo[];
  parameterInfo(target: readonly NnModule[], nameOrIndex: string | number, prefix?: string): ModuleParameterInfo | null;
  parameterInfo(target: OptimizerTarget, nameOrIndex: string | number, prefix?: string): ModuleParameterInfo | null;
  call<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, input: Tensor<S>): Tensor<ModuleTargetForwardShape<Target, S>>;
  call(target: NnModule | readonly NnModule[], input: TensorLike): Tensor;
  __call__<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, input: Tensor<S>): Tensor<ModuleTargetForwardShape<Target, S>>;
  __call__(target: NnModule | readonly NnModule[], input: TensorLike): Tensor;
  forward<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, input: Tensor<S>): Tensor<ModuleTargetForwardShape<Target, S>>;
  forward(target: NnModule | readonly NnModule[], input: TensorLike): Tensor;
  children(target: NnModule | readonly NnModule[]): readonly NnModule[];
  modules(target: NnModule | readonly NnModule[]): readonly NnModule[];
  namedChildren(target: NnModule | readonly NnModule[], prefix?: string): readonly ModuleTraversalEntry[];
  named_children(target: NnModule | readonly NnModule[], prefix?: string): readonly ModuleTraversalEntry[];
  namedModules(target: NnModule | readonly NnModule[], prefix?: string): readonly ModuleTraversalEntry[];
  named_modules(target: NnModule | readonly NnModule[], prefix?: string): readonly ModuleTraversalEntry[];
  getSubmodule(target: NnModule | readonly NnModule[], name: string): NnModule | null;
  get_submodule(target: NnModule | readonly NnModule[], name: string): NnModule | null;
  apply<const Layers extends readonly NnModule[]>(target: Layers, callback: (module: NnModule, entry: ModuleTraversalEntry) => void): SequentialModule<Layers>;
  apply<T extends NnModule>(target: T, callback: (module: NnModule, entry: ModuleTraversalEntry) => void): T;
  train<const Layers extends readonly NnModule[]>(target: Layers, mode?: boolean): SequentialModule<Layers>;
  train<T extends NnModule>(target: T, mode?: boolean): T;
  eval<const Layers extends readonly NnModule[]>(target: Layers): SequentialModule<Layers>;
  eval<T extends NnModule>(target: T): T;
  to<const Layers extends readonly NnModule[]>(target: Layers, placement?: TensorToTarget | TensorToOptions, options?: TensorToOptions): SequentialModule<Layers>;
  to<T extends NnModule>(target: T, placement?: TensorToTarget | TensorToOptions, options?: TensorToOptions): T;
  cpu<const Layers extends readonly NnModule[]>(target: Layers, options?: TensorToOptions): SequentialModule<Layers>;
  cpu<T extends NnModule>(target: T, options?: TensorToOptions): T;
  float<const Layers extends readonly NnModule[]>(target: Layers, options?: TensorToOptions): SequentialModule<Layers>;
  float<T extends NnModule>(target: T, options?: TensorToOptions): T;
  float32<const Layers extends readonly NnModule[]>(target: Layers, options?: TensorToOptions): SequentialModule<Layers>;
  float32<T extends NnModule>(target: T, options?: TensorToOptions): T;
  zeroGrad(target: readonly NnModule[], options?: ZeroGradOptions): void;
  zeroGrad(target: OptimizerTarget, options?: ZeroGradOptions): void;
  zero_grad(target: readonly NnModule[], options?: ZeroGradOptions): void;
  zero_grad(target: OptimizerTarget, options?: ZeroGradOptions): void;
  requiresGrad<const Layers extends readonly NnModule[]>(target: Layers, requiresGrad?: boolean): SequentialModule<Layers>;
  requiresGrad<T extends OptimizerTarget>(target: T, requiresGrad?: boolean): T;
  requiresGrad_<const Layers extends readonly NnModule[]>(target: Layers, requiresGrad?: boolean): SequentialModule<Layers>;
  requiresGrad_<T extends OptimizerTarget>(target: T, requiresGrad?: boolean): T;
  requires_grad_<const Layers extends readonly NnModule[]>(target: Layers, requiresGrad?: boolean): SequentialModule<Layers>;
  requires_grad_<T extends OptimizerTarget>(target: T, requiresGrad?: boolean): T;
  freeze<const Layers extends readonly NnModule[]>(target: Layers): SequentialModule<Layers>;
  freeze<T extends OptimizerTarget>(target: T): T;
  unfreeze<const Layers extends readonly NnModule[]>(target: Layers): SequentialModule<Layers>;
  unfreeze<T extends OptimizerTarget>(target: T): T;
  stateDict(target: readonly NnModule[], prefix?: string): ModuleStateSnapshot;
  stateDict(target: OptimizerTarget, prefix?: string): ModuleStateSnapshot;
  stateDict(target: readonly NnModule[], options: ModuleStateDictOptions): ModuleStateSnapshot;
  stateDict(target: OptimizerTarget, options: ModuleStateDictOptions): ModuleStateSnapshot;
  state_dict(target: readonly NnModule[], prefix?: string): ModuleStateSnapshot;
  state_dict(target: OptimizerTarget, prefix?: string): ModuleStateSnapshot;
  state_dict(target: readonly NnModule[], options: ModuleStateDictOptions): ModuleStateSnapshot;
  state_dict(target: OptimizerTarget, options: ModuleStateDictOptions): ModuleStateSnapshot;
  loadStateDict<const Layers extends readonly NnModule[]>(target: Layers, source: ModuleStateDict, options?: LoadStateDictOptions): SequentialModule<Layers>;
  loadStateDict<T extends OptimizerTarget>(target: T, source: ModuleStateDict, options?: LoadStateDictOptions): T;
  load_state_dict<const Layers extends readonly NnModule[]>(target: Layers, source: ModuleStateDict, options?: LoadStateDictOptions): SequentialModule<Layers>;
  load_state_dict<T extends OptimizerTarget>(target: T, source: ModuleStateDict, options?: LoadStateDictOptions): T;
  trace(target: NnModule | readonly NnModule[], options?: ModuleTraceOptions): ModuleProgramTrace;
  trace(target: LazyTensor, options?: CompileOptions): ModuleProgramTrace;
  compileSupport<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, ModuleTargetForwardShape<Target, S>>;
  compileSupport(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileSupport;
  compileSupport(target: LazyTensor, options?: CompileOptions): LazyCompileSupport;
  compile_support<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, ModuleTargetForwardShape<Target, S>>;
  compile_support(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileSupport;
  compile_support(target: LazyTensor, options?: CompileOptions): LazyCompileSupport;
  requireCompileSupport<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, ModuleTargetForwardShape<Target, S>>;
  requireCompileSupport(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileSupport;
  requireCompileSupport(target: LazyTensor, options?: CompileOptions): LazyCompileSupport;
  require_compile_support<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileSupport<S, ModuleTargetForwardShape<Target, S>>;
  require_compile_support(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileSupport;
  require_compile_support(target: LazyTensor, options?: CompileOptions): LazyCompileSupport;
  compilerSignatures(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleCompilerSignatures | null;
  compiler_signatures(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleCompilerSignatures | null;
  tensorProgramIr(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleTensorProgramIr | null;
  tensor_program_ir(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleTensorProgramIr | null;
  kernelPlan(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleKernelPlan | null;
  kernel_plan(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleKernelPlan | null;
  bufferLayout(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleKernelBufferLayout | null;
  buffer_layout(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleKernelBufferLayout | null;
  memoryLayout(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleKernelMemoryLayout | null;
  memory_layout(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleKernelMemoryLayout | null;
  inputShape(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): readonly number[] | null;
  input_shape(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): readonly number[] | null;
  outputShape<const Target extends EmbeddingModule, const S extends TensorShapeTuple>(target: Target, options: EmbeddingCompileOptions<S>): ModuleForwardShape<Target, S> | null;
  outputShape<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleTargetForwardShape<Target, S> | null;
  outputShape<const Shape extends TensorShapeTuple>(target: LazyTensor<Shape>, options?: CompileOptions): Shape | null;
  outputShape(target: NnModule | readonly NnModule[], options?: CompileOptions): readonly number[] | null;
  output_shape<const Target extends EmbeddingModule, const S extends TensorShapeTuple>(target: Target, options: EmbeddingCompileOptions<S>): ModuleForwardShape<Target, S> | null;
  output_shape<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleTargetForwardShape<Target, S> | null;
  output_shape<const Shape extends TensorShapeTuple>(target: LazyTensor<Shape>, options?: CompileOptions): Shape | null;
  output_shape(target: NnModule | readonly NnModule[], options?: CompileOptions): readonly number[] | null;
  shapeConstraints(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleKernelShapeConstraints | null;
  shape_constraints(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleKernelShapeConstraints | null;
  parameterLayout(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleKernelParameterLayout | null;
  parameter_layout(target: NnModule | readonly NnModule[] | LazyTensor, options?: CompileOptions): ModuleKernelParameterLayout | null;
  explain<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ModuleTargetForwardShape<Target, S>>;
  explain(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileExplanation;
  preflight<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ModuleTargetForwardShape<Target, S>>;
  preflight(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileExplanation;
  compileExplanation<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ModuleTargetForwardShape<Target, S>>;
  compileExplanation(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileExplanation;
  compile_explanation<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ModuleTargetForwardShape<Target, S>>;
  compile_explanation(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileExplanation;
  compilePlan<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ModuleTargetForwardShape<Target, S>>;
  compilePlan(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileExplanation;
  compile_plan<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ModuleTargetForwardShape<Target, S>>;
  compile_plan(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileExplanation;
  requireCompilePlan<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ModuleTargetForwardShape<Target, S>>;
  requireCompilePlan(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileExplanation;
  require_compile_plan<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ModuleTargetForwardShape<Target, S>>;
  require_compile_plan(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileExplanation;
  assertCompilePlan<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ModuleTargetForwardShape<Target, S>>;
  assertCompilePlan(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileExplanation;
  assert_compile_plan<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleCompileExplanation<S, ModuleTargetForwardShape<Target, S>>;
  assert_compile_plan(target: NnModule | readonly NnModule[], options?: CompileOptions): ModuleCompileExplanation;
  canCompile(target: NnModule | readonly NnModule[], options?: CompileOptions): boolean;
  canCompile(target: LazyTensor, options?: CompileOptions): boolean;
  can_compile(target: NnModule | readonly NnModule[], options?: CompileOptions): boolean;
  can_compile(target: LazyTensor, options?: CompileOptions): boolean;
  compile<const Target extends EmbeddingModule, const S extends TensorShapeTuple>(target: Target, options: EmbeddingCompileOptions<S>): Program<S, ModuleForwardShape<Target, S>>;
  compile<const Target extends NnCompilableModule, const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): Program<S, ModuleForwardShape<Target, S>>;
  compile<const Shape extends TensorShapeTuple>(target: LazyTensor<Shape>, options?: CompileOptions): Program<TensorShapeTuple, Shape>;
  compile(target: NnCompilableModule, options?: CompileOptions): Program;
  compile<const Target extends readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): Program<S, ModuleTargetForwardShape<Target, S>>;
  compile(target: readonly NnModule[], options?: CompileOptions): Program;
  native<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, ModuleTargetForwardShape<Target, S>>;
  native(target: NnModule | readonly NnModule[], options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  inference<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, ModuleTargetForwardShape<Target, S>>;
  inference(target: NnModule | readonly NnModule[], options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  forInference<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, ModuleTargetForwardShape<Target, S>>;
  forInference(target: NnModule | readonly NnModule[], options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  for_inference<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, ModuleTargetForwardShape<Target, S>>;
  for_inference(target: NnModule | readonly NnModule[], options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  compileInference<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, ModuleTargetForwardShape<Target, S>>;
  compileInference(target: NnModule | readonly NnModule[], options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  compile_inference<const Target extends NnModule | readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>, bindOptions?: ModuleParameterPlacementOptions): CompiledInference<S, ModuleTargetForwardShape<Target, S>>;
  compile_inference(target: NnModule | readonly NnModule[], options?: CompileOptions, bindOptions?: ModuleParameterPlacementOptions): CompiledInference;
  bindParameters<const Target extends EmbeddingModule, const S extends TensorShapeTuple>(target: Target, options: EmbeddingCompileOptions<S>): ModuleBindings<S, ModuleForwardShape<Target, S>>;
  bindParameters(target: EmbeddingModule, options: EmbeddingCompileOptions): ModuleBindings;
  bindParameters<const Target extends NnCompilableModule, const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleBindings<S, ModuleForwardShape<Target, S>>;
  bindParameters(target: NnCompilableModule, options?: CompileOptions): ModuleBindings;
  bindParameters<const Target extends readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleBindings<S, ModuleTargetForwardShape<Target, S>>;
  bindParameters(target: readonly NnModule[], options?: CompileOptions): ModuleBindings;
  bind_parameters<const Target extends EmbeddingModule, const S extends TensorShapeTuple>(target: Target, options: EmbeddingCompileOptions<S>): ModuleBindings<S, ModuleForwardShape<Target, S>>;
  bind_parameters(target: EmbeddingModule, options: EmbeddingCompileOptions): ModuleBindings;
  bind_parameters<const Target extends NnCompilableModule, const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleBindings<S, ModuleForwardShape<Target, S>>;
  bind_parameters(target: NnCompilableModule, options?: CompileOptions): ModuleBindings;
  bind_parameters<const Target extends readonly NnModule[], const S extends TensorShapeTuple>(target: Target, options: CompileOptionsWithInputShape<S>): ModuleBindings<S, ModuleTargetForwardShape<Target, S>>;
  bind_parameters(target: readonly NnModule[], options?: CompileOptions): ModuleBindings;
  placeParameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(target: EmbeddingModule, program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  placeParameters(target: EmbeddingModule, program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  placeParameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(target: NnCompilableModule, program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  placeParameters(target: NnCompilableModule, program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  placeParameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(target: readonly NnModule[], program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  placeParameters(target: readonly NnModule[], program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  place_parameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(target: EmbeddingModule, program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  place_parameters(target: EmbeddingModule, program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  place_parameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(target: NnCompilableModule, program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  place_parameters(target: NnCompilableModule, program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  place_parameters<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(target: readonly NnModule[], program: Program<S, O>, options?: ModuleParameterPlacementOptions): ModuleBindings<S, O>;
  place_parameters(target: readonly NnModule[], program: Program, options?: ModuleParameterPlacementOptions): ModuleBindings;
  bindingPlan<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(bindings: ModuleBindings<S, O>): ModuleBindingPlan<S, O>;
  bindingPlan<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(bindings: ProgramBindings<S, O>): ModuleBindingPlan<S, O>;
  bindingPlan(bindings: ProgramBindings | ModuleBindings): ModuleBindingPlan;
  binding_plan<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(bindings: ModuleBindings<S, O>): ModuleBindingPlan<S, O>;
  binding_plan<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(bindings: ProgramBindings<S, O>): ModuleBindingPlan<S, O>;
  binding_plan(bindings: ProgramBindings | ModuleBindings): ModuleBindingPlan;
  requireBindingPlan<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(bindings: ModuleBindings<S, O>): ModuleBindingPlan<S, O>;
  requireBindingPlan<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(bindings: ProgramBindings<S, O>): ModuleBindingPlan<S, O>;
  requireBindingPlan(bindings: ProgramBindings | ModuleBindings): ModuleBindingPlan;
  require_binding_plan<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(bindings: ModuleBindings<S, O>): ModuleBindingPlan<S, O>;
  require_binding_plan<const S extends TensorShapeTuple, const O extends TensorShapeTuple>(bindings: ProgramBindings<S, O>): ModuleBindingPlan<S, O>;
  require_binding_plan(bindings: ProgramBindings | ModuleBindings): ModuleBindingPlan;
  acceptsModuleBindingPlan(plan: unknown): plan is ModuleBindingPlan;
  requireModuleBindingPlan(plan: unknown): ModuleBindingPlan;
  assertModuleBindingPlan(plan: unknown): ModuleBindingPlan;
  assert_module_binding_plan(plan: unknown): ModuleBindingPlan;
  matchesModuleBindingPlanSignature(plan: unknown, signature: string): boolean;
  acceptsModuleCompilePlan(plan: unknown): plan is ModuleCompileExplanation;
  requireModuleCompilePlan(plan: unknown): ModuleCompileExplanation;
  assertModuleCompilePlan(plan: unknown): ModuleCompileExplanation;
  assert_module_compile_plan(plan: unknown): ModuleCompileExplanation;
  matchesModuleCompilePlanSignature(plan: unknown, signature: string): boolean;
}>;
export type PublicNnNamespace = Readonly<NnNamespace>;
export declare const nn: PublicNnNamespace;

export interface LossNamespace {
  meanSquaredError(pred: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  meanSquaredError(pred: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  meanSquaredError(pred: TensorData, target: TensorData, options?: LossReductionOptions): number;
  mse(pred: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  mse(pred: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  mse(pred: TensorData, target: TensorData, options?: LossReductionOptions): number;
  mseLoss(options?: LossReductionOptions): MSELoss;
  mse_loss(options?: LossReductionOptions): MSELoss;
  MSELoss: new (options?: LossReductionOptions) => MSELoss;
  meanAbsoluteError(pred: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  meanAbsoluteError(pred: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  meanAbsoluteError(pred: TensorData, target: TensorData, options?: LossReductionOptions): number;
  l1(pred: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  l1(pred: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  l1(pred: TensorData, target: TensorData, options?: LossReductionOptions): number;
  l1Loss(options?: LossReductionOptions): L1Loss;
  l1_loss(options?: LossReductionOptions): L1Loss;
  L1Loss: new (options?: LossReductionOptions) => L1Loss;
  huber(pred: Tensor, target: TensorLike, options?: HuberLossOptions): Tensor<readonly [1]>;
  huber(pred: TensorData, target: Tensor, options?: HuberLossOptions): Tensor<readonly [1]>;
  huber(pred: TensorData, target: TensorData, options?: HuberLossOptions): number;
  huberLoss(options?: HuberLossOptions): HuberLoss;
  huber_loss(options?: HuberLossOptions): HuberLoss;
  HuberLoss: new (options?: HuberLossOptions) => HuberLoss;
  smoothL1(pred: Tensor, target: TensorLike, options?: SmoothL1LossOptions): Tensor<readonly [1]>;
  smoothL1(pred: TensorData, target: Tensor, options?: SmoothL1LossOptions): Tensor<readonly [1]>;
  smoothL1(pred: TensorData, target: TensorData, options?: SmoothL1LossOptions): number;
  smooth_l1(pred: Tensor, target: TensorLike, options?: SmoothL1LossOptions): Tensor<readonly [1]>;
  smooth_l1(pred: TensorData, target: Tensor, options?: SmoothL1LossOptions): Tensor<readonly [1]>;
  smooth_l1(pred: TensorData, target: TensorData, options?: SmoothL1LossOptions): number;
  smoothL1Loss(options?: SmoothL1LossOptions): SmoothL1Loss;
  smooth_l1_loss(options?: SmoothL1LossOptions): SmoothL1Loss;
  SmoothL1Loss: new (options?: SmoothL1LossOptions) => SmoothL1Loss;
  binaryCrossEntropy(pred: Tensor, target: TensorLike, options?: BCELossOptions): Tensor<readonly [1]>;
  binaryCrossEntropy(pred: TensorData, target: Tensor, options?: BCELossOptions): Tensor<readonly [1]>;
  binaryCrossEntropy(pred: TensorData, target: TensorData, options?: BCELossOptions): number;
  binary_cross_entropy(pred: Tensor, target: TensorLike, options?: BCELossOptions): Tensor<readonly [1]>;
  binary_cross_entropy(pred: TensorData, target: Tensor, options?: BCELossOptions): Tensor<readonly [1]>;
  binary_cross_entropy(pred: TensorData, target: TensorData, options?: BCELossOptions): number;
  bce(pred: Tensor, target: TensorLike, options?: BCELossOptions): Tensor<readonly [1]>;
  bce(pred: TensorData, target: Tensor, options?: BCELossOptions): Tensor<readonly [1]>;
  bce(pred: TensorData, target: TensorData, options?: BCELossOptions): number;
  bceLoss(options?: BCELossOptions): BCELoss;
  bce_loss(options?: BCELossOptions): BCELoss;
  BCELoss: new (options?: BCELossOptions) => BCELoss;
  binaryCrossEntropyWithLogits(logits: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  binaryCrossEntropyWithLogits(logits: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  binaryCrossEntropyWithLogits(logits: TensorData, target: TensorData, options?: LossReductionOptions): number;
  binary_cross_entropy_with_logits(logits: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  binary_cross_entropy_with_logits(logits: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  binary_cross_entropy_with_logits(logits: TensorData, target: TensorData, options?: LossReductionOptions): number;
  bceWithLogits(logits: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  bceWithLogits(logits: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  bceWithLogits(logits: TensorData, target: TensorData, options?: LossReductionOptions): number;
  bce_with_logits(logits: Tensor, target: TensorLike, options?: LossReductionOptions): Tensor<readonly [1]>;
  bce_with_logits(logits: TensorData, target: Tensor, options?: LossReductionOptions): Tensor<readonly [1]>;
  bce_with_logits(logits: TensorData, target: TensorData, options?: LossReductionOptions): number;
  bceWithLogitsLoss(options?: LossReductionOptions): BCEWithLogitsLoss;
  bce_with_logits_loss(options?: LossReductionOptions): BCEWithLogitsLoss;
  BCEWithLogitsLoss: new (options?: LossReductionOptions) => BCEWithLogitsLoss;
  classTargets(classes: readonly number[]): Uint32Array;
  crossEntropy(logits: Tensor, targets: IndexLike, options?: ClassLossOptions): Tensor<readonly [1]>;
  crossEntropy(logits: TensorData, targets: IndexLike, options?: ClassLossOptions): number;
  cross_entropy(logits: Tensor, targets: IndexLike, options?: ClassLossOptions): Tensor<readonly [1]>;
  cross_entropy(logits: TensorData, targets: IndexLike, options?: ClassLossOptions): number;
  crossEntropyLoss(options?: ClassLossOptions): CrossEntropyLoss;
  cross_entropy_loss(options?: ClassLossOptions): CrossEntropyLoss;
  CrossEntropyLoss: new (options?: ClassLossOptions) => CrossEntropyLoss;
  negativeLogLikelihood(logProbabilities: Tensor, targets: IndexLike, options?: ClassLossOptions): Tensor<readonly [1]>;
  negativeLogLikelihood(logProbabilities: TensorData, targets: IndexLike, options?: ClassLossOptions): number;
  negative_log_likelihood(logProbabilities: Tensor, targets: IndexLike, options?: ClassLossOptions): Tensor<readonly [1]>;
  negative_log_likelihood(logProbabilities: TensorData, targets: IndexLike, options?: ClassLossOptions): number;
  nllLoss(logProbabilities: Tensor, targets: IndexLike, options?: ClassLossOptions): Tensor<readonly [1]>;
  nllLoss(logProbabilities: TensorData, targets: IndexLike, options?: ClassLossOptions): number;
  nll_loss(logProbabilities: Tensor, targets: IndexLike, options?: ClassLossOptions): Tensor<readonly [1]>;
  nll_loss(logProbabilities: TensorData, targets: IndexLike, options?: ClassLossOptions): number;
  nllLossModule(options?: ClassLossOptions): NLLLoss;
  nll_loss_module(options?: ClassLossOptions): NLLLoss;
  NLLLoss: new (options?: ClassLossOptions) => NLLLoss;
}

export type PublicLossNamespace = Readonly<LossNamespace>;
export declare const loss: PublicLossNamespace;

export type OptimNamespace = Readonly<{
  sgd(target: OptimizerTarget, config?: SGDConfig): Optimizer<"sgd">;
  adam(target: OptimizerTarget, config?: AdamConfig): Optimizer<"adam">;
  adamW(target: OptimizerTarget, config?: AdamConfig): Optimizer<"adamw">;
  rmsprop(target: OptimizerTarget, config?: RMSpropConfig): Optimizer<"rmsprop">;
  adagrad(target: OptimizerTarget, config?: AdagradConfig): Optimizer<"adagrad">;
  SGD: new (target: OptimizerTarget, config?: SGDConfig) => Optimizer<"sgd">;
  Adam: new (target: OptimizerTarget, config?: AdamConfig) => Optimizer<"adam">;
  AdamW: new (target: OptimizerTarget, config?: AdamConfig) => Optimizer<"adamw">;
  RMSprop: new (target: OptimizerTarget, config?: RMSpropConfig) => Optimizer<"rmsprop">;
  Adagrad: new (target: OptimizerTarget, config?: AdagradConfig) => Optimizer<"adagrad">;
  zeroGrad(target: OptimizerTarget, options?: ZeroGradOptions): void;
  zero_grad(target: OptimizerTarget, options?: ZeroGradOptions): void;
  config<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>): OptimizerConfigSnapshot<Kind>;
  isOptimizerConfigSnapshot(snapshot: unknown): snapshot is OptimizerConfigSnapshot;
  requireOptimizerConfigSnapshot(snapshot: unknown): OptimizerConfigSnapshot;
  assertOptimizerConfigSnapshot(snapshot: unknown): OptimizerConfigSnapshot;
  assert_optimizer_config_snapshot(snapshot: unknown): OptimizerConfigSnapshot;
  matchesOptimizerConfigSnapshotSignature(snapshot: unknown, signature: unknown): boolean;
  matches_optimizer_config_snapshot_signature(snapshot: unknown, signature: unknown): boolean;
  optimizerConfigSnapshotSignature(snapshot: OptimizerConfigSnapshot): string;
  setLearningRate<T extends Optimizer>(optimizer: T, lr: number): T;
  set_lr<T extends Optimizer>(optimizer: T, lr: number): T;
  getLearningRate(optimizer: Optimizer): number;
  get_lr(optimizer: Optimizer): number;
  addParamGroup<T extends Optimizer>(optimizer: T, group: OptimizerParamGroupInput): T;
  add_param_group<T extends Optimizer>(optimizer: T, group: OptimizerParamGroupInput): T;
  stateDict<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>): OptimizerStateSnapshot<Kind>;
  state_dict<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>): OptimizerStateSnapshot<Kind>;
  isOptimizerStateSnapshot(snapshot: unknown): snapshot is OptimizerStateSnapshot;
  requireOptimizerStateSnapshot(snapshot: unknown): OptimizerStateSnapshot;
  assertOptimizerStateSnapshot(snapshot: unknown): OptimizerStateSnapshot;
  assert_optimizer_state_snapshot(snapshot: unknown): OptimizerStateSnapshot;
  matchesOptimizerStateSnapshotSignature(snapshot: unknown, signature: unknown): boolean;
  matches_optimizer_state_snapshot_signature(snapshot: unknown, signature: unknown): boolean;
  optimizerStateSnapshotSignature(snapshot: OptimizerStateSnapshot): string;
  loadStateDict<const Kind extends OptimizerStateKind, T extends Optimizer<Kind>>(optimizer: T, source: OptimizerStateDict<Kind>, options?: LoadStateDictOptions): T;
  load_state_dict<const Kind extends OptimizerStateKind, T extends Optimizer<Kind>>(optimizer: T, source: OptimizerStateDict<Kind>, options?: LoadStateDictOptions): T;
  stepLR<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: StepLRConfig): LRScheduler<"step-lr", Kind>;
  step_lr<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: StepLRConfig): LRScheduler<"step-lr", Kind>;
  StepLR: new <const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: StepLRConfig) => LRScheduler<"step-lr", Kind>;
  exponentialLR<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: LRSchedulerConfig): LRScheduler<"exponential-lr", Kind>;
  exponential_lr<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: LRSchedulerConfig): LRScheduler<"exponential-lr", Kind>;
  ExponentialLR: new <const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: LRSchedulerConfig) => LRScheduler<"exponential-lr", Kind>;
  cosineAnnealingLR<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: CosineAnnealingLRConfig): LRScheduler<"cosine-annealing-lr", Kind>;
  cosine_annealing_lr<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: CosineAnnealingLRConfig): LRScheduler<"cosine-annealing-lr", Kind>;
  CosineAnnealingLR: new <const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: CosineAnnealingLRConfig) => LRScheduler<"cosine-annealing-lr", Kind>;
  reduceLROnPlateau<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: ReduceLROnPlateauConfig): LRScheduler<"reduce-lr-on-plateau", Kind>;
  reduce_lr_on_plateau<const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: ReduceLROnPlateauConfig): LRScheduler<"reduce-lr-on-plateau", Kind>;
  ReduceLROnPlateau: new <const Kind extends OptimizerStateKind>(optimizer: Optimizer<Kind>, config?: ReduceLROnPlateauConfig) => LRScheduler<"reduce-lr-on-plateau", Kind>;
  lrScheduler: LRSchedulerNamespace;
  lr_scheduler: LRSchedulerNamespace;
  isLRSchedulerStateSnapshot(snapshot: unknown): snapshot is LRSchedulerStateSnapshot;
  requireLRSchedulerStateSnapshot(snapshot: unknown): LRSchedulerStateSnapshot;
  assertLRSchedulerStateSnapshot(snapshot: unknown): LRSchedulerStateSnapshot;
  assert_lr_scheduler_state_snapshot(snapshot: unknown): LRSchedulerStateSnapshot;
  matchesLRSchedulerStateSnapshotSignature(snapshot: unknown, signature: unknown): boolean;
  matches_lr_scheduler_state_snapshot_signature(snapshot: unknown, signature: unknown): boolean;
  lrSchedulerStateSnapshotSignature(snapshot: LRSchedulerStateSnapshot): string;
}>;
export type PublicOptimNamespace = Readonly<OptimNamespace>;
export declare const optim: PublicOptimNamespace;

export type PublicTrainNamespace = Readonly<TrainNamespace>;
export declare const train: PublicTrainNamespace;
export type PublicCheckpointNamespace = Readonly<CheckpointNamespace>;
export declare const checkpoint: PublicCheckpointNamespace;
export declare function save(snapshot: ZgmlCheckpoint, space?: string | number): string;
export declare function save(snapshot: ZgmlCheckpoint, path: string, space?: string | number): string;
export declare function load(textOrPath: string): ZgmlCheckpoint;
export declare function load<const OptimizerKind extends OptimizerStateKind>(textOrPath: string, targets: CheckpointRestoreOptions<OptimizerKind>): CheckpointRestoreOptions<OptimizerKind>;

export type PublicTorchNamespace = Readonly<{
  Tensor: typeof Tensor;
  tensor: typeof tensor;
  asTensor: typeof asTensor;
  as_tensor: typeof as_tensor;
  asarray: typeof asarray;
  fromNumpy: typeof fromNumpy;
  from_numpy: typeof from_numpy;
  parameter: typeof parameter;
  param: typeof param;
  empty: typeof empty;
  emptyLike: typeof emptyLike;
  empty_like: typeof empty_like;
  zeros: typeof zeros;
  zerosLike: typeof zerosLike;
  zeros_like: typeof zeros_like;
  ones: typeof ones;
  onesLike: typeof onesLike;
  ones_like: typeof ones_like;
  full: typeof full;
  fullLike: typeof fullLike;
  full_like: typeof full_like;
  eye: typeof eye;
  scalar: typeof scalar;
  rand: typeof rand;
  randLike: typeof randLike;
  rand_like: typeof rand_like;
  randn: typeof randn;
  randnLike: typeof randnLike;
  randn_like: typeof randn_like;
  randint: typeof randint;
  randInt: typeof randInt;
  randperm: typeof randperm;
  randPerm: typeof randPerm;
  manual_seed: typeof manual_seed;
  manualSeed: typeof manualSeed;
  initialSeed: typeof initialSeed;
  initial_seed: typeof initial_seed;
  seededRng: typeof seededRng;
  arange: typeof arange;
  linspace: typeof linspace;
  cat: typeof cat;
  concat: typeof concat;
  concatenate: typeof concatenate;
  stack: typeof stack;
  vstack: typeof vstack;
  hstack: typeof hstack;
  einsum: typeof einsum;
  broadcastTo: typeof broadcastTo;
  expand: typeof expand;
  repeat: typeof repeat;
  tile: typeof tile;
  scatterAdd: typeof scatterAdd;
  scatter_add: typeof scatter_add;
  add: typeof add;
  sub: typeof sub;
  mul: typeof mul;
  div: typeof div;
  eq: typeof eq;
  ne: typeof ne;
  lt: typeof lt;
  le: typeof le;
  gt: typeof gt;
  ge: typeof ge;
  isclose: typeof isclose;
  matmul: typeof matmul;
  mm: typeof mm;
  dot: typeof dot;
  trace: typeof trace;
  diagonal: typeof diagonal;
  bmm: typeof bmm;
  clone: typeof clone;
  detach: typeof detach;
  relu<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  gelu<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  silu<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  sigmoid<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  tanh<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  softmax(input: Tensor, dim?: number): Tensor;
  softmax_dim(input: Tensor, dim?: number): Tensor;
  softmaxDim(input: Tensor, dim?: number): Tensor;
  logSoftmax(input: Tensor, dim?: number): Tensor;
  log_softmax(input: Tensor, dim?: number): Tensor;
  log_softmax_dim(input: Tensor, dim?: number): Tensor;
  logSoftmaxDim(input: Tensor, dim?: number): Tensor;
  to: typeof to;
  cpu: typeof cpu;
  float: typeof float;
  float32: typeof float32;
  typeAs: typeof typeAs;
  type_as: typeof type_as;
  sum(input: Tensor, dim?: number): Tensor;
  prod(input: Tensor, dim?: number): Tensor;
  cumsum(input: Tensor, dim?: number): Tensor;
  mean(input: Tensor, dim?: number): Tensor;
  max(input: Tensor, dim?: number): Tensor;
  min(input: Tensor, dim?: number): Tensor;
  argmax(input: Tensor, dim?: number): Tensor;
  argmin(input: Tensor, dim?: number): Tensor;
  any(input: Tensor, dim?: number): Tensor;
  all(input: Tensor, dim?: number): Tensor;
  logsumexp(input: Tensor, dim?: number): Tensor;
  logSumExp(input: Tensor, dim?: number): Tensor;
  variance(input: Tensor, dim?: number, correction?: number): Tensor;
  var(input: Tensor, dim?: number, correction?: number): Tensor;
  std(input: Tensor, dim?: number, correction?: number): Tensor;
  norm(input: Tensor, dim?: number, p?: number): Tensor;
  neg<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  negative<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  expm1<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  log1p<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  sqr<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  square<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  recip<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  reciprocal<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  sgn<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  sign<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  step<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  isnan<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  isinf<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  isfinite<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  floor<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  ceil<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  round<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  trunc<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  sin<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  cos<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  tan<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  sqrt<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  rsqrt<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  exp<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  log<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  abs<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<Shape>;
  pow<const Shape extends TensorShapeTuple>(input: Tensor<Shape>, exponent: number): Tensor<Shape>;
  clamp<const Shape extends TensorShapeTuple>(input: Tensor<Shape>, min?: number | null, max?: number | null): Tensor<Shape>;
  clip<const Shape extends TensorShapeTuple>(input: Tensor<Shape>, min?: number | null, max?: number | null): Tensor<Shape>;
  flatten(input: Tensor, startDim?: number, endDim?: number): Tensor;
  reshape<const Shape extends TensorShapeTuple>(input: Tensor, shape: Shape): Tensor<Shape>;
  view<const Shape extends TensorShapeTuple>(input: Tensor, shape: Shape): Tensor<Shape>;
  squeeze<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<SqueezeShape<Shape>>;
  squeeze<const Shape extends TensorShapeTuple, const Dim extends number>(input: Tensor<Shape>, dim: Dim): Tensor<SqueezeShape<Shape, Dim>>;
  squeeze(input: Tensor, dim?: number | null): Tensor;
  unsqueeze<const Shape extends TensorShapeTuple, const Dim extends number>(input: Tensor<Shape>, dim: Dim): Tensor<UnsqueezeShape<Shape, Dim>>;
  transpose<const Shape extends TensorShapeTuple>(input: Tensor<Shape>): Tensor<TransposeShape<Shape>>;
  transpose<const Shape extends TensorShapeTuple, const Dim0 extends number, const Dim1 extends number>(input: Tensor<Shape>, dim0: Dim0, dim1: Dim1): Tensor<TransposeShape<Shape, Dim0, Dim1>>;
  transpose(input: Tensor, dim0?: number, dim1?: number): Tensor;
  permute<const Shape extends TensorShapeTuple, const Dims extends readonly number[]>(input: Tensor<Shape>, dims: Dims): Tensor<PermuteShape<Shape, Dims>>;
  permute(input: Tensor, dims: readonly number[]): Tensor;
  flip: typeof flip;
  roll: typeof roll;
  select: typeof select;
  narrow: typeof narrow;
  slice: typeof slice;
  indexSelect: typeof indexSelect;
  index_select: typeof index_select;
  gather: typeof gather;
  take: typeof take;
  unbind: typeof unbind;
  maximum<const Shape extends TensorShapeTuple>(input: Tensor<Shape>, other: number): Tensor<Shape>;
  maximum<const Shape extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<Shape>, other: Tensor<OtherShape>): Tensor<BroadcastShape<Shape, OtherShape>>;
  maximum(input: Tensor, other: TensorLike): Tensor;
  minimum<const Shape extends TensorShapeTuple>(input: Tensor<Shape>, other: number): Tensor<Shape>;
  minimum<const Shape extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(input: Tensor<Shape>, other: Tensor<OtherShape>): Tensor<BroadcastShape<Shape, OtherShape>>;
  minimum(input: Tensor, other: TensorLike): Tensor;
  where<const Shape extends TensorShapeTuple>(condition: Tensor<Shape>, input: number, other: number): Tensor<Shape>;
  where<const Shape extends TensorShapeTuple, const InputShape extends TensorShapeTuple, const OtherShape extends TensorShapeTuple>(condition: Tensor<Shape>, input: Tensor<InputShape>, other: Tensor<OtherShape>): Tensor<WhereShape<Shape, InputShape, OtherShape>>;
  where(condition: Tensor, input: TensorLike, other: TensorLike): Tensor;
  maskedFill<const Shape extends TensorShapeTuple>(input: Tensor<Shape>, mask: TensorLike, value: number): Tensor<Shape>;
  maskedFill(input: Tensor, mask: TensorLike, value: TensorLike): Tensor;
  masked_fill<const Shape extends TensorShapeTuple>(input: Tensor<Shape>, mask: TensorLike, value: number): Tensor<Shape>;
  masked_fill(input: Tensor, mask: TensorLike, value: TensorLike): Tensor;
  allclose(actual: Tensor, expected: TensorLike, options?: AllCloseOptions): boolean;
  equal(actual: Tensor, expected: TensorLike): boolean;
  argsort<const Shape extends TensorShapeTuple>(input: Tensor<Shape>, dim?: number, descending?: boolean): Tensor<Shape>;
  sort<const Shape extends TensorShapeTuple>(input: Tensor<Shape>, dim?: number, descending?: boolean): TensorSortResult<Shape>;
  topk<const Shape extends TensorShapeTuple, const K extends number>(input: Tensor<Shape>, k: K): TensorTopkResult<TensorTopkShape<Shape, -1, K>>;
  topk<const Shape extends TensorShapeTuple, const K extends number, const Dim extends number>(input: Tensor<Shape>, k: K, dim: Dim, largest?: boolean, sorted?: boolean): TensorTopkResult<TensorTopkShape<Shape, Dim, K>>;
  topk(input: Tensor, k: number, dim?: number, largest?: boolean, sorted?: boolean): TensorTopkResult;
  split(input: Tensor, splitSizeOrSections: number | readonly number[], dim?: number): readonly Tensor[];
  chunk(input: Tensor, chunks: number, dim?: number): readonly Tensor[];
  hasShape: typeof hasShape;
  requireShape: typeof requireShape;
  no_grad: typeof no_grad;
  noGrad: typeof noGrad;
  inference_mode: typeof inference_mode;
  inferenceMode: typeof inferenceMode;
  enable_grad: typeof enable_grad;
  enableGrad: typeof enableGrad;
  is_grad_enabled: typeof is_grad_enabled;
  isGradEnabled: typeof isGradEnabled;
  set_grad_enabled: typeof set_grad_enabled;
  setGradEnabled: typeof setGradEnabled;
  gradMode: GradModeNamespace;
  nn: PublicNnNamespace;
  F: NnFunctionalNamespace;
  functional: NnFunctionalNamespace;
  compile: PublicCompileNamespace;
  native: PublicCompileNamespace["compileForInference"];
  inference: PublicCompileNamespace["compileForInference"];
  forInference: PublicCompileNamespace["compileForInference"];
  for_inference: PublicCompileNamespace["compile_for_inference"];
  compileInference: PublicCompileNamespace["compileForInference"];
  compile_inference: PublicCompileNamespace["compile_for_inference"];
  compileForInference: PublicCompileNamespace["compileForInference"];
  compile_for_inference: PublicCompileNamespace["compile_for_inference"];
  run: PublicCompileNamespace["run"];
  infer: PublicCompileNamespace["infer"];
  predict: PublicCompileNamespace["predict"];
  runInto: PublicCompileNamespace["runInto"];
  run_into: PublicCompileNamespace["run_into"];
  inferInto: PublicCompileNamespace["inferInto"];
  infer_into: PublicCompileNamespace["infer_into"];
  predictInto: PublicCompileNamespace["predictInto"];
  predict_into: PublicCompileNamespace["predict_into"];
  trainingStep: PublicCompileNamespace["trainingStep"];
  training_step: PublicCompileNamespace["training_step"];
  compileForTraining: PublicCompileNamespace["compileForTraining"];
  compile_for_training: PublicCompileNamespace["compile_for_training"];
  forTraining: PublicCompileNamespace["forTraining"];
  for_training: PublicCompileNamespace["for_training"];
  nativeEager: PublicNativeEagerNamespace;
  native_eager: PublicNativeEagerNamespace;
  nativeCore: NativeCoreFunction;
  native_core: NativeCoreFunction;
  lazy: PublicLazyNamespace;
  optim: PublicOptimNamespace;
  data: PublicDataNamespace;
  utils: Readonly<{
    data: Readonly<{
      Dataset: PublicDataNamespace["Dataset"];
      TensorDataset: PublicDataNamespace["TensorDataset"];
      DataLoader: PublicDataNamespace["DataLoader"];
      Subset: PublicDataNamespace["Subset"];
      ConcatDataset: PublicDataNamespace["ConcatDataset"];
      MapDataset: PublicDataNamespace["MapDataset"];
      SequentialSampler: PublicDataNamespace["SequentialSampler"];
      RandomSampler: PublicDataNamespace["RandomSampler"];
      BatchSampler: PublicDataNamespace["BatchSampler"];
      tensorDataset: PublicDataNamespace["tensorDataset"];
      tensor_dataset: PublicDataNamespace["tensor_dataset"];
      defaultCollate: PublicDataNamespace["defaultCollate"];
      default_collate: PublicDataNamespace["default_collate"];
      dataLoader: PublicDataNamespace["dataLoader"];
      dataloader: PublicDataNamespace["dataloader"];
      randomSplit: PublicDataNamespace["randomSplit"];
      random_split: PublicDataNamespace["random_split"];
      subset: PublicDataNamespace["subset"];
      take: PublicDataNamespace["take"];
      concatDataset: PublicDataNamespace["concatDataset"];
      concat_dataset: PublicDataNamespace["concat_dataset"];
      mapDataset: PublicDataNamespace["mapDataset"];
      map_dataset: PublicDataNamespace["map_dataset"];
    }>;
  }>;
  loss: PublicLossNamespace;
  train: PublicTrainNamespace;
  fit: PublicTrainNamespace["fit"];
  fitModule: PublicTrainNamespace["fitModule"];
  fit_module: PublicTrainNamespace["fit_module"];
  explainNative: PublicTrainNamespace["explainNative"];
  explain_native: PublicTrainNamespace["explain_native"];
  nativeTrainingPlan: PublicTrainNamespace["nativePlan"];
  native_training_plan: PublicTrainNamespace["native_plan"];
  fitNative: PublicTrainNamespace["fitNative"];
  fit_native: PublicTrainNamespace["fit_native"];
  checkpoint: PublicCheckpointNamespace;
  save(snapshot: ZgmlCheckpoint, space?: string | number): string;
  save(snapshot: ZgmlCheckpoint, path: string, space?: string | number): string;
  load(textOrPath: string): ZgmlCheckpoint;
  load<const OptimizerKind extends OptimizerStateKind>(textOrPath: string, targets: CheckpointRestoreOptions<OptimizerKind>): CheckpointRestoreOptions<OptimizerKind>;
  Program: typeof Program;
  Session: typeof Session;
  NativeBuffer: typeof NativeBuffer;
}>;
export type PublicZgmlNamespace = PublicTorchNamespace;
export type PublicSimpleNamespace = Readonly<Pick<PublicZgmlNamespace,
  | "Tensor"
  | "tensor"
  | "nn"
  | "F"
  | "functional"
  | "compile"
  | "native"
  | "inference"
  | "forInference"
  | "compileInference"
  | "compileForInference"
  | "run"
  | "infer"
  | "predict"
  | "runInto"
  | "inferInto"
  | "predictInto"
  | "trainingStep"
  | "compileForTraining"
  | "forTraining"
  | "nativeCore"
  | "lazy"
  | "optim"
  | "data"
  | "loss"
  | "train"
  | "fit"
  | "fitModule"
  | "fit_module"
  | "fitNative"
  | "fit_native"
  | "checkpoint"
  | "save"
  | "load"
  | "noGrad"
  | "no_grad"
  | "inferenceMode"
  | "inference_mode"
>>;
export declare const simple: PublicSimpleNamespace;
export declare const zgml: PublicZgmlNamespace;
export declare const torch: PublicTorchNamespace;
export declare const native: PublicCompileNamespace["compileForInference"];
export declare const inference: PublicCompileNamespace["compileForInference"];
export declare const forInference: PublicCompileNamespace["compileForInference"];
export declare const for_inference: PublicCompileNamespace["compile_for_inference"];
export declare const compileInference: PublicCompileNamespace["compileForInference"];
export declare const compile_inference: PublicCompileNamespace["compile_for_inference"];
export declare const compileForInference: PublicCompileNamespace["compileForInference"];
export declare const compile_for_inference: PublicCompileNamespace["compile_for_inference"];
export declare const run: PublicCompileNamespace["run"];
export declare const infer: PublicCompileNamespace["infer"];
export declare const predict: PublicCompileNamespace["predict"];
export declare const runInto: PublicCompileNamespace["runInto"];
export declare const run_into: PublicCompileNamespace["run_into"];
export declare const inferInto: PublicCompileNamespace["inferInto"];
export declare const infer_into: PublicCompileNamespace["infer_into"];
export declare const predictInto: PublicCompileNamespace["predictInto"];
export declare const predict_into: PublicCompileNamespace["predict_into"];
export declare const trainingStep: PublicCompileNamespace["trainingStep"];
export declare const training_step: PublicCompileNamespace["training_step"];
export declare const compileForTraining: PublicCompileNamespace["compileForTraining"];
export declare const compile_for_training: PublicCompileNamespace["compile_for_training"];
export declare const forTraining: PublicCompileNamespace["forTraining"];
export declare const for_training: PublicCompileNamespace["for_training"];
export declare const fit: PublicTrainNamespace["fit"];
export declare const fitModule: PublicTrainNamespace["fitModule"];
export declare const fit_module: PublicTrainNamespace["fit_module"];
export declare const explainNative: PublicTrainNamespace["explainNative"];
export declare const explain_native: PublicTrainNamespace["explain_native"];
export declare const nativeTrainingPlan: PublicTrainNamespace["nativePlan"];
export declare const native_training_plan: PublicTrainNamespace["native_plan"];
export declare const fitNative: PublicTrainNamespace["fitNative"];
export declare const fit_native: PublicTrainNamespace["fit_native"];
export declare const nativeEager: PublicNativeEagerNamespace;
export declare const native_eager: PublicNativeEagerNamespace;
export declare const nativeCore: NativeCoreFunction;
export declare const native_core: NativeCoreFunction;

export type TinyLinearDesc = {
  inputLen: number;
  outputLen: number;
  hiddenLen?: number;
  activation?: number;
  weightsLen?: number;
  biasLen?: number;
};

export type TinyMlpDesc = TinyLinearDesc & {
  hiddenLen: number;
  activation: number;
  weightsLen: number;
  biasLen: number;
};

export type ProgramHostBinding<Shape extends TensorShapeTuple = TensorShapeTuple> = TensorLike<Shape>;
export type ProgramInputBinding<Shape extends TensorShapeTuple = TensorShapeTuple> = TensorLike<Shape> | Uint32Array | Int32Array;
export type ProgramOutputBinding<Shape extends TensorShapeTuple = TensorShapeTuple> = Tensor<Shape> | Float32Array | NativeBuffer;
export type CompiledInference<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = Readonly<{
  native: true;
  program: Program<InputShape, OutputShape>;
  session: Session<InputShape, OutputShape>;
  executionPlan(): ProgramExecutionPlan<InputShape, OutputShape>;
  requireExecutionPlan(): ProgramExecutionPlan<InputShape, OutputShape>;
  explain(): ModuleCompileExplanation<InputShape, OutputShape> | ModuleCompileSupport<InputShape, OutputShape>;
  preflight(): ModuleCompileExplanation<InputShape, OutputShape> | ModuleCompileSupport<InputShape, OutputShape>;
  compileSupport(): ModuleCompileSupport<InputShape, OutputShape>;
  compileEvidence(): ProgramCompileEvidence | null;
  parameterBindingPlan(): ModuleBindingPlan<InputShape, OutputShape> | null;
  parameter_binding_plan(): ModuleBindingPlan<InputShape, OutputShape> | null;
  programBindingPlan(): ProgramBindingPlan<InputShape, OutputShape>;
  program_binding_plan(): ProgramBindingPlan<InputShape, OutputShape>;
  requirements(): ProgramRequirements;
  bufferLayout(): ProgramBufferLayout;
  bufferSlotNames(): readonly string[];
  bufferSlot(nameOrKind: ProgramDeviceBufferKind | string): ProgramBufferLayoutSlot | null;
  sessionBufferLayout(): ProgramBufferLayout;
  sessionBufferSlotNames(): readonly string[];
  sessionBufferSlot(nameOrKind: ProgramDeviceBufferKind | string): ProgramBufferLayoutSlot | null;
  stepContract(): SessionStepContract;
  hotPathPlan(params?: unknown): SessionExecutionPlan<InputShape, OutputShape>;
  inputShape(): InputShape;
  outputShape(): OutputShape;
  kernelPlan(): ModuleKernelPlan | null;
  compilerSignatures(): ModuleCompleteCompilerSignatures | null;
  forward(input: ProgramInputBinding<InputShape>): Tensor<OutputShape>;
  call(input: ProgramInputBinding<InputShape>): Tensor<OutputShape>;
  __call__(input: ProgramInputBinding<InputShape>): Tensor<OutputShape>;
  stepTensor(input: ProgramInputBinding<InputShape>): Tensor<OutputShape>;
  into(output: Float32Array, input: ProgramInputBinding<InputShape>): Float32Array;
  prepareInto(output: Float32Array, input: ProgramInputBinding<InputShape>): () => Float32Array;
  dispose(): void;
  free(): void;
}>;
declare const moduleBindingsBrand: unique symbol;
export type ProgramBindings<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = {
  weights?: ProgramHostBinding | NativeBuffer;
  bias?: ProgramHostBinding | NativeBuffer;
  input?: ProgramInputBinding<InputShape> | NativeBuffer;
  output?: ProgramOutputBinding<OutputShape>;
  readonly inputShape?: InputShape;
  readonly outputShape?: OutputShape;
};
export type ModuleBindings<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = Readonly<ProgramBindings<InputShape, OutputShape>> & {
  readonly [moduleBindingsBrand]: "ModuleBindings";
  readonly inputShape?: InputShape;
  readonly outputShape?: OutputShape;
};
export type ProgramBindingDiagnostic = Readonly<{
  code: "binding-invalid";
  message: string;
}>;
export type ProgramBindingMode = "host" | "native" | "invalid";
export type ProgramBindingPlan<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = Readonly<{
  kind: "zgml.program.binding-plan";
  signature: string;
  accepted: boolean;
  canBind: boolean;
  mode: ProgramBindingMode;
  reason: string | null;
  diagnostics: readonly ProgramBindingDiagnostic[];
  bufferLayout: ProgramBufferLayout | null;
  inputShape: InputShape | null;
  outputShape: OutputShape | null;
  weightsLen: number;
  biasLen: number;
  inputLen: number | null;
  outputLen: number | null;
  hasWeights: boolean;
  hasBias: boolean;
  hasInput: boolean;
  hasOutput: boolean;
  usesNativeBuffers: boolean;
}>;
export type ModuleBindingPlacementMode = "raw" | "empty" | "host" | "native";
export type ModuleBindingPlan<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = Readonly<{
  kind: "zgml.nn.module-bindings-plan";
  signature: string;
  moduleBindings: boolean;
  placementMode: ModuleBindingPlacementMode;
  usesNativeBuffers: boolean;
  nativeSlots: readonly string[];
  hostSlots: readonly string[];
  options: Readonly<CompileOptions>;
  support: ModuleCompileSupport | null;
  supported: boolean | null;
  reason: string | null;
  inputShape: InputShape | null;
  outputShape: OutputShape | null;
  parameterNames: readonly string[];
  parameterInfos: readonly ModuleParameterInfo[];
  diagnostics: readonly ModuleCompileDiagnostic[];
}>;
export type TinyLinearWeights = ProgramBindings;
export type ModuleParameterPlacementOptions = CompileOptions & ProgramCreateBufferOptions & {
  weights?: ProgramCreateBufferOptions;
  weight?: ProgramCreateBufferOptions;
  bias?: ProgramCreateBufferOptions;
};

export type CompileMode = "auto" | "tiny" | "module";
export type CompileOptions = {
  backend?: ZgmlBackend;
  mode?: CompileMode;
  contextLength?: number;
  batch?: number;
  inputShape?: readonly number[];
};
export type CompileOptionsWithInputShape<S extends TensorShapeTuple> = CompileOptions & { inputShape: S };
export type EmbeddingCompileOptions<S extends TensorShapeTuple = TensorShapeTuple> = CompileOptions & { inputShape: S };

export type LoadModelKind =
  | "auto"
  | "llama"
  | "llama-auto"
  | "tiny-llama"
  | "tinyllama"
  | "tiny-llama-2layer"
  | "tiny-llama-2-layer"
  | "tinyllama2layer"
  | "smollm"
  | "smollm135m"
  | "smollm-135m";

export type LoadModelOptions = {
  kind?: LoadModelKind;
  modelKind?: LoadModelKind;
};
export type LoadModelSource = string | Uint8Array | ArrayBuffer;
export type TokenIds = Uint32Array | readonly number[];
export type TokenWindowOptions = {
  tokensLen?: number;
  tokenLength?: number;
  activeTokenLength?: number;
  activeTokenCount?: number;
};
export type ExecuteTokensOptions = TokenWindowOptions & {
  output?: Float32Array | false;
};
export type LlamaLogitsTensorOptions = TensorOptions & {
  shape?: TensorShape;
  output?: Tensor | Float32Array;
};
export type LlamaTokenWindowTensorOptions = TokenWindowOptions & LlamaLogitsTensorOptions;
export type LlamaExecuteParams = ExecuteTokensOptions & (
  | { token: number; tokens?: never }
  | { tokens: TokenIds; token?: never }
);
export type LlamaExecuteTensorParams = LlamaTokenWindowTensorOptions & (
  | { token: number; tokens?: never }
  | { tokens: TokenIds; token?: never }
);
export type LlamaExecuteIntoParams = TokenWindowOptions & (
  | { token: number; tokens?: never }
  | { tokens: TokenIds; token?: never }
);
export type LlamaStepParams = {
  token: number;
};
export type LlamaSessionStepContract = Readonly<{
  kind: "llama-session-step-contract";
  signature: string;
  scalarType: "f32";
  scalarBytes: 4;
  tokenIdBytes: 4;
  logitsLen: number;
  outputLen: number;
  outputByteLength: number;
  outputSlotName: "output";
  outputSlotRole: "step-output";
  outputShape: readonly number[];
  contextLength: number;
  position: number;
  remainingContext: number;
  boundOutput: Exclude<SessionBoundBufferKind, "host">;
  defaultOutput: Exclude<SessionDefaultOutputKind, "bound-host">;
  defaultReadbackRequired: boolean;
  defaultAllocationFree: boolean;
  defaultHotPath: boolean;
  defaultOutputEffect: SessionStepParamsOutputEffect;
  defaultOutputOwnership: SessionStepParamsOutputOwnership;
  defaultOutputReturnOwnership: SessionStepParamsOutputReturnOwnership;
  hasBoundOutput: boolean;
  canReadOutput: boolean;
  acceptsToken: true;
  acceptsTokens: true;
  acceptsInlineOutput: true;
  acceptsNoOutput: true;
  noOutputEffect: SessionNoOutputEffect;
}>;
export type TokenArgmaxResult = Readonly<{
  token: number;
  logit: number;
}>;
export type TokenSampleOptions = TokenWindowOptions & {
  topK?: number;
  top_k?: number;
  temperature?: number;
  seed?: number;
};
export type TokenSampleResult = Readonly<{
  token: number;
  logit: number;
}>;
export type TokenGenerateArgmaxResult = Readonly<{
  tokens: Uint32Array;
  lastToken: number;
  lastLogit: number;
}>;
export type TokenGenerateSampleResult = Readonly<{
  tokens: Uint32Array;
  lastToken: number;
  lastLogit: number;
}>;

export type RuntimeInfo = Readonly<{
  abiVersion: number;
  sizeTBytes: number;
  pointerBytes: number;
  tokenIdBytes: number;
  featureFlags: bigint;
  features: RuntimeFeatures;
}>;

export type RuntimeFeatures = Readonly<{
  bufferHandle: boolean;
  modelAuto: boolean;
  runtimeProfile: boolean;
  webGpuCompileOnly: boolean;
  wasmExports: boolean;
  nativeBufferIo: boolean;
  nativeArgmax: boolean;
  nativeExecuteArgmax: boolean;
  nativeTopKSample: boolean;
  programRequirements: boolean;
  programOutputBuffer: boolean;
  nativeGenerateSample: boolean;
  nativeGenerateArgmax: boolean;
  sessionModelBinding: boolean;
  externalBuffer: boolean;
  externalResourceBuffer: boolean;
  llamaKvResourceBinding: boolean;
  llamaKvCacheRequirements: boolean;
  programBufferFactory: boolean;
  externalResourceAccess: boolean;
  modelInspection: boolean;
  programModelCompatibility: boolean;
  bufferInspection: boolean;
  sessionInspection: boolean;
  programResourceInspection: boolean;
  programMemoryInspection: boolean;
  programShapeInspection: boolean;
  programPatchEnvelopeInspection: boolean;
  sessionBindingShapeInspection: boolean;
  nativeWgpuExecution: boolean;
  programDeviceBuffer: boolean;
  programDeviceBufferImport: boolean;
  programDispatchPlanInspection: boolean;
  abiStructSize: boolean;
  experimentalLlamaWgpuExecution: boolean;
  modelPathProbe: boolean;
  supportedCheckpoints: boolean;
  safetensorsHeaderProbe: boolean;
  safetensorsDataLoad: boolean;
  safetensorsDataProbe: boolean;
  nativeTinyMlp: boolean;
  nativeModuleProgram: boolean;
  programBindingRequirements: boolean;
  sessionPersistentUpload: boolean;
  nativeModuleActivationChain: boolean;
  nativeEagerLinear: boolean;
  nativeEagerLinearActivation: boolean;
  nativeTrainingStep: boolean;
  nativeEagerSoftmax: boolean;
  nativeEagerMatmul: boolean;
  nativeEagerBmm: boolean;
  nativeEagerElementwise: boolean;
  nativeEagerElementwiseBroadcast: boolean;
  nativeEagerReduce: boolean;
  nativeEagerDot: boolean;
  nativeEagerConv2d: boolean;
  nativeEagerPool2d: boolean;
}>;

export type ModelInspection = Readonly<{
  kind: "zgml.model.inspection";
  signature: string;
  modelKind: ProgramRequirements["modelKind"];
  inputLen: number;
  outputLen: number;
  vocabSize: number;
  maxSeqLen: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  nKvHeads: number;
  dFF: number;
  ropeBase: number;
  rmsNormEps: number;
  tiedLmHead: boolean;
}>;

export type ProgramRequirements = Readonly<{
  kind: "zgml.program.requirements";
  signature: string;
  modelKind: "tiny-linear" | "tiny-mlp" | "module" | "tiny-llama" | "tiny-llama-2layer" | "smollm-135m" | `unknown:${number}`;
  scalarBytes: number;
  tokenIdBytes: number;
  inputLen: number;
  inputByteLength: number;
  outputLen: number;
  weightsLen: number;
  weightsByteLength: number;
  biasLen: number;
  biasByteLength: number;
  parameterLen: number;
  parameterByteLength: number;
  logitsLen: number;
  outputByteLength: number;
  contextLen: number;
  batch: number;
  maxTokenWindow: number;
}>;

export type ProgramBufferSizing = Readonly<{
  kind: "program-buffer-sizing";
  signature: string;
  scalarType: "f32";
  scalarBytes: number;
  inputLen: number;
  inputByteLength: number;
  outputLen: number;
  outputByteLength: number;
  weightsLen: number;
  weightsByteLength: number;
  biasLen: number;
  biasByteLength: number;
  parameterLen: number;
  parameterByteLength: number;
  modelKind: ProgramRequirements["modelKind"];
}>;

export type SessionBufferSizing = Readonly<Omit<ProgramBufferSizing, "kind"> & {
  kind: "session-buffer-sizing";
}>;

export type ProgramInspection = Readonly<{
  kind: "zgml.program.inspection";
  signature: string;
  backend: ZgmlBackend | `unknown:${number}`;
  executionSupported: boolean;
  externalResourcesSupported: boolean;
  bufferCount: number;
  bufferElementCount: number;
  bufferByteLength: number;
  initialUploadCount: number;
  qweightCount: number;
  opCount: number;
  commandCount: number;
  commandStencilHash: bigint;
  runtimePatchHoles: number;
  runtimePatchCacheWritePosHoles: number;
  runtimePatchAttentionSeqKvHoles: number;
  runtimePatchMaxCacheWritePos: number | null;
  runtimePatchMaxAttentionSeqKv: number | null;
  runtimePatchStencilHash: bigint;
  commandOpCount: number;
  commandRowCount: number;
  commandProjectionCount: number;
  commandAttentionCount: number;
  commandMovementCount: number;
  commandElementwiseCount: number;
  commandRopeCount: number;
  bindingRequirementHash: bigint;
  persistentRequirementCount: number;
  stepInputRequirementCount: number;
  stepOutputRequirementCount: number;
  backendDispatchCount: number;
  dispatchPlanSupported: boolean;
  dispatchPlanCoveredOpCount: number;
  dispatchPlanFirstUnsupportedOp: number | null;
  dispatchPlanProjectionCount: number;
  dispatchPlanRowCount: number;
  dispatchPlanAttentionCount: number;
  dispatchPlanMovementCount: number;
  dispatchPlanElementwiseCount: number;
  dispatchPlanRopeCount: number;
  dispatchPlanQuantizedProjectionCount: number;
  diagnostics: readonly ProgramRuntimeDiagnostic[];
}>;
export type ProgramExecutionMode = "executable" | "resource-probe" | "compile-only";
export type ProgramExecutionUnavailableDiagnostic = Readonly<{
  code: "execution-unavailable";
  message: string;
  backend: ProgramInspection["backend"];
  mode: ProgramExecutionMode;
}>;
export type ProgramDispatchPlanDiagnostic = Readonly<{
  code:
    | "dispatch-plan-unavailable"
    | "dispatch-plan-unsupported-op"
    | "dispatch-plan-incomplete";
  message: string;
  backend: ProgramInspection["backend"];
  opCount: number;
  coveredOpCount: number;
  firstUnsupportedOp: number | null;
}>;
export type ProgramRuntimeDiagnostic =
  | ProgramExecutionUnavailableDiagnostic
  | ProgramDispatchPlanDiagnostic;

export type ProgramExecutionCapabilities = Readonly<{
  kind: "zgml.program.capabilities";
  backend: ProgramInspection["backend"];
  mode: ProgramExecutionMode;
  signature: string;
  canExecute: boolean;
  canBindExternalResources: boolean;
  hasFullDispatchPlan: boolean;
  backendDispatchCount: number;
  dispatchPlanSupported: boolean;
  dispatchPlanCoveredOpCount: number;
  dispatchPlanFirstUnsupportedOp: number | null;
  dispatchPlanProjectionCount: number;
  dispatchPlanRowCount: number;
  dispatchPlanAttentionCount: number;
  dispatchPlanMovementCount: number;
  dispatchPlanElementwiseCount: number;
  dispatchPlanRopeCount: number;
  dispatchPlanQuantizedProjectionCount: number;
  diagnostics: readonly ProgramRuntimeDiagnostic[];
}>;
export type ProgramExecutionPlan<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> = Readonly<{
  kind: "zgml.program.execution-plan";
  signature: string;
  programKind: "generic" | "llama";
  requirements: ProgramRequirements;
  capabilities: ProgramExecutionCapabilities;
  compileEvidence: ProgramCompileEvidence | null;
  canExecute: boolean;
  executionMode: ProgramExecutionMode;
  canBindExternalResources: boolean;
  hasFullDispatchPlan: boolean;
  capabilitySignature: string;
  bufferSizing: ProgramBufferSizing;
  bufferLayout: ProgramBufferLayout;
  inputShape: InputShape | null;
  outputShape: OutputShape;
  trace: ModuleProgramTrace | null;
  tensorProgramIr: ModuleTensorProgramIr | null;
  kernelPlan: ModuleKernelPlan | null;
  memoryLayout: ModuleKernelMemoryLayout | null;
  shapeConstraints: ModuleKernelShapeConstraints | null;
  parameterLayout: ModuleKernelParameterLayout | null;
  parameterNames: readonly string[];
  parameterInfos: readonly ModuleKernelParameterLayoutEntry[];
  moduleCompatibility: ProgramModuleCompatibility | null;
  acceptsModule: boolean | null;
  diagnostics: readonly ProgramRuntimeDiagnostic[];
}>;

export type ProgramModelCompatibility = Readonly<{
  kind: "zgml.program.model-compatibility";
  signature: string;
  programModelKind: ProgramRequirements["modelKind"];
  modelKind: ProgramRequirements["modelKind"];
  compatible: boolean;
}>;

export type BufferInspection = Readonly<{
  kind: "zgml.buffer.inspection";
  signature: string;
  storage: "host" | "external-resource" | `unknown:${number}`;
  byteLength: number;
  placement: ZgmlBackend | `unknown:${number}` | null;
  accessFlags: number;
  access: Readonly<{ read: boolean; write: boolean }> | null;
  handle: number;
  byteOffset: number;
  resourceByteLength: number;
}>;

export type SessionInspection = Readonly<{
  kind: "zgml.session.inspection";
  signature: string;
  modelKind: ProgramRequirements["modelKind"];
  backend: ZgmlBackend | `unknown:${number}`;
  outputStorage: BufferInspection["storage"];
  kvCacheStorage: BufferInspection["storage"];
  position: number;
  contextLen: number;
  persistentBindingCount: number;
  stepInputCount: number;
  stepOutputCount: number;
  hostBindingCount: number;
  resourceBindingCount: number;
  bindingShapeHash: bigint;
}>;

export type RuntimeProfile = Readonly<{
  kind: "zgml.runtime.profile";
  signature: string;
  callCount: number;
  backendOpCount: number;
  fallbackOpCount: number;
  backendDispatchCount: number;
  syncCount: number;
  runtimePatchCallCount: number;
  runtimePatchChangedCount: number;
  runtimePatchInvalidCount: number;
  runtimePatchHoles: number;
  runtimePatchCacheWritePosHoles: number;
  runtimePatchAttentionSeqKvHoles: number;
  runtimePatchStencilHash: bigint;
  commandCount: number;
  commandStencilHash: bigint;
  commandOpCount: number;
  commandRowCount: number;
  commandProjectionCount: number;
  commandAttentionCount: number;
  commandMovementCount: number;
  commandElementwiseCount: number;
  commandRopeCount: number;
}>;
export type RuntimeProfileExpectation = Readonly<{
  kind: "zgml.runtime.profile-expectation";
  signature: string;
  fallbackOpCount: number;
  syncCount: number;
  runtimePatchInvalidCount: number;
  noFallback: boolean;
  noSync: boolean;
  noRuntimePatchInvalid: boolean;
}>;

export type LlamaProgramInspection = Readonly<{
  kind: "zgml.llama.program.inspection";
  signature: string;
  vocabSize: number;
  maxSeqLen: number;
  contextLen: number;
  batch: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  nKvHeads: number;
  semanticStageCount: number;
  semanticTokenCount: number;
  semanticLayerStageCount: number;
  semanticTerminalStageCount: number;
  semanticRuntimePatchHoles: number;
  semanticRuntimePatchCacheWritePosHoles: number;
  semanticRuntimePatchAttentionSeqKvHoles: number;
}>;

export type LlamaKvCacheRequirements = Readonly<{
  kind: "zgml.llama.kv-cache.requirements";
  signature: string;
  modelKind: ProgramRequirements["modelKind"];
  scalarBytes: number;
  layers: number;
  kBufferByteLength: number;
  vBufferByteLength: number;
  bufferByteLength: number;
  contextLength: number;
}>;
export type LlamaKvCacheSlot = Readonly<{
  kind: "k" | "v";
  name: "kv-k" | "kv-v";
  role: "persistent";
  scalarType: "f32";
  scalarBytes: number;
  layer: number;
  elementOffset: number;
  elementCount: number;
  byteOffset: number;
  byteLength: number;
  requirements: LlamaKvCacheRequirements;
  layout: LlamaKvCacheLayout;
  slot: LlamaKvCacheLayoutSlot;
}>;
export type LlamaKvCacheLayoutSlot = Readonly<{
  kind: "k" | "v";
  name: "kv-k" | "kv-v";
  role: "persistent";
  scalarType: "f32";
  scalarBytes: number;
  layer: number;
  elementOffset: number;
  elementCount: number;
  byteOffset: number;
  byteLength: number;
}>;
export type LlamaKvCacheLayout = Readonly<{
  kind: "zgml.llama.kv-cache.layout";
  signature: string;
  scalarType: "f32";
  scalarBytes: number;
  layers: number;
  contextLength: number;
  kBufferByteLength: number;
  vBufferByteLength: number;
  bufferByteLength: number;
  slots: readonly LlamaKvCacheLayoutSlot[];
  k: readonly LlamaKvCacheLayoutSlot[];
  v: readonly LlamaKvCacheLayoutSlot[];
}>;
export type LlamaKvCacheCreateOptions = {
  resource?: (slot: LlamaKvCacheSlot) => NativeBuffer;
  externalResource?: (slot: LlamaKvCacheSlot) => NativeBuffer;
  placement?: ZgmlBackend;
  backend?: ZgmlBackend;
  device?: ZgmlBackend;
};

export type ProgramBufferKind = "weights" | "bias" | "input";
export type ProgramDeviceBufferKind = ProgramBufferKind | "output" | "kv-k" | "kv-v";
export type ProgramOutputBufferSlot = ProgramBufferLayoutSlot & Readonly<{
  kind: "output";
  name: "output";
  role: "step-output";
  elementLength: number;
  requirements: ProgramRequirements;
  layout: ProgramBufferLayout;
  slot: ProgramBufferLayoutSlot;
}>;
export type ProgramOutputBufferCreateOptions = {
  resource?: (slot: ProgramOutputBufferSlot) => NativeBuffer;
  externalResource?: (slot: ProgramOutputBufferSlot) => NativeBuffer;
  placement?: ZgmlBackend;
  backend?: ZgmlBackend;
  device?: ZgmlBackend;
};
export type ProgramBufferSlot = ProgramBufferLayoutSlot & Readonly<{
  kind: ProgramBufferKind;
  name: ProgramBufferKind;
  role: "persistent" | "step-input";
  elementLength: number;
  requirements: ProgramRequirements;
  layout: ProgramBufferLayout;
  slot: ProgramBufferLayoutSlot;
}>;
export type ProgramBufferCreateOptions = {
  resource?: (slot: ProgramBufferSlot) => NativeBuffer;
  externalResource?: (slot: ProgramBufferSlot) => NativeBuffer;
  placement?: ZgmlBackend;
  backend?: ZgmlBackend;
  device?: ZgmlBackend;
};
export type ProgramCreateBufferOptions = ProgramBufferCreateOptions | ProgramOutputBufferCreateOptions;
export type ExternalResourceAccessName =
  | "read"
  | "readonly"
  | "read-only"
  | "write"
  | "writeonly"
  | "write-only"
  | "readwrite"
  | "read-write"
  | "readWrite";
export type ExternalResourceAccess =
  | ExternalResourceAccessName
  | readonly ExternalResourceAccessName[]
  | { read?: boolean; write?: boolean };
export type ExternalResourceOptions = {
  placement?: ZgmlBackend;
  backend?: ZgmlBackend;
  handle: number;
  byteOffset?: number;
  byteLength: number;
  access?: ExternalResourceAccess;
};

declare const zgmlWebGpuDeviceHandle: unique symbol;
declare const zgmlWebGpuBufferHandle: unique symbol;
declare const zgmlWebGpuByteLength: unique symbol;
declare const zgmlWebGpuByteOffset: unique symbol;
declare const zgmlWebGpuPlacement: unique symbol;
declare const zgmlWebGpuImportSource: unique symbol;

export type WebGpuInteropSymbols = Readonly<{
  deviceHandle: typeof zgmlWebGpuDeviceHandle;
  bufferHandle: typeof zgmlWebGpuBufferHandle;
  byteLength: typeof zgmlWebGpuByteLength;
  byteOffset: typeof zgmlWebGpuByteOffset;
  placement: typeof zgmlWebGpuPlacement;
  importSource: typeof zgmlWebGpuImportSource;
}>;

export type WebGpuInteropImportFields = {
  [zgmlWebGpuPlacement]?: ZgmlBackend;
  [zgmlWebGpuDeviceHandle]?: number;
  [zgmlWebGpuBufferHandle]?: number;
  [zgmlWebGpuByteOffset]?: number;
  [zgmlWebGpuByteLength]?: number;
};

export type WebGpuInteropImportSource = Record<string | symbol, unknown> & {
  [zgmlWebGpuImportSource]: () =>
    | ProgramDeviceBufferImportOptions
    | ProgramDeviceBufferImportDescriptor
    | HostGpuBufferImportSource
    | NativeBuffer;
};

export type ProgramDeviceBufferImportOptions = WebGpuInteropImportFields & {
  placement?: ZgmlBackend;
  backend?: ZgmlBackend;
  device?: ZgmlBackend;
  deviceHandle?: number;
  wgpuDeviceHandle?: number;
  byteOffset?: number;
  byteLength?: number;
  byteLen?: number;
} & (
  | { deviceHandle: number; wgpuDeviceHandle?: number }
  | { wgpuDeviceHandle: number; deviceHandle?: number }
  | { [zgmlWebGpuDeviceHandle]: number; deviceHandle?: number; wgpuDeviceHandle?: number }
) & (
  | { bufferHandle: number; handle?: number; wgpuBufferHandle?: number }
  | { handle: number; bufferHandle?: number; wgpuBufferHandle?: number }
  | { wgpuBufferHandle: number; bufferHandle?: number; handle?: number }
  | { [zgmlWebGpuBufferHandle]: number; bufferHandle?: number; handle?: number; wgpuBufferHandle?: number }
);
export type HostGpuBufferImportSource = Record<string | symbol, unknown> & WebGpuInteropImportFields & {
  placement?: ZgmlBackend;
  backend?: ZgmlBackend;
  device?: ZgmlBackend | unknown;
  deviceHandle?: number;
  wgpuDeviceHandle?: number;
  bufferHandle?: number;
  wgpuBufferHandle?: number;
  handle?: number;
  byteOffset?: number;
  byteLength?: number;
  byteLen?: number;
  size?: number;
} & (
  | { deviceHandle: number }
  | { wgpuDeviceHandle: number }
  | { [zgmlWebGpuDeviceHandle]: number }
) & (
  | { bufferHandle: number }
  | { wgpuBufferHandle: number }
  | { handle: number }
  | { [zgmlWebGpuBufferHandle]: number }
);
export type ProgramDeviceBufferImportDescriptor = WebGpuInteropImportFields & {
  placement?: ZgmlBackend;
  backend?: ZgmlBackend;
  device?: ZgmlBackend;
  deviceHandle?: number;
  wgpuDeviceHandle?: number;
  byteOffset?: number;
  byteLength?: number;
  byteLen?: number;
} & (
  | { bufferHandle: number; handle?: number; wgpuBufferHandle?: number }
  | { handle: number; bufferHandle?: number; wgpuBufferHandle?: number }
  | { wgpuBufferHandle: number; bufferHandle?: number; handle?: number }
  | { [zgmlWebGpuBufferHandle]: number; bufferHandle?: number; handle?: number; wgpuBufferHandle?: number }
);
export type ProgramDeviceBufferImportSource =
  | NativeBuffer
  | ProgramDeviceBufferImportOptions
  | HostGpuBufferImportSource
  | WebGpuInteropImportSource;
export type ProgramDeviceImportBufferSource =
  | NativeBuffer
  | HostGpuBufferImportSource
  | ProgramDeviceBufferImportDescriptor
  | WebGpuInteropImportSource;
export type ProgramDeviceInfo = Readonly<{
  kind: "zgml.program-device.info";
  signature: string;
  placement: ZgmlBackend;
  handle: number;
}>;

export declare class ZgmlError extends Error {
  readonly code: number;
  readonly status: string;
}

export type NativeBufferByteRangeOperation =
  | "write"
  | "read-float32-into"
  | "read-bytes-into";
export type NativeBufferByteRangeElementType = "bytes" | "f32";
export type NativeBufferByteRangeInfo = Readonly<{
  kind: "zgml.native-buffer.byte-range";
  signature: string;
  operation: NativeBufferByteRangeOperation;
  elementType: NativeBufferByteRangeElementType;
  byteOffset: number;
  byteLength: number;
  length: number | null;
}>;
export type ProgramDeviceBufferImportInfo = Readonly<{
  kind: "zgml.native-buffer.device-buffer-import";
  signature: string;
  placementName: string;
  placement: number;
  deviceHandle: number;
  bufferHandle: number;
  byteOffset: number;
  explicitByteLength?: number;
  byteLength: number;
}>;
export type NativeBufferDeviceImportInfo = Readonly<{
  kind: "zgml.native-buffer.device-import";
  signature: string;
  placement: string;
  deviceHandle: number;
  bufferHandle: number;
  byteOffset: number;
  byteLength: number;
}>;
export type NativeBufferExternalResourceInfo = Readonly<{
  kind: "zgml.native-buffer.external-resource";
  signature: string;
  placementName: string;
  placement: number;
  accessFlags: number;
  handle: number;
  byteOffset: number;
  byteLength: number;
}>;

export declare class NativeBuffer {
  readonly byteLength: number;
  static create(byteLength: number): NativeBuffer;
  static wrapBytes(data: ArrayBufferView): NativeBuffer;
  static wrapFloat32(data: Float32Array): NativeBuffer;
  static externalResource(options: ExternalResourceOptions): NativeBuffer;
  static fromFloat32(values: IndexLike): NativeBuffer;
  static fromBytes(values: ByteLike): NativeBuffer;

  size(): number;
  inspect(): BufferInspection;
  readFloat32(length?: number, byteOffset?: number): Float32Array;
  readFloat32Into(target: Float32Array, length?: number, byteOffset?: number): Float32Array;
  readBytes(length?: number, byteOffset?: number): Uint8Array;
  readBytesInto(target: Uint8Array, byteLength?: number, byteOffset?: number): Uint8Array;
  writeFloat32(values: IndexLike, byteOffset?: number): this;
  writeBytes(values: ByteLike, byteOffset?: number): this;
  free(): void;
  dispose(): void;
}

export declare class ProgramDevice {
  readonly placement: ZgmlBackend;
  readonly handle: number;
  info(): ProgramDeviceInfo;
  matchesInfoSignature(signature: string): boolean;
  createBuffer(kind: ProgramDeviceBufferKind): NativeBuffer;
  createOutputBuffer(): NativeBuffer;
  createWeightsBuffer(): NativeBuffer;
  createBiasBuffer(): NativeBuffer;
  createInputBuffer(): NativeBuffer;
  createKvCache(): LlamaKvCache;
  importBuffer(kind: ProgramDeviceBufferKind, source: ProgramDeviceImportBufferSource): NativeBuffer;
}

export declare class TinyLinearModel {
  readonly desc: TinyLinearDesc;
  static create(desc: TinyLinearDesc): TinyLinearModel;
  inspect(): ModelInspection;
  compile(options?: CompileOptions): Program;
  free(): void;
  dispose(): void;
}

export declare class TinyMlpModel {
  readonly desc: TinyMlpDesc;
  static create(desc: TinyMlpDesc): TinyMlpModel;
  inspect(): ModelInspection;
  compile(options?: CompileOptions): Program;
  free(): void;
  dispose(): void;
}

export declare class Program<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> {
  inspect(): ProgramInspection;
  compileEvidence(): ProgramCompileEvidence | null;
  requirements(): ProgramRequirements;
  bufferLayout(): ProgramBufferLayout;
  bufferSlotNames(): readonly string[];
  bufferSlot(nameOrKind: ProgramDeviceBufferKind | string): ProgramBufferLayoutSlot | null;
  inputLen(): number;
  outputLen(): number;
  inputByteLength(): number;
  outputByteLength(): number;
  weightsLen(): number;
  weightsByteLength(): number;
  biasLen(): number;
  biasByteLength(): number;
  parameterLen(): number;
  parameterByteLength(): number;
  bufferSizing(): ProgramBufferSizing;
  matchesBufferSizingSignature(signature: string): boolean;
  inputShape(): InputShape;
  outputShape(): OutputShape;
  trace(): ModuleProgramTrace | null;
  compilerSignatures(): ModuleCompleteCompilerSignatures | null;
  tensorProgramIr(): ModuleTensorProgramIr | null;
  kernelPlan(): ModuleKernelPlan | null;
  shapeConstraints(): ModuleKernelShapeConstraints | null;
  memoryLayout(): ModuleKernelMemoryLayout | null;
  parameterLayout(): ModuleKernelParameterLayout | null;
  parameterNames(): readonly string[];
  parameterInfos(): readonly ModuleKernelParameterLayoutEntry[];
  parameterInfo(nameOrIndex: string | number): ModuleKernelParameterLayoutEntry | null;
  modelCompatibility(model: unknown): ProgramModelCompatibility;
  acceptsModel(model: unknown): boolean;
  moduleCompatibility(module: EmbeddingModule, options?: ModuleParameterPlacementOptions): ProgramModuleCompatibility;
  moduleCompatibility(module: NnCompilableModule, options?: ModuleParameterPlacementOptions): ProgramModuleCompatibility;
  acceptsModule(module: EmbeddingModule, options?: ModuleParameterPlacementOptions): boolean;
  acceptsModule(module: NnCompilableModule, options?: ModuleParameterPlacementOptions): boolean;
  capabilities(): ProgramExecutionCapabilities;
  executionPlan(): ProgramExecutionPlan<InputShape, OutputShape>;
  executionPlan<const Target extends EmbeddingModule | NnCompilableModule>(module: ProgramCompatibleModule<InputShape, OutputShape, Target>, options?: ModuleParameterPlacementOptions): ProgramExecutionPlan<InputShape, OutputShape>;
  execution_plan(): ProgramExecutionPlan<InputShape, OutputShape>;
  execution_plan<const Target extends EmbeddingModule | NnCompilableModule>(module: ProgramCompatibleModule<InputShape, OutputShape, Target>, options?: ModuleParameterPlacementOptions): ProgramExecutionPlan<InputShape, OutputShape>;
  requireExecutionPlan(): ProgramExecutionPlan<InputShape, OutputShape>;
  requireExecutionPlan<const Target extends EmbeddingModule | NnCompilableModule>(module: ProgramCompatibleModule<InputShape, OutputShape, Target>, options?: ModuleParameterPlacementOptions): ProgramExecutionPlan<InputShape, OutputShape>;
  require_execution_plan(): ProgramExecutionPlan<InputShape, OutputShape>;
  require_execution_plan<const Target extends EmbeddingModule | NnCompilableModule>(module: ProgramCompatibleModule<InputShape, OutputShape, Target>, options?: ModuleParameterPlacementOptions): ProgramExecutionPlan<InputShape, OutputShape>;
  bindingPlan(params?: ProgramBindings<InputShape, OutputShape>): ProgramBindingPlan<InputShape, OutputShape>;
  binding_plan(params?: ProgramBindings<InputShape, OutputShape>): ProgramBindingPlan<InputShape, OutputShape>;
  requireBindingPlan(params?: ProgramBindings<InputShape, OutputShape>): ProgramBindingPlan<InputShape, OutputShape>;
  require_binding_plan(params?: ProgramBindings<InputShape, OutputShape>): ProgramBindingPlan<InputShape, OutputShape>;
  canExecute(): boolean;
  can_execute(): boolean;
  canBindExternalResources(): boolean;
  can_bind_external_resources(): boolean;
  hasFullDispatchPlan(): boolean;
  has_full_dispatch_plan(): boolean;
  executionMode(): ProgramExecutionCapabilities["mode"];
  execution_mode(): ProgramExecutionCapabilities["mode"];
  matchesCapabilitySignature(signature: string): boolean;
  matches_capability_signature(signature: string): boolean;
  createBuffer(kind: ProgramDeviceBufferKind, options?: ProgramCreateBufferOptions): NativeBuffer;
  createOutputBuffer(options?: ProgramOutputBufferCreateOptions): NativeBuffer;
  createWeightsBuffer(options?: ProgramBufferCreateOptions): NativeBuffer;
  createBiasBuffer(options?: ProgramBufferCreateOptions): NativeBuffer;
  createInputBuffer(options?: ProgramBufferCreateOptions): NativeBuffer;
  deviceHandle(placement?: ZgmlBackend): number;
  device(placement?: ZgmlBackend): ProgramDevice;
  importDeviceBuffer(kind: ProgramDeviceBufferKind, source: ProgramDeviceBufferImportSource): NativeBuffer;
  runtimeProfile(): RuntimeProfile;
  runtime_profile(): RuntimeProfile;
  matchesRuntimeProfileSignature(signature: string): boolean;
  matches_runtime_profile_signature(signature: string): boolean;
  resetRuntimeProfile(): void;
  reset_runtime_profile(): void;
  runtimeProfileExpectation(): RuntimeProfileExpectation;
  runtime_profile_expectation(): RuntimeProfileExpectation;
  runtimeProfileHasNoFallback(): boolean;
  runtime_profile_has_no_fallback(): boolean;
  runtimeProfileHasNoSync(): boolean;
  runtime_profile_has_no_sync(): boolean;
  runtimeProfileHasNoInvalidRuntimePatches(): boolean;
  runtime_profile_has_no_invalid_runtime_patches(): boolean;
  requireNoFallbackRuntimeProfile(): RuntimeProfileExpectation;
  require_no_fallback_runtime_profile(): RuntimeProfileExpectation;
  requireNoSyncRuntimeProfile(): RuntimeProfileExpectation;
  require_no_sync_runtime_profile(): RuntimeProfileExpectation;
  requireRuntimePatchValidProfile(): RuntimeProfileExpectation;
  require_runtime_patch_valid_profile(): RuntimeProfileExpectation;
  requireHotRuntimeProfile(): RuntimeProfileExpectation;
  require_hot_runtime_profile(): RuntimeProfileExpectation;
  bind<const Target extends EmbeddingModule | NnCompilableModule>(module: ProgramCompatibleModule<InputShape, OutputShape, Target>, options?: ModuleParameterPlacementOptions): Session<InputShape, OutputShape>;
  bind(params: ProgramBindings<InputShape, OutputShape>): Session<InputShape, OutputShape>;
  bindModule<const Target extends EmbeddingModule | NnCompilableModule>(module: ProgramCompatibleModule<InputShape, OutputShape, Target>, options?: ModuleParameterPlacementOptions): Session<InputShape, OutputShape>;
  free(): void;
  dispose(): void;
}

export declare class Session<InputShape extends TensorShapeTuple = TensorShapeTuple, OutputShape extends TensorShapeTuple = TensorShapeTuple> {
  inspect(): SessionInspection;
  bufferLayout(): ProgramBufferLayout;
  bufferSlotNames(): readonly string[];
  bufferSlot(nameOrKind: ProgramDeviceBufferKind | string): ProgramBufferLayoutSlot | null;
  inputLen(): number;
  outputLen(): number;
  inputByteLength(): number;
  outputByteLength(): number;
  weightsLen(): number;
  weightsByteLength(): number;
  biasLen(): number;
  biasByteLength(): number;
  parameterLen(): number;
  parameterByteLength(): number;
  bufferSizing(): SessionBufferSizing;
  matchesBufferSizingSignature(signature: string): boolean;
  inputShape(): InputShape;
  outputShape(): OutputShape;
  sessionCallProfile(): SessionCallProfile;
  session_call_profile(): SessionCallProfile;
  matchesSessionCallProfileSignature(signature: string): boolean;
  matches_session_call_profile_signature(signature: string): boolean;
  resetSessionCallProfile(): void;
  reset_session_call_profile(): void;
  stepContract(): SessionStepContract;
  step_contract(): SessionStepContract;
  preflightStepParams(params?: unknown): SessionStepParamsCompatibility;
  preflight_step_params(params?: unknown): SessionStepParamsCompatibility;
  stepParamsCompatibility(params?: unknown): SessionStepParamsCompatibility;
  step_params_compatibility(params?: unknown): SessionStepParamsCompatibility;
  acceptsStepParams(params?: unknown): boolean;
  accepts_step_params(params?: unknown): boolean;
  canExecuteStepParams(params?: unknown): boolean;
  can_execute_step_params(params?: unknown): boolean;
  requireCanExecuteStepParams(params?: unknown): SessionStepParamsCompatibility;
  require_can_execute_step_params(params?: unknown): SessionStepParamsCompatibility;
  acceptsAllocationFreeStepParams(params?: unknown): boolean;
  accepts_allocation_free_step_params(params?: unknown): boolean;
  requireAllocationFreeStepParams(params?: unknown): SessionStepParamsCompatibility;
  require_allocation_free_step_params(params?: unknown): SessionStepParamsCompatibility;
  acceptsRuntimeOutputAllocationFreeStepParams(params?: unknown): boolean;
  accepts_runtime_output_allocation_free_step_params(params?: unknown): boolean;
  requireRuntimeOutputAllocationFreeStepParams(params?: unknown): SessionStepParamsCompatibility;
  require_runtime_output_allocation_free_step_params(params?: unknown): SessionStepParamsCompatibility;
  acceptsNoReadbackStepParams(params?: unknown): boolean;
  accepts_no_readback_step_params(params?: unknown): boolean;
  requireNoReadbackStepParams(params?: unknown): SessionStepParamsCompatibility;
  require_no_readback_step_params(params?: unknown): SessionStepParamsCompatibility;
  acceptsReadbackFreeStepParams(params?: unknown): boolean;
  accepts_readback_free_step_params(params?: unknown): boolean;
  requireReadbackFreeStepParams(params?: unknown): SessionStepParamsCompatibility;
  require_readback_free_step_params(params?: unknown): SessionStepParamsCompatibility;
  acceptsHotStepParams(params?: unknown): boolean;
  accepts_hot_step_params(params?: unknown): boolean;
  requireHotStepParams(params?: unknown): SessionStepParamsCompatibility;
  require_hot_step_params(params?: unknown): SessionStepParamsCompatibility;
  executionPlan(params?: unknown): SessionExecutionPlan<InputShape, OutputShape>;
  execution_plan(params?: unknown): SessionExecutionPlan<InputShape, OutputShape>;
  requireExecutionPlan(params?: unknown): SessionExecutionPlan<InputShape, OutputShape>;
  require_execution_plan(params?: unknown): SessionExecutionPlan<InputShape, OutputShape>;
  hotPathPlan(params?: unknown): SessionExecutionPlan<InputShape, OutputShape>;
  hot_path_plan(params?: unknown): SessionExecutionPlan<InputShape, OutputShape>;
  matchesStepContractSignature(signature: string): boolean;
  matches_step_contract_signature(signature: string): boolean;
  matchesStepParamsSignature(params: unknown, signature: string): boolean;
  matches_step_params_signature(params: unknown, signature: string): boolean;
  matchesStepParamsCompatibility(params: unknown, compatibility: SessionStepParamsCompatibility): boolean;
  matches_step_params_compatibility(params: unknown, compatibility: SessionStepParamsCompatibility): boolean;
  uploadParameters(): void;
  uploadParameter(index: number): void;
  uploadParameterByName(name: string): void;
  uploadParameterRange(first: number, len: number): void;
  parameterLayout(): ModuleKernelParameterLayout | null;
  parameterNames(): readonly string[];
  parameterInfos(): readonly ModuleKernelParameterLayoutEntry[];
  parameterInfo(nameOrIndex: string | number): ModuleKernelParameterLayoutEntry | null;
  step(inputValues?: ProgramInputBinding<InputShape>, outputValues?: Tensor<OutputShape> | Float32Array): Float32Array;
  stepTensor<const S extends TensorShapeTuple>(inputValues: ProgramInputBinding<InputShape> | undefined, options: { output: Tensor<S> }): Tensor<S>;
  stepTensor<const S extends TensorShape>(inputValues: ProgramInputBinding<InputShape> | undefined, options: SessionStepTensorOptions & { shape: S }): Tensor<TensorShapeOf<S>>;
  stepTensor<const S extends TensorShapeTuple>(inputValues: ProgramInputBinding<InputShape> | undefined, options: SessionStepTensorOptions & { output: Tensor<S>; shape?: undefined }): Tensor<S>;
  stepTensor(inputValues?: ProgramInputBinding<InputShape>, options?: SessionStepTensorOptions): Tensor<OutputShape>;
  stepTensor(inputValues?: ProgramInputBinding<InputShape>, options?: SessionStepTensorOptions): Tensor;
  stepInto(target: Float32Array, inputValues?: ProgramInputBinding<InputShape>): Float32Array;
  execute(params: SessionStepParams<InputShape, OutputShape> & { output: false }): undefined;
  execute(params?: SessionStepParams<InputShape, OutputShape>): Float32Array | undefined;
  executeTensor<const S extends TensorShape>(params: SessionExecuteTensorParams<InputShape, OutputShape> & { shape: S }): Tensor<TensorShapeOf<S>>;
  executeTensor<const S extends TensorShapeTuple>(params: SessionExecuteTensorParams<InputShape, OutputShape> & { output: Tensor<S>; shape?: undefined }): Tensor<S>;
  executeTensor(params?: SessionExecuteTensorParams<InputShape, OutputShape>): Tensor<OutputShape>;
  executeTensor(params?: SessionExecuteTensorParams<InputShape, OutputShape>): Tensor;
  executeInto(target: Float32Array, params?: SessionExecuteIntoParams<InputShape>): Float32Array;
  prepareExecuteInto(target: Float32Array, params?: SessionExecuteIntoParams<InputShape>): () => Float32Array;
  readOutputInto(target: Float32Array, length?: number, byteOffset?: number): Float32Array;
  readOutputTensor<const S extends TensorShapeTuple>(options: { output: Tensor<S> }): Tensor<S>;
  readOutputTensor<const S extends TensorShape>(options: SessionReadOutputTensorOptions & { shape: S }): Tensor<TensorShapeOf<S>>;
  readOutputTensor<const S extends TensorShapeTuple>(options: SessionReadOutputTensorOptions & { output: Tensor<S>; shape?: undefined }): Tensor<S>;
  readOutputTensor(options?: SessionReadOutputTensorOptions): Tensor<OutputShape>;
  readOutputTensor(options?: SessionReadOutputTensorOptions): Tensor;
  advance(inputValues?: ProgramInputBinding<InputShape>): void;
  runtimeProfile(): RuntimeProfile;
  runtime_profile(): RuntimeProfile;
  matchesRuntimeProfileSignature(signature: string): boolean;
  matches_runtime_profile_signature(signature: string): boolean;
  resetRuntimeProfile(): void;
  reset_runtime_profile(): void;
  runtimeProfileExpectation(): RuntimeProfileExpectation;
  runtime_profile_expectation(): RuntimeProfileExpectation;
  runtimeProfileHasNoFallback(): boolean;
  runtime_profile_has_no_fallback(): boolean;
  runtimeProfileHasNoSync(): boolean;
  runtime_profile_has_no_sync(): boolean;
  runtimeProfileHasNoInvalidRuntimePatches(): boolean;
  runtime_profile_has_no_invalid_runtime_patches(): boolean;
  requireNoFallbackRuntimeProfile(): RuntimeProfileExpectation;
  require_no_fallback_runtime_profile(): RuntimeProfileExpectation;
  requireNoSyncRuntimeProfile(): RuntimeProfileExpectation;
  require_no_sync_runtime_profile(): RuntimeProfileExpectation;
  requireRuntimePatchValidProfile(): RuntimeProfileExpectation;
  require_runtime_patch_valid_profile(): RuntimeProfileExpectation;
  requireHotRuntimeProfile(): RuntimeProfileExpectation;
  require_hot_runtime_profile(): RuntimeProfileExpectation;
  reset(): void;
  free(): void;
  dispose(): void;
}

export type TinyLinearProgram = Program;
export declare const TinyLinearProgram: typeof Program;
export type TinyLinearSession = Session;
export declare const TinyLinearSession: typeof Session;

export type LlamaOutputBinding = NativeBuffer | "native" | true;
export type LlamaKvCacheResources = {
  k: readonly NativeBuffer[];
  v: readonly NativeBuffer[];
};
export type LlamaBindOptions = {
  output?: LlamaOutputBinding;
  model?: unknown;
  kvCache?: LlamaKvCacheResources;
};

export declare class LlamaKvCache implements LlamaKvCacheResources {
  readonly k: readonly NativeBuffer[];
  readonly v: readonly NativeBuffer[];
  free(): void;
  dispose(): void;
}

export declare class TinyLlamaModel {
  static create(): TinyLlamaModel;
  static load(path: string): TinyLlamaModel;
  static loadSafetensorsData(data: Uint8Array | ArrayBuffer): TinyLlamaModel;
  inspect(): ModelInspection;
  compile(options?: CompileOptions): TinyLlamaProgram;
  free(): void;
  dispose(): void;
}

export declare class TinyLlamaProgram {
  readonly vocabSize: number;
  inspect(): LlamaProgramInspection;
  inspectExecutable(): ProgramInspection;
  requirements(): ProgramRequirements;
  bufferLayout(): ProgramBufferLayout;
  bufferSlotNames(): readonly string[];
  bufferSlot(nameOrKind: ProgramDeviceBufferKind | string): ProgramBufferLayoutSlot | null;
  inputLen(): number;
  outputLen(): number;
  inputByteLength(): number;
  outputByteLength(): number;
  weightsLen(): number;
  weightsByteLength(): number;
  biasLen(): number;
  biasByteLength(): number;
  parameterLen(): number;
  parameterByteLength(): number;
  bufferSizing(): ProgramBufferSizing;
  matchesBufferSizingSignature(signature: string): boolean;
  outputShape(): readonly number[];
  modelCompatibility(model: unknown): ProgramModelCompatibility;
  acceptsModel(model: unknown): boolean;
  capabilities(): ProgramExecutionCapabilities;
  executionPlan(): ProgramExecutionPlan;
  execution_plan(): ProgramExecutionPlan;
  requireExecutionPlan(): ProgramExecutionPlan;
  require_execution_plan(): ProgramExecutionPlan;
  canExecute(): boolean;
  can_execute(): boolean;
  canBindExternalResources(): boolean;
  can_bind_external_resources(): boolean;
  hasFullDispatchPlan(): boolean;
  has_full_dispatch_plan(): boolean;
  executionMode(): ProgramExecutionCapabilities["mode"];
  execution_mode(): ProgramExecutionCapabilities["mode"];
  matchesCapabilitySignature(signature: string): boolean;
  matches_capability_signature(signature: string): boolean;
  kvCacheRequirements(): LlamaKvCacheRequirements;
  kvCacheLayout(): LlamaKvCacheLayout;
  createOutputBuffer(options?: ProgramOutputBufferCreateOptions): NativeBuffer;
  createBuffer(kind: ProgramDeviceBufferKind, options?: ProgramCreateBufferOptions): NativeBuffer;
  deviceHandle(placement?: ZgmlBackend): number;
  device(placement?: ZgmlBackend): ProgramDevice;
  importDeviceBuffer(kind: ProgramDeviceBufferKind, source: ProgramDeviceBufferImportSource): NativeBuffer;
  createKvCache(options?: LlamaKvCacheCreateOptions): LlamaKvCache;
  runtimeProfile(): RuntimeProfile;
  runtime_profile(): RuntimeProfile;
  matchesRuntimeProfileSignature(signature: string): boolean;
  matches_runtime_profile_signature(signature: string): boolean;
  resetRuntimeProfile(): void;
  reset_runtime_profile(): void;
  runtimeProfileExpectation(): RuntimeProfileExpectation;
  runtime_profile_expectation(): RuntimeProfileExpectation;
  runtimeProfileHasNoFallback(): boolean;
  runtime_profile_has_no_fallback(): boolean;
  runtimeProfileHasNoSync(): boolean;
  runtime_profile_has_no_sync(): boolean;
  runtimeProfileHasNoInvalidRuntimePatches(): boolean;
  runtime_profile_has_no_invalid_runtime_patches(): boolean;
  requireNoFallbackRuntimeProfile(): RuntimeProfileExpectation;
  require_no_fallback_runtime_profile(): RuntimeProfileExpectation;
  requireNoSyncRuntimeProfile(): RuntimeProfileExpectation;
  require_no_sync_runtime_profile(): RuntimeProfileExpectation;
  requireRuntimePatchValidProfile(): RuntimeProfileExpectation;
  require_runtime_patch_valid_profile(): RuntimeProfileExpectation;
  requireHotRuntimeProfile(): RuntimeProfileExpectation;
  require_hot_runtime_profile(): RuntimeProfileExpectation;
  bind(options?: LlamaBindOptions): TinyLlamaSession;
  free(): void;
  dispose(): void;
}

export declare class TinyLlamaSession {
  readonly vocabSize: number;
  inspect(): SessionInspection;
  bufferLayout(): ProgramBufferLayout;
  bufferSlotNames(): readonly string[];
  bufferSlot(nameOrKind: ProgramDeviceBufferKind | string): ProgramBufferLayoutSlot | null;
  kvCacheLayout(): LlamaKvCacheLayout;
  inputLen(): number;
  outputLen(): number;
  inputByteLength(): number;
  outputByteLength(): number;
  weightsLen(): number;
  weightsByteLength(): number;
  biasLen(): number;
  biasByteLength(): number;
  parameterLen(): number;
  parameterByteLength(): number;
  bufferSizing(): SessionBufferSizing;
  matchesBufferSizingSignature(signature: string): boolean;
  outputShape(): readonly number[];
  sessionCallProfile(): SessionCallProfile;
  session_call_profile(): SessionCallProfile;
  matchesSessionCallProfileSignature(signature: string): boolean;
  matches_session_call_profile_signature(signature: string): boolean;
  resetSessionCallProfile(): void;
  reset_session_call_profile(): void;
  stepContract(): LlamaSessionStepContract;
  step_contract(): LlamaSessionStepContract;
  preflightStepParams(params?: unknown): SessionStepParamsCompatibility;
  preflight_step_params(params?: unknown): SessionStepParamsCompatibility;
  stepParamsCompatibility(params?: unknown): SessionStepParamsCompatibility;
  step_params_compatibility(params?: unknown): SessionStepParamsCompatibility;
  acceptsStepParams(params?: unknown): boolean;
  accepts_step_params(params?: unknown): boolean;
  canExecuteStepParams(params?: unknown): boolean;
  can_execute_step_params(params?: unknown): boolean;
  requireCanExecuteStepParams(params?: unknown): SessionStepParamsCompatibility;
  require_can_execute_step_params(params?: unknown): SessionStepParamsCompatibility;
  acceptsAllocationFreeStepParams(params?: unknown): boolean;
  accepts_allocation_free_step_params(params?: unknown): boolean;
  requireAllocationFreeStepParams(params?: unknown): SessionStepParamsCompatibility;
  require_allocation_free_step_params(params?: unknown): SessionStepParamsCompatibility;
  acceptsRuntimeOutputAllocationFreeStepParams(params?: unknown): boolean;
  accepts_runtime_output_allocation_free_step_params(params?: unknown): boolean;
  requireRuntimeOutputAllocationFreeStepParams(params?: unknown): SessionStepParamsCompatibility;
  require_runtime_output_allocation_free_step_params(params?: unknown): SessionStepParamsCompatibility;
  acceptsNoReadbackStepParams(params?: unknown): boolean;
  accepts_no_readback_step_params(params?: unknown): boolean;
  requireNoReadbackStepParams(params?: unknown): SessionStepParamsCompatibility;
  require_no_readback_step_params(params?: unknown): SessionStepParamsCompatibility;
  acceptsReadbackFreeStepParams(params?: unknown): boolean;
  accepts_readback_free_step_params(params?: unknown): boolean;
  requireReadbackFreeStepParams(params?: unknown): SessionStepParamsCompatibility;
  require_readback_free_step_params(params?: unknown): SessionStepParamsCompatibility;
  acceptsHotStepParams(params?: unknown): boolean;
  accepts_hot_step_params(params?: unknown): boolean;
  requireHotStepParams(params?: unknown): SessionStepParamsCompatibility;
  require_hot_step_params(params?: unknown): SessionStepParamsCompatibility;
  executionPlan(params?: unknown): SessionExecutionPlan;
  execution_plan(params?: unknown): SessionExecutionPlan;
  requireExecutionPlan(params?: unknown): SessionExecutionPlan;
  require_execution_plan(params?: unknown): SessionExecutionPlan;
  hotPathPlan(params?: unknown): SessionExecutionPlan;
  hot_path_plan(params?: unknown): SessionExecutionPlan;
  matchesStepContractSignature(signature: string): boolean;
  matches_step_contract_signature(signature: string): boolean;
  matchesStepParamsSignature(params: unknown, signature: string): boolean;
  matches_step_params_signature(params: unknown, signature: string): boolean;
  matchesStepParamsCompatibility(params: unknown, compatibility: SessionStepParamsCompatibility): boolean;
  matches_step_params_compatibility(params: unknown, compatibility: SessionStepParamsCompatibility): boolean;
  step(token: number, outputValues?: Float32Array): Float32Array;
  stepTensor<const S extends TensorShapeTuple>(token: number, options: { output: Tensor<S> }): Tensor<S>;
  stepTensor<const S extends TensorShape>(token: number, options: LlamaLogitsTensorOptions & { shape: S }): Tensor<TensorShapeOf<S>>;
  stepTensor<const S extends TensorShapeTuple>(token: number, options: LlamaLogitsTensorOptions & { output: Tensor<S>; shape?: undefined }): Tensor<S>;
  stepTensor(token: number, options?: LlamaLogitsTensorOptions): Tensor;
  stepInto(target: Float32Array, token: number): Float32Array;
  advance(token: number): void;
  advanceTokens(tokens: TokenIds, options?: TokenWindowOptions): void;
  advance_tokens(tokens: TokenIds, options?: TokenWindowOptions): void;
  executeTokens(tokens: TokenIds, options?: ExecuteTokensOptions): Float32Array | undefined;
  execute_tokens(tokens: TokenIds, options?: ExecuteTokensOptions): Float32Array | undefined;
  execute(params: LlamaExecuteParams): Float32Array | undefined;
  executeTensor<const S extends TensorShape>(params: LlamaExecuteTensorParams & { shape: S }): Tensor<TensorShapeOf<S>>;
  executeTensor<const S extends TensorShapeTuple>(params: LlamaExecuteTensorParams & { output: Tensor<S>; shape?: undefined }): Tensor<S>;
  executeTensor(params: LlamaExecuteTensorParams): Tensor;
  execute_tensor<const S extends TensorShape>(params: LlamaExecuteTensorParams & { shape: S }): Tensor<TensorShapeOf<S>>;
  execute_tensor<const S extends TensorShapeTuple>(params: LlamaExecuteTensorParams & { output: Tensor<S>; shape?: undefined }): Tensor<S>;
  execute_tensor(params: LlamaExecuteTensorParams): Tensor;
  executeInto(target: Float32Array, params: LlamaExecuteIntoParams): Float32Array;
  execute_into(target: Float32Array, params: LlamaExecuteIntoParams): Float32Array;
  prefill(tokens: TokenIds, outputValues?: Float32Array, options?: TokenWindowOptions): Float32Array;
  prefillTensor<const S extends TensorShape>(tokens: TokenIds, options: LlamaTokenWindowTensorOptions & { shape: S }): Tensor<TensorShapeOf<S>>;
  prefillTensor<const S extends TensorShapeTuple>(tokens: TokenIds, options: LlamaTokenWindowTensorOptions & { output: Tensor<S>; shape?: undefined }): Tensor<S>;
  prefillTensor(tokens: TokenIds, options?: LlamaTokenWindowTensorOptions): Tensor;
  prefill_tensor<const S extends TensorShape>(tokens: TokenIds, options: LlamaTokenWindowTensorOptions & { shape: S }): Tensor<TensorShapeOf<S>>;
  prefill_tensor<const S extends TensorShapeTuple>(tokens: TokenIds, options: LlamaTokenWindowTensorOptions & { output: Tensor<S>; shape?: undefined }): Tensor<S>;
  prefill_tensor(tokens: TokenIds, options?: LlamaTokenWindowTensorOptions): Tensor;
  prefillInto(target: Float32Array, tokens: TokenIds, options?: TokenWindowOptions): Float32Array;
  prefill_into(target: Float32Array, tokens: TokenIds, options?: TokenWindowOptions): Float32Array;
  stepArgmax(token: number): TokenArgmaxResult;
  step_argmax(token: number): TokenArgmaxResult;
  stepSample(token: number, options?: TokenSampleOptions): TokenSampleResult;
  step_sample(token: number, options?: TokenSampleOptions): TokenSampleResult;
  executeTokensArgmax(tokens: TokenIds, options?: TokenWindowOptions): TokenArgmaxResult;
  execute_tokens_argmax(tokens: TokenIds, options?: TokenWindowOptions): TokenArgmaxResult;
  executeTokensSample(tokens: TokenIds, options?: TokenSampleOptions): TokenSampleResult;
  execute_tokens_sample(tokens: TokenIds, options?: TokenSampleOptions): TokenSampleResult;
  generateTokensArgmax(tokens: TokenIds, maxTokens: number, options?: TokenWindowOptions): TokenGenerateArgmaxResult;
  generate_tokens_argmax(tokens: TokenIds, maxTokens: number, options?: TokenWindowOptions): TokenGenerateArgmaxResult;
  generateTokensArgmaxInto(tokens: TokenIds, outputTokens: Uint32Array, options?: TokenWindowOptions): TokenGenerateArgmaxResult;
  generate_tokens_argmax_into(tokens: TokenIds, outputTokens: Uint32Array, options?: TokenWindowOptions): TokenGenerateArgmaxResult;
  generateTokensSample(tokens: TokenIds, maxTokens: number, options?: TokenSampleOptions): TokenGenerateSampleResult;
  generate_tokens_sample(tokens: TokenIds, maxTokens: number, options?: TokenSampleOptions): TokenGenerateSampleResult;
  generateTokensSampleInto(tokens: TokenIds, outputTokens: Uint32Array, options?: TokenSampleOptions): TokenGenerateSampleResult;
  generate_tokens_sample_into(tokens: TokenIds, outputTokens: Uint32Array, options?: TokenSampleOptions): TokenGenerateSampleResult;
  argmaxToken(logits?: TensorLike): TokenArgmaxResult;
  argmax_token(logits?: TensorLike): TokenArgmaxResult;
  sampleToken(logits?: TensorLike, options?: TokenSampleOptions): TokenSampleResult;
  sample_token(logits?: TensorLike, options?: TokenSampleOptions): TokenSampleResult;
  position(): number;
  reset(): void;
  runtimeProfile(): RuntimeProfile;
  runtime_profile(): RuntimeProfile;
  matchesRuntimeProfileSignature(signature: string): boolean;
  matches_runtime_profile_signature(signature: string): boolean;
  resetRuntimeProfile(): void;
  reset_runtime_profile(): void;
  runtimeProfileExpectation(): RuntimeProfileExpectation;
  runtime_profile_expectation(): RuntimeProfileExpectation;
  runtimeProfileHasNoFallback(): boolean;
  runtime_profile_has_no_fallback(): boolean;
  runtimeProfileHasNoSync(): boolean;
  runtime_profile_has_no_sync(): boolean;
  runtimeProfileHasNoInvalidRuntimePatches(): boolean;
  runtime_profile_has_no_invalid_runtime_patches(): boolean;
  requireNoFallbackRuntimeProfile(): RuntimeProfileExpectation;
  require_no_fallback_runtime_profile(): RuntimeProfileExpectation;
  requireNoSyncRuntimeProfile(): RuntimeProfileExpectation;
  require_no_sync_runtime_profile(): RuntimeProfileExpectation;
  requireRuntimePatchValidProfile(): RuntimeProfileExpectation;
  require_runtime_patch_valid_profile(): RuntimeProfileExpectation;
  requireHotRuntimeProfile(): RuntimeProfileExpectation;
  require_hot_runtime_profile(): RuntimeProfileExpectation;
  readOutputInto(target: Float32Array, length?: number, byteOffset?: number): Float32Array;
  readOutputTensor<const S extends TensorShapeTuple>(options: { output: Tensor<S> }): Tensor<S>;
  readOutputTensor<const S extends TensorShape>(options: SessionReadOutputTensorOptions & { shape: S }): Tensor<TensorShapeOf<S>>;
  readOutputTensor<const S extends TensorShapeTuple>(options: SessionReadOutputTensorOptions & { output: Tensor<S>; shape?: undefined }): Tensor<S>;
  readOutputTensor(options?: SessionReadOutputTensorOptions): Tensor;
  free(): void;
  dispose(): void;
}

export declare class SmolLM135MModel extends TinyLlamaModel {
  static load(path: string): SmolLM135MModel;
  static loadSafetensorsData(data: Uint8Array | ArrayBuffer): SmolLM135MModel;
}
export declare class SmolLM135MProgram extends TinyLlamaProgram {}
export declare class SmolLM135MSession extends TinyLlamaSession {}
export declare class LlamaModel extends TinyLlamaModel {}
export declare class LlamaProgram extends TinyLlamaProgram {}
export declare class LlamaSession extends TinyLlamaSession {}

export declare const TinyLinear: typeof TinyLinearModel;
export declare const TinyMlp: typeof TinyMlpModel;
export declare const TinyLlama: typeof TinyLlamaModel;
export declare const SmolLM135M: typeof SmolLM135MModel;
export declare const Llama: typeof LlamaModel;

export declare const frontendManifest: FrontendManifest;
export declare function runtimeInfo(): RuntimeInfo;
export declare const loadedRuntimeInfo: RuntimeInfo;
export declare function abiStructSize(kind: string | number): number;
export declare function abiStructSizes(): Readonly<Record<string, number>>;
export declare function probeModel(source: LoadModelSource, options?: LoadModelOptions | LoadModelKind): ModelInspection;
export declare function probeSafetensorsData(data: Uint8Array | ArrayBuffer, options?: LoadModelOptions | LoadModelKind): ModelInspection;
export declare function probeSafetensorsHeader(header: string | Uint8Array, options?: LoadModelOptions | LoadModelKind): ModelInspection;
export declare function supportedCheckpointModels(): readonly ModelInspection[];
export declare function loadModel(source: LoadModelSource, options?: LoadModelOptions | LoadModelKind): LlamaModel | TinyLlamaModel | SmolLM135MModel;
export declare function loadSafetensorsData(data: Uint8Array | ArrayBuffer, options?: LoadModelOptions | LoadModelKind): LlamaModel | TinyLlamaModel | SmolLM135MModel;

export declare const webgpuInterop: WebGpuInteropSymbols;
