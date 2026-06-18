import {
  type LlamaProgramInspection,
  type ModuleKernelShapeConstraints,
  type TensorShapeTuple,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
type ShapeDesc = Readonly<UnknownRecord>;
type ShapeFallback = (desc: ShapeDesc) => readonly number[];
type ProgramShapeRecord<THandle = unknown> = Readonly<UnknownRecord & {
  handle: THandle;
  desc?: ShapeDesc;
}>;
type ShapeConstraintsLike = Readonly<UnknownRecord & {
  inputShape?: readonly number[];
  outputShape?: readonly number[];
}>;
type ProgramShapeConstraintsInput = ModuleKernelShapeConstraints | ShapeConstraintsLike | null | undefined;
type ProgramShapeConstraintsProvider<TProgram extends ProgramShapeRecord = ProgramShapeRecord> = (program: TProgram) => ProgramShapeConstraintsInput;
type LlamaProgramInspectionLike = Readonly<UnknownRecord & {
  vocabSize: number;
}>;
type LlamaProgramInspectionInput = LlamaProgramInspection | LlamaProgramInspectionLike;

function frozenShape(shape: readonly number[]): TensorShapeTuple {
  return Object.freeze(shape.slice());
}

export function programInputShapeFromEvidence(
  desc: ShapeDesc,
  constraints: ProgramShapeConstraintsInput,
  fallback: ShapeFallback,
): TensorShapeTuple {
  if (constraints && Array.isArray(constraints.inputShape)) return frozenShape(constraints.inputShape);
  return fallback(desc);
}

export function programOutputShapeFromEvidence(
  desc: ShapeDesc,
  constraints: ProgramShapeConstraintsInput,
  fallback: ShapeFallback,
): TensorShapeTuple {
  if (constraints && Array.isArray(constraints.outputShape)) return frozenShape(constraints.outputShape);
  return fallback(desc);
}

export function llamaProgramOutputShapeFromInspection(inspection: LlamaProgramInspectionInput): TensorShapeTuple {
  return Object.freeze([inspection.vocabSize]);
}

export type GenericProgramShapeAccessorsOptions<
  THandle = unknown,
  TProgram extends ProgramShapeRecord<THandle> = ProgramShapeRecord<THandle>,
> = Readonly<{
  assertProgramAlive: (handle: THandle, label: string) => void;
  shapeConstraints: ProgramShapeConstraintsProvider<TProgram>;
  inputShapeFallback: ShapeFallback;
  outputShapeFallback: ShapeFallback;
}>;

export type LlamaProgramShapeAccessorsOptions<TProgram extends object = object> = Readonly<{
  inspect: (program: TProgram) => LlamaProgramInspectionInput;
}>;

export function createGenericProgramShapeAccessors<
  THandle = unknown,
  TProgram extends ProgramShapeRecord<THandle> = ProgramShapeRecord<THandle>,
>(options: GenericProgramShapeAccessorsOptions<THandle, TProgram>) {
  const assertProgramAlive = options.assertProgramAlive;
  const shapeConstraints = options.shapeConstraints;
  const inputShapeFallback = options.inputShapeFallback;
  const outputShapeFallback = options.outputShapeFallback;

  if (typeof assertProgramAlive !== "function") {
    throw new Error("createGenericProgramShapeAccessors requires assertProgramAlive");
  }
  if (typeof shapeConstraints !== "function") {
    throw new Error("createGenericProgramShapeAccessors requires shapeConstraints");
  }
  if (typeof inputShapeFallback !== "function" || typeof outputShapeFallback !== "function") {
    throw new Error("createGenericProgramShapeAccessors requires shape fallbacks");
  }

  function inputShape(program: TProgram): TensorShapeTuple {
    assertProgramAlive(program.handle, "program");
    return programInputShapeFromEvidence(program.desc || {}, shapeConstraints(program), inputShapeFallback);
  }

  function outputShape(program: TProgram): TensorShapeTuple {
    assertProgramAlive(program.handle, "program");
    return programOutputShapeFromEvidence(program.desc || {}, shapeConstraints(program), outputShapeFallback);
  }

  return Object.freeze({
    inputShape,
    outputShape,
  });
}

export function createLlamaProgramShapeAccessors<TProgram extends object = object>(options: LlamaProgramShapeAccessorsOptions<TProgram>) {
  const inspect = options.inspect;
  if (typeof inspect !== "function") {
    throw new Error("createLlamaProgramShapeAccessors requires inspect");
  }

  function outputShape(program: TProgram): TensorShapeTuple {
    return llamaProgramOutputShapeFromInspection(inspect(program));
  }

  return Object.freeze({
    outputShape,
  });
}
