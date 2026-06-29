"use strict";

import {
  sameShape,
  createShapedF32Helpers,
} from "../core/shape.js";

import {
  validateProgramBindModuleCompatibility,
} from "./module_bindings.js";
import type {
  ProgramBindingDiagnostic,
  ProgramBindingPlan,
  ProgramBindings,
  ProgramBufferLayout,
  ProgramModuleCompatibility,
  TensorShapeTuple,
  TensorToOptions,
  TensorToTarget,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
export type ProgramBindingsInput = ProgramBindings | UnknownRecord;
type TensorToTargetInput = TensorToTarget | TensorToOptions | null | undefined;
export type ProgramBindingDesc = UnknownRecord & {
  inputLen?: number;
  outputLen?: number;
  weightsLen?: number;
  biasLen?: number;
};
type ProgramPlacementTensor = Readonly<{
  data: Readonly<{ length: number }>;
  shape: readonly number[];
  dtype?: unknown;
  device?: unknown;
}>;
export type ProgramBindingPlanProgram = Readonly<{
  handle?: unknown;
  compileEvidence?: () => unknown;
  moduleCompatibility?: (module: unknown, options: unknown) => ProgramModuleCompatibility | null | undefined;
  bufferLayout?: () => ProgramBufferLayout | null;
  inputShape?(): readonly number[] | null;
  outputShape?(): readonly number[] | null;
}>;
export type ProgramBindProgram<TNativeBuffer extends ProgramBindNativeBufferTarget = ProgramBindNativeBufferTarget> =
  ProgramBindingPlanProgram & Readonly<{
    createBuffer?: (kind: "weights" | "bias" | "input" | "output" | string) => TNativeBuffer;
  }>;
type ProgramBindSlotValidationOptions = Readonly<{
  required?: boolean;
  allowLarger?: boolean;
  allowSameElementShape?: boolean;
  expectedShape?: readonly number[] | null;
}>;
type ProgramBindArrayLike = {
  readonly length: number;
};
type ProgramBindNativeBufferTarget = {
  writeFloat32(values: unknown): unknown;
  free(): void;
};

export type TensorPlacementHelpersOptions = Readonly<{
  cloneTensor: (tensor: unknown) => unknown;
}>;

export type ProgramBindValidationHelpersOptions = Readonly<{
  isNativeBuffer: (value: unknown) => boolean;
  nativeBufferByteLength: (value: unknown, name: string) => number;
  f32: (data: unknown) => Float32Array;
  prepareF32: (data: unknown) => { data: Float32Array; shape: readonly number[] };
  valueShape?: (value: unknown) => readonly number[] | null;
}>;

export type ProgramBindPreparedHost = Readonly<{
  kind: "host";
  layout: ProgramBufferLayout | UnknownRecord | null;
  weights: ProgramBindArrayLike;
  bias: ProgramBindArrayLike | null;
  input: ProgramBindArrayLike | null;
  output: ProgramBindArrayLike | null;
  outputShape: TensorShapeTuple | null;
}>;

export type ProgramBindPreparedNative<TNativeBuffer extends ProgramBindNativeBufferTarget = ProgramBindNativeBufferTarget> = Readonly<{
  kind: "native";
  layout: ProgramBufferLayout | UnknownRecord | null;
  bindParams: ProgramBindingsInput;
  ownedBuffers: readonly TNativeBuffer[];
  boundInput: ProgramBindArrayLike | TNativeBuffer | null;
  boundOutput: ProgramBindArrayLike | TNativeBuffer | null;
  outputShape: TensorShapeTuple | null;
}>;

export type ProgramBindPrepared<TNativeBuffer extends ProgramBindNativeBufferTarget = ProgramBindNativeBufferTarget> =
  | ProgramBindPreparedHost
  | ProgramBindPreparedNative<TNativeBuffer>;

export type ProgramNativeBufferBindFields<TNativeBuffer = unknown> = Readonly<{
  weights: TNativeBuffer | null;
  weightsLen: number;
  bias: TNativeBuffer | null;
  biasLen: number;
  input: TNativeBuffer | null;
  inputLen: number;
  output: TNativeBuffer | null;
  outputLen: number;
}>;

export type ProgramBindPreparationHelpers<TNativeBuffer extends ProgramBindNativeBufferTarget = ProgramBindNativeBufferTarget> = Readonly<{
  programBindingPlan(program: ProgramBindingPlanProgram, desc: ProgramBindingDesc, params?: ProgramBindingsInput): ProgramBindingPlan;
  prepareProgramBind(program: ProgramBindProgram<TNativeBuffer>, desc: ProgramBindingDesc, params: ProgramBindingsInput): ProgramBindPrepared<TNativeBuffer>;
  freeOwnedProgramBindBuffers(buffers: readonly TNativeBuffer[]): void;
}>;

export type ProgramNativeBufferBindFieldHelpers<TNativeBuffer = unknown> = Readonly<{
  programNativeBufferBindFields(desc: ProgramBindingDesc, params: ProgramBindingsInput): ProgramNativeBufferBindFields<TNativeBuffer>;
}>;

export type ProgramBindPreparationHelpersOptions<TNativeBuffer extends ProgramBindNativeBufferTarget = ProgramBindNativeBufferTarget> = Readonly<{
  isNativeBuffer: (value: unknown) => value is TNativeBuffer;
  f32: (data: unknown) => Float32Array;
  validateProgramBindParams: (layout: ProgramBufferLayout | UnknownRecord | null | undefined, params: ProgramBindingsInput, shapes?: UnknownRecord) => void;
  outputShape?: (value: unknown) => readonly number[] | null;
}>;

export type ProgramNativeBufferBindFieldHelpersOptions<TNativeBuffer = unknown> = Readonly<{
  isNativeBuffer: (value: unknown) => value is TNativeBuffer;
  requireNativeBuffer: (value: unknown, name: string) => TNativeBuffer | null;
  f32: (data: unknown) => Float32Array;
}>;

export function packedProgramWeightsLen(desc: ProgramBindingDesc): number {
  return desc.weightsLen ?? (Number(desc.inputLen ?? 0) * Number(desc.outputLen ?? 0));
}

export function packedProgramBiasLen(desc: ProgramBindingDesc): number {
  return desc.biasLen ?? Number(desc.outputLen ?? 0);
}

export function isEmptyHostBindingValue(value: unknown): boolean {
  if (Array.isArray(value)) return value.length === 0;
  if (ArrayBuffer.isView(value)) return value.byteLength === 0;
  return false;
}

export function createTensorPlacementHelpers(options: TensorPlacementHelpersOptions) {
  const cloneTensor = options.cloneTensor;
  if (typeof cloneTensor !== "function") {
    throw new Error("tensor placement helpers require a cloneTensor function");
  }

  function normalizeDType(dtype: unknown, label: string) {
    if (dtype === undefined || dtype === null) return "f32";
    if (dtype === "f32" || dtype === "float32") return "f32";
    throw new Error(`${label} dtype ${dtype} is unsupported; eager zgml Tensor currently stores f32 values`);
  }

  function normalizeDevice(device: unknown, label: string) {
    if (device === undefined || device === null || device === "auto") return "cpu";
    if (device === "cpu" || device === "host") return "cpu";
    if (device === "webgpu" || device === "metal") {
      throw new Error(`${label} device ${device} requires an explicit compiled Program; use Tensor.place(program, kind) or program.device(${JSON.stringify(device)})`);
    }
    throw new Error(`${label} device ${device} is unsupported`);
  }

  function parseToArgs(target: TensorToTargetInput, options: TensorToOptions = {}) {
    let dtype;
    let device;
    let copy = false;
    if (typeof target === "string") {
      if (target === "f32" || target === "float32") dtype = target;
      else device = target;
    } else if (target && typeof target === "object") {
      const targetOptions = target as TensorToOptions;
      dtype = targetOptions.dtype;
      device = targetOptions.device ?? targetOptions.backend ?? targetOptions.placement;
      copy = Boolean(targetOptions.copy);
    } else if (target !== undefined && target !== null) {
      throw new Error("Tensor.to target must be a dtype string, device string, or options object");
    }

    if (options && typeof options === "object") {
      if (options.dtype !== undefined) dtype = options.dtype;
      if (options.device !== undefined || options.backend !== undefined || options.placement !== undefined) {
        device = options.device ?? options.backend ?? options.placement;
      }
      if (options.copy !== undefined) copy = Boolean(options.copy);
    }

    return { dtype: normalizeDType(dtype, "Tensor.to"), device: normalizeDevice(device, "Tensor.to"), copy };
  }

  function to(tensor: unknown, target?: TensorToTarget | TensorToOptions, options: TensorToOptions = {}) {
    const spec = parseToArgs(target, options);
    return spec.copy ? cloneTensor(tensor) : tensor;
  }

  function cpu(tensor: unknown, options: TensorToOptions = {}) {
    return to(tensor, { ...options, device: "cpu" }, {});
  }

  function float32(tensor: unknown, options: TensorToOptions = {}) {
    return to(tensor, { ...options, dtype: "f32" }, {});
  }

  function typeAs(tensor: unknown, other: ProgramPlacementTensor, options: TensorToOptions = {}) {
    if (!other || typeof other !== "object") {
      throw new Error("Tensor.type_as requires a tensor-like target");
    }
    const target = { ...options, dtype: other.dtype, device: other.device } as TensorToOptions;
    return to(tensor, target, {});
  }

  function type_as(tensor: unknown, other: ProgramPlacementTensor, options: TensorToOptions = {}) {
    return typeAs(tensor, other, options);
  }

  function validateProgramPlacement(tensor: ProgramPlacementTensor, program: ProgramBindingPlanProgram, kind = "input") {
    if (!program || typeof program.bufferLayout !== "function") return;
    const layout = program.bufferLayout();
    if (!layout || typeof layout !== "object") return;
    const layoutRecord = layout as UnknownRecord;
    const keyedSlot = layoutRecord[kind];
    const slot = keyedSlot && typeof keyedSlot === "object"
      ? keyedSlot
      : Array.isArray(layoutRecord.slots)
        ? layoutRecord.slots.find((candidate) => (
          candidate &&
          typeof candidate === "object" &&
          (candidate as UnknownRecord).name === kind
        ))
        : null;
    if (!slot || typeof slot !== "object") {
      throw new Error(`Tensor.place ${kind} is not a Program buffer slot`);
    }
    const slotRecord = slot as UnknownRecord;
    if (slotRecord.scalarType !== "f32") {
      throw new Error(`Tensor.place ${kind} slot dtype ${slotRecord.scalarType} is unsupported; eager zgml Tensor stores f32 values`);
    }
    const elementCount = Number(slotRecord.elementCount ?? 0);
    if (tensor.data.length !== elementCount) {
      throw new Error(`Tensor.place ${kind} length ${tensor.data.length} does not match Program ${kind} slot length ${elementCount}`);
    }
    let expectedShape = null;
    if (kind === "input" && typeof program.inputShape === "function") {
      expectedShape = program.inputShape();
    } else if (kind === "output" && typeof program.outputShape === "function") {
      expectedShape = program.outputShape();
    }
    if (Array.isArray(expectedShape) && expectedShape.length > 0 && !sameShape(tensor.shape, expectedShape)) {
      throw new Error(`Tensor.place ${kind} shape must be [${expectedShape.join(", ")}], got [${tensor.shape.join(", ")}]`);
    }
  }

  return Object.freeze({
    dtype: () => "f32",
    device: () => "cpu",
    to,
    cpu,
    float: float32,
    float32,
    typeAs,
    type_as,
    validateProgramPlacement,
  });
}

export function createProgramBindValidationHelpers(options: ProgramBindValidationHelpersOptions) {
  const isNativeBuffer = options.isNativeBuffer;
  const nativeBufferByteLength = options.nativeBufferByteLength;
  const f32 = options.f32;
  const prepareF32 = options.prepareF32;
  const valueShape = typeof options.valueShape === "function" ? options.valueShape : () => null;
  if (typeof isNativeBuffer !== "function") {
    throw new Error("Program bind validation helpers require isNativeBuffer");
  }
  if (typeof nativeBufferByteLength !== "function") {
    throw new Error("Program bind validation helpers require nativeBufferByteLength");
  }
  if (typeof f32 !== "function") {
    throw new Error("Program bind validation helpers require f32");
  }
  if (typeof prepareF32 !== "function") {
    throw new Error("Program bind validation helpers require prepareF32");
  }
  const {
    prepareHostValue,
    validateHostValueShape,
  } = createShapedF32Helpers({ f32, prepareF32, valueShape, label: "Program bind validation helpers" });

  function shapeHint(value: unknown): readonly number[] | null {
    return Array.isArray(value) ? value : null;
  }

  function validateProgramBindSlot(layout: ProgramBufferLayout | UnknownRecord | null | undefined, kind: string, value: unknown, options: ProgramBindSlotValidationOptions = {}) {
    const slot = layout && typeof layout === "object" ? (layout as UnknownRecord)[kind] : null;
    if (!slot || typeof slot !== "object") return;
    const slotRecord = slot as UnknownRecord;
    const elementCount = Number(slotRecord.elementCount ?? 0);
    const byteLength = Number(slotRecord.byteLength ?? 0);
    const required = Boolean(options.required) && elementCount > 0;
    const allowLarger = Boolean(options.allowLarger);
    if (value == null) {
      if (required) throw new Error(`${kind} must be provided for this Program`);
      return;
    }
    if (isNativeBuffer(value)) {
      const actualByteLength = nativeBufferByteLength(value, kind);
      if (actualByteLength < byteLength) {
        throw new Error(`${kind} is too small for Program ${kind} slot: ${actualByteLength} < ${byteLength}`);
      }
      return;
    }
    if (elementCount === 0 && isEmptyHostBindingValue(value)) {
      return;
    }
    const prepared = prepareHostValue(value);
    const length = prepared.data.length;
    if (allowLarger ? length < elementCount : length !== elementCount) {
      throw new Error(`${kind} length ${length} does not match Program ${kind} slot length ${elementCount}`);
    }
    const expectedShape = options.expectedShape ?? [];
    validateHostValueShape(prepared, expectedShape, kind, {
      allowFlatCapacity: allowLarger,
      allowSameElementShape: Boolean(options.allowSameElementShape),
    });
  }

  function validateProgramBindParams(layout: ProgramBufferLayout | UnknownRecord | null | undefined, params: ProgramBindingsInput, shapes: UnknownRecord = {}): void {
    if (!params || typeof params !== "object") throw new Error("Program.bind requires a binding object");
    validateProgramBindSlot(layout, "weights", params.weights, { required: true });
    validateProgramBindSlot(layout, "bias", params.bias);
    validateProgramBindSlot(layout, "input", params.input, { expectedShape: shapeHint(shapes.input) });
    validateProgramBindSlot(layout, "output", params.output, { allowLarger: true, allowSameElementShape: true, expectedShape: shapeHint(shapes.output) });
  }

  return Object.freeze({
    validateProgramBindSlot,
    validateProgramBindParams,
  });
}

export function createProgramBindPreparationHelpers<TNativeBuffer extends ProgramBindNativeBufferTarget = ProgramBindNativeBufferTarget>(
  options: ProgramBindPreparationHelpersOptions<TNativeBuffer>,
): ProgramBindPreparationHelpers<TNativeBuffer> {
  const isNativeBuffer = options.isNativeBuffer;
  const f32 = options.f32;
  const validateProgramBindParams = options.validateProgramBindParams;
  const outputShape = options.outputShape;
  if (typeof isNativeBuffer !== "function") {
    throw new Error("Program bind preparation helpers require isNativeBuffer");
  }
  if (typeof f32 !== "function") {
    throw new Error("Program bind preparation helpers require f32");
  }
  if (typeof validateProgramBindParams !== "function") {
    throw new Error("Program bind preparation helpers require validateProgramBindParams");
  }

  function optionalF32(value: unknown) {
    return value == null ? null : f32(value);
  }

  function hostWeights(desc: ProgramBindingDesc, value: unknown) {
    if ((value == null || isEmptyHostBindingValue(value)) && packedProgramWeightsLen(desc) === 0) return new Float32Array(0);
    return f32(value);
  }

  function outputTensorShape(value: unknown) {
    if (typeof outputShape !== "function") return null;
    const shape = outputShape(value);
    return Array.isArray(shape) ? shape.slice() : null;
  }

  function programRetainedShape(program: ProgramBindingPlanProgram, method: "inputShape" | "outputShape") {
    if (!program || typeof program[method] !== "function") return null;
    const shape = program[method]();
    return Array.isArray(shape) ? shape.slice() : null;
  }

  function hasBindingValue(value: unknown): boolean {
    return value !== undefined && value !== null && !isEmptyHostBindingValue(value);
  }

  function frozenShape(shape: readonly number[] | null) {
    return shape ? Object.freeze(shape.slice()) : null;
  }

  function shapeSignature(shape: readonly number[] | null) {
    return shape ? shape.join("x") : "null";
  }

  function programBindingPlanSignature(fields: {
    readonly accepted: boolean;
    readonly mode: string;
    readonly layout: ProgramBufferLayout | null;
    readonly inputShape: TensorShapeTuple | null;
    readonly outputShape: TensorShapeTuple | null;
    readonly weightsLen: number;
    readonly biasLen: number;
    readonly inputLen: number | null;
    readonly outputLen: number | null;
    readonly hasWeights: boolean;
    readonly hasBias: boolean;
    readonly hasInput: boolean;
    readonly hasOutput: boolean;
    readonly usesNativeBuffers: boolean;
    readonly diagnostics: readonly ProgramBindingDiagnostic[];
  }): string {
    return [
      "program-binding-plan",
      `accepted=${fields.accepted ? 1 : 0}`,
      `mode=${fields.mode}`,
      `layout=${fields.layout && typeof fields.layout.signature === "string" ? fields.layout.signature : "null"}`,
      `inputShape=${shapeSignature(fields.inputShape)}`,
      `outputShape=${shapeSignature(fields.outputShape)}`,
      `weights=${fields.weightsLen}`,
      `bias=${fields.biasLen}`,
      `input=${fields.inputLen ?? "null"}`,
      `output=${fields.outputLen ?? "null"}`,
      `has=${fields.hasWeights ? 1 : 0}${fields.hasBias ? 1 : 0}${fields.hasInput ? 1 : 0}${fields.hasOutput ? 1 : 0}`,
      `native=${fields.usesNativeBuffers ? 1 : 0}`,
      `diagnostics=${fields.diagnostics.map((diagnostic) => diagnostic.code).join(",")}`,
    ].join("|");
  }

  function programBindingPlan(program: ProgramBindingPlanProgram, desc: ProgramBindingDesc, params: ProgramBindingsInput = {}): ProgramBindingPlan {
    const bindParams = params && typeof params === "object" ? params : {};
    const diagnostics: ProgramBindingDiagnostic[] = [];
    let accepted = false;
    let reason: string | null = null;
    let layout: ProgramBufferLayout | null = null;
    let inputShape: readonly number[] | null = null;
    let outputShape: readonly number[] | null = null;
    let usesNativeBuffers = false;
    try {
      if (!program || typeof program.bufferLayout !== "function") {
        throw new Error("Program.bindingPlan requires a Program with bufferLayout");
      }
      layout = program.bufferLayout();
      inputShape = programRetainedShape(program, "inputShape");
      outputShape = programRetainedShape(program, "outputShape");
      validateProgramBindModuleCompatibility(program, params);
      validateProgramBindParams(layout, params, {
        input: inputShape,
        output: outputShape,
      });
      usesNativeBuffers = [bindParams.weights, bindParams.bias, bindParams.input, bindParams.output].some(isNativeBuffer);
      accepted = true;
    } catch (err) {
      reason = err && typeof (err as Error).message === "string" ? (err as Error).message : String(err);
      diagnostics.push(Object.freeze({
        code: "binding-invalid",
        message: reason,
      }));
    }
    const mode = accepted ? (usesNativeBuffers ? "native" : "host") : "invalid";
    const frozenInputShape = frozenShape(inputShape);
    const frozenOutputShape = frozenShape(outputShape);
    const weightsLen = packedProgramWeightsLen(desc);
    const biasLen = packedProgramBiasLen(desc);
    const inputLen = desc.inputLen ?? null;
    const outputLen = desc.outputLen ?? null;
    const hasWeights = hasBindingValue(bindParams.weights);
    const hasBias = hasBindingValue(bindParams.bias);
    const hasInput = hasBindingValue(bindParams.input);
    const hasOutput = hasBindingValue(bindParams.output);
    const frozenDiagnostics = Object.freeze(diagnostics);
    return Object.freeze({
      kind: "zgml.program.binding-plan",
      signature: programBindingPlanSignature({
        accepted,
        mode,
        layout,
        inputShape: frozenInputShape,
        outputShape: frozenOutputShape,
        weightsLen,
        biasLen,
        inputLen,
        outputLen,
        hasWeights,
        hasBias,
        hasInput,
        hasOutput,
        usesNativeBuffers,
        diagnostics: frozenDiagnostics,
      }),
      accepted,
      canBind: accepted,
      mode,
      reason,
      diagnostics: frozenDiagnostics,
      bufferLayout: layout,
      inputShape: frozenInputShape,
      outputShape: frozenOutputShape,
      weightsLen,
      biasLen,
      inputLen,
      outputLen,
      hasWeights,
      hasBias,
      hasInput,
      hasOutput,
      usesNativeBuffers,
    }) as ProgramBindingPlan;
  }

  function freeOwnedProgramBindBuffers(buffers: readonly TNativeBuffer[]) {
    for (const buffer of buffers) buffer.free();
  }

  function prepareProgramBind(program: ProgramBindProgram<TNativeBuffer>, desc: ProgramBindingDesc, params: ProgramBindingsInput): ProgramBindPrepared<TNativeBuffer> {
    if (!program || typeof program.bufferLayout !== "function" || typeof program.createBuffer !== "function") {
      throw new Error("Program.bind requires a Program with bufferLayout and createBuffer");
    }
    validateProgramBindModuleCompatibility(program, params);
    const layout = program.bufferLayout();
    validateProgramBindParams(layout, params, {
      input: programRetainedShape(program, "inputShape"),
      output: programRetainedShape(program, "outputShape"),
    });
    const usesNativeBuffers = [params.weights, params.bias, params.input, params.output].some(isNativeBuffer);
    if (!usesNativeBuffers) {
      return {
        kind: "host",
        layout,
        weights: hostWeights(desc, params.weights),
        bias: optionalF32(params.bias),
        input: optionalF32(params.input),
        output: optionalF32(params.output),
        outputShape: outputTensorShape(params.output),
      };
    }

    const bindParams = { ...params };
    const ownedBuffers: TNativeBuffer[] = [];
    try {
      if (!isNativeBuffer(bindParams.weights) && packedProgramWeightsLen(desc) > 0) {
        const weightsBuffer = program.createBuffer("weights");
        weightsBuffer.writeFloat32(f32(bindParams.weights));
        bindParams.weights = weightsBuffer;
        ownedBuffers.push(weightsBuffer);
      }
      if (bindParams.bias != null && !isNativeBuffer(bindParams.bias) && packedProgramBiasLen(desc) > 0) {
        const biasBuffer = program.createBuffer("bias");
        biasBuffer.writeFloat32(f32(bindParams.bias));
        bindParams.bias = biasBuffer;
        ownedBuffers.push(biasBuffer);
      }
    } catch (err) {
      freeOwnedProgramBindBuffers(ownedBuffers);
      throw err;
    }

    return {
      kind: "native",
      layout,
      bindParams,
      ownedBuffers,
      boundInput: isNativeBuffer(params.input) ? params.input : optionalF32(params.input),
      boundOutput: isNativeBuffer(params.output) ? params.output : optionalF32(params.output),
      outputShape: outputTensorShape(params.output),
    };
  }

  return Object.freeze({
    programBindingPlan,
    prepareProgramBind,
    freeOwnedProgramBindBuffers,
  });
}

export function createProgramNativeBufferBindFieldHelpers<TNativeBuffer = unknown>(
  options: ProgramNativeBufferBindFieldHelpersOptions<TNativeBuffer>,
): ProgramNativeBufferBindFieldHelpers<TNativeBuffer> {
  const isNativeBuffer = options.isNativeBuffer;
  const requireNativeBuffer = options.requireNativeBuffer;
  const f32 = options.f32;
  if (typeof isNativeBuffer !== "function") {
    throw new Error("Program native-buffer bind field helpers require isNativeBuffer");
  }
  if (typeof requireNativeBuffer !== "function") {
    throw new Error("Program native-buffer bind field helpers require requireNativeBuffer");
  }
  if (typeof f32 !== "function") {
    throw new Error("Program native-buffer bind field helpers require f32");
  }

  function programNativeBufferBindFields(desc: ProgramBindingDesc, params: ProgramBindingsInput): ProgramNativeBufferBindFields<TNativeBuffer> {
    const expectedWeightsLen = packedProgramWeightsLen(desc);
    const expectedBiasLen = packedProgramBiasLen(desc);
    let weights = null;
    if (isNativeBuffer(params.weights)) {
      if (expectedWeightsLen === 0) throw new Error("weights must be empty for this Program");
      weights = requireNativeBuffer(params.weights, "weights");
    } else if (expectedWeightsLen !== 0) {
      throw new Error("weights must be a zgml NativeBuffer");
    } else if (params.weights != null && f32(params.weights).length !== 0) {
      throw new Error("weights must be empty for this Program");
    }

    let bias = null;
    if (isNativeBuffer(params.bias)) {
      if (expectedBiasLen === 0) throw new Error("bias must be empty for this Program");
      bias = requireNativeBuffer(params.bias, "bias");
    } else if (params.bias != null) {
      if (expectedBiasLen === 0 && f32(params.bias).length !== 0) {
        throw new Error("bias must be empty for this Program");
      }
      if (expectedBiasLen !== 0) throw new Error("bias must be a zgml NativeBuffer");
    }

    const input = isNativeBuffer(params.input) ? requireNativeBuffer(params.input, "input") : null;
    const output = isNativeBuffer(params.output) ? requireNativeBuffer(params.output, "output") : null;
    return {
      weights,
      weightsLen: weights ? expectedWeightsLen : 0,
      bias,
      biasLen: bias ? expectedBiasLen : 0,
      input,
      inputLen: input ? Number(desc.inputLen ?? 0) : 0,
      output,
      outputLen: output ? Number(desc.outputLen ?? 0) : 0,
    };
  }

  return Object.freeze({
    programNativeBufferBindFields,
  });
}
