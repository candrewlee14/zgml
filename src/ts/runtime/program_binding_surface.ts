import type {
  NativeBufferLifetimeHelpers,
  NativeBufferLifetimeTarget,
} from "./native_buffer.js";
import type {
  ProgramModuleBindingHelpers,
  ProgramModuleBindingOptions,
} from "./program_module_binding.js";
import type {
  ProgramBindPreparationHelpersOptions,
  ProgramBindValidationHelpersOptions,
  ProgramBindPrepared,
  ProgramBindPreparationHelpers,
  ProgramBindingDesc,
  ProgramBindingPlanProgram,
  ProgramBindProgram,
  ProgramNativeBufferBindFields,
  ProgramNativeBufferBindFieldHelpers,
  ProgramNativeBufferBindFieldHelpersOptions,
} from "./tensor_placement.js";
import { tsRuntimeManifestPolicy } from "../internal/product_manifest.js";
import type {
  ProgramBindingPlan,
  ProgramBindings,
  ProgramBufferLayout,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
export type ProgramBindingSurfacePreparedBind<TNativeBuffer extends NativeBufferLifetimeTarget> =
  ProgramBindPrepared<TNativeBuffer>;

export type ProgramBindingSurfaceNativeBufferBindFields<TNativeBuffer extends NativeBufferLifetimeTarget> =
  ProgramNativeBufferBindFields<TNativeBuffer>;

export type ProgramBindingSurfacePreparedF32 = Readonly<{
  data: Float32Array;
  shape: readonly number[];
}>;

export type ProgramBindingSurfaceValueShape = readonly number[] | null;

export type ProgramBindingSurfaceModuleHelpers<TNativeBuffer extends NativeBufferLifetimeTarget = NativeBufferLifetimeTarget> =
  ProgramModuleBindingHelpers<TNativeBuffer>;

export type ProgramBindingSurfaceValidationHelpers = Readonly<{
  validateProgramBindParams(layout: ProgramBufferLayout | UnknownRecord | null | undefined, params: ProgramBindings | UnknownRecord, shapes?: UnknownRecord): void;
}>;

export type ProgramBindingSurfacePreparationHelpers<TNativeBuffer extends NativeBufferLifetimeTarget> = Readonly<{
  programBindingPlan(program: ProgramBindingPlanProgram, desc: ProgramBindingDesc, params?: ProgramBindings | UnknownRecord): ProgramBindingPlan;
  prepareProgramBind(program: ProgramBindProgram<TNativeBuffer>, desc: ProgramBindingDesc, params: ProgramBindings | UnknownRecord): ProgramBindingSurfacePreparedBind<TNativeBuffer>;
  freeOwnedProgramBindBuffers(buffers: readonly TNativeBuffer[]): void;
}>;

export type ProgramBindingSurfaceNativeBufferFieldHelpers<TNativeBuffer extends NativeBufferLifetimeTarget> = Readonly<{
  programNativeBufferBindFields(desc: ProgramBindingDesc, params: ProgramBindings | UnknownRecord): ProgramBindingSurfaceNativeBufferBindFields<TNativeBuffer>;
}>;

export type ProgramBindingSurfaceOptions<TNativeBuffer extends NativeBufferLifetimeTarget = NativeBufferLifetimeTarget> = Readonly<{
  readonly isNativeBuffer: (value: unknown) => value is TNativeBuffer;
  readonly nativeBufferByteLength?: (value: unknown, name: string) => number;
  readonly f32: (data: unknown) => Float32Array;
  readonly prepareF32: (data: unknown) => ProgramBindingSurfacePreparedF32;
  readonly valueShape: (value: unknown) => ProgramBindingSurfaceValueShape;
  readonly createNativeBufferLifetimeHelpers: (options: {
    readonly isNativeBuffer: (value: unknown) => value is TNativeBuffer;
  }) => NativeBufferLifetimeHelpers<TNativeBuffer>;
  readonly createProgramModuleBindingHelpers: (
    options: ProgramModuleBindingOptions<TNativeBuffer>,
  ) => ProgramModuleBindingHelpers<TNativeBuffer>;
  readonly createProgramBindValidationHelpers: (
    options: ProgramBindValidationHelpersOptions,
  ) => ProgramBindingSurfaceValidationHelpers;
  readonly createProgramBindPreparationHelpers: (
    options: ProgramBindPreparationHelpersOptions<TNativeBuffer>,
  ) => ProgramBindPreparationHelpers<TNativeBuffer>;
  readonly createProgramNativeBufferBindFieldHelpers: (
    options: ProgramNativeBufferBindFieldHelpersOptions<TNativeBuffer>,
  ) => ProgramNativeBufferBindFieldHelpers<TNativeBuffer>;
}>;

export type ProgramBindingSurface<TNativeBuffer extends NativeBufferLifetimeTarget = NativeBufferLifetimeTarget> = Readonly<
  NativeBufferLifetimeHelpers<TNativeBuffer> &
  ProgramBindingSurfaceModuleHelpers<TNativeBuffer> &
  ProgramBindingSurfaceValidationHelpers &
  ProgramBindingSurfacePreparationHelpers<TNativeBuffer> &
  ProgramBindingSurfaceNativeBufferFieldHelpers<TNativeBuffer>
>;

export function createProgramBindingSurface<TNativeBuffer extends NativeBufferLifetimeTarget = NativeBufferLifetimeTarget>(
  options: ProgramBindingSurfaceOptions<TNativeBuffer>,
): ProgramBindingSurface<TNativeBuffer> {
  const { requireNativeBuffer, uniqueNativeBuffers } = options.createNativeBufferLifetimeHelpers({
    isNativeBuffer: options.isNativeBuffer,
  });
  const nativeBufferByteLength = options.nativeBufferByteLength ?? (
    (value: unknown, name: string): number => requireNativeBuffer(value, name)?.byteLength ?? 0
  );
  const { bindModuleThroughProgram } = options.createProgramModuleBindingHelpers({ uniqueNativeBuffers });
  const { validateProgramBindParams } = options.createProgramBindValidationHelpers({
    isNativeBuffer: options.isNativeBuffer,
    nativeBufferByteLength,
    f32: options.f32,
    prepareF32: options.prepareF32,
    valueShape: options.valueShape,
  });
  const {
    programBindingPlan,
    prepareProgramBind,
    freeOwnedProgramBindBuffers,
  } = options.createProgramBindPreparationHelpers({
    isNativeBuffer: options.isNativeBuffer,
    f32: options.f32,
    validateProgramBindParams,
    outputShape: options.valueShape,
  });
  const { programNativeBufferBindFields } = options.createProgramNativeBufferBindFieldHelpers({
    isNativeBuffer: options.isNativeBuffer,
    requireNativeBuffer,
    f32: options.f32,
  });

  return Object.freeze({
    requireNativeBuffer,
    uniqueNativeBuffers,
    bindModuleThroughProgram,
    validateProgramBindParams,
    programBindingPlan,
    prepareProgramBind,
    freeOwnedProgramBindBuffers,
    programNativeBufferBindFields,
  }) as ProgramBindingSurface<TNativeBuffer>;
}

export const programBindingSurfaceManifest = Object.freeze({
  kind: "zgml-program-binding-surface",
  ...tsRuntimeManifestPolicy("src/ts/runtime/program_binding_surface.ts", "Program binding helper composition"),
});
