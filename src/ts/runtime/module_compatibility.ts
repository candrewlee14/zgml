"use strict";

import {
  compilerSignaturesFromCompilerEvidence,
} from "./compiler_signatures.js";
import { compileSupportRejectionReason } from "./compile_support.js";
import {
  kernelPlanInputShape,
  kernelPlanOutputShape,
  kernelPlanInputLen,
  kernelPlanOutputLen,
} from "./kernel_plan.js";
import type {
  ModuleCompileSupport,
  ModuleParameterPlacementOptions,
  ProgramCompileEvidence,
  ProgramModuleCompatibility,
  ProgramModuleCompatibilityDiagnostic,
  ProgramModuleCompatibilityDiagnosticCode,
} from "../public_api.js";

type UnknownRecord = Record<string, unknown>;
type ModuleCompatibilityEvidence = ProgramCompileEvidence | UnknownRecord;
type ModuleCompatibilitySupport = ModuleCompileSupport | UnknownRecord;
type ModuleCompatibilityDiagnosticRecord = UnknownRecord & {
  code: ProgramModuleCompatibilityDiagnosticCode;
  message: string;
};
type ModuleCompatibilityLengthFields = Readonly<{
  inputLen: unknown;
  outputLen: unknown;
  weightsLen: unknown;
  biasLen: unknown;
}>;
type ModuleCompatibilityShapeFields = Readonly<{
  inputShape: readonly unknown[] | null;
  outputShape: readonly unknown[] | null;
}>;
type ModuleCompatibilityTraceLike = Readonly<{
  ops?: readonly UnknownRecord[];
  inputShape?: readonly unknown[];
  outputShape?: readonly unknown[];
}>;
type ModuleCompatibilityKernelPlanLike = UnknownRecord & {
  opCount?: unknown;
  dispatchCount?: unknown;
  descriptorCount?: unknown;
  elidedOpCount?: unknown;
  weightsLen?: unknown;
  biasLen?: unknown;
};

type ModuleCompatibilityModuleSource = Readonly<{
  readonly compileSupport?: unknown;
}>;

function sameNumberArray(left: unknown, right: unknown): boolean {
  return Array.isArray(left) &&
    Array.isArray(right) &&
    left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

function record(value: unknown): UnknownRecord | null {
  return value && typeof value === "object" ? value as UnknownRecord : null;
}

function kernelPlanRecord(value: unknown): ModuleCompatibilityKernelPlanLike | null {
  return record(value) as ModuleCompatibilityKernelPlanLike | null;
}

function traceRecord(value: unknown): ModuleCompatibilityTraceLike | null {
  return record(value) as ModuleCompatibilityTraceLike | null;
}

function moduleCompatibilityDiagnostic(code: ProgramModuleCompatibilityDiagnosticCode, message: string, extra: UnknownRecord = {}): ModuleCompatibilityDiagnosticRecord {
  const diagnostic: UnknownRecord = { code, message, ...extra };
  if (Array.isArray(diagnostic.programShape)) diagnostic.programShape = Object.freeze(diagnostic.programShape.slice());
  if (Array.isArray(diagnostic.moduleShape)) diagnostic.moduleShape = Object.freeze(diagnostic.moduleShape.slice());
  return Object.freeze(diagnostic) as ModuleCompatibilityDiagnosticRecord;
}

function freezeModuleCompatibilityDiagnostics(diagnostics: readonly ModuleCompatibilityDiagnosticRecord[]): readonly ProgramModuleCompatibilityDiagnostic[] {
  return Object.freeze(diagnostics.map((diagnostic) => {
    if (Object.isFrozen(diagnostic)) return diagnostic;
    return moduleCompatibilityDiagnostic(diagnostic.code, diagnostic.message, diagnostic);
  })) as readonly ProgramModuleCompatibilityDiagnostic[];
}

function traceSignature(trace: ModuleCompatibilityTraceLike | null | undefined): string {
  return trace && Array.isArray(trace.ops)
    ? trace.ops.map((op) => `${String(op.path)}:${String(op.op)}`).join("|")
    : "";
}

function moduleCompatibilityLengths(source: ModuleCompatibilityEvidence | ModuleCompatibilitySupport | null | undefined): ModuleCompatibilityLengthFields {
  const sourceRecord = record(source);
  const kernelPlan = kernelPlanRecord(sourceRecord?.kernelPlan);
  return {
    inputLen: sourceRecord && sourceRecord.inputLen !== undefined ? sourceRecord.inputLen : kernelPlanInputLen(kernelPlan),
    outputLen: sourceRecord && sourceRecord.outputLen !== undefined ? sourceRecord.outputLen : kernelPlanOutputLen(kernelPlan),
    weightsLen: sourceRecord && sourceRecord.weightsLen !== undefined ? sourceRecord.weightsLen : kernelPlan && kernelPlan.weightsLen,
    biasLen: sourceRecord && sourceRecord.biasLen !== undefined ? sourceRecord.biasLen : kernelPlan && kernelPlan.biasLen,
  };
}

function moduleCompatibilityShapes(source: ModuleCompatibilityEvidence | ModuleCompatibilitySupport | null | undefined): ModuleCompatibilityShapeFields {
  const sourceRecord = record(source);
  const kernelPlan = kernelPlanRecord(sourceRecord?.kernelPlan);
  const trace = traceRecord(sourceRecord?.trace);
  const inputShape = sourceRecord && Array.isArray(sourceRecord.inputShape)
    ? sourceRecord.inputShape
    : (kernelPlanInputShape(kernelPlan) ?? (trace && Array.isArray(trace.inputShape) ? trace.inputShape : null));
  const outputShape = sourceRecord && Array.isArray(sourceRecord.outputShape)
    ? sourceRecord.outputShape
    : (kernelPlanOutputShape(kernelPlan) ?? (trace && Array.isArray(trace.outputShape) ? trace.outputShape : null));
  return { inputShape, outputShape };
}

function signatureField(signatures: Record<string, unknown> | null | undefined, field: string): string {
  return signatures && typeof signatures[field] === "string" ? signatures[field] as string : "";
}

function signatureMismatchDiagnostic(
  code: Extract<ProgramModuleCompatibilityDiagnosticCode, "ir-mismatch" | "kernel-plan-mismatch" | "memory-layout-mismatch" | "parameter-layout-mismatch" | "buffer-layout-mismatch">,
  message: string,
  signatureKind: string,
  programSignature: string,
  moduleSignature: string,
) {
  return moduleCompatibilityDiagnostic(code, message, {
    signatureKind,
    programSignature,
    moduleSignature,
  });
}

function moduleCompatibilitySignature(fields: {
	  readonly compatible: boolean;
	  readonly programKind: unknown;
	  readonly moduleKind: unknown;
	  readonly diagnostics: readonly ModuleCompatibilityDiagnosticRecord[];
	}) {
  const diagnosticCodes = fields.diagnostics.map((diagnostic) => diagnostic.code).join(",");
  return [
    "program-module-compatibility",
    `compatible=${fields.compatible ? 1 : 0}`,
    `program=${fields.programKind ?? "null"}`,
    `module=${fields.moduleKind ?? "null"}`,
    `diagnostics=${fields.diagnostics.length}`,
    `codes=${diagnosticCodes}`,
  ].join("|");
}

function evidenceInputShape(evidence: ModuleCompatibilityEvidence | null | undefined) {
  const evidenceRecord = record(evidence);
  const kernelPlan = kernelPlanRecord(evidenceRecord?.kernelPlan);
  const inputShape = kernelPlanInputShape(kernelPlan);
  if (inputShape) return inputShape;
  const trace = traceRecord(evidenceRecord?.trace);
  if (trace && Array.isArray(trace.inputShape)) return trace.inputShape;
  return null;
}

export function moduleOptionsWithEvidenceShape(evidence: ModuleCompatibilityEvidence | null | undefined, options: ModuleParameterPlacementOptions = {}) {
  if (evidence && (evidence.kind === "tiny-linear" || evidence.nativePath === "tiny-linear")) {
    const { inputShape: _inputShape, ...rest } = options as UnknownRecord;
    return rest;
  }
  if (options && Object.prototype.hasOwnProperty.call(options, "inputShape") && options.inputShape !== undefined) {
    return options;
  }
  const inputShape = evidenceInputShape(evidence);
  if (!inputShape) return options;
  return {
    ...options,
    inputShape: inputShape.slice(),
  };
}

export function moduleCompatibilityForSupportEvidence(
  evidence: ModuleCompatibilityEvidence | null | undefined,
  support: ModuleCompatibilitySupport | null | undefined,
  initialDiagnostics: readonly ModuleCompatibilityDiagnosticRecord[] = [],
): ProgramModuleCompatibility {
  const diagnostics = initialDiagnostics.slice();
  if (!evidence) {
    diagnostics.push(moduleCompatibilityDiagnostic(
      "missing-program-evidence",
      "Program has no module compile evidence for module compatibility preflight",
    ));
  }
  if (!support) {
    diagnostics.push(moduleCompatibilityDiagnostic(
      "module-missing-compiler",
      "module bindings have no compileSupport evidence for compatibility preflight",
    ));
  } else if (!support.supported) {
    diagnostics.push(moduleCompatibilityDiagnostic(
      "module-unsupported",
      compileSupportRejectionReason(support, { fallbackReason: "module is unsupported by the native Program compiler" }),
    ));
  }

  if (evidence && support && support.supported) {
    if (evidence.nativePath !== support.nativePath) {
      diagnostics.push(moduleCompatibilityDiagnostic(
        "native-path-mismatch",
        `Program native path ${evidence.nativePath} does not match module native path ${support.nativePath}`,
        { programNativePath: evidence.nativePath, moduleNativePath: support.nativePath },
      ));
    }
    if (evidence.modelKind !== support.modelKind) {
      diagnostics.push(moduleCompatibilityDiagnostic(
        "model-kind-mismatch",
        `Program model kind ${evidence.modelKind} does not match module model kind ${support.modelKind}`,
        { programModelKind: evidence.modelKind, moduleModelKind: support.modelKind },
      ));
    }
    if (evidence.layerCount !== support.layerCount) {
      diagnostics.push(moduleCompatibilityDiagnostic(
        "layer-count-mismatch",
        `Program layer count ${evidence.layerCount} does not match module layer count ${support.layerCount}`,
        { programLayerCount: evidence.layerCount, moduleLayerCount: support.layerCount },
      ));
    }

    const evidenceRecord = record(evidence);
    const supportRecord = record(support);
    const evidenceKernelPlan = kernelPlanRecord(evidenceRecord?.kernelPlan);
    const supportKernelPlan = kernelPlanRecord(supportRecord?.kernelPlan);
    const programLengths = moduleCompatibilityLengths(evidenceKernelPlan ? evidenceKernelPlan : evidence);
    const moduleLengths = moduleCompatibilityLengths(support);
    for (const field of ["inputLen", "outputLen", "weightsLen", "biasLen"]) {
      if (programLengths[field as keyof typeof programLengths] !== moduleLengths[field as keyof typeof moduleLengths]) {
        diagnostics.push(moduleCompatibilityDiagnostic(
          `${field}-mismatch` as ProgramModuleCompatibilityDiagnosticCode,
          `Program ${field} ${programLengths[field as keyof typeof programLengths]} does not match module ${field} ${moduleLengths[field as keyof typeof moduleLengths]}`,
          { field, programValue: programLengths[field as keyof typeof programLengths], moduleValue: moduleLengths[field as keyof typeof moduleLengths] },
        ));
      }
    }

    const programShapes = moduleCompatibilityShapes(evidenceKernelPlan ? evidenceKernelPlan : evidence);
    const moduleShapes = moduleCompatibilityShapes(support);
    if (!sameNumberArray(programShapes.inputShape, moduleShapes.inputShape)) {
      diagnostics.push(moduleCompatibilityDiagnostic(
        "input-shape-mismatch",
        `Program input shape [${programShapes.inputShape ? programShapes.inputShape.join(",") : "unknown"}] does not match module input shape [${moduleShapes.inputShape ? moduleShapes.inputShape.join(",") : "unknown"}]`,
        { programShape: programShapes.inputShape ?? [], moduleShape: moduleShapes.inputShape ?? [] },
      ));
    }
    if (!sameNumberArray(programShapes.outputShape, moduleShapes.outputShape)) {
      diagnostics.push(moduleCompatibilityDiagnostic(
        "output-shape-mismatch",
        `Program output shape [${programShapes.outputShape ? programShapes.outputShape.join(",") : "unknown"}] does not match module output shape [${moduleShapes.outputShape ? moduleShapes.outputShape.join(",") : "unknown"}]`,
        { programShape: programShapes.outputShape ?? [], moduleShape: moduleShapes.outputShape ?? [] },
      ));
    }

    if (evidence.kind === "module" || (evidenceKernelPlan && supportKernelPlan)) {
      const programSignatures = compilerSignaturesFromCompilerEvidence(evidence);
      const moduleSignatures = compilerSignaturesFromCompilerEvidence(support);
      const programIrSignature = signatureField(programSignatures, "ir");
      const moduleIrSignature = signatureField(moduleSignatures, "ir");
      if (programIrSignature !== moduleIrSignature) {
        diagnostics.push(signatureMismatchDiagnostic(
          "ir-mismatch",
          "Program Tensor Program IR does not match module compile support",
          "ir",
          programIrSignature,
          moduleIrSignature,
        ));
      }
      if (!supportKernelPlan) {
        diagnostics.push(moduleCompatibilityDiagnostic(
          "missing-module-kernel-plan",
          "module support has no kernelPlan to compare with module Program evidence",
        ));
      } else {
        const programKernelPlanSignature = signatureField(programSignatures, "kernelPlan");
        const moduleKernelPlanSignature = signatureField(moduleSignatures, "kernelPlan");
        if (!evidenceKernelPlan ||
          evidenceKernelPlan.opCount !== supportKernelPlan.opCount ||
          evidenceKernelPlan.dispatchCount !== supportKernelPlan.dispatchCount ||
          (evidenceKernelPlan.descriptorCount ?? evidenceKernelPlan.dispatchCount) !==
            (supportKernelPlan.descriptorCount ?? supportKernelPlan.dispatchCount) ||
          (evidenceKernelPlan.elidedOpCount ?? 0) !== (supportKernelPlan.elidedOpCount ?? 0) ||
          programKernelPlanSignature !== moduleKernelPlanSignature) {
          diagnostics.push(signatureMismatchDiagnostic(
            "kernel-plan-mismatch",
            "Program kernel plan does not match module compile support",
            "kernel-plan",
            programKernelPlanSignature,
            moduleKernelPlanSignature,
          ));
        }
        const programMemoryLayoutSignature = signatureField(programSignatures, "memoryLayout");
        const moduleMemoryLayoutSignature = signatureField(moduleSignatures, "memoryLayout");
        if (programMemoryLayoutSignature !== moduleMemoryLayoutSignature) {
          diagnostics.push(signatureMismatchDiagnostic(
            "memory-layout-mismatch",
            "Program memory layout does not match module memory layout",
            "memory-layout",
            programMemoryLayoutSignature,
            moduleMemoryLayoutSignature,
          ));
        }
        const programParameterLayoutSignature = signatureField(programSignatures, "parameterLayout");
        const moduleParameterLayoutSignature = signatureField(moduleSignatures, "parameterLayout");
        if (programParameterLayoutSignature !== moduleParameterLayoutSignature) {
          diagnostics.push(signatureMismatchDiagnostic(
            "parameter-layout-mismatch",
            "Program parameter layout does not match module parameter layout",
            "parameter-layout",
            programParameterLayoutSignature,
            moduleParameterLayoutSignature,
          ));
        }
        const programBufferLayoutSignature = signatureField(programSignatures, "bufferLayout");
        const moduleBufferLayoutSignature = signatureField(moduleSignatures, "bufferLayout");
        if (programBufferLayoutSignature !== moduleBufferLayoutSignature) {
          diagnostics.push(signatureMismatchDiagnostic(
            "buffer-layout-mismatch",
            "Program buffer layout does not match module buffer layout",
            "buffer-layout",
            programBufferLayoutSignature,
            moduleBufferLayoutSignature,
          ));
        }
      }
    } else if (evidence.kind === "tiny-linear") {
      if (supportKernelPlan) {
        diagnostics.push(moduleCompatibilityDiagnostic(
          "program-kind-mismatch",
          "tiny-linear Program evidence cannot bind a module that lowers through kernelPlan evidence",
        ));
      }
      if (traceSignature(traceRecord(record(evidence)?.trace)) !== traceSignature(traceRecord(record(support)?.trace))) {
        diagnostics.push(moduleCompatibilityDiagnostic(
          "trace-mismatch",
          "Program trace does not match module trace",
        ));
      }
    }
  }

  const compatible = diagnostics.length === 0;
  const programKind = evidence && (evidence.kind === "module" || evidence.kind === "tiny-linear") ? evidence.kind : null;
  const moduleKind = support && (support.modelKind === "module" || support.modelKind === "tiny-linear" || support.modelKind === "tiny-mlp") ? support.modelKind : null;
  const frozenDiagnostics = freezeModuleCompatibilityDiagnostics(diagnostics);
  return Object.freeze({
    kind: "zgml.program.module-compatibility",
    signature: moduleCompatibilitySignature({
      compatible,
      programKind,
      moduleKind,
      diagnostics: frozenDiagnostics,
    }),
    compatible,
    reason: compatible ? null : diagnostics[0].message,
    programKind,
    moduleKind,
    diagnostics: frozenDiagnostics,
  });
}

export function moduleCompatibilityForProgramEvidence(evidence: ModuleCompatibilityEvidence | null | undefined, module: ModuleCompatibilityModuleSource | null | undefined, options: ModuleParameterPlacementOptions = {}): ProgramModuleCompatibility {
  const diagnostics: ModuleCompatibilityDiagnosticRecord[] = [];
  const moduleOptions = moduleOptionsWithEvidenceShape(evidence, options);
  let support: ModuleCompatibilitySupport | null = null;
  const compileSupport = module && module.compileSupport;
  if (!module || typeof compileSupport !== "function") {
    diagnostics.push(moduleCompatibilityDiagnostic(
      "module-missing-compiler",
      "module does not expose compileSupport for compatibility preflight",
    ));
  } else {
    try {
      support = compileSupport.call(module, moduleOptions) as ModuleCompatibilitySupport;
      if (!support || !support.supported) {
        diagnostics.push(moduleCompatibilityDiagnostic(
          "module-unsupported",
          compileSupportRejectionReason(support, { fallbackReason: "module is unsupported by the native Program compiler" }),
        ));
      }
    } catch (err) {
      diagnostics.push(moduleCompatibilityDiagnostic(
        "module-analysis-failed",
        err && (err as Error).message ? String((err as Error).message) : String(err),
      ));
    }
  }

  return moduleCompatibilityForSupportEvidence(evidence, support, diagnostics);
}
