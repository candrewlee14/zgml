"use strict";

import { tsProductManifestPolicy } from "./internal/product_manifest.js";

export const programDeviceManifest = Object.freeze({
  kind: "zgml-program-device",
  ...tsProductManifestPolicy("src/ts/program_device.ts"),
  runtimePath: "Program -> ProgramDevice -> NativeBuffer",
});

export type {
  HostGpuBufferImportSource,
  ProgramDeviceBufferImportDescriptor,
  ProgramDeviceBufferImportOptions,
  ProgramDeviceBufferImportSource,
  ProgramDeviceImportBufferSource,
  ProgramDeviceInfo,
  WebGpuInteropImportFields,
  WebGpuInteropImportSource,
  WebGpuInteropSymbols,
} from "./public_api.js";

export {
  assert_program_device_info,
  assertProgramDeviceInfo,
  createProgramDeviceClass,
  isProgramDeviceInfo,
  matches_program_device_info_signature,
  matchesProgramDeviceInfoSignature,
  programDeviceInfo,
  requireProgramDeviceInfo,
} from "./runtime/program_device.js";
