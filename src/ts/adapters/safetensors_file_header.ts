import {
  createSafetensorsFileHeaderHelpers,
} from "../runtime/model_source.js";

export type AdapterSafetensorsFileHeaderFs<Fd = unknown> = Readonly<{
  openSync(path: string, flags: string): Fd;
  closeSync(fd: Fd): void;
  readSync(fd: Fd, buffer: Uint8Array, offset: number, length: number, position: number): number;
}>;

export function createAdapterSafetensorsFileHeaderHelpers<Fd = unknown>(fs: AdapterSafetensorsFileHeaderFs<Fd>) {
  return createSafetensorsFileHeaderHelpers<Fd>({
    openFile: (modelPath) => fs.openSync(modelPath, "r"),
    closeFile: (fd) => fs.closeSync(fd),
    readFile: (fd, buffer, offset, length, position) => fs.readSync(fd, buffer, offset, length, position),
    allocBytes: (byteLength) => new Uint8Array(byteLength),
    readU64LE: (bytes) => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getBigUint64(0, true),
  });
}
