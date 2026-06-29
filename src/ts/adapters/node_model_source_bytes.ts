type ByteSource = Readonly<{
  byteLength: number;
}>;

type PathBytes = ByteSource & Readonly<{
  length: number;
}>;

type NodeBufferConstructor = Readonly<{
  from(value: string): PathBytes;
  from(buffer: ArrayBufferLike, byteOffset: number, length: number): ByteSource;
}>;

declare const Buffer: NodeBufferConstructor;

export type NodeModelSourceBytes = Readonly<{
  bytesFromString(value: string): PathBytes;
  bytesFromView(bytes: Uint8Array): ByteSource;
}>;

export function createNodeModelSourceBytes(): NodeModelSourceBytes {
  return Object.freeze({
    bytesFromString: (value) => Buffer.from(value),
    bytesFromView: (bytes) => Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength),
  });
}
