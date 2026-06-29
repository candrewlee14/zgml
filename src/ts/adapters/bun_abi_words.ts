export type BunNativeWordHandle = number;

export function setUSize(view: DataView, offset: number, value: number): void {
  view.setBigUint64(offset, BigInt(value), true);
}

export function setPtr(view: DataView, offset: number, value: BunNativeWordHandle): void {
  view.setBigUint64(offset, BigInt(value), true);
}

export function setU32(view: DataView, offset: number, value: number): void {
  view.setUint32(offset, value, true);
}
