declare module "bun:ffi" {
  export const FFIType: Readonly<Record<"cstring" | "i32" | "ptr" | "u32" | "u64", string>>;
  export function dlopen(path: string, symbols: Readonly<Record<string, Readonly<{
    args?: readonly string[];
    returns?: string;
  }>>>): Readonly<{ symbols: Record<string, unknown> }>;
  export function ptr(value: ArrayBuffer | ArrayBufferView): number;
  export const read: Readonly<{
    ptr(value: number, offset: number): number;
  }>;
}
