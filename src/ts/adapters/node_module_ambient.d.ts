declare module "node:module" {
  export function createRequire(filename: string): (id: string) => any;
}

declare module "node:fs" {
  export function closeSync(fd: number): void;
  export function existsSync(path: string): boolean;
  export function openSync(path: string, flags: string): number;
  export function readSync(fd: number, buffer: Uint8Array, offset: number, length: number, position: number): number;
}

declare module "node:path" {
  export function dirname(path: string): string;
  export function join(...paths: string[]): string;
}

declare module "node:url" {
  export function fileURLToPath(url: string): string;
  export function pathToFileURL(path: string): { href: string };
}

declare class TextEncoder {
  encode(input?: string): Uint8Array;
}

interface SymbolConstructor {
  readonly dispose: symbol;
}
