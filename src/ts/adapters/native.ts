import { frontendManifest } from "../frontend_manifest.js";

export type {
  ConcreteRuntimeLoaderEvidence,
  NativeRuntime,
} from "./concrete_runtime_loader.js";

export type HostRuntimeKind = "node" | "bun" | "web";

export type NativeAdapterCallbackKind =
  | "loadLibrary"
  | "bindAbi"
  | "readFile"
  | "pathJoin"
  | "ffiCall";

export type NativeAdapterCapability = Readonly<{
  kind: NativeAdapterCallbackKind;
  required: boolean;
}>;

export type NativeAdapterEvidence = Readonly<{
  kind: "native-adapter";
  host: HostRuntimeKind;
  frontendSource: typeof frontendManifest.source;
  productSemanticsOwner: typeof frontendManifest.productSemanticsOwner;
  nativeAlignment: typeof frontendManifest.nativeAlignment;
  ownsFrontendPolicy: typeof frontendManifest.handwrittenFrontendMirrors;
  ownsNativeLoading: true;
  capabilities: readonly NativeAdapterCapability[];
  signature: string;
}>;

export type NativeAdapterManifestPolicy<PolicyOwner extends string> = Readonly<{
  source: typeof frontendManifest.source;
  policyOwner: PolicyOwner;
}>;

export type NativeLibraryPathOptions = {
  envLibraryPath?: string;
  moduleDir: string;
  cwd: string;
  extension: string;
  joinPath: (...parts: string[]) => string;
  exists: (path: string) => boolean;
};

export type NativeLibraryLoadInfoOptions = Omit<NativeLibraryPathOptions, "extension"> & {
  platform: string;
};

export type NativeLibraryLoadInfo = Readonly<{
  packageRoot: string;
  suffix: string;
  libPath: string;
}>;

export type NodeKoffiFallbackPathOptions = {
  moduleDir: string;
  joinPath: (...parts: string[]) => string;
};

const adapterCapabilities: readonly NativeAdapterCapability[] = Object.freeze([
  Object.freeze({ kind: "loadLibrary", required: true }),
  Object.freeze({ kind: "bindAbi", required: true }),
  Object.freeze({ kind: "readFile", required: false }),
  Object.freeze({ kind: "pathJoin", required: false }),
  Object.freeze({ kind: "ffiCall", required: true }),
]);

export function nativeAdapterCapabilities(): readonly NativeAdapterCapability[] {
  return adapterCapabilities;
}

export function nativeAdapterEvidence(host: HostRuntimeKind): NativeAdapterEvidence {
  const capabilities = nativeAdapterCapabilities();
  return Object.freeze({
    kind: "native-adapter",
    host,
    frontendSource: frontendManifest.source,
    productSemanticsOwner: frontendManifest.productSemanticsOwner,
    nativeAlignment: frontendManifest.nativeAlignment,
    ownsFrontendPolicy: frontendManifest.handwrittenFrontendMirrors,
    ownsNativeLoading: true,
    capabilities,
    signature: `${host}:${frontendManifest.source}-frontend:native-loader:${capabilities.map((capability) => capability.kind).join("+")}`,
  });
}

export function nativeAdapterManifestPolicy<const PolicyOwner extends string>(
  policyOwner: PolicyOwner,
): NativeAdapterManifestPolicy<PolicyOwner> {
  return Object.freeze({
    source: frontendManifest.source,
    policyOwner,
  });
}

export function adapterOwnsFrontendPolicy(evidence: NativeAdapterEvidence): false {
  return evidence.ownsFrontendPolicy;
}

export function nativeLibraryExtensionForPlatform(platform: string): string {
  if (platform === "darwin") return "dylib";
  if (platform === "win32") return "dll";
  return "so";
}

export function nativeLibraryFilename(extension: string): string {
  return `libzgml_c.${extension}`;
}

export function resolveNativeLibraryPath(options: NativeLibraryPathOptions): string {
  const envLibraryPath = options.envLibraryPath;
  if (typeof envLibraryPath === "string" && envLibraryPath.length > 0) return envLibraryPath;
  const filename = nativeLibraryFilename(options.extension);
  const packageLibPath = options.joinPath(options.moduleDir, "zig-out", "lib", filename);
  const legacyModuleLibPath = options.joinPath(options.moduleDir, "..", "zig-out", "lib", filename);
  const cwdLibPath = options.joinPath(options.cwd, "zig-out", "lib", filename);
  if (options.exists(packageLibPath)) return packageLibPath;
  if (options.exists(legacyModuleLibPath)) return legacyModuleLibPath;
  if (options.exists(cwdLibPath)) return cwdLibPath;
  return packageLibPath;
}

export function resolveNativeLibraryLoadInfo(options: NativeLibraryLoadInfoOptions): NativeLibraryLoadInfo {
  const suffix = nativeLibraryExtensionForPlatform(options.platform);
  const libPath = resolveNativeLibraryPath({
    envLibraryPath: options.envLibraryPath,
    moduleDir: options.moduleDir,
    cwd: options.cwd,
    extension: suffix,
    joinPath: options.joinPath,
    exists: options.exists,
  });

  if (!options.exists(libPath)) {
    throw new Error(nativeLibraryMissingMessage(libPath, suffix));
  }

  return Object.freeze({
    packageRoot: options.moduleDir,
    suffix,
    libPath,
  });
}

export function nativeLibraryMissingMessage(path: string, extension: string): string {
  return `zgml native library not found at ${path}. Run "zig build ffi-c" from the zgml package root or set ZGML_C_DYLIB to a built ${nativeLibraryFilename(extension)}.`;
}

export function nodeKoffiFallbackPath(options: NodeKoffiFallbackPathOptions): string {
  return options.joinPath(options.moduleDir, "..", "examples", "node_ffi", "node_modules", "koffi");
}
