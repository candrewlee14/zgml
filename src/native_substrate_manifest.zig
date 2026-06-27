// Generated from src/ts/frontend_manifest.ts by scripts/generate_native_substrate_manifest.cjs.
// Do not edit by hand; TS owns product policy, Zig imports this native substrate view.

pub const native_substrate_manifest = .{
    .kind = "zgml-native-substrate",
    .role = "core-kernel-runtime",
    .product_language = "typescript",
    .product_source = "src/ts/**",
    .product_source_of_truth = "ts-api-zig-core",
    .product_semantics_owner = "src/ts/** + src/**/*.zig",
    .package_fanout = "tsdown",
    .frontend_sync = "none",
    .handwritten_frontend_mirrors = false,
    .native_alignment = "zig-core-contract-tested",
    .native_product_policy = "required-core",
    .native_contract_boundary = "JS/TS API -> Zig C ABI -> Program/Session kernels",
};
