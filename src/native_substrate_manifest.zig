// Generated from src/ts/frontend_manifest.ts by scripts/generate_native_substrate_manifest.cjs.
// Do not edit by hand; TS owns product policy, Zig imports this native substrate view.

pub const native_substrate_manifest = .{
    .kind = "zgml-native-substrate",
    .role = "runtime-kernel-abi-substrate",
    .product_language = "typescript",
    .product_source = "src/ts/**",
    .product_source_of_truth = "ts-only",
    .product_semantics_owner = "src/ts/**",
    .package_fanout = "tsdown",
    .frontend_sync = "none",
    .handwritten_frontend_mirrors = false,
    .native_alignment = "contract-tested-substrate",
    .native_product_policy = "forbidden",
    .native_contract_boundary = "Program/Session/ABI contracts",
};
