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
    .module_compiler_core = "zig-module-program",
    .native_compile_evidence = "native-program-inspection",
    .program_inspection_core = "zig-program-inspection",
    .eager_hot_path_core = "zig-native-eager-when-profitable",
    .inference_hot_path_core = "zig-program-session-required",
    .training_hot_path_core = "zig-ffi-compiled-step-when-supported",
    .unsupported_hot_path_policy = "explicit-evidence-no-silent-performance-claim",
};
