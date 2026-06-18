const testing = @import("std").testing;
pub const Tensor = @import("tensor.zig").Tensor;
pub const ComputeGraph = @import("graph.zig").ComputeGraph;
const native_manifest = @import("native_substrate_manifest.zig");

pub const nn = @import("nn.zig");
pub const loss = @import("loss.zig");
pub const optim = @import("optim.zig");
pub const train = @import("train.zig");
pub const llm = @import("llm.zig");

pub const native_substrate_manifest = native_manifest.native_substrate_manifest;

test "native root surface stays curated" {
    const root = @This();
    const expected = .{
        "Tensor",
        "ComputeGraph",
        "nn",
        "loss",
        "optim",
        "train",
        "llm",
        "native_substrate_manifest",
    };
    try testing.expectEqual(expected.len, @typeInfo(root).@"struct".decls.len);
    inline for (expected) |name| {
        try testing.expect(@hasDecl(root, name));
    }

    const Graph = ComputeGraph(f32);
    const graph_expected = .{
        "init",
        "allocator",
        "tensor",
        "fromSlice",
        "full",
        "zeros",
        "ones",
        "param",
        "scalar",
        "arange",
        "linspace",
        "rand",
        "randn",
        "deinit",
        "enableThreading",
        "run",
        "infer",
    };
    try testing.expectEqual(graph_expected.len, @typeInfo(Graph).@"struct".decls.len);
    inline for (graph_expected) |name| {
        try testing.expect(@hasDecl(Graph, name));
    }

    try testing.expectEqualStrings("zgml-native-substrate", native_substrate_manifest.kind);
    try testing.expectEqualStrings("runtime-kernel-abi-substrate", native_substrate_manifest.role);
    try testing.expectEqualStrings("typescript", native_substrate_manifest.product_language);
    try testing.expectEqualStrings("src/ts/**", native_substrate_manifest.product_source);
    try testing.expectEqualStrings("ts-only", native_substrate_manifest.product_source_of_truth);
    try testing.expectEqualStrings("src/ts/**", native_substrate_manifest.product_semantics_owner);
    try testing.expectEqualStrings("tsdown", native_substrate_manifest.package_fanout);
    try testing.expectEqualStrings("none", native_substrate_manifest.frontend_sync);
    try testing.expect(!native_substrate_manifest.handwritten_frontend_mirrors);
    try testing.expectEqualStrings("contract-tested-substrate", native_substrate_manifest.native_alignment);
    try testing.expectEqualStrings("forbidden", native_substrate_manifest.native_product_policy);
    try testing.expectEqualStrings("Program/Session/ABI contracts", native_substrate_manifest.native_contract_boundary);
}

test "ref all decls" {
    _ = @import("nn.zig");
    _ = @import("loss.zig");
    _ = @import("optim.zig");
    _ = @import("train.zig");
    _ = @import("llm.zig");
}
