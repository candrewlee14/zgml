pub const Tensor = @import("tensor.zig").Tensor;

const opts = @import("zgml_options");

pub const backend = @import("backend.zig");
pub const backend_cpu = @import("backend/cpu.zig");
pub const backend_metal = if (opts.use_metal and @import("builtin").os.tag == .macos) @import("backend/metal.zig") else struct {
    pub const MetalBackend = struct {};
};
pub const backend_program = @import("backend/program.zig");
pub const backend_stencil = @import("backend/stencil.zig");
pub const llama_inference = @import("llama_inference.zig");
pub const profile = @import("profile.zig");
pub const tensor_program_ir = @import("tensor_program_ir.zig");

test "internal Interface stays curated" {
    const testing = @import("std").testing;
    const root = @This();
    const expected = .{
        "Tensor",
        "backend",
        "backend_cpu",
        "backend_metal",
        "backend_program",
        "backend_stencil",
        "llama_inference",
        "profile",
        "tensor_program_ir",
    };
    try testing.expectEqual(expected.len, @typeInfo(root).@"struct".decls.len);
    inline for (expected) |name| {
        try testing.expect(@hasDecl(root, name));
    }
}

test "ref all internal decls" {
    _ = @import("device_inference.zig");
    _ = @import("models/gguf_loader.zig");
    _ = @import("std").testing.refAllDecls(@This());
}
