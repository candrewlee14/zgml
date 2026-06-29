const std = @import("std");
const wgpu = @import("backend/wgpu.zig");

test {
    std.testing.refAllDecls(wgpu);
}
