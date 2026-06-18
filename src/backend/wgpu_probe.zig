const std = @import("std");
const options = @import("zgml_options");

const c = if (options.use_wgpu) @cImport({
    @cInclude("webgpu/webgpu.h");
    @cInclude("webgpu/wgpu.h");
}) else struct {};

pub fn headersAvailable() bool {
    if (!options.use_wgpu) return false;
    return @hasDecl(c, "WGPUInstance") and
        @hasDecl(c, "WGPUInstanceDescriptor") and
        @hasDecl(c, "WGPUInstanceBackend_Metal") and
        @hasDecl(c, "wgpuCreateInstance");
}

pub fn createInstanceSymbolAddress() usize {
    if (!options.use_wgpu) return 0;
    return @intFromPtr(&c.wgpuCreateInstance);
}

test "wgpu-native headers and link symbol are available when enabled" {
    if (!options.use_wgpu) return error.SkipZigTest;
    try std.testing.expect(headersAvailable());
    try std.testing.expect(createInstanceSymbolAddress() != 0);
}
