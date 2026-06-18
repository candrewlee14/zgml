const std = @import("std");

pub fn build(b: *std.Build) void {
    // Package-only: the root build imports link/include helpers below.
    _ = b;
}

/// Resolve the pre-built wgpu-native archive for the given target and configure
/// `compile` with include paths, library paths, and platform link flags.
pub fn link(
    dep: *std.Build.Dependency,
    target: std.Build.ResolvedTarget,
    compile: *std.Build.Step.Compile,
) bool {
    const upstream = resolveUpstream(dep, target) orelse return false;
    const inc = upstream.path("include");
    const lib = upstream.path("lib");

    compile.root_module.addIncludePath(inc);
    compile.root_module.addLibraryPath(lib);
    compile.root_module.linkSystemLibrary("wgpu_native", .{});
    compile.root_module.link_libc = true;

    switch (target.result.os.tag) {
        .macos => {
            compile.root_module.linkFramework("Metal", .{});
            compile.root_module.linkFramework("QuartzCore", .{});
            compile.root_module.linkFramework("CoreFoundation", .{});
            compile.root_module.linkFramework("Foundation", .{});
        },
        .linux => {},
        .windows => {
            const win_libs = [_][]const u8{
                "d3d12",
                "dxgi",
                "dcomp",
                "advapi32",
                "cfgmgr32",
                "gdi32",
                "kernel32",
                "ntdll",
                "opengl32",
                "setupapi",
                "user32",
                "ole32",
                "oleaut32",
                "combase",
                "dbghelp",
                "rpcrt4",
                "ws2_32",
                "bcrypt",
            };
            for (win_libs) |name| compile.root_module.linkSystemLibrary(name, .{});
        },
        else => {},
    }
    return true;
}

pub fn includePath(dep: *std.Build.Dependency, target: std.Build.ResolvedTarget) ?std.Build.LazyPath {
    const upstream = resolveUpstream(dep, target) orelse return null;
    return upstream.path("include");
}

fn resolveUpstream(dep: *std.Build.Dependency, target: std.Build.ResolvedTarget) ?*std.Build.Dependency {
    const dep_name: []const u8 = switch (target.result.os.tag) {
        .macos => switch (target.result.cpu.arch) {
            .aarch64 => "wgpu-macos-aarch64",
            .x86_64 => "wgpu-macos-x86_64",
            else => return null,
        },
        .linux => switch (target.result.cpu.arch) {
            .x86_64 => "wgpu-linux-x86_64",
            .aarch64 => "wgpu-linux-aarch64",
            else => return null,
        },
        .windows => switch (target.result.cpu.arch) {
            .x86_64 => "wgpu-windows-x86_64",
            else => return null,
        },
        else => return null,
    };
    return dep.builder.lazyDependency(dep_name, .{});
}
