const std = @import("std");

pub fn build(b: *std.Build) void {
    // Package-only — nothing to build standalone.
    _ = b;
}

/// Resolve the pre-built wgpu-native archive for the given target and
/// configure `compile` with include paths, library paths, and link flags.
///
/// Returns `false` if the target platform is unsupported or the lazy
/// dependency hasn't been fetched yet.
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
        .linux => {
            // wgpu-native loads Vulkan at runtime via dlopen, so no direct
            // libvulkan link needed.  The static archive does need:
            //   - libdl (dlopen/dlsym for Vulkan loader)  — part of glibc
            //   - libpthread (threading)                    — part of glibc
            //   - libm (math)                               — part of glibc
            // link_libc = true above covers all three on modern glibc.
            //
            // Rust unwinding symbols (_Unwind_*) come from libgcc_s or
            // libunwind; the Zig toolchain provides its own unwinder so
            // these resolve automatically for native builds.  For
            // cross-compiled targets using a system linker, libgcc_s may
            // be needed — but that's already on the default search path.
        },
        .windows => {
            // wgpu-native's static lib (MinGW/GNU) imports from many
            // Windows system DLLs.  Zig's MinGW layer provides import
            // libs for all of these, but they must be listed explicitly.
            const win_libs = [_][]const u8{
                "d3d12",     // Direct3D 12
                "dxgi",      // DXGI (adapter enumeration)
                "dcomp",     // DirectComposition
                "advapi32",  // Registry, crypto helpers
                "cfgmgr32",  // Device configuration manager
                "gdi32",     // GDI (fallback surface)
                "kernel32",  // Core Win32
                "ntdll",     // NT internals (RtlAddFunctionTable, etc.)
                "opengl32",  // OpenGL (GL backend fallback)
                "setupapi",  // Device enumeration
                "user32",    // Window management
                "ole32",     // COM (CoInitializeEx, etc.)
                "oleaut32",  // OLE Automation
                "combase",   // COM base (RoInitialize)
                "dbghelp",   // Stack traces
                "rpcrt4",    // RPC runtime (UuidCreate, etc.)
                "ws2_32",    // Winsock (networking for shader cache)
                "bcrypt",    // Cryptographic RNG
            };
            for (win_libs) |name| {
                compile.root_module.linkSystemLibrary(name, .{});
            }
        },
        else => {},
    }
    return true;
}

/// Return an include LazyPath for adding to Zig modules (so @cImport works).
pub fn includePath(
    dep: *std.Build.Dependency,
    target: std.Build.ResolvedTarget,
) ?std.Build.LazyPath {
    const upstream = resolveUpstream(dep, target) orelse return null;
    return upstream.path("include");
}

// ── internal ──────────────────────────────────────────────────────

fn resolveUpstream(
    dep: *std.Build.Dependency,
    target: std.Build.ResolvedTarget,
) ?*std.Build.Dependency {
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
