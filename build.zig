const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const use_blas = b.option(bool, "use-blas", "use BLAS for matmuls") orelse false;
    const opts_mod = b.addOptions();
    opts_mod.addOption(bool, "use-blas", use_blas);


    const lib = b.addStaticLibrary(.{
        .name = "zgml",
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Creates a step for unit testing.
    const main_tests = b.addTest(.{
        .name = "zgml",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });


    lib.addOptions("options", opts_mod);
    main_tests.addOptions("options", opts_mod);

    if (use_blas) {
        linkBlas(lib);
        linkBlas(main_tests);
    }


    // This declares intent for the library to be installed into the standard
    // location when the user invokes the "install" step (the default step when
    // running `zig build`).
    lib.install();

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build test`
    // This will evaluate the `test` step rather than the default, which is "install".
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&main_tests.run().step);
}

fn linkBlas(step: *std.Build.CompileStep) void {
    step.linkLibC();
    const host = (std.zig.system.NativeTargetInfo.detect(step.target) catch unreachable).target;
    switch (host.os.tag) {
        .macos => {
            step.linkFramework("Accelerate");
        },
        .linux => {
            step.linkSystemLibrary("cblas");
        },
        else => @panic("Unsupported OS"),
    }
}