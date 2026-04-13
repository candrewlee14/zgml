const std = @import("std");

pub const Options = struct {
    use_blas: bool = false,
};

pub const Package = struct {
    target: std.Build.ResolvedTarget,
    options: Options,
    zgml: *std.Build.Module,
    zgml_options: *std.Build.Module,

    pub fn link(pkg: Package, exe: *std.Build.Step.Compile) void {
        exe.root_module.addImport("zgml", pkg.zgml);
        exe.root_module.addImport("zgml_options", pkg.zgml_options);

        if (pkg.options.use_blas) {
            exe.linkLibC();
            switch (pkg.target.result.os.tag) {
                .windows => {
                    exe.linkSystemLibrary("libopenblas");
                },
                .linux => {
                    exe.linkSystemLibrary("openblas");
                },
                .macos => {
                    exe.linkFramework("Accelerate");
                },
                else => {
                    @panic("Unsupported host OS");
                },
            }
        }
    }
};

fn linkBlas(target: std.Build.ResolvedTarget, exe: *std.Build.Step.Compile) void {
    exe.linkLibC();
    switch (target.result.os.tag) {
        .windows => exe.linkSystemLibrary("libopenblas"),
        .linux => exe.linkSystemLibrary("openblas"),
        .macos => exe.linkFramework("Accelerate"),
        else => @panic("Unsupported host OS"),
    }
}

pub fn package(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    args: struct {
        options: Options = .{},
    },
) Package {
    _ = optimize;
    const step = b.addOptions();
    step.addOption(bool, "use_blas", args.options.use_blas);

    const zgml_options = step.createModule();

    const zgml = b.addModule("zgml", .{
        .root_source_file = b.path("src/main.zig"),
        .imports = &.{
            .{ .name = "zgml_options", .module = zgml_options },
        },
        .link_libc = if (args.options.use_blas) true else null,
    });

    if (args.options.use_blas) {
        zgml.addIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }

    return .{
        .target = target,
        .options = args.options,
        .zgml = zgml,
        .zgml_options = zgml_options,
    };
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const use_blas = b.option(bool, "use-blas", "Use BLAS library") orelse false;

    _ = package(b, target, optimize, .{ .options = .{ .use_blas = use_blas } });

    const test_step = b.step("test", "Run zgml tests");
    test_step.dependOn(runTests(b, optimize, target, .{ .use_blas = use_blas }));

    // Benchmark — always built with ReleaseFast
    const zgml_bench_pkg = package(b, target, .ReleaseFast, .{ .options = .{ .use_blas = use_blas } });
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "zgml", .module = zgml_bench_pkg.zgml },
            .{ .name = "zgml_options", .module = zgml_bench_pkg.zgml_options },
        },
    });
    const bench_exe = b.addExecutable(.{
        .name = "zgml-bench",
        .root_module = bench_mod,
    });
    if (use_blas) {
        linkBlas(target, bench_exe);
        bench_exe.addIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    b.installArtifact(bench_exe);

    const bench_build_step = b.step("bench-build", "Build zgml benchmarks");
    bench_build_step.dependOn(&bench_exe.step);

    const bench_step = b.step("bench", "Build and run zgml benchmarks");
    bench_step.dependOn(&b.addRunArtifact(bench_exe).step);

    // MNIST benchmark — always built with ReleaseFast
    const zgml_mnist_pkg = package(b, target, .ReleaseFast, .{ .options = .{ .use_blas = use_blas } });
    const mnist_mod = b.createModule(.{
        .root_source_file = b.path("src/mnist_bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "zgml", .module = zgml_mnist_pkg.zgml },
            .{ .name = "zgml_options", .module = zgml_mnist_pkg.zgml_options },
        },
    });
    const mnist_exe = b.addExecutable(.{
        .name = "mnist-bench",
        .root_module = mnist_mod,
    });
    if (use_blas) {
        linkBlas(target, mnist_exe);
        mnist_exe.addIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    b.installArtifact(mnist_exe);

    const mnist_step = b.step("mnist-bench", "Build and run MNIST CNN training benchmark");
    mnist_step.dependOn(&b.addRunArtifact(mnist_exe).step);
}

pub fn runTests(
    b: *std.Build,
    optimize: std.builtin.OptimizeMode,
    target: std.Build.ResolvedTarget,
    options: Options,
) *std.Build.Step {
    const zgml_pkg = package(b, target, optimize, .{ .options = options });

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgml_options", .module = zgml_pkg.zgml_options },
        },
    });

    const test_exe = b.addTest(.{
        .name = "zgml-tests",
        .root_module = test_mod,
    });
    if (options.use_blas) {
        linkBlas(target, test_exe);
        test_exe.addIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    b.installArtifact(test_exe);

    return &b.addRunArtifact(test_exe).step;
}
