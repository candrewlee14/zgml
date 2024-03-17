const std = @import("std");

pub const Options = struct {
    use_blas: bool = false,
};

pub fn module(b: *std.Build) *std.Build.Module {
    return b.addModule("zgml", .{
        .root_source_file = .{ .path = (comptime thisDir()) ++ "/src/zgml.zig" },
        .dependencies = &.{
            .{ .name = "zgml_options", .module = b.getModule("zgml_options") },
        },
    });
}

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
                    // exe.addSystemLibrary("openblas");
                    exe.linkFramework("Accelerate");
                },
                else => {
                    @panic("Unsupported host OS");
                },
            }
        }
    }
};

pub fn package(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.Mode,
    args: struct {
        options: Options = .{},
    },
) Package {
    _ = optimize;
    const step = b.addOptions();
    step.addOption(bool, "use_blas", args.options.use_blas);

    const zgml_options = step.createModule();

    const zgml = b.addModule("zgml", .{
        .root_source_file = .{ .path = thisDir() ++ "/src/main.zig" },
        .imports = &.{
            .{ .name = "zgml_options", .module = zgml_options },
        },
    });

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

    // const benchmark_step = b.step("benchmark", "Run zgml benchmarks");
    // benchmark_step.dependOn(runBenchmarks(b, target, .{ .use_blas = use_blas }));
}

pub fn runTests(
    b: *std.Build,
    optimize: std.builtin.Mode,
    target: std.Build.ResolvedTarget,
    options: Options,
) *std.Build.Step {
    const test_exe = b.addTest(.{
        .name = "zgml-tests",
        .root_source_file = .{ .path = thisDir() ++ "/src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const zgml_pkg = package(b, target, optimize, .{ .options = options });
    zgml_pkg.link(test_exe);
    test_exe.root_module.addImport("zgml_options", zgml_pkg.zgml_options);
    b.installArtifact(test_exe);

    return &b.addRunArtifact(test_exe).step;
}

pub fn runBenchmarks(
    b: *std.Build,
    target: std.zig.CrossTarget,
    options: Options,
) *std.Build.Step {
    const exe = b.addExecutable(.{
        .name = "zgml-benchmarks",
        .root_source_file = .{ .path = thisDir() ++ "/src/benchmark.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    const zgml_pkg = package(b, target, .ReleaseFast, .{ .options = options });
    zgml_pkg.link(exe);
    exe.addModule("zgml", zgml_pkg.zgml);
    b.installArtifact(exe);
    return &exe.run().step;
}

inline fn thisDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file) orelse ".";
}
