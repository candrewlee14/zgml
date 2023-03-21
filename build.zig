const std = @import("std");

pub const Options = struct {
    use_blas: bool = false,
};

pub fn module(b: *std.Build) *std.Build.Module {
    return b.createModule(.{
        .name = "zgml",
        .source_file = .{ .path = (comptime thisDir()) ++ "/src/zgml.zig" },
        .dependencies = &.{
            .{ .name = "zgml_options", .module = b.getModule("zgml_options") },
        },
    });
}

pub const Package = struct {
    options: Options,
    zgml: *std.build.Module,
    zgml_options: *std.Build.Module,

    pub fn link(pkg: Package, exe: *std.Build.CompileStep) void {
        if (pkg.options.use_blas) {
            const host = (std.zig.system.NativeTargetInfo.detect(exe.target) catch unreachable).target;
            exe.linkLibC();
            switch (host.os.tag) {
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
    target: std.zig.CrossTarget,
    optimize: std.builtin.Mode,
    args: struct {
        options: Options = .{},
    },
) Package {
    _ = target;
    _ = optimize;
    const step = b.addOptions();
    step.addOption(bool, "use_blas", args.options.use_blas);
    const zgml_options = step.createModule();

    const zgml = b.createModule(.{
        .source_file = .{ .path = thisDir() ++ "/src/main.zig" },
        .dependencies = &.{
            .{ .name = "zgml_options", .module = zgml_options },
        },
    });
    return .{
        .options = args.options,
        .zgml = zgml,
        .zgml_options = zgml_options,
    };
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const use_blas = b.option(bool, "use-blas", "Use BLAS library") orelse false;

    const test_step = b.step("test", "Run zgml tests");
    test_step.dependOn(runTests(b, optimize, target, .{ .use_blas = use_blas }));

    const benchmark_step = b.step("benchmark", "Run zgml benchmarks");
    benchmark_step.dependOn(runBenchmarks(b, target, .{ .use_blas = use_blas }));
}

pub fn runTests(
    b: *std.Build,
    optimize: std.builtin.Mode,
    target: std.zig.CrossTarget,
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
    // std.debug.print("zgml_options: {any}\n", .{zgml_pkg.zgml_options});
    test_exe.addModule("zgml_options", zgml_pkg.zgml_options);

    return &test_exe.run().step;
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
    return &exe.run().step;
}

inline fn thisDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file) orelse ".";
}
