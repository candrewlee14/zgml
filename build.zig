const std = @import("std");

pub const Options = struct {
    use_blas: bool = false,
};

pub const Package = struct {
    target: std.Build.ResolvedTarget,
    options: Options,
    zgml: *std.Build.Module,
    zgml_options: *std.Build.Module,

    pub fn link(pkg: Package, b: *std.Build, exe: *std.Build.Step.Compile) void {
        exe.root_module.addImport("zgml", pkg.zgml);
        exe.root_module.addImport("zgml_options", pkg.zgml_options);
        linkMetal(b, pkg.target, exe);

        if (pkg.options.use_blas) {
            exe.root_module.link_libc = true;
            switch (pkg.target.result.os.tag) {
                .windows => {
                    exe.root_module.linkSystemLibrary("libopenblas", .{});
                },
                .linux => {
                    exe.root_module.linkSystemLibrary("openblas", .{});
                },
                .macos => {
                    exe.root_module.linkFramework("Accelerate", .{});
                },
                .freestanding => {
                    // WASM/freestanding targets cannot use BLAS — ignore silently.
                },
                else => {
                    @panic("Unsupported host OS for BLAS linking");
                },
            }
        }
    }
};

fn linkMetal(b: *std.Build, target: std.Build.ResolvedTarget, exe: *std.Build.Step.Compile) void {
    if (target.result.os.tag == .macos) {
        exe.root_module.addCSourceFile(.{
            .file = b.path("src/backend/metal_shim.m"),
            .flags = &.{"-fno-objc-arc"},
        });
        exe.root_module.addIncludePath(b.path("src/backend"));
        exe.root_module.linkFramework("Metal", .{});
        exe.root_module.linkFramework("Foundation", .{});
        exe.root_module.link_libc = true;
    }
}

fn linkBlas(target: std.Build.ResolvedTarget, exe: *std.Build.Step.Compile) void {
    exe.root_module.link_libc = true;
    switch (target.result.os.tag) {
        .windows => exe.root_module.linkSystemLibrary("libopenblas", .{}),
        .linux => exe.root_module.linkSystemLibrary("openblas", .{}),
        .macos => exe.root_module.linkFramework("Accelerate", .{}),
        .freestanding => {}, // WASM/freestanding — no BLAS available
        else => @panic("Unsupported host OS for BLAS linking"),
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

    // Metal backend: include path for the C shim header so metal.zig can @cImport it.
    // Actual linking happens in Package.link() and linkMetal() on the compile step.
    if (target.result.os.tag == .macos) {
        zgml.addIncludePath(b.path("src/backend"));
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

    const use_blas = b.option(bool, "use-blas", "Use BLAS library") orelse (target.result.os.tag == .macos);

    _ = package(b, target, optimize, .{ .options = .{ .use_blas = use_blas } });

    const test_step = b.step("test", "Run zgml tests");
    test_step.dependOn(runTests(b, optimize, target, .{ .use_blas = use_blas }));
    test_step.dependOn(runInferenceTests(b, optimize, target, .{ .use_blas = use_blas }));

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
        bench_exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
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
        mnist_exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    b.installArtifact(mnist_exe);

    const mnist_step = b.step("mnist-bench", "Build and run MNIST CNN training benchmark");
    mnist_step.dependOn(&b.addRunArtifact(mnist_exe).step);

    // MNIST micro-benchmark — isolates the training hotloop
    const zgml_micro_pkg = package(b, target, .ReleaseFast, .{ .options = .{ .use_blas = use_blas } });
    const micro_mod = b.createModule(.{
        .root_source_file = b.path("src/mnist_micro.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "zgml", .module = zgml_micro_pkg.zgml },
            .{ .name = "zgml_options", .module = zgml_micro_pkg.zgml_options },
        },
    });
    const micro_exe = b.addExecutable(.{
        .name = "mnist-micro",
        .root_module = micro_mod,
    });
    if (use_blas) {
        linkBlas(target, micro_exe);
        micro_exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    b.installArtifact(micro_exe);

    const micro_step = b.step("mnist-micro", "Build and run MNIST CNN training micro-benchmark");
    micro_step.dependOn(&b.addRunArtifact(micro_exe).step);

    const zgml_conv_phase_pkg = package(b, target, .ReleaseFast, .{ .options = .{ .use_blas = use_blas } });
    const conv_phase_mod = b.createModule(.{
        .root_source_file = b.path("src/conv_phase_bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "zgml", .module = zgml_conv_phase_pkg.zgml },
            .{ .name = "zgml_options", .module = zgml_conv_phase_pkg.zgml_options },
        },
    });
    const conv_phase_exe = b.addExecutable(.{
        .name = "conv-phase-bench",
        .root_module = conv_phase_mod,
    });
    if (use_blas) {
        linkBlas(target, conv_phase_exe);
        conv_phase_exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    b.installArtifact(conv_phase_exe);
    const conv_phase_step = b.step("conv-phase-bench", "Run parameterized conv phase benchmark");
    conv_phase_step.dependOn(&b.addRunArtifact(conv_phase_exe).step);

    // Gradient check — compare zgml vs PyTorch reference
    const zgml_gc_pkg = package(b, target, optimize, .{ .options = .{ .use_blas = use_blas } });
    const gc_mod = b.createModule(.{
        .root_source_file = b.path("benchmarks/grad_check.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgml", .module = zgml_gc_pkg.zgml },
            .{ .name = "zgml_options", .module = zgml_gc_pkg.zgml_options },
        },
    });
    const gc_exe = b.addExecutable(.{
        .name = "grad-check",
        .root_module = gc_mod,
    });
    if (use_blas) {
        linkBlas(target, gc_exe);
        gc_exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    b.installArtifact(gc_exe);
    const gc_step = b.step("grad-check", "Compare gradients with PyTorch reference");
    gc_step.dependOn(&b.addRunArtifact(gc_exe).step);

    // Per-op backward profiler
    const zgml_prof_pkg = package(b, target, .ReleaseFast, .{ .options = .{ .use_blas = use_blas } });
    const prof_mod = b.createModule(.{
        .root_source_file = b.path("src/profile_backward.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "zgml", .module = zgml_prof_pkg.zgml },
            .{ .name = "zgml_options", .module = zgml_prof_pkg.zgml_options },
        },
    });
    const prof_exe = b.addExecutable(.{
        .name = "profile-bwd",
        .root_module = prof_mod,
    });
    if (use_blas) {
        linkBlas(target, prof_exe);
        prof_exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    b.installArtifact(prof_exe);
    const prof_step = b.step("profile-bwd", "Profile backward pass per-op timing");
    prof_step.dependOn(&b.addRunArtifact(prof_exe).step);

    // Reduction microbenchmark
    const zgml_rbench_pkg = package(b, target, .ReleaseFast, .{ .options = .{ .use_blas = use_blas } });
    const rbench_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_reduce.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "zgml", .module = zgml_rbench_pkg.zgml },
            .{ .name = "zgml_options", .module = zgml_rbench_pkg.zgml_options },
        },
    });
    const rbench_exe = b.addExecutable(.{
        .name = "bench-reduce",
        .root_module = rbench_mod,
    });
    b.installArtifact(rbench_exe);
    const rbench_step = b.step("bench-reduce", "Benchmark reduction and broadcast ops");
    rbench_step.dependOn(&b.addRunArtifact(rbench_exe).step);

    // Inference benchmark
    {
        const pkg = package(b, target, .ReleaseFast, .{ .options = .{ .use_blas = use_blas } });
        const mod = b.createModule(.{
            .root_source_file = b.path("bench/inference.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zgml", .module = pkg.zgml },
                .{ .name = "zgml_options", .module = pkg.zgml_options },
            },
        });
        const exe = b.addExecutable(.{ .name = "bench-inference", .root_module = mod });
        if (use_blas) {
            linkBlas(target, exe);
            exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
        }
        b.installArtifact(exe);
        const step = b.step("bench-inference", "Benchmark inference session (f32 vs int8)");
        step.dependOn(&b.addRunArtifact(exe).step);
    }

    // Per-op microbenchmark
    const zgml_opbench_pkg = package(b, target, .ReleaseFast, .{ .options = .{ .use_blas = use_blas } });
    const opbench_mod = b.createModule(.{
        .root_source_file = b.path("benchmarks/op_bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "zgml", .module = zgml_opbench_pkg.zgml },
            .{ .name = "zgml_options", .module = zgml_opbench_pkg.zgml_options },
        },
    });
    const opbench_exe = b.addExecutable(.{
        .name = "op-bench",
        .root_module = opbench_mod,
    });
    if (use_blas) {
        linkBlas(target, opbench_exe);
        opbench_exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    b.installArtifact(opbench_exe);
    const opbench_step = b.step("op-bench", "Per-op microbenchmark for MNIST CNN ops");
    opbench_step.dependOn(&b.addRunArtifact(opbench_exe).step);

    // Metal backend benchmark
    {
        const pkg = package(b, target, .ReleaseFast, .{ .options = .{ .use_blas = use_blas } });
        const mod = b.createModule(.{
            .root_source_file = b.path("benchmarks/metal_bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zgml", .module = pkg.zgml },
                .{ .name = "zgml_options", .module = pkg.zgml_options },
            },
        });
        const exe = b.addExecutable(.{ .name = "bench-metal", .root_module = mod });
        if (use_blas) {
            linkBlas(target, exe);
            exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
        }
        linkMetal(b, target, exe);
        b.installArtifact(exe);
        const step = b.step("bench-metal", "Benchmark Metal GPU vs CPU BLAS matmul");
        step.dependOn(&b.addRunArtifact(exe).step);
    }

    // Metal inference benchmark
    {
        const pkg = package(b, target, .ReleaseFast, .{ .options = .{ .use_blas = use_blas } });
        const mod = b.createModule(.{
            .root_source_file = b.path("benchmarks/metal_inference_bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zgml", .module = pkg.zgml },
                .{ .name = "zgml_options", .module = pkg.zgml_options },
            },
        });
        const exe = b.addExecutable(.{ .name = "bench-metal-inference", .root_module = mod });
        if (use_blas) {
            linkBlas(target, exe);
            exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
        }
        linkMetal(b, target, exe);
        b.installArtifact(exe);
        const step = b.step("bench-metal-inference", "Benchmark CPU vs Metal inference tok/s");
        step.dependOn(&b.addRunArtifact(exe).step);
    }

    // Text generation binary
    const zgml_gen_pkg = package(b, target, .ReleaseFast, .{ .options = .{ .use_blas = use_blas } });
    const gen_mod = b.createModule(.{
        .root_source_file = b.path("scripts/generate.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "zgml", .module = zgml_gen_pkg.zgml },
            .{ .name = "zgml_options", .module = zgml_gen_pkg.zgml_options },
        },
    });
    const gen_exe = b.addExecutable(.{
        .name = "generate",
        .root_module = gen_mod,
    });
    if (use_blas) {
        linkBlas(target, gen_exe);
        gen_exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    b.installArtifact(gen_exe);
    const gen_step = b.step("generate", "Build text generation binary");
    gen_step.dependOn(&b.addInstallArtifact(gen_exe, .{}).step);

    // Training script
    const zgml_train_pkg = package(b, target, .ReleaseFast, .{ .options = .{ .use_blas = use_blas } });
    const train_mod = b.createModule(.{
        .root_source_file = b.path("scripts/train_tiny.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "zgml", .module = zgml_train_pkg.zgml },
            .{ .name = "zgml_options", .module = zgml_train_pkg.zgml_options },
        },
    });
    const train_exe = b.addExecutable(.{
        .name = "train-tiny",
        .root_module = train_mod,
    });
    if (use_blas) {
        linkBlas(target, train_exe);
        train_exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    const train_step = b.step("train-tiny", "Train a tiny GPT and save checkpoint");
    train_step.dependOn(&b.addInstallArtifact(train_exe, .{}).step);

    // Pretrained model generation
    const zgml_pt_pkg = package(b, target, .ReleaseFast, .{ .options = .{ .use_blas = use_blas } });
    const pt_mod = b.createModule(.{
        .root_source_file = b.path("scripts/generate_pretrained.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "zgml", .module = zgml_pt_pkg.zgml },
            .{ .name = "zgml_options", .module = zgml_pt_pkg.zgml_options },
        },
    });
    const pt_exe = b.addExecutable(.{
        .name = "generate-pretrained",
        .root_module = pt_mod,
    });
    if (use_blas) {
        linkBlas(target, pt_exe);
        pt_exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    b.installArtifact(pt_exe);
    const pt_step = b.step("generate-pretrained", "Generate text from a pretrained HF model");
    pt_step.dependOn(&b.addInstallArtifact(pt_exe, .{}).step);
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
        test_exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    linkMetal(b, target, test_exe);
    b.installArtifact(test_exe);

    return &b.addRunArtifact(test_exe).step;
}

fn runInferenceTests(
    b: *std.Build,
    optimize: std.builtin.OptimizeMode,
    target: std.Build.ResolvedTarget,
    options: Options,
) *std.Build.Step {
    const zgml_pkg = package(b, target, optimize, .{ .options = options });

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/inference.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgml_options", .module = zgml_pkg.zgml_options },
        },
    });

    const test_exe = b.addTest(.{
        .name = "zgml-inference-tests",
        .root_module = test_mod,
    });
    if (options.use_blas) {
        linkBlas(target, test_exe);
        test_exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    linkMetal(b, target, test_exe);

    return &b.addRunArtifact(test_exe).step;
}
