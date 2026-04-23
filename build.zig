const std = @import("std");
const wgpu_native = @import("wgpu_native");

pub const Options = struct {
    use_blas: bool = false,
    use_wgpu: bool = false,
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
        if (pkg.options.use_wgpu) linkWgpu(b, pkg.target, exe);

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

fn linkWgpu(b: *std.Build, target: std.Build.ResolvedTarget, exe: *std.Build.Step.Compile) void {
    const dep = b.dependency("wgpu_native", .{});
    _ = wgpu_native.link(dep, target, exe);
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
    step.addOption(bool, "use_wgpu", args.options.use_wgpu);

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

    // WebGPU backend: include path for webgpu.h / wgpu.h headers.
    if (args.options.use_wgpu) {
        const dep = b.dependency("wgpu_native", .{});
        if (wgpu_native.includePath(dep, target)) |inc| {
            zgml.addIncludePath(inc);
        }
    }

    return .{
        .target = target,
        .options = args.options,
        .zgml = zgml,
        .zgml_options = zgml_options,
    };
}

// ---------------------------------------------------------------
// Executable helper — eliminates per-target boilerplate
// ---------------------------------------------------------------

const ExeConfig = struct {
    name: []const u8,
    src: []const u8,
    step_name: []const u8,
    step_desc: []const u8,
    /// null = use the user's optimize option (for debug/test targets)
    optimize: ?std.builtin.OptimizeMode = .ReleaseFast,
    /// install-only targets don't get a run step
    install_only: bool = false,
    /// extra linkMetal call (bench-metal, generate-llama, etc.)
    extra_metal: bool = false,
    /// extra linkWgpu call (bench-wgpu, etc.)
    extra_wgpu: bool = false,
    /// skip BLAS linking (bench-reduce)
    skip_blas: bool = false,
};

fn addExe(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    options: Options,
    cfg: ExeConfig,
) *std.Build.Step.Compile {
    const opt = cfg.optimize orelse optimize;
    const pkg = package(b, target, opt, .{ .options = options });
    const mod = b.createModule(.{
        .root_source_file = b.path(cfg.src),
        .target = target,
        .optimize = opt,
        .imports = &.{
            .{ .name = "zgml", .module = pkg.zgml },
            .{ .name = "zgml_options", .module = pkg.zgml_options },
        },
    });
    const exe = b.addExecutable(.{ .name = cfg.name, .root_module = mod });
    if (options.use_blas and !cfg.skip_blas) {
        linkBlas(target, exe);
        exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    if (cfg.extra_metal) linkMetal(b, target, exe);
    if (cfg.extra_wgpu) linkWgpu(b, target, exe);
    b.installArtifact(exe);

    const step = b.step(cfg.step_name, cfg.step_desc);
    if (cfg.install_only) {
        step.dependOn(&b.addInstallArtifact(exe, .{}).step);
    } else {
        step.dependOn(&b.addRunArtifact(exe).step);
    }
    return exe;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const use_blas = b.option(bool, "use-blas", "Use BLAS library") orelse (target.result.os.tag == .macos);
    const use_wgpu = b.option(bool, "use-wgpu", "Use WebGPU backend via wgpu-native") orelse false;

    const build_opts = Options{ .use_blas = use_blas, .use_wgpu = use_wgpu };

    _ = package(b, target, optimize, .{ .options = build_opts });

    const test_step = b.step("test", "Run zgml tests");
    test_step.dependOn(runTests(b, optimize, target, build_opts));
    test_step.dependOn(runInferenceTests(b, optimize, target, build_opts));

    // -- Benchmarks --

    const bench_exe = addExe(b, target, optimize, build_opts, .{
        .name = "zgml-bench",
        .src = "src/bench.zig",
        .step_name = "bench",
        .step_desc = "Build and run zgml benchmarks",
    });
    const bench_build_step = b.step("bench-build", "Build zgml benchmarks");
    bench_build_step.dependOn(&bench_exe.step);

    const frontier_opt = .ReleaseFast;
    const frontier_options_step = b.addOptions();
    frontier_options_step.addOption(bool, "use_blas", build_opts.use_blas);
    frontier_options_step.addOption(bool, "use_wgpu", build_opts.use_wgpu);
    const frontier_options = frontier_options_step.createModule();
    const frontier_zgml = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = frontier_opt,
        .imports = &.{
            .{ .name = "zgml_options", .module = frontier_options },
        },
        .link_libc = if (build_opts.use_blas) true else null,
    });
    if (build_opts.use_blas) {
        frontier_zgml.addIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    if (target.result.os.tag == .macos) {
        frontier_zgml.addIncludePath(b.path("src/backend"));
    }
    if (build_opts.use_wgpu) {
        const dep = b.dependency("wgpu_native", .{});
        if (wgpu_native.includePath(dep, target)) |inc| {
            frontier_zgml.addIncludePath(inc);
        }
    }
    const frontier_mod = b.createModule(.{
        .root_source_file = b.path("benchmarks/frontier_bench.zig"),
        .target = target,
        .optimize = frontier_opt,
        .imports = &.{
            .{ .name = "zgml_options", .module = frontier_options },
            .{ .name = "zgml", .module = frontier_zgml },
        },
    });
    const frontier_exe = b.addExecutable(.{ .name = "bench-frontier", .root_module = frontier_mod });
    linkMetal(b, target, frontier_exe);
    if (build_opts.use_wgpu) linkWgpu(b, target, frontier_exe);
    if (build_opts.use_blas) {
        linkBlas(target, frontier_exe);
        frontier_exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }
    b.installArtifact(frontier_exe);
    const frontier_step = b.step("bench-frontier", "Run decision-grade benchmark frontier");
    frontier_step.dependOn(&b.addRunArtifact(frontier_exe).step);
    bench_build_step.dependOn(&frontier_exe.step);

    _ = addExe(b, target, optimize, build_opts, .{
        .name = "mnist-bench",
        .src = "src/mnist_bench.zig",
        .step_name = "mnist-bench",
        .step_desc = "Build and run MNIST CNN training benchmark",
    });
    _ = addExe(b, target, optimize, build_opts, .{
        .name = "mnist-micro",
        .src = "src/mnist_micro.zig",
        .step_name = "mnist-micro",
        .step_desc = "Build and run MNIST CNN training micro-benchmark",
    });
    _ = addExe(b, target, optimize, build_opts, .{
        .name = "conv-phase-bench",
        .src = "src/conv_phase_bench.zig",
        .step_name = "conv-phase-bench",
        .step_desc = "Run parameterized conv phase benchmark",
    });
    _ = addExe(b, target, optimize, build_opts, .{
        .name = "op-bench",
        .src = "benchmarks/op_bench.zig",
        .step_name = "op-bench",
        .step_desc = "Per-op microbenchmark for MNIST CNN ops",
    });
    _ = addExe(b, target, optimize, build_opts, .{
        .name = "bench-reduce",
        .src = "src/bench_reduce.zig",
        .step_name = "bench-reduce",
        .step_desc = "Benchmark reduction and broadcast ops",
        .skip_blas = true,
    });
    _ = addExe(b, target, optimize, build_opts, .{
        .name = "bench-inference",
        .src = "bench/inference.zig",
        .step_name = "bench-inference",
        .step_desc = "Benchmark inference session (f32 vs int8)",
    });
    _ = addExe(b, target, optimize, build_opts, .{
        .name = "bench-wgpu",
        .src = "benchmarks/wgpu_bench.zig",
        .step_name = "bench-wgpu",
        .step_desc = "Benchmark WebGPU GPU vs CPU BLAS matmul",
        .extra_wgpu = true,
    });
    _ = addExe(b, target, optimize, build_opts, .{
        .name = "bench-metal",
        .src = "benchmarks/metal_bench.zig",
        .step_name = "bench-metal",
        .step_desc = "Benchmark Metal GPU vs CPU BLAS matmul",
        .extra_metal = true,
    });
    _ = addExe(b, target, optimize, build_opts, .{
        .name = "bench-metal-inference",
        .src = "benchmarks/metal_inference_bench.zig",
        .step_name = "bench-metal-inference",
        .step_desc = "Benchmark CPU vs Metal inference tok/s",
        .extra_metal = true,
    });
    _ = addExe(b, target, optimize, build_opts, .{
        .name = "bench-llama-smollm",
        .src = "benchmarks/llama_smollm_bench.zig",
        .step_name = "bench-llama-smollm",
        .step_desc = "Benchmark SmolLM LLaMA inference throughput",
        .extra_metal = true,
        .extra_wgpu = build_opts.use_wgpu,
    });

    // -- Profiling & grad check --

    _ = addExe(b, target, optimize, build_opts, .{
        .name = "profile-bwd",
        .src = "src/profile_backward.zig",
        .step_name = "profile-bwd",
        .step_desc = "Profile backward pass per-op timing",
    });
    _ = addExe(b, target, optimize, build_opts, .{
        .name = "grad-check",
        .src = "benchmarks/grad_check.zig",
        .step_name = "grad-check",
        .step_desc = "Compare gradients with PyTorch reference",
        .optimize = null, // use user's optimize setting
    });

    // -- Scripts --

    _ = addExe(b, target, optimize, build_opts, .{
        .name = "generate",
        .src = "scripts/generate.zig",
        .step_name = "generate",
        .step_desc = "Build text generation binary",
        .install_only = true,
    });
    _ = addExe(b, target, optimize, build_opts, .{
        .name = "train-tiny",
        .src = "scripts/train_tiny.zig",
        .step_name = "train-tiny",
        .step_desc = "Train a tiny GPT and save checkpoint",
        .install_only = true,
    });
    _ = addExe(b, target, optimize, build_opts, .{
        .name = "generate-pretrained",
        .src = "scripts/generate_pretrained.zig",
        .step_name = "generate-pretrained",
        .step_desc = "Generate text from a pretrained HF model",
        .install_only = true,
    });
    _ = addExe(b, target, optimize, build_opts, .{
        .name = "generate-llama",
        .src = "scripts/generate_llama.zig",
        .step_name = "generate-llama",
        .step_desc = "Generate text from a pretrained LLaMA model",
        .install_only = true,
        .extra_metal = true,
    });
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
    if (options.use_wgpu) linkWgpu(b, target, test_exe);
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
    if (options.use_wgpu) linkWgpu(b, target, test_exe);

    return &b.addRunArtifact(test_exe).step;
}
