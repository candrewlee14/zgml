const std = @import("std");

fn pathExists(path: []const u8) bool {
    const path_z = std.posix.toPosixPath(path) catch return false;
    return std.c.access(&path_z, 0) == 0;
}

pub const Options = struct {
    use_blas: bool = false,
    use_metal: bool = true,
    use_wgpu: bool = false,
    experimental_llama_wgpu_execution: bool = false,
};

pub const Package = struct {
    target: std.Build.ResolvedTarget,
    options: Options,
    zgml: *std.Build.Module,

    pub fn link(pkg: Package, b: *std.Build, exe: *std.Build.Step.Compile) void {
        exe.root_module.addImport("zgml", pkg.zgml);
        linkConfiguredBackends(b, pkg.target, pkg.options, exe, .{});
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

fn wgpuDependency(b: *std.Build) *std.Build.Dependency {
    return b.dependency("wgpu_native", .{});
}

fn resolveWgpuNativeUpstream(dep: *std.Build.Dependency, target: std.Build.ResolvedTarget) ?*std.Build.Dependency {
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

fn wgpuNativeUpstream(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Dependency {
    const dep = wgpuDependency(b);
    return resolveWgpuNativeUpstream(dep, target) orelse @panic("wgpu-native does not provide artifacts for this target");
}

fn addWgpuIncludePath(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    mod: *std.Build.Module,
) void {
    mod.addIncludePath(wgpuNativeUpstream(b, target).path("include"));
}

fn linkWgpu(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    exe: *std.Build.Step.Compile,
) void {
    const upstream = wgpuNativeUpstream(b, target);
    exe.root_module.addIncludePath(upstream.path("include"));
    exe.root_module.addLibraryPath(upstream.path("lib"));
    exe.root_module.linkSystemLibrary("wgpu_native", .{});
    exe.root_module.link_libc = true;

    switch (target.result.os.tag) {
        .macos => {
            exe.root_module.linkFramework("Metal", .{});
            exe.root_module.linkFramework("QuartzCore", .{});
            exe.root_module.linkFramework("CoreFoundation", .{});
            exe.root_module.linkFramework("Foundation", .{});
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
            for (win_libs) |name| exe.root_module.linkSystemLibrary(name, .{});
        },
        else => {},
    }
}

fn linkConfiguredBackends(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    options: Options,
    exe: *std.Build.Step.Compile,
    link_options: struct {
        blas: bool = true,
        blas_include_path: bool = false,
        metal: bool = true,
        wgpu: bool = true,
    },
) void {
    if (link_options.blas and options.use_blas) {
        linkBlas(target, exe);
        if (link_options.blas_include_path) {
            exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
        }
    }
    if (link_options.metal and options.use_metal) linkMetal(b, target, exe);
    if (link_options.wgpu and options.use_wgpu) linkWgpu(b, target, exe);
}

fn addBackendIncludePaths(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    options: Options,
    mod: *std.Build.Module,
) void {
    if (options.use_blas) {
        mod.addIncludePath(.{ .cwd_relative = "/usr/include/openblas" });
    }

    // Metal backend: include path for the C shim header so metal.zig can @cImport it.
    // Actual linking happens in Package.link() and linkMetal() on the compile step.
    if (options.use_metal and target.result.os.tag == .macos) {
        mod.addIncludePath(b.path("src/backend"));
    }
    if (options.use_wgpu) addWgpuIncludePath(b, target, mod);
}

fn optionsModule(b: *std.Build, options: Options) *std.Build.Module {
    const step = b.addOptions();
    step.addOption(bool, "use_blas", options.use_blas);
    step.addOption(bool, "use_metal", options.use_metal);
    step.addOption(bool, "use_wgpu", options.use_wgpu);
    step.addOption(bool, "experimental_llama_wgpu_execution", options.experimental_llama_wgpu_execution);
    return step.createModule();
}

fn addZgmlModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    options: Options,
    name: []const u8,
    root: []const u8,
) struct {
    module: *std.Build.Module,
    zgml_options: *std.Build.Module,
} {
    const zgml_options = optionsModule(b, options);
    const module = b.addModule(name, .{
        .root_source_file = b.path(root),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgml_options", .module = zgml_options },
        },
        .link_libc = if (options.use_blas) true else null,
    });
    addBackendIncludePaths(b, target, options, module);
    return .{
        .module = module,
        .zgml_options = zgml_options,
    };
}

pub fn package(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    options: Options,
) Package {
    const module = addZgmlModule(b, target, optimize, options, "zgml", "src/main.zig");
    return .{
        .target = target,
        .options = options,
        .zgml = module.module,
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
    run_args: []const []const u8 = &.{},
    /// null = use the user's optimize option (for debug/test targets)
    optimize: ?std.builtin.OptimizeMode = .ReleaseFast,
    include_in_bench_build: bool = false,
    include_in_check: bool = false,
};

const AddedExe = struct {
    exe: *std.Build.Step.Compile,
    install_step: *std.Build.Step,
    run_step: *std.Build.Step,
};

const CApiArtifacts = struct {
    lib: *std.Build.Step.Compile,
    tests: *std.Build.Step.Compile,
    install: *std.Build.Step.InstallArtifact,
};

fn addCApi(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    options: Options,
) CApiArtifacts {
    const zgml_options = optionsModule(b, options);
    const mod = b.createModule(.{
        .root_source_file = b.path("src/c_api.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgml_options", .module = zgml_options },
        },
    });
    addBackendIncludePaths(b, target, options, mod);
    const lib = b.addLibrary(.{
        .name = "zgml_c",
        .root_module = mod,
        .linkage = .dynamic,
    });
    linkConfiguredBackends(b, target, options, lib, .{});
    lib.installHeader(b.path("include/zgml.h"), "zgml.h");
    const lib_install = b.addInstallArtifact(lib, .{});
    b.getInstallStep().dependOn(&lib_install.step);

    const tests = b.addTest(.{
        .name = "zgml-c-api-tests",
        .root_module = mod,
    });
    b.installArtifact(tests);

    return .{ .lib = lib, .tests = tests, .install = lib_install };
}

fn addCApiWasm(
    b: *std.Build,
    optimize: std.builtin.OptimizeMode,
    options: Options,
) *std.Build.Step.InstallArtifact {
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .wasi,
    });
    const zgml_options = optionsModule(b, options);
    const mod = b.createModule(.{
        .root_source_file = b.path("src/c_api.zig"),
        .target = wasm_target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgml_options", .module = zgml_options },
        },
    });
    const wasm = b.addExecutable(.{
        .name = "zgml_c",
        .root_module = mod,
    });
    wasm.entry = .disabled;
    wasm.rdynamic = true;
    return b.addInstallArtifact(wasm, .{});
}

fn addCApiSmoke(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    c_api_lib: *std.Build.Step.Compile,
) *std.Build.Step {
    const step = b.step("ffi-c-smoke", "Run C ABI smoke executable");
    if (!pathExists("examples/c_ffi_smoke.c")) return step;

    const mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addIncludePath(b.path("include"));
    mod.addCSourceFile(.{
        .file = b.path("examples/c_ffi_smoke.c"),
        .flags = &.{"-std=c11"},
    });
    mod.linkLibrary(c_api_lib);

    const exe = b.addExecutable(.{
        .name = "zgml-c-ffi-smoke",
        .root_module = mod,
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    step.dependOn(&run.step);
    return step;
}

fn addLlamaWgpuExperimentalSmoke(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    options: Options,
) *std.Build.Step {
    const pkg = package(b, target, optimize, options);
    const tests = b.addTest(.{
        .name = "zgml-llama-wgpu-experimental-tests",
        .root_module = pkg.zgml,
        .filters = &.{
            "llm facade exposes WebGPU executable or resource-probe target",
            "llm facade WebGPU resource handoff covers GQA multilayer shape",
            "llm facade WebGPU resource handoff covers MQA multilayer shape",
            "llm facade WebGPU resource handoff covers MHA multilayer shape",
            "llm facade WebGPU resource handoff covers tied LM head shape",
            "llm facade WebGPU resource handoff covers realistic head width",
            "llm facade WebGPU resource handoff covers realistic GQA multilayer shape",
            "llm facade WebGPU resource handoff covers quantized projections",
            "llm facade WebGPU resource handoff uses compiled context envelope",
            "llm facade WebGPU resource handoff rejects over-context without GPU work",
            "native WebGPU tiny llama MQA multilayer resource handoff matches CPU",
        },
    });
    linkConfiguredBackends(b, target, options, tests, .{});
    b.installArtifact(tests);

    const run = b.addRunArtifact(tests);
    return &run.step;
}

fn addExe(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    options: Options,
    cfg: ExeConfig,
) AddedExe {
    const opt = cfg.optimize orelse optimize;
    const pkg = addZgmlModule(b, target, opt, options, "zgml_internal", "src/internal.zig");
    const mod = b.createModule(.{
        .root_source_file = b.path(cfg.src),
        .target = target,
        .optimize = opt,
        .imports = &.{
            .{ .name = "zgml_internal", .module = pkg.module },
            .{ .name = "zgml_options", .module = pkg.zgml_options },
        },
    });
    const exe = b.addExecutable(.{ .name = cfg.name, .root_module = mod });
    linkConfiguredBackends(b, target, options, exe, .{ .blas_include_path = true });
    const install = b.addInstallArtifact(exe, .{});

    const run = b.addRunArtifact(exe);
    run.addArgs(cfg.run_args);
    const step = b.step(cfg.step_name, cfg.step_desc);
    step.dependOn(&run.step);
    return .{ .exe = exe, .install_step = &install.step, .run_step = &run.step };
}

fn addPublicExe(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    cfg: ExeConfig,
) AddedExe {
    const opt = cfg.optimize orelse optimize;
    const pkg = addZgmlModule(b, target, opt, .{}, "zgml_public_example", "src/main.zig");
    const mod = b.createModule(.{
        .root_source_file = b.path(cfg.src),
        .target = target,
        .optimize = opt,
        .imports = &.{
            .{ .name = "zgml", .module = pkg.module },
            .{ .name = "zgml_options", .module = pkg.zgml_options },
        },
    });
    const exe = b.addExecutable(.{ .name = cfg.name, .root_module = mod });
    const install = b.addInstallArtifact(exe, .{});

    const run = b.addRunArtifact(exe);
    run.addArgs(cfg.run_args);
    const step = b.step(cfg.step_name, cfg.step_desc);
    step.dependOn(&run.step);
    return .{ .exe = exe, .install_step = &install.step, .run_step = &run.step };
}

fn addWgpuLinkSmoke(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step {
    const options = Options{ .use_wgpu = true };
    const zgml_options = optionsModule(b, options);
    const mod = b.createModule(.{
        .root_source_file = b.path("src/backend/wgpu_probe.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgml_options", .module = zgml_options },
        },
    });
    addBackendIncludePaths(b, target, options, mod);

    const probe = b.addTest(.{
        .name = "zgml-wgpu-link-smoke",
        .root_module = mod,
    });
    linkConfiguredBackends(b, target, options, probe, .{ .blas = false, .metal = false });
    b.installArtifact(probe);

    const run = b.addRunArtifact(probe);
    return &run.step;
}

fn addWgpuExecSmoke(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step {
    const options = Options{ .use_wgpu = true };
    const zgml_options = optionsModule(b, options);
    const mod = b.createModule(.{
        .root_source_file = b.path("src/backend_wgpu_test.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zgml_options", .module = zgml_options },
        },
    });
    addBackendIncludePaths(b, target, options, mod);

    const probe = b.addTest(.{
        .name = "zgml-wgpu-exec-smoke",
        .root_module = mod,
    });
    linkConfiguredBackends(b, target, options, probe, .{ .blas = false, .metal = false });
    b.installArtifact(probe);

    const run = b.addRunArtifact(probe);
    return &run.step;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const use_blas = b.option(bool, "use-blas", "Use BLAS library") orelse (target.result.os.tag == .macos);
    const use_metal = b.option(bool, "use-metal", "Enable Metal backend on macOS") orelse (target.result.os.tag == .macos);
    const use_wgpu = b.option(bool, "use-wgpu", "Enable wgpu-native dependency and probes") orelse false;
    const experimental_llama_wgpu_execution = b.option(bool, "experimental-llama-wgpu-execution", "Enable LLaMA-family native WebGPU execution; defaults to -Duse-wgpu=true and can be set false for resource-probe-only builds") orelse use_wgpu;

    const build_opts = Options{
        .use_blas = use_blas,
        .use_metal = use_metal,
        .use_wgpu = use_wgpu,
        .experimental_llama_wgpu_execution = use_wgpu and experimental_llama_wgpu_execution,
    };

    _ = package(b, target, optimize, build_opts);

    const c_api = addCApi(b, target, optimize, build_opts);
    const c_api_wasm = addCApiWasm(b, optimize, .{ .use_blas = false, .use_metal = false });
    const ffi_wasm_step = b.step("ffi-wasm", "Build the exported Wasm C ABI module");
    ffi_wasm_step.dependOn(&c_api_wasm.step);
    const ffi_wasm_smoke_step = b.step("ffi-wasm-smoke", "Run Wasm C ABI smoke with Node/WASI");
    if (pathExists("examples/wasm_ffi/smoke.mjs")) {
        const wasm_ffi_smoke = b.addSystemCommand(&.{
            "node",
            "--no-warnings",
            "examples/wasm_ffi/smoke.mjs",
        });
        wasm_ffi_smoke.step.dependOn(&c_api_wasm.step);
        ffi_wasm_smoke_step.dependOn(&wasm_ffi_smoke.step);
    }
    const ffi_wasm_browser_smoke_step = b.step("ffi-wasm-browser-smoke", "Run browser Wasm C ABI smoke with Chrome/Chromium");
    if (pathExists("examples/wasm_ffi/browser_smoke_runner.mjs")) {
        const wasm_browser_smoke = b.addSystemCommand(&.{
            "node",
            "--no-warnings",
            "examples/wasm_ffi/browser_smoke_runner.mjs",
        });
        wasm_browser_smoke.step.dependOn(&c_api_wasm.step);
        ffi_wasm_browser_smoke_step.dependOn(&wasm_browser_smoke.step);
    }
    const ffi_wasm_browser_llama_focused_smoke_step = b.step("ffi-wasm-browser-llama-focused-smoke", "Run focused browser Wasm LLaMA family proof with Chrome/Chromium");
    if (pathExists("examples/wasm_ffi/browser_smoke_runner.mjs")) {
        const wasm_browser_llama_focused_smoke = b.addSystemCommand(&.{
            "node",
            "--no-warnings",
            "examples/wasm_ffi/browser_smoke_runner.mjs",
            "--llama-profile-label=gguf-smollm3-nope-gqa-pipeline",
            "--timeout-ms=300000",
        });
        wasm_browser_llama_focused_smoke.step.dependOn(&c_api_wasm.step);
        ffi_wasm_browser_llama_focused_smoke_step.dependOn(&wasm_browser_llama_focused_smoke.step);
    }
    const ffi_wasm_browser_gpu_smoke_step = b.step("ffi-wasm-browser-gpu-smoke", "Run browser Wasm C ABI smoke and require real GPUBuffer mode");
    if (pathExists("examples/wasm_ffi/browser_smoke_runner.mjs")) {
        const wasm_browser_gpu_smoke = b.addSystemCommand(&.{
            "node",
            "--no-warnings",
            "examples/wasm_ffi/browser_smoke_runner.mjs",
            "--enable-unsafe-webgpu",
            "--require-gpu",
            "--timeout-ms=1800000",
        });
        wasm_browser_gpu_smoke.step.dependOn(&c_api_wasm.step);
        ffi_wasm_browser_gpu_smoke_step.dependOn(&wasm_browser_gpu_smoke.step);
    }
    const c_api_smoke = addCApiSmoke(b, target, optimize, c_api.lib);
    const ffi_c_step = b.step("ffi-c", "Build the dynamic C ABI library for FFI hosts");
    ffi_c_step.dependOn(&c_api.install.step);
    const ffi_bun_smoke_step = b.step("ffi-bun-smoke", "Run Bun FFI smoke");
    if (pathExists("examples/bun_ffi/smoke.ts")) {
        const bun_ffi_smoke = b.addSystemCommand(&.{
            "bun",
            "run",
            "examples/bun_ffi/smoke.ts",
        });
        bun_ffi_smoke.step.dependOn(&c_api.install.step);
        ffi_bun_smoke_step.dependOn(&bun_ffi_smoke.step);
    }
    const ffi_node_smoke_step = b.step("ffi-node-smoke", "Run Node package-adapter FFI smoke after npm --prefix examples/node_ffi install");
    if (pathExists("examples/node_ffi/package.json")) {
        const node_ffi_smoke = b.addSystemCommand(&.{
            "npm",
            "--prefix",
            "examples/node_ffi",
            "run",
            "smoke",
        });
        node_ffi_smoke.step.dependOn(&c_api.install.step);
        ffi_node_smoke_step.dependOn(&node_ffi_smoke.step);
    }
    const wgpu_link_smoke_step = b.step("wgpu-link-smoke", "Compile and link the optional wgpu-native probe; pass -Duse-wgpu=true");
    if (use_wgpu) {
        wgpu_link_smoke_step.dependOn(addWgpuLinkSmoke(b, target, optimize));
    }
    const wgpu_exec_smoke_step = b.step("wgpu-exec-smoke", "Run the optional tiny-linear wgpu execution smoke; pass -Duse-wgpu=true");
    if (use_wgpu) {
        wgpu_exec_smoke_step.dependOn(addWgpuExecSmoke(b, target, optimize));
    }
    const llama_wgpu_experimental_smoke_step = b.step("llama-wgpu-experimental-smoke", "Run LLaMA WebGPU public Zig/C/Node/Bun smokes; pass -Duse-wgpu=true");
    if (use_wgpu and build_opts.experimental_llama_wgpu_execution) {
        llama_wgpu_experimental_smoke_step.dependOn(addLlamaWgpuExperimentalSmoke(b, target, optimize, build_opts));
        llama_wgpu_experimental_smoke_step.dependOn(c_api_smoke);
        llama_wgpu_experimental_smoke_step.dependOn(ffi_node_smoke_step);
        llama_wgpu_experimental_smoke_step.dependOn(ffi_bun_smoke_step);
    }
    const wgpu_check_step = b.step("wgpu-check", "Run all optional native WebGPU validation gates; pass -Duse-wgpu=true");
    if (use_wgpu) {
        wgpu_check_step.dependOn(wgpu_link_smoke_step);
        wgpu_check_step.dependOn(wgpu_exec_smoke_step);
        if (build_opts.experimental_llama_wgpu_execution) {
            wgpu_check_step.dependOn(llama_wgpu_experimental_smoke_step);
        }
    }

    const test_step = b.step("test", "Run zgml tests");
    test_step.dependOn(runTests(b, optimize, target, build_opts, c_api.tests));

    const bench_build_step = b.step("bench-build", "Build zgml benchmarks");
    const baseline_check_step = b.step("bench-baseline-check", "Verify checked benchmark baseline artifacts");
    if (pathExists("scripts/verify_bench_artifact.py") and
        pathExists("benchmarks/baselines/smollm-m5pro-p128-g200-r3.json") and
        pathExists("benchmarks/baselines/smollm-stencil-p128.json"))
    {
        const baseline_check = b.addSystemCommand(&.{
            "python3",
            "scripts/verify_bench_artifact.py",
            "--status",
            "benchmarks/baselines/smollm-m5pro-p128-g200-r3.json",
            "benchmarks/baselines/smollm-stencil-p128.json",
        });
        baseline_check.setEnvironmentVariable("PYTHONPYCACHEPREFIX", ".zig-cache/pycache");
        baseline_check_step.dependOn(&baseline_check.step);
    }

    const check_step = b.step("check", "Run local/subagent validation gate");
    check_step.dependOn(test_step);
    check_step.dependOn(bench_build_step);
    check_step.dependOn(baseline_check_step);
    check_step.dependOn(&c_api.lib.step);
    check_step.dependOn(c_api_smoke);
    check_step.dependOn(ffi_wasm_smoke_step);
    check_step.dependOn(ffi_wasm_browser_smoke_step);

    const exes = [_]ExeConfig{
        .{ .name = "bench-frontier", .src = "benchmarks/frontier_bench.zig", .step_name = "bench-frontier", .step_desc = "Run decision-grade benchmark frontier", .include_in_bench_build = true },
        .{ .name = "bench-llama-smollm", .src = "benchmarks/llama_smollm_bench.zig", .step_name = "bench-llama-smollm", .step_desc = "Run SmolLM copy-and-patch stencil probe", .run_args = &.{ "ignored", "128", "1", "1", "--stencil-only" }, .include_in_bench_build = true, .include_in_check = true },
    };

    for (exes) |cfg| {
        if (!pathExists(cfg.src)) continue;
        const added = addExe(b, target, optimize, build_opts, cfg);
        if (cfg.include_in_bench_build) bench_build_step.dependOn(added.install_step);
        if (cfg.include_in_check) check_step.dependOn(added.run_step);
    }

    if (pathExists("examples/train_linear.zig")) {
        const train_example = addPublicExe(b, target, optimize, .{
            .name = "native-helper-train-linear",
            .src = "examples/train_linear.zig",
            .step_name = "native-helper-train-linear",
            .step_desc = "Run native helper linear-training smoke",
            .optimize = null,
        });
        check_step.dependOn(train_example.run_step);
    }
}

fn runTests(
    b: *std.Build,
    optimize: std.builtin.OptimizeMode,
    target: std.Build.ResolvedTarget,
    options: Options,
    c_api_test: *std.Build.Step.Compile,
) *std.Build.Step {
    const zgml_pkg = package(b, target, optimize, options);

    const public_test = b.addTest(.{
        .name = "zgml-tests",
        .root_module = zgml_pkg.zgml,
    });
    linkConfiguredBackends(b, target, options, public_test, .{});
    b.installArtifact(public_test);

    const internal_pkg = addZgmlModule(b, target, optimize, options, "zgml_internal", "src/internal.zig");
    const internal_test = b.addTest(.{
        .name = "zgml-internal-tests",
        .root_module = internal_pkg.module,
    });
    linkConfiguredBackends(b, target, options, internal_test, .{});
    b.installArtifact(internal_test);

    const conformance_pkg = addZgmlModule(b, target, optimize, options, "zgml-backend-conformance", "src/backend_conformance_test.zig");
    const conformance_test = b.addTest(.{
        .name = "zgml-backend-conformance-tests",
        .root_module = conformance_pkg.module,
    });
    linkConfiguredBackends(b, target, options, conformance_test, .{});
    b.installArtifact(conformance_test);

    const tests = b.step("unit-tests", "Run public and internal zgml tests");
    tests.dependOn(&b.addRunArtifact(public_test).step);
    tests.dependOn(&b.addRunArtifact(internal_test).step);
    tests.dependOn(&b.addRunArtifact(conformance_test).step);
    tests.dependOn(&b.addRunArtifact(c_api_test).step);
    return tests;
}
