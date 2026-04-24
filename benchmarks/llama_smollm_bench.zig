//! SmolLM LLaMA inference benchmark for zgml.
//!
//! Measures prompt and decode throughput on the local SmolLM checkpoint using
//! the real `LlamaInferenceSession` path, without tokenizer or stdout noise.
//!
//! Run:
//!   zig build bench-llama-smollm
//!   ./zig-out/bin/bench-llama-smollm [model.safetensors|model.gguf] [prompt_tokens] [gen_tokens] [repetitions]

const std = @import("std");
const zgml = @import("zgml");

const CpuBackend = zgml.backend_cpu.CpuBackend;
const WgpuBackend = zgml.backend_wgpu.WgpuBackend;
const MetalBackend = zgml.backend_metal.MetalBackend;
const DeviceInference = zgml.device_inference.DeviceInference;
const Backend = zgml.backend.Backend;
const Tensor = zgml.Tensor;
const profile = zgml.profile;
const backend_program = zgml.backend_program;
const have_wgpu = @import("zgml_options").use_wgpu;
const is_macos = @import("builtin").os.tag == .macos;

const config = zgml.models.LlamaConfig{
    .vocab_size = 49152,
    .d_model = 576,
    .n_heads = 9,
    .n_kv_heads = 3,
    .d_ff = 1536,
    .n_layers = 30,
    .max_seq_len = 2048,
    .rope_base = 10000.0,
    .rms_norm_eps = 1e-5,
    .tied_lm_head = true,
};

const Session = zgml.llama_inference.LlamaInferenceSession(f32, config);

const BenchConfig = struct {
    model_path: []const u8,
    prompt_tokens: usize,
    gen_tokens: usize,
    repetitions: usize,
};

const BenchResult = struct {
    prompt_tok_s: f64,
    gen_tok_s: f64,
    prompt_avg_ms: f64,
    gen_avg_ms: f64,
};

fn loadWeights(session: *Session, alloc: std.mem.Allocator, model_path: []const u8, io: std.Io) !void {
    return switch (zgml.llm.inferFileKind(model_path) orelse return error.UnknownModelFormat) {
        .safetensors => {
            var sf = try zgml.safetensors.SafetensorsFile.open(alloc, model_path, io);
            defer sf.deinit();
            session.clearDirectQuantWeights();
            try zgml.models.llama_loader.loadLlama(f32, config, &session.model, &sf);
        },
        .gguf => {
            var gf = try zgml.gguf.GGUFFile.open(alloc, io, model_path);
            defer gf.deinit();
            try session.loadGGUFDirectQuantized(&gf);
        },
    };
}

fn isGGUF(model_path: []const u8) bool {
    return (zgml.llm.inferFileKind(model_path) orelse return false) == .gguf;
}

fn schedulePolicyForBackend(be: Backend) backend_program.SchedulePolicy {
    return switch (be.device_type) {
        .wgpu => .{
            .capabilities = be.capabilities,
            .native_kernels = .{
                .elementwise = true,
                .fused_elementwise = be.capabilities.fused_elementwise,
                .row = true,
                .reduce = true,
                .movement = true,
                .matmul = be.capabilities.dense_matmul_f32,
                .qmatvec = be.capabilities.qmatmul,
                .qmatmul = be.capabilities.qmatmul,
                .rope = true,
                .attention = be.capabilities.attention.supported,
            },
            .fine_grained = true,
            .min_backend_matmul_m = 0,
            .min_backend_qmatmul_m = 0,
        },
        .cpu => .{
            .capabilities = be.capabilities,
            .native_kernels = .{
                .elementwise = true,
                .fused_elementwise = be.capabilities.fused_elementwise,
                .row = true,
                .reduce = true,
                .movement = true,
                .matmul = be.capabilities.dense_matmul_f32,
                .qmatvec = be.capabilities.qmatmul,
                .qmatmul = be.capabilities.qmatmul,
                .rope = true,
                .attention = be.capabilities.attention.supported,
            },
            .fine_grained = true,
            .min_backend_matmul_m = 0,
            .min_backend_qmatmul_m = 0,
        },
        else => backend_program.SchedulePolicy.conservative(be.capabilities),
    };
}

fn runVariant(
    label: []const u8,
    maybe_backend: ?Backend,
    quantized: bool,
    quant_kv: bool,
    cfg: BenchConfig,
    writer: anytype,
    io: std.Io,
    alloc: std.mem.Allocator,
) !BenchResult {
    var session = if (maybe_backend) |backend|
        try Session.initWithBackend(alloc, backend)
    else
        try Session.init(alloc);
    defer session.deinit();

    try loadWeights(&session, alloc, cfg.model_path, io);
    if (quantized and !isGGUF(cfg.model_path)) try session.quantize();
    if (quant_kv) try session.quantizeKV();

    // Pre-build the prompt token buffer once. The session takes it verbatim.
    const prompt = try alloc.alloc(usize, cfg.prompt_tokens);
    defer alloc.free(prompt);
    for (prompt, 0..) |*t, i| t.* = (i + 1) % config.vocab_size;

    // Warm up the real decode path and prefill path once each to settle
    // kernel/backend selection and amortize plan-build cost.
    _ = try session.step(0);
    session.reset();
    _ = try session.prefill(prompt);
    session.reset();

    var prompt_total_ns: u128 = 0;
    var gen_total_ns: u128 = 0;

    for (0..cfg.repetitions) |_| {
        session.reset();
        const prompt_start = std.Io.Clock.awake.now(io).nanoseconds;
        _ = try session.prefill(prompt);
        const prompt_end = std.Io.Clock.awake.now(io).nanoseconds;
        prompt_total_ns += @intCast(prompt_end - prompt_start);

        const gen_start = std.Io.Clock.awake.now(io).nanoseconds;
        for (0..cfg.gen_tokens) |i| {
            _ = try session.step((cfg.prompt_tokens + i + 1) % config.vocab_size);
        }
        const gen_end = std.Io.Clock.awake.now(io).nanoseconds;
        gen_total_ns += @intCast(gen_end - gen_start);
    }

    const prompt_ns = @as(f64, @floatFromInt(prompt_total_ns));
    const gen_ns = @as(f64, @floatFromInt(gen_total_ns));
    const prompt_tok_s = @as(f64, @floatFromInt(cfg.prompt_tokens * cfg.repetitions)) / (prompt_ns / 1_000_000_000.0);
    const gen_tok_s = @as(f64, @floatFromInt(cfg.gen_tokens * cfg.repetitions)) / (gen_ns / 1_000_000_000.0);
    const prompt_avg_ms = prompt_ns / @as(f64, @floatFromInt(cfg.repetitions)) / 1_000_000.0;
    const gen_avg_ms = gen_ns / @as(f64, @floatFromInt(cfg.repetitions)) / 1_000_000.0;

    try writer.print(
        "  {s}: prompt {d:>7.1} tok/s ({d:>6.2} ms avg)  decode {d:>7.1} tok/s ({d:>6.2} ms avg)\n",
        .{ label, prompt_tok_s, prompt_avg_ms, gen_tok_s, gen_avg_ms },
    );
    writer.flush() catch {};

    return .{
        .prompt_tok_s = prompt_tok_s,
        .gen_tok_s = gen_tok_s,
        .prompt_avg_ms = prompt_avg_ms,
        .gen_avg_ms = gen_avg_ms,
    };
}

/// Run decode-only benchmark through DeviceInference (compiled GPU program).
fn runDeviceVariant(
    label: []const u8,
    be: Backend,
    cfg: BenchConfig,
    writer: anytype,
    io: std.Io,
    alloc: std.mem.Allocator,
) !BenchResult {
    const d_model = config.d_model;
    const d_head = d_model / config.n_heads;
    const max_seq = config.max_seq_len;

    // Build session without backend — graph captures all ops for device compilation.
    var session = try Session.init(alloc);
    defer session.deinit();
    try loadWeights(&session, alloc, cfg.model_path, io);

    // Build input tensor list: token_input, attn_mask, per-layer RoPE.
    const n_inputs = 2 + config.n_layers;
    const input_tensors = try alloc.alloc(*const Tensor(f32), n_inputs);
    defer alloc.free(input_tensors);
    input_tensors[0] = session.plan.token_input;
    input_tensors[1] = session.plan.attn_mask;
    for (session.plan.trace.layers, 0..) |lt, l| {
        input_tensors[2 + l] = lt.rope;
    }

    // Logits host buffer.
    const logits_buf = try alloc.alloc(f32, config.vocab_size);
    defer alloc.free(logits_buf);

    var device = try DeviceInference(f32).init(.{
        .graph = &session.plan.graph,
        .be = be,
        .alloc = alloc,
        .input_tensors = input_tensors,
        .output_tensor = session.plan.trace.logits,
        .output_host_buf = logits_buf.ptr,
        .output_len = config.vocab_size,
        .quant_weights = session.plan.quant_weights,
        .quant_map = &session.plan.quant_map,
    });
    defer device.deinit();

    const program = device.getProgram();
    const schedule_policy = schedulePolicyForBackend(be);
    profile.printProfile(profile.profileProgramWithSchedule(program, schedule_policy));
    const schedule = try backend_program.buildKernelSchedule(alloc, device.program_ops, schedule_policy);
    defer alloc.free(schedule);
    const qmatvec_regions = try backend_program.buildKernelRegions(alloc, schedule, backend_program.RegionPolicy.qmatvecCluster());
    defer alloc.free(qmatvec_regions);
    profile.printKernelRegionSummary("qmatvec clusters", qmatvec_regions);
    const qmatvec_anchor_runs = try backend_program.buildAnchorRunRegions(alloc, schedule, backend_program.RegionPolicy.qmatvecCluster());
    defer alloc.free(qmatvec_anchor_runs);
    profile.printKernelRegionSummary("qmatvec anchor runs", qmatvec_anchor_runs);
    try profile.printAnchorNeighborhoodSummary(2, alloc, "qmatvec", schedule, backend_program.RegionPolicy.qmatvecCluster(), 8);
    const qmatvec_rope_attention_pattern = [_]backend_program.KernelFamily{ .qmatvec, .rope, .qmatvec, .rope, .movement, .qmatvec, .movement, .attention };
    const qmatvec_rope_attention = try backend_program.buildFamilyPatternRegions(alloc, schedule, &qmatvec_rope_attention_pattern);
    defer alloc.free(qmatvec_rope_attention);
    profile.printKernelRegionSummary("qmatvec-rope-attention pattern", qmatvec_rope_attention);

    const rope = &session.model.blocks[0].rope;
    const tok_data = session.model.token_embed.inner.data;

    // Helper: execute one decode step through DeviceInference.
    const StepCtx = struct {
        fn doStep(
            dev: *DeviceInference(f32),
            plan_: *@TypeOf(session.plan),
            tok_data_: []const f32,
            rope_: anytype,
            token_id: usize,
            pos: usize,
        ) void {
            // 1. Patch token embedding.
            @memcpy(plan_.token_input.data[0..d_model], tok_data_[token_id * d_model ..][0..d_model]);
            // 2. Patch causal mask.
            const mask = plan_.attn_mask.data[0..max_seq];
            @memset(mask[0 .. pos + 1], 0);
            if (pos + 1 < max_seq) @memset(mask[pos + 1 ..], -std.math.inf(f32));
            // 3. Patch RoPE cos/sin for each layer.
            for (plan_.trace.layers) |lt| {
                @memcpy(lt.rope.data[0..d_head], rope_.cos_table.data[pos * d_head ..][0..d_head]);
                @memcpy(lt.rope.data[d_head .. 2 * d_head], rope_.sin_table.data[pos * d_head ..][0..d_head]);
            }
            // 4. Patch KV-cache write offsets and attention seq_kv.
            dev.patchSliceAssignOffset(@intCast(pos));
            dev.patchAttentionSeqKV(@intCast(pos + 1));
            // 5. Execute.
            dev.execute();
        }
    };

    // Warm up.
    StepCtx.doStep(&device, &session.plan, tok_data, rope, 0, 0);
    // Reset KV caches (zero shared memory).
    for (0..config.n_layers) |l| {
        @memset(session.k_caches[l].data, 0);
        @memset(session.v_caches[l].data, 0);
    }

    // Benchmark decode only (prompt is trivially 1-token, focus on decode throughput).
    var gen_total_ns: u128 = 0;
    for (0..cfg.repetitions) |_| {
        // Reset KV caches.
        for (0..config.n_layers) |l| {
            @memset(session.k_caches[l].data, 0);
            @memset(session.v_caches[l].data, 0);
        }

        const gen_start = std.Io.Clock.awake.now(io).nanoseconds;
        for (0..cfg.gen_tokens) |i| {
            StepCtx.doStep(&device, &session.plan, tok_data, rope, (i + 1) % config.vocab_size, i);
        }
        const gen_end = std.Io.Clock.awake.now(io).nanoseconds;
        gen_total_ns += @intCast(gen_end - gen_start);
    }

    const gen_ns = @as(f64, @floatFromInt(gen_total_ns));
    const gen_tok_s = @as(f64, @floatFromInt(cfg.gen_tokens * cfg.repetitions)) / (gen_ns / 1_000_000_000.0);
    const gen_avg_ms = gen_ns / @as(f64, @floatFromInt(cfg.repetitions)) / 1_000_000.0;

    try writer.print(
        "  {s}: prompt {s:>7} tok/s ({s:>6} ms avg)  decode {d:>7.1} tok/s ({d:>6.2} ms avg)\n",
        .{ label, "  —  ", " —  ", gen_tok_s, gen_avg_ms },
    );
    writer.flush() catch {};

    if (device.getRuntimeProfile()) |rt| {
        const est = profile.estimateProgram(device.program_ops);
        profile.printRuntimeProfile(rt.*, est);
    }

    return .{ .prompt_tok_s = 0, .gen_tok_s = gen_tok_s, .prompt_avg_ms = 0, .gen_avg_ms = gen_avg_ms };
}

fn parseArgOrDefault(args: []const []const u8, idx: usize, default: usize) !usize {
    if (idx >= args.len) return default;
    return try std.fmt.parseInt(usize, args[idx], 10);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    const stdout_file = std.Io.File.stdout();
    var stdout_buf: [4096]u8 = undefined;
    var stdout = stdout_file.writer(io, &stdout_buf);

    const cfg = BenchConfig{
        .model_path = if (args.len > 1) args[1] else "data/smollm/model.safetensors",
        .prompt_tokens = try parseArgOrDefault(args, 2, 4),
        .gen_tokens = try parseArgOrDefault(args, 3, 200),
        .repetitions = try parseArgOrDefault(args, 4, 3),
    };
    const model_is_gguf = isGGUF(cfg.model_path);

    try stdout.interface.print("\nSmolLM LLaMA Benchmark — zgml\n", .{});
    try stdout.interface.print("================================\n", .{});
    try stdout.interface.print("  model={s}\n", .{cfg.model_path});
    try stdout.interface.print("  prompt={d}, gen={d}, reps={d}\n\n", .{ cfg.prompt_tokens, cfg.gen_tokens, cfg.repetitions });
    stdout.interface.flush() catch {};

    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    _ = try runVariant(if (model_is_gguf) "default gguf    " else "default f32      ", null, false, false, cfg, &stdout.interface, io, alloc);

    var cpu_backend = CpuBackend{};
    _ = try runVariant(if (model_is_gguf) "cpu-backend gguf" else "cpu-backend f32  ", cpu_backend.backend(), false, false, cfg, &stdout.interface, io, alloc);
    if (!model_is_gguf) {
        _ = try runVariant("default int8     ", null, true, false, cfg, &stdout.interface, io, alloc);
        _ = try runVariant("cpu-backend i8   ", cpu_backend.backend(), true, false, cfg, &stdout.interface, io, alloc);
        _ = try runVariant("i8 + kv-i8       ", null, true, true, cfg, &stdout.interface, io, alloc);
        _ = try runVariant("kv-i8 only       ", null, false, true, cfg, &stdout.interface, io, alloc);
    } else {
        _ = try runVariant("gguf + kv-i8    ", null, false, true, cfg, &stdout.interface, io, alloc);
    }

    if (is_macos) metal: {
        var metal_be = MetalBackend.init() catch |err| {
            try stdout.interface.print("  metal init failed: {}\n", .{err});
            break :metal;
        };
        defer metal_be.deinit();
        _ = try runVariant(if (model_is_gguf) "metal gguf      " else "metal f32        ", metal_be.backend(), false, false, cfg, &stdout.interface, io, alloc);
        if (!model_is_gguf) _ = try runVariant("metal int8       ", metal_be.backend(), true, false, cfg, &stdout.interface, io, alloc);
        _ = try runDeviceVariant(if (model_is_gguf) "metal device q  " else "metal device f16", metal_be.backend(), cfg, &stdout.interface, io, alloc);
    }

    if (have_wgpu) {
        var wgpu_be = WgpuBackend.init() catch |err| {
            try stdout.interface.print("  wgpu init failed: {}\n", .{err});
            return;
        };
        defer wgpu_be.deinit();
        _ = try runVariant(if (model_is_gguf) "wgpu gguf       " else "wgpu f32         ", wgpu_be.backend(), false, false, cfg, &stdout.interface, io, alloc);
        if (!model_is_gguf) _ = try runVariant("wgpu int8        ", wgpu_be.backend(), true, false, cfg, &stdout.interface, io, alloc);
        _ = try runDeviceVariant(if (model_is_gguf) "wgpu device q   " else "wgpu device f16 ", wgpu_be.backend(), cfg, &stdout.interface, io, alloc);
    }

    try stdout.interface.writeByte('\n');
    stdout.interface.flush() catch {};
}
