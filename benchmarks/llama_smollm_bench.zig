//! SmolLM LLaMA inference benchmark for zgml.
//!
//! Measures prompt and decode throughput on the local SmolLM checkpoint using
//! the real `LlamaInferenceSession` path, without tokenizer or stdout noise.
//!
//! Run:
//!   zig build bench-llama-smollm
//!   ./zig-out/bin/bench-llama-smollm [model.safetensors] [prompt_tokens] [gen_tokens] [repetitions]

const std = @import("std");
const zgml = @import("zgml");

const CpuBackend = zgml.backend_cpu.CpuBackend;
const Backend = zgml.backend.Backend;

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
    var sf = try zgml.safetensors.SafetensorsFile.open(alloc, model_path, io);
    defer sf.deinit();
    try zgml.models.llama_loader.loadLlama(f32, config, &session.model, &sf);
}

fn runVariant(
    label: []const u8,
    maybe_backend: ?Backend,
    quantized: bool,
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
    if (quantized) try session.quantize();

    // Warm up the real decode path once to settle kernel/backend selection.
    _ = try session.step(0);
    session.reset();

    var prompt_total_ns: u128 = 0;
    var gen_total_ns: u128 = 0;

    for (0..cfg.repetitions) |_| {
        session.reset();
        const prompt_start = std.Io.Clock.awake.now(io).nanoseconds;
        for (0..cfg.prompt_tokens) |i| {
            _ = try session.step((i + 1) % config.vocab_size);
        }
        const prompt_end = std.Io.Clock.awake.now(io).nanoseconds;
        prompt_total_ns += @intCast(prompt_end - prompt_start);

        session.reset();
        for (0..cfg.prompt_tokens) |i| {
            _ = try session.step((i + 1) % config.vocab_size);
        }

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

    return .{
        .prompt_tok_s = prompt_tok_s,
        .gen_tok_s = gen_tok_s,
        .prompt_avg_ms = prompt_avg_ms,
        .gen_avg_ms = gen_avg_ms,
    };
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

    try stdout.interface.print("\nSmolLM LLaMA Benchmark — zgml\n", .{});
    try stdout.interface.print("================================\n", .{});
    try stdout.interface.print("  model={s}\n", .{cfg.model_path});
    try stdout.interface.print("  prompt={d}, gen={d}, reps={d}\n\n", .{ cfg.prompt_tokens, cfg.gen_tokens, cfg.repetitions });

    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    _ = try runVariant("default f32    ", null, false, cfg, &stdout.interface, io, alloc);

    var cpu_backend = CpuBackend{};
    _ = try runVariant("cpu-backend f32", cpu_backend.backend(), false, cfg, &stdout.interface, io, alloc);
    _ = try runVariant("default int8   ", null, true, cfg, &stdout.interface, io, alloc);
    _ = try runVariant("cpu-backend i8 ", cpu_backend.backend(), true, cfg, &stdout.interface, io, alloc);

    try stdout.interface.writeByte('\n');
    stdout.interface.flush() catch {};
}
