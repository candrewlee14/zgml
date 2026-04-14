//! Metal inference benchmark: compare CPU vs Metal backend inference.
//!
//! Measures tok/s and us/tok for CPU f32, CPU int8, and Metal int8
//! across small and medium GPT configs.
//!
//! Run with: zig build bench-metal-inference

const std = @import("std");
const zgml = @import("zgml");
const GPTConfig = zgml.models.GPTConfig;
const MetalBackend = zgml.backend_metal.MetalBackend;
const Backend = zgml.backend.Backend;

const WARMUP_TOKENS = 1;
const TIMED_TOKENS = 32;

fn runBenchmark(
    comptime label: []const u8,
    comptime config: GPTConfig,
    quantized: bool,
    backend: ?Backend,
    writer: anytype,
) !void {
    const Session = zgml.inference.InferenceSession(f32, config);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    var session = try Session.initWithBackend(alloc, backend);
    defer session.deinit();

    if (quantized) try session.quantize();

    // Warm up.
    for (0..WARMUP_TOKENS) |i| {
        _ = try session.step(i % config.vocab_size);
    }
    session.reset();

    const t_start = std.time.nanoTimestamp();
    for (0..TIMED_TOKENS) |i| {
        _ = try session.step(i % config.vocab_size);
    }
    const t_end = std.time.nanoTimestamp();

    const elapsed_ns: f64 = @floatFromInt(t_end - t_start);
    const elapsed_ms = elapsed_ns / 1_000_000.0;
    const tok_per_sec = @as(f64, @floatFromInt(TIMED_TOKENS)) / (elapsed_ns / 1_000_000_000.0);
    const us_per_tok = elapsed_ns / 1000.0 / @as(f64, @floatFromInt(TIMED_TOKENS));

    try writer.print("  {s}: {d:>8.1} tok/s  {d:>7.1} us/tok  {d:>6.1}ms total\n", .{
        label,
        tok_per_sec,
        us_per_tok,
        elapsed_ms,
    });
}

pub fn main() !void {
    const stdout_file = std.fs.File.stdout();
    var stdout_buf: [4096]u8 = undefined;
    var w = stdout_file.writer(&stdout_buf);

    try w.interface.print("\nMetal Inference Benchmark — CPU vs Metal\n", .{});
    try w.interface.print("==========================================\n", .{});
    try w.interface.print("  warmup={d}, timed={d} tokens\n\n", .{ WARMUP_TOKENS, TIMED_TOKENS });

    // Try to init Metal backend.
    var maybe_metal: ?MetalBackend = MetalBackend.init() catch |err| blk: {
        try w.interface.print("  Metal init failed: {} — skipping Metal tests\n\n", .{err});
        break :blk null;
    };
    defer if (maybe_metal) |*mb| mb.deinit();

    const metal: ?Backend = if (maybe_metal) |*mb| mb.backend() else null;

    const small = GPTConfig{
        .vocab_size = 256,
        .d_model = 64,
        .n_heads = 4,
        .d_ff = 256,
        .n_layers = 4,
        .max_seq_len = 128,
    };

    const medium = GPTConfig{
        .vocab_size = 512,
        .d_model = 128,
        .n_heads = 8,
        .d_ff = 512,
        .n_layers = 6,
        .max_seq_len = 256,
    };

    // --- Small ---
    try w.interface.print("Small (d=64, L=4, H=4):\n", .{});
    try runBenchmark("CPU f32   ", small, false, null, &w.interface);
    try runBenchmark("CPU int8  ", small, true, null, &w.interface);
    if (metal) |be| {
        try runBenchmark("Metal int8", small, true, be, &w.interface);
    }

    // --- Medium ---
    try w.interface.print("\nMedium (d=128, L=6, H=8):\n", .{});
    try runBenchmark("CPU f32   ", medium, false, null, &w.interface);
    try runBenchmark("CPU int8  ", medium, true, null, &w.interface);
    if (metal) |be| {
        try runBenchmark("Metal int8", medium, true, be, &w.interface);
    }

    try w.interface.print("\n", .{});
    w.interface.flush() catch {};
}
