//! Inference benchmarks for InferenceSession.
//!
//! Measures tok/s and memory for f32 and quantized (int8) inference
//! across model sizes.  Always built with ReleaseFast.
//!
//! Run: zig build bench-inference

const std = @import("std");
const zgml = @import("zgml");
const GPTConfig = zgml.models.GPTConfig;

fn runBenchmark(
    comptime name: []const u8,
    comptime config: GPTConfig,
    n_tokens: usize,
    quantized: bool,
    writer: anytype,
) !void {
    const Session = zgml.inference.InferenceSession(f32, config);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    var session = try Session.init(alloc);
    defer session.deinit();

    if (quantized) try session.quantize();

    // Warm up (1 token).
    _ = try session.step(0);
    session.reset();

    const t_start = std.time.nanoTimestamp();
    for (0..n_tokens) |i| {
        _ = try session.step(i % config.vocab_size);
    }
    const t_end = std.time.nanoTimestamp();

    const elapsed_ns: f64 = @floatFromInt(t_end - t_start);
    const elapsed_ms = elapsed_ns / 1_000_000.0;
    const tok_per_sec = @as(f64, @floatFromInt(n_tokens)) / (elapsed_ns / 1_000_000_000.0);
    const us_per_tok = elapsed_ns / 1000.0 / @as(f64, @floatFromInt(n_tokens));

    // Count weight params for memory estimate.
    var weight_bytes: usize = 0;
    for (session.model.params()) |p| weight_bytes += p.nElems() * @sizeOf(f32);
    const weight_mb: f64 = @as(f64, @floatFromInt(weight_bytes)) / (1024.0 * 1024.0);

    const quant_label = if (quantized) "int8" else "f32 ";
    try writer.print("  {s} {s}: {d:>8.1} tok/s  {d:>7.1} us/tok  {d:>6.1}ms total  weights {d:.2}MB\n", .{
        name,
        quant_label,
        tok_per_sec,
        us_per_tok,
        elapsed_ms,
        if (quantized) weight_mb / 4.0 else weight_mb,
    });
}

pub fn main() !void {
    const stdout_file = std.fs.File.stdout();
    var stdout_buf: [4096]u8 = undefined;
    var stdout = stdout_file.writer(&stdout_buf);

    // Small model: like the training config.
    const small = GPTConfig{
        .vocab_size = 256,
        .d_model = 64,
        .n_heads = 4,
        .d_ff = 256,
        .n_layers = 4,
        .max_seq_len = 128,
    };

    // Medium model: closer to TinyStories.
    const medium = GPTConfig{
        .vocab_size = 512,
        .d_model = 128,
        .n_heads = 8,
        .d_ff = 512,
        .n_layers = 6,
        .max_seq_len = 256,
    };

    const n_tokens = 64;

    try stdout.interface.writeAll("=== Inference Benchmarks ===\n\n");
    try stdout.interface.writeAll("Small (d=64, L=4, H=4):\n");
    try runBenchmark("small ", small, n_tokens, false, &stdout.interface);
    try runBenchmark("small ", small, n_tokens, true, &stdout.interface);

    try stdout.interface.writeAll("\nMedium (d=128, L=6, H=8):\n");
    try runBenchmark("medium", medium, n_tokens, false, &stdout.interface);
    try runBenchmark("medium", medium, n_tokens, true, &stdout.interface);

    try stdout.interface.writeAll("\n");
    stdout.interface.flush() catch {};
}
