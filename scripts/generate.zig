//! Text generation with frozen inference plan.
//!
//! Loads model weights once into an InferenceSession, then generates
//! tokens with zero per-step allocation — no graph rebuild, no weight
//! copies, no KV cache memcpy.
//!
//! Build: zig build generate
//! Run:   ./zig-out/bin/generate model.bin "The "

const std = @import("std");
const zgml = @import("zgml");
const checkpoint = zgml.checkpoint;

const config = zgml.models.GPTConfig{
    .vocab_size = 256,
    .d_model = 64,
    .n_heads = 4,
    .d_ff = 256,
    .n_layers = 4,
    .max_seq_len = 128,
};

const Session = zgml.inference.InferenceSession(f32, config);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    const stderr_file = std.fs.File.stderr();
    var stderr_buf: [4096]u8 = undefined;
    var stderr = stderr_file.writer(&stderr_buf);
    const stdout_file = std.fs.File.stdout();
    var stdout_buf: [4096]u8 = undefined;
    var stdout = stdout_file.writer(&stdout_buf);

    if (args.len < 2) {
        try stderr.interface.print("Usage: {s} <checkpoint.bin> [prompt]\n", .{args[0]});
        stderr.interface.flush() catch {};
        return;
    }

    const ckpt_path = args[1];
    const prompt = if (args.len > 2) args[2] else "The ";

    // Create session and load weights directly into it.
    var session = try Session.init(alloc);
    defer session.deinit();

    const params = session.model.params();
    checkpoint.load(f32, &params, ckpt_path) catch |err| {
        try stderr.interface.print("Error loading '{s}': {}\n", .{ ckpt_path, err });
        stderr.interface.flush() catch {};
        return;
    };

    try stdout.interface.writeAll(prompt);
    stdout.interface.flush() catch {};

    var next_token: usize = prompt[0];
    var prompt_idx: usize = 0;
    var in_prompt = true;
    var gen_tokens: usize = 0;

    const max_tokens = prompt.len + 200;
    const t_start = std.time.nanoTimestamp();

    for (0..max_tokens) |_| {
        const logits = try session.step(next_token);

        // Prefill prompt
        if (in_prompt) {
            prompt_idx += 1;
            if (prompt_idx < prompt.len) {
                next_token = prompt[prompt_idx];
                continue;
            }
            in_prompt = false;
        }

        gen_tokens += 1;

        // Greedy argmax
        var best: usize = 0;
        var best_val = logits[0];
        for (logits[1..], 1..) |v, i| {
            if (v > best_val) { best_val = v; best = i; }
        }

        if (best < 128) {
            try stdout.interface.writeByte(@intCast(best));
            stdout.interface.flush() catch {};
        }
        if (best == '\n' or best == 0 or session.position() >= config.max_seq_len) break;
        next_token = best;
    }

    const t_end = std.time.nanoTimestamp();
    const elapsed_ms: f64 = @as(f64, @floatFromInt(t_end - t_start)) / 1_000_000.0;
    const total_tokens = session.position();
    const toks_per_sec: f64 = if (elapsed_ms > 0)
        @as(f64, @floatFromInt(total_tokens)) / (elapsed_ms / 1000.0)
    else
        0;

    try stdout.interface.writeByte('\n');
    stdout.interface.flush() catch {};
    try stderr.interface.print(
        "{d} tokens in {d:.1}ms ({d:.1} tok/s, {d} prefill + {d} generated)\n",
        .{ total_tokens, elapsed_ms, toks_per_sec, total_tokens - gen_tokens, gen_tokens },
    );
    stderr.interface.flush() catch {};
}
