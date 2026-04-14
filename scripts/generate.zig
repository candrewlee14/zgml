//! Standalone text generation binary.
//!
//! Demonstrates zgml's complete inference pipeline:
//! 1. Initialize a GPT model (comptime-known architecture)
//! 2. Load trained weights from checkpoint
//! 3. Generate text token by token
//!
//! Build: zig build generate
//! Run:   ./zig-out/bin/generate model.bin "The "

const std = @import("std");
const zgml = @import("zgml");
const Tensor = zgml.Tensor;
const ComputeGraph = zgml.ComputeGraph;
const checkpoint = zgml.checkpoint;

// Model architecture (must match trained model)
const config = zgml.models.GPTConfig{
    .vocab_size = 256, // byte-level
    .d_model = 64,
    .n_heads = 4,
    .d_ff = 256,
    .n_layers = 4,
    .max_seq_len = 128,
};

const GPT = zgml.models.GPT(f32, config);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    const stderr_file = std.fs.File.stderr();
    var stderr_buf: [1024]u8 = undefined;
    var stderr = stderr_file.writer(&stderr_buf);
    if (args.len < 2) {
        try stderr.interface.print("Usage: {s} <checkpoint.bin> [prompt]\n\n", .{args[0]});
        try stderr.interface.print("Architecture: vocab={}, dim={}, heads={}, layers={}\n", .{
            config.vocab_size, config.d_model, config.n_heads, config.n_layers,
        });
        stderr.interface.flush() catch {};
        return;
    }

    const ckpt_path = args[1];
    const prompt = if (args.len > 2) args[2] else "The ";

    const stdout_file = std.fs.File.stdout();
    var stdout_buf: [4096]u8 = undefined;
    var stdout = stdout_file.writer(&stdout_buf);
    try stdout.interface.writeAll(prompt);
    stdout.interface.flush() catch {};

    // Tokenize: bytes → token IDs
    var tokens = std.ArrayList(usize){};
    defer tokens.deinit(alloc);
    for (prompt) |byte| try tokens.append(alloc,byte);

    // Generate
    const max_new = 200;
    for (0..max_new) |_| {
        // Fresh graph for each forward pass (no KV cache yet)
        var g = ComputeGraph(f32).init(alloc);
        defer g.deinit();
        const a = g.allocator();

        const model = try GPT.init(a);

        // Load weights
        const params = model.params();
        checkpoint.load(f32, &params, ckpt_path) catch |err| {
            try stderr.interface.print("Error loading '{s}': {}\n", .{ ckpt_path, err });
            stderr.interface.flush() catch {};
            return;
        };

        // Build input indices tensor
        const seq_len = @min(tokens.items.len, config.max_seq_len);
        const input_tokens = tokens.items[tokens.items.len - seq_len ..];
        const indices = try Tensor(f32).initIndexVectorCopy(a, input_tokens);

        // Forward
        const logits = model.forward(indices);
        try g.infer(logits);

        // Greedy argmax on last position
        const vs = config.vocab_size;
        const last_start = (seq_len - 1) * vs;
        const last_logits = logits.data[last_start..][0..vs];
        var best: usize = 0;
        var best_val = last_logits[0];
        for (last_logits[1..], 1..) |v, i| {
            if (v > best_val) { best_val = v; best = i; }
        }

        // Output and append
        if (best < 128) {
            try stdout.interface.writeByte(@intCast(best));
            stdout.interface.flush() catch {};
        }
        try tokens.append(alloc, best);

        // Stop at newline or EOS
        if (best == '\n' or best == 0) break;
    }
    try stdout.interface.writeByte('\n');
    stdout.interface.flush() catch {};
}
