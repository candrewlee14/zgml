//! Text generation with weight caching.
//!
//! Loads model weights once, rebuilds graph per step but avoids disk I/O.
//! KV cache integration is TODO — currently reprocesses full sequence each step.
//!
//! Build: zig build generate
//! Run:   ./zig-out/bin/generate model.bin "The "

const std = @import("std");
const zgml = @import("zgml");
const Tensor = zgml.Tensor;
const ComputeGraph = zgml.ComputeGraph;
const checkpoint = zgml.checkpoint;

const config = zgml.models.GPTConfig{
    .vocab_size = 256,
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
        try stderr.interface.print("Usage: {s} <checkpoint.bin> [prompt]\n", .{args[0]});
        stderr.interface.flush() catch {};
        return;
    }

    const ckpt_path = args[1];
    const prompt = if (args.len > 2) args[2] else "The ";

    const stdout_file = std.fs.File.stdout();
    var stdout_buf: [4096]u8 = undefined;
    var stdout = stdout_file.writer(&stdout_buf);

    // Load weights once into persistent buffers
    var load_g = ComputeGraph(f32).init(alloc);
    const load_model = try GPT.init(load_g.allocator());
    const load_params = load_model.params();
    checkpoint.load(f32, &load_params, ckpt_path) catch |err| {
        try stderr.interface.print("Error loading '{s}': {}\n", .{ ckpt_path, err });
        stderr.interface.flush() catch {};
        return;
    };

    // Copy weight data to GPA-owned buffers
    var weight_data: [load_params.len][]f32 = undefined;
    for (load_params, 0..) |p, i| {
        weight_data[i] = try alloc.alloc(f32, p.nElems());
        @memcpy(weight_data[i], p.data);
    }
    load_g.deinit();
    defer for (&weight_data) |d| alloc.free(d);

    // Build token sequence
    var tokens = std.ArrayList(usize){};
    defer tokens.deinit(alloc);
    for (prompt) |byte| try tokens.append(alloc, byte);

    try stdout.interface.writeAll(prompt);
    stdout.interface.flush() catch {};

    // Generate
    const max_new = 200;
    for (0..max_new) |_| {
        // Fresh graph per step (weight copy, no disk I/O)
        var g = ComputeGraph(f32).init(alloc);
        defer g.deinit();
        const a = g.allocator();

        const model = try GPT.init(a);
        const params = model.params();
        for (params, weight_data) |p, wd| @memcpy(p.data, wd);

        const seq_len = @min(tokens.items.len, config.max_seq_len);
        const indices = try Tensor(f32).initIndexVectorCopy(a, tokens.items[tokens.items.len - seq_len ..]);
        const logits = model.forward(indices);
        try g.infer(logits);

        // Greedy argmax on last position
        const vs = config.vocab_size;
        const last_logits = logits.data[(seq_len - 1) * vs ..][0..vs];
        var best: usize = 0;
        var best_val = last_logits[0];
        for (last_logits[1..], 1..) |v, i| {
            if (v > best_val) { best_val = v; best = i; }
        }

        if (best < 128) {
            try stdout.interface.writeByte(@intCast(best));
            stdout.interface.flush() catch {};
        }
        try tokens.append(alloc, best);
        if (best == '\n' or best == 0) break;
    }
    try stdout.interface.writeByte('\n');
    stdout.interface.flush() catch {};
}
