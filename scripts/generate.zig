//! Standalone text generation binary.
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

    // --- Load weights once into persistent storage ---
    // Create a temporary graph just to get the param shapes, then copy data out.
    const n_params = GPT.params(&undefined).len; // comptime-known
    _ = n_params;

    // Load weights: create a graph, init model, load checkpoint, steal the data pointers.
    var weight_graph = ComputeGraph(f32).init(alloc);
    const weight_model = try GPT.init(weight_graph.allocator());
    const weight_params = weight_model.params();
    checkpoint.load(f32, &weight_params, ckpt_path) catch |err| {
        try stderr.interface.print("Error: {}\n", .{err});
        stderr.interface.flush() catch {};
        return;
    };

    // Copy weight data to GPA-owned buffers so we can free the graph
    var weight_data: [weight_params.len][]f32 = undefined;
    for (weight_params, 0..) |p, i| {
        weight_data[i] = try alloc.alloc(f32, p.nElems());
        @memcpy(weight_data[i], p.data);
    }
    weight_graph.deinit(); // free all tensor metadata, but our copies survive

    defer for (weight_data) |d| alloc.free(d);

    try stderr.interface.print("Loaded weights from '{s}'\n", .{ckpt_path});
    stderr.interface.flush() catch {};

    // --- Tokenize prompt ---
    var tokens = std.ArrayList(usize){};
    defer tokens.deinit(alloc);
    for (prompt) |byte| try tokens.append(alloc, byte);

    try stdout.interface.writeAll(prompt);
    stdout.interface.flush() catch {};

    // --- Generate tokens ---
    const max_new = 200;
    for (0..max_new) |_| {
        // Lightweight graph per step (no disk I/O)
        var g = ComputeGraph(f32).init(alloc);
        defer g.deinit();
        const a = g.allocator();

        const model = try GPT.init(a);

        // Wire in the pre-loaded weight data (pointer swap, no copy)
        const params = model.params();
        for (params, 0..) |p, i| {
            @memcpy(p.data, weight_data[i]);
        }

        // Build input
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
            if (v > best_val) {
                best_val = v;
                best = i;
            }
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
