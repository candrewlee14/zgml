//! Text generation with KV cache.
//!
//! Loads model weights once, then generates tokens one at a time using
//! cached key/value tensors — O(1) per token instead of reprocessing
//! the full sequence.
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
const d_head = config.d_model / config.n_heads;

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

    // Load weights into temporary graph, copy to owned buffers
    var load_g = ComputeGraph(f32).init(alloc);
    const load_model = try GPT.init(load_g.allocator());
    const load_params = load_model.params();
    checkpoint.load(f32, &load_params, ckpt_path) catch |err| {
        try stderr.interface.print("Error loading '{s}': {}\n", .{ ckpt_path, err });
        stderr.interface.flush() catch {};
        return;
    };
    var weight_data: [load_params.len][]f32 = undefined;
    for (load_params, 0..) |p, i| {
        weight_data[i] = try alloc.alloc(f32, p.nElems());
        @memcpy(weight_data[i], p.data);
    }
    load_g.deinit();
    defer for (&weight_data) |d| alloc.free(d);

    try stdout.interface.writeAll(prompt);
    stdout.interface.flush() catch {};

    // Generate token by token using a fresh graph per step.
    // Model weights are copied in, KV cache data persists in alloc-owned buffers.
    var kv_bufs: [2 * config.n_layers * config.n_heads][]f32 = undefined;
    for (&kv_bufs) |*buf| {
        buf.* = try alloc.alloc(f32, d_head * config.max_seq_len);
        @memset(buf.*, 0);
    }
    defer for (&kv_bufs) |buf| alloc.free(buf);

    var pos: usize = 0;
    var next_token: usize = prompt[0]; // start with first prompt byte
    var in_prompt = true;
    var prompt_idx: usize = 0;

    const max_tokens = prompt.len + 200;
    for (0..max_tokens) |_| {
        var g = ComputeGraph(f32).init(alloc);
        defer g.deinit();
        const a = g.allocator();

        // Build model and copy weights
        const model = try GPT.init(a);
        const params = model.params();
        for (params, weight_data) |p, wd| @memcpy(p.data, wd);

        // Build KV caches and copy persistent data in
        var k_caches: [config.n_layers][config.n_heads]*Tensor(f32) = undefined;
        var v_caches: [config.n_layers][config.n_heads]*Tensor(f32) = undefined;
        for (0..config.n_layers) |l| {
            for (0..config.n_heads) |h| {
                const idx = l * config.n_heads + h;
                k_caches[l][h] = try Tensor(f32).init(a, &.{ d_head, config.max_seq_len });
                v_caches[l][h] = try Tensor(f32).init(a, &.{ d_head, config.max_seq_len });
                @memcpy(k_caches[l][h].data, kv_bufs[idx]);
                @memcpy(v_caches[l][h].data, kv_bufs[config.n_layers * config.n_heads + idx]);
            }
        }

        const logits = model.forwardCached(a, next_token, pos, k_caches, v_caches);
        try g.infer(logits);

        // Copy KV cache data back out
        for (0..config.n_layers) |l| {
            for (0..config.n_heads) |h| {
                const idx = l * config.n_heads + h;
                @memcpy(kv_bufs[idx], k_caches[l][h].data);
                @memcpy(kv_bufs[config.n_layers * config.n_heads + idx], v_caches[l][h].data);
            }
        }
        pos += 1;

        // During prompt prefill, advance through prompt tokens
        if (in_prompt) {
            prompt_idx += 1;
            if (prompt_idx < prompt.len) {
                next_token = prompt[prompt_idx];
                continue;
            }
            in_prompt = false;
        }

        // Greedy argmax
        const vs = config.vocab_size;
        var best: usize = 0;
        var best_val = logits.data[0];
        for (logits.data[1..vs], 1..) |v, i| {
            if (v > best_val) { best_val = v; best = i; }
        }

        if (best < 128) {
            try stdout.interface.writeByte(@intCast(best));
            stdout.interface.flush() catch {};
        }
        if (best == '\n' or best == 0 or pos >= config.max_seq_len) break;
        next_token = best;
    }
    try stdout.interface.writeByte('\n');
    stdout.interface.flush() catch {};
}
