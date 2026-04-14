//! Generate text from a pretrained HuggingFace GPT-Neo model.
//!
//! Loads weights from safetensors, tokenizes a prompt with GPT-2 BPE,
//! and generates tokens using KV-cached inference (O(1) per new token).
//!
//! Build: zig build generate-pretrained
//! Run:   ./zig-out/bin/generate-pretrained data/tinystories/model.safetensors \
//!            data/tinystories/vocab.json data/tinystories/merges.txt "Once upon a time"

const std = @import("std");
const zgml = @import("zgml");
const Tensor = zgml.Tensor;
const ComputeGraph = zgml.ComputeGraph;

const config = zgml.models.GPTConfig{
    .vocab_size = 50257,
    .d_model = 64,
    .n_heads = 16,
    .d_ff = 256,
    .n_layers = 8,
    .max_seq_len = 512,
    .learnable_pos_embed = true,
    .learnable_ln = true,
    .attn_bias = true,
    .tied_lm_head = true,
};

const Model = zgml.models.GPT(f32, config);
const d_head = config.d_model / config.n_heads;
const n_kv = config.n_layers * config.n_heads;

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

    if (args.len < 4) {
        try stderr.interface.print("Usage: {s} <model.safetensors> <vocab.json> <merges.txt> [prompt]\n", .{args[0]});
        stderr.interface.flush() catch {};
        return;
    }

    const model_path = args[1];
    const vocab_path = args[2];
    const merges_path = args[3];
    const prompt = if (args.len > 4) args[4] else "Once upon a time";

    // Load tokenizer
    var tok = try zgml.tokenizer.GPT2Tokenizer.init(alloc, vocab_path, merges_path);
    defer tok.deinit();

    const prompt_ids = try tok.encode(alloc, prompt);
    defer alloc.free(prompt_ids);
    if (prompt_ids.len == 0) {
        try stderr.interface.print("Error: empty prompt\n", .{});
        stderr.interface.flush() catch {};
        return;
    }

    // Load model weights
    var sf = try zgml.safetensors.SafetensorsFile.open(alloc, model_path);
    defer sf.deinit();

    var load_g = ComputeGraph(f32).init(alloc);
    const load_model = try Model.init(load_g.allocator());
    try zgml.models.gpt_loader.loadGPTNeo(f32, config, &load_model, &sf);

    const load_params = load_model.params();
    var weight_data: [load_params.len][]f32 = undefined;
    for (load_params, 0..) |p, i| {
        weight_data[i] = try alloc.alloc(f32, p.nElems());
        @memcpy(weight_data[i], p.data);
    }
    load_g.deinit();
    defer for (&weight_data) |d| alloc.free(d);

    try stderr.interface.print("Loaded {s} ({d} tokens)\n", .{ prompt, prompt_ids.len });
    stderr.interface.flush() catch {};

    // Persistent KV cache buffers
    var k_bufs: [n_kv][]f32 = undefined;
    var v_bufs: [n_kv][]f32 = undefined;
    for (0..n_kv) |i| {
        k_bufs[i] = try alloc.alloc(f32, d_head * config.max_seq_len);
        v_bufs[i] = try alloc.alloc(f32, d_head * config.max_seq_len);
        @memset(k_bufs[i], 0);
        @memset(v_bufs[i], 0);
    }
    defer for (0..n_kv) |i| {
        alloc.free(k_bufs[i]);
        alloc.free(v_bufs[i]);
    };

    try stdout.interface.writeAll(prompt);
    stdout.interface.flush() catch {};

    var pos: usize = 0;
    var next_token: usize = @intCast(prompt_ids[0]);
    var prompt_idx: usize = 0;

    for (0..prompt_ids.len + 200) |_| {
        var g = ComputeGraph(f32).init(alloc);
        defer g.deinit();
        const a = g.allocator();

        const model = try Model.init(a);
        for (model.params(), weight_data) |param, wd| @memcpy(param.data, wd);

        var k_caches: [config.n_layers][config.n_heads]*Tensor(f32) = undefined;
        var v_caches: [config.n_layers][config.n_heads]*Tensor(f32) = undefined;
        for (0..config.n_layers) |l| {
            for (0..config.n_heads) |h| {
                const idx = l * config.n_heads + h;
                k_caches[l][h] = try Tensor(f32).init(a, &.{ d_head, config.max_seq_len });
                v_caches[l][h] = try Tensor(f32).init(a, &.{ d_head, config.max_seq_len });
                @memcpy(k_caches[l][h].data, k_bufs[idx]);
                @memcpy(v_caches[l][h].data, v_bufs[idx]);
            }
        }

        const logits = model.forwardCached(a, next_token, pos, k_caches, v_caches);
        try g.infer(logits);

        for (0..config.n_layers) |l| {
            for (0..config.n_heads) |h| {
                const idx = l * config.n_heads + h;
                @memcpy(k_bufs[idx], k_caches[l][h].data);
                @memcpy(v_bufs[idx], v_caches[l][h].data);
            }
        }
        pos += 1;

        // Prefill prompt
        if (prompt_idx + 1 < prompt_ids.len) {
            prompt_idx += 1;
            next_token = @intCast(prompt_ids[prompt_idx]);
            continue;
        }

        // Greedy decode
        var best: u32 = 0;
        var best_val = logits.data[0];
        for (1..config.vocab_size) |i| {
            if (logits.data[i] > best_val) {
                best_val = logits.data[i];
                best = @intCast(i);
            }
        }

        const token_text = try tok.decode(alloc, &.{best});
        defer alloc.free(token_text);
        try stdout.interface.writeAll(token_text);
        stdout.interface.flush() catch {};

        if (best == 50256 or pos >= config.max_seq_len) break;
        next_token = @intCast(best);
    }

    try stdout.interface.writeByte('\n');
    stdout.interface.flush() catch {};
}
