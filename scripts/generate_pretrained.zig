//! Generate text from a pretrained HuggingFace GPT-Neo model.
//!
//! Loads weights from safetensors into an InferenceSession, tokenizes
//! a prompt with GPT-2 BPE, and generates tokens with zero per-step
//! allocation.
//!
//! Build: zig build generate-pretrained
//! Run:   ./zig-out/bin/generate-pretrained data/tinystories/model.safetensors \
//!            data/tinystories/vocab.json data/tinystories/merges.txt "Once upon a time"

const std = @import("std");
const zgml = @import("zgml");
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

const Session = zgml.inference.InferenceSession(f32, config);

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    const args = try init.minimal.args.toSlice(init.arena.allocator());

    const stderr_file = std.Io.File.stderr();
    var stderr_buf: [4096]u8 = undefined;
    var stderr = stderr_file.writer(io, &stderr_buf);
    const stdout_file = std.Io.File.stdout();
    var stdout_buf: [4096]u8 = undefined;
    var stdout = stdout_file.writer(io, &stdout_buf);

    if (args.len < 4) {
        try stderr.interface.print("Usage: {s} <model.safetensors> <vocab.json> <merges.txt> [prompt]\n", .{args[0]});
        stderr.interface.flush() catch {};
        return;
    }

    const model_path = args[1];
    const vocab_path = args[2];
    const merges_path = args[3];
    const prompt = if (args.len > 4) args[4] else "Once upon a time";

    // Load tokenizer.
    var tok = try zgml.tokenizer.GPT2Tokenizer.init(alloc, vocab_path, merges_path, io);
    defer tok.deinit();

    const prompt_ids = try tok.encode(alloc, prompt);
    defer alloc.free(prompt_ids);
    if (prompt_ids.len == 0) {
        try stderr.interface.print("Error: empty prompt\n", .{});
        stderr.interface.flush() catch {};
        return;
    }

    // Create session, then load weights into it via safetensors.
    // The loader needs a temporary ComputeGraph to parse the file, but
    // writes directly into the session's persistent model tensors.
    var session = try Session.init(alloc);
    defer session.deinit();

    var sf = try zgml.safetensors.SafetensorsFile.open(alloc, model_path, io);
    defer sf.deinit();
    try zgml.models.gpt_loader.loadGPTNeo(f32, config, &session.model, &sf);

    try stderr.interface.print("Loaded {s} ({d} tokens)\n", .{ prompt, prompt_ids.len });
    stderr.interface.flush() catch {};

    try stdout.interface.writeAll(prompt);
    stdout.interface.flush() catch {};

    var next_token: usize = @intCast(prompt_ids[0]);
    var prompt_idx: usize = 0;
    var gen_tokens: usize = 0;

    const t_start = std.Io.Clock.awake.now(io).nanoseconds;

    for (0..prompt_ids.len + 200) |_| {
        const logits = try session.step(next_token);

        // Prefill prompt.
        if (prompt_idx + 1 < prompt_ids.len) {
            prompt_idx += 1;
            next_token = @intCast(prompt_ids[prompt_idx]);
            continue;
        }

        gen_tokens += 1;

        // Greedy decode.
        var best: u32 = 0;
        var best_val = logits[0];
        for (1..config.vocab_size) |i| {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best = @intCast(i);
            }
        }

        const token_text = try tok.decode(alloc, &.{best});
        defer alloc.free(token_text);
        try stdout.interface.writeAll(token_text);
        stdout.interface.flush() catch {};

        if (best == 50256 or session.position() >= config.max_seq_len) break;
        next_token = @intCast(best);
    }

    const t_end = std.Io.Clock.awake.now(io).nanoseconds;
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
