//! Train a tiny byte-level GPT on a hardcoded text snippet.
//!
//! Produces a checkpoint for scripts/generate.zig.
//! Build: zig build train-tiny
//! Run:   ./zig-out/bin/train-tiny

const std = @import("std");
const zgml = @import("zgml");
const Tensor = zgml.Tensor;
const ComputeGraph = zgml.ComputeGraph;
const checkpoint = zgml.checkpoint;
const loss_lib = zgml.loss;

const config = zgml.models.GPTConfig{
    .vocab_size = 256,
    .d_model = 64,
    .n_heads = 4,
    .d_ff = 256,
    .n_layers = 4,
    .max_seq_len = 128,
};

const GPT = zgml.models.GPT(f32, config);

const seq_len = 16;

const corpus = "The cat sat on the mat. The dog ran in the park. A bird flew over the tree. The sun was warm and bright. ";

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const stdout_file = std.Io.File.stdout();
    var stdout_buf: [4096]u8 = undefined;
    var w = stdout_file.writer(io, &stdout_buf);

    try w.interface.print("Training tiny GPT (dim={}, layers={}, seq={})\n", .{
        config.d_model, config.n_layers, seq_len,
    });
    w.interface.flush() catch {};

    const alloc = init.gpa;

    var g = ComputeGraph(f32).init(alloc);
    defer g.deinit();
    const a = g.allocator();

    // Build model
    const model = try GPT.init(a);

    // Fixed-size input/target tensors (reused each iteration)
    const input_slot = try Tensor(f32).init(a, &.{seq_len});
    const target_slot = try Tensor(f32).init(a, &.{seq_len});

    // Forward: logits = model(input) → [vocab_size, seq_len]
    const logits = model.forward(input_slot);
    const loss_node = loss_lib.crossEntropy(f32, logits, target_slot);

    // Build graph once
    try g.buildForward(loss_node);
    try g.buildBackward(false);
    try g.fusionPass();

    const params = model.params();
    const n_samples = corpus.len - seq_len - 1;
    const lr: f32 = 0.003;
    const n_epochs = 500;

    try w.interface.print("Corpus: {} bytes, {} samples\n\n", .{ corpus.len, n_samples });
    w.interface.flush() catch {};

    for (0..n_epochs) |epoch| {
        var total_loss: f32 = 0;
        for (0..n_samples) |s| {
            // Fill input: bytes s..s+seq_len as token IDs
            for (0..seq_len) |i| {
                input_slot.data[i] = @floatFromInt(corpus[s + i]);
            }
            // Target: bytes s+1..s+seq_len+1 (next token at each position)
            for (0..seq_len) |i| {
                target_slot.data[i] = @floatFromInt(corpus[s + i + 1]);
            }

            // Forward + backward
            g.reset();
            g.resetGrads();
            if (loss_node.grad) |grad| _ = grad.setAllScalar(1);
            g.compute();
            total_loss += loss_node.data[0];

            // SGD step
            for (params) |p| {
                if (p.gradOrNull()) |grad| {
                    for (p.data, grad.data) |*wv, gv| {
                        wv.* -= lr * gv;
                    }
                }
            }
        }

        const avg_loss = total_loss / @as(f32, @floatFromInt(n_samples));
        if (epoch % 50 == 0 or epoch == n_epochs - 1) {
            try w.interface.print("epoch {d:>3}: loss={d:.4}\n", .{ epoch + 1, avg_loss });
            w.interface.flush() catch {};
        }
    }

    // Save
    const path = "tiny_gpt.bin";
    try checkpoint.save(f32, &params, path);
    try w.interface.print("\nSaved to '{s}'. Generate with:\n  ./zig-out/bin/generate {s} \"The \"\n", .{ path, path });
    w.interface.flush() catch {};
}
