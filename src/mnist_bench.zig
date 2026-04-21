//! MNIST CNN training benchmark.
//!
//! Run with: zig build mnist-bench
//! Built with ReleaseFast for meaningful measurements.

const std = @import("std");
const zgml = @import("zgml");
const Tensor = zgml.Tensor;
const IndexTensor = zgml.IndexTensor;
const loss_mod = zgml.loss;
const nn = zgml.nn;
const ConvClassifier = zgml.models.ConvClassifier;
const MnistDataset = zgml.data.mnist.MnistDataset;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    const stdout_file = std.Io.File.stdout();
    var buf: [4096]u8 = undefined;
    var w = stdout_file.writer(io, &buf);

    try w.interface.print("\nMNIST CNN Training Benchmark\n", .{});
    try w.interface.print("============================\n\n", .{});

    // Load data
    try w.interface.print("Loading MNIST data...\n", .{});
    w.interface.flush() catch {};

    var train = try MnistDataset.load(alloc, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", io);
    defer train.deinit(alloc);
    var test_data = try MnistDataset.load(alloc, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", io);
    defer test_data.deinit(alloc);

    try w.interface.print("  Train: {} images, Test: {} images\n", .{ train.n_samples, test_data.n_samples });
    try w.interface.print("  Image size: {}x{}\n\n", .{ train.cols, train.rows });

    const batch_size: usize = 32;
    const n_epochs: usize = 10;
    const train_batches = train.n_samples / batch_size;
    const pixels_per_image = train.pixelsPerImage();

    try w.interface.print("Architecture: Conv(5x5, 1->8) -> ReLU -> MaxPool(2x2) -> FC(1152->10)\n", .{});
    try w.interface.print("Optimizer: Adam (lr=1e-3)\n", .{});
    try w.interface.print("Batch size: {}, Epochs: {}\n\n", .{ batch_size, n_epochs });
    w.interface.flush() catch {};

    // Simple model — fast and effective for MNIST
    var model = try ConvClassifier(f32).build(alloc, 28, 28, 5, 8, 10, batch_size);
    defer model.deinit();
    try model.g.fusionPass();

    // Adam optimizer — converges faster than SGD
    const p = model.params();
    var sgd = try zgml.optim.adam.Adam(f32).init(alloc, p, .{ .lr = 1e-3 });
    defer sgd.deinit();

    // Batch index array for shuffling
    const batch_indices = try alloc.alloc(usize, train_batches);
    defer alloc.free(batch_indices);
    for (batch_indices, 0..) |*idx, i| idx.* = i;
    var prng = std.Random.DefaultPrng.init(42);

    // Training loop
    const total_t0 = std.Io.Clock.awake.now(io).nanoseconds;

    for (0..n_epochs) |epoch| {
        const epoch_t0 = std.Io.Clock.awake.now(io).nanoseconds;
        var epoch_loss: f64 = 0;
        var epoch_correct: usize = 0;
        var preds: [32]usize = undefined;

        // Shuffle batch order each epoch
        prng.random().shuffle(usize, batch_indices);

        for (0..train_batches) |step| {
            const batch_idx = batch_indices[step];

            // Copy batch data
            const img_off = batch_idx * batch_size * pixels_per_image;
            @memcpy(model.xs_batch.data, train.images[img_off..][0 .. batch_size * pixels_per_image]);
            const lbl_off = batch_idx * batch_size;
            @memcpy(model.ys_batch.data, train.labels[lbl_off..][0..batch_size]);

            model.g.reset();
            model.g.resetGrads();
            if (model.loss.grad) |grad| _ = grad.setAllScalar(1);
            model.g.compute();

            sgd.step();

            epoch_loss += model.loss.data[0];

            // Track accuracy
            model.predict(&preds);
            for (0..batch_size) |i| {
                if (preds[i] == @as(usize, @intFromFloat(train.labels[lbl_off + i]))) {
                    epoch_correct += 1;
                }
            }

            if ((step + 1) % 100 == 0) {
                const avg_loss = epoch_loss / @as(f64, @floatFromInt(step + 1));
                const acc = @as(f64, @floatFromInt(epoch_correct)) / @as(f64, @floatFromInt((step + 1) * batch_size)) * 100.0;
                try w.interface.print("  epoch {}/{} batch {}/{} loss={d:.4} train_acc={d:.1}%\r", .{
                    epoch + 1, n_epochs, step + 1, train_batches, avg_loss, acc,
                });
                w.interface.flush() catch {};
            }
        }

        const epoch_ns: u64 = @intCast(std.Io.Clock.awake.now(io).nanoseconds - epoch_t0);
        const epoch_ms = @as(f64, @floatFromInt(epoch_ns)) / 1_000_000.0;
        const avg_loss = epoch_loss / @as(f64, @floatFromInt(train_batches));
        const train_acc = @as(f64, @floatFromInt(epoch_correct)) / @as(f64, @floatFromInt(train_batches * batch_size)) * 100.0;
        const imgs_per_sec = @as(f64, @floatFromInt(train_batches * batch_size)) / (epoch_ms / 1000.0);

        try w.interface.print("  Epoch {}/{}: loss={d:.4}  train_acc={d:.1}%  {d:.0}ms  ({d:.0} img/s)\n", .{
            epoch + 1, n_epochs, avg_loss, train_acc, epoch_ms, imgs_per_sec,
        });
        w.interface.flush() catch {};
    }

    const total_ns: u64 = @intCast(std.Io.Clock.awake.now(io).nanoseconds - total_t0);
    const total_ms = @as(f64, @floatFromInt(total_ns)) / 1_000_000.0;

    // Test set evaluation
    try w.interface.print("\nEvaluating on test set...\n", .{});
    w.interface.flush() catch {};

    const test_batches = test_data.n_samples / batch_size;
    var test_correct: usize = 0;
    var preds_test: [32]usize = undefined;

    for (0..test_batches) |batch_idx| {
        const img_off = batch_idx * batch_size * pixels_per_image;
        @memcpy(model.xs_batch.data, test_data.images[img_off..][0 .. batch_size * pixels_per_image]);

        model.g.reset();
        model.g.compute();

        model.predict(&preds_test);
        for (0..batch_size) |i| {
            if (preds_test[i] == @as(usize, @intFromFloat(test_data.labels[batch_idx * batch_size + i]))) {
                test_correct += 1;
            }
        }
    }

    const test_acc = @as(f64, @floatFromInt(test_correct)) / @as(f64, @floatFromInt(test_batches * batch_size)) * 100.0;

    try w.interface.print("\nResults\n", .{});
    try w.interface.print("-------\n", .{});
    try w.interface.print("  Test accuracy: {d:.2}% ({}/{})  \n", .{ test_acc, test_correct, test_batches * batch_size });
    try w.interface.print("  Total training time: {d:.0}ms\n", .{total_ms});
    try w.interface.print("  Avg time per epoch: {d:.0}ms\n", .{total_ms / @as(f64, @floatFromInt(n_epochs))});
    try w.interface.print("\n", .{});
    w.interface.flush() catch {};
}
