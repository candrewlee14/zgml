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

const Alloc = std.mem.Allocator;

// ---------------------------------------------------------------------------
// MNIST IDX file loading
// ---------------------------------------------------------------------------

fn readU32Big(buf: []const u8) u32 {
    return std.mem.readInt(u32, buf[0..4], .big);
}

const MnistImages = struct {
    data: []f32,
    n: usize,
    rows: usize,
    cols: usize,

    fn load(alloc: Alloc, path: []const u8) !MnistImages {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const raw = try file.readToEndAlloc(alloc, 128 * 1024 * 1024);
        defer alloc.free(raw);

        const magic = readU32Big(raw[0..]);
        if (magic != 2051) return error.BadMagic;
        const n = readU32Big(raw[4..]);
        const rows = readU32Big(raw[8..]);
        const cols = readU32Big(raw[12..]);
        const pixels = raw[16..];

        const total = n * rows * cols;
        const data = try alloc.alloc(f32, total);
        for (0..total) |i| {
            data[i] = @as(f32, @floatFromInt(pixels[i])) / 255.0;
        }
        return .{ .data = data, .n = n, .rows = rows, .cols = cols };
    }

    fn deinit(self: *MnistImages, alloc: Alloc) void {
        alloc.free(self.data);
    }
};

const MnistLabels = struct {
    data: []f32,
    n: usize,

    fn load(alloc: Alloc, path: []const u8) !MnistLabels {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const raw = try file.readToEndAlloc(alloc, 16 * 1024 * 1024);
        defer alloc.free(raw);

        const magic = readU32Big(raw[0..]);
        if (magic != 2049) return error.BadMagic;
        const n = readU32Big(raw[4..]);
        const labels = raw[8..];

        const data = try alloc.alloc(f32, n);
        for (0..n) |i| {
            data[i] = @floatFromInt(labels[i]);
        }
        return .{ .data = data, .n = n };
    }

    fn deinit(self: *MnistLabels, alloc: Alloc) void {
        alloc.free(self.data);
    }
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const stdout_file = std.fs.File.stdout();
    var buf: [4096]u8 = undefined;
    var w = stdout_file.writer(&buf);

    try w.interface.print("\nMNIST CNN Training Benchmark\n", .{});
    try w.interface.print("============================\n\n", .{});

    // Load data
    try w.interface.print("Loading MNIST data...\n", .{});
    w.interface.flush() catch {};

    var train_images = try MnistImages.load(alloc, "data/train-images-idx3-ubyte");
    defer train_images.deinit(alloc);
    var train_labels = try MnistLabels.load(alloc, "data/train-labels-idx1-ubyte");
    defer train_labels.deinit(alloc);
    var test_images = try MnistImages.load(alloc, "data/t10k-images-idx3-ubyte");
    defer test_images.deinit(alloc);
    var test_labels = try MnistLabels.load(alloc, "data/t10k-labels-idx1-ubyte");
    defer test_labels.deinit(alloc);

    try w.interface.print("  Train: {} images, Test: {} images\n", .{ train_images.n, test_images.n });
    try w.interface.print("  Image size: {}x{}\n\n", .{ train_images.cols, train_images.rows });

    const batch_size: usize = 32;
    const n_epochs: usize = 10;
    const train_batches = train_images.n / batch_size;
    const pixels_per_image = train_images.rows * train_images.cols;

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
    var total_timer = std.time.Timer.start() catch unreachable;

    for (0..n_epochs) |epoch| {
        var epoch_timer = std.time.Timer.start() catch unreachable;
        var epoch_loss: f64 = 0;
        var epoch_correct: usize = 0;
        var preds: [32]usize = undefined;

        // Shuffle batch order each epoch
        prng.random().shuffle(usize, batch_indices);

        for (0..train_batches) |step| {
            const batch_idx = batch_indices[step];

            // Copy batch data
            const img_off = batch_idx * batch_size * pixels_per_image;
            @memcpy(model.xs_batch.data, train_images.data[img_off..][0 .. batch_size * pixels_per_image]);
            const lbl_off = batch_idx * batch_size;
            @memcpy(model.ys_batch.data, train_labels.data[lbl_off..][0..batch_size]);

            model.g.reset();
            model.g.resetGrads();
            if (model.loss.grad) |grad| _ = grad.setAllScalar(1);
            model.g.compute();

            sgd.step();

            epoch_loss += model.loss.data[0];

            // Track accuracy
            model.predict(&preds);
            for (0..batch_size) |i| {
                if (preds[i] == @as(usize, @intFromFloat(train_labels.data[lbl_off + i]))) {
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

        const epoch_ns = epoch_timer.read();
        const epoch_ms = @as(f64, @floatFromInt(epoch_ns)) / 1_000_000.0;
        const avg_loss = epoch_loss / @as(f64, @floatFromInt(train_batches));
        const train_acc = @as(f64, @floatFromInt(epoch_correct)) / @as(f64, @floatFromInt(train_batches * batch_size)) * 100.0;
        const imgs_per_sec = @as(f64, @floatFromInt(train_batches * batch_size)) / (epoch_ms / 1000.0);

        try w.interface.print("  Epoch {}/{}: loss={d:.4}  train_acc={d:.1}%  {d:.0}ms  ({d:.0} img/s)\n", .{
            epoch + 1, n_epochs, avg_loss, train_acc, epoch_ms, imgs_per_sec,
        });
        w.interface.flush() catch {};
    }

    const total_ns = total_timer.read();
    const total_ms = @as(f64, @floatFromInt(total_ns)) / 1_000_000.0;

    // Test set evaluation
    try w.interface.print("\nEvaluating on test set...\n", .{});
    w.interface.flush() catch {};

    const test_batches = test_images.n / batch_size;
    var test_correct: usize = 0;
    var preds_test: [32]usize = undefined;

    for (0..test_batches) |batch_idx| {
        const img_off = batch_idx * batch_size * pixels_per_image;
        @memcpy(model.xs_batch.data, test_images.data[img_off..][0 .. batch_size * pixels_per_image]);

        model.g.reset();
        model.g.compute();

        model.predict(&preds_test);
        for (0..batch_size) |i| {
            if (preds_test[i] == @as(usize, @intFromFloat(test_labels.data[batch_idx * batch_size + i]))) {
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
