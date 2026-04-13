//! MNIST CNN training benchmark.
//!
//! Run with: zig build mnist-bench
//! Built with ReleaseFast for meaningful measurements.

const std = @import("std");
const zgml = @import("zgml");
const Tensor = zgml.Tensor;
const ComputeGraph = zgml.ComputeGraph;
const loss_mod = zgml.loss;

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

        // Convert to f32 [W, H, 1, N] layout — pixel (x,y) of image n at index x + y*cols + n*cols*rows
        // IDX stores row-major: pixels[n*rows*cols + y*cols + x]
        // Our tensor is [cols, rows, 1, N] which is also x + y*cols + 0 + n*cols*rows — same layout
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
// CNN model (built directly, not using the model template)
// ---------------------------------------------------------------------------

const CnnMnist = struct {
    g: ComputeGraph(f32),
    xs_batch: *Tensor(f32),
    ys_batch: *Tensor(f32),
    logits: *Tensor(f32),
    loss_tensor: *Tensor(f32),
    params: [6]*Tensor(f32),

    fn build(backing_alloc: Alloc, batch_size: usize) !CnnMnist {
        var self: CnnMnist = undefined;
        self.g = ComputeGraph(f32).init(backing_alloc);
        const a2 = self.g.allocator();

        // Conv: 1->8, 5x5 kernel -> 24x24, pool -> 12x12
        const k1 = try Tensor(f32).init(a2, &.{ 5, 5, 1, 8 });
        const b1 = try Tensor(f32).init(a2, &.{8});
        // FC: 12*12*8=1152 -> 10
        const fc_w = try Tensor(f32).init(a2, &.{ 10, 12 * 12 * 8 });
        const fc_b = try Tensor(f32).init(a2, &.{10});

        // Xavier-ish init
        var prng = std.Random.DefaultPrng.init(42);
        const rng = prng.random();
        for (k1.data) |*d| d.* = (rng.float(f32) - 0.5) * 0.2;
        for (fc_w.data) |*d| d.* = (rng.float(f32) - 0.5) * 0.05;
        _ = b1.setAllScalar(0);
        _ = fc_b.setAllScalar(0);

        for ([_]*Tensor(f32){ k1, b1, fc_w, fc_b }) |p| p.setParam();
        self.params = .{ k1, b1, fc_w, fc_b, k1, b1 }; // pad to 6, only first 4 used

        self.xs_batch = try Tensor(f32).init(a2, &.{ 28, 28, 1, batch_size });
        self.ys_batch = try Tensor(f32).init(a2, &.{batch_size});

        // Forward
        const conv_out = self.xs_batch.conv2d(k1); // [24, 24, 8, batch]
        const b1_4d = b1.reshape(&.{ 1, 1, 8, 1 });
        const b1_rep = b1_4d.repeat(&conv_out.ne);
        const act1 = conv_out.add(b1_rep).relu();
        const pool1 = act1.maxPool2d(); // [12, 12, 8, batch]
        const flat = pool1.reshape(&.{ 12 * 12 * 8, batch_size });
        const fc_out = flat.matMul(false, fc_w, false);
        const fb_rep = fc_b.repeat(&fc_out.ne);
        self.logits = fc_out.add(fb_rep);
        self.loss_tensor = loss_mod.crossEntropy(f32, self.logits, self.ys_batch);

        try self.g.buildForward(self.loss_tensor);
        try self.g.buildBackward(true);
        try self.g.fusionPass();

        // Use only the real 4 params
        self.params = .{ k1, b1, fc_w, fc_b, k1, b1 };
        return self;
    }

    fn deinit(self: *CnnMnist) void {
        self.g.deinit();
    }

    fn computeForwardBackward(self: *CnnMnist) void {
        self.g.reset();
        self.g.resetGrads();
        if (self.loss_tensor.grad) |grad| _ = grad.setAllScalar(1);
        self.g.compute();
    }

    fn predict(self: *const CnnMnist, preds: []usize) void {
        const batch = self.logits.ne[1];
        for (0..batch) |s| {
            var best: usize = 0;
            var best_val: f32 = self.logits.data[s * 10];
            for (1..10) |c| {
                const val = self.logits.data[s * 10 + c];
                if (val > best_val) {
                    best_val = val;
                    best = c;
                }
            }
            preds[s] = best;
        }
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
    const n_epochs: usize = 3;
    const train_batches = train_images.n / batch_size;
    const pixels_per_image = train_images.rows * train_images.cols;

    try w.interface.print("Architecture: Conv(5x5, 1->8) -> ReLU -> MaxPool(2x2) -> FC(1152->10)\n", .{});
    try w.interface.print("Optimizer: SGD (lr=0.01, momentum=0.9)\n", .{});
    try w.interface.print("Batch size: {}, Epochs: {}\n\n", .{ batch_size, n_epochs });
    w.interface.flush() catch {};

    // Build model
    var model = try CnnMnist.build(alloc, batch_size);
    defer model.deinit();

    // SGD optimizer (only first 4 params are real)
    const real_params = model.params[0..4];
    var sgd = try zgml.optim.sgd.SGD(f32).init(alloc, real_params, .{ .lr = 0.01, .momentum = 0.9 });
    defer sgd.deinit();

    // Training loop
    var total_timer = std.time.Timer.start() catch unreachable;

    for (0..n_epochs) |epoch| {
        var epoch_timer = std.time.Timer.start() catch unreachable;
        var epoch_loss: f64 = 0;
        var epoch_correct: usize = 0;
        var preds: [32]usize = undefined; // max batch_size

        for (0..train_batches) |batch_idx| {
            // Copy batch data
            const img_off = batch_idx * batch_size * pixels_per_image;
            @memcpy(model.xs_batch.data, train_images.data[img_off..][0 .. batch_size * pixels_per_image]);
            const lbl_off = batch_idx * batch_size;
            @memcpy(model.ys_batch.data, train_labels.data[lbl_off..][0..batch_size]);

            sgd.zeroGrad();
            model.computeForwardBackward();
            sgd.step();

            epoch_loss += model.loss_tensor.data[0];

            // Track accuracy
            model.predict(&preds);
            for (0..batch_size) |i| {
                if (preds[i] == @as(usize, @intFromFloat(train_labels.data[lbl_off + i]))) {
                    epoch_correct += 1;
                }
            }

            if ((batch_idx + 1) % 100 == 0) {
                const avg_loss = epoch_loss / @as(f64, @floatFromInt(batch_idx + 1));
                const acc = @as(f64, @floatFromInt(epoch_correct)) / @as(f64, @floatFromInt((batch_idx + 1) * batch_size)) * 100.0;
                try w.interface.print("  epoch {}/{} batch {}/{} loss={d:.4} train_acc={d:.1}%\r", .{
                    epoch + 1, n_epochs, batch_idx + 1, train_batches, avg_loss, acc,
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
