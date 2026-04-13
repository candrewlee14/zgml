//! Convolutional classifier: CNN for image classification.
//!
//! Architecture: Conv(3x3) → ReLU → MaxPool(2x2) → Flatten → FC → Softmax
//! Demonstrates conv2d and max_pool2d ops with cross-entropy loss.

const std = @import("std");
const testing = std.testing;
const tac = testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Alloc = std.mem.Allocator;
const loss_mod = @import("../loss.zig");
const optim = @import("../optim.zig");
const nn = @import("../nn.zig");

pub fn Model(comptime T: type) type {
    return struct {
        const Self = @This();

        conv_k: *Tensor(T),
        conv_b: *Tensor(T),
        fc_w: *Tensor(T),
        fc_b: *Tensor(T),
        xs_batch: *Tensor(T),
        ys_batch: *Tensor(T),
        logits: *Tensor(T),
        loss: *Tensor(T),
        g: ComputeGraph(T),

        pub fn build(
            backing_alloc: Alloc,
            in_w: usize,
            in_h: usize,
            n_filters: usize,
            n_classes: usize,
            batch_size: usize,
        ) !Self {
            var g = ComputeGraph(T).init(backing_alloc);
            const a = g.allocator();

            const conv_k = try g.param(&.{ 3, 3, 1, n_filters });
            const conv_b = try g.param(&.{n_filters});
            const pool_w = (in_w - 2) / 2;
            const pool_h = (in_h - 2) / 2;
            const flat_dim = pool_w * pool_h * n_filters;
            const fc_w = try g.param(&.{ n_classes, flat_dim });
            const fc_b = try g.param(&.{n_classes});

            nn.kaimingUniform(T, conv_k, 42);
            nn.kaimingUniform(T, fc_w, 123);

            const xs_batch = try Tensor(T).init(a, &.{ in_w, in_h, 1, batch_size });
            const ys_batch = try Tensor(T).init(a, &.{batch_size});

            // Forward: conv -> bias -> relu -> pool -> flatten -> fc
            const conv_out = xs_batch.conv2d(conv_k);
            const cb_4d = conv_b.reshape(&.{ 1, 1, n_filters, 1 });
            var conv_ne = [_]usize{ conv_out.ne[0], conv_out.ne[1], conv_out.ne[2], conv_out.ne[3] };
            const conv_act = conv_out.add(cb_4d.repeat(conv_ne[0..])).relu();
            const flat = conv_act.maxPool2d().reshape(&.{ flat_dim, batch_size });
            const logits = nn.linear(T, flat, fc_w, fc_b);

            const loss = loss_mod.crossEntropy(T, logits, ys_batch);
            try g.buildForward(loss);
            try g.buildBackward(true);

            return .{
                .conv_k = conv_k, .conv_b = conv_b,
                .fc_w = fc_w, .fc_b = fc_b,
                .xs_batch = xs_batch, .ys_batch = ys_batch,
                .logits = logits, .loss = loss, .g = g,
            };
        }

        pub fn params(self: *const Self) [4]*Tensor(T) {
            return .{ self.conv_k, self.conv_b, self.fc_w, self.fc_b };
        }

        pub fn deinit(self: *Self) void {
            self.g.deinit();
        }

        pub fn predict(self: *const Self, preds: []usize) void {
            const n_classes = self.logits.ne[0];
            const batch = self.logits.ne[1];
            std.debug.assert(preds.len >= batch);
            for (0..batch) |s| {
                var best_class: usize = 0;
                var best_val: T = self.logits.data[s * n_classes];
                for (1..n_classes) |c| {
                    const val = self.logits.data[s * n_classes + c];
                    if (val > best_val) {
                        best_val = val;
                        best_class = c;
                    }
                }
                preds[s] = best_class;
            }
        }

        test "conv classifier learns synthetic patterns" {
            const in_w = 8;
            const in_h = 8;
            const n_filters = 4;
            const n_classes = 4;
            const n_samples = 8;

            const xs = try Tensor(T).init(tac, &.{ in_w, in_h, 1, n_samples });
            defer xs.deinit();
            @memset(xs.data, 0);

            const pixel_count = in_w * in_h;
            for (0..n_samples) |s| {
                const class = s / 2;
                const base = s * pixel_count;
                const variation: T = if (s % 2 == 0) 1.0 else 0.8;
                const x_start: usize = if (class % 2 == 1) 4 else 0;
                const y_start: usize = if (class >= 2) 4 else 0;
                for (y_start..y_start + 4) |y| {
                    for (x_start..x_start + 4) |x| {
                        xs.data[base + x + y * in_w] = variation;
                    }
                }
            }

            const ys = try Tensor(T).init(tac, &.{n_samples});
            defer ys.deinit();
            ys.setData(&.{ 0, 0, 1, 1, 2, 2, 3, 3 });

            var model = try Model(T).build(tac, in_w, in_h, n_filters, n_classes, n_samples);
            defer model.deinit();

            @memcpy(model.xs_batch.data, xs.data);
            @memcpy(model.ys_batch.data, ys.data);
            try model.g.run(model.loss);
            const initial_loss = model.loss.data[0];

            const p = model.params();
            var optimizer = try optim.sgd.SGD(T).init(tac, &p, .{ .lr = 0.1, .momentum = 0.9 });
            defer optimizer.deinit();
            try nn.trainSupervised(T, &model.g, model.loss, model.xs_batch, model.ys_batch, xs, ys, 1000, &optimizer);

            @memcpy(model.xs_batch.data, xs.data);
            try model.g.infer(model.logits);
            const final_loss = model.loss.data[0];

            try testing.expect(final_loss < initial_loss * 0.5);

            var preds: [n_samples]usize = undefined;
            model.predict(&preds);
            var correct: usize = 0;
            for (0..n_samples) |i| {
                if (preds[i] == @as(usize, @intFromFloat(ys.data[i]))) correct += 1;
            }
            try testing.expect(correct >= 6);
        }
    };
}
