//! Convolutional classifier: CNN for image classification.
//!
//! Two architectures:
//!   - `build`:     Conv(KxK) → ReLU → MaxPool → FC → Softmax  (simple baseline)
//!   - `buildDeep`: Conv(5x5) → BN → ReLU → MaxPool → FC → ReLU → FC → Softmax  (better accuracy)

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

        param_slice: []const *Tensor(T),
        backing_alloc: Alloc,
        xs_batch: *Tensor(T),
        ys_batch: *Tensor(T),
        logits: *Tensor(T),
        loss: *Tensor(T),
        g: ComputeGraph(T),

        /// Simple baseline: Conv(KxK) → ReLU → MaxPool(2x2) → FC
        pub fn build(
            backing_alloc: Alloc,
            in_w: usize,
            in_h: usize,
            kernel_size: usize,
            n_filters: usize,
            n_classes: usize,
            batch_size: usize,
        ) !Self {
            var g = ComputeGraph(T).init(backing_alloc);
            const a = g.allocator();
            const conv_k = try g.param(&.{ kernel_size, kernel_size, 1, n_filters });
            const conv_b = try g.param(&.{n_filters});
            const conv_w = in_w - kernel_size + 1;
            const conv_h = in_h - kernel_size + 1;
            const pool_w = conv_w / 2;
            const pool_h = conv_h / 2;
            const flat_dim = pool_w * pool_h * n_filters;
            const fc_w = try g.param(&.{ n_classes, flat_dim });
            const fc_b = try g.param(&.{n_classes});

            nn.kaimingUniform(T, conv_k, 42);
            nn.kaimingUniform(T, fc_w, 123);

            const param_slice = try backing_alloc.dupe(*Tensor(T), &.{ conv_k, conv_b, fc_w, fc_b });

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
            try g.buildBackward(false);

            return .{
                .param_slice = param_slice, .backing_alloc = backing_alloc,
                .xs_batch = xs_batch, .ys_batch = ys_batch,
                .logits = logits, .loss = loss, .g = g,
            };
        }

        /// Deeper model: Conv(5x5,1→F) → BN → ReLU → MaxPool → FC(flat→H) → ReLU → FC(H→C)
        pub fn buildDeep(
            backing_alloc: Alloc,
            in_w: usize,
            in_h: usize,
            n_filters: usize,
            n_hidden: usize,
            n_classes: usize,
            batch_size: usize,
        ) !Self {
            var g = ComputeGraph(T).init(backing_alloc);
            const a = g.allocator();
            const ks: usize = 5;

            // Conv layer
            const conv_k = try g.param(&.{ ks, ks, 1, n_filters });
            const conv_b = try g.param(&.{n_filters});
            nn.kaimingUniform(T, conv_k, 42);

            // Batch norm parameters (gamma=1, beta=0)
            const bn_g = try g.param(&.{n_filters});
            const bn_b = try g.param(&.{n_filters});
            for (bn_g.data) |*v| v.* = 1;
            @memset(bn_b.data, 0);

            // FC layers
            const conv_w = in_w - ks + 1;
            const conv_h = in_h - ks + 1;
            const pool_w = conv_w / 2;
            const pool_h = conv_h / 2;
            const flat_dim = pool_w * pool_h * n_filters;

            const fc1_w = try g.param(&.{ n_hidden, flat_dim });
            const fc1_b = try g.param(&.{n_hidden});
            nn.kaimingUniform(T, fc1_w, 123);

            const fc2_w = try g.param(&.{ n_classes, n_hidden });
            const fc2_b = try g.param(&.{n_classes});
            nn.kaimingUniform(T, fc2_w, 456);

            const param_slice = try backing_alloc.dupe(*Tensor(T), &.{ conv_k, conv_b, bn_g, bn_b, fc1_w, fc1_b, fc2_w, fc2_b });

            const xs_batch = try Tensor(T).init(a, &.{ in_w, in_h, 1, batch_size });
            const ys_batch = try Tensor(T).init(a, &.{batch_size});

            // Forward: conv → bias → BN → relu → pool → flatten → FC → relu → FC
            const conv_out = xs_batch.conv2d(conv_k);
            const cb_4d = conv_b.reshape(&.{ 1, 1, n_filters, 1 });
            var conv_ne = [_]usize{ conv_out.ne[0], conv_out.ne[1], conv_out.ne[2], conv_out.ne[3] };
            const conv_biased = conv_out.add(cb_4d.repeat(conv_ne[0..]));
            const bn_out = nn.batchNorm2d(T, conv_biased, bn_g, bn_b, 1e-5);
            const conv_act = bn_out.relu();
            const flat = conv_act.maxPool2d().reshape(&.{ flat_dim, batch_size });
            const hidden = nn.linear(T, flat, fc1_w, fc1_b).relu();
            const logits = nn.linear(T, hidden, fc2_w, fc2_b);

            const loss = loss_mod.crossEntropy(T, logits, ys_batch);
            try g.buildForward(loss);
            try g.buildBackward(false);

            return .{
                .param_slice = param_slice, .backing_alloc = backing_alloc,
                .xs_batch = xs_batch, .ys_batch = ys_batch,
                .logits = logits, .loss = loss, .g = g,
            };
        }

        pub fn params(self: *const Self) []const *Tensor(T) {
            return self.param_slice;
        }

        pub fn deinit(self: *Self) void {
            self.backing_alloc.free(self.param_slice);
            self.g.deinit();
        }

        pub fn predict(self: *const Self, preds: []usize) void {
            nn.argmax(T, self.logits, preds);
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

            var model = try Model(T).build(tac, in_w, in_h, 3, n_filters, n_classes, n_samples);
            defer model.deinit();

            @memcpy(model.xs_batch.data, xs.data);
            @memcpy(model.ys_batch.data, ys.data);
            try model.g.run(model.loss);
            const initial_loss = model.loss.data[0];

            const p = model.params();
            var optimizer = try optim.sgd.SGD(T).init(tac, p, .{ .lr = 0.1, .momentum = 0.9 });
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

        test "deep conv classifier learns synthetic patterns" {
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

            var model = try Model(T).buildDeep(tac, in_w, in_h, n_filters, 16, n_classes, n_samples);
            defer model.deinit();

            @memcpy(model.xs_batch.data, xs.data);
            @memcpy(model.ys_batch.data, ys.data);
            try model.g.run(model.loss);
            const initial_loss = model.loss.data[0];

            const p = model.params();
            var optimizer = try optim.adam.Adam(T).init(tac, p, .{ .lr = 0.01 });
            defer optimizer.deinit();
            try nn.trainSupervised(T, &model.g, model.loss, model.xs_batch, model.ys_batch, xs, ys, 500, &optimizer);

            @memcpy(model.xs_batch.data, xs.data);
            try model.g.infer(model.logits);
            const final_loss = model.loss.data[0];

            try testing.expect(final_loss < initial_loss * 0.5);
        }
    };
}
