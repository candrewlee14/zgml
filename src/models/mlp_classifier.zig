//! MLP Classifier: multi-class classification with cross-entropy loss.
//!
//! Architecture: 2 → 8 (relu) → 4 logits, trained with cross-entropy.
//! Demonstrates matmul-based layers, softmax, and multi-class output.

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

        w1: *Tensor(T),
        b1: *Tensor(T),
        w2: *Tensor(T),
        b2: *Tensor(T),
        xs_batch: *Tensor(T),
        ys_batch: *Tensor(T),
        logits: *Tensor(T),
        loss: *Tensor(T),
        g: ComputeGraph(T),

        pub fn build(backing_alloc: Alloc, batch_size: usize) !Self {
            var g = ComputeGraph(T).init(backing_alloc);
            const a = g.allocator();

            const w1 = try g.param(&.{ 8, 2 });
            const b1 = try g.param(&.{8});
            const w2 = try g.param(&.{ 4, 8 });
            const b2 = try g.param(&.{4});

            nn.kaimingUniform(T, w1, 42);
            nn.kaimingUniform(T, w2, 123);

            const xs_batch = try Tensor(T).init(a, &.{ 2, batch_size });
            const ys_batch = try Tensor(T).init(a, &.{batch_size});

            const hidden = nn.linear(T, xs_batch, w1, b1).relu();
            const logits = nn.linear(T, hidden, w2, b2);
            const loss = loss_mod.crossEntropy(T, logits, ys_batch);

            try g.buildForward(loss);
            try g.buildBackward(true);

            return .{
                .w1 = w1, .b1 = b1, .w2 = w2, .b2 = b2,
                .xs_batch = xs_batch, .ys_batch = ys_batch,
                .logits = logits, .loss = loss, .g = g,
            };
        }

        pub fn params(self: *const Self) [4]*Tensor(T) {
            return .{ self.w1, self.b1, self.w2, self.b2 };
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

        test "mlp classifier learns quadrants" {
            const n_samples = 16;
            const xs = try Tensor(T).init(tac, &.{ 2, n_samples });
            // zig fmt: off
            xs.setData(&.{
                 1,  1,           -1,  1,          -1, -1,           1, -1,
                 2,  2,           -2,  2,          -2, -2,           2, -2,
                 1,  2,           -1,  2,          -1, -2,           1, -2,
                 2,  1,           -2,  1,          -2, -1,           2, -1,
            });
            // zig fmt: on
            const ys = try Tensor(T).init(tac, &.{n_samples});
            ys.setData(&.{ 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 });
            defer xs.deinit();
            defer ys.deinit();

            var model = try Model(T).build(tac, n_samples);
            defer model.deinit();

            @memcpy(model.xs_batch.data, xs.data);
            @memcpy(model.ys_batch.data, ys.data);
            try model.g.run(model.loss);
            const initial_loss = model.loss.data[0];

            const p = model.params();
            var optimizer = try optim.sgd.SGD(T).init(tac, &p, .{ .lr = 0.1, .momentum = 0.5 });
            defer optimizer.deinit();
            try nn.trainSupervised(T, &model.g, model.loss, model.xs_batch, model.ys_batch, xs, ys, 2000, &optimizer);

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
            try testing.expect(correct >= 15);
        }
    };
}
