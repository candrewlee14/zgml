//! XOR MLP: a 2-layer neural network that learns the XOR function.
//!
//! Architecture: 2 → 4 (relu) → 1, trained with MSE loss.
//! Demonstrates that backpropagation handles non-linearly separable problems.

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
        out: *Tensor(T),
        loss: *Tensor(T),
        g: ComputeGraph(T),

        pub fn build(backing_alloc: Alloc, batch_size: usize) !Self {
            var g = ComputeGraph(T).init(backing_alloc);
            const a = g.allocator();

            const w1 = try g.param(&.{ 4, 2 });
            const b1 = try g.param(&.{4});
            const w2 = try g.param(&.{ 1, 4 });
            const b2 = try g.param(&.{1});

            nn.kaimingUniform(T, w1, 42);
            nn.kaimingUniform(T, w2, 123);

            const xs_batch = try Tensor(T).init(a, &.{ 2, batch_size });
            const ys_batch = try Tensor(T).init(a, &.{ 1, batch_size });

            const hidden = nn.linear(T, xs_batch, w1, b1).relu();
            const out = nn.linear(T, hidden, w2, b2);
            const loss = loss_mod.meanSqErr(T, out, ys_batch);

            try g.buildForward(loss);
            try g.buildBackward(true);

            return .{
                .w1 = w1, .b1 = b1, .w2 = w2, .b2 = b2,
                .xs_batch = xs_batch, .ys_batch = ys_batch,
                .out = out, .loss = loss, .g = g,
            };
        }

        pub fn params(self: *const Self) [4]*Tensor(T) {
            return .{ self.w1, self.b1, self.w2, self.b2 };
        }

        pub fn deinit(self: *Self) void {
            self.g.deinit();
        }

        test "xor mlp learns xor" {
            const xs = try Tensor(T).init(tac, &.{ 2, 4 });
            xs.setData(&.{
                0, 0, 0, 1, 1, 0, 1, 1,
            });
            const ys = try Tensor(T).init(tac, &.{ 1, 4 });
            ys.setData(&.{ 0, 1, 1, 0 });
            defer xs.deinit();
            defer ys.deinit();

            var model = try Model(T).build(tac, 4);
            defer model.deinit();

            const p = model.params();
            var optimizer = try optim.sgd.SGD(T).init(tac, &p, .{ .lr = 0.1, .momentum = 0.1 });
            defer optimizer.deinit();
            try nn.trainSupervised(T, &model.g, model.loss, model.xs_batch, model.ys_batch, xs, ys, 500, &optimizer);

            @memcpy(model.xs_batch.data, xs.data);
            try model.g.infer(model.out);

            try testing.expectApproxEqAbs(@as(T, 0), model.out.data[0], 0.3);
            try testing.expectApproxEqAbs(@as(T, 1), model.out.data[1], 0.3);
            try testing.expectApproxEqAbs(@as(T, 1), model.out.data[2], 0.3);
            try testing.expectApproxEqAbs(@as(T, 0), model.out.data[3], 0.3);
        }
    };
}
