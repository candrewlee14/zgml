//! Linear regression model: `y = m*x + b`.

const std = @import("std");
const testing = std.testing;
const tac = testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Alloc = std.mem.Allocator;
const loss_mod = @import("../loss.zig");
const optim = @import("../optim.zig");
const nn = @import("../nn.zig");

/// A simple linear model with two learnable scalars (slope `m` and bias `b`).
pub fn Model(comptime T: type) type {
    return struct {
        const Self = @This();

        m: *Tensor(T),
        b: *Tensor(T),
        xs_batch: *Tensor(T),
        ys_batch: *Tensor(T),
        out: *Tensor(T),
        loss: *Tensor(T),
        g: ComputeGraph(T),

        pub fn build(backing_alloc: Alloc, m_init: T, b_init: T, batch_size: usize) !Self {
            var g = ComputeGraph(T).init(backing_alloc);
            const a = g.allocator();

            const m = try Tensor(T).initScalar(a, m_init);
            m.setParam();
            const b = try Tensor(T).initScalar(a, b_init);
            b.setParam();

            const xs_batch = try Tensor(T).init(a, &.{batch_size});
            const ys_batch = try Tensor(T).init(a, &.{batch_size});

            var ne_batch: [1]usize = .{batch_size};
            const mx = xs_batch.mul(m.repeat(ne_batch[0..]));
            const out = mx.add(b.repeat(ne_batch[0..]));
            const loss = loss_mod.meanSqErr(T, out, ys_batch);

            try g.buildForward(loss);
            try g.buildBackward(true);

            return .{
                .m = m, .b = b,
                .xs_batch = xs_batch, .ys_batch = ys_batch,
                .out = out, .loss = loss, .g = g,
            };
        }

        pub fn params(self: *const Self) [2]*Tensor(T) {
            return .{ self.m, self.b };
        }

        pub fn deinit(self: *Self) void {
            self.g.deinit();
        }

        test "linear model with sgd optim" {
            const n = 20;
            const time = try Tensor(T).initLinspace(tac, &.{n}, 0, 20);
            const true_m = 30;
            const speed = try Tensor(T).initLinspace(tac, &.{n}, 0, 20 * true_m);
            defer time.deinit();
            defer speed.deinit();

            var model = try Model(T).build(tac, 0, 0, 1);
            defer model.deinit();

            const p = model.params();
            var optimizer = try optim.sgd.SGD(T).init(tac, &p, .{ .lr = 1e-3, .momentum = 0.2 });
            defer optimizer.deinit();
            try nn.trainSupervised(T, &model.g, model.loss, model.xs_batch, model.ys_batch, time, speed, 10, &optimizer);
            try testing.expectApproxEqAbs(@as(T, true_m), model.m.data[0], 5e-1);
        }
    };
}
