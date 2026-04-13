//! Polynomial regression model: `y = c0 + c1*x + c2*x^2 + ... + cn*x^n`.

const std = @import("std");
const testing = std.testing;
const tac = testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Alloc = std.mem.Allocator;
const loss_mod = @import("../loss.zig");
const optim = @import("../optim.zig");
const nn = @import("../nn.zig");

/// A polynomial model with `max_exp + 1` learnable coefficients.
pub fn Model(comptime T: type) type {
    return struct {
        const Self = @This();

        xs_batch: *Tensor(T),
        ys_batch: *Tensor(T),
        coeffs: std.ArrayList(*Tensor(T)),
        out: *Tensor(T),
        loss: *Tensor(T),
        g: ComputeGraph(T),

        pub fn build(backing_alloc: Alloc, max_exp: usize, batch_size: usize) !Self {
            var g = ComputeGraph(T).init(backing_alloc);
            const a = g.allocator();

            var coeffs = try std.ArrayList(*Tensor(T)).initCapacity(a, max_exp + 1);
            const xs_batch = try Tensor(T).init(a, &.{batch_size});
            const ys_batch = try Tensor(T).init(a, &.{batch_size});

            for (0..max_exp + 1) |_| {
                const param = try Tensor(T).initScalar(a, 0);
                param.setParam();
                coeffs.appendAssumeCapacity(param);
            }

            var total = try Tensor(T).initScalar(a, 0);
            var cur_term = try Tensor(T).initScalar(a, 1);
            for (coeffs.items, 0..) |param, i| {
                total = total.add(cur_term.mul(param));
                if (i < max_exp) cur_term = cur_term.mul(xs_batch);
            }

            const loss = loss_mod.meanSqErr(T, total, ys_batch);
            try g.buildForward(loss);
            try g.buildBackward(true);

            return .{
                .xs_batch = xs_batch, .ys_batch = ys_batch,
                .coeffs = coeffs, .out = total,
                .loss = loss, .g = g,
            };
        }

        pub fn deinit(self: *Self) void {
            self.g.deinit();
        }

        test "linear poly model with sgd optim" {
            const n = 20;
            const time = try Tensor(T).initLinspace(tac, &.{n}, 0, 20);
            const true_m = 30;
            const speed = try Tensor(T).initLinspace(tac, &.{n}, 0, 20 * true_m);
            defer time.deinit();
            defer speed.deinit();

            var model = try Model(T).build(tac, 1, 1);
            defer model.deinit();

            var optimizer = try optim.sgd.SGD(T).init(tac, model.coeffs.items, .{ .lr = 1e-3, .momentum = 0.2 });
            defer optimizer.deinit();
            try nn.trainSupervised(T, &model.g, model.loss, model.xs_batch, model.ys_batch, time, speed, 10, &optimizer);
            try testing.expectApproxEqAbs(@as(T, true_m), model.coeffs.items[1].data[0], 5e-1);
        }
    };
}
