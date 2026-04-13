//! Polynomial regression model: `y = c0 + c1*x + c2*x^2 + ... + cn*x^n`.

const std = @import("std");
const testing = std.testing;
const tac = testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Alloc = std.mem.Allocator;
const loss = @import("../loss.zig");
const optim = @import("../optim.zig");

/// A polynomial model with `max_exp + 1` learnable coefficients.
///
/// Owns a `ComputeGraph` internally. Call `build` to construct the graph,
/// then `train` to run mini-batch SGD.
pub fn Model(comptime T: type) type {
    return struct {
        const Self = @This();

        cur_batch: usize,
        batch_size: usize,
        xs_batch: *Tensor(T),
        ys_batch: *Tensor(T),
        params: std.ArrayList(*Tensor(T)),
        g: ComputeGraph(T),
        out: *Tensor(T),
        loss: *Tensor(T),

        pub fn build(backing_alloc: Alloc, max_exp: usize, batch_size: usize) !Self {
            var res = Self{
                .g = ComputeGraph(T).init(backing_alloc),
                .params = undefined,
                .xs_batch = undefined,
                .ys_batch = undefined,
                .out = undefined,
                .loss = undefined,
                .batch_size = batch_size,
                .cur_batch = 0,
            };
            const a = res.g.allocator();
            res.params = try std.ArrayList(*Tensor(T)).initCapacity(a, max_exp + 1);
            res.xs_batch = try Tensor(T).init(a, &.{batch_size});
            res.ys_batch = try Tensor(T).init(a, &.{batch_size});
            for (0..max_exp + 1) |_| {
                const param = try Tensor(T).initScalar(a, 0);
                param.setParam(a);
                res.params.appendAssumeCapacity(param);
            }
            // mul by xs_batch
            var total = try Tensor(T).initScalar(a, 0);
            var cur_term = try Tensor(T).initScalar(a, 1);
            for (res.params.items, 0..) |param, i| {
                total = total.add(a, cur_term.mul(a, param));
                if (i < max_exp) cur_term = cur_term.mul(a, res.xs_batch);
            }
            res.out = total;
            res.loss = loss.meanSqErr(T, a, res.out, res.ys_batch);
            res.loss.name = "loss";
            try res.g.buildForward(res.loss);
            try res.g.buildBackward(true);
            return res;
        }

        pub fn deinit(self: *Self) void {
            self.g.deinit();
        }

        pub fn compute(self: *Self) void {
            self.g.reset();
            self.g.resetGrads();
            if (self.loss.grad) |grad| _ = grad.setAllScalar(1);
            self.g.compute();
        }

        pub fn train(self: *Self, xs: *Tensor(T), ys: *Tensor(T), n_epochs: usize, batch_size: usize, optimizer: anytype) void {
            const n_elems = self.xs_batch.nElems();
            const n_batches = xs.nElems() / (n_elems * batch_size);
            for (0..n_epochs) |_| {
                for (0..n_batches) |b_idx| {
                    @memcpy(self.xs_batch.data, xs.data[b_idx * self.batch_size ..][0..n_elems]);
                    @memcpy(self.ys_batch.data, ys.data[b_idx * self.batch_size ..][0..n_elems]);
                    optimizer.zeroGrad();
                    self.compute();
                    optimizer.step();
                }
            }
        }

        test "linear poly model with sgd optim" {
            const n = 20;
            const time = try Tensor(T).initLinspace(tac, &.{n}, 0, 20);
            const true_m = 30;
            const speed = try Tensor(T).initLinspace(tac, &.{n}, 0, 20 * true_m);
            defer time.deinit(tac);
            defer speed.deinit(tac);

            var model = try Model(T).build(tac, 1, 1);
            defer model.deinit();

            var optimizer = try optim.sgd.SGD(T).init(tac, model.params.items, 1, model.loss, 1e-3, 0.2);
            defer optimizer.deinit();
            model.train(time, speed, 10, 1, &optimizer);
            try testing.expectApproxEqAbs(@as(T, true_m), model.params.items[1].data[0], 5e-1);
        }
    };
}
