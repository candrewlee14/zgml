//! Linear regression model: `y = m*x + b`.

const std = @import("std");
const testing = std.testing;
const tac = testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Alloc = std.mem.Allocator;
const loss = @import("../loss.zig");
const optim = @import("../optim.zig");

/// A simple linear model with two learnable parameters (slope `m` and bias `b`).
///
/// Owns a `ComputeGraph` internally. Call `build` to construct the graph,
/// then `train` to run mini-batch SGD.
pub fn Model(comptime T: type) type {
    return struct {
        const Self = @This();

        batch_size: usize,
        xs_batch: *Tensor(T),
        ys_batch: *Tensor(T),
        params: [2]*Tensor(T),
        g: ComputeGraph(T),
        out: *Tensor(T),
        loss: *Tensor(T),

        pub fn build(backing_alloc: Alloc, m: T, b: T, batch_size: usize) !Self {
            var p = Self{
                .g = ComputeGraph(T).init(backing_alloc),
                .params = undefined,
                .xs_batch = undefined,
                .ys_batch = undefined,
                .out = undefined,
                .loss = undefined,
                .batch_size = batch_size,
            };
            const a = p.g.allocator();
            // zig fmt: off
            p.params = .{
                try Tensor(T).initScalar(a, m),
                try Tensor(T).initScalar(a, b),
            };
            // zig fmt: on
            p.xs_batch = try Tensor(T).init(a, &.{batch_size});
            p.ys_batch = try Tensor(T).init(a, &.{batch_size});
            for (p.params) |param| {
                param.setParam(a);
            }

            var ne_batch: [1]usize = .{batch_size};
            const repeated0 = p.params[0].repeat(a, ne_batch[0..]);
            const mx = p.xs_batch.mul(a, repeated0);
            const repeated1 = p.params[1].repeat(a, ne_batch[0..]);
            p.out = mx.add(a, repeated1);
            p.loss = loss.meanSqErr(T, a, p.out, p.ys_batch);
            { // for debugging
                p.params[0].name = "m";
                p.params[1].name = "b";
                repeated0.name = "m repeated to batch size";
                p.xs_batch.name = "xs batch";
                mx.name = "m*x";
                repeated1.name = "b repeated to batch size";
                p.out.name = "m*x+b";
                p.loss.name = "loss";
            }
            try p.g.buildForward(p.loss);
            try p.g.buildBackward(true);
            return p;
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

        pub fn train(self: *Self, xs: *Tensor(T), ys: *Tensor(T), n_epochs: usize, optimizer: anytype) void {
            const n_elems = self.xs_batch.nElems();
            std.debug.assert(ys.nElems() == xs.nElems());
            std.debug.assert(self.batch_size == n_elems);
            std.debug.assert(xs.nElems() % self.batch_size == 0);
            const n_batches = xs.nElems() / self.batch_size;
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

        test "linear model with sgd optim" {
            const n = 20;
            const time = try Tensor(T).initLinspace(tac, &.{n}, 0, 20);
            const true_m = 30;
            const speed = try Tensor(T).initLinspace(tac, &.{n}, 0, 20 * true_m);
            defer time.deinit(tac);
            defer speed.deinit(tac);

            var model = try Model(T).build(tac, 0, 0, 1);
            defer model.deinit();

            var optimizer = try optim.sgd.SGD(T).init(tac, &model.params, model.loss, 1e-3, 0.2);
            defer optimizer.deinit();
            model.train(time, speed, 10, &optimizer);
            try testing.expectApproxEqAbs(@as(T, true_m), model.params[0].data[0], 5e-1);
        }
    };
}
