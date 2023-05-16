const std = @import("std");
const testing = std.testing;
const tac = testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Alloc = std.mem.Allocator;
const loss = @import("../loss.zig");
const optim = @import("../optim.zig");

pub fn Model(comptime T: type) type {
    return struct {
        const Self = @This();

        cur_batch: usize,
        batch_size: usize,
        xs: *Tensor(T),
        xsBatch: *Tensor(T),
        ys: *Tensor(T),
        mu: *Tensor(T),
        ysBatch: *Tensor(T),
        params: [2]*Tensor(T),
        momentum: [2]*Tensor(T),
        g: ComputeGraph(T),
        out: *Tensor(T),
        loss: *Tensor(T),

        pub fn build(alloc: Alloc, m: T, b: T, xs: *Tensor(T), ys: *Tensor(T), batch_size: usize, mu: T) !Self {
            var p = Self{
                // zig fmt: off
                .params = .{ 
                    try Tensor(T).initScalar(alloc, m), 
                    try Tensor(T).initScalar(alloc, b), 
                },
                .mu = try Tensor(T).initScalar(alloc, mu),
                .momentum = .{ 
                    try Tensor(T).initScalar(alloc, 0), 
                    try Tensor(T).initScalar(alloc, 0), 
                },
                // zig fmt: on
                .xs = xs,
                .xsBatch = try Tensor(T).init(alloc, &.{batch_size}),
                .ys = ys,
                .ysBatch = try Tensor(T).init(alloc, &.{batch_size}),
                .g = ComputeGraph(T).init(alloc),
                .out = undefined,
                .loss = undefined,
                .batch_size = batch_size,
                .cur_batch = 0,
            };
            for (p.params) |param| {
                param.setParam();
            }
            p.params[0].name = "m";
            p.params[1].name = "b";
            var ne_batch: [1]usize = .{batch_size};
            const repeated0 = p.params[0].repeat(ne_batch[0..]);
            repeated0.name = "m repeated to batch size";

            p.xsBatch.name = "xs batch";
            const mx = p.xsBatch.mul(repeated0);
            mx.name = "m*x";
            const repeated1 = p.params[1].repeat(ne_batch[0..]);
            repeated1.name = "b repeated to batch size";
            const mxPlusB = mx.add(repeated1);
            mxPlusB.name = "m*x+b";
            p.out = mxPlusB;
            p.loss = loss.meanSqErr(T, p.out, p.ysBatch);
            p.loss.name = "loss";
            try p.g.buildForward(p.loss);
            try p.g.buildBackward(true);
            return p;
        }

        pub fn deinit(self: *Self) void {
            self.g.deinit();
            for (self.momentum) |mom| {
                mom.deinit();
            }
            self.mu.deinit();
        }

        pub fn compute(self: *Self) void {
            self.g.reset();
            self.g.resetGrads();
            if (self.loss.grad) |grad| _ = grad.setAllScalar(1);
            self.g.compute();
        }

        pub fn train(self: *Self, n_epochs: usize, batch_size: usize, optimizer: anytype) void {
            const n_elems = self.xsBatch.nElems();
            const n_batches = self.xs.nElems() / (n_elems * batch_size);
            for (0..n_epochs) |_| {
                for (0..n_batches) |b| {
                    // move batch of xs & ys into batch data
                    @memcpy(self.xsBatch.data, self.xs.data[b * self.batch_size ..][0..n_elems]);
                    @memcpy(self.ysBatch.data, self.ys.data[b * self.batch_size ..][0..n_elems]);
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
            defer time.deinit();
            defer speed.deinit();

            var model = try Model(T).build(tac, 0, 0, time, speed, 1, 0.2);
            defer model.deinit();

            var optimizer = try optim.sgd.SGD(T).init(tac, &model.params, 1, model.loss, 1e-3, 0.2);
            defer optimizer.deinit();
            model.train(10, 1, &optimizer);
            try testing.expectApproxEqAbs(@as(T, true_m), model.params[0].data[0], 5e-1);
        }
    };
}
