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
        xs_batch: *Tensor(T),
        ys_batch: *Tensor(T),
        // TODO: refactor to store any number of params
        params: [2]*Tensor(T),
        g: ComputeGraph(T),
        out: *Tensor(T),
        loss: *Tensor(T),

        /// Create a model
        /// y = m*x + b
        pub fn build(alloc: Alloc, m: T, b: T, batch_size: usize) !Self {
            var p = Self{
                // zig fmt: off
                .params = .{ 
                    try Tensor(T).initScalar(alloc, m), 
                    try Tensor(T).initScalar(alloc, b), 
                },
                // zig fmt: on
                .xs_batch = try Tensor(T).init(alloc, &.{batch_size}),
                .ys_batch = try Tensor(T).init(alloc, &.{batch_size}),
                .g = ComputeGraph(T).init(alloc),
                .out = undefined,
                .loss = undefined,
                .batch_size = batch_size,
                .cur_batch = 0,
            };
            for (p.params) |param| {
                param.setParam();
            }

            var ne_batch: [1]usize = .{batch_size};
            const repeated0 = p.params[0].repeat(ne_batch[0..]);
            const mx = p.xs_batch.mul(repeated0);
            const repeated1 = p.params[1].repeat(ne_batch[0..]);
            p.out = mx.add(repeated1);
            p.loss = loss.meanSqErr(T, p.out, p.ys_batch);
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
            self.g.resetGrads();
            if (self.loss.grad) |grad| _ = grad.setAllScalar(1);
            self.g.compute();
        }

        pub fn train(self: *Self, xs: *Tensor(T), ys: *Tensor(T), n_epochs: usize, batch_size: usize, optimizer: anytype) void {
            const n_elems = self.xs_batch.nElems();
            const n_batches = xs.nElems() / (n_elems * batch_size);
            for (0..n_epochs) |_| {
                // TODO: write a random shuffle function for tensors
                for (0..n_batches) |b| {
                    // move batch of xs & ys into batch data
                    @memcpy(self.xs_batch.data, xs.data[b * self.batch_size ..][0..n_elems]);
                    @memcpy(self.ys_batch.data, ys.data[b * self.batch_size ..][0..n_elems]);
                    optimizer.zeroGrad();
                    self.compute();
                    optimizer.step();
                    // for (self.params, 0..) |param, p_i| {
                    //     std.debug.print("{s} {d}: {} grad={}\n", .{ param.name orelse "", p_i, param.data[0], param.grad.?.data[0] });
                    // }
                }
            }
        }

        test "linear model with sgd optim" {
            const n = 20;
            const time = try Tensor(T).linspace(tac, 0, 20, &.{n});
            const true_m: T = 30;
            const speed = try Tensor(T).linspace(tac, 0, 20 * true_m, &.{n});
            defer time.deinit();
            defer speed.deinit();

            var model = try Model(T).build(tac, 0, 0, 1);
            defer model.deinit();

            var optimizer: optim.sgd.SGDMomentum(T) = undefined;
            try optimizer.init(tac, &model.params, 1e-3, 0.2);
            defer optimizer.deinit();

            model.train(time, speed, 10, 1, &optimizer);
            try testing.expectApproxEqAbs(true_m, model.params[0].data[0], 5e-1);
        }
    };
}
