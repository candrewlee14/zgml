const std = @import("std");
const testing = std.testing;
const tac = testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Alloc = std.mem.Allocator;
const loss = @import("../loss.zig");

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

        fn build(alloc: Alloc, m: T, b: T, xs: *Tensor(T), ys: *Tensor(T), batch_size: usize, mu: T) !Self {
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

        fn deinit(self: *Self) void {
            self.g.deinit();
            for (self.momentum) |mom| {
                mom.deinit();
            }
            self.mu.deinit();
            // for (self.params) |p| {
            //     p.deinit();
            // }
        }

        fn compute(self: *Self) void {
            self.g.reset();
            self.g.resetGrads();
            if (self.loss.grad) |grad| _ = grad.setAllScalar(1);
            self.g.compute();
        }

        fn step(self: *Self, lr: *Tensor(T), cur_batch: usize) void {
            // move set of xs into xsBatch
            const n_elems = self.xsBatch.nElems();
            @memcpy(self.xsBatch.data, self.xs.data[cur_batch * self.batch_size ..][0..n_elems]);
            // move set of ys into ysBatch
            @memcpy(self.ysBatch.data, self.ys.data[cur_batch * self.batch_size ..][0..n_elems]);
            for (self.params, self.momentum) |p, mom| {
                const p_grad = p.grad.?;
                mom.computeMul(mom, self.mu); // multiply the momentum term by the momentum constant
                p_grad.computeMul(p_grad, lr);
                mom.computeAdd(mom, p_grad); // add the gradient term to the momentum term
                p.computeSub(p, mom); // update the parameter by subtracting the momentum term
            }
            for (lr.data) |*lr_v| {
                lr_v.* *= 0.99;
            }
        }

        // TODO: uncomment when optimizer is implemented
        test "linear model" {
            // const time = try Tensor(T).initLinspace(tac, &.{20}, 0, 20);
            // const speed = try Tensor(T).initLinspace(tac, &.{20}, 5, 25);
            const n = 20;
            const time = try Tensor(T).initLinspace(tac, &.{n}, 0, 20);
            const true_m = 30;
            const speed = try Tensor(T).initLinspace(tac, &.{n}, 0, 20 * true_m);
            defer time.deinit();
            defer speed.deinit();

            var model = try Model(T).build(tac, 0, 0, time, speed, 1, 0.2);
            defer model.deinit();

            const lr = try Tensor(T).initScalar(tac, 1e-3);
            defer lr.deinit();

            const n_batches = time.nElems() / model.batch_size;
            for (0..10) |epoch| {
                _ = epoch;
                for (0..n_batches) |batch| {
                    model.compute();
                    model.step(lr, batch);
                }
            }
            try testing.expectApproxEqAbs(@as(T, true_m), model.params[0].data[0], 10e-1);
            // TODO: it's difficult to get this type of model to converge on a bias term, reproduced in pytorch with multiple optimizers
            // try testing.expectApproxEqAbs(@as(T, 0), model.params[1].data[0], 1e-2);
        }
    };
}
