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

        params: [2]*Tensor(T),
        g: ComputeGraph(T),
        out: *Tensor(T),
        loss: *Tensor(T),

        fn build(alloc: Alloc, m: T, b: T, xs: *Tensor(T), ys: *Tensor(T)) !Self {
            var p = Self{
                // zig fmt: off
                .params = .{ 
                    try Tensor(T).initScalar(alloc, m), 
                    try Tensor(T).initScalar(alloc, b), 
                },
                // zig fmt: on
                .g = ComputeGraph(T).init(alloc),
                .out = undefined,
                .loss = undefined,
            };
            for (p.params) |param| {
                try param.setParam();
            }
            p.params[0].name = "m";
            p.params[1].name = "b";
            const mx = try xs.mul(try p.params[0].repeatLike(xs));
            mx.name = "m*x";
            const mxPlusB = try mx.add(try p.params[1].repeatLike(xs));
            mxPlusB.name = "m*x+b";
            p.out = mxPlusB;
            p.loss = try loss.meanSqErr(T, p.out, ys);
            p.loss.name = "loss";
            try p.g.buildForward(p.loss);
            try p.g.buildBackward(true);
            return p;
        }

        fn deinit(self: *Self) void {
            self.g.deinit();

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

        fn step(self: *Self, lr: *Tensor(T)) void {
            for (self.params) |p| {
                const p_grad = p.grad.?;
                p_grad.computeMul(p_grad, lr);
                p.computeSub(p, p_grad);
            }
            for (lr.data) |*lr_v| {
                lr_v.* *= 0.99;
            }
        }

        // TODO: uncomment when optimizer is implemented
        // test "linear model" {
        //     const time = try Tensor(T).initArange(tac, &.{5}, 0, 20);
        //     const speed = try Tensor(T).initArange(tac, &.{5}, 5, 25);

        //     var model = try Model(T).build(tac, 2, 1, time, speed);
        //     defer model.deinit();

        //     const lr = try Tensor(T).initScalar(tac, 1e-3);
        //     defer lr.deinit();

        //     for (0..100) |_| {
        //         model.compute();
        //         std.debug.print("m: {d}\n", .{model.params[0].data[0]});
        //         std.debug.print("b: {d}\n", .{model.params[1].data[0]});
        //         std.debug.print("loss: {d}\n", .{model.loss.data[0]});
        //         model.step(lr);
        //     }
        //     try testing.expectEqual(@as(T, 1), model.params[0].data[0]);
        //     try testing.expectEqual(@as(T, 5), model.params[1].data[0]);
        // }
    };
}
