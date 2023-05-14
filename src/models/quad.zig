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

        params: [3]*Tensor(T),
        g: ComputeGraph(T),
        out: *Tensor(T),
        loss: *Tensor(T),

        fn build(alloc: Alloc, a: T, b: T, c1: T, xs: *Tensor(T), ys: *Tensor(T)) !Self {
            var p = Self{
                // zig fmt: off
                .params = .{ 
                    try Tensor(T).initScalar(alloc, a), 
                    try Tensor(T).initScalar(alloc, b), 
                    try Tensor(T).initScalar(alloc, c1) 
                },
                // zig fmt: on
                .g = ComputeGraph(T).init(alloc),
                .out = undefined,
                .loss = undefined,
            };
            for (p.params) |param| {
                try param.setParam();
            }
            p.params[0].name = "a";
            p.params[1].name = "b";
            p.params[2].name = "c";
            const xsq = try xs.sqr(); // x^2
            xsq.name = "x^2";
            const axsq = try xsq.mul(try p.params[0].repeatLike(xsq)); // a*x^2
            axsq.name = "a*x^2";
            const bx = try xs.mul(try p.params[1].repeatLike(xs)); // b*x
            bx.name = "b*x";
            const axsqPlusBx = try axsq.add(bx); // a*x^2 + b*x
            axsqPlusBx.name = "a*x^2 + b*x";

            p.out = try axsqPlusBx.add(try p.params[2].repeatLike(xs)); // a*x^2 + b*x + c
            p.out.name = "a*x^2 + b*x + c";
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

        // TODO: write optimizer to make this work properly, uncomment when done
        // test "quadratic model" {
        //     const time = try Tensor(T).initLinspace(tac, &.{5}, 0, 20);
        //     const speed = try Tensor(T).initLinspace(tac, &.{5}, 5, 25);

        //     var model = try Model(T).build(tac, 1, 1, 1, time, speed);
        //     defer model.deinit();

        //     const lr = try Tensor(T).initScalar(tac, 1e-5);
        //     defer lr.deinit();

        //     for (0..100) |_| {
        //         model.compute();
        //         std.debug.print("a: {d}\n", .{model.params[0].data[0]});
        //         std.debug.print("b: {d}\n", .{model.params[1].data[0]});
        //         std.debug.print("c: {d}\n", .{model.params[2].data[0]});
        //         std.debug.print("loss: {d}\n", .{model.loss.data[0]});
        //         model.step(lr);
        //     }
        //     try testing.expectEqual(@as(T, 0), model.params[0].data[0]);
        //     try testing.expectEqual(@as(T, 1), model.params[1].data[0]);
        //     try testing.expectEqual(@as(T, 5), model.params[2].data[0]);
        // }
    };
}
