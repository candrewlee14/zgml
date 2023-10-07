const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Alloc = std.mem.Allocator;
const tac = std.testing.allocator;
const models = @import("../models.zig");

/// Stochastic Gradient Descent optimizer
/// Uses momentum and mini-batches
pub fn SGD(comptime T: type) type {
    return struct {
        const Self = @This();

        params: []const *Tensor(T),
        batch_size: usize,
        learning_rate: *Tensor(T),
        momentum: std.ArrayList(*Tensor(T)),
        lr_grad: std.ArrayList(*Tensor(T)),
        momentum_decay: *Tensor(T),
        loss: *Tensor(T),

        pub fn init(
            alloc: Alloc,
            params: []const *Tensor(T),
            batch_size: usize,
            loss: *Tensor(T),
            learning_rate: T,
            momentum_decay: T,
        ) Alloc.Error!Self {
            var res = Self{
                .batch_size = batch_size,
                .params = params,
                .learning_rate = try Tensor(T).initScalar(alloc, learning_rate),
                .loss = loss,
                .momentum = try std.ArrayList(*Tensor(T)).initCapacity(alloc, params.len),
                .lr_grad = try std.ArrayList(*Tensor(T)).initCapacity(alloc, params.len),
                .momentum_decay = try Tensor(T).initScalar(alloc, momentum_decay),
            };
            for (params) |param| {
                const mo = try Tensor(T).init(alloc, &param.ne);
                _ = mo.setAllScalar(0);
                res.momentum.appendAssumeCapacity(mo);

                const lr_grad = try Tensor(T).init(alloc, &param.ne);
                _ = lr_grad.setAllScalar(0);
                res.lr_grad.appendAssumeCapacity(lr_grad);
            }
            return res;
        }
        pub fn deinit(self: *Self) void {
            self.learning_rate.deinit();
            for (self.momentum.items, self.lr_grad.items) |mo, lrg| {
                mo.deinit();
                lrg.deinit();
            }
            self.momentum.deinit();
            self.lr_grad.deinit();
            self.momentum_decay.deinit();
        }
        // Must zero grad before calling step
        pub fn step(self: *Self) void {
            for (self.params, self.momentum.items, self.lr_grad.items) |param, mo, lrg| {
                mo.computeMul(mo, self.momentum_decay);
                lrg.computeMul(param.grad.?, self.learning_rate);
                mo.computeAdd(mo, lrg);
                param.computeSub(param, mo);
            }
        }
        pub fn zeroGrad(self: *Self) void {
            for (self.params) |param| {
                _ = param.grad.?.setAllScalar(0);
            }
        }
    };
}

test "optim - linear model with sgd optim" {
    const T = f32;
    const n = 100;
    const time = try Tensor(T).linspace(tac, 0, 20, &.{n});
    const true_m: T = 13.5;
    const speed = try Tensor(T).linspace(tac, 0, 20 * true_m, &.{n});
    defer time.deinit();
    defer speed.deinit();

    var model = try models.Linear(T).build(tac, 0, 0, 5);
    defer model.deinit();

    var optimizer = try SGD(T).init(tac, &model.params, 1, model.loss, 1e-3, 0.2);
    defer optimizer.deinit();
    model.train(time, speed, 10, 1, &optimizer);
    try std.testing.expectApproxEqAbs(@as(T, true_m), model.params[0].data[0], 5e-1);
}

// TODO: get this to pass, bias should be found correctly
//
// test "optim linear model with sgd, y = 4x + 3" {
//     const T = f32;
//     const n = 100;
//     const time = try Tensor(T).linspace(tac, 0, 20, &.{n});
//     var speed = try Tensor(T).linspace(tac, 0, 20 * 4, &.{n});
//     for (speed.data) |*d| {
//         d.* += 3;
//     }
//     defer time.deinit();
//     defer speed.deinit();
//
//     var model = try models.Linear(T).build(tac, -0.5, -0.5, 5);
//     defer model.deinit();
//
//     var optimizer = try SGD(T).init(tac, &model.params, 1, model.loss, 1e-3, 0.2);
//     defer optimizer.deinit();
//     model.train(time, speed, 10, 1, &optimizer);
//     try std.testing.expectApproxEqAbs(@as(T, 4), model.params[0].data[0], 5e-1);
//     try std.testing.expectApproxEqAbs(@as(T, 3), model.params[1].data[0], 5e-1);
// }
