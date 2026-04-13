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

        alloc: Alloc,
        params: []const *Tensor(T),
        learning_rate: *Tensor(T),
        momentum: std.ArrayList(*Tensor(T)),
        lr_grad: std.ArrayList(*Tensor(T)),
        momentum_decay: *Tensor(T),
        loss: *Tensor(T),

        pub fn init(
            alloc: Alloc,
            params: []const *Tensor(T),
            loss: *Tensor(T),
            learning_rate: T,
            momentum_decay: T,
        ) Alloc.Error!Self {
            var res = Self{
                .alloc = alloc,
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

                const lr_grad_t = try Tensor(T).init(alloc, &param.ne);
                _ = lr_grad_t.setAllScalar(0);
                res.lr_grad.appendAssumeCapacity(lr_grad_t);
            }
            return res;
        }
        pub fn deinit(self: *Self) void {
            self.learning_rate.deinit(self.alloc);
            for (self.momentum.items, self.lr_grad.items) |mo, lrg| {
                mo.deinit(self.alloc);
                lrg.deinit(self.alloc);
            }
            self.momentum.deinit(self.alloc);
            self.lr_grad.deinit(self.alloc);
            self.momentum_decay.deinit(self.alloc);
        }
        /// Perform one optimization step.
        ///
        /// Update rule with momentum:
        ///   momentum = momentum * decay + grad * lr
        ///   param   -= momentum
        ///
        /// Call `zeroGrad()` before the next forward pass.
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
    const time = try Tensor(T).initLinspace(tac, &.{n}, 0, 20);
    const true_m: T = 13.5;
    const speed = try Tensor(T).initLinspace(tac, &.{n}, 0, 20 * true_m);
    defer time.deinit(tac);
    defer speed.deinit(tac);

    var model = try models.Linear(T).build(tac, 0, 0, 5);
    defer model.deinit();

    var optimizer = try SGD(T).init(tac, &model.params, model.loss, 1e-3, 0.2);
    defer optimizer.deinit();
    model.train(time, speed, 10, &optimizer);
    try std.testing.expectApproxEqAbs(@as(T, true_m), model.params[0].data[0], 5e-1);
}
