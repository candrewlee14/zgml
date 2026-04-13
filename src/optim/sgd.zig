const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Alloc = std.mem.Allocator;
const tac = std.testing.allocator;
const models = @import("../models.zig");

pub const SgdConfig = struct {
    lr: f64 = 0.01,
    momentum: f64 = 0.0,
};

/// Stochastic Gradient Descent optimizer with momentum.
///
/// ```
/// var opt = try SGD(f32).init(alloc, &params, .{ .lr = 0.1, .momentum = 0.9 });
/// defer opt.deinit();
/// opt.zeroGrad();
/// g.run(loss);
/// opt.step();
/// ```
pub fn SGD(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: Alloc,
        params: []const *Tensor(T),
        lr: T,
        momentum: std.ArrayList(*Tensor(T)),
        decay: T,

        pub fn init(
            alloc: Alloc,
            params: []const *Tensor(T),
            config: SgdConfig,
        ) Alloc.Error!Self {
            var res = Self{
                .alloc = alloc,
                .params = params,
                .lr = @floatCast(config.lr),
                .momentum = try std.ArrayList(*Tensor(T)).initCapacity(alloc, params.len),
                .decay = @floatCast(config.momentum),
            };
            for (params) |param| {
                const mo = try Tensor(T).init(alloc, &param.ne);
                _ = mo.setAllScalar(0);
                res.momentum.appendAssumeCapacity(mo);
            }
            return res;
        }
        pub fn deinit(self: *Self) void {
            for (self.momentum.items) |mo| {
                mo.deinit();
            }
            self.momentum.deinit(self.alloc);
        }
        /// Perform one optimization step.
        ///
        /// Update rule with momentum:
        ///   momentum = momentum * decay + grad * lr
        ///   param   -= momentum
        ///
        /// Call `zeroGrad()` before the next forward pass.
        pub fn step(self: *Self) void {
            for (self.params, self.momentum.items) |param, mo| {
                const grad_data = param.grad.?.data;
                for (mo.data, param.data, grad_data) |*m, *p, g| {
                    m.* = m.* * self.decay + g * self.lr;
                    p.* -= m.*;
                }
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
    defer time.deinit();
    defer speed.deinit();

    var model = try models.Linear(T).build(tac, 0, 0, 5);
    defer model.deinit();

    const nn = @import("../nn.zig");
    const p = model.params();
    var optimizer = try SGD(T).init(tac, &p, .{ .lr = 1e-3, .momentum = 0.2 });
    defer optimizer.deinit();
    try nn.trainSupervised(T, &model.g, model.loss, model.xs_batch, model.ys_batch, time, speed, 10, &optimizer);
    try std.testing.expectApproxEqAbs(@as(T, true_m), model.m.data[0], 5e-1);
}
