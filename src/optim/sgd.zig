const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Alloc = std.mem.Allocator;

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
