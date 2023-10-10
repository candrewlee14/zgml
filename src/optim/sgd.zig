const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Alloc = std.mem.Allocator;
const tac = std.testing.allocator;
const models = @import("../models.zig");
const graph = @import("../graph.zig");

/// Stochastic Gradient Descent optimizer
/// Uses momentum and mini-batches
pub fn SGD(comptime T: type) type {
    return struct {
        const Self = @This();

        arena: std.heap.ArenaAllocator,
        params: []const *Tensor(T),
        learning_rate: *Tensor(T),
        momentum: []*Tensor(T),
        momentum_decay: *Tensor(T),
        graphs: []graph.ComputeGraph(T),

        pub fn init(
            self: *Self,
            alloc: Alloc,
            params: []const *Tensor(T),
            learning_rate: T,
            momentum_decay: T,
        ) Alloc.Error!void {
            var arena = std.heap.ArenaAllocator.init(alloc);
            self.arena = arena;
            const al = self.arena.allocator();
            self.* = Self{
                .arena = self.arena,
                .params = params,
                .learning_rate = try Tensor(T).initScalar(al, learning_rate),
                .momentum = try al.alloc(*Tensor(T), params.len),
                .momentum_decay = try Tensor(T).initScalar(al, momentum_decay),
                .graphs = try al.alloc(graph.ComputeGraph(T), params.len),
            };
            for (params, 0..) |orig_param, i| {
                var param = orig_param.detachedView(al);
                var momentum = try Tensor(T).zeros(al, &orig_param.ne);

                momentum = momentum.mulInplace(self.momentum_decay);
                momentum = momentum.subInplace(param.grad.?.mul(self.learning_rate));

                param = param.addInplace(momentum);

                self.momentum[i] = momentum;
                self.graphs[i] = graph.ComputeGraph(T).init(al);
                try self.graphs[i].buildForward(param);
            }
        }
        pub fn deinit(self: *Self) void {
            self.arena.deinit();
        }
        // Must zero grad before calling step
        pub fn step(self: *Self) void {
            for (self.graphs) |*g| {
                g.computeNoGrad();
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
    std.log.warn("Setup model\n", .{});

    var optimizer: SGD(T) = undefined;
    try optimizer.init(tac, &model.params, 1e-3, 0.2);
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
//     var optimizer = try SGD(T).init(tac, &model.params, 1e-3, 0.2);
//     defer optimizer.deinit();
//     model.train(time, speed, 10, 1, &optimizer);
//     try std.testing.expectApproxEqAbs(@as(T, 4), model.params[0].data[0], 5e-1);
//     try std.testing.expectApproxEqAbs(@as(T, 3), model.params[1].data[0], 5e-1);
// }
