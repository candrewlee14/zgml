const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Alloc = std.mem.Allocator;
const tac = std.testing.allocator;
const models = @import("../models.zig");
const graph = @import("../graph.zig");

/// Stochastic Gradient Descent optimizer
/// Uses momentum and mini-batches
pub fn SGDMomentum(comptime T: type) type {
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

/// Stochastic Gradient Descent optimizer
/// Uses mini-batches
pub fn SGD(comptime T: type) type {
    return struct {
        const Self = @This();

        arena: std.heap.ArenaAllocator,
        params: []const *Tensor(T),
        learning_rate: *Tensor(T),
        graphs: []graph.ComputeGraph(T),

        pub fn init(
            self: *Self,
            alloc: Alloc,
            params: []const *Tensor(T),
            learning_rate: T,
        ) Alloc.Error!void {
            var arena = std.heap.ArenaAllocator.init(alloc);
            self.arena = arena;
            const al = self.arena.allocator();
            self.* = Self{
                .arena = self.arena,
                .params = params,
                .learning_rate = try Tensor(T).initScalar(al, learning_rate),
                .graphs = try al.alloc(graph.ComputeGraph(T), params.len),
            };
            for (params, 0..) |orig_param, i| {
                var param = orig_param.detachedView(al);

                param = param.subInplace(param.grad.?.mul(self.learning_rate));

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

    var optimizer: SGDMomentum(T) = undefined;
    try optimizer.init(tac, &model.params, 1e-3, 0.2);
    defer optimizer.deinit();
    model.train(time, speed, 10, 1, &optimizer);
    try std.testing.expectApproxEqAbs(@as(T, true_m), model.params[0].data[0], 5e-1);
}

test "optim linear model with sgd, y = 4x + 3" {
    const T = f32;
    const n = 100;
    const true_m: T = 4;
    const true_b: T = 3;
    const time = try Tensor(T).linspace(tac, 0, 20, &.{n});
    var speed = try Tensor(T).linspace(tac, true_b, true_b + 20 * true_m, &.{n});
    defer time.deinit();
    defer speed.deinit();

    var model = try models.Linear(T).build(tac, -0.5, -0.5, 4);
    defer model.deinit();

    var optimizer: SGDMomentum(T) = undefined;
    try optimizer.init(tac, &model.params, 1e-3, 0.2);
    defer optimizer.deinit();

    model.train(time, speed, 100, 1, &optimizer);

    try std.testing.expectApproxEqAbs(true_m, model.params[0].data[0], 3e-1);
    try std.testing.expectApproxEqAbs(true_b, model.params[1].data[0], 3e-1);
}

// TODO: get this to pass, this shouldn't blow up to Nan values.
// Seems like there's some batch-dependent math going on that isn't working correctly.
// test "optim linear model with sgd, y = 4x + 3, big batch" {
//     const T = f32;
//     const n = 100;
//     const batch_size = 10;
//     const true_m: T = 4;
//     const true_b: T = 3;
//     const time = try Tensor(T).linspace(tac, 0, 20, &.{n});
//     var speed = try Tensor(T).linspace(tac, true_b, true_b + 20 * true_m, &.{n});
//     defer time.deinit();
//     defer speed.deinit();

//     var model = try models.Linear(T).build(tac, -0.5, -0.5, batch_size);
//     defer model.deinit();

//     var optimizer: SGDMomentum(T) = undefined;
//     try optimizer.init(tac, &model.params, 1e-3, 0.2);
//     defer optimizer.deinit();

//     model.train(time, speed, 100, 1, &optimizer);

//     try std.testing.expectApproxEqAbs(true_m, model.params[0].data[0], 3e-1);
//     try std.testing.expectApproxEqAbs(true_b, model.params[1].data[0], 3e-1);
// }

test "autograd SGD vs SGD model equivalent" {
    const T = f32;
    const n = 100;
    _ = n;
    const true_m: T = 4;
    _ = true_m;
    const true_b: T = 3;
    _ = true_b;

    var model = try models.Linear(T).build(tac, 0.1, -0.1, 1);
    defer model.deinit();

    const lr = 0.03;
    const iters = 1000;

    var optimizer: SGD(T) = undefined;
    try optimizer.init(tac, &model.params, lr);
    defer optimizer.deinit();

    var arena = std.heap.ArenaAllocator.init(tac);
    defer arena.deinit();
    const al = arena.allocator();

    const true_m_t = try Tensor(T).initScalar(al, 3);
    const true_b_t = try Tensor(T).initScalar(al, 4);

    const m = try Tensor(T).initScalar(al, 0.1);
    m.setParam();
    const b = try Tensor(T).initScalar(al, -0.1);
    b.setParam();

    const xs = try Tensor(T).arange(al, 10);

    // setup y = 3x+4
    const ys = try Tensor(T).arange(al, 10);
    ys.computeMul(ys, true_m_t);
    ys.computeAdd(ys, true_b_t);
    for (xs.data, ys.data) |x, y| {
        try std.testing.expectEqual(3 * x + 4, y);
    }

    // setup compute graph
    const xi = try Tensor(T).initScalar(al, 0);
    const yi = try Tensor(T).initScalar(al, 0);
    const y_pred = xi.mul(m).add(b);
    const loss = (y_pred.sub(yi)).sqr();

    var g = graph.ComputeGraph(T).init(al);
    try g.buildForward(loss);
    try g.buildBackward(true);

    for (0..iters) |i| {
        // autograd style
        g.resetGrads();
        const idx = i % xs.nElems();
        @memcpy(xi.data, xs.data[idx * xi.nElems() ..][0..1]);
        @memcpy(yi.data, ys.data[idx * yi.nElems() ..][0..1]);
        _ = loss.grad.?.setAllScalar(1);
        _ = m.grad.?.setAllScalar(0);
        _ = b.grad.?.setAllScalar(0);
        g.compute();
        m.data[0] -= lr * m.grad.?.data[0];
        b.data[0] -= lr * b.grad.?.data[0];

        // model style
        @memcpy(model.xs_batch.data, xs.data[idx * xi.nElems() ..][0..1]);
        @memcpy(model.ys_batch.data, ys.data[idx * xi.nElems() ..][0..1]);
        optimizer.zeroGrad();
        model.compute();
        optimizer.step();

        if (i % 100 == 0 or i < 10) {
            try std.testing.expectApproxEqAbs(m.data[0], model.params[0].data[0], 1e-5);
            try std.testing.expectApproxEqAbs(b.data[0], model.params[1].data[0], 1e-5);
        }
    }
}
