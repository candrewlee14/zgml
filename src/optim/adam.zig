//! Adam optimizer (Kingma & Ba, 2014).
//!
//! Maintains per-parameter first and second moment estimates with bias
//! correction. This is the standard optimizer for transformer training.
//!
//! Update rule:
//!   m = β₁ * m + (1 - β₁) * grad
//!   v = β₂ * v + (1 - β₂) * grad²
//!   m̂ = m / (1 - β₁ᵗ)
//!   v̂ = v / (1 - β₂ᵗ)
//!   param -= lr * m̂ / (√v̂ + ε)
//!
//! ```
//! var opt = try Adam(f32).init(alloc, &model.params(), loss, .{});
//! defer opt.deinit();
//! for (0..epochs) |_| {
//!     opt.zeroGrad();
//!     graph.compute();  // forward + backward
//!     opt.step();
//! }
//! ```

const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Alloc = std.mem.Allocator;
const tac = std.testing.allocator;
const models = @import("../models.zig");

pub const AdamConfig = struct {
    lr: f64 = 1e-3,
    beta1: f64 = 0.9,
    beta2: f64 = 0.999,
    eps: f64 = 1e-8,
    weight_decay: f64 = 0,
};

/// Adam optimizer with optional decoupled weight decay (AdamW).
pub fn Adam(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: Alloc,
        params: []const *Tensor(T),
        config: AdamConfig,
        t: usize, // timestep

        // Per-parameter state
        m: std.ArrayList(*Tensor(T)), // first moment (mean of gradients)
        v: std.ArrayList(*Tensor(T)), // second moment (mean of squared gradients)

        // Scratch tensors for intermediate computation (one per param)
        scratch: std.ArrayList(*Tensor(T)),

        pub fn init(
            alloc: Alloc,
            params: []const *Tensor(T),
            config: AdamConfig,
        ) Alloc.Error!Self {
            var res = Self{
                .alloc = alloc,
                .params = params,
                .config = config,
                .t = 0,
                .m = try std.ArrayList(*Tensor(T)).initCapacity(alloc, params.len),
                .v = try std.ArrayList(*Tensor(T)).initCapacity(alloc, params.len),
                .scratch = try std.ArrayList(*Tensor(T)).initCapacity(alloc, params.len),
            };
            for (params) |param| {
                const m_t = try Tensor(T).init(alloc, &param.ne);
                _ = m_t.setAllScalar(0);
                res.m.appendAssumeCapacity(m_t);

                const v_t = try Tensor(T).init(alloc, &param.ne);
                _ = v_t.setAllScalar(0);
                res.v.appendAssumeCapacity(v_t);

                const s = try Tensor(T).init(alloc, &param.ne);
                _ = s.setAllScalar(0);
                res.scratch.appendAssumeCapacity(s);
            }
            return res;
        }

        pub fn deinit(self: *Self) void {
            for (self.m.items, self.v.items, self.scratch.items) |m_t, v_t, s| {
                m_t.deinit();
                v_t.deinit();
                s.deinit();
            }
            self.m.deinit(self.alloc);
            self.v.deinit(self.alloc);
            self.scratch.deinit(self.alloc);
        }

        /// Perform one optimization step.
        pub fn step(self: *Self) void {
            self.t += 1;

            const beta1: T = @floatCast(self.config.beta1);
            const beta2: T = @floatCast(self.config.beta2);
            const lr: T = @floatCast(self.config.lr);
            const eps: T = @floatCast(self.config.eps);
            const wd: T = @floatCast(self.config.weight_decay);

            // Bias correction factors
            const beta1_t = std.math.pow(T, beta1, @floatFromInt(self.t));
            const beta2_t = std.math.pow(T, beta2, @floatFromInt(self.t));
            const bc1 = 1.0 / (1.0 - beta1_t);
            const bc2 = 1.0 / (1.0 - beta2_t);

            for (self.params, self.m.items, self.v.items, self.scratch.items) |param, m_t, v_t, scratch| {
                const grad = param.grad.?;
                const data = param.data;
                const m_data = m_t.data;
                const v_data = v_t.data;

                // Fused update: avoids creating temporary tensors
                for (data, grad.data, m_data, v_data, scratch.data) |*p, g, *m, *v, *s| {
                    // m = β₁ * m + (1 - β₁) * g
                    m.* = beta1 * m.* + (1.0 - beta1) * g;
                    // v = β₂ * v + (1 - β₂) * g²
                    v.* = beta2 * v.* + (1.0 - beta2) * g * g;
                    // m̂ = m / (1 - β₁ᵗ),  v̂ = v / (1 - β₂ᵗ)
                    const m_hat = m.* * bc1;
                    const v_hat = v.* * bc2;
                    // param -= lr * m̂ / (√v̂ + ε)
                    const update = lr * m_hat / (@sqrt(v_hat) + eps);
                    p.* -= update;
                    // Decoupled weight decay (AdamW)
                    if (wd > 0) {
                        p.* -= wd * lr * p.*;
                    }
                    _ = s;
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
const ComputeGraph = @import("../graph.zig").ComputeGraph;

test "adam - basic convergence on quadratic" {
    // Minimize f(x) = sum(x^2) starting from x = [3, 4, 5]
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 3.0, 4.0, 5.0 });
    x.setParam();

    const loss = x.sqr().sumAll();
    try g.buildForward(loss);
    try g.buildBackward(false);

    var opt = try Adam(f32).init(tac, &.{x}, .{ .lr = 0.1 });
    defer opt.deinit();

    for (0..200) |_| {
        opt.zeroGrad();
        _ = loss.grad.?.setAllScalar(1);
        g.compute();
        opt.step();
    }

    // Should converge close to zero
    for (x.data) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v, 0.05);
    }
}

test "adam - weight decay pushes params toward zero" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{2});
    x.setData(&.{ 10.0, -10.0 });
    x.setParam();

    // Loss = sum(x) — gradient is constant 1, so weight decay dominates
    const loss = x.sumAll();
    try g.buildForward(loss);
    try g.buildBackward(false);

    var opt = try Adam(f32).init(tac, &.{x}, .{ .lr = 0.01, .weight_decay = 0.1 });
    defer opt.deinit();

    const initial_norm = @abs(x.data[0]) + @abs(x.data[1]);

    for (0..100) |_| {
        opt.zeroGrad();
        _ = loss.grad.?.setAllScalar(1);
        g.compute();
        opt.step();
    }

    const final_norm = @abs(x.data[0]) + @abs(x.data[1]);
    try std.testing.expect(final_norm < initial_norm);
}

test "adam - implements optimizer interface" {
    const Optimizer = Adam(f32);
    try std.testing.expect(std.meta.hasFn(Optimizer, "init"));
    try std.testing.expect(std.meta.hasFn(Optimizer, "deinit"));
    try std.testing.expect(std.meta.hasFn(Optimizer, "step"));
    try std.testing.expect(std.meta.hasFn(Optimizer, "zeroGrad"));
}
