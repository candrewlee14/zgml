//! Autoencoder: learns to compress and reconstruct data.
//!
//! Architecture: 8 → 3 (gelu) → 8, trained with MSE reconstruction loss.
//! Demonstrates unsupervised learning with a bottleneck representation.

const std = @import("std");
const testing = std.testing;
const tac = testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Alloc = std.mem.Allocator;
const loss_mod = @import("../loss.zig");
const optim = @import("../optim.zig");
const nn = @import("../nn.zig");

pub fn Model(comptime T: type) type {
    return struct {
        const Self = @This();

        w_enc: *Tensor(T),
        b_enc: *Tensor(T),
        w_dec: *Tensor(T),
        b_dec: *Tensor(T),
        xs_batch: *Tensor(T),
        encoded: *Tensor(T),
        decoded: *Tensor(T),
        loss: *Tensor(T),
        g: ComputeGraph(T),

        pub fn build(backing_alloc: Alloc, batch_size: usize) !Self {
            var g = ComputeGraph(T).init(backing_alloc);
            const a = g.allocator();

            const w_enc = try g.param(&.{ 3, 8 });
            const b_enc = try g.param(&.{3});
            const w_dec = try g.param(&.{ 8, 3 });
            const b_dec = try g.param(&.{8});

            nn.kaimingUniform(T, w_enc, 42);
            nn.kaimingUniform(T, w_dec, 123);

            const xs_batch = try Tensor(T).init(a, &.{ 8, batch_size });

            const encoded = nn.linear(T, xs_batch, w_enc, b_enc).gelu();
            const decoded = nn.linear(T, encoded, w_dec, b_dec);
            const loss = loss_mod.meanSqErr(T, decoded, xs_batch);

            try g.buildForward(loss);
            try g.buildBackward(true);

            return .{
                .w_enc = w_enc, .b_enc = b_enc,
                .w_dec = w_dec, .b_dec = b_dec,
                .xs_batch = xs_batch, .encoded = encoded,
                .decoded = decoded, .loss = loss, .g = g,
            };
        }

        pub fn params(self: *const Self) [4]*Tensor(T) {
            return .{ self.w_enc, self.b_enc, self.w_dec, self.b_dec };
        }

        pub fn deinit(self: *Self) void {
            self.g.deinit();
        }

        test "autoencoder learns to reconstruct structured data" {
            const n_samples = 8;
            const xs = try Tensor(T).init(tac, &.{ 8, n_samples });
            // zig fmt: off
            xs.setData(&.{
                0.5, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.0,
                0.0, 0.5, 0.5,-0.5, 0.0, 1.0, 0.0, 0.5,
                0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 0.5, 0.5,
                0.5,-0.5, 0.0, 1.0, 1.0,-1.0, 0.5,-0.5,
                1.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 1.0,-1.0, 0.0, 2.0, 0.0, 1.0,
                1.0, 0.5, 1.5, 0.5, 2.0, 1.0, 1.0, 0.5,
                0.5, 1.0, 1.5,-0.5, 1.0, 2.0, 0.5, 1.0,
            });
            // zig fmt: on
            defer xs.deinit();

            var model = try Model(T).build(tac, n_samples);
            defer model.deinit();

            @memcpy(model.xs_batch.data, xs.data);
            try model.g.run(model.loss);
            const initial_loss = model.loss.data[0];

            const p = model.params();
            var optimizer = try optim.sgd.SGD(T).init(tac, &p, .{ .lr = 0.005, .momentum = 0.1 });
            defer optimizer.deinit();
            try nn.trainUnsupervised(T, &model.g, model.loss, model.xs_batch, xs, 500, &optimizer);

            @memcpy(model.xs_batch.data, xs.data);
            try model.g.infer(model.decoded);
            const final_loss = model.loss.data[0];

            try testing.expect(final_loss < initial_loss * 0.5);

            for (model.decoded.data, xs.data) |reconstructed, original| {
                try testing.expectApproxEqAbs(original, reconstructed, 2.0);
            }
        }
    };
}
