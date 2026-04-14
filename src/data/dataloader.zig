//! Generic batched data loader with batch-level shuffling.
//!
//! Shuffles the order in which contiguous batches are served each epoch.
//! Each `next()` call copies one batch into caller-provided destination slices
//! with a single `@memcpy` per array — optimal for cache and prefetch.
//!
//! ```
//! var loader = try DataLoader(f32).init(alloc, ds.images, ds.labels, ds.n_samples, 32, 42);
//! defer loader.deinit();
//! for (0..n_epochs) |_| {
//!     loader.shuffle();
//!     while (loader.next(model.xs_batch.data, model.ys_batch.data)) {
//!         try g.run(loss_node);
//!         optimizer.step();
//!     }
//! }
//! ```

const std = @import("std");
const Alloc = std.mem.Allocator;

pub fn DataLoader(comptime T: type) type {
    return struct {
        const Self = @This();

        xs: []const T,
        ys: ?[]const T,
        x_batch_len: usize,
        y_batch_len: usize,
        n_batches: usize,
        indices: []usize, // batch-level, length = n_batches
        rng: std.Random.DefaultPrng,
        alloc: Alloc,
        pos: usize,

        /// Create a data loader over `n_samples` samples with the given `batch_size`.
        ///
        /// Per-sample element counts are inferred: `sample_x_len = xs.len / n_samples`.
        /// The last partial batch (if any) is dropped.
        pub fn init(
            alloc: Alloc,
            xs: []const T,
            ys: ?[]const T,
            n_samples: usize,
            batch_size: usize,
            seed: u64,
        ) Alloc.Error!Self {
            std.debug.assert(xs.len > 0 and n_samples > 0 and batch_size > 0);
            std.debug.assert(xs.len % n_samples == 0);
            std.debug.assert(batch_size <= n_samples);

            const sample_x_len = xs.len / n_samples;
            const n_batches = n_samples / batch_size;

            const y_batch_len: usize = if (ys) |y| blk: {
                std.debug.assert(y.len % n_samples == 0);
                break :blk (y.len / n_samples) * batch_size;
            } else 0;

            const indices = try alloc.alloc(usize, n_batches);
            for (indices, 0..) |*idx, i| idx.* = i;

            return .{
                .xs = xs,
                .ys = ys,
                .x_batch_len = sample_x_len * batch_size,
                .y_batch_len = y_batch_len,
                .n_batches = n_batches,
                .indices = indices,
                .rng = std.Random.DefaultPrng.init(seed),
                .alloc = alloc,
                .pos = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            self.alloc.free(self.indices);
        }

        /// Shuffle batch order and reset the iterator for a new epoch.
        pub fn shuffle(self: *Self) void {
            self.rng.random().shuffle(usize, self.indices);
            self.pos = 0;
        }

        /// Reset the iterator to the beginning without shuffling.
        pub fn reset(self: *Self) void {
            self.pos = 0;
        }

        /// Copy the next batch into `xs_dst` (and `ys_dst` if labels exist).
        ///
        /// Returns `true` if a batch was written, `false` when the epoch is done.
        pub fn next(self: *Self, xs_dst: []T, ys_dst: ?[]T) bool {
            if (self.pos >= self.n_batches) return false;
            const b = self.indices[self.pos];

            std.debug.assert(xs_dst.len == self.x_batch_len);
            @memcpy(xs_dst, self.xs[b * self.x_batch_len ..][0..self.x_batch_len]);

            if (self.ys) |y_src| {
                const yd = ys_dst.?;
                std.debug.assert(yd.len == self.y_batch_len);
                @memcpy(yd, y_src[b * self.y_batch_len ..][0..self.y_batch_len]);
            }

            self.pos += 1;
            return true;
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "dataloader - sequential batching" {
    const xs = [_]f32{ 10, 20, 30, 40 };
    const ys = [_]f32{ 1, 2, 3, 4 };
    var xs_dst: [2]f32 = undefined;
    var ys_dst: [2]f32 = undefined;

    var loader = try DataLoader(f32).init(testing.allocator, &xs, &ys, 4, 2, 42);
    defer loader.deinit();

    try testing.expectEqual(@as(usize, 2), loader.n_batches);

    try testing.expect(loader.next(&xs_dst, &ys_dst));
    try testing.expectEqualSlices(f32, &.{ 10, 20 }, &xs_dst);
    try testing.expectEqualSlices(f32, &.{ 1, 2 }, &ys_dst);

    try testing.expect(loader.next(&xs_dst, &ys_dst));
    try testing.expectEqualSlices(f32, &.{ 30, 40 }, &xs_dst);
    try testing.expectEqualSlices(f32, &.{ 3, 4 }, &ys_dst);

    try testing.expect(!loader.next(&xs_dst, &ys_dst));
}

test "dataloader - multi-element samples" {
    // 3 samples, sample_x_len=2, sample_y_len=1, batch_size=1
    const xs = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const ys = [_]f32{ 10, 20, 30 };
    var xs_dst: [2]f32 = undefined;
    var ys_dst: [1]f32 = undefined;

    var loader = try DataLoader(f32).init(testing.allocator, &xs, &ys, 3, 1, 42);
    defer loader.deinit();

    try testing.expectEqual(@as(usize, 3), loader.n_batches);

    try testing.expect(loader.next(&xs_dst, &ys_dst));
    try testing.expectEqualSlices(f32, &.{ 1, 2 }, &xs_dst);
    try testing.expectEqualSlices(f32, &.{10}, &ys_dst);
}

test "dataloader - shuffle changes batch order" {
    const xs = [_]f32{ 10, 20, 30, 40 };
    const ys = [_]f32{ 1, 2, 3, 4 };
    var xs_dst: [1]f32 = undefined;
    var ys_dst: [1]f32 = undefined;

    var loader = try DataLoader(f32).init(testing.allocator, &xs, &ys, 4, 1, 12345);
    defer loader.deinit();

    loader.shuffle();

    // Collect all values across the epoch
    var seen = [_]bool{false} ** 4;
    while (loader.next(&xs_dst, &ys_dst)) {
        const idx = @as(usize, @intFromFloat(xs_dst[0] / 10.0)) - 1;
        seen[idx] = true;
        // Verify x/y pairing is preserved
        try testing.expectApproxEqAbs(xs_dst[0], ys_dst[0] * 10.0, 1e-6);
    }
    for (seen) |s| try testing.expect(s);
}

test "dataloader - shuffle auto-resets" {
    const xs = [_]f32{ 1, 2, 3, 4 };
    var xs_dst: [2]f32 = undefined;

    var loader = try DataLoader(f32).init(testing.allocator, &xs, null, 4, 2, 42);
    defer loader.deinit();

    while (loader.next(&xs_dst, null)) {}
    try testing.expect(!loader.next(&xs_dst, null));

    loader.shuffle();
    try testing.expect(loader.next(&xs_dst, null));
}

test "dataloader - reset without shuffle" {
    const xs = [_]f32{ 10, 20 };
    var xs_dst: [1]f32 = undefined;

    var loader = try DataLoader(f32).init(testing.allocator, &xs, null, 2, 1, 42);
    defer loader.deinit();

    try testing.expect(loader.next(&xs_dst, null));
    try testing.expectEqualSlices(f32, &.{10}, &xs_dst);

    // Exhaust
    while (loader.next(&xs_dst, null)) {}

    // Reset replays same order
    loader.reset();
    try testing.expect(loader.next(&xs_dst, null));
    try testing.expectEqualSlices(f32, &.{10}, &xs_dst);
}

test "dataloader - unsupervised (ys = null)" {
    const xs = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var xs_dst: [4]f32 = undefined;

    var loader = try DataLoader(f32).init(testing.allocator, &xs, null, 3, 2, 42);
    defer loader.deinit();

    try testing.expectEqual(@as(usize, 1), loader.n_batches);
    try testing.expect(loader.next(&xs_dst, null));
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, &xs_dst);
    try testing.expect(!loader.next(&xs_dst, null));
}

test "dataloader - drops partial last batch" {
    // 5 samples, batch_size=2 -> 2 full batches, 1 dropped
    const xs = [_]f32{ 1, 2, 3, 4, 5 };
    var xs_dst: [2]f32 = undefined;

    var loader = try DataLoader(f32).init(testing.allocator, &xs, null, 5, 2, 42);
    defer loader.deinit();

    try testing.expectEqual(@as(usize, 2), loader.n_batches);
    var count: usize = 0;
    while (loader.next(&xs_dst, null)) count += 1;
    try testing.expectEqual(@as(usize, 2), count);
}

test "dataloader - batch_size equals n_samples" {
    const xs = [_]f32{ 1, 2, 3 };
    const ys = [_]f32{ 10, 20, 30 };
    var xs_dst: [3]f32 = undefined;
    var ys_dst: [3]f32 = undefined;

    var loader = try DataLoader(f32).init(testing.allocator, &xs, &ys, 3, 3, 42);
    defer loader.deinit();

    try testing.expectEqual(@as(usize, 1), loader.n_batches);
    try testing.expect(loader.next(&xs_dst, &ys_dst));
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, &xs_dst);
    try testing.expectEqualSlices(f32, &.{ 10, 20, 30 }, &ys_dst);
    try testing.expect(!loader.next(&xs_dst, &ys_dst));
}

test "dataloader - multi-element labels" {
    // 4 samples, sample_x_len=1, sample_y_len=2 (one-hot)
    const xs = [_]f32{ 1, 2, 3, 4 };
    const ys = [_]f32{ 1, 0, 0, 1, 1, 0, 0, 1 }; // 4 * 2
    var xs_dst: [2]f32 = undefined; // batch_size=2
    var ys_dst: [4]f32 = undefined; // batch_size=2 * sample_y_len=2

    var loader = try DataLoader(f32).init(testing.allocator, &xs, &ys, 4, 2, 42);
    defer loader.deinit();

    try testing.expect(loader.next(&xs_dst, &ys_dst));
    try testing.expectEqualSlices(f32, &.{ 1, 2 }, &xs_dst);
    try testing.expectEqualSlices(f32, &.{ 1, 0, 0, 1 }, &ys_dst);
}
