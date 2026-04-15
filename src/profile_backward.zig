//! Per-op backward pass profiler.
//!
//! Runs one forward+backward pass on the ConvClassifier (WITHOUT fusion),
//! timing each backward node individually. Groups by op tag and prints
//! a summary sorted by total time descending.
//!
//! This deliberately skips fusionPass() so every primitive op is timed
//! separately — the goal is to find where raw compute time goes before
//! fusion hides it.
//!
//! Run with: zig build profile-bwd  (after adding build step)

const std = @import("std");
const zgml = @import("zgml");
const Tensor = zgml.Tensor;
const ConvClassifier = zgml.models.ConvClassifier;
const Bucket = struct {
    name: []const u8,
    count: u64 = 0,
    total_ns: u64 = 0,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    const batch_size: usize = 32;
    const n_warmup: usize = 3;
    const n_runs: usize = 5;

    var model = try ConvClassifier(f32).build(alloc, 28, 28, 5, 8, 10, batch_size);
    defer model.deinit();
    try model.g.fusionPass(); // WITH fusion to see what remains

    // Fill with deterministic synthetic data
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    for (model.xs_batch.data) |*v| v.* = rand.float(f32);
    for (model.ys_batch.data) |*v| v.* = @floatFromInt(rand.intRangeAtMost(u32, 0, 9));

    const bwd_nodes = model.g.nodes.items[model.g.forward_node_count..];

    // Warmup
    for (0..n_warmup) |_| {
        model.g.reset();
        model.g.resetGrads();
        if (model.loss.grad) |grad| _ = grad.setAllScalar(1);
        model.g.compute();
    }

    // Accumulate across multiple runs for stability.
    // Use a map from op tag name -> bucket.
    var map = std.StringHashMap(Bucket).init(alloc);
    defer map.deinit();
    var total_bwd_ns: u64 = 0;

    for (0..n_runs) |_| {
        // Reset and run forward
        model.g.reset();
        model.g.resetGrads();
        if (model.loss.grad) |grad| _ = grad.setAllScalar(1);
        model.g.computeNoGrad();

        // Time each backward node individually (skip fused nodes)
        for (bwd_nodes, model.g.forward_node_count..) |node, idx| {
            if (idx < model.g.fused_skip.items.len and model.g.fused_skip.items[idx]) continue;
            const tag_name = @tagName(node.opTag());

            const node_t0 = std.Io.Clock.awake.now(io).nanoseconds;
            node.compute();
            const elapsed: u64 = @intCast(std.Io.Clock.awake.now(io).nanoseconds - node_t0);

            const gop = try map.getOrPut(tag_name);
            if (!gop.found_existing) {
                gop.value_ptr.* = .{ .name = tag_name };
            }
            gop.value_ptr.count += 1;
            gop.value_ptr.total_ns += elapsed;
            total_bwd_ns += elapsed;
        }
    }

    // Collect into a sortable array
    var entries = try std.ArrayList(Bucket).initCapacity(alloc, map.count());
    defer entries.deinit(alloc);
    var it = map.iterator();
    while (it.next()) |entry| {
        entries.appendAssumeCapacity(entry.value_ptr.*);
    }

    // Sort by total_ns descending
    std.mem.sortUnstable(Bucket, entries.items, {}, struct {
        fn lessThan(_: void, a: Bucket, b: Bucket) bool {
            return a.total_ns > b.total_ns;
        }
    }.lessThan);

    // Print results
    const stdout_file = std.Io.File.stdout();
    var buf: [8192]u8 = undefined;
    var w = stdout_file.writer(io, &buf);

    try w.interface.print("\nBackward Per-Op Profile (batch={}, {} runs, {} bwd nodes)\n", .{ batch_size, n_runs, bwd_nodes.len });
    try w.interface.print("==========================================================\n", .{});
    try w.interface.print("{s:<25} {s:>6} {s:>12} {s:>10} {s:>7}\n", .{ "op", "count", "total_ms", "avg_us", "pct" });
    try w.interface.print("{s:-<25} {s:->6} {s:->12} {s:->10} {s:->7}\n", .{ "", "", "", "", "" });

    const total_f: f64 = @floatFromInt(total_bwd_ns);

    for (entries.items) |b| {
        const total_ms = @as(f64, @floatFromInt(b.total_ns)) / 1_000_000.0;
        const avg_us = @as(f64, @floatFromInt(b.total_ns)) / @as(f64, @floatFromInt(b.count)) / 1_000.0;
        const pct = @as(f64, @floatFromInt(b.total_ns)) / total_f * 100.0;
        try w.interface.print("{s:<25} {d:>6} {d:>12.3} {d:>10.1} {d:>6.1}%\n", .{
            b.name, b.count, total_ms, avg_us, pct,
        });
    }

    const total_ms = @as(f64, @floatFromInt(total_bwd_ns)) / 1_000_000.0;
    try w.interface.print("{s:-<25} {s:->6} {s:->12} {s:->10} {s:->7}\n", .{ "", "", "", "", "" });
    try w.interface.print("{s:<25} {d:>6} {d:>12.3}\n", .{ "TOTAL", bwd_nodes.len * n_runs, total_ms });
    try w.interface.print("\n", .{});
    w.interface.flush() catch {};
}
