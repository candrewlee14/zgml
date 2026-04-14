//! Parallel execution utilities for compute graphs.
//!
//! Provides `parFor` — a data-parallel range splitter that distributes
//! chunks of [0, n) across a thread pool. When no pool is available,
//! executes sequentially. All threading in zgml flows through this.

const std = @import("std");

/// Split [0, n) into chunks and execute `callback(ctx, chunk_start, chunk_end)`
/// in parallel across the pool. Caller thread participates as a worker.
///
/// Falls back to sequential `callback(ctx, 0, n)` when pool is null or n is small.
pub fn parFor(
    pool: ?*std.Thread.Pool,
    n: usize,
    ctx: anytype,
    comptime callback: fn (@TypeOf(ctx), usize, usize) void,
) void {
    if (n == 0) return;

    const tp = pool orelse {
        callback(ctx, 0, n);
        return;
    };

    const n_workers = tp.threads.len;
    if (n_workers == 0) {
        callback(ctx, 0, n);
        return;
    }

    const n_threads = n_workers + 1;
    const chunk = @max(1, (n + n_threads - 1) / n_threads);
    if (chunk >= n) {
        callback(ctx, 0, n);
        return;
    }

    // Spawn worker chunks
    var wg = std.Thread.WaitGroup{};
    var start: usize = chunk; // first chunk reserved for caller
    while (start < n) {
        const end = @min(start + chunk, n);
        tp.spawnWg(&wg, callback, .{ ctx, start, end });
        start = end;
    }

    // Caller does the first chunk
    callback(ctx, 0, @min(chunk, n));

    // Wait for workers
    wg.wait();
}
