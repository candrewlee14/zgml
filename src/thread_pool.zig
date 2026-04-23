const std = @import("std");

const Alloc = std.mem.Allocator;

/// Small persistent worker pool for graph-level CPU parallelism.
///
/// The pool owns background workers and lets the caller participate in each
/// job, so `workerCount() + 1` is the effective parallel width.
pub const ThreadPool = struct {
    const JobFn = *const fn (*anyopaque, usize) void;

    alloc: Alloc = undefined,
    threads: []std.Thread = &.{},
    mutex: std.Io.Mutex = .init,
    work_available: std.Io.Condition = .init,
    done: std.Io.Condition = .init,
    stopping: bool = false,
    job_fn: ?JobFn = null,
    job_ctx: ?*anyopaque = null,
    next_index: usize = 0,
    job_count: usize = 0,
    active: usize = 0,

    pub fn init(self: *ThreadPool, alloc: Alloc, worker_count: usize) !void {
        self.* = .{
            .alloc = alloc,
            .threads = try alloc.alloc(std.Thread, worker_count),
        };
        errdefer alloc.free(self.threads);

        var spawned: usize = 0;
        errdefer {
            self.lock();
            self.stopping = true;
            self.work_available.broadcast(io());
            self.unlock();
            for (self.threads[0..spawned]) |thread| thread.join();
        }

        while (spawned < worker_count) : (spawned += 1) {
            self.threads[spawned] = try std.Thread.spawn(.{}, workerMain, .{self});
        }
    }

    pub fn deinit(self: *ThreadPool) void {
        self.lock();
        self.stopping = true;
        self.work_available.broadcast(io());
        self.unlock();

        for (self.threads) |thread| thread.join();
        self.alloc.free(self.threads);
        self.* = .{};
    }

    pub fn workerCount(self: *const ThreadPool) usize {
        return self.threads.len;
    }

    pub fn threadCount(self: *const ThreadPool) usize {
        return self.threads.len + 1;
    }

    pub fn parallelFor(
        self: *ThreadPool,
        comptime Context: type,
        ctx: *Context,
        count: usize,
        comptime func: fn (*Context, usize) void,
    ) void {
        if (count == 0) return;
        if (self.threads.len == 0) {
            for (0..count) |i| func(ctx, i);
            return;
        }

        const Wrapper = struct {
            fn run(erased: *anyopaque, index: usize) void {
                const typed: *Context = @ptrCast(@alignCast(erased));
                func(typed, index);
            }
        };

        self.lock();
        std.debug.assert(self.job_fn == null);
        self.job_fn = Wrapper.run;
        self.job_ctx = ctx;
        self.next_index = 0;
        self.job_count = count;
        self.active = 0;
        self.work_available.broadcast(io());
        self.unlock();

        while (self.claimJob()) |index| {
            func(ctx, index);
            self.finishJob();
        }

        self.lock();
        while (self.job_fn != null) {
            self.done.waitUncancelable(io(), &self.mutex);
        }
        self.unlock();
    }

    fn workerMain(self: *ThreadPool) void {
        while (true) {
            const claimed = self.claimJobOrWait() orelse return;
            const func = claimed.func;
            const ctx = claimed.ctx;
            const index = claimed.index;
            func(ctx, index);
            self.finishJob();
        }
    }

    const ClaimedJob = struct {
        func: JobFn,
        ctx: *anyopaque,
        index: usize,
    };

    fn claimJobOrWait(self: *ThreadPool) ?ClaimedJob {
        self.lock();
        defer self.unlock();

        while (!self.stopping) {
            if (self.claimJobLocked()) |job| return job;
            self.work_available.waitUncancelable(io(), &self.mutex);
        }
        return null;
    }

    fn claimJob(self: *ThreadPool) ?usize {
        self.lock();
        defer self.unlock();
        const job = self.claimJobLocked() orelse return null;
        return job.index;
    }

    fn claimJobLocked(self: *ThreadPool) ?ClaimedJob {
        const func = self.job_fn orelse return null;
        if (self.next_index >= self.job_count) return null;

        const index = self.next_index;
        self.next_index += 1;
        self.active += 1;
        return .{
            .func = func,
            .ctx = self.job_ctx.?,
            .index = index,
        };
    }

    fn finishJob(self: *ThreadPool) void {
        self.lock();
        self.active -= 1;
        if (self.next_index >= self.job_count and self.active == 0) {
            self.job_fn = null;
            self.job_ctx = null;
            self.done.broadcast(io());
        }
        self.unlock();
    }

    fn lock(self: *ThreadPool) void {
        self.mutex.lockUncancelable(io());
    }

    fn unlock(self: *ThreadPool) void {
        self.mutex.unlock(io());
    }
};

fn io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

test "thread pool runs each job once" {
    var pool: ThreadPool = .{};
    try pool.init(std.testing.allocator, 2);
    defer pool.deinit();

    const Context = struct {
        counts: []usize,

        fn run(ctx: *@This(), index: usize) void {
            @atomicStore(usize, &ctx.counts[index], 1, .monotonic);
        }
    };

    var counts = [_]usize{0} ** 64;
    var ctx = Context{ .counts = &counts };
    pool.parallelFor(Context, &ctx, counts.len, Context.run);

    for (counts) |count| {
        try std.testing.expectEqual(@as(usize, 1), count);
    }
}
