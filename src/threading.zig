const std = @import("std");

const io = std.Options.debug_io;

pub const WaitGroup = struct {
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    pending: usize = 0,

    pub fn start(self: *WaitGroup) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.pending += 1;
    }

    pub fn finish(self: *WaitGroup) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.pending -= 1;
        if (self.pending == 0) self.cond.broadcast(io);
    }

    pub fn wait(self: *WaitGroup) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (self.pending != 0) {
            self.cond.waitUncancelable(io, &self.mutex);
        }
    }
};

pub const Pool = struct {
    allocator: std.mem.Allocator = undefined,
    threads: []std.Thread = &.{},

    pub const Options = struct {
        allocator: std.mem.Allocator,
        track_ids: bool = false,
    };

    pub fn init(self: *Pool, options: Options) !void {
        _ = options.track_ids;
        self.allocator = options.allocator;

        const cpu_count = std.Thread.getCpuCount() catch 1;
        const worker_count = cpu_count -| 1;
        self.threads = try self.allocator.alloc(std.Thread, worker_count);
    }

    pub fn deinit(self: *Pool) void {
        self.allocator.free(self.threads);
        self.threads = &.{};
    }

    pub fn spawnWg(self: *Pool, wg: *WaitGroup, comptime function: anytype, args: anytype) void {
        const Task = struct {
            allocator: std.mem.Allocator,
            wait_group: *WaitGroup,
            args: @TypeOf(args),

            fn run(task: *@This()) void {
                defer {
                    task.wait_group.finish();
                    task.allocator.destroy(task);
                }
                @call(.auto, function, task.args);
            }
        };

        const task = self.allocator.create(Task) catch @panic("OOM");
        task.* = .{
            .allocator = self.allocator,
            .wait_group = wg,
            .args = args,
        };

        wg.start();
        const thread = std.Thread.spawn(.{}, Task.run, .{task}) catch @panic("unable to spawn worker");
        thread.detach();
    }
};
