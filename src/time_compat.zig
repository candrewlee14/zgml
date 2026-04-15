const std = @import("std");

pub fn nanoTimestamp() i96 {
    return std.Io.Timestamp.now(std.Options.debug_io, .awake).nanoseconds;
}

pub const Timer = struct {
    start_ns: i96,

    pub fn start() Timer {
        return .{ .start_ns = nanoTimestamp() };
    }

    pub fn reset(self: *Timer) void {
        self.start_ns = nanoTimestamp();
    }

    pub fn read(self: *const Timer) u64 {
        return @intCast(nanoTimestamp() - self.start_ns);
    }
};
