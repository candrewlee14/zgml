const std = @import("std");

pub const io = std.Options.debug_io;

pub fn openRead(path: []const u8) !std.Io.File {
    if (std.fs.path.isAbsolute(path)) {
        return std.Io.Dir.openFileAbsolute(io, path, .{});
    }
    return std.Io.Dir.cwd().openFile(io, path, .{});
}

pub fn readToEndAlloc(alloc: std.mem.Allocator, path: []const u8, max_bytes: usize) ![]u8 {
    const file = try openRead(path);
    defer file.close(io);

    const stat = try file.stat(io);
    if (stat.size > max_bytes) return error.FileTooBig;

    const size: usize = @intCast(stat.size);
    const buf = try alloc.alloc(u8, size);
    errdefer alloc.free(buf);

    const bytes_read = try file.readPositionalAll(io, buf, 0);
    if (bytes_read != size) return error.UnexpectedEof;
    return buf;
}
