const std = @import("std");
pub const Io = std.Io;
pub const Dir = std.Io.Dir;
pub const File = std.Io.File;
pub const Writer = std.Io.Writer;
pub const Allocating = Writer.Allocating;

/// Read an entire file into a newly allocated aligned buffer.
pub fn readFileAlloc(alloc: std.mem.Allocator, io: Io, path: []const u8, comptime alignment: std.mem.Alignment, max_size: usize) ![]align(alignment.toByteUnits()) u8 {
    const file = try Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    const file_size: usize = @intCast(stat.size);
    if (file_size > max_size) return error.FileTooBig;

    const raw = try alloc.alignedAlloc(u8, alignment, file_size);
    errdefer alloc.free(raw);

    const n = try file.readPositionalAll(io, raw, 0);
    return raw[0..n];
}
