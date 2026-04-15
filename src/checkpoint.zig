//! Model checkpoint serialization (save/load).
//!
//! Saves and loads model parameters to/from binary files with a simple format:
//!
//!   [magic: u32] [version: u32] [n_params: u32]
//!   For each parameter:
//!     [n_dims: u32] [ne[0]: u32] [ne[1]: u32] [ne[2]: u32] [ne[3]: u32]
//!     [data: T * n_elems]
//!
//! Usage:
//! ```
//! // Save
//! try checkpoint.save(f32, &model.params(), "model.bin");
//!
//! // Load
//! try checkpoint.load(f32, &model.params(), "model.bin");
//! ```

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const max_dims = @import("tensor.zig").max_dims;

const MAGIC: u32 = 0x5A474D4C; // "ZGML" in little-endian
const VERSION: u32 = 1;
const io = std.Options.debug_io;

fn createFile(path: []const u8) !std.Io.File {
    if (std.fs.path.isAbsolute(path)) {
        return std.Io.Dir.createFileAbsolute(io, path, .{});
    }
    return std.Io.Dir.cwd().createFile(io, path, .{});
}

fn openFile(path: []const u8) !std.Io.File {
    if (std.fs.path.isAbsolute(path)) {
        return std.Io.Dir.openFileAbsolute(io, path, .{});
    }
    return std.Io.Dir.cwd().openFile(io, path, .{});
}

fn deleteFile(path: []const u8) !void {
    if (std.fs.path.isAbsolute(path)) {
        return std.Io.Dir.deleteFileAbsolute(io, path);
    }
    return std.Io.Dir.cwd().deleteFile(io, path);
}

fn writeU32(file: std.Io.File, offset: *u64, val: u32) !void {
    const le = std.mem.nativeToLittle(u32, val);
    try file.writePositionalAll(io, std.mem.asBytes(&le), offset.*);
    offset.* += 4;
}

fn readU32(file: std.Io.File, offset: *u64) !u32 {
    var le: u32 = undefined;
    const n = try file.readPositionalAll(io, std.mem.asBytes(&le), offset.*);
    if (n != 4) return error.UnexpectedEof;
    offset.* += 4;
    return std.mem.littleToNative(u32, le);
}

pub fn save(comptime T: type, params: []const *Tensor(T), path: []const u8) !void {
    const file = try createFile(path);
    defer file.close(io);
    var offset: u64 = 0;

    // Header
    try writeU32(file, &offset, MAGIC);
    try writeU32(file, &offset, VERSION);
    try writeU32(file, &offset, @intCast(params.len));

    // Parameters
    for (params) |param| {
        // Shape
        try writeU32(file, &offset, param.n_dims);
        for (param.ne) |dim| {
            try writeU32(file, &offset, @intCast(dim));
        }
        // Data (raw bytes)
        const bytes = std.mem.sliceAsBytes(param.data);
        try file.writePositionalAll(io, bytes, offset);
        offset += bytes.len;
    }
}

pub fn load(comptime T: type, params: []const *Tensor(T), path: []const u8) !void {
    const file = try openFile(path);
    defer file.close(io);
    var offset: u64 = 0;

    // Header
    const magic = try readU32(file, &offset);
    if (magic != MAGIC) return error.InvalidCheckpoint;
    const version = try readU32(file, &offset);
    if (version != VERSION) return error.UnsupportedVersion;
    const n_params = try readU32(file, &offset);
    if (n_params != params.len) return error.ParamCountMismatch;

    // Parameters
    for (params) |param| {
        // Shape — read and verify
        const n_dims = try readU32(file, &offset);
        if (n_dims != param.n_dims) return error.ShapeMismatch;
        for (param.ne) |dim| {
            const saved_dim = try readU32(file, &offset);
            if (saved_dim != dim) return error.ShapeMismatch;
        }
        // Data
        const bytes = std.mem.sliceAsBytes(param.data);
        const n = try file.readPositionalAll(io, bytes, offset);
        if (n != bytes.len) return error.UnexpectedEof;
        offset += bytes.len;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const builtin = @import("builtin");
const testing = std.testing;
const tac = testing.allocator;
const is_wasm = builtin.target.cpu.arch == .wasm32 or builtin.target.cpu.arch == .wasm64;

test "checkpoint - save and load roundtrip" {
    if (comptime is_wasm) return error.SkipZigTest;
    const alloc = tac;

    const a = try Tensor(f32).init(alloc, &.{ 3, 2 });
    defer a.deinit();
    a.setData(&.{ 1.0, 2.0, 3.0, 4.5, -1.0, 0.5 });
    const b = try Tensor(f32).init(alloc, &.{4});
    defer b.deinit();
    b.setData(&.{ 10.0, 20.0, 30.0, 40.0 });

    const params = [_]*Tensor(f32){ a, b };

    try save(f32, &params, "/tmp/zgml_test_ckpt.bin");

    // Modify data
    _ = a.setAllScalar(0);
    _ = b.setAllScalar(0);

    // Load
    try load(f32, &params, "/tmp/zgml_test_ckpt.bin");

    // Verify restored
    try testing.expectEqualSlices(f32, &.{ 1.0, 2.0, 3.0, 4.5, -1.0, 0.5 }, a.data);
    try testing.expectEqualSlices(f32, &.{ 10.0, 20.0, 30.0, 40.0 }, b.data);

    deleteFile("/tmp/zgml_test_ckpt.bin") catch {};
}

test "checkpoint - shape mismatch is detected" {
    if (comptime is_wasm) return error.SkipZigTest;
    const alloc = tac;

    const a = try Tensor(f32).init(alloc, &.{ 3, 2 });
    defer a.deinit();
    a.setData(&.{ 1, 2, 3, 4, 5, 6 });

    try save(f32, &.{a}, "/tmp/zgml_test_ckpt2.bin");

    const b = try Tensor(f32).init(alloc, &.{ 2, 3 });
    defer b.deinit();

    const result = load(f32, &.{b}, "/tmp/zgml_test_ckpt2.bin");
    try testing.expectError(error.ShapeMismatch, result);

    deleteFile("/tmp/zgml_test_ckpt2.bin") catch {};
}

test "checkpoint - param count mismatch is detected" {
    if (comptime is_wasm) return error.SkipZigTest;
    const alloc = tac;

    const a = try Tensor(f32).init(alloc, &.{2});
    defer a.deinit();
    a.setData(&.{ 1, 2 });

    try save(f32, &.{a}, "/tmp/zgml_test_ckpt3.bin");

    const b = try Tensor(f32).init(alloc, &.{2});
    defer b.deinit();
    const c = try Tensor(f32).init(alloc, &.{2});
    defer c.deinit();

    const result = load(f32, &.{ b, c }, "/tmp/zgml_test_ckpt3.bin");
    try testing.expectError(error.ParamCountMismatch, result);

    deleteFile("/tmp/zgml_test_ckpt3.bin") catch {};
}
