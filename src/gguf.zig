//! GGUF model file parser.
//!
//! GGUF (GPT-Generated Unified Format) is the successor to GGML's binary format,
//! used by llama.cpp and related projects for storing quantized LLM weights.
//!
//! File layout:
//!   [4 bytes: magic "GGUF" (0x46475547)]
//!   [4 bytes: version (u32 LE, 2 or 3)]
//!   [8 bytes: tensor_count (u64 LE for v3, u32 for v2)]
//!   [8 bytes: kv_count (u64 LE for v3, u32 for v2)]
//!   [kv_count KV pairs: key (string) + value_type (u32) + value]
//!   [tensor_count tensor info entries: name, n_dims, dims, type, offset]
//!   [padding to alignment boundary]
//!   [tensor data section]
//!
//! Strings are length-prefixed: [u64 len][len bytes]. All integers are little-endian.
//!
//! Usage:
//! ```
//! var gf = try GGUFFile.open(allocator, "model.gguf");
//! defer gf.deinit();
//! const arch = gf.getMetaString("general.architecture") orelse "unknown";
//! const info = gf.getTensorInfo("token_embd.weight").?;
//! const data = gf.getTensorData(info);
//! ```

const std = @import("std");

/// GGML tensor data types with block size and type size information.
pub const GGMLType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    // q4_2 = 4, // removed
    // q4_3 = 5, // removed
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,

    /// Number of elements per quantization block.
    /// Scalar types (f32, f16, i8, etc.) have a block size of 1.
    /// Quantized types pack 32 or 256 elements per block.
    pub fn blockSize(self: GGMLType) usize {
        return switch (self) {
            .f32, .f16, .f64 => 1,
            .i8, .i16, .i32, .i64 => 1,
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            .iq2_xxs, .iq2_xs, .iq2_s => 256,
            .iq3_xxs, .iq3_s => 256,
            .iq1_s, .iq1_m => 256,
            .iq4_nl => 32,
            .iq4_xs => 256,
        };
    }

    /// Bytes per quantization block.
    /// For scalar types this is simply @sizeOf(T).
    pub fn typeSize(self: GGMLType) usize {
        return switch (self) {
            .f32 => 4,
            .f16 => 2,
            .f64 => 8,
            .i8 => 1,
            .i16 => 2,
            .i32 => 4,
            .i64 => 8,
            .q4_0 => 18, // 32 * 4 bits / 8 + 2 (scale f16)
            .q4_1 => 20, // 32 * 4 bits / 8 + 2 (scale) + 2 (min)
            .q5_0 => 22, // 32 * 5 bits / 8 (round up) + 2 (scale)
            .q5_1 => 24, // 32 * 5 bits / 8 (round up) + 2 (scale) + 2 (min)
            .q8_0 => 34, // 32 * 8 bits / 8 + 2 (scale f16)
            .q8_1 => 40, // 32 * 8 bits / 8 + 4 (scale f32) + 4 (min f32)
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 292,
            .iq2_xxs => 66,
            .iq2_xs => 74,
            .iq2_s => 82,
            .iq3_xxs => 98,
            .iq3_s => 110,
            .iq1_s => 50,
            .iq1_m => 56,
            .iq4_nl => 18, // same as q4_0
            .iq4_xs => 136,
        };
    }
};

/// GGUF metadata value type tags.
pub const MetaValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool_ = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

/// A GGUF array value: element type + length + raw byte data.
/// The caller is responsible for interpreting the raw bytes according to elem_type.
pub const ArrayValue = struct {
    elem_type: MetaValueType,
    len: usize,
    data: []const u8,
};

/// A GGUF metadata value. Tagged union over MetaValueType.
/// String values are zero-copy slices into the raw file buffer.
pub const MetaValue = union(MetaValueType) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool_: bool,
    string: []const u8,
    array: ArrayValue,
    uint64: u64,
    int64: i64,
    float64: f64,
};

/// Information about a single tensor stored in the GGUF file.
pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dims: [4]u64,
    type_: GGMLType,
    offset: u64, // offset within the data section (not from file start)

    /// Total number of elements in this tensor.
    pub fn nElems(self: TensorInfo) u64 {
        var n: u64 = 1;
        for (self.dims[0..self.n_dims]) |d| n *= d;
        return n;
    }

    /// Total data size in bytes for this tensor.
    pub fn dataSize(self: TensorInfo) u64 {
        const n = self.nElems();
        const bs: u64 = self.type_.blockSize();
        return (n / bs) * self.type_.typeSize();
    }
};

/// A parsed GGUF file. Holds the raw file buffer (32-byte aligned),
/// parsed metadata KV pairs, and tensor info entries.
pub const GGUFFile = struct {
    raw_data: []align(32) const u8,
    version: u32,
    metadata: std.StringHashMapUnmanaged(MetaValue),
    tensors: std.StringHashMapUnmanaged(TensorInfo),
    data_offset: usize,
    alloc: std.mem.Allocator,

    const magic_value: u32 = 0x46475547; // "GGUF" in LE
    const default_alignment: usize = 32;

    /// Open and parse a GGUF file from disk.
    pub fn open(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !GGUFFile {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);

        const stat = try file.stat(io);
        const file_size: usize = @intCast(stat.size);
        if (file_size < 24) return error.InvalidFormat; // magic + version + counts

        const buf = try alloc.alignedAlloc(u8, .@"32", file_size);
        errdefer alloc.free(buf);
        const bytes_read = try file.readPositionalAll(io, buf, 0);
        if (bytes_read != file_size) return error.UnexpectedEof;

        return parseBuffer(alloc, buf);
    }

    /// Parse a GGUF file from an already-loaded buffer.
    /// The buffer must be 32-byte aligned. Ownership of `buf` is transferred
    /// to the returned GGUFFile (freed on deinit).
    pub fn parseBuffer(alloc: std.mem.Allocator, buf: []align(32) const u8) !GGUFFile {
        var cursor: usize = 0;

        // --- Header ---
        const magic = readVal(u32, buf, &cursor);
        if (magic != magic_value) return error.InvalidMagic;

        const version = readVal(u32, buf, &cursor);
        if (version < 2 or version > 3) return error.UnsupportedVersion;

        // v3 uses u64 for counts; v2 uses u32
        var tensor_count: u64 = undefined;
        var kv_count: u64 = undefined;
        if (version >= 3) {
            tensor_count = readVal(u64, buf, &cursor);
            kv_count = readVal(u64, buf, &cursor);
        } else {
            tensor_count = readVal(u32, buf, &cursor);
            kv_count = readVal(u32, buf, &cursor);
        }

        // --- KV pairs ---
        var metadata: std.StringHashMapUnmanaged(MetaValue) = .{};
        errdefer metadata.deinit(alloc);

        for (0..kv_count) |_| {
            const key = readString(buf, &cursor);
            const value = readMetaValue(buf, &cursor);
            try metadata.put(alloc, key, value);
        }

        // --- Tensor info entries ---
        var tensors: std.StringHashMapUnmanaged(TensorInfo) = .{};
        errdefer tensors.deinit(alloc);

        for (0..tensor_count) |_| {
            const name = readString(buf, &cursor);
            const n_dims = readVal(u32, buf, &cursor);
            var dims: [4]u64 = .{ 1, 1, 1, 1 };
            for (0..n_dims) |d| {
                dims[d] = readVal(u64, buf, &cursor);
            }
            const type_raw = readVal(u32, buf, &cursor);
            const type_: GGMLType = toGGMLType(type_raw) orelse return error.UnsupportedGGMLType;
            const offset = readVal(u64, buf, &cursor);

            const info = TensorInfo{
                .name = name,
                .n_dims = n_dims,
                .dims = dims,
                .type_ = type_,
                .offset = offset,
            };
            try tensors.put(alloc, name, info);
        }

        // --- Compute data offset (aligned) ---
        const alignment = blk: {
            if (metadata.get("general.alignment")) |val| {
                switch (val) {
                    .uint32 => |v| break :blk @as(usize, v),
                    .uint64 => |v| break :blk @as(usize, @intCast(v)),
                    .int32 => |v| break :blk @as(usize, @intCast(v)),
                    else => break :blk default_alignment,
                }
            }
            break :blk default_alignment;
        };
        const data_offset = alignUp(cursor, alignment);

        return .{
            .raw_data = buf,
            .version = version,
            .metadata = metadata,
            .tensors = tensors,
            .data_offset = data_offset,
            .alloc = alloc,
        };
    }

    pub fn deinit(self: *GGUFFile) void {
        self.metadata.deinit(self.alloc);
        self.tensors.deinit(self.alloc);
        const non_const: []align(32) u8 = @constCast(self.raw_data);
        self.alloc.free(non_const);
    }

    /// Look up a metadata value by key.
    pub fn getMeta(self: *const GGUFFile, key: []const u8) ?MetaValue {
        return self.metadata.get(key);
    }

    /// Look up a u32 metadata value by key.
    pub fn getMetaU32(self: *const GGUFFile, key: []const u8) ?u32 {
        const val = self.metadata.get(key) orelse return null;
        return switch (val) {
            .uint32 => |v| v,
            .int32 => |v| if (v >= 0) @intCast(v) else null,
            .uint64 => |v| if (v <= std.math.maxInt(u32)) @intCast(v) else null,
            else => null,
        };
    }

    /// Look up a string metadata value by key.
    pub fn getMetaString(self: *const GGUFFile, key: []const u8) ?[]const u8 {
        const val = self.metadata.get(key) orelse return null;
        return switch (val) {
            .string => |s| s,
            else => null,
        };
    }

    /// Look up tensor info by name.
    pub fn getTensorInfo(self: *const GGUFFile, name: []const u8) ?TensorInfo {
        return self.tensors.get(name);
    }

    /// Get the raw byte slice for a tensor's data.
    /// `info.offset` is relative to the start of the data section.
    pub fn getTensorData(self: *const GGUFFile, info: TensorInfo) []const u8 {
        const start = self.data_offset + @as(usize, @intCast(info.offset));
        const size = @as(usize, @intCast(info.dataSize()));
        return self.raw_data[start..][0..size];
    }

    /// Get tensor data reinterpreted as a f32 slice.
    /// Only valid when the tensor type is .f32.
    pub fn getTensorF32(self: *const GGUFFile, info: TensorInfo) []const f32 {
        std.debug.assert(info.type_ == .f32);
        const bytes = self.getTensorData(info);
        const n = bytes.len / @sizeOf(f32);
        return @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0..n];
    }

    // --- Internal parsing helpers ---

    /// Convert a raw u32 to a GGMLType, returning null for invalid/unsupported values.
    fn toGGMLType(raw: u32) ?GGMLType {
        const fields = @typeInfo(GGMLType).@"enum".fields;
        inline for (fields) |f| {
            if (raw == f.value) return @enumFromInt(raw);
        }
        return null;
    }

    fn readVal(comptime T: type, buf: []align(32) const u8, cursor: *usize) T {
        const size = @sizeOf(T);
        if (cursor.* + size > buf.len) @panic("GGUF: unexpected end of buffer");
        const result = std.mem.readInt(T, @ptrCast(buf[cursor.*..][0..size]), .little);
        cursor.* += size;
        return result;
    }

    /// Read a f32 from the buffer at the cursor position (little-endian).
    fn readF32(buf: []align(32) const u8, cursor: *usize) f32 {
        if (cursor.* + 4 > buf.len) @panic("GGUF: unexpected end of buffer");
        const bits = std.mem.readInt(u32, @ptrCast(buf[cursor.*..][0..4]), .little);
        cursor.* += 4;
        return @bitCast(bits);
    }

    /// Read a f64 from the buffer at the cursor position (little-endian).
    fn readF64(buf: []align(32) const u8, cursor: *usize) f64 {
        if (cursor.* + 8 > buf.len) @panic("GGUF: unexpected end of buffer");
        const bits = std.mem.readInt(u64, @ptrCast(buf[cursor.*..][0..8]), .little);
        cursor.* += 8;
        return @bitCast(bits);
    }

    /// Read a bool from the buffer (stored as u8, 0 = false).
    fn readBool(buf: []align(32) const u8, cursor: *usize) bool {
        const v = readVal(u8, buf, cursor);
        return v != 0;
    }

    /// Read a length-prefixed string (u64 len + bytes). Zero-copy slice into buf.
    fn readString(buf: []align(32) const u8, cursor: *usize) []const u8 {
        const len: usize = @intCast(readVal(u64, buf, cursor));
        if (cursor.* + len > buf.len) @panic("GGUF: string extends past buffer");
        const s = buf[cursor.*..][0..len];
        cursor.* += len;
        return s;
    }

    /// Read a metadata value based on its type tag.
    fn readMetaValue(buf: []align(32) const u8, cursor: *usize) MetaValue {
        const type_raw = readVal(u32, buf, cursor);
        const vtype: MetaValueType = @enumFromInt(type_raw);
        return readMetaValueOfType(buf, cursor, vtype);
    }

    /// Read a metadata value of a known type.
    fn readMetaValueOfType(buf: []align(32) const u8, cursor: *usize, vtype: MetaValueType) MetaValue {
        return switch (vtype) {
            .uint8 => .{ .uint8 = readVal(u8, buf, cursor) },
            .int8 => .{ .int8 = @bitCast(readVal(u8, buf, cursor)) },
            .uint16 => .{ .uint16 = readVal(u16, buf, cursor) },
            .int16 => .{ .int16 = @bitCast(readVal(u16, buf, cursor)) },
            .uint32 => .{ .uint32 = readVal(u32, buf, cursor) },
            .int32 => .{ .int32 = @bitCast(readVal(u32, buf, cursor)) },
            .float32 => .{ .float32 = readF32(buf, cursor) },
            .bool_ => .{ .bool_ = readBool(buf, cursor) },
            .string => .{ .string = readString(buf, cursor) },
            .array => blk: {
                const elem_type: MetaValueType = @enumFromInt(readVal(u32, buf, cursor));
                const len: usize = @intCast(readVal(u64, buf, cursor));
                const start = cursor.*;
                // Skip over the array elements to advance the cursor
                for (0..len) |_| {
                    skipMetaValue(buf, cursor, elem_type);
                }
                const end = cursor.*;
                break :blk .{ .array = .{
                    .elem_type = elem_type,
                    .len = len,
                    .data = buf[start..end],
                } };
            },
            .uint64 => .{ .uint64 = readVal(u64, buf, cursor) },
            .int64 => .{ .int64 = @bitCast(readVal(u64, buf, cursor)) },
            .float64 => .{ .float64 = readF64(buf, cursor) },
        };
    }

    /// Skip over a metadata value of a given type without storing it.
    fn skipMetaValue(buf: []align(32) const u8, cursor: *usize, vtype: MetaValueType) void {
        switch (vtype) {
            .uint8, .int8 => cursor.* += 1,
            .uint16, .int16 => cursor.* += 2,
            .uint32, .int32, .float32 => cursor.* += 4,
            .uint64, .int64, .float64 => cursor.* += 8,
            .bool_ => cursor.* += 1,
            .string => {
                const len: usize = @intCast(readVal(u64, buf, cursor));
                cursor.* += len;
            },
            .array => {
                const elem_type: MetaValueType = @enumFromInt(readVal(u32, buf, cursor));
                const len: usize = @intCast(readVal(u64, buf, cursor));
                for (0..len) |_| {
                    skipMetaValue(buf, cursor, elem_type);
                }
            },
        }
    }

    /// Align `offset` up to the next multiple of `alignment`.
    pub fn alignUp(offset: usize, alignment: usize) usize {
        return (offset + alignment - 1) & ~(alignment - 1);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Helper to append little-endian integers to an ArrayList(u8) for building test buffers.
fn appendU32(list: *std.ArrayList(u8), alloc: std.mem.Allocator, val: u32) !void {
    const bytes: [4]u8 = @bitCast(std.mem.nativeToLittle(u32, val));
    try list.appendSlice(alloc, &bytes);
}

fn appendU64(list: *std.ArrayList(u8), alloc: std.mem.Allocator, val: u64) !void {
    const bytes: [8]u8 = @bitCast(std.mem.nativeToLittle(u64, val));
    try list.appendSlice(alloc, &bytes);
}

fn appendStr(list: *std.ArrayList(u8), alloc: std.mem.Allocator, s: []const u8) !void {
    try appendU64(list, alloc, s.len);
    try list.appendSlice(alloc, s);
}

fn appendF32(list: *std.ArrayList(u8), alloc: std.mem.Allocator, val: f32) !void {
    try appendU32(list, alloc, @bitCast(val));
}

fn appendPadding(list: *std.ArrayList(u8), alloc: std.mem.Allocator, alignment: usize) !void {
    const target = GGUFFile.alignUp(list.items.len, alignment);
    while (list.items.len < target) {
        try list.append(alloc, 0);
    }
}

/// Copy an ArrayList into a 32-byte aligned allocation.
fn toAligned(alloc: std.mem.Allocator, list: *const std.ArrayList(u8)) ![]align(32) u8 {
    const result = try alloc.alignedAlloc(u8, .@"32", list.items.len);
    @memcpy(result, list.items);
    return result;
}

/// Build a synthetic GGUF v3 buffer in memory for testing.
fn buildTestBuffer(alloc: std.mem.Allocator) ![]align(32) u8 {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(alloc);

    // Magic
    try appendU32(&buf, alloc, 0x46475547);
    // Version 3
    try appendU32(&buf, alloc, 3);
    // Tensor count: 2
    try appendU64(&buf, alloc, 2);
    // KV count: 3
    try appendU64(&buf, alloc, 3);

    // --- KV pairs ---

    // KV 1: "general.architecture" = "llama" (string)
    try appendStr(&buf, alloc, "general.architecture");
    try appendU32(&buf, alloc, @intFromEnum(MetaValueType.string));
    try appendStr(&buf, alloc, "llama");

    // KV 2: "general.name" = "test-model" (string)
    try appendStr(&buf, alloc, "general.name");
    try appendU32(&buf, alloc, @intFromEnum(MetaValueType.string));
    try appendStr(&buf, alloc, "test-model");

    // KV 3: "llama.context_length" = 2048 (uint32)
    try appendStr(&buf, alloc, "llama.context_length");
    try appendU32(&buf, alloc, @intFromEnum(MetaValueType.uint32));
    try appendU32(&buf, alloc, 2048);

    // --- Tensor info entries ---

    // Tensor 1: "token_embd.weight" - f32 [128, 64]
    try appendStr(&buf, alloc, "token_embd.weight");
    try appendU32(&buf, alloc, 2); // n_dims
    try appendU64(&buf, alloc, 128); // dim 0
    try appendU64(&buf, alloc, 64); // dim 1
    try appendU32(&buf, alloc, @intFromEnum(GGMLType.f32)); // type
    try appendU64(&buf, alloc, 0); // offset within data section

    // Tensor 2: "output.weight" - f32 [64, 32]
    try appendStr(&buf, alloc, "output.weight");
    try appendU32(&buf, alloc, 2); // n_dims
    try appendU64(&buf, alloc, 64); // dim 0
    try appendU64(&buf, alloc, 32); // dim 1
    try appendU32(&buf, alloc, @intFromEnum(GGMLType.f32)); // type
    try appendU64(&buf, alloc, 128 * 64 * 4); // offset = 32768

    // Pad to default alignment (32 bytes)
    try appendPadding(&buf, alloc, 32);

    // Write dummy tensor data with identifiable pattern
    // token_embd.weight: 128*64 = 8192 f32s = 32768 bytes
    // output.weight: 64*32 = 2048 f32s = 8192 bytes
    const total_data = 128 * 64 * 4 + 64 * 32 * 4;
    for (0..total_data) |i| {
        try buf.append(alloc, @truncate(i));
    }

    return toAligned(alloc, &buf);
}

test "gguf - parse synthetic v3 buffer" {
    const alloc = testing.allocator;
    const buf = try buildTestBuffer(alloc);

    var gf = try GGUFFile.parseBuffer(alloc, buf);
    defer gf.deinit();

    try testing.expectEqual(@as(u32, 3), gf.version);
}

test "gguf - metadata retrieval" {
    const alloc = testing.allocator;
    const buf = try buildTestBuffer(alloc);

    var gf = try GGUFFile.parseBuffer(alloc, buf);
    defer gf.deinit();

    // String metadata
    const arch = gf.getMetaString("general.architecture");
    try testing.expect(arch != null);
    try testing.expectEqualStrings("llama", arch.?);

    const name = gf.getMetaString("general.name");
    try testing.expect(name != null);
    try testing.expectEqualStrings("test-model", name.?);

    // u32 metadata
    const ctx_len = gf.getMetaU32("llama.context_length");
    try testing.expect(ctx_len != null);
    try testing.expectEqual(@as(u32, 2048), ctx_len.?);

    // Generic getMeta
    const val = gf.getMeta("llama.context_length");
    try testing.expect(val != null);
    try testing.expectEqual(MetaValueType.uint32, std.meta.activeTag(val.?));

    // Non-existent key
    try testing.expect(gf.getMetaString("nonexistent.key") == null);
    try testing.expect(gf.getMetaU32("nonexistent.key") == null);
    try testing.expect(gf.getMeta("nonexistent.key") == null);
}

test "gguf - tensor info retrieval" {
    const alloc = testing.allocator;
    const buf = try buildTestBuffer(alloc);

    var gf = try GGUFFile.parseBuffer(alloc, buf);
    defer gf.deinit();

    // token_embd.weight
    const embd = gf.getTensorInfo("token_embd.weight");
    try testing.expect(embd != null);
    const embd_info = embd.?;
    try testing.expectEqual(@as(u32, 2), embd_info.n_dims);
    try testing.expectEqual(@as(u64, 128), embd_info.dims[0]);
    try testing.expectEqual(@as(u64, 64), embd_info.dims[1]);
    try testing.expectEqual(GGMLType.f32, embd_info.type_);
    try testing.expectEqual(@as(u64, 0), embd_info.offset);
    try testing.expectEqual(@as(u64, 128 * 64), embd_info.nElems());
    try testing.expectEqual(@as(u64, 128 * 64 * 4), embd_info.dataSize());

    // output.weight
    const out = gf.getTensorInfo("output.weight");
    try testing.expect(out != null);
    const out_info = out.?;
    try testing.expectEqual(@as(u32, 2), out_info.n_dims);
    try testing.expectEqual(@as(u64, 64), out_info.dims[0]);
    try testing.expectEqual(@as(u64, 32), out_info.dims[1]);
    try testing.expectEqual(GGMLType.f32, out_info.type_);
    try testing.expectEqual(@as(u64, 128 * 64 * 4), out_info.offset);
    try testing.expectEqual(@as(u64, 64 * 32), out_info.nElems());
    try testing.expectEqual(@as(u64, 64 * 32 * 4), out_info.dataSize());

    // Non-existent tensor
    try testing.expect(gf.getTensorInfo("nonexistent") == null);
}

test "gguf - data offset alignment" {
    const alloc = testing.allocator;
    const buf = try buildTestBuffer(alloc);

    var gf = try GGUFFile.parseBuffer(alloc, buf);
    defer gf.deinit();

    // Data offset must be aligned to 32 (default)
    try testing.expectEqual(@as(usize, 0), gf.data_offset % 32);
    // Data offset must be > 0 (after the header)
    try testing.expect(gf.data_offset > 0);
}

test "gguf - getTensorData and getTensorF32" {
    const alloc = testing.allocator;
    const buf = try buildTestBuffer(alloc);

    var gf = try GGUFFile.parseBuffer(alloc, buf);
    defer gf.deinit();

    const embd_info = gf.getTensorInfo("token_embd.weight").?;
    const data = gf.getTensorData(embd_info);
    try testing.expectEqual(@as(usize, 128 * 64 * 4), data.len);

    // Verify data starts at the right offset
    const out_info = gf.getTensorInfo("output.weight").?;
    const out_data = gf.getTensorData(out_info);
    try testing.expectEqual(@as(usize, 64 * 32 * 4), out_data.len);
}

test "gguf - GGMLType block and type sizes" {
    // Scalar types
    try testing.expectEqual(@as(usize, 1), GGMLType.f32.blockSize());
    try testing.expectEqual(@as(usize, 4), GGMLType.f32.typeSize());
    try testing.expectEqual(@as(usize, 1), GGMLType.f16.blockSize());
    try testing.expectEqual(@as(usize, 2), GGMLType.f16.typeSize());
    try testing.expectEqual(@as(usize, 1), GGMLType.f64.blockSize());
    try testing.expectEqual(@as(usize, 8), GGMLType.f64.typeSize());

    // Integer types
    try testing.expectEqual(@as(usize, 1), GGMLType.i8.blockSize());
    try testing.expectEqual(@as(usize, 1), GGMLType.i8.typeSize());
    try testing.expectEqual(@as(usize, 1), GGMLType.i32.blockSize());
    try testing.expectEqual(@as(usize, 4), GGMLType.i32.typeSize());

    // Quantized types
    try testing.expectEqual(@as(usize, 32), GGMLType.q4_0.blockSize());
    try testing.expectEqual(@as(usize, 18), GGMLType.q4_0.typeSize());
    try testing.expectEqual(@as(usize, 32), GGMLType.q8_0.blockSize());
    try testing.expectEqual(@as(usize, 34), GGMLType.q8_0.typeSize());
    try testing.expectEqual(@as(usize, 256), GGMLType.q4_k.blockSize());
    try testing.expectEqual(@as(usize, 144), GGMLType.q4_k.typeSize());
}

test "gguf - TensorInfo nElems and dataSize" {
    // 1D tensor
    const t1 = TensorInfo{
        .name = "test",
        .n_dims = 1,
        .dims = .{ 100, 1, 1, 1 },
        .type_ = .f32,
        .offset = 0,
    };
    try testing.expectEqual(@as(u64, 100), t1.nElems());
    try testing.expectEqual(@as(u64, 400), t1.dataSize());

    // 3D tensor
    const t3 = TensorInfo{
        .name = "test3d",
        .n_dims = 3,
        .dims = .{ 10, 20, 30, 1 },
        .type_ = .f16,
        .offset = 0,
    };
    try testing.expectEqual(@as(u64, 6000), t3.nElems());
    try testing.expectEqual(@as(u64, 12000), t3.dataSize());

    // Quantized tensor (q4_0: 32 elements per 18-byte block)
    const tq = TensorInfo{
        .name = "quantized",
        .n_dims = 1,
        .dims = .{ 256, 1, 1, 1 },
        .type_ = .q4_0,
        .offset = 0,
    };
    try testing.expectEqual(@as(u64, 256), tq.nElems());
    // 256 / 32 = 8 blocks * 18 bytes = 144
    try testing.expectEqual(@as(u64, 144), tq.dataSize());
}

test "gguf - v2 buffer parsing" {
    const alloc = testing.allocator;

    // Build a minimal v2 buffer (u32 counts instead of u64)
    var buf_list = std.ArrayList(u8).empty;
    defer buf_list.deinit(alloc);

    try appendU32(&buf_list, alloc, 0x46475547); // magic
    try appendU32(&buf_list, alloc, 2); // version 2
    try appendU32(&buf_list, alloc, 1); // tensor count (u32 for v2)
    try appendU32(&buf_list, alloc, 1); // kv count (u32 for v2)

    // KV: "general.architecture" = "test"
    try appendStr(&buf_list, alloc, "general.architecture");
    try appendU32(&buf_list, alloc, @intFromEnum(MetaValueType.string));
    try appendStr(&buf_list, alloc, "test");

    // Tensor: "w" - f32 [4]
    try appendStr(&buf_list, alloc, "w");
    try appendU32(&buf_list, alloc, 1); // n_dims
    try appendU64(&buf_list, alloc, 4); // dim 0
    try appendU32(&buf_list, alloc, @intFromEnum(GGMLType.f32));
    try appendU64(&buf_list, alloc, 0); // offset

    // Pad to alignment
    try appendPadding(&buf_list, alloc, 32);

    // 4 f32s of data (16 bytes)
    for (0..16) |_| try buf_list.append(alloc, 0);

    const buf = try toAligned(alloc, &buf_list);

    var gf = try GGUFFile.parseBuffer(alloc, buf);
    defer gf.deinit();

    try testing.expectEqual(@as(u32, 2), gf.version);
    try testing.expectEqualStrings("test", gf.getMetaString("general.architecture").?);

    const tensor_w = gf.getTensorInfo("w").?;
    try testing.expectEqual(@as(u64, 4), tensor_w.nElems());
    try testing.expectEqual(@as(u64, 16), tensor_w.dataSize());
}

test "gguf - custom alignment via general.alignment" {
    const alloc = testing.allocator;

    var buf_list = std.ArrayList(u8).empty;
    defer buf_list.deinit(alloc);

    try appendU32(&buf_list, alloc, 0x46475547); // magic
    try appendU32(&buf_list, alloc, 3); // version 3
    try appendU64(&buf_list, alloc, 1); // tensor count
    try appendU64(&buf_list, alloc, 1); // kv count

    // KV: "general.alignment" = 64 (uint32)
    try appendStr(&buf_list, alloc, "general.alignment");
    try appendU32(&buf_list, alloc, @intFromEnum(MetaValueType.uint32));
    try appendU32(&buf_list, alloc, 64);

    // Tensor: "x" - f32 [2]
    try appendStr(&buf_list, alloc, "x");
    try appendU32(&buf_list, alloc, 1); // n_dims
    try appendU64(&buf_list, alloc, 2); // dim 0
    try appendU32(&buf_list, alloc, @intFromEnum(GGMLType.f32));
    try appendU64(&buf_list, alloc, 0); // offset

    const header_end = buf_list.items.len;

    // Pad to alignment=64
    try appendPadding(&buf_list, alloc, 64);

    // 2 f32s of data (8 bytes)
    for (0..8) |_| try buf_list.append(alloc, 0);

    const buf = try toAligned(alloc, &buf_list);

    var gf = try GGUFFile.parseBuffer(alloc, buf);
    defer gf.deinit();

    // data_offset must be aligned to 64
    try testing.expectEqual(@as(usize, 0), gf.data_offset % 64);
    try testing.expect(gf.data_offset >= header_end);
}

test "gguf - invalid magic rejected" {
    const alloc = testing.allocator;
    const buf = try alloc.alignedAlloc(u8, .@"32", 32);
    defer alloc.free(buf);
    @memset(buf, 0);
    // Wrong magic
    std.mem.writeInt(u32, buf[0..4], 0xDEADBEEF, .little);

    const result = GGUFFile.parseBuffer(alloc, buf);
    try testing.expectError(error.InvalidMagic, result);
}

test "gguf - unsupported version rejected" {
    const alloc = testing.allocator;
    const buf = try alloc.alignedAlloc(u8, .@"32", 32);
    defer alloc.free(buf);
    @memset(buf, 0);
    // Correct magic, version 99
    std.mem.writeInt(u32, buf[0..4], 0x46475547, .little);
    std.mem.writeInt(u32, buf[4..8], 99, .little);

    const result = GGUFFile.parseBuffer(alloc, buf);
    try testing.expectError(error.UnsupportedVersion, result);
}

test "gguf - array metadata value" {
    const alloc = testing.allocator;

    var buf_list = std.ArrayList(u8).empty;
    defer buf_list.deinit(alloc);

    try appendU32(&buf_list, alloc, 0x46475547); // magic
    try appendU32(&buf_list, alloc, 3); // version
    try appendU64(&buf_list, alloc, 0); // no tensors
    try appendU64(&buf_list, alloc, 1); // 1 KV

    // KV: "tokenizer.scores" = array of float32 [1.5, 2.5, 3.5]
    try appendStr(&buf_list, alloc, "tokenizer.scores");
    try appendU32(&buf_list, alloc, @intFromEnum(MetaValueType.array));
    try appendU32(&buf_list, alloc, @intFromEnum(MetaValueType.float32)); // elem type
    try appendU64(&buf_list, alloc, 3); // length
    try appendF32(&buf_list, alloc, 1.5);
    try appendF32(&buf_list, alloc, 2.5);
    try appendF32(&buf_list, alloc, 3.5);

    // Pad to 32-byte alignment
    try appendPadding(&buf_list, alloc, 32);

    const buf = try toAligned(alloc, &buf_list);

    var gf = try GGUFFile.parseBuffer(alloc, buf);
    defer gf.deinit();

    const val = gf.getMeta("tokenizer.scores");
    try testing.expect(val != null);
    try testing.expectEqual(MetaValueType.array, std.meta.activeTag(val.?));

    const arr = val.?.array;
    try testing.expectEqual(MetaValueType.float32, arr.elem_type);
    try testing.expectEqual(@as(usize, 3), arr.len);
    try testing.expectEqual(@as(usize, 12), arr.data.len); // 3 * 4 bytes
}

test "gguf - alignUp" {
    try testing.expectEqual(@as(usize, 0), GGUFFile.alignUp(0, 32));
    try testing.expectEqual(@as(usize, 32), GGUFFile.alignUp(1, 32));
    try testing.expectEqual(@as(usize, 32), GGUFFile.alignUp(31, 32));
    try testing.expectEqual(@as(usize, 32), GGUFFile.alignUp(32, 32));
    try testing.expectEqual(@as(usize, 64), GGUFFile.alignUp(33, 32));
    try testing.expectEqual(@as(usize, 64), GGUFFile.alignUp(33, 64));
    try testing.expectEqual(@as(usize, 128), GGUFFile.alignUp(65, 64));
}
