//! Safetensors file format loader.
//!
//! Safetensors is a simple, safe tensor serialization format:
//!   [8 bytes: header_len (u64 LE)]
//!   [header_len bytes: JSON header]
//!   [remaining bytes: raw tensor data]
//!
//! The JSON header maps tensor names to {dtype, shape, data_offsets: [start, end]}.
//! Data offsets are relative to the start of the data section (after header).
//!
//! Usage:
//! ```
//! var sf = try SafetensorsFile.open(allocator, "model.safetensors");
//! defer sf.deinit();
//! const weight_data = sf.getTensorF32("model.layers.0.weight");
//! @memcpy(my_tensor.data, weight_data);
//! ```

const std = @import("std");
const Alloc = std.mem.Allocator;
const file_compat = @import("file_compat.zig");

pub const SafetensorsFile = struct {
    alloc: Alloc,
    raw_data: []align(4) u8,
    header_json: []const u8,
    data_start: usize,

    pub fn open(alloc: Alloc, path: []const u8) !SafetensorsFile {
        const file = try file_compat.openRead(path);
        defer file.close(file_compat.io);

        const file_size = (try file.stat(file_compat.io)).size;
        if (file_size < 8) return error.InvalidFormat;

        // Read entire file into aligned buffer
        const buf = try alloc.alignedAlloc(u8, .@"4", file_size);
        errdefer alloc.free(buf);
        const bytes_read = try file.readPositionalAll(file_compat.io, buf, 0);
        if (bytes_read != file_size) return error.UnexpectedEof;

        // Parse header length
        const header_len = std.mem.readInt(u64, buf[0..8], .little);
        if (8 + header_len > file_size) return error.InvalidFormat;

        const header_json = buf[8..][0..header_len];
        const data_start = 8 + header_len;

        return .{
            .alloc = alloc,
            .raw_data = buf,
            .header_json = header_json,
            .data_start = data_start,
        };
    }

    pub fn deinit(self: *SafetensorsFile) void {
        self.alloc.free(self.raw_data);
    }

    /// Get raw bytes for a tensor given its data offsets from the JSON header.
    pub fn getTensorBytes(self: *const SafetensorsFile, offset_start: usize, offset_end: usize) []const u8 {
        return self.raw_data[self.data_start + offset_start .. self.data_start + offset_end];
    }

    /// Get tensor data as f32 slice.
    pub fn getTensorF32(self: *const SafetensorsFile, offset_start: usize, offset_end: usize) []const f32 {
        const bytes = self.getTensorBytes(offset_start, offset_end);
        const n = bytes.len / @sizeOf(f32);
        return @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0..n];
    }

    /// Get tensor data as f16 slice.
    pub fn getTensorF16(self: *const SafetensorsFile, offset_start: usize, offset_end: usize) []const f16 {
        const bytes = self.getTensorBytes(offset_start, offset_end);
        const n = bytes.len / @sizeOf(f16);
        return @as([*]const f16, @ptrCast(@alignCast(bytes.ptr)))[0..n];
    }

    /// Simple JSON key lookup — finds "key": and returns the value.
    /// Works for simple cases in safetensors headers.
    pub fn findTensorMeta(self: *const SafetensorsFile, name: []const u8) ?TensorMeta {
        // Search for "name": { ... "data_offsets": [start, end] ... }
        const json = self.header_json;

        // Find the tensor key
        var search_start: usize = 0;
        while (search_start < json.len) {
            const key_start = std.mem.indexOf(u8, json[search_start..], name) orelse return null;
            const abs_pos = search_start + key_start;

            // Verify it's a quoted key: check for surrounding quotes
            if (abs_pos == 0 or json[abs_pos - 1] != '"') {
                search_start = abs_pos + name.len;
                continue;
            }
            const after_key = abs_pos + name.len;
            if (after_key >= json.len or json[after_key] != '"') {
                search_start = after_key;
                continue;
            }

            // Find the object for this tensor
            const obj_start = std.mem.indexOf(u8, json[after_key..], "{") orelse return null;
            const obj_abs = after_key + obj_start;

            // Find data_offsets within this object
            const offsets_key = "data_offsets";
            const offsets_start = std.mem.indexOf(u8, json[obj_abs..], offsets_key) orelse return null;
            const offsets_abs = obj_abs + offsets_start + offsets_key.len;

            // Parse [start, end]
            const bracket_start = std.mem.indexOf(u8, json[offsets_abs..], "[") orelse return null;
            const bracket_abs = offsets_abs + bracket_start + 1;
            const bracket_end = std.mem.indexOf(u8, json[bracket_abs..], "]") orelse return null;
            const range_str = json[bracket_abs .. bracket_abs + bracket_end];

            // Split on comma
            const comma = std.mem.indexOf(u8, range_str, ",") orelse return null;
            const start_str = std.mem.trim(u8, range_str[0..comma], " ");
            const end_str = std.mem.trim(u8, range_str[comma + 1 ..], " ");

            const start = std.fmt.parseInt(usize, start_str, 10) catch return null;
            const end = std.fmt.parseInt(usize, end_str, 10) catch return null;

            // Find dtype
            var dtype: DType = .f32;
            const dtype_key = "dtype";
            if (std.mem.indexOf(u8, json[obj_abs..], dtype_key)) |dt_start| {
                const dt_abs = obj_abs + dt_start + dtype_key.len;
                if (std.mem.indexOf(u8, json[dt_abs..], "F16") != null) {
                    // Check it's before the next key
                    const next_quote = std.mem.indexOf(u8, json[dt_abs..], "\"") orelse json.len - dt_abs;
                    if (std.mem.indexOf(u8, json[dt_abs..][0..next_quote + 10], "F16") != null) {
                        dtype = .f16;
                    }
                }
            }

            // Find shape
            var shape: [8]usize = .{0} ** 8;
            var n_dims: usize = 0;
            const shape_key = "shape";
            if (std.mem.indexOf(u8, json[obj_abs..], shape_key)) |sh_start| {
                const sh_abs = obj_abs + sh_start + shape_key.len;
                const sh_bracket = std.mem.indexOf(u8, json[sh_abs..], "[") orelse 0;
                const sh_end_bracket = std.mem.indexOf(u8, json[sh_abs + sh_bracket..], "]") orelse 0;
                if (sh_bracket > 0 or sh_end_bracket > 0) {
                    const shape_str = json[sh_abs + sh_bracket + 1 .. sh_abs + sh_bracket + sh_end_bracket];
                    var it = std.mem.splitSequence(u8, shape_str, ",");
                    while (it.next()) |dim_str| {
                        const trimmed = std.mem.trim(u8, dim_str, " ");
                        if (trimmed.len > 0) {
                            shape[n_dims] = std.fmt.parseInt(usize, trimmed, 10) catch continue;
                            n_dims += 1;
                        }
                    }
                }
            }

            return .{
                .offset_start = start,
                .offset_end = end,
                .dtype = dtype,
                .shape = shape,
                .n_dims = n_dims,
            };
        }
        return null;
    }

    /// List all tensor names in the file.
    pub fn tensorNames(self: *const SafetensorsFile, alloc: Alloc) ![][]const u8 {
        var names: std.ArrayList([]const u8) = .empty;
        errdefer names.deinit(alloc);

        const json = self.header_json;
        var pos: usize = 0;
        while (pos < json.len) {
            // Find next quoted string
            const quote_start = std.mem.indexOf(u8, json[pos..], "\"") orelse break;
            const abs_start = pos + quote_start + 1;
            const quote_end = std.mem.indexOf(u8, json[abs_start..], "\"") orelse break;
            const key = json[abs_start .. abs_start + quote_end];
            pos = abs_start + quote_end + 1;

            // Skip __metadata__ and internal keys
            if (std.mem.eql(u8, key, "__metadata__")) continue;
            if (key.len == 0) continue;

            // Check if this key maps to an object with data_offsets
            const colon = std.mem.indexOf(u8, json[pos..], ":") orelse continue;
            const after_colon = pos + colon + 1;
            const trimmed_start = std.mem.trimLeft(u8, json[after_colon..], " \t\n");
            if (trimmed_start.len > 0 and trimmed_start[0] == '{') {
                if (std.mem.indexOf(u8, trimmed_start, "data_offsets") != null) {
                    try names.append(alloc, key);
                }
            }
        }

        return names.toOwnedSlice(alloc);
    }
};

pub const DType = enum { f16, f32, f64, bf16 };

pub const TensorMeta = struct {
    offset_start: usize,
    offset_end: usize,
    dtype: DType,
    shape: [8]usize,
    n_dims: usize,

    pub fn nElems(self: TensorMeta) usize {
        if (self.n_dims == 0) return 0;
        var n: usize = 1;
        for (self.shape[0..self.n_dims]) |d| n *= d;
        return n;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "safetensors - parse header format" {
    // Test the JSON parsing with a synthetic header
    // We can't test file loading without a real file, but we can test the meta parser
    const alloc = std.testing.allocator;

    // Create a minimal safetensors file in memory
    const header =
        \\{"weight":{"dtype":"F32","shape":[3,2],"data_offsets":[0,24]},"bias":{"dtype":"F32","shape":[2],"data_offsets":[24,32]}}
    ;
    const header_len: u64 = header.len;
    var header_len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &header_len_bytes, header_len, .little);

    // Build the file bytes
    const data = [_]u8{0} ** 32; // dummy tensor data
    const file_bytes = try alloc.alignedAlloc(u8, .@"4", 8 + header.len + data.len);
    defer alloc.free(file_bytes);
    @memcpy(file_bytes[0..8], &header_len_bytes);
    @memcpy(file_bytes[8..][0..header.len], header);
    @memcpy(file_bytes[8 + header.len ..][0..data.len], &data);

    var sf = SafetensorsFile{
        .alloc = alloc,
        .raw_data = file_bytes,
        .header_json = file_bytes[8..][0..header.len],
        .data_start = 8 + header.len,
    };
    _ = &sf;

    // Test tensor lookup
    const weight_meta = sf.findTensorMeta("weight").?;
    try std.testing.expectEqual(@as(usize, 0), weight_meta.offset_start);
    try std.testing.expectEqual(@as(usize, 24), weight_meta.offset_end);
    try std.testing.expectEqual(@as(usize, 2), weight_meta.n_dims);
    try std.testing.expectEqual(@as(usize, 3), weight_meta.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), weight_meta.shape[1]);
    try std.testing.expectEqual(@as(usize, 6), weight_meta.nElems());

    const bias_meta = sf.findTensorMeta("bias").?;
    try std.testing.expectEqual(@as(usize, 24), bias_meta.offset_start);
    try std.testing.expectEqual(@as(usize, 32), bias_meta.offset_end);
    try std.testing.expectEqual(@as(usize, 1), bias_meta.n_dims);
    try std.testing.expectEqual(@as(usize, 2), bias_meta.shape[0]);

    // Non-existent tensor
    try std.testing.expect(sf.findTensorMeta("nonexistent") == null);
}
