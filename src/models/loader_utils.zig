//! Shared utilities for model weight loading from safetensors files.
//!
//! Provides row-major → column-major transpose, f16→f32 conversion,
//! and layer name formatting helpers used by all weight loaders.

const std = @import("std");
const SafetensorsFile = @import("../safetensors.zig").SafetensorsFile;

/// Transpose a 2D matrix from row-major [rows, cols] to column-major [cols, rows].
pub fn transposeRowToCol(dst: []f32, src: []const f32, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        for (0..cols) |c| {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

/// Transpose a 2D f16 matrix from row-major to column-major with f32 conversion.
pub fn transposeRowToColF16(dst: []f32, src: []const f16, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        for (0..cols) |c| {
            dst[c * rows + r] = @floatCast(src[r * cols + c]);
        }
    }
}

/// Transpose a [rows, cols] row-major matrix into a row-offset region of a
/// packed column-major destination. Used to stack Q/K/V into packed w_qkv.
pub fn transposeIntoPackedRows(
    dst: []f32,
    src: []const f32,
    rows: usize,
    cols: usize,
    dst_rows: usize,
    row_offset: usize,
) void {
    for (0..rows) |r| {
        for (0..cols) |c| {
            dst[c * dst_rows + row_offset + r] = src[r * cols + c];
        }
    }
}

/// f16 variant of transposeIntoPackedRows.
pub fn transposeIntoPackedRowsF16(
    dst: []f32,
    src: []const f16,
    rows: usize,
    cols: usize,
    dst_rows: usize,
    row_offset: usize,
) void {
    for (0..rows) |r| {
        for (0..cols) |c| {
            dst[c * dst_rows + row_offset + r] = @floatCast(src[r * cols + c]);
        }
    }
}

pub fn copyDirect(dst: []f32, src: []const f32) void {
    @memcpy(dst, src);
}

pub fn copyDirectF16(dst: []f32, src: []const f16) void {
    for (dst, src) |*d, s| d.* = @floatCast(s);
}

/// Load a 2D weight matrix, transposing from HF row-major to zgml column-major.
/// Auto-handles f16 → f32 conversion based on dtype.
pub fn loadWeight2D(
    dst: []f32,
    sf: *const SafetensorsFile,
    name: []const u8,
    rows: usize,
    cols: usize,
) !void {
    const meta = sf.findTensorMeta(name) orelse return error.TensorNotFound;
    switch (meta.dtype) {
        .f32 => transposeRowToCol(dst, sf.getTensorF32(meta.offset_start, meta.offset_end), rows, cols),
        .f16 => transposeRowToColF16(dst, sf.getTensorF16(meta.offset_start, meta.offset_end), rows, cols),
        else => return error.UnsupportedDtype,
    }
}

/// Load a 1D weight vector (bias, norm weight). No transposition needed.
pub fn loadWeight1D(
    dst: []f32,
    sf: *const SafetensorsFile,
    name: []const u8,
) !void {
    const meta = sf.findTensorMeta(name) orelse return error.TensorNotFound;
    switch (meta.dtype) {
        .f32 => copyDirect(dst, sf.getTensorF32(meta.offset_start, meta.offset_end)),
        .f16 => copyDirectF16(dst, sf.getTensorF16(meta.offset_start, meta.offset_end)),
        else => return error.UnsupportedDtype,
    }
}

/// Load a 1D tensor into a slice of the destination buffer.
pub fn loadIntoSlice(dst: []f32, start: usize, end: usize, sf: *const SafetensorsFile, name: []const u8) void {
    const meta = sf.findTensorMeta(name) orelse return;
    switch (meta.dtype) {
        .f32 => @memcpy(dst[start..end], sf.getTensorF32(meta.offset_start, meta.offset_end)),
        .f16 => {
            const src = sf.getTensorF16(meta.offset_start, meta.offset_end);
            for (dst[start..end], src) |*d, s| d.* = @floatCast(s);
        },
        else => {},
    }
}

/// Transpose from safetensors into a packed row-offset region of a destination buffer.
/// Used to load separate Q/K/V weights into a packed w_qkv matrix.
pub fn transposeIntoPacked(
    dst: []f32,
    sf: *const SafetensorsFile,
    name: []const u8,
    rows: usize,
    cols: usize,
    dst_rows: usize,
    row_offset: usize,
) void {
    const meta = sf.findTensorMeta(name) orelse return;
    switch (meta.dtype) {
        .f32 => transposeIntoPackedRows(dst, sf.getTensorF32(meta.offset_start, meta.offset_end), rows, cols, dst_rows, row_offset),
        .f16 => transposeIntoPackedRowsF16(dst, sf.getTensorF16(meta.offset_start, meta.offset_end), rows, cols, dst_rows, row_offset),
        else => {},
    }
}

/// Format a layer-scoped tensor name: "{prefix}{layer}.{suffix}".
pub fn layerName(buf: *[128]u8, prefix: []const u8, layer: usize, suffix: []const u8) ![]const u8 {
    var pos: usize = 0;
    @memcpy(buf[pos..][0..prefix.len], prefix);
    pos += prefix.len;
    const layer_str = try std.fmt.bufPrint(buf[pos..], "{d}", .{layer});
    pos += layer_str.len;
    buf[pos] = '.';
    pos += 1;
    @memcpy(buf[pos..][0..suffix.len], suffix);
    pos += suffix.len;
    return buf[0..pos];
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "transposeRowToCol matches expected layout" {
    const src = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var dst: [6]f32 = undefined;
    transposeRowToCol(&dst, &src, 2, 3);
    try testing.expectEqual(@as(f32, 1), dst[0]);
    try testing.expectEqual(@as(f32, 4), dst[1]);
    try testing.expectEqual(@as(f32, 2), dst[2]);
    try testing.expectEqual(@as(f32, 5), dst[3]);
    try testing.expectEqual(@as(f32, 3), dst[4]);
    try testing.expectEqual(@as(f32, 6), dst[5]);
}

test "transposeIntoPackedRows stacks correctly" {
    var dst: [8]f32 = undefined;
    @memset(&dst, 0);
    const q = [_]f32{ 1, 2, 3, 4 };
    const k = [_]f32{ 5, 6, 7, 8 };
    transposeIntoPackedRows(&dst, &q, 2, 2, 4, 0);
    transposeIntoPackedRows(&dst, &k, 2, 2, 4, 2);
    try testing.expectEqual(@as(f32, 1), dst[0]);
    try testing.expectEqual(@as(f32, 3), dst[1]);
    try testing.expectEqual(@as(f32, 5), dst[2]);
    try testing.expectEqual(@as(f32, 7), dst[3]);
    try testing.expectEqual(@as(f32, 2), dst[4]);
    try testing.expectEqual(@as(f32, 4), dst[5]);
    try testing.expectEqual(@as(f32, 6), dst[6]);
    try testing.expectEqual(@as(f32, 8), dst[7]);
}

test "transposeRowToColF16 converts and transposes" {
    const src = [_]f16{ 1, 2, 3, 4, 5, 6 };
    var dst: [6]f32 = undefined;
    transposeRowToColF16(&dst, &src, 2, 3);
    try testing.expectEqual(@as(f32, 1), dst[0]);
    try testing.expectEqual(@as(f32, 4), dst[1]);
}

test "layerName formats correctly" {
    var buf: [128]u8 = undefined;
    const name = try layerName(&buf, "model.layers.", 3, "self_attn.q_proj.weight");
    try testing.expectEqualStrings("model.layers.3.self_attn.q_proj.weight", name);
}
