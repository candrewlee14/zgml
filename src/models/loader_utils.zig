//! Shared utilities for model weight loading from safetensors files.

const std = @import("std");
const SafetensorsFile = @import("../safetensors.zig").SafetensorsFile;

/// Generic 2D weight loader: transpose from HF row-major to column-major,
/// converting to destination type T.
pub fn loadWeight2DGeneric(
    comptime T: type,
    dst: []T,
    sf: *const SafetensorsFile,
    name: []const u8,
    rows: usize,
    cols: usize,
) !void {
    const meta = sf.findTensorMeta(name) orelse return error.TensorNotFound;
    switch (meta.dtype) {
        .f32 => transposeRowToColGeneric(T, dst, sf.getTensorF32(meta.offset_start, meta.offset_end), rows, cols),
        .f16 => transposeRowToColGeneric(T, dst, sf.getTensorF16(meta.offset_start, meta.offset_end), rows, cols),
        else => return error.UnsupportedDtype,
    }
}

/// Generic 1D weight loader: copy with type conversion.
pub fn loadWeight1DGeneric(
    comptime T: type,
    dst: []T,
    sf: *const SafetensorsFile,
    name: []const u8,
) !void {
    const meta = sf.findTensorMeta(name) orelse return error.TensorNotFound;
    switch (meta.dtype) {
        .f32 => copyGeneric(T, dst, sf.getTensorF32(meta.offset_start, meta.offset_end)),
        .f16 => copyGeneric(T, dst, sf.getTensorF16(meta.offset_start, meta.offset_end)),
        else => return error.UnsupportedDtype,
    }
}

/// Transpose row-major [rows, cols] to column-major [cols, rows] with type conversion.
fn transposeRowToColGeneric(comptime T: type, dst: []T, src: anytype, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        for (0..cols) |c| {
            dst[c * rows + r] = @floatCast(src[r * cols + c]);
        }
    }
}

/// Copy with float type conversion.
fn copyGeneric(comptime T: type, dst: []T, src: anytype) void {
    for (dst, src) |*d, s| d.* = @floatCast(s);
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

test "layerName formats correctly" {
    var buf: [128]u8 = undefined;
    const name = try layerName(&buf, "model.layers.", 3, "self_attn.q_proj.weight");
    try std.testing.expectEqualStrings("model.layers.3.self_attn.q_proj.weight", name);
}
