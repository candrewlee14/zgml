//! GGUF weight loader for LLaMA-architecture models.
//!
//! Maps GGUF tensor names (llama.cpp convention) to zgml LLaMA model
//! parameters and handles dequantization from Q4_0, Q8_0, and F16 formats
//! to f32.
//!
//! ```
//! var gf = try GGUFFile.open(alloc, "model.gguf");
//! defer gf.deinit();
//! const cfg = configFromGGUF(&gf);
//! try loadDequantized(f32, cfg, &model, &gf);
//! ```

const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const gguf_mod = @import("../gguf.zig");
const GGUFFile = gguf_mod.GGUFFile;
const GGMLType = gguf_mod.GGMLType;
const TensorInfo = gguf_mod.TensorInfo;
const LLaMA = @import("llama.zig").LLaMA;
const LlamaConfig = @import("llama.zig").LlamaConfig;

const log = std.log.scoped(.gguf_loader);

// ---------------------------------------------------------------------------
// Dequantization helpers
// ---------------------------------------------------------------------------

/// Dequantize Q4_0 blocks to f32.
/// Q4_0 layout: 32 elements per 18-byte block (2-byte f16 scale + 16 nibble bytes).
/// Nibbles store unsigned [0,15] representing signed [-8,7] via val = nibble - 8.
fn dequantQ4_0(dst: []f32, src: []const u8, n_elems: usize) void {
    const block_size: usize = 32;
    const block_bytes: usize = 18;
    const n_blocks = (n_elems + block_size - 1) / block_size;

    for (0..n_blocks) |b| {
        const blk_off = b * block_bytes;
        const scale_bytes: [2]u8 = .{ src[blk_off], src[blk_off + 1] };
        const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bytes)));

        const elems_in_block = @min(block_size, n_elems - b * block_size);
        for (0..elems_in_block) |i| {
            const nib_byte = src[blk_off + 2 + i / 2];
            const nibble: u8 = if (i % 2 == 0) (nib_byte & 0x0F) else (nib_byte >> 4);
            const signed_val: f32 = @floatFromInt(@as(i8, @intCast(@as(i16, nibble) - 8)));
            dst[b * block_size + i] = signed_val * scale;
        }
    }
}

/// Dequantize Q8_0 blocks to f32.
/// Q8_0 layout: 32 elements per 34-byte block (2-byte f16 scale + 32 i8 values).
fn dequantQ8_0(dst: []f32, src: []const u8, n_elems: usize) void {
    const block_size: usize = 32;
    const block_bytes: usize = 34;
    const n_blocks = (n_elems + block_size - 1) / block_size;

    for (0..n_blocks) |b| {
        const blk_off = b * block_bytes;
        const scale_bytes: [2]u8 = .{ src[blk_off], src[blk_off + 1] };
        const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bytes)));

        const elems_in_block = @min(block_size, n_elems - b * block_size);
        for (0..elems_in_block) |i| {
            const q: i8 = @bitCast(src[blk_off + 2 + i]);
            dst[b * block_size + i] = @as(f32, @floatFromInt(q)) * scale;
        }
    }
}

/// Convert f16 elements to f32.
fn dequantF16(dst: []f32, src: []const u8, n_elems: usize) void {
    for (0..n_elems) |i| {
        const offset = i * 2;
        const f16_bytes: [2]u8 = .{ src[offset], src[offset + 1] };
        dst[i] = @floatCast(@as(f16, @bitCast(f16_bytes)));
    }
}

// ---------------------------------------------------------------------------
// Tensor loading helper
// ---------------------------------------------------------------------------

/// Load a single tensor from GGUF, dequantizing to f32 if needed.
fn loadTensor(comptime T: type, dst: *Tensor(T), gf: *const GGUFFile, name: []const u8) !void {
    const info = gf.getTensorInfo(name) orelse {
        log.warn("tensor not found: {s} (skipping)", .{name});
        return error.TensorNotFound;
    };

    const n_elems: usize = @intCast(info.nElems());
    if (dst.data.len != n_elems) {
        log.warn("tensor {s}: size mismatch (expected {d}, got {d})", .{ name, dst.data.len, n_elems });
        return error.SizeMismatch;
    }

    const raw = gf.getTensorData(info);

    switch (info.type_) {
        .f32 => {
            const src_f32 = @as([*]const f32, @ptrCast(@alignCast(raw.ptr)))[0..n_elems];
            @memcpy(dst.data, src_f32);
        },
        .f16 => {
            dequantF16(dst.data, raw, n_elems);
        },
        .q4_0 => {
            dequantQ4_0(dst.data, raw, n_elems);
        },
        .q8_0 => {
            dequantQ8_0(dst.data, raw, n_elems);
        },
        else => {
            log.warn("tensor {s}: unsupported GGML type {}", .{ name, info.type_ });
            return error.UnsupportedType;
        },
    }
}

// ---------------------------------------------------------------------------
// Config extraction
// ---------------------------------------------------------------------------

/// Extract a LlamaConfig from GGUF metadata.
///
/// Reads `general.architecture` to determine the metadata key prefix (e.g. "llama"),
/// then maps standard GGUF metadata keys to LlamaConfig fields.
pub fn configFromGGUF(gf: *const GGUFFile) LlamaConfig {
    const arch = gf.getMetaString("general.architecture") orelse "llama";
    const n_heads = getArchU32(gf, arch, "attention.head_count") orelse 32;

    return .{
        .d_model = getArchU32(gf, arch, "embedding_length") orelse 4096,
        .n_layers = getArchU32(gf, arch, "block_count") orelse 32,
        .n_heads = n_heads,
        .n_kv_heads = getArchU32(gf, arch, "attention.head_count_kv") orelse n_heads,
        .d_ff = getArchU32(gf, arch, "feed_forward_length") orelse 11008,
        .max_seq_len = getArchU32(gf, arch, "context_length") orelse 2048,
        .rope_base = getArchF32(gf, arch, "rope.freq_base") orelse 10000.0,
        .vocab_size = getArchU32(gf, arch, "vocab_size") orelse blk: {
            // Fall back to counting token_embd rows if metadata missing.
            if (gf.getTensorInfo("token_embd.weight")) |ti| {
                break :blk @intCast(ti.dims[1]);
            }
            break :blk 32000;
        },
    };
}

/// Helper: look up `{arch}.{suffix}` as u32 in GGUF metadata.
fn getArchU32(gf: *const GGUFFile, arch: []const u8, suffix: []const u8) ?usize {
    var buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, suffix }) catch return null;
    const val = gf.getMetaU32(key) orelse return null;
    return @intCast(val);
}

/// Helper: look up `{arch}.{suffix}` as f32 in GGUF metadata.
fn getArchF32(gf: *const GGUFFile, arch: []const u8, suffix: []const u8) ?f32 {
    var buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, suffix }) catch return null;
    const val = gf.getMeta(key) orelse return null;
    return switch (val) {
        .float32 => |f| f,
        .float64 => |f| @floatCast(f),
        .uint32 => |v| @floatFromInt(v),
        else => null,
    };
}

// ---------------------------------------------------------------------------
// Block-level name builder
// ---------------------------------------------------------------------------

fn blockTensorName(buf: *[128]u8, layer: usize, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ layer, suffix }) catch "?";
}

// ---------------------------------------------------------------------------
// Full model loading
// ---------------------------------------------------------------------------

/// Load GGUF weights into a LLaMA model, dequantizing Q4_0/Q8_0/F16 to f32.
///
/// Tensor name mapping (GGUF LLaMA convention):
///   token_embd.weight          -> token_embed.inner
///   blk.{i}.attn_norm.weight   -> blocks[i].rms_norm_1.inner
///   blk.{i}.attn_q.weight      -> blocks[i].w_q.inner
///   blk.{i}.attn_k.weight      -> blocks[i].w_k.inner
///   blk.{i}.attn_v.weight      -> blocks[i].w_v.inner
///   blk.{i}.attn_output.weight -> blocks[i].w_o.inner
///   blk.{i}.ffn_norm.weight    -> blocks[i].rms_norm_2.inner
///   blk.{i}.ffn_gate.weight    -> blocks[i].w_gate.inner
///   blk.{i}.ffn_up.weight      -> blocks[i].w_up.inner
///   blk.{i}.ffn_down.weight    -> blocks[i].w_down.inner
///   output_norm.weight          -> rms_norm_f.inner
///   output.weight               -> out_proj.inner
pub fn loadDequantized(
    comptime T: type,
    comptime config: LlamaConfig,
    model: *const LLaMA(T, config),
    gf: *const GGUFFile,
) !void {
    if (T != f32) @compileError("loadDequantized only supports f32 currently");

    var name_buf: [128]u8 = undefined;

    // Token embedding
    loadTensor(T, model.token_embed.inner, gf, "token_embd.weight") catch |err| {
        if (err != error.TensorNotFound) return err;
    };

    // Transformer blocks
    for (0..config.n_layers) |i| {
        // Attention norm
        loadTensor(T, model.blocks[i].rms_norm_1.inner, gf, blockTensorName(&name_buf, i, "attn_norm.weight")) catch |err| {
            if (err != error.TensorNotFound) return err;
        };

        // Q/K/V/O projections
        loadTensor(T, model.blocks[i].w_q.inner, gf, blockTensorName(&name_buf, i, "attn_q.weight")) catch |err| {
            if (err != error.TensorNotFound) return err;
        };
        loadTensor(T, model.blocks[i].w_k.inner, gf, blockTensorName(&name_buf, i, "attn_k.weight")) catch |err| {
            if (err != error.TensorNotFound) return err;
        };
        loadTensor(T, model.blocks[i].w_v.inner, gf, blockTensorName(&name_buf, i, "attn_v.weight")) catch |err| {
            if (err != error.TensorNotFound) return err;
        };
        loadTensor(T, model.blocks[i].w_o.inner, gf, blockTensorName(&name_buf, i, "attn_output.weight")) catch |err| {
            if (err != error.TensorNotFound) return err;
        };

        // FFN norm
        loadTensor(T, model.blocks[i].rms_norm_2.inner, gf, blockTensorName(&name_buf, i, "ffn_norm.weight")) catch |err| {
            if (err != error.TensorNotFound) return err;
        };

        // FFN gate/up/down (SwiGLU)
        loadTensor(T, model.blocks[i].w_gate.inner, gf, blockTensorName(&name_buf, i, "ffn_gate.weight")) catch |err| {
            if (err != error.TensorNotFound) return err;
        };
        loadTensor(T, model.blocks[i].w_up.inner, gf, blockTensorName(&name_buf, i, "ffn_up.weight")) catch |err| {
            if (err != error.TensorNotFound) return err;
        };
        loadTensor(T, model.blocks[i].w_down.inner, gf, blockTensorName(&name_buf, i, "ffn_down.weight")) catch |err| {
            if (err != error.TensorNotFound) return err;
        };
    }

    // Final RMS norm
    loadTensor(T, model.rms_norm_f.inner, gf, "output_norm.weight") catch |err| {
        if (err != error.TensorNotFound) return err;
    };

    // Output projection
    if (!config.tied_lm_head) {
        loadTensor(T, model.out_proj.inner, gf, "output.weight") catch |err| {
            if (err != error.TensorNotFound) return err;
        };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "dequantQ4_0 round-trips with known values" {
    // Build a single Q4_0 block (18 bytes) for 32 elements.
    // Scale = 1.0 (f16), nibbles encode signed values via val+8.
    var block: [18]u8 = undefined;
    const scale: f16 = 1.0;
    const scale_bytes: [2]u8 = @bitCast(scale);
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];

    // Fill nibbles: element 0 = nibble 8 (val 0), element 1 = nibble 15 (val 7)
    // Packed: byte[0] low=8, high=15 -> 0xF8
    block[2] = 0xF8;
    // Rest zeros (val = -8 for each)
    for (block[3..]) |*b| b.* = 0;

    var dst: [32]f32 = undefined;
    dequantQ4_0(&dst, &block, 32);

    // Element 0: (8 - 8) * 1.0 = 0.0
    try testing.expectApproxEqAbs(@as(f32, 0.0), dst[0], 1e-6);
    // Element 1: (15 - 8) * 1.0 = 7.0
    try testing.expectApproxEqAbs(@as(f32, 7.0), dst[1], 1e-6);
    // Element 2: nibble = 0 -> (0 - 8) * 1.0 = -8.0
    try testing.expectApproxEqAbs(@as(f32, -8.0), dst[2], 1e-6);
}

test "dequantQ8_0 round-trips with known values" {
    // Build a single Q8_0 block (34 bytes) for 32 elements.
    var block: [34]u8 = undefined;
    const scale: f16 = 0.5;
    const scale_bytes: [2]u8 = @bitCast(scale);
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];

    // quant[0] = 10, quant[1] = -5
    block[2] = @bitCast(@as(i8, 10));
    block[3] = @bitCast(@as(i8, -5));
    for (block[4..]) |*b| b.* = 0;

    var dst: [32]f32 = undefined;
    dequantQ8_0(&dst, &block, 32);

    try testing.expectApproxEqAbs(@as(f32, 5.0), dst[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, -2.5), dst[1], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.0), dst[2], 1e-3);
}

test "dequantF16 converts correctly" {
    const vals = [_]f16{ 1.0, -0.5, 3.14 };
    var src: [6]u8 = undefined;
    for (vals, 0..) |v, i| {
        const bytes: [2]u8 = @bitCast(v);
        src[i * 2] = bytes[0];
        src[i * 2 + 1] = bytes[1];
    }

    var dst: [3]f32 = undefined;
    dequantF16(&dst, &src, 3);

    try testing.expectApproxEqAbs(@as(f32, 1.0), dst[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, -0.5), dst[1], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 3.14), dst[2], 0.01);
}

test "configFromGGUF extracts metadata" {
    const alloc = testing.allocator;

    // Build a synthetic GGUF buffer with LLaMA metadata.
    var aw: std.Io.Writer.Allocating = .init(alloc);
    defer aw.deinit();
    const writer = &aw.writer;

    // Magic + version 3
    try testWriteInt(u32, writer, 0x46475547);
    try testWriteInt(u32, writer, 3);
    // Tensor count: 0, KV count: 4
    try testWriteInt(u64, writer, 0);
    try testWriteInt(u64, writer, 4);

    // KV 1: general.architecture = "llama"
    try writeTestString(writer, "general.architecture");
    try testWriteInt(u32, writer, @intFromEnum(gguf_mod.MetaValueType.string));
    try writeTestString(writer, "llama");

    // KV 2: llama.embedding_length = 256
    try writeTestString(writer, "llama.embedding_length");
    try testWriteInt(u32, writer, @intFromEnum(gguf_mod.MetaValueType.uint32));
    try testWriteInt(u32, writer, 256);

    // KV 3: llama.block_count = 4
    try writeTestString(writer, "llama.block_count");
    try testWriteInt(u32, writer, @intFromEnum(gguf_mod.MetaValueType.uint32));
    try testWriteInt(u32, writer, 4);

    // KV 4: llama.attention.head_count = 8
    try writeTestString(writer, "llama.attention.head_count");
    try testWriteInt(u32, writer, @intFromEnum(gguf_mod.MetaValueType.uint32));
    try testWriteInt(u32, writer, 8);

    // Pad to alignment
    const header_end = aw.writer.end;
    const data_offset = GGUFFile.alignUp(header_end, 32);
    for (0..data_offset - header_end) |_| try writer.writeByte(0);

    const raw_data = aw.writer.buffer[0..aw.writer.end];
    const buf = try alloc.alignedAlloc(u8, .@"32", raw_data.len);
    @memcpy(buf, raw_data);

    var gf = try GGUFFile.parseBuffer(alloc, buf);
    defer gf.deinit();

    const cfg = configFromGGUF(&gf);
    try testing.expectEqual(@as(usize, 256), cfg.d_model);
    try testing.expectEqual(@as(usize, 4), cfg.n_layers);
    try testing.expectEqual(@as(usize, 8), cfg.n_heads);
    // Defaults for missing keys:
    try testing.expectEqual(@as(usize, 8), cfg.n_kv_heads); // default to n_heads
    try testing.expectEqual(@as(f32, 10000.0), cfg.rope_base); // default
}

test "blockTensorName formats correctly" {
    var buf: [128]u8 = undefined;
    const name = blockTensorName(&buf, 3, "attn_q.weight");
    try testing.expectEqualStrings("blk.3.attn_q.weight", name);
}

fn writeTestString(writer: *std.Io.Writer, s: []const u8) !void {
    try testWriteInt(u64, writer, @intCast(s.len));
    try writer.writeAll(s);
}

fn testWriteInt(comptime T: type, w: *std.Io.Writer, val: T) !void {
    var le = std.mem.nativeToLittle(T, val);
    try w.writeAll(std.mem.asBytes(&le));
}
