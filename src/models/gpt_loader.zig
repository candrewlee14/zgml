//! Weight loader for HuggingFace GPT-Neo models (safetensors format).
//!
//! Maps HuggingFace tensor names to zgml GPT model parameters and handles
//! the row-major → column-major transpose for 2D weight matrices.
//!
//! The model uses packed QKV weights (`w_qkv [3*d_model, d_model]`) and a
//! packed output projection (`w_o [d_model, d_model]`).  This loader stacks
//! the separate HF Q/K/V weight matrices directly into the packed layout.
//!
//! ```
//! var sf = try SafetensorsFile.open(alloc, "model.safetensors");
//! defer sf.deinit();
//! try loadGPTNeo(f32, config, &model, &sf);
//! ```

const std = @import("std");
const SafetensorsFile = @import("../safetensors.zig").SafetensorsFile;
const Tensor = @import("../tensor.zig").Tensor;
const GPT = @import("gpt.zig").GPT;
const GPTConfig = @import("gpt.zig").GPTConfig;

/// Transpose a 2D matrix from row-major [rows, cols] to column-major [cols, rows].
fn transposeRowToCol(dst: []f32, src: []const f32, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        for (0..cols) |c| {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

/// Transpose a [rows, cols] row-major matrix into a row-offset region of a
/// packed column-major destination.  Used to stack Q/K/V into w_qkv.
///
/// `dst`: packed buffer, col-major with `dst_rows` total rows.
/// `row_offset`: first row in dst to write into.
fn transposeIntoPackedRows(
    dst: []f32,
    src: []const f32,
    rows: usize,
    cols: usize,
    dst_rows: usize,
    row_offset: usize,
) void {
    for (0..rows) |r| {
        for (0..cols) |c| {
            // dst col-major [dst_rows, cols]: element (row_offset+r, c) at c*dst_rows + row_offset + r
            dst[c * dst_rows + row_offset + r] = src[r * cols + c];
        }
    }
}

fn copyDirect(dst: []f32, src: []const f32) void {
    @memcpy(dst, src);
}

fn getTensor(sf: *const SafetensorsFile, name: []const u8) ![]const f32 {
    const meta = sf.findTensorMeta(name) orelse return error.TensorNotFound;
    return sf.getTensorF32(meta.offset_start, meta.offset_end);
}

fn getTensorOptional(sf: *const SafetensorsFile, name: []const u8) ?[]const f32 {
    const meta = sf.findTensorMeta(name) orelse return null;
    return sf.getTensorF32(meta.offset_start, meta.offset_end);
}

/// Load HuggingFace GPT-Neo weights from a safetensors file into a zgml GPT model.
pub fn loadGPTNeo(
    comptime T: type,
    comptime config: GPTConfig,
    model: *const GPT(T, config),
    sf: *const SafetensorsFile,
) !void {
    if (T != f32) @compileError("loadGPTNeo only supports f32");

    const d_model = config.d_model;

    // --- Embeddings ---
    const wte = try getTensor(sf, "transformer.wte.weight");
    copyDirect(model.embed.token_embed.inner.data, wte);

    if (config.learnable_pos_embed) {
        const wpe = try getTensor(sf, "transformer.wpe.weight");
        copyDirect(model.embed.pos_encode.inner.data, wpe);
    }

    // --- Transformer blocks ---
    for (0..config.n_layers) |layer| {
        var name_buf: [128]u8 = undefined;

        // Layer norm 1
        if (config.learnable_ln) {
            const ln1_w = try getTensor(sf, try layerName(&name_buf, layer, "ln_1.weight"));
            copyDirect(model.blocks[layer].ln1_gamma.inner.data, ln1_w);
            const ln1_b = try getTensor(sf, try layerName(&name_buf, layer, "ln_1.bias"));
            copyDirect(model.blocks[layer].ln1_beta.inner.data, ln1_b);
        }

        // Packed QKV: stack three [d_model, d_model] matrices into [3*d_model, d_model].
        const q_w = try getTensor(sf, try layerName(&name_buf, layer, "attn.attention.q_proj.weight"));
        const k_w = try getTensor(sf, try layerName(&name_buf, layer, "attn.attention.k_proj.weight"));
        const v_w = try getTensor(sf, try layerName(&name_buf, layer, "attn.attention.v_proj.weight"));

        const qkv_dst = model.blocks[layer].w_qkv.inner.data;
        transposeIntoPackedRows(qkv_dst, q_w, d_model, d_model, 3 * d_model, 0);
        transposeIntoPackedRows(qkv_dst, k_w, d_model, d_model, 3 * d_model, d_model);
        transposeIntoPackedRows(qkv_dst, v_w, d_model, d_model, 3 * d_model, 2 * d_model);

        // Output projection: [d_model, d_model] → direct transpose.
        const o_w = try getTensor(sf, try layerName(&name_buf, layer, "attn.attention.out_proj.weight"));
        transposeRowToCol(model.blocks[layer].w_o.inner.data, o_w, d_model, d_model);

        // Packed QKV bias: stack three [d_model] biases into [3*d_model].
        if (config.attn_bias) {
            const b_dst = model.blocks[layer].b_qkv.inner.data;
            if (getTensorOptional(sf, try layerName(&name_buf, layer, "attn.attention.q_proj.bias"))) |q_b| {
                @memcpy(b_dst[0..d_model], q_b);
            }
            if (getTensorOptional(sf, try layerName(&name_buf, layer, "attn.attention.k_proj.bias"))) |k_b| {
                @memcpy(b_dst[d_model..][0..d_model], k_b);
            }
            if (getTensorOptional(sf, try layerName(&name_buf, layer, "attn.attention.v_proj.bias"))) |v_b| {
                @memcpy(b_dst[2 * d_model ..][0..d_model], v_b);
            }
            const o_b = try getTensor(sf, try layerName(&name_buf, layer, "attn.attention.out_proj.bias"));
            copyDirect(model.blocks[layer].b_o.inner.data, o_b);
        }

        // Layer norm 2
        if (config.learnable_ln) {
            const ln2_w = try getTensor(sf, try layerName(&name_buf, layer, "ln_2.weight"));
            copyDirect(model.blocks[layer].ln2_gamma.inner.data, ln2_w);
            const ln2_b = try getTensor(sf, try layerName(&name_buf, layer, "ln_2.bias"));
            copyDirect(model.blocks[layer].ln2_beta.inner.data, ln2_b);
        }

        // FFN
        const fc_w = try getTensor(sf, try layerName(&name_buf, layer, "mlp.c_fc.weight"));
        transposeRowToCol(model.blocks[layer].w1.inner.data, fc_w, config.d_ff, d_model);
        const fc_b = try getTensor(sf, try layerName(&name_buf, layer, "mlp.c_fc.bias"));
        copyDirect(model.blocks[layer].b1.inner.data, fc_b);

        const proj_w = try getTensor(sf, try layerName(&name_buf, layer, "mlp.c_proj.weight"));
        transposeRowToCol(model.blocks[layer].w2.inner.data, proj_w, d_model, config.d_ff);
        const proj_b = try getTensor(sf, try layerName(&name_buf, layer, "mlp.c_proj.bias"));
        copyDirect(model.blocks[layer].b2.inner.data, proj_b);
    }

    // --- Final layer norm ---
    if (config.learnable_ln) {
        const ln_f_w = try getTensor(sf, "transformer.ln_f.weight");
        copyDirect(model.ln_f_gamma.inner.data, ln_f_w);
        const ln_f_b = try getTensor(sf, "transformer.ln_f.bias");
        copyDirect(model.ln_f_beta.inner.data, ln_f_b);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "transposeIntoPackedRows stacks Q/K/V correctly" {
    // Two 2×2 row-major matrices stacked into a packed [4, 2] col-major buffer.
    //   Q row-major: [[1,2],[3,4]]  →  col-major rows 0-1
    //   K row-major: [[5,6],[7,8]]  →  col-major rows 2-3
    var dst: [8]f32 = undefined;
    @memset(&dst, 0);

    const q = [_]f32{ 1, 2, 3, 4 }; // [2,2] row-major
    const k = [_]f32{ 5, 6, 7, 8 };

    transposeIntoPackedRows(&dst, &q, 2, 2, 4, 0);
    transposeIntoPackedRows(&dst, &k, 2, 2, 4, 2);

    // dst is col-major [4, 2]:
    //   col 0: [Q(0,0), Q(1,0), K(0,0), K(1,0)] = [1, 3, 5, 7]
    //   col 1: [Q(0,1), Q(1,1), K(0,1), K(1,1)] = [2, 4, 6, 8]
    try testing.expectEqual(@as(f32, 1), dst[0]); // col0 row0
    try testing.expectEqual(@as(f32, 3), dst[1]); // col0 row1
    try testing.expectEqual(@as(f32, 5), dst[2]); // col0 row2
    try testing.expectEqual(@as(f32, 7), dst[3]); // col0 row3
    try testing.expectEqual(@as(f32, 2), dst[4]); // col1 row0
    try testing.expectEqual(@as(f32, 4), dst[5]); // col1 row1
    try testing.expectEqual(@as(f32, 6), dst[6]); // col1 row2
    try testing.expectEqual(@as(f32, 8), dst[7]); // col1 row3
}

test "transposeRowToCol matches expected layout" {
    const src = [_]f32{ 1, 2, 3, 4, 5, 6 }; // [2, 3] row-major
    var dst: [6]f32 = undefined;

    transposeRowToCol(&dst, &src, 2, 3);

    // dst col-major [3, 2]:
    //   col 0: [src(0,0), src(1,0)] = [1, 4]
    //   col 1: [src(0,1), src(1,1)] = [2, 5]
    //   col 2: [src(0,2), src(1,2)] = [3, 6]
    try testing.expectEqual(@as(f32, 1), dst[0]);
    try testing.expectEqual(@as(f32, 4), dst[1]);
    try testing.expectEqual(@as(f32, 2), dst[2]);
    try testing.expectEqual(@as(f32, 5), dst[3]);
    try testing.expectEqual(@as(f32, 3), dst[4]);
    try testing.expectEqual(@as(f32, 6), dst[5]);
}

fn layerName(buf: *[128]u8, layer: usize, suffix: []const u8) ![]const u8 {
    const prefix = "transformer.h.";
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
