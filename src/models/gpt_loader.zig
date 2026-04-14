//! Weight loader for HuggingFace GPT-Neo models (safetensors format).
//!
//! Maps HuggingFace tensor names to zgml GPT model parameters and handles
//! the row-major → column-major transpose for 2D weight matrices.
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

/// Copy a 1D tensor directly (no transpose needed).
fn copyDirect(dst: []f32, src: []const f32) void {
    @memcpy(dst, src);
}

/// Load a named tensor from the safetensors file as f32 data.
fn getTensor(sf: *const SafetensorsFile, name: []const u8) ![]const f32 {
    const meta = sf.findTensorMeta(name) orelse return error.TensorNotFound;
    return sf.getTensorF32(meta.offset_start, meta.offset_end);
}

/// Try to load a named tensor, returning null if not found.
fn getTensorOptional(sf: *const SafetensorsFile, name: []const u8) ?[]const f32 {
    const meta = sf.findTensorMeta(name) orelse return null;
    return sf.getTensorF32(meta.offset_start, meta.offset_end);
}

/// Load HuggingFace GPT-Neo weights from a safetensors file into a zgml GPT model.
///
/// Expects the model to be configured with:
///   - learnable_pos_embed = true
///   - learnable_ln = true
///   - attn_bias = true
///   - tied_lm_head = true
pub fn loadGPTNeo(
    comptime T: type,
    comptime config: GPTConfig,
    model: *const GPT(T, config),
    sf: *const SafetensorsFile,
) !void {
    if (T != f32) @compileError("loadGPTNeo only supports f32");

    const d_model = config.d_model;
    const n_heads = config.n_heads;
    const d_head = d_model / n_heads;

    // --- Embeddings ---
    // HF wte.weight: [vocab_size, d_model] row-major → element (tok, dim) at tok*d_model+dim
    // zgml token_embed: [d_model, vocab_size] col-major → element (dim, tok) at tok*d_model+dim
    // Same flat layout — direct copy.
    const wte = try getTensor(sf, "transformer.wte.weight");
    copyDirect(model.embed.token_embed.inner.data, wte);

    // Same reasoning for positional embeddings.
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

        // Attention Q/K/V projections: [d_model, d_model] → per-head [d_head, d_model]
        const q_w = try getTensor(sf, try layerName(&name_buf, layer, "attn.attention.q_proj.weight"));
        const k_w = try getTensor(sf, try layerName(&name_buf, layer, "attn.attention.k_proj.weight"));
        const v_w = try getTensor(sf, try layerName(&name_buf, layer, "attn.attention.v_proj.weight"));

        for (0..n_heads) |h| {
            // HF q_proj.weight [d_model, d_model] row-major: rows h*d_head..(h+1)*d_head
            // → zgml w_q[h] [d_head, d_model] col-major (transpose of the head slice)
            transposeHeadSlice(model.blocks[layer].w_q[h].inner.data, q_w, h, d_head, d_model);
            transposeHeadSlice(model.blocks[layer].w_k[h].inner.data, k_w, h, d_head, d_model);
            transposeHeadSlice(model.blocks[layer].w_v[h].inner.data, v_w, h, d_head, d_model);
        }

        // Output projection: [d_model, d_model] → per-head [d_model, d_head]
        const o_w = try getTensor(sf, try layerName(&name_buf, layer, "attn.attention.out_proj.weight"));
        for (0..n_heads) |h| {
            transposeOutProjSlice(model.blocks[layer].w_o[h].inner.data, o_w, h, d_head, d_model);
        }

        // Attention biases (Q/K/V biases are optional — some models don't have them)
        if (config.attn_bias) {
            if (getTensorOptional(sf, try layerName(&name_buf, layer, "attn.attention.q_proj.bias"))) |q_b| {
                for (0..n_heads) |h| {
                    copyDirect(model.blocks[layer].b_q[h].inner.data, q_b[h * d_head ..][0..d_head]);
                }
            }
            if (getTensorOptional(sf, try layerName(&name_buf, layer, "attn.attention.k_proj.bias"))) |k_b| {
                for (0..n_heads) |h| {
                    copyDirect(model.blocks[layer].b_k[h].inner.data, k_b[h * d_head ..][0..d_head]);
                }
            }
            if (getTensorOptional(sf, try layerName(&name_buf, layer, "attn.attention.v_proj.bias"))) |v_b| {
                for (0..n_heads) |h| {
                    copyDirect(model.blocks[layer].b_v[h].inner.data, v_b[h * d_head ..][0..d_head]);
                }
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

        // FFN: c_fc [d_ff, d_model] and c_proj [d_model, d_ff] — both need transpose
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

/// Build a layer-specific tensor name like "transformer.h.3.attn.attention.q_proj.weight".
fn layerName(buf: *[128]u8, layer: usize, suffix: []const u8) ![]const u8 {
    const prefix = "transformer.h.";
    var pos: usize = 0;

    @memcpy(buf[pos..][0..prefix.len], prefix);
    pos += prefix.len;

    // Write layer number
    const layer_str = try std.fmt.bufPrint(buf[pos..], "{d}", .{layer});
    pos += layer_str.len;

    buf[pos] = '.';
    pos += 1;

    @memcpy(buf[pos..][0..suffix.len], suffix);
    pos += suffix.len;

    return buf[0..pos];
}

/// Extract and transpose a head slice from a fused Q/K/V weight matrix.
/// HF: [d_model, d_model] row-major, head h occupies rows [h*d_head..(h+1)*d_head].
/// zgml: [d_head, d_model] col-major — transpose of that row slice.
fn transposeHeadSlice(dst: []f32, src: []const f32, h: usize, d_head: usize, d_model: usize) void {
    for (0..d_head) |r| {
        const src_row = h * d_head + r;
        for (0..d_model) |c| {
            // src[src_row, c] in row-major = src[src_row * d_model + c]
            // dst[r, c] in col-major [d_head, d_model] = dst[c * d_head + r]
            dst[c * d_head + r] = src[src_row * d_model + c];
        }
    }
}

/// Extract and transpose a head slice from the output projection weight.
/// HF: [d_model, d_model] row-major, head h occupies columns [h*d_head..(h+1)*d_head].
/// zgml: [d_model, d_head] col-major — transpose of that column slice.
fn transposeOutProjSlice(dst: []f32, src: []const f32, h: usize, d_head: usize, d_model: usize) void {
    for (0..d_model) |r| {
        for (0..d_head) |c| {
            const src_col = h * d_head + c;
            // src[r, src_col] in row-major = src[r * d_model + src_col]
            // dst[r, c] in col-major [d_model, d_head] = dst[c * d_model + r]
            dst[c * d_model + r] = src[r * d_model + src_col];
        }
    }
}
