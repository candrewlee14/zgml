//! Weight loader for HuggingFace Whisper models (safetensors format).
//!
//! Maps HuggingFace tensor names to zgml Whisper model parameters and handles
//! the row-major → column-major transpose for 2D weight matrices.
//!
//! ```
//! var sf = try SafetensorsFile.open(alloc, "model.safetensors");
//! defer sf.deinit();
//! try loadWhisper(f32, config, &model, &sf);
//! ```

const std = @import("std");
const SafetensorsFile = @import("../safetensors.zig").SafetensorsFile;
const Tensor = @import("../tensor.zig").Tensor;
const Whisper = @import("whisper.zig").Whisper;
const WhisperConfig = @import("whisper.zig").WhisperConfig;
const lu = @import("loader_utils.zig");

/// Load HuggingFace Whisper weights from a safetensors file.
pub fn loadWhisper(
    comptime T: type,
    comptime config: WhisperConfig,
    model: *const Whisper(T, config),
    sf: *const SafetensorsFile,
) !void {
    if (T != f32) @compileError("loadWhisper only supports f32");

    const d = config.d_model;
    const enc_prefix = "model.encoder.layers.";
    const dec_prefix = "model.decoder.layers.";

    // --- Encoder convolutions ---
    lu.loadWeight1D(model.conv1_kernel.inner.data, sf, "model.encoder.conv1.weight") catch {};
    lu.loadWeight1D(model.conv1_bias.inner.data, sf, "model.encoder.conv1.bias") catch {};
    lu.loadWeight1D(model.conv2_kernel.inner.data, sf, "model.encoder.conv2.weight") catch {};
    lu.loadWeight1D(model.conv2_bias.inner.data, sf, "model.encoder.conv2.bias") catch {};

    lu.loadWeight1D(model.pos_embed_enc.inner.data, sf, "model.encoder.embed_positions.weight") catch {};

    // --- Encoder blocks ---
    for (0..config.n_encoder_layers) |layer| {
        var buf: [128]u8 = undefined;

        // Packed QKV: stack separate Q/K/V into [3*d, d]
        const qkv_dst = model.encoder_blocks[layer].w_qkv.inner.data;
        lu.transposeIntoPacked(qkv_dst, sf, lu.layerName(&buf, enc_prefix, layer, "self_attn.q_proj.weight") catch unreachable, d, d, 3 * d, 0);
        lu.transposeIntoPacked(qkv_dst, sf, lu.layerName(&buf, enc_prefix, layer, "self_attn.k_proj.weight") catch unreachable, d, d, 3 * d, d);
        lu.transposeIntoPacked(qkv_dst, sf, lu.layerName(&buf, enc_prefix, layer, "self_attn.v_proj.weight") catch unreachable, d, d, 3 * d, 2 * d);

        lu.loadWeight2D(model.encoder_blocks[layer].w_o.inner.data, sf, lu.layerName(&buf, enc_prefix, layer, "self_attn.out_proj.weight") catch unreachable, d, d) catch {};

        // Packed QKV bias
        const bqkv_dst = model.encoder_blocks[layer].b_qkv.inner.data;
        lu.loadIntoSlice(bqkv_dst, 0, d, sf, lu.layerName(&buf, enc_prefix, layer, "self_attn.q_proj.bias") catch unreachable);
        lu.loadIntoSlice(bqkv_dst, d, 2 * d, sf, lu.layerName(&buf, enc_prefix, layer, "self_attn.k_proj.bias") catch unreachable);
        lu.loadIntoSlice(bqkv_dst, 2 * d, 3 * d, sf, lu.layerName(&buf, enc_prefix, layer, "self_attn.v_proj.bias") catch unreachable);
        lu.loadWeight1D(model.encoder_blocks[layer].b_o.inner.data, sf, lu.layerName(&buf, enc_prefix, layer, "self_attn.out_proj.bias") catch unreachable) catch {};

        // Layer norms
        lu.loadWeight1D(model.encoder_blocks[layer].ln1_gamma.inner.data, sf, lu.layerName(&buf, enc_prefix, layer, "self_attn_layer_norm.weight") catch unreachable) catch {};
        lu.loadWeight1D(model.encoder_blocks[layer].ln1_beta.inner.data, sf, lu.layerName(&buf, enc_prefix, layer, "self_attn_layer_norm.bias") catch unreachable) catch {};
        lu.loadWeight1D(model.encoder_blocks[layer].ln2_gamma.inner.data, sf, lu.layerName(&buf, enc_prefix, layer, "final_layer_norm.weight") catch unreachable) catch {};
        lu.loadWeight1D(model.encoder_blocks[layer].ln2_beta.inner.data, sf, lu.layerName(&buf, enc_prefix, layer, "final_layer_norm.bias") catch unreachable) catch {};

        // FFN
        lu.loadWeight2D(model.encoder_blocks[layer].w1.inner.data, sf, lu.layerName(&buf, enc_prefix, layer, "fc1.weight") catch unreachable, config.d_ff, d) catch {};
        lu.loadWeight1D(model.encoder_blocks[layer].b1.inner.data, sf, lu.layerName(&buf, enc_prefix, layer, "fc1.bias") catch unreachable) catch {};
        lu.loadWeight2D(model.encoder_blocks[layer].w2.inner.data, sf, lu.layerName(&buf, enc_prefix, layer, "fc2.weight") catch unreachable, d, config.d_ff) catch {};
        lu.loadWeight1D(model.encoder_blocks[layer].b2.inner.data, sf, lu.layerName(&buf, enc_prefix, layer, "fc2.bias") catch unreachable) catch {};
    }

    lu.loadWeight1D(model.enc_ln_gamma.inner.data, sf, "model.encoder.layer_norm.weight") catch {};
    lu.loadWeight1D(model.enc_ln_beta.inner.data, sf, "model.encoder.layer_norm.bias") catch {};

    // --- Decoder ---
    lu.loadWeight1D(model.token_embed.inner.data, sf, "model.decoder.embed_tokens.weight") catch {};
    lu.loadWeight1D(model.pos_embed_dec.inner.data, sf, "model.decoder.embed_positions.weight") catch {};

    for (0..config.n_decoder_layers) |layer| {
        var buf: [128]u8 = undefined;

        // Self-attention
        lu.loadWeight2D(model.decoder_blocks[layer].sa_w_q.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "self_attn.q_proj.weight") catch unreachable, d, d) catch {};
        lu.loadWeight2D(model.decoder_blocks[layer].sa_w_k.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "self_attn.k_proj.weight") catch unreachable, d, d) catch {};
        lu.loadWeight2D(model.decoder_blocks[layer].sa_w_v.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "self_attn.v_proj.weight") catch unreachable, d, d) catch {};
        lu.loadWeight2D(model.decoder_blocks[layer].sa_w_o.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "self_attn.out_proj.weight") catch unreachable, d, d) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].sa_b_q.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "self_attn.q_proj.bias") catch unreachable) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].sa_b_k.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "self_attn.k_proj.bias") catch unreachable) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].sa_b_v.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "self_attn.v_proj.bias") catch unreachable) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].sa_b_o.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "self_attn.out_proj.bias") catch unreachable) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].sa_ln_gamma.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "self_attn_layer_norm.weight") catch unreachable) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].sa_ln_beta.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "self_attn_layer_norm.bias") catch unreachable) catch {};

        // Cross-attention
        lu.loadWeight2D(model.decoder_blocks[layer].ca_w_q.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "encoder_attn.q_proj.weight") catch unreachable, d, d) catch {};
        lu.loadWeight2D(model.decoder_blocks[layer].ca_w_k.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "encoder_attn.k_proj.weight") catch unreachable, d, d) catch {};
        lu.loadWeight2D(model.decoder_blocks[layer].ca_w_v.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "encoder_attn.v_proj.weight") catch unreachable, d, d) catch {};
        lu.loadWeight2D(model.decoder_blocks[layer].ca_w_o.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "encoder_attn.out_proj.weight") catch unreachable, d, d) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].ca_b_q.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "encoder_attn.q_proj.bias") catch unreachable) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].ca_b_k.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "encoder_attn.k_proj.bias") catch unreachable) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].ca_b_v.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "encoder_attn.v_proj.bias") catch unreachable) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].ca_b_o.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "encoder_attn.out_proj.bias") catch unreachable) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].ca_ln_gamma.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "encoder_attn_layer_norm.weight") catch unreachable) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].ca_ln_beta.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "encoder_attn_layer_norm.bias") catch unreachable) catch {};

        // FFN
        lu.loadWeight2D(model.decoder_blocks[layer].w1.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "fc1.weight") catch unreachable, config.d_ff, d) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].b1.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "fc1.bias") catch unreachable) catch {};
        lu.loadWeight2D(model.decoder_blocks[layer].w2.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "fc2.weight") catch unreachable, d, config.d_ff) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].b2.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "fc2.bias") catch unreachable) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].ffn_ln_gamma.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "final_layer_norm.weight") catch unreachable) catch {};
        lu.loadWeight1D(model.decoder_blocks[layer].ffn_ln_beta.inner.data, sf, lu.layerName(&buf, dec_prefix, layer, "final_layer_norm.bias") catch unreachable) catch {};
    }

    lu.loadWeight1D(model.dec_ln_gamma.inner.data, sf, "model.decoder.layer_norm.weight") catch {};
    lu.loadWeight1D(model.dec_ln_beta.inner.data, sf, "model.decoder.layer_norm.bias") catch {};
}
