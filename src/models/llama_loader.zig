//! Weight loader for HuggingFace LLaMA models (safetensors format).
//!
//! Maps HuggingFace tensor names to zgml LLaMA model parameters and handles
//! the row-major → column-major transpose for 2D weight matrices.
//!
//! Supports both f32 and f16 safetensors weights (auto-detected via dtype).
//! For multi-shard models, call `loadLlama` with each shard — already-loaded
//! tensors are silently skipped.
//!
//! ```
//! var sf = try SafetensorsFile.open(alloc, "model.safetensors");
//! defer sf.deinit();
//! try loadLlama(f32, config, &model, &sf);
//! ```

const std = @import("std");
const SafetensorsFile = @import("../safetensors.zig").SafetensorsFile;
const Tensor = @import("../tensor.zig").Tensor;
const LLaMA = @import("llama.zig").LLaMA;
const LlamaConfig = @import("llama.zig").LlamaConfig;
const lu = @import("loader_utils.zig");

/// Load HuggingFace LLaMA weights from a safetensors file into a zgml LLaMA model.
///
/// For multi-shard models, call once per shard file. Missing tensors are
/// silently skipped (they'll be in another shard).
pub fn loadLlama(
    comptime T: type,
    comptime config: LlamaConfig,
    model: *const LLaMA(T, config),
    sf: *const SafetensorsFile,
) !void {
    if (T != f32) @compileError("loadLlama only supports f32");

    const d_model = config.d_model;
    const d_head = d_model / config.n_heads;
    const kv_dim = config.n_kv_heads * d_head;
    const prefix = "model.layers.";

    // --- Token embedding ---
    lu.loadWeight1D(model.token_embed.inner.data, sf, "model.embed_tokens.weight") catch {};

    // --- Transformer blocks ---
    for (0..config.n_layers) |layer| {
        var buf: [128]u8 = undefined;

        lu.loadWeight2D(model.blocks[layer].w_q.inner.data, sf, lu.layerName(&buf, prefix, layer, "self_attn.q_proj.weight") catch unreachable, d_model, d_model) catch {};
        lu.loadWeight2D(model.blocks[layer].w_k.inner.data, sf, lu.layerName(&buf, prefix, layer, "self_attn.k_proj.weight") catch unreachable, kv_dim, d_model) catch {};
        lu.loadWeight2D(model.blocks[layer].w_v.inner.data, sf, lu.layerName(&buf, prefix, layer, "self_attn.v_proj.weight") catch unreachable, kv_dim, d_model) catch {};
        lu.loadWeight2D(model.blocks[layer].w_o.inner.data, sf, lu.layerName(&buf, prefix, layer, "self_attn.o_proj.weight") catch unreachable, d_model, d_model) catch {};

        lu.loadWeight2D(model.blocks[layer].w_gate.inner.data, sf, lu.layerName(&buf, prefix, layer, "mlp.gate_proj.weight") catch unreachable, config.d_ff, d_model) catch {};
        lu.loadWeight2D(model.blocks[layer].w_up.inner.data, sf, lu.layerName(&buf, prefix, layer, "mlp.up_proj.weight") catch unreachable, config.d_ff, d_model) catch {};
        lu.loadWeight2D(model.blocks[layer].w_down.inner.data, sf, lu.layerName(&buf, prefix, layer, "mlp.down_proj.weight") catch unreachable, d_model, config.d_ff) catch {};

        lu.loadWeight1D(model.blocks[layer].rms_norm_1.inner.data, sf, lu.layerName(&buf, prefix, layer, "input_layernorm.weight") catch unreachable) catch {};
        lu.loadWeight1D(model.blocks[layer].rms_norm_2.inner.data, sf, lu.layerName(&buf, prefix, layer, "post_attention_layernorm.weight") catch unreachable) catch {};
    }

    // --- Final RMSNorm ---
    lu.loadWeight1D(model.rms_norm_f.inner.data, sf, "model.norm.weight") catch {};

    // --- Output projection (lm_head) ---
    if (!config.tied_lm_head) {
        lu.loadWeight2D(model.out_proj.inner.data, sf, "lm_head.weight", config.vocab_size, d_model) catch {};
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "llama loader - layerName formatting" {
    var buf: [128]u8 = undefined;
    const name = try lu.layerName(&buf, "model.layers.", 3, "self_attn.q_proj.weight");
    try testing.expectEqualStrings("model.layers.3.self_attn.q_proj.weight", name);
}
