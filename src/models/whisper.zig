//! OpenAI Whisper speech-to-text model.
//!
//! Encoder-decoder transformer architecture:
//!   Encoder: Conv1d feature extraction → N encoder blocks (non-causal self-attention)
//!   Decoder: Token embedding → N decoder blocks (causal self-attn + cross-attn) → logits
//!
//! The encoder processes mel spectrograms [n_mels, n_frames] and the decoder
//! autoregressively generates token IDs.
//!
//! ```
//! const model = try Whisper(f32, .{
//!     .n_mels = 80,
//!     .d_model = 512,
//!     .n_heads = 8,
//!     .d_ff = 2048,
//!     .n_encoder_layers = 6,
//!     .n_decoder_layers = 6,
//!     .vocab_size = 51865,
//!     .max_audio_ctx = 1500,
//!     .max_decoder_len = 448,
//! }).init(alloc);
//!
//! const encoder_out = model.encode(mel_spectrogram);
//! const logits = model.decode(encoder_out, token_indices);
//! ```

const std = @import("std");
const testing = std.testing;
const tac = testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Alloc = std.mem.Allocator;
const shaped_mod = @import("../shaped.zig");
const Shaped = shaped_mod.Shaped;
const TransformerBlock = @import("transformer.zig").TransformerBlock;
const nn = @import("../nn.zig");

pub const WhisperConfig = struct {
    n_mels: usize = 80,
    d_model: usize = 512,
    n_heads: usize = 8,
    d_ff: usize = 2048,
    n_encoder_layers: usize = 6,
    n_decoder_layers: usize = 6,
    vocab_size: usize = 51865,
    max_audio_ctx: usize = 1500,
    max_decoder_len: usize = 448,
};

/// Whisper decoder block: causal self-attention + encoder cross-attention + FFN.
///
/// This extends the standard TransformerBlock with a cross-attention sublayer
/// where Q comes from the decoder and K/V come from the encoder output.
pub fn WhisperDecoderBlock(
    comptime T: type,
    comptime d_model: usize,
    comptime n_heads: usize,
    comptime d_ff: usize,
) type {
    const d_head = d_model / n_heads;

    const WShape = Shaped(T, .{ d_model, d_model });
    const BShape = Shaped(T, .{d_model});
    const LnShape = Shaped(T, .{d_model});
    const W1Shape = Shaped(T, .{ d_ff, d_model });
    const B1Shape = Shaped(T, .{d_ff});
    const W2Shape = Shaped(T, .{ d_model, d_ff });
    const B2Shape = Shaped(T, .{d_model});

    return struct {
        const Self = @This();

        // Self-attention (causal)
        sa_w_q: WShape,
        sa_w_k: WShape,
        sa_w_v: WShape,
        sa_w_o: WShape,
        sa_b_q: BShape,
        sa_b_k: BShape,
        sa_b_v: BShape,
        sa_b_o: BShape,
        sa_ln_gamma: LnShape,
        sa_ln_beta: LnShape,

        // Cross-attention (encoder → decoder)
        ca_w_q: WShape,
        ca_w_k: WShape,
        ca_w_v: WShape,
        ca_w_o: WShape,
        ca_b_q: BShape,
        ca_b_k: BShape,
        ca_b_v: BShape,
        ca_b_o: BShape,
        ca_ln_gamma: LnShape,
        ca_ln_beta: LnShape,

        // FFN
        w1: W1Shape,
        b1: B1Shape,
        w2: W2Shape,
        b2: B2Shape,
        ffn_ln_gamma: LnShape,
        ffn_ln_beta: LnShape,

        pub fn init(alloc: Alloc) !Self {
            var self: Self = undefined;
            var seed: u64 = 42;

            inline for (.{
                "sa_w_q",  "sa_w_k",  "sa_w_v",  "sa_w_o",
                "ca_w_q",  "ca_w_k",  "ca_w_v",  "ca_w_o",
            }) |name| {
                @field(self, name) = try WShape.init(alloc);
                nn.kaimingUniform(T, @field(self, name).inner, seed);
                @field(self, name).inner.setParam();
                seed +%= 1;
            }

            inline for (.{
                "sa_b_q",  "sa_b_k",  "sa_b_v",  "sa_b_o",
                "ca_b_q",  "ca_b_k",  "ca_b_v",  "ca_b_o",
            }) |name| {
                @field(self, name) = try BShape.init(alloc);
                _ = @field(self, name).inner.setAllScalar(0);
                @field(self, name).inner.setParam();
            }

            inline for (.{
                "sa_ln_gamma", "ca_ln_gamma", "ffn_ln_gamma",
            }) |name| {
                @field(self, name) = try LnShape.init(alloc);
                _ = @field(self, name).inner.setAllScalar(1);
                @field(self, name).inner.setParam();
            }

            inline for (.{
                "sa_ln_beta", "ca_ln_beta", "ffn_ln_beta",
            }) |name| {
                @field(self, name) = try LnShape.init(alloc);
                _ = @field(self, name).inner.setAllScalar(0);
                @field(self, name).inner.setParam();
            }

            self.w1 = try W1Shape.init(alloc);
            nn.kaimingUniform(T, self.w1.inner, seed);
            self.w1.inner.setParam();
            seed +%= 1;
            self.b1 = try B1Shape.init(alloc);
            _ = self.b1.inner.setAllScalar(0);
            self.b1.inner.setParam();

            self.w2 = try W2Shape.init(alloc);
            nn.kaimingUniform(T, self.w2.inner, seed);
            self.w2.inner.setParam();
            self.b2 = try B2Shape.init(alloc);
            _ = self.b2.inner.setAllScalar(0);
            self.b2.inner.setParam();

            return self;
        }

        fn applyLn(x: *Tensor(T), gamma: *Tensor(T), beta: *Tensor(T), ln_reduce: []usize) *Tensor(T) {
            const normed = x.layerNorm(ln_reduce, 1e-5);
            return normed.mul(gamma.repeatLike(normed)).addBias(beta);
        }

        fn multiHeadAttn(
            q_proj: *Tensor(T),
            k_proj: *Tensor(T),
            v_proj: *Tensor(T),
            w_o: *Tensor(T),
            b_o: *Tensor(T),
            mask: ?*Tensor(T),
            sm_reduce: []usize,
        ) *Tensor(T) {
            var attn_sum: ?*Tensor(T) = null;
            for (0..n_heads) |h| {
                const q_h = q_proj.sliceRows(h * d_head, (h + 1) * d_head);
                const k_h = k_proj.sliceRows(h * d_head, (h + 1) * d_head);
                const v_h = v_proj.sliceRows(h * d_head, (h + 1) * d_head);

                const scores = q_h.matMul(false, k_h, true);
                const dk: T = @floatFromInt(d_head);
                var scaled = scores.scaleByVal(1.0 / @sqrt(dk));
                if (mask) |m| scaled = scaled.add(m);
                const weights = scaled.softmax(sm_reduce);
                const attn_out = weights.matMul(false, v_h, false);

                // Output projection per head via column slice
                const w_o_h = w_o.sliceColumns(h * d_head, (h + 1) * d_head);
                const projected = attn_out.matMul(false, w_o_h, false);
                attn_sum = if (attn_sum) |acc| acc.add(projected) else projected;
            }

            return attn_sum.?.addBias(b_o);
        }

        /// Full-sequence decoder forward.
        /// `x`: decoder hidden state [d_model, dec_seq]
        /// `encoder_out`: encoder output [d_model, enc_seq]
        pub fn forward(self: *const Self, x: *Tensor(T), encoder_out: *Tensor(T)) *Tensor(T) {
            const alloc = x.alloc.?;
            const dec_seq = x.ne[1];

            // 1. Causal self-attention
            var ln_reduce = [_]usize{ 1, dec_seq };
            const sa_norm = applyLn(x, self.sa_ln_gamma.inner, self.sa_ln_beta.inner, &ln_reduce);
            const sa_q = nn.linear(T, sa_norm, self.sa_w_q.inner, self.sa_b_q.inner);
            const sa_k = nn.linear(T, sa_norm, self.sa_w_k.inner, self.sa_b_k.inner);
            const sa_v = nn.linear(T, sa_norm, self.sa_w_v.inner, self.sa_b_v.inner);

            const causal_mask = nn.buildCausalMask(T, alloc, dec_seq);

            var sm_reduce_sa = [_]usize{ 1, dec_seq };
            const sa_out = multiHeadAttn(sa_q, sa_k, sa_v, self.sa_w_o.inner, self.sa_b_o.inner, causal_mask, &sm_reduce_sa);
            const after_sa = x.add(sa_out);

            // 2. Cross-attention (Q from decoder, K/V from encoder)
            const ca_norm = applyLn(after_sa, self.ca_ln_gamma.inner, self.ca_ln_beta.inner, &ln_reduce);
            const ca_q = nn.linear(T, ca_norm, self.ca_w_q.inner, self.ca_b_q.inner);
            const ca_k = nn.linear(T, encoder_out, self.ca_w_k.inner, self.ca_b_k.inner);
            const ca_v = nn.linear(T, encoder_out, self.ca_w_v.inner, self.ca_b_v.inner);

            var sm_reduce_ca = [_]usize{ 1, dec_seq };
            const ca_out = multiHeadAttn(ca_q, ca_k, ca_v, self.ca_w_o.inner, self.ca_b_o.inner, null, &sm_reduce_ca);
            const after_ca = after_sa.add(ca_out);

            // 3. FFN
            const ffn_norm = applyLn(after_ca, self.ffn_ln_gamma.inner, self.ffn_ln_beta.inner, &ln_reduce);
            const activated = nn.linear(T, ffn_norm, self.w1.inner, self.b1.inner).gelu();
            const ffn_out = nn.linear(T, activated, self.w2.inner, self.b2.inner);
            return after_ca.add(ffn_out);
        }

        pub const n_block_params = 26;

        pub fn params(self: *const Self) [n_block_params]*Tensor(T) {
            return .{
                // Self-attention
                self.sa_w_q.inner,  self.sa_w_k.inner,  self.sa_w_v.inner,  self.sa_w_o.inner,
                self.sa_b_q.inner,  self.sa_b_k.inner,  self.sa_b_v.inner,  self.sa_b_o.inner,
                self.sa_ln_gamma.inner, self.sa_ln_beta.inner,
                // Cross-attention
                self.ca_w_q.inner,  self.ca_w_k.inner,  self.ca_w_v.inner,  self.ca_w_o.inner,
                self.ca_b_q.inner,  self.ca_b_k.inner,  self.ca_b_v.inner,  self.ca_b_o.inner,
                self.ca_ln_gamma.inner, self.ca_ln_beta.inner,
                // FFN
                self.w1.inner, self.b1.inner, self.w2.inner, self.b2.inner,
                self.ffn_ln_gamma.inner, self.ffn_ln_beta.inner,
            };
        }
    };
}

/// Full Whisper model: encoder + decoder.
pub fn Whisper(comptime T: type, comptime config: WhisperConfig) type {
    const d_model = config.d_model;
    const n_heads = config.n_heads;
    const d_ff = config.d_ff;

    // Encoder uses standard TransformerBlock (non-causal, with bias and learnable LN)
    const EncoderBlock = TransformerBlock(T, d_model, n_heads, d_ff, false, true, true);
    const DecoderBlock = WhisperDecoderBlock(T, d_model, n_heads, d_ff);

    // Conv1d weights: stored as [C_out, C_in, kW] in HF, transposed to [kW, C_in, C_out] for zgml conv2d
    // Conv1 shape: [d_model, n_mels, 3] (kernel_size=3)
    // Conv2 shape: [d_model, d_model, 3] (kernel_size=3, stride=2)
    const Conv1KernelShape = Shaped(T, .{ 3, 1, config.n_mels, d_model });
    const Conv2KernelShape = Shaped(T, .{ 3, 1, d_model, d_model });
    const ConvBiasShape = Shaped(T, .{d_model});

    const EmbedShape = Shaped(T, .{ d_model, config.vocab_size });
    const PosEmbedEncShape = Shaped(T, .{ d_model, config.max_audio_ctx });
    const PosEmbedDecShape = Shaped(T, .{ d_model, config.max_decoder_len });
    const LnShape = Shaped(T, .{d_model});

    const enc_block_params = EncoderBlock.n_block_params;
    const dec_block_params = DecoderBlock.n_block_params;
    // conv1 kernel + bias + conv2 kernel + bias + pos_embed_enc + encoder blocks + enc_ln
    // + token_embed + pos_embed_dec + decoder blocks + dec_ln
    const total_params = 2 + 2 + 1 + enc_block_params * config.n_encoder_layers + 2 + 1 + 1 + dec_block_params * config.n_decoder_layers + 2;

    return struct {
        const Self = @This();

        // Encoder
        conv1_kernel: Conv1KernelShape,
        conv1_bias: ConvBiasShape,
        conv2_kernel: Conv2KernelShape,
        conv2_bias: ConvBiasShape,
        pos_embed_enc: PosEmbedEncShape,
        encoder_blocks: [config.n_encoder_layers]EncoderBlock,
        enc_ln_gamma: LnShape,
        enc_ln_beta: LnShape,

        // Decoder
        token_embed: EmbedShape,
        pos_embed_dec: PosEmbedDecShape,
        decoder_blocks: [config.n_decoder_layers]DecoderBlock,
        dec_ln_gamma: LnShape,
        dec_ln_beta: LnShape,

        pub fn init(alloc: Alloc) !Self {
            var self: Self = undefined;
            var seed: u64 = 42;

            // Encoder convolutions
            self.conv1_kernel = try Conv1KernelShape.init(alloc);
            nn.kaimingUniform(T, self.conv1_kernel.inner, seed);
            self.conv1_kernel.inner.setParam();
            seed +%= 1;
            self.conv1_bias = try ConvBiasShape.init(alloc);
            _ = self.conv1_bias.inner.setAllScalar(0);
            self.conv1_bias.inner.setParam();

            self.conv2_kernel = try Conv2KernelShape.init(alloc);
            nn.kaimingUniform(T, self.conv2_kernel.inner, seed);
            self.conv2_kernel.inner.setParam();
            seed +%= 1;
            self.conv2_bias = try ConvBiasShape.init(alloc);
            _ = self.conv2_bias.inner.setAllScalar(0);
            self.conv2_bias.inner.setParam();

            // Positional embeddings (encoder: sinusoidal, decoder: learned)
            self.pos_embed_enc = try PosEmbedEncShape.init(alloc);
            for (0..config.max_audio_ctx) |pos| {
                for (0..d_model) |i| {
                    const p: T = @floatFromInt(pos);
                    const dim_pair: T = @floatFromInt(2 * (i / 2));
                    const dm: T = @floatFromInt(d_model);
                    const angle = p / std.math.pow(T, 10000.0, dim_pair / dm);
                    self.pos_embed_enc.inner.data[pos * d_model + i] = if (i % 2 == 0) @sin(angle) else @cos(angle);
                }
            }
            self.pos_embed_enc.inner.setParam();

            self.pos_embed_dec = try PosEmbedDecShape.init(alloc);
            _ = self.pos_embed_dec.inner.setAllScalar(0);
            self.pos_embed_dec.inner.setParam();

            // Encoder blocks
            for (0..config.n_encoder_layers) |i| {
                self.encoder_blocks[i] = try EncoderBlock.init(alloc);
            }

            // Encoder final LN
            self.enc_ln_gamma = try LnShape.init(alloc);
            self.enc_ln_beta = try LnShape.init(alloc);
            _ = self.enc_ln_gamma.inner.setAllScalar(1);
            _ = self.enc_ln_beta.inner.setAllScalar(0);
            self.enc_ln_gamma.inner.setParam();
            self.enc_ln_beta.inner.setParam();

            // Decoder
            self.token_embed = try EmbedShape.init(alloc);
            const scale: T = 1.0 / @sqrt(@as(T, @floatFromInt(d_model)));
            for (self.token_embed.inner.data, 0..) |*d, i| {
                const fi: T = @floatFromInt(i);
                d.* = scale * @sin(fi * 0.1 + 0.3) * @cos(fi * 0.07 + 0.5);
            }
            self.token_embed.inner.setParam();

            for (0..config.n_decoder_layers) |i| {
                self.decoder_blocks[i] = try DecoderBlock.init(alloc);
            }

            self.dec_ln_gamma = try LnShape.init(alloc);
            self.dec_ln_beta = try LnShape.init(alloc);
            _ = self.dec_ln_gamma.inner.setAllScalar(1);
            _ = self.dec_ln_beta.inner.setAllScalar(0);
            self.dec_ln_gamma.inner.setParam();
            self.dec_ln_beta.inner.setParam();

            return self;
        }

        /// Encode pre-processed audio features through transformer blocks.
        ///
        /// `audio_features`: [d_model, enc_seq] — pre-processed features
        /// (after conv layers and positional encoding, done externally or by loader).
        /// Returns: [d_model, enc_seq] encoder output.
        pub fn encodeFeatures(self: *const Self, audio_features: *Tensor(T)) *Tensor(T) {
            const enc_seq = audio_features.ne[1];

            // Add positional encoding
            const alloc = audio_features.alloc.?;
            const pos_enc = Tensor(T).init(alloc, &.{ d_model, enc_seq }) catch unreachable;
            @memcpy(pos_enc.data[0 .. d_model * enc_seq], self.pos_embed_enc.inner.data[0 .. d_model * enc_seq]);

            var x = audio_features.add(pos_enc.repeatLike(audio_features));

            // Encoder blocks (non-causal self-attention)
            for (0..config.n_encoder_layers) |i| {
                x = self.encoder_blocks[i].forward(x);
            }

            // Final layer norm
            var ln_reduce = [_]usize{ 1, enc_seq };
            const normed = x.layerNorm(&ln_reduce, 1e-5);
            return normed.mul(self.enc_ln_gamma.inner.repeatLike(normed)).addBias(self.enc_ln_beta.inner);
        }

        /// Decode: generate logits from encoder output and token indices.
        ///
        /// `encoder_out`: [d_model, enc_seq] — encoder hidden states
        /// `token_indices`: [dec_seq] — decoder input token IDs
        /// Returns: [vocab_size, dec_seq] logits
        pub fn decode(self: *const Self, encoder_out: *Tensor(T), token_indices: *Tensor(T)) *Tensor(T) {
            const dec_seq = token_indices.ne[0];
            const alloc = token_indices.alloc.?;

            // Token embedding + positional encoding
            var x = self.token_embed.inner.gatherRows(token_indices);
            const pos_enc = Tensor(T).init(alloc, &.{ d_model, dec_seq }) catch unreachable;
            @memcpy(pos_enc.data[0 .. d_model * dec_seq], self.pos_embed_dec.inner.data[0 .. d_model * dec_seq]);
            x = x.add(pos_enc.repeatLike(x));

            // Decoder blocks
            for (0..config.n_decoder_layers) |i| {
                x = self.decoder_blocks[i].forward(x, encoder_out);
            }

            // Final layer norm
            var ln_reduce = [_]usize{ 1, dec_seq };
            const normed = x.layerNorm(&ln_reduce, 1e-5);
            x = normed.mul(self.dec_ln_gamma.inner.repeatLike(normed)).addBias(self.dec_ln_beta.inner);

            // Output projection (tied with token embedding)
            return x.matMul(false, self.token_embed.inner, true);
        }

        /// Return all learnable parameters.
        pub fn params(self: *const Self) [total_params]*Tensor(T) {
            var result: [total_params]*Tensor(T) = undefined;
            var idx: usize = 0;

            // Encoder conv
            result[idx] = self.conv1_kernel.inner;
            idx += 1;
            result[idx] = self.conv1_bias.inner;
            idx += 1;
            result[idx] = self.conv2_kernel.inner;
            idx += 1;
            result[idx] = self.conv2_bias.inner;
            idx += 1;
            result[idx] = self.pos_embed_enc.inner;
            idx += 1;

            for (0..config.n_encoder_layers) |i| {
                for (self.encoder_blocks[i].params()) |p| {
                    result[idx] = p;
                    idx += 1;
                }
            }

            result[idx] = self.enc_ln_gamma.inner;
            idx += 1;
            result[idx] = self.enc_ln_beta.inner;
            idx += 1;

            result[idx] = self.token_embed.inner;
            idx += 1;
            result[idx] = self.pos_embed_dec.inner;
            idx += 1;

            for (0..config.n_decoder_layers) |i| {
                for (self.decoder_blocks[i].params()) |p| {
                    result[idx] = p;
                    idx += 1;
                }
            }

            result[idx] = self.dec_ln_gamma.inner;
            idx += 1;
            result[idx] = self.dec_ln_beta.inner;
            idx += 1;

            std.debug.assert(idx == total_params);
            return result;
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "whisper decoder block - forward produces valid output" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const block = try WhisperDecoderBlock(f32, 8, 2, 16).init(a);

    const x = try Tensor(f32).init(a, &.{ 8, 3 }); // decoder hidden [d_model, dec_seq]
    nn.uniform(f32, x, -0.1, 0.1, 42);
    const enc = try Tensor(f32).init(a, &.{ 8, 5 }); // encoder out [d_model, enc_seq]
    nn.uniform(f32, enc, -0.1, 0.1, 43);

    const out = block.forward(x, enc);
    try g.infer(out);

    try testing.expectEqual(@as(usize, 8), out.ne[0]);
    try testing.expectEqual(@as(usize, 3), out.ne[1]);
    for (out.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "whisper - encoder forward produces valid output" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const model = try Whisper(f32, .{
        .n_mels = 8,
        .d_model = 8,
        .n_heads = 2,
        .d_ff = 16,
        .n_encoder_layers = 1,
        .n_decoder_layers = 1,
        .vocab_size = 32,
        .max_audio_ctx = 16,
        .max_decoder_len = 8,
    }).init(a);

    // Pre-processed audio features (skip conv layers for now)
    const audio = try Tensor(f32).init(a, &.{ 8, 5 });
    nn.uniform(f32, audio, -0.1, 0.1, 42);

    const enc_out = model.encodeFeatures(audio);
    try g.infer(enc_out);

    try testing.expectEqual(@as(usize, 8), enc_out.ne[0]);
    try testing.expectEqual(@as(usize, 5), enc_out.ne[1]);
    for (enc_out.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "whisper - full encode+decode forward" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const model = try Whisper(f32, .{
        .n_mels = 8,
        .d_model = 8,
        .n_heads = 2,
        .d_ff = 16,
        .n_encoder_layers = 1,
        .n_decoder_layers = 1,
        .vocab_size = 32,
        .max_audio_ctx = 16,
        .max_decoder_len = 8,
    }).init(a);

    // Encoder
    const audio = try Tensor(f32).init(a, &.{ 8, 5 });
    nn.uniform(f32, audio, -0.1, 0.1, 42);
    const enc_out = model.encodeFeatures(audio);

    // Decoder
    const tokens = try Tensor(f32).initIndexVectorCopy(a, &.{ 1, 5, 10 });
    const logits = model.decode(enc_out, tokens);

    try g.infer(logits);

    try testing.expectEqual(@as(usize, 32), logits.ne[0]); // vocab_size
    try testing.expectEqual(@as(usize, 3), logits.ne[1]); // dec_seq
    for (logits.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "whisper - backward produces gradients" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const model = try Whisper(f32, .{
        .n_mels = 4,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_encoder_layers = 1,
        .n_decoder_layers = 1,
        .vocab_size = 8,
        .max_audio_ctx = 8,
        .max_decoder_len = 8,
    }).init(a);

    const audio = try Tensor(f32).init(a, &.{ 4, 3 });
    nn.uniform(f32, audio, -0.1, 0.1, 42);
    const enc_out = model.encodeFeatures(audio);

    const tokens = try Tensor(f32).initIndexVectorCopy(a, &.{ 1, 2 });
    const loss = model.decode(enc_out, tokens).sumAll();

    try g.run(loss);

    var has_grad = false;
    for (model.params()) |param| {
        if (param.grad) |grad| {
            for (grad.data) |v| {
                try testing.expect(!std.math.isNan(v));
                if (v != 0) has_grad = true;
            }
        }
    }
    try testing.expect(has_grad);
}
