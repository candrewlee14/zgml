//! Pure LLM stage plans.
//!
//! This is the small semantic layer between a typed model and backend lowering.
//! It intentionally describes transformer execution in model-sized units rather
//! than exposing every graph op. Backends can lower a layer stage into one
//! command-buffer program, a handful of fused kernels, or a conservative op
//! schedule without changing the public inference API.

const std = @import("std");

const backend_mod = @import("../backend.zig");
const LlamaConfig = @import("../models/llama.zig").LlamaConfig;

pub const StageCapabilities = struct {
    fused_elementwise: bool = false,
    f16_matmul: bool = false,
    qmatmul: bool = false,
    prefill_attention: bool = false,
    decode_attention: bool = false,
    quantized_kv: bool = false,
    command_buffer_execution: bool = false,

    pub fn fromBackendCapabilities(caps: backend_mod.Capabilities) StageCapabilities {
        return .{
            .fused_elementwise = caps.fused_elementwise,
            .f16_matmul = caps.dense_matmul_f16 or caps.f16_weight_promotion,
            .qmatmul = caps.qmatmul,
            .prefill_attention = caps.prefill_attention or caps.attention.supported,
            .decode_attention = caps.decode_attention or caps.attention.supported,
            .quantized_kv = caps.quantized_kv,
            .command_buffer_execution = caps.command_buffer_execution,
        };
    }
};

pub const WeightRole = enum {
    token_embedding,
    attn_norm,
    w_q,
    w_k,
    w_v,
    w_o,
    ffn_norm,
    w_gate,
    w_up,
    w_down,
    final_norm,
    output,
};

pub const RuntimeBinding = enum {
    token_id,
    position,
    seq_kv,

    pub fn label(self: RuntimeBinding) []const u8 {
        return switch (self) {
            .token_id => "token_id",
            .position => "position",
            .seq_kv => "seq_kv",
        };
    }
};

pub const runtime_bindings = [_]RuntimeBinding{ .token_id, .position, .seq_kv };

pub const TransformerShape = struct {
    vocab_size: u32,
    d_model: u32,
    d_head: u32,
    n_heads: u32,
    n_kv_heads: u32,
    d_ff: u32,
    max_seq_len: u32,
    tied_lm_head: bool,

    pub fn kvDim(self: TransformerShape) u32 {
        return self.n_kv_heads * self.d_head;
    }

    pub fn qDim(self: TransformerShape) u32 {
        return self.n_heads * self.d_head;
    }
};

pub const LayerStep = enum {
    attn_norm,
    qkv_projection,
    rope_kv_write,
    decode_attention,
    attn_out_residual,
    ffn_norm,
    ffn_gate_up,
    swiglu_down_residual,

    pub fn label(self: LayerStep) []const u8 {
        return switch (self) {
            .attn_norm => "attn_norm",
            .qkv_projection => "qkv",
            .rope_kv_write => "rope+kv_write",
            .decode_attention => "decode_attention",
            .attn_out_residual => "attn_out+residual",
            .ffn_norm => "ffn_norm",
            .ffn_gate_up => "ffn_gate+up",
            .swiglu_down_residual => "swiglu+down+residual",
        };
    }
};

pub const layer_step_count = 8;
pub const layer_steps = [_]LayerStep{
    .attn_norm,
    .qkv_projection,
    .rope_kv_write,
    .decode_attention,
    .attn_out_residual,
    .ffn_norm,
    .ffn_gate_up,
    .swiglu_down_residual,
};

pub const LayerDecodePlan = struct {
    layer_index: u32,
    shape: TransformerShape,

    pub fn logicalStageCount(_: LayerDecodePlan) u32 {
        return layer_step_count;
    }

    pub fn targetCommandBuffers(_: LayerDecodePlan, caps: StageCapabilities) u32 {
        return if (caps.command_buffer_execution) 1 else layer_step_count;
    }

    pub fn usesGQA(self: LayerDecodePlan) bool {
        return self.shape.n_heads != self.shape.n_kv_heads;
    }
};

pub const FinalDecodePlan = struct {
    shape: TransformerShape,

    pub fn logicalStageCount(_: FinalDecodePlan) u32 {
        return 2;
    }

    pub fn targetCommandBuffers(_: FinalDecodePlan, caps: StageCapabilities) u32 {
        return if (caps.command_buffer_execution) 1 else 2;
    }
};

pub fn LlamaDecodePlan(comptime config: LlamaConfig) type {
    const shape = llamaShape(config);

    return struct {
        const Self = @This();

        shape: TransformerShape,
        layers: [config.n_layers]LayerDecodePlan,
        final: FinalDecodePlan,

        pub fn init() Self {
            var layers: [config.n_layers]LayerDecodePlan = undefined;
            for (0..config.n_layers) |i| {
                layers[i] = .{
                    .layer_index = u32From(i),
                    .shape = shape,
                };
            }
            return .{
                .shape = shape,
                .layers = layers,
                .final = .{ .shape = shape },
            };
        }

        pub fn logicalStageCount(self: *const Self) u32 {
            return 1 + u32From(self.layers.len) * layer_step_count + self.final.logicalStageCount();
        }

        pub fn targetCommandBuffers(self: *const Self, caps: StageCapabilities) u32 {
            var count: u32 = 0;
            for (self.layers) |layer| count += layer.targetCommandBuffers(caps);
            count += self.final.targetCommandBuffers(caps);
            return count;
        }

        pub fn writeSummary(self: *const Self, writer: anytype, caps: ?StageCapabilities) !void {
            try writer.print("LLM decode stage plan\n", .{});
            try writer.print(
                "  layers={d} d_model={d} heads={d} kv_heads={d} d_head={d} kv_dim={d} ff={d} max_seq={d} vocab={d}\n",
                .{
                    self.layers.len,
                    self.shape.d_model,
                    self.shape.n_heads,
                    self.shape.n_kv_heads,
                    self.shape.d_head,
                    self.shape.kvDim(),
                    self.shape.d_ff,
                    self.shape.max_seq_len,
                    self.shape.vocab_size,
                },
            );
            try writer.print("  tied_lm_head={}\n", .{self.shape.tied_lm_head});
            try writer.print("  runtime bindings: ", .{});
            for (runtime_bindings, 0..) |binding, i| {
                if (i != 0) try writer.writeAll(", ");
                try writer.writeAll(binding.label());
            }
            try writer.writeByte('\n');
            try writer.print("  layer lowering: ", .{});
            for (layer_steps, 0..) |step, i| {
                if (i != 0) try writer.writeAll(" -> ");
                try writer.writeAll(step.label());
            }
            try writer.writeByte('\n');
            try writer.print("  logical stages/token={d}\n", .{self.logicalStageCount()});
            if (caps) |c| {
                try writer.print(
                    "  target command buffers/token={d} (fused_ew={} f16_matmul={} qmatmul={} decode_attention={} quantized_kv={})\n",
                    .{
                        self.targetCommandBuffers(c),
                        c.fused_elementwise,
                        c.f16_matmul,
                        c.qmatmul,
                        c.decode_attention,
                        c.quantized_kv,
                    },
                );
            }
        }
    };
}

pub fn buildLlamaDecodePlan(comptime config: LlamaConfig) LlamaDecodePlan(config) {
    return LlamaDecodePlan(config).init();
}

pub fn printLlamaDecodePlanSummary(comptime config: LlamaConfig, writer: anytype, caps: ?StageCapabilities) !void {
    const plan = buildLlamaDecodePlan(config);
    try plan.writeSummary(writer, caps);
}

pub fn llamaShape(comptime config: LlamaConfig) TransformerShape {
    if (config.d_model % config.n_heads != 0)
        @compileError("d_model must be divisible by n_heads");
    if (config.n_heads % config.n_kv_heads != 0)
        @compileError("n_heads must be divisible by n_kv_heads");

    return .{
        .vocab_size = u32From(config.vocab_size),
        .d_model = u32From(config.d_model),
        .d_head = u32From(config.d_model / config.n_heads),
        .n_heads = u32From(config.n_heads),
        .n_kv_heads = u32From(config.n_kv_heads),
        .d_ff = u32From(config.d_ff),
        .max_seq_len = u32From(config.max_seq_len),
        .tied_lm_head = config.tied_lm_head,
    };
}

fn u32From(value: usize) u32 {
    std.debug.assert(value <= std.math.maxInt(u32));
    return @intCast(value);
}

const testing = std.testing;

test "llama decode plan captures SmolLM stage geometry" {
    const cfg = LlamaConfig{
        .vocab_size = 49152,
        .d_model = 576,
        .n_heads = 9,
        .n_kv_heads = 3,
        .d_ff = 1536,
        .n_layers = 30,
        .max_seq_len = 2048,
        .tied_lm_head = true,
    };
    const plan = buildLlamaDecodePlan(cfg);

    try testing.expectEqual(@as(usize, 30), plan.layers.len);
    try testing.expectEqual(@as(u32, 64), plan.shape.d_head);
    try testing.expectEqual(@as(u32, 192), plan.shape.kvDim());
    try testing.expectEqual(@as(u32, 243), plan.logicalStageCount());
    try testing.expectEqual(@as(u32, 31), plan.targetCommandBuffers(.{ .command_buffer_execution = true }));
    try testing.expectEqual(@as(u32, 242), plan.targetCommandBuffers(.{}));
    try testing.expect(plan.layers[0].usesGQA());
}

test "stage capabilities mirror backend capability metadata" {
    const metal_caps = StageCapabilities.fromBackendCapabilities(backend_mod.Capabilities.metal);

    try testing.expect(metal_caps.fused_elementwise);
    try testing.expect(metal_caps.f16_matmul);
    try testing.expect(metal_caps.qmatmul);
    try testing.expect(metal_caps.prefill_attention);
    try testing.expect(metal_caps.decode_attention);
    try testing.expect(metal_caps.command_buffer_execution);
}
