//! LLaMA transformer block with Grouped Query Attention (GQA) and SwiGLU FFN.
//!
//! Architecture (LLaMA-style, pre-RMSNorm):
//!   1. x = x + GQA(rmsNorm(x))
//!   2. x = x + SwiGLU_FFN(rmsNorm(x))
//!
//! Key differences from GPT-2 TransformerBlock:
//!   - RMSNorm instead of LayerNorm (no bias)
//!   - No biases on any projections
//!   - SwiGLU FFN: silu(x @ W_gate) * (x @ W_up) @ W_down
//!   - Grouped Query Attention: n_kv_heads <= n_heads
//!   - Separate Q/K/V projections (required for GQA)
//!   - RoPE applied to Q and K per-head
//!
//! ```
//! const block = try LlamaBlock(f32, .{
//!     .d_model = 4096, .n_heads = 32, .n_kv_heads = 8,
//!     .d_ff = 11008, .max_seq_len = 4096,
//! }).init(alloc);
//! const y = block.forward(x);
//! ```

const std = @import("std");
const testing = std.testing;
const tac = testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Alloc = std.mem.Allocator;
const shaped_mod = @import("../shaped.zig");
const Shaped = shaped_mod.Shaped;
const nn = @import("../nn.zig");

pub const LlamaBlockConfig = struct {
    d_model: usize,
    n_heads: usize,
    n_kv_heads: usize,
    d_ff: usize,
    max_seq_len: usize,
    rope_base: f32 = 10000.0,
    rms_norm_eps: f32 = 1e-6,
};

pub fn LlamaBlock(comptime T: type, comptime cfg: LlamaBlockConfig) type {
    if (cfg.d_model % cfg.n_heads != 0)
        @compileError("d_model must be divisible by n_heads");
    if (cfg.n_heads % cfg.n_kv_heads != 0)
        @compileError("n_heads must be divisible by n_kv_heads");

    const d_head = cfg.d_model / cfg.n_heads;
    const n_rep = cfg.n_heads / cfg.n_kv_heads;
    const kv_dim = cfg.n_kv_heads * d_head;

    const WqShape = Shaped(T, .{ cfg.d_model, cfg.d_model });
    const WkShape = Shaped(T, .{ kv_dim, cfg.d_model });
    const WvShape = Shaped(T, .{ kv_dim, cfg.d_model });
    const WoShape = Shaped(T, .{ cfg.d_model, cfg.d_model });
    const WGateShape = Shaped(T, .{ cfg.d_ff, cfg.d_model });
    const WUpShape = Shaped(T, .{ cfg.d_ff, cfg.d_model });
    const WDownShape = Shaped(T, .{ cfg.d_model, cfg.d_ff });
    const NormShape = Shaped(T, .{cfg.d_model});
    const RoPEType = nn.RoPE(T, d_head, cfg.max_seq_len);

    return struct {
        const Self = @This();

        w_q: WqShape,
        w_k: WkShape,
        w_v: WvShape,
        w_o: WoShape,
        w_gate: WGateShape,
        w_up: WUpShape,
        w_down: WDownShape,
        rms_norm_1: NormShape,
        rms_norm_2: NormShape,
        rope: RoPEType,

        pub fn init(alloc: Alloc) !Self {
            var self: Self = undefined;
            self.w_q = try WqShape.init(alloc);
            self.w_k = try WkShape.init(alloc);
            self.w_v = try WvShape.init(alloc);
            self.w_o = try WoShape.init(alloc);
            self.w_gate = try WGateShape.init(alloc);
            self.w_up = try WUpShape.init(alloc);
            self.w_down = try WDownShape.init(alloc);
            self.rms_norm_1 = try NormShape.init(alloc);
            self.rms_norm_2 = try NormShape.init(alloc);
            self.rope = try RoPEType.init(alloc, cfg.rope_base);

            var seed: u64 = 42;
            for ([_]*Tensor(T){
                self.w_q.inner,    self.w_k.inner,     self.w_v.inner,
                self.w_o.inner,    self.w_gate.inner,   self.w_up.inner,
                self.w_down.inner,
            }) |w| {
                nn.kaimingUniform(T, w, seed);
                seed +%= 1;
                w.setParam();
            }

            _ = self.rms_norm_1.inner.setAllScalar(1);
            _ = self.rms_norm_2.inner.setAllScalar(1);
            self.rms_norm_1.inner.setParam();
            self.rms_norm_2.inner.setParam();

            return self;
        }

        fn applyRmsNorm(self: *const Self, x: *Tensor(T), norm_reduce: []usize, comptime which: enum { norm1, norm2 }) *Tensor(T) {
            const bare = x.rmsNorm(norm_reduce, @floatCast(cfg.rms_norm_eps));
            const gamma = switch (which) {
                .norm1 => self.rms_norm_1.inner,
                .norm2 => self.rms_norm_2.inner,
            };
            return bare.mul(gamma.repeatLike(bare));
        }

        fn swigluFfn(self: *const Self, x: *Tensor(T)) *Tensor(T) {
            const gate = x.matMul(false, self.w_gate.inner, false);
            const up = x.matMul(false, self.w_up.inner, false);
            return nn.silu(T, gate).mul(up).matMul(false, self.w_down.inner, false);
        }

        // ---------------------------------------------------------------
        // Full-sequence forward (training)
        // ---------------------------------------------------------------

        pub fn forward(self: *const Self, x: *Tensor(T)) *Tensor(T) {
            const alloc = x.alloc.?;
            const seq_len = x.ne[1];
            var norm_reduce = [_]usize{ 1, seq_len };

            const norm1 = self.applyRmsNorm(x, &norm_reduce, .norm1);

            const q_all = norm1.matMul(false, self.w_q.inner, false);
            const k_all = norm1.matMul(false, self.w_k.inner, false);
            const v_all = norm1.matMul(false, self.w_v.inner, false);

            const mask = nn.buildCausalMask(T, alloc, seq_len);
            const rope_cs = self.rope.getCosSinPacked(alloc, seq_len);

            // Pre-rotate K per KV head (avoid redundant rotation in GQA)
            var k_rotated: [cfg.n_kv_heads]*Tensor(T) = undefined;
            for (0..cfg.n_kv_heads) |kv_h| {
                const k_h = k_all.sliceRows(kv_h * d_head, (kv_h + 1) * d_head);
                k_rotated[kv_h] = self.rope.apply(k_h, rope_cs);
            }

            var sm_reduce = [_]usize{ 1, seq_len };
            var attn_sum: ?*Tensor(T) = null;

            for (0..cfg.n_heads) |h| {
                const q_h = q_all.sliceRows(h * d_head, (h + 1) * d_head);
                const kv_h = h / n_rep;
                const v_h = v_all.sliceRows(kv_h * d_head, (kv_h + 1) * d_head);

                const q_rot = self.rope.apply(q_h, rope_cs);

                const scores = q_rot.matMul(false, k_rotated[kv_h], true);
                const dk: T = @floatFromInt(d_head);
                const scaled = scores.scaleByVal(1.0 / @sqrt(dk)).add(mask);
                const weights = scaled.softmax(&sm_reduce);
                const attn_out = weights.matMul(false, v_h, false);

                const w_o_h = self.w_o.inner.sliceColumns(h * d_head, (h + 1) * d_head);
                const projected = attn_out.matMul(false, w_o_h, false);
                attn_sum = if (attn_sum) |acc| acc.add(projected) else projected;
            }

            const after_attn = x.add(attn_sum.?);

            const norm2 = self.applyRmsNorm(after_attn, &norm_reduce, .norm2);
            return after_attn.add(self.swigluFfn(norm2));
        }

        // ---------------------------------------------------------------
        // Frozen cached forward (for InferencePlan)
        // ---------------------------------------------------------------

        /// Cached forward with per-head KV caches.
        ///
        /// `k_caches`/`v_caches`: per-KV-head caches, each [d_head, max_seq_len].
        /// RoPE cos/sin are looked up at position `pos`.
        pub fn forwardCachedMasked(
            self: *const Self,
            x: *Tensor(T),
            k_caches: [cfg.n_kv_heads]*Tensor(T),
            v_caches: [cfg.n_kv_heads]*Tensor(T),
            pos: usize,
            attn_mask: *Tensor(T),
        ) *Tensor(T) {
            const alloc = x.alloc.?;
            var norm_reduce = [_]usize{ 1, 1 };
            const norm1 = self.applyRmsNorm(x, &norm_reduce, .norm1);

            const q_proj = norm1.matMul(false, self.w_q.inner, false);
            const k_proj = norm1.matMul(false, self.w_k.inner, false);
            const v_proj = norm1.matMul(false, self.w_v.inner, false);

            const rope_cs = self.rope.getCosSinPackedAtPos(alloc, pos);

            // Rotate and cache K/V per KV head
            var k_updated: [cfg.n_kv_heads]*Tensor(T) = undefined;
            var v_updated: [cfg.n_kv_heads]*Tensor(T) = undefined;
            for (0..cfg.n_kv_heads) |kv_h| {
                const k_h = k_proj.sliceRows(kv_h * d_head, (kv_h + 1) * d_head);
                const v_h = v_proj.sliceRows(kv_h * d_head, (kv_h + 1) * d_head);
                k_updated[kv_h] = k_caches[kv_h].sliceAssign(self.rope.apply(k_h, rope_cs), pos);
                v_updated[kv_h] = v_caches[kv_h].sliceAssign(v_h, pos);
            }

            var sm_reduce = [_]usize{ 1, 1 };
            var attn_sum: ?*Tensor(T) = null;

            for (0..cfg.n_heads) |h| {
                const kv_h = h / n_rep;
                const q_h = q_proj.sliceRows(h * d_head, (h + 1) * d_head);
                const q_rot = self.rope.apply(q_h, rope_cs);

                const scores = q_rot.matMul(false, k_updated[kv_h], true);
                const dk: T = @floatFromInt(d_head);
                const scaled = scores.scaleByVal(1.0 / @sqrt(dk));
                const masked = scaled.add(attn_mask);
                const weights = masked.softmax(&sm_reduce);
                const attn_out = weights.matMul(false, v_updated[kv_h], false);

                const w_o_h = self.w_o.inner.sliceColumns(h * d_head, (h + 1) * d_head);
                const projected = attn_out.matMul(false, w_o_h, false);
                attn_sum = if (attn_sum) |acc| acc.add(projected) else projected;
            }

            const attn_projected = attn_sum.?;

            const after_attn = x.add(attn_projected);

            const norm2 = self.applyRmsNorm(after_attn, &norm_reduce, .norm2);
            return after_attn.add(self.swigluFfn(norm2));
        }

        // ---------------------------------------------------------------
        // Parameters
        // ---------------------------------------------------------------

        pub const n_block_params = 9;

        pub fn params(self: *const Self) [n_block_params]*Tensor(T) {
            return .{
                self.w_q.inner,
                self.w_k.inner,
                self.w_v.inner,
                self.w_o.inner,
                self.w_gate.inner,
                self.w_up.inner,
                self.w_down.inner,
                self.rms_norm_1.inner,
                self.rms_norm_2.inner,
            };
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "llama block - forward produces valid output" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const block = try LlamaBlock(f32, .{
        .d_model = 8,
        .n_heads = 2,
        .n_kv_heads = 2,
        .d_ff = 16,
        .max_seq_len = 32,
    }).init(a);

    const x = try Tensor(f32).init(a, &.{ 8, 3 });
    nn.uniform(f32, x, -0.1, 0.1, 123);
    const out = block.forward(x);
    try g.infer(out);

    try testing.expectEqual(@as(usize, 8), out.ne[0]);
    try testing.expectEqual(@as(usize, 3), out.ne[1]);
    for (out.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "llama block - GQA forward (n_kv_heads < n_heads)" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const block = try LlamaBlock(f32, .{
        .d_model = 8,
        .n_heads = 4,
        .n_kv_heads = 2,
        .d_ff = 16,
        .max_seq_len = 32,
    }).init(a);

    const x = try Tensor(f32).init(a, &.{ 8, 4 });
    nn.uniform(f32, x, -0.1, 0.1, 456);
    const out = block.forward(x);
    try g.infer(out);

    try testing.expectEqual(@as(usize, 8), out.ne[0]);
    try testing.expectEqual(@as(usize, 4), out.ne[1]);
    for (out.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "llama block - backward produces gradients" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const block = try LlamaBlock(f32, .{
        .d_model = 4,
        .n_heads = 2,
        .n_kv_heads = 2,
        .d_ff = 8,
        .max_seq_len = 16,
    }).init(a);

    const x = try Tensor(f32).init(a, &.{ 4, 2 });
    nn.uniform(f32, x, -0.1, 0.1, 789);
    x.setParam();
    const loss = block.forward(x).sumAll();
    try g.run(loss);

    for (block.params()) |p| {
        if (p.grad) |grad| {
            for (grad.data) |v| {
                try testing.expect(!std.math.isNan(v));
            }
        }
    }
}

test "llama block - frozen cached masked forward" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const d = 4;
    const max_seq = 8;
    const n_kv = 2;
    const d_h = d / n_kv;
    const block = try LlamaBlock(f32, .{
        .d_model = d,
        .n_heads = 2,
        .n_kv_heads = n_kv,
        .d_ff = 8,
        .max_seq_len = max_seq,
    }).init(a);

    var k_caches: [n_kv]*Tensor(f32) = undefined;
    var v_caches: [n_kv]*Tensor(f32) = undefined;
    for (0..n_kv) |h| {
        k_caches[h] = try Tensor(f32).init(a, &.{ d_h, max_seq });
        v_caches[h] = try Tensor(f32).init(a, &.{ d_h, max_seq });
        @memset(k_caches[h].data, 0);
        @memset(v_caches[h].data, 0);
    }

    const attn_mask = try Tensor(f32).init(a, &.{ max_seq, 1 });
    @memset(attn_mask.data, -std.math.inf(f32));
    attn_mask.data[0] = 0;

    const x = try Tensor(f32).init(a, &.{ d, 1 });
    nn.uniform(f32, x, -0.1, 0.1, 101);

    const out = block.forwardCachedMasked(x, k_caches, v_caches, 0, attn_mask);
    try g.infer(out);

    try testing.expectEqual(@as(usize, d), out.ne[0]);
    try testing.expectEqual(@as(usize, 1), out.ne[1]);
    for (out.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}
