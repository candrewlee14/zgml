//! LLaMA transformer block with Grouped Query Attention (GQA) and SwiGLU FFN.
//!
//! Architecture (LLaMA-style, pre-RMSNorm):
//!   1. x = x + GQA(rmsNorm(x))
//!   2. x = x + SwiGLU_FFN(rmsNorm(x))
//!
//! Key differences from a standard decoder-only transformer block:
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
const nn = @import("../nn.zig");

fn buildCausalMask(comptime T: type, alloc: Alloc, seq_len: usize) *Tensor(T) {
    const mask = Tensor(T).init(alloc, &.{ seq_len, seq_len }) catch unreachable;
    for (0..seq_len) |qi| {
        for (0..seq_len) |ki| {
            mask.data[qi * seq_len + ki] = if (ki <= qi) 0 else -1e9;
        }
    }
    return mask;
}

fn RoPE(comptime T: type, comptime d: usize, comptime max_seq_len: usize) type {
    if (d % 2 != 0)
        @compileError("RoPE: d must be even, got " ++ std.fmt.comptimePrint("{}", .{d}));

    return struct {
        const Self = @This();

        cos_table: *Tensor(T),
        sin_table: *Tensor(T),

        pub fn init(alloc: Alloc, base: T) !Self {
            var self: Self = undefined;
            self.cos_table = try Tensor(T).init(alloc, &.{ d, max_seq_len });
            self.sin_table = try Tensor(T).init(alloc, &.{ d, max_seq_len });

            for (0..max_seq_len) |pos| {
                for (0..d / 2) |i| {
                    const p: f32 = @floatFromInt(pos);
                    const dim: f32 = @floatFromInt(2 * i);
                    const dm: f32 = @floatFromInt(d);
                    const base_f32: f32 = @floatCast(base);
                    const freq = p / std.math.pow(f32, base_f32, dim / dm);
                    const cos_val: T = @floatCast(@cos(freq));
                    const sin_val: T = @floatCast(@sin(freq));
                    self.cos_table.data[pos * d + i] = cos_val;
                    self.cos_table.data[pos * d + i + d / 2] = cos_val;
                    self.sin_table.data[pos * d + i] = sin_val;
                    self.sin_table.data[pos * d + i + d / 2] = sin_val;
                }
            }

            return self;
        }

        fn packCosSin(alloc: Alloc, cos: *Tensor(T), sin: *Tensor(T)) *Tensor(T) {
            const seq = cos.ne[1];
            const cs_buf = Tensor(T).init(alloc, &.{ 2 * d, seq }) catch unreachable;
            for (0..seq) |col| {
                @memcpy(cs_buf.data[col * 2 * d ..][0..d], cos.data[col * d ..][0..d]);
                @memcpy(cs_buf.data[col * 2 * d + d ..][0..d], sin.data[col * d ..][0..d]);
            }
            return cs_buf;
        }

        pub fn getCosSinPacked(self: *const Self, alloc: Alloc, seq_len: usize) *Tensor(T) {
            std.debug.assert(seq_len <= max_seq_len);
            const cos = Tensor(T).init(alloc, &.{ d, seq_len }) catch unreachable;
            const sin = Tensor(T).init(alloc, &.{ d, seq_len }) catch unreachable;
            const elems = d * seq_len;
            @memcpy(cos.data[0..elems], self.cos_table.data[0..elems]);
            @memcpy(sin.data[0..elems], self.sin_table.data[0..elems]);
            return packCosSin(alloc, cos, sin);
        }

        pub fn getCosSinPackedAtPos(self: *const Self, alloc: Alloc, pos: usize) *Tensor(T) {
            return self.getCosSinPackedRange(alloc, pos, 1);
        }

        pub fn getCosSinPackedRange(self: *const Self, alloc: Alloc, start_pos: usize, seq_len: usize) *Tensor(T) {
            std.debug.assert(start_pos + seq_len <= max_seq_len);
            std.debug.assert(seq_len >= 1);
            const cs_buf = Tensor(T).init(alloc, &.{ 2 * d, seq_len }) catch unreachable;
            for (0..seq_len) |col| {
                @memcpy(cs_buf.data[col * 2 * d ..][0..d], self.cos_table.data[(start_pos + col) * d ..][0..d]);
                @memcpy(cs_buf.data[col * 2 * d + d ..][0..d], self.sin_table.data[(start_pos + col) * d ..][0..d]);
            }
            return cs_buf;
        }

        pub fn apply(_: *const Self, x: *Tensor(T), cos_sin: *Tensor(T)) *Tensor(T) {
            return x.ropeRotate(cos_sin);
        }

        pub fn forward(self: *const Self, x: *Tensor(T), seq_len: usize) *Tensor(T) {
            std.debug.assert(x.ne[0] == d);
            std.debug.assert(x.ne[1] == seq_len);
            const cs = self.getCosSinPacked(x.alloc.?, seq_len);
            return self.apply(x, cs);
        }
    };
}

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

    const RoPEType = RoPE(T, d_head, cfg.max_seq_len);

    return struct {
        const Self = @This();

        /// Identified graph nodes from a single `forwardCachedMasked` trace.
        /// The plan layer above uses these pointers directly — no graph
        /// walking, no shape-based node matching.
        pub const CachedLayerTrace = struct {
            output: *Tensor(T),
            k_write: [cfg.n_kv_heads]*Tensor(T),
            v_write: [cfg.n_kv_heads]*Tensor(T),
            attention: [cfg.n_heads]*Tensor(T),
        };

        w_q: *Tensor(T),
        w_k: *Tensor(T),
        w_v: *Tensor(T),
        w_o: *Tensor(T),
        w_gate: *Tensor(T),
        w_up: *Tensor(T),
        w_down: *Tensor(T),
        rms_norm_1: *Tensor(T),
        rms_norm_2: *Tensor(T),
        rope: RoPEType,

        pub fn init(alloc: Alloc) !Self {
            var self: Self = undefined;
            self.w_q = try Tensor(T).init(alloc, &.{ cfg.d_model, cfg.d_model });
            self.w_k = try Tensor(T).init(alloc, &.{ kv_dim, cfg.d_model });
            self.w_v = try Tensor(T).init(alloc, &.{ kv_dim, cfg.d_model });
            self.w_o = try Tensor(T).init(alloc, &.{ cfg.d_model, cfg.d_model });
            self.w_gate = try Tensor(T).init(alloc, &.{ cfg.d_ff, cfg.d_model });
            self.w_up = try Tensor(T).init(alloc, &.{ cfg.d_ff, cfg.d_model });
            self.w_down = try Tensor(T).init(alloc, &.{ cfg.d_model, cfg.d_ff });
            self.rms_norm_1 = try Tensor(T).init(alloc, &.{cfg.d_model});
            self.rms_norm_2 = try Tensor(T).init(alloc, &.{cfg.d_model});
            self.rope = try RoPEType.init(alloc, cfg.rope_base);

            var seed: u64 = 42;
            for ([_]*Tensor(T){
                self.w_q,    self.w_k,    self.w_v,
                self.w_o,    self.w_gate, self.w_up,
                self.w_down,
            }) |w| {
                nn.kaimingUniform(T, w, seed);
                seed +%= 1;
                w.setParam();
            }

            _ = self.rms_norm_1.setAllScalar(1);
            _ = self.rms_norm_2.setAllScalar(1);
            self.rms_norm_1.setParam();
            self.rms_norm_2.setParam();

            return self;
        }

        fn applyRmsNorm(self: *const Self, x: *Tensor(T), norm_reduce: []usize, comptime which: enum { norm1, norm2 }) *Tensor(T) {
            const bare = x.rmsNorm(norm_reduce, @floatCast(cfg.rms_norm_eps));
            const gamma = switch (which) {
                .norm1 => self.rms_norm_1,
                .norm2 => self.rms_norm_2,
            };
            return bare.mul(gamma.repeatLike(bare));
        }

        fn swigluFfn(self: *const Self, x: *Tensor(T)) *Tensor(T) {
            const gate = x.matMul(false, self.w_gate, false);
            const up = x.matMul(false, self.w_up, false);
            return nn.silu(T, gate).mul(up).matMul(false, self.w_down, false);
        }

        // ---------------------------------------------------------------
        // Full-sequence forward (training)
        // ---------------------------------------------------------------

        pub fn forward(self: *const Self, x: *Tensor(T)) *Tensor(T) {
            const alloc = x.alloc.?;
            const seq_len = x.ne[1];
            var norm_reduce = [_]usize{ 1, seq_len };

            const norm1 = self.applyRmsNorm(x, &norm_reduce, .norm1);

            const q_all = norm1.matMul(false, self.w_q, false);
            const k_all = norm1.matMul(false, self.w_k, false);
            const v_all = norm1.matMul(false, self.w_v, false);

            const mask = buildCausalMask(T, alloc, seq_len);
            const rope_cs = self.rope.getCosSinPacked(alloc, seq_len);

            // Pre-rotate K per KV head (avoid redundant rotation in GQA)
            var k_rotated: [cfg.n_kv_heads]*Tensor(T) = undefined;
            for (0..cfg.n_kv_heads) |kv_h| {
                const k_h = k_all.sliceRows(kv_h * d_head, (kv_h + 1) * d_head);
                k_rotated[kv_h] = self.rope.apply(k_h, rope_cs);
            }

            const dk: T = @floatFromInt(d_head);
            const attn_scale: T = 1.0 / @sqrt(dk);
            var attn_sum: ?*Tensor(T) = null;

            for (0..cfg.n_heads) |h| {
                const q_h = q_all.sliceRows(h * d_head, (h + 1) * d_head);
                const kv_h = h / n_rep;
                const v_h = v_all.sliceRows(kv_h * d_head, (kv_h + 1) * d_head);

                const q_rot = self.rope.apply(q_h, rope_cs);
                const attn_out = q_rot.attention(k_rotated[kv_h], v_h, mask, attn_scale);

                const w_o_h = self.w_o.sliceColumns(h * d_head, (h + 1) * d_head);
                const projected = attn_out.matMul(false, w_o_h, false);
                attn_sum = if (attn_sum) |acc| acc.add(projected) else projected;
            }

            const after_attn = x.add(attn_sum.?);

            const norm2 = self.applyRmsNorm(after_attn, &norm_reduce, .norm2);
            return after_attn.add(self.swigluFfn(norm2));
        }

        // ---------------------------------------------------------------
        // Frozen cached forward (for reusable inference plans)
        // ---------------------------------------------------------------

        /// Cached forward with a consolidated KV cache.
        ///
        /// `k_cache`/`v_cache`: single `[d_head, cache_seq_len * n_kv_heads]` tensor each.
        /// Head `kv_h`'s `[d_head, cache_seq_len]` slab is a contiguous column range.
        /// RoPE cos/sin are looked up at position `pos`.
        pub fn forwardCachedMasked(
            self: *const Self,
            x: *Tensor(T),
            k_cache: *Tensor(T),
            v_cache: *Tensor(T),
            pos: usize,
            attn_mask: *Tensor(T),
            rope_cs: *Tensor(T),
        ) CachedLayerTrace {
            const alloc = x.alloc.?;
            std.debug.assert(k_cache.n_dims >= 2 and v_cache.n_dims >= 2);
            std.debug.assert(k_cache.ne[0] == d_head and v_cache.ne[0] == d_head);
            std.debug.assert(k_cache.ne[1] == v_cache.ne[1]);
            std.debug.assert(k_cache.ne[1] % cfg.n_kv_heads == 0);
            const cache_seq_len = k_cache.ne[1] / cfg.n_kv_heads;
            std.debug.assert(cache_seq_len <= cfg.max_seq_len);
            std.debug.assert(pos + x.ne[1] <= cache_seq_len);
            std.debug.assert(attn_mask.ne[0] >= cache_seq_len);
            var norm_reduce = [_]usize{ 1, x.ne[1] };
            const norm1 = self.applyRmsNorm(x, &norm_reduce, .norm1);

            const q_proj = norm1.matMul(false, self.w_q, false);
            const k_proj = norm1.matMul(false, self.w_k, false);
            const v_proj = norm1.matMul(false, self.w_v, false);

            // Rotate and write into the per-head slab of the consolidated cache.
            var k_write: [cfg.n_kv_heads]*Tensor(T) = undefined;
            var v_write: [cfg.n_kv_heads]*Tensor(T) = undefined;
            for (0..cfg.n_kv_heads) |kv_h| {
                const k_h = k_proj.sliceRows(kv_h * d_head, (kv_h + 1) * d_head);
                const v_h = v_proj.sliceRows(kv_h * d_head, (kv_h + 1) * d_head);
                const k_slab = k_cache.sliceColumns(kv_h * cache_seq_len, (kv_h + 1) * cache_seq_len);
                const v_slab = v_cache.sliceColumns(kv_h * cache_seq_len, (kv_h + 1) * cache_seq_len);
                k_write[kv_h] = k_slab.sliceAssign(self.rope.apply(k_h, rope_cs), pos);
                v_write[kv_h] = v_slab.sliceAssign(v_h, pos);
            }

            const dk: T = @floatFromInt(d_head);
            const attn_scale: T = 1.0 / @sqrt(dk);
            // Build the concatenated attention output directly, then apply a
            // single output projection instead of one tiny matmul per head.
            // This scratch buffer is fully populated by the per-head stores below.
            var attn_buf = Tensor(T).init(alloc, &.{ cfg.d_model, x.ne[1] }) catch unreachable;

            var attention_nodes: [cfg.n_heads]*Tensor(T) = undefined;
            for (0..cfg.n_heads) |h| {
                const kv_h = h / n_rep;
                const q_h = q_proj.sliceRows(h * d_head, (h + 1) * d_head);
                const q_rot = self.rope.apply(q_h, rope_cs);
                const attn_out = q_rot.attention(k_write[kv_h], v_write[kv_h], attn_mask, attn_scale);
                attention_nodes[h] = attn_out;

                attn_buf = attn_buf.sliceAssignRows(attn_out, h * d_head);
            }

            const attn_projected = attn_buf.matMul(false, self.w_o, false);

            const after_attn = x.add(attn_projected);

            const norm2 = self.applyRmsNorm(after_attn, &norm_reduce, .norm2);
            const output = after_attn.add(self.swigluFfn(norm2));
            return .{
                .output = output,
                .k_write = k_write,
                .v_write = v_write,
                .attention = attention_nodes,
            };
        }

        // ---------------------------------------------------------------
        // Parameters
        // ---------------------------------------------------------------

        pub const n_block_params = 9;

        pub fn params(self: *const Self) [n_block_params]*Tensor(T) {
            return .{
                self.w_q,
                self.w_k,
                self.w_v,
                self.w_o,
                self.w_gate,
                self.w_up,
                self.w_down,
                self.rms_norm_1,
                self.rms_norm_2,
            };
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "llama causal mask is lower triangular" {
    const mask = buildCausalMask(f32, tac, 3);
    defer mask.deinit();

    try testing.expectEqualSlices(f32, &.{
        0, -1e9, -1e9,
        0, 0,    -1e9,
        0, 0,    0,
    }, mask.data);
}

test "llama rope - position zero is identity" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var rope = try RoPE(f32, 4, 8).init(a, 10000.0);
    const x = try Tensor(f32).init(a, &.{ 4, 1 });
    x.setData(&.{ 1, 2, 3, 4 });

    const y = rope.forward(x, 1);
    try g.infer(y);

    for (x.data, y.data) |xv, yv| {
        try testing.expectApproxEqAbs(xv, yv, 1e-5);
    }
}

test "llama rope - backward produces gradients" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    var rope = try RoPE(f32, 4, 8).init(a, 10000.0);
    const x = try Tensor(f32).init(a, &.{ 4, 2 });
    for (x.data, 0..) |*d, i| d.* = @as(f32, @floatFromInt(i + 1)) * 0.5;
    x.setParam();

    const loss = rope.forward(x, 2).sumAll();
    try g.run(loss);

    var has_nonzero = false;
    for (x.grad.?.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
        if (v != 0) has_nonzero = true;
    }
    try testing.expect(has_nonzero);
}

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

    const k_cache = try Tensor(f32).init(a, &.{ d_h, max_seq * n_kv });
    const v_cache = try Tensor(f32).init(a, &.{ d_h, max_seq * n_kv });
    @memset(k_cache.data, 0);
    @memset(v_cache.data, 0);

    const attn_mask = try Tensor(f32).init(a, &.{ max_seq, 1 });
    @memset(attn_mask.data, -std.math.inf(f32));
    attn_mask.data[0] = 0;

    const x = try Tensor(f32).init(a, &.{ d, 1 });
    nn.uniform(f32, x, -0.1, 0.1, 101);

    const rope_cs = block.rope.getCosSinPackedRange(a, 0, x.ne[1]);
    const trace = block.forwardCachedMasked(x, k_cache, v_cache, 0, attn_mask, rope_cs);
    try g.infer(trace.output);

    try testing.expectEqual(@as(usize, d), trace.output.ne[0]);
    try testing.expectEqual(@as(usize, 1), trace.output.ne[1]);
    for (trace.output.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "llama block - frozen cached masked forward derives KV slab stride from cache shape" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const d = 4;
    const max_seq = 8;
    const cache_seq = 3;
    const n_kv = 2;
    const d_h = d / n_kv;
    const block = try LlamaBlock(f32, .{
        .d_model = d,
        .n_heads = 2,
        .n_kv_heads = n_kv,
        .d_ff = 8,
        .max_seq_len = max_seq,
    }).init(a);

    const k_cache = try Tensor(f32).init(a, &.{ d_h, cache_seq * n_kv });
    const v_cache = try Tensor(f32).init(a, &.{ d_h, cache_seq * n_kv });
    @memset(k_cache.data, 0);
    @memset(v_cache.data, 0);

    const attn_mask = try Tensor(f32).init(a, &.{ cache_seq, 1 });
    @memset(attn_mask.data, -std.math.inf(f32));
    attn_mask.data[0] = 0;
    attn_mask.data[1] = 0;

    const x = try Tensor(f32).init(a, &.{ d, 1 });
    nn.uniform(f32, x, -0.1, 0.1, 202);

    const pos = 1;
    const rope_cs = block.rope.getCosSinPackedRange(a, pos, x.ne[1]);
    const trace = block.forwardCachedMasked(x, k_cache, v_cache, pos, attn_mask, rope_cs);

    try testing.expectEqual(@as(usize, cache_seq), trace.k_write[0].ne[1]);
    try testing.expectEqual(@as(usize, cache_seq), trace.k_write[1].ne[1]);
    try testing.expectEqual(@as(usize, pos), trace.k_write[1].storage_offset);

    const second_head_offset = (@intFromPtr(trace.k_write[1].src1.?.data.ptr) - @intFromPtr(k_cache.data.ptr)) / @sizeOf(f32);
    try testing.expectEqual(d_h * cache_seq, second_head_offset);

    try g.infer(trace.output);
    try testing.expectEqual(@as(usize, d), trace.output.ne[0]);
    try testing.expectEqual(@as(usize, 1), trace.output.ne[1]);
    for (trace.output.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}
