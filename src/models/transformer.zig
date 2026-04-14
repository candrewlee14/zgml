//! Multi-head transformer block with packed QKV projections.
//!
//! Architecture (GPT-2 style, pre-norm):
//!   1. x = x + multiHeadAttention(layerNorm(x))
//!   2. x = x + ffn(layerNorm(x))
//!
//! Where ffn(x) = gelu(x @ W1 + b1) @ W2 + b2
//!
//! Attention uses packed weight matrices:
//!   - `w_qkv [3*d_model, d_model]` — one GEMM for all Q/K/V projections
//!   - `w_o   [d_model, d_model]`   — per-head column views, summed output
//!
//! Per-head slices are derived via `sliceRows`/`sliceColumns` (zero-copy
//! strided views).  KV caches are packed per-layer as `[d_model, max_seq]`.
//!
//! ```
//! const block = try TransformerBlock(f32, 64, 4, 256, false, false, false).init(alloc);
//! const x = try Tensor(f32).init(alloc, &.{64, seq_len});
//! const y = block.forward(x);   // → *Tensor(f32), shape [64, seq_len]
//! ```

const std = @import("std");
const testing = std.testing;
const tac = testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Alloc = std.mem.Allocator;
const shaped_mod = @import("../shaped.zig");
const Shaped = shaped_mod.Shaped;
const ShapedTensor = shaped_mod.ShapedTensor;
const nn = @import("../nn.zig");

pub fn TransformerBlock(
    comptime T: type,
    comptime d_model: usize,
    comptime n_heads: usize,
    comptime d_ff: usize,
    comptime causal: bool,
    comptime learnable_ln: bool,
    comptime attn_bias: bool,
) type {
    if (d_model % n_heads != 0)
        @compileError("d_model (" ++ std.fmt.comptimePrint("{}", .{d_model}) ++
            ") must be divisible by n_heads (" ++ std.fmt.comptimePrint("{}", .{n_heads}) ++ ")");

    const d_head = d_model / n_heads;

    const QkvShape = Shaped(T, .{ 3 * d_model, d_model });
    const OShape = Shaped(T, .{ d_model, d_model });
    const QkvBiasShape = Shaped(T, .{ 3 * d_model });
    const OBiasShape = Shaped(T, .{d_model});
    const LnShape = Shaped(T, .{d_model});
    const W1Shape = Shaped(T, .{ d_ff, d_model });
    const B1Shape = Shaped(T, .{d_ff});
    const W2Shape = Shaped(T, .{ d_model, d_ff });
    const B2Shape = Shaped(T, .{d_model});

    return struct {
        const Self = @This();

        w_qkv: QkvShape,
        w_o: OShape,
        b_qkv: if (attn_bias) QkvBiasShape else void,
        b_o: if (attn_bias) OBiasShape else void,
        ln1_gamma: if (learnable_ln) LnShape else void,
        ln1_beta: if (learnable_ln) LnShape else void,
        ln2_gamma: if (learnable_ln) LnShape else void,
        ln2_beta: if (learnable_ln) LnShape else void,
        w1: W1Shape,
        b1: B1Shape,
        w2: W2Shape,
        b2: B2Shape,

        pub fn init(alloc: Alloc) !Self {
            var self: Self = undefined;
            self.w_qkv = try QkvShape.init(alloc);
            self.w_o = try OShape.init(alloc);
            self.w1 = try W1Shape.init(alloc);
            self.b1 = try B1Shape.init(alloc);
            self.w2 = try W2Shape.init(alloc);
            self.b2 = try B2Shape.init(alloc);

            var seed: u64 = 42;
            for ([_]*Tensor(T){ self.w_qkv.inner, self.w_o.inner, self.w1.inner, self.w2.inner }) |w| {
                nn.kaimingUniform(T, w, seed);
                seed +%= 1;
                w.setParam();
            }
            for ([_]*Tensor(T){ self.b1.inner, self.b2.inner }) |b| {
                _ = b.setAllScalar(0);
                b.setParam();
            }

            if (attn_bias) {
                self.b_qkv = try QkvBiasShape.init(alloc);
                _ = self.b_qkv.inner.setAllScalar(0);
                self.b_qkv.inner.setParam();
                self.b_o = try OBiasShape.init(alloc);
                _ = self.b_o.inner.setAllScalar(0);
                self.b_o.inner.setParam();
            }

            if (learnable_ln) {
                self.ln1_gamma = try LnShape.init(alloc);
                self.ln1_beta = try LnShape.init(alloc);
                self.ln2_gamma = try LnShape.init(alloc);
                self.ln2_beta = try LnShape.init(alloc);
                _ = self.ln1_gamma.inner.setAllScalar(1);
                _ = self.ln1_beta.inner.setAllScalar(0);
                _ = self.ln2_gamma.inner.setAllScalar(1);
                _ = self.ln2_beta.inner.setAllScalar(0);
                self.ln1_gamma.inner.setParam();
                self.ln1_beta.inner.setParam();
                self.ln2_gamma.inner.setParam();
                self.ln2_beta.inner.setParam();
            }

            return self;
        }

        fn isShaped(comptime Input: type) bool {
            return @typeInfo(Input) == .@"struct" and @hasDecl(Input, "static_shape");
        }

        fn ForwardResult(comptime Input: type) type {
            if (isShaped(Input)) {
                if (Input.element_type != T)
                    @compileError("TransformerBlock(" ++ @typeName(T) ++ "): input element type mismatch");
                if (Input.static_shape[0] != d_model)
                    @compileError("TransformerBlock: input dim 0 must be d_model");
                return Input;
            } else if (Input == *Tensor(T)) {
                return *Tensor(T);
            } else {
                @compileError("TransformerBlock.forward: expected Shaped or *Tensor");
            }
        }

        pub fn forward(self: *const Self, x: anytype) ForwardResult(@TypeOf(x)) {
            const inner = if (comptime isShaped(@TypeOf(x))) x.inner else x;
            const result = self.forwardInner(inner);
            return if (comptime isShaped(@TypeOf(x)))
                @TypeOf(x).fromTensor(result)
            else
                result;
        }

        fn applyLayerNorm(self: *const Self, x: *Tensor(T), ln_reduce: []usize, comptime which: enum { ln1, ln2 }) *Tensor(T) {
            const bare = x.layerNorm(ln_reduce, 1e-5);
            if (!learnable_ln) return bare;
            const gamma = switch (which) {
                .ln1 => self.ln1_gamma.inner,
                .ln2 => self.ln2_gamma.inner,
            };
            const beta = switch (which) {
                .ln1 => self.ln1_beta.inner,
                .ln2 => self.ln2_beta.inner,
            };
            return bare.mul(gamma.repeatLike(bare)).addBias(beta);
        }

        /// Packed QKV projection: one matmul → [3*d_model, seq/1].
        fn projectQkv(self: *const Self, norm: *Tensor(T)) *Tensor(T) {
            var qkv = norm.matMul(false, self.w_qkv.inner, false);
            if (attn_bias) qkv = qkv.addBias(self.b_qkv.inner);
            return qkv;
        }

        /// Per-head Q/K/V row slices from packed QKV [3*d_model, *].
        fn headSlice(qkv: *Tensor(T), h: usize, group: usize) *Tensor(T) {
            const off = group * d_model + h * d_head;
            return qkv.sliceRows(off, off + d_head);
        }

        /// Per-head output projection via column view of packed w_o.
        fn projectAndSum(self: *const Self, attn_sum: ?*Tensor(T), attn_out: *Tensor(T), h: usize) *Tensor(T) {
            const w_o_h = self.w_o.inner.sliceColumns(h * d_head, (h + 1) * d_head);
            const projected = attn_out.matMul(false, w_o_h, false);
            return if (attn_sum) |acc| acc.add(projected) else projected;
        }

        /// FFN: residual connection, pre-norm, two linear layers with gelu.
        fn ffn(self: *const Self, after_attn: *Tensor(T), ln_reduce: []usize) *Tensor(T) {
            const norm2 = self.applyLayerNorm(after_attn, ln_reduce, .ln2);
            const activated = nn.linear(T, norm2, self.w1.inner, self.b1.inner).gelu();
            const ff_out = nn.linear(T, activated, self.w2.inner, self.b2.inner);
            return after_attn.add(ff_out);
        }

        // ---------------------------------------------------------------
        // Full-sequence forward (training)
        // ---------------------------------------------------------------

        fn forwardInner(self: *const Self, x: *Tensor(T)) *Tensor(T) {
            const alloc = x.alloc.?;
            const seq_len = x.ne[1];
            var ln_reduce = [_]usize{ 1, seq_len };
            const norm1 = self.applyLayerNorm(x, &ln_reduce, .ln1);

            const mask: ?*Tensor(T) = if (causal) blk: {
                const m = Tensor(T).init(alloc, &.{ seq_len, seq_len }) catch unreachable;
                for (0..seq_len) |qi| {
                    for (0..seq_len) |ki| {
                        m.data[qi * seq_len + ki] = if (ki <= qi) 0 else -1e9;
                    }
                }
                break :blk m;
            } else null;

            const qkv = self.projectQkv(norm1);
            var sm_reduce = [_]usize{ 1, seq_len };

            var attn_sum: ?*Tensor(T) = null;
            for (0..n_heads) |h| {
                const q_h = headSlice(qkv, h, 0);
                const k_h = headSlice(qkv, h, 1);
                const v_h = headSlice(qkv, h, 2);

                const scores = q_h.matMul(false, k_h, true);
                const dk: T = @floatFromInt(d_head);
                var scaled = scores.scaleByVal(1.0 / @sqrt(dk));
                if (causal) scaled = scaled.add(mask.?);
                const weights = scaled.softmax(&sm_reduce);
                const attn_out = weights.matMul(false, v_h, false);
                attn_sum = self.projectAndSum(attn_sum, attn_out, h);
            }

            if (attn_bias) attn_sum = attn_sum.?.addBias(self.b_o.inner);
            const after_attn = x.add(attn_sum.?);
            return self.ffn(after_attn, &ln_reduce);
        }

        // ---------------------------------------------------------------
        // Cached single-token forward (inference)
        // ---------------------------------------------------------------

        /// Cached forward.  KV caches are packed `[d_model, max_seq]` per layer.
        /// Attention window grows with `pos` (variable shapes — not freezable).
        pub fn forwardCached(
            self: *const Self,
            x: *Tensor(T),
            k_cache: *Tensor(T),
            v_cache: *Tensor(T),
            pos: usize,
        ) *Tensor(T) {
            var ln_reduce = [_]usize{ 1, 1 };
            const norm1 = self.applyLayerNorm(x, &ln_reduce, .ln1);
            const qkv = self.projectQkv(norm1);

            const k_all = qkv.sliceRows(d_model, 2 * d_model);
            const v_all = qkv.sliceRows(2 * d_model, 3 * d_model);
            const k_updated = k_cache.sliceAssign(k_all, pos);
            const v_updated = v_cache.sliceAssign(v_all, pos);

            var attn_sum: ?*Tensor(T) = null;
            for (0..n_heads) |h| {
                const q_h = headSlice(qkv, h, 0);
                const ck = k_updated.sliceRows(h * d_head, (h + 1) * d_head).sliceColumns(0, pos + 1);
                const cv = v_updated.sliceRows(h * d_head, (h + 1) * d_head).sliceColumns(0, pos + 1);

                const scores = q_h.matMul(false, ck, true);
                const dk: T = @floatFromInt(d_head);
                const scaled = scores.scaleByVal(1.0 / @sqrt(dk));
                var sm_reduce = [_]usize{ 1, 1 };
                const weights = scaled.softmax(&sm_reduce);
                const attn_out = weights.matMul(false, cv, false);
                attn_sum = self.projectAndSum(attn_sum, attn_out, h);
            }

            if (attn_bias) attn_sum = attn_sum.?.addBias(self.b_o.inner);
            const after_attn = x.add(attn_sum.?);
            return self.ffn(after_attn, &ln_reduce);
        }

        /// Frozen cached forward — all shapes position-independent (freezable).
        /// `attn_mask`: [cache_cols, 1] — 0 for valid positions, -inf for masked.
        pub fn forwardCachedMasked(
            self: *const Self,
            x: *Tensor(T),
            k_cache: *Tensor(T),
            v_cache: *Tensor(T),
            pos: usize,
            attn_mask: *Tensor(T),
        ) *Tensor(T) {
            var ln_reduce = [_]usize{ 1, 1 };
            const norm1 = self.applyLayerNorm(x, &ln_reduce, .ln1);
            const qkv = self.projectQkv(norm1);

            const k_all = qkv.sliceRows(d_model, 2 * d_model);
            const v_all = qkv.sliceRows(2 * d_model, 3 * d_model);
            const k_updated = k_cache.sliceAssign(k_all, pos);
            const v_updated = v_cache.sliceAssign(v_all, pos);

            var attn_sum: ?*Tensor(T) = null;
            for (0..n_heads) |h| {
                const q_h = headSlice(qkv, h, 0);
                const ck = k_updated.sliceRows(h * d_head, (h + 1) * d_head);
                const cv = v_updated.sliceRows(h * d_head, (h + 1) * d_head);

                const scores = q_h.matMul(false, ck, true);
                const dk: T = @floatFromInt(d_head);
                const scaled = scores.scaleByVal(1.0 / @sqrt(dk));
                const masked = scaled.add(attn_mask);
                var sm_reduce = [_]usize{ 1, 1 };
                const weights = masked.softmax(&sm_reduce);
                const attn_out = weights.matMul(false, cv, false);
                attn_sum = self.projectAndSum(attn_sum, attn_out, h);
            }

            if (attn_bias) attn_sum = attn_sum.?.addBias(self.b_o.inner);
            const after_attn = x.add(attn_sum.?);
            return self.ffn(after_attn, &ln_reduce);
        }

        // ---------------------------------------------------------------
        // Parameters
        // ---------------------------------------------------------------

        pub const n_block_params = 2 + 4 +
            (if (learnable_ln) 4 else 0) +
            (if (attn_bias) 2 else 0);

        pub fn params(self: *const Self) [n_block_params]*Tensor(T) {
            var result: [n_block_params]*Tensor(T) = undefined;
            var idx: usize = 0;

            result[idx] = self.w_qkv.inner;
            idx += 1;
            result[idx] = self.w_o.inner;
            idx += 1;

            if (attn_bias) {
                result[idx] = self.b_qkv.inner;
                idx += 1;
                result[idx] = self.b_o.inner;
                idx += 1;
            }

            if (learnable_ln) {
                result[idx] = self.ln1_gamma.inner;
                idx += 1;
                result[idx] = self.ln1_beta.inner;
                idx += 1;
                result[idx] = self.ln2_gamma.inner;
                idx += 1;
                result[idx] = self.ln2_beta.inner;
                idx += 1;
            }

            result[idx] = self.w1.inner;
            idx += 1;
            result[idx] = self.b1.inner;
            idx += 1;
            result[idx] = self.w2.inner;
            idx += 1;
            result[idx] = self.b2.inner;
            idx += 1;

            std.debug.assert(idx == n_block_params);
            return result;
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "transformer block - single head forward" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();
    const block = try TransformerBlock(f32, 4, 1, 8, false, false, false).init(a);

    const x = try Tensor(f32).init(a, &.{ 4, 3 });
    const out = block.forward(x);
    try g.infer(out);
    try testing.expectEqual(@as(usize, 4), out.ne[0]);
    try testing.expectEqual(@as(usize, 3), out.ne[1]);
    for (out.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "transformer block - multi-head forward" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();
    const block = try TransformerBlock(f32, 8, 2, 16, false, false, false).init(a);

    const x = try Tensor(f32).init(a, &.{ 8, 4 });
    const out = block.forward(x);
    try g.infer(out);
    try testing.expectEqual(@as(usize, 8), out.ne[0]);
    try testing.expectEqual(@as(usize, 4), out.ne[1]);
    for (out.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "transformer block - causal masking" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();
    const block = try TransformerBlock(f32, 4, 1, 8, true, false, false).init(a);

    const x = try Tensor(f32).init(a, &.{ 4, 3 });
    const out = block.forward(x);
    try g.infer(out);
    for (out.data) |v| {
        try testing.expect(!std.math.isNan(v));
        try testing.expect(!std.math.isInf(v));
    }
}

test "transformer block - backward produces gradients" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();
    const block = try TransformerBlock(f32, 4, 2, 8, true, false, false).init(a);

    const x = try Tensor(f32).init(a, &.{ 4, 3 });
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

test "transformer block - packed KV cached forward" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const d = 4;
    const max_seq = 8;
    const block = try TransformerBlock(f32, d, 2, 8, true, false, false).init(a);

    const k_cache = try Tensor(f32).init(a, &.{ d, max_seq });
    const v_cache = try Tensor(f32).init(a, &.{ d, max_seq });
    @memset(k_cache.data, 0);
    @memset(v_cache.data, 0);

    for (0..3) |pos| {
        const x = try Tensor(f32).init(a, &.{ d, 1 });
        const out = block.forwardCached(x, k_cache, v_cache, pos);
        try g.infer(out);

        try testing.expectEqual(@as(usize, d), out.ne[0]);
        try testing.expectEqual(@as(usize, 1), out.ne[1]);
        for (out.data) |v| {
            try testing.expect(!std.math.isNan(v));
            try testing.expect(!std.math.isInf(v));
        }

        g.built_forward = false;
        g.nodes.clearRetainingCapacity();
        g.visited_nodes = .empty;
    }
}
