//! Shared f32 reference executor for backend DevicePrograms.
//!
//! This module owns the CPU semantics for backend_mod.DeviceOp. Backends that
//! have host-visible buffers can reuse it instead of carrying their own copy of
//! elementwise, norm, cache update, RoPE, matmul, qmatmul, and attention logic.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const forward = @import("../tensor/forward.zig");

pub const Buffer = struct {
    ptr: [*]f32,
    len: usize,
};

pub const QWeight = struct {
    data: []const i8,
    scales: []const f32,
    block_size: usize,
};

pub const OwnedBufferTable = struct {
    alloc: std.mem.Allocator,
    buffers: []Buffer,

    pub fn init(alloc: std.mem.Allocator, sizes: []const usize) !OwnedBufferTable {
        const buffers = try alloc.alloc(Buffer, sizes.len);
        var n_buffers: usize = 0;
        errdefer {
            for (buffers[0..n_buffers]) |buf| alloc.free(buf.ptr[0..buf.len]);
            alloc.free(buffers);
        }

        for (buffers, sizes) |*buf, size| {
            const storage = try alloc.alloc(f32, @max(size, 1));
            @memset(storage, 0);
            buf.* = .{ .ptr = storage.ptr, .len = storage.len };
            n_buffers += 1;
        }

        return .{ .alloc = alloc, .buffers = buffers };
    }

    pub fn deinit(self: *OwnedBufferTable) void {
        for (self.buffers) |buf| self.alloc.free(buf.ptr[0..buf.len]);
        self.alloc.free(self.buffers);
    }

    pub fn upload(self: OwnedBufferTable, inputs: []const backend_mod.ProgramIO) void {
        uploadToBuffers(self.buffers, inputs);
    }

    pub fn download(self: OwnedBufferTable, outputs: []const backend_mod.ProgramIO) void {
        downloadFromBuffers(self.buffers, outputs);
    }
};

pub fn uploadToBuffers(buffers: []const Buffer, inputs: []const backend_mod.ProgramIO) void {
    for (inputs) |io| {
        const bytes = bufferBytes(buffers[@as(usize, io.buf_idx)]);
        std.debug.assert(@as(usize, io.offset) + @as(usize, io.size) <= bytes.len);
        @memcpy(bytes[io.offset..][0..io.size], io.host_ptr[0..io.size]);
    }
}

pub fn downloadFromBuffers(buffers: []const Buffer, outputs: []const backend_mod.ProgramIO) void {
    for (outputs) |io| {
        const bytes = bufferConstBytes(buffers[@as(usize, io.buf_idx)]);
        std.debug.assert(@as(usize, io.offset) + @as(usize, io.size) <= bytes.len);
        @memcpy(io.host_ptr[0..io.size], bytes[io.offset..][0..io.size]);
    }
}

pub fn executeProgram(buffers: []const Buffer, qweights: []const QWeight, ops: []const backend_mod.DeviceOp) void {
    for (ops) |op| executeOp(buffers, qweights, op);
}

pub fn executeOp(buffers: []const Buffer, qweights: []const QWeight, op: backend_mod.DeviceOp) void {
    const ctx = Context{ .buffers = buffers, .qweights = qweights };
    ctx.executeOp(op);
}

fn bufferBytes(buffer: Buffer) []u8 {
    const ptr: [*]u8 = @ptrCast(buffer.ptr);
    return ptr[0 .. buffer.len * @sizeOf(f32)];
}

fn bufferConstBytes(buffer: Buffer) []const u8 {
    const ptr: [*]const u8 = @ptrCast(buffer.ptr);
    return ptr[0 .. buffer.len * @sizeOf(f32)];
}

const Context = struct {
    buffers: []const Buffer,
    qweights: []const QWeight,

    fn bufF32(self: Context, idx: u16) [*]f32 {
        return self.buffers[@as(usize, idx)].ptr;
    }

    fn bufSlice(self: Context, idx: u16) []f32 {
        const b = self.buffers[@as(usize, idx)];
        return b.ptr[0..b.len];
    }

    fn executeOp(self: Context, op: backend_mod.DeviceOp) void {
        switch (op) {
            .matmul => |m| self.matmul(m),
            .qmatmul => |q| self.qmatmul(q),
            .elementwise => |e| self.elementwise(e),
            .softmax => |s| self.softmax(s),
            .layernorm => |l| self.layernorm(l),
            .rmsnorm => |r| self.rmsnorm(r),
            .reduce => |rd| self.reduce(rd),
            .repeat => |rp| self.repeat(rp),
            .slice_assign => |sa| self.sliceAssign(sa),
            .rope => |rr| self.rope(rr),
            .attention => |att| self.attention(att),
            .fused_elementwise => |fe| self.fusedElementwise(fe),
        }
    }

    const V = 8;

    fn simdBinaryLoop(dst: [*]f32, src0: [*]const f32, src1: [*]const f32, n: usize, comptime op: fn (@Vector(V, f32), @Vector(V, f32)) @Vector(V, f32)) void {
        const VecT = @Vector(V, f32);
        var i: usize = 0;
        while (i + V <= n) : (i += V) {
            const a: VecT = src0[i..][0..V].*;
            const b: VecT = src1[i..][0..V].*;
            dst[i..][0..V].* = op(a, b);
        }
        while (i < n) : (i += 1) dst[i] = op(@as(VecT, @splat(src0[i])), @as(VecT, @splat(src1[i])))[0];
    }

    fn simdUnaryLoop(dst: [*]f32, src: [*]const f32, n: usize, comptime op: fn (@Vector(V, f32)) @Vector(V, f32)) void {
        const VecT = @Vector(V, f32);
        var i: usize = 0;
        while (i + V <= n) : (i += V) {
            const a: VecT = src[i..][0..V].*;
            dst[i..][0..V].* = op(a);
        }
        while (i < n) : (i += 1) dst[i] = op(@as(VecT, @splat(src[i])))[0];
    }

    fn elementwise(self: Context, e: anytype) void {
        const dst = self.bufF32(e.dst) + @as(usize, e.dst_offset);
        const src0 = self.bufF32(e.src0) + @as(usize, e.src0_offset);
        const src1 = self.bufF32(e.src1) + @as(usize, e.src1_offset);
        const n: usize = e.n;
        switch (e.op) {
            .add => simdBinaryLoop(dst, src0, src1, n, struct {
                fn f(a: @Vector(V, f32), b: @Vector(V, f32)) @Vector(V, f32) {
                    return a + b;
                }
            }.f),
            .mul => simdBinaryLoop(dst, src0, src1, n, struct {
                fn f(a: @Vector(V, f32), b: @Vector(V, f32)) @Vector(V, f32) {
                    return a * b;
                }
            }.f),
            .neg => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    return -a;
                }
            }.f),
            .abs => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    return @abs(a);
                }
            }.f),
            .relu => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    return @max(a, @as(@Vector(V, f32), @splat(0.0)));
                }
            }.f),
            .sqrt => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    return @sqrt(a);
                }
            }.f),
            .recip => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    return @as(@Vector(V, f32), @splat(@as(f32, 1.0))) / a;
                }
            }.f),
            .exp => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    return @exp(a);
                }
            }.f),
            .log => simdUnaryLoop(dst, src0, n, struct {
                fn f(a: @Vector(V, f32)) @Vector(V, f32) {
                    return @log(a);
                }
            }.f),
            .gelu => {
                const VecT = @Vector(V, f32);
                const k0: VecT = @splat(0.7978845608);
                const k1: VecT = @splat(0.044715);
                const half: VecT = @splat(0.5);
                const one: VecT = @splat(1.0);
                var i: usize = 0;
                while (i + V <= n) : (i += V) {
                    const a: VecT = src0[i..][0..V].*;
                    const k = k0 * (a + k1 * a * a * a);
                    const e2k = @exp(k + k);
                    dst[i..][0..V].* = half * a * (one + (e2k - one) / (e2k + one));
                }
                while (i < n) : (i += 1) {
                    const a = src0[i];
                    const kk = 0.7978845608 * (a + 0.044715 * a * a * a);
                    dst[i] = 0.5 * a * (1.0 + std.math.tanh(kk));
                }
            },
            else => @memcpy(dst[0..n], src0[0..n]),
        }
    }

    fn fusedElementwise(self: Context, fe: anytype) void {
        const dst = self.bufF32(fe.dst) + @as(usize, fe.dst_offset);
        const src = self.bufF32(fe.src) + @as(usize, fe.src_offset);
        const n: usize = fe.n;
        for (0..n) |i| {
            var v = src[i];
            for (fe.steps) |step| {
                switch (step.op) {
                    .neg => v = -v,
                    .abs => v = @abs(v),
                    .relu => v = @max(v, 0.0),
                    .sqrt => v = @sqrt(v),
                    .recip => v = 1.0 / v,
                    .exp => v = @exp(v),
                    .log => v = @log(v),
                    .gelu => {
                        const kk = 0.7978845608 * (v + 0.044715 * v * v * v);
                        v = 0.5 * v * (1.0 + std.math.tanh(kk));
                    },
                    .add => {
                        const s_ptr = self.bufF32(step.secondary_buf) + @as(usize, step.secondary_offset);
                        v = if (step.is_swapped) s_ptr[i] + v else v + s_ptr[i];
                    },
                    .mul => {
                        const s_ptr = self.bufF32(step.secondary_buf) + @as(usize, step.secondary_offset);
                        v = if (step.is_swapped) s_ptr[i] * v else v * s_ptr[i];
                    },
                    else => {},
                }
            }
            dst[i] = v;
        }
    }

    fn softmax(self: Context, s: anytype) void {
        const src = self.bufF32(s.src);
        const dst = self.bufF32(s.dst);
        const cols: usize = s.cols;
        for (0..@as(usize, s.rows)) |row| {
            const sb: usize = @as(usize, s.src_offset) + row * cols;
            const db: usize = @as(usize, s.dst_offset) + row * cols;
            var m: f32 = -std.math.inf(f32);
            for (0..cols) |j| m = @max(m, src[sb + j]);
            var sum: f32 = 0;
            for (0..cols) |j| {
                const v = @exp(src[sb + j] - m);
                dst[db + j] = v;
                sum += v;
            }
            const inv = if (sum > 0.0) 1.0 / sum else 0.0;
            for (0..cols) |j| dst[db + j] *= inv;
        }
    }

    fn layernorm(self: Context, l: anytype) void {
        const src = self.bufF32(l.src);
        const dst = self.bufF32(l.dst);
        const cols: usize = l.cols;
        for (0..@as(usize, l.rows)) |row| {
            const base: usize = @as(usize, l.src_offset) + row * cols;
            const dbase: usize = @as(usize, l.dst_offset) + row * cols;
            var mu: f32 = 0;
            for (0..cols) |j| mu += src[base + j];
            mu /= @as(f32, @floatFromInt(cols));
            var v: f32 = 0;
            for (0..cols) |j| {
                const diff = src[base + j] - mu;
                v += diff * diff;
            }
            const inv_std = 1.0 / @sqrt(v / @as(f32, @floatFromInt(cols)) + l.eps);
            for (0..cols) |j| dst[dbase + j] = (src[base + j] - mu) * inv_std;
        }
    }

    fn rmsnorm(self: Context, r: anytype) void {
        const src = self.bufF32(r.src);
        const dst = self.bufF32(r.dst);
        const cols: usize = r.cols;
        const VecT = @Vector(V, f32);
        for (0..@as(usize, r.rows)) |row| {
            const s = src + @as(usize, r.src_offset) + row * cols;
            const d = dst + @as(usize, r.dst_offset) + row * cols;
            var acc: VecT = @splat(0);
            var i: usize = 0;
            while (i + V <= cols) : (i += V) {
                const v: VecT = s[i..][0..V].*;
                acc += v * v;
            }
            var ss: f32 = @reduce(.Add, acc);
            while (i < cols) : (i += 1) ss += s[i] * s[i];
            const inv_rms: VecT = @splat(1.0 / @sqrt(ss / @as(f32, @floatFromInt(cols)) + r.eps));
            i = 0;
            while (i + V <= cols) : (i += V) {
                const v: VecT = s[i..][0..V].*;
                d[i..][0..V].* = v * inv_rms;
            }
            const inv_s = inv_rms[0];
            while (i < cols) : (i += 1) d[i] = s[i] * inv_s;
        }
    }

    fn reduce(self: Context, rd: anytype) void {
        const src = self.bufF32(rd.src);
        const dst = self.bufF32(rd.dst);
        const rs: usize = rd.reduce_size;
        for (0..@as(usize, rd.n_out)) |i| {
            const sb: usize = @as(usize, rd.src_offset) + i * rs;
            var val: f32 = if (rd.op == .max) -std.math.inf(f32) else 0.0;
            for (0..rs) |k| {
                const v = src[sb + k];
                val = if (rd.op == .max) @max(val, v) else val + v;
            }
            dst[@as(usize, rd.dst_offset) + i] = val;
        }
    }

    fn repeat(self: Context, rp: anytype) void {
        const src = self.bufF32(rp.src);
        const dst = self.bufF32(rp.dst);
        const n: usize = rp.n;
        const d = dst + @as(usize, rp.dst_offset);
        const s = src + @as(usize, rp.src_offset);

        const src_n: usize = @as(usize, rp.src_ne[0]) * @as(usize, rp.src_ne[1]) *
            @as(usize, rp.src_ne[2]) * @as(usize, rp.src_ne[3]);

        if (src_n == 1) {
            @memset(d[0..n], s[0]);
            return;
        }
        if (src_n >= n) {
            @memcpy(d[0..n], s[0..n]);
            return;
        }
        if (n % src_n == 0 and rp.src_strides[0] == 1 and
            (rp.src_ne[1] <= 1 or rp.src_strides[1] == rp.src_ne[0]) and
            (rp.src_ne[2] <= 1 or rp.src_strides[2] == @as(u32, rp.src_ne[0]) * rp.src_ne[1]) and
            (rp.src_ne[3] <= 1 or rp.src_strides[3] == @as(u32, rp.src_ne[0]) * @as(u32, rp.src_ne[1]) * rp.src_ne[2]))
        {
            var off: usize = 0;
            while (off + src_n <= n) : (off += src_n) {
                @memcpy(d[off..][0..src_n], s[0..src_n]);
            }
            return;
        }

        for (0..n) |gid| {
            var idx = gid;
            var src_idx: usize = rp.src_offset;
            var dim: usize = 4;
            while (dim > 0) {
                dim -= 1;
                const coord = idx / @as(usize, rp.dst_strides[dim]);
                idx = idx % @as(usize, rp.dst_strides[dim]);
                src_idx += (coord % @as(usize, rp.src_ne[dim])) * @as(usize, rp.src_strides[dim]);
            }
            dst[@as(usize, rp.dst_offset) + gid] = src[src_idx];
        }
    }

    fn sliceAssign(self: Context, sa: anytype) void {
        const src = self.bufF32(sa.src);
        const dst = self.bufF32(sa.dst);
        const rows: usize = sa.rows;
        const cols: usize = sa.cols;
        const doff: usize = sa.dst_offset;
        const soff: usize = sa.src_offset;
        const drs: usize = sa.dst_row_stride;
        const dcs: usize = sa.dst_col_stride;
        const srs: usize = sa.src_row_stride;
        const scs: usize = sa.src_col_stride;
        if (drs == 1 and srs == 1 and dcs == rows and scs == rows) {
            @memcpy(dst[doff..][0 .. rows * cols], src[soff..][0 .. rows * cols]);
        } else {
            for (0..cols) |col| {
                for (0..rows) |row| {
                    dst[doff + row * drs + col * dcs] = src[soff + row * srs + col * scs];
                }
            }
        }
    }

    fn rope(self: Context, rr: anytype) void {
        const src = self.bufF32(rr.src);
        const cs = self.bufF32(rr.cos_sin);
        const dst = self.bufF32(rr.dst);
        const hd: usize = rr.half_d;
        const s_off: usize = rr.src_off;
        const c_off: usize = rr.cs_off;
        const d_off: usize = rr.dst_off;
        const s_rs: usize = rr.src_rs;
        const s_cs: usize = rr.src_cs;
        const c_cs: usize = rr.cs_cs;
        for (0..@as(usize, rr.seq_len)) |col| {
            for (0..hd) |pair| {
                const x_lo = src[s_off + pair * s_rs + col * s_cs];
                const x_hi = src[s_off + (pair + hd) * s_rs + col * s_cs];
                const cos_v = cs[c_off + pair + col * c_cs];
                const sin_v = cs[c_off + pair + hd + col * c_cs];
                dst[d_off + pair + col * 2 * hd] = x_lo * cos_v - x_hi * sin_v;
                dst[d_off + pair + hd + col * 2 * hd] = x_hi * cos_v + x_lo * sin_v;
            }
        }
    }

    fn matmul(self: Context, m: anytype) void {
        forward.blasSgemm(
            self.bufSlice(m.dst),
            self.bufSlice(m.a),
            self.bufSlice(m.b),
            m.geom.M,
            m.geom.N,
            m.geom.K,
            m.geom.a_row_stride,
            m.geom.a_col_stride,
            m.geom.b_row_stride,
            m.geom.b_col_stride,
            m.geom.a_offset,
            m.geom.b_offset,
            m.geom.dst_offset,
            m.geom.dst_row_stride,
        );
    }

    fn qmatmul(self: Context, q: anytype) void {
        const input = self.bufF32(q.input);
        const dst_ptr = self.bufF32(q.dst);
        const w = self.qweights[@as(usize, q.weight_idx)];
        const M: usize = q.M;
        const N: usize = q.N;
        const K: usize = q.K;
        const bs: usize = w.block_size;

        for (0..M) |row| {
            for (0..N) |col| {
                var sum: f32 = 0;
                for (0..K) |k| {
                    const w_idx = k * N + col;
                    sum += input[row * K + k] *
                        @as(f32, @floatFromInt(w.data[w_idx])) *
                        w.scales[w_idx / bs];
                }
                dst_ptr[row * N + col] = sum;
            }
        }
    }

    fn attention(self: Context, att: anytype) void {
        const q_ptr = self.bufF32(att.q);
        const k_ptr = self.bufF32(att.k);
        const v_ptr = self.bufF32(att.v);
        const mask_ptr = self.bufF32(att.mask);
        const dst = self.bufF32(att.dst);
        const dh: usize = att.d_head;
        const sq: usize = att.seq_q;
        const skv: usize = att.seq_kv;
        const k_off: usize = att.k_off;
        const v_off: usize = att.v_off;
        const m_off: usize = att.mask_off;
        const qrs: usize = att.q_rs;
        const qcs: usize = att.q_cs;
        const krs: usize = att.k_rs;
        const kcs: usize = att.k_cs;
        const vrs: usize = att.v_rs;
        const vcs: usize = att.v_cs;
        const mrs: usize = att.mask_rs;
        const mcs: usize = att.mask_cs;
        const drs: usize = att.dst_rs;
        const dcs: usize = att.dst_cs;
        const VecT = @Vector(V, f32);
        const neg_inf = -std.math.inf(f32);

        std.debug.assert(dh <= 512);
        const unit_k = (krs == 1);
        const unit_v = (vrs == 1);
        const unit_q = (qrs == 1);
        const unit_dst = (drs == 1);

        for (0..sq) |qi| {
            const q_off: usize = @as(usize, att.q_off) + qi * qcs;
            const d_off: usize = @as(usize, att.dst_off) + qi * dcs;
            const mask_q_off = m_off + qi * mcs;

            var m_val: f32 = neg_inf;
            var l: f32 = 0;
            var acc_buf: [512]f32 = undefined;
            const acc = acc_buf[0..dh];
            @memset(acc, 0);

            for (0..skv) |s| {
                const mask_add = if (att.has_mask) mask_ptr[mask_q_off + s * mrs] else 0;
                if (!std.math.isFinite(mask_add)) continue;

                var dot: f32 = 0;
                if (unit_q and unit_k) {
                    var dot_v: VecT = @splat(0);
                    var r: usize = 0;
                    const kb = k_off + s * kcs;
                    while (r + V <= dh) : (r += V) {
                        const qv: VecT = q_ptr[q_off + r ..][0..V].*;
                        const kv: VecT = k_ptr[kb + r ..][0..V].*;
                        dot_v += qv * kv;
                    }
                    dot = @reduce(.Add, dot_v);
                    while (r < dh) : (r += 1) dot += q_ptr[q_off + r] * k_ptr[kb + r];
                } else {
                    for (0..dh) |r| dot += q_ptr[q_off + r * qrs] * k_ptr[k_off + r * krs + s * kcs];
                }

                const score = dot * att.scale + mask_add;
                if (!std.math.isFinite(score)) continue;

                const new_m = @max(m_val, score);
                const alpha = if (m_val == neg_inf) @as(f32, 0) else @exp(m_val - new_m);
                const w = @exp(score - new_m);
                l = l * alpha + w;
                m_val = new_m;

                if (unit_v) {
                    const alpha_v: VecT = @splat(alpha);
                    const w_v: VecT = @splat(w);
                    var r: usize = 0;
                    const vb = v_off + s * vcs;
                    while (r + V <= dh) : (r += V) {
                        const av: VecT = acc[r..][0..V].*;
                        const vv: VecT = v_ptr[vb + r ..][0..V].*;
                        acc[r..][0..V].* = av * alpha_v + w_v * vv;
                    }
                    while (r < dh) : (r += 1) {
                        acc[r] = acc[r] * alpha + w * v_ptr[vb + r];
                    }
                } else {
                    for (0..dh) |r| {
                        acc[r] = acc[r] * alpha + w * v_ptr[v_off + r * vrs + s * vcs];
                    }
                }
            }

            const inv_l = if (l > 0) 1.0 / l else @as(f32, 0);
            if (unit_dst) {
                const inv_v: VecT = @splat(inv_l);
                var r: usize = 0;
                while (r + V <= dh) : (r += V) {
                    const av: VecT = acc[r..][0..V].*;
                    dst[d_off + r ..][0..V].* = av * inv_v;
                }
                while (r < dh) : (r += 1) dst[d_off + r] = acc[r] * inv_l;
            } else {
                for (0..dh) |r| dst[d_off + r * drs] = acc[r] * inv_l;
            }
        }
    }
};

test "reference executor elementwise add" {
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 10, 20, 30, 40 };
    var dst = [_]f32{0} ** 4;
    const buffers = [_]Buffer{
        .{ .ptr = &a, .len = a.len },
        .{ .ptr = &b, .len = b.len },
        .{ .ptr = &dst, .len = dst.len },
    };

    executeOp(&buffers, &.{}, .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 0, .src1 = 1, .n = 4 } });

    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44 }, &dst);
}

test "reference executor matmul" {
    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var dst = [_]f32{0} ** 4;
    const buffers = [_]Buffer{
        .{ .ptr = &a, .len = a.len },
        .{ .ptr = &b, .len = b.len },
        .{ .ptr = &dst, .len = dst.len },
    };

    executeOp(&buffers, &.{}, .{ .matmul = .{
        .dst = 2,
        .a = 0,
        .b = 1,
        .geom = .{ .M = 2, .N = 2, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
    } });

    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &dst);
}

test "reference executor qmatmul uses row-major quantized weights" {
    var input = [_]f32{ 1, 2, 3, -1, 0.5, 4 };
    var dst = [_]f32{0} ** 6;
    const data = [_]i8{ 2, -1, 3, 4, -2, 1, -3, 5, 2 };
    const scales = [_]f32{ 0.5, 0.25, 1.0 };
    const qweights = [_]QWeight{.{ .data = &data, .scales = &scales, .block_size = 4 }};
    const buffers = [_]Buffer{
        .{ .ptr = &input, .len = input.len },
        .{ .ptr = &dst, .len = dst.len },
    };

    executeOp(&buffers, &qweights, .{ .qmatmul = .{
        .dst = 1,
        .input = 0,
        .weight_idx = 0,
        .M = 2,
        .N = 3,
        .K = 3,
    } });

    try std.testing.expectApproxEqAbs(@as(f32, 2.75), dst[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.25), dst[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), dst[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -3.0), dst[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.25), dst[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.625), dst[5], 1e-6);
}
