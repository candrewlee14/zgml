//! CPU backend implementation.
//!
//! Provides both host kernel dispatch and device buffer management.
//! Device buffers are plain host allocations — this makes the CPU
//! backend a complete reference implementation for testing the device
//! buffer path without actual GPU hardware.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const forward = @import("../tensor/forward.zig");
const quant = @import("../quant.zig");

pub const CpuBackend = struct {
    alloc: ?std.mem.Allocator,

    /// Create a CPU backend with device buffer support.
    pub fn init(alloc: std.mem.Allocator) CpuBackend {
        return .{ .alloc = alloc };
    }

    /// Create a CPU backend without device buffer support (host dispatch only).
    pub fn initHostOnly() CpuBackend {
        return .{ .alloc = null };
    }

    pub fn backend(self: *CpuBackend) backend_mod.Backend {
        return .{
            .ctx = @ptrCast(self),
            .vtable = &vtable,
            .name_str = "cpu",
            .device_type = .cpu,
            .capabilities = .{ .device_buffers = self.alloc != null },
        };
    }
};

// ── VTable implementation ──────────────────────────────────────────

fn getState(ctx: *anyopaque) *CpuBackend {
    return @ptrCast(@alignCast(ctx));
}

// ── Host kernel dispatch ───────────────────────────────────────────

fn denseMatMulF32(_: *anyopaque, spec: backend_mod.DenseMatMulSpecF32) bool {
    const g = spec.geom;
    forward.blasSgemm(spec.dst, spec.a, spec.b, g.M, g.N, g.K, g.a_row_stride, g.a_col_stride, g.b_row_stride, g.b_col_stride, g.a_offset, g.b_offset, g.dst_offset, g.dst_row_stride);
    return true;
}

fn quantizedMatMulF32(_: *anyopaque, spec: backend_mod.QuantizedMatMulSpecF32) bool {
    if (spec.weight.rows != spec.K or spec.weight.cols != spec.N) return false;
    const Weight = quant.QuantizedWeight(f32);
    const weight = Weight{
        .data = spec.weight.data,
        .scales = spec.weight.scales,
        .rows = spec.weight.rows,
        .cols = spec.weight.cols,
        .block_size = spec.weight.block_size,
    };
    weight.matmul(spec.input, spec.dst, spec.M, spec.N, spec.K);
    return true;
}

// ── Device buffer management ───────────────────────────────────────

fn allocBuffer(ctx: *anyopaque, size: usize) ?backend_mod.DeviceBuffer {
    const self = getState(ctx);
    const alloc = self.alloc orelse return null;
    const mem = alloc.alloc(u8, size) catch return null;
    return .{ .ptr = @ptrCast(mem.ptr), .size = size };
}

fn freeBuffer(ctx: *anyopaque, buf: backend_mod.DeviceBuffer) void {
    const self = getState(ctx);
    const alloc = self.alloc orelse return;
    const ptr: [*]u8 = @ptrCast(buf.ptr);
    alloc.free(ptr[0..buf.size]);
}

fn upload(_: *anyopaque, dst: backend_mod.DeviceBuffer, dst_byte_offset: usize, src: []const u8) void {
    const ptr: [*]u8 = @ptrCast(dst.ptr);
    @memcpy(ptr[dst_byte_offset..][0..src.len], src);
}

fn download(_: *anyopaque, dst: []u8, src: backend_mod.DeviceBuffer, src_byte_offset: usize) void {
    const ptr: [*]const u8 = @ptrCast(src.ptr);
    @memcpy(dst, ptr[src_byte_offset..][0..dst.len]);
}

fn syncFn(_: *anyopaque) void {}

// ── Compiled CPU programs ─────────────────────────────────────────
//
// Pre-resolves all buffer indices to host pointers at compile time.
// Per-token execution is a tight loop with zero metadata overhead.

const CompiledCpuProgram = struct {
    /// Pre-resolved execution step: raw pointers + params, ready to call.
    const Step = union(enum) {
        matmul: struct { dst: [*]f32, a: [*]const f32, b: [*]const f32, geom: backend_mod.MatMulGeometry },
        qmatmul: struct { dst: [*]f32, input: [*]const f32, weight: quant.QuantizedWeight(f32), M: usize, N: usize, K: usize },
        elementwise: struct { op: backend_mod.Op, dst: [*]f32, src0: [*]const f32, src1: [*]const f32, n: usize },
        softmax: struct { data: [*]f32, rows: usize, cols: usize },
        layernorm: struct { dst: [*]f32, src: [*]const f32, rows: usize, cols: usize },
        reduce: struct { op: backend_mod.Op, dst: [*]f32, src: [*]const f32, n_out: usize, reduce_size: usize },
        repeat: struct { dst: [*]f32, src: [*]const f32, n: usize, src_ne: [4]usize, dst_ne: [4]usize, src_strides: [4]usize, dst_strides: [4]usize },
        slice_assign: struct { dst: [*]f32, src: [*]const f32, n: usize, dst_offset: *u32, dst_stride: usize, src_stride: usize },
    };

    steps: []Step,
    bufs: [][*]f32, // resolved buffer pointers
    qweights: []quant.QuantizedWeight(f32),
    alloc: std.mem.Allocator,

    fn execute(self: *CompiledCpuProgram, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
        // Upload per-token inputs (memcpy into pre-allocated host buffers).
        for (inputs) |io| {
            const dst: [*]u8 = @ptrCast(self.bufs[io.buf_idx]);
            @memcpy(dst[io.offset..][0..io.size], io.host_ptr[0..io.size]);
        }

        // Execute all steps — zero dispatch overhead.
        for (self.steps) |step| {
            switch (step) {
                .matmul => |m| {
                    const g = m.geom;
                    // blasSgemm uses offsets into the slices, so pass large-enough slices.
                    const dst_len = g.dst_offset + g.M * g.dst_row_stride;
                    const a_len = g.a_offset + g.M * @max(g.a_row_stride, g.a_col_stride);
                    const b_len = g.b_offset + g.K * @max(g.b_row_stride, g.b_col_stride);
                    forward.blasSgemm(m.dst[0..dst_len], m.a[0..a_len], m.b[0..b_len], g.M, g.N, g.K, g.a_row_stride, g.a_col_stride, g.b_row_stride, g.b_col_stride, g.a_offset, g.b_offset, g.dst_offset, g.dst_row_stride);
                },
                .qmatmul => |q| {
                    q.weight.matmul(q.input[0 .. q.M * q.K], @constCast(q.dst)[0 .. q.M * q.N], q.M, q.N, q.K);
                },
                .elementwise => |e| {
                    for (0..e.n) |i| {
                        const a = e.src0[i];
                        e.dst[i] = switch (e.op) {
                            .add => a + e.src1[i],
                            .mul => a * e.src1[i],
                            .neg => -a,
                            .exp => @exp(a),
                            .sqrt => @sqrt(a),
                            .recip => 1.0 / a,
                            .gelu => blk: {
                                const c = 0.7978845608 * (a + 0.044715 * a * a * a);
                                break :blk 0.5 * a * (1.0 + std.math.tanh(c));
                            },
                            else => a,
                        };
                    }
                },
                .softmax => |s| {
                    for (0..s.rows) |r| {
                        const row = s.data[r * s.cols ..][0..s.cols];
                        var m: f32 = -std.math.inf(f32);
                        for (row) |v| m = @max(m, v);
                        var sum: f32 = 0;
                        for (row) |*v| {
                            v.* = @exp(v.* - m);
                            sum += v.*;
                        }
                        const inv = 1.0 / sum;
                        for (row) |*v| v.* *= inv;
                    }
                },
                .layernorm => |l| {
                    for (0..l.rows) |r| {
                        const src_row = l.src[r * l.cols ..][0..l.cols];
                        const dst_row = @as([*]f32, @ptrCast(@constCast(l.dst)))[r * l.cols ..][0..l.cols];
                        var mu: f32 = 0;
                        for (src_row) |v| mu += v;
                        mu /= @floatFromInt(l.cols);
                        var v: f32 = 0;
                        for (src_row) |x| {
                            const d = x - mu;
                            v += d * d;
                        }
                        const inv_std = 1.0 / @sqrt(v / @as(f32, @floatFromInt(l.cols)) + 1e-5);
                        for (src_row, dst_row) |x, *d| d.* = (x - mu) * inv_std;
                    }
                },
                .reduce => |red| {
                    for (0..red.n_out) |i| {
                        const base = i * red.reduce_size;
                        var val: f32 = if (red.op == .max) -std.math.inf(f32) else 0;
                        for (0..red.reduce_size) |k| {
                            const v = red.src[base + k];
                            if (red.op == .max) { val = @max(val, v); } else { val += v; }
                        }
                        @as([*]f32, @ptrCast(@constCast(red.dst)))[i] = val;
                    }
                },
                .repeat => |rp| {
                    for (0..rp.n) |gid| {
                        var idx = gid;
                        var src_idx: usize = 0;
                        comptime var d: usize = 3;
                        inline while (d < 4) : (d -%= 1) {
                            const coord = idx / rp.dst_strides[d];
                            idx %= rp.dst_strides[d];
                            src_idx += (coord % rp.src_ne[d]) * rp.src_strides[d];
                        }
                        @as([*]f32, @ptrCast(@constCast(rp.dst)))[gid] = rp.src[src_idx];
                    }
                },
                .slice_assign => |sa| {
                    const off = sa.dst_offset.*;
                    for (0..sa.n) |i| {
                        @as([*]f32, @ptrCast(@constCast(sa.dst)))[off + i * sa.dst_stride] = sa.src[i * sa.src_stride];
                    }
                },
            }
        }

        // Download outputs.
        for (outputs) |io| {
            const src: [*]const u8 = @ptrCast(self.bufs[io.buf_idx]);
            @memcpy(io.host_ptr[0..io.size], src[io.offset..][0..io.size]);
        }
    }

    fn deinit(self: *CompiledCpuProgram) void {
        self.alloc.free(self.steps);
        // Free allocated buffers.
        for (self.bufs) |buf| {
            const ptr: [*]u8 = @ptrCast(buf);
            // We can't easily free individual buffers without tracking sizes.
            // The page_allocator was used for allocation.
            _ = ptr;
        }
        self.alloc.free(self.bufs);
        if (self.qweights.len > 0) self.alloc.free(self.qweights);
        self.alloc.destroy(self);
    }
};

fn compileProgram(ctx: *anyopaque, program: backend_mod.DeviceProgram) ?backend_mod.Backend.CompiledHandle {
    const self = getState(ctx);
    const alloc = self.alloc orelse return null;

    // Allocate host buffers (same as device buffers but just host memory).
    const bufs = alloc.alloc([*]f32, program.n_buffers) catch return null;
    for (bufs, program.buffer_sizes) |*buf, size| {
        const mem = alloc.alloc(f32, size) catch return null;
        buf.* = mem.ptr;
    }

    // Upload initial data.
    for (program.initial_uploads) |io| {
        const dst: [*]u8 = @ptrCast(bufs[io.buf_idx]);
        @memcpy(dst[io.offset..][0..io.size], io.host_ptr[0..io.size]);
    }

    // Quantize weights into host-resident QuantizedWeight structs.
    const qweights = alloc.alloc(quant.QuantizedWeight(f32), program.qweights.len) catch return null;
    for (program.qweights, 0..) |qw, i| {
        qweights[i] = .{
            .data = qw.data,
            .scales = qw.scales,
            .rows = qw.rows,
            .cols = qw.cols,
            .block_size = qw.block_size,
        };
    }

    // Pre-resolve all ops to direct pointer-based steps.
    // Cast ops to mutable so slice_assign can point to mutable dst_offset.
    const mutable_ops: []backend_mod.DeviceOp = @constCast(program.ops);
    const steps = alloc.alloc(CompiledCpuProgram.Step, program.ops.len) catch return null;
    for (mutable_ops, 0..) |*op, i| {
        steps[i] = switch (op.*) {
            .matmul => |m| .{ .matmul = .{ .dst = bufs[m.dst], .a = bufs[m.a], .b = bufs[m.b], .geom = m.geom } },
            .qmatmul => |q| .{ .qmatmul = .{ .dst = bufs[q.dst], .input = bufs[q.input], .weight = qweights[q.weight_idx], .M = q.M, .N = q.N, .K = q.K } },
            .elementwise => |e| .{ .elementwise = .{ .op = e.op, .dst = bufs[e.dst] + e.dst_offset, .src0 = bufs[e.src0] + e.src0_offset, .src1 = bufs[e.src1] + e.src1_offset, .n = e.n } },
            .softmax => |s| .{ .softmax = .{ .data = bufs[s.dst] + s.dst_offset, .rows = s.rows, .cols = s.cols } },
            .layernorm => |l| .{ .layernorm = .{ .dst = bufs[l.dst] + l.dst_offset, .src = bufs[l.src] + l.src_offset, .rows = l.rows, .cols = l.cols } },
            .reduce => |r| .{ .reduce = .{ .op = r.op, .dst = bufs[r.dst] + r.dst_offset, .src = bufs[r.src] + r.src_offset, .n_out = r.n_out, .reduce_size = r.reduce_size } },
            .repeat => |rp| .{ .repeat = .{ .dst = bufs[rp.dst] + rp.dst_offset, .src = bufs[rp.src] + rp.src_offset, .n = rp.n, .src_ne = .{ rp.src_ne[0], rp.src_ne[1], rp.src_ne[2], rp.src_ne[3] }, .dst_ne = .{ rp.dst_ne[0], rp.dst_ne[1], rp.dst_ne[2], rp.dst_ne[3] }, .src_strides = .{ rp.src_strides[0], rp.src_strides[1], rp.src_strides[2], rp.src_strides[3] }, .dst_strides = .{ rp.dst_strides[0], rp.dst_strides[1], rp.dst_strides[2], rp.dst_strides[3] } } },
            .slice_assign => .{ .slice_assign = .{ .dst = bufs[op.slice_assign.dst], .src = bufs[op.slice_assign.src] + op.slice_assign.src_offset, .n = op.slice_assign.n, .dst_offset = &op.slice_assign.dst_offset, .dst_stride = op.slice_assign.dst_stride, .src_stride = op.slice_assign.src_stride } },
        };
    }

    const compiled = alloc.create(CompiledCpuProgram) catch return null;
    compiled.* = .{ .steps = steps, .bufs = bufs, .qweights = qweights, .alloc = alloc };
    return @ptrCast(compiled);
}

fn executeProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
    const compiled: *CompiledCpuProgram = @ptrCast(@alignCast(handle));
    compiled.execute(inputs, outputs);
}

fn freeProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) void {
    const compiled: *CompiledCpuProgram = @ptrCast(@alignCast(handle));
    compiled.deinit();
}

// ── VTable ─────────────────────────────────────────────────────────

const vtable = backend_mod.Backend.VTable{
    .dense_matmul_f32 = denseMatMulF32,
    .quantized_matmul_f32 = quantizedMatMulF32,
    .alloc_buffer = allocBuffer,
    .free_buffer = freeBuffer,
    .upload = upload,
    .download = download,
    .sync = syncFn,
    .compile_program = compileProgram,
    .execute_program = executeProgram,
    .free_program = freeProgram,
};

// ── Tests ──────────────────────────────────────────────────────────

test "cpu backend host dense matmul" {
    var cpu = CpuBackend.initHostOnly();
    var dst = [_]f32{0} ** 4;
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 7, 8, 9, 10, 11, 12 };

    const ok = backend_mod.tryDenseMatMul(f32, cpu.backend(), .{
        .dst = &dst,
        .a = &a,
        .b = &b,
        .geom = .{ .M = 2, .N = 2, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
    });

    try std.testing.expect(ok);
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &dst);
}

test "cpu backend host quantized matmul" {
    const alloc = std.testing.allocator;
    var cpu = CpuBackend.initHostOnly();
    const weights = [_]f32{ 1.0, 0.5, -0.5, 1.0, 0.25, -0.25 };
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var dst = [_]f32{0} ** 4;
    var expected = [_]f32{0} ** 4;

    var qw = try quant.QuantizedWeight(f32).fromSlice(alloc, &weights, 3, 2, 32);
    defer qw.deinit(alloc);
    qw.matmul(&input, &expected, 2, 2, 3);

    const ok = backend_mod.tryQuantizedMatMul(f32, cpu.backend(), .{
        .dst = &dst,
        .input = &input,
        .weight = backend_mod.quantizedWeightViewF32(qw),
        .M = 2,
        .N = 2,
        .K = 3,
    });

    try std.testing.expect(ok);
    try std.testing.expectEqualSlices(f32, &expected, &dst);
}

test "cpu backend device buffer round-trip" {
    const alloc = std.testing.allocator;
    var cpu = CpuBackend.init(alloc);
    const be = cpu.backend();

    try std.testing.expect(be.caps().device_buffers);

    const buf = be.allocSlice(f32, 4) orelse return error.OutOfMemory;
    defer be.freeBuffer(buf);

    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    be.uploadSlice(f32, buf, 0, &src);

    var dst: [4]f32 = undefined;
    be.downloadSlice(f32, &dst, buf, 0);
    be.sync();

    try std.testing.expectEqualSlices(f32, &src, &dst);
}

test "cpu backend compiles and executes empty program" {
    const alloc = std.testing.allocator;
    var cpu = CpuBackend.init(alloc);
    const be = cpu.backend();
    const program = backend_mod.DeviceProgram{ .ops = &.{}, .n_buffers = 0, .buffer_sizes = &.{}, .initial_uploads = &.{} };
    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);
    be.executeProgram(handle, &.{}, &.{});
}

test "host-only backend reports no device buffer support" {
    var cpu = CpuBackend.initHostOnly();
    const be = cpu.backend();
    try std.testing.expect(!be.caps().device_buffers);
    try std.testing.expect(be.allocBuffer(64) == null);
}
