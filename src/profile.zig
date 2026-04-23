//! Profiling utilities for DeviceProgram analysis and timing breakdown.
//!
//! Usage:
//!   const profile = @import("profile.zig");
//!   const p = profile.profileProgram(program);
//!   profile.printProfile(p);
//!   profile.printTimingBreakdown("prefill", 128, elapsed_ns);

const std = @import("std");
const backend = @import("backend.zig");

pub const DeviceOp = backend.DeviceOp;
pub const DeviceProgram = backend.DeviceProgram;

/// Number of distinct DeviceOp tags.
const n_op_tags = 12;

/// Tag names in canonical order matching the DeviceOp union(enum) declaration.
const tag_names = [n_op_tags][]const u8{
    "elementwise",
    "matmul",
    "qmatmul",
    "softmax",
    "layernorm",
    "rmsnorm",
    "reduce",
    "repeat",
    "slice_assign",
    "rope",
    "attention",
    "fused_elementwise",
};

/// Indices of GPU-dispatched op types (matmul, qmatmul).
const gpu_tag_indices = [_]usize{ 1, 2 };

fn isGpuTag(idx: usize) bool {
    inline for (gpu_tag_indices) |gi| {
        if (idx == gi) return true;
    }
    return false;
}

/// Static profile summary of a compiled DeviceProgram.
pub const DeviceProgramProfile = struct {
    total_ops: u32,
    op_counts: [n_op_tags]u32,
    n_buffers: u16,
    total_buffer_bytes: usize,
    gpu_ops: u32,
    cpu_ops: u32,
};

/// Analyze a DeviceProgram and return a static profile summary.
pub fn profileProgram(program: DeviceProgram) DeviceProgramProfile {
    var counts = [_]u32{0} ** n_op_tags;

    for (program.ops) |op| {
        const idx: usize = @intFromEnum(op);
        counts[idx] += 1;
    }

    const total: u32 = @intCast(program.ops.len);

    var gpu: u32 = 0;
    inline for (gpu_tag_indices) |gi| {
        gpu += counts[gi];
    }

    var total_bytes: usize = 0;
    for (program.buffer_sizes) |sz| {
        total_bytes += sz * @sizeOf(f32);
    }

    return .{
        .total_ops = total,
        .op_counts = counts,
        .n_buffers = program.n_buffers,
        .total_buffer_bytes = total_bytes,
        .gpu_ops = gpu,
        .cpu_ops = total - gpu,
    };
}

/// Print a formatted profile summary to stderr.
pub fn printProfile(p: DeviceProgramProfile) void {
    // Build index array sorted by count descending.
    var order: [n_op_tags]usize = undefined;
    for (0..n_op_tags) |i| order[i] = i;

    // Insertion sort by count descending (n=15, trivial).
    for (1..n_op_tags) |i| {
        var j = i;
        while (j > 0 and p.op_counts[order[j]] > p.op_counts[order[j - 1]]) {
            const tmp = order[j];
            order[j] = order[j - 1];
            order[j - 1] = tmp;
            j -= 1;
        }
    }

    std.debug.print("\n=== Device Program Profile ===\n", .{});
    std.debug.print("Total ops: {d}\n", .{p.total_ops});

    const total_f: f64 = @floatFromInt(p.total_ops);

    for (order) |idx| {
        const count = p.op_counts[idx];
        if (count == 0) continue;
        const pct: f64 = if (p.total_ops > 0) @as(f64, @floatFromInt(count)) / total_f * 100.0 else 0.0;
        const label = if (isGpuTag(idx)) "[GPU]" else "[CPU]";
        std.debug.print("  {s:<22} {d:>5}  ({d:.1}%)  {s}\n", .{ tag_names[idx], count, pct, label });
    }

    const gpu_pct: f64 = if (p.total_ops > 0) @as(f64, @floatFromInt(p.gpu_ops)) / total_f * 100.0 else 0.0;
    const cpu_pct: f64 = if (p.total_ops > 0) @as(f64, @floatFromInt(p.cpu_ops)) / total_f * 100.0 else 0.0;

    std.debug.print("GPU dispatches: {d} ({d:.1}%)\n", .{ p.gpu_ops, gpu_pct });
    std.debug.print("CPU ops: {d} ({d:.1}%)\n", .{ p.cpu_ops, cpu_pct });

    const mb: f64 = @as(f64, @floatFromInt(p.total_buffer_bytes)) / (1024.0 * 1024.0);
    std.debug.print("Buffers: {d} ({d:.1} MB)\n\n", .{ p.n_buffers, mb });
}

/// Print a timing breakdown for a model inference run.
pub fn printTimingBreakdown(label: []const u8, n_tokens: u32, total_ns: u64) void {
    const total_ms: f64 = @as(f64, @floatFromInt(total_ns)) / 1_000_000.0;
    const tokens_f: f64 = @floatFromInt(n_tokens);
    const tok_s: f64 = if (total_ms > 0) tokens_f / (total_ms / 1000.0) else 0.0;
    const ms_per_tok: f64 = if (n_tokens > 0) total_ms / tokens_f else 0.0;

    std.debug.print("\n=== Timing: {s} ===\n", .{label});
    std.debug.print("{d} tokens in {d:.1} ms\n", .{ n_tokens, total_ms });
    std.debug.print("Throughput: {d:.1} tok/s ({d:.2} ms/tok)\n\n", .{ tok_s, ms_per_tok });
}

// ── Runtime profiling ──────────────────────────────────────────────

/// Accumulated per-op-type wall-clock time, populated by the backend
/// during CompiledProgram.execute(). Caller resets explicitly.
pub const RuntimeProfile = struct {
    time_ns: [n_op_tags]u64 = [_]u64{0} ** n_op_tags,
    call_count: u32 = 0,

    pub fn reset(self: *RuntimeProfile) void {
        self.* = .{};
    }
};

/// Estimate FLOPs for a single DeviceOp based on its geometry.
pub fn estimateFlops(op: DeviceOp) u64 {
    switch (op) {
        .matmul => |m| return 2 * @as(u64, m.geom.M) * @as(u64, m.geom.N) * @as(u64, m.geom.K),
        .qmatmul => |q| return 2 * @as(u64, q.M) * @as(u64, q.N) * @as(u64, q.K),
        .attention => |a| return @as(u64, a.seq_q) * (4 * @as(u64, a.seq_kv) * @as(u64, a.d_head) + 7 * @as(u64, a.seq_kv)),
        .rope => |r| return 6 * @as(u64, r.seq_len) * @as(u64, r.half_d),
        .softmax => |s| return 5 * @as(u64, s.rows) * @as(u64, s.cols),
        .layernorm => |l| return 5 * @as(u64, l.rows) * @as(u64, l.cols),
        .rmsnorm => |rn| return 3 * @as(u64, rn.rows) * @as(u64, rn.cols),
        .reduce => |rd| return @as(u64, rd.n_out) * @as(u64, rd.reduce_size),
        .elementwise => |e| return @as(u64, e.n),
        .fused_elementwise => |fe| return @as(u64, fe.n) * @as(u64, fe.steps.len),
        .repeat, .slice_assign => return 0,
    }
}

/// Estimate bytes transferred for a single DeviceOp based on its geometry.
pub fn estimateBytes(op: DeviceOp) u64 {
    switch (op) {
        .matmul => |m| {
            const M: u64 = m.geom.M;
            const N: u64 = m.geom.N;
            const K: u64 = m.geom.K;
            return (M * K + K * N + M * N) * 4;
        },
        .qmatmul => |q| {
            const M: u64 = q.M;
            const N: u64 = q.N;
            const K: u64 = q.K;
            // input f32 + weights i8 + output f32
            return M * K * 4 + K * N + M * N * 4;
        },
        .attention => |a| {
            const d: u64 = a.d_head;
            const s: u64 = a.seq_kv;
            const q: u64 = a.seq_q;
            // Q + K + V reads + scores + output
            return (q * d + 2 * q * s * d + q * s + q * d) * 4;
        },
        .rope => |r| {
            // read src(2*half_d) + cos_sin(2*half_d) + write dst(2*half_d)
            return 4 * @as(u64, r.seq_len) * @as(u64, r.half_d) * 4;
        },
        .softmax => |s| return 3 * @as(u64, s.rows) * @as(u64, s.cols) * 4,
        .layernorm => |l| return 3 * @as(u64, l.rows) * @as(u64, l.cols) * 4,
        .rmsnorm => |rn| return 2 * @as(u64, rn.rows) * @as(u64, rn.cols) * 4,
        .reduce => |rd| return (@as(u64, rd.n_out) * @as(u64, rd.reduce_size) + @as(u64, rd.n_out)) * 4,
        .elementwise => |e| {
            const n: u64 = e.n;
            return if (e.op.isBinary()) 3 * n * 4 else 2 * n * 4;
        },
        .fused_elementwise => |fe| return 2 * @as(u64, fe.n) * 4,
        .repeat => |rp| return @as(u64, rp.n) * 4,
        .slice_assign => |sa| return @as(u64, sa.rows) * @as(u64, sa.cols) * 4,
    }
}

/// Aggregated FLOP/byte estimates per op tag for a full program.
pub const ProgramEstimates = struct {
    flops: [n_op_tags]u64 = [_]u64{0} ** n_op_tags,
    bytes: [n_op_tags]u64 = [_]u64{0} ** n_op_tags,
};

/// Aggregate FLOP/byte estimates across all ops in a program, grouped by tag.
pub fn estimateProgram(ops: []const DeviceOp) ProgramEstimates {
    var est = ProgramEstimates{};
    for (ops) |op| {
        const idx: usize = @intFromEnum(op);
        est.flops[idx] += estimateFlops(op);
        est.bytes[idx] += estimateBytes(op);
    }
    return est;
}

/// Print a formatted runtime profile table with GFLOP/s and GB/s columns.
pub fn printRuntimeProfile(rt: RuntimeProfile, est: ProgramEstimates) void {
    if (rt.call_count == 0) return;

    // Sort by time descending.
    var order: [n_op_tags]usize = undefined;
    for (0..n_op_tags) |i| order[i] = i;
    for (1..n_op_tags) |i| {
        var j = i;
        while (j > 0 and rt.time_ns[order[j]] > rt.time_ns[order[j - 1]]) {
            const tmp = order[j];
            order[j] = order[j - 1];
            order[j - 1] = tmp;
            j -= 1;
        }
    }

    var total_ns: u64 = 0;
    for (rt.time_ns) |t| total_ns += t;
    const total_ms: f64 = @as(f64, @floatFromInt(total_ns)) / 1_000_000.0;
    const calls_f: f64 = @floatFromInt(rt.call_count);

    std.debug.print("\n=== Runtime Profile ({d} calls) ===\n", .{rt.call_count});
    std.debug.print("{s:<22} {s:>9} {s:>6}  {s:>10} {s:>9}  {s:>10} {s:>8}\n", .{ "op", "time_ms", "pct", "GFLOP", "GFLOP/s", "GB", "GB/s" });

    for (order) |idx| {
        const t = rt.time_ns[idx];
        if (t == 0) continue;
        const t_ms: f64 = @as(f64, @floatFromInt(t)) / 1_000_000.0;
        const pct: f64 = if (total_ns > 0) @as(f64, @floatFromInt(t)) / @as(f64, @floatFromInt(total_ns)) * 100.0 else 0.0;

        const total_flops: f64 = @as(f64, @floatFromInt(est.flops[idx])) * calls_f;
        const total_bytes: f64 = @as(f64, @floatFromInt(est.bytes[idx])) * calls_f;
        const gflop: f64 = total_flops / 1e9;
        const gb: f64 = total_bytes / 1e9;
        const t_s: f64 = @as(f64, @floatFromInt(t)) / 1e9;
        const gflop_s: f64 = if (t_s > 0) gflop / t_s else 0.0;
        const gb_s: f64 = if (t_s > 0) gb / t_s else 0.0;

        std.debug.print("{s:<22} {d:>9.2} {d:>5.1}%  {d:>10.3} {d:>9.1}  {d:>10.4} {d:>8.1}\n", .{ tag_names[idx], t_ms, pct, gflop, gflop_s, gb, gb_s });
    }
    std.debug.print("{s:<22} {d:>9.2}\n\n", .{ "TOTAL", total_ms });
}

// ── Tests ──────────────────────────────────────────────────────────

test "profileProgram counts ops correctly" {
    const ops = [_]DeviceOp{
        .{ .matmul = .{ .dst = 0, .a = 1, .b = 2, .geom = .{
            .M = 1, .N = 1, .K = 1,
            .a_row_stride = 1, .a_col_stride = 1,
            .b_row_stride = 1, .b_col_stride = 1,
            .a_offset = 0, .b_offset = 0,
            .dst_offset = 0, .dst_row_stride = 1,
        } } },
        .{ .matmul = .{ .dst = 0, .a = 1, .b = 2, .geom = .{
            .M = 1, .N = 1, .K = 1,
            .a_row_stride = 1, .a_col_stride = 1,
            .b_row_stride = 1, .b_col_stride = 1,
            .a_offset = 0, .b_offset = 0,
            .dst_offset = 0, .dst_row_stride = 1,
        } } },
        .{ .softmax = .{ .dst = 0, .src = 1, .rows = 1, .cols = 4 } },
    };
    const sizes = [_]usize{ 1024, 2048 };
    const program = DeviceProgram{
        .ops = &ops,
        .n_buffers = 3,
        .buffer_sizes = &sizes,
        .initial_uploads = &.{},
    };
    const p = profileProgram(program);

    try std.testing.expectEqual(@as(u32, 3), p.total_ops);
    try std.testing.expectEqual(@as(u32, 2), p.op_counts[1]); // matmul
    try std.testing.expectEqual(@as(u32, 1), p.op_counts[3]); // softmax
    try std.testing.expectEqual(@as(u32, 2), p.gpu_ops);
    try std.testing.expectEqual(@as(u32, 1), p.cpu_ops);
    try std.testing.expectEqual(@as(u16, 3), p.n_buffers);
    try std.testing.expectEqual(@as(usize, 3072 * @sizeOf(f32)), p.total_buffer_bytes);
}

test "profileProgram handles empty program" {
    const program = DeviceProgram{
        .ops = &.{},
        .n_buffers = 0,
        .buffer_sizes = &.{},
        .initial_uploads = &.{},
    };
    const p = profileProgram(program);

    try std.testing.expectEqual(@as(u32, 0), p.total_ops);
    try std.testing.expectEqual(@as(u32, 0), p.gpu_ops);
    try std.testing.expectEqual(@as(u32, 0), p.cpu_ops);
    try std.testing.expectEqual(@as(usize, 0), p.total_buffer_bytes);
}

test "estimateFlops matmul" {
    const op = DeviceOp{ .matmul = .{
        .dst = 0, .a = 1, .b = 2,
        .geom = .{ .M = 4, .N = 8, .K = 16, .a_row_stride = 16, .a_col_stride = 1, .b_row_stride = 8, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 8 },
    } };
    try std.testing.expectEqual(@as(u64, 2 * 4 * 8 * 16), estimateFlops(op));
    try std.testing.expectEqual(@as(u64, (4 * 16 + 16 * 8 + 4 * 8) * 4), estimateBytes(op));
}

test "estimateFlops elementwise binary vs unary" {
    const bin = DeviceOp{ .elementwise = .{ .op = .add, .dst = 0, .src0 = 1, .src1 = 2, .n = 100 } };
    const un = DeviceOp{ .elementwise = .{ .op = .neg, .dst = 0, .src0 = 1, .src1 = 0, .n = 100 } };
    try std.testing.expectEqual(@as(u64, 100), estimateFlops(bin));
    try std.testing.expectEqual(@as(u64, 100), estimateFlops(un));
    try std.testing.expectEqual(@as(u64, 3 * 100 * 4), estimateBytes(bin)); // binary: 3n*4
    try std.testing.expectEqual(@as(u64, 2 * 100 * 4), estimateBytes(un)); // unary: 2n*4
}

test "estimateProgram aggregates per tag" {
    const ops = [_]DeviceOp{
        .{ .matmul = .{ .dst = 0, .a = 1, .b = 2, .geom = .{ .M = 2, .N = 3, .K = 4, .a_row_stride = 4, .a_col_stride = 1, .b_row_stride = 3, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 3 } } },
        .{ .matmul = .{ .dst = 0, .a = 1, .b = 2, .geom = .{ .M = 2, .N = 3, .K = 4, .a_row_stride = 4, .a_col_stride = 1, .b_row_stride = 3, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 3 } } },
        .{ .softmax = .{ .dst = 0, .src = 1, .rows = 2, .cols = 5 } },
    };
    const est = estimateProgram(&ops);
    // matmul tag = 1: 2 * (2*2*3*4) = 96
    try std.testing.expectEqual(@as(u64, 2 * 2 * 3 * 4 * 2), est.flops[1]);
    // softmax tag = 3: 5*2*5 = 50
    try std.testing.expectEqual(@as(u64, 5 * 2 * 5), est.flops[3]);
}

test "RuntimeProfile reset" {
    var rt = RuntimeProfile{};
    rt.time_ns[0] = 42;
    rt.call_count = 5;
    rt.reset();
    try std.testing.expectEqual(@as(u64, 0), rt.time_ns[0]);
    try std.testing.expectEqual(@as(u32, 0), rt.call_count);
}
