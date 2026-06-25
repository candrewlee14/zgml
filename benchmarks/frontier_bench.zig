//! Decision-grade benchmark frontier for internal.
//!
//! Run with: zig build bench-frontier
//!
//! The harness adaptively batches each sample until the elapsed time is large
//! enough to avoid timer-resolution artifacts, then reports per-iteration
//! latency. Every sample checksums the output so optimized builds must keep the
//! computed values live.

const std = @import("std");
const opts = @import("zgml_options");
const internal = @import("zgml_internal");
const backend_mod = internal.backend;
const program_mod = internal.backend_program;
const profile_mod = internal.profile;

const Tensor = internal.Tensor;

const SampleCount = 15;
const WarmupSamples = 3;
const MinSampleNs: u64 = 8_000_000;
const MinRepeats: usize = 8;
const MaxRepeats: usize = 1 << 20;
const FrontierStencilVecLen = 16;

const FrontierFilter = struct {
    query: ?[]const u8,

    fn init() FrontierFilter {
        const raw = std.c.getenv("BENCH_FRONTIER_FILTER") orelse return .{ .query = null };
        const query = std.mem.span(raw);
        return .{ .query = if (query.len == 0) null else query };
    }

    fn enabled(self: FrontierFilter) bool {
        return self.query != null;
    }

    fn matches(self: FrontierFilter, name: []const u8) bool {
        const query = self.query orelse return true;
        return std.mem.indexOf(u8, name, query) != null;
    }

    fn matchesAny(self: FrontierFilter, names: []const []const u8) bool {
        if (!self.enabled()) return true;
        for (names) |name| {
            if (self.matches(name)) return true;
        }
        return false;
    }
};

const SemanticVariantFilter = struct {
    query: ?[]const u8,

    fn init() SemanticVariantFilter {
        const raw = std.c.getenv("BENCH_QSEMANTIC_VARIANTS") orelse return .{ .query = null };
        const query = std.mem.span(raw);
        return .{ .query = if (query.len == 0) null else query };
    }

    fn enabled(self: SemanticVariantFilter, name: []const u8) bool {
        const query = self.query orelse return true;
        var parts = std.mem.splitScalar(u8, query, ',');
        while (parts.next()) |part| {
            const trimmed = std.mem.trim(u8, part, " \t\r\n");
            if (trimmed.len == 0) continue;
            if (std.mem.eql(u8, trimmed, name)) return true;
        }
        return false;
    }
};

fn nowNs(io: std.Io) u64 {
    return @intCast(std.Io.Clock.awake.now(io).nanoseconds);
}

fn fillDeterministic(data: []f32, seed: u64, scale: f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (data) |*v| {
        v.* = (rng.float(f32) * 2.0 - 1.0) * scale;
    }
}

fn checksum(data: []const f32) f64 {
    var acc: f64 = 0;
    var finite_count: usize = 0;
    const stride = @max(@as(usize, 1), data.len / 4096);
    var i: usize = 0;
    while (i < data.len) : (i += stride) {
        const v = data[i];
        if (std.math.isFinite(v)) finite_count += 1;
        acc += @as(f64, @floatCast(v)) * @as(f64, @floatFromInt((i % 251) + 1));
    }
    if (finite_count == 0 or !std.math.isFinite(acc)) @panic("benchmark produced non-finite output");
    std.mem.doNotOptimizeAway(acc);
    return acc;
}

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) @panic("mismatched benchmark outputs");
    var max_diff: f32 = 0;
    for (a, b) |x, y| {
        const diff = @abs(x - y);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

const BenchStats = struct {
    repeats: usize,
    min_ns: f64,
    p50_ns: f64,
    p90_ns: f64,
    checksum: f64,
};

fn calibrateRepeats(io: std.Io, bench: anytype) usize {
    var repeats: usize = MinRepeats;
    while (true) {
        const t0 = nowNs(io);
        for (0..repeats) |_| bench.run();
        const elapsed = nowNs(io) - t0;
        _ = bench.consume();
        if (elapsed >= MinSampleNs or repeats >= MaxRepeats) return repeats;
        repeats = @min(repeats * 2, MaxRepeats);
    }
}

fn measure(io: std.Io, bench: anytype) BenchStats {
    const repeats = calibrateRepeats(io, bench);
    for (0..WarmupSamples) |_| {
        for (0..repeats) |_| bench.run();
        _ = bench.consume();
    }

    var times: [SampleCount]u64 = undefined;
    var total_check: f64 = 0;
    for (&times) |*t| {
        const t0 = nowNs(io);
        for (0..repeats) |_| bench.run();
        t.* = nowNs(io) - t0;
        total_check += bench.consume();
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));

    const denom = @as(f64, @floatFromInt(repeats));
    return .{
        .repeats = repeats,
        .min_ns = @as(f64, @floatFromInt(times[0])) / denom,
        .p50_ns = @as(f64, @floatFromInt(times[SampleCount / 2])) / denom,
        .p90_ns = @as(f64, @floatFromInt(times[(SampleCount * 9) / 10])) / denom,
        .checksum = total_check,
    };
}

fn printStats(
    w: *std.Io.Writer,
    name: []const u8,
    work_label: []const u8,
    work_units: f64,
    unit_suffix: []const u8,
    stats: BenchStats,
) !void {
    const seconds = stats.p50_ns / 1_000_000_000.0;
    const throughput = if (seconds > 0) work_units / seconds else 0;
    try w.print(
        "  {s:<28} p50={d:>10.1} ns  min={d:>10.1}  p90={d:>10.1}  reps={d:<7}  {s}={d:>9.2} {s}/s  check={d:.3}\n",
        .{ name, stats.p50_ns, stats.min_ns, stats.p90_ns, stats.repeats, work_label, throughput, unit_suffix, stats.checksum },
    );
}

fn printRatio(
    w: *std.Io.Writer,
    name: []const u8,
    numerator: BenchStats,
    denominator: BenchStats,
    diff: ?f32,
) !void {
    const ratio = numerator.p50_ns / denominator.p50_ns;
    if (diff) |max_diff| {
        try w.print("  {s:<28} speedup={d:.2}x  old_p50={d:.1} ns  fused_p50={d:.1} ns  max_abs_diff={d:.6}\n", .{
            name,
            ratio,
            numerator.p50_ns,
            denominator.p50_ns,
            max_diff,
        });
    } else {
        try w.print("  {s:<28} speedup={d:.2}x  old_p50={d:.1} ns  fused_p50={d:.1} ns\n", .{
            name,
            ratio,
            numerator.p50_ns,
            denominator.p50_ns,
        });
    }
}

fn printCommandShape(
    w: *std.Io.Writer,
    name: []const u8,
    ops: []const backend_mod.DeviceOp,
    commands: []const program_mod.ProgramCommand,
) !void {
    const shape = try program_mod.ProgramCommandStreamShape.fromOpsCommands(ops, commands);
    try writeCommandShapeMetricJson(w, name, shape);
    try w.print(
        "  {s:<28} shape_commands={d}  shape_semantic_ffn_sublayers={d}  shape_projection_row_chains={d}  shape_projection_row_chain_semantic_residual_bridges={d}  shape_covered_ops={d}  shape_saved_dispatches={d}  shape_projection_groups={d}\n",
        .{
            name,
            shape.command_count,
            shape.semantic_ffn_sublayers,
            shape.projection_row_chains,
            shape.projection_row_chain_semantic_residual_bridges,
            shape.covered_ops,
            shape.estimated_saved_dispatches,
            shape.projection_groups,
        },
    );
}

fn writeMetricJsonPrefix(w: *std.Io.Writer, label: []const u8, row_kind: []const u8) !std.json.Stringify {
    try w.writeAll("ZGML_FRONTIER_METRIC_JSON ");
    var jw: std.json.Stringify = .{ .writer = w };
    try jw.beginObject();
    try jw.objectField("label");
    try jw.write(label);
    try jw.objectField("row_kind");
    try jw.write(row_kind);
    return jw;
}

fn writeMetricJsonField(jw: *std.json.Stringify, key: []const u8, value: anytype) !void {
    try jw.objectField(key);
    try jw.write(value);
}

fn finishMetricJson(w: *std.Io.Writer, jw: *std.json.Stringify) !void {
    try jw.endObject();
    try w.writeByte('\n');
}

fn writeCommandShapeMetricJson(
    w: *std.Io.Writer,
    name: []const u8,
    shape: program_mod.ProgramCommandStreamShape,
) !void {
    var jw = try writeMetricJsonPrefix(w, name, "command_shape");
    try writeMetricJsonField(&jw, "shape_commands", shape.command_count);
    try writeMetricJsonField(&jw, "shape_semantic_ffn_sublayers", shape.semantic_ffn_sublayers);
    try writeMetricJsonField(&jw, "shape_projection_row_chains", shape.projection_row_chains);
    try writeMetricJsonField(&jw, "shape_projection_row_chain_semantic_residual_bridges", shape.projection_row_chain_semantic_residual_bridges);
    try writeMetricJsonField(&jw, "shape_covered_ops", shape.covered_ops);
    try writeMetricJsonField(&jw, "shape_saved_dispatches", shape.estimated_saved_dispatches);
    try writeMetricJsonField(&jw, "shape_projection_groups", shape.projection_groups);
    try finishMetricJson(w, &jw);
}

fn printProjectionRowChainRuntimeProfile(
    w: *std.Io.Writer,
    name: []const u8,
    be: backend_mod.Backend,
    handle: backend_mod.Backend.CompiledHandle,
    output_io: []const backend_mod.ProgramIO,
) !void {
    be.resetRuntimeProfile(handle);
    be.executeProgram(handle, &.{}, output_io);
    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeProfileTo(handle, &rt);
    try writeProjectionRowChainRuntimeMetricJson(w, name, rt);
    try w.print(
        "  {s:<28} runtime_command_dispatches={d}  qmatmul_row_chain_tiled_two_phase_count={d}  qmatmul_row_chain_tiled_finalize_tile_groups={d}  qmatmul_row_chain_tiled_finalize_elements={d}  qmatmul_row_chain_tiled_spilled_elementwise={d}  qmatmul_row_chain_tiled_spilled_input={d}  qmatmul_row_chain_tiled_output_spills={d}\n",
        .{
            name,
            rt.backend_dispatch_count,
            rt.qmatmul_row_chain_tiled_two_phase_count,
            rt.qmatmul_row_chain_tiled_finalize_tile_groups,
            rt.qmatmul_row_chain_tiled_finalize_elements,
            rt.qmatmul_row_chain_tiled_spilled_elementwise,
            rt.qmatmul_row_chain_tiled_spilled_input,
            rt.qmatmul_row_chain_tiled_output_spills,
        },
    );
}

fn writeProjectionRowChainRuntimeMetricJson(w: *std.Io.Writer, name: []const u8, rt: profile_mod.RuntimeProfile) !void {
    var jw = try writeMetricJsonPrefix(w, name, "projection_row_chain_runtime");
    try writeMetricJsonField(&jw, "runtime_command_dispatches", rt.backend_dispatch_count);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_two_phase_count", rt.qmatmul_row_chain_tiled_two_phase_count);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_finalize_tile_groups", rt.qmatmul_row_chain_tiled_finalize_tile_groups);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_finalize_elements", rt.qmatmul_row_chain_tiled_finalize_elements);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_spilled_elementwise", rt.qmatmul_row_chain_tiled_spilled_elementwise);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_spilled_input", rt.qmatmul_row_chain_tiled_spilled_input);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_output_spills", rt.qmatmul_row_chain_tiled_output_spills);
    try finishMetricJson(w, &jw);
}

fn printProjectionGroupRuntimeProfile(
    w: *std.Io.Writer,
    name: []const u8,
    be: backend_mod.Backend,
    handle: backend_mod.Backend.CompiledHandle,
    output_io: []const backend_mod.ProgramIO,
) !void {
    be.resetRuntimeProfile(handle);
    be.executeProgram(handle, &.{}, output_io);
    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeProfileTo(handle, &rt);
    try writeProjectionGroupRuntimeMetricJson(w, name, rt);
    try w.print(
        "  {s:<28} runtime_backend_dispatches={d}  runtime_projection_group_dispatches={d}  runtime_projection_cache_group_dispatches={d}  runtime_projection_group_count={d}  runtime_projection_cache_group_count={d}\n",
        .{
            name,
            rt.backend_dispatch_count,
            rt.program_command_dispatch_counts[@intFromEnum(program_mod.ProgramCommandKind.projection_group)],
            rt.program_command_dispatch_counts[@intFromEnum(program_mod.ProgramCommandKind.projection_cache_group)],
            rt.program_command_counts[@intFromEnum(program_mod.ProgramCommandKind.projection_group)],
            rt.program_command_counts[@intFromEnum(program_mod.ProgramCommandKind.projection_cache_group)],
        },
    );
}

fn writeProjectionGroupRuntimeMetricJson(w: *std.Io.Writer, name: []const u8, rt: profile_mod.RuntimeProfile) !void {
    const projection_group_idx = @intFromEnum(program_mod.ProgramCommandKind.projection_group);
    const projection_cache_group_idx = @intFromEnum(program_mod.ProgramCommandKind.projection_cache_group);
    var jw = try writeMetricJsonPrefix(w, name, "projection_group_runtime");
    try writeMetricJsonField(&jw, "runtime_backend_dispatches", rt.backend_dispatch_count);
    try writeMetricJsonField(&jw, "runtime_projection_group_dispatches", rt.program_command_dispatch_counts[projection_group_idx]);
    try writeMetricJsonField(&jw, "runtime_projection_cache_group_dispatches", rt.program_command_dispatch_counts[projection_cache_group_idx]);
    try writeMetricJsonField(&jw, "runtime_projection_group_count", rt.program_command_counts[projection_group_idx]);
    try writeMetricJsonField(&jw, "runtime_projection_cache_group_count", rt.program_command_counts[projection_cache_group_idx]);
    try finishMetricJson(w, &jw);
}

fn printSemanticSublayerRuntimeProfile(
    w: *std.Io.Writer,
    name: []const u8,
    be: backend_mod.Backend,
    handle: backend_mod.Backend.CompiledHandle,
    output_io: []const backend_mod.ProgramIO,
) !void {
    be.resetRuntimeProfile(handle);
    be.executeProgram(handle, &.{}, output_io);
    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeProfileTo(handle, &rt);
    const projection_row_chain_idx = @intFromEnum(program_mod.ProgramCommandKind.projection_row_chain);
    const semantic_idx = @intFromEnum(program_mod.ProgramCommandKind.semantic_ffn_sublayer);
    const semantic_with_input_idx = @intFromEnum(program_mod.ProgramCommandKind.semantic_ffn_sublayer_with_input_row_chain);
    const semantic_tile_groups = rt.semantic_ffn_sublayer_tile_parallel_groups;
    const semantic_row_serial_per_tile_group = if (semantic_tile_groups > 0) rt.semantic_ffn_sublayer_row_serial_dot_ops / semantic_tile_groups else 0;
    const semantic_total_row_serial_per_tile_group = if (semantic_tile_groups > 0) rt.semantic_ffn_sublayer_total_row_serial_dot_ops / semantic_tile_groups else 0;
    const semantic_width_lane_utilization_x1000 = if (rt.semantic_ffn_sublayer_width_lane_slots > 0) rt.semantic_ffn_sublayer_active_width_lanes * 1000 / rt.semantic_ffn_sublayer_width_lane_slots else 0;
    const semantic_thread_lane_utilization_x1000 = if (rt.semantic_ffn_sublayer_thread_lane_slots > 0) rt.semantic_ffn_sublayer_active_thread_lanes * 1000 / rt.semantic_ffn_sublayer_thread_lane_slots else 0;
    try writeSemanticSublayerRuntimeMetricJson(w, name, rt, semantic_row_serial_per_tile_group, semantic_total_row_serial_per_tile_group);
    try w.print(
        "  {s:<28} runtime_backend_dispatches={d}  semantic_target_dispatches=1  runtime_projection_row_chain_dispatches={d}  runtime_projection_row_chain_attempts={d}  runtime_projection_row_chain_refused={d}  runtime_semantic_ffn_dispatches={d}  runtime_semantic_ffn_with_input_dispatches={d}  runtime_semantic_ffn_with_input_attempts={d}  runtime_semantic_ffn_with_input_refused={d}  semantic_ffn_sublayer_fallback_pair_dispatches={d}  semantic_ffn_sublayer_fallback_tail_dispatches={d}  qmatmul_row_chain_tiled_count={d}  qmatmul_row_chain_tiled_row_tile_groups={d}  qmatmul_row_chain_tiled_n_tiles={d}  qmatmul_row_chain_tiled_serial_tile_loops={d}  qmatmul_row_chain_tiled_partial_slots={d}  qmatmul_row_chain_tiled_scratch_capacity={d}  qmatmul_row_chain_tiled_two_phase_count={d}  qmatmul_row_chain_tiled_finalize_tile_groups={d}  qmatmul_row_chain_tiled_finalize_elements={d}  qmatmul_row_chain_tiled_spilled_elementwise={d}  qmatmul_row_chain_tiled_spilled_input={d}  qmatmul_row_chain_tiled_output_spills={d}\n",
        .{
            name,
            rt.backend_dispatch_count,
            rt.program_command_dispatch_counts[projection_row_chain_idx],
            rt.program_command_attempt_counts[projection_row_chain_idx],
            rt.program_command_failed_counts[projection_row_chain_idx],
            rt.program_command_dispatch_counts[semantic_idx],
            rt.program_command_dispatch_counts[semantic_with_input_idx],
            rt.program_command_attempt_counts[semantic_with_input_idx],
            rt.program_command_failed_counts[semantic_with_input_idx],
            rt.semantic_ffn_sublayer_fallback_pair_dispatches,
            rt.semantic_ffn_sublayer_fallback_tail_dispatches,
            rt.qmatmul_row_chain_tiled_count,
            rt.qmatmul_row_chain_tiled_row_tile_groups,
            rt.qmatmul_row_chain_tiled_n_tiles,
            rt.qmatmul_row_chain_tiled_serial_tile_loops,
            rt.qmatmul_row_chain_tiled_partial_slots,
            rt.qmatmul_row_chain_tiled_scratch_capacity,
            rt.qmatmul_row_chain_tiled_two_phase_count,
            rt.qmatmul_row_chain_tiled_finalize_tile_groups,
            rt.qmatmul_row_chain_tiled_finalize_elements,
            rt.qmatmul_row_chain_tiled_spilled_elementwise,
            rt.qmatmul_row_chain_tiled_spilled_input,
            rt.qmatmul_row_chain_tiled_output_spills,
        },
    );
    try w.print(
        "  {s:<28} semantic_ffn_sublayer_count={d}  semantic_ffn_sublayer_rows={d}  semantic_ffn_sublayer_hidden={d}  semantic_ffn_sublayer_input={d}  semantic_ffn_sublayer_output={d}  semantic_ffn_sublayer_row_serial_dot_ops={d}  semantic_ffn_sublayer_total_row_serial_dot_ops={d}  semantic_ffn_sublayer_tile_row_groups={d}  semantic_ffn_sublayer_tile_hidden_tiles={d}  semantic_ffn_sublayer_tile_output_tiles={d}  semantic_ffn_sublayer_tile_parallel_groups={d}  semantic_ffn_sublayer_row_serial_dot_ops_per_tile_parallel_group={d}  semantic_ffn_sublayer_total_row_serial_dot_ops_per_tile_parallel_group={d}  semantic_ffn_sublayer_width_lane_slots={d}  semantic_ffn_sublayer_active_width_lanes={d}  semantic_ffn_sublayer_width_lane_utilization_x1000={d}  semantic_ffn_sublayer_thread_lane_slots={d}  semantic_ffn_sublayer_active_thread_lanes={d}  semantic_ffn_sublayer_thread_lane_utilization_x1000={d}\n",
        .{
            name,
            rt.semantic_ffn_sublayer_count,
            rt.semantic_ffn_sublayer_rows,
            rt.semantic_ffn_sublayer_hidden,
            rt.semantic_ffn_sublayer_input,
            rt.semantic_ffn_sublayer_output,
            rt.semantic_ffn_sublayer_row_serial_dot_ops,
            rt.semantic_ffn_sublayer_total_row_serial_dot_ops,
            rt.semantic_ffn_sublayer_tile_row_groups,
            rt.semantic_ffn_sublayer_tile_hidden_tiles,
            rt.semantic_ffn_sublayer_tile_output_tiles,
            rt.semantic_ffn_sublayer_tile_parallel_groups,
            semantic_row_serial_per_tile_group,
            semantic_total_row_serial_per_tile_group,
            rt.semantic_ffn_sublayer_width_lane_slots,
            rt.semantic_ffn_sublayer_active_width_lanes,
            semantic_width_lane_utilization_x1000,
            rt.semantic_ffn_sublayer_thread_lane_slots,
            rt.semantic_ffn_sublayer_active_thread_lanes,
            semantic_thread_lane_utilization_x1000,
        },
    );
    const direct_row_serial_per_threadgroup = if (rt.semantic_ffn_with_input_direct_row_threadgroups > 0) rt.semantic_ffn_with_input_direct_total_row_serial_dot_ops / rt.semantic_ffn_with_input_direct_row_threadgroups else 0;
    try w.print(
        "  {s:<28} semantic_ffn_with_input_direct_count={d}  semantic_ffn_with_input_direct_rows={d}  semantic_ffn_with_input_direct_input_projection={d}  semantic_ffn_with_input_direct_input={d}  semantic_ffn_with_input_direct_hidden={d}  semantic_ffn_with_input_direct_output={d}  semantic_ffn_with_input_direct_input_projection_dot_ops={d}  semantic_ffn_with_input_direct_gate_up_dot_ops={d}  semantic_ffn_with_input_direct_down_dot_ops={d}  semantic_ffn_with_input_direct_row_serial_dot_ops={d}  semantic_ffn_with_input_direct_total_row_serial_dot_ops={d}  semantic_ffn_with_input_direct_row_threadgroups={d}  semantic_ffn_with_input_direct_total_row_serial_dot_ops_per_row_threadgroup={d}\n",
        .{
            name,
            rt.semantic_ffn_with_input_direct_count,
            rt.semantic_ffn_with_input_direct_rows,
            rt.semantic_ffn_with_input_direct_input_projection,
            rt.semantic_ffn_with_input_direct_input,
            rt.semantic_ffn_with_input_direct_hidden,
            rt.semantic_ffn_with_input_direct_output,
            rt.semantic_ffn_with_input_direct_input_projection_dot_ops,
            rt.semantic_ffn_with_input_direct_gate_up_dot_ops,
            rt.semantic_ffn_with_input_direct_down_dot_ops,
            rt.semantic_ffn_with_input_direct_row_serial_dot_ops,
            rt.semantic_ffn_with_input_direct_total_row_serial_dot_ops,
            rt.semantic_ffn_with_input_direct_row_threadgroups,
            direct_row_serial_per_threadgroup,
        },
    );
}

fn writeSemanticSublayerRuntimeMetricJson(
    w: *std.Io.Writer,
    name: []const u8,
    rt: profile_mod.RuntimeProfile,
    semantic_row_serial_per_tile_group: u64,
    semantic_total_row_serial_per_tile_group: u64,
) !void {
    const projection_row_chain_idx = @intFromEnum(program_mod.ProgramCommandKind.projection_row_chain);
    const semantic_idx = @intFromEnum(program_mod.ProgramCommandKind.semantic_ffn_sublayer);
    const semantic_with_input_idx = @intFromEnum(program_mod.ProgramCommandKind.semantic_ffn_sublayer_with_input_row_chain);
    var jw = try writeMetricJsonPrefix(w, name, "semantic_sublayer_runtime");
    try writeMetricJsonField(&jw, "runtime_backend_dispatches", rt.backend_dispatch_count);
    try writeMetricJsonField(&jw, "semantic_target_dispatches", 1);
    try writeMetricJsonField(&jw, "runtime_projection_row_chain_dispatches", rt.program_command_dispatch_counts[projection_row_chain_idx]);
    try writeMetricJsonField(&jw, "runtime_projection_row_chain_attempts", rt.program_command_attempt_counts[projection_row_chain_idx]);
    try writeMetricJsonField(&jw, "runtime_projection_row_chain_refused", rt.program_command_failed_counts[projection_row_chain_idx]);
    try writeMetricJsonField(&jw, "runtime_semantic_ffn_dispatches", rt.program_command_dispatch_counts[semantic_idx]);
    try writeMetricJsonField(&jw, "runtime_semantic_ffn_with_input_dispatches", rt.program_command_dispatch_counts[semantic_with_input_idx]);
    try writeMetricJsonField(&jw, "runtime_semantic_ffn_with_input_attempts", rt.program_command_attempt_counts[semantic_with_input_idx]);
    try writeMetricJsonField(&jw, "runtime_semantic_ffn_with_input_refused", rt.program_command_failed_counts[semantic_with_input_idx]);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_fallback_pair_dispatches", rt.semantic_ffn_sublayer_fallback_pair_dispatches);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_fallback_tail_dispatches", rt.semantic_ffn_sublayer_fallback_tail_dispatches);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_decomposed_count", rt.semantic_ffn_with_input_decomposed_count);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_decomposed_dispatches", rt.semantic_ffn_with_input_decomposed_dispatches);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_decomposed_row_chain_dispatches", rt.semantic_ffn_with_input_decomposed_row_chain_dispatches);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_decomposed_pair_dispatches", rt.semantic_ffn_with_input_decomposed_pair_dispatches);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_decomposed_tail_dispatches", rt.semantic_ffn_with_input_decomposed_tail_dispatches);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_direct_count", rt.semantic_ffn_with_input_direct_count);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_direct_rows", rt.semantic_ffn_with_input_direct_rows);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_direct_input_projection", rt.semantic_ffn_with_input_direct_input_projection);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_direct_input", rt.semantic_ffn_with_input_direct_input);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_direct_hidden", rt.semantic_ffn_with_input_direct_hidden);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_direct_output", rt.semantic_ffn_with_input_direct_output);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_direct_input_projection_dot_ops", rt.semantic_ffn_with_input_direct_input_projection_dot_ops);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_direct_gate_up_dot_ops", rt.semantic_ffn_with_input_direct_gate_up_dot_ops);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_direct_down_dot_ops", rt.semantic_ffn_with_input_direct_down_dot_ops);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_direct_row_serial_dot_ops", rt.semantic_ffn_with_input_direct_row_serial_dot_ops);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_direct_total_row_serial_dot_ops", rt.semantic_ffn_with_input_direct_total_row_serial_dot_ops);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_direct_row_threadgroups", rt.semantic_ffn_with_input_direct_row_threadgroups);
    try writeMetricJsonField(&jw, "semantic_ffn_with_input_direct_total_row_serial_dot_ops_per_row_threadgroup", if (rt.semantic_ffn_with_input_direct_row_threadgroups > 0) rt.semantic_ffn_with_input_direct_total_row_serial_dot_ops / rt.semantic_ffn_with_input_direct_row_threadgroups else 0);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_count", rt.qmatmul_row_chain_tiled_count);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_row_tile_groups", rt.qmatmul_row_chain_tiled_row_tile_groups);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_n_tiles", rt.qmatmul_row_chain_tiled_n_tiles);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_serial_tile_loops", rt.qmatmul_row_chain_tiled_serial_tile_loops);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_partial_slots", rt.qmatmul_row_chain_tiled_partial_slots);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_scratch_capacity", rt.qmatmul_row_chain_tiled_scratch_capacity);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_two_phase_count", rt.qmatmul_row_chain_tiled_two_phase_count);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_finalize_tile_groups", rt.qmatmul_row_chain_tiled_finalize_tile_groups);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_finalize_elements", rt.qmatmul_row_chain_tiled_finalize_elements);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_spilled_elementwise", rt.qmatmul_row_chain_tiled_spilled_elementwise);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_spilled_input", rt.qmatmul_row_chain_tiled_spilled_input);
    try writeMetricJsonField(&jw, "qmatmul_row_chain_tiled_output_spills", rt.qmatmul_row_chain_tiled_output_spills);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_count", rt.semantic_ffn_sublayer_count);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_rows", rt.semantic_ffn_sublayer_rows);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_hidden", rt.semantic_ffn_sublayer_hidden);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_input", rt.semantic_ffn_sublayer_input);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_output", rt.semantic_ffn_sublayer_output);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_row_serial_dot_ops", rt.semantic_ffn_sublayer_row_serial_dot_ops);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_total_row_serial_dot_ops", rt.semantic_ffn_sublayer_total_row_serial_dot_ops);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_tile_row_groups", rt.semantic_ffn_sublayer_tile_row_groups);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_tile_hidden_tiles", rt.semantic_ffn_sublayer_tile_hidden_tiles);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_tile_output_tiles", rt.semantic_ffn_sublayer_tile_output_tiles);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_tile_parallel_groups", rt.semantic_ffn_sublayer_tile_parallel_groups);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_row_serial_dot_ops_per_tile_parallel_group", semantic_row_serial_per_tile_group);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_total_row_serial_dot_ops_per_tile_parallel_group", semantic_total_row_serial_per_tile_group);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_width_lane_slots", rt.semantic_ffn_sublayer_width_lane_slots);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_active_width_lanes", rt.semantic_ffn_sublayer_active_width_lanes);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_width_lane_utilization_x1000", if (rt.semantic_ffn_sublayer_width_lane_slots > 0) rt.semantic_ffn_sublayer_active_width_lanes * 1000 / rt.semantic_ffn_sublayer_width_lane_slots else 0);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_thread_lane_slots", rt.semantic_ffn_sublayer_thread_lane_slots);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_active_thread_lanes", rt.semantic_ffn_sublayer_active_thread_lanes);
    try writeMetricJsonField(&jw, "semantic_ffn_sublayer_thread_lane_utilization_x1000", if (rt.semantic_ffn_sublayer_thread_lane_slots > 0) rt.semantic_ffn_sublayer_active_thread_lanes * 1000 / rt.semantic_ffn_sublayer_thread_lane_slots else 0);
    try finishMetricJson(w, &jw);
}

const TensorComputeBench = struct {
    out: *Tensor(f32),

    fn run(self: *TensorComputeBench) void {
        self.out.compute();
    }

    fn consume(self: *TensorComputeBench) f64 {
        return checksum(self.out.data);
    }
};

const MatMulBench = struct {
    out: *Tensor(f32),

    fn run(self: *MatMulBench) void {
        self.out.compute();
    }

    fn consume(self: *MatMulBench) f64 {
        return checksum(self.out.data);
    }
};

const ChainBench = struct {
    x: *Tensor(f32),
    y: *Tensor(f32),
    bias: *Tensor(f32),
    tmp0: *Tensor(f32),
    tmp1: *Tensor(f32),
    tmp2: *Tensor(f32),
    tmp3: *Tensor(f32),
    out: *Tensor(f32),

    fn init(alloc: std.mem.Allocator, n: usize) !ChainBench {
        const x = try Tensor(f32).init(alloc, &.{n});
        errdefer x.deinit();
        const y = try Tensor(f32).init(alloc, &.{n});
        errdefer y.deinit();
        const bias = try Tensor(f32).init(alloc, &.{n});
        errdefer bias.deinit();
        const tmp0 = try Tensor(f32).init(alloc, &.{n});
        errdefer tmp0.deinit();
        const tmp1 = try Tensor(f32).init(alloc, &.{n});
        errdefer tmp1.deinit();
        const tmp2 = try Tensor(f32).init(alloc, &.{n});
        errdefer tmp2.deinit();
        const tmp3 = try Tensor(f32).init(alloc, &.{n});
        errdefer tmp3.deinit();
        const out = try Tensor(f32).init(alloc, &.{n});
        errdefer out.deinit();

        fillDeterministic(x.data, 101, 0.6);
        fillDeterministic(y.data, 102, 0.4);
        fillDeterministic(bias.data, 103, 0.2);

        return .{ .x = x, .y = y, .bias = bias, .tmp0 = tmp0, .tmp1 = tmp1, .tmp2 = tmp2, .tmp3 = tmp3, .out = out };
    }

    fn deinit(self: *ChainBench) void {
        self.x.deinit();
        self.y.deinit();
        self.bias.deinit();
        self.tmp0.deinit();
        self.tmp1.deinit();
        self.tmp2.deinit();
        self.tmp3.deinit();
        self.out.deinit();
    }

    fn run(self: *ChainBench) void {
        self.tmp0.computeMul(self.x, self.y);
        self.tmp1.computeAdd(self.tmp0, self.bias);
        self.tmp2.computeRelu(self.tmp1);
        self.tmp3.computeMul(self.tmp2, self.y);
        self.out.computeAdd(self.tmp3, self.x);
    }

    fn consume(self: *ChainBench) f64 {
        return checksum(self.out.data);
    }
};

const FusedChainBench = struct {
    x: *Tensor(f32),
    y: *Tensor(f32),
    bias: *Tensor(f32),
    out: *Tensor(f32),

    fn init(alloc: std.mem.Allocator, n: usize) !FusedChainBench {
        const x = try Tensor(f32).init(alloc, &.{n});
        errdefer x.deinit();
        const y = try Tensor(f32).init(alloc, &.{n});
        errdefer y.deinit();
        const bias = try Tensor(f32).init(alloc, &.{n});
        errdefer bias.deinit();
        const out = try Tensor(f32).init(alloc, &.{n});
        errdefer out.deinit();

        fillDeterministic(x.data, 101, 0.6);
        fillDeterministic(y.data, 102, 0.4);
        fillDeterministic(bias.data, 103, 0.2);

        return .{ .x = x, .y = y, .bias = bias, .out = out };
    }

    fn deinit(self: *FusedChainBench) void {
        self.x.deinit();
        self.y.deinit();
        self.bias.deinit();
        self.out.deinit();
    }

    fn run(self: *FusedChainBench) void {
        const vec_len = FrontierStencilVecLen;
        const Vec = @Vector(vec_len, f32);
        const zero: Vec = @splat(0);
        var i: usize = 0;
        while (i + vec_len <= self.out.data.len) : (i += vec_len) {
            const xv: Vec = self.x.data[i..][0..vec_len].*;
            const yv: Vec = self.y.data[i..][0..vec_len].*;
            const bv: Vec = self.bias.data[i..][0..vec_len].*;
            self.out.data[i..][0..vec_len].* = @max(xv * yv + bv, zero) * yv + xv;
        }
        while (i < self.out.data.len) : (i += 1) {
            const x = self.x.data[i];
            const y = self.y.data[i];
            const h = @max(x * y + self.bias.data[i], 0);
            self.out.data[i] = h * y + x;
        }
    }

    fn consume(self: *FusedChainBench) f64 {
        return checksum(self.out.data);
    }
};

fn benchElementwise(io: std.Io, alloc: std.mem.Allocator, w: *std.Io.Writer, filter: FrontierFilter) !void {
    if (!filter.matchesAny(&.{ "Elementwise", "chain" })) return;
    try w.print("\nElementwise Chain And Fusion\n", .{});
    try w.print("----------------------------\n", .{});

    const sizes = [_]usize{ 4_096, 262_144 };
    for (sizes) |n| {
        var unfused_name_buf: [64]u8 = undefined;
        const unfused_name = try std.fmt.bufPrint(&unfused_name_buf, "chain n={d} staged", .{n});
        var fused_name_buf: [64]u8 = undefined;
        const fused_name = try std.fmt.bufPrint(&fused_name_buf, "chain n={d} one-pass", .{n});
        if (!filter.matchesAny(&.{ unfused_name, fused_name })) continue;

        var unfused = try ChainBench.init(alloc, n);
        defer unfused.deinit();
        var fused = try FusedChainBench.init(alloc, n);
        defer fused.deinit();

        const unfused_stats = measure(io, &unfused);
        const fused_stats = measure(io, &fused);
        const elems = @as(f64, @floatFromInt(n));
        try printStats(w, unfused_name, "elems", elems, "elem", unfused_stats);

        try printStats(w, fused_name, "elems", elems, "elem", fused_stats);
    }
}

const MatmulCase = struct {
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    trans_b: bool = false,
};

fn benchMatmul(io: std.Io, alloc: std.mem.Allocator, w: *std.Io.Writer, filter: FrontierFilter) !void {
    if (!filter.matchesAny(&.{ "Matmul", "gemv", "projection", "attention" })) return;
    try w.print("\nMatmul Shape Regimes\n", .{});
    try w.print("--------------------\n", .{});

    const cases = [_]MatmulCase{
        .{ .name = "decode gemv", .m = 1, .n = 512, .k = 256 },
        .{ .name = "small square", .m = 128, .n = 128, .k = 128 },
        .{ .name = "batched projection", .m = 32, .n = 512, .k = 256 },
        .{ .name = "attention scores", .m = 64, .n = 128, .k = 64, .trans_b = true },
    };

    for (cases) |case| {
        if (!filter.matches(case.name)) continue;
        const a = try Tensor(f32).init(alloc, &.{ case.k, case.m });
        defer a.deinit();
        const b_shape = if (case.trans_b) [_]usize{ case.k, case.n } else [_]usize{ case.n, case.k };
        const b = try Tensor(f32).init(alloc, &b_shape);
        defer b.deinit();
        fillDeterministic(a.data, 201, 0.08);
        fillDeterministic(b.data, 202, 0.08);

        const out = a.matMul(false, b, case.trans_b);
        defer out.deinit();
        var bench = MatMulBench{ .out = out };
        const stats = measure(io, &bench);
        const flops = 2.0 * @as(f64, @floatFromInt(case.m)) * @as(f64, @floatFromInt(case.n)) * @as(f64, @floatFromInt(case.k));
        try printStats(w, case.name, "throughput", flops / 1_000_000_000.0, "GFLOP", stats);
    }
}

fn benchNorms(io: std.Io, alloc: std.mem.Allocator, w: *std.Io.Writer, filter: FrontierFilter) !void {
    if (!filter.matchesAny(&.{ "Softmax", "RMSNorm", "softmax", "rmsnorm" })) return;
    try w.print("\nSoftmax And RMSNorm\n", .{});
    try w.print("-------------------\n", .{});

    if (filter.matches("softmax 1024 x 32")) {
        const rows: usize = 1024;
        const cols: usize = 32;
        const logits = try Tensor(f32).init(alloc, &.{ rows, cols });
        defer logits.deinit();
        fillDeterministic(logits.data, 301, 2.0);
        const out = logits.softmax(&.{ 1, cols });
        defer out.deinit();
        var bench = TensorComputeBench{ .out = out };
        const stats = measure(io, &bench);
        try printStats(w, "softmax 1024 x 32", "elems", @floatFromInt(rows * cols), "elem", stats);
    }

    if (filter.matches("rmsnorm 768 x 64")) {
        const hidden: usize = 768;
        const tokens: usize = 64;
        const x = try Tensor(f32).init(alloc, &.{ hidden, tokens });
        defer x.deinit();
        fillDeterministic(x.data, 302, 0.5);
        const out = x.rmsNorm(&.{ 1, tokens }, 1e-5);
        defer out.deinit();
        var bench = TensorComputeBench{ .out = out };
        const stats = measure(io, &bench);
        try printStats(w, "rmsnorm 768 x 64", "elems", @floatFromInt(hidden * tokens), "elem", stats);
    }
}

const DecodeBench = struct {
    x: *Tensor(f32),
    norm_w: *Tensor(f32),
    wq: *Tensor(f32),
    k_cache: *Tensor(f32),
    v_cache: *Tensor(f32),
    mask: *Tensor(f32),
    wout: *Tensor(f32),
    rms: *Tensor(f32),
    norm: *Tensor(f32),
    q: *Tensor(f32),
    scores: *Tensor(f32),
    masked: *Tensor(f32),
    probs: *Tensor(f32),
    context: *Tensor(f32),
    logits: *Tensor(f32),

    fn init(alloc: std.mem.Allocator) !DecodeBench {
        const d_model: usize = 64;
        const seq: usize = 32;
        const vocab: usize = 512;

        const x = try Tensor(f32).init(alloc, &.{ d_model, 1 });
        errdefer x.deinit();
        const norm_w = try Tensor(f32).init(alloc, &.{ d_model, 1 });
        errdefer norm_w.deinit();
        const wq = try Tensor(f32).init(alloc, &.{ d_model, d_model });
        errdefer wq.deinit();
        const k_cache = try Tensor(f32).init(alloc, &.{ d_model, seq });
        errdefer k_cache.deinit();
        const v_cache = try Tensor(f32).init(alloc, &.{ d_model, seq });
        errdefer v_cache.deinit();
        const mask = try Tensor(f32).init(alloc, &.{ seq, 1 });
        errdefer mask.deinit();
        const wout = try Tensor(f32).init(alloc, &.{ vocab, d_model });
        errdefer wout.deinit();

        fillDeterministic(x.data, 401, 0.25);
        _ = norm_w.setAllScalar(1);
        fillDeterministic(wq.data, 402, 0.08);
        fillDeterministic(k_cache.data, 403, 0.05);
        fillDeterministic(v_cache.data, 404, 0.05);
        for (mask.data, 0..) |*v, i| v.* = if (i <= 15) 0 else -1e9;
        fillDeterministic(wout.data, 405, 0.08);

        const rms = x.rmsNorm(&.{ 1, 1 }, 1e-5);
        errdefer rms.deinit();
        const norm = rms.mul(norm_w);
        errdefer norm.deinit();
        const q = norm.matMul(false, wq, false);
        errdefer q.deinit();
        const scores0 = q.matMul(false, k_cache, true);
        errdefer scores0.deinit();
        const masked = scores0.add(mask);
        errdefer masked.deinit();
        const probs = masked.softmax(&.{ 1, 1 });
        errdefer probs.deinit();
        const context = probs.matMul(false, v_cache, false);
        errdefer context.deinit();
        const logits = context.matMul(false, wout, false);
        errdefer logits.deinit();

        return .{
            .x = x,
            .norm_w = norm_w,
            .wq = wq,
            .k_cache = k_cache,
            .v_cache = v_cache,
            .mask = mask,
            .wout = wout,
            .rms = rms,
            .norm = norm,
            .q = q,
            .scores = scores0,
            .masked = masked,
            .probs = probs,
            .context = context,
            .logits = logits,
        };
    }

    fn deinit(self: *DecodeBench) void {
        self.logits.deinit();
        self.context.deinit();
        self.probs.deinit();
        self.masked.deinit();
        self.scores.deinit();
        self.q.deinit();
        self.norm.deinit();
        self.rms.deinit();
        self.wout.deinit();
        self.mask.deinit();
        self.v_cache.deinit();
        self.k_cache.deinit();
        self.wq.deinit();
        self.norm_w.deinit();
        self.x.deinit();
    }

    fn run(self: *DecodeBench) void {
        self.rms.compute();
        self.norm.compute();
        self.q.compute();
        self.scores.compute();
        self.masked.compute();
        self.probs.compute();
        self.context.compute();
        self.logits.compute();
    }

    fn consume(self: *DecodeBench) f64 {
        return checksum(self.logits.data);
    }
};

fn benchDecodeGraph(io: std.Io, alloc: std.mem.Allocator, w: *std.Io.Writer, filter: FrontierFilter) !void {
    if (!filter.matchesAny(&.{ "Decode", "rmsnorm-attn-logits token" })) return;
    try w.print("\nDecode-ish Inference Path\n", .{});
    try w.print("-------------------------\n", .{});

    var bench = try DecodeBench.init(alloc);
    defer bench.deinit();
    const stats = measure(io, &bench);
    try printStats(w, "rmsnorm-attn-logits token", "tokens", 1.0, "tok", stats);
}

const ProjectionRowChainMetalBench = struct {
    be: backend_mod.Backend,
    handle: backend_mod.Backend.CompiledHandle,
    out: []f32,
    output_io: []const backend_mod.ProgramIO,

    fn run(self: *ProjectionRowChainMetalBench) void {
        self.be.executeProgram(self.handle, &.{}, self.output_io);
    }

    fn consume(self: *ProjectionRowChainMetalBench) f64 {
        return checksum(self.out);
    }
};

const ProjectionRowChainCase = struct {
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    hidden: ?usize = null,
    output: ?usize = null,
};

fn allocF32(alloc: std.mem.Allocator, n: usize, seed: u64, scale: f32) ![]f32 {
    const data = try alloc.alloc(f32, n);
    fillDeterministic(data, seed, scale);
    return data;
}

fn allocI8Weights(alloc: std.mem.Allocator, n: usize, seed: u64) ![]i8 {
    const data = try alloc.alloc(i8, n);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (data) |*v| {
        v.* = @intCast(rng.intRangeAtMost(i16, -12, 12));
    }
    return data;
}

fn programIo(buf_idx: u16, data: anytype) backend_mod.ProgramIO {
    return .{
        .buf_idx = buf_idx,
        .host_ptr = @ptrCast(data.ptr),
        .size = @intCast(data.len * @sizeOf(@TypeOf(data[0]))),
    };
}

fn benchProjectionChainMetalCase(
    io: std.Io,
    alloc: std.mem.Allocator,
    w: *std.Io.Writer,
    metal: *internal.backend_metal.MetalBackend,
    case: ProjectionRowChainCase,
) !void {
    const elems = case.m * case.n;
    const input_len = case.m * case.k;
    const block_size: usize = 32;
    const scale_len = (case.k * case.n + block_size - 1) / block_size;

    const input = try allocF32(alloc, input_len, 601, 0.25);
    defer alloc.free(input);
    const q_out = try allocF32(alloc, elems, 602, 0.0);
    defer alloc.free(q_out);
    const residual = try allocF32(alloc, elems, 603, 0.20);
    defer alloc.free(residual);
    const ew_out = try allocF32(alloc, elems, 604, 0.0);
    defer alloc.free(ew_out);
    const default_out = try allocF32(alloc, elems, 605, 0.0);
    defer alloc.free(default_out);
    const fused_out = try allocF32(alloc, elems, 606, 0.0);
    defer alloc.free(fused_out);
    const qdata = try allocI8Weights(alloc, case.k * case.n, 607);
    defer alloc.free(qdata);
    const qscales = try allocF32(alloc, scale_len, 608, 0.02);
    defer alloc.free(qscales);
    for (qscales) |*v| v.* = 0.02;

    const ops = [_]backend_mod.DeviceOp{
        .{ .qmatmul = .{ .dst = 1, .input = 0, .weight_idx = 0, .M = @intCast(case.m), .N = @intCast(case.n), .K = @intCast(case.k) } },
        .{ .elementwise = .{ .op = .add, .dst = 3, .src0 = 1, .src1 = 2, .n = @intCast(elems) } },
    };
    const buffer_sizes = [_]usize{ input_len, elems, elems, elems };
    const uploads = [_]backend_mod.ProgramIO{
        programIo(0, input),
        programIo(1, q_out),
        programIo(2, residual),
        programIo(3, ew_out),
    };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = qdata,
        .scales = qscales,
        .rows = case.k,
        .cols = case.n,
        .block_size = block_size,
    }};
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const be = metal.backend();
    var staged_policy = program_mod.CommandStreamPolicy.default();
    staged_policy.fuse_projection_chain = false;
    const staged_handle = metal.compileProgramWithCommandPolicy(program, staged_policy) orelse return error.CompileFailed;
    defer be.freeProgram(staged_handle);
    const fused_handle = metal.compileProgramWithCommandPolicy(program, program_mod.CommandStreamPolicy.default()) orelse return error.CompileFailed;
    defer be.freeProgram(fused_handle);

    const staged_output_io = [_]backend_mod.ProgramIO{programIo(3, default_out)};
    const fused_output_io = [_]backend_mod.ProgramIO{programIo(3, fused_out)};
    var staged_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = staged_handle,
        .out = default_out,
        .output_io = &staged_output_io,
    };
    var fused_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = fused_handle,
        .out = fused_out,
        .output_io = &fused_output_io,
    };

    const staged_stats = measure(io, &staged_bench);
    const fused_stats = measure(io, &fused_bench);
    const approx_work = 2.0 * @as(f64, @floatFromInt(case.m * case.n * case.k));

    var staged_name_buf: [96]u8 = undefined;
    const staged_name = try std.fmt.bufPrint(&staged_name_buf, "{s} staged", .{case.name});
    try printStats(w, staged_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", staged_stats);

    var fused_name_buf: [96]u8 = undefined;
    const fused_name = try std.fmt.bufPrint(&fused_name_buf, "{s} projection_chain", .{case.name});
    try printStats(w, fused_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", fused_stats);

    var ratio_name_buf: [96]u8 = undefined;
    const ratio_name = try std.fmt.bufPrint(&ratio_name_buf, "{s} projection_chain", .{case.name});
    try printRatio(w, ratio_name, staged_stats, fused_stats, maxAbsDiff(default_out, fused_out));
}

fn benchProjectionGroupMetalCase(
    comptime n_slots: usize,
    io: std.Io,
    alloc: std.mem.Allocator,
    w: *std.Io.Writer,
    metal: *internal.backend_metal.MetalBackend,
    case: ProjectionRowChainCase,
) !void {
    const elems = case.m * case.n;
    const input_len = case.m * case.k;
    const block_size: usize = 32;
    const scale_len = (case.k * case.n + block_size - 1) / block_size;

    const input = try allocF32(alloc, input_len, 701, 0.25);
    defer alloc.free(input);
    const q_out = try allocF32(alloc, n_slots * elems, 702, 0.0);
    defer alloc.free(q_out);
    const residual = try allocF32(alloc, n_slots * elems, 703, 0.20);
    defer alloc.free(residual);
    const staged_out = try allocF32(alloc, n_slots * elems, 704, 0.0);
    defer alloc.free(staged_out);
    const grouped_out = try allocF32(alloc, n_slots * elems, 705, 0.0);
    defer alloc.free(grouped_out);
    const qdata = try allocI8Weights(alloc, n_slots * case.k * case.n, 706);
    defer alloc.free(qdata);
    const qscales = try allocF32(alloc, n_slots * scale_len, 707, 0.02);
    defer alloc.free(qscales);
    for (qscales) |*v| v.* = 0.02;

    var ops: [n_slots * 2]backend_mod.DeviceOp = undefined;
    var buffer_sizes: [1 + n_slots * 3]usize = undefined;
    var uploads: [1 + n_slots * 3]backend_mod.ProgramIO = undefined;
    var qweights: [n_slots]backend_mod.QuantizedWeightUpload = undefined;

    buffer_sizes[0] = input_len;
    uploads[0] = programIo(0, input);
    for (0..n_slots) |slot| {
        const q_buf: u16 = @intCast(1 + slot * 3);
        const residual_buf: u16 = @intCast(2 + slot * 3);
        const out_buf: u16 = @intCast(3 + slot * 3);
        const base = slot * elems;
        const weight_base = slot * case.k * case.n;
        const scale_base = slot * scale_len;

        ops[slot * 2] = .{ .qmatmul = .{ .dst = q_buf, .input = 0, .weight_idx = @intCast(slot), .M = @intCast(case.m), .N = @intCast(case.n), .K = @intCast(case.k) } };
        ops[slot * 2 + 1] = .{ .elementwise = .{ .op = .add, .dst = out_buf, .src0 = q_buf, .src1 = residual_buf, .n = @intCast(elems) } };

        buffer_sizes[1 + slot * 3] = elems;
        buffer_sizes[2 + slot * 3] = elems;
        buffer_sizes[3 + slot * 3] = elems;
        uploads[1 + slot * 3] = programIo(q_buf, q_out[base .. base + elems]);
        uploads[2 + slot * 3] = programIo(residual_buf, residual[base .. base + elems]);
        uploads[3 + slot * 3] = programIo(out_buf, staged_out[base .. base + elems]);
        qweights[slot] = .{
            .data = qdata[weight_base .. weight_base + case.k * case.n],
            .scales = qscales[scale_base .. scale_base + scale_len],
            .rows = case.k,
            .cols = case.n,
            .block_size = block_size,
        };
    }

    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const be = metal.backend();
    var staged_policy = program_mod.CommandStreamPolicy.default();
    staged_policy.qmatmul_group_size = 1;
    const staged_handle = metal.compileProgramWithCommandPolicy(program, staged_policy) orelse return error.CompileFailed;
    defer be.freeProgram(staged_handle);
    const grouped_handle = metal.compileProgramWithCommandPolicy(program, program_mod.CommandStreamPolicy.default()) orelse return error.CompileFailed;
    defer be.freeProgram(grouped_handle);

    var staged_outputs: [n_slots]backend_mod.ProgramIO = undefined;
    var grouped_outputs: [n_slots]backend_mod.ProgramIO = undefined;
    for (0..n_slots) |slot| {
        const out_buf: u16 = @intCast(3 + slot * 3);
        const base = slot * elems;
        staged_outputs[slot] = programIo(out_buf, staged_out[base .. base + elems]);
        grouped_outputs[slot] = programIo(out_buf, grouped_out[base .. base + elems]);
    }

    var staged_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = staged_handle,
        .out = staged_out,
        .output_io = &staged_outputs,
    };
    var grouped_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = grouped_handle,
        .out = grouped_out,
        .output_io = &grouped_outputs,
    };

    const staged_stats = measure(io, &staged_bench);
    const grouped_stats = measure(io, &grouped_bench);
    const approx_work = 2.0 * @as(f64, @floatFromInt(n_slots * case.m * case.n * case.k));

    var staged_name_buf: [96]u8 = undefined;
    const staged_name = try std.fmt.bufPrint(&staged_name_buf, "{s} staged", .{case.name});
    try printStats(w, staged_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", staged_stats);

    var grouped_name_buf: [96]u8 = undefined;
    const grouped_name = try std.fmt.bufPrint(&grouped_name_buf, "{s} projection_group", .{case.name});
    try printStats(w, grouped_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", grouped_stats);

    var ratio_name_buf: [96]u8 = undefined;
    const ratio_name = try std.fmt.bufPrint(&ratio_name_buf, "{s} projection_group", .{case.name});
    try printRatio(w, ratio_name, staged_stats, grouped_stats, maxAbsDiff(staged_out, grouped_out));

    const grouped_commands = try program_mod.buildProgramCommands(alloc, &ops, program_mod.CommandStreamPolicy.default());
    defer alloc.free(grouped_commands);
    var profile_name_buf: [112]u8 = undefined;
    const profile_name = try std.fmt.bufPrint(&profile_name_buf, "{s} projection_group dispatch_profile", .{case.name});
    try printCommandShape(w, profile_name, &ops, grouped_commands);
    try printProjectionGroupRuntimeProfile(w, profile_name, be, grouped_handle, &grouped_outputs);
}

fn benchProjectionRowChainMetalCase(
    io: std.Io,
    alloc: std.mem.Allocator,
    w: *std.Io.Writer,
    metal: *internal.backend_metal.MetalBackend,
    case: ProjectionRowChainCase,
) !void {
    const elems = case.m * case.n;
    const input_len = case.m * case.k;
    const block_size: usize = 32;
    const scale_len = (case.k * case.n + block_size - 1) / block_size;

    const input = try allocF32(alloc, input_len, 501, 0.25);
    defer alloc.free(input);
    const q_out = try allocF32(alloc, elems, 502, 0.0);
    defer alloc.free(q_out);
    const residual = try allocF32(alloc, elems, 503, 0.20);
    defer alloc.free(residual);
    const norm_out = try allocF32(alloc, elems, 504, 0.0);
    defer alloc.free(norm_out);
    const scale = try allocF32(alloc, case.n, 505, 0.30);
    defer alloc.free(scale);
    const repeat_out = try allocF32(alloc, elems, 506, 0.0);
    defer alloc.free(repeat_out);
    const default_out = try allocF32(alloc, elems, 507, 0.0);
    defer alloc.free(default_out);
    const fused_out = try allocF32(alloc, elems, 508, 0.0);
    defer alloc.free(fused_out);
    const single_dispatch_out = try allocF32(alloc, elems, 511, 0.0);
    defer alloc.free(single_dispatch_out);
    const two_phase_out = try allocF32(alloc, elems, 512, 0.0);
    defer alloc.free(two_phase_out);
    const qdata = try allocI8Weights(alloc, case.k * case.n, 509);
    defer alloc.free(qdata);
    const qscales = try allocF32(alloc, scale_len, 510, 0.02);
    defer alloc.free(qscales);
    for (qscales) |*v| v.* = 0.02;

    const ops = [_]backend_mod.DeviceOp{
        .{ .qmatmul = .{ .dst = 1, .input = 0, .weight_idx = 0, .M = @intCast(case.m), .N = @intCast(case.n), .K = @intCast(case.k) } },
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 2, .n = @intCast(elems) } },
        .{ .rmsnorm = .{ .dst = 3, .src = 2, .rows = @intCast(case.m), .cols = @intCast(case.n), .eps = 1e-5 } },
        .{ .repeat = .{
            .dst = 5,
            .src = 4,
            .n = @intCast(elems),
            .src_ne = .{ @intCast(case.n), 1, 1, 1 },
            .dst_ne = .{ @intCast(case.n), @intCast(case.m), 1, 1 },
            .src_strides = .{ 1, @intCast(case.n), @intCast(case.n), @intCast(case.n) },
            .dst_strides = .{ 1, @intCast(case.n), @intCast(elems), @intCast(elems) },
        } },
        .{ .elementwise = .{ .op = .mul, .dst = 6, .src0 = 3, .src1 = 5, .n = @intCast(elems) } },
    };
    const buffer_sizes = [_]usize{ input_len, elems, elems, elems, case.n, elems, elems };
    const uploads = [_]backend_mod.ProgramIO{
        programIo(0, input),
        programIo(1, q_out),
        programIo(2, residual),
        programIo(3, norm_out),
        programIo(4, scale),
        programIo(5, repeat_out),
        programIo(6, default_out),
    };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = qdata,
        .scales = qscales,
        .rows = case.k,
        .cols = case.n,
        .block_size = block_size,
    }};
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const be = metal.backend();
    var legacy_policy = program_mod.CommandStreamPolicy.default();
    legacy_policy.fuse_projection_row_chain = false;
    const default_handle = metal.compileProgramWithCommandPolicy(program, legacy_policy) orelse return error.CompileFailed;
    defer be.freeProgram(default_handle);
    var fused_policy = program_mod.CommandStreamPolicy.default();
    fused_policy.fuse_projection_row_chain = true;
    fused_policy.fuse_projection_row_chain_qmatvec = true;
    const fused_handle = metal.compileProgramWithCommandPolicy(program, fused_policy) orelse return error.CompileFailed;
    defer be.freeProgram(fused_handle);
    var single_dispatch_policy = fused_policy;
    single_dispatch_policy.fuse_projection_row_chain_single_dispatch = true;
    const single_dispatch_handle = metal.compileProgramWithCommandPolicy(program, single_dispatch_policy) orelse return error.CompileFailed;
    defer be.freeProgram(single_dispatch_handle);
    const two_phase_policy = program_mod.CommandStreamPolicy.promptProjectionRowChainTwoPhaseCandidate();
    const two_phase_handle = metal.compileProgramWithCommandPolicy(program, two_phase_policy) orelse return error.CompileFailed;
    defer be.freeProgram(two_phase_handle);

    const default_output_io = [_]backend_mod.ProgramIO{programIo(6, default_out)};
    const fused_output_io = [_]backend_mod.ProgramIO{programIo(6, fused_out)};
    const single_dispatch_output_io = [_]backend_mod.ProgramIO{programIo(6, single_dispatch_out)};
    const two_phase_output_io = [_]backend_mod.ProgramIO{programIo(6, two_phase_out)};
    var default_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = default_handle,
        .out = default_out,
        .output_io = &default_output_io,
    };
    var fused_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = fused_handle,
        .out = fused_out,
        .output_io = &fused_output_io,
    };
    var single_dispatch_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = single_dispatch_handle,
        .out = single_dispatch_out,
        .output_io = &single_dispatch_output_io,
    };
    var two_phase_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = two_phase_handle,
        .out = two_phase_out,
        .output_io = &two_phase_output_io,
    };

    const default_stats = measure(io, &default_bench);
    const fused_stats = measure(io, &fused_bench);
    const single_dispatch_stats = measure(io, &single_dispatch_bench);
    const two_phase_stats = measure(io, &two_phase_bench);
    const approx_work = 2.0 * @as(f64, @floatFromInt(case.m * case.n * case.k));

    var default_name_buf: [96]u8 = undefined;
    const default_name = try std.fmt.bufPrint(&default_name_buf, "{s} default", .{case.name});
    try printStats(w, default_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", default_stats);

    var fused_name_buf: [96]u8 = undefined;
    const fused_name = try std.fmt.bufPrint(&fused_name_buf, "{s} projection_row_chain", .{case.name});
    try printStats(w, fused_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", fused_stats);

    var single_dispatch_name_buf: [112]u8 = undefined;
    const single_dispatch_name = try std.fmt.bufPrint(&single_dispatch_name_buf, "{s} projection_row_chain_single_dispatch", .{case.name});
    try printStats(w, single_dispatch_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", single_dispatch_stats);

    var two_phase_name_buf: [112]u8 = undefined;
    const two_phase_name = try std.fmt.bufPrint(&two_phase_name_buf, "{s} projection_row_chain_two_phase", .{case.name});
    try printStats(w, two_phase_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", two_phase_stats);

    var ratio_name_buf: [96]u8 = undefined;
    const ratio_name = try std.fmt.bufPrint(&ratio_name_buf, "{s} projection_row_chain", .{case.name});
    try printRatio(w, ratio_name, default_stats, fused_stats, maxAbsDiff(default_out, fused_out));

    var single_dispatch_ratio_name_buf: [128]u8 = undefined;
    const single_dispatch_ratio_name = try std.fmt.bufPrint(&single_dispatch_ratio_name_buf, "{s} projection_row_chain_single_dispatch", .{case.name});
    try printRatio(w, single_dispatch_ratio_name, default_stats, single_dispatch_stats, maxAbsDiff(default_out, single_dispatch_out));

    var two_phase_ratio_name_buf: [128]u8 = undefined;
    const two_phase_ratio_name = try std.fmt.bufPrint(&two_phase_ratio_name_buf, "{s} projection_row_chain_two_phase", .{case.name});
    try printRatio(w, two_phase_ratio_name, default_stats, two_phase_stats, maxAbsDiff(default_out, two_phase_out));

    const fused_commands = try program_mod.buildProgramCommands(alloc, &ops, fused_policy);
    defer alloc.free(fused_commands);
    var profile_name_buf: [112]u8 = undefined;
    const profile_name = try std.fmt.bufPrint(&profile_name_buf, "{s} projection_row_chain dispatch_profile", .{case.name});
    try printCommandShape(w, profile_name, &ops, fused_commands);
    try printProjectionRowChainRuntimeProfile(w, profile_name, be, fused_handle, &fused_output_io);

    const single_dispatch_commands = try program_mod.buildProgramCommands(alloc, &ops, single_dispatch_policy);
    defer alloc.free(single_dispatch_commands);
    var single_dispatch_profile_name_buf: [128]u8 = undefined;
    const single_dispatch_profile_name = try std.fmt.bufPrint(&single_dispatch_profile_name_buf, "{s} projection_row_chain_single_dispatch dispatch_profile", .{case.name});
    try printCommandShape(w, single_dispatch_profile_name, &ops, single_dispatch_commands);
    try printProjectionRowChainRuntimeProfile(w, single_dispatch_profile_name, be, single_dispatch_handle, &single_dispatch_output_io);

    const two_phase_commands = try program_mod.buildProgramCommands(alloc, &ops, two_phase_policy);
    defer alloc.free(two_phase_commands);
    var two_phase_profile_name_buf: [128]u8 = undefined;
    const two_phase_profile_name = try std.fmt.bufPrint(&two_phase_profile_name_buf, "{s} projection_row_chain_two_phase dispatch_profile", .{case.name});
    try printCommandShape(w, two_phase_profile_name, &ops, two_phase_commands);
    try printProjectionRowChainRuntimeProfile(w, two_phase_profile_name, be, two_phase_handle, &two_phase_output_io);
}

fn benchProjectionRowChainGroupMetalCase(
    comptime n_slots: usize,
    io: std.Io,
    alloc: std.mem.Allocator,
    w: *std.Io.Writer,
    metal: *internal.backend_metal.MetalBackend,
    case: ProjectionRowChainCase,
) !void {
    const elems = case.m * case.n;
    const input_len = case.m * case.k;
    const block_size: usize = 32;
    const scale_len = (case.k * case.n + block_size - 1) / block_size;

    const input = try allocF32(alloc, input_len, 801, 0.25);
    defer alloc.free(input);
    const q_out = try allocF32(alloc, n_slots * elems, 802, 0.0);
    defer alloc.free(q_out);
    const residual = try allocF32(alloc, n_slots * elems, 803, 0.20);
    defer alloc.free(residual);
    const norm_out = try allocF32(alloc, n_slots * elems, 804, 0.0);
    defer alloc.free(norm_out);
    const scale = try allocF32(alloc, n_slots * case.n, 805, 0.30);
    defer alloc.free(scale);
    const repeat_out = try allocF32(alloc, n_slots * elems, 806, 0.0);
    defer alloc.free(repeat_out);
    const staged_out = try allocF32(alloc, n_slots * elems, 807, 0.0);
    defer alloc.free(staged_out);
    const grouped_out = try allocF32(alloc, n_slots * elems, 808, 0.0);
    defer alloc.free(grouped_out);
    const two_phase_out = try allocF32(alloc, n_slots * elems, 811, 0.0);
    defer alloc.free(two_phase_out);
    const qdata = try allocI8Weights(alloc, n_slots * case.k * case.n, 809);
    defer alloc.free(qdata);
    const qscales = try allocF32(alloc, n_slots * scale_len, 810, 0.02);
    defer alloc.free(qscales);
    for (qscales) |*v| v.* = 0.02;

    var ops: [n_slots * 5]backend_mod.DeviceOp = undefined;
    var buffer_sizes: [1 + n_slots * 6]usize = undefined;
    var uploads: [1 + n_slots * 6]backend_mod.ProgramIO = undefined;
    var qweights: [n_slots]backend_mod.QuantizedWeightUpload = undefined;

    buffer_sizes[0] = input_len;
    uploads[0] = programIo(0, input);
    for (0..n_slots) |slot| {
        const q_buf: u16 = @intCast(1 + slot * 6);
        const residual_buf: u16 = @intCast(2 + slot * 6);
        const norm_buf: u16 = @intCast(3 + slot * 6);
        const scale_buf: u16 = @intCast(4 + slot * 6);
        const repeat_buf: u16 = @intCast(5 + slot * 6);
        const out_buf: u16 = @intCast(6 + slot * 6);
        const elem_base = slot * elems;
        const scale_base = slot * case.n;
        const weight_base = slot * case.k * case.n;
        const qscale_base = slot * scale_len;
        const op_base = slot * 5;

        ops[op_base] = .{ .qmatmul = .{ .dst = q_buf, .input = 0, .weight_idx = @intCast(slot), .M = @intCast(case.m), .N = @intCast(case.n), .K = @intCast(case.k) } };
        ops[op_base + 1] = .{ .elementwise = .{ .op = .add, .dst = residual_buf, .src0 = q_buf, .src1 = residual_buf, .n = @intCast(elems) } };
        ops[op_base + 2] = .{ .rmsnorm = .{ .dst = norm_buf, .src = residual_buf, .rows = @intCast(case.m), .cols = @intCast(case.n), .eps = 1e-5 } };
        ops[op_base + 3] = .{ .repeat = .{
            .dst = repeat_buf,
            .src = scale_buf,
            .n = @intCast(elems),
            .src_ne = .{ @intCast(case.n), 1, 1, 1 },
            .dst_ne = .{ @intCast(case.n), @intCast(case.m), 1, 1 },
            .src_strides = .{ 1, @intCast(case.n), @intCast(case.n), @intCast(case.n) },
            .dst_strides = .{ 1, @intCast(case.n), @intCast(elems), @intCast(elems) },
        } };
        ops[op_base + 4] = .{ .elementwise = .{ .op = .mul, .dst = out_buf, .src0 = norm_buf, .src1 = repeat_buf, .n = @intCast(elems) } };

        buffer_sizes[1 + slot * 6] = elems;
        buffer_sizes[2 + slot * 6] = elems;
        buffer_sizes[3 + slot * 6] = elems;
        buffer_sizes[4 + slot * 6] = case.n;
        buffer_sizes[5 + slot * 6] = elems;
        buffer_sizes[6 + slot * 6] = elems;
        uploads[1 + slot * 6] = programIo(q_buf, q_out[elem_base .. elem_base + elems]);
        uploads[2 + slot * 6] = programIo(residual_buf, residual[elem_base .. elem_base + elems]);
        uploads[3 + slot * 6] = programIo(norm_buf, norm_out[elem_base .. elem_base + elems]);
        uploads[4 + slot * 6] = programIo(scale_buf, scale[scale_base .. scale_base + case.n]);
        uploads[5 + slot * 6] = programIo(repeat_buf, repeat_out[elem_base .. elem_base + elems]);
        uploads[6 + slot * 6] = programIo(out_buf, staged_out[elem_base .. elem_base + elems]);
        qweights[slot] = .{
            .data = qdata[weight_base .. weight_base + case.k * case.n],
            .scales = qscales[qscale_base .. qscale_base + scale_len],
            .rows = case.k,
            .cols = case.n,
            .block_size = block_size,
        };
    }

    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const be = metal.backend();
    var staged_policy = program_mod.CommandStreamPolicy.default();
    staged_policy.fuse_projection_row_chain = false;
    const staged_handle = metal.compileProgramWithCommandPolicy(program, staged_policy) orelse return error.CompileFailed;
    defer be.freeProgram(staged_handle);
    var grouped_policy = program_mod.CommandStreamPolicy.default();
    grouped_policy.fuse_projection_row_chain = true;
    grouped_policy.fuse_projection_row_chain_qmatvec = true;
    const grouped_handle = metal.compileProgramWithCommandPolicy(program, grouped_policy) orelse return error.CompileFailed;
    defer be.freeProgram(grouped_handle);
    const two_phase_policy = program_mod.CommandStreamPolicy.promptProjectionRowChainTwoPhaseCandidate();
    const two_phase_handle = metal.compileProgramWithCommandPolicy(program, two_phase_policy) orelse return error.CompileFailed;
    defer be.freeProgram(two_phase_handle);

    var staged_outputs: [n_slots]backend_mod.ProgramIO = undefined;
    var grouped_outputs: [n_slots]backend_mod.ProgramIO = undefined;
    var two_phase_outputs: [n_slots]backend_mod.ProgramIO = undefined;
    for (0..n_slots) |slot| {
        const out_buf: u16 = @intCast(6 + slot * 6);
        const base = slot * elems;
        staged_outputs[slot] = programIo(out_buf, staged_out[base .. base + elems]);
        grouped_outputs[slot] = programIo(out_buf, grouped_out[base .. base + elems]);
        two_phase_outputs[slot] = programIo(out_buf, two_phase_out[base .. base + elems]);
    }

    var staged_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = staged_handle,
        .out = staged_out,
        .output_io = &staged_outputs,
    };
    var grouped_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = grouped_handle,
        .out = grouped_out,
        .output_io = &grouped_outputs,
    };
    var two_phase_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = two_phase_handle,
        .out = two_phase_out,
        .output_io = &two_phase_outputs,
    };

    be.executeProgram(staged_handle, &.{}, &staged_outputs);
    be.executeProgram(grouped_handle, &.{}, &grouped_outputs);
    be.executeProgram(two_phase_handle, &.{}, &two_phase_outputs);
    const grouped_max_abs_diff = maxAbsDiff(staged_out, grouped_out);
    const two_phase_max_abs_diff = maxAbsDiff(staged_out, two_phase_out);

    const staged_stats = measure(io, &staged_bench);
    const grouped_stats = measure(io, &grouped_bench);
    const two_phase_stats = measure(io, &two_phase_bench);
    const approx_work = 2.0 * @as(f64, @floatFromInt(n_slots * case.m * case.n * case.k));

    var staged_name_buf: [112]u8 = undefined;
    const staged_name = try std.fmt.bufPrint(&staged_name_buf, "{s} staged", .{case.name});
    try printStats(w, staged_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", staged_stats);

    var grouped_name_buf: [112]u8 = undefined;
    const grouped_name = try std.fmt.bufPrint(&grouped_name_buf, "{s} projection_row_chain_group", .{case.name});
    try printStats(w, grouped_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", grouped_stats);

    var two_phase_name_buf: [128]u8 = undefined;
    const two_phase_name = try std.fmt.bufPrint(&two_phase_name_buf, "{s} projection_row_chain_two_phase_group", .{case.name});
    try printStats(w, two_phase_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", two_phase_stats);

    var ratio_name_buf: [112]u8 = undefined;
    const ratio_name = try std.fmt.bufPrint(&ratio_name_buf, "{s} projection_row_chain_group", .{case.name});
    try printRatio(w, ratio_name, staged_stats, grouped_stats, grouped_max_abs_diff);

    var two_phase_ratio_name_buf: [144]u8 = undefined;
    const two_phase_ratio_name = try std.fmt.bufPrint(&two_phase_ratio_name_buf, "{s} projection_row_chain_two_phase_group", .{case.name});
    try printRatio(w, two_phase_ratio_name, staged_stats, two_phase_stats, two_phase_max_abs_diff);

    const grouped_commands = try program_mod.buildProgramCommands(alloc, &ops, grouped_policy);
    defer alloc.free(grouped_commands);
    var profile_name_buf: [128]u8 = undefined;
    const profile_name = try std.fmt.bufPrint(&profile_name_buf, "{s} projection_row_chain_group dispatch_profile", .{case.name});
    try printCommandShape(w, profile_name, &ops, grouped_commands);
    try printProjectionRowChainRuntimeProfile(w, profile_name, be, grouped_handle, &grouped_outputs);

    const two_phase_commands = try program_mod.buildProgramCommands(alloc, &ops, two_phase_policy);
    defer alloc.free(two_phase_commands);
    var two_phase_profile_name_buf: [144]u8 = undefined;
    const two_phase_profile_name = try std.fmt.bufPrint(&two_phase_profile_name_buf, "{s} projection_row_chain_two_phase_group dispatch_profile", .{case.name});
    try printCommandShape(w, two_phase_profile_name, &ops, two_phase_commands);
    try printProjectionRowChainRuntimeProfile(w, two_phase_profile_name, be, two_phase_handle, &two_phase_outputs);
}

fn benchSemanticSublayerMetalCase(
    io: std.Io,
    alloc: std.mem.Allocator,
    w: *std.Io.Writer,
    metal: *internal.backend_metal.MetalBackend,
    case: ProjectionRowChainCase,
) !void {
    const hidden = case.hidden orelse case.n;
    const output = case.output orelse case.n;
    const hidden_elems = case.m * hidden;
    const output_elems = case.m * output;
    const input_len = case.m * case.k;
    const block_size: usize = 32;
    const gate_scale_len = (case.k * hidden + block_size - 1) / block_size;
    const down_scale_len = (hidden * output + block_size - 1) / block_size;

    const input = try allocF32(alloc, input_len, 901, 0.25);
    defer alloc.free(input);
    const shared_q = try allocF32(alloc, hidden_elems, 902, 0.0);
    defer alloc.free(shared_q);
    const silu_out = try allocF32(alloc, hidden_elems, 903, 0.0);
    defer alloc.free(silu_out);
    const product = try allocF32(alloc, hidden_elems, 904, 0.0);
    defer alloc.free(product);
    const down_q = try allocF32(alloc, output_elems, 905, 0.0);
    defer alloc.free(down_q);
    const residual = try allocF32(alloc, output_elems, 906, 0.20);
    defer alloc.free(residual);
    const norm = try allocF32(alloc, output_elems, 907, 0.0);
    defer alloc.free(norm);
    const scale = try allocF32(alloc, output, 908, 0.30);
    defer alloc.free(scale);
    const repeat = try allocF32(alloc, output_elems, 909, 0.0);
    defer alloc.free(repeat);
    const staged_out = try allocF32(alloc, output_elems, 910, 0.0);
    defer alloc.free(staged_out);
    const command_out = try allocF32(alloc, output_elems, 911, 0.0);
    defer alloc.free(command_out);
    const two_phase_out = try allocF32(alloc, output_elems, 912, 0.0);
    defer alloc.free(two_phase_out);
    const single_dispatch_out = try allocF32(alloc, output_elems, 920, 0.0);
    defer alloc.free(single_dispatch_out);
    const throughput_out = try allocF32(alloc, output_elems, 921, 0.0);
    defer alloc.free(throughput_out);
    const target_out = try allocF32(alloc, output_elems, 919, 0.0);
    defer alloc.free(target_out);

    const gate_qdata = try allocI8Weights(alloc, case.k * hidden, 913);
    defer alloc.free(gate_qdata);
    const up_qdata = try allocI8Weights(alloc, case.k * hidden, 914);
    defer alloc.free(up_qdata);
    const down_qdata = try allocI8Weights(alloc, hidden * output, 915);
    defer alloc.free(down_qdata);
    const gate_scales = try allocF32(alloc, gate_scale_len, 916, 0.02);
    defer alloc.free(gate_scales);
    const up_scales = try allocF32(alloc, gate_scale_len, 917, 0.02);
    defer alloc.free(up_scales);
    const down_scales = try allocF32(alloc, down_scale_len, 918, 0.02);
    defer alloc.free(down_scales);
    for (gate_scales) |*v| v.* = 0.02;
    for (up_scales) |*v| v.* = 0.02;
    for (down_scales) |*v| v.* = 0.02;

    const ops = [_]backend_mod.DeviceOp{
        .{ .qmatmul = .{ .dst = 1, .input = 0, .weight_idx = 0, .M = @intCast(case.m), .N = @intCast(hidden), .K = @intCast(case.k) } },
        .{ .elementwise = .{ .op = .silu, .dst = 2, .src0 = 1, .src1 = 1, .n = @intCast(hidden_elems) } },
        .{ .qmatmul = .{ .dst = 1, .input = 0, .weight_idx = 1, .M = @intCast(case.m), .N = @intCast(hidden), .K = @intCast(case.k) } },
        .{ .elementwise = .{ .op = .mul, .dst = 3, .src0 = 2, .src1 = 1, .n = @intCast(hidden_elems) } },
        .{ .qmatmul = .{ .dst = 4, .input = 3, .weight_idx = 2, .M = @intCast(case.m), .N = @intCast(output), .K = @intCast(hidden) } },
        .{ .elementwise = .{ .op = .add, .dst = 5, .src0 = 4, .src1 = 5, .n = @intCast(output_elems) } },
        .{ .rmsnorm = .{ .dst = 6, .src = 5, .rows = @intCast(case.m), .cols = @intCast(output), .eps = 1e-5 } },
        .{ .repeat = .{
            .dst = 8,
            .src = 7,
            .n = @intCast(output_elems),
            .src_ne = .{ @intCast(output), 1, 1, 1 },
            .dst_ne = .{ @intCast(output), @intCast(case.m), 1, 1 },
            .src_strides = .{ 1, @intCast(output), @intCast(output), @intCast(output) },
            .dst_strides = .{ 1, @intCast(output), @intCast(output_elems), @intCast(output_elems) },
        } },
        .{ .elementwise = .{ .op = .mul, .dst = 9, .src0 = 6, .src1 = 8, .n = @intCast(output_elems) } },
    };
    const buffer_sizes = [_]usize{ input_len, hidden_elems, hidden_elems, hidden_elems, output_elems, output_elems, output_elems, output, output_elems, output_elems };
    const uploads = [_]backend_mod.ProgramIO{
        programIo(0, input),
        programIo(1, shared_q),
        programIo(2, silu_out),
        programIo(3, product),
        programIo(4, down_q),
        programIo(5, residual),
        programIo(6, norm),
        programIo(7, scale),
        programIo(8, repeat),
        programIo(9, staged_out),
    };
    const qweights = [_]backend_mod.QuantizedWeightUpload{
        .{ .data = gate_qdata, .scales = gate_scales, .rows = case.k, .cols = hidden, .block_size = block_size },
        .{ .data = up_qdata, .scales = up_scales, .rows = case.k, .cols = hidden, .block_size = block_size },
        .{ .data = down_qdata, .scales = down_scales, .rows = hidden, .cols = output, .block_size = block_size },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const be = metal.backend();
    var staged_policy = program_mod.CommandStreamPolicy.default();
    staged_policy.fuse_projection_row_chain = false;
    const staged_handle = metal.compileProgramWithCommandPolicy(program, staged_policy) orelse return error.CompileFailed;
    defer be.freeProgram(staged_handle);
    const command_policy = program_mod.CommandStreamPolicy.promptProjectionRowChainCommand();
    const command_handle = metal.compileProgramWithCommandPolicy(program, command_policy) orelse return error.CompileFailed;
    defer be.freeProgram(command_handle);
    const two_phase_policy = program_mod.CommandStreamPolicy.promptProjectionRowChainTwoPhaseCandidate();
    const two_phase_handle = metal.compileProgramWithCommandPolicy(program, two_phase_policy) orelse return error.CompileFailed;
    defer be.freeProgram(two_phase_handle);
    const single_dispatch_policy = program_mod.CommandStreamPolicy.promptProjectionRowChainSingleDispatchCandidate();
    const single_dispatch_handle = metal.compileProgramWithCommandPolicy(program, single_dispatch_policy) orelse return error.CompileFailed;
    defer be.freeProgram(single_dispatch_handle);
    const throughput_policy = program_mod.CommandStreamPolicy.promptSemanticFfnSublayerThroughputCandidate();
    const throughput_handle = metal.compileProgramWithCommandPolicy(program, throughput_policy) orelse return error.CompileFailed;
    defer be.freeProgram(throughput_handle);
    const target_policy = program_mod.CommandStreamPolicy.promptSemanticFfnSublayerTarget();
    const target_handle = metal.compileProgramWithCommandPolicy(program, target_policy) orelse return error.CompileFailed;
    defer be.freeProgram(target_handle);

    const staged_output_io = [_]backend_mod.ProgramIO{programIo(9, staged_out)};
    const command_output_io = [_]backend_mod.ProgramIO{programIo(9, command_out)};
    const two_phase_output_io = [_]backend_mod.ProgramIO{programIo(9, two_phase_out)};
    const single_dispatch_output_io = [_]backend_mod.ProgramIO{programIo(9, single_dispatch_out)};
    const throughput_output_io = [_]backend_mod.ProgramIO{programIo(9, throughput_out)};
    const target_output_io = [_]backend_mod.ProgramIO{programIo(9, target_out)};
    var staged_bench = ProjectionRowChainMetalBench{ .be = be, .handle = staged_handle, .out = staged_out, .output_io = &staged_output_io };
    var command_bench = ProjectionRowChainMetalBench{ .be = be, .handle = command_handle, .out = command_out, .output_io = &command_output_io };
    var two_phase_bench = ProjectionRowChainMetalBench{ .be = be, .handle = two_phase_handle, .out = two_phase_out, .output_io = &two_phase_output_io };
    var single_dispatch_bench = ProjectionRowChainMetalBench{ .be = be, .handle = single_dispatch_handle, .out = single_dispatch_out, .output_io = &single_dispatch_output_io };
    var throughput_bench = ProjectionRowChainMetalBench{ .be = be, .handle = throughput_handle, .out = throughput_out, .output_io = &throughput_output_io };
    var target_bench = ProjectionRowChainMetalBench{ .be = be, .handle = target_handle, .out = target_out, .output_io = &target_output_io };

    be.executeProgram(staged_handle, &.{}, &staged_output_io);
    be.executeProgram(command_handle, &.{}, &command_output_io);
    be.executeProgram(two_phase_handle, &.{}, &two_phase_output_io);
    be.executeProgram(single_dispatch_handle, &.{}, &single_dispatch_output_io);
    be.executeProgram(throughput_handle, &.{}, &throughput_output_io);
    be.executeProgram(target_handle, &.{}, &target_output_io);
    const command_max_abs_diff = maxAbsDiff(staged_out, command_out);
    const two_phase_max_abs_diff = maxAbsDiff(staged_out, two_phase_out);
    const single_dispatch_max_abs_diff = maxAbsDiff(staged_out, single_dispatch_out);
    const throughput_max_abs_diff = maxAbsDiff(staged_out, throughput_out);
    const target_max_abs_diff = maxAbsDiff(staged_out, target_out);

    const variant_filter = SemanticVariantFilter.init();
    const target_only = variant_filter.enabled("target") and
        !variant_filter.enabled("command") and
        !variant_filter.enabled("two_phase") and
        !variant_filter.enabled("single_dispatch") and
        !variant_filter.enabled("throughput_candidate");
    const throughput_only = variant_filter.enabled("throughput_candidate") and
        !variant_filter.enabled("command") and
        !variant_filter.enabled("two_phase") and
        !variant_filter.enabled("single_dispatch") and
        !variant_filter.enabled("target");
    const staged_stats = measure(io, &staged_bench);
    const approx_work = 2.0 * @as(f64, @floatFromInt(case.m * hidden * case.k * 2 + case.m * output * hidden));

    var staged_name_buf: [128]u8 = undefined;
    const staged_name = try std.fmt.bufPrint(&staged_name_buf, "{s} semantic staged", .{case.name});
    try printStats(w, staged_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", staged_stats);
    if (target_only) {
        const target_stats = measure(io, &target_bench);
        var target_name_buf: [128]u8 = undefined;
        const target_name = try std.fmt.bufPrint(&target_name_buf, "{s} semantic target", .{case.name});
        try printStats(w, target_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", target_stats);
        var target_ratio_name_buf: [128]u8 = undefined;
        const target_ratio_name = try std.fmt.bufPrint(&target_ratio_name_buf, "{s} semantic target", .{case.name});
        try printRatio(w, target_ratio_name, staged_stats, target_stats, target_max_abs_diff);
        const target_commands = try program_mod.buildProgramCommands(alloc, &ops, target_policy);
        defer alloc.free(target_commands);
        var target_profile_name_buf: [160]u8 = undefined;
        const target_profile_name = try std.fmt.bufPrint(&target_profile_name_buf, "{s} semantic target dispatch_profile", .{case.name});
        try printCommandShape(w, target_profile_name, &ops, target_commands);
        try printSemanticSublayerRuntimeProfile(w, target_profile_name, be, target_handle, &target_output_io);
        return;
    }
    if (throughput_only) {
        const throughput_stats = measure(io, &throughput_bench);
        var throughput_name_buf: [160]u8 = undefined;
        const throughput_name = try std.fmt.bufPrint(&throughput_name_buf, "{s} semantic throughput_candidate", .{case.name});
        try printStats(w, throughput_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", throughput_stats);
        var throughput_ratio_name_buf: [160]u8 = undefined;
        const throughput_ratio_name = try std.fmt.bufPrint(&throughput_ratio_name_buf, "{s} semantic throughput_candidate", .{case.name});
        try printRatio(w, throughput_ratio_name, staged_stats, throughput_stats, throughput_max_abs_diff);
        const throughput_commands = try program_mod.buildProgramCommands(alloc, &ops, throughput_policy);
        defer alloc.free(throughput_commands);
        var throughput_profile_name_buf: [176]u8 = undefined;
        const throughput_profile_name = try std.fmt.bufPrint(&throughput_profile_name_buf, "{s} semantic throughput_candidate dispatch_profile", .{case.name});
        try printCommandShape(w, throughput_profile_name, &ops, throughput_commands);
        try printSemanticSublayerRuntimeProfile(w, throughput_profile_name, be, throughput_handle, &throughput_output_io);
        return;
    }

    const command_stats = measure(io, &command_bench);
    const two_phase_stats = measure(io, &two_phase_bench);
    const single_dispatch_stats = measure(io, &single_dispatch_bench);
    const throughput_stats = measure(io, &throughput_bench);
    const target_stats = measure(io, &target_bench);
    var command_name_buf: [128]u8 = undefined;
    const command_name = try std.fmt.bufPrint(&command_name_buf, "{s} semantic command", .{case.name});
    try printStats(w, command_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", command_stats);
    var two_phase_name_buf: [144]u8 = undefined;
    const two_phase_name = try std.fmt.bufPrint(&two_phase_name_buf, "{s} semantic pair_row_chain_two_phase", .{case.name});
    try printStats(w, two_phase_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", two_phase_stats);
    var single_dispatch_name_buf: [160]u8 = undefined;
    const single_dispatch_name = try std.fmt.bufPrint(&single_dispatch_name_buf, "{s} semantic pair_row_chain_single_dispatch", .{case.name});
    try printStats(w, single_dispatch_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", single_dispatch_stats);
    var throughput_name_buf: [160]u8 = undefined;
    const throughput_name = try std.fmt.bufPrint(&throughput_name_buf, "{s} semantic throughput_candidate", .{case.name});
    try printStats(w, throughput_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", throughput_stats);
    var target_name_buf: [128]u8 = undefined;
    const target_name = try std.fmt.bufPrint(&target_name_buf, "{s} semantic target", .{case.name});
    try printStats(w, target_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", target_stats);

    var command_ratio_name_buf: [128]u8 = undefined;
    const command_ratio_name = try std.fmt.bufPrint(&command_ratio_name_buf, "{s} semantic command", .{case.name});
    try printRatio(w, command_ratio_name, staged_stats, command_stats, command_max_abs_diff);
    var two_phase_ratio_name_buf: [144]u8 = undefined;
    const two_phase_ratio_name = try std.fmt.bufPrint(&two_phase_ratio_name_buf, "{s} semantic pair_row_chain_two_phase", .{case.name});
    try printRatio(w, two_phase_ratio_name, staged_stats, two_phase_stats, two_phase_max_abs_diff);
    var single_dispatch_ratio_name_buf: [160]u8 = undefined;
    const single_dispatch_ratio_name = try std.fmt.bufPrint(&single_dispatch_ratio_name_buf, "{s} semantic pair_row_chain_single_dispatch", .{case.name});
    try printRatio(w, single_dispatch_ratio_name, staged_stats, single_dispatch_stats, single_dispatch_max_abs_diff);
    var throughput_ratio_name_buf: [160]u8 = undefined;
    const throughput_ratio_name = try std.fmt.bufPrint(&throughput_ratio_name_buf, "{s} semantic throughput_candidate", .{case.name});
    try printRatio(w, throughput_ratio_name, staged_stats, throughput_stats, throughput_max_abs_diff);
    var target_ratio_name_buf: [128]u8 = undefined;
    const target_ratio_name = try std.fmt.bufPrint(&target_ratio_name_buf, "{s} semantic target", .{case.name});
    try printRatio(w, target_ratio_name, staged_stats, target_stats, target_max_abs_diff);

    const command_commands = try program_mod.buildProgramCommands(alloc, &ops, command_policy);
    defer alloc.free(command_commands);
    var profile_name_buf: [144]u8 = undefined;
    const profile_name = try std.fmt.bufPrint(&profile_name_buf, "{s} semantic command dispatch_profile", .{case.name});
    try printCommandShape(w, profile_name, &ops, command_commands);
    try printSemanticSublayerRuntimeProfile(w, profile_name, be, command_handle, &command_output_io);

    const two_phase_commands = try program_mod.buildProgramCommands(alloc, &ops, two_phase_policy);
    defer alloc.free(two_phase_commands);
    var two_phase_profile_name_buf: [160]u8 = undefined;
    const two_phase_profile_name = try std.fmt.bufPrint(&two_phase_profile_name_buf, "{s} semantic pair_row_chain_two_phase dispatch_profile", .{case.name});
    try printCommandShape(w, two_phase_profile_name, &ops, two_phase_commands);
    try printSemanticSublayerRuntimeProfile(w, two_phase_profile_name, be, two_phase_handle, &two_phase_output_io);

    const single_dispatch_commands = try program_mod.buildProgramCommands(alloc, &ops, single_dispatch_policy);
    defer alloc.free(single_dispatch_commands);
    var single_dispatch_profile_name_buf: [176]u8 = undefined;
    const single_dispatch_profile_name = try std.fmt.bufPrint(&single_dispatch_profile_name_buf, "{s} semantic pair_row_chain_single_dispatch dispatch_profile", .{case.name});
    try printCommandShape(w, single_dispatch_profile_name, &ops, single_dispatch_commands);
    try printSemanticSublayerRuntimeProfile(w, single_dispatch_profile_name, be, single_dispatch_handle, &single_dispatch_output_io);

    const throughput_commands = try program_mod.buildProgramCommands(alloc, &ops, throughput_policy);
    defer alloc.free(throughput_commands);
    var throughput_profile_name_buf: [176]u8 = undefined;
    const throughput_profile_name = try std.fmt.bufPrint(&throughput_profile_name_buf, "{s} semantic throughput_candidate dispatch_profile", .{case.name});
    try printCommandShape(w, throughput_profile_name, &ops, throughput_commands);
    try printSemanticSublayerRuntimeProfile(w, throughput_profile_name, be, throughput_handle, &throughput_output_io);

    const target_commands = try program_mod.buildProgramCommands(alloc, &ops, target_policy);
    defer alloc.free(target_commands);
    var target_profile_name_buf: [160]u8 = undefined;
    const target_profile_name = try std.fmt.bufPrint(&target_profile_name_buf, "{s} semantic target dispatch_profile", .{case.name});
    try printCommandShape(w, target_profile_name, &ops, target_commands);
    try printSemanticSublayerRuntimeProfile(w, target_profile_name, be, target_handle, &target_output_io);
}

fn benchSemanticSublayerWithInputRowChainMetalCase(
    io: std.Io,
    alloc: std.mem.Allocator,
    w: *std.Io.Writer,
    metal: *internal.backend_metal.MetalBackend,
    case: ProjectionRowChainCase,
) !void {
    const model = case.k;
    const hidden = case.hidden orelse case.n;
    const output = case.output orelse model;
    const model_elems = case.m * model;
    const hidden_elems = case.m * hidden;
    const output_elems = case.m * output;
    const block_size: usize = 32;
    const input_scale_len = (model * model + block_size - 1) / block_size;
    const gate_scale_len = (model * hidden + block_size - 1) / block_size;
    const down_scale_len = (hidden * output + block_size - 1) / block_size;

    const input = try allocF32(alloc, model_elems, 1001, 0.25);
    defer alloc.free(input);
    const input_q = try allocF32(alloc, model_elems, 1002, 0.0);
    defer alloc.free(input_q);
    const input_residual_out = try allocF32(alloc, model_elems, 1003, 0.0);
    defer alloc.free(input_residual_out);
    const input_norm = try allocF32(alloc, model_elems, 1004, 0.0);
    defer alloc.free(input_norm);
    const input_repeat = try allocF32(alloc, model_elems, 1005, 0.0);
    defer alloc.free(input_repeat);
    const ffn_input = try allocF32(alloc, model_elems, 1006, 0.0);
    defer alloc.free(ffn_input);
    const shared_q = try allocF32(alloc, hidden_elems, 1007, 0.0);
    defer alloc.free(shared_q);
    const silu_out = try allocF32(alloc, hidden_elems, 1008, 0.0);
    defer alloc.free(silu_out);
    const product = try allocF32(alloc, hidden_elems, 1009, 0.0);
    defer alloc.free(product);
    const down_q = try allocF32(alloc, output_elems, 1010, 0.0);
    defer alloc.free(down_q);
    const input_residual = try allocF32(alloc, model_elems, 1011, 0.20);
    defer alloc.free(input_residual);
    const input_scale = try allocF32(alloc, model, 1012, 0.30);
    defer alloc.free(input_scale);
    const ffn_residual_out = try allocF32(alloc, output_elems, 1013, 0.0);
    defer alloc.free(ffn_residual_out);
    const output_norm = try allocF32(alloc, output_elems, 1014, 0.0);
    defer alloc.free(output_norm);
    const output_repeat = try allocF32(alloc, output_elems, 1015, 0.0);
    defer alloc.free(output_repeat);
    const output_scale = try allocF32(alloc, output, 1016, 0.30);
    defer alloc.free(output_scale);
    const staged_out = try allocF32(alloc, output_elems, 1017, 0.0);
    defer alloc.free(staged_out);
    const command_out = try allocF32(alloc, output_elems, 1018, 0.0);
    defer alloc.free(command_out);
    const absorbed_out = try allocF32(alloc, output_elems, 1019, 0.0);
    defer alloc.free(absorbed_out);

    const input_qdata = try allocI8Weights(alloc, model * model, 1020);
    defer alloc.free(input_qdata);
    const gate_qdata = try allocI8Weights(alloc, model * hidden, 1021);
    defer alloc.free(gate_qdata);
    const up_qdata = try allocI8Weights(alloc, model * hidden, 1022);
    defer alloc.free(up_qdata);
    const down_qdata = try allocI8Weights(alloc, hidden * output, 1023);
    defer alloc.free(down_qdata);
    const input_scales = try allocF32(alloc, input_scale_len, 1024, 0.02);
    defer alloc.free(input_scales);
    const gate_scales = try allocF32(alloc, gate_scale_len, 1025, 0.02);
    defer alloc.free(gate_scales);
    const up_scales = try allocF32(alloc, gate_scale_len, 1026, 0.02);
    defer alloc.free(up_scales);
    const down_scales = try allocF32(alloc, down_scale_len, 1027, 0.02);
    defer alloc.free(down_scales);
    for (input_scales) |*v| v.* = 0.02;
    for (gate_scales) |*v| v.* = 0.02;
    for (up_scales) |*v| v.* = 0.02;
    for (down_scales) |*v| v.* = 0.02;

    const ops = [_]backend_mod.DeviceOp{
        .{ .qmatmul = .{ .dst = 1, .input = 0, .weight_idx = 0, .M = @intCast(case.m), .N = @intCast(model), .K = @intCast(model) } },
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 10, .n = @intCast(model_elems) } },
        .{ .rmsnorm = .{ .dst = 3, .src = 2, .rows = @intCast(case.m), .cols = @intCast(model), .eps = 1e-5 } },
        .{ .repeat = .{
            .dst = 4,
            .src = 11,
            .n = @intCast(model_elems),
            .src_ne = .{ @intCast(model), 1, 1, 1 },
            .dst_ne = .{ @intCast(model), @intCast(case.m), 1, 1 },
            .src_strides = .{ 1, @intCast(model), @intCast(model), @intCast(model) },
            .dst_strides = .{ 1, @intCast(model), @intCast(model_elems), @intCast(model_elems) },
        } },
        .{ .elementwise = .{ .op = .mul, .dst = 5, .src0 = 3, .src1 = 4, .n = @intCast(model_elems) } },
        .{ .qmatmul = .{ .dst = 6, .input = 5, .weight_idx = 1, .M = @intCast(case.m), .N = @intCast(hidden), .K = @intCast(model) } },
        .{ .elementwise = .{ .op = .silu, .dst = 7, .src0 = 6, .src1 = 6, .n = @intCast(hidden_elems) } },
        .{ .qmatmul = .{ .dst = 6, .input = 5, .weight_idx = 2, .M = @intCast(case.m), .N = @intCast(hidden), .K = @intCast(model) } },
        .{ .elementwise = .{ .op = .mul, .dst = 8, .src0 = 7, .src1 = 6, .n = @intCast(hidden_elems) } },
        .{ .qmatmul = .{ .dst = 9, .input = 8, .weight_idx = 3, .M = @intCast(case.m), .N = @intCast(output), .K = @intCast(hidden) } },
        .{ .elementwise = .{ .op = .add, .dst = 12, .src0 = 9, .src1 = 2, .n = @intCast(output_elems) } },
        .{ .rmsnorm = .{ .dst = 13, .src = 12, .rows = @intCast(case.m), .cols = @intCast(output), .eps = 1e-5 } },
        .{ .repeat = .{
            .dst = 14,
            .src = 15,
            .n = @intCast(output_elems),
            .src_ne = .{ @intCast(output), 1, 1, 1 },
            .dst_ne = .{ @intCast(output), @intCast(case.m), 1, 1 },
            .src_strides = .{ 1, @intCast(output), @intCast(output), @intCast(output) },
            .dst_strides = .{ 1, @intCast(output), @intCast(output_elems), @intCast(output_elems) },
        } },
        .{ .elementwise = .{ .op = .mul, .dst = 16, .src0 = 13, .src1 = 14, .n = @intCast(output_elems) } },
    };
    const buffer_sizes = [_]usize{
        model_elems,  model_elems,  model_elems,  model_elems,  model_elems,  model_elems,
        hidden_elems, hidden_elems, hidden_elems, output_elems, model_elems,  model,
        output_elems, output_elems, output_elems, output,       output_elems,
    };
    const uploads = [_]backend_mod.ProgramIO{
        programIo(0, input),
        programIo(1, input_q),
        programIo(2, input_residual_out),
        programIo(3, input_norm),
        programIo(4, input_repeat),
        programIo(5, ffn_input),
        programIo(6, shared_q),
        programIo(7, silu_out),
        programIo(8, product),
        programIo(9, down_q),
        programIo(10, input_residual),
        programIo(11, input_scale),
        programIo(12, ffn_residual_out),
        programIo(13, output_norm),
        programIo(14, output_repeat),
        programIo(15, output_scale),
        programIo(16, staged_out),
    };
    const qweights = [_]backend_mod.QuantizedWeightUpload{
        .{ .data = input_qdata, .scales = input_scales, .rows = model, .cols = model, .block_size = block_size },
        .{ .data = gate_qdata, .scales = gate_scales, .rows = model, .cols = hidden, .block_size = block_size },
        .{ .data = up_qdata, .scales = up_scales, .rows = model, .cols = hidden, .block_size = block_size },
        .{ .data = down_qdata, .scales = down_scales, .rows = hidden, .cols = output, .block_size = block_size },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const be = metal.backend();
    var staged_policy = program_mod.CommandStreamPolicy.default();
    staged_policy.fuse_projection_row_chain = false;
    staged_policy.fuse_semantic_ffn_sublayer = false;
    staged_policy.fuse_semantic_ffn_sublayer_input_row_chain = false;
    const staged_handle = metal.compileProgramWithCommandPolicy(program, staged_policy) orelse return error.CompileFailed;
    defer be.freeProgram(staged_handle);
    const command_policy = program_mod.CommandStreamPolicy.promptProjectionRowChainCommand();
    const command_handle = metal.compileProgramWithCommandPolicy(program, command_policy) orelse return error.CompileFailed;
    defer be.freeProgram(command_handle);
    const absorbed_policy = program_mod.CommandStreamPolicy.promptSemanticFfnSublayerThroughputCandidate();
    const absorbed_handle = metal.compileProgramWithCommandPolicy(program, absorbed_policy) orelse return error.CompileFailed;
    defer be.freeProgram(absorbed_handle);

    const staged_output_io = [_]backend_mod.ProgramIO{programIo(16, staged_out)};
    const command_output_io = [_]backend_mod.ProgramIO{programIo(16, command_out)};
    const absorbed_output_io = [_]backend_mod.ProgramIO{programIo(16, absorbed_out)};
    var staged_bench = ProjectionRowChainMetalBench{ .be = be, .handle = staged_handle, .out = staged_out, .output_io = &staged_output_io };
    var command_bench = ProjectionRowChainMetalBench{ .be = be, .handle = command_handle, .out = command_out, .output_io = &command_output_io };
    var absorbed_bench = ProjectionRowChainMetalBench{ .be = be, .handle = absorbed_handle, .out = absorbed_out, .output_io = &absorbed_output_io };

    be.executeProgram(staged_handle, &.{}, &staged_output_io);
    be.executeProgram(command_handle, &.{}, &command_output_io);
    be.executeProgram(absorbed_handle, &.{}, &absorbed_output_io);
    const command_max_abs_diff = maxAbsDiff(staged_out, command_out);
    const absorbed_max_abs_diff = maxAbsDiff(staged_out, absorbed_out);

    const staged_stats = measure(io, &staged_bench);
    const command_stats = measure(io, &command_bench);
    const absorbed_stats = measure(io, &absorbed_bench);
    const approx_work = 2.0 * @as(f64, @floatFromInt(case.m * model * model + case.m * hidden * model * 2 + case.m * output * hidden));

    var staged_name_buf: [160]u8 = undefined;
    const staged_name = try std.fmt.bufPrint(&staged_name_buf, "{s} semantic input staged", .{case.name});
    try printStats(w, staged_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", staged_stats);
    var command_name_buf: [176]u8 = undefined;
    const command_name = try std.fmt.bufPrint(&command_name_buf, "{s} semantic input command", .{case.name});
    try printStats(w, command_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", command_stats);
    var absorbed_name_buf: [176]u8 = undefined;
    const absorbed_name = try std.fmt.bufPrint(&absorbed_name_buf, "{s} semantic input absorbed", .{case.name});
    try printStats(w, absorbed_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", absorbed_stats);

    try printRatio(w, command_name, staged_stats, command_stats, command_max_abs_diff);
    try printRatio(w, absorbed_name, staged_stats, absorbed_stats, absorbed_max_abs_diff);

    const command_commands = try program_mod.buildProgramCommands(alloc, &ops, command_policy);
    defer alloc.free(command_commands);
    var command_profile_name_buf: [192]u8 = undefined;
    const command_profile_name = try std.fmt.bufPrint(&command_profile_name_buf, "{s} semantic input command dispatch_profile", .{case.name});
    try printCommandShape(w, command_profile_name, &ops, command_commands);
    try printSemanticSublayerRuntimeProfile(w, command_profile_name, be, command_handle, &command_output_io);

    const absorbed_commands = try program_mod.buildProgramCommands(alloc, &ops, absorbed_policy);
    defer alloc.free(absorbed_commands);
    var absorbed_profile_name_buf: [192]u8 = undefined;
    const absorbed_profile_name = try std.fmt.bufPrint(&absorbed_profile_name_buf, "{s} semantic input absorbed dispatch_profile", .{case.name});
    try printCommandShape(w, absorbed_profile_name, &ops, absorbed_commands);
    try printSemanticSublayerRuntimeProfile(w, absorbed_profile_name, be, absorbed_handle, &absorbed_output_io);
}

fn benchProjectionRowChainMetal(io: std.Io, alloc: std.mem.Allocator, w: *std.Io.Writer, filter: FrontierFilter) !void {
    if (!filter.matchesAny(&.{
        "Metal",
        "qproj",
        "qsemantic",
        "qproj region",
        "qrow",
        "projection_chain",
        "projection_group",
        "projection_group_region",
        "projection_row_chain",
        "qrow region",
        "semantic command",
        "semantic pair_row_chain",
        "qsemantic bridge",
        "semantic bridge",
        "qsemantic input bridge",
        "semantic input",
        "qsemantic full-prefill",
        "qsemantic smollm-prompt",
        "projection_row_chain_two_phase_group",
    })) return;
    try w.print("\nMetal Projection Row-Chain Command\n", .{});
    try w.print("----------------------------------\n", .{});

    if (@import("builtin").os.tag != .macos or !opts.use_metal) {
        try w.print("  projection_row_chain metal unavailable\n", .{});
        return;
    }

    var metal = internal.backend_metal.MetalBackend.initWithAllocator(alloc) catch |err| switch (err) {
        error.MetalNotAvailable => {
            try w.print("  projection_row_chain metal unavailable\n", .{});
            return;
        },
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);

    const projection_chain_cases = [_]ProjectionRowChainCase{
        .{ .name = "qproj prompt m=32 n=512 k=512", .m = 32, .n = 512, .k = 512 },
        .{ .name = "qproj full-prefill m=128 n=512 k=512", .m = 128, .n = 512, .k = 512 },
        .{ .name = "qproj smollm-prompt m=128 n=576 k=576", .m = 128, .n = 576, .k = 576 },
    };
    for (projection_chain_cases) |case| {
        if (filter.matchesAny(&.{ case.name, "qproj", "projection_chain" })) {
            try benchProjectionChainMetalCase(io, alloc, w, &metal, case);
        }
    }

    const projection_group_cases = [_]ProjectionRowChainCase{
        .{ .name = "qproj group full-prefill x4 m=128 n=512 k=512", .m = 128, .n = 512, .k = 512 },
        .{ .name = "qproj group smollm-prompt x4 m=128 n=576 k=576", .m = 128, .n = 576, .k = 576 },
    };
    for (projection_group_cases) |case| {
        if (filter.matchesAny(&.{ case.name, "qproj group", "projection_group" })) {
            try benchProjectionGroupMetalCase(4, io, alloc, w, &metal, case);
        }
    }

    const projection_group_region_cases = [_]ProjectionRowChainCase{
        .{ .name = "qproj region full-prefill x7 m=128 n=512 k=512", .m = 128, .n = 512, .k = 512 },
        .{ .name = "qproj region smollm-prompt x7 m=128 n=576 k=576", .m = 128, .n = 576, .k = 576 },
    };
    for (projection_group_region_cases) |case| {
        if (filter.matchesAny(&.{ case.name, "qproj region", "projection_group_region" })) {
            try benchProjectionGroupMetalCase(7, io, alloc, w, &metal, case);
        }
    }

    const semantic_sublayer_cases = [_]ProjectionRowChainCase{
        .{ .name = "qsemantic full-prefill m=128 n=512 k=512", .m = 128, .n = 512, .k = 512 },
        .{ .name = "qsemantic smollm-prompt m=128 n=576 k=576", .m = 128, .n = 576, .k = 576 },
        .{ .name = "qsemantic bridge-ffn m=128 h=1536 k=576 o=576", .m = 128, .n = 1536, .k = 576, .output = 576 },
    };
    for (semantic_sublayer_cases) |case| {
        const bridge_case = case.output != null;
        const should_run = if (bridge_case)
            filter.matchesAny(&.{ case.name, "qsemantic bridge", "semantic bridge" })
        else
            filter.matchesAny(&.{ case.name, "qsemantic", "semantic command", "semantic pair_row_chain" });
        if (should_run) {
            try benchSemanticSublayerMetalCase(io, alloc, w, &metal, case);
        }
    }

    const semantic_with_input_cases = [_]ProjectionRowChainCase{
        .{ .name = "qsemantic input-bridge m=128 h=1536 k=576 o=576", .m = 128, .n = 1536, .k = 576, .output = 576 },
    };
    for (semantic_with_input_cases) |case| {
        if (filter.matchesAny(&.{ case.name, "qsemantic input bridge", "semantic input" })) {
            try benchSemanticSublayerWithInputRowChainMetalCase(io, alloc, w, &metal, case);
        }
    }

    const row_chain_group_cases = [_]ProjectionRowChainCase{
        .{ .name = "qrow group full-prefill x4 m=128 n=512 k=512", .m = 128, .n = 512, .k = 512 },
        .{ .name = "qrow group smollm-prompt x4 m=128 n=576 k=576", .m = 128, .n = 576, .k = 576 },
    };
    for (row_chain_group_cases) |case| {
        if (filter.matchesAny(&.{ case.name, "qrow group", "projection_row_chain_group" })) {
            try benchProjectionRowChainGroupMetalCase(4, io, alloc, w, &metal, case);
        }
    }

    const row_chain_region_cases = [_]ProjectionRowChainCase{
        .{ .name = "qrow region full-prefill x7 m=128 n=512 k=512", .m = 128, .n = 512, .k = 512 },
        .{ .name = "qrow region smollm-prompt x7 m=128 n=576 k=576", .m = 128, .n = 576, .k = 576 },
    };
    for (row_chain_region_cases) |case| {
        if (filter.matchesAny(&.{ case.name, "qrow region", "projection_row_chain_two_phase_group" })) {
            try benchProjectionRowChainGroupMetalCase(7, io, alloc, w, &metal, case);
        }
    }

    const cases = [_]ProjectionRowChainCase{
        .{ .name = "qrow decode m=1 n=512 k=512", .m = 1, .n = 512, .k = 512 },
        .{ .name = "qrow tiny m=2 n=128 k=128", .m = 2, .n = 128, .k = 128 },
        .{ .name = "qrow prompt m=32 n=512 k=512", .m = 32, .n = 512, .k = 512 },
        .{ .name = "qrow full-prefill m=128 n=512 k=512", .m = 128, .n = 512, .k = 512 },
        .{ .name = "qrow smollm-prompt m=128 n=576 k=576", .m = 128, .n = 576, .k = 576 },
    };
    for (cases) |case| {
        if (filter.matchesAny(&.{ case.name, "qrow", "projection_row_chain" })) {
            try benchProjectionRowChainMetalCase(io, alloc, w, &metal, case);
        }
    }
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    const stdout_file = std.Io.File.stdout();
    var buf: [16 * 1024]u8 = undefined;
    var writer = stdout_file.writer(io, &buf);
    const w = &writer.interface;

    try w.print("\nzgml benchmark frontier", .{});
    if (opts.use_blas) try w.print(" [BLAS enabled]", .{});
    try w.print("\n=======================\n", .{});
    try w.print("samples={d}, min_sample={d:.1} ms, adaptive repeats, ReleaseFast build step\n", .{
        SampleCount,
        @as(f64, @floatFromInt(MinSampleNs)) / 1_000_000.0,
    });
    const filter = FrontierFilter.init();
    if (filter.query) |query| {
        try w.print("filter={s}\n", .{query});
    }

    try benchElementwise(io, alloc, w, filter);
    try benchMatmul(io, alloc, w, filter);
    try benchNorms(io, alloc, w, filter);
    try benchDecodeGraph(io, alloc, w, filter);
    try benchProjectionRowChainMetal(io, alloc, w, filter);

    try w.print("\n", .{});
    writer.interface.flush() catch {};
}
