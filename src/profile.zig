//! Profiling utilities for backend placement and runtime evidence.

const std = @import("std");
const backend = @import("backend.zig");
const program_mod = @import("backend/program.zig");

const op_fields = @typeInfo(backend.DeviceOp).@"union".fields;
const n_op_tags = op_fields.len;
const n_program_command_kinds = @typeInfo(program_mod.ProgramCommandKind).@"enum".fields.len;
const max_schedule_region_patterns = 16;

const ScheduleRegionStats = struct {
    attempted: u64 = 0,
    lowered: u64 = 0,
    failed: u64 = 0,
    attempted_ops: u64 = 0,
    lowered_ops: u64 = 0,
    failed_ops: u64 = 0,

    fn recordAttempt(self: *ScheduleRegionStats, op_count: u32) void {
        self.attempted +%= 1;
        self.attempted_ops +%= op_count;
    }

    fn recordLowered(self: *ScheduleRegionStats, op_count: u32) void {
        self.lowered +%= 1;
        self.lowered_ops +%= op_count;
    }

    fn recordFailed(self: *ScheduleRegionStats, op_count: u32) void {
        self.failed +%= 1;
        self.failed_ops +%= op_count;
    }

    fn add(self: *ScheduleRegionStats, other: ScheduleRegionStats) void {
        self.attempted +%= other.attempted;
        self.lowered +%= other.lowered;
        self.failed +%= other.failed;
        self.attempted_ops +%= other.attempted_ops;
        self.lowered_ops +%= other.lowered_ops;
        self.failed_ops +%= other.failed_ops;
    }
};

const tag_names = blk: {
    var names: [n_op_tags][]const u8 = undefined;
    for (op_fields, 0..) |field, i| names[i] = field.name;
    break :blk names;
};

// ── Runtime profiling ──────────────────────────────────────────────

/// Accumulated backend placement and command evidence. Caller resets explicitly.
pub const RuntimeProfile = struct {
    program_command_counts: [n_program_command_kinds]u64 = [_]u64{0} ** n_program_command_kinds,
    program_command_dispatch_counts: [n_program_command_kinds]u64 = [_]u64{0} ** n_program_command_kinds,
    program_command_attempt_counts: [n_program_command_kinds]u64 = [_]u64{0} ** n_program_command_kinds,
    program_command_failed_counts: [n_program_command_kinds]u64 = [_]u64{0} ** n_program_command_kinds,
    program_op_command_counts: [n_op_tags]u64 = [_]u64{0} ** n_op_tags,
    projection_chain_qmatvec_sidecars: [n_op_tags]u64 = [_]u64{0} ** n_op_tags,
    projection_chain_qmatmul_sidecars: [n_op_tags]u64 = [_]u64{0} ** n_op_tags,
    schedule_regions: ScheduleRegionStats = .{},
    schedule_region_patterns: [max_schedule_region_patterns]ScheduleRegionStats = [_]ScheduleRegionStats{.{}} ** max_schedule_region_patterns,
    region_command_plan_cached_count: u64 = 0,
    region_command_plan_cached_command_count: u64 = 0,
    region_command_plan_dynamic_count: u64 = 0,
    backend_op_count: u64 = 0,
    fallback_op_count: u64 = 0,
    backend_dispatch_count: u64 = 0,
    sync_count: u64 = 0,
    runtime_patch_call_count: u64 = 0,
    runtime_patch_changed_count: u64 = 0,
    runtime_patch_invalid_count: u64 = 0,
    runtime_patch_shape: backend.RuntimePatchShape = .{},
    program_command_shape: program_mod.ProgramCommandStreamShape = .{},
    qmatmul_row_chain_tiled_count: u64 = 0,
    qmatmul_row_chain_tiled_row_tile_groups: u64 = 0,
    qmatmul_row_chain_tiled_n_tiles: u64 = 0,
    qmatmul_row_chain_tiled_serial_tile_loops: u64 = 0,
    qmatmul_row_chain_tiled_partial_slots: u64 = 0,
    qmatmul_row_chain_tiled_scratch_capacity: u64 = 0,
    qmatmul_row_chain_tiled_spilled_elementwise: u64 = 0,
    qmatmul_row_chain_tiled_spilled_input: u64 = 0,
    qmatmul_row_chain_tiled_output_spills: u64 = 0,
    qmatmul_row_chain_tiled_two_phase_count: u64 = 0,
    qmatmul_row_chain_tiled_finalize_tile_groups: u64 = 0,
    qmatmul_row_chain_tiled_finalize_elements: u64 = 0,
    semantic_ffn_sublayer_count: u64 = 0,
    semantic_ffn_sublayer_rows: u64 = 0,
    semantic_ffn_sublayer_hidden: u64 = 0,
    semantic_ffn_sublayer_input: u64 = 0,
    semantic_ffn_sublayer_output: u64 = 0,
    semantic_ffn_sublayer_row_serial_dot_ops: u64 = 0,
    semantic_ffn_sublayer_total_row_serial_dot_ops: u64 = 0,
    semantic_ffn_sublayer_tile_row_groups: u64 = 0,
    semantic_ffn_sublayer_tile_hidden_tiles: u64 = 0,
    semantic_ffn_sublayer_tile_output_tiles: u64 = 0,
    semantic_ffn_sublayer_tile_parallel_groups: u64 = 0,
    call_count: u32 = 0,

    pub fn reset(self: *RuntimeProfile) void {
        const runtime_patch_shape = self.runtime_patch_shape;
        const program_command_shape = self.program_command_shape;
        self.* = .{
            .runtime_patch_shape = runtime_patch_shape,
            .program_command_shape = program_command_shape,
        };
    }

    pub fn add(self: *RuntimeProfile, other: RuntimeProfile) void {
        for (&self.program_command_counts, other.program_command_counts) |*dst, value| dst.* +%= value;
        for (&self.program_command_dispatch_counts, other.program_command_dispatch_counts) |*dst, value| dst.* +%= value;
        for (&self.program_command_attempt_counts, other.program_command_attempt_counts) |*dst, value| dst.* +%= value;
        for (&self.program_command_failed_counts, other.program_command_failed_counts) |*dst, value| dst.* +%= value;
        for (&self.program_op_command_counts, other.program_op_command_counts) |*dst, value| dst.* +%= value;
        for (&self.projection_chain_qmatvec_sidecars, other.projection_chain_qmatvec_sidecars) |*dst, value| dst.* +%= value;
        for (&self.projection_chain_qmatmul_sidecars, other.projection_chain_qmatmul_sidecars) |*dst, value| dst.* +%= value;
        self.schedule_regions.add(other.schedule_regions);
        for (&self.schedule_region_patterns, other.schedule_region_patterns) |*dst, value| dst.add(value);
        self.region_command_plan_cached_count +%= other.region_command_plan_cached_count;
        self.region_command_plan_cached_command_count +%= other.region_command_plan_cached_command_count;
        self.region_command_plan_dynamic_count +%= other.region_command_plan_dynamic_count;
        self.backend_op_count +%= other.backend_op_count;
        self.fallback_op_count +%= other.fallback_op_count;
        self.backend_dispatch_count +%= other.backend_dispatch_count;
        self.sync_count +%= other.sync_count;
        self.runtime_patch_call_count +%= other.runtime_patch_call_count;
        self.runtime_patch_changed_count +%= other.runtime_patch_changed_count;
        self.runtime_patch_invalid_count +%= other.runtime_patch_invalid_count;
        self.runtime_patch_shape = self.runtime_patch_shape.merge(other.runtime_patch_shape);
        self.program_command_shape = self.program_command_shape.merge(other.program_command_shape);
        self.qmatmul_row_chain_tiled_count +%= other.qmatmul_row_chain_tiled_count;
        self.qmatmul_row_chain_tiled_row_tile_groups +%= other.qmatmul_row_chain_tiled_row_tile_groups;
        self.qmatmul_row_chain_tiled_n_tiles +%= other.qmatmul_row_chain_tiled_n_tiles;
        self.qmatmul_row_chain_tiled_serial_tile_loops +%= other.qmatmul_row_chain_tiled_serial_tile_loops;
        self.qmatmul_row_chain_tiled_partial_slots +%= other.qmatmul_row_chain_tiled_partial_slots;
        self.qmatmul_row_chain_tiled_scratch_capacity +%= other.qmatmul_row_chain_tiled_scratch_capacity;
        self.qmatmul_row_chain_tiled_spilled_elementwise +%= other.qmatmul_row_chain_tiled_spilled_elementwise;
        self.qmatmul_row_chain_tiled_spilled_input +%= other.qmatmul_row_chain_tiled_spilled_input;
        self.qmatmul_row_chain_tiled_output_spills +%= other.qmatmul_row_chain_tiled_output_spills;
        self.qmatmul_row_chain_tiled_two_phase_count +%= other.qmatmul_row_chain_tiled_two_phase_count;
        self.qmatmul_row_chain_tiled_finalize_tile_groups +%= other.qmatmul_row_chain_tiled_finalize_tile_groups;
        self.qmatmul_row_chain_tiled_finalize_elements +%= other.qmatmul_row_chain_tiled_finalize_elements;
        self.semantic_ffn_sublayer_count +%= other.semantic_ffn_sublayer_count;
        self.semantic_ffn_sublayer_rows +%= other.semantic_ffn_sublayer_rows;
        self.semantic_ffn_sublayer_hidden +%= other.semantic_ffn_sublayer_hidden;
        self.semantic_ffn_sublayer_input +%= other.semantic_ffn_sublayer_input;
        self.semantic_ffn_sublayer_output +%= other.semantic_ffn_sublayer_output;
        self.semantic_ffn_sublayer_row_serial_dot_ops +%= other.semantic_ffn_sublayer_row_serial_dot_ops;
        self.semantic_ffn_sublayer_total_row_serial_dot_ops +%= other.semantic_ffn_sublayer_total_row_serial_dot_ops;
        self.semantic_ffn_sublayer_tile_row_groups +%= other.semantic_ffn_sublayer_tile_row_groups;
        self.semantic_ffn_sublayer_tile_hidden_tiles +%= other.semantic_ffn_sublayer_tile_hidden_tiles;
        self.semantic_ffn_sublayer_tile_output_tiles +%= other.semantic_ffn_sublayer_tile_output_tiles;
        self.semantic_ffn_sublayer_tile_parallel_groups +%= other.semantic_ffn_sublayer_tile_parallel_groups;
        self.call_count +%= other.call_count;
    }

    pub fn recordProgramCommand(self: *RuntimeProfile, kind: program_mod.ProgramCommandKind) void {
        self.program_command_counts[@intFromEnum(kind)] +%= 1;
    }

    pub fn recordProgramCommandDispatch(self: *RuntimeProfile, kind: program_mod.ProgramCommandKind) void {
        self.program_command_dispatch_counts[@intFromEnum(kind)] +%= 1;
    }

    pub fn recordProgramCommandAttempt(self: *RuntimeProfile, kind: program_mod.ProgramCommandKind) void {
        self.program_command_attempt_counts[@intFromEnum(kind)] +%= 1;
    }

    pub fn recordProgramCommandFailed(self: *RuntimeProfile, kind: program_mod.ProgramCommandKind) void {
        self.program_command_failed_counts[@intFromEnum(kind)] +%= 1;
    }

    pub fn recordProgramCommandShapeDispatch(self: *RuntimeProfile, shape: program_mod.ProgramCommandStreamShape) void {
        for (
            &self.program_command_counts,
            &self.program_command_dispatch_counts,
            &self.program_command_attempt_counts,
            shape.command_kind_counts,
        ) |*encoded, *dispatched, *attempted, count| {
            const value: u64 = @intCast(count);
            encoded.* +%= value;
            dispatched.* +%= value;
            attempted.* +%= value;
        }
        self.backend_dispatch_count +%= @intCast(shape.command_count);
    }

    pub fn recordProgramOpCommand(self: *RuntimeProfile, op: backend.DeviceOp) void {
        self.program_op_command_counts[@intFromEnum(op)] +%= 1;
    }

    pub fn recordProjectionChainSidecar(self: *RuntimeProfile, is_qmatvec: bool, sidecar: backend.DeviceOp) void {
        const idx = @intFromEnum(sidecar);
        if (is_qmatvec) {
            self.projection_chain_qmatvec_sidecars[idx] +%= 1;
        } else {
            self.projection_chain_qmatmul_sidecars[idx] +%= 1;
        }
    }

    pub fn recordScheduleRegionAttempt(self: *RuntimeProfile, unit: program_mod.ScheduleUnit) void {
        self.schedule_regions.recordAttempt(unit.op_count);
        if (scheduleRegionPatternSlot(unit)) |slot| self.schedule_region_patterns[slot].recordAttempt(unit.op_count);
    }

    pub fn recordScheduleRegionLowered(self: *RuntimeProfile, unit: program_mod.ScheduleUnit) void {
        self.schedule_regions.recordLowered(unit.op_count);
        if (scheduleRegionPatternSlot(unit)) |slot| self.schedule_region_patterns[slot].recordLowered(unit.op_count);
    }

    pub fn recordScheduleRegionFailed(self: *RuntimeProfile, unit: program_mod.ScheduleUnit) void {
        self.schedule_regions.recordFailed(unit.op_count);
        if (scheduleRegionPatternSlot(unit)) |slot| self.schedule_region_patterns[slot].recordFailed(unit.op_count);
    }

    pub fn recordCachedRegionCommandPlan(self: *RuntimeProfile, command_count: usize) void {
        self.region_command_plan_cached_count +%= 1;
        self.region_command_plan_cached_command_count +%= command_count;
    }

    pub fn recordDynamicRegionCommandPlan(self: *RuntimeProfile) void {
        self.region_command_plan_dynamic_count +%= 1;
    }

    pub fn recordRuntimePatch(self: *RuntimeProfile, status: backend.RuntimePatchStatus) void {
        self.runtime_patch_call_count +%= 1;
        switch (status) {
            .changed => self.runtime_patch_changed_count +%= 1,
            .invalid => self.runtime_patch_invalid_count +%= 1,
            .unchanged => {},
        }
    }

    pub fn recordQMatmulRowChainTiled(self: *RuntimeProfile, m: u32, n: u32, tile: u32, write_elementwise_output: bool) void {
        self.recordQMatmulRowChainTiledSpill(m, n, 0, tile, write_elementwise_output, false);
    }

    pub fn recordQMatmulRowChainTiledSpill(self: *RuntimeProfile, m: u32, n: u32, k: u32, tile: u32, write_elementwise_output: bool, output_spill: bool) void {
        const row_tiles = divCeilU64(m, tile);
        const n_tiles = divCeilU64(n, tile);
        self.qmatmul_row_chain_tiled_count +%= 1;
        self.qmatmul_row_chain_tiled_row_tile_groups +%= row_tiles;
        self.qmatmul_row_chain_tiled_n_tiles +%= n_tiles;
        self.qmatmul_row_chain_tiled_serial_tile_loops +%= row_tiles *% n_tiles;
        self.qmatmul_row_chain_tiled_partial_slots +%= row_tiles *% @as(u64, tile) *% n_tiles;
        self.qmatmul_row_chain_tiled_scratch_capacity +%= @as(u64, m) *% @as(u64, n);
        if (write_elementwise_output) {
            self.qmatmul_row_chain_tiled_spilled_elementwise +%= 1;
            self.qmatmul_row_chain_tiled_spilled_input +%= k;
        }
        if (output_spill) self.qmatmul_row_chain_tiled_output_spills +%= 1;
    }

    pub fn recordQMatmulRowChainTwoPhaseTiled(self: *RuntimeProfile, m: u32, n: u32, tile: u32, write_elementwise_output: bool) void {
        self.recordQMatmulRowChainTwoPhaseTiledSpill(m, n, 0, tile, write_elementwise_output, false);
    }

    pub fn recordQMatmulRowChainTwoPhaseTiledSpill(self: *RuntimeProfile, m: u32, n: u32, k: u32, tile: u32, write_elementwise_output: bool, output_spill: bool) void {
        self.recordQMatmulRowChainTiledSpill(m, n, k, tile, write_elementwise_output, output_spill);
        self.qmatmul_row_chain_tiled_two_phase_count +%= 1;
        self.qmatmul_row_chain_tiled_finalize_tile_groups +%= divCeilU64(m, tile) *% divCeilU64(n, tile);
        self.qmatmul_row_chain_tiled_finalize_elements +%= @as(u64, m) *% @as(u64, n);
    }

    pub fn recordSemanticFfnSublayer(self: *RuntimeProfile, m: u32, h: u32, k: u32, o: u32) void {
        const hidden: u64 = h;
        const input: u64 = k;
        const output: u64 = o;
        const tile: u64 = 32;
        const row_groups = divCeilU64(m, tile);
        const hidden_tiles = divCeilU64(h, tile);
        const output_tiles = divCeilU64(o, tile);
        self.semantic_ffn_sublayer_count +%= 1;
        self.semantic_ffn_sublayer_rows +%= m;
        self.semantic_ffn_sublayer_hidden +%= hidden;
        self.semantic_ffn_sublayer_input +%= input;
        self.semantic_ffn_sublayer_output +%= output;
        const row_serial_dot_ops = (hidden *% input *% 2) +% (hidden *% output);
        self.semantic_ffn_sublayer_row_serial_dot_ops +%= row_serial_dot_ops;
        self.semantic_ffn_sublayer_total_row_serial_dot_ops +%= @as(u64, m) *% row_serial_dot_ops;
        self.semantic_ffn_sublayer_tile_row_groups +%= row_groups;
        self.semantic_ffn_sublayer_tile_hidden_tiles +%= hidden_tiles;
        self.semantic_ffn_sublayer_tile_output_tiles +%= output_tiles;
        self.semantic_ffn_sublayer_tile_parallel_groups +%= row_groups *% (hidden_tiles *% 2 +% output_tiles);
    }
};

fn perCall(count: u64, calls_f: f64) f64 {
    return if (calls_f > 0.0) @as(f64, @floatFromInt(count)) / calls_f else 0.0;
}

fn divCeilU64(n: u32, d: u32) u64 {
    if (d == 0) return 0;
    return (@as(u64, n) + @as(u64, d) - 1) / @as(u64, d);
}

fn writeJsonField(jw: *std.json.Stringify, key: []const u8, value: anytype) !void {
    try jw.objectField(key);
    try jw.write(value);
}

fn writeCountAndPerCall(jw: *std.json.Stringify, comptime prefix: []const u8, name: []const u8, count: u64, calls_f: f64) !void {
    var key_buf: [128]u8 = undefined;
    const count_key = try std.fmt.bufPrint(&key_buf, "{s}{s}", .{ prefix, name });
    try writeJsonField(jw, count_key, count);
    const per_call_key = try std.fmt.bufPrint(&key_buf, "{s}{s}_per_call", .{ prefix, name });
    try writeJsonField(jw, per_call_key, perCall(count, calls_f));
}

/// Write the structured runtime evidence used by benchmark gates.
pub fn writeRuntimeProfileJsonFields(rt: RuntimeProfile, jw: *std.json.Stringify) !void {
    const calls_f: f64 = @floatFromInt(rt.call_count);
    try writeJsonField(jw, "profile_calls", rt.call_count);

    const placed_ops = rt.backend_op_count + rt.fallback_op_count;
    if (placed_ops > 0) {
        const placed_f: f64 = @floatFromInt(placed_ops);
        try writeJsonField(jw, "backend_ops", rt.backend_op_count);
        try writeJsonField(jw, "fallback_ops", rt.fallback_op_count);
        try writeJsonField(jw, "fallback_pct", @as(f64, @floatFromInt(rt.fallback_op_count)) / placed_f * 100.0);
    }
    if (rt.backend_dispatch_count > 0) {
        try writeJsonField(jw, "dispatches", rt.backend_dispatch_count);
        try writeJsonField(jw, "dispatches_per_call", perCall(rt.backend_dispatch_count, calls_f));
    }
    if (rt.schedule_regions.attempted > 0) {
        const regions = rt.schedule_regions;
        const attempt_f: f64 = @floatFromInt(regions.attempted);
        try writeJsonField(jw, "schedule_region_attempts", regions.attempted);
        try writeJsonField(jw, "schedule_region_lowered", regions.lowered);
        try writeJsonField(jw, "schedule_region_lowered_pct", @as(f64, @floatFromInt(regions.lowered)) / attempt_f * 100.0);
        try writeJsonField(jw, "schedule_region_lowered_per_call", perCall(regions.lowered, calls_f));
        try writeJsonField(jw, "schedule_region_ops_per_call", perCall(regions.lowered_ops, calls_f));
        try writeJsonField(jw, "schedule_region_failed_ops", regions.failed_ops);
        try writeJsonField(jw, "schedule_region_failed_ops_per_call", perCall(regions.failed_ops, calls_f));
        if (rt.region_command_plan_cached_count > 0) {
            try writeJsonField(jw, "region_command_plan_cached", rt.region_command_plan_cached_count);
            try writeJsonField(jw, "region_command_plan_cached_per_call", perCall(rt.region_command_plan_cached_count, calls_f));
            try writeJsonField(jw, "commands_per_call", perCall(rt.region_command_plan_cached_command_count, calls_f));
        }
        try writeJsonField(jw, "dynamic_region_command_plans", rt.region_command_plan_dynamic_count);
        try writeJsonField(jw, "dynamic_region_command_plans_per_call", perCall(rt.region_command_plan_dynamic_count, calls_f));
    }
    if (rt.sync_count > 0) {
        try writeJsonField(jw, "sync_waits", rt.sync_count);
        try writeJsonField(jw, "syncs_per_call", perCall(rt.sync_count, calls_f));
    }
    if (rt.runtime_patch_call_count > 0) {
        try writeJsonField(jw, "runtime_patch_calls", rt.runtime_patch_call_count);
    }
    if (rt.runtime_patch_changed_count > 0 or rt.runtime_patch_invalid_count > 0) {
        try writeJsonField(jw, "runtime_patch_changed", rt.runtime_patch_changed_count);
        try writeJsonField(jw, "runtime_patch_invalid", rt.runtime_patch_invalid_count);
    }
    const patch_shape = rt.runtime_patch_shape;
    try writeJsonField(jw, "runtime_patch_holes", patch_shape.runtime_patch_holes);
    try writeJsonField(jw, "runtime_patch_cache_write_pos_holes", patch_shape.runtime_patch_cache_write_pos_holes);
    try writeJsonField(jw, "runtime_patch_attention_seq_kv_holes", patch_shape.runtime_patch_attention_seq_kv_holes);
    if (patch_shape.runtime_patch_stencil_hash != 0) {
        try writeJsonField(jw, "runtime_patch_stencil_hash", patch_shape.runtime_patch_stencil_hash);
    }

    const command_shape = rt.program_command_shape;
    if (command_shape.command_count > 0) {
        try writeJsonField(jw, "program_command_shape_commands", command_shape.command_count);
        try writeJsonField(jw, "program_command_shape_covered_ops", command_shape.covered_ops);
        try writeJsonField(jw, "program_command_shape_estimated_saved_dispatches", command_shape.estimated_saved_dispatches);
        try writeJsonField(jw, "program_command_shape_row_chains", command_shape.row_chains);
        try writeJsonField(jw, "program_command_shape_semantic_ffn_sublayers", command_shape.semantic_ffn_sublayers);
        try writeJsonField(jw, "program_command_shape_projection_row_chains", command_shape.projection_row_chains);
        try writeJsonField(jw, "program_command_shape_dense_projection_row_chains", command_shape.dense_projection_row_chains);
        try writeJsonField(jw, "program_command_shape_projection_chains", command_shape.projection_chains);
        try writeJsonField(jw, "program_command_shape_dense_projection_chains", command_shape.dense_projection_chains);
        try writeJsonField(jw, "program_command_shape_quantized_projection_chains", command_shape.quantized_projection_chains);
        try writeJsonField(jw, "program_command_shape_projection_chain_sidecars", command_shape.projection_chain_sidecars);
        try writeJsonField(jw, "program_command_shape_projection_chain_row_chain_frontiers", command_shape.projection_chain_row_chain_frontiers);
        try writeJsonField(jw, "program_command_shape_projection_row_chain_semantic_residual_bridges", command_shape.projection_row_chain_semantic_residual_bridges);
        try writeJsonField(jw, "program_command_shape_projection_groups", command_shape.projection_groups);
        try writeJsonField(jw, "program_command_shape_projection_anchors", command_shape.projection_anchors);
        try writeJsonField(jw, "program_command_shape_projection_sidecars", command_shape.projection_sidecars);
        try writeJsonField(jw, "program_command_shape_projection_cache_groups", command_shape.projection_cache_groups);
        try writeJsonField(jw, "program_command_shape_projection_cache_anchors", command_shape.projection_cache_anchors);
        try writeJsonField(jw, "program_command_shape_projection_cache_sidecars", command_shape.projection_cache_sidecars);
        try writeJsonField(jw, "program_command_shape_max_projection_span_ops", command_shape.max_projection_span_ops);
        try writeJsonField(jw, "program_command_shape_stencil_hash", command_shape.command_stencil_hash);
    }

    for (rt.program_command_counts, 0..) |count, i| {
        if (count == 0) continue;
        const kind: program_mod.ProgramCommandKind = @enumFromInt(i);
        try writeCountAndPerCall(jw, "program_command_encoded_", @tagName(kind), count, calls_f);
    }
    for (rt.program_command_dispatch_counts, 0..) |count, i| {
        if (count == 0) continue;
        const kind: program_mod.ProgramCommandKind = @enumFromInt(i);
        try writeCountAndPerCall(jw, "program_command_dispatches_", @tagName(kind), count, calls_f);
    }
    for (rt.program_command_attempt_counts, 0..) |count, i| {
        const failed = rt.program_command_failed_counts[i];
        if (count == 0 and failed == 0) continue;
        const kind: program_mod.ProgramCommandKind = @enumFromInt(i);
        try writeCountAndPerCall(jw, "program_command_attempts_", @tagName(kind), count, calls_f);
        try writeCountAndPerCall(jw, "program_command_refused_", @tagName(kind), failed, calls_f);
    }
    for (rt.program_op_command_counts, 0..) |count, i| {
        if (count == 0) continue;
        try writeCountAndPerCall(jw, "program_op_command_", tag_names[i], count, calls_f);
    }
    for (rt.projection_chain_qmatvec_sidecars, 0..) |count, i| {
        if (count == 0) continue;
        try writeCountAndPerCall(jw, "projection_chain_qmatvec_", tag_names[i], count, calls_f);
    }
    for (rt.projection_chain_qmatmul_sidecars, 0..) |count, i| {
        if (count == 0) continue;
        try writeCountAndPerCall(jw, "projection_chain_qmatmul_", tag_names[i], count, calls_f);
    }
    if (rt.qmatmul_row_chain_tiled_count > 0) {
        try writeCountAndPerCall(jw, "qmatmul_row_chain_tiled_", "count", rt.qmatmul_row_chain_tiled_count, calls_f);
        try writeCountAndPerCall(jw, "qmatmul_row_chain_tiled_", "row_tile_groups", rt.qmatmul_row_chain_tiled_row_tile_groups, calls_f);
        try writeCountAndPerCall(jw, "qmatmul_row_chain_tiled_", "n_tiles", rt.qmatmul_row_chain_tiled_n_tiles, calls_f);
        try writeCountAndPerCall(jw, "qmatmul_row_chain_tiled_", "serial_tile_loops", rt.qmatmul_row_chain_tiled_serial_tile_loops, calls_f);
        try writeCountAndPerCall(jw, "qmatmul_row_chain_tiled_", "partial_slots", rt.qmatmul_row_chain_tiled_partial_slots, calls_f);
        try writeCountAndPerCall(jw, "qmatmul_row_chain_tiled_", "scratch_capacity", rt.qmatmul_row_chain_tiled_scratch_capacity, calls_f);
        try writeCountAndPerCall(jw, "qmatmul_row_chain_tiled_", "spilled_elementwise", rt.qmatmul_row_chain_tiled_spilled_elementwise, calls_f);
        try writeCountAndPerCall(jw, "qmatmul_row_chain_tiled_", "spilled_input", rt.qmatmul_row_chain_tiled_spilled_input, calls_f);
        try writeCountAndPerCall(jw, "qmatmul_row_chain_tiled_", "output_spills", rt.qmatmul_row_chain_tiled_output_spills, calls_f);
        try writeCountAndPerCall(jw, "qmatmul_row_chain_tiled_", "two_phase_count", rt.qmatmul_row_chain_tiled_two_phase_count, calls_f);
        try writeCountAndPerCall(jw, "qmatmul_row_chain_tiled_", "finalize_tile_groups", rt.qmatmul_row_chain_tiled_finalize_tile_groups, calls_f);
        try writeCountAndPerCall(jw, "qmatmul_row_chain_tiled_", "finalize_elements", rt.qmatmul_row_chain_tiled_finalize_elements, calls_f);
    }
    if (rt.semantic_ffn_sublayer_count > 0) {
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_", "count", rt.semantic_ffn_sublayer_count, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_", "rows", rt.semantic_ffn_sublayer_rows, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_", "hidden", rt.semantic_ffn_sublayer_hidden, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_", "input", rt.semantic_ffn_sublayer_input, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_", "output", rt.semantic_ffn_sublayer_output, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_", "row_serial_dot_ops", rt.semantic_ffn_sublayer_row_serial_dot_ops, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_", "total_row_serial_dot_ops", rt.semantic_ffn_sublayer_total_row_serial_dot_ops, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_", "tile_row_groups", rt.semantic_ffn_sublayer_tile_row_groups, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_", "tile_hidden_tiles", rt.semantic_ffn_sublayer_tile_hidden_tiles, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_", "tile_output_tiles", rt.semantic_ffn_sublayer_tile_output_tiles, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_", "tile_parallel_groups", rt.semantic_ffn_sublayer_tile_parallel_groups, calls_f);
        if (rt.semantic_ffn_sublayer_tile_parallel_groups > 0) {
            try writeJsonField(
                jw,
                "semantic_ffn_sublayer_row_serial_dot_ops_per_tile_parallel_group",
                @as(f64, @floatFromInt(rt.semantic_ffn_sublayer_row_serial_dot_ops)) / @as(f64, @floatFromInt(rt.semantic_ffn_sublayer_tile_parallel_groups)),
            );
            try writeJsonField(
                jw,
                "semantic_ffn_sublayer_total_row_serial_dot_ops_per_tile_parallel_group",
                @as(f64, @floatFromInt(rt.semantic_ffn_sublayer_total_row_serial_dot_ops)) / @as(f64, @floatFromInt(rt.semantic_ffn_sublayer_tile_parallel_groups)),
            );
        }
    }
}

fn scheduleRegionPatternSlot(unit: program_mod.ScheduleUnit) ?usize {
    const idx: usize = @intCast(unit.pattern_index);
    return if (idx < max_schedule_region_patterns) idx else null;
}

test "RuntimeProfile reset" {
    var rt = RuntimeProfile{};
    rt.recordProgramCommandDispatch(.projection_cache_group);
    rt.schedule_regions = .{
        .attempted = 2,
        .lowered = 1,
        .failed = 1,
        .attempted_ops = 14,
        .lowered_ops = 7,
        .failed_ops = 7,
    };
    rt.schedule_region_patterns[3] = .{
        .attempted = 2,
        .lowered = 1,
        .failed = 1,
        .attempted_ops = 14,
        .lowered_ops = 7,
        .failed_ops = 7,
    };
    rt.region_command_plan_cached_count = 2;
    rt.region_command_plan_cached_command_count = 11;
    rt.region_command_plan_dynamic_count = 13;
    rt.backend_op_count = 7;
    rt.fallback_op_count = 11;
    rt.backend_dispatch_count = 3;
    rt.sync_count = 17;
    rt.runtime_patch_shape = backend.RuntimePatchShape.actual(7, 12, 99);
    rt.call_count = 5;
    rt.reset();
    try std.testing.expectEqual(@as(u64, 0), rt.program_command_dispatch_counts[@intFromEnum(program_mod.ProgramCommandKind.projection_cache_group)]);
    try std.testing.expectEqual(@as(u64, 0), rt.program_op_command_counts[@intFromEnum(std.meta.Tag(backend.DeviceOp).softmax)]);
    try std.testing.expectEqual(@as(u64, 0), rt.projection_chain_qmatvec_sidecars[@intFromEnum(std.meta.Tag(backend.DeviceOp).fused_elementwise)]);
    try std.testing.expectEqual(ScheduleRegionStats{}, rt.schedule_regions);
    try std.testing.expectEqual(ScheduleRegionStats{}, rt.schedule_region_patterns[3]);
    try std.testing.expectEqual(@as(u64, 0), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.region_command_plan_cached_count);
    try std.testing.expectEqual(@as(u64, 0), rt.region_command_plan_cached_command_count);
    try std.testing.expectEqual(@as(u64, 0), rt.region_command_plan_dynamic_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.backend_dispatch_count);
    try std.testing.expectEqual(@as(u64, 0), rt.sync_count);
    try std.testing.expectEqual(backend.RuntimePatchShape.actual(7, 12, 99), rt.runtime_patch_shape);
    try std.testing.expectEqual(@as(u32, 0), rt.call_count);
}

test "RuntimeProfile records schedule region lowerings" {
    var rt = RuntimeProfile{};
    const unit = program_mod.ScheduleUnit{
        .kind = .pattern_region,
        .pattern_index = 3,
        .start_item = 2,
        .item_count = 4,
        .op_start = 10,
        .op_count = 7,
    };

    rt.recordScheduleRegionAttempt(unit);
    rt.recordScheduleRegionLowered(unit);
    rt.recordScheduleRegionFailed(unit);

    const expected = ScheduleRegionStats{
        .attempted = 1,
        .lowered = 1,
        .failed = 1,
        .attempted_ops = 7,
        .lowered_ops = 7,
        .failed_ops = 7,
    };
    try std.testing.expectEqual(expected, rt.schedule_regions);
    try std.testing.expectEqual(expected, rt.schedule_region_patterns[3]);
}

test "RuntimeProfile records region command plan evidence" {
    var rt = RuntimeProfile{};
    rt.recordCachedRegionCommandPlan(9);
    rt.recordCachedRegionCommandPlan(4);
    rt.recordDynamicRegionCommandPlan();

    try std.testing.expectEqual(@as(u64, 2), rt.region_command_plan_cached_count);
    try std.testing.expectEqual(@as(u64, 13), rt.region_command_plan_cached_command_count);
    try std.testing.expectEqual(@as(u64, 1), rt.region_command_plan_dynamic_count);
}

test "RuntimeProfile records runtime patch status" {
    var rt = RuntimeProfile{};
    rt.recordRuntimePatch(.unchanged);
    try std.testing.expectEqual(@as(u64, 1), rt.runtime_patch_call_count);
    try std.testing.expectEqual(@as(u64, 0), rt.runtime_patch_changed_count);
    try std.testing.expectEqual(@as(u64, 0), rt.runtime_patch_invalid_count);
    rt.recordRuntimePatch(.changed);
    try std.testing.expectEqual(@as(u64, 2), rt.runtime_patch_call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.runtime_patch_changed_count);
    rt.recordRuntimePatch(.invalid);
    try std.testing.expectEqual(@as(u64, 3), rt.runtime_patch_call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.runtime_patch_invalid_count);
}

test "RuntimeProfile always serializes runtime patch-hole count" {
    var aw: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer aw.deinit();
    var jw: std.json.Stringify = .{ .writer = &aw.writer };

    try jw.beginObject();
    try writeRuntimeProfileJsonFields(.{}, &jw);
    try jw.endObject();
    var out = aw.writer.buffer[0..aw.writer.end];
    try std.testing.expect(std.mem.indexOf(u8, out, "\"runtime_patch_holes\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"runtime_patch_cache_write_pos_holes\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"runtime_patch_attention_seq_kv_holes\":0") != null);

    aw.clearRetainingCapacity();
    jw = .{ .writer = &aw.writer };
    try jw.beginObject();
    try writeRuntimeProfileJsonFields(.{
        .runtime_patch_shape = backend.RuntimePatchShape.actual(3, 4, 123),
    }, &jw);
    try jw.endObject();
    out = aw.writer.buffer[0..aw.writer.end];
    try std.testing.expect(std.mem.indexOf(u8, out, "\"runtime_patch_holes\":7") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"runtime_patch_cache_write_pos_holes\":3") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"runtime_patch_attention_seq_kv_holes\":4") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"runtime_patch_stencil_hash\":123") != null);
}

test "RuntimeProfile serializes program command shape compression evidence" {
    var rt = RuntimeProfile{};
    rt.program_command_shape = .{
        .command_count = 7,
        .covered_ops = 19,
        .estimated_saved_dispatches = 12,
        .row_chains = 2,
        .projection_chains = 3,
        .dense_projection_chains = 1,
        .quantized_projection_chains = 2,
        .projection_chain_sidecars = 3,
        .projection_chain_row_chain_frontiers = 2,
        .projection_row_chain_semantic_residual_bridges = 1,
        .projection_groups = 1,
        .projection_anchors = 4,
        .projection_sidecars = 2,
        .projection_cache_groups = 1,
        .projection_cache_anchors = 4,
        .projection_cache_sidecars = 4,
        .max_projection_span_ops = 8,
        .command_stencil_hash = 12345,
    };

    var aw: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer aw.deinit();
    var jw: std.json.Stringify = .{ .writer = &aw.writer };
    try jw.beginObject();
    try writeRuntimeProfileJsonFields(rt, &jw);
    try jw.endObject();
    const out = aw.writer.buffer[0..aw.writer.end];
    try std.testing.expect(std.mem.indexOf(u8, out, "\"program_command_shape_commands\":7") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"program_command_shape_covered_ops\":19") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"program_command_shape_estimated_saved_dispatches\":12") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"program_command_shape_projection_chains\":3") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"program_command_shape_dense_projection_chains\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"program_command_shape_quantized_projection_chains\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"program_command_shape_projection_chain_row_chain_frontiers\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"program_command_shape_projection_row_chain_semantic_residual_bridges\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"program_command_shape_projection_cache_sidecars\":4") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"program_command_shape_stencil_hash\":12345") != null);
}

test "RuntimeProfile serializes dynamic command-plan evidence from counters" {
    var rt = RuntimeProfile{};
    const unit = program_mod.ScheduleUnit{
        .kind = .pattern_region,
        .pattern_index = 0,
        .start_item = 0,
        .item_count = 1,
        .op_start = 0,
        .op_count = 1,
    };
    rt.recordScheduleRegionAttempt(unit);
    rt.recordCachedRegionCommandPlan(3);
    rt.recordDynamicRegionCommandPlan();
    rt.recordQMatmulRowChainTiledSpill(128, 576, 576, 32, true, true);
    rt.recordSemanticFfnSublayer(128, 576, 576, 576);
    rt.call_count = 2;

    var aw: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer aw.deinit();
    var jw: std.json.Stringify = .{ .writer = &aw.writer };

    try jw.beginObject();
    try writeRuntimeProfileJsonFields(rt, &jw);
    try jw.endObject();
    const out = aw.writer.buffer[0..aw.writer.end];
    try std.testing.expect(std.mem.indexOf(u8, out, "\"dynamic_region_command_plans\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"dynamic_region_command_plans_per_call\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"qmatmul_row_chain_tiled_count\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"qmatmul_row_chain_tiled_row_tile_groups\":4") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"qmatmul_row_chain_tiled_n_tiles\":18") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"qmatmul_row_chain_tiled_serial_tile_loops\":72") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"qmatmul_row_chain_tiled_partial_slots\":2304") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"qmatmul_row_chain_tiled_scratch_capacity\":73728") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"qmatmul_row_chain_tiled_spilled_elementwise\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"qmatmul_row_chain_tiled_spilled_input\":576") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"qmatmul_row_chain_tiled_output_spills\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"qmatmul_row_chain_tiled_two_phase_count\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"qmatmul_row_chain_tiled_finalize_tile_groups\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"qmatmul_row_chain_tiled_finalize_elements\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_count\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_row_serial_dot_ops\":995328") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_total_row_serial_dot_ops\":127401984") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_tile_parallel_groups\":216") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_row_serial_dot_ops_per_tile_parallel_group\":4608") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_total_row_serial_dot_ops_per_tile_parallel_group\":589824") != null);
}

test "RuntimeProfile accumulates evidence windows" {
    var total = RuntimeProfile{};
    var window = RuntimeProfile{};
    window.recordProgramCommand(.projection_chain);
    window.recordProgramCommandDispatch(.projection_chain);
    window.recordProgramCommandAttempt(.projection_chain);
    window.recordProgramCommandFailed(.projection_chain);
    window.recordProgramOpCommand(.{ .softmax = .{ .dst = 0, .src = 0, .rows = 1, .cols = 1 } });
    window.recordProjectionChainSidecar(true, .{ .fused_elementwise = .{ .steps = &.{}, .n = 0, .dst = 0, .src = 0, .dst_offset = 0, .src_offset = 0 } });
    window.recordProjectionChainSidecar(false, .{ .elementwise = .{ .op = .add, .dst = 0, .src0 = 0, .src1 = 0, .n = 1 } });
    window.schedule_regions.recordLowered(3);
    window.schedule_region_patterns[2].recordAttempt(3);
    window.region_command_plan_cached_count = 1;
    window.region_command_plan_cached_command_count = 7;
    window.region_command_plan_dynamic_count = 1;
    window.backend_op_count = 11;
    window.fallback_op_count = 13;
    window.backend_dispatch_count = 17;
    window.sync_count = 23;
    window.runtime_patch_call_count = 26;
    window.runtime_patch_changed_count = 27;
    window.runtime_patch_invalid_count = 28;
    window.runtime_patch_shape = backend.RuntimePatchShape.actual(17, 24, 4242);
    window.recordQMatmulRowChainTwoPhaseTiledSpill(128, 576, 576, 32, true, true);
    window.call_count = 37;

    total.add(window);
    total.add(window);

    const projection_chain = @intFromEnum(program_mod.ProgramCommandKind.projection_chain);
    try std.testing.expectEqual(@as(u64, 2), total.program_command_counts[projection_chain]);
    try std.testing.expectEqual(@as(u64, 2), total.program_command_dispatch_counts[projection_chain]);
    try std.testing.expectEqual(@as(u64, 2), total.program_command_attempt_counts[projection_chain]);
    try std.testing.expectEqual(@as(u64, 2), total.program_command_failed_counts[projection_chain]);
    try std.testing.expectEqual(@as(u64, 2), total.program_op_command_counts[@intFromEnum(std.meta.Tag(backend.DeviceOp).softmax)]);
    try std.testing.expectEqual(@as(u64, 2), total.projection_chain_qmatvec_sidecars[@intFromEnum(std.meta.Tag(backend.DeviceOp).fused_elementwise)]);
    try std.testing.expectEqual(@as(u64, 2), total.projection_chain_qmatmul_sidecars[@intFromEnum(std.meta.Tag(backend.DeviceOp).elementwise)]);
    try std.testing.expectEqual(@as(u64, 2), total.schedule_regions.lowered);
    try std.testing.expectEqual(@as(u64, 6), total.schedule_regions.lowered_ops);
    try std.testing.expectEqual(@as(u64, 2), total.schedule_region_patterns[2].attempted);
    try std.testing.expectEqual(@as(u64, 2), total.region_command_plan_cached_count);
    try std.testing.expectEqual(@as(u64, 14), total.region_command_plan_cached_command_count);
    try std.testing.expectEqual(@as(u64, 2), total.region_command_plan_dynamic_count);
    try std.testing.expectEqual(@as(u64, 22), total.backend_op_count);
    try std.testing.expectEqual(@as(u64, 26), total.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 34), total.backend_dispatch_count);
    try std.testing.expectEqual(@as(u64, 46), total.sync_count);
    try std.testing.expectEqual(@as(u64, 52), total.runtime_patch_call_count);
    try std.testing.expectEqual(@as(u64, 54), total.runtime_patch_changed_count);
    try std.testing.expectEqual(@as(u64, 56), total.runtime_patch_invalid_count);
    try std.testing.expectEqual(backend.RuntimePatchShape.actual(17, 24, 4242), total.runtime_patch_shape);
    try std.testing.expectEqual(@as(u64, 2), total.qmatmul_row_chain_tiled_count);
    try std.testing.expectEqual(@as(u64, 8), total.qmatmul_row_chain_tiled_row_tile_groups);
    try std.testing.expectEqual(@as(u64, 36), total.qmatmul_row_chain_tiled_n_tiles);
    try std.testing.expectEqual(@as(u64, 144), total.qmatmul_row_chain_tiled_serial_tile_loops);
    try std.testing.expectEqual(@as(u64, 4608), total.qmatmul_row_chain_tiled_partial_slots);
    try std.testing.expectEqual(@as(u64, 147456), total.qmatmul_row_chain_tiled_scratch_capacity);
    try std.testing.expectEqual(@as(u64, 2), total.qmatmul_row_chain_tiled_spilled_elementwise);
    try std.testing.expectEqual(@as(u64, 1152), total.qmatmul_row_chain_tiled_spilled_input);
    try std.testing.expectEqual(@as(u64, 2), total.qmatmul_row_chain_tiled_output_spills);
    try std.testing.expectEqual(@as(u64, 2), total.qmatmul_row_chain_tiled_two_phase_count);
    try std.testing.expectEqual(@as(u64, 144), total.qmatmul_row_chain_tiled_finalize_tile_groups);
    try std.testing.expectEqual(@as(u64, 147456), total.qmatmul_row_chain_tiled_finalize_elements);
    try std.testing.expectEqual(@as(u32, 74), total.call_count);
}

test "RuntimeProfile ignores out-of-range schedule region pattern counters" {
    var rt = RuntimeProfile{};
    const unit = program_mod.ScheduleUnit{
        .kind = .pattern_region,
        .pattern_index = 999,
        .start_item = 0,
        .item_count = 1,
        .op_start = 0,
        .op_count = 4,
    };

    rt.recordScheduleRegionAttempt(unit);
    rt.recordScheduleRegionLowered(unit);

    try std.testing.expectEqual(@as(u64, 1), rt.schedule_regions.attempted);
    try std.testing.expectEqual(@as(u64, 1), rt.schedule_regions.lowered);
    for (rt.schedule_region_patterns) |stats| {
        try std.testing.expectEqual(ScheduleRegionStats{}, stats);
    }
}
