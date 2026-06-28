//! Profiling utilities for backend placement and runtime evidence.

const std = @import("std");
const backend = @import("backend.zig");
const program_mod = @import("backend/program.zig");

const op_fields = @typeInfo(backend.DeviceOp).@"union".fields;
const n_op_tags = op_fields.len;
const n_program_command_kinds = @typeInfo(program_mod.ProgramCommandKind).@"enum".fields.len;
const max_schedule_region_patterns = 16;

pub const SemanticFfnSublayerSingleDispatchRefusalReason = enum {
    pair_chain,
    down_shape,
    down_input,
    residual_shape,
    rms_source,
    rms_shape,
    rms_fuse,
    dim,
    qweight,
    output_read,
    block_size,
    qparam_shape,
    qparam_input,
    down_param_shape,
};

const semantic_single_dispatch_refusal_fields = @typeInfo(SemanticFfnSublayerSingleDispatchRefusalReason).@"enum".fields;
const n_semantic_single_dispatch_refusal_reasons = semantic_single_dispatch_refusal_fields.len;

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
    qmatmul_row_chain_width_parallel_count: u64 = 0,
    qmatmul_row_chain_width_parallel_lanes: u64 = 0,
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
    semantic_ffn_sublayer_width_lane_slots: u64 = 0,
    semantic_ffn_sublayer_active_width_lanes: u64 = 0,
    semantic_ffn_sublayer_thread_lane_slots: u64 = 0,
    semantic_ffn_sublayer_active_thread_lanes: u64 = 0,
    semantic_ffn_sublayer_fallback_pair_dispatches: u64 = 0,
    semantic_ffn_sublayer_fallback_tail_dispatches: u64 = 0,
    semantic_ffn_with_input_decomposed_count: u64 = 0,
    semantic_ffn_with_input_decomposed_dispatches: u64 = 0,
    semantic_ffn_with_input_decomposed_extra_dispatches: u64 = 0,
    semantic_ffn_with_input_decomposed_row_chain_dispatches: u64 = 0,
    semantic_ffn_with_input_decomposed_pair_dispatches: u64 = 0,
    semantic_ffn_with_input_decomposed_tail_dispatches: u64 = 0,
    semantic_ffn_with_input_direct_count: u64 = 0,
    semantic_ffn_with_input_direct_rows: u64 = 0,
    semantic_ffn_with_input_direct_input_projection: u64 = 0,
    semantic_ffn_with_input_direct_input: u64 = 0,
    semantic_ffn_with_input_direct_hidden: u64 = 0,
    semantic_ffn_with_input_direct_output: u64 = 0,
    semantic_ffn_with_input_direct_input_projection_dot_ops: u64 = 0,
    semantic_ffn_with_input_direct_gate_up_dot_ops: u64 = 0,
    semantic_ffn_with_input_direct_down_dot_ops: u64 = 0,
    semantic_ffn_with_input_direct_row_serial_dot_ops: u64 = 0,
    semantic_ffn_with_input_direct_total_row_serial_dot_ops: u64 = 0,
    semantic_ffn_with_input_direct_row_threadgroups: u64 = 0,
    semantic_ffn_with_input_direct_width_parallel_count: u64 = 0,
    semantic_ffn_with_input_direct_width_parallel_rows: u64 = 0,
    semantic_ffn_with_input_direct_width_parallel_row_tile_groups: u64 = 0,
    semantic_ffn_with_input_direct_width_parallel_output_tiles: u64 = 0,
    semantic_ffn_with_input_direct_width_parallel_lanes: u64 = 0,
    semantic_ffn_with_input_direct_width_parallel_partial_slots: u64 = 0,
    semantic_ffn_sublayer_single_dispatch_attempts: u64 = 0,
    semantic_ffn_sublayer_single_dispatch_output_read_refusals: u64 = 0,
    semantic_ffn_sublayer_single_dispatch_block_size_refusals: u64 = 0,
    semantic_ffn_sublayer_single_dispatch_refusal_reasons: [n_semantic_single_dispatch_refusal_reasons]u64 = [_]u64{0} ** n_semantic_single_dispatch_refusal_reasons,
    semantic_ffn_sublayer_single_dispatch_dim_refusal_max_k: u64 = 0,
    semantic_ffn_sublayer_single_dispatch_dim_refusal_max_h: u64 = 0,
    semantic_ffn_sublayer_single_dispatch_dim_refusal_max_o: u64 = 0,
    semantic_ffn_sublayer_single_dispatch_dim_refusal_cap: u64 = 0,
    semantic_width_scratch_candidates: u64 = 0,
    semantic_width_scratch_bytes: u64 = 0,
    semantic_width_scratch_input_bytes: u64 = 0,
    semantic_width_scratch_product_bytes: u64 = 0,
    semantic_width_scratch_down_partial_bytes: u64 = 0,
    semantic_width_scratch_output_bytes: u64 = 0,
    semantic_width_scratch_down_partial_to_output_x1000: u64 = 0,
    semantic_width_scratch_allocated_bytes: u64 = 0,
    semantic_width_scratch_runtime_capacity_bytes: u64 = 0,
    semantic_width_scratch_runtime_uses: u64 = 0,
    semantic_width_scratch_runtime_bytes: u64 = 0,
    call_count: u32 = 0,

    pub fn reset(self: *RuntimeProfile) void {
        const runtime_patch_shape = self.runtime_patch_shape;
        const program_command_shape = self.program_command_shape;
        const semantic_width_scratch_candidates = self.semantic_width_scratch_candidates;
        const semantic_width_scratch_bytes = self.semantic_width_scratch_bytes;
        const semantic_width_scratch_input_bytes = self.semantic_width_scratch_input_bytes;
        const semantic_width_scratch_product_bytes = self.semantic_width_scratch_product_bytes;
        const semantic_width_scratch_down_partial_bytes = self.semantic_width_scratch_down_partial_bytes;
        const semantic_width_scratch_output_bytes = self.semantic_width_scratch_output_bytes;
        const semantic_width_scratch_down_partial_to_output_x1000 = self.semantic_width_scratch_down_partial_to_output_x1000;
        const semantic_width_scratch_allocated_bytes = self.semantic_width_scratch_allocated_bytes;
        const semantic_width_scratch_runtime_capacity_bytes = self.semantic_width_scratch_runtime_capacity_bytes;
        const semantic_width_scratch_runtime_uses = self.semantic_width_scratch_runtime_uses;
        const semantic_width_scratch_runtime_bytes = self.semantic_width_scratch_runtime_bytes;
        self.* = .{
            .runtime_patch_shape = runtime_patch_shape,
            .program_command_shape = program_command_shape,
            .semantic_width_scratch_candidates = semantic_width_scratch_candidates,
            .semantic_width_scratch_bytes = semantic_width_scratch_bytes,
            .semantic_width_scratch_input_bytes = semantic_width_scratch_input_bytes,
            .semantic_width_scratch_product_bytes = semantic_width_scratch_product_bytes,
            .semantic_width_scratch_down_partial_bytes = semantic_width_scratch_down_partial_bytes,
            .semantic_width_scratch_output_bytes = semantic_width_scratch_output_bytes,
            .semantic_width_scratch_down_partial_to_output_x1000 = semantic_width_scratch_down_partial_to_output_x1000,
            .semantic_width_scratch_allocated_bytes = semantic_width_scratch_allocated_bytes,
            .semantic_width_scratch_runtime_capacity_bytes = semantic_width_scratch_runtime_capacity_bytes,
            .semantic_width_scratch_runtime_uses = semantic_width_scratch_runtime_uses,
            .semantic_width_scratch_runtime_bytes = semantic_width_scratch_runtime_bytes,
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
        self.qmatmul_row_chain_width_parallel_count +%= other.qmatmul_row_chain_width_parallel_count;
        self.qmatmul_row_chain_width_parallel_lanes +%= other.qmatmul_row_chain_width_parallel_lanes;
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
        self.semantic_ffn_sublayer_width_lane_slots +%= other.semantic_ffn_sublayer_width_lane_slots;
        self.semantic_ffn_sublayer_active_width_lanes +%= other.semantic_ffn_sublayer_active_width_lanes;
        self.semantic_ffn_sublayer_thread_lane_slots +%= other.semantic_ffn_sublayer_thread_lane_slots;
        self.semantic_ffn_sublayer_active_thread_lanes +%= other.semantic_ffn_sublayer_active_thread_lanes;
        self.semantic_ffn_sublayer_fallback_pair_dispatches +%= other.semantic_ffn_sublayer_fallback_pair_dispatches;
        self.semantic_ffn_sublayer_fallback_tail_dispatches +%= other.semantic_ffn_sublayer_fallback_tail_dispatches;
        self.semantic_ffn_with_input_decomposed_count +%= other.semantic_ffn_with_input_decomposed_count;
        self.semantic_ffn_with_input_decomposed_dispatches +%= other.semantic_ffn_with_input_decomposed_dispatches;
        self.semantic_ffn_with_input_decomposed_extra_dispatches +%= other.semantic_ffn_with_input_decomposed_extra_dispatches;
        self.semantic_ffn_with_input_decomposed_row_chain_dispatches +%= other.semantic_ffn_with_input_decomposed_row_chain_dispatches;
        self.semantic_ffn_with_input_decomposed_pair_dispatches +%= other.semantic_ffn_with_input_decomposed_pair_dispatches;
        self.semantic_ffn_with_input_decomposed_tail_dispatches +%= other.semantic_ffn_with_input_decomposed_tail_dispatches;
        self.semantic_ffn_with_input_direct_count +%= other.semantic_ffn_with_input_direct_count;
        self.semantic_ffn_with_input_direct_rows +%= other.semantic_ffn_with_input_direct_rows;
        self.semantic_ffn_with_input_direct_input_projection +%= other.semantic_ffn_with_input_direct_input_projection;
        self.semantic_ffn_with_input_direct_input +%= other.semantic_ffn_with_input_direct_input;
        self.semantic_ffn_with_input_direct_hidden +%= other.semantic_ffn_with_input_direct_hidden;
        self.semantic_ffn_with_input_direct_output +%= other.semantic_ffn_with_input_direct_output;
        self.semantic_ffn_with_input_direct_input_projection_dot_ops +%= other.semantic_ffn_with_input_direct_input_projection_dot_ops;
        self.semantic_ffn_with_input_direct_gate_up_dot_ops +%= other.semantic_ffn_with_input_direct_gate_up_dot_ops;
        self.semantic_ffn_with_input_direct_down_dot_ops +%= other.semantic_ffn_with_input_direct_down_dot_ops;
        self.semantic_ffn_with_input_direct_row_serial_dot_ops +%= other.semantic_ffn_with_input_direct_row_serial_dot_ops;
        self.semantic_ffn_with_input_direct_total_row_serial_dot_ops +%= other.semantic_ffn_with_input_direct_total_row_serial_dot_ops;
        self.semantic_ffn_with_input_direct_row_threadgroups +%= other.semantic_ffn_with_input_direct_row_threadgroups;
        self.semantic_ffn_with_input_direct_width_parallel_count +%= other.semantic_ffn_with_input_direct_width_parallel_count;
        self.semantic_ffn_with_input_direct_width_parallel_rows +%= other.semantic_ffn_with_input_direct_width_parallel_rows;
        self.semantic_ffn_with_input_direct_width_parallel_row_tile_groups +%= other.semantic_ffn_with_input_direct_width_parallel_row_tile_groups;
        self.semantic_ffn_with_input_direct_width_parallel_output_tiles +%= other.semantic_ffn_with_input_direct_width_parallel_output_tiles;
        self.semantic_ffn_with_input_direct_width_parallel_lanes +%= other.semantic_ffn_with_input_direct_width_parallel_lanes;
        self.semantic_ffn_with_input_direct_width_parallel_partial_slots +%= other.semantic_ffn_with_input_direct_width_parallel_partial_slots;
        self.semantic_ffn_sublayer_single_dispatch_attempts +%= other.semantic_ffn_sublayer_single_dispatch_attempts;
        self.semantic_ffn_sublayer_single_dispatch_output_read_refusals +%= other.semantic_ffn_sublayer_single_dispatch_output_read_refusals;
        self.semantic_ffn_sublayer_single_dispatch_block_size_refusals +%= other.semantic_ffn_sublayer_single_dispatch_block_size_refusals;
        for (&self.semantic_ffn_sublayer_single_dispatch_refusal_reasons, other.semantic_ffn_sublayer_single_dispatch_refusal_reasons) |*dst, value| dst.* +%= value;
        self.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_k = @max(self.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_k, other.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_k);
        self.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_h = @max(self.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_h, other.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_h);
        self.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_o = @max(self.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_o, other.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_o);
        self.semantic_ffn_sublayer_single_dispatch_dim_refusal_cap = @max(self.semantic_ffn_sublayer_single_dispatch_dim_refusal_cap, other.semantic_ffn_sublayer_single_dispatch_dim_refusal_cap);
        self.semantic_width_scratch_candidates +%= other.semantic_width_scratch_candidates;
        self.semantic_width_scratch_bytes = @max(self.semantic_width_scratch_bytes, other.semantic_width_scratch_bytes);
        self.semantic_width_scratch_input_bytes = @max(self.semantic_width_scratch_input_bytes, other.semantic_width_scratch_input_bytes);
        self.semantic_width_scratch_product_bytes = @max(self.semantic_width_scratch_product_bytes, other.semantic_width_scratch_product_bytes);
        self.semantic_width_scratch_down_partial_bytes = @max(self.semantic_width_scratch_down_partial_bytes, other.semantic_width_scratch_down_partial_bytes);
        self.semantic_width_scratch_output_bytes = @max(self.semantic_width_scratch_output_bytes, other.semantic_width_scratch_output_bytes);
        self.semantic_width_scratch_down_partial_to_output_x1000 = @max(self.semantic_width_scratch_down_partial_to_output_x1000, other.semantic_width_scratch_down_partial_to_output_x1000);
        self.semantic_width_scratch_allocated_bytes = @max(self.semantic_width_scratch_allocated_bytes, other.semantic_width_scratch_allocated_bytes);
        self.semantic_width_scratch_runtime_capacity_bytes = @max(self.semantic_width_scratch_runtime_capacity_bytes, other.semantic_width_scratch_runtime_capacity_bytes);
        self.semantic_width_scratch_runtime_uses +%= other.semantic_width_scratch_runtime_uses;
        self.semantic_width_scratch_runtime_bytes = @max(self.semantic_width_scratch_runtime_bytes, other.semantic_width_scratch_runtime_bytes);
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

    pub fn recordQMatmulRowChainWidthParallelTiledSpill(self: *RuntimeProfile, m: u32, n: u32, k: u32, tile: u32, width_lanes: u32, write_elementwise_output: bool, output_spill: bool) void {
        self.recordQMatmulRowChainTwoPhaseTiledSpill(m, n, k, tile, write_elementwise_output, output_spill);
        self.qmatmul_row_chain_width_parallel_count +%= 1;
        self.qmatmul_row_chain_width_parallel_lanes +%= width_lanes;
    }

    pub fn recordSemanticFfnSublayer(self: *RuntimeProfile, m: u32, h: u32, k: u32, o: u32, thread_lanes: u32) void {
        const hidden: u64 = h;
        const input: u64 = k;
        const output: u64 = o;
        const tile: u64 = 32;
        const threads: u64 = thread_lanes;
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
        if (threads > 0) {
            const input_slots = divCeilU64(k, thread_lanes) *% threads;
            const hidden_slots = divCeilU64(h, thread_lanes) *% threads;
            const output_slots = divCeilU64(o, thread_lanes) *% threads;
            const active_width_lanes_per_row = input +% hidden +% output *% 2;
            const width_lane_slots_per_row = input_slots +% hidden_slots +% output_slots *% 2;
            const active_lanes_per_row = active_width_lanes_per_row +% threads;
            const lane_slots_per_row = width_lane_slots_per_row +% threads;
            self.semantic_ffn_sublayer_active_width_lanes +%= @as(u64, m) *% active_width_lanes_per_row;
            self.semantic_ffn_sublayer_width_lane_slots +%= @as(u64, m) *% width_lane_slots_per_row;
            self.semantic_ffn_sublayer_active_thread_lanes +%= @as(u64, m) *% active_lanes_per_row;
            self.semantic_ffn_sublayer_thread_lane_slots +%= @as(u64, m) *% lane_slots_per_row;
        }
    }

    pub fn recordSemanticFfnSublayerSingleDispatchAttempt(self: *RuntimeProfile) void {
        self.semantic_ffn_sublayer_single_dispatch_attempts +%= 1;
    }

    pub fn recordSemanticFfnSublayerFallbackDispatches(self: *RuntimeProfile, pair_dispatches: u64, tail_dispatches: u64) void {
        self.semantic_ffn_sublayer_fallback_pair_dispatches +%= pair_dispatches;
        self.semantic_ffn_sublayer_fallback_tail_dispatches +%= tail_dispatches;
    }

    pub fn recordSemanticFfnWithInputDecomposed(self: *RuntimeProfile, dispatches: u64, row_chain_dispatches: u64, pair_dispatches: u64, tail_dispatches: u64) void {
        self.semantic_ffn_with_input_decomposed_count +%= 1;
        self.semantic_ffn_with_input_decomposed_dispatches +%= dispatches;
        self.semantic_ffn_with_input_decomposed_extra_dispatches +%= dispatches -| 1;
        self.semantic_ffn_with_input_decomposed_row_chain_dispatches +%= row_chain_dispatches;
        self.semantic_ffn_with_input_decomposed_pair_dispatches +%= pair_dispatches;
        self.semantic_ffn_with_input_decomposed_tail_dispatches +%= tail_dispatches;
    }

    pub fn recordSemanticFfnWithInputDirect(self: *RuntimeProfile, m: u32, h: u32, k: u32, o: u32, input_projection_k: u32) void {
        const rows: u64 = m;
        const hidden: u64 = h;
        const input: u64 = k;
        const output: u64 = o;
        const input_projection: u64 = input_projection_k;
        const input_projection_dot_ops = input_projection *% input;
        const gate_up_dot_ops = hidden *% input *% 2;
        const down_dot_ops = hidden *% output;
        const row_serial_dot_ops = input_projection_dot_ops +% gate_up_dot_ops +% down_dot_ops;
        self.semantic_ffn_with_input_direct_count +%= 1;
        self.semantic_ffn_with_input_direct_rows +%= rows;
        self.semantic_ffn_with_input_direct_input_projection +%= input_projection;
        self.semantic_ffn_with_input_direct_input +%= input;
        self.semantic_ffn_with_input_direct_hidden +%= hidden;
        self.semantic_ffn_with_input_direct_output +%= output;
        self.semantic_ffn_with_input_direct_input_projection_dot_ops +%= input_projection_dot_ops;
        self.semantic_ffn_with_input_direct_gate_up_dot_ops +%= gate_up_dot_ops;
        self.semantic_ffn_with_input_direct_down_dot_ops +%= down_dot_ops;
        self.semantic_ffn_with_input_direct_row_serial_dot_ops +%= row_serial_dot_ops;
        self.semantic_ffn_with_input_direct_total_row_serial_dot_ops +%= rows *% row_serial_dot_ops;
        self.semantic_ffn_with_input_direct_row_threadgroups +%= rows;
    }

    pub fn recordSemanticFfnWithInputDirectWidthParallel(self: *RuntimeProfile, rows: u32, row_tile: u32, output: u32, output_tile: u32, width_lanes: u32) void {
        const row_groups = divCeilU64(rows, row_tile);
        const output_tiles = divCeilU64(output, output_tile);
        self.semantic_ffn_with_input_direct_width_parallel_count +%= 1;
        self.semantic_ffn_with_input_direct_width_parallel_rows +%= rows;
        self.semantic_ffn_with_input_direct_width_parallel_row_tile_groups +%= row_groups;
        self.semantic_ffn_with_input_direct_width_parallel_output_tiles +%= output_tiles;
        self.semantic_ffn_with_input_direct_width_parallel_lanes +%= width_lanes;
        self.semantic_ffn_with_input_direct_width_parallel_partial_slots +%= @as(u64, rows) *% output_tiles;
    }

    pub fn recordSemanticFfnSublayerSingleDispatchRefusal(self: *RuntimeProfile, reason: SemanticFfnSublayerSingleDispatchRefusalReason) void {
        self.semantic_ffn_sublayer_single_dispatch_refusal_reasons[@intFromEnum(reason)] +%= 1;
    }

    pub fn recordSemanticFfnSublayerSingleDispatchDimRefusal(self: *RuntimeProfile, k: u32, h: u32, o: u32, cap: u32) void {
        self.recordSemanticFfnSublayerSingleDispatchRefusal(.dim);
        self.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_k = @max(self.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_k, k);
        self.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_h = @max(self.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_h, h);
        self.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_o = @max(self.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_o, o);
        self.semantic_ffn_sublayer_single_dispatch_dim_refusal_cap = @max(self.semantic_ffn_sublayer_single_dispatch_dim_refusal_cap, cap);
    }

    pub fn recordSemanticFfnSublayerSingleDispatchOutputReadRefusal(self: *RuntimeProfile) void {
        self.semantic_ffn_sublayer_single_dispatch_output_read_refusals +%= 1;
        self.recordSemanticFfnSublayerSingleDispatchRefusal(.output_read);
    }

    pub fn recordSemanticFfnSublayerSingleDispatchBlockSizeRefusal(self: *RuntimeProfile) void {
        self.semantic_ffn_sublayer_single_dispatch_block_size_refusals +%= 1;
        self.recordSemanticFfnSublayerSingleDispatchRefusal(.block_size);
    }

    pub fn recordSemanticWidthScratch(self: *RuntimeProfile, candidates: u64, scratch_bytes: u64, input_bytes: u64, product_bytes: u64, down_partial_bytes: u64, output_bytes: u64, runtime_capacity_bytes: u64) void {
        self.semantic_width_scratch_candidates +%= candidates;
        self.semantic_width_scratch_bytes = @max(self.semantic_width_scratch_bytes, scratch_bytes);
        self.semantic_width_scratch_input_bytes = @max(self.semantic_width_scratch_input_bytes, input_bytes);
        self.semantic_width_scratch_product_bytes = @max(self.semantic_width_scratch_product_bytes, product_bytes);
        self.semantic_width_scratch_down_partial_bytes = @max(self.semantic_width_scratch_down_partial_bytes, down_partial_bytes);
        self.semantic_width_scratch_output_bytes = @max(self.semantic_width_scratch_output_bytes, output_bytes);
        self.semantic_width_scratch_runtime_capacity_bytes = @max(self.semantic_width_scratch_runtime_capacity_bytes, runtime_capacity_bytes);
        if (output_bytes > 0) {
            self.semantic_width_scratch_down_partial_to_output_x1000 = @max(
                self.semantic_width_scratch_down_partial_to_output_x1000,
                down_partial_bytes *% 1000 / output_bytes,
            );
        }
    }

    pub fn recordSemanticWidthScratchRuntimeUse(self: *RuntimeProfile, bytes: u64) void {
        self.semantic_width_scratch_runtime_uses +%= 1;
        self.semantic_width_scratch_runtime_bytes = @max(self.semantic_width_scratch_runtime_bytes, bytes);
    }

    pub fn recordSemanticWidthScratchAllocation(self: *RuntimeProfile, bytes: u64) void {
        self.semantic_width_scratch_allocated_bytes = @max(self.semantic_width_scratch_allocated_bytes, bytes);
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
        try writeCountAndPerCall(jw, "qmatmul_row_chain_width_parallel_", "count", rt.qmatmul_row_chain_width_parallel_count, calls_f);
        try writeCountAndPerCall(jw, "qmatmul_row_chain_width_parallel_", "lanes", rt.qmatmul_row_chain_width_parallel_lanes, calls_f);
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
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_", "width_lane_slots", rt.semantic_ffn_sublayer_width_lane_slots, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_", "active_width_lanes", rt.semantic_ffn_sublayer_active_width_lanes, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_", "thread_lane_slots", rt.semantic_ffn_sublayer_thread_lane_slots, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_", "active_thread_lanes", rt.semantic_ffn_sublayer_active_thread_lanes, calls_f);
        if (rt.semantic_ffn_sublayer_width_lane_slots > 0) {
            try writeJsonField(
                jw,
                "semantic_ffn_sublayer_width_lane_utilization",
                @as(f64, @floatFromInt(rt.semantic_ffn_sublayer_active_width_lanes)) / @as(f64, @floatFromInt(rt.semantic_ffn_sublayer_width_lane_slots)),
            );
        }
        if (rt.semantic_ffn_sublayer_thread_lane_slots > 0) {
            try writeJsonField(
                jw,
                "semantic_ffn_sublayer_thread_lane_utilization",
                @as(f64, @floatFromInt(rt.semantic_ffn_sublayer_active_thread_lanes)) / @as(f64, @floatFromInt(rt.semantic_ffn_sublayer_thread_lane_slots)),
            );
        }
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
    if (rt.semantic_ffn_sublayer_fallback_pair_dispatches > 0 or rt.semantic_ffn_sublayer_fallback_tail_dispatches > 0) {
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_fallback_", "pair_dispatches", rt.semantic_ffn_sublayer_fallback_pair_dispatches, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_fallback_", "tail_dispatches", rt.semantic_ffn_sublayer_fallback_tail_dispatches, calls_f);
    }
    if (rt.semantic_ffn_with_input_decomposed_count > 0) {
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_decomposed_", "count", rt.semantic_ffn_with_input_decomposed_count, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_decomposed_", "dispatches", rt.semantic_ffn_with_input_decomposed_dispatches, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_decomposed_", "extra_dispatches", rt.semantic_ffn_with_input_decomposed_extra_dispatches, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_decomposed_", "row_chain_dispatches", rt.semantic_ffn_with_input_decomposed_row_chain_dispatches, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_decomposed_", "pair_dispatches", rt.semantic_ffn_with_input_decomposed_pair_dispatches, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_decomposed_", "tail_dispatches", rt.semantic_ffn_with_input_decomposed_tail_dispatches, calls_f);
    }
    if (rt.semantic_ffn_with_input_direct_count > 0) {
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_", "count", rt.semantic_ffn_with_input_direct_count, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_", "rows", rt.semantic_ffn_with_input_direct_rows, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_", "input_projection", rt.semantic_ffn_with_input_direct_input_projection, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_", "input", rt.semantic_ffn_with_input_direct_input, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_", "hidden", rt.semantic_ffn_with_input_direct_hidden, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_", "output", rt.semantic_ffn_with_input_direct_output, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_", "input_projection_dot_ops", rt.semantic_ffn_with_input_direct_input_projection_dot_ops, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_", "gate_up_dot_ops", rt.semantic_ffn_with_input_direct_gate_up_dot_ops, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_", "down_dot_ops", rt.semantic_ffn_with_input_direct_down_dot_ops, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_", "row_serial_dot_ops", rt.semantic_ffn_with_input_direct_row_serial_dot_ops, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_", "total_row_serial_dot_ops", rt.semantic_ffn_with_input_direct_total_row_serial_dot_ops, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_", "row_threadgroups", rt.semantic_ffn_with_input_direct_row_threadgroups, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_width_parallel_", "count", rt.semantic_ffn_with_input_direct_width_parallel_count, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_width_parallel_", "rows", rt.semantic_ffn_with_input_direct_width_parallel_rows, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_width_parallel_", "row_tile_groups", rt.semantic_ffn_with_input_direct_width_parallel_row_tile_groups, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_width_parallel_", "output_tiles", rt.semantic_ffn_with_input_direct_width_parallel_output_tiles, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_width_parallel_", "lanes", rt.semantic_ffn_with_input_direct_width_parallel_lanes, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_with_input_direct_width_parallel_", "partial_slots", rt.semantic_ffn_with_input_direct_width_parallel_partial_slots, calls_f);
        if (rt.semantic_ffn_with_input_direct_row_threadgroups > 0) {
            try writeJsonField(
                jw,
                "semantic_ffn_with_input_direct_total_row_serial_dot_ops_per_row_threadgroup",
                @as(f64, @floatFromInt(rt.semantic_ffn_with_input_direct_total_row_serial_dot_ops)) / @as(f64, @floatFromInt(rt.semantic_ffn_with_input_direct_row_threadgroups)),
            );
        }
    }
    if (rt.semantic_ffn_sublayer_single_dispatch_attempts > 0) {
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_single_dispatch_", "attempts", rt.semantic_ffn_sublayer_single_dispatch_attempts, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_single_dispatch_", "output_read_refusals", rt.semantic_ffn_sublayer_single_dispatch_output_read_refusals, calls_f);
        try writeCountAndPerCall(jw, "semantic_ffn_sublayer_single_dispatch_", "block_size_refusals", rt.semantic_ffn_sublayer_single_dispatch_block_size_refusals, calls_f);
        inline for (semantic_single_dispatch_refusal_fields, 0..) |field, i| {
            const count = rt.semantic_ffn_sublayer_single_dispatch_refusal_reasons[i];
            if (count != 0) {
                try writeCountAndPerCall(jw, "semantic_ffn_sublayer_single_dispatch_refused_", field.name, count, calls_f);
            }
        }
        if (rt.semantic_ffn_sublayer_single_dispatch_refusal_reasons[@intFromEnum(SemanticFfnSublayerSingleDispatchRefusalReason.dim)] > 0) {
            try writeJsonField(jw, "semantic_ffn_sublayer_single_dispatch_dim_refusal_max_k", rt.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_k);
            try writeJsonField(jw, "semantic_ffn_sublayer_single_dispatch_dim_refusal_max_h", rt.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_h);
            try writeJsonField(jw, "semantic_ffn_sublayer_single_dispatch_dim_refusal_max_o", rt.semantic_ffn_sublayer_single_dispatch_dim_refusal_max_o);
            try writeJsonField(jw, "semantic_ffn_sublayer_single_dispatch_dim_refusal_cap", rt.semantic_ffn_sublayer_single_dispatch_dim_refusal_cap);
        }
    }
    if (rt.semantic_width_scratch_bytes > 0) {
        try writeJsonField(jw, "semantic_width_scratch_candidates", rt.semantic_width_scratch_candidates);
        try writeJsonField(jw, "semantic_width_scratch_bytes", rt.semantic_width_scratch_bytes);
        try writeJsonField(jw, "semantic_width_scratch_input_bytes", rt.semantic_width_scratch_input_bytes);
        try writeJsonField(jw, "semantic_width_scratch_product_bytes", rt.semantic_width_scratch_product_bytes);
        try writeJsonField(jw, "semantic_width_scratch_down_partial_bytes", rt.semantic_width_scratch_down_partial_bytes);
        try writeJsonField(jw, "semantic_width_scratch_output_bytes", rt.semantic_width_scratch_output_bytes);
        try writeJsonField(jw, "semantic_width_scratch_down_partial_to_output", @as(f64, @floatFromInt(rt.semantic_width_scratch_down_partial_to_output_x1000)) / 1000.0);
        try writeJsonField(jw, "semantic_width_scratch_allocated_bytes", rt.semantic_width_scratch_allocated_bytes);
        try writeJsonField(jw, "semantic_width_scratch_runtime_capacity_bytes", rt.semantic_width_scratch_runtime_capacity_bytes);
        try writeJsonField(jw, "semantic_width_scratch_runtime_uses", rt.semantic_width_scratch_runtime_uses);
        try writeJsonField(jw, "semantic_width_scratch_runtime_bytes", rt.semantic_width_scratch_runtime_bytes);
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
    rt.recordSemanticFfnSublayer(128, 576, 576, 576, 512);
    rt.recordSemanticFfnSublayerFallbackDispatches(1, 2);
    rt.recordSemanticFfnWithInputDecomposed(5, 2, 1, 2);
    rt.recordSemanticFfnWithInputDirect(128, 1536, 576, 576, 576);
    rt.recordSemanticFfnWithInputDirectWidthParallel(128, 32, 576, 32, 4);
    rt.recordSemanticFfnSublayerSingleDispatchAttempt();
    rt.recordSemanticFfnSublayerSingleDispatchDimRefusal(576, 1536, 576, 1024);
    rt.recordSemanticWidthScratch(1, 14155776, 0, 786432, 14155776, 294912, 9216);
    rt.recordSemanticWidthScratchAllocation(14155776);
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
    try std.testing.expect(std.mem.indexOf(u8, out, "\"qmatmul_row_chain_width_parallel_count\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"qmatmul_row_chain_width_parallel_lanes\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_count\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_row_serial_dot_ops\":995328") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_total_row_serial_dot_ops\":127401984") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_tile_parallel_groups\":216") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_row_serial_dot_ops_per_tile_parallel_group\":4608") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_total_row_serial_dot_ops_per_tile_parallel_group\":589824") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_width_lane_slots\":524288") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_active_width_lanes\":294912") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_width_lane_utilization\":0.5625") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_thread_lane_slots\":589824") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_active_thread_lanes\":360448") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_thread_lane_utilization\":0.611111") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_fallback_pair_dispatches\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_fallback_tail_dispatches\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_decomposed_count\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_decomposed_dispatches\":5") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_decomposed_extra_dispatches\":4") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_decomposed_row_chain_dispatches\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_decomposed_pair_dispatches\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_decomposed_tail_dispatches\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_direct_count\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_direct_rows\":128") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_direct_input_projection_dot_ops\":331776") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_direct_gate_up_dot_ops\":1769472") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_direct_down_dot_ops\":884736") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_direct_row_serial_dot_ops\":2985984") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_direct_total_row_serial_dot_ops\":382205952") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_direct_row_threadgroups\":128") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_direct_total_row_serial_dot_ops_per_row_threadgroup\":2985984") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_direct_width_parallel_count\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_direct_width_parallel_rows\":128") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_direct_width_parallel_row_tile_groups\":4") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_direct_width_parallel_output_tiles\":18") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_direct_width_parallel_lanes\":4") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_with_input_direct_width_parallel_partial_slots\":2304") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_single_dispatch_attempts\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_single_dispatch_refused_dim\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_single_dispatch_dim_refusal_max_k\":576") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_single_dispatch_dim_refusal_max_h\":1536") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_single_dispatch_dim_refusal_max_o\":576") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_single_dispatch_dim_refusal_cap\":1024") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_width_scratch_candidates\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_width_scratch_bytes\":14155776") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_width_scratch_input_bytes\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_width_scratch_product_bytes\":786432") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_width_scratch_down_partial_bytes\":14155776") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_width_scratch_output_bytes\":294912") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_width_scratch_down_partial_to_output\":48") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_width_scratch_allocated_bytes\":14155776") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_width_scratch_runtime_capacity_bytes\":9216") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_width_scratch_runtime_uses\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_width_scratch_runtime_bytes\":0") != null);
}

test "RuntimeProfile serializes semantic fallback dispatch evidence without semantic kernel count" {
    var rt = RuntimeProfile{};
    rt.recordSemanticFfnSublayerFallbackDispatches(1, 2);

    var aw: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer aw.deinit();
    var jw: std.json.Stringify = .{ .writer = &aw.writer };

    try jw.beginObject();
    try writeRuntimeProfileJsonFields(rt, &jw);
    try jw.endObject();
    const out = aw.writer.buffer[0..aw.writer.end];

    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_fallback_pair_dispatches\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_fallback_tail_dispatches\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"semantic_ffn_sublayer_count\"") == null);
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
    window.recordSemanticFfnWithInputDecomposed(5, 2, 1, 2);
    window.recordSemanticFfnWithInputDirectWidthParallel(128, 32, 576, 32, 4);
    window.recordSemanticWidthScratch(1, 14155776, 0, 786432, 14155776, 294912, 9216);
    window.recordSemanticWidthScratchAllocation(14155776);
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
    try std.testing.expectEqual(@as(u64, 0), total.qmatmul_row_chain_width_parallel_count);
    try std.testing.expectEqual(@as(u64, 0), total.qmatmul_row_chain_width_parallel_lanes);
    try std.testing.expectEqual(@as(u64, 2), total.semantic_ffn_with_input_decomposed_count);
    try std.testing.expectEqual(@as(u64, 10), total.semantic_ffn_with_input_decomposed_dispatches);
    try std.testing.expectEqual(@as(u64, 8), total.semantic_ffn_with_input_decomposed_extra_dispatches);
    try std.testing.expectEqual(@as(u64, 4), total.semantic_ffn_with_input_decomposed_row_chain_dispatches);
    try std.testing.expectEqual(@as(u64, 2), total.semantic_ffn_with_input_decomposed_pair_dispatches);
    try std.testing.expectEqual(@as(u64, 4), total.semantic_ffn_with_input_decomposed_tail_dispatches);
    try std.testing.expectEqual(@as(u64, 2), total.semantic_ffn_with_input_direct_width_parallel_count);
    try std.testing.expectEqual(@as(u64, 256), total.semantic_ffn_with_input_direct_width_parallel_rows);
    try std.testing.expectEqual(@as(u64, 8), total.semantic_ffn_with_input_direct_width_parallel_row_tile_groups);
    try std.testing.expectEqual(@as(u64, 36), total.semantic_ffn_with_input_direct_width_parallel_output_tiles);
    try std.testing.expectEqual(@as(u64, 8), total.semantic_ffn_with_input_direct_width_parallel_lanes);
    try std.testing.expectEqual(@as(u64, 4608), total.semantic_ffn_with_input_direct_width_parallel_partial_slots);
    try std.testing.expectEqual(@as(u64, 2), total.semantic_width_scratch_candidates);
    try std.testing.expectEqual(@as(u64, 14155776), total.semantic_width_scratch_bytes);
    try std.testing.expectEqual(@as(u64, 0), total.semantic_width_scratch_input_bytes);
    try std.testing.expectEqual(@as(u64, 786432), total.semantic_width_scratch_product_bytes);
    try std.testing.expectEqual(@as(u64, 14155776), total.semantic_width_scratch_down_partial_bytes);
    try std.testing.expectEqual(@as(u64, 294912), total.semantic_width_scratch_output_bytes);
    try std.testing.expectEqual(@as(u64, 48000), total.semantic_width_scratch_down_partial_to_output_x1000);
    try std.testing.expectEqual(@as(u64, 14155776), total.semantic_width_scratch_allocated_bytes);
    try std.testing.expectEqual(@as(u64, 9216), total.semantic_width_scratch_runtime_capacity_bytes);
    try std.testing.expectEqual(@as(u64, 0), total.semantic_width_scratch_runtime_uses);
    try std.testing.expectEqual(@as(u64, 0), total.semantic_width_scratch_runtime_bytes);
    try std.testing.expectEqual(@as(u32, 74), total.call_count);
}

test "RuntimeProfile records qmatmul row-chain width-parallel lowering" {
    var rt = RuntimeProfile{};
    rt.recordQMatmulRowChainWidthParallelTiledSpill(128, 576, 1536, 32, 4, false, false);

    try std.testing.expectEqual(@as(u64, 1), rt.qmatmul_row_chain_tiled_count);
    try std.testing.expectEqual(@as(u64, 1), rt.qmatmul_row_chain_tiled_two_phase_count);
    try std.testing.expectEqual(@as(u64, 1), rt.qmatmul_row_chain_width_parallel_count);
    try std.testing.expectEqual(@as(u64, 4), rt.qmatmul_row_chain_width_parallel_lanes);
    try std.testing.expectEqual(@as(u64, 0), rt.qmatmul_row_chain_tiled_spilled_input);
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
