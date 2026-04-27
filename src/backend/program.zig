//! Shared helpers for compiled backend programs.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const DeviceOpTag = std.meta.Tag(backend_mod.DeviceOp);

pub const compute_op_softmax: u32 = 100;
pub const compute_op_layernorm: u32 = 101;
pub const compute_op_rmsnorm: u32 = 102;

/// Broad, model-agnostic operation families used by backend schedulers.
/// These are intentionally coarser than DeviceOp tags: a backend can reason
/// about "row kernels" or "movement kernels" without knowing which model
/// produced the program.
pub const KernelFamily = enum {
    elementwise,
    fused_elementwise,
    row,
    reduce,
    movement,
    matmul,
    qmatvec,
    qmatmul,
    rope,
    attention,
};

pub const n_kernel_families = @typeInfo(KernelFamily).@"enum".fields.len;
comptime {
    if (n_kernel_families > 64) @compileError("KernelFamilyMask stores families in u64");
}

pub const KernelFamilyMask = struct {
    bits: u64 = 0,

    pub const empty: KernelFamilyMask = .{};
    pub const all: KernelFamilyMask = blk: {
        var bits: u64 = 0;
        for (0..n_kernel_families) |i| bits |= @as(u64, 1) << @intCast(i);
        break :blk .{ .bits = bits };
    };

    pub fn init(comptime families: []const KernelFamily) KernelFamilyMask {
        var mask: KernelFamilyMask = .empty;
        inline for (families) |family| {
            mask = mask.with(family);
        }
        return mask;
    }

    pub fn with(self: KernelFamilyMask, family: KernelFamily) KernelFamilyMask {
        return .{ .bits = self.bits | bit(family) };
    }

    pub fn contains(self: KernelFamilyMask, family: KernelFamily) bool {
        return (self.bits & bit(family)) != 0;
    }

    fn bit(family: KernelFamily) u64 {
        return @as(u64, 1) << @intCast(@intFromEnum(family));
    }
};

/// Whether a scheduled region is expected to run in the backend's native
/// execution path or through the backend's semantic fallback.
pub const ExecutionClass = enum {
    backend,
    fallback,
};

/// Native kernel families a backend can lower directly.
/// This is separate from backend.Capabilities.supportsOp(): a backend may
/// support a DeviceProgram by falling back for some ops.
pub const KernelSupport = struct {
    elementwise: bool = false,
    fused_elementwise: bool = false,
    row: bool = false,
    reduce: bool = false,
    movement: bool = false,
    matmul: bool = false,
    qmatvec: bool = false,
    qmatmul: bool = false,
    rope: bool = false,
    attention: bool = false,

    pub fn fromCapabilities(capabilities: backend_mod.Capabilities) KernelSupport {
        return .{
            .fused_elementwise = capabilities.fused_elementwise,
            .matmul = capabilities.dense_matmul_f32,
            .qmatvec = capabilities.qmatmul,
            .qmatmul = capabilities.qmatmul,
            .attention = capabilities.attention.supported,
        };
    }

    pub fn supports(self: KernelSupport, family: KernelFamily) bool {
        return switch (family) {
            .elementwise => self.elementwise,
            .fused_elementwise => self.fused_elementwise,
            .row => self.row,
            .reduce => self.reduce,
            .movement => self.movement,
            .matmul => self.matmul,
            .qmatvec => self.qmatvec,
            .qmatmul => self.qmatmul,
            .rope => self.rope,
            .attention => self.attention,
        };
    }
};

/// Pure scheduling policy. It has no backend state and no model knowledge:
/// given capabilities, native kernel availability, and dispatch thresholds,
/// the same DeviceProgram always maps to the same KernelItems.
pub const SchedulePolicy = struct {
    capabilities: backend_mod.Capabilities,
    native_kernels: KernelSupport = .{},
    fine_grained: bool = false,
    min_backend_matmul_m: u32 = 16,
    min_backend_qmatmul_m: u32 = 16,

    pub fn conservative(capabilities: backend_mod.Capabilities) SchedulePolicy {
        return .{
            .capabilities = capabilities,
            .native_kernels = KernelSupport.fromCapabilities(capabilities),
        };
    }
};

/// A contiguous DeviceOp range with the same broad family and execution class.
pub const KernelItem = struct {
    family: KernelFamily,
    execution: ExecutionClass,
    start: u32,
    len: u32,
};

pub const KernelRegion = struct {
    start_item: u32,
    item_count: u32,
    op_start: u32,
    op_count: u32,
    anchor_count: u32,
};

pub const invalid_pattern_index = std.math.maxInt(u32);

pub const PatternRegion = struct {
    pattern_index: u32,
    region: KernelRegion,
};

pub const FamilyPattern = struct {
    name: []const u8,
    families: []const KernelFamily,
};

pub const ScheduleUnitKind = enum {
    item,
    pattern_region,
};

/// A logical execution unit over a KernelItem schedule. Backends can start with
/// item units, then replace selected ranges with pattern regions as they grow
/// fused lowerings. This stays model-agnostic: a pattern is just a family
/// sequence with a backend-owned lowering.
pub const ScheduleUnit = struct {
    kind: ScheduleUnitKind,
    pattern_index: u32 = invalid_pattern_index,
    start_item: u32,
    item_count: u32,
    op_start: u32,
    op_count: u32,
};

pub const RegionExecutionSummary = struct {
    units: u32 = 0,
    backend_units: u32 = 0,
    fallback_units: u32 = 0,
    ops: u32 = 0,
    backend_ops: u32 = 0,
    fallback_ops: u32 = 0,
    backend_islands: u32 = 0,
    max_backend_island_units: u32 = 0,
    max_backend_island_ops: u32 = 0,
    execution_transitions: u32 = 0,
};

/// Describes reusable anchored regions over a KernelItem schedule. This is the
/// pure planning layer for future fusion: choose anchor families (for example
/// qmatvec) and families allowed to travel with them, then emit contiguous
/// candidate regions without knowing which model produced the ops.
pub const RegionPolicy = struct {
    anchor_families: KernelFamilyMask,
    member_families: KernelFamilyMask,

    pub fn qmatvecCluster() RegionPolicy {
        return .{
            .anchor_families = KernelFamilyMask.init(&.{.qmatvec}),
            .member_families = KernelFamilyMask.init(&.{
                .elementwise,
                .fused_elementwise,
                .row,
                .movement,
                .qmatvec,
                .rope,
                .attention,
            }),
        };
    }

    pub fn qmatmulCluster() RegionPolicy {
        return .{
            .anchor_families = KernelFamilyMask.init(&.{.qmatmul}),
            .member_families = KernelFamilyMask.init(&.{
                .elementwise,
                .fused_elementwise,
                .row,
                .reduce,
                .movement,
                .matmul,
                .qmatmul,
                .rope,
                .attention,
            }),
        };
    }
};

/// A named, backend-owned lowering target over a KernelItem schedule.
///
/// StagePolicy keeps the central idea tiny: a stage is a reusable anchored
/// region with a pattern id. Model code does not know about it, and backend
/// code can progressively replace the conservative per-op walk with a fused
/// stage lowering.
pub const StagePolicy = struct {
    name: []const u8,
    pattern_index: u32,
    region_policy: RegionPolicy,
    anchors_per_stage: u32,
    min_items_per_stage: u32 = 1,
    min_ops_per_stage: u32 = 1,

    pub fn anchored(
        name: []const u8,
        pattern_index: u32,
        region_policy: RegionPolicy,
        anchors_per_stage: u32,
    ) StagePolicy {
        return .{
            .name = name,
            .pattern_index = pattern_index,
            .region_policy = region_policy,
            .anchors_per_stage = anchors_per_stage,
        };
    }
};

pub const StageCommandKind = enum {
    op,
    row_chain,
    rope_chain,

    pub fn label(self: StageCommandKind) []const u8 {
        return @tagName(self);
    }
};

pub const StageCommand = struct {
    kind: StageCommandKind,
    op_start: u32,
    op_count: u32,

    pub fn dispatchCount(self: StageCommand) u32 {
        _ = self;
        return 1;
    }
};

pub const StageCommandSummary = struct {
    commands: u32 = 0,
    ops: u32 = 0,
    estimated_dispatches: u32 = 0,
    estimated_saved_dispatches: u32 = 0,
    row_chains: u32 = 0,
    row_chain_ops: u32 = 0,
    rope_chains: u32 = 0,
    rope_chain_ops: u32 = 0,
};

pub const ProgramCommandKind = enum {
    op,
    row_chain,
    rope_chain,
    rope_batch,
    rope_store_group,
    movement_batch,
    movement_group,
    attention_chain,
    attention_store_chain,
    attention_store_group,
    rope_attention_store_chain,
    rope_attention_store_group,
    attention_batch,
    attention_group,
    elementwise_batch,
    repeat_fused_elementwise_chain,
    projection_fused_elementwise_chain,
    projection_pair_fused_elementwise_chain,
    projection_chain,
    projection_group,
    projection_cache_group,

    pub fn label(self: ProgramCommandKind) []const u8 {
        return @tagName(self);
    }

    pub fn shape(self: ProgramCommandKind) ProgramCommandShape {
        return switch (self) {
            .op,
            .row_chain,
            .rope_chain,
            .rope_batch,
            .movement_batch,
            .attention_batch,
            .repeat_fused_elementwise_chain,
            .projection_fused_elementwise_chain,
            .projection_pair_fused_elementwise_chain,
            => .{},

            .projection_chain,
            .attention_chain,
            => .{ .coverage = .anchor_sidecars },

            .projection_group,
            .rope_store_group,
            .attention_store_chain,
            .attention_store_group,
            .rope_attention_store_chain,
            .rope_attention_store_group,
            => .{
                .coverage = .anchor_sidecars,
                .advance = .explicit_indices,
            },

            .projection_cache_group => .{
                .coverage = .anchor_sidecars,
                .sidecars = .flat,
                .advance = .explicit_indices,
            },

            .movement_group,
            .attention_group,
            .elementwise_batch,
            => .{
                .coverage = .anchors_only,
                .advance = .explicit_indices,
            },
        };
    }
};

pub const ProgramCommandCoverage = enum {
    contiguous,
    anchor_sidecars,
    anchors_only,
};

pub const ProgramCommandSidecarLayout = enum {
    anchor_aligned,
    flat,
};

pub const ProgramCommandAdvance = enum {
    contiguous,
    explicit_indices,
};

pub const ProgramCommandShape = struct {
    coverage: ProgramCommandCoverage = .contiguous,
    sidecars: ProgramCommandSidecarLayout = .anchor_aligned,
    advance: ProgramCommandAdvance = .contiguous,
};

pub const ProjectionGroupKind = enum {
    qmatvec,
    qmatmul,
};

/// Pure policy for projection batching. Backends still decide whether they
/// have a native kernel for the command; this only answers "is it legal to
/// batch these independent projections and carry simple side effects?"
pub const ProjectionGroupPolicy = struct {
    kind: ProjectionGroupKind,
    max_anchors: u32 = 4,
    carry_slice_sidecars: bool = true,

    pub fn decodeQMatvec(max_anchors: u32) ProjectionGroupPolicy {
        return .{ .kind = .qmatvec, .max_anchors = max_anchors, .carry_slice_sidecars = true };
    }

    pub fn prefillQMatmul(max_anchors: u32) ProjectionGroupPolicy {
        return .{ .kind = .qmatmul, .max_anchors = max_anchors };
    }
};

pub const CommandStreamPolicy = struct {
    stage_commands: bool = true,
    qmatvec_group_size: u32 = 4,
    qmatmul_group_size: u32 = 4,
    qmatmul_sidecars: bool = true,
    qmatmul_cache_sidecars_per_anchor: u32 = 8,
    projection_rope_cache_sidecars: bool = false,
    max_rope_batch: u32 = 16,
    max_movement_batch: u32 = 16,
    max_attention_batch: u32 = 16,
    max_attention_store_batch: u32 = 4,
    max_rope_attention_store_batch: u32 = 16,
    max_elementwise_batch: u32 = 8,
    fuse_repeat_fused_elementwise: bool = true,

    pub fn fromCapabilities(capabilities: backend_mod.Capabilities) CommandStreamPolicy {
        const c = capabilities.command_stream;
        return .{
            .stage_commands = c.stage_commands,
            .qmatvec_group_size = c.qmatvec_group_size,
            .qmatmul_group_size = c.qmatmul_group_size,
            .qmatmul_sidecars = c.qmatmul_sidecars,
            .qmatmul_cache_sidecars_per_anchor = c.qmatmul_cache_sidecars_per_anchor,
            .projection_rope_cache_sidecars = c.projection_rope_cache_sidecars,
            .max_rope_batch = c.max_rope_batch,
            .max_movement_batch = c.max_movement_batch,
            .max_attention_batch = c.max_attention_batch,
            .max_attention_store_batch = c.max_attention_store_batch,
            .max_rope_attention_store_batch = c.max_rope_attention_store_batch,
            .max_elementwise_batch = c.max_elementwise_batch,
            .fuse_repeat_fused_elementwise = c.fuse_repeat_fused_elementwise,
        };
    }

    pub fn metal(qmatvec_group_size: u32, qmatmul_group_size: u32) CommandStreamPolicy {
        var policy = CommandStreamPolicy.fromCapabilities(backend_mod.Capabilities.metal);
        policy.qmatvec_group_size = qmatvec_group_size;
        policy.qmatmul_group_size = qmatmul_group_size;
        return policy;
    }

    fn projectionPolicyFor(self: CommandStreamPolicy, q: anytype) ?ProjectionGroupPolicy {
        if (q.M == 1) {
            if (self.qmatvec_group_size < 2) return null;
            return ProjectionGroupPolicy.decodeQMatvec(self.qmatvec_group_size);
        }
        if (self.qmatmul_group_size < 2) return null;
        var policy = ProjectionGroupPolicy.prefillQMatmul(self.qmatmul_group_size);
        policy.carry_slice_sidecars = self.qmatmul_sidecars;
        return policy;
    }
};

pub const ProgramCommand = struct {
    kind: ProgramCommandKind,
    op_start: u32,
    op_count: u32,
    projection_kind: ProjectionGroupKind = .qmatmul,
    anchor_count: u32 = 0,
    sidecar_count: u32 = 0,
    indices: [max_projection_group_anchors]usize = [_]usize{0} ** max_projection_group_anchors,
    sidecar_indices: [max_projection_group_anchors]?usize = [_]?usize{null} ** max_projection_group_anchors,

    pub fn op(start: usize) ProgramCommand {
        return .{
            .kind = .op,
            .op_start = @intCast(start),
            .op_count = 1,
        };
    }

    pub fn fromStageCommand(command: StageCommand) ProgramCommand {
        return .{
            .kind = switch (command.kind) {
                .op => .op,
                .row_chain => .row_chain,
                .rope_chain => .rope_chain,
            },
            .op_start = command.op_start,
            .op_count = command.op_count,
        };
    }

    pub fn fromProjectionSelection(selection: ProjectionGroupSelection) ProgramCommand {
        var command = ProgramCommand{
            .kind = .projection_group,
            .op_start = @intCast(selection.start_op),
            .op_count = @intCast(selection.end_op - selection.start_op + 1),
            .projection_kind = selection.kind,
            .anchor_count = @intCast(selection.anchor_count),
            .sidecar_count = @intCast(selection.sidecar_count),
        };
        for (selection.anchorIndices(), 0..) |idx, slot| command.indices[slot] = idx;
        for (selection.sidecarIndices(), 0..) |idx, slot| command.sidecar_indices[slot] = idx;
        return command;
    }

    pub fn contiguous(kind: ProgramCommandKind, start: usize, count: usize) ProgramCommand {
        return .{
            .kind = kind,
            .op_start = @intCast(start),
            .op_count = @intCast(count),
        };
    }

    pub fn dispatchCount(_: ProgramCommand) u32 {
        return 1;
    }

    pub fn shape(self: ProgramCommand) ProgramCommandShape {
        return self.kind.shape();
    }

    pub fn coveredOpCount(self: ProgramCommand) u32 {
        return switch (self.shape().coverage) {
            .contiguous => self.op_count,
            .anchor_sidecars => self.anchor_count + self.sidecar_count,
            .anchors_only => self.anchor_count,
        };
    }

    pub fn advanceCount(self: ProgramCommand) u32 {
        return switch (self.shape().advance) {
            .contiguous => self.op_count,
            .explicit_indices => 1,
        };
    }

    pub fn coversAnchorSidecars(self: ProgramCommand) bool {
        return self.shape().coverage == .anchor_sidecars;
    }

    pub fn coversAnchorsOnly(self: ProgramCommand) bool {
        return self.shape().coverage == .anchors_only;
    }

    pub fn hasExplicitCoverage(self: ProgramCommand) bool {
        return self.shape().coverage != .contiguous;
    }

    pub fn usesExplicitIndices(self: ProgramCommand) bool {
        return self.shape().advance == .explicit_indices;
    }

    pub fn explicitIndexSet(self: *const ProgramCommand) CommandIndexSet {
        var set = CommandIndexSet{};
        for (self.anchorIndices()) |idx| {
            _ = set.append(idx);
        }
        for (self.carriedSidecarIndices()) |maybe_idx| {
            if (maybe_idx) |idx| _ = set.append(idx);
        }
        return set;
    }

    pub fn sortedExplicitIndexSet(self: *const ProgramCommand) CommandIndexSet {
        var set = self.explicitIndexSet();
        set.sort();
        return set;
    }

    pub fn coveredIndexIterator(self: *const ProgramCommand) CommandIndexIterator {
        return CommandIndexIterator.init(self);
    }

    pub fn anchorIndices(self: *const ProgramCommand) []const usize {
        return self.indices[0..self.anchor_count];
    }

    pub fn sidecarIndices(self: *const ProgramCommand) []const ?usize {
        return self.sidecar_indices[0..self.anchor_count];
    }

    pub fn flatSidecarIndices(self: *const ProgramCommand) []const ?usize {
        return self.sidecar_indices[0..self.sidecar_count];
    }

    pub fn carriedSidecarIndices(self: *const ProgramCommand) []const ?usize {
        return switch (self.shape().sidecars) {
            .anchor_aligned => self.sidecarIndices(),
            .flat => self.flatSidecarIndices(),
        };
    }
};

pub const max_command_indices = max_projection_group_anchors * 2;

pub const CommandIndexIterator = struct {
    mode: enum { contiguous, explicit },
    next_index: usize = 0,
    end_index: usize = 0,
    explicit: CommandIndexSet = .{},
    explicit_pos: usize = 0,

    pub fn init(command: *const ProgramCommand) CommandIndexIterator {
        if (command.hasExplicitCoverage()) {
            return .{
                .mode = .explicit,
                .explicit = command.sortedExplicitIndexSet(),
            };
        }

        const start: usize = @intCast(command.op_start);
        return .{
            .mode = .contiguous,
            .next_index = start,
            .end_index = start + @as(usize, command.op_count),
        };
    }

    pub fn next(self: *CommandIndexIterator) ?usize {
        return switch (self.mode) {
            .contiguous => {
                if (self.next_index >= self.end_index) return null;
                const idx = self.next_index;
                self.next_index += 1;
                return idx;
            },
            .explicit => {
                if (self.explicit_pos >= self.explicit.count) return null;
                const idx = self.explicit.indices[self.explicit_pos];
                self.explicit_pos += 1;
                return idx;
            },
        };
    }

    pub fn remainingCount(self: *const CommandIndexIterator) usize {
        return switch (self.mode) {
            .contiguous => self.end_index - self.next_index,
            .explicit => self.explicit.count - self.explicit_pos,
        };
    }
};

pub const CommandIndexSet = struct {
    indices: [max_command_indices]usize = undefined,
    count: usize = 0,

    pub fn append(self: *CommandIndexSet, idx: usize) bool {
        for (self.indices[0..self.count]) |existing| {
            if (existing == idx) return true;
        }
        if (self.count >= self.indices.len) return false;
        self.indices[self.count] = idx;
        self.count += 1;
        return true;
    }

    pub fn sort(self: *CommandIndexSet) void {
        std.mem.sort(usize, self.indices[0..self.count], {}, std.sort.asc(usize));
    }

    pub fn slice(self: *const CommandIndexSet) []const usize {
        return self.indices[0..self.count];
    }
};

pub const ProgramCommandSummary = struct {
    commands: u32 = 0,
    covered_ops: u32 = 0,
    estimated_dispatches: u32 = 0,
    estimated_saved_dispatches: u32 = 0,
    op_commands: u32 = 0,
    row_chains: u32 = 0,
    rope_chains: u32 = 0,
    rope_batches: u32 = 0,
    rope_store_groups: u32 = 0,
    rope_store_group_ops: u32 = 0,
    rope_store_group_sidecars: u32 = 0,
    movement_batches: u32 = 0,
    movement_groups: u32 = 0,
    movement_group_ops: u32 = 0,
    attention_chains: u32 = 0,
    attention_chain_sidecars: u32 = 0,
    attention_store_chains: u32 = 0,
    attention_store_chain_sidecars: u32 = 0,
    attention_store_groups: u32 = 0,
    attention_store_group_ops: u32 = 0,
    attention_store_group_sidecars: u32 = 0,
    rope_attention_store_chains: u32 = 0,
    rope_attention_store_chain_sidecars: u32 = 0,
    rope_attention_store_groups: u32 = 0,
    rope_attention_store_group_ops: u32 = 0,
    rope_attention_store_group_sidecars: u32 = 0,
    attention_batches: u32 = 0,
    attention_groups: u32 = 0,
    attention_group_ops: u32 = 0,
    elementwise_batches: u32 = 0,
    elementwise_ops: u32 = 0,
    repeat_fused_elementwise_chains: u32 = 0,
    projection_fused_elementwise_chains: u32 = 0,
    projection_pair_fused_elementwise_chains: u32 = 0,
    projection_chains: u32 = 0,
    projection_chain_sidecars: u32 = 0,
    projection_groups: u32 = 0,
    projection_anchors: u32 = 0,
    projection_sidecars: u32 = 0,
    projection_cache_groups: u32 = 0,
    projection_cache_anchors: u32 = 0,
    projection_cache_sidecars: u32 = 0,
    max_projection_span_ops: u32 = 0,

    pub fn add(self: *ProgramCommandSummary, other: ProgramCommandSummary) void {
        inline for (@typeInfo(ProgramCommandSummary).@"struct".fields) |field| {
            if (comptime std.mem.eql(u8, field.name, "max_projection_span_ops")) {
                @field(self.*, field.name) = @max(@field(self.*, field.name), @field(other, field.name));
            } else {
                @field(self.*, field.name) += @field(other, field.name);
            }
        }
    }
};

pub const AttentionStoreGroupCandidateSummary = struct {
    anchors: u32 = 0,
    first_store_missing: u32 = 0,
    candidate_attentions: u32 = 0,
    geometry_rejects: u32 = 0,
    hoist_rejects: u32 = 0,
    selected_conflict_rejects: u32 = 0,
    no_store_rejects: u32 = 0,
    pair_conflict_rejects: u32 = 0,
    formed_groups: u32 = 0,
    grouped_anchors: u32 = 0,
    max_group_anchors: u32 = 0,
};

pub const RopeAttentionStoreGroupCandidateSummary = struct {
    anchors: u32 = 0,
    first_pair_missing: u32 = 0,
    candidate_ropes: u32 = 0,
    geometry_rejects: u32 = 0,
    pair_missing_rejects: u32 = 0,
    before_emit_rejects: u32 = 0,
    delay_rejects: u32 = 0,
    attention_hoist_rejects: u32 = 0,
    sidecar_hoist_rejects: u32 = 0,
    selected_conflict_rejects: u32 = 0,
    formed_groups: u32 = 0,
    grouped_pairs: u32 = 0,
    max_group_pairs: u32 = 0,
};

pub const EarlyRopeAttentionStoreGroupCandidateSummary = struct {
    anchors: u32 = 0,
    first_pair_missing: u32 = 0,
    candidate_ropes: u32 = 0,
    geometry_rejects: u32 = 0,
    pair_missing_rejects: u32 = 0,
    rope_hoist_rejects: u32 = 0,
    attention_hoist_rejects: u32 = 0,
    selected_conflict_rejects: u32 = 0,
    formed_groups: u32 = 0,
    grouped_pairs: u32 = 0,
    max_group_pairs: u32 = 0,
};

pub const RopeStoreGroupCandidateSummary = struct {
    anchors: u32 = 0,
    first_pair_missing: u32 = 0,
    candidate_ropes: u32 = 0,
    geometry_rejects: u32 = 0,
    pair_missing_rejects: u32 = 0,
    hoist_rejects: u32 = 0,
    selected_conflict_rejects: u32 = 0,
    external_user_rejects: u32 = 0,
    formed_groups: u32 = 0,
    grouped_pairs: u32 = 0,
    max_group_pairs: u32 = 0,
};

pub const ProjectionSidecarSummary = struct {
    anchors: u32 = 0,
    immediate_sidecars: u32 = 0,
    compatible_sidecars: u32 = 0,
    primary_elidable_sidecars: u32 = 0,
    primary_required_sidecars: u32 = 0,
    slice_sidecars: u32 = 0,
    elementwise_sidecars: u32 = 0,
    fused_elementwise_sidecars: u32 = 0,
    incompatible_sidecars: u32 = 0,
};

pub const ProjectionRopeCacheSummary = struct {
    anchors: u32 = 0,
    rope_store_pairs: u32 = 0,
    compatible_pairs: u32 = 0,
    tile_pair_pairs: u32 = 0,
    rope_materializations: u32 = 0,
    materialization_attention_fusion_skips: u32 = 0,
};

pub const max_projection_group_anchors = 32;

pub const BufferSpan = struct {
    buf: u16,
    start: u64,
    end: u64,

    pub fn overlaps(self: BufferSpan, other: BufferSpan) bool {
        return self.buf == other.buf and self.start < other.end and other.start < self.end;
    }
};

const max_access_spans = 16;

const OpAccessSpans = struct {
    reads: [max_access_spans]BufferSpan = undefined,
    writes: [max_access_spans]BufferSpan = undefined,
    read_count: u8 = 0,
    write_count: u8 = 0,
    read_overflow: bool = false,
    write_overflow: bool = false,

    fn addRead(self: *OpAccessSpans, span: BufferSpan) void {
        if (span.start == span.end) return;
        if (self.read_count >= max_access_spans) {
            self.read_overflow = true;
            return;
        }
        self.reads[self.read_count] = span;
        self.read_count += 1;
    }

    fn addWrite(self: *OpAccessSpans, span: BufferSpan) void {
        if (span.start == span.end) return;
        if (self.write_count >= max_access_spans) {
            self.write_overflow = true;
            return;
        }
        self.writes[self.write_count] = span;
        self.write_count += 1;
    }

    fn readSpans(self: *const OpAccessSpans) []const BufferSpan {
        return self.reads[0..self.read_count];
    }

    fn writeSpans(self: *const OpAccessSpans) []const BufferSpan {
        return self.writes[0..self.write_count];
    }
};

pub const ProjectionGroup = struct {
    kind: ProjectionGroupKind,
    start_op: u32,
    op_count: u32,
    anchor_count: u32,
    sidecar_count: u32,

    pub fn coveredOpCount(self: ProjectionGroup) u32 {
        return self.anchor_count + self.sidecar_count;
    }

    pub fn dispatchCount(_: ProjectionGroup) u32 {
        return 1;
    }
};

pub const ProjectionGroupSummary = struct {
    groups: u32 = 0,
    anchors: u32 = 0,
    sidecars: u32 = 0,
    covered_ops: u32 = 0,
    estimated_dispatches: u32 = 0,
    estimated_saved_dispatches: u32 = 0,
    max_span_ops: u32 = 0,
};

pub const ProjectionGroupSelection = struct {
    kind: ProjectionGroupKind,
    start_op: usize,
    end_op: usize,
    anchor_count: usize,
    sidecar_count: usize,
    indices: [max_projection_group_anchors]usize = undefined,
    sidecar_indices: [max_projection_group_anchors]?usize = [_]?usize{null} ** max_projection_group_anchors,

    pub fn anchorIndices(self: *const ProjectionGroupSelection) []const usize {
        return self.indices[0..self.anchor_count];
    }

    pub fn sidecarIndices(self: *const ProjectionGroupSelection) []const ?usize {
        return self.sidecar_indices[0..self.anchor_count];
    }

    pub fn toGroup(self: ProjectionGroupSelection) ProjectionGroup {
        return .{
            .kind = self.kind,
            .start_op = @intCast(self.start_op),
            .op_count = @intCast(self.end_op - self.start_op + 1),
            .anchor_count = @intCast(self.anchor_count),
            .sidecar_count = @intCast(self.sidecar_count),
        };
    }
};

pub fn buildStageCommands(
    alloc: std.mem.Allocator,
    ops: []const backend_mod.DeviceOp,
) ![]StageCommand {
    var commands: std.ArrayListUnmanaged(StageCommand) = .empty;
    errdefer commands.deinit(alloc);

    var i: usize = 0;
    while (i < ops.len) {
        if (findStageCommand(ops, i)) |command| {
            try commands.append(alloc, command);
            i += command.op_count;
            continue;
        }

        try commands.append(alloc, .{
            .kind = .op,
            .op_start = @intCast(i),
            .op_count = 1,
        });
        i += 1;
    }

    return commands.toOwnedSlice(alloc);
}

pub fn findStageCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
) ?StageCommand {
    if (start >= ops.len) return null;

    if (start + 2 < ops.len and isRmsnormScaleChain(ops[start], ops[start + 1], ops[start + 2])) {
        return .{
            .kind = .row_chain,
            .op_start = @intCast(start),
            .op_count = 3,
        };
    }

    if (start + 1 < ops.len and isRopeSliceAssignChain(ops[start], ops[start + 1])) {
        return .{
            .kind = .rope_chain,
            .op_start = @intCast(start),
            .op_count = 2,
        };
    }

    return null;
}

pub fn summarizeStageCommands(commands: []const StageCommand) StageCommandSummary {
    var summary = StageCommandSummary{ .commands = @intCast(commands.len) };
    for (commands) |command| {
        summary.ops += command.op_count;
        const dispatches = command.dispatchCount();
        summary.estimated_dispatches += dispatches;
        if (command.op_count > dispatches) {
            summary.estimated_saved_dispatches += command.op_count - dispatches;
        }
        switch (command.kind) {
            .op => {},
            .row_chain => {
                summary.row_chains += 1;
                summary.row_chain_ops += command.op_count;
            },
            .rope_chain => {
                summary.rope_chains += 1;
                summary.rope_chain_ops += command.op_count;
            },
        }
    }
    return summary;
}

pub fn buildProjectionGroups(
    alloc: std.mem.Allocator,
    ops: []const backend_mod.DeviceOp,
    policy: ProjectionGroupPolicy,
) ![]ProjectionGroup {
    var groups: std.ArrayListUnmanaged(ProjectionGroup) = .empty;
    errdefer groups.deinit(alloc);

    const max_anchors = @min(policy.max_anchors, max_projection_group_anchors);
    if (max_anchors < 2) return groups.toOwnedSlice(alloc);

    const used = try alloc.alloc(bool, ops.len);
    defer alloc.free(used);
    @memset(used, false);

    var i: usize = 0;
    while (i < ops.len) : (i += 1) {
        if (used[i]) continue;
        const selection = findProjectionGroup(ops, i, policy, used) orelse continue;
        for (selection.anchorIndices()) |idx| {
            used[idx] = true;
        }
        for (selection.sidecarIndices()) |maybe_idx| {
            if (maybe_idx) |idx| used[idx] = true;
        }
        try groups.append(alloc, selection.toGroup());
    }

    return groups.toOwnedSlice(alloc);
}

pub fn findProjectionGroup(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: ProjectionGroupPolicy,
    used: ?[]const bool,
) ?ProjectionGroupSelection {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const max_anchors = @min(policy.max_anchors, max_projection_group_anchors);
    if (max_anchors < 2) return null;

    const first = switch (ops[start]) {
        .qmatmul => |q| q,
        else => return null,
    };
    if (!projectionMatchesPolicy(first, policy)) return null;

    var selection = ProjectionGroupSelection{
        .kind = policy.kind,
        .start_op = start,
        .end_op = start,
        .anchor_count = 1,
        .sidecar_count = 0,
    };
    selection.indices[0] = start;

    var scan = start + 1;
    while (scan < ops.len and selection.anchor_count < max_anchors) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const q = switch (ops[scan]) {
            .qmatmul => |q| q,
            else => continue,
        };
        if (!projectionMatchesPolicy(q, policy)) continue;
        if (!canHoistProjectionTo(ops, start, scan, q)) continue;
        if (projectionConflictsSelected(ops, selection.anchorIndices(), q)) continue;
        selection.indices[selection.anchor_count] = scan;
        selection.anchor_count += 1;
    }

    if (selection.anchor_count < 2) return null;

    for (selection.anchorIndices(), 0..) |idx, slot| {
        selection.end_op = @max(selection.end_op, idx);
        if (!policy.carry_slice_sidecars) continue;
        if (idx + 1 >= ops.len) continue;
        if (used) |used_ops| {
            if (idx + 1 >= used_ops.len or used_ops[idx + 1]) continue;
        }
        const q = ops[idx].qmatmul;
        const sidecar = ops[idx + 1];
        if (!projectionSidecarMatchesPolicy(policy, q, sidecar)) continue;

        var candidate = selection;
        candidate.sidecar_indices[slot] = idx + 1;
        candidate.sidecar_count += 1;
        candidate.end_op = @max(candidate.end_op, idx + 1);
        if (!canHoistProjectionSidecarToGroup(ops, selection.start_op, idx + 1, q, sidecar, &candidate)) continue;
        selection = candidate;
    }

    return selection;
}

fn findProjectionCacheGroupCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    const projection_policy = policy.projectionPolicyFor(ops[start].qmatmul) orelse return null;
    if (!projection_policy.carry_slice_sidecars) return null;
    const selection = findProjectionGroup(ops, start, projection_policy, used) orelse return null;

    var command = ProgramCommand{
        .kind = .projection_cache_group,
        .op_start = @intCast(selection.start_op),
        .op_count = @intCast(selection.end_op - selection.start_op + 1),
        .projection_kind = selection.kind,
        .anchor_count = @intCast(selection.anchor_count),
    };
    var per_anchor_sidecars = [_]u32{0} ** max_projection_group_anchors;
    for (selection.anchorIndices(), 0..) |idx, slot| command.indices[slot] = idx;
    for (selection.sidecarIndices(), 0..) |maybe_idx, slot| {
        const idx = maybe_idx orelse continue;
        const sa = switch (ops[idx]) {
            .slice_assign => |sa| sa,
            else => return null,
        };
        if (!appendProjectionCacheSidecar(&command, idx, selection.start_op)) return null;
        per_anchor_sidecars[slot] += 1;
        if (!projectionCacheSidecarsShareSink(ops, &command, slot, sa)) return null;
    }

    const initial_sidecars = command.sidecar_count;
    const max_sidecars_per_anchor = @max(1, policy.qmatmul_cache_sidecars_per_anchor);
    var scan = start + 1;
    while (scan < ops.len and command.sidecar_count < max_projection_group_anchors) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        if (commandContainsIndex(&command, scan)) continue;
        if (scan > selection.end_op and ops[scan] == .qmatmul) break;
        if (policy.projection_rope_cache_sidecars and ops[scan] == .rope) {
            const rr = ops[scan].rope;
            if (scan + 1 < ops.len) store: {
                if (used) |used_ops| {
                    if (scan + 1 >= used_ops.len or used_ops[scan + 1]) break :store;
                }
                const sa = switch (ops[scan + 1]) {
                    .slice_assign => |sa| sa,
                    else => break :store,
                };
                const slot = projectionCacheRopeStoreAnchorSlot(ops, &command, rr, sa) orelse break :store;
                if (per_anchor_sidecars[slot] >= max_sidecars_per_anchor) break :store;
                if (!projectionCacheSidecarsShareSink(ops, &command, slot, sa)) break :store;

                var candidate = command;
                if (!appendProjectionCacheRopeStoreSidecar(&candidate, scan, scan + 1, selection.start_op)) break;
                if (!canHoistProjectionCacheRopeStoreToGroup(ops, selection.start_op, scan, scan + 1, ops[command.indices[slot]].qmatmul, rr, sa, &candidate, executed)) break :store;
                if (projectionCacheSidecarConflictsSelected(ops, &command, sa)) break :store;
                if (projectionCacheRopeOutputHasExternalUsers(ops, scan, scan + 1)) break :store;

                command = candidate;
                per_anchor_sidecars[slot] += 1;
                scan += 1;
                continue;
            }

            const slot = projectionCacheRopeAnchorSlot(ops, &command, rr) orelse continue;
            if (per_anchor_sidecars[slot] >= max_sidecars_per_anchor) continue;
            if (projectionCacheRopeWouldBlockAttentionFusion(ops, scan, used, executed)) continue;
            var candidate = command;
            if (!appendProjectionCacheRopeSidecar(&candidate, scan, selection.start_op)) break;
            if (!canHoistProjectionCacheRopeToGroup(ops, selection.start_op, scan, ops[command.indices[slot]].qmatmul, rr, &candidate, executed)) continue;
            if (projectionCacheRopeConflictsSelected(ops, &command, rr)) continue;

            command = candidate;
            per_anchor_sidecars[slot] += 1;
            continue;
        }

        const sa = switch (ops[scan]) {
            .slice_assign => |sa| sa,
            else => continue,
        };
        const slot = projectionCacheSidecarAnchorSlot(ops, &command, sa) orelse continue;
        if (per_anchor_sidecars[slot] >= max_sidecars_per_anchor) continue;
        if (!projectionCacheSidecarsShareSink(ops, &command, slot, sa)) continue;

        var candidate = command;
        if (!appendProjectionCacheSidecar(&candidate, scan, selection.start_op)) break;
        if (!canHoistProjectionCacheSidecarToGroup(ops, selection.start_op, scan, ops[command.indices[slot]].qmatmul, sa, &candidate, executed)) continue;
        if (projectionCacheSidecarConflictsSelected(ops, &command, sa)) continue;

        command = candidate;
        per_anchor_sidecars[slot] += 1;
    }

    return if (command.sidecar_count > initial_sidecars) command else null;
}

fn appendProjectionCacheSidecar(command: *ProgramCommand, sidecar_index: usize, group_start: usize) bool {
    if (command.sidecar_count >= max_projection_group_anchors) return false;
    command.sidecar_indices[command.sidecar_count] = sidecar_index;
    command.sidecar_count += 1;
    command.op_count = @intCast(@max(
        group_start + @as(usize, command.op_count),
        sidecar_index + 1,
    ) - group_start);
    return true;
}

fn appendProjectionCacheRopeStoreSidecar(command: *ProgramCommand, rope_index: usize, sidecar_index: usize, group_start: usize) bool {
    if (command.sidecar_count + 2 > max_projection_group_anchors) return false;
    command.sidecar_indices[command.sidecar_count] = rope_index;
    command.sidecar_count += 1;
    command.sidecar_indices[command.sidecar_count] = sidecar_index;
    command.sidecar_count += 1;
    command.op_count = @intCast(@max(
        group_start + @as(usize, command.op_count),
        sidecar_index + 1,
    ) - group_start);
    return true;
}

fn appendProjectionCacheRopeSidecar(command: *ProgramCommand, rope_index: usize, group_start: usize) bool {
    if (command.sidecar_count >= max_projection_group_anchors) return false;
    command.sidecar_indices[command.sidecar_count] = rope_index;
    command.sidecar_count += 1;
    command.op_count = @intCast(@max(
        group_start + @as(usize, command.op_count),
        rope_index + 1,
    ) - group_start);
    return true;
}

fn projectionCacheSidecarAnchorSlot(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    sa: anytype,
) ?usize {
    for (command.anchorIndices(), 0..) |idx, slot| {
        if (idx >= ops.len) return null;
        const q = switch (ops[idx]) {
            .qmatmul => |q| q,
            else => return null,
        };
        if (projectionSidecarCompatible(q, .{ .slice_assign = sa })) return slot;
    }
    return null;
}

fn projectionCacheRopeStoreAnchorSlot(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    rr: anytype,
    sa: anytype,
) ?usize {
    for (command.anchorIndices(), 0..) |idx, slot| {
        if (idx >= ops.len) return null;
        const q = switch (ops[idx]) {
            .qmatmul => |q| q,
            else => return null,
        };
        if (projectionRopeStoreSidecarCompatible(q, rr, sa)) return slot;
    }
    return null;
}

fn projectionCacheRopeAnchorSlot(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    rr: anytype,
) ?usize {
    for (command.anchorIndices(), 0..) |idx, slot| {
        if (idx >= ops.len) return null;
        const q = switch (ops[idx]) {
            .qmatmul => |q| q,
            else => return null,
        };
        if (qmatvecRopeSidecarCompatible(q, rr)) return slot;
    }
    return null;
}

fn projectionCacheRopeWouldBlockAttentionFusion(
    ops: []const backend_mod.DeviceOp,
    rope_index: usize,
    used: ?[]const bool,
    executed: ?[]const bool,
) bool {
    return findDelayableRopeAttentionStorePair(ops, rope_index, used, executed) != null or
        findRopeAttentionStorePair(ops, rope_index, used, executed) != null;
}

fn projectionCacheSidecarsShareSink(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    anchor_slot: usize,
    sa: anytype,
) bool {
    var i: usize = 0;
    while (i < command.sidecar_count) : (i += 1) {
        const idx = command.sidecar_indices[i] orelse continue;
        switch (ops[idx]) {
            .slice_assign => |selected| {
                const selected_slot = projectionCacheSidecarAnchorSlot(ops, command, selected) orelse continue;
                if (selected_slot == anchor_slot and selected.dst != sa.dst) return false;
            },
            .rope => |rr| {
                if (i + 1 < command.sidecar_count) {
                    const sidecar_idx = command.sidecar_indices[i + 1] orelse return false;
                    const selected = switch (ops[sidecar_idx]) {
                        .slice_assign => |selected| selected,
                        else => continue,
                    };
                    const selected_slot = projectionCacheRopeStoreAnchorSlot(ops, command, rr, selected) orelse continue;
                    if (selected_slot == anchor_slot and selected.dst != sa.dst) return false;
                    i += 1;
                }
            },
            else => return false,
        }
    }
    return true;
}

fn projectionCacheSidecarConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    sa: anytype,
) bool {
    const sidecar_access = opAccessSpans(.{ .slice_assign = sa });
    for (command.anchorIndices()) |idx| {
        for (sidecar_access.writeSpans()) |write| {
            if (opReadsSpan(ops[idx], write)) return true;
        }
        if (sidecar_access.write_overflow and opReadsBuffer(ops[idx], sa.dst)) return true;
    }
    var i: usize = 0;
    while (i < command.sidecar_count) : (i += 1) {
        const idx = command.sidecar_indices[i] orelse continue;
        switch (ops[idx]) {
            .slice_assign => |selected| {
                if (sliceAssignWritesMayOverlap(selected, sa)) return true;
            },
            .rope => |rr| {
                if (opAccessConflicts(.{ .rope = rr }, .{ .slice_assign = sa })) return true;
                if (i + 1 < command.sidecar_count) {
                    const sidecar_idx = command.sidecar_indices[i + 1] orelse return true;
                    const selected = switch (ops[sidecar_idx]) {
                        .slice_assign => |selected| selected,
                        else => continue,
                    };
                    if (projectionCacheRopeStoreAnchorSlot(ops, command, rr, selected) == null) continue;
                    if (sliceAssignWritesMayOverlap(selected, sa)) return true;
                    i += 1;
                }
            },
            else => return true,
        }
    }
    return false;
}

fn projectionCacheRopeConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    rr: anytype,
) bool {
    const rope_op: backend_mod.DeviceOp = .{ .rope = rr };
    for (command.anchorIndices()) |idx| {
        if (idx >= ops.len) return true;
        if (ops[idx] == .qmatmul and projectionRopeSidecarCompatible(ops[idx].qmatmul, rr)) continue;
        if (opAccessConflicts(ops[idx], rope_op)) return true;
    }
    var i: usize = 0;
    while (i < command.sidecar_count) : (i += 1) {
        const idx = command.sidecar_indices[i] orelse continue;
        if (idx >= ops.len) return true;
        if (opAccessConflicts(ops[idx], rope_op)) return true;
        if (ops[idx] == .rope and i + 1 < command.sidecar_count) {
            const sidecar_idx = command.sidecar_indices[i + 1] orelse continue;
            if (sidecar_idx >= ops.len) return true;
            const selected = switch (ops[sidecar_idx]) {
                .slice_assign => |selected| selected,
                else => continue,
            };
            if (projectionCacheRopeStoreAnchorSlot(ops, command, ops[idx].rope, selected) != null) {
                if (opAccessConflicts(ops[sidecar_idx], rope_op)) return true;
                i += 1;
            }
        }
    }
    return false;
}

fn canHoistProjectionCacheSidecarToGroup(
    ops: []const backend_mod.DeviceOp,
    group_start: usize,
    sidecar_index: usize,
    q: anytype,
    sa: anytype,
    command: *const ProgramCommand,
    executed: ?[]const bool,
) bool {
    const sidecar_access = opAccessSpans(.{ .slice_assign = sa });
    for (ops[group_start..sidecar_index], group_start..) |op, idx| {
        if (commandContainsIndex(command, idx)) continue;
        if (executed) |executed_ops| {
            if (idx < executed_ops.len and executed_ops[idx]) continue;
        }
        for (sidecar_access.readSpans()) |read| {
            if (projectionWriteCoversRead(q, read)) continue;
            if (opWritesSpan(op, read)) return false;
        }
        for (sidecar_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return false;
        }
        if (sidecar_access.read_overflow or sidecar_access.write_overflow) {
            if (opAccessConflicts(op, .{ .slice_assign = sa })) return false;
        }
    }
    return true;
}

fn canHoistProjectionCacheRopeToGroup(
    ops: []const backend_mod.DeviceOp,
    group_start: usize,
    rope_index: usize,
    q: anytype,
    rr: anytype,
    command: *const ProgramCommand,
    executed: ?[]const bool,
) bool {
    const rope_access = opAccessSpans(.{ .rope = rr });
    for (ops[group_start..rope_index], group_start..) |op, idx| {
        if (commandContainsIndex(command, idx)) continue;
        if (executed) |executed_ops| {
            if (idx < executed_ops.len and executed_ops[idx]) continue;
        }
        for (rope_access.readSpans()) |read| {
            if (projectionWriteCoversRead(q, read)) continue;
            if (opWritesSpan(op, read)) return false;
        }
        for (rope_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return false;
        }
        if (rope_access.read_overflow and (opWritesBuffer(op, rr.src) or opWritesBuffer(op, rr.cos_sin))) return false;
        if (rope_access.write_overflow and opTouchesBuffer(op, rr.dst)) return false;
    }
    return true;
}

fn canHoistProjectionCacheRopeStoreToGroup(
    ops: []const backend_mod.DeviceOp,
    group_start: usize,
    rope_index: usize,
    sidecar_index: usize,
    q: anytype,
    rr: anytype,
    sa: anytype,
    command: *const ProgramCommand,
    executed: ?[]const bool,
) bool {
    const rope_access = opAccessSpans(.{ .rope = rr });
    for (ops[group_start..rope_index], group_start..) |op, idx| {
        if (commandContainsIndex(command, idx)) continue;
        if (executed) |executed_ops| {
            if (idx < executed_ops.len and executed_ops[idx]) continue;
        }
        for (rope_access.readSpans()) |read| {
            if (projectionWriteCoversRead(q, read)) continue;
            if (opWritesSpan(op, read)) return false;
        }
        if (rope_access.read_overflow and (opWritesBuffer(op, rr.src) or opWritesBuffer(op, rr.cos_sin))) return false;
    }

    const sidecar_access = opAccessSpans(.{ .slice_assign = sa });
    for (ops[group_start..sidecar_index], group_start..) |op, idx| {
        if (commandContainsIndex(command, idx)) continue;
        if (executed) |executed_ops| {
            if (idx < executed_ops.len and executed_ops[idx]) continue;
        }
        for (sidecar_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return false;
        }
        if (sidecar_access.write_overflow and opTouchesBuffer(op, sa.dst)) return false;
    }

    return true;
}

fn projectionCacheRopeOutputHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    rope_index: usize,
    sidecar_index: usize,
) bool {
    var command = ProgramCommand{
        .kind = .rope_store_group,
        .op_start = @intCast(rope_index),
        .op_count = 1,
    };
    appendRopeStorePair(&command, rope_index, sidecar_index, rope_index);
    return ropeStoreGroupOutputsHaveExternalUsers(ops, &command);
}

pub fn summarizeProjectionGroups(groups: []const ProjectionGroup) ProjectionGroupSummary {
    var summary = ProjectionGroupSummary{ .groups = @intCast(groups.len) };
    for (groups) |group| {
        const covered = group.coveredOpCount();
        const dispatches = group.dispatchCount();
        summary.anchors += group.anchor_count;
        summary.sidecars += group.sidecar_count;
        summary.covered_ops += covered;
        summary.estimated_dispatches += dispatches;
        summary.max_span_ops = @max(summary.max_span_ops, group.op_count);
        if (covered > dispatches) {
            summary.estimated_saved_dispatches += covered - dispatches;
        }
    }
    return summary;
}

pub fn summarizeProjectionSidecars(ops: []const backend_mod.DeviceOp) ProjectionSidecarSummary {
    var summary = ProjectionSidecarSummary{};
    for (ops, 0..) |op, i| {
        const q = switch (op) {
            .qmatmul => |q| q,
            else => continue,
        };
        summary.anchors += 1;
        if (i + 1 >= ops.len) continue;

        const sidecar = ops[i + 1];
        switch (sidecar) {
            .slice_assign, .elementwise, .fused_elementwise => {},
            else => continue,
        }
        summary.immediate_sidecars += 1;
        if (!projectionSidecarCompatible(q, sidecar)) {
            summary.incompatible_sidecars += 1;
            continue;
        }

        summary.compatible_sidecars += 1;
        if (projectionPrimaryOutputHasExternalUsers(ops, i, i + 1)) {
            summary.primary_required_sidecars += 1;
        } else {
            summary.primary_elidable_sidecars += 1;
        }
        switch (sidecar) {
            .slice_assign => summary.slice_sidecars += 1,
            .elementwise => summary.elementwise_sidecars += 1,
            .fused_elementwise => summary.fused_elementwise_sidecars += 1,
            else => unreachable,
        }
    }
    return summary;
}

pub fn summarizeProjectionRopeCacheSidecars(ops: []const backend_mod.DeviceOp, tile_cols: u32) ProjectionRopeCacheSummary {
    var summary = ProjectionRopeCacheSummary{};
    for (ops) |op| {
        const q = switch (op) {
            .qmatmul => |q| q,
            else => continue,
        };
        summary.anchors += 1;

        for (ops[0..ops.len -| 1], 0..) |candidate, i| {
            const rr = switch (candidate) {
                .rope => |rr| rr,
                else => continue,
            };
            const sa = switch (ops[i + 1]) {
                .slice_assign => |sa| sa,
                else => continue,
            };
            if (rr.src != q.dst) continue;
            summary.rope_store_pairs += 1;
            if (!projectionRopeStoreSidecarCompatible(q, rr, sa)) continue;
            summary.compatible_pairs += 1;
            if (qmatmulRopeStoreTilePairCompatible(q, rr, sa, tile_cols)) {
                summary.tile_pair_pairs += 1;
            }
        }

        for (ops, 0..) |candidate, i| {
            const rr = switch (candidate) {
                .rope => |rr| rr,
                else => continue,
            };
            if (!projectionRopeSidecarCompatible(q, rr)) continue;
            if (i + 1 < ops.len and ops[i + 1] == .slice_assign and projectionRopeStoreSidecarCompatible(q, rr, ops[i + 1].slice_assign)) continue;
            summary.rope_materializations += 1;
            if (projectionCacheRopeWouldBlockAttentionFusion(ops, i, null, null)) {
                summary.materialization_attention_fusion_skips += 1;
            }
        }
    }
    return summary;
}

pub fn buildProgramCommands(
    alloc: std.mem.Allocator,
    ops: []const backend_mod.DeviceOp,
    policy: CommandStreamPolicy,
) ![]ProgramCommand {
    var commands: std.ArrayListUnmanaged(ProgramCommand) = .empty;
    errdefer commands.deinit(alloc);
    var pending: std.ArrayListUnmanaged(PendingProgramCommand) = .empty;
    defer pending.deinit(alloc);

    const used = try alloc.alloc(bool, ops.len);
    defer alloc.free(used);
    @memset(used, false);
    const executed = try alloc.alloc(bool, ops.len);
    defer alloc.free(executed);
    @memset(executed, false);

    var i: usize = 0;
    while (i < ops.len) {
        try emitPendingProgramCommandsAt(alloc, &commands, &pending, used, executed, i);
        if (used[i]) {
            i += 1;
            continue;
        }

        if (findDelayedProgramCommand(ops, i, policy, used, executed)) |command| {
            markProgramCommandUsed(used, command);
            try pending.append(alloc, .{ .emit_at = command.op_start, .command = command });
            i += 1;
            continue;
        }

        if (findProgramCommandWithExecuted(ops, i, policy, used, executed)) |command| {
            markProgramCommandUsed(used, command);
            markProgramCommandUsed(executed, command);
            try commands.append(alloc, command);
            i += command.advanceCount();
            continue;
        }

        const command = ProgramCommand.op(i);
        markProgramCommandUsed(used, command);
        markProgramCommandUsed(executed, command);
        try commands.append(alloc, command);
        i += 1;
    }
    while (pending.items.len != 0) {
        const pending_command = pending.orderedRemove(0);
        try commands.append(alloc, pending_command.command);
    }

    return commands.toOwnedSlice(alloc);
}

fn findDelayedProgramCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    return findProgramCommandIn(delayed_program_command_finders, ops, start, policy, used, executed);
}

const PendingProgramCommand = struct {
    emit_at: u32,
    command: ProgramCommand,
};

fn emitPendingProgramCommandsAt(
    alloc: std.mem.Allocator,
    commands: *std.ArrayListUnmanaged(ProgramCommand),
    pending: *std.ArrayListUnmanaged(PendingProgramCommand),
    used: []bool,
    executed: []bool,
    index: usize,
) !void {
    var p: usize = 0;
    while (p < pending.items.len) {
        if (pending.items[p].emit_at != index) {
            p += 1;
            continue;
        }
        const pending_command = pending.orderedRemove(p);
        markProgramCommandUsed(used, pending_command.command);
        markProgramCommandUsed(executed, pending_command.command);
        try commands.append(alloc, pending_command.command);
    }
}

pub fn findProgramCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
) ?ProgramCommand {
    return findProgramCommandWithExecuted(ops, start, policy, used, null);
}

fn findProgramCommandWithExecuted(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }

    return findProgramCommandIn(program_command_finders, ops, start, policy, used, executed);
}

fn findProgramCommandIn(
    comptime finders: anytype,
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    inline for (finders) |finder| {
        if (finder.run(ops, start, policy, used, executed)) |command| return command;
    }
    return null;
}

const ProgramCommandFinderFn = *const fn ([]const backend_mod.DeviceOp, usize, CommandStreamPolicy, ?[]const bool, ?[]const bool) ?ProgramCommand;

const ProgramCommandFinderFeature = enum {
    always,
    stage_commands,
    elementwise_batch,
    repeat_fused_elementwise,
    rope_batch,
    movement_batch,
    attention_batch,
    rope_attention_store_batch,

    pub fn enabled(self: ProgramCommandFinderFeature, policy: CommandStreamPolicy) bool {
        return switch (self) {
            .always => true,
            .stage_commands => policy.stage_commands,
            .elementwise_batch => policy.max_elementwise_batch >= 2,
            .repeat_fused_elementwise => policy.fuse_repeat_fused_elementwise,
            .rope_batch => policy.max_rope_batch >= 2,
            .movement_batch => policy.max_movement_batch >= 2,
            .attention_batch => policy.max_attention_batch >= 2,
            .rope_attention_store_batch => policy.max_rope_attention_store_batch >= 2,
        };
    }
};

const ProgramCommandFinder = struct {
    name: []const u8,
    start_tag: ?DeviceOpTag = null,
    feature: ProgramCommandFinderFeature = .always,
    find: ProgramCommandFinderFn,

    fn run(
        self: ProgramCommandFinder,
        ops: []const backend_mod.DeviceOp,
        start: usize,
        policy: CommandStreamPolicy,
        used: ?[]const bool,
        executed: ?[]const bool,
    ) ?ProgramCommand {
        if (self.start_tag) |tag| {
            if (!opIs(ops, start, tag)) return null;
        }
        if (!self.feature.enabled(policy)) return null;
        return self.find(ops, start, policy, used, executed);
    }
};

const program_command_finders = [_]ProgramCommandFinder{
    .{ .name = "projection_pair_fused_elementwise_chain", .start_tag = .qmatmul, .find = finderUsedOnly(findProjectionPairFusedElementwiseChainCommand) },
    .{ .name = "projection_fused_elementwise_chain", .start_tag = .qmatmul, .find = finderUsedOnly(findProjectionFusedElementwiseChainCommand) },
    .{ .name = "projection_cache_group", .start_tag = .qmatmul, .find = findProjectionCacheGroupCommand },
    .{ .name = "projection_group", .start_tag = .qmatmul, .find = findProjectionGroupAt },
    .{ .name = "projection_chain", .start_tag = .qmatmul, .find = finderUsedOnly(findProjectionChainCommand) },
    .{ .name = "elementwise_batch", .start_tag = .elementwise, .feature = .elementwise_batch, .find = finderPolicyUsed(findElementwiseBatchCommand) },
    .{ .name = "repeat_fused_elementwise_chain", .start_tag = .repeat, .feature = .repeat_fused_elementwise, .find = finderUsedOnly(findRepeatFusedElementwiseCommand) },
    .{ .name = "rope_store_group", .start_tag = .rope, .feature = .rope_batch, .find = findRopeStoreGroupCommand },
    .{ .name = "stage_command", .feature = .stage_commands, .find = findStageProgramCommandAt },
    .{ .name = "contiguous_batch", .find = finderPolicyUsed(findContiguousBatchCommand) },
    .{ .name = "rope_attention_store_group", .start_tag = .rope, .feature = .rope_attention_store_batch, .find = findRopeAttentionStoreGroupCommand },
    .{ .name = "rope_attention_store_chain", .start_tag = .rope, .find = finderUsedExecuted(findRopeAttentionStoreChainCommand) },
    .{ .name = "attention_chain", .start_tag = .slice_assign, .find = finderUsedOnly(findAttentionChainCommand) },
    .{ .name = "attention_store_group", .start_tag = .attention, .feature = .attention_batch, .find = findAttentionStoreGroupCommand },
    .{ .name = "attention_store_chain", .start_tag = .attention, .find = finderUsedOnly(findAttentionStoreChainCommand) },
    .{ .name = "movement_group", .start_tag = .slice_assign, .feature = .movement_batch, .find = finderPolicyUsed(findMovementGroupCommand) },
    .{ .name = "attention_group", .start_tag = .attention, .feature = .attention_batch, .find = finderPolicyUsed(findAttentionGroupCommand) },
};

const delayed_program_command_finders = [_]ProgramCommandFinder{
    .{ .name = "delayed_rope_attention_store_group", .start_tag = .rope, .find = findDelayedRopeAttentionStoreGroupCommand },
};

fn requireUniqueFinderNames(comptime label: []const u8, comptime finders: anytype) void {
    inline for (finders, 0..) |finder, i| {
        if (finder.name.len == 0) @compileError("empty " ++ label ++ " program command finder name");
        inline for (finders[0..i]) |previous| {
            if (std.mem.eql(u8, finder.name, previous.name)) {
                @compileError("duplicate " ++ label ++ " program command finder name: " ++ finder.name);
            }
        }
    }
}

comptime {
    requireUniqueFinderNames("normal", program_command_finders);
    requireUniqueFinderNames("delayed", delayed_program_command_finders);
}

fn opIs(ops: []const backend_mod.DeviceOp, start: usize, tag: DeviceOpTag) bool {
    if (start >= ops.len) return false;
    return std.meta.activeTag(ops[start]) == tag;
}

fn findProjectionGroupAt(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    _ = executed;
    const q = switch (ops[start]) {
        .qmatmul => |q| q,
        else => return null,
    };
    const projection_policy = policy.projectionPolicyFor(q) orelse return null;
    const selection = findProjectionGroup(ops, start, projection_policy, used) orelse return null;
    return ProgramCommand.fromProjectionSelection(selection);
}

fn findStageProgramCommandAt(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    _ = executed;
    _ = policy;
    const stage_command = findStageCommand(ops, start) orelse return null;
    if (commandRangeTouchesUsed(stage_command.op_start, stage_command.op_count, used)) return null;
    return ProgramCommand.fromStageCommand(stage_command);
}

fn finderUsedOnly(comptime find: anytype) ProgramCommandFinderFn {
    return struct {
        fn run(
            ops: []const backend_mod.DeviceOp,
            start: usize,
            policy: CommandStreamPolicy,
            used: ?[]const bool,
            executed: ?[]const bool,
        ) ?ProgramCommand {
            _ = policy;
            _ = executed;
            return find(ops, start, used);
        }
    }.run;
}

fn finderPolicyUsed(comptime find: anytype) ProgramCommandFinderFn {
    return struct {
        fn run(
            ops: []const backend_mod.DeviceOp,
            start: usize,
            policy: CommandStreamPolicy,
            used: ?[]const bool,
            executed: ?[]const bool,
        ) ?ProgramCommand {
            _ = executed;
            return find(ops, start, policy, used);
        }
    }.run;
}

fn finderUsedExecuted(comptime find: anytype) ProgramCommandFinderFn {
    return struct {
        fn run(
            ops: []const backend_mod.DeviceOp,
            start: usize,
            policy: CommandStreamPolicy,
            used: ?[]const bool,
            executed: ?[]const bool,
        ) ?ProgramCommand {
            _ = policy;
            return find(ops, start, used, executed);
        }
    }.run;
}

fn findProjectionPairFusedElementwiseChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start + 5 >= ops.len) return null;
    if (commandRangeTouchesUsed(@intCast(start), 6, used)) return null;
    const gate = switch (ops[start]) {
        .qmatmul => |q| q,
        else => return null,
    };
    const first = switch (ops[start + 1]) {
        .fused_elementwise => |fe| fe,
        else => return null,
    };
    const rp = switch (ops[start + 2]) {
        .repeat => |rp| rp,
        else => return null,
    };
    const second = switch (ops[start + 3]) {
        .fused_elementwise => |fe| fe,
        else => return null,
    };
    const up = switch (ops[start + 4]) {
        .qmatmul => |q| q,
        else => return null,
    };
    const product = switch (ops[start + 5]) {
        .elementwise => |e| e,
        else => return null,
    };
    if (!projectionPairFusedElementwiseChainCompatible(gate, first, rp, second, up, product)) return null;
    if (projectionPairFusedElementwiseChainHasExternalUsers(ops, start)) return null;
    return ProgramCommand.contiguous(.projection_pair_fused_elementwise_chain, start, 6);
}

fn findProjectionFusedElementwiseChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start + 3 >= ops.len) return null;
    if (commandRangeTouchesUsed(@intCast(start), 4, used)) return null;
    const q = switch (ops[start]) {
        .qmatmul => |q| q,
        else => return null,
    };
    const first = switch (ops[start + 1]) {
        .fused_elementwise => |fe| fe,
        else => return null,
    };
    const rp = switch (ops[start + 2]) {
        .repeat => |rp| rp,
        else => return null,
    };
    const second = switch (ops[start + 3]) {
        .fused_elementwise => |fe| fe,
        else => return null,
    };
    if (!projectionFusedElementwiseChainCompatible(q, first, rp, second)) return null;
    if (projectionFusedElementwiseChainHasExternalUsers(ops, start)) return null;
    return ProgramCommand.contiguous(.projection_fused_elementwise_chain, start, 4);
}

fn findRepeatFusedElementwiseCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start + 1 >= ops.len) return null;
    if (commandRangeTouchesUsed(@intCast(start), 2, used)) return null;
    const rp = switch (ops[start]) {
        .repeat => |rp| rp,
        else => return null,
    };
    const fe = switch (ops[start + 1]) {
        .fused_elementwise => |fe| fe,
        else => return null,
    };
    if (!repeatFusedElementwiseCompatible(rp, fe)) return null;
    if (repeatOutputHasExternalUsers(ops, start, start + 1)) return null;
    return ProgramCommand.contiguous(.repeat_fused_elementwise_chain, start, 2);
}

fn findAttentionChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start + 1 >= ops.len) return null;
    if (commandRangeTouchesUsed(@intCast(start), 2, used)) return null;
    const sa = switch (ops[start]) {
        .slice_assign => |sa| sa,
        else => return null,
    };
    const att = switch (ops[start + 1]) {
        .attention => |att| att,
        else => return null,
    };
    if (attentionSliceAssignOperand(sa, att) == null) return null;
    if (attentionHasFusableStoreSidecar(ops, start + 1, used)) return null;

    var command = ProgramCommand{
        .kind = .attention_chain,
        .op_start = @intCast(start),
        .op_count = 2,
        .anchor_count = 1,
        .sidecar_count = 1,
    };
    command.indices[0] = start + 1;
    command.sidecar_indices[0] = start;
    return command;
}

fn attentionHasFusableStoreSidecar(
    ops: []const backend_mod.DeviceOp,
    attention_index: usize,
    used: ?[]const bool,
) bool {
    if (attention_index >= ops.len) return false;
    if (used) |used_ops| {
        if (attention_index >= used_ops.len or used_ops[attention_index]) return false;
    }
    const att = switch (ops[attention_index]) {
        .attention => |att| att,
        else => return false,
    };

    var scan = attention_index + 1;
    while (scan < ops.len) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const sa = switch (ops[scan]) {
            .slice_assign => |sa| sa,
            else => continue,
        };
        if (sa.src != att.dst) continue;
        if (!attentionSliceStoreCompatible(att, sa)) continue;
        if (!canFuseAttentionStoreSidecar(ops, attention_index, scan, sa)) continue;
        return true;
    }
    return false;
}

const RopeStorePair = struct {
    sidecar_index: usize,
};

fn findRopeStoreGroupCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const max_pairs: usize = @intCast(@min(policy.max_rope_batch, max_projection_group_anchors));
    if (max_pairs < 2) return null;

    const first_pair = findRopeStorePair(ops, start, used) orelse return null;
    const first_rope = ops[start].rope;
    const first_sa = ops[first_pair.sidecar_index].slice_assign;

    var command = ProgramCommand{
        .kind = .rope_store_group,
        .op_start = @intCast(start),
        .op_count = 1,
        .anchor_count = 0,
        .sidecar_count = 0,
    };
    appendRopeStorePair(&command, start, first_pair.sidecar_index, start);

    var scan = start + 1;
    while (scan < ops.len and command.anchor_count < max_pairs) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const rr = switch (ops[scan]) {
            .rope => |rr| rr,
            else => continue,
        };
        const pair = findRopeStorePair(ops, scan, used) orelse continue;
        const sa = ops[pair.sidecar_index].slice_assign;
        if (!ropeStoreGroupCompatible(first_rope, first_sa, rr, sa)) continue;

        var candidate = command;
        if (candidate.anchor_count + 1 > max_projection_group_anchors) break;
        appendRopeStorePair(&candidate, scan, pair.sidecar_index, start);
        if (!canHoistRopeStorePairToGroup(ops, start, scan, pair.sidecar_index, rr, sa, &candidate, executed)) continue;
        if (ropeStorePairConflictsSelected(ops, &command, scan, sa)) continue;

        command = candidate;
    }

    if (ropeStoreGroupOutputsHaveExternalUsers(ops, &command)) return null;
    return if (command.anchor_count >= 2) command else null;
}

fn findRopeStorePair(
    ops: []const backend_mod.DeviceOp,
    rope_index: usize,
    used: ?[]const bool,
) ?RopeStorePair {
    if (rope_index + 1 >= ops.len) return null;
    if (used) |used_ops| {
        if (rope_index >= used_ops.len or used_ops[rope_index]) return null;
        if (rope_index + 1 >= used_ops.len or used_ops[rope_index + 1]) return null;
    }
    if (!isRopeSliceAssignChain(ops[rope_index], ops[rope_index + 1])) return null;
    return .{ .sidecar_index = rope_index + 1 };
}

fn appendRopeStorePair(
    command: *ProgramCommand,
    rope_index: usize,
    sidecar_index: usize,
    group_start: usize,
) void {
    const slot = command.anchor_count;
    command.indices[slot] = rope_index;
    command.sidecar_indices[slot] = sidecar_index;
    command.anchor_count += 1;
    command.sidecar_count += 1;
    command.op_count = @intCast(@max(
        group_start + @as(usize, command.op_count),
        sidecar_index + 1,
    ) - group_start);
}

fn canHoistRopeStorePairToGroup(
    ops: []const backend_mod.DeviceOp,
    group_start: usize,
    rope_index: usize,
    sidecar_index: usize,
    rr: anytype,
    sa: anytype,
    command: *const ProgramCommand,
    executed: ?[]const bool,
) bool {
    const rope_access = opAccessSpans(.{ .rope = rr });
    for (ops[group_start..rope_index], group_start..) |op, idx| {
        if (commandContainsIndex(command, idx)) continue;
        if (executed) |executed_ops| {
            if (idx < executed_ops.len and executed_ops[idx]) continue;
        }
        for (rope_access.readSpans()) |read| {
            if (opWritesSpan(op, read)) return false;
        }
        if (rope_access.read_overflow and (opWritesBuffer(op, rr.src) or opWritesBuffer(op, rr.cos_sin))) return false;
    }

    const sidecar_access = opAccessSpans(.{ .slice_assign = sa });
    for (ops[group_start..sidecar_index], group_start..) |op, idx| {
        if (commandContainsIndex(command, idx)) continue;
        if (executed) |executed_ops| {
            if (idx < executed_ops.len and executed_ops[idx]) continue;
        }
        for (sidecar_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return false;
        }
        if (sidecar_access.write_overflow and opTouchesBuffer(op, sa.dst)) return false;
    }

    return true;
}

fn ropeStorePairConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    rope_index: usize,
    sa: anytype,
) bool {
    const rope_op = ops[rope_index];
    const sidecar_op: backend_mod.DeviceOp = .{ .slice_assign = sa };
    const sidecar_access = opAccessSpans(sidecar_op);
    for (command.anchorIndices()) |idx| {
        for (sidecar_access.writeSpans()) |write| {
            if (opReadsSpan(ops[idx], write)) return true;
        }
        if (sidecar_access.write_overflow and opReadsBuffer(ops[idx], sa.dst)) return true;
    }
    for (command.sidecarIndices()) |maybe_idx| {
        const idx = maybe_idx orelse continue;
        const selected_sa = switch (ops[idx]) {
            .slice_assign => |selected| selected,
            else => return true,
        };
        if (sliceAssignWritesMayOverlap(selected_sa, sa)) return true;
        const selected_access = opAccessSpans(ops[idx]);
        for (selected_access.writeSpans()) |write| {
            if (opReadsSpan(rope_op, write)) return true;
        }
        if (selected_access.write_overflow and opReadsBuffer(rope_op, selected_sa.dst)) return true;
    }
    return false;
}

fn ropeStoreGroupOutputsHaveExternalUsers(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
) bool {
    for (command.anchorIndices()) |rope_index| {
        const rr = switch (ops[rope_index]) {
            .rope => |rr| rr,
            else => return true,
        };
        const rope_access = opAccessSpans(.{ .rope = rr });
        var live_writes = [_]bool{false} ** max_access_spans;
        for (rope_access.writeSpans(), 0..) |_, slot| live_writes[slot] = true;
        var overflow_live = rope_access.write_overflow;

        for (ops[rope_index + 1 ..], rope_index + 1..) |op, idx| {
            if (commandContainsIndex(command, idx)) continue;
            for (rope_access.writeSpans(), 0..) |write, slot| {
                if (!live_writes[slot]) continue;
                if (opReadsSpan(op, write)) return true;
            }
            if (overflow_live and opReadsBuffer(op, rr.dst)) return true;

            var any_live = false;
            for (rope_access.writeSpans(), 0..) |write, slot| {
                if (!live_writes[slot]) continue;
                if (opWritesCoverSpan(op, write)) {
                    live_writes[slot] = false;
                } else {
                    any_live = true;
                }
            }
            if (overflow_live and opWritesBuffer(op, rr.dst)) overflow_live = false;
            if (!any_live and !overflow_live) break;
        }
    }
    return false;
}

pub fn projectionPrimaryOutputHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    q_index: usize,
    sidecar_index: usize,
) bool {
    const sidecars = [_]?usize{sidecar_index};
    return projectionPrimaryOutputHasExternalUsersExcept(ops, q_index, sidecars[0..]);
}

pub fn projectionPrimaryOutputHasExternalUsersExcept(
    ops: []const backend_mod.DeviceOp,
    q_index: usize,
    sidecar_indices: []const ?usize,
) bool {
    if (q_index >= ops.len) return true;
    const q = switch (ops[q_index]) {
        .qmatmul => |q| q,
        else => return true,
    };
    for (sidecar_indices) |maybe_idx| {
        const idx = maybe_idx orelse continue;
        if (idx >= ops.len or idx <= q_index) return true;
        if (opReadsBuffer(ops[idx], q.dst) and !projectionSidecarCompatible(q, ops[idx]) and !projectionRopeSidecarOpCompatible(q, ops[idx])) return true;
    }

    const q_access = opAccessSpans(.{ .qmatmul = q });
    var live_writes = [_]bool{false} ** max_access_spans;
    for (q_access.writeSpans(), 0..) |_, slot| live_writes[slot] = true;
    var overflow_live = q_access.write_overflow;

    var scan = q_index + 1;
    while (scan < ops.len) : (scan += 1) {
        const op = ops[scan];
        if (!optionalIndexContains(sidecar_indices, scan)) {
            for (q_access.writeSpans(), 0..) |write, slot| {
                if (!live_writes[slot]) continue;
                if (opReadsSpan(op, write)) return true;
            }
            if (overflow_live and opReadsBuffer(op, q.dst)) return true;
        }

        var any_live = false;
        for (q_access.writeSpans(), 0..) |write, slot| {
            if (!live_writes[slot]) continue;
            if (opWritesCoverSpan(op, write)) {
                live_writes[slot] = false;
            } else {
                any_live = true;
            }
        }
        if (overflow_live and opWritesBuffer(op, q.dst)) overflow_live = false;
        if (!any_live and !overflow_live) break;
    }

    return false;
}

fn optionalIndexContains(indices: []const ?usize, candidate: usize) bool {
    for (indices) |maybe_idx| {
        if (maybe_idx) |idx| {
            if (idx == candidate) return true;
        }
    }
    return false;
}

pub fn repeatFusedElementwiseCompatible(rp: anytype, fe: anytype) bool {
    if (rp.src == rp.dst) return false;
    if (fe.src == rp.dst) return false;

    var found_secondary = false;
    for (fe.steps) |step| {
        if (!step.op.isBinary() or step.secondary_buf != rp.dst) continue;
        if (step.secondary_offset < rp.dst_offset) return false;
        const rel = step.secondary_offset - rp.dst_offset;
        if (@as(u64, rel) + @as(u64, fe.n) > @as(u64, rp.n)) return false;
        found_secondary = true;
    }
    return found_secondary;
}

pub fn projectionFusedElementwiseChainCompatible(q: anytype, first: anytype, rp: anytype, second: anytype) bool {
    if (!qmatmulFusedElementwiseSidecarCompatible(q, first)) return false;
    if (!repeatFusedElementwiseCompatible(rp, second)) return false;
    if (rp.dst == q.dst) return false;
    if (first.dst == q.dst) return false;
    if (first.dst == rp.dst) return false;
    if (second.src != first.dst or second.src_offset != first.dst_offset) return false;
    if (second.n != first.n) return false;
    if (second.dst == q.dst or second.dst == first.dst or second.dst == rp.dst) return false;

    var reads_primary = false;
    for (second.steps) |step| {
        if (!step.op.isBinary()) continue;
        if (step.secondary_buf == first.dst) return false;
        if (step.secondary_buf == q.dst) {
            if (step.secondary_offset != q.dst_offset) return false;
            reads_primary = true;
        }
    }
    return reads_primary;
}

fn qmatmulPairGeometryCompatible(a: anytype, b: anytype) bool {
    return a.M != 1 and
        a.M == b.M and
        a.N == b.N and
        a.K == b.K and
        a.input == b.input and
        a.input_offset == b.input_offset and
        a.input_row_stride == b.input_row_stride and
        qmatmulDstRowStride(a) == a.N and
        qmatmulDstRowStride(b) == b.N;
}

pub fn projectionPairFusedElementwiseChainCompatible(
    gate: anytype,
    first: anytype,
    rp: anytype,
    second: anytype,
    up: anytype,
    product: anytype,
) bool {
    if (!projectionFusedElementwiseChainCompatible(gate, first, rp, second)) return false;
    if (!qmatmulPairGeometryCompatible(gate, up)) return false;
    if (product.op != .mul) return false;
    if (product.n != gate.M * gate.N or product.n != second.n) return false;
    const second_is_src0 = product.src0 == second.dst and product.src0_offset == second.dst_offset;
    const second_is_src1 = product.src1 == second.dst and product.src1_offset == second.dst_offset;
    const up_is_src0 = product.src0 == up.dst and product.src0_offset == up.dst_offset;
    const up_is_src1 = product.src1 == up.dst and product.src1_offset == up.dst_offset;
    if (!((second_is_src0 and up_is_src1) or (second_is_src1 and up_is_src0))) return false;
    if (product.dst == gate.dst or product.dst == first.dst or product.dst == second.dst or product.dst == up.dst) return false;
    return true;
}

fn spanHasExternalReadAfter(
    ops: []const backend_mod.DeviceOp,
    producer_index: usize,
    included_start: usize,
    included_end: usize,
    span: BufferSpan,
) bool {
    var scan = producer_index + 1;
    while (scan < ops.len) : (scan += 1) {
        const included = scan >= included_start and scan < included_end;
        if (!included and opReadsSpan(ops[scan], span)) return true;
        if (opWritesCoverSpan(ops[scan], span)) break;
    }
    return false;
}

pub fn projectionPairFusedElementwiseChainHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    q_index: usize,
) bool {
    if (q_index + 5 >= ops.len) return true;
    const gate = switch (ops[q_index]) {
        .qmatmul => |q| q,
        else => return true,
    };
    const first = switch (ops[q_index + 1]) {
        .fused_elementwise => |fe| fe,
        else => return true,
    };
    const rp = switch (ops[q_index + 2]) {
        .repeat => |rp| rp,
        else => return true,
    };
    const second = switch (ops[q_index + 3]) {
        .fused_elementwise => |fe| fe,
        else => return true,
    };
    const up = switch (ops[q_index + 4]) {
        .qmatmul => |q| q,
        else => return true,
    };
    const product = switch (ops[q_index + 5]) {
        .elementwise => |e| e,
        else => return true,
    };
    if (!projectionPairFusedElementwiseChainCompatible(gate, first, rp, second, up, product)) return true;

    const included_start = q_index + 1;
    const included_end = q_index + 6;
    if (spanHasExternalReadAfter(ops, q_index, included_start, included_end, bufferSpan(gate.dst, gate.dst_offset, gate.M * gate.N))) return true;
    if (spanHasExternalReadAfter(ops, q_index + 1, included_start, included_end, bufferSpan(first.dst, first.dst_offset, first.n))) return true;
    if (spanHasExternalReadAfter(ops, q_index + 2, included_start, included_end, bufferSpan(rp.dst, rp.dst_offset, rp.n))) return true;
    if (spanHasExternalReadAfter(ops, q_index + 3, included_start, included_end, bufferSpan(second.dst, second.dst_offset, second.n))) return true;
    if (spanHasExternalReadAfter(ops, q_index + 4, included_start, included_end, bufferSpan(up.dst, up.dst_offset, up.M * up.N))) return true;
    return false;
}

pub fn projectionFusedElementwiseChainHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    q_index: usize,
) bool {
    if (q_index + 3 >= ops.len) return true;
    const q = switch (ops[q_index]) {
        .qmatmul => |q| q,
        else => return true,
    };
    const first = switch (ops[q_index + 1]) {
        .fused_elementwise => |fe| fe,
        else => return true,
    };
    const rp = switch (ops[q_index + 2]) {
        .repeat => |rp| rp,
        else => return true,
    };
    const second = switch (ops[q_index + 3]) {
        .fused_elementwise => |fe| fe,
        else => return true,
    };
    if (!projectionFusedElementwiseChainCompatible(q, first, rp, second)) return true;

    const included_start = q_index + 1;
    const included_end = q_index + 4;
    const q_write = bufferSpan(q.dst, q.dst_offset, q.M * q.N);
    const first_write = bufferSpan(first.dst, first.dst_offset, first.n);
    const repeat_write = bufferSpan(rp.dst, rp.dst_offset, rp.n);
    if (spanHasExternalReadAfter(ops, q_index, included_start, included_end, q_write)) return true;
    if (spanHasExternalReadAfter(ops, q_index + 1, included_start, included_end, first_write)) return true;
    if (spanHasExternalReadAfter(ops, q_index + 2, included_start, included_end, repeat_write)) return true;
    return false;
}

pub fn repeatOutputHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    repeat_index: usize,
    fused_index: usize,
) bool {
    if (repeat_index >= ops.len or fused_index >= ops.len) return true;
    if (fused_index <= repeat_index) return true;
    const rp = switch (ops[repeat_index]) {
        .repeat => |rp| rp,
        else => return true,
    };
    const fe = switch (ops[fused_index]) {
        .fused_elementwise => |fe| fe,
        else => return true,
    };
    if (!repeatFusedElementwiseCompatible(rp, fe)) return true;

    const repeat_write = bufferSpan(rp.dst, rp.dst_offset, rp.n);

    var scan = repeat_index + 1;
    while (scan < ops.len) : (scan += 1) {
        const op = ops[scan];
        if (scan != fused_index) {
            if (opReadsSpan(op, repeat_write)) return true;
        }
        if (opWritesCoverSpan(op, repeat_write)) break;
    }

    return false;
}

pub fn summarizeRopeStoreGroupCandidates(
    ops: []const backend_mod.DeviceOp,
    policy: CommandStreamPolicy,
) RopeStoreGroupCandidateSummary {
    var summary = RopeStoreGroupCandidateSummary{};
    const max_pairs: usize = @intCast(@min(policy.max_rope_batch, max_projection_group_anchors));
    if (max_pairs < 2) return summary;

    for (ops, 0..) |op, start| {
        const first_rope = switch (op) {
            .rope => |rr| rr,
            else => continue,
        };
        summary.anchors += 1;

        const first_pair = findRopeStorePair(ops, start, null) orelse {
            summary.first_pair_missing += 1;
            continue;
        };
        const first_sa = ops[first_pair.sidecar_index].slice_assign;
        var command = ProgramCommand{
            .kind = .rope_store_group,
            .op_start = @intCast(start),
            .op_count = 1,
            .anchor_count = 0,
            .sidecar_count = 0,
        };
        appendRopeStorePair(&command, start, first_pair.sidecar_index, start);

        var scan = start + 1;
        while (scan < ops.len and command.anchor_count < max_pairs) : (scan += 1) {
            const rr = switch (ops[scan]) {
                .rope => |rr| rr,
                else => continue,
            };
            summary.candidate_ropes += 1;
            const pair = findRopeStorePair(ops, scan, null) orelse {
                summary.pair_missing_rejects += 1;
                continue;
            };
            const sa = ops[pair.sidecar_index].slice_assign;
            if (!ropeStoreGroupCompatible(first_rope, first_sa, rr, sa)) {
                summary.geometry_rejects += 1;
                continue;
            }

            var candidate = command;
            if (candidate.anchor_count + 1 > max_projection_group_anchors) break;
            appendRopeStorePair(&candidate, scan, pair.sidecar_index, start);
            if (!canHoistRopeStorePairToGroup(ops, start, scan, pair.sidecar_index, rr, sa, &candidate, null)) {
                summary.hoist_rejects += 1;
                continue;
            }
            if (ropeStorePairConflictsSelected(ops, &command, scan, sa)) {
                summary.selected_conflict_rejects += 1;
                continue;
            }
            command = candidate;
        }

        if (command.anchor_count < 2) continue;
        if (ropeStoreGroupOutputsHaveExternalUsers(ops, &command)) {
            summary.external_user_rejects += 1;
            continue;
        }
        summary.formed_groups += 1;
        summary.grouped_pairs += command.anchor_count;
        summary.max_group_pairs = @max(summary.max_group_pairs, command.anchor_count);
    }

    return summary;
}

fn findAttentionStoreChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const att = switch (ops[start]) {
        .attention => |att| att,
        else => return null,
    };

    var sidecar_idx: ?usize = null;
    var scan = start + 1;
    while (scan < ops.len) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const sa = switch (ops[scan]) {
            .slice_assign => |sa| sa,
            else => continue,
        };
        if (!attentionSliceStoreCompatible(att, sa)) continue;
        if (!canFuseAttentionStoreSidecar(ops, start, scan, sa)) continue;
        sidecar_idx = scan;
        break;
    }
    const found = sidecar_idx orelse return null;

    var command = ProgramCommand{
        .kind = .attention_store_chain,
        .op_start = @intCast(start),
        .op_count = @intCast(found - start + 1),
        .anchor_count = 1,
        .sidecar_count = 1,
    };
    command.indices[0] = start;
    command.sidecar_indices[0] = found;
    return command;
}

fn findAttentionStoreGroupCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const first = switch (ops[start]) {
        .attention => |att| att,
        else => return null,
    };

    const max_ops: usize = @intCast(@min(policy.max_attention_store_batch, max_projection_group_anchors));
    if (max_ops < 2) return null;

    var command = ProgramCommand{
        .kind = .attention_store_group,
        .op_start = @intCast(start),
        .op_count = 1,
        .anchor_count = 0,
        .sidecar_count = 0,
    };
    if (!appendAttentionStorePair(ops, &command, start, first, start, used)) return null;

    var scan = start + 1;
    while (scan < ops.len and command.anchor_count < max_ops) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const next = switch (ops[scan]) {
            .attention => |att| att,
            else => continue,
        };
        if (!attentionGeometryCompatible(first, next)) continue;
        if (!canHoistAttentionForStoreGroup(ops, start, scan, next, &command, executed)) continue;
        if (opConflictsSelected(ops, command.anchorIndices(), .{ .attention = next })) continue;
        if (!appendAttentionStorePair(ops, &command, scan, next, start, used)) continue;
    }

    return if (command.anchor_count >= 2) command else null;
}

fn findRopeAttentionStoreChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const rr = switch (ops[start]) {
        .rope => |rr| rr,
        else => return null,
    };

    var scan = start + 1;
    while (scan < ops.len) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const att = switch (ops[scan]) {
            .attention => |att| att,
            else => continue,
        };
        if (!ropeAttentionCompatible(rr, att)) continue;

        var command = ProgramCommand{
            .kind = .rope_attention_store_chain,
            .op_start = @intCast(start),
            .op_count = @intCast(scan - start + 1),
            .anchor_count = 2,
            .sidecar_count = 0,
        };
        command.indices[0] = start;
        command.indices[1] = scan;
        if (ropeOutputHasExternalUsers(ops, start, scan, rr)) continue;
        if (!canHoistAttentionForStoreGroup(ops, start, scan, att, &command, executed)) continue;

        const sidecar_index = findAttentionStoreSidecarIndex(ops, scan, att, used) orelse return null;
        command.sidecar_indices[0] = sidecar_index;
        command.sidecar_count = 1;
        command.op_count = @intCast(sidecar_index - start + 1);
        return command;
    }
    return null;
}

fn findRopeAttentionStoreGroupCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const max_pairs: usize = @intCast(@min(policy.max_rope_attention_store_batch, max_projection_group_anchors / 2));
    if (max_pairs < 2) return null;

    const first_pair = findRopeAttentionStorePair(ops, start, used, executed) orelse return null;
    var command = ProgramCommand{
        .kind = .rope_attention_store_group,
        .op_start = @intCast(start),
        .op_count = 1,
        .anchor_count = 0,
        .sidecar_count = 0,
    };
    appendRopeAttentionStorePair(&command, start, first_pair.attention_index, first_pair.sidecar_index, start);

    const first_att = ops[first_pair.attention_index].attention;
    const first_rope = ops[start].rope;
    var scan = start + 1;
    while (scan < ops.len and command.sidecar_count < max_pairs) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const rr = switch (ops[scan]) {
            .rope => |rr| rr,
            else => continue,
        };
        if (!ropeStoreBatchGeometryCompatible(first_rope, rr)) continue;
        const pair = findRopeAttentionStorePair(ops, scan, used, executed) orelse continue;
        const att = ops[pair.attention_index].attention;
        if (!attentionGeometryCompatible(first_att, att)) continue;

        var candidate = command;
        const slot = candidate.anchor_count;
        if (slot + 2 > max_projection_group_anchors) break;
        candidate.indices[slot] = scan;
        candidate.indices[slot + 1] = pair.attention_index;
        candidate.anchor_count += 2;
        if (!canHoistOpToRopeAttentionStoreGroup(ops, start, scan, .{ .rope = rr }, &candidate, executed)) continue;
        if (!canHoistAttentionForStoreGroup(ops, start, pair.attention_index, att, &candidate, executed)) continue;
        if (ropeAttentionStorePairConflictsSelected(ops, &command, scan, pair.attention_index, ops[pair.sidecar_index].slice_assign)) continue;

        var selected = command;
        appendRopeAttentionStorePair(&selected, scan, pair.attention_index, pair.sidecar_index, start);
        if (selected.sidecar_count > policy.max_attention_store_batch and !ropeAttentionStoreCompactBatchCompatible(ops, &selected)) continue;
        command = selected;
    }

    return if (command.sidecar_count >= 2) command else null;
}

fn findDelayedRopeAttentionStoreGroupCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const max_pairs: usize = @max(1, @as(usize, @intCast(@min(policy.max_rope_attention_store_batch, max_projection_group_anchors / 2))));

    const first_pair = findDelayableRopeAttentionStorePair(ops, start, used, executed) orelse return null;
    var emit_at = first_pair.attention_index;
    if (emit_at <= start) return null;

    var command = ProgramCommand{
        .kind = .rope_attention_store_group,
        .op_start = @intCast(emit_at),
        .op_count = 1,
        .anchor_count = 0,
        .sidecar_count = 0,
    };
    appendRopeAttentionStorePair(&command, start, first_pair.attention_index, first_pair.sidecar_index, emit_at);

    const first_att = ops[first_pair.attention_index].attention;
    const first_rope = ops[start].rope;
    var scan = start + 1;
    while (scan < ops.len and command.sidecar_count < max_pairs) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const rr = switch (ops[scan]) {
            .rope => |rr| rr,
            else => continue,
        };
        if (!ropeStoreBatchGeometryCompatible(first_rope, rr)) continue;
        const pair = findDelayableRopeAttentionStorePair(ops, scan, used, executed) orelse continue;
        if (pair.attention_index < emit_at) continue;
        const att = ops[pair.attention_index].attention;
        if (!attentionGeometryCompatible(first_att, att)) continue;
        const sa = ops[pair.sidecar_index].slice_assign;

        var candidate = command;
        if (candidate.anchor_count + 2 > max_projection_group_anchors) break;
        appendRopeAttentionStorePair(&candidate, scan, pair.attention_index, pair.sidecar_index, emit_at);
        if (ropeAttentionStorePairConflictsSelected(ops, &command, scan, pair.attention_index, sa)) continue;
        const candidate_emit_at = @max(emit_at, pair.attention_index);
        setRopeAttentionStoreCommandEmit(&candidate, candidate_emit_at);
        if (!ropeAttentionStoreCommandLegalAt(ops, &candidate, candidate_emit_at, executed)) continue;
        if (candidate.sidecar_count > policy.max_attention_store_batch and !ropeAttentionStoreCompactBatchCompatible(ops, &candidate)) continue;

        command = candidate;
        emit_at = candidate_emit_at;
    }

    if (command.sidecar_count >= 2) return command;
    if (findRopeAttentionStoreGroupCommand(ops, start, policy, used, executed) != null) return null;
    if (findRopeAttentionStoreChainCommand(ops, start, used, executed) != null) return null;
    command.kind = .rope_attention_store_chain;
    command.op_count = @intCast(first_pair.sidecar_index - emit_at + 1);
    return command;
}

const RopeAttentionStorePair = struct {
    attention_index: usize,
    sidecar_index: usize,
};

fn findRopeAttentionStorePair(
    ops: []const backend_mod.DeviceOp,
    rope_index: usize,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?RopeAttentionStorePair {
    const rr = switch (ops[rope_index]) {
        .rope => |rr| rr,
        else => return null,
    };
    var scan = rope_index + 1;
    while (scan < ops.len) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const att = switch (ops[scan]) {
            .attention => |att| att,
            else => continue,
        };
        if (!ropeAttentionCompatible(rr, att)) continue;
        if (ropeOutputHasExternalUsers(ops, rope_index, scan, rr)) continue;
        var command = ProgramCommand{
            .kind = .rope_attention_store_chain,
            .op_start = @intCast(rope_index),
            .op_count = @intCast(scan - rope_index + 1),
            .anchor_count = 2,
            .sidecar_count = 0,
        };
        command.indices[0] = rope_index;
        command.indices[1] = scan;
        if (!canHoistAttentionForStoreGroup(ops, rope_index, scan, att, &command, executed)) continue;
        const sidecar_index = findAttentionStoreSidecarIndex(ops, scan, att, used) orelse return null;
        return .{ .attention_index = scan, .sidecar_index = sidecar_index };
    }
    return null;
}

fn findDelayableRopeAttentionStorePair(
    ops: []const backend_mod.DeviceOp,
    rope_index: usize,
    used: ?[]const bool,
    executed: ?[]const bool,
) ?RopeAttentionStorePair {
    const rr = switch (ops[rope_index]) {
        .rope => |rr| rr,
        else => return null,
    };
    var scan = rope_index + 1;
    while (scan < ops.len) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const att = switch (ops[scan]) {
            .attention => |att| att,
            else => continue,
        };
        if (!ropeAttentionCompatible(rr, att)) continue;
        if (ropeOutputHasExternalUsers(ops, rope_index, scan, rr)) continue;

        var command = ProgramCommand{
            .kind = .rope_attention_store_group,
            .op_start = @intCast(scan),
            .op_count = 1,
            .anchor_count = 2,
            .sidecar_count = 0,
        };
        command.indices[0] = rope_index;
        command.indices[1] = scan;
        if (!canDelayOpToCommand(ops, rope_index, scan, .{ .rope = rr }, &command)) continue;

        const sidecar_index = findAttentionStoreSidecarIndex(ops, scan, att, used) orelse return null;
        command.sidecar_indices[0] = sidecar_index;
        command.sidecar_count = 1;
        const sa = ops[sidecar_index].slice_assign;
        if (!canHoistOpToRopeAttentionStoreGroup(ops, scan, sidecar_index, .{ .slice_assign = sa }, &command, executed)) continue;
        return .{ .attention_index = scan, .sidecar_index = sidecar_index };
    }
    return null;
}

fn appendRopeAttentionStorePair(
    command: *ProgramCommand,
    rope_index: usize,
    attention_index: usize,
    sidecar_index: usize,
    group_start: usize,
) void {
    const slot = command.anchor_count;
    command.indices[slot] = rope_index;
    command.indices[slot + 1] = attention_index;
    command.sidecar_indices[command.sidecar_count] = sidecar_index;
    command.anchor_count += 2;
    command.sidecar_count += 1;
    command.op_count = @intCast(@max(
        group_start + @as(usize, command.op_count),
        sidecar_index + 1,
    ) - group_start);
}

fn setRopeAttentionStoreCommandEmit(command: *ProgramCommand, emit_at: usize) void {
    command.op_start = @intCast(emit_at);
    var end = emit_at + 1;
    for (command.sidecarIndices()[0..command.sidecar_count]) |maybe_idx| {
        if (maybe_idx) |idx| end = @max(end, idx + 1);
    }
    command.op_count = @intCast(end - emit_at);
}

fn ropeAttentionStoreCommandLegalAt(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    emit_at: usize,
    executed: ?[]const bool,
) bool {
    var i: usize = 0;
    while (i < command.sidecar_count) : (i += 1) {
        const rope_index = command.indices[i * 2];
        const attention_index = command.indices[i * 2 + 1];
        const sidecar_index = command.sidecar_indices[i] orelse return false;
        const rr = switch (ops[rope_index]) {
            .rope => |rr| rr,
            else => return false,
        };
        const att = switch (ops[attention_index]) {
            .attention => |att| att,
            else => return false,
        };
        const sa = switch (ops[sidecar_index]) {
            .slice_assign => |sa| sa,
            else => return false,
        };
        if (!canMoveOpToRopeAttentionStoreEmit(ops, rope_index, emit_at, .{ .rope = rr }, command, executed)) return false;
        if (!canMoveAttentionToRopeAttentionStoreEmit(ops, attention_index, emit_at, att, command, executed)) return false;
        if (!canMoveOpToRopeAttentionStoreEmit(ops, sidecar_index, emit_at, .{ .slice_assign = sa }, command, executed)) return false;
    }
    return true;
}

fn canMoveOpToRopeAttentionStoreEmit(
    ops: []const backend_mod.DeviceOp,
    candidate_index: usize,
    emit_at: usize,
    candidate: backend_mod.DeviceOp,
    command: *const ProgramCommand,
    executed: ?[]const bool,
) bool {
    if (candidate_index < emit_at) return canDelayOpToCommand(ops, candidate_index, emit_at, candidate, command);
    if (candidate_index > emit_at) return canHoistOpToRopeAttentionStoreGroup(ops, emit_at, candidate_index, candidate, command, executed);
    return true;
}

fn canMoveAttentionToRopeAttentionStoreEmit(
    ops: []const backend_mod.DeviceOp,
    attention_index: usize,
    emit_at: usize,
    att: anytype,
    command: *const ProgramCommand,
    executed: ?[]const bool,
) bool {
    if (attention_index < emit_at) return canDelayOpToCommand(ops, attention_index, emit_at, .{ .attention = att }, command);
    if (attention_index > emit_at) return canHoistAttentionForStoreGroup(ops, emit_at, attention_index, att, command, executed);
    return true;
}

fn canDelayOpToCommand(
    ops: []const backend_mod.DeviceOp,
    candidate_index: usize,
    emit_at: usize,
    candidate: backend_mod.DeviceOp,
    command: *const ProgramCommand,
) bool {
    if (candidate_index > emit_at) return false;
    const candidate_access = opAccessSpans(candidate);
    for (ops[candidate_index + 1 .. emit_at + 1], candidate_index + 1..) |op, idx| {
        if (commandContainsIndex(command, idx)) continue;
        for (candidate_access.readSpans()) |read| {
            if (opWritesSpan(op, read)) return false;
        }
        for (candidate_access.writeSpans()) |write| {
            if (hoistWriteConflict(op, candidate, write)) return false;
        }
        if (candidate_access.read_overflow or candidate_access.write_overflow) {
            if (opAccessConflicts(op, candidate)) return false;
        }
    }
    return true;
}

fn canHoistOpToRopeAttentionStoreGroup(
    ops: []const backend_mod.DeviceOp,
    group_start: usize,
    candidate_index: usize,
    candidate: backend_mod.DeviceOp,
    command: *const ProgramCommand,
    executed: ?[]const bool,
) bool {
    const candidate_access = opAccessSpans(candidate);
    for (ops[group_start..candidate_index], group_start..) |op, idx| {
        if (commandContainsIndex(command, idx)) continue;
        if (executed) |executed_ops| {
            if (idx < executed_ops.len and executed_ops[idx]) continue;
        }
        for (candidate_access.readSpans()) |read| {
            if (opWritesSpan(op, read)) return false;
        }
        for (candidate_access.writeSpans()) |write| {
            if (hoistWriteConflict(op, candidate, write)) return false;
        }
        if (candidate_access.read_overflow or candidate_access.write_overflow) {
            if (opAccessConflicts(op, candidate)) return false;
        }
    }
    return true;
}

fn hoistWriteConflict(op: backend_mod.DeviceOp, candidate: backend_mod.DeviceOp, write: BufferSpan) bool {
    if (!opTouchesSpan(op, write)) return false;
    const candidate_sa = switch (candidate) {
        .slice_assign => |sa| sa,
        else => return true,
    };
    const op_sa = switch (op) {
        .slice_assign => |sa| sa,
        else => return true,
    };
    return opReadsSpan(op, write) or sliceAssignWritesMayOverlap(op_sa, candidate_sa);
}

fn ropeAttentionStorePairConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    rope_index: usize,
    attention_index: usize,
    sa: anytype,
) bool {
    const rope_op = ops[rope_index];
    const attention_op = ops[attention_index];
    const sidecar_op: backend_mod.DeviceOp = .{ .slice_assign = sa };
    const sidecar_access = opAccessSpans(sidecar_op);
    for (command.anchorIndices()) |idx| {
        for (sidecar_access.writeSpans()) |write| {
            if (opReadsSpan(ops[idx], write)) return true;
        }
        if (sidecar_access.write_overflow and opReadsBuffer(ops[idx], sa.dst)) return true;
    }
    for (command.sidecarIndices()) |maybe_idx| {
        const idx = maybe_idx orelse continue;
        const selected_sa = switch (ops[idx]) {
            .slice_assign => |selected| selected,
            else => return true,
        };
        if (sliceAssignWritesMayOverlap(selected_sa, sa)) return true;
        const selected_access = opAccessSpans(ops[idx]);
        for (selected_access.writeSpans()) |write| {
            if (opReadsSpan(rope_op, write) or opReadsSpan(attention_op, write)) return true;
        }
        if (selected_access.write_overflow and
            (opReadsBuffer(rope_op, selected_sa.dst) or opReadsBuffer(attention_op, selected_sa.dst))) return true;
    }
    return false;
}

fn ropeOutputHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    rope_index: usize,
    attention_index: usize,
    rr: anytype,
) bool {
    const rope_access = opAccessSpans(.{ .rope = rr });
    for (ops[rope_index + 1 .. attention_index], rope_index + 1..) |op, idx| {
        if (idx == attention_index) continue;
        for (rope_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return true;
        }
        if (rope_access.write_overflow and opTouchesBuffer(op, rr.dst)) return true;
    }
    return false;
}

fn attentionOutputHasExternalUsers(
    ops: []const backend_mod.DeviceOp,
    attention_index: usize,
    sidecar_index: usize,
    att: anytype,
) bool {
    const attention_access = opAccessSpans(.{ .attention = att });
    for (ops[attention_index + 1 ..], attention_index + 1..) |op, idx| {
        if (idx == sidecar_index) continue;
        for (attention_access.writeSpans()) |write| {
            if (opReadsSpan(op, write)) return true;
        }
        if (attention_access.write_overflow and opReadsBuffer(op, att.dst)) return true;
    }
    return false;
}

fn canHoistAttentionForStoreGroup(
    ops: []const backend_mod.DeviceOp,
    group_start: usize,
    attention_index: usize,
    att: anytype,
    command: *const ProgramCommand,
    executed: ?[]const bool,
) bool {
    const candidate_access = opAccessSpans(.{ .attention = att });
    for (ops[group_start..attention_index], group_start..) |op, idx| {
        if (commandContainsIndex(command, idx)) continue;
        if (executed) |executed_ops| {
            if (idx < executed_ops.len and executed_ops[idx]) continue;
        }
        for (candidate_access.readSpans()) |read| {
            if (opWritesSpan(op, read)) return false;
        }
        for (candidate_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return false;
        }
        if (candidate_access.read_overflow or candidate_access.write_overflow) {
            if (opAccessConflicts(op, .{ .attention = att })) return false;
        }
    }
    return true;
}

fn commandContainsIndex(command: *const ProgramCommand, candidate: usize) bool {
    for (command.anchorIndices()) |idx| {
        if (idx == candidate) return true;
    }
    for (command.carriedSidecarIndices()) |maybe_idx| {
        if (maybe_idx) |idx| {
            if (idx == candidate) return true;
        }
    }
    return false;
}

fn appendAttentionStorePair(
    ops: []const backend_mod.DeviceOp,
    command: *ProgramCommand,
    attention_index: usize,
    att: anytype,
    group_start: usize,
    used: ?[]const bool,
) bool {
    const sidecar_index = findAttentionStoreSidecarIndex(ops, attention_index, att, used) orelse return false;
    const sa = ops[sidecar_index].slice_assign;
    if (attentionStorePairConflictsSelected(ops, command, attention_index, sa)) return false;

    const slot = command.anchor_count;
    command.indices[slot] = attention_index;
    command.sidecar_indices[slot] = sidecar_index;
    command.anchor_count += 1;
    command.sidecar_count += 1;
    command.op_count = @intCast(@max(
        group_start + @as(usize, command.op_count),
        sidecar_index + 1,
    ) - group_start);
    return true;
}

fn findAttentionStoreSidecarIndex(
    ops: []const backend_mod.DeviceOp,
    attention_index: usize,
    att: anytype,
    used: ?[]const bool,
) ?usize {
    var scan = attention_index + 1;
    while (scan < ops.len) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const sa = switch (ops[scan]) {
            .slice_assign => |sa| sa,
            else => continue,
        };
        if (!attentionSliceStoreCompatible(att, sa)) continue;
        if (!canFuseAttentionStoreSidecar(ops, attention_index, scan, sa)) continue;
        return scan;
    }
    return null;
}

fn attentionStorePairConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
    attention_index: usize,
    sa: anytype,
) bool {
    const attention_op = ops[attention_index];
    const sidecar_op: backend_mod.DeviceOp = .{ .slice_assign = sa };
    for (command.anchorIndices()) |idx| {
        if (opAccessConflicts(ops[idx], attention_op)) return true;
        if (opAccessConflicts(ops[idx], sidecar_op)) return true;
    }
    for (command.sidecarIndices()) |maybe_idx| {
        const idx = maybe_idx orelse continue;
        const selected_sa = switch (ops[idx]) {
            .slice_assign => |selected| selected,
            else => return true,
        };
        if (sliceAssignWritesMayOverlap(selected_sa, sa)) return true;
    }
    return false;
}

pub fn summarizeAttentionStoreGroupCandidates(
    ops: []const backend_mod.DeviceOp,
    policy: CommandStreamPolicy,
) AttentionStoreGroupCandidateSummary {
    var summary = AttentionStoreGroupCandidateSummary{};
    const max_ops: usize = @intCast(@min(policy.max_attention_store_batch, max_projection_group_anchors));
    if (max_ops < 2) return summary;

    for (ops, 0..) |op, start| {
        const first = switch (op) {
            .attention => |att| att,
            else => continue,
        };
        summary.anchors += 1;

        var command = ProgramCommand{
            .kind = .attention_store_group,
            .op_start = @intCast(start),
            .op_count = 1,
            .anchor_count = 0,
            .sidecar_count = 0,
        };
        if (!appendAttentionStorePair(ops, &command, start, first, start, null)) {
            summary.first_store_missing += 1;
            continue;
        }

        var scan = start + 1;
        while (scan < ops.len and command.anchor_count < max_ops) : (scan += 1) {
            const next = switch (ops[scan]) {
                .attention => |att| att,
                else => continue,
            };
            summary.candidate_attentions += 1;
            if (!attentionGeometryCompatible(first, next)) {
                summary.geometry_rejects += 1;
                continue;
            }
            if (!canHoistAttentionForStoreGroup(ops, start, scan, next, &command, null)) {
                summary.hoist_rejects += 1;
                continue;
            }
            if (opConflictsSelected(ops, command.anchorIndices(), .{ .attention = next })) {
                summary.selected_conflict_rejects += 1;
                continue;
            }
            const sidecar_index = findAttentionStoreSidecarIndex(ops, scan, next, null) orelse {
                summary.no_store_rejects += 1;
                continue;
            };
            const sa = ops[sidecar_index].slice_assign;
            if (attentionStorePairConflictsSelected(ops, &command, scan, sa)) {
                summary.pair_conflict_rejects += 1;
                continue;
            }
            _ = appendAttentionStorePair(ops, &command, scan, next, start, null);
        }

        if (command.anchor_count >= 2) {
            summary.formed_groups += 1;
            summary.grouped_anchors += command.anchor_count;
            summary.max_group_anchors = @max(summary.max_group_anchors, command.anchor_count);
        }
    }

    return summary;
}

pub fn summarizeRopeAttentionStoreGroupCandidates(
    ops: []const backend_mod.DeviceOp,
    policy: CommandStreamPolicy,
) RopeAttentionStoreGroupCandidateSummary {
    var summary = RopeAttentionStoreGroupCandidateSummary{};
    const max_pairs: usize = @intCast(@min(policy.max_rope_attention_store_batch, max_projection_group_anchors / 2));
    if (max_pairs < 2) return summary;

    for (ops, 0..) |op, start| {
        switch (op) {
            .rope => {},
            else => continue,
        }
        summary.anchors += 1;

        if (findDelayableRopeAttentionStorePair(ops, start, null, null) == null) {
            summary.first_pair_missing += 1;
            continue;
        }

        const command = findDelayedRopeAttentionStoreGroupCommand(ops, start, policy, null, null) orelse continue;
        if (command.kind == .rope_attention_store_group and command.sidecar_count >= 2) {
            summary.formed_groups += 1;
            summary.grouped_pairs += command.sidecar_count;
            summary.max_group_pairs = @max(summary.max_group_pairs, command.sidecar_count);
        }
    }

    return summary;
}

pub fn summarizeEarlyRopeAttentionStoreGroupCandidates(
    ops: []const backend_mod.DeviceOp,
    policy: CommandStreamPolicy,
) EarlyRopeAttentionStoreGroupCandidateSummary {
    var summary = EarlyRopeAttentionStoreGroupCandidateSummary{};
    const max_pairs: usize = @intCast(@min(policy.max_rope_attention_store_batch, max_projection_group_anchors / 2));
    if (max_pairs < 2) return summary;

    for (ops, 0..) |op, start| {
        const first_rope = switch (op) {
            .rope => |rr| rr,
            else => continue,
        };
        summary.anchors += 1;

        const first_pair = findRopeAttentionStorePair(ops, start, null, null) orelse {
            summary.first_pair_missing += 1;
            continue;
        };
        var command = ProgramCommand{
            .kind = .rope_attention_store_group,
            .op_start = @intCast(start),
            .op_count = 1,
            .anchor_count = 0,
            .sidecar_count = 0,
        };
        appendRopeAttentionStorePair(&command, start, first_pair.attention_index, first_pair.sidecar_index, start);

        const first_att = ops[first_pair.attention_index].attention;
        var scan = start + 1;
        while (scan < ops.len and command.sidecar_count < max_pairs) : (scan += 1) {
            const rr = switch (ops[scan]) {
                .rope => |rr| rr,
                else => continue,
            };
            summary.candidate_ropes += 1;
            if (!ropeStoreBatchGeometryCompatible(first_rope, rr)) {
                summary.geometry_rejects += 1;
                continue;
            }
            const pair = findRopeAttentionStorePair(ops, scan, null, null) orelse {
                summary.pair_missing_rejects += 1;
                continue;
            };
            const att = ops[pair.attention_index].attention;
            if (!attentionGeometryCompatible(first_att, att)) {
                summary.geometry_rejects += 1;
                continue;
            }

            var candidate = command;
            const slot = candidate.anchor_count;
            if (slot + 2 > max_projection_group_anchors) break;
            candidate.indices[slot] = scan;
            candidate.indices[slot + 1] = pair.attention_index;
            candidate.anchor_count += 2;
            if (!canHoistOpToRopeAttentionStoreGroup(ops, start, scan, .{ .rope = rr }, &candidate, null)) {
                summary.rope_hoist_rejects += 1;
                continue;
            }
            if (!canHoistAttentionForStoreGroup(ops, start, pair.attention_index, att, &candidate, null)) {
                summary.attention_hoist_rejects += 1;
                continue;
            }
            if (ropeAttentionStorePairConflictsSelected(ops, &command, scan, pair.attention_index, ops[pair.sidecar_index].slice_assign)) {
                summary.selected_conflict_rejects += 1;
                continue;
            }
            appendRopeAttentionStorePair(&command, scan, pair.attention_index, pair.sidecar_index, start);
        }

        if (command.sidecar_count >= 2) {
            summary.formed_groups += 1;
            summary.grouped_pairs += command.sidecar_count;
            summary.max_group_pairs = @max(summary.max_group_pairs, command.sidecar_count);
        }
    }

    return summary;
}

fn findMovementGroupCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const first = switch (ops[start]) {
        .slice_assign => |sa| sa,
        else => return null,
    };

    const max_ops: usize = @intCast(@min(policy.max_movement_batch, max_projection_group_anchors));
    if (max_ops < 2) return null;

    var command = ProgramCommand{
        .kind = .movement_group,
        .op_start = @intCast(start),
        .op_count = 1,
        .anchor_count = 1,
    };
    command.indices[0] = start;

    var scan = start + 1;
    while (scan < ops.len and command.anchor_count < max_ops) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const next = switch (ops[scan]) {
            .slice_assign => |sa| sa,
            else => continue,
        };
        if (!sliceAssignBatchCompatible(first, next)) continue;
        if (!canHoistOpTo(ops, start, scan, .{ .slice_assign = next })) continue;
        if (opConflictsSelected(ops, command.anchorIndices(), .{ .slice_assign = next })) continue;
        command.indices[command.anchor_count] = scan;
        command.anchor_count += 1;
        command.op_count = @intCast(scan - start + 1);
    }

    return if (command.anchor_count >= 2) command else null;
}

fn findAttentionGroupCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const first = switch (ops[start]) {
        .attention => |att| att,
        else => return null,
    };

    const max_ops: usize = @intCast(@min(policy.max_attention_batch, max_projection_group_anchors));
    if (max_ops < 2) return null;

    var command = ProgramCommand{
        .kind = .attention_group,
        .op_start = @intCast(start),
        .op_count = 1,
        .anchor_count = 1,
    };
    command.indices[0] = start;

    var scan = start + 1;
    while (scan < ops.len and command.anchor_count < max_ops) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const next = switch (ops[scan]) {
            .attention => |att| att,
            else => continue,
        };
        if (!attentionBatchCompatible(first, next)) continue;
        if (!canHoistOpTo(ops, start, scan, .{ .attention = next })) continue;
        if (opConflictsSelected(ops, command.anchorIndices(), .{ .attention = next })) continue;
        command.indices[command.anchor_count] = scan;
        command.anchor_count += 1;
        command.op_count = @intCast(scan - start + 1);
    }

    return if (command.anchor_count >= 2) command else null;
}

fn findProjectionChainCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start + 1 >= ops.len) return null;
    if (commandRangeTouchesUsed(@intCast(start), 2, used)) return null;
    const q = switch (ops[start]) {
        .qmatmul => |q| q,
        else => return null,
    };
    if (!projectionSidecarCompatible(q, ops[start + 1])) return null;

    var command = ProgramCommand{
        .kind = .projection_chain,
        .op_start = @intCast(start),
        .op_count = 2,
        .projection_kind = if (q.M == 1) .qmatvec else .qmatmul,
        .anchor_count = 1,
        .sidecar_count = 1,
    };
    command.indices[0] = start;
    command.sidecar_indices[0] = start + 1;
    return command;
}

fn findElementwiseBatchCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }
    const first = switch (ops[start]) {
        .elementwise => |e| e,
        else => return null,
    };
    if (!canBatchElementwiseOp(first)) return null;

    const max_ops: usize = @intCast(@min(policy.max_elementwise_batch, max_projection_group_anchors));
    if (max_ops < 2) return null;

    var command = ProgramCommand{
        .kind = .elementwise_batch,
        .op_start = @intCast(start),
        .op_count = 1,
        .anchor_count = 1,
    };
    command.indices[0] = start;

    var scan = start + 1;
    while (scan < ops.len and command.anchor_count < max_ops) : (scan += 1) {
        if (used) |used_ops| {
            if (scan >= used_ops.len or used_ops[scan]) continue;
        }
        const e = switch (ops[scan]) {
            .elementwise => |e| e,
            else => continue,
        };
        if (!canBatchElementwiseOp(e)) continue;
        if (!canHoistElementwiseTo(ops, start, scan, e)) continue;
        if (elementwiseConflictsSelected(ops, command.anchorIndices(), e)) continue;
        command.indices[command.anchor_count] = scan;
        command.anchor_count += 1;
        command.op_count = @intCast(scan - start + 1);
    }

    return if (command.anchor_count >= 2) command else null;
}

fn findContiguousBatchCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
) ?ProgramCommand {
    if (policy.max_attention_batch >= 2) {
        const n = attentionBatchRunLen(ops[start..], policy.max_attention_batch);
        if (n >= 2 and !commandRangeTouchesUsed(@intCast(start), @intCast(n), used)) {
            return ProgramCommand.contiguous(.attention_batch, start, n);
        }
    }

    if (policy.max_rope_batch >= 2) {
        const n = ropeBatchRunLen(ops[start..], policy.max_rope_batch);
        if (n >= 2 and !commandRangeTouchesUsed(@intCast(start), @intCast(n), used)) {
            return ProgramCommand.contiguous(.rope_batch, start, n);
        }
    }

    if (policy.max_movement_batch >= 2) {
        const n = sliceAssignBatchRunLen(ops[start..], policy.max_movement_batch);
        if (n >= 2 and !commandRangeTouchesUsed(@intCast(start), @intCast(n), used)) {
            return ProgramCommand.contiguous(.movement_batch, start, n);
        }
    }

    return null;
}

pub fn summarizeProgramCommands(commands: []const ProgramCommand) ProgramCommandSummary {
    var summary = ProgramCommandSummary{ .commands = @intCast(commands.len) };
    for (commands) |command| {
        const covered = command.coveredOpCount();
        const dispatches = command.dispatchCount();
        summary.covered_ops += covered;
        summary.estimated_dispatches += dispatches;
        if (covered > dispatches) {
            summary.estimated_saved_dispatches += covered - dispatches;
        }

        switch (command.kind) {
            .op => summary.op_commands += 1,
            .row_chain => summary.row_chains += 1,
            .rope_chain => summary.rope_chains += 1,
            .rope_batch => summary.rope_batches += 1,
            .rope_store_group => {
                summary.rope_store_groups += 1;
                summary.rope_store_group_ops += command.anchor_count;
                summary.rope_store_group_sidecars += command.sidecar_count;
            },
            .movement_batch => summary.movement_batches += 1,
            .movement_group => {
                summary.movement_groups += 1;
                summary.movement_group_ops += command.anchor_count;
            },
            .attention_chain => {
                summary.attention_chains += 1;
                summary.attention_chain_sidecars += command.sidecar_count;
            },
            .attention_store_chain => {
                summary.attention_store_chains += 1;
                summary.attention_store_chain_sidecars += command.sidecar_count;
            },
            .attention_store_group => {
                summary.attention_store_groups += 1;
                summary.attention_store_group_ops += command.anchor_count;
                summary.attention_store_group_sidecars += command.sidecar_count;
            },
            .rope_attention_store_chain => {
                summary.rope_attention_store_chains += 1;
                summary.rope_attention_store_chain_sidecars += command.sidecar_count;
            },
            .rope_attention_store_group => {
                summary.rope_attention_store_groups += 1;
                summary.rope_attention_store_group_ops += command.anchor_count;
                summary.rope_attention_store_group_sidecars += command.sidecar_count;
            },
            .attention_batch => summary.attention_batches += 1,
            .attention_group => {
                summary.attention_groups += 1;
                summary.attention_group_ops += command.anchor_count;
            },
            .elementwise_batch => {
                summary.elementwise_batches += 1;
                summary.elementwise_ops += command.anchor_count;
            },
            .repeat_fused_elementwise_chain => summary.repeat_fused_elementwise_chains += 1,
            .projection_fused_elementwise_chain => summary.projection_fused_elementwise_chains += 1,
            .projection_pair_fused_elementwise_chain => summary.projection_pair_fused_elementwise_chains += 1,
            .projection_chain => {
                summary.projection_chains += 1;
                summary.projection_chain_sidecars += command.sidecar_count;
            },
            .projection_group => {
                summary.projection_groups += 1;
                summary.projection_anchors += command.anchor_count;
                summary.projection_sidecars += command.sidecar_count;
                summary.max_projection_span_ops = @max(summary.max_projection_span_ops, command.op_count);
            },
            .projection_cache_group => {
                summary.projection_cache_groups += 1;
                summary.projection_cache_anchors += command.anchor_count;
                summary.projection_cache_sidecars += command.sidecar_count;
                summary.max_projection_span_ops = @max(summary.max_projection_span_ops, command.op_count);
            },
        }
    }
    return summary;
}

pub fn markProgramCommandUsed(used: []bool, command: ProgramCommand) void {
    var indices = command.coveredIndexIterator();
    while (indices.next()) |idx| {
        if (idx < used.len) used[idx] = true;
    }
}

fn commandRangeTouchesUsed(op_start: u32, op_count: u32, used: ?[]const bool) bool {
    const used_ops = used orelse return false;
    const start: usize = @intCast(op_start);
    if (start >= used_ops.len) return true;
    const end = @min(used_ops.len, start + @as(usize, op_count));
    for (used_ops[start..end]) |slot| {
        if (slot) return true;
    }
    return false;
}

fn projectionMatchesPolicy(q: anytype, policy: ProjectionGroupPolicy) bool {
    return switch (policy.kind) {
        .qmatvec => q.M == 1,
        .qmatmul => q.M != 1,
    };
}

pub fn opReadsBuffer(op: backend_mod.DeviceOp, buf: u16) bool {
    return switch (op) {
        .elementwise => |e| e.src0 == buf or e.src1 == buf,
        .matmul => |m| m.a == buf or m.b == buf,
        .qmatmul => |q| q.input == buf,
        .softmax => |s| s.src == buf,
        .layernorm => |l| l.src == buf,
        .rmsnorm => |r| r.src == buf,
        .reduce => |r| r.src == buf,
        .repeat => |rp| rp.src == buf,
        .slice_assign => |sa| sa.src == buf,
        .rope => |rr| rr.src == buf or rr.cos_sin == buf,
        .attention => |att| att.q == buf or att.k == buf or att.v == buf or att.mask == buf,
        .fused_elementwise => |fe| {
            if (fe.src == buf) return true;
            for (fe.steps) |step| {
                if (step.op.isBinary() and step.secondary_buf == buf) return true;
            }
            return false;
        },
    };
}

pub fn opWritesBuffer(op: backend_mod.DeviceOp, buf: u16) bool {
    return switch (op) {
        .elementwise => |e| e.dst == buf,
        .matmul => |m| m.dst == buf,
        .qmatmul => |q| q.dst == buf,
        .softmax => |s| s.dst == buf,
        .layernorm => |l| l.dst == buf,
        .rmsnorm => |r| r.dst == buf,
        .reduce => |r| r.dst == buf,
        .repeat => |rp| rp.dst == buf,
        .slice_assign => |sa| sa.dst == buf,
        .rope => |rr| rr.dst == buf,
        .attention => |att| att.dst == buf,
        .fused_elementwise => |fe| fe.dst == buf,
    };
}

pub fn opTouchesBuffer(op: backend_mod.DeviceOp, buf: u16) bool {
    return opReadsBuffer(op, buf) or opWritesBuffer(op, buf);
}

fn bufferSpan(buf: u16, offset: anytype, len: anytype) BufferSpan {
    const start: u64 = @intCast(offset);
    const n: u64 = @intCast(len);
    return .{ .buf = buf, .start = start, .end = start + n };
}

fn stridedSpan(buf: u16, offset: anytype, rows: anytype, cols: anytype, row_stride: anytype, col_stride: anytype) BufferSpan {
    const start: u64 = @intCast(offset);
    const r: u64 = @intCast(rows);
    const c: u64 = @intCast(cols);
    if (r == 0 or c == 0) return .{ .buf = buf, .start = start, .end = start };
    const rs: u64 = @intCast(row_stride);
    const cs: u64 = @intCast(col_stride);
    const last = start + (r - 1) * rs + (c - 1) * cs;
    return .{ .buf = buf, .start = start, .end = last + 1 };
}

fn strided4Span(buf: u16, offset: anytype, ne: [4]u32, strides: [4]u32) BufferSpan {
    const start: u64 = @intCast(offset);
    var last = start;
    for (ne, 0..) |extent, i| {
        if (extent == 0) return .{ .buf = buf, .start = start, .end = start };
        last += @as(u64, extent - 1) * @as(u64, strides[i]);
    }
    return .{ .buf = buf, .start = start, .end = last + 1 };
}

fn opAccessSpans(op: backend_mod.DeviceOp) OpAccessSpans {
    var access = OpAccessSpans{};
    switch (op) {
        .elementwise => |e| {
            access.addRead(bufferSpan(e.src0, e.src0_offset, e.n));
            if (e.op.isBinary()) access.addRead(bufferSpan(e.src1, e.src1_offset, e.n));
            access.addWrite(bufferSpan(e.dst, e.dst_offset, e.n));
        },
        .matmul => |m| {
            const g = m.geom;
            access.addRead(stridedSpan(m.a, g.a_offset, g.M, g.K, g.a_row_stride, g.a_col_stride));
            access.addRead(stridedSpan(m.b, g.b_offset, g.K, g.N, g.b_row_stride, g.b_col_stride));
            access.addWrite(stridedSpan(m.dst, g.dst_offset, g.M, g.N, g.dst_row_stride, 1));
        },
        .qmatmul => |q| {
            const input_row_stride = if (q.input_row_stride != 0) q.input_row_stride else q.K;
            access.addRead(stridedSpan(q.input, q.input_offset, q.M, q.K, input_row_stride, 1));
            access.addWrite(stridedSpan(q.dst, q.dst_offset, q.M, q.N, qmatmulDstRowStride(q), 1));
        },
        .softmax => |s| {
            access.addRead(bufferSpan(s.src, s.src_offset, s.rows * s.cols));
            access.addWrite(bufferSpan(s.dst, s.dst_offset, s.rows * s.cols));
        },
        .layernorm => |l| {
            access.addRead(bufferSpan(l.src, l.src_offset, l.rows * l.cols));
            access.addWrite(bufferSpan(l.dst, l.dst_offset, l.rows * l.cols));
        },
        .rmsnorm => |r| {
            access.addRead(bufferSpan(r.src, r.src_offset, r.rows * r.cols));
            access.addWrite(bufferSpan(r.dst, r.dst_offset, r.rows * r.cols));
        },
        .reduce => |r| {
            access.addRead(bufferSpan(r.src, r.src_offset, r.n_out * r.reduce_size));
            access.addWrite(bufferSpan(r.dst, r.dst_offset, r.n_out));
        },
        .repeat => |rp| {
            access.addRead(strided4Span(rp.src, rp.src_offset, rp.src_ne, rp.src_strides));
            access.addWrite(strided4Span(rp.dst, rp.dst_offset, rp.dst_ne, rp.dst_strides));
        },
        .slice_assign => |sa| {
            access.addRead(stridedSpan(sa.src, sa.src_offset, sa.rows, sa.cols, sa.src_row_stride, sa.src_col_stride));
            access.addWrite(stridedSpan(sa.dst, sa.dst_offset, sa.rows, sa.cols, sa.dst_row_stride, sa.dst_col_stride));
        },
        .rope => |rr| {
            const d = rr.half_d * 2;
            access.addRead(stridedSpan(rr.src, rr.src_off, d, rr.seq_len, rr.src_rs, rr.src_cs));
            access.addRead(stridedSpan(rr.cos_sin, rr.cs_off, d, rr.seq_len, 1, rr.cs_cs));
            access.addWrite(stridedSpan(rr.dst, rr.dst_off, d, rr.seq_len, 1, d));
        },
        .attention => |att| {
            access.addRead(stridedSpan(att.q, att.q_off, att.d_head, att.seq_q, att.q_rs, att.q_cs));
            access.addRead(stridedSpan(att.k, att.k_off, att.d_head, att.seq_kv, att.k_rs, att.k_cs));
            access.addRead(stridedSpan(att.v, att.v_off, att.d_head, att.seq_kv, att.v_rs, att.v_cs));
            if (att.has_mask) access.addRead(stridedSpan(att.mask, att.mask_off, att.seq_kv, att.seq_q, att.mask_rs, att.mask_cs));
            access.addWrite(stridedSpan(att.dst, att.dst_off, att.d_head, att.seq_q, att.dst_rs, att.dst_cs));
        },
        .fused_elementwise => |fe| {
            access.addRead(bufferSpan(fe.src, fe.src_offset, fe.n));
            for (fe.steps) |step| {
                if (step.op.isBinary()) access.addRead(bufferSpan(step.secondary_buf, step.secondary_offset, fe.n));
            }
            access.addWrite(bufferSpan(fe.dst, fe.dst_offset, fe.n));
        },
    }
    return access;
}

pub fn opReadsSpan(op: backend_mod.DeviceOp, target: BufferSpan) bool {
    const access = opAccessSpans(op);
    for (access.readSpans()) |read| {
        if (read.overlaps(target)) return true;
    }
    return access.read_overflow and opReadsBuffer(op, target.buf);
}

pub fn opWritesSpan(op: backend_mod.DeviceOp, target: BufferSpan) bool {
    const access = opAccessSpans(op);
    for (access.writeSpans()) |write| {
        if (write.overlaps(target)) return true;
    }
    return access.write_overflow and opWritesBuffer(op, target.buf);
}

pub fn opWritesCoverSpan(op: backend_mod.DeviceOp, target: BufferSpan) bool {
    const access = opAccessSpans(op);
    for (access.writeSpans()) |write| {
        if (write.buf == target.buf and write.start <= target.start and write.end >= target.end) return true;
    }
    return access.write_overflow and opWritesBuffer(op, target.buf);
}

pub fn opTouchesSpan(op: backend_mod.DeviceOp, target: BufferSpan) bool {
    return opReadsSpan(op, target) or opWritesSpan(op, target);
}

pub fn opAccessConflicts(a: backend_mod.DeviceOp, b: backend_mod.DeviceOp) bool {
    const a_access = opAccessSpans(a);
    const b_access = opAccessSpans(b);
    for (a_access.writeSpans()) |write| {
        for (b_access.writeSpans()) |other_write| {
            if (write.overlaps(other_write)) return true;
        }
        for (b_access.readSpans()) |read| {
            if (write.overlaps(read)) return true;
        }
        if ((b_access.read_overflow and opReadsBuffer(b, write.buf)) or (b_access.write_overflow and opWritesBuffer(b, write.buf))) return true;
    }
    for (a_access.readSpans()) |read| {
        for (b_access.writeSpans()) |write| {
            if (read.overlaps(write)) return true;
        }
        if (b_access.write_overflow and opWritesBuffer(b, read.buf)) return true;
    }
    if (a_access.read_overflow or a_access.write_overflow) {
        const b_may_touch_overflowed_buffer = for (b_access.readSpans()) |read| {
            if (opTouchesBuffer(a, read.buf)) break true;
        } else for (b_access.writeSpans()) |write| {
            if (opTouchesBuffer(a, write.buf)) break true;
        } else false;
        if (b_may_touch_overflowed_buffer) return true;
    }
    return false;
}

pub fn canHoistOpTo(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    candidate_index: usize,
    candidate: backend_mod.DeviceOp,
) bool {
    const candidate_access = opAccessSpans(candidate);
    for (ops[start..candidate_index]) |op| {
        for (candidate_access.readSpans()) |read| {
            if (opWritesSpan(op, read)) return false;
        }
        for (candidate_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return false;
        }
        if (candidate_access.read_overflow or candidate_access.write_overflow) {
            if (opAccessConflicts(op, candidate)) return false;
        }
    }
    return true;
}

pub fn canFuseAttentionStoreSidecar(
    ops: []const backend_mod.DeviceOp,
    attention_index: usize,
    sidecar_index: usize,
    sa: anytype,
) bool {
    return attentionStoreSidecarBlocker(ops, attention_index, sidecar_index, sa) == .none;
}

pub const AttentionStoreSidecarBlocker = enum {
    none,
    invalid_range,
    candidate_read_written,
    sidecar_write_read,
    sidecar_write_written,
    overflow_conflict,
};

pub fn attentionStoreSidecarBlocker(
    ops: []const backend_mod.DeviceOp,
    attention_index: usize,
    sidecar_index: usize,
    sa: anytype,
) AttentionStoreSidecarBlocker {
    if (attention_index >= sidecar_index or sidecar_index > ops.len) return .invalid_range;
    const candidate_access = opAccessSpans(.{ .slice_assign = sa });
    for (ops[attention_index + 1 .. sidecar_index]) |op| {
        for (candidate_access.readSpans()) |read| {
            if (opWritesSpan(op, read)) return .candidate_read_written;
        }
        for (candidate_access.writeSpans()) |write| {
            if (opReadsSpan(op, write)) return .sidecar_write_read;
            if (opWritesSpan(op, write)) {
                const other_sa = switch (op) {
                    .slice_assign => |other| other,
                    else => return .sidecar_write_written,
                };
                if (sliceAssignWritesMayOverlap(other_sa, sa)) return .sidecar_write_written;
            }
        }
        if (candidate_access.read_overflow or candidate_access.write_overflow) {
            if (opAccessConflicts(op, .{ .slice_assign = sa })) return .overflow_conflict;
        }
    }
    return .none;
}

fn sliceAssignWritesMayOverlap(a: anytype, b: anytype) bool {
    if (a.dst != b.dst) return false;
    if (a.dst_row_stride == 1 and
        b.dst_row_stride == 1 and
        a.dst_col_stride == b.dst_col_stride and
        a.dst_col_stride > 0)
    {
        const stride: i64 = @intCast(a.dst_col_stride);
        const diff: i64 = @as(i64, @intCast(b.dst_offset)) - @as(i64, @intCast(a.dst_offset));
        const min_delta = -@as(i64, @intCast(a.cols)) + 1;
        const max_delta = @as(i64, @intCast(b.cols)) - 1;
        var delta = min_delta;
        while (delta <= max_delta) : (delta += 1) {
            const start_delta = diff + delta * stride;
            if (start_delta < @as(i64, @intCast(a.rows)) and -start_delta < @as(i64, @intCast(b.rows))) return true;
        }
        return false;
    }
    const a_span = stridedSpan(a.dst, a.dst_offset, a.rows, a.cols, a.dst_row_stride, a.dst_col_stride);
    const b_span = stridedSpan(b.dst, b.dst_offset, b.rows, b.cols, b.dst_row_stride, b.dst_col_stride);
    return a_span.overlaps(b_span);
}

pub fn opConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    indices: []const usize,
    candidate: backend_mod.DeviceOp,
) bool {
    for (indices) |idx| {
        if (opAccessConflicts(ops[idx], candidate)) return true;
    }
    return false;
}

pub fn canHoistProjectionTo(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    candidate_index: usize,
    q: anytype,
) bool {
    return canHoistOpTo(ops, start, candidate_index, .{ .qmatmul = q });
}

pub fn canBatchElementwiseOp(e: anytype) bool {
    return e.op.isFusible();
}

pub fn canHoistElementwiseTo(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    candidate_index: usize,
    e: anytype,
) bool {
    return canHoistOpTo(ops, start, candidate_index, .{ .elementwise = e });
}

pub fn elementwiseConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    indices: []const usize,
    e: anytype,
) bool {
    return opConflictsSelected(ops, indices, .{ .elementwise = e });
}

pub fn projectionConflictsSelected(
    ops: []const backend_mod.DeviceOp,
    indices: []const usize,
    q: anytype,
) bool {
    return opConflictsSelected(ops, indices, .{ .qmatmul = q });
}

pub fn projectionSidecarCompatible(q: anytype, op: backend_mod.DeviceOp) bool {
    return switch (op) {
        .slice_assign => |sa| if (q.M == 1) qmatvecSliceSidecarCompatible(q, sa) else qmatmulSliceSidecarCompatible(q, sa),
        .elementwise => |e| if (q.M == 1) qmatvecElementwiseSidecarCompatible(q, e) else qmatmulElementwiseSidecarCompatible(q, e),
        .fused_elementwise => |fe| qmatmulFusedElementwiseSidecarCompatible(q, fe),
        else => false,
    };
}

fn projectionSidecarMatchesPolicy(policy: ProjectionGroupPolicy, q: anytype, op: backend_mod.DeviceOp) bool {
    return switch (policy.kind) {
        .qmatvec => switch (op) {
            .slice_assign => |sa| qmatvecSliceSidecarCompatible(q, sa),
            .elementwise => |e| qmatvecElementwiseSidecarCompatible(q, e),
            else => false,
        },
        .qmatmul => switch (op) {
            .slice_assign => |sa| qmatmulSliceSidecarCompatible(q, sa),
            .elementwise => |e| qmatmulElementwiseSidecarCompatible(q, e),
            else => false,
        },
    };
}

fn projectionSelectionContainsIndex(selection: *const ProjectionGroupSelection, candidate: usize) bool {
    for (selection.anchorIndices()) |idx| {
        if (idx == candidate) return true;
    }
    for (selection.sidecarIndices()) |maybe_idx| {
        if (maybe_idx) |idx| {
            if (idx == candidate) return true;
        }
    }
    return false;
}

fn projectionWriteCoversRead(q: anytype, read: BufferSpan) bool {
    const q_access = opAccessSpans(.{ .qmatmul = q });
    for (q_access.writeSpans()) |write| {
        if (write.buf == read.buf and write.start <= read.start and write.end >= read.end) return true;
    }
    return q_access.write_overflow and opWritesBuffer(.{ .qmatmul = q }, read.buf);
}

fn canHoistProjectionSidecarToGroup(
    ops: []const backend_mod.DeviceOp,
    group_start: usize,
    sidecar_index: usize,
    q: anytype,
    sidecar: backend_mod.DeviceOp,
    selection: *const ProjectionGroupSelection,
) bool {
    const sidecar_access = opAccessSpans(sidecar);
    for (ops[group_start..sidecar_index], group_start..) |op, idx| {
        if (projectionSelectionContainsIndex(selection, idx)) continue;
        for (sidecar_access.readSpans()) |read| {
            if (projectionWriteCoversRead(q, read)) continue;
            if (opWritesSpan(op, read)) return false;
        }
        for (sidecar_access.writeSpans()) |write| {
            if (opTouchesSpan(op, write)) return false;
        }
        if (sidecar_access.read_overflow or sidecar_access.write_overflow) {
            if (opAccessConflicts(op, sidecar)) return false;
        }
    }
    return true;
}

pub fn qmatmulElementwiseSidecarCompatible(q: anytype, e: anytype) bool {
    if (q.M == 1) return false;
    if (e.op != .add and e.op != .mul) return false;
    if (e.n != q.M * q.N) return false;
    if (qmatmulDstRowStride(q) != q.N) return false;
    return (e.src0 == q.dst and e.src0_offset == q.dst_offset) or
        (e.src1 == q.dst and e.src1_offset == q.dst_offset);
}

pub fn qmatvecElementwiseSidecarCompatible(q: anytype, e: anytype) bool {
    if (q.M != 1) return false;
    if (e.op != .add and e.op != .mul) return false;
    if (e.n != q.N) return false;
    if (q.dst_row_stride != 0 and q.dst_row_stride != q.N) return false;
    const src0_primary = e.src0 == q.dst and e.src0_offset == q.dst_offset;
    const src1_primary = e.src1 == q.dst and e.src1_offset == q.dst_offset;
    return src0_primary != src1_primary;
}

pub fn qmatmulFusedElementwiseSidecarCompatible(q: anytype, fe: anytype) bool {
    if (q.M == 1) return false;
    if (fe.n != q.M * q.N) return false;
    if (qmatmulDstRowStride(q) != q.N) return false;
    return fe.src == q.dst and fe.src_offset == q.dst_offset;
}

pub fn qmatmulSliceSrcColStart(q: anytype, sa: anytype) ?u32 {
    if (sa.src_offset < q.dst_offset) return null;
    const delta = sa.src_offset - q.dst_offset;
    const dst_row_stride = qmatmulDstRowStride(q);
    if (delta >= dst_row_stride) return null;
    return delta;
}

pub fn qmatmulRopeSrcColStart(q: anytype, rr: anytype) ?u32 {
    if (rr.src_off < q.dst_offset) return null;
    const delta = rr.src_off - q.dst_offset;
    const dst_row_stride = qmatmulDstRowStride(q);
    if (delta >= dst_row_stride) return null;
    return delta;
}

pub fn qmatmulSliceSidecarCompatible(q: anytype, sa: anytype) bool {
    const slice_src_col_start = qmatmulSliceSrcColStart(q, sa) orelse return false;
    return q.M != 1 and
        q.dst == sa.src and
        slice_src_col_start + sa.rows <= q.N and
        q.M == sa.cols and
        sa.src_row_stride == 1 and
        sa.src_col_stride == qmatmulDstRowStride(q);
}

pub fn qmatmulRopeSidecarCompatible(q: anytype, rr: anytype) bool {
    return q.M != 1 and projectionRopeSidecarCompatible(q, rr);
}

pub fn qmatvecRopeSidecarCompatible(q: anytype, rr: anytype) bool {
    return q.M == 1 and projectionRopeSidecarCompatible(q, rr);
}

pub fn projectionRopeSidecarCompatible(q: anytype, rr: anytype) bool {
    const rope_src_col_start = qmatmulRopeSrcColStart(q, rr) orelse return false;
    const d = rr.half_d * 2;
    return q.dst == rr.src and
        rope_src_col_start + d <= q.N and
        q.M == rr.seq_len and
        rr.src_rs == 1 and
        rr.src_cs == qmatmulDstRowStride(q);
}

pub fn qmatmulRopeStoreSidecarCompatible(q: anytype, rr: anytype, sa: anytype) bool {
    return qmatmulRopeSidecarCompatible(q, rr) and ropeSliceAssignCompatible(rr, sa);
}

pub fn qmatvecRopeStoreSidecarCompatible(q: anytype, rr: anytype, sa: anytype) bool {
    return qmatvecRopeSidecarCompatible(q, rr) and ropeSliceAssignCompatible(rr, sa);
}

pub fn projectionRopeStoreSidecarCompatible(q: anytype, rr: anytype, sa: anytype) bool {
    return projectionRopeSidecarCompatible(q, rr) and ropeSliceAssignCompatible(rr, sa);
}

pub fn qmatmulRopeStoreTilePairCompatible(q: anytype, rr: anytype, sa: anytype, tile_cols: u32) bool {
    return tile_cols != 0 and
        qmatmulRopeStoreSidecarCompatible(q, rr, sa) and
        rr.half_d >= tile_cols and
        rr.half_d % tile_cols == 0;
}

fn projectionRopeSidecarOpCompatible(q: anytype, op: backend_mod.DeviceOp) bool {
    return switch (op) {
        .rope => |rr| projectionRopeSidecarCompatible(q, rr),
        else => false,
    };
}

pub fn qmatvecSliceSidecarCompatible(q: anytype, sa: anytype) bool {
    const slice_src_col_start = qmatmulSliceSrcColStart(q, sa) orelse return false;
    return q.M == 1 and
        q.dst == sa.src and
        slice_src_col_start + sa.rows <= q.N and
        sa.cols == 1 and
        sa.src_row_stride == 1 and
        (sa.src_col_stride == q.N or sa.src_col_stride == sa.rows);
}

pub fn qmatmulDstRowStride(q: anytype) u32 {
    return if (q.dst_row_stride != 0) q.dst_row_stride else q.N;
}

pub const AttentionOperand = enum { q, k, v };

pub fn attentionSliceAssignOperand(sa: anytype, att: anytype) ?AttentionOperand {
    if (attentionSliceMatches(sa, att.q, att.q_off, att.d_head, att.seq_q, att.q_rs, att.q_cs)) return .q;
    if (attentionSliceMatches(sa, att.k, att.k_off, att.d_head, att.seq_kv, att.k_rs, att.k_cs)) return .k;
    if (attentionSliceMatches(sa, att.v, att.v_off, att.d_head, att.seq_kv, att.v_rs, att.v_cs)) return .v;
    return null;
}

pub fn attentionSliceStoreCompatible(att: anytype, sa: anytype) bool {
    return sa.src == att.dst and
        sa.src_offset == att.dst_off and
        sa.rows == att.d_head and
        sa.cols == att.seq_q and
        sa.src_row_stride == att.dst_rs and
        sa.src_col_stride == att.dst_cs;
}

pub fn ropeAttentionCompatible(rr: anytype, att: anytype) bool {
    const d = rr.half_d * 2;
    return rr.dst == att.q and
        rr.dst_off == att.q_off and
        d == att.d_head and
        rr.seq_len == att.seq_q and
        att.q_rs == 1 and
        att.q_cs == d;
}

pub fn ropeStoreBatchGeometryCompatible(first: anytype, next: anytype) bool {
    return first.half_d == next.half_d and
        first.seq_len == next.seq_len and
        first.src_rs == next.src_rs and
        first.src_cs == next.src_cs and
        first.cs_cs == next.cs_cs;
}

pub fn ropeAttentionStoreCompactBatchCompatible(
    ops: []const backend_mod.DeviceOp,
    command: *const ProgramCommand,
) bool {
    if (command.kind != .rope_attention_store_group) return false;
    if (command.sidecar_count < 2 or command.anchor_count != command.sidecar_count * 2) return false;
    const first_sa_idx = command.sidecar_indices[0] orelse return false;
    if (command.indices[0] >= ops.len or command.indices[1] >= ops.len or first_sa_idx >= ops.len) return false;
    const first_rope = switch (ops[command.indices[0]]) {
        .rope => |rr| rr,
        else => return false,
    };
    const first_att = switch (ops[command.indices[1]]) {
        .attention => |att| att,
        else => return false,
    };
    const first_sa = switch (ops[first_sa_idx]) {
        .slice_assign => |sa| sa,
        else => return false,
    };

    var i: usize = 0;
    while (i < command.sidecar_count) : (i += 1) {
        const rope_idx = command.indices[i * 2];
        const att_idx = command.indices[i * 2 + 1];
        const sa_idx = command.sidecar_indices[i] orelse return false;
        if (rope_idx >= ops.len or att_idx >= ops.len or sa_idx >= ops.len) return false;
        const rr = switch (ops[rope_idx]) {
            .rope => |rr| rr,
            else => return false,
        };
        const att = switch (ops[att_idx]) {
            .attention => |att| att,
            else => return false,
        };
        const sa = switch (ops[sa_idx]) {
            .slice_assign => |sa| sa,
            else => return false,
        };
        if (!ropeAttentionCompatible(rr, att)) return false;
        if (!ropeStoreBatchGeometryCompatible(first_rope, rr)) return false;
        if (!attentionGeometryCompatible(first_att, att)) return false;
        if (!attentionSliceStoreCompatible(att, sa)) return false;
        if (rr.src != first_rope.src or rr.cos_sin != first_rope.cos_sin or
            att.k != first_att.k or att.v != first_att.v or att.mask != first_att.mask or
            sa.dst != first_sa.dst) return false;
        if (sa.dst_row_stride != first_sa.dst_row_stride or
            sa.dst_col_stride != first_sa.dst_col_stride) return false;
    }
    return true;
}

fn attentionSliceMatches(
    sa: anytype,
    buf: u16,
    offset: u32,
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
) bool {
    return sa.dst == buf and
        sa.dst_offset == offset and
        sa.rows == rows and
        sa.cols == cols and
        sa.dst_row_stride == row_stride and
        sa.dst_col_stride == col_stride;
}

pub fn isRopeSliceAssignChain(
    a: backend_mod.DeviceOp,
    b: backend_mod.DeviceOp,
) bool {
    const rr = switch (a) {
        .rope => |rr| rr,
        else => return false,
    };
    const sa = switch (b) {
        .slice_assign => |sa| sa,
        else => return false,
    };
    return ropeSliceAssignCompatible(rr, sa);
}

pub fn ropeSliceAssignCompatible(rr: anytype, sa: anytype) bool {
    const d = rr.half_d * 2;
    return rr.dst == sa.src and
        rr.dst_off == sa.src_offset and
        sa.rows == d and
        sa.cols == rr.seq_len and
        sa.src_row_stride == 1 and
        sa.src_col_stride == d;
}

pub fn ropeStoreGroupCompatible(first_rope: anytype, first_sa: anytype, next_rope: anytype, next_sa: anytype) bool {
    return first_rope.cos_sin == next_rope.cos_sin and
        first_sa.dst == next_sa.dst and
        first_rope.half_d == next_rope.half_d and
        first_rope.seq_len == next_rope.seq_len and
        first_rope.src_rs == next_rope.src_rs and
        first_rope.src_cs == next_rope.src_cs and
        first_rope.cs_cs == next_rope.cs_cs and
        first_sa.dst_row_stride == next_sa.dst_row_stride and
        first_sa.dst_col_stride == next_sa.dst_col_stride;
}

pub fn ropeBatchCompatible(first: anytype, next: anytype) bool {
    return first.src == next.src and
        first.cos_sin == next.cos_sin and
        first.dst == next.dst and
        first.half_d == next.half_d and
        first.seq_len == next.seq_len and
        first.src_rs == next.src_rs and
        first.src_cs == next.src_cs and
        first.cs_cs == next.cs_cs;
}

pub fn ropeBatchRunLen(ops: []const backend_mod.DeviceOp, max_ops: u32) usize {
    if (ops.len < 2 or max_ops < 2) return 0;
    const first = switch (ops[0]) {
        .rope => |rr| rr,
        else => return 0,
    };
    var n: usize = 1;
    const limit = @min(ops.len, @as(usize, @intCast(max_ops)));
    while (n < limit) : (n += 1) {
        const next = switch (ops[n]) {
            .rope => |rr| rr,
            else => break,
        };
        if (!ropeBatchCompatible(first, next)) break;
    }
    return if (n >= 2) n else 0;
}

pub fn sliceAssignBatchCompatible(first: anytype, next: anytype) bool {
    return first.src == next.src and first.dst == next.dst;
}

pub fn sliceAssignBatchRunLen(ops: []const backend_mod.DeviceOp, max_ops: u32) usize {
    if (ops.len < 2 or max_ops < 2) return 0;
    const first = switch (ops[0]) {
        .slice_assign => |sa| sa,
        else => return 0,
    };
    var n: usize = 1;
    const limit = @min(ops.len, @as(usize, @intCast(max_ops)));
    while (n < limit) : (n += 1) {
        const next = switch (ops[n]) {
            .slice_assign => |sa| sa,
            else => break,
        };
        if (!sliceAssignBatchCompatible(first, next)) break;
    }
    return if (n >= 2) n else 0;
}

pub fn attentionBatchCompatible(first: anytype, next: anytype) bool {
    return first.q == next.q and
        first.k == next.k and
        first.v == next.v and
        first.mask == next.mask and
        first.dst == next.dst and
        first.has_mask == next.has_mask and
        first.d_head == next.d_head and
        first.seq_q == next.seq_q and
        first.seq_kv == next.seq_kv and
        first.scale == next.scale and
        first.q_rs == next.q_rs and
        first.q_cs == next.q_cs and
        first.k_rs == next.k_rs and
        first.k_cs == next.k_cs and
        first.v_rs == next.v_rs and
        first.v_cs == next.v_cs and
        first.mask_rs == next.mask_rs and
        first.mask_cs == next.mask_cs and
        first.dst_rs == next.dst_rs and
        first.dst_cs == next.dst_cs;
}

pub fn attentionGeometryCompatible(first: anytype, next: anytype) bool {
    return first.has_mask == next.has_mask and
        first.d_head == next.d_head and
        first.seq_q == next.seq_q and
        first.seq_kv == next.seq_kv and
        first.scale == next.scale and
        first.q_rs == next.q_rs and
        first.q_cs == next.q_cs and
        first.k_rs == next.k_rs and
        first.k_cs == next.k_cs and
        first.v_rs == next.v_rs and
        first.v_cs == next.v_cs and
        first.mask_rs == next.mask_rs and
        first.mask_cs == next.mask_cs and
        first.dst_rs == next.dst_rs and
        first.dst_cs == next.dst_cs;
}

pub fn attentionBatchRunLen(ops: []const backend_mod.DeviceOp, max_ops: u32) usize {
    if (ops.len < 2 or max_ops < 2) return 0;
    const first = switch (ops[0]) {
        .attention => |att| att,
        else => return 0,
    };
    var n: usize = 1;
    const limit = @min(ops.len, @as(usize, @intCast(max_ops)));
    while (n < limit) : (n += 1) {
        const next = switch (ops[n]) {
            .attention => |att| att,
            else => break,
        };
        if (!attentionBatchCompatible(first, next)) break;
    }
    return if (n >= 2) n else 0;
}

pub fn isRmsnormScaleChain(
    a: backend_mod.DeviceOp,
    b: backend_mod.DeviceOp,
    c: backend_mod.DeviceOp,
) bool {
    const rn = switch (a) {
        .rmsnorm => |rn| rn,
        else => return false,
    };
    const rp = switch (b) {
        .repeat => |rp| rp,
        else => return false,
    };
    const e = switch (c) {
        .elementwise => |e| e,
        else => return false,
    };

    const n = rn.rows * rn.cols;
    return e.op == .mul and
        rp.n == n and
        e.n == n and
        rp.dst == (if (e.src0 == rn.dst and e.src0_offset == rn.dst_offset) e.src1 else e.src0) and
        rp.src_ne[0] == rn.cols and
        rp.src_ne[1] == 1 and
        rp.dst_ne[0] == rn.cols and
        rp.dst_ne[1] == rn.rows and
        rp.src_strides[0] == 1 and
        rp.dst_strides[0] == 1 and
        rp.dst_strides[1] == rn.cols and
        mulSourcesMatchNormAndScale(e, rn.dst, rn.dst_offset, rp.dst, rp.dst_offset);
}

fn mulSourcesMatchNormAndScale(e: anytype, norm_buf: u16, norm_offset: u32, scale_buf: u16, scale_offset: u32) bool {
    return (e.src0 == norm_buf and e.src0_offset == norm_offset and e.src1 == scale_buf and e.src1_offset == scale_offset) or
        (e.src1 == norm_buf and e.src1_offset == norm_offset and e.src0 == scale_buf and e.src0_offset == scale_offset);
}

pub fn kernelFamily(op: backend_mod.DeviceOp) KernelFamily {
    return switch (op) {
        .elementwise => .elementwise,
        .fused_elementwise => .fused_elementwise,
        .softmax, .layernorm, .rmsnorm => .row,
        .reduce => .reduce,
        .repeat, .slice_assign => .movement,
        .matmul => .matmul,
        .qmatmul => |q| if (q.M == 1) .qmatvec else .qmatmul,
        .rope => .rope,
        .attention => .attention,
    };
}

pub fn executionClass(op: backend_mod.DeviceOp, policy: SchedulePolicy) ExecutionClass {
    if (!policy.capabilities.supportsOp(op)) return .fallback;

    const family = kernelFamily(op);
    if (!policy.native_kernels.supports(family)) return .fallback;

    const can_use_backend = switch (op) {
        .matmul => |m| policy.fine_grained or m.geom.M >= @as(usize, policy.min_backend_matmul_m),
        .qmatmul => |q| if (q.M == 1) policy.fine_grained else policy.fine_grained or q.M >= policy.min_backend_qmatmul_m,
        .elementwise,
        .fused_elementwise,
        .softmax,
        .layernorm,
        .rmsnorm,
        .reduce,
        .repeat,
        .slice_assign,
        .rope,
        .attention,
        => policy.fine_grained,
    };

    return if (can_use_backend) .backend else .fallback;
}

pub fn buildKernelSchedule(
    alloc: std.mem.Allocator,
    ops: []const backend_mod.DeviceOp,
    policy: SchedulePolicy,
) ![]KernelItem {
    var items: std.ArrayListUnmanaged(KernelItem) = .empty;
    errdefer items.deinit(alloc);

    for (ops, 0..) |op, i| {
        const next = KernelItem{
            .family = kernelFamily(op),
            .execution = executionClass(op, policy),
            .start = @intCast(i),
            .len = 1,
        };

        if (items.items.len > 0) {
            const last = &items.items[items.items.len - 1];
            if (last.family == next.family and
                last.execution == next.execution and
                last.start + last.len == next.start)
            {
                last.len += 1;
                continue;
            }
        }

        try items.append(alloc, next);
    }

    return items.toOwnedSlice(alloc);
}

pub fn scheduleShapeMatches(
    ops: []const backend_mod.DeviceOp,
    items: []const KernelItem,
    policy: SchedulePolicy,
) bool {
    var item_index: usize = 0;
    var current: ?KernelItem = null;

    for (ops, 0..) |op, i| {
        const next = KernelItem{
            .family = kernelFamily(op),
            .execution = executionClass(op, policy),
            .start = @intCast(i),
            .len = 1,
        };

        if (current) |*item| {
            if (item.family == next.family and
                item.execution == next.execution and
                item.start + item.len == next.start)
            {
                item.len += 1;
                continue;
            }

            if (item_index >= items.len or !kernelItemsEqual(item.*, items[item_index])) return false;
            item_index += 1;
            current = next;
        } else {
            current = next;
        }
    }

    if (current) |item| {
        if (item_index >= items.len or !kernelItemsEqual(item, items[item_index])) return false;
        item_index += 1;
    }

    return item_index == items.len;
}

fn kernelItemsEqual(a: KernelItem, b: KernelItem) bool {
    return a.family == b.family and
        a.execution == b.execution and
        a.start == b.start and
        a.len == b.len;
}

pub fn buildKernelRegions(
    alloc: std.mem.Allocator,
    items: []const KernelItem,
    policy: RegionPolicy,
) ![]KernelRegion {
    var regions: std.ArrayListUnmanaged(KernelRegion) = .empty;
    errdefer regions.deinit(alloc);

    var run_start: usize = 0;
    while (run_start < items.len) {
        while (run_start < items.len and !policy.member_families.contains(items[run_start].family)) {
            run_start += 1;
        }
        if (run_start >= items.len) break;

        var run_end = run_start;
        var anchor_count: u32 = 0;
        while (run_end < items.len and policy.member_families.contains(items[run_end].family)) : (run_end += 1) {
            if (policy.anchor_families.contains(items[run_end].family)) anchor_count += items[run_end].len;
        }

        if (anchor_count > 0) {
            const first = items[run_start];
            const last = items[run_end - 1];
            const op_end = last.start + last.len;
            try regions.append(alloc, .{
                .start_item = @intCast(run_start),
                .item_count = @intCast(run_end - run_start),
                .op_start = first.start,
                .op_count = op_end - first.start,
                .anchor_count = anchor_count,
            });
        }

        run_start = run_end;
    }

    return regions.toOwnedSlice(alloc);
}

/// Emit one region per contiguous run of anchor-family items. Because
/// buildKernelSchedule already coalesces adjacent ops of the same family,
/// this exposes reusable projection groups such as "three qmatvec ops in a
/// row" without knowing whether they came from attention, FFN, or any model.
pub fn buildAnchorRunRegions(
    alloc: std.mem.Allocator,
    items: []const KernelItem,
    policy: RegionPolicy,
) ![]KernelRegion {
    var regions: std.ArrayListUnmanaged(KernelRegion) = .empty;
    errdefer regions.deinit(alloc);

    var i: usize = 0;
    while (i < items.len) {
        while (i < items.len and !policy.anchor_families.contains(items[i].family)) {
            i += 1;
        }
        if (i >= items.len) break;

        const start = i;
        var anchor_count: u32 = 0;
        while (i < items.len and policy.anchor_families.contains(items[i].family)) : (i += 1) {
            anchor_count += items[i].len;
        }

        const first = items[start];
        const last = items[i - 1];
        const op_end = last.start + last.len;
        try regions.append(alloc, .{
            .start_item = @intCast(start),
            .item_count = @intCast(i - start),
            .op_start = first.start,
            .op_count = op_end - first.start,
            .anchor_count = anchor_count,
        });
    }

    return regions.toOwnedSlice(alloc);
}

/// Split anchored member runs into non-overlapping windows with at least
/// `anchors_per_region` anchors. Schedule items are not split, so a coalesced
/// anchor item may make a region contain more anchors than requested.
pub fn buildAnchorWindowRegions(
    alloc: std.mem.Allocator,
    items: []const KernelItem,
    policy: RegionPolicy,
    anchors_per_region: u32,
) ![]KernelRegion {
    var regions: std.ArrayListUnmanaged(KernelRegion) = .empty;
    errdefer regions.deinit(alloc);

    if (anchors_per_region == 0) return regions.toOwnedSlice(alloc);

    var run_start: usize = 0;
    while (run_start < items.len) {
        while (run_start < items.len and !policy.member_families.contains(items[run_start].family)) {
            run_start += 1;
        }
        if (run_start >= items.len) break;

        var run_end = run_start;
        while (run_end < items.len and policy.member_families.contains(items[run_end].family)) : (run_end += 1) {}

        var window_start = run_start;
        while (window_start < run_end) {
            var window_end = window_start;
            var anchor_count: u32 = 0;

            while (window_end < run_end and anchor_count < anchors_per_region) : (window_end += 1) {
                if (policy.anchor_families.contains(items[window_end].family)) {
                    anchor_count += items[window_end].len;
                }
            }

            if (anchor_count < anchors_per_region) break;

            while (window_end < run_end and !policy.anchor_families.contains(items[window_end].family)) : (window_end += 1) {}

            const first = items[window_start];
            const last = items[window_end - 1];
            const op_end = last.start + last.len;
            try regions.append(alloc, .{
                .start_item = @intCast(window_start),
                .item_count = @intCast(window_end - window_start),
                .op_start = first.start,
                .op_count = op_end - first.start,
                .anchor_count = anchor_count,
            });

            window_start = window_end;
        }

        run_start = run_end;
    }

    return regions.toOwnedSlice(alloc);
}

pub fn buildStagePatternRegions(
    alloc: std.mem.Allocator,
    items: []const KernelItem,
    stage: StagePolicy,
) ![]PatternRegion {
    var pattern_regions: std.ArrayListUnmanaged(PatternRegion) = .empty;
    errdefer pattern_regions.deinit(alloc);

    const regions = try buildAnchorWindowRegions(
        alloc,
        items,
        stage.region_policy,
        stage.anchors_per_stage,
    );
    defer alloc.free(regions);

    for (regions) |region| {
        if (region.item_count < stage.min_items_per_stage) continue;
        if (region.op_count < stage.min_ops_per_stage) continue;
        try pattern_regions.append(alloc, .{
            .pattern_index = stage.pattern_index,
            .region = region,
        });
    }

    return pattern_regions.toOwnedSlice(alloc);
}

pub fn buildStagePlan(
    alloc: std.mem.Allocator,
    items: []const KernelItem,
    stages: []const StagePolicy,
) ![]PatternRegion {
    var candidates: std.ArrayListUnmanaged(PatternRegion) = .empty;
    defer candidates.deinit(alloc);

    for (stages) |stage| {
        const regions = try buildStagePatternRegions(alloc, items, stage);
        defer alloc.free(regions);
        for (regions) |region| {
            try candidates.append(alloc, region);
        }
    }

    return selectPatternRegions(alloc, candidates.items);
}

pub fn buildStageRegionSchedule(
    alloc: std.mem.Allocator,
    items: []const KernelItem,
    stages: []const StagePolicy,
) ![]ScheduleUnit {
    const plan = try buildStagePlan(alloc, items, stages);
    defer alloc.free(plan);
    return buildRegionSchedule(alloc, items, plan);
}

pub fn buildFamilyPatternRegions(
    alloc: std.mem.Allocator,
    items: []const KernelItem,
    pattern: []const KernelFamily,
) ![]KernelRegion {
    var regions: std.ArrayListUnmanaged(KernelRegion) = .empty;
    errdefer regions.deinit(alloc);

    if (pattern.len == 0 or pattern.len > items.len) return regions.toOwnedSlice(alloc);

    var i: usize = 0;
    while (i + pattern.len <= items.len) : (i += 1) {
        for (pattern, 0..) |family, j| {
            if (items[i + j].family != family) break;
        } else {
            const first = items[i];
            const last = items[i + pattern.len - 1];
            const op_end = last.start + last.len;
            var anchor_count: u32 = 0;
            for (items[i .. i + pattern.len]) |item| {
                anchor_count += item.len;
            }
            try regions.append(alloc, .{
                .start_item = @intCast(i),
                .item_count = @intCast(pattern.len),
                .op_start = first.start,
                .op_count = op_end - first.start,
                .anchor_count = anchor_count,
            });
        }
    }

    return regions.toOwnedSlice(alloc);
}

pub fn buildFamilyPatternPlan(
    alloc: std.mem.Allocator,
    items: []const KernelItem,
    patterns: []const FamilyPattern,
) ![]PatternRegion {
    var candidates: std.ArrayListUnmanaged(PatternRegion) = .empty;
    defer candidates.deinit(alloc);

    for (patterns, 0..) |pattern, pattern_index| {
        const matches = try buildFamilyPatternRegions(alloc, items, pattern.families);
        defer alloc.free(matches);
        for (matches) |region| {
            try candidates.append(alloc, .{
                .pattern_index = @intCast(pattern_index),
                .region = region,
            });
        }
    }

    sortPatternRegions(candidates.items);

    var selected: std.ArrayListUnmanaged(PatternRegion) = .empty;
    errdefer selected.deinit(alloc);

    var next_free_item: u32 = 0;
    for (candidates.items) |candidate| {
        if (candidate.region.start_item < next_free_item) continue;
        try selected.append(alloc, candidate);
        next_free_item = candidate.region.start_item + candidate.region.item_count;
    }

    return selected.toOwnedSlice(alloc);
}

pub fn selectPatternRegions(
    alloc: std.mem.Allocator,
    candidates: []const PatternRegion,
) ![]PatternRegion {
    const sorted = try alloc.dupe(PatternRegion, candidates);
    defer alloc.free(sorted);
    sortPatternRegions(sorted);

    var selected: std.ArrayListUnmanaged(PatternRegion) = .empty;
    errdefer selected.deinit(alloc);

    var next_free_item: u32 = 0;
    for (sorted) |candidate| {
        if (candidate.region.start_item < next_free_item) continue;
        try selected.append(alloc, candidate);
        next_free_item = candidate.region.start_item + candidate.region.item_count;
    }

    return selected.toOwnedSlice(alloc);
}

pub fn buildRegionSchedule(
    alloc: std.mem.Allocator,
    items: []const KernelItem,
    pattern_regions: []const PatternRegion,
) ![]ScheduleUnit {
    var units: std.ArrayListUnmanaged(ScheduleUnit) = .empty;
    errdefer units.deinit(alloc);

    var item_index: usize = 0;
    for (pattern_regions) |pattern_region| {
        const region = pattern_region.region;
        const region_start: usize = @intCast(region.start_item);
        if (region_start < item_index) continue;
        while (item_index < region_start) : (item_index += 1) {
            try units.append(alloc, itemScheduleUnit(@intCast(item_index), items[item_index]));
        }

        try units.append(alloc, .{
            .kind = .pattern_region,
            .pattern_index = pattern_region.pattern_index,
            .start_item = region.start_item,
            .item_count = region.item_count,
            .op_start = region.op_start,
            .op_count = region.op_count,
        });
        item_index = @intCast(region.start_item + region.item_count);
    }

    while (item_index < items.len) : (item_index += 1) {
        try units.append(alloc, itemScheduleUnit(@intCast(item_index), items[item_index]));
    }

    return units.toOwnedSlice(alloc);
}

pub const RegionCommandPlan = struct {
    commands: []const ProgramCommand = &.{},

    pub fn deinit(self: RegionCommandPlan, alloc: std.mem.Allocator) void {
        if (self.commands.len > 0) alloc.free(self.commands);
    }
};

pub fn deinitRegionCommandPlans(alloc: std.mem.Allocator, plans: []const RegionCommandPlan) void {
    for (plans) |plan| plan.deinit(alloc);
}

pub fn buildRegionCommandPlans(
    alloc: std.mem.Allocator,
    ops: []const backend_mod.DeviceOp,
    units: []const ScheduleUnit,
    policy: CommandStreamPolicy,
) ![]RegionCommandPlan {
    if (units.len == 0) return &.{};

    const plans = try alloc.alloc(RegionCommandPlan, units.len);
    errdefer alloc.free(plans);
    @memset(plans, .{});
    errdefer deinitRegionCommandPlans(alloc, plans);

    for (units, 0..) |unit, i| {
        if (unit.kind != .pattern_region or unit.op_count > 256) continue;
        const start: usize = @intCast(unit.op_start);
        const end = start + @as(usize, unit.op_count);
        if (end > ops.len) continue;
        plans[i].commands = try buildProgramCommands(alloc, ops[start..end], policy);
    }

    return plans;
}

pub const ExecutionPlan = struct {
    schedule: []const KernelItem = &.{},
    regions: []const ScheduleUnit = &.{},
    region_commands: []const RegionCommandPlan = &.{},
    region_command_policy: CommandStreamPolicy = .{},

    pub fn deinit(self: ExecutionPlan, alloc: std.mem.Allocator) void {
        if (self.schedule.len > 0) alloc.free(self.schedule);
        if (self.regions.len > 0) alloc.free(self.regions);
        deinitRegionCommandPlans(alloc, self.region_commands);
        if (self.region_commands.len > 0) alloc.free(self.region_commands);
    }

    pub fn shapeMatches(
        self: ExecutionPlan,
        ops: []const backend_mod.DeviceOp,
        policy: SchedulePolicy,
    ) bool {
        return scheduleShapeMatches(ops, self.schedule, policy);
    }

    pub fn regionCommandPlan(
        self: ExecutionPlan,
        unit_index: usize,
        policy: CommandStreamPolicy,
    ) []const ProgramCommand {
        if (!std.meta.eql(self.region_command_policy, policy)) return &.{};
        if (unit_index >= self.region_commands.len) return &.{};
        return self.region_commands[unit_index].commands;
    }
};

pub fn buildExecutionPlan(
    alloc: std.mem.Allocator,
    ops: []const backend_mod.DeviceOp,
    schedule_policy: SchedulePolicy,
    stages: []const StagePolicy,
    command_policy: CommandStreamPolicy,
) !ExecutionPlan {
    const schedule = try buildKernelSchedule(alloc, ops, schedule_policy);
    errdefer if (schedule.len > 0) alloc.free(schedule);

    const regions = buildStageRegionSchedule(alloc, schedule, stages) catch &.{};
    errdefer if (regions.len > 0) alloc.free(regions);

    const region_commands = buildRegionCommandPlans(alloc, ops, regions, command_policy) catch &.{};
    errdefer {
        deinitRegionCommandPlans(alloc, region_commands);
        if (region_commands.len > 0) alloc.free(region_commands);
    }

    return .{
        .schedule = schedule,
        .regions = regions,
        .region_commands = region_commands,
        .region_command_policy = command_policy,
    };
}

pub fn summarizeRegionExecution(
    units: []const ScheduleUnit,
    items: []const KernelItem,
    backend_pattern_indices: []const u32,
) RegionExecutionSummary {
    var summary = RegionExecutionSummary{ .units = @intCast(units.len) };
    var prev_execution: ?ExecutionClass = null;
    var current_backend_island_units: u32 = 0;
    var current_backend_island_ops: u32 = 0;

    for (units) |unit| {
        const execution = scheduleUnitExecution(unit, items, backend_pattern_indices);
        summary.ops += unit.op_count;
        switch (execution) {
            .backend => {
                summary.backend_units += 1;
                summary.backend_ops += unit.op_count;
                if (prev_execution == null or prev_execution.? != .backend) {
                    summary.backend_islands += 1;
                    current_backend_island_units = 0;
                    current_backend_island_ops = 0;
                }
                current_backend_island_units += 1;
                current_backend_island_ops += unit.op_count;
                summary.max_backend_island_units = @max(summary.max_backend_island_units, current_backend_island_units);
                summary.max_backend_island_ops = @max(summary.max_backend_island_ops, current_backend_island_ops);
            },
            .fallback => {
                summary.fallback_units += 1;
                summary.fallback_ops += unit.op_count;
            },
        }
        if (prev_execution) |prev| {
            if (prev != execution) summary.execution_transitions += 1;
        }
        prev_execution = execution;
    }

    return summary;
}

pub fn scheduleUnitExecution(
    unit: ScheduleUnit,
    items: []const KernelItem,
    backend_pattern_indices: []const u32,
) ExecutionClass {
    return switch (unit.kind) {
        .item => items[@intCast(unit.start_item)].execution,
        .pattern_region => if (containsPatternIndex(backend_pattern_indices, unit.pattern_index)) .backend else .fallback,
    };
}

fn containsPatternIndex(indices: []const u32, pattern_index: u32) bool {
    for (indices) |idx| {
        if (idx == pattern_index) return true;
    }
    return false;
}

fn itemScheduleUnit(item_index: u32, item: KernelItem) ScheduleUnit {
    return .{
        .kind = .item,
        .start_item = item_index,
        .item_count = 1,
        .op_start = item.start,
        .op_count = item.len,
    };
}

fn sortPatternRegions(regions: []PatternRegion) void {
    if (regions.len < 2) return;
    for (1..regions.len) |i| {
        const tmp = regions[i];
        var j = i;
        while (j > 0 and patternRegionLess(tmp, regions[j - 1])) : (j -= 1) {
            regions[j] = regions[j - 1];
        }
        regions[j] = tmp;
    }
}

fn patternRegionLess(lhs: PatternRegion, rhs: PatternRegion) bool {
    if (lhs.region.start_item != rhs.region.start_item) {
        return lhs.region.start_item < rhs.region.start_item;
    }
    if (lhs.region.item_count != rhs.region.item_count) {
        return lhs.region.item_count > rhs.region.item_count;
    }
    return lhs.pattern_index < rhs.pattern_index;
}

fn testElementwise(op: backend_mod.Op) backend_mod.DeviceOp {
    return .{ .elementwise = .{ .op = op, .dst = 0, .src0 = 0, .src1 = 0, .n = 1 } };
}

fn testMatmul(rows: usize) backend_mod.DeviceOp {
    return .{ .matmul = .{
        .dst = 0,
        .a = 0,
        .b = 0,
        .geom = .{
            .M = rows,
            .N = 4,
            .K = 4,
            .a_row_stride = 4,
            .a_col_stride = 1,
            .b_row_stride = 4,
            .b_col_stride = 1,
            .a_offset = 0,
            .b_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 4,
        },
    } };
}

fn testQMatmul(rows: u32) backend_mod.DeviceOp {
    return .{ .qmatmul = .{ .dst = 0, .input = 0, .weight_idx = 0, .M = rows, .N = 4, .K = 4 } };
}

fn testQMatmulWith(dst: u16, input: u16, rows: u32) backend_mod.DeviceOp {
    return .{ .qmatmul = .{ .dst = dst, .input = input, .weight_idx = 0, .M = rows, .N = 4, .K = 4 } };
}

fn testSliceAssign(dst_offset: u32) backend_mod.DeviceOp {
    return .{ .slice_assign = .{
        .dst = 0,
        .src = 1,
        .rows = 1,
        .cols = 4,
        .dst_base_offset = 0,
        .dst_offset = dst_offset,
        .dst_row_stride = 4,
        .dst_col_stride = 1,
        .src_offset = 0,
        .src_row_stride = 4,
        .src_col_stride = 1,
        .patch_stride = 4,
    } };
}

fn testQMatmulSidecar(src: u16, dst: u16) backend_mod.DeviceOp {
    return .{ .slice_assign = .{
        .dst = dst,
        .src = src,
        .rows = 4,
        .cols = 2,
        .dst_base_offset = 0,
        .dst_offset = 4,
        .dst_row_stride = 1,
        .dst_col_stride = 4,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 4,
        .patch_stride = 4,
    } };
}

fn testRopeSliceAssignOps() [2]backend_mod.DeviceOp {
    return .{
        .{ .rope = .{
            .dst = 2,
            .src = 0,
            .cos_sin = 1,
            .half_d = 2,
            .seq_len = 3,
            .src_off = 0,
            .cs_off = 0,
            .dst_off = 8,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 4,
        } },
        .{ .slice_assign = .{
            .dst = 3,
            .src = 2,
            .rows = 4,
            .cols = 3,
            .dst_base_offset = 0,
            .dst_offset = 4,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 8,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
    };
}

fn testAttention(offset: u32) backend_mod.DeviceOp {
    return .{ .attention = .{
        .dst = 4,
        .q = 0,
        .k = 1,
        .v = 2,
        .mask = 3,
        .has_mask = true,
        .d_head = 4,
        .seq_q = 2,
        .seq_kv = 4,
        .scale = 0.5,
        .q_off = offset,
        .k_off = 0,
        .v_off = 0,
        .mask_off = 0,
        .dst_off = offset,
        .q_rs = 1,
        .q_cs = 4,
        .k_rs = 1,
        .k_cs = 4,
        .v_rs = 1,
        .v_cs = 4,
        .mask_rs = 1,
        .mask_cs = 4,
        .dst_rs = 1,
        .dst_cs = 4,
    } };
}

fn testRmsnormScaleOps() [3]backend_mod.DeviceOp {
    return .{
        .{ .rmsnorm = .{
            .dst = 1,
            .src = 0,
            .rows = 2,
            .cols = 4,
            .eps = 1e-5,
        } },
        .{ .repeat = .{
            .dst = 2,
            .src = 3,
            .n = 8,
            .src_ne = .{ 4, 1, 1, 1 },
            .dst_ne = .{ 4, 2, 1, 1 },
            .src_strides = .{ 1, 4, 4, 4 },
            .dst_strides = .{ 1, 4, 8, 8 },
        } },
        .{ .elementwise = .{
            .op = .mul,
            .dst = 4,
            .src0 = 1,
            .src1 = 2,
            .n = 8,
        } },
    };
}

fn expectKernelItem(item: KernelItem, family: KernelFamily, execution: ExecutionClass, start: u32, len: u32) !void {
    try std.testing.expectEqual(family, item.family);
    try std.testing.expectEqual(execution, item.execution);
    try std.testing.expectEqual(start, item.start);
    try std.testing.expectEqual(len, item.len);
}

test "program command index set carries flat sidecars once" {
    var command = ProgramCommand{
        .kind = .projection_cache_group,
        .op_start = 2,
        .op_count = 8,
        .anchor_count = 2,
        .sidecar_count = 3,
    };
    command.indices[0] = 5;
    command.indices[1] = 2;
    command.sidecar_indices[0] = 6;
    command.sidecar_indices[1] = 5;
    command.sidecar_indices[2] = 9;

    const indices = command.explicitIndexSet();
    const expected = [_]usize{ 5, 2, 6, 9 };
    try std.testing.expectEqualSlices(usize, &expected, indices.slice());

    var sorted = command.sortedExplicitIndexSet();
    const expected_sorted = [_]usize{ 2, 5, 6, 9 };
    try std.testing.expectEqualSlices(usize, &expected_sorted, sorted.slice());

    var iter = command.coveredIndexIterator();
    var got: [4]usize = undefined;
    var count: usize = 0;
    while (iter.next()) |idx| {
        got[count] = idx;
        count += 1;
    }
    try std.testing.expectEqualSlices(usize, &expected_sorted, got[0..count]);
}

test "program command shape drives coverage and advancement" {
    try std.testing.expectEqual(ProgramCommandCoverage.anchor_sidecars, ProgramCommandKind.projection_chain.shape().coverage);
    try std.testing.expectEqual(ProgramCommandAdvance.contiguous, ProgramCommandKind.projection_chain.shape().advance);
    try std.testing.expectEqual(ProgramCommandSidecarLayout.flat, ProgramCommandKind.projection_cache_group.shape().sidecars);

    const chain = ProgramCommand{
        .kind = .projection_chain,
        .op_start = 3,
        .op_count = 2,
        .anchor_count = 1,
        .sidecar_count = 1,
    };
    try std.testing.expect(chain.hasExplicitCoverage());
    try std.testing.expect(!chain.usesExplicitIndices());
    try std.testing.expectEqual(@as(u32, 2), chain.coveredOpCount());
    try std.testing.expectEqual(@as(u32, 2), chain.advanceCount());

    const group = ProgramCommand{
        .kind = .movement_group,
        .op_start = 3,
        .op_count = 8,
        .anchor_count = 4,
    };
    try std.testing.expect(group.hasExplicitCoverage());
    try std.testing.expect(group.usesExplicitIndices());
    try std.testing.expectEqual(@as(u32, 4), group.coveredOpCount());
    try std.testing.expectEqual(@as(u32, 1), group.advanceCount());

    const contiguous = ProgramCommand.contiguous(.row_chain, 4, 3);
    var iter = contiguous.coveredIndexIterator();
    const expected = [_]usize{ 4, 5, 6 };
    for (expected) |idx| {
        try std.testing.expectEqual(idx, iter.next().?);
    }
    try std.testing.expectEqual(@as(?usize, null), iter.next());
    try std.testing.expectEqual(@as(usize, 0), iter.remainingCount());
}

test "kernel schedule groups contiguous fallback ops by family" {
    const ops = [_]backend_mod.DeviceOp{
        testElementwise(.add),
        testElementwise(.relu),
        .{ .softmax = .{ .dst = 0, .src = 0, .rows = 1, .cols = 4 } },
        .{ .rmsnorm = .{ .dst = 0, .src = 0, .rows = 1, .cols = 4 } },
        .{ .reduce = .{ .op = .sum, .dst = 0, .src = 0, .n_out = 1, .reduce_size = 4 } },
    };
    const policy = SchedulePolicy{ .capabilities = backend_mod.Capabilities.reference_cpu };

    const items = try buildKernelSchedule(std.testing.allocator, &ops, policy);
    defer std.testing.allocator.free(items);

    try std.testing.expectEqual(@as(usize, 3), items.len);
    try expectKernelItem(items[0], .elementwise, .fallback, 0, 2);
    try expectKernelItem(items[1], .row, .fallback, 2, 2);
    try expectKernelItem(items[2], .reduce, .fallback, 4, 1);
}

test "kernel schedule uses coarse backend thresholds for matmul families" {
    const ops = [_]backend_mod.DeviceOp{
        testMatmul(1),
        testMatmul(16),
        testMatmul(32),
        testQMatmul(1),
        testQMatmul(16),
    };
    const policy = SchedulePolicy{
        .capabilities = backend_mod.Capabilities.metal,
        .native_kernels = .{ .matmul = true, .qmatvec = true, .qmatmul = true },
    };

    const items = try buildKernelSchedule(std.testing.allocator, &ops, policy);
    defer std.testing.allocator.free(items);

    try std.testing.expectEqual(@as(usize, 4), items.len);
    try expectKernelItem(items[0], .matmul, .fallback, 0, 1);
    try expectKernelItem(items[1], .matmul, .backend, 1, 2);
    try expectKernelItem(items[2], .qmatvec, .fallback, 3, 1);
    try expectKernelItem(items[3], .qmatmul, .backend, 4, 1);
}

test "kernel schedule treats single-row quantized matmul as qmatvec" {
    const op = testQMatmul(1);
    var policy = SchedulePolicy{
        .capabilities = backend_mod.Capabilities.metal,
        .native_kernels = .{ .qmatvec = true },
    };

    try std.testing.expectEqual(KernelFamily.qmatvec, kernelFamily(op));
    try std.testing.expectEqual(ExecutionClass.fallback, executionClass(op, policy));
    policy.fine_grained = true;
    try std.testing.expectEqual(ExecutionClass.backend, executionClass(op, policy));
}

test "kernel schedule respects fused elementwise capability limits" {
    const small_steps = [_]backend_mod.FusedEwStep{.{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 }};
    const large_steps = [_]backend_mod.FusedEwStep{.{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 }} ** 9;
    const ops = [_]backend_mod.DeviceOp{
        .{ .fused_elementwise = .{ .steps = &small_steps, .n = 1, .dst = 0, .src = 0, .dst_offset = 0, .src_offset = 0 } },
        .{ .fused_elementwise = .{ .steps = &large_steps, .n = 1, .dst = 0, .src = 0, .dst_offset = 0, .src_offset = 0 } },
    };
    const policy = SchedulePolicy{
        .capabilities = backend_mod.Capabilities.metal,
        .native_kernels = .{ .fused_elementwise = true },
        .fine_grained = true,
    };

    const items = try buildKernelSchedule(std.testing.allocator, &ops, policy);
    defer std.testing.allocator.free(items);

    try std.testing.expectEqual(@as(usize, 2), items.len);
    try expectKernelItem(items[0], .fused_elementwise, .backend, 0, 1);
    try expectKernelItem(items[1], .fused_elementwise, .fallback, 1, 1);
}

test "kernel schedule fine grained policy unlocks small native kernels" {
    const op = testElementwise(.add);
    var policy = SchedulePolicy{
        .capabilities = backend_mod.Capabilities.metal,
        .native_kernels = .{ .elementwise = true },
    };

    try std.testing.expectEqual(ExecutionClass.fallback, executionClass(op, policy));
    policy.fine_grained = true;
    try std.testing.expectEqual(ExecutionClass.backend, executionClass(op, policy));
}

test "schedule shape match ignores dynamic offsets but catches family changes" {
    var ops = [_]backend_mod.DeviceOp{
        testSliceAssign(0),
        testQMatmul(1),
        testQMatmul(16),
    };
    const policy = SchedulePolicy{
        .capabilities = backend_mod.Capabilities.reference_cpu,
        .native_kernels = .{ .movement = true, .qmatvec = true, .qmatmul = true },
        .fine_grained = true,
        .min_backend_qmatmul_m = 0,
    };

    const items = try buildKernelSchedule(std.testing.allocator, &ops, policy);
    defer std.testing.allocator.free(items);

    ops[0].slice_assign.dst_offset = 128;
    try std.testing.expect(scheduleShapeMatches(&ops, items, policy));

    ops[1].qmatmul.M = 16;
    try std.testing.expect(!scheduleShapeMatches(&ops, items, policy));
}

test "kernel regions group qmatvec anchored member runs" {
    const items = [_]KernelItem{
        .{ .family = .movement, .execution = .fallback, .start = 0, .len = 2 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 2, .len = 1 },
        .{ .family = .elementwise, .execution = .fallback, .start = 3, .len = 2 },
        .{ .family = .matmul, .execution = .backend, .start = 5, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 6, .len = 1 },
    };

    const regions = try buildKernelRegions(std.testing.allocator, &items, RegionPolicy.qmatvecCluster());
    defer std.testing.allocator.free(regions);

    try std.testing.expectEqual(@as(usize, 2), regions.len);
    try std.testing.expectEqual(@as(u32, 0), regions[0].start_item);
    try std.testing.expectEqual(@as(u32, 3), regions[0].item_count);
    try std.testing.expectEqual(@as(u32, 0), regions[0].op_start);
    try std.testing.expectEqual(@as(u32, 5), regions[0].op_count);
    try std.testing.expectEqual(@as(u32, 1), regions[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 4), regions[1].start_item);
    try std.testing.expectEqual(@as(u32, 1), regions[1].item_count);
    try std.testing.expectEqual(@as(u32, 6), regions[1].op_start);
    try std.testing.expectEqual(@as(u32, 1), regions[1].op_count);
    try std.testing.expectEqual(@as(u32, 1), regions[1].anchor_count);
}

test "kernel regions skip member runs without anchors" {
    const items = [_]KernelItem{
        .{ .family = .movement, .execution = .fallback, .start = 0, .len = 2 },
        .{ .family = .elementwise, .execution = .fallback, .start = 2, .len = 1 },
    };

    const regions = try buildKernelRegions(std.testing.allocator, &items, RegionPolicy.qmatvecCluster());
    defer std.testing.allocator.free(regions);

    try std.testing.expectEqual(@as(usize, 0), regions.len);
}

test "anchor run regions expose contiguous anchor groups" {
    const items = [_]KernelItem{
        .{ .family = .movement, .execution = .fallback, .start = 0, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 1, .len = 3 },
        .{ .family = .elementwise, .execution = .fallback, .start = 4, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 5, .len = 2 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 7, .len = 1 },
    };

    const regions = try buildAnchorRunRegions(std.testing.allocator, &items, RegionPolicy.qmatvecCluster());
    defer std.testing.allocator.free(regions);

    try std.testing.expectEqual(@as(usize, 2), regions.len);
    try std.testing.expectEqual(@as(u32, 1), regions[0].start_item);
    try std.testing.expectEqual(@as(u32, 1), regions[0].item_count);
    try std.testing.expectEqual(@as(u32, 1), regions[0].op_start);
    try std.testing.expectEqual(@as(u32, 3), regions[0].op_count);
    try std.testing.expectEqual(@as(u32, 3), regions[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 3), regions[1].start_item);
    try std.testing.expectEqual(@as(u32, 2), regions[1].item_count);
    try std.testing.expectEqual(@as(u32, 5), regions[1].op_start);
    try std.testing.expectEqual(@as(u32, 3), regions[1].op_count);
    try std.testing.expectEqual(@as(u32, 3), regions[1].anchor_count);
}

test "anchor window regions split member runs by anchor count" {
    const items = [_]KernelItem{
        .{ .family = .movement, .execution = .fallback, .start = 0, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 1, .len = 1 },
        .{ .family = .elementwise, .execution = .fallback, .start = 2, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 3, .len = 2 },
        .{ .family = .row, .execution = .fallback, .start = 5, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 6, .len = 1 },
        .{ .family = .attention, .execution = .fallback, .start = 7, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 8, .len = 1 },
        .{ .family = .matmul, .execution = .backend, .start = 9, .len = 1 },
    };

    const regions = try buildAnchorWindowRegions(std.testing.allocator, &items, RegionPolicy.qmatvecCluster(), 3);
    defer std.testing.allocator.free(regions);

    try std.testing.expectEqual(@as(usize, 1), regions.len);
    try std.testing.expectEqual(@as(u32, 0), regions[0].start_item);
    try std.testing.expectEqual(@as(u32, 5), regions[0].item_count);
    try std.testing.expectEqual(@as(u32, 0), regions[0].op_start);
    try std.testing.expectEqual(@as(u32, 6), regions[0].op_count);
    try std.testing.expectEqual(@as(u32, 3), regions[0].anchor_count);
}

test "qmatmul cluster windows keep prefill attention inside layer regions" {
    const items = [_]KernelItem{
        .{ .family = .row, .execution = .fallback, .start = 0, .len = 1 },
        .{ .family = .qmatmul, .execution = .backend, .start = 1, .len = 3 },
        .{ .family = .rope, .execution = .fallback, .start = 4, .len = 2 },
        .{ .family = .attention, .execution = .fallback, .start = 6, .len = 1 },
        .{ .family = .qmatmul, .execution = .backend, .start = 7, .len = 1 },
        .{ .family = .fused_elementwise, .execution = .fallback, .start = 8, .len = 1 },
        .{ .family = .qmatmul, .execution = .backend, .start = 9, .len = 3 },
        .{ .family = .matmul, .execution = .backend, .start = 12, .len = 1 },
    };

    const regions = try buildAnchorWindowRegions(std.testing.allocator, &items, RegionPolicy.qmatmulCluster(), 7);
    defer std.testing.allocator.free(regions);

    try std.testing.expectEqual(@as(usize, 1), regions.len);
    try std.testing.expectEqual(@as(u32, 0), regions[0].start_item);
    try std.testing.expectEqual(@as(u32, 8), regions[0].item_count);
    try std.testing.expectEqual(@as(u32, 0), regions[0].op_start);
    try std.testing.expectEqual(@as(u32, 13), regions[0].op_count);
    try std.testing.expectEqual(@as(u32, 7), regions[0].anchor_count);
}

test "stage plan builds named anchored layer windows" {
    const items = [_]KernelItem{
        .{ .family = .row, .execution = .fallback, .start = 0, .len = 1 },
        .{ .family = .qmatmul, .execution = .backend, .start = 1, .len = 3 },
        .{ .family = .rope, .execution = .backend, .start = 4, .len = 2 },
        .{ .family = .movement, .execution = .backend, .start = 6, .len = 2 },
        .{ .family = .attention, .execution = .backend, .start = 8, .len = 1 },
        .{ .family = .qmatmul, .execution = .backend, .start = 9, .len = 1 },
        .{ .family = .fused_elementwise, .execution = .backend, .start = 10, .len = 1 },
        .{ .family = .qmatmul, .execution = .backend, .start = 11, .len = 3 },
        .{ .family = .matmul, .execution = .backend, .start = 14, .len = 1 },
        .{ .family = .qmatvec, .execution = .backend, .start = 15, .len = 1 },
    };
    const stages = [_]StagePolicy{
        StagePolicy.anchored("prefill-layer", 7, RegionPolicy.qmatmulCluster(), 7),
    };

    const plan = try buildStagePlan(std.testing.allocator, &items, &stages);
    defer std.testing.allocator.free(plan);

    try std.testing.expectEqual(@as(usize, 1), plan.len);
    try std.testing.expectEqual(@as(u32, 7), plan[0].pattern_index);
    try std.testing.expectEqual(@as(u32, 0), plan[0].region.start_item);
    try std.testing.expectEqual(@as(u32, 9), plan[0].region.item_count);
    try std.testing.expectEqual(@as(u32, 15), plan[0].region.op_count);
    try std.testing.expectEqual(@as(u32, 7), plan[0].region.anchor_count);

    const units = try buildStageRegionSchedule(std.testing.allocator, &items, &stages);
    defer std.testing.allocator.free(units);

    try std.testing.expectEqual(@as(usize, 2), units.len);
    try std.testing.expectEqual(ScheduleUnitKind.pattern_region, units[0].kind);
    try std.testing.expectEqual(@as(u32, 7), units[0].pattern_index);
    try std.testing.expectEqual(ScheduleUnitKind.item, units[1].kind);
    try std.testing.expectEqual(@as(u32, 15), units[1].op_start);
}

test "stage commands collapse rmsnorm scale row chains" {
    const row_chain = testRmsnormScaleOps();
    const ops = [_]backend_mod.DeviceOp{
        row_chain[0],
        row_chain[1],
        row_chain[2],
        testQMatmul(16),
    };

    try std.testing.expect(isRmsnormScaleChain(ops[0], ops[1], ops[2]));

    const commands = try buildStageCommands(std.testing.allocator, &ops);
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(StageCommandKind.row_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 0), commands[0].op_start);
    try std.testing.expectEqual(@as(u32, 3), commands[0].op_count);
    try std.testing.expectEqual(StageCommandKind.op, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 3), commands[1].op_start);

    const summary = summarizeStageCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 4), summary.ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.row_chains);
    try std.testing.expectEqual(@as(u32, 3), summary.row_chain_ops);
}

test "stage commands leave nonmatching row triples as ops" {
    var row_chain = testRmsnormScaleOps();
    row_chain[2].elementwise.op = .add;

    const commands = try buildStageCommands(std.testing.allocator, &row_chain);
    defer std.testing.allocator.free(commands);

    try std.testing.expect(!isRmsnormScaleChain(row_chain[0], row_chain[1], row_chain[2]));
    try std.testing.expectEqual(@as(usize, 3), commands.len);
    for (commands, 0..) |command, i| {
        try std.testing.expectEqual(StageCommandKind.op, command.kind);
        try std.testing.expectEqual(@as(u32, @intCast(i)), command.op_start);
        try std.testing.expectEqual(@as(u32, 1), command.op_count);
    }
}

test "stage commands collapse rope cache-write chains" {
    const rope_chain = testRopeSliceAssignOps();
    const ops = [_]backend_mod.DeviceOp{
        rope_chain[0],
        rope_chain[1],
        testQMatmul(16),
    };

    try std.testing.expect(isRopeSliceAssignChain(ops[0], ops[1]));
    try std.testing.expectEqual(StageCommandKind.rope_chain, findStageCommand(&ops, 0).?.kind);

    const commands = try buildStageCommands(std.testing.allocator, &ops);
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(StageCommandKind.rope_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 0), commands[0].op_start);
    try std.testing.expectEqual(@as(u32, 2), commands[0].op_count);
    try std.testing.expectEqual(StageCommandKind.op, commands[1].kind);

    const summary = summarizeStageCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 3), summary.ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_chains);
    try std.testing.expectEqual(@as(u32, 2), summary.rope_chain_ops);
}

test "stage commands leave nonmatching rope pairs as ops" {
    var rope_chain = testRopeSliceAssignOps();
    rope_chain[1].slice_assign.src_col_stride = 8;

    const commands = try buildStageCommands(std.testing.allocator, &rope_chain);
    defer std.testing.allocator.free(commands);

    try std.testing.expect(!isRopeSliceAssignChain(rope_chain[0], rope_chain[1]));
    try std.testing.expectEqual(@as(usize, 2), commands.len);
    for (commands, 0..) |command, i| {
        try std.testing.expectEqual(StageCommandKind.op, command.kind);
        try std.testing.expectEqual(@as(u32, @intCast(i)), command.op_start);
        try std.testing.expectEqual(@as(u32, 1), command.op_count);
    }
}

test "projection groups batch independent prefill qmatmuls" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 16),
        testQMatmulWith(2, 0, 16),
        testQMatmulWith(3, 0, 16),
        testQMatmulWith(4, 0, 16),
        testQMatmulWith(5, 0, 16),
    };

    const groups = try buildProjectionGroups(std.testing.allocator, &ops, ProjectionGroupPolicy.prefillQMatmul(4));
    defer std.testing.allocator.free(groups);

    try std.testing.expectEqual(@as(usize, 1), groups.len);
    try std.testing.expectEqual(ProjectionGroupKind.qmatmul, groups[0].kind);
    try std.testing.expectEqual(@as(u32, 0), groups[0].start_op);
    try std.testing.expectEqual(@as(u32, 4), groups[0].op_count);
    try std.testing.expectEqual(@as(u32, 4), groups[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 0), groups[0].sidecar_count);

    const summary = summarizeProjectionGroups(groups);
    try std.testing.expectEqual(@as(u32, 1), summary.groups);
    try std.testing.expectEqual(@as(u32, 4), summary.anchors);
    try std.testing.expectEqual(@as(u32, 4), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
}

test "projection groups carry compatible qmatmul cache-store sidecars" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        testQMatmulSidecar(1, 8),
        .{ .elementwise = .{ .op = .add, .dst = 9, .src0 = 9, .src1 = 9, .n = 1 } },
        testQMatmulWith(2, 0, 2),
    };

    const groups = try buildProjectionGroups(std.testing.allocator, &ops, ProjectionGroupPolicy.prefillQMatmul(4));
    defer std.testing.allocator.free(groups);

    const selection = findProjectionGroup(&ops, 0, ProjectionGroupPolicy.prefillQMatmul(4), null).?;
    try std.testing.expectEqual(@as(usize, 0), selection.indices[0]);
    try std.testing.expectEqual(@as(usize, 3), selection.indices[1]);
    try std.testing.expectEqual(@as(?usize, 1), selection.sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, null), selection.sidecar_indices[1]);

    try std.testing.expect(qmatmulSliceSidecarCompatible(ops[0].qmatmul, ops[1].slice_assign));
    try std.testing.expectEqual(@as(usize, 1), groups.len);
    try std.testing.expectEqual(@as(u32, 0), groups[0].start_op);
    try std.testing.expectEqual(@as(u32, 4), groups[0].op_count);
    try std.testing.expectEqual(@as(u32, 2), groups[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), groups[0].sidecar_count);

    const summary = summarizeProjectionGroups(groups);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_saved_dispatches);
}

test "projection groups carry compatible qmatvec cache-store sidecars" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 1),
        .{ .slice_assign = .{
            .dst = 8,
            .src = 1,
            .rows = 4,
            .cols = 1,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
        testQMatmulWith(2, 0, 1),
    };

    const selection = findProjectionGroup(&ops, 0, ProjectionGroupPolicy.decodeQMatvec(4), null).?;
    try std.testing.expectEqual(@as(usize, 0), selection.indices[0]);
    try std.testing.expectEqual(@as(usize, 2), selection.indices[1]);
    try std.testing.expectEqual(@as(?usize, 1), selection.sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, null), selection.sidecar_indices[1]);
    try std.testing.expect(qmatvecSliceSidecarCompatible(ops[0].qmatmul, ops[1].slice_assign));

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_group, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatvec, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_saved_dispatches);
}

test "projection groups carry compatible qmatvec elementwise sidecars" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 1),
        .{ .elementwise = .{
            .op = .add,
            .dst = 8,
            .src0 = 1,
            .src1 = 9,
            .n = 4,
            .dst_offset = 0,
            .src0_offset = 0,
            .src1_offset = 0,
        } },
        testQMatmulWith(2, 0, 1),
    };

    const selection = findProjectionGroup(&ops, 0, ProjectionGroupPolicy.decodeQMatvec(4), null).?;
    try std.testing.expectEqual(@as(usize, 0), selection.indices[0]);
    try std.testing.expectEqual(@as(usize, 2), selection.indices[1]);
    try std.testing.expectEqual(@as(?usize, 1), selection.sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, null), selection.sidecar_indices[1]);
    try std.testing.expect(qmatvecElementwiseSidecarCompatible(ops[0].qmatmul, ops[1].elementwise));

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_group, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatvec, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_groups);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_sidecars);
}

test "projection groups carry compatible qmatmul elementwise sidecars" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 10, .src0 = 1, .src1 = 20, .n = 8 } },
        testQMatmulWith(2, 0, 2),
        .{ .elementwise = .{ .op = .mul, .dst = 11, .src0 = 21, .src1 = 2, .n = 8 } },
    };

    const selection = findProjectionGroup(&ops, 0, ProjectionGroupPolicy.prefillQMatmul(4), null).?;
    try std.testing.expectEqual(@as(usize, 0), selection.indices[0]);
    try std.testing.expectEqual(@as(usize, 2), selection.indices[1]);
    try std.testing.expectEqual(@as(?usize, 1), selection.sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 3), selection.sidecar_indices[1]);

    const groups = try buildProjectionGroups(std.testing.allocator, &ops, ProjectionGroupPolicy.prefillQMatmul(4));
    defer std.testing.allocator.free(groups);
    try std.testing.expectEqual(@as(usize, 1), groups.len);
    try std.testing.expectEqual(@as(u32, 2), groups[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 2), groups[0].sidecar_count);

    const summary = summarizeProjectionGroups(groups);
    try std.testing.expectEqual(@as(u32, 4), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
}

test "projection groups reject conflicting or nonhoistable projections" {
    const conflict_ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 16),
        testQMatmulWith(1, 0, 16),
    };
    const conflict_groups = try buildProjectionGroups(std.testing.allocator, &conflict_ops, ProjectionGroupPolicy.prefillQMatmul(4));
    defer std.testing.allocator.free(conflict_groups);
    try std.testing.expectEqual(@as(usize, 0), conflict_groups.len);

    const blocked_ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 16),
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 0, .src1 = 0, .n = 1 } },
        testQMatmulWith(3, 2, 16),
    };
    const blocked_groups = try buildProjectionGroups(std.testing.allocator, &blocked_ops, ProjectionGroupPolicy.prefillQMatmul(4));
    defer std.testing.allocator.free(blocked_groups);
    try std.testing.expectEqual(@as(usize, 0), blocked_groups.len);
}

test "projection groups use access spans instead of whole-buffer conflicts" {
    var first = testQMatmulWith(1, 0, 2);
    first.qmatmul.dst_offset = 0;
    var second = testQMatmulWith(1, 0, 2);
    second.qmatmul.dst_offset = 8;
    const disjoint_ops = [_]backend_mod.DeviceOp{ first, second };

    const groups = try buildProjectionGroups(std.testing.allocator, &disjoint_ops, ProjectionGroupPolicy.prefillQMatmul(4));
    defer std.testing.allocator.free(groups);
    try std.testing.expectEqual(@as(usize, 1), groups.len);
    try std.testing.expectEqual(@as(u32, 2), groups[0].anchor_count);

    second.qmatmul.dst_offset = 4;
    const overlapping_ops = [_]backend_mod.DeviceOp{ first, second };
    const overlapping_groups = try buildProjectionGroups(std.testing.allocator, &overlapping_ops, ProjectionGroupPolicy.prefillQMatmul(4));
    defer std.testing.allocator.free(overlapping_groups);
    try std.testing.expectEqual(@as(usize, 0), overlapping_groups.len);
}

test "program command stream merges stage and projection commands" {
    const rope_chain = testRopeSliceAssignOps();
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        testQMatmulSidecar(1, 8),
        rope_chain[0],
        rope_chain[1],
        testQMatmulWith(4, 0, 2),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_group, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatmul, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(usize, 4), commands[0].indices[1]);
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(ProgramCommandKind.rope_chain, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[1].op_start);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 5), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_groups);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_chains);
}

test "program command stream derives defaults from backend capabilities" {
    const policy = CommandStreamPolicy.fromCapabilities(backend_mod.Capabilities.metal);
    try std.testing.expect(policy.stage_commands);
    try std.testing.expectEqual(@as(u32, 4), policy.qmatvec_group_size);
    try std.testing.expectEqual(@as(u32, 4), policy.qmatmul_group_size);
    try std.testing.expect(policy.qmatmul_sidecars);
    try std.testing.expectEqual(@as(u32, 8), policy.qmatmul_cache_sidecars_per_anchor);
    try std.testing.expect(!policy.projection_rope_cache_sidecars);
    try std.testing.expectEqual(@as(u32, 16), policy.max_rope_batch);
    try std.testing.expectEqual(@as(u32, 16), policy.max_movement_batch);
    try std.testing.expectEqual(@as(u32, 16), policy.max_attention_batch);
    try std.testing.expectEqual(@as(u32, 4), policy.max_attention_store_batch);
    try std.testing.expectEqual(@as(u32, 16), policy.max_rope_attention_store_batch);
    try std.testing.expectEqual(@as(u32, 8), policy.max_elementwise_batch);
    try std.testing.expect(policy.fuse_repeat_fused_elementwise);
}

test "program command stream carries multiple projection cache stores" {
    var store0 = testQMatmulSidecar(3, 8);
    store0.slice_assign.dst_offset = 0;
    var store1 = testQMatmulSidecar(3, 8);
    store1.slice_assign.dst_offset = 8;
    var store2 = testQMatmulSidecar(3, 8);
    store2.slice_assign.dst_offset = 16;

    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        testQMatmulWith(2, 0, 2),
        testQMatmulWith(3, 0, 2),
        store0,
        .{ .elementwise = .{ .op = .add, .dst = 20, .src0 = 20, .src1 = 20, .n = 1 } },
        store1,
        store2,
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_cache_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 3), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 3), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(?usize, 3), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 5), commands[0].sidecar_indices[1]);
    try std.testing.expectEqual(@as(?usize, 6), commands[0].sidecar_indices[2]);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_cache_groups);
    try std.testing.expectEqual(@as(u32, 3), summary.projection_cache_anchors);
    try std.testing.expectEqual(@as(u32, 3), summary.projection_cache_sidecars);
    try std.testing.expectEqual(@as(u32, 7), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 5), summary.estimated_saved_dispatches);
}

test "program command stream carries projection rope cache stores" {
    const rope_pair = testRopeSliceAssignOps();
    var rope = rope_pair[0];
    rope.rope.src = 1;
    rope.rope.src_off = 0;

    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 3),
        rope,
        rope_pair[1],
        testQMatmulWith(4, 0, 3),
    };

    var policy = CommandStreamPolicy.metal(4, 4);
    policy.projection_rope_cache_sidecars = true;
    const commands = try buildProgramCommands(std.testing.allocator, &ops, policy);
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_cache_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 2), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 2), commands[0].sidecar_indices[1]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_cache_groups);
    try std.testing.expectEqual(@as(u32, 2), summary.projection_cache_anchors);
    try std.testing.expectEqual(@as(u32, 2), summary.projection_cache_sidecars);
    try std.testing.expectEqual(@as(u32, 4), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
}

test "program command stream carries qmatvec rope cache stores" {
    const rope_pair = testRopeSliceAssignOps();
    var rope = rope_pair[0];
    rope.rope.src = 1;
    rope.rope.src_off = 0;
    rope.rope.seq_len = 1;
    var store = rope_pair[1];
    store.slice_assign.cols = 1;

    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 1),
        rope,
        store,
        testQMatmulWith(4, 0, 1),
    };

    var policy = CommandStreamPolicy.metal(4, 4);
    policy.projection_rope_cache_sidecars = true;
    const commands = try buildProgramCommands(std.testing.allocator, &ops, policy);
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_cache_group, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatvec, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 2), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 2), commands[0].sidecar_indices[1]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_cache_groups);
    try std.testing.expectEqual(@as(u32, 2), summary.projection_cache_anchors);
    try std.testing.expectEqual(@as(u32, 2), summary.projection_cache_sidecars);
    try std.testing.expectEqual(@as(u32, 4), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
}

test "program command stream carries qmatvec rope materialization" {
    const rope_pair = testRopeSliceAssignOps();
    var rope = rope_pair[0];
    rope.rope.src = 1;
    rope.rope.src_off = 0;
    rope.rope.seq_len = 1;

    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 1),
        rope,
        testQMatmulWith(4, 0, 1),
    };

    var policy = CommandStreamPolicy.metal(4, 4);
    policy.projection_rope_cache_sidecars = true;
    const commands = try buildProgramCommands(std.testing.allocator, &ops, policy);
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_cache_group, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatvec, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecar_indices[0]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_cache_groups);
    try std.testing.expectEqual(@as(u32, 2), summary.projection_cache_anchors);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_cache_sidecars);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_saved_dispatches);
}

test "program command stream preserves q rope attention fusion" {
    var att = testAttention(0);
    att.attention.q = 4;
    att.attention.k = 10;
    att.attention.v = 11;
    att.attention.dst = 5;
    att.attention.seq_q = 1;

    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 1),
        .{ .rope = .{
            .dst = 4,
            .src = 1,
            .cos_sin = 7,
            .half_d = 2,
            .seq_len = 1,
            .src_off = 0,
            .cs_off = 0,
            .dst_off = 0,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 4,
        } },
        testQMatmulWith(8, 0, 1),
        att,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 5,
            .rows = 4,
            .cols = 1,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
    };

    var policy = CommandStreamPolicy.metal(4, 4);
    policy.projection_rope_cache_sidecars = true;
    const commands = try buildProgramCommands(std.testing.allocator, &ops, policy);
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_group, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatvec, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 0), commands[0].sidecar_count);
    try std.testing.expectEqual(ProgramCommandKind.rope_attention_store_chain, commands[1].kind);
    try std.testing.expectEqual(@as(usize, 1), commands[1].indices[0]);
    try std.testing.expectEqual(@as(usize, 3), commands[1].indices[1]);
    try std.testing.expectEqual(@as(?usize, 4), commands[1].sidecar_indices[0]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_groups);
    try std.testing.expectEqual(@as(u32, 0), summary.projection_cache_groups);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_attention_store_chains);
    try std.testing.expectEqual(@as(u32, 5), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);

    const rope_summary = summarizeProjectionRopeCacheSidecars(&ops, 2);
    try std.testing.expectEqual(@as(u32, 2), rope_summary.anchors);
    try std.testing.expectEqual(@as(u32, 1), rope_summary.rope_materializations);
    try std.testing.expectEqual(@as(u32, 1), rope_summary.materialization_attention_fusion_skips);
}

test "projection rope cache summary counts tile-pair opportunities" {
    const rope_pair = testRopeSliceAssignOps();
    var rope = rope_pair[0];
    rope.rope.src = 1;
    rope.rope.src_off = 0;

    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 3),
        rope,
        rope_pair[1],
        testQMatmulWith(4, 0, 3),
    };

    const summary = summarizeProjectionRopeCacheSidecars(&ops, 2);
    try std.testing.expectEqual(@as(u32, 2), summary.anchors);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_store_pairs);
    try std.testing.expectEqual(@as(u32, 1), summary.compatible_pairs);
    try std.testing.expectEqual(@as(u32, 1), summary.tile_pair_pairs);
    try std.testing.expectEqual(@as(u32, 0), summary.rope_materializations);
    try std.testing.expectEqual(@as(u32, 0), summary.materialization_attention_fusion_skips);
}

test "program command stream groups rope slice stores" {
    const first = testRopeSliceAssignOps();
    var second = testRopeSliceAssignOps();
    second[0].rope.src = 11;
    second[0].rope.src_off = 8;
    second[1].slice_assign.dst_offset = 16;

    const ops = [_]backend_mod.DeviceOp{
        first[0],
        first[1],
        testQMatmulWith(8, 9, 2),
        second[0],
        second[1],
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.rope_store_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 2), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(usize, 3), commands[0].indices[1]);
    try std.testing.expectEqual(@as(?usize, 4), commands[0].sidecar_indices[1]);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[1].op_start);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 5), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.op_commands);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_store_groups);
    try std.testing.expectEqual(@as(u32, 2), summary.rope_store_group_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.rope_store_group_sidecars);
}

test "program command stream keeps used noncontiguous ops single-owned" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 9, .src0 = 9, .src1 = 9, .n = 1 } },
        testQMatmulWith(4, 0, 2),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_group, commands[0].kind);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[1].op_start);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.op_commands);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_groups);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
}

test "program command stream emits projection sidecar chains" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 8 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_chain, commands[0].kind);
    try std.testing.expectEqual(ProjectionGroupKind.qmatmul, commands[0].projection_kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecar_indices[0]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.commands);
    try std.testing.expectEqual(@as(u32, 2), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_chains);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_chain_sidecars);
}

test "projection sidecar chains reject incompatible consumers" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 7 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[0].kind);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
}

test "projection primary output liveness ignores internal sidecar reads" {
    const scratch_only = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 8 } },
    };
    try std.testing.expect(!projectionPrimaryOutputHasExternalUsers(&scratch_only, 0, 1));

    const external_read = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 8 } },
        .{ .elementwise = .{ .op = .mul, .dst = 4, .src0 = 1, .src1 = 5, .n = 8 } },
    };
    try std.testing.expect(projectionPrimaryOutputHasExternalUsers(&external_read, 0, 1));

    const overwritten = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 3, .n = 8 } },
        testQMatmulWith(1, 4, 2),
        .{ .elementwise = .{ .op = .mul, .dst = 5, .src0 = 1, .src1 = 6, .n = 8 } },
    };
    try std.testing.expect(!projectionPrimaryOutputHasExternalUsers(&overwritten, 0, 1));

    const inplace_sidecar = [_]backend_mod.DeviceOp{
        testQMatmulWith(1, 0, 2),
        .{ .elementwise = .{ .op = .add, .dst = 1, .src0 = 1, .src1 = 3, .n = 8 } },
        .{ .elementwise = .{ .op = .mul, .dst = 4, .src0 = 1, .src1 = 5, .n = 8 } },
    };
    try std.testing.expect(!projectionPrimaryOutputHasExternalUsers(&inplace_sidecar, 0, 1));
}

test "program command stream emits contiguous batch commands" {
    var rope_pair = testRopeSliceAssignOps();
    rope_pair[1] = rope_pair[0];
    rope_pair[1].rope.src_off = 4;
    rope_pair[1].rope.dst_off = 12;

    const ops = [_]backend_mod.DeviceOp{
        rope_pair[0],
        rope_pair[1],
        testQMatmulSidecar(1, 8),
        testQMatmulSidecar(1, 8),
        testAttention(0),
        testAttention(8),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 3), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.rope_batch, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].op_count);
    try std.testing.expectEqual(ProgramCommandKind.movement_batch, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[1].op_count);
    try std.testing.expectEqual(ProgramCommandKind.attention_batch, commands[2].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[2].op_count);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 3), summary.commands);
    try std.testing.expectEqual(@as(u32, 6), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_batches);
    try std.testing.expectEqual(@as(u32, 1), summary.movement_batches);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_batches);
}

test "program command stream emits noncontiguous attention groups" {
    const ops = [_]backend_mod.DeviceOp{
        testAttention(0),
        testQMatmulWith(9, 8, 2),
        testAttention(8),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.attention_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(usize, 2), commands[0].indices[1]);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[1].op_start);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_groups);
    try std.testing.expectEqual(@as(u32, 2), summary.attention_group_ops);
}

test "program command stream emits attention producer chains" {
    const ops = [_]backend_mod.DeviceOp{
        .{ .slice_assign = .{
            .dst = 2,
            .src = 9,
            .rows = 4,
            .cols = 4,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
        testAttention(0),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.attention_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 1), commands[0].indices[0]);
    try std.testing.expectEqual(@as(?usize, 0), commands[0].sidecar_indices[0]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.commands);
    try std.testing.expectEqual(@as(u32, 2), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_chains);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_chain_sidecars);
}

test "program command stream preserves attention output store over input sidecar" {
    const ops = [_]backend_mod.DeviceOp{
        .{ .slice_assign = .{
            .dst = 2,
            .src = 9,
            .rows = 4,
            .cols = 4,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
        testAttention(0),
        .{ .slice_assign = .{
            .dst = 9,
            .src = 4,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 8,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 0), commands[0].op_start);
    try std.testing.expectEqual(ProgramCommandKind.attention_store_chain, commands[1].kind);
    try std.testing.expectEqual(@as(usize, 1), commands[1].indices[0]);
    try std.testing.expectEqual(@as(?usize, 2), commands[1].sidecar_indices[0]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.op_commands);
    try std.testing.expectEqual(@as(u32, 0), summary.attention_chains);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_store_chains);
}

test "program command stream emits attention output store chains" {
    const ops = [_]backend_mod.DeviceOp{
        testAttention(0),
        .{ .slice_assign = .{
            .dst = 9,
            .src = 4,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 8,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.attention_store_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecar_indices[0]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.commands);
    try std.testing.expectEqual(@as(u32, 2), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_store_chains);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_store_chain_sidecars);
}

test "program command stream groups attention output stores" {
    const att0 = testAttention(0);
    var att1 = testAttention(8);
    att1.attention.q = 10;
    att1.attention.k = 11;
    att1.attention.v = 12;
    att1.attention.dst = 13;

    const ops = [_]backend_mod.DeviceOp{
        att0,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 4,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 8,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
        att1,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 13,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 4,
            .dst_row_stride = 1,
            .dst_col_stride = 8,
            .src_offset = 8,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.attention_store_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 2), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(?usize, 1), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(usize, 2), commands[0].indices[1]);
    try std.testing.expectEqual(@as(?usize, 3), commands[0].sidecar_indices[1]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.commands);
    try std.testing.expectEqual(@as(u32, 4), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_store_groups);
    try std.testing.expectEqual(@as(u32, 2), summary.attention_store_group_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.attention_store_group_sidecars);
}

test "program command stream emits rope attention output store chains" {
    var att = testAttention(0);
    att.attention.q = 4;
    att.attention.k = 10;
    att.attention.dst = 5;

    const ops = [_]backend_mod.DeviceOp{
        .{ .rope = .{
            .dst = 4,
            .src = 0,
            .cos_sin = 1,
            .half_d = 2,
            .seq_len = 2,
            .src_off = 0,
            .cs_off = 0,
            .dst_off = 0,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 4,
        } },
        att,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 5,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.rope_attention_store_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(usize, 1), commands[0].indices[1]);
    try std.testing.expectEqual(@as(?usize, 2), commands[0].sidecar_indices[0]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_attention_store_chains);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_attention_store_chain_sidecars);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
}

test "program command stream groups rope attention output stores" {
    var att0 = testAttention(0);
    var att1 = testAttention(8);
    att0.attention.q = 4;
    att0.attention.dst = 5;
    att1.attention.q = 6;
    att1.attention.dst = 7;

    const ops = [_]backend_mod.DeviceOp{
        .{ .rope = .{
            .dst = 4,
            .src = 0,
            .cos_sin = 1,
            .half_d = 2,
            .seq_len = 2,
            .src_off = 0,
            .cs_off = 0,
            .dst_off = 0,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 4,
        } },
        att0,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 5,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 8,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
        .{ .rope = .{
            .dst = 6,
            .src = 10,
            .cos_sin = 1,
            .half_d = 2,
            .seq_len = 2,
            .src_off = 8,
            .cs_off = 0,
            .dst_off = 8,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 4,
        } },
        att1,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 7,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 4,
            .dst_row_stride = 1,
            .dst_col_stride = 8,
            .src_offset = 8,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.rope_attention_store_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 4), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 2), commands[0].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(usize, 1), commands[0].indices[1]);
    try std.testing.expectEqual(@as(usize, 3), commands[0].indices[2]);
    try std.testing.expectEqual(@as(usize, 4), commands[0].indices[3]);
    try std.testing.expectEqual(@as(?usize, 2), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 5), commands[0].sidecar_indices[1]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_attention_store_groups);
    try std.testing.expectEqual(@as(u32, 4), summary.rope_attention_store_group_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.rope_attention_store_group_sidecars);
    try std.testing.expectEqual(@as(u32, 6), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
}

test "program command stream groups wide shared-offset rope attention stores" {
    var ops: [15]backend_mod.DeviceOp = undefined;
    for (0..5) |h| {
        const off: u32 = @intCast(h * 8);
        ops[h * 3] = .{ .rope = .{
            .dst = 4,
            .src = 0,
            .cos_sin = 1,
            .half_d = 2,
            .seq_len = 2,
            .src_off = off,
            .cs_off = off,
            .dst_off = off,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 4,
        } };
        ops[h * 3 + 1] = testAttention(off);
        ops[h * 3 + 1].attention.q = 4;
        ops[h * 3 + 1].attention.dst = 5;
        ops[h * 3 + 1].attention.k_off = off;
        ops[h * 3 + 1].attention.v_off = off;
        ops[h * 3 + 1].attention.mask_off = off;
        ops[h * 3 + 2] = .{ .slice_assign = .{
            .dst = 9,
            .src = 5,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = off,
            .dst_row_stride = 1,
            .dst_col_stride = 20,
            .src_offset = off,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } };
    }

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.rope_attention_store_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 10), commands[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 5), commands[0].sidecar_count);
    try std.testing.expect(ropeAttentionStoreCompactBatchCompatible(&ops, &commands[0]));

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_attention_store_groups);
    try std.testing.expectEqual(@as(u32, 10), summary.rope_attention_store_group_ops);
    try std.testing.expectEqual(@as(u32, 5), summary.rope_attention_store_group_sidecars);
    try std.testing.expectEqual(@as(u32, 15), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
}

test "program command stream delays rope attention groups until inputs are ready" {
    var att0 = testAttention(0);
    var att1 = testAttention(8);
    att0.attention.q = 4;
    att0.attention.dst = 5;
    att1.attention.q = 6;
    att1.attention.dst = 7;

    const ops = [_]backend_mod.DeviceOp{
        .{ .rope = .{
            .dst = 4,
            .src = 0,
            .cos_sin = 1,
            .half_d = 2,
            .seq_len = 2,
            .src_off = 0,
            .cs_off = 0,
            .dst_off = 0,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 4,
        } },
        testQMatmulWith(10, 8, 2),
        .{ .rope = .{
            .dst = 6,
            .src = 10,
            .cos_sin = 1,
            .half_d = 2,
            .seq_len = 2,
            .src_off = 0,
            .cs_off = 0,
            .dst_off = 8,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 4,
        } },
        att0,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 5,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 8,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
        att1,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 7,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 4,
            .dst_row_stride = 1,
            .dst_col_stride = 8,
            .src_offset = 8,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].op_start);
    try std.testing.expectEqual(ProgramCommandKind.rope_attention_store_group, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 5), commands[1].op_start);
    try std.testing.expectEqual(@as(u32, 4), commands[1].anchor_count);
    try std.testing.expectEqual(@as(u32, 2), commands[1].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[1].indices[0]);
    try std.testing.expectEqual(@as(usize, 3), commands[1].indices[1]);
    try std.testing.expectEqual(@as(usize, 2), commands[1].indices[2]);
    try std.testing.expectEqual(@as(usize, 5), commands[1].indices[3]);
    try std.testing.expectEqual(@as(?usize, 4), commands[1].sidecar_indices[0]);
    try std.testing.expectEqual(@as(?usize, 6), commands[1].sidecar_indices[1]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 7), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 5), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.op_commands);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_attention_store_groups);
    try std.testing.expectEqual(@as(u32, 4), summary.rope_attention_store_group_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.rope_attention_store_group_sidecars);
}

test "program command stream delays single rope attention store chains" {
    var att = testAttention(0);
    att.attention.q = 4;
    att.attention.k = 10;
    att.attention.dst = 5;

    const ops = [_]backend_mod.DeviceOp{
        .{ .rope = .{
            .dst = 4,
            .src = 0,
            .cos_sin = 1,
            .half_d = 2,
            .seq_len = 2,
            .src_off = 0,
            .cs_off = 0,
            .dst_off = 0,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 4,
        } },
        .{ .elementwise = .{
            .op = .add,
            .dst = 10,
            .src0 = 8,
            .src1 = 8,
            .n = 16,
        } },
        att,
        .{ .slice_assign = .{
            .dst = 9,
            .src = 5,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 8,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 0,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[0].op_start);
    try std.testing.expectEqual(ProgramCommandKind.rope_attention_store_chain, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[1].op_start);
    try std.testing.expectEqual(@as(u32, 2), commands[1].anchor_count);
    try std.testing.expectEqual(@as(u32, 1), commands[1].sidecar_count);
    try std.testing.expectEqual(@as(usize, 0), commands[1].indices[0]);
    try std.testing.expectEqual(@as(usize, 2), commands[1].indices[1]);
    try std.testing.expectEqual(@as(?usize, 3), commands[1].sidecar_indices[0]);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 4), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.op_commands);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_attention_store_chains);
    try std.testing.expectEqual(@as(u32, 1), summary.rope_attention_store_chain_sidecars);
}

test "program command stream carries delayed attention output stores" {
    const ops = [_]backend_mod.DeviceOp{
        testAttention(0),
        testQMatmulWith(8, 0, 2),
        .{ .slice_assign = .{
            .dst = 9,
            .src = 4,
            .rows = 4,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 8,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.attention_store_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 3), commands[0].op_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(?usize, 2), commands[0].sidecar_indices[0]);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[1].op_start);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.op_commands);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_store_chains);
    try std.testing.expectEqual(@as(u32, 1), summary.attention_store_chain_sidecars);
}

test "program command stream emits noncontiguous movement groups" {
    const ops = [_]backend_mod.DeviceOp{
        testSliceAssign(0),
        testQMatmulWith(9, 8, 2),
        testSliceAssign(4),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.movement_group, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(usize, 2), commands[0].indices[1]);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[1].op_start);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.movement_groups);
    try std.testing.expectEqual(@as(u32, 2), summary.movement_group_ops);
}

test "program command stream emits noncontiguous elementwise batch commands" {
    const ops = [_]backend_mod.DeviceOp{
        .{ .elementwise = .{ .op = .add, .dst = 1, .src0 = 0, .src1 = 0, .n = 4 } },
        testQMatmulWith(9, 8, 2),
        .{ .elementwise = .{ .op = .mul, .dst = 2, .src0 = 0, .src1 = 0, .n = 4 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.elementwise_batch, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].anchor_count);
    try std.testing.expectEqual(@as(usize, 0), commands[0].indices[0]);
    try std.testing.expectEqual(@as(usize, 2), commands[0].indices[1]);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
    try std.testing.expectEqual(@as(u32, 1), commands[1].op_start);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.op_commands);
    try std.testing.expectEqual(@as(u32, 1), summary.elementwise_batches);
    try std.testing.expectEqual(@as(u32, 2), summary.elementwise_ops);
}

test "program command stream fuses repeat feeding fused elementwise secondary" {
    const steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 2, .secondary_offset = 0 },
        .{ .op = .recip, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = true, .secondary_buf = 3, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        .{ .repeat = .{
            .dst = 2,
            .src = 1,
            .n = 8,
            .src_ne = .{ 1, 1, 1, 1 },
            .dst_ne = .{ 4, 2, 1, 1 },
            .src_strides = .{ 1, 1, 1, 1 },
            .dst_strides = .{ 1, 4, 8, 8 },
        } },
        .{ .fused_elementwise = .{
            .steps = &steps,
            .n = 8,
            .dst = 4,
            .src = 5,
            .dst_offset = 0,
            .src_offset = 0,
        } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.repeat_fused_elementwise_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 2), commands[0].op_count);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 1), summary.commands);
    try std.testing.expectEqual(@as(u32, 2), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.repeat_fused_elementwise_chains);
}

test "program command stream fuses projection activation expression chain" {
    const exp_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .neg, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .exp, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };
    const silu_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 4, .secondary_offset = 0 },
        .{ .op = .recip, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = true, .secondary_buf = 2, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(2, 8, 2),
        .{ .fused_elementwise = .{
            .steps = &exp_steps,
            .n = 8,
            .dst = 3,
            .src = 2,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        .{ .repeat = .{
            .dst = 4,
            .src = 5,
            .n = 8,
            .src_ne = .{ 1, 1, 1, 1 },
            .dst_ne = .{ 4, 2, 1, 1 },
            .src_strides = .{ 1, 1, 1, 1 },
            .dst_strides = .{ 1, 4, 8, 8 },
        } },
        .{ .fused_elementwise = .{
            .steps = &silu_steps,
            .n = 8,
            .dst = 6,
            .src = 3,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        testQMatmulWith(2, 8, 2),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_fused_elementwise_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 4), commands[0].op_count);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 5), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 3), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_fused_elementwise_chains);
}

test "program command stream fuses paired projection activation product chain" {
    const exp_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .neg, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .exp, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };
    const silu_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 4, .secondary_offset = 0 },
        .{ .op = .recip, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = true, .secondary_buf = 2, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        testQMatmulWith(2, 8, 2),
        .{ .fused_elementwise = .{
            .steps = &exp_steps,
            .n = 8,
            .dst = 3,
            .src = 2,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        .{ .repeat = .{
            .dst = 4,
            .src = 5,
            .n = 8,
            .src_ne = .{ 1, 1, 1, 1 },
            .dst_ne = .{ 4, 2, 1, 1 },
            .src_strides = .{ 1, 1, 1, 1 },
            .dst_strides = .{ 1, 4, 8, 8 },
        } },
        .{ .fused_elementwise = .{
            .steps = &silu_steps,
            .n = 8,
            .dst = 6,
            .src = 3,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        testQMatmulWith(2, 8, 2),
        .{ .elementwise = .{ .op = .mul, .dst = 4, .src0 = 6, .src1 = 2, .n = 8 } },
        testQMatmulWith(7, 8, 2),
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.projection_pair_fused_elementwise_chain, commands[0].kind);
    try std.testing.expectEqual(@as(u32, 6), commands[0].op_count);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 7), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.estimated_dispatches);
    try std.testing.expectEqual(@as(u32, 5), summary.estimated_saved_dispatches);
    try std.testing.expectEqual(@as(u32, 1), summary.projection_pair_fused_elementwise_chains);
}

test "program command stream keeps live repeat outputs materialized" {
    const steps = [_]backend_mod.FusedEwStep{.{ .op = .add, .is_swapped = false, .secondary_buf = 2, .secondary_offset = 0 }};
    const ops = [_]backend_mod.DeviceOp{
        .{ .repeat = .{
            .dst = 2,
            .src = 1,
            .n = 4,
            .src_ne = .{ 1, 1, 1, 1 },
            .dst_ne = .{ 4, 1, 1, 1 },
            .src_strides = .{ 1, 1, 1, 1 },
            .dst_strides = .{ 1, 4, 4, 4 },
        } },
        .{ .fused_elementwise = .{
            .steps = &steps,
            .n = 4,
            .dst = 4,
            .src = 5,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        .{ .elementwise = .{ .op = .add, .dst = 6, .src0 = 2, .src1 = 5, .n = 4 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 3), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[0].kind);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[2].kind);
}

test "program command stream keeps conflicting elementwise ops separate" {
    const ops = [_]backend_mod.DeviceOp{
        .{ .elementwise = .{ .op = .add, .dst = 1, .src0 = 0, .src1 = 0, .n = 4 } },
        .{ .elementwise = .{ .op = .mul, .dst = 1, .src0 = 0, .src1 = 0, .n = 4 } },
    };

    const commands = try buildProgramCommands(std.testing.allocator, &ops, CommandStreamPolicy.metal(4, 4));
    defer std.testing.allocator.free(commands);

    try std.testing.expectEqual(@as(usize, 2), commands.len);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[0].kind);
    try std.testing.expectEqual(ProgramCommandKind.op, commands[1].kind);

    const summary = summarizeProgramCommands(commands);
    try std.testing.expectEqual(@as(u32, 2), summary.commands);
    try std.testing.expectEqual(@as(u32, 2), summary.covered_ops);
    try std.testing.expectEqual(@as(u32, 0), summary.elementwise_batches);
    try std.testing.expectEqual(@as(u32, 2), summary.op_commands);
}

test "family pattern regions match exact contiguous family sequences" {
    const items = [_]KernelItem{
        .{ .family = .movement, .execution = .fallback, .start = 0, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 1, .len = 1 },
        .{ .family = .rope, .execution = .fallback, .start = 2, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 3, .len = 1 },
        .{ .family = .attention, .execution = .fallback, .start = 4, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 5, .len = 1 },
        .{ .family = .rope, .execution = .fallback, .start = 6, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 7, .len = 1 },
    };
    const pattern = [_]KernelFamily{ .qmatvec, .rope, .qmatvec };

    const regions = try buildFamilyPatternRegions(std.testing.allocator, &items, &pattern);
    defer std.testing.allocator.free(regions);

    try std.testing.expectEqual(@as(usize, 2), regions.len);
    try std.testing.expectEqual(@as(u32, 1), regions[0].start_item);
    try std.testing.expectEqual(@as(u32, 3), regions[0].item_count);
    try std.testing.expectEqual(@as(u32, 1), regions[0].op_start);
    try std.testing.expectEqual(@as(u32, 3), regions[0].op_count);
    try std.testing.expectEqual(@as(u32, 3), regions[0].anchor_count);
    try std.testing.expectEqual(@as(u32, 5), regions[1].start_item);
}

test "family pattern plan selects non-overlapping longest matches" {
    const items = [_]KernelItem{
        .{ .family = .qmatvec, .execution = .fallback, .start = 0, .len = 1 },
        .{ .family = .rope, .execution = .fallback, .start = 1, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 2, .len = 1 },
        .{ .family = .attention, .execution = .fallback, .start = 3, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 4, .len = 1 },
        .{ .family = .rope, .execution = .fallback, .start = 5, .len = 1 },
    };
    const short = [_]KernelFamily{ .qmatvec, .rope };
    const long = [_]KernelFamily{ .qmatvec, .rope, .qmatvec };
    const patterns = [_]FamilyPattern{
        .{ .name = "short", .families = &short },
        .{ .name = "long", .families = &long },
    };

    const regions = try buildFamilyPatternPlan(std.testing.allocator, &items, &patterns);
    defer std.testing.allocator.free(regions);

    try std.testing.expectEqual(@as(usize, 2), regions.len);
    try std.testing.expectEqual(@as(u32, 1), regions[0].pattern_index);
    try std.testing.expectEqual(@as(u32, 0), regions[0].region.start_item);
    try std.testing.expectEqual(@as(u32, 3), regions[0].region.item_count);
    try std.testing.expectEqual(@as(u32, 0), regions[1].pattern_index);
    try std.testing.expectEqual(@as(u32, 4), regions[1].region.start_item);
}

test "select pattern regions sorts candidates and removes overlaps" {
    const candidates = [_]PatternRegion{
        .{ .pattern_index = 0, .region = .{ .start_item = 2, .item_count = 2, .op_start = 2, .op_count = 2, .anchor_count = 2 } },
        .{ .pattern_index = 1, .region = .{ .start_item = 0, .item_count = 3, .op_start = 0, .op_count = 3, .anchor_count = 3 } },
        .{ .pattern_index = 2, .region = .{ .start_item = 4, .item_count = 1, .op_start = 4, .op_count = 1, .anchor_count = 1 } },
    };

    const selected = try selectPatternRegions(std.testing.allocator, &candidates);
    defer std.testing.allocator.free(selected);

    try std.testing.expectEqual(@as(usize, 2), selected.len);
    try std.testing.expectEqual(@as(u32, 1), selected[0].pattern_index);
    try std.testing.expectEqual(@as(u32, 0), selected[0].region.start_item);
    try std.testing.expectEqual(@as(u32, 2), selected[1].pattern_index);
    try std.testing.expectEqual(@as(u32, 4), selected[1].region.start_item);
}

test "region schedule covers items once and replaces selected patterns" {
    const items = [_]KernelItem{
        .{ .family = .movement, .execution = .fallback, .start = 0, .len = 2 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 2, .len = 1 },
        .{ .family = .rope, .execution = .fallback, .start = 3, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 4, .len = 1 },
        .{ .family = .attention, .execution = .fallback, .start = 5, .len = 1 },
    };
    const pattern = [_]KernelFamily{ .qmatvec, .rope, .qmatvec };
    const patterns = [_]FamilyPattern{.{ .name = "q-r-q", .families = &pattern }};

    const regions = try buildFamilyPatternPlan(std.testing.allocator, &items, &patterns);
    defer std.testing.allocator.free(regions);
    const units = try buildRegionSchedule(std.testing.allocator, &items, regions);
    defer std.testing.allocator.free(units);

    try std.testing.expectEqual(@as(usize, 3), units.len);
    try std.testing.expectEqual(ScheduleUnitKind.item, units[0].kind);
    try std.testing.expectEqual(@as(u32, 0), units[0].op_start);
    try std.testing.expectEqual(@as(u32, 2), units[0].op_count);
    try std.testing.expectEqual(ScheduleUnitKind.pattern_region, units[1].kind);
    try std.testing.expectEqual(@as(u32, 0), units[1].pattern_index);
    try std.testing.expectEqual(@as(u32, 1), units[1].start_item);
    try std.testing.expectEqual(@as(u32, 3), units[1].item_count);
    try std.testing.expectEqual(@as(u32, 2), units[1].op_start);
    try std.testing.expectEqual(@as(u32, 3), units[1].op_count);
    try std.testing.expectEqual(ScheduleUnitKind.item, units[2].kind);
    try std.testing.expectEqual(@as(u32, 5), units[2].op_start);
}

test "region command plans compile commands only for pattern regions" {
    const ops = [_]backend_mod.DeviceOp{
        testElementwise(.add),
        testElementwise(.mul),
        testMatmul(1),
        testElementwise(.relu),
    };
    const units = [_]ScheduleUnit{
        .{
            .kind = .pattern_region,
            .pattern_index = 0,
            .start_item = 0,
            .item_count = 1,
            .op_start = 0,
            .op_count = 3,
        },
        .{
            .kind = .item,
            .start_item = 1,
            .item_count = 1,
            .op_start = 3,
            .op_count = 1,
        },
    };

    const plans = try buildRegionCommandPlans(std.testing.allocator, &ops, &units, .{});
    defer {
        deinitRegionCommandPlans(std.testing.allocator, plans);
        std.testing.allocator.free(plans);
    }

    try std.testing.expectEqual(@as(usize, units.len), plans.len);
    try std.testing.expect(plans[0].commands.len > 0);
    try std.testing.expectEqual(@as(usize, 0), plans[1].commands.len);

    const summary = summarizeProgramCommands(plans[0].commands);
    try std.testing.expectEqual(@as(u32, 3), summary.covered_ops);
}

test "execution plan bundles schedule regions and cached command plans" {
    const ops = [_]backend_mod.DeviceOp{
        testQMatmul(16),
        testElementwise(.add),
        testQMatmul(16),
    };
    const schedule_policy = SchedulePolicy{
        .capabilities = backend_mod.Capabilities.metal,
        .native_kernels = .{ .qmatmul = true, .elementwise = true },
        .fine_grained = true,
        .min_backend_qmatmul_m = 0,
    };
    const stages = [_]StagePolicy{
        StagePolicy.anchored("qmatmul-stage", 0, RegionPolicy.qmatmulCluster(), 1),
    };
    const command_policy = CommandStreamPolicy.metal(4, 4);

    const plan = try buildExecutionPlan(
        std.testing.allocator,
        &ops,
        schedule_policy,
        &stages,
        command_policy,
    );
    defer plan.deinit(std.testing.allocator);

    try std.testing.expect(plan.schedule.len > 0);
    try std.testing.expect(plan.regions.len > 0);
    try std.testing.expectEqual(plan.regions.len, plan.region_commands.len);
    try std.testing.expect(plan.shapeMatches(&ops, schedule_policy));
    try std.testing.expect(plan.regionCommandPlan(0, command_policy).len > 0);

    var different_command_policy = command_policy;
    different_command_policy.max_elementwise_batch = 1;
    try std.testing.expectEqual(@as(usize, 0), plan.regionCommandPlan(0, different_command_policy).len);
}

test "region execution summary counts backend islands and transitions" {
    const items = [_]KernelItem{
        .{ .family = .movement, .execution = .fallback, .start = 0, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 1, .len = 1 },
        .{ .family = .rope, .execution = .fallback, .start = 2, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 3, .len = 1 },
        .{ .family = .elementwise, .execution = .fallback, .start = 4, .len = 1 },
        .{ .family = .qmatvec, .execution = .fallback, .start = 5, .len = 1 },
        .{ .family = .rope, .execution = .fallback, .start = 6, .len = 1 },
    };
    const pattern = [_]KernelFamily{ .qmatvec, .rope };
    const patterns = [_]FamilyPattern{.{ .name = "q-r", .families = &pattern }};

    const regions = try buildFamilyPatternPlan(std.testing.allocator, &items, &patterns);
    defer std.testing.allocator.free(regions);
    const units = try buildRegionSchedule(std.testing.allocator, &items, regions);
    defer std.testing.allocator.free(units);

    const backend_patterns = [_]u32{0};
    const summary = summarizeRegionExecution(units, &items, &backend_patterns);
    try std.testing.expectEqual(@as(u32, 5), summary.units);
    try std.testing.expectEqual(@as(u32, 2), summary.backend_units);
    try std.testing.expectEqual(@as(u32, 3), summary.fallback_units);
    try std.testing.expectEqual(@as(u32, 4), summary.backend_ops);
    try std.testing.expectEqual(@as(u32, 3), summary.fallback_ops);
    try std.testing.expectEqual(@as(u32, 2), summary.backend_islands);
    try std.testing.expectEqual(@as(u32, 1), summary.max_backend_island_units);
    try std.testing.expectEqual(@as(u32, 2), summary.max_backend_island_ops);
    try std.testing.expectEqual(@as(u32, 3), summary.execution_transitions);
}

pub const StepDynamicParams = extern struct {
    slice_pos: u32,
    seq_kv: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

pub const StepDynamicState = struct {
    params: StepDynamicParams = .{ .slice_pos = 0, .seq_kv = 0 },
    has_slice_assign: bool = false,
    has_attention: bool = false,

    pub fn needsUpload(self: StepDynamicState) bool {
        return self.has_slice_assign or self.has_attention;
    }
};

pub fn stepDynamicStateFromOps(ops: []const backend_mod.DeviceOp) StepDynamicState {
    var state = StepDynamicState{};
    for (ops) |op| {
        switch (op) {
            .slice_assign => |sa| {
                if (!state.has_slice_assign and sa.patch_stride != 0 and sa.dst_offset >= sa.dst_base_offset) {
                    state.params.slice_pos = @intCast((sa.dst_offset - sa.dst_base_offset) / sa.patch_stride);
                    state.has_slice_assign = true;
                }
            },
            .attention => |att| {
                if (!state.has_attention) {
                    state.params.seq_kv = att.seq_kv;
                    state.has_attention = true;
                }
            },
            else => {},
        }
        if (state.has_slice_assign and state.has_attention) break;
    }
    return state;
}

test "step dynamic state derives from ops" {
    const ops = [_]backend_mod.DeviceOp{
        .{ .elementwise = .{ .op = .add, .dst = 0, .src0 = 0, .src1 = 0, .n = 1 } },
        .{ .slice_assign = .{
            .dst = 0,
            .src = 0,
            .rows = 4,
            .cols = 1,
            .dst_base_offset = 8,
            .dst_offset = 20,
            .dst_row_stride = 1,
            .dst_col_stride = 4,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 4,
        } },
        .{ .attention = .{
            .dst = 0,
            .q = 0,
            .k = 0,
            .v = 0,
            .mask = 0,
            .has_mask = false,
            .d_head = 4,
            .seq_q = 1,
            .seq_kv = 17,
            .scale = 1.0,
            .q_off = 0,
            .k_off = 0,
            .v_off = 0,
            .mask_off = 0,
            .dst_off = 0,
            .q_rs = 1,
            .q_cs = 4,
            .k_rs = 1,
            .k_cs = 4,
            .v_rs = 1,
            .v_cs = 4,
            .mask_rs = 0,
            .mask_cs = 0,
            .dst_rs = 1,
            .dst_cs = 4,
        } },
    };

    const state = stepDynamicStateFromOps(&ops);
    try std.testing.expect(state.has_slice_assign);
    try std.testing.expect(state.has_attention);
    try std.testing.expect(state.needsUpload());
    try std.testing.expectEqual(@as(u32, 3), state.params.slice_pos);
    try std.testing.expectEqual(@as(u32, 17), state.params.seq_kv);
}
