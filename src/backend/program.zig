//! Shared helpers for compiled backend programs.

const std = @import("std");
const backend_mod = @import("../backend.zig");

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
        return switch (self) {
            .op => "op",
            .row_chain => "row_chain",
            .rope_chain => "rope_chain",
        };
    }
};

pub const StageCommand = struct {
    kind: StageCommandKind,
    op_start: u32,
    op_count: u32,

    pub fn dispatchCount(self: StageCommand) u32 {
        return switch (self.kind) {
            .op, .row_chain, .rope_chain => 1,
        };
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
    movement_batch,
    attention_batch,
    elementwise_batch,
    projection_chain,
    projection_group,

    pub fn label(self: ProgramCommandKind) []const u8 {
        return switch (self) {
            .op => "op",
            .row_chain => "row_chain",
            .rope_chain => "rope_chain",
            .rope_batch => "rope_batch",
            .movement_batch => "movement_batch",
            .attention_batch => "attention_batch",
            .elementwise_batch => "elementwise_batch",
            .projection_chain => "projection_chain",
            .projection_group => "projection_group",
        };
    }
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
        return .{ .kind = .qmatvec, .max_anchors = max_anchors, .carry_slice_sidecars = false };
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
    max_rope_batch: u32 = 16,
    max_movement_batch: u32 = 16,
    max_attention_batch: u32 = 16,
    max_elementwise_batch: u32 = 8,

    pub fn metal(qmatvec_group_size: u32, qmatmul_group_size: u32) CommandStreamPolicy {
        return .{
            .qmatvec_group_size = qmatvec_group_size,
            .qmatmul_group_size = qmatmul_group_size,
        };
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

    pub fn coveredOpCount(self: ProgramCommand) u32 {
        return switch (self.kind) {
            .projection_group, .projection_chain => self.anchor_count + self.sidecar_count,
            .elementwise_batch => self.anchor_count,
            else => self.op_count,
        };
    }

    pub fn advanceCount(self: ProgramCommand) u32 {
        return switch (self.kind) {
            .projection_group, .elementwise_batch => 1,
            else => self.op_count,
        };
    }

    pub fn anchorIndices(self: *const ProgramCommand) []const usize {
        return self.indices[0..self.anchor_count];
    }

    pub fn sidecarIndices(self: *const ProgramCommand) []const ?usize {
        return self.sidecar_indices[0..self.anchor_count];
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
    movement_batches: u32 = 0,
    attention_batches: u32 = 0,
    elementwise_batches: u32 = 0,
    elementwise_ops: u32 = 0,
    projection_chains: u32 = 0,
    projection_chain_sidecars: u32 = 0,
    projection_groups: u32 = 0,
    projection_anchors: u32 = 0,
    projection_sidecars: u32 = 0,
    max_projection_span_ops: u32 = 0,
};

pub const max_projection_group_anchors = 8;

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
        const sa = switch (ops[idx + 1]) {
            .slice_assign => |sa| sa,
            else => continue,
        };
        const q = ops[idx].qmatmul;
        const compatible = switch (policy.kind) {
            .qmatvec => qmatvecSliceSidecarCompatible(q, sa),
            .qmatmul => qmatmulSliceSidecarCompatible(q, sa),
        };
        if (!compatible) continue;
        selection.sidecar_indices[slot] = idx + 1;
        selection.sidecar_count += 1;
        selection.end_op = @max(selection.end_op, idx + 1);
    }

    return selection;
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

pub fn buildProgramCommands(
    alloc: std.mem.Allocator,
    ops: []const backend_mod.DeviceOp,
    policy: CommandStreamPolicy,
) ![]ProgramCommand {
    var commands: std.ArrayListUnmanaged(ProgramCommand) = .empty;
    errdefer commands.deinit(alloc);

    const used = try alloc.alloc(bool, ops.len);
    defer alloc.free(used);
    @memset(used, false);

    var i: usize = 0;
    while (i < ops.len) {
        if (used[i]) {
            i += 1;
            continue;
        }

        if (findProgramCommand(ops, i, policy, used)) |command| {
            markProgramCommandUsed(used, command);
            try commands.append(alloc, command);
            i += command.advanceCount();
            continue;
        }

        const command = ProgramCommand.op(i);
        markProgramCommandUsed(used, command);
        try commands.append(alloc, command);
        i += 1;
    }

    return commands.toOwnedSlice(alloc);
}

pub fn findProgramCommand(
    ops: []const backend_mod.DeviceOp,
    start: usize,
    policy: CommandStreamPolicy,
    used: ?[]const bool,
) ?ProgramCommand {
    if (start >= ops.len) return null;
    if (used) |used_ops| {
        if (start >= used_ops.len or used_ops[start]) return null;
    }

    const op = ops[start];
    if (op == .qmatmul) {
        if (policy.projectionPolicyFor(op.qmatmul)) |projection_policy| {
            if (findProjectionGroup(ops, start, projection_policy, used)) |selection| {
                return ProgramCommand.fromProjectionSelection(selection);
            }
        }
        if (findProjectionChainCommand(ops, start, used)) |command| {
            return command;
        }
    }

    if (op == .elementwise and policy.max_elementwise_batch >= 2) {
        if (findElementwiseBatchCommand(ops, start, policy, used)) |command| {
            return command;
        }
    }

    if (policy.stage_commands) {
        if (findStageCommand(ops, start)) |stage_command| {
            if (!commandRangeTouchesUsed(stage_command.op_start, stage_command.op_count, used)) {
                return ProgramCommand.fromStageCommand(stage_command);
            }
        }
    }

    if (findContiguousBatchCommand(ops, start, policy, used)) |command| {
        return command;
    }

    return null;
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
            .movement_batch => summary.movement_batches += 1,
            .attention_batch => summary.attention_batches += 1,
            .elementwise_batch => {
                summary.elementwise_batches += 1;
                summary.elementwise_ops += command.anchor_count;
            },
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
        }
    }
    return summary;
}

pub fn markProgramCommandUsed(used: []bool, command: ProgramCommand) void {
    switch (command.kind) {
        .projection_group, .elementwise_batch => {
            for (command.anchorIndices()) |idx| {
                if (idx < used.len) used[idx] = true;
            }
            if (command.kind == .projection_group) {
                for (command.sidecarIndices()) |maybe_idx| {
                    if (maybe_idx) |idx| {
                        if (idx < used.len) used[idx] = true;
                    }
                }
            }
        },
        else => {
            const start: usize = @intCast(command.op_start);
            const end = @min(used.len, start + @as(usize, command.op_count));
            for (used[start..end]) |*slot| slot.* = true;
        },
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
        .elementwise => |e| qmatmulElementwiseSidecarCompatible(q, e),
        .fused_elementwise => |fe| qmatmulFusedElementwiseSidecarCompatible(q, fe),
        else => false,
    };
}

pub fn qmatmulElementwiseSidecarCompatible(q: anytype, e: anytype) bool {
    if (q.M == 1) return false;
    if (e.op != .add and e.op != .mul) return false;
    if (e.n != q.M * q.N) return false;
    if (qmatmulDstRowStride(q) != q.N) return false;
    return (e.src0 == q.dst and e.src0_offset == q.dst_offset) or
        (e.src1 == q.dst and e.src1_offset == q.dst_offset);
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

pub fn qmatmulSliceSidecarCompatible(q: anytype, sa: anytype) bool {
    const slice_src_col_start = qmatmulSliceSrcColStart(q, sa) orelse return false;
    return q.M != 1 and
        q.dst == sa.src and
        slice_src_col_start + sa.rows <= q.N and
        q.M == sa.cols and
        sa.src_row_stride == 1 and
        sa.src_col_stride == qmatmulDstRowStride(q);
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
