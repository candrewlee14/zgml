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
};

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
