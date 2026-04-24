//! Shared helpers for compiled backend programs.

const std = @import("std");
const backend_mod = @import("../backend.zig");

pub const compute_op_softmax: u32 = 100;
pub const compute_op_layernorm: u32 = 101;
pub const compute_op_rmsnorm: u32 = 102;

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
