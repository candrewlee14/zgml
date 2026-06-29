//! Backend conformance tests against the shared reference executor.

const std = @import("std");
const builtin = @import("builtin");
const opts = @import("zgml_options");

const backend_mod = @import("../backend.zig");
const profile_mod = @import("../profile.zig");
const program_mod = @import("program.zig");
const reference = @import("reference.zig");
const cpu_mod = @import("cpu.zig");
const metal_mod = if (opts.use_metal and builtin.os.tag == .macos) @import("metal.zig") else struct {};

fn expectedOutput(alloc: std.mem.Allocator, program: backend_mod.DeviceProgram, out_idx: u16, out_len: usize) ![]f32 {
    var table = try reference.OwnedBufferTable.init(alloc, program.buffer_sizes);
    defer table.deinit();

    var qweights: []reference.QWeight = &.{};
    if (program.qweights.len > 0) {
        qweights = try alloc.alloc(reference.QWeight, program.qweights.len);
        for (program.qweights, 0..) |qw, i| {
            qweights[i] = .{ .data = qw.data, .scales = qw.scales, .block_size = qw.block_size };
        }
    }
    defer if (qweights.len > 0) alloc.free(qweights);

    table.upload(program.initial_uploads);
    reference.executeProgram(table.buffers, qweights, program.ops);

    const out = try alloc.alloc(f32, out_len);
    var io = [_]backend_mod.ProgramIO{.{ .buf_idx = out_idx, .host_ptr = @ptrCast(out.ptr), .size = @intCast(out_len * @sizeOf(f32)) }};
    table.download(&io);
    return out;
}

fn backendOutput(alloc: std.mem.Allocator, be: backend_mod.Backend, program: backend_mod.DeviceProgram, out_idx: u16, out_len: usize) ![]f32 {
    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    const out = try alloc.alloc(f32, out_len);
    var io = [_]backend_mod.ProgramIO{.{ .buf_idx = out_idx, .host_ptr = @ptrCast(out.ptr), .size = @intCast(out_len * @sizeOf(f32)) }};
    be.executeProgram(handle, &.{}, &io);
    return out;
}

fn expectApproxSlices(want: []const f32, got: []const f32, tol: f32) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| {
        try std.testing.expectApproxEqAbs(w, g, tol);
    }
}

fn assertBackendMatchesReference(be: backend_mod.Backend, program: backend_mod.DeviceProgram, out_idx: u16, out_len: usize, tol: f32) !void {
    if (!be.supportsProgram(program)) return;
    const alloc = std.testing.allocator;
    const expected = try expectedOutput(alloc, program, out_idx, out_len);
    defer alloc.free(expected);
    const got = try backendOutput(alloc, be, program, out_idx, out_len);
    defer alloc.free(got);
    try expectApproxSlices(expected, got, tol);
}

fn assertBackendRuntimePatchSequenceMatchesReference(be: backend_mod.Backend, program: backend_mod.DeviceProgram, windows: []const backend_mod.RuntimeWindow, out_idx: u16, out_len: usize, tol: f32) !void {
    if (!be.supportsProgram(program)) return;
    const alloc = std.testing.allocator;

    var table = try reference.OwnedBufferTable.init(alloc, program.buffer_sizes);
    defer table.deinit();

    var qweights: []reference.QWeight = &.{};
    if (program.qweights.len > 0) {
        qweights = try alloc.alloc(reference.QWeight, program.qweights.len);
        for (program.qweights, 0..) |qw, i| {
            qweights[i] = .{ .data = qw.data, .scales = qw.scales, .block_size = qw.block_size };
        }
    }
    defer if (qweights.len > 0) alloc.free(qweights);

    var ref_stencil = try program_mod.ProgramStencil.initProgram(alloc, program);
    defer ref_stencil.deinit(alloc);

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    const expected = try alloc.alloc(f32, out_len);
    defer alloc.free(expected);
    const got = try alloc.alloc(f32, out_len);
    defer alloc.free(got);

    table.upload(program.initial_uploads);
    for (windows) |window| {
        const ref_status = ref_stencil.patchRuntimeWindow(window);
        if (ref_status == .invalid) return error.InvalidRuntimePatch;
        reference.executeProgram(table.buffers, qweights, ref_stencil.ops);
        var expected_io = [_]backend_mod.ProgramIO{.{ .buf_idx = out_idx, .host_ptr = @ptrCast(expected.ptr), .size = @intCast(out_len * @sizeOf(f32)) }};
        table.download(&expected_io);

        try std.testing.expectEqual(ref_status, be.patchRuntimeWindow(handle, window));
        var got_io = [_]backend_mod.ProgramIO{.{ .buf_idx = out_idx, .host_ptr = @ptrCast(got.ptr), .size = @intCast(out_len * @sizeOf(f32)) }};
        be.executeProgram(handle, &.{}, &got_io);
        try expectApproxSlices(expected, got, tol);
    }
}

fn assertBackendRuntimePatchProfileEvidence(be: backend_mod.Backend) !void {
    const ops = [_]backend_mod.DeviceOp{
        .{ .slice_assign = .{
            .dst = 1,
            .src = 0,
            .rows = 1,
            .cols = 1,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 1,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 1,
            .patch_stride = 1,
        } },
        .{ .attention = .{
            .dst = 4,
            .q = 2,
            .k = 3,
            .v = 3,
            .mask = 2,
            .has_mask = false,
            .d_head = 4,
            .seq_q = 1,
            .seq_kv = 1,
            .scale = 0.5,
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
            .mask_rs = 1,
            .mask_cs = 1,
            .dst_rs = 1,
            .dst_cs = 4,
            .patch_seq_kv = true,
        } },
    };
    const sizes = [_]usize{ 1, 8, 4, 20, 4 };
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = sizes.len, .buffer_sizes = &sizes, .initial_uploads = &.{} };
    if (!be.supportsProgram(program)) return;

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var before = profile_mod.RuntimeProfile{};
    be.addRuntimeProfileTo(handle, &before);
    try std.testing.expectEqual(@as(u32, 2), before.runtime_patch_shape.runtime_patch_holes);
    try std.testing.expectEqual(@as(u32, 1), before.runtime_patch_shape.runtime_patch_cache_write_pos_holes);
    try std.testing.expectEqual(@as(u32, 1), before.runtime_patch_shape.runtime_patch_attention_seq_kv_holes);
    try std.testing.expect(before.runtime_patch_shape.runtime_patch_stencil_hash != 0);
    try std.testing.expectEqual(@as(u32, 0), before.call_count);

    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, be.patchRuntimeWindow(handle, try backend_mod.RuntimeWindow.init(0, 1)));
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.unchanged, be.patchRuntimeWindow(handle, try backend_mod.RuntimeWindow.init(0, 1)));
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, be.patchRuntimeWindow(handle, try backend_mod.RuntimeWindow.init(2, 3)));
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.invalid, be.patchRuntimeWindow(handle, try backend_mod.RuntimeWindow.init(8, 1)));

    var after = profile_mod.RuntimeProfile{};
    be.addRuntimeProfileTo(handle, &after);
    try std.testing.expectEqual(before.runtime_patch_shape, after.runtime_patch_shape);
    try std.testing.expectEqual(@as(u64, 4), after.runtime_patch_call_count);
    try std.testing.expectEqual(@as(u64, 2), after.runtime_patch_changed_count);
    try std.testing.expectEqual(@as(u64, 1), after.runtime_patch_invalid_count);
    try std.testing.expectEqual(@as(u32, 0), after.call_count);
}

fn compileProgramFromTemporaryState(be: backend_mod.Backend, alloc: std.mem.Allocator) !?backend_mod.Backend.CompiledHandle {
    var src = [_]f32{ 0, 4, 0, -4 };
    var dst_init = [_]f32{ 99, 99, 99, 99 };

    const qdata = try alloc.alloc(i8, 16);
    defer alloc.free(qdata);
    @memcpy(qdata, &[_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    });

    const scales = try alloc.alloc(f32, 4);
    defer alloc.free(scales);
    @memset(scales, 1);

    const qweights = try alloc.alloc(backend_mod.QuantizedWeightUpload, 1);
    defer alloc.free(qweights);
    qweights[0] = .{ .data = qdata, .scales = scales, .rows = 4, .cols = 4, .block_size = 4 };

    const steps = try alloc.alloc(backend_mod.FusedEwStep, 1);
    defer alloc.free(steps);
    steps[0] = .{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 };

    const ops = try alloc.alloc(backend_mod.DeviceOp, 2);
    defer alloc.free(ops);
    ops[0] = .{ .qmatmul = .{ .dst = 1, .input = 0, .weight_idx = 0, .M = 1, .N = 4, .K = 4 } };
    ops[1] = .{ .fused_elementwise = .{ .steps = steps, .n = 4, .dst = 2, .src = 1, .dst_offset = 0, .src_offset = 0 } };

    const sizes = try alloc.alloc(usize, 3);
    defer alloc.free(sizes);
    @memcpy(sizes, &[_]usize{ 4, 4, 4 });

    const uploads = try alloc.alloc(backend_mod.ProgramIO, 2);
    defer alloc.free(uploads);
    @memcpy(uploads, &[_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&src), .size = src.len * @sizeOf(f32) },
        .{ .buf_idx = 2, .host_ptr = @ptrCast(&dst_init), .size = dst_init.len * @sizeOf(f32) },
    });

    const program = backend_mod.DeviceProgram{ .ops = ops, .n_buffers = 3, .buffer_sizes = sizes, .initial_uploads = uploads, .qweights = qweights };
    if (!be.supportsProgram(program)) return null;
    return be.compileProgram(program) orelse error.CompileFailed;
}

fn assertBackendOwnsCompiledProgramState(be: backend_mod.Backend, tol: f32) !void {
    const alloc = std.testing.allocator;
    const handle = (try compileProgramFromTemporaryState(be, alloc)) orelse return;
    defer be.freeProgram(handle);

    const poison = try alloc.alloc(backend_mod.FusedEwStep, 1);
    defer alloc.free(poison);
    poison[0] = .{ .op = .neg, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 };

    const poison_data = try alloc.alloc(i8, 16);
    defer alloc.free(poison_data);
    @memset(poison_data, 0);

    const poison_scales = try alloc.alloc(f32, 4);
    defer alloc.free(poison_scales);
    @memset(poison_scales, 0);

    const poison_qweights = try alloc.alloc(backend_mod.QuantizedWeightUpload, 1);
    defer alloc.free(poison_qweights);
    poison_qweights[0] = .{ .data = poison_data, .scales = poison_scales, .rows = 4, .cols = 4, .block_size = 4 };

    var got = [_]f32{0} ** 4;
    const out = [_]backend_mod.ProgramIO{.{ .buf_idx = 2, .host_ptr = @ptrCast(&got), .size = got.len * @sizeOf(f32) }};
    be.executeProgram(handle, &.{}, &out);
    try expectApproxSlices(&.{ 0, 4, 0, 0 }, &got, tol);
}

fn runCoreCases(be: backend_mod.Backend, tol: f32) !void {
    {
        var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
        var b = [_]f32{ 7, 8, 9, 10, 11, 12 };
        const ops = [_]backend_mod.DeviceOp{.{ .matmul = .{
            .dst = 2,
            .a = 0,
            .b = 1,
            .geom = .{ .M = 2, .N = 2, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
        } }};
        const sizes = [_]usize{ 6, 6, 4 };
        const uploads = [_]backend_mod.ProgramIO{
            .{ .buf_idx = 0, .host_ptr = @ptrCast(&a), .size = 6 * @sizeOf(f32) },
            .{ .buf_idx = 1, .host_ptr = @ptrCast(&b), .size = 6 * @sizeOf(f32) },
        };
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        try assertBackendMatchesReference(be, program, 2, 4, tol);
    }

    {
        var input = [_]f32{ 99, 1, 2, 3, 99, -1, 0.5, 4, 99 };
        var dst_init = [_]f32{-7} ** 9;
        const qdata = [_]i8{ 2, -1, 3, 4, -2, 1, -3, 5, 2 };
        const scales = [_]f32{ 0.5, 0.25, 1.0 };
        const ops = [_]backend_mod.DeviceOp{.{ .qmatmul = .{
            .dst = 1,
            .input = 0,
            .weight_idx = 0,
            .M = 2,
            .N = 3,
            .K = 3,
            .input_offset = 1,
            .input_row_stride = 4,
            .dst_offset = 1,
            .dst_row_stride = 4,
        } }};
        const sizes = [_]usize{ 9, 9 };
        const uploads = [_]backend_mod.ProgramIO{
            .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * @sizeOf(f32) },
            .{ .buf_idx = 1, .host_ptr = @ptrCast(&dst_init), .size = dst_init.len * @sizeOf(f32) },
        };
        const qweights = [_]backend_mod.QuantizedWeightUpload{.{
            .data = &qdata,
            .scales = &scales,
            .rows = 3,
            .cols = 3,
            .block_size = 4,
        }};
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &sizes, .initial_uploads = &uploads, .qweights = &qweights };
        try assertBackendMatchesReference(be, program, 1, 9, tol);
    }

    {
        var a = [_]f32{ 1, 2, 3, 4 };
        var b = [_]f32{ 10, 20, 30, 40 };
        const ops = [_]backend_mod.DeviceOp{.{ .elementwise = .{
            .op = .add,
            .dst = 2,
            .src0 = 0,
            .src1 = 1,
            .n = 4,
        } }};
        const sizes = [_]usize{ 4, 4, 4 };
        const uploads = [_]backend_mod.ProgramIO{
            .{ .buf_idx = 0, .host_ptr = @ptrCast(&a), .size = 4 * @sizeOf(f32) },
            .{ .buf_idx = 1, .host_ptr = @ptrCast(&b), .size = 4 * @sizeOf(f32) },
        };
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        try assertBackendMatchesReference(be, program, 2, 4, tol);
    }

    {
        var src = [_]f32{ 1, -2, 3, 4, 5, -6 };
        const ops = [_]backend_mod.DeviceOp{
            .{ .reduce = .{
                .op = .sum,
                .dst = 1,
                .src = 0,
                .n_out = 2,
                .reduce_size = 3,
            } },
            .{ .reduce = .{
                .op = .max,
                .dst = 1,
                .src = 0,
                .n_out = 2,
                .reduce_size = 3,
                .dst_offset = 2,
            } },
        };
        const sizes = [_]usize{ 6, 4 };
        const uploads = [_]backend_mod.ProgramIO{.{ .buf_idx = 0, .host_ptr = @ptrCast(&src), .size = 6 * @sizeOf(f32) }};
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        try assertBackendMatchesReference(be, program, 1, 4, tol);
    }

    {
        var src = [_]f32{ 7, 8 };
        const ops = [_]backend_mod.DeviceOp{.{ .repeat = .{
            .dst = 1,
            .src = 0,
            .n = 6,
            .src_ne = .{ 2, 1, 1, 1 },
            .dst_ne = .{ 2, 3, 1, 1 },
            .src_strides = .{ 1, 2, 2, 2 },
            .dst_strides = .{ 1, 2, 6, 6 },
        } }};
        const sizes = [_]usize{ 2, 6 };
        const uploads = [_]backend_mod.ProgramIO{.{ .buf_idx = 0, .host_ptr = @ptrCast(&src), .size = 2 * @sizeOf(f32) }};
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        try assertBackendMatchesReference(be, program, 1, 6, tol);
    }

    {
        var src = [_]f32{ 99, 2, 3, 5, 6, 77 };
        var dst = [_]f32{ 10, 11, 12, 13, 14, 15, 16, 17 };
        const ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
            .dst = 1,
            .src = 0,
            .rows = 2,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 2,
            .dst_row_stride = 1,
            .dst_col_stride = 2,
            .src_offset = 1,
            .src_row_stride = 1,
            .src_col_stride = 2,
            .patch_stride = 2,
        } }};
        const sizes = [_]usize{ 6, 8 };
        const uploads = [_]backend_mod.ProgramIO{
            .{ .buf_idx = 0, .host_ptr = @ptrCast(&src), .size = 6 * @sizeOf(f32) },
            .{ .buf_idx = 1, .host_ptr = @ptrCast(&dst), .size = 8 * @sizeOf(f32) },
        };
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        try assertBackendMatchesReference(be, program, 1, 8, tol);
    }

    {
        var src = [_]f32{ 1, 2, 3, -1, 0, 1 };
        const ops = [_]backend_mod.DeviceOp{.{ .softmax = .{
            .dst = 1,
            .src = 0,
            .rows = 2,
            .cols = 3,
        } }};
        const sizes = [_]usize{ 6, 6 };
        const uploads = [_]backend_mod.ProgramIO{.{ .buf_idx = 0, .host_ptr = @ptrCast(&src), .size = 6 * @sizeOf(f32) }};
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        try assertBackendMatchesReference(be, program, 1, 6, tol);
    }

    {
        var src = [_]f32{ 1, 2, 3, 4, -1, 0, 1, 2 };
        const ops = [_]backend_mod.DeviceOp{
            .{ .layernorm = .{
                .dst = 1,
                .src = 0,
                .rows = 2,
                .cols = 4,
                .eps = 1e-5,
            } },
            .{ .rmsnorm = .{
                .dst = 1,
                .src = 0,
                .rows = 2,
                .cols = 4,
                .eps = 1e-5,
                .dst_offset = 8,
            } },
        };
        const sizes = [_]usize{ 8, 16 };
        const uploads = [_]backend_mod.ProgramIO{.{ .buf_idx = 0, .host_ptr = @ptrCast(&src), .size = 8 * @sizeOf(f32) }};
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        try assertBackendMatchesReference(be, program, 1, 16, tol);
    }

    {
        var src = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        var cs = [_]f32{
            1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 1, 1,
        };
        const ops = [_]backend_mod.DeviceOp{.{ .rope = .{
            .dst = 2,
            .src = 0,
            .cos_sin = 1,
            .half_d = 2,
            .seq_len = 2,
            .src_off = 0,
            .cs_off = 0,
            .dst_off = 0,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 8,
        } }};
        const sizes = [_]usize{ 8, 16, 8 };
        const uploads = [_]backend_mod.ProgramIO{
            .{ .buf_idx = 0, .host_ptr = @ptrCast(&src), .size = 8 * @sizeOf(f32) },
            .{ .buf_idx = 1, .host_ptr = @ptrCast(&cs), .size = 16 * @sizeOf(f32) },
        };
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        try assertBackendMatchesReference(be, program, 2, 8, tol);
    }

    {
        var q = [_]f32{
            0.2,  0.1, -0.3, 0.4,
            -0.1, 0.5, 0.2,  -0.4,
        };
        var k = [_]f32{
            0.1,  0.2,  0.3, 0.4,
            -0.2, 0.3,  0.1, -0.1,
            0.5,  -0.4, 0.2, 0.1,
        };
        var v = [_]f32{
            1,    2,     3,   4,
            -1,   0.5,   2,   -0.5,
            0.25, -0.75, 1.5, 2.5,
        };
        var mask = [_]f32{
            0, 0,     -std.math.inf(f32),
            0, -0.25, 0,
        };
        const ops = [_]backend_mod.DeviceOp{.{ .attention = .{
            .dst = 4,
            .q = 0,
            .k = 1,
            .v = 2,
            .mask = 3,
            .has_mask = true,
            .d_head = 4,
            .seq_q = 2,
            .seq_kv = 3,
            .scale = 0.5,
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
            .mask_rs = 1,
            .mask_cs = 3,
            .dst_rs = 1,
            .dst_cs = 4,
        } }};
        const sizes = [_]usize{ 8, 12, 12, 6, 8 };
        const uploads = [_]backend_mod.ProgramIO{
            .{ .buf_idx = 0, .host_ptr = @ptrCast(&q), .size = 8 * @sizeOf(f32) },
            .{ .buf_idx = 1, .host_ptr = @ptrCast(&k), .size = 12 * @sizeOf(f32) },
            .{ .buf_idx = 2, .host_ptr = @ptrCast(&v), .size = 12 * @sizeOf(f32) },
            .{ .buf_idx = 3, .host_ptr = @ptrCast(&mask), .size = 6 * @sizeOf(f32) },
        };
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 5, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        try assertBackendMatchesReference(be, program, 4, 8, tol);
    }

    {
        var src = [_]f32{ 1, -2, 4, 9 };
        var addend = [_]f32{ 10, 20, 30, 40 };
        const steps = [_]backend_mod.FusedEwStep{
            .{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
            .{ .op = .sqrt, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
            .{ .op = .add, .is_swapped = false, .secondary_buf = 1, .secondary_offset = 0 },
        };
        const ops = [_]backend_mod.DeviceOp{.{ .fused_elementwise = .{
            .steps = &steps,
            .n = 4,
            .dst = 2,
            .src = 0,
            .dst_offset = 0,
            .src_offset = 0,
        } }};
        const sizes = [_]usize{ 4, 4, 4 };
        const uploads = [_]backend_mod.ProgramIO{
            .{ .buf_idx = 0, .host_ptr = @ptrCast(&src), .size = 4 * @sizeOf(f32) },
            .{ .buf_idx = 1, .host_ptr = @ptrCast(&addend), .size = 4 * @sizeOf(f32) },
        };
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        try assertBackendMatchesReference(be, program, 2, 4, tol);
    }
}

fn runRuntimePatchCases(be: backend_mod.Backend, tol: f32) !void {
    {
        var src = [_]f32{ 1, 2, 3, 4 };
        var dst = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1 };
        const ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
            .dst = 1,
            .src = 0,
            .rows = 1,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 1,
            .src_offset = 1,
            .src_row_stride = 1,
            .src_col_stride = 1,
            .patch_stride = 2,
        } }};
        const sizes = [_]usize{ 4, 8 };
        const uploads = [_]backend_mod.ProgramIO{
            .{ .buf_idx = 0, .host_ptr = @ptrCast(&src), .size = src.len * @sizeOf(f32) },
            .{ .buf_idx = 1, .host_ptr = @ptrCast(&dst), .size = dst.len * @sizeOf(f32) },
        };
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 2, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        const windows = [_]backend_mod.RuntimeWindow{
            try backend_mod.RuntimeWindow.init(3, 1),
            try backend_mod.RuntimeWindow.init(1, 1),
        };
        try assertBackendRuntimePatchSequenceMatchesReference(be, program, &windows, 1, dst.len, tol);
    }

    {
        var q = [_]f32{ 0.2, 0.1, -0.3, 0.4 };
        var k = [_]f32{
            0.1,  0.2,  0.3, 0.4,
            -0.2, 0.3,  0.1, -0.1,
            0.5,  -0.4, 0.2, 0.1,
        };
        var v = [_]f32{
            1,    2,     3,   4,
            -1,   0.5,   2,   -0.5,
            0.25, -0.75, 1.5, 2.5,
        };
        const ops = [_]backend_mod.DeviceOp{.{ .attention = .{
            .dst = 3,
            .q = 0,
            .k = 1,
            .v = 2,
            .mask = 0,
            .has_mask = false,
            .d_head = 4,
            .seq_q = 1,
            .seq_kv = 1,
            .scale = 0.5,
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
            .mask_rs = 1,
            .mask_cs = 1,
            .dst_rs = 1,
            .dst_cs = 4,
            .patch_seq_kv = true,
        } }};
        const sizes = [_]usize{ 4, 12, 12, 4 };
        const uploads = [_]backend_mod.ProgramIO{
            .{ .buf_idx = 0, .host_ptr = @ptrCast(&q), .size = q.len * @sizeOf(f32) },
            .{ .buf_idx = 1, .host_ptr = @ptrCast(&k), .size = k.len * @sizeOf(f32) },
            .{ .buf_idx = 2, .host_ptr = @ptrCast(&v), .size = v.len * @sizeOf(f32) },
        };
        const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 4, .buffer_sizes = &sizes, .initial_uploads = &uploads };
        const windows = [_]backend_mod.RuntimeWindow{
            try backend_mod.RuntimeWindow.init(0, 2),
            try backend_mod.RuntimeWindow.init(0, 3),
        };
        try assertBackendRuntimePatchSequenceMatchesReference(be, program, &windows, 3, 4, tol);
    }
}

test "cpu backend conforms to reference core ops" {
    var cpu = cpu_mod.CpuBackend{};
    try runCoreCases(cpu.backend(), 1e-5);
}

test "cpu backend applies runtime patch values" {
    var cpu = cpu_mod.CpuBackend{};
    try runRuntimePatchCases(cpu.backend(), 1e-5);
}

test "cpu backend reports runtime patch profile evidence" {
    var cpu = cpu_mod.CpuBackend{};
    try assertBackendRuntimePatchProfileEvidence(cpu.backend());
}

test "cpu backend compiled programs own program state" {
    var cpu = cpu_mod.CpuBackend{};
    try assertBackendOwnsCompiledProgramState(cpu.backend(), 1e-5);
}

test "metal backend conforms to reference core ops" {
    if (!opts.use_metal or builtin.os.tag != .macos) return;
    var metal = metal_mod.MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setFineGrainedProgramDispatch(true);
    try runCoreCases(metal.backend(), 1e-5);
}

test "metal backend applies runtime patch values" {
    if (!opts.use_metal or builtin.os.tag != .macos) return;
    var metal = metal_mod.MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setFineGrainedProgramDispatch(true);
    try runRuntimePatchCases(metal.backend(), 1e-5);
}

test "metal backend reports runtime patch profile evidence" {
    if (!opts.use_metal or builtin.os.tag != .macos) return;
    var metal = metal_mod.MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setFineGrainedProgramDispatch(true);
    try assertBackendRuntimePatchProfileEvidence(metal.backend());
}

test "metal backend compiled programs own program state" {
    if (!opts.use_metal or builtin.os.tag != .macos) return;
    var metal = metal_mod.MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setFineGrainedProgramDispatch(true);
    try assertBackendOwnsCompiledProgramState(metal.backend(), 1e-5);
}
