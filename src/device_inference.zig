//! Device-accelerated inference wrapper.
//!
//! Takes an InferencePlan and a Backend, builds a DeviceProgram from
//! the plan's execution steps, compiles it via the backend, and
//! executes it per token. This is an opt-in layer — the core
//! InferencePlan stays clean and backend-agnostic.
//!
//! ```
//! var metal = try MetalBackend.init();
//! var device = try DeviceInference(f32, config).init(&session.plan, metal.backend(), alloc);
//! defer device.deinit();
//! const logits = device.step(&session, token_id, pos);
//! ```

const std = @import("std");
const backend_mod = @import("backend.zig");
const fused = @import("tensor/fused.zig");

const Tensor = @import("tensor.zig").Tensor;
const GPTConfig = @import("models/gpt.zig").GPTConfig;
const QuantizedWeight = @import("quant.zig").QuantizedWeight;

pub fn DeviceInference(comptime T: type, comptime config: GPTConfig) type {
    const Plan = @import("inference.zig").InferencePlan(T, config);
    const d_model = config.d_model;
    const max_seq = config.max_seq_len;

    return struct {
        const Self = @This();

        be: backend_mod.Backend,
        alloc: std.mem.Allocator,
        compiled: backend_mod.Backend.CompiledHandle,
        program_ops: []backend_mod.DeviceOp,
        program_inputs: []backend_mod.ProgramIO,
        program_outputs: []backend_mod.ProgramIO,
        slice_assign_op_indices: []u32,

        pub fn init(plan: *Plan, be: backend_mod.Backend, alloc: std.mem.Allocator) !Self {
            const nodes = plan.graph.nodes.items[0..plan.graph.forward_node_count];
            const steps = plan.graph.forward_execution_steps.items;

            // ── Assign buffer indices to unique host data pointers ────
            var ptr_to_idx = std.AutoHashMap([*]T, u16).init(alloc);
            defer ptr_to_idx.deinit();
            var buf_sizes: std.ArrayListUnmanaged(usize) = .empty;
            defer buf_sizes.deinit(alloc);
            var uploads_list: std.ArrayListUnmanaged(backend_mod.ProgramIO) = .empty;
            defer uploads_list.deinit(alloc);

            for (nodes) |node| {
                const tensors = [_]?*const Tensor(T){ node, node.src0, node.src1 };
                for (tensors) |maybe_t| {
                    const t = maybe_t orelse continue;
                    const entry = try ptr_to_idx.getOrPut(t.data.ptr);
                    if (!entry.found_existing) {
                        entry.value_ptr.* = @intCast(buf_sizes.items.len);
                        try buf_sizes.append(alloc, @max(t.data.len, 1));
                        if (t.opTag() == .none and t.data.len > 0) {
                            try uploads_list.append(alloc, .{
                                .buf_idx = entry.value_ptr.*,
                                .host_ptr = @ptrCast(t.data.ptr),
                                .size = @intCast(t.data.len * @sizeOf(T)),
                            });
                        }
                    } else {
                        buf_sizes.items[entry.value_ptr.*] = @max(buf_sizes.items[entry.value_ptr.*], @max(t.data.len, 1));
                    }
                }
            }

            // ── Build DeviceOp list ───────────────────────────────────
            var ops_list: std.ArrayListUnmanaged(backend_mod.DeviceOp) = .empty;
            defer ops_list.deinit(alloc);

            if (steps.len > 0) {
                for (steps) |exec_step| {
                    switch (exec_step) {
                        .fusion => |idx| {
                            const fp = plan.graph.fused_chains.items[idx];
                            switch (fp.kind()) {
                                .softmax => |sm_tag| {
                                    _ = sm_tag;
                                    const sm = fp.payload.softmax;
                                    try ops_list.append(alloc, .{ .softmax = .{
                                        .dst = bufIdx(&ptr_to_idx, sm.output),
                                        .src = bufIdx(&ptr_to_idx, sm.input),
                                        .rows = @intCast(sm.input.ne[1]),
                                        .cols = @intCast(sm.input.ne[0]),
                                    } });
                                },
                                .layer_norm => {
                                    const ln = fp.payload.layer_norm;
                                    try ops_list.append(alloc, .{ .layernorm = .{
                                        .dst = bufIdx(&ptr_to_idx, ln.output),
                                        .src = bufIdx(&ptr_to_idx, ln.input),
                                        .rows = @intCast(ln.input.ne[1]),
                                        .cols = @intCast(ln.input.ne[0]),
                                        .eps = ln.eps_like.data[0],
                                    } });
                                },
                                .elementwise_chain => {
                                    for (fp.payload.elementwise_chain.nodes) |node| {
                                        try appendNodeOp(plan, &ops_list, &ptr_to_idx, node, alloc);
                                    }
                                },
                                // CNN and training fusions — not supported on GPU inference path.
                                .conv2d, .conv2d_bwd_input, .conv2d_bwd_kernel,
                                .max_pool2d, .max_pool2d_bwd,
                                .log_softmax, .cross_entropy => {},
                            }
                        },
                        .node => |node_ptr| try appendNodeOp(plan, &ops_list, &ptr_to_idx, node_ptr, alloc),
                    }
                }
            } else {
                for (nodes) |node| try appendNodeOp(plan, &ops_list, &ptr_to_idx, node, alloc);
            }

            // ── Per-step I/O ──────────────────────────────────────────
            const inputs = try alloc.alloc(backend_mod.ProgramIO, 3);
            inputs[0] = .{ .buf_idx = bufIdx(&ptr_to_idx, plan.token_input), .host_ptr = @ptrCast(plan.token_input.data.ptr), .size = @intCast(d_model * @sizeOf(T)) };
            inputs[1] = .{ .buf_idx = bufIdx(&ptr_to_idx, plan.pos_input), .host_ptr = @ptrCast(plan.pos_input.data.ptr), .size = @intCast(d_model * @sizeOf(T)) };
            inputs[2] = .{ .buf_idx = bufIdx(&ptr_to_idx, plan.attn_mask), .host_ptr = @ptrCast(plan.attn_mask.data.ptr), .size = @intCast(max_seq * @sizeOf(T)) };

            const outputs = try alloc.alloc(backend_mod.ProgramIO, 1);
            outputs[0] = .{ .buf_idx = bufIdx(&ptr_to_idx, plan.logits), .host_ptr = @ptrCast(plan.logits_buf.ptr), .size = @intCast(config.vocab_size * @sizeOf(T)) };

            // ── Quantized weights ─────────────────────────────────────
            const qw_uploads = try alloc.alloc(backend_mod.QuantizedWeightUpload, plan.quant_weights.len);
            defer alloc.free(qw_uploads);
            for (plan.quant_weights, 0..) |qw, i| {
                qw_uploads[i] = .{ .data = qw.data, .scales = qw.scales, .rows = qw.rows, .cols = qw.cols, .block_size = qw.block_size };
            }

            // ── Slice assign tracking ─────────────────────────────────
            var sa_list: std.ArrayListUnmanaged(u32) = .empty;
            defer sa_list.deinit(alloc);
            for (ops_list.items, 0..) |dop, i| {
                if (dop == .slice_assign) try sa_list.append(alloc, @intCast(i));
            }

            const owned_ops = try alloc.dupe(backend_mod.DeviceOp, ops_list.items);

            // ── Compile ───────────────────────────────────────────────
            const program = backend_mod.DeviceProgram{
                .ops = owned_ops,
                .n_buffers = @intCast(buf_sizes.items.len),
                .buffer_sizes = buf_sizes.items,
                .initial_uploads = uploads_list.items,
                .qweights = qw_uploads,
            };

            const compiled = be.compileProgram(program) orelse return error.CompileFailed;

            return .{
                .be = be,
                .alloc = alloc,
                .compiled = compiled,
                .program_ops = owned_ops,
                .program_inputs = inputs,
                .program_outputs = outputs,
                .slice_assign_op_indices = try sa_list.toOwnedSlice(alloc),
            };
        }

        pub fn deinit(self: *Self) void {
            self.be.freeProgram(self.compiled);
            self.alloc.free(self.program_ops);
            self.alloc.free(self.program_inputs);
            self.alloc.free(self.program_outputs);
            if (self.slice_assign_op_indices.len > 0) self.alloc.free(self.slice_assign_op_indices);
        }

        /// Execute one token on device. Patches inputs, runs compiled program, returns logits.
        pub fn step(self: *Self, plan: *Plan, model: anytype, token_id: usize, pos: usize) []const T {
            std.debug.assert(token_id < config.vocab_size);
            std.debug.assert(pos < config.max_seq_len);

            // Patch host-side inputs (shared memory — visible to GPU).
            const tok_data = model.embed.token_embed.inner.data;
            @memcpy(plan.token_input.data[0..d_model], tok_data[token_id * d_model ..][0..d_model]);

            const pe_data = if (config.learnable_pos_embed) model.embed.pos_encode.inner.data else model.embed.pos_encode.data;
            @memcpy(plan.pos_input.data[0..d_model], pe_data[pos * d_model ..][0..d_model]);

            for (plan.attn_mask.data[0..max_seq], 0..) |*v, i| {
                v.* = if (i <= pos) 0 else -std.math.inf(T);
            }

            // Patch slice_assign offsets for current position.
            for (self.slice_assign_op_indices) |idx| {
                self.program_ops[idx].slice_assign.dst_offset = @intCast(pos);
            }

            // Execute compiled program.
            self.be.executeProgram(self.compiled, self.program_inputs, self.program_outputs);
            return plan.logits_buf;
        }

        // ── Helpers ──────────────────────────────────────────────

        fn bufIdx(map: *const std.AutoHashMap([*]T, u16), tensor: *const Tensor(T)) u16 {
            return map.get(tensor.data.ptr).?;
        }

        fn appendNodeOp(plan: *const Plan, ops: *std.ArrayListUnmanaged(backend_mod.DeviceOp), ptr_to_idx: *const std.AutoHashMap([*]T, u16), node: *Tensor(T), alloc: std.mem.Allocator) !void {
            const op = node.opTag();
            if (op == .none or op == .view or op == .as_strided or op == .reshape or
                op == .transpose or op == .permute or op == .broadcast_to) return;

            if (op == .matmul) {
                const s0 = node.src0.?;
                const s1 = node.src1.?;
                const flags = node.matmul_flags;
                const M = if (flags.trans0) s0.ne[0] else s0.ne[1];
                const N = if (flags.trans1) s1.ne[1] else s1.ne[0];
                const K = if (flags.trans0) s0.ne[1] else s0.ne[0];

                if (plan.quant_map.get(node)) |qi| {
                    const input_tensor = if (s1.isParam()) s0 else s1;
                    try ops.append(alloc, .{ .qmatmul = .{
                        .dst = bufIdx(ptr_to_idx, node), .input = bufIdx(ptr_to_idx, input_tensor),
                        .weight_idx = @intCast(qi), .M = @intCast(M), .N = @intCast(N), .K = @intCast(K),
                    } });
                } else {
                    try ops.append(alloc, .{ .matmul = .{
                        .dst = bufIdx(ptr_to_idx, node), .a = bufIdx(ptr_to_idx, s0), .b = bufIdx(ptr_to_idx, s1),
                        .geom = .{
                            .M = M, .N = N, .K = K,
                            .a_row_stride = if (flags.trans0) s0.strides[0] else s0.strides[1],
                            .a_col_stride = if (flags.trans0) s0.strides[1] else s0.strides[0],
                            .b_row_stride = if (flags.trans1) s1.strides[0] else s1.strides[1],
                            .b_col_stride = if (flags.trans1) s1.strides[1] else s1.strides[0],
                            .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = N,
                        },
                    } });
                }
                return;
            }

            const dst_idx = bufIdx(ptr_to_idx, node);
            const src0_idx = if (node.src0) |s| bufIdx(ptr_to_idx, s) else dst_idx;
            const src1_idx = if (node.src1) |s| bufIdx(ptr_to_idx, s) else dst_idx;

            switch (op) {
                .add, .mul, .neg, .abs, .sgn, .step, .relu, .sqrt, .recip, .exp, .log, .gelu => {
                    try ops.append(alloc, .{ .elementwise = .{
                        .op = op, .dst = dst_idx, .src0 = src0_idx, .src1 = src1_idx,
                        .n = @intCast(node.nElems()),
                        .dst_offset = @intCast(node.storage_offset),
                        .src0_offset = if (node.src0) |s| @intCast(s.storage_offset) else 0,
                        .src1_offset = if (node.src1) |s| @intCast(s.storage_offset) else 0,
                    } });
                },
                .sum, .max => {
                    try ops.append(alloc, .{ .reduce = .{ .op = op, .dst = dst_idx, .src = src0_idx, .n_out = @intCast(node.nElems()), .reduce_size = @intCast(node.src0.?.ne[0]) } });
                },
                .repeat => {
                    const src = node.src0.?;
                    try ops.append(alloc, .{ .repeat = .{
                        .dst = dst_idx, .src = src0_idx, .n = @intCast(node.nElems()),
                        .src_ne = .{ @intCast(src.ne[0]), @intCast(src.ne[1]), @intCast(src.ne[2]), @intCast(src.ne[3]) },
                        .dst_ne = .{ @intCast(node.ne[0]), @intCast(node.ne[1]), @intCast(node.ne[2]), @intCast(node.ne[3]) },
                        .src_strides = .{ @intCast(src.strides[0]), @intCast(src.strides[1]), @intCast(src.strides[2]), @intCast(src.strides[3]) },
                        .dst_strides = .{ @intCast(node.strides[0]), @intCast(node.strides[1]), @intCast(node.strides[2]), @intCast(node.strides[3]) },
                    } });
                },
                .rmsnorm => {
                    const src = node.src0.?;
                    try ops.append(alloc, .{ .rmsnorm = .{
                        .dst = dst_idx, .src = src0_idx,
                        .rows = @intCast(src.ne[1]),
                        .cols = @intCast(src.ne[0]),
                        .eps = node.op_eps,
                    } });
                },
                .slice_assign => {
                    try ops.append(alloc, .{ .slice_assign = .{
                        .dst = if (node.src1) |s| bufIdx(ptr_to_idx, s) else dst_idx,
                        .src = src0_idx, .n = @intCast(node.src0.?.nElems()),
                        .dst_offset = @intCast(node.storage_offset),
                        .dst_stride = @intCast(if (node.src1) |s| s.strides[0] else 1),
                        .src_offset = @intCast(node.src0.?.storage_offset),
                        .src_stride = @intCast(node.src0.?.strides[0]),
                    } });
                },
                // Structural ops are filtered at function entry; matmul handled above.
                // Scatter/gather and fused composite ops not yet supported on GPU.
                .none, .view, .reshape, .transpose, .permute, .as_strided, .broadcast_to,
                .matmul, .softmax, .attention,
                .gather_rows, .scatter_add_rows, .pick_rows, .scatter_add_picks, .scatter_add_view,
                .rope, .slice_assign_rows => {},
            }
        }
    };
}
