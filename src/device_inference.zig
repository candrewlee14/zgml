//! Device-accelerated inference wrapper.
//!
//! Compiles a ComputeGraph into a DeviceProgram, executes it per token
//! via the backend. Model-agnostic — works with any graph (GPT, LLaMA, etc.).
//!
//! ```
//! var metal = try MetalBackend.init();
//! var device = try DeviceInference(f32).init(.{
//!     .graph = &plan.graph,
//!     .be = metal.backend(),
//!     .alloc = alloc,
//!     .input_tensors = &.{ plan.token_input, plan.attn_mask },
//!     .output_tensor = plan.logits,
//!     .output_host_buf = logits_buf.ptr,
//!     .output_len = vocab_size,
//! });
//! defer device.deinit();
//!
//! // Caller patches tensor data each step, then:
//! device.patchSliceAssignOffset(pos);
//! device.execute();
//! ```

const std = @import("std");
const backend_mod = @import("backend.zig");

const QuantizedWeight = @import("quant.zig").QuantizedWeight;

pub fn DeviceInference(comptime T: type) type {
    const Tensor = @import("tensor.zig").Tensor(T);
    const ComputeGraph = @import("graph.zig").ComputeGraph(T);

    return struct {
        const Self = @This();

        be: backend_mod.Backend,
        alloc: std.mem.Allocator,
        compiled: backend_mod.Backend.CompiledHandle,
        program_ops: []backend_mod.DeviceOp,
        n_buffers: u16,
        buffer_sizes: []const usize,
        step_inputs: []backend_mod.ProgramIO,
        step_outputs: []backend_mod.ProgramIO,
        slice_assign_op_indices: []u32,
        attention_op_indices: []u32,

        pub const InitOptions = struct {
            graph: *ComputeGraph,
            be: backend_mod.Backend,
            alloc: std.mem.Allocator,
            input_tensors: []const *const Tensor,
            output_tensor: *const Tensor,
            output_host_buf: [*]T,
            output_len: usize,
            quant_weights: []const QuantizedWeight(T) = &.{},
            quant_map: *const std.AutoHashMapUnmanaged(*Tensor, usize) = &empty_quant_map,
        };

        const empty_quant_map: std.AutoHashMapUnmanaged(*Tensor, usize) = .empty;

        pub fn init(opts: InitOptions) !Self {
            const graph = opts.graph;
            const alloc = opts.alloc;
            const nodes = graph.nodes.items[0..graph.forward_node_count];
            const steps = graph.forward_execution_steps.items;

            // ── Assign buffer indices to unique host data pointers ────
            var ptr_to_idx = std.AutoHashMap([*]T, u16).init(alloc);
            defer ptr_to_idx.deinit();
            var buf_sizes: std.ArrayListUnmanaged(usize) = .empty;
            defer buf_sizes.deinit(alloc);
            var uploads_list: std.ArrayListUnmanaged(backend_mod.ProgramIO) = .empty;
            defer uploads_list.deinit(alloc);

            for (nodes) |node| {
                const tensors = [_]?*const Tensor{ node, node.src0, node.src1, node.src2, node.src3 };
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
                            const fp = graph.fused_chains.items[idx];
                            switch (fp.kind()) {
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
                                    const chain = fp.payload.elementwise_chain;
                                    const ew_steps = try alloc.alloc(backend_mod.FusedEwStep, chain.nodes.len);
                                    for (chain.nodes, 0..) |node, k| {
                                        const node_op = node.opTag();
                                        const role = chain.otherOperandRole(k);
                                        const is_swapped = (role == .src0);
                                        var secondary_buf: u16 = 0;
                                        var secondary_offset: u32 = 0;
                                        if (node_op.isBinary()) {
                                            const other = if (is_swapped) node.src0.? else node.src1.?;
                                            secondary_buf = bufIdx(&ptr_to_idx, other);
                                            secondary_offset = @intCast(other.storage_offset);
                                        }
                                        ew_steps[k] = .{
                                            .op = node_op,
                                            .is_swapped = is_swapped,
                                            .secondary_buf = secondary_buf,
                                            .secondary_offset = secondary_offset,
                                        };
                                    }
                                    const out_node = chain.nodes[chain.nodes.len - 1];
                                    try ops_list.append(alloc, .{ .fused_elementwise = .{
                                        .steps = ew_steps,
                                        .n = @intCast(out_node.nElems()),
                                        .dst = bufIdx(&ptr_to_idx, out_node),
                                        .src = bufIdx(&ptr_to_idx, chain.input),
                                        .dst_offset = @intCast(out_node.storage_offset),
                                        .src_offset = @intCast(chain.input.storage_offset),
                                    } });
                                },
                                .conv2d, .conv2d_bwd_input, .conv2d_bwd_kernel,
                                .max_pool2d, .max_pool2d_bwd,
                                .log_softmax, .cross_entropy => {},
                            }
                        },
                        .node => |node_ptr| try appendNodeOp(opts.quant_map, &ops_list, &ptr_to_idx, node_ptr, alloc),
                    }
                }
            } else {
                for (nodes) |node| try appendNodeOp(opts.quant_map, &ops_list, &ptr_to_idx, node, alloc);
            }

            // ── Per-step I/O ──────────────────────────────────────────
            const inputs = try alloc.alloc(backend_mod.ProgramIO, opts.input_tensors.len);
            for (opts.input_tensors, 0..) |t, i| {
                inputs[i] = .{
                    .buf_idx = bufIdx(&ptr_to_idx, t),
                    .host_ptr = @ptrCast(t.data.ptr),
                    .size = @intCast(t.data.len * @sizeOf(T)),
                };
            }

            const outputs = try alloc.alloc(backend_mod.ProgramIO, 1);
            outputs[0] = .{
                .buf_idx = bufIdx(&ptr_to_idx, opts.output_tensor),
                .host_ptr = @ptrCast(opts.output_host_buf),
                .size = @intCast(opts.output_len * @sizeOf(T)),
            };

            // ── Quantized weights ─────────────────────────────────────
            const qw_uploads = try alloc.alloc(backend_mod.QuantizedWeightUpload, opts.quant_weights.len);
            defer alloc.free(qw_uploads);
            for (opts.quant_weights, 0..) |qw, i| {
                qw_uploads[i] = .{ .data = qw.data, .scales = qw.scales, .rows = qw.rows, .cols = qw.cols, .block_size = qw.block_size };
            }

            // ── Op index tracking ─────────────────────────────────────
            var sa_list: std.ArrayListUnmanaged(u32) = .empty;
            defer sa_list.deinit(alloc);
            var attn_list: std.ArrayListUnmanaged(u32) = .empty;
            defer attn_list.deinit(alloc);
            for (ops_list.items, 0..) |dop, i| {
                switch (dop) {
                    .slice_assign => try sa_list.append(alloc, @intCast(i)),
                    .attention => try attn_list.append(alloc, @intCast(i)),
                    else => {},
                }
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

            const compiled = opts.be.compileProgram(program) orelse return error.CompileFailed;

            const owned_buf_sizes = try alloc.dupe(usize, buf_sizes.items);

            return .{
                .be = opts.be,
                .alloc = alloc,
                .compiled = compiled,
                .program_ops = owned_ops,
                .n_buffers = @intCast(buf_sizes.items.len),
                .buffer_sizes = owned_buf_sizes,
                .step_inputs = inputs,
                .step_outputs = outputs,
                .slice_assign_op_indices = try sa_list.toOwnedSlice(alloc),
                .attention_op_indices = try attn_list.toOwnedSlice(alloc),
            };
        }

        /// Patch all slice_assign destination offsets (KV-cache write positions).
        pub fn patchSliceAssignOffset(self: *Self, pos: u32) void {
            for (self.slice_assign_op_indices) |idx| {
                self.program_ops[idx].slice_assign.dst_offset = pos;
            }
        }

        /// Patch attention seq_kv to the actual valid KV length (pos + 1).
        /// Avoids scanning masked positions in the attention kernel.
        pub fn patchAttentionSeqKV(self: *Self, seq_kv: u32) void {
            for (self.attention_op_indices) |idx| {
                self.program_ops[idx].attention.seq_kv = seq_kv;
            }
        }

        /// Execute the compiled program. Caller must have already patched
        /// input tensor data and slice_assign offsets before calling.
        pub fn execute(self: *Self) void {
            self.be.executeProgram(self.compiled, self.step_inputs, self.step_outputs);
        }

        /// Return pointer to accumulated runtime profile, or null if unsupported.
        pub fn getRuntimeProfile(self: *const Self) ?*@import("profile.zig").RuntimeProfile {
            return self.be.getRuntimeProfile(self.compiled);
        }

        /// Return a DeviceProgram descriptor for profiling.
        pub fn getProgram(self: *const Self) backend_mod.DeviceProgram {
            return .{
                .ops = self.program_ops,
                .n_buffers = self.n_buffers,
                .buffer_sizes = self.buffer_sizes,
                .initial_uploads = &.{},
            };
        }

        pub fn deinit(self: *Self) void {
            self.be.freeProgram(self.compiled);
            self.alloc.free(self.program_ops);
            self.alloc.free(self.buffer_sizes);
            self.alloc.free(self.step_inputs);
            self.alloc.free(self.step_outputs);
            if (self.slice_assign_op_indices.len > 0) self.alloc.free(self.slice_assign_op_indices);
            if (self.attention_op_indices.len > 0) self.alloc.free(self.attention_op_indices);
        }

        // ── Helpers ──────────────────────────────────────────────

        fn bufIdx(map: *const std.AutoHashMap([*]T, u16), tensor: *const Tensor) u16 {
            return map.get(tensor.data.ptr).?;
        }

        fn appendNodeOp(quant_map: *const std.AutoHashMapUnmanaged(*Tensor, usize), ops: *std.ArrayListUnmanaged(backend_mod.DeviceOp), ptr_to_idx: *const std.AutoHashMap([*]T, u16), node: *Tensor, alloc: std.mem.Allocator) !void {
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

                if (quant_map.get(node)) |qi| {
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
                .slice_assign_rows => {
                    // For seq_q=1 (decode), this is a contiguous copy of src_rows elements.
                    const src = node.src0.?;
                    try ops.append(alloc, .{ .slice_assign = .{
                        .dst = if (node.src1) |s| bufIdx(ptr_to_idx, s) else dst_idx,
                        .src = src0_idx,
                        .n = @intCast(src.nElems()),
                        .dst_offset = @intCast(node.storage_offset),
                        .dst_stride = 1,
                        .src_offset = @intCast(src.storage_offset),
                        .src_stride = 1,
                    } });
                },
                .rope => {
                    const src = node.src0.?;
                    const cs = node.src1.?;
                    const d = src.ne[0];
                    try ops.append(alloc, .{ .rope = .{
                        .dst = dst_idx,
                        .src = src0_idx,
                        .cos_sin = src1_idx,
                        .half_d = @intCast(d / 2),
                        .seq_len = @intCast(src.ne[1]),
                        .src_off = @intCast(src.storage_offset),
                        .cs_off = @intCast(cs.storage_offset),
                        .dst_off = @intCast(node.storage_offset),
                        .src_rs = @intCast(src.strides[0]),
                        .src_cs = @intCast(src.strides[1]),
                        .cs_cs = @intCast(cs.strides[1]),
                    } });
                },
                .attention => {
                    const q = node.src0.?;
                    const k = node.src1.?;
                    const v = node.src2.?;
                    const mask = node.src3;
                    try ops.append(alloc, .{ .attention = .{
                        .dst = dst_idx,
                        .q = src0_idx,
                        .k = src1_idx,
                        .v = bufIdx(ptr_to_idx, v),
                        .mask = if (mask) |m| bufIdx(ptr_to_idx, m) else dst_idx,
                        .d_head = @intCast(q.ne[0]),
                        .seq_kv = @intCast(k.ne[1]),
                        .scale = node.op_scale,
                        .q_off = @intCast(q.storage_offset),
                        .k_off = @intCast(k.storage_offset),
                        .v_off = @intCast(v.storage_offset),
                        .mask_off = if (mask) |m| @intCast(m.storage_offset) else 0,
                        .dst_off = @intCast(node.storage_offset),
                        .k_rs = @intCast(k.strides[0]),
                        .k_cs = @intCast(k.strides[1]),
                        .v_rs = @intCast(v.strides[0]),
                        .v_cs = @intCast(v.strides[1]),
                    } });
                },
                // Structural ops filtered at entry; matmul handled above.
                // Scatter/gather and fused composite ops not yet on GPU.
                .none, .view, .reshape, .transpose, .permute, .as_strided, .broadcast_to,
                .matmul, .softmax,
                .gather_rows, .scatter_add_rows, .pick_rows, .scatter_add_picks, .scatter_add_view => {},
            }
        }
    };
}
