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

            // ── Assign buffer indices to alias-aware base storages ────
            var buffers = BufferMap.init(alloc);
            defer buffers.deinit();

            for (nodes) |node| {
                const tensors = [_]?*const Tensor{ node, node.src0, node.src1, node.src2, node.src3 };
                for (tensors) |maybe_t| {
                    const t = maybe_t orelse continue;
                    try buffers.ensure(t);
                }
            }
            for (opts.input_tensors) |t| try buffers.ensure(t);
            try buffers.ensure(opts.output_tensor);

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
                                    if (!ln.input.isDenseLayout() or !ln.output.isDenseLayout()) return error.UnsupportedDeviceOp;
                                    try ops_list.append(alloc, .{ .layernorm = .{
                                        .dst = buffers.idx(ln.output),
                                        .src = buffers.idx(ln.input),
                                        .rows = @intCast(ln.input.ne[1]),
                                        .cols = @intCast(ln.input.ne[0]),
                                        .eps = ln.eps_like.data[0],
                                        .src_offset = @intCast(buffers.offset(ln.input)),
                                        .dst_offset = @intCast(buffers.offset(ln.output)),
                                    } });
                                },
                                .elementwise_chain => {
                                    const chain = fp.payload.elementwise_chain;
                                    if (opts.be.device_type == .wgpu) {
                                        try appendElementwiseChainOps(&ops_list, &buffers, chain, alloc);
                                    } else {
                                        const out_node = chain.nodes[chain.nodes.len - 1];
                                        if (!chain.input.isDenseLayout() or !out_node.isDenseLayout()) return error.UnsupportedDeviceOp;
                                        const ew_steps = try alloc.alloc(backend_mod.FusedEwStep, chain.nodes.len);
                                        errdefer alloc.free(ew_steps);
                                        for (chain.nodes, 0..) |node, k| {
                                            const node_op = node.opTag();
                                            const role = chain.otherOperandRole(k);
                                            const is_swapped = (role == .src0);
                                            var secondary_buf: u16 = 0;
                                            var secondary_offset: u32 = 0;
                                            if (node_op.isBinary()) {
                                                const other = if (is_swapped) node.src0.? else node.src1.?;
                                                if (!other.isDenseLayout()) return error.UnsupportedDeviceOp;
                                                secondary_buf = buffers.idx(other);
                                                secondary_offset = @intCast(buffers.offset(other));
                                            }
                                            ew_steps[k] = .{
                                                .op = node_op,
                                                .is_swapped = is_swapped,
                                                .secondary_buf = secondary_buf,
                                                .secondary_offset = secondary_offset,
                                            };
                                        }
                                        try ops_list.append(alloc, .{ .fused_elementwise = .{
                                            .steps = ew_steps,
                                            .n = @intCast(out_node.nElems()),
                                            .dst = buffers.idx(out_node),
                                            .src = buffers.idx(chain.input),
                                            .dst_offset = @intCast(buffers.offset(out_node)),
                                            .src_offset = @intCast(buffers.offset(chain.input)),
                                        } });
                                    }
                                },
                                .conv2d, .conv2d_bwd_input, .conv2d_bwd_kernel, .max_pool2d, .max_pool2d_bwd, .log_softmax, .cross_entropy => {},
                            }
                        },
                        .node => |node_ptr| try appendNodeOp(opts.quant_map, &ops_list, &buffers, node_ptr, alloc),
                    }
                }
            } else {
                for (nodes) |node| try appendNodeOp(opts.quant_map, &ops_list, &buffers, node, alloc);
            }

            // ── Per-step I/O ──────────────────────────────────────────
            const inputs = try alloc.alloc(backend_mod.ProgramIO, opts.input_tensors.len);
            for (opts.input_tensors, 0..) |t, i| {
                inputs[i] = .{
                    .buf_idx = buffers.idx(t),
                    .offset = @intCast(buffers.offset(t) * @sizeOf(T)),
                    .host_ptr = @ptrCast(t.data.ptr),
                    .size = @intCast(t.data.len * @sizeOf(T)),
                };
            }

            const outputs = try alloc.alloc(backend_mod.ProgramIO, 1);
            outputs[0] = .{
                .buf_idx = buffers.idx(opts.output_tensor),
                .offset = @intCast(buffers.offset(opts.output_tensor) * @sizeOf(T)),
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
                .n_buffers = @intCast(buffers.buf_sizes.items.len),
                .buffer_sizes = buffers.buf_sizes.items,
                .initial_uploads = buffers.uploads.items,
                .qweights = qw_uploads,
            };

            const compiled = opts.be.compileProgram(program) orelse return error.CompileFailed;

            const owned_buf_sizes = try alloc.dupe(usize, buffers.buf_sizes.items);

            return .{
                .be = opts.be,
                .alloc = alloc,
                .compiled = compiled,
                .program_ops = owned_ops,
                .n_buffers = @intCast(buffers.buf_sizes.items.len),
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
                const sa = &self.program_ops[idx].slice_assign;
                sa.dst_offset = sa.dst_base_offset + pos * sa.patch_stride;
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
            self.be.refreshProgram(self.compiled, self.program_ops);
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
            for (self.program_ops) |op| {
                switch (op) {
                    .fused_elementwise => |fe| if (fe.steps.len > 0) self.alloc.free(fe.steps),
                    else => {},
                }
            }
            self.alloc.free(self.program_ops);
            self.alloc.free(self.buffer_sizes);
            self.alloc.free(self.step_inputs);
            self.alloc.free(self.step_outputs);
            if (self.slice_assign_op_indices.len > 0) self.alloc.free(self.slice_assign_op_indices);
            if (self.attention_op_indices.len > 0) self.alloc.free(self.attention_op_indices);
        }

        // ── Helpers ──────────────────────────────────────────────

        const BufferRef = struct {
            idx: u16,
            offset: usize,
        };

        const StorageRef = struct {
            ptr: [*]T,
            len: usize,
            offset: usize,
            base: *const Tensor,
        };

        const BufferMap = struct {
            alloc: std.mem.Allocator,
            ptr_to_idx: std.AutoHashMap([*]T, u16),
            tensor_to_ref: std.AutoHashMap(*const Tensor, BufferRef),
            buf_sizes: std.ArrayListUnmanaged(usize) = .empty,
            uploads: std.ArrayListUnmanaged(backend_mod.ProgramIO) = .empty,

            fn init(alloc: std.mem.Allocator) BufferMap {
                return .{
                    .alloc = alloc,
                    .ptr_to_idx = std.AutoHashMap([*]T, u16).init(alloc),
                    .tensor_to_ref = std.AutoHashMap(*const Tensor, BufferRef).init(alloc),
                };
            }

            fn deinit(self: *BufferMap) void {
                self.ptr_to_idx.deinit();
                self.tensor_to_ref.deinit();
                self.buf_sizes.deinit(self.alloc);
                self.uploads.deinit(self.alloc);
            }

            fn idx(self: *const BufferMap, tensor: *const Tensor) u16 {
                return self.tensor_to_ref.get(tensor).?.idx;
            }

            fn offset(self: *const BufferMap, tensor: *const Tensor) usize {
                return self.tensor_to_ref.get(tensor).?.offset;
            }

            fn ensure(self: *BufferMap, tensor: *const Tensor) !void {
                if (self.tensor_to_ref.contains(tensor)) return;
                const sr = storageRef(tensor);
                const entry = try self.ptr_to_idx.getOrPut(sr.ptr);
                if (!entry.found_existing) {
                    entry.value_ptr.* = @intCast(self.buf_sizes.items.len);
                    try self.buf_sizes.append(self.alloc, @max(sr.len, 1));
                    if (sr.base.opTag() == .none and sr.base.data.len > 0) {
                        try self.uploads.append(self.alloc, .{
                            .buf_idx = entry.value_ptr.*,
                            .host_ptr = @ptrCast(sr.base.data.ptr),
                            .size = @intCast(sr.base.data.len * @sizeOf(T)),
                        });
                    }
                } else {
                    self.buf_sizes.items[entry.value_ptr.*] = @max(self.buf_sizes.items[entry.value_ptr.*], @max(sr.len, 1));
                }
                try self.tensor_to_ref.put(tensor, .{ .idx = entry.value_ptr.*, .offset = sr.offset });
            }
        };

        fn storageBase(tensor: *const Tensor) *const Tensor {
            return switch (tensor.opTag()) {
                .view, .reshape, .transpose, .permute, .as_strided, .broadcast_to => storageBase(tensor.src0.?),
                .slice_assign, .slice_assign_rows => storageBase(tensor.src1.?),
                else => tensor,
            };
        }

        fn appendElementwiseChainOps(
            ops: *std.ArrayListUnmanaged(backend_mod.DeviceOp),
            buffers: *BufferMap,
            chain: @import("tensor/fused.zig").ElementwiseFusionPlan(T),
            alloc: std.mem.Allocator,
        ) !void {
            for (chain.nodes) |node| {
                if (!node.isDenseLayout()) return error.UnsupportedDeviceOp;
                const src0 = node.src0.?;
                if (!src0.isDenseLayout()) return error.UnsupportedDeviceOp;
                const src1 = node.src1;
                if (src1) |other| if (!other.isDenseLayout()) return error.UnsupportedDeviceOp;

                try ops.append(alloc, .{ .elementwise = .{
                    .op = node.opTag(),
                    .dst = buffers.idx(node),
                    .src0 = buffers.idx(src0),
                    .src1 = if (src1) |other| buffers.idx(other) else buffers.idx(src0),
                    .n = @intCast(node.nElems()),
                    .dst_offset = @intCast(buffers.offset(node)),
                    .src0_offset = @intCast(buffers.offset(src0)),
                    .src1_offset = if (src1) |other| @intCast(buffers.offset(other)) else @intCast(buffers.offset(src0)),
                } });
            }
        }

        fn storageRef(tensor: *const Tensor) StorageRef {
            if (tensor.opTag() == .slice_assign or tensor.opTag() == .slice_assign_rows) {
                return storageRef(tensor.src1.?);
            }
            const base = storageBase(tensor);
            const base_addr = @intFromPtr(base.data.ptr);
            const tensor_addr = @intFromPtr(tensor.data.ptr);
            std.debug.assert(tensor_addr >= base_addr);
            const ptr_delta = (tensor_addr - base_addr) / @sizeOf(T);
            return .{
                .ptr = base.data.ptr,
                .len = base.data.len,
                .offset = ptr_delta + tensor.storage_offset,
                .base = base,
            };
        }

        fn isCanonicalRowSoftmax(node: *Tensor) bool {
            const src = node.src0 orelse return false;
            if (!node.isSameShape(src)) return false;
            if (!node.isDenseLayout() or !src.isDenseLayout()) return false;
            if (node.reduce_ne[0] != 1) return false;

            var dim: usize = 1;
            while (dim < src.n_dims) : (dim += 1) {
                if (node.reduce_ne[dim] != src.ne[dim]) return false;
            }
            while (dim < @import("tensor.zig").max_dims) : (dim += 1) {
                if (node.reduce_ne[dim] != 1) return false;
            }
            return true;
        }

        fn shape4(tensor: *const Tensor) [4]u32 {
            return .{
                @intCast(tensor.ne[0]),
                @intCast(tensor.ne[1]),
                @intCast(tensor.ne[2]),
                @intCast(tensor.ne[3]),
            };
        }

        fn strides4(tensor: *const Tensor) [4]u32 {
            return .{
                @intCast(tensor.strides[0]),
                @intCast(tensor.strides[1]),
                @intCast(tensor.strides[2]),
                @intCast(tensor.strides[3]),
            };
        }

        fn appendMatmulOp(
            quant_map: *const std.AutoHashMapUnmanaged(*Tensor, usize),
            ops: *std.ArrayListUnmanaged(backend_mod.DeviceOp),
            buffers: *const BufferMap,
            node: *Tensor,
            alloc: std.mem.Allocator,
        ) !void {
            const s0 = node.src0.?;
            const s1 = node.src1.?;
            const flags = node.matmul_flags;
            const M = if (flags.trans0) s0.ne[0] else s0.ne[1];
            const N = if (flags.trans1) s1.ne[1] else s1.ne[0];
            const K = if (flags.trans0) s0.ne[1] else s0.ne[0];

            if (quant_map.get(node)) |qi| {
                const input_tensor = if (s1.isParam()) s0 else s1;
                if (!input_tensor.isDenseLayout() or !node.isDenseLayout()) return error.UnsupportedDeviceOp;
                if (buffers.offset(input_tensor) != 0 or buffers.offset(node) != 0) return error.UnsupportedDeviceOp;
                try ops.append(alloc, .{ .qmatmul = .{
                    .dst = buffers.idx(node),
                    .input = buffers.idx(input_tensor),
                    .weight_idx = @intCast(qi),
                    .M = @intCast(M),
                    .N = @intCast(N),
                    .K = @intCast(K),
                } });
                return;
            }

            try ops.append(alloc, .{ .matmul = .{
                .dst = buffers.idx(node),
                .a = buffers.idx(s0),
                .b = buffers.idx(s1),
                .geom = .{
                    .M = M,
                    .N = N,
                    .K = K,
                    .a_row_stride = if (flags.trans0) s0.strides[0] else s0.strides[1],
                    .a_col_stride = if (flags.trans0) s0.strides[1] else s0.strides[0],
                    .b_row_stride = if (flags.trans1) s1.strides[0] else s1.strides[1],
                    .b_col_stride = if (flags.trans1) s1.strides[1] else s1.strides[0],
                    .a_offset = buffers.offset(s0),
                    .b_offset = buffers.offset(s1),
                    .dst_offset = buffers.offset(node),
                    .dst_row_stride = N,
                },
            } });
        }

        fn elementwiseDeviceOp(buffers: *const BufferMap, node: *Tensor, dst_idx: u16, src0_idx: u16, src1_idx: u16) backend_mod.DeviceOp {
            return .{ .elementwise = .{
                .op = node.opTag(),
                .dst = dst_idx,
                .src0 = src0_idx,
                .src1 = src1_idx,
                .n = @intCast(node.nElems()),
                .dst_offset = @intCast(buffers.offset(node)),
                .src0_offset = if (node.src0) |s| @intCast(buffers.offset(s)) else 0,
                .src1_offset = if (node.src1) |s| @intCast(buffers.offset(s)) else 0,
            } };
        }

        fn reduceDeviceOp(buffers: *const BufferMap, node: *Tensor, src: *const Tensor, dst_idx: u16, src_idx: u16) backend_mod.DeviceOp {
            return .{ .reduce = .{
                .op = node.opTag(),
                .dst = dst_idx,
                .src = src_idx,
                .n_out = @intCast(node.nElems()),
                .reduce_size = @intCast(src.ne[0]),
                .src_offset = @intCast(buffers.offset(src)),
                .dst_offset = @intCast(buffers.offset(node)),
            } };
        }

        fn repeatDeviceOp(buffers: *const BufferMap, node: *Tensor, src: *const Tensor, dst_idx: u16, src_idx: u16) backend_mod.DeviceOp {
            return .{ .repeat = .{
                .dst = dst_idx,
                .src = src_idx,
                .n = @intCast(node.nElems()),
                .src_ne = shape4(src),
                .dst_ne = shape4(node),
                .src_strides = strides4(src),
                .dst_strides = strides4(node),
                .src_offset = @intCast(buffers.offset(src)),
                .dst_offset = @intCast(buffers.offset(node)),
            } };
        }

        fn softmaxDeviceOp(buffers: *const BufferMap, node: *Tensor, src: *const Tensor, dst_idx: u16, src_idx: u16) backend_mod.DeviceOp {
            return .{ .softmax = .{
                .dst = dst_idx,
                .src = src_idx,
                .rows = @intCast(src.nElems() / src.ne[0]),
                .cols = @intCast(src.ne[0]),
                .src_offset = @intCast(buffers.offset(src)),
                .dst_offset = @intCast(buffers.offset(node)),
            } };
        }

        fn rmsnormDeviceOp(buffers: *const BufferMap, node: *Tensor, src: *const Tensor, dst_idx: u16, src_idx: u16) backend_mod.DeviceOp {
            return .{ .rmsnorm = .{
                .dst = dst_idx,
                .src = src_idx,
                .rows = @intCast(src.ne[1]),
                .cols = @intCast(src.ne[0]),
                .eps = node.op_eps,
                .src_offset = @intCast(buffers.offset(src)),
                .dst_offset = @intCast(buffers.offset(node)),
            } };
        }

        fn sliceAssignDeviceOp(
            buffers: *const BufferMap,
            src: *const Tensor,
            dst: *const Tensor,
            src_idx: u16,
            rows: usize,
            cols: usize,
            dst_base_offset: usize,
            dst_offset: usize,
            patch_stride: usize,
        ) backend_mod.DeviceOp {
            return .{ .slice_assign = .{
                .dst = buffers.idx(dst),
                .src = src_idx,
                .rows = @intCast(rows),
                .cols = @intCast(cols),
                .dst_base_offset = @intCast(dst_base_offset),
                .dst_offset = @intCast(dst_offset),
                .dst_row_stride = @intCast(dst.strides[0]),
                .dst_col_stride = @intCast(dst.strides[1]),
                .src_offset = @intCast(buffers.offset(src)),
                .src_row_stride = @intCast(src.strides[0]),
                .src_col_stride = @intCast(src.strides[1]),
                .patch_stride = @intCast(patch_stride),
            } };
        }

        fn ropeDeviceOp(
            buffers: *const BufferMap,
            node: *Tensor,
            src: *const Tensor,
            cs: *const Tensor,
            dst_idx: u16,
            src_idx: u16,
            cos_sin_idx: u16,
        ) backend_mod.DeviceOp {
            return .{ .rope = .{
                .dst = dst_idx,
                .src = src_idx,
                .cos_sin = cos_sin_idx,
                .half_d = @intCast(src.ne[0] / 2),
                .seq_len = @intCast(src.ne[1]),
                .src_off = @intCast(buffers.offset(src)),
                .cs_off = @intCast(buffers.offset(cs)),
                .dst_off = @intCast(buffers.offset(node)),
                .src_rs = @intCast(src.strides[0]),
                .src_cs = @intCast(src.strides[1]),
                .cs_cs = @intCast(cs.strides[1]),
            } };
        }

        fn attentionDeviceOp(
            buffers: *const BufferMap,
            node: *Tensor,
            q: *const Tensor,
            k: *const Tensor,
            v: *const Tensor,
            mask: ?*const Tensor,
            dst_idx: u16,
            q_idx: u16,
            k_idx: u16,
        ) backend_mod.DeviceOp {
            return .{ .attention = .{
                .dst = dst_idx,
                .q = q_idx,
                .k = k_idx,
                .v = buffers.idx(v),
                .mask = if (mask) |m| buffers.idx(m) else dst_idx,
                .has_mask = mask != null,
                .d_head = @intCast(q.ne[0]),
                .seq_q = @intCast(q.ne[1]),
                .seq_kv = @intCast(k.ne[1]),
                .scale = node.op_scale,
                .q_off = @intCast(buffers.offset(q)),
                .k_off = @intCast(buffers.offset(k)),
                .v_off = @intCast(buffers.offset(v)),
                .mask_off = if (mask) |m| @intCast(buffers.offset(m)) else 0,
                .dst_off = @intCast(buffers.offset(node)),
                .q_rs = @intCast(q.strides[0]),
                .q_cs = @intCast(q.strides[1]),
                .k_rs = @intCast(k.strides[0]),
                .k_cs = @intCast(k.strides[1]),
                .v_rs = @intCast(v.strides[0]),
                .v_cs = @intCast(v.strides[1]),
                .mask_rs = if (mask) |m| @intCast(m.strides[0]) else 0,
                .mask_cs = if (mask) |m| @intCast(m.strides[1]) else 0,
                .dst_rs = @intCast(node.strides[0]),
                .dst_cs = @intCast(node.strides[1]),
            } };
        }

        fn appendNodeOp(quant_map: *const std.AutoHashMapUnmanaged(*Tensor, usize), ops: *std.ArrayListUnmanaged(backend_mod.DeviceOp), buffers: *const BufferMap, node: *Tensor, alloc: std.mem.Allocator) !void {
            const op = node.opTag();
            if (op == .none or op == .view or op == .as_strided or op == .reshape or
                op == .transpose or op == .permute or op == .broadcast_to) return;

            if (op == .matmul) {
                try appendMatmulOp(quant_map, ops, buffers, node, alloc);
                return;
            }

            const dst_idx = buffers.idx(node);
            const src0_idx = if (node.src0) |s| buffers.idx(s) else dst_idx;
            const src1_idx = if (node.src1) |s| buffers.idx(s) else dst_idx;

            switch (op) {
                .add, .mul, .neg, .abs, .sgn, .step, .relu, .sqrt, .recip, .exp, .log, .gelu => {
                    if (!node.isDenseLayout()) return error.UnsupportedDeviceOp;
                    if (node.src0) |s| if (!s.isDenseLayout()) return error.UnsupportedDeviceOp;
                    if (node.src1) |s| if (!s.isDenseLayout()) return error.UnsupportedDeviceOp;
                    try ops.append(alloc, elementwiseDeviceOp(buffers, node, dst_idx, src0_idx, src1_idx));
                },
                .sum, .max => {
                    const src = node.src0.?;
                    if (!src.isDenseLayout() or !node.isDenseLayout()) return error.UnsupportedDeviceOp;
                    try ops.append(alloc, reduceDeviceOp(buffers, node, src, dst_idx, src0_idx));
                },
                .repeat => {
                    const src = node.src0.?;
                    try ops.append(alloc, repeatDeviceOp(buffers, node, src, dst_idx, src0_idx));
                },
                .softmax => {
                    if (!isCanonicalRowSoftmax(node)) return error.UnsupportedDeviceOp;
                    const src = node.src0.?;
                    try ops.append(alloc, softmaxDeviceOp(buffers, node, src, dst_idx, src0_idx));
                },
                .rmsnorm => {
                    const src = node.src0.?;
                    if (!src.isDenseLayout() or !node.isDenseLayout()) return error.UnsupportedDeviceOp;
                    try ops.append(alloc, rmsnormDeviceOp(buffers, node, src, dst_idx, src0_idx));
                },
                .slice_assign => {
                    const src = node.src0.?;
                    const dst = node.src1 orelse node;
                    const rows = dst.ne[0];
                    const cols = if (src.n_dims >= 2) src.ne[1] else 1;
                    const base = buffers.offset(dst);
                    const dst_offset = base + node.storage_offset * dst.strides[1];
                    try ops.append(alloc, sliceAssignDeviceOp(buffers, src, dst, src0_idx, rows, cols, base, dst_offset, dst.strides[1]));
                },
                .slice_assign_rows => {
                    const src = node.src0.?;
                    const dst = node.src1 orelse node;
                    const base = buffers.offset(dst);
                    const dst_offset = base + node.storage_offset * dst.strides[0];
                    try ops.append(alloc, sliceAssignDeviceOp(buffers, src, dst, src0_idx, src.ne[0], src.ne[1], base, dst_offset, dst.strides[0]));
                },
                .rope => {
                    const src = node.src0.?;
                    const cs = node.src1.?;
                    try ops.append(alloc, ropeDeviceOp(buffers, node, src, cs, dst_idx, src0_idx, src1_idx));
                },
                .attention => {
                    const q = node.src0.?;
                    const k = node.src1.?;
                    const v = node.src2.?;
                    const mask = node.src3;
                    try ops.append(alloc, attentionDeviceOp(buffers, node, q, k, v, mask, dst_idx, src0_idx, src1_idx));
                },
                // Structural ops filtered at entry; matmul handled above.
                // Scatter/gather and fused composite ops not yet on device programs.
                .none, .view, .reshape, .transpose, .permute, .as_strided, .broadcast_to, .matmul, .gather_rows, .scatter_add_rows, .pick_rows, .scatter_add_picks, .scatter_add_view => return error.UnsupportedDeviceOp,
            }
        }
    };
}

const testing = std.testing;
const TensorF32 = @import("tensor.zig").Tensor(f32);
const ComputeGraphF32 = @import("graph.zig").ComputeGraph(f32);

const TestBackendState = struct {
    compile_calls: usize = 0,
    refresh_calls: usize = 0,
};

fn testDenseMatMul(_: *anyopaque, _: backend_mod.DenseMatMulSpecF32) bool {
    return false;
}

fn testCompile(ctx: *anyopaque, _: backend_mod.DeviceProgram) ?backend_mod.Backend.CompiledHandle {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    state.compile_calls += 1;
    return @ptrFromInt(1);
}

fn testExecute(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: []const backend_mod.ProgramIO, _: []const backend_mod.ProgramIO) void {}

fn testRefresh(ctx: *anyopaque, _: backend_mod.Backend.CompiledHandle, _: []const backend_mod.DeviceOp) void {
    const state: *TestBackendState = @ptrCast(@alignCast(ctx));
    state.refresh_calls += 1;
}

fn testFree(_: *anyopaque, _: backend_mod.Backend.CompiledHandle) void {}

fn testProfile(_: *anyopaque, _: backend_mod.Backend.CompiledHandle) ?*@import("profile.zig").RuntimeProfile {
    return null;
}

const test_vtable = backend_mod.Backend.VTable{
    .dense_matmul_f32 = testDenseMatMul,
    .compile_program = testCompile,
    .refresh_program = testRefresh,
    .execute_program = testExecute,
    .free_program = testFree,
    .get_runtime_profile = testProfile,
};

fn testBackend(state: *TestBackendState) backend_mod.Backend {
    return testBackendForDevice(state, .cpu);
}

fn testBackendForDevice(state: *TestBackendState, device_type: backend_mod.Device) backend_mod.Backend {
    return .{
        .ctx = state,
        .vtable = &test_vtable,
        .name_str = "test",
        .device_type = device_type,
    };
}

test "DeviceInference lowers canonical softmax" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 4, 3 });
    for (x.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.125;
    const y = x.softmax(&.{ 1, 3 });
    try graph.infer(y);

    var out = [_]f32{0} ** 12;
    var state = TestBackendState{};
    var dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer dev.deinit();

    const program = dev.getProgram();
    try testing.expectEqual(@as(usize, 1), state.compile_calls);
    try testing.expectEqual(@as(usize, 1), program.ops.len);
    try testing.expectEqual(@as(u32, 3), program.ops[0].softmax.rows);
    try testing.expectEqual(@as(u32, 4), program.ops[0].softmax.cols);
}

test "DeviceInference aliases dense views to base buffers" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 4, 4 });
    for (x.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));
    const view = x.sliceColumns(1, 3);
    const y = view.relu();
    try graph.infer(y);

    var out = [_]f32{0} ** 8;
    var state = TestBackendState{};
    var dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer dev.deinit();

    const program = dev.getProgram();
    try testing.expectEqual(@as(usize, 1), program.ops.len);
    try testing.expectEqual(@as(u16, 2), program.n_buffers);
    try testing.expectEqual(dev.step_inputs[0].buf_idx, program.ops[0].elementwise.src0);
    try testing.expectEqual(@as(u32, 4), program.ops[0].elementwise.src0_offset);
    try testing.expectEqual(@as(u32, 0), program.ops[0].elementwise.dst_offset);
}

test "DeviceInference lowers prefill slice_assign with 2D strides" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const cache = try TensorF32.init(a, &.{ 4, 8 });
    const src_full = try TensorF32.init(a, &.{ 12, 2 });
    const src = src_full.sliceRows(4, 8);
    const y = cache.sliceAssign(src, 3);
    try graph.infer(y);

    var out = [_]f32{0} ** 32;
    var state = TestBackendState{};
    var dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{src_full},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer dev.deinit();

    const program = dev.getProgram();
    try testing.expectEqual(@as(usize, 1), program.ops.len);
    const sa = program.ops[0].slice_assign;
    try testing.expectEqual(@as(u32, 4), sa.rows);
    try testing.expectEqual(@as(u32, 2), sa.cols);
    try testing.expectEqual(@as(u32, 12), sa.dst_offset);
    try testing.expectEqual(@as(u32, 1), sa.dst_row_stride);
    try testing.expectEqual(@as(u32, 4), sa.dst_col_stride);
    try testing.expectEqual(@as(u32, 4), sa.src_offset);
    try testing.expectEqual(@as(u32, 1), sa.src_row_stride);
    try testing.expectEqual(@as(u32, 12), sa.src_col_stride);

    dev.patchSliceAssignOffset(5);
    try testing.expectEqual(@as(u32, 20), dev.program_ops[0].slice_assign.dst_offset);
    dev.execute();
    try testing.expectEqual(@as(usize, 1), state.refresh_calls);
}

test "DeviceInference lowers batched attention geometry" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const q = try TensorF32.init(a, &.{ 4, 2 });
    const k_full = try TensorF32.init(a, &.{ 8, 8 });
    const v_full = try TensorF32.init(a, &.{ 8, 8 });
    const k = k_full.sliceRows(2, 6);
    const v = v_full.sliceRows(2, 6);
    const mask = try TensorF32.init(a, &.{ 8, 2 });
    const y = q.attention(k, v, mask, 0.5);
    try graph.infer(y);

    var out = [_]f32{0} ** 8;
    var state = TestBackendState{};
    var dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{ q, k_full, v_full, mask },
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer dev.deinit();

    const program = dev.getProgram();
    try testing.expectEqual(@as(usize, 1), program.ops.len);
    const att = program.ops[0].attention;
    try testing.expect(att.has_mask);
    try testing.expectEqual(@as(u32, 4), att.d_head);
    try testing.expectEqual(@as(u32, 2), att.seq_q);
    try testing.expectEqual(@as(u32, 8), att.seq_kv);
    try testing.expectEqual(@as(u32, 2), att.k_off);
    try testing.expectEqual(@as(u32, 1), att.k_rs);
    try testing.expectEqual(@as(u32, 8), att.k_cs);
    try testing.expectEqual(@as(u32, 8), att.mask_cs);

    dev.patchAttentionSeqKV(5);
    try testing.expectEqual(@as(u32, 5), dev.program_ops[0].attention.seq_kv);
}

test "DeviceInference lowers WGPU elementwise chains as primitive ops" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 4, 3 });
    for (x.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.25;
    const y = x.relu().sqrt();
    try graph.infer(y);

    var out = [_]f32{0} ** 12;
    var state = TestBackendState{};
    var dev = try DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackendForDevice(&state, .wgpu),
        .alloc = testing.allocator,
        .input_tensors = &.{x},
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    });
    defer dev.deinit();

    const program = dev.getProgram();
    try testing.expectEqual(@as(usize, 2), program.ops.len);
    try testing.expectEqual(backend_mod.Op.relu, program.ops[0].elementwise.op);
    try testing.expectEqual(backend_mod.Op.sqrt, program.ops[1].elementwise.op);
}

test "DeviceInference rejects unsupported graph ops" {
    var graph = ComputeGraphF32.init(testing.allocator);
    defer graph.deinit();
    const a = graph.allocator();

    const x = try TensorF32.init(a, &.{ 4, 3 });
    const idx = try TensorF32.initIndexVectorCopy(a, &.{ 2, 0 });
    const y = x.gatherRows(idx);
    try graph.infer(y);

    var out = [_]f32{0} ** 8;
    var state = TestBackendState{};
    try testing.expectError(error.UnsupportedDeviceOp, DeviceInference(f32).init(.{
        .graph = &graph,
        .be = testBackend(&state),
        .alloc = testing.allocator,
        .input_tensors = &.{ x, idx },
        .output_tensor = y,
        .output_host_buf = &out,
        .output_len = out.len,
    }));
    try testing.expectEqual(@as(usize, 0), state.compile_calls);
}
