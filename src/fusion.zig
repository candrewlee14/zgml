//! Unified fusion detection for the computation graph.
//!
//! Single pass over the node list: tries high-level patterns first (largest
//! subgraphs), then elementwise chains. Works directly on the tensor graph —
//! no intermediate IR. All results are `FusionPlan(T)` consumed by the
//! execution kernels in `tensor/fused.zig`.

const std = @import("std");
const Op = @import("op.zig").Op;
const Tensor = @import("tensor.zig").Tensor;
const fused = @import("tensor/fused.zig");

pub const FusionPlan = fused.FusionPlan;
pub const FusionPayload = fused.FusionPayload;
pub const FusionKind = fused.FusionKind;

pub fn FusionDetector(comptime T: type) type {
    return struct {
        nodes: []*Tensor(T),
        forward_node_count: usize,
        fused_skip: []bool,
        fused_chains: std.ArrayListUnmanaged(FusionPlan(T)),
        use_count: []u16,
        ptr_to_idx: std.AutoHashMapUnmanaged(*Tensor(T), usize),

        const Self = @This();

        pub fn init(alloc: std.mem.Allocator, nodes: []*Tensor(T), forward_node_count: usize) !Self {
            const n = nodes.len;
            const skip = try alloc.alloc(bool, n);
            @memset(skip, false);

            var map = std.AutoHashMapUnmanaged(*Tensor(T), usize){};
            try map.ensureTotalCapacity(alloc, @intCast(n));
            for (nodes, 0..) |node, i| map.putAssumeCapacity(node, i);

            const uc = try alloc.alloc(u16, n);
            @memset(uc, 0);
            for (nodes) |node| {
                if (node.source0()) |s| if (map.get(s)) |j| {
                    uc[j] += 1;
                };
                if (node.source1()) |s| if (map.get(s)) |j| {
                    uc[j] += 1;
                };
            }

            return .{
                .nodes = nodes,
                .forward_node_count = forward_node_count,
                .fused_skip = skip,
                .fused_chains = .{},
                .use_count = uc,
                .ptr_to_idx = map,
            };
        }

        pub fn deinit(self: *Self, alloc: std.mem.Allocator) void {
            self.ptr_to_idx.deinit(alloc);
        }

        // =================================================================
        // Main entry point
        // =================================================================

        pub fn detect(self: *Self, alloc: std.mem.Allocator) !void {
            // Forward high-level patterns: scan high-to-low (maximal munch).
            // The graph is topologically ordered, so composed patterns like
            // relu(bias(conv(x))) always have higher output indices than their
            // sub-patterns. Scanning in reverse ensures the widest match wins.
            {
                var idx: usize = self.forward_node_count;
                while (idx > 0) {
                    idx -= 1;
                    if (self.fused_skip[idx]) continue;
                    _ = try self.tryForwardPattern(alloc, self.nodes[idx], idx);
                }
            }

            // Backward patterns (conv2d, maxpool2d)
            for (self.nodes[self.forward_node_count..], self.forward_node_count..) |node, idx| {
                if (self.fused_skip[idx]) continue;
                if (self.detectConv2dBwd(node, idx)) |plan| {
                    try self.fused_chains.append(alloc, plan);
                    self.markBwdConvSkip(plan, idx);
                } else if (self.detectMaxPool2dBwd(node, idx)) |plan| {
                    try self.fused_chains.append(alloc, plan);
                    self.markBwdSkipChain(idx);
                }
            }

            // Elementwise chains (forward + backward, fills gaps)
            try self.detectElementwiseChains(alloc);

            std.mem.sortUnstable(FusionPlan(T), self.fused_chains.items, {}, struct {
                fn cmp(_: void, a: FusionPlan(T), b: FusionPlan(T)) bool {
                    return a.output_idx < b.output_idx;
                }
            }.cmp);
        }

        // =================================================================
        // Forward patterns
        // =================================================================

        fn tryForwardPattern(self: *Self, alloc: std.mem.Allocator, node: *Tensor(T), idx: usize) !bool {
            if (self.detectCrossEntropy(node, idx)) |plan| {
                try self.fused_chains.append(alloc, plan);
                return true;
            }
            if (self.detectLogSoftmax(node, idx)) |plan| {
                try self.fused_chains.append(alloc, plan);
                return true;
            }
            if (self.detectSoftmax(node, idx)) |plan| {
                try self.fused_chains.append(alloc, plan);
                return true;
            }
            if (self.detectLayerNorm(node, idx)) |plan| {
                try self.fused_chains.append(alloc, plan);
                return true;
            }
            if (self.detectConv2dForward(node, idx)) |plan| {
                try self.fused_chains.append(alloc, plan);
                return true;
            }
            if (self.detectMaxPool2dForward(node, idx)) |plan| {
                try self.fused_chains.append(alloc, plan);
                return true;
            }
            return false;
        }

        /// softmax: mul(exp(x - rep(max(x))), recip(rep(sum(exp(...)))))
        fn detectSoftmax(self: *Self, node: *Tensor(T), idx: usize) ?FusionPlan(T) {
            if (node.opTag() != .mul) return null;
            const exp_node = expect(node.source0(), .exp) orelse return null;
            const recip_rep_sum = expect(node.source1(), .recip) orelse return null;
            const rep_sum = expect(recip_rep_sum.source0(), .repeat) orelse return null;
            const sum_node = expect(rep_sum.source0(), .sum) orelse return null;
            if (sum_node.source0() != exp_node) return null;

            const shifted = expect(exp_node.source0(), .add) orelse return null;
            const neg_rep_max = expect(shifted.source1(), .neg) orelse return null;
            const rep_max = expect(neg_rep_max.source0(), .repeat) orelse return null;
            const max_node = expect(rep_max.source0(), .max) orelse return null;
            const input = shifted.source0() orelse return null;
            if (max_node.source0() != input) return null;

            const plan = fused.SoftmaxPlan(T){
                .input = input,
                .max_node = max_node,
                .rep_max = rep_max,
                .neg_rep_max = neg_rep_max,
                .shifted = shifted,
                .exp_node = exp_node,
                .sum_node = sum_node,
                .rep_sum = rep_sum,
                .recip_rep_sum = recip_rep_sum,
                .output = node,
            };
            if (!fused.validateSoftmaxPlan(T, plan)) return null;
            self.markNodes(&.{ max_node, rep_max, neg_rep_max, shifted, exp_node, sum_node, rep_sum, recip_rep_sum, node });
            return .{ .output_idx = idx, .payload = .{ .softmax = plan } };
        }

        /// log_softmax: add(shifted, neg(rep(log(sum(exp(shifted))))))
        fn detectLogSoftmax(self: *Self, node: *Tensor(T), idx: usize) ?FusionPlan(T) {
            return self.detectLogSoftmaxImpl(node, idx, true);
        }

        fn detectLogSoftmaxImpl(self: *Self, node: *Tensor(T), idx: usize, mark: bool) ?FusionPlan(T) {
            if (node.opTag() != .add) return null;
            const shifted = node.source0() orelse return null;
            const neg_rep_log = expect(node.source1(), .neg) orelse return null;
            const rep_log = expect(neg_rep_log.source0(), .repeat) orelse return null;
            const log_node = expect(rep_log.source0(), .log) orelse return null;
            const sum_node = expect(log_node.source0(), .sum) orelse return null;
            const exp_node = expect(sum_node.source0(), .exp) orelse return null;
            if (exp_node.source0() != shifted) return null;

            if (shifted.opTag() != .add) return null;
            const neg_rep_max = expect(shifted.source1(), .neg) orelse return null;
            const rep_max = expect(neg_rep_max.source0(), .repeat) orelse return null;
            const max_node = expect(rep_max.source0(), .max) orelse return null;
            const input = shifted.source0() orelse return null;
            if (max_node.source0() != input) return null;

            const plan = fused.LogSoftmaxPlan(T){
                .input = input,
                .max_node = max_node,
                .rep_max = rep_max,
                .neg_rep_max = neg_rep_max,
                .shifted = shifted,
                .exp_node = exp_node,
                .sum_node = sum_node,
                .log_node = log_node,
                .rep_log = rep_log,
                .neg_rep_log = neg_rep_log,
                .output = node,
            };
            if (!fused.validateLogSoftmaxPlan(T, plan)) return null;
            if (mark) self.markNodes(&.{ max_node, rep_max, neg_rep_max, shifted, exp_node, sum_node, log_node, rep_log, neg_rep_log, node });
            return .{ .output_idx = idx, .payload = .{ .log_softmax = plan } };
        }

        /// cross_entropy: mean(neg(pick_rows(log_softmax(x), targets)))
        fn detectCrossEntropy(self: *Self, node: *Tensor(T), idx: usize) ?FusionPlan(T) {
            if (node.opTag() != .mul) return null;
            const sum_node = expect(node.source0(), .sum) orelse return null;
            const scale = node.source1() orelse return null;
            if (!scale.isScalar()) return null;

            const neg_picked = expect(sum_node.source0(), .neg) orelse return null;
            const picked = expect(neg_picked.source0(), .pick_rows) orelse return null;
            const log_softmax_output = picked.source0() orelse return null;
            const targets = picked.source1() orelse return null;

            const ls_idx = self.ptr_to_idx.get(log_softmax_output) orelse return null;
            const ls = self.detectLogSoftmaxImpl(log_softmax_output, ls_idx, false) orelse return null;
            const inner = switch (ls.payload) {
                .log_softmax => |p| p,
                else => return null,
            };

            self.markNodes(&.{
                inner.max_node,    inner.rep_max,  inner.neg_rep_max, inner.shifted,
                inner.exp_node,    inner.sum_node, inner.log_node,    inner.rep_log,
                inner.neg_rep_log, inner.output,   picked,            neg_picked,
                sum_node,          node,
            });
            return .{ .output_idx = idx, .payload = .{ .cross_entropy = .{
                .log_softmax = inner,
                .targets = targets,
                .picked = picked,
                .neg_picked = neg_picked,
                .sum_node = sum_node,
                .mean_node = node,
            } } };
        }

        /// layer_norm: mul(centered, rep(recip(sqrt(var + eps))))
        fn detectLayerNorm(self: *Self, node: *Tensor(T), idx: usize) ?FusionPlan(T) {
            if (node.opTag() != .mul) return null;
            const centered = node.source0() orelse return null;
            const rep_std_inv = expect(node.source1(), .repeat) orelse return null;
            const recip_node = expect(rep_std_inv.source0(), .recip) orelse return null;
            const sqrt_node = expect(recip_node.source0(), .sqrt) orelse return null;
            const var_eps = expect(sqrt_node.source0(), .add) orelse return null;
            const var_node = var_eps.source0() orelse return null;
            const eps_like = var_eps.source1() orelse return null;
            if (!isScalarBroadcast(eps_like)) return null;

            if (var_node.opTag() != .mul) return null;
            const var_sum = expect(var_node.source0(), .sum) orelse return null;
            if (var_node.source1() == null) return null;
            const sqr_node = var_sum.source0() orelse return null;
            if (sqr_node.opTag() != .mul) return null;
            if (sqr_node.source0() != centered or sqr_node.source1() != centered) return null;

            if (centered.opTag() != .add) return null;
            const neg_rep_mean = expect(centered.source1(), .neg) orelse return null;
            const rep_mean = expect(neg_rep_mean.source0(), .repeat) orelse return null;
            const mean_node = rep_mean.source0() orelse return null;
            if (mean_node.opTag() != .mul) return null;
            if (mean_node.source1() == null) return null;
            const sum_node = expect(mean_node.source0(), .sum) orelse return null;
            const input = centered.source0() orelse return null;
            if (sum_node.source0() != input) return null;

            const plan = fused.LayerNormPlan(T){
                .input = input,
                .sum_node = sum_node,
                .mean_node = mean_node,
                .rep_mean = rep_mean,
                .neg_rep_mean = neg_rep_mean,
                .centered = centered,
                .sqr_node = sqr_node,
                .var_sum = var_sum,
                .var_node = var_node,
                .eps_like = eps_like,
                .var_eps = var_eps,
                .sqrt_node = sqrt_node,
                .recip_node = recip_node,
                .rep_std_inv = rep_std_inv,
                .output = node,
            };
            if (!fused.validateLayerNormPlan(T, plan)) return null;
            self.markNodes(&.{
                sum_node,  mean_node,  rep_mean,    neg_rep_mean, centered,
                sqr_node,  var_sum,    var_node,    eps_like,     var_eps,
                sqrt_node, recip_node, rep_std_inv, node,
            });
            return .{ .output_idx = idx, .payload = .{ .layer_norm = plan } };
        }

        /// conv2d: reshape(sum(mul(as_strided(input), as_strided(kernel)))) [+ bias] [+ relu]
        fn detectConv2dForward(self: *Self, node: *Tensor(T), idx: usize) ?FusionPlan(T) {
            var core = node;
            var activation: ?*Tensor(T) = null;
            var step_node: ?*Tensor(T) = null;
            var bias: ?*Tensor(T) = null;
            var bias_node: ?*Tensor(T) = null;
            var bias_add: ?*Tensor(T) = null;

            // Peel optional relu: mul(x, step(x))
            if (core.opTag() == .mul) {
                const s0 = core.source0() orelse return null;
                const s1 = core.source1() orelse return null;
                if (s1.opTag() == .step and s1.source0() == s0) {
                    activation = core;
                    step_node = s1;
                    core = s0;
                } else if (s0.opTag() == .step and s0.source0() == s1) {
                    activation = core;
                    step_node = s0;
                    core = s1;
                }
            }

            // Peel optional bias: add(core, repeat(bias))
            if (core.opTag() == .add) {
                const s0 = core.source0() orelse return null;
                const s1 = core.source1() orelse return null;
                const bias_side = if (s1.opTag() == .repeat and !s1.isScalar()) s1 else if (s0.opTag() == .repeat and !s0.isScalar()) s0 else null;
                if (bias_side) |bs| {
                    const other = if (bs == s1) s0 else s1;
                    if (other.opTag() == .reshape) {
                        bias_add = core;
                        bias = bs.source0();
                        bias_node = bs;
                        core = other;
                    }
                }
            }

            if (core.opTag() != .reshape or core.n_dims != 4) return null;
            // Guard: skip if core was already claimed by another fusion plan.
            if (self.ptr_to_idx.get(core)) |ci| {
                if (self.fused_skip[ci]) return null;
            }
            const sum_node = expect(core.source0(), .sum) orelse return null;
            const mul_node = expect(sum_node.source0(), .mul) orelse return null;
            const input_view = expect(mul_node.source0(), .as_strided) orelse return null;
            const kernel_view = expect(mul_node.source1(), .as_strided) orelse return null;
            if (input_view.n_dims != 7 or kernel_view.n_dims != 7) return null;
            const input = input_view.source0() orelse return null;
            const kernel = kernel_view.source0() orelse return null;
            if (input.n_dims != 4 or kernel.n_dims != 4) return null;

            self.markNodes(&.{ input_view, kernel_view, mul_node, sum_node, core, node });
            if (bias_node) |bn| self.markNodes(&.{bn});
            if (bias_add) |ba| self.markNodes(&.{ba});
            if (step_node) |sn| self.markNodes(&.{sn});

            return .{ .output_idx = idx, .payload = .{ .conv2d = .{
                .input = input,
                .kernel = kernel,
                .conv_out = core,
                .input_view = input_view,
                .kernel_view = kernel_view,
                .bias = bias,
                .bias_node = bias_node,
                .bias_add = bias_add,
                .activation = activation,
                .step_node = step_node,
                .mul_node = mul_node,
                .sum_node = sum_node,
                .output = node,
            } } };
        }

        /// maxpool2d: reshape(max(as_strided_6d(input))) with 2x2 window
        fn detectMaxPool2dForward(self: *Self, node: *Tensor(T), idx: usize) ?FusionPlan(T) {
            if (node.opTag() != .reshape or node.n_dims != 4) return null;
            const max_node = expect(node.source0(), .max) orelse return null;
            const strided = expect(max_node.source0(), .as_strided) orelse return null;
            if (strided.n_dims != 6 or strided.ne[2] != 2 or strided.ne[3] != 2) return null;
            const input = strided.source0() orelse return null;
            if (input.n_dims != 4) return null;

            self.markNodes(&.{ strided, max_node, node });
            return .{ .output_idx = idx, .payload = .{ .max_pool2d = .{
                .input = input,
                .strided = strided,
                .max_node = max_node,
                .output = node,
            } } };
        }

        // =================================================================
        // Backward patterns
        // =================================================================

        /// conv2d backward: scatter_add_view with 7D as_strided view
        fn detectConv2dBwd(self: *Self, scatter: *Tensor(T), scatter_idx: usize) ?FusionPlan(T) {
            _ = self;
            if (scatter.opTag() != .scatter_add_view) return null;
            const view = scatter.source1() orelse return null;
            if (view.opTag() != .as_strided or view.n_dims != 7) return null;
            const view_source = view.source0() orelse return null;
            if (view_source.n_dims != 4) return null;

            const mul_node = findPastAdd(scatter.source0() orelse return null, .mul) orelse return null;
            const bcast = findPastAdd(mul_node.source0() orelse return null, .broadcast_to) orelse
                findPastAdd(mul_node.source1() orelse return null, .broadcast_to) orelse return null;
            const other_view = findPastAdd(mul_node.source0() orelse return null, .as_strided) orelse
                findPastAdd(mul_node.source1() orelse return null, .as_strided) orelse return null;
            const other_source = other_view.source0() orelse return null;
            if (other_source.n_dims != 4) return null;
            const reshape = findPastAdd(bcast.source0() orelse return null, .reshape) orelse return null;
            const output_grad = reshape.source0() orelse return null;
            if (output_grad.n_dims != 4) return null;

            const is_kernel = view_source.nElems() <= other_source.nElems();
            return if (is_kernel)
                .{ .output_idx = scatter_idx, .payload = .{ .conv2d_bwd_kernel = .{
                    .input = other_source,
                    .output_grad = output_grad,
                    .reshape_node = reshape,
                    .repeat_node = bcast,
                    .mul_node = mul_node,
                    .output = scatter,
                } } }
            else
                .{ .output_idx = scatter_idx, .payload = .{ .conv2d_bwd_input = .{
                    .output_grad = output_grad,
                    .kernel = other_source,
                    .reshape_node = reshape,
                    .repeat_node = bcast,
                    .mul_node = mul_node,
                    .output = scatter,
                } } };
        }

        /// maxpool2d backward: scatter_add_view with 6D as_strided, 2x2 window
        fn detectMaxPool2dBwd(self: *Self, scatter: *Tensor(T), scatter_idx: usize) ?FusionPlan(T) {
            _ = self;
            if (scatter.opTag() != .scatter_add_view) return null;
            const view = scatter.source1() orelse return null;
            if (view.opTag() != .as_strided or view.n_dims != 6) return null;
            const input = view.source0() orelse return null;
            if (input.n_dims != 4 or view.ne[2] != 2 or view.ne[3] != 2) return null;

            const bcast = findBroadcastFromReshape(scatter.source0() orelse return null) orelse return null;
            const reshape = findPastAdd(bcast.source0() orelse return null, .reshape) orelse return null;
            const output_grad = reshape.source0() orelse return null;
            if (output_grad.n_dims != 4) return null;

            return .{ .output_idx = scatter_idx, .payload = .{ .max_pool2d_bwd = .{
                .input = input,
                .output_grad = output_grad,
                .output = scatter,
            } } };
        }

        // =================================================================
        // Elementwise chain detection
        // =================================================================

        fn detectElementwiseChains(self: *Self, alloc: std.mem.Allocator) !void {
            const n = self.nodes.len;
            var i: usize = 0;
            while (i < n) : (i += 1) {
                const node = self.nodes[i];
                if (!node.opTag().isFusible() or self.fused_skip[i]) continue;
                if (!isSafeBinaryOperand(node)) continue;

                const start = i;
                var end = i;
                while (end + 1 < n) {
                    const next = self.nodes[end + 1];
                    if (!next.opTag().isFusible() or self.fused_skip[end + 1]) break;
                    if (self.use_count[end] != 1 or !next.isSameShape(self.nodes[end])) break;

                    const is_chain = (next.source0().? == self.nodes[end]) or
                        (isCommutative(next) and (if (next.source1()) |s1| s1 == self.nodes[end] else false));
                    if (!is_chain) break;

                    const other = if (next.source0().? == self.nodes[end]) next.source1() else next.source0();
                    if (other) |o| {
                        if (!o.isScalar() and !(next.isSameShape(o) and o.isContiguous() and o.data.len == next.nElems())) break;
                    }
                    end += 1;
                }

                if (end - start < 1) continue;

                const chain_len = end - start + 1;
                const chain_nodes = try alloc.alloc(*Tensor(T), chain_len);
                for (start..end + 1, 0..) |idx, k| {
                    chain_nodes[k] = self.nodes[idx];
                    self.fused_skip[idx] = true;
                }

                const input = self.nodes[start].source0().?;
                const roles = try buildOperandRoles(alloc, input, chain_nodes);
                try self.fused_chains.append(alloc, .{
                    .output_idx = end,
                    .payload = .{ .elementwise_chain = .{
                        .input = input,
                        .nodes = chain_nodes,
                        .other_operand_roles = roles,
                    } },
                });
                i = end;
            }
        }

        // =================================================================
        // Helpers
        // =================================================================

        /// Check if node's optional source (op) matches expected op tag.
        fn expect(maybe: ?*Tensor(T), op: Op) ?*Tensor(T) {
            const node = maybe orelse return null;
            return if (node.opTag() == op) node else null;
        }

        fn isScalarBroadcast(node: *Tensor(T)) bool {
            if (node.isScalar()) return true;
            return (node.opTag() == .repeat or node.opTag() == .broadcast_to) and
                (if (node.source0()) |src| src.isScalar() else false);
        }

        fn isCommutative(node: *Tensor(T)) bool {
            return node.opTag() == .add or node.opTag() == .mul;
        }

        fn isSafeBinaryOperand(node: *Tensor(T)) bool {
            const src1 = node.source1() orelse return true;
            return src1.isScalar() or (node.isSameShape(src1) and src1.isContiguous() and src1.data.len == node.nElems());
        }

        fn findPastAdd(start: *Tensor(T), target: Op) ?*Tensor(T) {
            if (start.opTag() == target) return start;
            if (start.opTag() != .add) return null;
            if (start.source0()) |s| if (s.opTag() == target) return s;
            if (start.source1()) |s| if (s.opTag() == target) return s;
            return null;
        }

        fn findOp(node: *Tensor(T), target: Op) ?*Tensor(T) {
            if (node.source0()) |s| if (s.opTag() == target) return s;
            if (node.source1()) |s| if (s.opTag() == target) return s;
            return null;
        }

        fn findBroadcastFromReshape(node: *Tensor(T)) ?*Tensor(T) {
            if (node.opTag() == .broadcast_to) {
                const src = node.source0() orelse return null;
                const reshape = findPastAdd(src, .reshape) orelse return null;
                const output_grad = reshape.source0() orelse return null;
                if (output_grad.n_dims == 4) return node;
            }
            if (node.source0()) |s| {
                if (findBroadcastFromReshape(s)) |hit| return hit;
            }
            if (node.source1()) |s| {
                if (findBroadcastFromReshape(s)) |hit| return hit;
            }
            return null;
        }

        fn markNodes(self: *Self, nodes: []const *Tensor(T)) void {
            for (nodes) |node| {
                if (self.ptr_to_idx.get(node)) |i| self.fused_skip[i] = true;
            }
        }

        fn buildOperandRoles(alloc: std.mem.Allocator, input: *Tensor(T), nodes: []const *Tensor(T)) ![]const fused.BinaryOperandRole {
            const roles = try alloc.alloc(fused.BinaryOperandRole, nodes.len);
            var any_swapped = false;
            var prev: *Tensor(T) = input;
            for (nodes, 0..) |node, k| {
                if (node.opTag().isBinary() and isCommutative(node) and
                    node.source1().? == prev and node.source0().? != prev)
                {
                    roles[k] = .src0;
                    any_swapped = true;
                } else {
                    roles[k] = .src1;
                }
                prev = node;
            }
            if (!any_swapped) {
                alloc.free(roles);
                return &.{};
            }
            return roles;
        }

        // Backward skip helpers

        fn markBwdConvSkip(self: *Self, plan: FusionPlan(T), scatter_idx: usize) void {
            self.fused_skip[scatter_idx] = true;
            const intermediates: [3]*Tensor(T) = switch (plan.payload) {
                .conv2d_bwd_kernel => |p| .{ p.mul_node, p.repeat_node, p.reshape_node },
                .conv2d_bwd_input => |p| .{ p.mul_node, p.repeat_node, p.reshape_node },
                else => return,
            };
            for (intermediates) |node| {
                const i = self.ptr_to_idx.get(node) orelse continue;
                if (self.use_count[i] > 1) continue;
                self.fused_skip[i] = true;
                // Also skip gradient accumulation add if single-consumer
                for (self.nodes[self.forward_node_count..], self.forward_node_count..) |n, ni| {
                    if (n.opTag() == .add and (n.source0() == node or n.source1() == node) and self.use_count[ni] <= 1) {
                        self.fused_skip[ni] = true;
                        break;
                    }
                }
            }
        }

        fn markBwdSkipChain(self: *Self, scatter_idx: usize) void {
            self.fused_skip[scatter_idx] = true;
            var cur = self.nodes[scatter_idx].source0() orelse return;
            while (true) {
                const i = self.ptr_to_idx.get(cur) orelse break;
                if (self.use_count[i] > 1) break;
                self.fused_skip[i] = true;
                cur = cur.source0() orelse break;
            }
        }
    };
}
