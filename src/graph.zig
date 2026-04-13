//! Computation graph for automatic differentiation.
//!
//! A `ComputeGraph` owns an arena allocator and manages the lifecycle of all
//! tensors created within it. Call `allocator()` to get the arena for tensor
//! creation, then `buildForward` / `buildBackward` to wire up the graph.
//! A single `deinit()` frees everything.

const std = @import("std");

const tensorlib = @import("./tensor.zig");
const Tensor = tensorlib.Tensor;
const Op = @import("op.zig").Op;
const fused = @import("tensor/fused.zig");
const assert = std.debug.assert;
const testing = std.testing;
const Alloc = std.mem.Allocator;
const tac = std.testing.allocator;

/// Manages forward and backward passes over a tensor computation graph.
///
/// All tensors should be allocated from `allocator()` so that `deinit()`
/// can free them in bulk via the arena.
pub fn ComputeGraph(comptime T: type) type {
    return struct {
        const Self = @This();

        built_forward: bool = false,
        built_backward: bool = false,
        forward_node_count: usize = 0,

        arena: std.heap.ArenaAllocator,
        nodes: std.ArrayList(*Tensor(T)),
        grads: std.ArrayList(?*Tensor(T)),
        leaves: std.ArrayList(*Tensor(T)),
        scratch: std.ArrayList(*Tensor(T)),

        /// Fusion state — populated by `fusionPass()`.
        fused_chains: std.ArrayList(fused.FusionPlan(T)),
        /// Per-node flag: true means this node is part of a fused chain
        /// and should be skipped during normal compute iteration.
        fused_skip: std.ArrayList(bool),

        /// Set up resources for compute graph.
        /// Must call `buildForward` (then optionally `buildBackward`) to be able to do computation.
        pub fn init(backing_alloc: Alloc) Self {
            return .{
                .arena = std.heap.ArenaAllocator.init(backing_alloc),
                .nodes = .{},
                .grads = .{},
                .leaves = .{},
                .scratch = .{},
                .fused_chains = .{},
                .fused_skip = .{},
            };
        }

        /// Returns the arena allocator for allocating tensors within this graph.
        pub fn allocator(self: *Self) Alloc {
            return self.arena.allocator();
        }

        /// Clean up all the resources for this compute graph
        pub fn deinit(self: *Self) void {
            const alloc = self.arena.allocator();
            self.fused_chains.deinit(alloc);
            self.fused_skip.deinit(alloc);
            self.nodes.deinit(alloc);
            self.grads.deinit(alloc);
            self.leaves.deinit(alloc);
            self.scratch.deinit(alloc);
            self.arena.deinit();
        }

        /// Build a graph where the provided tensor is the final output node
        pub fn buildForward(self: *Self, tensor: *Tensor(T)) Alloc.Error!void {
            try self.buildForwardHelper(tensor);
            self.built_forward = true;
            self.forward_node_count = self.nodes.items.len;
        }
        fn buildForwardHelper(self: *Self, tensor: *Tensor(T)) Alloc.Error!void {
            const n_before = self.nodes.items.len;
            try self.addParentsThenSelf(tensor);
            // tensor should be last node
            const n_change = self.nodes.items.len - n_before;
            if (n_change > 0) assert(self.nodes.items[self.nodes.items.len - 1] == tensor);
        }
        /// Build a backward graph
        pub fn buildBackward(self: *Self, keep: bool) Alloc.Error!void {
            assert(self.nodes.items.len > 0);
            const alloc = self.arena.allocator();
            const nodes_len = self.nodes.items.len;
            for (0..nodes_len) |j| {
                const i = nodes_len - j - 1;
                const node = self.nodes.items[i];

                // because we detached the grad nodes from the original graph, we can afford inplace operations
                if (node.grad != null) {
                    try node.backward(alloc, &self.scratch, keep);
                }
            }
            for (0..nodes_len) |j| {
                const i = nodes_len - j - 1;
                const node = self.nodes.items[i];
                if (node.is_param) {
                    assert(node.grad != null);
                    try self.buildForwardHelper(node.grad.?);
                }
            }

            self.built_backward = true;
            self.resetGrads();
        }

        /// Detect chains of fusible elementwise ops in the forward graph
        /// and prepare comptime-specialized kernels for them.
        ///
        /// Call after `buildForward()` (and optionally after `buildBackward()`).
        /// Fusion only affects the forward pass — backward still uses
        /// the original nodes and their stored intermediate values.
        pub fn fusionPass(self: *Self) Alloc.Error!void {
            const alloc = self.arena.allocator();
            const fwd_count = self.forward_node_count;
            if (fwd_count < 2) return;

            self.fused_chains.clearRetainingCapacity();
            self.fused_skip.clearRetainingCapacity();

            // Build use-count: how many forward nodes read each node's output
            const use_count = try alloc.alloc(u16, fwd_count);
            @memset(use_count, 0);

            for (self.nodes.items[0..fwd_count]) |node| {
                if (node.src0) |src| {
                    for (self.nodes.items[0..fwd_count], 0..) |n, j| {
                        if (n == src) {
                            use_count[j] += 1;
                            break;
                        }
                    }
                }
                if (node.src1) |src| {
                    for (self.nodes.items[0..fwd_count], 0..) |n, j| {
                        if (n == src) {
                            use_count[j] += 1;
                            break;
                        }
                    }
                }
            }

            // Initialize skip bitmap
            self.fused_skip = try std.ArrayList(bool).initCapacity(alloc, fwd_count);
            self.fused_skip.items.len = fwd_count;
            @memset(self.fused_skip.items, false);

            // Scan for fusible chains
            var i: usize = 0;
            while (i < fwd_count) : (i += 1) {
                const node = self.nodes.items[i];
                if (detectCrossEntropyPattern(self, i)) |plan| {
                    for (i..plan.output_idx + 1) |idx| self.fused_skip.items[idx] = true;
                    try self.fused_chains.append(alloc, plan);
                    i = plan.output_idx;
                    continue;
                }
                if (detectSoftmaxPattern(self, i)) |plan| {
                    for (i..plan.output_idx + 1) |idx| self.fused_skip.items[idx] = true;
                    try self.fused_chains.append(alloc, plan);
                    i = plan.output_idx;
                    continue;
                }
                if (!node.op.isFusible()) continue;
                if (self.fused_skip.items[i]) continue;

                // Start a chain from this node
                const chain_start = i;
                var chain_end = i; // inclusive

                // Extend forward: look for the next node that reads chain_end via src0
                var extending = true;
                while (extending) {
                    extending = false;
                    if (chain_end + 1 < fwd_count) {
                        const next = self.nodes.items[chain_end + 1];
                        if (next.op.isFusible() and
                            next.src0.? == self.nodes.items[chain_end] and
                            next.isSameShape(self.nodes.items[chain_end]) and
                            use_count[chain_end] == 1)
                        {
                            chain_end += 1;
                            extending = true;
                        }
                    }
                }

                const chain_len = chain_end - chain_start + 1;
                if (chain_len < 2) continue;

                // Record the fused chain
                const chain_nodes = try alloc.alloc(*Tensor(T), chain_len);
                for (chain_start..chain_end + 1, 0..) |idx, k| {
                    chain_nodes[k] = self.nodes.items[idx];
                    self.fused_skip.items[idx] = true;
                }

                try self.fused_chains.append(alloc, .{
                    .nodes = chain_nodes,
                    .output_idx = chain_end,
                    .kind = .elementwise_chain,
                });

                i = chain_end; // skip past the chain
            }
        }

        fn detectSoftmaxPattern(self: *Self, start: usize) ?fused.FusionPlan(T) {
            const nodes = self.nodes.items;

            if (start + 8 < self.forward_node_count) {
                const n0 = nodes[start + 0];
                const n1 = nodes[start + 1];
                const n2 = nodes[start + 2];
                const n3 = nodes[start + 3];
                const n4 = nodes[start + 4];
                const n5 = nodes[start + 5];
                const n6 = nodes[start + 6];
                const n7 = nodes[start + 7];
                const n8 = nodes[start + 8];

                if (n0.op == .max and
                    n1.op == .repeat and n1.src0.? == n0 and
                    n2.op == .neg and n2.src0.? == n1 and
                    n3.op == .add and n3.src0.? == n0.src0.? and n3.src1.? == n2 and
                    n4.op == .exp and n4.src0.? == n3 and
                    n5.op == .sum and n5.src0.? == n4 and
                    n6.op == .repeat and n6.src0.? == n5 and
                    n7.op == .recip and n7.src0.? == n6 and
                    n8.op == .mul and n8.src0.? == n4 and n8.src1.? == n7)
                {
                    return .{ .nodes = nodes[start .. start + 9], .output_idx = start + 8, .kind = .softmax };
                }
            }

            if (start + 9 < self.forward_node_count) {
                const n0 = nodes[start + 0];
                const n1 = nodes[start + 1];
                const n2 = nodes[start + 2];
                const n3 = nodes[start + 3];
                const n4 = nodes[start + 4];
                const n5 = nodes[start + 5];
                const n6 = nodes[start + 6];
                const n7 = nodes[start + 7];
                const n8 = nodes[start + 8];
                const n9 = nodes[start + 9];

                if (n0.op == .max and
                    n1.op == .repeat and n1.src0.? == n0 and
                    n2.op == .neg and n2.src0.? == n1 and
                    n3.op == .add and n3.src0.? == n0.src0.? and n3.src1.? == n2 and
                    n4.op == .exp and n4.src0.? == n3 and
                    n5.op == .sum and n5.src0.? == n4 and
                    n6.op == .log and n6.src0.? == n5 and
                    n7.op == .repeat and n7.src0.? == n6 and
                    n8.op == .neg and n8.src0.? == n7 and
                    n9.op == .add and n9.src0.? == n3 and n9.src1.? == n8)
                {
                    return .{ .nodes = nodes[start .. start + 10], .output_idx = start + 9, .kind = .log_softmax };
                }
            }

            return null;
        }

        fn detectCrossEntropyPattern(self: *Self, start: usize) ?fused.FusionPlan(T) {
            const nodes = self.nodes.items;
            if (start + 13 >= self.forward_node_count) return null;

            const n0 = nodes[start + 0];
            const n1 = nodes[start + 1];
            const n2 = nodes[start + 2];
            const n3 = nodes[start + 3];
            const n4 = nodes[start + 4];
            const n5 = nodes[start + 5];
            const n6 = nodes[start + 6];
            const n7 = nodes[start + 7];
            const n8 = nodes[start + 8];
            const n9 = nodes[start + 9];
            const n10 = nodes[start + 10];
            const n11 = nodes[start + 11];
            const n12 = nodes[start + 12];
            const n13 = nodes[start + 13];

            if (n0.op == .max and
                n1.op == .repeat and n1.src0.? == n0 and
                n2.op == .neg and n2.src0.? == n1 and
                n3.op == .add and n3.src0.? == n0.src0.? and n3.src1.? == n2 and
                n4.op == .exp and n4.src0.? == n3 and
                n5.op == .sum and n5.src0.? == n4 and
                n6.op == .log and n6.src0.? == n5 and
                n7.op == .repeat and n7.src0.? == n6 and
                n8.op == .neg and n8.src0.? == n7 and
                n9.op == .add and n9.src0.? == n3 and n9.src1.? == n8 and
                n10.op == .pick_rows and n10.src0.? == n9 and
                n11.op == .neg and n11.src0.? == n10 and
                n12.op == .sum and n12.src0.? == n11 and
                n13.op == .mul and n13.src0.? == n12 and n13.src1.?.isScalar())
            {
                return .{ .nodes = nodes[start .. start + 14], .output_idx = start + 13, .kind = .cross_entropy };
            }

            return null;
        }

        fn addParentsThenSelf(self: *Self, tensor: *Tensor(T)) Alloc.Error!void {
            const alloc = self.arena.allocator();
            // check if already visited
            for (self.nodes.items) |node| {
                if (tensor == node) {
                    return;
                }
            }
            for (self.leaves.items) |node| {
                if (tensor == node) {
                    return;
                }
            }
            // visit parents
            if (tensor.src0) |ts0| try self.addParentsThenSelf(ts0);
            if (tensor.src1) |ts1| try self.addParentsThenSelf(ts1);
            if (tensor.op == .none and tensor.grad == null) {
                // is leaf
                try self.leaves.append(alloc, tensor);
            } else {
                try self.nodes.append(alloc, tensor);
                try self.grads.append(alloc, tensor.grad);
            }
        }
        pub fn toGraphViz(self: *const Self, alloc: Alloc) Alloc.Error!std.ArrayList(u8) {
            var str = std.ArrayList(u8){};
            const writer = str.writer(alloc);
            try writer.print("digraph G {{\n  node [shape=box];\n", .{});

            for (self.nodes.items) |node| {
                try writer.print("  \"{*}\" [shape=\"none\",label=<<table>", .{node});
                if (node.op == .none) {
                    try writer.print("<tr><td>{any}</td></tr>", .{node.data});
                } else {
                    try writer.print("<tr><td>{any}</td></tr>", .{node.data});
                    try writer.print("<tr><td>{s}</td></tr>", .{node.op.symbol()});
                }
                if (node.name) |name| {
                    try writer.print("<tr><td>{s}</td></tr>", .{name});
                }
                try writer.print("<tr><td>{any}</td></tr>", .{node.ne});
                try writer.print("</table>>];\n", .{});
                if (node.src0) |src0| {
                    try writer.print("  \"{*}\" -> \"{*}\";\n", .{ src0, node });
                }
                if (node.src1) |src1| {
                    try writer.print("  \"{*}\" -> \"{*}\";\n", .{ src1, node });
                }
                if (node.grad) |grad| {
                    try writer.print("  \"{*}\" -> \"{*}\" [style=dashed];\n", .{ node, grad });
                }
                if (!node.data_owned) {
                    try writer.print("  \"{*}\" [style=filled fillcolor=gray];\n", .{node});
                }
            }
            for (self.leaves.items) |leaf| {
                try writer.print("  \"{*}\" [style=filled fillcolor=green label=\"{any}\"];\n", .{ leaf, leaf.data });
            }
            for (self.scratch.items) |item| {
                try writer.print("  \"{*}\" [style=filled fillcolor=gray label=\"{any}\"];\n", .{ item, item.data });
            }
            try writer.print("}}\n", .{});
            return str;
        }

        pub fn resetGrads(self: *Self) void {
            for (self.grads.items) |grad_o| {
                if (grad_o) |grad| {
                    _ = grad.setAllScalar(0);
                }
            }
        }
        pub fn reset(self: *Self) void {
            for (self.nodes.items) |node| {
                if (node.op != .none) _ = node.setAllScalar(0);
            }
        }

        /// Execute the forward pass over all nodes.
        /// If `fusionPass()` was called, fused chains execute as single-pass
        /// comptime-specialized kernels.
        pub fn compute(self: *const Self) void {
            if (self.fused_chains.items.len == 0) {
                for (self.nodes.items) |node| {
                    node.compute();
                }
                return;
            }
            var next_chain: usize = 0;
            for (self.nodes.items, 0..) |node, i| {
                if (i < self.fused_skip.items.len and self.fused_skip.items[i]) {
                    // When we reach a chain's output node, execute the fused kernel
                    if (next_chain < self.fused_chains.items.len and
                        self.fused_chains.items[next_chain].output_idx == i)
                    {
                        fused.executeFusionPlan(T, self.fused_chains.items[next_chain]);
                        next_chain += 1;
                    }
                    continue;
                }
                node.compute();
            }
        }

        pub fn computeNoGrad(self: *const Self) void {
            if (self.fused_chains.items.len == 0) {
                for (self.nodes.items[0..self.forward_node_count]) |node| {
                    node.compute();
                }
                return;
            }
            var next_chain: usize = 0;
            for (self.nodes.items[0..self.forward_node_count], 0..) |node, i| {
                if (i < self.fused_skip.items.len and self.fused_skip.items[i]) {
                    if (next_chain < self.fused_chains.items.len and
                        self.fused_chains.items[next_chain].output_idx == i)
                    {
                        fused.executeFusionPlan(T, self.fused_chains.items[next_chain]);
                        next_chain += 1;
                    }
                    continue;
                }
                node.compute();
            }
        }
    };
}

test "ref all decls" {
    _ = testing.refAllDeclsRecursive(ComputeGraph(f32));
}

//#region Tests

test "tensor compute graph - matmul" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const t1 = try Tensor(f32).init(a, &.{ 2, 3 });
    t1.setData(&[_]f32{
        1, 2,
        3, 4,
        5, 6,
    });
    t1.setParam(a);

    const t2 = try Tensor(f32).init(a, &.{ 3, 2 });
    t2.setData(&[_]f32{
        1, 2, 3,
        4, 5, 6,
    });

    const dst = t1.matMul(a, false, t2, false);
    try g.buildForward(dst);
    try g.buildBackward(false);

    _ = dst.grad.?.setAllScalar(1);
    g.compute();
    {
        const expected = [_]f32{
            9,  12, 15,
            19, 26, 33,
            29, 40, 51,
        };
        try testing.expectEqualSlices(f32, &expected, dst.data);
    }
    {
        const expected = [_]f32{
            6, 15,
            6, 15,
            6, 15,
        };
        try testing.expectEqualSlices(f32, &expected, t1.grad.?.data);
    }
}

test "build compute graph - forward mul" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const t0 = try Tensor(f32).init(a, &.{1});
    t0.data[0] = 5;
    const t1 = try Tensor(f32).init(a, &.{1});
    t1.data[0] = 6;
    const out = t0.mul(a, t1);
    try g.buildForward(out);
    try g.buildBackward(false);
    g.compute();
    {
        const expected = [_]f32{30};
        try testing.expectEqualSlices(f32, &expected, out.data);
    }
}

test "build computeNoGrad graph - forward mul" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const t0 = try Tensor(f32).init(a, &.{1});
    t0.data[0] = 5;
    t0.setParam(a);
    const t1 = try Tensor(f32).init(a, &.{1});
    t1.data[0] = 6;
    const out = t0.mul(a, t1);
    try g.buildForward(out);
    try g.buildBackward(false);
    const dummy_val: f32 = -23;
    t0.grad.?.data[0] = dummy_val;
    g.computeNoGrad();
    {
        const expected = [_]f32{30};
        try testing.expectEqualSlices(f32, &expected, out.data);
        try testing.expectEqual(dummy_val, t0.grad.?.data[0]);
    }
}

test "build compute graph - forward matMul" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const t1 = try Tensor(f32).init(a, &.{ 2, 3 });
    t1.setData(&[_]f32{
        1, 2,
        3, 4,
        5, 6,
    });
    const intermed = t1.matMul(a, true, t1, false);
    const out = intermed.matMul(a, false, t1, true);
    try g.buildForward(out);
    g.compute();
    {
        const expected = [_]f32{
            35, 44,
            44, 56,
        };
        try testing.expectEqualSlices(f32, &expected, intermed.data);
    }
    {
        const expected = [_]f32{
            123, 281, 439,
            156, 356, 556,
        };
        try testing.expectEqualSlices(f32, &expected, out.data);
    }
}

test "build compute graph - forward mul & add" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 3);
    const w = try Tensor(f32).initScalar(a, 2);
    w.setParam(a);
    const b = try Tensor(f32).initScalar(a, 5);
    b.setParam(a);
    const intermed = w.mul(a, x);
    const out = intermed.add(a, b);
    try g.buildForward(out);
    g.compute();
    {
        const expected = [_]f32{11};
        try testing.expectEqualSlices(f32, &expected, out.data);
    }
}

test "build compute graph - backward" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 3);
    const w = try Tensor(f32).initScalar(a, 2);
    w.setParam(a);
    const b = try Tensor(f32).initScalar(a, 5);
    b.setParam(a);
    const intermed = w.mul(a, x);
    const out = intermed.add(a, b);
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();
    {
        const expected = [_]f32{11};
        try testing.expectEqualSlices(f32, &expected, out.data);
    }
    {
        const expected = [_]f32{3};
        try testing.expectEqualSlices(f32, &expected, w.grad.?.data);
    }
    {
        const expected = [_]f32{1};
        try testing.expectEqualSlices(f32, &expected, b.grad.?.data);
    }
}

fn testSqrFunc(alloc: Alloc, x: *Tensor(f32)) *Tensor(f32) {
    return x.sqr(alloc);
}

test "build compute graph - backward - testSqrFunc" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 3);
    x.setParam(a);
    const out = testSqrFunc(a, x);
    try g.buildForward(out);
    try g.buildBackward(true);

    _ = out.grad.?.setAllScalar(1);
    g.compute();
    {
        const expected = [_]f32{9};
        try testing.expectEqualSlices(f32, &expected, out.data);
    }
    {
        const expected = [_]f32{6};
        try testing.expectEqualSlices(f32, &expected, x.grad.?.data);
    }
    const iters = 10;
    for (0..iters) |_| {
        g.compute();
    }
    {
        const expected = [_]f32{9};
        try testing.expectEqualSlices(f32, &expected, out.data);
    }
    // accumulated gradient
    {
        const expected = [_]f32{6 * (iters + 1)};
        try testing.expectEqualSlices(f32, &expected, x.grad.?.data);
    }
}

fn testSqrSumFunc(alloc: Alloc, x: *Tensor(f32)) *Tensor(f32) {
    return x.sqr(alloc).sumAll(alloc);
}

test "build compute graph - backward - testSqrSumFunc" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    const data = [_]f32{ 3, 4, 10 };
    x.setData(&data);
    x.setParam(a);
    const out = testSqrSumFunc(a, x);
    try g.buildForward(out);
    try g.buildBackward(true);

    _ = out.grad.?.setAllScalar(1);
    g.compute();
    {
        const expected = [_]f32{125};
        try testing.expectEqualSlices(f32, &expected, out.data);
    }
    {
        const expected = [_]f32{ 6, 8, 20 };
        try testing.expectEqualSlices(f32, &expected, x.grad.?.data);
    }
}

test "time speed equation test" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const time = try Tensor(f32).initLinspace(a, &.{20}, 0, 20);

    const c0 = try Tensor(f32).initScalar(a, 0.75);
    const c1 = try Tensor(f32).initScalar(a, 9.5);
    const c2 = try Tensor(f32).initScalar(a, 1);

    const inner = time.sub(a, c1.repeatLike(a, time));
    const inner2 = inner.sqr(a);
    const inner3 = inner2.mul(a, c0.repeatLike(a, inner2));
    const speed = inner3.add(a, c2.repeatLike(a, inner3));

    try g.buildForward(speed);
    g.compute();

    try testing.expectEqual(@as(usize, 20), time.nElems());
    for (time.data, speed.data) |t, s| {
        const t1 = t - 9.5;
        try testing.expectEqual(0.75 * (t1 * t1) + 1, s);
    }
}

test "a*x^2" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 4);
    x.name = "x";
    const coeff = try Tensor(f32).initScalar(a, 2);
    coeff.name = "a";
    coeff.setParam(a);
    const xsq = x.sqr(a);
    xsq.name = "x^2";
    const axsq = xsq.mul(a, coeff);
    axsq.name = "a*x^2";
    try g.buildForward(axsq);
    try g.buildBackward(false);
    g.compute();
    try testing.expectEqual(@as(f32, 32), axsq.data[0]);
}

test "arange a*x^2" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initLinspace(a, &.{5}, 0, 5);
    x.name = "x";
    const coeff = try Tensor(f32).initScalar(a, 2);
    coeff.name = "a";
    coeff.setParam(a);
    const xsq = x.sqr(a);
    xsq.name = "x^2";
    const axsq = xsq.mul(a, coeff.repeatLike(a, xsq));
    axsq.name = "a*x^2";
    try g.buildForward(axsq);
    try g.buildBackward(true);
    g.compute();
    const expected = [_]f32{ 0, 2, 8, 18, 32 };
    try testing.expectEqualSlices(f32, &expected, axsq.data);
}

test "arange" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const t = try Tensor(f32).initLinspace(a, &.{5}, 0, 5);
    t.setParam(a);
    try testing.expectEqual(@as(usize, 5), t.nElems());
    const expected = [_]f32{ 0, 1, 2, 3, 4 };
    try testing.expectEqualSlices(f32, &expected, t.data);

    const out = t.sqr(a).sumAll(a);
    try g.buildForward(out);
    try g.buildBackward(true);
    out.grad.?.data[0] = 1;
    g.compute();

    try testing.expectEqual(@as(f32, 30), out.data[0]);
    try testing.expectEqualSlices(f32, &.{ 0, 2, 4, 6, 8 }, t.grad.?.data);
    const lr = try Tensor(f32).initScalar(a, 0.01);
    for (0..1000) |_| {
        g.reset();
        g.resetGrads();
        out.grad.?.data[0] = 1;
        g.compute();
        t.grad.?.computeMul(t.grad.?, lr);
        t.computeSub(t, t.grad.?);
    }
    try testing.expectApproxEqAbs(@as(f32, 0), out.data[0], 0.00001);
}

test "backward - sqrt" {
    // f(x) = sqrt(x), f'(x) = 0.5 / sqrt(x)
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 4);
    x.setParam(a);
    const out = x.sqrt(a);
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectApproxEqAbs(@as(f32, 2.0), out.data[0], 1e-6);
    // f'(4) = 0.5 / sqrt(4) = 0.25
    try testing.expectApproxEqAbs(@as(f32, 0.25), x.grad.?.data[0], 1e-6);
}

test "backward - abs" {
    // f(x) = abs(x), f'(x) = sgn(x)
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&[_]f32{ -3, 0, 5 });
    x.setParam(a);
    const out = x.abs(a).sumAll(a);
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectApproxEqAbs(@as(f32, 8.0), out.data[0], 1e-6);
    try testing.expectEqualSlices(f32, &.{ -1, 0, 1 }, x.grad.?.data);
}

test "backward - neg" {
    // f(x) = -x, f'(x) = -1
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 7);
    x.setParam(a);
    const out = x.neg(a);
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectApproxEqAbs(@as(f32, -7.0), out.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -1.0), x.grad.?.data[0], 1e-6);
}

test "backward - relu" {
    // f(x) = relu(x), f'(x) = step(x)
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{4});
    x.setData(&[_]f32{ -2, 0, 3, 5 });
    x.setParam(a);
    const out = x.relu(a).sumAll(a);
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    // relu(-2)=0, relu(0)=0, relu(3)=3, relu(5)=5 → sum=8
    try testing.expectApproxEqAbs(@as(f32, 8.0), out.data[0], 1e-6);
    // gradients: step(-2)=0, step(0)=0, step(3)=1, step(5)=1
    try testing.expectEqualSlices(f32, &.{ 0, 0, 1, 1 }, x.grad.?.data);
}

test "backward - recip" {
    // f(x) = 1/x, f'(x) = -1/x^2
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 4);
    x.setParam(a);
    const out = x.recip(a);
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    try testing.expectApproxEqAbs(@as(f32, 0.25), out.data[0], 1e-6);
    // f'(4) = -1/16 = -0.0625
    try testing.expectApproxEqAbs(@as(f32, -0.0625), x.grad.?.data[0], 1e-6);
}

test "fusion - fused compute matches unfused" {
    // Compute x.sub(y).sqr() both with and without fusion, verify identical results
    var g1 = ComputeGraph(f32).init(tac);
    defer g1.deinit();
    const a1 = g1.allocator();
    const x1 = try Tensor(f32).init(a1, &.{4});
    x1.setData(&.{ 1, 2, 3, 4 });
    const y1 = try Tensor(f32).init(a1, &.{4});
    y1.setData(&.{ 4, 3, 2, 1 });
    const out1 = x1.sub(a1, y1).sqr(a1);
    try g1.buildForward(out1);
    g1.compute();

    var g2 = ComputeGraph(f32).init(tac);
    defer g2.deinit();
    const a2 = g2.allocator();
    const x2 = try Tensor(f32).init(a2, &.{4});
    x2.setData(&.{ 1, 2, 3, 4 });
    const y2 = try Tensor(f32).init(a2, &.{4});
    y2.setData(&.{ 4, 3, 2, 1 });
    const out2 = x2.sub(a2, y2).sqr(a2);
    try g2.buildForward(out2);
    try g2.fusionPass();
    g2.compute();

    for (out1.data, out2.data) |v1, v2| {
        try testing.expectApproxEqAbs(v1, v2, 1e-6);
    }
    try testing.expectEqualSlices(f32, &.{ 9, 1, 1, 9 }, out2.data);
}

test "fusion - backward works after fusion" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).initScalar(a, 3);
    x.setParam(a);
    const y = try Tensor(f32).initScalar(a, 2);
    const out = x.mul(a, y).neg(a); // mul -> neg: fusible chain of 2
    try g.buildForward(out);
    try g.fusionPass();
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    // f(x) = -(x * 2) = -2x, f'(x) = -2
    try testing.expectApproxEqAbs(@as(f32, -6.0), out.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -2.0), x.grad.?.data[0], 1e-6);
}

test "fusion - chain detection skips non-fusible ops" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{ 2, 3 });
    x.setData(&.{ 1, 2, 3, 4, 5, 6 });
    // neg -> sumAll: neg is fusible but sum breaks the chain
    const out = x.neg(a).sumAll(a);
    try g.buildForward(out);
    try g.fusionPass();

    try testing.expectEqual(@as(usize, 0), g.fused_chains.items.len);
    g.compute();
    try testing.expectApproxEqAbs(@as(f32, -21.0), out.data[0], 1e-6);
}

test "fusion - detects softmax pattern" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const out = x.softmax(a, &.{1});
    try g.buildForward(out);
    try g.fusionPass();

    try testing.expectEqual(@as(usize, 1), g.fused_chains.items.len);
    try testing.expectEqual(fused.FusionKind.softmax, g.fused_chains.items[0].kind);

    g.compute();
    try testing.expectApproxEqAbs(@as(f32, 1.0), out.data[0] + out.data[1] + out.data[2], 1e-6);
}

test "fusion - detects logSoftmax pattern" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ 1, 2, 3 });
    const out = x.logSoftmax(a, &.{1});
    try g.buildForward(out);
    try g.fusionPass();

    try testing.expectEqual(@as(usize, 1), g.fused_chains.items.len);
    try testing.expectEqual(fused.FusionKind.log_softmax, g.fused_chains.items[0].kind);

    g.compute();
    const probs = std.math.exp(out.data[0]) + std.math.exp(out.data[1]) + std.math.exp(out.data[2]);
    try testing.expectApproxEqAbs(@as(f32, 1.0), probs, 1e-6);
}

test "fusion - detects cross entropy pattern" {
    const loss = @import("loss.zig");

    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const logits = try Tensor(f32).init(a, &.{ 3, 2 });
    logits.setData(&.{
        2.0, 0.0, 1.0,
        0.0, 3.0, 1.0,
    });
    const targets = try Tensor(f32).init(a, &.{2});
    targets.setData(&.{ 0, 1 });

    const out = loss.crossEntropy(f32, a, logits, targets);
    try g.buildForward(out);
    try g.fusionPass();

    try testing.expectEqual(@as(usize, 1), g.fused_chains.items.len);
    // Fusion pass may detect log_softmax subpattern within cross-entropy
    try testing.expect(g.fused_chains.items[0].kind == .cross_entropy or
        g.fused_chains.items[0].kind == .log_softmax);

    g.compute();

    const row0_sum = std.math.exp(@as(f32, 2.0)) + std.math.exp(@as(f32, 0.0)) + std.math.exp(@as(f32, 1.0));
    const row1_sum = std.math.exp(@as(f32, 0.0)) + std.math.exp(@as(f32, 3.0)) + std.math.exp(@as(f32, 1.0));
    const expected = (-std.math.log(f32, std.math.e, std.math.exp(@as(f32, 2.0)) / row0_sum) -
        std.math.log(f32, std.math.e, std.math.exp(@as(f32, 3.0)) / row1_sum)) / 2.0;
    try testing.expectApproxEqAbs(expected, out.data[0], 1e-5);
}

test "layerNorm forward" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{4});
    x.setData(&.{ 1, 2, 3, 4 });
    // mean=2.5, var=1.25, result = (x - 2.5) / sqrt(1.25 + 1e-5)
    const out = x.layerNorm(a, &.{1}, 1e-5);
    try g.buildForward(out);
    g.compute();

    const std_dev = @sqrt(@as(f32, 1.25) + 1e-5);
    try testing.expectApproxEqAbs((1.0 - 2.5) / std_dev, out.data[0], 1e-4);
    try testing.expectApproxEqAbs((2.0 - 2.5) / std_dev, out.data[1], 1e-4);
    try testing.expectApproxEqAbs((3.0 - 2.5) / std_dev, out.data[2], 1e-4);
    try testing.expectApproxEqAbs((4.0 - 2.5) / std_dev, out.data[3], 1e-4);
}

test "backward - gelu" {
    // Numerical gradient check: (gelu(x+h) - gelu(x-h)) / 2h ≈ gelu'(x)
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{3});
    x.setData(&.{ -1.0, 0.0, 1.5 });
    x.setParam(a);
    const out = x.gelu(a).sumAll(a);
    try g.buildForward(out);
    try g.buildBackward(false);
    _ = out.grad.?.setAllScalar(1);
    g.compute();

    // Compute expected gradients numerically
    const h: f32 = 1e-4;
    const test_vals = [_]f32{ -1.0, 0.0, 1.5 };
    for (test_vals, 0..) |xv, i| {
        const gelu_plus = geluScalar(xv + h);
        const gelu_minus = geluScalar(xv - h);
        const numerical_grad = (gelu_plus - gelu_minus) / (2.0 * h);
        try testing.expectApproxEqAbs(numerical_grad, x.grad.?.data[i], 1e-3);
    }
}

fn geluScalar(x: f32) f32 {
    const a = 0.79788456080286535587989211986876 * x * (1.0 + 0.044715 * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(a));
}

//#endregion
