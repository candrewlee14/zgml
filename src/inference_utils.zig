//! Shared utilities for inference plans (GPT and LLaMA).
//!
//! Contains workspace optimisation (liveness analysis + buffer reuse) and
//! quantized-matmul dispatch helpers used by both `InferencePlan` and
//! `LlamaInferencePlan`.

const std = @import("std");

const Tensor = @import("tensor.zig").Tensor;
const ComputeGraph = @import("graph.zig").ComputeGraph;
const QuantizedWeight = @import("quant.zig").QuantizedWeight;

/// Analyse tensor liveness across the forward schedule and assign
/// shared workspace buffers so that dead temporaries are recycled.
pub fn optimizeWorkspace(comptime T: type, graph: *ComputeGraph(T), alloc: std.mem.Allocator, out_bufs: *[][]T) !void {
    const nodes = graph.nodes.items[0..graph.forward_node_count];
    if (nodes.len == 0) return;

    // Build pointer → index map.
    var ptr_to_idx = std.AutoHashMap(*Tensor(T), u32).init(alloc);
    defer ptr_to_idx.deinit();
    try ptr_to_idx.ensureTotalCapacity(@intCast(nodes.len));
    for (nodes, 0..) |node, i| ptr_to_idx.putAssumeCapacity(node, @intCast(i));

    // Compute last-use step for each node.
    const last_use = try alloc.alloc(u32, nodes.len);
    defer alloc.free(last_use);
    for (0..nodes.len) |i| last_use[i] = @intCast(i);

    for (nodes, 0..) |node, step| {
        inline for (.{ node.src0, node.src1 }) |maybe_src| {
            if (maybe_src) |s| {
                if (ptr_to_idx.get(s)) |idx| {
                    last_use[idx] = @max(last_use[idx], @as(u32, @intCast(step)));
                }
            }
        }
    }

    // Mark tensors that are parents of views — their data buffers
    // must not be relocated because child views hold raw pointers
    // into them.
    const is_view_parent = try alloc.alloc(bool, nodes.len);
    defer alloc.free(is_view_parent);
    @memset(is_view_parent, false);
    for (nodes) |node| {
        const op = node.opTag();
        if (op == .as_strided or op == .view) {
            if (node.src0) |parent| {
                if (ptr_to_idx.get(parent)) |idx| is_view_parent[idx] = true;
            }
        }
    }

    // Identify workspace-eligible nodes: intermediate, owned data,
    // not a view parent, not fused-skip.
    const eligible = try alloc.alloc(bool, nodes.len);
    defer alloc.free(eligible);
    for (nodes, 0..) |node, i| {
        eligible[i] = node.opTag() != .none and
            node.ownsData() and
            !is_view_parent[i] and
            !(i < graph.fused_skip.items.len and graph.fused_skip.items[i]);
    }

    // Slot assignment: best-fit (smallest adequate free slot).
    const Slot = struct { free_after: u32, size: usize };
    var slots: std.ArrayList(Slot) = .empty;
    defer slots.deinit(alloc);

    const assignment = try alloc.alloc(i32, nodes.len);
    defer alloc.free(assignment);
    @memset(assignment, -1);

    for (nodes, 0..) |node, step_usize| {
        if (!eligible[step_usize]) continue;
        const step: u32 = @intCast(step_usize);
        const size = node.nElems();

        var best: ?usize = null;
        for (slots.items, 0..) |slot, si| {
            if (slot.free_after < step and slot.size >= size) {
                if (best == null or slot.size < slots.items[best.?].size) {
                    best = si;
                }
            }
        }

        if (best) |si| {
            slots.items[si].free_after = last_use[step_usize];
            assignment[step_usize] = @intCast(si);
        } else {
            assignment[step_usize] = @intCast(slots.items.len);
            try slots.append(alloc, .{ .free_after = last_use[step_usize], .size = size });
        }
    }

    if (slots.items.len == 0) return;

    // Allocate workspace buffers and redirect tensor data pointers.
    const bufs = try alloc.alloc([]T, slots.items.len);
    errdefer {
        for (bufs) |b| if (b.len > 0) alloc.free(b);
        alloc.free(bufs);
    }
    for (slots.items, 0..) |slot, si| {
        bufs[si] = try alloc.alloc(T, slot.size);
    }

    for (nodes, 0..) |node, i| {
        if (assignment[i] < 0) continue;
        const si: usize = @intCast(assignment[i]);
        node.data = bufs[si][0..node.nElems()];
    }

    out_bufs.* = bufs;
}

/// Check if a matmul node has at least one parameter (weight) source.
pub fn isWeightMatmul(comptime T: type, node: *Tensor(T)) bool {
    if (node.src0) |s| {
        if (s.isParam()) return true;
    }
    if (node.src1) |s| {
        if (s.isParam()) return true;
    }
    return false;
}

/// Dispatch a quantized matmul for a node whose weight has been quantized.
pub fn executeQuantizedMatmul(comptime T: type, node: *Tensor(T), qw: *const QuantizedWeight(T)) void {
    const src0 = node.src0.?;
    const src1 = node.src1.?;
    const flags = node.matmul_flags;

    const input = if (src1.isParam()) src0 else src1;
    const M = if (flags.trans0) src0.ne[0] else src0.ne[1];
    const N = if (flags.trans1) src1.ne[1] else src1.ne[0];
    const K = if (flags.trans0) src0.ne[1] else src0.ne[0];

    qw.matmul(input.data, node.data, M, N, K);
}
