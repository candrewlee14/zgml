//! Comptime model definition and compilation (inference only).
//!
//! When the model architecture is known at compile time, the execution
//! plan is baked into the binary — zero runtime graph construction.
//!
//! ```
//! const Model = ComptimeModel(f32, struct {
//!     pub fn define(b: *Builder) void {
//!         const x = b.input(.{ 3, 2 });
//!         const y = b.gelu(b.linear(x, .{ 2, 3 }, true));
//!         b.output(y);
//!     }
//! });
//! var model = try Model.init(allocator);
//! defer model.deinit();
//! model.forward(&input_data, &output_data);
//! ```
//!
//! ## Supported ops
//! Elementwise: neg, exp, log, sqrt, recip, abs, step, gelu
//! Binary: add, mul (with cyclic broadcast for different-sized operands)
//! Matmul: uses optimized SIMD-tiled kernel (row-major, no transpose)
//! Layers: linear (matmul + optional bias)
//! Reduction: sum
//!
//! ## Limitations
//! - Inference only (no backward pass / training)
//! - Matmul assumes row-major layout, no transpose flags
//! - No conv2d, softmax, layerNorm, maxPool2d
//! - No weight serialization (params must be filled manually)
//! - No connection to the dynamic ComputeGraph

const std = @import("std");
const tensorlib = @import("tensor.zig");
const Tensor = tensorlib.Tensor;
const max_dims = tensorlib.max_dims;
const Op = @import("op.zig").Op;
const fused = @import("tensor/fused.zig");
const fwd = @import("tensor/forward.zig");

// ---------------------------------------------------------------------------
// Comptime graph recording
// ---------------------------------------------------------------------------

/// A node in the comptime computation graph. No data — just shape and op info.
const Node = struct {
    shape: [max_dims]usize,
    op: Op,
    src0: ?usize = null, // index into node list
    src1: ?usize = null,
    is_param: bool = false,
    is_input: bool = false,
};

/// Maximum nodes in a comptime model graph.
const max_nodes = 256;

/// Comptime graph builder. Records operations without allocating any data.
pub const Builder = struct {
    nodes: [max_nodes]Node = undefined,
    count: usize = 0,

    pub fn input(self: *Builder, comptime dims: anytype) usize {
        return self.addNode(.{
            .shape = normShape(dims),
            .op = .none,
            .is_input = true,
        });
    }

    pub fn param(self: *Builder, comptime dims: anytype) usize {
        return self.addNode(.{
            .shape = normShape(dims),
            .op = .none,
            .is_param = true,
        });
    }

    // -- Elementwise ops --

    pub fn add(self: *Builder, a: usize, b: usize) usize {
        return self.binary(.add, a, b);
    }

    pub fn mul(self: *Builder, a: usize, b: usize) usize {
        return self.binary(.mul, a, b);
    }

    pub fn neg(self: *Builder, a: usize) usize {
        return self.unary(.neg, a);
    }

    pub fn exp(self: *Builder, a: usize) usize {
        return self.unary(.exp, a);
    }

    pub fn log(self: *Builder, a: usize) usize {
        return self.unary(.log, a);
    }

    pub fn gelu(self: *Builder, a: usize) usize {
        return self.unary(.gelu, a);
    }

    pub fn relu(self: *Builder, a: usize) usize {
        const step = self.unary(.step, a);
        return self.binary(.mul, a, step);
    }

    pub fn sqrt(self: *Builder, a: usize) usize {
        return self.unary(.sqrt, a);
    }

    pub fn recip(self: *Builder, a: usize) usize {
        return self.unary(.recip, a);
    }

    // -- Higher-level ops --

    pub fn matmul(self: *Builder, a: usize, b: usize) usize {
        const sa = self.nodes[a].shape;
        const sb = self.nodes[b].shape;
        var out_shape = sa;
        out_shape[0] = sb[0]; // N = cols of B
        // out_shape[1] stays = rows of A
        return self.addNode(.{
            .shape = out_shape,
            .op = .matmul,
            .src0 = a,
            .src1 = b,
        });
    }

    /// Linear layer: matmul(x, w) + bias
    pub fn linear(self: *Builder, x: usize, weight_dims: anytype, bias: bool) usize {
        const w = self.param(weight_dims);
        const mm = self.matmul(x, w);
        if (bias) {
            const b_shape = normShape(weight_dims);
            const b = self.param(.{b_shape[0]});
            // repeat bias to match matmul output, then add
            const rep = self.addNode(.{
                .shape = self.nodes[mm].shape,
                .op = .repeat,
                .src0 = b,
            });
            return self.binary(.add, mm, rep);
        }
        return mm;
    }

    pub fn sumAll(self: *Builder, a: usize) usize {
        return self.addNode(.{
            .shape = [_]usize{1} ** max_dims,
            .op = .sum,
            .src0 = a,
        });
    }

    pub fn output(self: *Builder, node: usize) void {
        _ = self;
        _ = node;
        // Mark as output — the forward function returns this node's data
    }

    // -- Internal --

    fn unary(self: *Builder, op: Op, a: usize) usize {
        return self.addNode(.{
            .shape = self.nodes[a].shape,
            .op = op,
            .src0 = a,
        });
    }

    fn binary(self: *Builder, op: Op, a: usize, b: usize) usize {
        return self.addNode(.{
            .shape = self.nodes[a].shape, // assume same shape (broadcast handled separately)
            .op = op,
            .src0 = a,
            .src1 = b,
        });
    }

    fn addNode(self: *Builder, node: Node) usize {
        const idx = self.count;
        self.nodes[idx] = node;
        self.count += 1;
        return idx;
    }

    fn normShape(comptime dims: anytype) [max_dims]usize {
        comptime {
            var result = [_]usize{1} ** max_dims;
            const fields = @typeInfo(@TypeOf(dims)).@"struct".fields;
            for (fields, 0..) |f, i| {
                result[i] = @field(dims, f.name);
            }
            return result;
        }
    }

    // -- Analysis --

    pub fn countParams(self: *const Builder) usize {
        var count: usize = 0;
        for (self.nodes[0..self.count]) |n| {
            if (n.is_param) count += 1;
        }
        return count;
    }

    pub fn countOps(self: *const Builder) usize {
        var count: usize = 0;
        for (self.nodes[0..self.count]) |n| {
            if (n.op != .none) count += 1;
        }
        return count;
    }

    pub fn totalParamElems(self: *const Builder) usize {
        var total: usize = 0;
        for (self.nodes[0..self.count]) |n| {
            if (n.is_param) {
                var elems: usize = 1;
                for (n.shape) |d| elems *= d;
                total += elems;
            }
        }
        return total;
    }
};

// ---------------------------------------------------------------------------
// Comptime model type generator
// ---------------------------------------------------------------------------

/// Generate a model type from a comptime graph definition.
///
/// `Def` must have a `pub fn define(b: *Builder) void` method.
/// The returned type has:
///   - `init(allocator)` — allocate runtime buffers
///   - `deinit()` — free everything
///   - `forward(input, output)` — run the model
///   - `param_count`, `op_count` — comptime-known stats
pub fn ComptimeModel(comptime T: type, comptime Def: type) type {
    // Run the builder at comptime to capture the graph
    const graph = comptime blk: {
        var b = Builder{};
        Def.define(&b);
        break :blk b;
    };

    const n_nodes = graph.count;
    const n_params = comptime graph.countParams();
    const n_ops = comptime graph.countOps();
    const total_param_elems = comptime graph.totalParamElems();

    return struct {
        const Self = @This();
        const nodes = graph.nodes;

        pub const param_count = n_params;
        pub const op_count = n_ops;
        pub const node_count = n_nodes;
        pub const param_elems = total_param_elems;

        /// Runtime state: actual tensor data
        allocator: std.mem.Allocator,
        buffers: [n_nodes]?[]T,
        param_data: [n_params][]T,

        pub fn init(alloc: std.mem.Allocator) !Self {
            var self = Self{
                .allocator = alloc,
                .buffers = .{null} ** n_nodes,
                .param_data = undefined,
            };

            // Allocate parameter buffers
            var pi: usize = 0;
            inline for (0..n_nodes) |i| {
                if (comptime nodes[i].is_param) {
                    const elems = comptime nodeElems(i);
                    self.param_data[pi] = try alloc.alloc(T, elems);
                    self.buffers[i] = self.param_data[pi];
                    pi += 1;
                }
            }

            // Allocate intermediate buffers (not params, not inputs)
            inline for (0..n_nodes) |i| {
                if (comptime !nodes[i].is_param and !nodes[i].is_input and nodes[i].op != .none) {
                    const elems = comptime nodeElems(i);
                    self.buffers[i] = try alloc.alloc(T, elems);
                }
            }

            return self;
        }

        pub fn deinit(self: *Self) void {
            for (self.param_data) |p| self.allocator.free(p);
            inline for (0..n_nodes) |i| {
                if (comptime !nodes[i].is_param and !nodes[i].is_input and nodes[i].op != .none) {
                    if (self.buffers[i]) |buf| self.allocator.free(buf);
                }
            }
        }

        /// Run the forward pass. Input data is copied in, output is written to out_buf.
        pub fn forward(self: *Self, input_data: []const T, out_buf: []T) void {
            // Set input buffer
            inline for (0..n_nodes) |i| {
                if (comptime nodes[i].is_input) {
                    // Input node — point to provided data (cast away const for uniform access)
                    self.buffers[i] = @constCast(input_data);
                }
            }

            // Execute ops in order (nodes are topologically sorted by construction)
            inline for (0..n_nodes) |i| {
                const node = comptime nodes[i];
                if (comptime node.op == .none) continue;

                const dst = self.buffers[i].?;
                const src0 = if (comptime node.src0) |s| self.buffers[s].? else undefined;

                switch (comptime node.op) {
                    .neg => for (dst, src0) |*d, s| { d.* = -s; },
                    .exp => for (dst, src0) |*d, s| { d.* = @exp(s); },
                    .log => for (dst, src0) |*d, s| { d.* = @log(s); },
                    .sqrt => for (dst, src0) |*d, s| { d.* = @sqrt(s); },
                    .recip => for (dst, src0) |*d, s| { d.* = 1.0 / s; },
                    .abs => for (dst, src0) |*d, s| { d.* = @abs(s); },
                    .step => for (dst, src0) |*d, s| { d.* = if (s > 0) 1.0 else 0.0; },
                    .gelu => for (dst, src0) |*d, s| {
                        const c = @as(T, @sqrt(2.0 / std.math.pi));
                        d.* = 0.5 * s * (1.0 + std.math.tanh(c * s * (1.0 + 0.044715 * s * s)));
                    },
                    .add => {
                        const src1 = self.buffers[node.src1.?].?;
                        if (src1.len == dst.len) {
                            for (dst, src0, src1) |*d, a, b| d.* = a + b;
                        } else {
                            // Broadcast: src1 is smaller, repeat cyclically
                            for (dst, src0, 0..) |*d, a, j| d.* = a + src1[j % src1.len];
                        }
                    },
                    .mul => {
                        const src1 = self.buffers[node.src1.?].?;
                        if (src1.len == dst.len) {
                            for (dst, src0, src1) |*d, a, b| d.* = a * b;
                        } else {
                            for (dst, src0, 0..) |*d, a, j| d.* = a * src1[j % src1.len];
                        }
                    },
                    .sum => {
                        // Simple sum-all for now
                        var acc: T = 0;
                        for (src0) |s| acc += s;
                        dst[0] = acc;
                    },
                    .repeat => {
                        // Simple repeat: tile src0 into dst
                        for (dst, 0..) |*d, j| d.* = src0[j % src0.len];
                    },
                    .matmul => {
                        const src1 = self.buffers[node.src1.?].?;
                        const s = comptime nodes[i].shape;
                        const s0 = comptime nodes[node.src0.?].shape;
                        const N = s[0]; // output cols
                        const M = s[1]; // output rows
                        const K = s0[0]; // inner dim
                        // Use optimized SIMD-tiled kernel
                        const kernel = fwd.selectMatMulKernel(T);
                        kernel(dst, src0, src1, M, N, K, K, 1, N, 1, 0, 0, 0, N);
                    },
                    else => {},
                }
            }

            // Copy output (last non-input, non-param node)
            const last_buf = comptime blk: {
                var last: usize = 0;
                for (0..n_nodes) |i| {
                    if (nodes[i].op != .none) last = i;
                }
                break :blk last;
            };
            const result = self.buffers[last_buf].?;
            @memcpy(out_buf[0..result.len], result);
        }

        fn nodeElems(comptime i: usize) usize {
            comptime {
                var e: usize = 1;
                for (nodes[i].shape) |d| e *= d;
                return e;
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "comptime model - simple elementwise" {
    const Model = ComptimeModel(f32, struct {
        pub fn define(b: *Builder) void {
            const x = b.input(.{3});
            const y = b.exp(x);
            b.output(y);
        }
    });

    try std.testing.expectEqual(@as(usize, 0), Model.param_count);
    try std.testing.expectEqual(@as(usize, 1), Model.op_count);

    var model = try Model.init(std.testing.allocator);
    defer model.deinit();

    const input = [_]f32{ 1.0, 2.0, 3.0 };
    var output: [3]f32 = undefined;
    model.forward(&input, &output);

    try std.testing.expectApproxEqAbs(@exp(@as(f32, 1.0)), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@exp(@as(f32, 2.0)), output[1], 1e-6);
    try std.testing.expectApproxEqAbs(@exp(@as(f32, 3.0)), output[2], 1e-6);
}

test "comptime model - chain of ops" {
    const Model = ComptimeModel(f32, struct {
        pub fn define(b: *Builder) void {
            const x = b.input(.{4});
            const y = b.neg(b.exp(x));
            b.output(y);
        }
    });

    try std.testing.expectEqual(@as(usize, 2), Model.op_count);

    var model = try Model.init(std.testing.allocator);
    defer model.deinit();

    const input = [_]f32{ 0.0, 1.0, -1.0, 2.0 };
    var output: [4]f32 = undefined;
    model.forward(&input, &output);

    for (input, output) |x, y| {
        try std.testing.expectApproxEqAbs(-@exp(x), y, 1e-6);
    }
}

test "comptime model - matmul" {
    const Model = ComptimeModel(f32, struct {
        pub fn define(b: *Builder) void {
            const x = b.input(.{ 2, 2 }); // 2x2 input
            const w = b.param(.{ 2, 2 }); // 2x2 weight
            const y = b.matmul(x, w);
            b.output(y);
        }
    });

    try std.testing.expectEqual(@as(usize, 1), Model.param_count);
    try std.testing.expectEqual(@as(usize, 4), Model.param_elems);

    var model = try Model.init(std.testing.allocator);
    defer model.deinit();

    // Set weights to identity
    model.param_data[0][0] = 1;
    model.param_data[0][1] = 0;
    model.param_data[0][2] = 0;
    model.param_data[0][3] = 1;

    const input = [_]f32{ 1, 2, 3, 4 };
    var output: [4]f32 = undefined;
    model.forward(&input, &output);

    // Identity matmul should return input
    try std.testing.expectApproxEqAbs(@as(f32, 1), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), output[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), output[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), output[3], 1e-6);
}

test "comptime model - linear layer end-to-end" {
    // Linear: y = x @ W + b
    // x: [2, 1] (2 features, 1 sample)
    // W: [3, 2] (project 2→3)
    // b: [3]
    const Model = ComptimeModel(f32, struct {
        pub fn define(b: *Builder) void {
            const x = b.input(.{ 2, 1 });
            const y = b.linear(x, .{ 3, 2 }, true);
            b.output(y);
        }
    });

    try std.testing.expectEqual(@as(usize, 2), Model.param_count); // weight + bias

    var model = try Model.init(std.testing.allocator);
    defer model.deinit();

    // W = [[1, 0], [0, 1], [1, 1]] — projects [a,b] → [a, b, a+b]
    model.param_data[0][0] = 1;
    model.param_data[0][1] = 0;
    model.param_data[0][2] = 0;
    model.param_data[0][3] = 1;
    model.param_data[0][4] = 1;
    model.param_data[0][5] = 1;
    // b = [0.5, -0.5, 0]
    model.param_data[1][0] = 0.5;
    model.param_data[1][1] = -0.5;
    model.param_data[1][2] = 0.0;

    const input = [_]f32{ 2.0, 3.0 };
    var output: [3]f32 = undefined;
    model.forward(&input, &output);

    // Expected: [2+0.5, 3-0.5, 2+3+0] = [2.5, 2.5, 5.0]
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), output[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), output[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), output[2], 1e-5);
}
