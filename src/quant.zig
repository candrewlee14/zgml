//! Weight-only int8 quantization for inference.
//!
//! Stores weights as int8 with per-block f32 scales. During matmul,
//! int8 weights are dequantized on the fly — 4x memory reduction with
//! minimal accuracy loss for inference.
//!
//! ```
//! var qw = try QuantizedWeight(f32).fromTensor(allocator, weight_tensor, 32);
//! defer qw.deinit(allocator);
//! qw.matmul(input_data, output_data, M, N, K);
//! ```

const std = @import("std");

/// Block size for quantization scales. Each block of `block_size` int8 values
/// shares one f32 scale factor.
pub const default_block_size: usize = 32;

pub fn QuantizedWeight(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []i8,
        scales: []T,
        rows: usize, // K (inner dim)
        cols: usize, // N (output dim)
        block_size: usize,

        /// Quantize a f32 weight matrix to int8 with per-block scaling.
        /// Weight layout: [K, N] (row-major, K rows, N cols).
        pub fn fromSlice(alloc: std.mem.Allocator, weights: []const T, rows: usize, cols: usize, block_size: usize) !Self {
            std.debug.assert(weights.len == rows * cols);
            const n_elems = rows * cols;
            const n_blocks = (n_elems + block_size - 1) / block_size;

            const data = try alloc.alloc(i8, n_elems);
            errdefer alloc.free(data);
            const scales = try alloc.alloc(T, n_blocks);
            errdefer alloc.free(scales);

            // Quantize each block
            for (0..n_blocks) |b| {
                const start = b * block_size;
                const end = @min(start + block_size, n_elems);
                const block = weights[start..end];

                // Find max absolute value in block
                var max_abs: T = 0;
                for (block) |v| {
                    const a = @abs(v);
                    if (a > max_abs) max_abs = a;
                }

                const scale = if (max_abs > 0) max_abs / 127.0 else 1.0;
                scales[b] = scale;
                const inv_scale = if (max_abs > 0) 127.0 / max_abs else 0.0;

                for (block, start..) |v, j| {
                    const q = v * inv_scale;
                    data[j] = @intFromFloat(std.math.clamp(q, -127.0, 127.0));
                }
            }

            return .{
                .data = data,
                .scales = scales,
                .rows = rows,
                .cols = cols,
                .block_size = block_size,
            };
        }

        /// Quantize from a col-major 2D Tensor.
        ///
        /// Col-major [ne0, ne1] has the same flat layout as row-major [ne1, ne0].
        /// The quantized weight stores [K=ne1, N=ne0] for matmul dispatch.
        pub fn fromTensor(alloc: std.mem.Allocator, tensor: anytype, block_size: usize) !Self {
            return fromSlice(alloc, tensor.data, tensor.ne[1], tensor.ne[0], block_size);
        }

        pub fn deinit(self: Self, alloc: std.mem.Allocator) void {
            alloc.free(self.data);
            alloc.free(self.scales);
        }

        /// Dequantize a single element.
        fn dequant(self: *const Self, idx: usize) T {
            const block = idx / self.block_size;
            return @as(T, @floatFromInt(self.data[idx])) * self.scales[block];
        }

        /// Quantized matmul: dst[m,n] = sum_k(input[m,k] * dequant(weight[k,n]))
        /// input: [M, K] row-major f32
        /// weight (self): [K, N] row-major int8
        /// dst: [M, N] row-major f32
        pub fn matmul(self: *const Self, input: []const T, dst: []T, M: usize, N: usize, K: usize) void {
            std.debug.assert(self.rows == K);
            std.debug.assert(self.cols == N);
            std.debug.assert(input.len >= M * K);
            std.debug.assert(dst.len >= M * N);

            const bs = self.block_size;

            for (0..M) |m| {
                for (0..N) |n| {
                    var acc: T = 0;
                    // Process in blocks for better scale reuse
                    var k: usize = 0;
                    while (k < K) {
                        const block_end = @min(k + bs, K);
                        // All elements in this column-block share a scale
                        // (if block_size divides K; otherwise approximate)
                        while (k < block_end) : (k += 1) {
                            const w_idx = k * N + n;
                            const w_val = self.dequant(w_idx);
                            acc += input[m * K + k] * w_val;
                        }
                    }
                    dst[m * N + n] = acc;
                }
            }
        }

        /// Quantized matmul with bias addition.
        pub fn matmulBias(self: *const Self, input: []const T, bias: []const T, dst: []T, M: usize, N: usize, K: usize) void {
            self.matmul(input, dst, M, N, K);
            // Add bias (broadcast over M rows)
            for (0..M) |m| {
                for (0..N) |n| {
                    dst[m * N + n] += bias[n];
                }
            }
        }

        /// Compute quantization error (RMSE) vs original weights.
        pub fn quantizationError(self: *const Self, original: []const T) T {
            var sum_sq: T = 0;
            for (original, 0..) |orig, j| {
                const diff = orig - self.dequant(j);
                sum_sq += diff * diff;
            }
            return @sqrt(sum_sq / @as(T, @floatFromInt(original.len)));
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "quantize and dequantize preserves values approximately" {
    const alloc = testing.allocator;
    const weights = [_]f32{ 1.0, -0.5, 0.25, -1.0, 0.0, 0.75, -0.3, 0.9 };

    var qw = try QuantizedWeight(f32).fromSlice(alloc, &weights, 2, 4, 4);
    defer qw.deinit(alloc);

    // Quantization error should be small
    const err = qw.quantizationError(&weights);
    try testing.expect(err < 0.01);
}

test "quantized matmul matches float matmul approximately" {
    const alloc = testing.allocator;

    // Weight: 3x2 (K=3, N=2)
    const weights = [_]f32{
        1.0,  0.5,
        -0.5, 1.0,
        0.25, -0.25,
    };
    // Input: 2x3 (M=2, K=3)
    const input = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };

    // Reference float matmul
    var ref_output: [4]f32 = undefined;
    for (0..2) |m| {
        for (0..2) |n| {
            var acc: f32 = 0;
            for (0..3) |k| {
                acc += input[m * 3 + k] * weights[k * 2 + n];
            }
            ref_output[m * 2 + n] = acc;
        }
    }

    // Quantized matmul
    var qw = try QuantizedWeight(f32).fromSlice(alloc, &weights, 3, 2, 32);
    defer qw.deinit(alloc);

    var q_output: [4]f32 = undefined;
    qw.matmul(&input, &q_output, 2, 2, 3);

    // Should be close (within quantization error)
    for (ref_output, q_output) |r, q| {
        try testing.expectApproxEqAbs(r, q, 0.1);
    }
}

test "quantized matmul with bias" {
    const alloc = testing.allocator;
    const weights = [_]f32{ 1.0, 0.0, 0.0, 1.0 }; // 2x2 identity
    const bias = [_]f32{ 0.5, -0.5 };
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 }; // 2x2

    var qw = try QuantizedWeight(f32).fromSlice(alloc, &weights, 2, 2, 32);
    defer qw.deinit(alloc);

    var output: [4]f32 = undefined;
    qw.matmulBias(&input, &bias, &output, 2, 2, 2);

    // Identity + bias: [1.5, 1.5, 3.5, 3.5]
    try testing.expectApproxEqAbs(@as(f32, 1.5), output[0], 0.1);
    try testing.expectApproxEqAbs(@as(f32, 1.5), output[1], 0.1);
    try testing.expectApproxEqAbs(@as(f32, 3.5), output[2], 0.1);
    try testing.expectApproxEqAbs(@as(f32, 3.5), output[3], 0.1);
}

test "block size affects quantization error" {
    const alloc = testing.allocator;
    // Random-ish weights with varying magnitudes
    const weights = [_]f32{ 0.1, 10.0, -0.01, 5.0, 0.5, -8.0, 0.001, 3.0 };

    // Smaller blocks = better accuracy (each block has its own scale)
    var qw_large = try QuantizedWeight(f32).fromSlice(alloc, &weights, 1, 8, 8);
    defer qw_large.deinit(alloc);
    var qw_small = try QuantizedWeight(f32).fromSlice(alloc, &weights, 1, 8, 2);
    defer qw_small.deinit(alloc);

    const err_large = qw_large.quantizationError(&weights);
    const err_small = qw_small.quantizationError(&weights);

    // Smaller blocks should have less error
    try testing.expect(err_small <= err_large);
}
