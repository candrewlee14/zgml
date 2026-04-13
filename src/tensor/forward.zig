//! Forward computation implementations for tensor operations.
//!
//! Each `compute*` function executes the actual math for one operation,
//! reading from source tensors and writing into the destination tensor.
//! The `compute` dispatcher routes based on `tensor.op`.
//!
//! Element-wise ops use portable SIMD via Zig's @Vector for throughput.

const std = @import("std");
const assert = std.debug.assert;
const builtin = @import("builtin");
const Op = @import("../op.zig").Op;
const opts = @import("zgml_options");

const c = if (opts.use_blas)
    switch (builtin.os.tag) {
        .linux, .windows => @cImport(@cInclude("cblas.h")),
        .macos => @cImport(@cInclude("Accelerate/Accelerate.h")),
        else => @cImport(@compileError("Unsupported OS")),
    }
else
    void;

const max_dims = 4;
const GELU_COEF_A = 0.044715;
const SQRT_2_OVER_PI = 0.79788456080286535587989211986876;

/// Returns the optimal SIMD vector width (in lanes) for type T.
/// Targets 256-bit vectors (e.g. AVX2), minimum 4 lanes.
/// LLVM will split wider vectors automatically on narrower hardware.
fn simdVecSize(comptime T: type) comptime_int {
    const lanes = 32 / @sizeOf(T);
    return if (lanes >= 4) lanes else 4;
}

/// Process contiguous data with a SIMD unary operation.
/// Returns the index where SIMD processing stopped; caller handles the tail.
fn simdUnaryLoop(
    comptime T: type,
    src: []const T,
    dst: []T,
    comptime mapFn: fn (@Vector(simdVecSize(T), T)) @Vector(simdVecSize(T), T),
) usize {
    const vec_size = comptime simdVecSize(T);
    const len = src.len;
    var i: usize = 0;
    while (i + vec_size <= len) : (i += vec_size) {
        const v: @Vector(vec_size, T) = src[i..][0..vec_size].*;
        dst[i..][0..vec_size].* = mapFn(v);
    }
    return i;
}

/// Process contiguous data with a SIMD binary operation.
/// Returns the index where SIMD processing stopped; caller handles the tail.
fn simdBinaryLoop(
    comptime T: type,
    src0: []const T,
    src1: []const T,
    dst: []T,
    comptime mapFn: fn (@Vector(simdVecSize(T), T), @Vector(simdVecSize(T), T)) @Vector(simdVecSize(T), T),
) usize {
    const vec_size = comptime simdVecSize(T);
    const len = src0.len;
    var i: usize = 0;
    while (i + vec_size <= len) : (i += vec_size) {
        const v0: @Vector(vec_size, T) = src0[i..][0..vec_size].*;
        const v1: @Vector(vec_size, T) = src1[i..][0..vec_size].*;
        dst[i..][0..vec_size].* = mapFn(v0, v1);
    }
    return i;
}

/// SIMD scalar-broadcast loop: apply `op(src[i], scalar)` across the source.
/// Returns the index where SIMD processing stopped.
fn simdScalarRhsLoop(
    comptime T: type,
    src: []const T,
    scalar: T,
    dst: []T,
    comptime mapFn: fn (@Vector(simdVecSize(T), T), @Vector(simdVecSize(T), T)) @Vector(simdVecSize(T), T),
) usize {
    const vec_size = comptime simdVecSize(T);
    const sv: @Vector(vec_size, T) = @splat(scalar);
    const len = src.len;
    var i: usize = 0;
    while (i + vec_size <= len) : (i += vec_size) {
        const v: @Vector(vec_size, T) = src[i..][0..vec_size].*;
        dst[i..][0..vec_size].* = mapFn(v, sv);
    }
    return i;
}

/// SIMD scalar-broadcast loop: apply `op(scalar, src[i])` across the source.
/// Returns the index where SIMD processing stopped.
fn simdScalarLhsLoop(
    comptime T: type,
    scalar: T,
    src: []const T,
    dst: []T,
    comptime mapFn: fn (@Vector(simdVecSize(T), T), @Vector(simdVecSize(T), T)) @Vector(simdVecSize(T), T),
) usize {
    const vec_size = comptime simdVecSize(T);
    const sv: @Vector(vec_size, T) = @splat(scalar);
    const len = src.len;
    var i: usize = 0;
    while (i + vec_size <= len) : (i += vec_size) {
        const v: @Vector(vec_size, T) = src[i..][0..vec_size].*;
        dst[i..][0..vec_size].* = mapFn(sv, v);
    }
    return i;
}

/// Forward compute operations parameterized on the tensor type.
pub fn Ops(comptime Self: type, comptime T: type) type {
    const vec_size = comptime simdVecSize(T);
    const Vec = @Vector(vec_size, T);

    return struct {
        // -- SIMD map functions for use with the loop helpers --

        fn addVec(a: Vec, b: Vec) Vec {
            return a + b;
        }
        fn subVec(a: Vec, b: Vec) Vec {
            return a - b;
        }
        fn mulVec(a: Vec, b: Vec) Vec {
            return a * b;
        }
        fn divVec(a: Vec, b: Vec) Vec {
            return a / b;
        }
        fn sqrVec(v: Vec) Vec {
            return v * v;
        }
        fn sqrtVec(v: Vec) Vec {
            return @sqrt(v);
        }
        fn absVec(v: Vec) Vec {
            return @abs(v);
        }
        fn negVec(v: Vec) Vec {
            return -v;
        }

        // -----------------------------------------------------------
        // Element-wise unary ops
        // -----------------------------------------------------------

        /// Copy src0 into dst. Both must be contiguous and same size.
        pub fn computeDup(dst: *Self, src0: *Self) void {
            assert(dst.isContiguous());
            assert(dst.nElems() == src0.nElems());
            if (src0.isContiguous()) {
                @memcpy(dst.data, src0.data);
                return;
            }
            @panic("Unimplemented forward dup for non-contiguous src");
        }

        /// Element-wise square: dst[i] = src0[i]^2
        pub fn computeSqr(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            const i = simdUnaryLoop(T, src0.data, dst.data, sqrVec);
            for (src0.data[i..], dst.data[i..]) |s, *d| d.* = s * s;
        }

        /// Element-wise square root.
        pub fn computeSqrt(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            const i = simdUnaryLoop(T, src0.data, dst.data, sqrtVec);
            for (src0.data[i..], dst.data[i..]) |s, *d| d.* = std.math.sqrt(s);
        }

        /// Element-wise absolute value.
        pub fn computeAbs(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            const i = simdUnaryLoop(T, src0.data, dst.data, absVec);
            for (src0.data[i..], dst.data[i..]) |s, *d| d.* = @abs(s);
        }

        /// Element-wise negation.
        pub fn computeNeg(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            const i = simdUnaryLoop(T, src0.data, dst.data, negVec);
            for (src0.data[i..], dst.data[i..]) |s, *d| d.* = -s;
        }

        /// Element-wise sign: returns -1, 0, or 1.  Uses @select for branchless SIMD.
        pub fn computeSgn(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            const zero: Vec = @splat(0);
            const one: Vec = @splat(1);
            const neg_one: Vec = @splat(-1);
            const len = src0.data.len;
            var i: usize = 0;
            while (i + vec_size <= len) : (i += vec_size) {
                const v: Vec = src0.data[i..][0..vec_size].*;
                const pos = @select(T, v > zero, one, zero);
                const neg = @select(T, v < zero, neg_one, zero);
                dst.data[i..][0..vec_size].* = pos + neg;
            }
            while (i < len) : (i += 1) {
                const s = src0.data[i];
                dst.data[i] = if (s > 0) 1 else if (s < 0) @as(T, -1) else 0;
            }
        }

        /// Element-wise step function: 1 if positive, 0 otherwise.
        pub fn computeStep(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            const zero: Vec = @splat(0);
            const one: Vec = @splat(1);
            const len = src0.data.len;
            var i: usize = 0;
            while (i + vec_size <= len) : (i += vec_size) {
                const v: Vec = src0.data[i..][0..vec_size].*;
                dst.data[i..][0..vec_size].* = @select(T, v > zero, one, zero);
            }
            while (i < len) : (i += 1) {
                dst.data[i] = if (src0.data[i] > 0) 1 else 0;
            }
        }

        /// Element-wise ReLU: max(0, x).  Uses @select for branchless SIMD.
        pub fn computeRelu(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            const zero: Vec = @splat(0);
            const len = src0.data.len;
            var i: usize = 0;
            while (i + vec_size <= len) : (i += vec_size) {
                const v: Vec = src0.data[i..][0..vec_size].*;
                dst.data[i..][0..vec_size].* = @select(T, v > zero, v, zero);
            }
            while (i < len) : (i += 1) {
                const s = src0.data[i];
                dst.data[i] = if (s > 0) s else 0;
            }
        }

        /// Element-wise GeLU approximation (scalar — tanh not yet vectorized).
        pub fn computeGelu(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            for (src0.data, dst.data) |x, *d| {
                d.* = 0.5 * x * (1 + std.math.tanh(SQRT_2_OVER_PI * x * (1 + GELU_COEF_A * x * x)));
            }
        }

        pub fn computeNorm(dst: *Self, src0: *Self) void {
            _ = src0;
            _ = dst;
            @panic("Not implemented");
        }

        pub fn computeRMSNorm(dst: *Self, src0: *Self) void {
            _ = src0;
            _ = dst;
            @panic("Not implemented");
        }

        // -----------------------------------------------------------
        // Element-wise binary ops (with scalar broadcasting)
        // -----------------------------------------------------------

        /// Element-wise addition. Supports same-shape and scalar broadcasting.
        pub fn computeAdd(dst: *Self, src0: *Self, src1: *Self) void {
            if (src0.isSameShape(src1)) {
                assert(dst.isSameShape(src0));
                const i = simdBinaryLoop(T, src0.data, src1.data, dst.data, addVec);
                for (src0.data[i..], src1.data[i..], dst.data[i..]) |a, b, *d| d.* = a + b;
            } else if (src1.isScalar()) {
                assert(dst.isSameShape(src0));
                const i = simdScalarRhsLoop(T, src0.data, src1.data[0], dst.data, addVec);
                for (src0.data[i..], dst.data[i..]) |a, *d| d.* = a + src1.data[0];
            } else if (src0.isScalar()) {
                assert(dst.isSameShape(src1));
                const i = simdScalarLhsLoop(T, src0.data[0], src1.data, dst.data, addVec);
                for (src1.data[i..], dst.data[i..]) |b, *d| d.* = src0.data[0] + b;
            }
        }

        /// Element-wise subtraction. Supports same-shape and scalar broadcasting.
        pub fn computeSub(dst: *Self, src0: *Self, src1: *Self) void {
            if (src0.isSameShape(src1)) {
                assert(dst.isSameShape(src0));
                const i = simdBinaryLoop(T, src0.data, src1.data, dst.data, subVec);
                for (src0.data[i..], src1.data[i..], dst.data[i..]) |a, b, *d| d.* = a - b;
            } else if (src1.isScalar()) {
                assert(dst.isSameShape(src0));
                const i = simdScalarRhsLoop(T, src0.data, src1.data[0], dst.data, subVec);
                for (src0.data[i..], dst.data[i..]) |a, *d| d.* = a - src1.data[0];
            } else if (src0.isScalar()) {
                assert(dst.isSameShape(src1));
                const i = simdScalarLhsLoop(T, src0.data[0], src1.data, dst.data, subVec);
                for (src1.data[i..], dst.data[i..]) |b, *d| d.* = src0.data[0] - b;
            } else {
                @panic("Unimplemented forward sub for src sizes");
            }
        }

        /// Element-wise multiplication. Supports same-shape and scalar broadcasting.
        pub fn computeMul(dst: *Self, src0: *Self, src1: *Self) void {
            if (src0.isScalar()) {
                assert(dst.isSameShape(src1));
                const i = simdScalarLhsLoop(T, src0.data[0], src1.data, dst.data, mulVec);
                for (src1.data[i..], dst.data[i..]) |b, *d| d.* = src0.data[0] * b;
            } else if (src1.isScalar()) {
                assert(dst.isSameShape(src0));
                const i = simdScalarRhsLoop(T, src0.data, src1.data[0], dst.data, mulVec);
                for (src0.data[i..], dst.data[i..]) |a, *d| d.* = a * src1.data[0];
            } else {
                assert(dst.isSameShape(src0));
                assert(src0.isSameShape(src1));
                const i = simdBinaryLoop(T, src0.data, src1.data, dst.data, mulVec);
                for (src0.data[i..], src1.data[i..], dst.data[i..]) |a, b, *d| d.* = a * b;
            }
        }

        /// Element-wise division. Supports same-shape and scalar broadcasting.
        pub fn computeDiv(dst: *Self, src0: *Self, src1: *Self) void {
            if (src0.isScalar()) {
                assert(dst.isSameShape(src1));
                const i = simdScalarLhsLoop(T, src0.data[0], src1.data, dst.data, divVec);
                for (src1.data[i..], dst.data[i..]) |b, *d| d.* = src0.data[0] / b;
            } else if (src1.isScalar()) {
                assert(dst.isSameShape(src0));
                const i = simdScalarRhsLoop(T, src0.data, src1.data[0], dst.data, divVec);
                for (src0.data[i..], dst.data[i..]) |a, *d| d.* = a / src1.data[0];
            } else {
                assert(dst.isSameShape(src0));
                assert(src0.isSameShape(src1));
                const i = simdBinaryLoop(T, src0.data, src1.data, dst.data, divVec);
                for (src0.data[i..], src1.data[i..], dst.data[i..]) |a, b, *d| d.* = a / b;
            }
        }

        // -----------------------------------------------------------
        // Reduction / broadcast ops (stride-indexed, not SIMD'd)
        // -----------------------------------------------------------

        /// Inverse-broadcast mean: average src0 elements into dst's smaller shape.
        /// Divides each contribution to avoid overflow.
        pub fn computeMean(dst: *Self, src0: *Self) void {
            assert(max_dims == 4);
            assert(src0.canSumTo(dst));
            const src0_ne_v: @Vector(4, usize) = src0.ne;
            const div_elems: T = @floatFromInt(@reduce(.Mul, src0_ne_v / dst.ne));

            for (0..src0.ne[3]) |ne3| {
                for (0..src0.ne[2]) |ne2| {
                    for (0..src0.ne[1]) |ne1| {
                        for (0..src0.ne[0]) |ne0| {
                            const src0_nes = @Vector(4, usize){ ne0, ne1, ne2, ne3 };
                            const dst_ne_v: @Vector(4, usize) = dst.ne;
                            const dst_nes = src0_nes % dst_ne_v;
                            const src0_stride_v: @Vector(4, usize) = src0.strides;
                            const dst_stride_v: @Vector(4, usize) = dst.strides;

                            const src0_idx = @reduce(.Add, src0_nes * src0_stride_v);
                            const dst_idx = @reduce(.Add, dst_nes * dst_stride_v);

                            dst.data[dst_idx] += src0.data[src0_idx] / div_elems;
                        }
                    }
                }
            }
        }

        /// Inverse-broadcast sum: accumulate src0 elements into dst's smaller shape.
        pub fn computeSum(dst: *Self, src0: *Self) void {
            assert(max_dims == 4);
            assert(src0.canSumTo(dst));
            for (0..src0.ne[3]) |ne3| {
                for (0..src0.ne[2]) |ne2| {
                    for (0..src0.ne[1]) |ne1| {
                        for (0..src0.ne[0]) |ne0| {
                            const src0_nes = @Vector(4, usize){ ne0, ne1, ne2, ne3 };
                            const dst_ne_v: @Vector(4, usize) = dst.ne;
                            const dst_nes = src0_nes % dst_ne_v;
                            const src0_stride_v: @Vector(4, usize) = src0.strides;
                            const dst_stride_v: @Vector(4, usize) = dst.strides;

                            const src0_idx = @reduce(.Add, src0_nes * src0_stride_v);
                            const dst_idx = @reduce(.Add, dst_nes * dst_stride_v);

                            dst.data[dst_idx] += src0.data[src0_idx];
                        }
                    }
                }
            }
        }

        /// Broadcast src0 into dst's larger shape by repeating elements.
        pub fn computeRepeat(dst: *Self, src0: *Self) void {
            assert(max_dims == 4);
            assert(src0.canRepeatTo(dst));
            for (0..dst.ne[3]) |ne3| {
                for (0..dst.ne[2]) |ne2| {
                    for (0..dst.ne[1]) |ne1| {
                        for (0..dst.ne[0]) |ne0| {
                            const nes = @Vector(4, usize){ ne0, ne1, ne2, ne3 };
                            const src0_ne_v: @Vector(4, usize) = src0.ne;
                            const src0_nes = nes % src0_ne_v;
                            const src0_stride_v: @Vector(4, usize) = src0.strides;
                            const dst_stride_v: @Vector(4, usize) = dst.strides;

                            const src0_idx = @reduce(.Add, src0_nes * src0_stride_v);
                            const dst_idx = @reduce(.Add, nes * dst_stride_v);

                            dst.data[dst_idx] = src0.data[src0_idx];
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------
        // Matrix multiplication
        // -----------------------------------------------------------

        fn shouldUseBlasForMatMul(dst: *Self, src0: *Self, src1: *Self) bool {
            return src0.isContiguous() and src1.isContiguous() and
                (dst.ne[0] >= 32 and dst.ne[1] >= 32 and src1.ne[0] >= 32);
        }

        /// Validate that dimension constraints for matrix multiplication are satisfied.
        pub fn assertValidMatMulDims(dst: *Self, src0: *Self, trans0: bool, src1: *Self, trans1: bool) void {
            assert(src0.ne[3] == src1.ne[3]);
            assert(src0.ne[2] == src1.ne[2]);
            assert(dst.ne[2] == src0.ne[2]);
            assert(dst.ne[3] == src0.ne[3]);

            if (!trans0 and !trans1) {
                assert(src0.ne[0] == src1.ne[1]);
                assert(dst.ne[1] == src0.ne[1]);
                assert(dst.ne[0] == src1.ne[0]);
            } else if (!trans0 and trans1) {
                assert(src0.ne[0] == src1.ne[0]);
                assert(dst.ne[1] == src0.ne[1]);
                assert(dst.ne[0] == src1.ne[1]);
            } else if (trans0 and !trans1) {
                assert(src0.ne[1] == src1.ne[1]);
                assert(dst.ne[1] == src0.ne[0]);
                assert(dst.ne[0] == src1.ne[0]);
            } else if (trans0 and trans1) {
                assert(src0.ne[1] == src1.ne[0]);
                assert(dst.ne[1] == src0.ne[0]);
                assert(dst.ne[0] == src1.ne[1]);
            }
        }

        /// Matrix multiplication with optional transposition of either operand.
        /// Supports BLAS acceleration when enabled and operands are f32.
        pub fn computeMatMul(dst: *Self, src0: *Self, comptime trans0: bool, src1: *Self, comptime trans1: bool) void {
            assert(max_dims == 4);
            dst.assertValidMatMulDims(src0, trans0, src1, trans1);
            assert(dst.strides[0] == 1);
            assert(dst.strides[0] <= dst.strides[1]);
            assert(dst.strides[1] <= dst.strides[2]);
            assert(dst.strides[2] <= dst.strides[3]);

            const src0_ne3 = src0.ne[3];
            const src0_ne2 = src0.ne[2];
            const src0_ne1 = src0.ne[1];
            const src0_ne0 = src0.ne[0];
            const src1_ne1 = src1.ne[1];
            const src1_ne0 = src1.ne[0];

            const src0_ne1c: c_int = @intCast(src0_ne1);
            const src0_ne0c: c_int = @intCast(src0_ne0);
            const src1_ne1c: c_int = @intCast(src1_ne1);
            const src1_ne0c: c_int = @intCast(src1_ne0);
            const dst_ne0c: c_int = @intCast(dst.ne[0]);

            for (0..src0_ne3) |src0_i3| {
                for (0..src0_ne2) |src0_i2| {
                    if (opts.use_blas and T == f32) {
                        c.cblas_sgemm(
                            c.CblasRowMajor,
                            if (trans0) c.CblasTrans else c.CblasNoTrans,
                            if (trans1) c.CblasTrans else c.CblasNoTrans,
                            if (trans0) src0_ne0c else src0_ne1c,
                            if (trans1) src1_ne1c else src1_ne0c,
                            if (trans0) src0_ne1c else src0_ne0c,
                            1.0,
                            &src0.data[src0_i3 * src0.strides[3] + src0_i2 * src0.strides[2]],
                            src0_ne0c,
                            &src1.data[src0_i3 * src1.strides[3] + src0_i2 * src1.strides[2]],
                            src1_ne0c,
                            0.0,
                            &dst.data[src0_i3 * dst.strides[3] + src0_i2 * dst.strides[2]],
                            dst_ne0c,
                        );
                    } else {
                        // Tiled matmul with SIMD accumulation.
                        //
                        // Determine M, N, K and index strides based on transpose flags.
                        // Since trans0/trans1 are comptime, unused branches are eliminated.
                        const M = if (trans0) src0_ne0 else src0_ne1; // dst rows
                        const N = if (trans1) src1_ne1 else src1_ne0; // dst cols
                        const K = if (trans0) src0_ne1 else src0_ne0; // contraction dim

                        // Strides for walking A[m, k] and B[k, n]:
                        const a_m_stride = if (trans0) src0.strides[0] else src0.strides[1];
                        const a_k_stride = if (trans0) src0.strides[1] else src0.strides[0];
                        const b_n_stride = if (trans1) src1.strides[1] else src1.strides[0];
                        const b_k_stride = if (trans1) src1.strides[0] else src1.strides[1];

                        const a_base = src0_i3 * src0.strides[3] + src0_i2 * src0.strides[2];
                        const b_base = src0_i3 * src1.strides[3] + src0_i2 * src1.strides[2];
                        const d_base = src0_i3 * dst.strides[3] + src0_i2 * dst.strides[2];

                        const tile_n = vec_size;
                        const tile_m = 8;

                        // Process full N-tiles with SIMD
                        var mi: usize = 0;
                        while (mi < M) : (mi += tile_m) {
                            const m_end = @min(mi + tile_m, M);

                            var ni: usize = 0;
                            while (ni + tile_n <= N) : (ni += tile_n) {
                                // Accumulator: one vector per row of the tile
                                var acc: [tile_m]Vec = @splat(@as(Vec, @splat(0)));

                                // Can we do a contiguous vector load on B's N-dimension?
                                const b_contig = (b_n_stride == 1);

                                for (0..K) |k| {
                                    // Load a vector of B[k, ni..ni+tile_n]
                                    const b_row_base = b_base + k * b_k_stride + ni * b_n_stride;
                                    const b_vec: Vec = if (b_contig)
                                        src1.data[b_row_base..][0..tile_n].*
                                    else blk: {
                                        var bv: [tile_n]T = undefined;
                                        for (0..tile_n) |j| {
                                            bv[j] = src1.data[b_row_base + j * b_n_stride];
                                        }
                                        break :blk bv;
                                    };

                                    // Broadcast each A element and multiply-accumulate
                                    for (mi..m_end, 0..) |m, mi_local| {
                                        const a_val = src0.data[a_base + m * a_m_stride + k * a_k_stride];
                                        const a_vec: Vec = @splat(a_val);
                                        acc[mi_local] += a_vec * b_vec;
                                    }
                                }

                                // Write tile to dst
                                for (mi..m_end, 0..) |m, mi_local| {
                                    const row_base = d_base + m * dst.strides[1] + ni;
                                    dst.data[row_base..][0..tile_n].* = acc[mi_local];
                                }
                            }

                            // Scalar tail for remaining N columns
                            while (ni < N) : (ni += 1) {
                                for (mi..m_end) |m| {
                                    var s: T = 0;
                                    for (0..K) |k| {
                                        const a_val = src0.data[a_base + m * a_m_stride + k * a_k_stride];
                                        const b_val = src1.data[b_base + k * b_k_stride + ni * b_n_stride];
                                        s += a_val * b_val;
                                    }
                                    dst.data[d_base + m * dst.strides[1] + ni] = s;
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Dispatch forward computation based on the tensor's operation.
        pub fn compute(tensor: *Self) void {
            const src0 = tensor.src0;
            const src1 = tensor.src1;
            switch (tensor.op) {
                .none, .view => {},
                .dup => tensor.computeDup(src0.?),
                .add => tensor.computeAdd(src0.?, src1.?),
                .sub => tensor.computeSub(src0.?, src1.?),
                .mul => tensor.computeMul(src0.?, src1.?),
                .div => tensor.computeDiv(src0.?, src1.?),
                .repeat => tensor.computeRepeat(src0.?),
                .sqr => tensor.computeSqr(src0.?),
                .sqrt => tensor.computeSqrt(src0.?),
                .sum => tensor.computeSum(src0.?),
                .mean => tensor.computeMean(src0.?),
                .abs => tensor.computeAbs(src0.?),
                .sgn => tensor.computeSgn(src0.?),
                .neg => tensor.computeNeg(src0.?),
                .step => tensor.computeStep(src0.?),
                .relu => tensor.computeRelu(src0.?),
                .gelu => tensor.computeGelu(src0.?),
                .norm => tensor.computeNorm(src0.?),
                .matmul => tensor.computeMatMul(src0.?, false, src1.?, false),
                .matmul_t0 => tensor.computeMatMul(src0.?, true, src1.?, false),
                .matmul_t1 => tensor.computeMatMul(src0.?, false, src1.?, true),
                .matmul_t0t1 => tensor.computeMatMul(src0.?, true, src1.?, true),
                else => @panic("Unimplemented forward OP"),
            }
        }
    };
}
