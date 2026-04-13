//! Forward computation implementations for tensor operations.
//!
//! Each `compute*` function executes the actual math for one operation,
//! reading from source tensors and writing into the destination tensor.
//! The `compute` dispatcher routes based on `tensor.op`.
//!
//! Element-wise ops use portable SIMD via Zig's @Vector for throughput.
//! Matrix multiplication uses cache-friendly tiling with SIMD accumulation.

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

// ---------------------------------------------------------------------------
// SIMD primitives
// ---------------------------------------------------------------------------

/// Returns the optimal SIMD vector width (in lanes) for type T.
/// Targets 256-bit vectors (e.g. AVX2), minimum 4 lanes.
/// LLVM will split wider vectors automatically on narrower hardware.
fn simdVecSize(comptime T: type) comptime_int {
    const lanes = 32 / @sizeOf(T);
    return if (lanes >= 4) lanes else 4;
}

/// Apply a unary SIMD operation across contiguous data, including the scalar tail.
fn simdMapUnary(
    comptime T: type,
    src: []const T,
    dst: []T,
    comptime vecFn: fn (@Vector(simdVecSize(T), T)) @Vector(simdVecSize(T), T),
    comptime scalarFn: fn (T) T,
) void {
    const vec_size = comptime simdVecSize(T);
    const len = src.len;
    var i: usize = 0;
    while (i + vec_size <= len) : (i += vec_size) {
        const v: @Vector(vec_size, T) = src[i..][0..vec_size].*;
        dst[i..][0..vec_size].* = vecFn(v);
    }
    while (i < len) : (i += 1) {
        dst[i] = scalarFn(src[i]);
    }
}

/// How the two operands of a binary op are supplied.
const BinaryMode = enum { vec_vec, scalar_lhs, scalar_rhs };

/// Apply a binary SIMD operation across contiguous data, including the scalar tail.
/// In `scalar_lhs` / `scalar_rhs` mode the scalar operand is taken from index 0
/// of the corresponding slice and broadcast across every lane.
fn simdMapBinary(
    comptime T: type,
    comptime mode: BinaryMode,
    lhs: []const T,
    rhs: []const T,
    dst: []T,
    comptime vecFn: fn (@Vector(simdVecSize(T), T), @Vector(simdVecSize(T), T)) @Vector(simdVecSize(T), T),
    comptime scalarFn: fn (T, T) T,
) void {
    const vec_size = comptime simdVecSize(T);
    const len = dst.len;

    var i: usize = 0;
    switch (mode) {
        .vec_vec => {
            while (i + vec_size <= len) : (i += vec_size) {
                const vl: @Vector(vec_size, T) = lhs[i..][0..vec_size].*;
                const vr: @Vector(vec_size, T) = rhs[i..][0..vec_size].*;
                dst[i..][0..vec_size].* = vecFn(vl, vr);
            }
            while (i < len) : (i += 1) dst[i] = scalarFn(lhs[i], rhs[i]);
        },
        .scalar_lhs => {
            const sv: @Vector(vec_size, T) = @splat(lhs[0]);
            const s = lhs[0];
            while (i + vec_size <= len) : (i += vec_size) {
                const vr: @Vector(vec_size, T) = rhs[i..][0..vec_size].*;
                dst[i..][0..vec_size].* = vecFn(sv, vr);
            }
            while (i < len) : (i += 1) dst[i] = scalarFn(s, rhs[i]);
        },
        .scalar_rhs => {
            const sv: @Vector(vec_size, T) = @splat(rhs[0]);
            const s = rhs[0];
            while (i + vec_size <= len) : (i += vec_size) {
                const vl: @Vector(vec_size, T) = lhs[i..][0..vec_size].*;
                dst[i..][0..vec_size].* = vecFn(vl, sv);
            }
            while (i < len) : (i += 1) dst[i] = scalarFn(lhs[i], s);
        },
    }
}

// ---------------------------------------------------------------------------
// Forward compute operations parameterized on the tensor type
// ---------------------------------------------------------------------------

pub fn Ops(comptime Self: type, comptime T: type) type {
    const vec_size = comptime simdVecSize(T);
    const Vec = @Vector(vec_size, T);

    return struct {
        // -- Vector map functions ----------------------------------------

        fn addVec(a: Vec, b: Vec) Vec { return a + b; }
        fn subVec(a: Vec, b: Vec) Vec { return a - b; }
        fn mulVec(a: Vec, b: Vec) Vec { return a * b; }
        fn divVec(a: Vec, b: Vec) Vec { return a / b; }
        fn sqrVec(v: Vec) Vec { return v * v; }
        fn sqrtVec(v: Vec) Vec { return @sqrt(v); }
        fn absVec(v: Vec) Vec { return @abs(v); }
        fn negVec(v: Vec) Vec { return -v; }
        fn recipVec(v: Vec) Vec {
            const one: Vec = @splat(@as(T, 1.0));
            return one / v;
        }

        // -- Scalar map functions ----------------------------------------

        fn addScalar(a: T, b: T) T { return a + b; }
        fn subScalar(a: T, b: T) T { return a - b; }
        fn mulScalar(a: T, b: T) T { return a * b; }
        fn divScalar(a: T, b: T) T { return a / b; }
        fn sqrScalar(x: T) T { return x * x; }
        fn sqrtScalar(x: T) T { return std.math.sqrt(x); }
        fn absScalar(x: T) T { return @abs(x); }
        fn negScalar(x: T) T { return -x; }
        fn recipScalar(x: T) T { return 1.0 / x; }

        // -- Vectorized tanh (3,3) Pade approximant ----------------------

        fn tanhApprox(x: Vec) Vec {
            const lo: Vec = @splat(@as(T, -4.97));
            const hi: Vec = @splat(@as(T, 4.97));
            const xc = @min(@max(x, lo), hi);
            const x2 = xc * xc;
            const x4 = x2 * x2;
            const x6 = x4 * x2;
            const c135135: Vec = @splat(@as(T, 135135.0));
            const c17325: Vec = @splat(@as(T, 17325.0));
            const c378: Vec = @splat(@as(T, 378.0));
            const c62370: Vec = @splat(@as(T, 62370.0));
            const c3150: Vec = @splat(@as(T, 3150.0));
            const c28: Vec = @splat(@as(T, 28.0));
            const num = xc * (c135135 + c17325 * x2 + c378 * x4 + x6);
            const den = c135135 + c62370 * x2 + c3150 * x4 + c28 * x6;
            return num / den;
        }

        // ---------------------------------------------------------------
        // Element-wise unary ops
        // ---------------------------------------------------------------

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

        pub fn computeSqr(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            simdMapUnary(T, src0.data, dst.data, sqrVec, sqrScalar);
        }

        pub fn computeSqrt(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            simdMapUnary(T, src0.data, dst.data, sqrtVec, sqrtScalar);
        }

        /// Element-wise reciprocal: dst[i] = 1 / src0[i].
        pub fn computeRecip(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            simdMapUnary(T, src0.data, dst.data, recipVec, recipScalar);
        }

        pub fn computeAbs(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            simdMapUnary(T, src0.data, dst.data, absVec, absScalar);
        }

        pub fn computeNeg(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            simdMapUnary(T, src0.data, dst.data, negVec, negScalar);
        }

        /// Element-wise sign: -1, 0, or 1.  Uses @select for branchless SIMD.
        pub fn computeSgn(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            const zero: Vec = @splat(0);
            const one: Vec = @splat(1);
            const neg_one: Vec = @splat(-1);
            const len = src0.data.len;
            var i: usize = 0;
            while (i + vec_size <= len) : (i += vec_size) {
                const v: Vec = src0.data[i..][0..vec_size].*;
                dst.data[i..][0..vec_size].* = @select(T, v > zero, one, zero) +
                    @select(T, v < zero, neg_one, zero);
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

        /// Element-wise GeLU with vectorized tanh approximation.
        pub fn computeGelu(dst: *Self, src0: *Self) void {
            assert(dst.isSameShape(src0));
            const half: Vec = @splat(@as(T, 0.5));
            const one: Vec = @splat(@as(T, 1.0));
            const coef_a: Vec = @splat(@as(T, GELU_COEF_A));
            const s2op: Vec = @splat(@as(T, SQRT_2_OVER_PI));
            const len = src0.data.len;
            var i: usize = 0;
            while (i + vec_size <= len) : (i += vec_size) {
                const x: Vec = src0.data[i..][0..vec_size].*;
                const inner = s2op * x * (one + coef_a * x * x);
                dst.data[i..][0..vec_size].* = half * x * (one + tanhApprox(inner));
            }
            while (i < len) : (i += 1) {
                const x = src0.data[i];
                dst.data[i] = 0.5 * x * (1.0 + std.math.tanh(SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)));
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

        // ---------------------------------------------------------------
        // Element-wise binary ops (with scalar broadcasting)
        // ---------------------------------------------------------------

        pub fn computeAdd(dst: *Self, src0: *Self, src1: *Self) void {
            if (src0.isSameShape(src1)) {
                assert(dst.isSameShape(src0));
                simdMapBinary(T, .vec_vec, src0.data, src1.data, dst.data, addVec, addScalar);
            } else if (src1.isScalar()) {
                assert(dst.isSameShape(src0));
                simdMapBinary(T, .scalar_rhs, src0.data, src1.data, dst.data, addVec, addScalar);
            } else if (src0.isScalar()) {
                assert(dst.isSameShape(src1));
                simdMapBinary(T, .scalar_lhs, src0.data, src1.data, dst.data, addVec, addScalar);
            }
        }

        pub fn computeSub(dst: *Self, src0: *Self, src1: *Self) void {
            if (src0.isSameShape(src1)) {
                assert(dst.isSameShape(src0));
                simdMapBinary(T, .vec_vec, src0.data, src1.data, dst.data, subVec, subScalar);
            } else if (src1.isScalar()) {
                assert(dst.isSameShape(src0));
                simdMapBinary(T, .scalar_rhs, src0.data, src1.data, dst.data, subVec, subScalar);
            } else if (src0.isScalar()) {
                assert(dst.isSameShape(src1));
                simdMapBinary(T, .scalar_lhs, src0.data, src1.data, dst.data, subVec, subScalar);
            } else {
                @panic("Unimplemented forward sub for src sizes");
            }
        }

        pub fn computeMul(dst: *Self, src0: *Self, src1: *Self) void {
            if (src0.isScalar()) {
                assert(dst.isSameShape(src1));
                simdMapBinary(T, .scalar_lhs, src0.data, src1.data, dst.data, mulVec, mulScalar);
            } else if (src1.isScalar()) {
                assert(dst.isSameShape(src0));
                simdMapBinary(T, .scalar_rhs, src0.data, src1.data, dst.data, mulVec, mulScalar);
            } else {
                assert(dst.isSameShape(src0));
                assert(src0.isSameShape(src1));
                simdMapBinary(T, .vec_vec, src0.data, src1.data, dst.data, mulVec, mulScalar);
            }
        }

        pub fn computeDiv(dst: *Self, src0: *Self, src1: *Self) void {
            if (src0.isScalar()) {
                assert(dst.isSameShape(src1));
                simdMapBinary(T, .scalar_lhs, src0.data, src1.data, dst.data, divVec, divScalar);
            } else if (src1.isScalar()) {
                assert(dst.isSameShape(src0));
                simdMapBinary(T, .scalar_rhs, src0.data, src1.data, dst.data, divVec, divScalar);
            } else {
                assert(dst.isSameShape(src0));
                assert(src0.isSameShape(src1));
                simdMapBinary(T, .vec_vec, src0.data, src1.data, dst.data, divVec, divScalar);
            }
        }

        // ---------------------------------------------------------------
        // Reduction / broadcast ops (stride-indexed, not SIMD'd)
        // ---------------------------------------------------------------

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

        // ---------------------------------------------------------------
        // Matrix multiplication  (tiled, K-blocked, SIMD-accumulated)
        // ---------------------------------------------------------------

        /// Gather vec_size elements from data with stride, starting at offset.
        fn gatherB(data: []const T, base: usize, stride: usize, offset: usize) Vec {
            var v: [vec_size]T = undefined;
            for (0..vec_size) |j| {
                v[j] = data[base + (offset + j) * stride];
            }
            return v;
        }

        fn shouldUseBlasForMatMul(dst: *Self, src0: *Self, src1: *Self) bool {
            return src0.isContiguous() and src1.isContiguous() and
                (dst.ne[0] >= 32 and dst.ne[1] >= 32 and src1.ne[0] >= 32);
        }

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
                        const M = if (trans0) src0_ne0 else src0_ne1;
                        const N = if (trans1) src1_ne1 else src1_ne0;
                        const K = if (trans0) src0_ne1 else src0_ne0;

                        const a_m_stride = if (trans0) src0.strides[0] else src0.strides[1];
                        const a_k_stride = if (trans0) src0.strides[1] else src0.strides[0];
                        const b_n_stride = if (trans1) src1.strides[1] else src1.strides[0];
                        const b_k_stride = if (trans1) src1.strides[0] else src1.strides[1];

                        const a_base = src0_i3 * src0.strides[3] + src0_i2 * src0.strides[2];
                        const b_base = src0_i3 * src1.strides[3] + src0_i2 * src1.strides[2];
                        const d_base = src0_i3 * dst.strides[3] + src0_i2 * dst.strides[2];

                        // Micro-kernel: mr rows x 2*vec_size columns.
                        // 12 accumulator regs + 2 B + 1 A broadcast = 15, fits in 16 YMM.
                        // kc sized for L1 cache (~32KB working set).
                        const mr = 6;
                        const nr = 2 * vec_size; // 16 for f32
                        const kc = 128;
                        const b_contig = (b_n_stride == 1);

                        var mi: usize = 0;
                        while (mi < M) : (mi += mr) {
                            const m_end = @min(mi + mr, M);

                            // --- Wide tiles (2 vectors per row) ---
                            var ni: usize = 0;
                            while (ni + nr <= N) : (ni += nr) {
                                var acc0 = [_]Vec{@as(Vec, @splat(0))} ** mr;
                                var acc1 = [_]Vec{@as(Vec, @splat(0))} ** mr;

                                var ki: usize = 0;
                                while (ki < K) : (ki += kc) {
                                    const k_end = @min(ki + kc, K);
                                    for (ki..k_end) |k| {
                                        const b_off = b_base + k * b_k_stride + ni * b_n_stride;
                                        const b0: Vec = if (b_contig)
                                            src1.data[b_off..][0..vec_size].*
                                        else
                                            gatherB(src1.data, b_off, b_n_stride, 0);
                                        const b1: Vec = if (b_contig)
                                            src1.data[(b_off + vec_size)..][0..vec_size].*
                                        else
                                            gatherB(src1.data, b_off, b_n_stride, vec_size);

                                        for (mi..m_end, 0..) |m, ml| {
                                            const av: Vec = @splat(src0.data[a_base + m * a_m_stride + k * a_k_stride]);
                                            acc0[ml] = @mulAdd(Vec, av, b0, acc0[ml]);
                                            acc1[ml] = @mulAdd(Vec, av, b1, acc1[ml]);
                                        }
                                    }
                                }

                                for (mi..m_end, 0..) |m, ml| {
                                    const rb = d_base + m * dst.strides[1] + ni;
                                    dst.data[rb..][0..vec_size].* = acc0[ml];
                                    dst.data[(rb + vec_size)..][0..vec_size].* = acc1[ml];
                                }
                            }

                            // --- Single-vector remainder ---
                            if (ni + vec_size <= N) {
                                var acc = [_]Vec{@as(Vec, @splat(0))} ** mr;
                                var ki: usize = 0;
                                while (ki < K) : (ki += kc) {
                                    const k_end = @min(ki + kc, K);
                                    for (ki..k_end) |k| {
                                        const b_off = b_base + k * b_k_stride + ni * b_n_stride;
                                        const bv: Vec = if (b_contig)
                                            src1.data[b_off..][0..vec_size].*
                                        else
                                            gatherB(src1.data, b_off, b_n_stride, 0);
                                        for (mi..m_end, 0..) |m, ml| {
                                            const av: Vec = @splat(src0.data[a_base + m * a_m_stride + k * a_k_stride]);
                                            acc[ml] = @mulAdd(Vec, av, bv, acc[ml]);
                                        }
                                    }
                                }
                                for (mi..m_end, 0..) |m, ml| {
                                    const rb = d_base + m * dst.strides[1] + ni;
                                    dst.data[rb..][0..vec_size].* = acc[ml];
                                }
                                ni += vec_size;
                            }

                            // --- Scalar tail ---
                            while (ni < N) : (ni += 1) {
                                for (mi..m_end) |m| {
                                    var s: T = 0;
                                    for (0..K) |k| {
                                        s = @mulAdd(T, src0.data[a_base + m * a_m_stride + k * a_k_stride], src1.data[b_base + k * b_k_stride + ni * b_n_stride], s);
                                    }
                                    dst.data[d_base + m * dst.strides[1] + ni] = s;
                                }
                            }
                        }
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // Dispatch
        // ---------------------------------------------------------------

        /// Dispatch forward computation for this tensor's primitive op.
        pub fn compute(tensor: *Self) void {
            const src0 = tensor.src0;
            const src1 = tensor.src1;
            switch (tensor.op) {
                .none, .view, .reshape, .transpose => {},
                .add => tensor.computeAdd(src0.?, src1.?),
                .mul => tensor.computeMul(src0.?, src1.?),
                .neg => tensor.computeNeg(src0.?),
                .abs => tensor.computeAbs(src0.?),
                .sgn => tensor.computeSgn(src0.?),
                .step => tensor.computeStep(src0.?),
                .sqrt => tensor.computeSqrt(src0.?),
                .recip => tensor.computeRecip(src0.?),
                .gelu => tensor.computeGelu(src0.?),
                .sum => tensor.computeSum(src0.?),
                .repeat => tensor.computeRepeat(src0.?),
                .matmul => tensor.computeMatMul(src0.?, false, src1.?, false),
                .matmul_t0 => tensor.computeMatMul(src0.?, true, src1.?, false),
                .matmul_t1 => tensor.computeMatMul(src0.?, false, src1.?, true),
                .matmul_t0t1 => tensor.computeMatMul(src0.?, true, src1.?, true),
            }
        }
    };
}
