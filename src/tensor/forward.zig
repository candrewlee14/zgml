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
const opts = if (@hasDecl(@import("root"), "zgml_options"))
    @import("zgml_options")
else
    struct {
        pub const use_blas = false;
    };

const c = if (opts.use_blas)
    switch (builtin.os.tag) {
        .linux, .windows => @cImport(@cInclude("cblas.h")),
        .macos => @cImport(@cInclude("Accelerate/Accelerate.h")),
        else => @cImport(@compileError("Unsupported OS")),
    }
else
    void;

const cpuinfo = @import("cpuinfo.zig");

const max_dims = @import("../tensor.zig").max_dims;
/// Coefficient for the GeLU tanh approximation.
const GELU_COEF_A: comptime_float = 0.044715;
/// √(2/π), computed from std.math.pi.
const SQRT_2_OVER_PI: comptime_float = @sqrt(2.0 / std.math.pi);

// ---------------------------------------------------------------------------
// SIMD primitives
// ---------------------------------------------------------------------------

/// Returns the default SIMD vector width (in lanes) for type T.
/// Used for element-wise ops. Targets 256-bit (conservative portable default).
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

fn elemOffset(strides: [max_dims]usize, c0: usize, c1: usize, c2: usize, c3: usize) usize {
    return c0 * strides[0] + c1 * strides[1] + c2 * strides[2] + c3 * strides[3];
}

fn first4(arr: [max_dims]usize) @Vector(4, usize) {
    return @Vector(4, usize){ arr[0], arr[1], arr[2], arr[3] };
}

/// SIMD contiguous reduction: sum src into dst using chunk-based accumulation.
///
/// `chunk` is the number of contiguous source elements that map to the same
/// destination element (product of leading reduced dimensions). Each chunk
/// is SIMD-reduced horizontally, then accumulated into the correct dst slot.
///
/// For [W,H,C,N] → [1,1,C,1]: chunk = W*H = 576. Each group of 576
/// contiguous elements sums into one of C channel bins.
fn simdReduceSum(comptime T: type, src: []const T, dst: []T, dst_len: usize, chunk: usize) void {
    const vec_size = comptime simdVecSize(T);
    const Vec = @Vector(vec_size, T);

    @memset(dst, 0);
    const num_chunks = src.len / chunk;

    if (chunk >= vec_size) {
        // SIMD horizontal sum per chunk.
        for (0..num_chunks) |ci| {
            const base = ci * chunk;
            var vacc: Vec = @splat(0);
            var j: usize = 0;
            while (j + vec_size <= chunk) : (j += vec_size) {
                vacc += src[base + j ..][0..vec_size].*;
            }
            var acc: T = @reduce(.Add, vacc);
            while (j < chunk) : (j += 1) acc += src[base + j];
            dst[ci % dst_len] += acc;
        }
    } else {
        // Chunk too small for SIMD — scalar accumulation.
        for (0..num_chunks) |ci| {
            const base = ci * chunk;
            var acc: T = 0;
            for (0..chunk) |j| acc += src[base + j];
            dst[ci % dst_len] += acc;
        }
    }
}

/// SIMD contiguous broadcast: expand src into dst using chunk-based tiling.
///
/// `chunk` is the number of contiguous dst elements filled from the same
/// src element (product of leading broadcast dimensions).
///
/// For [1,1,C,1] → [W,H,C,N]: chunk = W*H. Each src[c] fills W*H
/// contiguous destination elements.
fn simdBroadcastRepeat(comptime T: type, src: []const T, dst: []T, src_len: usize, chunk: usize) void {
    const dst_len = dst.len;
    const num_chunks = dst_len / chunk;

    if (chunk == 1) {
        // No leading broadcast dims — just tile src cyclically.
        if (src_len == 1) {
            @memset(dst, src[0]);
        } else {
            for (dst, 0..) |*d, i| d.* = src[i % src_len];
        }
    } else {
        // Fill chunk contiguous elements from each src value.
        for (0..num_chunks) |ci| {
            const base = ci * chunk;
            const val = src[ci % src_len];
            @memset(dst[base..][0..chunk], val);
        }
    }
}

fn nextCoord(coords: []usize, shape: []const usize) bool {
    var axis: usize = 0;
    while (axis < shape.len) : (axis += 1) {
        coords[axis] += 1;
        if (coords[axis] < shape[axis]) return true;
        coords[axis] = 0;
    }
    return false;
}

fn offsetFor(comptime Self: type, tensor: *const Self, coords: []const usize) usize {
    var idx = tensor.storage_offset;
    for (coords, 0..) |coord, i| idx += coord * tensor.strides[i];
    return idx;
}

fn computeStructuralDup(comptime Self: type, dst: *Self, src0: *const Self) void {
    if (dst.nElems() == 0) return;
    var coords: [max_dims]usize = [_]usize{0} ** max_dims;
    while (true) {
        const idx = offsetFor(Self, dst, coords[0..dst.n_dims]);
        dst.data[idx] = src0.data[offsetFor(Self, src0, coords[0..src0.n_dims])];
        if (!nextCoord(coords[0..dst.n_dims], dst.ne[0..dst.n_dims])) break;
    }
}

fn computeBinaryGeneric(comptime Self: type, comptime Tt: type, dst: *Self, src0: *const Self, src1: *const Self, comptime f: fn (Tt, Tt) Tt) void {
    var coords: [max_dims]usize = [_]usize{0} ** max_dims;
    var lhs_coords: [max_dims]usize = [_]usize{0} ** max_dims;
    var rhs_coords: [max_dims]usize = [_]usize{0} ** max_dims;
    while (true) {
        const idx = offsetFor(Self, dst, coords[0..dst.n_dims]);
        var i: usize = 0;
        while (i < dst.n_dims) : (i += 1) {
            lhs_coords[i] = if (i < src0.n_dims and src0.ne[i] != 1) coords[i] else 0;
            rhs_coords[i] = if (i < src1.n_dims and src1.ne[i] != 1) coords[i] else 0;
        }
        dst.data[idx] = f(
            src0.data[offsetFor(Self, src0, lhs_coords[0..src0.n_dims])],
            src1.data[offsetFor(Self, src1, rhs_coords[0..src1.n_dims])],
        );
        if (!nextCoord(coords[0..dst.n_dims], dst.ne[0..dst.n_dims])) break;
    }
}

fn computeReduceGeneric(comptime Self: type, comptime Tt: type, dst: *Self, src0: *const Self, comptime op: enum { sum, max }, mean_divisor: ?Tt) void {
    switch (op) {
        .sum => @memset(dst.data, 0),
        .max => @memset(dst.data, -std.math.inf(Tt)),
    }

    var src_coords: [max_dims]usize = [_]usize{0} ** max_dims;
    var dst_coords: [max_dims]usize = [_]usize{0} ** max_dims;
    while (true) {
        @memset(&dst_coords, 0);
        var i: usize = 0;
        while (i < src0.n_dims) : (i += 1) {
            if (i < dst.n_dims) dst_coords[i] = src_coords[i] % dst.ne[i];
        }
        const src_idx = offsetFor(Self, src0, src_coords[0..src0.n_dims]);
        const dst_idx = offsetFor(Self, dst, dst_coords[0..dst.n_dims]);
        switch (op) {
            .sum => dst.data[dst_idx] += if (mean_divisor) |div| src0.data[src_idx] / div else src0.data[src_idx],
            .max => dst.data[dst_idx] = @max(dst.data[dst_idx], src0.data[src_idx]),
        }
        if (!nextCoord(src_coords[0..src0.n_dims], src0.ne[0..src0.n_dims])) break;
    }
}

fn computeRepeatGeneric(comptime Self: type, dst: *Self, src0: *const Self) void {
    var dst_coords: [max_dims]usize = [_]usize{0} ** max_dims;
    var src_coords: [max_dims]usize = [_]usize{0} ** max_dims;
    while (true) {
        @memset(&src_coords, 0);
        var i: usize = 0;
        while (i < dst.n_dims) : (i += 1) {
            if (i < src0.n_dims) src_coords[i] = dst_coords[i] % src0.ne[i];
        }
        const src_idx = offsetFor(Self, src0, src_coords[0..src0.n_dims]);
        const dst_idx = offsetFor(Self, dst, dst_coords[0..dst.n_dims]);
        dst.data[dst_idx] = src0.data[src_idx];
        if (!nextCoord(dst_coords[0..dst.n_dims], dst.ne[0..dst.n_dims])) break;
    }
}

fn computeScatterAddView(comptime Self: type, dst: *Self, grad_view: *const Self, view_tensor: *const Self) void {
    const base = view_tensor.source0().?;
    std.debug.assert(dst.n_dims == base.n_dims);
    std.debug.assert(dst.hasShape(base.ne[0..base.n_dims]));
    @memset(dst.data, 0);
    var coords: [max_dims]usize = [_]usize{0} ** max_dims;
    while (true) {
        var src_idx: usize = view_tensor.storage_offset;
        var i: usize = 0;
        while (i < view_tensor.n_dims) : (i += 1) src_idx += coords[i] * view_tensor.strides[i];

        var out_idx: usize = grad_view.storage_offset;
        i = 0;
        while (i < grad_view.n_dims) : (i += 1) out_idx += coords[i] * grad_view.strides[i];
        dst.data[src_idx] += grad_view.data[out_idx];

        if (!nextCoord(coords[0..grad_view.n_dims], grad_view.ne[0..grad_view.n_dims])) break;
    }
}

// ---------------------------------------------------------------------------
// Multi-width tiled matmul micro-kernel
// ---------------------------------------------------------------------------

/// Tiled matmul kernel parameterized on vector lane count.
/// Works on raw data slices and strides, decoupled from the Tensor type.
/// Uses FMA, K-blocking, and a 2-vector-wide micro-kernel.
fn TiledMatMul(comptime T: type, comptime vl: comptime_int) type {
    return struct {
        const V = @Vector(vl, T);
        // mr sized to keep 2*mr accumulators in the register file.
        // vl=8 (16 YMM regs): mr=6 → 12 acc + 2 B + 1 A + 1 spare = 16.
        // vl=16 (32 ZMM regs): mr=14 → 28 acc + 2 B + 1 A + 1 spare = 32.
        const mr: usize = if (vl <= 8) 6 else 14;
        const kc: usize = 128;

        pub fn run(
            dst_data: []T,
            src0_data: []const T,
            src1_data: []const T,
            M: usize,
            N: usize,
            K: usize,
            a_m_stride: usize,
            a_k_stride: usize,
            b_k_stride: usize,
            b_n_stride: usize,
            a_base: usize,
            b_base: usize,
            d_base: usize,
            d_row_stride: usize,
        ) void {
            runRange(dst_data, src0_data, src1_data, 0, M, N, K, a_m_stride, a_k_stride, b_k_stride, b_n_stride, a_base, b_base, d_base, d_row_stride);
        }

        /// Execute matmul for a subset of M-rows [m_start, m_limit).
        /// Thread-safe: each thread gets a disjoint row range.
        pub fn runRange(
            dst_data: []T,
            src0_data: []const T,
            src1_data: []const T,
            m_start: usize,
            m_limit: usize,
            N: usize,
            K: usize,
            a_m_stride: usize,
            a_k_stride: usize,
            b_k_stride: usize,
            b_n_stride: usize,
            a_base: usize,
            b_base: usize,
            d_base: usize,
            d_row_stride: usize,
        ) void {
            const nr = 2 * vl;
            const b_contig = (b_n_stride == 1);

            var mi: usize = m_start;
            while (mi < m_limit) : (mi += mr) {
                const m_end = @min(mi + mr, m_limit);

                var ni: usize = 0;
                // --- Wide tiles (2 vectors per row) ---
                while (ni + nr <= N) : (ni += nr) {
                    var acc0: [mr]V = .{@as(V, @splat(0))} ** mr;
                    var acc1: [mr]V = .{@as(V, @splat(0))} ** mr;

                    var ki: usize = 0;
                    while (ki < K) : (ki += kc) {
                        const k_end = @min(ki + kc, K);
                        for (ki..k_end) |k| {
                            const b_off = b_base + k * b_k_stride + ni * b_n_stride;
                            const b0: V = if (b_contig) src1_data[b_off..][0..vl].* else gatherVec(T, vl, src1_data, b_off, b_n_stride, 0);
                            const b1: V = if (b_contig) src1_data[(b_off + vl)..][0..vl].* else gatherVec(T, vl, src1_data, b_off, b_n_stride, vl);
                            for (mi..m_end, 0..) |m, ml| {
                                const av: V = @splat(src0_data[a_base + m * a_m_stride + k * a_k_stride]);
                                acc0[ml] = @mulAdd(V, av, b0, acc0[ml]);
                                acc1[ml] = @mulAdd(V, av, b1, acc1[ml]);
                            }
                        }
                    }
                    for (mi..m_end, 0..) |m, ml| {
                        const rb = d_base + m * d_row_stride + ni;
                        dst_data[rb..][0..vl].* = acc0[ml];
                        dst_data[(rb + vl)..][0..vl].* = acc1[ml];
                    }
                }

                // --- Single-vector remainder ---
                if (ni + vl <= N) {
                    var acc: [mr]V = .{@as(V, @splat(0))} ** mr;
                    var ki: usize = 0;
                    while (ki < K) : (ki += kc) {
                        const k_end = @min(ki + kc, K);
                        for (ki..k_end) |k| {
                            const b_off = b_base + k * b_k_stride + ni * b_n_stride;
                            const bv: V = if (b_contig) src1_data[b_off..][0..vl].* else gatherVec(T, vl, src1_data, b_off, b_n_stride, 0);
                            for (mi..m_end, 0..) |m, ml| {
                                const av: V = @splat(src0_data[a_base + m * a_m_stride + k * a_k_stride]);
                                acc[ml] = @mulAdd(V, av, bv, acc[ml]);
                            }
                        }
                    }
                    for (mi..m_end, 0..) |m, ml| {
                        dst_data[(d_base + m * d_row_stride + ni)..][0..vl].* = acc[ml];
                    }
                    ni += vl;
                }

                // --- Scalar tail ---
                while (ni < N) : (ni += 1) {
                    for (mi..m_end) |m| {
                        var s: T = 0;
                        for (0..K) |k| {
                            s = @mulAdd(T, src0_data[a_base + m * a_m_stride + k * a_k_stride], src1_data[b_base + k * b_k_stride + ni * b_n_stride], s);
                        }
                        dst_data[d_base + m * d_row_stride + ni] = s;
                    }
                }
            }
        }
    };
}

/// Gather vl elements from data[base + (offset+j)*stride] for j in 0..vl.
fn gatherVec(comptime T: type, comptime vl: comptime_int, data: []const T, base: usize, stride: usize, offset: usize) @Vector(vl, T) {
    var v: [vl]T = undefined;
    for (0..vl) |j| v[j] = data[base + (offset + j) * stride];
    return v;
}

/// Matmul function signature for runtime dispatch.
pub fn MatMulFnType(comptime T: type) type {
    return *const fn ([]T, []const T, []const T, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize) void;
}

/// Matmul range function signature — operates on [m_start, m_limit) rows.
pub fn MatMulRangeFnType(comptime T: type) type {
    return *const fn ([]T, []const T, []const T, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize) void;
}

pub fn selectMatMulRangeKernel(comptime T: type) MatMulRangeFnType(T) {
    const State = struct {
        var cached: MatMulRangeFnType(T) = undefined;
        var initialized: bool = false;
    };
    if (State.initialized) return State.cached;

    const width = cpuinfo.detectSimdWidth();
    const lanes = @as(usize, @intFromEnum(width)) / @sizeOf(T);

    State.cached = if (lanes >= 16)
        &TiledMatMul(T, 16).runRange
    else
        &TiledMatMul(T, 8).runRange;
    State.initialized = true;
    return State.cached;
}

/// Select the best matmul kernel at runtime based on CPU SIMD capabilities.
/// Cached after first call.
pub fn selectMatMulKernel(comptime T: type) MatMulFnType(T) {
    const State = struct {
        var cached: MatMulFnType(T) = undefined;
        var initialized: bool = false;
    };
    if (State.initialized) return State.cached;

    const width = cpuinfo.detectSimdWidth();
    const lanes = @as(usize, @intFromEnum(width)) / @sizeOf(T);

    State.cached = if (lanes >= 16)
        &TiledMatMul(T, 16).run
    else
        &TiledMatMul(T, 8).run;
    State.initialized = true;
    return State.cached;
}

// ---------------------------------------------------------------------------
// Forward compute operations parameterized on the tensor type
// ---------------------------------------------------------------------------

pub fn Ops(comptime Self: type, comptime T: type) type {
    const vec_size = comptime simdVecSize(T);
    const Vec = @Vector(vec_size, T);

    return struct {
        // -- Vector map functions ----------------------------------------

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
        fn recipVec(v: Vec) Vec {
            const one: Vec = @splat(@as(T, 1.0));
            return one / v;
        }

        // -- Scalar map functions ----------------------------------------

        fn addScalar(a: T, b: T) T {
            return a + b;
        }
        fn subScalar(a: T, b: T) T {
            return a - b;
        }
        fn mulScalar(a: T, b: T) T {
            return a * b;
        }
        fn divScalar(a: T, b: T) T {
            return a / b;
        }
        fn sqrScalar(x: T) T {
            return x * x;
        }
        fn sqrtScalar(x: T) T {
            return std.math.sqrt(x);
        }
        fn absScalar(x: T) T {
            return @abs(x);
        }
        fn negScalar(x: T) T {
            return -x;
        }
        fn recipScalar(x: T) T {
            return 1.0 / x;
        }
        fn expScalar(x: T) T {
            return std.math.exp(x);
        }
        fn logScalar(x: T) T {
            return std.math.log(T, std.math.e, x);
        }

        fn indexFromValue(v: T) usize {
            if (@typeInfo(T) == .float) {
                assert(v >= 0);
                const iv: usize = @intFromFloat(v);
                assert(@as(T, @floatFromInt(iv)) == v);
                return iv;
            }
            return @intCast(v);
        }

        fn indexAt(indices: *const Self, i: usize) usize {
            if (indices.indexData()) |idx| return idx[i];
            return indexFromValue(indices.data[i]);
        }

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
        pub fn computeDup(dst: *Self, src0: *const Self) void {
            assert(dst.isContiguous());
            assert(dst.nElems() == src0.nElems());
            if (src0.isContiguous()) {
                @memcpy(dst.data, src0.data);
                return;
            }
            @panic("Unimplemented forward dup for non-contiguous src");
        }

        pub fn computeSqrt(dst: *Self, src0: *const Self) void {
            assert(dst.isSameShape(src0));
            simdMapUnary(T, src0.data, dst.data, sqrtVec, sqrtScalar);
        }

        /// Element-wise reciprocal: dst[i] = 1 / src0[i].
        pub fn computeRecip(dst: *Self, src0: *const Self) void {
            assert(dst.isSameShape(src0));
            simdMapUnary(T, src0.data, dst.data, recipVec, recipScalar);
        }

        pub fn computeExp(dst: *Self, src0: *const Self) void {
            assert(dst.isSameShape(src0));
            var i: usize = 0;
            while (i < src0.data.len) : (i += 1) {
                dst.data[i] = expScalar(src0.data[i]);
            }
        }

        pub fn computeLog(dst: *Self, src0: *const Self) void {
            assert(dst.isSameShape(src0));
            var i: usize = 0;
            while (i < src0.data.len) : (i += 1) {
                dst.data[i] = logScalar(src0.data[i]);
            }
        }

        pub fn computeAbs(dst: *Self, src0: *const Self) void {
            assert(dst.isSameShape(src0));
            simdMapUnary(T, src0.data, dst.data, absVec, absScalar);
        }

        pub fn computeNeg(dst: *Self, src0: *const Self) void {
            assert(dst.isSameShape(src0));
            simdMapUnary(T, src0.data, dst.data, negVec, negScalar);
        }

        /// Element-wise sign: -1, 0, or 1.  Uses @select for branchless SIMD.
        pub fn computeSgn(dst: *Self, src0: *const Self) void {
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
        pub fn computeStep(dst: *Self, src0: *const Self) void {
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

        /// Element-wise GeLU with vectorized tanh approximation.
        pub fn computeGelu(dst: *Self, src0: *const Self) void {
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

        pub fn computeNorm(dst: *Self, src0: *const Self) void {
            _ = src0;
            _ = dst;
            @panic("Not implemented");
        }

        pub fn computeRMSNorm(dst: *Self, src0: *const Self) void {
            _ = src0;
            _ = dst;
            @panic("Not implemented");
        }

        // ---------------------------------------------------------------
        // Element-wise binary ops (with scalar broadcasting)
        // ---------------------------------------------------------------

        pub fn computeAdd(dst: *Self, src0: *const Self, src1: *const Self) void {
            if (src0.isSameShape(src1) and dst.isSameShape(src0)) {
                if (dst.isContiguous() and src0.isContiguous() and src1.isContiguous()) {
                    simdMapBinary(T, .vec_vec, src0.data, src1.data, dst.data, addVec, addScalar);
                } else {
                    if (dst.n_dims > 4) {
                        computeBinaryGeneric(Self, T, dst, src0, src1, addScalar);
                        return;
                    }
                    for (0..dst.ne[3]) |d3| {
                        for (0..dst.ne[2]) |d2| {
                            for (0..dst.ne[1]) |d1| {
                                for (0..dst.ne[0]) |d0| {
                                    const dst_idx = elemOffset(dst.strides, d0, d1, d2, d3);
                                    const src0_idx = elemOffset(src0.strides, d0, d1, d2, d3);
                                    const src1_idx = elemOffset(src1.strides, d0, d1, d2, d3);
                                    dst.data[dst_idx] = src0.data[src0_idx] + src1.data[src1_idx];
                                }
                            }
                        }
                    }
                }
            } else if (src1.isScalar() and dst.isSameShape(src0)) {
                assert(dst.isSameShape(src0));
                simdMapBinary(T, .scalar_rhs, src0.data, src1.data, dst.data, addVec, addScalar);
            } else if (src0.isScalar() and dst.isSameShape(src1)) {
                assert(dst.isSameShape(src1));
                simdMapBinary(T, .scalar_lhs, src0.data, src1.data, dst.data, addVec, addScalar);
            } else {
                computeBinaryGeneric(Self, T, dst, src0, src1, addScalar);
            }
        }

        pub fn computeSub(dst: *Self, src0: *const Self, src1: *const Self) void {
            if (src0.isSameShape(src1) and dst.isSameShape(src0)) {
                if (dst.isContiguous() and src0.isContiguous() and src1.isContiguous()) {
                    simdMapBinary(T, .vec_vec, src0.data, src1.data, dst.data, subVec, subScalar);
                } else {
                    if (dst.n_dims > 4) {
                        computeBinaryGeneric(Self, T, dst, src0, src1, subScalar);
                        return;
                    }
                    for (0..dst.ne[3]) |d3| {
                        for (0..dst.ne[2]) |d2| {
                            for (0..dst.ne[1]) |d1| {
                                for (0..dst.ne[0]) |d0| {
                                    const dst_idx = elemOffset(dst.strides, d0, d1, d2, d3);
                                    const src0_idx = elemOffset(src0.strides, d0, d1, d2, d3);
                                    const src1_idx = elemOffset(src1.strides, d0, d1, d2, d3);
                                    dst.data[dst_idx] = src0.data[src0_idx] - src1.data[src1_idx];
                                }
                            }
                        }
                    }
                }
            } else if (src1.isScalar() and dst.isSameShape(src0)) {
                assert(dst.isSameShape(src0));
                simdMapBinary(T, .scalar_rhs, src0.data, src1.data, dst.data, subVec, subScalar);
            } else if (src0.isScalar() and dst.isSameShape(src1)) {
                assert(dst.isSameShape(src1));
                simdMapBinary(T, .scalar_lhs, src0.data, src1.data, dst.data, subVec, subScalar);
            } else {
                computeBinaryGeneric(Self, T, dst, src0, src1, subScalar);
            }
        }

        pub fn computeMul(dst: *Self, src0: *const Self, src1: *const Self) void {
            if (src0.isScalar()) {
                assert(dst.isSameShape(src1));
                simdMapBinary(T, .scalar_lhs, src0.data, src1.data, dst.data, mulVec, mulScalar);
            } else if (src1.isScalar()) {
                assert(dst.isSameShape(src0));
                simdMapBinary(T, .scalar_rhs, src0.data, src1.data, dst.data, mulVec, mulScalar);
            } else {
                assert(dst.isBroadcastable(src0));
                assert(src0.isBroadcastable(src1));
                if (dst.isContiguous() and src0.isContiguous() and src1.isContiguous()) {
                    simdMapBinary(T, .vec_vec, src0.data, src1.data, dst.data, mulVec, mulScalar);
                } else {
                    if (dst.n_dims > 4 or !dst.isSameShape(src0) or !src0.isSameShape(src1)) {
                        computeBinaryGeneric(Self, T, dst, src0, src1, mulScalar);
                        return;
                    }
                    for (0..dst.ne[3]) |d3| {
                        for (0..dst.ne[2]) |d2| {
                            for (0..dst.ne[1]) |d1| {
                                for (0..dst.ne[0]) |d0| {
                                    const dst_idx = elemOffset(dst.strides, d0, d1, d2, d3);
                                    const src0_idx = elemOffset(src0.strides, d0, d1, d2, d3);
                                    const src1_idx = elemOffset(src1.strides, d0, d1, d2, d3);
                                    dst.data[dst_idx] = src0.data[src0_idx] * src1.data[src1_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        pub fn computeDiv(dst: *Self, src0: *const Self, src1: *const Self) void {
            if (src0.isScalar()) {
                assert(dst.isSameShape(src1));
                simdMapBinary(T, .scalar_lhs, src0.data, src1.data, dst.data, divVec, divScalar);
            } else if (src1.isScalar()) {
                assert(dst.isSameShape(src0));
                simdMapBinary(T, .scalar_rhs, src0.data, src1.data, dst.data, divVec, divScalar);
            } else {
                assert(dst.isBroadcastable(src0));
                assert(src0.isBroadcastable(src1));
                if (dst.isContiguous() and src0.isContiguous() and src1.isContiguous()) {
                    simdMapBinary(T, .vec_vec, src0.data, src1.data, dst.data, divVec, divScalar);
                } else {
                    if (dst.n_dims > 4 or !dst.isSameShape(src0) or !src0.isSameShape(src1)) {
                        computeBinaryGeneric(Self, T, dst, src0, src1, divScalar);
                        return;
                    }
                    for (0..dst.ne[3]) |d3| {
                        for (0..dst.ne[2]) |d2| {
                            for (0..dst.ne[1]) |d1| {
                                for (0..dst.ne[0]) |d0| {
                                    const dst_idx = elemOffset(dst.strides, d0, d1, d2, d3);
                                    const src0_idx = elemOffset(src0.strides, d0, d1, d2, d3);
                                    const src1_idx = elemOffset(src1.strides, d0, d1, d2, d3);
                                    dst.data[dst_idx] = src0.data[src0_idx] / src1.data[src1_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // Reduction / broadcast ops (stride-indexed, not SIMD'd)
        // ---------------------------------------------------------------

        pub fn computeSum(dst: *Self, src0: *const Self) void {
            assert(src0.canSumTo(dst));
            if (src0.n_dims > 4 or dst.n_dims > 4) {
                computeReduceGeneric(Self, T, dst, src0, .sum, null);
                return;
            }
            // Fast path: contiguous source and destination.
            if (src0.isContiguous() and dst.isContiguous()) {
                // Chunk = product of leading reduced dimensions.
                // For [W,H,C,N] → [1,1,C,1]: chunk = W*H (576 contiguous elements per channel).
                var chunk: usize = 1;
                for (0..src0.n_dims) |d| {
                    if (d < dst.n_dims and dst.ne[d] > 1) break;
                    chunk *= src0.ne[d];
                }
                simdReduceSum(T, src0.data, dst.data, dst.nElems(), chunk);
                return;
            }
            @memset(dst.data, 0);
            for (0..src0.ne[3]) |ne3| {
                for (0..src0.ne[2]) |ne2| {
                    for (0..src0.ne[1]) |ne1| {
                        for (0..src0.ne[0]) |ne0| {
                            const src0_nes = @Vector(4, usize){ ne0, ne1, ne2, ne3 };
                            const dst_ne_v = first4(dst.ne);
                            const dst_nes = src0_nes % dst_ne_v;
                            const src0_stride_v = first4(src0.strides);
                            const dst_stride_v = first4(dst.strides);
                            const src0_idx = @reduce(.Add, src0_nes * src0_stride_v);
                            const dst_idx = @reduce(.Add, dst_nes * dst_stride_v);
                            dst.data[dst_idx] += src0.data[src0_idx];
                        }
                    }
                }
            }
        }

        pub fn computeMax(dst: *Self, src0: *const Self) void {
            assert(src0.canSumTo(dst));
            if (src0.n_dims > 4 or dst.n_dims > 4) {
                computeReduceGeneric(Self, T, dst, src0, .max, null);
                return;
            }
            @memset(dst.data, -std.math.inf(T));
            for (0..src0.ne[3]) |ne3| {
                for (0..src0.ne[2]) |ne2| {
                    for (0..src0.ne[1]) |ne1| {
                        for (0..src0.ne[0]) |ne0| {
                            const src0_nes = @Vector(4, usize){ ne0, ne1, ne2, ne3 };
                            const dst_ne_v = first4(dst.ne);
                            const dst_nes = src0_nes % dst_ne_v;
                            const src0_stride_v = first4(src0.strides);
                            const dst_stride_v = first4(dst.strides);
                            const src0_idx = @reduce(.Add, src0_nes * src0_stride_v);
                            const dst_idx = @reduce(.Add, dst_nes * dst_stride_v);
                            dst.data[dst_idx] = @max(dst.data[dst_idx], src0.data[src0_idx]);
                        }
                    }
                }
            }
        }

        pub fn computeRepeat(dst: *Self, src0: *const Self) void {
            assert(src0.canRepeatTo(dst));
            if (dst.n_dims > 4 or src0.n_dims > 4) {
                computeRepeatGeneric(Self, dst, src0);
                return;
            }
            // Fast path: contiguous tensors.
            if (src0.isContiguous() and dst.isContiguous()) {
                var chunk: usize = 1;
                for (0..dst.n_dims) |d| {
                    if (d < src0.n_dims and src0.ne[d] > 1) break;
                    chunk *= dst.ne[d];
                }
                simdBroadcastRepeat(T, src0.data, dst.data, src0.nElems(), chunk);
                return;
            }
            for (0..dst.ne[3]) |ne3| {
                for (0..dst.ne[2]) |ne2| {
                    for (0..dst.ne[1]) |ne1| {
                        for (0..dst.ne[0]) |ne0| {
                            const nes = @Vector(4, usize){ ne0, ne1, ne2, ne3 };
                            const src0_ne_v = first4(src0.ne);
                            const src0_nes = nes % src0_ne_v;
                            const src0_stride_v = first4(src0.strides);
                            const dst_stride_v = first4(dst.strides);
                            const src0_idx = @reduce(.Add, src0_nes * src0_stride_v);
                            const dst_idx = @reduce(.Add, nes * dst_stride_v);
                            dst.data[dst_idx] = src0.data[src0_idx];
                        }
                    }
                }
            }
        }

        pub fn computeGatherRows(dst: *Self, src0: *const Self, indices: *const Self) void {
            assert(src0.isMatrix());
            assert(indices.isVector());
            assert(dst.ne[0] == src0.ne[0]);
            assert(dst.ne[1] == indices.ne[0]);

            const width = src0.ne[0];
            const rows = src0.ne[1];
            const count = indices.ne[0];
            for (0..count) |out_row| {
                const src_row = indexAt(indices, out_row);
                assert(src_row < rows);
                const src_off = src_row * src0.strides[1];
                const dst_off = out_row * dst.strides[1];
                @memcpy(dst.data[dst_off .. dst_off + width], src0.data[src_off .. src_off + width]);
            }
        }

        pub fn computeScatterAddRows(dst: *Self, updates: *const Self, indices: *const Self) void {
            assert(dst.isMatrix());
            assert(updates.isMatrix());
            assert(indices.isVector());
            assert(updates.ne[0] == dst.ne[0]);
            assert(updates.ne[1] == indices.ne[0]);

            const width = dst.ne[0];
            const rows = dst.ne[1];
            const count = indices.ne[0];
            @memset(dst.data, 0);
            for (0..count) |update_row| {
                const dst_row = indexAt(indices, update_row);
                assert(dst_row < rows);
                const upd_off = update_row * updates.strides[1];
                const dst_off = dst_row * dst.strides[1];
                for (0..width) |col| {
                    dst.data[dst_off + col] += updates.data[upd_off + col];
                }
            }
        }

        pub fn computePickRows(dst: *Self, src0: *const Self, indices: *const Self) void {
            assert(src0.isMatrix());
            assert(indices.isVector());
            assert(dst.isVector());
            assert(indices.ne[0] == src0.ne[1]);
            assert(dst.ne[0] == src0.ne[1]);

            const width = src0.ne[0];
            const rows = src0.ne[1];
            for (0..rows) |row| {
                const col = indexAt(indices, row);
                assert(col < width);
                dst.data[row] = src0.data[row * src0.strides[1] + col];
            }
        }

        pub fn computeScatterAddPicks(dst: *Self, updates: *const Self, indices: *const Self) void {
            assert(dst.isMatrix());
            assert(updates.isVector());
            assert(indices.isVector());
            assert(indices.ne[0] == dst.ne[1]);
            assert(updates.ne[0] == dst.ne[1]);

            const width = dst.ne[0];
            const rows = dst.ne[1];
            @memset(dst.data, 0);
            for (0..rows) |row| {
                const col = indexAt(indices, row);
                assert(col < width);
                dst.data[row * dst.strides[1] + col] += updates.data[row];
            }
        }

        /// Transpose the first two dimensions, copying data into contiguous layout.
        /// dst shape is [src.ne[1], src.ne[0], ...] with standard strides.
        pub fn computeTranspose(dst: *Self, src0: *const Self) void {
            assert(dst.ne[0] == src0.ne[1]);
            assert(dst.ne[1] == src0.ne[0]);
            const cols = src0.ne[0];
            const rows = src0.ne[1];
            for (0..src0.ne[3]) |d3| {
                for (0..src0.ne[2]) |d2| {
                    const src_batch = d3 * src0.strides[3] + d2 * src0.strides[2];
                    const dst_batch = d3 * dst.strides[3] + d2 * dst.strides[2];
                    for (0..rows) |row| {
                        for (0..cols) |col| {
                            // src[col, row] → dst[row, col]
                            dst.data[dst_batch + row * dst.strides[0] + col * dst.strides[1]] =
                                src0.data[src_batch + col * src0.strides[0] + row * src0.strides[1]];
                        }
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // Matrix multiplication  (runtime-dispatched SIMD width)
        // ---------------------------------------------------------------

        pub fn assertValidMatMulDims(dst: *const Self, src0: *const Self, trans0: bool, src1: *const Self, trans1: bool) void {
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

        pub fn computeMatMul(dst: *Self, src0: *const Self, comptime trans0: bool, src1: *const Self, comptime trans1: bool) void {
            assert(max_dims >= 4);
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

            // Resolve transpose-dependent dimensions once
            const M = if (trans0) src0_ne0 else src0_ne1;
            const N = if (trans1) src1_ne1 else src1_ne0;
            const K = if (trans0) src0_ne1 else src0_ne0;
            const a_m_stride = if (trans0) src0.strides[0] else src0.strides[1];
            const a_k_stride = if (trans0) src0.strides[1] else src0.strides[0];
            const b_n_stride = if (trans1) src1.strides[1] else src1.strides[0];
            const b_k_stride = if (trans1) src1.strides[0] else src1.strides[1];

            // Get the best matmul kernel for this CPU (cached after first call)
            const kernel = selectMatMulKernel(T);

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
                        const a_base = src0_i3 * src0.strides[3] + src0_i2 * src0.strides[2];
                        const b_base = src0_i3 * src1.strides[3] + src0_i2 * src1.strides[2];
                        const d_base = src0_i3 * dst.strides[3] + src0_i2 * dst.strides[2];
                        kernel(dst.data, src0.data, src1.data, M, N, K, a_m_stride, a_k_stride, b_k_stride, b_n_stride, a_base, b_base, d_base, dst.strides[1]);
                    }
                }
            }
        }

        /// Parallel matmul: splits M-rows across threads for each batch.
        pub fn computeMatMulParallel(dst: *Self, src0: *const Self, comptime trans0: bool, src1: *const Self, comptime trans1: bool, pool: *std.Thread.Pool) void {
            dst.assertValidMatMulDims(src0, trans0, src1, trans1);
            assert(dst.strides[0] == 1);

            const M = if (trans0) src0.ne[0] else src0.ne[1];
            const N = if (trans1) src1.ne[1] else src1.ne[0];
            const K = if (trans0) src0.ne[1] else src0.ne[0];
            const a_m_stride = if (trans0) src0.strides[0] else src0.strides[1];
            const a_k_stride = if (trans0) src0.strides[1] else src0.strides[0];
            const b_n_stride = if (trans1) src1.strides[1] else src1.strides[0];
            const b_k_stride = if (trans1) src1.strides[0] else src1.strides[1];
            const kernel = selectMatMulRangeKernel(T);

            const n_workers = pool.threads.len + 1;
            const min_rows_per_thread = 4; // don't split below this

            for (0..src0.ne[3]) |b3| {
                for (0..src0.ne[2]) |b2| {
                    const a_base = b3 * src0.strides[3] + b2 * src0.strides[2];
                    const b_base = b3 * src1.strides[3] + b2 * src1.strides[2];
                    const d_base = b3 * dst.strides[3] + b2 * dst.strides[2];

                    if (M < min_rows_per_thread * 2 or n_workers <= 1) {
                        kernel(dst.data, src0.data, src1.data, 0, M, N, K, a_m_stride, a_k_stride, b_k_stride, b_n_stride, a_base, b_base, d_base, dst.strides[1]);
                    } else {
                        const chunk = @max(min_rows_per_thread, (M + n_workers - 1) / n_workers);
                        var wg = std.Thread.WaitGroup{};
                        var m_start: usize = chunk;
                        while (m_start < M) {
                            const m_end = @min(m_start + chunk, M);
                            pool.spawnWg(&wg, struct {
                                fn work(d: []T, s0: []const T, s1: []const T, ms: usize, me: usize, n: usize, k: usize, am: usize, ak: usize, bk: usize, bn: usize, ab: usize, bb: usize, db: usize, dr: usize, kfn: MatMulRangeFnType(T)) void {
                                    kfn(d, s0, s1, ms, me, n, k, am, ak, bk, bn, ab, bb, db, dr);
                                }
                            }.work, .{ dst.data, src0.data, src1.data, m_start, m_end, N, K, a_m_stride, a_k_stride, b_k_stride, b_n_stride, a_base, b_base, d_base, dst.strides[1], kernel });
                            m_start += chunk;
                        }
                        // Caller does first chunk
                        kernel(dst.data, src0.data, src1.data, 0, @min(chunk, M), N, K, a_m_stride, a_k_stride, b_k_stride, b_n_stride, a_base, b_base, d_base, dst.strides[1]);
                        wg.wait();
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // Dispatch
        // ---------------------------------------------------------------

        pub fn compute(tensor: *Self) void {
            const src0 = tensor.src0;
            const src1 = tensor.src1;
            switch (tensor.op) {
                .none, .view, .reshape, .transpose, .permute, .as_strided, .broadcast_to => {},
                .scatter_add_view => computeScatterAddView(Self, tensor, src0.?, src1.?),
                .add => tensor.computeAdd(src0.?, src1.?),
                .mul => tensor.computeMul(src0.?, src1.?),
                .neg => tensor.computeNeg(src0.?),
                .abs => tensor.computeAbs(src0.?),
                .sgn => tensor.computeSgn(src0.?),
                .step => tensor.computeStep(src0.?),
                .sqrt => tensor.computeSqrt(src0.?),
                .recip => tensor.computeRecip(src0.?),
                .exp => tensor.computeExp(src0.?),
                .log => tensor.computeLog(src0.?),
                .gelu => tensor.computeGelu(src0.?),
                .sum => tensor.computeSum(src0.?),
                .max => tensor.computeMax(src0.?),
                .repeat => tensor.computeRepeat(src0.?),
                .gather_rows => tensor.computeGatherRows(src0.?, src1.?),
                .scatter_add_rows => tensor.computeScatterAddRows(src0.?, src1.?),
                .pick_rows => tensor.computePickRows(src0.?, src1.?),
                .scatter_add_picks => tensor.computeScatterAddPicks(src0.?, src1.?),
                .matmul => {
                    const flags = tensor.matmul_flags;
                    if (flags.trans0) {
                        if (flags.trans1) tensor.computeMatMul(src0.?, true, src1.?, true) else tensor.computeMatMul(src0.?, true, src1.?, false);
                    } else {
                        if (flags.trans1) tensor.computeMatMul(src0.?, false, src1.?, true) else tensor.computeMatMul(src0.?, false, src1.?, false);
                    }
                },
            }
        }
    };
}
