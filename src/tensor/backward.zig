//! Backward (reverse-mode autodiff) implementations for primitive tensor operations.
//!
//! Only primitive ops need backward rules here. Decomposed ops (sub, sqr, div,
//! relu, scale, mean) get their gradients automatically through the chain rule
//! on the primitives they decompose into.

const std = @import("std");
const assert = std.debug.assert;
const Alloc = std.mem.Allocator;

/// Coefficient for the GeLU tanh approximation: `gelu(x) ≈ 0.5x(1 + tanh(√(2/π) · x · (1 + Ax²)))`.
const GELU_COEF_A: comptime_float = 0.044715;
/// √(2/π), used in the GeLU approximation.
const SQRT_2_OVER_PI: comptime_float = @sqrt(2.0 / std.math.pi);

/// Backward operations parameterized on the tensor type.
pub fn Ops(comptime Self: type) type {
    return struct {
        /// Accumulate `contribution` into `grad`.
        fn accumGrad(grad: *Self, contribution: *Self, inplace: bool) *Self {
            return if (inplace) grad.addInplace(contribution) else grad.add(contribution);
        }

        /// Remove the gradient from an intermediate tensor created during backward.
        fn stripGrad(tensor: *Self) void {
            tensor.setGrad(null);
        }

        fn addToScratchUniq(alloc: Alloc, scratch: *std.ArrayList(*Self), tensor: *Self) Alloc.Error!void {
            for (scratch.items) |item| {
                if (item == tensor) return;
            }
            try scratch.append(alloc, tensor);
        }

        /// Propagate gradients backward through the primitive op that created this tensor.
        /// `alloc` is used only for scratch list management, not for tensor ops.
        pub fn backward(tensor: *Self, alloc: Alloc, scratch: *std.ArrayList(*Self), inplace: bool) Alloc.Error!void {
            _ = scratch;
            const src0_o = tensor.source0();
            const src1_o = tensor.source1();
            const out_grad = tensor.gradOrNull().?;

            switch (tensor.opTag()) {
                .none => {},

                .reshape, .broadcast_to => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const contribution = if (tensor.opTag() == .broadcast_to)
                            out_grad.sumInto(grad)
                        else
                            out_grad.reshapeLike(grad);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                .transpose => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const contribution = out_grad.transpose();
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                .permute => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const axes = tensor.source1().?.index_data.?;
                        var inv: [8]usize = [_]usize{0} ** 8;
                        var i: usize = 0;
                        while (i < tensor.n_dims) : (i += 1) inv[axes[i]] = i;
                        const contribution = out_grad.permute(inv[0..out_grad.n_dims]);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                .view => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const contribution = out_grad.reshapeLike(grad);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                .as_strided => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const contribution = out_grad.scatterAddView(tensor);

                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                .scatter_add_view => {},

                // d/d(src0) [src0 + src1] = out_grad
                // d/d(src1) [src0 + src1] = out_grad
                .add => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const contribution = if (out_grad.isSameShape(grad)) out_grad else out_grad.sumInto(grad);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                    if (src1.gradOrNull()) |grad| {
                        const contribution = if (out_grad.isSameShape(grad)) out_grad else out_grad.sumInto(grad);
                        stripGrad(contribution);
                        src1.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                // d/d(src0) [src0 * src1] = src1 * out_grad
                // d/d(src1) [src0 * src1] = src0 * out_grad
                .mul => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const raw = src1.mul(out_grad);
                        stripGrad(raw);
                        const contribution = if (raw.isSameShape(grad)) raw else raw.sumInto(grad);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                    if (src1.gradOrNull()) |grad| {
                        const raw = src0.mul(out_grad);
                        stripGrad(raw);
                        const contribution = if (raw.isSameShape(grad)) raw else raw.sumInto(grad);
                        stripGrad(contribution);
                        src1.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                // d/d(src0) [-src0] = -out_grad
                .neg => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const neg_grad = out_grad.neg();
                        stripGrad(neg_grad);
                        src0.setGrad(accumGrad(grad, neg_grad, inplace));
                    }
                },

                // d/d(src0) [abs(src0)] = sgn(src0) * out_grad
                .abs => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const sign = src0.sgn();
                        stripGrad(sign);
                        const contribution = sign.mul(out_grad);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                // d/d(src0) [sqrt(src0)] = 0.5 / sqrt(src0) * out_grad
                // tensor already holds sqrt(src0) from the forward pass
                .sqrt => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const half = try Self.initScalar(alloc, 0.5);
                        const half_rep = half.repeatLike(src0);
                        const inv_2sqrt = half_rep.div(tensor);
                        stripGrad(inv_2sqrt);
                        const contribution = inv_2sqrt.mul(out_grad);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                // d/d(src0) [1/src0] = -1/src0^2 * out_grad
                // tensor already holds 1/src0 from the forward pass
                .recip => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const neg_recip_sq = tensor.mul(tensor).neg();
                        stripGrad(neg_recip_sq);
                        const contribution = neg_recip_sq.mul(out_grad);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                // d/d(src0) [exp(src0)] = exp(src0) * out_grad
                .exp => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const contribution = tensor.mul(out_grad);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                // d/d(src0) [log(src0)] = out_grad / src0
                .log => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const contribution = out_grad.div(src0);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                // d/d(src0) [repeat(src0)] = sum(out_grad) back to src0's shape
                .repeat => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        src0.setGrad(out_grad.sumInto(grad));
                    }
                },

                // Gradient of sum broadcasts back to source shape
                .sum => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        src0.setGrad(accumGrad(grad, out_grad.repeatLike(grad), inplace));
                    }
                },

                .max => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const repeated_max = tensor.repeatLike(src0);
                        stripGrad(repeated_max);
                        const diff = src0.sub(repeated_max);
                        stripGrad(diff);
                        const one = try Self.initScalar(alloc, 1);
                        const zero_mask = diff.abs().step();
                        stripGrad(zero_mask);
                        const mask = one.repeatLike(src0).sub(zero_mask);
                        stripGrad(mask);
                        const expanded_out_grad = out_grad.repeatLike(src0);
                        stripGrad(expanded_out_grad);
                        const contribution = mask.mul(expanded_out_grad);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                .gather_rows => {
                    const src0 = src0_o.?;
                    const indices = src1_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const contribution = grad.scatterAddRows(indices, out_grad);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                .scatter_add_rows => {},

                .pick_rows => {
                    const src0 = src0_o.?;
                    const indices = src1_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const contribution = grad.scatterAddPicks(indices, out_grad);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                .scatter_add_picks => {},

                // d/d(A) [A @ B] = out_grad @ B^T
                // d/d(B) [A @ B] = A^T @ out_grad
                .matmul => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const z = out_grad.matMul(false, src1, true);
                        stripGrad(z);
                        src0.setGrad(accumGrad(grad, z, inplace));
                    }
                    if (src1.gradOrNull()) |grad| {
                        const z = src0.matMul(true, out_grad, false);
                        stripGrad(z);
                        src1.setGrad(accumGrad(grad, z, inplace));
                    }
                },
                .matmul_t0 => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const z = src1.matMul(false, out_grad, true);
                        stripGrad(z);
                        src0.setGrad(accumGrad(grad, z, inplace));
                    }
                    if (src1.gradOrNull()) |grad| {
                        const z = src0.matMul(false, out_grad, false);
                        stripGrad(z);
                        src1.setGrad(accumGrad(grad, z, inplace));
                    }
                },
                .matmul_t1 => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const z = out_grad.matMul(false, src1, false);
                        stripGrad(z);
                        src0.setGrad(accumGrad(grad, z, inplace));
                    }
                    if (src1.gradOrNull()) |grad| {
                        const z = out_grad.matMul(true, src0, false);
                        stripGrad(z);
                        src1.setGrad(accumGrad(grad, z, inplace));
                    }
                },
                .matmul_t0t1 => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const z = src1.matMul(true, out_grad, true);
                        stripGrad(z);
                        src0.setGrad(accumGrad(grad, z, inplace));
                    }
                    if (src1.gradOrNull()) |grad| {
                        const z = out_grad.matMul(true, src0, true);
                        stripGrad(z);
                        src1.setGrad(accumGrad(grad, z, inplace));
                    }
                },

                // max_pool2d backward: route gradient to the max element in each 2x2 window
                .max_pool2d => {
                    const input = src0_o.?;
                    if (input.gradOrNull()) |grad| {
                        const in_w = input.ne[0];
                        const out_w = tensor.ne[0];
                        const out_h = tensor.ne[1];
                        const channels = tensor.ne[2];
                        const batch = tensor.ne[3];
                        const in_s_h = in_w;
                        const in_s_c = in_w * input.ne[1];
                        const in_s_n = in_s_c * channels;
                        const o_s_h = out_w;
                        const o_s_c = out_w * out_h;
                        const o_s_n = o_s_c * channels;

                        const gi = try Self.init(alloc, grad.ne[0..grad.n_dims]);
                        @memset(gi.data, 0);
                        for (0..batch) |n| {
                            for (0..channels) |ch| {
                                for (0..out_h) |oy| {
                                    for (0..out_w) |ox| {
                                        const ix = ox * 2;
                                        const iy = oy * 2;
                                        const base = ix + iy * in_s_h + ch * in_s_c + n * in_s_n;
                                        // Find which of the 4 elements was max
                                        var max_val = input.data[base];
                                        var max_idx = base;
                                        const idx1 = base + 1;
                                        if (input.data[idx1] > max_val) {
                                            max_val = input.data[idx1];
                                            max_idx = idx1;
                                        }
                                        const idx2 = base + in_s_h;
                                        if (input.data[idx2] > max_val) {
                                            max_val = input.data[idx2];
                                            max_idx = idx2;
                                        }
                                        const idx3 = base + in_s_h + 1;
                                        if (input.data[idx3] > max_val) {
                                            max_idx = idx3;
                                        }
                                        gi.data[max_idx] += out_grad.data[ox + oy * o_s_h + ch * o_s_c + n * o_s_n];
                                    }
                                }
                            }
                        }
                        stripGrad(gi);
                        input.setGrad(accumGrad(grad, gi, inplace));
                    }
                },

                .sgn, .step => {}, // non-differentiable

                // d/dx[gelu(x)] = 0.5*(1+tanh(a)) + 0.5*x*sech²(a)*a'
                // where a = sqrt(2/π)*x*(1 + 0.044715*x²), a' = sqrt(2/π)*(1 + 3*0.044715*x²)
                .gelu => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const ElemType = @TypeOf(src0.data[0]);
                        const gelu_grad = try src0.map(struct {
                            fn f(x: ElemType) ElemType {
                                const coef: ElemType = GELU_COEF_A;
                                const s2pi: ElemType = SQRT_2_OVER_PI;
                                const x2 = x * x;
                                const a = s2pi * x * (1.0 + coef * x2);
                                const tanh_a = std.math.tanh(a);
                                const sech2_a = 1.0 - tanh_a * tanh_a;
                                const da = s2pi * (1.0 + 3.0 * coef * x2);
                                return 0.5 * (1.0 + tanh_a) + 0.5 * x * sech2_a * da;
                            }
                        }.f);
                        const contribution = gelu_grad.mul(out_grad);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },
            }
        }
    };
}
