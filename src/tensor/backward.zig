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
        /// Accumulate `contribution` into `grad`, either in-place or by creating a new node.
        fn accumGrad(grad: *Self, alloc: Alloc, contribution: *Self, inplace: bool) *Self {
            return if (inplace) grad.addInplace(alloc, contribution) else grad.add(alloc, contribution);
        }

        /// Remove the gradient from an intermediate tensor created during backward.
        fn stripGrad(tensor: *Self, alloc: Alloc) void {
            _ = alloc;
            tensor.grad = null;
        }

        fn addToScratchUniq(alloc: Alloc, scratch: *std.ArrayList(*Self), tensor: *Self) Alloc.Error!void {
            for (scratch.items) |item| {
                if (item == tensor) return;
            }
            try scratch.append(alloc, tensor);
        }

        /// Propagate gradients backward through the primitive op that created this tensor.
        pub fn backward(tensor: *Self, alloc: Alloc, scratch: *std.ArrayList(*Self), inplace: bool) Alloc.Error!void {
            _ = scratch;
            const src0_o = tensor.src0;
            const src1_o = tensor.src1;
            const out_grad = tensor.grad.?;

            switch (tensor.op) {
                .none => {},

                .view, .reshape => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const contribution = out_grad.reshapeLike(alloc, grad);
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                .transpose => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const contribution = out_grad.transpose(alloc);
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                // d/d(src0) [src0 + src1] = out_grad
                // d/d(src1) [src0 + src1] = out_grad
                .add => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| src0.grad = accumGrad(grad, alloc, out_grad, inplace);
                    if (src1.grad) |grad| src1.grad = accumGrad(grad, alloc, out_grad, inplace);
                },

                .sub => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| src0.grad = accumGrad(grad, alloc, out_grad, inplace);
                    if (src1.grad) |grad| {
                        const contribution = out_grad.neg(alloc);
                        stripGrad(contribution, alloc);
                        src1.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                // d/d(src0) [src0 * src1] = src1 * out_grad
                // d/d(src1) [src0 * src1] = src0 * out_grad
                .mul => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        const contribution = src1.mul(alloc, out_grad);
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                    if (src1.grad) |grad| {
                        const contribution = src0.mul(alloc, out_grad);
                        stripGrad(contribution, alloc);
                        src1.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                .div => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        const contribution = out_grad.div(alloc, src1);
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                    if (src1.grad) |grad| {
                        const numer = src0.mul(alloc, out_grad).neg(alloc);
                        stripGrad(numer, alloc);
                        const denom = src1.mul(alloc, src1);
                        stripGrad(denom, alloc);
                        const contribution = numer.div(alloc, denom);
                        stripGrad(contribution, alloc);
                        src1.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                // d/d(src0) [-src0] = -out_grad
                .neg => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const neg_grad = out_grad.neg(alloc);
                        stripGrad(neg_grad, alloc);
                        src0.grad = accumGrad(grad, alloc, neg_grad, inplace);
                    }
                },

                .relu => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const mask = src0.step(alloc);
                        stripGrad(mask, alloc);
                        const contribution = mask.mul(alloc, out_grad);
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                // d/d(src0) [abs(src0)] = sgn(src0) * out_grad
                .abs => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const sign = src0.sgn(alloc);
                        stripGrad(sign, alloc);
                        const contribution = sign.mul(alloc, out_grad);
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                // d/d(src0) [sqrt(src0)] = 0.5 / sqrt(src0) * out_grad
                // tensor already holds sqrt(src0) from the forward pass
                .sqrt => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const half = try Self.initScalar(alloc, 0.5);
                        const half_rep = half.repeatLike(alloc, src0);
                        const inv_2sqrt = half_rep.div(alloc, tensor);
                        stripGrad(inv_2sqrt, alloc);
                        const contribution = inv_2sqrt.mul(alloc, out_grad);
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                // d/d(src0) [1/src0] = -1/src0^2 * out_grad
                // tensor already holds 1/src0 from the forward pass
                .recip => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const neg_recip_sq = tensor.mul(alloc, tensor).neg(alloc);
                        stripGrad(neg_recip_sq, alloc);
                        const contribution = neg_recip_sq.mul(alloc, out_grad);
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                // d/d(src0) [exp(src0)] = exp(src0) * out_grad
                .exp => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const contribution = tensor.mul(alloc, out_grad);
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                // d/d(src0) [log(src0)] = out_grad / src0
                .log => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const contribution = out_grad.div(alloc, src0);
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                // d/d(src0) [repeat(src0)] = sum(out_grad) back to src0's shape
                .repeat => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        src0.grad = out_grad.sumInto(alloc, grad);
                    }
                },

                // Gradient of sum broadcasts back to source shape
                .sum => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        src0.grad = accumGrad(grad, alloc, out_grad.repeatLike(alloc, grad), inplace);
                    }
                },

                .mean => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const count: f64 = @floatFromInt(src0.nElems() / tensor.nElems());
                        const inv_count = try Self.initScalar(alloc, @as(@TypeOf(src0.data[0]), @floatCast(1.0 / count)));
                        const expanded = out_grad.repeatLike(alloc, grad);
                        stripGrad(expanded, alloc);
                        const contribution = expanded.mul(alloc, inv_count.repeatLike(alloc, expanded));
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                .max => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const repeated_max = tensor.repeatLike(alloc, src0);
                        stripGrad(repeated_max, alloc);
                        const diff = src0.sub(alloc, repeated_max);
                        stripGrad(diff, alloc);
                        const one = try Self.initScalar(alloc, 1);
                        const zero_mask = diff.abs(alloc).step(alloc);
                        stripGrad(zero_mask, alloc);
                        const mask = one.repeatLike(alloc, src0).sub(alloc, zero_mask);
                        stripGrad(mask, alloc);
                        const expanded_out_grad = out_grad.repeatLike(alloc, src0);
                        stripGrad(expanded_out_grad, alloc);
                        const contribution = mask.mul(alloc, expanded_out_grad);
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                .gather_rows => {
                    const src0 = src0_o.?;
                    const indices = src1_o.?;
                    if (src0.grad) |grad| {
                        const contribution = grad.scatterAddRows(alloc, indices, out_grad);
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                .scatter_add_rows => {},

                .pick_rows => {
                    const src0 = src0_o.?;
                    const indices = src1_o.?;
                    if (src0.grad) |grad| {
                        const contribution = grad.scatterAddPicks(alloc, indices, out_grad);
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                .scatter_add_picks => {},

                // d/d(A) [A @ B] = out_grad @ B^T
                // d/d(B) [A @ B] = A^T @ out_grad
                .matmul => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        const z = out_grad.matMul(alloc, false, src1, true);
                        stripGrad(z, alloc);
                        src0.grad = accumGrad(grad, alloc, z, inplace);
                    }
                    if (src1.grad) |grad| {
                        const z = src0.matMul(alloc, true, out_grad, false);
                        stripGrad(z, alloc);
                        src1.grad = accumGrad(grad, alloc, z, inplace);
                    }
                },
                .matmul_t0 => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        const z = src1.matMul(alloc, false, out_grad, true);
                        stripGrad(z, alloc);
                        src0.grad = accumGrad(grad, alloc, z, inplace);
                    }
                    if (src1.grad) |grad| {
                        const z = src0.matMul(alloc, false, out_grad, false);
                        stripGrad(z, alloc);
                        src1.grad = accumGrad(grad, alloc, z, inplace);
                    }
                },
                .matmul_t1 => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        const z = out_grad.matMul(alloc, false, src1, false);
                        stripGrad(z, alloc);
                        src0.grad = accumGrad(grad, alloc, z, inplace);
                    }
                    if (src1.grad) |grad| {
                        const z = out_grad.matMul(alloc, true, src0, false);
                        stripGrad(z, alloc);
                        src1.grad = accumGrad(grad, alloc, z, inplace);
                    }
                },
                .matmul_t0t1 => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        const z = src1.matMul(alloc, true, out_grad, true);
                        stripGrad(z, alloc);
                        src0.grad = accumGrad(grad, alloc, z, inplace);
                    }
                    if (src1.grad) |grad| {
                        const z = out_grad.matMul(alloc, true, src0, true);
                        stripGrad(z, alloc);
                        src1.grad = accumGrad(grad, alloc, z, inplace);
                    }
                },

                .sgn, .step => {}, // non-differentiable

                // d/dx[gelu(x)] = 0.5*(1+tanh(a)) + 0.5*x*sech²(a)*a'
                // where a = sqrt(2/π)*x*(1 + 0.044715*x²), a' = sqrt(2/π)*(1 + 3*0.044715*x²)
                .gelu => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const ElemType = @TypeOf(src0.data[0]);
                        const gelu_grad = try src0.map(alloc, struct {
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
                        const contribution = gelu_grad.mul(alloc, out_grad);
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },
            }
        }
    };
}
