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
                        const axes = tensor.source1().?.indexData().?;
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

                // d/d(src0) [relu(src0)] = 1{tensor > 0} * out_grad
                // Uses the forward output so backward does not depend on any
                // decomposed step-mask subgraph.
                .relu => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const mask = tensor.step();
                        stripGrad(mask);
                        const contribution = mask.mul(out_grad);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
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
                        const half_rep = half.broadcastTo(src0.ne[0..src0.n_dims]);
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
                        const expanded = out_grad.broadcastTo(grad.ne[0..grad.n_dims]);
                        stripGrad(expanded);
                        src0.setGrad(accumGrad(grad, expanded, inplace));
                    }
                },

                .max => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const repeated_max = tensor.broadcastTo(src0.ne[0..src0.n_dims]);
                        stripGrad(repeated_max);
                        const diff = src0.sub(repeated_max);
                        stripGrad(diff);
                        const one = try Self.initScalar(alloc, 1);
                        const zero_mask = diff.abs().step();
                        stripGrad(zero_mask);
                        const mask = one.broadcastTo(src0.ne[0..src0.n_dims]).sub(zero_mask);
                        stripGrad(mask);
                        const tie_count = mask.sum(tensor.ne[0..tensor.n_dims]);
                        stripGrad(tie_count);
                        const expanded_tie_count = tie_count.broadcastTo(src0.ne[0..src0.n_dims]);
                        stripGrad(expanded_tie_count);
                        const expanded_out_grad = out_grad.broadcastTo(src0.ne[0..src0.n_dims]);
                        stripGrad(expanded_out_grad);
                        const shared_out_grad = expanded_out_grad.div(expanded_tie_count);
                        stripGrad(shared_out_grad);
                        const contribution = mask.mul(shared_out_grad);
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

                // Slice assign: inference-only, no gradient needed
                .slice_assign, .slice_assign_rows => {},

                .rope => {
                    // Backward of rope: inverse rotation (negate sin).
                    // dx[i] = dy[i]*cos[i] + dy[i+half]*sin[i]
                    // dx[i+half] = dy[i+half]*cos[i] - dy[i]*sin[i]
                    // This is rope with negated sin applied to out_grad.
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        // Build cos_sin_neg: cos unchanged, sin negated.
                        const cos_sin = src1_o.?;
                        const d = src0.ne[0];
                        const seq = src0.ne[1];
                        const neg_cs = Self.init(alloc, &.{ 2 * d, seq }) catch unreachable;
                        // Copy cos rows (0..d)
                        for (0..seq) |col| {
                            const cs_off = col * 2 * d;
                            @memcpy(neg_cs.data[cs_off..][0..d], cos_sin.data[cs_off..][0..d]);
                            // Negate sin rows (d..2d)
                            for (0..d) |i| {
                                neg_cs.data[cs_off + d + i] = -cos_sin.data[cs_off + d + i];
                            }
                        }
                        const contribution = out_grad.ropeRotate(neg_cs);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                // d/dx[rmsnorm(x, axis, eps)] = s*dy - y*(sum(y*dy, axis) / N)
                // where s = 1/sqrt(mean(x², axis) + eps), N = elems per group.
                .rmsnorm => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const y = tensor;
                        const reduce_shape = y.reduce_ne[0..y.n_dims];
                        const eps = y.op_eps;

                        var total: usize = 1;
                        for (src0.ne[0..src0.n_dims]) |e| total *= e;
                        var red: usize = 1;
                        for (reduce_shape) |e| red *= e;
                        const count_per_group: @TypeOf(eps) = @floatFromInt(total / red);

                        // Recompute per-group scale s = 1/sqrt(mean(x²) + eps).
                        const sq = src0.sqr();
                        stripGrad(sq);
                        const mean_sq = sq.mean(reduce_shape);
                        stripGrad(mean_sq);
                        const eps_scalar = try Self.initScalar(alloc, eps);
                        stripGrad(eps_scalar);
                        const eps_bcast = eps_scalar.broadcastTo(mean_sq.ne[0..mean_sq.n_dims]);
                        stripGrad(eps_bcast);
                        const var_eps = mean_sq.add(eps_bcast);
                        stripGrad(var_eps);
                        const rsqrt = var_eps.sqrt().recip();
                        stripGrad(rsqrt);
                        const s_expanded = rsqrt.broadcastTo(src0.ne[0..src0.n_dims]);
                        stripGrad(s_expanded);

                        // inner = sum(y * dy, axis) / N, broadcast back.
                        const y_d = y.mul(out_grad);
                        stripGrad(y_d);
                        const inner = y_d.sum(reduce_shape);
                        stripGrad(inner);
                        const inner_scaled = inner.scaleByVal(1.0 / count_per_group);
                        stripGrad(inner_scaled);
                        const inner_bcast = inner_scaled.broadcastTo(src0.ne[0..src0.n_dims]);
                        stripGrad(inner_bcast);

                        const term1 = s_expanded.mul(out_grad);
                        stripGrad(term1);
                        const term2 = y.mul(inner_bcast);
                        stripGrad(term2);
                        const contribution = term1.sub(term2);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                // d/dx[softmax(x, axis)] = y * (dy - sum(dy * y, axis))
                .softmax => {
                    const src0 = src0_o.?;
                    if (src0.gradOrNull()) |grad| {
                        const y = tensor;
                        const inner = out_grad.mul(y);
                        stripGrad(inner);
                        const reduce_shape = y.reduce_ne[0..y.n_dims];
                        const s = inner.sum(reduce_shape);
                        stripGrad(s);
                        const s_expanded = s.broadcastTo(src0.ne[0..src0.n_dims]);
                        stripGrad(s_expanded);
                        const diff = out_grad.sub(s_expanded);
                        stripGrad(diff);
                        const contribution = y.mul(diff);
                        stripGrad(contribution);
                        src0.setGrad(accumGrad(grad, contribution, inplace));
                    }
                },

                // Matmul backward: dispatch based on transpose flags.
                // For fwd C = op(A) @ op(B) where op is identity or transpose:
                //   no trans:  d/dA = G @ B^T,      d/dB = A^T @ G
                //   trans0:    d/dA = B @ G^T,       d/dB = A @ G
                //   trans1:    d/dA = G @ B,         d/dB = G^T @ A
                //   trans0t1:  d/dA = B^T @ G^T,     d/dB = G^T @ A^T
                .matmul => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    const t0 = tensor.matmul_flags.trans0;
                    const t1 = tensor.matmul_flags.trans1;
                    if (src0.gradOrNull()) |grad| {
                        const z = if (!t0 and !t1)
                            out_grad.matMul(false, src1, true)
                        else if (t0 and !t1)
                            src1.matMul(false, out_grad, true)
                        else if (!t0 and t1)
                            out_grad.matMul(false, src1, false)
                        else
                            src1.matMul(true, out_grad, true);
                        stripGrad(z);
                        src0.setGrad(accumGrad(grad, z, inplace));
                    }
                    if (src1.gradOrNull()) |grad| {
                        const z = if (!t0 and !t1)
                            src0.matMul(true, out_grad, false)
                        else if (t0 and !t1)
                            src0.matMul(false, out_grad, false)
                        else if (!t0 and t1)
                            out_grad.matMul(true, src0, false)
                        else
                            out_grad.matMul(true, src0, true);
                        stripGrad(z);
                        src1.setGrad(accumGrad(grad, z, inplace));
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
                                const xf: f32 = @floatCast(x);
                                const coef: f32 = GELU_COEF_A;
                                const s2pi: f32 = SQRT_2_OVER_PI;
                                const x2 = xf * xf;
                                const a = s2pi * xf * (1.0 + coef * x2);
                                const tanh_a = std.math.tanh(a);
                                const sech2_a = 1.0 - tanh_a * tanh_a;
                                const da = s2pi * (1.0 + 3.0 * coef * x2);
                                return @floatCast(0.5 * (1.0 + tanh_a) + 0.5 * xf * sech2_a * da);
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
