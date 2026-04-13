//! Backward (reverse-mode autodiff) implementations for tensor operations.
//!
//! Given a tensor produced by an operation, `backward` propagates gradients
//! from the output back through to the source tensors. Each operation has a
//! corresponding gradient rule.

const std = @import("std");
const assert = std.debug.assert;
const Alloc = std.mem.Allocator;

/// Backward operations parameterized on the tensor type.
pub fn Ops(comptime Self: type) type {
    return struct {
        /// Accumulate `contribution` into `grad`, either in-place or by creating a new node.
        fn accumGrad(grad: *Self, alloc: Alloc, contribution: *Self, inplace: bool) *Self {
            return if (inplace) grad.addInplace(alloc, contribution) else grad.add(alloc, contribution);
        }

        /// Same as accumGrad but for subtraction.
        fn accumGradSub(grad: *Self, alloc: Alloc, contribution: *Self, inplace: bool) *Self {
            return if (inplace) grad.subInplace(alloc, contribution) else grad.sub(alloc, contribution);
        }

        /// Remove the gradient from an intermediate tensor created during backward.
        /// These intermediates shouldn't track their own gradients.
        fn stripGrad(tensor: *Self, alloc: Alloc) void {
            if (tensor.grad) |gradp| {
                gradp.deinit(alloc);
                tensor.grad = null;
            }
        }

        fn addToScratchUniq(alloc: Alloc, scratch: *std.ArrayList(*Self), tensor: *Self) Alloc.Error!void {
            for (scratch.items) |item| {
                if (item == tensor) return;
            }
            try scratch.append(alloc, tensor);
        }

        /// Propagate gradients backward through the operation that created this tensor.
        ///
        /// For each source tensor with a gradient, accumulates the contribution from
        /// this tensor's gradient according to the chain rule. When `inplace` is true,
        /// gradient tensors may be modified in-place for efficiency.
        pub fn backward(tensor: *Self, alloc: Alloc, scratch: *std.ArrayList(*Self), inplace: bool) Alloc.Error!void {
            const src0_o = tensor.src0;
            const src1_o = tensor.src1;
            const out_grad = tensor.grad.?;

            switch (tensor.op) {
                .none, .view => {},

                .dup => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const new_grad = accumGrad(grad, alloc, out_grad, inplace);
                        assert(new_grad.isSameShape(grad));
                        src0.grad = new_grad;
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

                // d/d(src0) [src0 - src1] = out_grad
                // d/d(src1) [src0 - src1] = -out_grad
                .sub => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| src0.grad = accumGrad(grad, alloc, out_grad, inplace);
                    if (src1.grad) |grad| src1.grad = accumGradSub(grad, alloc, out_grad, inplace);
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

                // d/d(src0) [src0 / src1] = out_grad / src1
                // d/d(src1) [src0 / src1] = -src0 * out_grad / src1^2
                .div => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        const contribution = src1.div(alloc, out_grad);
                        stripGrad(contribution, alloc);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                    if (src1.grad) |grad| {
                        const contribution = src0.div(alloc, out_grad);
                        stripGrad(contribution, alloc);
                        src1.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                // d/d(src0) [repeat(src0)] = sum(out_grad) back to src0's shape
                .repeat => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        src0.grad = out_grad.sumInto(alloc, grad);
                    }
                },

                // d/d(src0) [src0^2] = 2 * src0 * out_grad
                .sqr => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const two = try Self.initScalar(alloc, 2);
                        const two_rep = two.repeatLike(alloc, src0);
                        const src0_x2 = src0.mul(alloc, two_rep);
                        stripGrad(src0_x2, alloc);
                        const contribution = src0_x2.mul(alloc, out_grad);
                        src0.grad = accumGrad(grad, alloc, contribution, inplace);
                    }
                },

                // d/d(src0) [sqrt(src0)] = out_grad / (2 * sqrt(src0))
                .sqrt => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const half = try Self.initScalar(alloc, 0.5);
                        const half_rep = half.repeatLike(alloc, src0);
                        // dst already holds sqrt(src0) from forward pass
                        const inv_2sqrt = half_rep.div(alloc, tensor);
                        stripGrad(inv_2sqrt, alloc);
                        const contribution = inv_2sqrt.mul(alloc, out_grad);
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

                // d/d(src0) [-src0] = -out_grad
                .neg => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const neg_grad = out_grad.neg(alloc);
                        stripGrad(neg_grad, alloc);
                        src0.grad = accumGrad(grad, alloc, neg_grad, inplace);
                    }
                },

                // d/d(src0) [relu(src0)] = step(src0) * out_grad
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

                // Gradient of sum/mean broadcasts back to source shape
                .sum, .mean => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        src0.grad = accumGrad(grad, alloc, out_grad.repeatLike(alloc, grad), inplace);
                    }
                },

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
                        try addToScratchUniq(alloc, scratch, out_grad.matMul(alloc, false, src1, true));
                        src0.grad = accumGrad(grad, alloc, scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(alloc, scratch, grad);
                    }
                    if (src1.grad) |grad| {
                        try addToScratchUniq(alloc, scratch, src0.matMul(alloc, false, out_grad, false));
                        src1.grad = accumGrad(grad, alloc, scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(alloc, scratch, grad);
                    }
                },
                .matmul_t1 => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        try addToScratchUniq(alloc, scratch, out_grad.matMul(alloc, false, src1, false));
                        src0.grad = accumGrad(grad, alloc, scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(alloc, scratch, grad);
                    }
                    if (src1.grad) |grad| {
                        try addToScratchUniq(alloc, scratch, src0.matMul(alloc, true, out_grad, false));
                        src1.grad = accumGrad(grad, alloc, scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(alloc, scratch, grad);
                    }
                },
                .matmul_t0t1 => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        try addToScratchUniq(alloc, scratch, out_grad.matMul(alloc, true, src1, false));
                        src0.grad = accumGrad(grad, alloc, scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(alloc, scratch, grad);
                    }
                    if (src1.grad) |grad| {
                        try addToScratchUniq(alloc, scratch, src0.matMul(alloc, false, out_grad, true));
                        src1.grad = accumGrad(grad, alloc, scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(alloc, scratch, grad);
                    }
                },
                else => @panic("Unimplemented backward OP"),
            }
        }
    };
}
