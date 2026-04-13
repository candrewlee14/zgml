//! Backward (reverse-mode autodiff) implementations for primitive tensor operations.
//!
//! Only primitive ops need backward rules here. Decomposed ops (sub, sqr, div,
//! relu, scale, mean) get their gradients automatically through the chain rule
//! on the primitives they decompose into.

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
            const src0_o = tensor.src0;
            const src1_o = tensor.src1;
            const out_grad = tensor.grad.?;

            switch (tensor.op) {
                .none, .view, .reshape, .transpose => {},

                // d/d(src0) [src0 + src1] = out_grad
                // d/d(src1) [src0 + src1] = out_grad
                .add => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| src0.grad = accumGrad(grad, alloc, out_grad, inplace);
                    if (src1.grad) |grad| src1.grad = accumGrad(grad, alloc, out_grad, inplace);
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

                // d/d(src0) [-src0] = -out_grad
                .neg => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const neg_grad = out_grad.neg(alloc);
                        stripGrad(neg_grad, alloc);
                        src0.grad = accumGrad(grad, alloc, neg_grad, inplace);
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

                // sgn, step, gelu: not differentiable / not yet implemented
                .sgn, .step => {},
                .gelu => @panic("gelu backward not yet implemented"),
            }
        }
    };
}
