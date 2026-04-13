//! Backward (reverse-mode autodiff) implementations for tensor operations.
//!
//! Given a tensor produced by an operation, `backward` propagates gradients
//! from the output back through to the source tensors. Each operation has a
//! corresponding gradient rule.

const std = @import("std");
const assert = std.debug.assert;
const Alloc = std.mem.Allocator;

/// Backward operations parameterized on the tensor type.
pub fn Ops(comptime Self: type, comptime T: type) type {
    _ = T;
    return struct {
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
            switch (tensor.op) {
                .none, .view => {},
                .dup => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const new_grad = grad.addImpl(alloc, tensor.grad.?, inplace);
                        assert(new_grad.isSameShape(grad));
                        src0.grad = new_grad;
                    }
                },
                .add => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        src0.grad = grad.addImpl(alloc, tensor.grad.?, inplace);
                    }
                    if (src1.grad) |grad| {
                        src1.grad = grad.addImpl(alloc, tensor.grad.?, inplace);
                    }
                },
                .sub => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        src0.grad = grad.addImpl(alloc, tensor.grad.?, inplace);
                    }
                    if (src1.grad) |grad| {
                        src1.grad = grad.subImpl(alloc, tensor.grad.?, inplace);
                    }
                },
                // d/d(src0) [src0 * src1] = src1 * out_grad
                // d/d(src1) [src0 * src1] = src0 * out_grad
                .mul => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        const src1_x = src1.mul(alloc, tensor.grad.?);
                        if (src1_x.grad) |gradp| {
                            gradp.deinit(alloc);
                            src1_x.grad = null;
                        }
                        src0.grad = grad.addImpl(alloc, src1_x, inplace);
                    }
                    if (src1.grad) |grad| {
                        const src0_x = src0.mul(alloc, tensor.grad.?);
                        if (src0_x.grad) |gradp| {
                            gradp.deinit(alloc);
                            src0_x.grad = null;
                        }
                        src1.grad = grad.addImpl(alloc, src0_x, inplace);
                    }
                },
                .div => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        const src1_x = src1.div(alloc, tensor.grad.?);
                        if (src1_x.grad) |gradp| {
                            gradp.deinit(alloc);
                            src1_x.grad = null;
                        }
                        src0.grad = grad.addImpl(alloc, src1_x, inplace);
                    }
                    if (src1.grad) |grad| {
                        const src0_x = src0.div(alloc, tensor.grad.?);
                        if (src0_x.grad) |gradp| {
                            gradp.deinit(alloc);
                            src0_x.grad = null;
                        }
                        src1.grad = grad.addImpl(alloc, src0_x, inplace);
                    }
                },
                .repeat => {
                    const src0 = src0_o.?;
                    const t_grad = tensor.grad.?;
                    if (src0.grad) |grad| {
                        src0.grad = t_grad.sumInto(alloc, grad);
                    }
                },
                // d/d(src0) [src0^2] = 2 * src0 * out_grad
                .sqr => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        const t2 = try Self.initScalar(alloc, 2);
                        const t2_rep = t2.repeatLike(alloc, src0);
                        const src0_2 = src0.mul(alloc, t2_rep);
                        if (src0_2.grad) |gradp| {
                            gradp.deinit(alloc);
                            src0_2.grad = null;
                        }
                        const src0_2_grad = src0_2.mul(alloc, tensor.grad.?);
                        src0.grad = grad.addImpl(alloc, src0_2_grad, inplace);
                    }
                },
                // Gradient of sum/mean broadcasts back to source shape
                .sum, .mean => {
                    const src0 = src0_o.?;
                    if (src0.grad) |grad| {
                        src0.grad = grad.addImpl(alloc, tensor.grad.?.repeatLike(alloc, grad), inplace);
                    }
                },
                // d/d(A) [A @ B] = out_grad @ B^T
                // d/d(B) [A @ B] = A^T @ out_grad
                .matmul => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        const z = tensor.grad.?.matMul(alloc, false, src1, true);
                        if (z.grad) |gradp| {
                            gradp.deinit(alloc);
                            z.grad = null;
                        }
                        src0.grad = grad.addImpl(alloc, z, inplace);
                    }
                    if (src1.grad) |grad| {
                        const z = src0.matMul(alloc, true, tensor.grad.?, false);
                        if (z.grad) |gradp| {
                            gradp.deinit(alloc);
                            z.grad = null;
                        }
                        src1.grad = grad.addImpl(alloc, z, inplace);
                    }
                },
                .matmul_t0 => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        try addToScratchUniq(alloc, scratch, tensor.grad.?.matMul(alloc, false, src1, true));
                        src0.grad = grad.addImpl(alloc, scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(alloc, scratch, grad);
                    }
                    if (src1.grad) |grad| {
                        try addToScratchUniq(alloc, scratch, src0.matMul(alloc, false, tensor.grad.?, false));
                        src1.grad = grad.addImpl(alloc, scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(alloc, scratch, grad);
                    }
                },
                .matmul_t1 => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        try addToScratchUniq(alloc, scratch, tensor.grad.?.matMul(alloc, false, src1, false));
                        src0.grad = grad.addImpl(alloc, scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(alloc, scratch, grad);
                    }
                    if (src1.grad) |grad| {
                        try addToScratchUniq(alloc, scratch, src0.matMul(alloc, true, tensor.grad.?, false));
                        src1.grad = grad.addImpl(alloc, scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(alloc, scratch, grad);
                    }
                },
                .matmul_t0t1 => {
                    const src0 = src0_o.?;
                    const src1 = src1_o.?;
                    if (src0.grad) |grad| {
                        try addToScratchUniq(alloc, scratch, tensor.grad.?.matMul(alloc, true, src1, false));
                        src0.grad = grad.addImpl(alloc, scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(alloc, scratch, grad);
                    }
                    if (src1.grad) |grad| {
                        try addToScratchUniq(alloc, scratch, src0.matMul(alloc, false, tensor.grad.?, true));
                        src1.grad = grad.addImpl(alloc, scratch.items[scratch.items.len - 1], inplace);
                        try addToScratchUniq(alloc, scratch, grad);
                    }
                },
                else => @panic("Unimplemented backward OP"),
            }
        }
    };
}
