//! Enumerates all tensor operations supported by the computation graph.
//!
//! Each variant corresponds to a forward compute implementation in
//! `tensor/forward.zig` and (where implemented) a backward rule in
//! `tensor/backward.zig`.

/// A tensor operation. Stored on each tensor to record how it was produced.
pub const Op = enum {
    const Self = @This();

    none,
    dup,
    add,
    sub,
    mul,
    div,
    sqr,
    sqrt,
    sum,
    mean,
    repeat,
    abs,
    sgn,
    neg,
    step,
    relu,
    gelu,
    norm,
    //
    matmul,
    matmul_t0,
    matmul_t1,
    matmul_t0t1,
    //
    scale,
    cpy,
    reshape,
    view,
    permute,
    transpose,
    get_rows,
    // diag_max_inf,
    soft_max,
    rope,
    // conv_1d_1s
    // conv_1d_2s
    //
    // flash_attn,
    // flash_ff,
    //

    /// Human-readable symbol for this operation, used in debug output and GraphViz export.
    pub fn symbol(self: *Self) []const u8 {
        return switch (self.*) {
            .none => "none",
            .dup => "x",
            .add => "x+y",
            .sub => "x-y",
            .mul => "x*y",
            .div => "x/y",
            .sqr => "x^2",
            .sqrt => "√x",
            .sum => "Σx",
            .mean => "Σx/n",
            .repeat => "repeat(x)",
            .abs => "abs(x)",
            .sgn => "sgn(x)",
            .neg => "-x",
            .step => "step(x)",
            .relu => "relu(x)",
            .gelu => "gelu(x)",
            .norm => "norm(x)",
            //
            .matmul => "X*Y",
            .matmul_t0 => "XT*Y",
            .matmul_t1 => "X*YT",
            .matmul_t0t1 => "XT*YT",

            //
            .scale => "x*v",
            .cpy => "x->y",
            .reshape => "reshape(x)",
            .view => "view(x)",
            .permute => "permute(x)",
            .transpose => "transpose(x)",
            .get_rows => "get_rows(x)",
            // diag_max_inf,
            .soft_max => "soft_max(x)",
            .rope => "rope(x)",
            // conv_1d_1s
            // conv_1d_2s
            //
            // flash_attn,
            // flash_ff,
            //
        };
    }
};
