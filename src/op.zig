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
    mul_mat,
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

    pub fn symbol(self: *Self) []const u8 {
        return switch (self) {
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
            .mul_mat => "X*Y",
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
