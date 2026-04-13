//! Enumerates all primitive tensor operations supported by the computation graph.
//!
//! Higher-level ops (sub, sqr, div, relu, scale, mean) decompose into
//! these primitives via the lazy API in `tensor.zig`.
//!
//! Each variant has a forward compute implementation in `tensor/forward.zig`
//! and (where implemented) a backward rule in `tensor/backward.zig`.

/// A primitive tensor operation. Stored on each tensor to record how it was produced.
pub const Op = enum {
    const Self = @This();

    // -- Structural --
    none,
    view,
    reshape,
    transpose,

    // -- Element-wise binary --
    add,
    mul,

    // -- Element-wise unary --
    neg,
    abs,
    sgn,
    step,
    sqrt,
    recip,
    gelu,

    // -- Reductions & broadcast --
    sum,
    repeat,

    // -- Matrix multiplication (4 transpose variants) --
    matmul,
    matmul_t0,
    matmul_t1,
    matmul_t0t1,

    /// Human-readable symbol for this operation, used in debug output and GraphViz export.
    pub fn symbol(self: *Self) []const u8 {
        return switch (self.*) {
            .none => "none",
            .view => "view(x)",
            .reshape => "reshape(x)",
            .transpose => "transpose(x)",
            .add => "x+y",
            .mul => "x*y",
            .neg => "-x",
            .abs => "abs(x)",
            .sgn => "sgn(x)",
            .step => "step(x)",
            .sqrt => "√x",
            .recip => "1/x",
            .gelu => "gelu(x)",
            .sum => "Σx",
            .repeat => "repeat(x)",
            .matmul => "X*Y",
            .matmul_t0 => "XT*Y",
            .matmul_t1 => "X*YT",
            .matmul_t0t1 => "XT*YT",
        };
    }
};
