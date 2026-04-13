//! Enumerates all primitive tensor operations supported by the computation graph.
//!
//! This enum is the graph IR. Keep it small and low-level; prefer expressing
//! higher-level tensor APIs as compositions in `tensor.zig`, then fuse or
//! rewrite those compositions later as an optimization step.
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
    sub,
    mul,
    div,

    // -- Element-wise unary --
    neg,
    relu,
    abs,
    sgn,
    step,
    sqrt,
    recip,
    exp,
    log,
    gelu,

    // -- Reductions & broadcast --
    sum,
    mean,
    max,
    repeat,
    gather_rows,
    scatter_add_rows,
    pick_rows,
    scatter_add_picks,

    // -- Matrix multiplication (4 transpose variants) --
    matmul,
    matmul_t0,
    matmul_t1,
    matmul_t0t1,

    /// True if this op is elementwise (shape-preserving) and can participate in fusion.
    pub fn isFusible(self: Self) bool {
        return switch (self) {
            .add, .sub, .mul, .div, .neg, .relu, .abs, .sgn, .step, .sqrt, .recip, .exp, .log, .gelu => true,
            else => false,
        };
    }

    /// True if this is a binary op (takes two operands).
    pub fn isBinary(self: Self) bool {
        return switch (self) {
            .add, .sub, .mul, .div => true,
            else => false,
        };
    }

    /// Human-readable symbol for this operation, used in debug output and GraphViz export.
    pub fn symbol(self: *Self) []const u8 {
        return switch (self.*) {
            .none => "none",
            .view => "view(x)",
            .reshape => "reshape(x)",
            .transpose => "transpose(x)",
            .add => "x+y",
            .sub => "x-y",
            .mul => "x*y",
            .div => "x/y",
            .neg => "-x",
            .relu => "relu(x)",
            .abs => "abs(x)",
            .sgn => "sgn(x)",
            .step => "step(x)",
            .sqrt => "√x",
            .recip => "1/x",
            .exp => "exp(x)",
            .log => "log(x)",
            .gelu => "gelu(x)",
            .sum => "Σx",
            .mean => "mean(x)",
            .max => "max(x)",
            .repeat => "repeat(x)",
            .gather_rows => "gather_rows(x)",
            .scatter_add_rows => "scatter_add_rows(x)",
            .pick_rows => "pick_rows(x)",
            .scatter_add_picks => "scatter_add_picks(x)",
            .matmul => "X*Y",
            .matmul_t0 => "XT*Y",
            .matmul_t1 => "X*YT",
            .matmul_t0t1 => "XT*YT",
        };
    }
};
