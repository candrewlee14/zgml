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
    permute,
    as_strided,
    broadcast_to,

    // -- Element-wise binary --
    add,
    mul,

    // -- Element-wise unary --
    neg,
    abs,
    sgn,
    step,
    relu,
    sqrt,
    recip,
    exp,
    log,
    gelu,

    // -- Reductions & broadcast --
    sum,
    max,
    repeat,
    gather_rows,
    scatter_add_rows,
    pick_rows,
    scatter_add_picks,
    scatter_add_view,

    // -- Slice --
    slice_assign, // write src0 into src1 at a position (mutates src1)

    // -- Matrix multiplication --
    matmul,

    /// True if this op is elementwise (shape-preserving) and can participate in fusion.
    pub fn isFusible(self: Self) bool {
        return switch (self) {
            .add, .mul, .neg, .abs, .sgn, .step, .relu, .sqrt, .recip, .exp, .log, .gelu => true,
            else => false,
        };
    }

    /// True if this is a binary op (takes two operands).
    pub fn isBinary(self: Self) bool {
        return switch (self) {
            .add, .mul => true,
            else => false,
        };
    }

    /// Human-readable symbol for this operation, used in debug output and GraphViz export.
    pub fn symbol(self: Self) []const u8 {
        return switch (self) {
            .none => "none",
            .view => "view(x)",
            .reshape => "reshape(x)",
            .transpose => "transpose(x)",
            .permute => "permute(x)",
            .as_strided => "as_strided(x)",
            .broadcast_to => "broadcast_to(x)",
            .add => "x+y",
            .mul => "x*y",
            .neg => "-x",
            .abs => "abs(x)",
            .sgn => "sgn(x)",
            .step => "step(x)",
            .relu => "relu(x)",
            .sqrt => "√x",
            .recip => "1/x",
            .exp => "exp(x)",
            .log => "log(x)",
            .gelu => "gelu(x)",
            .sum => "Σx",
            .max => "max(x)",
            .repeat => "repeat(x)",
            .gather_rows => "gather_rows(x)",
            .scatter_add_rows => "scatter_add_rows(x)",
            .pick_rows => "pick_rows(x)",
            .scatter_add_picks => "scatter_add_picks(x)",
            .scatter_add_view => "scatter_add_view(x)",
            .slice_assign => "x[pos]=y",
            .matmul => "X*Y",
        };
    }
};
